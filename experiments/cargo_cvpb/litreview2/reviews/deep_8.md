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
session id: 019ef835-812a-7053-be45-023fdfdbd22d
--------
user
你是 ReID 创新方法论拆解员 8。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(8)==
Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则尝试读取目标文件，再用 `pdftotext` 抽取每篇前 8 页的方法部分。随后只在判断先例和视角新意需要时联网核对，最后按你给的 5 个问题逐篇拆。
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
/bin/zsh -lc "printf '%s\\n' *.pdf" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc "find .. -name GOAL.md -o -name CLAUDE.md -o -name AGENTS.md -o -path '*/experiments/STATUS.md'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
codex
我已确认目录里能看到你列出的 6 个 PDF，其中 `Cross-Modal Full-Mode...` 有 ACM 和 arXiv 两份同题版本。项目上层的 `GOAL.md` 等文件在当前受限根目录里没有读到；这轮我只做论文库拆解，不改文件。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Cross-Modal Full-Mode Fine-Grained Alignment for
Text-to-Image Person Retrieval
HAO YIN and XIN MAN, Shenzhen Institute for Advanced Study, University of Electronic Science and
Technology of China, Shenzhen, China

FEIYU CHEN, JIE SHAO, and HENG TAO SHEN, University of Electronic Science and Technology
of China, Chengdu, China and Sichuan Artificial Intelligence Research Institute, Yibin, China
Text-to-Image Person Retrieval (TIPR) is a cross-modal matching task designed to identify the person images
that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
between the textual and visual modalities within a unified latent representation space. To address this
challenge, prior approaches incorporate attention mechanisms for implicit cross-modal local alignment.
However, they lack the ability to verify whether all local features are correctly aligned. Moreover, existing
methods tend to emphasize the utilization of hard negative samples during model optimization to strengthen
discrimination between positive and negative pairs, often neglecting incorrectly matched positive pairs. To
mitigate these problems, we propose FMFA, a cross-modal Full-Mode Fine-Grained Alignment framework,
which enhances global matching through Explicit Fine-Grained Alignment (EFA) and existing implicit relational
reasoning—hence the term “full-mode”—without introducing extra supervisory signals. In particular, we
propose an Adaptive Similarity Distribution Matching (A-SDM) module to rectify unmatched positive sample
pairs. A-SDM adaptively pulls the unmatched positive pairs closer in the joint embedding space, thereby
achieving more precise global alignment. Additionally, we introduce an EFA module, which makes up for the
lack of verification capability of implicit relational reasoning. EFA strengthens explicit cross-modal fine-grained
interactions by sparsifying the similarity matrix and employs a hard coding method for local alignment.
We evaluate our method on three public datasets, where it attains state-of-the-art results among all global
matching methods. The code for our method is publicly accessible at https://github.com/yinhao1102/FMFA.
CCS Concepts: • Information systems → Image search; • Computing methodologies → Object identification;
Additional Key Words and Phrases: Cross-modal retrieval, Person search, Fine-grained alignment

This work was supported by the National Natural Science Foundation of China (No. 62302080), Guangxi Key Research and
Development Program (No. Guike AB24010112), National Foreign Expert Project of China (No. S20240327), Sichuan Science
and Technology Program (No. 2025HJRC0021), and Sichuan Province Innovative Talent Funding Project for Postdoctoral
Fellows (No. BX202312).
Authors’ Contact Information: Hao Yin, Shenzhen Institute for Advanced Study, University of Electronic Science and
Technology of China, Shenzhen, China; e-mail: yinhao1102@std.uestc.edu.cn; Xin Man, Shenzhen Institute for Advanced
Study, University of Electronic Science and Technology of China, Shenzhen, China; e-mail: manxin@std.uestc.edu.cn;
Feiyu Chen, University of Electronic Science and Technology of China, Chengdu, China and Sichuan Artificial Intelligence
Research Institute, Yibin, China; e-mail: chenfeiyu@uestc.edu.cn; Jie Shao (corresponding author), University of Electronic
Science and Technology of China, Chengdu, China and Sichuan Artificial Intelligence Research Institute, Yibin, China;
e-mail: shaojie@uestc.edu.cn; Heng Tao Shen, University of Electronic Science and Technology of China, Chengdu, China
and Sichuan Artificial Intelligence Research Institute, Yibin, China; e-mail: shenhengtao@hotmail.com.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2026/4-ART135
https://doi.org/10.1145/3786798
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

135:2

H. Yin et al.

ACM Reference format:
Hao Yin, Xin Man, Feiyu Chen, Jie Shao, and Heng Tao Shen. 2026. Cross-Modal Full-Mode Fine-Grained
Alignment for Text-to-Image Person Retrieval. ACM Trans. Multimedia Comput. Commun. Appl. 22, 5, Article 135
(April 2026), 21 pages.
https://doi.org/10.1145/3786798

1

Introduction

Text-to-Image Person Retrieval (TIPR) seeks to understand natural language descriptions and
identify the most relevant person image within a large gallery [31]. Unlike general image-text
retrieval [5, 6, 44, 47, 50], which tends to achieve semantic-based matching between text and image,
TIPR is specifically designed for identifying individuals. TIPR requires the accurate modeling of
fine-grained correspondences between textual and visual modalities, owing to the large intra-class
variance and small inter-class difference. This substantial intra-class variation arises from two
aspects: (1) visual appearances of the same identity exhibit dramatic variations under different poses,
viewpoints, and illumination conditions, and (2) textual descriptions are influenced by differences
in phrasing, word order, and textual ambiguities. Therefore, the primary challenges in TIPR are
how to extract discriminative global representations from image-text pairs and how to achieve
precise cross-modal fine-grained alignment. Existing methods for tackling these challenges can be
roughly divided into two main categories: global matching methods and local matching methods.
Some global matching methods [58, 59] obtain discriminative global representations by aligning
images and texts, which are projected into a joint embedding space. Their widely adopted loss
functions include the Cross-Modal Projection Matching (CMPM) loss [58] and the Similarity Distribution Matching (SDM) loss [21]. The CMPM loss highlights the gap between the
scalar projections of image-text pairs and their matched label indicators. In comparison, the SDM
loss boosts global matching performance by minimizing the Kullback-Leibler (KL) divergence
between the normalized similarity profile of image-text pairs and the true label distribution. In
addition, the SDM loss incorporates a temperature hyperparameter to make model updates concentrate on hard negative samples, yet it leads to the neglect of unmatched positive pairs, as shown
in Figure 1(a). However, in TIPR, the accurate matching of positive pairs is prioritized over merely
distinguishing between positive and negative pairs. Meanwhile, some local matching methods
[2, 11, 35] incorporate attention mechanisms to achieve cross-modal fine-grained alignment. For
instance, RaSa [2] constructs a cross-modal encoder to generate multimodal representations for
subsequent fine-grained alignment. Building on RaSa, MARS [11] integrates a Masked AutoEncoder
decoder [14] to reconstruct masked image patch sequences into their original unmasked form,
thereby facilitating cross-modal fine-grained alignment. However, these methods rely on attention
mechanisms to implicitly aggregate local image-text representations. As a result, they yield only the
final multimodal representation, without revealing the details of the aggregation process. Consequently, these implicit aggregation methods make it difficult to determine whether the aggregated
multimodal representations correctly encode the corresponding visual and textual information.
To remedy these concerns, we propose a cross-modal Full-Mode Fine-Grained Alignment
(FMFA) framework that enhances global matching by including explicit fine-grained image-text
alignment and existing implicit relational reasoning. Specifically, we design an Adaptive Similarity
Distribution Matching (A-SDM) module to ensure the correct matching of positive image-text
pairs. Within the joint embedding space, the A-SDM module adaptively pulls positive pairs closer
together. In cases of mismatched positive pairs, the A-SDM module adaptively regulates the pulling
force based on their relative distance within the joint embedding space, as shown in Figure 1(b),
thus improving cross-modal global alignment. Based on the insight that each word in a caption can
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

Cross-Modal Full-Mode Fine-Grained Alignment for TIPR

135:3

Fig. 1. Evolution of global matching methods for TIPR. (a) Existing global matching methods focus on
hard negative samples to learn a discriminative boundary in a common latent space, thus enhancing the
distinction between positive and negative samples. (b) Our adaptive global matching method builds on a
discriminative boundary and concentrates on unmatched positive samples, adaptively pulling them closer to
the corresponding query text.

be associated with several image patches [3], we introduce an Explicit Fine-Grained Alignment
(EFA) module. The EFA module derives multimodal representations through explicit aggregation
with a sparse similarity matrix. During this process, the sparse similarity matrix between text
and image reflects the contribution of textual and visual representations to the final multimodal
representation. To minimize redundancy and reduce the computational cost during training, the
EFA module employs hard coding alignment between the aggregated multimodal representation
and its original visual and textual representations. These designs allow EFA to realize fine-grained
cross-modal interactions and assist the backbone network in learning more distinctive global
image-text representations without introducing additional supervision. FMFA is evaluated on three
public benchmarks [10, 27, 60], and it attains competitive top-level performance along with high
inference efficiency. We highlight our key contributions below:
— We introduce FMFA to explicitly leverage fine-grained interactions for improving cross-modal
alignment, without incurring extra supervision or inference overhead.
— We present an A-SDM module aimed at precisely aligning image-text pairs in a shared
embedding space. It adaptively adjusts to narrow the distance between mismatched positive
pairs, ensuring more precise matching.
— We develop an EFA module, which leverages the sparse similarity matrix for explicit aggregation and employs a hard coding method in cross-modal fine-grained alignment to minimize
redundant information.
2

Related Work

TIPR was initially proposed by Li et al. [27], who created the CUHK-PEDES dataset. Unlike visualbased person retrieval [7, 8, 16, 56], the core challenge of TIPR lies in constructing a shared latent
space that enables coherent alignment between visual and textual representations. Existing methods
can be typically classified into global and local matching approaches.
Early global methods [59, 60] directly aligned the global representations of images and text in
a joint embedding space. Schroff et al. [39] proposed a triplet ranking loss to enforce a margin
constraint between positive and negative pairs, and Zhang and Lu [58] introduced the CMPM/C
loss to minimize the discrepancy between the scalar projection of image-text pairs and their labels.
However, these global methods lack cross-modal fine-grained interactions, which restrict their
ability to capture detailed semantic correspondences. To address this limitation, early local matching
methods [12, 42, 49] explicitly aligned local visual and textual features to achieve fine-grained
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

135:4

H. Yin et al.

cross-modal interactions. Nevertheless, they rely on unimodal pre-trained models (e.g., BERT [9]
and ResNet [15]), failing to exploit the strong cross-modal alignment capability of recent pre-trained
Vision-Language Models (VLMs) [25, 26, 53].
Recent local matching methods [11, 19, 32, 36, 52] have benefited greatly from VLMs and
introduced VLMs to enhance cross-modal alignment. Park et al. [35] utilized a modified Contrastive
Language-Image Pre-training (CLIP) [37] model as the feature extractor and designed a slot
attention-based [30] part discovery module to identify discriminative human parts without extra
supervision, while Bai et al. [2] used the align-before-fuse model [26] as the backbone and introduced
a cross-modal encoder for fine-grained alignment. Although effective, these methods involve
complex computations during inference, leading to high time and memory costs, which limit their
applicability to real-time systems.
On another line of research, several studies [41, 45, 55] have explored leveraging large-scale
image-text pairs in the person Re-IDentification (ReID) domain to VLMs. Zuo et al. [61] utilized
CUHK-PEDES and ICFG-PEDES to train an image captioner, aiming to generate comprehensive
textual descriptions for pedestrian images. Yang et al. [55] employed BLIP-2 [24] to produce
attribute-aware captions for diffusion-generated pedestrian images [38], while Jiang et al. [22]
leveraged recent Multimodal Large Language Models, such as Qwen-VL [1] and LLaVA [28], to
automatically annotate large-scale ReID datasets in a human-like manner. The CLIP models pretrained on large-scale ReID datasets exhibit strong zero-shot performance. Their compatibility
with global matching methods—which relies solely on global features and has a simple inference
pipeline—makes them particularly suitable for direct fine-tuning in such settings.
Recent global matching methods [17, 21, 43] have integrated local fine-grained alignment modules
into global matching frameworks to obtain more discriminative global representations. Shu et al.
[43] introduced a bidirectional mask modeling mechanism that randomly masks image patches
and text words, encouraging the model to infer missing semantics and implicitly learn local visualtextual correspondences. He et al. [17] proposed the Vision-Guided Semantic-Group (VGSG)
network to cluster textual tokens into semantic groups and align them with corresponding visual
regions under the guidance of vision features, achieving group-level fine-grained alignment within
a global representation space. Similarly, Jiang and Ye [21] developed IRRA to employ an Implicit
Relation Reasoning (IRR) module based on attention mechanisms to capture latent cross-modal
relations, enhancing global alignment. Although these methods enhance fine-grained cross-modal
interactions within global matching frameworks, their implicit or group-level alignment strategies
may still fail to guarantee precise local correspondences. In light of these limitations, we propose
FMFA, which aims to enhance the global matching ability of the model by achieving cross-modal
FMFA, including EFA and IRR.
3

Method

This section introduces the proposed FMFA framework. Figure 2 presents an overview of FMFA,
and further details of the framework are elaborated in the subsequent subsections.
3.1

Feature Extraction

Motivated by the success of IRRA [21], we use the modified full CLIP [37] visual and textual
encoders to enhance cross-modal alignment capabilities while reducing inference costs.
Visual Modality. Given an input image 𝐼 ∈ R𝐻 ×𝑊 ×𝐶 , we employ a CLIP-pre-trained Vision
Transformer (ViT) to attain its image representation. An image is first divided into 𝑁 = 𝐻 ×𝑊 /𝑃 2
𝑁 via
distinct patches of size 𝑃 × 𝑃, which are then transformed into 1D token embeddings {𝑓𝑖𝑣 }𝑖=1
a learnable linear projection. After adding positional encodings and a [CLS] token, the sequence
𝑣 , 𝑓 𝑣 , . . . , 𝑓 𝑣 } is passed through 𝐿 transformer layers to capture dependencies among patches.
{𝑓𝑐𝑙𝑠
1
𝑁
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

Cross-Modal Full-Mode Fine-Grained Alignment for TIPR

135:5

Fig. 2. The illustration of our FMFA framework. (a) Our FMFA contains a two-stream feature extraction
network and four distinct modules for structured representation learning, namely Identity Identification
(ID loss), A-SDM, EFA, and IRR. The former two are coarse-grained alignment modules, and the latter two
are fine-grained alignment modules. Modules linked via dashed connections are omitted during inference.
(b) A-SDM obtains the adaptive weight through the similarity score of unmatched positive pairs, adaptively
pulling positive pairs closer and dynamically adjusting the pulling force. (c) EFA achieves fine-grained
interaction by hard coding alignment of token or patch embeddings and their corresponding languagegrouped vision embedding, which is derived from an aggregated sparse similarity matrix.

𝑣 is linearly mapped into the joint image-text embedding
Finally, the [CLS] token embedding 𝑓𝑐𝑙𝑠
space, producing the compact global feature of the image.
Textual Modality. Given an input text 𝑇 , we utilize the CLIP-Xformer textual extractor [37] to
obtain its embedding. The text is first tokenized through lower-cased Byte Pair Encoding [40] and
framed with [SOS] and [EOS] tokens to indicate sequence boundaries. The resulting token sequence
𝑡 , 𝑓 𝑡 , . . . , 𝑓 𝑡 } is processed by the transformer encoder, which models dependencies among
{𝑓𝑠𝑜𝑠
𝑒𝑜𝑠
1
𝑡 , is
tokens via masked self-attention. Finally, the [EOS] token embedding from the top layer, 𝑓𝑒𝑜𝑠
linearly mapped into the joint image-text representation space, generating a compressed global
textual representation.

3.2

A-SDM

Adopted from IRRA [21], we introduce a novel A-SDM module, which aims to adaptively pull the
unmatched positive image-text pairs into a shared representation space, further enhancing the
cross-modal global matching capability of the model.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

135:6

H. Yin et al.

Let the mini-batch contain 𝐵 image-text pairs, we pair each text embedding 𝑔𝑖𝑡 with its global
image embedding 𝑔𝑣𝑗 to form the set {(𝑔𝑖𝑡 , 𝑔𝑣𝑗 ), 𝑦𝑖,𝑗 }𝐵𝑗=1 , where 𝑦𝑖,𝑗 serves as the matching indicator.
Specifically, 𝑦𝑖,𝑗 = 1 denotes a matched pair, while 𝑦𝑖,𝑗 = 0 denotes a mismatched pair. Let 𝑐𝑜𝑠 (a, c) =
a> c/kakkck denotes the similarity of a and c. Subsequently, like SDM [21], the similarity matrix of
image-text pairs is obtained through the following softmax function:
𝑒𝑥𝑝 (𝑐𝑜𝑠 (𝑔𝑖𝑡 , 𝑔𝑣𝑗 )/𝜏1 )

𝑝𝑖,𝑗 = Í𝐵

,

𝑡 𝑣
𝑘=1 𝑒𝑥𝑝 (𝑐𝑜𝑠 (𝑔𝑖 , 𝑔𝑘 )/𝜏1 )

(1)

where 𝜏1 acts as a temperature term that modulates the spread of the resulting distribution.
The probability 𝑝𝑖,𝑗 quantifies how much the similarity between the text embedding 𝑔𝑖𝑡 and the
image embedding 𝑔𝑣𝑗 contributes relative to the sum of all similarities between 𝑔𝑖𝑡 and every image
embedding in the mini-batch.
Let the 𝑖th text 𝑇𝑖 from the batch be designated as the query text and 𝐼𝑖 be the corresponding
image for 𝑇𝑖 at rank-𝑘, where 𝑘 > 1. Different from IRRA [21], we propose to derive an adaptive
weighting factor by assessing the similarity between the query text 𝑇𝑖 and all image representations:


𝑡 2𝑖
𝑤𝑖 = 𝛼 · max 𝑝𝑖,𝑘 − 𝑝𝑖,𝑖 + 1,
(2)
𝑘

where 𝛼 is a weight factor reflecting the contribution of unmatched image-text pairs to the crossmodal global matching ability of the model. Here, max𝑘 𝑝𝑖,𝑘 indicates the top similarity value
between the text 𝑇𝑖 and every image within the mini-batch, while 𝑝𝑖,𝑖 refers to the similarity
associated with its corresponding positive image. The constant term “+1” ensures that when 𝑇𝑖
and its corresponding image 𝐼𝑖 are correctly matched, the weight 𝑤𝑖𝑡 2𝑖 defaults to 1. In this case,
the A-SDM loss reduces to the SDM loss [21], preventing overemphasis on correctly matched
pairs while allowing the model to focus adaptively on harder and misaligned pairs. Conversely,
𝑤𝑖𝑡 2𝑖 > 1 indicates that 𝑇𝑖 and 𝐼𝑖 are unmatched, increasing their contribution to the loss to enhance
global cross-modal alignment. The A-SDM loss for mapping text to image within a mini-batch is
subsequently formulated as:


𝐵
𝐵
𝑝𝑖,𝑗
1 Õ 𝑡 2𝑖 Õ
L𝑡 2𝑖 = 𝑊 𝑡 2𝑖 ∗ 𝐾𝐿(pi kqi ) =
𝑤𝑖
𝑝𝑖,𝑗 log
,
(3)
𝐵 𝑖=1
𝑞𝑖,𝑗 + 𝜖
𝑗=1
Í𝐵
where 𝜖 is a tiny offset added to safeguard the computation from unstable values, and 𝑞𝑖,𝑗 = 𝑦𝑖,𝑗 / 𝑘=1
denotes the ground-truth matching probability.
In a complementary manner, the A-SDM loss for the image-to-text branch L𝑖2𝑡 is derived by
swapping the roles of the text and image features. The bidirectional A-SDM loss is formulated as:
L𝐴-𝑆𝐷𝑀 = L𝑖2𝑡 + L𝑡 2𝑖 .
3.3

(4)

EFA

To effectively leverage fine-grained information, it is necessary to narrow the underlying disparity
between visual and textual modalities. Although many attention-based fine-grained alignment
approaches have shown effectiveness by implicitly associating local regions in images with textual
fragments, they provide no direct means to verify whether these localized correspondences are
accurately aligned. We propose an explicit cross-modal aggregation approach that leverages the
sparse similarity matrix between the local image and text features. To further reduce redundant
information and minimize memory and time costs during fine-grained alignment, we use hard
coding to align the aggregated language-grouped vision embeddings with both image and text
embeddings, as shown in Figure 2(c).
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

Cross-Modal Full-Mode Fine-Grained Alignment for TIPR

135:7

Fig. 3. Illustration of the EFA module. (a) EFA imposes a sparse structure on the similarity matrix relating
token embeddings to patch embeddings and obtains the language-grouped vision embeddings by aggregating
the sparse similarity with its corresponding patch. (b) EFA obtains hard similarity through hard coding and
LSE pooling, where the calculated hard similarity reflects the relationship between the language-grouped
vision embeddings and their original token or patch embeddings.

Sparse Similarity Matrix Aggregation. Some methods [33, 57] incur substantial computational and
memory overhead, as they evaluate pairwise relationships between every image patch and every
text token, which limits scalability to large batch sizes. Therefore, we apply a sparsification strategy
to reduce the full pairwise similarity computation. While softmax is commonly used for such sparse
processing, it tends to produce low-entropy similarity distributions that impede effective gradient
flow [18]. Thus, we further adopt a max–min normalization scheme to achieve a more stable and
expressive sparse similarity aggregation.
An image 𝐼 and its corresponding text 𝑇 are encoded through the visual and textual encoders,
respectively. As presented in Figure 3(a), the similarity between image patches and text tokens is
𝐿 of the text transformer and {𝑓 𝑣 } 𝑁
computed via the inner product of the last hidden states {𝑓𝑖𝑡 }𝑖=1
𝑖 𝑖=1
𝑡
𝑣
of the ViT. 𝑠𝑖,𝑗 = 𝑓𝑖 · 𝑓 𝑗 measures the similarity between the text token 𝑓𝑖𝑡 and the image patch 𝑓 𝑗𝑣 ,
where · denotes the inner product. To obtain the aggregation weight, each token 𝑖 is first scaled to
the range [0, 1] through the following min-max normalization:
𝑠ˆ𝑖,𝑗 =

𝑠𝑖,𝑗 − min𝑘 𝑠𝑖,𝑘
.
max𝑘 𝑠𝑖,𝑘 − min𝑘 𝑠𝑖,𝑘

(5)

We sparsify the normalized similarity matrix to encourage cross-modal interactions between
each token and its patches with higher similarity:
(
𝑠ˆ𝑖,𝑗 if 𝑠ˆ𝑖,𝑗 ≥ 𝜎
𝑠˜𝑖,𝑗 =
(6)
0
otherwise,
where 𝜎 is the sparsity threshold. 𝜎 is assigned the value 1/𝑁 , where 𝑁 corresponds to the total
count of patches in the image. This ensures that each token has a minimum of one corresponding
image patch for alignment. We compute the aggregation weights by:
𝑠˜𝑖,𝑗
𝑎𝑔𝑔𝑖,𝑗 = Í𝑀

𝑚=1 𝑠˜𝑖,𝑗

,

(7)

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

135:8

H. Yin et al.

where 𝑀 is the number of image patches retained with high similarity to the token 𝑖, and 𝑎𝑔𝑔𝑖,𝑗
quantifies the influence of patch 𝑗 in forming the language-grouped vision embedding (referred to as
joint embedding) associated with token 𝑖. This explicit aggregation strategy ensures a comprehensive
interaction between token 𝑖 and its corresponding patch 𝑗 during local alignment. In particular, the
aggregation weight 𝑎𝑔𝑔𝑖,𝑗 effectively captures the semantic relevance between token 𝑖 and patch 𝑗,
thereby facilitating precise alignment.
Next, we derive the corresponding joint embedding 𝑒𝑖 as:
𝑒𝑖 =

𝑁
Õ

(8)

𝑎𝑔𝑔𝑖,𝑗 · 𝑓 𝑗𝑣 ,

𝑗=1

where 𝑁 is the count of image patches. The resulting set of joint embedding 𝑒𝑖 has the same length
𝐿 as the text token 𝑓𝑖𝑡 .
𝐿 and
Hard Coding Alignment. We calculate the similarity between the joint embeddings {𝑒𝑖 }𝑖=1
𝐿 as well as image embeddings {𝑓 𝑣 } 𝑁 , retheir corresponding original text embeddings {𝑓𝑖𝑡 }𝑖=1
𝑖 𝑖=1
spectively. To reduce both computational and memory costs, we adopt a hard coding similarity
computation between the joint embeddings and their corresponding text and image embeddings,
and the theoretical analysis of the hard coding is provided in Appendix A. For simplicity, we only
present the calculation between the joint embeddings and the text embeddings, while the remaining
computations follow a similar and symmetric approach.
For the text 𝑇 and its corresponding joint embedding 𝐸, we calculate the original similarity matrix
𝐿 and their joint embeddings {𝑒 }𝐿 , where 𝑜
𝑡 >
𝑡
𝑂 between all text tokens {𝑓𝑖𝑡 }𝑖=1
𝑖 𝑖=1
𝑖,𝑗 = 𝑓𝑖 𝑒 𝑗 /k𝑓𝑖 kk𝑒 𝑗 k
𝑡
𝑡
means the cosine similarity of 𝑓𝑖 and 𝑒 𝑗 . For the token 𝑓𝑖 , we compute the weight factor between
it and all joint embeddings using the following hard coding way:


if 𝑗 = argmax (𝑜𝑖,𝑗 0 )
1

𝑗 0 =1···𝐿
𝜔𝑖,𝑗 =
(9)

 0 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒.

Then, we utilize the LSE pooling [23] to compute the hard similarity between text 𝑇 and its
corresponding joint embedding 𝐸 by:
!
𝐿
Õ
ℎ𝑎𝑟𝑑_𝑠 (𝑇 , 𝐸) = 𝐿𝑆𝐸 − 𝑃𝑜𝑜𝑙𝑖𝑛𝑔
𝜔𝑖,𝑗 𝑜𝑖,𝑗 ,
𝑗=1
𝐿
Õ

(10)




1
= log
exp 𝜆 max 𝑜𝑖,𝑗 ,
𝑗=1···𝐿
𝜆
𝑖=1
where 𝜆 controls the degree to which the most relevant text embeddings and their corresponding
joint embeddings are emphasized.
Given a batch containing 𝐵 text embeddings along with their associated joint embeddings, we
compute the hard coding similarity matrix 𝐻𝑎𝑟𝑑_𝑆 following Equations (9) and (10), as illustrated
in Figure 3(b). We calculate the EFA loss from the text to its joint embedding, adapted from the
triplet ranking loss [39]:


Õ
𝐻𝑎𝑟𝑑_𝑆 neg − 𝐻𝑎𝑟𝑑_𝑆 pos + 𝑚𝑎𝑟𝑔𝑖𝑛
1
L𝑡 2𝑒 = log
exp
,
(11)
𝐵
𝜏2
neg
where 𝜏2 is a scaling factor adjusting the spread of the loss, and 𝑚𝑎𝑟𝑔𝑖𝑛 is a distance hyperparameter
defining the minimal gap separating positive and negative pairs.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.


 succeeded in 0ms:
Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Cross-modal Collaborative Representation Learning for Text-to-Image Person
Retrieval
Shuanglin Yan1 , Jun Liu2 , Neng Dong1 , Liyan Zhang3∗ and Jinhui Tang1
1
Nanjing University of Science and Technology
2
Lancaster University
3
Nanjing University of Aeronautics and Astronautics
{shuanglinyan, neng.dong, jinhuitang}@njust.edu.cn, j.liu81@lancaster.ac.uk, zhangliyan@nuaa.edu.cn
Abstract

1

Identity i

i

The young man
is wearing sky
blue jacket....

j

A man in his
late twenties
with short....

j

The young man
is wearing sky
blue jacket....

Labled
Identity i

i

Identity j

j

i
j
i

i

i

j

i

j

i

j

Image Embedding

Text Embedding

Visual
Backbone
Image feature
Virtualtext
Branch
Virtual-text
feature

i

j
j

j j
j j

i

j

j

i

i

j

i
j

i
j

j

i

i

Labled
Identity j

0
β
α

i

Labled

i

j

Similarity Metric Space

/ j j Strong Positive Pairs
/ j j Weak Positive Pairs
i j / j i Negative Pairs

i

i

i

i

LIBM

Enhanced
Text
image feature feature

Textual
Backbone

A man in his
late twenties
with short....

Figure 1: (1) Examples of different types of image-text pairs.
Identity-bounded Matching (IBM) loss is designed to handle different types of image-text pairs distinctly by establishing different
similarity boundaries, leading to more accurate cross-modal correspondences. (2) We propose a cross-modal collaborative representation learning framework that leverages the inter-modality complementarity to enhance image features without relying on actual texts.

Introduction

Person re-identification (ReID) aims to retrieve a personof-interest across different camera networks. ReID models [Gong et al., 2022; Li et al., 2023b; Li et al., 2019a]
trained on extensive labeled cross-camera image pairs have
shown impressive retrieval capabilities. However, the closeset assumption of paired cross-camera images severely limits
∗

i

i

Cross-modal
Collaboration

Text-to-image person retrieval (TIPR) aims to find
images of the same identity that match a given
text description. Current TIPR methods mainly
focus on mining the association between images
and texts, ignoring their potential complementarity. Besides, existing matching losses treat all positive pairs from the same identity equally, leading to noisy correspondences. In this paper, we
propose CoRL: a cross-modal Collaborative Representation Learning framework designed to improve
TIPR by effectively leveraging the complementarity between modalities. The text typically contains
identity details with less noise, which helps distinguish visually similar pedestrians. This inspires us
to integrate it into the corresponding image to emphasize identity-related and modality-shared visual
information. However, corresponding text for each
image is not always available, especially during inference. Accordingly, we introduce a Virtual-text
Embedding Synthesizer that generates high-quality
virtual-text features for cross-modal collaboration,
eliminating the need for actual texts. We then design a Cross-Modal Collaboration learning process,
incorporating a Cross-modal Relation Consistency
loss to promote interaction and fusion between image and virtual-text features for mutual enhancement. Additionally, an Identity-bounded Matching
loss is proposed to handle different types of imagetext pairs distinctly, leading to more accurate crossmodal correspondences. Extensive experiments on
multiple benchmarks demonstrate the superiority
of CoRL over existing TIPR methods.

Identity-bounded Matching

A man in his
late twenties
with short....

Labled

Corresponding author

2152

the practical deployment of these models. Recently, Text-toimage person retrieval (TIPR) [Li et al., 2017] has been proposed to address situations where images are missing under
certain cameras. As a realistic extension to ReID, TIPR assumes that text descriptions are available for the missing images, allowing the retrieval of target person images via these
descriptions. The model must accurately discover identity information and establish correspondences between images and
texts, making TIPR a challenging and meaningful task.
Previous TIPR methods [Chen et al., 2022; Yan et
al., 2023d] have relied on single-modal pre-trained models [Dosovitskiy et al., 2021; Devlin et al., 2019] as backbone networks, leveraging their robust initial representations
to facilitate fine-tuning on training data. However, these

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

single-modal pre-trained models tend to overemphasize information from one modality, leading to significant modality gaps and alignment challenges. With the rise of visionlanguage pre-training (VLP), some works [Yan et al., 2023c;
Jiang and Ye, 2023] have adopted VLP models as backbone
networks, allowing TIPR to benefit from rich multi-modal information. Although promising results have been achieved,
existing methods mainly focus on modeling the association
between images and texts, often overlooking their potential
complementarity. In reality, the text primarily contains pedestrian identity details with less noise. Even when pedestrians
are visually similar, their text descriptions are often distinct,
making text information more discriminative and robust. This
leads us to explore whether inter-modality complementary information can benefit the TIPR task. Inspired by this, we propose integrating text information into images to emphasize
discriminative and modality-shared identity details, aiding in
distinguishing fine-grained differences and enhancing robustness to background noise. However, this approach requires
access to the corresponding text for each image, which is not
always available, especially during inference.
Accordingly, we propose a cross-modal Collaborative
Representation Learning (CoRL) framework that leverages
the inter-modality complementary information to enhance
image features without relying on actual texts. Specifically,
we introduce a Virtual-text Embedding Synthesizer (VES),
which utilizes CLIP’s vision-language alignment capabilities
to produce high-quality virtual-text embeddings directly from
images, eliminating the need for actual texts. These generated
virtual-text embeddings can replace actual texts in subsequent
cross-modal collaboration. To ensure consistency, we align
the generated virtual-text embeddings with actual text embeddings at both the feature and semantic levels. To fully exploit
the complementarity between images and texts, we design a
dual-branch cross-modal collaboration learning process, incorporating a cross-modal relation consistency loss (CRCL).
The visual backbone branch focuses on generating image features. The virtual-text branch uses VES to create virtual-text
embeddings from images, which are then fed into an Adapter
to produce virtual-text features adapted to the target domain.
The CRCL loss enforces that image and virtual-text features
maintain the same relationship with modality-specific prototypes. This encourages information exchange and collaboration between the two branches, allowing virtual textual information to emphasize the discriminative and modality-shared
identity details in the images, thereby enhancing feature discriminability and reducing modality gaps.
Cross-modal matching loss is essential for learning accurate correspondences between modalities. However, existing
losses [Zhang and Lu, 2018; Ding et al., 2021; Jiang and Ye,
2023] treat all positive pairs from the same identity equally,
resulting in noisy correspondences. Typically, each batch
contains three types of image-text pairs: single-view strong
positive pairs, cross-view weak positive pairs, and negative
pairs, as illustrated in Figure 1. Due to significant appearance differences under the same identity caused by view variations, cross-view weak positive pairs may suffer from noisy
correspondences. Consequently, the similarity between these
three types of image-text pairs should decrease progressively.

2153

To address this, we propose an Identity-bounded Matching
(IBM) loss, which defines distinct similarity boundaries for
each category of image-text pairs, thereby leading to more
precise cross-modal correspondences.
Here are the main contributions of our paper: (1) We
propose a cross-modal collaborative representation learning
framework, which is the first to leverage the inter-modality
complementary information to improve the TIPR task without relying on actual texts. (2) Identity-bounded matching loss is proposed to learn precise cross-modal correspondences. (3) Extensive experiments verify the effectiveness
of our method and achieve superior performance on multiple
benchmarks.

2

Related Work

2.1

Text-to-Image Person Retrieval

TIPR extends ReID [Gong et al., 2024; Dong et al., 2024a;
Dong et al., 2024b] to a more realistic scenario. The TIPR
model mainly contains two parts: the backbone network and
the feature alignment network [Tang et al., 2025]. A common practice is to use pre-trained backbones to leverage their
strong initial representation capabilities, facilitating effective
fine-tuning on TIPR data. Earlier methods [Shen et al., 2023;
Yan et al., 2023b] employed single-modal pre-trained models
such as ViT (pre-trained on ImageNet) and BERT. Recently,
the success of vision-language pre-trained models (VLPs) has
led to their widespread adoption in TIPR [Yan et al., 2023c;
Jiang and Ye, 2023], achieving promising results by exploiting their rich multi-modal knowledge. Notably, some recent
works [Yang et al., 2023; Tan et al., 2024] have advanced
further by retraining VLPs specifically for TIPR using largescale datasets, yielding additional performance gains.
For feature alignment network, various strategies have
been proposed to align images and texts. Early methods [Li et al., 2017; Zhang and Lu, 2018] directly aligned
the global features of images and texts. To achieve finegrained correspondences, later methods [Chen et al., 2022;
Yan et al., 2023d] introduced feature aggregation schemes to
generate multiple local features, modeling fine-grained alignment through interaction or guidance between these local features. However, this approach increases storage costs and inference time, reducing practicality. To avoid explicitly generating local features, recent methods [Jiang and Ye, 2023;
Li et al., 2023a] propose to inject fine-grained information [Tang et al., 2023; Yan et al., 2023a] into global features
by designing auxiliary tasks (e.g., masked language/region
modeling) to model fine-grained matching.
Despite these advancements, existing methods mainly emphasize image-text associations but overlook their complementarity. In this study, we explore inter-modality complementarity to enhance image features with text information.
Besides, existing matching losses treat image-text pairs of the
same identity equally. Although RaSa [Bai et al., 2023] incorporates a discriminator to differentiate between positive
pairs, it still relies on conventional matching losses, with the
discriminator serving only as a regularizer. In contrast, our
IBM loss directly addresses this issue by establishing distinct
boundaries for different types of image-text pairs.

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

The young man
is wearing sky
blue jacket....

Stage 1: Virtual-text Embedding
Synthesizer (VES)

Et

Lscor

LIBM

Visual Backbone
Branch

CRCL

s
Lalign
2

1

...

z v,i

Freezed Network
Learnable Network
CLIP Visual/Textual FC Single Fully
Ev Et Encoders
Connected Layer

Batch

g()
Adapter

Ev

s1

Lalign

...
Virtual-text features

Virtual-text Branch

Text features

...

...

...

VES

...

z^ t,i

Visual
Backbone

Lid

Textual
Backbone

Visual/Tex
tual Spaces
Visual/Textu
al Prototypes
Visual/Virtualtext Features
Relations
Information
Transfer

A man in his
lateThe
twenties
young man
with
short.... sky
is wearing
blue jacket....

...

FC

z t,i

Image features

...

Ti
Ii

...

Stage 2: Cross-modal Collaboration

A man in his
lateThe
twenties
young man
with
short.... sky
is wearing
blue jacket....

Batch

Cross-modal Relation Consistency Loss
(CRCL)

Figure 2: Overview of our CoRL. In the first stage, we introduce a Virtual-text Embedding Synthesizer to generate a virtual-text embedding
from each image. The generated virtual-text embedding is then used to replace actual text in modeling multi-modal complementarity with
the image. In the second stage, we propose a dual-branch Cross-Modal Collaboration learning process equipped with a Cross-modal Relation
Consistency Loss, which encourages the exchange of information and mutual enhancement between image and virtual-text features.

2.2

Vision-Language Pre-Training

Vision-language pre-training involves training large-scale
models on vast collections of image-text pairs, demonstrating exceptional capabilities in semantic understanding [Shen
and Tang, 2024; Shen et al., 2025], multi-modal alignment,
and generalization. It has been widely used in diverse downstream tasks. The primary objective of pre-training is to
achieve a deep understanding of both image and text semantics and their correspondences. Some studies [Li et al.,
2019b; Li et al., 2021] focus on generating multi-modal representations by encoding interactions between images and
texts with tasks such as mask language modeling and image captioning. However, these methods typically necessitate pairwise interactions for all image-text pairs, leading
to inefficiency during training and inference, thus limiting
their practicality in large-scale applications. Consequently,
some studies [Radford et al., 2021; Yao et al., 2022] integrate
contrastive representation learning into Vision-language pretraining. These methods encode images and texts separately
into a joint space and learn modality-aligned representations
by contrasting positive and negative pairs, such as the wellknown CLIP [Radford et al., 2021], trained on 400 million
image-text pairs. The strong multi-modal alignment capability of CLIP prompts us to introduce it into the TIPR task.

3

Methods

This section presents our proposed CoRL framework, with an
overview in Figure 2 and details in the following subsections.

3.1

Problem Formulation

The TIPR dataset contains image-text pairs of persons with
multiple identities, where each identity has multiple images
collected from different cameras with their annotated text descriptions. The goal is to accurately identify images that belong to the same identity as a given text description. This requires the TIPR model to extract identity-discriminative representations and accurately establish cross-modal correspondences. To achieve this, we first use a pre-trained CLIP as

2154

the backbone to provide high-quality initialization and facilitate fine-tuning on the TIPR dataset. Next, we propose a
cross-modal collaborative representation learning framework
that leverages inter-modality complementary information to
enhance image features. Additionally, we design an identitybounded matching loss to fully leverage identity information
and precisely establish cross-modal correspondences.

3.2

Cross-modal Collaborative Representation
Learning

Current TIPR methods mainly emphasize image-text associations but overlook their complementarity. Texts offer
the following advantages over images: texts primarily contain pedestrian identity details with less noise. Even when
pedestrians are visually similar, their text descriptions are
often distinct, making text information more discriminative
and robust. Complementing images with textual information
enhances fine-grained discrimination and reduces modality
gaps. However, this requires access to corresponding text for
each image, which is not always available, especially during inference. In this paper, we tackle two key challenges:
eliminating the dependency on actual text and effectively utilizing text information to enhance image representations. To
achieve this, we propose a two-stage cross-modal collaborative representation learning strategy. In the first stage, we
develop a Virtual-text Embedding Synthesizer (VES) to generate virtual-text embeddings directly from each image, eliminating the need for actual paired text. In the second stage, we
implement a dual-branch cross-modal collaboration learning
process to integrate the virtual-text information into images,
thereby enhancing image features.
Virtual-text Embedding Synthesizer. To eliminate reliance on actual paired text, we propose a Virtual-text Embedding Synthesizer (VES) that generates a virtual-text embedding directly from each image, as illustrated in Figure 2.
CLIP, trained on vast amounts of image-text data, can generate modality-aligned image-text features. We utilize its
alignment capabilities to convert images into text embeddings. VES functions as a CLIP-based encoder-decoder
model. Specifically, an image Ii is first processed by CLIP’s

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

visual encoder Ev to produce an image embedding zv,i .
This embedding is then transformed into virtual-text tokens
through a fully connected layer. Finally, CLIP’s textual encoder Et decodes these virtual-text tokens to generate the
virtual-text embedding ẑt,i .
This generated virtual-text embedding ẑt,i can serve as a
substitute for the actual text embedding to enhance its image
feature, given that the virtual-text and actual-text embeddings
are consistent. To ensure this consistency, we introduce an
alignment loss Ls1
align that aligns the generated virtual-text
embedding with the actual text embedding zt,i at both feature
and semantic levels, inheriting the advantages of the actual
text Ti .
s1
s1
Ls1
align = Lnce + Lmse

Ls1
nce = −

B
T
exp(zt,i
ẑt,i )
1 X
log PB
T
B i=1
j=1 exp(zt,i ẑt,j )

B
T
exp(ẑt,i
zt,i )
1 X
−
log PB
T
B i=1
j=1 exp(ẑt,i zt,j )

(1)

(2)

B

Ls1
mse =

1 X
∥ẑt,i − zt,i ∥2
B i=1

(3)

where zt,i denotes the actual text embedding of text Ti , generated by CLIP’s visual encoder. ∥ · ∥2 denotes the L2 distance,
and τa denotes the temperature factor. To facilitate effective
conversion between images and virtual-text embeddings, we
impose a correlation loss Ls1
cor to ensure that valuable image
information is retained as much as possible throughout the
conversion process.
Ls1
cor = −

B
T
exp(zv,i
ẑt,i /τa )
1 X
log PB
T
B i=1
j=1,lj ̸=li exp(zv,i ẑt,j /τa )

B
T
exp(ẑt,i
zv,i /τa )
1 X
−
log PB
T
B i=1
j=1,lj ̸=li exp(ẑt,i zv,j /τa )

(4)

s2
s2
Ls2
align = λ1 Lnce + Lmse .
B

Ls2
nce = −

B

2155

(5)

1 X
exp(t̂Ti sg(ti ))
−
log PB
T
B i=1
j=1 exp(t̂i sg(tj ))
B

Ls2
mse =

1 X
∥t̂i − sg(ti )∥2
B i=1

(6)

Where sg(·) means stop-gradient operator, which ensures that
only the Adapter is updated. With the image feature vi and
the virtual-text feature t̂i , we introduce a cross-modal relation consistency loss (CRCL). This loss enforces that the image and virtual-text features share the same relationship with
modality-specific prototypes, promoting the two branches to
exchange information and cooperate with each other. For
each batch B, we cluster the image and text features based
on identity labels to obtain visual and textual prototypes, pv
and pt . We then compute the cosine similarity between the
image feature vi and the visual prototypes pv to determine
their relationship riv . Similarly, we calculate the relationship
rit between the virtual-text feature t̂i and these textual prototypes pt . Thus, the CRCL is formulated as:
B

Ls2
CRCL =


1 X
Dkl (rit ||riv ) + Dkl (riv ||rit )
2|B| i=1

(7)

where Dkl is the Kullback-Leibler divergence. Mutual
knowledge distillation on instance-to-prototype relations between the visual embedding vi and virtual-text embedding t̂i
allows them to exchange information and benefit from one another. Incorporating virtual-text information emphasizes the
discriminative and modality-shared identity details in the image Ii , boosting the discriminability and robustness of the image feature vi , making it easier to differentiate visually similar pedestrians while effectively reducing the modality gap
between the image Ii and the text Ti .

3.3

where li represents the identity label of Ii /Ti . The overall
s1
s1
objective of our VES is calculated as: Ls1
V ES = Lalign +Lcor .
VES enables the generation of virtual-text embedding that is
aligned with the actual text for each image, eliminating the
need for actual text.
Cross-modal Collaboration. With the virtual-text embedding, we design a dual-branch cross-modal collaboration
learning process to integrate it with the corresponding image, thereby enhancing the image feature. Specifically, for an
image-text pair (Ii , Ti ), the visual backbone branch generates
the image feature vi of image Ii . Simultaneously, the virtualtext branch first produces a virtual-text embedding ẑt,i from
image Ii using VES, which is then processed by an Adapter
g(·) to yield a virtual-text feature t̂i adapted to the target domain. Additionally, the text feature ti for text Ti is generated
via the textual backbone. We optimize the Adapter by aligning the virtual-text feature t̂i with the text feature ti through

1 X
exp(sg(ti )T t̂i )
log PB
T
B i=1
j=1 exp(sg(ti ) t̂j )

Identity-bounded Matching

When a batch contains multiple image-text pairs of the same
identity, the model must perform pairwise matching between
all images and texts in the batch. Due to view variations,
significant differences among samples of the same identity
can lead to noisy matching of cross-view image-text pairs of
the same identity. Specifically, within the same identity, each
image and its corresponding text form a single-view strong
positive pair, while image-text pairs across different views
are considered as cross-view weak positive pairs. Imagetext pairs with different identities are categorized as negative
pairs. Existing matching losses often treat strong and weak
positive pairs equally, overlooking noisy correspondences of
weak positive pairs. To address the issue, we propose an
identity-bounded matching (IBM) loss to establish more precise cross-modal correspondences.
To fully leverage identity information, we use a P K sampling strategy to construct batch samples. For each batch,

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

we randomly sample K identities and then randomly select P images for each identity, with each image annotated
with the corresponding text. Thus, each batch contains B
= P K images and their corresponding texts. This results in
P K strong positive pairs, KP (P − 1) weak positive pairs,
and P 2 K(K − 1) negative pairs. We compute the cosine
PK
similarity for these image-text pairs, denoted as {ssp
i }i=1 ,
P 2 K(K−1)

KP (P −1)

{swp
, and {sni }i=1
respectively. To distini }i=1
guish different image-text pairs, we propose an IBM loss that
enforces the condition: ssp >swp >sn . This can be expressed
equivalently as: ssp >α, sn <β, β<swp <α. We further reformulate this condition as follows:
(ssp − α)>0, −(sn − β)>0, (swp − β)>0, −(swp − α)>0
(8)
where α and β are the upper and lower bounds for ssp and sn ,
with α>β. We further implement IBM loss based on logistic
loss as follows:
(P K
i
h
X
sp
1
LIBM =
log 1 + e−τsp (si −α)
P K i=1
KP (P −1)

+

X

h
i
wp
log 1 + e−τwp (si −β)

i=1

(9)

KP (P −1)

+

X

h

log 1 + e

τwp (swp
i −α)

i

i=1
P 2 K(K−1)

+

X

h

log 1 + e

i=1

τn (sn
i −β)


i


where τsp , τwp and τn are the temperature factors. Besides,
we compute the cross entropy loss Lid on image and text features to classify them by identity. This loss forces the network
to focus on the identity information, enabling it to correctly
recognize pedestrians of the same identity while distinguishing between those of different identities.

3.4

Optimization and Inference

The optimization of CoRL involves two stages. In the first
stage, we train a virtual-text embedding synthesizer, where
CLIP’s visual and textual encoders are frozen and only a fully
connected layer is optimized via Ls1
V ES to convert image embeddings to virtual-text tokens. We pre-extract and save features from the dataset, allowing us to focus on optimizing this
layer, reducing computational cost. In the second stage, VES
is frozen, and the backbone network and Adapter are optimized. The objective of the second stage is as follows:
s2
Ls2 = LIBM + Lid + Ls2
(10)
align + LCRCL
During inference, both visual and virtual-text features of
gallery images are extracted, and calculate their similarities
to the query text features. The final retrieval score is obtained
by summing these similarities without weighting.

4

Experiments

4.1

Experiment Settings

40,206 images and 80,412 descriptions of 13,003 people.
Each image has 2 descriptions, averaging 23 words. The
dataset is split into 34,054 images for training, 3,078 for validation, and 3,074 for testing. ICFG-PEDES [Ding et al.,
2021] consists of 54,522 image-text pairs of 4,102 persons,
with descriptions averaging 37 words. Training uses 34,674
pairs from 3,102 people, with the remaining 1,000 people reserved for evaluation. RSTPReid [Zhu et al., 2021] includes
20,505 images of 4,101 people, each with 2 descriptions averaging 23 words. Training includes 3,701 people, while validation and testing include 200 people each. Performance is
evaluated using Rank-k accuracy (R@k, k=1, 5, 10).
Implementation Details: Images are resized to 384×128
and augmented with random horizontal flipping, cropping
with padding, and random erasing. The maximum length of
the text sequence is set to 77, and random masking is employed for text augmentation. We use CLIP-ViT-B/16 as the
backbone. Temperature factors are set to τa = 0.02, τsp = 10,
τwp = 5, and τn = 40. Loss weight λ1 is 0.1, and the boundaries α and β in IBM loss are 0.6 and 0.4. Each mini-batch
comprises B = P × K images, , with P = 32 identities and
K = 4 images per identity. In the first stage, only a fully connected layer is optimized for 60 epochs using a cosine learning rate schedule, starting at 1×10−4 . In the second stage, we
fine-tune the visual/textual backbones with an initial learning
rate of 1 × 10−5 and the Adapter with 5 × 10−5 , also using a
cosine schedule and trained for 60 epochs. Both stages adopt
the Adam optimizer with a linear warm-up over the first 5
epochs. Experiments are implemented using the PyTorch library on a single NVIDIA RTX 3090 (24GB) GPU.

Datasets and Metrics: The evaluations are conducted on
three TIPR datasets. CUHK-PEDES [Li et al., 2017] has

2156

4.2

Comparisons with State-of-the-art Models

Table 1 compares our CoRL with current state-of-the-art
methods across three TIPR benchmarks: CUHK-PEDES,
ICFG-PEDES, and RSTPReid. Our CoRL achieves leading performance on these benchmarks, underscoring its effectiveness and advantages. On the CUHK-PEDES dataset,
CoRL achieves an R@1 accuracy of 78.15%, surpassing the
second-best AUL method by 0.92%. For the ICFG-PEDES
dataset, our CoRL sets a new state-of-the-art with R@1 and
R@5 accuracies of 69.50% and 85.63%, respectively. On
the RSTPReid dataset, CoRL delivers impressive results with
69.10%, 87.30%, and 92.90% on R@1, R@5, and R@10.
These results highlight the robustness and versatility of CoRL
across different scenarios. The superior performance of our
method is attributed to its innovative approach in leveraging
cross-modal complementarity and establishing precise crossmodal correspondences. CoRL effectively integrates textual information into images to enhance their discriminability
and bridge the modality gap, and employs identity-bounded
matching loss to address noisy correspondences among different types of image-text pairs.

4.3

Ablation Studies and Analysis

Effectiveness of different components: We conduct an ablation study to assess the effectiveness of various components on CUHK-PEDES in Table 2. 0# represents the result
of Baseline, which involves only the backbone network and
is trained using SDM and cross-entropy losses. 5# denotes

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Methods

Reference

CFine [Yan et al., 2023c]
VGSG [He et al., 2023]
IRRA [Jiang and Ye, 2023]
TCB [Zang et al., 2023]
DCEL [Li et al., 2023a]
RaSa [Bai et al., 2023]
CFAM [Zuo et al., 2024]
MACF [Sun et al., 2024]
TBPS-CLIP [Cao et al., 2024]
SAMC [Lu et al., 2024]
UMSA [Zhao et al., 2024b]
LSPM [Li et al., 2024a]
IRLT [Liu et al., 2024]
DCGA [Zhao et al., 2024a]
MDRL [Yang et al., 2024]
Propot [Yan et al., 2024]
DP [Song et al., 2024]
RDE [Qin et al., 2024]
FSRL [Wang et al., 2024]
APTM [Yang et al., 2023]
MLLM [Tan et al., 2024]
AUL [Li et al., 2024b]
CoRL (Ours)

TIP’23
TIP’23
CVPR’23
MM’23
MM’23
IJCAI’23
CVPR’24
IJCV’24
AAAI’24
TIFS’24
AAAI’24
TMM’24
AAAI’24
ICASSP’24
AAAI’24
MM’24
AAAI’24
CVPR’24
ICMR’24
MM’23
CVPR’24
AAAI’24
IJCAI’25

CUHK-PEDES
R@1 R@5 R@10
69.57 85.93
91.15
71.38 86.75
91.86
73.38 89.93
93.71
74.45 90.07
94.66
75.02 90.89
94.52
76.51 90.29
94.25
72.87 88.61
92.87
73.33 88.57
93.02
73.54 88.19
92.35
74.03 89.18
93.31
74.25 89.83
93.58
74.38 89.51
93.42
74.46 90.19
94.01
74.54 90.22
95.10
74.56 92.56
96.30
74.89 89.90
94.17
75.66 90.59
94.07
75.94 90.63
94.04
74.86 89.97
94.14
76.17 89.47
93.57
76.82 91.16
94.46
77.23 90.43
94.41
78.15 92.16
95.57

ICFG-PEDES
R@1 R@5 R@10
60.83 76.55
82.42
63.05 78.43
84.36
63.46 80.25
85.82
61.60 76.33
81.90
64.88 81.34
86.72
65.28 80.40
85.12
62.17 79.57
85.32
62.95 79.93
85.04
65.05 80.34
85.47
63.68 79.69
85.21
65.62 80.54
85.83
64.40 79.96
85.41
64.72 81.35
86.31
66.56 85.04
86.98
65.88 85.25
90.38
65.12 81.57
86.97
65.61 81.73
86.95
67.60 82.47
87.17
64.93 80.71
86.19
68.22 82.87
87.50
67.05 82.16
87.33
69.16 83.32
88.37
69.50 85.63
88.86

R@1
50.55
60.20
65.80
61.35
66.90
59.40
61.95
60.80
63.40
61.49
60.49
61.87
62.48
65.00
60.65
66.45
68.50
71.65
69.10

RSTPReid
R@5 R@10
72.50
81.60
81.30
88.20
82.85
88.20
83.95
90.45
86.50
91.35
81.35
88.50
83.55
88.75
82.35
89.00
83.30
90.30
82.26
89.23
80.60
89.32
83.63
89.70
83.77
89.93
84.75
90.60
83.05
89.60
85.60
90.60
87.15
92.10
87.55
92.05
87.30
92.90

Table 1: Performance comparison with state-of-the-art methods on three TIPR benchmarks. The first part lists non-pretrained methods, while
the following section presents pretrained methods. R@1, R@5, and R@10 are listed.
No.
0#
1#
2#
3#
4#
5#
6#
7#
8#
9#

IBM

VES

CMC

Pre

✓
✓

✓
✓
✓

✓
✓

✓
✓

✓
✓
✓

✓
✓

✓
✓
✓
✓
✓

R@1
70.63
74.66
73.58
74.24
75.48
75.02
77.32
75.89
76.41
78.15

R@5
87.67
89.46
89.08
89.70
90.45
90.21
91.76
90.33
90.77
92.16

Ls1
mse
✓

R@10
92.27
93.47
93.29
93.54
94.22
93.92
94.81
94.04
94.51
95.57

✓
✓
✓

Ls1
nce

Ls1
cor

✓
✓
✓
✓

✓
✓
✓

R@1
73.31
73.18
73.45
73.78
73.64
74.24

R@5
89.31
89.39
89.46
89.43
89.38
89.70

R@10
93.29
93.58
93.50
93.44
93.36
93.54

Table 3: Effectiveness of different losses in VES on CUHK-PEDES.

Table 2: Effectiveness of different components on CUHK-PEDES.

Baseline with additional pre-training (Pre) on LUPersonMLLM [Tan et al., 2024]. The SDM [Jiang and Ye, 2023]
loss treats image-text pairs of the same identity equally, leading to noisy correspondences and poor performance. Comparing 1# (6#) and 0# (5#), our IBM loss creates distinct
boundaries for differentiating various categories of imagetext pairs and establishes more precise cross-modal correspondences, achieving a 4.03% (2.30%) R@1 improvement.
Comparing 2# (7#) and 0# (5#), VES generates a virtual-text
feature for each image to enhance its image feature, resulting in a 2.95% (0.87%) R@1 improvement. Besides, CMC
facilitates information exchange between image and virtualtext features, allowing them to benefit from each other and
resulting in an additional 0.66% (0.52%) improvement. The
combination of these components achieves an R@1 accuracy
of 75.48% (78.15%), surpassing all methods listed in Table 1
across various settings. These results underscore the positive

2157

contributions of each component to the overall performance.
Effectiveness of different losses in VES. Table 3 summarizes the impact of different losses in VES, leading to the fols1
lowing conclusions: (1) Ls1
mse and Lnce jointly align virtual
and actual text embeddings at both feature and semantic levels, which is crucial for ensuring that VES generates virtualtext embeddings aligned with actual text. (2) The introduction of Ls1
cor evidently improves performance, highlighting
its importance in preserving information during conversion.
(3) combining all losses effectively ensures the generation of
high-quality virtual-text embeddings.
Impact of boundaries α and β in IBM: Figure 3 illustrates the effects of α and β. α defines the boundary between
strong and weak positive pairs. A large α may cause the
model to ignore weak positives, while a small α may disrupt the cross-view matching for weak positive pairs. We
set α = 0.6. β denotes the boundary between weak positive and negative pairs. An improper β can introduce noisy
correspondences—too large makes the model overly tolerant
to negatives, while too small blurs the distinction with weak
positives. We set β = 0.4 to balance these trade-offs.

75

75

74.5

74.5

74

R@1 (%)

R@1 (%)

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

73.5
73

74
73.5
73
72.5

72.5
72

Query

0.4

0.5

0.6

0.7

0.8

72

0.9

0.1

0.2

0.3

0.4

0.5

0.6

Figure 3: Effects of boundaries α (left) and β (right) of IBM loss on
CUHK-PEDES.
Method
SDM
IBM∗
IBM (Ours)

R@1
70.63
73.03
74.66

R@5
87.67
87.75
89.46

R@10
92.27
92.66
93.47

Effectiveness of IBM. As shown in Table 4, the SDM loss
treats strong and weak positive pairs equally, and its performance heavily relies on the number of negative pairs within
a batch. When using the P K sampling strategy to construct
batches, performance deteriorates significantly due to the reduced number of negative pairs. We modified IBM by removing the middle two lines of Equation 9 (IBM∗ ) to treat
strong and weak positive pairs equally, which led to a notable
performance drop (-1.63%). This indicates the importance
of distinguishing between strong and weak positive pairs for
learning accurate cross-modal correspondence. Despite this
drop, IBM∗ still outperforms SDM significantly and is less
affected by the number of negative pairs. Our IBM loss offers greater robustness for cross-modal retrieval tasks.
Computational Complexity: We compare the computational cost and inference time of CoRL with classic methods in Table 5. TIPCB and CFine build fine-grained correspondences by learning local features, reducing retrieval efficiency due to pairwise similarity calculations. IRRA and
Propot enhance global features with auxiliary fine-grained
tasks, improving retrieval efficiency but increasing parameters and storage. In contrast, CoRL avoids fine-grained tasks,
reducing parameters. The virtual-text feature in CoRL adds
minimal inference time, balancing accuracy and efficiency.
Qualitative Results: We qualitatively assess the effectiveness of our CoRL in Figure 4, showcasing the Top-10 retrieved images for each query text using both Baseline and
CoRL. The comparison reveals that our CoRL outperforms
Baseline even in cases where Baseline fails, ensuring that imParms
155.26
184.75
204.74
194.54
245.91
155.66

FLOPs
20.27
43.86
27.69
26.36
37.35
37.60

Time
18.7s
25.1s
37.2s
18.7s
18.7s
20.8s

This elderly gentleman has
white hair. He wears a shortsleeved shirt of light-colored
plaid. His pants and shoes are
black. A black and white jacket
is held at his side by one hand.

The woman is wearing a white
vest over a grey and black dress
with pointed shoes and white
socks.

Table 4: Effectiveness of IBM on CUHK-PEDES.

Method
Baseline
TIPCB [Chen et al., 2022]
CFine [Yan et al., 2023c]
IRRA [Jiang and Ye, 2023]
Propot [Yan et al., 2024]
CoRL (Ours)

Top-10 Retrieval Results

There is a male and female
walking down path interlocking
arms. Both wearing foot wear
blue in color while the female is
dressed in white dress, male is
dressed in grey shirt and shorts.

R@1
70.63
64.26
69.57
73.38
74.89
78.15

Small framed woman wearing
white top, white cardigan and
long grey skirt is hurrying
along.

Figure 4: Retrieval results of Baseline (the 1st row) and CoRL (the
2nd row) on CUHK-PEDES. The matched and mismatched person
images are marked with green and red rectangles, respectively.

ages with the same identity as the given query text are ranked
highly. This success is attributed to our method’s ability to
effectively exploit cross-modal complementarity to learn discriminative and modality-shared feature representations, and
establish precise cross-modal correspondences.

5

Conclusion

In this paper, we propose leveraging multi-modal complementarity to improve the TIPR task. Specifically, we introduce a virtual-text embedding synthesizer that generates
high-quality virtual-text embeddings aligned with actual text
from images. These virtual-text embeddings replace actual
texts, allowing us to explore multimodal complementarity
with images and eliminating the dependency on actual texts.
Meanwhile, we design a dual-branch cross-modal collaborative learning process equipped with a cross-modal relation
consistency loss, which enforces the exchange of information and mutual enhancement between image and virtual-text
features, resulting in more discriminative and robust image
features for retrieval. Additionally, we propose an identitybounded matching loss to distinguish between different types
of image-text pairs, establishing more accurate cross-modal
correspondences. The superior performance of CoRL across
multiple TIPR benchmarks underscores its effectiveness.

Acknowledgments

Table 5: Computational complexity comparison with several stateof-the-art methods on CUHK-PEDES.

2158

This work was supported in part by the National Natural
Science Foundation of China under Grant 62172212 and
Grant 62427819, the Natural Science Foundation of Jiangsu
Province under Grant BK20230031, the Jiangsu Provincial Science and Technology Major Project under Grant
BG2024042.

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

References
[Bai et al., 2023] Yang Bai, Min Cao, Daming Gao, Ziqiang
Cao, Chen Chen, Zhenfeng Fan, Liqiang Nie, and Min
Zhang. Rasa: Relation and sensitivity aware representation learning for text-based person search. In International
Joint Conference on Artificial Intelligence, IJCAI, 2023.
[Cao et al., 2024] Min Cao, Yang Bai, Ziyin Zeng, Mang Ye,
and Min Zhang. An empirical study of clip for text-based
person search. In AAAI Conference on Artificial Intelligence, AAAI, 2024.
[Chen et al., 2022] Yuhao Chen, Guoqing Zhang, Yujiang
Lu, Zhenxing Wang, and Yuhui Zheng. TIPCB: A simple but effective part-based convolutional baseline for
text-based person search. Neurocomputing, 494:171–181,
2022.
[Devlin et al., 2019] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of
deep bidirectional transformers for language understanding. In The North American Chapter of the Association
for Computational Linguistics, NAACL, 2019.
[Ding et al., 2021] Zefeng Ding, Changxing Ding, Zhiyin
Shao, and Dacheng Tao. Semantically self-aligned network for text-to-image part-aware person re-identification,
2021.
[Dong et al., 2024a] Neng Dong, Shuanglin Yan, Hao Tang,
Jinhui Tang, and Liyan Zhang. Multi-view information integration and propagation for occluded person reidentification. Information Fusion, 104:102201, 2024.
[Dong et al., 2024b] Neng Dong, Liyan Zhang, Shuanglin
Yan, Hao Tang, and Jinhui Tang. Erasing, transforming, and noising defense network for occluded person reidentification. IEEE Transactions on Circuits and Systems
for Video Technology, 34(6):4458–4472, 2024.
[Dosovitskiy et al., 2021] Alexey Dosovitskiy, Lucas Beyer,
Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and
Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International
Conference on Learning Representations, ICLR, 2021.
[Gong et al., 2022] Yunpeng Gong, Liqing Huang, and Lifei
Chen. Person re-identification method based on color attack and joint defence. In IEEE conference on computer
vision and pattern recognition, CVPR, 2022.
[Gong et al., 2024] Yunpeng Gong, Zhun Zhong, Yansong
Qu, Zhiming Luo, Rongrong Ji, and Min Jiang. Crossmodality perturbation synergy attack for person reidentification. In Advances in neural information processing systems, NeurIPS, 2024.
[He et al., 2023] Shuting He, Hao Luo, Wei Jiang, Xudong
Jiang, and Henghui Ding. Vgsg: Vision-guided semanticgroup network for text-based person search. IEEE Transactions on Image Processing, 2023.
[Jiang and Ye, 2023] Ding Jiang and Mang Ye. Cross-modal
implicit relation reasoning and aligning for text-to-image

2159

person retrieval. In IEEE Conference on Computer Vision
and Pattern Recognition, CVPR, 2023.
[Li et al., 2017] Shuang Li, Tong Xiao, Hongsheng Li, Bolei
Zhou, Dayu Yue, and Xiaogang Wang. Person search
with natural language description. In IEEE Conference on
Computer Vision and Pattern Recognition, CVPR, 2017.
[Li et al., 2019a] Huafeng Li, Shuanglin Yan, Zhengtao Yu,
and Dapeng Tao. Attribute-identity embedding and selfsupervised learning for scalable person re-identification.
IEEE Transactions on Circuits and Systems for Video
Technology, 30(10):3472–3485, 2019.
[Li et al., 2019b] Liunian Harold Li, Mark Yatskar, Da Yin,
Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple
and performant baseline for vision and language, 2019.
[Li et al., 2021] Junnan Li,
Ramprasaath Selvaraju,
Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and
Steven Chu Hong Hoi. Align before fuse: Vision and
language representation learning with momentum distillation. In Advances in Neural Information Processing
Systems, NeruIPS, 2021.
[Li et al., 2023a] Shenshen Li, Xing Xu, Yang Yang, Fumin
Shen, Yijun Mo, Yujie Li, and Heng Tao Shen. Dcel: Deep
cross-modal evidential learning for text-based person retrieval. In ACM International Conference on Multimedia,
MM, 2023.
[Li et al., 2023b] Shuang Li, Fan Li, Jinxing Li, Huafeng Li,
Bob Zhang, Dapeng Tao, and Xinbo Gao. Logical relation inference and multiview information interaction for
domain adaptation person re-identification. IEEE Transactions on Neural Networks and Learning Systems, 2023.
[Li et al., 2024a] Jiayi Li, Min Jiang, Jun Kong, Xuefeng
Tao, and Xi Luo. Learning semantic polymorphic mapping for text-based person retrieval. IEEE Transactions on
Multimedia, pages 1–14, 2024.
[Li et al., 2024b] Shenshen Li, Chen He, Xing Xu, Fumin
Shen, Yang Yang, and Heng Tao Shen.
Adaptive
uncertainty-based learning for text-based person retrieval.
In AAAI Conference on Artificial Intelligence, AAAI, 2024.
[Liu et al., 2024] Yu Liu, Guihe Qin, Haipeng Chen, Zhiyong Cheng, and Xun Yang. Causality-inspired invariant
representation learning for text-based person retrieval. In
AAAI Conference on Artificial Intelligence, AAAI, 2024.
[Lu et al., 2024] Zefeng Lu, Ronghao Lin, and Haifeng Hu.
Mind the inconsistent semantics in positive pairs: Semantic aligning and multimodal contrastive learning for textbased pedestrian search. IEEE Transactions on Information Forensics and Security, 19:6409–6424, 2024.
[Qin et al., 2024] Yang Qin, Yingke Chen, Dezhong Peng,
Xi Peng, Joey Tianyi Zhou, and Peng Hu. Noisycorrespondence learning for text-to-image person reidentification. In IEEE Conference on Computer Vision
and Pattern Recognition, CVPR, 2024.
[Radford et al., 2021] Alec Radford, Jong Wook Kim, Chris
Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Cross-modal Full-mode Fine-grained Alignment for
Text-to-Image Person Retrieval
HAO YIN and XIN MAN, Shenzhen Institute for Advanced Study, University of Electronic Science and
Technology of China, China

FEIYU CHEN, JIE SHAO∗ , and HENG TAO SHEN, University of Electronic Science and Technology

arXiv:2509.13754v2 [cs.CV] 28 Dec 2025

of China, China Sichuan Artificial Intelligence Research Institute, China
Text-to-Image Person Retrieval (TIPR) is a cross-modal matching task designed to identify the person images
that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
between the textual and visual modalities within a unified latent representation space. To address this
challenge, prior approaches incorporate attention mechanisms for implicit cross-modal local alignment.
However, they lack the ability to verify whether all local features are correctly aligned. Moreover, existing
methods tend to emphasize the utilization of hard negative samples during model optimization to strengthen
discrimination between positive and negative pairs, often neglecting incorrectly matched positive pairs. To
mitigate these problems, we propose FMFA, a cross-modal Full-Mode Fine-grained Alignment framework,
which enhances global matching through explicit fine-grained alignment and existing implicit relational
reasoning—hence the term “full-mode”—without introducing extra supervisory signals. In particular, we
propose an Adaptive Similarity Distribution Matching (A-SDM) module to rectify unmatched positive sample
pairs. A-SDM adaptively pulls the unmatched positive pairs closer in the joint embedding space, thereby
achieving more precise global alignment. Additionally, we introduce an Explicit Fine-grained Alignment
(EFA) module, which makes up for the lack of verification capability of implicit relational reasoning. EFA
strengthens explicit cross-modal fine-grained interactions by sparsifying the similarity matrix and employs a
hard coding method for local alignment. We evaluate our method on three public datasets, where it attains
state-of-the-art results among all global matching methods. The code for our method is publicly accessible at
https://github.com/yinhao1102/FMFA.
CCS Concepts: • Information systems → Image search; • Computing methodologies → Object identification.
Additional Key Words and Phrases: Cross-modal retrieval, Person search, Fine-grained alignment
ACM Reference Format:
Hao Yin, Xin Man, Feiyu Chen, Jie Shao, and Heng Tao Shen. 2025. Cross-modal Full-mode Fine-grained
Alignment for Text-to-Image Person Retrieval. ACM Trans. Multimedia Comput. Commun. Appl. 1, 1, Article 1
(January 2025), 22 pages. https://doi.org/10.1145/3786798
∗ Corresponding author.

Authors’ Contact Information: Hao Yin, yinhao1102@std.uestc.edu.cn; Xin Man, manxin@std.uestc.edu.cn, Shenzhen
Institute for Advanced Study, University of Electronic Science and Technology of China, Shenzhen, China; Feiyu Chen,
chenfeiyu@uestc.edu.cn; Jie Shao, shaojie@uestc.edu.cn; Heng Tao Shen, shenhengtao@hotmail.com, University of Electronic Science and Technology of China, Chengdu, China and Sichuan Artificial Intelligence Research Institute, Yibin,
China.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/1-ART1
https://doi.org/10.1145/3786798
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

1:2

1

Yin et al.

Introduction

Text-to-Image Person Retrieval (TIPR) seeks to understand natural language descriptions and
identify the most relevant person image within a large gallery [31]. Unlike general image-text
retrieval [5, 6, 44, 47, 50], which tends to achieve semantic-based matching between text and image,
TIPR is specifically designed for identifying individuals. TIPR requires the accurate modeling of
fine-grained correspondences between textual and visual modalities, owing to the large intra-class
variance and small inter-class difference. This substantial intra-class variation arises from two
aspects: (1) visual appearances of the same identity exhibit dramatic variations under different poses,
viewpoints, and illumination conditions, and (2) textual descriptions are influenced by differences
in phrasing, word order and textual ambiguities. Therefore, the primary challenges in TIPR are
how to extract discriminative global representations from image-text pairs and how to achieve
precise cross-modal fine-grained alignment. Existing methods for tackling these challenges can be
roughly divided into two main categories: global matching methods and local matching methods.
Some global matching methods [58, 59] obtain discriminative global representations by aligning
images and texts, which are projected into a joint embedding space. Their widely adopted loss
functions include the Cross-Modal Projection Matching (CMPM) loss [58] and the Similarity
Distribution Matching (SDM) loss [21]. The CMPM loss highlights the gap between the scalar
projections of image-text pairs and their matched label indicators. In comparison, the SDM loss
boosts global matching performance by minimizing the Kullback-Leibler (KL) divergence between
the normalized similarity profile of image-text pairs and the true label distribution. In addition,
the SDM loss incorporates a temperature hyperparameter to make model updates concentrate
on hard negative samples, yet it leads to the neglect of unmatched positive pairs, as shown in
Figure 1a. However, in TIPR, the accurate matching of positive pairs is prioritized over merely
distinguishing between positive and negative pairs. Meanwhile, some local matching methods
[2, 11, 35] incorporate attention mechanisms to achieve cross-modal fine-grained alignment. For
instance, RaSa [2] constructs a cross-modal encoder to generate multimodal representations for
subsequent fine-grained alignment. Building on RaSa, MARS [11] integrates a Masked AutoEncoder
(MAE) decoder [14] to reconstruct masked image patch sequences into their original unmasked
form, thereby facilitating cross-modal fine-grained alignment. However, these methods rely on
attention mechanisms to implicitly aggregate local image-text representations. As a result, they
yield only the final multimodal representation, without revealing the details of the aggregation
process. Consequently, these implicit aggregation methods make it difficult to determine whether
the aggregated multimodal representations correctly encode the corresponding visual and textual
information.
To remedy these concerns, we propose FMFA, a cross-modal Full-Mode Fine-grained Alignment framework, which enhances global matching through full-mode fine-grained alignment,
including explicit fine-grained image-text alignment and existing implicit relational reasoning.
Specifically, we design an Adaptive Similarity Distribution Matching (A-SDM) module to
ensure the correct matching of positive image-text pairs. Within the joint embedding space, the
A-SDM module adaptively pulls positive pairs closer together. In cases of mismatched positive pairs,
the A-SDM module adaptively regulates the pulling force based on their relative distance within
the joint embedding space, as shown in Figure 1b, thus improving cross-modal global alignment.
Based on the insight that each word in a caption can be associated with several image patches [3],
we introduce an Explicit Fine-Grained Alignment (EFA) module. The EFA module derives multimodal representations through explicit aggregation with a sparse similarity matrix. During this
process, the sparse similarity matrix between text and image reflects the contribution of textual and
visual representations to the final multimodal representation. To minimize redundancy and reduce

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

Cross-modal Full-mode Fine-grained Alignment for Text-to-Image Person Retrieval

1:3

Query text
Query text

Matched positive image
Matched positive image

Unmatched positive image

Unmatched positive image

Negative image
Negative image

Hard negative image
Hard negative image

Adaptive Pull

(a) Existing global matching methods.

(b) Our adaptive global matching method.

Fig. 1. Evolution of global matching methods for text-to-image person retrieval. (a) Existing global matching
methods focus on hard negative samples to learn a discriminative boundary in a common latent space, thus
enhancing the distinction between positive and negative samples. (b) Our adaptive global matching method
builds on a discriminative boundary and concentrates on unmatched positive samples, adaptively pulling
them closer to the corresponding query text.

the computational cost during training, the EFA module employs hard coding alignment between
the aggregated multimodal representation and its original visual and textual representations. These
designs allow EFA to realize fine-grained cross-modal interactions and assist the backbone network
in learning more distinctive global image-text representations without introducing additional
supervision. FMFA is evaluated on three public benchmarks [10, 27, 60], and it attains competitive
top-level performance along with high inference efficiency. We highlight our key contributions
below:
• We introduce FMFA to explicitly leverage fine-grained interactions for improving cross-modal
alignment, without incurring extra supervision or inference overhead.
• We present an adaptive similarity distribution matching module aimed at precisely aligning
image-text pairs in a shared embedding space. It adaptively adjusts to narrow the distance
between mismatched positive pairs, ensuring more precise matching.
• We develop an explicit fine-grained alignment module, which leverages the sparse similarity
matrix for explicit aggregation and employs a hard coding method in cross-modal fine-grained
alignment to minimize redundant information.
2

Related Work

Text-to-Image Person Retrieval (TIPR) was initially proposed by Li et al. [27], who created
the CUHK-PEDES dataset. Unlike visual-based person retrieval [7, 8, 16, 56], the core challenge of
TIPR lies in constructing a shared latent space that enables coherent alignment between visual and
textual representations. Existing methods can be typically classified into global and local matching
approaches.
Early global methods [59, 60] directly aligned the global representations of images and text in
a joint embedding space. Schroff et al. [39] proposed a triplet ranking loss to enforce a margin
constraint between positive and negative pairs, and Zhang et al. [58] introduced the CMPM/C loss
to minimize the discrepancy between the scalar projection of image-text pairs and their labels.
However, these global methods lack cross-modal fine-grained interactions, which restrict their
ability to capture detailed semantic correspondences. To address this limitation, early local matching
methods [12, 42, 49] explicitly aligned local visual and textual features to achieve fine-grained
cross-modal interactions. Nevertheless, they rely on unimodal pre-trained models (e.g., BERT [9]
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

1:4

Yin et al.

and ResNet [15]), failing to exploit the strong cross-modal alignment capability of recent pre-trained
Vision-Language Models (VLMs) [25, 26, 53].
Recent local matching methods [11, 19, 32, 36, 52] have benefited greatly from VLMs and introduced VLMs to enhance cross-modal alignment. Park et al. [35] utilized a modified Contrastive
Language-Image Pre-training (CLIP) [37] model as the feature extractor and designed a slot attentionbased [30] part discovery module to identify discriminative human parts without extra supervision,
while Bai et al. [2] used the align-before-fuse model [26] as the backbone and introduced a crossmodal encoder for fine-grained alignment. Although effective, these methods involve complex
computations during inference, leading to high time and memory costs, which limit their applicability to real-time systems.
On another line of research, several studies [41, 45, 55] have explored leveraging large-scale
image-text pairs in the person Re-IDentification (ReID) domain to VLMs. Zuo et al. [61] utilized
CUHK-PEDES and ICFG-PEDES to train an image captioner, aiming to generate comprehensive
textual descriptions for pedestrian images. Yang et al. [55] employed BLIP-2 [24] to produce
attribute-aware captions for diffusion-generated pedestrian images [38], while Jiang et al. [22]
leveraged recent Multi-modal Large Language Models (MLLMs), such as Qwen-VL [1] and LLaVA
[28], to automatically annotate large-scale ReID datasets in a human-like manner. The CLIP models
pre-trained on large-scale ReID datasets exhibit strong zero-shot performance. Their compatibility
with global matching methods—which relies solely on global features and has a simple inference
pipeline—makes them particularly suitable for direct fine-tuning in such settings.
Recent global matching methods [17, 21, 43] have integrated local fine-grained alignment modules
into global matching frameworks to obtain more discriminative global representations. Shu et al.
[43] introduced a bidirectional mask modeling mechanism that randomly masks image patches
and text words, encouraging the model to infer missing semantics and implicitly learn local
visual-textual correspondences. He et al. [17] proposed the Vision-Guided Semantic-Group (VGSG)
network to cluster textual tokens into semantic groups and align them with corresponding visual
regions under the guidance of vision features, achieving group-level fine-grained alignment within
a global representation space. Similarly, Jiang et al. [21] developed IRRA to employ an Implicit
Relation Reasoning (IRR) module based on attention mechanisms to capture latent cross-modal
relations, enhancing global alignment. Although these methods enhance fine-grained cross-modal
interactions within global matching frameworks, their implicit or group-level alignment strategies
may still fail to guarantee precise local correspondences. In light of these limitations, we propose
FMFA, which aims to enhance the global matching ability of the model by achieving cross-modal
full-mode fine-grained alignment, including explicit fine-grained alignment and implicit relation
reasoning.
3

Method

This section introduces the proposed FMFA framework. Figure 2 presents an overview of FMFA,
and further details of the framework are elaborated in the subsequent subsections.
3.1

Feature Extraction

Motivated by the success of IRRA [21], we use the modified full CLIP [37] visual and textual
encoders to enhance cross-modal alignment capabilities while reducing inference costs.
Visual Modality. Given an input image 𝐼 ∈ R𝐻 ×𝑊 ×𝐶 , we employ a CLIP-pretrained Vision
Transformer (ViT) to attain its image representation. An image is first divided into 𝑁 = 𝐻 × 𝑊 /𝑃 2
distinct patches of size 𝑃 × 𝑃, which are then transformed into one-dimensional token embeddings
𝑁 via a learnable linear projection. After adding positional encodings and a [CLS] token,
{𝑓𝑖𝑣 }𝑖=1
𝑣 , 𝑓 𝑣 , . . . , 𝑓 𝑣 } is passed through 𝐿 transformer layers to capture dependencies
the sequence {𝑓𝑐𝑙𝑠
1
𝑁
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

Cross-modal Full-mode Fine-grained Alignment for Text-to-Image Person Retrieval

1:5

CLIP
ViT-B/16
ASDM

EFA

IRR

CLIP
Xformer

Coarse-grained
Alignment

ID

A woman with long
black hair is wearing
a black patterned
shirt, black capri
pants and is carrying
a purple purse and a
green bag.

Fine-grained
Alignment

(a) The architecture of FMFA.
Hard Coding
Alignment
Similarity
Score
Adaptive
Weight

Sparse Similarity
Matrix Aggregation

KL

[EOS]
Embedding
[CLS]
Embedding

Hard Coding
Alignment
True Matching Label

(b) Adaptive Similarity Distribution Matching (ASDM).

Token
Embedding

Patch
Embedding

Language-grouped
Vision Embedding

(c) Explicit Fine-grained Alignment (EFA).

Fig. 2. The illustration of our FMFA framework. (a) Our FMFA contains a two-stream feature extraction
network and four distinct modules for structured representation learning, namely Identity Identification (ID
loss), Adaptive Similarity Distribution Matching (A-SDM), Explicit Fine-grained Alignment (EFA) and Implicit
Relation Reasoning (IRR). The former two are coarse-grained alignment modules and the latter two are finegrained alignment modules. Modules linked via dashed connections are omitted during inference. (b) A-SDM
obtains the adaptive weight through the similarity score of unmatched positive pairs, adaptively pulling
positive pairs closer and dynamically adjusting the pulling force. (c) EFA achieves fine-grained interaction
by hard coding alignment of token or patch embeddings and their corresponding language-grouped vision
embedding, which is derived from an aggregated sparse similarity matrix.

𝑣 is linearly mapped into the joint image-text
among patches. Finally, the [CLS] token embedding 𝑓𝑐𝑙𝑠
embedding space, producing the compact global feature of the image.
Textual Modality. Given an input text 𝑇 , we utilize the CLIP-Xformer textual extractor [37] to
obtain its embedding. The text is first tokenized through lower-cased Byte Pair Encoding (BPE)
[40] and framed with [SOS] and [EOS] tokens to indicate sequence boundaries. The resulting token
𝑡 , 𝑓 𝑡 , . . . , 𝑓 𝑡 } is processed by the transformer encoder, which models dependencies
sequence {𝑓𝑠𝑜𝑠
𝑒𝑜𝑠
1
among tokens via masked self-attention. Finally, the [EOS] token embedding from the top layer,
𝑡 , is linearly mapped into the joint image-text representation space, generating a compressed
𝑓𝑒𝑜𝑠
global textual representation.

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

1:6

3.2

Yin et al.

Adaptive Similarity Distribution Matching

Adopted from IRRA [21], we introduce a novel Adaptive Similarity Distribution Matching (A-SDM)
module, which aims to adaptively pull the unmatched positive image-text pairs into a shared
representation space, further enhancing the cross-modal global matching capability of the model.
Let the mini-batch contain 𝐵 image-text pairs, we pair each text embedding 𝑔𝑖𝑡 with its global
image embedding 𝑔𝑣𝑗 to form the set {(𝑔𝑖𝑡 , 𝑔𝑣𝑗 ), 𝑦𝑖,𝑗 }𝐵𝑗=1 , where 𝑦𝑖,𝑗 serves as the matching indicator.
Specifically, 𝑦𝑖,𝑗 = 1 denotes a matched pair, while 𝑦𝑖,𝑗 = 0 denotes an mismatched pair. Let
𝑐𝑜𝑠 (a, c) = a⊤ c/∥a∥ ∥c∥ denotes the similarity of a and c. Subsequently, like SDM [21], the similarity
matrix of image-text pairs is obtained through the following softmax function:
𝑒𝑥𝑝 (𝑐𝑜𝑠 (𝑔𝑖𝑡 , 𝑔𝑣𝑗 )/𝜏1 )
𝑝𝑖,𝑗 = Í𝐵
,
𝑡 𝑣
𝑘=1 𝑒𝑥𝑝 (𝑐𝑜𝑠 (𝑔𝑖 , 𝑔𝑘 )/𝜏1 )

(1)

where 𝜏1 acts as a temperature term that modulates the spread of the resulting distribution. The
probability 𝑝𝑖,𝑗 quantifies how much the similarity between the text embedding 𝑔𝑖𝑡 and the image
embedding 𝑔𝑣𝑗 contributes relative to the sum of all similarities between 𝑔𝑖𝑡 and every image
embedding in the mini-batch.
Let the 𝑖 − 𝑡ℎ text 𝑇𝑖 from the batch be designated as the query text and 𝐼𝑖 be the corresponding
image for 𝑇𝑖 at rank-𝑘, where 𝑘 > 1. Different from IRRA [21], we propose to derive an adaptive
weighting factor by assessing the similarity between the query text 𝑇𝑖 and all image representations:


𝑡 2𝑖
𝑤𝑖 = 𝛼 · max 𝑝𝑖,𝑘 − 𝑝𝑖,𝑖 + 1,
(2)
𝑘

where 𝛼 is a weight factor reflecting the contribution of unmatched image-text pairs to the crossmodal global matching ability of the model. Here, max𝑘 𝑝𝑖,𝑘 indicates the top similarity value
between the text 𝑇𝑖 and every image within the mini-batch, while 𝑝𝑖,𝑖 refers to the similarity
associated with its corresponding positive image. The constant term “+1” ensures that when 𝑇𝑖
and its corresponding image 𝐼𝑖 are correctly matched, the weight 𝑤𝑖𝑡 2𝑖 defaults to 1. In this case,
the A-SDM loss reduces to the SDM loss [21], preventing overemphasis on correctly matched
pairs while allowing the model to focus adaptively on harder and misaligned pairs. Conversely,
𝑤𝑖𝑡 2𝑖 > 1 indicates that 𝑇𝑖 and 𝐼𝑖 are unmatched, increasing their contribution to the loss to enhance
global cross-modal alignment. The A-SDM loss for mapping text to image within a mini-batch is
subsequently formulated as:
L𝑡 2𝑖 = 𝑊 𝑡 2𝑖 ∗ 𝐾𝐿(pi ∥qi ) =

𝐵
𝐵
𝑝𝑖,𝑗
1 ∑︁ 𝑡 2𝑖 ∑︁
𝑤𝑖
𝑝𝑖,𝑗 log(
),
𝐵 𝑖=1
𝑞
𝑖,𝑗 + 𝜖
𝑗=1

(3)

Í𝐵
where 𝜖 is a tiny offset added to safeguard the computation from unstable values, and 𝑞𝑖,𝑗 = 𝑦𝑖,𝑗 / 𝑘=1
denotes the ground-truth matching probability.
In a complementary manner, the A-SDM loss for the image-to-text branch L𝑖2𝑡 is derived by
swapping the roles of the text and image features. The bi-directional A-SDM loss is formulated as:
L𝐴−𝑠𝑑𝑚 = L𝑖2𝑡 + L𝑡 2𝑖 .
3.3

(4)

Explicit Fine-grained Alignment

To effectively leverage fine-grained information, it is necessary to narrow the underlying disparity
between visual and textual modalities. Although many attention-based fine-grained alignment
approaches have shown effectiveness by implicitly associating local regions in images with textual
fragments, they provide no direct means to verify whether these localized correspondences are
accurately aligned. We propose an explicit cross-modal aggregation approach that leverages the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

Cross-modal Full-mode Fine-grained Alignment for Text-to-Image Person Retrieval

1:7

Hard
Coding

LSE
Pooling

Sparse Similarity Matrix
Token
Embedding

Cosine
Similarity

Patch
Embedding

Language-grouped
Vision Embedding

Dot Product
and Sum

(a) Sparse similarity matrix aggregation.

Token/Patch Embedding

Hard Similarity

Language-grouped Vision Embedding

Cosine Similarity

(b) Hard coding alignment.

Fig. 3. Illustration of the EFA module. (a) EFA imposes a sparse structure on the similarity matrix relating
token embeddings to patch embeddings, and obtains the language-grouped vision embeddings by aggregating
the sparse similarity with its corresponding patch. (b) EFA obtains hard similarity through hard coding and
LSE pooling, where the calculated hard similarity reflects the relationship between the language-grouped
vision embeddings and their original token or patch embeddings.

sparse similarity matrix between the local image and text features. To further reduce redundant
information and minimize memory and time costs during fine-grained alignment, we use hard
coding to align the aggregated language-grouped vision embeddings with both image and text
embeddings, as shown in Figure 2c.
Sparse Similarity Matrix Aggregation. Some methods [33, 57] incur substantial computational
and memory overhead, as they evaluate pairwise relationships between every image patch and
every text token, which limits scalability to large batch sizes. Therefore, we apply a sparsification
strategy to reduce the full pairwise similarity computation. While softmax is commonly used
for such sparse processing, it tends to produce low-entropy similarity distributions that impede
effective gradient flow [18]. Thus, we further adopt a max-min normalization scheme to achieve a
more stable and expressive sparse similarity aggregation.
An image 𝐼 and its corresponding text 𝑇 are encoded through the visual and textual encoders,
respectively. As presented in Figure 3a, the similarity between image patches and text tokens is
𝐿 of the text transformer and {𝑓 𝑣 } 𝑁
computed via the inner product of the last hidden states {𝑓𝑖𝑡 }𝑖=1
𝑖 𝑖=1
of the vision transformer. 𝑠𝑖,𝑗 = 𝑓𝑖𝑡 · 𝑓 𝑗𝑣 measures the similarity between the text token 𝑓𝑖𝑡 and the
image patch 𝑓 𝑗𝑣 , where · denotes the inner product. To obtain the aggregation weight, each token 𝑖
is first scaled to the range [0,1] through the following min-max normalization:
𝑠ˆ𝑖,𝑗 =

𝑠𝑖,𝑗 − min𝑘 𝑠𝑖,𝑘
.
max𝑘 𝑠𝑖,𝑘 − min𝑘 𝑠𝑖,𝑘

(5)

We sparsify the normalized similarity matrix to encourage cross-modal interactions between
each token and its patches with higher similarity:
(
𝑠ˆ𝑖,𝑗 if 𝑠ˆ𝑖,𝑗 ≥ 𝜎
𝑠˜𝑖,𝑗 =
,
(6)
0
otherwise
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

1:8

Yin et al.

where 𝜎 is the sparsity threshold. 𝜎 is assigned the value 1/𝑁 , where 𝑁 corresponds to the total
count of patches in the image. This ensures that each token has a minimum of one corresponding
image patch for alignment. We compute the aggregation weights by:
𝑠˜𝑖,𝑗
𝑎𝑔𝑔𝑖,𝑗 = Í𝑀

(7)

,

𝑚=1 𝑠˜𝑖,𝑗

where 𝑀 is the number of image patches retained with high similarity to the token 𝑖, and 𝑎𝑔𝑔𝑖,𝑗
quantifies the influence of patch 𝑗 in forming the language-grouped vision embedding (referred to as
joint embedding) associated with token 𝑖. This explicit aggregation strategy ensures a comprehensive
interaction between token 𝑖 and its corresponding patch 𝑗 during local alignment. In particular, the
aggregation weight 𝑎𝑔𝑔𝑖,𝑗 effectively captures the semantic relevance between token 𝑖 and patch 𝑗,
thereby facilitating precise alignment.
Next, we derive the corresponding joint embedding 𝑒𝑖 as:
𝑒𝑖 =

𝑁
∑︁

𝑎𝑔𝑔𝑖,𝑗 · 𝑓 𝑗𝑣 ,

(8)

𝑗=1

where 𝑁 is the count of image patches. The resulting set of joint embedding 𝑒𝑖 has the same length
𝐿 as the text token 𝑓𝑖𝑡 .
𝐿
Hard Coding Alignment. We calculate the similarity between the joint embeddings {𝑒𝑖 }𝑖=1
𝑡
𝐿
𝑣
𝑁
and their corresponding original text embeddings {𝑓𝑖 }𝑖=1 as well as image embeddings {𝑓𝑖 }𝑖=1 ,
respectively. To reduce both computational and memory costs, we adopt a hard coding similarity
computation between the joint embeddings and their corresponding text and image embeddings,
and the theoretical analysis of the hard coding is provided in Appendix A. For simplicity, we only
present the calculation between the joint embeddings and the text embeddings, while the remaining
computations follow a similar and symmetric approach.
For the text 𝑇 and its corresponding joint embedding 𝐸, we calculate the original similarity matrix
𝐿 and their joint embeddings {𝑒 }𝐿 , where 𝑜
𝑡 ⊤
𝑡
𝑂 between all text tokens {𝑓𝑖𝑡 }𝑖=1
𝑖 𝑖=1
𝑖,𝑗 = 𝑓𝑖 𝑒 𝑗 /∥ 𝑓𝑖 ∥ ∥𝑒 𝑗 ∥
𝑡
𝑡
means the cosine similarity of 𝑓𝑖 and 𝑒 𝑗 . For the token 𝑓𝑖 , we compute the weight factor between
it and all joint embeddings using the following hard coding way:


(𝑜𝑖,𝑗 ′ )
 1 if 𝑗 = argmax

𝑗 ′ =1···𝐿
𝜔𝑖,𝑗 =
.

 0 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒


(9)

Then, we utilize the LSE pooling [23] to compute the hard similarity between text 𝑇 and its
corresponding joint embedding 𝐸 by:
!
𝐿
∑︁
ℎ𝑎𝑟𝑑_𝑠 (𝑇 , 𝐸) = 𝐿𝑆𝐸 − 𝑃𝑜𝑜𝑙𝑖𝑛𝑔
𝜔𝑖,𝑗 𝑜𝑖,𝑗
𝑗=1
𝐿
∑︁

(10)




1
= log
exp 𝜆 max 𝑜𝑖,𝑗 ,
𝑗=1···𝐿
𝜆
𝑖=1
where 𝜆 controls the degree to which the most relevant text embeddings and their corresponding
joint embeddings are emphasized.
Given a batch containing 𝐵 text embeddings along with their associated joint embeddings, we
compute the hard coding similarity matrix 𝐻𝑎𝑟𝑑_𝑆 following Eq. (9) and Eq. (10), as illustrated in
Figure 3b. We calculate the EFA loss from the text to its joint embedding, adapted from the triplet
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-modality average precision optimization for visible thermal person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 164 (2025) 111489

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Cross-modality average precision optimization for visible thermal person
re-identification
Yongguo Ling a,e , Zhiming Luo b , Dazhen Lin b , Shaozi Li b , Min Jiang b , Nicu Sebe c ,
Zhun Zhong d ,∗
a

School of Computer, Electronics and Information, Guangxi University, Nanning 530004, China

b Department of Artificial Intelligence, Xiamen University, Xiamen 361005, China
c Department of Information Engineering and Computer Science, University of Trento, Trento 38123, Italy
d School of Computer Science and Information Engineering, Hefei University of Technology, 230009, Hefei, China
e

Guangxi Key Laboratory of Multimedia Communications and Network Technology, Nanning, 530004, Guangxi, China

ARTICLE

INFO

Keywords:
Person re-identification
Cross-modality
Average precision
Modality alignment

ABSTRACT
Metric learning has emerged as a popular approach for addressing the challenges of visible thermal person
re-identification (VT-ReID), such as the cross-modality discrepancy and intra-class variations. However, existing
metric learning-based methods often focus on optimizing the model for hard positive samples, neglecting the
importance of high-ranking ones, due to failing to consider the overall ranking order within a batch. To
overcome this limitation, we propose a novel approach called Cross-modality Average Precision (CAP) that
directly optimizes the cross-modality overall ranking order in VT-ReID. Unlike the recently introduced Smooth
Average Precision (Smooth-AP), which primarily corrects misordered samples at high ranks, CAP specifically
targets the main challenge of cross-modality discrepancy in VT-ReID. Our method involves setting a query
instance from one modality and calculating the CAP using galleries from another modality. CAP encompasses
two complementary aspects: CAP with Visible queries (CAPV) and CAP with Thermal queries (CAPT). By
jointly optimizing these two aspects, we can effectively improve the cross-modality overall ranking order.
Additionally, to enhance the effectiveness of CAP, we introduce two techniques. The first technique is Dynamic
Modality Alignment (DMA), which reduces the cross-modality discrepancy by adaptively adjusting the weights
of modality alignment. The second technique involves implementing CAP and DMA on the Global and Local
Features (GLF), enabling us to optimize the model at both global and local levels, further enhancing the
advantages of CAP and DMA. We conducted extensive experiments on two VT-ReID datasets, and the results
demonstrate the effectiveness of our proposed method, which achieves state-of-the-art performance.

1. Introduction
Person re-identification (ReID) is a task that involves matching a
specific query person from a set of gallery images captured by nonoverlapping cameras. Traditional ReID [1] assumes that both the query
and gallery images are obtained from RGB cameras. However, the
reliance on visible (RGB) images in illumination environments makes
them susceptible to significant changes under poor lighting conditions,
such as night-time. To address this issue, researchers have proposed
the collection of thermal images using thermal cameras in scenarios
with poor illumination [2]. In this paper, we focus on the problem of
matching person images between visible and thermal cameras, which
is commonly referred to as visible-thermal person re-identification (VTReID) in the research community [3]. Compared to traditional ReID,

VT-ReID presents more significant challenges. The primary difficulties
stem from the substantial inter-modality discrepancy caused by the utilization of different modalities and the presence of intra-class variations
resulting from environmental factors (e.g., illumination) and personal
changes (e.g., pose). (Fig. 1).
The primary objective of visible-thermal person re-identification
(VT-ReID) is to align features from different modalities and learn a
shared semantic embedding space that facilitates accurate matching
of person identities across modalities. To address this goal, several
metric learning methods [4,5] have been proposed to mitigate the
cross-modality discrepancy and intra-class variations. However, these
approaches typically perform gradient updates using a small number
of sample pairs, such as the triplet loss [6] (Fig. 2(a)) , leading to

∗ Corresponding author at: School of Computer Science and Information Engineering, Hefei University of Technology, 230009, Hefei, China.

E-mail addresses: ygling@gxu.edu.cn (Y. Ling), zhiming.luo@xmu.edu.cn (Z. Luo), dzlin@xmu.edu.cn (D. Lin), szlig@xmu.edu.cn (S. Li),
minjiang@xmu.edu.cn (M. Jiang), niculae.sebe@unitn.it (N. Sebe), zhunzhong@hfut.edu.cn (Z. Zhong).
https://doi.org/10.1016/j.patcog.2025.111489
Received 7 October 2023; Received in revised form 31 December 2024; Accepted 19 February 2025
Available online 28 February 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

Fig. 1. Illustration of the main challenges in VT-ReID. The inter-modality discrepancy
caused by different modalities and the intra-class variations caused by different
illumination, view, and pose.

Fig. 2. Illustration of the difference between triplet loss, original Smooth-AP, and
our Cross-modality Average Precision (CAP). Shapes indicate the identities and colors
represent the modalities (blue for visible and yellow for thermal). (a) Triplet loss
performs gradient updates with a couple of sample pairs, potentially encountering
difficulties in escaping local optima because of its constrained ability to capture the
overall ranking order. (b) Original Smooth-AP will prioritize optimizing the ranking
of the intra-modality instances (e.g. 𝑝1, 𝑝2) while ignoring the inter-modality ones
(e.g. 𝑝3) due to its inherent characteristics. (c) Our CAP considers handling the large
cross-modality gap in VT-ReID, which can directly optimize the cross-modality overall
ranking orders. (For interpretation of the references to color in this figure legend, the
reader is referred to the web version of this article.)

an over-focus on optimizing low-ranking (located at the rear of the
ranking list with lower similarity) positive samples at the expense
of high-ranking (located at the front of the ranking list with higher
similarity) ones, due to the failure to consider the overall ranking
order [7]. This issue is particularly problematic in VT-ReID due to
the simultaneous presence of cross-modality discrepancy and intraclass variations. Furthermore, these metric learning methods often rely
on intricate sampling strategies and pairing losses, requiring extensive
experimentation and empirical fine-tuning.
Recently, Smooth-AP [8] introduced an approximation method to
calculate differentiable average precision, which directly optimizes the
overall ranking order. However, the original Smooth-AP fails to address
the main challenge of cross-modality in VT-ReID. Specifically, SmoothAP prioritizes correcting misordered samples at high ranks [8], while
in the cross-modality VT-ReID task, the inter-modality distance is typically larger than the intra-modality distance. As a result, it prioritizes
optimizing the ranking of intra-modality instances (high rank 𝑝1, 𝑝2)
while disregarding the ranking of inter-modality instances (low rank
𝑝3) (Fig. 2(b)), leading to inferior performance.
To overcome these limitations, we propose a differentiable Crossmodality Average Precision (CAP) method that explicitly addresses
the substantial cross-modality discrepancy in VT-ReID. CAP directly
optimizes the cross-modality overall ranking order within a batch.
Specifically, we introduce a query instance from one modality and
calculate CAP using galleries from another modality. CAP incorporates
two aspects 2(c): CAP with Visible queries (CAPV) and CAP with
Thermal queries (CAPT). These two aspects mutually reinforce each
other and jointly optimize the cross-modality overall ranking order.
To further enhance the effectiveness of CAP, we introduce two
techniques. Firstly, recognizing that the large cross-modality discrepancy hinders CAP performance, we propose Dynamic Modality Alignment (DMA) to mitigate this discrepancy. DMA involves constructing
a dynamic Cross-modality Affiliated Matrix (CAM) that assigns higher
weights to cross-modality sample pairs exhibiting smaller non-modality
variations (e.g., same view and pose). By encouraging the network
to focus on reducing the cross-modality discrepancy while ignoring
non-modality variations during alignment, DMA mitigates the influence
of non-modality variations and achieves superior modality alignment.
On the second aspect, some elaborate works have demonstrated the
effectiveness of local features [9,10] in VT-ReID. Motivated by these
findings, we extract Global and Local Features (GLF) jointly to enhance
feature discrimination. CAP and DMA are then applied individually
to these features, further improving the performance of the proposed
method.

Our contributions can be summarized as follows:
• We develop a differentiable Cross-modality Average Precision
(CAP) to directly optimize the cross-modality overall ranking orders, which explicitly handle the large cross-modality discrepancy
in the VT-ReID task.
• We propose a Dynamic Modality Alignment (DMA) to reduce
the cross-modality discrepancy by constructing a dynamic Crossmodality Affiliated Matrix (CAM), which can alleviate the influence of the non-modality variations and achieve better modality
alignment, facilitating the effectiveness of CAP.
• We apply our CAP and DMA to both Global and Local Features (GLF) individually. By incorporating them into a carefully
designed global–local structure, we achieve state-of-the-art performance on two datasets.
2. Related work
Object re-identification (ReID) [11] is a sub-task within the broader
field of image retrieval. It encompasses various domains, such as
building retrieval [12], drone-based geo-localization [13], vehicle reidentification [14,15], and person re-identification [16].
2.1. VT-ReID
Visible-thermal person re-identification (VT-ReID) was first introduced by [2], which aims to match the query person of one modality
from the gallery of another one. Since then, many methods have been
proposed for VT-ReID, which can be mainly divided into four groups.
(1) Feature extractor based methods aim to design a cross-modality
structure to extract modality-invariant and discriminative features. For
example,
Ye et al. [17] introduced a modality-aware collaborative ensemble learning method and middle-level sharable two-stream network
to handle modality discrepancies at both feature and classifier levels.
Some methods [18,19] focus on extracting the identity-related feature
2

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

by explicitly removing the irrelevant information. Fu et al. [20] used
a neural architecture search method to automatically search for the
best segmentation scheme, and determine which BN layer needs to be
segmented. To improve the feature discrimination, the global and local
features are applied to the cross-modality matching [10,21].
(2) Metric learning based methods aim to learn an embedding space
by explicitly reducing the distances between intra-identify samples of
two modalities. Hao et al. [4] map the features of two modalities into a
hypersphere manifold, and then constrain the intra-modality variations
and inter-modality discrepancy in this manifold. Ye et al. [5] introduce
a triplet loss with a bidirectional exponential angle to optimize the
angle discriminative features of two modalities samples. In addition,
Liu et al. [22] proposed dual-granularity triplet loss hierarchically
integrates sample-based and center-based triplet losses using simple
configurations like pooling and batch normalization.
(3) Distribution alignment based methods are designed to learn
modality-invariant features by decreasing the distribution discrepancy
of two modalities. Wu et al. [23] propose a modality alleviation
structure and a pattern alignment structure to align two modalities.
Zhao et al. [24] leverages color-irrelevant consistency learning to
extract color-agnostic features, and identity-aware modality adaptation
to align feature distributions at the identity level. In order to explore
nuances of information, Zhang et al. [25] introduce a method to embed
two modalities images into a 3D public space, and use a contrastive
association structure to learn contrastive features. Park et al. [26]
introduce a dense correspondence relationship between visible and
thermal modalities to match the corresponding pedestrian parts of the
two modalities.
(4) Image generation based methods attempt to bridge the modality
gap in the image-level by image translation techniques. Li et al. [27]
and Wei et al. [28] utilize a lightweight network to generate images
from one modality to another modality, and learn modality-invariant
embedding representation from these three modalities. In order to
reduce the influence of id-unrelated factors in features, some methods [29,30] use variational autoencoders and generative adversarial
networks to decompose features into two factors, id-related and idunrelated features, where id-related features with rich identify information are used for cross-modality retrieval. Liu et al. [31] attempt to
generate high-quality images to smooth the large inter-modality gap.

identity classification (𝐿ID ), the losses of Cross-modality Average Precision (𝐿CAP ), and the losses of Dynamic Modality Alignment (𝐿DMA )
with each type of feature. These combined losses are jointly used as the
objective function to optimize the network in an end-to-end manner.
Specifically, the 𝐿ID is mainly used to reduce intra-class variations, the
𝐿CAP can directly optimize the cross-modality overall ranking orders in
a batch (Fig. 4(a → c)), and the 𝐿DMA can alleviate the influence of
non-modality variations and effectively smooth the inter-modality gap
(Fig. 4(a → b)).
3.1. Cross-modality average precision
Existing metric learning methods will over-focus on optimizing lowranking positive samples at the expense of high-ranking ones due to
a lack of consideration of the overall ranking orders. Moreover, these
methods usually require to be well-designed for jointly handling crossmodality discrepancy and intra-class variations, which require huge
experimentation and empirical practice. Inspired by Smooth-AP [8], we
propose a Cross-modality Average Precision (CAP) to directly optimize
the cross-modality overall ranking orders. We next introduce CAP in
detail.
True average precision. We set 𝑠𝑖 as the cosine similarity between
the query and the sample 𝑖. The ranking of sample 𝑖 in any set 𝑆 can
be defined as:
∑
𝑅(𝑖, 𝑆) = 1 +
𝑢(𝑠𝑖 − 𝑠𝑗 ),
(1)
𝑗∈𝑆 ,𝑗≠𝑖

where 𝑢(𝑥) is a Heaviside step function, which will be set to 1 when
𝑥 > 0, otherwise set to 0.
Given a query 𝑓 𝑞 , and gallery set 𝐺𝑞 = {𝑓 𝑖 , 𝑖 = 1, … , 𝑁}. For
each query 𝑓 𝑞 , the gallery set can split into a positive set 𝑃 𝑞 and a
negative set 𝑁 𝑞 , which are formed by samples with the same ID and
with different IDS, respectively. The average precision of a query 𝑓 𝑞 is
defined as:
1 ∑ 𝑅 (𝑖, 𝑃 𝑞 )
𝐴𝑃 (𝑓 𝑞 ) = 𝑞
|𝑃 | 𝑖∈𝑃 𝑞 𝑅 (𝑖, 𝐺𝑞 )
∑
(2)
1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑢(𝑠𝑖𝑗 )
1 ∑
,
= 𝑞
∑
∑
|𝑃 | 𝑖∈𝑃 𝑞 1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑢(𝑠𝑖𝑗 ) + 𝑗∈𝑁 𝑞 𝑢(𝑠𝑖𝑗 )
where 𝑠𝑖𝑗 = 𝑠𝑖 − 𝑠𝑗 , |𝑃 𝑞 | is the instance number of 𝑃 𝑞 set. It is noticed
that the derivative of the Heaviside step function (𝛿(𝑥) = 𝑑 𝑢(𝑥)∕𝑑 𝑥) is
flat with zero except 𝑥 = 0 point (Fig. 5(a)), and thus cannot be used
to optimize the model.
Approximate differentiable average precision. To address this
issue, we use an approximate method to calculate the average precision
by replacing the Heaviside step function 𝑢(𝑥) with a sigmoid-derived
function 𝑔(𝑥), which is defined as:
1
(3)
𝑔(𝑥) =
−𝑥 ,
1+𝑒 𝜏

2.2. Optimizing average precision
Average precision is a standard metric for retrieval tasks. Recently,
directly optimizing average precision based methods [32–34] have
been proposed to address the challenge of non-differentiability average
precision in the retrieval community. Such as using an approximation
derived from distance quantization [35], and a histogram binning
approximation [7], relaxing indicator function with a sigmoid function [8]. Recently, Ramziet al. [36] used an upper bound to optimize the average precision. Li et al. [37] propose PNP to optimize
the negative instances before the positive ones. Distinguishing itself
from previous research efforts, our approach explicitly addresses the
formidable challenge posed by the substantial cross-modality discrepancy. We introduce a novel concept termed ‘‘Cross-modality Average
Precision’’, which allows us to directly optimize the global ranking
orders across different modalities, specifically involving queries and
galleries. This marks a pioneering endeavor in the realm of addressing
the VT-ReID problem by directly optimizing cross-modality average
precision.

where 𝜏 is a hyper-parameter that adjusts the sharpness. The sigmoid
derived function 𝑔(𝑥) and its derivative 𝑑 𝑔(𝑥)∕𝑑 𝑥 are shown in Fig. 5(b).
Therefore, the average precision in Eq. (2) can be approximated as:
∑
1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑖𝑗 )
1 ∑
𝐴𝑃 (𝑓 𝑞 ) ≈ 𝑞
.
(4)
∑
∑
|𝑃 | 𝑖∈𝑃 𝑞 1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑖𝑗 ) + 𝑗∈𝑁 𝑞 𝑔(𝑠𝑖𝑗 )
This approximate calculation can be directly optimized using
gradient-descent methods.
Cross-modality average precision. For VT-ReID, the main challenge is the coexistence of inter-modality discrepancy and intramodality variations, and the former is usually larger than the latter.
So the original Smooth-AP will prioritize optimizing the ranking of
intra-modality instances while neglecting to optimize the ranking of
inter-modality ones due to its inherent attributes. To handle the issue,
we calculate Cross-modality Average Precision (CAP) to handle the
main challenge of the cross-modality VT-ReID, in which queries and

3. Proposed method
In this section, we provide a detailed description of our proposed
framework. As illustrated in Fig. 3, we adopt a Global–Local Structure
(GLS) to extract both Global and Local Features (GLF) from visible and
thermal modality images, respectively. Then we calculate the losses of
3

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

Fig. 3. The overview of our proposed method. (a) Illustration of our framework. Two independent streams are utilized for visible modality and thermal modality in the shallow
layers, which include the first three layers of the ResNet-50 [38]. After the shallow layers, we design two pipelines for extracting global-based features and local-based features.
Each pipeline consists of the last two layers of the ResNet-50 [38], which are shared by both modalities. The parameters of the two pipelines are different. Given the inputs,
we first use a Global–Local Structure (GLS) to extract Global and Local Features (GLF). We then calculate the losses of identity classification (𝐿ID ), the losses of Cross-modality
Average Precision (𝐿CAP ), and the losses of Dynamic Modality Alignment (𝐿DMA ) with each type of feature. (b) Illustration of the CAP loss (bottom) and the calculation process
of DMA loss (top). Shapes indicate the identities and colors represent the modalities (blue for visible and yellow for thermal). (For interpretation of the references to color in this
figure legend, the reader is referred to the web version of this article.)

galleries are from different modalities. Given a training mini-batch, we
have 𝑛𝑣 visible modality samples and 𝑛𝑡 thermal modality samples. The
features obtained by the model are defined as F𝑣 and F𝑡 for visible
modality and thermal modality, respectively. We will introduce how to
obtain the representation of samples in Section 3.3. We perform CAP
by considering two aspects, CAP with Visible queries (CAPV), and CAP
with Thermal queries (CAPT).
At the CAPV aspect, we set 𝑓𝑣𝑞 as a query instance from visible
modality, and set 𝑃𝑡𝑞 and 𝑁𝑡𝑞 as its corresponding positive set and
negative set from thermal modality. The CAPV is calculated as:
∑
1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑖𝑗 )
1 ∑
𝑡
𝑞
𝐴𝑃 (𝑓𝑣 ) ≈ 𝑞
.
(5)
∑
∑
𝑖
𝑖
|𝑃 |
| 𝑡 | 𝑖∈𝑃𝑡𝑞 1 + 𝑗∈𝑃𝑡𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑗 ) + 𝑗∈𝑁𝑡𝑞 𝑔(𝑠𝑗 )
The loss of CAPV is calculated as:
𝑛𝑣
)
1 ∑(
𝐿𝐶 𝐴𝑃 𝑉 (F𝑣 , F𝑡 ) =
1 − 𝐴𝑃 (𝑓𝑣𝑞 ) .
𝑛𝑣 𝑞=1

Fig. 4. Illustrate the complementary results of CAP and DMA. The pairs with high
weights are connected with wider lines. Shapes indicate the identities and colors
represent the modalities (blue for visible and yellow for thermal). DMA can effectively
smooth the cross-modality gap (a → b). CAP can directly improve the cross-modality
overall ranking orders (a → c). These two aspects jointly optimize the model and gain
mutual benefit results (d). (For interpretation of the references to color in this figure
legend, the reader is referred to the web version of this article.)

(6)

Similarly, at the CAPT aspect, we set 𝑓𝑡𝑞 as a query instance from
thermal modality, and set 𝑃𝑣𝑞 and 𝑁𝑣𝑞 as its corresponding positive set
and negative set from visible modality. The CAPT is calculated as:
∑
1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑖𝑗 )
1 ∑
𝑣
.
(7)
𝐴𝑃 (𝑓𝑡𝑞 ) ≈ 𝑞
∑
𝑖) + ∑
𝑖
|𝑃𝑣 |
𝑞
𝑞
1
+
𝑔(𝑠
𝑞
| | 𝑖∈𝑃𝑣
𝑗∈𝑃𝑣 ,𝑗≠𝑖
𝑗∈𝑁𝑣 𝑔(𝑠𝑗 )
𝑗
The loss of CAPT is calculated as:
𝑛𝑡
)
1 ∑(
𝐿𝐶 𝐴𝑃 𝑇 (F𝑣 , F𝑡 ) =
1 − 𝐴𝑃 (𝑓𝑡𝑞 ) .
𝑛𝑡 𝑞=1
The CAP loss combines these two aspects of losses:
}
1{
𝐿𝐶 𝐴𝑃 𝑉 (F𝑣 , F𝑡 ) + 𝐿𝐶 𝐴𝑃 𝑇 (F𝑣 , F𝑡 )
2
3.2. Dynamic modality alignment
𝐿𝐶 𝐴𝑃 (F𝑣 , F𝑡 ) =

(8)

(9)

In the VT-ReID task, the large cross-modality discrepancy has a negative effect on the CAP. To cope with this issue, we propose a Dynamic
Modality Alignment (DMA) to smooth the large inter-modality gap by
constructing a dynamic Cross-modality Affiliated Matrix (CAM), which
can mitigate the influence of non-modality variations and achieve
better modality alignment. The details of the DMA loss calculation are
shown in Fig. 3 (b-top).
𝑛 ×𝑛
We set 𝑫 ∈ R+𝑣 𝑡 as the distance matrix between visible modality
and thermal modality, which elements are calculated as:
‖
‖2
𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ) = ‖𝑓𝑣,𝑖 − 𝑓𝑡,𝑗 ‖ ,
(10)
‖
‖2
where 𝑓𝑣,𝑖 ∈ F𝑣 and 𝑓𝑡,𝑗 ∈ F𝑡 , ‖.‖ represents the L2-norm. The distance includes cross-modality discrepancy and non-modality variations
(e.g. different identities, views, and poses).
In order to reduce the influence of non-modality variations during
modality alignment, we utilize a dynamic Cross-modality Affiliated
Matrix (CAM) to calculate the distribution distance between visible

Fig. 5. (a) The Heaviside step function 𝑢(𝑥) (top) and its derivative 𝑑 𝑢(𝑥)∕𝑑 𝑥 (bottom).
The derivative of the Heaviside step function is flat with zero except at 𝑥 = 0, which
prevents effective model optimization. (b) The sigmoid-derived function 𝑔(𝑥; 𝜏) (top)
and its derivative 𝑑 𝑔(𝑥)∕𝑑 𝑥 (bottom), which can be directly optimized using gradientdescent methods. 𝜏 = 0.1.

4

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

and DMA on them.
Global Feature. We directly apply the generalized-mean pooling
(GeM) [41] and the batch normalization neck (BNNeck) [42] on the
last residual convolution layer of global stream 𝑓 𝑔,𝑟𝑒𝑠 and obtain the
global feature 𝑓 𝑔 ∈ R1,𝐶 .
Local Features. We first evenly divide the last residual convolution
layer of local stream 𝑓 𝑙,𝑟𝑒𝑠 into 𝐾 parts along the vertical direction and
perform the GeM and BNNeck on them, obtaining 𝐾 local features,
{𝑓 𝑙,1 , … , 𝑓 𝑙,𝐾 }.
Losses for CAP. Given the global feature and local features, the
losses of CAP can be calculated as:
𝐿𝐶 𝐴𝑃 = 𝐿𝑔𝐶 𝐴𝑃 + 𝐿𝑙𝐶 𝐴𝑃
(
)
∑
(
) 𝐾
= 𝐿𝐶 𝐴𝑃 F𝑔𝑣 , F𝑔𝑡 +
𝐿𝐶 𝐴𝑃 F𝑙𝑣,𝑘 , F𝑙𝑡,𝑘 ,

(13)

𝑘=1

where 𝐾 is the part number, 𝐿𝑔CAP is the global-based loss of CAP and
𝐿𝑙CAP is the local-based loss of CAP.
Losses for DMA. Similarly, the losses of DMA can be calculated as:
𝐿𝐷𝑀 𝐴 = 𝐿𝑔𝐷𝑀 𝐴 + 𝐿𝑙𝐷𝑀 𝐴

Fig. 6. Illustration of the negative correlation between the weights of cross-modality
pairs, 𝑾 𝑖, 𝑗, and their distance, 𝑫(𝑓 𝑣, 𝑖, 𝑓𝑡,𝑗 ). Larger weights are assigned to crossmodality pairs that are closer to each other. Pairs with higher weights are represented
by thicker lines.

𝐾
(
)
(
) ∑
= 𝐿𝐷𝑀 𝐴 F𝑔𝑣 , F𝑔𝑡 +
𝐿𝐷𝑀 𝐴 F𝑙𝑣,𝑘 , F𝑙𝑡,𝑘 ,

(14)

𝑘=1

Losses for Identity Classification. We add classification heads (1
for global feature, 𝐾 for local part features), which are fully connected
layers with an output length of #identities, after these two types of
features. We then calculated identity classification losses according to
the outputs of these classification heads, formulated as:
𝐿ID = 𝐿𝑔ID + 𝐿𝑙ID

𝑛 ×𝑛

and thermal modality. We set 𝑾 ∈ R+𝑣 𝑡 as CAM. Intuitively, a
larger distance between samples from the visible and thermal modalities indicates greater non-modality variations, such as differences in
viewpoint and pose. At this time, a smaller weight 𝑾 𝑖,𝑗 should be assigned. This enables the model to prioritize cross-modality sample pairs
with larger weights, thus focusing on handling the cross-modality discrepancy, while simultaneously neglecting those with smaller weights
(cross-modality pairs characterized by more substantial non-modality
variation). Therefore, the elements of 𝑾 matrix can be calculated as:
exp(1 − 𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ))
𝑾 𝑖,𝑗 = ∑
.
(11)
𝑗∈𝑁𝑡 exp(1 − 𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ))

= Cr ossEnt r opy (H𝑔 , Y) +

𝐾
∑

(
)
Cr ossEnt r opy H𝑙,𝑘 , Y ,

(15)

𝑘=1

where H indicates the predictions of the corresponding classification
heads and Y is the identity labels.
3.4. Overall
Training. The proposed framework can be trained in an end-toend manner by minimizing the loss of identity classification, the loss
of DMA, and the loss of CAP, which are calculated as follows:

We calculate the distribution distance between visible and thermal
modality by the weighted summation of cross-modality pairs distance,
and treat it as the DMA loss, which is calculated as:
𝑛𝑣 𝑛𝑡
1 ∑∑
𝐿𝐷𝑀 𝐴 (F𝑣 , F𝑡 ) =
𝑾 𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ).
(12)
𝑛𝑣 + 𝑛𝑡 𝑖=1 𝑗=1 𝑖,𝑗

ar g min 𝐿𝐼 𝐷 + 𝐿𝐶 𝐴𝑃 + 𝛼 𝐿𝐷𝑀 𝐴 ,
𝜃

(16)

where 𝛼 is the hyper-parameters.
Testing. In the testing phase, different from other global and local
combination methods (e.g. [10,21]), we obtain the final Global–Local
Feature (GLF) by concatenating the global feature and local features
with a ratio:
[
]
(17)
𝑓𝑡𝑒𝑠𝑡 = 𝛽(𝑓 𝑙,1 ∣ ⋯ ∣ 𝑓 𝑙,𝐾 )|(1 − 𝛽)𝑓 𝑔 ,

During optimization, due to the negative correlation between crossmodality pairs distance 𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ) and their weights 𝑾 𝑖,𝑗 , the larger
weight will be assigned for cross-modality pairs (𝑓𝑣,𝑖 and 𝑓𝑡,𝑗 ) when they
are close to each other, otherwise a small weight will be assigned for
them (Fig. 6). In such a scenario, when aligning two modalities with the
DMA loss, the model would concentrate on reducing the discrepancy
between cross-modality pairs that have less non-modality variations
(i.e., closer pair). Nevertheless, the pairs with larger non-modality
variations (i.e., distant pairs) would receive less attention from model
optimization, since they are assigned with low weights. Therefore, the
DMA can efficiently reduce the influence of non-modality variations
and achieve better modality alignment, which can further enhance the
CAP result.

where [⋅ ∣ ⋅] indicates the concatenation operation, 𝛽 is the hyperparameter that controls the importance of global and local features. The
dimensions of the global features, local features, and testing features
are 2048, 256, and 3584, respectively.
4. Experiments
4.1. Experimental settings

3.3. Losses on global and local feature

Datasets. We evaluate the proposed methods on two publicly available VT-ReID datasets, SYSU-MM01 [2] and RegDB [43].
SYSU-MM01 [2] contains a total of 303,420 images: including
287,628 RGB images and 15,792 infrared images. This dataset is captured by four RGB cameras and two thermal cameras. The training
set contains 22,258 RGB images and 11,909 infrared images of 395
identities. The testing set consists of 3803 query (infrared) images and
301 galleries (RGB) images of 96 identities. For evaluation, we use two

The PCB method [39] in ReID has demonstrated that local information can improve feature discrimination. Furthermore, some elaborate
works also evaluate the effectiveness of local features [9,40], and the
effectiveness of global and local combined features [10,21] in VT-ReID.
Inspired by this, we jointly extract Global and Local Features (GLF) to
improve feature discrimination and then implement the proposed CAP
5

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

testing modes: all-search mode and indoor-search mode, where each
mode has two settings (single-shot settings and multi-shot settings).
Please refer to [2] for details of the dataset and the evaluation setting.
RegDB [43] is collected by one RGB camera and one infrared
camera. It contains 4120 RGB images and 4120 infrared images of
412 identities. For each identity, there are 10 RGB images and 10
infrared images, respectively. We divide RegDB into two halves, one for
training and another for testing. For evaluation, we apply two testing
settings: Thermal (query) to Visible (gallery) setting and Visible (query)
to Thermal (gallery) setting. The difference between these two settings
lies in the modalities of the query and gallery samples.
Evaluation metrics. The Cumulative Matching Characteristics
(CMC) and mean Average Precision (mAP) are used to evaluate the
retrieval performance. For CMC, we report the rank-1 (R1), rank-10
(R10), and rank-20 (R20) accuracies.
Implementation Details. We implement our method with PyTorch
and train our model with an NVIDIA 3090 GPU. We construct the
baseline model based on [44]. The baseline includes two network
streams, which are used to extract features for the images of visible
modality and thermal modality, respectively. Only the identification
loss is used to optimize the baseline model. For the baseline model
and our model, we use the same training settings as described below.
For data augmentation, we use Random Erasing (RE) [45], and random
cropping during training. In terms of mini-batch, the identity number
𝐶 is set to 6, and each identity contains 8 RGB images and 8 infrared
images (i.e. 𝑛𝑣 = 𝑛𝑡 = 8) for SYSU-MM01, and 4 RGB images and 4
infrared images (i.e. 𝑛𝑣 = 𝑛𝑡 = 4) for RegDB. The input images are
resized to 384 × 192 × 3 for both RGB and infrared images. Following
other methods, we adopt the ResNet-50 [38] as our backbone of the
feature extractor. We use the SGD optimizer for training and set the
initial learning rate to 0.1 with a warm-up strategy [42]. We divide
the learning rate by 10 after 20 and 50 epochs. The overall training
epoch is 100 for both SYSU-MM01 and RegDB datasets. For the hyperparameters of our model, the features are divided into 6 parts, i.e. 𝐾 =
6. we set weight 𝜏 (Eq. (3)) to 0.01, 𝛼 (Eq. (16)) to 0.1 for both
SYSU-MM01 and RegDB datasets. In the testing phase, we set weight 𝛽
(Eq. (17)) to 0.6 for both datasets.

4.3. Evaluation
In this section, we evaluate the effectiveness of each component of
our proposed, i.e., GLS, CAP, and DMA. Results are reported on the
single-shot setting under the all-search model for SYSU-MM01 and the
‘‘Visible to Thermal’’ setting for RegDB.
Effectiveness of GLS. To demonstrate the superiority of the Global–
Local Structure (GLS), we conduct experiments on the baseline mode
with the global features, local features, and Global and Local Features
(GLF) obtained by GLS. The results are shown in Table 3 (#0 vs #4
vs #8). We can find that using local features can achieve higher performance than using global features on both datasets. Moreover, using
GLF can further improve the performance, indicating the effectiveness
of our GLS.
Effectiveness of CAP and DMA. To evaluate the effectiveness and
demonstrate the generalization of our proposed Cross-modality Average
Precision (CAP) and Dynamic Modality Alignment (DMA), we conduct
experiments of ablation study under the baseline with global feature
(Table 3 (#0–#3)), local feature (Table 3 (#4–#7)), and our global–
local feature (Table 3 (#8–#11)) on both datasets. We can conclude
that: (1) Considering each component (i.e., CAP, DMA) independently
can improve the performance of all baselines on both datasets. (2) Combining these two components into each baseline can further enhance
the results. (3) We achieve the best results when applying our CAP and
DMA on Baseline with GLF (Baseline w/GLF). These observations verify
the generalization ability of CAP and DMA, and demonstrate that the
proposed DMA and GLS can facilitate the effectiveness of our CAP.
Effectiveness of two components in CAP. To evaluate the effectiveness of two components in CAP, i.e., CAP with Visible queries
(CAPV), and CAP with Thermal queries (CAPT), we conduct experiments of ablation study under the baseline with local features. The
results are shown in Table 4, we can find that CAPV and CAPT can
improve the performance over the baseline. Moreover, combining the
two of them can further improve the results on both datasets. These
observations verify the effectiveness of our CAPV and CAPT, and the
complementary and mutual benefits of these two aspects.

4.2. Comparison with state of the art
To demonstrate the superiority of our method, we first compare the
proposed method with state-of-the-art approaches on SYSU-MM01 and
RegDB datasets.
Results on SYSU-MM01. We first compare our method with stateof-the-art methods on the SYSU-MM001 dataset. The results are reported in Table 1. We can find that our method outperforms the
state-of-the-art methods by a large margin in all evaluation settings.
Specifically, for the all-search mode, we achieve rank-1 = 75.05% and
mAP = 71.72% for the single-shot setting, and, rank-1 = 78.94% and
mAP = 65.67% for the multi-shot setting. For indoor-search mode,
we achieve rank-1 = 83.92% and mAP = 85.44% for the single-shot
setting, and, rank-1 = 87.77% and mAP = 77.88% for the multishot setting. Compared to the best competitor (MAUM [67]), we show
significant improvements. For example, our method is higher than it
by 3.37% and 6.96% in rank-1 accuracy for single-shot setting under
all-search and indoor-search mode, respectively.
Results on RegDB. The comparisons on the RegDB dataset are
reported in Table 2. It is clear that our method outperforms the state-ofthe-art methods by a large margin in both evaluation settings. Specifically, we obtain rank-1 = 93.93% and mAP = 89.17% for the ‘‘Visible
to Thermal’’ setting, and, rank-1 = 92.09% and mAP = 87.19% for
‘‘Thermal to Visible’’ setting. Compared to the current best competitor
(GLMC [10]), our method is higher than it by 2.09% in rank-1 accuracy
and 7.75% in mAP accuracy on the ‘‘Visible to Thermal’’ setting, and,
by 0.97% in rank-1 accuracy and 8.16% in mAP accuracy on the
‘‘Thermal to Visible’’ setting, respectively.
In conclusion, our approach produces new state-of-the-art results on
both datasets.

Comparison of distribution alignment methods. In Table 5, we
compare DMA with its variant and KL divergence. For the DMA variant, we directly use Average Weights (AW) to replace the weights
obtained by DMA. The results show that DMA largely outperforms
the other two methods, further demonstrating the superiority of the
proposed DMA. It is because the KL and AW cannot consider the
influence of non-modality variation, while our DMA dynamic assigns
a large weight for cross-modality sample pairs that have a smaller
non-modality variation, which encourages the network to focus on
reducing cross-modality discrepancy while ignoring the non-modality
variations during cross-modality alignment, resulting in better modality
alignment.
Comparison of CAP with other methods. In Table 6, we compare CAP with other popular metric learning methods (i.e., center
loss [72], triplet loss [6], circle loss [73], and instance loss [74]), and
AP approximating methods (i.e., Fast AP [35], Blackbox AP [34], and
Smooth-AP [8]). We can find that metric learning methods and AP
approximating methods can improve the results over the baseline. In
addition, our CAP achieves greater improvements than other methods
on both datasets. Compared to these metric learning methods [6,72–
74], our CAP can directly optimize the overall ranking orders and
reduce the risk of getting stuck in local optimization. Compared to
the original Smooth-AP [8] used in single modality retrieval, our CAP
explicitly handles the challenge of the large cross-modality discrepancy
ignored by Smooth-AP.
6

Pattern Recognition 164 (2025) 111489

Y. Ling et al.
Table 1
Comparison with the state-of-the-art methods on the SYSU-MM01 dataset.
Methods

All-search

Indoor-search

Single-shot

Multi-shot

Single-shot

Multi-shot

R1

R10

R20

mAP

R1

R10

R20

mAP

R1

R10

R20

mAP

R1

R10

R20

mAP

Zero-padding
[2]
HCML [46]
cmGAN [47]
BDTR [48]
D-HSME [4]
D2RL [49]
SDL [50]
DGD+MSR
[51]
AlignGAN
[52]
Hi-CMD [29]
PIG [30]
DFE [53]
Xmodal [27]
DDAG [54]
DG-VAE [55]
SIM [56]
cm-SSFT [57]
AGW [44]
CMAlign [26]
NFS [18]
MSO [58]
VCD-VML
[19]
MID [59]
HCT [9]
CM-NAS [20]
GLMC [10]
MCLNet [60]
FMCNet [61]
TTM [62]
SMCL [28]
PMT [63]
MRCN [64]
DART [65]
CAJL [66]
MPANet [23]
MAUM [67]

14.8

54.12

71.33

15.95

19.13

61.4

78.41

10.89

20.58

68.38

85.79

26.92

24.43

75.86

91.32

18.64

14.32
26.97
17.01
20.68
28.9
28.12
37.35

53.16
67.51
55.43
62.74
70.6
70.23
83.4

69.17
80.56
71.96
77.95
82.4
83.67
93.44

16.16
27.8
19.66
23.12
29.2
29.01
38.11

/
31.49
/
/
/
/
43.86

/
72.74
/
/
/
/
86.94

/
85.01
/
/
/
/
95.68

/
22.27
/
/
/
/
30.48

/
31.63
/
/
/
32.56
39.64

/
77.23
/
/
/
80.45
89.29

/
89.18
/
/
/
90.67
97.66

/
42.19
/
/
/
39.56
50.88

/
37
/
/
/
/
46.56

/
80.94
/
/
/
/
93.57

/
92.11
/
/
/
/
98.8

/
32.76
/
/
/
/
40.08

42.4

85

93.7

40.7

51.5

89.4

95.7

33.9

45.9

87.6

94.4

54.3

57.1

92.7

97.4

45.3

34.94
38.1
48.71
49.92
54.75
59.49
60.88
61.6
47.50
55.41
56.91
58.70
60.02

77.58
80.7
88.86
89.79
90.39
/
/
89.2
84.39
/
91.34
92.06
94.18

/
89.9
95.27
95.96
95.81
93.77
/
93.9
92.14
/
96.52
/
98.14

35.94
36.9
48.59
50.73
53.02
58.46
56.93
63.2
47.65
54.14
55.45
56.42
58.8

/
45.1
54.63
/
/
/
/
63.4
/
/
63.51
65.85
/

/
85.7
91.62
/
/
/
/
91.2
/
/
94.42
94.37
/

/
93.8
96.83
/
/
/
/
95.7
/
/
97.81
/
/

/
29.5
42.14
/
/
/
/
62.0
/
/
48.56
49.56
/

/
43.8
52.25
/
61.02
/
/
70.5
54.17
58.46
62.79
63.09
66.05

/
86.2
89.86
/
94.06
/
/
94.9
91.14
/
96.53
96.61
96.59

/
94.2
95.85
/
98.41
/
/
97.7
95.98
/
99.07
/
99.38

/
52.9
59.68
/
67.98
/
/
72.6
62.97
66.33
69.79
70.31
72.98

/
52.7
59.62
/
/
/
/
73.0
/
/
70.03
72.06
/

/
91.1
94.45
/
/
/
/
96.3
/
/
97.7
97.77
/

/
96.4
98.07
/
/
/
/
99.1
/
/
99.51
/
/

/
42.7
50.6
/
/
/
/
72.4
/
/
61.45
61.69
/

60.27
61.68
61.99
64.37
65.4
66.34
67.25
67.39
67.53
68.9
68.72
69.88
70.58
71.68

92.90
93.10
92.87
93.90
93.33
/
95.38
92.87
95.36
95.2
96.39
95.71
96.21
/

/
97.17
97.25
97.53
97.14
/
98.46
96.76
98.64
98.4
98.96
98.46
98.8
/

59.40
57.51
60.02
63.43
61.98
62.51
64.29
61.78
64.98
65.5
66.29
66.89
68.24
68.79

/
/
68.68
66.66
/
/
72.95
72.15
/
/
/
/
75.58
/

/
/
94.92
95.87
/
/
96.94
90.66
/
/
/
/
97.91
/

/
/
98.36
98.63
/
/
99.27
94.32
/
/
/
/
99.43
/

/
/
53.45
54.46
/
/
57.62
54.93
/
/
/
/
62.91
/

64.86
63.41
67.01
67.35
72.56
68.15
69.58
68.84
71.66
76.0
72.52
76.26
76.74
76.97

96.12
91.69
97.02
98.10
96.98
/
96.66
96.55
96.73
98.3
97.84
97.88
98.21
/

/
95.28
99.32
99.77
99.2
/
99.03
98.77
99.25
99.7
99.46
99.49
99.57
/

70.12
68.17
72.95
74.02
76.58
74.09
74.37
75.56
76.52
79.8
78.17
80.37
80.95
81.94

/
/
76.48
77.45
/
/
80.39
79.57
/
/
/
/
84.22
/

/
/
98.68
97.69
/
/
98.80
95.33
/
/
/
/
99.66
/

/
/
99.91
99.59
/
/
99.83
98.00
/
/
/
/
99.96
/

/
/
65.11
67.18
/
/
68.60
66.57
/
/
/
/
75.11
/

Ours

75.05

96.66

98.90

71.72

78.94

97.92

99.45

65.67

83.92

98.83

99.68

85.44

87.77

99.73

99.98

77.88

Fig. 7. (a) Evaluation of different weights 𝛼 of DMA loss on the RegDB dataset. (b) Evaluation of different ratios 𝛽 of local features and global features on the SYSU-MM01
dataset. (c) Evaluation of the part number K on the SYSU-MM01 dataset.

features on the SYSU-MM01 dataset under the single-shot setting of the
all-search mode. The results are shown in Fig. 7(b). 𝛽 = 0 means only
global features are used for testing, and 𝛽 = 1 represents that only local
features are applied for testing. We gain the best performance when 𝛽
is set to 0.6, indicating the good performance of our proposed GLS, and
demonstrating the effectiveness of the combination of global and local
features. This is because the global features carry the overall information of the pedestrian, while the local features pay more attention to the
fine-grained information of the pedestrian. Therefore, the combination
of global and local features can enhance the discrimination of the
feature.
Evaluation of the part number K. we evaluate the impact of
part number K. The results are shown in Fig. 7(c). We find that the
performance improves as K increases, achieving the best results when

4.4. Hyperparameter and complexity analysis
Evaluation of weight 𝛼. We first evaluate the performance of different weights 𝛼 (Eq. (16)) on the RegDB dataset under the global feature.
The results are shown in Fig. 7(a). We gain the best performance when
𝛼 is set to 0.1. It demonstrates that the loss of the proposed DMA can facilitate the advantage of CAP. However, the DMA has a negative effect
on the CAP when 𝛼 increases to near 0.9. This is because the process of
modality alignment inevitably reduces feature discrimination. We can
balance the weight of CAP and DMA to achieve the best performance,
demonstrating the mutual benefit and complementary of the CAP and
DMA.
Evaluation of weight 𝛽. We then evaluate the performance of
different combination ratios 𝛽 (Eq. (17)) of global features and local
7

Pattern Recognition 164 (2025) 111489

Y. Ling et al.
Table 2
Comparison with the state-of-the-art methods on the RegDB dataset.
Settings

Visible to Thermal

Thermal to Visible

Method

Venue

R1

R10

R20

mAP

R1

R10

R20

mAP

Zero-Padding [2]
HCML [46]
BDTR [48]
DGD+MSR [51]
D2RL [49]
D-HSME [4]
AlignGAN [52]
DFE [53]
Hi-CMD [29]
PIG [30]
MANN [68]
Xmodal [27]
DDAG [54]
DG-VAE [55]
cm-SSFT [57]
SIM [56]
AGW [44]
CMAlign [26]
NFS [18]
VCD-VML [19]
MSO [58]
MCLNet [60]
MPANet [23]
SMCL [28]
CM-NAS [20]
CAJL [66]
DART [65]
MID [59]
MAUM [67]
FMCNet [61]
TTM [62]
G2DA [69]
MTMFE [70]
PMT [63]
HCT [9]
DEEN [71]
MRCN [64]
GLMC [10]

ICCV17
AAAI18
IJCAI18
TIP19
CVPR19
AAAI19
ICCV19
ACMMM19
CVPR20
AAAI20
PR20
AAAI20
ECCV20
ACMMM20
CVPR20
IJCAI20
TPAMI21
ICCV21
CVPR21
CVPR21
ACMMM21
ICCV21
CVPR21
ICCV21
ICCV21
ICCV21
CVPR22
AAAI22
CVPR22
CVPR22
PR22
PR23
PR23
AAAI23
TMM20
CVPR23
AAAI23
TNNLS21

17.75
24.44
33.47
48.43
/
50.85
57.9
70.13
/
48.50
48.67
62.21
69.34
72.97
73.3
75.29
70.05
74.17
80.54
73.2
73.6
80.3
83.7
83.93
84.54
85.03
83.60
87.45
87.87
89.84
89.91
71.72
76.10
84.83
91.05
91.1
91.4
91.84

34.21
47.53
58.42
70.32
/
73.36
/
86.32
/
/
71.55
83.13
86.19
/
/
/
86.21
/
91.96
/
88.6
92.7
/
/
95.18
95.49
/
95.73
/
/
96.57
87.13
88.86
/
97.16
97.8
98.0
97.86

44.35
56.78
67.52
79.95
/
81.66
/
91.96
/
/
80.06
91.72
91.49
86.89
/
/
91.55
/
95.07
/
/
96.03
/
/
97.85
97.54
/
/
/
/
98.33
91.92
92.41
/
98.57
98.9
99.0
98.98

18.9
20.8
31.83
48.67
/
47
53.6
69.14
/
49.3
41.11
60.18
63.46
71.78
72.9
74.47
66.37
67.64
72.1
71.6
66.9
73.07
80.9
79.83
80.32
79.14
75.67
84.85
85.09
84.43
85.64
65.90
74.39
76.55
83.28
85.1
84.6
81.42

/
21.70
32.92
/
43.4
50.15
56.3
67.99
70.93
48.1
38.68
/
68.06
/
71.0
78.30
70.49
72.43
77.95
71.8
74.6
75.93
82.8
83.05
82.57
84.75
81.97
84.29
86.95
88.38
88.34
69.50
72.18
84.16
89.30
89.5
88.3
91.12

/
45.02
58.46
/
66.1
72.40
/
85.56
86.39
/
60.82
/
85.15
/
/
/
87.12
/
90.45
/
88.7
90.93
/
/
94.51
95.33
/
93.44
/
/
96.16
84.87
87.06
/
96.41
96.8
96.7
97.86

/
55.58
68.43
/
76.3
81.07
/
91.41
/
/
69.10
/
90.31
/
/
/
91.84
/
93.62
/
/
94.59
/
/
97.37
97.51
/
/
/
/
97.98
89.85
92.38
/
98.16
98.4
98.5
98.69

/
22.24
31.96
/
44.1
46.16
53.4
66.70
66.04
48.90
32.61
/
61.80
/
71.7
75.24
65.90
65.46
69.79
70.1
67.5
69.49
80.7
78.57
78.31
77.82
73.78
81.41
84.34
83.86
84.06
63.88
71.04
75.13
81.46
83.4
81.9
81.06

Ours

–

93.93

98.83

99.61

89.17

92.09

97.57

98.53

87.19

Table 3
Ablation study of the proposed components. DMA: Dynamic Modality Alignment, CAP:
Cross-modality Average Precision, Global: global-based feature, Local: Local-based
features, GLF: Global and Local Feature (GLF).
#

Method

SYSU-MM01

RegDB

R1

mAP

R1

mAP

0
1
2
3

Baseline w/ Global
+DMA
+CAP
+DMA+CAP

57.53
65.03
64.08
67.08

54.71
60.97
60.42
64.36

65.99
82.45
81.67
86.72

58.27
73.09
73.56
79.64

4
5
6
7

Baseline w/ Local
+DMA
+CAP
+DMA+CAP

60.02
64.69
70.37
72.73

53.92
57.40
65.55
68.16

75.26
88.08
88.17
91.86

69.03
82.05
82.01
85.78

8
9
10
11

Baseline w/ GLF
+DMA
+CAP
+DMA+CAP

65.26
72.68
72.63
75.05

60.41
66.41
68.57
71.72

80.21
90.75
91.38
93.93

71.75
84.76
86.27
89.17

K = 6. However, further increasing K leads to performance degradation,
likely due to the inclusion of meaningless regions.
Computational complexity analysis. we compare the computational complexity of our model with the competitive method DEEN
[71]. The results are shown in Table 7. Due to the incorporation of
Global–Local Structure (GLS), the number of parameters in our model
is slightly higher than DEEN [71]. However, our model demonstrates a
competitive advantage in terms of FLOPs, indicating its computational
efficiency.

Fig. 8. Illustration of feature distribution on RegDB and SYSU-MM01 datasets. For
better visualization, we randomly show the distribution of 10 identities with 10 samples
for each modality from the testing set. Colors represent the identities. Circle-dot and
cross-mark denote the visible and thermal modalities, respectively.

8


codex
前四份文本已经抽出来了。`CoRL` 和 `CAP` 的前 8 页覆盖了方法与部分实验，文本抽取很长有截断，我会再针对方法小节精确抽取，避免只凭摘要判断。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
CVAF: A CLIP-Based View-Consistent Alignment Framework
for Aerial-Ground Person Re-Identification
DONGXU MAO, SHANGZHI TENG, and XUEQIANG LYU, School of Computer Science, Beijing
Information Science and Technology University, Beijing, China
With the increasing adoption of UAV platforms in areas such as public safety and smart cities, AerialGround Person Re-Identification (AGPReID) has emerged as a crucial yet highly challenging task, garnering
growing interest from the research community. While existing approaches have leveraged identity attributes
and viewpoint disentanglement strategies to improve cross-view matching, their heavy reliance on prior
knowledge often compromises model generalization. Furthermore, some methods that explicitly separate
viewpoints may unintentionally discard identity-related, view-invariant features, leading to incomplete
identity representations. To address these limitations, we propose a CLIP-based View-Consistent Alignment
Framework (CVAF) with two training stages. In the first stage, learnable text tokens are employed to represent
identity-aware textual descriptions. To promote consistent alignment across varying viewpoints, we introduce
a Text Consistency Loss (TCL) that regularizes the stability of text-token interactions with multi-view images.
In the second stage, we present a Semantic Filtering Module (SFM) that jointly modulates image patch tokens
along spatial and channel dimensions. A text-guided cross-attention mechanism generates spatial attention
maps to explicitly emphasize identity-relevant regions, while semantic matching between textual features
and visual tokens enables adaptive reweighting of image representations, effectively suppressing background
clutter and view-specific noise. Extensive experiments on multiple AGPReID datasets demonstrate that our
CVAF outperforms the state-of-the-art methods.
CCS Concepts: • Information systems → Information retrieval; • Computing methodologies → Image
representations;
Additional Key Words and Phrases: Vision-language Learning, Aerial-Ground View, Person Re-Identification,
Image Retrieval
ACM Reference format:
Dongxu Mao, Shangzhi Teng, and Xueqiang Lyu. 2026. CVAF: A CLIP-Based View-Consistent Alignment
Framework for Aerial-Ground Person Re-Identification. ACM Trans. Multimedia Comput. Commun. Appl. 22, 3,
Article 85 (February 2026), 19 pages.
https://doi.org/10.1145/3785482

This work is supported by the National Natural Science Foundation of China (Grants Nos. 62202061 and 62171043), the
Beijing Natural Science Foundation (Grant Nos. 4232025 and 4254096), and the Research Program of Beijing Municipal
Education Commission (Grant No. KM202311232002).
Authors’ Contact Information: Dongxu Mao, School of Computer Science, Beijing Information Science and Technology
University, Beijing, China; e-mail: 2023020673@bistu.edu.cn; Shangzhi Teng (corresponding author), School of Computer
Science, Beijing Information Science and Technology University, Beijing, China; e-mail: tengshangzhi@bistu.edu.cn;
Xueqiang Lyu, School of Computer Science, Beijing Information Science and Technology University, Beijing, China; e-mail:
lxq@bistu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2026/2-ART85
https://doi.org/10.1145/3785482
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

85:2
1

D. Mao et al.

Introduction

Person Re-Identification (ReID) is a fundamental task in computer vision that aims to identify
and match pedestrian instances across images captured by multiple non-overlapping cameras.
Traditional ReID methods primarily focus on image sourced from ground-only cameras. With
the proliferation of Unmanned Aerial Vehicles (UAVs) [17, 20, 51], their high mobility and
top-down viewpoints offer valuable complements to ground cameras by enhancing coverage,
reducing occlusion, and enabling more flexible deployment. The integration of aerial and ground
perspectives thus facilitates the construction of more comprehensive and adaptive intelligent
surveillance systems. However, the substantial appearance variations and scale discrepancies
introduced by heterogeneous camera views in aerial-ground networks also pose new challenges
for robust person ReID.
Figure 1 presents a visual comparison between pedestrian images captured by ground-only
cameras and those captured under aerial-ground cameras. Ground-only cameras typically provide
frontal or side views of pedestrians, where clothing and appearance features are clearly visible. In
contrast, aerial cameras—due to their elevated viewpoints and top-down perspectives—primarily
observe the head and upper body regions. As a result, pedestrians appear smaller, with blurred
visual details, distorted poses, and inconsistent aspect ratios. Despite significant progress in person
ReID, existing methods [7, 8, 13, 30, 32, 34, 37] often fall short in aerial-ground scenarios due to their
reliance on datasets captured exclusively under ground-only camera settings. These approaches are
typically optimized for consistent viewpoints and limited intra-class variation, yet they encounter
challenges when applied to aerial-ground scenarios. In Aerial-Ground Person Re-Identification
(AGPReID) task, the same identity frequently appears across drastically different views, resulting
in substantial intra-class discrepancies and making cross-view matching particularly challenging.
To address this issue, AG-ReID [18] leverages identity attributes as auxiliary, view-invariant feature
to bridge the appearance gap. More recent efforts such as VDT [40] and ViT-based disentanglement
frameworks attempt to explicitly decouple viewpoint and identity information, enabling more
robust feature learning across heterogeneous camera views.
While both attribute-based and view-disentanglement approaches have shown promise in addressing the severe viewpoint discrepancies of AGPReID, they each exhibit inherent limitations.
Attribute-based methods depend on predefined or detector-generated soft-biometric cues such as
color, clothing type, or accessories. However, these cues become unreliable under aerial viewpoints
due to low resolution, missing body parts, and reduced visual detail. Such coarse and incomplete attributes restrict the model from autonomously discovering fine-grained, view-invariant
patterns—such as stable body proportions, global silhouette geometry, or consistent structural
layouts—that remain shared across aerial and ground views but fall outside the scope of manually defined labels. Conversely, view-disentanglement methods introduce explicit mechanisms
(e.g., view tokens, subtractive separation, orthogonality constraints) to isolate view-related factors from identity features. Yet, in aerial-ground scenarios, identity cues and viewpoint cues are
intrinsically entangled. Many cross-view shared cues—such as approximate body shape, global
contour, and structural transitions—are partially view-dependent and thus cannot be perfectly
separated. Over-aggressive disentanglement may inadvertently discard these shared cues, weakening discriminability and generalization across views. These limitations motivate the need for an
approach that preserves such view-consistent structural information while enabling the model to
learn robust, view-invariant representations without relying on coarse attributes or rigid factor
decomposition.
To address the aforementioned limitations, we leverage the strong cross-modal alignment capabilities of CLIP as the foundation of our method. CLIP is pretrained on a large-scale corpus of
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

CVAF for Aerial-Ground Person Re-Identification

85:3

Fig. 1. (a) AGPReID vs. (b) traditional person ReID. AGPReID integrates views from aerial and ground
perspectives, introducing significant intra-class variations due to drastic viewpoint changes and posing
greater challenges for robust feature learning. In contrast, traditional person ReID focuses on ground-view
images from a single perspective, where intra-class variations are relatively small. AGPReID, Aerial-Ground
Person Re-Identification.

image-text pairs spanning diverse scenes, viewpoints, and visual appearances, which implicitly
endows it with the ability to associate semantically consistent content across different views. This
inherent generalization ability makes CLIP particularly suitable for the AGPReID task, where large
viewpoint and modality discrepancies pose significant challenges. Building upon this foundation,
we propose a CLIP-Based View-Consistent Alignment Framework (CVAF) with two training
stages for AGPReID task. CVAF leverages the powerful vision-language pretrained model CLIP [10]
to extract robust and view-invariant identity representations. In the first stage, we introduce a set of
learnable identity-aware text tokens, optimized via a supervised contrastive loss to align images and
textual descriptions across views. To ensure that these tokens capture consistent identity semantics,
we further propose a Text Consistency Loss (TCL), which enforces stable alignment between
multi-view image features and their shared identity description. This encourages the model to learn
stable, semantic-level representations that generalize across diverse viewpoints without relying on
manually labeled attributes. In the second stage, the learned text tokens are reused to guide image
feature refinement via a Semantic Filtering Module (SFM). SFM modulates image patch tokens
jointly across spatial and channel dimensions. It first applies text-guided cross-attention to generate
a spatial attention map, explicitly highlighting identity-relevant regions. Then, it reweights image
tokens based on their semantic similarity to the text features, suppressing background noise and
view-specific artifacts. Without the need for explicit viewpoint separation, SFM eliminates the risk
of identity feature loss while still extracting robust, view-invariant representations.
In summary, our main contributions are as follows:
— We present CVAF, a CLIP-based AGPReID framework that effectively aligns multimodal
representations and significantly improves cross-view matching performance.
— We introduce a novel TCL to enhance the stability and invariance of learned identity-aware
text tokens, and design a lightweight SFM to emphasize identity-relevant features.
— Extensive experiments on multiple AGPReID benchmarks demonstrate that CVAF consistently
outperforms existing state-of-the-art methods.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

85:4
2
2.1

D. Mao et al.

Related Work
Single-View Person ReID

Single-view person ReID primarily deals with pedestrian images captured from either ground or
aerial view networks. Ground-view based ReID has garnered the most attention in recent years, with
several benchmark datasets being introduced [12, 23, 29, 45]. Building on these datasets, numerous
innovative methods have been proposed, including many excellent CNN-based approaches [3, 42,
44, 49], such as PCB [24] and MGN [25] segment pedestrian images into horizontal stripes to extract
multi-granular local features from each part. OSNet [47] introduces multi-stream residual blocks
with unified aggregation gates to capture discriminative features across different spatial scales.
With the success of Transformer in computer vision [6], it has also been widely adopted in person
ReID [14, 16, 28, 33, 38, 39, 48]. TransReID [7] is the first to apply Transformer architectures to ReID,
introducing the JPM module that shuffles and regroups patch embeddings to enhance local feature
extraction and significantly improve performance. DC-Former [13] incorporates multiple class
tokens into the ViT architecture to represent different embedding spaces and explicitly encourages
them to be orthogonal, allowing each space to focus on distinct discriminative cues. AAFormer [50]
introduces learnable vectors to cluster patch embeddings into groups, enabling the model to learn
diverse local representations, which are then aligned through integrated self-attention mechanisms.
In contrast to ground-view person ReID, aerial-view person ReID has received relatively limited
attention. To date, only a few pioneering studies have contributed benchmark datasets [11, 41]
and proposed initial methodologies. RoExViT [26] introduces multi-rotation tokens to capture
rotation-specific features of pedestrians and employs a cross-camera similarity loss to reduce view
discrepancies. Additionally, it proposes an iterative adaptive batch construction strategy to mitigate
overfitting on small-scale aerial datasets, addressing the challenges of rotation variance and background style shifts in aerial ReID. SPAKD [21] leverages attention maps from ViT to extract salient
image patches, and integrates them with body part information derived from human parsing models.
This fusion enhances the precision of pedestrian representation. RotTrans [2] adopts feature-level
rotational data augmentation to improve robustness against viewpoint rotations in aerial images.
In datasets for single-view ReID, all images are captured from similar viewpoints, with minimal
viewpoint variation between cameras. These datasets rely on pedestrian appearance patterns specific
to the given view, where the global appearance of the pedestrian remains relatively consistent.
However, when applied to AGPReID scenarios characterized by extreme viewpoint variations,
these methods often struggle to maintain robust cross-view matching performance.
2.2

Cross-View AGPReID

Cross-view AGPReID aims to handle pedestrian images captured from both aerial and ground
view-based camera networks, focusing on achieving person matching between aerial and ground
perspectives. This task has gained attention in recent years, with the core challenge being the
extraction of robust, view-invariant features to address the significant appearance variations caused
by drastic viewpoint changes.
Before the emergence of AGPReID, cross-view person ReID tasks have been widely studied to
handle occlusion and viewpoint variations across multiple cameras. For example, ETNDNet [5]
enhances occlusion-robust feature learning through adversarial perturbations, while MVI²P [4]
aggregates features across multiple views to propagate reliable identity information. These methods
demonstrate effective strategies for leveraging multi-view cues, but they assume the availability of
multiple overlapping camera views, which is not the case in aerial-ground ReID scenarios where
images are captured from drastically different viewpoints with minimal overlap.
AG-ReID [18] contributed the first AGPReID dataset, AG-ReID, and proposed leveraging pedestrian identity attributes to address cross-view matching problem. AG-ReID.v2 [19] expanded the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

CVAF for Aerial-Ground Person Re-Identification

85:5

AG-ReID dataset and introduced high-angle attention flow to extract head-level local features as
view-invariant information, significantly enhancing model performance. However, both methods
rely heavily on prior attributes, limiting the model’s ability to autonomously learn view-invariant
features. View-Decoupled [40] introduced the CARGO dataset, a synthetic AGPReID dataset, and
proposed the VDT, which adds a view token to the Vision Transformer [6]. It extracts view-invariant
features by removing the view token from the CLS token through explicit subtraction, and introduces
an orthogonal loss to enforce the independence of the two features. SeCap [27] provided two realworld AGPReID datasets, LAGPeR and G2APS-ReID, and further enhanced VDT by incorporating
self-calibration and adaptive prompting. This method refines the view-invariant features extracted
by VDT using a prompt recalibration module and local feature refinement module, achieving
efficient cross-view identity matching. While view-decoupling methods have shown performance
improvements, excessive decoupling inevitably risks diluting some of the underlying view-invariant
features. In contrast to the above methods, our CVAF leverages the inherent view-invariant modeling capability of CLIP, explicitly guiding feature alignment and filtering, thus maximizing the
extraction of view-invariant features while preserving feature integrity and generalization.
2.3

Vision-Language Learning

Vision-language learning aims to enable multimodal perception and generation by jointly modeling
images and texts. Cross-modal alignment has been shown to significantly boost the performance of
various downstream tasks. CLIP [22] achieves such alignment by optimizing a bidirectional InfoNCE
loss between image and text features. Thanks to its remarkable cross-modal retrieval accuracy
and transferability, CLIP has been rapidly adopted across a wide range of vision and language
tasks. For example, CoOp [46] integrates prompt learning into CLIP for image classification, while
TCM [35] directly applies CLIP for text detection. CLIP-ReID [10] extends CLIP’s capabilities to
person ReID, marking the first application of vision-language pretraining in this domain. CFine
[31] aligns fine-grained image patches with textual descriptions to improve identity discrimination
in text-image retrieval. CSDN [36] leverages CLIP-based semantic cues to bridge modality gaps,
while MCGI [43] applies multi-granularity and cross-modality fusion to enhance fine-grained
instance retrieval. These works demonstrate the potential of CLIP for learning transferable and
discriminative representations.
As vision-language pretrained models continue to evolve, their superior cross-modal alignment
and generalization capabilities are continually pushing the boundaries of multimodal recognition
tasks. However, their potential in highly view-variant scenarios, such as AGPReID, remains underexplored. Built upon CLIP-ReID, our method introduces two key improvements to enhance
cross-view robustness. In the first stage, a TCL is incorporated to encourage the learnable identityspecific text tokens to capture more generalized and discriminative semantic representations across
diverse viewpoints. In the second stage, a SFM is proposed to parse the textual features and leverage them to guide the image encoder. Through cross-modal attention, SFM enables the model to
focus on semantically meaningful regions and learn view-invariant discriminative features. These
enhancements further exploit the vision-language alignment to address the challenges of AGPReID.
3
3.1

Methodology
Recap of CLIP-ReID

CLIP-ReID is the first framework that introduces the powerful vision–language model CLIP into
the person ReID domain. Its key insight lies in fully exploiting CLIP’s cross-modal representation
ability through a set of learnable text tokens assigned to each identity. These tokens are fed into
the text encoder to form semantically ambiguous yet identity-related textual descriptions, enabling
flexible alignment between image and text representations.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

85:6

D. Mao et al.

Fig. 2. Comparison between CLIP-ReID and the proposed CVAF. (a) CLIP-ReID employs learnable text tokens
and a two-stage training paradigm for image–text alignment. (b) CVAF extends CLIP-ReID by introducing
the TCL and SFM.

To effectively optimize such a cross-modal embedding space, CLIP-ReID proposes a two-stage
training strategy. In the first stage, the image and text encoders inherited from CLIP are frozen,
while the learnable text tokens are trained from scratch using a supervised contrastive loss computed within each batch. This stage aims to establish robust image–text associations by learning
discriminative text tokens for each identity. In the second stage, the learned ID-specific text tokens
and the text encoder are fixed, serving as stable semantic constraints to fine-tune the image encoder.
This progressive training paradigm allows the model to gradually bridge the visual–semantic gap
and enhance identity-level alignment across modalities, leading to more stable optimization and
superior cross-view matching performance.
3.2

Method Overview

As illustrated in Figure 2(b), we propose CVAF, a two-stage CLIP-based framework for AGPReID.
The framework builds upon the two-stage training paradigm of CLIP-ReID while introducing
two novel components: the TCL and SFM. In the first stage, we introduce an identity-level textsupervised view-consistency contrastive learning mechanism. For each identity, 𝑀 learnable text
tokens are initialized and fed into the frozen text encoder to generate ambiguous descriptions.
During this stage, both the image and text encoders from CLIP are kept frozen, and only the
text tokens are optimized from scratch based on the contrastive loss computed within a batch.
By combining the Supervised Contrastive Loss and the TCL, the model encourages the learned
text tokens to capture discriminative semantic representations for each identity across different
views. This stage promotes robust cross-modal associations by aligning the text features with the
corresponding pedestrian samples under diverse viewpoints. In the second stage, the image features
are enhanced and filtered in both the spatial and channel dimensions. The SFM is integrated into
the deeper layers of the image encoder. During this stage, the learnable text tokens and the text
encoder are kept frozen to provide a stable semantic constraint for fine-tuning the image encoder.
Through cross-modal attention, the regions of the image corresponding to spatial terms in the text
descriptions are aligned, optimizing the spatial attention map. This design ensures that the encoder
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

CVAF for Aerial-Ground Person Re-Identification

85:7

more accurately focuses on semantically relevant areas. Meanwhile, the SFM employs an attention
reweighting mechanism guided by textual semantics to filter out noise interference in the visual
tokens, further enhancing semantic relevance.
3.3

TCL

During the first-stage training of learnable text tokens, each identity in the training set is associated
with a textual description, while each description corresponds to multiple images of the same
ID captured from different views. As a result, the learned text tokens are expected to encode
view-invariant identity semantics, enabling them to consistently represent the same person across
diverse viewpoints. In the aerial-to-ground person ReID scenario, view-invariant representation
is particularly crucial due to the extreme viewpoint variations. To encourage the text encoder to
capture more robust, view-invariant semantic features, we introduce the TCL. The core idea of TCL
is to enforce consistency between the text embedding and image features from different views of
the same ID, thereby constraining the cross-modal similarity to remain stable across viewpoints.
The design of TCL is as follows. For each ID, we collect its associated text feature and a set of
image features captured from multiple views. To ensure fair similarity comparison, all features
are L2-normalized and projected onto a unit hypersphere, such that cosine similarity can be
directly computed. Given an ID with 𝑁 image features 𝐼 = {𝑖 1, 𝑖 2, . . . , 𝑖 𝑁 } and its corresponding
text embedding 𝑡, the similarity between each image feature and the text feature is computed as:
𝑖𝑘 · 𝑡
𝑠𝑖𝑚𝑘 =
, 𝑘 = {1, 2, . . . , 𝑁 }.
(1)
k𝑖𝑘 kk𝑡 k
Ideally, we believe that the similarity between different images and text features belonging to
the same ID should be as close as possible, which indicates that the text feature effectively captures
the shared, view-invariant information across all views. To quantify the variation in similarity, we
adopt variance as the core optimization objective of TCL, as it directly measures the dispersion of
similarity scores and sensitively reflects deviations caused by view-specific noise.
The final TCL formulation is:
𝑀
n
o
1 Õ
L𝑡𝑐𝑙 =
𝑉 𝑎𝑟 𝑠𝑘(𝑚) ,
(2)
𝑀 𝑚=1
where 𝑀 denotes the number of IDs in the current mini-batch that have valid multi-view image pairs,
and 𝑉 𝑎𝑟 () computes the variance over the similarities for each ID. This loss explicitly encourages
the textual representation to maintain consistent correlations with image features across varying
views, enhancing its ability to encode view-invariant features.
3.4

SFM

To enhance the sensitivity of the image encoder to the regions described by text features and
optimize image representations, we designed the SFM, which integrates both spatial and channel
semantic filtering functionalities, as depicted in Figure 3. SFM effectively combines textual features,
guiding the image encoder to suppress semantically irrelevant regions and enhance identity-related
areas. Specifically, given an image patch sequence 𝑋 ∈ R𝐵× (𝑁 +1) ×𝐶 , the CLS token 𝑋 cls and image
patch tokens 𝑋 patch are separated, where 𝐵 denotes batch size, 𝑁 represents the number of image
patches, and 𝐶 is the feature dimension.
After the first-stage training, the CLIP-encoded textual features 𝑡 ∈ R𝐵×𝐷 are obtained. These
features are projected into the visual space through a learnable linear transformation to generate
the query vectors 𝑄 = 𝑡𝑊𝑞 . Simultaneously, the image patch features are normalized using a
LayerNorm operation and then projected to obtain the key vectors 𝐾 = LN(𝑋 patch )𝑊𝑘 . Next, a
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

85:8

D. Mao et al.

Fig. 3. Overview of the proposed SFM.

cross-modal attention map is computed to measure the semantic relevance between the textual
queries and spatial locations in the image. This attention mechanism is particularly effective when
the textual features contain spatial cues, allowing the model to highlight regions in the image that
are semantically aligned with the description. The computation of the cross-modal attention map 𝐴
can be formulated as:


𝑄 · 𝐾𝑇
𝐴 =𝜎 √
+𝑏 ,
(3)
𝐶
where 𝑄 denotes the query vectors derived from the textual features 𝑡 and the learnable projection
matrix 𝑊𝑞 , 𝐾 represents the key vectors obtained from the normalized image patch representations,
√
𝜎 denotes the Sigmoid activation function. The attention computation includes a scaling factor 𝐶
to stabilize training, and a learnable bias term 𝑏 is added to enhance the flexibility of the attention
learning.
The obtained attention map 𝐴 are used as gating coefficients to modulate the contribution of
each image patch, resulting in the gated image representation 𝑋𝑔𝑎𝑡𝑒𝑑 :
𝑋𝑔𝑎𝑡𝑒𝑑 = 𝑋𝑝𝑎𝑡𝑐ℎ

𝐴𝑇 ,

(4)

refers to the element-wise multiplication, 𝐴𝑇 is the transposed matrix of attention map 𝐴.

where
Meanwhile, to further refine the image features, a text-guided dynamic channel weighting
mechanism is introduced. The text feature is mapped to a dynamic channel weight vector w𝑐 via a
network consisting of two fully connected layers and a ReLU activation, followed by a Sigmoid
function to constrain the output within the range [0, 1]:
w𝑐 = 𝜎 (𝑊2 𝑅𝑒𝐿𝑈 (𝑊1𝑡 + 𝑏 1 ) + 𝑏 2 ),

(5)

where 𝑊1 , 𝑊2 and 𝑏 1 , 𝑏 2 are the learnable weights and biases of the two layers, 𝜎 denotes the
Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
features in a fine-grained manner.
Finally, the generated weight vector w𝑐 is then applied to the gated image tokens 𝑋𝑔𝑎𝑡𝑒𝑑 via
element-wise multiplication. To preserve representational consistency, the re-weighted features are
fused with the original input using a residual connection, followed by a linear transformation to
project them back to the original token space. The final semantically filtered image representation
𝑜𝑢𝑡 can be formulated as:
𝑋𝑆𝐹
𝑀

𝑜𝑢𝑡
𝑋𝑆𝐹
w𝑐 ,
(6)
𝑀 = 𝑋𝑔𝑎𝑡𝑒𝑑 + 𝐹𝐶 𝑋𝑔𝑎𝑡𝑒𝑑
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf' - | sed -n '/3[[:space:]]*Methods/,/4[[:space:]]*Experiments/p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-modality average precision optimization for visible thermal person re-identification.pdf' - | sed -n '/3\\. Proposed method/,/4\\. Experiments/p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3. Proposed method
In this section, we provide a detailed description of our proposed
framework. As illustrated in Fig. 3, we adopt a Global–Local Structure
(GLS) to extract both Global and Local Features (GLF) from visible and
thermal modality images, respectively. Then we calculate the losses of
3

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

Fig. 3. The overview of our proposed method. (a) Illustration of our framework. Two independent streams are utilized for visible modality and thermal modality in the shallow
layers, which include the first three layers of the ResNet-50 [38]. After the shallow layers, we design two pipelines for extracting global-based features and local-based features.
Each pipeline consists of the last two layers of the ResNet-50 [38], which are shared by both modalities. The parameters of the two pipelines are different. Given the inputs,
we first use a Global–Local Structure (GLS) to extract Global and Local Features (GLF). We then calculate the losses of identity classification (𝐿ID ), the losses of Cross-modality
Average Precision (𝐿CAP ), and the losses of Dynamic Modality Alignment (𝐿DMA ) with each type of feature. (b) Illustration of the CAP loss (bottom) and the calculation process
of DMA loss (top). Shapes indicate the identities and colors represent the modalities (blue for visible and yellow for thermal). (For interpretation of the references to color in this
figure legend, the reader is referred to the web version of this article.)

galleries are from different modalities. Given a training mini-batch, we
have 𝑛𝑣 visible modality samples and 𝑛𝑡 thermal modality samples. The
features obtained by the model are defined as F𝑣 and F𝑡 for visible
modality and thermal modality, respectively. We will introduce how to
obtain the representation of samples in Section 3.3. We perform CAP
by considering two aspects, CAP with Visible queries (CAPV), and CAP
with Thermal queries (CAPT).
At the CAPV aspect, we set 𝑓𝑣𝑞 as a query instance from visible
modality, and set 𝑃𝑡𝑞 and 𝑁𝑡𝑞 as its corresponding positive set and
negative set from thermal modality. The CAPV is calculated as:
∑
1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑖𝑗 )
1 ∑
𝑡
𝑞
𝐴𝑃 (𝑓𝑣 ) ≈ 𝑞
.
(5)
∑
∑
𝑖
𝑖
|𝑃 |
| 𝑡 | 𝑖∈𝑃𝑡𝑞 1 + 𝑗∈𝑃𝑡𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑗 ) + 𝑗∈𝑁𝑡𝑞 𝑔(𝑠𝑗 )
The loss of CAPV is calculated as:
𝑛𝑣
)
1 ∑(
𝐿𝐶 𝐴𝑃 𝑉 (F𝑣 , F𝑡 ) =
1 − 𝐴𝑃 (𝑓𝑣𝑞 ) .
𝑛𝑣 𝑞=1

Fig. 4. Illustrate the complementary results of CAP and DMA. The pairs with high
weights are connected with wider lines. Shapes indicate the identities and colors
represent the modalities (blue for visible and yellow for thermal). DMA can effectively
smooth the cross-modality gap (a → b). CAP can directly improve the cross-modality
overall ranking orders (a → c). These two aspects jointly optimize the model and gain
mutual benefit results (d). (For interpretation of the references to color in this figure
legend, the reader is referred to the web version of this article.)

(6)

Similarly, at the CAPT aspect, we set 𝑓𝑡𝑞 as a query instance from
thermal modality, and set 𝑃𝑣𝑞 and 𝑁𝑣𝑞 as its corresponding positive set
and negative set from visible modality. The CAPT is calculated as:
∑
1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑖𝑗 )
1 ∑
𝑣
.
(7)
𝐴𝑃 (𝑓𝑡𝑞 ) ≈ 𝑞
∑
𝑖) + ∑
𝑖
|𝑃𝑣 |
𝑞
𝑞
1
+
𝑔(𝑠
𝑞
| | 𝑖∈𝑃𝑣
𝑗∈𝑃𝑣 ,𝑗≠𝑖
𝑗∈𝑁𝑣 𝑔(𝑠𝑗 )
𝑗
The loss of CAPT is calculated as:
𝑛𝑡
)
1 ∑(
𝐿𝐶 𝐴𝑃 𝑇 (F𝑣 , F𝑡 ) =
1 − 𝐴𝑃 (𝑓𝑡𝑞 ) .
𝑛𝑡 𝑞=1
The CAP loss combines these two aspects of losses:
}
1{
𝐿𝐶 𝐴𝑃 𝑉 (F𝑣 , F𝑡 ) + 𝐿𝐶 𝐴𝑃 𝑇 (F𝑣 , F𝑡 )
2
3.2. Dynamic modality alignment
𝐿𝐶 𝐴𝑃 (F𝑣 , F𝑡 ) =

(8)

(9)

In the VT-ReID task, the large cross-modality discrepancy has a negative effect on the CAP. To cope with this issue, we propose a Dynamic
Modality Alignment (DMA) to smooth the large inter-modality gap by
constructing a dynamic Cross-modality Affiliated Matrix (CAM), which
can mitigate the influence of non-modality variations and achieve
better modality alignment. The details of the DMA loss calculation are
shown in Fig. 3 (b-top).
𝑛 ×𝑛
We set 𝑫 ∈ R+𝑣 𝑡 as the distance matrix between visible modality
and thermal modality, which elements are calculated as:
‖
‖2
𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ) = ‖𝑓𝑣,𝑖 − 𝑓𝑡,𝑗 ‖ ,
(10)
‖
‖2
where 𝑓𝑣,𝑖 ∈ F𝑣 and 𝑓𝑡,𝑗 ∈ F𝑡 , ‖.‖ represents the L2-norm. The distance includes cross-modality discrepancy and non-modality variations
(e.g. different identities, views, and poses).
In order to reduce the influence of non-modality variations during
modality alignment, we utilize a dynamic Cross-modality Affiliated
Matrix (CAM) to calculate the distribution distance between visible

Fig. 5. (a) The Heaviside step function 𝑢(𝑥) (top) and its derivative 𝑑 𝑢(𝑥)∕𝑑 𝑥 (bottom).
The derivative of the Heaviside step function is flat with zero except at 𝑥 = 0, which
prevents effective model optimization. (b) The sigmoid-derived function 𝑔(𝑥; 𝜏) (top)
and its derivative 𝑑 𝑔(𝑥)∕𝑑 𝑥 (bottom), which can be directly optimized using gradientdescent methods. 𝜏 = 0.1.

4

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

and DMA on them.
Global Feature. We directly apply the generalized-mean pooling
(GeM) [41] and the batch normalization neck (BNNeck) [42] on the
last residual convolution layer of global stream 𝑓 𝑔,𝑟𝑒𝑠 and obtain the
global feature 𝑓 𝑔 ∈ R1,𝐶 .
Local Features. We first evenly divide the last residual convolution
layer of local stream 𝑓 𝑙,𝑟𝑒𝑠 into 𝐾 parts along the vertical direction and
perform the GeM and BNNeck on them, obtaining 𝐾 local features,
{𝑓 𝑙,1 , … , 𝑓 𝑙,𝐾 }.
Losses for CAP. Given the global feature and local features, the
losses of CAP can be calculated as:
𝐿𝐶 𝐴𝑃 = 𝐿𝑔𝐶 𝐴𝑃 + 𝐿𝑙𝐶 𝐴𝑃
(
)
∑
(
) 𝐾
= 𝐿𝐶 𝐴𝑃 F𝑔𝑣 , F𝑔𝑡 +
𝐿𝐶 𝐴𝑃 F𝑙𝑣,𝑘 , F𝑙𝑡,𝑘 ,

(13)

𝑘=1

where 𝐾 is the part number, 𝐿𝑔CAP is the global-based loss of CAP and
𝐿𝑙CAP is the local-based loss of CAP.
Losses for DMA. Similarly, the losses of DMA can be calculated as:
𝐿𝐷𝑀 𝐴 = 𝐿𝑔𝐷𝑀 𝐴 + 𝐿𝑙𝐷𝑀 𝐴

Fig. 6. Illustration of the negative correlation between the weights of cross-modality
pairs, 𝑾 𝑖, 𝑗, and their distance, 𝑫(𝑓 𝑣, 𝑖, 𝑓𝑡,𝑗 ). Larger weights are assigned to crossmodality pairs that are closer to each other. Pairs with higher weights are represented
by thicker lines.

𝐾
(
)
(
) ∑
= 𝐿𝐷𝑀 𝐴 F𝑔𝑣 , F𝑔𝑡 +
𝐿𝐷𝑀 𝐴 F𝑙𝑣,𝑘 , F𝑙𝑡,𝑘 ,

(14)

𝑘=1

Losses for Identity Classification. We add classification heads (1
for global feature, 𝐾 for local part features), which are fully connected
layers with an output length of #identities, after these two types of
features. We then calculated identity classification losses according to
the outputs of these classification heads, formulated as:
𝐿ID = 𝐿𝑔ID + 𝐿𝑙ID

𝑛 ×𝑛

and thermal modality. We set 𝑾 ∈ R+𝑣 𝑡 as CAM. Intuitively, a
larger distance between samples from the visible and thermal modalities indicates greater non-modality variations, such as differences in
viewpoint and pose. At this time, a smaller weight 𝑾 𝑖,𝑗 should be assigned. This enables the model to prioritize cross-modality sample pairs
with larger weights, thus focusing on handling the cross-modality discrepancy, while simultaneously neglecting those with smaller weights
(cross-modality pairs characterized by more substantial non-modality
variation). Therefore, the elements of 𝑾 matrix can be calculated as:
exp(1 − 𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ))
𝑾 𝑖,𝑗 = ∑
.
(11)
𝑗∈𝑁𝑡 exp(1 − 𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ))

= Cr ossEnt r opy (H𝑔 , Y) +

𝐾
∑

(
)
Cr ossEnt r opy H𝑙,𝑘 , Y ,

(15)

𝑘=1

where H indicates the predictions of the corresponding classification
heads and Y is the identity labels.
3.4. Overall
Training. The proposed framework can be trained in an end-toend manner by minimizing the loss of identity classification, the loss
of DMA, and the loss of CAP, which are calculated as follows:

We calculate the distribution distance between visible and thermal
modality by the weighted summation of cross-modality pairs distance,
and treat it as the DMA loss, which is calculated as:
𝑛𝑣 𝑛𝑡
1 ∑∑
𝐿𝐷𝑀 𝐴 (F𝑣 , F𝑡 ) =
𝑾 𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ).
(12)
𝑛𝑣 + 𝑛𝑡 𝑖=1 𝑗=1 𝑖,𝑗

ar g min 𝐿𝐼 𝐷 + 𝐿𝐶 𝐴𝑃 + 𝛼 𝐿𝐷𝑀 𝐴 ,
𝜃

(16)

where 𝛼 is the hyper-parameters.
Testing. In the testing phase, different from other global and local
combination methods (e.g. [10,21]), we obtain the final Global–Local
Feature (GLF) by concatenating the global feature and local features
with a ratio:
[
]
(17)
𝑓𝑡𝑒𝑠𝑡 = 𝛽(𝑓 𝑙,1 ∣ ⋯ ∣ 𝑓 𝑙,𝐾 )|(1 − 𝛽)𝑓 𝑔 ,

During optimization, due to the negative correlation between crossmodality pairs distance 𝑫(𝑓𝑣,𝑖 , 𝑓𝑡,𝑗 ) and their weights 𝑾 𝑖,𝑗 , the larger
weight will be assigned for cross-modality pairs (𝑓𝑣,𝑖 and 𝑓𝑡,𝑗 ) when they
are close to each other, otherwise a small weight will be assigned for
them (Fig. 6). In such a scenario, when aligning two modalities with the
DMA loss, the model would concentrate on reducing the discrepancy
between cross-modality pairs that have less non-modality variations
(i.e., closer pair). Nevertheless, the pairs with larger non-modality
variations (i.e., distant pairs) would receive less attention from model
optimization, since they are assigned with low weights. Therefore, the
DMA can efficiently reduce the influence of non-modality variations
and achieve better modality alignment, which can further enhance the
CAP result.

where [⋅ ∣ ⋅] indicates the concatenation operation, 𝛽 is the hyperparameter that controls the importance of global and local features. The
dimensions of the global features, local features, and testing features
are 2048, 256, and 3584, respectively.
4. Experiments

 succeeded in 127ms:
���=1 |𝑝(𝑦𝑖 )| 𝑝 ∈𝑝(𝑦 )
exp(𝑠(𝒇
,
𝒇𝑦𝑡 ))
𝑗
𝑗=1
𝑖
𝑖

(1)

(2)

𝑖

𝑛𝑏

𝐿𝑖𝑑 = −
3

1 ∑
𝑞 𝑙𝑜𝑔(𝑾 (𝒇𝑖 )),
𝑛𝑏 𝑖=1 𝑖

(3)

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

𝐿𝑖2𝑡𝑐𝑒 = −

and Convnet-based visual encoder. Among them, the transformer makes
use of the attention mechanism and is more conducive to modeling
global information. As for CNN, convolutional operations are good
at mining local information from images and are suitable for visual
tasks with small datasets such as pedestrian re-identification. Therefore,
in order to enhance the model’s ability to ultimately extract visual
features, we introduced the VSS module that combines the advantages
of transformer and CNN.
Specifically, to obtain global features, we divide images of shape
(B, C, H, W) into patches and project them into tokens of shape (B,
N, L). Then, we use the Multi-Head Self-Attention (MHSA) mechanism (Vaswani, 2017) to build the global information between different
tokens. To obtain local features, we use a series of standard convolution
blocks following the structure of ResNet-50 and reduce the size of the
features by increasing the stride at the end of each layer. As shown
in Fig. 3, for fusing 𝑓𝑖𝑙 and 𝑓𝑖𝑔 , we multiplied 𝑓𝑖𝑙 , 𝑓𝑖𝑔 with 𝑊𝑙 , 𝑊𝑔 and
added together. The integration process can be expressed as follows:

(4)

𝐿𝑡𝑟𝑖 = max(𝑑𝑝 − 𝑑𝑛 + 𝛼, 0),
𝑛𝑏
exp(𝑠(𝒇𝑖 , 𝒇𝑦𝑡 ))
1 ∑
𝑖
𝑞𝑖 log( ∑𝑁
),
𝑐
𝑛𝑏 𝑖=1
exp(𝑠(𝒇
,
𝒇𝑦𝑡 ))
𝑖
𝑦 =1

(5)

𝑎

𝑎

𝐿𝑠𝑡𝑎𝑔𝑒1 = 𝐿𝑡2𝑖 + 𝐿𝑖2𝑡 ,

(6)

𝐿𝑠𝑡𝑎𝑔𝑒2 = 𝐿𝑖𝑑 + 𝐿𝑡𝑟𝑖 + 𝐿𝑖2𝑡𝑐𝑒 ,

(7)

where 𝒇𝑖 𝑡 is extracted by T(⋅), 𝑓𝑖 is extracted by I(⋅), 𝑛𝑏 is for batch size,
N is the total number of batches. s(⋅) represents the similarity between
two vectors; 𝒇𝑦𝑖 𝑡 is the text feature with identity 𝑦𝑖 ; 𝑃 (𝑦𝑖 ) denotes a set
composed of indexes of samples with identity 𝑦𝑖 ; |𝑃 (𝑦𝑖 )| indicates the
cardinality of 𝑃 (𝑦𝑖 ). 𝑞𝑖 is the one-hot vector of identity label 𝑦𝑖 ; 𝑁𝑐 is the
total number of identities; W is identity classifier; 𝑑𝑝 and 𝑑𝑛 are feature
distances of positive pair and negative pair, while 𝛼 is the margin of
𝐿𝑡𝑟𝑖 .

𝑓𝑖𝑙 , 𝑓𝑖𝑔 = (𝑊𝑙 , 𝑊𝑔 ) ⊗ (𝑓𝑖𝑙 , 𝑓𝑖𝑔 ),
B Our proposed CSGN

where 𝑊𝑙 , 𝑊𝑔 are one-dimensional matrices of the same length as
(𝑓𝑖𝑙 , 𝑓𝑖𝑔 ); ⊗ represents the matrix multiplication; 𝑓𝑖𝑙 and 𝑓𝑖𝑔 are features containing rich complementary visual information. For inference,
we deploy a triplet loss to ensure that 𝑓𝑖𝑙 and 𝑓𝑖𝑔 can learn unique
identity-independent features.
(3) HSF: For the multiple language descriptions learned by MDG,
it is crucial to effectively leverage the semantic information. CLIPVIReID (Yu et al., 2024) gives a scheme, in which the learnable
prompt for two specific modalities is designed and the Attention Fusion
Module was proposed to decrease the modal difference. Specifically,
they treat the text features corresponding to the visible image as
‘Key’, and the text features corresponding to the infrared image as
‘Query’ and ‘Value’, by attention mechanism and contrastive losses to
improve the semantic richness of text features. However, for clotheschanging person re-identification, there is no modal difference between
multiple descriptions but only informational difference, e.g., language
descriptions generated from RGB images focus on information such
as clothing, skin color, hair style, etc., whereas language descriptions
generated from gray-scale images focus on information such as body
shape, pose, etc. For this reason, in order to avoid visual features being
biased towards a single text feature, we propose the HSF loss to learn
more comprehensive individual features. As shown in Eq. (11), the first
term will direct the learned features to align with the description of
the color image, while the second term will guide the visual features
to favor the description of the grayscale image. Since the description
of the grayscale image emphasizes body shape information, whereas
the description of the color image focuses on color-related details,
we balance the contributions of both pseudo-text descriptions using a
weighted computational approach. This ensures that the final model
incorporates all relevant discriminative information in the image.
Specifically, as shown in Eq. (11), we employ 𝐿ℎ𝑠𝑓 to constrain the
𝑡
𝑡
visual features using 𝑓𝑖 𝑅 and 𝑓𝑖 𝐺 obtained by MDG. Where the pair
𝑡𝐺
𝑡𝑅
𝑣
𝑣
(𝑓𝑖 , 𝑓𝑖 )’s and (𝑓𝑖 , 𝑓𝑖 )’s distance wound be decreased. 𝑓𝑖𝑣 represent
the visual feature.

For CC-ReID, as time progresses, the change in a person’s clothing
introduces significant challenges to the accuracy of the model. Given
the challenges mentioned above, we propose a CLIP-driven Semantic
Guidance Network (CSGN), which consists of a Multi-Description Generator (MDG), Heterogeneous Semantic Fusion loss (HSF), and Visual
Semantic Steering module (VSS).
(1) MDG: In CSGN, we introduce a learnable prompt for pedestrians
with the same identity. However, unlike other subtasks in person reidentification, CC-ReID is more complex because different individuals
may wear similar clothing, while the same person may wear different
clothing. Specifically, for CLIP, differences in descriptions of a person
are often reflected in changes in their clothing. To address this, we
design the Multi-Description Generator (MDG) to mitigate the effects
of clothing changes by generating diverse descriptions for the same
pedestrian.
Specifically, we assign two learnable language descriptions to each
of the RGB image and its corresponding grayscale image, denoted as
𝑇𝑖𝑅 and 𝑇𝑖𝐺 , respectively. At the start of the first stage, 𝑇𝑖𝑅 and 𝑇𝑖𝐺 are
initialized to ‘‘a photo of [𝑇1 ][𝑇2 ]...[𝑇𝑀 ][𝑐𝑙𝑠] person’’. To differentiate
between language descriptions, we further use a technique called MetaInsert, which generates a conditional token (vector) for each input. This
token is then combined with the text vectors. Specifically, we design
a mini-network consisting of a two-layer bottleneck structure (LinearReLU-Linear) to project image features into the text space. Let ℎ𝜃 ()
denote the mini-network parameterized by 𝜃, so that each text token
is now obtained by 𝑇𝑚 (𝑥) = 𝑇𝑚 (𝑥) + ℎ𝜃 (𝑥). The loss for the first stage is
then reformulated as follows:
𝑡

𝐿1𝑖2𝑡 = −

𝑁
exp(𝑠(𝑓𝑖𝑅 , 𝑓𝑖 𝑅 ))
1 ∑
log( ∑𝑛
)
𝑡
𝑏
𝑁 𝑖=1
exp(𝑠(𝑓𝑖𝑅 , 𝑓𝑗 𝑅 ))
𝑗=1

(8)

𝑡

−

𝑁
exp(𝑠(𝑓𝑖𝐺 , 𝑓𝑖 𝐺 ))
1 ∑
log( ∑𝑛
),
𝑡
𝑏
𝑁 𝑖=1
exp(𝑠(𝑓𝑖𝐺 , 𝑓𝑗 𝐺 ))
𝑗=1
𝑡

𝐿1𝑡2𝑖 = −

𝑁
exp(𝑠(𝑓𝑝𝑅 , 𝑓𝑦𝑅𝑖 ))
∑
1 ∑ 1
𝑖
log( ∑𝑛
)
𝑡𝑅
𝑏
𝑅
𝑁 𝑖=1 |𝑝(𝑦𝑖 )| 𝑝 ∈𝑝(𝑦 )
exp(𝑠(𝑓
𝑗 , 𝑓𝑦𝑖 ))
𝑖
𝑖
𝑗=1
𝑡

𝑁
exp(𝑠(𝑓𝑝𝐺 , 𝑓𝑦𝐺𝑖 ))
∑
1 ∑ 1
𝑖
−
log( ∑𝑛
),
𝑡𝐺
𝐺
𝑏
𝑁 𝑖=1 |𝑝(𝑦𝑖 )| 𝑝 ∈𝑝(𝑦 )
exp(𝑠(𝑓
𝑖
𝑖
𝑗 , 𝑓𝑦𝑖 ))
𝑗=1
𝑡

(10)

𝑡

𝐿2ℎ𝑠𝑓 = −

(9)

𝑛𝑏
exp(𝑠(𝑓𝑖𝑉 , 𝑓𝑦𝑅𝑖 ))
𝛼 ∑
𝑞𝑖 log( ∑𝑁
)
𝑡
𝑐
𝑛𝑏 𝑖=1
exp(𝑠(𝑓𝑖𝑉 , 𝑓𝑦𝑅𝑎 ))
𝑦 =1
𝑎

𝑡

𝑛𝑏
exp(𝑠(𝑓𝑖𝑉 , 𝑓𝑦𝐺𝑖 ))
1−𝛼 ∑
−
𝑞𝑖 log( ∑𝑁
),
𝑡
𝑐
𝑛𝑏 𝑖=1
exp(𝑠(𝑓𝑖𝑉 , 𝑓𝑦𝐺𝑎 ))
𝑦 =1

𝑡

(11)

𝑎

where 𝑓𝑖 𝑅 and 𝑓𝑖 𝐺 denote the text features of 𝑇𝑖𝑅 and 𝑇𝑖𝐺 . 𝑇𝑚 (𝑥)
represent the text token of image 𝑥.
𝑡
𝑡
(2) VSS: It is feasible to utilize 𝑓𝑖 𝑅 and 𝑓𝑖 𝐺 to facilitate 𝑓𝑖𝑣 learning.
However, CLIP’s visual encoder fails to adequately extract visual features of the target identity. The visual encoder currently used in CLIP
can be divided into two categories, Transformer-based visual encoder

1 ∑
𝑞 𝑙𝑜𝑔(𝑊 (𝑓𝑖𝑔 ))
(12)
𝑁 𝑖=1
𝑁 𝑖=1 𝑖
Note that superscript 1 of the formula represents the loss function
used in the first stage of training and superscript 2 represents the loss
function used in the second stage.
𝐿2𝑖𝑑 = −

4

𝑁
1 ∑

𝑁

𝑞𝑖 𝑙𝑜𝑔(𝑊 (𝑓𝑖𝑙 )) −

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Fig. 3. VSS module.
Table 1
Comparison with state-of-the-art methods on Celeb-ReID, Celeb-Light and VC-Clothes
datasets.

C Training and inference
The approach in this paper divides the entire training process into
two stages. In the first stage, under the constraints of Eqs. (8) and (9),
we train the MDG equipped with two learnable prompts for generating
text descriptions corresponding to RGB and grayscale images. Upon
entering the second stage, we input the RGB and gray-scale images
to obtain the pseudo-text generated during the first stage. We then
impose the constraints defined in Eq. (4), Eq. (11), and Eq. (12) to train
the VSS module, which aims to integrate the visual information in the
global and local features to obtain more robust features. The constraints
defined in Eq. (12) can improve the discriminative properties of the
visual representations. It is important to note that, during the training
phase, only RGB images are fed into the VSS module. During the
inference phase, however, only the features extracted by the visual
encoder are used for identity matching through the cosine similarity
metric. The other components are not required, ensuring that the utility
of our proposed CSGN remains unaffected.

Mthod

Venue

dataset
Celeb-ReID

Celeb-Light

VC-Clothes

rank-1 mAP rank-1 mAP rank-1 mAP

4. Experiments

FSAM (Hong et al., 2021)
CVPR21 –
IRANet (Shi et al., 2022)
IVC22
64.1
MBUNet (Zhang et al., 2023)
ICME22 55.3
DeSKPro (Wu et al., 2022)
ICIP22
68.6
CCAT (Ren et al., 2022)
IJCNN22 –
UCAD (Yan et al., 2022)
IJCAI22 –
CC-ReID (Mu et al., 2022)
BMVC22 –
MVSE (Gao et al., 2022)
ACM22 64.5
Co-attention (Wang et al., 2022) ACCV22 57.5
CIHANet (Zhou et al., 2022a)
ICPR22 –
DSA (Chen et al., 2021)
TMM22 –
CCIL (Li et al., 2023a)
Arxiv23 –
DualBCtnet (Li et al., 2022)
TCSVT23 –
ACID (Yang et al., 2023b)
TIP23
52.5
IGCL (Gao et al., 2023)
CVPR23 –

–
–
19
46.2
12.1 33.9
22.7 52
–
–
–
–
–
–
19.2 –
12.3 –
–
47.2
–
–
–
–
–
–
11.4 27.9
–
41

–
25.4
21.3
29.8
–
–
–
–
–
26.8
–
–
–
15.8
23.4

78.6
–
–
–
83.8
92.6
84.5
–
–
–
82
80.4
92.2
84.3
82.9

78.9
–
–
–
–
81.1
84.3
–
–
–
80
–
73.6
74.2
84.8

CSGN(our)

23.3 56.4

38.4 92.7

85.1

64.6

A Experimental settings
padding, and cropping. In the first stage, we train MDG for 120 epochs.
The initial learning rate is set to 3 × 10−4 and decayed according to the
cosine schedule (Li et al., 2023b). Subsequently, in the second stage,
we focus on designing a visual encoder and classifier fusing global and
local features for 60 epochs of training. The learning rate in this phase
increases linearly from 3×10−6 to 3×10−4 during the first 10 epochs and
decays by a factor of 0.1 at the 20th and 40th epochs. The batch size
is set to 64, with 8 RGB images per pedestrian. The training process of
the proposed CSGN uses the Adam optimizer (Kingma and Ba, 2014).

(1)Datasets: As shown in Fig. 4, we conduct experiments on four
public CC-ReID datasets: PRCC, Celeb-ReID, Celeb-light, and
VC-Clothes. PRCC is a rigorous clothes-changing dataset, consisting
of 33,698 images from 221 individuals, captured from 3 different
views. It also provides contour sketch images to facilitate the extraction of silhouette information. Celeb-ReID is one of the first datasets,
comprising 590 individuals and 10,842 images, while Celeb-light is a
lighter version of Celeb-ReID. VC-Clothes is a virtual dataset from the
game GTA-V, containing 512 IDs and 19,060 images. It also includes
a real-world-based test dataset with 4324 images from 28 IDs. All of
these datasets were acquired using multiple cameras over various time
periods and contain multiple composite disturbances, such as lighting
changes, pose variations, occlusions, clothing changes, and more.
(2) Evaluation metrics: In this paper, we primarily use Cumulative
Matching Precision (rank-k) and Mean Average Precision (mAP) to
evaluate the model’s performance. Rank-k represents the probability of
a correct match appearing in the top K results, sorted by confidence.
mAP treats pedestrian re-identification as a retrieval task and calculates
the mean of the average precision scores across all classes in the
dataset.
(3) Implementation details: we implemented the proposed CSGN
using the PyTorch deep learning framework, and all experiments were
deployed on a GTX 4090 GPU. We use CLIP (Radford et al., 2021) as
the backbone, and the input images are uniformly resized to 256 × 128
with common data augmentation strategies such as random flipping,

B Comparison with state-of-the-art methods
In this section, we compare the proposed CSGN with the state-ofthe-art methods across four widely used CC-ReID datasets. The results
are summarized in Tables 1 and 2. The best result is highlighted in bold,
while the second-best result is underlined.
Experiment on Celeb-ReID, Celeb-Light, and VC-Clothes: To evaluate the effectiveness of our approach, we first conducted experiments
on three widely used datasets, Celeb-ReID, Celeb-Light, and VC-Clothes,
and compared our method with the latest studies. As shown in Table
1, the semantic segmentation-based approach achieves satisfactory performance when handling complex clothing transformation scenarios.
For example, DeSKPro (Wu et al., 2022) and IGCL (Gao et al., 2023)
achieve rank-1 accuracies of 68.6% and 52.0%, and mAP of 22.7%
and 29.8% on Celeb-ReID and Celeb-Light, respectively. This significant
5

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Fig. 4. Illustration of examples in four datasets.
Table 2
Comparison with state-of-the-art methods at the PRCC dataset.
Mthod

Venue

Table 3
Verify the effectiveness of each component in CSGN on PRCC.

PRCC
SC

CC

rank-1rank-10mAP rank-1rank-10mAP
FSAM (Hong et al., 2021)
CVPR21 98.8 100
RCSANet (Huang et al., 2021)
ICCV21 100 –
GI-ReID (Jin et al., 2022)
CVPR22 –
–
CAL (Gu et al., 2022)
CVPR22 100 –
IRANet (Shi et al., 2022)
IVC22 100 –
self-attention (Bansal et al., 2022)WACV22 99.7 100
CC-ReID (Mu et al., 2022)
BMVC22 99.6 –
MVSE (Gao et al., 2022)
ACM22 –
–
DLAW (Liu et al., 2023)
TIP23
98.42 –
CCIL (Li et al., 2023a)
Arxiv23 58.2 86.7
DualBCt-net (Li et al., 2022)
TCSVT2385.0 91.3
DCR-ReID (Cui et al., 2023)
TCSVT23100 –
CCFA (Han et al., 2023)
CVPR23 99.6 –
AIM (Yang et al., 2023a)
CVPR23 100 –
ADC (Yang et al., 2022)
ICME22 99.6 –
CSGN(our)

100

100

–
54.5 86.4
97.2 50.2 –
–
33.26 75.09
99.8 55.2 –
99.4 54.9 –
–
47
84.93
97.3 61.8 –
–
47.4 –
98.7756.22 –
–
–
–
–
–
–
99.7 57.2 –
98.7 61.2 –
99.9 57.9 –
98.8 51.6 –

–
48.6
–
55.8
53.0
–
58.7
52.5
57.11
–
–
57.4
58.4
58.3
56.2

99.4 63.2 73.8

59.7

Method

MDG

VSS

HSF

rank-1

rank-10

mAP

Baseline(R50)
Baseline (V-B-16)
CLIP-ReID
Stage1
Stage2
Stage2

–
–
–
✓
✓
✓

–
–
–
–
✓
✓

–
–
–
–
–
✓

41.0
45.8
48.7
50.9
51.7
63.2

–
–
57.1
60.1
60.3
72.0

37.5
46.1
48.5
50.8
51.5
59.7

The above results comprehensively demonstrate the superiority of
our approach across the four benchmarks. This superiority is mainly attributed to the following key factors:(1) We introduce CLIP to facilitate
CCReID learning, enabling the model to acquire high-level semantic information. This helps alleviate the challenges posed by high intra-class
and inter-class variation. (2) Rather than simply relying on pre-trained
CLIP models, we adopt MDG to generate language descriptions that
enhance visual feature learning, thus fully leveraging CLIP’s potential.
(3) To address the incomplete visual information, we design a VSS
module that integrates both global and local information. Additionally,
we propose HSF loss to comprehensively constrain the learning of
visual features by combining identity loss and triplet loss.

improvement is primarily attributed to their use of semantic segmentation maps to guide feature learning, along with high-resolution facial
images provided to the teacher model, which in turn helps the student
model learn discriminative features from low-resolution facial images.
Nevertheless, under the same test conditions, our CSGN outperforms
DeSKPro by 0.6% in terms of mAP on the Celeb-ReID dataset, and by
4.4% and 18.6% in terms of rank-1 and mAP on the Celeb-Light dataset,
respectively. Moreover, several recently competitive methods have also
shown excellent performance, such as ACID (Yang et al., 2023b). In
comparison, on the Celeb-Light dataset, our CSGN outperforms ACID
by 22.6% in terms of mAP and 28.5% in terms of rank-1 accuracy. On
the VC-Clothes dataset, our CSGN surpasses ACID by 10.9% in mAP and
by 8.4% in rank-1 accuracy, demonstrating a considerable advantage.
Experiment on PRCC: To further validate the effectiveness of our
method, we conducted experiments on PRCC. As shown in Table 2, the
proposed CSGN performs excellently across all environments. Specifically, our algorithm achieves 100.0% rank-1 accuracy and 99.4% mAP
when the clothing remains unchanged, and 63.2% rank-1 accuracy, and
59.7% mAP when the clothing changes.

C Ablation experiments
In this section, we conduct an ablation study to validate the efficacy
of each module in the proposed method. Specifically, we first qualitatively analyze the performance gains associated with each component,
and the results are summarized in Table 3. We compare our approach
with CLIP-ReID as the baseline. Starting with the first stage of CSGN,
where we add MDG, we observe an improvement of 2.2% in rank-1
accuracy and 2.3% in mAP. When we further incorporate the VSS module in stage 2 of CSGN, performance improves by 3.0% in both rank-1
accuracy and mAP. Finally, adding the HSF module in stage 2 leads
to a significant performance boost of 14.5% in rank-1 accuracy and
11.2% in mAP. All experiments were conducted on the PRCC dataset
in the clothes-changing scenario, demonstrating the effectiveness and
superiority of our design components.
(1) Effectiveness of MDG: In CSGN, we introduce a learnable prompt
for pedestrians with the same identity. The results presented in Table 4
6

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Table 4
Verify the effectiveness of MDG on PRCC, MI is our Meta-Insert skill.

Table 6
Verify the effectiveness of HSF on PRCC.

Method

rank-1

rank-10

mAP

Method

rank-1

rank-10

mAP

CoOp
CoCoOp
MDG(without MI)
MDG(with MI)

48.7
46.8
47.7
50.9

57.1
57.5
55.4
60.1

48.5
49.8
49.2
50.8

Base
RGB
Gray-scale

48.7
52.6
51.0

57.1
66.5
64.7

48.5
56.3
55.3

HSF

63.2

72.0

59.7

Table 5
Verify the effectiveness of VSS on PRCC, The ‘*’ indicates results reproduced using the
ViT backbone through the source code.
Methods

Backbones

rank-1

rank-10

mAP

PCB (Sun et al., 2018)
MGN (Wang et al., 2018)
ResNet-50(He et al., 2016)
FSAM (Hong et al., 2021)
GI-ReID (Jin et al., 2022)
CAL (Jin et al., 2022)
CLIP(R50)(Radford et al., 2021)

ResNet
ResNet
ResNet
ResNet
ResNet
ResNet
ResNet

22.9
25.9
19.6
54.5
37.6
55.2
41.0

–
–
–
–
–
–
–

–
–
8.1
–
–
55.8
37.5

DenseNet-121(Huang et al., 2017)
RCSANet (Huang et al., 2021)
MVSE (Gao et al., 2022)

DenseNet
DenseNet
DenseNet

18.7
31.6
47.4

–
–
–

23.7
31.5
52.5

ViT-VIBE Hybrid (Bansal et al., 2022)
CAL*(Jin et al., 2022)
MVSE*(Gao et al., 2022)
CLIP(Vit-B-16)(Radford et al., 2021)

VIT
VIT
VIT
VIT

47.0
32.7
27.3
45.8

–
–
–
–

–
33.6
28.8
46.1

VSS
VSS
VSS

CNN+CNN
ViT+ViT
Hybrid

41.6
45.9
51.7

50.9
52.2
60.3

37.8
46.6
51.5

𝛼 is big or small, the performance of our VSS module wound decrease.
The best 𝛼 value is 0.4. This is attributed to the structural design of the
HSF loss, whose first term guides the visual features to bias the pseudotext space generated by RGB images, and the second term guides the
visual features to bias the pseudo-text space generated by gray-scale
images. Since the two together constitute the HSF loss in an additive
manner, its overall optimization is the result of integrating both color
distribution and shape information in the visual features.
(4) Effectiveness of CSGN: Based on the above discussion and experimental validation, our proposed CSGN contains two different description generators for generating language descriptions of RGB images
and gray-scale images respectively, which alleviates the problem of
inaccurate CLIP-ReID prompt generation. Based on this, the HSF loss
plays the role of fusing descriptions, whose target is enriching textual
features and guiding the learning of the CCReID model. Notably, the
algorithm achieves the best performance on Celeb, Celeb-Light, and VCClothes datasets, with a rank-1 accuracy of 64.6%,56.4%, and 92.7%
and a mAP of 23.3%,38.4%, and 85.1%, respectively. Furthermore, the
visualization in Fig. 6 where ‘red’ presents the wrong match and ‘green’
presents the right match vividly shows that the proposed CSGN enables
the model to accurately identify the target pedestrian. These results
demonstrate the positive contribution of each module to the proposed
method. In conclusion, our CSGN is both rational and effective and
exhibits excellent performance.

(Stage 1) show that this approach outperforms pre-training the model
with CLIP alone, with improvements of 2.2%, 3.0%, and 2.3% in rank1, rank-10, and mAP on the PRCC dataset, respectively. Additionally, in
Table 4, we compare MDG with popular prompt generation methods.
When comparing MDG with CoOP, our MDG’s performance surpasses
CoOP by 2.2% in rank-1 accuracy and 2.3% in mAP. Similarly, when
comparing MDG with CoCoOP, our MDG outperforms it by 4.1% in
rank-1 accuracy and 1.0% in mAP. These results highlight the superiority of our method. It can be demonstrated that our MDG generates
richer semantic information, largely due to the inclusion of grayscale images. While RGB images provide pseudo-text focusing on color
distribution, gray-scale images are less affected by color interference
and emphasize shape information, such as body shape. When used together in training the MDGs, they generate more instructive pseudo-text
descriptions, combining the strengths of both modalities.
(2) Effectiveness of VSS: As shown in Table 5, CLIP achieves the best
performance whether using ResNet or ViT as the backbone. However,
when comparing VSS with CLIP(R50), our VSS surpasses it by 10.7%
in rank-1 and 14.0% in mAP. Similarly, when comparing VSS with
CLIP(ViT-B-16), our VSS outperforms it by 5.9% in rank-1 and 5.4%
in mAP. This demonstrates that our VSS module effectively combines
the strengths of both ViT and ResNet. This improvement is made
possible by the introduction of a lightweight fusion header, which
fuses global and local features in a balanced manner. It assigns equallength weight vectors to both feature types, and the fused feature
is obtained by summing the products of the two features with their
corresponding weight vectors. This approach is conceptually similar to
the fully connected layer but avoids the complex computations by using
simple weight vector products instead.
(3) Effectiveness of HSF: As shown in Table 6, comparing line 1 with
our HSF loss, our method outperforms 14.5% and 11.2% in rank-1 and
mAP than no use. Comparing line 2 with our HSF loss, our method
outperforms 10.6% and 3.4% in rank-1 and mAP. Comparing line 3
with our HSF loss, our method outperforms 12.2% and 4.4% in rank-1
and mAP. This demonstrates that our HSF loss can utilize high-level
semantic information to guide the learning of visual features, which in
turn breaks the bottleneck. As shown in Fig. 5, we can see that when

5. Conclusion
In this paper, we apply CLIP to the CCReID task and propose a twostage CSGN framework. In the first phase, we introduce the innovative
Multi-Description Generator (MDG), which generates pseudo-text descriptions in both RGB and grayscale modes. This dual-mode approach,
combined with a text-image mutual loss mechanism, effectively reduces the color sensitivity of the text encoder. Notably, each linguistic
description captures different semantic nuances. In the second phase,
we design the Visual Semantic Steering (VSS) module, a crucial component that enhances visual feature representation by leveraging the
complementary nature of ResNet and ViT feature extractors. To further
refine our model, we incorporate Heterogeneous Semantic Fusion (HSF)
loss, which imposes stricter constraints on the individual components,
ensuring a comprehensive and robust model.
However, this work represents only a preliminary exploration in
the field of clothing change re-identification. More in-depth research is
required to fully exploit the advantages of visual-language models for
improving the feature extraction capabilities of visual models. Firstly,
the linguistic descriptions generated by our method are still relatively
coarse. This is largely due to the lack of accurate linguistic descriptions
in the clothing change re-identification datasets, which limits our ability to fine-tune large models effectively and fully utilize their potential.
This constraint impacts the richness of the semantic information. To address this, we plan to construct a new clothing change re-identification
dataset with accurate linguistic descriptions. Secondly, clothing change
re-identification still faces challenges in improving the visual encoder.
Specifically, extracting global and local discriminative features with
fewer parameters and computations remains a significant challenge.
As such, we intend to investigate more efficient and accurate network
architectures in future work.
7

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Fig. 5. The effect of the selection of the hyperparameter 𝛼.

Fig. 6. Comparing visual sorting results of different algorithms.

8


codex
`CSGN` 和 `CVAF` 也读完了前 8 页。几篇长文的方法小节跨页，我会补读后续 2 到 3 页来把训练目标、模块闭环和消融证据读完整，这不改变你要求的主读法，只是补全 method 末尾。
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
/bin/zsh -lc "pdftotext -f 8 -l 12 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
135:8

H. Yin et al.

where 𝑀 is the number of image patches retained with high similarity to the token 𝑖, and 𝑎𝑔𝑔𝑖,𝑗
quantifies the influence of patch 𝑗 in forming the language-grouped vision embedding (referred to as
joint embedding) associated with token 𝑖. This explicit aggregation strategy ensures a comprehensive
interaction between token 𝑖 and its corresponding patch 𝑗 during local alignment. In particular, the
aggregation weight 𝑎𝑔𝑔𝑖,𝑗 effectively captures the semantic relevance between token 𝑖 and patch 𝑗,
thereby facilitating precise alignment.
Next, we derive the corresponding joint embedding 𝑒𝑖 as:
𝑒𝑖 =

𝑁
Õ

(8)

𝑎𝑔𝑔𝑖,𝑗 · 𝑓 𝑗𝑣 ,

𝑗=1

where 𝑁 is the count of image patches. The resulting set of joint embedding 𝑒𝑖 has the same length
𝐿 as the text token 𝑓𝑖𝑡 .
𝐿 and
Hard Coding Alignment. We calculate the similarity between the joint embeddings {𝑒𝑖 }𝑖=1
𝐿 as well as image embeddings {𝑓 𝑣 } 𝑁 , retheir corresponding original text embeddings {𝑓𝑖𝑡 }𝑖=1
𝑖 𝑖=1
spectively. To reduce both computational and memory costs, we adopt a hard coding similarity
computation between the joint embeddings and their corresponding text and image embeddings,
and the theoretical analysis of the hard coding is provided in Appendix A. For simplicity, we only
present the calculation between the joint embeddings and the text embeddings, while the remaining
computations follow a similar and symmetric approach.
For the text 𝑇 and its corresponding joint embedding 𝐸, we calculate the original similarity matrix
𝐿 and their joint embeddings {𝑒 }𝐿 , where 𝑜
𝑡 >
𝑡
𝑂 between all text tokens {𝑓𝑖𝑡 }𝑖=1
𝑖 𝑖=1
𝑖,𝑗 = 𝑓𝑖 𝑒 𝑗 /k𝑓𝑖 kk𝑒 𝑗 k
𝑡
𝑡
means the cosine similarity of 𝑓𝑖 and 𝑒 𝑗 . For the token 𝑓𝑖 , we compute the weight factor between
it and all joint embeddings using the following hard coding way:


if 𝑗 = argmax (𝑜𝑖,𝑗 0 )
1

𝑗 0 =1···𝐿
𝜔𝑖,𝑗 =
(9)

 0 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒.

Then, we utilize the LSE pooling [23] to compute the hard similarity between text 𝑇 and its
corresponding joint embedding 𝐸 by:
!
𝐿
Õ
ℎ𝑎𝑟𝑑_𝑠 (𝑇 , 𝐸) = 𝐿𝑆𝐸 − 𝑃𝑜𝑜𝑙𝑖𝑛𝑔
𝜔𝑖,𝑗 𝑜𝑖,𝑗 ,
𝑗=1
𝐿
Õ

(10)




1
= log
exp 𝜆 max 𝑜𝑖,𝑗 ,
𝑗=1···𝐿
𝜆
𝑖=1
where 𝜆 controls the degree to which the most relevant text embeddings and their corresponding
joint embeddings are emphasized.
Given a batch containing 𝐵 text embeddings along with their associated joint embeddings, we
compute the hard coding similarity matrix 𝐻𝑎𝑟𝑑_𝑆 following Equations (9) and (10), as illustrated
in Figure 3(b). We calculate the EFA loss from the text to its joint embedding, adapted from the
triplet ranking loss [39]:


Õ
𝐻𝑎𝑟𝑑_𝑆 neg − 𝐻𝑎𝑟𝑑_𝑆 pos + 𝑚𝑎𝑟𝑔𝑖𝑛
1
L𝑡 2𝑒 = log
exp
,
(11)
𝐵
𝜏2
neg
where 𝜏2 is a scaling factor adjusting the spread of the loss, and 𝑚𝑎𝑟𝑔𝑖𝑛 is a distance hyperparameter
defining the minimal gap separating positive and negative pairs.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

Cross-Modal Full-Mode Fine-Grained Alignment for TIPR

135:9

Similarly, the EFA loss from the joint embedding to its original text can be computed following
Equation (11), and we can calculate the EFA loss between image and its joint embedding through
Equations (9)–(11). Then, we obtain a full EFA loss by:
L𝐸𝐹𝐴 = L𝑡 2𝑒 + L𝑒2𝑡 + L𝑖2𝑒 + L𝑒2𝑖 .
3.4

(12)

Training Objective

As mentioned, FMFA aims to improve both the global and local cross-modal alignment of image-text
features within the shared embedding space. To realize this goal, the widely adopted ID loss [59]
and IRR loss [21], together with the proposed EFA and A-SDM losses, are jointly utilized to train
FMFA. The ID loss directly classifies the global features obtained from both the image and the text
according to their identities, thereby enhancing the global alignment of the model. The IRR loss,
based on the Masked Language Modeling task [46], leverages an attention mechanism for implicit
cross-modal interaction to obtain a joint embedding, and then predicts the [MASK] text token to
enhance the local alignment of the model.
FMFA is trained end-to-end, with the complete training objective formulated as:
L = L𝐼 𝐷 + L𝐼𝑅𝑅 + L𝐸𝐹𝐴 + L𝐴-𝑆𝐷𝑀.
4
4.1

(13)

Experiments
Datasets and Settings

Datasets. We assess FMFA on three widely used text-based person retrieval datasets, following the
data splits introduced in IRRA [21]. CUHK-PEDES [27] contains 40,206 images associated with
13,003 identities, where each image is paired with two textual descriptions. Of these identities,
11,003 are designated for training, while the remaining 1,000 identities are allocated separately to
validation and test sets. ICFG-PEDES [10] includes 54,522 images belonging to 4,102 individuals,
each image linked to a single sentence. The conventional setup utilizes 3,102 identities for training
and reserves 1,000 identities for testing. RSTPReid [60] comprises 20,505 images from 4,101 identities
captured across 15 camera views. Every identity corresponds to five images taken from different
viewpoints, and each image is annotated with two descriptive captions. The dataset follows a split
with 3,701 identities for training and 200 identities each for validation and testing.
Evaluation Metrics. To gauge retrieval quality, we primarily report Rank-K results (K = 1, 5, 10),
which measure how often the correct item appears within the top-K predictions. Additionally, Mean
Average Precision (mAP) is adopted to summarize ranking accuracy over all query outcomes. In
both cases, higher metric values correspond to superior model behavior.
Implementation Details. We utilize either the original CLIP model [37] or its ReID-domain pretrained variants [22, 45] as encoders tailored to each modality. To maintain consistency, we employ
the identical CLIP-ViT-B/16 model for visual encoding and Xformer for text encoding, following
the setup used in IRRA [21] for our experiments. Specifically, images are resized to 384 × 128 pixels,
and the maximum sequence length 𝐿 for input word tokens is set to 77. The model is trained using
the Adam optimizer for 60 epochs with a default cosine learning rate decay schedule, in contrast to
the 100 epochs employed for the ICFG-PEDES dataset. The original CLIP model parameters are
trained with an initial learning rate of 1𝑒 − 5 and a batch size of 64. In particular, the temperature
𝜏1 in the A-SDM loss is set to 0.02, while the temperature 𝜏2 in the EFA loss takes a value of 1.0.
The weight factor 𝛼 of A-SDM is set to 10.0 by default, and set to 1 in the RSTPReid dataset, and
the factor 𝜆 in the LSE pooling is set to 1.0. Due to variations in data distribution, the margins
used in the EFA loss differ across the three datasets. The specific margins used in Equation (11)
for each dataset are provided in Table 1. When using ReID-domain pre-trained CLIP models, we
adopt the same initial learning rate and batch size as in NAM [45] and HAM [22], while keeping all
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

135:10

H. Yin et al.
Table 1. The Margins Utilized in the EFA Loss
CUHK-PEDES

ICFG-PEDES

RSTPReid

0.1
0.1

0.2
1.0

0.2
0.8

T to E
V to E

“T to E” means the EFA loss from textual embeddings to
the corresponding joint embeddings, and “V to E” means
the EFA loss from visual embeddings to the corresponding joint embeddings.

Table 2. The Hardware
Configuration of Our
Experimental Environment
Hardware

Details

CPU
GPU
RAM

Intel Xeon Gold 6330
NVIDIA RTX A6000
755 GB DDR4

other settings unchanged. The hardware configuration used in our experiments is shown in Table 2,
while the detailed software environment is supplied in the code repository we have released.
4.2

Comparison with State-of-the-Art Methods

In this subsection, we provide a comparison with current state-of-the-art methods (e.g., NAM
[45] and HAM [22]) on three public benchmark datasets. The methods are grouped into two
types according to their underlying network architecture, as listed in Tables 3–5: those using VLBackbones without ReID-domain pre-training and those incorporating ReID-domain pre-training.
Furthermore, according to whether local features are utilized during inference, the baselines are
further classified into local and global matching methods (denoted as “L” and “G” in the “Type”
column, respectively). It should be noted that the baseline model presented in Tables 3–5 is referred
to as IRRA𝑅 , which represents the performance of our reimplementation of the IRRA model. CLIP
means the ViT-B/16 architecture after fine-tuning under the InfoNCE loss [34].
Evaluation Results on CUHK-PEDES. We measure the performance of FMFA on the CUHK-PEDES
dataset, as presented in Table 3. When using the VL-Backbones without ReID-domain pre-training,
FMFA achieves superior performance over advanced global matching methods, attaining 74.16%
Rank-1 and 66.66% mAP, while surpassing IRRA by 0.74% in Rank-5 and 0.41% in Rank-10. When
adopting the VL-Backbones with ReID-domain pre-training, FMFA maintains its superiority and
achieves Rank-5 accuracy exceeding 95% with the HAM-based backbone. Notably, FMFA with
NAM-based backbone attains 91.33% in Rank-5, outperforming IRRA with the HAM-based backbone
by 0.13%.
Evaluation Results on RSTPReid. We assess FMFA on the latest RSTPReid benchmark, as presented in Table 4. Using the VL-Backbones without ReID-domain pre-training, FMFA achieves
competitive performance, attaining 61.05% Rank-1, 83.85% Rank-5, 89.80% Rank-10, and 48.22%
mAP, respectively, outperforming IRRA by 1.55% in Rank-1 and 2.05% in Rank-5. When adopting
the VL-Backbones with ReID-domain pre-training, our method achieves further gains, exceeding
IRRA by 0.45% in Rank-1 with both the NAM-based and HAM-based backbones. Notably, FMFA
achieves Rank-5 accuracy higher than 88% with the HAM-based backbone.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

Cross-Modal Full-Mode Fine-Grained Alignment for TIPR

135:11

Table 3. Comparisons with State-of-the-Art Methods on the CUHK-PEDES Dataset
Type Method
Ref.
VL-Backbones w/o ReID-Domain Pre-Training
LGUR [42]
MM22
IVT [43]
ECCV22
VGSG [17]
TIP23
G
CLIP [37]
ICML21
DM-Adapter [29]
AAAI25
IRRA𝑅 [21]
CVPR23
TBPS-CLIP [4]
AAAI24
FMFA (ours)
ACSA [20]
TMM22
Han et al. [13]
arXiv21
PLOT [35]
ECCV24
L
RaSa [2]
IJCAI23
PTMI [32]
TIFS25
APTM [55]
MM23
SCVD [51]
TCSVT24
VL-Backbones with ReID-Domain Pre-Training
UniPT [41] + IRRA [21]
ICCV23
PLIP [61] + IRRA [21]
NeurIPS24
NAM [45] + IRRA𝑅
CVPR24
P+G
NAM [45] + FMFA (ours)
𝑅
HAM [22] + IRRA
CVPR25
HAM [22] + FMFA (ours)

Image Enc.

Text Enc.

Rank-1

Rank-5

Rank-10

mAP

ResNet50
ViT-B/16
ResNet50
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
Swin-B
CLIP-RN101
CLIP-ViT
CLIP-ViT
CLIP-ViT
Swin-B
CLIP-RN50

BERT
BERT
Transformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
BERT
CLIP-Xformer
CLIP-Xformer
BERT-base
CLIP-Xformer
BERT-base
CLIP-Xformer

65.25
65.59
67.52
68.19
72.17
73.45
73.54
74.16
63.56
64.08
75.28
76.51
76.02
76.53
76.72

83.12
83.11
84.37
86.47
88.74
89.38
88.19
90.12
81.49
81.73
90.42
90.29
89.93
90.04
90.38

89.00
89.21
90.26
91.47
92.85
93.69
92.35
94.10
87.70
88.19
94.12
94.25
94.14
94.15
94.89

61.12
64.33
66.13
65.38
66.66
60.08
69.38
70.85
66.91
-

CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT

CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer

74.37
74.25
76.67
77.23
77.32
77.46

89.51
89.49
91.11
91.33
91.20
91.36

93.97
93.68
94.60
94.75
94.95
95.01

66.60
66.52
68.42
68.53
68.87
68.89

“G,” “L,” and “P” in the “Type” column stand for global-matching method, local-matching method, and pretrained model with ReID-domain, respectively. “Image Enc.” and “Text Enc.” mean the backbone of image
encoder and text encoder, respectively. “IRRA𝑅 ” means the model that we reproduce. Bold denotes best result.

Table 4. Comparisons with State-of-the-Art Methods on the
RSTPReid Dataset
Type Method
Rank-1 Rank-5
VL-Backbones w/o ReID-Domain Pre-Training
DSSL [60]
39.05
62.60
IVT [43]
46.70
70.00
CLIP [37]
54.05
80.70
G
IRRA𝑅 [21]
59.50
81.80
DM-Adapter [29]
60.00
82.10
FMFA (ours)
61.05
83.85
ACSA [32]
48.40
71.85
CFine [53]
50.55
72.50
L
PLOT [35]
61.80
82.85
RaSa [2]
66.90
86.50
APTM [55]
67.50
85.70
VL-Backbones with ReID-Domain Pre-Training
UniPT [41] + IRRA [21]
62.20
83.30
PLIP [61] + IRRA [21]
64.35
83.75
NAM [45] + IRRA𝑅
68.25
86.75
P+G
NAM [45] + FMFA (ours)
68.70
87.05
HAM [22] + IRRA𝑅
71.35
87.60
HAM [22] + FMFA (ours)
71.80
88.05

Rank-10

mAP

73.95
78.80
88.00
88.85
87.90
89.80
81.45
81.60
89.45
91.35
91.45

43.41
47.44
47.37
48.22
52.31
52.56

89.75
91.00
92.30
92.35
93.05
93.15

48.33
50.93
52.92
53.14
55.40
55.72

Bold denotes best result.

Evaluation Results on ICFG-PEDES. We assess FMFA on the ICFG-PEDES benchmark, with the
results displayed in Table 5. Using VL-Backbones without ReID-domain pre-training, FMFA obtains
the leading results across all metrics, attaining 64.29% Rank-1 and 39.43% mAP. Compared with
IRRA, FMFA shows a notable improvement of 0.81% Rank-1 and 1.23% mAP, which is meaningful
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

135:12

H. Yin et al.
Table 5. Comparisons with State-of-the-Art Methods on the
ICFG-PEDES Dataset
Type Method
Rank-1 Rank-5
VL-Backbones w/o ReID-Domain Pre-Training
DuaL Path [59]
38.99
59.44
IVT [43]
56.04
73.60
CLIP [37]
56.74
75.72
G
VGSG [17]
60.34
76.01
DM-Adapter [29]
62.64
79.53
IRRA𝑅 [21]
63.48
80.16
FMFA (ours)
64.29
80.48
SSAN [10]
54.23
72.63
ISANet [54]
57.73
75.42
L
CFine [53]
60.83
76.55
RaSa [2]
65.28
80.40
PLOT [35]
65.76
81.39
VL-Backbones with ReID-Domain Pre-Training
UniPT [41] + IRRA [21]
64.50
80.24
PLIP [61] + IRRA [21]
65.79
81.94
NAM [45] + IRRA𝑅
66.34
81.94
P+G
NAM [45] + FMFA (ours)
66.58
81.94
HAM [22] + IRRA𝑅
68.21
83.28
HAM [22] + FMFA (ours)
68.37
83.28

Rank-10

mAP

68.41
80.22
82.26
82.01
85.32
85.78
85.93
79.53
81.72
82.42
85.12
86.73

31.84
36.50
38.20
39.43
41.29

85.74
87.32
86.73
87.04
88.04
88.10

38.22
39.43
40.14
40.17
41.72
41.76

Bold denotes best result.

Table 6. Ablation Analysis of FMFA Modules across Three Public Benchmarks
No.

Methods

0
1
2
3

Baseline
+A-SDM
+EFA
FMFA

Components
SDM A-SDM EFA
Ø
Ø
Ø
Ø
Ø
Ø

Rank-1
73.45
74.04
73.73
74.16

CUHK-PEDES
Rank-5 Rank-10
89.38
93.69
89.86
93.89
89.64
94.04
90.12
94.10

mAP
66.13
66.45
66.40
66.66

Rank-1
63.48
64.26
63.77
64.29

ICFG-PEDES
Rank-5 Rank-10
80.16
85.78
80.59
85.90
80.39
85.86
80.48
85.93

mAP
38.20
39.39
39.17
39.43

Rank-1
59.50
60.25
60.45
61.05

RSTPReid
Rank-5 Rank-10
81.80
88.85
81.45
88.70
82.30
89.25
83.85
89.80

mAP
47.44
47.69
47.64
48.22

Bold denotes best result.

for practical applications. When adopting VL-Backbones with ReID-domain pre-training, FMFA
yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
HAM-based backbones, respectively.
In conclusion, FMFA attains the highest performance across all evaluation metrics on the three
widely used public benchmarks. As far as we are aware, FMFA is the best method for all global
matching methods. This highlights the ability of our method to generalize well and maintain
robustness.
4.3

Ablation Study

In this subsection, we examine our proposed components in the FMFA framework. For simplicity,
we omit the components of L𝐼 𝐷 and the IRR module that were proposed by IRRA and used in all
experiments. Only one of SDM and A-SDM can be used at the same time.
To thoroughly assess the contribution of our FMFA modules, we undertake an empirical analysis
on three widely used datasets. Table 6 summarizes the Rank-1/5/10 accuracies (%) together with
the mAP (%) performance.
Effect of the A-SDM Module. To evaluate the contribution of the A-SDM module, we perform
ablation experiments by replacing the A-SDM module with the SDM module, keeping all hyperparameters unchanged. Specifically, as shown in Table 6, replacing A-SDM with SDM results in a
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.


exec
/bin/zsh -lc "pdftotext -f 3 -l 5 'Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

The young man
is wearing sky
blue jacket....

Stage 1: Virtual-text Embedding
Synthesizer (VES)

Et

Lscor

LIBM

Visual Backbone
Branch

CRCL

s
Lalign
2

1

...

z v,i

Freezed Network
Learnable Network
CLIP Visual/Textual FC Single Fully
Ev Et Encoders
Connected Layer

Batch

g()
Adapter

Ev

s1

Lalign

...
Virtual-text features

Virtual-text Branch

Text features

...

...

...

VES

...

z^ t,i

Visual
Backbone

Lid

Textual
Backbone

Visual/Tex
tual Spaces
Visual/Textu
al Prototypes
Visual/Virtualtext Features
Relations
Information
Transfer

A man in his
lateThe
twenties
young man
with
short.... sky
is wearing
blue jacket....

...

FC

z t,i

Image features

...

Ti
Ii

...

Stage 2: Cross-modal Collaboration

A man in his
lateThe
twenties
young man
with
short.... sky
is wearing
blue jacket....

Batch

Cross-modal Relation Consistency Loss
(CRCL)

Figure 2: Overview of our CoRL. In the first stage, we introduce a Virtual-text Embedding Synthesizer to generate a virtual-text embedding
from each image. The generated virtual-text embedding is then used to replace actual text in modeling multi-modal complementarity with
the image. In the second stage, we propose a dual-branch Cross-Modal Collaboration learning process equipped with a Cross-modal Relation
Consistency Loss, which encourages the exchange of information and mutual enhancement between image and virtual-text features.

2.2

Vision-Language Pre-Training

Vision-language pre-training involves training large-scale
models on vast collections of image-text pairs, demonstrating exceptional capabilities in semantic understanding [Shen
and Tang, 2024; Shen et al., 2025], multi-modal alignment,
and generalization. It has been widely used in diverse downstream tasks. The primary objective of pre-training is to
achieve a deep understanding of both image and text semantics and their correspondences. Some studies [Li et al.,
2019b; Li et al., 2021] focus on generating multi-modal representations by encoding interactions between images and
texts with tasks such as mask language modeling and image captioning. However, these methods typically necessitate pairwise interactions for all image-text pairs, leading
to inefficiency during training and inference, thus limiting
their practicality in large-scale applications. Consequently,
some studies [Radford et al., 2021; Yao et al., 2022] integrate
contrastive representation learning into Vision-language pretraining. These methods encode images and texts separately
into a joint space and learn modality-aligned representations
by contrasting positive and negative pairs, such as the wellknown CLIP [Radford et al., 2021], trained on 400 million
image-text pairs. The strong multi-modal alignment capability of CLIP prompts us to introduce it into the TIPR task.

3

Methods

This section presents our proposed CoRL framework, with an
overview in Figure 2 and details in the following subsections.

3.1

Problem Formulation

The TIPR dataset contains image-text pairs of persons with
multiple identities, where each identity has multiple images
collected from different cameras with their annotated text descriptions. The goal is to accurately identify images that belong to the same identity as a given text description. This requires the TIPR model to extract identity-discriminative representations and accurately establish cross-modal correspondences. To achieve this, we first use a pre-trained CLIP as

2154

the backbone to provide high-quality initialization and facilitate fine-tuning on the TIPR dataset. Next, we propose a
cross-modal collaborative representation learning framework
that leverages inter-modality complementary information to
enhance image features. Additionally, we design an identitybounded matching loss to fully leverage identity information
and precisely establish cross-modal correspondences.

3.2

Cross-modal Collaborative Representation
Learning

Current TIPR methods mainly emphasize image-text associations but overlook their complementarity. Texts offer
the following advantages over images: texts primarily contain pedestrian identity details with less noise. Even when
pedestrians are visually similar, their text descriptions are
often distinct, making text information more discriminative
and robust. Complementing images with textual information
enhances fine-grained discrimination and reduces modality
gaps. However, this requires access to corresponding text for
each image, which is not always available, especially during inference. In this paper, we tackle two key challenges:
eliminating the dependency on actual text and effectively utilizing text information to enhance image representations. To
achieve this, we propose a two-stage cross-modal collaborative representation learning strategy. In the first stage, we
develop a Virtual-text Embedding Synthesizer (VES) to generate virtual-text embeddings directly from each image, eliminating the need for actual paired text. In the second stage, we
implement a dual-branch cross-modal collaboration learning
process to integrate the virtual-text information into images,
thereby enhancing image features.
Virtual-text Embedding Synthesizer. To eliminate reliance on actual paired text, we propose a Virtual-text Embedding Synthesizer (VES) that generates a virtual-text embedding directly from each image, as illustrated in Figure 2.
CLIP, trained on vast amounts of image-text data, can generate modality-aligned image-text features. We utilize its
alignment capabilities to convert images into text embeddings. VES functions as a CLIP-based encoder-decoder
model. Specifically, an image Ii is first processed by CLIP’s

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

visual encoder Ev to produce an image embedding zv,i .
This embedding is then transformed into virtual-text tokens
through a fully connected layer. Finally, CLIP’s textual encoder Et decodes these virtual-text tokens to generate the
virtual-text embedding ẑt,i .
This generated virtual-text embedding ẑt,i can serve as a
substitute for the actual text embedding to enhance its image
feature, given that the virtual-text and actual-text embeddings
are consistent. To ensure this consistency, we introduce an
alignment loss Ls1
align that aligns the generated virtual-text
embedding with the actual text embedding zt,i at both feature
and semantic levels, inheriting the advantages of the actual
text Ti .
s1
s1
Ls1
align = Lnce + Lmse

Ls1
nce = −

B
T
exp(zt,i
ẑt,i )
1 X
log PB
T
B i=1
j=1 exp(zt,i ẑt,j )

B
T
exp(ẑt,i
zt,i )
1 X
−
log PB
T
B i=1
j=1 exp(ẑt,i zt,j )

(1)

(2)

B

Ls1
mse =

1 X
∥ẑt,i − zt,i ∥2
B i=1

(3)

where zt,i denotes the actual text embedding of text Ti , generated by CLIP’s visual encoder. ∥ · ∥2 denotes the L2 distance,
and τa denotes the temperature factor. To facilitate effective
conversion between images and virtual-text embeddings, we
impose a correlation loss Ls1
cor to ensure that valuable image
information is retained as much as possible throughout the
conversion process.
Ls1
cor = −

B
T
exp(zv,i
ẑt,i /τa )
1 X
log PB
T
B i=1
j=1,lj ̸=li exp(zv,i ẑt,j /τa )

B
T
exp(ẑt,i
zv,i /τa )
1 X
−
log PB
T
B i=1
j=1,lj ̸=li exp(ẑt,i zv,j /τa )

(4)

s2
s2
Ls2
align = λ1 Lnce + Lmse .
B

Ls2
nce = −

B

2155

(5)

1 X
exp(t̂Ti sg(ti ))
−
log PB
T
B i=1
j=1 exp(t̂i sg(tj ))
B

Ls2
mse =

1 X
∥t̂i − sg(ti )∥2
B i=1

(6)

Where sg(·) means stop-gradient operator, which ensures that
only the Adapter is updated. With the image feature vi and
the virtual-text feature t̂i , we introduce a cross-modal relation consistency loss (CRCL). This loss enforces that the image and virtual-text features share the same relationship with
modality-specific prototypes, promoting the two branches to
exchange information and cooperate with each other. For
each batch B, we cluster the image and text features based
on identity labels to obtain visual and textual prototypes, pv
and pt . We then compute the cosine similarity between the
image feature vi and the visual prototypes pv to determine
their relationship riv . Similarly, we calculate the relationship
rit between the virtual-text feature t̂i and these textual prototypes pt . Thus, the CRCL is formulated as:
B

Ls2
CRCL =


1 X
Dkl (rit ||riv ) + Dkl (riv ||rit )
2|B| i=1

(7)

where Dkl is the Kullback-Leibler divergence. Mutual
knowledge distillation on instance-to-prototype relations between the visual embedding vi and virtual-text embedding t̂i
allows them to exchange information and benefit from one another. Incorporating virtual-text information emphasizes the
discriminative and modality-shared identity details in the image Ii , boosting the discriminability and robustness of the image feature vi , making it easier to differentiate visually similar pedestrians while effectively reducing the modality gap
between the image Ii and the text Ti .

3.3

where li represents the identity label of Ii /Ti . The overall
s1
s1
objective of our VES is calculated as: Ls1
V ES = Lalign +Lcor .
VES enables the generation of virtual-text embedding that is
aligned with the actual text for each image, eliminating the
need for actual text.
Cross-modal Collaboration. With the virtual-text embedding, we design a dual-branch cross-modal collaboration
learning process to integrate it with the corresponding image, thereby enhancing the image feature. Specifically, for an
image-text pair (Ii , Ti ), the visual backbone branch generates
the image feature vi of image Ii . Simultaneously, the virtualtext branch first produces a virtual-text embedding ẑt,i from
image Ii using VES, which is then processed by an Adapter
g(·) to yield a virtual-text feature t̂i adapted to the target domain. Additionally, the text feature ti for text Ti is generated
via the textual backbone. We optimize the Adapter by aligning the virtual-text feature t̂i with the text feature ti through

1 X
exp(sg(ti )T t̂i )
log PB
T
B i=1
j=1 exp(sg(ti ) t̂j )

Identity-bounded Matching

When a batch contains multiple image-text pairs of the same
identity, the model must perform pairwise matching between
all images and texts in the batch. Due to view variations,
significant differences among samples of the same identity
can lead to noisy matching of cross-view image-text pairs of
the same identity. Specifically, within the same identity, each
image and its corresponding text form a single-view strong
positive pair, while image-text pairs across different views
are considered as cross-view weak positive pairs. Imagetext pairs with different identities are categorized as negative
pairs. Existing matching losses often treat strong and weak
positive pairs equally, overlooking noisy correspondences of
weak positive pairs. To address the issue, we propose an
identity-bounded matching (IBM) loss to establish more precise cross-modal correspondences.
To fully leverage identity information, we use a P K sampling strategy to construct batch samples. For each batch,

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

we randomly sample K identities and then randomly select P images for each identity, with each image annotated
with the corresponding text. Thus, each batch contains B
= P K images and their corresponding texts. This results in
P K strong positive pairs, KP (P − 1) weak positive pairs,
and P 2 K(K − 1) negative pairs. We compute the cosine
PK
similarity for these image-text pairs, denoted as {ssp
i }i=1 ,
P 2 K(K−1)

KP (P −1)

{swp
, and {sni }i=1
respectively. To distini }i=1
guish different image-text pairs, we propose an IBM loss that
enforces the condition: ssp >swp >sn . This can be expressed
equivalently as: ssp >α, sn <β, β<swp <α. We further reformulate this condition as follows:
(ssp − α)>0, −(sn − β)>0, (swp − β)>0, −(swp − α)>0
(8)
where α and β are the upper and lower bounds for ssp and sn ,
with α>β. We further implement IBM loss based on logistic
loss as follows:
(P K
i
h
X
sp
1
LIBM =
log 1 + e−τsp (si −α)
P K i=1
KP (P −1)

+

X

h
i
wp
log 1 + e−τwp (si −β)

i=1

(9)

KP (P −1)

+

X

h

log 1 + e

τwp (swp
i −α)

i

i=1
P 2 K(K−1)

+

X

h

log 1 + e

i=1

τn (sn
i −β)


i


where τsp , τwp and τn are the temperature factors. Besides,
we compute the cross entropy loss Lid on image and text features to classify them by identity. This loss forces the network
to focus on the identity information, enabling it to correctly
recognize pedestrians of the same identity while distinguishing between those of different identities.

3.4

Optimization and Inference

The optimization of CoRL involves two stages. In the first
stage, we train a virtual-text embedding synthesizer, where
CLIP’s visual and textual encoders are frozen and only a fully
connected layer is optimized via Ls1
V ES to convert image embeddings to virtual-text tokens. We pre-extract and save features from the dataset, allowing us to focus on optimizing this
layer, reducing computational cost. In the second stage, VES
is frozen, and the backbone network and Adapter are optimized. The objective of the second stage is as follows:
s2
Ls2 = LIBM + Lid + Ls2
(10)
align + LCRCL
During inference, both visual and virtual-text features of
gallery images are extracted, and calculate their similarities
to the query text features. The final retrieval score is obtained
by summing these similarities without weighting.

4

Experiments

4.1

Experiment Settings

40,206 images and 80,412 descriptions of 13,003 people.
Each image has 2 descriptions, averaging 23 words. The
dataset is split into 34,054 images for training, 3,078 for validation, and 3,074 for testing. ICFG-PEDES [Ding et al.,
2021] consists of 54,522 image-text pairs of 4,102 persons,
with descriptions averaging 37 words. Training uses 34,674
pairs from 3,102 people, with the remaining 1,000 people reserved for evaluation. RSTPReid [Zhu et al., 2021] includes
20,505 images of 4,101 people, each with 2 descriptions averaging 23 words. Training includes 3,701 people, while validation and testing include 200 people each. Performance is
evaluated using Rank-k accuracy (R@k, k=1, 5, 10).
Implementation Details: Images are resized to 384×128
and augmented with random horizontal flipping, cropping
with padding, and random erasing. The maximum length of
the text sequence is set to 77, and random masking is employed for text augmentation. We use CLIP-ViT-B/16 as the
backbone. Temperature factors are set to τa = 0.02, τsp = 10,
τwp = 5, and τn = 40. Loss weight λ1 is 0.1, and the boundaries α and β in IBM loss are 0.6 and 0.4. Each mini-batch
comprises B = P × K images, , with P = 32 identities and
K = 4 images per identity. In the first stage, only a fully connected layer is optimized for 60 epochs using a cosine learning rate schedule, starting at 1×10−4 . In the second stage, we
fine-tune the visual/textual backbones with an initial learning
rate of 1 × 10−5 and the Adapter with 5 × 10−5 , also using a
cosine schedule and trained for 60 epochs. Both stages adopt
the Adam optimizer with a linear warm-up over the first 5
epochs. Experiments are implemented using the PyTorch library on a single NVIDIA RTX 3090 (24GB) GPU.

Datasets and Metrics: The evaluations are conducted on
three TIPR datasets. CUHK-PEDES [Li et al., 2017] has

2156

4.2

Comparisons with State-of-the-art Models

Table 1 compares our CoRL with current state-of-the-art
methods across three TIPR benchmarks: CUHK-PEDES,
ICFG-PEDES, and RSTPReid. Our CoRL achieves leading performance on these benchmarks, underscoring its effectiveness and advantages. On the CUHK-PEDES dataset,
CoRL achieves an R@1 accuracy of 78.15%, surpassing the
second-best AUL method by 0.92%. For the ICFG-PEDES
dataset, our CoRL sets a new state-of-the-art with R@1 and
R@5 accuracies of 69.50% and 85.63%, respectively. On
the RSTPReid dataset, CoRL delivers impressive results with
69.10%, 87.30%, and 92.90% on R@1, R@5, and R@10.
These results highlight the robustness and versatility of CoRL
across different scenarios. The superior performance of our
method is attributed to its innovative approach in leveraging
cross-modal complementarity and establishing precise crossmodal correspondences. CoRL effectively integrates textual information into images to enhance their discriminability
and bridge the modality gap, and employs identity-bounded
matching loss to address noisy correspondences among different types of image-text pairs.

4.3

Ablation Studies and Analysis

Effectiveness of different components: We conduct an ablation study to assess the effectiveness of various components on CUHK-PEDES in Table 2. 0# represents the result
of Baseline, which involves only the backbone network and
is trained using SDM and cross-entropy losses. 5# denotes


exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:8

Yin et al.

where 𝜎 is the sparsity threshold. 𝜎 is assigned the value 1/𝑁 , where 𝑁 corresponds to the total
count of patches in the image. This ensures that each token has a minimum of one corresponding
image patch for alignment. We compute the aggregation weights by:
𝑠˜𝑖,𝑗
𝑎𝑔𝑔𝑖,𝑗 = Í𝑀

(7)

,

𝑚=1 𝑠˜𝑖,𝑗

where 𝑀 is the number of image patches retained with high similarity to the token 𝑖, and 𝑎𝑔𝑔𝑖,𝑗
quantifies the influence of patch 𝑗 in forming the language-grouped vision embedding (referred to as
joint embedding) associated with token 𝑖. This explicit aggregation strategy ensures a comprehensive
interaction between token 𝑖 and its corresponding patch 𝑗 during local alignment. In particular, the
aggregation weight 𝑎𝑔𝑔𝑖,𝑗 effectively captures the semantic relevance between token 𝑖 and patch 𝑗,
thereby facilitating precise alignment.
Next, we derive the corresponding joint embedding 𝑒𝑖 as:
𝑒𝑖 =

𝑁
∑︁

𝑎𝑔𝑔𝑖,𝑗 · 𝑓 𝑗𝑣 ,

(8)

𝑗=1

where 𝑁 is the count of image patches. The resulting set of joint embedding 𝑒𝑖 has the same length
𝐿 as the text token 𝑓𝑖𝑡 .
𝐿
Hard Coding Alignment. We calculate the similarity between the joint embeddings {𝑒𝑖 }𝑖=1
𝑡
𝐿
𝑣
𝑁
and their corresponding original text embeddings {𝑓𝑖 }𝑖=1 as well as image embeddings {𝑓𝑖 }𝑖=1 ,
respectively. To reduce both computational and memory costs, we adopt a hard coding similarity
computation between the joint embeddings and their corresponding text and image embeddings,
and the theoretical analysis of the hard coding is provided in Appendix A. For simplicity, we only
present the calculation between the joint embeddings and the text embeddings, while the remaining
computations follow a similar and symmetric approach.
For the text 𝑇 and its corresponding joint embedding 𝐸, we calculate the original similarity matrix
𝐿 and their joint embeddings {𝑒 }𝐿 , where 𝑜
𝑡 ⊤
𝑡
𝑂 between all text tokens {𝑓𝑖𝑡 }𝑖=1
𝑖 𝑖=1
𝑖,𝑗 = 𝑓𝑖 𝑒 𝑗 /∥ 𝑓𝑖 ∥ ∥𝑒 𝑗 ∥
𝑡
𝑡
means the cosine similarity of 𝑓𝑖 and 𝑒 𝑗 . For the token 𝑓𝑖 , we compute the weight factor between
it and all joint embeddings using the following hard coding way:


(𝑜𝑖,𝑗 ′ )
 1 if 𝑗 = argmax

𝑗 ′ =1···𝐿
𝜔𝑖,𝑗 =
.

 0 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒


(9)

Then, we utilize the LSE pooling [23] to compute the hard similarity between text 𝑇 and its
corresponding joint embedding 𝐸 by:
!
𝐿
∑︁
ℎ𝑎𝑟𝑑_𝑠 (𝑇 , 𝐸) = 𝐿𝑆𝐸 − 𝑃𝑜𝑜𝑙𝑖𝑛𝑔
𝜔𝑖,𝑗 𝑜𝑖,𝑗
𝑗=1
𝐿
∑︁

(10)




1
= log
exp 𝜆 max 𝑜𝑖,𝑗 ,
𝑗=1···𝐿
𝜆
𝑖=1
where 𝜆 controls the degree to which the most relevant text embeddings and their corresponding
joint embeddings are emphasized.
Given a batch containing 𝐵 text embeddings along with their associated joint embeddings, we
compute the hard coding similarity matrix 𝐻𝑎𝑟𝑑_𝑆 following Eq. (9) and Eq. (10), as illustrated in
Figure 3b. We calculate the EFA loss from the text to its joint embedding, adapted from the triplet
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

Cross-modal Full-mode Fine-grained Alignment for Text-to-Image Person Retrieval

1:9

ranking loss [39]:


∑︁
𝐻𝑎𝑟𝑑_𝑆 neg − 𝐻𝑎𝑟𝑑_𝑆 pos + 𝑚𝑎𝑟𝑔𝑖𝑛
1
L𝑡 2𝑒 = log
,
exp
𝐵
𝜏2
neg

(11)

where 𝜏2 is a scaling factor adjusting the spread of the loss, and 𝑚𝑎𝑟𝑔𝑖𝑛 is a distance hyperparameter
defining the minimal gap separating positive and negative pairs.
Similarly, the EFA loss from the joint embedding to its original text can be computed following
Eq. (11), and we can calculate the EFA loss between image and its joint embedding through Eq. (9),
Eq. (10) and Eq. (11). Then, we obtain a full EFA loss by:
L𝑒 𝑓 𝑎 = L𝑡 2𝑒 + L𝑒2𝑡 + L𝑖2𝑒 + L𝑒2𝑖 .
3.4

(12)

Training Objective

As mentioned, FMFA aims to improve both the global and local cross-modal alignment of image-text
features within the shared embedding space. To realize this goal, the widely adopted ID loss [59]
and IRR loss [21], together with the proposed EFA and A-SDM loss, are jointly utilized to train
FMFA. The ID loss directly classifies the global features obtained from both the image and the text
according to their identities, thereby enhancing the global alignment of the model. The IRR loss,
based on the Masked Language Modeling (MLM) task [46], leverages an attention mechanism for
implicit cross-modal interaction to obtain a joint embedding, and then predicts the [MASK] text
token to enhance the local alignment of the model.
FMFA is trained end-to-end, with the complete training objective formulated as:
L = L𝑖𝑑 + L𝑖𝑟𝑟 + L𝑒 𝑓 𝑎 + L𝐴−𝑠𝑑𝑚 .
4
4.1

(13)

Experiments
Datasets and Settings

Datasets. We assess FMFA on three widely used text-based person retrieval datasets, following
the data splits introduced in IRRA [21]. CUHK-PEDES [27] contains 40,206 images associated with
13,003 identities, where each image is paired with two textual descriptions. Of these identities,
11,003 are designated for training, while the remaining 1,000 identities are allocated separately to
validation and test sets. ICFG-PEDES [10] includes 54,522 images belonging to 4,102 individuals,
each image linked to a single sentence. The conventional setup utilizes 3,102 identities for training
and reserves 1,000 identities for testing. RSTPReid [60] comprises 20,505 images from 4,101 identities
captured across 15 camera views. Every identity corresponds to five images taken from different
viewpoints, and each image is annotated with two descriptive captions. The dataset follows a split
with 3,701 identities for training and 200 identities each for validation and testing.
Evaluation Metrics. To gauge retrieval quality, we primarily report Rank-K results (K = 1, 5,
10), which measure how often the correct item appears within the top-K predictions. Additionally,
mean Average Precision (mAP) is adopted to summarize ranking accuracy over all query outcomes.
In both cases, higher metric values correspond to superior model behavior.
Implementation Details. We utilize either the original CLIP model [37] or its ReID-domain
pre-trained variants [22, 45] as encoders tailored to each modality. To maintain consistency, we
employ the identical CLIP-ViT-B/16 model for visual encoding and Xformer for text encoding,
following the setup used in IRRA [21] for our experiments. Specifically, images are resized to
384×128 pixels, and the maximum sequence length 𝐿 for input word tokens is set to 77. The model
is trained using the Adam optimizer for 60 epochs with a default cosine learning rate decay schedule,
in contrast to the 100 epochs employed for the ICFG-PEDES dataset. The original CLIP model
parameters are trained with an initial learning rate of 1𝑒 − 5 and a batch size of 64. In particular,
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

1:10

Yin et al.

Table 1. The margins utilized in the EFA loss. “T. to E.” means the EFA loss from textual embeddings to
the corresponding joint embeddings, and “V. to E.” means the EFA loss from visual embeddings to the
corresponding joint embeddings.
CUHK-PEDES

ICFG-PEDES

RSTPReid

0.1
0.1

0.2
1.0

0.2
0.8

T. to E.
V. to E.

Table 2. The hardware configuration of our experimental environment.
Hardware

Details

CPU
GPU
RAM

Intel Xeon Gold 6330
NVIDIA RTX A6000
755 GB DDR4

the temperature 𝜏1 in the A-SDM loss is set to 0.02, while the temperature 𝜏2 in the EFA loss takes a
value of 1.0. The weight factor 𝛼 of A-SDM is set to 10.0 by default, and set to 1 in the RSTPReid
dataset, and the factor 𝜆 in the LSE pooling is set to 1.0. Due to variations in data distribution, the
margins used in the EFA loss differ across the three datasets. The specific margins used in Eq. (11)
for each dataset are provided in Table 1. When using ReID-domain pre-trained CLIP models, we
adopt the same initial learning rate and batch size as in NAM [45] and HAM [22], while keeping all
other settings unchanged. The hardware configuration used in our experiments is shown in Table 2,
while the detailed software environment is supplied in the code repository we have released.
4.2

Comparison with State-of-the-Art Methods

In this subsection, we provide a comparison with current state-of-the-art methods (e.g., NAM [45]
and HAM [22]) on three public benchmark datasets. The methods are grouped into two types
according to their underlying network architecture, as listed in Table 3, Table 4, and Table 5: those
using VL-Backbones without ReID-domain pre-training and those incorporating ReID-domain
pre-training. Furthermore, according to whether local features are utilized during inference, the
baselines are further classified into local and global matching methods (denoted as “L"” and “G” in the
“Type” column, respectively). It should be noted that the baseline model presented in Table 3, Table 4,
and Table 5 is referred to as IRRA𝑅 , which represents the performance of our reimplementation of
the IRRA model. CLIP means the ViT-B/16 architecture after fine-tuning under the InfoNCE loss
[34].
Evaluation Results on CUHK-PEDES We measure the performance of FMFA on the CUHKPEDES dataset, as presented in Table 3. When using the VL-Backbones without ReID-domain pretraining, FMFA achieves superior performance over advanced global matching methods, attaining
74.16% Rank-1 and 66.66% mAP, while surpassing IRRA by 0.74% in Rank-5 and 0.41% in Rank-10.
When adopting the VL-Backbones with ReID-domain pre-training, FMFA maintains its superiority,
and achieves Rank-5 accuracy exceeding 95% with the HAM-based backbone. Notably, FMFA
with NAM-based backbone attains 91.33% in Rank-5, outperforming IRRA with the HAM-based
backbone by 0.13%.
Evaluation Results on RSTPReid. We assess FMFA on the latest RSTPReid benchmark, as
presented in Table 4. Using the VL-Backbones without ReID-domain pre-training, FMFA achieves
competitive performance, attaining 61.05% Rank-1, 83.85% Rank-5, 89.80% Rank-10, and 48.22%
mAP, respectively, outperforming IRRA by 1.55% in Rank-1 and 2.05% in Rank-5. When adopting
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

Cross-modal Full-mode Fine-grained Alignment for Text-to-Image Person Retrieval

1:11

Table 3. Comparisons with state-of-the-art methods on the CUHK-PEDES dataset. “G”, “L” and “P” in the
“Type” column stand for global-matching method, local-matching method and pre-trained model with ReIDdomain respectively. “Image Enc.” and “Text Enc.” mean the backbone of image encoder and text encoder
respectively. “IRRA𝑅 ” means the model that we reproduce.
Type Method
Ref.
VL-Backbones w/o ReID-domain pre-training:
LGUR [42]
MM22
IVT [43]
ECCV22
VGSG [17]
TIP23
G
CLIP [37]
ICML21
DM-Adapter [29]
AAAI25
IRRA𝑅 [21]
CVPR23
TBPS-CLIP [4]
AAAI24
FMFA (ours)
ACSA [20]
TMM22
Han et al. [13]
arXiv21
PLOT [35]
ECCV24
L
RaSa [2]
IJCAI23
PTMI [32]
TIFS25
APTM [55]
MM23
SCVD [51]
TCSVT24
VL-Backbones with ReID-domain pre-training:
UniPT [41] + IRRA [21]
ICCV23
PLIP [61] + IRRA [21]
NeurIPS24
NAM [45] + IRRA𝑅
CVPR24
P+G
NAM [45] + FMFA (ours)
HAM [22] + IRRA𝑅
CVPR25
HAM [22] + FMFA (ours)

Image Enc.

Text Enc.

Rank-1

Rank-5

Rank-10

mAP

ResNet50
ViT-B/16
ResNet50
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
Swin-B
CLIP-RN101
CLIP-ViT
CLIP-ViT
CLIP-ViT
Swin-B
CLIP-RN50

BERT
BERT
Transformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
BERT
CLIP-Xformer
CLIP-Xformer
BERT-base
CLIP-Xformer
BERT-base
CLIP-Xformer

65.25
65.59
67.52
68.19
72.17
73.45
73.54
74.16
63.56
64.08
75.28
76.51
76.02
76.53
76.72

83.12
83.11
84.37
86.47
88.74
89.38
88.19
90.12
81.49
81.73
90.42
90.29
89.93
90.04
90.38

89.00
89.21
90.26
91.47
92.85
93.69
92.35
94.10
87.70
88.19
94.12
94.25
94.14
94.15
94.89

61.12
64.33
66.13
65.38
66.66
60.08
69.38
70.85
66.91
-

CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT

CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer

74.37
74.25
76.67
77.23
77.32
77.46

89.51
89.49
91.11
91.33
91.20
91.36

93.97
93.68
94.60
94.75
94.95
95.01

66.60
66.52
68.42
68.53
68.87
68.89

Table 4. Comparisons with state-of-the-art methods on the RSTPReid dataset.
Type Method
Rank-1 Rank-5
VL-Backbones w/o ReID-domain pre-training:
DSSL [60]
39.05
62.60
IVT [43]
46.70
70.00
CLIP [37]
54.05
80.70
G
IRRA𝑅 [21]
59.50
81.80
DM-Adapter [29]
60.00
82.10
FMFA (ours)
61.05
83.85
ACSA [32]
48.40
71.85
CFine [53]
50.55
72.50
L
PLOT [35]
61.80
82.85
RaSa [2]
66.90
86.50
APTM [55]
67.50
85.70
VL-Backbones with ReID-domain pre-training:
UniPT [41] + IRRA [21]
62.20
83.30
PLIP [61] + IRRA [21]
64.35
83.75
NAM [45] + IRRA𝑅
68.25
86.75
P+G
NAM [45] + FMFA (ours)
68.70
87.05
HAM [22] + IRRA𝑅
71.35
87.60
HAM [22] + FMFA (ours)
71.80
88.05

Rank-10

mAP

73.95
78.80
88.00
88.85
87.90
89.80
81.45
81.60
89.45
91.35
91.45

43.41
47.44
47.37
48.22
52.31
52.56

89.75
91.00
92.30
92.35
93.05
93.15

48.33
50.93
52.92
53.14
55.40
55.72

the VL-Backbones with ReID-domain pre-training, our method achieves further gains, exceeding
IRRA by 0.45% in Rank-1 with both the NAM-based and HAM-based backbones. Notably, FMFA
achieves Rank-5 accuracy higher than 88% with the HAM-based backbone.
Evaluation Results on ICFG-PEDES. We assess FMFA on the ICFG-PEDES benchmark, with the
results displayed in Table 5. Using VL-Backbones without ReID-domain pre-training, FMFA obtains
the leading results across all metrics, attaining 64.29% Rank-1 and 39.43% mAP. Compared with
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

1:12

Yin et al.

Table 5. Comparisons with state-of-the-art methods on the ICFG-PEDES dataset.
Type Method
Rank-1 Rank-5
VL-Backbones w/o ReID-domain pre-training:
Dual Path [59]
38.99
59.44
IVT [43]
56.04
73.60
CLIP [37]
56.74
75.72
G
VGSG [17]
60.34
76.01
DM-Adapter [29]
62.64
79.53
IRRA𝑅 [21]
63.48
80.16
FMFA (ours)
64.29
80.48
SSAN [10]
54.23
72.63
ISANet [54]
57.73
75.42
L
CFine [53]
60.83
76.55
RaSa [2]
65.28
80.40
PLOT [35]
65.76
81.39
VL-Backbones with ReID-domain pre-training:
UniPT [41] + IRRA [21]
64.50
80.24
PLIP [61] + IRRA [21]
65.79
81.94
NAM [45] + IRRA𝑅
66.34
81.94
P+G
NAM [45] + FMFA (ours)
66.58
81.94
HAM [22] + IRRA𝑅
68.21
83.28
HAM [22] + FMFA (ours)
68.37
83.28

Rank-10

mAP

68.41
80.22
82.26
82.01
85.32
85.78
85.93
79.53
81.72
82.42
85.12
86.73

31.84
36.50
38.20
39.43
41.29

85.74
87.32
86.73
87.04
88.04
88.10

38.22
39.43
40.14
40.17
41.72
41.76

IRRA, FMFA shows a notable improvement of 0.81% Rank-1 and 1.23% mAP, which is meaningful
for practical applications. When adopting VL-Backbones with ReID-domain pre-training, FMFA
yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
HAM-based backbones, respectively.
In conclusion, FMFA attains the highest performance across all evaluation metrics on the three
widely used public benchmarks. As far as we are aware, FMFA is the best method for all global
matching methods. This highlights the ability of our method to generalize well and maintain
robustness.
4.3

Ablation Study

In this subsection, we examine our proposed components in the FMFA framework. For simplicity,
we omit the components of L𝑖𝑑 and the IRR module that were proposed by IRRA and used in all
experiments. Only one of SDM and A-SDM can be used at the same time.
To thoroughly assess the contribution of our FMFA modules, we undertake an empirical analysis
on three widely used datasets. Table 6 summarizes the Rank-1/5/10 accuracies (%) together with
the mAP (%) performance.
Effect of The A-SDM Module. To evaluate the contribution of the Adaptive Similarity Distribution Matching (A-SDM) module, we perform ablation experiments by replacing the A-SDM
module with the SDM module, keeping all hyperparameters unchanged. Specifically, as shown in
Table 6, replacing A-SDM with SDM results in a reduction of Rank-1 accuracy by 0.59%, 0.78%, and
0.75% across the three datasets, and also causes a 1.19% drop in mAP on the ICFG-PEDES dataset, as
observed in No. 0 vs. No. 1. Additionally, all evaluation metrics on CUHK-PEDES and ICFG-PEDES
degrade, further confirming the superiority of A-SDM. Moreover, when combined with the EFA
module, the advantage of A-SDM becomes even more pronounced. As shown in No. 2 vs. No. 3,
replacing the A-SDM module with the SDM module results in 0.43% and 0.48% decrease in Rank-1
and Rank-5 on the CUHK-PEDES dataset, respectively, as well as a 1.55% drop in Rank-5 and a
0.58% decline in mAP on the RSTPReid dataset. These results collectively validate the consistent
and significant impact of A-SDM to performance.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.


exec
/bin/zsh -lc "pdftotext -f 8 -l 11 'CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
85:8

D. Mao et al.

Fig. 3. Overview of the proposed SFM.

cross-modal attention map is computed to measure the semantic relevance between the textual
queries and spatial locations in the image. This attention mechanism is particularly effective when
the textual features contain spatial cues, allowing the model to highlight regions in the image that
are semantically aligned with the description. The computation of the cross-modal attention map 𝐴
can be formulated as:


𝑄 · 𝐾𝑇
𝐴 =𝜎 √
+𝑏 ,
(3)
𝐶
where 𝑄 denotes the query vectors derived from the textual features 𝑡 and the learnable projection
matrix 𝑊𝑞 , 𝐾 represents the key vectors obtained from the normalized image patch representations,
√
𝜎 denotes the Sigmoid activation function. The attention computation includes a scaling factor 𝐶
to stabilize training, and a learnable bias term 𝑏 is added to enhance the flexibility of the attention
learning.
The obtained attention map 𝐴 are used as gating coefficients to modulate the contribution of
each image patch, resulting in the gated image representation 𝑋𝑔𝑎𝑡𝑒𝑑 :
𝑋𝑔𝑎𝑡𝑒𝑑 = 𝑋𝑝𝑎𝑡𝑐ℎ

𝐴𝑇 ,

(4)

refers to the element-wise multiplication, 𝐴𝑇 is the transposed matrix of attention map 𝐴.

where
Meanwhile, to further refine the image features, a text-guided dynamic channel weighting
mechanism is introduced. The text feature is mapped to a dynamic channel weight vector w𝑐 via a
network consisting of two fully connected layers and a ReLU activation, followed by a Sigmoid
function to constrain the output within the range [0, 1]:
w𝑐 = 𝜎 (𝑊2 𝑅𝑒𝐿𝑈 (𝑊1𝑡 + 𝑏 1 ) + 𝑏 2 ),

(5)

where 𝑊1 , 𝑊2 and 𝑏 1 , 𝑏 2 are the learnable weights and biases of the two layers, 𝜎 denotes the
Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
features in a fine-grained manner.
Finally, the generated weight vector w𝑐 is then applied to the gated image tokens 𝑋𝑔𝑎𝑡𝑒𝑑 via
element-wise multiplication. To preserve representational consistency, the re-weighted features are
fused with the original input using a residual connection, followed by a linear transformation to
project them back to the original token space. The final semantically filtered image representation
𝑜𝑢𝑡 can be formulated as:
𝑋𝑆𝐹
𝑀

𝑜𝑢𝑡
𝑋𝑆𝐹
w𝑐 ,
(6)
𝑀 = 𝑋𝑔𝑎𝑡𝑒𝑑 + 𝐹𝐶 𝑋𝑔𝑎𝑡𝑒𝑑
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

CVAF for Aerial-Ground Person Re-Identification
where FC denotes the fully connected layer,
3.5

85:9

represents the multiplication by elements.

Optimization

Overall Loss. In addition to the proposed TCL, we retain the Supervised Contrastive Loss adopted in
the first stage of CLIP-ReID [10]. It is worth noting that CLIP-ReID introduces two complementary
losses: the image-to-text loss L𝑖2𝑡 and the text-to-image loss L𝑡 2𝑖 , which enforce bidirectional
image-text alignment. Both losses share the same supervised contrastive form:

𝑁
exp 𝑎𝑖>𝑏𝑖 + /𝜏
1 Õ
L (𝐴, 𝐵) = −
log Í𝑀
(7)
,
>
𝑁 𝑖=1
𝑘=1 exp 𝑎𝑖 𝑏𝑘 /𝜏
where 𝑎𝑖 is a feature from set 𝐴, 𝑏𝑖 + denotes the positive sample in set 𝐵 with the same identity,
and 𝜏 is the temperature. Specifically, L𝑖2𝑡 = L (image, text) and L𝑡 2𝑖 = L (text, image). These two
losses are essential for strengthening positive pairs. By explicitly encouraging features of the same
identity across different modalities to be close in the embedding space, they ensure that positive
image-text pairs achieve high similarity. Without these supervised contrastive objectives, TCL
alone is insufficient to enforce sufficient alignment for cross-modal positive pairs, especially when
the intra-class variance is large. Furthermore, we incorporate the text-to-image cross-entropy
alignment loss L𝑖2𝑡𝑐𝑒 , together with the commonly used identity classification loss L𝐶𝐸 and triplet
loss L𝑇 𝑟𝑖 . The overall optimization objectives of CVAF can thus be summarized as:
L𝑠𝑡𝑎𝑔𝑒1 = L𝑖2𝑡 + L𝑡 2𝑖 + L𝑡𝑐𝑙 ,

(8)

L𝑠𝑡𝑎𝑔𝑒2 = L𝐶𝐸 + L𝑇 𝑟𝑖 + L𝑖2𝑡𝑐𝑒 .

(9)

4 Experiment
4.1 Dataset and Evaluation Metrics
We evaluate the proposed CVAF on three benchmark AGPReID datasets, including the existing
AG-ReID [18], AG-ReID.v2 [19], and CARGO [40], following standard train/test splits.
AG-ReID is the first real-world ground-to-aerial person ReID dataset, collected in outdoor environments. It contains 21,983 images of 388 identities captured by two cameras from distinct
viewpoints. We follow the standard two evaluation protocols: 𝐴 → 𝐺 and 𝐺 → 𝐴.
AG-ReID.v2 extends AG-ReID by incorporating more diverse camera sources, including static
ground cameras, drones, and wearable devices. It contains a total of 100,502 images from 1,615
identities. The evaluation is conducted under four view-shift scenarios: 𝐴 → 𝐶, 𝐶 → 𝐴, 𝐴 → 𝑊 ,
and 𝑊 → 𝐴, where A, C, and W refer to aerial, CCTV, and wearable views, respectively.
CARGO is a large-scale synthetic AGPReID dataset created using MakeHuman [1], Unity3D, and
other 3D modeling tools. It includes 108,563 images of 5,000 synthetic identities, captured by eight
ground-level and five aerial cameras. Two evaluation protocols are adopted: 𝐴𝐿𝐿, using the full
dataset, and 𝐴 ↔ 𝐺, focusing on cross-view matching between aerial and ground cameras.
Evaluation Metrics. We adopt Rank-1 accuracy and Mean Average Precision (mAP) as the
standard evaluation metrics across all datasets and protocols.
4.2

Implementation Details

Our method is implemented using PyTorch, and all experiments are conducted on a single NVIDIA
Tesla A40 GPU. In the first training stage, we adopt the text encoder from CLIP as the backbone
for extracting textual features, and the number of learnable text tokens 𝑀 is set to 4. The model is
trained for 120 epochs with a batch size of 64 using the Adam optimizer. The initial learning rate is
set to 3.5 × 10e-4 and decayed using a cosine annealing schedule. In the second stage, we utilize
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

85:10

D. Mao et al.
Table 1. Performance Comparison on AG-ReID and AG-ReID.v2 Dataset
AG-ReID
Method

A→G

AG-ReID.v2
G→A

A→C

C→A

A→W

W→A

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

TransReID [7]
CLIP-ReID [10]

81.28
83.72
85.08

72.38
74.14
78.84

82.64
85.95
86.82

73.35
75.45
81.00

85.40
85.82
88.10

77.03
79.50
81.91

84.65
85.87
86.70

75.90
78.10
81.15

89.77
90.02
91.83

80.48
82.81
86.22

84.65
85.77
90.16

75.90
79.13
83.42

AG-ReID [18]a
AG-ReID.v2 [19]a
VDT [40]a
SeCap [27]a

81.47
–
82.91
84.03

72.61
–
74.44
76.16

82.85
–
86.59
87.01

73.39
–
78.57
78.34

87.70
88.77
86.46
88.12

79.00
80.72
79.13
80.84

87.35
87.86
86.14
88.24

72.84
78.51
78.12
79.99

93.67
93.62
90.00
91.44

83.14
84.85
82.21
84.01

87.73
88.61
85.26
87.56

79.08
80.11
78.52
80.15

Ours

87.23

79.45

90.12

82.59

88.77

83.35

88.40

83.17

92.49

87.81

90.85

85.08

BoT(ViT) [15]a

“A → G” refers to using the aerial perspective images as the query, and G refers to the ground perspective images. “C” in “A → C” represents
CCTV, and “W” in “A → W” represents wearable camera. The best performance are represented in bold.
a Performance results that are cited from the original paper SeCap [27].

the ViT-B/16 image encoder from CLIP as the visual backbone. The training lasts for 60 epochs,
with the input image resized to 256 × 128. Standard data augmentation techniques are applied,
including random cropping, color jittering, random erasing, and so on. The batch size remains
64, and we sample four images per identity within each batch. The optimizer is Adam, with an
initial learning rate of 5 × 10e-6, which is decayed at the 30th and 50th epochs. We employ the
cross-entropy loss and soft version of triplet loss [9] for supervision. During inference, no data
augmentation or re-ranking strategies are applied.
4.3

Comparison with SoTA Methods

As shown in Tables 1 and 2, we conduct comprehensive comparisons on the AG-ReID, AG-ReID.v2,
and CARGO datasets against both conventional ReID baselines based on Vision Transformers (BoT
[15], CLIP-ReID [10], and TransReID [7]) and state-of-the-art AGPReID approaches (AG-ReID [18],
AG-ReID.v2 [19], VDT [40], and SeCap [27]). Compared to traditional ViT-based ReID methods, our
method exhibits significantly improved cross-view matching capability, consistently outperforming
CLIP-ReID under various evaluation settings. Furthermore, CVAF achieves superior performance
over the latest AGPReID models, including AG-ReID.v2, VDT, and SeCap, across all evaluation
protocols. These results clearly demonstrate the effectiveness of CVAF in modeling cross-view
consistency and extracting robust view-invariant features.
The performance improvement mainly benefits from the fact that, the TCL effectively reinforces
view-invariant textual representations for the same identity, thereby providing stable semantic
guidance that enables the SFM to focus on consistent discriminative regions and suppress viewspecific noise.
4.4

Ablation Study

To comprehensively validate the effectiveness and generalization of each component in CVAF, we
conduct ablation studies on both the AG-ReID and CARGO datasets using two different backbones:
CLIP-ViT B/16 and CLIP-ResNet50, as presented in Tables 3 and 4. Comparing the “+TCL” setting
with the baseline shows that introducing the TCL consistently improves cross-view matching
performance across both architectures. This indicates that TCL effectively enhances cross-view
semantic alignment between image and text features. Moreover, Adding the SFM alone brings more
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

CVAF for Aerial-Ground Person Re-Identification

85:11

Table 2. Performance Comparison on the
CARGO Dataset
CARGO
Method

ALL

𝐴↔𝐺

R-1

mAP

R-1

mAP

CLIP-ReID [10]

61.54
70.12

53.54
64.07

43.13
63.75

40.11
56.54

VDT [40]a
SeCap [27]a

64.10
68.59

55.20
60.19

48.12
69.43

42.76
58.94

Ours

73.15

66.65

68.00

60.26

BoT(ViT) [15]a

“ALL” denotes the overall retrieval performance of each
method. “𝐴 ↔ 𝐺 ” represents the performance of each model
in a specific retrieval pattern. The best performance are represented in bold.
a Performance results that are cited from the original paper
SeCap [27].

Table 3. The Efficacy of Components in CVAF Is Evaluated on the AG-ReID Datasets
AG-ReID (A → G)

AG-ReID (G → A)

R-1

mAP

R-1

mAP

Baseline (CLIP-ViT-B/16)
+ TCL
+ SFM
+ TCL + SFM (Ours)

85.08
85.94 (+0.86)
86.76 (+1.68)
87.23 (+2.15)

78.84
79.02 (+0.18)
79.43 (+0.59)
79.45 (+0.61)

86.82
87.36 (+0.74)
89.43 (+2.81)
90.12 (+3.50)

81.00
81.63 (+0.63)
82.18 (+1.18)
82.59 (+1.59)

Baseline (CLIP-ResNet50)
+ TCL
+ SFM
+ TCL + SFM (Ours)

74.81
75.62 (+0.81)
77.02 (+2.21)
77.62 (+2.81)

64.18
64.93 (+0.75)
65.77 (+1.59)
65.99 (+1.81)

74.82
75.97 (+1.15)
77.47 (+2.65)
78.12 (+3.30)

66.11
67.02 (+0.91)
67.47 (+1.36)
68.01 (+1.90)

Settings

The best performance and best improvement are in bold.
SFM, Semantic Filtering Module; TCL, Text Consistency Loss.

notable gains over the baseline, verifying its ability to refine visual representations by emphasizing
identity-relevant regions while suppressing redundant background information. When both TCL
and SFM are jointly applied, the model achieves the best results under all evaluation protocols
and across both backbones. The synergy between TCL and SFM allows the model to learn more
robust, view-invariant, and semantically consistent representations. These results confirm that
the proposed modules are not only effective but also architecture-agnostic, providing consistent
improvements for both Transformer- and CNN-based encoders.
Effect of TCL. The effectiveness of the proposed TCL is demonstrated on both the AG-ReID and
CARGO datasets, as shown in the second row of Tables 3 and 4. By introducing TCL into the baseline
model (CLIP-ReID), consistent performance improvements are observed across both datasets. This
indicates that the learnable text tokens are capable of aggregating multi-view information from
images of the same identity during training, enabling the extraction of the most discriminative,
view-invariant features as textual descriptions. These view-invariant text embeddings then serve as
reliable guidance for the image encoder in subsequent stages. Notably, TCL requires no additional
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.


exec
/bin/zsh -lc "pdftotext -f 3 -l 5 'CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Fig. 2. Overview of the proposed method. Our CSGN includes two stages. Stage 1: Training MDG using RGB maps and gray-scale maps with Meta-Insert skill. Stage 2: Training
VSS to fuse local and global features, and reduce the feature bias by the HSF loss function.

In contrast to the static prompt in CoOp, the dynamic prompt in
CoCoOp is instance-adaptive, making it more robust to class migration.
Adapter (Houlsby et al., 2019) offers a different approach by preserving
the original pre-trained model parameters while introducing a small set
of trainable parameters. During fine-tuning, only these new parameters
are trained, thus avoiding excessive computational costs.
However, in a ReID scenario, all IDs inherently belong to the same
category, making it challenging to generate distinct and appropriate
text descriptions for each image. To address this challenge, we introduce a novel framework named CSGN, which consists of three core
components: MDG, VSS, and HSF. In the first stage, our MDG module
generates two learnable prompt tokens specific to each identity in
the training set. This process incorporates a mutual loss mechanism
between text and image representations, effectively mitigating the
text encoder’s sensitivity to color variations. In the second stage, VSS
enhances the visual feature representation by leveraging the complementary strengths of both ResNet and ViT feature extractors in parallel.
This parallel processing ensures a richer and more nuanced visual understanding. Finally, recognizing the synergy between these two types
of representations, the HSF loss is applied, providing a comprehensive
and constraining force that further strengthens the identity of each
individual. In summary, our CSGN framework offers a holistic approach
to generating discriminative textual and visual representations for ReID
scenarios.

sets {𝑇1 , 𝑇2 , … , 𝑇𝑁 } and {𝐼1 , 𝐼2 , … , 𝐼𝑁 } are one-to-one matched, where
when i = j, 𝑇𝑖 and 𝐼𝑗 form positive sample pairs, and otherwise,
𝑇𝑖 and 𝐼𝑗 are negative sample pairs. Next, we calculate the cosine
similarity between 𝑇𝑖 and 𝐼𝑗 ; the greater the cosine similarity, the
stronger the correspondence between 𝑇𝑖 and 𝐼𝑗 , and vice versa. Finally,
we maximize the cosine similarity for the 𝑁 positive samples and
minimize the cosine similarity for the N(N-1) negative samples by
training the parameters of the Text Encoder and Image Encoder. This
process ensures a one-to-one correspondence between images and texts.
We next briefly review CLIP-ReID (Li et al., 2023b), which consists
of an image encoder I(⋅) and a language encoder T(⋅), both pre-trained
on CLIP. For the image encoder I(⋅), ViT-B/16 or ResNet-50 are commonly used to extract feature vectors from images. For the language
encoder T(⋅), ViT-B/16 is implemented as a Transformer, which generates a representation from a sentence. Specifically, given a description
such as ‘‘A photo of a [class]’’, where [class] is typically replaced by
concrete text labels, T(⋅) first converts each word into a unique numeric
ID using lower-cased byte pair encoding (BPE) with a vocabulary size of
49,152. Each ID is then mapped to a 512-dimensional word embedding.
To enable parallel computation, each text sequence has a fixed length
of 77 tokens, including the start [SOS] and end [EOS] tokens. After
passing through a 12-layer model with 8 attention heads, the [EOS]
token is considered as the feature representation of the text, which
is then layer-normalized and linearly projected into the cross-modal
embedding space.
CLIP-ReID consists of two training phases. In the first phase, both
I(⋅) and T(⋅) are frozen, and the language prompt is optimized by 𝐿𝑖2𝑡
and 𝐿𝑡2𝑖 in Eq. (6). In the second phase, T(⋅) and the language prompt
are fixed, while I(⋅) is trained using 𝐿𝑖𝑑 , 𝐿𝑡𝑟𝑖 , and 𝐿𝑖2𝑡𝑐𝑒 in Eq. (7) to
achieve the best alignment between the visual and text features.

3. The proposed methods
A Preliminaries: Overview of CLIP and CLIP-ReID
We first briefly review CLIP (Radford et al., 2021). The CLIP model
structure consists of two parts: the Text Encoder and the Image Encoder. The Text Encoder is based on a Transformer model, while two
models are used for the Image Encoder: one is ResNet based on CNN,
and the other is ViT based on Transformer. Assuming that a batch in
the dataloader contains 𝑁 (text-image) pairs, we first encode these 𝑁
texts into one-dimensional vectors of length d using the Text Encoder.
This batch of text data is represented as {𝑇1 , 𝑇2 , … , 𝑇𝑁 }. Similarly,
the 𝑁 images are encoded by the Image Encoder as {𝐼1 , 𝐼2 , … , 𝐼𝑁 },
where each image vector has the same length as the text vector. The

𝐿𝑖2𝑡 = −

𝑁
exp(𝑠(𝒇𝑖 , 𝒇𝑖𝑡 ))
1 ∑
log( ∑𝑛
),
𝑏
𝑁 𝑖=1
exp(𝑠(𝒇𝑖 , 𝒇𝑗𝑡 ))
𝑗=1

𝐿𝑡2𝑖 = −

𝑁
exp(𝑠(𝒇𝑝𝑖 , 𝒇𝑦𝑡 ))
∑
1 ∑ 1
𝑖
log( ∑𝑛
),
𝑏
𝑁 𝑖=1 |𝑝(𝑦𝑖 )| 𝑝 ∈𝑝(𝑦 )
exp(𝑠(𝒇
,
𝒇𝑦𝑡 ))
𝑗
𝑗=1
𝑖
𝑖

(1)

(2)

𝑖

𝑛𝑏

𝐿𝑖𝑑 = −
3

1 ∑
𝑞 𝑙𝑜𝑔(𝑾 (𝒇𝑖 )),
𝑛𝑏 𝑖=1 𝑖

(3)

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

𝐿𝑖2𝑡𝑐𝑒 = −

and Convnet-based visual encoder. Among them, the transformer makes
use of the attention mechanism and is more conducive to modeling
global information. As for CNN, convolutional operations are good
at mining local information from images and are suitable for visual
tasks with small datasets such as pedestrian re-identification. Therefore,
in order to enhance the model’s ability to ultimately extract visual
features, we introduced the VSS module that combines the advantages
of transformer and CNN.
Specifically, to obtain global features, we divide images of shape
(B, C, H, W) into patches and project them into tokens of shape (B,
N, L). Then, we use the Multi-Head Self-Attention (MHSA) mechanism (Vaswani, 2017) to build the global information between different
tokens. To obtain local features, we use a series of standard convolution
blocks following the structure of ResNet-50 and reduce the size of the
features by increasing the stride at the end of each layer. As shown
in Fig. 3, for fusing 𝑓𝑖𝑙 and 𝑓𝑖𝑔 , we multiplied 𝑓𝑖𝑙 , 𝑓𝑖𝑔 with 𝑊𝑙 , 𝑊𝑔 and
added together. The integration process can be expressed as follows:

(4)

𝐿𝑡𝑟𝑖 = max(𝑑𝑝 − 𝑑𝑛 + 𝛼, 0),
𝑛𝑏
exp(𝑠(𝒇𝑖 , 𝒇𝑦𝑡 ))
1 ∑
𝑖
𝑞𝑖 log( ∑𝑁
),
𝑐
𝑛𝑏 𝑖=1
exp(𝑠(𝒇
,
𝒇𝑦𝑡 ))
𝑖
𝑦 =1

(5)

𝑎

𝑎

𝐿𝑠𝑡𝑎𝑔𝑒1 = 𝐿𝑡2𝑖 + 𝐿𝑖2𝑡 ,

(6)

𝐿𝑠𝑡𝑎𝑔𝑒2 = 𝐿𝑖𝑑 + 𝐿𝑡𝑟𝑖 + 𝐿𝑖2𝑡𝑐𝑒 ,

(7)

where 𝒇𝑖 𝑡 is extracted by T(⋅), 𝑓𝑖 is extracted by I(⋅), 𝑛𝑏 is for batch size,
N is the total number of batches. s(⋅) represents the similarity between
two vectors; 𝒇𝑦𝑖 𝑡 is the text feature with identity 𝑦𝑖 ; 𝑃 (𝑦𝑖 ) denotes a set
composed of indexes of samples with identity 𝑦𝑖 ; |𝑃 (𝑦𝑖 )| indicates the
cardinality of 𝑃 (𝑦𝑖 ). 𝑞𝑖 is the one-hot vector of identity label 𝑦𝑖 ; 𝑁𝑐 is the
total number of identities; W is identity classifier; 𝑑𝑝 and 𝑑𝑛 are feature
distances of positive pair and negative pair, while 𝛼 is the margin of
𝐿𝑡𝑟𝑖 .

𝑓𝑖𝑙 , 𝑓𝑖𝑔 = (𝑊𝑙 , 𝑊𝑔 ) ⊗ (𝑓𝑖𝑙 , 𝑓𝑖𝑔 ),
B Our proposed CSGN

where 𝑊𝑙 , 𝑊𝑔 are one-dimensional matrices of the same length as
(𝑓𝑖𝑙 , 𝑓𝑖𝑔 ); ⊗ represents the matrix multiplication; 𝑓𝑖𝑙 and 𝑓𝑖𝑔 are features containing rich complementary visual information. For inference,
we deploy a triplet loss to ensure that 𝑓𝑖𝑙 and 𝑓𝑖𝑔 can learn unique
identity-independent features.
(3) HSF: For the multiple language descriptions learned by MDG,
it is crucial to effectively leverage the semantic information. CLIPVIReID (Yu et al., 2024) gives a scheme, in which the learnable
prompt for two specific modalities is designed and the Attention Fusion
Module was proposed to decrease the modal difference. Specifically,
they treat the text features corresponding to the visible image as
‘Key’, and the text features corresponding to the infrared image as
‘Query’ and ‘Value’, by attention mechanism and contrastive losses to
improve the semantic richness of text features. However, for clotheschanging person re-identification, there is no modal difference between
multiple descriptions but only informational difference, e.g., language
descriptions generated from RGB images focus on information such
as clothing, skin color, hair style, etc., whereas language descriptions
generated from gray-scale images focus on information such as body
shape, pose, etc. For this reason, in order to avoid visual features being
biased towards a single text feature, we propose the HSF loss to learn
more comprehensive individual features. As shown in Eq. (11), the first
term will direct the learned features to align with the description of
the color image, while the second term will guide the visual features
to favor the description of the grayscale image. Since the description
of the grayscale image emphasizes body shape information, whereas
the description of the color image focuses on color-related details,
we balance the contributions of both pseudo-text descriptions using a
weighted computational approach. This ensures that the final model
incorporates all relevant discriminative information in the image.
Specifically, as shown in Eq. (11), we employ 𝐿ℎ𝑠𝑓 to constrain the
𝑡
𝑡
visual features using 𝑓𝑖 𝑅 and 𝑓𝑖 𝐺 obtained by MDG. Where the pair
𝑡𝐺
𝑡𝑅
𝑣
𝑣
(𝑓𝑖 , 𝑓𝑖 )’s and (𝑓𝑖 , 𝑓𝑖 )’s distance wound be decreased. 𝑓𝑖𝑣 represent
the visual feature.

For CC-ReID, as time progresses, the change in a person’s clothing
introduces significant challenges to the accuracy of the model. Given
the challenges mentioned above, we propose a CLIP-driven Semantic
Guidance Network (CSGN), which consists of a Multi-Description Generator (MDG), Heterogeneous Semantic Fusion loss (HSF), and Visual
Semantic Steering module (VSS).
(1) MDG: In CSGN, we introduce a learnable prompt for pedestrians
with the same identity. However, unlike other subtasks in person reidentification, CC-ReID is more complex because different individuals
may wear similar clothing, while the same person may wear different
clothing. Specifically, for CLIP, differences in descriptions of a person
are often reflected in changes in their clothing. To address this, we
design the Multi-Description Generator (MDG) to mitigate the effects
of clothing changes by generating diverse descriptions for the same
pedestrian.
Specifically, we assign two learnable language descriptions to each
of the RGB image and its corresponding grayscale image, denoted as
𝑇𝑖𝑅 and 𝑇𝑖𝐺 , respectively. At the start of the first stage, 𝑇𝑖𝑅 and 𝑇𝑖𝐺 are
initialized to ‘‘a photo of [𝑇1 ][𝑇2 ]...[𝑇𝑀 ][𝑐𝑙𝑠] person’’. To differentiate
between language descriptions, we further use a technique called MetaInsert, which generates a conditional token (vector) for each input. This
token is then combined with the text vectors. Specifically, we design
a mini-network consisting of a two-layer bottleneck structure (LinearReLU-Linear) to project image features into the text space. Let ℎ𝜃 ()
denote the mini-network parameterized by 𝜃, so that each text token
is now obtained by 𝑇𝑚 (𝑥) = 𝑇𝑚 (𝑥) + ℎ𝜃 (𝑥). The loss for the first stage is
then reformulated as follows:
𝑡

𝐿1𝑖2𝑡 = −

𝑁
exp(𝑠(𝑓𝑖𝑅 , 𝑓𝑖 𝑅 ))
1 ∑
log( ∑𝑛
)
𝑡
𝑏
𝑁 𝑖=1
exp(𝑠(𝑓𝑖𝑅 , 𝑓𝑗 𝑅 ))
𝑗=1

(8)

𝑡

−

𝑁
exp(𝑠(𝑓𝑖𝐺 , 𝑓𝑖 𝐺 ))
1 ∑
log( ∑𝑛
),
𝑡
𝑏
𝑁 𝑖=1
exp(𝑠(𝑓𝑖𝐺 , 𝑓𝑗 𝐺 ))
𝑗=1
𝑡

𝐿1𝑡2𝑖 = −

𝑁
exp(𝑠(𝑓𝑝𝑅 , 𝑓𝑦𝑅𝑖 ))
∑
1 ∑ 1
𝑖
log( ∑𝑛
)
𝑡𝑅
𝑏
𝑅
𝑁 𝑖=1 |𝑝(𝑦𝑖 )| 𝑝 ∈𝑝(𝑦 )
exp(𝑠(𝑓
𝑗 , 𝑓𝑦𝑖 ))
𝑖
𝑖
𝑗=1
𝑡

𝑁
exp(𝑠(𝑓𝑝𝐺 , 𝑓𝑦𝐺𝑖 ))
∑
1 ∑ 1
𝑖
−
log( ∑𝑛
),
𝑡𝐺
𝐺
𝑏
𝑁 𝑖=1 |𝑝(𝑦𝑖 )| 𝑝 ∈𝑝(𝑦 )
exp(𝑠(𝑓
𝑖
𝑖
𝑗 , 𝑓𝑦𝑖 ))
𝑗=1
𝑡

(10)

𝑡

𝐿2ℎ𝑠𝑓 = −

(9)

𝑛𝑏
exp(𝑠(𝑓𝑖𝑉 , 𝑓𝑦𝑅𝑖 ))
𝛼 ∑
𝑞𝑖 log( ∑𝑁
)
𝑡
𝑐
𝑛𝑏 𝑖=1
exp(𝑠(𝑓𝑖𝑉 , 𝑓𝑦𝑅𝑎 ))
𝑦 =1
𝑎

𝑡

𝑛𝑏
exp(𝑠(𝑓𝑖𝑉 , 𝑓𝑦𝐺𝑖 ))
1−𝛼 ∑
−
𝑞𝑖 log( ∑𝑁
),
𝑡
𝑐
𝑛𝑏 𝑖=1
exp(𝑠(𝑓𝑖𝑉 , 𝑓𝑦𝐺𝑎 ))
𝑦 =1

𝑡

(11)

𝑎

where 𝑓𝑖 𝑅 and 𝑓𝑖 𝐺 denote the text features of 𝑇𝑖𝑅 and 𝑇𝑖𝐺 . 𝑇𝑚 (𝑥)
represent the text token of image 𝑥.
𝑡
𝑡
(2) VSS: It is feasible to utilize 𝑓𝑖 𝑅 and 𝑓𝑖 𝐺 to facilitate 𝑓𝑖𝑣 learning.
However, CLIP’s visual encoder fails to adequately extract visual features of the target identity. The visual encoder currently used in CLIP
can be divided into two categories, Transformer-based visual encoder

1 ∑
𝑞 𝑙𝑜𝑔(𝑊 (𝑓𝑖𝑔 ))
(12)
𝑁 𝑖=1
𝑁 𝑖=1 𝑖
Note that superscript 1 of the formula represents the loss function
used in the first stage of training and superscript 2 represents the loss
function used in the second stage.
𝐿2𝑖𝑑 = −

4

𝑁
1 ∑

𝑁

𝑞𝑖 𝑙𝑜𝑔(𝑊 (𝑓𝑖𝑙 )) −

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Fig. 3. VSS module.
Table 1
Comparison with state-of-the-art methods on Celeb-ReID, Celeb-Light and VC-Clothes
datasets.

C Training and inference
The approach in this paper divides the entire training process into
two stages. In the first stage, under the constraints of Eqs. (8) and (9),
we train the MDG equipped with two learnable prompts for generating
text descriptions corresponding to RGB and grayscale images. Upon
entering the second stage, we input the RGB and gray-scale images
to obtain the pseudo-text generated during the first stage. We then
impose the constraints defined in Eq. (4), Eq. (11), and Eq. (12) to train
the VSS module, which aims to integrate the visual information in the
global and local features to obtain more robust features. The constraints
defined in Eq. (12) can improve the discriminative properties of the
visual representations. It is important to note that, during the training
phase, only RGB images are fed into the VSS module. During the
inference phase, however, only the features extracted by the visual
encoder are used for identity matching through the cosine similarity
metric. The other components are not required, ensuring that the utility
of our proposed CSGN remains unaffected.

Mthod

Venue

dataset
Celeb-ReID

Celeb-Light

VC-Clothes

rank-1 mAP rank-1 mAP rank-1 mAP

4. Experiments

FSAM (Hong et al., 2021)
CVPR21 –
IRANet (Shi et al., 2022)
IVC22
64.1
MBUNet (Zhang et al., 2023)
ICME22 55.3
DeSKPro (Wu et al., 2022)
ICIP22
68.6
CCAT (Ren et al., 2022)
IJCNN22 –
UCAD (Yan et al., 2022)
IJCAI22 –
CC-ReID (Mu et al., 2022)
BMVC22 –
MVSE (Gao et al., 2022)
ACM22 64.5
Co-attention (Wang et al., 2022) ACCV22 57.5
CIHANet (Zhou et al., 2022a)
ICPR22 –
DSA (Chen et al., 2021)
TMM22 –
CCIL (Li et al., 2023a)
Arxiv23 –
DualBCtnet (Li et al., 2022)
TCSVT23 –
ACID (Yang et al., 2023b)
TIP23
52.5
IGCL (Gao et al., 2023)
CVPR23 –

–
–
19
46.2
12.1 33.9
22.7 52
–
–
–
–
–
–
19.2 –
12.3 –
–
47.2
–
–
–
–
–
–
11.4 27.9
–
41

–
25.4
21.3
29.8
–
–
–
–
–
26.8
–
–
–
15.8
23.4

78.6
–
–
–
83.8
92.6
84.5
–
–
–
82
80.4
92.2
84.3
82.9

78.9
–
–
–
–
81.1
84.3
–
–
–
80
–
73.6
74.2
84.8

CSGN(our)

23.3 56.4

38.4 92.7

85.1

64.6

A Experimental settings
padding, and cropping. In the first stage, we train MDG for 120 epochs.
The initial learning rate is set to 3 × 10−4 and decayed according to the
cosine schedule (Li et al., 2023b). Subsequently, in the second stage,
we focus on designing a visual encoder and classifier fusing global and
local features for 60 epochs of training. The learning rate in this phase
increases linearly from 3×10−6 to 3×10−4 during the first 10 epochs and
decays by a factor of 0.1 at the 20th and 40th epochs. The batch size
is set to 64, with 8 RGB images per pedestrian. The training process of
the proposed CSGN uses the Adam optimizer (Kingma and Ba, 2014).

(1)Datasets: As shown in Fig. 4, we conduct experiments on four
public CC-ReID datasets: PRCC, Celeb-ReID, Celeb-light, and
VC-Clothes. PRCC is a rigorous clothes-changing dataset, consisting
of 33,698 images from 221 individuals, captured from 3 different
views. It also provides contour sketch images to facilitate the extraction of silhouette information. Celeb-ReID is one of the first datasets,
comprising 590 individuals and 10,842 images, while Celeb-light is a
lighter version of Celeb-ReID. VC-Clothes is a virtual dataset from the
game GTA-V, containing 512 IDs and 19,060 images. It also includes
a real-world-based test dataset with 4324 images from 28 IDs. All of
these datasets were acquired using multiple cameras over various time
periods and contain multiple composite disturbances, such as lighting
changes, pose variations, occlusions, clothing changes, and more.
(2) Evaluation metrics: In this paper, we primarily use Cumulative
Matching Precision (rank-k) and Mean Average Precision (mAP) to
evaluate the model’s performance. Rank-k represents the probability of
a correct match appearing in the top K results, sorted by confidence.
mAP treats pedestrian re-identification as a retrieval task and calculates
the mean of the average precision scores across all classes in the
dataset.
(3) Implementation details: we implemented the proposed CSGN
using the PyTorch deep learning framework, and all experiments were
deployed on a GTX 4090 GPU. We use CLIP (Radford et al., 2021) as
the backbone, and the input images are uniformly resized to 256 × 128
with common data augmentation strategies such as random flipping,

B Comparison with state-of-the-art methods
In this section, we compare the proposed CSGN with the state-ofthe-art methods across four widely used CC-ReID datasets. The results
are summarized in Tables 1 and 2. The best result is highlighted in bold,
while the second-best result is underlined.
Experiment on Celeb-ReID, Celeb-Light, and VC-Clothes: To evaluate the effectiveness of our approach, we first conducted experiments
on three widely used datasets, Celeb-ReID, Celeb-Light, and VC-Clothes,
and compared our method with the latest studies. As shown in Table
1, the semantic segmentation-based approach achieves satisfactory performance when handling complex clothing transformation scenarios.
For example, DeSKPro (Wu et al., 2022) and IGCL (Gao et al., 2023)
achieve rank-1 accuracies of 68.6% and 52.0%, and mAP of 22.7%
and 29.8% on Celeb-ReID and Celeb-Light, respectively. This significant
5


web search: 
web search: "Cross-modal Collaborative Representation Learning" "Text-to-Image Person Retrieval" CoRL IJCAI 2025
web search: 
web search: "CVAF" "CLIP-Based View-Consistent Alignment Framework" "Aerial-Ground Person Re-Identification"
web search: 
web search: "CVAF" "Aerial-Ground Person Re-Identification" "3785482"
web search: 
web search: "Cross-modality average precision optimization" "Pattern Recognition" "111489"
web search: 
web search: https://doi.org/10.1016/j.patcog.2025.111489
web search: 
web search: 10.1016/j.patcog.2025.111489
web search: 
web search: "CLIP-ReID" "Exploiting Vision-Language Model" person re-identification arxiv
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
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
/bin/zsh -lc "pdftotext -f 12 -l 15 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
135:12

H. Yin et al.
Table 5. Comparisons with State-of-the-Art Methods on the
ICFG-PEDES Dataset
Type Method
Rank-1 Rank-5
VL-Backbones w/o ReID-Domain Pre-Training
DuaL Path [59]
38.99
59.44
IVT [43]
56.04
73.60
CLIP [37]
56.74
75.72
G
VGSG [17]
60.34
76.01
DM-Adapter [29]
62.64
79.53
IRRA𝑅 [21]
63.48
80.16
FMFA (ours)
64.29
80.48
SSAN [10]
54.23
72.63
ISANet [54]
57.73
75.42
L
CFine [53]
60.83
76.55
RaSa [2]
65.28
80.40
PLOT [35]
65.76
81.39
VL-Backbones with ReID-Domain Pre-Training
UniPT [41] + IRRA [21]
64.50
80.24
PLIP [61] + IRRA [21]
65.79
81.94
NAM [45] + IRRA𝑅
66.34
81.94
P+G
NAM [45] + FMFA (ours)
66.58
81.94
HAM [22] + IRRA𝑅
68.21
83.28
HAM [22] + FMFA (ours)
68.37
83.28

Rank-10

mAP

68.41
80.22
82.26
82.01
85.32
85.78
85.93
79.53
81.72
82.42
85.12
86.73

31.84
36.50
38.20
39.43
41.29

85.74
87.32
86.73
87.04
88.04
88.10

38.22
39.43
40.14
40.17
41.72
41.76

Bold denotes best result.

Table 6. Ablation Analysis of FMFA Modules across Three Public Benchmarks
No.

Methods

0
1
2
3

Baseline
+A-SDM
+EFA
FMFA

Components
SDM A-SDM EFA
Ø
Ø
Ø
Ø
Ø
Ø

Rank-1
73.45
74.04
73.73
74.16

CUHK-PEDES
Rank-5 Rank-10
89.38
93.69
89.86
93.89
89.64
94.04
90.12
94.10

mAP
66.13
66.45
66.40
66.66

Rank-1
63.48
64.26
63.77
64.29

ICFG-PEDES
Rank-5 Rank-10
80.16
85.78
80.59
85.90
80.39
85.86
80.48
85.93

mAP
38.20
39.39
39.17
39.43

Rank-1
59.50
60.25
60.45
61.05

RSTPReid
Rank-5 Rank-10
81.80
88.85
81.45
88.70
82.30
89.25
83.85
89.80

mAP
47.44
47.69
47.64
48.22

Bold denotes best result.

for practical applications. When adopting VL-Backbones with ReID-domain pre-training, FMFA
yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
HAM-based backbones, respectively.
In conclusion, FMFA attains the highest performance across all evaluation metrics on the three
widely used public benchmarks. As far as we are aware, FMFA is the best method for all global
matching methods. This highlights the ability of our method to generalize well and maintain
robustness.
4.3

Ablation Study

In this subsection, we examine our proposed components in the FMFA framework. For simplicity,
we omit the components of L𝐼 𝐷 and the IRR module that were proposed by IRRA and used in all
experiments. Only one of SDM and A-SDM can be used at the same time.
To thoroughly assess the contribution of our FMFA modules, we undertake an empirical analysis
on three widely used datasets. Table 6 summarizes the Rank-1/5/10 accuracies (%) together with
the mAP (%) performance.
Effect of the A-SDM Module. To evaluate the contribution of the A-SDM module, we perform
ablation experiments by replacing the A-SDM module with the SDM module, keeping all hyperparameters unchanged. Specifically, as shown in Table 6, replacing A-SDM with SDM results in a
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

Cross-Modal Full-Mode Fine-Grained Alignment for TIPR

135:13

reduction of Rank-1 accuracy by 0.59%, 0.78%, and 0.75% across the three datasets, and also causes
a 1.19% drop in mAP on the ICFG-PEDES dataset, as observed in No. 0 vs. No. 1. Additionally, all
evaluation metrics on CUHK-PEDES and ICFG-PEDES degrade, further confirming the superiority
of A-SDM. Moreover, when combined with the EFA module, the advantage of A-SDM becomes
even more pronounced. As shown in No. 2 vs. No. 3, replacing the A-SDM module with the SDM
module results in 0.43% and 0.48% decrease in Rank-1 and Rank-5 on the CUHK-PEDES dataset,
respectively, as well as a 1.55% drop in Rank-5 and a 0.58% decline in mAP on the RSTPReid dataset.
These results collectively validate the consistent and significant impact of A-SDM to performance.
Effect of the EFA Module. To improve the model’s global matching performance, the EFA module
introduces fine-grained cross-modal interaction based on a sparse similarity matrix. The impact
of the EFA module is illustrated by comparing the results of No. 0 vs. No. 2 and No. 1 vs. No. 3.
Specifically, as shown in No. 1 vs. No. 3, removing the EFA module from FMFA leads to a performance
drop of 0.26% and 0.21% in Rank-5 and Rank-10 on the CUHK-PEDES dataset, and a more significant
decline of 2.40% and 1.10% in Rank-5 and Rank-10, along with a 0.53% decrease in mAP on the
RSTPReid dataset. However, EFA causes a 0.11% drop in Rank-5 on ICFG-PEDES, suggesting that
its sparse and hard coding strategy, which focuses only on the most relevant patches, may overlook
other informative ones and lead to information loss. Notably, this comparison also reflects the
joint ablation of EFA and the A-SDM module, further verifying the complementary effect between
the two modules. Moreover, to further validate the individual contribution of the EFA module,
we design an additional experiment that retains the SDM module while removing only the EFA
module, as observed in No. 0 vs. No. 2. In this setting, the absence of EFA results in 0.38%, 0.29%,
and 0.95% drops in Rank-1 on CUHK-PEDES, ICFG-PEDES, and RSTPReid, and causes a 1.23% drop
in mAP on ICFG-PEDES. These evaluations highlight the effectiveness of the EFA module.
4.4

Parameter Study

We perform a parameter study on the CUHK-PEDES dataset, examining three hyperparameters—the
weight factor 𝛼, the sparsity threshold 𝜎, and the factor 𝜆—as well as the contribution weights of
the proposed loss functions, L𝐴-𝑆𝐷𝑀 and L𝐸𝐹𝐴 . When examining a specific parameter, all other
parameters are maintained as specified in Section 4.1.
Hyperparameters Analysis. As shown in Figure 4(a), we vary the weight factor 𝛼 from 1 to 20.
The results on CUHK-PEDES show that setting 𝛼 to 10 achieves the highest performance across the
evaluated metrics, suggesting that the adaptive pull force on unmatched positive pairs is optimal.
However, when 𝛼 is increased to 20, the performance drops significantly because the pull force
becomes excessively strong, causing unmatched positive pairs to fail to align properly and instead
over-pulling mismatched positives, which leads to false positives. We further vary the sparsity
threshold 𝜎 from 0.1/𝑁 to 20/𝑁 and the LSE factor 𝜆 from 0.1 to 10, where 𝑁 denotes the number of
image patches. As illustrated in Figure 4(b) and (c), setting 𝜎 to 1/𝑁 and 𝜆 to 1 yields the best overall
performance. When 𝜎 is set too high, only a few highly relevant patches are retained, leading to the
loss of semantic information, whereas an excessively low 𝜎 preserves most patches and weakens
the ability to capture discriminative features. According to Equation (10), a large 𝜆 emphasizes the
differences among patch responses and makes the pooling operation less robust to noise, while
a small 𝜆 smooths these responses excessively and reduces feature discrimination. Moreover, to
evaluate the stability of these hyperparameters, we further set 𝜎 around 1/𝑁 (i.e., 0.9/𝑁 and 1.1/𝑁 )
and 𝜆 around 1 (i.e., 0.9 and 1.1). The results show that although slightly adjusting 𝜎 and 𝜆 leads to
a minor performance drop, our model still obtains high performance, demonstrating the robustness
of our method.
Loss Function Weights. We perform experiments using the CUHK-PEDES dataset to investigate
the influence of the weights of the two proposed loss functions, L𝐴-𝑆𝐷𝑀 and L𝐸𝐹𝐴 , varying them
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

135:14

H. Yin et al.

Fig. 4. The sensitivity analysis of hyperparameters of FMFA on the CUHK-PEDES dataset.
Table 7. The Sensitivity Analysis of the Weight of L𝐴-𝑆𝐷𝑀 and
L𝐸𝐹𝐴 on the CUHK-PEDES Dataset
Weight
0.1
0.5
1.0
5.0
10.0

Rank-1
71.47
71.23
74.16
71.86
72.45

L𝐴-𝑆𝐷𝑀
Rank-5 Rank-10
88.62
93.03
88.06
93.16
90.12
94.10
88.27
93.12
88.62
93.24

Rank-1
68.32
58.49
74.16
0.09
0.09

L𝐸𝐹 𝐴
Rank-5
85.75
80.05
90.12
0.19
0.19

Rank-10
91.04
87.13
94.10
0.32
0.32

One weight is varied, and the other is fixed at 1.0.

Table 8. Ablation Study Verifying the Necessity
of the Constant Term “+1” in A-SDM on the
CUHK-PEDES Dataset
Setting
A-SDM w/o “+1”
A-SDM w “+1”

“+1” Term
Ø

Rank-1
26.12
74.16

Rank-5
48.36
90.12

Rank-10
59.18
94.10

from 0.1 to 10.0, as shown in Table 7. To comprehensively explore the optimal combination of the
proposed loss weights, we only vary one weight at a time while keeping the other fixed at 1 to
ensure a controlled comparison. The results show that setting both weights to 1 yields the best
performance. Notably, increasing the weight of L𝐸𝐹𝐴 to 5.0 or 10.0 results in gradient explosion,
making the model untrainable and causing all metrics to drop below 1.
The “+1” Term in A-SDM. We carry out experiments on the CUHK-PEDES dataset to validate
the necessity of the constant “+1” in A-SDM. As defined in Equation (2), the constant term “+1”
ensures a positive lower bound for the adaptive weight, preventing it from becoming zero when a
pair is correctly matched. Without this term, gradients of correctly matched positive pairs diminish,
leading to unstable optimization and degraded performance during training. Table 8 indicates that
omitting the “+1” leads to a notable decline in retrieval accuracy, illustrating that the effectiveness of
model is reduced. Therefore, the “+1” term is a crucial component that ensures stable optimization
and consistent performance of A-SDM.
4.5 Qualitative Results
Visualization of the Sparse Process. We visualize the similarity maps before and after the sparse
process (i.e., Equations (5) and (6)) in the EFA module, as shown in Figure 5. Patches with high similarity (greater than 0.75) are preserved, as indicated by the black frames in the left part of Figure 5.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

Cross-Modal Full-Mode Fine-Grained Alignment for TIPR

135:15

Fig. 5. Visualization of similarity maps before and after the sparse process in the EFA module.

In contrast, patches with low similarity (below 0.5) are suppressed to values under 0.25 and thus
omitted during aggregation, as shown by the pink frames on the left. When most patches in an
image exhibit relatively high similarity (around 0.75), the sparse process retains only those with
the highest similarity while reducing the similarity of the remaining patches below 0.5, highlighted
by the purple frames in the right part of Figure 5. This guarantees that the subsequent aggregation
emphasizes only the most pertinent patches, decreasing computational and memory overhead while
preserving performance. Unlike implicit aggregation methods based on attention mechanisms,
EFA explicitly aggregates image patches and text tokens, allowing us to observe whether the most
relevant patches are effectively aggregated by visualizing the similarity between patches and tokens.
Inference Time Comparison. We compare the inference time between FMFA and recent matching
methods (e.g., PLOT [35] and RaSa [2]) on the test sets of three datasets, as shown in Table 9. As a
global matching method, FMFA only computes global features during inference, thus achieving a
higher inference speed than local matching methods. Even when compared with the recent global
matching method DM-Adapter [29], FMFA still achieves consistently faster inference across all
three datasets. Moreover, the efficiency advantage of FMFA becomes more pronounced with the
growth of the test set. For instance, the inference time of FMFA vs. PLOT increases from 3 s vs. 5 s
on RSTPReid to 50 s vs. 91 s on ICFG-PEDES. These comparisons clearly demonstrate that FMFA
achieves superior inference speed compared with recent methods.
Visualization of Top-5 Retrieval Results. Figure 6 compares the top-5 retrieval results between the
baseline IRRA𝑅 and our proposed FMFA on the CUHK-PEDES dataset. Figure 6 illustrates that FMFA
delivers more precise retrieval results, correctly identifying images that the baseline fails to match.
For query texts where the baseline performs well, FMFA further improves performance by retrieving
more relevant pedestrian images (e.g., Figure 6(a) and (c)). Even for hard negative samples, where the
baseline struggles to retrieve the correct image, FMFA still enhances the similarity between positive
pairs (e.g., Figure 6(b) and (d)). This is because our proposed FMFA focuses on the unmatched
positive pairs and adaptively pulls the positive pairs closer. More comparisons of the top-5 retrieved
results are provided in Appendix B.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.


exec
/bin/zsh -lc "pdftotext -f 11 -l 14 'CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
CVAF for Aerial-Ground Person Re-Identification

85:11

Table 2. Performance Comparison on the
CARGO Dataset
CARGO
Method

ALL

𝐴↔𝐺

R-1

mAP

R-1

mAP

CLIP-ReID [10]

61.54
70.12

53.54
64.07

43.13
63.75

40.11
56.54

VDT [40]a
SeCap [27]a

64.10
68.59

55.20
60.19

48.12
69.43

42.76
58.94

Ours

73.15

66.65

68.00

60.26

BoT(ViT) [15]a

“ALL” denotes the overall retrieval performance of each
method. “𝐴 ↔ 𝐺 ” represents the performance of each model
in a specific retrieval pattern. The best performance are represented in bold.
a Performance results that are cited from the original paper
SeCap [27].

Table 3. The Efficacy of Components in CVAF Is Evaluated on the AG-ReID Datasets
AG-ReID (A → G)

AG-ReID (G → A)

R-1

mAP

R-1

mAP

Baseline (CLIP-ViT-B/16)
+ TCL
+ SFM
+ TCL + SFM (Ours)

85.08
85.94 (+0.86)
86.76 (+1.68)
87.23 (+2.15)

78.84
79.02 (+0.18)
79.43 (+0.59)
79.45 (+0.61)

86.82
87.36 (+0.74)
89.43 (+2.81)
90.12 (+3.50)

81.00
81.63 (+0.63)
82.18 (+1.18)
82.59 (+1.59)

Baseline (CLIP-ResNet50)
+ TCL
+ SFM
+ TCL + SFM (Ours)

74.81
75.62 (+0.81)
77.02 (+2.21)
77.62 (+2.81)

64.18
64.93 (+0.75)
65.77 (+1.59)
65.99 (+1.81)

74.82
75.97 (+1.15)
77.47 (+2.65)
78.12 (+3.30)

66.11
67.02 (+0.91)
67.47 (+1.36)
68.01 (+1.90)

Settings

The best performance and best improvement are in bold.
SFM, Semantic Filtering Module; TCL, Text Consistency Loss.

notable gains over the baseline, verifying its ability to refine visual representations by emphasizing
identity-relevant regions while suppressing redundant background information. When both TCL
and SFM are jointly applied, the model achieves the best results under all evaluation protocols
and across both backbones. The synergy between TCL and SFM allows the model to learn more
robust, view-invariant, and semantically consistent representations. These results confirm that
the proposed modules are not only effective but also architecture-agnostic, providing consistent
improvements for both Transformer- and CNN-based encoders.
Effect of TCL. The effectiveness of the proposed TCL is demonstrated on both the AG-ReID and
CARGO datasets, as shown in the second row of Tables 3 and 4. By introducing TCL into the baseline
model (CLIP-ReID), consistent performance improvements are observed across both datasets. This
indicates that the learnable text tokens are capable of aggregating multi-view information from
images of the same identity during training, enabling the extraction of the most discriminative,
view-invariant features as textual descriptions. These view-invariant text embeddings then serve as
reliable guidance for the image encoder in subsequent stages. Notably, TCL requires no additional
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

85:12

D. Mao et al.
Table 4. The Efficacy of Components in CVAF on the CARGO Datasets
Settings

CARGO (ALL)

CARGO (A ↔ G)

R-1

mAP

R-1

mAP

Baseline (CLIP-ViT-B/16)
+ TCL
+ SFM
+ TCL + SFM (Ours)

70.12
70.79 (+0.67)
72.40 (+2.28)
73.15 (+3.03)

64.07
64.78 (+0.71)
65.25 (+1.18)
66.65 (+2.58)

63.75
64.89 (+1.14)
66.76 (+3.01)
68.00 (+4.25)

56.54
58.77 (+2.23)
59.23 (+2.69)
60.26 (+3.72)

Baseline (CLIP-ResNet50)
+ TCL
+ SFM
+ TCL + SFM (Ours)

61.52
61.83 (+0.31)
63.30 (+1.78)
63.53 (+2.01)

55.33
55.57 (+0.24)
56.22 (+0.89)
56.91 (+1.58)

46.93
48.26 (+1.33)
48.92 (+1.99)
49.81 (+2.88)

43.85
45.67 (+1.82)
45.70 (+1.85)
46.51 (+2.66)

The best performance and best improvement are in bold.
SFM, Semantic Filtering Module; TCL, Text Consistency Loss.

network structures or parameters and functions purely as an auxiliary loss term, yet it delivers stable
and significant performance gains. This demonstrates its strong generalizability and transferability,
validating that enhancing the view-invariance of text token representations is an effective strategy
for improving cross-modal recognition. It is worth noting that the TCL alone may yield moderate
performance improvements and should not be viewed merely as a regularization term. Instead,
its true strength emerges when combined with the SFM. The synergy between TCL and SFM
enables more effective extraction of robust, view-invariant identity features, leading to significantly
enhanced overall performance.
To further validate the effectiveness of the proposed TCL, we design an additional analysis
experiment to explicitly examine how TCL enhances the cross-view consistency between image
features and the learned text embeddings. In our framework, each textual description corresponds
to multiple images of the same identity captured from diverse viewpoints. Ideally, the learned
text embedding should align equally well with all associated image features, regardless of view.
Therefore, we hypothesize that applying TCL should reduce the variance of image-text similarities
across views and simultaneously improve the overall alignment. To validate this hypothesis, we
randomly selected five identities from the training set and computed the cosine similarity between
their fixed textual embeddings and image features under different views, both with and without TCL.
For each identity, we report the mean similarity and variance across multi-view image–text pairs,
as shown in Figure 4(a), where the error bars denote the variance. It can be observed that the mean
similarity consistently increases after applying TCL, while the variance decreases in most cases,
indicating that TCL effectively enhances cross-view semantic consistency. Furthermore, to capture
the global variance change across identities, we plotted the variance distribution in a boxplot, as
shown in Figure 4(b). Compared with the case without TCL, the variance distribution with TCL
exhibits a shorter interquartile range and a more compact overall spread, further demonstrating that
our TCL reduces intra-ID similarity dispersion and thereby strengthens global stability. It is worth
noting that a few identities exhibit slightly increased variance after applying TCL. We attribute
this to the model becoming more confident in aligning the text embedding with certain dominant
views, resulting in a sharper distribution of similarity scores—i.e., higher similarity with one or two
representative images, while others contribute less. This sharper focus can cause a minor increase
in variance, especially when some views are less informative or contain occlusion. Nonetheless,
this behavior is not contradictory to our objective. The overall trend of reduced variance across
identities clearly supports that TCL improves the robustness and view-invariance of the learned
textual representation.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

CVAF for Aerial-Ground Person Re-Identification

85:13

Fig. 4. Comparison of image-text similarity before and after applying TCL. (a) Mean similarity for selected
IDs, with error bars representing intra-ID standard deviation, showing that TCL generally improves alignment
consistency. (b) Boxplot of variance across all IDs, illustrating that TCL reduces the spread of intra-ID similarity
and thus enhances global stability.

Effect of SFM. The effectiveness of incorporating the SFM is further validated on both the AG-ReID
and CARGP datasets, as shown in the third row of Tables 3 and 4. With the addition of SFM, the
model achieves notable performance improvements on AG-ReID: +1.68% Rank-1 and +0.59% mAP
under the A → G setting, and +2.81% Rank-1 and +1.18% mAP under the G → A setting. Similarly,
on the CARGP dataset, the model gains +2.28% Rank-1 and +1.18% mAP in the ALL setting, and
+3.01% Rank-1 and +2.69% mAP in the A ↔ G setting. These consistent improvements demonstrate
that, guided by the learned text representations, SFM effectively enhances the image encoder’s
ability to focus on view-invariant features while suppressing irrelevant or view-specific information
not aligned with the semantic cues.
As shown in Figure 5, we conduct ablation studies to validate the necessity of each component
within the SFM. Specifically, we design a series of simplified models by individually removing
channel modulation, spatial modulation, and the residual connection. We then compare their
performance on the AGPReID task. The results consistently show that the absence of either channelwise or spatial-wise modulation leads to a noticeable drop in performance, underscoring the
importance of multi-dimensional feature calibration in suppressing view-sensitive information and
enhancing identity discrimination. Notably, removing the residual connection results in the most
severe performance degradation, indicating that the residual path plays a crucial role in preserving
the integrity of the original features and maintaining effective feature modulation. These findings
collectively demonstrate that each component of SFM contributes to its ability to guide the model
in learning more robust and view-invariant representations.
Analysis of SFM Placement. To investigate the optimal placement of the SFM in the image encoder,
we conducted a position ablation study by inserting a single SFM module after different transformer
layers L = {6, 9, 101, 112} and evaluated its effect on cross-view performance on the AG-ReID
dataset. Table 5 reports the performance for both A → G and G → A retrieval settings. It can be
observed that placing SFM after shallow layers (L = 6 or 9) leads to degraded performance. This
is because shallow features primarily capture low-level textures and spatial details, which are
often noisy and lack stable semantic information, making them less suitable for semantic filtering.
Inserting SFM after a higher layer (L = 10) shows noticeable improvement, indicating that the
features are becoming sufficiently abstract and discriminative. Inserting SFM after the deepest
layers (L = 11 or 12) yields the best performance, with almost identical results for both layers. This
demonstrates that the SFM module is robust to the last few layers of the encoder. For clarity in
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

85:14

D. Mao et al.

Fig. 5. Ablation study of SFM on AG-ReID dataset.
Table 5. Ablation Study of SFM Placement in the Image Encoder
Settings
L =6
L =9
L = 10
L = 11
L = 12

AG-ReID (A → G)

AG-ReID (G → A)

R-1

mAP

R-1

mAP

81.04
82.94
85.65
87.21
87.23

76.43
77.02
78.62
79.45
79.45

82.87
84.36
88.66
90.13
90.12

76.05
78.63
81.27
82.58
82.59

The best performance in each column is highlighted in bold.

Table 6. Model Complexity Comparison of Baseline CLIP-ReID and CVAF
under Two Backbones

Model
CLIP-ReID (ViT-B/16)
CLIP-ReID (ResNet50)
CVAF (ViT-B/16)
CVAF (ResNet50)

Parameters (M)
132.59
89.68
134.66
91.76

FLOPs (G)
11.10
6.57
11.33
7.51

Model Size (MB)
506.12
342.67
514.38
350.52

illustrations, we depict the SFM module after the 11th layer. Overall, this study confirms that SFM
is most effective when applied to high-level semantic features, aligning with its design purpose
of enhancing identity-relevant representations for cross-view alignment. Exploring multiple SFM
modules is left for future work.
Analysis of Model Complexity. To evaluate the computational cost of the proposed CVAF module,
we compare the model parameters, Floating-Point Operations (FLOPs), and model size of the
baseline CLIP-ReID and our CVAF-enhanced models, as summarized in Table 6.
As shown in Table 6, the CVAF module introduces only minor increases in model complexity: For
the ViT-B/16 backbone, the parameter count increases by 2.07M (+1.56%), FLOPs by 0.23G (+2.07%),
and model size by 8.26 MB (+1.63%). For the ResNet50 backbone, the parameter count increases
by 2.08M (+2.32%), FLOPs by 0.94G (+14.31%), and model size by 7.85 MB (+2.29%). The relatively
higher FLOPs increase observed on ResNet50 is due to the SFM module requiring conversion of
convolutional feature maps into patch representations for computation, followed by conversion
back to the original feature map format. In contrast, for ViT-B/16, SFM operates directly on patch
tokens, so no additional conversions are needed, resulting in negligible computational overhead.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.


exec
/bin/zsh -lc "pdftotext -f 7 -l 9 'CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Table 4
Verify the effectiveness of MDG on PRCC, MI is our Meta-Insert skill.

Table 6
Verify the effectiveness of HSF on PRCC.

Method

rank-1

rank-10

mAP

Method

rank-1

rank-10

mAP

CoOp
CoCoOp
MDG(without MI)
MDG(with MI)

48.7
46.8
47.7
50.9

57.1
57.5
55.4
60.1

48.5
49.8
49.2
50.8

Base
RGB
Gray-scale

48.7
52.6
51.0

57.1
66.5
64.7

48.5
56.3
55.3

HSF

63.2

72.0

59.7

Table 5
Verify the effectiveness of VSS on PRCC, The ‘*’ indicates results reproduced using the
ViT backbone through the source code.
Methods

Backbones

rank-1

rank-10

mAP

PCB (Sun et al., 2018)
MGN (Wang et al., 2018)
ResNet-50(He et al., 2016)
FSAM (Hong et al., 2021)
GI-ReID (Jin et al., 2022)
CAL (Jin et al., 2022)
CLIP(R50)(Radford et al., 2021)

ResNet
ResNet
ResNet
ResNet
ResNet
ResNet
ResNet

22.9
25.9
19.6
54.5
37.6
55.2
41.0

–
–
–
–
–
–
–

–
–
8.1
–
–
55.8
37.5

DenseNet-121(Huang et al., 2017)
RCSANet (Huang et al., 2021)
MVSE (Gao et al., 2022)

DenseNet
DenseNet
DenseNet

18.7
31.6
47.4

–
–
–

23.7
31.5
52.5

ViT-VIBE Hybrid (Bansal et al., 2022)
CAL*(Jin et al., 2022)
MVSE*(Gao et al., 2022)
CLIP(Vit-B-16)(Radford et al., 2021)

VIT
VIT
VIT
VIT

47.0
32.7
27.3
45.8

–
–
–
–

–
33.6
28.8
46.1

VSS
VSS
VSS

CNN+CNN
ViT+ViT
Hybrid

41.6
45.9
51.7

50.9
52.2
60.3

37.8
46.6
51.5

𝛼 is big or small, the performance of our VSS module wound decrease.
The best 𝛼 value is 0.4. This is attributed to the structural design of the
HSF loss, whose first term guides the visual features to bias the pseudotext space generated by RGB images, and the second term guides the
visual features to bias the pseudo-text space generated by gray-scale
images. Since the two together constitute the HSF loss in an additive
manner, its overall optimization is the result of integrating both color
distribution and shape information in the visual features.
(4) Effectiveness of CSGN: Based on the above discussion and experimental validation, our proposed CSGN contains two different description generators for generating language descriptions of RGB images
and gray-scale images respectively, which alleviates the problem of
inaccurate CLIP-ReID prompt generation. Based on this, the HSF loss
plays the role of fusing descriptions, whose target is enriching textual
features and guiding the learning of the CCReID model. Notably, the
algorithm achieves the best performance on Celeb, Celeb-Light, and VCClothes datasets, with a rank-1 accuracy of 64.6%,56.4%, and 92.7%
and a mAP of 23.3%,38.4%, and 85.1%, respectively. Furthermore, the
visualization in Fig. 6 where ‘red’ presents the wrong match and ‘green’
presents the right match vividly shows that the proposed CSGN enables
the model to accurately identify the target pedestrian. These results
demonstrate the positive contribution of each module to the proposed
method. In conclusion, our CSGN is both rational and effective and
exhibits excellent performance.

(Stage 1) show that this approach outperforms pre-training the model
with CLIP alone, with improvements of 2.2%, 3.0%, and 2.3% in rank1, rank-10, and mAP on the PRCC dataset, respectively. Additionally, in
Table 4, we compare MDG with popular prompt generation methods.
When comparing MDG with CoOP, our MDG’s performance surpasses
CoOP by 2.2% in rank-1 accuracy and 2.3% in mAP. Similarly, when
comparing MDG with CoCoOP, our MDG outperforms it by 4.1% in
rank-1 accuracy and 1.0% in mAP. These results highlight the superiority of our method. It can be demonstrated that our MDG generates
richer semantic information, largely due to the inclusion of grayscale images. While RGB images provide pseudo-text focusing on color
distribution, gray-scale images are less affected by color interference
and emphasize shape information, such as body shape. When used together in training the MDGs, they generate more instructive pseudo-text
descriptions, combining the strengths of both modalities.
(2) Effectiveness of VSS: As shown in Table 5, CLIP achieves the best
performance whether using ResNet or ViT as the backbone. However,
when comparing VSS with CLIP(R50), our VSS surpasses it by 10.7%
in rank-1 and 14.0% in mAP. Similarly, when comparing VSS with
CLIP(ViT-B-16), our VSS outperforms it by 5.9% in rank-1 and 5.4%
in mAP. This demonstrates that our VSS module effectively combines
the strengths of both ViT and ResNet. This improvement is made
possible by the introduction of a lightweight fusion header, which
fuses global and local features in a balanced manner. It assigns equallength weight vectors to both feature types, and the fused feature
is obtained by summing the products of the two features with their
corresponding weight vectors. This approach is conceptually similar to
the fully connected layer but avoids the complex computations by using
simple weight vector products instead.
(3) Effectiveness of HSF: As shown in Table 6, comparing line 1 with
our HSF loss, our method outperforms 14.5% and 11.2% in rank-1 and
mAP than no use. Comparing line 2 with our HSF loss, our method
outperforms 10.6% and 3.4% in rank-1 and mAP. Comparing line 3
with our HSF loss, our method outperforms 12.2% and 4.4% in rank-1
and mAP. This demonstrates that our HSF loss can utilize high-level
semantic information to guide the learning of visual features, which in
turn breaks the bottleneck. As shown in Fig. 5, we can see that when

5. Conclusion
In this paper, we apply CLIP to the CCReID task and propose a twostage CSGN framework. In the first phase, we introduce the innovative
Multi-Description Generator (MDG), which generates pseudo-text descriptions in both RGB and grayscale modes. This dual-mode approach,
combined with a text-image mutual loss mechanism, effectively reduces the color sensitivity of the text encoder. Notably, each linguistic
description captures different semantic nuances. In the second phase,
we design the Visual Semantic Steering (VSS) module, a crucial component that enhances visual feature representation by leveraging the
complementary nature of ResNet and ViT feature extractors. To further
refine our model, we incorporate Heterogeneous Semantic Fusion (HSF)
loss, which imposes stricter constraints on the individual components,
ensuring a comprehensive and robust model.
However, this work represents only a preliminary exploration in
the field of clothing change re-identification. More in-depth research is
required to fully exploit the advantages of visual-language models for
improving the feature extraction capabilities of visual models. Firstly,
the linguistic descriptions generated by our method are still relatively
coarse. This is largely due to the lack of accurate linguistic descriptions
in the clothing change re-identification datasets, which limits our ability to fine-tune large models effectively and fully utilize their potential.
This constraint impacts the richness of the semantic information. To address this, we plan to construct a new clothing change re-identification
dataset with accurate linguistic descriptions. Secondly, clothing change
re-identification still faces challenges in improving the visual encoder.
Specifically, extracting global and local discriminative features with
fewer parameters and computations remains a significant challenge.
As such, we intend to investigate more efficient and accurate network
architectures in future work.
7

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Fig. 5. The effect of the selection of the hyperparameter 𝛼.

Fig. 6. Comparing visual sorting results of different algorithms.

8

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

CRediT authorship contribution statement

Houlsby, Neil, Giurgiu, Andrei, Jastrzebski, Stanislaw, Morrone, Bruna, De Laroussilhe, Quentin, Gesmundo, Andrea, Attariyan, Mona, Gelly, Sylvain, 2019.
Parameter-efficient transfer learning for NLP. In: International Conference on
Machine Learning. PMLR, pp. 2790–2799. http://dx.doi.org/10.48550/arXiv.1902.
00751.
Huang, Gao, Liu, Zhuang, Van Der Maaten, Laurens, Weinberger, Kilian Q, 2017.
Densely connected convolutional networks. In: Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition. pp. 4700–4708. http://dx.doi.org/
10.1109/CVPR.2017.243.
Huang, Yan, Wu, Qiang, Xu, Jingsong, Zhong, Yi, 2019. Celebrities-reid: A benchmark
for clothes variation in long-term person re-identification. In: 2019 International
Joint Conference on Neural Networks. IJCNN, IEEE, pp. 1–8. http://dx.doi.org/10.
1109/IJCNN.2019.8851957.
Huang, Yan, Wu, Qiang, Xu, JingSong, Zhong, Yi, Zhang, ZhaoXiang, 2021. Clothing
status awareness for long-term person re-identification. In: Proceedings of the
IEEE/CVF International Conference on Computer Vision. pp. 11895–11904. http:
//dx.doi.org/10.1109/ICCV48922.2021.01168.
Jia, Chao, Yang, Yinfei, Xia, Ye, Chen, Yi-Ting, Parekh, Zarana, Pham, Hieu, Le, Quoc,
Sung, Yun-Hsuan, Li, Zhen, Duerig, Tom, 2021. Scaling up visual and visionlanguage representation learning with noisy text supervision. In: International
Conference on Machine Learning. PMLR, pp. 4904–4916. http://dx.doi.org/10.
48550/arXiv.2102.05918.
Jin, Xin, He, Tianyu, Zheng, Kecheng, Yin, Zhiheng, Shen, Xu, Huang, Zhen,
Feng, Ruoyu, Huang, Jianqiang, Chen, Zhibo, Hua, Xian-Sheng, 2022. Clothchanging person re-identification from a single image with gait prediction and
regularization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 14278–14287. http://dx.doi.org/10.1109/CVPR52688.
2022.01388.
Kingma, Diederik P., Ba, Jimmy, 2014. Adam: A method for stochastic optimization.
arXiv preprint arXiv:1412.6980.
Li, Shihua, Chen, Haobin, Yu, Shijie, He, Zhiqun, Zhu, Feng, Zhao, Rui, Chen, Jie,
Qiao, Yu, 2022. Cocas+: Large-scale clothes-changing person re-identification with
clothes templates. IEEE Trans. Circuits Syst. Video Technol. 33 (4), 1839–1853.
http://dx.doi.org/10.1109/TCSVT.2022.3216769.
Li, Xulin, Lu, Yan, Liu, Bin, Hou, Yuenan, Liu, Yating, Chu, Qi, Ouyang, Wanli,
Yu, Nenghai, 2023a. Clothes-invariant feature learning by causal intervention for
clothes-changing person re-identification. http://dx.doi.org/10.48550/arXiv.2305.
06145, arXiv preprint arXiv:2305.06145.
Li, Yu-Jhe, Luo, Zhengyi, Weng, Xinshuo, Kitani, Kris M., 2020. Learning shape
representations for clothing variations in person re-identification. http://dx.doi.org/
10.48550/arXiv.2003.07340, arXiv preprint arXiv:2003.07340.
Li, Junnan, Selvaraju, Ramprasaath, Gotmare, Akhilesh, Joty, Shafiq, Xiong, Caiming,
Hoi, Steven Chu Hong, 2021a. Align before fuse: Vision and language representation
learning with momentum distillation. Adv. Neural Inf. Process. Syst. 34, 9694–9705.
http://dx.doi.org/10.48550/arXiv.2107.07651.
Li, Siyuan, Sun, Li, Li, Qingli, 2023b. CLIP-ReID: exploiting vision-language model
for image re-identification without concrete text labels. In: Proceedings of the
AAAI Conference on Artificial Intelligence, vol. 37, (1), pp. 1405–1413. http:
//dx.doi.org/10.1609/aaai.v37i1.25225.
Li, Yu-Jhe, Weng, Xinshuo, Kitani, Kris M., 2021b. Learning shape representations for
person re-identification under clothing change. In: Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision. pp. 2432–2441. http:
//dx.doi.org/10.1109/WACV48630.2021.00248.
Lin, Yin, Liu, Cong, Chen, Yehansen, Hu, Jinshui, Yin, Bing, Yin, Baocai, Wang, Zengfu,
2023. Exploring part-informed visual-language learning for person re-identification.
http://dx.doi.org/10.48550/arXiv.2308.02738, arXiv preprint arXiv:2308.02738.
Liu, Fangyi, Ye, Mang, Du, Bo, 2023. Dual level adaptive weighting for cloth-changing
person re-identification. IEEE Trans. Image Process. http://dx.doi.org/10.1109/TIP.
2023.3310307.
Mu, Jingyi, Li, Yong, Li, Jun, Yang, Jian, 2022. Learning clothes-irrelevant cues for
clothes-changing person re-identification.. In: BMVC. p. 337.
Qian, Xuelin, Wang, Wenxuan, Zhang, Li, Zhu, Fangrui, Fu, Yanwei, Xiang, Tao,
Jiang, Yu-Gang, Xue, Xiangyang, 2020. Long-term cloth-changing person reidentification. In: Proceedings of the Asian Conference on Computer Vision. http:
//dx.doi.org/10.1007/978-3-030-69535-4_5.
Radford, Alec, Kim, Jong Wook, Hallacy, Chris, Ramesh, Aditya, Goh, Gabriel, Agarwal, Sandhini, Sastry, Girish, Askell, Amanda, Mishkin, Pamela, Clark, Jack, et
al., 2021. Learning transferable visual models from natural language supervision.
In: International Conference on Machine Learning. PMLR, pp. 8748–8763. http:
//dx.doi.org/10.48550/arXiv.2103.00020.
Ren, Xuena, Zhang, Dongming, Bao, Xiuguo, 2022. Person re-identification with a
cloth-changing aware transformer. In: 2022 International Joint Conference on
Neural Networks. IJCNN, IEEE, pp. 1–8. http://dx.doi.org/10.1109/IJCNN55064.
2022.9892916.
Shi, Wei, Liu, Hong, Liu, Mengyuan, 2022. IRANet: Identity-relevance aware representation for cloth-changing person re-identification. Image Vis. Comput. 117, 104335.
http://dx.doi.org/10.1016/j.imavis.2021.104335.
Shu, Xiujun, Li, Ge, Wang, Xiao, Ruan, Weijian, Tian, Qi, 2021. Semantic-guided pixel
sampling for cloth-changing person re-identification. IEEE Signal Process. Lett. 28,
1365–1369. http://dx.doi.org/10.1109/LSP.2021.3091924.

Yang Lu: Writing – original draft, Visualization, Formal analysis,
Conceptualization. Bin Ge: Project administration, Methodology, Investigation. Chenxing Xia: Supervision, Resources, Funding acquisition.
Junming Guan: Validation, Investigation, Data curation.
Funding
This work was supported by National Natural Science Foundation of China (62102003), Anhui Postdoctoral Science Foundation
(2022B623), Anhui Province Young Key Teachers Overseas Study and
Research Funding Project (JWFX2023016), Huainan City Science and
Technology Plan Project (2023A316), Anhui University of Science and
Technology Graduate Student Innovation Fund Project (2023CX2125).
Declaration of competing interest
All authors disclosed no relevant relationships.
Data availability
Data will be made available on request.

References
Bansal, Vaibhav, Foresti, Gian Luca, Martinel, Niki, 2022. Cloth-changing person
re-identification with self-attention. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 602–610. http://dx.doi.org/10.
1109/WACVW54805.2022.00066.
Chen, Jiaxing, Zheng, Wei-Shi, Yang, Qize, Meng, Jingke, Hong, Richang, Tian, Qi,
2021. Deep shape-aware person re-identification for overcoming moderate clothing
changes. IEEE Trans. Multimed. 24, 4285–4300. http://dx.doi.org/10.1109/TMM.
2021.3114539.
Cui, Zhenyu, Zhou, Jiahuan, Peng, Yuxin, Zhang, Shiliang, Wang, Yaowei, 2023. Dcrreid: Deep component reconstruction for cloth-changing person re-identification.
IEEE Trans. Circuits Syst. Video Technol. http://dx.doi.org/10.1109/TCSVT.2023.
3241988.
Fan, Chao, Peng, Yunjie, Cao, Chunshui, Liu, Xu, Hou, Saihui, Chi, Jiannan,
Huang, Yongzhen, Li, Qing, He, Zhiqiang, 2020. Gaitpart: Temporal part-based
model for gait recognition. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. pp. 14225–14233. http://dx.doi.org/10.
1109/CVPR42600.2020.01423.
Gao, Zan, Wei, Hongwei, Guan, Weili, Nie, Weizhi, Liu, Meng, Wang, Meng,
2022. Multigranular visual-semantic embedding for cloth-changing person reidentification. In: Proceedings of the 30th ACM International Conference on
Multimedia. pp. 3703–3711. http://dx.doi.org/10.1145/3503161.3547884.
Gao, Zan, Wei, Shengxun, Guan, Weili, Zhu, Lei, Wang, Meng, Chen, Shengyong, 2023.
Identity-guided collaborative learning for cloth-changing person reidentification.
IEEE Trans. Pattern Anal. Mach. Intell. http://dx.doi.org/10.1109/TPAMI.2023.
3334741.
Gu, Xinqian, Chang, Hong, Ma, Bingpeng, Bai, Shutao, Shan, Shiguang, Chen, Xilin,
2022. Clothes-changing person re-identification with rgb modality only. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
pp. 1060–1069, doi: Arxiv-2204.06890.
Han, Ke, Gong, Shaogang, Huang, Yan, Wang, Liang, Tan, Tieniu, 2023. Clothingchange feature augmentation for person re-identification. In: Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp.
22066–22075.
He, Shuting, Chen, Weihua, Wang, Kai, Luo, Hao, Wang, Fan, Jiang, Wei,
Ding, Henghui, 2023. Region generation and assessment network for occluded
person re-identification. IEEE Trans. Inf. Forensics Secur. http://dx.doi.org/10.
1109/TIFS.2023.3318956.
He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian, 2016. Deep residual learning
for image recognition. In: Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition. pp. 770–778. http://dx.doi.org/10.1109/CVPR.2016.90.
Hong, Peixian, Wu, Tao, Wu, Ancong, Han, Xintong, Zheng, Wei-Shi, 2021. Fine-grained
shape-appearance mutual learning for cloth-changing person re-identification.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10513–10522. http://dx.doi.org/10.1109/CVPR46437.2021.
01037.
9


exec
/bin/zsh -lc "pdftotext -f 8 -l 10 'Cross-modality average precision optimization for visible thermal person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 164 (2025) 111489

Y. Ling et al.
Table 2
Comparison with the state-of-the-art methods on the RegDB dataset.
Settings

Visible to Thermal

Thermal to Visible

Method

Venue

R1

R10

R20

mAP

R1

R10

R20

mAP

Zero-Padding [2]
HCML [46]
BDTR [48]
DGD+MSR [51]
D2RL [49]
D-HSME [4]
AlignGAN [52]
DFE [53]
Hi-CMD [29]
PIG [30]
MANN [68]
Xmodal [27]
DDAG [54]
DG-VAE [55]
cm-SSFT [57]
SIM [56]
AGW [44]
CMAlign [26]
NFS [18]
VCD-VML [19]
MSO [58]
MCLNet [60]
MPANet [23]
SMCL [28]
CM-NAS [20]
CAJL [66]
DART [65]
MID [59]
MAUM [67]
FMCNet [61]
TTM [62]
G2DA [69]
MTMFE [70]
PMT [63]
HCT [9]
DEEN [71]
MRCN [64]
GLMC [10]

ICCV17
AAAI18
IJCAI18
TIP19
CVPR19
AAAI19
ICCV19
ACMMM19
CVPR20
AAAI20
PR20
AAAI20
ECCV20
ACMMM20
CVPR20
IJCAI20
TPAMI21
ICCV21
CVPR21
CVPR21
ACMMM21
ICCV21
CVPR21
ICCV21
ICCV21
ICCV21
CVPR22
AAAI22
CVPR22
CVPR22
PR22
PR23
PR23
AAAI23
TMM20
CVPR23
AAAI23
TNNLS21

17.75
24.44
33.47
48.43
/
50.85
57.9
70.13
/
48.50
48.67
62.21
69.34
72.97
73.3
75.29
70.05
74.17
80.54
73.2
73.6
80.3
83.7
83.93
84.54
85.03
83.60
87.45
87.87
89.84
89.91
71.72
76.10
84.83
91.05
91.1
91.4
91.84

34.21
47.53
58.42
70.32
/
73.36
/
86.32
/
/
71.55
83.13
86.19
/
/
/
86.21
/
91.96
/
88.6
92.7
/
/
95.18
95.49
/
95.73
/
/
96.57
87.13
88.86
/
97.16
97.8
98.0
97.86

44.35
56.78
67.52
79.95
/
81.66
/
91.96
/
/
80.06
91.72
91.49
86.89
/
/
91.55
/
95.07
/
/
96.03
/
/
97.85
97.54
/
/
/
/
98.33
91.92
92.41
/
98.57
98.9
99.0
98.98

18.9
20.8
31.83
48.67
/
47
53.6
69.14
/
49.3
41.11
60.18
63.46
71.78
72.9
74.47
66.37
67.64
72.1
71.6
66.9
73.07
80.9
79.83
80.32
79.14
75.67
84.85
85.09
84.43
85.64
65.90
74.39
76.55
83.28
85.1
84.6
81.42

/
21.70
32.92
/
43.4
50.15
56.3
67.99
70.93
48.1
38.68
/
68.06
/
71.0
78.30
70.49
72.43
77.95
71.8
74.6
75.93
82.8
83.05
82.57
84.75
81.97
84.29
86.95
88.38
88.34
69.50
72.18
84.16
89.30
89.5
88.3
91.12

/
45.02
58.46
/
66.1
72.40
/
85.56
86.39
/
60.82
/
85.15
/
/
/
87.12
/
90.45
/
88.7
90.93
/
/
94.51
95.33
/
93.44
/
/
96.16
84.87
87.06
/
96.41
96.8
96.7
97.86

/
55.58
68.43
/
76.3
81.07
/
91.41
/
/
69.10
/
90.31
/
/
/
91.84
/
93.62
/
/
94.59
/
/
97.37
97.51
/
/
/
/
97.98
89.85
92.38
/
98.16
98.4
98.5
98.69

/
22.24
31.96
/
44.1
46.16
53.4
66.70
66.04
48.90
32.61
/
61.80
/
71.7
75.24
65.90
65.46
69.79
70.1
67.5
69.49
80.7
78.57
78.31
77.82
73.78
81.41
84.34
83.86
84.06
63.88
71.04
75.13
81.46
83.4
81.9
81.06

Ours

–

93.93

98.83

99.61

89.17

92.09

97.57

98.53

87.19

Table 3
Ablation study of the proposed components. DMA: Dynamic Modality Alignment, CAP:
Cross-modality Average Precision, Global: global-based feature, Local: Local-based
features, GLF: Global and Local Feature (GLF).
#

Method

SYSU-MM01

RegDB

R1

mAP

R1

mAP

0
1
2
3

Baseline w/ Global
+DMA
+CAP
+DMA+CAP

57.53
65.03
64.08
67.08

54.71
60.97
60.42
64.36

65.99
82.45
81.67
86.72

58.27
73.09
73.56
79.64

4
5
6
7

Baseline w/ Local
+DMA
+CAP
+DMA+CAP

60.02
64.69
70.37
72.73

53.92
57.40
65.55
68.16

75.26
88.08
88.17
91.86

69.03
82.05
82.01
85.78

8
9
10
11

Baseline w/ GLF
+DMA
+CAP
+DMA+CAP

65.26
72.68
72.63
75.05

60.41
66.41
68.57
71.72

80.21
90.75
91.38
93.93

71.75
84.76
86.27
89.17

K = 6. However, further increasing K leads to performance degradation,
likely due to the inclusion of meaningless regions.
Computational complexity analysis. we compare the computational complexity of our model with the competitive method DEEN
[71]. The results are shown in Table 7. Due to the incorporation of
Global–Local Structure (GLS), the number of parameters in our model
is slightly higher than DEEN [71]. However, our model demonstrates a
competitive advantage in terms of FLOPs, indicating its computational
efficiency.

Fig. 8. Illustration of feature distribution on RegDB and SYSU-MM01 datasets. For
better visualization, we randomly show the distribution of 10 identities with 10 samples
for each modality from the testing set. Colors represent the identities. Circle-dot and
cross-mark denote the visible and thermal modalities, respectively.

8

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

Fig. 9. Comparison the retrieval results of the (a) Baseline /w GLF with (b) Ours. For each retrieval case, the sample of the first column represents the query image from thermal
modality. The retrieved samples with blue rectangles are positive samples of the query, and those with red rectangles are negative samples of the query. For visualization, we
select two queries and show their top-10 retrieval results on the SYSU-MM01 dataset.
Table 4
Effectiveness of two components in CAP. CAPV: Cross-modality Average Precision with
Visible queries, CAPT: Cross-modality Average Precision with Thermal queries.
Method

Baseline w/ Local
+CAPV
+CAPT
+CAP

SYSU-MM01

Retrieval Results. We then visualize retrieval results of the (a)
Baseline /w GLF and (b) Ours. The top-10 retrieval results of two
queries on the SYSU-MM01 dataset are illustrated in Fig. 9. We can
see that the top-10 retrieval ranking list of baseline contains many
negative samples, and our proposed method can significantly improve
the ranking list with more positive samples ranked in the top positions.

RegDB

R1

mAP

R1

mAP

60.02
65.26
68.87
70.37

53.92
59.88
64.62
65.55

75.26
87.47
86.41
88.17

69.03
81.28
80.71
82.01

5. Conclusion

Table 5
Comparison of distribution alignment methods. KL: Kullback–Leibler divergence, AW:
use average weights to replace the weights obtained by DMA.
Method

Baseline w/GLF
+KL
+AW
+DMA

SYSU-MM01

This paper presents a novel approach called Cross-modality Average Precision (CAP) for directly optimizing the overall ranking order in visible-thermal person re-identification (VT-ReID) tasks, where
queries and galleries belong to different modalities. We introduce two
techniques to enhance the effectiveness of CAP. First, we develop
a Dynamic Modality Alignment (DMA) to reduce cross-modality discrepancy, which alleviates the impact of non-modality variations and
achieves better modality alignment. Second, we apply our CAP and
DMA to both global and local features. We achieve state-of-the-art
performance on two datasets by integrating these techniques into a
specially designed global–local structure. Our work is limited by the
potential trade-off introduced by the DMA loss coefficient. If set too
high, this coefficient may reduce cross-modality feature discriminability, leading to inferior retrieval performance. In future work, we aim
to better balance cross-modality alignment and feature discrimination
to enhance retrieval performance. Additionally, we plan to explore the
scalability of our approach to general image retrieval tasks.

RegDB

R1

mAP

R1

mAP

65.26
68.24
69.52
72.68

60.41
61.93
62.54
66.41

80.21
81.70
86.77
90.75

71.75
74.46
78.39
84.76

Table 6
Comparison of CAP with other methods.
Method

SYSU-MM01

RegDB

R1

mAP

R1

mAP

Baseline w/ MGF

65.26

60.41

80.21

71.75

+ Center loss [72]
+ Triplet loss [6]
+ Circle loss [73]
+ Instance loss [74]

67.24
68.31
69.43
69.96

62.54
65.83
66.78
66.73

83.64
87.67
88.64
88.92

76.94
80.44
83.32
83.28

+ FastAP [35]
+ Blackbox AP [34]
+ Smooth-AP [8]

65.81
66.59
69.18

63.18
63.96
66.10

85.00
87.26
89.12

81.32
82.12
83.10

+ CAP

72.63

68.57

91.38

86.27

CRediT authorship contribution statement
Yongguo Ling: Formal analysis, Methodology, Project administration, Writing – original draft, Writing – review & editing. Zhiming
Luo: Formal analysis, Methodology, Visualization. Dazhen Lin: Formal
analysis, Investigation, Visualization. Shaozi Li: Project administration,
Supervision, Validation. Min Jiang: Formal analysis, Methodology.
Nicu Sebe: Methodology, Supervision, Validation. Zhun Zhong: Formal analysis, Investigation, Methodology, Validation, Writing – original
draft.

Table 7
Computational complexity analysis.
Method

Parameters

FLOPs

DEEN [71]
Our

41.23 MB
49.13 MB

49.03 G
31.16 G

4.5. Visualization

Declaration of competing interest

t-SNE of Feature Distribution. To better reflect the advantage of
our method, we show the t-SNE [75] of features distribution on the
testing set for RegDB and SYSU-MM01. The results on the baseline and
our method are shown in Fig. 8. We can see that there is a large crossmodality discrepancy in the feature distribution of the baseline, and
our proposed method can effectively bridge the cross-modality gap.

The authors declare that they have no financial and personal relationships with other people or organizations that can inappropriately
influence our work, there is no professional or other personal interest of
any nature or kind in any product, service and/or company that could
be construed as influencing the position presented in, or the review of,
the manuscript entitled.
9

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

Acknowledgments

[26] H. Park, S. Lee, J. Lee, B. Ham, Learning by aligning: Visible-infrared person
re-identification using cross-modal correspondences, in: ICCV, 2021.
[27] D. Li, X. Wei, X. Hong, Y. Gong, Infrared-visible cross-modal person
re-identification with an X modality, in: AAAI, 2020.
[28] Z. Wei, X. Yang, N. Wang, X. Gao, Syncretic modality collaborative learning for
visible infrared person re-identification, in: ICCV, 2021.
[29] S. Choi, S. Lee, Y. Kim, T. Kim, C. Kim, Hi-CMD: Hierarchical cross-modality
disentanglement for visible-infrared person re-identification, in: CVPR, 2020.
[30] G.-A. Wang, T.Z. Yang, J. Cheng, J. Chang, X. Liang, Z. Hou, et al., Crossmodality paired-images generation for RGB-infrared person re-identification, in:
AAAI, 2020.
[31] J. Liu, J. Wang, N. Huang, Q. Zhang, J. Han, Revisiting modality-specific feature
compensation for visible-infrared person re-identification, IEEE Trans. Circuits
Syst. Video Technol. (2022).
[32] M. Engilberge, L. Chevallier, P. Pérez, M. Cord, Sodeep: a sorting deep net to
learn ranking loss surrogates, in: CVPR, 2019.
[33] K. He, F. Cakir, S.A. Bargal, S. Sclaroff, Hashing as tie-aware learning to rank,
in: CVPR, 2018.
[34] M. Rolinek, V. Musil, A. Paulus, M. Vlastelica, C. Michaelis, G. Martius,
Optimizing rank-based metrics with blackbox differentiation, in: CVPR, 2020.
[35] F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff, Deep metric learning to rank, in:
CVPR, 2019.
[36] E. Ramzi, N. Thome, C. Rambour, N. Audebert, X. Bitot, Robust and decomposable average precision for image retrieval, Adv. Neural Inf. Process. Syst.
(2021).
[37] Z. Li, W. Min, J. Song, Y. Zhu, L. Kang, X. Wei, X. Wei, S. Jiang, Rethinking
the optimization of average precision: Only penalizing negative instances before
positive ones is enough, in: AAAI, 2022.
[38] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition,
in: CVPR, 2016.
[39] Y. Sun, L. Zheng, Y. Yang, Q. Tian, S. Wang, Beyond part models: Person retrieval
with refined part pooling (and a strong convolutional baseline), in: ECCV, 2018.
[40] Y. Ling, Z. Luo, Y. Lin, S. Li, A multi-constraint similarity learning with adaptive
weighting for visible-thermal person re-identification, in: IJCAI, 2021.
[41] F. Radenović, G. Tolias, O. Chum, Fine-tuning CNN image retrieval with no
human annotation, IEEE Trans. Pattern Anal. Mach. Intell. (2018).
[42] H. Luo, W. Jiang, Y. Gu, F. Liu, X. Liao, S. Lai, J. Gu, A strong baseline and batch
normalization neck for deep person re-identification, IEEE Trans. Multimed.
(2019).
[43] D. Nguyen, H. Hong, K. Kim, K. Park, Person recognition system based on a
combination of body images from visible light and thermal cameras, Sensors
(2017).
[44] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, S.C. Hoi, Deep learning for person
re-identification: A survey and outlook, IEEE Trans. Pattern Anal. Mach. Intell.
(2021).
[45] Z. Zhong, L. Zheng, G. Kang, S. Li, Y. Yang, Random erasing data augmentation,
in: AAAI, 2020.
[46] M. Ye, X. Lan, J. Li, P.C. Yuen, Hierarchical discriminative learning for visible
thermal person re-identification, in: AAAI, 2018.
[47] P. Dai, R. Ji, H. Wang, Q. Wu, Y. Huang, Cross-modality person re-identification
with generative adversarial training, in: IJCAI, 2018.
[48] M. Ye, Z. Wang, X. Lan, P.C. Yuen, Visible thermal person re-identification via
dual-constrained top-ranking, in: IJCAI, 2018.
[49] Z. Wang, Z. Wang, Y. Zheng, Y.-Y. Chuang, S. Satoh, Learning to reduce
dual-level discrepancy for infrared-visible person re-identification, in: CVPR,
2019.
[50] K. Kansal, A.V. Subramanyam, Z. Wang, S. Satoh, SDL: Spectrum-disentangled
representation learning for visible-infrared person re-identification, IEEE Trans.
Circuits Syst. Video Technol. (2020).
[51] Z. Feng, J. Lai, X. Xie, Learning modality-specific representations for
visible-infrared person re-identification, IEEE Trans. Image Process. (2019).
[52] G. Wang, T. Zhang, J. Cheng, S. Liu, Y. Yang, Z. Hou, RGB-infrared crossmodality person re-identification via joint pixel and feature alignment, in: ICCV,
2019.
[53] Y. Hao, N. Wang, X. Gao, J. Li, X. Wang, Dual-alignment feature embedding for
cross-modality person re-identification, in: ACM MM, 2019.
[54] M. Ye, J. Shen, D.J. Crandall, L. Shao, J. Luo, Dynamic dual-attentive aggregation
learning for visible-infrared person re-identification, in: ECCV, 2020.
[55] N. Pu, W. Chen, Y. Liu, E.M. Bakker, M.S. Lew, Dual Gaussian-based variational
subspace disentanglement for visible-infrared person re-identification, in: ACM
MM, 2020.
[56] M. Jia, Y. Zhai, S. Lu, S. Ma, J. Zhang, A similarity inference metric for
RGB-infrared cross-modality person re-identification, in: IJCAI, 2020.
[57] Y. Lu, Y. Wu, B. Liu, T. Zhang, B. Li, Q. Chu, N. Yu, Cross-modality person
re-identification with shared-specific feature transfer, in: CVPR, 2020.

This work is supported by the National Natural Science Foundation of China (No. 62466003, No. 62276221, No. 62376232), the
Guangxi Natural Science Foundation of China (No. 2023JJB170012,
No. 2024JJA170003), the Fujian Provincial Natural Science Foundation
of China (No. 2022J01002). We acknowledge the Hefei Advanced
Computing Center, for the availability of high-performance computing
resources and support.
Data availability
Data will be made available on request.

References
[1] Y. Lin, L. Zheng, Z. Zheng, Y. Wu, Z. Hu, C. Yan, Y. Yang, Improving person
re-identification by attribute and identity learning, Pattern Recognit. (2019).
[2] A. Wu, W.-S. Zheng, H.-X. Yu, S. Gong, J. Lai, Rgb-infrared cross-modality person
re-identification, in: ICCV, 2017.
[3] M. Ye, X. Lan, Z. Wang, P.C. Yuen, Bi-directional center-constrained top-ranking
for visible thermal person re-identification, TIFS (2019).
[4] Y. Hao, N. Wang, J. Li, X. Gao, HSME: hypersphere manifold embedding for
visible thermal person re-identification, in: AAAI, 2019.
[5] H. Ye, H. Liu, F. Meng, X. Li, Bi-directional exponential angular triplet loss for
RGB-infrared person re-identification, IEEE Trans. Image Process. (2020).
[6] A. Hermans, L. Beyer, B. Leibe, In defense of the triplet loss for person
re-identification, 2017, arXiv preprint arXiv:1703.07737.
[7] J. Revaud, J. Almazán, R.S. Rezende, C.R.d. Souza, Learning with average
precision: Training image retrieval with a listwise loss, in: ICCV, 2019.
[8] A. Brown, W. Xie, V. Kalogeiton, A. Zisserman, Smooth-ap: Smoothing the path
towards large-scale image retrieval, in: ECCV, 2020.
[9] H. Liu, X. Tan, X. Zhou, Parameter sharing exploration and hetero-center triplet
loss for visible-thermal person re-identification, IEEE Trans. Multimed. (2020).
[10] L. Zhang, G. Du, F. Liu, H. Tu, X. Shu, Global-local multiple granularity learning
for cross-modality visible-infrared person reidentification, IEEE Trans. Neural
Netw. Learn. Syst. (2021).
[11] Z. Zheng, L. Zheng, Object re-identification: Problems, algorithms and responsible
research practice, Boundaries Data (2024).
[12] Y. Ge, H. Wang, F. Zhu, R. Zhao, H. Li, Self-supervising fine-grained region
similarities for large-scale image localization, in: ECCV, 2020.
[13] Z. Zheng, Y. Wei, Y. Yang, University-1652: A multi-view multi-source
benchmark for drone-based geo-localization, in: ACM MM, 2020.
[14] X. Liu, W. Liu, H. Ma, H. Fu, Large-scale vehicle re-identification in urban
surveillance videos, in: ICME, 2016.
[15] Z. Zheng, T. Ruan, Y. Wei, Y. Yang, T. Mei, VehicleNet: Learning robust visual
representation for vehicle re-identification, IEEE Trans. Multimed. (2020).
[16] X. Bai, M. Yang, T. Huang, Z. Dou, R. Yu, Y. Xu, Deep-person: Learning
discriminative deep features for person re-identification, Pattern Recognit. (2020).
[17] M. Ye, X. Lan, Q. Leng, J. Shen, Cross-modality person re-identification via
modality-aware collaborative ensemble learning, IEEE Trans. Image Process.
(2020).
[18] Y. Chen, L. Wan, Z. Li, Q. Jing, Z. Sun, Neural feature search for RGB-infrared
person re-identification, in: CVPR, 2021.
[19] X. Tian, Z. Zhang, S. Lin, Y. Qu, Y. Xie, L. Ma, Farewell to mutual information:
Variational distillation for cross-modal person re-identification, in: CVPR, 2021.
[20] C. Fu, Y. Hu, X. Wu, H. Shi, T. Mei, R. He, Cm-nas: Cross-modality neural
architecture search for visible-infrared person re-identification, in: ICCV, 2021.
[21] Z. Wei, X. Yang, N. Wang, X. Gao, Flexible body partition-based adversarial
learning for visible infrared person re-identification, IEEE Trans. Neural Netw.
Learn. Syst. (2021).
[22] H. Liu, Y. Chai, X. Tan, D. Li, X. Zhou, Strong but simple baseline with dualgranularity triplet loss for visible-thermal person re-identification, IEEE Signal
Process. Lett. (2021).
[23] Q. Wu, P. Dai, J. Chen, C.-W. Lin, Y. Wu, F. Huang, B. Zhong, R. Ji, Discover
cross-modality nuances for visible-infrared person re-identification, in: CVPR,
2021.
[24] Z. Zhao, B. Liu, Q. Chu, Y. Lu, N. Yu, Joint color-irrelevant consistency learning
and identity-aware modality adaptation for visible-infrared cross modality person
re-identification, in: AAAI, 2021.
[25] S. Zhang, Y. Yang, P. Wang, G. Liang, X. Zhang, Y. Zhang, Attend to the
difference: Cross-modality person re-identification via contrastive correlation,
IEEE Trans. Image Process. (2021).
10


web search: 
web search: CLIP-ReID exploiting vision-language model for image re-identification AAAI 2023
web search: 
web search: "Cross-Modal Full-Mode Fine-Grained Alignment" FMFA arXiv
web search: 
web search: "Cross-modal Collaborative Representation Learning" "Text-to-Image Person Retrieval" "IJCAI" "CoRL"
web search: 
web search: https://www.sciencedirect.com/science/article/pii/S0031320325001769
web search: 
web search: "10.1016/j.patcog.2025.111489"
web search: 
web search: "Cross-modal Implicit Relation Reasoning and Aligning" "Text-to-Image Person Retrieval"
web search: 
web search: "View-Decoupled Transformer" "Aerial-Ground Person Re-Identification" CARGO arxiv
codex
你列出的实际是 6 个 PDF，其中 `Cross-Modal Full-Mode...` 的 ACM 版和 arXiv 版是同一篇工作。我两份都读了，不把它们伪装成两个不同创新。

**1. CoRL: Cross-modal Collaborative Representation Learning**
1. 触发观察：作者抓住两个具体问题。第一，TIPR 过去总在做图文“关联”，但文本本身比图像更干净，身份细节更集中，能补足视觉噪声。第二，同一身份的图文正样本并不等价，单视角对应文本是强正样本，跨视角同身份文本可能有外观差异，是弱正样本。
2. 重定义动作：他们把问题从“图文对齐”改写成“跨模态互补协作”，再把正样本从一个类别拆成“强正、弱正、负样本”的有界匹配问题。关键词是 `complementarity`、`virtual-text`、`collaboration`、`identity-bounded matching`。
3. 机制怎么长出来：如果文本有用但推理时图像没有配套文本，就自然推出 `VES`，从图像合成虚拟文本特征；如果虚拟文本要真有用，就要和真实文本特征对齐，再用 `CRCL` 让视觉特征和虚拟文本特征对原型的关系一致。强弱正样本的观察则直接推出 `IBM`，让强正相似度高于上界，弱正落在中间，负样本低于下界。
4. 证据闭环：关键不是总表涨点，而是 `IBM*` 去掉强弱正区分后掉点，证明“正样本内部有层级”这个重定义有意义；`VES` 的各个对齐损失消融证明虚拟文本不是随便加的噪声；检索可视化展示 CoRL 能把视觉相似但文本语义不同的人区分开。
5. reviewer 为什么买账：真正卖点不是一个新模块，而是两个视角：TIPR 不只是对齐，还能做模态互补；同身份正样本不是全等。机制和视角绑定比较紧，所以审稿人容易接受它不是“又加了一个 CLIP 分支”。

**2. FMFA ACM 版: Cross-Modal Full-Mode Fine-Grained Alignment**
1. 触发观察：作者指出两个旧方法盲点。全局匹配损失偏向 hard negative，却忽略“正样本没有排到前面”的 unmatched positive；局部注意力方法只给最终融合结果，无法检查局部词和图像块到底有没有对齐。
2. 重定义动作：他们把问题从“挖 hard negative”改成“修正 unmatched positive”，又把 fine-grained alignment 从隐式注意力改成“显式可检查的全模式对齐”。关键词是 `full-mode`、`explicit fine-grained alignment`、`unmatched positive pairs`、`verification capability`。
3. 机制怎么长出来：`A-SDM` 很自然，若真实正样本相似度低于 batch 中最高相似样本，就增大该正样本的拉近权重。`EFA` 也顺着“可检查”长出来，用词 token 和图像 patch 的稀疏相似矩阵显式聚合，再用 hard coding 和排序损失约束 joint embedding。
4. 证据闭环：表 6 分别消融 `A-SDM` 和 `EFA`，证明两者都不是装饰；参数实验里 `A-SDM` 的拉力过强会掉点，说明它确实在控制 unmatched positive 的拉近；稀疏相似图可视化展示哪些 patch 被保留，回应“显式对齐是否可检查”。
5. reviewer 为什么买账：它卖的是视角加轻量机制。创新性不在“做局部对齐”本身，而在把已有的 hard negative 叙事反过来，强调 missed positive，并把隐式局部交互改造成可观察的显式过程。

**3. FMFA arXiv 版**
1. 触发观察：和 ACM 版一致，核心仍是 SDM 关注 hard negative、忽略 unmatched positive，以及隐式局部对齐不可验证。
2. 重定义动作：同样把 TIPR 讲成“全局正样本修正 + 显式细粒度校验”的问题，而不是单纯再堆 attention。
3. 机制怎么长出来：`A-SDM`、`EFA`、`IRR`、`ID loss` 的组合与 ACM 版一致。arXiv 版没有提供一个独立的新机制。
4. 证据闭环：同样依赖三数据集的模块消融、稀疏相似图、top-k 检索展示。它的证据服务于同一个重定义。
5. reviewer 为什么买账：这份应视为同一工作的预印本，不应在我们调研里算两篇独立先例。对我们有用的是它的构造套路，不是重复计数。

**4. CAP: Cross-modality Average Precision Optimization**
1. 触发观察：VT-ReID 最终看排序指标，但训练常用 triplet、circle、center 这类局部度量损失，不能直接优化整体排序。Smooth-AP 虽然优化 AP，但在 VT-ReID 中会优先修正同模态高排名样本，反而忽略真正要解决的跨模态排序。
2. 重定义动作：作者把问题从“学一个跨模态嵌入空间”改成“直接优化跨模态 AP”。关键词是 `Cross-modality Average Precision`、`CAPV`、`CAPT`、`overall ranking order`、`cross-modality discrepancy`。
3. 机制怎么长出来：如果检索是 visible 查 thermal 和 thermal 查 visible，就自然得到 `CAPV` 和 `CAPT` 两个方向；如果跨模态距离太大影响 CAP，就加 `DMA`，只重点对齐那些距离较近、非模态变化较小的跨模态样本；全局局部特征只是把这个思想放到更强表征上。
4. 证据闭环：表 6 直接比较 CAP、Smooth-AP、FastAP、Blackbox AP、triplet、circle 等，证明不是“AP 损失都行”，而是跨模态定义更合适；表 4 拆 `CAPV/CAPT`，证明双向互补；表 5 证明 `DMA` 优于普通 KL 或平均权重；t-SNE 和检索图证明跨模态分布确实被拉近。
5. reviewer 为什么买账：这是典型“目标函数错配”式创新。它卖的主要是视角，不是结构。因为评价指标就是 AP，作者把 AP 改成 VT-ReID 专用的跨模态 AP，逻辑非常直接。

**5. CSGN: CLIP-driven Semantic Guidance Network**
1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
2. 重定义动作：作者把 CCReID 从“学习衣服不变视觉特征”改写成“用异质语义描述引导视觉特征摆脱衣服偏置”。关键词是 `semantic guidance`、`visual bottleneck`、`multi-description`、`RGB and gray-scale pseudo-text`、`heterogeneous semantic fusion`。
3. 机制怎么长出来：既然 RGB 文本偏颜色，灰度图可以弱化颜色并突出体型、姿态，就自然推出 `MDG` 同时生成 RGB 和灰度伪文本；既然两种文本各有偏向，就用 `HSF` 同时约束视觉特征；`VSS` 的 ResNet+ViT 融合是补强视觉编码，和核心重定义的绑定不如 MDG/HSF 紧。
4. 证据闭环：表 3 显示 `HSF` 从 51.7 提到 63.2，说明真正起作用的是异质语义融合；表 4 比 CoOp、CoCoOp 和带 Meta-Insert 的 MDG，证明不是普通 prompt；表 6 比 RGB-only、gray-only 和 HSF，证明“两种描述互补”这个说法成立。
5. reviewer 为什么买账：这篇主要卖“语义引导换衣 ReID”的视角，机制有点模块化堆叠，但 `HSF` 的消融很强，能支撑“RGB/灰度语义互补”这个核心故事。

**6. CVAF: CLIP-Based View-Consistent Alignment for AGPReID**
1. 触发观察：AGPReID 里航拍和地面视角差异极大。属性方法依赖人工或检测属性，但航拍低分辨率下属性不可靠；视角解耦方法看似合理，却可能把身体轮廓、结构过渡这类身份相关但带视角性的线索一起删掉。
2. 重定义动作：作者把问题从“显式去掉视角因素”改成“保持跨视角一致的身份语义，同时过滤视角噪声”。关键词是 `view-consistent alignment`、`Text Consistency Loss`、`Semantic Filtering Module`、`view-specific noise`。
3. 机制怎么长出来：如果同一身份的文本 token 应该稳定对齐多个视角图像，就自然推出 `TCL`，约束同一身份多视角图像与文本的相似度方差；如果文本 token 已经学到稳定身份语义，就用 `SFM` 通过文本引导的空间注意力和通道重权重过滤图像 patch。
4. 证据闭环：表 3、表 4 在 AG-ReID 和 CARGO 上分别消融 `TCL`、`SFM` 和组合，且 ViT/ResNet 都有效；更关键的是他们额外画了图文相似度均值和方差，证明 `TCL` 确实降低同一身份跨视角相似度波动；`SFM` 还做了空间、通道、残差和插入层位置消融。
5. reviewer 为什么买账：这篇卖的是“别粗暴解耦视角，而是做语义一致性约束”的视角。机制不复杂，但 `TCL` 的方差证据非常对题，能证明作者不是只在 CLIP-ReID 上加模块涨点。

这批论文共同套路很清楚：先找一个旧叙事里的盲区，比如 hard negative 之外的 unmatched positive、强弱正样本差异、视觉瓶颈、视角解耦误伤；再把它命名成一个更具体的新问题；最后设计一个几乎能从这个新问题直接推出的损失或分支。真正值得学的是：好创新不是“模块看起来新”，而是观察、重定义、机制、消融四件事能互相咬住。
tokens used
151,579
你列出的实际是 6 个 PDF，其中 `Cross-Modal Full-Mode...` 的 ACM 版和 arXiv 版是同一篇工作。我两份都读了，不把它们伪装成两个不同创新。

**1. CoRL: Cross-modal Collaborative Representation Learning**
1. 触发观察：作者抓住两个具体问题。第一，TIPR 过去总在做图文“关联”，但文本本身比图像更干净，身份细节更集中，能补足视觉噪声。第二，同一身份的图文正样本并不等价，单视角对应文本是强正样本，跨视角同身份文本可能有外观差异，是弱正样本。
2. 重定义动作：他们把问题从“图文对齐”改写成“跨模态互补协作”，再把正样本从一个类别拆成“强正、弱正、负样本”的有界匹配问题。关键词是 `complementarity`、`virtual-text`、`collaboration`、`identity-bounded matching`。
3. 机制怎么长出来：如果文本有用但推理时图像没有配套文本，就自然推出 `VES`，从图像合成虚拟文本特征；如果虚拟文本要真有用，就要和真实文本特征对齐，再用 `CRCL` 让视觉特征和虚拟文本特征对原型的关系一致。强弱正样本的观察则直接推出 `IBM`，让强正相似度高于上界，弱正落在中间，负样本低于下界。
4. 证据闭环：关键不是总表涨点，而是 `IBM*` 去掉强弱正区分后掉点，证明“正样本内部有层级”这个重定义有意义；`VES` 的各个对齐损失消融证明虚拟文本不是随便加的噪声；检索可视化展示 CoRL 能把视觉相似但文本语义不同的人区分开。
5. reviewer 为什么买账：真正卖点不是一个新模块，而是两个视角：TIPR 不只是对齐，还能做模态互补；同身份正样本不是全等。机制和视角绑定比较紧，所以审稿人容易接受它不是“又加了一个 CLIP 分支”。

**2. FMFA ACM 版: Cross-Modal Full-Mode Fine-Grained Alignment**
1. 触发观察：作者指出两个旧方法盲点。全局匹配损失偏向 hard negative，却忽略“正样本没有排到前面”的 unmatched positive；局部注意力方法只给最终融合结果，无法检查局部词和图像块到底有没有对齐。
2. 重定义动作：他们把问题从“挖 hard negative”改成“修正 unmatched positive”，又把 fine-grained alignment 从隐式注意力改成“显式可检查的全模式对齐”。关键词是 `full-mode`、`explicit fine-grained alignment`、`unmatched positive pairs`、`verification capability`。
3. 机制怎么长出来：`A-SDM` 很自然，若真实正样本相似度低于 batch 中最高相似样本，就增大该正样本的拉近权重。`EFA` 也顺着“可检查”长出来，用词 token 和图像 patch 的稀疏相似矩阵显式聚合，再用 hard coding 和排序损失约束 joint embedding。
4. 证据闭环：表 6 分别消融 `A-SDM` 和 `EFA`，证明两者都不是装饰；参数实验里 `A-SDM` 的拉力过强会掉点，说明它确实在控制 unmatched positive 的拉近；稀疏相似图可视化展示哪些 patch 被保留，回应“显式对齐是否可检查”。
5. reviewer 为什么买账：它卖的是视角加轻量机制。创新性不在“做局部对齐”本身，而在把已有的 hard negative 叙事反过来，强调 missed positive，并把隐式局部交互改造成可观察的显式过程。

**3. FMFA arXiv 版**
1. 触发观察：和 ACM 版一致，核心仍是 SDM 关注 hard negative、忽略 unmatched positive，以及隐式局部对齐不可验证。
2. 重定义动作：同样把 TIPR 讲成“全局正样本修正 + 显式细粒度校验”的问题，而不是单纯再堆 attention。
3. 机制怎么长出来：`A-SDM`、`EFA`、`IRR`、`ID loss` 的组合与 ACM 版一致。arXiv 版没有提供一个独立的新机制。
4. 证据闭环：同样依赖三数据集的模块消融、稀疏相似图、top-k 检索展示。它的证据服务于同一个重定义。
5. reviewer 为什么买账：这份应视为同一工作的预印本，不应在我们调研里算两篇独立先例。对我们有用的是它的构造套路，不是重复计数。

**4. CAP: Cross-modality Average Precision Optimization**
1. 触发观察：VT-ReID 最终看排序指标，但训练常用 triplet、circle、center 这类局部度量损失，不能直接优化整体排序。Smooth-AP 虽然优化 AP，但在 VT-ReID 中会优先修正同模态高排名样本，反而忽略真正要解决的跨模态排序。
2. 重定义动作：作者把问题从“学一个跨模态嵌入空间”改成“直接优化跨模态 AP”。关键词是 `Cross-modality Average Precision`、`CAPV`、`CAPT`、`overall ranking order`、`cross-modality discrepancy`。
3. 机制怎么长出来：如果检索是 visible 查 thermal 和 thermal 查 visible，就自然得到 `CAPV` 和 `CAPT` 两个方向；如果跨模态距离太大影响 CAP，就加 `DMA`，只重点对齐那些距离较近、非模态变化较小的跨模态样本；全局局部特征只是把这个思想放到更强表征上。
4. 证据闭环：表 6 直接比较 CAP、Smooth-AP、FastAP、Blackbox AP、triplet、circle 等，证明不是“AP 损失都行”，而是跨模态定义更合适；表 4 拆 `CAPV/CAPT`，证明双向互补；表 5 证明 `DMA` 优于普通 KL 或平均权重；t-SNE 和检索图证明跨模态分布确实被拉近。
5. reviewer 为什么买账：这是典型“目标函数错配”式创新。它卖的主要是视角，不是结构。因为评价指标就是 AP，作者把 AP 改成 VT-ReID 专用的跨模态 AP，逻辑非常直接。

**5. CSGN: CLIP-driven Semantic Guidance Network**
1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
2. 重定义动作：作者把 CCReID 从“学习衣服不变视觉特征”改写成“用异质语义描述引导视觉特征摆脱衣服偏置”。关键词是 `semantic guidance`、`visual bottleneck`、`multi-description`、`RGB and gray-scale pseudo-text`、`heterogeneous semantic fusion`。
3. 机制怎么长出来：既然 RGB 文本偏颜色，灰度图可以弱化颜色并突出体型、姿态，就自然推出 `MDG` 同时生成 RGB 和灰度伪文本；既然两种文本各有偏向，就用 `HSF` 同时约束视觉特征；`VSS` 的 ResNet+ViT 融合是补强视觉编码，和核心重定义的绑定不如 MDG/HSF 紧。
4. 证据闭环：表 3 显示 `HSF` 从 51.7 提到 63.2，说明真正起作用的是异质语义融合；表 4 比 CoOp、CoCoOp 和带 Meta-Insert 的 MDG，证明不是普通 prompt；表 6 比 RGB-only、gray-only 和 HSF，证明“两种描述互补”这个说法成立。
5. reviewer 为什么买账：这篇主要卖“语义引导换衣 ReID”的视角，机制有点模块化堆叠，但 `HSF` 的消融很强，能支撑“RGB/灰度语义互补”这个核心故事。

**6. CVAF: CLIP-Based View-Consistent Alignment for AGPReID**
1. 触发观察：AGPReID 里航拍和地面视角差异极大。属性方法依赖人工或检测属性，但航拍低分辨率下属性不可靠；视角解耦方法看似合理，却可能把身体轮廓、结构过渡这类身份相关但带视角性的线索一起删掉。
2. 重定义动作：作者把问题从“显式去掉视角因素”改成“保持跨视角一致的身份语义，同时过滤视角噪声”。关键词是 `view-consistent alignment`、`Text Consistency Loss`、`Semantic Filtering Module`、`view-specific noise`。
3. 机制怎么长出来：如果同一身份的文本 token 应该稳定对齐多个视角图像，就自然推出 `TCL`，约束同一身份多视角图像与文本的相似度方差；如果文本 token 已经学到稳定身份语义，就用 `SFM` 通过文本引导的空间注意力和通道重权重过滤图像 patch。
4. 证据闭环：表 3、表 4 在 AG-ReID 和 CARGO 上分别消融 `TCL`、`SFM` 和组合，且 ViT/ResNet 都有效；更关键的是他们额外画了图文相似度均值和方差，证明 `TCL` 确实降低同一身份跨视角相似度波动；`SFM` 还做了空间、通道、残差和插入层位置消融。
5. reviewer 为什么买账：这篇卖的是“别粗暴解耦视角，而是做语义一致性约束”的视角。机制不复杂，但 `TCL` 的方差证据非常对题，能证明作者不是只在 CLIP-ReID 上加模块涨点。

这批论文共同套路很清楚：先找一个旧叙事里的盲区，比如 hard negative 之外的 unmatched positive、强弱正样本差异、视觉瓶颈、视角解耦误伤；再把它命名成一个更具体的新问题；最后设计一个几乎能从这个新问题直接推出的损失或分支。真正值得学的是：好创新不是“模块看起来新”，而是观察、重定义、机制、消融四件事能互相咬住。
