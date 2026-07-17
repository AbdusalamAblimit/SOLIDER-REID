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
session id: 019ef835-b891-7103-b339-88fcacaae89d
--------
user
你是 ReID 创新方法论拆解员 15。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(15)==
HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则和 PDF 技能确认读取方式，然后逐篇用 `pdftotext` 抽方法页。当前环境是只读，所以我只会读取和分析，不会改任何文件。
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

codex
PDF 技能只要求优先用 Poppler 工具读取或渲染；这里用户指定了 `pdftotext -f 1 -l 8`，我会按这个口径抽文本。目录里列出的 6 篇目标文件都存在，用户说“15”但实际名单是 6 篇，我会按名单逐篇拆。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
arXiv:2501.05851v1 [cs.CV] 10 Jan 2025

Identity-aware Feature Decoupling Learning for
Clothing-change Person Re-identification
1st Haoxuan Xu

2nd Bo Li

3rd Guanglin Niu∗

School of Artificial Intelligence
Beihang University
Beijing, China
xhaoxuan@buaa.edu.cn

School of Artificial Intelligence
Beihang University
Beijing, China
boli@buaa.edu.cn

School of Artificial Intelligence
Beihang University
Beijing, China
beihangngl@buaa.edu.cn

Abstract—Clothing-change person re-identification (CC Re-ID)
has attracted increasing attention in recent years due to its
application prospect. Most existing works struggle to adequately
extract the ID-related information from the original RGB images.
In this paper, we propose an Identity-aware Feature Decoupling
(IFD) learning framework to mine identity-related features.
Particularly, IFD exploits a dual stream architecture that consists
of a main stream and an attention stream. The attention stream
takes the clothing-masked images as inputs and derives the
identity attention weights for effectively transferring the spatial
knowledge to the main stream and highlighting the regions with
abundant identity-related information. To eliminate the semantic
gap between the inputs of two streams, we propose a clothing
bias diminishing module specific to the main stream to regularize
the features of clothing-relevant regions. Extensive experimental
results demonstrate that our framework outperforms other
baseline models on several widely-used CC Re-ID datasets.
Index Terms—Clothing-change Person Re-Identification, IDbased Knowledge Transfer, Clothing Bias Diminishing Module

I. I NTRODUCTION
Person re-identification (Re-ID) aims at matching the same
pedestrian across different cameras [1]. Most existing methods
predominantly utilize global representations for matching,
which are only applicable to pedestrians without clothing
change [2]–[11]. Whereas, a person changing clothing is a
widespread phenomenon in practice. Consequently, the more
challenging task clothing-change person re-identification (CC
Re-ID) has received significant attention recently, which attempts to associate the same pedestrian with changed clothes.
Driven by different motivations, existing methods for CC
Re-ID can be broadly classified into two categories: 1) data
augmentation methods and 2) biometrics-based methods. Data
augmentation methods argue that the scale of current CC ReID datasets is insufficient to fully capture identity-related (IDrelated) features and attempt to augment training data [12]–
[14]. However, these methods are subject to the quality of
the generated virtual data, and their effectiveness is typically
difficult to interpret. On the other hand, biometrics-based
methods aim to explicitly capture stable biometrics features.
* corresponding author. This work was supported by the National Natural
Science Foundation of China (No. 62376016).

All ID-related (e.g. face, hairstyle)

Decouple

ID-unrelated

All ID-related (e.g. face, hairstyle)

ID-unrelated

ID 1 Cloth A

ID 1 Cloth B

Head region
Body region

Decouple

matched
ID-related

ID-related

unmatched

Fig. 1. Illustration of the ID-related features distribution. The head regions
contain purely ID-related features, while ID-related features and ID-unrelated
features are coupled in the body regions.

These methods can be further subdivided into multi-modality
and single-modality methods. The multi-modality methods
exploit extra modalities as auxiliary information to highlight
ID-related features. Wang et al. enhance ID-related features
by extracting face features [15]. Methods such as SPT+ASE,
CESD, FASM, and GI-ReID learn ID-related features utilizing
sketches, keypoints, human contours, and gait information,
respectively [16]–[19]. Additionally, some studies argue that
the 3D shape contains rich ID-related information and attempt
to leverage 3D information [20], [21]. In contrast, single
modality methods such as CAL, RCSANet and AIM attempt to
directly mine ID-related features by clothing adversarial loss,
clothing status awareness and causality, respectively [22]–[24].
In the real world, humans can recognize their acquaintances
through various identity clues (e.g., face, hairstyle, body shape,
height, gait) even if these individuals wear unfamiliar clothing.
However, certain clues, such as height and gait, cannot be
reliably estimated from a single image, and the remaining
clues are distributed across both clothing-irrelevant regions
(i.e. head regions) and clothing-relevant regions (i.e. body
regions). As illustrated in Fig. 1, the head regions primarily
contain ID-related features, such as the face, hairstyle and head
contour. In contrast, the body regions exhibit a mixture of

Clothing Bias Diminishing
Resize

Main Stream

Clothing Weight 𝑾𝑾𝒄𝒄

Clothing Feature Maps 𝑭𝑭𝒄𝒄 Clothing Features 𝒇𝒇𝒄𝒄
Inference

BN + GAP

Global Feature Maps 𝑭𝑭𝒈𝒈

Refined Feature Maps 𝑭𝑭𝒓𝒓𝒓𝒓 Refined Features 𝒇𝒇𝒓𝒓𝒓𝒓

Parsing

Max Pool

IKT Module

…
…

…

Attention Stream

Concat

Attention Feature Maps 𝑭𝑭𝒂𝒂

Necessary during both training and inference

Attention Features 𝒇𝒇𝒂𝒂

Predicted Identities

Only Necessary during training

ℒ𝐶𝐶𝐶𝐶𝐶𝐶

𝑚𝑚
ℒ𝐼𝐼𝐼𝐼

conv

ID-based Knowledge Transfer

BN + GAP
Backbone 𝑔𝑔𝑎𝑎

Predicted Identities

Average Pool
𝑭𝑭𝒂𝒂

…
…

…

Backbone 𝑔𝑔𝑚𝑚

𝝍𝝍𝒎𝒎

BN + GAP

ID-based Weight 𝑾𝑾𝑰𝑰

𝑎𝑎
ℒ𝐼𝐼𝐼𝐼

Element-wise matrix product

Fig. 2. Overview of our proposed IFD, which consists of an attention stream and a main stream. The attention stream learns a weight matrix with high values
for identity-relevant regions and low values for identity-irrelevant regions at the feature level. The main stream aims to learn ID-related features under the
guidance of the attention stream and clothing bias diminishing module.

identity-related and identity-unrelated features, which can lead
to erroneous match results. Thus, the key challenges of CC
Re-ID can be concluded: 1) leveraging only specific part of
ID-related features can not comprehensively represent an
individual, 2) clothing-relevant regions actually couple the
implicit ID-related features, which is hard to be extracted
for CC Re-ID. Although existing approaches mitigate the
effect of clothing changes to some extent, they still lack
effective constraints to ensure that the model consistently
focuses on all critical ID-related features. Specifically, each of
the aforementioned multi-modality methods tends to capture
only one category of ID-related features, leaving other critical
discriminative ID-related features underutilized. As for singlemodality methods, they can only extract coarse ID-related
features by overlooking the semantics of body parts.
To enhance the ID-related features for CC Re-ID, we
propose a novel Identity-aware Feature Decoupling (IFD)
learning framework that consists of an attention stream and
a main stream. Both streams employ the same backbone for
feature extraction but operate independently without shared
weights. The attention stream processes clothing-masked images, while the main stream takes the original images as input.
To ensure that the main stream focuses comprehensively on
the regions implying identity information, we incorporate an
ID-based Knowledge Transfer (IKT) module between the two
streams. Additionally, to decouple the ID-related features from
the clothing-relevant region, we introduce a Clothing Bias

Diminishing (CBD) module, which helps model the consistent
clothing features with regard to the same individual.
In summary, our contributions are listed as follows:
• We are the first to propose a dual-stream identity-attention
model that effectively compels the network to focus
comprehensively on the regions containing distinctive
identity information.
• An effective CBD module is developed to maintain the
consistency of clothing features for the same individual.
• Extensive experiments demonstrate that our method
achieves state-of-the-art on several clothing-change ReID datasets including PRCC and LTCC [18], [19].
II. M ETHODOLOGY
IFD aims to comprehensively mine the ID-related features
to address the CC Re-ID issue. As illustrated in Fig. 2, the
framework begins by extracting a clothing-masked image M
from the input image I using an off-the-shelf human parsing
network [25]. The clothing-masked image contains the critical
head and contour information, which is essential for learning
ID-related features, but it discards all the color information
of body parts, which is critical to re-identify persons in the
conventional Re-ID scenarios. To robustly extract ID-related
features, we implement a dual-stream framework with an IDbased knowledge transfer module to guide the main stream
toward comprehensively emphasizing ID-related regions.
The IKT module can help locate the ID-related regions.
However, the semantic gap between the inputs of the two

streams may introduce error. The body regions of inputs
merely contain shape information in the attention stream,
while clothing-relevant features can still couple with IDrelated features in body parts of main stream. Thus, the
IKT module might inadvertently amplify the influence of IDunrelated features while enhancing ID-related features, which
can limit the overall performance of our model. To guarantee
that the final features are exclusive of clothing color and
texture, we introduce a CBD module.

To restrain the contribution of clothing at the feature level,
we propose Clothing Contrastive Loss LCCL . Let i be the
index of an arbitrary sample in a batch, the LCCL is defined:
i
p
N
e(fc ·fc /τ )
1 X 1 X
wp log
LCCL = −
P (f i ·fcj /τ )
p
i
N i=1 |Pi |
e(fc ·fc /τ ) +
e c

p∈Pi

j∈Ni

(
wp =

A. ID-based Knowledge Transfer
Motivated by the spatial attention mechanisms that are
widely used in computer vision to capture fine-grained local
features and precisely locate task-relevant regions, we attempt
to enhance the attention toward ID-related regions for the CC
Re-ID task [26], [27]. However, it is naive to directly utilize
spatial attention due to the difficulty of learning effective
spatial attention weights without auxiliary supervision. To
address this, we propose an ID-based Knowledge Transfer
module that facilitates the learning of robust and effective
spatial attention weights.
As shown in Fig. 2, we design a mutual learning framework
with two stream, gm (.) and ga (.) denotes the backbone of
main stream and attention stream, respectively. The original
image I is passed through gm (.) to extract the global feature
maps Fg . Simultaneously, the masked image M is fed into
ga (.) to obtain the attention feature maps Fa . Taking Fa as
input, our IKT module derives ID-based attention matrix WI :
WI = σ (Wconv ∗ [mp (Fa ) ; ap (Fa )])

(1)

where mp denotes max pooling along the channel, ap denotes
average pooling along the channel, ∗ denotes convolution
operation, Wconv indicates the weights of convolution filters,
and σ (.) denotes the sigmoid function. The ID-based attention
matrix is then applied to the global feature Fg as formulated:
Frg = WI ⊗ Fg

(2)

where ⊗ denotes the element-wise matrix product, Frg denotes
refined feature maps.
B. Clothing Bias Diminishing Module
To decouple the ID-related features in clothing-relevant
regions, we propose a Clothing Bias Diminishing module. As
illustrated in the top of Fig. 2, we estimate a fine-grained mask
ψ m for clothing-relevant parts from the original image I. The
pixel value of ψ m can be formulated as:
(
1 , if Ii,j ∈ C
m
ψ(i,j) =
(3)
0 , Otherwise
where C denotes the set of clothing part categories. Then we
resize the ψ m to match the dimensions of WI , resulting in
Wc , which is used to perform an element-wise matrix product
with Frg to derive the clothing-related feature maps Fc .
By applying batch normalization and global average pooling
operation to Fc , we obtain the clothing features fc .

(4)
1
T

1

, if cp ̸= ci
, Otherwise

(5)

where N is the batch size, Pi (Ni ) denotes the set of samples
with the same ID as (different ID from) i, and τ is a
temperature parameter. T ∈ R+ is a variable parameter, and cx
denotes the clothing label. Equation (5) serves as an incentive
function, encouraging the network to focus more on pairs with
the same ID label but different clothing labels during training.
Existing Re-ID methods typically employ PK sampling
strategy during training, where K samples per ID and P × K
samples per batch, but PK sampling overlooks the appearance
diversity. However, in this paper, we encourage the network to
learn ID-related features through our proposed loss function
LCCL , and its effectiveness hinges on diversity in both persons
and appearances within a batch. RAS sampling strategy can
maintain appearance diversity by sampling A appearances of
each person and fixed K images of each appearance in a batch
[28]. whereas, this sampling strategy may discard a significant
number of samples for appearances with large proportion by
ignoring the maldistribution of different appearances. Thus, we
modify RAS by replacing the fixed number of each appearance
with a proportion number to accelerate the effect of LCCL .
C. Training
We adopt ResNet-50 pre-trained on ImageNet as our backbone [29], [30]. During the training stage, we first train the
attention stream with LaID to obtain effective feature maps
with abundant ID-related information, and then we jointly train
the dual streams under the guidance of the overall loss:
a
Lall = Lm
ID + LID + λ · LCCL

(6)

where λ denotes the weight of the LCCL , which is set to 1.0.
III. E XPERIMENTS
A. Datasets and Evaluation Protocols
We mainly evaluate our proposed method on two popular
CC Re-ID benchmark datasets PRCC and LTCC. We employ
two frequently-used metrics rank-1 and mAP to perform the
evaluation results. Three kinds of test settings are defined
as following: (i) general mode (both clothing-change and
clothing-consistent ground truth samples are used to evaluate
accuracy), (ii) same-clothing mode (only clothing-consistent
ground truth samples are used to evaluate accuracy), (iii)
clothing-change mode (only clothing-change ground truth
samples are used to evaluate accuracy). In terms of PRCC,
we report the evaluation results of same-clothing mode and
clothing-change mode. As for LTCC, the accuracies for general mode and clothing-change mode are provided.

TABLE I
C OMPARISON R ESULTS WITH STATE - OF - THE - ART METHODS ON THE PRCC AND LTCC DATASET (%), WHERE ’ SKETCH ’, ’ SIL’, ’3D’, ’ POSE ’,
’ PARSING ’,’ AUG ’ DENOTE THE CONTOUR SKETCHES , SILHOUETTES , 3D SHAPE , KEYPOINTS , HUMAN PARSING AND DATA AUGMENTATION ,
RESPECTIVELY. T HE BEST RESULTS ARE INDICATED BY RED .
Method
PCB [4]
IANet [2]
SPT+ASE [19]
CESD [18]
RCSANet [23]
AFD-Net [13]
FASM [16]
GI-ReID [17]
CAL [22]
3DSL [20]
3DInvarReID [21]
AIM [24]
CCFA [12]
IFD

Modality
RGB
RGB
sketch
RGB+pose
RGB
RGB+aug
RGB+pos+sil
RGB+sil
RGB
RGB+3D
RGB+3D
RGB
RGB+aug
RGB+parsing

PRCC
SC Mode
CC Mode
rank-1
mAP rank-1
mAP
99.8
97.0
41.8
38.7
99.4
98.3
46.3
45.9
64.2
34.4
100
97.2
50.2
48.6
95.7
42.8
98.8
54.5
80.0
33.3
100
99.8
55.2
55.8
51.3
51.6
52.5
100
99.9
57.9
58.3
99.6
98.7
61.2
58.4
99.0
98.2
65.3
61.7

LTCC
General Mode
CC Mode
rank-1
mAP
rank-1
mAP
65.1
30.6
23.5
10.0
63.7
31
25.0
12.6
71.4
34.3
26.2
12.4
73.2
40.8
38.5
16.2
63.2
29.4
23.7
10.4
74.2
40.8
40.1
18.0
31.2
14.8
37.8
16.7
76.3
41.1
40.6
19.1
75.8
42.5
45.3
22.1
81.4
65.3
64.3
42.3

TABLE II
T HE ABLATION STUDIES OF IFD ON PRCC AND LTCC.
Method
baseline
w/ IKT
w/ CBD
IFD w/ CL
IFD

PRCC
rank-1
mAP
24.3
11.2
50.3
49.2
51.2
45.3
58.9
52.3
65.3
61.7

LTCC
rank-1
mAP
16.4
9.2
50.7
24.6
49.6
23.4
57.8
30.6
64.3
42.3

B. Comparison With State-of-the-art Methods
We compare our IFD with two traditional Re-ID methods, namely PCB, IANet, and eleven clothing-change ReID methods, including SPT+ASE, CESD, RCSANet, AFDNet, FASM, GI-ReID, CAL, 3DSL, 3DInvarReID, AIM, and
CCFA. As is illustrated in Tab. I, our proposed IFD achieves
superior performance with 19.0% / 4.1% absolute improvements in rank-1 on LTCC/PRCC of CC Mode, illustrating the
effectiveness of our method that effectively mines comprehensive ID-related information whereas other methods typically
capture only a single category of ID-related information. In
the SC mode of PRCC, the 99.0% rank-1 of IFD is close to
saturation but inferior to some methods. The reason is that our
IFD aims to capture clothing-irrelevant features but there are
only clothing-consistent ground truth samples in this mode.
C. Ablation Studies
The effectiveness of components. To verify the effectiveness of each contribution in our framework IFD, we reproduce
a baseline method (ResNet-50) that only remains Lm
ID of IFD
for training. As shown in Tab. II, the two ablated models
that integrate IKT or CBD to the baseline individually both
improve the performance significantly compared with the baseline. Furthermore, the whole model IFD introducing both IKT
and CBD into the baseline obtains superior performance than
the two ablated models. These results illustrate that both IKT
and CBD could effectively facilitate our model to decouple
clothing-relevant features and enhance ID-related features.

0.43

ID1

ID1

ID2

(a) The triple hard sample

ID1

0.92

ID1

0.56

0.89

ID2

(b) The performance of baseline

ID1

ID1

ID2

(c) The performance of IFD

Fig. 3. An intuitive comparison of the baseline and our model IFD specific to
hard triples, namely the positive sample has absolutely different outfit with the
anchor while the negative sample dressing similar with the anchor, together
with their visualization results derived by grad-cam [31].

Comparison of CCL and standard contrastive loss.
We compare our clothing contrastive loss with widely used
supervised contrastive loss [32]. As shown in Tab. II, IFD with
contrastive loss namely IFD w/CL is superior to the baseline.
Besides, IFD with clothing contrastive loss namely IFD further
outperforms IFD w/CL significantly, which illustrates the effectiveness of the designed clothing bias diminishing strategy.
D. Visualization
As the similarity between sample pairs and the heatmap
visualization results shown in Fig. 3, the baseline believes the
negative sample is more similar to the anchor since it focuses
more attention on ID-unrelated features such as the texture of
clothing. In contrast, our IFD can consistently highlight the
critical ID-related head and human contour features, facilitating the correct matching results.
IV. CONCLUSION
In this paper, we propose a novel Identity-aware Feature
Decoupling learning framework for the CC Re-ID task. Our
approach could effectively transfer the ID-based spatial knowledge into the main stream guided by the ID-related features
derived from the attention stream and diminish the clothing
bias to enhance the robustness and discriminate feature extraction capability under clothing variations. Extensive experiments demonstrate that our method achieves state-of-the-art
performance on several widely used CC Re-ID datasets.

R EFERENCES
[1] Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang,
and Qi Tian, “Scalable person re-identification: A benchmark,” in
Proceedings of the IEEE international conference on computer vision,
2015, pp. 1116–1124.
[2] Ruibing Hou, Bingpeng Ma, Hong Chang, Xinqian Gu, Shiguang Shan,
and Xilin Chen, “Interaction-and-aggregation network for person reidentification,” in Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2019, pp. 9317–9326.
[3] Wei Li, Xiatian Zhu, and Shaogang Gong, “Harmonious attention
network for person re-identification,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2018, pp. 2285–
2294.
[4] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang, “Beyond
part models: Person retrieval with refined part pooling (and a strong
convolutional baseline),” in Proceedings of the European conference on
computer vision (ECCV), 2018, pp. 480–496.
[5] Tianlong Chen, Shaojin Ding, Jingyi Xie, Ye Yuan, Wuyang Chen, Yang
Yang, Zhou Ren, and Zhangyang Wang, “Abd-net: Attentive but diverse
person re-identification,” in Proceedings of the IEEE/CVF international
conference on computer vision, 2019, pp. 8351–8361.
[6] Jing Xu, Rui Zhao, Feng Zhu, Huaming Wang, and Wanli Ouyang,
“Attention-aware compositional network for person re-identification,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 2119–2128.
[7] Jinrui Yang, Wei-Shi Zheng, Qize Yang, Ying-Cong Chen, and Qi Tian,
“Spatial-temporal graph convolutional network for video-based person
re-identification,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2020, pp. 3289–3299.
[8] Tong Xiao, Hongsheng Li, Wanli Ouyang, and Xiaogang Wang, “Learning deep feature representations with domain guided dropout for person
re-identification,” in Proceedings of the IEEE conference on computer
vision and pattern recognition, 2016, pp. 1249–1258.
[9] Zhizheng Zhang, Cuiling Lan, Wenjun Zeng, Xin Jin, and Zhibo
Chen, “Relation-aware global attention for person re-identification,” in
Proceedings of the ieee/cvf conference on computer vision and pattern
recognition, 2020, pp. 3186–3195.
[10] Hao Luo, Youzhi Gu, Xingyu Liao, Shenqi Lai, and Wei Jiang, “Bag
of tricks and a strong baseline for deep person re-identification,” in
Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition workshops, 2019, pp. 0–0.
[11] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang,
“Transreid: Transformer-based object re-identification,” in Proceedings
of the IEEE/CVF international conference on computer vision, 2021,
pp. 15013–15022.
[12] Ke Han, Shaogang Gong, Yan Huang, Liang Wang, and Tieniu Tan,
“Clothing-change feature augmentation for person re-identification,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2023, pp. 22066–22075.
[13] Wanlu Xu, Hong Liu, Wei Shi, Ziling Miao, Zhisheng Lu, and Feihu
Chen, “Adversarial feature disentanglement for long-term person reidentification.,” in IJCAI, 2021, pp. 1201–1207.
[14] Zhedong Zheng, Xiaodong Yang, Zhiding Yu, Liang Zheng, Yi Yang,
and Jan Kautz, “Joint discriminative and generative learning for person
re-identification,” in proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2019, pp. 2138–2147.
[15] Fangbin Wan, Yang Wu, Xuelin Qian, Yixiong Chen, and Yanwei Fu,
“When person re-identification meets changing clothes,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition
workshops, 2020, pp. 830–831.
[16] Peixian Hong, Tao Wu, Ancong Wu, Xintong Han, and Wei-Shi Zheng,
“Fine-grained shape-appearance mutual learning for cloth-changing person re-identification,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2021, pp. 10513–10522.
[17] Xin Jin, Tianyu He, Kecheng Zheng, Zhiheng Yin, Xu Shen, Zhen
Huang, Ruoyu Feng, Jianqiang Huang, Zhibo Chen, and Xian-Sheng
Hua, “Cloth-changing person re-identification from a single image with
gait prediction and regularization,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2022, pp.
14278–14287.
[18] Xuelin Qian, Wenxuan Wang, Li Zhang, Fangrui Zhu, Yanwei Fu, Tao
Xiang, Yu-Gang Jiang, and Xiangyang Xue, “Long-term cloth-changing

person re-identification,” in Proceedings of the Asian Conference on
Computer Vision, 2020.
[19] Qize Yang, Ancong Wu, and Wei-Shi Zheng, “Person re-identification
by contour sketch under moderate clothing change,” IEEE transactions
on pattern analysis and machine intelligence, vol. 43, no. 6, pp. 2029–
2046, 2019.
[20] Jiaxing Chen, Xinyang Jiang, Fudong Wang, Jun Zhang, Feng Zheng,
Xing Sun, and Wei-Shi Zheng, “Learning 3d shape feature for textureinsensitive person re-identification,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2021, pp. 8146–
8155.
[21] Feng Liu, Minchul Kim, ZiAng Gu, Anil Jain, and Xiaoming Liu,
“Learning clothing and pose invariant 3d shape representation for
long-term person re-identification,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 19617–19626.
[22] Xinqian Gu, Hong Chang, Bingpeng Ma, Shutao Bai, Shiguang Shan,
and Xilin Chen, “Clothes-changing person re-identification with rgb
modality only,” in Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2022, pp. 1060–1069.
[23] Yan Huang, Qiang Wu, JingSong Xu, Yi Zhong, and ZhaoXiang Zhang,
“Clothing status awareness for long-term person re-identification,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 11895–11904.
[24] Zhengwei Yang, Meng Lin, Xian Zhong, Yu Wu, and Zheng Wang,
“Good is bad: Causality inspired cloth-debiasing for cloth-changing
person re-identification,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2023, pp. 1472–1481.
[25] Peike Li, Yunqiu Xu, Yunchao Wei, and Yi Yang, “Self-correction for
human parsing,” IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 44, no. 6, pp. 3260–3271, 2020.
[26] Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al., “Spatial
transformer networks,” Advances in neural information processing
systems, vol. 28, 2015.
[27] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon,
“Cbam: Convolutional block attention module,” in Proceedings of the
European conference on computer vision (ECCV), 2018, pp. 3–19.
[28] Mengmeng Liu, Zhi Ma, Tao Li, Yanfeng Jiang, and Kai Wang, “Longterm person re-identification with dramatic appearance change: Algorithm and benchmark,” in Proceedings of the 30th ACM International
Conference on Multimedia, 2022, pp. 6406–6415.
[29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep
residual learning for image recognition,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2016, pp. 770–
778.
[30] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei,
“Imagenet: A large-scale hierarchical image database,” in 2009 IEEE
conference on computer vision and pattern recognition. Ieee, 2009, pp.
248–255.
[31] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra, “Grad-cam: Visual
explanations from deep networks via gradient-based localization,” in
Proceedings of the IEEE international conference on computer vision,
2017, pp. 618–626.
[32] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong
Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan,
“Supervised contrastive learning,” Advances in neural information
processing systems, vol. 33, pp. 18661–18673, 2020.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

147

HPRNet: Human Parsing Reconstruction With
Non-Local Multi-Scale Perception Network for
Cloth-Changing Person Re-Identification
Mingfu Xiong , Longlong Ge , Ruimin Hu , Senior Member, IEEE,
Khan Muhammad , Senior Member, IEEE, Sambit Bakshi , Senior Member, IEEE,
Javier Del Ser , Senior Member, IEEE, Xiaokang Yang , Fellow, IEEE, and Bin Sheng , Member, IEEE

Abstract—Cloth-changing Person Re-Identification (CC-ReID)
is a challenging data modeling task that involves identifying
specific pedestrians wearing different outfits. Existing methods primarily focus on altering clothing color and directly
reconstructing appearance to extract features independent of
the clothes. Real pedestrians differ in height, body shape, etc.
Such methods are prone to losing the intrinsic information
of the original sample (i.e., the person identity) owing to the
absence of contextual phenomena (e.g., texture structure and
local correlation), which decreases the recognition performance.
To address this problem, we propose a framework called HPRNet,
or “Human Parsing Reconstruction with Non-Local Multi-Scale
Perception Network,” which includes a non-local weighted multiscale perception (NWMP) module and a parsing reconstruction
exploration (PRE) module. In particular, the proposed NWMP
module effectively captures the global receptive field of a sample
and obtains a contextual correlation between non-neighboring
Received 9 April 2025; revised 24 June 2025; accepted 31 July 2025.
Date of publication 8 August 2025; date of current version 22 January 2026.
This work was supported in part by the National Natural Science Foundation
of China under Grant 62272298 and Grant U1903214 and in part by the
Natural Science Foundation of Hubei Province under Grant 2021CFB568. The
work of Javier Del Ser was supported by the Basque Government (Eusko
Jaurlaritza) through the Consolidated Research Group MATHMODE under
Grant IT1456-22. This article was recommended by Associate Editor Z. Yang.
(Corresponding authors: Khan Muhammad; Bin Sheng.)
Mingfu Xiong is with the School of Computer Science and Artificial
Intelligence, Wuhan Textile University, Wuhan 430200, China, and also with
the School of Cyber Science and Engineering, Wuhan University, Wuhan
430072, China (e-mail: xmf2013@whu.edu.cn).
Longlong Ge is with the School of Computer Science and Artificial
Intelligence, Wuhan Textile University, Wuhan 430200, China (e-mail:
ge longlong@163.com).
Ruimin Hu is with the School of Cyber Science and Engineering, Wuhan
University, Wuhan 430072, China (e-mail: hrm@whu.edu.cn).
Khan Muhammad is with the Visual Analytics for Knowledge Laboratory
(VIS2KNOW Lab), Department of Applied Artificial Intelligence, School of
Convergence, College of Computing and Informatics, Sungkyunkwan University, Seoul 03063, Republic of Korea (e-mail: khan.muhammad@ieee.org).
Sambit Bakshi is with the Visual Surveillance Laboratory, Department
of Computer Science and Engineering, National Institute of Technology
Rourkela, Rourkela, Odisha 769008, India (e-mail: bakshisambit@ieee.org).
Javier Del Ser is with TECNALIA, Basque Research and Technology
Alliance (BRTA), 48160 Derio, Spain, and also with Department of Mathematics, the University of the Basque Country (UPV/EHU), 48940 Leioa,
Spain (e-mail: javier.delser@tecnalia.com).
Xiaokang Yang is with the MOE Key Laboratory of AI, School of Electronic Information and Electrical Engineering, Shanghai Jiao Tong University,
Shanghai 200240, China (e-mail: xkyang@sjtu.edu.cn).
Bin Sheng is with the Department of Computer Science and Engineering,
School of Electronic Information and Electrical Engineering, Shanghai Jiao
Tong University, Shanghai 200240, China (e-mail: shengbin@sjtu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3597210

pixels within the sample image. The PRE module was used
to achieve a more accurate reconstruction of human body
components with a clothing parsing model to better distinguish
features related to or unrelated to clothes. Extensive experiments
were conducted on CC-ReID public datasets (LTCC, PRCC, and
CCVID) to demonstrate the effectiveness and competitiveness of
the proposed method with state-of-the-art (SOTA) baselines for
this complex modeling task.
Index Terms—Cloth-changing person re-identification, human
reconstruction, non-local exploration, multi-scale perception,
visual scene understanding.

I. I NTRODUCTION
LOTH-CHANGING Person Re-identification (CC-ReID)
aims to recognize specific pedestrians with different
outfits in nonoverlapping surveillance systems at various times
and locations. This paradigm has received widespread attention in the multimedia and computer vision communities [1],
[2]. Due to the unique nature of this task, it is widely applied
in related fields such as visual scene understanding [3], [4],
target tracking [5], criminal investigation [6], [7], to mention
a few. Meanwhile, owing to objective factors such as lighting,
clothing changes, and hardware conditions, CC-ReID remains
challenging and has not yet been fully resolved [2], [6], [8].
Existing CC-ReID-based techniques can be classified into
two main categories: (1) data-driven disentanglement methods
[9], [10] and (2) feature-driven disentanglement methods [11],
[12], [13], [14]. The first category employs a simulation
process to evoke alterations in attire and generate a spectrum
of appearances. This approach entails the random alteration
of pixels within the designated area of the garment, thereby
reducing reliance on specific attributes such as color and
texture. Despite the efficacy of these techniques in simulating
changes in attire and generating samples, their performance
is constrained by the absence of intricate details inherent in
authentic garments and the introduction of unnatural variations in appearance [9], [15]. Feature-driven disentanglement
methods focus on introducing additional branches to capture
clothing-independent features specifically and harness them
to learn from each other in CC-ReID tasks. Although these
methods can identify some discriminative features, they ignore
clothing-related attributes such as human skin, bags, and
shoes [12].

C

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.

148

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

Fig. 1. (a) Overall diagram illustrating the general design approach followed
by existing CC-ReID methods, which typically emphasize clothing-irrelevant
feature extraction and pedestrian reconstruction in a relatively coarse manner.
Methods following this approach often result in lower predictive performance
due to classification bias and the model’s focus on non-ID regions that
may appear related to identity. In contrast, our proposed multi-scale perception HPRNet framework depicted in (b) incorporates a non-local weighted
multi-scale perception module and a parsing reconstruction module, both
contributing to improved performance on the CC-ReID task.

Recently, novel approaches have been proposed as potential
solutions to the CC-ReID problem by adopting one of the
two general modeling strategies explained above. For example,
Yang et al. [16] proposed a causality-based Auto-Intervention
Model (AIM) to guarantee the learning of the feature representation of pedestrians, which is unaffected by clothing
bias. Cui et al. [1] presented a disentanglement DCR-ReID
framework called Deep Component Reconstruction, which
discriminates the clothes-irrelevant or relevant features in a
controllable strategy for CC-ReID. They exploited human
component reconstruction in a deeply assembled manner to
improve the performance of previously disentangled features.
Although promising progress has been made in the CC-ReID
task [17], [18], existing methods mainly focus on changing
the color of clothes or performing appearance reconstruction
directly to distinguish features related and unrelated to clothes,
as shown in Fig. 1 (a). Real pedestrians have differences
in height, body shape, and other characteristics, and such
methods lack real sample cases, which causes the generated
features to easily lose essential information (such as person
identity and ID) of the original samples, resulting in insufficient discriminative performance of learned features unrelated
to clothes. Although the work in [1] performs loss summation
by analyzing pedestrian appearance parsing and reconstructing
human body parts, the method described directly decomposes
feature maps from the backbone network using channellevel splitting. It employs human-component region parsing
features as supervisory signals to reconstruct different parts
(contours, clothing, and non-clothing). However, the absence
of the actual ground truth in the feature decomposition process
hinders contextual texture structure segmentation and local
correlation, leading to interference from subtle noise such as
pedestrian background contour features, ultimately reducing
the recognition performance.
To address the aforementioned problems, this study proposes a modeling framework coined as “Human Parsing

Reconstruction Network with Non-Local Multi-Scale Perception Network” (HPRNet), which incorporates a non-local
weighted multi-scale perception (NWMP) module and a parsing reconstruction exploration (PRE) module for CC-ReID.
The general workflow of the proposed framework is illustrated
in Fig. 1 (b). Specifically, to obtain the contextual correlation
between non-neighboring pixels of the same pedestrian appearance, the NWMP module is proposed to effectively capture
the global receptive field via a multi-scale progressive learning strategy for features unrelated to clothing, which differs
from changing the color of clothes directly. Unlike existing
human component-based reconstruction methods, the PRE
module enhances human component reconstruction by leveraging clothing parsing and a multi-local component generation
strategy, effectively distinguishing between clothing-related
and unrelated features.
Extensive experiments were conducted to evaluate the
performance of the proposed method, demonstrating that it
performs better than existing CC-ReID algorithms. In addition,
human appearance nonlocal exploration and human component
multiscale perception schemes have been proven to significantly improve the accuracy of CC-ReID (Rank-01 and mAP)
compared to existing CC-ReID methods over the LTCC [19],
PRCC [20], and CCVID [11] datasets. Our experiments also
examined the contribution of each element of the proposed
HPRNet approach to the performance gains through ablation
studies and visualizations.
The main contributions of this study are summarized as
follows.
• We propose the HPRNet framework, which comprises
a non-local weighted multi-scale perception (NWMP)
module and a parsing reconstructed exploration (PRE)
module for the CC-ReID task.
• The proposed NWMP module effectively captures the
global receptive field through a multi-scale progressive
learning strategy for features unrelated to clothing. The
PRE module incorporates clothing parsing and a human
local component generation strategy to achieve more
accurate human component reconstruction and better distinguish between clothing-related and unrelated features.
• Extensive experiments are conducted on the public CCReID datasets LTCC, PRCC, and CCVID to demonstrate
the competitive accuracy of our proposed method. Additionally, ablation studies and visualizations are performed
to verify the contribution of each module to the overall
performance of HPRNet.
The rest of the manuscript is organized as follows: firstly
Section II briefly revisits existing CC-ReID methods, including
data-driven disentanglement methods, feature-driven disentanglement methods, and Human Reconstruction-based CC-ReID
methods. In Section III, we describe the proposed HPRNet
method. Section IV presents and discusses the experimental
results and ablation study. Finally, concluding remarks and
future work are presented in Section V.
II. R ELATED W ORK
In this section, related work on CC-ReID is reviewed.
First, we describe the data-driven disentanglement methods for

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.

XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK

CC-ReID (Subsection II-A). Feature-driven disentanglement
strategies are revisited in Subsection II-B. Finally, we revise
the human reconstruction-based methods in Subsection II-C,
followed by a short statement of the novelty of this work
compared to existing works (Subsection II-D).
A. Data-Driven Disentanglement CC-ReID Methods
As mentioned previously, CC-ReID methods based on datadriven disentanglement focus on entailing random alterations
within the designated outfit area to reduce reliance on color
or texture attributes. The primary strategy employed by these
methods involves increasing the amount of dataset at the data
level, which includes expanding data templates and synthesizing clothing-relevant features [9], [10], [21]. Specifically,
a semantic-guided sampling strategy was proposed in [10]
to enforce the learning of clothing-independent features by
recognizing pedestrians’ outfit appearances (e.g., tops and
pants) and sampling features from other individuals’ images
for CC-ReID. The method for this same task presented in [22]
incorporated identity-aware Mixstyle and graph enhancement
modules to construct variable clothing-based fine-grained style
transformation features and cross-domain style transfer-based
enhanced samples. CCPG [23] was used to enhance data
diversity of different pedestrians wearing the same clothes and
the same one with different clothes. Other similar methods
have been proposed in the literature [24], [25] addressing the CC-ReID task by embracing different data-driven
disentanglement-based strategies.

149

images utilizing generative models (such as GANs [37] and
diffusion models [38]), combining them with the original
pedestrian instances to perform contrastive training on the
reconstructed images, to reduce the interference of clothingrelevant features [13], [14], [39]. In addition, other methods
such as [40], [41], [42] adopted the idea of pedestrian reconstruction to address the CC-ReID task.
D. Contribution
Although existing CC-ReID technologies have achieved
promising performance, they lack realistic guidance on generated samples, leading to a potentially corrupted feature space
and uncontrollable results. By contrast, we propose HPRNet, a novel framework that integrates a non-local weighted
multi-scale perception module and a parsing-generation
reconstructed exploration module for the CC-ReID task.
By leveraging component parsing reconstruction and a
multi-scale non-local weighted attention interaction strategy,
HPRNet effectively enhances CC-ReID performance and mitigates the performance degradation caused by the feature
decomposition process present in state-of-the-art approaches.
III. P ROPOSED HPRN ET F RAMEWORK
In this section, we introduce the proposed HPRNet framework for CC-ReID tasks. A specific sub-model constituting the
overall framework of this study is introduced. The loss function and differences from existing methods are also described
separately, with the details as follows:

B. Feature-Driven Disentanglement CC-ReID Methods

A. Overview

Differently than the previous category, feature-driven methods for the CC-ReID problem hinge on distinguishing
clothing-related and clothing-unrelated attributes by using
feature extraction [2], [11], [12], [26]. An adversarial feature
learning method text description-based, called DIFFER [11],
has been proposed for the separation of identity features.
Specifically, the feature space is partitioned into multiple nonoverlapping subspaces. Gradient inversion is then employed
to distinguish identity-relevant features from non-biological
ones, informing the model to better solve the CC-ReID task.
A 3D InvarReID framework was designed in [27] to disentangle and reconstruct 3D outfit body shapes for the CC-ReID
task. DLAW [3] incorporated an adaptive cloth-changed region
localization strategy and a modeling scheme that captures the
correlation of cloth changes at both the image and feature
levels. Other related strategies [28], [29], [30], [31], [32]
rely on feature-driven disentanglement, employing a learning
approach that is unaffected by clothing features. Overall, these
methods distinguish related and unrelated cues to clothing at
the feature level.

In this section, we first establish the mathematical notations
and formally state the modeling problem addressed by the
CC-ReID methods (Section III-B). Subsequently, the proposed
HPRNet framework is introduced, as shown in Fig. 2. Subsequently, the nonlocal weighted multi-scale perception module
(Section III-C) and the human body parsing reconstructed
exploration module (Section III-D) are described. Finally, the
optimization process and the differences between the proposed
method and other baselines are discussed in Sections III-E
and III-F, respectively.

C. Human Reconstruction-Based CC-ReID Methods
Recently, human reconstruction learning-based solutions,
which are essentially feature-driven methods, have gained
widespread attention within the CC-ReID community [33],
[34], [35], [36]. These methods mainly reconstruct pedestrian

B. Notation and Problem Statement
Given a ReID gallery dataset, which is denoted as G =
{(g1 , y1 ), . . . , (ga , ya ), . . . , (gN , yL )}, where N is the size of the
gallery set, and L is the amount of different identities in the
set. The probe set q p contains M pedestrian images, which are
the targets to be queried. In the specific ReID task, the goal
was to accurately retrieve pedestrian images from a gallery
set that matched a probe image (a query image). It involves
sorting all images in the gallery set based on their similarity
scores with the probe set, which is given by
a∗ = arg min d(φ(q; θ), φ(ga ; θ)),

(1)

a=1,...,N

where d(·, ·) denotes the measurement function calculated
using the similarity between the two feature vectors. φ(·; θ) is a

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.

150

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

Fig. 2. Diagram showing the design of the HPRNet framework. The framework comprises three key components: parsing, reconstruction, and feature extraction
models. These components are integrated into the NWMP and PRE modules. The NWMP module is designed to capture the global receptive field of each
person, while the PRE module further distinguishes clothing-related and clothing-independent features through a parsing-reconstruction pipeline.

parametric function whose parameters are collectively denoted
by θ. We aim to find the image ga in the gallery set that is most
similar to probe q p , that is, a∗ , such that distance d minimizes
the image index in the gallery set.
However, the CC-ReID task poses a greater challenge than
its original ReID form, as it necessitates distinguishing pedestrians wearing the same clothing but belonging to different
identities, or the same person wearing different clothes. In
this case, the model must be able to differentiate between
those who are dressed similarly but are different people.
Mathematically, the differences imprinted by the formulation
of the CC-ReID task to ReID give rise to the following
objectives:
min

M X
M
X

(d(φ(q p ; θ), φ(ga ; θ)) − d(φ(qn ; θ), φ(ga ; θ))),

(2)

p=1 n=1

where p and a are the p-th and a-th samples in each batch,
respectively. q p and ga are pedestrians with the same identity
but wearing different clothing. qn denotes dressed clothes
similar to ga but with different identities. M is the size of
each mini-batch.
C. Non-Local Multi-Scale Perception Module
As mentioned at the end of Section II, existing methods mainly focus on pedestrian clothing color change and
appearance transfer, but ignore the local contextual correlation
features of human appearance, resulting in decreased model
performance. This section proposes a human appearance nonlocal multi-scale perception module involving a weighted

non-local perception mechanism (WNPM) and a dual-scale
progressive learning strategy (DPLS). Details are provided in
the following sections.
1) WNPM: Specifically, the weighted non-local perception,
inspired by the attention mechanism and AGW-ReID [43],
[44], captures both prominent human appearance features
and local region correlations by assigning weights to each
pedestrian region. This strategy is devoted to obtaining the
local contextual receptive field of pedestrian appearance and
determining its multi-perspective receptive field. Weights are
proposed to associate the region’s relationships (such as
similarity or contextual correlation), which determine the
importance of each person’s feature points by calculating their
relationship with others. This study constructed an attentionmap-based weight learning, which reflects the interactive
relationships between different regions. The contribution of
each feature region depends not only on its own, but also on
its relationship with other feature regions, thus achieving an
understanding of global receptive fields (such as the overall
posture and shape contour of pedestrians). Meanwhile, it suppresses the local features that may change owing to changes
in perspective or occlusion and improves the accuracy and
robustness of the model under harsh conditions. Specifically,
the procedure followed by the WNPM module to produce
map-based weights wi j , where i and j denote i-th and j-th
regions, respectively, is given by
wi j =

T
1 X fi · f j
,
T
k fnle k2
i=1, j=1

fnlp =

T
X

wi j · fki j ,

i=1, j=1

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.

(3)

XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK

where fi and f j are the features of the two nearest neighboring
local regions. fnle denotes the output of the Non-local Encoder
with the ResNet50 [45] as the backbone. T denotes the size
of the local region. fki j denotes the k-th feature map in region
(i, j) and fnlp is the nonlocal perception feature.
2) DPLS: To acquire the multi-scale pedestrian features
and achieve perceptual learning from local to whole, we
propose a dual-scale progressive learning strategy (DPLS),
which is based on the non-local perception patterns. Specifically, feature fnlp from WNPM is split into K equal blocks
first. Then, the max-pooling operations of 2 × 2 and 4 × 4
were implemented in each case. Next, we combine the maxpooling results of 2 × 2 and 4 × 4 separately to obtain two
different scales of local features. Then, fnlp is included to
form three different scales of pedestrian features to achieve
a progressive feature learning strategy. Finally, a 1 × 1 global
pooling operation was performed on each of the three scale
features mentioned above to complete the combination and
classification. This process can be summarized as follows.
!
K 
X
fk − 2
,
(4a)
fk2 = concat
s
k=1
!
K 
X
fk − 4
fk4 = concat
,
(4b)
s
k=1


fcls = concat conv fnlp , 1 × 1
+ conv ( fk2 , 1 × 1) + conv ( fk4 , 1 × 1))),

(5)

where fk denotes the k-th feature block. bc is the pooling
operation and s is the stride. concat() is the feature concatenation. fk2 and fk4 are the concatenated features after the max
pooling results with kernel sizes 2 × 2 and 4 × 4, respectively.
conv( f , 1 × 1) denotes a 1 × 1 convolution operation on f
for normalization to a uniform dimension. fcls is a multi-scale
progressive learning feature (the dimensions of fcls are 6144
and are averaged from fk2 , fk4 and fnlp ).
After this feature is learned, the multi-classification loss Lcla
is expressed as
Lcla = −

L−1
X

yi j log(ybi j ),

(6)

j=0

where yi j is the ground truth when sample i belongs to
class j, and ybi j is the predicted value.
In addition, to preserve the consistency of the pedestrian identity and clothing, we also exploit the identity (ID)
loss Lid as in [11], which is given by
!
N
X
y(xi , li )
Lid = −
log PNid
,
(7)
j=1 y(xi , l j )
i=1
where N denotes the total number of samples in the training
set. y(xi , li ) is the probability that the model
PNid predicts that
sample xi belongs to its true label li .
j=1 y(xi , l j ) is the
normalized sum of the predicted probabilities for all possible
identity categories l j .

151

Partly inspired by CAL [11], we extract the invariant features for clothing variations by penalizing the model’s ability
to predict clothing attributes, which is described as follows:
min Lclo (CϕC (gθ (xi ), yCi )),

(8)

ϕ

Lclo = −

N
X

e( fi ·ϕyi c/τ)
log PLC
( fi ·ϕ j /τ)
j=1 e
i=1

!
,

(9)

where CϕC (•) is the clothes classifier and CϕC (gθ (xi ) denotes
the predicted clothes. yCi is the ground truth for the clothing
label. gθ (xi ) denotes the result of fi after l2 -normalization, and
ϕ j is the output of the weights for j-th clothes classifier after
l2 -normalization. N is the size of the input batch. LC is the
volume of the category data for clothing. τ is a temperature
parameter used to control the model training.
Subsequently, we disrupt the clothing features of different
identities and randomly select those belonging to other identities to generate adversarial samples and improve the robustness
of the model.
f˜i = fi − ficlo + fĩclo ,
(10)
where i represents the current sample, ĩ represents a noncurrent sample, and fi represents the current sample feature.
ficlo denotes the clothing-related feature of the current sample,
and fĩclo is the non-current sample’s clothing-related feature by
disrupting the above clothing-related (unrelated) features and
reclassifying them Many adversarial samples can be simulated
to help the model learn and adapt to different input changes,
thereby improving the model’s generalization ability. Similar
to Eq. (9) as follows:
!
N
˜
X
e( fi ·ϕyi c/τ)
adv
.
(11)
Lclo = −
log PLC ˜
( fi ·ϕ j /τ)
j=1 e
i=1
D. Parsing Reconstructed Exploration Module
To acquire clothing-independent features better, we also
propose a pedestrian-based appearance parsing reconstructed
exploration model, which is accomplished using a multiregion
pedestrian local reconstruction and clothing parsing model.
Specifically, the multiregion pedestrian local reconstruction
model was inspired by [1] and [11] but differs from them.
Based on the basic structure of the human body, we split
the output of the reconstructed encoder using ResNet50 as
the backbone into three local regions (head, body, and lower
limbs), which are represented as multilocal components in
Fig. 2. The multilocal maps are then fed into the generation model [1] and projected onto a deconvolution model
[46] to obtain the corresponding local region reconstructed
maps. Finally, these maps are concatenated to form the final
reconstructed feature, fr . In addition, we computed the reconstruction loss Lor with the original pedestrian feature fo to
ensure reconstruction accuracy, which is expressed as
v
u N
uX
(12)
L = t ( f − f )2 ,
ri

or

oi

i=1

where fri and foi are the ith reconstructed and original pedestrian features, respectively. N is the sample size.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.

152

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

To extract pedestrian clothing color-independent features
more effectively, we adopted a cloth parsing module to obtain
pedestrian contour features independent of clothing color.
Specifically, we fed the output of the Parsing Encoder with
the ResNet50 as the backbone into the Clothing Parsing Model
[47], which involved channel, position, and color parsing networks. It completes the pedestrian contour profile parsing from
three aspects: pedestrian feature channels, human contextual
position, and clothing color. The parsed pedestrian sample is
represented by f p in Fig. 2. As before, to ensure the accuracy
of the pedestrian sample parsing, we also used it to calculate
the loss with the original feature map fo , and the parsing loss
Lop is represented by Eq. (13).
v
u N
uX
(13)
Lop = t ( f p j − fo j )2 ,
j=1

where f p j and fo j are the jth parsing and original pedestrian
features, respectively. N is the sample size as mentioned above.
E. Overall Loss Function of the Framework
Once we complete the non-local multi-scale perception and
parsing reconstructed exploration model learning, the total loss
function L is represented as
L = Lid + αLcla + Lclo + βLadv
clo + Lor + Lop .

(14)

where Lid and Lclo are the pedestrian identity and clothing category loss, respectively. Lcla is the multi-classification loss for
nonlocal perception loss. Lor and Lop are the reconstructed and
parsing losses, respectively. α and β are the hyperparameters
that balance the relative importance of every loss term in the
overall training of the framework.
F. Differences to Existing CC-ReID Methods
In this section, we discuss the differences between existing
methods and our method. Part of the inspiration for the
proposed framework comes from existing DCR-ReID [1],
AGW-ReID [43], and CAL [11]. However, the proposed
HPRNet method differs substantially from these state-of-theart techniques, as described below.
• Different design principles: The DCR-ReID method
relies on component reconstruction and deep learningbased fusion for CC-ReID. The CAL method leverages clothes-based adversarial loss to extract clothingirrelevant features directly from original RGB images,
while the AGW-ReID method incorporates a non-local
operation within the ResNet backbone to retrieve the
weights of such features. In contrast, HPRNet employs
a combined strategy of non-local weighted multi-scale
perception and human appearance parsing reconstruction
to model contextual similarity (correlation) for nonneighboring pixels and clothing-independent features in
CC-ReID. Unlike AGW-ReID, HPRNet treats the nonlocal attention operation as an independent module,
integrating it with a multi-scale perception module to
enhance model flexibility. Additionally, the proposed
framework is optimized through a loss function that

addresses both global modeling and local fine-grained
feature learning, improving the robustness of fine-grained
feature representation in complex scenes and making it
particularly suitable for ReID tasks involving clothing
changes.
• Different framework structures: Although all the above
methods utilize multi-branch strategies for learning and
training clothing-irrelevant features, their approaches differ significantly. On one hand, DCR-ReID employs
component reconstruction, clothing identification, and
person identity branches. On the other hand, CAL
incorporates clothing category loss and person identity branches for CC-ReID. Lastly, the AGW-ReID
method integrates a non-local attention model within
the backbone network to enhance pedestrian feature
learning. Our proposed framework, however, adopts a
structurally distinct architecture. It leverages a non-local
weighted perception mechanism and a multi-scale progressive learning strategy to effectively capture the global
receptive field and progressively learn local-to-whole
pedestrian appearance features. Furthermore, HPRNet
integrates a parsing reconstruction auxiliary model, forming a framework that, while inspired by state-of-the-art
methods, introduces novel elements to effectively address
the CC-ReID task.
Algorithm 1 Training Procedure for the HPRNet Framework
N
Require: ResNet50 model G; Training dataset D = {xi , ji }i=1
;
p
Number of images N;Number of identities N ; Maximum
epoch F; Threshold t0 (0 < t0 < F)
Initialization: Model weights θG ; Hyper-parameters α, β
1: for t ← 1 to F do
2:
for i ← 1 to N do
3:
Extract image features fi0 = G(xi )
4:
Calculate Weight result fi = I( fi0 ) via Eq. (3)
5:
B, C, H, W ← fi
6:
Stripe dim S ← C  (max pooling kernel size)
7:
Segment fi with S size
8:
for z ← 1 to S do
z
9:
2 × 2 max-pooling extract fk2
via (4)
z
10:
4 × 4 max-pooling extract fk4 via Eq. (4)
11:
Concat in the channel dimension
12:
end for
13:
Concat fk2 , fk4 with origin feature fi via Eq. (5)
14:
Compute Lcla via Eq. (6)
15:
Reconstruction Yiv via Eq. (12), (13)
16:
Compute Lor via Eq. (12)
17:
Compute Lop via Eq. (13)
18:
Compute Lclo Eq. (9)
19:
Shuffle fi cloth region compute Ladv
clo via Eq. (11)
20:
Compute LID via Eq. (7)
21:
L = Lid + αLcla + Lclo + βLadv
clo + Lor + Lop
22:
end for
23: end for
Output: Optimal weights θG∗
• Different updating strategies: In terms of model optimization, DCR-ReID and CAL utilize a component

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.

XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK

reconstruction loss term and a clothes-based adversarial loss term to represent the person’s identity, while
AGW-ReID relies on ID loss as its primary optimization objective. In contrast, HPRNet introduces additional
optimization strategies beyond clothing and identity
losses, incorporating parsing reconstruction and multiclassification losses to enhance performance for the
CC-ReID task.
In summary, essential differences exist between the proposed method and existing algorithms in terms of technical
principles, network structures, and optimization strategies.
IV. E XPERIMENTAL R ESULTS AND D ISCUSSION
In this part, the experimental results and related analyses are
described to validate the effectiveness of black the proposed
method. The details of this process are provided below.
A. Experimental Setup
1) Datasets: We conduct a series of experiments on
three real-world benchmark CC-ReID datasets, which include
LTCC [19], PRCC [20], and CCVID [11]. The details of
these datasets are summarized as follows.
• The LTCC [19] dataset is one of the most widely used
CC-ReID datasets and contains 17, 119 images of 152
different identities. On average, each identity has five
different outfits, with the number of outfit changes ranging
from 2 to 14.
• The PRCC [20] dataset contains 33, 698 images covering 221 identities and primarily focuses on evaluating
algorithm performance under moderate clothing change
conditions. Although it has limitations in scene diversity
(collected using only three cameras) and clothing variations (only two clothing changes per person), it remains
a valuable resource for developing and testing pedestrian
recognition algorithms designed to adapt to real-world
complexity.
• The CCVID [11] dataset is a large-scale video CC-ReID
dataset, which contains 2, 856 video sequences from 226
different identities. It has a total of 347, 833 images with 2
to 5 different sets of clothing for each identity. The length
of the sequences in the CCVID dataset ranged from 27
to 410 frames, with an average length of 122 frames. It
includes 834 sequences, which are used as the query set,
and 1074 sequences that form the library set in the testing
phase.
2) Evaluation Protocol: As usual, there are also three
main settings for evaluating the CC-ReID task, which include:
(i) General Setting (abbreviated as General, both clothingchanging and unchanging ground truth instances are utilized
to evaluate performance), (ii) Clothing-Changing Setting
(denoted as CC, only clothing-changing ground truth instances
were utilized to evaluate the performance); and (iii) SameClothing Setting (denoted as SC, only clothes-consistent
(unchanging) ground truth instances were utilized to evaluate
the performance). Similar to [1] and [11], the performances
of both general and CC were evaluated on the CCVID and
LTCC datasets. The SC and CC datasets are reported in the

153

PRCC dataset. As is typical in ReID experiments, we also
exploited the cumulative matching characteristics (CMC) [11]
and mean average precision (mAP), which are commonly used
as evaluation metrics, with the rank metric being demonstrated
by Rank@1.
3) Implementation Details: We utilize ResNet50 [45],
which has been pre-trained on ImageNet [48], as the backbone
of our model. The proposed HPRNet method was implemented on a hardware platform that included NVIDIA RTX
4080 SUPER GPUs. The software environment was based
on Python 3.8.10, and Ubuntu 22.04 operating system. The
versatility of Python supports scientific computing and deep
learning. The version of PyTorch used is 2.1.0, with its rich
library functions for building, optimizing, and assessing the
proposed CC-ReID framework. Following [11], we exploited
a data augmentation operation, which included random cropping, erasing, and flipping. The input images were resized
to 384 × 192. Global pooling is a combination of the maxand average-pooling strategies. Similar to [11], we employed
a two-stage optimization strategy to decouple the clothingindependent features. It optimized the loss function Lid +Lclo +
Ladv
clo + Lcla in the first stage and the full loss function L in
the second stage during the training phase. During testing, we
removed the cloth-relevant branch to calculate the similarity
between the query image and images in the gallery.
4) Parameter Setting: The learning rate (lr) of the Adam
optimizer is set as 3.5×10−4 , and the lr drops 10% of its initial
value every 20 epochs for PRCC and LTCC datasets, whereas
it descends every 40 epochs for the CCVID dataset. The
temperature parameter τ is set to 1/16. The hyperparameters α
and β are set to specific values for the LTCC (α = 0.2, β = 0),
PRCC (α = 1, β = 1), and CCVID (α = 0.2, β = 1) datasets.
In addition, the batch size was set to 64, which included 8
instances of 8 people with different IDs in the PRCC and
LTCC datasets. For the CCVID dataset, the batch size was set
to 8, and each batch contained 4 instances of 2 people with
different IDs.
B. Comparison With State-of-the-Art Methods
1) Comparisons on the LTCC and PRCC Datasets: We
summarize the performance of our presented HPRNet method
and that of the latest competitors on the LTCC and PRCC
datasets, which is shown in Table I. In the comparison benchmark, 18 methods were used: HACNN [49], PCB [50], IANet
[51], CESD [19], SPT+ASE [20], RCSANet [52], 3DSL [53],
FSMA [12], GI-ReID [54], CAL [11], AIM [16], DCR-ReID
[1], AFL [55], JIAMGP [22], FIRe2 [32], LIFTCAP [42],
MCSC-CAL [56], CLIP3DReID [57], RFFR-Net [58]. It can
be shown that our HPRNet model has achieved competitive results compared with the latest methods. Although our
evaluation metrics are lower than those of the most recent
methods available, they have the most balanced performance
and are relatively stable overall. Specifically, our proposed
method (HPRNet) aims to achieve balanced performance
across different datasets (LTCC and PRCC) and evaluation
settings (General, CC and SC). Although some methods (e.g.,
JIAMGP [22], FIRe2 [32]) have higher CC scores on specific
datasets, they tend to rely on dataset-specific strategies or focus

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.

154

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

TABLE I
C OMPARISON W ITH THE L ATEST S TATE - OF - THE -A RT M ETHODS ON THE LTCC AND PRCC DATASETS . T HE O NES W ITH THE B EST P ERFORMANCE
A RE H IGHLIGHTED IN B OLD B LACK , AND S IMILAR R EPRESENTATIONS A RE A LSO F OUND IN THE OTHER TABLES

TABLE II
C OMPARISON W ITH THE L ATEST M ETHODS ON THE CCVID DATASET

effective for CC-ReID tasks. For the comparison benchmark,
we introduced seven latest methods: AP3D [59], TCLNet
[60], CAL [11], DCR-ReID [1], SEMI [61], CLIP3DReID
[57], ASGL [62]. These comparisons demonstrated that the
proposed approach achieved competitive results in multiple
dimensions.
C. Ablation Study and Sensitivity Analysis

excessively on clothing-independent features. This makes their
performance in other settings (e.g., General or SC) or on
different datasets less consistent. By contrast, by extracting
a combination of clothing-independent and fine-grained features, HPRNet achieves a stable generalisation performance
across datasets and settings, which is reflected in the overall
improvement in Rank@1 and mAP for the LTCC and PRCC
datasets. Compared to the LTCC dataset, the PRCC dataset
has different characteristics in terms of the distribution and
difficulty of garment variation. For example, PRCC places
more emphasis on comparison tests with significant garment
changes and well-controlled environments, which are more
favourable for certain methods specifically optimized for this
scenario. In practice, however, clothing variation is only one of
many factors (e.g., pose, lighting, and background) that affect
re-identification performance.
2) Comparisons on the CCVID Dataset: To demonstrate
the performance of the proposed HPRNet method, we also
conduct experiments on the CCVID dataset for the CC-ReID
task, which is shown in Table II. Our technique is also

To demonstrate the performance of each component of
the proposed HPRNet framework, we conducted a series
of ablation studies and sensitivity analyzes, which included
the following: 1) effectiveness of the parameter setting,
2) performance of the sub-branch, 3) impact of loss functions,
4) performance of different feature scaling ratios, 5) computational complexity of the model, and 6) visualization results
on the above-mentioned datasets. The details are as follows.
1) Effectiveness of the Parameters Setting: Our proposed
HPRNet framework introduces two vital training hyperparameters, α and β, which are directly related to the performance of
the CC-ReID. To demonstrate the impact of these parameters
on the overall framework effectiveness, we conducted a series
of experiments on the mainstream CC-ReID datasets (LTCC
and PRCC), as shown in Fig. 3 and Fig. 4. From the figures,
it can be seen that the values of the parameters α and β affect
the performance of the model. Specifically, the parameters α
and β were set to specific values for the LTCC (α = 0.2,
β = 0) and PRCC (α = 1, β = 1) to obtain the best
performance.
2) Performance of the Sub-Branch: As described above,
our proposed HPRNet framework mainly includes NWMP and
PRE modules to capture the global receptive field and to distinguish features related and unrelated to clothing, respectively.
To verify the performance of the single-branch (module), we

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 107ms:
U activation function that sets
similarity values less than 0 to 0 and the mean operation,
respectively.
Different from the fine-grain alignment that needs to refine
the details of each node-level feature, by compacting the
feature nodes (e.g., classify each node and merge similar
nodes into a single node), the representation ability of the
feature nodes can be further enhanced, and the computational
complexity of the feature alignment process can be reduced.
So, we introduce a coarse-grain graph attention alignment
(CGA) to improve the efficiency of generating intermediate
features and enhance the semantic association of the middle
features.
To begin with, we classify each feature node of Rvis
L into Y
regions (i.e., head, arm, torso, leg, and so on) by a learnable
classifier, and achieve the total region probability P2 RN ⇥Y ,
that is,
P = Classifier(Rvis
L ) = [p1 , p2 , · · · , pN ],

(6)

where Classifier(·) is a fully connected layer which consists of
learnable parameters ✓cls 2 RC⇥Y ; pn 2 R1⇥Y represents the

(8)

where Mask(·) sets the value of the n-th feature node to 0
when the corresponding idxn,z is 0; otherwise keep the same
value as before.
Finally, we perform an average operation on feature nodes
vis,L
vis,L
N ⇥C
Ovis,L
= [ovis,L
to obtain the
z
z,1 , oz,2 , · · · , oz,N ] 2 R
vis,L
z-th compact representation Ôz
, i.e.,
PN
vis,L
n=1 oz,n
Ôvis,L
=
,
(9)
z
vis,L
Nz
+✏
where ovis,L
represents the n-th feature node of Ovis,L
;
z,n
z
vis,L
Nz
means the number of non-zero feature nodes in Ovis,L
;
z
✏ is a very small value (i.e., e 5 ) added to the denominator
for numerical stability. Similarly to Eq. (9), we can get the
other Y -1 compact feature nodes and thus the final compact
Y ⇥C
feature node set can be represented as Õvis
.
L 2R
Moreover, to improve the efficiency of feature alignment,
and explore contextual semantic associations in different
modalities and ranges. Hence, we design a coarse-grain graph
attention (CGA) to align Õvis
and Õir
L
S , and the aligned
C
vis,L
compact feature node set Ōir,S 2 RN ⇥ r can be formulated
as
vis
ir
Ōvis,L
ir,S = CGA(ÕL , ÕS )
ir
T
ir
= ✓c,u ReLU(Softmax(Õvis
L ✓q,c (ÕS ✓k,c ) ))(ÕS ✓v,c ),
(10)
C

C

C

where ✓q,c 2 RC⇥ r , ✓k,c 2 RC⇥ r , ✓v,c 2 RC⇥ r and
✓c,u 2 RN ⇥Y are the linear transformations. Unlike FGA,
the CGA allocates all feature nodes to the limited Y regions,
without suppressing low similarity feature regions to ensure
the efficiency of the coarse-grain alignment.
Based on the above, we employ an effective and efficient
fine-coarse graph attention alignment (FCGA) module to align
different modalities and ranges of feature nodes from nodelevel and region-level, respectively, as follows
ir
vis
ir
FCGA(Rvis
L , RS ) = FGA(RL , RS )+
ir
CGA(Rvis
L , RS ).

(11)

Page 7 of 26

IEEE Transactions on Circuits and Systems for Video Technology

PREPRINT SUBMITTED TO TCSVT, JULY 2025

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60

7

Through the FCGA module with a small number of paN ⇥C
rameters, a middle feature Mvis
can be obtained
L 2 R
ir
by aligning Rvis
with
the
other
three
features Rir
L
S , RL , and
Rvis
in
R,
that
is,
S
vis
ir
vis
ir
Mvis
L = (FCGA(RL , RS ) + FCGA(RL , RL )+
vis
vis
FCGA(Rvis
L , RS ))✓u + RL ,

(12)

C

where ✓u 2 R r ⇥C represents the linear transformation.
We can get the other reliable middle features similar to
Eq. (12). Hence, we obtain the middle feature set R̃ =
vis
ir
ir
{Mvis
L , MS , ML , MS }.
D. Hierarchical Middle-Feature Agent Learning (HMAL) Loss
We generate agents of middle features at each modalityshared feature extraction stage to assist the network in learning
a better common feature space and reducing the modality discrepancy hierarchically. For example, we generate the middlefeature agent A1 2 R1⇥1⇥C1 for Stage 1 of the network based
on the above middle feature set R̃, that is,
A1 = ✓m,1 (Avg(R̃)),

(13)

where Avg(·) is the average pooling operation; ✓m,1 is the
learnable parameter that adjusts the feature size to 1 ⇥ 1 ⇥ C1
(C1 means the number of channels of the features in Stage 1).
Then, we introduce the modality discrepancy reduction
(MDR) loss to reduce the difference between the pooled VIS
feature (F̃1,vis ), IR feature (F̃1,ir ) and middle-feature agent
(A1 ) in Stage 1 of the network, as follows
1 $A1
LF̃
M DR = L1 (F̃1,vis + F̃1,ir , 2A1 ) + L1 (F̃1,vis , F̃1,ir ).
(14)

where L1 (·) represents the L1 distance; F̃1,vis 2 R
and F̃1,ir 2 R1⇥1⇥C1 mean the VIS feature F1,vis and IR
feature F1,ir after the pooling operation, respectively.
Similarly to Eq. (13), we can achieve other middle-feature
agents (i.e., A2 , A3 , and A4 ), and the total MDR loss can be
expressed as:
1⇥1⇥C1

F̃1 $A1
F̃2 $A2
F̃3 $A3
F̃4 $A4
DR
LM
HM AL = LM DR + LM DR + LM DR + LM DR .
(15)

With the agent of the middle features, our method can
learn discriminative features from all the network stages and
hierarchically enhance feature representations to achieve a
reliable common feature space between different modalities.
The existing VI-ReID methods [1], [40] follow low-to-high
feature extraction, which ignores the interaction of features at
different stages. The features of the low stage contain more
detailed information, while the features of the high stage have
rich semantic relationships. In this subsection, we will use
the BFE loss to build the mutual interaction bridge and thus
achieve the bi-directional enhancement between the features
of different stages. The interactions of the high-to-low and the
low-to-high can improve the ability of the network to capture
discriminative features. In other words, the low-stage network
can focus on detailed features guided by semantic relationships
from the high-stage, and the high-stage network can enhance

the semantic relationship by using detailed information from
the low-stage.
We take the bi-directional enhancement between the features
of Stage 1 (S1 ) and Stage 2 (S2 ) as an example. First, we
adopt the pooling operation on the features of S1 and S2
to obtain the feature representations F̃1 2 R1⇥1⇥C1 and
F̃2 2 R1⇥1⇥C2 (C2 is the number of channels of the features
in S2 ), that is, F̃1 = Avg(F1 ) and F̃2 = Avg(F2 ). Here
Avg(·) is the average pooling operation; F1 2 RH1 ⇥W1 ⇥C1
and F2 2 RH2 ⇥W2 ⇥C2 are the features of S1 and S2 ,
respectively (H1 /H2 and W1 /W2 correspond to the height and
width of the features in S1 /S2 , respectively).
Then, we perform the upsample and downsample operations
1⇥1⇥C2
on the features F̃1 and F̃2 to achieve F̃up
and
1 2 R
1⇥1⇥C1
F̃down
2
R
,
respectively,
which
can
be
formulated
as
2
F̃up
1 = Upsample(F̃1 );

F̃down
= Downsample(F̃2 ), (16)
2

where the Upsample(·) and Downsample(·) operations make
down
F̃up
become the same size as F2 and F1 by linear
1 and F̃2
transformations.
To provide detailed information from low-stage features to
high-stage features and transfer high-stage semantic information to low-stage features, we adopt the L1 distance to perform
the bi-directional interaction between the features of S1 and
S2 , which is defined as
up
down
1 $S2
LSBF
, F̃1 ).
E = L1 (F̃1 , F̃2 ) + L1 (F̃2

(17)

We also perform bi-directional enhancement between S2
and S3 and between S3 and S4 , and thus the final BFE loss
can be written as
S1 $S2
S2 $S3
S3 $S4
E
LBF
HM AL = LBF E + LBF E + LBF E .

(18)

The final HMAL loss, which is defined as
DR
BF E
LHM AL = LM
HM AL + LHM AL .

(19)

E. Modality-Range Identity-Center Contrastive (MRIC) Loss
To reduce the intra-class difference and increase inter-class
discrepancy, we introduce the MRIC loss to improve feature
representations and minimize the modality gaps among the
VIS, IR, and middle features. The MRIC loss consists of three
items: an intra-range loss, a middle feature loss, and an intermodality loss based on identity centers. The illustration of the
MRIC loss is presented in Fig. 5.
Following previous works [1], [7], we apply the holistic
and partial generalized mean pooling to each feature in R̃
and concatenate the pooling features to obtain the 1D middle
features, and we can get the 1D middle feature set R̃0 =
vis
ir
ir
{mvis
L , mS , mL , mS }. Analogously, we apply the same
pooling and concatenation operations to each feature in R
vis ir ir
and thus obtain the 1D feature set R0 = {rvis
L , rS , rL , rS }.
The robustness of the identity centers ensures they are not
influenced by pedestrian appearance changes. Technically, we
first obtain identity centers through the weighted average of the
features of each person at the specific modality and range. For
example, the center of the relation-enhanced features for the

IEEE Transactions on Circuits and Systems for Video Technology

Page 8 of 26

PREPRINT SUBMITTED TO TCSVT, JULY 2025

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60

Intra-range distance

8

The inter-modality loss LVMIRIC is leveraged to mitigate the
intra-class discrepancy and enlarge the inter-class distances
between VIS, IR, and middle features, which is formulated as

Inter-modality distance

Middle-feature distance

vis

ir

vis

mid

C
,C
C
,C
LVMIM
RIC = LM RIC + LM RIC

Short-/long range VIS feature centers

Push

Middle feature centers

Short-/long range IR feature centers

Pull

Modality representations

Fig. 5. Illustration of the proposed MRIC loss. Different colors represent
different identities.

pedestrian with the identity i at the long-range VIS modality
can be achieved as
PK
K
vis T
X
exp( k=1 rvis
L,i,j rL,i,k )
vis
cvis
=
(20)
PK
PK vis vis T rL,i,j ,
L,i
j=1
j=1 exp(
k=1 rL,i,j rL,i,k )

where K represents the number of VIS features of each
1⇥C 0
person; rvis
denotes the k-th 1D relation-enhanced
L,i,k 2 R
long-range VIS feature with the identity i in R0 .
Accordingly, we can obtain the identity center sets
P
P
P
CLvis ({cvis
CSvis ({cvis
CLir ({cir
L,i }i=1 ),
S,i }i=1 ),
L,i }i=1 ),
ir
ir P
vis
vis P
vis
vis P
˜
˜
CS ({cS,i }i=1 ), CL ({c̃L,i }i=1 ), CS ({c̃S,i }i=1 ), C˜Lir
P
ir P
˜ir
˜
({c̃ir
L,i }i=1 ), and CS ({c̃S,i }i=1 ), where C and C represent
the center sets for the enhanced features and the middle
features at a specific range and modality, respectively; P is
the number of pedestrian identities in the training set.
The intra-range loss LSL
M RIC is to reduce the distances
between the same-range VIS and IR features from the same
pedestrian while enlarging the distances between the samerange VIS and IR features from different pedestrian, that is,
C vis ,C ir

C vis ,C ir

S
S
L
L
LSL
M RIC = LM RIC + LM RIC ,

(21)

where
exp(MA,B
i,i )
log PP
A,B
z=1 exp(Mi,z )
i=1

P
X

LA,B
M RIC =

exp(MA,B
i,i )
log PP
A,B
z=1 exp(Mz,i )
i=1

P
X

+

P
X

L1 (ai

(22)

bi ).

i=1

Here, MA,B 2 RP ⇥P denotes the cosine similarity matrix between A and B. MA,B
i,j denotes the cosine similarity between
the i-th row (ai ) of matrix A and the j-th row (bj ) of matrix
B; L1 (·) represents the L1 norm. By minimizing the LA,B
M RIC ,
we can effectively decrease and increase the distance between
the same pedestrian and different pedestrians in the feature
space, respectively.
ID
The middle-feature loss LM
M RIC reduces the distances between different middle features, which is defined as
vis
vis
C̃S
,C̃L

ID
LM
M RIC = LM RIC

C̃ vis ,C̃ ir

vis
ir
C̃S
,C̃S

vis
ir
C̃S
,C̃L

+ LM RIC + LM RIC +
C̃ vis ,C̃ ir

C̃ ir ,C̃ ir

L
LMLRICS + LMLRICL + LMSRIC
.

(23)

ir

mid

,C
+ LCM RIC
,

(24)

where C vis , C ir , and C mid represent the identity center sets
of VIS, IR, and middle features, respectively; C vis and C ir
denote the averaged features from the same modality for each
person; C mid is obtained by averaging all the middle features
for each person. Thus, the MRIC loss is
M ID
V IM
LM RIC = LSL
M RIC + LM RIC + LM RIC .

(25)

Finally, we adopt the cross-entropy loss (LCE [41]), the
triplet loss (LT RI [42]), the HMAL loss (LHM AL ), and the
MRIC loss (LM RIC ) to jointly train the HOH-Net. The joint
loss L is defined as
L = LCE + LT RI + LHM AL + LM RIC .

(26)

IV. E XPERIMENTS
A. Experimental Settings
Image-based Datasets. The SYSU-MM01 [22] dataset contains 491 identities. Its training set includes 395 identities with
22,258 VIS and 11,909 IR images, while the test set has 96
identities with 301 VIS and 3,803 IR images. The RegDB [43]
dataset consists of 412 identities, each with 10 VIS and 10
IR images captured by two overlapping cameras. The LLCM
[25] dataset provides 713 identities in the training set and 351
identities in the test set.
Video-based Dataset. The HITSZ-VCM dataset [44] is captured by 12 RGB and 12 IR cameras. Its training set includes
500 identities with 11,061 tracklets, while the test set contains
427 identities with 10,802 tracklets.
Implementation Details. During the training phase, all images
are resized to 3 ⇥ 288 ⇥ 144 with data augmentation [45]. For
each mini-batch, we randomly select 8 identities with 4 VIS
images and 4 IR images for each identity. We adopt AGW
[1] as our backbone network. The learning rate is warmed
up from 0.01 to 0.1 over the first 10 epochs, then decays to
0.01 at epoch 20 and 0.001 at epoch 50. We use SGD as the
optimizer with a momentum parameter set to 0.9. The number
of hyperedges M in the HSL module is set to 256. In the
FCGA module, the reduction ratio r is set to 32. in Eq. (5)
is set to 1.3, 1.1, 1.3, and 1.3 on the SYSU-MM01, RegDB,
LLCM, and HITSZ-VCM datasets, respectively. In the FCGA
module, the number Y of person regions is set to 9, 8, 9, and
9 on the SYSU-MM01, RegDB, LLCM, and HITSZ-VCM
datasets, respectively. For HITSZ-VCM, each video sequence
consists of 14 frames, averaged as the video representation. We
train the HOH-Net for 120 epochs. The proposed HOH-Net is
implemented in PyTorch on an NVIDIA A40 GPU.
B. Comparison with State-of-the-Art Methods
Our proposed HOH-Net is compared with some SOTA
image-based and video-based models, including LbA [39],
TSME [5], SPOT [20], DFLN-ViT [10], PMT [8], CAL [46],


codex
前三篇文本已经抽到，方法部分有的跨页被工具输出截断，我会再用章节关键词补读缺口。接下来先把剩下三篇的前 8 页也抽出来，再逐篇核对方法链条。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3160

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025

InfinitePerson: Innovating Synthetic Data Creation
for Generalization Person Re-Identification
Guoqing Zhang , Member, IEEE, Jin Li , Yuhui Zheng , Member, IEEE, and Ruili Wang

Abstract— Recently, large-scale synthetic datasets have
effectively alleviated the issue of insufficient person reidentification (Re-ID) datasets. However, synthetic datasets
grapple with inherent challenges, including the subpar quality
of synthetic pedestrians and single data collection. This paper
presents InfinitePerson, a costless pipeline that fully utilizes the
infinite generation capability of diffusion models to produce
diverse UV texture images and effortlessly constructs high-quality
synthetic datasets by simulating a real surveillance network.
Specifically, we innovatively propose the utilization of diffusion
models to generate high-quality, realistic, and diverse UV
texture images to address the limitations of clothing textures.
This ensures that our 3D character models have complete
clothing texture information and look very similar to real-world
pedestrians. Moreover, in response to the challenges in replicating
synthetic data collection pipelines, we propose a sub-monitoring
network data collection method, which can collect pedestrians
data from different viewpoints, backgrounds, and lighting
conditions through simple scene layout. Finally, a more scalable
and realistic large synthetic dataset called InfinitePerson
is created, containing 4,700 identities and 535,636 images.
Experimental evidence demonstrates show that models trained
on InfinitePerson exhibit superior generalization performance,
surpassing those trained on both popular real-world and
synthetic person Re-ID datasets. The InfinitePerson project is
available at https://github.com/zhguoqing/InfinitePerson.
Index Terms— Generalization person re-identification, synthetic Re-ID dataset, stable diffusion, sub-monitoring network.

Received 8 June 2024; revised 20 October 2024; accepted 16 November
2024. Date of publication 22 November 2024; date of current version
7 April 2025. This work was supported in part by the National Natural
Science Foundation of China under Grant 62172231, Grant 92470202, and
Grant U20B2065; in part by the Natural Science Foundation of Jiangsu
Province under Grant BK20220107; in part by Wuxi Industrial Innovation Research Institute-Visual Intelligent Analysis of Worker Behavior
and Anomaly Warning; and in part by 2020 Catalyst: Strategic New
Zealand–Singapore Data Science Research Programme Fund by MBIE, New
Zealand. This article was recommended by Associate Editor J.-H. Xue.
(Corresponding author: Ruili Wang.)
Guoqing Zhang is with the School of Computer Science, Nanjing University
of Information Science and Technology, Nanjing 210044, China, and also with
the School of Mathematical and Computational Sciences, Massey University,
Auckland 4442, New Zealand (e-mail: guoqingzhang@nuist.edu.cn).
Jin Li is with the School of Computer Science, Nanjing University
of Information Science and Technology, Nanjing 210044, China (e-mail:
jin_li@nuist.edu.cn).
Yuhui Zheng is with the Key Laboratory of Tibetan Information Processing,
Ministry of Education, Qinghai Normal University, Xining 810008, China
(e-mail: zheng_yuhui@nuist.edu.cn).
Ruili Wang is with the School of Mathematical and Computational Sciences,
Massey University, Auckland 4442, New Zealand, and also with the School of
Computer Science, University of Nottingham China, Ningbo 315104, China
(e-mail: ruili.wang@massey.ac.nz).
Digital Object Identifier 10.1109/TCSVT.2024.3504722

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) has a focal point within
the domain computer vision, attracting significant attention due to its expansive potential application prospects in
intelligent security, video surveillance, and other fields [1], [2].
However, existing Re-ID datasets are insufficient to meet
increasingly diverse practical needs [3], [4], [5], and creating
new real datasets faces various challenges. Firstly, annotating
precise identities in real-life scenarios requires a significant
amount of manpower. Secondly, collecting large amounts of
characters data faces the problem of violating personal privacy. Therefore, synthetic person Re-ID datasets have recently
gained popularity among researchers due to their inherent
advantages, such as the avoidance of privacy issues and lower
annotation costs [6], [7], [8]. However, synthetic datasets still
face some challenges, such as poor quality of 3D character
models, which are insufficient to fully simulate real pedestrians, and the difficulty in reproducing the process of collecting
synthetic data, all of which limit the further development of
synthetic datasets.
In existing synthetic datasets [7], [8], there are significant
differences between the 3D character models used and realworld pedestrians. The main reason is that they all ignore the
texture definition of 3D clothing when generating UV texture
maps, leading to a misalignment between the generated UV
maps with the corresponding 3D clothing models, making
the 3D character that bears a resemblance to cartoon characters, as shown in Figure 1. The significant gap between 3D
characters and real-world pedestrians directly affects the effectiveness of the synthetic dataset and the generalization ability
of models trained on this dataset. Additionally, these synthetic
data collection methods suffer from shortcomings in terms of
replicability and adaptability, hindering the extension of these
datasets to address specific Re-ID scenarios [9]. Therefore,
we create a new reproducible synthetic data collection pipeline
suitable for the ReID.
In this work, we propose a costless pipeline called Infiniteperson to address the above issues. To generate a variety
of high-quality 3D character models, we innovatively use
diffusion models to generate high quality and diverse UV
texture maps and use normal maps dataset as additional input
to train ControlNet [10] to guide the Stable Diffusion [11] in
generating reasonable UV texture maps. Compared to previous
work [7], [8], the UV images we have generated are more comprehensive and better aligned with defined clothing textures.
In addition, we also propose a sub-monitoring network data

1051-8215 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID

Fig. 1.
Illustration of selected examples from (a) UnrealPerson and
(b) RandPerson datasets, respectively. (c) the 3D character models with our
generated UV texture maps rendered by UE4.

collection method, which enables researchers to more easily
design and build complex Re-ID scenes to better simulate
the real world. Our major contributions are summarized as
follows:
• We innovatively use diffusion models to generate UV
texture images to resolve the problem of limited clothing
texture resources in synthetic datasets and enhance both
the diversity and quality of 3D character models.
• We propose a sub-monitoring network data collection
method, which can more accurately simulate the operation of the monitoring network, effectively control data
quality and collection process, and facilitate subsequent
workers to design specific Re-ID scenarios.
• We create a large-scale synthetic person Re-ID dataset
called InfinitePerson, which contains 535,636 images
of 4,700 identities, covering various scenes and lighting conditions, providing researchers with powerful data
resources.
• Experimental results show that models trained on InfnitePerson have better generalization performance than
models trained on other widely used real-world and
synthetic Re-ID datasets.
II. R ELATED W ORK
A. Generating Images for Re-ID
The generation of realistic and sensible textures is a critical
aspect of simulating authentic 3D character models, as textures
carry essential information for describing and identifying
object instances [17]. The early incorporation of texture information into Re-ID tasks was primarily aimed at mitigating
the issue of spatial semantic misalignment. Wang et al. [18]

3161

employed the Re-ID model to compute the similarity between
the source image and the synthetic image, aiming to produce high-quality texture images. Zhang et al. [19] used
DensePose [20] to distort the input task image into a standardized UV coordinate system. This approach established
a dense correspondence between 2D pedestrian images and
standardized human body representations based on 3D surface
space, effectively resolving the problem of semantic alignment
and misalignment in Re-ID task images. Jin et al. [21] used
trained networks to produce texture images of each pedestrian
in the Re-ID dataset. Since the pedestrian surface in the generated texture image is semantically aligned and encompasses all
three-dimensional surfaces of pedestrians, it could effectively
guide the Re-ID network to learn semantically aligned feature
representations and gain a more comprehensive understanding
of pedestrian features. Therefore, we believe that 3D character
models with high-quality and authentic clothing textures can
benefit the Re-ID network.
With the remarkable advancements in diffusion models for
image generation [10], [11], some recent works had made it
possible to generate realistic and diverse images based on
text prompts [22], [23]. Bhunia et al. [24] adopted diffusion
models to generate realistic character images based on pose
information, and then added these generated images to Market1501 [4], which enriches the diversity of characters in the
dataset to a certain extent. However, these generated character images often lack background information and label
information, which limits their applicability in Re-ID tasks.
In addition to generating images, OWD [25] also provided
diverse real-world data, but collecting and expanding this
data is still limited by time, resources, and privacy issues.
Therefore, the flexibility of synthetic data can help researchers
generate large-scale datasets in special application scenarios,
which is difficult to achieve with real-world datasets such as
OWD [25].
B. Learning From Synthetic Dataset
As deep learning progresses within the realm of person
Re-ID [26], [27], [28], [29], [30] the demand for large-scale
and diverse datasets continues to increase. Compared to the
time-consuming and labor-intensive process of collecting and
manually annotating Re-ID pedestrian images in the real
world, synthetic data can minimize the expense of manual
annotation and avoid security and privacy issues. In recent
years, game engines have been able to effectively simulate real-life scenes, and even convert the environmental
factors of the scene into controllable parameters (such as
lighting and perspective) with their excellent rendering techniques and diverse scene design capabilities. For example,
VC-Clothes [31] and SynPerson [6] are synthetic person Re-ID
datasets collected on the GTA5 game engine. SynPerson [6]
studied the impact of lighting on Re-ID systems by varying
weather conditions and lighting, and VC-Clothes [31] is a
dataset under clothing changes. In real scenarios, associating
the same identity with clothing changes and labeling them
correctly is quite challenging for manual labeling.
In recent years, various synthetic datasets have been continuously proposed. SOMAset [12] is a synthetic dataset

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.

3162

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025

TABLE I
D ETAILED C OMPARISONS OF S OME S YNTHETIC P ERSON R E -ID DATASETS . “#I LLUM ” R EFERS TO W HETHER THE DATASET H AS L IGHT C HANGES .
“S CALABLE ” R EFERS TO W HETHER THE DATASET I S S CALABLE . “M ONITORING N ETWORK ”: R EFERS TO W HETHER A M ONITORING N ETWORK
I S S IMULATED D URING THE DATA C OLLECTION P ROCESS . “R ELIABLE T EXTURE ”: R EFERS TO W HETHER THE 3D C HARACTER M ODELS
U SED IN THE S YNTHETIC DATASET H AVE C OMPLETE AND R ELIABLE C LOTHING T EXTURES

comprising instances generated using photo-realistic human
body simulation software. SyRI [13] used 100 virtual humans
illuminated with multiple HDR environment maps. Both
SOMAset [12] and SyRI [13] are relatively small and
have limited diversity in background and human appearance.
PersonX [14] is a large-scale synthetic dataset including 1,266
human models. However, due to the static nature of the
characters, this dataset lacks scalability. UnrealPerson [8] is
another large synthetic dataset that significantly enhances the
accuracy and diversity of the dataset by using Unreal Engine
(UE4) to render four large realistic scenes and randomly
generate a variety of 3D character models. However, the
clothing textures of the 3D characters in this dataset are
directly replaced by cropping clothing image blocks from
a real clothing dataset [32], [33], resulting in significant
difference between the 3D character models and real-world
pedestrians. SynPerson [6] and FineGPR [16] are primarily
synthetic datasets collected to represent various lighting and
weather conditions, but they did not simulate pedestrian movement. The detailed introduction of some synthetic datasets is
presented in Table I.
III. T HE I NFINITE P ERSON P IPELINE
A. Problem: Person Re-Identification
N
Given an annotated image dataset S = {xi , yi , ci }i=1
, where
xi represents the i-th image, yi and ci represent the identity
label and camera label of image xi . The goal of the Re-ID
task is to learn a suitable network model to map images
to the feature space H, H = {hi |hi = f (δ; xi ) , 1 ≤ i ≤ N },
where f (δ; xi ) is the feature extracted by the network model.
A direct method to reducing the distance between instances
of the same identity compared to different identities is to
minimize the error in predicting identities within the dataset S:


minE(xi ,yi )∈S yi − g( f (xi )) ,
(1)

In this formula, where g represents the classifier, the quality
of features learned by the network model is influenced by the
data distribution in the dataset S.
The above statement reveals two major shortcomings.
A challenge is to collect and annotate the training dataset S.

Fig. 2. Demonstrating the creation of a 3D character using MakeHuman, and
adding the generated UV texture maps to the 3D character models in UE4.

It often takes a lot of time and manpower to construct and
accurately label pedestrians correctly in multiple different
scenarios, which is very challenging for manual annotation.
In addition, the data distribution of dataset S is susceptible to
interference from changes in background, lighting, and other
factors. Therefore, Re-ID data collected in a single scenario
is usually difficult to successfully transfer to other scenarios.
As shown in Table II, the evaluation results of the Re-ID
models trained on CUHK03 and Market-1501 were transferred
to the MSMT17 test set show that both Rank-1 and mAP
scores are very low, indicating a significant gap between the
domains and also revealing the weaknesses of the current
Re-ID task.
B. 3D Virtual Character Generation
1) Character Model and Animation: We create a
diverse range of male and female models, we employ
MakeHuman [34] to generate various character types based
on factors such as gender, age, torso, facial features, and
more. To further enhance diversity, we leverage different
hairstyles, beards, skin textures, and other assets to customize
and adorn our characters. In this project, we export the 3D
character models as.fbx files, which are then imported into
Unreal Engine 4 for adding animations. In total, we create
4,700 3D character models, all without clothing textures.
However, one limitation of MakeHuman-generated characters
is the absence of walking animations. To address this issue,
we add skeleton and walking animations suitable for game

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID

Fig. 3.

3163

The InfinitePerson pipeline, which shows the comprehensive process of generating our synthetic datasets.

engines to these models. In this way, our 3D character models
can execute a range of actions in scenes, including walking,
running, jumping, etc. which provides crucial support for
simulating real-world pedestrian behavior.
2) UV Mapping: The surface of clothing in real life has
rich texture details, while the 3D model of clothing itself
has no texture information, as shown in Figure 2. UV mapping [35] is a method of enhancing the shading details of
3D models by mapping spatial coordinate points of objects
to texture coordinates and retrieving the corresponding values
from texture images. When adding textures, there are two
essential coordinate systems to consider: one represents spatial
coordinates describing the vertex positions (X, Y, Z ), and
the other represents the UV coordinates. The UV coordinate
system is used to determine the position information of each
point on the texture image, aligning it with the geometric
structure of the 3D model. In the case of complex geometric
mappings, UV mapping is often necessary to project points
from a 2D texture image onto a 3D model, followed by
rendering through a rendering engine. This process enables the
3D clothing model to have rich texture information, as shown
in Figure 2.
3) UV Texture Maps: UV texture is a technique that ensures
that every point of an image is precisely corresponded to
the surface of the model, while the gaps between points are
processed for image smoothing interpolation. UV Mapping
technology allows us to map UV texture maps onto 3D
clothing models, providing them with rich texture information.
However, each 3D clothing model is usually equipped with
one UV texture map, which means that each clothing style
has only one texture map available, thus limiting the diversity
of 3D character models that can be created. In addition,
although the MakeHuman community has provided a variety
of clothing types, including those we wear daily, their selection
and variety are still limited. Therefore, this paper proposes to

generate diverse sets of clothing models by changing the UV
texture maps of existing clothing models. This method can not
only expand the diversity of 3D character models, but also
provide designers with a broader creative space. To obtain
diverse clothing, we need to generate different UV texture
maps that adhere to the definition of clothing texture. Stable
Diffusion [11] is a powerful large-scale generative model
within the realm of computer vision. However, it often lacks
precise control over image generation. To generate a reasonable UV texture map, it is necessary to ensure that the large
diffusion model has the ability to generate high-quality texture
images in a controllable manner. ControlNet [10] effectively
solves this problem by controlling Stable Diffusion [11].
We use normal maps dataset [36] as additional input to train
ControlNet [10] to guide Stable Diffusion [11] in generating
reasonable UV maps. The benefit of using normal maps
dataset as additional input is that the generated UV texture
maps align with the model’s surface structure. This enables
the generation of high-quality textures on various types of
clothing. We import the 3D clothing models and UV texture
maps into the Unreal Engine 4 [37], and utilize the material
system provided by the game engine to generate a wide variety
of clothing models infinitely to some extent. As can be seen
from Figure 4, various 3D character models can be obtained
using the generated UV maps.
C. Sub-Monitoring Network
Previous data collection methods for synthetic datasets
often focused on specific scenarios or research objectives.
For instance, SynPerson [6] focuses on studying the effects
of lighting without simulating pedestrian movement, while
UnrealPerson [8] randomly collects pedestrian images in large
scenes. This approach restricts future researchers from designing complex Re-ID scenarios and results in a lack of flexibility
in data collection.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.

3164

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025

appearing in the camera’s field of view and maximizes the
efficiency of data collection. This automated process helps to
reduce human intervention and improve data consistency.
Our method provides several benefits in terms of data collection. Firstly, it allows us to better control over essential factors
such as viewpoint, background, and lighting when collecting
pedestrian images, which greatly affect the performance of the
Re-ID tasks. The monitoring scene shown in Figure 5 includes
varied lighting regions. Secondly, simulating pedestrian occlusion is also an important aspect and excessive occlusion may
lead to incorrect annotation and recognition results. With our
method, we can design pedestrian routes by controlling the
distance between characters and the camera’s angle, and introduce moderate occlusion without compromising the accuracy
of data annotation.This allows us to simulate more realistic
occlusion scenarios in the data and thus better assess the
performance of the Re-ID tasks.
D. InfinitPerson Dataset Overview

Fig. 4. Different categories of UV texture maps. (a) Regular UV texture maps
maintain the corresponding shape of the clothes. (b) Dresses and shoes with
irregular UV texture maps from top to bottom. The dashed box contains the
original UV map of the clothes, and others are generated by Stable Diffusion.

In real surveillance scenarios, monitoring networks within
an area typically consist of multiple sub-networks, and each
subnet covers the limited field of view of surveillance cameras.
To ensure comprehensive monitoring and security, multiple sub-networks must operate simultaneously, with each
subnet responsible for monitoring different areas or locations. Inspired by this practical monitoring network structure,
we introduce a sub-network data collection method in our
research, aiming to collect pedestrian images within each subnetwork using a limited number of cameras, typically four
to six. This approach replicates the operating mode of a real
monitoring network, where each sub-network is accountable
for distinct monitoring zones. In this way, we can more
accurately simulate the operation of the monitoring network,
while controlling the quality of data and the collection process
more effectively and flexibly.
To simulate the movement of characters, we establish a
circular pedestrian path, as shown in Figure 5. During the
image collection process, we introduce pedestrians in batches
and let them follow the predetermined paths to simulate their
movements. After completing one round, the program will
automatically switch to the next batch of pedestrians. In terms
of data collection, our circular pedestrian path strategy is
an effective method to ensure the probability of pedestrians

1) Data Engine: Our Infiniteperson dataset is created using
Unreal Engine (UE4), one of the most well-known and widely
licensed game engines. This mature game engine boasts powerful rendering capabilities and a realistic physical material
system, which can simulate real surveillance scenes and render
lifelike 3D characters effectively. The Unreal Engine market
offers a wealth of high-quality scene resources, enabling the
generation of Re-ID datasets for a wide range of scenarios.
Additionally, visual variables such as weather, lighting, and
viewpoint can be customized to suit requirements, making the
synthetic dataset highly scalable.
2) Identities: To create 3D human models, we employ
open-source software called MakeHuman [34] to generate
highly realistic human models. Similar to OWD [25], which
sets pedestrians face features invisible, our dataset does not
rely on facial features to distinguish identities. We randomly
generate a large number of character models with different skin
tones, body types (including height and weight), and hairstyles
by using the plugin of MakeHuman [34]. For 3D clothing
models, we carefully select 141 different types of clothing,
as well as 40 types of pedestrian accessories such as hats and
glasses. In our research, we combine these clothing models,
which initially lack UV textures, with character models, and
then import them into the game engine, and use UV texture
maps generated by Stable diffusion [11] as texture images
of the characters clothing texture. Ultimately, we generate
4,700 3D character models. Figure 6 (a) shows images of
different pedestrians in different scenes, with realistic and
diverse clothing textures.
3) Scenario Design: Unreal Engine (UE4) has a
well-established community with rich resources. In this
community, we have selected 9 diverse scenes, including
indoor and outdoor scenes such as streets, banks, shopping
malls, fields, rural areas and so on, all of which are readily
available from the Unreal Engine community. Within the
Unreal Engine environment, we employ our proposed
sub-monitoring network to position the cameras and define
circular paths for pedestrians, providing effective and flexible
control over data quality and collection. Figure 6 (b) shows

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID

3165

Fig. 5. Example configuration of sub-monitoring network and 3D character models movements. A total of cameras are installed and the monitoring scenes
show each camera’s field of view. The yellow dot represents the destination points we set, and the yellow line represents our predetermined pedestrian
movement route.

535,636 images of 4,700 pedestrians using a total of 72 virtual
cameras. Figure 6 provides some images from the InfinitePerson dataset. Cropped images typically have diverse lighting,
viewpoints, backgrounds, poses, resolutions and occlusions.
IV. E XPERIMENTS
A. Datasets and Evaluation

Fig. 6. Sample images from the proposed InfinitePerson dataset, (a) Identity:
Some images from the InfinitePerson dataset have different pedestrians in
different scenes. (b) Illustrations: There are three images of pedestrians, and
under different lighting conditions, we divide the lighting scene into bright
and dark scenes. (c) Screens: Images of the same pedestrian in 6 different
scenes.

some pedestrian images under different lighting conditions in
the proposed InfinitePerson dataset, including bright and dark
lighting scenes. Figure 6 (c) presents images of the same
pedestrian in six different scenes, demonstrating the diversity
of pedestrian appearances captured by the dataset in different
environmental backgrounds.
4) Data Annotation: During data collection, we use
UnrealCV [38], [39] to gather instance segmentation annotations for every image captured by the virtual cameras. After
collecting the data, we use appropriate tools to precisely crop
pedestrians from the image.
5) Summary: We set up 12 sub-monitoring networks across
9 different scenarios and successfully captured a total of

Three popular real-world person Re-ID datasets are
used for generalization evaluation, including CUHK03 [3],
Market-1501 [4] and MSMT17 [5]. The CUHK03 [3] dataset
contains 14,097 images of 1,467 identities. The CUHK03-NP
is split into a training set and a testing set. The training set
consists of 7,365 images of 767 identities, while the testing
set consists of the remaining 6,732 images of 700 identities.
The Market-1501 dataset1 [4], gathered on the campus of
Tsinghua University, comprises 1,501 pedestrians recorded
by 6 cameras, totaling 32,668 labeled images. The training
set consists of 12,936 images of 751 identities, and the
testing set consists of 19,732 images of 750 identities. The
MSMT17 [5] dataset encompasses 126,441 labeled images of
4,101 identities. The training set comprises 32,621 images
corresponding to 1,041 identities. In the testing set, 11,659
bounding boxes are designated as query images, while another
82,161 bounding boxes are designated as gallery images.
We compare our dataset with multiple synthetic datasets
to verify its generalization ability for Re-ID. All evaluations
adhere to the single-query evaluation protocol [40]. We utilize
the Cumulative Matching Characteristic (CMC) [41], with
specific emphasis on Rank-1 accuracy, Rank-5 accuracy, and
mean Average Precision (mAP) [42] as the primary performance metrics. The synthetic dataset’s quality is evaluated
through direct transfer.
B. Implementation Details
We employ ResNet-50 [46] as our backbone for all our
experiments. The cross-entropy loss function is utilized, and
all pedestrian images are resized to 256 × 128. The training
batch size is configured to 64 samples. For optimization,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.

3166

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025

TABLE II
P ERFORMANCE C OMPARISON W ITH E XISTING R EAL -W ORLD AND S YNTHETIC DATASETS ON CUHK03-NP, M ARKET-1501 AND MSMT17,
R ESPECTIVELY. U NREAL P ERSON * M EANS THE T RAINING DATA E XTRACTED F ROM UNREAL _ V 1.1,
UNREAL _ V 2.1, UNREAL _ V 3.1 AND UNREAL _ V 4.1

we employ the SGD optimizer with a momentum of 0.9. The
initial learning rate for the backbone network is set to 0.005,
which is then decayed to 0.0005 at the 10th epoch. Training
concludes after 40 epochs.
C. Direct Cross-Dataset Evaluation
Direct cross-dataset evaluation involves training a model on
a particular dataset and subsequently assessing its performance
on another dataset, avoiding any intermediate fine-tuning or
adjustments. This evaluation method offers a robust assessment of the model’s ability to generalize across diverse
datasets, providing valuable insights into its overall performance on different datasets. We evaluate our synthetic
dataset’s quality through direct cross-dataset evaluation.
To assess the effectiveness of our synthetic dataset, we use
direct transfer performance on real-world datasets as the
primary indicator of synthetic data quality, since direct transfer
is the foundation of all other tasks. We employ CUHK03 [3],
Market-1501 [4] and MSMT17 [5] as testing sets. The evaluation results are provided in Table II. It is clear from the results
that our synthetic dataset outperforms existing synthetic and
real-world datasets in direct transfer experiments. Compared
to the UnrealPerson [8] collected using the same game engine
as ours, our synthetic dataset also achieved better performance.
The reason why we can surpass other composite datasets
is that the pedestrian data collected by our proposed sub
monitoring network data collection method includes various
key factors (such as lighting, perspective, occlusion, etc.), and
the Unreal Engine (UE4) used in our dataset has stronger simulation capabilities. Compared to UnrealPerson [8] collected
using the same game engine as ours, our 3D character models
have more complete and reasonable clothing textures than
UnrealPerson’s 3D character models, so the models trained
on our synthetic dataset has better performance.
D. Generalizable Person Re-ID
To verify the generalization of our dataset, we choose to use
various methods in the Re-ID field for comparison, including
QAConv2.0 [43] based on ResNet-50 [46], TransReID [26]
based on Transformer [47], PHA [44] based on Discrete Haar

Wavelet Transform [48], and Domain generalizable Re-ID
method SuA-SpML [45]. The results of direct cross dataset
evaluation are presented in Table III. It can be observed that the
optimal performance is achieved among all methods using the
InfinitePerson dataset, surpassing all other competing synthetic
datasets. This proves that the Re-ID model trained on our
dataset has stronger generalization ability. What’s noteworthy
is that our dataset’s superior performance compared to others
stems from its high-quality 3D pedestrian models and diverse
scene variations. These factors collectively promote the model
to have stronger robustness.

E. Quantitative Analysis
To demonstrate that the UV texture maps generated by
Stable Diffusion [11] can increase the diversity of 3D character
models and to evaluate the effectiveness of the proposed
sub-monitoring network for collecting synthetic datasets,
we perform quantitative analysis.
1) Identities: In contrast to UnrealPerson, which randomly
matches clothing to create a large number of different clothing
sets, we generate diverse sets of clothing models by replacing
the UV texture maps of existing clothing models with those
generated by diffusion models. To verify whether our proposed
method of generating clothing textures can improve pedestrian
diversity, we use 141 3D clothing models, including those with
regular and irregular UV texture maps. For these 141 clothing
models, unlike Unrealperson [8], we do not use random matching to obtain clothing sets. Instead, we choose to manually
pair them into 152 clothing sets, resulting in 4,700 virtual
characters with distinct clothing textures. Figure 8 clearly
demonstrates that as the number of characters increases, the
accuracy of Rank-1 and mAP consistently increases on all
three real-world datasets, despite using only 152 clothing
sets. This shows that the UV texture maps we generate can
indeed increase the diversity of virtual characters, as shown
in Table II. Furthermore, it’s worth noting that even if we
increase the number of 3D character models from 3,000 to
4,700, we observe no performance degradation. This finding
challenges the notion as posited by UnrealPerson [8] that

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID

3167

TABLE III
D IRECT C ROSS DATASET E VALUATION R ESULTS U SING R E -ID M ETHODS . U NREAL P ERSON * M EANS THE T RAINING DATA E XTRACTED
F ROM UNREAL _ V 1.1, UNREAL _ V 2.1, UNREAL _ V 3.1 AND UNREAL _ V 4.1

Fig. 7. Results of direct transfer evaluation on CUHK03-NP, Market-1501 and MSMT17, varying the quantity of sub-monitoring networks in synthetic
datasets from 2 to 12.

optimal performance can only be achieved with 3,000 3D
character models.
2) Sub-Monitoring Network: To demonstrate the effectiveness of our proposed sub-monitoring network, we analyze it
by setting the number of sub-monitoring networks. As shown
in Figure 7, when we increase the number of sub-monitoring
networks, the accuracy of Rank-1 and mAP continues to
increase on all three real-world datasets. Even when we deploy
sub-monitoring networks in different regions of the same large
scene or use distinct networks for the same character group, the
accuracy of Rank-1 and mAP consistently exhibits a consistent
increase on all three real-world datasets. As the number of 3D
character models and sub-monitoring networks increases, the
performance remains consistent without showing any decline.
This demonstrates the enormous potential of our method to
further produce larger synthetic datasets.
3) Dataset Parameter Analysis: To fully validate the
effectiveness and applicability of our dataset, we also use
QAConv2.0 [43] as the baseline and adjusted various parameters of our dataset for experiments. We first fix the number
of pedestrians and change the number of clothing sets
used, as shown in Table IV. The trend reveals that as the

number of clothing types increases, the model’s generalization
performance improves. Moreover, increasing the number of
sub-monitoring networks and cameras further enhances the
model’s generalization performance. This not only verifies the
effectiveness of our sub monitoring network, but also verifies
that using pedestrian images in multiple scenes for model
training can improve the model’s generalization ability.
F. Qualitative Comparisons
As shown in Figure 9, there are significant differences in the
UV texture maps generated using ControlNet [10] to control
Stable Diffusion [11] based on different training datasets.
When compared to ControlNet [10] trained on the normal
map [36] dataset, the performance of ControlNet [10] trained
on the Canny edge dataset [49] and that trained on the Instruct
Pix2Pix dataset is notably inferior. Specifically, using the
Instruct Pix2Pix dataset [50] as additional input to control the
images generated by Stable Diffusion [11] are unsuitable for
use as UV texture maps. Although employing the Canny maps
as additional input to control the UV texture map generated by
Stable Diffusion yields reasonable results, the generated image

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

5097

Improving Text-Based Person Retrieval by
Excavating All-Round Information
Beyond Color
Aichun Zhu , Zijie Wang, Jingyi Xue, Xili Wan , Member, IEEE, Jing Jin,
Tian Wang , Senior Member, IEEE, and Hichem Snoussi

Abstract— Text-based person retrieval is the process of searching a massive visual resource library for images of a particular
pedestrian, based on a textual query. Existing approaches often
suffer from a problem of color (CLR) over-reliance, which
can result in a suboptimal person retrieval performance by
distracting the model from other important visual cues such
as texture and structure information. To handle this problem,
we propose a novel framework to Excavate All-round Information
Beyond Color for the task of text-based person retrieval, which
is therefore termed EAIBC. The EAIBC architecture includes
four branches, namely an RGB branch, a grayscale (GRS)
branch, a high-frequency (HFQ) branch, and a CLR branch.
Furthermore, we introduce a mutual learning (ML) mechanism
to facilitate communication and learning among the branches,
enabling them to take full advantage of all-round information
in an effective and balanced manner. We evaluate the proposed
method on three benchmark datasets, including CUHK-PEDES,
ICFG-PEDES, and RSTPReid. The experimental results demonstrate that EAIBC significantly outperforms existing methods
and achieves state-of-the-art (SOTA) performance in supervised,
weakly supervised, and cross-domain settings.
Index Terms— Color (CLR) information, cross-modal retrieval,
frequency, person reidentification (ReID), text-based person
retrieval.

I. I NTRODUCTION
EXT-BASED person retrieval is the process of searching a massive visual resource library for images of a
particular pedestrian, based on a textual query. Since query
sentences are easier to access than other types of query (such
as images), this approach is particularly important in video
surveillance applications and has gained increasing attention.

T

Manuscript received 4 March 2023; revised 4 November 2023;
accepted 14 February 2024. Date of publication 28 February 2024; date of
current version 1 March 2025. This work was supported in part by the National
Natural Science Foundation of China under Grant 62101245 and Grant
61972016, and in part by the Natural Science Research of Jiangsu Higher
Education Institutions of China under Grant 21KJB520008. (Corresponding
author: Aichun Zhu.)
Aichun Zhu, Zijie Wang, Jingyi Xue, Xili Wan, and Jing Jin are
with the College of Computer and Information Engineering, Nanjing
Tech University, Nanjing 211816, China (e-mail: aichun.zhu@njtech.edu.cn;
zijiewang9928@gmail.com).
Tian Wang is with the Institute of Artificial Intelligence, SKLCCSE,
Beihang University, Zhongguancun Laboratory, Beijing 100191, China
(e-mail: wangtian@buaa.edu.cn).
Hichem Snoussi is with the Institute Charles Delaunay-LM2S FRE CNRS
2019, University of Technology of Troyes, Troyes 10004, France (e-mail:
hichem.snoussi@utt.fr).
Digital Object Identifier 10.1109/TNNLS.2024.3368217

Fig. 1. Text-based person retrieval examples given by a single-branch basic
model. The targeted/untargeted person images are marked with green/red
borders.

However, while image-based person retrieval (also known as
person ReID) has been extensively studied, the text-based
person retrieval task is still in its early stages of development.
The primary challenge in the text-based person retrieval
task is to extract and align relevant clues from multimodal
data sources, including RGB pedestrian images and natural
language queries. Various methods have been employed to
tackle this task, but many still struggle with the problem of
color (CLR) over-reliance. Utilizing a single-branch basic
model (which is detailed in Section III-B), we display some
retrieval examples in Fig. 1. The targeted (untargeted) pedestrian images are marked with green (red) borders. It can be
noticed that within the top-ten retrieval results, quite a few
images that do not align with the query description still share
similar CLRs with the targeted images. This indicates a heavy

2162-237X © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.

5098

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

Fig. 2. Illustration of some feature response maps on pairs of RGB and GRS
pedestrian images, which are obtained via two single-branch basic models are
trained, respectively, on RGB and GRS data. By averaging along the channel
dimension, we compute the feature response maps for all the images. In each
column (numbered from 1 to 7), the first and third rows are, respectively, the
RGB and GRS images while the second and fourth ones are the corresponding
RGB and GRS response maps.

reliance on CLR information in existing text-based person
retrieval methods. However, the model may ignore subtle yet
discriminative cues, such as the presence of items like “a
camera bag,” “a cross-body bag,” or “high top shoes,” leading
to retrieval failures. In certain cases, an image may still rank
at the top of the result list by the model, even if the CLRs in
local regions do not accurately match the given textual query.
While CLR information undoubtedly plays a significant role
in computing cross-modal affinity, the over-reliance on CLR
information can divert the attention of the model from other
crucial visual cues such as texture and structural information, ultimately resulting in suboptimal retrieval performance.
A typical case can be observed in the second example from
Fig. 1. In this instance, the CLR of the T-shirt in the target
images is a very pale shade of blue, almost akin to gray.
However, the query text merely states “a blue t-shirt” without
specifying its lightness. Consequently, a single-branch basic
model tends to retrieve numerous unrelated images where the
clothing is distinctly blue. This mismatch in CLR description
consequently results in a failed retrieval case. Therefore,
alleviating the problem of CLR over-reliance can be a crucial
factor in further promoting future research. To be more clear,
two single-branch basic models are, respectively, trained on
RGB and grayscale (GRS) data, by which the feature response
maps are generated on pairs of RGB and GRS person images
and illustrated in Fig. 2. As can be seen, for the RGB and
GRS data, the attention of the models are drawn to varied
local parts, implying that the complementary effects between

the RGB and GRS data could be taken advantage and hence
the problem of CLR over-reliance could be alleviated.
To this end, in this article, a novel framework is designed
to Excavate All-round Information Beyond Color for the textbased person retrieval task, which is therefore termed EAIBC.
Specifically, to address the problem of CLR over-reliance,
we introduce a jointly optimized multibranch architecture
consists of four branches, namely an RGB branch, a GRS
branch, a high-frequency (HFQ) branch and a CLR
branch. The GRS branch employs a color deprivation module
(CDM) to obtain GRS images, while the color masking module (CMM) masks the words related to CLR information in
textual descriptions. This ensures that the GRS branch focuses
on non-CLR clues in retrieving. Besides, within an image, the
low-frequency information cares more about appearance and
CLR, while the HFQ information majorly attends to details
like texture and contour information. Therefore, for the HFQ
branch, a high-frequency extraction module (HEM) is adopted
to obtain the HFQ information from the input raw image,
which enables EAIBC to explicitly key cues like textural and
structural information other than CLR. Furthermore, in order
to ensure that EAIBC fully utilizes all-round information in
an effective and balanced way, and does not overly emphasize certain information while ignoring others, we include a
CLR branch which is specifically designed to focus on CLR
information. Additionally, a mutual learning (ML) mechanism [1] has been implemented to enable the four branches
to communicate with and learn from each other. We evaluate
our proposed method on three text-based person retrieval
datasets, namely CUHK-PEDES [2], ICFG-PEDES [3] and
RSTPReid [4]. Our experimental results show that EAIBC
outperforms existing methods and achieves state of the art
(SOTA) performance in supervised [2], weakly supervised
[5] and cross-domain [6] text-based person retrieval tasks.
To sum up, the major contributions of this article include
the following.
1) This article proposes a jointly optimized multibranch
architecture termed as EAIBC to Excavate All-round
Information Beyond Color and address the problem
of CLR over-reliance. The framework includes four
branches including an RGB branch, a GRS branch,
an HFQ branch, and a CLR branch.
2) An ML mechanism is introduced to facilitate communication and learning among the four branches, which
allows for an effective and balanced use of all-round
information.
3) To our knowledge, this article is the first to use GRS
data in addition to RGB data to improve performance in
text-based person retrieval.
4) The experimental results on CUHK-PEDES, ICFGPEDES and RSTPReid, as well as extensive ablation
analysis, demonstrating that EAIBC outperforms existing methods and achieves SOTA performance in
supervised [2], weakly supervised [5], and cross-domain
[6] text-based person retrieval tasks.
This work is an extension of our previous ACM MM 2022 conference paper CAIBC [7]. The contributions of this article over
CAIBC can be concluded as follows.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.

ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL

1) An HFQ branch is proposed to explicitly care for
extreme signals like texture and contour information
within a person image.
2) In addition to CUHK-PEDES and RSTPReid, the
ICFG-PEDES dataset is further utilized to evaluate the
performance of EAIBC.
3) EAIBC is further evaluated in the cross-domain textbased person retrieval setting to validate its generalization ability and robustness.
4) A more comprehensive quantitative analysis is carried
out to systematically demonstrate the effectiveness of
the proposed components within EAIBC.
5) After visualization of the feature response maps along
with text-based person retrieval examples given by
EAIBC, a detailed discussion is carried out to understand
the mechanism behind EAIBC. This analysis identifies
some of the current challenges in text-based person
retrieval and suggests areas for further research.
II. R ELATED W ORKS
A. Person Reidentification
Person reidentification (ReID) aims to match person images
across disjoint cameras [8], [9], [10], [11], [12], [13].
To address this problem, previous ReID methods focus either
on designing discriminative representations for human appearance or on learning a reliable affinity metric for the input
data. To further enhance the representation capability of deep
neural networks, researchers have proposed novel modules
such as the second-order non-local attention (SONA) module
by Xia et al. [14], which learns multigranular information and
relationships in an end-to-end manner. A Gabor convolution
module [15] is constructed based on the Gabor function for
capturing texture representation, which is particularly effective
when integrated into the lower layers of the network. Besides,
based on the hinge function, a novel regularizer loss is
proposed to further enhance this module. Bak and Carr [16]
propose a one-shot learning method that decomposes the ReID
metric. Hao et al. [17] present the modality confusion learning
network (MCLNet), with the aim of confusing two modalities
during optimization to focus solely on modality-irrelevant
information.
B. Cross-Modal Retrieval
Text-based person retrieval can be considered as a subtask
of cross-modal retrieval. Yu et al. [18] propose a novel
ranking model based on the learning to rank framework.
This model simultaneously leverages visual features and click
features to construct the ranking model. To be more precise,
their approach is rooted in large-margin structured output
learning, and it integrates visual consistency with click features using a hypergraph regularizer term. Wang et al. [19]
present a robust multiview hashing (RMVH) framework to
better handle the information loss problem during learning
the common semantic subspace. Meng et al. [20] propose
an asymmetric supervised consistent and specific hashing
(ASCSH) method to improve multimodal mapping learning.
Yang et al. [21] introduce a controlled semantic embedding

5099

(CSE) framework, which focuses on learning disentangled representations characterized by a controlled semantic structure
for cross-modal retrieval. A semantic disentanglement adversarial hashing (SDAH) [22] is designed to separate the original
features of each modality into two distinct components:
modality-common features with semantic information and
modality-private features containing disturbing information.
Following this initial disentanglement, the modality-private
features are shuffled and treated as positive interactions to
enhance the learning of modality-common features. This
approach significantly enhances the discriminative capabilities
and robustness of semantic embeddings.
C. Text-Based Person Retrieval
Text-based person retrieval involves searching for a pedestrian image in a large database based on a given text query.
In 2015, Ye et al. [23] introduced the Specific Person Retrieval
via Incomplete Text Description task with the aim of identifying pedestrian images according to attributes provided by
users, which can be considered as a basis for the task of
text-based person retrieval. Along with this task, a specific
attribute completion method is designed to enhance and convert a text query into a more vector for the attributes. And
after that, the text-based person retrieval task is formally
proposed by Li et al. [2] in the year of 2017. To tackle
this task, researchers have developed various methods that
adopt cross-modality attention mechanisms to match visual
and textual features. The goal of these approaches is to
improve the relevance of image-text matching by calculating
weights for the cross-modal alignments. Chen et al. [24]
propose an efficient patch-word matching model to catch
the fine-grained similarities between images and sentences.
To exploit the multilevel visual information, Jing et al. [25]
develop a posture-guided multigranularity attention network
(PMA) and use posture clues as semantic masks to locate
key parts. However, when applying pretrained models to this
task, external clues obtained from these models may deviate
significantly from the target data because of the gap of domain
between the pretraining and targeted data. In addition to the
attention-based methods, there are also common embeddingbased approaches that measure the affinities for cross-modal
sample pairs in a latent common space. For instance, a new
system presented by Zheng et al. [26] is capable of mapping
multimodal data into a common space. To explicitly consider
the intramodal data distribution, they propose the instance loss.
Besides, text-image modality adversarial matching (TIMAM)
which is introduced by Sarafianos et al. [27] take the adversarial learning mechanism into consideration so as to learn
feature representations that are invariant to modality. However,
these methods tend to ignore the importance of part representations in text-based person retrieval. To address this issue,
some recent papers seek more fine-grained paradigms. For
example, Niu et al. [28] propose the multigranularity imagetext alignments (MIA) model, which exploits the possibility of
measuring affinities across different granularities. Additionally,
some works have utilized technique like graph learning to
better model the relationship within the multimodal data, such
as the adversarial graph attention network (A-GANet) model

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.

5100

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

Fig. 3. Illustration of the EAIBC framework, which consists of RGB, GRS, HFQ, and CLR branches. The terms CDM, HEM, CMM, and CPM, respectively,
denote color deprivation module, high-frequency extraction module, color masking module, and color prior module.

by Liu et al. [29]. By means of the semantic scene graphs, the
structural features of the visual and textual data are modeled.
Aggarwal et al. [30] propose cross modal attribute aided
matching (CMAAM), a method that learns two spaces which
are, respectively, driven by attributes and class information,
by leveraging the attribute annotations and predictions. With
a Gumbel attention module, a hierarchical adaptive matching
model is introduced by Zheng et al. [31] to tackle the problem
of matching redundancy. A deep surroundings-person separation learning (DSSL) model is proposed by Zhu et al. [4]
along with a new RSTPReid dataset. Ding et al. [3] propose the semantically self-aligned network (SSAN) method to
extract semantically aligned part features for the multimodal
data. Besides, an ICFG-PEDES dataset is proposed as well.
Recently, with the development of contrastive language-image
pretraining (CLIP) [32], some researchers begin to exploit the
great cross-modal alignment capabilities of vision-language
pretrained models. Han et al. [33] introduce a cross-modal
momentum contrastive learning framework with the goal of
enriching the training data within a given mini-batch. This
approach addresses issues stemming from limited data availability. Yan et al. [34] propose the CLIP-driven Fine-grained
information excavation framework (CFine) to fully harness
the robust knowledge of CLIP for text-based person retrieval.
To effectively transfer multimodal knowledge, they conduct
fine-grained information excavation to extract intramodal discriminative clues and establish intermodal correspondences.
Jiang and Ye [35] introduce the cross-modal implicit relation
reasoning and aligning (IRRA) framework, which focuses
on learning relations between local visual-textual tokens and
enhancing global image-text matching without the need for
additional prior supervision. To be specific, Lapscore [36]
enhances the model’s ability to capture CLR information

and fine-grained cross-modal association by introducing two
subtasks on CLR reasoning, namely text completion and image
colorization (TC&IC), during training. On the contrary, our
proposed EAIBC method aims to address the issue of CLR
over-reliance by utilizing other important cues.
III. M ETHODOLOGY
A. Problem Formulation
The proposed framework, as illustrated in Fig. 3, aims to
measure the affinity between textual queries and person images
in a database. Formally, within a dataset D = {pi , qi }i = 1 N
consisting of N image-text pairs, each pair includes a person
image pi which is captured by a surveillance camera along
with its corresponding textual query qi . The Q unique identifiQ
cations (IDs) of the pedestrians are represented as Y = {yi }i=1
.
The objective is to search for the most relevant images of the
pedestrians in the gallery given a textual query.
B. Branch Prototype
To begin with, a branch prototype that extracts and aligns
the multimodal multigranular representations is introduced.
This prototype can then be used for implementing the RGB,
GRS and HFQ branches in EAIBC.
a) Visual representation extraction: To obtain the global
and local visual representations from an input image, a pretrained ResNet-50 [37] backbone is utilized. For the global
granularity, a global max pooling operation is employed to
down-scale the feature map before the last pooling layer of
ResNet-50 to the shape of 2048 × 1 × 1, which is then converted to a 2048-D vector. This vector is then passed through
a fully-connected (FC) layer and transformed to P-dim as the
global representation vg ∈ R P . For the local granularity, the

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.

ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL

5101

global max pooling operation is adopted to down-scale the
same feature map to the shape of 2048 × K × 1, so as
to horizontally K -partition the feature map. Each of the local
strips is then separately passed through a fully-connect layer
to generate K P-dim fine-grained local visual representations
K
V f = {v f k }k=1
.
b) Textual representation extraction: For extracting textual representations, we utilize a bidirectional Gated Recurrent
Unit (bi-GRU) after word embedding to process the input
sentence. The ith last hidden states of the forward and backward GRUs are averaged to represent the ith word in the
query sentence as ei . To represent the entire sentence, the
C word representations are concatenated to form E, which
undergoes a rowwise max pooling (RMP) operation along with
a fully-connect layer to obtain the global textual representation
tg ∈ R P . To obtain K local textual representations based
on word-part correspondences, a Word Attention Module is
employed based on [3]


C
sik = σ Wkp ei , Ek = sik ei i=1 , k = {1, 2, . . . , K }
(1)
where σ stands for the Sigmoid function and Wkp ∈ R1×C
is a linear transformation operation. Next, we process each
modified sentence representation Ek separately using RMP +
FC and stack them together to create the fine-grained local
K
textual representations T f = {t f k }k=1
.
After obtaining the fine-grained visual and textual representations V f and T f , they are concatenated to form vl and tl
∈ R K P . The cross-modal global and local similarities are then
computed as follows:
Sg =

vgT tg
||vg ||||tg ||

, Sl =

vlT tl
.
||vl ||||tl ||

(2)

C. Color Deprivation and Masking
To obtain the corresponding GRS image Igrs for a given
RGB image Irgb ∈ R3×H ×W , we propose a CDM that can be
formulated as follows:



 R(i, j)
GRS(i, j) = 0.299 0.587 0.114 G(i, j)
(3)
B(i, j)
where i ∈ {1, 2, . . . , H }, j ∈ {1, 2, . . . , W }. GRS(i, j),
R(i, j), G(i, j) and B(i, j) stand for the values of the pixels
in the ith row and the jth column of the GRS and RGB
channels, respectively. The empirical coefficients 0.299, 0.587,
and 0.114 are commonly adopted in image processing software
such as OpenCV and Photoshop for converting RGB images
to GRS. In order to using the ResNet-50 visual backbone to
deal with the obtained GRS images, we replicate the singlechannel GRS images three times along the dimension of the
channel.
For removing CLR information from the textual modality,
a CMM is designed. After collecting the words related to
CLR information that appear with HFQ from the textual query
corpus, a CLR bank is formed. The frequency-based CLRrelated word cloud is displayed in Fig. 4. Consequently, when a
textual query sentence is given, all of the words related to CLR
are converted into a masking token [CLR]. For instance, given

Fig. 4. Frequency-based CLR-related word cloud constructed from the CLR
bank.

a textual query “This man is wearing an orange coat with a
gray-orange backpack. He has light blue jeans and black-white
sneakers”, the output of CMM will be “This man is wearing
an [CLR] coat with a [CLR] backpack. He has light [CLR]
jeans and [CLR] sneakers.”
It is worth noting that both CDM and CMM do not require
learning any parameters, which can be directly utilized to
handle the input raw multimodal data.
D. High-Frequency Information Extraction
To extract HFQ components from a raw image I rgb ∈
R
and obtain the corresponding HFQ image I hfq ∈
3×H ×W
R
, an HEM is employed. In practice, HEM can be
implemented in varied manners, and in this article the Gaussian filter is utilized for extracting the HFQ information.
3×H ×W

E. Excavating All-Round Information via Multibranch
Architecture
1) RGB Branch: In this branch, an RGB image and a textual
query sentence are provided as inputs. The output consists of
visual and textual multigranular RGB representations, denoted
rgb
rgb
rgb
rgb
as vg / tg ∈ R P and vl / tl ∈ R K P , respectively.
To calculate the similarities for this branch, the following
formula is used:
 T
 T
rgb
rgb
rgb
rgb
vg
tg
vl
tl
rgb
Srgb
=
,
S
=
.
(4)
g
l
rgb
rgb
rgb
rgb
||vg ||||tg ||
||vl ||||tl ||
2) GRS Branch: The GRS branch accepts a GRS image
obtained through CLR deprivation and a textual description
after masking the words related to CLR information. It also
grs
grs
grs
grs
produces vg / tg ∈ R P and vl / tl ∈ R K P , just like the
RGB branch. The similarities for the GRS branch are then
obtained by
grs T grs
grs T grs
vg
tg
vl
tl
grs
grs
Sg = grs
(5)
grs , Sl =
grs
grs .
||vg ||||tg ||
||vl ||||tl ||
3) HFQ Branch: The HFQ branch processes the obtained
HFQ image and the original textual description. It generates
hfq
hfq
hfq
hfq
vg / tg ∈ R P and vl / tl ∈ R K P , similar to the RGB

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.

5102

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

branch. The similarities for the HFQ branch can be computed
as
 T
 T
hfq
hfq
hfq
hfq
vl
tl
tg
vg
hfq
hfq
, Sl = hfq
.
(6)
Sg = hfq
hfq
hfq
||vg ||||tg ||
||vl ||||tl ||
4) CLR Branch: As stated in Section I, the aim of EAIBC
is to ensure an effective and balanced use of all available
information, instead of prioritizing some information while
ignoring others. Therefore, it is undesirable for the model to
excel at extracting non-CLR clues but fail to attend to critical
CLR information. To ensure that CLR information is not
neglected, we employ the CLR branch which focuses explicitly
on clues related to CLR. Intuitively, the RGB branch is
responsible for capturing general information, while the GRS
branch is responsible for capturing other crucial details beyond
CLR. As a result, the information that is not shared between
these two branches can be considered pure information about
CLR. In the visual modality, we subtract the output feature
map Mgrs of the visual backbone within the GRS branch from
Mrgb , the output feature map of the RGB branch, to obtain
a pure CLR feature map Mclr . Then, we apply global max
pooling and a fully-connect layer to Mclr to obtain the visual
CLR representation vclr ∈ R P . And in the textual modality,
we propose a color prior module (CPM). We first subtract the
masked sentence representation Egrs from the representation
Ergb of the whole sentence to obtain Eclr . And then we use
the words related to CLR in the sentence as a CLR prior to
enhancing the CLR information in Eclr . To do so, we sum
the word embeddings of the CLR-related words and convert
them to the same dimension as the word representations eiclr
in Eclr . We add this CLR prior to each eiclr to obtain a CLR
prior enhanced representation Ecp . Finally, we apply an RMP
operation and a fully-connect layer to Ecp to obtain the textual
CLR representation tclr ∈ R P . Finally, the similarity score for
the CLR branch is computed by
T
vclr tclr
clr
S = clr
.
(7)
||v ||||tclr ||
F. Optimization
As mentioned previously, the major target of EAIBC is
to simultaneously optimize the four branches and utilize
all-round information in an effective and balanced manner.
Therefore, an ML mechanism is used to enable knowledge sharing among the branches. To achieve this, the
Kullback–Leibler (KL) Divergence is used to measure the
affinity between the logits of varied branches. Let vibr ∈
rgb grs hfq
rgb grs hfq
{vgi , vgi , vgi , viclr } and tibr ∈ {tgi , tgi , tgi , ticlr } represent the
visual and textual features extracted from the ith sample pair,
br
they are then utilized to compute the probabilities pbr
vm and ptm
of class (person ID) m as


exp γ Wm vibr
br
br
(8)
pvm vi = P M

br
k=1 exp γ Wk vi


exp γ Wm tibr
br br
ptm ti = P M
(9)

br
k=1 exp γ Wk ti

where γ Wm vibr and γ Wm tibr are logits input to the softmax
layer in the br branch, and br can be rgb, grs, hfq or clr.
Under the constraint of an ML loss, each branch is optimized,
which can be calculated following:


N
M
br
1  X X X br br  pbr
vm vi
br

p v log s
LML =
2 s∈B/{br } i=1 m=1 vm i
pvm vibr

N X
M
br br
X X

p
t
tm i 
br

pbr
+
(10)
tm ti log s
br
p
tm ti
s∈B/{br } i=1 m=1
where B = {rgb, grs, hfq, clr}.
Furthermore, to train EAIBC, we also utilize the triplet
ranking loss and ID loss, which are combined to form the
ReID loss. It is worth noting that the ID loss is applied to
hfq
hfq
grs
grs
rgb
rgb
each local representation v f k / t f k , v f k / t f k , and v f k / t f k ,
grs
rgb
rgb
whereas the triplet ranking loss is applied to vl / tl , vl /
grs
hfq
hfq
tl , and vl / tl .
IV. E XPERIMENTS
A. Experimental Setup
1) Dataset: The proposed EAIBC is trained and evaluated
on three challenging datasets for the task of Text-based Person
Retrieval, namely CUHK-PEDES [2], ICFG-PEDES [3] and
RSTPReid [4].
a) CUHK-PEDES: As per the official data split approach
described in [2], the CUHK-PEDES dataset is divided into
three sets: training set, validation set, and test set. The training
set includes 34 054 images of 11 003 persons and 68 126
textual descriptions. The validation set consists of 3078 images
of 1000 persons and 6158 textual descriptions, while the test
set includes 3074 images of 1000 persons and 6156 textual
descriptions.
b) ICFG-PEDES: The ICFG-PEDES dataset consists
of 54 522 images of 4102 persons based on the MSMT17
dataset [38]. This dataset is divided into three subsets, with
34 674 images of 3102 identities and 19 848 images of
1000 identities used for training, test, respectively.
c) RSTPReid: The RSTPReid dataset consists of 20 505
images of 4101 persons, with each person having five images
captured by different cameras. Each image is annotated with
two textual descriptions. The dataset is divided into three
subsets, with 3701, 200, and 200 identities used for training,
validation, and testing, respectively.
2) Evaluation Metrics: The performance of our approach is
evaluated based on the Rank-k accuracy metric. Specifically,
for each test query natural language sentence, all images in
the test set are ranked based on their similarities with the
textual query. A search is considered successful if at least one
image of the targeted pedestrian is ranked within the top-k.
Rank-k accuracy stands for the probability of finding at least
one matching person image within the top-k candidate list
when given a textual description as a query. The evaluation
is conducted by reporting the Rank-1/5/10 accuracies for all
ablation experiments.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.

ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL

5103

TABLE I
A BLATION A NALYSIS OF K EY C OMPONENTS ON CUHK-PEDES AND RSTPR EID

B. Implementation Details
For implementing EAIBC, the dimension P for the representations is set to 2048. Two independent pretrained
ResNet-50 [37] models are selected for the visual backbones
within the RGB and GRS branches. The word embedding
dimension is 768 or 500 with or without the pretrained bidirectional encoder representations for transformers (BERT) [39].
Images from all of the datasets are resized to 384 × 128 × 3.
Random horizontal flipping along with the CLR jittering
strategies are adopted for augmenting the visual data. To optimize the parameters of EAIBC, an Adam optimizer [40] is
used. We set the initial learning rate for the visual backbones
as 0.0001 while 0.001 for other parameters of EAIBC. The
batch size is 64 and the training process lasts for 100 epochs.
C. Ablation Analysis
To evaluate the contribution and effectiveness of proposed
components in EAIBC, comprehensive ablation experiments
are conducted on the CUHK-PEDES and RSTPReid datasets.
The results are reported in Table I and numbered from 1 to 29.
Abl.1, Abl.2, and Abl.3 are conducted using a single
RGB, GRS, and HFQ branch basic model (as described in
Section III-B). Abl.4–Abl.9 are conducted without the CLR
branch, and V or T in Table I indicate that the GRS branch
is used only for the visual or textual modality. Abl.10–Abl.16
are performed after adding the CLR branch, and Abl.28/29

are carried out with the complete EAIBC model without/with
a pretrained BERT model [39] for word embedding. In
Abl.17–Abl.27, we analyze the performance of our proposed
EAIBC method on the task of weakly supervised person
retrieval. The Rank-1, Rank-5, and Rank-10 accuracies are
reported for all ablation experiments.
1) Impact of Multibranch Learning: As shown in Table I,
while a single RGB branch model may outperform individual
GRS or HFQ branch models, combining any two of these
branches leads to a notable improvement in performance.
Furthermore, adding the CLR branch, which specifically
focuses on CLR information, further improves the retrieval
performance. These observations suggest that the jointly optimized multibranch architecture in EAIBC allows for separate
consideration of varied types of information from different
perspectives, and the four branches are enabled to complement
each other for superior retrieval performance.
2) Impact of Mutual Learning: Comparing Abl.6 with
Abl.7 in Table I, we observe that by utilizing the ML mechanism, the retrieval accuracies are improved by 0.91%, 0.42%,
0.37% and 0.95%, 0.35% 0.40% under the Rank-1/5/10
accuracies, respectively, on CUHK-PEDES and RSTPReid.
Similarly, comparing Abl.13 with Abl.11, we observe a performance increase of 0.65%, 0.47%, 0.17% and 1.30%, 0.80%,
0.60% on CUHK-PEDES and RSTPReid. Furthermore, from
Abl.15 to Abl.28, the performance gain is 0.72%, 0.62%,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.

5104

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

TABLE II
A BLATION A NALYSIS ON THE L OCAL PART N UMBER K ON CUHK-PEDES AND RSTPR EID

0.45% and 1.10%, 0.70%, 0.60%. These results provide evidence for the effectiveness of employing the ML mechanism
to facilitate knowledge exchange across different branches.
3) Impact of Local Part Number K : The number K of
local parts and the way in which they are partitioned are
also of significance for the proposed model to achieve an
accurate retrieval performance. Therefore, we carry out comprehensive ablation experiments to analyze the impact of the
choice of K and the partitioning paradigm. On either the
CUHK-PEDES or the RSTPReid dataset, there are totally
26 different partitioning paradigms employed with K varies
from 2 to 12 for ablation analysis. The experimental results
are reported in Table II. First, it can be observed from the
results that the horizontal partitioning strategy with K equals
to 6 and 8 are, respectively, the best choice for the CUHKPEDES and RSTPReid datasets. Second, compared with the
horizontal partitioning strategy, partitioning the feature map
vertically can result in an obvious performance drop. This
observation is reasonable according to the characteristics of
person images utilized for text-based person retrieval. As the
input person images are resized as a 384 × 128 rectangle and
commonly most of the images are exactly filled by a certain
pedestrian, the horizontally split stripes are more likely to
contain information for different body parts, while each of
the vertically split ones may just contain partial information
of multiple body parts, and hence leads to the performance gap
between these two strategies. Besides, as shown in Table II,
when K goes relatively large, the retrieval performance also
starts to decrease. This can be attributed to the fact that with

a relatively large K , each split part is small and thereby some
of these local parts may fail to catch complete discriminative
local clues and may even become noise information. Thus,
it seems that the horizontal partitioning strategy with a proper
local part number K best meets the nature of the text-based
person retrieval task.
4) Impact of Color Prior: Comparing several pairs of
experiments such as (Abl.10, Abl.11), (Abl.12, Abl.13),
(Abl.14, Abl.15) and (Abl.16, Abl.28), it can be seen that
taking the CLR prior information into consideration leads to
a further improvement in the performance. This observation
confirms the effectiveness of the CLR prior information to
enhance the CLR branch.
5) Comparison Between KL Divergence and JS Divergence
for Mutual Learning: As mentioned in Section III-E4, for ML,
the KL Divergence is used to quantify the affinity of logits
from varied branches. For two probabilities p1 and p2 , the KL
distance from p1 to p2 is calculated as
DKL (p2 ||p1 ) =

N X
M
X

pm
2 (xi )
pm
2 (xi )log m
p
1 (xi )
i=1 m=1

(11)

where N denotes the sample number and M is the class
number. Due to the properties of the KL divergence, the loss
based on it is asymmetric, which can replaced by a symmetric
Jensen–Shannon (JS) Divergence loss
1
(DKL (p2 ||p1 ) + DKL (p1 ||p2 )).
2

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.

(12)


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 165 (2025) 111561

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Identity-aware infrared person image generation and re-identification via
controllable diffusion model
Xizhuo Yu a , Chaojie Fan b ,∗, Zhizhong Zhang c , Yongbo Wang c , Chunyang Chen a ,
Tianjian Yu a , Yong Peng b
a

School of Traffic & Transportation Engineering, Central South University, Changsha, 410075, Hunan, China

b Key Laboratory of Traffic Safety on Track of Ministry of Education, School of Traffic & Transportation Engineering, Central South

University, Changsha, 410075, Hunan, China
c School of Computer Science and Technology, East China Normal University, Shanghai, 200062, China

ARTICLE

INFO

Keywords:
Cross-modality person re-identification
Image generation

ABSTRACT
Visible–infrared person re-identification (VI-ReID) aims to learn the identity-aware features between visible and
infrared person images. However, most works rely on two publicly available datasets, 𝑖.𝑒., SYSU-MM01 and
RegDB, which is limited by the limited amount of training data and the lack of rich scenes and perspectives.
In this paper, we propose a controllable diffusion framework for infrared person image generation and reidentification. Our approach is beyond the existing diffusion model in two perspectives: (1) we use LoRA
to fine-tune the existing diffusion models with VI-ReID dataset and therefore it helps the diffusion model
understand the infrared modality. A text adapter is then utilized to transfer the semantic understanding ability
of Large Language Model (LLMs) to our generation models; (2) we design a controllable generation module
to make the generated person images, from the same textual description, identity-aware. After meticulous
post-processing operations, our approach is capable of producing diverse visible and infrared person images,
allowing for improving the discrimination of existing VI-ReID model without any annotations. We expand
the VI-ReID dataset with our generated images, and conduct extensive experiments on VI-ReID models.
Experimental results demonstrate the effectiveness of our method.

1. Introduction
The proliferation of surveillance cameras in public spaces, transportation hubs, and commercial establishments has underscored the
need for efficient and accurate methods to track individuals across
multiple camera feeds. In this context, person re-identification (ReID)
has emerged to identify specific persons in different cameras and
aroused extensive research attentions. However, most existing ReID
methods focus on the visible environment. Despite excellent results,
the limitations of visible cameras under low-light conditions restrict
their applications in complex scenarios such as nighttime and adverse
weather conditions. Therefore, visible–infrared person re-identification
(VI-ReID) is introduced. VI-ReID conduct the person image matching
between visible cameras and infrared cameras, providing the possibility
for achieving 24-h surveillance in video monitoring systems.
Due to the substantial cross modal image disparities, learning
identity-aware features between visible and infrared images is more
challenging than traditional ReID task. To deal with this issue, recent

works resort to cross-modal feature alignment, which learns modalagnostic features in a common embedding space. However, this
paradigm requires huge data and annotation efforts, while current
researches mostly rely on two publicly available datasets, 𝑖.𝑒., SYSUMM01 [1] and RegDB [2]. For example, the largest SYSU-MM01 dataset
only includes 22,258 visible images and 11,909 infrared images. The
training data is limited, with most images originating from similar
perspective, which deviates from real-world practical applications.
Furthermore, due to the privacy issues and the government of AI
algorithm, it is challenging to manually collect a sufficiently large and
diverse cross-modal person dataset.
With the recent success of AIGC, it is natural to consider that using
the diffusion model to generate the sufficient training samples. But as
shown in Fig. 1(a), it appears that existing pre-trained diffusion model,
e.g., Stable Diffusion, which may learned from a plenty of colored images, cannot understand the infrared modality and therefore even given
the prompt ‘‘an infrared photo’’, it still generates the colored images.
Another nonnegligible issue is the diffusion process is uncontrollable

∗ Corresponding author.

E-mail address: fcjgszx@csu.edu.cn (C. Fan).
https://doi.org/10.1016/j.patcog.2025.111561
Received 8 August 2024; Received in revised form 20 February 2025; Accepted 4 March 2025
Available online 14 March 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 165 (2025) 111561

X. Yu et al.

Fig. 1. Illustration of our proposed method. (a) shows that our method performs better in semantic understanding than Stable Diffusion v1.5 and is capable of generating infrared
images. (b) exhibits controllable generation results for VI-ReID with our method.

such that it is unable to generate the identity-aware infrared person
images.
In this paper, we propose a controllable diffusion framework for
infrared person image generation and re-identification. In our approach, we employ commonly used and pre-trained Stable Diffusion
model to generate both visible and infrared person images with text
descriptions. Towards this goal, we first use LoRA [3] to fine-tune the
existing diffusion models with current VI-ReID dataset, which helps
the model learn the concept of ‘‘infrared images’’. After that, we
propose a text adapter to transfer the semantic understanding ability
of LLMs to the fine-tuned diffusion models, and allows us to describe
the person images with satisfactory text prompts. Then, we design a
controllable generation module to ensure that generated person images
from the same textual description are identity-aware, such that it
enables us to learn a discriminative VI-ReID model. After meticulous
post-processing operations, our image generation model is capable of
producing diverse visible and infrared person images based on textual
descriptions. Finally, we expand the existing real-world dataset with the
generated dataset and conduct training and testing on VI-ReID model.
Experimental results validate the effectiveness of our approach.
The contributions of our approach are summarized as follows:

presents detailed description. Experimental analysis and completion
results are shown in Section 4 to verify our approach. Finally, we
conclude the proposed method in Section 5.
2. Related work
2.1. Traditional person ReID
Traditional person re-identification [4,5] aims to match person
images captured by different visible cameras, whose challenges mainly
lie in the changes in perspective and pedestrian posture. Previous
supervised traditional person ReID works concentrate on feature representation learning [5] and distance metric learning [6]. He et al. [7]
introduce TransReID, a framework completely based on transformer,
which offers enhanced representation capabilities while demanding
less computational resources compared to methods relying on CNNs.
Considering the complexity of manually annotating data, unsupervised
approaches have gained popularity. Several works [8,9] resort to Unsupervised Domain Adaptation (UDA) and utilize transfer learning to
enhance the robustness of the model. The other approaches [10,11] use
Unsupervised Self-Learning (USL), using clustering algorithms to assign
pseudo labels for learning identity features. Dai et al. [12] introduce the
Cluster Contrast for unsupervised ReID, and use momentum updating
to strengthen the consistency of cluster features.

• We propose a new cross-modal person image generation framework, which is capable of producing diverse visible and infrared
person images with the help of fine-tuned Stable Diffusion model.
• We design a controllable module to ensure that the generated
person images is identity-aware, and therefore allowing for improving the discrimination of existing VI-ReID model.
• We construct a new VI-ReID dataset, called GCP, using text-toimage generation instead of real camera capture, thereby diminishing privacy concerns.
• We conduct extensive experiments with the SYSU-MM01 and
RegDB datasets, which demonstrate the effectiveness of our
method.

2.2. Visible–infrared person ReID
Visible–infrared person re-identification [13,14] has emerged as a
popular research area due to its ability to remain effective in lowlight environments. To overcome the great contrast between RGB and
infrared images, many works [15,16] utilize image translating to transform one modality to the other. Different from the image-level methods,
several other approaches [17,18] employ the feature-level methods
and align cross-modal features in a shared feature space. To discover
more discriminative features that are shared across modalities, Feng
et al. [19] attempt to extract additional varied modality-shared information by eliminating body shape-related semantic content from

The rest of this paper is organized as follows. Section 2 briefly
introduces related works and analyzes the difference between ours
and prior works. Section 3 gives the general idea of our method and
2

Pattern Recognition 165 (2025) 111561

X. Yu et al.

the learned features. Recently, to alleviate the complexity and difficulties associated with manual annotation, several unsupervised VI-ReID
works have made great progress. OTLA [20] tries to assign the infrared images to the generated visible pseudo classes by employing
optimal transport methods. CCLNet [21] utilizes the text information from CLIP [22] to improve subsequent unsupervised training.
To overcome camera variation and modality discrepancy, GUR [23]
leverages a bottom-up domain learning strategy, further narrowing the
gap between supervised and unsupervised VI-ReID.

3.2. Diffusion model fine-tuning via infrared images
Using existing diffusion models in generating the cross-modal person photos may fail because of their limitations in generating infrared
photos and high costs for training. To remedy this issue, we use LoRA,
a parameter-efficient fine-tuning method, to fine-tune the pre-trained
Stable Diffusion. In our approach, We leverage the real-world VI-ReID
dataset SYSU-MM01 as image training dataset, and get corresponding
text training dataset by BLIP. As a result, the fine-tuned diffusion model
can generate high-quality visible and infrared person photos based on
appropriate text prompts.
As shown in Fig. 2, the fine-tuned diffusion model is comprised of
a pretrained autoencoder, a text conditional encoder, and an image
generator. In it, the autoencoder, comprised of an encoder  and a
decoder , is used to realize the conversion between image space and
latent space and lower the computational complexity. For an image
𝑥 ∈ R𝐻×𝑊 ×3 , the encoder  encodes 𝑥 and gets 𝑧 = (𝑥) as its latent
representation, and the decoder  reconstruct the latent representation
and gets 𝑥̃ = (𝑧) = ((𝑥)). Besides, latent diffusion model also introduces a domain specific encoder 𝜏𝜃 , e.g., the text conditional encoder,
to control the synthesis process. On this basis, we choose pretrained
CLIP ViT-L/14 as our text encoder to encode the input texts that are
generated from BLIP. In particular, for a generated text description,
the text encoder encodes it to text embedding, which is then mapped
to the intermediate layers of the UNet, i.e., the backbone of our image
generator.
The time-conditional UNet is utilized in our image generator as
denoising autoencoders 𝜖𝜃 (𝑥𝑡 , 𝑡), 𝑡 = 1, … , 𝑇 , where 𝑥𝑡 is a noisy version
of the input 𝑥. In UNet, the cross-attention mechanism can establish an effective association between the image–text pair. It helps
the model better understand the text information, so as to realize
the transformation from the diffusion model to the conditional image
generator.
For learning the style of infrared images, we propose to further finetune the pre-trained diffusion model. Compared with full fine-tuning,
which updates all the parameters of the pre-trained model, Low-Rank
Adaptation (LoRA) is more efficient in storage and computation. For
a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
constrain its update:

2.3. Text-to-image generation
Text-to-image generation is proposed to realize image generation
with the guidance of text descriptions. Some works, which utilize Generative Adversarial Networks (GANs) [24,25], autoregressive models
(ARM) [26] and Vector Quantized Variational AutoEncoder (VQ-VAE)
Transformer-based methods [27], have made remarkable progress in
text-to-image generation. Recently, diffusion models (DMs) [28,29],
which leverage a denoising process, progressively refining noise to
generate high-quality images through a series of iterative steps, have
shown great success in image generation [30,31]. LDM [32] extends
DMs to latent space and significantly alleviate computational demands.
ControlNet [33] is proposed to control DMs with task-specific conditions by an additional trainable copy of pre-trained models. Imagen [34] demonstrates the efficacy of utilizing pre-trained, frozen large
language models as text encoders in the context of text-to-image generation via diffusion models. SUR-adapter [35] aligns the semantic representation between simple narrative prompts and complex keywordbased prompts with the help of large language models, further improving the performance of text-to-image generation models.
Despite the remarkable progress achieved in the field of text-toimage generation, the generation of images in new styles remains a
challenging task, which limits its further applications. Different from
methods above, we approach the adaptation of existing diffusion models from both the textual and visual perspectives to cater to the requirements of VI-ReID task.
3. Method
3.1. Overview
The shortcomings of existing VI-ReID datasets are obvious, including the limited amount of data available for training, and the lack of
rich scenes and perspectives. These shortcomings make it difficult for
VI-ReID to benefit from large-scale cross-modal pretraining.
To deal with this issue, we propose a generative framework for
cross-modal person re-identification. Our method mainly consists of
three steps: (1) Diffusion Model Fine-tuning via Infrared Images. (2)
Generative Text Adapter Alignment. (3) Controllable Image Generation
and Inference, which is presented in Fig. 2. Based on our generative framework, we also introduce a Generated Cross-modal Person
Dataset, called GCP for further training and evaluation. Thanks to
the great success of AIGC, our diffusion model enables us to generate
extensive infrared person photos and therefore allows for learning the
discriminative ReID model without any annotations.
Our proposed cross-modal person image generation framework is
shown in Fig. 2. During the fine-tuning, we first feed the original
visible and infrared images from SYSU-MM01 into BLIP to generate
corresponding text prompts. Specific prefix words, like ’a RGB photo’
or ’an infrared photo’, are added to the prompts, which encourages the
diffusion model to recognize the concept of infrared images. We then
freeze all parameters of Stable Diffusion and use LoRA to fine-tune the
U-Net of Image Generator with our constructed image–text pairs. After
that, we design a new text adapter to refine and align the output of
the text conditional encoder with that of LLMs, and therefore enables
us to generate the generation-friendly-friendly embeddings. Finally, we
propose a controllable module to make the generation be identityaware when using the same text prompts. Meticulous post-processing
is adopted to obtain appropriate cross-modal person images.

𝑊 ′ = 𝑊0 + 𝛥𝑊 = 𝑊0 + 𝐴𝐵 𝑇

(1)

where 𝐴 ∈ R𝑑×𝑟 , 𝐵 ∈ R𝑘×𝑟 . During training process, only A and B are
trainable parameters. In our method, LoRA is only used to optimize our
image generator. We select visible and infrared photos in SYSU-MM01
as the image training dataset 𝑋𝑣 = {𝑥𝑖𝑣 }, 𝑋𝑟 = {𝑥𝑖𝑟 }. Then we use BLIP
to get the original text description 𝑃𝑣 = {𝑝𝑖𝑣 }, 𝑃𝑟 = {𝑝𝑖𝑟 }. Note that here
we choose both visible and infrared images, targeting on teaching the
diffusion model to learn the difference of the cross-modal images. Our
experimental results also show that even the subset of SYSU-MM01,
e.g., images from Cam1 and Cam2, is sufficient for this fine-tuning
process. At last, to better reflect the characteristics of infrared images,
we also modify the original text by adding the prefix words such as
𝑌𝑣 = {𝑦𝑖𝑣 } =‘‘A RGB photo, [𝑝𝑖𝑣 ]’’, 𝑌𝑟 = {𝑦𝑖𝑟 } =‘‘An infrared photo, [𝑝𝑖𝑟 ]’’.
Finally, we utilize both the image and text information to fine-tune
Stable Diffusion model.
3.3. Generative text adapter alignment
Note that the text conditional encoder of the original Stable Diffusion model is CLIP ViT-L/14, which, however, is trained with text–
image contrastive learning, that not specifically designed for image
generation tasks. This leads to inadequate semantic understanding
when guiding image creation with natural language prompts.
Confronting this challenge, large language models (LLMs) attract
our attention due to their remarkable capability in semantic comprehension. Specifically, we utilize the open-sourced large language
3

Pattern Recognition 165 (2025) 111561

X. Yu et al.

Fig. 2. The pipeline of our proposed method. Left: we first use BLIP to generate some text prompts for the images from SYSU-MM01. Specific prefix words, like ’A RGB photo’
or ’An infrared photo’, are added to the prompts. For model fine-tuning, we freeze the original Stable Diffusion Model and use LoRA to fine-tune it. For text adapter training,
we design a text adapter to refine the output of the text encoder by leveraging the capability of large language models. Right Top: We design a two-stage strategy to generate
multi-perspective RGB person images belonging to the same ID based on text descriptions. Right Bottom: we use SoftEdge to detect edge maps of generated RGB images, and then
combine infrared image embeddings and edge maps to transfer RGB images to infrared images.

model LLaMA [36], and save the mean value of each token embeddings
generated by the LLM as the knowledge representation.
To transfer the semantic understanding ability of the LLM to our
fine-tuned diffusion model, we design a text adapter. We first freeze
the parameters of the large language model 𝑓𝐿𝐿𝑀 , the text conditional
encoder 𝑓𝐸𝑡 and the image generator 𝑓𝑔𝑒𝑛 of the fine-tuned diffusion
model. Then we only optimize our text adapter. It consists of a fully
connected neural networks (FCN) and a Transformer subnetwork, denoted as ℎ𝑖 . For the text description 𝑝𝑖 , we send it to both text encoders
(LLM’s and diffusion model) and use the cross-attention to associate
them. Formally, we define 𝑄𝑖 = ℎ1 [𝑓𝐸𝑡 (𝑝𝑖 )] and 𝐾𝑖 = ℎ2 [𝑓𝐸𝑡 (𝑝𝑖 )], and
calculate an attention value as:
𝑄𝑖 𝐾 𝑇
𝑎𝑡𝑡𝑖 = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥( √ 𝑖 )
𝑑

visible and infrared photo, respectively. Here 𝑃 means original text
descriptions from CUHK-PEDES and ICFG-PEDES. Note that we remove
color keywords in 𝑃 to avoid the influence of color information for
infrared photo generating. For each prompt, we first generate a RGB
photo as the reference image for subsequent controllable generation.
Then we use IP-Adapter [39], an effective and lightweight adapter
to achieve image prompt capability for the pre-trained text-to-image
diffusion models, wherein the reference image, serving as the image
prompt, is fed into the adapter to generate multi-perspective RGB
person images while preserving the identity. Since we have generated
enough RGB photos for each identity, we utilize ControlNet [33] with
SoftEdge to transfer some RGB photos to infrared photos. Specifically,
we use real-world infrared photos together with edge detection map
of generated RGB photos as the inputs of ControlNet, and thus we
realize controllable visible and infrared person photo generating for
each prompt. Finally, we generate 40 RGB photos and 40 infrared
photos for each person identity, totaling about 20 000 generated photos
with 257 person identities.
In our endeavor to create a high-quality dataset, we have identified
that the initial generation process may yield images that are not up
to the desired standard. These images may be blurry, contain multiple
persons, or exhibit other forms of noise, which can be attributed to
the inherent limitations of the generation abilities. To address these
issues, we have implemented a series of post-processing operations to
enhance the quality of our dataset. Specifically, we first remove some
low-quality images, like grayscale or blur images, in generated RGB
images. For the generated infrared images, we have observed that some
images contain color information, which is not typical for this type of
imagery. To maintain the authenticity and consistency of the infrared
images, we convert these images into grayscale, which ensures that all
infrared images in our dataset adhere to the standard characteristics of
infrared imagery, thereby enhancing the reliability of our dataset. To
further refine our dataset, we employ OpenPose [40–43], a widely used
tool for human keypoint detection. With OpenPose, we can identify
images that contain fewer than 18 detected key points, which typically
indicate low-quality or incomplete human figures. These images are
subsequently removed from our dataset. Additionally, OpenPose allows
us to resize the images with tight bounding boxes, ensuring that each
image focuses on the human figure without unnecessary background
noise. At this point, we are capable of generating visible and infrared
person images, as shown in Fig. 4, resulting in the creation of the Gen-

(2)

where 𝑑 is the feature dimension of 𝑄𝑖 and 𝐾𝑖 . To guarantee that
the meaning conveyed by the prompt remains unaffected, we directly
set 𝑉𝑖 = 𝑓𝐸𝑡 (𝑝𝑖 ) without any transformation. Moreover, we design a
distillation loss 𝐿𝐿𝑀 to incorporate the semantic understanding of the
LLM into 𝑎𝑡𝑡𝑖 as following:
𝐿𝐿𝑀 = 𝐊𝐋[𝐖𝟎 𝑓𝐿𝐿𝑀 (𝑝𝑖 )∕𝜏, 𝑄𝑖 ∕𝜏]

(3)

where 𝜏 is the temperature, 𝐊𝐋 is the KL divergence, and 𝐖𝟎 is
a randomly initialized matrix using Kaiming initialization, ensuring
maximal preservation of the LLM’s semantic content while adjusting the
dimensional alignment between 𝑓𝐿𝐿𝑀 and 𝑄. Therefore, we construct
𝑉𝑖′ = 𝑉𝑖 ⊗ 𝑎𝑡𝑡𝑖 and get calibrated semantic information. Finally, through
a fully connected layer, the output generated by our text adapter is
transformed to 𝑐𝐿𝐿𝑀 , a representation that encapsulates the semantic
interpretation capabilities inherent to the LLM, and we combine the
output of the text encoder 𝑓𝐸𝑡 (𝑝𝑖 ) and our text adapter 𝑐𝐿𝐿𝑀 as the
input to the generator as follows:
′
𝑐𝐿𝐿𝑀
= 𝜂 ⋅ 𝑐𝐿𝐿𝑀 + (1 − 𝜂) ⋅ 𝑓𝐸𝑡 (𝑝𝑖 )

(4)

where 𝜂 is a constant hyper-parameter.
3.4. Controllable image generation and inference
To make the generated images more close to the real-world photos,
we utilize the text description of the CUHK-PEDES dataset [37] and the
ICFG-PEDES [38] dataset as the text condition for the diffusion model.
We design text prompts as ‘‘a RGB photo/an infrared photo, [𝑃 ]’’ for
4

Pattern Recognition 165 (2025) 111561

X. Yu et al.

Fig. 3. Utilization of large language models during the inference stage.

3.5. Model pre-training
To verify the validity of GCP dataset, we combine GCP with the existing training dataset and use it to pretrain CLIP-Enhanced Contrastive
Learning Network (CCLNet) [21] for VI-ReID task. With the help of the
visual-language representation ability from CLIP, CCLNet establishes a
learnable cluster-aware prompt for person images and obtains textual
descriptions allowing for subsequent unsupervised training. In addition,
CCLNet leverages Hungarian matching method to fuse cross-modal information, enabling the network to better learn modality-independent
features.
In addition, we also conduct evaluation on OTPA [20], a mutual information guided optimal transport approach for unsupervised
VI-ReID. Under three learning principles, 𝑖.𝑒., ‘‘Sharpness’’ (entropy
minimization), ‘‘Fairness’’ (uniform label distribution), and ‘‘Fitness’’
(reliable cross-modality matching), OTPA designs a loop iterative training strategy alternating between model training and cross-modality
matching.
Another training baseline we adopt is RPNR [44], a robust pseudolabel learning framework designed to improve the reliability of pseudolabels and establish more reliable cross-modality correspondences for
the unsupervised Visible–Infrared Person Re-identification (VI-ReID)
task.
Furthermore, we conduct experiments on the superior baseline
SDCL [45], a collaborative learning framework based on the transformer, which effectively combines shallow and deep features for
improved performance. By integrating these features, SDCL enhances
the robustness and accuracy of cross-modality learning.

Fig. 4. Text-to-image generation results by our generation framework.

Table 1
Statistics about our generated dataset.
Modal

# of ID

Mean

25%

50%

75%

RGB
IR

257
257

20.52
19.46

12
12

21
19

29
26

erated Cross-modal Person Dataset (GCP). Detailed statistics about GCP
are presented in Table 1. Each identity in the dataset has approximately
20 visible images and 20 infrared images, which ensures a high degree
of ID consistency and diversity for subsequent ReID model training.
Furthermore, we also explore the use of large language models to
generate person textual descriptions, as shown in Fig. 3, aiming to
enhance the automation of generating both visible and infrared images
within the entire framework. This approach eliminates the reliance on
existing textual descriptions from the CUHK-PEDES dataset and the
ICFG-PEDES dataset, further improving the scalability and applicability
of our method.

4. Experiments
In this section, we conduct comprehensive experiments to evaluate our approach. First, we introduce our experimental settings and
implementation details. The main results and the comparison with
state-of-the-art methods are then shown in 4.2. Finally, we present the
ablation studies and visualization results of our approach. To promote
the culture of reproducible research, source codes and experimental
results accompanying this paper will be released later.
5

Pattern Recognition 165 (2025) 111561

X. Yu et al.

model using LoRA for 5000 steps; in the second stage, we train an
additional text adapter to enhance the model’s semantic understanding
capability for 10,000 steps. For LoRA fine-tuning, we select visible
images from cam1, infrared images from cam3 in SYSU-MM01 as
the image training dataset. Notably, only the UNet of the model is
finetuned while the others are frozen. For text adapter training, we
utilize LLaMA2 with the size of 13B as the large language model for
knowledge distillation, and set 𝜂 in Eq. (4) to 0.1. During the stage
of inference and dataset generating, we use OpenCLIP-ViT-H-14 as the
image encoder, and choose pretrained IP-Adapter for Stable Diffusion
v1.5. For ControlNet with SoftEdge, softedge_pidinet is employed to
detect edge maps. For ReID pre-training, we initially integrate our
proposed dataset, 𝑖.𝑒., GCP, with the original training sets of SYSUMM01 and RegDB to construct a new training set, while the test sets
remain unaltered to validate the effectiveness of our method. All our
methods are implemented with PyTorch and trained on GPU NVIDIA
RTX 4090.
ReID Baseline. After integrating GCP, our generated dataset, with
the training sets of SYSU-MM01 or RegDB, we proceed to conduct
training on four established baseline models: CCLNet, OTPA, RPNR, and
SDCL. By utilizing both the synthetic data from GCP and the real-world
data from SYSU-MM01 and RegDB, we aim to improve the models’
ability to generalize across diverse scenarios and conditions.

Fig. 5. Comparisons between real-world photos and generated images. For each image
pair, the real-world person images captured by camera is on the left, while the image
on the right is generated by our generation framework.

4.2. Comparison with state-of-the-art methods

4.1. Experimental settings

We conduct the experiments on SYSU-MM01 and RegDB under the
settings described in 4.1, and compare our method with the state-ofthe-art methods, including fully-supervised VI-ReID and unsupervised
VI-ReID. We take the test results of CCLNet as the baseline. The main
result are presented in Tables 2 and 3. We report some fully-supervised
methods (Zero-Padding, AlignGAN, DDAG, CM-MAS, DART, CMT) and
some unsupervised methods (ClusterContrast, H2H, OTLA, ADCA).
Comparison on SYSU-MM01. As reported in Table 2, our proposed
method almost achieves the best results in terms of both all-search
mode and indoor-search mode. In particular, we achieve 64.95% Rank1 and 63.59% mAP for all-search, and 71.57% Rank-1 and 77.44% mAP
for indoor-search, respectively. It is worth noting that our approach
has demonstrated effectiveness on all four baselines. We improve OTPA
by 2.87% Rank-1 and 1.95% mAP in term of all-search, and enhance
RPNR by 2.71% Rank-1 and 1.71% mAP in term of indoor-search. Our
unsupervised results are closely approaching the supervised ones.
Despite the effectiveness of previous methods, it seems that the
limited scale of VI-ReID dataset restrict the performance of existing
models. It verifies that the generated images can be effectively added to
the SYSU-MM01 dataset, demonstrating the effectiveness and usability
of our generative framework in pre-training of VI-ReID models. This
result also proves that expanding the scale of the SYSU-MM01 dataset
helps the VI-ReID model better benefit from cross modal pre-training,
without requiring any modifications.
Comparison on RegDB. Table 3 presents the results of our methods and other SOTA methods tested on RegDB. Consistent with the
experimental results observed on SYSU-MM01, our approach performs
excellent results on RegDB. To be specific, our method demonstrates superiority over both baselines, with 88.88% Rank-1 and 82.41% mAP for
Vis-to-IR, and 87.52% Rank-1 and 81.18% mAP for IR-to-Vis. Note that
the infrared images in the RegDB dataset are thermographic infrared
images, which differ significantly from the infrared images generated
based on the SYSU-MM01 dataset. However, our fine-tuning process is
only conducted on SYSU-MM01 because the data quality of RegDB is
insufficient. Therefore, there is a slight decline in the mAP on CCLNet,
but consistent improvement for OTPA.

Datasets. We evaluate our method on two widely-used VI-ReID
datasets: SYSU-MM01 and RegDB.
• SYSU-MM01. SYSU-MM01 is captured by 4 visible cameras and
2 infrared cameras, which contains 395 identities for training and
96 identities for testing, with a total of 22,258 visible images and
11,909 infrared images.
• RegDB. RegDB consists of 412 identities with 10 visible images
and 10 infrared images for each identity collected by a pair of
cameras, where half of the IDs are used for training and half of
the IDs are for testing.
Evaluation Protocols. Cumulative match characteristic (CMC) and
mean average precision (mAP) are used to evaluate the performance
of VI-ReID models. To make a fair comparison, we test both all-search
mode and indoor-search mode with official code on SYSU-MM01. On
the all-search mode, indoor and outdoor images from all cameras are
used to constitute the gallery set, while on indoor-search mode only
indoor images from 1st, 2nd, 3rd, 6th cameras are used. we also report
the results of RegDB by randomly splitting of training and testing set
10 times.
We also employ two prominent evaluation metrics to assess the
quality and relevance of synthesized images, including the CLIP Score
[46] and FID [47]. CLIP Score assesses image–text alignment by comparing embeddings from a jointly trained visual-language model, with
higher scores indicating better semantic coherence. The Frechet Inception Distance (FID) evaluates the similarity between real and synthetic
image distributions using deep features from an Inception-v3 network.
Lower FID values signal closer resemblance, reflecting the model’s ability to generate realistic and diverse images. We evaluate our method on
1000 image–text pairs of CUHK-PEDES.
Implementation Details. We adopt Stable Diffusion Model v1.5
as our pre-trained text-to-image generation model. In it, the height and
width of input and output images are all 512. The image augmentation
techniques such as center crop and random flip are utilized to enhance
the training data. We use AdamW optimizer and set the learning rate
to 1𝑒 − 4 by a constant schedule. Our experiments are conducted in two
stages: in the first stage, we fine-tune the pretrained Stable Diffusion
6

Pattern Recognition 165 (2025) 111561

X. Yu et al.

Fig. 6. Visualization of Controllable Person Image Generation. Left: images generated with the given prompts. The image enclosed by a green bounding box denotes our selected
reference image for controllable generation. Middle: we exhibit some RGB person images after postprocessing the output of IP-Adapter. Right: the corresponding infrared images
with the same identities.

Table 2
Performance comparison on SYSU-MM01(%). GUR* denotes the results without camera information, RPNR† denotes the results we reproduced.
Type

Supervised

Unsupervised

Method

All-Search

Indoor-Search

Rank-1

Rank-10

Rank-20

mAP

Rank-1

Rank-10

Rank-20

mAP

Zero-Padding [48]
AlignGAN [49]
DDAG [50]
CM-MAS [51]
MTL [52]
CMT [53]
MTMFE [54]

14.80
42.4
54.75
61.99
67.25
71.88
69.47

54.12
85.0
90.39
92.87
95.38
96.45
96.42

71.33
93.7
95.81
97.25
98.46
98.87
99.11

15.95
40.7
53.02
60.02
64.29
68.75
66.41

20.58
45.9
61.02
67.01
69.58
76.90
71.72

68.38
87.6
94.06
97.02
96.66
97.68
97.19

85.79
94.4
98.41
99.32
99.03
99.64
98.97

26.92
54.3
67.98
72.95
74.37
79.91
76.38

ClusterContrast [12]
H2H [55]
OTLA [20]
ADCA [56]
PGM [57]
GUR* [23]

20.16
30.15
29.98
45.51
57.27
60.95

59.27
65.92
71.79
85.29
92.48
–

72.50
77.32
83.85
93.16
97.23
–

22.00
29.40
27.13
42.73
51.78
56.99

23.33
–
29.8
50.60
56.23
64.22

68.13
–
–
89.66
90.19
–

82.66
–
–
96.15
95.39
–

34.01
–
38.8
59.11
62.74
64.81

CCLNet [21]
Ours + CCLNet
OTPA [20]
Ours + OTPA
RPNR† [44]
Ours + RPNR
SDCL [45]
Ours + SDCL

54.03
55.40↑
58.90
61.77↑
62.73
63.01↑
64.49
64.95↑

88.80
90.21↑
90.64
91.09↑
93.00
93.47↑
–
95.06↑

95.02
95.80↑
96.27
96.40↑
96.77
97.33↑
–
98.31↑

50.19
50.92↑
54.40
56.35↑
57.64
58.41↑
63.24
63.59↑

56.68
57.05↑
62.41
64.67↑
66.79
69.50↑
71.37
71.57↑

91.14
92.98↑
91.30
91.67↑
94.84
95.53↑
–
97.48↑

97.23
97.17↓
94.70
95.56↑
97.92
98.18↑
–
99.44↑

65.12
62.24↓
67.01
68.80↑
72.45
74.16↑
76.90
77.44↑

Table 3
Performance comparison on RegDB(%). GUR* denotes the results without camera
information.
Type

Method

Rank-1

mAP

Rank-1

mAP

Supervised

Zero-Padding [48]
AlignGAN [49]
DDAG [50]
CM-MAS [51]
MTL [52]
CMT [53]
MTMFE [54]

17.75
57.9
69.3
84.25
89.9
95.17
85.04

18.90
53.6
63.5
80.32
85.6
87.30
82.52

16.63
56.3
68.1
82.57
88.3
91.97
81.11

17.82
53.4
61.8
78.31
84.1
84.46
79.59

ClusterContrast [12]
H2H [55]
OTLA [20]
ADCA [56]
PGM [57]
GUR* [23]

11.76
23.81
32.9
67.2
69.48
73.91

13.88
18.87
29.7
64.05
65.41
70.23

11.14
–
32.1
68.48
69.85
75.00

12.99
–
28.6
63.81
65.17
69.94

CCLNet [21]
Ours + CCLNet
OTPA [20]
Ours + OTPA

69.94
70.86↑
86.60
88.88↑

65.53
65.14↓
79.15
82.41↑

70.17
71.18↑
86.18
87.52↑

66.66
64.60↓
78.28
81.18↑

Unsupervised

Vis-to-IR

4.3. Ablation studies and visualization results
In order to show the generation results, we present extensive ablation studies and visual experiments in this section.
Given the text prompt, we take the diffusion model fine-tuned
by LoRA, as the basic image generation model. Due to the lack of
controllable generation ability in this baseline model, although the
generated images satisfy most of the text prompts, there are significant
differences between the generated images in terms of details, such as
clothing style, hairstyle, and other character features, which is shown
in the left of Fig. 6. However, VI-ReID is sensitive to person identities,
requiring images belong to the same ID to exhibit visual similarity.
Thus we randomly select one of the images generated by the basic
image generation model as the reference image for controllable image
generation, which is enclosed by a green bounding box in the left of
Fig. 6. With our controllable generation module, the generated person
images looks with similar person appearance while changing postures
and perspectives. Notably, the selected original person image in third
column has sunglasses on the face, which is also preserved in other
images. That indicates our approach is capable of making the generated
person images, from the same textual description, identity-aware.
Validation of GCP in VI-ReID. To verify whether the generated
dataset (GCP) provided by our method can effectively generalize to

IR-to-Vis

7

Pattern Recognition 165 (2025) 111561

X. Yu et al.

have yielded commendable results in controlling person details, but
images generated using the SoftEdge detection algorithm better align
with the characteristics of infrared imagery, such as blurred edges and
missing textures, thereby exhibiting superior performance in generating
infrared-style images.
Ablation on Generation Quality. We conduct ablation results in
terms of semantic metrics for assessing the quality and relevance of
synthesized images shown in Table 5. After LoRA fine-tuning, there
was no significant change in term of CLIPScore, indicating that learning
the style of infrared images does not impact the generative capability
of the diffusion model. Additionally, our designed text adapter show
competence in higher CLIP Score while lower FID, which suggests that
our generation framework performs better in comprehension of textual
conditions, and further reduces the discrepancy between synthesized
images and real ones, allowing for improving the discrimination of
existing VI-ReID model. Furthermore, we also show visualization results
of ablation studies in Fig. 9. Although the original Stable Diffusion
model are capable of generating textual relevant images, the images
may still suffer from semantic loss and confusion, as evidenced by
lower CLIP Score values. With the help of our proposed text adapter
alignment, our generation framework can produce images that more
accurately reflect textual information and achieve higher CLIP Score
ratings.
Compared with Real-world Images. In order to visually apprehend the difference between real-world images and generated images,
we show comparisons in Fig. 5. For each image pair, the image on
the left is the real-world image captured by camera in CUHK-PEDES,
while the image on the right is the generated image by our generation
framework with the corresponding text descriptions in CUHK-PEDES.
It appears that our generated images exhibit a strong resemblance to
real-world images.
Compared with Existing VI-ReID Datasets. The comparisons between the existing VI-ReID datasets and our generated dataset GCP is
shown in Fig. 8. We exhibit images from three datasets with each row
displaying 5 visible images and 5 infrared images corresponding to the
same person identity. It is evident that our introduced GCP significantly
enrich the variety of person scenarios in the VI-ReID dataset.
t-SNE visualization of pre-training. To verify the efficacy of GCP
dataset in VI-ReID pre-training, we randomly select 20 identities and
visualize the feature embedding space with t-SNE in Fig. 10. Different
colors indicate different identities, while different markers mean different modals. The result demonstrates that despite the baseline already
being capable of identifying ID discriminative features, with the help of
GCP, the VI-ReID model is able to further narrow the distance between
samples belonging to the same ID.
Training loss visualization. We document the variation of 𝐿𝐿𝑀
in Eq. (3) throughout the training process of the text adapter, illustrated
in Fig. 11. As training progresses, the loss values notably decrease and
eventually stabilize, indicating a heightened alignment between the
outputs of the original text conditional encoder and the LLM.

Fig. 7. Different controlling conditions for ControlNet and the corresponding generated
results. Top figures show the conditional image information under different edge
and depth detection methods. Transferred infrared images are shown in the bottom
respectively.

Table 4
Performance Comparison on GCP(%).
Method

mAP

Rank-1

Rank-10

Rank-20

RPNR Baseline
RPNR Trained on GCP
RPNR Trained on SYSU-MM01 + GCP

44.06
77.00
87.55

31.85
65.83
79.96

68.48
95.53
98.12

82.59
98.06
98.68

Table 5
Ablation results of our fine-tuned diffusion model in terms of various semantic metrics.
Order

1
2
3
4

Component
LoRA

Adapter

–
✓
–
✓

–
–
✓
✓

CLIP Score↑

FID↓

0.72
0.72
0.79
0.79

185.85
184.73
179.49
182.39

real-world scenarios, we attempt to train and/or test the model using
only GCP dataset. We partition the GCP generated dataset with reference to the SYSU-MM01 dataset. Specifically, our training set contains
206 person identities, and the testing set includes 51 person identities.
We conduct tests based on RPNR in the all-search testing mode
similar to that of the SYSU-MM01 dataset. The detailed results are
presented in Table 4. It can be observed in the first row that even
without training on the GCP dataset, the model still exhibits a certain
person re-identification capability on the GCP testing set. Moreover,
the models trained on the GCP dataset or the SYSU-MM01 dataset
augmented with the GCP dataset show significant improvements in the
results on the GCP testing set. Notably, when the SYSU-MM01 training
set is added to the GCP training set, the model’s performance is further
enhanced to a substantial extent. This indicates that training on a
real dataset can effectively augment the model’s recognition capability
on the GCP testing set. Consequently, it indirectly demonstrates that
the GCP dataset has the capacity to validate the performance of the
model in real person re-identification scenarios, thus providing strong
evidence to validate the effectiveness of our method.
Controlling Conditions. Fig. 7 shows the different controlling
conditions for ControlNet and the corresponding generated results in
our style transfer module. CannyEdge and SoftEdge detect the edge
map while Depth detect depth information of the image. In the process of infrared style transfer, it is necessary to control the person
characters as much as possible, but the depth map lacks such detailed
information, and therefore fail to meet the requirements of generating
the same person. On the other hand, both CannyEdge and SoftEdge

5. Conclusion
In this paper, due to the limited amount of training data in VI-ReID,
we propose a controllable diffusion framework for infrared person image generation and re-identification. This framework utilizes LoRA for
fine-tuning with VI-ReID dataset to understand the infrared modality,
and leverage a text adapter to transfer the semantic understanding
ability of LLMs to the fine-tuned diffusion models. To keep the identities of generated person images from the same textual description,
we design a controllable generation module that make the generated
images identity-aware. Our approach is capable of generating diverse
visible and infrared person images with textual descriptions and thus
obtain a new generated VI-ReID dataset. This dataset enables us to
expand the existing VI-ReID dataset, which consistently improves the
model performance. Experimental results validate the effectiveness of
our method.
8


codex
我已经读到每篇的核心方法框架。现在做一轮补读，重点补那些输出中被截断的段落，比如 HPRNet 的非局部多尺度模块、扩散生成文的可控生成部分、InfinitePerson 的数据管线细节。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf' - | nl -ba | sed -n '240,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   240	
   241	minE(xi ,yi )∈S yi − g( f (xi )) ,
   242	(1)
   243	
   244	In this formula, where g represents the classifier, the quality
   245	of features learned by the network model is influenced by the
   246	data distribution in the dataset S.
   247	The above statement reveals two major shortcomings.
   248	A challenge is to collect and annotate the training dataset S.
   249	
   250	Fig. 2. Demonstrating the creation of a 3D character using MakeHuman, and
   251	adding the generated UV texture maps to the 3D character models in UE4.
   252	
   253	It often takes a lot of time and manpower to construct and
   254	accurately label pedestrians correctly in multiple different
   255	scenarios, which is very challenging for manual annotation.
   256	In addition, the data distribution of dataset S is susceptible to
   257	interference from changes in background, lighting, and other
   258	factors. Therefore, Re-ID data collected in a single scenario
   259	is usually difficult to successfully transfer to other scenarios.
   260	As shown in Table II, the evaluation results of the Re-ID
   261	models trained on CUHK03 and Market-1501 were transferred
   262	to the MSMT17 test set show that both Rank-1 and mAP
   263	scores are very low, indicating a significant gap between the
   264	domains and also revealing the weaknesses of the current
   265	Re-ID task.
   266	B. 3D Virtual Character Generation
   267	1) Character Model and Animation: We create a
   268	diverse range of male and female models, we employ
   269	MakeHuman [34] to generate various character types based
   270	on factors such as gender, age, torso, facial features, and
   271	more. To further enhance diversity, we leverage different
   272	hairstyles, beards, skin textures, and other assets to customize
   273	and adorn our characters. In this project, we export the 3D
   274	character models as.fbx files, which are then imported into
   275	Unreal Engine 4 for adding animations. In total, we create
   276	4,700 3D character models, all without clothing textures.
   277	However, one limitation of MakeHuman-generated characters
   278	is the absence of walking animations. To address this issue,
   279	we add skeleton and walking animations suitable for game
   280	
   281	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   282	
   283	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   284	
   285	Fig. 3.
   286	
   287	3163
   288	
   289	The InfinitePerson pipeline, which shows the comprehensive process of generating our synthetic datasets.
   290	
   291	engines to these models. In this way, our 3D character models
   292	can execute a range of actions in scenes, including walking,
   293	running, jumping, etc. which provides crucial support for
   294	simulating real-world pedestrian behavior.
   295	2) UV Mapping: The surface of clothing in real life has
   296	rich texture details, while the 3D model of clothing itself
   297	has no texture information, as shown in Figure 2. UV mapping [35] is a method of enhancing the shading details of
   298	3D models by mapping spatial coordinate points of objects
   299	to texture coordinates and retrieving the corresponding values
   300	from texture images. When adding textures, there are two
   301	essential coordinate systems to consider: one represents spatial
   302	coordinates describing the vertex positions (X, Y, Z ), and
   303	the other represents the UV coordinates. The UV coordinate
   304	system is used to determine the position information of each
   305	point on the texture image, aligning it with the geometric
   306	structure of the 3D model. In the case of complex geometric
   307	mappings, UV mapping is often necessary to project points
   308	from a 2D texture image onto a 3D model, followed by
   309	rendering through a rendering engine. This process enables the
   310	3D clothing model to have rich texture information, as shown
   311	in Figure 2.
   312	3) UV Texture Maps: UV texture is a technique that ensures
   313	that every point of an image is precisely corresponded to
   314	the surface of the model, while the gaps between points are
   315	processed for image smoothing interpolation. UV Mapping
   316	technology allows us to map UV texture maps onto 3D
   317	clothing models, providing them with rich texture information.
   318	However, each 3D clothing model is usually equipped with
   319	one UV texture map, which means that each clothing style
   320	has only one texture map available, thus limiting the diversity
   321	of 3D character models that can be created. In addition,
   322	although the MakeHuman community has provided a variety
   323	of clothing types, including those we wear daily, their selection
   324	and variety are still limited. Therefore, this paper proposes to
   325	
   326	generate diverse sets of clothing models by changing the UV
   327	texture maps of existing clothing models. This method can not
   328	only expand the diversity of 3D character models, but also
   329	provide designers with a broader creative space. To obtain
   330	diverse clothing, we need to generate different UV texture
   331	maps that adhere to the definition of clothing texture. Stable
   332	Diffusion [11] is a powerful large-scale generative model
   333	within the realm of computer vision. However, it often lacks
   334	precise control over image generation. To generate a reasonable UV texture map, it is necessary to ensure that the large
   335	diffusion model has the ability to generate high-quality texture
   336	images in a controllable manner. ControlNet [10] effectively
   337	solves this problem by controlling Stable Diffusion [11].
   338	We use normal maps dataset [36] as additional input to train
   339	ControlNet [10] to guide Stable Diffusion [11] in generating
   340	reasonable UV maps. The benefit of using normal maps
   341	dataset as additional input is that the generated UV texture
   342	maps align with the model’s surface structure. This enables
   343	the generation of high-quality textures on various types of
   344	clothing. We import the 3D clothing models and UV texture
   345	maps into the Unreal Engine 4 [37], and utilize the material
   346	system provided by the game engine to generate a wide variety
   347	of clothing models infinitely to some extent. As can be seen
   348	from Figure 4, various 3D character models can be obtained
   349	using the generated UV maps.
   350	C. Sub-Monitoring Network
   351	Previous data collection methods for synthetic datasets
   352	often focused on specific scenarios or research objectives.
   353	For instance, SynPerson [6] focuses on studying the effects
   354	of lighting without simulating pedestrian movement, while
   355	UnrealPerson [8] randomly collects pedestrian images in large
   356	scenes. This approach restricts future researchers from designing complex Re-ID scenarios and results in a lack of flexibility
   357	in data collection.
   358	
   359	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   360	
   361	3164
   362	
   363	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025
   364	
   365	appearing in the camera’s field of view and maximizes the
   366	efficiency of data collection. This automated process helps to
   367	reduce human intervention and improve data consistency.
   368	Our method provides several benefits in terms of data collection. Firstly, it allows us to better control over essential factors
   369	such as viewpoint, background, and lighting when collecting
   370	pedestrian images, which greatly affect the performance of the
   371	Re-ID tasks. The monitoring scene shown in Figure 5 includes
   372	varied lighting regions. Secondly, simulating pedestrian occlusion is also an important aspect and excessive occlusion may
   373	lead to incorrect annotation and recognition results. With our
   374	method, we can design pedestrian routes by controlling the
   375	distance between characters and the camera’s angle, and introduce moderate occlusion without compromising the accuracy
   376	of data annotation.This allows us to simulate more realistic
   377	occlusion scenarios in the data and thus better assess the
   378	performance of the Re-ID tasks.
   379	D. InfinitPerson Dataset Overview
   380	
   381	Fig. 4. Different categories of UV texture maps. (a) Regular UV texture maps
   382	maintain the corresponding shape of the clothes. (b) Dresses and shoes with
   383	irregular UV texture maps from top to bottom. The dashed box contains the
   384	original UV map of the clothes, and others are generated by Stable Diffusion.
   385	
   386	In real surveillance scenarios, monitoring networks within
   387	an area typically consist of multiple sub-networks, and each
   388	subnet covers the limited field of view of surveillance cameras.
   389	To ensure comprehensive monitoring and security, multiple sub-networks must operate simultaneously, with each
   390	subnet responsible for monitoring different areas or locations. Inspired by this practical monitoring network structure,
   391	we introduce a sub-network data collection method in our
   392	research, aiming to collect pedestrian images within each subnetwork using a limited number of cameras, typically four
   393	to six. This approach replicates the operating mode of a real
   394	monitoring network, where each sub-network is accountable
   395	for distinct monitoring zones. In this way, we can more
   396	accurately simulate the operation of the monitoring network,
   397	while controlling the quality of data and the collection process
   398	more effectively and flexibly.
   399	To simulate the movement of characters, we establish a
   400	circular pedestrian path, as shown in Figure 5. During the
   401	image collection process, we introduce pedestrians in batches
   402	and let them follow the predetermined paths to simulate their
   403	movements. After completing one round, the program will
   404	automatically switch to the next batch of pedestrians. In terms
   405	of data collection, our circular pedestrian path strategy is
   406	an effective method to ensure the probability of pedestrians
   407	
   408	1) Data Engine: Our Infiniteperson dataset is created using
   409	Unreal Engine (UE4), one of the most well-known and widely
   410	licensed game engines. This mature game engine boasts powerful rendering capabilities and a realistic physical material
   411	system, which can simulate real surveillance scenes and render
   412	lifelike 3D characters effectively. The Unreal Engine market
   413	offers a wealth of high-quality scene resources, enabling the
   414	generation of Re-ID datasets for a wide range of scenarios.
   415	Additionally, visual variables such as weather, lighting, and
   416	viewpoint can be customized to suit requirements, making the
   417	synthetic dataset highly scalable.
   418	2) Identities: To create 3D human models, we employ
   419	open-source software called MakeHuman [34] to generate
   420	highly realistic human models. Similar to OWD [25], which
   421	sets pedestrians face features invisible, our dataset does not
   422	rely on facial features to distinguish identities. We randomly
   423	generate a large number of character models with different skin
   424	tones, body types (including height and weight), and hairstyles
   425	by using the plugin of MakeHuman [34]. For 3D clothing
   426	models, we carefully select 141 different types of clothing,
   427	as well as 40 types of pedestrian accessories such as hats and
   428	glasses. In our research, we combine these clothing models,
   429	which initially lack UV textures, with character models, and
   430	then import them into the game engine, and use UV texture
   431	maps generated by Stable diffusion [11] as texture images
   432	of the characters clothing texture. Ultimately, we generate
   433	4,700 3D character models. Figure 6 (a) shows images of
   434	different pedestrians in different scenes, with realistic and
   435	diverse clothing textures.
   436	3) Scenario Design: Unreal Engine (UE4) has a
   437	well-established community with rich resources. In this
   438	community, we have selected 9 diverse scenes, including
   439	indoor and outdoor scenes such as streets, banks, shopping
   440	malls, fields, rural areas and so on, all of which are readily
   441	available from the Unreal Engine community. Within the
   442	Unreal Engine environment, we employ our proposed
   443	sub-monitoring network to position the cameras and define
   444	circular paths for pedestrians, providing effective and flexible
   445	control over data quality and collection. Figure 6 (b) shows
   446	
   447	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   448	
   449	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   450	
   451	3165
   452	
   453	Fig. 5. Example configuration of sub-monitoring network and 3D character models movements. A total of cameras are installed and the monitoring scenes
   454	show each camera’s field of view. The yellow dot represents the destination points we set, and the yellow line represents our predetermined pedestrian
   455	movement route.
   456	
   457	535,636 images of 4,700 pedestrians using a total of 72 virtual
   458	cameras. Figure 6 provides some images from the InfinitePerson dataset. Cropped images typically have diverse lighting,
   459	viewpoints, backgrounds, poses, resolutions and occlusions.
   460	IV. E XPERIMENTS
   461	A. Datasets and Evaluation
   462	
   463	Fig. 6. Sample images from the proposed InfinitePerson dataset, (a) Identity:
   464	Some images from the InfinitePerson dataset have different pedestrians in
   465	different scenes. (b) Illustrations: There are three images of pedestrians, and
   466	under different lighting conditions, we divide the lighting scene into bright
   467	and dark scenes. (c) Screens: Images of the same pedestrian in 6 different
   468	scenes.
   469	
   470	some pedestrian images under different lighting conditions in
   471	the proposed InfinitePerson dataset, including bright and dark
   472	lighting scenes. Figure 6 (c) presents images of the same
   473	pedestrian in six different scenes, demonstrating the diversity
   474	of pedestrian appearances captured by the dataset in different
   475	environmental backgrounds.
   476	4) Data Annotation: During data collection, we use
   477	UnrealCV [38], [39] to gather instance segmentation annotations for every image captured by the virtual cameras. After
   478	collecting the data, we use appropriate tools to precisely crop
   479	pedestrians from the image.
   480	5) Summary: We set up 12 sub-monitoring networks across
   481	9 different scenarios and successfully captured a total of
   482	
   483	Three popular real-world person Re-ID datasets are
   484	used for generalization evaluation, including CUHK03 [3],
   485	Market-1501 [4] and MSMT17 [5]. The CUHK03 [3] dataset
   486	contains 14,097 images of 1,467 identities. The CUHK03-NP
   487	is split into a training set and a testing set. The training set
   488	consists of 7,365 images of 767 identities, while the testing
   489	set consists of the remaining 6,732 images of 700 identities.
   490	The Market-1501 dataset1 [4], gathered on the campus of
   491	Tsinghua University, comprises 1,501 pedestrians recorded
   492	by 6 cameras, totaling 32,668 labeled images. The training
   493	set consists of 12,936 images of 751 identities, and the
   494	testing set consists of 19,732 images of 750 identities. The
   495	MSMT17 [5] dataset encompasses 126,441 labeled images of
   496	4,101 identities. The training set comprises 32,621 images
   497	corresponding to 1,041 identities. In the testing set, 11,659
   498	bounding boxes are designated as query images, while another
   499	82,161 bounding boxes are designated as gallery images.
   500	We compare our dataset with multiple synthetic datasets
   501	to verify its generalization ability for Re-ID. All evaluations
   502	adhere to the single-query evaluation protocol [40]. We utilize
   503	the Cumulative Matching Characteristic (CMC) [41], with
   504	specific emphasis on Rank-1 accuracy, Rank-5 accuracy, and
   505	mean Average Precision (mAP) [42] as the primary performance metrics. The synthetic dataset’s quality is evaluated
   506	through direct transfer.
   507	B. Implementation Details
   508	We employ ResNet-50 [46] as our backbone for all our
   509	experiments. The cross-entropy loss function is utilized, and
   510	all pedestrian images are resized to 256 × 128. The training
   511	batch size is configured to 64 samples. For optimization,
   512	
   513	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   514	
   515	3166
   516	
   517	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025
   518	
   519	TABLE II
   520	P ERFORMANCE C OMPARISON W ITH E XISTING R EAL -W ORLD AND S YNTHETIC DATASETS ON CUHK03-NP, M ARKET-1501 AND MSMT17,
   521	R ESPECTIVELY. U NREAL P ERSON * M EANS THE T RAINING DATA E XTRACTED F ROM UNREAL _ V 1.1,
   522	UNREAL _ V 2.1, UNREAL _ V 3.1 AND UNREAL _ V 4.1
   523	
   524	we employ the SGD optimizer with a momentum of 0.9. The
   525	initial learning rate for the backbone network is set to 0.005,
   526	which is then decayed to 0.0005 at the 10th epoch. Training
   527	concludes after 40 epochs.
   528	C. Direct Cross-Dataset Evaluation
   529	Direct cross-dataset evaluation involves training a model on
   530	a particular dataset and subsequently assessing its performance
   531	on another dataset, avoiding any intermediate fine-tuning or
   532	adjustments. This evaluation method offers a robust assessment of the model’s ability to generalize across diverse
   533	datasets, providing valuable insights into its overall performance on different datasets. We evaluate our synthetic
   534	dataset’s quality through direct cross-dataset evaluation.
   535	To assess the effectiveness of our synthetic dataset, we use
   536	direct transfer performance on real-world datasets as the
   537	primary indicator of synthetic data quality, since direct transfer
   538	is the foundation of all other tasks. We employ CUHK03 [3],
   539	Market-1501 [4] and MSMT17 [5] as testing sets. The evaluation results are provided in Table II. It is clear from the results
   540	that our synthetic dataset outperforms existing synthetic and
   541	real-world datasets in direct transfer experiments. Compared
   542	to the UnrealPerson [8] collected using the same game engine
   543	as ours, our synthetic dataset also achieved better performance.
   544	The reason why we can surpass other composite datasets
   545	is that the pedestrian data collected by our proposed sub
   546	monitoring network data collection method includes various
   547	key factors (such as lighting, perspective, occlusion, etc.), and
   548	the Unreal Engine (UE4) used in our dataset has stronger simulation capabilities. Compared to UnrealPerson [8] collected
   549	using the same game engine as ours, our 3D character models
   550	have more complete and reasonable clothing textures than
   551	UnrealPerson’s 3D character models, so the models trained
   552	on our synthetic dataset has better performance.
   553	D. Generalizable Person Re-ID
   554	To verify the generalization of our dataset, we choose to use
   555	various methods in the Re-ID field for comparison, including
   556	QAConv2.0 [43] based on ResNet-50 [46], TransReID [26]
   557	based on Transformer [47], PHA [44] based on Discrete Haar
   558	
   559	Wavelet Transform [48], and Domain generalizable Re-ID
   560	method SuA-SpML [45]. The results of direct cross dataset

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '180,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   180	a non-local weighted multi-scale perception (NWMP)
   181	module and a parsing reconstructed exploration (PRE)
   182	module for the CC-ReID task.
   183	• The proposed NWMP module effectively captures the
   184	global receptive field through a multi-scale progressive
   185	learning strategy for features unrelated to clothing. The
   186	PRE module incorporates clothing parsing and a human
   187	local component generation strategy to achieve more
   188	accurate human component reconstruction and better distinguish between clothing-related and unrelated features.
   189	• Extensive experiments are conducted on the public CCReID datasets LTCC, PRCC, and CCVID to demonstrate
   190	the competitive accuracy of our proposed method. Additionally, ablation studies and visualizations are performed
   191	to verify the contribution of each module to the overall
   192	performance of HPRNet.
   193	The rest of the manuscript is organized as follows: firstly
   194	Section II briefly revisits existing CC-ReID methods, including
   195	data-driven disentanglement methods, feature-driven disentanglement methods, and Human Reconstruction-based CC-ReID
   196	methods. In Section III, we describe the proposed HPRNet
   197	method. Section IV presents and discusses the experimental
   198	results and ablation study. Finally, concluding remarks and
   199	future work are presented in Section V.
   200	II. R ELATED W ORK
   201	In this section, related work on CC-ReID is reviewed.
   202	First, we describe the data-driven disentanglement methods for
   203	
   204	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   205	
   206	XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK
   207	
   208	CC-ReID (Subsection II-A). Feature-driven disentanglement
   209	strategies are revisited in Subsection II-B. Finally, we revise
   210	the human reconstruction-based methods in Subsection II-C,
   211	followed by a short statement of the novelty of this work
   212	compared to existing works (Subsection II-D).
   213	A. Data-Driven Disentanglement CC-ReID Methods
   214	As mentioned previously, CC-ReID methods based on datadriven disentanglement focus on entailing random alterations
   215	within the designated outfit area to reduce reliance on color
   216	or texture attributes. The primary strategy employed by these
   217	methods involves increasing the amount of dataset at the data
   218	level, which includes expanding data templates and synthesizing clothing-relevant features [9], [10], [21]. Specifically,
   219	a semantic-guided sampling strategy was proposed in [10]
   220	to enforce the learning of clothing-independent features by
   221	recognizing pedestrians’ outfit appearances (e.g., tops and
   222	pants) and sampling features from other individuals’ images
   223	for CC-ReID. The method for this same task presented in [22]
   224	incorporated identity-aware Mixstyle and graph enhancement
   225	modules to construct variable clothing-based fine-grained style
   226	transformation features and cross-domain style transfer-based
   227	enhanced samples. CCPG [23] was used to enhance data
   228	diversity of different pedestrians wearing the same clothes and
   229	the same one with different clothes. Other similar methods
   230	have been proposed in the literature [24], [25] addressing the CC-ReID task by embracing different data-driven
   231	disentanglement-based strategies.
   232	
   233	149
   234	
   235	images utilizing generative models (such as GANs [37] and
   236	diffusion models [38]), combining them with the original
   237	pedestrian instances to perform contrastive training on the
   238	reconstructed images, to reduce the interference of clothingrelevant features [13], [14], [39]. In addition, other methods
   239	such as [40], [41], [42] adopted the idea of pedestrian reconstruction to address the CC-ReID task.
   240	D. Contribution
   241	Although existing CC-ReID technologies have achieved
   242	promising performance, they lack realistic guidance on generated samples, leading to a potentially corrupted feature space
   243	and uncontrollable results. By contrast, we propose HPRNet, a novel framework that integrates a non-local weighted
   244	multi-scale perception module and a parsing-generation
   245	reconstructed exploration module for the CC-ReID task.
   246	By leveraging component parsing reconstruction and a
   247	multi-scale non-local weighted attention interaction strategy,
   248	HPRNet effectively enhances CC-ReID performance and mitigates the performance degradation caused by the feature
   249	decomposition process present in state-of-the-art approaches.
   250	III. P ROPOSED HPRN ET F RAMEWORK
   251	In this section, we introduce the proposed HPRNet framework for CC-ReID tasks. A specific sub-model constituting the
   252	overall framework of this study is introduced. The loss function and differences from existing methods are also described
   253	separately, with the details as follows:
   254	
   255	B. Feature-Driven Disentanglement CC-ReID Methods
   256	
   257	A. Overview
   258	
   259	Differently than the previous category, feature-driven methods for the CC-ReID problem hinge on distinguishing
   260	clothing-related and clothing-unrelated attributes by using
   261	feature extraction [2], [11], [12], [26]. An adversarial feature
   262	learning method text description-based, called DIFFER [11],
   263	has been proposed for the separation of identity features.
   264	Specifically, the feature space is partitioned into multiple nonoverlapping subspaces. Gradient inversion is then employed
   265	to distinguish identity-relevant features from non-biological
   266	ones, informing the model to better solve the CC-ReID task.
   267	A 3D InvarReID framework was designed in [27] to disentangle and reconstruct 3D outfit body shapes for the CC-ReID
   268	task. DLAW [3] incorporated an adaptive cloth-changed region
   269	localization strategy and a modeling scheme that captures the
   270	correlation of cloth changes at both the image and feature
   271	levels. Other related strategies [28], [29], [30], [31], [32]
   272	rely on feature-driven disentanglement, employing a learning
   273	approach that is unaffected by clothing features. Overall, these
   274	methods distinguish related and unrelated cues to clothing at
   275	the feature level.
   276	
   277	In this section, we first establish the mathematical notations
   278	and formally state the modeling problem addressed by the
   279	CC-ReID methods (Section III-B). Subsequently, the proposed
   280	HPRNet framework is introduced, as shown in Fig. 2. Subsequently, the nonlocal weighted multi-scale perception module
   281	(Section III-C) and the human body parsing reconstructed
   282	exploration module (Section III-D) are described. Finally, the
   283	optimization process and the differences between the proposed
   284	method and other baselines are discussed in Sections III-E
   285	and III-F, respectively.
   286	
   287	C. Human Reconstruction-Based CC-ReID Methods
   288	Recently, human reconstruction learning-based solutions,
   289	which are essentially feature-driven methods, have gained
   290	widespread attention within the CC-ReID community [33],
   291	[34], [35], [36]. These methods mainly reconstruct pedestrian
   292	
   293	B. Notation and Problem Statement
   294	Given a ReID gallery dataset, which is denoted as G =
   295	{(g1 , y1 ), . . . , (ga , ya ), . . . , (gN , yL )}, where N is the size of the
   296	gallery set, and L is the amount of different identities in the
   297	set. The probe set q p contains M pedestrian images, which are
   298	the targets to be queried. In the specific ReID task, the goal
   299	was to accurately retrieve pedestrian images from a gallery
   300	set that matched a probe image (a query image). It involves
   301	sorting all images in the gallery set based on their similarity
   302	scores with the probe set, which is given by
   303	a∗ = arg min d(φ(q; θ), φ(ga ; θ)),
   304	
   305	(1)
   306	
   307	a=1,...,N
   308	
   309	where d(·, ·) denotes the measurement function calculated
   310	using the similarity between the two feature vectors. φ(·; θ) is a
   311	
   312	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   313	
   314	150
   315	
   316	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   317	
   318	Fig. 2. Diagram showing the design of the HPRNet framework. The framework comprises three key components: parsing, reconstruction, and feature extraction
   319	models. These components are integrated into the NWMP and PRE modules. The NWMP module is designed to capture the global receptive field of each
   320	person, while the PRE module further distinguishes clothing-related and clothing-independent features through a parsing-reconstruction pipeline.
   321	
   322	parametric function whose parameters are collectively denoted
   323	by θ. We aim to find the image ga in the gallery set that is most
   324	similar to probe q p , that is, a∗ , such that distance d minimizes
   325	the image index in the gallery set.
   326	However, the CC-ReID task poses a greater challenge than
   327	its original ReID form, as it necessitates distinguishing pedestrians wearing the same clothing but belonging to different
   328	identities, or the same person wearing different clothes. In
   329	this case, the model must be able to differentiate between
   330	those who are dressed similarly but are different people.
   331	Mathematically, the differences imprinted by the formulation
   332	of the CC-ReID task to ReID give rise to the following
   333	objectives:
   334	min
   335	
   336	M X
   337	M
   338	X
   339	
   340	(d(φ(q p ; θ), φ(ga ; θ)) − d(φ(qn ; θ), φ(ga ; θ))),
   341	
   342	(2)
   343	
   344	p=1 n=1
   345	
   346	where p and a are the p-th and a-th samples in each batch,
   347	respectively. q p and ga are pedestrians with the same identity
   348	but wearing different clothing. qn denotes dressed clothes
   349	similar to ga but with different identities. M is the size of
   350	each mini-batch.
   351	C. Non-Local Multi-Scale Perception Module
   352	As mentioned at the end of Section II, existing methods mainly focus on pedestrian clothing color change and
   353	appearance transfer, but ignore the local contextual correlation
   354	features of human appearance, resulting in decreased model
   355	performance. This section proposes a human appearance nonlocal multi-scale perception module involving a weighted
   356	
   357	non-local perception mechanism (WNPM) and a dual-scale
   358	progressive learning strategy (DPLS). Details are provided in
   359	the following sections.
   360	1) WNPM: Specifically, the weighted non-local perception,
   361	inspired by the attention mechanism and AGW-ReID [43],
   362	[44], captures both prominent human appearance features
   363	and local region correlations by assigning weights to each
   364	pedestrian region. This strategy is devoted to obtaining the
   365	local contextual receptive field of pedestrian appearance and
   366	determining its multi-perspective receptive field. Weights are
   367	proposed to associate the region’s relationships (such as
   368	similarity or contextual correlation), which determine the
   369	importance of each person’s feature points by calculating their
   370	relationship with others. This study constructed an attentionmap-based weight learning, which reflects the interactive
   371	relationships between different regions. The contribution of
   372	each feature region depends not only on its own, but also on
   373	its relationship with other feature regions, thus achieving an
   374	understanding of global receptive fields (such as the overall
   375	posture and shape contour of pedestrians). Meanwhile, it suppresses the local features that may change owing to changes
   376	in perspective or occlusion and improves the accuracy and
   377	robustness of the model under harsh conditions. Specifically,
   378	the procedure followed by the WNPM module to produce
   379	map-based weights wi j , where i and j denote i-th and j-th
   380	regions, respectively, is given by
   381	wi j =
   382	
   383	T
   384	1 X fi · f j
   385	,
   386	T
   387	k fnle k2
   388	i=1, j=1
   389	
   390	fnlp =
   391	
   392	T
   393	X
   394	
   395	wi j · fki j ,
   396	
   397	i=1, j=1
   398	
   399	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   400	
   401	(3)
   402	
   403	XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK
   404	
   405	where fi and f j are the features of the two nearest neighboring
   406	local regions. fnle denotes the output of the Non-local Encoder
   407	with the ResNet50 [45] as the backbone. T denotes the size
   408	of the local region. fki j denotes the k-th feature map in region
   409	(i, j) and fnlp is the nonlocal perception feature.
   410	2) DPLS: To acquire the multi-scale pedestrian features
   411	and achieve perceptual learning from local to whole, we
   412	propose a dual-scale progressive learning strategy (DPLS),
   413	which is based on the non-local perception patterns. Specifically, feature fnlp from WNPM is split into K equal blocks
   414	first. Then, the max-pooling operations of 2 × 2 and 4 × 4
   415	were implemented in each case. Next, we combine the maxpooling results of 2 × 2 and 4 × 4 separately to obtain two
   416	different scales of local features. Then, fnlp is included to
   417	form three different scales of pedestrian features to achieve
   418	a progressive feature learning strategy. Finally, a 1 × 1 global
   419	pooling operation was performed on each of the three scale
   420	features mentioned above to complete the combination and
   421	classification. This process can be summarized as follows.
   422	!
   423	K 
   424	X
   425	fk − 2
   426	,
   427	(4a)
   428	fk2 = concat
   429	s
   430	k=1
   431	!
   432	K 
   433	X
   434	fk − 4
   435	fk4 = concat
   436	,
   437	(4b)
   438	s
   439	k=1
   440	
   441	
   442	fcls = concat conv fnlp , 1 × 1
   443	+ conv ( fk2 , 1 × 1) + conv ( fk4 , 1 × 1))),
   444	
   445	(5)
   446	
   447	where fk denotes the k-th feature block. bc is the pooling
   448	operation and s is the stride. concat() is the feature concatenation. fk2 and fk4 are the concatenated features after the max
   449	pooling results with kernel sizes 2 × 2 and 4 × 4, respectively.
   450	conv( f , 1 × 1) denotes a 1 × 1 convolution operation on f
   451	for normalization to a uniform dimension. fcls is a multi-scale
   452	progressive learning feature (the dimensions of fcls are 6144
   453	and are averaged from fk2 , fk4 and fnlp ).
   454	After this feature is learned, the multi-classification loss Lcla
   455	is expressed as
   456	Lcla = −
   457	
   458	L−1
   459	X
   460	
   461	yi j log(ybi j ),
   462	
   463	(6)
   464	
   465	j=0
   466	
   467	where yi j is the ground truth when sample i belongs to
   468	class j, and ybi j is the predicted value.
   469	In addition, to preserve the consistency of the pedestrian identity and clothing, we also exploit the identity (ID)
   470	loss Lid as in [11], which is given by
   471	!
   472	N
   473	X
   474	y(xi , li )
   475	Lid = −
   476	log PNid
   477	,
   478	(7)
   479	j=1 y(xi , l j )
   480	i=1
   481	where N denotes the total number of samples in the training
   482	set. y(xi , li ) is the probability that the model
   483	PNid predicts that
   484	sample xi belongs to its true label li .
   485	j=1 y(xi , l j ) is the
   486	normalized sum of the predicted probabilities for all possible
   487	identity categories l j .
   488	
   489	151
   490	
   491	Partly inspired by CAL [11], we extract the invariant features for clothing variations by penalizing the model’s ability
   492	to predict clothing attributes, which is described as follows:
   493	min Lclo (CϕC (gθ (xi ), yCi )),
   494	
   495	(8)
   496	
   497	ϕ
   498	
   499	Lclo = −
   500	
   501	N
   502	X
   503	
   504	e( fi ·ϕyi c/τ)
   505	log PLC
   506	( fi ·ϕ j /τ)
   507	j=1 e
   508	i=1
   509	
   510	!
   511	,
   512	
   513	(9)
   514	
   515	where CϕC (•) is the clothes classifier and CϕC (gθ (xi ) denotes
   516	the predicted clothes. yCi is the ground truth for the clothing
   517	label. gθ (xi ) denotes the result of fi after l2 -normalization, and
   518	ϕ j is the output of the weights for j-th clothes classifier after
   519	l2 -normalization. N is the size of the input batch. LC is the
   520	volume of the category data for clothing. τ is a temperature

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf' - | nl -ba | sed -n '220,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   220	VI-ReID to benefit from large-scale cross-modal pretraining.
   221	To deal with this issue, we propose a generative framework for
   222	cross-modal person re-identification. Our method mainly consists of
   223	three steps: (1) Diffusion Model Fine-tuning via Infrared Images. (2)
   224	Generative Text Adapter Alignment. (3) Controllable Image Generation
   225	and Inference, which is presented in Fig. 2. Based on our generative framework, we also introduce a Generated Cross-modal Person
   226	Dataset, called GCP for further training and evaluation. Thanks to
   227	the great success of AIGC, our diffusion model enables us to generate
   228	extensive infrared person photos and therefore allows for learning the
   229	discriminative ReID model without any annotations.
   230	Our proposed cross-modal person image generation framework is
   231	shown in Fig. 2. During the fine-tuning, we first feed the original
   232	visible and infrared images from SYSU-MM01 into BLIP to generate
   233	corresponding text prompts. Specific prefix words, like ’a RGB photo’
   234	or ’an infrared photo’, are added to the prompts, which encourages the
   235	diffusion model to recognize the concept of infrared images. We then
   236	freeze all parameters of Stable Diffusion and use LoRA to fine-tune the
   237	U-Net of Image Generator with our constructed image–text pairs. After
   238	that, we design a new text adapter to refine and align the output of
   239	the text conditional encoder with that of LLMs, and therefore enables
   240	us to generate the generation-friendly-friendly embeddings. Finally, we
   241	propose a controllable module to make the generation be identityaware when using the same text prompts. Meticulous post-processing
   242	is adopted to obtain appropriate cross-modal person images.
   243	
   244	𝑊 ′ = 𝑊0 + 𝛥𝑊 = 𝑊0 + 𝐴𝐵 𝑇
   245	
   246	(1)
   247	
   248	where 𝐴 ∈ R𝑑×𝑟 , 𝐵 ∈ R𝑘×𝑟 . During training process, only A and B are
   249	trainable parameters. In our method, LoRA is only used to optimize our
   250	image generator. We select visible and infrared photos in SYSU-MM01
   251	as the image training dataset 𝑋𝑣 = {𝑥𝑖𝑣 }, 𝑋𝑟 = {𝑥𝑖𝑟 }. Then we use BLIP
   252	to get the original text description 𝑃𝑣 = {𝑝𝑖𝑣 }, 𝑃𝑟 = {𝑝𝑖𝑟 }. Note that here
   253	we choose both visible and infrared images, targeting on teaching the
   254	diffusion model to learn the difference of the cross-modal images. Our
   255	experimental results also show that even the subset of SYSU-MM01,
   256	e.g., images from Cam1 and Cam2, is sufficient for this fine-tuning
   257	process. At last, to better reflect the characteristics of infrared images,
   258	we also modify the original text by adding the prefix words such as
   259	𝑌𝑣 = {𝑦𝑖𝑣 } =‘‘A RGB photo, [𝑝𝑖𝑣 ]’’, 𝑌𝑟 = {𝑦𝑖𝑟 } =‘‘An infrared photo, [𝑝𝑖𝑟 ]’’.
   260	Finally, we utilize both the image and text information to fine-tune
   261	Stable Diffusion model.
   262	3.3. Generative text adapter alignment
   263	Note that the text conditional encoder of the original Stable Diffusion model is CLIP ViT-L/14, which, however, is trained with text–
   264	image contrastive learning, that not specifically designed for image
   265	generation tasks. This leads to inadequate semantic understanding
   266	when guiding image creation with natural language prompts.
   267	Confronting this challenge, large language models (LLMs) attract
   268	our attention due to their remarkable capability in semantic comprehension. Specifically, we utilize the open-sourced large language
   269	3
   270	
   271	Pattern Recognition 165 (2025) 111561
   272	
   273	X. Yu et al.
   274	
   275	Fig. 2. The pipeline of our proposed method. Left: we first use BLIP to generate some text prompts for the images from SYSU-MM01. Specific prefix words, like ’A RGB photo’
   276	or ’An infrared photo’, are added to the prompts. For model fine-tuning, we freeze the original Stable Diffusion Model and use LoRA to fine-tune it. For text adapter training,
   277	we design a text adapter to refine the output of the text encoder by leveraging the capability of large language models. Right Top: We design a two-stage strategy to generate
   278	multi-perspective RGB person images belonging to the same ID based on text descriptions. Right Bottom: we use SoftEdge to detect edge maps of generated RGB images, and then
   279	combine infrared image embeddings and edge maps to transfer RGB images to infrared images.
   280	
   281	model LLaMA [36], and save the mean value of each token embeddings
   282	generated by the LLM as the knowledge representation.
   283	To transfer the semantic understanding ability of the LLM to our
   284	fine-tuned diffusion model, we design a text adapter. We first freeze
   285	the parameters of the large language model 𝑓𝐿𝐿𝑀 , the text conditional
   286	encoder 𝑓𝐸𝑡 and the image generator 𝑓𝑔𝑒𝑛 of the fine-tuned diffusion
   287	model. Then we only optimize our text adapter. It consists of a fully
   288	connected neural networks (FCN) and a Transformer subnetwork, denoted as ℎ𝑖 . For the text description 𝑝𝑖 , we send it to both text encoders
   289	(LLM’s and diffusion model) and use the cross-attention to associate
   290	them. Formally, we define 𝑄𝑖 = ℎ1 [𝑓𝐸𝑡 (𝑝𝑖 )] and 𝐾𝑖 = ℎ2 [𝑓𝐸𝑡 (𝑝𝑖 )], and
   291	calculate an attention value as:
   292	𝑄𝑖 𝐾 𝑇
   293	𝑎𝑡𝑡𝑖 = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥( √ 𝑖 )
   294	𝑑
   295	
   296	visible and infrared photo, respectively. Here 𝑃 means original text
   297	descriptions from CUHK-PEDES and ICFG-PEDES. Note that we remove
   298	color keywords in 𝑃 to avoid the influence of color information for
   299	infrared photo generating. For each prompt, we first generate a RGB
   300	photo as the reference image for subsequent controllable generation.
   301	Then we use IP-Adapter [39], an effective and lightweight adapter
   302	to achieve image prompt capability for the pre-trained text-to-image
   303	diffusion models, wherein the reference image, serving as the image
   304	prompt, is fed into the adapter to generate multi-perspective RGB
   305	person images while preserving the identity. Since we have generated
   306	enough RGB photos for each identity, we utilize ControlNet [33] with
   307	SoftEdge to transfer some RGB photos to infrared photos. Specifically,
   308	we use real-world infrared photos together with edge detection map
   309	of generated RGB photos as the inputs of ControlNet, and thus we
   310	realize controllable visible and infrared person photo generating for
   311	each prompt. Finally, we generate 40 RGB photos and 40 infrared
   312	photos for each person identity, totaling about 20 000 generated photos
   313	with 257 person identities.
   314	In our endeavor to create a high-quality dataset, we have identified
   315	that the initial generation process may yield images that are not up
   316	to the desired standard. These images may be blurry, contain multiple
   317	persons, or exhibit other forms of noise, which can be attributed to
   318	the inherent limitations of the generation abilities. To address these
   319	issues, we have implemented a series of post-processing operations to
   320	enhance the quality of our dataset. Specifically, we first remove some
   321	low-quality images, like grayscale or blur images, in generated RGB
   322	images. For the generated infrared images, we have observed that some
   323	images contain color information, which is not typical for this type of
   324	imagery. To maintain the authenticity and consistency of the infrared
   325	images, we convert these images into grayscale, which ensures that all
   326	infrared images in our dataset adhere to the standard characteristics of
   327	infrared imagery, thereby enhancing the reliability of our dataset. To
   328	further refine our dataset, we employ OpenPose [40–43], a widely used
   329	tool for human keypoint detection. With OpenPose, we can identify
   330	images that contain fewer than 18 detected key points, which typically
   331	indicate low-quality or incomplete human figures. These images are
   332	subsequently removed from our dataset. Additionally, OpenPose allows
   333	us to resize the images with tight bounding boxes, ensuring that each
   334	image focuses on the human figure without unnecessary background
   335	noise. At this point, we are capable of generating visible and infrared
   336	person images, as shown in Fig. 4, resulting in the creation of the Gen-
   337	
   338	(2)
   339	
   340	where 𝑑 is the feature dimension of 𝑄𝑖 and 𝐾𝑖 . To guarantee that
   341	the meaning conveyed by the prompt remains unaffected, we directly
   342	set 𝑉𝑖 = 𝑓𝐸𝑡 (𝑝𝑖 ) without any transformation. Moreover, we design a
   343	distillation loss 𝐿𝐿𝑀 to incorporate the semantic understanding of the
   344	LLM into 𝑎𝑡𝑡𝑖 as following:
   345	𝐿𝐿𝑀 = 𝐊𝐋[𝐖𝟎 𝑓𝐿𝐿𝑀 (𝑝𝑖 )∕𝜏, 𝑄𝑖 ∕𝜏]
   346	
   347	(3)
   348	
   349	where 𝜏 is the temperature, 𝐊𝐋 is the KL divergence, and 𝐖𝟎 is
   350	a randomly initialized matrix using Kaiming initialization, ensuring
   351	maximal preservation of the LLM’s semantic content while adjusting the
   352	dimensional alignment between 𝑓𝐿𝐿𝑀 and 𝑄. Therefore, we construct
   353	𝑉𝑖′ = 𝑉𝑖 ⊗ 𝑎𝑡𝑡𝑖 and get calibrated semantic information. Finally, through
   354	a fully connected layer, the output generated by our text adapter is
   355	transformed to 𝑐𝐿𝐿𝑀 , a representation that encapsulates the semantic
   356	interpretation capabilities inherent to the LLM, and we combine the
   357	output of the text encoder 𝑓𝐸𝑡 (𝑝𝑖 ) and our text adapter 𝑐𝐿𝐿𝑀 as the
   358	input to the generator as follows:
   359	′
   360	𝑐𝐿𝐿𝑀
   361	= 𝜂 ⋅ 𝑐𝐿𝐿𝑀 + (1 − 𝜂) ⋅ 𝑓𝐸𝑡 (𝑝𝑖 )
   362	
   363	(4)
   364	
   365	where 𝜂 is a constant hyper-parameter.
   366	3.4. Controllable image generation and inference
   367	To make the generated images more close to the real-world photos,
   368	we utilize the text description of the CUHK-PEDES dataset [37] and the
   369	ICFG-PEDES [38] dataset as the text condition for the diffusion model.
   370	We design text prompts as ‘‘a RGB photo/an infrared photo, [𝑃 ]’’ for
   371	4
   372	
   373	Pattern Recognition 165 (2025) 111561
   374	
   375	X. Yu et al.
   376	
   377	Fig. 3. Utilization of large language models during the inference stage.
   378	
   379	3.5. Model pre-training
   380	To verify the validity of GCP dataset, we combine GCP with the existing training dataset and use it to pretrain CLIP-Enhanced Contrastive
   381	Learning Network (CCLNet) [21] for VI-ReID task. With the help of the
   382	visual-language representation ability from CLIP, CCLNet establishes a
   383	learnable cluster-aware prompt for person images and obtains textual
   384	descriptions allowing for subsequent unsupervised training. In addition,
   385	CCLNet leverages Hungarian matching method to fuse cross-modal information, enabling the network to better learn modality-independent
   386	features.
   387	In addition, we also conduct evaluation on OTPA [20], a mutual information guided optimal transport approach for unsupervised
   388	VI-ReID. Under three learning principles, 𝑖.𝑒., ‘‘Sharpness’’ (entropy
   389	minimization), ‘‘Fairness’’ (uniform label distribution), and ‘‘Fitness’’
   390	(reliable cross-modality matching), OTPA designs a loop iterative training strategy alternating between model training and cross-modality
   391	matching.
   392	Another training baseline we adopt is RPNR [44], a robust pseudolabel learning framework designed to improve the reliability of pseudolabels and establish more reliable cross-modality correspondences for
   393	the unsupervised Visible–Infrared Person Re-identification (VI-ReID)
   394	task.
   395	Furthermore, we conduct experiments on the superior baseline
   396	SDCL [45], a collaborative learning framework based on the transformer, which effectively combines shallow and deep features for
   397	improved performance. By integrating these features, SDCL enhances
   398	the robustness and accuracy of cross-modality learning.
   399	
   400	Fig. 4. Text-to-image generation results by our generation framework.
   401	
   402	Table 1
   403	Statistics about our generated dataset.
   404	Modal
   405	
   406	# of ID
   407	
   408	Mean
   409	
   410	25%
   411	
   412	50%
   413	
   414	75%
   415	
   416	RGB
   417	IR
   418	
   419	257
   420	257
   421	
   422	20.52
   423	19.46
   424	
   425	12
   426	12
   427	
   428	21
   429	19
   430	
   431	29
   432	26
   433	
   434	erated Cross-modal Person Dataset (GCP). Detailed statistics about GCP
   435	are presented in Table 1. Each identity in the dataset has approximately
   436	20 visible images and 20 infrared images, which ensures a high degree
   437	of ID consistency and diversity for subsequent ReID model training.
   438	Furthermore, we also explore the use of large language models to
   439	generate person textual descriptions, as shown in Fig. 3, aiming to
   440	enhance the automation of generating both visible and infrared images
   441	within the entire framework. This approach eliminates the reliance on
   442	existing textual descriptions from the CUHK-PEDES dataset and the
   443	ICFG-PEDES dataset, further improving the scalability and applicability
   444	of our method.
   445	
   446	4. Experiments
   447	In this section, we conduct comprehensive experiments to evaluate our approach. First, we introduce our experimental settings and
   448	implementation details. The main results and the comparison with
   449	state-of-the-art methods are then shown in 4.2. Finally, we present the
   450	ablation studies and visualization results of our approach. To promote
   451	the culture of reproducible research, source codes and experimental
   452	results accompanying this paper will be released later.
   453	5
   454	
   455	Pattern Recognition 165 (2025) 111561
   456	
   457	X. Yu et al.
   458	
   459	model using LoRA for 5000 steps; in the second stage, we train an
   460	additional text adapter to enhance the model’s semantic understanding
   461	capability for 10,000 steps. For LoRA fine-tuning, we select visible
   462	images from cam1, infrared images from cam3 in SYSU-MM01 as
   463	the image training dataset. Notably, only the UNet of the model is
   464	finetuned while the others are frozen. For text adapter training, we
   465	utilize LLaMA2 with the size of 13B as the large language model for
   466	knowledge distillation, and set 𝜂 in Eq. (4) to 0.1. During the stage
   467	of inference and dataset generating, we use OpenCLIP-ViT-H-14 as the
   468	image encoder, and choose pretrained IP-Adapter for Stable Diffusion
   469	v1.5. For ControlNet with SoftEdge, softedge_pidinet is employed to
   470	detect edge maps. For ReID pre-training, we initially integrate our
   471	proposed dataset, 𝑖.𝑒., GCP, with the original training sets of SYSUMM01 and RegDB to construct a new training set, while the test sets
   472	remain unaltered to validate the effectiveness of our method. All our
   473	methods are implemented with PyTorch and trained on GPU NVIDIA
   474	RTX 4090.
   475	ReID Baseline. After integrating GCP, our generated dataset, with
   476	the training sets of SYSU-MM01 or RegDB, we proceed to conduct
   477	training on four established baseline models: CCLNet, OTPA, RPNR, and
   478	SDCL. By utilizing both the synthetic data from GCP and the real-world
   479	data from SYSU-MM01 and RegDB, we aim to improve the models’
   480	ability to generalize across diverse scenarios and conditions.
   481	
   482	Fig. 5. Comparisons between real-world photos and generated images. For each image
   483	pair, the real-world person images captured by camera is on the left, while the image
   484	on the right is generated by our generation framework.
   485	
   486	4.2. Comparison with state-of-the-art methods
   487	
   488	4.1. Experimental settings
   489	
   490	We conduct the experiments on SYSU-MM01 and RegDB under the
   491	settings described in 4.1, and compare our method with the state-ofthe-art methods, including fully-supervised VI-ReID and unsupervised
   492	VI-ReID. We take the test results of CCLNet as the baseline. The main
   493	result are presented in Tables 2 and 3. We report some fully-supervised
   494	methods (Zero-Padding, AlignGAN, DDAG, CM-MAS, DART, CMT) and
   495	some unsupervised methods (ClusterContrast, H2H, OTLA, ADCA).
   496	Comparison on SYSU-MM01. As reported in Table 2, our proposed
   497	method almost achieves the best results in terms of both all-search
   498	mode and indoor-search mode. In particular, we achieve 64.95% Rank1 and 63.59% mAP for all-search, and 71.57% Rank-1 and 77.44% mAP
   499	for indoor-search, respectively. It is worth noting that our approach
   500	has demonstrated effectiveness on all four baselines. We improve OTPA
   501	by 2.87% Rank-1 and 1.95% mAP in term of all-search, and enhance
   502	RPNR by 2.71% Rank-1 and 1.71% mAP in term of indoor-search. Our
   503	unsupervised results are closely approaching the supervised ones.
   504	Despite the effectiveness of previous methods, it seems that the
   505	limited scale of VI-ReID dataset restrict the performance of existing
   506	models. It verifies that the generated images can be effectively added to
   507	the SYSU-MM01 dataset, demonstrating the effectiveness and usability
   508	of our generative framework in pre-training of VI-ReID models. This
   509	result also proves that expanding the scale of the SYSU-MM01 dataset
   510	helps the VI-ReID model better benefit from cross modal pre-training,
   511	without requiring any modifications.
   512	Comparison on RegDB. Table 3 presents the results of our methods and other SOTA methods tested on RegDB. Consistent with the
   513	experimental results observed on SYSU-MM01, our approach performs
   514	excellent results on RegDB. To be specific, our method demonstrates superiority over both baselines, with 88.88% Rank-1 and 82.41% mAP for
   515	Vis-to-IR, and 87.52% Rank-1 and 81.18% mAP for IR-to-Vis. Note that
   516	the infrared images in the RegDB dataset are thermographic infrared
   517	images, which differ significantly from the infrared images generated
   518	based on the SYSU-MM01 dataset. However, our fine-tuning process is
   519	only conducted on SYSU-MM01 because the data quality of RegDB is
   520	insufficient. Therefore, there is a slight decline in the mAP on CCLNet,

codex
HPRNet 的关键断点已经补齐：它不是简单“重建人体”，而是把现有重建方法的失败归因到缺少非邻域上下文和真实解析约束。还要补一小段它的服装对抗和解析重建损失，避免把方法链条说漏。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	arXiv:2501.05851v1 [cs.CV] 10 Jan 2025
     2	
     3	Identity-aware Feature Decoupling Learning for
     4	Clothing-change Person Re-identification
     5	1st Haoxuan Xu
     6	
     7	2nd Bo Li
     8	
     9	3rd Guanglin Niu∗
    10	
    11	School of Artificial Intelligence
    12	Beihang University
    13	Beijing, China
    14	xhaoxuan@buaa.edu.cn
    15	
    16	School of Artificial Intelligence
    17	Beihang University
    18	Beijing, China
    19	boli@buaa.edu.cn
    20	
    21	School of Artificial Intelligence
    22	Beihang University
    23	Beijing, China
    24	beihangngl@buaa.edu.cn
    25	
    26	Abstract—Clothing-change person re-identification (CC Re-ID)
    27	has attracted increasing attention in recent years due to its
    28	application prospect. Most existing works struggle to adequately
    29	extract the ID-related information from the original RGB images.
    30	In this paper, we propose an Identity-aware Feature Decoupling
    31	(IFD) learning framework to mine identity-related features.
    32	Particularly, IFD exploits a dual stream architecture that consists
    33	of a main stream and an attention stream. The attention stream
    34	takes the clothing-masked images as inputs and derives the
    35	identity attention weights for effectively transferring the spatial
    36	knowledge to the main stream and highlighting the regions with
    37	abundant identity-related information. To eliminate the semantic
    38	gap between the inputs of two streams, we propose a clothing
    39	bias diminishing module specific to the main stream to regularize
    40	the features of clothing-relevant regions. Extensive experimental
    41	results demonstrate that our framework outperforms other
    42	baseline models on several widely-used CC Re-ID datasets.
    43	Index Terms—Clothing-change Person Re-Identification, IDbased Knowledge Transfer, Clothing Bias Diminishing Module
    44	
    45	I. I NTRODUCTION
    46	Person re-identification (Re-ID) aims at matching the same
    47	pedestrian across different cameras [1]. Most existing methods
    48	predominantly utilize global representations for matching,
    49	which are only applicable to pedestrians without clothing
    50	change [2]–[11]. Whereas, a person changing clothing is a
    51	widespread phenomenon in practice. Consequently, the more
    52	challenging task clothing-change person re-identification (CC
    53	Re-ID) has received significant attention recently, which attempts to associate the same pedestrian with changed clothes.
    54	Driven by different motivations, existing methods for CC
    55	Re-ID can be broadly classified into two categories: 1) data
    56	augmentation methods and 2) biometrics-based methods. Data
    57	augmentation methods argue that the scale of current CC ReID datasets is insufficient to fully capture identity-related (IDrelated) features and attempt to augment training data [12]–
    58	[14]. However, these methods are subject to the quality of
    59	the generated virtual data, and their effectiveness is typically
    60	difficult to interpret. On the other hand, biometrics-based
    61	methods aim to explicitly capture stable biometrics features.
    62	* corresponding author. This work was supported by the National Natural
    63	Science Foundation of China (No. 62376016).
    64	
    65	All ID-related (e.g. face, hairstyle)
    66	
    67	Decouple
    68	
    69	ID-unrelated
    70	
    71	All ID-related (e.g. face, hairstyle)
    72	
    73	ID-unrelated
    74	
    75	ID 1 Cloth A
    76	
    77	ID 1 Cloth B
    78	
    79	Head region
    80	Body region
    81	
    82	Decouple
    83	
    84	matched
    85	ID-related
    86	
    87	ID-related
    88	
    89	unmatched
    90	
    91	Fig. 1. Illustration of the ID-related features distribution. The head regions
    92	contain purely ID-related features, while ID-related features and ID-unrelated
    93	features are coupled in the body regions.
    94	
    95	These methods can be further subdivided into multi-modality
    96	and single-modality methods. The multi-modality methods
    97	exploit extra modalities as auxiliary information to highlight
    98	ID-related features. Wang et al. enhance ID-related features
    99	by extracting face features [15]. Methods such as SPT+ASE,
   100	CESD, FASM, and GI-ReID learn ID-related features utilizing
   101	sketches, keypoints, human contours, and gait information,
   102	respectively [16]–[19]. Additionally, some studies argue that
   103	the 3D shape contains rich ID-related information and attempt
   104	to leverage 3D information [20], [21]. In contrast, single
   105	modality methods such as CAL, RCSANet and AIM attempt to
   106	directly mine ID-related features by clothing adversarial loss,
   107	clothing status awareness and causality, respectively [22]–[24].
   108	In the real world, humans can recognize their acquaintances
   109	through various identity clues (e.g., face, hairstyle, body shape,
   110	height, gait) even if these individuals wear unfamiliar clothing.
   111	However, certain clues, such as height and gait, cannot be
   112	reliably estimated from a single image, and the remaining
   113	clues are distributed across both clothing-irrelevant regions
   114	(i.e. head regions) and clothing-relevant regions (i.e. body
   115	regions). As illustrated in Fig. 1, the head regions primarily
   116	contain ID-related features, such as the face, hairstyle and head
   117	contour. In contrast, the body regions exhibit a mixture of
   118	
   119	Clothing Bias Diminishing
   120	Resize
   121	
   122	Main Stream
   123	
   124	Clothing Weight 𝑾𝑾𝒄𝒄
   125	
   126	Clothing Feature Maps 𝑭𝑭𝒄𝒄 Clothing Features 𝒇𝒇𝒄𝒄
   127	Inference
   128	
   129	BN + GAP
   130	
   131	Global Feature Maps 𝑭𝑭𝒈𝒈
   132	
   133	Refined Feature Maps 𝑭𝑭𝒓𝒓𝒓𝒓 Refined Features 𝒇𝒇𝒓𝒓𝒓𝒓
   134	
   135	Parsing
   136	
   137	Max Pool
   138	
   139	IKT Module
   140	
   141	…
   142	…
   143	
   144	…
   145	
   146	Attention Stream
   147	
   148	Concat
   149	
   150	Attention Feature Maps 𝑭𝑭𝒂𝒂
   151	
   152	Necessary during both training and inference
   153	
   154	Attention Features 𝒇𝒇𝒂𝒂
   155	
   156	Predicted Identities
   157	
   158	Only Necessary during training
   159	
   160	ℒ𝐶𝐶𝐶𝐶𝐶𝐶
   161	
   162	𝑚𝑚
   163	ℒ𝐼𝐼𝐼𝐼
   164	
   165	conv
   166	
   167	ID-based Knowledge Transfer
   168	
   169	BN + GAP
   170	Backbone 𝑔𝑔𝑎𝑎
   171	
   172	Predicted Identities
   173	
   174	Average Pool
   175	𝑭𝑭𝒂𝒂
   176	
   177	…
   178	…
   179	
   180	…
   181	
   182	Backbone 𝑔𝑔𝑚𝑚
   183	
   184	𝝍𝝍𝒎𝒎
   185	
   186	BN + GAP
   187	
   188	ID-based Weight 𝑾𝑾𝑰𝑰
   189	
   190	𝑎𝑎
   191	ℒ𝐼𝐼𝐼𝐼
   192	
   193	Element-wise matrix product
   194	
   195	Fig. 2. Overview of our proposed IFD, which consists of an attention stream and a main stream. The attention stream learns a weight matrix with high values
   196	for identity-relevant regions and low values for identity-irrelevant regions at the feature level. The main stream aims to learn ID-related features under the
   197	guidance of the attention stream and clothing bias diminishing module.
   198	
   199	identity-related and identity-unrelated features, which can lead
   200	to erroneous match results. Thus, the key challenges of CC
   201	Re-ID can be concluded: 1) leveraging only specific part of
   202	ID-related features can not comprehensively represent an
   203	individual, 2) clothing-relevant regions actually couple the
   204	implicit ID-related features, which is hard to be extracted
   205	for CC Re-ID. Although existing approaches mitigate the
   206	effect of clothing changes to some extent, they still lack
   207	effective constraints to ensure that the model consistently
   208	focuses on all critical ID-related features. Specifically, each of
   209	the aforementioned multi-modality methods tends to capture
   210	only one category of ID-related features, leaving other critical
   211	discriminative ID-related features underutilized. As for singlemodality methods, they can only extract coarse ID-related
   212	features by overlooking the semantics of body parts.
   213	To enhance the ID-related features for CC Re-ID, we
   214	propose a novel Identity-aware Feature Decoupling (IFD)
   215	learning framework that consists of an attention stream and
   216	a main stream. Both streams employ the same backbone for
   217	feature extraction but operate independently without shared
   218	weights. The attention stream processes clothing-masked images, while the main stream takes the original images as input.
   219	To ensure that the main stream focuses comprehensively on
   220	the regions implying identity information, we incorporate an
   221	ID-based Knowledge Transfer (IKT) module between the two
   222	streams. Additionally, to decouple the ID-related features from
   223	the clothing-relevant region, we introduce a Clothing Bias
   224	
   225	Diminishing (CBD) module, which helps model the consistent
   226	clothing features with regard to the same individual.
   227	In summary, our contributions are listed as follows:
   228	• We are the first to propose a dual-stream identity-attention
   229	model that effectively compels the network to focus
   230	comprehensively on the regions containing distinctive
   231	identity information.
   232	• An effective CBD module is developed to maintain the
   233	consistency of clothing features for the same individual.
   234	• Extensive experiments demonstrate that our method
   235	achieves state-of-the-art on several clothing-change ReID datasets including PRCC and LTCC [18], [19].
   236	II. M ETHODOLOGY
   237	IFD aims to comprehensively mine the ID-related features
   238	to address the CC Re-ID issue. As illustrated in Fig. 2, the
   239	framework begins by extracting a clothing-masked image M
   240	from the input image I using an off-the-shelf human parsing
   241	network [25]. The clothing-masked image contains the critical
   242	head and contour information, which is essential for learning
   243	ID-related features, but it discards all the color information
   244	of body parts, which is critical to re-identify persons in the
   245	conventional Re-ID scenarios. To robustly extract ID-related
   246	features, we implement a dual-stream framework with an IDbased knowledge transfer module to guide the main stream
   247	toward comprehensively emphasizing ID-related regions.
   248	The IKT module can help locate the ID-related regions.
   249	However, the semantic gap between the inputs of the two
   250	
   251	streams may introduce error. The body regions of inputs
   252	merely contain shape information in the attention stream,
   253	while clothing-relevant features can still couple with IDrelated features in body parts of main stream. Thus, the
   254	IKT module might inadvertently amplify the influence of IDunrelated features while enhancing ID-related features, which
   255	can limit the overall performance of our model. To guarantee
   256	that the final features are exclusive of clothing color and
   257	texture, we introduce a CBD module.
   258	
   259	To restrain the contribution of clothing at the feature level,
   260	we propose Clothing Contrastive Loss LCCL . Let i be the
   261	index of an arbitrary sample in a batch, the LCCL is defined:
   262	i
   263	p
   264	N
   265	e(fc ·fc /τ )
   266	1 X 1 X
   267	wp log
   268	LCCL = −
   269	P (f i ·fcj /τ )
   270	p
   271	i
   272	N i=1 |Pi |
   273	e(fc ·fc /τ ) +
   274	e c
   275	
   276	p∈Pi
   277	
   278	j∈Ni
   279	
   280	(
   281	wp =
   282	
   283	A. ID-based Knowledge Transfer
   284	Motivated by the spatial attention mechanisms that are
   285	widely used in computer vision to capture fine-grained local
   286	features and precisely locate task-relevant regions, we attempt
   287	to enhance the attention toward ID-related regions for the CC
   288	Re-ID task [26], [27]. However, it is naive to directly utilize
   289	spatial attention due to the difficulty of learning effective
   290	spatial attention weights without auxiliary supervision. To
   291	address this, we propose an ID-based Knowledge Transfer
   292	module that facilitates the learning of robust and effective
   293	spatial attention weights.
   294	As shown in Fig. 2, we design a mutual learning framework
   295	with two stream, gm (.) and ga (.) denotes the backbone of
   296	main stream and attention stream, respectively. The original
   297	image I is passed through gm (.) to extract the global feature
   298	maps Fg . Simultaneously, the masked image M is fed into
   299	ga (.) to obtain the attention feature maps Fa . Taking Fa as
   300	input, our IKT module derives ID-based attention matrix WI :
   301	WI = σ (Wconv ∗ [mp (Fa ) ; ap (Fa )])
   302	
   303	(1)
   304	
   305	where mp denotes max pooling along the channel, ap denotes
   306	average pooling along the channel, ∗ denotes convolution
   307	operation, Wconv indicates the weights of convolution filters,
   308	and σ (.) denotes the sigmoid function. The ID-based attention
   309	matrix is then applied to the global feature Fg as formulated:
   310	Frg = WI ⊗ Fg
   311	
   312	(2)
   313	
   314	where ⊗ denotes the element-wise matrix product, Frg denotes
   315	refined feature maps.
   316	B. Clothing Bias Diminishing Module
   317	To decouple the ID-related features in clothing-relevant
   318	regions, we propose a Clothing Bias Diminishing module. As
   319	illustrated in the top of Fig. 2, we estimate a fine-grained mask
   320	ψ m for clothing-relevant parts from the original image I. The
   321	pixel value of ψ m can be formulated as:
   322	(
   323	1 , if Ii,j ∈ C
   324	m
   325	ψ(i,j) =
   326	(3)
   327	0 , Otherwise
   328	where C denotes the set of clothing part categories. Then we
   329	resize the ψ m to match the dimensions of WI , resulting in
   330	Wc , which is used to perform an element-wise matrix product
   331	with Frg to derive the clothing-related feature maps Fc .
   332	By applying batch normalization and global average pooling
   333	operation to Fc , we obtain the clothing features fc .
   334	
   335	(4)
   336	1
   337	T
   338	
   339	1
   340	
   341	, if cp ̸= ci
   342	, Otherwise
   343	
   344	(5)
   345	
   346	where N is the batch size, Pi (Ni ) denotes the set of samples
   347	with the same ID as (different ID from) i, and τ is a
   348	temperature parameter. T ∈ R+ is a variable parameter, and cx
   349	denotes the clothing label. Equation (5) serves as an incentive
   350	function, encouraging the network to focus more on pairs with
   351	the same ID label but different clothing labels during training.
   352	Existing Re-ID methods typically employ PK sampling
   353	strategy during training, where K samples per ID and P × K
   354	samples per batch, but PK sampling overlooks the appearance
   355	diversity. However, in this paper, we encourage the network to
   356	learn ID-related features through our proposed loss function
   357	LCCL , and its effectiveness hinges on diversity in both persons
   358	and appearances within a batch. RAS sampling strategy can
   359	maintain appearance diversity by sampling A appearances of
   360	each person and fixed K images of each appearance in a batch

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '520,700p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   520	volume of the category data for clothing. τ is a temperature
   521	parameter used to control the model training.
   522	Subsequently, we disrupt the clothing features of different
   523	identities and randomly select those belonging to other identities to generate adversarial samples and improve the robustness
   524	of the model.
   525	f˜i = fi − ficlo + fĩclo ,
   526	(10)
   527	where i represents the current sample, ĩ represents a noncurrent sample, and fi represents the current sample feature.
   528	ficlo denotes the clothing-related feature of the current sample,
   529	and fĩclo is the non-current sample’s clothing-related feature by
   530	disrupting the above clothing-related (unrelated) features and
   531	reclassifying them Many adversarial samples can be simulated
   532	to help the model learn and adapt to different input changes,
   533	thereby improving the model’s generalization ability. Similar
   534	to Eq. (9) as follows:
   535	!
   536	N
   537	˜
   538	X
   539	e( fi ·ϕyi c/τ)
   540	adv
   541	.
   542	(11)
   543	Lclo = −
   544	log PLC ˜
   545	( fi ·ϕ j /τ)
   546	j=1 e
   547	i=1
   548	D. Parsing Reconstructed Exploration Module
   549	To acquire clothing-independent features better, we also
   550	propose a pedestrian-based appearance parsing reconstructed
   551	exploration model, which is accomplished using a multiregion
   552	pedestrian local reconstruction and clothing parsing model.
   553	Specifically, the multiregion pedestrian local reconstruction
   554	model was inspired by [1] and [11] but differs from them.
   555	Based on the basic structure of the human body, we split
   556	the output of the reconstructed encoder using ResNet50 as
   557	the backbone into three local regions (head, body, and lower
   558	limbs), which are represented as multilocal components in
   559	Fig. 2. The multilocal maps are then fed into the generation model [1] and projected onto a deconvolution model
   560	[46] to obtain the corresponding local region reconstructed
   561	maps. Finally, these maps are concatenated to form the final
   562	reconstructed feature, fr . In addition, we computed the reconstruction loss Lor with the original pedestrian feature fo to
   563	ensure reconstruction accuracy, which is expressed as
   564	v
   565	u N
   566	uX
   567	(12)
   568	L = t ( f − f )2 ,
   569	ri
   570	
   571	or
   572	
   573	oi
   574	
   575	i=1
   576	
   577	where fri and foi are the ith reconstructed and original pedestrian features, respectively. N is the sample size.
   578	
   579	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   580	
   581	152
   582	
   583	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   584	
   585	To extract pedestrian clothing color-independent features
   586	more effectively, we adopted a cloth parsing module to obtain
   587	pedestrian contour features independent of clothing color.
   588	Specifically, we fed the output of the Parsing Encoder with
   589	the ResNet50 as the backbone into the Clothing Parsing Model
   590	[47], which involved channel, position, and color parsing networks. It completes the pedestrian contour profile parsing from
   591	three aspects: pedestrian feature channels, human contextual
   592	position, and clothing color. The parsed pedestrian sample is
   593	represented by f p in Fig. 2. As before, to ensure the accuracy
   594	of the pedestrian sample parsing, we also used it to calculate
   595	the loss with the original feature map fo , and the parsing loss
   596	Lop is represented by Eq. (13).
   597	v
   598	u N
   599	uX
   600	(13)
   601	Lop = t ( f p j − fo j )2 ,
   602	j=1
   603	
   604	where f p j and fo j are the jth parsing and original pedestrian
   605	features, respectively. N is the sample size as mentioned above.
   606	E. Overall Loss Function of the Framework
   607	Once we complete the non-local multi-scale perception and
   608	parsing reconstructed exploration model learning, the total loss
   609	function L is represented as
   610	L = Lid + αLcla + Lclo + βLadv
   611	clo + Lor + Lop .
   612	
   613	(14)
   614	
   615	where Lid and Lclo are the pedestrian identity and clothing category loss, respectively. Lcla is the multi-classification loss for
   616	nonlocal perception loss. Lor and Lop are the reconstructed and
   617	parsing losses, respectively. α and β are the hyperparameters
   618	that balance the relative importance of every loss term in the
   619	overall training of the framework.
   620	F. Differences to Existing CC-ReID Methods
   621	In this section, we discuss the differences between existing
   622	methods and our method. Part of the inspiration for the
   623	proposed framework comes from existing DCR-ReID [1],
   624	AGW-ReID [43], and CAL [11]. However, the proposed
   625	HPRNet method differs substantially from these state-of-theart techniques, as described below.
   626	• Different design principles: The DCR-ReID method
   627	relies on component reconstruction and deep learningbased fusion for CC-ReID. The CAL method leverages clothes-based adversarial loss to extract clothingirrelevant features directly from original RGB images,
   628	while the AGW-ReID method incorporates a non-local
   629	operation within the ResNet backbone to retrieve the
   630	weights of such features. In contrast, HPRNet employs
   631	a combined strategy of non-local weighted multi-scale
   632	perception and human appearance parsing reconstruction
   633	to model contextual similarity (correlation) for nonneighboring pixels and clothing-independent features in
   634	CC-ReID. Unlike AGW-ReID, HPRNet treats the nonlocal attention operation as an independent module,
   635	integrating it with a multi-scale perception module to
   636	enhance model flexibility. Additionally, the proposed
   637	framework is optimized through a loss function that
   638	
   639	addresses both global modeling and local fine-grained
   640	feature learning, improving the robustness of fine-grained
   641	feature representation in complex scenes and making it
   642	particularly suitable for ReID tasks involving clothing
   643	changes.
   644	• Different framework structures: Although all the above
   645	methods utilize multi-branch strategies for learning and
   646	training clothing-irrelevant features, their approaches differ significantly. On one hand, DCR-ReID employs
   647	component reconstruction, clothing identification, and
   648	person identity branches. On the other hand, CAL
   649	incorporates clothing category loss and person identity branches for CC-ReID. Lastly, the AGW-ReID
   650	method integrates a non-local attention model within
   651	the backbone network to enhance pedestrian feature
   652	learning. Our proposed framework, however, adopts a
   653	structurally distinct architecture. It leverages a non-local
   654	weighted perception mechanism and a multi-scale progressive learning strategy to effectively capture the global
   655	receptive field and progressively learn local-to-whole
   656	pedestrian appearance features. Furthermore, HPRNet
   657	integrates a parsing reconstruction auxiliary model, forming a framework that, while inspired by state-of-the-art
   658	methods, introduces novel elements to effectively address
   659	the CC-ReID task.
   660	Algorithm 1 Training Procedure for the HPRNet Framework
   661	N
   662	Require: ResNet50 model G; Training dataset D = {xi , ji }i=1
   663	;
   664	p
   665	Number of images N;Number of identities N ; Maximum
   666	epoch F; Threshold t0 (0 < t0 < F)
   667	Initialization: Model weights θG ; Hyper-parameters α, β
   668	1: for t ← 1 to F do
   669	2:
   670	for i ← 1 to N do
   671	3:
   672	Extract image features fi0 = G(xi )
   673	4:
   674	Calculate Weight result fi = I( fi0 ) via Eq. (3)
   675	5:
   676	B, C, H, W ← fi
   677	6:
   678	Stripe dim S ← C  (max pooling kernel size)
   679	7:
   680	Segment fi with S size
   681	8:
   682	for z ← 1 to S do
   683	z
   684	9:
   685	2 × 2 max-pooling extract fk2
   686	via (4)
   687	z
   688	10:
   689	4 × 4 max-pooling extract fk4 via Eq. (4)
   690	11:
   691	Concat in the channel dimension
   692	12:
   693	end for
   694	13:
   695	Concat fk2 , fk4 with origin feature fi via Eq. (5)
   696	14:
   697	Compute Lcla via Eq. (6)
   698	15:
   699	Reconstruction Yiv via Eq. (12), (13)
   700	16:

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '300,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 109ms:
   300	modalities and ranges of high-order enhanced features at node-level and
   301	region-level simultaneously to generate the reliable middle-feature agents, and
   302	leverages the bi-directional feature enhancement to hierarchically reduce the
   303	modality discrepancy.
   304	
   305	and full utilization between different modality features.
   306	To address the above issues, we propose a novel highorder hierarchical middle-feature learning network (HOHNet), which is shown in Fig. 1. The HOH-Net is made up of
   307	a high-order structure learning (HSL) module, a fine-coarse
   308	graph attention alignment (FCGA) module, a hierarchical
   309	middle-feature agent learning (HMAL) loss, and a modalityrange identity-center contrastive (MRIC) loss for VI-ReID.
   310	The key innovation of our method lies in the novel formulation
   311	of exploiting high-order structure information and hierarchical
   312	middle-feature learning to learn a discriminative and reliable
   313	common feature space, thereby significantly mitigating the
   314	modality gap.
   315	Specifically, given a VIS-IR image pair, the HSL module
   316	captures the high-order relationships between the short-range
   317	and long-range features that are extracted from the short- and
   318	long-range feature extraction (SLE) module using a whitened
   319	hypergraph. Instead of directly adding or concatenating features from different modalities and ranges, we design an
   320	FCGA module that aligns these features appropriately and
   321	effectively at node-level and region-level simultaneously to
   322	achieve reliable middle features. Besides, we propose a HMAL
   323	loss to address the modality gap hierarchically by utilizing
   324	middle-feature agents and executing bi-directional interactions
   325	between different stages to enhance feature representation.
   326	Finally, we reduce the distances among VIS, IR, and middle
   327	center features by an MRIC loss, thereby smoothing the
   328	learning process of the common feature space between modalities. On the SYSU-MM01, RegDB, LLCM, and HITSZVCM datasets, our method achieves impressive 76.2%, 95.1%,
   329	65.7%, and 74.8% in Rank-1, respectively.
   330	The main contributions of our work are as follows:
   331	•
   332	
   333	Page 2 of 26
   334	
   335	We propose an HSL module to learn high-order structure
   336	information of both short and long-range features. Such
   337	
   338	a novel way effectively models high-order relationships
   339	across different feature nodes of each pedestrian image
   340	and avoids the problem of model collapse.
   341	• We design a lightweight yet effective FCGA module
   342	that can refine the details of each high-order node-level
   343	feature and perceive the semantic association of regionlevel features simultaneously to achieve reliable middle
   344	features.
   345	• An HMAL loss is designed to hierarchically reduce
   346	the modality discrepancy at each stage network by the
   347	middle-feature agents and perform the bi-directional feature enhancement between different stages to enhance the
   348	detailed representation and the semantic relationship of
   349	features.
   350	• An MRIC loss is designed to minimize the distances
   351	between VIS, IR, and middle features in the embedding space. This is beneficial to extracting discriminative
   352	modality-shared pedestrian features.
   353	This paper significantly extends our previous conference
   354	work HOS-Net [12]. The limitations of our previous work
   355	include the following: First, the computational cost of generating the middle features through graph attention is high
   356	and did not make full use of the middle features. Second,
   357	the previous method extracted modality-shared features from
   358	the low stage to the high stage, ignoring the importance of
   359	bi-directional interaction between different stages that can
   360	enhance feature representation. The HOH-Net addresses these
   361	limitations in two main ways. (1) We further develop a finecoarse graph attention alignment (FCGA) module to refine
   362	the high-order node-level features and perceive the contextual
   363	relationship between region-level features to achieve more
   364	reliable middle features with less model complexity. (2) We
   365	design an HMAL loss to mitigate modality discrepancy from
   366	a hierarchical view by introducing the agents of the middle
   367	features at each VIS and IR modality-shared feature extraction
   368	stage. The proposed HMAL loss also enables the bi-directional
   369	interaction of features between different stages, for obtaining
   370	richer semantic and more detailed feature information than
   371	the previous HOS-Net. In the experiments, we also provide
   372	more comprehensive experimental evaluations, including comparative experiments, ablation studies, parameter analyses,
   373	and visualization analyses. Compared to the previous HOSNet, the HOH-Net achieves lower computational cost and
   374	superior retrieval accuracy than our previous work (the number
   375	of parameters of the HOH-Net is reduced by 29.5%) and
   376	the Rank-1 of our method is improved by 0.6%, 0.4%, and
   377	0.8% on the three image-based VI-ReID datasets, i.e., SYSUMM01, RegDB, and LLCM, respectively. In addition, our
   378	method can also be easily extended to the video-based VIReID field, and compared to the existing video-based methods,
   379	our HOH-Net achieves the best 74.8% Rank-1 on the HITSZVCM dataset.
   380	II. R ELATED W ORK
   381	A. Visible-Infrared Person Re-Identification (VI-ReID)
   382	VI-ReID methods can be divided into image-level and
   383	feature-level methods to reduce the modality discrepancy. The
   384	
   385	Page 3 of 26
   386	
   387	IEEE Transactions on Circuits and Systems for Video Technology
   388	
   389	PREPRINT SUBMITTED TO TCSVT, JULY 2025
   390	
   391	1
   392	2
   393	3
   394	4
   395	5
   396	6
   397	7
   398	8
   399	9
   400	10
   401	11
   402	12
   403	13
   404	14
   405	15
   406	16
   407	17
   408	18
   409	19
   410	20
   411	21
   412	22
   413	23
   414	24
   415	25
   416	26
   417	27
   418	28
   419	29
   420	30
   421	31
   422	32
   423	33
   424	34
   425	35
   426	36
   427	37
   428	38
   429	39
   430	40
   431	41
   432	42
   433	43
   434	44
   435	45
   436	46
   437	47
   438	48
   439	49
   440	50
   441	51
   442	52
   443	53
   444	54
   445	55
   446	56
   447	57
   448	58
   449	59
   450	60
   451	
   452	image-level methods [4], [5], [13] often minimize the modality
   453	gap by generating middle-modality images or new modality
   454	images. Wang et al. [13] attempted to introduce a generative
   455	adversarial network to generate new modality images from
   456	VIS and IR modalities by jointly aligning the pixel-level and
   457	feature-level features. Liu et al. [5] proposed a two-stage
   458	modality enhancement network to perform the cross-modality
   459	style translation and optimized the structures of images for
   460	VI-ReID. Besides, Li et al. [14] leveraged the anaglyph data
   461	of the pedestrian as the middle modality images to reduce the
   462	modality gap. Du et al. [15] proposed a channel-blended transformation mechanism to confuse the VIS and IR information
   463	and reduce the influence of modality-specific features, thereby
   464	facilitating the learning of modality-shared features. However,
   465	image-level methods easily encounter issues such as color
   466	inconsistency or the loss of image details when generating
   467	images by the generative adversarial network (GAN), which
   468	is less reliable for training and subsequent visible-infrared
   469	retrieval.
   470	The feature-level methods seek to reduce the modality
   471	discrepancy by mapping the features of different modalities
   472	into a common feature space. A few methods [1], [8], [16]
   473	leverage the weight-shared CNN or ViT as the backbone
   474	to extract modality-shared features. Hybrid models of CNN
   475	and Transformer [10], [17]–[20] can effectively extract shortrange and long-range features. For example, Zhao et al. [10]
   476	enhanced the spatial-channel information of the pedestrian by
   477	adopting the CNN-Transformer hybrid network. Chen et al.
   478	[20] attempt to introduce the off-the-shelf key point extractors
   479	(e.g., OpenPose [21]) to generate key point labels of person
   480	images and achieve features based on the CNN-Transformer
   481	hybrid network, aiming to learn modality-irrelevant features.
   482	But the key point extractor may bring noisy labels, deteriorating the discriminability of final ReID features. However,
   483	the above feature-level methods neglect the high-order structure information of features (i.e., the complex and diverse
   484	relationships across features) that is important for VI-ReID.
   485	To solve the above problem, our work introduces the highorder structure learning to obtain the high-order relationships
   486	between the short- and long-range features and avoid the
   487	model collapse by a whitened hypergraph.
   488	To obtain a common feature space, a lot of feature-level
   489	methods [1], [8], [22]–[24] employ the contrastive-based loss
   490	that directly minimizes the distances between VIS and IR
   491	features. However, it is not a trivial task to learn a reliable
   492	common feature space due to the large modality discrepancy
   493	between modalities. Different from these methods that tend
   494	to minimize the distances between VIS and IR features directly, Zhang et al. [25] tried to generate diverse VIS or
   495	IR embeddings for learning informative feature representations to mitigate the modality gap. Jiang et al. [26] adopted
   496	the modality-level and instance-level alignments for learning robust modality compensation. Li et al. [27] introduced
   497	the cross-modality semantic alignment to explore the intermodality correlation for eliminating the modality discrepancy.
   498	However, they ignored the importance of fine-coarse alignment
   499	for generating reliable middle features from different modalities and ranges to narrow the difference between VIS and IR
   500	
   501	3
   502	
   503	images. Different from these methods, our method generates
   504	reliable hierarchical middle-feature agents via the fine-coarse
   505	graph attention alignment, greatly promoting our method to
   506	learn a discriminative and reliable common feature space.
   507	In addition, to improve the discriminative ability of the
   508	network, Yang et al. [28] designed a saliency response module that adopts the location attention mechanism to build
   509	contextual connections between person features. Tian et al.
   510	[29] adopted the variational self-distillation to fit the mutual
   511	information between the input feature and its representation,
   512	thus obtaining the multi-view information for VI-ReID. The
   513	above methods follow low-to-high feature extraction, which
   514	ignores the interaction between features at different stages. To
   515	this end, the proposed HOH-Net performs the bi-directional
   516	enhancement between different stages to enhance the detailed
   517	representation and the semantic relationship of features. Moreover, we reduce the distances among VIS, IR, and middle
   518	center features by a modality-range identity-center contrastive
   519	loss, thereby smoothing the learning process of the common
   520	feature space between ranges and modalities.
   521	B. Graph Neural Network
   522	Graph neural network (GNN) is a type of neural network
   523	to process graph-structured data. Zhang et al. [30] adopted the
   524	GNN to select correlated nodes for information aggregation,
   525	thereby establishing the robust connection between the target
   526	and the search regions. Zhang et al. [31] introduced the
   527	GNN to perform the progressive relationship-mining for textto-image ReID. Contrasting with the vanilla graph models
   528	that only allow connections between two nodes, Feng et al.
   529	[32] proposed the novel hypergraph neural network (HGNN)
   530	to represent high-order feature correlations by utilizing a
   531	hypergraph structure. Wadhwa et al. [33] adopted the HGNN
   532	to learn the complex relationship among the incomplete features for the image inpainting. Han et al. [34] utilized the
   533	power of the hypergraph to encode image information and
   534	update the hypergraph structure by the fuzzy c-means method
   535	that can reduce the computational burden. Nevertheless, the
   536	above methods that rely on the HGNN may easily suffer
   537	from the model collapse (i.e., complex and diverse highorder correlations collapse to a single correlation) since the
   538	small differences in the feature nodes of pedestrians and the
   539	hyperedge can connect an arbitrary number of nodes. Different
   540	from the above methods, this paper introduces the whitening
   541	operation to HGNN, which can play the role of “scattering” on
   542	the nodes of the hypergraph, thereby significantly alleviating
   543	model collapse.
   544	Besides, to establish the correspondence between feature
   545	nodes, several methods [35]–[37] attempt to introduce the
   546	graph attention network (GAT) to enhance the representation
   547	of features. For instance, Dong et al. [35] fused the characteristics of CNN and GAT to discover feature connections for hyperspectral image classification. However, the above methods
   548	consider the correspondence between feature nodes at nodelevel, and ignore the semantic connections between regionlevel features that can encapsulate the context of features. In
   549	this work, we develop a fine-coarse graph attention alignment
   550	
   551	IEEE Transactions on Circuits and Systems for Video Technology
   552	
   553	Page 4 of 26
   554	
   555	PREPRINT SUBMITTED TO TCSVT, JULY 2025
   556	
   557	�
   558	
   559	SLE
   560	
   561	HSL
   562	
   563	�
   564	�
   565	
   566	Stage
   567	
   568	�
   569	
   570	�
   571	
   572	FCGA
   573	
   574	�
   575	
   576	×
   577	
   578	�
   579	
   580	�
   581	
   582	FCGA
   583	
   584	�
   585	
   586	×
   587	
   588	×
   589	×
   590	
   591	� ,
   592	
   593	2
   594	
   595	1
   596	
   597	e
   598	
   599	�1,
   600	
   601	�2,
   602	
   603	� ,
   604	
   605	�
   606	
   607	�
   608	
   609	�
   610	
   611	FCGA
   612	
   613	�
   614	
   615	×
   616	
   617	�
   618	
   619	�
   620	
   621	�
   622	
   623	FCGA
   624	
   625	�
   626	
   627	×
   628	
   629	� ,
   630	
   631	×
   632	×
   633	�
   634	
   635	�
   636	
   637	�
   638	x
   639	
   640	Stage
   641	
   642	IR
   643	
   644	×
   645	×
   646	
   647	�
   648	
   649	Stage
   650	
   651	Stage
   652	
   653	Stage
   654	
   655	Stage
   656	�
   657	
   658	�
   659	�
   660	
   661	� ,
   662	
   663	�
   664	
   665	×
   666	×
   667	
   668	�2,
   669	
   670	�
   671	
   672	�
   673	
   674	g
   675	
   676	Stage
   677	
   678	Stage
   679	�1,
   680	
   681	�
   682	
   683	eg t
   684	
   685	VIS
   686	
   687	Stage
   688	
   689	�
   690	
   691	Stage
   692	
   693	g
   694	
   695	1
   696	2
   697	3
   698	4
   699	5
   700	6
   701	7
   702	8
   703	9
   704	10
   705	11
   706	12
   707	13
   708	14
   709	15
   710	16
   711	17
   712	18
   713	19
   714	20
   715	21
   716	22
   717	23
   718	24
   719	25
   720	26
   721	27
   722	28
   723	29
   724	30
   725	31
   726	32
   727	33
   728	34
   729	35
   730	36
   731	37
   732	38
   733	39
   734	40
   735	41
   736	42
   737	43
   738	44
   739	45
   740	46
   741	47
   742	48
   743	49
   744	50
   745	51
   746	52
   747	53
   748	54
   749	55
   750	56
   751	57
   752	58
   753	59
   754	60
   755	
   756	4
   757	
   758	�
   759	�
   760	

codex
IFD 的方法部分也补到关键细节：它把“衣服区域不能一概丢掉”作为核心观察，再用被遮衣图分支给原图分支提供身份注意力。还缺它的训练采样和训练损失结尾，我继续补完。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf' - | nl -ba | sed -n '360,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   360	each person and fixed K images of each appearance in a batch
   361	[28]. whereas, this sampling strategy may discard a significant
   362	number of samples for appearances with large proportion by
   363	ignoring the maldistribution of different appearances. Thus, we
   364	modify RAS by replacing the fixed number of each appearance
   365	with a proportion number to accelerate the effect of LCCL .
   366	C. Training
   367	We adopt ResNet-50 pre-trained on ImageNet as our backbone [29], [30]. During the training stage, we first train the
   368	attention stream with LaID to obtain effective feature maps
   369	with abundant ID-related information, and then we jointly train
   370	the dual streams under the guidance of the overall loss:
   371	a
   372	Lall = Lm
   373	ID + LID + λ · LCCL
   374	
   375	(6)
   376	
   377	where λ denotes the weight of the LCCL , which is set to 1.0.
   378	III. E XPERIMENTS
   379	A. Datasets and Evaluation Protocols
   380	We mainly evaluate our proposed method on two popular
   381	CC Re-ID benchmark datasets PRCC and LTCC. We employ
   382	two frequently-used metrics rank-1 and mAP to perform the
   383	evaluation results. Three kinds of test settings are defined
   384	as following: (i) general mode (both clothing-change and
   385	clothing-consistent ground truth samples are used to evaluate
   386	accuracy), (ii) same-clothing mode (only clothing-consistent
   387	ground truth samples are used to evaluate accuracy), (iii)
   388	clothing-change mode (only clothing-change ground truth
   389	samples are used to evaluate accuracy). In terms of PRCC,
   390	we report the evaluation results of same-clothing mode and
   391	clothing-change mode. As for LTCC, the accuracies for general mode and clothing-change mode are provided.
   392	
   393	TABLE I
   394	C OMPARISON R ESULTS WITH STATE - OF - THE - ART METHODS ON THE PRCC AND LTCC DATASET (%), WHERE ’ SKETCH ’, ’ SIL’, ’3D’, ’ POSE ’,
   395	’ PARSING ’,’ AUG ’ DENOTE THE CONTOUR SKETCHES , SILHOUETTES , 3D SHAPE , KEYPOINTS , HUMAN PARSING AND DATA AUGMENTATION ,
   396	RESPECTIVELY. T HE BEST RESULTS ARE INDICATED BY RED .
   397	Method
   398	PCB [4]
   399	IANet [2]
   400	SPT+ASE [19]
   401	CESD [18]
   402	RCSANet [23]
   403	AFD-Net [13]
   404	FASM [16]
   405	GI-ReID [17]
   406	CAL [22]
   407	3DSL [20]
   408	3DInvarReID [21]
   409	AIM [24]
   410	CCFA [12]
   411	IFD
   412	
   413	Modality
   414	RGB
   415	RGB
   416	sketch
   417	RGB+pose
   418	RGB
   419	RGB+aug
   420	RGB+pos+sil
   421	RGB+sil
   422	RGB
   423	RGB+3D
   424	RGB+3D
   425	RGB
   426	RGB+aug
   427	RGB+parsing
   428	
   429	PRCC
   430	SC Mode
   431	CC Mode
   432	rank-1
   433	mAP rank-1
   434	mAP
   435	99.8
   436	97.0
   437	41.8
   438	38.7
   439	99.4
   440	98.3
   441	46.3
   442	45.9
   443	64.2
   444	34.4
   445	100
   446	97.2
   447	50.2
   448	48.6
   449	95.7
   450	42.8
   451	98.8
   452	54.5
   453	80.0
   454	33.3
   455	100
   456	99.8
   457	55.2
   458	55.8
   459	51.3
   460	51.6
   461	52.5
   462	100
   463	99.9
   464	57.9
   465	58.3
   466	99.6
   467	98.7
   468	61.2
   469	58.4
   470	99.0
   471	98.2
   472	65.3
   473	61.7
   474	
   475	LTCC
   476	General Mode
   477	CC Mode
   478	rank-1
   479	mAP
   480	rank-1
   481	mAP
   482	65.1
   483	30.6
   484	23.5
   485	10.0
   486	63.7
   487	31
   488	25.0
   489	12.6
   490	71.4
   491	34.3
   492	26.2
   493	12.4
   494	73.2
   495	40.8
   496	38.5
   497	16.2
   498	63.2
   499	29.4
   500	23.7
   501	10.4
   502	74.2
   503	40.8
   504	40.1
   505	18.0
   506	31.2
   507	14.8
   508	37.8
   509	16.7
   510	76.3
   511	41.1
   512	40.6
   513	19.1
   514	75.8
   515	42.5
   516	45.3
   517	22.1
   518	81.4
   519	65.3
   520	64.3

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf' - | nl -ba | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
     2	
     3	5097
     4	
     5	Improving Text-Based Person Retrieval by
     6	Excavating All-Round Information
     7	Beyond Color
     8	Aichun Zhu , Zijie Wang, Jingyi Xue, Xili Wan , Member, IEEE, Jing Jin,
     9	Tian Wang , Senior Member, IEEE, and Hichem Snoussi
    10	
    11	Abstract— Text-based person retrieval is the process of searching a massive visual resource library for images of a particular
    12	pedestrian, based on a textual query. Existing approaches often
    13	suffer from a problem of color (CLR) over-reliance, which
    14	can result in a suboptimal person retrieval performance by
    15	distracting the model from other important visual cues such
    16	as texture and structure information. To handle this problem,
    17	we propose a novel framework to Excavate All-round Information
    18	Beyond Color for the task of text-based person retrieval, which
    19	is therefore termed EAIBC. The EAIBC architecture includes
    20	four branches, namely an RGB branch, a grayscale (GRS)
    21	branch, a high-frequency (HFQ) branch, and a CLR branch.
    22	Furthermore, we introduce a mutual learning (ML) mechanism
    23	to facilitate communication and learning among the branches,
    24	enabling them to take full advantage of all-round information
    25	in an effective and balanced manner. We evaluate the proposed
    26	method on three benchmark datasets, including CUHK-PEDES,
    27	ICFG-PEDES, and RSTPReid. The experimental results demonstrate that EAIBC significantly outperforms existing methods
    28	and achieves state-of-the-art (SOTA) performance in supervised,
    29	weakly supervised, and cross-domain settings.
    30	Index Terms— Color (CLR) information, cross-modal retrieval,
    31	frequency, person reidentification (ReID), text-based person
    32	retrieval.
    33	
    34	I. I NTRODUCTION
    35	EXT-BASED person retrieval is the process of searching a massive visual resource library for images of a
    36	particular pedestrian, based on a textual query. Since query
    37	sentences are easier to access than other types of query (such
    38	as images), this approach is particularly important in video
    39	surveillance applications and has gained increasing attention.
    40	
    41	T
    42	
    43	Manuscript received 4 March 2023; revised 4 November 2023;
    44	accepted 14 February 2024. Date of publication 28 February 2024; date of
    45	current version 1 March 2025. This work was supported in part by the National
    46	Natural Science Foundation of China under Grant 62101245 and Grant
    47	61972016, and in part by the Natural Science Research of Jiangsu Higher
    48	Education Institutions of China under Grant 21KJB520008. (Corresponding
    49	author: Aichun Zhu.)
    50	Aichun Zhu, Zijie Wang, Jingyi Xue, Xili Wan, and Jing Jin are
    51	with the College of Computer and Information Engineering, Nanjing
    52	Tech University, Nanjing 211816, China (e-mail: aichun.zhu@njtech.edu.cn;
    53	zijiewang9928@gmail.com).
    54	Tian Wang is with the Institute of Artificial Intelligence, SKLCCSE,
    55	Beihang University, Zhongguancun Laboratory, Beijing 100191, China
    56	(e-mail: wangtian@buaa.edu.cn).
    57	Hichem Snoussi is with the Institute Charles Delaunay-LM2S FRE CNRS
    58	2019, University of Technology of Troyes, Troyes 10004, France (e-mail:
    59	hichem.snoussi@utt.fr).
    60	Digital Object Identifier 10.1109/TNNLS.2024.3368217
    61	
    62	Fig. 1. Text-based person retrieval examples given by a single-branch basic
    63	model. The targeted/untargeted person images are marked with green/red
    64	borders.
    65	
    66	However, while image-based person retrieval (also known as
    67	person ReID) has been extensively studied, the text-based
    68	person retrieval task is still in its early stages of development.
    69	The primary challenge in the text-based person retrieval
    70	task is to extract and align relevant clues from multimodal
    71	data sources, including RGB pedestrian images and natural
    72	language queries. Various methods have been employed to
    73	tackle this task, but many still struggle with the problem of
    74	color (CLR) over-reliance. Utilizing a single-branch basic
    75	model (which is detailed in Section III-B), we display some
    76	retrieval examples in Fig. 1. The targeted (untargeted) pedestrian images are marked with green (red) borders. It can be
    77	noticed that within the top-ten retrieval results, quite a few
    78	images that do not align with the query description still share
    79	similar CLRs with the targeted images. This indicates a heavy
    80	
    81	2162-237X © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
    82	See https://www.ieee.org/publications/rights/index.html for more information.
    83	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
    84	
    85	5098
    86	
    87	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
    88	
    89	Fig. 2. Illustration of some feature response maps on pairs of RGB and GRS
    90	pedestrian images, which are obtained via two single-branch basic models are
    91	trained, respectively, on RGB and GRS data. By averaging along the channel
    92	dimension, we compute the feature response maps for all the images. In each
    93	column (numbered from 1 to 7), the first and third rows are, respectively, the
    94	RGB and GRS images while the second and fourth ones are the corresponding
    95	RGB and GRS response maps.
    96	
    97	reliance on CLR information in existing text-based person
    98	retrieval methods. However, the model may ignore subtle yet
    99	discriminative cues, such as the presence of items like “a
   100	camera bag,” “a cross-body bag,” or “high top shoes,” leading
   101	to retrieval failures. In certain cases, an image may still rank
   102	at the top of the result list by the model, even if the CLRs in
   103	local regions do not accurately match the given textual query.
   104	While CLR information undoubtedly plays a significant role
   105	in computing cross-modal affinity, the over-reliance on CLR
   106	information can divert the attention of the model from other
   107	crucial visual cues such as texture and structural information, ultimately resulting in suboptimal retrieval performance.
   108	A typical case can be observed in the second example from
   109	Fig. 1. In this instance, the CLR of the T-shirt in the target
   110	images is a very pale shade of blue, almost akin to gray.
   111	However, the query text merely states “a blue t-shirt” without
   112	specifying its lightness. Consequently, a single-branch basic
   113	model tends to retrieve numerous unrelated images where the
   114	clothing is distinctly blue. This mismatch in CLR description
   115	consequently results in a failed retrieval case. Therefore,
   116	alleviating the problem of CLR over-reliance can be a crucial
   117	factor in further promoting future research. To be more clear,
   118	two single-branch basic models are, respectively, trained on
   119	RGB and grayscale (GRS) data, by which the feature response
   120	maps are generated on pairs of RGB and GRS person images
   121	and illustrated in Fig. 2. As can be seen, for the RGB and
   122	GRS data, the attention of the models are drawn to varied
   123	local parts, implying that the complementary effects between
   124	
   125	the RGB and GRS data could be taken advantage and hence
   126	the problem of CLR over-reliance could be alleviated.
   127	To this end, in this article, a novel framework is designed
   128	to Excavate All-round Information Beyond Color for the textbased person retrieval task, which is therefore termed EAIBC.
   129	Specifically, to address the problem of CLR over-reliance,
   130	we introduce a jointly optimized multibranch architecture
   131	consists of four branches, namely an RGB branch, a GRS
   132	branch, a high-frequency (HFQ) branch and a CLR
   133	branch. The GRS branch employs a color deprivation module
   134	(CDM) to obtain GRS images, while the color masking module (CMM) masks the words related to CLR information in
   135	textual descriptions. This ensures that the GRS branch focuses
   136	on non-CLR clues in retrieving. Besides, within an image, the
   137	low-frequency information cares more about appearance and
   138	CLR, while the HFQ information majorly attends to details
   139	like texture and contour information. Therefore, for the HFQ
   140	branch, a high-frequency extraction module (HEM) is adopted
   141	to obtain the HFQ information from the input raw image,
   142	which enables EAIBC to explicitly key cues like textural and
   143	structural information other than CLR. Furthermore, in order
   144	to ensure that EAIBC fully utilizes all-round information in
   145	an effective and balanced way, and does not overly emphasize certain information while ignoring others, we include a
   146	CLR branch which is specifically designed to focus on CLR
   147	information. Additionally, a mutual learning (ML) mechanism [1] has been implemented to enable the four branches
   148	to communicate with and learn from each other. We evaluate
   149	our proposed method on three text-based person retrieval
   150	datasets, namely CUHK-PEDES [2], ICFG-PEDES [3] and
   151	RSTPReid [4]. Our experimental results show that EAIBC
   152	outperforms existing methods and achieves state of the art
   153	(SOTA) performance in supervised [2], weakly supervised
   154	[5] and cross-domain [6] text-based person retrieval tasks.
   155	To sum up, the major contributions of this article include
   156	the following.
   157	1) This article proposes a jointly optimized multibranch
   158	architecture termed as EAIBC to Excavate All-round
   159	Information Beyond Color and address the problem
   160	of CLR over-reliance. The framework includes four
   161	branches including an RGB branch, a GRS branch,
   162	an HFQ branch, and a CLR branch.
   163	2) An ML mechanism is introduced to facilitate communication and learning among the four branches, which
   164	allows for an effective and balanced use of all-round
   165	information.
   166	3) To our knowledge, this article is the first to use GRS
   167	data in addition to RGB data to improve performance in
   168	text-based person retrieval.
   169	4) The experimental results on CUHK-PEDES, ICFGPEDES and RSTPReid, as well as extensive ablation
   170	analysis, demonstrating that EAIBC outperforms existing methods and achieves SOTA performance in
   171	supervised [2], weakly supervised [5], and cross-domain
   172	[6] text-based person retrieval tasks.
   173	This work is an extension of our previous ACM MM 2022 conference paper CAIBC [7]. The contributions of this article over
   174	CAIBC can be concluded as follows.
   175	
   176	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   177	
   178	ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL
   179	
   180	1) An HFQ branch is proposed to explicitly care for
   181	extreme signals like texture and contour information
   182	within a person image.
   183	2) In addition to CUHK-PEDES and RSTPReid, the
   184	ICFG-PEDES dataset is further utilized to evaluate the
   185	performance of EAIBC.
   186	3) EAIBC is further evaluated in the cross-domain textbased person retrieval setting to validate its generalization ability and robustness.
   187	4) A more comprehensive quantitative analysis is carried
   188	out to systematically demonstrate the effectiveness of
   189	the proposed components within EAIBC.
   190	5) After visualization of the feature response maps along
   191	with text-based person retrieval examples given by
   192	EAIBC, a detailed discussion is carried out to understand
   193	the mechanism behind EAIBC. This analysis identifies
   194	some of the current challenges in text-based person
   195	retrieval and suggests areas for further research.
   196	II. R ELATED W ORKS
   197	A. Person Reidentification
   198	Person reidentification (ReID) aims to match person images
   199	across disjoint cameras [8], [9], [10], [11], [12], [13].
   200	To address this problem, previous ReID methods focus either
   201	on designing discriminative representations for human appearance or on learning a reliable affinity metric for the input
   202	data. To further enhance the representation capability of deep
   203	neural networks, researchers have proposed novel modules
   204	such as the second-order non-local attention (SONA) module
   205	by Xia et al. [14], which learns multigranular information and
   206	relationships in an end-to-end manner. A Gabor convolution
   207	module [15] is constructed based on the Gabor function for
   208	capturing texture representation, which is particularly effective
   209	when integrated into the lower layers of the network. Besides,
   210	based on the hinge function, a novel regularizer loss is
   211	proposed to further enhance this module. Bak and Carr [16]
   212	propose a one-shot learning method that decomposes the ReID
   213	metric. Hao et al. [17] present the modality confusion learning
   214	network (MCLNet), with the aim of confusing two modalities
   215	during optimization to focus solely on modality-irrelevant
   216	information.
   217	B. Cross-Modal Retrieval
   218	Text-based person retrieval can be considered as a subtask
   219	of cross-modal retrieval. Yu et al. [18] propose a novel
   220	ranking model based on the learning to rank framework.
   221	This model simultaneously leverages visual features and click
   222	features to construct the ranking model. To be more precise,
   223	their approach is rooted in large-margin structured output
   224	learning, and it integrates visual consistency with click features using a hypergraph regularizer term. Wang et al. [19]
   225	present a robust multiview hashing (RMVH) framework to
   226	better handle the information loss problem during learning
   227	the common semantic subspace. Meng et al. [20] propose
   228	an asymmetric supervised consistent and specific hashing
   229	(ASCSH) method to improve multimodal mapping learning.
   230	Yang et al. [21] introduce a controlled semantic embedding
   231	
   232	5099
   233	
   234	(CSE) framework, which focuses on learning disentangled representations characterized by a controlled semantic structure
   235	for cross-modal retrieval. A semantic disentanglement adversarial hashing (SDAH) [22] is designed to separate the original
   236	features of each modality into two distinct components:
   237	modality-common features with semantic information and
   238	modality-private features containing disturbing information.
   239	Following this initial disentanglement, the modality-private
   240	features are shuffled and treated as positive interactions to
   241	enhance the learning of modality-common features. This
   242	approach significantly enhances the discriminative capabilities
   243	and robustness of semantic embeddings.
   244	C. Text-Based Person Retrieval
   245	Text-based person retrieval involves searching for a pedestrian image in a large database based on a given text query.
   246	In 2015, Ye et al. [23] introduced the Specific Person Retrieval
   247	via Incomplete Text Description task with the aim of identifying pedestrian images according to attributes provided by
   248	users, which can be considered as a basis for the task of
   249	text-based person retrieval. Along with this task, a specific
   250	attribute completion method is designed to enhance and convert a text query into a more vector for the attributes. And
   251	after that, the text-based person retrieval task is formally
   252	proposed by Li et al. [2] in the year of 2017. To tackle
   253	this task, researchers have developed various methods that
   254	adopt cross-modality attention mechanisms to match visual
   255	and textual features. The goal of these approaches is to
   256	improve the relevance of image-text matching by calculating
   257	weights for the cross-modal alignments. Chen et al. [24]
   258	propose an efficient patch-word matching model to catch
   259	the fine-grained similarities between images and sentences.
   260	To exploit the multilevel visual information, Jing et al. [25]
   261	develop a posture-guided multigranularity attention network
   262	(PMA) and use posture clues as semantic masks to locate
   263	key parts. However, when applying pretrained models to this
   264	task, external clues obtained from these models may deviate
   265	significantly from the target data because of the gap of domain
   266	between the pretraining and targeted data. In addition to the
   267	attention-based methods, there are also common embeddingbased approaches that measure the affinities for cross-modal
   268	sample pairs in a latent common space. For instance, a new
   269	system presented by Zheng et al. [26] is capable of mapping
   270	multimodal data into a common space. To explicitly consider
   271	the intramodal data distribution, they propose the instance loss.
   272	Besides, text-image modality adversarial matching (TIMAM)
   273	which is introduced by Sarafianos et al. [27] take the adversarial learning mechanism into consideration so as to learn
   274	feature representations that are invariant to modality. However,
   275	these methods tend to ignore the importance of part representations in text-based person retrieval. To address this issue,
   276	some recent papers seek more fine-grained paradigms. For
   277	example, Niu et al. [28] propose the multigranularity imagetext alignments (MIA) model, which exploits the possibility of
   278	measuring affinities across different granularities. Additionally,
   279	some works have utilized technique like graph learning to
   280	better model the relationship within the multimodal data, such
   281	as the adversarial graph attention network (A-GANet) model
   282	
   283	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   284	
   285	5100
   286	
   287	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
   288	
   289	Fig. 3. Illustration of the EAIBC framework, which consists of RGB, GRS, HFQ, and CLR branches. The terms CDM, HEM, CMM, and CPM, respectively,
   290	denote color deprivation module, high-frequency extraction module, color masking module, and color prior module.
   291	
   292	by Liu et al. [29]. By means of the semantic scene graphs, the
   293	structural features of the visual and textual data are modeled.
   294	Aggarwal et al. [30] propose cross modal attribute aided
   295	matching (CMAAM), a method that learns two spaces which
   296	are, respectively, driven by attributes and class information,
   297	by leveraging the attribute annotations and predictions. With
   298	a Gumbel attention module, a hierarchical adaptive matching
   299	model is introduced by Zheng et al. [31] to tackle the problem
   300	of matching redundancy. A deep surroundings-person separation learning (DSSL) model is proposed by Zhu et al. [4]
   301	along with a new RSTPReid dataset. Ding et al. [3] propose the semantically self-aligned network (SSAN) method to
   302	extract semantically aligned part features for the multimodal
   303	data. Besides, an ICFG-PEDES dataset is proposed as well.
   304	Recently, with the development of contrastive language-image
   305	pretraining (CLIP) [32], some researchers begin to exploit the
   306	great cross-modal alignment capabilities of vision-language
   307	pretrained models. Han et al. [33] introduce a cross-modal
   308	momentum contrastive learning framework with the goal of
   309	enriching the training data within a given mini-batch. This
   310	approach addresses issues stemming from limited data availability. Yan et al. [34] propose the CLIP-driven Fine-grained
   311	information excavation framework (CFine) to fully harness
   312	the robust knowledge of CLIP for text-based person retrieval.
   313	To effectively transfer multimodal knowledge, they conduct
   314	fine-grained information excavation to extract intramodal discriminative clues and establish intermodal correspondences.
   315	Jiang and Ye [35] introduce the cross-modal implicit relation
   316	reasoning and aligning (IRRA) framework, which focuses
   317	on learning relations between local visual-textual tokens and
   318	enhancing global image-text matching without the need for
   319	additional prior supervision. To be specific, Lapscore [36]
   320	enhances the model’s ability to capture CLR information
   321	
   322	and fine-grained cross-modal association by introducing two
   323	subtasks on CLR reasoning, namely text completion and image
   324	colorization (TC&IC), during training. On the contrary, our
   325	proposed EAIBC method aims to address the issue of CLR
   326	over-reliance by utilizing other important cues.
   327	III. M ETHODOLOGY
   328	A. Problem Formulation
   329	The proposed framework, as illustrated in Fig. 3, aims to
   330	measure the affinity between textual queries and person images
   331	in a database. Formally, within a dataset D = {pi , qi }i = 1 N
   332	consisting of N image-text pairs, each pair includes a person
   333	image pi which is captured by a surveillance camera along
   334	with its corresponding textual query qi . The Q unique identifiQ
   335	cations (IDs) of the pedestrians are represented as Y = {yi }i=1
   336	.
   337	The objective is to search for the most relevant images of the
   338	pedestrians in the gallery given a textual query.
   339	B. Branch Prototype
   340	To begin with, a branch prototype that extracts and aligns
   341	the multimodal multigranular representations is introduced.
   342	This prototype can then be used for implementing the RGB,
   343	GRS and HFQ branches in EAIBC.
   344	a) Visual representation extraction: To obtain the global
   345	and local visual representations from an input image, a pretrained ResNet-50 [37] backbone is utilized. For the global
   346	granularity, a global max pooling operation is employed to
   347	down-scale the feature map before the last pooling layer of
   348	ResNet-50 to the shape of 2048 × 1 × 1, which is then converted to a 2048-D vector. This vector is then passed through
   349	a fully-connected (FC) layer and transformed to P-dim as the
   350	global representation vg ∈ R P . For the local granularity, the
   351	
   352	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   353	
   354	ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL
   355	
   356	5101
   357	
   358	global max pooling operation is adopted to down-scale the
   359	same feature map to the shape of 2048 × K × 1, so as
   360	to horizontally K -partition the feature map. Each of the local
   361	strips is then separately passed through a fully-connect layer
   362	to generate K P-dim fine-grained local visual representations
   363	K
   364	V f = {v f k }k=1
   365	.
   366	b) Textual representation extraction: For extracting textual representations, we utilize a bidirectional Gated Recurrent
   367	Unit (bi-GRU) after word embedding to process the input
   368	sentence. The ith last hidden states of the forward and backward GRUs are averaged to represent the ith word in the
   369	query sentence as ei . To represent the entire sentence, the
   370	C word representations are concatenated to form E, which
   371	undergoes a rowwise max pooling (RMP) operation along with
   372	a fully-connect layer to obtain the global textual representation
   373	tg ∈ R P . To obtain K local textual representations based
   374	on word-part correspondences, a Word Attention Module is
   375	employed based on [3]
   376	
   377	
   378	C
   379	sik = σ Wkp ei , Ek = sik ei i=1 , k = {1, 2, . . . , K }
   380	(1)
   381	where σ stands for the Sigmoid function and Wkp ∈ R1×C
   382	is a linear transformation operation. Next, we process each
   383	modified sentence representation Ek separately using RMP +
   384	FC and stack them together to create the fine-grained local
   385	K
   386	textual representations T f = {t f k }k=1
   387	.
   388	After obtaining the fine-grained visual and textual representations V f and T f , they are concatenated to form vl and tl
   389	∈ R K P . The cross-modal global and local similarities are then
   390	computed as follows:
   391	Sg =
   392	
   393	vgT tg
   394	||vg ||||tg ||
   395	
   396	, Sl =
   397	
   398	vlT tl
   399	.
   400	||vl ||||tl ||
   401	
   402	(2)
   403	
   404	C. Color Deprivation and Masking
   405	To obtain the corresponding GRS image Igrs for a given
   406	RGB image Irgb ∈ R3×H ×W , we propose a CDM that can be
   407	formulated as follows:
   408	
   409	
   410	
   411	 R(i, j)
   412	GRS(i, j) = 0.299 0.587 0.114 G(i, j)
   413	(3)
   414	B(i, j)
   415	where i ∈ {1, 2, . . . , H }, j ∈ {1, 2, . . . , W }. GRS(i, j),
   416	R(i, j), G(i, j) and B(i, j) stand for the values of the pixels
   417	in the ith row and the jth column of the GRS and RGB
   418	channels, respectively. The empirical coefficients 0.299, 0.587,
   419	and 0.114 are commonly adopted in image processing software
   420	such as OpenCV and Photoshop for converting RGB images
   421	to GRS. In order to using the ResNet-50 visual backbone to
   422	deal with the obtained GRS images, we replicate the singlechannel GRS images three times along the dimension of the
   423	channel.
   424	For removing CLR information from the textual modality,
   425	a CMM is designed. After collecting the words related to
   426	CLR information that appear with HFQ from the textual query
   427	corpus, a CLR bank is formed. The frequency-based CLRrelated word cloud is displayed in Fig. 4. Consequently, when a
   428	textual query sentence is given, all of the words related to CLR
   429	are converted into a masking token [CLR]. For instance, given
   430	
   431	Fig. 4. Frequency-based CLR-related word cloud constructed from the CLR
   432	bank.
   433	
   434	a textual query “This man is wearing an orange coat with a
   435	gray-orange backpack. He has light blue jeans and black-white
   436	sneakers”, the output of CMM will be “This man is wearing
   437	an [CLR] coat with a [CLR] backpack. He has light [CLR]
   438	jeans and [CLR] sneakers.”
   439	It is worth noting that both CDM and CMM do not require
   440	learning any parameters, which can be directly utilized to
   441	handle the input raw multimodal data.
   442	D. High-Frequency Information Extraction
   443	To extract HFQ components from a raw image I rgb ∈
   444	R
   445	and obtain the corresponding HFQ image I hfq ∈
   446	3×H ×W
   447	R
   448	, an HEM is employed. In practice, HEM can be
   449	implemented in varied manners, and in this article the Gaussian filter is utilized for extracting the HFQ information.
   450	3×H ×W
   451	
   452	E. Excavating All-Round Information via Multibranch
   453	Architecture
   454	1) RGB Branch: In this branch, an RGB image and a textual
   455	query sentence are provided as inputs. The output consists of
   456	visual and textual multigranular RGB representations, denoted
   457	rgb
   458	rgb
   459	rgb
   460	rgb
   461	as vg / tg ∈ R P and vl / tl ∈ R K P , respectively.
   462	To calculate the similarities for this branch, the following
   463	formula is used:
   464	 T
   465	 T
   466	rgb
   467	rgb
   468	rgb
   469	rgb
   470	vg
   471	tg
   472	vl
   473	tl
   474	rgb
   475	Srgb
   476	=
   477	,
   478	S
   479	=
   480	.
   481	(4)
   482	g
   483	l
   484	rgb
   485	rgb
   486	rgb
   487	rgb
   488	||vg ||||tg ||
   489	||vl ||||tl ||
   490	2) GRS Branch: The GRS branch accepts a GRS image
   491	obtained through CLR deprivation and a textual description
   492	after masking the words related to CLR information. It also
   493	grs
   494	grs
   495	grs
   496	grs
   497	produces vg / tg ∈ R P and vl / tl ∈ R K P , just like the
   498	RGB branch. The similarities for the GRS branch are then
   499	obtained by
   500	grs T grs
   501	grs T grs
   502	vg
   503	tg
   504	vl
   505	tl
   506	grs
   507	grs
   508	Sg = grs
   509	(5)
   510	grs , Sl =
   511	grs
   512	grs .
   513	||vg ||||tg ||
   514	||vl ||||tl ||
   515	3) HFQ Branch: The HFQ branch processes the obtained
   516	HFQ image and the original textual description. It generates
   517	hfq
   518	hfq
   519	hfq
   520	hfq

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '760,1180p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   760	
   761	HSL
   762	
   763	SLE
   764	
   765	�
   766	
   767	eg t
   768	
   769	a
   770	
   771	VIS IR eat e
   772	�
   773	�
   774	
   775	Stage VIS IR eat e
   776	
   777	e e ge
   778	
   779	ae
   780	
   781	S
   782	
   783	t a ge VIS IR eat e
   784	
   785	S
   786	
   787	t a ge e at
   788	
   789	e
   790	
   791	�
   792	�
   793	
   794	g a ge e at
   795	
   796	g a ge VIS IR eat e
   797	
   798	�
   799	
   800	�
   801	
   802	�
   803	
   804	�
   805	
   806	e eat e age t
   807	
   808	e
   809	
   810	a e VIS IR eat e
   811	
   812	e eat e
   813	
   814	�
   815	�
   816	
   817	×
   818	
   819	a e VIS IR eat e
   820	
   821	m
   822	
   823	e eat e
   824	
   825	�
   826	�
   827	
   828	S
   829	
   830	t a
   831	
   832	g a ge eat e et
   833	
   834	Re at
   835	
   836	e
   837	
   838	a e
   839	
   840	eat e et
   841	
   842	t a ge e at
   843	
   844	e
   845	
   846	a e VIS IR eat e
   847	
   848	g a ge e at
   849	
   850	e
   851	
   852	a e VIS IR eat e
   853	
   854	e eat e et
   855	
   856	Fig. 2. Overall of the proposed HOH-Net, including a high-order structure learning (HSL) module and a fine-coarse graph attention alignment (FCGA)
   857	module. The HOH-Net is jointly optimized by LCE , LT RI , and a hierarchical middle-feature agent learning (HMAL) loss LHM AL , and a modality-range
   858	identity-center contrastive loss LM RIC .
   859	
   860	(FCGA) module to leverage high-order node-level and regionlevel features for achieving reliable middle features. Besides,
   861	we also generate the middle-feature agents to hierarchically
   862	mitigate the modality gap at each modality-shared feature
   863	extraction stage by introducing a hierarchical middle-feature
   864	agent learning (HMAL) loss.
   865	III. P ROPOSED M ETHOD
   866	A. Overview
   867	The overall of our proposed HOH-Net is given in Fig. 2. The
   868	HOH-Net mainly consists of an HSL module and an FCGA
   869	module with the HMAL loss and the MRIC loss. In this paper,
   870	we adopt a two-stream AGW [1] as the backbone. Firstly,
   871	we feed the VIS-IR image pair with the same identity to the
   872	backbone for obtaining paired VIS-IR features. Then, the HSL
   873	module introduces a whitened hypergraph network to exploit
   874	high-order structure information of short-range and long-range
   875	features that are obtained from the short- and long-range
   876	feature extraction (SLE) module. Furthermore, the FCGA
   877	module aligns different modalities and ranges of features to
   878	generate reliable middle features effectively at the node-level
   879	and region-level. The HMAL loss mitigates the modality discrepancy hierarchically based on the middle-feature agents and
   880	can constrain the bi-directional interaction between different
   881	stages to improve the representation of features. Besides, in
   882	the embedding space, we develop an MRIC loss to reduce the
   883	distances between the VIS, IR, and middle features, greatly
   884	smoothing the process of learning the common feature space.
   885	
   886	B. High-Order Structure Learning (HSL) Module
   887	Suppose that we have paired VIS-IR images, denoted as
   888	{Ivis , Iir } with the same identity label. We first extract VIS
   889	features Bvis and IR features Bir from the backbone network,
   890	respectively. Then, Bvis and Bir are passed through the SLE
   891	module (more details of SLE can be seen in our previous work
   892	ir
   893	[12]) to extract both short-range features (Fvis
   894	S /FS ) and longir
   895	range features (Fvis
   896	/F
   897	)
   898	for
   899	VIS
   900	and
   901	IR
   902	modalities.
   903	Thus,
   904	L
   905	L
   906	vis
   907	ir
   908	ir
   909	we can obtain a feature set Q = {Fvis
   910	,
   911	F
   912	,
   913	F
   914	,
   915	F
   916	}.
   917	The
   918	L
   919	S
   920	L
   921	S
   922	sizes of each feature in Q are RH⇥W ⇥C , where H, W , and
   923	C correspond to the height, the width, and the number of
   924	channels of the features, respectively.
   925	The backbone network and the SLE module just capture
   926	pixel-level and region-level dependencies within person images. However, they can not fully exploit the high-order
   927	structural information that delineates complex relationships
   928	among features (e.g., the head, torso, upper arm, and lower
   929	arm are parts of the upper body while head, torso, arm, and
   930	leg belong to the whole body). Inspired by the Hypergraph
   931	Neural Network (HGNN) [32], we introduce an HSL module
   932	to better capture high-order correlations, thereby enriching the
   933	feature representations. Besides, due to the small differences
   934	in the feature nodes of pedestrians, the conventional HGNN
   935	tends to suffer from the problem of model collapse that leads
   936	to the diverse and complex relationships tending to be the
   937	same. To deal with this problem, we make good use of the
   938	whitening operation and apply it to the hypergraph network,
   939	
   940	Page 5 of 26
   941	
   942	IEEE Transactions on Circuits and Systems for Video Technology
   943	
   944	PREPRINT SUBMITTED TO TCSVT, JULY 2025
   945	
   946	e m
   947	
   948	(� − 1)−1
   949	
   950	⊖
   951	
   952	⨂
   953	
   954	(� − 1�� )
   955	
   956	e
   957	
   958	�(�’)
   959	
   960	t
   961	
   962	(�)−1
   963	
   964	�(�’)
   965	
   966	⨂
   967	�
   968	
   969	I
   970	
   971	t
   972	I ta
   973	
   974	[
   975	
   976	e
   977	
   978	]
   979	
   980	��T − �T�
   981	
   982	a e�
   983	
   984	e
   985	�� , � �
   986	
   987	S
   988	te
   989	
   990	�
   991	
   992	e
   993	
   994	�’
   995	
   996	a
   997	
   998	g
   999	
  1000	t
  1001	
  1002	at
  1003	
  1004	⊕
  1005	
  1006	t
  1007	
  1008	⊕
  1009	
  1010	�
  1011	
  1012	�
  1013	
  1014	1
  1015	−
  1016	
  1017	� 2 �−1
  1018	
  1019	�2
  1020	
  1021	T
  1022	
  1023	�−1 ⨂
  1024	
  1025	the F0 . The diagonal operation in ⇤(·) is used to capture the
  1026	context relationship of the whitened nodes and determine their
  1027	distance contributions to the corresponding hyperedges, where
  1028	⇤(F0 ) 2 RC⇥C . "(·) is the step function. Hence in this endto-end trainable way, the high-order structure information in
  1029	person features can be well exploited with H.
  1030	Based on the above H, we introduce the hypergraph convolutional operation [32] to aggregate high-order structure
  1031	information and the high-order relation-enhanced feature R 2
  1032	RN ⇥C can be obtained as
  1033	
  1034	⊖S t a t
  1035	⨂
  1036	
  1037	e
  1038	
  1039	�(�’)T
  1040	
  1041	1
  1042	
  1043	�
  1044	
  1045	e e ge
  1046	
  1047	e e ge
  1048	
  1049	�(�’)
  1050	
  1051	��T
  1052	
  1053	(� − 1�� )T
  1054	
  1055	ea
  1056	
  1057	�
  1058	e
  1059	
  1060	1
  1061	2
  1062	3
  1063	4
  1064	5
  1065	6
  1066	7
  1067	8
  1068	9
  1069	10
  1070	11
  1071	12
  1072	13
  1073	14
  1074	15
  1075	16
  1076	17
  1077	18
  1078	19
  1079	20
  1080	21
  1081	22
  1082	23
  1083	24
  1084	25
  1085	26
  1086	27
  1087	28
  1088	29
  1089	30
  1090	31
  1091	32
  1092	33
  1093	34
  1094	35
  1095	36
  1096	37
  1097	38
  1098	39
  1099	40
  1100	41
  1101	42
  1102	43
  1103	44
  1104	45
  1105	46
  1106	47
  1107	48
  1108	49
  1109	50
  1110	51
  1111	52
  1112	53
  1113	54
  1114	55
  1115	56
  1116	57
  1117	58
  1118	59
  1119	60
  1120	
  1121	5
  1122	
  1123	ega
  1124	
  1125	t
  1126	
  1127	t t
  1128	e
  1129	
  1130	g
  1131	
  1132	a e �’
  1133	
  1134	e t
  1135	
  1136	t e eat e �
  1137	
  1138	ega
  1139	t
  1140	
  1141	e at
  1142	
  1143	Fig. 3. The detailed network architecture of the proposed HSL module.
  1144	
  1145	R = (I
  1146	as shown in Fig. 3.
  1147	Different from the typical graph models that connect only
  1148	pairwise nodes, hypergraphs provide a more sophisticated
  1149	structure by allowing connections between an arbitrary number of nodes, thereby describing the high-order structural
  1150	information. For each feature within the set Q, we construct
  1151	a whitened hypergraph, denoted as G = (V, E, W). Here,
  1152	V = {v1 , v2 , · · · , vN }, E = {e1 , e2 , · · · , eM } and
  1153	W represent the node set, the hyperedge set and the weight
  1154	matrix, respectively. N = HW and M correspond to the numbers of nodes and hyperedges, respectively. In this paper, each
  1155	1 ⇥ 1 ⇥ C grid from the feature in Q is considered as a feature
  1156	node. The n-th node is represented as fn 2 R1⇥C , and thus all
  1157	nodes can be represented by F = [f1 ; f2 ; · · · ; fN ] 2 RN ⇥C .
  1158	The conventional hypergraph network [32] is designed to
  1159	enable unrestricted node connections to capture high-order
  1160	structural information. However, it easily suffers from model
  1161	collapse (i.e., the nodes connected by different hyperedges
  1162	are the same) during hypergraph learning. To overcome this
  1163	difficulty, we introduce a whitening operation to project the
  1164	nodes into a spherical distribution and facilitate the learning
  1165	of subtle high-order relationships. The whitening operation
  1166	plays the role of “scattering” on the nodes, thereby preventing
  1167	the diverse high-order connections from converging into a
  1168	single connection. As a result, this approach enables us to
  1169	explore various high-order relationships across these features
  1170	effectively.
  1171	The whitened node fn0 can be obtained as
  1172	fn0 =
  1173	
  1174	n(
  1175	
  1176	1
  1177	
  1178	(fnT
  1179	
  1180	T

codex
EAIBC 的前 8 页覆盖了完整方法：观察来自检索失败样例和 RGB/灰度响应图差异，机制是四分支互学。还要补它的颜色分支和互学习优化段落，随后可以进入综合拆解。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf' - | nl -ba | sed -n '560,780p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf' - | nl -ba | sed -n '520,820p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   560	method SuA-SpML [45]. The results of direct cross dataset
   561	evaluation are presented in Table III. It can be observed that the
   562	optimal performance is achieved among all methods using the
   563	InfinitePerson dataset, surpassing all other competing synthetic
   564	datasets. This proves that the Re-ID model trained on our
   565	dataset has stronger generalization ability. What’s noteworthy
   566	is that our dataset’s superior performance compared to others
   567	stems from its high-quality 3D pedestrian models and diverse
   568	scene variations. These factors collectively promote the model
   569	to have stronger robustness.
   570	
   571	E. Quantitative Analysis
   572	To demonstrate that the UV texture maps generated by
   573	Stable Diffusion [11] can increase the diversity of 3D character
   574	models and to evaluate the effectiveness of the proposed
   575	sub-monitoring network for collecting synthetic datasets,
   576	we perform quantitative analysis.
   577	1) Identities: In contrast to UnrealPerson, which randomly
   578	matches clothing to create a large number of different clothing
   579	sets, we generate diverse sets of clothing models by replacing
   580	the UV texture maps of existing clothing models with those
   581	generated by diffusion models. To verify whether our proposed
   582	method of generating clothing textures can improve pedestrian
   583	diversity, we use 141 3D clothing models, including those with
   584	regular and irregular UV texture maps. For these 141 clothing
   585	models, unlike Unrealperson [8], we do not use random matching to obtain clothing sets. Instead, we choose to manually
   586	pair them into 152 clothing sets, resulting in 4,700 virtual
   587	characters with distinct clothing textures. Figure 8 clearly
   588	demonstrates that as the number of characters increases, the
   589	accuracy of Rank-1 and mAP consistently increases on all
   590	three real-world datasets, despite using only 152 clothing
   591	sets. This shows that the UV texture maps we generate can
   592	indeed increase the diversity of virtual characters, as shown
   593	in Table II. Furthermore, it’s worth noting that even if we
   594	increase the number of 3D character models from 3,000 to
   595	4,700, we observe no performance degradation. This finding
   596	challenges the notion as posited by UnrealPerson [8] that
   597	
   598	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   599	
   600	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   601	
   602	3167
   603	
   604	TABLE III
   605	D IRECT C ROSS DATASET E VALUATION R ESULTS U SING R E -ID M ETHODS . U NREAL P ERSON * M EANS THE T RAINING DATA E XTRACTED
   606	F ROM UNREAL _ V 1.1, UNREAL _ V 2.1, UNREAL _ V 3.1 AND UNREAL _ V 4.1
   607	
   608	Fig. 7. Results of direct transfer evaluation on CUHK03-NP, Market-1501 and MSMT17, varying the quantity of sub-monitoring networks in synthetic
   609	datasets from 2 to 12.
   610	
   611	optimal performance can only be achieved with 3,000 3D
   612	character models.
   613	2) Sub-Monitoring Network: To demonstrate the effectiveness of our proposed sub-monitoring network, we analyze it
   614	by setting the number of sub-monitoring networks. As shown
   615	in Figure 7, when we increase the number of sub-monitoring
   616	networks, the accuracy of Rank-1 and mAP continues to
   617	increase on all three real-world datasets. Even when we deploy
   618	sub-monitoring networks in different regions of the same large
   619	scene or use distinct networks for the same character group, the
   620	accuracy of Rank-1 and mAP consistently exhibits a consistent
   621	increase on all three real-world datasets. As the number of 3D
   622	character models and sub-monitoring networks increases, the
   623	performance remains consistent without showing any decline.
   624	This demonstrates the enormous potential of our method to
   625	further produce larger synthetic datasets.
   626	3) Dataset Parameter Analysis: To fully validate the
   627	effectiveness and applicability of our dataset, we also use
   628	QAConv2.0 [43] as the baseline and adjusted various parameters of our dataset for experiments. We first fix the number
   629	of pedestrians and change the number of clothing sets
   630	used, as shown in Table IV. The trend reveals that as the
   631	
   632	number of clothing types increases, the model’s generalization
   633	performance improves. Moreover, increasing the number of
   634	sub-monitoring networks and cameras further enhances the
   635	model’s generalization performance. This not only verifies the
   636	effectiveness of our sub monitoring network, but also verifies
   637	that using pedestrian images in multiple scenes for model
   638	training can improve the model’s generalization ability.
   639	F. Qualitative Comparisons
   640	As shown in Figure 9, there are significant differences in the
   641	UV texture maps generated using ControlNet [10] to control
   642	Stable Diffusion [11] based on different training datasets.
   643	When compared to ControlNet [10] trained on the normal
   644	map [36] dataset, the performance of ControlNet [10] trained
   645	on the Canny edge dataset [49] and that trained on the Instruct
   646	Pix2Pix dataset is notably inferior. Specifically, using the
   647	Instruct Pix2Pix dataset [50] as additional input to control the
   648	images generated by Stable Diffusion [11] are unsuitable for
   649	use as UV texture maps. Although employing the Canny maps
   650	as additional input to control the UV texture map generated by
   651	Stable Diffusion yields reasonable results, the generated image
   652	
   653	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   654	
   655	
 succeeded in 0ms:
   520	hfq
   521	vg / tg ∈ R P and vl / tl ∈ R K P , similar to the RGB
   522	
   523	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   524	
   525	5102
   526	
   527	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
   528	
   529	branch. The similarities for the HFQ branch can be computed
   530	as
   531	 T
   532	 T
   533	hfq
   534	hfq
   535	hfq
   536	hfq
   537	vl
   538	tl
   539	tg
   540	vg
   541	hfq
   542	hfq
   543	, Sl = hfq
   544	.
   545	(6)
   546	Sg = hfq
   547	hfq
   548	hfq
   549	||vg ||||tg ||
   550	||vl ||||tl ||
   551	4) CLR Branch: As stated in Section I, the aim of EAIBC
   552	is to ensure an effective and balanced use of all available
   553	information, instead of prioritizing some information while
   554	ignoring others. Therefore, it is undesirable for the model to
   555	excel at extracting non-CLR clues but fail to attend to critical
   556	CLR information. To ensure that CLR information is not
   557	neglected, we employ the CLR branch which focuses explicitly
   558	on clues related to CLR. Intuitively, the RGB branch is
   559	responsible for capturing general information, while the GRS
   560	branch is responsible for capturing other crucial details beyond
   561	CLR. As a result, the information that is not shared between
   562	these two branches can be considered pure information about
   563	CLR. In the visual modality, we subtract the output feature
   564	map Mgrs of the visual backbone within the GRS branch from
   565	Mrgb , the output feature map of the RGB branch, to obtain
   566	a pure CLR feature map Mclr . Then, we apply global max
   567	pooling and a fully-connect layer to Mclr to obtain the visual
   568	CLR representation vclr ∈ R P . And in the textual modality,
   569	we propose a color prior module (CPM). We first subtract the
   570	masked sentence representation Egrs from the representation
   571	Ergb of the whole sentence to obtain Eclr . And then we use
   572	the words related to CLR in the sentence as a CLR prior to
   573	enhancing the CLR information in Eclr . To do so, we sum
   574	the word embeddings of the CLR-related words and convert
   575	them to the same dimension as the word representations eiclr
   576	in Eclr . We add this CLR prior to each eiclr to obtain a CLR
   577	prior enhanced representation Ecp . Finally, we apply an RMP
   578	operation and a fully-connect layer to Ecp to obtain the textual
   579	CLR representation tclr ∈ R P . Finally, the similarity score for
   580	the CLR branch is computed by
   581	T
   582	vclr tclr
   583	clr
   584	S = clr
   585	.
   586	(7)
   587	||v ||||tclr ||
   588	F. Optimization
   589	As mentioned previously, the major target of EAIBC is
   590	to simultaneously optimize the four branches and utilize
   591	all-round information in an effective and balanced manner.
   592	Therefore, an ML mechanism is used to enable knowledge sharing among the branches. To achieve this, the
   593	Kullback–Leibler (KL) Divergence is used to measure the
   594	affinity between the logits of varied branches. Let vibr ∈
   595	rgb grs hfq
   596	rgb grs hfq
   597	{vgi , vgi , vgi , viclr } and tibr ∈ {tgi , tgi , tgi , ticlr } represent the
   598	visual and textual features extracted from the ith sample pair,
   599	br
   600	they are then utilized to compute the probabilities pbr
   601	vm and ptm
   602	of class (person ID) m as
   603	
   604	
   605	exp γ Wm vibr
   606	br
   607	br
   608	(8)
   609	pvm vi = P M
   610	
   611	br
   612	k=1 exp γ Wk vi
   613	
   614	
   615	exp γ Wm tibr
   616	br br
   617	ptm ti = P M
   618	(9)
   619	
   620	br
   621	k=1 exp γ Wk ti
   622	
   623	where γ Wm vibr and γ Wm tibr are logits input to the softmax
   624	layer in the br branch, and br can be rgb, grs, hfq or clr.
   625	Under the constraint of an ML loss, each branch is optimized,
   626	which can be calculated following:
   627	
   628	
   629	N
   630	M
   631	br
   632	1  X X X br br  pbr
   633	vm vi
   634	br
   635	
   636	p v log s
   637	LML =
   638	2 s∈B/{br } i=1 m=1 vm i
   639	pvm vibr
   640	
   641	N X
   642	M
   643	br br
   644	X X
   645	
   646	p
   647	t
   648	tm i 
   649	br
   650	
   651	pbr
   652	+
   653	(10)
   654	tm ti log s
   655	br
   656	p
   657	tm ti
   658	s∈B/{br } i=1 m=1
   659	where B = {rgb, grs, hfq, clr}.
   660	Furthermore, to train EAIBC, we also utilize the triplet
   661	ranking loss and ID loss, which are combined to form the
   662	ReID loss. It is worth noting that the ID loss is applied to
   663	hfq
   664	hfq
   665	grs
   666	grs
   667	rgb
   668	rgb
   669	each local representation v f k / t f k , v f k / t f k , and v f k / t f k ,
   670	grs
   671	rgb
   672	rgb
   673	whereas the triplet ranking loss is applied to vl / tl , vl /
   674	grs
   675	hfq
   676	hfq
   677	tl , and vl / tl .
   678	IV. E XPERIMENTS
   679	A. Experimental Setup
   680	1) Dataset: The proposed EAIBC is trained and evaluated
   681	on three challenging datasets for the task of Text-based Person
   682	Retrieval, namely CUHK-PEDES [2], ICFG-PEDES [3] and
   683	RSTPReid [4].
   684	a) CUHK-PEDES: As per the official data split approach
   685	described in [2], the CUHK-PEDES dataset is divided into
   686	three sets: training set, validation set, and test set. The training
   687	set includes 34 054 images of 11 003 persons and 68 126
   688	textual descriptions. The validation set consists of 3078 images
   689	of 1000 persons and 6158 textual descriptions, while the test
   690	set includes 3074 images of 1000 persons and 6156 textual
   691	descriptions.
   692	b) ICFG-PEDES: The ICFG-PEDES dataset consists
   693	of 54 522 images of 4102 persons based on the MSMT17
   694	dataset [38]. This dataset is divided into three subsets, with
   695	34 674 images of 3102 identities and 19 848 images of
   696	1000 identities used for training, test, respectively.
   697	c) RSTPReid: The RSTPReid dataset consists of 20 505
   698	images of 4101 persons, with each person having five images
   699	captured by different cameras. Each image is annotated with
   700	two textual descriptions. The dataset is divided into three
   701	subsets, with 3701, 200, and 200 identities used for training,
   702	validation, and testing, respectively.
   703	2) Evaluation Metrics: The performance of our approach is
   704	evaluated based on the Rank-k accuracy metric. Specifically,
   705	for each test query natural language sentence, all images in
   706	the test set are ranked based on their similarities with the
   707	textual query. A search is considered successful if at least one
   708	image of the targeted pedestrian is ranked within the top-k.
   709	Rank-k accuracy stands for the probability of finding at least
   710	one matching person image within the top-k candidate list
   711	when given a textual description as a query. The evaluation
   712	is conducted by reporting the Rank-1/5/10 accuracies for all
   713	ablation experiments.
   714	
   715	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   716	
   717	ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL
   718	
   719	5103
   720	
   721	TABLE I
   722	A BLATION A NALYSIS OF K EY C OMPONENTS ON CUHK-PEDES AND RSTPR EID
   723	
   724	B. Implementation Details
   725	For implementing EAIBC, the dimension P for the representations is set to 2048. Two independent pretrained
   726	ResNet-50 [37] models are selected for the visual backbones
   727	within the RGB and GRS branches. The word embedding
   728	dimension is 768 or 500 with or without the pretrained bidirectional encoder representations for transformers (BERT) [39].
   729	Images from all of the datasets are resized to 384 × 128 × 3.
   730	Random horizontal flipping along with the CLR jittering
   731	strategies are adopted for augmenting the visual data. To optimize the parameters of EAIBC, an Adam optimizer [40] is
   732	used. We set the initial learning rate for the visual backbones
   733	as 0.0001 while 0.001 for other parameters of EAIBC. The
   734	batch size is 64 and the training process lasts for 100 epochs.
   735	C. Ablation Analysis
   736	To evaluate the contribution and effectiveness of proposed
   737	components in EAIBC, comprehensive ablation experiments
   738	are conducted on the CUHK-PEDES and RSTPReid datasets.
   739	The results are reported in Table I and numbered from 1 to 29.
   740	Abl.1, Abl.2, and Abl.3 are conducted using a single
   741	RGB, GRS, and HFQ branch basic model (as described in
   742	Section III-B). Abl.4–Abl.9 are conducted without the CLR
   743	branch, and V or T in Table I indicate that the GRS branch
   744	is used only for the visual or textual modality. Abl.10–Abl.16
   745	are performed after adding the CLR branch, and Abl.28/29
   746	
   747	are carried out with the complete EAIBC model without/with
   748	a pretrained BERT model [39] for word embedding. In
   749	Abl.17–Abl.27, we analyze the performance of our proposed
   750	EAIBC method on the task of weakly supervised person
   751	retrieval. The Rank-1, Rank-5, and Rank-10 accuracies are
   752	reported for all ablation experiments.
   753	1) Impact of Multibranch Learning: As shown in Table I,
   754	while a single RGB branch model may outperform individual
   755	GRS or HFQ branch models, combining any two of these
   756	branches leads to a notable improvement in performance.
   757	Furthermore, adding the CLR branch, which specifically
   758	focuses on CLR information, further improves the retrieval
   759	performance. These observations suggest that the jointly optimized multibranch architecture in EAIBC allows for separate
   760	consideration of varied types of information from different
   761	perspectives, and the four branches are enabled to complement
   762	each other for superior retrieval performance.
   763	2) Impact of Mutual Learning: Comparing Abl.6 with
   764	Abl.7 in Table I, we observe that by utilizing the ML mechanism, the retrieval accuracies are improved by 0.91%, 0.42%,
   765	0.37% and 0.95%, 0.35% 0.40% under the Rank-1/5/10
   766	accuracies, respectively, on CUHK-PEDES and RSTPReid.
   767	Similarly, comparing Abl.13 with Abl.11, we observe a performance increase of 0.65%, 0.47%, 0.17% and 1.30%, 0.80%,
   768	0.60% on CUHK-PEDES and RSTPReid. Furthermore, from
   769	Abl.15 to Abl.28, the performance gain is 0.72%, 0.62%,
   770	
   771	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   772	
   773	5104
   774	
   775	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
   776	
   777	TABLE II
   778	A BLATION A NALYSIS ON THE L OCAL PART N UMBER K ON CUHK-PEDES AND RSTPR EID
   779	
   780	0.45% and 1.10%, 0.70%, 0.60%. These results provide evidence for the effectiveness of employing the ML mechanism
   781	to facilitate knowledge exchange across different branches.
   782	3) Impact of Local Part Number K : The number K of
   783	local parts and the way in which they are partitioned are
   784	also of significance for the proposed model to achieve an
   785	accurate retrieval performance. Therefore, we carry out comprehensive ablation experiments to analyze the impact of the
   786	choice of K and the partitioning paradigm. On either the
   787	CUHK-PEDES or the RSTPReid dataset, there are totally
   788	26 different partitioning paradigms employed with K varies
   789	from 2 to 12 for ablation analysis. The experimental results
   790	are reported in Table II. First, it can be observed from the
   791	results that the horizontal partitioning strategy with K equals
   792	to 6 and 8 are, respectively, the best choice for the CUHKPEDES and RSTPReid datasets. Second, compared with the
   793	horizontal partitioning strategy, partitioning the feature map
   794	vertically can result in an obvious performance drop. This
   795	observation is reasonable according to the characteristics of
   796	person images utilized for text-based person retrieval. As the
   797	input person images are resized as a 384 × 128 rectangle and
   798	commonly most of the images are exactly filled by a certain
   799	pedestrian, the horizontally split stripes are more likely to
   800	contain information for different body parts, while each of
   801	the vertically split ones may just contain partial information
   802	of multiple body parts, and hence leads to the performance gap
   803	between these two strategies. Besides, as shown in Table II,
   804	when K goes relatively large, the retrieval performance also
   805	starts to decrease. This can be attributed to the fact that with
   806	
   807	a relatively large K , each split part is small and thereby some
   808	of these local parts may fail to catch complete discriminative
   809	local clues and may even become noise information. Thus,
   810	it seems that the horizontal partitioning strategy with a proper
   811	local part number K best meets the nature of the text-based
   812	person retrieval task.
   813	4) Impact of Color Prior: Comparing several pairs of
   814	experiments such as (Abl.10, Abl.11), (Abl.12, Abl.13),
   815	(Abl.14, Abl.15) and (Abl.16, Abl.28), it can be seen that
   816	taking the CLR prior information into consideration leads to
   817	a further improvement in the performance. This observation
   818	confirms the effectiveness of the CLR prior information to
   819	enhance the CLR branch.
   820	5) Comparison Between KL Divergence and JS Divergence

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1180,1700p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1180	T
  1181	µT
  1182	F )) +
  1183	
  1184	n,
  1185	
  1186	(1)
  1187	
  1188	where 2 R
  1189	denotes the lower triangular matrix that is
  1190	T
  1191	obtained by the cholesky decomposition
  1192	= N 1 1 (F
  1193	T
  1194	1⇥C
  1195	1µF ) (F 1µF ); µF 2 R
  1196	denotes the mean vector of
  1197	F; 1 2 RN ⇥1 is a column vector of all ones; n 2 R1⇥1 and
  1198	1⇥C
  1199	are the learnable affine parameters. In such a way,
  1200	n 2R
  1201	0
  1202	F0 = [f10 ; f20 ; · · · ; fN
  1203	] 2 RN ⇥C can represent all the whitened
  1204	nodes, where fn 2 R1⇥C is the n-th node in F0 .
  1205	Similarly to [38], we use cross-correlation to learn the
  1206	incidence matrix H 2 RN ⇥M , i.e.,
  1207	C⇥C
  1208	
  1209	H = "( (F0 )⇤(F0 ) (F0 )T ⌦(F0 )),
  1210	
  1211	(2)
  1212	
  1213	where (F0 ) 2 RN ⇥C introduces the learnable parameters
  1214	to perform the linear transformation for the whitened nodes
  1215	for all the whitened nodes F0 . ⌦(F0 ) 2 RN ⇥M gets M
  1216	hyperedges of whitened nodes by trainable parameters for
  1217	
  1218	D1/2 HWB 1 HT D 1/2 )F0 ⇥ + F,
  1219	
  1220	(3)
  1221	
  1222	where I 2 RN ⇥N is the identity matrix; W 2 RM ⇥M denotes
  1223	the weight matrix; D 2 RN ⇥N and B 2 RM ⇥M represent the
  1224	node degree matrix and the hyperedge degree matrix obtained
  1225	by the broadcast operation, respectively; ⇥ 2 RC⇥C denotes
  1226	the learnable parameters. Following the above steps, we feed
  1227	features from Q into the HSL module and obtain a relationvis
  1228	ir
  1229	ir
  1230	enhanced feature set R = {Rvis
  1231	L , RS , RS , RL }, where each
  1232	feature in R is obtained by Eq. (3).
  1233	C. Fine-Coarse Graph Attention Alignment (FCGA) Module
  1234	Some existing feature-level methods [1], [7] try to directly
  1235	reduce the distances between VIS and IR features by the loss
  1236	function, which can not achieve a reliable common feature
  1237	space because of the large modality gap. Later, some methods
  1238	[25], [26], [39] generate the auxiliary features from one or two
  1239	modalities to mitigate the modality discrepancy, but they still
  1240	lack efficient alignment and full utilization between different
  1241	modality features. In order to effectively mitigate modality
  1242	discrepancy, we leverage a fine-coarse graph attention alignment (FCGA) module, which aligns the features from different
  1243	modalities and ranges by combining the fine-grain graph attention alignment (FGA) with the coarse-grain graph attention
  1244	alignment (CGA), so as to generate reliable middle features,
  1245	as shown in Fig. 4. In the FCGA module, the short-range
  1246	features can offer local details to long-range features, making
  1247	pedestrian feature representation more discriminative, while
  1248	the long-range features can provide contextual information
  1249	for short-range features to focus on the global relationship
  1250	between detailed features.
  1251	During the feature alignment, the fine-grain graph attention
  1252	establishes the dense connections between feature nodes that
  1253	can reserve the details of middle features. Besides, the coarsegrain graph attention perceives the semantic associations of
  1254	regional feature nodes to improve the quality of the overall
  1255	middle feature. Specifically, we align each feature with the
  1256	other three features in R and generate a middle feature, which
  1257	involves the information from different modalities and ranges.
  1258	ir
  1259	We take the alignment between two features Rvis
  1260	L and RS as
  1261	an example.
  1262	For the fine-grain graph attention alignment (FGA), first,
  1263	we establish the similarity matrix U 2 RN ⇥N between Rvis
  1264	L
  1265	and Rir
  1266	S by using the inner product and the softmax function,
  1267	which can be formulated as
  1268	ir
  1269	T
  1270	U = Softmax(Rvis
  1271	L ✓q,f (RS ✓k,f ) ),
  1272	
  1273	(4)
  1274	
  1275	IEEE Transactions on Circuits and Systems for Video Technology
  1276	
  1277	Page 6 of 26
  1278	
  1279	PREPRINT SUBMITTED TO TCSVT, JULY 2025
  1280	
  1281	FGA Branch
  1282	a ega a g �
  1283	a �
  1284	a ega a g �
  1285	a �
  1286	a ega a g �
  1287	a �
  1288	
  1289	�
  1290	S tma
  1291	
  1292	⊖ ⨂Re
  1293	
  1294	⨂
  1295	
  1296	⨂
  1297	
  1298	�
  1299	Middle
  1300	feature
  1301	
  1302	�
  1303	CGA Branch
  1304	a ega a g �
  1305	a �
  1306	a ega a g �
  1307	a �
  1308	a ega a g �
  1309	a �
  1310	
  1311	Feature
  1312	Com act
  1313	
  1314	�
  1315	
  1316	�
  1317	
  1318	S tma
  1319	
  1320	.
  1321	
  1322	Reg
  1323	
  1324	.
  1325	
  1326	.
  1327	.
  1328	node node
  1329	
  1330	�
  1331	
  1332	t
  1333	
  1334	.
  1335	
  1336	.
  1337	
  1338	.
  1339	.
  1340	e
  1341	
  1342	.
  1343	.
  1344	
  1345	e
  1346	
  1347	e
  1348	
  1349	.
  1350	
  1351	.
  1352	
  1353	.
  1354	.
  1355	e
  1356	
  1357	a
  1358	onma
  1359	
  1360	at
  1361	
  1362	Feature Compact
  1363	
  1364	�
  1365	
  1366	er Bod
  1367	node
  1368	node
  1369	node
  1370	node
  1371	
  1372	0
  1373	1
  1374	0
  1375	1
  1376	
  1377	node n
  1378	
  1379	0
  1380	
  1381	node
  1382	node
  1383	node
  1384	node
  1385	
  1386	0
  1387	0
  1388	1
  1389	0
  1390	
  1391	a
  1392	
  1393	�
  1394	
  1395	a
  1396	
  1397	�
  1398	
  1399	a
  1400	
  1401	�
  1402	
  1403	a
  1404	
  1405	�
  1406	
  1407	Lower Bod
  1408	
  1409	Head
  1410	
  1411	Foot
  1412	
  1413	eration
  1414	
  1415	a
  1416	
  1417	a
  1418	.
  1419	
  1420	t
  1421	
  1422	A erage
  1423	
  1424	Reg
  1425	
  1426	⨂
  1427	
  1428	⊖ S ta t
  1429	⊕ t
  1430	
  1431	Re
  1432	
  1433	⨂
  1434	
  1435	Reg
  1436	Reg
  1437	
  1438	�
  1439	
  1440	⨂
  1441	
  1442	Feature
  1443	Com act
  1444	
  1445	1
  1446	2
  1447	3
  1448	4
  1449	5
  1450	6
  1451	7
  1452	8
  1453	9
  1454	10
  1455	11
  1456	12
  1457	13
  1458	14
  1459	15
  1460	16
  1461	17
  1462	18
  1463	19
  1464	20
  1465	21
  1466	22
  1467	23
  1468	24
  1469	25
  1470	26
  1471	27
  1472	28
  1473	29
  1474	30
  1475	31
  1476	32
  1477	33
  1478	34
  1479	35
  1480	36
  1481	37
  1482	38
  1483	39
  1484	40
  1485	41
  1486	42
  1487	43
  1488	44
  1489	45
  1490	46
  1491	47
  1492	48
  1493	49
  1494	50
  1495	51
  1496	52
  1497	53
  1498	54
  1499	55
  1500	56
  1501	57
  1502	58
  1503	59
  1504	60
  1505	
  1506	6
  1507	
  1508	Compact
  1509	node set
  1510	
  1511	−
  1512	
  1513	node n 0
  1514	
  1515	C
  1516	
  1517	where ✓q,f 2 RC⇥ r and ✓k,f 2 RC⇥ r are linear transformations; r is a reduction ratio used to perform the squeeze
  1518	and excitation, thereby reducing the number of parameters;
  1519	Softmax(·) denotes the softmax function.
  1520	Then, we adopt the graph attention [37] to perform alignir
  1521	ment between Rvis
  1522	matrix.
  1523	L and RS according to the similarity
  1524	N⇥ C
  1525	r is
  1526	Therefore, the aggregated node R̄vis,L
  1527	ir,S 2 R
  1528	vis
  1529	ir
  1530	R̄vis,L
  1531	ir,S = FGA(RL , RS )
  1532	
  1533	= ReLU(U
  1534	
  1535	where arg max(·) returns the index of the maximum probability value in pn = [pn,1 , pn,2 , · · · , pn,Y ] 2 R1⇥Y , and
  1536	pn,y means the probability value of the n-th feature node
  1537	belonging to the y-th region.
  1538	For the z-th region, we can get the index vector idxz =
  1539	[idx1,z , idx2,z , · · · , idxN,z ], and then the feature node set
  1540	Ovis,L
  1541	with the same z-th region can be obtained as
  1542	z
  1543	Ovis,L
  1544	= Mask(Rvis
  1545	z
  1546	L , idxz ),
  1547	
  1548	Fig. 4. The detailed network architecture of the proposed FCGA module
  1549	ir
  1550	ir
  1551	vis
  1552	(Taking aligning Rvis
  1553	L with RS , RL , and RS as an example).
  1554	C
  1555	
  1556	probability values that the n-th (n 2 {1, 2, · · · , N }) feature
  1557	node respectively belongs to the Y regions.
  1558	Then, we denote idxn,z as the index value of the n-th
  1559	feature node belonging to the z-th (z 2 {1, 2, · · · , Y }) region.
  1560	The idxn,z can be defined as
  1561	8
  1562	< 1, if arg max (pn,y ) = z
  1563	y2{1,2,··· ,Y }
  1564	idxn,z =
  1565	,
  1566	(7)
  1567	: 0, otherwise
  1568	
  1569	Mean(redU)11T )(Rir
  1570	S ✓v,f ),
  1571	
  1572	(5)
  1573	
  1574	]where FGA(·) denotes the fine-grain graph attention alignC
  1575	ment operation; ✓v,f 2 RC⇥ r is the linear transformation;
  1576	is the balancing parameter that reduces nodes with low
  1577	similarity; 11T 2 RN ⇥N is a matrix of all ones; and ReLU(·)
  1578	and Mean(·) represent the ReLU activation function that sets
  1579	similarity values less than 0 to 0 and the mean operation,
  1580	respectively.
  1581	Different from the fine-grain alignment that needs to refine
  1582	the details of each node-level feature, by compacting the
  1583	feature nodes (e.g., classify each node and merge similar
  1584	nodes into a single node), the representation ability of the
  1585	feature nodes can be further enhanced, and the computational
  1586	complexity of the feature alignment process can be reduced.
  1587	So, we introduce a coarse-grain graph attention alignment
  1588	(CGA) to improve the efficiency of generating intermediate
  1589	features and enhance the semantic association of the middle
  1590	features.
  1591	To begin with, we classify each feature node of Rvis
  1592	L into Y
  1593	regions (i.e., head, arm, torso, leg, and so on) by a learnable
  1594	classifier, and achieve the total region probability P2 RN ⇥Y ,
  1595	that is,
  1596	P = Classifier(Rvis
  1597	L ) = [p1 , p2 , · · · , pN ],
  1598	
  1599	(6)
  1600	
  1601	where Classifier(·) is a fully connected layer which consists of
  1602	learnable parameters ✓cls 2 RC⇥Y ; pn 2 R1⇥Y represents the
  1603	
  1604	(8)
  1605	
  1606	where Mask(·) sets the value of the n-th feature node to 0
  1607	when the corresponding idxn,z is 0; otherwise keep the same
  1608	value as before.
  1609	Finally, we perform an average operation on feature nodes
  1610	vis,L
  1611	vis,L
  1612	N ⇥C
  1613	Ovis,L
  1614	= [ovis,L
  1615	to obtain the
  1616	z
  1617	z,1 , oz,2 , · · · , oz,N ] 2 R
  1618	vis,L
  1619	z-th compact representation Ôz
  1620	, i.e.,
  1621	PN
  1622	vis,L
  1623	n=1 oz,n
  1624	Ôvis,L
  1625	=
  1626	,
  1627	(9)
  1628	z
  1629	vis,L
  1630	Nz
  1631	+✏
  1632	where ovis,L
  1633	represents the n-th feature node of Ovis,L
  1634	;
  1635	z,n
  1636	z
  1637	vis,L
  1638	Nz
  1639	means the number of non-zero feature nodes in Ovis,L
  1640	;
  1641	z
  1642	✏ is a very small value (i.e., e 5 ) added to the denominator
  1643	for numerical stability. Similarly to Eq. (9), we can get the
  1644	other Y -1 compact feature nodes and thus the final compact
  1645	Y ⇥C
  1646	feature node set can be represented as Õvis
  1647	.
  1648	L 2R
  1649	Moreover, to improve the efficiency of feature alignment,
  1650	and explore contextual semantic associations in different
  1651	modalities and ranges. Hence, we design a coarse-grain graph
  1652	attention (CGA) to align Õvis
  1653	and Õir
  1654	L
  1655	S , and the aligned
  1656	C
  1657	vis,L
  1658	compact feature node set Ōir,S 2 RN ⇥ r can be formulated
  1659	as
  1660	vis
  1661	ir
  1662	Ōvis,L
  1663	ir,S = CGA(ÕL , ÕS )
  1664	ir
  1665	T
  1666	ir
  1667	= ✓c,u ReLU(Softmax(Õvis
  1668	L ✓q,c (ÕS ✓k,c ) ))(ÕS ✓v,c ),
  1669	(10)
  1670	C
  1671	
  1672	C
  1673	
  1674	C
  1675	
  1676	where ✓q,c 2 RC⇥ r , ✓k,c 2 RC⇥ r , ✓v,c 2 RC⇥ r and
  1677	✓c,u 2 RN ⇥Y are the linear transformations. Unlike FGA,
  1678	the CGA allocates all feature nodes to the limited Y regions,
  1679	without suppressing low similarity feature regions to ensure
  1680	the efficiency of the coarse-grain alignment.
  1681	Based on the above, we employ an effective and efficient
  1682	fine-coarse graph attention alignment (FCGA) module to align
  1683	different modalities and ranges of feature nodes from nodelevel and region-level, respectively, as follows
  1684	ir
  1685	vis
  1686	ir
  1687	FCGA(Rvis
  1688	L , RS ) = FGA(RL , RS )+
  1689	ir
  1690	CGA(Rvis
  1691	L , RS ).
  1692	
  1693	(11)
  1694	
  1695	Page 7 of 26
  1696	
  1697	IEEE Transactions on Circuits and Systems for Video Technology
  1698	
  1699	PREPRINT SUBMITTED TO TCSVT, JULY 2025
  1700	

codex
HOH-Net 的方法比其他几篇更“模块堆叠”，但它的主线是中间特征不是生成物，而是跨模态公共空间的分层代理。我再补完它的 HMAL 和 MRIC，避免把它误解成普通对齐损失。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf' - | nl -ba | sed -n '1,240p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	3160
     2	
     3	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025
     4	
     5	InfinitePerson: Innovating Synthetic Data Creation
     6	for Generalization Person Re-Identification
     7	Guoqing Zhang , Member, IEEE, Jin Li , Yuhui Zheng , Member, IEEE, and Ruili Wang
     8	
     9	Abstract— Recently, large-scale synthetic datasets have
    10	effectively alleviated the issue of insufficient person reidentification (Re-ID) datasets. However, synthetic datasets
    11	grapple with inherent challenges, including the subpar quality
    12	of synthetic pedestrians and single data collection. This paper
    13	presents InfinitePerson, a costless pipeline that fully utilizes the
    14	infinite generation capability of diffusion models to produce
    15	diverse UV texture images and effortlessly constructs high-quality
    16	synthetic datasets by simulating a real surveillance network.
    17	Specifically, we innovatively propose the utilization of diffusion
    18	models to generate high-quality, realistic, and diverse UV
    19	texture images to address the limitations of clothing textures.
    20	This ensures that our 3D character models have complete
    21	clothing texture information and look very similar to real-world
    22	pedestrians. Moreover, in response to the challenges in replicating
    23	synthetic data collection pipelines, we propose a sub-monitoring
    24	network data collection method, which can collect pedestrians
    25	data from different viewpoints, backgrounds, and lighting
    26	conditions through simple scene layout. Finally, a more scalable
    27	and realistic large synthetic dataset called InfinitePerson
    28	is created, containing 4,700 identities and 535,636 images.
    29	Experimental evidence demonstrates show that models trained
    30	on InfinitePerson exhibit superior generalization performance,
    31	surpassing those trained on both popular real-world and
    32	synthetic person Re-ID datasets. The InfinitePerson project is
    33	available at https://github.com/zhguoqing/InfinitePerson.
    34	Index Terms— Generalization person re-identification, synthetic Re-ID dataset, stable diffusion, sub-monitoring network.
    35	
    36	Received 8 June 2024; revised 20 October 2024; accepted 16 November
    37	2024. Date of publication 22 November 2024; date of current version
    38	7 April 2025. This work was supported in part by the National Natural
    39	Science Foundation of China under Grant 62172231, Grant 92470202, and
    40	Grant U20B2065; in part by the Natural Science Foundation of Jiangsu
    41	Province under Grant BK20220107; in part by Wuxi Industrial Innovation Research Institute-Visual Intelligent Analysis of Worker Behavior
    42	and Anomaly Warning; and in part by 2020 Catalyst: Strategic New
    43	Zealand–Singapore Data Science Research Programme Fund by MBIE, New
    44	Zealand. This article was recommended by Associate Editor J.-H. Xue.
    45	(Corresponding author: Ruili Wang.)
    46	Guoqing Zhang is with the School of Computer Science, Nanjing University
    47	of Information Science and Technology, Nanjing 210044, China, and also with
    48	the School of Mathematical and Computational Sciences, Massey University,
    49	Auckland 4442, New Zealand (e-mail: guoqingzhang@nuist.edu.cn).
    50	Jin Li is with the School of Computer Science, Nanjing University
    51	of Information Science and Technology, Nanjing 210044, China (e-mail:
    52	jin_li@nuist.edu.cn).
    53	Yuhui Zheng is with the Key Laboratory of Tibetan Information Processing,
    54	Ministry of Education, Qinghai Normal University, Xining 810008, China
    55	(e-mail: zheng_yuhui@nuist.edu.cn).
    56	Ruili Wang is with the School of Mathematical and Computational Sciences,
    57	Massey University, Auckland 4442, New Zealand, and also with the School of
    58	Computer Science, University of Nottingham China, Ningbo 315104, China
    59	(e-mail: ruili.wang@massey.ac.nz).
    60	Digital Object Identifier 10.1109/TCSVT.2024.3504722
    61	
    62	I. I NTRODUCTION
    63	
    64	P
    65	
    66	ERSON re-identification (Re-ID) has a focal point within
    67	the domain computer vision, attracting significant attention due to its expansive potential application prospects in
    68	intelligent security, video surveillance, and other fields [1], [2].
    69	However, existing Re-ID datasets are insufficient to meet
    70	increasingly diverse practical needs [3], [4], [5], and creating
    71	new real datasets faces various challenges. Firstly, annotating
    72	precise identities in real-life scenarios requires a significant
    73	amount of manpower. Secondly, collecting large amounts of
    74	characters data faces the problem of violating personal privacy. Therefore, synthetic person Re-ID datasets have recently
    75	gained popularity among researchers due to their inherent
    76	advantages, such as the avoidance of privacy issues and lower
    77	annotation costs [6], [7], [8]. However, synthetic datasets still
    78	face some challenges, such as poor quality of 3D character
    79	models, which are insufficient to fully simulate real pedestrians, and the difficulty in reproducing the process of collecting
    80	synthetic data, all of which limit the further development of
    81	synthetic datasets.
    82	In existing synthetic datasets [7], [8], there are significant
    83	differences between the 3D character models used and realworld pedestrians. The main reason is that they all ignore the
    84	texture definition of 3D clothing when generating UV texture
    85	maps, leading to a misalignment between the generated UV
    86	maps with the corresponding 3D clothing models, making
    87	the 3D character that bears a resemblance to cartoon characters, as shown in Figure 1. The significant gap between 3D
    88	characters and real-world pedestrians directly affects the effectiveness of the synthetic dataset and the generalization ability
    89	of models trained on this dataset. Additionally, these synthetic
    90	data collection methods suffer from shortcomings in terms of
    91	replicability and adaptability, hindering the extension of these
    92	datasets to address specific Re-ID scenarios [9]. Therefore,
    93	we create a new reproducible synthetic data collection pipeline
    94	suitable for the ReID.
    95	In this work, we propose a costless pipeline called Infiniteperson to address the above issues. To generate a variety
    96	of high-quality 3D character models, we innovatively use
    97	diffusion models to generate high quality and diverse UV
    98	texture maps and use normal maps dataset as additional input
    99	to train ControlNet [10] to guide the Stable Diffusion [11] in
   100	generating reasonable UV texture maps. Compared to previous
   101	work [7], [8], the UV images we have generated are more comprehensive and better aligned with defined clothing textures.
   102	In addition, we also propose a sub-monitoring network data
   103	
   104	1051-8215 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
   105	See https://www.ieee.org/publications/rights/index.html for more information.
   106	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   107	
   108	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   109	
   110	Fig. 1.
   111	Illustration of selected examples from (a) UnrealPerson and
   112	(b) RandPerson datasets, respectively. (c) the 3D character models with our
   113	generated UV texture maps rendered by UE4.
   114	
   115	collection method, which enables researchers to more easily
   116	design and build complex Re-ID scenes to better simulate
   117	the real world. Our major contributions are summarized as
   118	follows:
   119	• We innovatively use diffusion models to generate UV
   120	texture images to resolve the problem of limited clothing
   121	texture resources in synthetic datasets and enhance both
   122	the diversity and quality of 3D character models.
   123	• We propose a sub-monitoring network data collection
   124	method, which can more accurately simulate the operation of the monitoring network, effectively control data
   125	quality and collection process, and facilitate subsequent
   126	workers to design specific Re-ID scenarios.
   127	• We create a large-scale synthetic person Re-ID dataset
   128	called InfinitePerson, which contains 535,636 images
   129	of 4,700 identities, covering various scenes and lighting conditions, providing researchers with powerful data
   130	resources.
   131	• Experimental results show that models trained on InfnitePerson have better generalization performance than
   132	models trained on other widely used real-world and
   133	synthetic Re-ID datasets.
   134	II. R ELATED W ORK
   135	A. Generating Images for Re-ID
   136	The generation of realistic and sensible textures is a critical
   137	aspect of simulating authentic 3D character models, as textures
   138	carry essential information for describing and identifying
   139	object instances [17]. The early incorporation of texture information into Re-ID tasks was primarily aimed at mitigating
   140	the issue of spatial semantic misalignment. Wang et al. [18]
   141	
   142	3161
   143	
   144	employed the Re-ID model to compute the similarity between
   145	the source image and the synthetic image, aiming to produce high-quality texture images. Zhang et al. [19] used
   146	DensePose [20] to distort the input task image into a standardized UV coordinate system. This approach established
   147	a dense correspondence between 2D pedestrian images and
   148	standardized human body representations based on 3D surface
   149	space, effectively resolving the problem of semantic alignment
   150	and misalignment in Re-ID task images. Jin et al. [21] used
   151	trained networks to produce texture images of each pedestrian
   152	in the Re-ID dataset. Since the pedestrian surface in the generated texture image is semantically aligned and encompasses all
   153	three-dimensional surfaces of pedestrians, it could effectively
   154	guide the Re-ID network to learn semantically aligned feature
   155	representations and gain a more comprehensive understanding
   156	of pedestrian features. Therefore, we believe that 3D character
   157	models with high-quality and authentic clothing textures can
   158	benefit the Re-ID network.
   159	With the remarkable advancements in diffusion models for
   160	image generation [10], [11], some recent works had made it
   161	possible to generate realistic and diverse images based on
   162	text prompts [22], [23]. Bhunia et al. [24] adopted diffusion
   163	models to generate realistic character images based on pose
   164	information, and then added these generated images to Market1501 [4], which enriches the diversity of characters in the
   165	dataset to a certain extent. However, these generated character images often lack background information and label
   166	information, which limits their applicability in Re-ID tasks.
   167	In addition to generating images, OWD [25] also provided
   168	diverse real-world data, but collecting and expanding this
   169	data is still limited by time, resources, and privacy issues.
   170	Therefore, the flexibility of synthetic data can help researchers
   171	generate large-scale datasets in special application scenarios,
   172	which is difficult to achieve with real-world datasets such as
   173	OWD [25].
   174	B. Learning From Synthetic Dataset
   175	As deep learning progresses within the realm of person
   176	Re-ID [26], [27], [28], [29], [30] the demand for large-scale
   177	and diverse datasets continues to increase. Compared to the
   178	time-consuming and labor-intensive process of collecting and
   179	manually annotating Re-ID pedestrian images in the real
   180	world, synthetic data can minimize the expense of manual
   181	annotation and avoid security and privacy issues. In recent
   182	years, game engines have been able to effectively simulate real-life scenes, and even convert the environmental
   183	factors of the scene into controllable parameters (such as
   184	lighting and perspective) with their excellent rendering techniques and diverse scene design capabilities. For example,
   185	VC-Clothes [31] and SynPerson [6] are synthetic person Re-ID
   186	datasets collected on the GTA5 game engine. SynPerson [6]
   187	studied the impact of lighting on Re-ID systems by varying
   188	weather conditions and lighting, and VC-Clothes [31] is a
   189	dataset under clothing changes. In real scenarios, associating
   190	the same identity with clothing changes and labeling them
   191	correctly is quite challenging for manual labeling.
   192	In recent years, various synthetic datasets have been continuously proposed. SOMAset [12] is a synthetic dataset
   193	
   194	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   195	
   196	3162
   197	
   198	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025
   199	
   200	TABLE I
   201	D ETAILED C OMPARISONS OF S OME S YNTHETIC P ERSON R E -ID DATASETS . “#I LLUM ” R EFERS TO W HETHER THE DATASET H AS L IGHT C HANGES .
   202	“S CALABLE ” R EFERS TO W HETHER THE DATASET I S S CALABLE . “M ONITORING N ETWORK ”: R EFERS TO W HETHER A M ONITORING N ETWORK
   203	I S S IMULATED D URING THE DATA C OLLECTION P ROCESS . “R ELIABLE T EXTURE ”: R EFERS TO W HETHER THE 3D C HARACTER M ODELS
   204	U SED IN THE S YNTHETIC DATASET H AVE C OMPLETE AND R ELIABLE C LOTHING T EXTURES
   205	
   206	comprising instances generated using photo-realistic human
   207	body simulation software. SyRI [13] used 100 virtual humans
   208	illuminated with multiple HDR environment maps. Both
   209	SOMAset [12] and SyRI [13] are relatively small and
   210	have limited diversity in background and human appearance.
   211	PersonX [14] is a large-scale synthetic dataset including 1,266
   212	human models. However, due to the static nature of the
   213	characters, this dataset lacks scalability. UnrealPerson [8] is
   214	another large synthetic dataset that significantly enhances the
   215	accuracy and diversity of the dataset by using Unreal Engine
   216	(UE4) to render four large realistic scenes and randomly
   217	generate a variety of 3D character models. However, the
   218	clothing textures of the 3D characters in this dataset are
   219	directly replaced by cropping clothing image blocks from
   220	a real clothing dataset [32], [33], resulting in significant
   221	difference between the 3D character models and real-world
   222	pedestrians. SynPerson [6] and FineGPR [16] are primarily
   223	synthetic datasets collected to represent various lighting and
   224	weather conditions, but they did not simulate pedestrian movement. The detailed introduction of some synthetic datasets is
   225	presented in Table I.
   226	III. T HE I NFINITE P ERSON P IPELINE
   227	A. Problem: Person Re-Identification
   228	N
   229	Given an annotated image dataset S = {xi , yi , ci }i=1
   230	, where
   231	xi represents the i-th image, yi and ci represent the identity
   232	label and camera label of image xi . The goal of the Re-ID
   233	task is to learn a suitable network model to map images
   234	to the feature space H, H = {hi |hi = f (δ; xi ) , 1 ≤ i ≤ N },
   235	where f (δ; xi ) is the feature extracted by the network model.
   236	A direct method to reducing the distance between instances
   237	of the same identity compared to different identities is to
   238	minimize the error in predicting identities within the dataset S:
   239	
   240	

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf' - | nl -ba | sed -n '1,220p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1700,2300p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1700	
  1701	1
  1702	2
  1703	3
  1704	4
  1705	5
  1706	6
  1707	7
  1708	8
  1709	9
  1710	10
  1711	11
  1712	12
  1713	13
  1714	14
  1715	15
  1716	16
  1717	17
  1718	18
  1719	19
  1720	20
  1721	21
  1722	22
  1723	23
  1724	24
  1725	25
  1726	26
  1727	27
  1728	28
  1729	29
  1730	30
  1731	31
  1732	32
  1733	33
  1734	34
  1735	35
  1736	36
  1737	37
  1738	38
  1739	39
  1740	40
  1741	41
  1742	42
  1743	43
  1744	44
  1745	45
  1746	46
  1747	47
  1748	48
  1749	49
  1750	50
  1751	51
  1752	52
  1753	53
  1754	54
  1755	55
  1756	56
  1757	57
  1758	58
  1759	59
  1760	60
  1761	
  1762	7
  1763	
  1764	Through the FCGA module with a small number of paN ⇥C
  1765	rameters, a middle feature Mvis
  1766	can be obtained
  1767	L 2 R
  1768	ir
  1769	by aligning Rvis
  1770	with
  1771	the
  1772	other
  1773	three
  1774	features Rir
  1775	L
  1776	S , RL , and
  1777	Rvis
  1778	in
  1779	R,
  1780	that
  1781	is,
  1782	S
  1783	vis
  1784	ir
  1785	vis
  1786	ir
  1787	Mvis
  1788	L = (FCGA(RL , RS ) + FCGA(RL , RL )+
  1789	vis
  1790	vis
  1791	FCGA(Rvis
  1792	L , RS ))✓u + RL ,
  1793	
  1794	(12)
  1795	
  1796	C
  1797	
  1798	where ✓u 2 R r ⇥C represents the linear transformation.
  1799	We can get the other reliable middle features similar to
  1800	Eq. (12). Hence, we obtain the middle feature set R̃ =
  1801	vis
  1802	ir
  1803	ir
  1804	{Mvis
  1805	L , MS , ML , MS }.
  1806	D. Hierarchical Middle-Feature Agent Learning (HMAL) Loss
  1807	We generate agents of middle features at each modalityshared feature extraction stage to assist the network in learning
  1808	a better common feature space and reducing the modality discrepancy hierarchically. For example, we generate the middlefeature agent A1 2 R1⇥1⇥C1 for Stage 1 of the network based
  1809	on the above middle feature set R̃, that is,
  1810	A1 = ✓m,1 (Avg(R̃)),
  1811	
  1812	(13)
  1813	
  1814	where Avg(·) is the average pooling operation; ✓m,1 is the
  1815	learnable parameter that adjusts the feature size to 1 ⇥ 1 ⇥ C1
  1816	(C1 means the number of channels of the features in Stage 1).
  1817	Then, we introduce the modality discrepancy reduction
  1818	(MDR) loss to reduce the difference between the pooled VIS
  1819	feature (F̃1,vis ), IR feature (F̃1,ir ) and middle-feature agent
  1820	(A1 ) in Stage 1 of the network, as follows
  1821	1 $A1
  1822	LF̃
  1823	M DR = L1 (F̃1,vis + F̃1,ir , 2A1 ) + L1 (F̃1,vis , F̃1,ir ).
  1824	(14)
  1825	
  1826	where L1 (·) represents the L1 distance; F̃1,vis 2 R
  1827	and F̃1,ir 2 R1⇥1⇥C1 mean the VIS feature F1,vis and IR
  1828	feature F1,ir after the pooling operation, respectively.
  1829	Similarly to Eq. (13), we can achieve other middle-feature
  1830	agents (i.e., A2 , A3 , and A4 ), and the total MDR loss can be
  1831	expressed as:
  1832	1⇥1⇥C1
  1833	
  1834	F̃1 $A1
  1835	F̃2 $A2
  1836	F̃3 $A3
  1837	F̃4 $A4
  1838	DR
  1839	LM
  1840	HM AL = LM DR + LM DR + LM DR + LM DR .
  1841	(15)
  1842	
  1843	With the agent of the middle features, our method can
  1844	learn discriminative features from all the network stages and
  1845	hierarchically enhance feature representations to achieve a
  1846	reliable common feature space between different modalities.
  1847	The existing VI-ReID methods [1], [40] follow low-to-high
  1848	feature extraction, which ignores the interaction of features at
  1849	different stages. The features of the low stage contain more
  1850	detailed information, while the features of the high stage have
  1851	rich semantic relationships. In this subsection, we will use
  1852	the BFE loss to build the mutual interaction bridge and thus
  1853	achieve the bi-directional enhancement between the features
  1854	of different stages. The interactions of the high-to-low and the
  1855	low-to-high can improve the ability of the network to capture
  1856	discriminative features. In other words, the low-stage network
  1857	can focus on detailed features guided by semantic relationships
  1858	from the high-stage, and the high-stage network can enhance
  1859	
  1860	the semantic relationship by using detailed information from
  1861	the low-stage.
  1862	We take the bi-directional enhancement between the features
  1863	of Stage 1 (S1 ) and Stage 2 (S2 ) as an example. First, we
  1864	adopt the pooling operation on the features of S1 and S2
  1865	to obtain the feature representations F̃1 2 R1⇥1⇥C1 and
  1866	F̃2 2 R1⇥1⇥C2 (C2 is the number of channels of the features
  1867	in S2 ), that is, F̃1 = Avg(F1 ) and F̃2 = Avg(F2 ). Here
  1868	Avg(·) is the average pooling operation; F1 2 RH1 ⇥W1 ⇥C1
  1869	and F2 2 RH2 ⇥W2 ⇥C2 are the features of S1 and S2 ,
  1870	respectively (H1 /H2 and W1 /W2 correspond to the height and
  1871	width of the features in S1 /S2 , respectively).
  1872	Then, we perform the upsample and downsample operations
  1873	1⇥1⇥C2
  1874	on the features F̃1 and F̃2 to achieve F̃up
  1875	and
  1876	1 2 R
  1877	1⇥1⇥C1
  1878	F̃down
  1879	2
  1880	R
  1881	,
  1882	respectively,
  1883	which
  1884	can
  1885	be
  1886	formulated
  1887	as
  1888	2
  1889	F̃up
  1890	1 = Upsample(F̃1 );
  1891	
  1892	F̃down
  1893	= Downsample(F̃2 ), (16)
  1894	2
  1895	
  1896	where the Upsample(·) and Downsample(·) operations make
  1897	down
  1898	F̃up
  1899	become the same size as F2 and F1 by linear
  1900	1 and F̃2
  1901	transformations.
  1902	To provide detailed information from low-stage features to
  1903	high-stage features and transfer high-stage semantic information to low-stage features, we adopt the L1 distance to perform
  1904	the bi-directional interaction between the features of S1 and
  1905	S2 , which is defined as
  1906	up
  1907	down
  1908	1 $S2
  1909	LSBF
  1910	, F̃1 ).
  1911	E = L1 (F̃1 , F̃2 ) + L1 (F̃2
  1912	
  1913	(17)
  1914	
  1915	We also perform bi-directional enhancement between S2
  1916	and S3 and between S3 and S4 , and thus the final BFE loss
  1917	can be written as
  1918	S1 $S2
  1919	S2 $S3
  1920	S3 $S4
  1921	E
  1922	LBF
  1923	HM AL = LBF E + LBF E + LBF E .
  1924	
  1925	(18)
  1926	
  1927	The final HMAL loss, which is defined as
  1928	DR
  1929	BF E
  1930	LHM AL = LM
  1931	HM AL + LHM AL .
  1932	
  1933	(19)
  1934	
  1935	E. Modality-Range Identity-Center Contrastive (MRIC) Loss
  1936	To reduce the intra-class difference and increase inter-class
  1937	discrepancy, we introduce the MRIC loss to improve feature
  1938	representations and minimize the modality gaps among the
  1939	VIS, IR, and middle features. The MRIC loss consists of three
  1940	items: an intra-range loss, a middle feature loss, and an intermodality loss based on identity centers. The illustration of the
  1941	MRIC loss is presented in Fig. 5.
  1942	Following previous works [1], [7], we apply the holistic
  1943	and partial generalized mean pooling to each feature in R̃
  1944	and concatenate the pooling features to obtain the 1D middle
  1945	features, and we can get the 1D middle feature set R̃0 =
  1946	vis
  1947	ir
  1948	ir
  1949	{mvis
  1950	L , mS , mL , mS }. Analogously, we apply the same
  1951	pooling and concatenation operations to each feature in R
  1952	vis ir ir
  1953	and thus obtain the 1D feature set R0 = {rvis
  1954	L , rS , rL , rS }.
  1955	The robustness of the identity centers ensures they are not
  1956	influenced by pedestrian appearance changes. Technically, we
  1957	first obtain identity centers through the weighted average of the
  1958	features of each person at the specific modality and range. For
  1959	example, the center of the relation-enhanced features for the
  1960	
  1961	IEEE Transactions on Circuits and Systems for Video Technology
  1962	
  1963	Page 8 of 26
  1964	
  1965	PREPRINT SUBMITTED TO TCSVT, JULY 2025
  1966	
  1967	1
  1968	2
  1969	3
  1970	4
  1971	5
  1972	6
  1973	7
  1974	8
  1975	9
  1976	10
  1977	11
  1978	12
  1979	13
  1980	14
  1981	15
  1982	16
  1983	17
  1984	18
  1985	19
  1986	20
  1987	21
  1988	22
  1989	23
  1990	24
  1991	25
  1992	26
  1993	27
  1994	28
  1995	29
  1996	30
  1997	31
  1998	32
  1999	33
  2000	34
  2001	35
  2002	36
  2003	37
  2004	38
  2005	39
  2006	40
  2007	41
  2008	42
  2009	43
  2010	44
  2011	45
  2012	46
  2013	47
  2014	48
  2015	49
  2016	50
  2017	51
  2018	52
  2019	53
  2020	54
  2021	55
  2022	56
  2023	57
  2024	58
  2025	59
  2026	60
  2027	
  2028	Intra-range distance
  2029	
  2030	8
  2031	
  2032	The inter-modality loss LVMIRIC is leveraged to mitigate the
  2033	intra-class discrepancy and enlarge the inter-class distances
  2034	between VIS, IR, and middle features, which is formulated as
  2035	
  2036	Inter-modality distance
  2037	
  2038	Middle-feature distance
  2039	
  2040	vis
  2041	
  2042	ir
  2043	
  2044	vis
  2045	
  2046	mid
  2047	
  2048	C
  2049	,C
  2050	C
  2051	,C
  2052	LVMIM
  2053	RIC = LM RIC + LM RIC
  2054	
  2055	Short-/long range VIS feature centers
  2056	
  2057	Push
  2058	
  2059	Middle feature centers
  2060	
  2061	Short-/long range IR feature centers
  2062	
  2063	Pull
  2064	
  2065	Modality representations
  2066	
  2067	Fig. 5. Illustration of the proposed MRIC loss. Different colors represent
  2068	different identities.
  2069	
  2070	pedestrian with the identity i at the long-range VIS modality
  2071	can be achieved as
  2072	PK
  2073	K
  2074	vis T
  2075	X
  2076	exp( k=1 rvis
  2077	L,i,j rL,i,k )
  2078	vis
  2079	cvis
  2080	=
  2081	(20)
  2082	PK
  2083	PK vis vis T rL,i,j ,
  2084	L,i
  2085	j=1
  2086	j=1 exp(
  2087	k=1 rL,i,j rL,i,k )
  2088	
  2089	where K represents the number of VIS features of each
  2090	1⇥C 0
  2091	person; rvis
  2092	denotes the k-th 1D relation-enhanced
  2093	L,i,k 2 R
  2094	long-range VIS feature with the identity i in R0 .
  2095	Accordingly, we can obtain the identity center sets
  2096	P
  2097	P
  2098	P
  2099	CLvis ({cvis
  2100	CSvis ({cvis
  2101	CLir ({cir
  2102	L,i }i=1 ),
  2103	S,i }i=1 ),
  2104	L,i }i=1 ),
  2105	ir
  2106	ir P
  2107	vis
  2108	vis P
  2109	vis
  2110	vis P
  2111	˜
  2112	˜
  2113	CS ({cS,i }i=1 ), CL ({c̃L,i }i=1 ), CS ({c̃S,i }i=1 ), C˜Lir
  2114	P
  2115	ir P
  2116	˜ir
  2117	˜
  2118	({c̃ir
  2119	L,i }i=1 ), and CS ({c̃S,i }i=1 ), where C and C represent
  2120	the center sets for the enhanced features and the middle
  2121	features at a specific range and modality, respectively; P is
  2122	the number of pedestrian identities in the training set.
  2123	The intra-range loss LSL
  2124	M RIC is to reduce the distances
  2125	between the same-range VIS and IR features from the same
  2126	pedestrian while enlarging the distances between the samerange VIS and IR features from different pedestrian, that is,
  2127	C vis ,C ir
  2128	
  2129	C vis ,C ir
  2130	
  2131	S
  2132	S
  2133	L
  2134	L
  2135	LSL
  2136	M RIC = LM RIC + LM RIC ,
  2137	
  2138	(21)
  2139	
  2140	where
  2141	exp(MA,B
  2142	i,i )
  2143	log PP
  2144	A,B
  2145	z=1 exp(Mi,z )
  2146	i=1
  2147	
  2148	P
  2149	X
  2150	
  2151	LA,B
  2152	M RIC =
  2153	
  2154	exp(MA,B
  2155	i,i )
  2156	log PP
  2157	A,B
  2158	z=1 exp(Mz,i )
  2159	i=1
  2160	
  2161	P
  2162	X
  2163	
  2164	+
  2165	
  2166	P
  2167	X
  2168	
  2169	L1 (ai
  2170	
  2171	(22)
  2172	
  2173	bi ).
  2174	
  2175	i=1
  2176	
  2177	Here, MA,B 2 RP ⇥P denotes the cosine similarity matrix between A and B. MA,B
  2178	i,j denotes the cosine similarity between
  2179	the i-th row (ai ) of matrix A and the j-th row (bj ) of matrix
  2180	B; L1 (·) represents the L1 norm. By minimizing the LA,B
  2181	M RIC ,
  2182	we can effectively decrease and increase the distance between
  2183	the same pedestrian and different pedestrians in the feature
  2184	space, respectively.
  2185	ID
  2186	The middle-feature loss LM
  2187	M RIC reduces the distances between different middle features, which is defined as
  2188	vis
  2189	vis
  2190	C̃S
  2191	,C̃L
  2192	
  2193	ID
  2194	LM
  2195	M RIC = LM RIC
  2196	
  2197	C̃ vis ,C̃ ir
  2198	
  2199	vis
  2200	ir
  2201	C̃S
  2202	,C̃S
  2203	
  2204	vis
  2205	ir
  2206	C̃S
  2207	,C̃L
  2208	
  2209	+ LM RIC + LM RIC +
  2210	C̃ vis ,C̃ ir
  2211	
  2212	C̃ ir ,C̃ ir
  2213	
  2214	L
  2215	LMLRICS + LMLRICL + LMSRIC
  2216	.
  2217	
  2218	(23)
  2219	
  2220	ir
  2221	
  2222	mid
  2223	
  2224	,C
  2225	+ LCM RIC
  2226	,
  2227	
  2228	(24)
  2229	
  2230	where C vis , C ir , and C mid represent the identity center sets
  2231	of VIS, IR, and middle features, respectively; C vis and C ir
  2232	denote the averaged features from the same modality for each
  2233	person; C mid is obtained by averaging all the middle features
  2234	for each person. Thus, the MRIC loss is
  2235	M ID
  2236	V IM
  2237	LM RIC = LSL
  2238	M RIC + LM RIC + LM RIC .
  2239	
  2240	(25)
  2241	
  2242	Finally, we adopt the cross-entropy loss (LCE [41]), the
  2243	triplet loss (LT RI [42]), the HMAL loss (LHM AL ), and the
  2244	MRIC loss (LM RIC ) to jointly train the HOH-Net. The joint
  2245	loss L is defined as
  2246	L = LCE + LT RI + LHM AL + LM RIC .
  2247	
  2248	(26)
  2249	
  2250	IV. E XPERIMENTS
  2251	A. Experimental Settings
  2252	Image-based Datasets. The SYSU-MM01 [22] dataset contains 491 identities. Its training set includes 395 identities with
  2253	22,258 VIS and 11,909 IR images, while the test set has 96
  2254	identities with 301 VIS and 3,803 IR images. The RegDB [43]
  2255	dataset consists of 412 identities, each with 10 VIS and 10
  2256	IR images captured by two overlapping cameras. The LLCM
  2257	[25] dataset provides 713 identities in the training set and 351
  2258	identities in the test set.
  2259	Video-based Dataset. The HITSZ-VCM dataset [44] is captured by 12 RGB and 12 IR cameras. Its training set includes
  2260	500 identities with 11,061 tracklets, while the test set contains
  2261	427 identities with 10,802 tracklets.
  2262	Implementation Details. During the training phase, all images
  2263	are resized to 3 ⇥ 288 ⇥ 144 with data augmentation [45]. For
  2264	each mini-batch, we randomly select 8 identities with 4 VIS
  2265	images and 4 IR images for each identity. We adopt AGW
  2266	[1] as our backbone network. The learning rate is warmed
  2267	up from 0.01 to 0.1 over the first 10 epochs, then decays to
  2268	0.01 at epoch 20 and 0.001 at epoch 50. We use SGD as the
  2269	optimizer with a momentum parameter set to 0.9. The number
  2270	of hyperedges M in the HSL module is set to 256. In the
  2271	FCGA module, the reduction ratio r is set to 32. in Eq. (5)
  2272	is set to 1.3, 1.1, 1.3, and 1.3 on the SYSU-MM01, RegDB,
  2273	LLCM, and HITSZ-VCM datasets, respectively. In the FCGA
  2274	module, the number Y of person regions is set to 9, 8, 9, and
  2275	9 on the SYSU-MM01, RegDB, LLCM, and HITSZ-VCM
  2276	datasets, respectively. For HITSZ-VCM, each video sequence
  2277	consists of 14 frames, averaged as the video representation. We
  2278	train the HOH-Net for 120 epochs. The proposed HOH-Net is
  2279	implemented in PyTorch on an NVIDIA A40 GPU.
  2280	B. Comparison with State-of-the-Art Methods
  2281	Our proposed HOH-Net is compared with some SOTA
  2282	image-based and video-based models, including LbA [39],
  2283	TSME [5], SPOT [20], DFLN-ViT [10], PMT [8], CAL [46],
  2284	
  2285	
 succeeded in 0ms:
     1	Pattern Recognition 165 (2025) 111561
     2	
     3	Contents lists available at ScienceDirect
     4	
     5	Pattern Recognition
     6	journal homepage: www.elsevier.com/locate/pr
     7	
     8	Identity-aware infrared person image generation and re-identification via
     9	controllable diffusion model
    10	Xizhuo Yu a , Chaojie Fan b ,∗, Zhizhong Zhang c , Yongbo Wang c , Chunyang Chen a ,
    11	Tianjian Yu a , Yong Peng b
    12	a
    13	
    14	School of Traffic & Transportation Engineering, Central South University, Changsha, 410075, Hunan, China
    15	
    16	b Key Laboratory of Traffic Safety on Track of Ministry of Education, School of Traffic & Transportation Engineering, Central South
    17	
    18	University, Changsha, 410075, Hunan, China
    19	c School of Computer Science and Technology, East China Normal University, Shanghai, 200062, China
    20	
    21	ARTICLE
    22	
    23	INFO
    24	
    25	Keywords:
    26	Cross-modality person re-identification
    27	Image generation
    28	
    29	ABSTRACT
    30	Visible–infrared person re-identification (VI-ReID) aims to learn the identity-aware features between visible and
    31	infrared person images. However, most works rely on two publicly available datasets, 𝑖.𝑒., SYSU-MM01 and
    32	RegDB, which is limited by the limited amount of training data and the lack of rich scenes and perspectives.
    33	In this paper, we propose a controllable diffusion framework for infrared person image generation and reidentification. Our approach is beyond the existing diffusion model in two perspectives: (1) we use LoRA
    34	to fine-tune the existing diffusion models with VI-ReID dataset and therefore it helps the diffusion model
    35	understand the infrared modality. A text adapter is then utilized to transfer the semantic understanding ability
    36	of Large Language Model (LLMs) to our generation models; (2) we design a controllable generation module
    37	to make the generated person images, from the same textual description, identity-aware. After meticulous
    38	post-processing operations, our approach is capable of producing diverse visible and infrared person images,
    39	allowing for improving the discrimination of existing VI-ReID model without any annotations. We expand
    40	the VI-ReID dataset with our generated images, and conduct extensive experiments on VI-ReID models.
    41	Experimental results demonstrate the effectiveness of our method.
    42	
    43	1. Introduction
    44	The proliferation of surveillance cameras in public spaces, transportation hubs, and commercial establishments has underscored the
    45	need for efficient and accurate methods to track individuals across
    46	multiple camera feeds. In this context, person re-identification (ReID)
    47	has emerged to identify specific persons in different cameras and
    48	aroused extensive research attentions. However, most existing ReID
    49	methods focus on the visible environment. Despite excellent results,
    50	the limitations of visible cameras under low-light conditions restrict
    51	their applications in complex scenarios such as nighttime and adverse
    52	weather conditions. Therefore, visible–infrared person re-identification
    53	(VI-ReID) is introduced. VI-ReID conduct the person image matching
    54	between visible cameras and infrared cameras, providing the possibility
    55	for achieving 24-h surveillance in video monitoring systems.
    56	Due to the substantial cross modal image disparities, learning
    57	identity-aware features between visible and infrared images is more
    58	challenging than traditional ReID task. To deal with this issue, recent
    59	
    60	works resort to cross-modal feature alignment, which learns modalagnostic features in a common embedding space. However, this
    61	paradigm requires huge data and annotation efforts, while current
    62	researches mostly rely on two publicly available datasets, 𝑖.𝑒., SYSUMM01 [1] and RegDB [2]. For example, the largest SYSU-MM01 dataset
    63	only includes 22,258 visible images and 11,909 infrared images. The
    64	training data is limited, with most images originating from similar
    65	perspective, which deviates from real-world practical applications.
    66	Furthermore, due to the privacy issues and the government of AI
    67	algorithm, it is challenging to manually collect a sufficiently large and
    68	diverse cross-modal person dataset.
    69	With the recent success of AIGC, it is natural to consider that using
    70	the diffusion model to generate the sufficient training samples. But as
    71	shown in Fig. 1(a), it appears that existing pre-trained diffusion model,
    72	e.g., Stable Diffusion, which may learned from a plenty of colored images, cannot understand the infrared modality and therefore even given
    73	the prompt ‘‘an infrared photo’’, it still generates the colored images.
    74	Another nonnegligible issue is the diffusion process is uncontrollable
    75	
    76	∗ Corresponding author.
    77	
    78	E-mail address: fcjgszx@csu.edu.cn (C. Fan).
    79	https://doi.org/10.1016/j.patcog.2025.111561
    80	Received 8 August 2024; Received in revised form 20 February 2025; Accepted 4 March 2025
    81	Available online 14 March 2025
    82	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    83	
    84	Pattern Recognition 165 (2025) 111561
    85	
    86	X. Yu et al.
    87	
    88	Fig. 1. Illustration of our proposed method. (a) shows that our method performs better in semantic understanding than Stable Diffusion v1.5 and is capable of generating infrared
    89	images. (b) exhibits controllable generation results for VI-ReID with our method.
    90	
    91	such that it is unable to generate the identity-aware infrared person
    92	images.
    93	In this paper, we propose a controllable diffusion framework for
    94	infrared person image generation and re-identification. In our approach, we employ commonly used and pre-trained Stable Diffusion
    95	model to generate both visible and infrared person images with text
    96	descriptions. Towards this goal, we first use LoRA [3] to fine-tune the
    97	existing diffusion models with current VI-ReID dataset, which helps
    98	the model learn the concept of ‘‘infrared images’’. After that, we
    99	propose a text adapter to transfer the semantic understanding ability
   100	of LLMs to the fine-tuned diffusion models, and allows us to describe
   101	the person images with satisfactory text prompts. Then, we design a
   102	controllable generation module to ensure that generated person images
   103	from the same textual description are identity-aware, such that it
   104	enables us to learn a discriminative VI-ReID model. After meticulous
   105	post-processing operations, our image generation model is capable of
   106	producing diverse visible and infrared person images based on textual
   107	descriptions. Finally, we expand the existing real-world dataset with the
   108	generated dataset and conduct training and testing on VI-ReID model.
   109	Experimental results validate the effectiveness of our approach.
   110	The contributions of our approach are summarized as follows:
   111	
   112	presents detailed description. Experimental analysis and completion
   113	results are shown in Section 4 to verify our approach. Finally, we
   114	conclude the proposed method in Section 5.
   115	2. Related work
   116	2.1. Traditional person ReID
   117	Traditional person re-identification [4,5] aims to match person
   118	images captured by different visible cameras, whose challenges mainly
   119	lie in the changes in perspective and pedestrian posture. Previous
   120	supervised traditional person ReID works concentrate on feature representation learning [5] and distance metric learning [6]. He et al. [7]
   121	introduce TransReID, a framework completely based on transformer,
   122	which offers enhanced representation capabilities while demanding
   123	less computational resources compared to methods relying on CNNs.
   124	Considering the complexity of manually annotating data, unsupervised
   125	approaches have gained popularity. Several works [8,9] resort to Unsupervised Domain Adaptation (UDA) and utilize transfer learning to
   126	enhance the robustness of the model. The other approaches [10,11] use
   127	Unsupervised Self-Learning (USL), using clustering algorithms to assign
   128	pseudo labels for learning identity features. Dai et al. [12] introduce the
   129	Cluster Contrast for unsupervised ReID, and use momentum updating
   130	to strengthen the consistency of cluster features.
   131	
   132	• We propose a new cross-modal person image generation framework, which is capable of producing diverse visible and infrared
   133	person images with the help of fine-tuned Stable Diffusion model.
   134	• We design a controllable module to ensure that the generated
   135	person images is identity-aware, and therefore allowing for improving the discrimination of existing VI-ReID model.
   136	• We construct a new VI-ReID dataset, called GCP, using text-toimage generation instead of real camera capture, thereby diminishing privacy concerns.
   137	• We conduct extensive experiments with the SYSU-MM01 and
   138	RegDB datasets, which demonstrate the effectiveness of our
   139	method.
   140	
   141	2.2. Visible–infrared person ReID
   142	Visible–infrared person re-identification [13,14] has emerged as a
   143	popular research area due to its ability to remain effective in lowlight environments. To overcome the great contrast between RGB and
   144	infrared images, many works [15,16] utilize image translating to transform one modality to the other. Different from the image-level methods,
   145	several other approaches [17,18] employ the feature-level methods
   146	and align cross-modal features in a shared feature space. To discover
   147	more discriminative features that are shared across modalities, Feng
   148	et al. [19] attempt to extract additional varied modality-shared information by eliminating body shape-related semantic content from
   149	
   150	The rest of this paper is organized as follows. Section 2 briefly
   151	introduces related works and analyzes the difference between ours
   152	and prior works. Section 3 gives the general idea of our method and
   153	2
   154	
   155	Pattern Recognition 165 (2025) 111561
   156	
   157	X. Yu et al.
   158	
   159	the learned features. Recently, to alleviate the complexity and difficulties associated with manual annotation, several unsupervised VI-ReID
   160	works have made great progress. OTLA [20] tries to assign the infrared images to the generated visible pseudo classes by employing
   161	optimal transport methods. CCLNet [21] utilizes the text information from CLIP [22] to improve subsequent unsupervised training.
   162	To overcome camera variation and modality discrepancy, GUR [23]
   163	leverages a bottom-up domain learning strategy, further narrowing the
   164	gap between supervised and unsupervised VI-ReID.
   165	
   166	3.2. Diffusion model fine-tuning via infrared images
   167	Using existing diffusion models in generating the cross-modal person photos may fail because of their limitations in generating infrared
   168	photos and high costs for training. To remedy this issue, we use LoRA,
   169	a parameter-efficient fine-tuning method, to fine-tune the pre-trained
   170	Stable Diffusion. In our approach, We leverage the real-world VI-ReID
   171	dataset SYSU-MM01 as image training dataset, and get corresponding
   172	text training dataset by BLIP. As a result, the fine-tuned diffusion model
   173	can generate high-quality visible and infrared person photos based on
   174	appropriate text prompts.
   175	As shown in Fig. 2, the fine-tuned diffusion model is comprised of
   176	a pretrained autoencoder, a text conditional encoder, and an image
   177	generator. In it, the autoencoder, comprised of an encoder  and a
   178	decoder , is used to realize the conversion between image space and
   179	latent space and lower the computational complexity. For an image
   180	𝑥 ∈ R𝐻×𝑊 ×3 , the encoder  encodes 𝑥 and gets 𝑧 = (𝑥) as its latent
   181	representation, and the decoder  reconstruct the latent representation
   182	and gets 𝑥̃ = (𝑧) = ((𝑥)). Besides, latent diffusion model also introduces a domain specific encoder 𝜏𝜃 , e.g., the text conditional encoder,
   183	to control the synthesis process. On this basis, we choose pretrained
   184	CLIP ViT-L/14 as our text encoder to encode the input texts that are
   185	generated from BLIP. In particular, for a generated text description,
   186	the text encoder encodes it to text embedding, which is then mapped
   187	to the intermediate layers of the UNet, i.e., the backbone of our image
   188	generator.
   189	The time-conditional UNet is utilized in our image generator as
   190	denoising autoencoders 𝜖𝜃 (𝑥𝑡 , 𝑡), 𝑡 = 1, … , 𝑇 , where 𝑥𝑡 is a noisy version
   191	of the input 𝑥. In UNet, the cross-attention mechanism can establish an effective association between the image–text pair. It helps
   192	the model better understand the text information, so as to realize
   193	the transformation from the diffusion model to the conditional image
   194	generator.
   195	For learning the style of infrared images, we propose to further finetune the pre-trained diffusion model. Compared with full fine-tuning,
   196	which updates all the parameters of the pre-trained model, Low-Rank
   197	Adaptation (LoRA) is more efficient in storage and computation. For
   198	a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
   199	constrain its update:
   200	
   201	2.3. Text-to-image generation
   202	Text-to-image generation is proposed to realize image generation
   203	with the guidance of text descriptions. Some works, which utilize Generative Adversarial Networks (GANs) [24,25], autoregressive models
   204	(ARM) [26] and Vector Quantized Variational AutoEncoder (VQ-VAE)
   205	Transformer-based methods [27], have made remarkable progress in
   206	text-to-image generation. Recently, diffusion models (DMs) [28,29],
   207	which leverage a denoising process, progressively refining noise to
   208	generate high-quality images through a series of iterative steps, have
   209	shown great success in image generation [30,31]. LDM [32] extends
   210	DMs to latent space and significantly alleviate computational demands.
   211	ControlNet [33] is proposed to control DMs with task-specific conditions by an additional trainable copy of pre-trained models. Imagen [34] demonstrates the efficacy of utilizing pre-trained, frozen large
   212	language models as text encoders in the context of text-to-image generation via diffusion models. SUR-adapter [35] aligns the semantic representation between simple narrative prompts and complex keywordbased prompts with the help of large language models, further improving the performance of text-to-image generation models.
   213	Despite the remarkable progress achieved in the field of text-toimage generation, the generation of images in new styles remains a
   214	challenging task, which limits its further applications. Different from
   215	methods above, we approach the adaptation of existing diffusion models from both the textual and visual perspectives to cater to the requirements of VI-ReID task.
   216	3. Method
   217	3.1. Overview
   218	The shortcomings of existing VI-ReID datasets are obvious, including the limited amount of data available for training, and the lack of
   219	rich scenes and perspectives. These shortcomings make it difficult for
   220	VI-ReID to benefit from large-scale cross-modal pretraining.

web search: 
web search: "HOH-Net" "High-Order Hierarchical Middle-Feature Learning Network"
web search: 
web search: "Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color"
codex
方法链条已经够了，但“证据闭环”不能只靠作者口头说模块有效。我会再用全文文本按 `Ablation`、`Visualization`、`Effectiveness` 等关键词定位实验段落，只抽关键证据，不展开读无关结果表。
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
exec
/bin/zsh -lc "pdftotext 'Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf' - | nl -ba | rg -i -C 4 'ablation|visualization|effectiveness|IKT|CBD|CCL|Grad|hard triple|baseline'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    38	gap between the inputs of two streams, we propose a clothing
    39	bias diminishing module specific to the main stream to regularize
    40	the features of clothing-relevant regions. Extensive experimental
    41	results demonstrate that our framework outperforms other
    42	baseline models on several widely-used CC Re-ID datasets.
    43	Index Terms—Clothing-change Person Re-Identification, IDbased Knowledge Transfer, Clothing Bias Diminishing Module
    44	
    45	I. I NTRODUCTION
    46	Person re-identification (Re-ID) aims at matching the same
--
    55	Re-ID can be broadly classified into two categories: 1) data
    56	augmentation methods and 2) biometrics-based methods. Data
    57	augmentation methods argue that the scale of current CC ReID datasets is insufficient to fully capture identity-related (IDrelated) features and attempt to augment training data [12]–
    58	[14]. However, these methods are subject to the quality of
    59	the generated virtual data, and their effectiveness is typically
    60	difficult to interpret. On the other hand, biometrics-based
    61	methods aim to explicitly capture stable biometrics features.
    62	* corresponding author. This work was supported by the National Natural
    63	Science Foundation of China (No. 62376016).
--
   135	Parsing
   136	
   137	Max Pool
   138	
   139	IKT Module
   140	
   141	…
   142	…
   143	
--
   217	feature extraction but operate independently without shared
   218	weights. The attention stream processes clothing-masked images, while the main stream takes the original images as input.
   219	To ensure that the main stream focuses comprehensively on
   220	the regions implying identity information, we incorporate an
   221	ID-based Knowledge Transfer (IKT) module between the two
   222	streams. Additionally, to decouple the ID-related features from
   223	the clothing-relevant region, we introduce a Clothing Bias
   224	
   225	Diminishing (CBD) module, which helps model the consistent
   226	clothing features with regard to the same individual.
   227	In summary, our contributions are listed as follows:
   228	• We are the first to propose a dual-stream identity-attention
   229	model that effectively compels the network to focus
   230	comprehensively on the regions containing distinctive
   231	identity information.
   232	• An effective CBD module is developed to maintain the
   233	consistency of clothing features for the same individual.
   234	• Extensive experiments demonstrate that our method
   235	achieves state-of-the-art on several clothing-change ReID datasets including PRCC and LTCC [18], [19].
   236	II. M ETHODOLOGY
--
   244	of body parts, which is critical to re-identify persons in the
   245	conventional Re-ID scenarios. To robustly extract ID-related
   246	features, we implement a dual-stream framework with an IDbased knowledge transfer module to guide the main stream
   247	toward comprehensively emphasizing ID-related regions.
   248	The IKT module can help locate the ID-related regions.
   249	However, the semantic gap between the inputs of the two
   250	
   251	streams may introduce error. The body regions of inputs
   252	merely contain shape information in the attention stream,
   253	while clothing-relevant features can still couple with IDrelated features in body parts of main stream. Thus, the
   254	IKT module might inadvertently amplify the influence of IDunrelated features while enhancing ID-related features, which
   255	can limit the overall performance of our model. To guarantee
   256	that the final features are exclusive of clothing color and
   257	texture, we introduce a CBD module.
   258	
   259	To restrain the contribution of clothing at the feature level,
   260	we propose Clothing Contrastive Loss LCCL . Let i be the
   261	index of an arbitrary sample in a batch, the LCCL is defined:
   262	i
   263	p
   264	N
   265	e(fc ·fc /τ )
   266	1 X 1 X
   267	wp log
   268	LCCL = −
   269	P (f i ·fcj /τ )
   270	p
   271	i
   272	N i=1 |Pi |
--
   296	main stream and attention stream, respectively. The original
   297	image I is passed through gm (.) to extract the global feature
   298	maps Fg . Simultaneously, the masked image M is fed into
   299	ga (.) to obtain the attention feature maps Fa . Taking Fa as
   300	input, our IKT module derives ID-based attention matrix WI :
   301	WI = σ (Wconv ∗ [mp (Fa ) ; ap (Fa )])
   302	
   303	(1)
   304	
--
   353	strategy during training, where K samples per ID and P × K
   354	samples per batch, but PK sampling overlooks the appearance
   355	diversity. However, in this paper, we encourage the network to
   356	learn ID-related features through our proposed loss function
   357	LCCL , and its effectiveness hinges on diversity in both persons
   358	and appearances within a batch. RAS sampling strategy can
   359	maintain appearance diversity by sampling A appearances of
   360	each person and fixed K images of each appearance in a batch
   361	[28]. whereas, this sampling strategy may discard a significant
   362	number of samples for appearances with large proportion by
   363	ignoring the maldistribution of different appearances. Thus, we
   364	modify RAS by replacing the fixed number of each appearance
   365	with a proportion number to accelerate the effect of LCCL .
   366	C. Training
   367	We adopt ResNet-50 pre-trained on ImageNet as our backbone [29], [30]. During the training stage, we first train the
   368	attention stream with LaID to obtain effective feature maps
   369	with abundant ID-related information, and then we jointly train
   370	the dual streams under the guidance of the overall loss:
   371	a
   372	Lall = Lm
   373	ID + LID + λ · LCCL
   374	
   375	(6)
   376	
   377	where λ denotes the weight of the LCCL , which is set to 1.0.
   378	III. E XPERIMENTS
   379	A. Datasets and Evaluation Protocols
   380	We mainly evaluate our proposed method on two popular
   381	CC Re-ID benchmark datasets PRCC and LTCC. We employ
--
   520	64.3
   521	42.3
   522	
   523	TABLE II
   524	T HE ABLATION STUDIES OF IFD ON PRCC AND LTCC.
   525	Method
   526	baseline
   527	w/ IKT
   528	w/ CBD
   529	IFD w/ CL
   530	IFD
   531	
   532	PRCC
--
   560	B. Comparison With State-of-the-art Methods
   561	We compare our IFD with two traditional Re-ID methods, namely PCB, IANet, and eleven clothing-change ReID methods, including SPT+ASE, CESD, RCSANet, AFDNet, FASM, GI-ReID, CAL, 3DSL, 3DInvarReID, AIM, and
   562	CCFA. As is illustrated in Tab. I, our proposed IFD achieves
   563	superior performance with 19.0% / 4.1% absolute improvements in rank-1 on LTCC/PRCC of CC Mode, illustrating the
   564	effectiveness of our method that effectively mines comprehensive ID-related information whereas other methods typically
   565	capture only a single category of ID-related information. In
   566	the SC mode of PRCC, the 99.0% rank-1 of IFD is close to
   567	saturation but inferior to some methods. The reason is that our
   568	IFD aims to capture clothing-irrelevant features but there are
   569	only clothing-consistent ground truth samples in this mode.
   570	C. Ablation Studies
   571	The effectiveness of components. To verify the effectiveness of each contribution in our framework IFD, we reproduce
   572	a baseline method (ResNet-50) that only remains Lm
   573	ID of IFD
   574	for training. As shown in Tab. II, the two ablated models
   575	that integrate IKT or CBD to the baseline individually both
   576	improve the performance significantly compared with the baseline. Furthermore, the whole model IFD introducing both IKT
   577	and CBD into the baseline obtains superior performance than
   578	the two ablated models. These results illustrate that both IKT
   579	and CBD could effectively facilitate our model to decouple
   580	clothing-relevant features and enhance ID-related features.
   581	
   582	0.43
   583	
--
   600	0.89
   601	
   602	ID2
   603	
   604	(b) The performance of baseline
   605	
   606	ID1
   607	
   608	ID1
--
   610	ID2
   611	
   612	(c) The performance of IFD
   613	
   614	Fig. 3. An intuitive comparison of the baseline and our model IFD specific to
   615	hard triples, namely the positive sample has absolutely different outfit with the
   616	anchor while the negative sample dressing similar with the anchor, together
   617	with their visualization results derived by grad-cam [31].
   618	
   619	Comparison of CCL and standard contrastive loss.
   620	We compare our clothing contrastive loss with widely used
   621	supervised contrastive loss [32]. As shown in Tab. II, IFD with
   622	contrastive loss namely IFD w/CL is superior to the baseline.
   623	Besides, IFD with clothing contrastive loss namely IFD further
   624	outperforms IFD w/CL significantly, which illustrates the effectiveness of the designed clothing bias diminishing strategy.
   625	D. Visualization
   626	As the similarity between sample pairs and the heatmap
   627	visualization results shown in Fig. 3, the baseline believes the
   628	negative sample is more similar to the anchor since it focuses
   629	more attention on ID-unrelated features such as the texture of
   630	clothing. In contrast, our IFD can consistently highlight the
   631	critical ID-related head and human contour features, facilitating the correct matching results.
--
   650	conference on computer vision and pattern recognition, 2018, pp. 2285–
   651	2294.
   652	[4] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang, “Beyond
   653	part models: Person retrieval with refined part pooling (and a strong
   654	convolutional baseline),” in Proceedings of the European conference on
   655	computer vision (ECCV), 2018, pp. 480–496.
   656	[5] Tianlong Chen, Shaojin Ding, Jingyi Xie, Ye Yuan, Wuyang Chen, Yang
   657	Yang, Zhou Ren, and Zhangyang Wang, “Abd-net: Attentive but diverse
   658	person re-identification,” in Proceedings of the IEEE/CVF international
--
   672	Chen, “Relation-aware global attention for person re-identification,” in
   673	Proceedings of the ieee/cvf conference on computer vision and pattern
   674	recognition, 2020, pp. 3186–3195.
   675	[10] Hao Luo, Youzhi Gu, Xingyu Liao, Shenqi Lai, and Wei Jiang, “Bag
   676	of tricks and a strong baseline for deep person re-identification,” in
   677	Proceedings of the IEEE/CVF conference on computer vision and
   678	pattern recognition workshops, 2019, pp. 0–0.
   679	[11] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang,
   680	“Transreid: Transformer-based object re-identification,” in Proceedings
--
   750	[30] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei,
   751	“Imagenet: A large-scale hierarchical image database,” in 2009 IEEE
   752	conference on computer vision and pattern recognition. Ieee, 2009, pp.
   753	248–255.
   754	[31] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra, “Grad-cam: Visual
   755	explanations from deep networks via gradient-based localization,” in
   756	Proceedings of the IEEE international conference on computer vision,
   757	2017, pp. 618–626.
   758	[32] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong
   759	Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan,

exec
/bin/zsh -lc "pdftotext 'HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | rg -i -C 4 'ablation|visualization|effectiveness|NWMP|PRE|non-local|parsing|sub-branch|loss'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
     2	
     3	147
     4	
     5	HPRNet: Human Parsing Reconstruction With
     6	Non-Local Multi-Scale Perception Network for
     7	Cloth-Changing Person Re-Identification
     8	Mingfu Xiong , Longlong Ge , Ruimin Hu , Senior Member, IEEE,
     9	Khan Muhammad , Senior Member, IEEE, Sambit Bakshi , Senior Member, IEEE,
    10	Javier Del Ser , Senior Member, IEEE, Xiaokang Yang , Fellow, IEEE, and Bin Sheng , Member, IEEE
--
    18	of the original sample (i.e., the person identity) owing to the
    19	absence of contextual phenomena (e.g., texture structure and
    20	local correlation), which decreases the recognition performance.
    21	To address this problem, we propose a framework called HPRNet,
    22	or “Human Parsing Reconstruction with Non-Local Multi-Scale
    23	Perception Network,” which includes a non-local weighted multiscale perception (NWMP) module and a parsing reconstruction
    24	exploration (PRE) module. In particular, the proposed NWMP
    25	module effectively captures the global receptive field of a sample
    26	and obtains a contextual correlation between non-neighboring
    27	Received 9 April 2025; revised 24 June 2025; accepted 31 July 2025.
    28	Date of publication 8 August 2025; date of current version 22 January 2026.
--
    57	School of Electronic Information and Electrical Engineering, Shanghai Jiao
    58	Tong University, Shanghai 200240, China (e-mail: shengbin@sjtu.edu.cn).
    59	Digital Object Identifier 10.1109/TCSVT.2025.3597210
    60	
    61	pixels within the sample image. The PRE module was used
    62	to achieve a more accurate reconstruction of human body
    63	components with a clothing parsing model to better distinguish
    64	features related to or unrelated to clothes. Extensive experiments
    65	were conducted on CC-ReID public datasets (LTCC, PRCC, and
    66	CCVID) to demonstrate the effectiveness and competitiveness of
    67	the proposed method with state-of-the-art (SOTA) baselines for
    68	this complex modeling task.
    69	Index Terms—Cloth-changing person re-identification, human
    70	reconstruction, non-local exploration, multi-scale perception,
    71	visual scene understanding.
    72	
    73	I. I NTRODUCTION
    74	LOTH-CHANGING Person Re-identification (CC-ReID)
    75	aims to recognize specific pedestrians with different
    76	outfits in nonoverlapping surveillance systems at various times
    77	and locations. This paradigm has received widespread attention in the multimedia and computer vision communities [1],
    78	[2]. Due to the unique nature of this task, it is widely applied
    79	in related fields such as visual scene understanding [3], [4],
    80	target tracking [5], criminal investigation [6], [7], to mention
    81	a few. Meanwhile, owing to objective factors such as lighting,
--
   113	
   114	Fig. 1. (a) Overall diagram illustrating the general design approach followed
   115	by existing CC-ReID methods, which typically emphasize clothing-irrelevant
   116	feature extraction and pedestrian reconstruction in a relatively coarse manner.
   117	Methods following this approach often result in lower predictive performance
   118	due to classification bias and the model’s focus on non-ID regions that
   119	may appear related to identity. In contrast, our proposed multi-scale perception HPRNet framework depicted in (b) incorporates a non-local weighted
   120	multi-scale perception module and a parsing reconstruction module, both
   121	contributing to improved performance on the CC-ReID task.
   122	
   123	Recently, novel approaches have been proposed as potential
   124	solutions to the CC-ReID problem by adopting one of the
   125	two general modeling strategies explained above. For example,
   126	Yang et al. [16] proposed a causality-based Auto-Intervention
   127	Model (AIM) to guarantee the learning of the feature representation of pedestrians, which is unaffected by clothing
   128	bias. Cui et al. [1] presented a disentanglement DCR-ReID
   129	framework called Deep Component Reconstruction, which
   130	discriminates the clothes-irrelevant or relevant features in a
   131	controllable strategy for CC-ReID. They exploited human
   132	component reconstruction in a deeply assembled manner to
   133	improve the performance of previously disentangled features.
   134	Although promising progress has been made in the CC-ReID
   135	task [17], [18], existing methods mainly focus on changing
   136	the color of clothes or performing appearance reconstruction
   137	directly to distinguish features related and unrelated to clothes,
--
   139	in height, body shape, and other characteristics, and such
   140	methods lack real sample cases, which causes the generated
   141	features to easily lose essential information (such as person
   142	identity and ID) of the original samples, resulting in insufficient discriminative performance of learned features unrelated
   143	to clothes. Although the work in [1] performs loss summation
   144	by analyzing pedestrian appearance parsing and reconstructing
   145	human body parts, the method described directly decomposes
   146	feature maps from the backbone network using channellevel splitting. It employs human-component region parsing
   147	features as supervisory signals to reconstruct different parts
   148	(contours, clothing, and non-clothing). However, the absence
   149	of the actual ground truth in the feature decomposition process
   150	hinders contextual texture structure segmentation and local
   151	correlation, leading to interference from subtle noise such as
   152	pedestrian background contour features, ultimately reducing
   153	the recognition performance.
   154	To address the aforementioned problems, this study proposes a modeling framework coined as “Human Parsing
   155	
   156	Reconstruction Network with Non-Local Multi-Scale Perception Network” (HPRNet), which incorporates a non-local
   157	weighted multi-scale perception (NWMP) module and a parsing reconstruction exploration (PRE) module for CC-ReID.
   158	The general workflow of the proposed framework is illustrated
   159	in Fig. 1 (b). Specifically, to obtain the contextual correlation
   160	between non-neighboring pixels of the same pedestrian appearance, the NWMP module is proposed to effectively capture
   161	the global receptive field via a multi-scale progressive learning strategy for features unrelated to clothing, which differs
   162	from changing the color of clothes directly. Unlike existing
   163	human component-based reconstruction methods, the PRE
   164	module enhances human component reconstruction by leveraging clothing parsing and a multi-local component generation
   165	strategy, effectively distinguishing between clothing-related
   166	and unrelated features.
   167	Extensive experiments were conducted to evaluate the
   168	performance of the proposed method, demonstrating that it
--
   171	multiscale perception schemes have been proven to significantly improve the accuracy of CC-ReID (Rank-01 and mAP)
   172	compared to existing CC-ReID methods over the LTCC [19],
   173	PRCC [20], and CCVID [11] datasets. Our experiments also
   174	examined the contribution of each element of the proposed
   175	HPRNet approach to the performance gains through ablation
   176	studies and visualizations.
   177	The main contributions of this study are summarized as
   178	follows.
   179	• We propose the HPRNet framework, which comprises
   180	a non-local weighted multi-scale perception (NWMP)
   181	module and a parsing reconstructed exploration (PRE)
   182	module for the CC-ReID task.
   183	• The proposed NWMP module effectively captures the
   184	global receptive field through a multi-scale progressive
   185	learning strategy for features unrelated to clothing. The
   186	PRE module incorporates clothing parsing and a human
   187	local component generation strategy to achieve more
   188	accurate human component reconstruction and better distinguish between clothing-related and unrelated features.
   189	• Extensive experiments are conducted on the public CCReID datasets LTCC, PRCC, and CCVID to demonstrate
   190	the competitive accuracy of our proposed method. Additionally, ablation studies and visualizations are performed
   191	to verify the contribution of each module to the overall
   192	performance of HPRNet.
   193	The rest of the manuscript is organized as follows: firstly
   194	Section II briefly revisits existing CC-ReID methods, including
   195	data-driven disentanglement methods, feature-driven disentanglement methods, and Human Reconstruction-based CC-ReID
   196	methods. In Section III, we describe the proposed HPRNet
   197	method. Section IV presents and discusses the experimental
   198	results and ablation study. Finally, concluding remarks and
   199	future work are presented in Section V.
   200	II. R ELATED W ORK
   201	In this section, related work on CC-ReID is reviewed.
   202	First, we describe the data-driven disentanglement methods for
   203	
   204	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   205	
   206	XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK
   207	
   208	CC-ReID (Subsection II-A). Feature-driven disentanglement
   209	strategies are revisited in Subsection II-B. Finally, we revise
   210	the human reconstruction-based methods in Subsection II-C,
   211	followed by a short statement of the novelty of this work
   212	compared to existing works (Subsection II-D).
   213	A. Data-Driven Disentanglement CC-ReID Methods
   214	As mentioned previously, CC-ReID methods based on datadriven disentanglement focus on entailing random alterations
   215	within the designated outfit area to reduce reliance on color
   216	or texture attributes. The primary strategy employed by these
   217	methods involves increasing the amount of dataset at the data
   218	level, which includes expanding data templates and synthesizing clothing-relevant features [9], [10], [21]. Specifically,
   219	a semantic-guided sampling strategy was proposed in [10]
   220	to enforce the learning of clothing-independent features by
   221	recognizing pedestrians’ outfit appearances (e.g., tops and
   222	pants) and sampling features from other individuals’ images
   223	for CC-ReID. The method for this same task presented in [22]
   224	incorporated identity-aware Mixstyle and graph enhancement
   225	modules to construct variable clothing-based fine-grained style
   226	transformation features and cross-domain style transfer-based
   227	enhanced samples. CCPG [23] was used to enhance data
--
   239	such as [40], [41], [42] adopted the idea of pedestrian reconstruction to address the CC-ReID task.
   240	D. Contribution
   241	Although existing CC-ReID technologies have achieved
   242	promising performance, they lack realistic guidance on generated samples, leading to a potentially corrupted feature space
   243	and uncontrollable results. By contrast, we propose HPRNet, a novel framework that integrates a non-local weighted
   244	multi-scale perception module and a parsing-generation
   245	reconstructed exploration module for the CC-ReID task.
   246	By leveraging component parsing reconstruction and a
   247	multi-scale non-local weighted attention interaction strategy,
   248	HPRNet effectively enhances CC-ReID performance and mitigates the performance degradation caused by the feature
   249	decomposition process present in state-of-the-art approaches.
   250	III. P ROPOSED HPRN ET F RAMEWORK
   251	In this section, we introduce the proposed HPRNet framework for CC-ReID tasks. A specific sub-model constituting the
   252	overall framework of this study is introduced. The loss function and differences from existing methods are also described
   253	separately, with the details as follows:
   254	
   255	B. Feature-Driven Disentanglement CC-ReID Methods
   256	
   257	A. Overview
   258	
   259	Differently than the previous category, feature-driven methods for the CC-ReID problem hinge on distinguishing
   260	clothing-related and clothing-unrelated attributes by using
   261	feature extraction [2], [11], [12], [26]. An adversarial feature
   262	learning method text description-based, called DIFFER [11],
   263	has been proposed for the separation of identity features.
--
   277	In this section, we first establish the mathematical notations
   278	and formally state the modeling problem addressed by the
   279	CC-ReID methods (Section III-B). Subsequently, the proposed
   280	HPRNet framework is introduced, as shown in Fig. 2. Subsequently, the nonlocal weighted multi-scale perception module
   281	(Section III-C) and the human body parsing reconstructed
   282	exploration module (Section III-D) are described. Finally, the
   283	optimization process and the differences between the proposed
   284	method and other baselines are discussed in Sections III-E
   285	and III-F, respectively.
   286	
   287	C. Human Reconstruction-Based CC-ReID Methods
   288	Recently, human reconstruction learning-based solutions,
   289	which are essentially feature-driven methods, have gained
   290	widespread attention within the CC-ReID community [33],
   291	[34], [35], [36]. These methods mainly reconstruct pedestrian
   292	
   293	B. Notation and Problem Statement
   294	Given a ReID gallery dataset, which is denoted as G =
--
   314	150
   315	
   316	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   317	
   318	Fig. 2. Diagram showing the design of the HPRNet framework. The framework comprises three key components: parsing, reconstruction, and feature extraction
   319	models. These components are integrated into the NWMP and PRE modules. The NWMP module is designed to capture the global receptive field of each
   320	person, while the PRE module further distinguishes clothing-related and clothing-independent features through a parsing-reconstruction pipeline.
   321	
   322	parametric function whose parameters are collectively denoted
   323	by θ. We aim to find the image ga in the gallery set that is most
   324	similar to probe q p , that is, a∗ , such that distance d minimizes
--
   347	respectively. q p and ga are pedestrians with the same identity
   348	but wearing different clothing. qn denotes dressed clothes
   349	similar to ga but with different identities. M is the size of
   350	each mini-batch.
   351	C. Non-Local Multi-Scale Perception Module
   352	As mentioned at the end of Section II, existing methods mainly focus on pedestrian clothing color change and
   353	appearance transfer, but ignore the local contextual correlation
   354	features of human appearance, resulting in decreased model
   355	performance. This section proposes a human appearance nonlocal multi-scale perception module involving a weighted
   356	
   357	non-local perception mechanism (WNPM) and a dual-scale
   358	progressive learning strategy (DPLS). Details are provided in
   359	the following sections.
   360	1) WNPM: Specifically, the weighted non-local perception,
   361	inspired by the attention mechanism and AGW-ReID [43],
   362	[44], captures both prominent human appearance features
   363	and local region correlations by assigning weights to each
   364	pedestrian region. This strategy is devoted to obtaining the
--
   371	relationships between different regions. The contribution of
   372	each feature region depends not only on its own, but also on
   373	its relationship with other feature regions, thus achieving an
   374	understanding of global receptive fields (such as the overall
   375	posture and shape contour of pedestrians). Meanwhile, it suppresses the local features that may change owing to changes
   376	in perspective or occlusion and improves the accuracy and
   377	robustness of the model under harsh conditions. Specifically,
   378	the procedure followed by the WNPM module to produce
   379	map-based weights wi j , where i and j denote i-th and j-th
--
   399	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   400	
   401	(3)
   402	
   403	XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK
   404	
   405	where fi and f j are the features of the two nearest neighboring
   406	local regions. fnle denotes the output of the Non-local Encoder
   407	with the ResNet50 [45] as the backbone. T denotes the size
   408	of the local region. fki j denotes the k-th feature map in region
   409	(i, j) and fnlp is the nonlocal perception feature.
   410	2) DPLS: To acquire the multi-scale pedestrian features
   411	and achieve perceptual learning from local to whole, we
   412	propose a dual-scale progressive learning strategy (DPLS),
   413	which is based on the non-local perception patterns. Specifically, feature fnlp from WNPM is split into K equal blocks
   414	first. Then, the max-pooling operations of 2 × 2 and 4 × 4
   415	were implemented in each case. Next, we combine the maxpooling results of 2 × 2 and 4 × 4 separately to obtain two
   416	different scales of local features. Then, fnlp is included to
   417	form three different scales of pedestrian features to achieve
--
   450	conv( f , 1 × 1) denotes a 1 × 1 convolution operation on f
   451	for normalization to a uniform dimension. fcls is a multi-scale
   452	progressive learning feature (the dimensions of fcls are 6144
   453	and are averaged from fk2 , fk4 and fnlp ).
   454	After this feature is learned, the multi-classification loss Lcla
   455	is expressed as
   456	Lcla = −
   457	
   458	L−1
   459	X
--
   464	
   465	j=0
   466	
   467	where yi j is the ground truth when sample i belongs to
   468	class j, and ybi j is the predicted value.
   469	In addition, to preserve the consistency of the pedestrian identity and clothing, we also exploit the identity (ID)
   470	loss Lid as in [11], which is given by
   471	!
   472	N
   473	X
   474	y(xi , li )
--
   479	j=1 y(xi , l j )
   480	i=1
   481	where N denotes the total number of samples in the training
   482	set. y(xi , li ) is the probability that the model
   483	PNid predicts that
   484	sample xi belongs to its true label li .
   485	j=1 y(xi , l j ) is the
   486	normalized sum of the predicted probabilities for all possible
   487	identity categories l j .
   488	
   489	151
   490	
   491	Partly inspired by CAL [11], we extract the invariant features for clothing variations by penalizing the model’s ability
   492	to predict clothing attributes, which is described as follows:
   493	min Lclo (CϕC (gθ (xi ), yCi )),
   494	
   495	(8)
   496	
--
   512	
   513	(9)
   514	
   515	where CϕC (•) is the clothes classifier and CϕC (gθ (xi ) denotes
   516	the predicted clothes. yCi is the ground truth for the clothing
   517	label. gθ (xi ) denotes the result of fi after l2 -normalization, and
   518	ϕ j is the output of the weights for j-th clothes classifier after
   519	l2 -normalization. N is the size of the input batch. LC is the
   520	volume of the category data for clothing. τ is a temperature
--
   523	identities and randomly select those belonging to other identities to generate adversarial samples and improve the robustness
   524	of the model.
   525	f˜i = fi − ficlo + fĩclo ,
   526	(10)
   527	where i represents the current sample, ĩ represents a noncurrent sample, and fi represents the current sample feature.
   528	ficlo denotes the clothing-related feature of the current sample,
   529	and fĩclo is the non-current sample’s clothing-related feature by
   530	disrupting the above clothing-related (unrelated) features and
   531	reclassifying them Many adversarial samples can be simulated
--
   544	log PLC ˜
   545	( fi ·ϕ j /τ)
   546	j=1 e
   547	i=1
   548	D. Parsing Reconstructed Exploration Module
   549	To acquire clothing-independent features better, we also
   550	propose a pedestrian-based appearance parsing reconstructed
   551	exploration model, which is accomplished using a multiregion
   552	pedestrian local reconstruction and clothing parsing model.
   553	Specifically, the multiregion pedestrian local reconstruction
   554	model was inspired by [1] and [11] but differs from them.
   555	Based on the basic structure of the human body, we split
   556	the output of the reconstructed encoder using ResNet50 as
   557	the backbone into three local regions (head, body, and lower
   558	limbs), which are represented as multilocal components in
   559	Fig. 2. The multilocal maps are then fed into the generation model [1] and projected onto a deconvolution model
   560	[46] to obtain the corresponding local region reconstructed
   561	maps. Finally, these maps are concatenated to form the final
   562	reconstructed feature, fr . In addition, we computed the reconstruction loss Lor with the original pedestrian feature fo to
   563	ensure reconstruction accuracy, which is expressed as
   564	v
   565	u N
   566	uX
   567	(12)
--
   582	
   583	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   584	
   585	To extract pedestrian clothing color-independent features
   586	more effectively, we adopted a cloth parsing module to obtain
   587	pedestrian contour features independent of clothing color.
   588	Specifically, we fed the output of the Parsing Encoder with
   589	the ResNet50 as the backbone into the Clothing Parsing Model
   590	[47], which involved channel, position, and color parsing networks. It completes the pedestrian contour profile parsing from
   591	three aspects: pedestrian feature channels, human contextual
   592	position, and clothing color. The parsed pedestrian sample is
   593	represented by f p in Fig. 2. As before, to ensure the accuracy
   594	of the pedestrian sample parsing, we also used it to calculate
   595	the loss with the original feature map fo , and the parsing loss
   596	Lop is represented by Eq. (13).
   597	v
   598	u N
   599	uX
   600	(13)
   601	Lop = t ( f p j − fo j )2 ,
   602	j=1
   603	
   604	where f p j and fo j are the jth parsing and original pedestrian
   605	features, respectively. N is the sample size as mentioned above.
   606	E. Overall Loss Function of the Framework
   607	Once we complete the non-local multi-scale perception and
   608	parsing reconstructed exploration model learning, the total loss
   609	function L is represented as
   610	L = Lid + αLcla + Lclo + βLadv
   611	clo + Lor + Lop .
   612	
   613	(14)
   614	
   615	where Lid and Lclo are the pedestrian identity and clothing category loss, respectively. Lcla is the multi-classification loss for
   616	nonlocal perception loss. Lor and Lop are the reconstructed and
   617	parsing losses, respectively. α and β are the hyperparameters
   618	that balance the relative importance of every loss term in the
   619	overall training of the framework.
   620	F. Differences to Existing CC-ReID Methods
   621	In this section, we discuss the differences between existing
   622	methods and our method. Part of the inspiration for the
   623	proposed framework comes from existing DCR-ReID [1],
   624	AGW-ReID [43], and CAL [11]. However, the proposed
   625	HPRNet method differs substantially from these state-of-theart techniques, as described below.
   626	• Different design principles: The DCR-ReID method
   627	relies on component reconstruction and deep learningbased fusion for CC-ReID. The CAL method leverages clothes-based adversarial loss to extract clothingirrelevant features directly from original RGB images,
   628	while the AGW-ReID method incorporates a non-local
   629	operation within the ResNet backbone to retrieve the
   630	weights of such features. In contrast, HPRNet employs
   631	a combined strategy of non-local weighted multi-scale
   632	perception and human appearance parsing reconstruction
   633	to model contextual similarity (correlation) for nonneighboring pixels and clothing-independent features in
   634	CC-ReID. Unlike AGW-ReID, HPRNet treats the nonlocal attention operation as an independent module,
   635	integrating it with a multi-scale perception module to
   636	enhance model flexibility. Additionally, the proposed
   637	framework is optimized through a loss function that
   638	
   639	addresses both global modeling and local fine-grained
   640	feature learning, improving the robustness of fine-grained
   641	feature representation in complex scenes and making it
   642	particularly suitable for ReID tasks involving clothing
   643	changes.
   644	• Different framework structures: Although all the above
   645	methods utilize multi-branch strategies for learning and
   646	training clothing-irrelevant features, their approaches differ significantly. On one hand, DCR-ReID employs
   647	component reconstruction, clothing identification, and
   648	person identity branches. On the other hand, CAL
   649	incorporates clothing category loss and person identity branches for CC-ReID. Lastly, the AGW-ReID
   650	method integrates a non-local attention model within
   651	the backbone network to enhance pedestrian feature
   652	learning. Our proposed framework, however, adopts a
   653	structurally distinct architecture. It leverages a non-local
   654	weighted perception mechanism and a multi-scale progressive learning strategy to effectively capture the global
   655	receptive field and progressively learn local-to-whole
   656	pedestrian appearance features. Furthermore, HPRNet
   657	integrates a parsing reconstruction auxiliary model, forming a framework that, while inspired by state-of-the-art
   658	methods, introduces novel elements to effectively address
   659	the CC-ReID task.
   660	Algorithm 1 Training Procedure for the HPRNet Framework
   661	N
--
   718	• Different updating strategies: In terms of model optimization, DCR-ReID and CAL utilize a component
   719	
   720	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   721	
   722	XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK
   723	
   724	reconstruction loss term and a clothes-based adversarial loss term to represent the person’s identity, while
   725	AGW-ReID relies on ID loss as its primary optimization objective. In contrast, HPRNet introduces additional
   726	optimization strategies beyond clothing and identity
   727	losses, incorporating parsing reconstruction and multiclassification losses to enhance performance for the
   728	CC-ReID task.
   729	In summary, essential differences exist between the proposed method and existing algorithms in terms of technical
   730	principles, network structures, and optimization strategies.
   731	IV. E XPERIMENTAL R ESULTS AND D ISCUSSION
   732	In this part, the experimental results and related analyses are
   733	described to validate the effectiveness of black the proposed
   734	method. The details of this process are provided below.
   735	A. Experimental Setup
   736	1) Datasets: We conduct a series of experiments on
   737	three real-world benchmark CC-ReID datasets, which include
--
   772	153
   773	
   774	PRCC dataset. As is typical in ReID experiments, we also
   775	exploited the cumulative matching characteristics (CMC) [11]
   776	and mean average precision (mAP), which are commonly used
   777	as evaluation metrics, with the rank metric being demonstrated
   778	by Rank@1.
   779	3) Implementation Details: We utilize ResNet50 [45],
   780	which has been pre-trained on ImageNet [48], as the backbone
   781	of our model. The proposed HPRNet method was implemented on a hardware platform that included NVIDIA RTX
   782	4080 SUPER GPUs. The software environment was based
   783	on Python 3.8.10, and Ubuntu 22.04 operating system. The
   784	versatility of Python supports scientific computing and deep
--
   786	library functions for building, optimizing, and assessing the
   787	proposed CC-ReID framework. Following [11], we exploited
   788	a data augmentation operation, which included random cropping, erasing, and flipping. The input images were resized
   789	to 384 × 192. Global pooling is a combination of the maxand average-pooling strategies. Similar to [11], we employed
   790	a two-stage optimization strategy to decouple the clothingindependent features. It optimized the loss function Lid +Lclo +
   791	Ladv
   792	clo + Lcla in the first stage and the full loss function L in
   793	the second stage during the training phase. During testing, we
   794	removed the cloth-relevant branch to calculate the similarity
   795	between the query image and images in the gallery.
   796	4) Parameter Setting: The learning rate (lr) of the Adam
--
   806	to 8, and each batch contained 4 instances of 2 people with
   807	different IDs.
   808	B. Comparison With State-of-the-Art Methods
   809	1) Comparisons on the LTCC and PRCC Datasets: We
   810	summarize the performance of our presented HPRNet method
   811	and that of the latest competitors on the LTCC and PRCC
   812	datasets, which is shown in Table I. In the comparison benchmark, 18 methods were used: HACNN [49], PCB [50], IANet
   813	[51], CESD [19], SPT+ASE [20], RCSANet [52], 3DSL [53],
   814	FSMA [12], GI-ReID [54], CAL [11], AIM [16], DCR-ReID
--
   831	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   832	
   833	TABLE I
   834	C OMPARISON W ITH THE L ATEST S TATE - OF - THE -A RT M ETHODS ON THE LTCC AND PRCC DATASETS . T HE O NES W ITH THE B EST P ERFORMANCE
   835	A RE H IGHLIGHTED IN B OLD B LACK , AND S IMILAR R EPRESENTATIONS A RE A LSO F OUND IN THE OTHER TABLES
   836	
   837	TABLE II
   838	C OMPARISON W ITH THE L ATEST M ETHODS ON THE CCVID DATASET
   839	
--
   842	[60], CAL [11], DCR-ReID [1], SEMI [61], CLIP3DReID
   843	[57], ASGL [62]. These comparisons demonstrated that the
   844	proposed approach achieved competitive results in multiple
   845	dimensions.
   846	C. Ablation Study and Sensitivity Analysis
   847	
   848	excessively on clothing-independent features. This makes their
   849	performance in other settings (e.g., General or SC) or on
   850	different datasets less consistent. By contrast, by extracting
--
   866	task, which is shown in Table II. Our technique is also
   867	
   868	To demonstrate the performance of each component of
   869	the proposed HPRNet framework, we conducted a series
   870	of ablation studies and sensitivity analyzes, which included
   871	the following: 1) effectiveness of the parameter setting,
   872	2) performance of the sub-branch, 3) impact of loss functions,
   873	4) performance of different feature scaling ratios, 5) computational complexity of the model, and 6) visualization results
   874	on the above-mentioned datasets. The details are as follows.
   875	1) Effectiveness of the Parameters Setting: Our proposed
   876	HPRNet framework introduces two vital training hyperparameters, α and β, which are directly related to the performance of
   877	the CC-ReID. To demonstrate the impact of these parameters
   878	on the overall framework effectiveness, we conducted a series
   879	of experiments on the mainstream CC-ReID datasets (LTCC
   880	and PRCC), as shown in Fig. 3 and Fig. 4. From the figures,
   881	it can be seen that the values of the parameters α and β affect
   882	the performance of the model. Specifically, the parameters α
   883	and β were set to specific values for the LTCC (α = 0.2,
   884	β = 0) and PRCC (α = 1, β = 1) to obtain the best
   885	performance.
   886	2) Performance of the Sub-Branch: As described above,
   887	our proposed HPRNet framework mainly includes NWMP and
   888	PRE modules to capture the global receptive field and to distinguish features related and unrelated to clothing, respectively.
   889	To verify the performance of the single-branch (module), we
   890	
   891	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   892	
   893	XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK
   894	
   895	155
   896	
   897	Fig. 3. Influence of parameters α and β on the LTCC dataset: (a) for fixed β, results (Rank@1 and mAP) of adjusting α; (b) results corresponding to variable
   898	β for a fixed α value. In each subplot, the vertical axis represents accuracy, whereas the horizontal axis represents parameter values.
   899	
   900	Fig. 4. Influence of parameters α and β on the PRCC dataset: (a) for fixed β, results (Rank@1 and mAP) of adjusting α; (b) results corresponding to variable
   901	β for a fixed α value. In each subplot, the vertical axis represents accuracy, whereas the horizontal axis represents parameter values.
   902	TABLE III
   903	P ERFORMANCE OF D IFFERENT S UB -B RANCHES FOR O UR HPRN ET M ETHOD ON THE LTCC AND PRCC DATASETS
   904	
   905	conducted a series of ablation experiments on the LTCC and
   906	PRCC datasets; the results are shown in Table III. From
   907	the table, it can be observed that our single model achieved
   908	good performance based on the DCR-ReID [1] method. The
   909	interaction learning between the two sub-branches proposed
   910	in this study achieved the best performance. A combination
   911	of the NWMP and PRE modules can capture the global
   912	contextual and local detailed features of each person. This
   913	provides a comprehensive macro-to-micro representation of
   914	features, thereby enhancing the discrimination capability of
   915	the model. Their complementarity not only enhances the
   916	robustness of the model to unfavorable conditions (such as
   917	occlusion and view-angle changes) but also creates richer and
   918	
   919	more accurate pedestrian features through a feature-interaction
   920	strategy.
   921	3) Impact of Loss Functions: To evaluate the impact of
   922	different loss functions on the CC-ReID performance, we also
   923	conduct the ablation experiments on the LTCC and PRCC
   924	datasets, described in Table IV. We assess the following
   925	loss functions: the pedestrian ID and clothing category losses
   926	adv
   927	(`ico = `id + `clo + `clo
   928	), Reconstructed Loss (`or ), Parsing
   929	Loss (`op ), and Multi-classification Losses (`cla ). Specifically,
   930	L1 represents a combination of `ico , `or and `cla . L2 includes
   931	`ico , `op and `cla . `or , `op , and `cla are consisted of L3 . L4
   932	includes `ico , `or , and `op . Finally, L5 indicates that all loss
   933	functions are included, that is, the proposed HPRNet method.
   934	
   935	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   936	
--
   947	TABLE VI
   948	C OMPARISON OF M ODEL C OMPLEXITY. “PARAM .”
   949	D ENOTES PARAMETERS
   950	
   951	Fig. 5. Visual comparison of our reconstruction method without the PRE
   952	module against SCHP [63], CE2P [64], and PGECNet [65]. The presented
   953	visual results demonstrate that the proposed method achieves competitive
   954	performance regarding reconstruction quality.
   955	
   956	As shown in Table IV, the combination of all the proposed
   957	loss functions in our method has an essential impact on the
   958	performance of the model.
   959	4) Effectiveness of the Non-Local Multi-Scale Perception
   960	Module: To verify the performance of our proposed non-local
   961	multi-scale perception module, we have conducted a series
   962	of experiments on the LTCC and PRCC datasets, which are
   963	shown in Table V.
   964	
--
   969	key features (body shape and gait) and perform a reduction
   970	operation (i.e., reduce the number of feature channels) on the
   971	original input features to reduce the computational complexity
   972	and redundancy of the module. The results are listed in
   973	Table V. The left column “Reduction” represents reducing the
   974	number of feature channels. As demonstrated in the table, the
   975	optimal performance at Rank@1 for the LTCC dataset was
   976	obtained when the reduction was set to 8, whereas it was 16
   977	for the PRCC dataset.
--
   979	of our proposed approach. The main comparison was made
   980	
   981	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
   982	
   983	XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK
   984	
   985	157
   986	
   987	Fig. 6. Clustering results based on t-SNE visualization comparing our method with DCR-ReID [1] on the LTCC and PRCC datasets. By mapping the features
   988	to a low-dimensional space, the figure illustrates the distribution of samples with the same identity. It can be observed that after processing with our method,
   989	the samples belonging to the same identity are grouped more cohesively in the specialized space, demonstrating improved feature clustering performance.
   990	
   991	Fig. 7. Visualization of retrieval results and heatmaps for the LTCC and PRCC datasets. The heatmaps are generated using Grad-CAM [67], where colors
   992	closer to red indicate higher model interest. Green boxes denote correctly retrieved instances, while red boxes indicate incorrectly retrieved instances. These
   993	visualized results demonstrate the effectiveness of the proposed method in accurately retrieving target results.
   994	
   995	between the parameters of the model, multiple–accumulate
   996	operations (MACs), and training time (s) for each epoch, as
   997	shown in Table VI. We compared the most typical CC-ReID
--
  1008	practicality of HPRNet for resource-constrained computing
  1009	devices by virtue of its better balance between modeling
  1010	performance and memory footprint.
  1011	6) Visual Inspection of the Results: In addition to more
  1012	intuitively demonstrating the effectiveness of our model, we
  1013	also conduct a series of intermediate process visualizations
  1014	to display the results. First, to verify the effectiveness of our
  1015	reconstructed model, we provide the results of its visualization,
  1016	as shown in Fig. 5. Then, we present the sorting results of the
  1017	proposed method for specific pedestrian retrieval. In addition,
  1018	we demonstrate the effectiveness of the proposed method for
  1019	CC-ReID classification from a clustering perspective, as shown
  1020	
  1021	Fig. 8. Visualization parsing results on the LTCC and PRCC datasets,
  1022	respectively. The top row is the example of the original image, and the bottom
  1023	is the parsing results for the corresponding target.
  1024	
  1025	in Fig. 6, where each rectangular box represents the clustering
  1026	effect of an individual. The figure shows that, compared
  1027	to DCR-ReID [1], our method has more compact clusters.
  1028	Meanwhile, we demonstrated a heatmap of each person’s focus
  1029	using the proposed HPRNet method, indicating that the model
  1030	focuses on key areas of interest, as shown in Fig. 7. We
  1031	also provided intermediate examples of pedestrian parsing,
  1032	as shown in Fig. 8. It can be observed from the figure that
  1033	some features unrelated to clothing are well displayed using
  1034	the parsing strategy.
  1035	V. C ONCLUSION
  1036	This work has proposed a framework called HPRNet, namely “Human Parsing Reconstruction Network with
  1037	
  1038	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
  1039	
  1040	158
  1041	
  1042	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
  1043	
  1044	Multi-modal Fundamental Models”, which involves a nonlocal weighted multi-scale perception (NWMP) module and a
  1045	parsing reconstruction exploration (PRE) module for the CCReID task. The NWMP module captures the global receptive
  1046	field of the sample to obtain the contextual correlation between
  1047	non-neighboring pixels within the sample image. The PPE
  1048	module aims to achieve a more accurate reconstruction of
  1049	human body components using a clothing parsing model
  1050	that distinguishes between features related or unrelated to
  1051	outfits. Extensive experiments were conducted on the CCReID public datasets to demonstrate the promising results
  1052	of the method used in this study. In addition, although our
  1053	method is effective in solving the CC-ReID task, it may have
  1054	some weaknesses in addressing real-world scenarios, such as
  1055	adverse weather conditions or blurry artifacts caused by rapid
  1056	camera dynamics. We acknowledge that the current parsing
  1057	module may lose some fine-grained features that contribute
  1058	to recognition when extracting garment-independent features.
  1059	Therefore, a key direction for future research is to overcome
  1060	performance degradation.
--
  1074	pp. 5398–5412, Aug. 2024.
  1075	[5] M. Dunnhofer, A. Furnari, G. M. Farinella, and C. Micheloni, “Visual
  1076	object tracking in first person vision,” Int. J. Comput. Vis., vol. 131,
  1077	no. 1, pp. 259–283, Jan. 2023.
  1078	[6] M. Ye, W. Shen, J. Zhang, Y. Yang, and B. Du, “SecureReID: Privacypreserving anonymization for person re-identification,” IEEE Trans. Inf.
  1079	Forensics Security, vol. 19, pp. 2840–2853, 2024.
  1080	[7] L. Li, W. Wang, T. Zhou, R. Quan, and Y. Yang, “Semantic hierarchyaware segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46,
  1081	no. 4, pp. 2123–2138, Apr. 2024.
  1082	[8] J. Zhang, Z. Tu, J. Weng, J. Yuan, and B. Du, “A modular neural motion
--
  1092	pp. 1050–1059.
  1093	[12] P. Hong, T. Wu, A. Wu, X. Han, and W.-S. Zheng, “Fine-grained shapeappearance mutual learning for cloth-changing person re-identification,”
  1094	in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
  1095	2021, pp. 10508–10517.
  1096	[13] Y.-J. Li, Z. Luo, X. Weng, and K. M. Kitani, “Learning shape representations for clothing variations in person re-identification,” 2020,
  1097	arXiv:2003.07340.
  1098	[14] W. Xu, H. Liu, W. Shi, Z. Miao, Z. Lu, and F. Chen, “Adversarial
  1099	feature disentanglement for long-term person re-identification,” in Proc.
  1100	13th Int. Joint Conf. Artif. Intell., Aug. 2021, pp. 1201–1207.
--
  1106	inspired cloth-debiasing for cloth-changing person re-identification,” in
  1107	Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
  1108	2023, pp. 1472–1481.
  1109	[17] T. Zhou, Y. Yang, and W. Wang, “Differentiable multi-granularity human
  1110	parsing,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–14, 2023.
  1111	[18] J. Wu et al., “A two-stream hybrid convolution-transformer network
  1112	architecture for clothing-change person re-identification,” IEEE Trans.
  1113	Multimedia, vol. 26, pp. 5326–5339, 2024.
  1114	[19] X. Qian et al., “Long-term cloth-changing person re-identification,” in
--
  1136	[26] X. Liang and Y. S. Rawat, “DIFFER: Disentangling identity features via
  1137	semantic cues for clothes-changing person re-ID,” in Proc. Comput. Vis.
  1138	Pattern Recognit. Conf., Jun. 2025, pp. 13980–13989.
  1139	[27] F. Liu, M. Kim, Z. Gu, A. Jain, and X. Liu, “Learning clothing
  1140	and pose invariant 3D shape representation for long-term person reidentification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct.
  1141	2023, pp. 19617–19626.
  1142	[28] G. Zhang, J. Liu, Y. Chen, Y. Zheng, and H. Zhang, “Multi-biometric
  1143	unified network for cloth-changing person re-identification,” IEEE
  1144	Trans. Image Process., vol. 32, pp. 4555–4566, 2023.
--
  1152	[31] Y. Huang, J. Xu, Q. Wu, Y. Zhong, P. Zhang, and Z. Zhang, “Beyond
  1153	scalar neuron: Adopting vector-neuron capsules for long-term person
  1154	re-identification,” IEEE Trans. Circuits Syst. Video Technol., vol. 30,
  1155	no. 10, pp. 3459–3471, Oct. 2020.
  1156	[32] Q. Wang, X. Qian, B. Li, X. Xue, and Y. Fu, “Exploring finegrained representation and recomposition for cloth-changing person re-identification,” IEEE Trans. Inf. Forensics Security, vol. 19,
  1157	pp. 6280–6292, 2024.
  1158	[33] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked
  1159	autoencoders are scalable vision learners,” in Proc. IEEE/CVF Conf.
  1160	Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 15979–15988.
  1161	[34] Z. Xie et al., “SimMIM: A simple framework for masked image
  1162	modeling,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
  1163	(CVPR), Jun. 2022, pp. 9653–9663.
  1164	[35] T. Zhan, Y. Song, J. Wang, and L. Wang, “VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training,”
  1165	in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2022, pp. 10078–10093.
  1166	[36] Z. Tu et al., “Consistent 3D hand reconstruction in video via selfsupervised learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45,
  1167	no. 8, pp. 9469–9485, Aug. 2023.
  1168	[37] Z. Zheng, L. Zheng, and Y. Yang, “Unlabeled samples generated by
--
  1174	Comput. Vis. (WACV), Feb. 2025, pp. 1608–1617.
  1175	
  1176	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:29 UTC from IEEE Xplore. Restrictions apply.
  1177	
  1178	XIONG et al.: HPRNet: HUMAN PARSING RECONSTRUCTION WITH NON-LOCAL MULTI-SCALE PERCEPTION NETWORK
  1179	
  1180	[39] Z. Zheng, X. Yang, Z. Yu, L. Zheng, Y. Yang, and J. Kautz, “Joint
  1181	discriminative and generative learning for person re-identification,” in
  1182	Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
--
  1187	shape embedding for cloth-changing person re-identification via 2D-3D
  1188	correspondences,” in Proc. 31st ACM Int. Conf. Multimedia, Oct. 2023,
  1189	pp. 7121–7130.
  1190	[42] M. Xiong et al., “Cloth-changing person re-identification with invariant
  1191	feature parsing for UAVs applications,” IEEE Trans. Veh. Technol.,
  1192	vol. 73, no. 9, pp. 12448–12457, Sep. 2024.
  1193	[43] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C. H. Hoi, “Deep
  1194	learning for person re-identification: A survey and outlook,” IEEE Trans.
  1195	Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 2872–2893, Jun. 2022.
  1196	[44] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural
  1197	networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.,
  1198	Jun. 2018, pp. 7794–7803.
  1199	[45] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
  1200	image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.
  1201	(CVPR), Jun. 2016, pp. 770–778.
  1202	[46] H. Noh, S. Hong, and B. Han, “Learning deconvolution network for
  1203	semantic segmentation,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV),
  1204	Dec. 2015, pp. 1520–1528.
  1205	[47] M. Xiong et al., “A triplet appearance parsing network for person reidentification,” in Proc. ICASSP - IEEE Int. Conf. Acoust., Speech Signal
  1206	Process. (ICASSP), Jun. 2021, pp. 4245–4249.
  1207	[48] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet:
  1208	A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput.
  1209	Vis. Pattern Recognit., Jun. 2009, pp. 248–255.
--
  1223	[53] J. Chen et al., “Learning 3D shape feature for texture-insensitive
  1224	person re-identification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
  1225	Recognit., Jun. 2021, pp. 8146–8155.
  1226	[54] X. Jin et al., “Cloth-changing person re-identification from a single
  1227	image with gait prediction and regularization,” in Proc. IEEE/CVF Conf.
  1228	Comput. Vis. Pattern Recognit., Jun. 2022, pp. 14278–14287.
  1229	[55] Y. Liu, H. Ge, Z. Wang, Y. Hou, and M. Zhao, “Clothes-changing person
  1230	re-identification via universal framework with association and forgetting
  1231	learning,” IEEE Trans. Multimedia, vol. 26, pp. 4294–4307, 2024.
  1232	[56] Y. Huang et al., “Meta clothing status calibration for long-term person
  1233	re-identification,” IEEE Trans. Image Process., vol. 33, pp. 2334–2346,
  1234	2024.
  1235	[57] F. Liu, M. Kim, Z. Ren, and X. Liu, “Distilling CLIP with dual guidance
  1236	for learning discriminative human body shape representation,” in Proc.
  1237	IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024,
  1238	pp. 256–266.
  1239	[58] M. Xiong, X. Yang, Z. Sun, X. Hu, A. I. Alzahrani, and K. Muhammad, “RFFR-net: Robust feature fusion and reconstruction network for
  1240	clothing-change person re-identification,” Inf. Fusion, vol. 118, Jun.
  1241	2025, Art. no. 102885.
  1242	[59] X. Gu, H. Chang, B. Ma, H. Zhang, and X. Chen, “Appearancepreserving 3D convolution for video-based person re-identification,” in
  1243	Proc. Eur. Conf. Comput. Vis., 2020, pp. 228–243.
  1244	[60] R. Hou, H. Chang, B. Ma, S. Shan, and X. Chen, “Temporal complementary learning for video person re-identification,” in Proc. Eur. Conf.
  1245	Comput. Vis. Cham, Switzerland: Springer, 2020, pp. 388–405.
  1246	[61] V. D. Nguyen, P. Mantini, and S. K. Shah, “Temporal 3D shape
--
  1250	
  1251	159
  1252	
  1253	[62] V. D. Nguyen, S. Mirza, P. Mantini, and S. K. Shah, “Attention-based
  1254	shape and gait representations learning for video-based cloth-changing
  1255	person re-identification,” 2024, arXiv:2402.03716.
  1256	[63] P. Li, Y. Xu, Y. Wei, and Y. Yang, “Self-correction for human parsing,”
  1257	IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 3260–3271,
  1258	Jun. 2022.
  1259	[64] T. Ruan, T. Liu, Z. Huang, Y. Wei, S. Wei, and Y. Zhao, “Devil in the
  1260	details: Towards accurate single and multiple human parsing,” in Proc.
  1261	AAAI Conf. Artif. Intell., vol. 33, Jul. 2019, pp. 4814–4821.
  1262	[65] S. Zhang, G.-J. Qi, X. Cao, Z. Song, and J. Zhou, “Human parsing
  1263	with pyramidical gather-excite context,” IEEE Trans. Circuits Syst. Video
  1264	Technol., vol. 31, no. 3, pp. 1016–1030, Mar. 2021.
  1265	[66] Z. Cui, J. Zhou, and Y. Peng, “Dkc: Differentiated knowledge consolidation for cloth-hybrid lifelong person re-identification,” in Proc. Comput.
  1266	Vis. Pattern Recognit. Conf. (CVPR), Jun. 2025, pp. 3573–3582.

exec
/bin/zsh -lc "pdftotext 'HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -i -C 4 'ablation|visualization|effectiveness|HSL|FCGA|HMAL|MRIC|model collapse'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 158ms:
  1938	representations and minimize the modality gaps among the
  1939	VIS, IR, and middle features. The MRIC loss consists of three
  1940	items: an intra-range loss, a middle feature loss, and an intermodality loss based on identity centers. The illustration of the
  1941	MRIC loss is presented in Fig. 5.
  1942	Following previous works [1], [7], we apply the holistic
  1943	and partial generalized mean pooling to each feature in R̃
  1944	and concatenate the pooling features to obtain the 1D middle
  1945	features, and we can get the 1D middle feature set R̃0 =
--
  2063	Pull
  2064	
  2065	Modality representations
  2066	
  2067	Fig. 5. Illustration of the proposed MRIC loss. Different colors represent
  2068	different identities.
  2069	
  2070	pedestrian with the identity i at the long-range VIS modality
  2071	can be achieved as
--
  2230	where C vis , C ir , and C mid represent the identity center sets
  2231	of VIS, IR, and middle features, respectively; C vis and C ir
  2232	denote the averaged features from the same modality for each
  2233	person; C mid is obtained by averaging all the middle features
  2234	for each person. Thus, the MRIC loss is
  2235	M ID
  2236	V IM
  2237	LM RIC = LSL
  2238	M RIC + LM RIC + LM RIC .
  2239	
  2240	(25)
  2241	
  2242	Finally, we adopt the cross-entropy loss (LCE [41]), the
  2243	triplet loss (LT RI [42]), the HMAL loss (LHM AL ), and the
  2244	MRIC loss (LM RIC ) to jointly train the HOH-Net. The joint
  2245	loss L is defined as
  2246	L = LCE + LT RI + LHM AL + LM RIC .
  2247	
  2248	(26)
--
  2266	[1] as our backbone network. The learning rate is warmed
  2267	up from 0.01 to 0.1 over the first 10 epochs, then decays to
  2268	0.01 at epoch 20 and 0.001 at epoch 50. We use SGD as the
  2269	optimizer with a momentum parameter set to 0.9. The number
  2270	of hyperedges M in the HSL module is set to 256. In the
  2271	FCGA module, the reduction ratio r is set to 32. in Eq. (5)
  2272	is set to 1.3, 1.1, 1.3, and 1.3 on the SYSU-MM01, RegDB,
  2273	LLCM, and HITSZ-VCM datasets, respectively. In the FCGA
  2274	module, the number Y of person regions is set to 9, 8, 9, and
  2275	9 on the SYSU-MM01, RegDB, LLCM, and HITSZ-VCM
  2276	datasets, respectively. For HITSZ-VCM, each video sequence
  2277	consists of 14 frames, averaged as the video representation. We
--
  2553	SYSU-MM01 RegDB
  2554	Params / FOPLs / Inference
  2555	DR
  2556	BF E
  2557	SLE HSL FCGA LM
  2558	HM AL LHM AL LM RIC R-1 / mAP R-1 / mAP
  2559	69.9 / 66.9 85.0 / 79.1
  2560	5.2 / 23.5 / 69.3
  2561	X
--
  2769	structure information of pedestrians and better reduce the
  2770	modality gap. In addition, compared to the previous HOS-Net,
  2771	the HOH-Net outperforms the previous it by 1.2% Rank-1 for
  2772	the VIS to IR search mode. These results indicate that the
  2773	HOH-Net can adopt the HMAL loss to generate middle-feature
  2774	agents and perform the bi-directional feature enhancement to
  2775	hierarchically mitigate the modality gap between VIS and
  2776	IR video frames, thus improving the video-based VI-ReID
  2777	performance.
--
  2786	learning. In addition, compared with our previous HOS-Net
  2787	[12] method, the number of parameters of the HOH-Net is
  2788	reduced by 29.5%, and the Rank-1 of the VIS to IR search
  2789	mode is improved by 0.8% on the LLCM dataset.
  2790	C. Ablation Studies
  2791	Effectiveness of Key Components. As shown in Table III, we
  2792	conduct ablation studies to validate the effectiveness of each
  2793	key component of the proposed HOH-Net (including HSL,
  2794	FCGA, LHM AL , and LM RIC ). #1 represents the Baseline [1]
  2795	method.
  2796	SLE: By incorporating the SLE (+1.9M, +15.3G) into the
  2797	Baseline, #2 outperforms #1, achieving about 2.5% increase in
  2798	mAP on the SYSU-MM01 dataset. This demonstrates the effectiveness of our SLE, leveraging both CNN and Transformer,
  2799	to extract both the short- and long-range person features.
  2800	HSL: By introducing HSL (+2.1M, +13.1G), #3 further
  2801	improves 3.0% increase in mAP on the SYSU-MM01 dataset,
  2802	compared with #2. This verifies the effectiveness of HSL
  2803	adopting a whitened hypergraph network to effectively model
  2804	the high-order relationships between different feature nodes.
  2805	FCGA: By introducing FCGA (+0.7M, +1.0G) to #3, #5 has
  2806	improved accuracy by 0.8% in Rank-1 on the SYSU-MM01
  2807	
  2808	10
  2809	
--
  2822	#8 simultaneously adopts LM
  2823	HM AL and LHM AL to hierarchically mitigate the modality gap and improve the feature
  2824	representation. Compared with #5, the Rank-1 performance of
  2825	#8 has been improved by 2.0% on the RegDB dataset.
  2826	LM RIC : When we apply the MRIC loss (#10) to train the
  2827	network, we achieve the best performance results (i.e., the a
  2828	remarkable 74.5% and 90.7% mAP on the SYSU-MM01 and
  2829	RegDB datasets, respectively). These results demonstrate the
  2830	effectiveness of the MRIC loss that can reduce discrepancies
  2831	between the VIS and IR modalities, obtaining a discriminative
  2832	and reliable common embedding space.
  2833	Though, compared to the Baseline, our method increases the
  2834	computational complexity (+6.3M) and has a higher number
  2835	of parameters (+35.3G), its performance gains are significantly greater (e.g., our HOH-Net outperforms the Baseline
  2836	by 7.6% in mAP on the SYSU-MM01 dataset). Additionally,
  2837	the inference time of the HOH-Net is about 1.2 times longer
  2838	than that of the Baseline, yet it remains acceptable for realworld applications. Though gradually adding modules (i.e.,
  2839	SLE, HSL, FCGA, and loss functions (LHM AL and LM RIC ))
  2840	slightly increase complexity, they also significantly improve
  2841	the mAP from 79.1% to 90.7% on the RegDB dataset. It is
  2842	essential to highlight that the FCGA module generates reliable
  2843	middle features and middle-feature agents through the groundtruth labels of VIS and IR images at the stage of training.
  2844	During the inference stage, the FCGA module and the loss
  2845	functions are not used for cross-modality retrieval, because the
  2846	label information is unavailable. Consequently, the inference
  2847	time of variants with and without the FCGA module and the
  2848	loss functions is the same.
  2849	TABLE IV
  2850	T HE INFLUENCE OF SLE MODULE AND THE DIFFERENT NODE
  2851	RELATIONSHIP MODELING METHODS ON THE SYSU-MM01 AND R EG DB
--
  2861	+Vision-RWKV [58]
  2862	+GNN [59]
  2863	+Hypergraph [32]
  2864	+MambaVision [60]
  2865	+HSL (PCA)
  2866	+HSL (ZCA)
  2867	+HSL (Cholesky)
  2868	
  2869	SYSU-MM01
  2870	R-1 / mAP
  2871	69.9 / 66.9
--
  2981	surpasses the “Baseline+SLE w/o TB” and “Baseline+SLE
  2982	w/o CB” by 1.1% and 0.7% in Rank-1 on the SYSU-MM01
  2983	dataset, respectively. This indicates that the SLE can effectively explore different ranges of person features by combining
  2984	CNN with Transformer.
  2985	Effectiveness of the Different Node Relationship Modeling
  2986	Methods. To verify the effectiveness of our proposed HSL
  2987	module with Cholesky, we compare it with the different node
  2988	relationship modeling methods, i.e., PConv [56], DEConv
  2989	[57], Vision-RWKV [58], GNN [59], Hypergraph [32], and
  2990	MambaVision [60] under the setting of “Baseline+SLE [12]”.
--
  2992	layer to extract features, and the GNN method models node
  2993	relationships by discovering fixed nearest neighbor feature
  2994	nodes. Different from the PConv, DEConv, and GNN that only
  2995	focus on the limited node relationship modeling, our proposed
  2996	HSL (Cholesky) module provides a more sophisticated structure by allowing connections between an arbitrary number
  2997	of whitened nodes. As shown in Table IV, our proposed
  2998	HSL (Cholesky) outperforms the DEConv by 1.8% in mAP
  2999	on the SYSU-MM01 dataset. Compared to the long-range
  3000	spatial-channel mix-based Vision-RWKV [58] and the state
  3001	space model-based MambaVision [60], the HSL (Cholesky)
  3002	method outperforms them by 1.0% and 0.6% in Rank-1 on
  3003	the SYSU-MM01 dataset, respectively. Especially, compared
  3004	to the GNN, the proposed HSL (Cholesky) brings 1.1%
  3005	and 0.7% improvements in mAP on the SYSU-MM01 and
  3006	RegDB datasets, respectively. It is important to note that
  3007	the original hypergraph [32] allows unrestricted connections
  3008	between nodes to capture high-order structural information,
  3009	but it might suffer from the influence of model collapse during
  3010	the hypergraph learning. We also analyze the impact of different whitening methods (i.e., PCA, ZCA, and Cholesky) on
  3011	hypergraphs. Compared to hypergraph [32], our proposed HSL
  3012	with PCA/ZCA/Cholesky can explore high-order relationships
  3013	and achieves 71.0% (+0.7%), 71.9% (+1.6%), and 72.4%
  3014	(2.1%) in mAP on the SYSU-MM01 dataset, respectively.
  3015	This indicates that our HSL module can effectively model
  3016	the complex and diverse high-order structure relationships
  3017	between pedestrians, and can avoid the model collapse (i.e.,
  3018	the nodes connected by different hyperedges are the same),
  3019	thus achieving discriminative pedestrian features.
  3020	Influence of the FCGA Module and the HMAL Loss.
  3021	In this subsection, we evaluate the effectiveness of the FGA
  3022	and CGA in the FCGA module, as shown in Table V.
  3023	“w/o FGA” and “w/o CGA” refer to the FCGA module
  3024	without FGA or CGA, respectively. We also compare models
  3025	trained with and without HMAL loss, denoted as “+FCGA
  3026	
  3027	11
  3028	
  3029	TABLE V
  3030	T HE INFLUENCE OF THE FCGA MODULE AND THE HMAL LOSS ON THE
  3031	SYSU-MM01 AND R EG DB DATASETS . R-1 (%) AND M AP (%) ARE
  3032	REPORTED .
  3033	Settings
  3034	Baseline+SLE+HSL
  3035	+FCGA w/o FGA
  3036	+FCGA w/o CGA
  3037	+FCGA
  3038	+FCGA w/o FGA
  3039	+FCGA w/o CGA
  3040	+FCGA
  3041	+FCGA w/o FGA
  3042	+FCGA w/o CGA
  3043	+FCGA
  3044	
  3045	w/o LHM AL
  3046	w/ LHM AL w/o Agents
  3047	w/ LHM AL
--
  3071	93.9 / 89.6
  3072	94.5 / 90.0
  3073	94.5 / 90.2
  3074	
  3075	w/ LHM AL ” and “+FCGA w/o LHM AL ”, to evaluate the
  3076	influence of the HMAL loss. Besides, we denote the FCGA
  3077	using the LHM AL without middle-feature agents and with
  3078	middle-feature agents as “+FCGA w/ LHM AL w/o Agents”
  3079	and “+FCGA w/ LHM AL ”, respectively.
  3080	As shown in Table V, on the SYSU-MM01 dataset, the
  3081	Rank-1 of “+FCGA w/o FCGA w/o LHM AL ” and “+FGA w/o
  3082	CGA w/o LHM AL ” methods are 0.3% and 0.2% higher, respectively, than the “Baseline+SLE+HSL” method. When we
  3083	combine the FGA and the CGA without LHM AL (“+FCGA
  3084	w/o LHM AL ”), it achieves 74.1% in Rank-1 on the SYSUMM01 dataset, compared with the “Baseline+SLE+HSL”
  3085	method. This shows the positive influence of the FCGA
  3086	module that aligns features at the node-level and region-level
  3087	perspectives simultaneously to achieve reliable middle features
  3088	and thus mitigate the modality gap. We adopt the HMAL
  3089	loss to hierarchically reduce the modality discrepancy with
  3090	the agents and exchange detailed and semantic information
  3091	between low- and high-stage networks (i.e., “+FCGA w/
  3092	LHM AL ”) and achieve 73.9% in mAP on the SYSU-MM01
  3093	dataset. Furthermore, “+FCGA w/ LHM AL ” achieves 0.8%
  3094	higher Rank-1 than the method without middle-feature agents
  3095	(i.e., “+FCGA w/ LHM AL w/o Agents” ) on the RegDB
  3096	dataset. These results prove that middle-feature agents and
  3097	the MDR loss can effectively and hierarchically reduce the
  3098	modality gap.
  3099	TABLE VI
  3100	T HE INFLUENCE OF THE FEATURE ENHANCEMENT IN THE HMAL LOSS
  3101	ON THE SYSU-MM01 AND R EG DB DATASETS . R-1 (%) AND M AP (%)
  3102	ARE REPORTED .
  3103	Settings
  3104	Baseline+SLE+HSL+FCGA
  3105	+LHM AL w/o Low
  3106	High
  3107	+LHM AL w/o Low ! High
  3108	+LHM AL
--
  3201	
  3202	SYSU-MM01
  3203	
  3204	TABLE VII
  3205	T HE INFLUENCE OF THE HMAL LOSS AT DIFFERENT STAGES OF THE
  3206	BACKBONE NETWORK ON THE SYSU-MM01 AND R EG DB DATASETS .
  3207	R-1 (%) AND M AP (%) ARE REPORTED .
  3208	Settings
  3209	Baseline+SLE+HSL+FCGA
  3210	+LHM AL (Stages 1-2)
  3211	+LHM AL (Stages 2-3)
  3212	+LHM AL (Stages 3-4)
  3213	+LHM AL (Stages 1-4)
--
  3230	RegDB
  3231	
  3232	bring about 0.4% and 0.7% improvements in mAP on the
  3233	SYSU-MM01 and RegDB datasets, respectively. Notably, the
  3234	“+LHM AL ” surpasses the “Baseline+SLE+HSL+FCGA” by
  3235	2.2% in mAP on the RegDB datasets. These results demonstrate that our HMAL loss can better improve the ability of
  3236	the network to capture discriminative features by performing
  3237	the bi-directional feature enhancement.
  3238	The Influence of the HMAL Loss at Different Stages. We
  3239	analyze the influence of the different stages of the backbone
  3240	network for the HMAL loss on the SYSU-MM01 and RegDB
  3241	datasets to verify its effectiveness. As shown in Table VII,
  3242	we can observe that the HAML loss achieves the best 75.5%
  3243	Rank-1 when applied to all the weight-shared feature extraction network stages (i.e., Stages 1-4 of the backbone). This
  3244	indicates that the HMAL loss can more effectively reduce the
  3245	modality gap and obtain richer semantic and more detailed
  3246	feature information when all the stages are equipped.
  3247	TABLE VIII
  3248	T HE INFLUENCE OF EACH TERM IN THE MRIC LOSS ON THE
  3249	SYSU-MM01 AND R EG DB DATASETS . R-1 (%) AND M AP (%) ARE
  3250	REPORTED .
  3251	Settings
  3252	Baseline+SLE+HSL+FCGA+LHM AL
  3253	ID
  3254	V IM
  3255	+LM RIC w/o LM
  3256	M RIC w/o LM RIC
--
  3277	94.7 / 90.2
  3278	94.9 / 90.6
  3279	95.1 / 90.7
  3280	
  3281	Influence of Each Term in the MRIC Loss. To enhance
  3282	feature representation and align VIS, IR, and middle features, we propose the MRIC loss (LM RIC ), composed of
  3283	M ID
  3284	V IM
  3285	three terms: LSL
  3286	M RIC , LM RIC , and LM RIC . As shown in
  3287	Table VIII, ablation studies on the RegDB dataset confirm the effectiveness of each term, with mAP progressively increasing from 73.9% to 74.5% as more components are included. Compared with the model trained without
  3288	MRIC (i.e., “Baseline+SLE+HSL+FCGA+LHM AL ”), adding
  3289	the complete MRIC loss yields a 0.7% Rank-1 improvement
  3290	on SYSU-MM01 dataset, which demonstrates that the HOHNet trained with the MRIC loss can obtain a more discriminative and more reliable common feature space.
  3291	Influence of the Stage Locations of the Key Components.
  3292	Our key components (SLE, HSL, and FCGA) can be easily
  3293	integrated at any stage of the backbone network. In this paper,
  3294	we utilize the AGW as the backbone for the imaged-based
  3295	VI-ReID task, which is structured into five stages (i.e., Stages
  3296	
  3297	Fig. 6. The influence of the number M of hyperedges, the balancing
  3298	parameter in Eq. (5), and the number Y of person regions in the FCGA
  3299	module on the SYSU-MM01 and RegDB datasets. R-1 (%) and mAP (%) are
  3300	reported.
  3301	
  3302	0-4). We plug these key components into different stages of
--
  3427	AND PARAMS (M) ARE REPORTED .
  3428	
  3429	(a) Traditional Hypergraph
  3430	
  3431	Fig. 7. Visualization of the high-order relationship obtained by different
  3432	methods. In each column, the green square represents that the node is
  3433	connected by a hyperedge, while the gray square represents that the node
  3434	has no connections.
  3435	SYSU-MM01
--
  3457	(i) HOH-Net Similarity
  3458	
  3459	Frequency
  3460	
  3461	Eq. (5) and the number Y of person regions in the FCGA
  3462	module in the ranges of {32, 64, 128, 256, 512}, {0.9, 1.1,
  3463	1.3, 1.5, 1.7} and {6, 7, 8, 9, 10}, respectively. The results are
  3464	illustrated in Fig. 6. We first discuss the impact of parameter
  3465	M on the performance of the proposed HOH-Net. As shown
--
  3567	and the negative (for the different identities) matching from the VIS and
  3568	IR modalities, respectively. The larger the distance D and the smaller the
  3569	overlapping area E, the better the model performance.
  3570	
  3571	D. Visualization Analysis
  3572	Visualization of High-Order Relationship. As illustrated in
  3573	Fig. 7, we present visualizations of the high-order relationships
  3574	derived from the traditional hypergraph network and our
  3575	whitened hypergraph network on the SYSU-MM01 dataset.
  3576	For the traditional hypergraph network, many nodes share the
  3577	same hyperedges, and thus the diverse and complex high-order
  3578	connections collapse into a single connection. In contrast,
  3579	our whitened hypergraph network prevents model collapse by
  3580	
  3581	using a whitening operation to project the feature nodes into
  3582	a spherical distribution.
  3583	Feature Distribution Visualization. We randomly select
  3584	90,000 positive and negative pairs from the query and gallery
  3585	sets and visualize the cosine similarity distributions on SYSUMM01, RegDB, and HITSZ-VCM (see Figs. 8(a–i)). As
  3586	shown in Figs. 8(b–c), (e–f), and (h-i), the HOH-Net shows
  3587	larger intra-/inter-class distribution differences (D2 , D4 , and
--
  3664	demonstrates that our HOH-Net is able to achieve a more
  3665	discriminative common feature space for effective VI-ReID,
  3666	and the discrepancy between VIS and IR modalities can be
  3667	effectively mitigated.
  3668	Retrieval Results. To further evaluate the effectiveness of
  3669	our proposed HOH-Net, we provide the attention maps and
  3670	retrieval results produced by both the Baseline and our method
  3671	on the SYSU-MM01 and HITSZ-VCM datasets, respectively.
  3672	As illustrated in Fig. 10(b), different from the Baseline, our
--
  3750	modality pedestrian retrieval, while the Baseline is easily
  3751	affected by modality discrepancy, resulting in poor retrieval
  3752	results.
  3753	V. C ONCLUSION
  3754	In this paper, we propose a novel HOH-Net mainly consisting of the HSL module, and the FCGA module with the
  3755	HMAL loss and the MRIC loss for VI-ReID. The HSL module
  3756	exploits diverse and complex high-order structure information
  3757	of shot- and long-range features that are extracted from the
  3758	SLE module and prevents model collapse by employing a
  3759	whitened hypergraph. Moreover, the FCGA module generates reliable middle features from the node-level and regionlevel perspectives. In particular, the HMAL loss hierarchically
  3760	reduces the modality gap by leveraging the middle-feature
  3761	agents and performs the bi-directional feature enhancement
  3762	between different stages to obtain the discriminative features.
  3763	Finally, the MRIC loss minimizes the distance between the
  3764	VIS, IR, and middle features, thereby establishing a discriminative and reliable common feature space. The quantitative
  3765	and qualitative experiments on the four challenging VI-ReID
  3766	datasets confirm the superiority of the HOH-Net in comparison
  3767	with several state-of-the-art methods. Our HOH-Net achieves

codex
证据段落也够用了：HOH 有分模块消融、关系可视化和相似度分布；HPR 有分支消融、损失消融、重建可视化、聚类和热力图；IFD 有 IKT/CBD 消融和 hard triple Grad-CAM。最后补剩下三篇的证据定位。
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
/bin/zsh -lc "pdftotext 'InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf' - | nl -ba | rg -i -C 4 'ablation|quantitative|qualitative|sub-monitoring|UV texture|direct transfer|generalization|normal maps|Canny|ControlNet|number of'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     2	
     3	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 4, APRIL 2025
     4	
     5	InfinitePerson: Innovating Synthetic Data Creation
     6	for Generalization Person Re-Identification
     7	Guoqing Zhang , Member, IEEE, Jin Li , Yuhui Zheng , Member, IEEE, and Ruili Wang
     8	
     9	Abstract— Recently, large-scale synthetic datasets have
    10	effectively alleviated the issue of insufficient person reidentification (Re-ID) datasets. However, synthetic datasets
    11	grapple with inherent challenges, including the subpar quality
    12	of synthetic pedestrians and single data collection. This paper
    13	presents InfinitePerson, a costless pipeline that fully utilizes the
    14	infinite generation capability of diffusion models to produce
    15	diverse UV texture images and effortlessly constructs high-quality
    16	synthetic datasets by simulating a real surveillance network.
    17	Specifically, we innovatively propose the utilization of diffusion
    18	models to generate high-quality, realistic, and diverse UV
    19	texture images to address the limitations of clothing textures.
    20	This ensures that our 3D character models have complete
    21	clothing texture information and look very similar to real-world
    22	pedestrians. Moreover, in response to the challenges in replicating
    23	synthetic data collection pipelines, we propose a sub-monitoring
    24	network data collection method, which can collect pedestrians
    25	data from different viewpoints, backgrounds, and lighting
    26	conditions through simple scene layout. Finally, a more scalable
    27	and realistic large synthetic dataset called InfinitePerson
    28	is created, containing 4,700 identities and 535,636 images.
    29	Experimental evidence demonstrates show that models trained
    30	on InfinitePerson exhibit superior generalization performance,
    31	surpassing those trained on both popular real-world and
    32	synthetic person Re-ID datasets. The InfinitePerson project is
    33	available at https://github.com/zhguoqing/InfinitePerson.
    34	Index Terms— Generalization person re-identification, synthetic Re-ID dataset, stable diffusion, sub-monitoring network.
    35	
    36	Received 8 June 2024; revised 20 October 2024; accepted 16 November
    37	2024. Date of publication 22 November 2024; date of current version
    38	7 April 2025. This work was supported in part by the National Natural
--
    80	synthetic data, all of which limit the further development of
    81	synthetic datasets.
    82	In existing synthetic datasets [7], [8], there are significant
    83	differences between the 3D character models used and realworld pedestrians. The main reason is that they all ignore the
    84	texture definition of 3D clothing when generating UV texture
    85	maps, leading to a misalignment between the generated UV
    86	maps with the corresponding 3D clothing models, making
    87	the 3D character that bears a resemblance to cartoon characters, as shown in Figure 1. The significant gap between 3D
    88	characters and real-world pedestrians directly affects the effectiveness of the synthetic dataset and the generalization ability
    89	of models trained on this dataset. Additionally, these synthetic
    90	data collection methods suffer from shortcomings in terms of
    91	replicability and adaptability, hindering the extension of these
    92	datasets to address specific Re-ID scenarios [9]. Therefore,
--
    94	suitable for the ReID.
    95	In this work, we propose a costless pipeline called Infiniteperson to address the above issues. To generate a variety
    96	of high-quality 3D character models, we innovatively use
    97	diffusion models to generate high quality and diverse UV
    98	texture maps and use normal maps dataset as additional input
    99	to train ControlNet [10] to guide the Stable Diffusion [11] in
   100	generating reasonable UV texture maps. Compared to previous
   101	work [7], [8], the UV images we have generated are more comprehensive and better aligned with defined clothing textures.
   102	In addition, we also propose a sub-monitoring network data
   103	
   104	1051-8215 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
   105	See https://www.ieee.org/publications/rights/index.html for more information.
   106	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   107	
   108	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   109	
   110	Fig. 1.
   111	Illustration of selected examples from (a) UnrealPerson and
   112	(b) RandPerson datasets, respectively. (c) the 3D character models with our
   113	generated UV texture maps rendered by UE4.
   114	
   115	collection method, which enables researchers to more easily
   116	design and build complex Re-ID scenes to better simulate
   117	the real world. Our major contributions are summarized as
--
   119	• We innovatively use diffusion models to generate UV
   120	texture images to resolve the problem of limited clothing
   121	texture resources in synthetic datasets and enhance both
   122	the diversity and quality of 3D character models.
   123	• We propose a sub-monitoring network data collection
   124	method, which can more accurately simulate the operation of the monitoring network, effectively control data
   125	quality and collection process, and facilitate subsequent
   126	workers to design specific Re-ID scenarios.
   127	• We create a large-scale synthetic person Re-ID dataset
   128	called InfinitePerson, which contains 535,636 images
   129	of 4,700 identities, covering various scenes and lighting conditions, providing researchers with powerful data
   130	resources.
   131	• Experimental results show that models trained on InfnitePerson have better generalization performance than
   132	models trained on other widely used real-world and
   133	synthetic Re-ID datasets.
   134	II. R ELATED W ORK
   135	A. Generating Images for Re-ID
--
   247	The above statement reveals two major shortcomings.
   248	A challenge is to collect and annotate the training dataset S.
   249	
   250	Fig. 2. Demonstrating the creation of a 3D character using MakeHuman, and
   251	adding the generated UV texture maps to the 3D character models in UE4.
   252	
   253	It often takes a lot of time and manpower to construct and
   254	accurately label pedestrians correctly in multiple different
   255	scenarios, which is very challenging for manual annotation.
--
   279	we add skeleton and walking animations suitable for game
   280	
   281	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   282	
   283	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   284	
   285	Fig. 3.
   286	
   287	3163
--
   308	from a 2D texture image onto a 3D model, followed by
   309	rendering through a rendering engine. This process enables the
   310	3D clothing model to have rich texture information, as shown
   311	in Figure 2.
   312	3) UV Texture Maps: UV texture is a technique that ensures
   313	that every point of an image is precisely corresponded to
   314	the surface of the model, while the gaps between points are
   315	processed for image smoothing interpolation. UV Mapping
   316	technology allows us to map UV texture maps onto 3D
   317	clothing models, providing them with rich texture information.
   318	However, each 3D clothing model is usually equipped with
   319	one UV texture map, which means that each clothing style
   320	has only one texture map available, thus limiting the diversity
   321	of 3D character models that can be created. In addition,
   322	although the MakeHuman community has provided a variety
   323	of clothing types, including those we wear daily, their selection
--
   326	generate diverse sets of clothing models by changing the UV
   327	texture maps of existing clothing models. This method can not
   328	only expand the diversity of 3D character models, but also
   329	provide designers with a broader creative space. To obtain
   330	diverse clothing, we need to generate different UV texture
   331	maps that adhere to the definition of clothing texture. Stable
   332	Diffusion [11] is a powerful large-scale generative model
   333	within the realm of computer vision. However, it often lacks
   334	precise control over image generation. To generate a reasonable UV texture map, it is necessary to ensure that the large
   335	diffusion model has the ability to generate high-quality texture
   336	images in a controllable manner. ControlNet [10] effectively
   337	solves this problem by controlling Stable Diffusion [11].
   338	We use normal maps dataset [36] as additional input to train
   339	ControlNet [10] to guide Stable Diffusion [11] in generating
   340	reasonable UV maps. The benefit of using normal maps
   341	dataset as additional input is that the generated UV texture
   342	maps align with the model’s surface structure. This enables
   343	the generation of high-quality textures on various types of
   344	clothing. We import the 3D clothing models and UV texture
   345	maps into the Unreal Engine 4 [37], and utilize the material
   346	system provided by the game engine to generate a wide variety
   347	of clothing models infinitely to some extent. As can be seen
   348	from Figure 4, various 3D character models can be obtained
   349	using the generated UV maps.
   350	C. Sub-Monitoring Network
   351	Previous data collection methods for synthetic datasets
   352	often focused on specific scenarios or research objectives.
   353	For instance, SynPerson [6] focuses on studying the effects
   354	of lighting without simulating pedestrian movement, while
--
   377	occlusion scenarios in the data and thus better assess the
   378	performance of the Re-ID tasks.
   379	D. InfinitPerson Dataset Overview
   380	
   381	Fig. 4. Different categories of UV texture maps. (a) Regular UV texture maps
   382	maintain the corresponding shape of the clothes. (b) Dresses and shoes with
   383	irregular UV texture maps from top to bottom. The dashed box contains the
   384	original UV map of the clothes, and others are generated by Stable Diffusion.
   385	
   386	In real surveillance scenarios, monitoring networks within
   387	an area typically consist of multiple sub-networks, and each
   388	subnet covers the limited field of view of surveillance cameras.
   389	To ensure comprehensive monitoring and security, multiple sub-networks must operate simultaneously, with each
   390	subnet responsible for monitoring different areas or locations. Inspired by this practical monitoring network structure,
   391	we introduce a sub-network data collection method in our
   392	research, aiming to collect pedestrian images within each subnetwork using a limited number of cameras, typically four
   393	to six. This approach replicates the operating mode of a real
   394	monitoring network, where each sub-network is accountable
   395	for distinct monitoring zones. In this way, we can more
   396	accurately simulate the operation of the monitoring network,
--
   419	open-source software called MakeHuman [34] to generate
   420	highly realistic human models. Similar to OWD [25], which
   421	sets pedestrians face features invisible, our dataset does not
   422	rely on facial features to distinguish identities. We randomly
   423	generate a large number of character models with different skin
   424	tones, body types (including height and weight), and hairstyles
   425	by using the plugin of MakeHuman [34]. For 3D clothing
   426	models, we carefully select 141 different types of clothing,
   427	as well as 40 types of pedestrian accessories such as hats and
   428	glasses. In our research, we combine these clothing models,
   429	which initially lack UV textures, with character models, and
   430	then import them into the game engine, and use UV texture
   431	maps generated by Stable diffusion [11] as texture images
   432	of the characters clothing texture. Ultimately, we generate
   433	4,700 3D character models. Figure 6 (a) shows images of
   434	different pedestrians in different scenes, with realistic and
--
   439	indoor and outdoor scenes such as streets, banks, shopping
   440	malls, fields, rural areas and so on, all of which are readily
   441	available from the Unreal Engine community. Within the
   442	Unreal Engine environment, we employ our proposed
   443	sub-monitoring network to position the cameras and define
   444	circular paths for pedestrians, providing effective and flexible
   445	control over data quality and collection. Figure 6 (b) shows
   446	
   447	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   448	
   449	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   450	
   451	3165
   452	
   453	Fig. 5. Example configuration of sub-monitoring network and 3D character models movements. A total of cameras are installed and the monitoring scenes
   454	show each camera’s field of view. The yellow dot represents the destination points we set, and the yellow line represents our predetermined pedestrian
   455	movement route.
   456	
   457	535,636 images of 4,700 pedestrians using a total of 72 virtual
--
   476	4) Data Annotation: During data collection, we use
   477	UnrealCV [38], [39] to gather instance segmentation annotations for every image captured by the virtual cameras. After
   478	collecting the data, we use appropriate tools to precisely crop
   479	pedestrians from the image.
   480	5) Summary: We set up 12 sub-monitoring networks across
   481	9 different scenarios and successfully captured a total of
   482	
   483	Three popular real-world person Re-ID datasets are
   484	used for generalization evaluation, including CUHK03 [3],
   485	Market-1501 [4] and MSMT17 [5]. The CUHK03 [3] dataset
   486	contains 14,097 images of 1,467 identities. The CUHK03-NP
   487	is split into a training set and a testing set. The training set
   488	consists of 7,365 images of 767 identities, while the testing
--
   497	corresponding to 1,041 identities. In the testing set, 11,659
   498	bounding boxes are designated as query images, while another
   499	82,161 bounding boxes are designated as gallery images.
   500	We compare our dataset with multiple synthetic datasets
   501	to verify its generalization ability for Re-ID. All evaluations
   502	adhere to the single-query evaluation protocol [40]. We utilize
   503	the Cumulative Matching Characteristic (CMC) [41], with
   504	specific emphasis on Rank-1 accuracy, Rank-5 accuracy, and
   505	mean Average Precision (mAP) [42] as the primary performance metrics. The synthetic dataset’s quality is evaluated
   506	through direct transfer.
   507	B. Implementation Details
   508	We employ ResNet-50 [46] as our backbone for all our
   509	experiments. The cross-entropy loss function is utilized, and
   510	all pedestrian images are resized to 256 × 128. The training
--
   532	adjustments. This evaluation method offers a robust assessment of the model’s ability to generalize across diverse
   533	datasets, providing valuable insights into its overall performance on different datasets. We evaluate our synthetic
   534	dataset’s quality through direct cross-dataset evaluation.
   535	To assess the effectiveness of our synthetic dataset, we use
   536	direct transfer performance on real-world datasets as the
   537	primary indicator of synthetic data quality, since direct transfer
   538	is the foundation of all other tasks. We employ CUHK03 [3],
   539	Market-1501 [4] and MSMT17 [5] as testing sets. The evaluation results are provided in Table II. It is clear from the results
   540	that our synthetic dataset outperforms existing synthetic and
   541	real-world datasets in direct transfer experiments. Compared
   542	to the UnrealPerson [8] collected using the same game engine
   543	as ours, our synthetic dataset also achieved better performance.
   544	The reason why we can surpass other composite datasets
   545	is that the pedestrian data collected by our proposed sub
--
   550	have more complete and reasonable clothing textures than
   551	UnrealPerson’s 3D character models, so the models trained
   552	on our synthetic dataset has better performance.
   553	D. Generalizable Person Re-ID
   554	To verify the generalization of our dataset, we choose to use
   555	various methods in the Re-ID field for comparison, including
   556	QAConv2.0 [43] based on ResNet-50 [46], TransReID [26]
   557	based on Transformer [47], PHA [44] based on Discrete Haar
   558	
--
   561	evaluation are presented in Table III. It can be observed that the
   562	optimal performance is achieved among all methods using the
   563	InfinitePerson dataset, surpassing all other competing synthetic
   564	datasets. This proves that the Re-ID model trained on our
   565	dataset has stronger generalization ability. What’s noteworthy
   566	is that our dataset’s superior performance compared to others
   567	stems from its high-quality 3D pedestrian models and diverse
   568	scene variations. These factors collectively promote the model
   569	to have stronger robustness.
   570	
   571	E. Quantitative Analysis
   572	To demonstrate that the UV texture maps generated by
   573	Stable Diffusion [11] can increase the diversity of 3D character
   574	models and to evaluate the effectiveness of the proposed
   575	sub-monitoring network for collecting synthetic datasets,
   576	we perform quantitative analysis.
   577	1) Identities: In contrast to UnrealPerson, which randomly
   578	matches clothing to create a large number of different clothing
   579	sets, we generate diverse sets of clothing models by replacing
   580	the UV texture maps of existing clothing models with those
   581	generated by diffusion models. To verify whether our proposed
   582	method of generating clothing textures can improve pedestrian
   583	diversity, we use 141 3D clothing models, including those with
   584	regular and irregular UV texture maps. For these 141 clothing
   585	models, unlike Unrealperson [8], we do not use random matching to obtain clothing sets. Instead, we choose to manually
   586	pair them into 152 clothing sets, resulting in 4,700 virtual
   587	characters with distinct clothing textures. Figure 8 clearly
   588	demonstrates that as the number of characters increases, the
   589	accuracy of Rank-1 and mAP consistently increases on all
   590	three real-world datasets, despite using only 152 clothing
   591	sets. This shows that the UV texture maps we generate can
   592	indeed increase the diversity of virtual characters, as shown
   593	in Table II. Furthermore, it’s worth noting that even if we
   594	increase the number of 3D character models from 3,000 to
   595	4,700, we observe no performance degradation. This finding
   596	challenges the notion as posited by UnrealPerson [8] that
   597	
   598	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   599	
   600	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   601	
   602	3167
   603	
   604	TABLE III
   605	D IRECT C ROSS DATASET E VALUATION R ESULTS U SING R E -ID M ETHODS . U NREAL P ERSON * M EANS THE T RAINING DATA E XTRACTED
   606	F ROM UNREAL _ V 1.1, UNREAL _ V 2.1, UNREAL _ V 3.1 AND UNREAL _ V 4.1
   607	
   608	Fig. 7. Results of direct transfer evaluation on CUHK03-NP, Market-1501 and MSMT17, varying the quantity of sub-monitoring networks in synthetic
   609	datasets from 2 to 12.
   610	
   611	optimal performance can only be achieved with 3,000 3D
   612	character models.
   613	2) Sub-Monitoring Network: To demonstrate the effectiveness of our proposed sub-monitoring network, we analyze it
   614	by setting the number of sub-monitoring networks. As shown
   615	in Figure 7, when we increase the number of sub-monitoring
   616	networks, the accuracy of Rank-1 and mAP continues to
   617	increase on all three real-world datasets. Even when we deploy
   618	sub-monitoring networks in different regions of the same large
   619	scene or use distinct networks for the same character group, the
   620	accuracy of Rank-1 and mAP consistently exhibits a consistent
   621	increase on all three real-world datasets. As the number of 3D
   622	character models and sub-monitoring networks increases, the
   623	performance remains consistent without showing any decline.
   624	This demonstrates the enormous potential of our method to
   625	further produce larger synthetic datasets.
   626	3) Dataset Parameter Analysis: To fully validate the
   627	effectiveness and applicability of our dataset, we also use
   628	QAConv2.0 [43] as the baseline and adjusted various parameters of our dataset for experiments. We first fix the number
   629	of pedestrians and change the number of clothing sets
   630	used, as shown in Table IV. The trend reveals that as the
   631	
   632	number of clothing types increases, the model’s generalization
   633	performance improves. Moreover, increasing the number of
   634	sub-monitoring networks and cameras further enhances the
   635	model’s generalization performance. This not only verifies the
   636	effectiveness of our sub monitoring network, but also verifies
   637	that using pedestrian images in multiple scenes for model
   638	training can improve the model’s generalization ability.
   639	F. Qualitative Comparisons
   640	As shown in Figure 9, there are significant differences in the
   641	UV texture maps generated using ControlNet [10] to control
   642	Stable Diffusion [11] based on different training datasets.
   643	When compared to ControlNet [10] trained on the normal
   644	map [36] dataset, the performance of ControlNet [10] trained
   645	on the Canny edge dataset [49] and that trained on the Instruct
   646	Pix2Pix dataset is notably inferior. Specifically, using the
   647	Instruct Pix2Pix dataset [50] as additional input to control the
   648	images generated by Stable Diffusion [11] are unsuitable for
   649	use as UV texture maps. Although employing the Canny maps
   650	as additional input to control the UV texture map generated by
   651	Stable Diffusion yields reasonable results, the generated image
   652	
   653	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   654	
--
   659	TABLE IV
   660	T HE P ERFORMANCE OF I NFINITE P ERSON IN D IRECT T RANSFER TO T HREE R EAL -W ORLD DATASETS I S E VALUATED . W E M ANAGE VARIOUS
   661	PARAMETERS W ITHIN O UR S YNTHETIC DATASET, I NCLUDING I DENTITIES , S UB -M ONITORING N ETWORK , C AMERAS AND C LOTHES S ET
   662	
   663	Fig. 9. UV texture maps generated by Stable Diffusion using ControlNet
   664	with different training data.
   665	Fig. 8. The change in experimental results as the number of pedestrians
   666	increase from 1,000 to 4,700.
   667	
   668	textures remain relatively simplistic. In addition, when dealing
   669	with irregular UV texture maps such as shoe, using Canny
   670	maps as additional input to control the UV texture generated
   671	by Stable Diffusion [11] cannot be used as UV texture maps
   672	either.
   673	G. Synthetic-Real Data Fusion Experiment
   674	To further verify the generalization ability of the model,
   675	we mix the real data with the synthetic data we generated
   676	for model training to enrich the diversity of the real data.
   677	The experimental results are shown in Table VI. It can been
   678	seen that compared with using only real data, the model
--
   693	we make some summaries as follows:
   694	
   695	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   696	
   697	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   698	
   699	TABLE V
   700	T ESTING THE O RIGINAL DATASET OF MARKET 1501 AND THE
   701	MARKET 1501 W ITH E NHANCED I MAGES U SING T RANS R E ID
--
   705	3169
   706	
   707	to enhance control over data quality and the data collection
   708	process by more accurately simulating the monitoring network’s operations. The experimental results demonstrate that
   709	the large synthetic dataset obtained through our method significantly improves the model’s generalization capability. Looking
   710	ahead, we can further extend our pipeline to encompass a
   711	wider array of Re-ID scenarios and explore the development
   712	of additional data sources.
   713	
--
   731	Firstly, the clothing textures we employed are generated
   732	by controlling Stable Diffusion [11], which usually demands
   733	a substantial computational resource allocation for image
   734	generation. Secondly, despite the effectiveness of using normal
   735	maps [36] as input conditions to guide the UV texture maps
   736	generated by Stable Diffusion [11], there are still some texture
   737	images with poor quality. Filtering these texture images is the
   738	key to improve generation efficiency.
   739	C. Extension of Synthetic Datasets
--
   745	severe weather [53]. At present, continuously adding realworld-simulated attributes to the synthetic dataset can yield
   746	significant benefits.
   747	VI. C ONCLUSION
   748	This paper introduces an innovative approach by employing
   749	diffusion models to generate UV texture images, aiming to
   750	improve the quality of character models and bridge the disparity between synthetic and real pedestrians. Additionally,
   751	a sub-monitoring network data collection method is designed
   752	
   753	[1] G. Zhang, Y. Zhang, H. Zhang, Y. Chen, and Y. Zheng, “Learning
   754	dual attention enhancement feature for visible–infrared person reidentification,” J. Vis. Commun. Image Represent., vol. 99, Mar. 2024,
   755	Art. no. 104076.
--
   898	Process. Syst. (NIPS), 2017, pp. 6000–6010.
   899	[48] S. G. Mallat, “A theory for multiresolution signal decomposition:
   900	The wavelet representation,” IEEE Trans. Pattern Anal. Mach. Intell.,
   901	vol. 11, no. 7, pp. 674–693, Jul. 1989.
   902	[49] J. Canny, “A computational approach to edge detection,” IEEE
   903	Trans. Pattern Anal. Mach. Intell., vol. PAMI-8, no. 6, pp. 679–698,
   904	Nov. 1986.
   905	[50] T. Brooks, A. Holynski, and A. A. Efros, “InstructPix2Pix:
   906	Learning to follow image editing instructions,” in Proc. IEEE/CVF
--
   930	pattern recognition, and machine learning.
   931	
   932	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:57 UTC from IEEE Xplore. Restrictions apply.
   933	
   934	ZHANG et al.: InfinitePerson: INNOVATING SYNTHETIC DATA CREATION FOR GENERALIZATION PERSON Re-ID
   935	
   936	Jin Li is currently pursuing the master’s degree
   937	with the School of Computer Science, Nanjing
   938	University of Information Science and Technology,

exec
/bin/zsh -lc "pdftotext 'Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf' - | nl -ba | rg -i -C 4 'ablation|visualization|response maps|over-reliance|GRS|HFQ|CLR|mutual learning|cross-domain|color prior'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     9	Tian Wang , Senior Member, IEEE, and Hichem Snoussi
    10	
    11	Abstract— Text-based person retrieval is the process of searching a massive visual resource library for images of a particular
    12	pedestrian, based on a textual query. Existing approaches often
    13	suffer from a problem of color (CLR) over-reliance, which
    14	can result in a suboptimal person retrieval performance by
    15	distracting the model from other important visual cues such
    16	as texture and structure information. To handle this problem,
    17	we propose a novel framework to Excavate All-round Information
    18	Beyond Color for the task of text-based person retrieval, which
    19	is therefore termed EAIBC. The EAIBC architecture includes
    20	four branches, namely an RGB branch, a grayscale (GRS)
    21	branch, a high-frequency (HFQ) branch, and a CLR branch.
    22	Furthermore, we introduce a mutual learning (ML) mechanism
    23	to facilitate communication and learning among the branches,
    24	enabling them to take full advantage of all-round information
    25	in an effective and balanced manner. We evaluate the proposed
    26	method on three benchmark datasets, including CUHK-PEDES,
    27	ICFG-PEDES, and RSTPReid. The experimental results demonstrate that EAIBC significantly outperforms existing methods
    28	and achieves state-of-the-art (SOTA) performance in supervised,
    29	weakly supervised, and cross-domain settings.
    30	Index Terms— Color (CLR) information, cross-modal retrieval,
    31	frequency, person reidentification (ReID), text-based person
    32	retrieval.
    33	
    34	I. I NTRODUCTION
--
    70	task is to extract and align relevant clues from multimodal
    71	data sources, including RGB pedestrian images and natural
    72	language queries. Various methods have been employed to
    73	tackle this task, but many still struggle with the problem of
    74	color (CLR) over-reliance. Utilizing a single-branch basic
    75	model (which is detailed in Section III-B), we display some
    76	retrieval examples in Fig. 1. The targeted (untargeted) pedestrian images are marked with green (red) borders. It can be
    77	noticed that within the top-ten retrieval results, quite a few
    78	images that do not align with the query description still share
    79	similar CLRs with the targeted images. This indicates a heavy
    80	
    81	2162-237X © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
    82	See https://www.ieee.org/publications/rights/index.html for more information.
    83	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
--
    85	5098
    86	
    87	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
    88	
    89	Fig. 2. Illustration of some feature response maps on pairs of RGB and GRS
    90	pedestrian images, which are obtained via two single-branch basic models are
    91	trained, respectively, on RGB and GRS data. By averaging along the channel
    92	dimension, we compute the feature response maps for all the images. In each
    93	column (numbered from 1 to 7), the first and third rows are, respectively, the
    94	RGB and GRS images while the second and fourth ones are the corresponding
    95	RGB and GRS response maps.
    96	
    97	reliance on CLR information in existing text-based person
    98	retrieval methods. However, the model may ignore subtle yet
    99	discriminative cues, such as the presence of items like “a
   100	camera bag,” “a cross-body bag,” or “high top shoes,” leading
   101	to retrieval failures. In certain cases, an image may still rank
   102	at the top of the result list by the model, even if the CLRs in
   103	local regions do not accurately match the given textual query.
   104	While CLR information undoubtedly plays a significant role
   105	in computing cross-modal affinity, the over-reliance on CLR
   106	information can divert the attention of the model from other
   107	crucial visual cues such as texture and structural information, ultimately resulting in suboptimal retrieval performance.
   108	A typical case can be observed in the second example from
   109	Fig. 1. In this instance, the CLR of the T-shirt in the target
   110	images is a very pale shade of blue, almost akin to gray.
   111	However, the query text merely states “a blue t-shirt” without
   112	specifying its lightness. Consequently, a single-branch basic
   113	model tends to retrieve numerous unrelated images where the
   114	clothing is distinctly blue. This mismatch in CLR description
   115	consequently results in a failed retrieval case. Therefore,
   116	alleviating the problem of CLR over-reliance can be a crucial
   117	factor in further promoting future research. To be more clear,
   118	two single-branch basic models are, respectively, trained on
   119	RGB and grayscale (GRS) data, by which the feature response
   120	maps are generated on pairs of RGB and GRS person images
   121	and illustrated in Fig. 2. As can be seen, for the RGB and
   122	GRS data, the attention of the models are drawn to varied
   123	local parts, implying that the complementary effects between
   124	
   125	the RGB and GRS data could be taken advantage and hence
   126	the problem of CLR over-reliance could be alleviated.
   127	To this end, in this article, a novel framework is designed
   128	to Excavate All-round Information Beyond Color for the textbased person retrieval task, which is therefore termed EAIBC.
   129	Specifically, to address the problem of CLR over-reliance,
   130	we introduce a jointly optimized multibranch architecture
   131	consists of four branches, namely an RGB branch, a GRS
   132	branch, a high-frequency (HFQ) branch and a CLR
   133	branch. The GRS branch employs a color deprivation module
   134	(CDM) to obtain GRS images, while the color masking module (CMM) masks the words related to CLR information in
   135	textual descriptions. This ensures that the GRS branch focuses
   136	on non-CLR clues in retrieving. Besides, within an image, the
   137	low-frequency information cares more about appearance and
   138	CLR, while the HFQ information majorly attends to details
   139	like texture and contour information. Therefore, for the HFQ
   140	branch, a high-frequency extraction module (HEM) is adopted
   141	to obtain the HFQ information from the input raw image,
   142	which enables EAIBC to explicitly key cues like textural and
   143	structural information other than CLR. Furthermore, in order
   144	to ensure that EAIBC fully utilizes all-round information in
   145	an effective and balanced way, and does not overly emphasize certain information while ignoring others, we include a
   146	CLR branch which is specifically designed to focus on CLR
   147	information. Additionally, a mutual learning (ML) mechanism [1] has been implemented to enable the four branches
   148	to communicate with and learn from each other. We evaluate
   149	our proposed method on three text-based person retrieval
   150	datasets, namely CUHK-PEDES [2], ICFG-PEDES [3] and
   151	RSTPReid [4]. Our experimental results show that EAIBC
   152	outperforms existing methods and achieves state of the art
   153	(SOTA) performance in supervised [2], weakly supervised
   154	[5] and cross-domain [6] text-based person retrieval tasks.
   155	To sum up, the major contributions of this article include
   156	the following.
   157	1) This article proposes a jointly optimized multibranch
   158	architecture termed as EAIBC to Excavate All-round
   159	Information Beyond Color and address the problem
   160	of CLR over-reliance. The framework includes four
   161	branches including an RGB branch, a GRS branch,
   162	an HFQ branch, and a CLR branch.
   163	2) An ML mechanism is introduced to facilitate communication and learning among the four branches, which
   164	allows for an effective and balanced use of all-round
   165	information.
   166	3) To our knowledge, this article is the first to use GRS
   167	data in addition to RGB data to improve performance in
   168	text-based person retrieval.
   169	4) The experimental results on CUHK-PEDES, ICFGPEDES and RSTPReid, as well as extensive ablation
   170	analysis, demonstrating that EAIBC outperforms existing methods and achieves SOTA performance in
   171	supervised [2], weakly supervised [5], and cross-domain
   172	[6] text-based person retrieval tasks.
   173	This work is an extension of our previous ACM MM 2022 conference paper CAIBC [7]. The contributions of this article over
   174	CAIBC can be concluded as follows.
   175	
   176	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   177	
   178	ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL
   179	
   180	1) An HFQ branch is proposed to explicitly care for
   181	extreme signals like texture and contour information
   182	within a person image.
   183	2) In addition to CUHK-PEDES and RSTPReid, the
   184	ICFG-PEDES dataset is further utilized to evaluate the
   185	performance of EAIBC.
   186	3) EAIBC is further evaluated in the cross-domain textbased person retrieval setting to validate its generalization ability and robustness.
   187	4) A more comprehensive quantitative analysis is carried
   188	out to systematically demonstrate the effectiveness of
   189	the proposed components within EAIBC.
   190	5) After visualization of the feature response maps along
   191	with text-based person retrieval examples given by
   192	EAIBC, a detailed discussion is carried out to understand
   193	the mechanism behind EAIBC. This analysis identifies
   194	some of the current challenges in text-based person
--
   285	5100
   286	
   287	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
   288	
   289	Fig. 3. Illustration of the EAIBC framework, which consists of RGB, GRS, HFQ, and CLR branches. The terms CDM, HEM, CMM, and CPM, respectively,
   290	denote color deprivation module, high-frequency extraction module, color masking module, and color prior module.
   291	
   292	by Liu et al. [29]. By means of the semantic scene graphs, the
   293	structural features of the visual and textual data are modeled.
   294	Aggarwal et al. [30] propose cross modal attribute aided
--
   316	reasoning and aligning (IRRA) framework, which focuses
   317	on learning relations between local visual-textual tokens and
   318	enhancing global image-text matching without the need for
   319	additional prior supervision. To be specific, Lapscore [36]
   320	enhances the model’s ability to capture CLR information
   321	
   322	and fine-grained cross-modal association by introducing two
   323	subtasks on CLR reasoning, namely text completion and image
   324	colorization (TC&IC), during training. On the contrary, our
   325	proposed EAIBC method aims to address the issue of CLR
   326	over-reliance by utilizing other important cues.
   327	III. M ETHODOLOGY
   328	A. Problem Formulation
   329	The proposed framework, as illustrated in Fig. 3, aims to
   330	measure the affinity between textual queries and person images
--
   339	B. Branch Prototype
   340	To begin with, a branch prototype that extracts and aligns
   341	the multimodal multigranular representations is introduced.
   342	This prototype can then be used for implementing the RGB,
   343	GRS and HFQ branches in EAIBC.
   344	a) Visual representation extraction: To obtain the global
   345	and local visual representations from an input image, a pretrained ResNet-50 [37] backbone is utilized. For the global
   346	granularity, a global max pooling operation is employed to
   347	down-scale the feature map before the last pooling layer of
--
   401	
   402	(2)
   403	
   404	C. Color Deprivation and Masking
   405	To obtain the corresponding GRS image Igrs for a given
   406	RGB image Irgb ∈ R3×H ×W , we propose a CDM that can be
   407	formulated as follows:
   408	
   409	
   410	
   411	 R(i, j)
   412	GRS(i, j) = 0.299 0.587 0.114 G(i, j)
   413	(3)
   414	B(i, j)
   415	where i ∈ {1, 2, . . . , H }, j ∈ {1, 2, . . . , W }. GRS(i, j),
   416	R(i, j), G(i, j) and B(i, j) stand for the values of the pixels
   417	in the ith row and the jth column of the GRS and RGB
   418	channels, respectively. The empirical coefficients 0.299, 0.587,
   419	and 0.114 are commonly adopted in image processing software
   420	such as OpenCV and Photoshop for converting RGB images
   421	to GRS. In order to using the ResNet-50 visual backbone to
   422	deal with the obtained GRS images, we replicate the singlechannel GRS images three times along the dimension of the
   423	channel.
   424	For removing CLR information from the textual modality,
   425	a CMM is designed. After collecting the words related to
   426	CLR information that appear with HFQ from the textual query
   427	corpus, a CLR bank is formed. The frequency-based CLRrelated word cloud is displayed in Fig. 4. Consequently, when a
   428	textual query sentence is given, all of the words related to CLR
   429	are converted into a masking token [CLR]. For instance, given
   430	
   431	Fig. 4. Frequency-based CLR-related word cloud constructed from the CLR
   432	bank.
   433	
   434	a textual query “This man is wearing an orange coat with a
   435	gray-orange backpack. He has light blue jeans and black-white
   436	sneakers”, the output of CMM will be “This man is wearing
   437	an [CLR] coat with a [CLR] backpack. He has light [CLR]
   438	jeans and [CLR] sneakers.”
   439	It is worth noting that both CDM and CMM do not require
   440	learning any parameters, which can be directly utilized to
   441	handle the input raw multimodal data.
   442	D. High-Frequency Information Extraction
   443	To extract HFQ components from a raw image I rgb ∈
   444	R
   445	and obtain the corresponding HFQ image I hfq ∈
   446	3×H ×W
   447	R
   448	, an HEM is employed. In practice, HEM can be
   449	implemented in varied manners, and in this article the Gaussian filter is utilized for extracting the HFQ information.
   450	3×H ×W
   451	
   452	E. Excavating All-Round Information via Multibranch
   453	Architecture
--
   486	rgb
   487	rgb
   488	||vg ||||tg ||
   489	||vl ||||tl ||
   490	2) GRS Branch: The GRS branch accepts a GRS image
   491	obtained through CLR deprivation and a textual description
   492	after masking the words related to CLR information. It also
   493	grs
   494	grs
   495	grs
   496	grs
   497	produces vg / tg ∈ R P and vl / tl ∈ R K P , just like the
   498	RGB branch. The similarities for the GRS branch are then
   499	obtained by
   500	grs T grs
   501	grs T grs
   502	vg
   503	tg
   504	vl
   505	tl
   506	grs
   507	grs
   508	Sg = grs
   509	(5)
   510	grs , Sl =
   511	grs
   512	grs .
   513	||vg ||||tg ||
   514	||vl ||||tl ||
   515	3) HFQ Branch: The HFQ branch processes the obtained
   516	HFQ image and the original textual description. It generates
   517	hfq
   518	hfq
   519	hfq
   520	hfq
   521	vg / tg ∈ R P and vl / tl ∈ R K P , similar to the RGB
   522	
   523	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   524	
   525	5102
   526	
   527	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
   528	
   529	branch. The similarities for the HFQ branch can be computed
   530	as
   531	 T
   532	 T
   533	hfq
   534	hfq
   535	hfq
   536	hfq
   537	vl
   538	tl
   539	tg
   540	vg
   541	hfq
   542	hfq
   543	, Sl = hfq
   544	.
   545	(6)
   546	Sg = hfq
   547	hfq
   548	hfq
   549	||vg ||||tg ||
   550	||vl ||||tl ||
   551	4) CLR Branch: As stated in Section I, the aim of EAIBC
   552	is to ensure an effective and balanced use of all available
   553	information, instead of prioritizing some information while
   554	ignoring others. Therefore, it is undesirable for the model to
   555	excel at extracting non-CLR clues but fail to attend to critical
   556	CLR information. To ensure that CLR information is not
   557	neglected, we employ the CLR branch which focuses explicitly
   558	on clues related to CLR. Intuitively, the RGB branch is
   559	responsible for capturing general information, while the GRS
   560	branch is responsible for capturing other crucial details beyond
   561	CLR. As a result, the information that is not shared between
   562	these two branches can be considered pure information about
   563	CLR. In the visual modality, we subtract the output feature
   564	map Mgrs of the visual backbone within the GRS branch from
   565	Mrgb , the output feature map of the RGB branch, to obtain
   566	a pure CLR feature map Mclr . Then, we apply global max
   567	pooling and a fully-connect layer to Mclr to obtain the visual
   568	CLR representation vclr ∈ R P . And in the textual modality,
   569	we propose a color prior module (CPM). We first subtract the
   570	masked sentence representation Egrs from the representation
   571	Ergb of the whole sentence to obtain Eclr . And then we use
   572	the words related to CLR in the sentence as a CLR prior to
   573	enhancing the CLR information in Eclr . To do so, we sum
   574	the word embeddings of the CLR-related words and convert
   575	them to the same dimension as the word representations eiclr
   576	in Eclr . We add this CLR prior to each eiclr to obtain a CLR
   577	prior enhanced representation Ecp . Finally, we apply an RMP
   578	operation and a fully-connect layer to Ecp to obtain the textual
   579	CLR representation tclr ∈ R P . Finally, the similarity score for
   580	the CLR branch is computed by
   581	T
   582	vclr tclr
   583	clr
   584	S = clr
   585	.
   586	(7)
   587	||v ||||tclr ||
   588	F. Optimization
   589	As mentioned previously, the major target of EAIBC is
   590	to simultaneously optimize the four branches and utilize
   591	all-round information in an effective and balanced manner.
   592	Therefore, an ML mechanism is used to enable knowledge sharing among the branches. To achieve this, the
   593	Kullback–Leibler (KL) Divergence is used to measure the
   594	affinity between the logits of varied branches. Let vibr ∈
   595	rgb grs hfq
   596	rgb grs hfq
   597	{vgi , vgi , vgi , viclr } and tibr ∈ {tgi , tgi , tgi , ticlr } represent the
   598	visual and textual features extracted from the ith sample pair,
   599	br
   600	they are then utilized to compute the probabilities pbr
   601	vm and ptm
--
   620	br
   621	k=1 exp γ Wk ti
   622	
   623	where γ Wm vibr and γ Wm tibr are logits input to the softmax
   624	layer in the br branch, and br can be rgb, grs, hfq or clr.
   625	Under the constraint of an ML loss, each branch is optimized,
   626	which can be calculated following:
   627	
   628	
--
   655	br
   656	p
   657	tm ti
   658	s∈B/{br } i=1 m=1
   659	where B = {rgb, grs, hfq, clr}.
   660	Furthermore, to train EAIBC, we also utilize the triplet
   661	ranking loss and ID loss, which are combined to form the
   662	ReID loss. It is worth noting that the ID loss is applied to
   663	hfq
   664	hfq
   665	grs
   666	grs
   667	rgb
   668	rgb
   669	each local representation v f k / t f k , v f k / t f k , and v f k / t f k ,
   670	grs
   671	rgb
   672	rgb
   673	whereas the triplet ranking loss is applied to vl / tl , vl /
   674	grs
   675	hfq
   676	hfq
   677	tl , and vl / tl .
   678	IV. E XPERIMENTS
   679	A. Experimental Setup
   680	1) Dataset: The proposed EAIBC is trained and evaluated
--
   709	Rank-k accuracy stands for the probability of finding at least
   710	one matching person image within the top-k candidate list
   711	when given a textual description as a query. The evaluation
   712	is conducted by reporting the Rank-1/5/10 accuracies for all
   713	ablation experiments.
   714	
   715	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   716	
   717	ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL
--
   723	
   724	B. Implementation Details
   725	For implementing EAIBC, the dimension P for the representations is set to 2048. Two independent pretrained
   726	ResNet-50 [37] models are selected for the visual backbones
   727	within the RGB and GRS branches. The word embedding
   728	dimension is 768 or 500 with or without the pretrained bidirectional encoder representations for transformers (BERT) [39].
   729	Images from all of the datasets are resized to 384 × 128 × 3.
   730	Random horizontal flipping along with the CLR jittering
   731	strategies are adopted for augmenting the visual data. To optimize the parameters of EAIBC, an Adam optimizer [40] is
   732	used. We set the initial learning rate for the visual backbones
   733	as 0.0001 while 0.001 for other parameters of EAIBC. The
   734	batch size is 64 and the training process lasts for 100 epochs.
   735	C. Ablation Analysis
   736	To evaluate the contribution and effectiveness of proposed
   737	components in EAIBC, comprehensive ablation experiments
   738	are conducted on the CUHK-PEDES and RSTPReid datasets.
   739	The results are reported in Table I and numbered from 1 to 29.
   740	Abl.1, Abl.2, and Abl.3 are conducted using a single
   741	RGB, GRS, and HFQ branch basic model (as described in
   742	Section III-B). Abl.4–Abl.9 are conducted without the CLR
   743	branch, and V or T in Table I indicate that the GRS branch
   744	is used only for the visual or textual modality. Abl.10–Abl.16
   745	are performed after adding the CLR branch, and Abl.28/29
   746	
   747	are carried out with the complete EAIBC model without/with
   748	a pretrained BERT model [39] for word embedding. In
   749	Abl.17–Abl.27, we analyze the performance of our proposed
   750	EAIBC method on the task of weakly supervised person
   751	retrieval. The Rank-1, Rank-5, and Rank-10 accuracies are
   752	reported for all ablation experiments.
   753	1) Impact of Multibranch Learning: As shown in Table I,
   754	while a single RGB branch model may outperform individual
   755	GRS or HFQ branch models, combining any two of these
   756	branches leads to a notable improvement in performance.
   757	Furthermore, adding the CLR branch, which specifically
   758	focuses on CLR information, further improves the retrieval
   759	performance. These observations suggest that the jointly optimized multibranch architecture in EAIBC allows for separate
   760	consideration of varied types of information from different
   761	perspectives, and the four branches are enabled to complement
   762	each other for superior retrieval performance.
   763	2) Impact of Mutual Learning: Comparing Abl.6 with
   764	Abl.7 in Table I, we observe that by utilizing the ML mechanism, the retrieval accuracies are improved by 0.91%, 0.42%,
   765	0.37% and 0.95%, 0.35% 0.40% under the Rank-1/5/10
   766	accuracies, respectively, on CUHK-PEDES and RSTPReid.
   767	Similarly, comparing Abl.13 with Abl.11, we observe a performance increase of 0.65%, 0.47%, 0.17% and 1.30%, 0.80%,
--
   781	to facilitate knowledge exchange across different branches.
   782	3) Impact of Local Part Number K : The number K of
   783	local parts and the way in which they are partitioned are
   784	also of significance for the proposed model to achieve an
   785	accurate retrieval performance. Therefore, we carry out comprehensive ablation experiments to analyze the impact of the
   786	choice of K and the partitioning paradigm. On either the
   787	CUHK-PEDES or the RSTPReid dataset, there are totally
   788	26 different partitioning paradigms employed with K varies
   789	from 2 to 12 for ablation analysis. The experimental results
   790	are reported in Table II. First, it can be observed from the
   791	results that the horizontal partitioning strategy with K equals
   792	to 6 and 8 are, respectively, the best choice for the CUHKPEDES and RSTPReid datasets. Second, compared with the
   793	horizontal partitioning strategy, partitioning the feature map
--
   809	local clues and may even become noise information. Thus,
   810	it seems that the horizontal partitioning strategy with a proper
   811	local part number K best meets the nature of the text-based
   812	person retrieval task.
   813	4) Impact of Color Prior: Comparing several pairs of
   814	experiments such as (Abl.10, Abl.11), (Abl.12, Abl.13),
   815	(Abl.14, Abl.15) and (Abl.16, Abl.28), it can be seen that
   816	taking the CLR prior information into consideration leads to
   817	a further improvement in the performance. This observation
   818	confirms the effectiveness of the CLR prior information to
   819	enhance the CLR branch.
   820	5) Comparison Between KL Divergence and JS Divergence
   821	for Mutual Learning: As mentioned in Section III-E4, for ML,
   822	the KL Divergence is used to quantify the affinity of logits
   823	from varied branches. For two probabilities p1 and p2 , the KL
   824	distance from p1 to p2 is calculated as
   825	DKL (p2 ||p1 ) =
--
   878	Table III. It can be noticed that utilizing either the asymmetric KL divergence or the symmetric JS divergence for ML
   879	achieves similar retrieval performance, with the utilization of
   880	KL divergence slightly better.
   881	D. Comparison With SOTA And Discussion on Superiority of
   882	EAIBC for Alleviating Color Over-Reliance Problem
   883	We present a comparison of EAIBC with previous
   884	approaches on CUHK-PEDES, ICFG-PEDES and RSTPReid
   885	in Tables IV–VI, respectively. It can be observed that EAIBC
   886	achieves SOTA performance on all of the datasets by excavating all-round information beyond CLR. To be specific,
   887	Lapscore [36] enhances the model’s ability to capture CLR
   888	information and fine-grained cross-modal association by introducing two subtasks on CLR reasoning, namely TC&IC,
   889	during training. On the contrary, our proposed EAIBC method
   890	
   891	aims to address the issue of CLR over-reliance by utilizing
   892	other important cues.
   893	Furthermore, we evaluate EAIBC for weakly supervised
   894	person retrieval [5] and compare it with existing methods
   895	in Table VII as well. Notably, EAIBC achieves competitive
   896	or even superior performance than existing SOTA approaches
   897	without using any clustering or pseudo label generation-based
   898	techniques, indicating that EAIBC effectively addresses the
   899	problem of CLR over-reliance in person retrieval.
   900	To further explore this issue, it is worth noting that in the
   901	conventional task of supervised text-based person retrieval, the
   902	problem of CLR over-reliance can be partially mitigated by
   903	using identity annotations for each multimodal sample pair
   904	as prior knowledge. More specifically, when the proposed
   905	algorithm struggles to differentiating between images of pedestrians that have similar CLR characteristics and tends to rely
   906	too much on CLR cues, the prior information about person
   907	
   908	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
   909	
   910	5106
--
   914	TABLE VIII
   915	P ERFORMANCE C OMPARISONS ON CUHK-PEDES IN THE C ROSS -D OMAIN S ETTINGS
   916	
   917	identity can assist the algorithm in determining whether the
   918	two samples are similar in CLR belong to the same person
   919	or not. However, relying solely on the annotation of identity
   920	to address the problem of CLR over-reliance is insufficient.
   921	While the identity annotation can help the algorithm differentiate two CLR-similar samples, it cannot pinpoint the exact
   922	details that the model has overlooked. Therefore, the positive
   923	impact of the identity annotation is limited during the training
   924	process.
   925	Furthermore, the problem of CLR over-reliance can get
   926	even more serious when there is no accessible information
   927	about person identity and only pairwise relationships, as in the
   928	weakly supervised case. Many previous works have proposed
   929	pseudo label generating-based approaches to address this task,
--
   931	up for lacking identity clues. However, since these approaches
   932	still rely on some form of prior knowledge, most of which
   933	are even less precise than directly annotated labels for person
   934	identity, they are unlikely to effectively alleviate the problem
   935	of CLR over-reliance.
   936	To summarize, as discussed above, the majority of previous
   937	either supervised or weakly supervised approaches depend on
   938	some form of prior knowledge to inform the algorithm if
   939	it is affected by the problem of CLR over-reliance or not,
   940	with no further detailed guidance provided. These paradigms
   941	are not so direct and therefore just lead to suboptimal
   942	retrieval performance. In contrast to existing methods, our
   943	proposed EAIBC method adopts a multibranch architecture
   944	that explicitly handles different types of information. The GRS
   945	branch focuses on multimodal data from which the CLRrelated information is removed, forcing the model to rely on
   946	other discriminative cues. The HFQ branch is designed to
   947	consider the HFQ information such as texture and contour,
   948	while the CLR branch explicitly considers information about
   949	CLR. Besides, for processing the original multimodal data,
   950	the RGB branch is proposed. The four branches in EAIBC
   951	have distinct contributions that guide and complement each
   952	other, facilitating the extraction and alignment of multimodal
   953	information. Consequently, the need for relying solely on prior
   954	knowledge to address the problem of CLR over-reliance will
   955	be greatly reduced. The branches can work together to identify
   956	and fill in gaps in information that may be missed by one
   957	branch alone. Consequently, the EAIBC model is able to
   958	focus on sufficient discriminative clues without solely relying
--
   960	on prior knowledge to identify key information. As a result,
   961	the experimental results presented in Table VII demonstrate
   962	that EAIBC outperforms existing approaches for weaklysupervised person retrieval, without relying on clustering or
   963	pseudo-label generating paradigms.
   964	E. Performance Comparisons in Cross-Domain Settings
   965	We evaluated the generalization ability of EAIBC on
   966	CUHK-PEDES with the cross-domain settings proposed in
   967	moment alignment network (MAN) [6]. The dataset was
   968	divided into five domains based on the source of the image:
   969	CUHK01 [62], CUHK03 [63], Market-1501 [64], SSM [65],
   970	and VIPER [66]. SSM (S) is selected as the source domain,
--
   976	labeled source data and unlabeled target data. The results,
   977	shown in Table VIII, demonstrate that EAIBC outperforms
   978	all existing approaches in both SO and ST settings, indicating
   979	the robustness of our proposed method for text-based person
   980	retrieval in cross-domain scenarios.
   981	F. Visualization of Feature Response Maps
   982	We visualize and compare the feature response maps from
   983	RGB and GRS branches in Fig. 5 using a complete EAIBC
   984	model. Fig. 5 displays a total of 32 examples, with 8 ×
   985	4 examples shown. By averaging along the channel dimension,
   986	we compute the feature response maps for all the images.
   987	In each example, the first and third columns are, respectively,
   988	the RGB and GRS images while the second and fourth ones
   989	are the corresponding RGB and GRS response maps.
   990	It can be observed that the RGB and GRS branches usually
   991	care for different regions. When some discriminative clues are
   992	neglected by one certain branch of EAIBC, it would be caught
   993	by the other branch and stilled taken into consideration for a
   994	robust retrieval. Taking the sample in the first row and the first
   995	column of Fig. 5 (referred to as R1C1) as an example, we can
   996	see that the RGB branch mainly focuses on the part of the
   997	“pink shirt,” which is exactly the most salient region in terms
   998	of CLR information in the RGB image. On the contrary, the
   999	GRS branch tends to focus on the “bicycle” region instead.
  1000	
  1001	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
  1002	
  1003	ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL
  1004	
  1005	5107
  1006	
  1007	Fig. 5. We visualize and compare the feature response maps from RGB and GRS branches of a complete EAIBC model. By averaging along the channel
  1008	dimension, the feature response maps for all the images are computed. In each example, the first and third columns are, respectively, the RGB and GRS
  1009	images while the second and fourth ones are the corresponding RGB and GRS response maps.
  1010	
  1011	For R4C1, it can be noticed that the major attention of the
  1012	RGB branch is focused on the region of the “pink track pants”
  1013	while the attention of the GRS branch is focused on the part
  1014	related to “a dark bag over her left shoulder.” And for R2C4,
  1015	it can be observed that the RGB branch pays its attention to
  1016	the “white printed folded long sleeves shirt,” while the GRS
  1017	branch focuses more on the “suitcase” related region. Many of
  1018	the other examples are also consistent with this observation,
  1019	which further proves the complementary effect between the
  1020	multiple branches proposed in EAIBC. Consequently, EAIBC
  1021	benefits from this complementary effect and is able to better
  1022	solve the problem of CLR over-reliance.
  1023	Nevertheless, it is not always the case that the RGB and
  1024	GRS branches of EAIBC would care for totally different parts
  1025	of the images. Some typical examples like R4C3, R4C4 and
  1026	R5C3 are illustrated to explain this point. It is reasonable that
  1027	the attention of the two branches would sometimes overlap
  1028	with each other, especially when some of the fine-grained clues
--
  1074	green and red borders, respectively. (a) Results on CUHK-PEDES. (b) Results on RSTPReid.
  1075	
  1076	in a given image. Therefore, taking R5C4 as an example,
  1077	for the nonpedestrian occlusion case, both the RGB and
  1078	GRS branches can care for the parts related to the person.
  1079	Although the attention of the RGB branch covers extra regions
  1080	besides the person, as the GRS branch focuses exactly on
  1081	the pedestrian, the complete EAIBC can still pay its major
  1082	attention to the person-related parts due to the complementary
  1083	effect between the branches. However, when it comes to the
  1084	Nontarget Pedestrian Occlusion case, the situation becomes
--
  1095	pedestrians in the image theoretically have equal chances to
  1096	be chosen. For some of the examples like R6C3, R8C3 and
  1097	R8C4, the two branches focus on different persons in the
  1098	image, while for examples like R6C4, R7C3 and R7C4, the
  1099	attention of both the RGB and the GRS branches are somehow
  1100	paid to the averaged region of the multiple pedestrians. All
  1101	these mentioned situations may distract the proposed model
  1102	from the targeted person, and hence give a suboptimal retrieval
  1103	performance. Therefore, the nontarget pedestrian occlusion
--
  1106	
  1107	G. Illustration of Text-Based Person Retrieval Examples by
  1108	EAIBC
  1109	To further demonstrate the effectiveness of EAIBC which
  1110	excavates all-round information beyond CLR, more text-based
  1111	person retrieval examples are illustrated in Fig. 6(a) and (b),
  1112	which are, respectively, generated on the CUHK-PEDES and
  1113	RSTPReid datasets. Note that for each of the examples, the
  1114	first row is obtained by a single RGB branch baseline model,
--
  1116	model. The matched person images in the retrieved lists are
  1117	marked with green borders, and the mismatched ones are
  1118	marked with red borders.
  1119	It can be observed that in these typical examples, the single
  1120	RGB branch baseline model suffers from the CLR overreliance problem and ends up in failed cases. Specifically,
  1121	quite a few untargeted images ranked high in the result lists
  1122	of the single RGB branch baseline model just well meet
  1123	the CLR description in the textual query and have similar
  1124	CLRs with the matched images. Many of these mismatched
  1125	images only have tiny but crucial difference with the matched
  1126	ones. And in some hard cases for the single RGB branch
  1127	model, although the CLRs in images are corresponding to
  1128	the wrong local parts, it still ranks high. On the contrary,
  1129	however, the problem of CLR over-reliance will not be an
  1130	obstacle for EAIBC. Regardless of the failure of the single
  1131	RGB branch model on these cases, the complete EAIBC model
  1132	gives different and better performance. It is worth noting that
  1133	as CLR information is undoubtedly of great significance for
  1134	text-based person retrieval, many images besides the targeted
  1135	ones ranked high in the result lists of EAIBC are also still
  1136	
  1137	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:36 UTC from IEEE Xplore. Restrictions apply.
  1138	
  1139	ZHU et al.: IMPROVING TEXT-BASED PERSON RETRIEVAL
  1140	
  1141	generally similar in CLR with the targeted ones. However,
  1142	the target images with more subtle local parts matched with
  1143	the queries are ranked higher than the untargeted ones as the
  1144	complementary effects among the four proposed branches can
  1145	effectively excavate all-round information beyond CLR and
  1146	alleviate the problem of CLR over-reliance.
  1147	V. C ONCLUSION
  1148	Text-based person retrieval is the process of searching
  1149	a massive visual resource library for images of a particular pedestrian, based on a textual query. However, existing
  1150	approaches often rely too heavily on CLR information, which
  1151	can result in a suboptimal person retrieval performance due
  1152	to the neglect of other important clues such as texture
  1153	and structure information. To address this problem of CLR
  1154	over-reliance, we propose a jointly optimized multibranch
  1155	architecture called EAIBC which includes an RGB branch,
  1156	a GRS branch, an HFQ branch, and a CLR branch. To balance
  1157	and effectively utilize all types of information, we employ
  1158	an ML mechanism that allows the branches to communicate
  1159	and learn from one another. Our proposed EAIBC method
  1160	achieves SOTA performance on supervised, supervised and
  1161	cross-domain text-based person retrieval tasks on the CUHKPEDES, ICFG-PEDES, and RSTPReid datasets.
  1162	R EFERENCES
  1163	[1] Y. Zhang, T. Xiang, T. M. Hospedales, and H. Lu, “Deep mutual
  1164	learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.,
  1165	Jun. 2018, pp. 4320–4328.
--
  1174	2021, pp. 209–217.
  1175	[5] S. Zhao, C. Gao, Y. Shao, W.-S. Zheng, and N. Sang, “Weakly
  1176	supervised text-based person re-identification,” in Proc. IEEE/CVF Int.
  1177	Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 11375–11384.
  1178	[6] Y. Jing, W. Wang, L. Wang, and T. Tan, “Cross-modal cross-domain
  1179	moment alignment network for person search,” in Proc. IEEE/CVF Conf.
  1180	Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020, pp. 10675–10683.
  1181	[7] Z. Wang et al., “CAIBC: Capturing all-round information beyond
  1182	color for text-based person retrieval,” in Proc. 30th ACM Int. Conf.

exec
/bin/zsh -lc "pdftotext 'Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf' - | nl -ba | rg -i -C 4 'ablation|visualization|controllable|GCP|LoRA|Adapter|ControlNet|CLIP Score|FID|t-SNE|pre-training|quality'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     5	Pattern Recognition
     6	journal homepage: www.elsevier.com/locate/pr
     7	
     8	Identity-aware infrared person image generation and re-identification via
     9	controllable diffusion model
    10	Xizhuo Yu a , Chaojie Fan b ,∗, Zhizhong Zhang c , Yongbo Wang c , Chunyang Chen a ,
    11	Tianjian Yu a , Yong Peng b
    12	a
    13	
--
    29	ABSTRACT
    30	Visible–infrared person re-identification (VI-ReID) aims to learn the identity-aware features between visible and
    31	infrared person images. However, most works rely on two publicly available datasets, 𝑖.𝑒., SYSU-MM01 and
    32	RegDB, which is limited by the limited amount of training data and the lack of rich scenes and perspectives.
    33	In this paper, we propose a controllable diffusion framework for infrared person image generation and reidentification. Our approach is beyond the existing diffusion model in two perspectives: (1) we use LoRA
    34	to fine-tune the existing diffusion models with VI-ReID dataset and therefore it helps the diffusion model
    35	understand the infrared modality. A text adapter is then utilized to transfer the semantic understanding ability
    36	of Large Language Model (LLMs) to our generation models; (2) we design a controllable generation module
    37	to make the generated person images, from the same textual description, identity-aware. After meticulous
    38	post-processing operations, our approach is capable of producing diverse visible and infrared person images,
    39	allowing for improving the discrimination of existing VI-ReID model without any annotations. We expand
    40	the VI-ReID dataset with our generated images, and conduct extensive experiments on VI-ReID models.
--
    70	the diffusion model to generate the sufficient training samples. But as
    71	shown in Fig. 1(a), it appears that existing pre-trained diffusion model,
    72	e.g., Stable Diffusion, which may learned from a plenty of colored images, cannot understand the infrared modality and therefore even given
    73	the prompt ‘‘an infrared photo’’, it still generates the colored images.
    74	Another nonnegligible issue is the diffusion process is uncontrollable
    75	
    76	∗ Corresponding author.
    77	
    78	E-mail address: fcjgszx@csu.edu.cn (C. Fan).
--
    85	
    86	X. Yu et al.
    87	
    88	Fig. 1. Illustration of our proposed method. (a) shows that our method performs better in semantic understanding than Stable Diffusion v1.5 and is capable of generating infrared
    89	images. (b) exhibits controllable generation results for VI-ReID with our method.
    90	
    91	such that it is unable to generate the identity-aware infrared person
    92	images.
    93	In this paper, we propose a controllable diffusion framework for
    94	infrared person image generation and re-identification. In our approach, we employ commonly used and pre-trained Stable Diffusion
    95	model to generate both visible and infrared person images with text
    96	descriptions. Towards this goal, we first use LoRA [3] to fine-tune the
    97	existing diffusion models with current VI-ReID dataset, which helps
    98	the model learn the concept of ‘‘infrared images’’. After that, we
    99	propose a text adapter to transfer the semantic understanding ability
   100	of LLMs to the fine-tuned diffusion models, and allows us to describe
   101	the person images with satisfactory text prompts. Then, we design a
   102	controllable generation module to ensure that generated person images
   103	from the same textual description are identity-aware, such that it
   104	enables us to learn a discriminative VI-ReID model. After meticulous
   105	post-processing operations, our image generation model is capable of
   106	producing diverse visible and infrared person images based on textual
--
   130	to strengthen the consistency of cluster features.
   131	
   132	• We propose a new cross-modal person image generation framework, which is capable of producing diverse visible and infrared
   133	person images with the help of fine-tuned Stable Diffusion model.
   134	• We design a controllable module to ensure that the generated
   135	person images is identity-aware, and therefore allowing for improving the discrimination of existing VI-ReID model.
   136	• We construct a new VI-ReID dataset, called GCP, using text-toimage generation instead of real camera capture, thereby diminishing privacy concerns.
   137	• We conduct extensive experiments with the SYSU-MM01 and
   138	RegDB datasets, which demonstrate the effectiveness of our
   139	method.
   140	
--
   164	gap between supervised and unsupervised VI-ReID.
   165	
   166	3.2. Diffusion model fine-tuning via infrared images
   167	Using existing diffusion models in generating the cross-modal person photos may fail because of their limitations in generating infrared
   168	photos and high costs for training. To remedy this issue, we use LoRA,
   169	a parameter-efficient fine-tuning method, to fine-tune the pre-trained
   170	Stable Diffusion. In our approach, We leverage the real-world VI-ReID
   171	dataset SYSU-MM01 as image training dataset, and get corresponding
   172	text training dataset by BLIP. As a result, the fine-tuned diffusion model
   173	can generate high-quality visible and infrared person photos based on
   174	appropriate text prompts.
   175	As shown in Fig. 2, the fine-tuned diffusion model is comprised of
   176	a pretrained autoencoder, a text conditional encoder, and an image
   177	generator. In it, the autoencoder, comprised of an encoder  and a
--
   193	the transformation from the diffusion model to the conditional image
   194	generator.
   195	For learning the style of infrared images, we propose to further finetune the pre-trained diffusion model. Compared with full fine-tuning,
   196	which updates all the parameters of the pre-trained model, Low-Rank
   197	Adaptation (LoRA) is more efficient in storage and computation. For
   198	a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
   199	constrain its update:
   200	
   201	2.3. Text-to-image generation
   202	Text-to-image generation is proposed to realize image generation
--
   204	(ARM) [26] and Vector Quantized Variational AutoEncoder (VQ-VAE)
   205	Transformer-based methods [27], have made remarkable progress in
   206	text-to-image generation. Recently, diffusion models (DMs) [28,29],
   207	which leverage a denoising process, progressively refining noise to
   208	generate high-quality images through a series of iterative steps, have
   209	shown great success in image generation [30,31]. LDM [32] extends
   210	DMs to latent space and significantly alleviate computational demands.
   211	ControlNet [33] is proposed to control DMs with task-specific conditions by an additional trainable copy of pre-trained models. Imagen [34] demonstrates the efficacy of utilizing pre-trained, frozen large
   212	language models as text encoders in the context of text-to-image generation via diffusion models. SUR-adapter [35] aligns the semantic representation between simple narrative prompts and complex keywordbased prompts with the help of large language models, further improving the performance of text-to-image generation models.
   213	Despite the remarkable progress achieved in the field of text-toimage generation, the generation of images in new styles remains a
   214	challenging task, which limits its further applications. Different from
   215	methods above, we approach the adaptation of existing diffusion models from both the textual and visual perspectives to cater to the requirements of VI-ReID task.
   216	3. Method
--
   220	VI-ReID to benefit from large-scale cross-modal pretraining.
   221	To deal with this issue, we propose a generative framework for
   222	cross-modal person re-identification. Our method mainly consists of
   223	three steps: (1) Diffusion Model Fine-tuning via Infrared Images. (2)
   224	Generative Text Adapter Alignment. (3) Controllable Image Generation
   225	and Inference, which is presented in Fig. 2. Based on our generative framework, we also introduce a Generated Cross-modal Person
   226	Dataset, called GCP for further training and evaluation. Thanks to
   227	the great success of AIGC, our diffusion model enables us to generate
   228	extensive infrared person photos and therefore allows for learning the
   229	discriminative ReID model without any annotations.
   230	Our proposed cross-modal person image generation framework is
--
   232	visible and infrared images from SYSU-MM01 into BLIP to generate
   233	corresponding text prompts. Specific prefix words, like ’a RGB photo’
   234	or ’an infrared photo’, are added to the prompts, which encourages the
   235	diffusion model to recognize the concept of infrared images. We then
   236	freeze all parameters of Stable Diffusion and use LoRA to fine-tune the
   237	U-Net of Image Generator with our constructed image–text pairs. After
   238	that, we design a new text adapter to refine and align the output of
   239	the text conditional encoder with that of LLMs, and therefore enables
   240	us to generate the generation-friendly-friendly embeddings. Finally, we
   241	propose a controllable module to make the generation be identityaware when using the same text prompts. Meticulous post-processing
   242	is adopted to obtain appropriate cross-modal person images.
   243	
   244	𝑊 ′ = 𝑊0 + 𝛥𝑊 = 𝑊0 + 𝐴𝐵 𝑇
   245	
   246	(1)
   247	
   248	where 𝐴 ∈ R𝑑×𝑟 , 𝐵 ∈ R𝑘×𝑟 . During training process, only A and B are
   249	trainable parameters. In our method, LoRA is only used to optimize our
   250	image generator. We select visible and infrared photos in SYSU-MM01
   251	as the image training dataset 𝑋𝑣 = {𝑥𝑖𝑣 }, 𝑋𝑟 = {𝑥𝑖𝑟 }. Then we use BLIP
   252	to get the original text description 𝑃𝑣 = {𝑝𝑖𝑣 }, 𝑃𝑟 = {𝑝𝑖𝑟 }. Note that here
   253	we choose both visible and infrared images, targeting on teaching the
--
   258	we also modify the original text by adding the prefix words such as
   259	𝑌𝑣 = {𝑦𝑖𝑣 } =‘‘A RGB photo, [𝑝𝑖𝑣 ]’’, 𝑌𝑟 = {𝑦𝑖𝑟 } =‘‘An infrared photo, [𝑝𝑖𝑟 ]’’.
   260	Finally, we utilize both the image and text information to fine-tune
   261	Stable Diffusion model.
   262	3.3. Generative text adapter alignment
   263	Note that the text conditional encoder of the original Stable Diffusion model is CLIP ViT-L/14, which, however, is trained with text–
   264	image contrastive learning, that not specifically designed for image
   265	generation tasks. This leads to inadequate semantic understanding
   266	when guiding image creation with natural language prompts.
--
   272	
   273	X. Yu et al.
   274	
   275	Fig. 2. The pipeline of our proposed method. Left: we first use BLIP to generate some text prompts for the images from SYSU-MM01. Specific prefix words, like ’A RGB photo’
   276	or ’An infrared photo’, are added to the prompts. For model fine-tuning, we freeze the original Stable Diffusion Model and use LoRA to fine-tune it. For text adapter training,
   277	we design a text adapter to refine the output of the text encoder by leveraging the capability of large language models. Right Top: We design a two-stage strategy to generate
   278	multi-perspective RGB person images belonging to the same ID based on text descriptions. Right Bottom: we use SoftEdge to detect edge maps of generated RGB images, and then
   279	combine infrared image embeddings and edge maps to transfer RGB images to infrared images.
   280	
   281	model LLaMA [36], and save the mean value of each token embeddings
   282	generated by the LLM as the knowledge representation.
   283	To transfer the semantic understanding ability of the LLM to our
   284	fine-tuned diffusion model, we design a text adapter. We first freeze
   285	the parameters of the large language model 𝑓𝐿𝐿𝑀 , the text conditional
   286	encoder 𝑓𝐸𝑡 and the image generator 𝑓𝑔𝑒𝑛 of the fine-tuned diffusion
   287	model. Then we only optimize our text adapter. It consists of a fully
   288	connected neural networks (FCN) and a Transformer subnetwork, denoted as ℎ𝑖 . For the text description 𝑝𝑖 , we send it to both text encoders
   289	(LLM’s and diffusion model) and use the cross-attention to associate
   290	them. Formally, we define 𝑄𝑖 = ℎ1 [𝑓𝐸𝑡 (𝑝𝑖 )] and 𝐾𝑖 = ℎ2 [𝑓𝐸𝑡 (𝑝𝑖 )], and
   291	calculate an attention value as:
--
   296	visible and infrared photo, respectively. Here 𝑃 means original text
   297	descriptions from CUHK-PEDES and ICFG-PEDES. Note that we remove
   298	color keywords in 𝑃 to avoid the influence of color information for
   299	infrared photo generating. For each prompt, we first generate a RGB
   300	photo as the reference image for subsequent controllable generation.
   301	Then we use IP-Adapter [39], an effective and lightweight adapter
   302	to achieve image prompt capability for the pre-trained text-to-image
   303	diffusion models, wherein the reference image, serving as the image
   304	prompt, is fed into the adapter to generate multi-perspective RGB
   305	person images while preserving the identity. Since we have generated
   306	enough RGB photos for each identity, we utilize ControlNet [33] with
   307	SoftEdge to transfer some RGB photos to infrared photos. Specifically,
   308	we use real-world infrared photos together with edge detection map
   309	of generated RGB photos as the inputs of ControlNet, and thus we
   310	realize controllable visible and infrared person photo generating for
   311	each prompt. Finally, we generate 40 RGB photos and 40 infrared
   312	photos for each person identity, totaling about 20 000 generated photos
   313	with 257 person identities.
   314	In our endeavor to create a high-quality dataset, we have identified
   315	that the initial generation process may yield images that are not up
   316	to the desired standard. These images may be blurry, contain multiple
   317	persons, or exhibit other forms of noise, which can be attributed to
   318	the inherent limitations of the generation abilities. To address these
   319	issues, we have implemented a series of post-processing operations to
   320	enhance the quality of our dataset. Specifically, we first remove some
   321	low-quality images, like grayscale or blur images, in generated RGB
   322	images. For the generated infrared images, we have observed that some
   323	images contain color information, which is not typical for this type of
   324	imagery. To maintain the authenticity and consistency of the infrared
   325	images, we convert these images into grayscale, which ensures that all
--
   327	infrared imagery, thereby enhancing the reliability of our dataset. To
   328	further refine our dataset, we employ OpenPose [40–43], a widely used
   329	tool for human keypoint detection. With OpenPose, we can identify
   330	images that contain fewer than 18 detected key points, which typically
   331	indicate low-quality or incomplete human figures. These images are
   332	subsequently removed from our dataset. Additionally, OpenPose allows
   333	us to resize the images with tight bounding boxes, ensuring that each
   334	image focuses on the human figure without unnecessary background
   335	noise. At this point, we are capable of generating visible and infrared
--
   350	a randomly initialized matrix using Kaiming initialization, ensuring
   351	maximal preservation of the LLM’s semantic content while adjusting the
   352	dimensional alignment between 𝑓𝐿𝐿𝑀 and 𝑄. Therefore, we construct
   353	𝑉𝑖′ = 𝑉𝑖 ⊗ 𝑎𝑡𝑡𝑖 and get calibrated semantic information. Finally, through
   354	a fully connected layer, the output generated by our text adapter is
   355	transformed to 𝑐𝐿𝐿𝑀 , a representation that encapsulates the semantic
   356	interpretation capabilities inherent to the LLM, and we combine the
   357	output of the text encoder 𝑓𝐸𝑡 (𝑝𝑖 ) and our text adapter 𝑐𝐿𝐿𝑀 as the
   358	input to the generator as follows:
   359	′
   360	𝑐𝐿𝐿𝑀
   361	= 𝜂 ⋅ 𝑐𝐿𝐿𝑀 + (1 − 𝜂) ⋅ 𝑓𝐸𝑡 (𝑝𝑖 )
   362	
   363	(4)
   364	
   365	where 𝜂 is a constant hyper-parameter.
   366	3.4. Controllable image generation and inference
   367	To make the generated images more close to the real-world photos,
   368	we utilize the text description of the CUHK-PEDES dataset [37] and the
   369	ICFG-PEDES [38] dataset as the text condition for the diffusion model.
   370	We design text prompts as ‘‘a RGB photo/an infrared photo, [𝑃 ]’’ for
--
   375	X. Yu et al.
   376	
   377	Fig. 3. Utilization of large language models during the inference stage.
   378	
   379	3.5. Model pre-training
   380	To verify the validity of GCP dataset, we combine GCP with the existing training dataset and use it to pretrain CLIP-Enhanced Contrastive
   381	Learning Network (CCLNet) [21] for VI-ReID task. With the help of the
   382	visual-language representation ability from CLIP, CCLNet establishes a
   383	learnable cluster-aware prompt for person images and obtains textual
   384	descriptions allowing for subsequent unsupervised training. In addition,
--
   430	
   431	29
   432	26
   433	
   434	erated Cross-modal Person Dataset (GCP). Detailed statistics about GCP
   435	are presented in Table 1. Each identity in the dataset has approximately
   436	20 visible images and 20 infrared images, which ensures a high degree
   437	of ID consistency and diversity for subsequent ReID model training.
   438	Furthermore, we also explore the use of large language models to
--
   446	4. Experiments
   447	In this section, we conduct comprehensive experiments to evaluate our approach. First, we introduce our experimental settings and
   448	implementation details. The main results and the comparison with
   449	state-of-the-art methods are then shown in 4.2. Finally, we present the
   450	ablation studies and visualization results of our approach. To promote
   451	the culture of reproducible research, source codes and experimental
   452	results accompanying this paper will be released later.
   453	5
   454	
   455	Pattern Recognition 165 (2025) 111561
   456	
   457	X. Yu et al.
   458	
   459	model using LoRA for 5000 steps; in the second stage, we train an
   460	additional text adapter to enhance the model’s semantic understanding
   461	capability for 10,000 steps. For LoRA fine-tuning, we select visible
   462	images from cam1, infrared images from cam3 in SYSU-MM01 as
   463	the image training dataset. Notably, only the UNet of the model is
   464	finetuned while the others are frozen. For text adapter training, we
   465	utilize LLaMA2 with the size of 13B as the large language model for
   466	knowledge distillation, and set 𝜂 in Eq. (4) to 0.1. During the stage
   467	of inference and dataset generating, we use OpenCLIP-ViT-H-14 as the
   468	image encoder, and choose pretrained IP-Adapter for Stable Diffusion
   469	v1.5. For ControlNet with SoftEdge, softedge_pidinet is employed to
   470	detect edge maps. For ReID pre-training, we initially integrate our
   471	proposed dataset, 𝑖.𝑒., GCP, with the original training sets of SYSUMM01 and RegDB to construct a new training set, while the test sets
   472	remain unaltered to validate the effectiveness of our method. All our
   473	methods are implemented with PyTorch and trained on GPU NVIDIA
   474	RTX 4090.
   475	ReID Baseline. After integrating GCP, our generated dataset, with
   476	the training sets of SYSU-MM01 or RegDB, we proceed to conduct
   477	training on four established baseline models: CCLNet, OTPA, RPNR, and
   478	SDCL. By utilizing both the synthetic data from GCP and the real-world
   479	data from SYSU-MM01 and RegDB, we aim to improve the models’
   480	ability to generalize across diverse scenarios and conditions.
   481	
   482	Fig. 5. Comparisons between real-world photos and generated images. For each image
--
   504	Despite the effectiveness of previous methods, it seems that the
   505	limited scale of VI-ReID dataset restrict the performance of existing
   506	models. It verifies that the generated images can be effectively added to
   507	the SYSU-MM01 dataset, demonstrating the effectiveness and usability
   508	of our generative framework in pre-training of VI-ReID models. This
   509	result also proves that expanding the scale of the SYSU-MM01 dataset
   510	helps the VI-ReID model better benefit from cross modal pre-training,
   511	without requiring any modifications.
   512	Comparison on RegDB. Table 3 presents the results of our methods and other SOTA methods tested on RegDB. Consistent with the
   513	experimental results observed on SYSU-MM01, our approach performs
   514	excellent results on RegDB. To be specific, our method demonstrates superiority over both baselines, with 88.88% Rank-1 and 82.41% mAP for
   515	Vis-to-IR, and 87.52% Rank-1 and 81.18% mAP for IR-to-Vis. Note that
   516	the infrared images in the RegDB dataset are thermographic infrared
   517	images, which differ significantly from the infrared images generated
   518	based on the SYSU-MM01 dataset. However, our fine-tuning process is
   519	only conducted on SYSU-MM01 because the data quality of RegDB is
   520	insufficient. Therefore, there is a slight decline in the mAP on CCLNet,
   521	but consistent improvement for OTPA.
   522	
   523	Datasets. We evaluate our method on two widely-used VI-ReID
--
   539	indoor images from 1st, 2nd, 3rd, 6th cameras are used. we also report
   540	the results of RegDB by randomly splitting of training and testing set
   541	10 times.
   542	We also employ two prominent evaluation metrics to assess the
   543	quality and relevance of synthesized images, including the CLIP Score
   544	[46] and FID [47]. CLIP Score assesses image–text alignment by comparing embeddings from a jointly trained visual-language model, with
   545	higher scores indicating better semantic coherence. The Frechet Inception Distance (FID) evaluates the similarity between real and synthetic
   546	image distributions using deep features from an Inception-v3 network.
   547	Lower FID values signal closer resemblance, reflecting the model’s ability to generate realistic and diverse images. We evaluate our method on
   548	1000 image–text pairs of CUHK-PEDES.
   549	Implementation Details. We adopt Stable Diffusion Model v1.5
   550	as our pre-trained text-to-image generation model. In it, the height and
   551	width of input and output images are all 512. The image augmentation
--
   558	Pattern Recognition 165 (2025) 111561
   559	
   560	X. Yu et al.
   561	
   562	Fig. 6. Visualization of Controllable Person Image Generation. Left: images generated with the given prompts. The image enclosed by a green bounding box denotes our selected
   563	reference image for controllable generation. Middle: we exhibit some RGB person images after postprocessing the output of IP-Adapter. Right: the corresponding infrared images
   564	with the same identities.
   565	
   566	Table 2
   567	Performance comparison on SYSU-MM01(%). GUR* denotes the results without camera information, RPNR† denotes the results we reproduced.
--
   929	Unsupervised
   930	
   931	Vis-to-IR
   932	
   933	4.3. Ablation studies and visualization results
   934	In order to show the generation results, we present extensive ablation studies and visual experiments in this section.
   935	Given the text prompt, we take the diffusion model fine-tuned
   936	by LoRA, as the basic image generation model. Due to the lack of
   937	controllable generation ability in this baseline model, although the
   938	generated images satisfy most of the text prompts, there are significant
   939	differences between the generated images in terms of details, such as
   940	clothing style, hairstyle, and other character features, which is shown
   941	in the left of Fig. 6. However, VI-ReID is sensitive to person identities,
   942	requiring images belong to the same ID to exhibit visual similarity.
   943	Thus we randomly select one of the images generated by the basic
   944	image generation model as the reference image for controllable image
   945	generation, which is enclosed by a green bounding box in the left of
   946	Fig. 6. With our controllable generation module, the generated person
   947	images looks with similar person appearance while changing postures
   948	and perspectives. Notably, the selected original person image in third
   949	column has sunglasses on the face, which is also preserved in other
   950	images. That indicates our approach is capable of making the generated
   951	person images, from the same textual description, identity-aware.
   952	Validation of GCP in VI-ReID. To verify whether the generated
   953	dataset (GCP) provided by our method can effectively generalize to
   954	
   955	IR-to-Vis
   956	
   957	7
--
   964	images generated using the SoftEdge detection algorithm better align
   965	with the characteristics of infrared imagery, such as blurred edges and
   966	missing textures, thereby exhibiting superior performance in generating
   967	infrared-style images.
   968	Ablation on Generation Quality. We conduct ablation results in
   969	terms of semantic metrics for assessing the quality and relevance of
   970	synthesized images shown in Table 5. After LoRA fine-tuning, there
   971	was no significant change in term of CLIPScore, indicating that learning
   972	the style of infrared images does not impact the generative capability
   973	of the diffusion model. Additionally, our designed text adapter show
   974	competence in higher CLIP Score while lower FID, which suggests that
   975	our generation framework performs better in comprehension of textual
   976	conditions, and further reduces the discrepancy between synthesized
   977	images and real ones, allowing for improving the discrimination of
   978	existing VI-ReID model. Furthermore, we also show visualization results
   979	of ablation studies in Fig. 9. Although the original Stable Diffusion
   980	model are capable of generating textual relevant images, the images
   981	may still suffer from semantic loss and confusion, as evidenced by
   982	lower CLIP Score values. With the help of our proposed text adapter
   983	alignment, our generation framework can produce images that more
   984	accurately reflect textual information and achieve higher CLIP Score
   985	ratings.
   986	Compared with Real-world Images. In order to visually apprehend the difference between real-world images and generated images,
   987	we show comparisons in Fig. 5. For each image pair, the image on
   988	the left is the real-world image captured by camera in CUHK-PEDES,
   989	while the image on the right is the generated image by our generation
   990	framework with the corresponding text descriptions in CUHK-PEDES.
   991	It appears that our generated images exhibit a strong resemblance to
   992	real-world images.
   993	Compared with Existing VI-ReID Datasets. The comparisons between the existing VI-ReID datasets and our generated dataset GCP is
   994	shown in Fig. 8. We exhibit images from three datasets with each row
   995	displaying 5 visible images and 5 infrared images corresponding to the
   996	same person identity. It is evident that our introduced GCP significantly
   997	enrich the variety of person scenarios in the VI-ReID dataset.
   998	t-SNE visualization of pre-training. To verify the efficacy of GCP
   999	dataset in VI-ReID pre-training, we randomly select 20 identities and
  1000	visualize the feature embedding space with t-SNE in Fig. 10. Different
  1001	colors indicate different identities, while different markers mean different modals. The result demonstrates that despite the baseline already
  1002	being capable of identifying ID discriminative features, with the help of
  1003	GCP, the VI-ReID model is able to further narrow the distance between
  1004	samples belonging to the same ID.
  1005	Training loss visualization. We document the variation of 𝐿𝐿𝑀
  1006	in Eq. (3) throughout the training process of the text adapter, illustrated
  1007	in Fig. 11. As training progresses, the loss values notably decrease and
  1008	eventually stabilize, indicating a heightened alignment between the
  1009	outputs of the original text conditional encoder and the LLM.
  1010	
  1011	Fig. 7. Different controlling conditions for ControlNet and the corresponding generated
  1012	results. Top figures show the conditional image information under different edge
  1013	and depth detection methods. Transferred infrared images are shown in the bottom
  1014	respectively.
  1015	
  1016	Table 4
  1017	Performance Comparison on GCP(%).
  1018	Method
  1019	
  1020	mAP
  1021	
--
  1025	
  1026	Rank-20
  1027	
  1028	RPNR Baseline
  1029	RPNR Trained on GCP
  1030	RPNR Trained on SYSU-MM01 + GCP
  1031	
  1032	44.06
  1033	77.00
  1034	87.55
--
  1045	98.06
  1046	98.68
  1047	
  1048	Table 5
  1049	Ablation results of our fine-tuned diffusion model in terms of various semantic metrics.
  1050	Order
  1051	
  1052	1
  1053	2
  1054	3
  1055	4
  1056	
  1057	Component
  1058	LoRA
  1059	
  1060	Adapter
  1061	
  1062	–
  1063	✓
  1064	–
--
  1068	–
  1069	✓
  1070	✓
  1071	
  1072	CLIP Score↑
  1073	
  1074	FID↓
  1075	
  1076	0.72
  1077	0.72
  1078	0.79
--
  1083	179.49
  1084	182.39
  1085	
  1086	real-world scenarios, we attempt to train and/or test the model using
  1087	only GCP dataset. We partition the GCP generated dataset with reference to the SYSU-MM01 dataset. Specifically, our training set contains
  1088	206 person identities, and the testing set includes 51 person identities.
  1089	We conduct tests based on RPNR in the all-search testing mode
  1090	similar to that of the SYSU-MM01 dataset. The detailed results are
  1091	presented in Table 4. It can be observed in the first row that even
  1092	without training on the GCP dataset, the model still exhibits a certain
  1093	person re-identification capability on the GCP testing set. Moreover,
  1094	the models trained on the GCP dataset or the SYSU-MM01 dataset
  1095	augmented with the GCP dataset show significant improvements in the
  1096	results on the GCP testing set. Notably, when the SYSU-MM01 training
  1097	set is added to the GCP training set, the model’s performance is further
  1098	enhanced to a substantial extent. This indicates that training on a
  1099	real dataset can effectively augment the model’s recognition capability
  1100	on the GCP testing set. Consequently, it indirectly demonstrates that
  1101	the GCP dataset has the capacity to validate the performance of the
  1102	model in real person re-identification scenarios, thus providing strong
  1103	evidence to validate the effectiveness of our method.
  1104	Controlling Conditions. Fig. 7 shows the different controlling
  1105	conditions for ControlNet and the corresponding generated results in
  1106	our style transfer module. CannyEdge and SoftEdge detect the edge
  1107	map while Depth detect depth information of the image. In the process of infrared style transfer, it is necessary to control the person
  1108	characters as much as possible, but the depth map lacks such detailed
  1109	information, and therefore fail to meet the requirements of generating
  1110	the same person. On the other hand, both CannyEdge and SoftEdge
  1111	
  1112	5. Conclusion
  1113	In this paper, due to the limited amount of training data in VI-ReID,
  1114	we propose a controllable diffusion framework for infrared person image generation and re-identification. This framework utilizes LoRA for
  1115	fine-tuning with VI-ReID dataset to understand the infrared modality,
  1116	and leverage a text adapter to transfer the semantic understanding
  1117	ability of LLMs to the fine-tuned diffusion models. To keep the identities of generated person images from the same textual description,
  1118	we design a controllable generation module that make the generated
  1119	images identity-aware. Our approach is capable of generating diverse
  1120	visible and infrared person images with textual descriptions and thus
  1121	obtain a new generated VI-ReID dataset. This dataset enables us to
  1122	expand the existing VI-ReID dataset, which consistently improves the
--
  1127	Pattern Recognition 165 (2025) 111561
  1128	
  1129	X. Yu et al.
  1130	
  1131	Fig. 8. Comparisons between the existing VI-ReID datasets, 𝑖.𝑒., SYSU-MM01 and RegDB, and our generated dataset GCP, with each row displaying 5 visible images and 5 infrared
  1132	images corresponding to the same person identity.
  1133	
  1134	Fig. 11. Training loss 𝐿𝐿𝑀 of the text adapter over steps.
  1135	
  1136	CRediT authorship contribution statement
  1137	Xizhuo Yu: Writing – original draft, Visualization, Methodology.
  1138	Chaojie Fan: Validation, Methodology. Zhizhong Zhang: Writing –
  1139	review & editing, Data curation. Yongbo Wang: Formal analysis, Data
  1140	curation. Chunyang Chen: Writing – review & editing, Supervision.
  1141	Tianjian Yu: Writing – review & editing, Visualization, Validation.
  1142	Yong Peng: Writing – review & editing, Supervision.
  1143	
  1144	Fig. 9. Visualization results of ablation studies on text-to-image generating. The
  1145	value under each image is its CLIP Score which is used to measure the degree of
  1146	correspondence between images and texts.
  1147	
  1148	Declaration of competing interest
  1149	The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
--
  1153	This project was supported in part by the National Natural Science Foundation of China (U24B20123), the science and technology
  1154	innovation Program of Hunan Province (2022RC3044).
  1155	
  1156	Data availability
  1157	Fig. 10. t-SNE visualization results on SYSU-MM01. Different colors indicates different
  1158	identities, while different markers means different modals. The marker ‘o’ means the
  1159	instance from visible modal, and the marker ’+’ means the instance from infrared
  1160	modal.
  1161	
--
  1201	[34] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E.L. Denton, K. Ghasemipour, R.
  1202	Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al., Photorealistic text-to-image
  1203	diffusion models with deep language understanding, Adv. Neural Inf. Process.
  1204	Syst. 35 (2022) 36479–36494.
  1205	[35] S. Zhong, Z. Huang, W. Wen, J. Qin, L. Lin, Sur-adapter: Enhancing text-to-image
  1206	pre-trained diffusion models with large language models, in: Proceedings of the
  1207	31st ACM International Conference on Multimedia, 2023, pp. 567–578.
  1208	[36] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B.
  1209	Rozière, N. Goyal, E. Hambro, F. Azhar, et al., Llama: Open and efficient
--
  1212	language description, in: Proceedings of the IEEE Conference on Computer Vision
  1213	and Pattern Recognition, 2017, pp. 1970–1979.
  1214	[38] Z. Ding, C. Ding, Z. Shao, D. Tao, Semantically self-aligned network for textto-image part-aware person re-identification, 2021, arXiv preprint arXiv:2107.
  1215	12666.
  1216	[39] H. Ye, J. Zhang, S. Liu, X. Han, W. Yang, Ip-adapter: Text compatible image
  1217	prompt adapter for text-to-image diffusion models, 2023, arXiv preprint arXiv:
  1218	2308.06721.
  1219	[40] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, Y.A. Sheikh, OpenPose: Realtime
  1220	multi-person 2D pose estimation using part affinity fields, IEEE Trans. Pattern
  1221	Anal. Mach. Intell. (2019).
--
  1246	[2] D.T. Nguyen, H.G. Hong, K.W. Kim, K.R. Park, Person recognition system based
  1247	on a combination of body images from visible light and thermal cameras, Sensors
  1248	17 (3) (2017) 605.
  1249	[3] E.J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen,
  1250	Lora: Low-rank adaptation of large language models, 2021, arXiv preprint arXiv:
  1251	2106.09685.
  1252	[4] W. Ge, C. Pan, A. Wu, H. Zheng, W.-S. Zheng, Cross-camera feature prediction
  1253	for intra-camera supervised person re-identification across distant scenes, in:
  1254	Proceedings of the 29th ACM International Conference on Multimedia, 2021,

codex
你给的名单实际是 6 篇，不是 15 篇。我按这 6 篇读了前 8 页方法部分，并额外用全文关键词补了消融和可视化证据。

**1. HOH-Net**
1. **触发观察**：作者先抓住 VI-ReID 里两个常见失败：一是大家直接拉近 VIS 和 IR 特征，但忽略了人体局部之间更复杂的高阶关系；二是已有“中间模态”或辅助特征不够可靠，往往只是简单生成、相加或拼接。更细的观察是，普通超图容易把复杂关系学塌，低层到高层的单向特征流也浪费了低层细节和高层语义之间的互动。
2. **重定义动作**：他们把问题从“跨模态差异太大”改写成“缺少高阶结构建模和分层中间特征代理”。关键词是 `high-order structure`、`reliable middle features`、`hierarchical middle-feature agents`、`modality-range identity-center`。也就是，大家以为 VIS 和 IR 只需要直接对齐，其实可靠公共空间要靠短程、长程、节点级、区域级和层级代理共同支撑。
3. **机制怎么长出来**：机制基本是顺着重定义推出来的。短程和长程特征先由 SLE 得到，再用白化超图 HSL 建高阶关系，白化是为了解决高阶关系塌成单一连接的问题。随后 FCGA 同时做细粒度节点对齐和粗粒度区域对齐，生成中间特征。HMAL 把中间特征变成每个 stage 的代理，并做低层和高层的双向增强。MRIC 再用身份中心把 VIS、IR 和 middle features 拉到更稳的共同空间。
4. **证据闭环**：关键不是只看涨点，而是证明“高阶关系和中间代理确实必要”。他们做了 SLE、HSL、FCGA、HMAL、MRIC 的逐项消融；把 HSL 和 GNN、普通 hypergraph、多种白化方式比较；还可视化高阶连接，显示普通 hypergraph 有关系坍缩，而白化后连接更分散。相似度分布、注意力图和检索结果用来证明公共空间更分离，模态差异被压小。
5. **reviewer 为什么买账**：这篇卖的是“视角加系统化机制”，不是单个模块。真正 novelty 是把中间特征从“辅助生成物”升级成“分层公共空间代理”，再用高阶结构和中心约束把它说圆。风险是模块很多，像堆料，所以它必须靠消融和关系可视化证明每个模块都服务于这个重定义。

**2. HPRNet**
1. **触发观察**：作者看到 CC-ReID 里很多方法只是在衣服区域做颜色扰动、外观重建或粗粒度人体组件分解，但真实人有身高、体型、纹理结构和局部相关性。直接重建或通道切分容易丢掉原始样本里的身份信息，还会把背景轮廓等非身份噪声当成有用线索。
2. **重定义动作**：他们把问题从“去掉衣服影响”改写成“粗糙重建缺少真实的人体解析约束和非邻域上下文”。关键词是 `human parsing reconstruction`、`non-local multi-scale perception`、`contextual correlation`、`global receptive field`、`multi-local component generation`。大家以为只要分离衣服相关和无关特征，其实分离过程本身如果没有上下文和解析约束，会制造新的偏差。
3. **机制怎么长出来**：NWMP 用非局部加权感知建模不同人体区域之间的关系，DPLS 用 2×2 和 4×4 池化做局部到整体的多尺度学习。PRE 则用人体结构把重建分成头部、身体、下肢，并引入 clothing parsing，从通道、位置和颜色角度得到更稳定的人体轮廓和衣服无关特征。整体损失把 ID、衣服分类对抗、重建、解析和多分类损失合在一起。
4. **证据闭环**：它的闭环主要靠分支消融和可视化。NWMP 和 PRE 单独加都有提升，组合最好，说明“全局上下文”和“局部解析重建”互补。损失消融说明重建、解析和多尺度分类不是装饰项。重建图、t-SNE 聚类、Grad-CAM 热力图和检索结果进一步说明模型更关注人体关键区域，而不是背景轮廓或衣服噪声。
5. **reviewer 为什么买账**：这篇主要卖“重建方法的诊断视角”。它不是说我又做了一个 reconstruction branch，而是说已有重建因为缺少上下文和解析约束，所以会把身份信息重建坏。机制本身不算特别新，但和这个诊断绑定得比较紧。

**3. Identity-aware Feature Decoupling Learning**
1. **触发观察**：作者的观察很清楚：人换衣后仍能靠脸、发型、头部轮廓、体型等线索识别，但这些线索分布不均。头部区域比较“纯”，主要是身份相关；身体区域不是纯噪声，而是身份相关信息和衣服信息耦合在一起。已有方法要么只抓一种生物线索，要么粗暴压制衣服，导致身份线索不完整。
2. **重定义动作**：它把问题从“学习衣服无关特征”改成“在衣服相关区域中解耦身份相关信息”。关键词是 `identity-aware feature decoupling`、`ID-based knowledge transfer`、`clothing bias diminishing`、`clothing contrastive loss`。大家以为衣服区域要被抑制，其实衣服区域里还藏着体型、轮廓等身份线索，关键是别让颜色纹理主导。
3. **机制怎么长出来**：注意力分支吃 clothing-masked image，保留头部和轮廓，学习身份注意力；主分支吃原图，保留普通 ReID 需要的信息。IKT 把 masked 分支学到的身份空间注意力转移给主分支。CBD 再用人体解析得到衣服区域 mask，从主分支里抽出 clothing feature，用 clothing contrastive loss 降低同一身份不同衣服之间的衣服偏差。采样上也调整为更重视同一身份多外观样本。
4. **证据闭环**：最关键的是 hard triple 可视化：anchor 和 positive 衣服完全不同，negative 衣服相似。baseline 会被衣服纹理吸走，认为 negative 更像；IFD 则更关注头部和人体轮廓。消融里 IKT、CBD 单独有效，二者合用更强；clothing contrastive loss 明显优于普通 supervised contrastive loss，这正好证明“衣服区域要解耦而不是简单忽略”。
5. **reviewer 为什么买账**：这篇卖的是视角，机制相对简洁。它把“衣服是干扰”推进到“衣服区域是身份和干扰耦合体”，这个说法比单纯 cloth-agnostic 更有解释力，也更容易设计干净消融。

**4. Identity-aware Infrared Person Image Generation via Controllable Diffusion**
1. **触发观察**：VI-ReID 数据少，SYSU-MM01 和 RegDB 场景、视角有限，真实采集又受隐私和标注成本限制。直接用 Stable Diffusion 也不行，因为它不理解红外模态，给 “infrared photo” 仍可能出彩色图；同时普通扩散生成不可控，同一文本会生成不同身份，这对 ReID 是致命问题。
2. **重定义动作**：作者把问题从“做数据增强”改成“生成能训练 ReID 的身份一致跨模态数据”。关键词是 `controllable diffusion`、`infrared modality understanding`、`text adapter alignment`、`identity-aware generation`、`Generated Cross-modal Person Dataset`。大家以为生成更多图片就行，其实 ReID 合成数据必须同时满足红外风格、文本语义、身份一致和跨模态配对。
3. **机制怎么长出来**：每个机制对应一个失败点。LoRA 用 SYSU-MM01 的可见光和红外图微调 UNet，让扩散模型学会红外概念。BLIP 生成图文对，再加 “RGB photo” 和 “infrared photo” 前缀。Text adapter 用 LLaMA 的语义能力校准原始 CLIP text encoder。生成阶段先用文本生成 RGB reference，再用 IP-Adapter 保持身份生成多视角 RGB，最后用 ControlNet SoftEdge 把 RGB 转成红外。OpenPose 和灰度化等后处理用于清洗低质量样本。
4. **证据闭环**：他们用生成可视化证明同一文本下身份更一致，比如眼镜等细节能保留。ControlNet 条件比较证明 SoftEdge 比 depth 更适合红外，因为它保留边缘和身份细节。LoRA 和 text adapter 的消融用 CLIP Score 和 FID 证明语义理解更好。GCP 加到 CCLNet、OTPA、RPNR、SDCL 等基线后都有提升，t-SNE 显示同一身份跨模态距离更近。
5. **reviewer 为什么买账**：这篇卖的是数据构造范式，不是 ReID 网络。novelty 来自“把扩散生成约束成 ReID 可用数据”，尤其是身份一致的 RGB-IR 生成。它的说服力取决于生成数据是否真的改善多个现有基线，而不是只展示好看的图。

**5. EAIBC**
1. **触发观察**：作者先展示了文本行人检索的典型失败：错误结果排在前面，不是因为整体更像，而是颜色对上了。比如文本写“blue t-shirt”，目标其实是很浅的蓝，模型却召回一堆明显蓝色衣服的人。RGB 和灰度模型的响应图也显示它们关注不同区域。
2. **重定义动作**：它把问题从“跨模态对齐不够好”改成“颜色过度依赖导致模型忽略纹理、结构和局部物品”。关键词是 `color over-reliance`、`all-round information beyond color`、`grayscale branch`、`high-frequency branch`、`color branch`、`mutual learning`。大家以为颜色是最强线索要强化，其实颜色太强会遮蔽其他可判别线索。
3. **机制怎么长出来**：机制非常顺。RGB branch 保留常规信息；GRS branch 去掉图像颜色，同时把文本颜色词 mask 掉，强迫模型看非颜色线索；HFQ branch 用高频信息关注纹理和轮廓；CLR branch 又专门保留颜色，防止模型矫枉过正。最后四个分支用 mutual learning 交流，避免各看各的。
4. **证据闭环**：这篇证据很完整。先用检索失败样例和 RGB/GRS response maps 证明颜色过度依赖存在。消融证明单分支不如多分支，加入 CLR branch 和 mutual learning 后更好。feature response map 显示 RGB 分支看粉色衣服，GRS 分支看自行车、包、箱子等非颜色线索。检索例子说明 EAIBC 能把颜色相似但细节不符的人往后排。
5. **reviewer 为什么买账**：这篇真正卖的是“偏差诊断”。模块本身不复杂，但问题被讲得非常具体：不是颜色没用，而是颜色太强，需要让非颜色证据有发言权，同时保留颜色分支。这个平衡比“去颜色”更容易让 reviewer 接受。

**6. InfinitePerson**
1. **触发观察**：真实 ReID 数据难采、难标、涉及隐私；合成数据虽然便宜，但已有合成集有两个明显缺陷。第一，3D 人物像卡通，原因是衣服 UV 纹理和 3D clothing model 不匹配。第二，采集流程单一，缺少真实监控网络里的多视角、背景、光照和遮挡变化，也不容易复现和扩展。
2. **重定义动作**：它把问题从“合成数据规模不够大”改成“合成数据的泛化能力由纹理真实性和监控采集协议决定”。关键词是 `diffusion-generated UV texture`、`reliable clothing texture`、`sub-monitoring network`、`scalable synthetic pipeline`、`direct transfer`。大家以为扩大身份数和场景数就能提升合成数据，其实 UV 纹理定义和监控网络模拟才是关键瓶颈。
3. **机制怎么长出来**：MakeHuman 生成不同体型、肤色、发型的人体模型，再加骨架和行走动画。衣服模型本身没有足够纹理，于是用 normal maps 条件训练 ControlNet，引导 Stable Diffusion 生成和衣服表面结构对齐的 UV texture maps，再导入 UE4 材质系统。采集上设计 sub-monitoring network，每个子网络用少量摄像头覆盖局部区域，并让人物沿循环路径移动，控制视角、背景、光照和遮挡。UnrealCV 自动得到实例分割和裁剪。
4. **证据闭环**：核心证据不是看合成图，而是 direct transfer。模型只在 InfinitePerson 训练，再直接测 CUHK03、Market-1501、MSMT17，泛化优于多个真实和合成数据集。定量分析显示身份数从 1000 增到 4700 时性能持续上升，子监控网络数量从 2 到 12 增加时也持续变好。UV 生成比较里，normal map 控制优于 Canny 和 InstructPix2Pix，说明“纹理要贴合 3D 表面结构”这个重定义是有效的。
5. **reviewer 为什么买账**：这篇卖的是数据集和管线视角。novelty 不在 ReID 模型，而在指出合成 ReID 的真实瓶颈是“衣服纹理几何一致性”和“监控采集协议”。只要 direct transfer 和参数分析站得住，这类工作容易被当成可复用资源接受。

最后看这批论文，套路很一致：先抓一个可视化、失败样例或数据缺陷，把它命名成一个更高层的问题，再让机制逐项对应这个问题。能发的方法稿通常不是“加了一个模块”，而是“我证明大家一直优化的目标其实漏了一个关键变量，然后我设计机制控制这个变量，并用消融和可视化证明这个变量真的变了”。
tokens used
230,917
你给的名单实际是 6 篇，不是 15 篇。我按这 6 篇读了前 8 页方法部分，并额外用全文关键词补了消融和可视化证据。

**1. HOH-Net**
1. **触发观察**：作者先抓住 VI-ReID 里两个常见失败：一是大家直接拉近 VIS 和 IR 特征，但忽略了人体局部之间更复杂的高阶关系；二是已有“中间模态”或辅助特征不够可靠，往往只是简单生成、相加或拼接。更细的观察是，普通超图容易把复杂关系学塌，低层到高层的单向特征流也浪费了低层细节和高层语义之间的互动。
2. **重定义动作**：他们把问题从“跨模态差异太大”改写成“缺少高阶结构建模和分层中间特征代理”。关键词是 `high-order structure`、`reliable middle features`、`hierarchical middle-feature agents`、`modality-range identity-center`。也就是，大家以为 VIS 和 IR 只需要直接对齐，其实可靠公共空间要靠短程、长程、节点级、区域级和层级代理共同支撑。
3. **机制怎么长出来**：机制基本是顺着重定义推出来的。短程和长程特征先由 SLE 得到，再用白化超图 HSL 建高阶关系，白化是为了解决高阶关系塌成单一连接的问题。随后 FCGA 同时做细粒度节点对齐和粗粒度区域对齐，生成中间特征。HMAL 把中间特征变成每个 stage 的代理，并做低层和高层的双向增强。MRIC 再用身份中心把 VIS、IR 和 middle features 拉到更稳的共同空间。
4. **证据闭环**：关键不是只看涨点，而是证明“高阶关系和中间代理确实必要”。他们做了 SLE、HSL、FCGA、HMAL、MRIC 的逐项消融；把 HSL 和 GNN、普通 hypergraph、多种白化方式比较；还可视化高阶连接，显示普通 hypergraph 有关系坍缩，而白化后连接更分散。相似度分布、注意力图和检索结果用来证明公共空间更分离，模态差异被压小。
5. **reviewer 为什么买账**：这篇卖的是“视角加系统化机制”，不是单个模块。真正 novelty 是把中间特征从“辅助生成物”升级成“分层公共空间代理”，再用高阶结构和中心约束把它说圆。风险是模块很多，像堆料，所以它必须靠消融和关系可视化证明每个模块都服务于这个重定义。

**2. HPRNet**
1. **触发观察**：作者看到 CC-ReID 里很多方法只是在衣服区域做颜色扰动、外观重建或粗粒度人体组件分解，但真实人有身高、体型、纹理结构和局部相关性。直接重建或通道切分容易丢掉原始样本里的身份信息，还会把背景轮廓等非身份噪声当成有用线索。
2. **重定义动作**：他们把问题从“去掉衣服影响”改写成“粗糙重建缺少真实的人体解析约束和非邻域上下文”。关键词是 `human parsing reconstruction`、`non-local multi-scale perception`、`contextual correlation`、`global receptive field`、`multi-local component generation`。大家以为只要分离衣服相关和无关特征，其实分离过程本身如果没有上下文和解析约束，会制造新的偏差。
3. **机制怎么长出来**：NWMP 用非局部加权感知建模不同人体区域之间的关系，DPLS 用 2×2 和 4×4 池化做局部到整体的多尺度学习。PRE 则用人体结构把重建分成头部、身体、下肢，并引入 clothing parsing，从通道、位置和颜色角度得到更稳定的人体轮廓和衣服无关特征。整体损失把 ID、衣服分类对抗、重建、解析和多分类损失合在一起。
4. **证据闭环**：它的闭环主要靠分支消融和可视化。NWMP 和 PRE 单独加都有提升，组合最好，说明“全局上下文”和“局部解析重建”互补。损失消融说明重建、解析和多尺度分类不是装饰项。重建图、t-SNE 聚类、Grad-CAM 热力图和检索结果进一步说明模型更关注人体关键区域，而不是背景轮廓或衣服噪声。
5. **reviewer 为什么买账**：这篇主要卖“重建方法的诊断视角”。它不是说我又做了一个 reconstruction branch，而是说已有重建因为缺少上下文和解析约束，所以会把身份信息重建坏。机制本身不算特别新，但和这个诊断绑定得比较紧。

**3. Identity-aware Feature Decoupling Learning**
1. **触发观察**：作者的观察很清楚：人换衣后仍能靠脸、发型、头部轮廓、体型等线索识别，但这些线索分布不均。头部区域比较“纯”，主要是身份相关；身体区域不是纯噪声，而是身份相关信息和衣服信息耦合在一起。已有方法要么只抓一种生物线索，要么粗暴压制衣服，导致身份线索不完整。
2. **重定义动作**：它把问题从“学习衣服无关特征”改成“在衣服相关区域中解耦身份相关信息”。关键词是 `identity-aware feature decoupling`、`ID-based knowledge transfer`、`clothing bias diminishing`、`clothing contrastive loss`。大家以为衣服区域要被抑制，其实衣服区域里还藏着体型、轮廓等身份线索，关键是别让颜色纹理主导。
3. **机制怎么长出来**：注意力分支吃 clothing-masked image，保留头部和轮廓，学习身份注意力；主分支吃原图，保留普通 ReID 需要的信息。IKT 把 masked 分支学到的身份空间注意力转移给主分支。CBD 再用人体解析得到衣服区域 mask，从主分支里抽出 clothing feature，用 clothing contrastive loss 降低同一身份不同衣服之间的衣服偏差。采样上也调整为更重视同一身份多外观样本。
4. **证据闭环**：最关键的是 hard triple 可视化：anchor 和 positive 衣服完全不同，negative 衣服相似。baseline 会被衣服纹理吸走，认为 negative 更像；IFD 则更关注头部和人体轮廓。消融里 IKT、CBD 单独有效，二者合用更强；clothing contrastive loss 明显优于普通 supervised contrastive loss，这正好证明“衣服区域要解耦而不是简单忽略”。
5. **reviewer 为什么买账**：这篇卖的是视角，机制相对简洁。它把“衣服是干扰”推进到“衣服区域是身份和干扰耦合体”，这个说法比单纯 cloth-agnostic 更有解释力，也更容易设计干净消融。

**4. Identity-aware Infrared Person Image Generation via Controllable Diffusion**
1. **触发观察**：VI-ReID 数据少，SYSU-MM01 和 RegDB 场景、视角有限，真实采集又受隐私和标注成本限制。直接用 Stable Diffusion 也不行，因为它不理解红外模态，给 “infrared photo” 仍可能出彩色图；同时普通扩散生成不可控，同一文本会生成不同身份，这对 ReID 是致命问题。
2. **重定义动作**：作者把问题从“做数据增强”改成“生成能训练 ReID 的身份一致跨模态数据”。关键词是 `controllable diffusion`、`infrared modality understanding`、`text adapter alignment`、`identity-aware generation`、`Generated Cross-modal Person Dataset`。大家以为生成更多图片就行，其实 ReID 合成数据必须同时满足红外风格、文本语义、身份一致和跨模态配对。
3. **机制怎么长出来**：每个机制对应一个失败点。LoRA 用 SYSU-MM01 的可见光和红外图微调 UNet，让扩散模型学会红外概念。BLIP 生成图文对，再加 “RGB photo” 和 “infrared photo” 前缀。Text adapter 用 LLaMA 的语义能力校准原始 CLIP text encoder。生成阶段先用文本生成 RGB reference，再用 IP-Adapter 保持身份生成多视角 RGB，最后用 ControlNet SoftEdge 把 RGB 转成红外。OpenPose 和灰度化等后处理用于清洗低质量样本。
4. **证据闭环**：他们用生成可视化证明同一文本下身份更一致，比如眼镜等细节能保留。ControlNet 条件比较证明 SoftEdge 比 depth 更适合红外，因为它保留边缘和身份细节。LoRA 和 text adapter 的消融用 CLIP Score 和 FID 证明语义理解更好。GCP 加到 CCLNet、OTPA、RPNR、SDCL 等基线后都有提升，t-SNE 显示同一身份跨模态距离更近。
5. **reviewer 为什么买账**：这篇卖的是数据构造范式，不是 ReID 网络。novelty 来自“把扩散生成约束成 ReID 可用数据”，尤其是身份一致的 RGB-IR 生成。它的说服力取决于生成数据是否真的改善多个现有基线，而不是只展示好看的图。

**5. EAIBC**
1. **触发观察**：作者先展示了文本行人检索的典型失败：错误结果排在前面，不是因为整体更像，而是颜色对上了。比如文本写“blue t-shirt”，目标其实是很浅的蓝，模型却召回一堆明显蓝色衣服的人。RGB 和灰度模型的响应图也显示它们关注不同区域。
2. **重定义动作**：它把问题从“跨模态对齐不够好”改成“颜色过度依赖导致模型忽略纹理、结构和局部物品”。关键词是 `color over-reliance`、`all-round information beyond color`、`grayscale branch`、`high-frequency branch`、`color branch`、`mutual learning`。大家以为颜色是最强线索要强化，其实颜色太强会遮蔽其他可判别线索。
3. **机制怎么长出来**：机制非常顺。RGB branch 保留常规信息；GRS branch 去掉图像颜色，同时把文本颜色词 mask 掉，强迫模型看非颜色线索；HFQ branch 用高频信息关注纹理和轮廓；CLR branch 又专门保留颜色，防止模型矫枉过正。最后四个分支用 mutual learning 交流，避免各看各的。
4. **证据闭环**：这篇证据很完整。先用检索失败样例和 RGB/GRS response maps 证明颜色过度依赖存在。消融证明单分支不如多分支，加入 CLR branch 和 mutual learning 后更好。feature response map 显示 RGB 分支看粉色衣服，GRS 分支看自行车、包、箱子等非颜色线索。检索例子说明 EAIBC 能把颜色相似但细节不符的人往后排。
5. **reviewer 为什么买账**：这篇真正卖的是“偏差诊断”。模块本身不复杂，但问题被讲得非常具体：不是颜色没用，而是颜色太强，需要让非颜色证据有发言权，同时保留颜色分支。这个平衡比“去颜色”更容易让 reviewer 接受。

**6. InfinitePerson**
1. **触发观察**：真实 ReID 数据难采、难标、涉及隐私；合成数据虽然便宜，但已有合成集有两个明显缺陷。第一，3D 人物像卡通，原因是衣服 UV 纹理和 3D clothing model 不匹配。第二，采集流程单一，缺少真实监控网络里的多视角、背景、光照和遮挡变化，也不容易复现和扩展。
2. **重定义动作**：它把问题从“合成数据规模不够大”改成“合成数据的泛化能力由纹理真实性和监控采集协议决定”。关键词是 `diffusion-generated UV texture`、`reliable clothing texture`、`sub-monitoring network`、`scalable synthetic pipeline`、`direct transfer`。大家以为扩大身份数和场景数就能提升合成数据，其实 UV 纹理定义和监控网络模拟才是关键瓶颈。
3. **机制怎么长出来**：MakeHuman 生成不同体型、肤色、发型的人体模型，再加骨架和行走动画。衣服模型本身没有足够纹理，于是用 normal maps 条件训练 ControlNet，引导 Stable Diffusion 生成和衣服表面结构对齐的 UV texture maps，再导入 UE4 材质系统。采集上设计 sub-monitoring network，每个子网络用少量摄像头覆盖局部区域，并让人物沿循环路径移动，控制视角、背景、光照和遮挡。UnrealCV 自动得到实例分割和裁剪。
4. **证据闭环**：核心证据不是看合成图，而是 direct transfer。模型只在 InfinitePerson 训练，再直接测 CUHK03、Market-1501、MSMT17，泛化优于多个真实和合成数据集。定量分析显示身份数从 1000 增到 4700 时性能持续上升，子监控网络数量从 2 到 12 增加时也持续变好。UV 生成比较里，normal map 控制优于 Canny 和 InstructPix2Pix，说明“纹理要贴合 3D 表面结构”这个重定义是有效的。
5. **reviewer 为什么买账**：这篇卖的是数据集和管线视角。novelty 不在 ReID 模型，而在指出合成 ReID 的真实瓶颈是“衣服纹理几何一致性”和“监控采集协议”。只要 direct transfer 和参数分析站得住，这类工作容易被当成可复用资源接受。

最后看这批论文，套路很一致：先抓一个可视化、失败样例或数据缺陷，把它命名成一个更高层的问题，再让机制逐项对应这个问题。能发的方法稿通常不是“加了一个模块”，而是“我证明大家一直优化的目标其实漏了一个关键变量，然后我设计机制控制这个变量，并用消融和可视化证明这个变量真的变了”。
