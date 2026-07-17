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
session id: 019ef71f-af06-7f83-b510-d8d3e0a1b973
--------
user
你是 ReID 论文创新挖掘员 14/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(14/20):
Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf
Occluded person Re-Identification with noise injection.pdf
Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf
Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf
Privacy preserving person re-identification via anonymizing diffusion model.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我会按 `pdf` 技能处理 PDF，但这次只用你指定的 `pdftotext -l 3`，不做全文解析。先快速确认目录和项目状态文件是否能读到，然后逐篇抽取前 3 页。
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
/bin/zsh -lc 'for f in ../../../../GOAL.md ../../../../CLAUDE.md ../../../STATUS.md; do if [ -r "$f" ]; then printf '"'===== %s =====\\n' \""'$f"; sed -n '"'1,120p' \""'$f"; else printf '"'UNREADABLE %s\\n' \""'$f"; fi; done' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
UNREADABLE ../../../../GOAL.md
UNREADABLE ../../../../CLAUDE.md
UNREADABLE ../../../STATUS.md

exec
/bin/zsh -lc "sed -n '1,180p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.622.11653/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
项目根目录的 `GOAL.md`、`CLAUDE.md` 和 `experiments/STATUS.md` 在当前只读沙箱里不可读，我会依据你这次给出的任务和目录规则继续。现在分批抽取 9 篇论文前三页，先看可直接支撑创新挖掘的摘要、引言和贡献点。
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
/bin/zsh -lc "pdftotext -l 3 'Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

1

Mutual Distillation Driven Dual-Space Matching for
Visible–Infrared Person Re-Identification
Linfeng Jiang, Member, IEEE, Dongcan Liu, Jinsheng Ji, Member, IEEE, and Ting Bai*, Member, IEEE

Abstract—Visible–infrared person re-identification (VI-ReID)
aims to match pedestrian images across heterogeneous modalities.
As a key technology in intelligent transportation systems, VIReID supports cross-camera tracking, behavior analysis, and
security monitoring, particularly in nighttime or low-illumination
scenarios. Despite recent advances, existing methods still encounter two critical challenges: (i) semantic misalignment between low-level and high-level features across modalities, and (ii)
distribution discrepancies between visible and infrared images.
To address these challenges, we propose a novel framework,
Mutual Distillation Driven Dual-Space Matching (MDDM), which
performs modality alignment in two complementary spaces. For
challenge (i), we design a Dual Level Fusion (DLF) module
to capture and adaptively fuse hierarchical features, aligning
modalities by integrating both low- and high-level semantics
across spatial and channel dimensions. In addition, a Modality
Invariant Augmentation (MIA) module is developed to extract
fine-grained semantic cues and enhance identity discrimination,
thereby reinforcing the correlation between visible and infrared
modalities and facilitating the learning of robust shared representations. For challenge (ii), we introduce Dual-Space Matching
(DSM), which aligns features in both Hilbert and Euclidean
spaces. Furthermore, a mutual distillation strategy is incorporated to promote cross-space consistency and alleviate modalityspecific discrepancies. Extensive experiments on widely used VIReID benchmarks demonstrate the superiority and flexibility of
the proposed method, which consistently achieves competitive
performance across multiple datasets. Our code is available at
https://github.com/lfjiang-cn/MDDM.
Index Terms—Visible-infrared person re-identification, mutual
distillation, dual-space.

I. I NTRODUCTION

P

ERSON re-identification (ReID) aims to match pedestrian images across cameras with non-overlapping views.
Owing to its practical significance in modern surveillance
and forensic systems, Re-ID has attracted increasing attention
and has become a key technology in intelligent transportation
This work was supported by the National Natural Science Foundation of
China (Grant No. 61971078), the Natural Science Foundation of Chongqing,
China (Grant No. cstc2021jcyj-msxmX0605), and Science and Technology Foundation of Chongqing Education Commission (Grant Nos. KJZDK202301102, KJQN202101104). (Corresponding author: Ting Bai)
Linfeng Jiang is with the School of Artificial Intelligence, Chongqing
University of Technology, Chongqing, 404100, China (e-mail: linfengjiang@cqut.edu.cn).
Dongcan Liu is with the School of Artificial Intelligence, Chongqing
University of Technology, Chongqing, 404100, China (e-mail: dongcanliu@stu.cqut.edu.cn).
Jinsheng Ji is with the School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore 639798 (e-mail: jinsheng.ji@ntu.edu.sg).
Ting Bai is with the School of Automation and Intelligent Sensing, Shanghai Jiao Tong University, Shanghai, 200240, China (email:
tingbai@sjtu.edu.cn).

Fig. 1: Illustration of the differences between the proposed
model and existing VI-ReID paradigms: (a) image-level
modality compensation (e.g., GAN-based methods) that rely
on generation and may introduce instability and artifacts; (b)
modality-shared feature learning in a single embedding space
that may overlook complex cross-modality relationships; and
(c) the proposed dual-space matching, which aligns features
in both Euclidean and Hilbert spaces via mutual distillation,
enabling more robust cross-modality association.

systems [1]–[8]. Most existing Re-ID methods [9]–[14] focus
on matching RGB pedestrian images captured by visible
(VIS) cameras. However, VIS cameras often fail to capture
sufficient human information in nighttime or low-illumination
conditions, leading to unsatisfactory performance. To address
this limitation, many advanced surveillance systems employ
infrared (IR) cameras, which can automatically switch between
VIS and IR modes based on real-time illumination. Consequently, a number of visible–infrared person re-identification
(VI-ReID) methods [15]–[18] have been proposed to enable
cross-modal retrieval between VIS and IR images.
Compared with conventional Re-ID, VI-ReID is more challenging due to the significant modality discrepancy between
VIS and IR images. This discrepancy arises from differences
in illumination, appearance, and texture, making cross-modal

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

matching difficult. Specifically, VIS light has a wavelength
range of 400–700 nm, and is typically stored as RGB threechannel images in Re-ID tasks, providing rich color and
texture information. In contrast, IR light ranges from 700
nm to 1 mm, among which near-infrared (NIR, 700–1000
nm) is most commonly used in Re-ID. NIR images are usually single-channel grayscale images that reflect the intensity
of infrared radiation, lacking color information and certain
fine details [19], [20]. Moreover, the inherent heterogeneity
across modalities results in distinct feature distributions. While
VIS images can focus on pedestrian regions with reduced
background interference, IR images often suffer from blurred
boundaries between pedestrians and background. This discrepancy leads to severe feature misalignment and erroneous
matches in the latent feature space. In addition, VI-ReID faces
intra-modality variations such as viewpoint, pose, clothing, and
background changes, further increasing task complexity.
Recently, numerous approaches have been proposed to address VI-ReID [21]–[25], which can generally be categorized
into image-level and feature-level methods, as illustrated in
Fig. 1. Image-level methods [15], [26] typically employ Generative Adversarial Network (GAN) to synthesize VIS images
from IR inputs or vice versa, thereby bridging the modality
gap. However, these methods require precisely aligned VIS–IR
image pairs with consistent viewpoint, time, and resolution.
Despite their promising results, image-level approaches are
often computationally expensive, prone to noise generation,
and limited in flexibility. Feature-level methods [27]–[30], in
contrast, generally adopt dual-stream architectures to extract
both modality-specific and modality-shared representations in
the latent space. Owing to their higher efficiency and stability,
feature-level techniques have become the dominant paradigm
in recent VI-ReID research. Nevertheless, due to the large
modality discrepancy and the unbounded nature of latent
feature spaces, these methods generally capture only coarsegrained patterns (e.g., human contours and structures) and
struggle to learn discriminative identity-consistent knowledge
across modalities. Furthermore, as the network deepens, a
substantial amount of discriminative semantic information may
be lost, and the intrinsic modality gap continues to drive the
model toward inconsistent cues, thereby hindering accurate
cross-modal matching, as empirically observed in previous
studies [31]–[33]. In essence, the large modality gap remains
the primary bottleneck limiting VI-ReID performance, and
effectively bridging this gap to discover cross-modal consistent
information is still a major challenge.
Based on these observations, we summarize the challenges
of VI-ReID into two aspects: (i) difficulty in aligning lowand high-level semantics across modalities, and (ii) distribution discrepancies between VIS and IR features. To address
these challenges, we propose a simple yet effective framework named Mutual Distillation driven Dual-Space Matching (MDDM). Specifically, for challenge (i), we design a
Dual Level Fusion (DLF) module and a Modality Invariant
Augmentation (MIA) module. DLF captures and adaptively
fuses hierarchical features, aligning modalities by integrating
both low- and high-level semantics across spatial and channel
dimensions. MIA extracts fine-grained semantic cues and

2

enhances identity discrimination, thereby reinforcing VIS–IR
correlations and enabling the learning of robust modalityshared representations. For challenge (ii), we propose a novel
Dual-Space Matching (DSM), which aligns modalities in both
Euclidean and Hilbert spaces at the pixel level. By mapping
shared features into these two complementary spaces and
employing a mutual distillation strategy, DSM encourages
cross-space consistency, alleviates the unboundedness of latent spaces, and strengthens the representation capacity of
shared features. Moreover, DSM guides the model to focus
on consistent cues across spaces, which can be regarded as
invariant, robust, and discriminative, thereby reducing modality discrepancies. In summary, the contributions of this paper
are as follows:
• We propose a novel Mutual Distillation driven DualSpace Matching (MDDM) framework, which performs
pixel-level alignment in both Euclidean and Hilbert
spaces to mitigate modality discrepancies in VI-ReID.
• We design a Dual Level Fusion (DLF) to capture hierarchical semantics across channel and spatial dimensions,
and a Modality Invariant Augmentation (MIA) module
to learn discriminative modality-shared knowledge by
establishing VIS–IR correlations.
• Extensive experiments on SYSU-MM01, RegDB, and
LLCM demonstrate the effectiveness of MDDM, achieving competitive performance compared with state-of-theart methods.
II. R ELATED W ORK
A. Person Re-Identification
Person re-identification (ReID) [10], [34], [35] is a singlemodality matching task in which both query and gallery
images belong to the visible spectrum. The objective is to
match the same pedestrian across non-overlapping camera
views. The primary challenges of conventional ReID arise
from substantial intra-modality variations, including clothing
changes, viewpoints, and poses. Existing studies [11], [36]–
[39] mainly focus on feature representation learning and metric
learning, and have achieved impressive performance on benchmark datasets. In real-world scenarios, however, illumination
conditions often vary widely, and infrared images are typically captured under poor lighting. These practical constraints
naturally motivate the development of visible-infrared person
re-identification, where conventional single-modality methods
struggle to address the considerable modality discrepancy.
B. Visible-Infrared Person Re-Identification
Visible-infrared person re-identification (VI-ReID) [40]–
[42] is a challenging cross-modality task that has recently
attracted increasing attention. Its goal is to match visible and
infrared pedestrian images based on feature similarity ranking.
VI-ReID plays a critical role in modern surveillance systems,
enabling seamless 24-hour operation. Existing approaches can
be broadly divided into unsupervised and supervised methods.
Unsupervised VI-ReID assigns pseudo-labels to training
data through clustering to reduce annotation costs and has

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

become a popular research direction. For instance, Shi et
al. [43] integrated hard and dynamic prototypes into unsupervised VI-ReID and proposed Progressive Contrastive Learning
with Hard and Dynamic (PCLHD) prototypes to exploit crossmodal divergence and diversity. Yang et al. [44] designed
Progressive Cross-modal Association Learning (PCAL), which
combined triple-modal adversarial learning with cross-modal
neighbor expansion to progressively eliminate modality bias
and establish reliable correspondences. Yang et al. [45] introduced Shallow-Deep Collaborative Learning (SDCL), coupled shallow details with deep semantics via collaborative ranking association to enhance cross-modal consistency.
Xia et al. [46] proposed Camera-proxy Enhanced Identityrecalibration Learning (CEIL), employing a two-stage pipeline
and Graph-based Camera-constraint Re-ranking (GCR) to recalibrate cross-modal identities and narrow the modality gap.
Although these methods avoid costly annotations, their performance still lags behind supervised approaches. Moreover,
pseudo-label assignment through clustering is computationally
expensive and time-consuming, limiting their applicability in
real-world scenarios.
Supervised VI-ReID leverages label information to extract
modality invariant features and mitigate the large discrepancy
between visible and infrared modalities. Existing works can
be categorized into image-level methods [15], [26], [47] and
feature-level methods [27]–[30].
Image-level methods typically generate synthetic images
or intermediate modalities to bridge the modality gap. For
example, Qian et al. [48] proposed a patch-mixed framework
that stitched RGB and IR patches of the same identity into
a unified image, enabling the model to learn semantic correspondence and reduce modality imbalance. Dai et al. [47]
introduced DiVE, a diffusion-based framework that disentangled identity and modality to synthesize large-scale RGBIR paired data, significantly improving VI-ReID performance.
Zhang et al. [49] designed a non-linear Middle Modality
Generator (MMG) that projected visible and infrared images
into a Unified Middle-Modality Image (UMMI) space, thereby
reducing modality discrepancy. Lu et al. [50] presented Trilevel Modality-information Disentanglement (TMD), which
disentangled image-, distribution-, and instance-level information to eliminate modality variance. Pan et al. [51] proposed
a unified conditional image generation method that produced
intra-modality, cross-modality, and middle-modality images to
alleviate discrepancy and augment training data. Despite their
advances, these methods are often sensitive to noise, lack
reliable modality transformation, and incur high computational
costs due to the joint training of auxiliary and original modalities, which constrains their practicality.
Feature-level methods aim to extract shared representations across modalities to mitigate discrepancies. Liang et
al. [52] proposed the Multi-level Cross-modality Joint Alignment (MCJA), which aligns training and testing objectives
at multiple levels to enhance cross-modal performance. CycleTrans [33] constructed cycles to capture modality-related
semantics and transformed them into neutral yet discriminative features. Cui et al. [32] presented Dual Modality-aware
Alignment (DMA), which performed data-level alignment in

3

the HSV color space and suppressed misleading information.
MSALNet [53] devised a multi-stage auxiliary learning strategy that progressively trains on grayscale-equalized and original image pairs, effectively reducing modality discrepancies
and computational costs. RHL [54] formulated a robust hybrid
loss that categorizes noisy data into clean, obviously noisy,
and indistinguishably noisy subsets for tailored optimization,
effectively mitigating the impact of real-world label noise.
TransVI [55] employed a dual-stream Transformer architecture
to capture long-range dependencies, incorporating a CrossModality Dissimilarity-based Maximum Mean Discrepancy
(CMD-MMD) to mitigate cross-modality discrepancies from
the perspective of global distance distributions. CSC-Net [56]
explored intra- and inter-modality correlations via semantic alignment and cross-granularity awareness to discover
modality-irrelevant features and mitigate modality bias. Despite these advances, these methods remain susceptible to
instability in the latent space during feature alignment, which
may lead to misalignment and limit the ability to fully capture
complex correlations between visible and infrared modalities.
To further reduce modality discrepancies, some studies
focus on distribution-level alignment. For example, Hao et
al. [57] mapped shared features onto a hyperspherical manifold
and aligned them based on angular distances between modalities. Zhang et al. [58] proposed the Diverse Embedding Expansion Network (DEEN), which leverages multi-level feature
representations to generate diverse embeddings and exploit
modality-shared information for VI-ReID. Jiang et al. [59]
introduced a domain shifting method (DNS) that enhances
both modality-specific and shared knowledge through crossmodality alignment. Despite these advances, such methods
generally follow a single-space alignment paradigm, where
multi-level features are fused and aligned within a unified
embedding space. This design primarily focuses on improving
feature discriminability, but may be insufficient to capture
comprehensive cross-modality relationships.
To address these limitations, we move beyond this paradigm
by establishing cross-modality interactions in both the Euclidean space and an implicit Hilbert space. To realize this
dual-space modeling, we introduce a mutual distillation mechanism that enables cross-space knowledge transfer, facilitating
the extraction of modality-invariant features while preserving
complementary information across spaces, thereby effectively
reducing cross-modality discrepancies.

III. M ETHOD
In this section, we provide a detailed explanation of the
proposed Mutual Distillation driven Dual-space Matching
(MDDM) framework. Firstly, we present the overall architecture in Section III-A. Secondly, the proposed Dual Level Fusion (DLF) is described in Section III-B. Thirdly, Section III-C
and Section III-D introduce the Modality Invariant Augmentation (MIA) module and the Dual-Space Matching (DSM),
respectively. Finally, the optimization scheme is discussed in
Section III-E.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


exec
/bin/zsh -lc "pdftotext -l 3 'Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5608

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 4, APRIL 2026

Nearest Neighbor Sample Constraint and ODE
Guided Feature Reconstruction for Unsupervised
Person Re-Identification
Xi Yang , Senior Member, IEEE, Wenjiao Dong, Gu Zheng, and Nannan Wang , Senior Member, IEEE

Abstract—Unsupervised person re-identification aims to
retrieve a given pedestrian image from unlabeled data. The
method of clustering and assigning pseudo-labels has become
mainstream, but there are still some problems that will reduce
recognition accuracy. On the one hand, in the process of clustering, poor classification of hard samples between neighboring
classes leads to inadequate clustering accuracy, which affects the
quality of pseudo-labels. On the other hand, the representational
capacity of features extracted by the backbone network is
also crucial for the model’s performance. To this end, this
paper proposes an unsupervised person re-identification method
based on nearest neighbor sample constraint and ordinary
differential equation guided feature reconstruction (NNSC-FR)
to improve the clustering accuracy and pseudo-label quality
while enhancing the representation of features. Specifically, we
present a novel nearest neighbor sample constraint (NNSC) after
neighbor sample mining for each instance sample to recognize
the hard samples’ fine classification between classes. To further
improve clustering accuracy, an inter-class balance loss (CB loss)
is introduced to better identify the hard samples between the
nearest neighbor classes. In addition, guided by the thirdorder adam solution of the Ordinary Differential Equation, we
design a Feature Reconstruction (ODE-FR) module with residual
structure to improve the model representation ability. Extensive
experimental results on Market-1501, DukeMTMC-reID, and
MSMT17 demonstrate that our proposed method is superior to
the state-of-the-art methods.
Index Terms—Unsupervised person re-identification, nearest
neighbor constraint, ordinary differential equation, feature reconstruction.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) aims to identify the
same person in different camera views, and it has been

Received 9 June 2025; revised 27 October 2025; accepted 10 November
2025. Date of publication 13 November 2025; date of current version 7 April
2026. This work was supported in part by the National Natural Science
Foundation of China under Grant 62372348, Grant U22A2096, and Grant
62036007; in part by the Key Research and Development Program of Shaanxi
under Grant 2024GX-ZDCYL-02-10; in part by Shaanxi Outstanding Youth
Science Fund Project under Grant 2023-JC-JQ-53; in part by the Scientific and Technological Innovation Teams in Shaanxi Province under Grant
2025RS-CXTD-011; in part by Shaanxi Province Core Technology Research
and Development Project under Grant 2024QY2-GJHX-11; and in part by
the Fundamental Research Funds for the Central Universities under Grant
QTZX23042. This article was recommended by Associate Editor Z. Mao.
(Corresponding author: Xi Yang.)
The authors are with the State Key Laboratory of Integrated Services
Networks, School of Telecommunications Engineering, Xidian University,
Xi’an 710071, China (e-mail: yangx@xidian.edu.cn; dwj@stu.xidian.edu.cn;
zhengguyan@stu.xidian.edu.cn; nnwang@xidian.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3632324

widely used in urban intelligent security and large-scale
monitoring systems. Significant progress has been made in
supervised person Re-ID tasks [1], [2], [3], [4], [5], [6],
but its extensive manual annotation of labels has greatly
limited landing applications. In real life, it is not difficult
for police to obtain untagged pedestrian images while using
security surveillance to track criminal suspects. As a result,
increasing research has focused on unsupervised person Re-ID,
which learns directly from unlabeled data, exploring potential
feature representations. This technique is more scalable and
has more significant potential for deployment in the real
world.
Existing unsupervised person Re-ID faces two core challenges, which together constitute the motivation for our
proposed method. First, insufficient feature representation
capability: although existing deep networks (e.g., ResNet)
can extract multi-level features, there is a lack of effective
fusion mechanisms among different hierarchical feature information, and traditional residual connections lack mathematical
modeling of the feature evolution process. Second, lack of
optimization mechanisms for cluster boundary regions: in
the iterative training process of existing methods, boundary
hard samples often receive unstable pseudo-labels, and these
errors accumulate and amplify in subsequent training, leading
to insufficient inter-class separability and degraded model
performance.
Recently, many advanced methods [7], [8] utilize memory
banks to store all instance features, treating each image as an
independent class and learning representations by matching
features of the same instance across different augmented
views. However, in person Re-ID datasets, each class typically
contains multiple positive instances. The SpCL [9] method
alleviates this issue by matching instances with multiple centroids of positive values. To further ensure that each positive
value converges to its centroid at a uniform rate, the cluster
contrast learning method [10] updates the memory bank and
computes the contrastive loss at the cluster level, making the
contrastive loss the most suitable metric learning algorithm for
updating memory bank features.
Although cluster contrast learning achieves impressive performance, applying contrast learning only at the cluster level
does not consider the relationships among difficult samples at
the instance level. Furthermore, previous deep metric learning works focus on difficult sample mining to place greater
emphasis on hard samples within a class. These methods
aim to distinguish between different classes of samples and
bring samples of the same class closer together. However,

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:00 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION

Fig. 1. Examples of confusing images for the unsupervised person Re-ID
task.

these methods often employ small batch-based depth measurement losses, such as hard triplet losses [11] and multiple
similarity losses [12], which utilize only a small subset of
data and do not account for information across all classes,
and they are incompatible with memory bank architectures
designed to ensure consistency across the entire dataset. As
shown in Fig. 1, appropriate constraints are needed between
easily confused sample classes due to various reasons such
as pedestrian occlusion or similar clothing. At the same time,
features with rich representation capabilities can refine pseudolabels, thereby improving the accuracy of model training.
Therefore, we also need to find ways to enhance the model’s
representation ability.
To solve these problems, we propose an unsupervised
person Re-ID method based on nearest neighbor sample
constraint and ordinary differential equation guided feature
reconstruction (NNSC-FR) to improve the clustering accuracy
and pseudo-label quality while enhancing the representation of
features. Specifically, we propose a nearest neighbor constraint
(NNSC) based relative distance, which explicitly identifies
hard samples located near decision boundaries by computing
sample pair distances. NNSC enables the model to focus
computational resources on the most informative samples
rather than treating all samples uniformly. Based on NNSC,
we also design an inter-class balance loss (CB loss), which
works together to classify the hard samples between the
neighboring domains and distinguish the two clusters to the
greatest extent by forcing feature vectors to escape from the
uniform classification distribution state. In addition, works on
the design of residual structures guided by the solution of
ODEs [13], [14], [15] have shown that it can significantly
help the representation and generalization of the model. Thus,
we apply the implicit third-order adam solution of ODEs to
design a suitable residual structure guided by the solution of
differential equations in numerical form. Different from the
methods of mining encoder features at various depths, such
as feature fusion and cascade, the residual structure can be
inserted at any depth of the encoder, and function modules
can be designed according to different task characteristics,
so we call it ODE guided feature reconstruction (ODE-FR)

5609

module. Meanwhile, we insert a simple PReLU layer [16] and
a Conv layer to reconstruct the backbone network’s features
appropriately. Combined with the anti-degradation ability of
the residual structure, we improve the feature characterization
ability of the unsupervised model.
Overall, we improve the unsupervised person Re-ID model’s
performance from two perspectives: improving clustering
accuracy and feature representation capability. Our contribution can be summarized as follows:
• We introduce the nearest neighbor sample constraint
strategy (NNSC), which aims to enhance clustering accuracy by mining positive samples within the same class
and negative samples between neighboring classes at the
instance level.
• We design the inter-class balance loss (CB loss) better to
identify the hard samples between the nearest neighbor
classes and improve the clustering accuracy and pseudolabel quality.
• We incorporate an ODE-guided feature reconstruction
(ODE-FR) module, which leverages the implicit thirdorder Adam solution of an ordinary differential equation
to create the residual structure of the architecture and
improve the model representation ability.
• Extensive experiments verify that our proposed NNSC-FR
is superior to existing unsupervised person Re-ID methods on Market-1501, DukeMTMC-reID, and MSMT17
datasets.
II. R ELATED W ORK
A. Unsupervised Person Re-Identification
The clustering algorithm is the most common method for
unsupervised person Re-ID tasks, which is used to cluster pedestrian images into clusters of different IDs without
identification labels. Kodinariya et al. [17] proposed a kmeans clustering algorithm, which assigns image features to
k clustering centers to maximize image similarity within each
cluster. DBSCAN proposed by Schubert et al. [18] is currently
the most widely used clustering algorithm in the field, which
can automatically determine the number of clusters. With the
development of the clustering algorithm, the optimization of
pseudo-labels has become the focus of researchers, and the
two complement each other. Ge et al. [19] first proposed to
refine the one-hot form of pseudo-labels and used the mutual
supervision teacher-student model to refine the pseudo-labels
globally.
Recently, some methods are seeking better clustering methods to improve the quality of pseudo-labels. For example,
Zhang et al. [20] proposed a time-integrated approach to
improve pseudo-labels by dynamically changing categories
round by round. Han et al. [21] modulated the representation
learning process to enhance the statistical stability of feature
representations incrementally. The above methods improved
the mainstream method of generating pseudo-labels by clustering. In addition, some methods [22], [23], [24], [25], [26],
[27] made full use of the complementary nature of global
features and local features to optimize clustering. Tao et al.
[28] proposed a diffusion model-based unsupervised person
re-identification framework, optimizing pseudo-labels through
a spatial diffusion model and semantic consistency loss, thus

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:00 UTC from IEEE Xplore. Restrictions apply.

5610

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 4, APRIL 2026

enhancing current state-of-the-art methods. The recently introduced Relation-Preserving Feature Embedding model [29]
further improves performance by leveraging structural relationships among samples to address challenging instances. In the
meanwhile, Sheng et al. [30] creatively proposed introducing
neural architecture search into unsupervised tasks. It is worth
mentioning that GCL [31] and LIGV [32] proposed the gridbased view generator, where the 3D grid-based guided image
generator can create data enhancement algorithms for contrast
learning.
Despite continuous improvements in mainstream clustering
algorithms, accurately clustering visually ambiguous samples
caused by factors such as foreground occlusion and highly
similar clothing colors remains a significant challenge. To
address this issue, Pang et al. proposed the CIFL framework
[33] and the MSC-GDC method [34], both of which incorporate camera label information: CIFL employs an ensemble
clustering strategy to learn camera-invariant features, while
MSC-GDC introduces a camera penalty factor to mitigate
interference from cross-camera style variations in clustering
results. In contrast to these camera-aware approaches, this
study proposes a nearest neighbor constraint mechanism that
directly optimizes the decision boundaries for ambiguous
samples in the feature space, thereby enhancing classification
accuracy.
B. Sampling Strategy and Contrast Learning
The sampling strategy refers to selecting unlabeled pedestrian images or features for training. Fan et al. [35] proposed
a progressive learning method integrating self-step learning
into training. In the initial training of the model, only a small
number of high-quality samples around the center of mass
were considered for training. The model would self-adapt
and select more pedestrian image samples in the subsequent
training. However, it used a fixed confidence threshold sampling, and the threshold cannot be changed adaptively. In this
regard, Wu et al. [36] proposed a dynamic sampling algorithm,
constantly updating the clustering center and sample sampling
range according to the changing threshold λ, thus expanding
the reliable sample. In unsupervised person Re-ID tasks, how
to mine low-confidence samples and classify them accurately
has become a challenging problem. Therefore, the contrast
learning method has become a vital helper.
Ge et al. [9] provided a robust baseline network that
introduces a memory bank and proposes a new selfscheduling contrast learning framework with mixed memory.
Subsequently, Sun et al. [37] optimized the contrast loss
and proposed a contrast learning method based on multigranularity cluster integration to mine the priority information
between pseudo-positive sample pairs. Li et al. [38] proposed
a cluster-guided asymmetric contrast learning method that
successfully combines classification loss and contrast loss by
using cluster structures to guide feature learning in a suitably
designed asymmetric contrast learning framework. In addition,
Dai et al. [10] proposed clustering contrast and introduced a
memory bank to store feature vectors and calculate contrast
loss at the cluster level. Moreover, the real-time memory
updating strategy (RTMem) [39] updates cluster centroids
using randomly sampled instance features, significantly boosting performance in unsupervised person Re-ID.

Recently, several works have explored cross-modality scenarios and multi-source semantic information fusion to
enhance unsupervised person Re-ID. CHCR [40] proposed a
cross-modality hierarchical clustering algorithm coupled with
modality contrastive loss, which establishes cross-modal constraints at the cluster centroid level to promote the generation
of positive sample pairs across visible-infrared modalities,
thereby mitigating the adverse effects of modality discrepancies on feature learning. FuseDSI [41] introduced a diverse
semantic information fusion framework that leverages nonparametric graph convolution to estimate pair-wise similarities
and employs relaxed contrastive loss to integrate hard pseudolabels, pair-wise semantics, and neighborhood semantics as
composite supervision signals. Nevertheless, these methods
predominantly operate at the cluster or mini-batch level,
lacking the capability to precisely locate the most challenging
samples near decision boundaries. In contrast, our proposed
NNSC explicitly identifies the hardest sample pairs at the
instance level through arg min/max operations, accurately
pinpointing the farthest positive samples within the same
class and the nearest negative samples from neighboring
classes. Coupled with the class balance loss, which is theoretically grounded in concave function optimization via Jensen’s
inequality, our approach dynamically repels confusing samples
from ambiguous regions, thereby achieving more fine-grained
feature space partitioning at clustering boundaries.

C. Network Design With ODEs
ResNet [42] has been widely used in various tasks of neural
networks in recent years due to its stable training process and
good performance. It is worth mentioning that ResNet is not
only simple in structure but also very efficient in the design of
residual connections. Therefore, many researchers explore its
good topology structure and improve it further. Zhang et al.
[43] designed a PolyInception module for each residual block
to enhance the network’s generalization ability. These methods
[43], [44] discussed how neural networks should integrate
mathematical systems from a mathematical point of view.
Moreover, some works [13], [45], [46] explained the design
of deep neural networks in terms of differential equations.
The relationship between ResNet and ordinary differential
equations was first discovered in [13]. Specifically, they treated
deep neural networks as a discrete dynamical system, identifying similarities between ResNet and ODE discretization
and laying the foundation for subsequent research on network
design. Subsequently, Lu et al. [14] combined the network with
the discretization of differential equations, studied the linear
multistep solution method of ordinary differential equations,
and proposed the linear multistep architecture. Inspired by
Leapfrog et al. [46] in numerical ordinary differential equations, He et al. [45] designed a network module for image
super-resolution, enhancing the feature representation ability
of the network. As far as we know, this pattern is not applied in
the person Re-ID task. Therefore, in the process of pushing the
Re-ID task to fine granularity, we can use ODEs to design an
efficient residual network topology and embed some function
modules to improve the representation ability and robustness
of the Re-ID model.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:00 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Occluded person Re-Identification with noise injection.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113689

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Occluded person Re-Identification with noise injection
Can Yao a , Xi Du b , Rui Li b , Ye Fan b , Deng Cai c , Shenqi Lai c
a
b

,∗

The School of Computer Science and School of Artificial Intelligence Optics and Electronics (iOPEN), Northwestern Polytechnical University, Xi’an 710072, China
China Mobile Group Jiangsu Co., Ltd., Nanjing 210029, China

c The State Key Laboratory of Computer-Aided Design (CAD) and Computer Graphics (CG), Zhejiang University, Hangzhou 310027, China

ARTICLE

INFO

Keywords:
Re-Identification
Noise learning
Vehicle
Occluded person

ABSTRACT
Occluded person Re-Identification (ReID) has recently been a hot topic. While most existing approaches focus
on developing novel architectures or incorporating additional manual annotations, few studies address the
construction of large-scale datasets, which is a critical factor in the success of deep learning. However,
creating such datasets for person ReID is challenging due to the prevalence of individuals wearing similar
clothing across different camera views, further exacerbated by occlusions. In this study, we propose an
innovative solution by leveraging large-scale vehicle images, which can be automatically associated using
consistent license plate information. Although vehicle images are typically considered noise in person ReID,
we demonstrate that incorporating them into training significantly improves discrimination in occluded ReID.
Specifically, we propose two vehicle noise injection strategies: Epoch-wise Noise Injection (ENI) and Imagewise Noise Injection (INI). ENI periodically alternates the training dataset between person and vehicle images,
while INI synthesizes occluded person images by inserting vehicle segments to generate physically plausible
occlusion patterns. Both strategies deliberately introduce controlled noise during training, bridging the domain
gap between vehicle and pedestrian images while avoiding overfitting. Extensive experiments show that our
method achieves state-of-the-art performance on two occluded person ReID benchmarks: Occluded-Duke and
P-DukeMTMC. Additionally, we validate the approach on four standard person ReID benchmarks, Market1501, DukeMTMC-ReID, CUHK03-NP, and MSMT17, further demonstrating its effectiveness. The code is at
https://github.com/xiaolai-sqlai/NIReID.

1. Introduction
Occluded person re-identification (Re-ID), which focuses on retrieving partially occluded pedestrians across non-overlapping camera
views, has become a crucial research area for intelligent security systems. Although existing methods have made significant progress on
standardized benchmarks, their deployment in real-world scenarios
remains challenging due to two inherent complexities. The primary
difficulty stems from dual occlusion patterns: (1) object-induced occlusion, where environmental elements (e.g., vehicles, buildings) obstruct
visibility, and (2) inter-person occlusion, where pedestrians in dense
crowds obscure one another. Both scenarios lead to the loss of critical
biometric and appearance information. For example, facial features
may be entirely blocked, while distinctive clothing patterns often appear fragmented. This dual degradation mechanism not only forces
models to depend on ambiguous local cues but also causes severe
part misalignment. Consequently, this fundamental conflict makes it
difficult to converge.

Recent research has focused on architectural innovations, including
attention mechanisms and part-based feature alignment, which often
rely on manual annotations like human parsing masks to address occlusion. However, this dependence on annotated pose and parsing data
substantially increases implementation costs. Moreover, the scale of
existing occluded person ReID datasets (e.g., 2000 images in OccludedREID [1]) is several orders of magnitude smaller than that of standard
benchmarks (e.g., 126,441 images in MSMT17 [2]), severely limiting
the generalizability of the models. Current data collection protocols
also prioritize static pedestrian scenes, neglecting the dynamic occlusion patterns caused by moving vehicles that are prevalent in real-world
urban environments. Paradoxically, although vehicle-induced occlusions account for over 40% (statistics in Occluded-Duke [3] query set)
in cross-camera retrieval, vehicle images are typically discarded as
noise in conventional ReID pipelines.
This study challenges the conventional view that vehicle data constitute mere noise in person ReID. Specifically, large-scale vehicle images,

∗ Corresponding author.

E-mail addresses: carlyaocan@gmail.com (C. Yao), duxi@js.chinamobile.com (X. Du), liruisgs2@js.chinamobile.com (R. Li), fanye@js.chinamobile.com
(Y. Fan), dengcai@gmail.com (D. Cai), laishenqi@qq.com (S. Lai).
https://doi.org/10.1016/j.patcog.2026.113689
Received 16 June 2025; Received in revised form 6 April 2026; Accepted 7 April 2026
Available online 17 April 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 179 (2026) 113689

C. Yao et al.

Fig. 1. Our noise injection enhances the Rank-1 accuracy on Occluded-Duke [3] of various vision backbones without additional manual annotations.

(a)

(b)

(c)

(d)

Fig. 2. The visualization of data. (a) Person, (b) Inter-person occlusion, (c) Vehicle-induced occlusion, (d) Our synthetic Vehicle-Person.

which can be readily collected via automated license plate recognition
systems, offer two principal advantages: First, their inherent visual
diversity (e.g., in color and texture) encourages models to focus on discriminative pedestrian features, reducing dependence on global apparel
patterns and mitigating overfitting. Second, vehicle images captured
from varying camera viewpoints introduce additional camera bias and
domain-specific variations, thereby improving model generalization.
Consequently, we incorporate vehicle images into the training process
for occluded person ReID, which can also be interpreted as a form of
noisy learning. We systematically examine their dual function as both
a regularizer and a semantic enhancer.
Our approach consists of two key components: (1) Epoch-wise
Noise Injection (ENI) strategy, which periodically alternates the training dataset between person and vehicle images, serving as a regularizer
to prevent overfitting. (2) Image-wise Noise Injection (INI) strategy, which synthesizes occluded person images by inserting vehicle
segments to create physically plausible occlusion patterns, thereby
enriching the semantic content of the training data. As shown in Fig.
1, based on the above strategies, we could improve the performance
significantly. Besides, we visualize several training images in Fig. 2. Our
synthetic images are very similar to person images with vehicle-induced
occlusion.

Our contributions can be summarized as follows:
• We demonstrate that vehicle data is not merely noise but also
contributes positively to the training process in occluded person
ReID.
• We propose Epoch-wise Noise Injection and Image-wise Noise
Injection strategies, which inject controlled noise to bridge the domain gap between vehicle and pedestrian images while avoiding
overfitting.
• Extensive experiments on two occluded person ReID datasets and
four standard person ReID benchmarks show that incorporating vehicle data yields significant performance improvements,
even surpassing state-of-the-art methods that rely on manual
annotations.
2. Related works
2.1. Person ReID
Early studies on convolutional neural networks (CNNs) established
robust representation learning frameworks for person ReID, achieving
2

Pattern Recognition 179 (2026) 113689

C. Yao et al.

significant progress through global descriptor learning [4] and local
feature alignment strategies. Although these methods perform well in
holistic scenarios, they suffer from inherent limitations in generalization and efficiency. For instance, rigid partitioning strategies (e.g., fixed
stripes in MPPM [5]) assume consistent spatial alignment of body parts
across viewpoints, leading to misalignment when parts are occluded,
deformed, or inconsistently visible. To mitigate these issues, poseguided methods have been proposed, leveraging keypoints or semantic
parsing for improved part alignment. However, such approaches introduce new challenges: auxiliary modules (e.g., pose estimators in
HOReID [6] and co-parsing networks in HCGA [7]) require additional manual annotations and increase inference latency compared to
standard CNNs.

the concept of ‘‘Positive-incentive noise’’ (𝜋-noise), which can reduce
task complexity under certain conditions. This framework demonstrates
that even random noise can benefit learning by simplifying the task
entropy when applied appropriately.
In computer vision, this principle manifests in various forms. Adversarial training deliberately introduces worst-case perturbations to
enhance model robustness against distribution shifts. AutoAugment
and RandAugment search for optimal noise policies that expand the
effective training distribution without compromising label integrity.
These methods validate that controlled noise injection can improve
generalization beyond traditional data augmentation.
Cross-domain learning presents a compelling application of noise
principles. While domain gaps are traditionally considered harmful
[20], recent studies demonstrate that they can be beneficial under appropriate constraints. For instance, progressive domain adaptation [21]
reveals that incrementally introducing target domain characteristics
as ‘‘noise’’ facilitates the learning of domain-invariant features. Similarly, CleanAdapt [22] treats source-free video domain adaptation as
a noisy-label learning problem and improves training through reliable pseudo-label refinement. These studies suggest that structured
cross-domain variation, when properly controlled, can act as useful regularization rather than pure interference. This insight is also relevant to
person re-identification. Recent work has shown that background clutter, occlusion, and model imperfections can introduce substantial noise
into person representations. For example, Pose2ID [23] reduces such
noise through feature centralization and identity-preserving generation,
improving representation stability across standard, cross-modality, and
occluded ReID settings. In contrast, our method introduces structured
cross-domain noise directly during training, allowing the network to
learn occlusion-robust representations from augmented observations.
Our approach extends these insights by treating vehicle images as
structured noise for person ReID. Unlike random perturbations, vehicle
data provides semantic consistency through license plate associations
while introducing appearance variations that enhance occlusion robustness. The person–vehicle copy–paste strategy bridges domain gaps
by creating hybrid samples that preserve identity information while
diversifying visual patterns. This represents a novel instantiation of
𝜋-noise theory, where cross-domain data serves as constructive noise
that reduces the complexity of occluded person ReID without requiring
additional human annotations.

2.2. Occluded person ReID
The introduction of Vision Transformers marked a turning point in
handling occlusion challenges. TransReID [8] pioneered this direction
by treating images as patch sequences and leveraging self-attention to
capture long-range dependencies. Building upon this foundation, subsequent works have explored various strategies to enhance occlusion
robustness while addressing computational efficiency.
DRL-Net [9] decouples identity-relevant features from occlusion
patterns through disentangled representation learning. By introducing
semantic preference queries, it learns to focus on discriminative regions without explicit part annotations. Data-centric approaches offer
complementary solutions to architectural innovations. SPT [10] synthesizes realistic occlusion scenarios by transferring salient person regions
onto diverse backgrounds, creating a rich augmentation pipeline. The
method employs IoU-based filtering to ensure occlusion realism while
maintaining identity consistency. CAAO [11] advances this concept
through reinforcement learning, where an occlusion controller learns
optimal augmentation policies based on model feedback. The controller
adapts occlusion patterns to current model weaknesses, creating a
curriculum learning effect.
Recent studies have further advanced occluded person ReID from
several complementary perspectives. Cross-erasure [12] enhanced network improves robustness by strengthening feature learning under
partial visibility and occlusion-induced corruption. Occlusion-attribute
based methods [13] explicitly exploit occlusion cues to guide identity learning under incomplete observations. Feature complement and
dual-attention based approaches [14] enhance the interaction between
visible body cues and attention refinement, improving discriminability
when informative regions are partially missing. In addition, maskaware hierarchical transformer architectures [15] incorporate mask semantics to emphasize non-occluded body parts and suppress corrupted
regions.
Despite their effectiveness, these recent methods still mainly rely
on explicit occlusion modeling, auxiliary cues, or more sophisticated
architectures. In contrast, our method improves occluded ReID from a
data-centric perspective by injecting structured cross-domain noise during training, without requiring extra manual annotations or specialized
occlusion masks.

3. Proposed approach
As illustrated in Fig. 3, our proposed framework addresses the
challenge of occluded person ReID through a noise injection strategy
based on a dual-branch architecture [5]. The core innovation lies in
reconceptualizing vehicle data not as mere noise but as a valuable
training resource in occluded scenarios.
We think that controlled noise injection during training improves
model robustness and generalization. Specifically, we leverage largescale vehicle datasets, readily available from automated license plate
recognition systems, to augment conventional person ReID training.
The training begins by preparing two distinct data sources: person
images from standard ReID datasets and vehicle images with consistent
license plate associations. Our noise injection strategy dynamically
modulates input composition, alternating between pure vehicle training
phases and mixed person-vehicle phases. This approach serves a dual
purpose: it regularizes the model to prevent overfitting to specific appearance patterns while enriching semantic diversity through realistic
occlusion simulation.
The extracted features from both branches are fused into a comprehensive representation that captures both holistic appearance and discriminative local patterns. This dual-representation approach ensures
robust performance across diverse occlusion scenarios while maintaining high discriminative capability for person ReID.

2.3. Noise data learning
Noise-aware learning has been explored in different forms across
representation learning and recognition tasks. Click Prediction [16]
highlights early noise-aware representation learning. SFE-Net [17] provides a complementary view on robustness under sparse and noisy
signals. GAGA [18] contextualizes our work from an adversarial robustness perspective. These studies collectively suggest that noise should
not always be viewed as purely harmful, but can sometimes play a
constructive role in improving representation quality and robustness.
Recent theoretical work by Li [19] further challenges the conventional
view that noise is always detrimental to learning systems. Li introduces
3


codex
前三篇已经读完。当前能看到一个值得注意的趋势：B 类文章常把“噪声、空间、边界样本”重新解释成有用监督，而不是只堆模块。继续读跨模态补全、光照距离度量和无监督代理挖掘这三篇。
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
/bin/zsh -lc "pdftotext -l 3 'Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708

1

Occlusion-aware Cross-modality Completion
Network for Occluded Visible-Infrared Person
Re-Identification
Jian Yu, Yujian Feng, Shuai You, Zhengjun Jing, Fei Wu, Yimu Ji, Xiao-yuan Jing

Abstract—Occlusion is a key challenge in real-world person reidentification (ReID), as partial occlusions often remove critical
identity cues. Most existing methods focus on feature extraction
for visible regions while ignoring the holistic contextual information of pedestrians, leading to incomplete feature representations.
This issue is particularly severe in visible-infrared ReID (VIReID), where spatial inconsistencies of occluded regions and
modality-specific feature degradation jointly exacerbate feature
misalignment, but has received little attention in previous studies. Therefore, we propose an Occlusion-Aware Cross-Modality
Completion Network (OCCNet) to explicitly complete missing
identity cues, which integrates three complementary components.
The Graph-guided Feature Completion Transformer (GFCT)
embeds a dynamically constructed semantic graph into selfattention to complete occluded tokens within each modality,
while the Semantic-Aware Region Completion Module (SRCM)
aggregates tokens into coherent regions and adaptively transfers complementary cues across modalities. To ensure semantic
coherence, a Multi-Distribution Consistency Alignment (MCA)
loss is employed to regularize token- and region-level completions
toward a unified distribution. By integrating these components,
OCCNet recovers missing identity information at multiple levels
and mitigates modality gaps caused by occlusion. To evaluate its
effectiveness under realistic occlusion scenarios, we first construct
two realistic occluded benchmarks, Occluded-SYSU-MM01 and
Occluded-RegDB, on which extensive experiments demonstrate
that our OCCNet outperforms state-of-the-art methods.
Index
Terms—Occluded
visible-infrared
person
reidentification, Cross-modality feature completion, Distribution
alignment, Occluded VI-ReID datasets.

Woman
Black backpack
Short hair
...

Single
Modality

Original
Image

Pose
Estimation

Human
Parsing

Instance
Segmentation

Attribute
Annotations

(a)

Vis
Modality

Intra-modality
Complete

Matching

Complete
Region
Inter-modality
Complete

Complete
Region

NIS
Modality

Matching
Intra-modality
Complete

(b)

Fig. 1: The difference between the existing methods and
our approach. (a) Existing methods. (b) Our Occlusion-aware
Cross-modality Completion Network (OCCNet).

I. I NTRODUCTION
Visible-infrared person re-identification (VI-ReID) plays a
critical role in intelligent surveillance systems by matching
the same individual across visible and near-infrared cameras.
Although numerous VI-ReID methods [1]–[5] have been proposed in recent years, most of them assume full body visibility. In real-world scenarios, however, occlusions (e.g., signs,
vehicles or other pedestrians) frequently occur in crowded
environments such as railway stations, schools, and streets,
leading to severe degradation of recognition performance.
This motivates the study of occluded visible-infrared person
Jian Yu, Yujian Feng, and Zhengjun Jing are with the School of Computer Engineering,
Jiangsu University of Technology, Changzhou, China (email: yujian 928@163.com;
fengyujian 904@163.com; jzjing@jsut.edu.cn).
Shuai You and Xiao-Yuan Jing are with the School of Computer, Wuhan University,
Wuhan, China (email: youshuai 666@163.com; jingxy 2000@126.com).
Fei Wu is with the College of Automation, Nanjing University of Posts and Telecommunications, Nanjing, China (email: wufei 8888@126.com).
Yimu Ji is with the School of Computer Science and Technology, Nanjing University
of Posts and Telecommunications, Nanjing, China (email: jiym@njupt.edu.cn).
Yujian Feng is the corresponding author.

re-identification (Occluded VI-ReID), which addresses crossmodality matching under partial occlusion. However, to date,
only a single work has explicitly explored this problem [6].
Local detail preservation and cross-modality alignment are
both important in complex multi-modality scenarios [7]–[9].
These requirements are particularly critical for Occluded VIReID, which presents two main challenges. (1) Occlusion
leads to the loss of critical identity cues such as facial features, clothing textures, or distinctive accessories, resulting in
incomplete and less discriminative pedestrian representations;
(2) Occlusion significantly exacerbates modality discrepancies,
mainly caused by two factors: On the one hand, the spatial
locations of occluded regions often differ across modalities,
resulting in positional misalignment. On the other hand,
occlusion affects each modality in different ways, leading
to inconsistent feature degradation. For example, in visible
light images, occlusions usually obscure visual cues such as
color and texture, while in infrared images, occlusions often

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708

2

mask structural information such as contours and shapes.
This asymmetry significantly increases the difficulty of crossmodality feature alignment.
As illustrated in Fig. 1(a), most existing occluded ReID
methods [10]–[13] are developed from a single-modality perspective. They typically handle occlusion by discarding the
occluded parts and extracting features only from the visible body regions, often relying on auxiliary cues such as
pose estimation, human parsing, instance segmentation, or
attribute annotations. While effective to some extent, these
approaches inevitably neglect the holistic contextual semantics
of pedestrians, such as the overall silhouette, the swing range
of the arms, and the movement posture of the legs, which
are crucial for discriminative alignment. As a result, the
learned representations remain incomplete and less robust
under severe occlusions. This limitation also exists in the only
study on Occluded VI-ReID, OCMF [6], and becomes even
more critical in the cross-modality scenario, where the visible
regions in different modalities are often spatially inconsistent.
Such asymmetry easily leads to severe feature misalignment
and degraded retrieval performance.
In contrast, as shown in Fig. 1(b), our method addresses
occlusion from a completion perspective rather than a discarding one. Specifically, we propose to fully exploit both intramodality contextual cues and inter-modality complementary
information to complete the occluded regions. By jointly completing missing features within modality and across modalities,
our framework generates semantically holistic and structurally
consistent representations, thus enabling more robust and
discriminative cross-modality matching under occlusion.
To implement this framework, we propose an OcclusionAware Cross-Modality Completion Network (OCCNet), which
explicitly completes missing identity cues by integrating
token-level and region-level completion with distribution consistency constraints. Specifically, OCCNet is composed of
three complementary modules. First, the proposed Graphguided Feature Completion Transformer (GFCT) incorporates
dynamically constructed semantic graphs into self-attention to
complete occluded tokens, ensuring that completed features
are structurally consistent with surrounding contexts within the
same modality. Second, a Semantic-Aware Region Completion
Module (SRCM) is designed to aggregate tokens into semantic
regions and performs adaptive cross-modality completion,
allowing complementary cues from another modality to fill
missing information and alleviate spatial misalignment.
To further ensure semantic coherence across different completion pathways, we proposed a Multi-Distribution Consistency Alignment (MCA) loss to align token- and regionlevel completions with a pseudo-holistic reference through
symmetric distribution matching. By jointly modeling these
three components, OCCNet produces semantically holistic and
structurally consistent representations, significantly improving
robustness against occlusion in cross-modality matching.
The main contributions of this paper can be summarized as
follows:
(1) A Graph-guided Feature Completion Transformer
(GFCT) is proposed to embed dynamic semantic graphs into

self-attention, constraining interactions to semantically relevant pairs and enabling robust token-level feature completion.
(2) A Semantic-aware Region Completion module (SRCM)
is proposed to cluster tokens into semantic regions and perform
adaptive cross-modality feature completion, effectively alleviating spatial misalignment and modality-specific degradation.
(3) A Multi-Distribution Consistency Alignment (MCA)
loss is designed to align token- and region-level completions
with a unified pseudo-holistic reference via symmetric KL
divergence, improving multi-path semantic coherence.
(4) Two new benchmarks, Occluded-SYSU-MM01 and
Occluded-RegDB, are built with realistic occlusions, which
are the first real occluded VI-ReID datasets. Extensive experiments show that our method achieves state-of-the-art performance in cross-modality occlusion scenarios.
II. R ELATED W ORK
A. Visible-infrared Person Re-identification
Visible-infrared person re-identification (VI-ReID) aims to
match pedestrian identities across visible and infrared cameras. Its main challenges lie in the inter-modality discrepancy caused by different imaging mechanisms and the intramodality variation caused by viewpoint changes.
To address these challenges, many VI-ReID methods have
been proposed from the perspective of modality alignment.
For example, Zhang et al. [1] investigated the real-world label
noise problem in VI-ReID and proposed a robust hybrid loss to
improve model robustness under instance-dependent and realworld noisy annotations. Pang et al. [2] proposed a hierarchical
clustering and refinement framework for unsupervised VIReID. Wu et al. [14] developed a modality- and instance-aware
adaptation framework to alleviate cross-modality discrepancy.
Liang et al. [4] introduced a multi-level joint alignment method
to reduce modality discrepancy in image and feature spaces.
Zhao et al. [15] proposed a historical consistency learning
strategy to correct noisy labels in VI-ReID.
However, the effectiveness of these methods relies on
obtaining complete, unobstructed images. When pedestrians
are occluded by trees, buildings, vehicles, text, and other
factors, these methods fail to effectively extract discriminative
features. Therefore, this paper proposes an Occlusion-aware
Cross-modality Completion Network (OCCNet) to complete
the features of pedestrians in occluded areas, thus improving
cross-modality matching performance.
B. Occluded Person Re-identification
In real-world scenarios, pedestrians frequently encounter
occlusion from various obstacles, necessitating the matching
of occluded pedestrians with those in the gallery, defining
the task of Occluded Person ReID [16]–[21]. The traditional
methods focus on single-modality pedestrian matching for
visible light/RGB images. For example, Wang et al. [16]
proposed a text-guided hierarchical context blending network
for occluded person re-identification, which integrates textual
information with visual features to enhance contextual representation under occlusion. Chen et al. [17] introduced a
hierarchical token-aware reconstruction framework to alleviate

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708

3

Fig. 2: Overview of the proposed OCCNet. Modality-specific features are first extracted by a ResNet50 backbone and then
enhanced through two completion stages: the Graph-guided Feature Completion Transformer (GFCT) for token-level completion
and the Semantic-Aware Region Completion Module (SRCM) for cross-modality region-level completion. The network is trained
with a total loss that enforces consistency between token- and region-level completions, resulting in discriminative features for
robust cross-modality person retrieval.

modality discrepancy in VI-ReID. Qiu et al. [18] proposed a
hierarchical middle-feature learning network to build a more
reliable common feature space for VI-ReID. Zhang et al. [19]
developed a mask-aware hierarchical transformer to improve
visible-part localization and feature completion under occlusion. Dong et al. [20] addressed Occluded Person ReID from
an adversarial defense perspective, while further introduced a
multi-view framework that transfers complementary identity
cues to single-image representations [21].
Although these methods have advanced occlusion-aware
representation learning or cross-modality feature modeling,
they remain insufficient for occluded VI-ReID. This task
must simultaneously address missing identity cues caused
by occlusion and feature inconsistency caused by modality
variation, resulting in more severe cross-modality misalignment. Moreover, single-modality completion cannot exploit
complementary cues from another modality, while existing
VI-ReID methods generally emphasize modality alignment
rather than explicit recovery of missing identity information.
To address these issues, our OCCNet jointly performs intramodality and cross-modality feature completion, producing

more complete and discriminative representations for robust
matching under occlusion.
III. O UR A PPROACH
In this section, we provide a detailed introduction to the proposed Occlusion-aware Cross-modality Completion Network
(OCCNet). We first present the Graph-guided Feature Completion Transformer (GFCT), which performs token-level feature
completion with graph-guided attention. Then, we describe the
Semantic-aware Region Completion Module (SRCM), which
extends the completion to region-level through cross-modality
alignment. Finally, the Multi-distribution Consistency Alignment (MCA) loss is introduced to enforce consistency among
different completion paths.
A. Overview
Fig. 2 illustrates the overall architecture of the proposed
OCCNet. Given a pair of visible and near-infrared images,
a shared ResNet50 [22], pre-trained on ImageNet [23], is
vis
employed as the backbone to extract feature maps Xcnn

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


exec
/bin/zsh -lc "pdftotext -l 3 'Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

2095

Optimal Proxy Mining Contrastive Network for
Unsupervised Person Re-Identification
Ge Cao , Member, IEEE, Qing Tang , Member, IEEE, Xuan-Thuy Vo , Member, IEEE,
Adri Priadana , Member, IEEE, and Kang-Hyun Jo , Senior Member, IEEE
Abstract—Unsupervised person re-identification (Re-ID) performance enhancement hinges on extracting the most informative
features from unlabeled person datasets. In recent approaches,
proxy-based contrastive learning with awareness of camera labels
has been adopted for model training, thereby achieving highly
promising results. However, inappropriate selections of contrastive pairs can significantly degrade the performance of these
models. To address this issue, we propose the Optimal Proxy Mining Contrastive Network (OPMCN), a novel framework designed
to strategically optimize the selection of proxies for positive and
negative pair formation, thus enhancing the efficacy of contrastive
training. The OPMCN framework proposes two specific contrastive losses: Hardest Camera Proxy Mining (HCPM) and False
Negative Proxies Mining (FNPM), each essential for enhancing
model performance in unsupervised settings. The HCPM loss
targets proxies from the most challenging cameras to maximize
semantic differences between pairs while ensuring minimal background shifts. In contrast, the FNPM loss counters noise in
pseudo labels by prioritizing similarity rankings over clustering
results to effectively identify and correct false negatives among
proxies. Moreover, we have developed the Pyramid Kernel Global
Context (PKGC) block, which employs an attention mechanism
that focuses on identity-invariant semantic cues in instances.
This module utilizes optimally sized convolutional kernels to
enhance identity recognition consistency across camera-based
variations, thereby improving the precision of feature extraction.
Experimental results on several popular datasets prove that our
work surpasses existing unsupervised person Re-ID approaches
to a remarkable extent.
Index Terms—Unsupervised person re-identification, contrastive learning, attention mechanism.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) aims at precisely identifying the target pedestrian from a vast pool of instances. It
serves as a cornerstone in non-overlapping cross-camera video
surveillance systems, offering essential support for various
downstream computer vision tasks by enabling person ID
tracking. Over the past few years, deep learning-based person
Re-ID methods [1], [2], [3], [4] have become highly dependent
on the representational quality of the training datasets, causing
Received 10 October 2024; revised 12 May 2025; accepted 24 August 2025.
Date of publication 27 August 2025; date of current version 5 February 2026.
This result was supported by the “Regional Innovation System & Education
(RISE)” through the Ulsan RISE Center, funded by the Ministry of Education
(MOE) and the Ulsan Metropolitan City, Republic of Korea. (2025-RISE07-001). This article was recommended by Associate Editor Y. S. Rawat.
(Corresponding author: Kang-Hyun Jo.)
Ge Cao, Xuan-Thuy Vo, Adri Priadana, and Kang-Hyun Jo are with the
Department of Electrical, Electronic and Computer Engineering, University
of Ulsan, Ulsan 44610, Republic of Korea (e-mail: acejo@ulsan.ac.kr).
Qing Tang is with the Department of Research and Development center,
INTERX, Ulsan 44542, Republic of Korea.
Digital Object Identifier 10.1109/TCSVT.2025.3603199

a dramatic decline in the test performance of these models
when deployed in real-world surveillance environments. To
combat this decline, it is essential to collect data from real
scenarios; however, manual annotation of such data is both
time-consuming and labor-intensive.
To mitigate this issue, unsupervised learning offers a viable
solution that enhances testing performance without the ponderous manual annotations. Furthermore, with ongoing research
in person Re-ID under unsupervised training scheme, numerous approaches [5], [6], [7], [8], [9] have shown progressively
improved outcomes on public datasets, steadily approaching
the efficacy of supervised training models. Unsupervised person Re-ID is divided into unsupervised domain adaptation
[10], [11], [12], [13], [14], [15], [16] and purely unsupervised
methods [17], [18], [19], [20] based on the use of labeled data
from other domains. The latter, not relying on any annotated
labels, presents greater challenges and offers increased practical value. These methods underpin foundational technologies
for cross-camera tracking [21], [22] and pose estimation
[23], [24] and are integral to developing intelligent video
surveillance and smart traffic systems [25], [26]. In this paper,
we concentrate on developing purely unsupervised methods to
enhance flexibility in person Re-ID.
Recent works indicate that pseudo label based methods
[5], [6], [27] facilitate the stable and efficient training for
unsupervised training for person Re-Identification, while typically adhering to the following pipeline: (1) obtain the feature
representation through a designed extractor; (2) generate the
pseudo labels by clustering methods (i.e. DBSCAN [28]); (3)
optimize the extractor with a set of loss functions applying
contrastive learning. As training progresses, the model incrementally masters the semantic features of analogous instances
and more adeptly articulates the characteristics of these
instances.
With the aim of achieving notable accuracy, prior works
[6], [11], [13], [15], [18], [20], [29] have utilized camera
labels to further split identity clusters into multiple proxies,
enhancing the granularity and effectiveness of person Re-ID
models through what is known as proxy-based contrastive
learning [6], [19], [20], [30]. Among these, CAP [6] has
played a pivotal role by identifying the high intra-camera
similarity of instances captured by the same camera, based
on inherent characteristics. It proposed camera-aware proxies
and designed both intra-camera and inter-camera contrastive
learning strategies to improve ID-discrimination within and
across cameras. CAP’s innovative approach has significantly
influenced many subsequent proxy-based contrastive learning
methods, cementing its importance in advancing the field.

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.

2096

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

Although the existing methods have achieved promising performance in unsupervised person Re-ID, they are still unable
to resolve the following issues:
• Misaligned contrastive pairs: The core of contrastive
learning in purely unsupervised person Re-ID lies in
selectively choosing appropriate positive and negative
samples to aid semantic distinction, a process where nonselective methods may fail to yield optimal results due to
inherent variability and noise in pseudo labels. Such noise
often misrepresents the ID-distinctive feature of instances,
leading to misalignment that introduce false negatives and
undermine learning by reinforcing semantical noise. The
failure to accurately adjust both positive and negative
instances in contrastive learning significantly limits the
effectiveness and generalization of the model, a challenge
that current methods rarely address.
• Insufficient global information: In the Re-ID task,
instances are often captured from cluttered environments, leading models to inadvertently neglect identitydistinctive features and erroneously focus on background
noise. The capability of convolution neural networks
(CNNs), including architectures like ResNet [31], to
extract complex or subtle features is limited. These
networks tend to prioritize local feature extraction, inadequately addressing the integration of global contextual
information. Consequently, current methodologies are
insufficiently robust in extracting pertinent information
for effective model performance.
To address the challenges outlined, this paper introduces the
Optimal Proxy Mining Contrastive Network (OPMCN) for the
unsupervised person Re-ID task. Within OPMCN, we develop
two specialized contrastive loss functions: the Hardest Camera
Proxy Mining (HCPM) and the False Negative Proxies Mining
(FNPM). HCPM operates under the assumption that pseudo
labels are accurate, selecting proxies from the hardest camera.
These proxies, although in the same cluster as the query,
exhibit the largest semantic distinctions, thus providing rich
discriminative information. This method effectively reduces
background variations, allowing the model to concentrate on
ID-distinctive features. Conversely, FNPM directly addresses
the performance degradation due to the inevitable noise in
pseudo labels by introducing an anti-noise similarity ranking
system that refines the selection of contrastive pairs. Together,
HCPM and FNPM identify optimal proxies, substantially
enhancing model training. Subsequently, the paper presents
the Pyramid Kernel Global Context (PKGC) block, which
supports the backbone Pyramid Kernel Attention (PKA) network by enhancing key features and facilitating the integration
of global information. This module pyramidally adjusts the
processing of spatial information according to the feature map
size, effectively managing different scales of data and aiding
the network in recognizing identity-distinctive features while
adapting to background changes.
In summary, the main contributions of our model are as
follows:
• We introduced the Hardest Camera Proxy Mining
(HCPM) approach to mine proxies from the most

challenging cameras, leveraging the richest discriminative
information within each cluster to focus the model on
learning identity-invariant features while reducing background noise.
• The False Negative Proxies Mining (FNPM) method
counters the adverse effects of noisy pseudo labels, which
mines potential false negatives and refines the selection
of contrastive pairs to improve overall accuracy and
robustness.
• The PKGC block is proposed to enhance key features
by pyramidally adjusting kernel sizes, enabling the backbone network to better integrate global information and
extract identity-distinctive features. This design improves
adaptability and recognition accuracy.
• Experiments conducted across multiple large-scale Re-ID
datasets confirm that the proposed method significantly
enhances the performance of purely unsupervised person
Re-ID, surpassing other approaches by a considerable
margin.
II. R ELATED W ORKS
A. Unsupervised Person Re-Identification
With the rapidly increasing demand for social security and
public services, video surveillance systems, as the cornerstone
of smart cities, have attracted widespread attention [21], [22],
[23], [24], [25], [26]. For the person Re-ID system, obtaining
instances of pedestrians is not difficult, but the challenge lies
in the fact that modern industry is incapable of affording
the cost of manually labeling a large number of pedestrian
instances. Fortunately, the emergence of some large-scale
datasets [32], [33], [34] has shown the possibility of not
relying entirely on annotated instances for researchers, i.e.,
the unsupervised person Re-ID. The unsupervised Re-ID is
divided into unsupervised domain adaptation (UDA) methods
[5], [10], [11], [12], [13], [14], [15], [16] and purely unsupervised methods [6], [7], [8], [17], [18], [19], [20], based on
whether it relies on annotated data beyond the target unlabeled
domain.
Existing methods commonly apply the pseudo label based
pipeline [5], [6], [27] for training and achieving notable
performance. By utilizing camera factors for unsupervised
learning, prior works have focused on different aspects of
model optimization. Although almost all methods directly
apply DBSCAN [28] and K-means as the clustering algorithm,
some methods [17], [35], [36] proposed new clustering methods to generate better pseudo label results. BUC [17] and
HCT [37] implement learning in a camera-agnostic manner,
effectively capturing similarity within identities but often overlooking variance within the same identity due to changes in
camera angles. Conversely, DARC [35] segments the target
domain into distinct camera-specific groups to enhance global
clustering. Similarly, STS [36] generates a camera offset
matrix alongside a unified distance matrix to mitigate camerainduced variances. Alternatively, several approaches [6], [19],
[20] bifurcate the Re-ID process into intra- and inter-camera
learning phases, allowing the discriminative capabilities honed
within individual cameras to bolster identity matching across

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.

CAO et al.: OPTIMAL PROXY MINING CONTRASTIVE NETWORK FOR UNSUPERVISED PERSON RE-IDENTIFICATION

different cameras. Notably, CAP [6] introduces camera-aware
proxies to address intra-ID variance, pursuing unsupervised
learning from both intra- and inter-camera viewpoints. Additionally, IICS [20] establishes proxy memories based on
separate clustering outcomes for each camera, facilitating the
computation of intra- and inter-camera similarities to inform
model training. Beyond these real-image-based approaches,
Privacy-Protected Re-ID [38] introduces a novel paradigm
where no real person images are used during training, relying
instead on synthetic samples generated from privacy-irrelevant
cues. However, these methods do not systematically select
positive and negative samples for contrastive learning while
incorporating camera information. This paper introduces the
optimal proxy mining contrastive method to address this issue
and provides an analysis and results of various positive and
negative sample selection methods.
B. Metric Learning With Proxies
Metric learning is crucial for person Re-ID and various
other vision tasks. The Proxy-NCA [39] method introduced
the use of proxies to represent groups of instances, enhancing
the contextual information available for similarity calculations.
Subsequent research [6], [19], [20] has confirmed the value
of proxies in unsupervised person Re-ID, employing proxybased strategies to divide each cluster into multiple proxies
based on camera labels. This division helps better capture local
structures, addressing the substantial inter-camera variability
inherent in the Re-ID task. This study further refines this
approach by partitioning clusters into varying numbers of
proxies according to camera views. Additionally, it introduces
two strategic contrastive learning loss functions designed to
optimize unsupervised training.
C. Contrastive Learning and Hard Negative Mining
Contrastive learning (CL), a subset of metric learning
techniques, focuses on assessing sample similarities and has
recently gained popularity in unsupervised representation
learning [40], [41], [42]. To improve the robustness of learned
representations, recent methods such as MixIR [41] enforce
invariance by aligning the prediction of mixed augmented
views with aggregated features, encouraging consistency under
stronger intra-instance variations. Key contrastive losses like
InfoNCE [42] are structured non-parametrically, designed to
attract positive samples and repel negative ones. Originally
centered on individual instances, these losses have evolved to
encompass prototypes [43] and proxies [44]. Moreover, CL
has been extensively adopted for both Unsupervised Domain
Adaptation (UDA) [5], [10], [45] and purely unsupervised
person Re-ID tasks [6], [8], [19], [20]. For example, SpCL [5]
integrates a hybrid memory, facilitating contrastive learning
across various levels: source-domain classes, target-domain
clusters, and individual unclustered instances. Meanwhile,
CAP [6] employs a proxy-level memory bank to enable both
intra- and inter-camera contrastive learning at the proxy level,
whereas ICE [19] enhances both SpCL and CAP by incorporating instance-level contrastive learning elements. This
revision avoids direct repetition, restructures sentences for

2097

clarity, and refines the description of how contrastive learning
has been applied in different research contexts.
In contrast, this paper not only applies CL in unsupervised
training but also utilizes a hard negative mining (HNM) algorithm for selecting suitable negative samples. A key challenge
in CL is the selection of positive and negative samples for
effective comparison. Despite the introduction of several hard
negative mining strategies [46], this issue continues to be
unresolved in the field. CAP [6] applied this scheme when
computing the inter-camera contrastive learning and proved
the performance of HNM but lacked deep exploration. In
this work, we apply HNM with camera information on two
contrastive loss functions, which demonstrates notable performance in unsupervised training.
D. Attention-Based Person Re-ID
The attention mechanism focuses on identifying salient
regions within images, which is crucial for extracting discriminative information from visual data. Various attention-based
models [47], [48], [49], [50] enhance feature extraction by
prioritizing spatial and channel-wise information that is crucial
for recognizing contextual and detailed aspects within images.
SEA [47] enhances specific feature activations by adaptively reweighting channel weights based on their importance.
CBAM [48] applies attention sequentially across channels and
spatial dimensions to focus on relevant features selectively.
CA [49] encodes positional information to capture long-range
dependencies, enhancing attention across spatial features. GC
[50] utilizes global context to enrich local features, aggregating
comprehensive global information into a compact descriptor.
In person Re-ID field, prior works [27], [51], [52] also
attempt to improve feature extraction by attention mechanism.
Reference [27] integrated SEA and CA to capture attention
in three dimensions, while HOA [51] and RGA [52] have
developed attention modules that leverage complex, highorder statistical information to capture global structural details,
thereby generating more discriminatively effective attention
proposals. Beyond these CNN-based mechanisms, AGCL [53]
introduces a graph-based attention strategy for partial Re-ID,
where body keypoints and attributes are used to guide identityconsistent representation learning through structured message
passing. While AGCL relies on auxiliary annotations and is
designed for partial-body inputs, it shares our goal of refining
semantic features for identity alignment—achieved in our case
through a fully unsupervised, attention-driven module without
external supervision. In this paper, we introduce the Pyramid
Kernel Global Context block to refine features at every spatial
position, effectively managing the balance between spatialchannel attention and computational demands. This module
adeptly extracts identity-distinctive semantic information for
person Re-ID tasks by incorporating spatial insights to adjust
channel attention throughout the feature map, thereby enhancing performance.
III. M ETHODOLOGY
As depicted in Fig. 1, the proposed OPMCN framework
includes three important parts, the Hardest Camera Proxy

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Optimal Illumination Distance Metrics for Person
Re-Identification in Complex Lighting Conditions
CHAO WANG, ZHONGYUAN WANG, RUIMIN HU, and XIAOCHEN WANG, National
Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University,
Wuhan, China
WEN ZHOU, Nanjing University of Finance and Economics, Nanjing, China
Person re-identification is extensively applied in public security and surveillance. However, environmental factors like time and location often lead to varying lighting conditions in captured pedestrian images, significantly
impacting identification accuracy. Current approaches mitigate this issue through lighting transformation
techniques, aiming to normalize images to a standard lighting condition for consistent person re-identification
results. Yet, these methods overlook the fact that different content may hold distinct identification values
under diverse lighting conditions. To address this, we conducted an analysis on the identification distance
between images of the same or different pedestrians under pre-defined lighting conditions. From this analysis,
we introduce the concept of optimal lighting: a condition where the distance between image pairs is minimized
compared to other lighting scenarios. We propose utilizing this optimal lighting distance in the image retrieval
process for final ranking. Our study, validated on synthetic datasets Market-IA and Duke-IA, demonstrates
that optimal lighting is independent of image texture information. Each image pair exhibits a unique optimal
lighting, yet consistently shows a minimum distance value.
CCS Concepts: • Computing methodologies → Scene understanding; Computer Vision; Activity
recognition and understanding; • Information systems → Data mining;
Additional Key Words and Phrases: Person re-identification, Complex Lighting, Optimal Illumination Distance
ACM Reference format:
Chao Wang, Zhongyuan Wang, Ruimin Hu, Xiaochen Wang, and Wen Zhou. 2025. Optimal Illumination
Distance Metrics for Person Re-Identification in Complex Lighting Conditions. ACM Trans. Multimedia Comput.
Commun. Appl. 21, 1, Article 26 (January 2025), 18 pages.
https://doi.org/10.1145/3700771

This work was supported by the National Nature Science Foundation of China (Nos. U22A2035, 62371350, 62071339,
U1903214, and 2042023kf1033).
Authors’ Contact Information: Chao Wang, National Engineering Research Center for Multimedia Software, School of
Computer Science, Wuhan University, Wuhan, China; e-mail: snipercwang@whu.edu.cn; Zhongyuan Wang (corresponding
author), National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University,
Wuhan, China; e-mail: wzy_hope@163.com; Ruimin Hu (corresponding author), National Engineering Research Center for
Multimedia Software, School of Computer Science, Wuhan University, Wuhan, China; e-mail: hrm@whu.edu.cn; Xiaochen
Wang, National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University,
Wuhan, China; e-mail: clowang@163.com; Wen Zhou, Nanjing University of Finance and Economics, Nanjing, China;
e-mail: zw_academic 08@163.com.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/1-ART26
https://doi.org/10.1145/3700771
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.

26:2
1

C. Wang et al.

Introduction

Person Re-Identification (Re-ID) [1, 3, 5], which is widely utilized in public safety and security
domains, retrieves images of the same pedestrian from different non-overlapping camera regions
based on a given pedestrian image. Although existing Re-ID tasks perform exceptionally well in
pedestrian recognition under slightly changing lighting conditions, real-world scenarios present
significant challenges due to complex lighting variations. Pedestrians may appear at different times
and locations in different cameras, leading to diverse illumination information in the captured
images. Such illumination variations pose a major obstacle to recognition performance [28]. Consequently, there is an urgent need to develop Re-ID techniques that are highly robust to lighting
variations and specifically tailored for real-world applications.
The article investigates the task of person Re-ID in complex lighting scenarios, particularly for
pedestrians. The task, as illustrated in Figure 1, can be defined as follows: Given an image of a
pedestrian taken under unknown lighting conditions, the objective is to retrieve all images of the
same pedestrian from a gallery consisting of images captured by cameras located in non-overlapping
areas with varying lighting conditions.
Compared to traditional Re-ID tasks, Re-ID under complex lighting scenarios presents additional
challenges due to lighting noise. Illumination changes can alter the color representation of pedestrian
images, making it unreliable to use color as a distinguishing feature. Furthermore, significant
lighting differences between images can cause the intra-class distance (distance between images
of the same pedestrian) to be larger than the inter-class distance (distance between images of
different pedestrians). To address this, researchers have attempted to extract pedestrian features
that remain stable under varying lighting conditions [2]. For example, Kviatkovsky et al. note that
while color is susceptible to lighting changes, transforming the color space can mitigate this effect.
They propose a multi-dimensional histogram method to describe image content [12]. Yu et al. argue
that color distribution is a multi-modal structure that can remain consistent under different lighting
conditions and suggest using shape content to describe this structure as an identification feature
[27]. However, these shallow, content-based semantic features are only effective when there are
significant differences in clothing color and type. They are less effective when pedestrians wear
similar clothing.
To enable effective pedestrian recognition, deep semantic representation features are crucial.
Based on the Retinex theory [10], Zeng et al. decouple the essential features from pedestrian
images, independent of lighting conditions, and combine them with lighting-independent features
for feature matching [28].
While this approach can separate the lighting information from the pedestrian features that
eliminate the impact of lighting changes, it may also reduce the discriminative features of pedestrians. Zhang et al. [29] employ lighting transformation techniques to transform images taken under
different lighting conditions into images with the same lighting, and then use traditional pedestrian
recognition methods to extract more identifiable features. However, this approach assumes that
all images can perfectly express semantic features under normalized lighting conditions, which
may not be valid because different image contents have distinct identification values under various
lighting conditions. This assumption may lead to additional losses introduced by uniform lighting
conditions.
As shown in Figure 2, existing approaches to solving the task of person Re-ID in complex illumination conditions can be broadly classified into two categories: (1) methods based on illuminationinvariant features and (2) methods based on illumination unification. The former often leads to a
reduction in the distinctiveness of pedestrian features, while the latter introduces additional errors
due to the process of normalizing lighting conditions. To address these limitations, we conducted
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.

Optimal Illumination Distance Metrics for Person Re-Identification

26:3

Fig. 1. Given a pedestrian retrieval image with unknown illumination, from a retrieval gallery consisting of
images taken by cameras in non-overlapping areas under different illumination conditions, retrieve all images
containing the same pedestrian.

Fig. 2. Existing approaches for person Re-ID in complex lighting: (1) illumination invariance feature-based
methods and (2) illumination unification methods. Our method selects the optimal lighting condition for
person Re-ID.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.


codex
第二批读完。这里的可借鉴点是“不要追求统一到一个标准状态”，而是为每对样本选择最有判别力的条件，光照论文尤其明显。最后读部件去噪、姿态骨架融合和隐私扩散三篇。
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
/bin/zsh -lc "pdftotext -l 3 'Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

2137

Part-Based Feature Complementary Denoising for
Unsupervised Person Re-Identification
Qing Tian , Bin Wang, Xiang Liu , Jiashuo Shen, Keyang Cheng , Member, IEEE, Weihua Ou,
and Zhen Lei , Fellow, IEEE
Abstract—Unsupervised Person re-identification (ReID) aims
to automatically capture and match images of the same person across different camera viewpoints without any manual
annotations. Current methods primarily generate pseudo-labels
by clustering global features and employ contrastive learning
strategies for training. Despite the promising advancements made
by these approaches, effectively addressing the inherent bias of
global features and mitigating the impact of pseudo-label noise
remains an unresolved issue. To tackle this challenge, we propose
a part-based features complementary denoising method (PFCD).
Specifically, we design the combined features (CF) module and
the partial-features fusion and contrastive scheme (PFCS), which
capture fine-grained clues from a local perspective and combine
global and local features for clustering with consistent pseudolabel assignment, thereby achieving a complementarity between
global and local features. Furthermore, to diminish the influence
of pseudo-label noise on the model, we design the GMM features
denoising (GFD) module, which employs a Gaussian Mixture
Model to categorize features within each pseudo-class based
on confidence levels and performs denoising on low-confidence
features. Lastly, we construct a modular knowledge distillation (MKD) to enhance feature representation capabilities and
effectively reduce pseudo-label noise. In addition, our method
is confirmed to be effective through extensive experiments on
four challenging ReID datasets, remarkably surpassing numerous
state-of-the-art methods. Code has been made available at https://
github.com/xfltdzzz/PFCS ReID

Received 17 February 2025; revised 3 June 2025 and 12 July 2025;
accepted 8 September 2025. Date of publication 12 September 2025; date
of current version 5 February 2026. This work was supported in part by the
National Natural Science Foundation of China under Grant 62176128, in part
by the Basic Research Program of Jiangsu under Grant BK20231143, in part
by the Fundamental Research Funds for the Central Universities under Grant
NJ2023032, in part by the Project Funded by the Priority Academic Program
Development of Jiangsu Higher Education Institutions (PAPD) Fund, and in
part by the 333 High-Level Talent Project of Jiangsu Province. This article
was recommended by Associate Editor M. Devanne. (Corresponding author:
Qing Tian.)
Qing Tian is with the School of Software, Nanjing University of Information
Science and Technology, Nanjing 210044, China, also with Wuxi Institute
of Technology, Nanjing University of Information Science and Technology,
Wuxi 214000, China, and also with the MIIT Key Laboratory of Pattern
Analysis and Machine Intelligence, Nanjing University of Aeronautics and
Astronautics, Nanjing 211106, China (e-mail: tianqing@nuist.edu.cn).
Bin Wang, Xiang Liu, and Jiashuo Shen are with the School of
Software, Nanjing University of Information Science and Technology, Nanjing 210044, China (e-mail: wangbin@nuist.edu.cn; liuxiang@nuist.edu.cn;
shenjiashuo@nuist.edu.cn).
Keyang Cheng is with the School of Computer Science and Communication Engineering, Jiangsu University, Zhenjiang 212013, China (e-mail:
kycheng@ujs.edu.cn).
Weihua Ou is with the School of Big Data and Computer Science, Guizhou
Normal University, Guiyang 550025, China (e-mail: ouweihua@gznu.edu.cn).
Zhen Lei is with the National Laboratory of Pattern Recognition, Institute
of Automation, Chinese Academy of Sciences, Beijing 100190, China (e-mail:
zhen.lei@ia.ac.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3609570

Index Terms—Combined feature, knowledge distillation, contrastive scheme, feature denoising, unsupervised person reidentification.

I. I NTRODUCTION

P

ERSON re-identification (ReID) strives to recognize the
identical pedestrian across monitoring cameras [1]. Currently, ReID is frequently employed in practical scenarios
such as intelligent security and protection and pedestrian
analysis. Nevertheless, ReID faces significant challenges due
to variability in appearance caused by factors such as posture, clothing, occlusion, and changes in body shape [2],
[3], [4]. As a result, researchers have been paying growing
attention to tackling these challenges present in ReID in
recent times.
According to the modeling paradigm of current ReID works,
they can be implemented through both supervised [5], [6], [7]
and unsupervised learning [8], [9] methodologies. While the
former usually surpasses the latter in performance, annotating
data with labels is typically time-consuming and high-cost,
which limits the applicability and scalability of supervised
learning-based methods. To this end, growing attention is paid
to the unsupervised modeling paradigm.
Unsupervised learning-induced ReID can be further grouped
into unsupervised domain adaptation (UDA) ReID [10], [11]
and fully unsupervised learning (USL) ReID [12], [13],
[14]. For UDA ReID methods [9], [15], [16], [17], they
usually involve pre-training a model on a labeled source
domain and subsequently fine-tuning it on an unlabeled target
domain. In contrast, USL ReID methods [18], [19] do not
require supervised annotation knowledge from the domains
while learning directly from unlabeled datasets. Although
the USL ReID approaches enjoy better scalability to large
data applications, their performance seriously relies on the
quality of the predicted pseudo-labels, thus requiring more
sophisticated training strategies to improve the accuracy of
pseudo-labels.
The majority of USL ReID methods [20], [21], [22], [23]
have achieved significant advancements by leveraging clustering algorithms to generate pseudo-labels [24], which are
subsequently utilized to train the models. These approaches
presume that images of the same person are more similar,
making them more likely to be clustered together within the
same cluster. In addition, contrastive learning has shown great
promise in USL ReID tasks, in which the memory bank [25] is
commonly employed in contrastive learning to store previous
feature representations, which is particularly beneficial for

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.

2138

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

this domain. By providing a larger pool of negative samples
for comparison, the memory bank enhances the effectiveness
of contrastive learning and consequently alleviates feature
instability issues caused by small batch sizes.
However, despite the great progress of above USL ReID
methods, they still face a critical limitation of discriminating
ability when relying solely on global feature representations. It
is because that compared to local feature representations, the
global representations are inferior in characterizing the person
body details. We empirically analyzed this issue through
evaluations and found that nearly all of related works primarily
depend on global features to capture and thus recognize people
identities. When people of different identities have highly
similar salient features in global visual appearance, relying
solely on global features may tend to incur representationbias and thus result in the ReID classifier failing to distinguish
these identities. Therefore, local features, by providing additional fine-grained discriminative information, can effectively
compensate for the deficiencies of global features. Although a
small number of current works [26], [27] also focus on local
features, they treat these feature representations independently,
for example, by assigning pseudo-labels to these local features
separately or clustering them individually. In this manner, the
complementarities between global and local feature representations are overlooked and, unfortunately, remain unexplored.
To tackle the issues mentioned above, we propose a novel
USL Re-ID framework consisting of three synergistic modules: PFCS, GFD, and MKD, along with the CF module.
The CF module integrates local and global features to capture
both dominant semantic appearances and fine-grained body
details, providing more robust and discriminative combined
features that indirectly enhance the denoising capability of
the GFD module. Building on this, the PFCS module innovates the clustering process by computing separate distance
matrices for global and local features and fusing them by
weights, effectively mitigating the feature bias inherent in
traditional global-only clustering methods. To further address
the pseudo-label noise common in USL Re-ID, we introduce
two complementary denoising mechanisms. The GFD module
applies a Gaussian Mixture Model on the combined features
to estimate confidence levels and suppress noisy samples
within the same pseudo-class. Meanwhile, the MKD module
strengthens early-stage robustness by employing an external
teacher network to guide the student network through knowledge distillation, thereby reducing noise impact and enhancing
representation learning.
It is worth pointing out that although our method is built
upon the contrastive learning framework proposed in CCL
[20], it fundamentally differs from global-only iterative training methods such as RTMem [28] and CCL itself. Specifically,
we specially design fine-grained local features and assign
them to independent memory banks, allowing these features to
actively participate in both the clustering and contrastive learning processes. This joint participation continually enhances
the discriminative capability of model. Moreover, while
PPLR [26] also utilizes local features, its underlying design
philosophy is substantially different. Unlike our approach
which incorporates the designed local features directly into

clustering, PPLR focuses on improving pseudo-label quality
through intra-image consistency analysis. Notably, PPLR does
not adopt a contrastive learning framework.
Compared with recent denoising-oriented methods such as
DCCT [29] and DKD-MPL [30], our approach also exhibits
distinct differences in both motivation and model-design.
DKD-MPL introduces a tri-view pseudo-labeling scheme and
employs dual knowledge distillation modules to perform selfrefinement via consistency filtering and soft-label supervision.
DCCT, on the other hand, adopts a dual-network co-teaching
framework where pseudo-labels are generated from two separate clustering processes, and training is performed only on
consistent, high-confidence samples shared by both networks.
In contrast, our method introduces two synergistic modules,
GFD and MKD, to jointly mitigate the impact of noisy pseudo
labels. Unlike DKD-MPL and DCCT, our framework neither
relies on generating additional pseudo labels nor requires
multiple clustering processes. Specifically, the GFD module
performs explicit denoising at the feature level by modeling
the distribution of intra-class samples with a Gaussian Mixture
Model. In parallel, the MKD module adopts an independently
designed teacher network that provides consistent and structured guidance throughout the entire training process, thereby
enhancing the stability and robustness of representation
learning.
These design choices collectively improve pseudo-label
quality and enable a more robust training pipeline under fully
unsupervised settings. Based on the above motivations and
designs, our main contributions can be summarized as follows:
• We design a novel feature fusion method. First, during
the feature extraction stage, after the global features are
extracted using a ResNet-based backbone network, the
features are divided and fused using adaptive weights.
In the clustering stage, we fuse the Jaccard distance
matrices of both global and local features with different
weights. This method not only effectively captures the
most prominent clues in the global features but also
leverages the local features to complement the missing
fine-grained information in the global features, achieving
a complementary fusion of global and local features.
• An innovative method for pseudo-labels denoising is
presented, employing Gaussian Mixture Model (GMM)
to identify high-confidence and low-confidence features.
This method leverages Gaussian distribution assumption to denoise low-confidence features within the same
pseudo-class. Additionally, knowledge from the teacher
model is harnessed to further reduce the noise in pseudolabels, ensuring the optimization of the denoising effect.
• Comprehensive experiments conducted on the Market1501, DukeMTMC-ReID, MSMT17, and PersonX
datasets have shown that our method surpasses existing
state-of-the-art USL ReID methods.
II. R ELATED W ORK
A. Unsupervised Person ReID
The current unsupervised person ReID methods can be
broadly classified into two types: Unsupervised Domain

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.

TIAN et al.: PART-BASED FEATURE COMPLEMENTARY DENOISING FOR UNSUPERVISED PERSON RE-IDENTIFICATION

Adaptation (UDA) ReID [31], [32], [33] and Fully Unsupervised Learning (USL) ReID [18], [34], [35], [36]. UDA ReID
[37], [38] transfers knowledge from the source domain to the
target domain, assuming that the discrepancies between the
two domains are minimal. In contrast, USL ReID is trained
directly on the target dataset, making it more scalable than
UDA ReID. Thus, we focus on USL ReID, as it aligns more
closely with our research objectives.
To fully leverage unlabeled datasets in USL ReID, existing clustering algorithms are usually used to generate
pseudo-labels, such as K-means [39] and DBSCAN [40]. Subsequently, the pseudo-labels generated are used in conjunction
with the unlabeled dataset for iterative training. Therefore, how
to boost the precision of pseudo-labels is the central issue
in USL ReID. Early methods introduced various strategies
to address this challenge. BUC [12] introduced a bottom-up
clustering method that utilized the inherent diversity between
identities and the similarity within each identity to obtain
more discriminative features. PLM [13] presented a progressive learning method based on multi-scale fusion networks,
enabling inference on large datasets without annotations.
MMT [8] learned feature representations through the joint
supervision of offline refined hard pseudo-labels and online
refined soft pseudo-labels, which helped hinder the gradual
accumulation of label noise. ISE [41] proposed an implicit
sample extension method that enhanced the effectiveness of
the clustering module by generating support samples.
Building upon these earlier works, the state-of-the-art USL
ReID methods [9], [30], [42] primarily utilized memorybased contrastive learning frameworks. Specifically, SPCL [9]
introduced a self-paced approach, progressively utilizing more
dependable clustering to train the hybrid memory and learning objectives. CCL [20] introduced a brand-new clustering
contrastive learning approach that was built on a cluster-level
clustering memory bank to solve the problem of inconsistency.
Furthermore, various modules for selection or improvement
have been proposed to boost the accuracy of USL ReID.
For instance, DHCCN [43] improved the performance of
USL ReID by introducing local clues and cross-granularity
consistency constraints. Wu et al. [44] constructed patch proxy
classes as initial supervision, they proposed the method for
assigning pseudo-labels to samples using pairwise gradientguided similarity separation.
In this work, to enhance representation learning and improve
model performance, we focus on generating more precise
pseudo-labels. Specifically, we combine global and local features, and leverage a Gaussian Mixture Model along with
teacher knowledge to reduce pseudo-label noise. The following
parts will discuss these techniques and related works in detail.
B. Part-Based Approaches for Person ReID
Most deep learning-based ReID methods [9], [20], [29],
[36], primarily rely on global features of individuals, which
may lead to the neglect of important information contained
in local features. In recent years, some studies have focused
on leveraging discriminative local features, aiming to improve
retrieval accuracy by incorporating these local parts. Currently,
the most widely used methods for part-based ReID involve

2139

directly using predefined stripes. This approach assumes that
person images are vertically aligned, does not require additional pre-trained models, and offers good scalability.
For example, MGN [45], SSG [27], and PPLR [26] have
all utilized local features to further refine the ReID model.
Different from USL ReID works, MGN, which was designed
for supervised ReID, could easily learn more discriminative
feature representations given the presence of labels. Similar
to our approach, SSG and PPLR were both employed for
USL ReID. SSG [27] generated separate pseudo-labels for
each group of global and local features without employing
any fusion strategy, potentially overlooking the consistency
between the different branches. PPLR [26] proposed a cross
agreement score to obtain reliable complementary knowledge
in both global and local features, but exclusively used global
features for clustering, neglecting the impact of local clues
during the clustering process.
In contrast to these methods, our approach enhances global
features by complementing them with local features through
the CF and PFCS modules. We further integrate global and
local features for clustering, ensuring the same pseudo-labels
are assigned to both and utilizing their respective memory
banks to enrich their feature representations. This synergistic
fusion of global and local features allows the model to
better capture fine-grained details while maintaining overall
consistency, thus enhancing its discriminative power.
C. Person ReID With Noisy Labels
Over the past few years, an increasing number of studies
have focused on minimizing the impact of noisy samples
to enhance the reliability of pseudo-labels. Existing methods
primarily aim to filter out the interference from low-confidence
samples to obtain more accurate features, based on the
assumption that low-confidence samples typically incur greater
losses than high-confidence ones. For instance, Arazo et al.
[46] fitted the Beta mixture model to distinguish between highconfidence and low-confidence samples, while DivideMix [47]
used the Gaussian mixture model instead of the Beta mixture
model, offering a more flexible distribution. Additionally,
strategies employing knowledge distillation to reduce label
noise have also been widely studied. MMT [8] proposed a
collaborative training strategy that used the temporal averaging
model of each network to generate soft pseudo-labels for
supervising other networks, aiming to prevent the amplification
of training errors. HDCPD [48] aimed to match the statistical
distributions between the network and the teacher model,
updating parameters via the Exponential Moving Average
(EMA) method. While these methods have made significant
strides in mitigating label noise, substantial label noise may
still remain during the initial training phase, as most models
are initialized with parameters pre-trained on ImageNet. Moreover, NRNet [49] proposed a dual-network framework that
uses networks of different depths (ResNet-50 and ResNet-152)
to extract complementary features and decouples clustering
from training to reduce noise accumulation. While effective for
unsupervised domain adaptation, it relies on a labeled source
domain and is therefore not applicable to fully unsupervised
person re-identification.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8598

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025

Pose-Skeleton Guided Cross-Attention
Representation Fusion for Occluded Pedestrian
Re-Identification
Shuze Geng , Yifan Liu, Zijin Wang, Gang Yan , Yang Yu , and Yingchun Guo , Member, IEEE
Abstract—Most methods address occluded pedestrian ReIdentification (Re-ID) by employing external auxiliary models
in the feature output stage of the backbone network to locate
visible appearance areas. Nevertheless, these approaches suffer
from issues such as occlusion information diffusion and imprecise
masks generated by external models, indicating the need for
further exploration in the decoupling of pedestrian features
from occlusion information. In light of these challenges, we
propose an innovative algorithm called Pose-Skeleton guided
Cross-attention Representation fusion (PSCR) method. Firstly,
we introduce the Visible Appearance Region Attention (VARA)
model designed to leverage pose information for guiding the
backbone network in effectively distinguishing between occlusion
information and pedestrian features at the intermediate layer. By
employing a suppression strategy, the model is able to effectively
suppress occlusion interference and alleviate the diffusion of
occlusion information. Next, to achieve precise localization of
pedestrian-specific semantic regions, a groundbreaking Skeletal
Area Modeling (SAM) is proposed. Leveraging the principles of
mathematical modeling and capitalizing on the efficacy of human
keypoint confidence, this module generates finely-grained masks
for local skeleton regions and extracts an exhaustive set of local
features. Lastly, under the constraints imposed by spatial attention masks, a cross-attention mechanism is employed to fuse the
features acquired from the previous two steps with local features.
This fusion process results in the generation of enhanced local
features that seamlessly integrate aligning high-level semantic
information. Extensive experimentation demonstrates that the
proposed algorithm exhibits notable performance advancements
when compared to existing methodologies.
Index Terms—Occluded, re-identification, pose-skeleton, crossattention model.

P

I. I NTRODUCTION
ERSON re-identification (re-id) aims to identify individuals across different camera views [1]. However,

Received 3 March 2025; accepted 27 March 2025. Date of publication
31 March 2025; date of current version 8 September 2025. This work was
supported in part by the National Natural Science Foundation of China under
Grant 61806071 and Grant 62102129; in part by Tianjin Municipal Education
Commission Research Plan Project under Grant 2022KJ110; in part by the
Sci-Tech Research Projects of Colleges and Universities in Hebei Province,
China, under Grant QN2019207; in part by the Natural Science Foundation
of Hebei Province (General Program), China, under Grant F2019202381
and Grant 2019202464; and in part by the Key Project of Tianjin University of Technology and Education under Grant KRKC012106. This article
was recommended by Associate Editor Z. Ding. (Corresponding author:
Shuze Geng.)
Shuze Geng and Yifan Liu are with the School of Information Technology
and Engineering, Tianjin University of Technology and Education, Tianjin
300350, China (e-mail: gengshuze@tute.edu.cn).
Zijin Wang, Gang Yan, Yang Yu, and Yingchun Guo are with the School
of Artificial Intelligence, Hebei University of Technology, Tianjin 300401,
China.
Digital Object Identifier 10.1109/TCSVT.2025.3556250

challenges such as lighting changes, pedestrian occlusion,
and background variations make cross-scene matching highly
difficult [2], [3]. Among these challenges, occlusion stands out
as a key issue with a significant impact [4], [5], [6], [7].
The central problem in pedestrian re-identification under
occlusion is to decouple pedestrian appearance information
from occlusion objects in images and create an effective representation for matching. Current approaches often incorporate
advanced pedestrian pose estimation models or part parsing
models to assist in extracting features from non-occluded
regions of pedestrians. Typical methods include PGFA [8],
PGPAN [9], PVPM [10] and PFD [11], and so on. While these
methods do employ semantic and spatial position information
from pedestrian pose keypoints to achieve pedestrian region
localization and enhance re-identification performance to some
degree [12], they still exhibit two limitations. One major
limitation is that these methods only perform visible feature
extraction at the output of the backbone network, thus failing
to effectively separate occlusion information. As the network
deepens, the receptive field of spatial units in the feature map
expands, causing non-occluded units to cover occluded regions
and transmit occlusion information to originally un-occluded
areas. Another drawback is the adoption of pedestrian feature
extraction techniques, such as mask filtering or semantic alignment, on the feature map generated by the backbone network.
This approach restricts the backbone network’s ability to
perceive and make decisions based on occlusion information.
It selectively updates a subset of the network’s neurons,
impairing the learning capacity of neurons responsible for
occlusion and background information. These issues make
it challenging to decouple pedestrian features and occlusion
information. Additionally, the pedestrian masks generated by
external models are not detailed enough, resulting in some
information loss and occlusion interference. From Fig. 1, it
can be observed that the utilization of conventional approaches
based on pose or part parsing methods fails to effectively
decouple pedestrian regions from occluded regions.
To address the aforementioned issues, this paper proposes
a new Pose-Skeleton guided Cross-attention Representation
fusion (PSCR) algorithm,as shown in Fig. 2. Building upon
the feature fusion paradigm, our proposed method synergistically integrates pose information with the backbone
network and a dedicated local feature extraction network.
This novel integration empowers the network with enhanced
perceptual capabilities and bolstered decision-making prowess
specifically tailored for the challenge of decoupling pedestrian

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.

GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION

8599

work features and local features. This approach not only
enhances discriminative local features but also aligns
high-level semantic features.
The remainder of this paper is structured as follows: Section II
provides a comprehensive review of related works in the field.
In Section III, we present the architecture of the proposed
framework and elaborate on the implementation details. Experimental results and a thorough analysis of the effectiveness of
our method are presented in Section IV. Finally, Section V
concludes our work by summarizing the key findings and
contributions.
II. R ELATED W ORK
Fig. 1. Illustration of the Original Image (a) and pedestrian attention regions
obtained using traditional pose or parsing-based methods (b), as well as our
method (c).

features and occlusion information. Specifically, an innovative
Visible Appearance Region Attention (VARA) module is constructed, employing a multi-layer pose information soft fusion
technique to facilitate the intermediate layer of the backbone
network in effectively learning occlusion information. This
approach attempts to resolve the issue of occlusion information
diffusion, employing a suppression method instead of relying
solely on pose estimation to filter high-level features [8],
[12], [13], [14]. Furthermore, a pioneering Skeletal Area
Modeling (SAM) technique is developed, leveraging human
keypoint information to generate pre-defined joint masks in
a mathematically modeled manner. By comparison, these
masks outperform those generated by semantic segmentation
[15], [16], [17] or pose points since they provide pixellevel visibility confidence and exhibit greater stability when
confronted with low-resolution pedestrian images. Lastly, a
Skeleton Modulated Cross-Attention Fusion (SMCF) module
is established to explore the interplay between local features
derived from the pedestrian skeleton and features outputted by
the backbone network. Through the fusion of local contextual
information, this module effectively learns pedestrian retrieval
features, enabling enhanced retrieval performance.
The main contributions of our work can be summarized as
follows:
• Firstly, VARA module is proposed that is organically
integrated with the backbone network, which guides
the backbone network to perceive pedestrian appearance
information through the foreground-perceiving channel
attention mechanism (pedestrian area) and the crossattention fusion mechanism of key feature spaces, and
improves the model’s perception and discrimination abilities for occlusion information.
• Secondly, the SAM technique is developed to construct
a comprehensive set of local features by capturing local
skeleton regions of the human body. This compensates for
potential information loss in appearance regions between
key points extracted from the heat map generated by the
pose estimation model.
• Thirdly, the SMCF module utilizes the local region
weight map provided by the joint mask to constrain
the cross-attention fusion process between backbone net-

Currently, existing occluded pedestrian re-identification
methods are primarily categorized into two research
paradigms: methods utilizing external pedestrian models (e.g.,
pose estimation and semantic segmentation) [5], [15], [16],
[17], [18], [19], and approaches leveraging Vision Transformer
(ViT) architectures [20], [21].
A. Pedestrian Re-Identification Method Based on External
Semantic Clues
Currently, methods that employ external semantic clues have
gained prominence in the field of re-ID. These approaches
often utilize external pedestrian models, such as pedestrian
parsing, pose estimation, or foreground segmentation (pedestrian area), to locate the visible area of pedestrians [22],
[23], [24], [25], [26], [27]. For instance, in [8], a posture
information branch is added to the PCB network. The heat
maps of key points are treated as masks for the pedestrian
key areas, which are then applied to the output feature map
of the re-identification network to extract the pedestrian’s
appearance features. Similarly, to emphasize the local regions
of pedestrians, study [11] introduces a Pose-guided Feature
Disentangling (PFD) method based on transformers. It leverages pose information to disentangle semantic components
and selectively match non-occluded parts. To obtain complete
pedestrian features under occlusion, Hou et al. [24] simultaneously introduce a foreground extractor and a pose estimation
model. The pose estimation result guides the structural segmentation of the feature map, while the foreground extractor
filters out occlusion information. Spatial correlation between
local features is then used to reconstruct the appearance
features lost in the occluded areas. With a similar motivation,
Human Co-Parsing Guided Alignment (HCGA) entails the
amalgamation of a co-parsing network and a guided alignment
module, leading to the extraction of crucial foreground and
human part features for occluded person ReID [25].
The aforementioned approaches have partially addressed the
issue of occlusion on re-identification models. However, these
methods typically incorporate modules post feature extraction
by the backbone network (e.g., CNN or Transformer). This
approach presents a challenge as occlusion information may
spread to previously un-occluded areas due to the expanding
pixel receptive field in the backbone network. Consequently,
the higher or output layers struggle to filter out the diffused
interference information. In contrast, our proposed method

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.

8600

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025

Fig. 2. The overall architecture of the proposed PSCR network. Our method contains VARA module(Section III-B), SAM module (Section III-C), SMCF
module (Section III-D). Module VARA: Focusing on visible light area and suppressing occlusion interference in the backbone; Module SAM: Achieving more
precise localization and extracting local features of pedestrians; Module SMCF: Incorporating Cross-Attention Mechanism to dynamically capture semantic
∗ and the global feature f obtained from
correlations between features for local feature alignment and fusion. During matching stage, the local features fout
g
the backbone are concatenated together for occluded re-id.

introduces the VARA module and leverages multi-layer pose
information soft fusion to guide the backbone network in
learning occlusion information at intermediate layers. Moreover, we introduce a novel SAM module to address the loss
of pedestrian appearance information in the pose heatmap.
This effectively mitigates the decoupling problem between
pedestrian information and occlusion information within our
method, elevating its performance.
B. Methods Based on Transformer
The initial introduction of Transformer in the Re-ID field
was proposed by [20] as TransReID. However, TransReID
primarily focuses on global feature learning, neglecting the
correlation between local features. Thus, in literature [21], a
spatial slice strategy is proposed to incorporate the spatiality
of input images into the image patch sequence, enhancing the
short-distance correlation among patches. Similarly, Lu [28]
presents a dual-branch Transformer network that operates in an
end-to-end manner. The first branch is a global branch based
on transformers, responsible for extracting global features.
The second branch, known as the local branch, utilizes the
Selective Token Attention (STA) module, employing multiheaded self-attention to efficiently extract unique tokens for
obtaining local features.
Moreover, an auto-aligned Transformer model is proposed in [29] which avoids the use of a fixed partition to
acquire local features. Instead, it utilizes clustering for patch
allocation and introduces part encoding to simultaneously
adaptively learn and align local features. However, despite the
discriminative power of Transformers, they lack the desired

properties of translation, scale, distortion invariance, as well
as the hierarchical structures found in CNN convolution
operations. Addressing this limitation, [30] combines a CNN
backbone to extract hierarchical features and harnesses the
global interaction characteristics of Transformers to aggregate features of different scales from a global perspective.
Another notable approach, PAT [31], employs CNNs as a
backbone model and incorporates a transformer encoder to
extract additional features. It utilizes cross-attention with a
group of learnable part prototypes to selectively mask specific
features. Similarly, our proposed method also leverages the
fusion of CNN and transformer encoder. However, what sets
our method apart is that our method is the incorporation of
local skeletal regions of the human body, which allows for
the construction of a comprehensive set of local features.
Furthermore, our method utilizes the local region weight map
derived from the joint mask to regulate the cross-attention
fusion process between the backbone network’s features and
the local features. As a result, we achieve enhanced discriminative local features and alignment of high-level semantic
features.

III. M ETHODOLOGY
In this section, we present the implementation details of our
PSCR method for occluded person re-id, as shown in Fig. 2.
An overview of the proposed method is introduced in Section
III-A. Each component of our method is then described in
detail in Sections III-B to Section III-D.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Privacy preserving person re-identification via anonymizing diffusion model.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 174 (2026) 112939

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Privacy preserving person re-identiﬁcation via anonymizing diﬀusion
model
Rui Wang
a

a,1 , Ruikang Lin

b,1 , Jiabing Yang

a , Hao Ni

b , Jingkuan Song

c,∗

Yingcai Honor College, University of Electronic Science and Technology of China, Chengdu, 611731, Sichuan, China

b School of Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu, 611731, Sichuan, China
c

Shanghai Innovation Institute, Shanghai, 200231, Shanghai, China

a r t i c l e
Keywords:
Person re-identiﬁcation
Privacy protection
Diﬀusion

i n f o

a b s t r a c t
Existing public person re-identiﬁcation (ReID) datasets face signiﬁcant privacy challenges that raise critical
ethical and legal concerns, particularly regarding the collection, and utilization of pedestrian imagery. Those
datasets contain sensitive biometric information of real pedestrians, and privacy attributes(e.g. identity, age,
location) may be deduced by attackers. To address this issue, numerous anonymization methods have been
proposed to reconstruct pedestrian images. However, the privacy protection capability of these methods is limited due to incomplete removal of private information. In this paper, we propose an anonymizing diﬀusion
model (ADM), a novel generation-based privacy protection approach for ReID. Through 2-stage ﬁne-tuning
of diﬀusion model and a novel prompt construction algorithm, ADM is able to maintain the usability of the
generated data while preventing the real ID from being retrieved by either neural networks or human observers. Building on ADM, we release three privacy-compliant ReID benchmarks-Market1501-A, MSMT17-A,
and CUHK-SYSU-A-which systematically replace original pedestrian identities with synthetic yet discriminative counterparts. These datasets retain re-ID utility (e.g., posture, clothing styles) while eliminating biometric
identiﬁers, oﬀering the ﬁrst standardized anonymized alternatives to their non-compliant originals. Furthermore, we introduce a new metric called ID separation degree (ISD) to measure the visual diﬀerence between
real and generated samples. To the best of our knowledge, this is the ﬁrst exploration of utilizing diﬀusion
models to generate new IDs for privacy protection ReID. We release an open-source implementation ofADM at
https://github.com/Physics-wr/Anonymizing-Diﬀusion-Model.

1. Introduction
Person re-identiﬁcation (ReID) is to detect and track the same target
person across diﬀerent camera views based on image features. However,
the collection of training data for ReID tasks faces severe privacy leakage
issues. Those data often contain high-resolution images captured in public spaces without explicit informed consent from individuals. A primary
issue lies in the irreversible exposure of sensitive biometric information,
including facial features, gait patterns, and clothing details, which can
be exploited for unauthorized identity tracking or linkage to auxiliary
personal data. Furthermore, the open accessibility of these datasets exacerbates potential misuse, as malicious actors could repurpose them
to develop surveillance tools or infer private attributes (e.g., health status, location history) through attacks. For example, as shown in Fig. 1,
as long as a photo of these pedestrians is obtained from social media,

attackers can retrieve other images of pedestrians from public datasets
for more privacy information, such as their whereabouts. Mainstream
ReID datasets such as Market-1501 [1], DukeMTMC-reID [2], CUHKSYSU [3] and MSMT17 [4] are mostly collected from public scenarios
such as malls and campuses, but whether the collection of these datasets
has fully considered the participants’ right to information and portrait
remains in doubt.
Although these datasets have made important contributions to the
advancement of ReID research, insuﬃcient privacy protection hinders
the adoption of the technology in real application environments. To address this issue, some existing works on privacy protection for ReID
attempt to reconstruct datasets using image encryption or adversarial attacks. Image encryption aims to selectively blur sensitive regions [5], add noise to the image [5], or encrypt [6] before releasing
datasets. The images obtained by encryption-based methods require a

∗

Corresponding author.
E-mail addresses: wangrui604@foxmail.com (R. Wang), linrk.proton@gmail.com (R. Lin), jiabingyang01@gmail.com (J. Yang), haoni0812@gmail.com (H. Ni),
jingkuansong@tongji.edu.cn (J. Song).
1
Contributed equally to this work.
https://doi.org/10.1016/j.patcog.2025.112939
Received 26 May 2025; Received in revised form 4 November 2025; Accepted 14 December 2025
Available online 17 December 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 174 (2026) 112939

R. Wang et al.

Fig. 1. Most benchmark datasets aggregate high-resolution pedestrian images captured across multiple non-overlapping surveillance cameras, often with detailed
metadata such as camera IDs, sequence numbers, and timestamps. This granular spatiotemporal information enables adversaries to reconstruct individual movement
patterns by cross-referencing camera deployment locations and timestamps, thereby exposing sensitive trajectories and daily routines of real-world pedestrians.

complex decryption process before being utilized by ReID models. Adversarial attack approaches [7–9] add carefully crafted perturbations
to samples to reduce their recognizability during retrieval. However,
to ensure data usability, most of these methods do not change the
IDs of pedestrians. For example, adversarial attack methods only add
slight perturbations to the images, allowing the true IDs to still be retrieved by humans or other ReID models. Essentially, these anonymization methods hide private information rather than eliminate it. Therefore, these methods still have privacy leakage risks. Despite those efforts to construct privacy-preserving datasets, the inherent trade-oﬀ
between data utility and anonymization eﬃcacy remains unresolved,
often leading to suboptimal model performance or residual identity
leakage.
To eliminate the privacy information of the original data while preserving data utility, we propose an anonymizing diﬀusion model (ADM),
a novel generation-based privacy protection approach for ReID. ADM
aims to anonymize ReID datasets by generating visually distinct new
IDs, preventing being retrieved by either neural networks or human observers. To this end, ADM utilizes Stable Diﬀusion (SD) [10] to generate anonymous images with new IDs through text prompts. As shown
in Fig. 2, compared with previous methods that edit real images, the
generated images originate from noise, so private information is eliminated at its source without relying on potentially reversible alterations.
To enable diﬀusion model to produce synthetic ReID dataset, we ﬁrst
ﬁne-tune it on the entire dataset. To achieve ﬁne-grained control over
pedestrian visual attributes, a second stage ﬁne-tuning on subsets is
conducted to facilitate intra-id consistency.A new anonymous dataset
is generated based on novel text prompts constructed through transformation of original image captions. Finally, some outlier may exist among speciﬁc generated ID, which hinders data utility. ADM ﬁlters outlier samples from the generated dataset through the proposed
griddle.
Additionally, to quantitatively measure the visual diﬀerence between original and generated samples, we introduce a new metric called
ID separation degree (ISD). ISD measures the anonymization capability of a privacy protection method based on the feature distance in the
embedding space. A higher ISD indicates greater dissimilarity between
the real and generated samples, and thus enhanced privacy protection
level.
To analyze the eﬀectiveness of our proposed method, we conduct
thorough experiments on commonly used ReID datasets and compare
them with other privacy-preserving methods. The results show that ADM

achieves both competitive utility capability and privacy-preserving capability.
Experiments on public datasets demonstrate that ADM achieves
excellent privacy protection performance while maintaining competitive ReID accuracy. Based on ADM, we obtain three privacy-compliant
derivatives of widely adopted benchmarks Market1501, MSMT17 and
CUHK-SYSU, enabling researchers to develop robust re-ID models without compromising individual privacy. The contributions of our work can
be concluded as:
•

We introduce the ﬁrst set of anonymized ReID datasets (Market1501A, MSMT17-A, CUHK-SYSU-A) that eliminate personally identiﬁable
information while retaining ReID eﬃcacy, providing ethically compliant alternatives to mainstream benchmarks.
• Through 2-stage ﬁne-tuning and novel prompt construction algorithm, ADM is able to synthetize ReID dataset with a trade-oﬀ between data utility and privacy protection capability. We also propose ISD as a new metric to evaluate privacy protection eﬃcacy of
the entire synthetic dataset.
• We conduct experiments on benchmark datasets to demonstrate
ADM can eﬀectively remove identity associations between real and
generated data, with superior performance in preventing retrieving
private identities from public datasets, while maintaining competitive ReID accuracy.
2. Related work
2.1. Person re-identiﬁcation
Person re-identiﬁcation (ReID) is an important person retrieval
task, which aims to retrieve a person of interest across multiple nonoverlapping camera views. It has wide applications in video surveillance, intelligent security, and other ﬁelds. In recent years, many deep
learning-based methods have been proposed to solve this problem. Traditionally, CNNs have served as the dominant backbone [11–14] for extracting discriminative features from pedestrian images. Beyond CNNbased approaches, Transformer architectures [15–17] have emerged as
a powerful alternative in recent years. Metrics-based learning methods are commonly used to calculate the similarity between person images [18–20]. Chen et al.[21] apply GAN to ReID to solve the domain adaptation problems. However, person ReID still faces severe privacy issues. For example, the widely used dataset DukeMTMC-reID
2

Pattern Recognition 174 (2026) 112939

R. Wang et al.

Fig. 2. The previous methods protect ID privacy through attacking, encryption, etc. These methods fail to erase the correlation with the original ID, leaving private
information that can still be retrieved through human observation or decryption. ADM (ours) protects ID privacy by generating new IDs to replace the original ones.

2.3. CLIP and stable diﬀusion

[2] was withdrawn due to privacy issues. Most existing ReID models rarely consider privacy protection problems, and some methods
proposed have diﬃculty in balancing privacy protection and model
performance.

CLIP [34], as a pre-trained model based on a transformer architecture, has been widely applied to various multimodal tasks [35–38].
Through contrastive learning on a huge dataset, CLIP builds semantic
connections between text and images to achieve mutual understanding between images and text. ClipCap [39] uses CLIP encodings as image caption preﬁxes, and ﬁne-tunes a language model GPT2 [40] to
generate image captions through a simple mapping network. Shao et
al.[41] utilize a divide-conquer-combine strategy to generate text descriptions of images through the CLIP paradigm, constructing a largescale text-labeled person dataset “LUPerson-T”. encoder of CLIP to generate prompts’ latent embeddings as conditions to guide image generation. Diﬀusion models [42] are generative models that include a process
of forward noise injection diﬀusion and a reverse process of denoising
reconstruction. An improvement of Stable Diﬀusion [10] over Diﬀusion
models is that it performs diﬀusion in the latent space of images, achieving higher computational eﬃciency. In addition, Stable Diﬀusion introduces conditional control. It uses the text encoder of CLIP to generate
prompts’ latent embeddings as conditions to guide image generation.

2.2. Privacy-preserving methods
Traditional privacy-preserving methods focus on blurring, pixelating, or adding noise to images, which may aﬀect the ReID performance
to some extent. Dietlmeier et al.[5] show that blurring faces has little impact on the performance of ReID systems, and propose an anonymous ReID dataset DAA [5] with blurred faces. However, the face only
occupies a small part of the pedestrian image, and privacy attackers
can still retrieve the target person through other major features such
as clothing, posture, background, etc. Zhang et al.[22] propose a reversible anonymous framework based on joint learning. They adapt desensitized images generated by conventional methods as the initial supervision to generate anonymous images for privacy protection, and the
original images can be recovered for ReID research. PIS [23] uses images from other identities in the dataset to weakly encrypt the original
image, generating anonymized images with relatively consistent identities. Diﬀerential privacy [24,25] and secure computation [26,27] have
also been used for privacy protection, yet they introduce relatively high
computational overhead [28]. Some synthetic virtual datasets [29–31]
have better privacy protection eﬀects while the large domain gap results in poor generalizability. Recently, event cameras have also been
used for privacy-preserving in ReID [32,33], but their application is not
widespread.

3. Methodology
The goal of ADM is to generate new IDs that are anonymized and
ReID usable. In the following, we describe the main components of our
method: Image-text Pair generation (3.1), Anonymizing Diﬀusion Model
(3.2), and ID separation degree (3.3).
3


codex
已按每篇前三页读完，下面只基于摘要和引言判断。

**逐篇提取**

1. `MDDM`
创新类型是新机制。它把 VI-ReID 的问题拆成低高层语义错位和特征分布错位，用欧氏空间和 Hilbert 空间双空间匹配加互蒸馏来讲。能发的点在于它没有只做单一 embedding 对齐，而是把“单空间对齐不够”包装成 gap。story 是：跨模态不应只在一个潜空间里拉近，而要在互补空间中找一致身份线索。

2. `NNSC-FR`
创新类型是新机制加工程组合。它抓住无监督 ReID 中聚类边界 hard sample 的伪标签不稳定问题，用最近邻样本约束和类间平衡损失处理边界，再用 ODE 引导的残差重构增强特征。能发靠的是伪标签边界样本这个明确痛点。story 是：无监督失败不是所有样本都错，而是边界样本反复污染聚类。

3. `Occluded person Re-ID with noise injection`
创新类型是问题重定义和新数据利用。它把车辆图像从“噪声”改写成“有益结构噪声”，用车辆数据模拟真实遮挡并做正则。能发的 gap 是 occluded ReID 数据小，真实车辆遮挡常见但被丢弃。story 很强：噪声不是要清除，而是要可控注入，让模型学会在遮挡下识别人。

4. `OCCNet`
创新类型是新数据设定加新机制。它定义 occluded VI-ReID，指出遮挡和跨模态会叠加造成空间错位和模态退化，并构造 Occluded-SYSU、Occluded-RegDB。能发靠新设定和补全机制双支撑。story 是：遮挡 ReID 不应只看可见区域，而要利用单模态上下文和跨模态互补信息补全缺失身份线索。

5. `Optimal Illumination Distance`
创新类型是问题重定义。它反对“光照不变”和“统一光照”这两条老路，提出每对图像有自己的最优光照距离。能发的 gap 是统一归一化会损失对某些身份有判别力的内容。story 是：不要把所有图像推到同一个标准光照，而要为每对样本选择最有判别力的光照条件。

6. `OPMCN`
创新类型是新机制。它聚焦无监督代理对比学习里的正负样本选择错误，尤其是假负样本和相机代理选择。能发靠的是把 camera proxy 从普通分组变成“最优代理挖掘”。story 是：无监督 ReID 的关键不是更多对比，而是选对最有信息量的代理对。

7. `PFCD`
创新类型是新机制偏工程组合。它用全局和局部特征互补生成伪标签，再用 GMM 和知识蒸馏去伪标签噪声。能发的 gap 是全局特征聚类有偏，局部信息过去没有真正参与聚类。story 是：伪标签质量来自全局语义和局部细节共同投票，而不是单一全局特征。

8. `PSCR`
创新类型是新机制。它批评已有姿态方法只在 backbone 输出端做 mask，导致遮挡信息已经扩散到高层特征里。它把姿态和骨架区域提前引入中间层，并用骨架区域建模和 cross-attention 融合。story 是：姿态不该只是后处理过滤器，而应该在特征形成过程中阻断遮挡扩散。

9. `ADM`
创新类型是问题重定义加新数据设定。它把 ReID 隐私问题从“模糊、加扰”改成“生成新身份替换真实身份”，并提出匿名化数据集和 ISD 指标。能发靠的是伦理合规和可用性之间的明确矛盾。story 是：隐私保护不是隐藏真实 ID，而是从源头替换成可训练但不可追踪的新 ID。

**这批论文反映出的可发套路**

最值得借的是三类：第一，把传统噪声重新定义成有益监督，例如车辆噪声、伪标签噪声、边界样本。第二，把“统一不变表示”改成“样本对自适应的最优条件”，光照论文给了很好的叙事模板。第三，把辅助信息从输出端后处理前移到特征形成过程，PSCR 这个角度比普通姿态 mask 更能讲机制创新。

**面向团队资产的强创新候选**

1. Headline：从跨视角 ReID 改写为“跨视角身体表面可见性补全”问题。  
挂载资产：aerial-ground 跨视角 ReID、SMPL 3D 几何、SOLIDER-Swin。  
区别：最像 OCCNet 和 PSCR，但它们在二维 token、姿态热图或跨模态补全里做文章；我们把航拍和地面的差异定义为三维身体表面可见性不一致，用 SMPL mesh 或 joints 建立 canonical body surface，再做可见表面对齐和缺失表面补全。这个切开点比“加姿态注意力”强。  
cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。

2. Headline：不要学习一个统一跨视角距离，学习每对样本的最优几何可见距离。  
挂载资产：SMPL 3D 几何、SOLIDER-Swin、aerial-ground。  
区别：最像 Optimal Illumination Distance 和 MDDM。光照论文是为每对样本找最优光照，MDDM 是双空间对齐；我们的点是为每对航拍和地面样本选择最可信的身体表面、视角或关节子空间，不再强迫所有样本共享同一种全局距离。  
cheap kill-switch：先做纯评测级首验。提取全局特征和按 SMPL 区域聚合的局部特征，对每个 query-gallery 对取可见区域 soft-min 距离或 visibility-penalized 距离。若不训练都没有稳定改善，说明“最优几何距离”信号弱，不继续写复杂损失。

3. Headline：把航拍遮挡、尺度退化和背景错配变成正噪声注入，而不是数据增强小技巧。  
挂载资产：aerial-ground、pose 热图门控、SMPL。  
区别：最像车辆 noise injection，但那篇服务于地面 occluded ReID，噪声来源是车辆图像。我们的噪声是跨视角特有的结构噪声，包括俯视身体压缩、低清、地面大背景、非人体遮挡和可见身体区域缺失。关键是噪声由姿态或 SMPL 可见性控制，不能只是随机遮挡。  
cheap kill-switch：先做三种极简注入：按姿态热图遮挡非关键区域、按 SMPL 可见性 drop 掉地面图像的部分身体区域、把航拍背景块注入地面图。只跑 Swin-Tiny 小规模快跑。如果损失正常但验证集无一致提升，说明正噪声假设不成立。

4. Headline：用三维可见性重判无监督跨视角伪标签，解决“不是不同人，而是看见的身体不同”这一类假负样本。  
挂载资产：SMPL 3D 几何、aerial-ground、SOLIDER-Swin。  
区别：最像 OPMCN、NNSC-FR 和 PFCD。它们在普通 ReID 里用相机代理、最近邻边界和局部全局特征去修伪标签；我们专门处理 aerial-ground 的几何假负样本，用可见身体表面重叠度和视角差异判断 hard proxy，而不是只看特征相似度或 camera id。  
cheap kill-switch：在有标签的 AG 数据上先模拟无监督聚类，统计 false negative 是否集中在低可见重叠样本对。然后加入一个 visibility-aware proxy 合并规则，看聚类纯度、NMI 或后续小训练是否提升。若伪标签错误和可见性没有相关性，这条直接停。

我最建议优先首验第 1 和第 2。它们最能挂住团队的 SMPL 基建和 aerial-ground 场景，也最容易和这批工作切开：不是再做遮挡补全、姿态注意力或双空间对齐，而是把跨视角 ReID 的核心矛盾定义成三维身体表面可见性不一致。
tokens used
54,819
已按每篇前三页读完，下面只基于摘要和引言判断。

**逐篇提取**

1. `MDDM`
创新类型是新机制。它把 VI-ReID 的问题拆成低高层语义错位和特征分布错位，用欧氏空间和 Hilbert 空间双空间匹配加互蒸馏来讲。能发的点在于它没有只做单一 embedding 对齐，而是把“单空间对齐不够”包装成 gap。story 是：跨模态不应只在一个潜空间里拉近，而要在互补空间中找一致身份线索。

2. `NNSC-FR`
创新类型是新机制加工程组合。它抓住无监督 ReID 中聚类边界 hard sample 的伪标签不稳定问题，用最近邻样本约束和类间平衡损失处理边界，再用 ODE 引导的残差重构增强特征。能发靠的是伪标签边界样本这个明确痛点。story 是：无监督失败不是所有样本都错，而是边界样本反复污染聚类。

3. `Occluded person Re-ID with noise injection`
创新类型是问题重定义和新数据利用。它把车辆图像从“噪声”改写成“有益结构噪声”，用车辆数据模拟真实遮挡并做正则。能发的 gap 是 occluded ReID 数据小，真实车辆遮挡常见但被丢弃。story 很强：噪声不是要清除，而是要可控注入，让模型学会在遮挡下识别人。

4. `OCCNet`
创新类型是新数据设定加新机制。它定义 occluded VI-ReID，指出遮挡和跨模态会叠加造成空间错位和模态退化，并构造 Occluded-SYSU、Occluded-RegDB。能发靠新设定和补全机制双支撑。story 是：遮挡 ReID 不应只看可见区域，而要利用单模态上下文和跨模态互补信息补全缺失身份线索。

5. `Optimal Illumination Distance`
创新类型是问题重定义。它反对“光照不变”和“统一光照”这两条老路，提出每对图像有自己的最优光照距离。能发的 gap 是统一归一化会损失对某些身份有判别力的内容。story 是：不要把所有图像推到同一个标准光照，而要为每对样本选择最有判别力的光照条件。

6. `OPMCN`
创新类型是新机制。它聚焦无监督代理对比学习里的正负样本选择错误，尤其是假负样本和相机代理选择。能发靠的是把 camera proxy 从普通分组变成“最优代理挖掘”。story 是：无监督 ReID 的关键不是更多对比，而是选对最有信息量的代理对。

7. `PFCD`
创新类型是新机制偏工程组合。它用全局和局部特征互补生成伪标签，再用 GMM 和知识蒸馏去伪标签噪声。能发的 gap 是全局特征聚类有偏，局部信息过去没有真正参与聚类。story 是：伪标签质量来自全局语义和局部细节共同投票，而不是单一全局特征。

8. `PSCR`
创新类型是新机制。它批评已有姿态方法只在 backbone 输出端做 mask，导致遮挡信息已经扩散到高层特征里。它把姿态和骨架区域提前引入中间层，并用骨架区域建模和 cross-attention 融合。story 是：姿态不该只是后处理过滤器，而应该在特征形成过程中阻断遮挡扩散。

9. `ADM`
创新类型是问题重定义加新数据设定。它把 ReID 隐私问题从“模糊、加扰”改成“生成新身份替换真实身份”，并提出匿名化数据集和 ISD 指标。能发靠的是伦理合规和可用性之间的明确矛盾。story 是：隐私保护不是隐藏真实 ID，而是从源头替换成可训练但不可追踪的新 ID。

**这批论文反映出的可发套路**

最值得借的是三类：第一，把传统噪声重新定义成有益监督，例如车辆噪声、伪标签噪声、边界样本。第二，把“统一不变表示”改成“样本对自适应的最优条件”，光照论文给了很好的叙事模板。第三，把辅助信息从输出端后处理前移到特征形成过程，PSCR 这个角度比普通姿态 mask 更能讲机制创新。

**面向团队资产的强创新候选**

1. Headline：从跨视角 ReID 改写为“跨视角身体表面可见性补全”问题。  
挂载资产：aerial-ground 跨视角 ReID、SMPL 3D 几何、SOLIDER-Swin。  
区别：最像 OCCNet 和 PSCR，但它们在二维 token、姿态热图或跨模态补全里做文章；我们把航拍和地面的差异定义为三维身体表面可见性不一致，用 SMPL mesh 或 joints 建立 canonical body surface，再做可见表面对齐和缺失表面补全。这个切开点比“加姿态注意力”强。  
cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。

2. Headline：不要学习一个统一跨视角距离，学习每对样本的最优几何可见距离。  
挂载资产：SMPL 3D 几何、SOLIDER-Swin、aerial-ground。  
区别：最像 Optimal Illumination Distance 和 MDDM。光照论文是为每对样本找最优光照，MDDM 是双空间对齐；我们的点是为每对航拍和地面样本选择最可信的身体表面、视角或关节子空间，不再强迫所有样本共享同一种全局距离。  
cheap kill-switch：先做纯评测级首验。提取全局特征和按 SMPL 区域聚合的局部特征，对每个 query-gallery 对取可见区域 soft-min 距离或 visibility-penalized 距离。若不训练都没有稳定改善，说明“最优几何距离”信号弱，不继续写复杂损失。

3. Headline：把航拍遮挡、尺度退化和背景错配变成正噪声注入，而不是数据增强小技巧。  
挂载资产：aerial-ground、pose 热图门控、SMPL。  
区别：最像车辆 noise injection，但那篇服务于地面 occluded ReID，噪声来源是车辆图像。我们的噪声是跨视角特有的结构噪声，包括俯视身体压缩、低清、地面大背景、非人体遮挡和可见身体区域缺失。关键是噪声由姿态或 SMPL 可见性控制，不能只是随机遮挡。  
cheap kill-switch：先做三种极简注入：按姿态热图遮挡非关键区域、按 SMPL 可见性 drop 掉地面图像的部分身体区域、把航拍背景块注入地面图。只跑 Swin-Tiny 小规模快跑。如果损失正常但验证集无一致提升，说明正噪声假设不成立。

4. Headline：用三维可见性重判无监督跨视角伪标签，解决“不是不同人，而是看见的身体不同”这一类假负样本。  
挂载资产：SMPL 3D 几何、aerial-ground、SOLIDER-Swin。  
区别：最像 OPMCN、NNSC-FR 和 PFCD。它们在普通 ReID 里用相机代理、最近邻边界和局部全局特征去修伪标签；我们专门处理 aerial-ground 的几何假负样本，用可见身体表面重叠度和视角差异判断 hard proxy，而不是只看特征相似度或 camera id。  
cheap kill-switch：在有标签的 AG 数据上先模拟无监督聚类，统计 false negative 是否集中在低可见重叠样本对。然后加入一个 visibility-aware proxy 合并规则，看聚类纯度、NMI 或后续小训练是否提升。若伪标签错误和可见性没有相关性，这条直接停。

我最建议优先首验第 1 和第 2。它们最能挂住团队的 SMPL 基建和 aerial-ground 场景，也最容易和这批工作切开：不是再做遮挡补全、姿态注意力或双空间对齐，而是把跨视角 ReID 的核心矛盾定义成三维身体表面可见性不一致。
