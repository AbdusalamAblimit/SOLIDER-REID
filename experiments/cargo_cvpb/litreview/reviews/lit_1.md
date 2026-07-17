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
session id: 019ef71f-4843-7a53-b235-0a4ffa417b15
--------
user
你是 ReID 论文创新挖掘员 1/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(1/20):
'Knowledge and experience' for visible-infrared person re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf
A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
ADA framework for unsupervised domain adaptation person re-identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我会先按项目规则读目标和状态文件，然后只用 `pdftotext -l 3` 读取这 9 篇的前三页，最后从团队资产角度反推候选创新点。
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
/bin/zsh -lc 'pwd && ls' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.622.11653/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
目标文件确认过了，当前目标就是文献差距分析，不开实验。我现在分批抽取前三页，重点看摘要、引言里的问题定义、机制和证据组织。
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
/bin/zsh -lc "pdftotext -l 3 'Knowledge and experience'\\'' for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
I/O Error: Couldn't open file 'Knowledge and experience' for visible-infrared person re-identification.pdf': No such file or directory.

exec
/bin/zsh -lc "pdftotext -l 3 'A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

1

A Semantic-Aware Attention and Visual Shielding
Network for Cloth-Changing
Person Re-Identification
Zan Gao , Member, IEEE, Hongwei Wei, Weili Guan , Member, IEEE, Jie Nie , Member, IEEE,
Meng Wang , Fellow, IEEE, and Shengyong Chen , Senior Member, IEEE

Abstract— Cloth-changing person re-identification (ReID) is a
newly emerging research topic that aims to retrieve pedestrians
whose clothes are changed. Since the human appearance with
different clothes exhibits large variations, it is very difficult
for existing approaches to extract discriminative and robust
feature representations. Current works mainly focus on body
shape or contour sketches, but the human semantic information
and the potential consistency of pedestrian features before and
after changing clothes are not fully explored or are ignored. To
solve these issues, in this work, a novel semantic-aware attention
and visual shielding network for cloth-changing person ReID
(abbreviated as SAVS) is proposed where the key idea is to shield
clues related to the appearance of clothes and only focus on visual
semantic information that is not sensitive to view/posture changes.
Specifically, a visual semantic encoder is first employed to locate
the human body and clothing regions based on human semantic
segmentation information. Then, a human semantic attention
(HSA) module is proposed to highlight the human semantic
information and reweight the visual feature map. In addition,
a visual clothes shielding (VCS) module is also designed to extract
a more robust feature representation for the cloth-changing task
by covering the clothing regions and focusing the model on
the visual semantic information unrelated to the clothes. Most
importantly, these two modules are jointly explored in an endto-end unified framework. Extensive experiments demonstrate
Manuscript received 17 July 2022; revised 9 October 2023; accepted
29 October 2023. This work was supported in part by the National Natural
Science Foundation of China under Grant 62372325 and Grant 61872270; in
part by the Young Creative Team in universities of Shandong Province under
Grant 2020KJN012; in part by the Jinan 20 Projects in universities under Grant
2020GXRC040; and in part by the Shandong Project toward the Integration
of Education and Industry under Grant 2022PYI001, Grant 2022PY009, and
Grant 2022JBZ01-03. (Corresponding authors: Hongwei Wei; Zan Gao.)
Zan Gao is with the Shandong Artificial Intelligence Institute, Qilu University of Technology (Shandong Academy of Sciences), Jinan 250014, China,
and also with the Key Laboratory of Computer Vision and System, Ministry of
Education, Tianjin University of Technology, Tianjin 300384, China (e-mail:
zangaonsh4522@gmail.com).
Hongwei Wei is with the Shandong Artificial Intelligence Institute, Qilu
University of Technology (Shandong Academy of Sciences), Jinan 250014,
China (e-mail: wmolang@163.com).
Weili Guan is with the Faculty of Information Technology, Monash University, Clayton, VIC 3800, Australia (e-mail: honeyguan@gmail.com).
Jie Nie is with the College of Information Science and Engineering, Ocean
University of China, Qingdao 266100, China (e-mail: niejie@ouc.edu.cn).
Meng Wang is with the School of Computer Science and Information
Engineering, Hefei University of Technology, Hefei 230009, China (e-mail:
eric.mengwang@gmail.com).
Shengyong Chen is with the Key Laboratory of Computer Vision and
System, Ministry of Education, Tianjin University of Technology, Tianjin
300384, China (e-mail: sy@ieee.org).
Color versions of one or more figures in this article are available at
https://doi.org/10.1109/TNNLS.2023.3329384.
Digital Object Identifier 10.1109/TNNLS.2023.3329384

that the proposed method can significantly outperform state-ofthe-art methods, and more robust features can be extracted for
cloth-changing persons. Compared with multibiometric unified
network (MBUNet) (published in TIP2023), this method can
achieve improvements of 17.5% (30.9%) and 8.5% (10.4%) on
the LTCC and Celeb-reID datasets in terms of mean average
precision (mAP) (rank-1), respectively. When compared with the
Swin Transformer (Swin-T), the improvements can reach 28.6%
(17.3%), 22.5% (10.0%), 19.5% (10.2%), and 8.6% (10.1%) on
the PRCC, LTCC, Celeb, and NKUP datasets in terms of rank-1
(mAP), respectively.
Index Terms— Cloth-changing person re-identification (ReID),
human semantic attention (HSA), semantic-aware, visual clothes
shielding (VCS).

I. I NTRODUCTION

T

HE person re-identification (ReID) task is to explore the
usefulness of image retrieval techniques in the public
security domain. It is an upstream task of the person detection
or person localization tasks, while the ReID task needs to
find additional clues about the target person based on the
already-acquired person images or video sequences. A typical
person ReID system aims to discover matching persons from
a gallery library and return the retrieval sequence based on
the query probe. Furthermore, as air pollution continues to
rise, individuals frequently wear face masks as a precautionary
measure in their everyday routines. Additionally, surveillance
cameras often capture face images at a noticeably reduced
size. Consequently, even advanced face recognition methods
often struggle to accurately identify individuals under such
circumstances. To solve this issue, researchers [1], [2], [3],
[4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
[16], [17] have developed the person ReID technique, which
is an important supplement to the face recognition technique
[18], [19] and a special case of feature extraction [20], [21],
[22], and several person ReID datasets [23], [24], [25] have
been released. Moreover, Ye et al. [17] have surveyed the
existing person ReID approaches. We can find that these
approaches are very effective for the person ReID task with
short time spans where the human appearance features are
fully used for visual matching, but when the surveillance
acquisition period becomes longer, the complexity of clothing
changes subsequently increases. Fig. 1 shows some examples
of cloth-changing person ReID images, where each row displays the images of the same person wearing different clothes.
From them, we can observe that the differences in the visual
appearances of the same person with different clothes are

2162-237X © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
2

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

Fig. 1. Examples of cloth-changing person ReID images. The images in
each row belong to the same person with different clothes.

very large, and it is also very difficult for humans to identify
them. In other words, when data from different cameras are
collected for a long time, the appearance of the person’s
clothing cannot be used as the feature representation. If the
existing person ReID approaches are directly applied in this
case, their performance deteriorates dramatically, and they
often fail. The reason for this is that the above approaches
mainly rely on visual clothing appearance, and they cannot
provide a robust feature representation for each identity (ID)
with different clothes; thus, discriminative and robust feature
extraction is an urgent issue for the cloth-changing person
ReID task.
To date, a few researchers [26], [27], [28], [29], [30], [31],
[32], [33], [34], [35], [36] have made useful attempts for
the cloth-changing person ReID task. For example, Huang
et al. [27] proposed an augmented representation with vector
neurons for cloth-changing. Yang et al. [29] proposed human
contour information and polar coordinate transformation to
obtain the results of pedestrian matching. Qian et al. [30] proposed a shape embedding module and a clothing-eliminating
shape-distillation module (SE + CESD). To enrich clothing
styles, Zheng et al. [31] proposed a generative adversarial
model (GAM) module to expand the training data. Yu et al.
[32] proposed a new solution by involving rich clothing
templates in training, and in the retrieval, different clothing
templates are added into the query in sequence. Jin et al.
[33] proposed a framework called GI-ReID that leverages gait
recognition as an auxiliary task to learn cloth-agnostic representations for efficient and latency-free person ID matching
in surveillance. Yang et al. [34] proposed SirNet-based on
positive and negative sample clustering to increase interclass
differences and reduce intraclass gaps. These approaches are
very good for trying to solve the cloth-changing person ReID
problem, but since the human appearance exhibits large variations with different clothes, it is very difficult for existing
approaches to extract discriminative and robust feature representations. Moreover, current works mainly focus on body
shape or contour sketches, but the human semantic information
and the potential consistency of pedestrian features before and
after changing clothes are not fully explored or are ignored.

To solve these issues, in this work, we propose a novel
end-to-end SAVS algorithm for the cloth-changing person
ReID task to obtain more discriminative and robust features
that are irrelevant to clothes. The two key points of the
problem to be solved in the field of cloth-changing person
ReID: 1) the same person wearing different clothes and
2) different people wearing the same clothes. The SAVS
method has already eliminated both effects from appearance
simultaneously by attentional weighting and shielding pixels.
Extensive experimental results on four cloth-changing person
ReID datasets demonstrate that SAVS can outperform stateof-the-art person ReID approaches, and more discriminative
and robust features can be obtained that can effectively solve
the cloth-changing issue. The main contributions of this article
are summarized as follows.
1) We develop a novel end-to-end SAVS network for
cloth-changing person ReID that consists of visual
semantic encoding and visual semantic decoding. The
key idea is to shield clues related to the appearance of
clothes and only focus on visual semantic information
that is not sensitive to view/posture changes. In this way,
the negative effect of the clothing information can be
reduced as much as possible.
2) We design a human semantic attention (HSA) module
to highlight the human information and reweight the
visual feature map that is very helpful for obtaining
more discriminative features, and then we develop a
visual clothes shielding (VCS) module to extract a more
robust feature representation by focusing the model
on the visual semantic information unrelated to the
clothes. Most importantly, these two modules are jointly
explored in an end-to-end unified framework. In this
way, more discriminative and robust features can be
extracted that are irrelevant to the cloth-changing or pose
variants.
3) We systematically and comprehensively evaluate the
SAVS algorithm on four public cloth-changing person
ReID datasets, and the experimental results demonstrate
that the SAVS approach can obtain more discriminative and robust features that are irrelevant to clothes;
moreover, it can significantly outperform state-of-theart cloth-changing person ReID methods in terms of the
mean average precision (mAP) and rank-1.
The remainder of this article is organized as follows.
Section II introduces the related work, and Section III
describes the proposed SAVS method. Section IV describes the
experimental settings and the analysis of the results. Section V
presents the details of the ablation study, and concluding
remarks are presented in Section VI.
II. R ELATED W ORK
To date, many person ReID approaches have been proposed.
According to the person’s visual appearance, these methods
can be roughly divided into clothing-consistent person ReID
and cloth-changing person ReID. In the following, we will
separately introduce them.
A. Clothing-Consistent Person ReID
In earlier times, people [31], [37], [38] made efforts to
develop related methods for clothing-consistent person ReID
where the visual appearance of the clothes was consistent
for the same person. For example, Sun et al. [9] proposed

Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
GAO et al.: SEMANTIC-AWARE ATTENTION AND VISUAL SHIELDING NETWORK

a part-based convolutional baseline (PCB) module where a
base feature map was first obtained and then equally divided
into six feature blocks in the horizontal direction. This method
is simple but very effective and has become an important
benchmark in the field of person ReID. Wang et al. [14]
proposed the multiple granularity network (MGN), a multibranch deep network architecture where one branch is built
for the global feature representation and two branches are built
for the local feature representation to capture the preference
information of the pedestrian ID categories from the whole
image. In this way, discriminative information with various
granularities can be obtained via an end-to-end feature learning
strategy. Gao et al. [16] proposed a deep spatial pyramid-based
collaborative feature reconstruction model (DCR) where all
blocks of the person were jointly reconstructed; in this way, the
issues of occlusion, pose changes, and observation perspective
changes can be solved. In addition, some methods employ
human skeleton points or human surface texture as a priori
knowledge to guide pedestrian ReID. Song et al. [39] proposed
a mask-guided contrastive attention model (MGCAM) to learn
features separately from the body and background regions;
moreover, a novel region-level triplet loss was designed to
restrain the features learned from different regions. Miao et al.
[40] proposed a novel pose-guided feature alignment (PGFA)
method where a pose estimator was utilized to detect key
points of the human body in pedestrian images, and then
these keypoints were used to decide whether a specific body
part was occluded. Gao et al. [5] proposed a novel texture
semantic alignment (TSA) approach with visibility awareness
for the partial person ReID task, where the occlusion issue
and changes in poses were simultaneously explored in an endto-end unified framework. Zheng et al. [41] proposed for the
first time to solve the person ReID problem in 3-D space by
learning features from human appearance and 3-D geometric
structure in a coherent manner. Wang et al. [15] proposed
a novel spatial rescaling (SpaRs) layer to help convolutional
neural networks (CNNs) to see more, and it introduced spatial
relations among the feature map activations back to guide
the model to focus on a broad area in the feature map.
Hou et al. [2] proposed a novel interaction-aggregation-update
(IAU) block to comprehensively leverage the spatial–temporal
context information for high-performance person reID. Zhang
et al. [1] introduced a unified attribute-guided collaborative
learning scheme tailored for partial person ReID. In their work,
they proposed an adaptive threshold-guided masked graph
convolutional network, which effectively incorporates human
attributes and a cyclic heterogeneous graph convolutional
network. This integration facilitates the fusion of cross-modal
pedestrian information through both intragraph and intergraph interactions. Ye et al. [17] performed a comprehensive
examination of closed-world person ReID from three distinct
viewpoints, providing an in-depth analysis. They further evaluated the strengths of existing person ReID approaches and
proposed a robust attention generalized mean pooling with
weighted triplet loss (AGW) baseline method. Although these
models are robust to changes caused by poses, lighting, and
viewing angles, they are vulnerable to clothing changes, as the
models heavily rely on the consistency of the appearance of
clothes.
B. Cloth-Changing Person ReID
Since the visual appearance of the pedestrians in the
cloth-changing person ReID task changes dramatically after

3

a long period, it is very difficult to extract more discriminative and robust feature representations. If the existing
clothing-consistent person ReID methods are directly applied
to this task, their performance is unsatisfactory. Therefore,
to accelerate the development of cloth-changing person ReID
techniques, some cloth-changing person ReID datasets have
been built and released, such as LTCC [30], PRCC [29], CelebreID [27], and NKUP [42]. Moreover, several researchers [27],
[28], [29], [30], [35], [36], [43], [44], [45], [46] have made
some attempts to address this problem and then assessed their
performance on a certain dataset. For example, Yang et al. [29]
proposed a spatial polar transformation (SPT) + angle-specific
extractor (ASE) module, where human contour sketching
information was used to substitute for human color information. Moreover, an SPT layer was designed to transform the
contour sketch image, and then a multistream network was
used to aggregate multiple granular features to better discriminate people by changing the sampling range of the SPT layer.
In this way, the changes in visual appearance caused by clothing changes could be reduced. Qian et al. [30] proposed an
SE + CESD module, where the main idea was to completely
delete information related to the appearance of clothes and
only focus on body shape information that is not sensitive to
changes in perspective and posture. The former was used to
encode shape information from human body keypoints, and the
latter was utilized to adaptively distill the ID-relevant shape
features. Huang et al. [27] designed a ReIDCaps module where
a vector neuron concept was proposed. For each vector neuron,
its direction was used to represent the changes in clothing
information, and its length was utilized to identify the people.
In this way, the clothing changes of a specific person can be
perceived, and the auxiliary modules can be used to enhance
the robustness of the module. Zheng et al. [31] proposed a
jointly couples discriminative and generative learning in a
unified network (DG-Net) module where a generative model
was utilized to automatically generate person images with
different appearances regarding clothing. Li et al. [26] and
Yu et al. [32] proposed a new solution for changing clothes
called clothes changing person set (COCAS) and COCAS plus
(COCAS+), respectively, where rich clothing templates were
supplied; thus, in the query, both the clothing template image
and an image of the target person wearing other clothes were
fed into the module to find the target image. Gao et al. [45]
proposed a novel multigranular visual-semantic embedding
algorithm (MVSE) for cloth-changing person ReID, where
visual semantic information and human attributes are embedded into the network. Hong et al. [44] proposed a fine-grained
shape-appearance mutual learning framework that can learn
fine-grained discriminative body shape knowledge in a shaped
stream and transfer it to an appearance stream to complement
the clothing-unrelated knowledge in the appearance features.
Shu et al. [46] proposed a semantic-guided pixel sampling
approach for the cloth-changing person ReID task which
forces the model to automatically learn clothing-irrelevant
cues that are irrelevant to upper clothes and pants. Gu et al.
[47] proposed a clothes-based adversarial loss (CAL) to mine
clothes irrelevant features from the original RGB images by
penalizing the predictive power of the ReID model. Yang
et al. [36] proposed a causality-based autointervention model
(AIM) to mitigate clothing bias for robust cloth-changing
person ReID. Yang et al. [43] proposed an Auxiliaryfree Competitive IDentification (ACID) model to achieve a
win–win situation by enriching the ID-preserving information
conveyed by the appearance and structure features while

Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 '3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
12830

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025

3D-Aided Pedestrian Representation Learning for
Video-Based Person Re-Identification
Guquan Jing , Graduate Student Member, IEEE, Peng Gao , Graduate Student Member, IEEE, Yujian Lee,
Yiyang Hu, and Hui Zhang , Senior Member, IEEE

Abstract—Video-based person re-identification (Re-ID) aims
to match the target pedestrian from video sequences. Recent
methods perform frame-level feature extraction followed by
temporal aggregation to obtain video representations. However,
they pay insufficient attention to the quality of frame-level
features, which suffer from issues including multi-frame misalignment, partial occlusion and appearance confusion. People
live in a 3D space. 3D pedestrian representations can provide
rich geometric information and shape cues that offer promising
solutions to these challenges in video-based Re-ID. To mitigate these issues, this paper proposes a 3D-Aid Pedestrian
Representation Learning (3DAPRL) network, which introduces
3D modality to video-based Re-ID. Specifically, two novel modules
are designed, i.e., the Cross-Modal Fusion (CMF) module and
the Shape-aware Spatial-Temporal Interaction (SSTI) module,
to enhance pedestrian representation learning. The CMF module generates discriminative fusion representations by utilizing
3D pedestrian data, while the SSTI module learns spatialtemporal 3D shape representation which are distinguishable for
finding the target pedestrian in video scenarios. Both features
generated from the CMF and SSTI modules contribute to
the final video representation. Extensive experiments on four
challenging video-based Re-ID datasets demonstrate that our
3DAPRL network reaches better performance than state-of-thearts methods.
Index Terms—Video-based person re-identification, crossmodal feature fusion, 3D pedestrian representation, video
representation.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) [1], [2] is a critical
task in intelligent surveillance, aiming to identify specific
pedestrians across non-overlapping cameras. Early studies

Received 30 November 2024; revised 19 May 2025; accepted 1 July 2025.
Date of publication 7 July 2025; date of current version 8 December 2025.
This work was supported in part by Guangdong Provincial Key Laboratory of
Interdisciplinary Research and Application for Data Science, Beijing Normal
University-Hong Kong Baptist University United International College (BNUHKBU) United International College under Grant 2022B1212010006; in part
by the Natural Science Foundation of China under Grant 62076029; in part
by the National Key Research and Development Program of China under
Grant 2022YFE0201400; and in part by Guangdong Science and Technology
Department under Grant 2022B1212010006. This article was recommended
by Associate Editor Y. Wang. (Corresponding author: Hui Zhang.)
Guquan Jing, Peng Gao, and Yiyang Hu are with the Department
of Computer Science, Hong Kong Baptist University, Hong Kong, SAR,
China, and also with Beijing Normal University-Hong Kong Baptist University (BNU-HKBU) United International College, Zhuhai 519000, China
(e-mail: guquanjing1@gmail.com; gaopeng1225@gmail.com; yiyanghu124@
gmail.com).
Yujian Lee and Hui Zhang are with Guangdong Provincial Key Laboratory of Interdisciplinary Research and Application for Data Science,
BNU-HKBU United International College, Zhuhai 519087, China (e-mail:
r130034019@mail.uic.edu.cn; amyzhang@uic.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3586808

[3], [4], [5], [6], [7], [8] focused on image-based Re-ID,
i.e., identifying pedestrians from single-shot images. With
the development of video devices, video-based Re-ID (video
Re-ID) [9], [10], [11] that uses multi-frame image sequences
as input has attracted widespread attention. Compared to
image-based Re-ID, video Re-ID offers a more promising scenario for capturing richer temporal and spatial
information.
Early video Re-ID methods primarily adopt models
designed for video recognition (including RNNs [12], [13],
[14], [15], 3D CNNs [16], [17], [18], optical flow [19], [20],
etc.) to learn video temporal information directly, which are
not well-suited for video Re-ID tasks. Recent studies [9],
[21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31],
[32], [33], [34] have made significant progress by proposing
novel modules to aggregate extracted frame-level features
temporally; however, they fail to adequately consider the
quality of the frame-level features. Multi-frame misalignment,
partial occlusion and appearance confusion frequently occur
in video data, compromising frame-level features and causing
the generation of suboptimal video representations. In specific,
some of these methods [35], [36], [37], [38], [39] retain the
local information of pedestrians through designed modules,
such as those that account for the importance of each feature’s
spatial position [39]. Attention mechanisms or local pooling
are used to highlight relevant parts of the pedestrian in these
methods. However, partial occlusion introduces spatial noise,
which integrates irrelevant information into frame-level features and degrades the overall performance of these methods.
In addition, spatial misalignment across frames often yields
uneven feature quality, which in turn can induce temporal
misalignment during feature aggregation. Other methods [21],
[31], [32], [33] attempt to derive global representations by
pooling frame-level features [9]; some [31] also refine framelevel features using these global cues. Nevertheless, their
performance is significantly undermined by the aforementioned multi-frame misalignment and partial occlusion. These
issues can corrupt global features and make it difficult to
spatially emphasize the target pedestrian, leading to a loss of
discriminative spatial information and causing temporal misalignment. Meanwhile, both methods struggle with appearance
confusion, where negative samples with similar appearance
are misidentified as the target pedestrian, ultimately leading to non-discriminative video representations. Figure 1 (a)
shows these challenges in video Re-ID, where two pedestrians with similar appearance are affected by multi-frame

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID

Fig. 1. (a) Two pedestrian sequences from video Re-ID datasets exhibiting
similar appearance, with challenges including multi-frame misalignment (red
box) and partial occlusion (blue box). (b) An example of our 3D pedestrian representation including mesh, point cloud and colored point cloud.
(c) Overview of our proposed framework. Note that in (c), black lines/boxes
denote the basic structure of the previous methods. On this basis, red
lines/boxes represent our additional processes.

misalignment (red boxes) and partial occlusion (blue boxes). In
such scenarios, effective spatial emphasis is hindered directly,
leading to discriminative spatial information loss and temporal
misalignment.
The inherent limitations of relying solely on 2D image to
overcome these issues and other challenges in Re-ID motivate the exploration of auxiliary information. Recent studies
incorporate various modalities (such as text [30], events [28]
and 3D data [40], [41], [42], [43], [44], etc.). Among these,
3D data, such as meshes rendered by models like SMPL [45],
is particularly promising. It offers rich geometric information
and shape cues, which are valuable for extracting detailed and
robust pedestrian features. Motivated by this, Zheng et al. [40]
propose a 3D pedestrian feature generation module that utilizes
both 3D geometric information from reconstructed meshes and
2D appearance cues from images to enhance feature representation. While the extracted 3D feature contributes to pedestrian
representation, their methods lack sufficient consideration for
the effective fusion of cross-modal information, limiting the
potential of fused features. Specifically, their fusion strategies
project RGB information directly into 3D space [40] or simply
concatenate 2D image features with 3D features [41], [42],
leading to coarse feature aggregation and limited cross-modal
interaction. We consider that the main challenge in feature
fusion for Re-ID with 3D representations lies in the modality
gap between 2D images and 3D data, causing inconsistency
in cross-modal features. Such a feature inconsistency are
manifested in feature misalignment across dimensions and
the disparate information captured from different modalities.
On the one hand, the dimensional misalignment between
2D and 3D pedestrian features leads to the loss of crucial
information during the fusion process. On the other hand,
3D pedestrian feature could provide incomplete pedestrian
representation due to the coarse extraction process, resulting
in inconsistency with 2D features and yielding suboptimal
fusion outcomes. Some methods [41], [42] leverage the 3D
shape parameter included with SMPL model to obtain spatialtemporal pedestrian shape cues using attention mechanisms,
but only limited and coarse-grained information is extracted.

12831

In addition, although they utilize 3D information in video
cloth-changing Re-ID (VCCReID) scenarios, its potential in
video Re-ID benchmark remains underexplored.
To address the aforementioned problems, we propose a
novel 3D-Aid Pedestrian Representation Learning (3DAPRL)
network to incoperate 3D modality into video-based Re-ID
(see Figure 1 (b) for an example of the 3D pedestrian representation and (c) for a conceptual illustration of our framework).
Our network exploit a Cross-modal Fusion (CMF) module and a Shape-aware Spatial-Temporal Interaction (SSTI)
module to obtain accurate frame-level fusion representations
and spatial-temporal 3D shape representations, respectively.
Before that, we first convert 3D meshes to point clouds, and
then employ a pillar-based backbone network [46] to extract
pillar-based features (i.e., pseudo-image features), effectively
reducing the feature inconsistency between 2D and 3D modalities. The generated features contain pedestrian geometry,
appearance, and shape information, while being dimensionally aligned with image features. Besides, the pillar-based
backbone network captures the complete pedestrian geometry
in 3D space, enabling a comprehensive feature representation. Consequently, this step narrows the modality gap and
establishes a more consistent feature space for subsequent
processing stages in our framework. After obtaining pillarbased features, the CMF and SSTI modules are employed to
overcome challenges in video Re-ID. Detailed explanations of
these modules are provided below.
The CMF module is introduced to effectively mitigate the
impacts of multi-frame misalignment and partial occlusion.
It achieves this by adaptively fusing the pillar-based features
described above with image features. This fusion process
emphasizes crucial areas of the target pedestrian and integrates
essential geometric and shape cues. In detail, the fusion
strategy exploits deformable cross-attention [47] to learn
correlations between image and 3D modalities, generating
3D-enhanced features. Accurate pedestrian positions derived
from the mapping relationship between 2D images and
3D point clouds serve as reference points, guiding the model
to focus on pedestrian-relevant areas and improving crossmodal feature alignment in the latent space. Additionally, the
CMF module adaptively determines relevant regions for fusion
and capture the local context of pedestrians based on offsets
learned from pillar-based features. Moreover, we retain the
generated pillar-based feature for preserving 3D geometric
and shape cues. Thus, the proposed CMF module generates discriminative fusion pedestrian representations that fuses
frame-level features with 3D geometric and shape information,
mitigating the effects of multi-frame misalignment and partial
occlusion.
To further enhance model ability against these issues, and
in particular to address the challenge of appearance confusion,
it is necessary to utilize a new stable pedestrian characteristic,
i.e., video-level 3D shape. The pedestrian shape in 3D space
offers robustness against similar appearances [41], [42] and
provides a complementary shape representation. Our Shapeaware Spatial-Temporal Interaction (SSTI) module is therefore
proposed to effectively learn and leverage these discriminative 3D shape representations of the target pedestrian from

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

12832

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025

video sequences for improved identification. Specifically, the
3D shape parameters estimated by the SMPL model are
used as the initial input. Subsequently, we create temporal
and spatial streams in parallel to adopt feature contextualization along space and time axes the attention. The spatial
stream focuses on obtaining 3D shape information within each
frame, while the temporal stream models 3D shape variations
across frames. The extracted spatial and temporal features
are then fused to produce spatial-temporal representation.
Our SSTI module allows the network to fully exploit both
spatial and temporal dependencies. Its receptive field captures
spatial-temporal interactions. Hence, discriminative videolevel 3D shape representations are yielded, which contributes
to the final video representation. Extensive experiments on
four challenging video Re-ID datasets demonstrate that the
3DAPRL outperforms the state-of-the-art methods.
In summary, the contributions of this study are as follows:
• We propose a novel 3D-Aid Pedestrian Representation
Learning (3DAPRL) network that introduces 3D modality
to video-based Re-ID. To the best of our knowledge, this
is the first method to leverage 3D modality for addressing
video-based Re-ID (video Re-ID) task.
• We propose a Cross-modal Fusion (CMF) module to
fuse image features and pillar-based features. In the
CMF module, the pillar-based feature helps the model
to focus on the crucial area while providing pedestrian
geometric information and shape cues. Consequently, the
discriminative frame-level pedestrian feature is generated
to reduce the influence of the multi-frame misalignment
and partial occlusion.
• We design a Shape-aware Spatial-Temporal Interaction
(SSTI) module to extract distinctive spatial-temporal
3D shape representations, which contributes to the video
representation. It serves as a new pedestrian characteristic
to further mitigate the impact of appearance confusion,
multi-frame misalignment and partial occlusion.
• Extensive experiments on four widely used video Re-ID
datasets including MARS, iLIDS-VID, PRID-2011 and
DukeMTMC-VID demonstrate that our proposed method
outperforms the state-of-the-art methods.
II. R ELATED W ORKS
Related works are organized as follows. In Section II-A, we
provide a summary of recent video Re-ID methods and discuss
their limitations. Section II-B offers a detailed review of existing methods that incorporate 3D modality for person Re-ID.
We also emphasize the novelty of our approach and highlight
how it differs from these methods. Eventually, methods in the
3D feature learning are introduced in Section II-C.
A. Video-Based Person Re-ID
Remarkable progress has been made in video-based Re-ID
in recent years. Early studies exploit deep learning techniques
for video recognition task including optical flow [19], [20],
3D convolutions [16], [17], [48], [49] and recurrent models
[12], [13], [14], [15], [50] to obtain temporal features directly.
Chung et al. [19] proposed a Siamese network with two

separately streams to extract spatial and temporal features.
Specifically, one stream processes RGB images while the other
processes optical flow. McLaughlin et al. [12] employ CNNs
to extract image features, followed by RNNs for temporal
information extraction. To tackle the spatial misalignment
issue in methods using RNNs, Liao et al. [48] apply 3D convolutions to extract and aggregate spatial and temporal features.
Li et al. [49] propose a multi-scale 3D convolutional layer,
which offers a more compact and efficient solution compared
to traditional 3D convolutions. Despite effectively encoding
temporal information, the techniques used in these methods
are not specifically designed for video Re-ID. They also face
challenges such as multi-frame misalignment, partial occlusion, and appearance confusion in video datasets, which often
lead to performance degradation. Most of methods [1], [20],
[21], [22], [24], [27], [29], [31], [36], [38], [39], [51], [52],
[53], [54], [55], [56], [57], [58], [59], [60] focus on aggregate
frame-level feature temporally by using attention mechanisms
in video Re-ID to obtain discriminative video representations.
For instances, Li et al. [36] build the spatial and temporal
attention to extract useful information from visible body parts.
Wang et al. [21] introduced a pyramid structure that aggregates
spatial and temporal features by utilizing attention-based modules designed to capture spatial and temporal information. In
order to address the spatial misalignment in video-level data,
Chen et al. [27] propose an attention mechanism to interact
each frame with others in the video, enhancing both local
and global features. Despite the effective temporal aggregation of frame-level features using attention, challenges such
as multi-frame misalignment, partial occlusion, and appearance confusion still affect model performance. To mitigate
these issues, some methods incorporate additional modalities
such as event data [28] to provide supplementary information. However, the limited information provided by event
data is insufficient to resolve these challenges. To overcome
these limitations, we introduce a novel network incorporating
3D modality.
B. Person Re-ID With 3D Pedestrian Representation
Different from single-modality Re-ID, cross-modality
Re-ID [28], [40], [41], [43], [61], [62] focuses on retrieving
pedestrians using diverse modality representations, such as
3D pedestrian mesh. In the context of person Re-ID with
3D representation, existing approaches [40], [41], [43], [61],
[62], [63] primarily concentrate on extracting distinctive 3D
features. The incorporation of 3D modality is shown to
enhance the Re-ID performance by providing complementary
3D information, which positively impacts retrieval accuracy.
To obtain distinguishable 3D pedestrian features, some
methods use the SMPL model [45] to generate 3D representation. OGNet [40] first employs 3D representation for image
Re-ID by converting 2D pedestrian images into 3D space via
off-the-shelf 3D human reconstruction networks [64], [65],
[66]. It leverages a GNN structure to build the location
topology and extracts multi-scale semantic information of 3D
pedestrians. Liu et al. [62] reconstruct 3D clothed pedestrians
from images and learn the distinguished feature from the
3D clothed body, naked body and RGB information for

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 '4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 180 (2026) 113966

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

4SNet: Spatial and Spectrum Self-adaptive Synergy Network for
Visible-Infrared Person Re-identification
Mingfu Xiong a , Feiyang Luo a , Junjie Huang a ,∗, Yifei Guo b , Aziz Alotaibi c , Sambit Bakshi d ,
Javier Del Ser e,f , Khan Muhammad g ,∗
a

School of Computer Science and Artificial Intelligence, Wuhan Textile University, Wuhan, 430200, Hubei, China
School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan, 430074, Hubei, China
c Department of Computer Science, College of Computers and Information Technology, Taif University, Taif, 21974, Saudi Arabia
d Visual Surveillance Laboratory, Department of Computer Science and Engineering, National Institute of Technology Rourkela, Rourkela, 769008, Odisha, India
e TECNALIA, Basque Research and Technology Alliance (BRTA), Derio, 48160, Bizkaia, Spain
f
Department of Mathematics, University of the Basque Country (UPV/EHU), Leioa, 48940, Bizkaia, Spain
g
Visual Analytics for Knowledge Laboratory (VIS2KNOW Lab), Department of Applied Artificial Intelligence, School of Convergence, College of Computing and
Informatics, Sungkyunkwan University, Seoul, 03063, Republic of Korea
b

ARTICLE

INFO

Keywords:
Visible-Infrared Person Re-identification
Adaptive Frequency Filter
Spectrum Synergy Module
Adaptive filtering
Identity-shared patterns

ABSTRACT
Visible-Infrared Person Re-identification (VI-ReID) is a cross-modality retrieval task that involves searching
for a specific pedestrian across different modalities (visible or infrared). Existing methods have explored crossmodality alignment from different perspectives, including spatial-domain cues and frequency-domain modeling.
Nevertheless, how to adaptively capture modality-specific frequency characteristics and effectively integrate
them into cross-modality feature alignment remains an open challenge. To address this challenge, this work
presents a novel and efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet) for the VI-ReID
task, which incorporates an Adaptive Frequency Filter (AFF) module and a Spectrum Synergy Module (SSM)
to extract robust cross-modality-invariant features. Specifically, in contrast to traditional fixed convolutionkernel operations, the AFF module reconstructs a set of learnable frequency bases by adopting a dual spatialand channel attention-guided strategy, which dynamically synthesizes a customized Spectrum filter for each
input. Complementarily, the SSM decomposes features into multiple frequency bands, which explicitly decouple
identity-shared patterns from modality-specific traits and synergistically enhance them, thereby achieving
consistent feature representation across cross-modal pedestrian images at the cross-modality feature alignment
level. Extensive experiments have been conducted on the public VI-ReID datasets (SYSU-MM01, LLCM, and
RegDB), demonstrating that our proposed method achieves competitive performance when compared to stateof-the-art approaches. Source code implementing 4SNet is publicly available at: https://github.com/dekusaklasdroid/4SNet.

1. Introduction
Visible-Infrared Person Re-identification (VI-ReID) is a crossmodality retrieval task that aims to search for images of the same pedestrian across a non-overlapping, distributed camera network, which
captures visible images during the daytime and infrared images during
nighttime or low-light conditions [1,2]. Due to its cross-perspective
search trait, VI-ReID has found widespread applications in criminal
investigations, target tracking, and robotic systems [3,4], among other
application domains [5,6]. It has also emerged as a significant academic
concern in fields like pattern recognition and artificial intelligence [7,
8]. However, the large differences between daytime and low-light

nighttime conditions, coupled with the constant variations in the
appearance and clothing of the same individual across modalities,
have exacerbated the difficulty and challenges of this modeling task,
remaining as an unresolved task yet [9,10].
Existing VI-ReID methods can be roughly classified into two categories: (1) image-level based methods, and (2) feature-level based
methods. Specifically, image-based methods resort to generative models, such as Generative Adversarial Networks (GANs) [11], and Diffusion models [12], to transform images from one modality to another.
Alternatively, some of them generate an intermediate auxiliary embedding modality that lies between the distributions corresponding to

∗ Corresponding authors.

E-mail addresses: xmf2013@whu.edu.cn (M. Xiong), 2415283016@wtu.edu.cn (F. Luo), jjhuang@wtu.edu.cn (J. Huang), yifeiguo@hust.edu.cn (Y. Guo),
azotaibi@tu.edu.sa (A. Alotaibi), bakshisambit@ieee.org (S. Bakshi), javier.delser@tecnalia.com (J. Del Ser), khan.muhammad@ieee.org (K. Muhammad).
https://doi.org/10.1016/j.patcog.2026.113966
Received 6 January 2026; Received in revised form 26 April 2026; Accepted 12 May 2026
Available online 19 May 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 180 (2026) 113966

M. Xiong et al.

the visible and infrared modalities [13,14], aimed at shrinking the
gap between them. Despite the practical and intuitive nature of these
methods, their heavy reliance on additional models renders the converted (generative) modalities vulnerable to noise [15,16]. Moreover,
the complexity of their underlying learning algorithms adds to the
overall computational burden of the VI-ReID approach, leaving ample
room for further performance enhancements. In contrast, feature-based
methods hinge on projecting data from diverse modalities into a unified
feature embedding space [17,18]. Subsequently, feature enhancement
techniques (e.g., attention-based mechanisms [19,20]) are employed
for recognizing patterns corresponding to the same pedestrian. Although these techniques diminish the reliance on additional models,
direct projection also struggles to entirely bridge the domain gaps
caused by substantial differences in the data distribution characteristics
across various modalities. Several recent methods have embraced this
strategy [21,22]. One notable example is DEEN [23], which employs a
center-guided pair mining strategy to create diverse auxiliary embedding spaces between visible and infrared modalities, thereby reducing
their differences. Another example is FDNM [24], which employs a
salience-guided and amplitude nuance mining strategy to extract crossmodality frequency domain information, thereby decreasing modality
discrepancies and enhancing VI-ReID performance.
Although these methods have achieved promising performance,
they still face two main issues:
1. When it comes to mining intrinsic attributes within a single modality,
existing methods primarily rely on a fixed, pre-learned set of convolution kernels that are uniformly applied to samples from all modalities. As a result, these static operators fail to capture modalityspecific frequency-domain characteristics (such as structural textures and body contours) present within individual modalities. For
instance, a filter that is optimal for extracting fine textures in visible
images may inadvertently amplify interference noise when applied
to infrared images (e.g., strong background–foreground contrast, as
illustrated in Fig. 1(a)). This limitation ultimately constrains the
network’s ability to discover highly discriminative, modality-specific
features.
2. Regarding feature alignment across different modalities, although several recent studies have introduced frequency-domain modeling,
many existing approaches still rely primarily on spatial-domain
cues, such as appearance attributes and relative positional relationships, for cross-modality alignment. However, due to the substantial distributional differences between visible and infrared images,
spatial representations are often sensitive to modality-specific variations, which limits their robustness for alignment. In this context, frequency-domain information (encompassing global structure,
coarse appearance distribution, and complementary edge-like responses) provides features that are relatively more stable across
modalities. Therefore, effectively exploiting these spectral characteristics remains crucial for improving cross-modality feature alignment.

Fig. 1. Motivation for the overall architecture of the 4SNet framework. (a)
Grayscale Fast Fourier Transform (FFT) magnitude spectra illustrate that
infrared images concentrate most of their energy in low-frequency bands,
whereas visible images contain richer medium- and high-frequency details.
This clear spectral divergence suggests that different modalities exhibit distinct
frequency responses, making fixed shared filters inadequate for capturing
modality-specific discriminative features. (b) Motivated by this observation,
4SNet introduces two complementary frequency-domain modules, namely the
Adaptive Frequency Filter (AFF) and the Spectrum Synergy Module (SSM). A
comparison between a baseline fixed stem (denoted as Conv1) and AFF under
visible and infrared inputs shows that fixed shared filters are suboptimal for
early-stage feature extraction, which justifies the use of AFF for input-adaptive
and modality-specific frequency filtering. Meanwhile, the visualization of
low-, middle-, and high-frequency decompositions reveals that different spectral bands preserve complementary structural and fine-grained details. The
SSM explicitly models this band-wise decomposition and their synergistic
relationships to achieve more robust cross-modality feature alignment.. (For
interpretation of the references to colour in this figure legend, the reader is
referred to the web version of this article.)

2. For the second problem, the SSM decomposes the characteristic
spectrum into low-, medium-, and high-frequency components.
These components are then processed in parallel using band-specific
operators (e.g., lightweight convolutional blocks). This strategy retains stable low-frequency body structures and coarse appearances,
while exploiting complementary mid-to-high frequency variations
and fine edge details.

To overcome these limitations, this study introduces a novel and
efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet),
which integrates an Adaptive Frequency Filter (AFF) module and a
Spectrum Synergy Module (SSM) to respectively extract singlemodality-robust features and accomplish cross-modality feature alignment for the VI-ReID task, as shown in Fig. 1(b). Specifically:

Finally, a fusion module is utilized to integrate these features,
with a particular focus on the enhanced low-frequency components to
guide the alignment of cross-domain features. Extensive experiments
on three public VI-ReID datasets (SYSU-MM01 [25], RegDB [26], and
LLCM [23]) demonstrate that 4SNet achieves competitive performance.
Ablation studies and feature visualizations further validate its superior
retrieval capability.

1. For the first issue, the AFF module dynamically generates inputspecific filters to address the limitations of static convolutional
kernels. The module first learns a set of frequency bases from images
using spatial attention. These bases are then adaptively combined
under channel attention guidance, synthesizing customized filters
that extract modality-inherent frequency-domain features.
2

Pattern Recognition 180 (2026) 113966

M. Xiong et al.

The rest of the manuscript is organized as follows: Section 2 comprehensively reviews related work in the VI-ReID field. Section 3
elaborates on the proposed 4SNet framework. Section 4 validates the
effectiveness of the proposed method through a series of extensive experiments described previously. Finally, Section 5 concludes the paper
and outlines potential future research directions rooted on our findings.

2.3. Frequency domain analysis in vision tasks
The frequency domain offers a complementary perspective for image representation and understanding, providing intrinsic robustness
against certain spatial transformations and local perturbations [40].
By decomposing visual signals into components that capture global
structure and fine-grained details at different spectral scales, frequencydomain analysis enables models to disentangle information that is often
entangled in the spatial domain. As a result, frequency-aware modeling
has been successfully applied across a wide range of computer vision
tasks. In particular, studies in domain adaptation and generalization
have demonstrated that low-frequency components, which predominantly encode global shape and coarse structural information, tend to
be more stable and domain-invariant than high-frequency components
that capture texture and local appearance variations [1,41]. These
properties make frequency representations especially suitable for scenarios involving significant domain shifts. Beyond domain adaptation,
frequency-domain insights have also been leveraged to improve adversarial robustness, where sensitivity to high-frequency perturbations
plays a critical role [42].
Motivated by this observation, several works have incorporated
frequency-domain operations directly into deep neural networks. Some
approaches have highlighted the effectiveness of frequency-aware channel attention in selectively emphasizing informative spectral responses
[43,44], while others have gone a step further by partially or fully replacing standard spatial convolutions with frequency-domain counterparts [45,46]. More recently, wavelet-guided designs such as PST [47]
have demonstrated the benefits of frequency-aware decomposition in
surface-defect detection, where carefully separating frequency components helps enhance meaningful details while suppressing irrelevant
noise. Along this line, several recent VI-ReID methods, including FDMNet [48] and FDNM [24], have explored frequency-domain modeling
through amplitude–phase decomposition, motivating further investigation into frequency-aware designs for cross-modality matching
problems.
Contribution. While prior frequency-based methods such as FDMNet [48] and FDNM [24] have achieved promising performance
through amplitude–phase decomposition, they typically treat frequency
modeling as a single, unified mechanism for reducing modality discrepancy. In these approaches, amplitude and phase components are
jointly manipulated to encourage modality-invariant representations,
implicitly assuming that a common frequency processing strategy is
sufficient for both intra-modality feature extraction and inter-modality
alignment. As a consequence, the intrinsic differences in frequency
distribution characteristics between visible and infrared modalities are
not explicitly addressed at different stages of the network. In contrast,
4SNet introduces a role-decoupled frequency modeling strategy, in
which frequency information is explicitly assigned to two complementary and well-defined roles within the framework. Specifically, the
AFF operates at the input stage in a dual-stem, non-shared manner
to perform modality-specific and input-adaptive frequency filtering,
enabling visible and infrared images to be characterized by customized
frequency responses before entering the shared backbone. The Spectrum Synergy Module (SSM) addresses cross-modality alignment as
a separate modeling problem by adopting a coarse low-, medium, and high-frequency band decomposition and performing band-wise
processing and fusion. This design emphasizes robust shared structural
cues while preserving complementary discriminative details, thereby
better meeting the practical requirements of day–night visible–infrared
matching.

2. Related work
2.1. Image-based VI-ReID methods
As previously mentioned, image-based methods depend on generative models (such as GANs [11] or Diffusion models [12]) to synthesize
an auxiliary data modality that bridges the gap between visible and
infrared modalities, minimizing the differences between the two modalities and effectively addressing the VI-ReID task. DiVE [12] is a recent
proposal that adopts this strategy: it generates visible-IR paired images
from samples with the same identity ID using a fine-tuned diffusion
model, thereby extending the text-driven approach to achieve identity
consistency for the VI-ReID task. Pan et al. [27] employ a conditional diffusion model, which is grounded on binary modal indicators
and modal independence, to respectively generate intermediate modal
images of the target modality and pedestrian identity. The bidirectional modal interaction network proposed in [2,28] also follows an
image-based strategy by separately generating samples for each other’s
modality to lower the difficulty of modal recognition. CSDN [29] and
DMA [30] have proposed, respectively, a CLIP-driven semantic discovery network and a dual modality-aware alignment method, designed to
generate intermediate modalities. Other similar approaches, like [31,
32], also adopt the strategy of generating intermediate modalities at
the image level to address the VI-ReID problem.

2.2. Feature-based VI-ReID methods
Unlike their image-based counterparts, feature-based methods [33]
mainly map the features of the two modalities to a shared embedding
space, and subsequently unify the modeling and mapping processes to
achieve cross-modality feature alignment. Some approaches [25,31]
employ a two-stream architecture, which focuses on aligning feature distributions through metric learning and adversarial training.
CPN [34] tackles VIS-NIR person ReID through angular metric learning, projecting features into a modality-invariant hypersphere space.
IDKL [17] explored feature disentanglement to separate modalityinvariant from modality-specific information, enhancing the shared
representation. TSKD [1] proposes a two-stage distillation for progressive cross-modality alignment and intra-modality reduction, robust
under weak supervision. MSCMNet [32] extracts multi-scale shared
semantic features to accomplish cross-modality feature alignment tasks.
DMDL [35] further introduces a dual-level modality debiasing learning
framework for unsupervised VI-ReID, where modality bias is reduced
at both the model and optimization levels. In addition, related works
such as [36,37], aim to achieve cross-modality feature alignment
by extracting intrinsic features and cross-domain invariant features
separately, and using other auxiliary information (such as semantic
text-guided) to bridge the modality gap. Recent endeavors [38,39]
have also pivoted towards part-level alignment, introducing dynamic
and semantically-aware strategies to surpass the limitations of simple
horizontal striping (namely, dividing an image into a fixed number of
equal-sized horizontal segments, which fails to account for changes
in pose or body part location). Different from these methods, 4SNet
addresses supervised VI-ReID from a frequency-modeling perspective.

3. Proposed 4SNet framework
In this section we introduce the proposed 4SNet network, whose
structure is summarized in Fig. 2. Firstly, the AFF module dynamically detailed in Section 3.2 generates a unique filter to capture the
3


codex
前三篇已经显示出一个规律：能发的论文通常先把失败来源重新命名成具体 gap，再配一个能被可视化或消融验证的机制。比如 3D 视频 ReID 不是说“加 3D”，而是把它绑定到帧级错位、遮挡和外观混淆。
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
/bin/zsh -lc "pdftotext -l 3 \"Knowledge and experience' for visible-infrared person re-identification.pdf\" -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
I/O Error: Couldn't open file 'Knowledge and experience' for visible-infrared person re-identification.pdf': No such file or directory.

exec
/bin/zsh -lc "pdftotext -l 3 'A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
A Survey on 3D Skeleton Based Person Re-Identification: Taxonomy, Advances,
Challenges, and Interdisciplinary Prospects
Haocong Rao and Chunyan Miao
College of Computing and Data Science, Nanyang Technological University (NTU), Singapore
Joint NTU-UBC Research Centre of Excellence in Active Living for the Elderly (LILY), NTU, Singapore
{haocong001, ascymiao}@ntu.edu.sg

Person re-identification via 3D skeletons is an important emerging research area that attracts increasing attention within the pattern recognition community. With distinctive advantages across various
application scenarios, numerous 3D skeleton based
person re-identification (SRID) methods with diverse skeleton modeling and learning paradigms
have been proposed in recent years. In this paper, we provide a comprehensive review and analysis of recent SRID advances. First of all, we
define the SRID task and provide an overview
of its origin and major advancements. Secondly,
we formulate a systematic taxonomy that organizes existing methods into three categories centered on hand-crafted, sequence-based, and graphbased modeling. Then, we elaborate on the representative models along these three types with an
illustration of foundational mechanisms. Meanwhile, we provide an overview of mainstream supervised, self-supervised, and unsupervised SRID
learning paradigms and corresponding common
methods. A thorough evaluation of state-of-theart SRID methods is further conducted over various
types of benchmarks and protocols to compare their
effectiveness, efficiency, and key properties. Finally, we present the key challenges and prospects
to advance future research, and highlight interdisciplinary applications of SRID with a case study. A
curated collection of valuable resources is available
at https://github.com/Kali-Hac/3D-SRID-Survey.

1

Introduction

Person re-identification (re-ID) is an essential pattern recognition task of matching and retrieving a person-of-interest
across different views or scenes, which has been widely applied to security authentication, smart surveillance, healthcare, and embodied AI [Nambiar et al., 2019; Ye et al.,
2021]. Recent economical and precise skeleton-tracking devices (e.g., Kinect [Shotton et al., 2011]) have simplified
the acquisition of 3D skeleton data, enabling them to be a
prevalent and versatile data modality for gait analysis and

Hand-Crafted

Similarity

I=1

Sequences

...
3D Skeletons
�1 , �2 , ⋯, ��

Graphs

I=2
Query ID
I=3

Body/Motion Feature Learning
Database Records ��

•••

arXiv:2401.15296v3 [cs.CV] 4 Jan 2026

Abstract

Figure 1: Overview of 3D skeleton based person re-ID (SRID) task
with hand-crafted, sequence-based or graph-based modeling to learn
effective body and motion features for identity recognition.

person re-ID [Liao et al., 2020; Rao et al., 2024]. Unlike conventional person re-ID methods that rely on appearance or facial characteristics [Ye et al., 2021], 3D Skeleton
based person Re-ID (SRID) models typically exploit bodystructure features and motion patterns (e.g., gait [Murray et
al., 1964]) from 3D positions of key body joints to identify
different persons. With unique merits such as small input
data, light-weight models, privacy-preserving without using
appearances, and robustness against view and background
variations [Han et al., 2017], SRID has attracted surging attention from both academia and industry [Rao et al., 2022].
In recent years, research on SRID has gained significant momentum, leading to diverse skeleton modeling and
learning paradigms. Early endeavors [Barbosa et al., 2012;
Munaro et al., 2014a; Andersson and Araujo, 2015; Pala et
al., 2019] mainly extract hand-crafted features such as skeleton descriptors in terms of anthropometric, geometric and gait
attributes of body. As these methods often require domain
expertise such as anatomy and kinematics [Yoo et al., 2002]
for skeleton modeling, they lack the ability to fully mine latent high-level features beyond human cognition. To resolve
this challenge, recent mainstream methods [Liao et al., 2020;
Huynh-The et al., 2020; Rao et al., 2022; Rashmi and Guddeti, 2022] leverage deep neural networks to automatically
perform skeleton representation learning for SRID. One of
exemplar methods (termed “sequence-based modeling”) is to
model sequential dynamics and motion semantics from raw
or normalized skeletons (e.g., joint trajectory) based on long
short-term memory (LSTM) and its variants [Wei et al., 2020;
Rao et al., 2022]. However, they rarely investigate the intrinsic body relationships such as inter-joint motion correlations,

First depth-based
person re-ID method
(Haque et al.) (2016)
First skeleton-based
First deep learning based
person re-ID (SRID) &
SRID method
hand-crafted method
SM-SGE;
(AGE)
(Munaro et al.)
CTR-GCN

First image-based
person re-ID method
(Gheissari et al.)

2006

...

2010

...

First video-based
person re-ID Method
(Farenzana et al.)

2014

...

2017

...

First visible-Infrared
person re-ID Method
(Wu et al.)

First piont cloud based
person re-ID Method
(Munaro et al.)

2020

2021

First Transformer-based
model for SRID (TranSG);
GaitTR; GPGait
SGE-LA

2022

2023

First graph-based
model for SRID
(MG-SCR)

First sequence-based
& self-supervised
SRID method (AGE)

2025

SkeletonGait;
Hi-MPC
(2024)

First unsupervised
SRID method (SimMC)

First motif-based model
for SRID (MoCos)

(a) Origin and Advancements of SRID Research

(b) Model Accuracy & Efficiency Comparison

Figure 2: (a) Overview of research origin and technical advancements of SRID within the person re-ID community (Zoom in and follow the
timeline for the best view). (b) Parameter sizes (Millions (M)), computational complexity (Giga Floating Point Operations (GFLOPs)), and
KS20 Rank-1 accuracy of state-of-the-art deep learning methods for SRID (Red: Sequence-based models; Green: Graph-based models).

thereby possibly overlooking some valuable skeleton patterns. Another paradigm (termed “graph-based modeling”)
mitigates this challenge by constructing skeleton graphs to
model discriminative structural and actional features based on
the interrelations of body parts [Rao and Miao, 2023]. This
often requires multi-granularity body modeling and efficient
relational reasoning mechanisms (e.g., collaborative learning) based on skeleton graphs. Despite the great progress of
SRID, this rapidly evolving technique still lacks a systematic
review, making it difficult for researchers to gain a holistic
view of this field and embark on new research endeavors.
In light of this, we present the first survey on SRID, elucidating recent advancements of skeleton modeling, learning
paradigms, evaluation benchmarks, current challenges, and
interdisciplinary applications. Firstly, we define the SRID
task and provide a milestone overview to illustrate the origin
and key advancements of SRID as shown in Fig. 2a. Secondly, we propose a systematic taxonomy of SRID methods
to categorize them into hand-crafted, sequence-based, and
graph-based modeling, and elaborate on their foundational
mechanisms and representative approaches. We also illustrate the basic definitions and common methods within three
mainstream SRID paradigms (supervised, self-supervised,
unsupervised). Thirdly, we introduce existing public benchmarks, evaluation metrics, and protocols for SRID, while
comprehensively evaluating state-of-the-art methods across
different benchmarks to compare their performance and efficiency. Meanwhile, we conduct qualitative analysis of different methods to compare their key properties with a discussion of advantages and disadvantages. Finally, we discuss the
current challenges in SRID and identify potential directions
for future research. An overview of promising SRID applications in interdisciplinary areas, spanning healthcare, embodied AI, and security, is further provided and illustrated with
a case study. The structure of this survey, including skeleton
modeling (Sec. 3), learning paradigms (Sec. 4), benchmarks
and evaluation (Sec. 5), challenges and prospects (Sec. 6) is
shown in Fig. 3. We hope our survey can bring new insights
to researchers and expedite future research in SRID.

2

Overview

2.1

Task Description

As illustrated in Fig. 1, the input of SRID task is a 3D skeleton sequence that belongs to a certain pedestrian, and the
output is the predicted identity. Formally, we denote a 3D
skeleton sequence as S = (s1 , · · · , sf ) ∈ Rf ×j×3 , where
st ∈ Rj×3 denotes the tth skeleton with 3D coordinates of
j body joints. Each skeleton sequence S corresponds to a
person identity I, where I ∈ {1, · · · , C} and C is the number of different classes (i.e., identities). In the SRID task, we
generally have training set, probe set, and gallery set, respecn oN2
n oN1
, and
, ΦP = S P
tively denoted as ΦT = S Ti
i
i=1
i=1
n oN3
that contain N1 , N2 , and N3 skeleton seΦG = S G
i
i=1
quences of different persons collected from different scenes
or views. The task target is to learn a hand-crafted, sequencebased or graph-based model (detailed in Sec. 3) that maps
3D skeleton sequences into effective representations, so that
we can query the correct identity of an encoded skeleton sequence representation in the probe set via matching it with
the sequence representations in the database (i.e., gallery set).
SRID is essentially a retrieving and matching problem.

2.2

Origin and Advancements

As presented in Fig. 2a, the first SRID research [Munaro
et al., 2014a] using hand-crafted skeleton descriptors commenced in 2014, coming after the first RGB video based
method [Farenzena et al., 2010] and before the first depthbased approach [Haque et al., 2016]. Then, Rao et al. proposed the first deep learning based SRID paradigm in 2020,
followed by the first self-supervised, unsupervised, and supervised paradigms [Rao et al., 2020; Rao and Miao, 2022;
Rao and Miao, 2023]. Over the past five years, an increasing number of innovative models have been devised specifically for SRID and related emerging tasks, including LSTM
models (e.g., AGE [Rao et al., 2020]), CNN models (e.g.,
SkeletonGait [Fan et al., 2024]), Transformer models (e.g.,
TranSG [Rao and Miao, 2023]), GCN models (e.g., CTR-

Body Relations

MoCos

SM-SGE

Hi-MPC

BIWI

Rank-1

Probe-Gallery
Matching

Scarcity

Robustness

Antropometric

Pose Dynamics

Graph Dynamics

TranSG

AGE

SimMC

IAS-Lab

mAP

RGB-Estimated

Imbalance

Generalizability

Geometric

Semantics Learning

Multi-Level
Graphs

GPGait

SGELA

...

KGBD

CMC

Cross-View

Noise

Interpretability

...

...

...

...

...

Gait

Joint Motion

Other Attributes

Importance Inference

Figure 3: Structure of this survey with the taxonomy of SRID research. Representative branches and SRID methods are listed.

GCN [Chen et al., 2021]), MLP models (e.g., SimMC [Rao
and Miao, 2022]), and hybrid/composite models (e.g., SMSGE [Rao et al., 2021a]) (compared in Table 2).

2.3

Taxonomy of SRID Methods

As shown in Fig. 3, we propose a systematic taxonomy
for SRID approaches based on skeleton modeling (see Sec.
3) and learning paradigms (see Sec. 4). For skeleton
modeling, we divide them into three categories, including
(1) hand-crafted modeling using manually-extracted features
(e.g., skeleton descriptors), (2) sequence-based modeling that
focuses on sequential features (e.g., joint trajectory) of 3D
skeletons, and (3) graph-based modeling that represents 3D
skeletons as graphs, and further subcategorize them by different learning focuses such as body relations or graph dynamics. In terms of learning paradigms, we group them
into (1) supervised SRID paradigms that require skeleton labels for feature learning, (2) self-supervised SRID paradigms
that combine pretext tasks for skeleton representation learning with labeled fine-tuning, and (3) unsupervised SRID
paradigms that learn skeleton features without using labels.

3

Skeleton Modeling

We elaborate on different skeleton modeling including their
foundational mechanisms and representative approaches.

3.1

Hand-Crafted Modeling

Gait Attributes. Extracting gait features is a common way
to characterize unique walking patterns of an individual [Cunado et al., 2003], typically including (1) kinematic parameters (e.g., angles of hips, knees, and feet), and (2) spatiotemporal parameters (e.g., stride length, gait cycle time, velocity). They can be manually computed with:
fangles = {(αij , βij ) | (i, j) ∈ Ψ},
(1)
αij = arctan

yi − yj
zi − z j
,
, βij = arctan p
xi − xj
(xi − xj )2 + (yi − yj )2
(2)

where xi , yi , zi denote 3D coordinates of the ith joint, the set
Ψ defines adjacent joints constrained by the human skeleton
model, and two kinds of angles αij and βij are calculated
from these joint pairs. The velocity can be calculated by
Pn strideLength
n
V elocity = i=1
,
(3)
cycleT ime

where
strideLength = 2 ∗ stepLength.
(4)
We calculate the step length by averaging the highest values
of the difference between the right and left feet, and adopt the
mean stride length over all n strides following [Andersson
and Araujo, 2015].
Anthropometric Attributes. The Euclidean distance between two joints, such as bone lengths, limb dimensions,
height, are usually computed as the anthropometric features
(fA ) to differentiate individuals [Barbosa et al., 2012]:
fA = {∥Ji − Jj ∥2 | (i, j) ∈ Ψ∗ },
(5)
where Ji =(xi , yi , zi ), Jj =(xj , yj , zj ), Ψ∗ not only contains
adjacent joints but also can be extended to cover more anthropometric properties, such as 13 (D13 ) or 16 (D16 ) skeleton
descriptors in [Munaro et al., 2014b; Pala et al., 2019].
Geometric Attributes. The incorporation of geometric
skeletal measurements, including body-part ratios and interjoint geodesic distances on the mesh surface, can enhance
feature representations in person re-ID [Barbosa et al., 2012].
The above hand-crafted features are often learned by different classifiers (e.g., KNN) to perform person re-ID [Andersson and Araujo, 2015; Nambiar et al., 2017]. They
are also combined with different metric algorithms [Pala
et al., 2015] or other modalities such as 3D point clouds
[Gharghabi et al., 2015; Bondi et al., 2016; Pala et al., 2019;
Munaro et al., 2014a] to further boost person re-ID accuracy.

3.2

Sequence-Based Modeling

Joint Motion. The walking patterns are characterized by the
motion of joints, which can be defined as the differences between body joint positions [Liao et al., 2020]:
fmotion = st − st−1 ,
(6)
where st = {J1t , J2t , · · · , Jjt }, Jit = (xti , yit , zit ) denotes
the 3D coordinates of ith joint in the tth skeleton, and i ∈
{1, 2, · · · , j}. The occluded or masked joints during walking
can also help models learn useful motion patterns [Rashmi
and Guddeti, 2022].
Pose Dynamics. The consecutive skeletons typically conveys dynamics of unique body poses, which can be encoded
sequentially by temporal learning models (e.g., LSTM):

ϕ(s1 )
if t = 1
ht =
,
(7)
ϕ(ht−1 , st ) if 1 < t ≤ f


exec
/bin/zsh -lc "pdftotext -l 3 'A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112567

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

A Semantic-guided occlusion simulation based local feature semantic
expansion network for person re-identiﬁcation
Zelin Deng a , Mingxuan Tang a , Ke Nai
a

a,∗, Guiji Li b , Shaomiao Chen c , Pei He d

School of Computer, Changsha University of Science and Technology, Changsha, 410114, China

b School of Computer Science and Engineering, Changsha University, Changsha, 410022, China
c

School of Computer Science and Engineering, Hunan University of Science and Technology, Xiangtan, 411201, China

d School of Computer Science and Cyber Engineering, Guangzhou University, Guangzhou, 510000, China

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Local semantic features
Occlusion simulation

a b s t r a c t
Person re-identiﬁcation is a challenging task which aims to retrieve images of a target pedestrian across diﬀerent
cameras. Currently, existing models generally neglect the semantic relevance of local features, which may cause
to limited identiﬁcation performance. To address this issue, we propose a semantic-guided occlusion simulation
based local feature semantic expansion network (FOSENet) to pursue satisfactory performance. Firstly, we propose a semantic-guided occlusion simulation (SGOS) method, which generates semantically relevant occlusion
patches and further simulates the occlusion based on the semantic information of pedestrians to improve the diversity of the occluded pedestrian samples. Then, we propose a local feature semantic expansion (LFSE) method,
which obtains several local areas around each key area and selects some useful local areas as additional cues of
the key area to enhance the discrimination ability. Finally, we introduce a cross-center oﬀset loss function, which
enlarges the distance of a pair of nearest neighbor samples from two classes to optimize the model. Thus, the
samples belonging to diﬀerent identities are pushed further away. Extensive experiments are conducted on four
challenging datasets, and the proposed method can achieve competitive results compared to multiple state-ofthe-art works. The code is publicly available at https://github.com/xuanthan-art/code-with-paper-FOSENet.git

1. Introduction
Person re-identiﬁcation aims to determine whether images captured by diﬀerent cameras contain the same individual by retrieving
pedestrian information across these cameras, which has been widely
used in artiﬁcial intelligence and security monitoring [1,2]. With the
rapid development of deep learning and the release of pedestrians reidentiﬁcation (Re-ID) image datasets, many powerful Re-ID methods
have been proposed to pursue impressive performance, such as multiscale methods [3,4], ﬁne-grained methods [5,6] and adversarial network learning [7–9]. However, person re-identiﬁcation still faces multiple challenges: (1) occlusion. In real-world environments like train
stations, airports, and shopping malls, pedestrian images captured by
stationary surveillance cameras often suﬀer from occlusion caused by
vegetation, vehicles, and other individuals. The occlusion may cause
a pedestrian to exhibit signiﬁcant appearance changes under diﬀerent
environments, which greatly limits the re-identiﬁcation accuracy [17].
(2) local features semantic representation.It is widely acknowledged
that CNN features are extracted from pre-trained models, as the depth

of the layers increases, CNN features tend to overly concentrate on a
single local center, which may lead to the neglect of other important
details. As a result, the semantic discriminability of the extracted features is likely to decrease, potentially aﬀecting the overall performance
of the model. In particular, some regions in pedestrian images, such
as ground surfaces, walls, and background objects, are repeatedly extracted, thereby generating redundant semantic information that is frequently attended by the model but does not contribute to identity discrimination or occlusion interference representations. These redundant
semantics may mislead feature learning and increase similarity across
diﬀerent identities. Therefore, how to remove the interfering semantic information and obtain more additional semantic information to enhance the feature representations of pedestrian identity is a challenge for
person re-identiﬁcation [10,11]. To address these issues, existing methods mainly focus on enhancing the robustness of the model to deal with
occluded samples and the extraction of semantic information from local
features.
On the one hand, to solve the occlusion problem, many methods
try to enhance the robustness of the model by simulating real-world

∗ Corresponding author.

E-mail addresses: zelin.deng@csust.edu.cn (Z. Deng), xuanthan@163.com (M. Tang), naike_hnu@hnu.edu.cn (K. Nai), guiji.li@hnu.edu.cn (G. Li),
csm123@hnust.edu.cn (S. Chen), bk_he@126.com (P. He).
https://doi.org/10.1016/j.patcog.2025.112567
Received 5 August 2024; Received in revised form 20 June 2025; Accepted 3 October 2025
Available online 10 October 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112567

Z. Deng et al.

occlusions through random erasure of image regions [12,13]. Then,
to make the occlusion patches more realistic for simulated occlusion,
Chen et al.[13] proposed a method to crop other images to create occlusion patches, and these patches were used at conventional occlusion
locations (such as top, bottom, left, and right) in pedestrian images to
enhance the robustness of the model to recognize occlusion situations.
Subsequently, in order to better model the feature extraction during occlusion, some works have proposed feature erasure [6]. Wang et al. [14]
introduced a feature erasure and diﬀusion network, which generates an
occlusion mask to guide the model to erase the feature map. As a result,
the model can focus on unobstructed regions and diﬀuse the pedestrian
representation in the feature space to boost the identiﬁcation capability of the target pedestrian. However, existing occlusion simulation or
erasure methods are positionally arbitrary and heavily dependent on
manual design. These methods fail to adequately simulate the occlusions that models encounter in real-world scenarios, where pedestrians
are frequently partially obscured by irregular objects. Moreover, the positions of such occlusions tend to interfere with critical semantic information. Thus, feature extraction and semantic recognition in diﬀerent
occlusion scenarios become more diﬃcult, and the model may focus on
the useless feature areas, leading to a signiﬁcant performance degradation in recognition accuracy.
On the other hand, to extract more semantic information, some methods employ auxiliary models to acquire additional semantic information
to address the issue of limited representation ability in local feature semantics [15]. Yang et al. [16] proposed to introduce discretizing posture
information into visibility labels of body parts to suppress the impact
of interfering information, thereby obtaining more posture information
from visible areas as additional semantic cues to enhance the recognition eﬃciency under various scenes. Subsequently, to extract more
precise semantic information, some methods employ multi-scale information to learn additional semantic information to improve the feature
representation ability of pedestrian images [17]. Gao et al.[18] proposed
a teacher-student ecoder (TSD) framework to aggregate pedestrian area
information through feature distillation and a mask generator was designed to provide discriminative regions for better re-identiﬁcation performance. However, the extraction of semantic information heavily relies on prior training outcomes. To extract more semantic information
that is not focused on by pre-trained models, Zhang et al. [19] introduced a complementary network for person re-identiﬁcation, which utilizes two branches to independently extract potentially discriminative
information that is beneﬁcial for pedestrian recognition from both background and salient regions across the global scope. The proposed network can eﬀectively capture additional semantic information such as
backpacks and handbags to improve the identiﬁcation performance in
pedestrian recognition tasks. However, the loss of partially discriminative features caused by real-world occlusions ampliﬁes the inﬂuence of
unoccluded key regions in recognition decisions, which may increase
the similarity between diﬀerent pedestrians and reduce the recognition
accuracy of the model. Meanwhile, existing methods rely heavily on the
unoccluded key regions, making them highly susceptible to misidentiﬁcation.
Based on the above analysis, we propose a semantic-guided occlusion
simulation based local feature semantic expansion network. Firstly, we
try to exploit semantic information to guide the simulation of occlusions to generate semantically occlusion patches. Specially, occlusion
operations are performed based on their position, shape, and context to
ensure that the generated occlusions are more consistent with the actual content of the images to better simulate real-world scenarios where
pedestrians are obstructed. Subsequently, to obtain richer semantic information for each individual pedestrian, some local areas near the key
area are selected, which are treated as additional cues to enhance semantic diversity. Finally, we introduce a cross-center oﬀset loss function, which enlarges the distance of a pair of nearest neighbor samples
from two classes during the training process to ensure that the samples
belonging to diﬀerent identities are pushed further away. We conducted

extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
realistic occlusion scenarios, which achieves better adaptability to deal
with complex visual interference. In addition, by mining local regions
with rich identity-related cues, it can obtain better discriminative ability
to perform person re-identiﬁcation under occlusion scenes.
The main contributions can be summarized as follows:
•

We propose a semantic-guide occlusion simulation method, which
generates semantically relevant occlusion patches with semantic information of object pedestrians and further realizes occlusion simulation based on their position, shape, and context to generate more
diverse occluded pedestrian samples.
• We develope a local feature semantic extension method, which selects some useful local regions near the key areas to obtain additional
semantic information, thereby enhancing the discriminative ability
of pedestrian features.
• We introduce a cross-center oﬀset loss function, which increases the
distance between a pair of nearest neighbor samples from two classes
to optimize the model and enhance its generalization ability.
• We conduct extensive experiments to demonstrate the eﬀectiveness of our proposed method, which achieves promising performance on four large-scale benchmark datasets: Market-1501
[20], DukeMTMC-REID [21], CUHK03-NP [22], and OccludedDukeMTMC [15].
2. Related work
In this section, we provide a brief overview of methods related to
person re-identiﬁcation, occluded person re-identiﬁcation, and partial
person re-identiﬁcation.
2.1. Person re-identiﬁcation
Pedestrian re-identiﬁcation is a crucial computer vision task that enables the re-identiﬁcation of the same individual across diﬀerent scenes
by analyzing and comparing images of pedestrians, and its research and
applications have obtained substantial progress due to the advancement
of convolutional neural networks [1,2] . One core challenge of person
re-identiﬁcation is how to accurately identify the same pedestrian under varying lighting scenes, viewpoints, and occlusions [23] Typically,
existing pedestrian re-identiﬁcation methods attempt to learn more reﬁned local feature semantic information to achieve ﬁne-grained feature matching [5]. Zhang et al.[24] proposed a method that leverages
graph neural networks to learn global and local features. The proposed
method can utilize the characteristics of graph neural networks to acquire more locally joint semantic information. Subsequently, to enhance
the discriminative ability across diﬀerent regions, some methods have
employed attention mechanisms to eﬀectively focus on pedestrian areas.
Chen et al. [3] proposed an attention pyramid method that employs attention regions in a multi-scale way to learn corresponding attention at
various ﬁne-grained levels to learn beneﬁcial human foreground areas,
thereby producing eﬀective features enriched with identity information.
Additionally, some methods enrich the diversity of features by extracting them from pedestrian images in a region speciﬁc or layered manner
[23]. Zhang et al. [25] proposed a Hierarchical Aggregation Transformer
method, which aggregates diﬀerent feature maps with the same Transformer and further calibrates the features with another Transformer
to enrich high-level semantic information to capture a more comprehensive set of global and local information. However, these methods
overlook the occlusion issue and fail to separate individuals from obstructions when interfering elements are semantically similar to the
pedestrian information, which will signiﬁcantly reduce the recognition
ability.
2

Pattern Recognition 172 (2026) 112567

Z. Deng et al.

2.2. Occluded person re-identiﬁcation

ages. Therefore, the goal of local pedestrian re-identiﬁcation is to match
the partial information in pedestrian images with the overall images in
the database [31]. To better learn diﬀerent local information, Dai et
al. [6] proposed the Batch DropBlock Network (BDB), which erases the
same area of images in a batch to concentrate on other areas and learn
richer local information. Subsequently, to better learn the local features
of pedestrians, Zhou et al. [32] simulated occlusions and introduced a
co-saliency branch to learn those parts of the pedestrians that are not
occluded. Additionally, Dong et al. [33] generated adversarial samples
by randomly erasing, transforming, and adding noise to feature maps
to learn undisturbed areas to obtain more reliable local information.
Recently, Yan et al. [34] leveraged the correlation between local features for aggregation and employed a reﬁned partitioning strategy to
concentrate and learn the visible areas more eﬀectively. Diﬀerent from
the above methods, Huang et al. [35] performed horizontal strip partitioning at various scales to better capture the representable information
of diﬀerent local parts of pedestrians to eﬀectively learn semantic information from local to global levels. To address the issue of partial occlusion, we propose selectively incorporating the features of adjacent local
regions of important local regions as supplementary information into
pedestrian features,to improve the discriminative ability to distinguish
pedestrians in locally similar scenarios.

In real-world scenarios, due to the ﬁxed positions of cameras, occlusions are inevitable when pedestrians are captured, which greatly
limits the re-identiﬁcation performance [26]. To address the issues arising from occluded pedestrians, some works have focused on simulating
occlusion scenarios to enhance the robustness against occlusions [12].
Huang et al [27]. proposed an Ad-Occluded Samples method, which
introduces deliberately designed occlusions into the training data and
trains the model after identifying key areas to better cope with realworld occlusion situations. Subsequently, to enable the model to better
learn with occlusions under diﬀerent scenes, Zhao et al [28]. proposed
an Incremental Generative Occlusion Adversarial Suppression Network,
which simulates occlusions from easy to diﬃcult to make the model
more robust to occlusions. The CBE module [29], which stands for Composite Batch Erasing, combines Batch-level Erasing (BCE) with Imagelevel Erasing (RE). These two erasing operations are applied to the original pedestrian images within the same batch to obtain two batches
of augmented images that are merged into a larger batch for training,
thereby enabling the model to be learned under occlusion scenes which
aﬀect diﬀerent body parts and of varying sizes. Some other methods
focus on directing the attention to visible areas to suppress the interference caused by occlusions. Jia et al. [30]proposed a set-matching
method without a collection, which uses a set matching approach to
concentrate on visible regions of the model. Subsequently, Wang et al.
[17] proposed a Key Point-Aware Occlusion Suppression and Semantic
Alignment (POS) method, which focuses on the unobstructed regions of
pedestrians to achieve semantic alignment to eﬀectively extract semantic information from the visible areas. However, these methods still have
some drawbacks. Firstly, their simulation strategy shows poor adaptability to the interference caused by occlusion in diﬀerent scenarios. Moreover, in real-world scenarios, they are unable to handle occlusions of
various shapes and may mistake the semantic information of occluding
objects for that of pedestrian features, which signiﬁcantly aﬀects the
recognition performance. In our proposed approach, we aim to more
eﬀectively simulate a diverse range of occlusion scenarios in order to
signiﬁcantly enhance recognition performance under various occlusion
interferences.

3. Methodology
3.1. Overview of the proposed method
The overall network structure proposed in this paper is shown in
Fig. 1. We propose a semantic-guided occlusion simulation based local
feature semantic expansion network (FOSENet) to solve the problem
of limited local semantic information representation ability caused by
semantic occlusion in real-world scenarios. Firstly, FOSENet integrates
occlusion patches with the semantic information of pedestrians to generate semantically relevant occlusion patches and further simulates the
occlusion with the generated occlusion patches and the location, shape,
and context of the semantic information of pedestrians to improve the
samples. Then, ResNet-50 is employed as a pretrained extractor to extract features from pedestrian images and obtain the attention weights
for each channel through an attention module to capture the key areas
and surrounding local areas. To ensure that the key areas have more
individual distinctiveness for representing pedestrian information, we
propose a local feature semantic expansion (LFSE) method. Speciﬁcally,
LFSE sorts the surrounding local areas of each key area extracted by the
attention module to select several local areas and fuses them with the

2.3. Partial person re-identiﬁcation
Due to issues such as occlusion, lighting, and changes in viewing
angle, the representable information of pedestrian images is limited,
which means useful information only partially exists on pedestrian im-

Fig. 1. Overall architecture of FOSENet, FOSENet mainly contains SGOS, LFSE, cross-center oﬀset loss function. Then, ResNet-50 is used as a backbone network.
FOSENet is designed to enhance model adaptability under occlusion conditions by generating diverse occlusion samples and mining local features with stronger
identity-discriminative ability.
3


exec
/bin/zsh -lc "pdftotext -l 3 'A training-free framework for text-to-image person re-identification via query-prototype matching.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113705

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

A training-free framework for text-to-image person re-identification via
query-prototype matching
Hao Yang a , Quan Zhang a , Jian-Fang Hu a ,∗, Jianhuang Lai a,b,c,d

,∗∗

a

Sun Yat-Sen University, Guangzhou, 510006, China
Pazhou Lab (HuangPu), Guangdong, 510000, China
c
Guangdong Province Key Laboratory of Information Security Technology, Sun Yat-Sen University, Guangzhou, 510006, China
d
Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education, Sun Yat-Sen University, Guangzhou, 510006, China
b

ARTICLE

INFO

Keywords:
Person re-identification
Text-to-image person retrieval
Training-free

ABSTRACT
Text-to-Image Person Re-identification (TI-ReID) aims to retrieve all images of a target individual from a gallery
given a natural-language description, yet existing methods still rely on instance-level similarity from pretrained
Vision–Language Models (VLMs), despite relevance being defined at the identity level. This instance–identity
mismatch, together with the modality gap and large intra-identity appearance variations, makes cross-modal
alignment fragile and leads to unstable rankings. We introduce a Prototype-Driven Enhancement (PDE)
module that exploits identity-level structure in the image modality purely at inference: gallery features are
clustered into identity prototypes, and the resulting Query–Prototype Similarity (QP-Similarity) is fused with
conventional Query–Gallery Similarity (QG-Similarity) to refine retrieval scores. Building on this, a cascaded
refinement scheme with adaptive prototype granularity yields a training-free, plug-and-play framework that
can be seamlessly attached to VLM-based TI-ReID backbones without modifying their parameters or using
additional data. Experiments on CUHK-PEDES, ICFG-PEDES, and RSTPReid show consistent improvements
over strong baselines and establish new state-of-the-art performance on all three benchmarks.

1. Introduction
Text-to-Image Person Re-identification (TI-ReID) is a cross-modal
retrieval task that retrieves images of a target person from a large
gallery given a natural-language description [1]. It is important for intelligent surveillance, human–computer interaction, and forensic analysis, where textual descriptions provide an accessible and privacypreserving query form. Unlike generic text–image retrieval, TI-ReID is
inherently identity-aware: each query is associated with one reference
image, while the retrieval target is all gallery images of the same
identity. This creates a notable instance–identity mismatch, because
supervision is given at the instance level whereas relevance is defined at the identity level. The problem is further complicated by
large intra-identity variations, such as changes in pose, illumination,
occlusion, and background, which make stable identity-level matching
particularly challenging.
Existing methods mainly address TI-ReID by learning joint representations to reduce the text–image modality gap. Early approaches
aligned entire images with sentences [2], while later works refined finegrained correspondences between body parts and textual phrases [3].

More recently, Vision-Language Models (VLMs) such as CLIP [4] and
ALBEF [5], pretrained with contrastive objectives [2,6], have provided
stronger multimodal representations and become the dominant backbone for TI-ReID. Nevertheless, retrieval is still typically performed
by directly matching a text query against individual gallery images,
leaving identity-level discrimination to emerge implicitly from limited
ReID supervision. Fig. 1 highlights an additional challenge behind
this paradigm: compared with the relatively balanced text–image distribution in Flickr30k, TI-ReID text embeddings in ICFG-PEDES are
noticeably more compact and clustered, partly because pedestrian descriptions are often highly templated and semantically similar across
identities. As a result, Query–Gallery Similarity (QG-Similarity) becomes more vulnerable to cross-identity ambiguity, suggesting that
reliable identity cues may need to be sought beyond text-side instance
matching alone.
In contrast, while text-side matching is prone to ambiguity, the
visual side of TI-ReID exhibits a clearer identity-level structure: images
of the same person form compact clusters, while different identities

∗ Corresponding author.
∗∗ Correspondence to: School of Computer Science and Engineering, Sun Yat-Sen University, Guangzhou 510006, China.

E-mail addresses: yangh529@mail2.sysu.edu.cn (H. Yang), zhangq689@mail.sysu.edu.cn (Q. Zhang), hujianfang@mail.sysu.edu.cn (J.-F. Hu),
stsljh@mail.sysu.edu.cn (J. Lai).
https://doi.org/10.1016/j.patcog.2026.113705
Received 5 January 2026; Received in revised form 10 April 2026; Accepted 10 April 2026
Available online 21 April 2026
0031-3203/© 2026 Published by Elsevier Ltd.

Pattern Recognition 179 (2026) 113705

H. Yang et al.

Fig. 1. Feature distribution in CLIP embedding space for text–image retrieval and TI-ReID, with TI-ReID text embeddings more compact due to highly templated
pedestrian descriptions.

Fig. 2. Identity-level matching is more reliable than instance-level matching for TI-ReID under the modality gap. The prototype-driven scheme aggregates
gallery images into identity prototypes and uses them as a prior to correct instance-level misranking, yielding a consistent ordering of text–image distances:
𝐷𝑞𝑔+ + 𝜆𝐷𝑞𝑝+ < 𝐷𝑞𝑔− + 𝜆𝐷𝑞𝑝− .

are well separated. As shown in Fig. 2, a text query 𝑞 is surrounded
by multiple positive and negative instances (𝑔 + , 𝑔 − ) on the hypersphere, and instance-level matching can be misled by hard positives
or negatives (𝐷𝑞𝑔 − < 𝐷𝑞𝑔 + ). Aggregating gallery images into identity
prototypes (𝑝+ , 𝑝− ) captures dominant identity-level patterns and reduces sensitivity to outliers. These cues are embedded in the gallery
itself, allowing extraction without modifying the pretrained backbone
and enabling training-free calibration. We define Query–Prototype Similarity (QP-Similarity) to measure the affinity between a query and
cluster-based prototypes across the gallery. Fused with QG-Similarity,
this hybrid similarity provides more reliable, semantically coherent
rankings aligned with the identity-aware nature of TI-ReID.
Based on this observation, we propose a Prototype-Driven Enhancement (PDE) module and build a fully training-free, plug-and-play
framework around it to refine retrieval results purely at inference. PDE
combines QP-Similarity and QG-Similarity to correct mismatched rankings without modifying pretrained backbones. Its cascaded extension,
PDE𝑀 , performs iterative refinement by progressively adjusting prototype granularity and applying stage-wise decayed fusion. Furthermore,
an Adaptive Cluster Estimation (ACE) mechanism is introduced to
automatically determine the number of prototypes from gallery density
via kernel density estimation (KDE) and peak detection, enabling robust
and scalable deployment. As demonstrated in Fig. 3, this prototypedriven paradigm seamlessly integrates with existing VLM-based TI-ReID

models and improves ranking consistency across datasets. Extensive
experiments on CUHK-PEDES, ICFG-PEDES, and RSTPReid datasets
show that our framework consistently outperforms state-of-the-art TIReID baselines under different backbones, confirming the effectiveness
and robustness of the proposed inference-time refinement strategy.
The main contributions of this work are summarized as follows:
• We propose a plug-and-play, training-free framework for TI-ReID
that operates entirely at the inference stage, mitigating the modality gap and instance–identity mismatch without any additional
training data or parameter updates. The framework can be seamlessly attached to existing pretrained VLM-based TI-ReID models.
• We introduce a Prototype-Driven Enhancement (PDE) module
that augments instance-level Query–Gallery Similarity with
Query–Prototype Similarity, where identity prototypes are obtained via unsupervised clustering to aggregate identity-level
structure in the image space. On top of PDE, we further develop
PDE𝑀 , a multi-stage refinement scheme that progressively adjusts
prototype granularity, and an Adaptive Cluster Estimation (ACE)
mechanism that automatically selects the number of prototypes
from gallery density, yielding a robust and scalable inference-time
refinement pipeline.
• Extensive experiments on CUHK-PEDES, ICFG-PEDES, and RSTPReid demonstrate that the proposed framework consistently
2

Pattern Recognition 179 (2026) 113705

H. Yang et al.

Fig. 3. Our training-free framework for TI-ReID with cascaded Prototype-Driven Enhancement (PDE) modules. PDE exploits intrinsic clustering in gallery features
to form identity prototypes. ACE adaptively estimates the prototype count 𝑁 from gallery density, removing manual tuning. Iterative refinement strengthens the
final similarity 𝑆̂ during inference without additional training or explicit modality alignment.

achieves state-of-the-art performance, significantly improving retrieval accuracy and ranking stability under a unified inference paradigm and showing strong generalization across different
backbones and datasets.

relation-aware local feature learning [14], similarity-guided interaction
with mismatched feature emphasis [15], and disentangled concept representations for hierarchical part-level alignment [16], or by improving
the stability and transferability of learned representations, for example
through language-based semantic regularization across domains [17]
and domain-aligned adaptation to reduce the synthetic-to-real pretraining gap [18]. Collectively, these methods primarily advance TI-ReID by
shaping the representation space during training.
Alongside these approaches, TI-ReID has made significant strides
with the advent of VLMs [19], particularly those based on dual-encoder
architectures such as ALBEF [5] and CLIP [4]. RaSa [20] adopts ALBEF
with Relation-Aware Learning and Sensitivity-Aware Learning to mitigate modality-specific noise. APTM [21] proposes a dual-branch learning scheme with Attribute Prompt Learning and Text Matching Learning, optimizing both attribute reasoning and cross-modal alignment.
MARS [22] strengthens identity-level supervision with a masked autoencoder and attribute loss. IRRA [23] leverages CLIP’s dual-encoder
structure with Re-ID-specific constraints. RDE [24] improves robustness
to annotation noise with a dual embedding mechanism and softened
triplet alignment loss. AUL [25] estimates confidence using subjective logic and refines supervision quality, while masked modeling
enhances cross-modal interaction. Propot [26] leverages CLIP-derived
identity prototypes and conditional prompts for improved instancelevel alignment. PLOT [27] incorporates confidence-based filtering and
loss smoothing to address noisy supervision in contrastive training.
HAM [28] addresses generalization by simulating human annotation
variability during training.
Despite these advances, most existing TI-ReID methods still perform
retrieval at the instance level, in which a text query is directly matched
to each gallery image, and even prototype-based approaches [26]
mainly exploit prototypes to shape training signals rather than to support identity-level reasoning at inference. Combined with the modality
gap discussed above, such instance-level QG-Similarity is highly sensitive to outlier instances, leading to unstable identity-level matching.
In this work, we explicitly exploit the inherent identity-level structure of TI-ReID by constructing image-modality prototypes from the
gallery and using them as test-time priors to refine rankings, providing
an inference-time, identity-aware complement to existing VLM-based
methods.

2. Related work
2.1. Vision-Language Models in text-to-image retrieval
VLMs have significantly advanced text-to-image retrieval. CLIP [4]
uses contrastive learning on large-scale datasets, enabling zero-shot
retrieval across diverse tasks. ALBEF [5] refines feature alignment between text and image to improve retrieval performance. ALIGN [7] enhances retrieval robustness by capturing complex relationships through
large-scale pretraining. T5 [8] unifies text-to-text and image-to-text
transformations, boosting retrieval flexibility. BLIP [9] jointly learns
vision and language features through bootstrapping, improving both
visual understanding and text generation. These models enforce crossmodal constraints, aligning feature distributions between text and image, thereby enhancing the accuracy and efficiency of text-to-image
retrieval.
However, recent studies [10,11] have pointed out that VLMs still
suffer from inherent heterogeneous distribution gaps, with text and
image embeddings occupying distinct regions on the hypersphere. Further analysis [12,13] reveals that text features tend to form compact,
less discriminative clusters, whereas image features exhibit stronger
class-wise separability. These inherent modality gaps pose particular
challenges for TI-ReID, where identity-level discrimination is required
but only limited task-specific data are available. The disparity between
feature distributions hampers accurate text-to-identity matching and
makes instance-level similarity highly sensitive to cross-modal noise.
2.2. Text-to-image person re-identification
Early research on text-to-image person re-identification (TI-ReID)
mainly followed a supervised cross-modal representation learning
paradigm, where task-specific image and text encoders were jointly optimized to reduce modality discrepancies and enhance fine-grained correspondence. Within this paradigm, representative methods mainly improve TI-ReID either by strengthening fine-grained cross-modal alignment through local or concept-level correspondence learning, such as
3


codex
有一篇文件名里包含成对单引号，我的前两次 shell 引号没有精确命中。先不跳过它，我会用 `ls -b` 确认原始文件名后再抽取。
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
/bin/zsh -lc "ls -b | sed -n '1,12p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc "pdftotext -l 3 'ADA framework for unsupervised domain adaptation person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 171 (2026) 112238

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

ADA framework for unsupervised domain adaptation person
re-identiﬁcation
Wei Zhang a , Peijun Ye b , Dihu Chen c , Tao Su a,∗
School of Electronics and Information Technology, Sun Yat-sen University, Guangzhou Higher Education Mega Center, Guangzhou, China
Peng Cheng Laboratory, No. 2 Xingke 1st Street,Nanshan District, Shenzhen, China
c School of Integrated Circuits, Sun Yat-sen University, NO 66 Gongchang Road, Guangming District, Shenzhen, China
a

b

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Domain adaptation
Sparse self-attention

a b s t r a c t
Domain shift remains a critical barrier for generalizing person re-identiﬁcation (ReID) models across datasets.
To address this challenge, we present a sparse self-Attention augmented Domain Adaptation (ADA) framework
that learns domain-invariant identity features through three key innovations: (1) Sandwich Attention Primitive
(SAP), a novel computational unit designed to boost primitive-level domain adaptation. (2) Sparse self-Attention
Augmented Bottleneck block (SAAB block), a hierarchical block integrating SAP to enhance adaptation at the
architecture level. (3) Scalable Design, if necessary, SAAB block can be ﬂexibly cascaded to construct taskspeciﬁc ADA framework. Experiments on three benchmarks validate ADA’s superiority: (1) Achieves state-of-theart performance across domains (e.g., 16.5 % mAP gain on CUHK03→Market-1501). (2) Demonstrates consistent
generalizability and adaptability.

1. Introduction
Unsupervised Domain Adaptive Person Re-Identiﬁcation (UDA ReID) transfers recognition from labeled source to unlabeled target
domains via domain-invariant identity features. This task addresses
cross-camera or cross-domain identity matching without target domain
annotations, making it a critical technology for intelligent missing person search and smart city surveillance systems. Consequently, UDA ReID has garnered signiﬁcant research interest due to its practical applicability and technical challenges.
The domain shift in UDA Re-ID primarily stems from dataset-speciﬁc
variations in pedestrian appearance. Key factors include: (1) Illumination discrepancies: Lighting condition diﬀerences across domains alter
visual features. (2) Viewpoint variations: Camera perspectives aﬀect
pose and occlusion patterns. (3) Background clutter: Irrelevant scene elements introduce noise to feature representations. (4) Camera-speciﬁc
biases: Sensor characteristics (e.g., resolution, color proﬁles) distort appearance consistency. To mitigate these challenges, learning domaininvariant pedestrian features-representations robust to dataset-speciﬁcbiases is essential for eﬀective cross-domain adaptation.
Prior studies on UDA Re-ID have primarily focused on three dominant paradigms: (1) Clustering-based adaptation, (2) Image style
translation, and (3) Feature-level alignment. While these methods
have advanced cross-domain adaptation research, they exhibit critical

limitations: (1) Feature-level alignment often relies on auxiliary annotations or information to align distributions, increasing implementation
complexity. (2) Image style translation suﬀers from unstable optimization due to its dependency on generative adversarial networks (GANs).
(3) Clustering-based methods are prone to error propagation from noisy
pseudo-labels and imbalanced cluster distributions.
To address these challenges, we present a ADA framework -a
lightweight, sparse self-Attention augmented Domain Adaptation framework that integrates sparse token interactions to learn domain-invariant
identity representations. Our ADA framework eliminates the need for
auxiliary supervision while achieving superior cross-domain generalization.
In brief, our ADA framework advances domain adaptation in person
re-identiﬁcation (ReID) through following innovations:
•

Sandwich Attention Primitive (SAP): We introduce a novel computational primitive(SAP) integrating a sandwich attention mechanism
to enhance domain adaptation. The SAP ﬂexibly adapts to sparse
self-attention mechanisms across heterogeneous design paradigms,
eﬀectively bridging domain shifts by synergizing the complementary strengths of convolutional inductive bias and sparse token
interactions.
• Sparse self-Attention Augmented Bottleneck block (SAAB block):
Building on SAP, we propose the SAAB block, a hierarchical

∗ Corresponding author.

E-mail addresses: yepj@pcl.ac.cn (P. Ye), stscdh@mail.sysu.edu.cn (D. Chen), sutao@mail.sysu.edu.cn (T. Su).
https://doi.org/10.1016/j.patcog.2025.112238
Received 18 October 2023; Received in revised form 18 April 2025; Accepted 5 August 2025
Available online 8 August 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

bottleneck structure that systematically improves domain-invariant
feature learning. If necessary, by cascading SAAB blocks, we construct a scalable sparse self-Attention augmented Domain Adaptation framework (ADA framework), enabling ﬂexible integration into
existing architectures.
• Robustness and Superiority: Extensive experiments on three ReID
benchmarks demonstrate that the ADA framework achieves consistent, generalizable, and adaptable performance across varying sparse
self-attention conﬁgurations. As a plug-and-play module, the ADA
framework enhances model accuracy at multiple deployment stages
while surpassing many SOTA methods in cross-domain scenarios.

In contrast to these approaches, our methodology circumvents multiple limitations through three key innovations: First, it operates without
requiring auxiliary annotations from either domain, reducing implementation complexity. Second, it eliminates dependence on GAN-generated
synthetic images, thereby avoiding quality constraints of image translation. Third, the proposed architecture demonstrates superior training
stability and faster convergence compared to clustering-based alternatives while maintaining deployment ﬂexibility.
2.2. Self-attention augmented convolution
The synergistic integration of self-attention with convolution enables
complementary feature learning, achieving performance enhancements
with moderate computational overhead. This fusion paradigm addresses
the intrinsic limitations of CNNs in modeling long-range dependencies
while preserving local feature extraction capabilities.
Self-attention augmented convolution addresses the inherent locality constraints of conventional convolution through long-range dependency modeling. AA-ResNet [10] pioneered a parallel implementation
where convolutional and self-attention operators process identical input feature maps simultaneously. The ﬁnal output is generated through
channel-wise concatenation of the convolution-derived local features
and self-attention-extracted global contextual features. BoTNet [11] enhanced this paradigm by replacing spatial convolutions in ResNet’s ﬁnal
bottleneck stage with attention modules, realizing measurable performance gains.
These approaches integrate standard self-attention mechanisms with
quadratic spatial complexity 𝑂((𝐻𝑊 )2 ) (where H and W denote feature map height and width) within supervised learning frameworks.
In contrast, our method introduces two key upgradation: (1) sparse
self-attention with reduced spatial complexity, and (2) unsupervised
domain adaptation deployment without backbone architecture modiﬁcation. The proposed modular design enables ﬂexible implementation
while eﬀectively bridging domain gaps in person re-identiﬁcation tasks.
Existing supervised methods universally require structural alterations to backbone networks for performance enhancement. Our framework diﬀerentiates itself by operating in unsupervised domain adaptation (UDA) scenarios, maintaining the original backbone architecture
while achieving comparable adaptability. This non-invasive integration
ensures deployment ﬂexibility without compromising target domain
generalization.

2. Related work
Prior research relevant to this work primarily encompasses three
technical directions in unsupervised domain adaptation (UDA) for
person re-identiﬁcation (Re-ID): self-attention augmented convolution, transformer-based person re-identiﬁcation, and domain adaptation
methodologies for person re-identiﬁcation. This section provides a concise review of these foundational approaches.
2.1. UDA methods for person re-identiﬁcation
Current UDA-ReID solutions primarily address domain gaps through
three principal paradigms mentioned earlier. Each methodology
presents distinct advantages and limitations as analyzed below:
Feature-Level Alignment. This category mitigates domain discrepancies through hierarchical feature alignment across source and target
domains. Through dynamic token matching and ﬂexible allocation of
local tokens to human joint points, POSR [1] propose pose-aligned feature learning module to obtain pose-aligned features. HUDA [2] implements joint global distribution alignment and local instance matching
to enhance knowledge transferability. Similarly, TJ-AIDL [3] employs a
dual-branch network architecture with identity inference space, leveraging both semantic labels and auxiliary attribute annotations from source
domains. While eﬀective, these methods often require supplementary
annotations (e.g., attribute labels), increasing implementation complexity and data acquisition costs.
Image Style Translation. GAN-based approaches attempt to bridge
domain gaps through cross-domain image synthesis. PTGAN [4] exempliﬁes this strategy by preserving source-domain identity attributes
while transferring target-domain background characteristics. PDA-Net
[5] learns domain invariant representation with disentangled pose
and domain information. However, such methods face two inherent
constraints: (1) The training instability characteristic of adversarial
networks often prolongs convergence time compared to conventional
CNNs; (2) The synthetic image quality remains fundamentally constrained by ﬁtting quality of GANs.
Clustering-Based Adaptation. Pseudo-labeling techniques typically
employ pre-trained models for feature extraction followed by clusterguided pseudo-label assignment. Recent advances address critical limitations in this paradigm: MGCN [6] learns noise-resistant features
via prototype-, triplet-, and list-wise consistency. 𝐶 3 𝐴𝐵 [7] introduces cluster-aware boundary adaptation to reﬁne pseudo-label reliability while mitigating class imbalance through its Cluster-Adaptive Balance (CAB) loss. The MLC framework [8] further enhances robustness
through multi-granularity feature learning, memory bank-based label
assignment, and noise-resistant self-paced clustering. The CMC framework [9] employs a graph convolutional network to construct a connection estimator, which facilitates the transfer of relational knowledge
from the source domain to the target domain. Nevertheless, clusteringbased approaches inherently suﬀer from two limitations: (1) Performance ceilings imposed by clustering algorithm precision; (2) Error
propagation from noisy pseudo-labels that exacerbate intra-cluster data
imbalance.

2.3. Transformer-based person re-identiﬁcation
Transformer-based person re-identiﬁcation (ReID) methods broadly
fall into two categories: (1) hybrid architectures combining CNNs and
Transformers, and (2) pure Transformer-based approaches. APD [12]
uses a CNN as a feature extractor, followed by a modiﬁed visual Transformer for Part Mask Generation and Part Merging. TMGF [13] leverages
the ﬁnal Transformer layer to split features in a dual-branch backbone
network, enabling multi-granularity feature learning.
While these methods advance ReID research by exploring hybrid
or multi-branch architectures, they signiﬁcantly increase computational
complexity due to dense matrix operations within backbone network. In
contrast, our approach retains the original backbone architecture and
requires only sparse self-attention operations, drastically reducing computational overhead.
3. Methods
3.1. Description of the overall workﬂow
The proposed domain adaptation framework operates in joint training mode through the following workﬂow: Labeled source domain images and unlabeled target domain samples are aggregated and partitioned into mini-batches containing equal proportions of both domains.
2

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Fig. 1. The ﬂowchart for the ADA framework: the ADA framework is deployed only once, and the eﬀective deployment location is behind any one of the stages
1/2/3 of the backbone network. The backbone is inserted a hybrid classiﬁer. The pseudo labels were generated after DBSCAN clustering and updated once at the
starting of each training epoch.

Each mini-batch is sequentially processed through our network pipeline,
which comprises:(1) A ResNet-50 backbone [14] (pre-trained on ImageNet, containing stages 0–4). (2) The ADA framework. (3) Cluster computation module. (4) Composite Re-ID loss 𝐿𝑅𝑒𝐼𝐷 .
As illustrated in Fig. 1, the Re-ID loss 𝐿𝑅𝑒𝐼𝐷 combines classiﬁcation
loss 𝐿𝑐𝑙𝑠 [15] and triplet loss 𝐿𝑡𝑟𝑖 [16]. The complete training loss is
formulated as:
𝐿𝑅𝑒𝐼𝐷 = 𝐿𝑐𝑙𝑠 + 𝐿𝑡𝑟𝑖

(1)

The ADA framework’s deployment requires balancing between computational overhead and performance optimization. Operating as a plugand-play module across both training and inference phases, the framework’s architecture permits ﬂexible scalability. In our implementation,
a simple conﬁguration is adopted with single SAAB block integration.
Note: (1)after stages 1/2/3 are all eﬀective deployment locations. This
design choice ensures computational eﬃciency while maintaining adaptation eﬀectiveness. (2) Inspired by the review [17,18], with the rapid
evolution of sparse self-attention mechanisms, it is possible to ﬂexibly
decide whether to stack SAAB blocks based on practical needs. Fig. 2
provides detailed architectural speciﬁcations.

Fig. 2. The diagram of the ADA framework: for simplicity, the BN and ReLU
operations are not drawn in the ﬁgure. In this article, stackable SAAB block is
only stacked once, and the performance improvement is signiﬁcant. Therefore,
its memory usage is relatively reasonable.

3.2. Introduction to the ADA framework

The SAP augments the capacity of domain adaptation through: (1)
Sparse Attention Eﬃciency: Implements Sparse complexity attention(<
𝑂(𝑁 2 )) versus conventional approaches(= 𝑂(𝑁 2 )). (2) Modular Flexibility: Permits plug-and-play integration of various sparse self-attention
variants. (3) Symmetrical Feature Encoding: Channel-wise symmetric
distributions emerge through bottleneck-residual coupling. Fig. 3 depicts the functional block diagram of SAP.

3.2.1. Introduction to the SAAB block and SAP
The ADA framework enhances cross-domain discriminative capability through domain-invariant identity representation learning.
Its core innovation lies in dual attention mechanisms: strong focus
on identity-salient features (pedestrian attributes) while suppressing
domain-speciﬁc distractors (background elements) via long-range dependency modeling. As joint training progresses, the framework demonstrates progressive domain adaptation through the SAAB block and the
computational primitive SAP.
To enhance domain adaptation performance, the SAAB block introduces the following key improvements over the bottleneck block structure: (1) Core Computational Primitive. The bottleneck block relies on
standard convolution, whereas the SAAB block replaces it with the SAP
(Sandwich Attention Primitive). (2) Bi-Level Bottleneck Structure. The
ﬁrst level is embedded via SAP. The second level is constructed by
the SAAB block based on SAP. (3) Expanded Receptive Field. In bottleneck block, the receptive ﬁeld is limited by a single 3 × 3 convolution. The SAAB block achieves broader contextual awareness through
self-attention mechanism. (4) Overﬁtting Integration. The SAAB block
incorporates dropout operations, which are absent in bottleneck block,
further improving robustness.

3.2.2. Algorithm description for the SAP
To rigorously characterize the SAP, we present its mathematical implementation through PyTorch-style pseudocode across ﬁve sparse selfattention variants(Linear Self-Attention [19], Permuted Self-Attention
[20], Focused Linear Attention [21], Bi-Level Routing Attention [22],
Window based Self-Attention [23]). The pseudocode description is presented in the following algorithmic listing.
#i n p u t f e a t u r e Xin a r e ( Cin , H, W) . o u t p u t f e a t u r e a r e
( Cout , H, W)
#2D c o n v o l u t i o n Conv2d_head , ( Cout , H, W)
x1 = Conv2d_head ( Xin )
#2D c o n v o l u t i o n Conv2d_down , ( Cin /4 ,H, W)
x = Conv2d_down ( Xin )
# p e r m u t e from ( Cin /4 ,H,W) t o (H,W, Cin /4)
3


exec
/bin/zsh -lc "pdftotext -l 3 'A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 180 (2026) 114100

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

A2HA: Attribute-aware hierarchical alignment for text–image person
re-identification
Ziyu Zhu a , Runmin Wang a ,∗, Han Xu a , Qiuju Dai a , Lingxin Cui a , Xingdong Song a ,
Congzhen Yu a , Xin Hu a , Changxin Gao b , Nong Sang b
a

School of Information Science and Engineering, Hunan Normal University, Changsha, 410081, Hunan, China

b School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan, 430074, Hubei, China

ARTICLE

INFO

Keywords:
Text–image person re-identification
Cross-modal alignment
Multi-scale alignment
Attribute-level enhancement
Semantic understanding

ABSTRACT
Text–Image Person Re-Identification (TI-ReID) aims to retrieve a target individual from an image gallery based
on auxiliary textual descriptions. Current approaches face two major limitations: (1) global matching methods
struggle to capture fine-grained attribute relations; and (2) textual descriptions are treated as a one-directional
auxiliary modality and are often dominated by global visual representations. To address these challenges,
we propose A2HA (Attribute-Aware Hierarchical Alignment), a framework designed for precise cross-modal
alignment through hierarchical optimization. Specifically, we present a dynamically weighted Hierarchical
Alignment (HA) framework that performs coarse-to-fine optimization across instance-, identity-, and attributelevel representations, enabling a unified modeling of global and fine-grained semantics. Rather than treating
objectives as an isolated combination of existing losses, this design leverages attribute-enriched benchmarks
to enable coordinated multi-level semantic alignment within a single integrated framework. To this end,
we propose a dual-dimensional and dual-directional attribute-pairing strategy, capturing semantic and modal
consistency while enabling bidirectional interaction between textual and visual attributes. Furthermore, we
extend three widely-used benchmarks by incorporating attribute-aware segmented images and annotations,
constructed with large language models and segmentation models, resulting in attribute-enriched versions of
CUHK-PEDES, ICFG-PEDES, and RSTPReid. Experimental results show that A2HA achieves improvements of
+5.02%, +4.68%, and +8.98% in mean Average Precision (mAP) on the three datasets, respectively. The code
will be released at https://github.com/ziyuzhu452-create/A2HA.

1. Introduction
Text–Image Person Re-Identification (TI-ReID) has demonstrated
significant research relevance in various security and surveillance applications [1]. In TI-ReID tasks, the model is designed to extract features from two distinct modalities and establish alignment between
them. The first challenge is bridging the modality gap to establish
unified representations. With the prevalence of Transformer [2], Vision
Transformer (ViT) [3], and Visual Language Pre-trained (VLP) models,
most current works [4–7] utilize these models to encode cross-modal information into a shared feature space to address this issue. The unified
feature space provides a gateway to alleviate modal disparities while
conducting global matching. The second challenge is alignment. Existing methods must not only bridge the modality gap but also address
the challenge of semantically aligning visual information with textual
descriptions [8,9]. As illustrated in Fig. 1(a), traditional methods [10]
that rely on global text features 𝐹𝑔𝑡 and global image features 𝐹𝑔𝑣

for alignment are inherently misleading. The global features primarily
capture coarse-grained semantics and tend to overlook fine-grained
attribute-level correspondences that are crucial for distinguishing visually similar identities. Consequently, the lack of explicit attribute-level
modeling weakens the semantic alignment between textual descriptions and corresponding visual parts, particularly in scenarios involving
subtle inter-person differences.
Although some attribute-aware methods [11–13] have considered
these issues, they have the following flaws: (1) Most of these works [12,
14] extract textual attributes and match them with global image features, which hinders their ability to achieve strong attribute-level alignment, as shown in Fig. 1(b). Noise information continues to play a
significant role in the recognition of specific attributes. This is mainly
because the semantic granularity of localized textual attributes is fundamentally mismatched with holistic visual representations, leading

∗ Corresponding author.

E-mail address: runminwang@hunnu.edu.cn (R. Wang).
https://doi.org/10.1016/j.patcog.2026.114100
Received 28 February 2026; Received in revised form 29 April 2026; Accepted 24 May 2026
Available online 29 May 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Fig. 1. A comparison of (a) traditional methods, (b) existing attribute-aware methods, and (c) our approach to aligning image–text features. 𝐹 𝑡 and 𝐹 𝑣 refer to
𝑡
𝑣
the textual and visual features. 𝐹𝑔𝑡 and 𝐹𝑔𝑣 stand for the global features of text and image. 𝐹𝑎𝑡𝑡𝑁
and 𝐹𝑎𝑡𝑡𝑁
represent the textual and visual attribute features of
the 𝑁𝑡ℎ attribute. GMP is the global max pooling.

• We propose A2HA, an attribute-aware hierarchical alignment
framework for TI-ReID, which jointly optimizes instance-level,
identity-level, and attribute-level objectives in a coarse-to-fine
manner, enabling balanced and fine-grained semantic alignment.
• To support effective attribute-level optimization, we identify and
formalize diverse attribute-level misalignment patterns in TIReID and introduce a dual-dimensional and dual-directional attribute pairing strategy, which provides explicit and balanced
cross-modal attribute supervision.
• We construct attribute-enriched versions of three public TI-ReID
benchmarks and demonstrate that A2HA achieves state-of-theart (SOTA) performance across multiple evaluation metrics on
CUHK-PEDES [17], ICFG-PEDES [18], and RSTPReid [19].

to ambiguous and noisy attribute-visual associations. (2) These methods [11,13] treat attributes as auxiliary rather than core alignment
targets and rely mostly on instance-level or identity-level constraints,
thus attribute information is implicitly optimized and easily overshadowed by dominant identity discrimination objectives. (3) Methods [11,
15,16] that segment both text and image attributes typically rely on
static attribute mapping or predefined structural priors about person attributes. Such designs assume fixed attribute importance across
samples and scenarios, ignoring the fact that different attributes contribute unequally under varying semantic and visual contexts. Without
systematic attribute-level optimization and dynamic weight adaptation, these methods struggle to flexibly emphasize discriminative attributes while suppressing less informative ones, resulting in rigid
and suboptimal attribute alignment. As a result, existing attributeaware approaches are limited in their ability to establish fine-grained
and balanced attribute-level alignment across modalities, limiting their
effectiveness in handling complex and subtle person distinctions.
To address the aforementioned challenges, we propose A2HA, an
attribute-aware hierarchical alignment framework built upon a VLP
backbone, which explicitly promotes fine-grained and balanced semantic alignment between textual descriptions and visual illustrations. At
the core of A2HA is a hierarchical alignment method that defines a unified coarse-to-fine optimization process across instance-, identity-, and
attribute-level representations. This multi-level formulation presents as
an explicitly coupled optimization scheme, where different semantic
levels are progressively refined under a shared alignment objective,
ensuring global identity consistency while progressively strengthening
attribute correspondence. As a result, A2HA enables robust cross-modal
alignment under complex and fine-grained scenarios.
To effectively support attribute-level alignment, we propose a dualdimensional and dual-directional attribute pairing strategy that defines
how image–text pairs are organized for optimization across both intramodal and inter-modal spaces. Serving as a fundamental basis for
contrastive and triplet-based objectives, this strategy goes beyond conventional instance-level pairing schemes based on global similarity
by explicitly modeling fine-grained attribute relations. It distinguishes
multiple attribute misalignment patterns, enabling the model to better identify hard false positives and false negatives at the attribute
level and to emphasize discriminative attributes during training. Furthermore, to provide reliable attribute-level supervision, we construct
attribute-enriched versions of three widely used TI-ReID benchmarks:
CUHK-PEDES [17], ICFG-PEDES [18], and RSTPReid [19]. By jointly
leveraging a powerful semantic segmentation model and a large language model, the resulting datasets offer precise and interpretable
attribute annotations, substantially reducing semantic ambiguity and
noise in cross-modal attribute alignment.
In summary, our key contributions are as follows:

2. Related works
2.1. Text–Image Person Re-Identification
Previous solutions for TI-ReID use ResNet [20] and LSTM [21]
as backbones for feature extraction, emphasizing semantic alignment
between textual tokens and visual patches. Li, et al. [17] introduces a
pioneering work with the release of the widely used CUHK-PEDES [17].
Subsequent studies [22,23] address limitations such as spatial insensitivity and an over-reliance on specific features by designing novel
loss functions and enhancing local dependency modeling. For instance,
Zhang, et al. [24] proposes Cross-Modal Projection Matching (CMPM)
and Classification (CMPC) losses to enhance discriminative feature
extraction. In recent years, with the prevalence of Vision Transformer
(ViT) [3] and BERT [25], most works adopt them as backbones. TransReID [26] is the first to apply ViT [3] to ReID, inspiring subsequent works [27–29] to explore its potential in both single and cross
modal person re-identification. In the field of text–image person reidentification, IVT [30] introduces two paradigms for implicit semantic
alignment, achieving finer and more accurate alignment between visual
and textual modalities. SDG [31] proposes a structure-level distribution guidance to calibrate instance-level feature bias which employs
inter-class and intra-class loss functions with an iterative optimization
mechanism. Meanwhile, TPAN [32] leverages aggregated text prototypes to guide the model to focus on the prime identity-related features
of target persons, mitigating the feature ambiguity caused by detection
misalignment. TP-LReID [33] proposes a novel lifelong learning method
to dynamically generate identity-consistent text prompts, enabling the
model to learn discriminative features across successive domains without catastrophic forgetting. Despite the effectiveness, these methods
still focus on the instance-level, overlooking the fine-grained alignment
among cross-modal attribute features.
2

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Fig. 2. The architecture of A2HA. The parameters in the segmentation model and LLM are frozen during training. All attribute sample pairs include both
𝑣
𝑡
inter-modal and intra-modal cases. 𝐹 𝑣 and 𝐹 𝑡 represent image and text features, while 𝐹𝑎𝑡𝑡
and 𝐹𝑎𝑡𝑡
are representation of attribute features for both image and
text inputs.

pre-trained semantic segmentation model Grounded SAM [39], both
with frozen parameters, to obtain attribute descriptions and segmenta𝑣
tion masks. In the visual branch, attribute-specific visual features 𝐹𝑎𝑡𝑡𝑞
𝑣
are obtained by fusing the global visual feature 𝐹𝑔 with the corresponding attribute mask under a ratio 𝑟. In parallel, textual attribute
descriptions are encoded to generate attribute-level textual features
𝑡 . Based on these attribute features, we employ a dual-dimensional
𝐹𝑎𝑡𝑡𝑞
and dual-directional attribute pairing strategy to categorize attribute
sample pairs across intra-modal and inter-modal spaces. The resulting
attribute pairs are then optimized through the Attribute-Level Alignment(ALA) module, which performs contrastive learning over diverse
attribute relations. Through the joint optimization of GLA and ALA,
A2HA achieves hierarchical alignment across instance, identity, and
attribute levels.

2.2. Attribute-aware methods
Numerous works [34,35] perform cross-modal alignment at the
global level. For example, IRRA [36] matches global image and text
representations based on feature similarity. To alleviate this limitation, several attribute-aware methods have been proposed [11,12,14].
ViTAA [11] pioneers attribute-level segmentation across visual and
textual modalities, but relies on NLKT-based noun phrase extraction,
which may fragment descriptive semantics, and adopts static attributevisual mappings that assume fixed attribute importance. APTM [13]
extracts attribute nouns to construct prompt-based representations;
however, decoupling attributes from their original textual context disrupts syntactic coherence and leads to misalignment when matching
attribute descriptions with global visual features. VGSG [15] further
explores attribute-aware alignment, yet still employs static attribute
mappings and performs alignment at a single semantic level, limiting
its ability to capture hierarchical and fine-grained attribute correspondence. BAMG [16] models attribute relations via graph convolution, but depends on predefined human structural priors, which may
constrain flexibility under diverse visual conditions.

3.2. Feature extraction
A2HA employs CLIP-ViT/B-16 [37] as image encoder and
BERT [25] as text encoder. In ViT-B/16 [3], the letter ‘B’ denotes the
base network scale, while ‘16’ indicates the use of a 16 × 16 pixel patch
size.
Image Encoder. The input image 𝐼 ∈ R𝐻×𝑊 ×𝐶 is divided into 𝑁
patches and passed through a linear projection layer with a learnable
position embedding 𝑃𝐸 . After concatenation with a [CLS] token, 𝐼 is
represented as 𝑉 ∈ R(𝑁+1)×𝑑 , where 𝑑 denotes the embedding dimension. The [CLS] token encodes the global information of all 𝑁 patches.
{ 𝑣
}
𝑣 , where
The extracted features are tokenized into 𝐹 𝑣 = 𝑓𝑐𝑙𝑠
, 𝑓1𝑣 , … , 𝑓𝑁
𝑣
𝑣
𝑓𝑖 represents the feature of the 𝑖𝑡ℎ patch and 𝑓𝑐𝑙𝑠 serves as the global
representation 𝐹𝑔𝑣 of the image.
Text Encoder. Following a similar approach as the image encoder,
the input text 𝑇 is tokenized by a token embedding layer. The textual
token is obtained through lower-cased Byte Pair Encoding (BPE). Starting with a [SOS] token and ending{with an [EOS] token,
we obtain
}
𝑡 , 𝑓 𝑡, … , 𝑓 𝑡
a sequence represented as 𝐹 𝑡 = 𝑓SOS
.
Additionally,
1
EOS
since we have adopted BERT as the backbone, we incorporate masked
language modeling to improve prediction accuracy.

3. Methodology
3.1. Overview
As illustrated in Fig. 2, A2HA follows an attribute-aware hierarchical alignment paradigm, consisting of a Global-Level Alignment
(GLA) branch and an Attribute-Level Alignment (ALA) branch. The two
branches collaboratively enforce coarse-to-fine cross-modal semantic
alignment. Given an input image–text pair, we adopt CLIP-ViT/B16 [37] as the backbone to extract visual and textual features, denoted
as 𝐹 𝑣 and 𝐹 𝑡 . The corresponding global representations 𝐹𝑔𝑣 and 𝐹𝑔𝑡
are first utilized to perform Global-Level Alignment(GLA), ensuring
instance-level and identity-level consistency across modalities. To enable fine-grained attribute-level alignment, the input texts and images
are further processed by a large language model ChatGPT [38] and the
3


codex
现在还剩一篇可见红外的文件名需要用通配符绕过引号问题；另外 9 篇的正文信息已经足够覆盖文本、遮挡、换衣、3D、终身和域适应几个套路。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'pdftotext -l 3 ./*Knowledge*experience*visible-infrared*.pdf -' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112553

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

‘Knowledge and experience’ for visible-infrared person re-identiﬁcation
Nianchang Huang a,b , Jialiang Wang a,b , Qiang Zhang

a,b,∗, Jungong Han c , Jin Huang a

a State Key Laboratory of Electromechanical Integrated Manufacturing of High-Performance Electronic Equipment, Xidian University, Xi’an, 710071, Shaanxi, China
b

Center for Complex Systems, School of Mechano-Electronic Engineering, Xidian University, Xi’an, 710071, Shaanxi, China

c Department of Automation, Tsinghua University, Beijing, 100084, China

a r t i c l e

i n f o

Keywords:
Cross-modality person re-identiﬁcation
‘Knowledge and experience’
Modality-invariant person information

a b s t r a c t
Most existing models mainly focus on learning those modality-shared features unique to individuals for crossmodality Visible-Infrared Person Re-Identiﬁcation (VI-ReID). Diﬀerently, the human visual systems identify different persons by using not only the information within the input images but also their learned knowledge and
experience in history. Speciﬁcally, given a query image, our visual systems will ﬁrst extract lots of information
about the person from the given image and then deduce more information about this person with the aid of our
knowledge and experience for searching this person from a set of images. Inspired by that, we will propose a
novel Cross-modality Knowledge and Experience Transformer (CKT) for cross-modality VI-ReID in this paper.
In detail, our CKT will ﬁrst memorize the discriminative modality-shared information of each identity in the
training set, and then convert them into our model’s ‘knowledge and experience’. Accordingly, in addition to the
information within those input images, our CKT can infer more modality-invariant person information from its
‘knowledge and experience’ according to their corresponding modality-shared information for VI-ReID. Comprehensive experiments show that our proposed model achieves signiﬁcant performance improvements over most
state-of-the-art ones by combining the information within the input images and their induced information from
our model’s ‘knowledge and experience’.

1. Introduction
Person Re-Identiﬁcation (ReID) aims to match the given identity in
a query image from a set of gallery images captured by disjoint surveillance camera views [1]. Based on the assumption that the query and
gallery images are captured under good lighting conditions, most existing methods [2] mainly match a visible image from a visible image
based gallery set, i.e., VV-ReID. Although they have achieved great success, these VV-ReID methods usually rely on bright lighting conditions,
thus restricting their applicability in real-life settings. To overcome this
limitation, many surveillance systems now extensively incorporate thermal infrared (IR) cameras with RGB cameras to capture valuable information in low-light environments. As a result of that, cross-modality
Visible-Infrared (VI-) ReID has gained a growing interest recently.
Compared with VV-ReID, the major challenges of VI-ReID is the
modality discrepancy caused by the intrinsically distinct imaging mechanisms of RGB and IR cameras [3–5], which will lead to heterogenous
distributions between single-modality RGB and IR features. The major
idea of addressing such an issue is to extract those common features (i.e.,
modality-shared features) within RGB and IR images for identifying dif-

ferent individuals. However, solely relying on such a limited amount of
modality-shared person information between RGB and IR images may
hinder VI-ReID systems to reach the next milestone.
Alternatively, we will overcome the cross-modality VI-ReID issue by
imitating some characteristics of the human visual system. Speciﬁcally,
if a person wants to tell whether the identities in two images are the
same or not, he (she) uses not only the information obtained from the
two images but also his (her) learned knowledge and experience in the
past. For example, as shown in Fig. 1, when we judge whether person
A in the RGB image and person B in the IR image are the same person
or not, we may ﬁrst extract their modality-invariant person attributes
from the corresponding images, e.g., person A has thin arms and a thin
neck, but person B has a thick neck and a large waist. Then, given these
attributes extracted from input images, alongside our knowledge and
experience, we may infer more modality-invariant attributes about the
two persons in other views or poses, e.g., person A may have a round
face and a slim ﬁgure, and person B may have a stubby neck and a plump
body. By fusing these extracted and inferred person attributes, we will
make our ﬁnal judgment. In this process, besides the person information
extracted from the input images, our knowledge and experience also

∗ Corresponding author.

E-mail addresses: huangnianchang@xidian.edu.cn (N. Huang), Jialiang_Wang@stu.xidian.edu.cn (J. Wang), qzhang@xidian.edu.cn (Q. Zhang),
jungonghan77@gmail.com (J. Han), jhuang@mail.xidian.edu.cn (J. Huang).
https://doi.org/10.1016/j.patcog.2025.112553
Received 30 November 2023; Received in revised form 19 August 2025; Accepted 3 October 2025
Available online 8 October 2025
0031-3203/© 2025 Published by Elsevier Ltd.

Pattern Recognition 172 (2026) 112553

N. Huang et al.

Fig. 1. General process of identifying diﬀerent identities in the human visual system.

play important roles, since they provide more modality-invariant person
information for better identifying diﬀerent persons.
Inspired by that, a novel Cross-modality Knowledge and Experience
Transformer (CKT) will be designed in this paper for VI-ReID, which will
employ its ‘knowledge and experience’ to infer more modality-invariant
information for VI-ReID, in addition to some information extracted from
the input images. Speciﬁcally, as the human visual systems, which ﬁrst
capture information from the given images, our proposed CKT will also
ﬁrst extract those discriminative modality-shared features from the images of diﬀerent modalities by employing a cross-modality VI-ReID subnetwork. On top of that, our proposed CKT will imitate the human visual
system to infer more modality-invariant information for VI-ReID by using its ‘knowledge and experience’.
To this end, we observe that the information within all the training
images not only contains rich discriminative person features/attributes
of a speciﬁc person identity, but also contains the relations among person features/attributes across diﬀerent person identities. This means
that we may infer more information about this person from some existing information by virtue of those learned relations among diﬀerent
person features/attributes. For example, if a person looks overweight in
the given image, it may be inferred that the person probably carries a
heavy belly rather than a ﬂat stomach by exploring the relations among
the features from diﬀerent identities, i.e. such relations may be obtained
by comparing those features from a set of overweight persons and those
from a set of thin persons. Based on such an observation, a novel Knowledge and Experience Transformer (KET) will be further designed in our
proposed CKT, which will transfer the information within the training
set into our VI-ReID model’s ‘knowledge and experience’ and employ
them for inferring more modality-invariant person information based
on the features from the given images as that in the human visual system.
More speciﬁcally, the KET will ﬁrst save the modality-shared feature center of each identity in the training set as our model’s memory,
just like a person’s memory. Here, each modality-shared feature center can represent some unique characteristics of its corresponding identity. Accordingly, the memory features can well save the information of
diﬀerent identities in the training set. Then, a knowledge module will
be designed in the proposed KET for establishing its ‘knowledge and
experience’ by exploring the interactions across the feature centers of
diﬀerent identities. Similar to that in the human visual system, these
‘knowledge and experience’ will enable our model to infer more hidden modality-invariant person information, given such modality-shared
features extracted from the input images. Therefore, the proposed KET
will further design a novel reasoning module, which will utilize the information extracted from the images of one identity for inferring more

modality-shared information about the identity from the ‘knowledge
and experience’ of our proposed CKT. By doing so, our proposed model
can combine the modality-shared features from the input images and
those inferred modality-shared features from its ‘knowledge and experience’ for VI-ReID in the testing stage.
The main contributions of our work are as follows:
(1) Inspired by the human visual system, we endue our proposed VIReID model with the ability to learn its ‘knowledge and experience’ by
exploring the relations among all the identities in the training set. By
virtue of the learned ‘knowledge and experience’, our proposed model
can infer more modality-shared information, thereby helping to reach
new performance milestone for VI-ReID models.
(2) We propose a novel Cross-modality Knowledge and Experience Transformer (CKT) to implement the above idea. Speciﬁcally, the
modality-shared information extracted from each individual in the training set is memorized and transferred into our model’s ‘knowledge and
experience’ via a novel Transformer to infer more modality-invariant
information for VI-ReID during the testing stage.
(3) Extensive experiments prove that our proposed model achieves
a substantial improvement over existing algorithms on the widely-used
SYSU-MM01 and datasets
In the following contents, we will ﬁrst provide a brief overview for
previous research on ReID and VI-ReID, respectively, in Section 2. Then,
we will elaborate our proposed model’s design. After that, we report
several experiments in Section 4 to validate our proposed model. Finally,
we provide a brief conclusion for this study.
2. Related work
2.1. Single-modality VV-ReID
Earlier solutions of VV-ReID mainly rely on some hand-crafted features and probabilistic patch matching algorithms [6]. Recently, with
the remarkable achievements in computer vision, Deep Convolutional
Neural Network (DCNN) based VV-ReID models have taken over as the
mainstream. Generally speaking, there are currently two major categories of DCNN based VV-ReID models: representation learning based
models [7] and metric learning based models [2]. Representation learning based models [7] handle resolution, lighting, view or pose changes
by extracting some invariant features from the inputs. For instance, in
[7], a Pose-Guided Feature Alignment (PGFA) model was introduced
for solving the occluded ReID problem by utilizing pose landmarks to
extract useful information and reduce occlusion noise. Metric learningbased models [2] aim to learn discriminative metric spaces for ReID by
minimizing the intra-person distance and maximizing the inter-person
2

Pattern Recognition 172 (2026) 112553

N. Huang et al.

3. Proposed model

distance. For instance, considering that optimizing metric losses in a
hand-designed way are sub-optimal and time-consuming, Gu et al. [2]
proposed an AutoML-based methods, termed LFS-ReID, for searching
optimal margin-based softmax loss function. Eventually, their optimized
metric loss achieved signiﬁcant performance.

3.1. Problem formulation
In this section, we take the RGB-to-IR matching as an example to theoretically compare existing VI-ReID models with our proposed model.
𝐾

𝑔
Suppose that 𝑞 is a query image of RGB modality and 𝐺 = {𝑔𝑖 }𝑖=1
denotes the gallery set with a set of IR images. Existing VI-ReID models
will ﬁrst extract their features 𝐅𝑞 and 𝐅𝑔𝑖 from these input images via a
feature extractor 𝑓 (∗), i.e.,

2.2. Cross-modality VI-ReID
VI-ReID has gained increasing interest in research recently, but summarizing the extensive existing research on VI-ReID models is beyond
the scope of this paper. We recommend interested readers to recent surveys on VI-ReID [8] for further introduction.
Generally speaking, there are two major solutions to VI-ReID, i.e.,
modality-shared feature learning and modality-speciﬁc feature compensation. Modality-shared feature learning [9,10] based models aim to
transfer the features extracted from the input image of diﬀerent modalities into the same feature space for extracting their discriminative
modality-shared person features. For example, considering that shape
information is one of the important modality-invariant features, [10]
proposed a multi-task learning-based model that leverages person segmentation to extract modality-invariant shape information for VI-ReID.
Diﬀerently, [9] proposed to exploit the relations among diﬀerent person
parts to complement those modality-shared appearance features, achieving great performance improvements.
Diﬀerently, modality-speciﬁc feature compensation based models
[11–13] ﬁrst generate the missing speciﬁc information from the existing
ones and then use those generated information together with the original images for VI-ReID. For example, [11] ﬁrst analyzed the drawbacks
of existing models and proposed to optimize their image generator in the
network structures and their ReID network in cross-modal feature fusion
strategies. Accordingly, a new two-stage GAN based model is designed,
which achieves new state-of-the-art performance.
Compared with modality-speciﬁc feature compensation based models, modality-shared feature learning based ones are usually more
concise, thus obtaining more attentions [14,15]. However, existing
modality-shared feature learning based models only focus on exploring the features within the input images, ignoring the fact that human
visual systems also employ those information inferred from their ‘knowledge and experience’. Diﬀerently, as that in human visual systems, our
proposed model can not only extract discriminative modality-shared information from the input images but also infer more modality-shared
information from its ‘knowledge and experience’, thus achieving better
results.

𝐅𝑞 = 𝑓 (𝑞, 𝛼), 𝐅𝑔𝑖 = 𝑓 (𝑔𝑖 , 𝛼),

(1)

where 𝛼 denotes the parameters of the feature extractor 𝑓 (∗). Then, they
will achieve person ReID by optimizing the following objective function

𝑖∗ = 𝑎𝑟𝑔𝑚𝑎𝑥𝑖=1,2,…,𝐾𝑔 𝑆𝑖𝑚(𝐅𝑞 , 𝐅𝑔𝑖 ).

(2)

Here, 𝑖∗ denotes that 𝑔𝑖∗ is the matched image. 𝑆𝑖𝑚(∗) denotes a metric
function.
Diﬀerently, our proposed model will use its ‘knowledge and experience’ to infer more discriminative information for VI-ReID. Here, we
denote the ‘knowledge and experience’ of our proposed model by 𝐊𝐄,
which can be seen as a set of person features/attributes for better explanation. Speciﬁcally, after extracting the features 𝐅𝑞 or 𝐅𝑔𝑖 , our proposed
model will further infer more information about the identity from our
model’s ‘knowledge and experience’ according to 𝐅𝑞 or 𝐅𝑔𝑖 by

𝐅𝐾𝑞 = 𝑓𝑧 (𝐅𝑞 , 𝐊𝐄, 𝛽), 𝐅𝐾𝑔𝑖 = 𝑓𝑧 (𝐅𝑔𝑖 , 𝐊𝐄, 𝛽),

(3)

where 𝑓𝑧 (∗, 𝛽) denotes an inference function with its parameters 𝛽. After
that, our proposed model will further fuse those inferred features and
the features extracted from the input images, i.e.,

̂
𝐅𝑞 = ℎ(𝐅𝑞 , 𝐅𝐾𝑞 , 𝛾), ̂
𝐅𝑔𝑖 = ℎ(𝐅𝑔𝑖 , 𝐅𝐾𝑔𝑖 , 𝛾),

(4)

where ℎ(∗, 𝛾) denotes a fusion function with its parameters 𝛾. Accordingly, our proposed model will achieve person ReID via optimizing the
following objective function

𝑖∗ = 𝑎𝑟𝑔𝑚𝑎𝑥𝑖=1,2,…,𝐾𝑔 𝑆𝑖𝑚(̂
𝐅𝑞 , ̂
𝐅𝑔𝑖 ).

(5)

Accordingly, Fig. 2 illustrates the diagram of our proposed Crossmodality Knowledge and Experience Transformer (CKT), which mainly
consists of a VI-ReID subnetwork (corresponding to 𝑓 (∗)), a Knowledge
and Experience Transformer (corresponding to 𝑓𝑧 (∗)) and a fusion module (corresponding to ℎ(∗)).
Speciﬁcally, the VI-ReID subnetwork begins by extracting modalityshared features from input RGB or IR images via two modality-speciﬁc

Fig. 2. Illustration of the proposed model.
3


codex
只基于每篇前三页摘要和引言反推，不代表全文细节。

**逐篇反推**

1. `‘Knowledge and experience’ for visible-infrared person re-identification.pdf`  
创新类型：新机制。  
能发点：把 VI-ReID 的 gap 从“只学 RGB/IR 共享特征”改成“共享信息本身不够，需要从训练身份关系中推断隐藏的模态不变信息”。证据链是身份中心记忆、Transformer 推理、输入特征和推理特征融合，再做 SYSU-MM01、RegDB 一类标准对照。  
Story 套路：人不是只看当前图，而是用过去经验补全判断，模型也应有“经验记忆”。

2. `3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf`  
创新类型：新机制加新模态。  
能发点：把视频 ReID 的失败源具体拆成帧级错位、局部遮挡、外观混淆，认为时间聚合救不了低质量帧特征。用 SMPL/3D mesh 转点云和 pillar 特征，再做 2D-3D 融合与 3D 形状时序建模。  
Story 套路：人在 3D 空间里，2D 视频的错位和遮挡要靠 3D 几何补上。

3. `4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf`  
创新类型：新机制。  
能发点：不是泛泛做跨模态对齐，而是指出 RGB 和 IR 的频谱分布不同，固定卷积核不适合两种模态。证据链是 FFT 频谱差异、输入自适应频率滤波、低中高频分解对齐、三数据集消融。  
Story 套路：跨模态差异不只在空间外观里，也藏在频谱响应里。

4. `A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf`  
创新类型：新机制，偏工程组合。  
能发点：换衣 ReID 不再只靠轮廓和体型，而是用人体语义分割定位衣服和非衣服区域，做语义注意力和衣服遮蔽。证据链是四个换衣数据集上对 Swin-T、MBUNet 等大幅提升。  
Story 套路：换衣场景里，衣服是干扰源，要主动遮掉，让模型看不依赖衣服的语义区域。

5. `A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf`  
创新类型：工程组合，但 gap 包装较清楚。  
能发点：指出随机擦除和固定遮挡模拟不真实，真实遮挡有语义、形状、位置和上下文。方法用语义引导遮挡模拟，再扩展关键局部区域的邻近语义，并加跨中心偏移损失。  
Story 套路：不是让模型见更多随机洞，而是让它见更像真实世界的语义遮挡。

6. `A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf`  
创新类型：新数据设定和综述。  
能发点：3D skeleton ReID 被定义成独立方向，强调轻量、隐私、视角鲁棒、背景无关。证据链是手工特征、序列建模、图建模三类范式和基准评测。  
Story 套路：当外观不可靠时，身体结构和运动模式可以成为独立身份线索。

7. `A training-free framework for text-to-image person re-identification via query-prototype matching.pdf`  
创新类型：问题重定义加新机制。  
能发点：把 TI-ReID 的核心矛盾定义成“实例级相似度和身份级检索目标不一致”。方法不训练模型，而是从 gallery 聚类出身份原型，用 query-prototype 相似度修正 query-gallery 排序。  
Story 套路：要找的是人，不是某一张图，所以检索也应按身份结构校准。

8. `A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf`  
创新类型：新机制加新数据设定。  
能发点：指出全局图文对齐太粗，已有属性方法又常把文本属性和整图特征硬配。它用实例、身份、属性三级对齐，并用 LLM 和分割模型构造属性增强版数据集。  
Story 套路：文本描述天然是属性级的，图文 ReID 应该在属性层面对齐，而不是只做整图对齐。

9. `ADA framework for unsupervised domain adaptation person re-identification.pdf`  
创新类型：工程组合偏多。  
能发点：把 UDA ReID 的缺陷归结为伪标签噪声、GAN 不稳、辅助标注复杂，提出稀疏自注意力增强的可插拔适配块。证据链是跨域 benchmark 上提升和部署位置消融。  
Story 套路：用轻量稀疏长程关系替代重型生成或伪标签依赖。

**针对团队资产的强候选**

1. Headline：把 aerial-ground ReID 重定义为“3D 可见表面匹配”，而不是普通跨域匹配。  
挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
区别：最像 `3D-Aided...`，但它解决视频帧错位和遮挡，我们解决航拍和地面之间的极端视角可见面不一致。方法核心不是简单加 3D 特征，而是把 2D 特征投到 SMPL 规范表面，再按可见性做跨视角匹配。  
低成本首验：冻结 SOLIDER，在 CARGO 或 AG-ReID.v2 小子集上用现有 SMPL 投影做表面分区池化，对比全局特征、水平条带、2D pose part。若规范表面池化不能稳定提高跨视角正负间隔，先杀。

2. Headline：用 3D 几何记忆补全航拍视角缺失的人体证据。  
挂靠资产：aerial-ground、SMPL、SOLIDER-Swin。  
区别：最像 `Knowledge and experience`，但它记忆训练身份中心来补 VI 模态不变信息；我们不做身份记忆，而是学习“某个 3D 可见区域在另一视角通常对应什么稳定证据”。重点是视角造成的缺失证据补全，不是 RGB/IR 模态补偿。  
低成本首验：冻结特征后，只训练一个很小的 part-to-part 预测头，用地面视角的可见部位特征预测航拍不可见或弱可见部位特征。若预测补全后的跨视角同身份相似度没有明显提升，杀。

3. Headline：做几何条件的频谱可靠性建模，让模型知道航拍低清下哪些频率还能信。  
挂靠资产：aerial-ground、SOLIDER-Swin、pose 热图门控，必要时加 SMPL 可见性。  
区别：最像 `4SNet`，但它是 RGB/IR 的模态频谱差异；我们是航拍和地面在视角、尺度、低清下的频谱可靠性差异。机制不是固定低频对齐，而是由姿态、视角和可见性决定每个身体区域该信低频结构还是中高频纹理。  
低成本首验：把图像或特征分成低中高频，在 aerial-ground 配对上统计各频段的同身份和异身份分离度，并按俯视角、人体面积、SMPL 可见率分组。若没有稳定的频段可靠性规律，杀。

4. Headline：用 3D 视角投影生成真实的自遮挡课程，而不是随机遮挡增广。  
挂靠资产：SMPL、pose 热图门控、aerial-ground。  
区别：最像 `FOSENet`，但它模拟外部物体遮挡；我们模拟航拍和地面跨视角天然产生的自遮挡、截断和部位压缩。卖点是“跨视角可见性课程”，不是普通遮挡增强。  
低成本首验：用 SMPL 或 pose 估计得到真实 aerial-ground 的部位可见分布，再把地面图按该分布合成航拍式缺失，做冻结特征检索或很短训练。若合成分布和真实航拍错误模式对不上，杀。

我的排序是：1 最强，2 次强，3 有机制新意但需要先证明确有频谱规律，4 最便宜但容易被审稿人看成增强方法，除非把“视角诱导自遮挡”这个问题定义讲得很硬。
tokens used
79,253
只基于每篇前三页摘要和引言反推，不代表全文细节。

**逐篇反推**

1. `‘Knowledge and experience’ for visible-infrared person re-identification.pdf`  
创新类型：新机制。  
能发点：把 VI-ReID 的 gap 从“只学 RGB/IR 共享特征”改成“共享信息本身不够，需要从训练身份关系中推断隐藏的模态不变信息”。证据链是身份中心记忆、Transformer 推理、输入特征和推理特征融合，再做 SYSU-MM01、RegDB 一类标准对照。  
Story 套路：人不是只看当前图，而是用过去经验补全判断，模型也应有“经验记忆”。

2. `3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf`  
创新类型：新机制加新模态。  
能发点：把视频 ReID 的失败源具体拆成帧级错位、局部遮挡、外观混淆，认为时间聚合救不了低质量帧特征。用 SMPL/3D mesh 转点云和 pillar 特征，再做 2D-3D 融合与 3D 形状时序建模。  
Story 套路：人在 3D 空间里，2D 视频的错位和遮挡要靠 3D 几何补上。

3. `4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf`  
创新类型：新机制。  
能发点：不是泛泛做跨模态对齐，而是指出 RGB 和 IR 的频谱分布不同，固定卷积核不适合两种模态。证据链是 FFT 频谱差异、输入自适应频率滤波、低中高频分解对齐、三数据集消融。  
Story 套路：跨模态差异不只在空间外观里，也藏在频谱响应里。

4. `A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf`  
创新类型：新机制，偏工程组合。  
能发点：换衣 ReID 不再只靠轮廓和体型，而是用人体语义分割定位衣服和非衣服区域，做语义注意力和衣服遮蔽。证据链是四个换衣数据集上对 Swin-T、MBUNet 等大幅提升。  
Story 套路：换衣场景里，衣服是干扰源，要主动遮掉，让模型看不依赖衣服的语义区域。

5. `A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf`  
创新类型：工程组合，但 gap 包装较清楚。  
能发点：指出随机擦除和固定遮挡模拟不真实，真实遮挡有语义、形状、位置和上下文。方法用语义引导遮挡模拟，再扩展关键局部区域的邻近语义，并加跨中心偏移损失。  
Story 套路：不是让模型见更多随机洞，而是让它见更像真实世界的语义遮挡。

6. `A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf`  
创新类型：新数据设定和综述。  
能发点：3D skeleton ReID 被定义成独立方向，强调轻量、隐私、视角鲁棒、背景无关。证据链是手工特征、序列建模、图建模三类范式和基准评测。  
Story 套路：当外观不可靠时，身体结构和运动模式可以成为独立身份线索。

7. `A training-free framework for text-to-image person re-identification via query-prototype matching.pdf`  
创新类型：问题重定义加新机制。  
能发点：把 TI-ReID 的核心矛盾定义成“实例级相似度和身份级检索目标不一致”。方法不训练模型，而是从 gallery 聚类出身份原型，用 query-prototype 相似度修正 query-gallery 排序。  
Story 套路：要找的是人，不是某一张图，所以检索也应按身份结构校准。

8. `A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf`  
创新类型：新机制加新数据设定。  
能发点：指出全局图文对齐太粗，已有属性方法又常把文本属性和整图特征硬配。它用实例、身份、属性三级对齐，并用 LLM 和分割模型构造属性增强版数据集。  
Story 套路：文本描述天然是属性级的，图文 ReID 应该在属性层面对齐，而不是只做整图对齐。

9. `ADA framework for unsupervised domain adaptation person re-identification.pdf`  
创新类型：工程组合偏多。  
能发点：把 UDA ReID 的缺陷归结为伪标签噪声、GAN 不稳、辅助标注复杂，提出稀疏自注意力增强的可插拔适配块。证据链是跨域 benchmark 上提升和部署位置消融。  
Story 套路：用轻量稀疏长程关系替代重型生成或伪标签依赖。

**针对团队资产的强候选**

1. Headline：把 aerial-ground ReID 重定义为“3D 可见表面匹配”，而不是普通跨域匹配。  
挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
区别：最像 `3D-Aided...`，但它解决视频帧错位和遮挡，我们解决航拍和地面之间的极端视角可见面不一致。方法核心不是简单加 3D 特征，而是把 2D 特征投到 SMPL 规范表面，再按可见性做跨视角匹配。  
低成本首验：冻结 SOLIDER，在 CARGO 或 AG-ReID.v2 小子集上用现有 SMPL 投影做表面分区池化，对比全局特征、水平条带、2D pose part。若规范表面池化不能稳定提高跨视角正负间隔，先杀。

2. Headline：用 3D 几何记忆补全航拍视角缺失的人体证据。  
挂靠资产：aerial-ground、SMPL、SOLIDER-Swin。  
区别：最像 `Knowledge and experience`，但它记忆训练身份中心来补 VI 模态不变信息；我们不做身份记忆，而是学习“某个 3D 可见区域在另一视角通常对应什么稳定证据”。重点是视角造成的缺失证据补全，不是 RGB/IR 模态补偿。  
低成本首验：冻结特征后，只训练一个很小的 part-to-part 预测头，用地面视角的可见部位特征预测航拍不可见或弱可见部位特征。若预测补全后的跨视角同身份相似度没有明显提升，杀。

3. Headline：做几何条件的频谱可靠性建模，让模型知道航拍低清下哪些频率还能信。  
挂靠资产：aerial-ground、SOLIDER-Swin、pose 热图门控，必要时加 SMPL 可见性。  
区别：最像 `4SNet`，但它是 RGB/IR 的模态频谱差异；我们是航拍和地面在视角、尺度、低清下的频谱可靠性差异。机制不是固定低频对齐，而是由姿态、视角和可见性决定每个身体区域该信低频结构还是中高频纹理。  
低成本首验：把图像或特征分成低中高频，在 aerial-ground 配对上统计各频段的同身份和异身份分离度，并按俯视角、人体面积、SMPL 可见率分组。若没有稳定的频段可靠性规律，杀。

4. Headline：用 3D 视角投影生成真实的自遮挡课程，而不是随机遮挡增广。  
挂靠资产：SMPL、pose 热图门控、aerial-ground。  
区别：最像 `FOSENet`，但它模拟外部物体遮挡；我们模拟航拍和地面跨视角天然产生的自遮挡、截断和部位压缩。卖点是“跨视角可见性课程”，不是普通遮挡增强。  
低成本首验：用 SMPL 或 pose 估计得到真实 aerial-ground 的部位可见分布，再把地面图按该分布合成航拍式缺失，做冻结特征检索或很短训练。若合成分布和真实航拍错误模式对不上，杀。

我的排序是：1 最强，2 次强，3 有机制新意但需要先证明确有频谱规律，4 最便宜但容易被审稿人看成增强方法，除非把“视角诱导自遮挡”这个问题定义讲得很硬。
e precision (mAP) and rank-1.
The remainder of this article is organized as follows.
Section II introduces the related work, and Section III
describes the proposed SAVS method. Section IV describes the
experimental settings and the analysis of the results. Section V
presents the details of the ablation study, and concluding
remarks are presented in Section VI.
II. R ELATED W ORK
To date, many person ReID approaches have been proposed.
According to the person’s visual appearance, these methods
can be roughly divided into clothing-consistent person ReID
and cloth-changing person ReID. In the following, we will
separately introduce them.
A. Clothing-Consistent Person ReID
In earlier times, people [31], [37], [38] made efforts to
develop related methods for clothing-consistent person ReID
where the visual appearance of the clothes was consistent
for the same person. For example, Sun et al. [9] proposed

Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

codex
前四篇已经读到足够信息，另外几篇的输出里有截断，我现在只对被截断的论文做定向抽取，重点抓摘要、问题陈述、贡献段和引言里的 gap。
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
/bin/zsh -lc 'pdftotext -l 3 "3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf" - | rg -n -C 3 "Abstract|However|To mitigate|Specifically|contributions|main contributions|we propose|we introduce|challenge|misalignment|occlusion|appearance confusion|3D"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2-
3-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025
4-
5:3D-Aided Pedestrian Representation Learning for
6-Video-Based Person Re-Identification
7-Guquan Jing , Graduate Student Member, IEEE, Peng Gao , Graduate Student Member, IEEE, Yujian Lee,
8-Yiyang Hu, and Hui Zhang , Senior Member, IEEE
9-
10:Abstract—Video-based person re-identification (Re-ID) aims
11-to match the target pedestrian from video sequences. Recent
12-methods perform frame-level feature extraction followed by
13:temporal aggregation to obtain video representations. However,
14-they pay insufficient attention to the quality of frame-level
15:features, which suffer from issues including multi-frame misalignment, partial occlusion and appearance confusion. People
16:live in a 3D space. 3D pedestrian representations can provide
17-rich geometric information and shape cues that offer promising
18:solutions to these challenges in video-based Re-ID. To mitigate these issues, this paper proposes a 3D-Aid Pedestrian
19:Representation Learning (3DAPRL) network, which introduces
20:3D modality to video-based Re-ID. Specifically, two novel modules
21-are designed, i.e., the Cross-Modal Fusion (CMF) module and
22-the Shape-aware Spatial-Temporal Interaction (SSTI) module,
23-to enhance pedestrian representation learning. The CMF module generates discriminative fusion representations by utilizing
24:3D pedestrian data, while the SSTI module learns spatialtemporal 3D shape representation which are distinguishable for
25-finding the target pedestrian in video scenarios. Both features
26-generated from the CMF and SSTI modules contribute to
27-the final video representation. Extensive experiments on four
28-challenging video-based Re-ID datasets demonstrate that our
29:3DAPRL network reaches better performance than state-of-thearts methods.
30:Index Terms—Video-based person re-identification, crossmodal feature fusion, 3D pedestrian representation, video
31-representation.
32-
33-I. I NTRODUCTION
--
67-information.
68-Early video Re-ID methods primarily adopt models
69-designed for video recognition (including RNNs [12], [13],
70:[14], [15], 3D CNNs [16], [17], [18], optical flow [19], [20],
71-etc.) to learn video temporal information directly, which are
72-not well-suited for video Re-ID tasks. Recent studies [9],
73-[21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31],
74-[32], [33], [34] have made significant progress by proposing
75-novel modules to aggregate extracted frame-level features
76-temporally; however, they fail to adequately consider the
77:quality of the frame-level features. Multi-frame misalignment,
78:partial occlusion and appearance confusion frequently occur
79-in video data, compromising frame-level features and causing
80-the generation of suboptimal video representations. In specific,
81-some of these methods [35], [36], [37], [38], [39] retain the
--
83-such as those that account for the importance of each feature’s
84-spatial position [39]. Attention mechanisms or local pooling
85-are used to highlight relevant parts of the pedestrian in these
86:methods. However, partial occlusion introduces spatial noise,
87-which integrates irrelevant information into frame-level features and degrades the overall performance of these methods.
88:In addition, spatial misalignment across frames often yields
89-uneven feature quality, which in turn can induce temporal
90:misalignment during feature aggregation. Other methods [21],
91-[31], [32], [33] attempt to derive global representations by
92-pooling frame-level features [9]; some [31] also refine framelevel features using these global cues. Nevertheless, their
93:performance is significantly undermined by the aforementioned multi-frame misalignment and partial occlusion. These
94-issues can corrupt global features and make it difficult to
95-spatially emphasize the target pedestrian, leading to a loss of
96:discriminative spatial information and causing temporal misalignment. Meanwhile, both methods struggle with appearance
97-confusion, where negative samples with similar appearance
98-are misidentified as the target pedestrian, ultimately leading to non-discriminative video representations. Figure 1 (a)
99:shows these challenges in video Re-ID, where two pedestrians with similar appearance are affected by multi-frame
100-
101-1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
102-similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
103-See https://www.ieee.org/publications/rights/index.html for more information.
104-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
105-
106:JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID
107-
108-Fig. 1. (a) Two pedestrian sequences from video Re-ID datasets exhibiting
109:similar appearance, with challenges including multi-frame misalignment (red
110:box) and partial occlusion (blue box). (b) An example of our 3D pedestrian representation including mesh, point cloud and colored point cloud.
111-(c) Overview of our proposed framework. Note that in (c), black lines/boxes
112-denote the basic structure of the previous methods. On this basis, red
113-lines/boxes represent our additional processes.
114-
115:misalignment (red boxes) and partial occlusion (blue boxes). In
116-such scenarios, effective spatial emphasis is hindered directly,
117-leading to discriminative spatial information loss and temporal
118:misalignment.
119-The inherent limitations of relying solely on 2D image to
120:overcome these issues and other challenges in Re-ID motivate the exploration of auxiliary information. Recent studies
121-incorporate various modalities (such as text [30], events [28]
122:and 3D data [40], [41], [42], [43], [44], etc.). Among these,
123:3D data, such as meshes rendered by models like SMPL [45],
124-is particularly promising. It offers rich geometric information
125-and shape cues, which are valuable for extracting detailed and
126-robust pedestrian features. Motivated by this, Zheng et al. [40]
127:propose a 3D pedestrian feature generation module that utilizes
128:both 3D geometric information from reconstructed meshes and
129:2D appearance cues from images to enhance feature representation. While the extracted 3D feature contributes to pedestrian
130-representation, their methods lack sufficient consideration for
131-the effective fusion of cross-modal information, limiting the
132:potential of fused features. Specifically, their fusion strategies
133:project RGB information directly into 3D space [40] or simply
134:concatenate 2D image features with 3D features [41], [42],
135-leading to coarse feature aggregation and limited cross-modal
136:interaction. We consider that the main challenge in feature
137:fusion for Re-ID with 3D representations lies in the modality
138:gap between 2D images and 3D data, causing inconsistency
139-in cross-modal features. Such a feature inconsistency are
140:manifested in feature misalignment across dimensions and
141-the disparate information captured from different modalities.
142:On the one hand, the dimensional misalignment between
143:2D and 3D pedestrian features leads to the loss of crucial
144-information during the fusion process. On the other hand,
145:3D pedestrian feature could provide incomplete pedestrian
146-representation due to the coarse extraction process, resulting
147-in inconsistency with 2D features and yielding suboptimal
148:fusion outcomes. Some methods [41], [42] leverage the 3D
149-shape parameter included with SMPL model to obtain spatialtemporal pedestrian shape cues using attention mechanisms,
150-but only limited and coarse-grained information is extracted.
151-
152-12831
153-
154:In addition, although they utilize 3D information in video
155-cloth-changing Re-ID (VCCReID) scenarios, its potential in
156-video Re-ID benchmark remains underexplored.
157:To address the aforementioned problems, we propose a
158:novel 3D-Aid Pedestrian Representation Learning (3DAPRL)
159:network to incoperate 3D modality into video-based Re-ID
160:(see Figure 1 (b) for an example of the 3D pedestrian representation and (c) for a conceptual illustration of our framework).
161-Our network exploit a Cross-modal Fusion (CMF) module and a Shape-aware Spatial-Temporal Interaction (SSTI)
162-module to obtain accurate frame-level fusion representations
163:and spatial-temporal 3D shape representations, respectively.
164:Before that, we first convert 3D meshes to point clouds, and
165-then employ a pillar-based backbone network [46] to extract
166-pillar-based features (i.e., pseudo-image features), effectively
167:reducing the feature inconsistency between 2D and 3D modalities. The generated features contain pedestrian geometry,
168-appearance, and shape information, while being dimensionally aligned with image features. Besides, the pillar-based
169-backbone network captures the complete pedestrian geometry
170:in 3D space, enabling a comprehensive feature representation. Consequently, this step narrows the modality gap and
171-establishes a more consistent feature space for subsequent
172-processing stages in our framework. After obtaining pillarbased features, the CMF and SSTI modules are employed to
173:overcome challenges in video Re-ID. Detailed explanations of
174-these modules are provided below.
175-The CMF module is introduced to effectively mitigate the
176:impacts of multi-frame misalignment and partial occlusion.
177-It achieves this by adaptively fusing the pillar-based features
178-described above with image features. This fusion process
179-emphasizes crucial areas of the target pedestrian and integrates
180-essential geometric and shape cues. In detail, the fusion
181-strategy exploits deformable cross-attention [47] to learn
182:correlations between image and 3D modalities, generating
183:3D-enhanced features. Accurate pedestrian positions derived
184-from the mapping relationship between 2D images and
185:3D point clouds serve as reference points, guiding the model
186-to focus on pedestrian-relevant areas and improving crossmodal feature alignment in the latent space. Additionally, the
187-CMF module adaptively determines relevant regions for fusion
188-and capture the local context of pedestrians based on offsets
189-learned from pillar-based features. Moreover, we retain the
190:generated pillar-based feature for preserving 3D geometric
191-and shape cues. Thus, the proposed CMF module generates discriminative fusion pedestrian representations that fuses
192:frame-level features with 3D geometric and shape information,
193:mitigating the effects of multi-frame misalignment and partial
194:occlusion.
195-To further enhance model ability against these issues, and
196:in particular to address the challenge of appearance confusion,
197-it is necessary to utilize a new stable pedestrian characteristic,
198:i.e., video-level 3D shape. The pedestrian shape in 3D space
199-offers robustness against similar appearances [41], [42] and
200-provides a complementary shape representation. Our Shapeaware Spatial-Temporal Interaction (SSTI) module is therefore
201:proposed to effectively learn and leverage these discriminative 3D shape representations of the target pedestrian from
202-
203-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
204-
--
206-
207-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025
208-
209:video sequences for improved identification. Specifically, the
210:3D shape parameters estimated by the SMPL model are
211-used as the initial input. Subsequently, we create temporal
212-and spatial streams in parallel to adopt feature contextualization along space and time axes the attention. The spatial
213:stream focuses on obtaining 3D shape information within each
214:frame, while the temporal stream models 3D shape variations
215-across frames. The extracted spatial and temporal features
216-are then fused to produce spatial-temporal representation.
217-Our SSTI module allows the network to fully exploit both
218-spatial and temporal dependencies. Its receptive field captures
219:spatial-temporal interactions. Hence, discriminative videolevel 3D shape representations are yielded, which contributes
220-to the final video representation. Extensive experiments on
221-four challenging video Re-ID datasets demonstrate that the
222:3DAPRL outperforms the state-of-the-art methods.
223:In summary, the contributions of this study are as follows:
224:• We propose a novel 3D-Aid Pedestrian Representation
225:Learning (3DAPRL) network that introduces 3D modality
226-to video-based Re-ID. To the best of our knowledge, this
227:is the first method to leverage 3D modality for addressing
228-video-based Re-ID (video Re-ID) task.
229-• We propose a Cross-modal Fusion (CMF) module to
230-fuse image features and pillar-based features. In the
--
232-to focus on the crucial area while providing pedestrian
233-geometric information and shape cues. Consequently, the
234-discriminative frame-level pedestrian feature is generated
235:to reduce the influence of the multi-frame misalignment
236:and partial occlusion.
237-• We design a Shape-aware Spatial-Temporal Interaction
238-(SSTI) module to extract distinctive spatial-temporal
239:3D shape representations, which contributes to the video
240-representation. It serves as a new pedestrian characteristic
241:to further mitigate the impact of appearance confusion,
242:multi-frame misalignment and partial occlusion.
243-• Extensive experiments on four widely used video Re-ID
244-datasets including MARS, iLIDS-VID, PRID-2011 and
245-DukeMTMC-VID demonstrate that our proposed method
--
247-II. R ELATED W ORKS
248-Related works are organized as follows. In Section II-A, we
249-provide a summary of recent video Re-ID methods and discuss
250:their limitations. Section II-B offers a detailed review of existing methods that incorporate 3D modality for person Re-ID.
251-We also emphasize the novelty of our approach and highlight
252-how it differs from these methods. Eventually, methods in the
253:3D feature learning are introduced in Section II-C.
254-A. Video-Based Person Re-ID
255-Remarkable progress has been made in video-based Re-ID
256-in recent years. Early studies exploit deep learning techniques
257-for video recognition task including optical flow [19], [20],
258:3D convolutions [16], [17], [48], [49] and recurrent models
259-[12], [13], [14], [15], [50] to obtain temporal features directly.
260-Chung et al. [19] proposed a Siamese network with two
261-
262-separately streams to extract spatial and temporal features.
263:Specifically, one stream processes RGB images while the other
264-processes optical flow. McLaughlin et al. [12] employ CNNs
265-to extract image features, followed by RNNs for temporal
266:information extraction. To tackle the spatial misalignment
267:issue in methods using RNNs, Liao et al. [48] apply 3D convolutions to extract and aggregate spatial and temporal features.
268:Li et al. [49] propose a multi-scale 3D convolutional layer,
269-which offers a more compact and efficient solution compared
270:to traditional 3D convolutions. Despite effectively encoding
271-temporal information, the techniques used in these methods
272-are not specifically designed for video Re-ID. They also face
273:challenges such as multi-frame misalignment, partial occlusion, and appearance confusion in video datasets, which often
274-lead to performance degradation. Most of methods [1], [20],
275-[21], [22], [24], [27], [29], [31], [36], [38], [39], [51], [52],
276-[53], [54], [55], [56], [57], [58], [59], [60] focus on aggregate
--
280-attention to extract useful information from visible body parts.
281-Wang et al. [21] introduced a pyramid structure that aggregates
282-spatial and temporal features by utilizing attention-based modules designed to capture spatial and temporal information. In
283:order to address the spatial misalignment in video-level data,
284-Chen et al. [27] propose an attention mechanism to interact
285-each frame with others in the video, enhancing both local
286:and global features. Despite the effective temporal aggregation of frame-level features using attention, challenges such
287:as multi-frame misalignment, partial occlusion, and appearance confusion still affect model performance. To mitigate
288-these issues, some methods incorporate additional modalities
289:such as event data [28] to provide supplementary information. However, the limited information provided by event
290:data is insufficient to resolve these challenges. To overcome
291:these limitations, we introduce a novel network incorporating
292:3D modality.
293:B. Person Re-ID With 3D Pedestrian Representation
294-Different from single-modality Re-ID, cross-modality
295-Re-ID [28], [40], [41], [43], [61], [62] focuses on retrieving
296-pedestrians using diverse modality representations, such as
297:3D pedestrian mesh. In the context of person Re-ID with
298:3D representation, existing approaches [40], [41], [43], [61],
299:[62], [63] primarily concentrate on extracting distinctive 3D
300:features. The incorporation of 3D modality is shown to
301-enhance the Re-ID performance by providing complementary
302:3D information, which positively impacts retrieval accuracy.
303:To obtain distinguishable 3D pedestrian features, some
304:methods use the SMPL model [45] to generate 3D representation. OGNet [40] first employs 3D representation for image
305:Re-ID by converting 2D pedestrian images into 3D space via
306:off-the-shelf 3D human reconstruction networks [64], [65],
307-[66]. It leverages a GNN structure to build the location
308:topology and extracts multi-scale semantic information of 3D
309:pedestrians. Liu et al. [62] reconstruct 3D clothed pedestrians
310-from images and learn the distinguished feature from the
311:3D clothed body, naked body and RGB information for
312-
313-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
314-

exec
/bin/zsh -lc 'pdftotext -l 3 "A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf" - | rg -n -C 3 "Abstract|Introduction|challenge|prospects|future|3D skeleton|privacy|view|background|taxonomy|survey|applications"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc 'pdftotext -l 3 "4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf" - | rg -n -C 3 "Abstract|However|Specifically|contributions|Contribution|we propose|we introduce|frequency|spectrum|low|medium|high|modality"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5-Joint NTU-UBC Research Centre of Excellence in Active Living for the Elderly (LILY), NTU, Singapore
6-{haocong001, ascymiao}@ntu.edu.sg
7-
8:Person re-identification via 3D skeletons is an important emerging research area that attracts increasing attention within the pattern recognition community. With distinctive advantages across various
9:application scenarios, numerous 3D skeleton based
10-person re-identification (SRID) methods with diverse skeleton modeling and learning paradigms
11:have been proposed in recent years. In this paper, we provide a comprehensive review and analysis of recent SRID advances. First of all, we
12:define the SRID task and provide an overview
13-of its origin and major advancements. Secondly,
14:we formulate a systematic taxonomy that organizes existing methods into three categories centered on hand-crafted, sequence-based, and graphbased modeling. Then, we elaborate on the representative models along these three types with an
15:illustration of foundational mechanisms. Meanwhile, we provide an overview of mainstream supervised, self-supervised, and unsupervised SRID
16-learning paradigms and corresponding common
17-methods. A thorough evaluation of state-of-theart SRID methods is further conducted over various
18-types of benchmarks and protocols to compare their
19:effectiveness, efficiency, and key properties. Finally, we present the key challenges and prospects
20:to advance future research, and highlight interdisciplinary applications of SRID with a case study. A
21-curated collection of valuable resources is available
22-at https://github.com/Kali-Hac/3D-SRID-Survey.
23-
24-1
25-
26:Introduction
27-
28-Person re-identification (re-ID) is an essential pattern recognition task of matching and retrieving a person-of-interest
29:across different views or scenes, which has been widely applied to security authentication, smart surveillance, healthcare, and embodied AI [Nambiar et al., 2019; Ye et al.,
30-2021]. Recent economical and precise skeleton-tracking devices (e.g., Kinect [Shotton et al., 2011]) have simplified
31:the acquisition of 3D skeleton data, enabling them to be a
32-prevalent and versatile data modality for gait analysis and
33-
34-Hand-Crafted
--
56-
57-arXiv:2401.15296v3 [cs.CV] 4 Jan 2026
58-
59:Abstract
60-
61:Figure 1: Overview of 3D skeleton based person re-ID (SRID) task
62-with hand-crafted, sequence-based or graph-based modeling to learn
63-effective body and motion features for identity recognition.
64-
--
66-based person Re-ID (SRID) models typically exploit bodystructure features and motion patterns (e.g., gait [Murray et
67-al., 1964]) from 3D positions of key body joints to identify
68-different persons. With unique merits such as small input
69:data, light-weight models, privacy-preserving without using
70:appearances, and robustness against view and background
71-variations [Han et al., 2017], SRID has attracted surging attention from both academia and industry [Rao et al., 2022].
72-In recent years, research on SRID has gained significant momentum, leading to diverse skeleton modeling and
73-learning paradigms. Early endeavors [Barbosa et al., 2012;
--
76-attributes of body. As these methods often require domain
77-expertise such as anatomy and kinematics [Yoo et al., 2002]
78-for skeleton modeling, they lack the ability to fully mine latent high-level features beyond human cognition. To resolve
79:this challenge, recent mainstream methods [Liao et al., 2020;
80-Huynh-The et al., 2020; Rao et al., 2022; Rashmi and Guddeti, 2022] leverage deep neural networks to automatically
81-perform skeleton representation learning for SRID. One of
82-exemplar methods (termed “sequence-based modeling”) is to
--
167-
168-(b) Model Accuracy & Efficiency Comparison
169-
170:Figure 2: (a) Overview of research origin and technical advancements of SRID within the person re-ID community (Zoom in and follow the
171:timeline for the best view). (b) Parameter sizes (Millions (M)), computational complexity (Giga Floating Point Operations (GFLOPs)), and
172-KS20 Rank-1 accuracy of state-of-the-art deep learning methods for SRID (Red: Sequence-based models; Green: Graph-based models).
173-
174-thereby possibly overlooking some valuable skeleton patterns. Another paradigm (termed “graph-based modeling”)
175:mitigates this challenge by constructing skeleton graphs to
176-model discriminative structural and actional features based on
177-the interrelations of body parts [Rao and Miao, 2023]. This
178-often requires multi-granularity body modeling and efficient
179-relational reasoning mechanisms (e.g., collaborative learning) based on skeleton graphs. Despite the great progress of
180-SRID, this rapidly evolving technique still lacks a systematic
181:review, making it difficult for researchers to gain a holistic
182:view of this field and embark on new research endeavors.
183:In light of this, we present the first survey on SRID, elucidating recent advancements of skeleton modeling, learning
184:paradigms, evaluation benchmarks, current challenges, and
185:interdisciplinary applications. Firstly, we define the SRID
186:task and provide a milestone overview to illustrate the origin
187:and key advancements of SRID as shown in Fig. 2a. Secondly, we propose a systematic taxonomy of SRID methods
188-to categorize them into hand-crafted, sequence-based, and
189-graph-based modeling, and elaborate on their foundational
190-mechanisms and representative approaches. We also illustrate the basic definitions and common methods within three
--
192-unsupervised). Thirdly, we introduce existing public benchmarks, evaluation metrics, and protocols for SRID, while
193-comprehensively evaluating state-of-the-art methods across
194-different benchmarks to compare their performance and efficiency. Meanwhile, we conduct qualitative analysis of different methods to compare their key properties with a discussion of advantages and disadvantages. Finally, we discuss the
195:current challenges in SRID and identify potential directions
196:for future research. An overview of promising SRID applications in interdisciplinary areas, spanning healthcare, embodied AI, and security, is further provided and illustrated with
197:a case study. The structure of this survey, including skeleton
198-modeling (Sec. 3), learning paradigms (Sec. 4), benchmarks
199:and evaluation (Sec. 5), challenges and prospects (Sec. 6) is
200:shown in Fig. 3. We hope our survey can bring new insights
201:to researchers and expedite future research in SRID.
202-
203-2
204-
205:Overview
206-
207-2.1
208-
209-Task Description
210-
211:As illustrated in Fig. 1, the input of SRID task is a 3D skeleton sequence that belongs to a certain pedestrian, and the
212-output is the predicted identity. Formally, we denote a 3D
213-skeleton sequence as S = (s1 , · · · , sf ) ∈ Rf ×j×3 , where
214-st ∈ Rj×3 denotes the tth skeleton with 3D coordinates of
--
227-i
228-i=1
229-quences of different persons collected from different scenes
230:or views. The task target is to learn a hand-crafted, sequencebased or graph-based model (detailed in Sec. 3) that maps
231:3D skeleton sequences into effective representations, so that
232-we can query the correct identity of an encoded skeleton sequence representation in the probe set via matching it with
233-the sequence representations in the database (i.e., gallery set).
234-SRID is essentially a retrieving and matching problem.
--
328-
329-Importance Inference
330-
331:Figure 3: Structure of this survey with the taxonomy of SRID research. Representative branches and SRID methods are listed.
332-
333-GCN [Chen et al., 2021]), MLP models (e.g., SimMC [Rao
334-and Miao, 2022]), and hybrid/composite models (e.g., SMSGE [Rao et al., 2021a]) (compared in Table 2).
--
337-
338-Taxonomy of SRID Methods
339-
340:As shown in Fig. 3, we propose a systematic taxonomy
341-for SRID approaches based on skeleton modeling (see Sec.
342-3) and learning paradigms (see Sec. 4). For skeleton
343-modeling, we divide them into three categories, including

 succeeded in 0ms:
35-Identity-shared patterns
36-
37-ABSTRACT
38:Visible-Infrared Person Re-identification (VI-ReID) is a cross-modality retrieval task that involves searching
39:for a specific pedestrian across different modalities (visible or infrared). Existing methods have explored crossmodality alignment from different perspectives, including spatial-domain cues and frequency-domain modeling.
40:Nevertheless, how to adaptively capture modality-specific frequency characteristics and effectively integrate
41:them into cross-modality feature alignment remains an open challenge. To address this challenge, this work
42-presents a novel and efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet) for the VI-ReID
43-task, which incorporates an Adaptive Frequency Filter (AFF) module and a Spectrum Synergy Module (SSM)
44:to extract robust cross-modality-invariant features. Specifically, in contrast to traditional fixed convolutionkernel operations, the AFF module reconstructs a set of learnable frequency bases by adopting a dual spatialand channel attention-guided strategy, which dynamically synthesizes a customized Spectrum filter for each
45:input. Complementarily, the SSM decomposes features into multiple frequency bands, which explicitly decouple
46:identity-shared patterns from modality-specific traits and synergistically enhance them, thereby achieving
47:consistent feature representation across cross-modal pedestrian images at the cross-modality feature alignment
48-level. Extensive experiments have been conducted on the public VI-ReID datasets (SYSU-MM01, LLCM, and
49-RegDB), demonstrating that our proposed method achieves competitive performance when compared to stateof-the-art approaches. Source code implementing 4SNet is publicly available at: https://github.com/dekusaklasdroid/4SNet.
50-
51-1. Introduction
52:Visible-Infrared Person Re-identification (VI-ReID) is a crossmodality retrieval task that aims to search for images of the same pedestrian across a non-overlapping, distributed camera network, which
53-captures visible images during the daytime and infrared images during
54:nighttime or low-light conditions [1,2]. Due to its cross-perspective
55-search trait, VI-ReID has found widespread applications in criminal
56-investigations, target tracking, and robotic systems [3,4], among other
57-application domains [5,6]. It has also emerged as a significant academic
58-concern in fields like pattern recognition and artificial intelligence [7,
59:8]. However, the large differences between daytime and low-light
60-
61-nighttime conditions, coupled with the constant variations in the
62-appearance and clothing of the same individual across modalities,
63-have exacerbated the difficulty and challenges of this modeling task,
64-remaining as an unresolved task yet [9,10].
65-Existing VI-ReID methods can be roughly classified into two categories: (1) image-level based methods, and (2) feature-level based
66:methods. Specifically, image-based methods resort to generative models, such as Generative Adversarial Networks (GANs) [11], and Diffusion models [12], to transform images from one modality to another.
67:Alternatively, some of them generate an intermediate auxiliary embedding modality that lies between the distributions corresponding to
68-
69-∗ Corresponding authors.
70-
--
95-strategy [21,22]. One notable example is DEEN [23], which employs a
96-center-guided pair mining strategy to create diverse auxiliary embedding spaces between visible and infrared modalities, thereby reducing
97-their differences. Another example is FDNM [24], which employs a
98:salience-guided and amplitude nuance mining strategy to extract crossmodality frequency domain information, thereby decreasing modality
99-discrepancies and enhancing VI-ReID performance.
100-Although these methods have achieved promising performance,
101-they still face two main issues:
102:1. When it comes to mining intrinsic attributes within a single modality,
103:existing methods primarily rely on a fixed, pre-learned set of convolution kernels that are uniformly applied to samples from all modalities. As a result, these static operators fail to capture modalityspecific frequency-domain characteristics (such as structural textures and body contours) present within individual modalities. For
104-instance, a filter that is optimal for extracting fine textures in visible
105-images may inadvertently amplify interference noise when applied
106-to infrared images (e.g., strong background–foreground contrast, as
107-illustrated in Fig. 1(a)). This limitation ultimately constrains the
108:network’s ability to discover highly discriminative, modality-specific
109-features.
110:2. Regarding feature alignment across different modalities, although several recent studies have introduced frequency-domain modeling,
111-many existing approaches still rely primarily on spatial-domain
112:cues, such as appearance attributes and relative positional relationships, for cross-modality alignment. However, due to the substantial distributional differences between visible and infrared images,
113:spatial representations are often sensitive to modality-specific variations, which limits their robustness for alignment. In this context, frequency-domain information (encompassing global structure,
114-coarse appearance distribution, and complementary edge-like responses) provides features that are relatively more stable across
115:modalities. Therefore, effectively exploiting these spectral characteristics remains crucial for improving cross-modality feature alignment.
116-
117-Fig. 1. Motivation for the overall architecture of the 4SNet framework. (a)
118-Grayscale Fast Fourier Transform (FFT) magnitude spectra illustrate that
119:infrared images concentrate most of their energy in low-frequency bands,
120:whereas visible images contain richer medium- and high-frequency details.
121-This clear spectral divergence suggests that different modalities exhibit distinct
122:frequency responses, making fixed shared filters inadequate for capturing
123:modality-specific discriminative features. (b) Motivated by this observation,
124:4SNet introduces two complementary frequency-domain modules, namely the
125-Adaptive Frequency Filter (AFF) and the Spectrum Synergy Module (SSM). A
126-comparison between a baseline fixed stem (denoted as Conv1) and AFF under
127-visible and infrared inputs shows that fixed shared filters are suboptimal for
128-early-stage feature extraction, which justifies the use of AFF for input-adaptive
129:and modality-specific frequency filtering. Meanwhile, the visualization of
130:low-, middle-, and high-frequency decompositions reveals that different spectral bands preserve complementary structural and fine-grained details. The
131-SSM explicitly models this band-wise decomposition and their synergistic
132:relationships to achieve more robust cross-modality feature alignment.. (For
133-interpretation of the references to colour in this figure legend, the reader is
134-referred to the web version of this article.)
135-
136-2. For the second problem, the SSM decomposes the characteristic
137:spectrum into low-, medium-, and high-frequency components.
138-These components are then processed in parallel using band-specific
139:operators (e.g., lightweight convolutional blocks). This strategy retains stable low-frequency body structures and coarse appearances,
140:while exploiting complementary mid-to-high frequency variations
141-and fine edge details.
142-
143-To overcome these limitations, this study introduces a novel and
144-efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet),
145-which integrates an Adaptive Frequency Filter (AFF) module and a
146:Spectrum Synergy Module (SSM) to respectively extract singlemodality-robust features and accomplish cross-modality feature alignment for the VI-ReID task, as shown in Fig. 1(b). Specifically:
147-
148-Finally, a fusion module is utilized to integrate these features,
149:with a particular focus on the enhanced low-frequency components to
150-guide the alignment of cross-domain features. Extensive experiments
151-on three public VI-ReID datasets (SYSU-MM01 [25], RegDB [26], and
152-LLCM [23]) demonstrate that 4SNet achieves competitive performance.
--
154-retrieval capability.
155-
156-1. For the first issue, the AFF module dynamically generates inputspecific filters to address the limitations of static convolutional
157:kernels. The module first learns a set of frequency bases from images
158-using spatial attention. These bases are then adaptively combined
159-under channel attention guidance, synthesizing customized filters
160:that extract modality-inherent frequency-domain features.
161-2
162-
163-Pattern Recognition 180 (2026) 113966
164-
165-M. Xiong et al.
166-
167:The rest of the manuscript is organized as follows: Section 2 comprehensively reviews related work in the VI-ReID field. Section 3
168-elaborates on the proposed 4SNet framework. Section 4 validates the
169-effectiveness of the proposed method through a series of extensive experiments described previously. Finally, Section 5 concludes the paper
170-and outlines potential future research directions rooted on our findings.
171-
172-2.3. Frequency domain analysis in vision tasks
173:The frequency domain offers a complementary perspective for image representation and understanding, providing intrinsic robustness
174-against certain spatial transformations and local perturbations [40].
175-By decomposing visual signals into components that capture global
176:structure and fine-grained details at different spectral scales, frequencydomain analysis enables models to disentangle information that is often
177:entangled in the spatial domain. As a result, frequency-aware modeling
178-has been successfully applied across a wide range of computer vision
179-tasks. In particular, studies in domain adaptation and generalization
180:have demonstrated that low-frequency components, which predominantly encode global shape and coarse structural information, tend to
181:be more stable and domain-invariant than high-frequency components
182-that capture texture and local appearance variations [1,41]. These
183:properties make frequency representations especially suitable for scenarios involving significant domain shifts. Beyond domain adaptation,
184:frequency-domain insights have also been leveraged to improve adversarial robustness, where sensitivity to high-frequency perturbations
185-plays a critical role [42].
186-Motivated by this observation, several works have incorporated
187:frequency-domain operations directly into deep neural networks. Some
188:approaches have highlighted the effectiveness of frequency-aware channel attention in selectively emphasizing informative spectral responses
189:[43,44], while others have gone a step further by partially or fully replacing standard spatial convolutions with frequency-domain counterparts [45,46]. More recently, wavelet-guided designs such as PST [47]
190:have demonstrated the benefits of frequency-aware decomposition in
191:surface-defect detection, where carefully separating frequency components helps enhance meaningful details while suppressing irrelevant
192:noise. Along this line, several recent VI-ReID methods, including FDMNet [48] and FDNM [24], have explored frequency-domain modeling
193:through amplitude–phase decomposition, motivating further investigation into frequency-aware designs for cross-modality matching
194-problems.
195:Contribution. While prior frequency-based methods such as FDMNet [48] and FDNM [24] have achieved promising performance
196:through amplitude–phase decomposition, they typically treat frequency
197:modeling as a single, unified mechanism for reducing modality discrepancy. In these approaches, amplitude and phase components are
198:jointly manipulated to encourage modality-invariant representations,
199:implicitly assuming that a common frequency processing strategy is
200:sufficient for both intra-modality feature extraction and inter-modality
201:alignment. As a consequence, the intrinsic differences in frequency
202-distribution characteristics between visible and infrared modalities are
203-not explicitly addressed at different stages of the network. In contrast,
204:4SNet introduces a role-decoupled frequency modeling strategy, in
205:which frequency information is explicitly assigned to two complementary and well-defined roles within the framework. Specifically, the
206-AFF operates at the input stage in a dual-stem, non-shared manner
207:to perform modality-specific and input-adaptive frequency filtering,
208-enabling visible and infrared images to be characterized by customized
209:frequency responses before entering the shared backbone. The Spectrum Synergy Module (SSM) addresses cross-modality alignment as
210:a separate modeling problem by adopting a coarse low-, medium, and high-frequency band decomposition and performing band-wise
211-processing and fusion. This design emphasizes robust shared structural
212-cues while preserving complementary discriminative details, thereby
213-better meeting the practical requirements of day–night visible–infrared
--
216-2. Related work
217-2.1. Image-based VI-ReID methods
218-As previously mentioned, image-based methods depend on generative models (such as GANs [11] or Diffusion models [12]) to synthesize
219:an auxiliary data modality that bridges the gap between visible and
220-infrared modalities, minimizing the differences between the two modalities and effectively addressing the VI-ReID task. DiVE [12] is a recent
221-proposal that adopts this strategy: it generates visible-IR paired images
222-from samples with the same identity ID using a fine-tuned diffusion
223-model, thereby extending the text-driven approach to achieve identity
224-consistency for the VI-ReID task. Pan et al. [27] employ a conditional diffusion model, which is grounded on binary modal indicators
225-and modal independence, to respectively generate intermediate modal
226:images of the target modality and pedestrian identity. The bidirectional modal interaction network proposed in [2,28] also follows an
227-image-based strategy by separately generating samples for each other’s
228:modality to lower the difficulty of modal recognition. CSDN [29] and
229:DMA [30] have proposed, respectively, a CLIP-driven semantic discovery network and a dual modality-aware alignment method, designed to
230-generate intermediate modalities. Other similar approaches, like [31,
231-32], also adopt the strategy of generating intermediate modalities at
232-the image level to address the VI-ReID problem.
--
235-Unlike their image-based counterparts, feature-based methods [33]
236-mainly map the features of the two modalities to a shared embedding
237-space, and subsequently unify the modeling and mapping processes to
238:achieve cross-modality feature alignment. Some approaches [25,31]
239-employ a two-stream architecture, which focuses on aligning feature distributions through metric learning and adversarial training.
240:CPN [34] tackles VIS-NIR person ReID through angular metric learning, projecting features into a modality-invariant hypersphere space.
241:IDKL [17] explored feature disentanglement to separate modalityinvariant from modality-specific information, enhancing the shared
242:representation. TSKD [1] proposes a two-stage distillation for progressive cross-modality alignment and intra-modality reduction, robust
243-under weak supervision. MSCMNet [32] extracts multi-scale shared
244:semantic features to accomplish cross-modality feature alignment tasks.
245:DMDL [35] further introduces a dual-level modality debiasing learning
246:framework for unsupervised VI-ReID, where modality bias is reduced
247-at both the model and optimization levels. In addition, related works
248:such as [36,37], aim to achieve cross-modality feature alignment
249-by extracting intrinsic features and cross-domain invariant features
250-separately, and using other auxiliary information (such as semantic
251:text-guided) to bridge the modality gap. Recent endeavors [38,39]
252-have also pivoted towards part-level alignment, introducing dynamic
253-and semantically-aware strategies to surpass the limitations of simple
254-horizontal striping (namely, dividing an image into a fixed number of
255-equal-sized horizontal segments, which fails to account for changes
256-in pose or body part location). Different from these methods, 4SNet
257:addresses supervised VI-ReID from a frequency-modeling perspective.
258-
259-3. Proposed 4SNet framework
260:In this section we introduce the proposed 4SNet network, whose
261-structure is summarized in Fig. 2. Firstly, the AFF module dynamically detailed in Section 3.2 generates a unique filter to capture the
262-3
263-

exec
/bin/zsh -lc 'pdftotext -l 3 "ADA framework for unsupervised domain adaptation person re-identification.pdf" - | rg -n -C 3 "Abstract|However|Specifically|contributions|we propose|we introduce|domain|adaptation|sparse|attention|pseudo|clustering|SAAB|SAP"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5-Pattern Recognition
6-journal homepage: www.elsevier.com/locate/pr
7-
8:ADA framework for unsupervised domain adaptation person
9-re-identiﬁcation
10-Wei Zhang a , Peijun Ye b , Dihu Chen c , Tao Su a,∗
11-School of Electronics and Information Technology, Sun Yat-sen University, Guangzhou Higher Education Mega Center, Guangzhou, China
--
21-
22-Keywords:
23-Person re-identiﬁcation
24:Domain adaptation
25:Sparse self-attention
26-
27-a b s t r a c t
28-Domain shift remains a critical barrier for generalizing person re-identiﬁcation (ReID) models across datasets.
29:To address this challenge, we present a sparse self-Attention augmented Domain Adaptation (ADA) framework
30:that learns domain-invariant identity features through three key innovations: (1) Sandwich Attention Primitive
31:(SAP), a novel computational unit designed to boost primitive-level domain adaptation. (2) Sparse self-Attention
32:Augmented Bottleneck block (SAAB block), a hierarchical block integrating SAP to enhance adaptation at the
33:architecture level. (3) Scalable Design, if necessary, SAAB block can be ﬂexibly cascaded to construct taskspeciﬁc ADA framework. Experiments on three benchmarks validate ADA’s superiority: (1) Achieves state-of-theart performance across domains (e.g., 16.5 % mAP gain on CUHK03→Market-1501). (2) Demonstrates consistent
34-generalizability and adaptability.
35-
36-1. Introduction
37-Unsupervised Domain Adaptive Person Re-Identiﬁcation (UDA ReID) transfers recognition from labeled source to unlabeled target
38:domains via domain-invariant identity features. This task addresses
39:cross-camera or cross-domain identity matching without target domain
40-annotations, making it a critical technology for intelligent missing person search and smart city surveillance systems. Consequently, UDA ReID has garnered signiﬁcant research interest due to its practical applicability and technical challenges.
41:The domain shift in UDA Re-ID primarily stems from dataset-speciﬁc
42:variations in pedestrian appearance. Key factors include: (1) Illumination discrepancies: Lighting condition diﬀerences across domains alter
43-visual features. (2) Viewpoint variations: Camera perspectives aﬀect
44-pose and occlusion patterns. (3) Background clutter: Irrelevant scene elements introduce noise to feature representations. (4) Camera-speciﬁc
45:biases: Sensor characteristics (e.g., resolution, color proﬁles) distort appearance consistency. To mitigate these challenges, learning domaininvariant pedestrian features-representations robust to dataset-speciﬁcbiases is essential for eﬀective cross-domain adaptation.
46:Prior studies on UDA Re-ID have primarily focused on three dominant paradigms: (1) Clustering-based adaptation, (2) Image style
47-translation, and (3) Feature-level alignment. While these methods
48:have advanced cross-domain adaptation research, they exhibit critical
49-
50-limitations: (1) Feature-level alignment often relies on auxiliary annotations or information to align distributions, increasing implementation
51-complexity. (2) Image style translation suﬀers from unstable optimization due to its dependency on generative adversarial networks (GANs).
52-(3) Clustering-based methods are prone to error propagation from noisy
53:pseudo-labels and imbalanced cluster distributions.
54-To address these challenges, we present a ADA framework -a
55:lightweight, sparse self-Attention augmented Domain Adaptation framework that integrates sparse token interactions to learn domain-invariant
56-identity representations. Our ADA framework eliminates the need for
57:auxiliary supervision while achieving superior cross-domain generalization.
58:In brief, our ADA framework advances domain adaptation in person
59-re-identiﬁcation (ReID) through following innovations:
60-•
61-
62:Sandwich Attention Primitive (SAP): We introduce a novel computational primitive(SAP) integrating a sandwich attention mechanism
63:to enhance domain adaptation. The SAP ﬂexibly adapts to sparse
64:self-attention mechanisms across heterogeneous design paradigms,
65:eﬀectively bridging domain shifts by synergizing the complementary strengths of convolutional inductive bias and sparse token
66-interactions.
67:• Sparse self-Attention Augmented Bottleneck block (SAAB block):
68:Building on SAP, we propose the SAAB block, a hierarchical
69-
70-∗ Corresponding author.
71-
--
79-
80-W. Zhang et al.
81-
82:bottleneck structure that systematically improves domain-invariant
83:feature learning. If necessary, by cascading SAAB blocks, we construct a scalable sparse self-Attention augmented Domain Adaptation framework (ADA framework), enabling ﬂexible integration into
84-existing architectures.
85-• Robustness and Superiority: Extensive experiments on three ReID
86:benchmarks demonstrate that the ADA framework achieves consistent, generalizable, and adaptable performance across varying sparse
87:self-attention conﬁgurations. As a plug-and-play module, the ADA
88-framework enhances model accuracy at multiple deployment stages
89:while surpassing many SOTA methods in cross-domain scenarios.
90-
91-In contrast to these approaches, our methodology circumvents multiple limitations through three key innovations: First, it operates without
92:requiring auxiliary annotations from either domain, reducing implementation complexity. Second, it eliminates dependence on GAN-generated
93-synthetic images, thereby avoiding quality constraints of image translation. Third, the proposed architecture demonstrates superior training
94:stability and faster convergence compared to clustering-based alternatives while maintaining deployment ﬂexibility.
95:2.2. Self-attention augmented convolution
96:The synergistic integration of self-attention with convolution enables
97-complementary feature learning, achieving performance enhancements
98-with moderate computational overhead. This fusion paradigm addresses
99-the intrinsic limitations of CNNs in modeling long-range dependencies
100-while preserving local feature extraction capabilities.
101:Self-attention augmented convolution addresses the inherent locality constraints of conventional convolution through long-range dependency modeling. AA-ResNet [10] pioneered a parallel implementation
102:where convolutional and self-attention operators process identical input feature maps simultaneously. The ﬁnal output is generated through
103-channel-wise concatenation of the convolution-derived local features
104:and self-attention-extracted global contextual features. BoTNet [11] enhanced this paradigm by replacing spatial convolutions in ResNet’s ﬁnal
105:bottleneck stage with attention modules, realizing measurable performance gains.
106:These approaches integrate standard self-attention mechanisms with
107-quadratic spatial complexity 𝑂((𝐻𝑊 )2 ) (where H and W denote feature map height and width) within supervised learning frameworks.
108:In contrast, our method introduces two key upgradation: (1) sparse
109:self-attention with reduced spatial complexity, and (2) unsupervised
110:domain adaptation deployment without backbone architecture modiﬁcation. The proposed modular design enables ﬂexible implementation
111:while eﬀectively bridging domain gaps in person re-identiﬁcation tasks.
112:Existing supervised methods universally require structural alterations to backbone networks for performance enhancement. Our framework diﬀerentiates itself by operating in unsupervised domain adaptation (UDA) scenarios, maintaining the original backbone architecture
113-while achieving comparable adaptability. This non-invasive integration
114:ensures deployment ﬂexibility without compromising target domain
115-generalization.
116-
117-2. Related work
118-Prior research relevant to this work primarily encompasses three
119:technical directions in unsupervised domain adaptation (UDA) for
120:person re-identiﬁcation (Re-ID): self-attention augmented convolution, transformer-based person re-identiﬁcation, and domain adaptation
121-methodologies for person re-identiﬁcation. This section provides a concise review of these foundational approaches.
122-2.1. UDA methods for person re-identiﬁcation
123:Current UDA-ReID solutions primarily address domain gaps through
124-three principal paradigms mentioned earlier. Each methodology
125-presents distinct advantages and limitations as analyzed below:
126:Feature-Level Alignment. This category mitigates domain discrepancies through hierarchical feature alignment across source and target
127:domains. Through dynamic token matching and ﬂexible allocation of
128-local tokens to human joint points, POSR [1] propose pose-aligned feature learning module to obtain pose-aligned features. HUDA [2] implements joint global distribution alignment and local instance matching
129-to enhance knowledge transferability. Similarly, TJ-AIDL [3] employs a
130-dual-branch network architecture with identity inference space, leveraging both semantic labels and auxiliary attribute annotations from source
131:domains. While eﬀective, these methods often require supplementary
132-annotations (e.g., attribute labels), increasing implementation complexity and data acquisition costs.
133-Image Style Translation. GAN-based approaches attempt to bridge
134:domain gaps through cross-domain image synthesis. PTGAN [4] exempliﬁes this strategy by preserving source-domain identity attributes
135:while transferring target-domain background characteristics. PDA-Net
136:[5] learns domain invariant representation with disentangled pose
137:and domain information. However, such methods face two inherent
138-constraints: (1) The training instability characteristic of adversarial
139-networks often prolongs convergence time compared to conventional
140-CNNs; (2) The synthetic image quality remains fundamentally constrained by ﬁtting quality of GANs.
141-Clustering-Based Adaptation. Pseudo-labeling techniques typically
142:employ pre-trained models for feature extraction followed by clusterguided pseudo-label assignment. Recent advances address critical limitations in this paradigm: MGCN [6] learns noise-resistant features
143:via prototype-, triplet-, and list-wise consistency. 𝐶 3 𝐴𝐵 [7] introduces cluster-aware boundary adaptation to reﬁne pseudo-label reliability while mitigating class imbalance through its Cluster-Adaptive Balance (CAB) loss. The MLC framework [8] further enhances robustness
144-through multi-granularity feature learning, memory bank-based label
145:assignment, and noise-resistant self-paced clustering. The CMC framework [9] employs a graph convolutional network to construct a connection estimator, which facilitates the transfer of relational knowledge
146:from the source domain to the target domain. Nevertheless, clusteringbased approaches inherently suﬀer from two limitations: (1) Performance ceilings imposed by clustering algorithm precision; (2) Error
147:propagation from noisy pseudo-labels that exacerbate intra-cluster data
148-imbalance.
149-
150-2.3. Transformer-based person re-identiﬁcation
--
158-or multi-branch architectures, they signiﬁcantly increase computational
159-complexity due to dense matrix operations within backbone network. In
160-contrast, our approach retains the original backbone architecture and
161:requires only sparse self-attention operations, drastically reducing computational overhead.
162-3. Methods
163-3.1. Description of the overall workﬂow
164:The proposed domain adaptation framework operates in joint training mode through the following workﬂow: Labeled source domain images and unlabeled target domain samples are aggregated and partitioned into mini-batches containing equal proportions of both domains.
165-2
166-
167-Pattern Recognition 171 (2026) 112238
--
169-W. Zhang et al.
170-
171-Fig. 1. The ﬂowchart for the ADA framework: the ADA framework is deployed only once, and the eﬀective deployment location is behind any one of the stages
172:1/2/3 of the backbone network. The backbone is inserted a hybrid classiﬁer. The pseudo labels were generated after DBSCAN clustering and updated once at the
173-starting of each training epoch.
174-
175-Each mini-batch is sequentially processed through our network pipeline,
--
182-(1)
183-
184-The ADA framework’s deployment requires balancing between computational overhead and performance optimization. Operating as a plugand-play module across both training and inference phases, the framework’s architecture permits ﬂexible scalability. In our implementation,
185:a simple conﬁguration is adopted with single SAAB block integration.
186-Note: (1)after stages 1/2/3 are all eﬀective deployment locations. This
187:design choice ensures computational eﬃciency while maintaining adaptation eﬀectiveness. (2) Inspired by the review [17,18], with the rapid
188:evolution of sparse self-attention mechanisms, it is possible to ﬂexibly
189:decide whether to stack SAAB blocks based on practical needs. Fig. 2
190-provides detailed architectural speciﬁcations.
191-
192-Fig. 2. The diagram of the ADA framework: for simplicity, the BN and ReLU
193:operations are not drawn in the ﬁgure. In this article, stackable SAAB block is
194-only stacked once, and the performance improvement is signiﬁcant. Therefore,
195-its memory usage is relatively reasonable.
196-
197-3.2. Introduction to the ADA framework
198-
199:The SAP augments the capacity of domain adaptation through: (1)
200:Sparse Attention Eﬃciency: Implements Sparse complexity attention(<
201:𝑂(𝑁 2 )) versus conventional approaches(= 𝑂(𝑁 2 )). (2) Modular Flexibility: Permits plug-and-play integration of various sparse self-attention
202-variants. (3) Symmetrical Feature Encoding: Channel-wise symmetric
203:distributions emerge through bottleneck-residual coupling. Fig. 3 depicts the functional block diagram of SAP.
204-
205:3.2.1. Introduction to the SAAB block and SAP
206:The ADA framework enhances cross-domain discriminative capability through domain-invariant identity representation learning.
207:Its core innovation lies in dual attention mechanisms: strong focus
208-on identity-salient features (pedestrian attributes) while suppressing
209:domain-speciﬁc distractors (background elements) via long-range dependency modeling. As joint training progresses, the framework demonstrates progressive domain adaptation through the SAAB block and the
210:computational primitive SAP.
211:To enhance domain adaptation performance, the SAAB block introduces the following key improvements over the bottleneck block structure: (1) Core Computational Primitive. The bottleneck block relies on
212:standard convolution, whereas the SAAB block replaces it with the SAP
213-(Sandwich Attention Primitive). (2) Bi-Level Bottleneck Structure. The
214:ﬁrst level is embedded via SAP. The second level is constructed by
215:the SAAB block based on SAP. (3) Expanded Receptive Field. In bottleneck block, the receptive ﬁeld is limited by a single 3 × 3 convolution. The SAAB block achieves broader contextual awareness through
216:self-attention mechanism. (4) Overﬁtting Integration. The SAAB block
217-incorporates dropout operations, which are absent in bottleneck block,
218-further improving robustness.
219-
220:3.2.2. Algorithm description for the SAP
221:To rigorously characterize the SAP, we present its mathematical implementation through PyTorch-style pseudocode across ﬁve sparse selfattention variants(Linear Self-Attention [19], Permuted Self-Attention
222-[20], Focused Linear Attention [21], Bi-Level Routing Attention [22],
223:Window based Self-Attention [23]). The pseudocode description is presented in the following algorithmic listing.
224-#i n p u t f e a t u r e Xin a r e ( Cin , H, W) . o u t p u t f e a t u r e a r e
225-( Cout , H, W)
226-#2D c o n v o l u t i o n Conv2d_head , ( Cout , H, W)

codex
说明一下：你这条消息里没有列出具体 PDF 文件名。我按目录里已有批次清单的第一批 8 篇 `out_aa` 处理，并且只用 `pdftotext -l 3` 读了前三页，没有读全文。

**逐篇提取**

1. `'Knowledge and experience' for visible-infrared person re-identification`
创新类型：新机制。  
为什么能发：把 VI-ReID 从“只看输入图像中的模态共享特征”改成“输入特征加训练集身份关系记忆推理”。证据链是身份特征中心记忆、知识模块、推理模块，再做输入特征和推理特征的消融。  
story 套路：人识别目标时会调用历史经验，模型也应从训练集身份关系里推断当前图像缺失的模态不变信息。

2. `3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification`
创新类型：新数据模态加新机制。  
为什么能发：视频 ReID 的核心痛点被定义为帧级特征质量差，包括多帧错位、遮挡、外观混淆。它引入 SMPL/3D mesh/point cloud，把 3D 几何和形状作为稳定补充。证据链是 3D 融合、时空形状交互、四个视频数据集和消融。  
story 套路：人生活在 3D 空间，2D 视频帧不稳定，3D 几何能给 ReID 提供更稳定的身份线索。

3. `4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification`
创新类型：新机制。  
为什么能发：把 VI-ReID 的模态差异进一步拆成频谱响应差异。不是泛泛做频域增强，而是区分两个角色：单模态自适应频率滤波和跨模态低中高频协同对齐。证据链是 FFT 可视化、AFF、SSM、三数据集和频带消融。  
story 套路：昼夜模态差不是普通外观差，而是频谱结构差，所以要按频带分工对齐。

4. `A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification`
创新类型：新机制，但方向已拥挤。  
为什么能发：换衣 ReID 中衣服是干扰源，现有形状或轮廓方法没有充分用人体语义和衣服区域屏蔽。证据链是人体语义注意力、衣服遮挡屏蔽、四个换衣数据集。  
story 套路：换衣识别不是学更多外观，而是主动屏蔽衣服，把模型逼到非衣服语义上。

5. `A Survey on 3D Skeleton Based Person Re-Identification`
创新类型：问题重定义和领域综述。  
为什么能发：把 ReID 的身份载体从外观转到 3D skeleton，强调隐私、轻量、抗视角和背景。证据链是任务定义、方法谱系、数据集和挑战归纳。  
story 套路：外观不是唯一身份载体，骨架和运动能支撑更长期、更隐私友好的 ReID。

6. `A training-free framework for text-to-image person re-identification via query-prototype matching`
创新类型：问题重定义加推理机制。  
为什么能发：指出 TI-ReID 的核心不是单图匹配，而是实例匹配和身份级相关性不一致。证据链是 gallery 聚类成身份原型，Query-Prototype 相似度补充 Query-Gallery 相似度，多粒度级联和自适应聚类数。  
story 套路：ReID 的答案本来是身份级的，所以检索时不该只看单个 gallery 实例。

7. `ADA framework for unsupervised domain adaptation person re-identification`
创新类型：偏工程组合。  
为什么能发：把 UDA ReID 的痛点归纳为辅助标注复杂、GAN 不稳定、伪标签噪声，然后用稀疏自注意力瓶颈做轻量插件。证据链是不同插入位置、不同稀疏注意力配置、跨域实验。  
story 套路：不用额外标注和生成图像，只用稀疏长程交互增强域不变特征。

8. `Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification`
创新类型：新机制。  
为什么能发：把 USVI-ReID 的瓶颈明确成伪标签噪声造成三类伤害：记忆库错误、跨模态对应错误、负样本采样偏差。证据链是伪标签校准与划分、最优传输原型匹配、去偏记忆学习、邻域关系学习。  
story 套路：无监督跨模态 ReID 的敌人不是模态差一个，而是伪标签闭环被噪声带偏。

**反推出的强候选**

1. Headline：把 aerial-ground ReID 从 2D 图像对齐改成 3D 人体表面对齐。  
挂靠资产：CARGO/AG-ReID.v2、SMPL、pose 热图门控、SOLIDER-Swin。  
区别：最像 `3D-Aided Pedestrian Representation Learning`，但它解决普通视频 ReID 的帧错位和遮挡，我们解决航拍和地面之间的极端视角错位。关键不是“加 3D 特征”，而是用 SMPL 投影把 aerial 和 ground 的 token 对齐到同一人体表面或关节结构上。  
便宜首验：冻结 SOLIDER，在 AG-ReID/CARGO 上只做 SMPL 部位投影池化和可见性重加权，不训练或只训一层线性融合。若跨视角 mAP/rank1 连噪声线都过不了，直接杀。

2. Headline：为低清航拍构造“频率可靠性门控”，让模型只在可信频带里做跨视角匹配。  
挂靠资产：aerial-ground、SOLIDER-Swin、SMPL。  
区别：最像 `4SNet`，但 4SNet 是可见光和红外的频谱分工，我们针对航拍低清、俯视、压缩和地面高清之间的信息不对称。机制上用 SMPL 或姿态估计判断哪些区域和频带可靠，高频衣服纹理在航拍端不可靠时降权，低频身体结构和中频轮廓成为主对齐信号。  
便宜首验：把地面图按航拍尺度退化，比较 SOLIDER 特征在低频、中频、高频分量上的同身份稳定性。若高频降权不能提升正负样本间隔，杀。

3. Headline：用 3D 姿态一致性审计无监督 aerial-ground 伪标签，而不是相信外观聚类。  
挂靠资产：aerial-ground、SMPL、pose 热图门控。  
区别：最像 `APPD`，但 APPD 是 VI-ReID 里清洗可见光和红外伪标签。我们的切开点是航拍和地面外观差异极大，单靠外观聚类天然会把相似衣服、相似背景和视角偏差混在一起；SMPL/pose 提供一个独立于衣服纹理的几何一致性审计器。  
便宜首验：在有标签的 AG-ReID/CARGO 上先隐藏标签跑 DBSCAN，再用真实标签只评估伪标签纯度。加入姿态可见性、骨架比例、投影质量作为置信度过滤后，如果高置信伪标签纯度没有明显提升，杀。

4. Headline：把 gallery 从实例库变成“跨视角身份原型场”，补偿航拍 query 缺失的地面细节。  
挂靠资产：SOLIDER-Swin、aerial-ground、SMPL。  
区别：最像 `Knowledge and experience` 和 `training-free query-prototype matching`。它们分别做 VI-ReID 的身份记忆推理和 TI-ReID 的推理期原型修正。我们的切开点是跨视角信息缺失：航拍 query 缺少正面纹理和细节，地面 gallery 有更完整的人体表面信息。不能做简单测试期重排序，而要训练一个视角条件的原型补偿模块，并用严格对照证明不是普通原型平滑。  
便宜首验：冻结 backbone，把 gallery 按 pose/view/SMPL 部位聚成原型，给 aerial query 加一个原型相似度分数。若只在训练集调参就能在验证集稳定改善，再考虑做成训练期模块；若只对少数 query 有用或高度依赖聚类数，杀。

我的排序：首推第 1 个，问题最像真 gap，且最能吃到团队已有 SMPL 和航拍地面资产；第 3 个适合作为无监督或标注效率副线；第 2 个可以作为第 1 个里的可靠性子模块，不建议单独当主线；第 4 个风险最大，容易被审稿人看成推理期原型技巧。
tokens used
108,651
说明一下：你这条消息里没有列出具体 PDF 文件名。我按目录里已有批次清单的第一批 8 篇 `out_aa` 处理，并且只用 `pdftotext -l 3` 读了前三页，没有读全文。

**逐篇提取**

1. `'Knowledge and experience' for visible-infrared person re-identification`
创新类型：新机制。  
为什么能发：把 VI-ReID 从“只看输入图像中的模态共享特征”改成“输入特征加训练集身份关系记忆推理”。证据链是身份特征中心记忆、知识模块、推理模块，再做输入特征和推理特征的消融。  
story 套路：人识别目标时会调用历史经验，模型也应从训练集身份关系里推断当前图像缺失的模态不变信息。

2. `3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification`
创新类型：新数据模态加新机制。  
为什么能发：视频 ReID 的核心痛点被定义为帧级特征质量差，包括多帧错位、遮挡、外观混淆。它引入 SMPL/3D mesh/point cloud，把 3D 几何和形状作为稳定补充。证据链是 3D 融合、时空形状交互、四个视频数据集和消融。  
story 套路：人生活在 3D 空间，2D 视频帧不稳定，3D 几何能给 ReID 提供更稳定的身份线索。

3. `4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification`
创新类型：新机制。  
为什么能发：把 VI-ReID 的模态差异进一步拆成频谱响应差异。不是泛泛做频域增强，而是区分两个角色：单模态自适应频率滤波和跨模态低中高频协同对齐。证据链是 FFT 可视化、AFF、SSM、三数据集和频带消融。  
story 套路：昼夜模态差不是普通外观差，而是频谱结构差，所以要按频带分工对齐。

4. `A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification`
创新类型：新机制，但方向已拥挤。  
为什么能发：换衣 ReID 中衣服是干扰源，现有形状或轮廓方法没有充分用人体语义和衣服区域屏蔽。证据链是人体语义注意力、衣服遮挡屏蔽、四个换衣数据集。  
story 套路：换衣识别不是学更多外观，而是主动屏蔽衣服，把模型逼到非衣服语义上。

5. `A Survey on 3D Skeleton Based Person Re-Identification`
创新类型：问题重定义和领域综述。  
为什么能发：把 ReID 的身份载体从外观转到 3D skeleton，强调隐私、轻量、抗视角和背景。证据链是任务定义、方法谱系、数据集和挑战归纳。  
story 套路：外观不是唯一身份载体，骨架和运动能支撑更长期、更隐私友好的 ReID。

6. `A training-free framework for text-to-image person re-identification via query-prototype matching`
创新类型：问题重定义加推理机制。  
为什么能发：指出 TI-ReID 的核心不是单图匹配，而是实例匹配和身份级相关性不一致。证据链是 gallery 聚类成身份原型，Query-Prototype 相似度补充 Query-Gallery 相似度，多粒度级联和自适应聚类数。  
story 套路：ReID 的答案本来是身份级的，所以检索时不该只看单个 gallery 实例。

7. `ADA framework for unsupervised domain adaptation person re-identification`
创新类型：偏工程组合。  
为什么能发：把 UDA ReID 的痛点归纳为辅助标注复杂、GAN 不稳定、伪标签噪声，然后用稀疏自注意力瓶颈做轻量插件。证据链是不同插入位置、不同稀疏注意力配置、跨域实验。  
story 套路：不用额外标注和生成图像，只用稀疏长程交互增强域不变特征。

8. `Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification`
创新类型：新机制。  
为什么能发：把 USVI-ReID 的瓶颈明确成伪标签噪声造成三类伤害：记忆库错误、跨模态对应错误、负样本采样偏差。证据链是伪标签校准与划分、最优传输原型匹配、去偏记忆学习、邻域关系学习。  
story 套路：无监督跨模态 ReID 的敌人不是模态差一个，而是伪标签闭环被噪声带偏。

**反推出的强候选**

1. Headline：把 aerial-ground ReID 从 2D 图像对齐改成 3D 人体表面对齐。  
挂靠资产：CARGO/AG-ReID.v2、SMPL、pose 热图门控、SOLIDER-Swin。  
区别：最像 `3D-Aided Pedestrian Representation Learning`，但它解决普通视频 ReID 的帧错位和遮挡，我们解决航拍和地面之间的极端视角错位。关键不是“加 3D 特征”，而是用 SMPL 投影把 aerial 和 ground 的 token 对齐到同一人体表面或关节结构上。  
便宜首验：冻结 SOLIDER，在 AG-ReID/CARGO 上只做 SMPL 部位投影池化和可见性重加权，不训练或只训一层线性融合。若跨视角 mAP/rank1 连噪声线都过不了，直接杀。

2. Headline：为低清航拍构造“频率可靠性门控”，让模型只在可信频带里做跨视角匹配。  
挂靠资产：aerial-ground、SOLIDER-Swin、SMPL。  
区别：最像 `4SNet`，但 4SNet 是可见光和红外的频谱分工，我们针对航拍低清、俯视、压缩和地面高清之间的信息不对称。机制上用 SMPL 或姿态估计判断哪些区域和频带可靠，高频衣服纹理在航拍端不可靠时降权，低频身体结构和中频轮廓成为主对齐信号。  
便宜首验：把地面图按航拍尺度退化，比较 SOLIDER 特征在低频、中频、高频分量上的同身份稳定性。若高频降权不能提升正负样本间隔，杀。

3. Headline：用 3D 姿态一致性审计无监督 aerial-ground 伪标签，而不是相信外观聚类。  
挂靠资产：aerial-ground、SMPL、pose 热图门控。  
区别：最像 `APPD`，但 APPD 是 VI-ReID 里清洗可见光和红外伪标签。我们的切开点是航拍和地面外观差异极大，单靠外观聚类天然会把相似衣服、相似背景和视角偏差混在一起；SMPL/pose 提供一个独立于衣服纹理的几何一致性审计器。  
便宜首验：在有标签的 AG-ReID/CARGO 上先隐藏标签跑 DBSCAN，再用真实标签只评估伪标签纯度。加入姿态可见性、骨架比例、投影质量作为置信度过滤后，如果高置信伪标签纯度没有明显提升，杀。

4. Headline：把 gallery 从实例库变成“跨视角身份原型场”，补偿航拍 query 缺失的地面细节。  
挂靠资产：SOLIDER-Swin、aerial-ground、SMPL。  
区别：最像 `Knowledge and experience` 和 `training-free query-prototype matching`。它们分别做 VI-ReID 的身份记忆推理和 TI-ReID 的推理期原型修正。我们的切开点是跨视角信息缺失：航拍 query 缺少正面纹理和细节，地面 gallery 有更完整的人体表面信息。不能做简单测试期重排序，而要训练一个视角条件的原型补偿模块，并用严格对照证明不是普通原型平滑。  
便宜首验：冻结 backbone，把 gallery 按 pose/view/SMPL 部位聚成原型，给 aerial query 加一个原型相似度分数。若只在训练集调参就能在验证集稳定改善，再考虑做成训练期模块；若只对少数 query 有用或高度依赖聚类数，杀。

我的排序：首推第 1 个，问题最像真 gap，且最能吃到团队已有 SMPL 和航拍地面资产；第 3 个适合作为无监督或标注效率副线；第 2 个可以作为第 1 个里的可靠性子模块，不建议单独当主线；第 4 个风险最大，容易被审稿人看成推理期原型技巧。
