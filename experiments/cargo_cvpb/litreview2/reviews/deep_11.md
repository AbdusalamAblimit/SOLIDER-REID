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
session id: 019ef835-98f0-79d3-9441-fbda7dfddb9a
--------
user
你是 ReID 创新方法论拆解员 11。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(11)==
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认工作区上下文，再用 `pdftotext` 读每篇前 8 页，重点截取方法、问题定义和实验闭环相关内容。这个任务不需要改文件。
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
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
../../../CLAUDE.md
../../../AGENTS.md
../../../GOAL.md
../../../experiments/STATUS.md
../../STATUS.md

codex
我已经确认 PDF 技能要求和本目录文件。接下来先读 `GOAL.md`、`CLAUDE.md`、`AGENTS.md`、`experiments/STATUS.md`，然后集中提取这 6 篇的方法部分。
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
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,220p' ../../../experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,260p' ../../../AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
上下文里当前目标仍然是文献差距分析，不开实验。现在我会先用标题和目录式检索定位每篇的方法章节边界，再逐篇读完整方法段落。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[A-Z][A-Za-z ]+"'$|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*)"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3:Dynamic Token Selective Transformer for
7:Maryam Pishgar
9:Department of Computer Science
10:University of Southern California
14:Department of Industrial and Systems Engineering
15:University of Southern California
19:Abstract—Aerial-Ground Person Re-identification (AGPReID)
35:1.18% mAP improvements when compared to the second place.
43:I. I NTRODUCTION
114:II. R ELATED W ORK
143:Token selection is crucial for addressing redundancy issues
151:These approaches have demonstrated substantial reductions in
161:III. M ETHOD
199:VTS mechanism can be understood as a dynamic token
219:These tokens are then ranked based on their importance
260:C. Comparisons with State-of-the-art Methods
307:III. As depicted in Table III, we compare two challenging
328:1.6 × 10−6 over 120 epochs. The batch size is set to 128,
345:IV. E XPERIMENTS
346:A. Experiment settings
352:Dataset
356:Data
357:Real
358:Synthetic
381:Method
423:50.32
424:51.00
425:54.81
426:54.81
427:45.83
428:60.26
429:61.54
430:62.82
431:64.42
433:43.09
434:44.50
435:46.49
436:49.08
437:38.84
438:53.44
439:53.54
440:54.22
441:55.73
443:29.76
444:32.20
445:32.40
446:36.52
447:39.57
448:40.22
449:39.62
450:39.92
451:41.92
453:72.31
454:74.10
455:77.68
456:83.93
457:72.31
458:81.25
459:82.14
460:79.46
461:78.57
463:62.99
464:67.60
465:66.47
466:71.05
467:62.99
468:71.66
469:71.34
470:70.60
471:72.40
473:48.24
474:55.10
475:51.34
476:55.20
477:48.24
478:58.09
479:57.55
480:57.89
481:62.10
483:67.50
484:55.00
485:65.00
486:65.00
487:67.50
488:67.50
489:80.00
490:82.50
491:80.00
493:49.73
494:44.60
495:49.79
496:52.96
497:49.73
498:56.48
499:64.47
500:64.06
501:63.31
503:29.32
504:27.00
505:29.82
506:36.78
507:29.32
508:40.40
509:47.07
510:44.67
511:44.67
513:31.25
514:34.40
515:36.25
516:31.87
517:31.25
518:43.57
519:43.13
520:47.50
521:50.63
523:29.00
524:30.40
525:32.56
526:33.47
527:29.00
528:40.90
529:40.11
530:42.21
531:43.39
533:18.71
534:20.10
535:21.46
536:24.64
537:18.71
538:29.39
539:28.20
540:29.03
541:29.46
547:73.54
548:59.77
549:70.01
550:55.47
551:72.59
552:58.32
553:81.28
554:72.38
555:82.91
556:74.44
557:51.06
558:83.48
559:74.51
560:49.86
566:73.70
567:62.27
568:71.20
569:58.83
570:74.22
571:60.99
572:82.64
573:73.35
574:83.68
575:75.96
576:49.39
577:84.72
578:76.05
579:50.04
584:Method
596:Method
600:Visual Token Selector
608:45.00
609:42.05
610:30.26
611:50.63
612:43.39
613:29.46
634:Method
674:46.25
675:42.56
676:30.16
677:45.00
678:41.28
679:28.83
680:46.88
681:41.04
682:28.12
683:40.00
684:36.58
685:24.73
686:46.88
687:42.46
688:29.79
689:50.63
690:43.39
691:29.46
716:1.28% in mAP, and 1.33% in mINP, highlighting the tradeoff between token quantity and model’s focus on essential
722:The reason behind this could be that tokens in shallow layers
727:V. C ONCLUSION AND F UTURE W ORK
730:transformer specifically for the AGReID task. Experiments
744:R EFERENCES
748:617.
758:2023.
768:3123.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[A-Z][A-Za-z ]+"'$|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*)"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[A-Z][A-Za-z ]+"'$|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*)"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
37:Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
40:Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
59:Introduction
123:Related Work
125:2.1
154:2.2
170:2.3
204:3.1
206:Methodology
207:Problem Formulation and Overview
244:3.2
246:CSM
334:3.3
336:CRLR
386:3.4
388:WMM

 succeeded in 0ms:
13:Abstract—Unsupervised learning visible-infrared person reidentification (USL-VI-ReID) offers a more flexible and costeffective alternative compared to supervised methods. This field
36:I. I NTRODUCTION
49:Yiming Yang and Haifeng Hu are with the School of Electronics and
135:II. R ELATED WORK
224:III. T HE P ROPOSED M ODEL
299:Dj
306:Pn
484:The clustering method plays a pivotal role in unsupervised
496:Unsupervised methods commonly take refined InfoNCE
498:PX
596:PX
625:To explore the instance relationships during the training
643:PX
732:Lm
734:Lm
812:IV. E XPERIMENTS E VALUATION
881:C. Comparison with State-of-the-art Methods

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[A-Z][A-Za-z ]+"'$|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*)"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3:Contents lists available at ScienceDirect
5:Neural Networks
8:Full Length Article
28:Contrastive learning
30:Cluster proxies
44:1. Introduction
128:We propose a novel dynamic adaptive camera Jaccard distance
146:2. Releted works
171:2.1. Unsupervised person Re-ID
185:Our approach addresses the more demanding requirements of the
222:2.2. Contrastive learning
235:2.3. Knowledge distillation
236:Knowledge distillation aims to transfer knowledge from a teacher
250:3. Method
251:3.1. Overview
314:3.2. Dynamic adaptive camera jaccard
459:3.3. Dynamic adaptive proxies
522:We then apply the DOCP loss function on the cluster memory bank
609:3.4. Dynamic adaptive knowledge distillation
646:Compute dynamic adaptive camera Jaccard distance matrixs
664:Compute dynamic adaptive camera Jaccard distance
707:3.5. Overall loss
766:GEM pooling is applied to obtain the corresponding feature vector sets
784:3.6. Training process
797:Compute dynamic adaptive camera Jaccard distance

codex
这批里有 6 个文件，不是 11 个标题。我会按你列出的 6 篇完整拆解；如果还有漏列的 5 篇，需要你再补文件名。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[A-Z][A-Za-z ]+"'$|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*)"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
20:global and partial feature branches to capture richer pedestrian features. Experiments on public and real-world
40:4.0.
58:Introduction
130:2.1
132:Related Work
152:2.2
185:2.3
187:Domain Adaptation
217:Methods
233:This model can provide richer pedestrian image features and improve the extraction capability in
247:3.1
249:WPOSNet
333:3.2
348:SuA achieves style enhancement by introducing Gaussian noise to perturb the mean and variance

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[A-Z][A-Za-z ]+"'$|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*)"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
7:Adaptation Learning
11:Abstract—The Visible-Infrared Person Re-identification (VI
13:A crucial challenge in this task is mitigating the impact of
24:ReID model with sufficient flexibility to adapt diverse modality
38:This work was supported in part by the National Natural Science Foundation
66:I. I NTRODUCTION
230:This version provides a more comprehensive theoretical
235:2) Methodological Enhancements:
236:This version includes significant improvements in the
253:3) Expanded Experimental Validation:
254:This version reports more extensive new experimental
263:II. R ELATED W ORK
301:Modality compensation represents a VI ReID paradigm that
311:GAN for style transfer of training images across different
391:These studies have demonstrated the potential of visual prompt
400:III. A PPROACH
516:We treat the adaptation of the model to visible and infrared
615:Three types of prompts are supplied into the model and
651:To avoid potential conflicts between the two kinds of
696:Unlike the first strategy simply concatenating different
829:IV. E XPERIMENT
883:Our proposed MIP method is implemented on the Pytorch

exec
/bin/zsh -lc "for f in 'Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf' 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' 'Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf' 'Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf' 'Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf' 'Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf'; do printf '%s: ' \""'$f"; pdfinfo "$f" | rg '"'"'^Pages:'"'; done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf: Pages:           24
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf: Pages:           13
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf: Pages:           13
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf: Pages:           6
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf: Pages:           18
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf: Pages:           18

codex
我已定位到每篇的方法章节边界。现在开始逐篇读具体段落，先读无监督可见光红外的几篇，因为它们的“问题重定义”通常藏在聚类、伪标签和模态相机偏差里。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '55,450p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    55	https://doi.org/10.1145/3724397
    56	
    57	1
    58	
    59	Introduction
    60	
    61	Visible-infrared person re-identification (VI-ReID) aims at matching pedestrian images of the
    62	same identity captured by visible and infrared cameras when provided with a query image from
    63	a different modality. The advancements in 24-hour security surveillance systems have led to the
    64	widespread usage of multi-modal cameras that automatically switch between visible and infrared
    65	modes based on lighting conditions. Consequently, VI-ReID has garnered significant attention
    66	[32, 65, 69, 71]. In contrast to traditional person re-identification [14, 43, 51, 61], VI-ReID not only
    67	addresses the challenges like appearance and viewpoint variations in ReID, but further confronts
    68	the significant modality gap between visible and infrared images. To address the modality gap,
    69	some studies designed network architectures and loss functions [13, 24, 31, 53, 64, 71], while some
    70	explored auxiliary features to compensate for modality-shared information [35, 48, 49, 63, 69].
    71	However, these methods rely on manually annotated pedestrian datasets for training, which is
    72	costly and time-intensive.
    73	To advance the practical application of VI-ReID, we investigate unsupervised VI-ReID (USVIReID) in this study. The main goal of USVI-ReID is to accurately associate samples from different
    74	modalities while effectively guiding the model in handling the significant modality difference.
    75	Many USVI-ReID methods [5, 28, 50] employ a two-stage framework to learn modality-invariant
    76	representations. However, these approaches require an additional annotated visible dataset for
    77	model training, limiting the generalizability of the methods. Recently, pseudo labels generated by
    78	advanced clustering algorithms [11, 54] have shown effectiveness in unsupervised learning (USL).
    79	Besides, memory bank-based approaches have garnered increasing attention for unsupervised
    80	single-modality person re-identification [8, 18]. These methods iteratively assign pseudo labels
    81	to samples and train the model with a contrastive learning framework. To successfully associate
    82	samples from different modalities for USVI-ReID, various attempts [4, 55, 59] have been made
    83	utilizing different matching algorithms, achieving impressive performance. However, most existing
    84	methods overlook the noise present in pseudo labels. Furthermore, many methods initialize memory
    85	banks by averaging the features of samples within a cluster, neglecting the hard samples in the
    86	dataset and hindering the effectiveness of cross-modality learning.
    87	To address the aforementioned challenges, we propose a novel method for USVI-ReID called
    88	dual-modality-shared learning and label refinement (DLLR). As shown in Figure 1, the
    89	framework consists of a cluster similarity matching (CSM) module, a cluster relationshipbased label refinement (CRLR) algorithm, and a weighted modality-shared memory (WMM)
    90	module. Initially, intra-modality relationships among samples are established through clustering.
    91	Subsequently, the CSM associates samples from different modalities by assigning modality-shared
    92	pseudo labels to clusters based on cluster similarity. Next, the CRLR is used to enhance the quality
    93	of modality-shared pseudo labels by separating clusters that potentially contain samples with
    94	different identities and merging those that may belong to the same identity. This approach proves
    95	effective by using readily available information from matched cross-modality clusters, providing
    96	high-quality homogeneous instance-level associations that aid the refinement of pseudo labels.
    97	Moreover, considering the diverse difficulties of samples, we introduce a WMM for dual-modalityshared learning, in which a weighted strategy based on the feature distribution is used to assign
    98	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
    99	
   100	DLLR for Unsupervised Visible-Infrared Person ReID
   101	
   102	325:3
   103	
   104	Fig. 1. The overall pipeline of our proposed method. Samples from each modality are clustered into clusters,
   105	and the cluster similarity matching (CSM) module matches clusters from different modalities. The cluster
   106	relationship-based label refinement (CRLR) algorithm is then used to refine modality-shared pseudo labels.
   107	Next, dual-modality-shared learning is conducted using the weighted modality-shared memory (WMM) to
   108	narrow the modality gap while considering sample difficulties.
   109	
   110	different weights to samples for initializing memory banks. Consequently, the proposed model
   111	excels in learning from two modalities by a dual-modality-shared learning framework, while
   112	eliminating the need for labels or additional labeled single-modality datasets.
   113	In summary, our main contributions are listed as follows:
   114	— We propose a DLLR framework for USVI-ReID. By incorporating the CSM and CRLR algorithms, our framework can effectively establish associations between unlabeled samples
   115	across modalities, and then generate high-quality pseudo labels for model training.
   116	— We design the WMM to assign different weights to samples for constructing memory banks,
   117	which can enhance the model’s capacity to learn modality-invariant features by considering
   118	hard samples.
   119	— Extensive experiments on three public benchmarks demonstrate the superiority of our proposed method, outperforming state-of-the-art USVI-ReID methods and even surpassing many
   120	supervised VI-ReID methods.
   121	2
   122	
   123	Related Work
   124	
   125	2.1
   126	
   127	VI-ReID
   128	
   129	VI-ReID is a challenging task due to the substantial modality gap between visible and infrared
   130	images [7, 16, 19, 29, 46, 71]. Existing approaches for VI-ReID can be categorized into the following
   131	two categories. (1) The first category focuses on developing novel network architectures and loss
   132	functions to extract modality-shared features [12, 13, 30, 33, 52, 53, 62]. For instance, Wu et al. [52]
   133	conducted pioneering research in VI-ReID, introducing a zero-padding single-stream network,
   134	generating multiple representations of each image, and then fusing multi-level features to enhance
   135	feature discrimination. Lu et al. [34] proposed a progressive learning strategy with modality-shared
   136	enhancement loss and discriminative center loss to improve the discrimination of modality-shared
   137	features. Hua et al. [22] designed a multi-scale semantic correlation mining network to exploit
   138	semantic features at multiple scales, which improves the issue of losing valuable information during
   139	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   140	
   141	325:4
   142	
   143	L. Dai et al.
   144	
   145	the feature extracting process. (2) The second category mainly aims to generate auxiliary features
   146	to complement modality-shared features during model training [35, 63, 67, 69, 72]. Ye et al. [63]
   147	introduced a random channel augmentation approach to generate an intermediate modality from
   148	the visible modality for enhancing robustness. Wang et al. [49] implemented a cross-modality
   149	paired-image generation and bridge the gap between visible and infrared modalities at both set-level
   150	and instance-level. Additionally, Zhang et al. [69] generated missing modality-specific information
   151	at the feature level to introduce more discriminative features. Nonetheless, a notable limitation of
   152	these VI-ReID methods is their reliance on annotated cross-modality person datasets, which can be
   153	time-consuming and labor-intensive to obtain.
   154	2.2
   155	
   156	Unsupervised Single-Modality Person Re-Identification
   157	
   158	Unsupervised single-modality person re-identification methods can be categorized into two main
   159	categories: unsupervised domain adaptation (UDA) and fully USL. UDA methods [2, 78, 79]
   160	focus on the domain adaptation from a labeled source dataset to an unlabeled target dataset. For
   161	example, Ge et al. [18] utilized contrastive learning with hybrid memory to learn modality-invariant
   162	features in the target domain. Some uncertainty-based ReID methods [74, 76] have been proposed
   163	to mitigate the negative effects of noisy labels during model training. However, assessing the degree
   164	of uncertainty in pseudo labels is challenging in the unsupervised scenario, thereby complicating
   165	the process of label correction. Conversely, USL methods [1, 6, 14, 17, 68, 70] aim to learn the model
   166	using a single unlabeled dataset. Dai et al. [8] constructed a memory bank and iteratively updated
   167	it to compute the cluster contrast loss at the cluster level. Chen et al. [1] investigated instance-level
   168	relationships to enhance class-level contrastive ReID methods. However, these USL-ReID methods
   169	cannot be directly applied to USVI-ReID for the significant modality gap.
   170	2.3
   171	
   172	USVI-ReID
   173	
   174	The primary challenge of USVI-ReID is the association of samples from different modalities and the
   175	reduction of the modality gap [28, 42, 50, 55, 59]. Liang et al. [28] conducted pioneering research
   176	by proposing a two-stage training framework that incorporates both homogeneous learning and
   177	heterogeneous learning. Wang et al. [50] utilized a UDA method to generate labels for visible
   178	samples, then assigned labels for infrared samples using an optimal transport strategy. However,
   179	these methods require additional labeled datasets for training, which limits the model’s robustness.
   180	Recently, memory bank-based methods [3, 27, 45, 58, 66] have shown remarkable performance
   181	for USVI-ReID. To tackle the challenge of missing cross-modality correspondences, Yang et al. [59]
   182	proposed a dual-contrastive learning framework that integrates a cross-modality memory aggregation module to enhance the learning of modality-invariant features. Wu et al. [55] utilized
   183	a progressive graph-matching algorithm to globally mine cross-modality correspondences and
   184	then perform contrastive learning. He et al. [4] designed a bilateral cluster-matching algorithm
   185	that incorporates contrastive learning with both modality-specific and modality-agnostic memory
   186	banks. In addition, some recent works employ novel network architectures to extract features
   187	that are representative of the identity, e.g., Chen et al. [3] leveraged the visual-text representation
   188	ability from CLIP [39] to propose a new prompt learning paradigm, and Yang et al. [58] proposed a
   189	strategy of collaborative learning of deep and shallow features with transformer [10]. However,
   190	these methods overlook the presence of noisy pseudo labels or assign a constant weight to samples
   191	for memory initialization, which limits the model’s ability to extract discriminative information.
   192	In contrast to existing works, we address the noise in pseudo labels through a CRLR module. This
   193	module leverages matched cross-modality clusters to enhance pseudo label quality through cluster
   194	combination and separation. Furthermore, our approach incorporates a weighted strategy that
   195	considers the balance between cross-modality samples and feature differences, thereby facilitating
   196	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   197	
   198	DLLR for Unsupervised Visible-Infrared Person ReID
   199	
   200	325:5
   201	
   202	hard sample learning. With the refined pseudo labels and the proposed WMM, our dual-modalityshared learning framework effectively learns modality-invariant features without any annotations.
   203	3
   204	3.1
   205	
   206	Methodology
   207	Problem Formulation and Overview
   208	
   209	𝑁 represents the visible subset
   210	Given an unlabeled VI-ReID dataset X = {V, R}, where V = {𝑥𝑖𝑣 }𝑖=1
   211	𝑟
   212	𝑟
   213	𝑁
   214	𝑣
   215	and R = {𝑥𝑖 }𝑖=1 denotes the infrared subset, containing 𝑁 and 𝑁 𝑟 images, respectively. A deep
   216	neural network 𝐹𝜃 (·) is trained to project a pedestrian image into the shared feature space, obtaining
   217	a modality-invariant representation 𝑓 ∈ R𝑑 , for matching the cross-modality pedestrians.
   218	In the initial training stage, we employ the Augmented Dual Contrastive learning framework
   219	[59] following recent USVI-ReID methods [4, 55, 59], which alternates between pseudo label
   220	generation and network training. The DBSCAN [11] is utilized to generate modality-specific
   221	𝑁 𝑣 and Y 𝑟 = {𝑦𝑟 } 𝑁 𝑟 for samples from two modalities. Two memory
   222	pseudo labels Y 𝑣 = {𝑦𝑖𝑣 }𝑖=1
   223	𝑖 𝑖=1
   224	𝑣
   225	𝐾 and 𝑀 𝑟 = {𝑐 𝑟 }𝐾 𝑟 , are initialized and updated using a momentum updating
   226	banks, 𝑀 𝑣 = {𝑐𝑖𝑣 }𝑖=1
   227	𝑖 𝑖=1
   228	strategy [8], where 𝐾 𝑣 /𝐾 𝑟 is the number of cluster centroids 𝑐 𝑣 /𝑐 𝑟 in visible/infrared modality.
   229	b = {𝑥 𝑣b}𝑁 𝑣 are generated to support visible stream
   230	Meanwhile, augmented visible samples V
   231	𝑖 𝑖=1
   232	learning process [63]. Additionally, to mitigate the sensitivity of the clustering algorithm against
   233	the environment of different visible cameras, we use a segmentation model [26] to remove the
   234	background from visible images.
   235	Our proposed method is implemented in the second training stage. As illustrated in Figure 2,
   236	after generating pseudo labels by clustering, we design the CSM module to establish cross-modality
   237	correspondences for assigning pseudo labels in another modality to samples. Then, we introduce
   238	a CRLR module, which enhances the quality of pseudo labels by merging and splitting clusters
   239	based on their relationships. Additionally, we design the WMM module to facilitate dual-modalityshared learning. By employing a weighted strategy, the memory banks are effective for learning
   240	modality-shared representations. Algorithm 1 is provided to illustrate the working principle of our
   241	framework.
   242	𝑣
   243	
   244	3.2
   245	
   246	CSM
   247	
   248	In the CSM module, we employ a cluster-level matching strategy to construct a similarity matrix,
   249	find the corresponding clusters through bidirectional matching, and then assign pseudo labels of
   250	another modality to samples, finally forming modality-shared labels for training. Next, we provide
   251	a detailed elaboration on this module.
   252	After applying a clustering algorithm to each modality, we obtain modality-specific pseudo labels
   253	for training the samples. To bridge the modality gap, it is essential to associate samples from the two
   254	modalities to construct cross-modality correspondences for guiding network learning. Intuitively,
   255	samples with the same identity from different modalities should exhibit higher similarity than those
   256	with different identities. Therefore, we calculate the cosine similarities between cross-modality
   257	features 𝑓 𝑣 and 𝑓 𝑟 . Rather than performing instance-level matching, we adopt a cluster-level
   258	matching approach to associate clusters from the two modalities, which preserves the accurate
   259	intra-modality relationships. We define the similarities between clusters from two modalities as
   260	the average similarity between samples within each cluster, thereby constructing a cross-modality
   261	𝑣
   262	𝑟
   263	cluster similarity matrix 𝑆 𝑐 ∈ R𝐾 ×𝐾 :
   264	Í𝑇𝑖𝑣 Í𝑇𝑗𝑟
   265	𝑐
   266	𝑆𝑖,𝑗
   267	=
   268	
   269	𝑓 𝑣 · 𝑓𝑙𝑟
   270	𝑙=1 𝑘
   271	,
   272	𝑇𝑖𝑣 × 𝑇 𝑗𝑟
   273	
   274	𝑘=1
   275	
   276	(1)
   277	
   278	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   279	
   280	325:6
   281	
   282	L. Dai et al.
   283	
   284	Fig. 2. The overall framework of our proposed method. It mainly comprises three components: the CSM
   285	module, the CRLR algorithm, and the WMM module. Different shapes denote different identities, with blue
   286	and yellow representing the infrared and visible modalities, respectively. The clustering step generates pseudo
   287	labels for samples. With CSM, pseudo labels are converted to modality-shared labels by matching to another
   288	modality. CRLR is utilized for label refinement. Dual-modality-shared learning with two WMM banks is
   289	performed with the modality-shared labels.
   290	
   291	Algorithm 1: DLLR
   292	
   293	where 𝑇𝑖𝑣 /𝑇 𝑗𝑟 denote the number of instances within the 𝑖th/𝑗th cluster from visible/infrared
   294	modality. Since we use the l2-norm to normalize the features extracted by the encoder during
   295	pseudo label generation, we directly utilize the dot product of the features to represent the cosine
   296	similarity.
   297	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   298	
   299	DLLR for Unsupervised Visible-Infrared Person ReID
   300	
   301	325:7
   302	
   303	Fig. 3. Illustration of the effects of CRLR. Samples representing different identities are depicted using
   304	different shapes, while different colors indicate different modalities. (a) Samples with different identities may
   305	be erroneously clustered together and assigned the same pseudo label (pid), while samples with the same
   306	identity may be dispersed into different clusters and assigned different pids. (b) Incorrectly clustered samples
   307	and noisy correspondences are rectified.
   308	
   309	Next, we perform a bidirectional matching across modalities that exhibit the highest similarity
   310	to match the clusters. The visible to infrared cross-modality correspondences 𝑄 𝑣𝑟 and the infrared
   311	to visible correspondences 𝑄 𝑟 𝑣 can be represented as:
   312	𝑐 ),
   313	𝑄𝑖𝑣𝑟 = arg max (𝑆𝑖,𝑗
   314	0≤ 𝑗 ≤𝐾 𝑟
   315	𝑐 ).
   316	𝑄 𝑟𝑗 𝑣 = arg max (𝑆𝑖,𝑗
   317	
   318	(2)
   319	
   320	0≤𝑖 ≤𝐾 𝑣
   321	
   322	Based on the obtained cross-modality association corresponding to two matching directions, we
   323	assign pseudo labels from another modality to samples. For instance, we assign pseudo labels of
   324	infrared to the visible sample 𝑥𝑖𝑣 with its pseudo label 𝑦𝑖𝑣 using the correspondence 𝑄 𝑣𝑟 :
   325	𝑦𝑖𝑣𝑟 = 𝑄 𝑦𝑣𝑟𝑣 .
   326	
   327	(3)
   328	
   329	𝑖
   330	
   331	Finally, we form the modality-shared labels (infrared to visible) with Y 𝑣𝑟 of visible samples and
   332	Y𝑟 of infrared samples for training. Similarly, the modality-shared labels for visible to infrared can
   333	be obtained in the same manner.
   334	3.3
   335	
   336	CRLR
   337	
   338	In the CRLR module, we find the matched cross-modality clusters of each cluster and merge them
   339	that satisfy the conditions, and then separate the cluster according to the number of matched
   340	clusters, finally assign refined modality-shared pseudo labels for training. Then, we delve into a
   341	comprehensive description of this module.
   342	In the context of the previous CSM, cross-modality correspondences are established for model
   343	learning. However, the noise inherent in the pseudo labels has not been examined. Some samples
   344	sharing the same identity may be erroneously assigned to multiple clusters, while different identities
   345	might be incorrectly grouped into a single cluster. These inaccuracies typically manifest as manyto-one cross-modality correspondences, as illustrated in Figure 3(a). Such noise not only impacts
   346	the accuracy of cross-modality matching but also hinders the efficiency of model training.
   347	To address this issue, we introduce CRLR to refine the modality-shared labels. Taking the case
   348	of visible-to-infrared matching as an example, the CSM assigns a pseudo label from the matched
   349	infrared cluster to the visible clusters with Equation (3) while despising the fact that these visible
   350	clusters may be distinctly different. However, if clusters from the same modality share the same
   351	identity, a high degree of similarity is expected among them. Therefore, we first obtain the matched
   352	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   353	
   354	325:8
   355	
   356	L. Dai et al.
   357	
   358	cross-modality cluster set 𝐷 𝑟 ∈ 𝑄 𝑣𝑟 for each infrared cluster:
   359	𝐷𝑖𝑟 = { 𝑗 |𝑄 𝑣𝑟
   360	𝑗 = 𝑖},
   361	
   362	(4)
   363	
   364	where j denotes the modality-specific label of infrared clusters. Then, we examine each 𝐷 𝑟 to
   365	merge visible clusters that satisfy the following two conditions before assigning pseudo labels from
   366	another modality: (1) they match the same infrared cluster; (2) the similarity between the clusters
   367	exceeds a specified threshold 𝜖. Subsequently, we consider that an infrared cluster tends to contain
   368	multiple identities if it is matched by k (𝑘 > 1) merged visible clusters. To achieve a high-quality
   369	division, we employ the k-means to partition the infrared cluster into k parts, thereby updating the
   370	𝑁𝑟 .
   371	infrared modality-specific labels Y𝑟 , obtaining Y𝑟𝑑 = {𝑦𝑖𝑟𝑑 }𝑖=1
   372	After the processes of merging and separating, we establish the refined matching relationship
   373	e𝑣𝑟 with the merged visible clusters and the separated infrared labels Y𝑟𝑑 .
   374	of visible to infrared 𝑄
   375	e𝑣𝑟 using
   376	Then, the separated infrared labels Y𝑟𝑑 are assigned to the merged visible clusters with 𝑄
   377	𝑣𝑟
   378	e
   379	Equation (3), obtaining Y . Consequently, the refined modality-shared labels (visible to infrared)
   380	e𝑣𝑟 , Y𝑟𝑑 }. As illustrated in Figure 3(b), the CRLR enhances the
   381	are established and represented as {Y
   382	quality of modality-shared labels through cross-modality correspondences. The refined modalitye𝑟 𝑣 }, which consists of separated visible labels of visible
   383	shared labels (infrared to visible) {Y 𝑣𝑑 , Y
   384	samples and the separated visible labels assigned to the merged infrared clusters, can be obtained
   385	in the same way.
   386	3.4
   387	
   388	WMM
   389	
   390	To build WMM, we initialize the modality-shared centroids by averaging the two modality-specific
   391	centroids, then assign weights to samples according to the similarity between the sample and
   392	modality-shared centroids to re-construct modality-specific centroids, and finally initialize WMM
   393	banks by fusing the modality-shared centroids and weighted version through a momentum strategy.
   394	Then, we give a thorough introduction of this module.
   395	Following the implementation of CSM and CRLR, modality-shared pseudo labels are assigned to
   396	samples from both matching directions. To facilitate the interaction of cross-modality information
   397	and achieve modality-invariant feature extraction, we construct two memory banks: the visible
   398	𝐾 𝑣 and the infrared MM 𝑀 𝑟 = {𝑐 𝑟 }𝐾 𝑟 .
   399	modality-shared memory (MM) 𝑀 𝑣 = {𝑐𝑖𝑣 }𝑖=1
   400	𝑖 𝑖=1
   401	For the visible MM, it is initialized using modality-shared cluster centroids {𝑐 1𝑣 , 𝑐 2𝑣 , . . . 𝑐 𝐾𝑣 𝑣 } at the
   402	beginning of each epoch. The cluster centroids are defined as:
   403	𝑐𝑖𝑣 =
   404	
   405	1 Õ
   406	𝑓,
   407	|H𝑖𝑣 |
   408	𝑣
   409	
   410	(5)
   411	
   412	𝑓 ∈ H𝑖
   413	
   414	𝑟𝑣
   415	where H𝑖𝑣 = H𝑖𝑣𝑑 ∪ H𝑖𝑟 𝑣 . The set H𝑖𝑣𝑑 comprises features {𝑓 𝑗𝑣 |𝑦 𝑣𝑑
   416	𝑗 = 𝑖} and the set H𝑖 consists of
   417	𝑟
   418	𝑟
   419	𝑣
   420	features {𝑓 𝑗 𝑦
   421	|e𝑗 = 𝑖}. Here | · | indicates the number of instances per cluster.
   422	However, discrepancies in the number of samples across cross-modality clusters can lead
   423	to a bias in the modality-shared centers, favoring the modality with more samples. Furthermore, this initialization strategy assigns equal weight to different samples, despite their varying
   424	representations.
   425	To simultaneously account for both the difference in features and the unequal number of crossmodality samples, we introduce the WMM. For the visible MM, we begin computing the cluster
   426	representation for each modality separately. The modality-shared cluster centroids are then initialized by averaging the two modality-specific cluster centers, thereby mitigating the imbalance in
   427	
   428	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   429	
   430	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '44,830p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    44	1. Introduction
    45	Person re-identiﬁcation (Re-ID) focuses on recognizing and matching
    46	the same individual across diﬀerent camera views. It has gained considerable interest in intelligent surveillance systems because of its significant potential in practical applications. Most supervised Re-ID methods (He et al., 2021; Li et al., 2024; Zheng et al., 2019) leverage humanannotated labels to learn discriminative features and have achieved signiﬁcant progress. However, these methods rely heavily on human annotation, which severely limits the application. Hence, recent studies have
    47	shifted focus to unsupervised Re-ID. Most unsupervised methods (Chen
    48	et al., 2021; Dai et al., 2022; Ge et al., 2020b; Lan et al., 2023; Wang
    49	et al., 2021; Yin et al., 2023; Zou et al., 2023) employ a two-stage alternating training scheme: 1) generating pseudo-labels using Jaccard
    50	distance (Zhong et al., 2017) to measure similarity between all training
    51	samples, followed by a clustering algorithm (Ester et al., 1996; Lloyd,
    52	1982); 2) training the model with a memory bank constructed from
    53	these pseudo-labels. Leveraging this scheme, memory-based contrastive
    54	learning approaches have demonstrated superior performance by utilizing proxies stored within the memory bank.
    55	
    56	Despite signiﬁcant advancements achieved in previous methods,
    57	memory-based contrastive learning framework still faces three inherent
    58	and challenging issues.
    59	Firstly, noise within clustering-generated pseudo-labels propagates
    60	erroneous supervisory signals during training. This arises from substantial cross-camera variations, including diﬀerences in backgrounds,
    61	illumination, perspectives, and other factors. As shown in Fig. 1, in
    62	some cases, camera variations can cause within-camera neighbors of
    63	a query image (from diﬀerent identities) to exhibit higher similarity
    64	than its cross-camera counterparts. Consequently, clustering algorithms
    65	frequently incorrectly group within-camera samples from distinct
    66	identities due to their inﬂated similarity, yet neglect to associate
    67	cross-camera samples sharing the same identity. This leads to numerous
    68	pseudo-labels noise. Secondly, inaccurate cluster proxies negatively impact the contrastive learning system. Previous studies have adopted the
    69	mean feature as the proxy for the cluster centroid, operating under the
    70	assumption that data within each cluster adheres to a high-dimensional
    71	spherical distribution (Chen et al., 2021; Dai et al., 2022; Ge et al.,
    72	2020b; Lan et al., 2023). However, in practice, the feature clusters
    73	formed in the Re-ID feature space are often complex, non-spherical
    74	
    75	∗ Corresponding author.
    76	
    77	E-mail address: sbchen@ahu.edu.cn (S.-B. Chen).
    78	https://doi.org/10.1016/j.neunet.2026.109083
    79	Received 4 August 2025; Received in revised form 20 January 2026; Accepted 8 May 2026
    80	Available online 10 May 2026
    81	0893-6080/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    82	
    83	Neural Networks 203 (2026) 109083
    84	
    85	Z.-H. Li et al.
    86	
    87	after computing DACJ, performing clustering can yield cleaner pseudolabels. Secondly, we propose a Dynamic Adaptive Proxies (DAP) module
    88	integrating two synergistic submodules: the Dynamic Optimal Cluster
    89	Proxies (DOCP) and Dynamic Instance Proxies (DIP). The DOCP constructs the cluster proxy by selecting the medoid—the instance within
    90	each cluster whose sum of distances to all other members is minimal.
    91	By serving as the optimal feature representation, the medoid preserves
    92	the intrinsic data manifold of DBSCAN clustering, diverging from prior
    93	spherical distribution assumptions. At the same time, the DIP treats
    94	each sample as an anchor, performing contrastive learning with all positive and negative instances stored in the instance-level memory bank.
    95	This enables comprehensive exploitation of global information to mitigate within-cluster variations and adaptively reﬁne clustering outcomes.
    96	Thirdly, we introduce a novel dynamic adaptive knowledge distillation
    97	(DAKD) approach that generates high-quality soft labels through knowledge distillation and DAP, using class prediction consistency to improve
    98	robustness against pseudo-label noise. Finally, we integrate two local
    99	features extracted from distinct views (upper and lower body regions)
   100	to capture more discriminative information that may be overlooked by
   101	the global feature.
   102	The key contributions of this work are summarized as follows:
   103	
   104	Fig. 1. Illustration of the camera variations problem. Shapes represent camera
   105	sources, with identical shapes denoting images captured by the same camera,
   106	while colors represent identity labels, with identical colors indicating images of
   107	the same individual.
   108	
   109	due to signiﬁcant variations in pose, viewpoint, and illumination. This
   110	inherent complexity indicates that the data points are actually distributed on a manifold rather than strictly conforming to the traditional
   111	spherical cluster structure (Bengio et al., 2013; Yin et al., 2023).
   112	Speciﬁcally, as illustrated in Fig. 2, real-world unsupervised person
   113	Re-ID data exhibits manifold distributions, which consequently leads to
   114	proxy deviations from the true cluster centers and potential intrusion
   115	into foreign clusters. Concurrently, the momentum update strategy,
   116	by compelling features to converge to historical averages, distorts the
   117	data’s intrinsic manifold structures. This distortion is fundamentally
   118	at odds with the local Euclidean properties upon which DBSCAN
   119	clustering is predicated. This theoretical incompatibility between proxy
   120	modeling and update mechanisms collectively induces feature representation distortion and misguided optimization, severely limiting model
   121	performance ceilings. Thirdly, typical single hard targets (i.e., one-hot
   122	vectors representing predicted labels) are not robust to pseudo-label
   123	noise and can mislead model optimization, amplifying errors during
   124	training.
   125	
   126	•
   127	
   128	We propose a novel dynamic adaptive camera Jaccard distance
   129	(DACJ), which explicitly incorporates camera-speciﬁc distribution
   130	characteristics into similarity measurement. By adaptively adjusting
   131	the weight of sample correlations within the same camera and across
   132	diﬀerent cameras according to the variation in camera conditions,
   133	this approach decreases pseudo-label noise and improves the robustness of feature representation learning.
   134	• We propose a novel dynamic adaptive proxies (DAP) module that
   135	mitigates the negative impact of inaccurate cluster proxies on contrastive learning. Unlike conventional static proxy methods, DAP addresses the error accumulation caused by ﬁxed cluster centroids in
   136	scenarios with imbalanced data distributions.
   137	• We propose a dynamic adaptive knowledge distillation (DAKD) module that generates high-quality soft labels. This design enables adaptive adjustment of knowledge transfer intensity across diﬀerent samples, eﬀectively suppressing error propagation from noisy pseudolabels.
   138	
   139	Therefore, to address the existing challenges, we introduce a new
   140	contrastive learning framework called dynamic adaptive multi-view
   141	contrastive learning (DAMCL). We ﬁrst propose a Dynamic Adaptive
   142	Camera Jaccard (DACJ) distance to dynamically estimate camera variations per training epoch and perform adaptive variation reduction. Crucially, when camera variations for a sample exceed a threshold, DACJ
   143	proportionally adjusts the within-/cross-camera neighbor ratio in its
   144	feature space, ensuring highly informative positive samples from crosscamera sources are eﬀectively included in the relevant neighbors. Thus,
   145	
   146	2. Releted works
   147	This section discusses the most relevant studies from three angles: 1)
   148	unsupervised person Re-ID, encompassing unsupervised domain adaptation (UDA) and fully unsupervised learning (USL) approaches for person
   149	Re-ID; 2) contrastive learning, which develops robust feature representations by leveraging positive and negative sample pairs; and 3) knowledge distillation, which involves techniques to improve model perfor-
   150	
   151	Fig. 2. DBSCAN clustering on two no-spherical distributed datasets. Diﬀerent colors of points represent distinct clusters, and the red pentagram indicates the mean
   152	feature proxy point for each cluster.
   153	2
   154	
   155	Neural Networks 203 (2026) 109083
   156	
   157	Z.-H. Li et al.
   158	
   159	mance by transferring knowledge from a larger model to a more compact one.
   160	
   161	incorporating camera information, eﬀectively mitigating camera variations to improve pseudo-label generation and re-ranking accuracy. However, this method rely on static parameter settings, which constrain
   162	their adaptability to dynamic feature distributions and complex, nonspherical data across training epochs. Our DACJ introduces a dynamic
   163	camera variations degree vector Δ, which continuously estimates and
   164	adapts to camera variations at each epoch. This approach enables DACJ
   165	to produce cleaner pseudo-labels and achieve superior performance, delivering substantial improvements in mean Average Precision (mAP).
   166	Additionally, methods employing memory banks for model training,
   167	which depend on mean proxies, often neglect the challenges posed by
   168	data stream distribution. These limitations can hinder the eﬀectiveness
   169	of model training.
   170	
   171	2.1. Unsupervised person Re-ID
   172	Existing unsupervised person Re-ID methods can be roughly divided
   173	into unsupervised domain adaptation (UDA) methods (Dai et al., 2021;
   174	Ge et al., 2020b; Han et al., 2022a; He et al., 2022; Lee et al., 2023;
   175	Li et al., 2023; Yin et al., 2022) and purely unsupervised learning
   176	(USL) (Cho et al., 2022; Dai et al., 2022; Ji et al., 2024a; Lan et al.,
   177	2023; Ran et al., 2025; Wang et al., 2021; Yin et al., 2023; Yu et al.,
   178	2023; Zhang et al., 2024, 2022; Zou et al., 2023) methods. UDA methods transfer knowledge from a labeled source domain to an unlabeled
   179	target domain, while USL methods train directly on unlabeled target
   180	datasets. The core challenge in UDA Re-ID methods lies in eﬀectively
   181	utilizing a labeled source dataset to enhance model performance on an
   182	unlabeled target dataset. These methods typically rely on the premise
   183	that the diﬀerences between the source and target domains are minimal,
   184	employing transfer learning techniques to address this challenge.
   185	Our approach addresses the more demanding requirements of the
   186	USL framework. In unsupervised person Re-ID frameworks, a selfsupervision paradigm is typically used to leverage unlabeled target data:
   187	clustering algorithms (i.e., K-means Lloyd, 1982, DBSCAN Ester et al.,
   188	1996) partition target domain samples into pseudo-label clusters, followed by joint training strategies that combine these pseudo-labels with
   189	the unlabeled data through iterative optimization cycles (Dai et al.,
   190	2022; Ge et al., 2020b). This cyclical process enables progressive reﬁnement of feature embeddings while mitigating error accumulation from
   191	imperfect cluster assignments.
   192	To enhance the reliability of pseudo-labels, several advanced methods for pseudo-label generation have been developed. MMT (Ge et al.,
   193	2020a) enhances pseudo-labels by training two neural networks with
   194	combined supervision from oﬄine reﬁned hard pseudo-labels and online reﬁned soft pseudo-labels. SPCL (Ge et al., 2020b) proposed a selfpaced approach that incrementally builds more reliable clusters to reﬁne
   195	hybrid memory and learning objectives.
   196	To tackle inconsistencies in memory updates, CCL (Dai et al.,
   197	2022) developed a novel cluster contrast learning framework using a
   198	cluster-level memory dictionary, achieving remarkable performance.
   199	ISE (Zhang et al., 2022) employs a progressive linear interpolation strategy to create support samples from real samples and adjacent clusters in
   200	the embedding space, enabling more precise cluster representations by
   201	capturing underlying information. In Li et al. (2022), a self-guided hard
   202	negative generation approach is introduced, utilizing adversarial training between a hard negative generation network and a re-ID network
   203	to mutually improve their performance. In Han et al. (2022b), a group
   204	sampling approach is introduced to mitigate overﬁtting by minimizing
   205	the negative inﬂuence of individual samples on statistical stability. In
   206	GMT (Cai et al., 2025), a balancing multi-task oriented fusing method is
   207	introduced to explicitly enhance Re-ID sub-task learning through additional Re-ID training and target-guided multi-model parameters fusion.
   208	Recent works have achieved signiﬁcant progress in multi-view clustering tasks. Among them, BWIC-TIMC (Yao et al., 2025) jointly exploits
   209	both inter-view and intra-view information and employs an adaptive
   210	fused graph learning strategy, demonstrating remarkable eﬀectiveness.
   211	IMVC-TGR (Wang et al., 2025) separates redundant information in each
   212	aﬃnity graph through graph reﬁnement operations, stacks the feature
   213	graphs into a tensor, and introduces semantic consistency constraints
   214	along with a self-weighted fusion strategy, which not only maintains
   215	inter-view consistency but also signiﬁcantly enhances the quality of
   216	the feature graphs. Recent approaches have tackled camera variations
   217	in unsupervised person Re-ID with innovative methods. CGMAL (Ran
   218	et al., 2025) constructing a heterogeneous graph to model semantic
   219	relationships, including positive pairs across cameras (same identity)
   220	and negative pairs within cameras (diﬀerent identities). Similarly, CAJaccard (Chen et al., 2024) enhances the traditional Jaccard distance by
   221	
   222	2.2. Contrastive learning
   223	Contrastive learning (Chen et al., 2020; He et al., 2020; Wang & Qi,
   224	2022), a leading self-supervised learning approach, focuses on maximizing the similarity between representations derived from various augmented or distorted versions of a sample while minimizing similarity
   225	with representations of other samples in the latent space. CLSA (Wang
   226	& Qi, 2022) introduces a contrastive learning framework that combines
   227	weakly and strongly augmented images using distribution divergence
   228	minimization, enhancing the performance of self-supervised learning.
   229	In SimCLR (Chen et al., 2020), it directly extract representations from
   230	a large batch. In MoCo (He et al., 2020), a novel contrastive learning framework is introduced, featuring a momentum-based encoder
   231	and a queue-structured memory bank to dynamically update negative
   232	samples, enhancing the eﬀectiveness of contrastive learning. Distinct
   233	from MoCo’s approach, our proposed method simultaneously performs
   234	cluster-level and instance-level contrastive learning through our innovative DAP.
   235	2.3. Knowledge distillation
   236	Knowledge distillation aims to transfer knowledge from a teacher
   237	network to a smaller student network. Recently, there has been a focus on self-knowledge distillation, where both networks share the same
   238	structure (Lan et al., 2023; Tian et al., 2020; Ye et al., 2020, 2021; Yun
   239	et al., 2020). These methods leveraged outputs from a structurally identical teacher. In Tian et al. (2020), a straightforward yet eﬀective baseline for few-shot learning was proposed, focusing on minimizing the
   240	loss tied to the class probability distribution generated by the teacher
   241	model. CS-KD (Yun et al., 2020) introduced an innovative regularization
   242	approach that aligns distribution predictions across samples within the
   243	same class. Meanwhile, HDCPD (Cheng et al., 2022) aims to harmonize
   244	the probability distribution between the network and the teacher network, which is updated using the Exponential Moving Average (EMA)
   245	method. CORE (Ye et al., 2021) jointly reﬁnes networks and label predictions by leveraging knowledge distillation from peer networks, relying
   246	on minimal and possibly imprecise annotations. Additionally, a collaborative ensemble learning scheme in Ye et al. (2020) utilizes relationships
   247	among classiﬁers for cross-modality person re-identiﬁcation, enhancing
   248	discriminability through consistent ensemble outputs. Although knowledge distillation has led to signiﬁcant progress, these methods may not
   249	entirely mitigate the susceptibility of traditional hard labels to pseudolabel noise.
   250	3. Method
   251	3.1. Overview
   252	Given a person Re-ID dataset 𝑃 = {𝑥𝑖 }𝑁
   253	, where 𝑁 represents the
   254	𝑖=1
   255	total number of images and 𝑥𝑖 denotes the 𝑖th image, the goal is to train
   256	a robust network 𝑓𝜃 that maps each sample 𝑥𝑖 to a corresponding feature
   257	vector 𝑿 𝑖 = 𝑓𝜃 (𝑥𝑖 ) in the embedding space  . Additionally, each image
   258	3
   259	
   260	Neural Networks 203 (2026) 109083
   261	
   262	Z.-H. Li et al.
   263	
   264	Fig. 3. An overview framework of the proposed DAMCL. It alternates between two steps. (a) In the clustering step, the encoder ﬁrst extracts the global feature map
   265	sets. Horizontal splitting and GEM pooling are then applied to obtain feature map sets for the upper and lower views, followed by computing and fusing the DACJ
   266	distances across these three views for clustering. Using the generated pseudo-labels and multi-view features, it initializes global DIP and three types of DOCPs. (b)
   267	In the training step, we exploit three DOCPs and global-DIP to perform contrastive learning. During the training process, the teacher model’s knowledge is utilized
   268	to guide the student model.
   269	
   270	𝑥𝑖 is paired with a camera label 𝑐𝑖 .
   271	As shown in Fig. 3, in the clustering step, the encoder 𝑓𝜃 is used to
   272	extract the global feature map sets 𝑀𝑔𝑏 . These feature maps are then
   273	horizontally split into upper and lower parts, denoted as 𝑀𝑢𝑝 and 𝑀𝑙𝑤 ,
   274	respectively. Subsequently, GEM pooling is applied to obtain the corresponding feature vector sets {𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 }. We then calculate DACJ
   275	distances for three views, which are denoted as 𝐷𝑔𝑏 , 𝐷𝑢𝑝 and 𝐷𝑙𝑤 . Subsequently, a re-weighted distance matrix 𝐷𝑐𝑝 is computed, which is then
   276	used to generate pseudo-labels {𝑦𝑖 𝑠} via the DBSCAN clustering algorithm, according to the following function:
   277	(1)
   278	
   279	sample pairs. First, the camera label relationship matrix 𝑀 is constructed using the camera IDs in the dataset to indicate whether samples 𝑥𝑖 and 𝑥𝑗 share the same camera label or not. Next, based on 𝐷
   280	and 𝑀, we compute the k-nearest neighbors (KNNs) 𝐾, as well as the
   281	within-camera KNNs 𝐾𝐼 and the cross-camera KNNs 𝐾𝐶 for each example. Finally, we introduce the camera variations degree vector Δ to
   282	estimate the diﬀerences in feature distributions between cross-camera
   283	and within-camera sample pairs. The calculation process for Δ is as
   284	follows:
   285	1 ∑
   286	1 ∑
   287	Δ𝑖 =
   288	𝑑 −
   289	𝑑 ,
   290	(2)
   291	𝑘1 𝑗∈𝐾 (𝑖) 𝑖,𝑗 𝑘1 𝑗∈𝐾 (𝑖) 𝑖,𝑗
   292	
   293	where 𝜇 is the balancing factor. Using {𝑦𝑖 𝑠} and {𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 }, we initialize global DIP and three types of DOCP.
   294	During training, we feed mini-batch images into the encoder to extract the features of the three views. These features are then used with
   295	the three DOCPs and DIP for contrastive learning. Importantly, the
   296	teacher model continuously supervises the training of the student model
   297	in an oﬄine manner throughout the process.
   298	
   299	where 𝑘1 is the number of nearest neighbors, Δ𝑖 means camera variations degree of sample 𝑥𝑖 and 𝑑𝑖,𝑗 represents the feature distance between
   300	samples 𝑥𝑖 and 𝑥𝑗 . During model optimization, our Δ is continuously and
   301	dynamically updated to reﬂect the latest changes.
   302	Dynamic adaptive camera k-reciprocal nearest neighbors. For
   303	sample 𝑥𝑖 , we query Δ𝑖 . If Δ𝑖 is greater than 0, we calculate the dynamic
   304	adaptive camera k-reciprocal nearest neighbors(DAC-KRNNs) of 𝑥𝑖 :
   305	
   306	𝐷𝑐𝑝 = 𝜇𝐷𝑢𝑝 + 𝜇𝐷𝑙𝑤 + (1 − 𝜇)𝐷𝑔𝑏 ,
   307	
   308	𝐶
   309	
   310	𝐼
   311	
   312	𝑅(𝑥𝑖 , 𝑘𝑐1 , 𝑘𝑖1 ) = Unique(
   313	
   314	3.2. Dynamic adaptive camera jaccard
   315	
   316	{𝑥𝑗 |𝑥𝑗 ∈ 𝐾𝐶 (𝑖)[∶ 𝑘𝑐1 ] ∩ 𝑥𝑖 ∈ 𝐾𝐶 (𝑗)[∶ 𝑘𝑐1 ]}∪
   317	
   318	(3)
   319	
   320	{𝑥𝑗 |𝑥𝑗 ∈ 𝐾𝐼 (𝑖)[∶ 𝑘𝑖1 ] ∩ 𝑥𝑖 ∈ 𝐾𝐼 (𝑗)[∶ 𝑘𝑖1 ]}),
   321	
   322	In unsupervised Re-ID, the Jaccard distance is a commonly employed
   323	metric, particularly for clustering tasks. However, camera variations often introduce numerous high-weight within-camera negative samples
   324	into the relevant neighbors while impeding the inclusion of informative
   325	cross-camera positive samples. As a result, relying on the traditional
   326	camera-unaware Jaccard distance causes clustering algorithms to prioritize connections between similar within-camera sample pairs of diﬀerent identities, while failing to group cross-camera samples of the same
   327	identity. To address these challenges and reﬁne the Jaccard distance
   328	before clustering to reduce pseudo-label noise, we introduce the DACJ
   329	distance.
   330	The original distance matrix 𝐷 is computed based on either cosine
   331	similarity or Euclidean distance metrics.
   332	Dynamic adaptive camera variations estimation. To reduce the
   333	impact of camera variations, it is essential to precisely assess the diﬀerences in feature distributions between cross-camera and within-camera
   334	
   335	where Unique(⋅) denotes removing duplicate elements, 𝑘𝑐1 and 𝑘𝑖1 mean
   336	diﬀerent k are used in 𝐾𝑐 and 𝐾𝑖 , 𝑘𝑐1 + 𝑘𝑖1 = 𝑘1 . To reduce the number
   337	of within-camera negative samples, we increase the proportion of crosscamera positive samples in DAC-KRNNs by using a larger 𝑘𝑐1 .
   338	Vectorization of neighbors. After the DAC-KRNNs of sample 𝑥𝑖 are
   339	obtained, they need to be encoded into a weighted DAC-KRNNs vector
   340	𝑉𝑖 = [𝑉𝑖,1 , 𝑉𝑖,2 , …, 𝑉𝑖,𝑁 ]. The calculation formula is as follows:
   341	−𝑑𝑖,𝑗
   342	
   343	⎧ ∑𝑒 −𝑑
   344	𝑒 𝑖,𝑘
   345	⎪
   346	𝑉𝑖,𝑗 = ⎨ 𝑥𝑘 ∈𝑅
   347	⎪
   348	⎩0
   349	
   350	if 𝑥𝑗 ∈ 𝑅
   351	(4)
   352	otherwise,
   353	
   354	where 𝑅 is the DAC-KRNNs of sample 𝑥𝑖 .
   355	Dynamic adaptive camera local query expansion. Due to camera variations, a large number of within-camera negative samples are
   356	4
   357	
   358	Neural Networks 203 (2026) 109083
   359	
   360	Z.-H. Li et al.
   361	
   362	𝑐 ∈ ℝ𝐶×𝑑 contains 𝐶 entries, with 𝑑 representing the feature dimension. Selecting the medoid rather than the centroid (mean) as the cluster
   363	proxy is a deliberate design choice to better cope with the complexity
   364	and noise inherent in the feature space of unsupervised person Re-ID.
   365	First, as noted in Section 1, due to signiﬁcant variations in pose,
   366	viewpoint, illumination, and cross-camera discrepancies, Re-ID feature
   367	clusters often exhibit complex, non-spherical. In such irregularly shaped
   368	distributions, the centroid which is computed as the arithmetic mean,
   369	tends to fall into low-density “empty” regions, deviating substantially
   370	from the true data manifold. By contrast, the medoid is an actual data
   371	point that lies on the manifold itself, enabling it to more faithfully represent the complex geometric structure captured by density-based clustering methods such as DBSCAN.
   372	Second, pseudo-labels in unsupervised Re-ID are inevitably noisy,
   373	containing misclassiﬁcations and outliers primarily caused by crosscamera appearance gaps. The centroid (mean) is highly sensitive to these
   374	outliers because averaging incorporates every sample equally, allowing
   375	extreme points to disproportionately shift the proxy toward erroneous
   376	regions. The medoid, deﬁned as the point that minimizes the total distance to all other points in the cluster, exhibits strong robustness to outliers and thus provides a more stable and reliable proxy for contrastive
   377	learning.
   378	Third, since DBSCAN deﬁnes clusters based on local density and local
   379	Euclidean distances, using a synthetic centroid (especially when updated
   380	via momentum) risks causing progressive drift of the proxy features
   381	away from the original data manifold, ultimately distorting the very
   382	local structure that DBSCAN has discovered. By adopting the medoid,
   383	which is an actual sample feature, we preserve the intrinsic geometric
   384	properties of the feature space, ensuring strong consistency between the
   385	optimization objective of contrastive learning and the underlying manifold structure revealed by clustering.
   386	Additionally, selecting the medoid as the proxy is well-justiﬁed.
   387	As shown in Fig. 5, compared to the traditional mean centroid proxy
   388	strategy, our proposed medoid-based optimal proxy strategy exhibits a
   389	steeper and faster loss decline during the training phase, which directly
   390	demonstrates that the DOCP module can signiﬁcantly accelerate convergence. Moreover, with minimal diﬀerence in the ﬁnal loss values, our
   391	proxy strategy displays reduced ﬂuctuations and oscillations throughout the training process, thereby proving its superior robustness against
   392	pseudo-label noise and outliers.
   393	Updating the memory bank during back-propagation is crucial. To
   394	preserve the intrinsic manifold structure of the data and maintain the in-
   395	
   396	introduced into KRNNs, making LQE generate the unreliable weighted
   397	expanded neighbors vector 𝑉𝑖𝑒 of sample 𝑥𝑖 , which reassign high weights
   398	to these negative samples. So we propose a dynamic adaptive camera
   399	local query expansion(DAC-LQE) to boost the reliability of vectors:
   400	⎧⎛
   401	⎞
   402	∑
   403	∑
   404	⎪⎜
   405	𝑉𝑗 +
   406	𝑉𝑘 ⎟
   407	⎪ ⎜𝑥𝑗 ∈𝐾𝐶 (𝑖)[∶𝑘𝑐 ]
   408	⎟
   409	𝑥𝑘 ∈𝐾𝐼 (𝑖)[∶𝑘𝑖2 ]
   410	2
   411	⎠
   412	⎪⎝
   413	𝑉𝑖𝑒 = ⎨
   414	𝑐 + 𝑘𝑖
   415	𝑘
   416	2
   417	2
   418	⎪
   419	∑
   420	⎪1
   421	𝑉𝑗
   422	⎪ 𝑘2
   423	⎩ 𝑥𝑗 ∈𝐾(𝑖)[∶𝑘2 ]
   424	
   425	if Δ𝑖 > 0
   426	(5)
   427	otherwise,
   428	
   429	where 𝑘𝑐2 + 𝑘𝑖2 = 𝑘2 , 𝑉𝑗 and 𝑉𝑘 are the weighted DAC-KRNNs vectors of
   430	𝑥𝑗 and 𝑥𝑘 .
   431	Overlap computation. The DACJ 𝐷𝑖,𝑗 between 𝑥𝑖 and 𝑥𝑗 is computed as follows:
   432	𝑁
   433	∑
   434	
   435	𝐷𝑖,𝑗 = 1 −
   436	
   437	𝑘=1
   438	𝑁
   439	∑
   440	𝑘=1
   441	
   442	(
   443	)
   444	𝑒 ,𝑉 𝑒
   445	min 𝑉𝑖,𝑘
   446	𝑗,𝑘
   447	(
   448	)
   449	𝑒 ,𝑉 𝑒
   450	max 𝑉𝑖,𝑘
   451	𝑗,𝑘
   452	
   453	.
   454	
   455	(6)
   456	
   457	By dynamically estimating the camera variations for each instance, our
   458	DACJ eﬀectively incorporates more cross-camera positive samples, offering greater reliability and robustness compared to the traditional Jaccard distance.
   459	3.3. Dynamic adaptive proxies
   460	As mentioned in Section 1, we argue that using mean features as
   461	proxies for cluster centroids may adversely aﬀect memory-based contrastive learning.
   462	Thus, we propose a DAP module, which includes DOCP and DIP.
   463	Dynamic optimal cluster proxies. For each cluster, we employ dynamic optimization to select a medoid that serves as the proxy, which
   464	is then stored in the memory bank 𝑐 . Speciﬁcally, as shown in Fig. 4,
   465	for the proxy of the 𝑗th cluster, we initialize by identifying the index
   466	𝑖𝑑𝑥𝑗 corresponding to the optimal sample within the cluster:
   467	(
   468	)
   469	∑
   470	𝑖𝑑𝑥𝑗 = arg min
   471	‖𝑿 𝑖 − 𝑿 𝑘 ‖2 ,
   472	(7)
   473	𝑖
   474	
   475	𝑿 𝑘 ∈𝑗 ,𝑿 𝑘 ≠𝑿 𝑖
   476	
   477	where 𝑗 denotes the 𝑗th cluster and 𝑿 𝑖 is a sample from 𝑗 . Consequently, the proxy for the 𝑗th cluster is 𝑿 𝑖𝑑𝑥𝑗 , and the memory bank
   478	
   479	Fig. 4. The construction process of DOCP. The SUM(𝑥𝑛 ) represents the sum of distances between the feature point 𝑥𝑛 and other data points within the cluster.
   480	5
   481	
   482	Neural Networks 203 (2026) 109083
   483	
   484	Z.-H. Li et al.
   485	
   486	Fig. 5. Loss curves under diﬀerent cluster proxy selection strategies. Optimal represents our proposed medoid-based optimal proxy, while Mean represents the
   487	traditional mean centroid proxy. Compared to the conventional centroid proxy, the medoid-based optimal proxy signiﬁcantly accelerates the convergence speed and
   488	notably improves the stability of the optimization process.
   489	
   490	Fig. 6. The t-SNE visualization of 10 randomly selected classes from the MSMT17 test set is presented. Diﬀerent colors represent distinct identities, while diﬀerent
   491	shapes correspond to camera labels.
   492	
   493	where 𝜏 is a temperature hyper-parameter that plays a crucial role in
   494	controlling the scale of the similarities.
   495	Dynamic instance proxies. Since pseudo-label clusters generated
   496	by DBSCAN may contain samples with signiﬁcant appearance variations (e.g., the same identity under diﬀerent clothing scenarios), solely
   497	adopting DOCP-based contrastive learning would force samples to align
   498	narrowly with cluster centroids, failing to capture ﬁne-grained interinstance correlations and address within-cluster feature dispersion. Furthermore, in unsupervised training frameworks, continuous parameter
   499	updates of the model may induce non-stationary drift in the feature
   500	space (feature drift), resulting in cross-epoch inconsistency of pseudolabels. Therefore, we additionally propose DIP on the basis of DOCP.
   501	We establish an instance memory bank 𝑖 ∈ ℝ𝑁×𝑑 , which stores the
   502	features of all images in the training set, with 𝑁 entries and 𝑑 representing the feature dimension. Consistent with DOCP, we forgo the momentum update strategy and, in each iteration, directly replace the 𝑗th entry
   503	in 𝑖 with the corresponding feature 𝑿 𝑗 from the current mini-batch
   504	:
   505	
   506	tegrity of the optimal sample features, we abandon the conventional momentum update strategy and instead directly adopt the optimal sample
   507	feature. Speciﬁcally, we select one sample from each mini-batch pseudocluster using Eq. (7). Supposed that for the cluster with pseudo-label 𝑗,
   508	the sample we select is 𝑿 𝑜𝑝𝑡𝑗 . Then, we directly update the 𝑗th entry in
   509	the memory bank 𝑐 with 𝑿 𝑜𝑝𝑡𝑗 in each iteration. The formulation is
   510	as follows:
   511	𝑐 [𝑗] ← 𝑿 𝑜𝑝𝑡𝑗 .
   512	
   513	(8)
   514	
   515	This critical simpliﬁcation guarantees that all proxy features stored in
   516	the memory bank originate from real sample embeddings, rather than
   517	artiﬁcial mean proxy representations. It not only considers the diversity
   518	of the real data distribution but also preserves the original distribution
   519	of the sample features. In this way, we ensure that the optimal proxies
   520	for the current phase are stored in the 𝑐 during each epoch, while
   521	discarding the momentum updating strategy to align with the DBSCAN.
   522	We then apply the DOCP loss function on the cluster memory bank
   523	𝑐 to ensure that the sample is close to its respective cluster proxy
   524	while maintaining distance from other proxies. This can be formulated
   525	as:
   526	(
   527	)
   528	exp 𝑿 𝑇𝑖𝑑𝑥 ⋅ 𝑐 [𝑗]∕𝜏
   529	𝑗
   530	𝑑𝑜𝑐𝑝 = − log ∑𝐶
   531	(9)
   532	( 𝑇
   533	),
   534	𝑘=1 exp 𝑿 𝑖𝑑𝑥 ⋅ 𝑐 [𝑘]∕𝜏
   535	
   536	𝑖 [𝑗] ← 𝑿 𝑗 ,
   537	
   538	(10)
   539	
   540	where 𝑖 [𝑗] is the 𝑗th entry of 𝑖 . In this way, each iteration directly
   541	updates the memory bank by replacing outdated features with the current ones Eq. (10), ensuring that contrastive learning consistently oper-
   542	
   543	𝑗
   544	
   545	6
   546	
   547	Neural Networks 203 (2026) 109083
   548	
   549	Z.-H. Li et al.
   550	
   551	Fig. 7. Parameter analysis of 𝑘𝑖1 , 𝑘𝑐1 , 𝑘𝑖2 , 𝑘𝑐2 , 𝜇 on Market1501, DukeMTMC-reID and MSMT17 and parameter analysis of 𝛼 and 𝛽 on MSMT17.
   552	
   553	ates on the most up-to-date features generated by the latest model. With
   554	DIP, we form a DIP loss as follows:
   555	( 𝑇
   556	)
   557	∑
   558	𝑦𝑙 =𝑦𝑗 exp 𝑿 𝑗 ⋅ 𝑖 [𝑙]∕𝜏
   559	𝑑𝑖𝑝 = − log ∑𝑁
   560	(11)
   561	( 𝑇
   562	),
   563	𝑘=1 exp 𝑿 𝑗 ⋅ 𝑖 [𝑘]∕𝜏
   564	
   565	Speciﬁcally, the teacher model is trained using cluster contrastive
   566	learning along with our proposed DACJ and DAP modules, incorporating the loss functions deﬁned in Eqs. (9) and (11). More details about
   567	the training process of the teacher model can be found in Section 3.6.
   568	Once the teacher model is fully trained, we proceed to apply dynamic
   569	adaptive knowledge distillation on the student model, a process that
   570	consists of two stages. More details about these stages can be found in
   571	Section 3.6.
   572	Since the initialized student model exhibits poor performance on person Re-ID tasks, its early pseudo-labels tend to contain substantial noise,
   573	which may lead to biased feature representations. To mitigate this issue,
   574	in the ﬁrst stage, we directly leverage the well-trained teacher model
   575	to generate pseudo-labels and initialize memory banks using its feature
   576	vectors (as shown in Eq. (5)). Subsequently, the student model is trained
   577	for a short period using these teacher-generated pseudo-labels and ﬁxed
   578	cluster proxy. Given the signiﬁcant domain gap between the ImageNet
   579	pre-training dataset and person Re-ID datasets, the student model encounters more severe label noise during the initial training phase. Therefore, this stage enables the student model to rapidly acquire knowledge
   580	from the teacher model, thereby generating more accurate pseudo-labels
   581	in the early stages of training. During the second training stage, the student model is used to generate soft pseudo-labels and update the model
   582	parameters under the guidance of the teacher model. The query image
   583	samples 𝑥𝑖 𝑠 are passed through the student encoder network to obtain
   584	the feature representations 𝑿 𝑖 𝑠. We then apply the softmax to the distance between these feature representations and each cluster proxy to
   585	obtain soft labels 𝑦̂𝑖 𝑠, which can be formulated as:
   586	(
   587	)
   588	exp − ‖𝑿 𝑖 − 𝑐 [𝑗]‖2
   589	𝑦̂𝑖 [𝑗] = ∑𝐶
   590	(12)
   591	(
   592	),
   593	𝑘=1 exp − ‖𝑿 𝑖 − 𝑐 [𝑘]‖2
   594	
   595	where 𝑦𝑙 and 𝑦𝑗 are pseudo labels of samples 𝑋𝑙 and 𝑋𝑗 . The Eq. (11)
   596	incorporates all samples, leveraging comprehensive global information
   597	to eﬀectively mitigate within-cluster variations.
   598	Discussion. Building upon the aforementioned DOCP and DIP modules, we perform periodic global reinitialization and updating of proxy
   599	features in both 𝑐 and 𝑖 at the start of each training epoch. As
   600	model parameters progressively optimize, the feature space dynamically evolves. This periodic reinitialization and updating mechanism ensures memory features are consistently derived from the currently optimal model, thereby eﬀectively eliminating potential interference from
   601	outdated proxy features during contrastive learning.
   602	Additionally,
   603	the synergistic integration of DOCP and DIP achieves complementary
   604	enhancement without redundancy. We maintain a core principle that
   605	each image should align its features with all samples within its assigned
   606	pseudo-cluster. To this end, DOCP ensures samples are closely aligned
   607	with their cluster proxy while repelling foreign proxies using DBSCANgenerated pseudo-labels, while DIP enhances DBSCAN’s clustering by
   608	leveraging global sample relationships.
   609	3.4. Dynamic adaptive knowledge distillation
   610	Traditional hard pseudo-targets, such as one-hot label vectors, often
   611	lacks robustness against pseudo-label noise. This can result in inaccurate
   612	guidance for model optimization and may amplify errors as training progresses. Thus, we propose a DAKD, which adopts knowledge distillation
   613	to reﬁne feature learning. We designate the model trained on the same
   614	dataset as the teacher model and a new ImageNet (Deng et al., 2009)
   615	pre-trained initialized model as the student model, requiring no additional information.
   616	
   617	where 𝑦̂𝑖 [𝑗] is a softmax probability between 𝑿 𝑖 and the 𝑗th entry in
   618	𝑐 . Given the pseudo-label 𝑦𝑖 , which is one-hot vector obtained by extracting feature from 𝑥𝑖 with teacher model and then clustering, the soft
   619	label 𝑦̂𝑖 obtained via Eq. (12) is encouraged to be close to 𝑦𝑖 . Thus, the
   620	7
   621	
   622	Neural Networks 203 (2026) 109083
   623	
   624	Z.-H. Li et al.
   625	
   626	DAKD loss can be expressed as:
   627	𝑑𝑎𝑘𝑑 = −
   628	
   629	||
   630	∑
   631	
   632	𝑦𝑖 ⋅ log 𝑦̂𝑖 ,
   633	
   634	Algorithm 2: Training process of the student model.
   635	Require: Unlabeled training data 𝑃 ; initialize the encoder 𝑓𝜃
   636	with ImageNet-pretrained ResNet-50; the teacher
   637	encoder 𝑓̃𝜃 trained on the unlabeled training data 𝑃
   638	using Algorithm 1; balancing factors 𝜇, 𝛼 and 𝛽 for
   639	Eqs. (1), (15) and (17); hyper-parameter 𝑘𝑐1 and 𝑘𝑖1 for
   640	Eq. (3); hyper-parameter 𝑘𝑐2 and 𝑘𝑖2 for Eq. (5);
   641	temperature hyper-parameter 𝜏 for Eqs. (9) and (11)
   642	// first stage
   643	{
   644	}
   645	Extract feature vector sets 𝐹̃𝑔𝑏 , 𝐹̃𝑢𝑝 , 𝐹̃𝑙𝑤 from 𝑃 by 𝑓̃𝜃 ;
   646	Compute dynamic adaptive camera Jaccard distance matrixs
   647	{
   648	}
   649	𝐷̃ 𝑔𝑏 , 𝐷̃ 𝑢𝑝 , 𝐷̃ 𝑙𝑤 with Eq. (6);
   650	{
   651	}
   652	Clustering 𝐹̃𝑔𝑏 , 𝐹̃𝑢𝑝 , 𝐹̃𝑙𝑤 from 𝑃 by 𝑓̃𝜃 into 𝐶 clusters with Eq.
   653	(1) and DBSCAN;
   654	Initialize four memory banks individually with DAP ;
   655	for 𝑖 in [1, 𝑛𝑢𝑚_𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠 × 2] do
   656	Sample 𝑃 × 𝐾 query images from 𝑃 ;
   657	Compute objective function with Eq. (17) ;
   658	end
   659	// second stage
   660	for 𝑛 in [1,num_epochs] do
   661	{
   662	}
   663	Extract feature vector sets 𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 from 𝑃 by 𝑓𝜃 ;
   664	Compute dynamic adaptive camera Jaccard distance
   665	{
   666	}
   667	matrixs 𝐷𝑔𝑏 , 𝐷𝑢𝑝 , 𝐷𝑙𝑤 with Eq. (6);
   668	{
   669	}
   670	Clustering 𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 into 𝐶 clusters with Eq. (1) and
   671	DBSCAN;
   672	Initialize four memory banks individually with DAP ;
   673	for 𝑖 in [1,num_iterations] do
   674	Sample 𝑃 × 𝐾 query images from 𝑃 ;
   675	Obtain soft pseudo-labels with Eq. (12) ;
   676	Compute objective function with Eq. (19) ;
   677	Update feature representations with Eqs. (8) and (10);
   678	end
   679	end
   680	
   681	(13)
   682	
   683	𝑖=1
   684	
   685	where || denotes the number of samples in the mini-batch . Finally,
   686	we introduce a regularization term to improve the stability of the features and ensure that the student model remains robust across various
   687	scenarios. This allows the DAKD loss to be optimized as follows:
   688	𝑑𝑎𝑘𝑑 = −
   689	
   690	||
   691	∑
   692	
   693	2
   694	‖ 𝑿̂ 𝑖
   695	𝑿𝑖 ‖
   696	‖
   697	𝑦𝑖 ⋅ log 𝑦̂𝑖 + ‖
   698	‖ ̂ − ‖𝑿 ‖ ‖ ,
   699	‖ ‖𝑿 𝑖 ‖
   700	𝑖 ‖2
   701	𝑖=1
   702	
   703	(14)
   704	
   705	where 𝑿̂ 𝑖 and 𝑿 𝑖 are the feature vectors of query 𝑥𝑖 in the teacher model
   706	and student model, respectively.
   707	3.5. Overall loss
   708	According to Eqs. (9) and (11), the cluster proxies loss 𝑡𝑒𝑎
   709	𝑐𝑝 and instance proxies loss 𝑖𝑝 are deﬁned as follows:
   710	𝑔𝑏
   711	𝑢𝑝
   712	𝑙𝑤
   713	𝑡𝑒𝑎
   714	𝑐𝑝 = (1 − 𝛼)𝑑𝑜𝑐𝑝 + 𝛼(𝑑𝑜𝑐𝑝 + 𝑑𝑜𝑐𝑝 ),
   715	
   716	(15)
   717	
   718	𝑖𝑝 = 𝑑𝑖𝑝 ,
   719	
   720	(16)
   721	
   722	where 𝛼 is the loss weight to balance the importance between global and
   723	local features. {𝑔𝑏
   724	, 𝑢𝑝
   725	, 𝑙𝑜
   726	} are the DOCP losses corresponding
   727	𝑑𝑜𝑐𝑝
   728	𝑑𝑜𝑐𝑝
   729	𝑑𝑜𝑐𝑝
   730	to the global, upper and lower views. Thus, the loss function of teacher
   731	model is:
   732	𝑡𝑒𝑎 = 𝑡𝑒𝑎
   733	𝑐𝑝 + 𝛽𝑖𝑝 ,
   734	
   735	(17)
   736	
   737	where 𝛽 controls the degree of two loss functions.
   738	For the student model, the 𝑠𝑡𝑢
   739	𝑐𝑝 is deﬁned as follows:
   740	𝑔𝑏
   741	𝑔𝑏
   742	𝑢𝑝
   743	𝑢𝑝
   744	𝑙𝑤
   745	𝑙𝑤
   746	𝑠𝑡𝑢
   747	𝑐𝑝 = (1 − 𝛼)(𝑑𝑜𝑐𝑝 + 𝑑𝑎𝑘𝑑 ) + 𝛼(𝑑𝑜𝑐𝑝 + 𝑑𝑎𝑘𝑑 + 𝑑𝑜𝑐𝑝 + 𝑑𝑎𝑘𝑑 ),
   748	
   749	(18)
   750	
   751	where {𝑔𝑏
   752	, 𝑢𝑝
   753	, 𝑙𝑜
   754	} are the DAKD losses as in Eq. (14) corre𝑑𝑎𝑘𝑑
   755	𝑑𝑎𝑘𝑑
   756	𝑑𝑎𝑘𝑑
   757	sponding to the global, upper and lower views. The instance proixes is
   758	not change. Thus, the loss function of student model is :
   759	𝑠𝑡𝑢 = 𝑠𝑡𝑢
   760	𝑐𝑝 + 𝛽𝑖𝑝 .
   761	
   762	Training Process of the Teacher Model: The detailed training process of the teacher model is shown in Algorithm 1. Given the unlabeled
   763	dataset 𝑃 and the encoder 𝑓𝜃 initialized with ResNet-50 parameters pretrained on ImageNet (Deng et al., 2009), the training procedure is as follows. In each epoch, the encoder 𝑓𝜃 is used to extract the global feature
   764	map set 𝑀𝑔𝑏 . These feature maps are then horizontally split into upper
   765	and lower parts, denoted as 𝑀𝑢𝑝 and 𝑀𝑙𝑤 , respectively. Subsequently,
   766	GEM pooling is applied to obtain the corresponding feature vector sets
   767	{𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 }. Based on the feature vector sets, we apply the DACJ
   768	module to compute the DACJ distances for the three views, denoted as
   769	{𝐷𝑔𝑏 , 𝐷𝑢𝑝 , 𝐷𝑙𝑤 }. Subsequently, pseudo-labels are generated by applying
   770	Eq. (1) in conjunction with the DBSCAN clustering algorithm. Finally,
   771	the DAP module is employed to initialize three cluster-level memory
   772	banks and one instance-level memory bank. Once the pseudo-labels and
   773	memory banks are prepared, model training begins. Speciﬁcally, in each
   774	iteration, a mini-batch of 𝑝 × 𝑘 query images is sampled from 𝑃 (where 𝑝
   775	represents the number of selected identities, and 𝑘 denotes the number
   776	of images per identity), and the model parameters are updated according to Eq. (17). Finally, the features stored in the memory banks are
   777	updated using Eqs. (8) and (10).
   778	Training Process of the Student Model: The detailed training process of the teacher model is shown in Algorithm 2. Apart from the unlabeled dataset 𝑃 and the encoder 𝑓𝜃 initialized with ResNet-50 parameters pretrained on ImageNet, a teacher model trained via Algorithm 1
   779	is also needed. The student model’s training procedure comprises two
   780	stages.
   781	
   782	(19)
   783	
   784	3.6. Training process
   785	
   786	Algorithm 1: Training process of the teacher model.
   787	Require: Unlabeled training dataset 𝑃 ; initialize the encoder 𝑓𝜃
   788	with ImageNet-pretrained ResNet-50; balancing
   789	factors 𝜇, 𝛼 and 𝛽 for Eqs. (1),(15) and (17);
   790	hyper-parameter 𝑘𝑐1 and 𝑘𝑖1 for Eq. (3);
   791	hyper-parameter 𝑘𝑐2 and 𝑘𝑖2 for Eq. (5); temperature
   792	hyper-parameter 𝜏 for Eqs. (9) and (11)
   793	for 𝑛 in [1,num_epochs] do
   794	{
   795	}
   796	Extract feature vector sets 𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 from 𝑃 by 𝑓𝜃 ;
   797	Compute dynamic adaptive camera Jaccard distance
   798	{
   799	}
   800	matrixs 𝐷𝑔𝑏 , 𝐷𝑢𝑝 , 𝐷𝑙𝑤 with Eq. (6);
   801	{
   802	}
   803	Clustering 𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 into 𝐶 clusters with Eq. (1) and
   804	DBSCAN;
   805	Initialize four memory banks individually with DAP ;
   806	for 𝑖 in [1,num_iterations] do
   807	Sample 𝑃 × 𝐾 query images from 𝑃 ;
   808	Compute objective function with Eq. (17) ;
   809	Update feature representations with Eqs. (8) and (10);
   810	end
   811	end
   812	
   813	8
   814	
   815	
codex
第一篇的页码比较长，前 8 页只读到 WMM 开头，方法还没结束。我会补读后续方法页，避免把损失函数和训练流程漏掉。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '1,420p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	325:8
     2	
     3	L. Dai et al.
     4	
     5	cross-modality cluster set 𝐷 𝑟 ∈ 𝑄 𝑣𝑟 for each infrared cluster:
     6	𝐷𝑖𝑟 = { 𝑗 |𝑄 𝑣𝑟
     7	𝑗 = 𝑖},
     8	
     9	(4)
    10	
    11	where j denotes the modality-specific label of infrared clusters. Then, we examine each 𝐷 𝑟 to
    12	merge visible clusters that satisfy the following two conditions before assigning pseudo labels from
    13	another modality: (1) they match the same infrared cluster; (2) the similarity between the clusters
    14	exceeds a specified threshold 𝜖. Subsequently, we consider that an infrared cluster tends to contain
    15	multiple identities if it is matched by k (𝑘 > 1) merged visible clusters. To achieve a high-quality
    16	division, we employ the k-means to partition the infrared cluster into k parts, thereby updating the
    17	𝑁𝑟 .
    18	infrared modality-specific labels Y𝑟 , obtaining Y𝑟𝑑 = {𝑦𝑖𝑟𝑑 }𝑖=1
    19	After the processes of merging and separating, we establish the refined matching relationship
    20	e𝑣𝑟 with the merged visible clusters and the separated infrared labels Y𝑟𝑑 .
    21	of visible to infrared 𝑄
    22	e𝑣𝑟 using
    23	Then, the separated infrared labels Y𝑟𝑑 are assigned to the merged visible clusters with 𝑄
    24	𝑣𝑟
    25	e
    26	Equation (3), obtaining Y . Consequently, the refined modality-shared labels (visible to infrared)
    27	e𝑣𝑟 , Y𝑟𝑑 }. As illustrated in Figure 3(b), the CRLR enhances the
    28	are established and represented as {Y
    29	quality of modality-shared labels through cross-modality correspondences. The refined modalitye𝑟 𝑣 }, which consists of separated visible labels of visible
    30	shared labels (infrared to visible) {Y 𝑣𝑑 , Y
    31	samples and the separated visible labels assigned to the merged infrared clusters, can be obtained
    32	in the same way.
    33	3.4
    34	
    35	WMM
    36	
    37	To build WMM, we initialize the modality-shared centroids by averaging the two modality-specific
    38	centroids, then assign weights to samples according to the similarity between the sample and
    39	modality-shared centroids to re-construct modality-specific centroids, and finally initialize WMM
    40	banks by fusing the modality-shared centroids and weighted version through a momentum strategy.
    41	Then, we give a thorough introduction of this module.
    42	Following the implementation of CSM and CRLR, modality-shared pseudo labels are assigned to
    43	samples from both matching directions. To facilitate the interaction of cross-modality information
    44	and achieve modality-invariant feature extraction, we construct two memory banks: the visible
    45	𝐾 𝑣 and the infrared MM 𝑀 𝑟 = {𝑐 𝑟 }𝐾 𝑟 .
    46	modality-shared memory (MM) 𝑀 𝑣 = {𝑐𝑖𝑣 }𝑖=1
    47	𝑖 𝑖=1
    48	For the visible MM, it is initialized using modality-shared cluster centroids {𝑐 1𝑣 , 𝑐 2𝑣 , . . . 𝑐 𝐾𝑣 𝑣 } at the
    49	beginning of each epoch. The cluster centroids are defined as:
    50	𝑐𝑖𝑣 =
    51	
    52	1 Õ
    53	𝑓,
    54	|H𝑖𝑣 |
    55	𝑣
    56	
    57	(5)
    58	
    59	𝑓 ∈ H𝑖
    60	
    61	𝑟𝑣
    62	where H𝑖𝑣 = H𝑖𝑣𝑑 ∪ H𝑖𝑟 𝑣 . The set H𝑖𝑣𝑑 comprises features {𝑓 𝑗𝑣 |𝑦 𝑣𝑑
    63	𝑗 = 𝑖} and the set H𝑖 consists of
    64	𝑟
    65	𝑟
    66	𝑣
    67	features {𝑓 𝑗 𝑦
    68	|e𝑗 = 𝑖}. Here | · | indicates the number of instances per cluster.
    69	However, discrepancies in the number of samples across cross-modality clusters can lead
    70	to a bias in the modality-shared centers, favoring the modality with more samples. Furthermore, this initialization strategy assigns equal weight to different samples, despite their varying
    71	representations.
    72	To simultaneously account for both the difference in features and the unequal number of crossmodality samples, we introduce the WMM. For the visible MM, we begin computing the cluster
    73	representation for each modality separately. The modality-shared cluster centroids are then initialized by averaging the two modality-specific cluster centers, thereby mitigating the imbalance in
    74	
    75	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
    76	
    77	DLLR for Unsupervised Visible-Infrared Person ReID
    78	
    79	325:9
    80	
    81	the number of samples across modalities:
    82	𝑐𝑖𝑣𝑠ℎ =
    83	
    84	Õ
    85	Õ
    86	1© 1
    87	1
    88	ª
    89	𝑓𝑣 +
    90	𝑓 𝑟® .
    91	­ 𝑣𝑑
    92	𝑟
    93	𝑣
    94	2 |H𝑖 | 𝑣 𝑣𝑑
    95	|H𝑖 | 𝑟 𝑟 𝑣
    96	𝑓 ∈ H𝑖
    97	𝑓 ∈ H𝑖
    98	«
    99	¬
   100	
   101	(6)
   102	
   103	To construct more suitable memory banks for contrastive learning, particularly with respect to
   104	the representation of samples from a global perspective, we introduce a weighted strategy to assign
   105	different weights to each sample for the construction of memory banks. Drawing inspiration from
   106	[21, 40], we allocate corresponding weights to similar instances relative to each query instance,
   107	ensuring that harder instances are given greater weight. For a feature 𝑓 𝑗 ∈ H𝑖𝑙 , where 𝑙 ∈ {𝑟𝑣, 𝑣𝑑 },
   108	𝑙 used to construct modality-shared cluster center 𝑐 𝑣 as:
   109	we define the weight 𝑤𝑖,𝑗
   110	𝑖
   111	𝑙
   112	𝑤𝑖,𝑗
   113	=Í
   114	
   115	exp (−𝑐𝑖𝑣𝑠ℎ · 𝑓 𝑗 /𝜏𝑤 )
   116	,
   117	
   118	| H𝑖𝑙 |
   119	exp(−𝑐𝑖𝑣𝑠ℎ · 𝑓𝑘 /𝜏𝑤 )
   120	𝑘=1
   121	
   122	(7)
   123	
   124	where 𝜏𝑤 is the temperature coefficient hyper-parameter. By combining these two strategies,
   125	we develop a weighted memory initialization strategy. The modality-shared cluster center 𝑐𝑖𝑣 is
   126	initialized through a momentum updating process that averages the two weighted centers as
   127	follows:
   128	1 −𝛾 ©
   129	𝑐𝑖𝑣 = 𝛾𝑐𝑖𝑣𝑠ℎ +
   130	­
   131	
   132	𝑣𝑑
   133	|H
   134	𝑖 |
   135	Õ
   136	
   137	|Õ
   138	H𝑖𝑟 𝑣 |
   139	
   140	𝑟𝑣 ª
   141	𝑤𝑖,𝑘
   142	𝑓𝑘 ® ,
   143	(8)
   144	𝑗=1
   145	𝑘=1
   146	«
   147	¬
   148	where 𝛾 serves as a tradeoff parameter that balances the average center and the weighted center
   149	for memory initialization. By thoroughly considering both the distribution and the sample size
   150	across modalities, the memory bank is more cohesive and effective for USVI-ReID compared to
   151	existing methods.
   152	
   153	2
   154	
   155	3.5
   156	
   157	𝑣𝑑
   158	𝑤𝑖,𝑗
   159	𝑓𝑗 +
   160	
   161	Dual-Modality-Shared Learning
   162	
   163	In dual-modality-shared learning, we update memory banks during training, employ modalityshared pseudo labels in two matching directions of samples to calculate the contrastive loss as the
   164	objective function for model training, and also add a modality classification loss to object functions.
   165	In the following section, we present a detailed overview.
   166	To conduct effective contrastive learning, we train the encoder within the dual-modality-shared
   167	learning framework. In this framework, we simultaneously utilize both matching directions of
   168	modality-shared labels for training. Although the modality-shared labels in the two matching
   169	directions may exhibit inconsistencies, this dual-modality-shared learning approach mitigates the
   170	noise present in a single matching direction and encourages correctly matched samples to learn
   171	cross-modality features. Furthermore, this pattern can solve the problem that some clusters are not
   172	matched by cross-modality clusters in the single matching direction and thereby cannot participate
   173	in cross-modality learning. This is further discussed in Appendix A.
   174	Based on the modality-shared labels of each matching direction, we sample P identities from
   175	each modality and K instances of each identity to form the mini-batch. Our algorithm alternates
   176	between pseudo labels generation and model learning, iterating in each epoch. To ensure that the
   177	memory banks adequately represent each identity during training, the visible and infrared MM
   178	bank 𝑐𝑖𝑡 at t iteration is updated using a momentum strategy:
   179	𝑐𝑖𝑡 ← 𝜇𝑐𝑖𝑡 −1 + (1 − 𝜇) 𝑓 ,
   180	
   181	(9)
   182	
   183	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   184	
   185	325:10
   186	
   187	L. Dai et al.
   188	
   189	where 𝜇 is the momentum updating factor. Here, f represents the feature of a sampled image that
   190	participated in the iterative model training, with its modality-shared label denoted as i.
   191	The encoder 𝐹𝜃 (·) is optimized using contrastive loss functions [8], which comprise two loss
   192	functions for the infrared MM and the visible MM, respectively. The loss functions for the visible
   193	MM 𝑀 𝑣 consists of L𝑣𝑣 and L𝑟 𝑣 . The former objective function pertains to visible features 𝑓𝑖𝑣 ∈
   194	𝑁𝑣
   195	
   196	{𝐹𝜃 (𝑥𝑖𝑣 ), 𝐹𝜃 (𝑥𝑖𝑣b)}𝑖=1 with their corresponding labels 𝑦𝑖𝑣𝑑 , and is formulated as:
   197	!
   198	𝑁
   199	exp (𝑓𝑖𝑣 · 𝑐 𝑣 𝑣𝑑 /𝜏)
   200	𝐵
   201	Õ
   202	𝑦𝑖
   203	L𝑣𝑣 = −
   204	log Í𝐾 𝑣
   205	,
   206	𝑣
   207	𝑣
   208	𝑗=1 exp (𝑓𝑖 · 𝑐 𝑗 /𝜏)
   209	𝑖=1
   210	
   211	(10)
   212	
   213	where 𝑁𝐵 is the number of samples in a mini-batch and 𝜏 is a temperature hyper-parameter [8].
   214	𝑁𝑟
   215	The latter objective function pertains to infrared features 𝑓𝑖𝑟 ∈ {𝐹𝜃 (𝑥𝑖𝑟 )}𝑖=1
   216	with their labels 𝑦e𝑖𝑟 𝑣 ,
   217	expressed as:
   218	!
   219	𝑁𝐵
   220	exp (𝑓𝑖𝑟 · 𝑐 𝑦𝑣e𝑟 𝑣 /𝜏)
   221	Õ
   222	𝑖
   223	L𝑟 𝑣 = −
   224	log Í𝐾 𝑣
   225	.
   226	(11)
   227	𝑟
   228	𝑣
   229	𝑗=1 exp (𝑓𝑖 · 𝑐 𝑗 /𝜏)
   230	𝑖=1
   231	The loss function for infrared MM is symmetrical to Equations (10) and (11). Furthermore,
   232	to generate modality-invariant features, we introduce a modality classifier 𝜙 for classifying the
   233	modality of features [15, 50]. The objective function L𝑑 is formulated in an adversarial form:
   234	L𝑑 = − max min E 𝑓𝑖𝑣 [log(1 − 𝜙 (𝑓𝑖𝑣 ))] + E 𝑓𝑖𝑟 [log(𝜙 (𝑓𝑖𝑟 ))].
   235	𝐹𝜃
   236	
   237	(12)
   238	
   239	𝜙
   240	
   241	The final training loss L utilized for training our framework is expressed as:
   242	L = L𝑣𝑣 + L𝑟 𝑣 + L𝑣𝑟 + L𝑟𝑟 + L𝑑 .
   243	4
   244	4.1
   245	
   246	(13)
   247	
   248	Experiments
   249	Experimental Settings
   250	
   251	We evaluate our framework on three public VI-ReID datasets: SYSU-MM01 [52], RegDB [36], and
   252	LLCM [71].
   253	SYSU-MM01 consists of images captured from two infrared cameras and four visible cameras.
   254	The dataset includes a total of 287,628 visible images and 15,972 infrared images, covering 491 IDs.
   255	Among these, 22,258 visible images and 11,909 near-infrared images from 395 IDs are utilized for
   256	training, while the remaining images from the rest 96 IDs are used for testing. We report the results
   257	under two testing modes: All-Search and Indoor-Search. The All-Search mode covers both indoor
   258	and outdoor environments, while the Indoor-Search mode is limited to indoor shooting scenarios.
   259	These two settings enable a comprehensive evaluation across various conditions.
   260	RegDB comprises images captured by a pair of aligned cameras, including 4,120 visible images
   261	and 4,120 infrared images from 412 IDs, where each ID has 10 corresponding images. Among the
   262	images, 156 IDs are captured from a frontal view, and the rest 256 are captured from a rear view.
   263	We ran the experiments ten times with random selection and reported the average accuracy for the
   264	final evaluation [59, 63]. This dataset offers two testing modes: Visible to Infrared and Infrared to
   265	Visible, utilizing visible images as the gallery and query, respectively.
   266	LLCM is a new and challenging dataset for cross-modality ReID. It was collected under a ninecamera network deployed in low-light environments. The dataset contains 46,767 images from
   267	1,064 IDs, presenting challenges such as motion blurring, occlusion, and low resolution. Similar to
   268	RegDB, it also provides two testing modes: Visible to Infrared and Infrared to Visible.
   269	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   270	
   271	DLLR for Unsupervised Visible-Infrared Person ReID
   272	
   273	325:11
   274	
   275	Evaluation Metrics. Following previous studies [52, 65], we adopt mean average precision
   276	(mAP) and cumulative matching characteristics (CMC) as our primary evaluation metrics.
   277	Additionally, we report the mean inverse negative penalty (mINP) [65], a metric specifically
   278	formulated to evaluate the performance of retrieving harder samples. The mINP is calculated as
   279	follows:
   280	1Õ
   281	1 Õ |𝐺𝑖 |
   282	𝑚𝐼 𝑁 𝑃 =
   283	(1 − 𝑁 𝑃𝑖 ) =
   284	,
   285	(14)
   286	𝑛 𝑖
   287	𝑛 𝑖 𝑅𝑖ℎ𝑎𝑟𝑑
   288	where 𝑅𝑖ℎ𝑎𝑟𝑑 denotes the rank position of the hardest match, and |𝐺𝑖 | indicates the total number of
   289	correct matches for query i. Unlike CMC and mAP, the mINP assesses the cost of finding all the
   290	correct matches.
   291	4.2
   292	
   293	Implementation Details
   294	
   295	Our proposed method is implemented using PyTorch. We adopt a two-stream ResNet-50 [20]
   296	pre-trained on ImageNet [9] as the encoder 𝐹𝜃 (·). Following the modality-shared layers, a GeM [38]
   297	pooling layer is employed, followed by a batch normalization layer [23]. The modality classifier
   298	implementation is based on [50], and we only process the final layer output. Additionally, our
   299	method involves alternating between pseudo label generation and model training. The features
   300	output by the encoder are l2-normalized during both pseudo label generation and testing. Before
   301	each training epoch, DBSCAN [11] is utilized to generate pseudo labels.
   302	During training, images are resized to 288 × 144 and data augmentations follow the method in
   303	Wu et al. [55]. The framework is trained for a total of 100 epochs: the first 50 epochs are allocated
   304	for initial training, and the remaining 50 epochs are for training our framework. For RegDB, we set
   305	the number of identities 𝑃 = 8 and instances for each identity 𝐾 = 4 during initial training, which is
   306	then reset to 𝑃 = 4. For SYSU-MM01 and LLCM, we set 𝑃 = 9 and 𝐾 = 8 during initial training and
   307	then reset them to 8 and 4, respectively. The maximum distance of DBSCAN for visible modality is
   308	set to 0.6 + (0.0008 × 𝑒𝑝𝑜𝑐ℎ) for initial training [21] and 0.68 for training our framework. We use
   309	the Adam with a weight decay of 5e-4 for training. The initial learning rate is set to 3.5e-4, which
   310	is reduced by a factor of 0.1 at every 20 epochs. Following [59], we set 𝜏 = 0.05 in Equations (10)
   311	and (11), and 𝜇 = 0.1 in Equation (9). Following [21], the 𝜏𝑤 in Equation (7) is 0.09. The remaining
   312	settings for contrastive learning follow [8].
   313	4.3
   314	
   315	Comparison with State-of-the-Art Methods
   316	
   317	We compare our method with state-of-the-art supervised learning (SL), semi-SL (SSL), and USL
   318	VI-ReID methods. The results on SYSU-MM01 and RegDB are reported in Table 1, and the results
   319	on LLCM are reported in Table 2.
   320	Comparison with Unsupervised Methods. The results demonstrate the effectiveness of our proposed
   321	framework, which surpasses the state-of-the-art method MBCCM [4] by 3.42% mAP on SYSU-MM01
   322	(All-Search) and 6.45% mAP on RegDB (Visible to Infrared). Notably, our framework operates with a
   323	smaller batch size compared to advanced methods [4, 5, 55, 59]indicating that our approach achieves
   324	superior performance while utilizing fewer GPU memory resources (detailed in supplementary
   325	materials). It is important to highlight that PGM [55] focuses on matching visible and infrared
   326	clusters in pairs at a minimum cost, which makes their method superior to SYSU-MM01 (All-Search).
   327	In contrast, our method employs a bidirectional approach and places greater emphasis on balanced
   328	cross-modality learning through MM, thus achieving better performance under various settings.
   329	Furthermore, the result of our method on SYSU-MM01 (All-Search) exceeds PGM [55] 2.97% in
   330	mINP, illustrating our method’s enhanced capability in handling hard samples. Additionally, we
   331	compare our method with other approaches on LLCM under the same settings. The dataset’s
   332	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   333	
   334	325:12
   335	
   336	L. Dai et al.
   337	
   338	Table 1. Comparison of Our Method with the State-of-the-Art VI-ReID Methods on RegDB and
   339	SYSU-MM01; It contains SL, SSL, and USL VI-ReID Methods
   340	Settings
   341	
   342	RegDB
   343	
   344	SYSU-MM01
   345	
   346	Visible to Infrared
   347	Infrared to Visible
   348	All-Search
   349	Indoor-Search
   350	Type
   351	Method
   352	Venue Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP
   353	Zero-Pad [52]
   354	ICCV-17 17.75 18.90
   355	16.63 17.82
   356	14.80 15.95
   357	20.58 26.92
   358	AlignGAN [48]
   359	ICCV-19
   360	57.9
   361	53.6
   362	56.3
   363	53.4
   364	42.4
   365	40.7
   366	45.9
   367	54.3
   368	DDAG [64]
   369	ECCV-20 69.34 63.46 49.24 68.06 61.80 48.62 54.75 53.02 39.62 61.02 67.98 62.61
   370	AGW [65]
   371	TPAMI-21 70.05 66.37 50.19 70.49 65.90 51.24 47.50 47.65 35.30 54.17 62.97 59.23
   372	CA [63]
   373	ICCV-21 85.03 79.14 65.33 84.75 77.82 61.56 69.88 66.89 53.61 76.26 80.37 76.79
   374	SL
   375	FMCNet [69]
   376	CVPR-22 89.12 84.43
   377	88.38 83.86
   378	66.34 52.51
   379	68.15 74.09
   380	MPANet [53]
   381	CVPR-21
   382	82.8
   383	80.7
   384	83.7
   385	80.9
   386	70.58 68.24
   387	76.74 80.96
   388	MAUM [33]
   389	CVPR-22 87.87 85.09
   390	86.95 84.83
   391	71.68 68.79
   392	76.97 81.94
   393	PartMix [24]
   394	CVPR-23 84.93 82.52
   395	85.66 82.27
   396	77.78 74.62
   397	81.52 84.38
   398	DEEN [71]
   399	CVPR-23
   400	91.1
   401	85.1
   402	89.5
   403	83.4
   404	74.7
   405	71.8
   406	80.3
   407	83.3
   408	MAUM-50 [33]
   409	CVPR-22
   410	28.8
   411	36.1
   412	MAUM-100 [33]
   413	CVPR-22
   414	38.5
   415	39.2
   416	SSL
   417	OTLA [50]
   418	ECCV-22
   419	32.9
   420	29.7

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '250,458p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   250	3. Method
   251	3.1. Overview
   252	Given a person Re-ID dataset 𝑃 = {𝑥𝑖 }𝑁
   253	, where 𝑁 represents the
   254	𝑖=1
   255	total number of images and 𝑥𝑖 denotes the 𝑖th image, the goal is to train
   256	a robust network 𝑓𝜃 that maps each sample 𝑥𝑖 to a corresponding feature
   257	vector 𝑿 𝑖 = 𝑓𝜃 (𝑥𝑖 ) in the embedding space  . Additionally, each image
   258	3
   259	
   260	Neural Networks 203 (2026) 109083
   261	
   262	Z.-H. Li et al.
   263	
   264	Fig. 3. An overview framework of the proposed DAMCL. It alternates between two steps. (a) In the clustering step, the encoder ﬁrst extracts the global feature map
   265	sets. Horizontal splitting and GEM pooling are then applied to obtain feature map sets for the upper and lower views, followed by computing and fusing the DACJ
   266	distances across these three views for clustering. Using the generated pseudo-labels and multi-view features, it initializes global DIP and three types of DOCPs. (b)
   267	In the training step, we exploit three DOCPs and global-DIP to perform contrastive learning. During the training process, the teacher model’s knowledge is utilized
   268	to guide the student model.
   269	
   270	𝑥𝑖 is paired with a camera label 𝑐𝑖 .
   271	As shown in Fig. 3, in the clustering step, the encoder 𝑓𝜃 is used to
   272	extract the global feature map sets 𝑀𝑔𝑏 . These feature maps are then
   273	horizontally split into upper and lower parts, denoted as 𝑀𝑢𝑝 and 𝑀𝑙𝑤 ,
   274	respectively. Subsequently, GEM pooling is applied to obtain the corresponding feature vector sets {𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 }. We then calculate DACJ
   275	distances for three views, which are denoted as 𝐷𝑔𝑏 , 𝐷𝑢𝑝 and 𝐷𝑙𝑤 . Subsequently, a re-weighted distance matrix 𝐷𝑐𝑝 is computed, which is then
   276	used to generate pseudo-labels {𝑦𝑖 𝑠} via the DBSCAN clustering algorithm, according to the following function:
   277	(1)
   278	
   279	sample pairs. First, the camera label relationship matrix 𝑀 is constructed using the camera IDs in the dataset to indicate whether samples 𝑥𝑖 and 𝑥𝑗 share the same camera label or not. Next, based on 𝐷
   280	and 𝑀, we compute the k-nearest neighbors (KNNs) 𝐾, as well as the
   281	within-camera KNNs 𝐾𝐼 and the cross-camera KNNs 𝐾𝐶 for each example. Finally, we introduce the camera variations degree vector Δ to
   282	estimate the diﬀerences in feature distributions between cross-camera
   283	and within-camera sample pairs. The calculation process for Δ is as
   284	follows:
   285	1 ∑
   286	1 ∑
   287	Δ𝑖 =
   288	𝑑 −
   289	𝑑 ,
   290	(2)
   291	𝑘1 𝑗∈𝐾 (𝑖) 𝑖,𝑗 𝑘1 𝑗∈𝐾 (𝑖) 𝑖,𝑗
   292	
   293	where 𝜇 is the balancing factor. Using {𝑦𝑖 𝑠} and {𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 }, we initialize global DIP and three types of DOCP.
   294	During training, we feed mini-batch images into the encoder to extract the features of the three views. These features are then used with
   295	the three DOCPs and DIP for contrastive learning. Importantly, the
   296	teacher model continuously supervises the training of the student model
   297	in an oﬄine manner throughout the process.
   298	
   299	where 𝑘1 is the number of nearest neighbors, Δ𝑖 means camera variations degree of sample 𝑥𝑖 and 𝑑𝑖,𝑗 represents the feature distance between
   300	samples 𝑥𝑖 and 𝑥𝑗 . During model optimization, our Δ is continuously and
   301	dynamically updated to reﬂect the latest changes.
   302	Dynamic adaptive camera k-reciprocal nearest neighbors. For
   303	sample 𝑥𝑖 , we query Δ𝑖 . If Δ𝑖 is greater than 0, we calculate the dynamic
   304	adaptive camera k-reciprocal nearest neighbors(DAC-KRNNs) of 𝑥𝑖 :
   305	
   306	𝐷𝑐𝑝 = 𝜇𝐷𝑢𝑝 + 𝜇𝐷𝑙𝑤 + (1 − 𝜇)𝐷𝑔𝑏 ,
   307	
   308	𝐶
   309	
   310	𝐼
   311	
   312	𝑅(𝑥𝑖 , 𝑘𝑐1 , 𝑘𝑖1 ) = Unique(
   313	
   314	3.2. Dynamic adaptive camera jaccard
   315	
   316	{𝑥𝑗 |𝑥𝑗 ∈ 𝐾𝐶 (𝑖)[∶ 𝑘𝑐1 ] ∩ 𝑥𝑖 ∈ 𝐾𝐶 (𝑗)[∶ 𝑘𝑐1 ]}∪
   317	
   318	(3)
   319	
   320	{𝑥𝑗 |𝑥𝑗 ∈ 𝐾𝐼 (𝑖)[∶ 𝑘𝑖1 ] ∩ 𝑥𝑖 ∈ 𝐾𝐼 (𝑗)[∶ 𝑘𝑖1 ]}),
   321	
   322	In unsupervised Re-ID, the Jaccard distance is a commonly employed
   323	metric, particularly for clustering tasks. However, camera variations often introduce numerous high-weight within-camera negative samples
   324	into the relevant neighbors while impeding the inclusion of informative
   325	cross-camera positive samples. As a result, relying on the traditional
   326	camera-unaware Jaccard distance causes clustering algorithms to prioritize connections between similar within-camera sample pairs of diﬀerent identities, while failing to group cross-camera samples of the same
   327	identity. To address these challenges and reﬁne the Jaccard distance
   328	before clustering to reduce pseudo-label noise, we introduce the DACJ
   329	distance.
   330	The original distance matrix 𝐷 is computed based on either cosine
   331	similarity or Euclidean distance metrics.
   332	Dynamic adaptive camera variations estimation. To reduce the
   333	impact of camera variations, it is essential to precisely assess the diﬀerences in feature distributions between cross-camera and within-camera
   334	
   335	where Unique(⋅) denotes removing duplicate elements, 𝑘𝑐1 and 𝑘𝑖1 mean
   336	diﬀerent k are used in 𝐾𝑐 and 𝐾𝑖 , 𝑘𝑐1 + 𝑘𝑖1 = 𝑘1 . To reduce the number
   337	of within-camera negative samples, we increase the proportion of crosscamera positive samples in DAC-KRNNs by using a larger 𝑘𝑐1 .
   338	Vectorization of neighbors. After the DAC-KRNNs of sample 𝑥𝑖 are
   339	obtained, they need to be encoded into a weighted DAC-KRNNs vector
   340	𝑉𝑖 = [𝑉𝑖,1 , 𝑉𝑖,2 , …, 𝑉𝑖,𝑁 ]. The calculation formula is as follows:
   341	−𝑑𝑖,𝑗
   342	
   343	⎧ ∑𝑒 −𝑑
   344	𝑒 𝑖,𝑘
   345	⎪
   346	𝑉𝑖,𝑗 = ⎨ 𝑥𝑘 ∈𝑅
   347	⎪
   348	⎩0
   349	
   350	if 𝑥𝑗 ∈ 𝑅
   351	(4)
   352	otherwise,
   353	
   354	where 𝑅 is the DAC-KRNNs of sample 𝑥𝑖 .
   355	Dynamic adaptive camera local query expansion. Due to camera variations, a large number of within-camera negative samples are
   356	4
   357	
   358	Neural Networks 203 (2026) 109083
   359	
   360	Z.-H. Li et al.
   361	
   362	𝑐 ∈ ℝ𝐶×𝑑 contains 𝐶 entries, with 𝑑 representing the feature dimension. Selecting the medoid rather than the centroid (mean) as the cluster
   363	proxy is a deliberate design choice to better cope with the complexity
   364	and noise inherent in the feature space of unsupervised person Re-ID.
   365	First, as noted in Section 1, due to signiﬁcant variations in pose,
   366	viewpoint, illumination, and cross-camera discrepancies, Re-ID feature
   367	clusters often exhibit complex, non-spherical. In such irregularly shaped
   368	distributions, the centroid which is computed as the arithmetic mean,
   369	tends to fall into low-density “empty” regions, deviating substantially
   370	from the true data manifold. By contrast, the medoid is an actual data
   371	point that lies on the manifold itself, enabling it to more faithfully represent the complex geometric structure captured by density-based clustering methods such as DBSCAN.
   372	Second, pseudo-labels in unsupervised Re-ID are inevitably noisy,
   373	containing misclassiﬁcations and outliers primarily caused by crosscamera appearance gaps. The centroid (mean) is highly sensitive to these
   374	outliers because averaging incorporates every sample equally, allowing
   375	extreme points to disproportionately shift the proxy toward erroneous
   376	regions. The medoid, deﬁned as the point that minimizes the total distance to all other points in the cluster, exhibits strong robustness to outliers and thus provides a more stable and reliable proxy for contrastive
   377	learning.
   378	Third, since DBSCAN deﬁnes clusters based on local density and local
   379	Euclidean distances, using a synthetic centroid (especially when updated
   380	via momentum) risks causing progressive drift of the proxy features
   381	away from the original data manifold, ultimately distorting the very
   382	local structure that DBSCAN has discovered. By adopting the medoid,
   383	which is an actual sample feature, we preserve the intrinsic geometric
   384	properties of the feature space, ensuring strong consistency between the
   385	optimization objective of contrastive learning and the underlying manifold structure revealed by clustering.
   386	Additionally, selecting the medoid as the proxy is well-justiﬁed.
   387	As shown in Fig. 5, compared to the traditional mean centroid proxy
   388	strategy, our proposed medoid-based optimal proxy strategy exhibits a
   389	steeper and faster loss decline during the training phase, which directly
   390	demonstrates that the DOCP module can signiﬁcantly accelerate convergence. Moreover, with minimal diﬀerence in the ﬁnal loss values, our
   391	proxy strategy displays reduced ﬂuctuations and oscillations throughout the training process, thereby proving its superior robustness against
   392	pseudo-label noise and outliers.
   393	Updating the memory bank during back-propagation is crucial. To
   394	preserve the intrinsic manifold structure of the data and maintain the in-
   395	
   396	introduced into KRNNs, making LQE generate the unreliable weighted
   397	expanded neighbors vector 𝑉𝑖𝑒 of sample 𝑥𝑖 , which reassign high weights
   398	to these negative samples. So we propose a dynamic adaptive camera
   399	local query expansion(DAC-LQE) to boost the reliability of vectors:
   400	⎧⎛
   401	⎞
   402	∑
   403	∑
   404	⎪⎜
   405	𝑉𝑗 +
   406	𝑉𝑘 ⎟
   407	⎪ ⎜𝑥𝑗 ∈𝐾𝐶 (𝑖)[∶𝑘𝑐 ]
   408	⎟
   409	𝑥𝑘 ∈𝐾𝐼 (𝑖)[∶𝑘𝑖2 ]
   410	2
   411	⎠
   412	⎪⎝
   413	𝑉𝑖𝑒 = ⎨
   414	𝑐 + 𝑘𝑖
   415	𝑘
   416	2
   417	2
   418	⎪
   419	∑
   420	⎪1
   421	𝑉𝑗
   422	⎪ 𝑘2
   423	⎩ 𝑥𝑗 ∈𝐾(𝑖)[∶𝑘2 ]
   424	
   425	if Δ𝑖 > 0
   426	(5)
   427	otherwise,
   428	
   429	where 𝑘𝑐2 + 𝑘𝑖2 = 𝑘2 , 𝑉𝑗 and 𝑉𝑘 are the weighted DAC-KRNNs vectors of
   430	𝑥𝑗 and 𝑥𝑘 .
   431	Overlap computation. The DACJ 𝐷𝑖,𝑗 between 𝑥𝑖 and 𝑥𝑗 is computed as follows:
   432	𝑁
   433	∑
   434	
   435	𝐷𝑖,𝑗 = 1 −
   436	
   437	𝑘=1
   438	𝑁
   439	∑
   440	𝑘=1
   441	
   442	(
   443	)
   444	𝑒 ,𝑉 𝑒
   445	min 𝑉𝑖,𝑘
   446	𝑗,𝑘
   447	(
   448	)
   449	𝑒 ,𝑉 𝑒
   450	max 𝑉𝑖,𝑘
   451	𝑗,𝑘
   452	
   453	.
   454	
   455	(6)
   456	
   457	By dynamically estimating the camera variations for each instance, our
   458	DACJ eﬀectively incorporates more cross-camera positive samples, offering greater reliability and robustness compared to the traditional Jaccard distance.

codex
前两篇已经读到足够的方法细节。第一篇把创新压在“跨模态伪标签关系可被反过来修标签”上，第二篇把常规均值代理重定义成和 DBSCAN 局部流形假设冲突，这是很典型的把工程弱点升格成论文问题。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf' - | nl -ba | sed -n '43,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    43	I. I NTRODUCTION
    44	Person Re-identification (ReID) is crucial for surveillance
    45	and tracking, identifying individuals across camera views. Advances in deep learning have improved feature extraction and
    46	matching accuracy [1]–[5]. However, most methods rely on
    47	global image features, making them vulnerable to background
    48	noise and irrelevant regions, particularly in cases of occlusion
    49	or complex backgrounds. This limits their effectiveness in
    50	diverse real-world scenarios with cross-camera variations and
    51	environmental inconsistencies [6]–[8].
    52	To address these challenges, recent studies have emphasized the importance of more targeted and efficient feature
    53	extraction approaches. For instance, Zhang et al. [9] propose
    54	a separable attention mechanism to focus on discriminative regions while suppressing irrelevant background features.
    55	Tang et al. [10] introduce adaptive context-aware selection to
    56	dynamically enhance feature representations under complex
    57	conditions. Similarly, Qiu et al. [11] develop a salient feature
    58	*Corresponding Author, yuhaiwan@usc.edu
    59	
    60	Fig. 1: A straightforward description of Aerial-Ground Person Re-identification (AGPReID) involves the utilization of
    61	an aerial-ground mixed camera network, enabling matching
    62	across aerial-aerial, ground-ground, and aerial-ground scenarios. Thus, it presents greater challenges and practical applications compared to traditional single-camera person ReID
    63	methods.
    64	
    65	extraction framework that prioritizes key object parts even in
    66	scenarios involving significant occlusion. These advancements
    67	show promising progress in overcoming the limitations of
    68	the reliance on global feature in View-homogeneous person
    69	ReID. However, when applied to Aerial-Ground Person Reidentification (AGPReID) tasks (View-heterogeneous person
    70	ReID), which are valuable in real-world scenarios for addressing complex aerial-to-ground matching challenges and
    71	encompassing diverse camera perspectives [12], these methods
    72	often fall short. Fig. 1 demonstrates the AGPReID problem.
    73	This discrepancy may stem from the scale diversity and redundancy characteristics observed in large-area observational
    74	scenarios, leading to notable appearance differences for the
    75	same individual across various cameras. Therefore, there is an
    76	urgent need to develop innovative strategies that effectively
    77	address these specific challenges in AGPReID.
    78	To this end, we propose a Dynamic Token Selective Transformer (DTST) that enhances identity representation by focusing on the most critical spatial features. Our DTST module
    79	contains two steps: First, a Predictor Local-Global network
    80	
    81	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:56 UTC from IEEE Xplore. Restrictions apply.
    82	
    83	Fig. 2: Illustration of the proposed Dynamic Token Selective Transformer (DTST) framework. The framework incorporates N
    84	Token Selection view-decoupled transformer (VDT) blocks, where each block consists of an encoder layer and a visual token
    85	selector. The loss function is designed to account for both view-related and view-unrelated features, while an orthogonal loss
    86	ensures that these features remain independent from each other, further enhancing feature disentanglement and robustness.
    87	
    88	Fig. 3: The Illustration of Visual Token Selector (VTS). The process involves selecting the Top-K informative tokens from the
    89	original token set to be used in the subsequent feature aggregation.
    90	
    91	computes relevance scores for each token, integrating local and
    92	global spatial semantics using multi-head attention. Second, a
    93	Perturbation-Based Top-K Selector chooses the most relevant
    94	tokens based on the predicted scores, ensuring robustness by
    95	adding noise perturbations. The selected tokens are combined
    96	with a global class token, enabling efficient and compact representation while reducing computational overhead. Extensive
    97	experiments validate our method’s state-of-the-art performance
    98	on AGPReID tasks, showcasing its robustness in handling
    99	occlusions, complex backgrounds, and viewpoint variations.
   100	Our main contributions are as follows.
   101	We propose a Top-k Token Selective Transformer for
   102	AGPReID, to better model identity representation spatially. We further comprehensively study the impact of
   103	the insertion position and the number of tokens selected
   104	on the model’s performance.
   105	• To eliminate the interference of irrelevant tokens, our
   106	method adaptively selects the most critical tokens based
   107	on the top-k selective mechanism, making the long-range
   108	modeling more effective and compact.
   109	• Extensive experiments on various datasets demonstrate
   110	that our proposed model achieves state-of-the-art performance on AGPReID tasks.
   111	
   112	•
   113	
   114	II. R ELATED W ORK
   115	A. Person Re-identification
   116	Person re-identification (ReID) is essential for retrieving
   117	images of the same individual across different camera views.
   118	It can be categorized into view-homogeneous and viewheterogeneous ReID. View-homogeneous ReID pertains to
   119	scenarios with a single camera type, such as ground-only or
   120	aerial-only networks, while view-heterogeneous ReID such
   121	as Aerial-Ground Person ReID (AGPReID), deals with networks featuring diverse camera perspectives. In terms of
   122	view-homogeneous ReID, ground-only camera networks have
   123	received more attention compared to aerial-only networks. For
   124	example, some ground-only datasets are well established such
   125	as Market1501 [13] and MSMT17 [14]. As a consequence,
   126	a multitude of methods have been proposed, such as handcrafted feature-based, CNN-based, and transformer-based approaches, facilitating the development of ReID. However, these
   127	methods overlook the significant view differences between
   128	aerial and ground cameras, leading to poor performance
   129	faced with diverse view-point scenarios. Fortunately, viewheterogeneous ReID can address this issue. Recently, researchers in [12] propose the AG-ReID dataset, which includes
   130	
   131	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:56 UTC from IEEE Xplore. Restrictions apply.
   132	
   133	identity and attribute labels, and put forward an attributeguided model. Another work extends this by introducing
   134	the CARGO dataset with multiple matching scenarios and
   135	proposes a view-decoupled transformer (VDT) that decouples
   136	view-related features using hierarchical separation and orthogonal loss, improving performance and reducing reliance on
   137	extensive attribute labeling [15]. However, this approach does
   138	not dynamically select key tokens related to the target object,
   139	fails to reduce redundant computation, and lacks enhanced
   140	model capability to focus specifically on critical regions of
   141	interest.
   142	B. Token Selection in Vision Transformers
   143	Token selection is crucial for addressing redundancy issues
   144	in transformer-based vision models, particularly in tasks involving dense visual data. Despite their success, transformers
   145	often suffer from computational inefficiencies due to the need
   146	to process numerous redundant tokens. Token selection methods can effectively mitigate this issue by focusing on only the
   147	most informative tokens for further processing. For example,
   148	STTS [16], as a representative work, utilizes token selection
   149	to enhance computational efficiency by dynamically reducing
   150	the number of tokens processed at each transformer layer.
   151	These approaches have demonstrated substantial reductions in
   152	computation while maintaining performance. To address the
   153	challenge of differentiability in token selection, a perturbed
   154	maximum strategy is introduced [17], enabling top-K selection
   155	to be differentiable, thereby facilitating end-to-end training.
   156	Building on the principles of differentiable top-K selection
   157	[18], we develop a lightweight token selection module specifically designed to enhance temporal-spatial modeling in our
   158	view-decoupled transformer. By selecting only the most informative tokens, this module reduces redundancy and improves
   159	both efficiency and performance, especially in modeling visual
   160	data across multiple viewpoints.
   161	III. M ETHOD
   162	A. Formulation
   163	Aerial-Ground Person ReID aims to match images from
   164	ground- or aerial-only camera networks. In a training dataset
   165	|D tr |
   166	Dtr = {(xi , yi , vi )}i=1 , each instance consists of an image
   167	xi depicting a person, along with identity label yi and view
   168	label vi . The view label vi ∈ {v a , v g } is determined by
   169	the known camera labels in D, distinguishing between aerial
   170	(v a ) and ground (v g ) views. A substantial distinction in
   171	views between v a and v g results in a biased feature space,
   172	characterized by low intra-identity similarity and high interidentity dissimilarity.
   173	B. Overview
   174	As illustrated in Fig.2, we propose a token enhanced framework based on the View-Decoupled Transformer (VDT) to
   175	tackle the view discrepancy challenge in AGPReID. Input
   176	images that include both aerial (va ) and ground (vg ) views
   177	are tokenized into a sequence of tokens. To encompass both
   178	global and view-specific details, meta tokens and view tokens
   179	
   180	are added to these image tokens before they are inputted into
   181	our VDT.
   182	Comprising N blocks, the VDT framework initiates each
   183	block with a conventional self-attention encoding process,
   184	succeeded by a subtraction operation between meta and view
   185	tokens to explicitly disentangle view-specific characteristics
   186	from the overarching ones. This facilitates a distinct segregation of features influenced by diverse viewpoints.
   187	Subsequently, the updated meta and view tokens produced
   188	by the VDT are supervised by identity and view classifiers. To
   189	enforce the independence of meta and view tokens, we introduce an orthogonal loss, facilitating the successful separation
   190	of view-based and view-agnostic attributes. To select the most
   191	critical tokens, a visual token selector module is proposed to
   192	enhance the identity representation, with further elaboration
   193	provided in subsequent sections.
   194	We introduce the Visual Token Selector (VTS), as shown in
   195	Fig. 3, designed to dynamically refine the token representation
   196	by selecting the most informative tokens for subsequent analysis. This module aims to reduce redundancy and enhance the
   197	model’s ability to focus on critical regions, thereby optimizing
   198	computational efficiency while preserving feature quality. The
   199	VTS mechanism can be understood as a dynamic token
   200	selection process that leverages attention scores to determine
   201	the importance of each token.
   202	For a sequence of tokens {ti }M
   203	i=1 , where M is the number
   204	of tokens, the VTS computes importance scores for each token
   205	si using a lightweight attention mechanism. The score si is
   206	obtained as:
   207	
   208	 ⊤
   209	ti Wq Wk⊤ ti
   210	√
   211	,
   212	si = softmax
   213	d
   214	where ti is the i-th token, Wq and Wk are learnable matrices
   215	representing query and key transformations, and d is the
   216	dimensionality of the tokens. The softmax function normalizes
   217	the scores to ensure they sum to 1, thus creating a probabilistic
   218	distribution over the tokens.
   219	These tokens are then ranked based on their importance
   220	scores, and we select the top-K tokens with the highest
   221	scores, where K < M is a hyperparameter that controls the
   222	number of tokens retained. Mathematically, this selection can
   223	be represented as:
   224	{tselected
   225	} = TopK({si }M
   226	i
   227	i=1 ),
   228	where TopK(·) returns the indices corresponding to the topK scores. The retained tokens, {tselected
   229	}, are then passed to the
   230	i
   231	subsequent layers or directly to the final classification head.
   232	To ensure that the VTS can be used in an end-to-end
   233	training fashion, we adopt a differentiable approach for the
   234	token selection. Specifically, we use a continuous relaxation
   235	of the TopK function by employing a Gumbel-Softmax trick:
   236	exp((si + gi )/τ )
   237	ŝi = PM
   238	,
   239	j=1 exp((sj + gj )/τ )
   240	
   241	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:56 UTC from IEEE Xplore. Restrictions apply.
   242	
   243	where gi are Gumbel noise samples and τ is the temperature
   244	parameter that controls the smoothness of the approximation.
   245	This differentiable approximation allows the selection of tokens to be included in backpropagation, facilitating end-to-end
   246	optimization.
   247	By incorporating the Visual Token Selector, we achieve
   248	several key benefits:
   249	• Reduce redundancy: By selecting only the most informative tokens, we minimize the amount of redundant
   250	information processed by the model.
   251	• Enhance discriminability: The model can focus on the
   252	most critical aspects of the input, leading to improved
   253	performance on tasks requiring fine-grained feature analysis.
   254	• Improve computational efficiency: Reducing the number of tokens processed helps in reducing the overall
   255	computational cost, making the model more efficient for
   256	both training and inference.
   257	
   258	that selects the top two rated tokens, integrated after the final
   259	transformer encoder block for enhanced performance.
   260	C. Comparisons with State-of-the-art Methods

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '36,812p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    36	I. I NTRODUCTION
    37	
    38	P
    39	
    40	ERSON re-identification is employed to identify and
    41	locate specific individuals among pedestrians captured in
    42	multiple camera surveillance scenarios [1], [16]. This technology have been significantly developed during the last decade
    43	due to its important role in the fields of multimedia data
    44	retrieval and criminal investigation [3], [4], [5]. Earlier works
    45	on person Re-ID focus on the retrieval of pedestrian images
    46	captured by RGB cameras. However, such methods falter in
    47	low-light conditions. This is primarily due to the inherent
    48	limitation of RGB cameras in acquiring high-definition images
    49	Yiming Yang and Haifeng Hu are with the School of Electronics and
    50	Information Technology, Sun Yat-sen University, Guangzhou 510006, China.
    51	(e-mail: yangym53@mail2.sysu.edu.cn; huhaif@mail.sysu.edu.cn).
    52	Weipeng Hu is with the School of Electrical and Electronic Engineering (EEE), Nanyang Technological University, Singapore 639798. (e-mail:
    53	weipeng.hu@ntu.edu.sg).
    54	
    55	Fig. 1. Illustration of cross-modality and cross-camera discrepancies in
    56	clustering. Large variations caused by these discrepancies lead to identities
    57	splitting. Fine-tuning the network using these inaccurate labels obtains suboptimal results.
    58	
    59	in darkness. Therefore, visible-infrared person re-identification
    60	(VI-ReID) is proposed and employed to form 24-hour surveillance system, which aims to match infrared images under poor
    61	illumination with visible images under good illumination.
    62	The current VI-ReID methods focus on generation and subspace mapping technique to learn modality-invariant representations, achieving notable success [6], [7], [8], [9]. However,
    63	their dependence on manually annotated associations between
    64	visible and infrared modalities can hinder the scalability and
    65	deployment of the VI-ReID model. Unsupervised Learning
    66	Visible-Infrared Person Re-Identification (USL-VI-ReID) is
    67	introduced to eliminate this reliance on annotations and gains
    68	increasing attention due to its promising potential.
    69	The USL-VI-ReID method explores cross-modal associations, eliminating the requirement for manual identity labeling [10], [11]. Although DBSCAN [12] is an effective
    70	clustering algorithm for pseudo-labeling unlabeled data, challenges still arise due to substantial cross-modality and crosscamera variations within the images from the same ground
    71	truth ID [6], [13], [14]. Fig. 1 illustrates the challenge in
    72	clustering cross-modal person data. Variations across cameras
    73	and modalities lead to excessive identity splitting and hinder
    74	accurate label assignment. Fine-tuning the network using these
    75	
    76	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
    77	
    78	labels may amplify the distances within the same class more
    79	than those between different classes. In this case, additional
    80	noise can be potentially introduced to adversely affect the
    81	performance of the model. Existing methods [15], [16], [17],
    82	[18], [19] primarily focus on eliminating differences between
    83	different modalities while neglecting the challenge of crosscamera discrepancy. For instance, previous approaches utilize
    84	graph matching [16] and optimal transmission [20] techniques
    85	to facilitate cross-modal cluster association. However, these
    86	methods are impeded by the severe issue of excessive identity
    87	splitting, which may affect the accuracy of the association.
    88	To address the above problems, we propose a novel Dynamic Modality-Camera Invariant Clustering (DMIC) framework to eliminate cross-modality and cross-camera discrepancies at the clustering level. The flowchart of DMIC is displayed
    89	in Fig. 2. To be specific, DMIC naturally combines ModalityCamera Invariant Expansion (MIE), Dynamic Neighborhood
    90	Clustering (DNC) and Hybrid Modality Contrastive Learning
    91	(HMCL) into a joint framework. The MIE integrates intermodal and inter-camera distance coding, generating robust
    92	embeddings for the clustering algorithm. This eradicates the
    93	implicit inclusion of modality and camera information in the
    94	distance coding, resulting in modality-camera invariant embeddings. Consequently, modality-camera invariant associations
    95	can be established. To tackle the issue of intra-class distances surpassing inter-class distances due to excessive identity
    96	splitting, the DNC employs two dynamic search strategies.
    97	Specifically, for the first stragegy, we dynamically narrow the
    98	search radius to include the reliable positive samples within
    99	clusters, thereby enhancing the model’s ability to distinguish
   100	between relevant and irrelevant samples. Subsequently, we
   101	dynamically broaden the search radius, using the model’s
   102	refined discrimination to effectively incorporate reliable crosscamera and cross-modality samples into the cluster. The
   103	second strategy involves recalibrating the expanded distance
   104	coding. This refinement facilitates a broaden affinities of more
   105	cross-camera instances, which can be leveraged to contribute
   106	to cross-camera invariant learning. Importantly, our strategies
   107	do not require additional parameters as a cost, effectively
   108	enhancing the model’s performance. Taking inspiration from
   109	[21], we design HMCL to optimize instance-level and clusterlevel distributions. We randomly select instance from different
   110	modalities as cluster’s centroid and update the representations
   111	of clusters in a real-time manner, which effectively reduces
   112	modal gap.
   113	To sum up, the main contributions of this paper are list as
   114	follows:
   115	• We propose a novel DMIC network for USL-VI-ReID
   116	that simultaneously eliminates both the cross-modality
   117	and cross-camera discrepancies in clustering.
   118	• The MIE fuses distance coding between inter-modal and
   119	inter-camera instances, which bridges cross-modality and
   120	cross-camera gaps.
   121	• The DNC consists of two dynamic search strategies
   122	that do not require additional parameters, facilitating
   123	early optimization of model discriminability and gradually extending generalization to different cameras and
   124	modalities.
   125	
   126	2
   127	
   128	The HMCL performs cluster-level and instance-level contrastive learning for intra-modality and inter-modality
   129	training. The cluster representations are updated using
   130	randomly selected cross-modal samples to obtain compact cross-modal distribution.
   131	• Extensive experiments on the SYSU-MM01 and RegDB
   132	datasets showcase the effectiveness and competitive performance of our DMIC framework.
   133	•
   134	
   135	II. R ELATED WORK
   136	In this section, we provide a brief overview of the following areas: Supervised Visible-Infrared person ReID (SVIReID), Unsupervised Single-Modality Person ReID (USLReID), and Unsupervised Learning Visible-Infrared Person
   137	Re-Identification (USL-VI-ReID).
   138	A. Supervised Visible-Infrared person ReID
   139	Supervised Visible-Infrared Person ReID (SVI-ReID) can
   140	be roughly divided into feature-level and image-level modality
   141	alignment methods.
   142	The feature-level modality alignment methods aim to project
   143	cross-modal features into a shared subspace and employs metric learning techniques to narrow the distribution. In pursuit of
   144	this goal, Ye et al. [6] introduce a modality-aware collaborative
   145	ensemble learning approach to eliminate modal discrepancies
   146	at both the instance and classifier levels. Lu et al. [7] propose a
   147	two-step Wasserstein loss to align modality-unrelated information, which includes viewpoint, background, and posture. To
   148	fully leverage a wide array of cross-modality cues, a diverse
   149	embedding expansion network [22] is introduced to expand
   150	the feature set and reduce modal differences through triplelevel constraints. Hao et al. [23] design a camera-aware and
   151	modality-aware framework to enhance the discriminability and
   152	generalization of cross-modal representations.
   153	Image-level modality alignment methods use generative
   154	networks or image enhancement techniques to eliminate modal
   155	differences between pixels. Ye et al. [24] introduce a Channel
   156	exchangeable Augmentation (CA) method to enhance the generalization of visible stream. Li et al. [25] design a lightweight
   157	network to transform visible images into X -modality in selfsupervised manner. Zhang et al. [26] propose a non-linear
   158	simple generator to synthesize cross-modal images to middle
   159	modality. Furthermore, several works [8], [9], [27], [28] successfully employ Generative Adversarial Networks (GANs) to
   160	achieve the transitions between visible and infrared modalities
   161	while preserving identity information. However, it is noted that
   162	modality alignment methods may inevitably introduce noise
   163	and may not be suitable for scenarios demanding high realtime performance.
   164	B. Unsupervised Single-Modality person ReID
   165	Unsupervised Single-Modality Person ReID (USL-ReID)
   166	tries to tackle the demanding and time-intensive task of assigning labels to visible modality images. Recently, mainstream
   167	methods fine-tune the network by assigning labels to the
   168	data through clustering algorithms. Memory-based learning
   169	
   170	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
   171	
   172	3
   173	
   174	Fig. 2. The flowchart of Dynamic Modality-Camera Invariant Clustering (DMIC) model. Our DMIC model is composed of three key modules: ModalityCamera Invariant Expansion (MIE), Dynamic Neighborhood Clustering (DNC), and Hybrid Modality Contrastive Learning (HMCL). MIE fuses the distance
   175	encoding from multiple cameras to generate modality-camera invariant embeddings for clustering. DNC employs two dynamic search strategies that optimize
   176	the network’s performance, transitioning from improving discriminability to enhancing generalization. The estimated pseudo labels from MIE and DNC are
   177	used to initialize instance-level and cluster-level memories. HMCL includes intra-modality and inter-modality contrastive learning to learn modality-camera
   178	invariant representations. During the testing phase, our framework only utilizes the backbone for testing purposes.
   179	
   180	methods are then used to optimize the relationships between
   181	instances and clusters. SPCL [29] design unified contrastive
   182	learning to distinguish inter-cluster distributions. ClusterContrast [30] refines InfoNCE [31] and presents ClusterNCE,
   183	effectively enhancing the optimization of unsupervised clusters
   184	and improving the performance of unsupervised methods.
   185	Lan et al. [32] introduce multi-view features to contrastive
   186	framework, which enables efficient mining of partial cues and
   187	the refinement of pseudo labels. To solve camera discrepancy
   188	problem, Xuan et al. [33] divide unsupervised learning into
   189	inter-camera training and intra-camera training to generate
   190	reliable pseudo labels for cross-camera data. Zhang et al. [34]
   191	propose time-based camera contrastive learning to select the
   192	hardest camera centroid as a proxy for each cluster. ICE [35]
   193	design cross-camera proxy contrastive loss to mitigate camera
   194	discrepancy.
   195	C. Unsupervised Visible-Infrared person ReID
   196	The existing Unsupervised Learning Visible-Infrared Person
   197	Re-Identification (USL-VI-ReID) methods mainly focus on
   198	establishing cross-modal associations. ADCA [15] adopt count
   199	priority selection method to facilitate cross-modal fusion.
   200	Wu et al. [16] design two cross-modal graphs to discover
   201	correspondences between different modalities. Pang et al.
   202	[17] extract three channels from visible images, conduct
   203	clustering with infrared images, and subsequently utilize IoU
   204	for label refinement. Liang et al. [18] pretrain model with
   205	
   206	labeled single-modality dataset and introduce a homogeneousto-heterogeneous training method. Wang et al. [20] utilize
   207	optimal transport techniques to transfer label knowledge from
   208	the visible modality to the infrared modality. However, the
   209	above-mentioned methods do not take into account the impact
   210	of camera differences in clustering phase. Especially, the
   211	susceptibility of visible images to camera discrepancy will
   212	lead to excessive identity splitting. In this case, using one-toone or one-to-many association methods can exacerbate the
   213	impact of noisy labels. Unlike the above-mentioned methods,
   214	we fully utilize camera information throughout the clustering
   215	process to solve the identity splitting problem. Like ICE [35]
   216	and CAP [55], GUR [37] and DCCL [36] cluster the samples
   217	within camera and employ cross-camera proxy contrastive
   218	loss to solve camera discrepancy. Unlike these methods, we
   219	do not need to perform clustering within individual cameras.
   220	Instead, we integrate camera information in a global clustering
   221	approach and introduce two dynamic search strategies to
   222	address camera differences.
   223	
   224	III. T HE P ROPOSED M ODEL
   225	In this section, we present a Dynamic Modality-Camera
   226	Invariant Clustering (DMIC) framework to simultaneously
   227	reduce cross-modality and cross-camera discrepancies. Our
   228	framework is illustrated in Fig. 2.
   229	
   230	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
   231	
   232	4
   233	
   234	A. Problem Modeling
   235	In USL-VI-ReID, we adopt dual-stream backbone AGW
   236	[3] as backbone f . Given visible-infrared pedestrian datasets,
   237	we discard all identity labels, which means we are unable
   238	to use manual annotations as supervison. The visible-infrared
   239	pedestrian datasets can be represented as D = {V, R}, where
   240	r Nr
   241	v
   242	V = {xvi }N
   243	i=1 and R = {xi }i=1 indicate visible and infrared
   244	images, respectively. Nv denotes the number of visible images,
   245	and Nr indicates the number of infrared images. We employ
   246	Channel exchangeable Augmentation (CA) [24] to enhance
   247	the generalization ability of visible stream. CA data can be
   248	v
   249	represented as C = {xci }N
   250	i=1 . Notably, we only enhance visible
   251	images using CA technique during the training process.
   252	It should be noted that within our framework, we include
   253	two training phases, i.e., intra-modality training and intermodality training. The intra-modality training aims to enhance the initial discriminability of model, while the intermodality training aims to develop the cross-modality and
   254	cross-camera generalization of model. For intra-modality training, we employ intra-modality clustering to assign modalityspecific labels {ŷ r , ŷ v } to the data in two different modalities,
   255	separately. In the context of inter-modality training, we utilize
   256	both intra-modality clustering and inter-modality clustering.
   257	Inter-modality clustering involves taking both infrared and
   258	visible data as inputs to the clustering algorithm, which
   259	assigns modality-shared labels {ŷ m }. Therefore, during intermodality training, each sample generates two pseudo-labels,
   260	i.e., {ŷ r , ŷ m } for infrared samples and {ŷ v , ŷ m } for visible
   261	samples.
   262	B. Modality-Camera Invariant Expansion
   263	In prior research [15], [16], [18], a commonly used clustering approach involved calculating Jaccard distances and
   264	utilizing distance encoding to generate embeddings for clustering infrared and visible data. Nevertheless, clusters within
   265	the visible modality tend to experience identity splitting due
   266	to significant variations in lighting and viewpoint caused by
   267	cross-camera discrepancy [37]. It’s worth noting that prior
   268	works mainly focus on enhancing the strategy of using the
   269	clustering algorithm, like bottom-up clustering [37], [36], [38],
   270	with less comprehensive consideration of improving the algorithm itself. To address the problem of cross-camera and crossmodal variation, we introduce a simple and effective modalcamera invariant expansion (MIE) to improve the clustering
   271	algorithm in order to enhance the performance of clustering
   272	cross-camera and cross-modal samples.
   273	Let’s start with the method of obtaining distance encoding
   274	in existing clustering-based approaches. To obtain distance
   275	encoding, each feature is taken as a probe to compute the
   276	k-reciprocal encoding vector [39] with other features:
   277	Di = [di,1 , di,2 , · · · , di,n ]
   278	
   279	(1)
   280	
   281	(
   282	exp(−M(fi , fj )) if fj ∈ R(fi , k1 )
   283	di,j =
   284	0
   285	otherwise.
   286	
   287	(2)
   288	
   289	where fi and fj represent probe and gallery feature, respectively. The function M(·, ·) represents the Mahalanobis
   290	distance, while R(fi , k1 ) refers to the set of k-reciprocal
   291	nearest neighbors for fi , with k1 serving as a hyperparameter
   292	for adjusting the proximity range for distinguishing these
   293	nearest neighbors.
   294	The distance encoding of fi is then expanded by incorporating the distance encodings of the top-k2 most similar
   295	gallery instances from its reciprocal neighbors. In this case,
   296	the distance encoding is effectively fused with contextual
   297	information from neighboring elements:
   298	1 Pk2
   299	Dj
   300	D̃i =
   301	(3)
   302	k2 j=1
   303	where k2 is smaller than k1 to avoid introducing noisy
   304	instances, and D̃i is the expanded distance encoding. Subsequently, we compute the Jaccard distance between the probe
   305	instance and other instances to obtain embedding for clustering:
   306	Pn
   307	min(di,π , dj,π )
   308	J (fi , fj ) = 1 − Pnπ=1
   309	π=1 max(di,π , dj,π )
   310	
   311	(4)
   312	
   313	J (fi ) = [J (fi , f1 ), J (fi , f2 ), · · · , J (fi , fn )]
   314	
   315	(5)
   316	
   317	where min and max operate the element-based minimization and maximization for two input vectors. J (fi ) denotes
   318	the embedding of fi for clustering. Then we take J =
   319	[J (f1 ), J (f2 ), · · · , J (fn )]⊤ as input to clustering algorithm
   320	[12] for assigning pseudo labels ŷ = DBSCAN(J ).
   321	However, the majority of gallery instances are most similar
   322	to the query instances that have the same camera information.
   323	Consequently, the expanded distance encoding is unable to effectively capture the cross-camera neighborhood relationships.
   324	Thus, we equally fuse information from different cameras,
   325	ensuring that samples from different cameras can contribute
   326	equally to the expanded distance encoding. Eq. 3 can be
   327	rewritten as:
   328	D̃i =
   329	
   330	Dicamera =
   331	
   332	1 Pnc camera
   333	D
   334	nc j=1 j
   335	
   336	1 Pk2
   337	1{lcamera = c}Dj
   338	nci j=1 j
   339	
   340	(6)
   341	
   342	(7)
   343	
   344	where 1{·} is the indicator function, nci denotes the number of
   345	instance from camera c, nc indicates the number of camera,
   346	and ljcamera represents the camera label.
   347	Notably, we employ MIE in the phase of intra-modality
   348	clustering and inter-modality clustering. Especially, we can
   349	bridge cross-modality discrepancy when fusing distance coding from visible camera domains and infrared camera domains.
   350	Unlike recent methods [35], [36], [37], [40], [55] focusing
   351	on camera differences, which employ camera proxies for
   352	contrastive learning, we introduce camera information during
   353	global clustering.
   354	
   355	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
   356	
   357	5
   358	
   359	C. Dynamic Neighborhood Clustering
   360	The clustering performance of USL-VI-ReID methods is
   361	critically influenced by hyperparameters in clustering algorithm, such as eps, k1 , and k2 . These hyperparameters affects
   362	clustering objectives, thereby influencing the optimization path
   363	of the network. Specifically, eps defines the search radius for
   364	identifying neighborhoods, k1 adjusts the proximity range to
   365	differentiate nearest neighbors, and k2 determines the top-k2
   366	most similar gallery instances used for expanding distance
   367	encoding. However, in existing methods [15], [16], [17],
   368	these hyperparameters are set empirically and remain constant
   369	during training. It is important to note that the choice of eps
   370	determines the inclusion of noisy instances in the clusters. If
   371	eps is set to be too large, it might result in including noisy
   372	instances, whereas if eps is too small, it might lead to the
   373	exclusion of many valid cross-camera and cross-modality samples. Although some methods take into account the influence
   374	of eps, their consideration is not comprehensive. PUL [10]
   375	maintains small eps to select reliable samples, but this may
   376	not utilize valuable hard positive samples. And this method
   377	requires a predetermined number of identity categories, which
   378	is unknowable. DCCC [11] only consider dynamic downsizing
   379	adjustment of eps and do not consider further scaling up to incorporate more hard positive samples with the help of network
   380	discriminatory power. Additionally, during the early stages
   381	of training, the model’s discriminative capability is limited,
   382	and setting a higher value for k2 may incorporate inaccurate
   383	correlation data into the expanded distance coding. On the
   384	contrary, maintaining an appropriate value of k2 during the
   385	middle and late stages of training prevents the aggregation of
   386	distance coding with cross-modality and cross-camera samples
   387	that exhibit relatively low similarity. The dynamic adjustment
   388	of k2 is not considered in current methods.
   389	In this subsection, we introduce Dynamic Neighborhood
   390	Clustering (DNC) to dynamically adjust the clustering objective. We anticipate that DNC, in collaboration with MIE,
   391	can effectively address the identity splitting issue. First and
   392	foremost, we begin with an assumption: during the early stages
   393	of model optimization (intra-modality training), we aim to
   394	minimize the inclusion of noisy instances within clusters to
   395	enhance the model’s discriminative power. As the optimization
   396	progresses into the middle and later stages (inter-modality
   397	training), our objective shifts towards gradually incorporating
   398	cross-modality and cross-camera positive samples into the
   399	clusters to improve the model’s ability to generalize. To
   400	achieve this, we implement dynamic exponential schedulers
   401	for eps and k2 :
   402	π1 = π2 ∗ σnepochs
   403	
   404	(8)
   405	
   406	π2 = π1 ∗ σbepochs
   407	
   408	(9)
   409	
   410	ϵ2 = ϵ1 ∗ σkepochs
   411	
   412	(10)
   413	
   414	where σn ∈ [0, 1) denote the decay ratio, while σb ∈ (1, +∞]
   415	
   416	Fig. 3. Dynamic schedular in DNC. (a) is the dynamic schedular for eps in
   417	Eq. 11 and Eq. 12, while (b) is for k2 in Eq. 11.
   418	
   419	Fig. 4. Illustratiion of the clustering results of DNC. Taking eps for an
   420	example, we establish upper and lower dynamic range limits for eps denoted
   421	as π2 and π1 , respectively. Initially, eps decreases from π2 to π1 , excluding
   422	noisy instances from clusters. Subsequently, eps expands from π1 to π2 ,
   423	progressively incorporating cross-modality and cross-camera instances into
   424	clusters. In this manner, the model first improves discriminability and then
   425	gradually develops cross-modality and cross-camera generalizability.
   426	
   427	and σk ∈ (1, +∞] indicate growth ratio. π2 is the upper
   428	bound and π1 is the lower bound of eps. ϵ2 (> ϵ1 ) represents
   429	the upper bound of k2 . To better illustrate the use of dynamic strategies for intra-modality and inter-modality training,
   430	we initially define the estimation of modality-specific labels
   431	{ŷ r , ŷ v }:
   432	ŷ r = DBSCAN(J r ; k1 , k2 , eps)
   433	ŷ v = DBSCAN(J v ; k1 , k2 , eps)
   434	
   435	(11)
   436	
   437	where J r and J v are embeddings obtained from MIE. Please
   438	note that k1 and k2 are not hyperparameters for DBSCAN.
   439	They are hyperparameters calculated within MIE. However,
   440	we include them here for clarity in our further demonstration.
   441	Then, we also define estimation of modality-share labels
   442	{ŷ m }:
   443	ŷ m = DBSCAN(J m ; k1 , k2 , eps)
   444	(12)
   445	where J m is the embedding of infrared and visible data
   446	together as input for the MIE module.
   447	As previously mentioned, intra-modality training focuses
   448	on improving the initial discrimination capabilities of the
   449	model, whereas inter-modality training aims to enhance the
   450	model’s ability to generalize across different modalities and
   451	cameras. Consequently, we employ distinct dynamic strategies
   452	for these two stages of training to accomplish this objective,
   453	as shown in Fig. 3. (1) Intra-modality training: The eps
   454	
   455	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
   456	
   457	6
   458	
   459	in Eq. 11 undergoes large-to-small process according to Eq.
   460	8, while k2 in Eq. 11 experiences the small-to-large process
   461	according to Eq. 10. With regards to eps, the clustering
   462	algorithm systematically eliminates noisy samples. For k2 , the
   463	clustering algorithm gradually explores affinities within intracamera instances and subsequently transitions to exploring
   464	affinities between inter-camera instances in collaboration with
   465	eps. (2) Inter-modality training: The eps in Eq. 11 and Eq.
   466	12 first go through a small period of epochs (set to 10 epochs)
   467	from large to small, and then go through the rest (set to 40
   468	epochs) from small to large. The small-to-large stage can serve
   469	as a warm-up process for cross-modal clustering, aiding in
   470	the separation of some solid modality-specific clusters. In the
   471	small-to-large phase for eps, the clustering algorithm progressively incorporates reliable cross-modality and cross-camera
   472	instances into clusters. However, the adjustment of k2 in Eq.
   473	11 and Eq. 12 follows distinct paths. The former experiences
   474	the small-to-large process, while the latter remains the a larger
   475	value of ϵ3 . Combining visible and infrared samples in the
   476	clustering algorithm requires a relatively large value for k2
   477	to enhance the likelihood of linking cross-modal samples with
   478	substantial modal differences. This is essential for maintaining
   479	effective connections between visible and infrared samples in
   480	the clustering algorithm, achieved by keeping a relatively large
   481	value for k2 .
   482	Overall, the dynamic adjustment of eps serves to illustrate
   483	the purpose of DNC, which can be visualized in Fig. 4.
   484	The clustering method plays a pivotal role in unsupervised
   485	approaches, directly influencing network performance. Our
   486	DNC dynamically fine-tunes the clustering objective to control
   487	the network’s learning objectives. Notably, DNC cooperates
   488	with MIE, effectively addressing the challenge of identity
   489	splitting resulting from variations in camera and modal distributions. Further details regarding the effectiveness of DNC
   490	are presented in the experiment section.
   491	
   492	With the intra-modal pseudo labels {ŷ r , ŷ v } and inter-modal
   493	pseudo labels {ŷ m } obtained from MIE and DNC, we propose
   494	Hybrid Modality Contrastive Learning (HMCL) to refine the
   495	distributions between clusters and instances.
   496	Unsupervised methods commonly take refined InfoNCE
   497	[30], [31] as their loss function, which can be defined as:
   498	PX
   499	×Z
   500	exp(qi · ϕ[ŷi ]/τ )
   501	(13)
   502	L=−
   503	log PI
   504	k=0 exp(qi · ϕ[k]/τ )
   505	i=1
   506	where P and Z indicate the number of sampled individuals
   507	and instances per individual, respectively. qi denotes L2normalized query instance in training mini-batch. ϕ[ŷi ] is
   508	the positive cluster representation of qi and ϕ[k] represents
   509	each cluster representation stored in memory, where cluster
   510	representations are obtained by averaging the instance features
   511	in the clusters. τ is the temperature factor and I is the total
   512	number of clusters. Then a momentum updating strategy [31]
   513	is employed to update the memory after each iteration:
   514	(i = 1, 2, ..., P × Z)
   515	
   516	ϕrI [i] = fir
   517	ϕvI [i] = fiv
   518	ϕrC [ŷir ] =
   519	
   520	(14)
   521	
   522	1
   523	
   524	X
   525	
   526	Oŷr r
   527	
   528	ϕvC [ŷiv ] =
   529	where ϕrI
   530	
   531	fir
   532	
   533	fir ∈Oŷr r
   534	i
   535	
   536	i
   537	
   538	1
   539	
   540	X
   541	
   542	Oŷvv
   543	i
   544	
   545	fiv ∈Oŷvv
   546	
   547	(15)
   548	
   549	fiv
   550	i
   551	
   552	and ϕvI
   553	
   554	represent instance-level memories for infrared and visible modalities, while ϕrC and ϕvC indicate
   555	r(v)
   556	r(v)
   557	cluster-level memories. O r(v) denotes the ŷi -th cluster set
   558	ŷi
   559	
   560	in infrared or visible modality, and |·| represents the number
   561	of instances in specific cluster. To establish a stable starting
   562	point for optimization, we initialize the cluster-level memory
   563	by taking the average value. After each iteration, we randomly
   564	select instances to update memory instead of employing momentum updating strategy, ensuring the real-time update of
   565	memory. In the case of the visible modality, to fully leverage
   566	the CA modality, we randomly update visible memory using
   567	(v,c)
   568	CA and visible features, specifically ϕvI [i] ← fi
   569	and
   570	(v,c)
   571	v
   572	v
   573	. As for infrared modality, we update memory
   574	ϕC [ŷi ] ← fi
   575	by ϕrI [i] ← fir and ϕrC [ŷir ] ← fir .
   576	The cluster-level contrastive loss for visible and infrared
   577	modalities can be expressed as:
   578	LvC = −
   579	LrC = −
   580	
   581	D. Hybrid Modality Contrastive Learning
   582	
   583	ϕ[ŷi ] ← λϕ[ŷi ] + (1 − λ)qi
   584	
   585	Similar to [21], we adopt a strategy of randomly sampling
   586	instances to update cluster representations. Subsequently, we
   587	introduce cluster-level and instance-level losses to refine global
   588	and partial distributions, respectively.
   589	For intra-modality training, we construct two kinds of
   590	memories, i.e., cluster-level and instance-level memories:
   591	
   592	2×P
   593	×Z
   594	X
   595	i=1
   596	PX
   597	×Z
   598	
   599	(v,c)
   600	
   601	exp(q
   602	· ϕvC [ŷiv ]/τ )
   603	log PI i
   604	v
   605	v
   606	k=0 exp(qi · ϕC [k]/τ )
   607	
   608	exp(qir · ϕrC [ŷir ]/τ )
   609	log PI
   610	r
   611	r
   612	k=0 exp(qi · ϕC [k]/τ )
   613	i=1
   614	
   615	(16)
   616	
   617	(17)
   618	
   619	(v,c)
   620	
   621	where qi
   622	denote the query features in the mini-batch from
   623	visible and CA modalites, and qir indicates the query feature
   624	from infrared modality.
   625	To explore the instance relationships during the training
   626	process, the instance-level contrastive loss can be formulated
   627	as:
   628	2×P
   629	×Z
   630	(v,c)
   631	X
   632	exp(q
   633	· ϕvI [ŷiv ]/τ )
   634	LvI = −
   635	log PI i
   636	(18)
   637	v
   638	v
   639	k=0 exp(qi · ϕI [k]/τ )
   640	i=1
   641	LrI = −
   642	
   643	PX
   644	×Z
   645	
   646	exp(qir · ϕrI [ŷir ]/τ )
   647	log PI
   648	r
   649	r
   650	k=0 exp(qi · ϕI [k]/τ )
   651	i=1
   652	
   653	(19)
   654	
   655	For inter-modality training, we enhance the training process
   656	by introducing global cluster-level memory and instance-level
   657	memory, in addition to the intra-modality training, denoted as:
   658	(v,r)
   659	ϕm
   660	I [i] = fi
   661	X
   662	1
   663	(v,r)
   664	m
   665	(20)
   666	ϕm
   667	fi
   668	C [ŷi ] =
   669	m
   670	Oŷm f (v,r) ∈Om
   671	i
   672	
   673	i
   674	
   675	ŷ m
   676	i
   677	
   678	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
   679	
   680	7
   681	
   682	Algorithm 1: DMIC approach
   683	Input: Unlabeled infrared training data R, unlabeled visible training data
   684	V, epochs number epoch and training iterations number iter.
   685	Output: Backbone network parameters θ.
   686	1: for κ = 1, 2, ..., epoch do #Intra − modality training
   687	2: Adjust eps and k2 by Eq. (8)-(10);
   688	3: Generate intra-modality pseudo labels {ŷ r , ŷ v } by Eq. (1), (2), (4),
   689	(5), (6), (7) and (11);
   690	4: Initialize memories {ϕrI , ϕvI , ϕrC , ϕvC } by Eq.(15);
   691	5: for π = 1, 2, ..., iter do
   692	6:
   693	Update θ by minimizing Eq. (16)-(19);
   694	7:
   695	Update memories {ϕrI , ϕvI , ϕrC , ϕvC }.
   696	8: end for
   697	9: end for
   698	10: for κ = 1, 2, ..., epoch do #Inter − modality training
   699	11: Adjust eps and k2 by Eq. (8)-(10);
   700	12: Generate intra-modality labels {ŷ r , ŷ v } and inter-modality pseudo
   701	labels {ŷ m } by Eq. (1), (2), (4), (5), (6), (7), (11) and (12);
   702	r
   703	v
   704	m
   705	13: Initialize memories {ϕrI , ϕvI , ϕm
   706	I , ϕC , ϕC , ϕC } by Eq. (15) and
   707	(20);
   708	14: for π = 1, 2, ..., iter do
   709	15:
   710	Update θ by minimizing Eq. (16)-(19) and (21)-(22);
   711	r
   712	v
   713	m
   714	16:
   715	Update memories {ϕrI , ϕvI , ϕm
   716	I , ϕC , ϕC , ϕC }.
   717	17: end for
   718	18: end for
   719	19: Return θ
   720	
   721	where Oŷmm denotes the ŷim -th cluster set according to DBi
   722	SCAN in the process of inter-modality clustering. When updating these memories, CA modality features can be included
   723	(v,r,c)
   724	through random selection, as indicated by ϕm
   725	I [i] ← fi
   726	(v,r,c)
   727	and ϕm
   728	.
   729	C [i] ← fi
   730	The cluster-level and instance-level contrastive losses for
   731	inter-modality training can be expressed as:
   732	Lm
   733	C =−
   734	Lm
   735	I =−
   736	
   737	3×P
   738	×Z
   739	X
   740	i=1
   741	3×P
   742	×Z
   743	X
   744	i=1
   745	
   746	(v,r,c)
   747	
   748	m
   749	exp(qi
   750	· ϕm
   751	C [ŷi ]/τ )
   752	log PI
   753	(v,r,c)
   754	· ϕm
   755	C [k]/τ )
   756	k=0 exp(qi
   757	
   758	(21)
   759	
   760	(v,r,c)
   761	
   762	m
   763	exp(qi
   764	· ϕm
   765	I [ŷi ]/τ )
   766	log PI
   767	(v,r,c)
   768	· ϕm
   769	I [k]/τ )
   770	k=0 exp(qi
   771	
   772	Fig. 5. Example images from visible-infrared pedestrian databases. The
   773	images from the upper row are in visible modality, the images from the
   774	second row are in infrared modality, and the images from the last row are in
   775	CA modality.
   776	
   777	(22)
   778	
   779	(v,r,c)
   780	where qi
   781	indicate that the query features are sampled from
   782	
   783	visible, infrared and CA modalities.
   784	During the training process, HMCL firstly perform intramodality learning and explore reliable intra-modal clusters.
   785	This preference stems from the simplicity of initiating intramodal learning before inter-modal learning. The implementation of direct inter-modal clustering results in an excessive
   786	introduction of noise. Such an increase in noise, especially
   787	during the early stages of model training, accumulates progressively, thereby negatively impacting the final efficacy of
   788	the model. As a result, transitioning from intra-modal to intermodal learning aids in steering the model toward a more
   789	organized optimization path. Moreover, with the combined
   790	influence of MIE and DNC, cross-modal discrepancies are
   791	eliminated. Simultaneously, cross-camera information steadily
   792	aggregates, providing robustness against cross-camera disturbances.
   793	E. Optimization Algorithm
   794	The proposed DMIC network naturally integrates MIE,
   795	DNC and HMCL into a unified framework. These key modules
   796	can establish a synergistic effect to bolster the model’s robustness against cross-modality and cross-camera discrepancies.
   797	
   798	Since we have both intra-modal and inter-modal training
   799	stages, the overall loss function for DMIC can be formally
   800	expressed as:
   801	r
   802	v
   803	m
   804	L = φ1 (LrI + LvI + Lm
   805	I ) + φ2 (LC + LC + LC )
   806	
   807	(23)
   808	
   809	where φ1 and φ2 denote the trade-off weights for the clusterlevel and instance-level loss functions. The optimization process of our method is outlined in Algorithm 1, with epoch
   810	representing the number of epochs, κ signifying the epoch
   811	index, and iter indicating the number of training iterations.
   812	IV. E XPERIMENTS E VALUATION

codex
第三篇 DMIC 的叙事很清楚：它不是说“跨模态关联难”，而是说“聚类阶段同时被模态和相机差异污染，导致身份拆裂”。第四篇 DTST 的方法部分很短，机制相对更像把已有 VDT 加上动态令牌选择，我会顺手读它后面的消融来判断 reviewer 买账点。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf' - | nl -ba | sed -n '260,727p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   260	C. Comparisons with State-of-the-art Methods
   261	
   262	Datasets. We conduct experiments on the CARGO [15]
   263	dataset and AG-ReID dataset [19]. Compared to AG-ReID,
   264	the CARGO dataset offers a larger scale, greater diversity, and
   265	is the first large-scale synthetic dataset for AGPReID. Table I
   266	summarizes both datasets. For CARGO, 51,451 images with
   267	2,500 IDs are used for training, and 51,024 images with 2,500
   268	IDs for testing. Four evaluation protocols (ALL, A↔A, G↔G,
   269	and A↔G) assess model performance, with A↔A and G↔G
   270	testing aerial and ground data separately, and A↔G using
   271	cross-view retrieval. The training set is consistent across all
   272	protocols.
   273	For AG-ReID, 11,554 images with 199 IDs are used for
   274	training, and 12,464 images with 189 IDs for testing. Two
   275	protocols, A→G and G→A, evaluate cross-view retrieval, with
   276	the former testing 1,701 aerial queries against 3,331 ground
   277	galleries, and the latter 962 ground queries against 7,204 aerial
   278	galleries.
   279	Evaluation Metrics. Following the common setting, we
   280	utilize three metrics to evaluate our model: the cumulative
   281	matching characteristic at Rank1, mean Average Precision
   282	(mAP), and mean Inverse Negative Penalty (mINP).
   283	
   284	We evaluate our proposed DTST against state-of-the-art
   285	methods on the CARGO and AG-ReID datasets, comprising
   286	CNN-based approaches (BoT [21], SBS [20], MGN [22],
   287	AGW [23]) and transformer-based methods (ViT [24], VDT
   288	[15]).
   289	Performance on CARGO. Table II shows the results of
   290	our proposed DTST and other competitive methods on the
   291	CARGO dataset. The proposed DTST achieves state-of-the-art
   292	performance. For example, DTST surpasses the mAP/Rank1/mINP of the baseline by 1.18%/3.13%/0.43% on the aerialto-ground (A↔G) protocol of CARGO. Besides, DTST also
   293	brings different degree of benefits to other CARGO protocols.
   294	Specifically, our proposed DTST exceeds VDT on mAP/Rank1/mINP by 1.51%/1.60%/2.00% on the ALL of AG-ReID.
   295	Demonstrating the effectiveness of the Dynamic Token Selective Transformer in mitigating view bias and improving identity representation. Previous view-homogeneous ReID
   296	methods show significant performance degradation under the
   297	view-heterogeneous AGPReID protocols, especially in cases
   298	of considerable view variation. This decline underscores how
   299	view bias hampers the consistency of identity features across
   300	views. Unlike existing methods that overlook this key challenge and struggle to generalize in heterogeneous scenarios,
   301	our approach adaptively selects the most critical tokens using
   302	a top-k selective mechanism. This token selection not only
   303	maintains accuracy but even enhances it, resulting in more
   304	effective and compact long-range modeling.
   305	Performance on AG-ReID. To further demonstrate the performance of our model, we also carry out similar experiments
   306	on the AG-ReID dataset. The outcomes are detailed in Table
   307	III. As depicted in Table III, we compare two challenging
   308	protocols: A→G and G→A. It is noteworthy that VDT serves
   309	as a strong baseline. However, our proposed method, DTST,
   310	demonstrates a significant enhancement, outperforming VDT
   311	by 0.57% for the A→G Rank-1 protocol and 1.04% for the
   312	G→A Rank-1 protocol. This consistent improvement suggests
   313	that the superior performance of DTST does not stem from a
   314	robust baseline VDT but from the proposed method itself.
   315	
   316	B. Implementation Details
   317	
   318	D. Ablation Study
   319	
   320	Our model is implemented using the PyTorch framework,
   321	with experiments conducted on an NVIDIA 4090 GPU. We
   322	use the View-decoupled Transformer (VDT) as the baseline, which includes 12 transformer encoder blocks based on
   323	the ViT-Base architecture, pre-trained on ImageNet with a
   324	patch size and stride of 16×16. Input images are resized to
   325	256×128 during preprocessing. The training process employs
   326	the Stochastic Gradient Descent (SGD) optimizer with a cosine
   327	learning rate decay, starting at 8 × 10−3 and reducing to
   328	1.6 × 10−6 over 120 epochs. The batch size is set to 128,
   329	comprising 32 identities with four images per identity. Our
   330	token selector module features a two-head transformer encoder
   331	
   332	In this section, we provide ablation study to investigate
   333	several key components of our DTST. We also delved into
   334	the number of attention heads, token quantities, and token
   335	positions. Notably, all ablation experiments are conducted on
   336	the on the CARGO dataset.
   337	Effects of Visual Token Selector (VTS). We first explore the
   338	effectiveness with placing the Visual Token Selector before the
   339	final layer of the View-decoupled Transformer. In this setup,
   340	all other settings,such as the number of attention heads and
   341	selected tokens, remain constant. Table IV shows the results,
   342	where model-a lacks a visual token selector, whereas model-b
   343	incorporates one. From the Table, we can observe a 5.63%
   344	
   345	IV. E XPERIMENTS
   346	A. Experiment settings
   347	
   348	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:56 UTC from IEEE Xplore. Restrictions apply.
   349	
   350	TABLE I: THE DETAILED SUMMARY OF THE DATASET PROPERTIES INVOLVED IN THIS PAPER, INCLUDING
   351	AG-ReID and CARGO.
   352	Dataset
   353	AG-ReID [19]
   354	CARGO [15]
   355	
   356	Data
   357	Real
   358	Synthetic
   359	
   360	#PersonID
   361	388
   362	5,000
   363	
   364	#Camera
   365	2 (1A+1G)
   366	13 (5A+8G)
   367	
   368	#Image
   369	21,893
   370	108,563
   371	
   372	#Height
   373	15 ∼ 45m
   374	5 ∼ 75m
   375	
   376	TABLE II: Performance comparison of the mainstream methods under four settings of the proposed CARGO dataset. “ALL”
   377	denotes the overall retrieval performance of each method. “G↔G,” “A↔A,” and “A↔G” represent the performance of each
   378	model in several specific retrieval patterns. Rank1, mAP, and mINP are reported (%). The best performance is shown in bold.
   379	Protocol 1: ALL
   380	
   381	Method
   382	SBS [20]
   383	PCB [37]
   384	BoT [21]
   385	MGN [22]
   386	VV [40, 41]
   387	AGW [23]
   388	ViT [24]
   389	VDT [15]
   390	DTST (Ours)
   391	
   392	Protocol 2: G↔G
   393	
   394	SBS [20]
   395	BoT [21]
   396	OSNet [25]
   397	ViT [24]
   398	VDT [15]
   399	DTST (ours)
   400	
   401	mAP
   402	
   403	mINP
   404	
   405	Rank1
   406	
   407	mAP
   408	
   409	mINP
   410	
   411	Rank1
   412	
   413	mAP
   414	
   415	mINP
   416	
   417	Rank1
   418	
   419	mAP
   420	
   421	mINP
   422	
   423	50.32
   424	51.00
   425	54.81
   426	54.81
   427	45.83
   428	60.26
   429	61.54
   430	62.82
   431	64.42
   432	
   433	43.09
   434	44.50
   435	46.49
   436	49.08
   437	38.84
   438	53.44
   439	53.54
   440	54.22
   441	55.73
   442	
   443	29.76
   444	32.20
   445	32.40
   446	36.52
   447	39.57
   448	40.22
   449	39.62
   450	39.92
   451	41.92
   452	
   453	72.31
   454	74.10
   455	77.68
   456	83.93
   457	72.31
   458	81.25
   459	82.14
   460	79.46
   461	78.57
   462	
   463	62.99
   464	67.60
   465	66.47
   466	71.05
   467	62.99
   468	71.66
   469	71.34
   470	70.60
   471	72.40
   472	
   473	48.24
   474	55.10
   475	51.34
   476	55.20
   477	48.24
   478	58.09
   479	57.55
   480	57.89
   481	62.10
   482	
   483	67.50
   484	55.00
   485	65.00
   486	65.00
   487	67.50
   488	67.50
   489	80.00
   490	82.50
   491	80.00
   492	
   493	49.73
   494	44.60
   495	49.79
   496	52.96
   497	49.73
   498	56.48
   499	64.47
   500	64.06
   501	63.31
   502	
   503	29.32
   504	27.00
   505	29.82
   506	36.78
   507	29.32
   508	40.40
   509	47.07
   510	44.67
   511	44.67
   512	
   513	31.25
   514	34.40
   515	36.25
   516	31.87
   517	31.25
   518	43.57
   519	43.13
   520	47.50
   521	50.63
   522	
   523	29.00
   524	30.40
   525	32.56
   526	33.47
   527	29.00
   528	40.90
   529	40.11
   530	42.21
   531	43.39
   532	
   533	18.71
   534	20.10
   535	21.46
   536	24.64
   537	18.71
   538	29.39
   539	28.20
   540	29.03
   541	29.46
   542	
   543	Protocol 1: A→G
   544	Rank1
   545	mAP
   546	mINP
   547	73.54
   548	59.77
   549	70.01
   550	55.47
   551	72.59
   552	58.32
   553	81.28
   554	72.38
   555	82.91
   556	74.44
   557	51.06
   558	83.48
   559	74.51
   560	49.86
   561	
   562	Protocol 2: G→A
   563	Rank1
   564	mAP
   565	mINP
   566	73.70
   567	62.27
   568	71.20
   569	58.83
   570	74.22
   571	60.99
   572	82.64
   573	73.35
   574	83.68
   575	75.96
   576	49.39
   577	84.72
   578	76.05
   579	50.04
   580	
   581	TABLE IV: Ablation study of model key designs on CARGO
   582	dataset. Rank1, mMAP, and mINP are reported(%). Best in
   583	blod.
   584	Method
   585	model-a
   586	model-b (Ours)
   587	
   588	Protocol 4: A↔G
   589	
   590	Rank1
   591	
   592	TABLE III: Quantitative evaluation of the mainstream methods
   593	under two settings of AG-ReID dataset. “A↔G”, and “G↔A”
   594	represent the performance in two specific patterns. Rank1,
   595	mAP, and mINP are reported (%). Best marked in bold.
   596	Method
   597	
   598	Protocol 3: A↔A
   599	
   600	Visual Token Selector
   601	✘
   602	✔
   603	
   604	Protocol: A↔G
   605	Rank1
   606	mAP
   607	mINP
   608	45.00
   609	42.05
   610	30.26
   611	50.63
   612	43.39
   613	29.46
   614	
   615	improvement in rank-1 accuracy and 1.34% increase in mAP
   616	accuracy under Protocol A→G, which indicates that the token
   617	selection strategy effectively filters out tokens with discriminative features and eliminates identity-irrelevant tokens, thereby
   618	enhancing better identity representation.
   619	Number of Heads. We also evaluate the performance of
   620	VTS with different numbers of heads, specifically 2, 4, and
   621	8 heads in Table V. Interestingly, using more heads results
   622	in a decrease in accuracy. Specifically, when increasing the
   623	number of heads from 2 to 4, there is a 3.64% decline in
   624	rank-1 accuracy and 0.93% drop in mAP. This suggests that
   625	a higher number of heads may dilute the model’s ability to
   626	
   627	TABLE V: Ablation study on the number of attention heads,
   628	token quantities, and token positions using the CARGO
   629	dataset. “Head-Num.” signifies the quantity of attention heads,
   630	“T-Num.” demotes the number of token, and “T-Position.”
   631	indicates the specific position where each token is locate.
   632	Performance is assessed through Rank1, mAP, and mINP(%),
   633	with the best results highlighted in blod.
   634	Method
   635	
   636	Head-Num.
   637	
   638	T-Num.
   639	
   640	T-Position.
   641	
   642	model-1
   643	model-2
   644	model-3
   645	model-4
   646	model-5
   647	model-6 (Ours)
   648	
   649	8
   650	8
   651	8
   652	8
   653	4
   654	2
   655	
   656	2
   657	3
   658	3
   659	32
   660	2
   661	2
   662	
   663	last layer
   664	last layer
   665	second to last layer
   666	second to last layer
   667	last layer
   668	last layer
   669	
   670	Protocol: A↔G
   671	Rank1
   672	mAP
   673	mINP
   674	46.25
   675	42.56
   676	30.16
   677	45.00
   678	41.28
   679	28.83
   680	46.88
   681	41.04
   682	28.12
   683	40.00
   684	36.58
   685	24.73
   686	46.88
   687	42.46
   688	29.79
   689	50.63
   690	43.39
   691	29.46
   692	
   693	focus on critical identity features, potentially introducing noise
   694	and decreasing overall model performance. One underlying
   695	reason may be model over-fitting, as a greater number of heads
   696	could increase the model’s complexity without corresponding
   697	improvements in performance. Another potential explanation
   698	might be that more heads may dilute the importance of the
   699	most vital tokens, leading to less effective feature aggregation.
   700	Number of Tokens Selected. Keeping other variables constant, we analyze the impact of different numbers of token
   701	selections on model performance in Table V. We vary the
   702	number of tokens to 2, 3, 5. The findings reveal that selecting
   703	2 or 3 tokens yields superior results across all evaluation
   704	metrics, i.e. Rank-1 accuracy, mAP, and mINP. Specifically,
   705	we increase the number of selected tokens beyond 3, but the
   706	performance fails to show any improvement, indicating that
   707	opting for fewer but more critical tokens enables the model
   708	to concentrate better on pivotal identity features. In contrast,
   709	selecting more tokens may introduce irrelevant information,
   710	thereby compromising overall accuracy. When our method is
   711	applied in the same setup, choosing 3 tokens compared to
   712	
   713	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:56 UTC from IEEE Xplore. Restrictions apply.
   714	
   715	2 tokens results in a decrease of 1.25% in rank-1 accuracy,
   716	1.28% in mAP, and 1.33% in mINP, highlighting the tradeoff between token quantity and model’s focus on essential
   717	features.
   718	Token positions. The insertion position of VST, whether in the
   719	last or second-to-last layer, also affects model performance, as
   720	shown in Table V. When the fixed number of heads is 8 and
   721	the number of tokens is 3, model-3 achieves a higher Rank1 accuracy at 46.88%, but both mAP and mINP decrease.
   722	The reason behind this could be that tokens in shallow layers
   723	contain more detailed information, while tokens in deeper
   724	layers extract higher-level semantic information. As a result,
   725	the information within each token becomes more refined,
   726	leading to a higher compressibility ratio.
   727	V. C ONCLUSION AND F UTURE W ORK

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf' - | nl -ba | sed -n '58,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    58	Introduction
    59	
    60	Person re-identification (Re-ID), a technology focused on cross-camera identity matching in
    61	disjoint surveillance networks, has emerged as a critical research area owing to its vital role in
    62	intelligent security and public safety applications. Supervised Re-ID methods have seen remarkable
    63	improvements due to the development of deep learning structures, particularly convolutional neural
    64	network architectures. Concurrently, researchers have progressively developed expanded benchmark datasets with growing image volumes to support methodological innovations in this domain
    65	[15]. However, in the context of training sets and test sets coming from the same domain, even when
    66	trained on extensive datasets, Re-ID models typically exhibit considerable performance degradation
    67	because of domain gaps resulting from differences in illumination and camera perspective. On the
    68	other hand, annotating datasets to improve performance consumes a lot of manpower and resources.
    69	Therefore, scholars have expressed interest in unsupervised domain adaptation (UDA).
    70	In UDA, a model initially trained on labeled source data is adapted to perform effectively on an
    71	unlabeled target domain, making it a cross-domain problem. Existing approaches can be classified
    72	into two categories: one leverages domain alignment techniques such as using cross-domain similarity transfer frameworks to enhance domain alignment [7]. The other employs clustering algorithms
    73	to group unlabeled datasets by ID and trains the network using pseudo-labels generated from clustering. Generally, pseudo-label-based algorithms achieve better performance on the target domain.
    74	Most recent UDA methods focus on refining pseudo-labels to reduce the impact of noisy labels [29],
    75	but these methods ignore the rich source domain knowledge. If a more powerful and better learning
    76	initial model can be learned from multiple source domains first, then the subsequent pseudo-label
    77	learning can be supervised from less noisy labels, making feature learning more accurate, thereby
    78	further refining more accurate pseudo-labels. How to obtain a model with cross-domain capabilities
    79	from multiple existing datasets, which introduces domain generalization (DG). DG refers to
    80	learning a generalizable model from one or more datasets that can be applied to any unknown
    81	domain. Existing DG methods use meta-learning, style transfer, and instance normalization (IN)
    82	to learn domain-invariant knowledge from multiple source domains [23]. However, most DG works
    83	only focus on generalization capabilities in unseen domains. With the development of algorithms
    84	such as object detection, it is becoming easier to obtain unlabeled person image data. Therefore,
    85	the combination of DG and domain adaptation methods is an important part of UDA.
    86	As a key technology in Internet of Things (IoT) surveillance systems, person Re-ID has certain
    87	requirements for the models deployed on related devices. Edge devices typically face constraints
    88	in processing power and memory capacity, making it difficult to support complex and largescale models [21]. Additionally, many IoT applications require local data processing at the edge
    89	to safeguard user privacy or minimize data transmission needs. To address these challenges,
    90	lightweight models can operate locally in real-time using low resources, eliminating the need
    91	for cloud transmission while meeting the demands of low resource consumption and privacy. In
    92	conclusion, developing a lightweight, UDA Re-ID method is essential for the practical deployment
    93	of person Re-ID systems.
    94	In addition to lightweight requirements, person Re-ID performance is often affected by various
    95	factors in real-world scenarios, such as significant variations in pose and lighting, degraded pixel
    96	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
    97	
    98	Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID
    99	
   100	171:3
   101	
   102	quality, and occlusions [22]. Most domain adaptation methods focus solely on global features
   103	during feature extraction, which may fail to accurately represent pedestrian information in practical
   104	settings. In recent years, transformer-based models have gained popularity among researchers
   105	due to their ability to directly model relationships between arbitrary image patches. For instance,
   106	introducing attention pyramid Transformers or improving positional encoding can enhance the
   107	accuracy of visual tasks [16]. Additionally, some approaches integrate other pedestrian information,
   108	such as human pose estimation [10] and motion [26] into Re-ID. However, these models tend to
   109	be large and complex. Therefore, if a model can learn more fine-grained and richer features while
   110	remaining lightweight, it would be better suited for adapting to new domains.
   111	In general, in view of the fact that existing methods do not fully utilize multiple source domains
   112	and do not consider environmental issues such as device limitations and light occlusion in actual
   113	scenes, this study proposes a lightweight multi-source UDA method based on overall level features
   114	and partial level features. Specifically, the work of this study can be summarized as follows:
   115	— In order to make full use of the source domain data in multi-source domain UDA, this article
   116	combines the pseudo-label-based domain adaptation method with the meta-learning method.
   117	Through Self-paced Meta-Learning (SpML) and style enhancement technology, the model
   118	simulates the human learning process and learns the source domain knowledge from easy
   119	to difficult so that the model can learn domain-invariant knowledge, thereby improving the
   120	quality of pseudo-labels in the domain adaptation stage.
   121	— In response to the problems of limited processing equipment resources in actual scenarios
   122	and environmental factors such as object occlusion and light changes that interfere with
   123	image recognition accuracy, this article uses deep separable convolution to build a lightweight
   124	network and designs a feature extraction network consisting of overall level branches and
   125	partial level branches to obtain richer feature information from pedestrian images and improve
   126	recognition capabilities.
   127	— This article validates the model on multiple public datasets and their different combinations
   128	and also tests the model performance using a set of datasets from real application scenarios.
   129	2
   130	2.1
   131	
   132	Related Work
   133	Supervised Person Re-ID
   134	
   135	Supervised person Re-ID is the most widely studied and outstanding direction. This method requires
   136	a big amount of training data with annotated information, which can fully utilize the advantages
   137	of annotated data and achieve high recognition accuracy in specific scenarios. With the continuous
   138	development of deep learning and attention mechanisms, supervised Re-ID methods have achieved
   139	superior performance. For example, Luo et al. [18] presented an efficient multi-scale channel
   140	attention network to learn robust and more discriminative features through cross-channel attention
   141	modules. In recent years, with the popularity of Transformer, He et al. [9] proposed a Transformerbased Re-ID framework and tried to use pure Transformer for person Re-ID research for the first
   142	time. Subsequently, Li et al. [14] studied how to apply the pre-trained visual language model
   143	contrastive language–image pretraining to fine-grained image Re-ID tasks, further improving
   144	the accuracy of Re-ID. Furthermore, Liu et al. [17] proposed Mix-Modality Re-ID with CrossIdentity Discrimination Harmonization loss for hyperspherical identity-modality harmonization
   145	and bridge-based similarity optimization strategy to enable robust day-night mixed retrieval.
   146	While supervised techniques have demonstrated outstanding performance in the same scenario, in
   147	practical applications, due to the difficulty of data annotation, more and more researchers have
   148	turned their attention to the task of unsupervised method.
   149	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   150	
   151	171:4
   152	2.2
   153	
   154	X. Zhang et al.
   155	Unsupervised Person Re-ID
   156	
   157	According to whether the target domain can be used for training, unsupervised cross-domain
   158	pedestrian Re-ID methods are broadly classified into two types: domain adaptation methods and
   159	DG methods. Domain adaptation methods investigate the relationship between source and target
   160	domain pictures, as well as pseudo-label training for the target domain. Li et al. [12] propose a
   161	task-specific feature reorganization mechanism, as well as a feature learning approach for alignment
   162	of feature distributions at the instance level and ensuring identity consistency across cameras
   163	to address the problem of paired sample loss across cameras. The DG method hopes to obtain
   164	a generalized representation of pedestrians using only a limited source domain, which can be
   165	broadly classified into four categories: representation learning-based, distribution alignmentbased, meta-learning-based, and data enhancement-based methods. Zhou et al. [33] introduce IN
   166	in the omni-scale network (OSNet) system, calibrate samples through internal statistics, and
   167	thus eliminate the influence of specific environmental shadows on style. Ni et al. [20] propose a
   168	similarity learning agent to reduce domain-specific style, enable the model to learn common features
   169	between different identities, and design a locally guided self-distillation algorithm to enhance the
   170	generalization capabilities of global characteristics. Zhao et al. [31] propose a memory-based multisource meta-learning framework (M3L), which significantly improves the model’s generalization
   171	ability to unseen domains by simulating the DG training-test process and combining nonparametric
   172	memory identity loss with meta-batch normalization layers. Xu et al. [28] propose a Style Screening
   173	and Continuous Utilization framework, which accumulates positive styles by Generalization GainGuided Dynamic Style Memory and continuously utilizes them through a dual-branch Collaborative
   174	Style Training. This framework achieves efficient positive and negative style management for the
   175	first time in federated DG Re-ID, improving cross-domain generalization performance. Choi et al.
   176	[3] propose Meta Batch-IN, which simulates under- and over-style-normalization failures in a
   177	meta-learning pipeline using learnable batch-IN layers, and enhances generalization via diversified
   178	meta-train loss and cyclic inner-updating, enabling the model to robustly adapt to unseen domains
   179	without overfitting to source styles. Xu et al. [27] propose a Bi-Stream Generative Model that learns
   180	camera-invariant global features by filtering random interference in one stream and pedestrianaligned local features via information-complete densely semantically aligned part maps in another
   181	stream, while introducing a part-weighted loss to mitigate missing part influence, enhancing
   182	generalization in unseen domains. Although domain adaptation and DG have certain differences in
   183	theory and methods, their goal is to improve the performance of the model in different domains.
   184	Therefore, effectively combining the two is critical for enhancing unsupervised person Re-ID.
   185	2.3
   186	
   187	Domain Adaptation
   188	
   189	In the field of person Re-ID, most UDA methods use pseudo-labeling methods. Zheng et al. [32]
   190	proposed a domain alignment method based on a ladder alignment matcher and integrated it
   191	into the clustering framework. Lee et al. [11] proposed a curriculum learning framework and
   192	scheduler based on camera labels, along with a loss of camera-diversity to address the challenges
   193	of significant intraclass variations and interclass distinctions. Li et al. [13] proposed a method for
   194	logical reasoning on pseudo-labels and optimizing noisy labels and proposed a module for multiview
   195	information exchange to obtain the global consistency of pedestrians. Han et al. [8] proposed label
   196	correction based on silhouette coefficient and contrast loss regularization based on loose-tight
   197	clustering guidance to reduce the negative impact of pseudo-label noise on model training. Zhu
   198	et al. [34] propose a cluster environment-aware learning method for active domain adaptation. By
   199	exploring the informativeness and representativeness of target samples in a cluster-aware manner,
   200	it selects potentially most valuable instances under domain shift. The approach further employs
   201	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   202	
   203	Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID
   204	
   205	171:5
   206	
   207	Fig. 1. Overall architecture diagram of the algorithm. The upper left corner is the experimental data required;
   208	Figure (a) is the model structure diagram, Figure (b) is the multi-source domain learning process, and Figure
   209	(c) is the unsupervised training on the target domain.
   210	
   211	information maximization to enhance the reliability of target pseudo-labels, thereby improving
   212	model performance. Despite these works have shown remarkable results, these approaches rely
   213	solely on restricted data from a single domain for pre-training, leaving the abundant labeled data
   214	underutilized.
   215	3
   216	
   217	Methods
   218	
   219	Multi-source domain adaptation problem setting: For a source domain with 𝑁 datasets, it is repre
   220	𝑀𝑖
   221	𝑁
   222	sented as 𝑆 = {𝐷𝑖 }𝑖=1
   223	, where 𝐷𝑖 = 𝑃 𝑗 , 𝐿 𝑗 𝑗=1
   224	, 𝑀 represents the total image count in the dataset, 𝑃
   225	is the image, and 𝐿 is the label corresponding to the image. The target domain 𝑇 = {𝐷𝑡 }, where
   226	 𝑀𝑡
   227	𝐷𝑡 = 𝑃 𝑗 𝑗=1
   228	. The goal of this article is to learn domain-invariant knowledge from 𝑁 source domains
   229	and use this knowledge for pseudo-label acquisition and optimization of the target domain.
   230	Based on the above goals, the framework for multi-source domain adaptation is proposed in this
   231	article as shown in Figure 1. First, we will introduce a lightweight feature extraction network based
   232	on the whole and part as shown in Figure 1(a), which is abbreviated as WPOSNet in this article.
   233	This model can provide richer pedestrian image features and improve the extraction capability in
   234	complex situations. After that, based on this model, we use a SpML training method [30] to learn
   235	feature extraction capabilities that are not affected by domain differences from multiple source
   236	domains, as shown in Figure 1(b). Finally, we will introduce the unsupervised learning process on
   237	the target domain, which is based on the mutual mean-teaching (MMT) framework, as shown
   238	in Figure 1(c).
   239	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   240	
   241	171:6
   242	
   243	X. Zhang et al.
   244	
   245	Fig. 2. Basic block OSblock structure.
   246	
   247	3.1
   248	
   249	WPOSNet
   250	
   251	Based on the lightweight OSNet, this article designs feature extraction branches based on the whole
   252	and parts. As shown in Figure 1(a), the upper branch extracts features at the component level,
   253	and the lower branch extracts overall features. Finally, based on the extraction results of the two
   254	branches, features that focus on both the parts and the whole are obtained.
   255	First, the entire network is composed of the basic block OSblock, and its structure is shown in
   256	Figure 2. This module first performs 1 × 1 convolution on the input and then performs multi-branch
   257	3 × 3 convolution. To achieve lightweight, the 3 × 3 convolution here uses depthwise separable
   258	convolution, namely Lite Conv in the figure. By setting the number of Lite Conv layers, the receptive
   259	field of each branch can be controlled. Finally, after processing four feature streams with different
   260	receptive fields, features focusing on different scales are obtained. In addition, the OSblock basic
   261	block is designed as a residual block, and the scale index 𝑆 is set to control the number of network
   262	branches. For an input x, the residual 𝑥˜ is:
   263	𝑥˜ =
   264	
   265	𝑆
   266	Õ
   267	
   268	𝐹 𝑠 (𝑥) ,
   269	
   270	(1)
   271	
   272	𝑠
   273	
   274	where 𝐹 𝑠 represents the convolution of the stacked 𝑠 Lite Conv.
   275	To enhance the representation of multi-scale characteristics, a dynamic feature aggregation
   276	mechanism is implemented through an aggregation gate (AG) module. Although squeeze-andexcitation networks (SE-Net) employs channel-wise attention to enhance feature representations,
   277	SE-Net concentrates on refining channels within a single branch while the AG focuses on adaptively
   278	combining global and local features. This component, consisting of a multi-layer perceptron and a
   279	global average pooling operation, adaptively assigns importance weights to features across varying
   280	scales. The integration of AG into the network architecture modifies the residual computation, as
   281	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   282	
   283	Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID
   284	
   285	171:7
   286	
   287	defined by the equation:
   288	𝑥˜ =
   289	
   290	𝑆
   291	Õ
   292	
   293	𝐴 (𝐹 𝑠 (𝑥))
   294	
   295	𝐹 𝑠 (𝑥) ,
   296	
   297	(2)
   298	
   299	𝑠
   300	
   301	where 𝐴 corresponds to the AG and indicates element-wise multiplication. The AG generates a
   302	weighting vector that is used to perform channel weighting for features at different scales.
   303	After extracting a pedestrian image through the backbone network of partial branches, feature
   304	𝐹𝑃 is obtained, and then feature 𝐹𝑊 is obtained through the whole-level branch, with a dimension
   305	of (𝐶, 𝐻,𝑊 ). For the partial branch, we divide pedestrian features into 𝑁 blocks from top to
   306	bottom. We adopt vertical partitioning from top to bottom rather than horizontal splitting because
   307	pedestrian images in Re-ID datasets are usually vertically aligned, and most discriminative cues
   308	such as hat, clothing, bag, and shoes appear along the vertical axis. Vertical slicing ensures that each
   309	sub-region corresponds to a semantically meaningful body part including head, trunk, and legs.
   310	This facilitates the learning of localized representations consistent with human body structure. In
   311	contrast, horizontal division disrupts spatial continuity and mixes unrelated regions. The features
   312	obtained from the ith part are as follows:
   313	𝑓𝑖 = (0 : 𝐶, 𝑖 − 1 : 𝐻𝑖/𝑛, 0 : 𝑊 ).
   314	
   315	(3)
   316	
   317	Average pooling is performed on each obtained part feature to obtain a feature vector 𝐴𝑃 (𝑓𝑖 ).
   318	Compared to predicting each part vector separately, merging all part vectors will result in a more
   319	comprehensive feature description. Although the final integrated feature vector combines all partial
   320	representations, dividing the pedestrian feature map into multiple local regions before integration is
   321	crucial. This division allows each local branch to focus on fine-grained cues such as clothing textures,
   322	carried objects, or local pose variations that might be diluted in a single global representation. By
   323	first learning discriminative local embeddings and then concatenating them, the model preserves
   324	detailed spatial information while still benefiting from the complementary context of the global
   325	feature. The comprehensive part feature vector obtained by concatenation is as follows:
   326	𝑓𝑃 = [𝐴𝑃 (𝑓1 )𝑇 , 𝐴𝑃 (𝑓2 )𝑇 , ..., 𝐴𝑃 (𝑓𝑛 )𝑇 ]𝑇 .
   327	
   328	(4)
   329	
   330	Finally, the part-level features and whole-level features are joined together to form the final
   331	feature embedding as shown below:
   332	𝐹 = [𝑓𝑃𝑇 , 𝑓𝑊𝑇 ]𝑇 .
   333	3.2
   334	
   335	(5)
   336	
   337	Meta-Learning-Based Multi-Source Domain Learning Framework
   338	
   339	The multi-source domain learning framework used in this study is shown in Figure 1(b). The learning
   340	process is as follows: in each training round, each source domain is set as a task, and Gaussian
   341	noise perturbation is used to enhance the style of each task, which is called style uncertainty
   342	augment (SuA). According to the degree of perturbation of style enhancement, the tasks are
   343	divided into simple tasks and complex tasks. By simulating the learning process of humans from
   344	easy to difficult, the model will first learn on simple tasks, and then use the knowledge learned on
   345	simple tasks to learn on more difficult new tasks until the most difficult tasks are learned.
   346	Compared with image enhancement methods such as generative adversarial network, SuA
   347	enhances features with lower computational and training overhead. Specifically, during training,
   348	SuA achieves style enhancement by introducing Gaussian noise to perturb the mean and variance
   349	across feature space dimensions. For a given feature 𝑥 ∈ 𝑅 𝐵×𝐶 ×𝐻 ×𝑊 , SuA begins by normalizing
   350	the tensor and computing its mean 𝜇 (𝑥) and standard deviation 𝜎 (𝑥) for each channel according
   351	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   352	
   353	171:8
   354	
   355	X. Zhang et al.
   356	
   357	to the following formula:
   358	𝐻
   359	
   360	𝜇 (𝑥) =
   361	
   362	𝑊
   363	
   364	1 ÕÕ
   365	𝑥𝑖 𝑗 ,
   366	𝐻𝑊 𝑖=1 𝑗=1
   367	
   368	(6)
   369	
   370	and
   371	v
   372	u
   373	t
   374	𝜎 (𝑥) =
   375	
   376	𝐻
   377	
   378	𝑊
   379	
   380	1 ÕÕ
   381	(𝑥𝑖 𝑗 − 𝜇 (𝑥)) 2 .
   382	𝐻𝑊 𝑖=1 𝑗=1
   383	
   384	(7)
   385	
   386	Then, the mean 𝜇 (𝑥) and standard deviation 𝜎 (𝑥) are perturbed with Gaussian noise to randomize
   387	its style, as follows:
   388	𝑆𝑢𝐴(𝑥) = 𝜎𝑆 (𝑥)(𝑥 − 𝜇 (𝑥))/𝜎 (𝑥) + 𝜇𝑆 (𝑥),
   389	
   390	(8)
   391	
   392	Among them, the mean and variance are perturbed by sampling from a normal distribution with a
   393	mean of 0 and a variance of 𝑚1 and 𝑚2 as follows:
   394	𝜇𝑆 (𝑥) = 𝜇 (𝑥) + 𝛼, 𝛼 ∼ 𝑁 (0, 𝑚1),
   395	
   396	(9)
   397	
   398	𝜎𝑆 (𝑥) = 𝜎 (𝑥) + 𝛽, 𝛽 ∼ 𝑁 (0, 𝑚2),
   399	
   400	(10)
   401	
   402	where 𝛼, 𝛽 ∈ 𝑅𝐶 . For the network WPOSNet proposed in this article, we add the style enhancement
   403	module after conv2 and conv3 of the OSNet backbone network. Regarding the setting of the standard
   404	deviation m1 and m2, the larger its value is, the stronger the style interference to the feature is.
   405	By setting different standard deviations, source tasks of different difficulty levels can be obtained.
   406	Note that the style enhancement module is only used during the training phase, it is not applied in
   407	the network during the inference phase.
   408	For the multi-source domain training stage, unlike the traditional meta-learning mode, the
   409	SpML method assigns tasks from easy to difficult for the model to learn, which can make the
   410	model training more stable and enhance the model’s generalization ability on novel target domains.
   411	First, set the initial task as the meta-training task, which does not perform style enhancement, set
   412	the initial model parameters to Θ, and then calculate the meta-training loss 𝐿𝑚𝑡𝑟 of the task. The
   413	formula is as follows:
   414	𝐿𝑚𝑡𝑟 = 𝐿𝑚 (Θ, 𝑥) + 𝐿𝑡𝑟𝑖 (Θ, 𝑥).
   415	
   416	(11)
   417	
   418	Then we use the meta-training loss of the first task for optimization and get the model Θ1 of the
   419	second task. On the second task, we use the new source domain data and introduce simple style
   420	enhancement and calculate the loss 𝐿𝑚𝑡𝑒1 of the second task. The relevant formula is as follows:
   421	Θ1 = Θ − ∇𝐿𝑚𝑡𝑟 (Θ),
   422	
   423	(12)
   424	
   425	𝐿𝑚𝑡𝑒1 = 𝐿𝑚 (𝑥, Θ1 ) + 𝐿𝑡𝑟𝑖 (𝑥, Θ1 ) + 𝐿𝑑𝑔 (𝑥, Θ1 ).
   426	
   427	(13)
   428	
   429	Next, we use the losses of the first two tasks to optimize together and obtain the model Θ2 for
   430	the third task. The formula is as follows:
   431	𝑎𝑟𝑔min𝐿𝑚𝑡𝑟 (Θ) + 𝐿𝑚𝑡𝑒1 (Θ1 ).
   432	Θ
   433	
   434	(14)
   435	
   436	At this time, using new source domain data and setting a more difficult style diversity enhancement, the loss 𝐿𝑚𝑡𝑒1 of the model can be obtained by calculation, and its calculation method is
   437	similar to formula 13. Similarly, for meta-learning with 𝑛 tasks, the model will use the losses on
   438	these 𝑛 tasks to jointly optimize, and the formula is as follows:
   439	𝑎𝑟𝑔min𝐿𝑚𝑡𝑟 (Θ) + 𝐿𝑚𝑡𝑒1 (Θ1 ) + ... + 𝐿𝑚𝑡𝑒𝑛−1 (Θ𝑛−1 ).
   440	Θ
   441	
   442	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   443	
   444	(15)
   445	
   446	
codex
第五篇 Efficient Lightweight 的方法在第 8 页后还没结束，后面还有 SpML 的损失和目标域 MMT 适配。我会补读后续方法页。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf' - | nl -ba | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	171:8
     2	
     3	X. Zhang et al.
     4	
     5	to the following formula:
     6	𝐻
     7	
     8	𝜇 (𝑥) =
     9	
    10	𝑊
    11	
    12	1 ÕÕ
    13	𝑥𝑖 𝑗 ,
    14	𝐻𝑊 𝑖=1 𝑗=1
    15	
    16	(6)
    17	
    18	and
    19	v
    20	u
    21	t
    22	𝜎 (𝑥) =
    23	
    24	𝐻
    25	
    26	𝑊
    27	
    28	1 ÕÕ
    29	(𝑥𝑖 𝑗 − 𝜇 (𝑥)) 2 .
    30	𝐻𝑊 𝑖=1 𝑗=1
    31	
    32	(7)
    33	
    34	Then, the mean 𝜇 (𝑥) and standard deviation 𝜎 (𝑥) are perturbed with Gaussian noise to randomize
    35	its style, as follows:
    36	𝑆𝑢𝐴(𝑥) = 𝜎𝑆 (𝑥)(𝑥 − 𝜇 (𝑥))/𝜎 (𝑥) + 𝜇𝑆 (𝑥),
    37	
    38	(8)
    39	
    40	Among them, the mean and variance are perturbed by sampling from a normal distribution with a
    41	mean of 0 and a variance of 𝑚1 and 𝑚2 as follows:
    42	𝜇𝑆 (𝑥) = 𝜇 (𝑥) + 𝛼, 𝛼 ∼ 𝑁 (0, 𝑚1),
    43	
    44	(9)
    45	
    46	𝜎𝑆 (𝑥) = 𝜎 (𝑥) + 𝛽, 𝛽 ∼ 𝑁 (0, 𝑚2),
    47	
    48	(10)
    49	
    50	where 𝛼, 𝛽 ∈ 𝑅𝐶 . For the network WPOSNet proposed in this article, we add the style enhancement
    51	module after conv2 and conv3 of the OSNet backbone network. Regarding the setting of the standard
    52	deviation m1 and m2, the larger its value is, the stronger the style interference to the feature is.
    53	By setting different standard deviations, source tasks of different difficulty levels can be obtained.
    54	Note that the style enhancement module is only used during the training phase, it is not applied in
    55	the network during the inference phase.
    56	For the multi-source domain training stage, unlike the traditional meta-learning mode, the
    57	SpML method assigns tasks from easy to difficult for the model to learn, which can make the
    58	model training more stable and enhance the model’s generalization ability on novel target domains.
    59	First, set the initial task as the meta-training task, which does not perform style enhancement, set
    60	the initial model parameters to Θ, and then calculate the meta-training loss 𝐿𝑚𝑡𝑟 of the task. The
    61	formula is as follows:
    62	𝐿𝑚𝑡𝑟 = 𝐿𝑚 (Θ, 𝑥) + 𝐿𝑡𝑟𝑖 (Θ, 𝑥).
    63	
    64	(11)
    65	
    66	Then we use the meta-training loss of the first task for optimization and get the model Θ1 of the
    67	second task. On the second task, we use the new source domain data and introduce simple style
    68	enhancement and calculate the loss 𝐿𝑚𝑡𝑒1 of the second task. The relevant formula is as follows:
    69	Θ1 = Θ − ∇𝐿𝑚𝑡𝑟 (Θ),
    70	
    71	(12)
    72	
    73	𝐿𝑚𝑡𝑒1 = 𝐿𝑚 (𝑥, Θ1 ) + 𝐿𝑡𝑟𝑖 (𝑥, Θ1 ) + 𝐿𝑑𝑔 (𝑥, Θ1 ).
    74	
    75	(13)
    76	
    77	Next, we use the losses of the first two tasks to optimize together and obtain the model Θ2 for
    78	the third task. The formula is as follows:
    79	𝑎𝑟𝑔min𝐿𝑚𝑡𝑟 (Θ) + 𝐿𝑚𝑡𝑒1 (Θ1 ).
    80	Θ
    81	
    82	(14)
    83	
    84	At this time, using new source domain data and setting a more difficult style diversity enhancement, the loss 𝐿𝑚𝑡𝑒1 of the model can be obtained by calculation, and its calculation method is
    85	similar to formula 13. Similarly, for meta-learning with 𝑛 tasks, the model will use the losses on
    86	these 𝑛 tasks to jointly optimize, and the formula is as follows:
    87	𝑎𝑟𝑔min𝐿𝑚𝑡𝑟 (Θ) + 𝐿𝑚𝑡𝑒1 (Θ1 ) + ... + 𝐿𝑚𝑡𝑒𝑛−1 (Θ𝑛−1 ).
    88	Θ
    89	
    90	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
    91	
    92	(15)
    93	
    94	Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID
    95	
    96	171:9
    97	
    98	By learning tasks from easy to difficult, the model can more stably learn domain-independent
    99	knowledge and improve its generalization ability.
   100	For the design of the loss function, memory-based loss, triplet loss, and distance-graph alignment
   101	loss are considered. Memory-based loss is implemented by maintaining a feature center for each
   102	pedestrian category. For the feature 𝑓 𝑗 from the 𝑗th pedestrian, the loss is obtained by calculating
   103	the similarity between 𝑓 𝑗 and the feature centers of other pedestrians. The specific formula is as
   104	follows:
   105	𝑒𝑥𝑝 (𝑀𝑒𝑚[ 𝑗] 𝑓 𝑗 /𝛾)
   106	𝐿𝑚 = −𝑙𝑜𝑔 Í𝑁
   107	,
   108	(16)
   109	𝑖=1𝑒𝑥𝑝 (𝑀𝑒𝑚[𝑖] 𝑓 𝑗 /𝛾)
   110	where 𝑀𝑒𝑚 is the memory used to store identity features center, 𝑁 is the number of identities in
   111	the current dataset, and 𝛾 is the temperature factor. To ensure that the memory maintains both
   112	stability and adaptability, we adopt a momentum-based update scheme formulated as follows:
   113	𝑀𝑒𝑚[ 𝑗] = 𝜓 𝑀𝑒𝑚[ 𝑗] + (1 − 𝜓 ) 𝑓 𝑗 ,
   114	
   115	(17)
   116	
   117	where 𝜓 ∈ [0, 1] is the momentum coefficient controlling the update speed. A larger 𝜓 retains more
   118	historical information, while a smaller 𝜓 allows faster adaptation to the new feature distribution.
   119	Secondly, triple loss is also used to make similar samples close together and different samples far
   120	apart. Its formula can be expressed as:
   121	𝐿𝑡𝑟𝑖 = 𝑚𝑎𝑥 (𝑑𝑖𝑠𝑝 − 𝑑𝑖𝑠𝑛 + 𝜉, 0),
   122	
   123	(18)
   124	
   125	where 𝑑𝑖𝑠𝑝 is the positive sample distance, 𝑑𝑖𝑠𝑛 is the negative sample distance, and 𝜉 is an adjustable
   126	threshold that controls the distance between the two.
   127	Finally, the distance map alignment loss is used to ensure that the data can retain its original
   128	semantic content after Sua enhancement. The relevant formula is as follows:
   129	Í𝑁 Í𝑁
   130	0 0
   131	𝑖=1 𝑗=1 𝑑𝑖𝑠 (𝑓𝑖 , 𝑓 𝑗 ) − 𝑑𝑖𝑠 (𝑓𝑖 , 𝑓 𝑗 )
   132	2
   133	𝐿𝑑𝑔 =
   134	,
   135	(19)
   136	𝑁 ×𝑁
   137	where 𝑁 represents the number of batchsize, 𝑓 is the feature of the image, 𝑓 0 is its enhanced feature,
   138	and k·k is the operation of the L2 norm.
   139	3.3
   140	
   141	Pseudo-Label Learning Based on MMT
   142	
   143	The MMT framework is used in the pseudo-label learning stage. The framework contains four
   144	networks, two of which are 𝑁 𝑒𝑡1 and 𝑁 𝑒𝑡2 in the figure. They are updated by back-propagation of
   145	the loss function, and hard labels are updated by clustering before each training epoch. The other two
   146	networks, 𝑀𝑒𝑎𝑛1 and 𝑀𝑒𝑎𝑛2, are weighted by the time average parameters of the corresponding
   147	networks. The soft pseudo-labels are obtained by calculating the classification prediction values
   148	of the average model 𝑀𝑒𝑎𝑛1/2. The confidence of the soft pseudo-labels is less than 1, and they
   149	are optimized online as the network is updated. The soft pseudo-labels generated by 𝑀𝑒𝑎𝑛1/2 of
   150	𝑁 𝑒𝑡1/2 are used to supervise 𝑁 𝑒𝑡2/1, and the effect of mutual supervision is achieved under the
   151	collaborative training of hard pseudo-labels and soft pseudo-labels, thereby avoiding overfitting of
   152	the output error of the network itself. Its overall loss is composed of the classification loss 𝐿𝑖𝑑 and
   153	triplet loss 𝐿𝑡𝑟𝑖 of the hard pseudo-labels of the two networks and the classification loss 𝐿𝑠𝑖𝑑 and
   154	triplet loss 𝐿𝑠𝑡𝑟𝑖 of the soft pseudo-labels. The formula is as follows:
   155	𝐿 = (𝛼𝐿𝑖𝑑 (𝜃 1, 𝜃 2 ) + (1 − 𝛼)𝐿𝑠𝑖𝑑 (𝜃 1, 𝜃 2 )
   156	+ 𝛽𝐿𝑡𝑟𝑖 (𝜃 1, 𝜃 2 ) + (1 − 𝛽)𝐿𝑠𝑡𝑟𝑖 (𝜃 1, 𝜃 2 )),
   157	
   158	(20)
   159	
   160	where 𝜃 1 is the network parameter of 𝑁 𝑒𝑡1, 𝜃 2 is the network parameter of 𝑁 𝑒𝑡2, and 𝛼 and 𝛽 are
   161	weighted parameters.
   162	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   163	
   164	171:10
   165	
   166	X. Zhang et al.
   167	
   168	The soft cross-entropy loss function is employed to determine the soft classification loss, and the
   169	formula is as follows:
   170	𝑁 
   171	Õ
   172	𝐿𝑠𝑖𝑑 (𝜃 1, 𝜃 2 ) =
   173	cls(𝑥𝑖 |𝜃 𝑇2 𝐴 ) · log cls(𝑥𝑖 |𝜃 1 )
   174	𝑖=1
   175	
   176	
   177	+ cls(𝑥𝑖 |𝜃 𝑇1 𝐴 ) · log cls(𝑥𝑖 |𝜃 2 ) ,
   178	
   179	(21)
   180	
   181	where 𝑁 is the number of images in the target domain, 𝜃 𝑇1 𝐴 is the network parameters of the average
   182	network 𝑀𝑒𝑎𝑛1 of 𝑁 𝑒𝑡1, 𝜃 𝑇2 𝐴 is the network parameters of the average network 𝑀𝑒𝑎𝑛2 of 𝑁 𝑒𝑡2, and
   183	𝑐𝑙𝑠 (·) is the classifier of the corresponding network. This formula aims to make the classification prediction
   184	value
   185	of 𝑁 𝑒𝑡1 close to the classification prediction value of 𝑀𝑒𝑎𝑛2, and to make the classification prediction value of 𝑁 𝑒𝑡2 close to the classification prediction value of 𝑀𝑒𝑎𝑛1.
   186	For the soft triplet loss, the softmax-triplet calculated by the image features encoded by the
   187	average model is used as a soft pseudo-label to supervise the training of the triplet. The function
   188	aims to make the softmax-triplet output by 𝑁 𝑒𝑡1 close to the softmax-triplet prediction value of
   189	𝑀𝑒𝑎𝑛2, and the softmax-triplet output by 𝑁 𝑒𝑡2 close to the softmax-triplet prediction value of
   190	𝑀𝑒𝑎𝑛1. The formula is as follows:
   191	𝑁 
   192	Õ
   193	𝐿𝑠𝑡𝑟𝑖 (𝜃 1, 𝜃 2 ) =
   194	𝐿𝑏 (Γ(𝑥𝑖 |𝜃 1 ), Γ(𝑥𝑖 |𝜃 𝑇2 𝐴 ))
   195	𝑖=1
   196	
   197	+ 𝐿𝑏 (Γ(𝑥𝑖 |𝜃 2 ), Γ(𝑥𝑖 |𝜃 𝑇1 𝐴 ))
   198	
   199	
   200	,
   201	
   202	(22)
   203	
   204	where 𝐿𝑏 represents the binary cross-entropy loss, Γ represents the softmax triplet, and its formula
   205	is as follows:
   206	𝑒𝑥𝑝 (𝑑𝑖𝑠 (𝑓𝑥 , 𝑓𝑥𝑛 ))
   207	Γ(𝑥) =
   208	,
   209	(23)
   210	𝑒𝑥𝑝 (𝑑𝑖𝑠 (𝑓𝑥 , 𝑓𝑥𝑝 )) + 𝑒𝑥𝑝 (𝑑𝑖𝑠 (𝑓𝑥 , 𝑓𝑥𝑛 ))
   211	where 𝑓𝑥 represents the feature of the input image 𝑥 calculated by the network, and 𝑑𝑖𝑠 () represents
   212	the L2 norm distance between two samples. The formula is as follows:
   213	𝑑𝑖𝑠 (𝑥, 𝑦) = k𝑥 − 𝑦 k 2 .
   214	
   215	(24)
   216	
   217	Finally, by jointly training the offline-optimized hard pseudo-labels and the online-optimized soft
   218	pseudo-labels, the model obtained by multi-source domain learning based on WPOSNet performed
   219	effective domain adaptation in the target domain.
   220	3.4
   221	
   222	Experiment
   223	
   224	This section will introduce the data required for the experiment, hyperparameter settings, comparative experiments with other methods, ablation tests to validate the impact of individual algorithm
   225	components, and visual analysis of the results.
   226	3.5
   227	
   228	Dataset
   229	
   230	This experiment uses four public person Re-ID datasets and one private dataset from real Re-ID
   231	application scenarios to verify the proposed algorithm. Among them, the private dataset Railway
   232	collects and organizes eight surveillance cameras at Shenyang Railway Station and Shenyangbei
   233	Railway Station. The camera distribution is shown in Figure 3. The left side of the picture shows
   234	the distribution of Shenyangbei Railway Station, covering the area from the security check to the
   235	waiting hall. The right side of the picture shows the distribution of Shenyang Railway Station,
   236	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   237	
   238	Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID
   239	
   240	171:11
   241	
   242	Fig. 3. Schematic diagram of the distribution of train station dataset. The left half of the picture is located
   243	at Shenyangbei Railway Station, and the right half is located at Shenyang Railway Station. The area is the
   244	inbound security check and waiting hall. The arrows indicate the direction of movement of pedestrians.
   245	Table 1. Details of the Dataset Used in the Experiment
   246	
   247	Name
   248	
   249	Training Set
   250	
   251	Test Set
   252	
   253	Images
   254	
   255	IDs
   256	
   257	Images
   258	
   259	IDs
   260	
   261	Market1501
   262	
   263	12,936
   264	
   265	751
   266	
   267	19,732
   268	
   269	759
   270	
   271	DukeMTMC
   272	
   273	6,522
   274	
   275	702
   276	
   277	19,889
   278	
   279	1,110
   280	
   281	CUHK03
   282	
   283	7,365
   284	
   285	767
   286	
   287	6,732
   288	
   289	700
   290	
   291	MSMT17V1
   292	
   293	32,621
   294	
   295	1,041
   296	
   297	93,820
   298	
   299	3,060
   300	
   301	Railway
   302	
   303	12,394
   304	
   305	982
   306	
   307	12,647
   308	
   309	813
   310	
   311	covering the area before and after the security check. The four public datasets are Market1501,
   312	CUHK03, DukeMTMC, and MSMT17. The details of the datasets used in the experiments are shown
   313	in Table 1.
   314	3.6
   315	
   316	Implementation Details
   317	
   318	For the multi-source domain learning stage, we set the input image size to 256 × 128. During
   319	training, a batch size of 32 is used, comprising 8 identities with 4 images per identity, and the
   320	triplet loss 𝜉 is set to 0.3. For SuA-based domain augmentation, this article follows the same settings
   321	as those in reference [30]. Specifically, the SuA augmentation is divided into three stages with
   322	progressively increasing difficulty. In the initial stage, the Gaussian noise intensities m1 and m2 are
   323	both set to 0.1, and then each of them is increased by 0.05 in every subsequent stage. In addition,
   324	the Adam optimizer is employed with a weight decay of 0.0005, and the learning rate is initialized
   325	at 3.5 × 10−5 for the first 30 epochs, followed by decays of 0.1 at the 30th and 40th epochs. The
   326	training process spans 50 epochs in total. For the domain adaptation stage, we set the batch size
   327	to 64, and the entire training process requires 50 epochs. Other settings refer to the same settings
   328	as [6].
   329	For the experimental evaluation, mean Average Precision (mAP), Rank 1/5/10, and lightweight
   330	model metrics, including model parameters and Floating Point Operations (FLOPs), were
   331	employed.
   332	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   333	
   334	171:12
   335	
   336	X. Zhang et al.
   337	Table 2. Comparison with Other Domain Adaptation Methods
   338	
   339	Method
   340	
   341	3.7
   342	
   343	D+C+MS → M
   344	
   345	M+C+MS → D
   346	
   347	D+C+M → MS
   348	
   349	mAP
   350	
   351	R1
   352	
   353	R5
   354	
   355	mAP
   356	
   357	R1
   358	
   359	R5
   360	
   361	mAP
   362	
   363	R1
   364	
   365	R5
   366	
   367	BORT [4]
   368	
   369	79.3%
   370	
   371	92.1%
   372	
   373	95.9%
   374	
   375	65.1%
   376	
   377	79.5%
   378	
   379	88.9%
   380	
   381	24.3%
   382	
   383	50.6%
   384	
   385	62.5%
   386	
   387	DaDiL-E [19]
   388	
   389	82.6%
   390	
   391	92.8%
   392	
   393	97.5%
   394	
   395	64.9%
   396	
   397	79.0%
   398	
   399	88.8%
   400	
   401	30.1%
   402	
   403	58.4%
   404	
   405	65.0%
   406	
   407	VDD [25]
   408	
   409	85.7%
   410	
   411	94.5%
   412	
   413	97.0%
   414	
   415	68.3%
   416	
   417	81.6%
   418	
   419	89.9%
   420	
   421	33.4%
   422	
   423	62.5%
   424	
   425	73.9%
   426	
   427	MMT+GRL [5]
   428	
   429	77.1%
   430	
   431	90.4%
   432	
   433	96.8%
   434	
   435	64.3%
   436	
   437	77.6%
   438	
   439	88.1%
   440	
   441	22.6%
   442	
   443	46.3%
   444	
   445	59.8%
   446	
   447	MMT+DSBN [2]
   448	
   449	81.1%
   450	
   451	92.8%
   452	
   453	97.3%
   454	
   455	65.6%
   456	
   457	79.6%
   458	
   459	89.1%
   460	
   461	22.6%
   462	
   463	49.2%
   464	
   465	62.1%
   466	
   467	RDSBN [1]
   468	
   469	86.0%
   470	
   471	94.8%
   472	
   473	97.9%
   474	
   475	68.9%
   476	
   477	82.1%
   478	
   479	90.4%
   480	
   481	34.9%
   482	
   483	64.7%
   484	
   485	76.2%
   486	
   487	FFCCLS [24]
   488	
   489	86.2%
   490	
   491	95.1%
   492	
   493	97.9%
   494	
   495	69.6%
   496	
   497	83.5%
   498	
   499	91.3%
   500	
   501	35.8%
   502	
   503	65.3%
   504	
   505	77.3%
   506	
   507	Ours
   508	
   509	85.1%
   510	
   511	94.3%
   512	
   513	98.1%
   514	
   515	72.7%
   516	
   517	83.3%
   518	
   519	91.4%
   520	

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf' - | nl -ba | sed -n '66,829p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    66	I. I NTRODUCTION
    67	
    68	P
    69	
    70	ERSON Re-Identification (ReID) aims to retrieve images
    71	of the same individuals from different cameras. With
    72	its wide-ranging applications in public security and video
    73	surveillance, ReID has sparked significant interest and notable
    74	advancements in the field. Many existing methods [1], [2], [3],
    75	[4] primarily focus on re-identification in daytime scenarios,
    76	overlooking low-light conditions. However, treating ReID as
    77	a single-modality problem is unreasonable, as this inevitably
    78	causes existing methods to underperform in low-light environments. To overcome this limitation, the adoption of infrared
    79	camera technology for continuous, all-conditions surveillance
    80	has given rise to the Visible-Infrared Person Re-Identification
    81	(VI ReID) task, enhancing the system’s robustness and effectiveness across diverse scenarios.
    82	Different from the ReID task merely built upon the visible illumination, the query and gallery sets in VI ReID
    83	are captured by cameras with distinct modalities, resulting
    84	in significant modality gaps among the compared person
    85	images. Existing methods typically focus on reducing modality
    86	discrepancies by eliminating information gaps. For example,
    87	some modality supplementing methods [5], [6], [7], [8] utilize Generative Adversarial Networks (GAN) [9] to generate
    88	specific information from another modality for supplementing
    89	inputs, achieving effective cross-modality matching. However, challenges arise because the brightness of infrared
    90	images may not perfectly correspond to the color of visible
    91	images [10], and the high computational cost of GANs also
    92	makes stable modality transfer difficult [11]. Moreover, some
    93	methods involve extracting discriminative modality-invariant
    94	features and focusing on the commonality of visual features
    95	between different modalities, to address modality discrepancies. For instance, SPOT [12] employs multi-level alignment
    96	mechanisms and leverages physics knowledge, such as body
    97	structure, to learn discriminative cross-modality invariant features.
    98	However, these existing methods may overly focus on
    99	bridging the information gap. In fact, we contend that merely
   100	
   101	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
   102	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   103	See https://www.ieee.org/publications/rights/index.html for more information.
   104	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   105	
   106	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
   107	
   108	eliminating the information gap does not fully address all the
   109	challenges of the VI ReID task, such as capturing discriminative pedestrian clues for identification. Moreover, completely
   110	bridging the information gap between diverse modalities could
   111	be a very challenging issue, potentially even more than crossmodality re-identification itself. Here, let us consider the
   112	fundamental challenge of the VI ReID task, which we believe
   113	involves training a model to consistently extract discriminative
   114	information for ReID from various instances within different
   115	modalities. Based on this, in this work, we propose a more
   116	straightforward and effective strategy that directly enhances
   117	the model’s adaptability, enabling it to flexibly adapt to various
   118	instances and diverse modalities. The core insight of our
   119	work is that when a model has sufficient flexibility, it can
   120	dynamically adapt to the characteristics of different modalities
   121	and consistently extract significant information from diverse
   122	modality inputs for identification. This enables our model
   123	to effectively mitigate the impact of modality divergence.
   124	Besides, this flexibility also allows our model to adapt to
   125	different instance inputs and adaptively recognize their discriminative clues.
   126	Based on this idea, in this paper, we propose a novel and
   127	effective Modality-aware and Instance-aware Visual Prompts
   128	(MIP) network to address the VI ReID task. Our key innovation is to endow the model with sufficient flexibility, enabling
   129	it to adapt to various modalities and instances. In this step, we
   130	notice that visual prompts could be a good tool to accomplish
   131	this. Recently, Visual Prompt Tuning (VPT) [13] and its
   132	extensive use in numerous existing works [13], [14], [15],
   133	[16], [17], [18] showcases its ability to adapt the origin
   134	models efficiently to various target tasks. Inspired by this,
   135	in this work, we customize two types of visual prompts
   136	namely, modality-aware prompts and instance-aware prompts
   137	to adapt our model. The modality-aware prompts are designed
   138	to learn and equip our model with the characteristics of
   139	the current inputs. This enables the model to dynamically
   140	adapt to diverse modalities, thereby alleviating the interference
   141	caused by modality divergence. Regarding the instance-aware
   142	prompts, they are responsible for guiding our model to adapt
   143	to the input instances, thereby enabling our model to capture instance-aware discriminative clues for identification. As
   144	shown in Fig. 1, our method focuses on model adaptability to
   145	different modalities and instances. Using modality and instance
   146	prompts, the model’s parameters and feature extraction process
   147	are adjusted to dynamically adapt to the feature distributions of
   148	input images from different modalities and instances, learning
   149	discriminative information in an adaptive feature space and
   150	overcoming more out-of-distribution samples that existing
   151	methods struggle to process, thereby improving identification
   152	performance. In contrast, existing methods typically focus
   153	on reducing the distribution gap between different modalities
   154	by mapping input images to a shared feature space, aiming
   155	to facilitate subsequent matching and recognition. However,
   156	eliminating the distribution gap is a challenging task, and
   157	certain hard samples may fail to map correctly to the shared
   158	space, impacting identification accuracy.
   159	Practically, the MIP network comprises a global backbone
   160	and three prompt learning modules: a Modality-aware Prompt
   161	
   162	8087
   163	
   164	Fig. 1. The illustrations of the modality- and instance-aware adaptation,
   165	and the difference between existing methods and our method. The circles in
   166	different colors represent inputs from different modalities. Existing methods
   167	typically focus on reducing the distribution gap between different modalities
   168	to map input images to a shared feature space, aiming to facilitate subsequent
   169	matching and recognition. However, eliminating the distribution gap is a
   170	challenging task, and certain hard samples may fail to map correctly to the
   171	shared space, impacting recognition accuracy. In contrast, our method avoids
   172	this challenge by focusing on model adaptability to different modalities and
   173	instances. Using modality and instance prompts, the model’s parameters and
   174	feature extraction process are adjusted to dynamically adapt to the feature
   175	distributions of input images from different modalities and instances, learning
   176	discriminative information in an adaptive feature space and overcoming outof-distribution samples, thereby improving identification performance.
   177	
   178	Learning (MPL) module and two Instance-aware Prompt
   179	Generator (IPG) modules, i.e., a Self-guiding IPG (SIPG)
   180	module, and a Query-guiding IPG (QIPG) module. In terms
   181	of structure, the MPL consists of two sets of learnable vectors corresponding to infrared and visible modalities. These
   182	prompt vectors are responsible for learning the characteristics
   183	of distinct modalities and guiding our model to adapt to
   184	them. Moreover, we have devised two innovative instanceaware prompt generators, namely SIPG and QIPG, based on
   185	transformer architecture. In the two IPG modules, we employ
   186	a transformer layer to transfer the identity-related information
   187	from the image features into a group of learnable vectors
   188	to construct instance-aware prompts. The distinction between
   189	SIPG and QIPG lies in the source of image features: SIPG
   190	receives features from the current input instances, while QIPG
   191	receives features from the query instances to be matched. The
   192	prompts generated by two IPG modules are supplied to the
   193	backbone model to guide it in dynamically adapting to the
   194	input instances and the query instances, respectively, thereby
   195	capturing discriminative clues for identification. Additionally,
   196	we have designed a Customized Prompt Fusion (CPF) module
   197	to adaptively integrate modality-aware and instance-aware
   198	prompts, so as to provide more effective guidance for our
   199	model. We also designed some auxiliary loss functions, the
   200	Instance-aware Enhancement Loss (IAEL) and the Causality
   201	Enhancement Loss (CEL) to help the modules we designed to
   202	function more effectively.
   203	We summarize the contribution of our work as follows:
   204	• We propose a novel method, the Modality-aware and
   205	Instance-aware Visual Prompts (MIP) network, incorporating visual prompt learning into the VI ReID field.
   206	• We design a Modality-aware Prompt Learning (MPL)
   207	module, two Instance-aware Prompt Generators (IPG),
   208	
   209	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   210	
   211	8088
   212	
   213	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025
   214	
   215	i.e. the Self-guiding IPG (SIPG) and the Query-guiding
   216	IPG (QIPG), and the Customized Prompt Fusion (CPF)
   217	module to generate modality-aware and instance-aware
   218	prompts for the ReID model, which guide the model
   219	to adapt to the diverse modalities and instances. These
   220	enable our model to alleviate the impact of modality
   221	divergence and effectively capture discriminative instance
   222	clues for identification.
   223	• We execute extensive experiments on VI ReID benchmarks SYSU-MM01, RegDB, LLCM, and RGBN300 (for
   224	Vehicle), which validate the effectiveness of both our
   225	designed modules and demonstrate that MIP performs
   226	better than most state-of-the-art methods.
   227	Statement. This paper is an extended version of our previous work [19]. This version significantly expands upon the
   228	original work by following several key aspects.
   229	1) Theoretical Improvement:
   230	This version provides a more comprehensive theoretical
   231	analysis of how the model’s adaptability and flexibility help
   232	in VI ReID. Then this version extends the adaptation on
   233	modalities and instances to the adaptation on input modalities,
   234	input instances, and the query instances to be matched.
   235	2) Methodological Enhancements:
   236	This version includes significant improvements in the
   237	methodology, as follows:
   238	• a) Based on the model with only two modules, the
   239	Modality-aware Prompt Learning (MPL) module and
   240	the Instance-aware Prompt Generator (IPG) module, the
   241	journal version extra designs two IPG modules, namely
   242	SIPG and QIPG. The SIPG takes the place of IPG in the
   243	previous version, and the QIPG produces query-guiding
   244	instance-aware prompts to guide the model to adapt to
   245	the query instances to be matched.
   246	• b) This version introduces a novel CFP module to dynamically integrate modality-aware prompts and instanceaware prompts and discusses how to deploy the prompts
   247	better, comparing two kinds of fusion strategies to
   248	integrate modality-aware prompts and instance-aware
   249	prompts, so as to provide more effective guidance for
   250	the model.
   251	• c) This version designs a new CEL loss and introduces a
   252	cross-modality triplet loss to help the model training.
   253	3) Expanded Experimental Validation:
   254	This version reports more extensive new experimental
   255	results, validating the effectiveness of our proposed methods,
   256	including: comparisons with more existing methods (including
   257	some CLIP-based methods); comparisons with existing methods on two other mainstream datasets, namely LLCM and
   258	RGBN300 (vehicle); ablations about the trade-off parameters
   259	in hybrid objection functions; ablations about the new proposed modules; ablation about the length of prompts; ablation
   260	about the QIPG module and its CEL loss constraint; discussion
   261	on the parameters and computational complexity.
   262	
   263	II. R ELATED W ORK
   264	A. Person Re-Identification
   265	Person Re-identification (ReID) is a crucial task that
   266	involves matching query images of individuals with corresponding target images from a gallery set. Its importance
   267	in real-world applications has garnered significant attention
   268	and driven the development of various methodologies [1],
   269	[3], [20], [21], [22], [23], [24], [25], [26], [27], [28]. Similar
   270	tasks include vehicle ReID [29], [30], animal ReID [31],
   271	etc.
   272	ReID methods typically comprise two key components:
   273	feature representation learning and deep metric learning.
   274	Global-based approaches like VLAD [20], BNNeck [21],
   275	among others, have been introduced to extract global-level
   276	feature representations for individuals’ images. Furthermore,
   277	part-based methods, such as PCB-RPP [3], leverage part-level
   278	clues to amalgamate more robust representations for retrieval
   279	purposes. Some algorithms combine both global and local features to exploit their respective advantages. For instance, Wang
   280	et al. [22] proposed a multiple granularity network with one
   281	branch for global feature representation and two branches for
   282	local feature representation. Deep metric learning techniques
   283	[23], [25], [26], such as triplet-loss [25] and quadrupleloss [26], aim to increase inter-identity feature distance and
   284	reduce intra-identity variation. While many ReID methods
   285	excel merely built upon the visible illumination, they may face
   286	challenges in low-light scenarios due to insufficient handling
   287	of the significant domain gap between visible and infrared
   288	modalities. Addressing this gap is essential for improving the
   289	versatility and applicability of ReID techniques in various
   290	environmental conditions.
   291	B. Visible-Infrared Person Re-Identification
   292	Visible-Infrared Person Re-identification (VI ReID) focuses
   293	on matching visible and infrared images of the same individuals across different cameras. The query and gallery sets
   294	are captured by cameras using different modalities. However,
   295	directly applying ReID methods designed for visible-light
   296	images to VI ReID results in poor performance due to modality
   297	discrepancies and differing distributions between the modalities [32]. Existing methods mainly focus on bridging the
   298	modality gap, and they can generally be grouped into two
   299	approaches: modality compensation and modality-invariant
   300	feature learning.
   301	Modality compensation represents a VI ReID paradigm that
   302	supplements inputs with specific information from another
   303	modality. This kind of method usually initially employs a Generative Adversarial Network (GAN) [9] to generate another
   304	modality from the available one, then compensates the original
   305	image with the generated image, to mitigate the modality
   306	discrepancy. The cmGAN [5] is the first research introducing
   307	GAN into the VI ReID task, but it utilizes GAN to better
   308	distinguish between different modalities of images rather than
   309	to generate images of missing modalities. CycleGAN [33] are
   310	typical methodologies via modality compensation, which uses
   311	GAN for style transfer of training images across different
   312	cameras to create an augmented training set, which helps
   313	
   314	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   315	
   316	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
   317	
   318	smooth camera style disparities and increase data diversity
   319	in cross-modality ReID. Wang et al. [6] further proposed
   320	to perform pixel-level and feature-level alignment to reduce
   321	modality discrepancy. Liu et al. [8] introduce a new Two-Stage
   322	Modality Enhancement Network (TSME) to address the shortcomings, such as poor quality of generated images, of existing
   323	modality compensation methods for VI ReID. TSME achieves
   324	higher-quality image generation via Deeper Skip-connection
   325	Generative Adversarial Networks and a feature-level fusion
   326	with attention mechanisms, reducing modality discrepancies
   327	and improving VI ReID performance. Zhang et al. [34]
   328	proposed a Feature-level Modality Compensation Network
   329	(FMCNet) that operates at the feature level, rather than
   330	the image level, to address modality discrepancies. FMCNet
   331	includes a single-modality feature decomposition module to
   332	separate features into modality-specific and shared types,
   333	and a feature-level compensation module to generate missing modality-specific features from shared ones. To address
   334	the challenges of traditional modality compensation methods,
   335	which are often difficult and computationally expensive, some
   336	methods [35], [36] attempt to generate auxiliary modalities to
   337	help learn more cross-modality-related information.
   338	In contrast to modality compensation methods [5], [6], [8],
   339	[33], [34], [35], [36], several methods [6], [10], [11], [12],
   340	[37], [38], [39], [40], [41], [42] attempt to extract modalityinvariant features from images of different modalities. SDL
   341	[37] proposed to disentangle spectrum information from
   342	identity discriminative features using a dual-branch network
   343	with disentanglement and identity-dispeller losses, enhancing
   344	identity-relevant features, and demonstrating superior performance compared to traditional adversarial learning methods.
   345	Chen et al. [12] proposed to model the modality invariant
   346	structural features of each modality and use the information of
   347	human body structure and part position to learn discriminative
   348	cross-modality invariant features at the part level. Liang et al.
   349	[10] innovatively proposed to design modality embeddings for
   350	the ViT [43] backbone and applied pure transformer [44] networks to VI ReID task for the first time. Existing methods [6],
   351	[12] involve computationally intensive feature alignment and
   352	fusion operations to address modality discrepancies. Zhong
   353	et al. [35] a Grayscale Enhancement Colorization Network
   354	(GECNet) to improve VI ReID performance by utilizing
   355	intermediate grayscale images and enhancing the colorization
   356	process to help bridge the modality gap between singlechannel infrared and three-channel visible images. Chai et
   357	al. [39] presents TransVI, a pure Transformer-based network
   358	with a conventional two-stream structure, that addresses crossmodality discrepancies by capturing long-range dependencies
   359	and aligning distance distributions with a Cross-Modality
   360	Dissimilarity-based Maximum Mean Discrepancy approach.
   361	Some approaches [11], [40] normalize or optimize the training
   362	process to eliminate interference from modality-specific information on shared feature learning. For example, Lu et al. [11]
   363	introduced grayscale images as auxiliary modalities and proposed a progressive learning strategy to help extract invariant
   364	features. Recently, Zhang et al. [36] presented MSALNet, a multi-stage auxiliary learning strategy that improves
   365	cross-modality consistency feature extraction and alignment
   366	
   367	8089
   368	
   369	through a two-stage training process and auxiliary modality
   370	pairs.
   371	As we have summarized, existing methods mainly focus
   372	on closing the information gap between inputs to address
   373	modality divergence. However, it is important to note that
   374	enhancing the model’s flexibility and adaptability to both
   375	modalities and instances can also be beneficial for VI ReID.
   376	C. Visual Prompt Learning
   377	Inspired by the success of textual prompts [45], [46] in NLP,
   378	visual prompt learning has been widely used in several computer vision tasks. VPT [13] designed a set of learnable vectors
   379	prepended to the input sequence of each encoder layer, thereby
   380	achieving better or equivalent results on 24 downstream recognition benchmarks. Bahng et al. [17] proposed to construct
   381	prompt in the form of perturbations, exploring the use of
   382	visual prompt guiding the large-scale language model CLIP
   383	[47] to adapt to downstream tasks. Chen et al. [18] proposed
   384	to introduce learnable parameters into the pre-trained model
   385	for adapting to other downstream tasks, but different from the
   386	VPT, the learnable parameters are added to the transformer
   387	MLP layers rather than to the input sequence. Yang et al.
   388	[14] and Zhu et al. [15] incorporated prompt learning into the
   389	multi-modality track field, showcasing its efficacy in multimodality tasks. Yu et al. [48] and Liu and Yang [49] attempted
   390	to apply prompt in UDA ReID and DG ReID, respectively.
   391	These studies have demonstrated the potential of visual prompt
   392	learning in adapting original models to other tasks, but there
   393	is still no method to introduce visual prompt learning into the
   394	VI ReID field. Wu et al. [19] firstly attempt to use prompt
   395	learning for the VI ReID task. In this work, we propose that
   396	the enhancement of the model’s adaptability and flexibility
   397	will improve the VI ReID performance, and we use specific
   398	prompts for different modality inputs and different instances
   399	to endow the model with sufficient flexibility.
   400	III. A PPROACH
   401	A. Preliminary
   402	Our model is based on the transformer-based CLIP-ReID
   403	[50] baseline equipped with commonly used VI ReID tricks,
   404	e.g. modality-balanced data sampler and data augmentations.
   405	For the text branch, we adopt the same setup with CLIP-ReID,
   406	with all the proposed modules applied solely to the image
   407	branch.
   408	In VI ReID, for an input image, we denote it as xm ∈
   409	C×H×W
   410	R
   411	, where the C, H, W denote channel dimension,
   412	height, width of the images, and m ∈ {vis, ir} denotes its
   413	modality flag. For each input image xm , we split it into a
   414	patch sequence sm ∈ Rl×C×b×b , where l denotes the length of
   415	the sequence and b denotes the size of the patch. Then, sm
   416	l×D
   417	is converted to embedding em
   418	by a linear projection
   419	0 ∈ R
   420	layer, where D denotes the embedding dimension. Thereafter, an extra learnable class token a0 will be concatenated
   421	m
   422	with the embedding em
   423	0 , and [a0 , e0 ] will be sent into the
   424	transformer encoder layers as the initial input before the
   425	first layer, where [·] indicates concatenation. Additionally, we
   426	N
   427	denote the transformer encoder layers as {Li }i=0
   428	, where N is
   429	
   430	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   431	
   432	8090
   433	
   434	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025
   435	
   436	Fig. 2. The overall framework of our proposed Modality-aware and Instance-aware Visual Prompts (MIP) network. Modality-aware prompts (produced by
   437	the MPL module), self-guiding instance-aware prompts (produced by the SIPG module), and query-guiding instance-aware are supplied into the backbone via
   438	different strategies. The ‘CE Loss’, ‘TRI Loss’, and ‘CM-TRI Loss’ is cross-entropy loss, triplet loss, and cross-modality triplet loss, respectively. The ‘CEL
   439	loss’ is our proposed Causality Enhancement Loss. © means concatenation operation; CLS denotes cls token. The ‘Query Instances’ are the images being
   440	matched, and the ‘Visual Embeddings of Query Instances’ are also obtained through the vision encoder on the left side of the figure.
   441	
   442	the number of transformer encoder layers. Subsequently, the
   443	feature extraction stage of the i-th layer can be described as
   444	the following,
   445	
   446	
   447	
   448	
   449	m
   450	(1)
   451	ai+1 , em
   452	i+1 = Li ai , ei
   453	where ai+1 and em
   454	i+1 denote the output class token and the
   455	output embedding of the i-th layer and i = 0, . . . , N − 1.
   456	B. Overall Framework
   457	In this paper, we introduce a MIP network designed for the
   458	VI ReID task. Our main objective is to adapt the model to
   459	handle varying modality and instance inputs, thereby enabling
   460	the model to effectively mitigate the impact of modality
   461	divergence and adaptively recognize the discriminative clues
   462	across different instances, thereby helping to facilitate VI
   463	ReID. We achieve these goals by employing three distinct sets
   464	of visual prompts. Specifically, we generate modality-aware
   465	prompts, self-guiding instance-aware prompts, and queryguiding instance-aware prompts based on the current modality,
   466	instance, and query. These three sets of prompts are then provided to the model. By leveraging prompts tailored to the input
   467	modalities, instances, and specific query tasks, the model can
   468	adjust its parameters to effectively adapt to varying modalities,
   469	instances, and queries, thereby enhancing the performance of
   470	VI ReID.
   471	The overall framework of our MIP is depicted in Fig. 2.
   472	As shown in Fig. 2, we utilize the pre-trained CLIP [47]
   473	images encoder for extracting features of images, and we insert
   474	visual prompts produced by our designed modules, namely the
   475	Modality-aware Prompt Learning (MPL) module (in section
   476	III-C and Fig. 3), the Self-guiding Instance-aware Prompt
   477	Generator (SIPG) module (in section III-D and Fig. 4), and
   478	the Query-guiding Instance-aware Prompt Generator (QIPG)
   479	module (in section III-D), into the backbone model. These
   480	three kinds of visual prompts are responsible for guiding
   481	the model to adapt to current modalities, input instances,
   482	and query tasks, respectively. These prompts are utilized at
   483	
   484	Fig. 3. Illustration of the Modality-aware Prompt Learning (MPL) module.
   485	The MPL module maintains a library of modality-aware prompts corresponding to each layer and each modality, essentially comprising multiple sets of
   486	learnable vectors. It selects modality-aware prompts for each layer’s visual
   487	embedding based on the input modality label, which is then processed through
   488	a prompt projection layer and supplied to the backbone.
   489	
   490	different stages, and the specific deployment strategies for
   491	these prompts and the discussion on the design of the fusion
   492	module will be detailed in section III-E.
   493	C. Modality-Aware Prompt Learning
   494	In this work, we aim to empower our model with sufficient
   495	flexibility, to enable it to dynamically adapt to the characteristics of different modalities and consistently extract significant
   496	
   497	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   498	
   499	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
   500	
   501	Fig. 4. Illustration of the Self-guiding Instance-aware Prompt Generator
   502	(SIPG) module. The SIPG module generates self-guiding instance-aware
   503	prompts from visual embedding via a vision encoder layer, and the generated
   504	prompts are supervised by ‘IAEL loss’ and supplied to the backbone. The
   505	‘IAEL Loss’ is our proposed Instance-aware Enhancement Loss.
   506	
   507	information from diverse modality inputs for identification.
   508	Based on this idea, we attempt to customize model parameters
   509	and feature extraction processes for different modalities. In
   510	this step, we notice that many studies [14], [15], [16], [17],
   511	[18], e.g., VPT [13] that achieved superior results across
   512	24 downstream vision benchmarks, have demonstrated the
   513	potential of visual prompt learning in adapting original models
   514	to other tasks without complex operations and large extra computations, so we consider exploiting this advantage of prompt
   515	learning to help the model adapt to different modality inputs.
   516	We treat the adaptation of the model to visible and infrared
   517	modalities as two different tasks, applying two independent
   518	sets of learnable visual prompts for the two modalities. To
   519	achieve this, we propose an MPL module to produce modalityaware prompts, which is shown in Fig. 3.
   520	Formally, the MPL module initializes and maintains a
   521	modality-aware prompts library with a set of learnable vectors. The modality-aware prompts library can be written as
   522	ir N
   523	P M = {pvis
   524	i , pi }i=1 , in which N is the number of transformer
   525	encoder layers; pi ∈ R j×D ; the j is the length of the modalityaware prompt, in this case, we set j = 16; pvis
   526	and piri
   527	i
   528	denote the visible-specific and infrared-specific prompts for
   529	the input before the i-th layer. Before the input embedding
   530	is sent into each encoder layer, the MPL module selects the
   531	target modality-aware prompt from the library according to the
   532	modality label of the image and the index of the layer to be
   533	input. The chosen modality-aware prompt will be concatenated
   534	with the input after being projected by a linear layer. The above
   535	processing can be written as,
   536	
   537	p̀iM = Γ P M , m, i
   538	(2)
   539	(
   540	
   541	pvis , if m = vis
   542	(3)
   543	Γ P M , m, i = iir
   544	pi , if m = ir
   545	
   546	piM = ϕ p̀iM
   547	(4)
   548	where i = 1, . . . , N; Γ(·) denotes the MPL Selection operation;
   549	ϕ(·) denotes the prompt projection linear layer.
   550	D. Instance-Aware Prompt Generator
   551	In addition to tailoring the model for different modalities,
   552	we also strive to customize our model for distinct instances
   553	by leveraging instance-aware prompts. The construction of
   554	
   555	8091
   556	
   557	instance-aware prompts is non-trivial. Here, we notice a great
   558	challenge, namely, we cannot train a set of independent
   559	prompts for each instance, because the test instances are
   560	unlimited and unpredictable. Therefore, in this work, we
   561	adopt a generation-based structure and design two dynamic
   562	prompting modules, capable of adaptively generating instanceaware prompts according to the instance, including the current
   563	input instances and the query instances to be matched.
   564	1) Self-Guiding Instance-Aware Prompt Generator: For
   565	adaptation on input instances, as shown in Fig. 4, we employ a
   566	transformer layer as the SIPG module to transfer the instanceaware knowledge from the input image features into a group
   567	of learnable vectors. The outputs of these learnable vectors
   568	then serve as instance-aware prompts to adapt our model to
   569	various input instances.
   570	Formally, given the input embedding em
   571	i before the i-th
   572	layer, we employ a transformer encoder layer to transfer the
   573	information in the embedding em
   574	i into a set of learnable vectors
   575	vi ∈ Rk×D , where i = 1, . . . , N, and k is the length of the
   576	instance-aware prompt, which is set as k = 16 in this work. The
   577	process of constructing a self-guiding instance-aware prompt
   578	can be written as follows,
   579	
   580	
   581	 S 
   582	= GS vi , em
   583	(5)
   584	pi ,
   585	i
   586	where the pSi is the output of vi , i.e., self-guiding instanceaware prompt; the GS (·) is a transformer encoder layer used as
   587	the self-guiding instance-aware generator; the is the output
   588	of em
   589	i , which is discarded.
   590	2) Query-Guiding Instance-Aware Prompt Generator: The
   591	adaptation to query instances is essential for responding to
   592	dynamically changing query conditions. The design structure
   593	of the QIPG module closely mirrors that of the SIPG shown
   594	in Fig. 4, but it focuses on integrating information related to
   595	the query, and its inputs are the visual embeddings of query
   596	instances. For this generator, the input is derived from the
   597	embedding of the current query instance rather than the current
   598	input instance. The prompts it generates incorporate additional
   599	context or cues from the query instance. This information
   600	allows the model to more precisely tailor its feature extraction
   601	process to the specifics of the query, thereby enhancing the
   602	relevance and discriminability of the extracted feature.
   603	Formally, the process can be written as follows,
   604	 Q 
   605	
   606	
   607	p ,
   608	= GQ vi , emQ
   609	(6)
   610	where the piQ is the output of vi , i.e., query-guiding instanceaware prompt; the emQ is the visual embedding of query
   611	is
   612	instance; the GQ (·) is a transformer encoder layer; the
   613	the output of emQ , which is discarded.
   614	E. Deployment Strategies for Prompts
   615	Three types of prompts are supplied into the model and
   616	utilized at different stages, as shown in Fig. 2.
   617	1) Modality-Aware Prompts and Self-Guiding InstanceAware Prompts: The modality-aware prompts and the selfguiding instance-aware prompts are used in the backbone
   618	model. These two prompts are processed through a fusion
   619	
   620	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   621	
   622	8092
   623	
   624	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025
   625	
   626	module to obtain the fused prompts, which will be concatenated with the input before each transformer encoder layer of
   627	the backbone, like the following,
   628	
   629	
   630	
   631	
   632	C
   633	= Li ai , em
   634	(7)
   635	ai+1 , em
   636	i , pi
   637	i+1 ,
   638	where pCi is the fused prompt consisted of modality-aware
   639	prompts piM and self-guiding instance-aware prompts pSi ; Li (·)
   640	means the i-th transformer encoder layer and i = 0, . . . , N − 1;
   641	the is the output of pCi , which is discarded.
   642	As for the composition of fused prompt pCi , i.e., the
   643	design of ‘Fusion Module’ in Fig. 2, we propose two kinds
   644	of strategies to fuse modality-aware prompts piM and selfguiding instance-aware prompts pSi . The first strategy is simply
   645	concatenating the two prompts in the sequence dimension,
   646	which can be formally written as follows,
   647	
   648	
   649	pCi = piM , pSi
   650	(8)
   651	To avoid potential conflicts between the two kinds of
   652	prompts, and to ensure a more refined integration that enhances
   653	their effectiveness in the backbone, we have developed a more
   654	meticulous strategy named Customized Prompt Fusion (CPF)
   655	module. The CPF module is designed to integrate modalityaware prompts and instance-aware prompts while minimizing
   656	the conflicts between the effect of adaptations toward the input
   657	instance and modality. Inspired by attention mechanisms [51],
   658	The CPF module leverages instance-dependent feature recalibration to ensure a balanced integration of modality adaptation
   659	and instance adaptation, preventing conflicts between the two
   660	types of prompts. Unlike direct concatenation, the CPF module
   661	allows the model to dynamically emphasize the instancerelated information in modality-specific prompts as needed via
   662	adaptively reweighting the modality-aware prompts based on
   663	instance-dependent information, thereby avoiding conflicts and
   664	improving feature discriminability. This design aligns with the
   665	theoretical principles of feature recalibration in SE-Net [51],
   666	where adaptive weighting enhances generalization by reducing
   667	redundancy and preserving relevant information.
   668	First, we utilize the inter-channel relationships of the current
   669	visual embedding to generate a customized attention map MiC ,
   670	then multiply the modality-aware prompt with this attention
   671	map to obtain a customized modality-aware prompt ṕiM , this
   672	prompt is added to the self-guiding instance-aware prompt and
   673	then mapped through a linear layer δ(·) to produce the fused
   674	prompt pCi . The above process can be described as:
   675	
   676	
   677	(9)
   678	pCi = C piM , pSi = δ ṕiM + pSi
   679	ṕiM = MiC ⊗ piM
   680	
   681	(10)
   682	
   683	where ⊗ denotes element-wise multiplication; the composition
   684	of customized attention map MiC can be described as
   685	
   686	
   687	MiC = σ MLP avgPool em
   688	+ MLP maxPool em
   689	i
   690	i
   691	(11)
   692	MLP (e) = W1 (ReLU (W0 (e)))
   693	(12)
   694	where σ(·) denotes the sigmoid function; ReLU(·) is the ReLU
   695	function; W1 and W0 are the weights of the MLP.
   696	Unlike the first strategy simply concatenating different
   697	prompts, the CPF module adaptively reweights the modalityaware prompt based on the instance, ensuring that modality
   698	
   699	and instance adaptation do not interfere with each other. This
   700	dynamic fusion mechanism enhances feature discriminability
   701	and improves retrieval performance.
   702	We used the CPF module in our full model, and the performance of the above two strategies is meticulously validated in
   703	the ablation study section.
   704	2) Query-Guiding Instance-Aware Prompts: The queryguiding instance-aware prompt is only used in an additional
   705	transformer encoder layer, whose input is the final feature
   706	extracted by the backbone network, like the following,
   707	
   708	
   709	 m 
   710	(13)
   711	= L̃ aN , emN , pQ
   712	ã, ẽ ,
   713	where L̃(·) is the additional transformer encoder layer; aN and
   714	emN denote the output class token and output embedding of
   715	the last layer of backbone, and ã and ẽm are the output of
   716	this additional transformer encoder layer. The output ã of the
   717	additional transformer encoder layer, and features aN extracted
   718	by the backbone, are used for inference together.
   719	F. Objective Function
   720	We employ our designed Instance-aware Enhancement Loss
   721	(IAEL), Causality Enhancement Loss (CEL), and some losses
   722	commonly used in ReID tasks for our model.
   723	1) Instance-Aware Enhancement Loss: In practice, we
   724	observe that the modality-aware and instance-aware prompts
   725	are not easy to optimize. Particularly, the instance-aware
   726	prompts may degenerate to only get a trivial solution, which
   727	means it could be invariant to different instances while failing
   728	to be customized to them. To ensure that instance-aware
   729	prompts are indeed instance-aware, we have designed the IPG
   730	module in a generation-based manner rather than fusion-based
   731	ones like previous work [52], and we additionally designed a
   732	loss function called Instance Aware Enhancement Loss (IAEL)
   733	to force the generated prompts being instance-adaptive. The
   734	IAEL loss can be written formally as follows,
   735	LIAEL = −
   736	
   737	N
   738	X
   739	
   740	  
   741	
   742	y · log ψi σi pSi /Q
   743	
   744	(14)
   745	
   746	i=1
   747	
   748	where i = 1, . . . , N means for the i-th layer; y is the groundtruth label of current input instance; ψi (·) denotes the classifier
   749	for instance-aware prompts before the i-th layer; σi (·) denotes
   750	Batch Normalization [53] layer (BN layer).
   751	2) Causality Enhancement Loss: To enhance the effectiveness of query-guiding instance-aware prompts, we designed
   752	a loss function called Causality Enhancement Loss. Inspired
   753	by metric learning and SNR [54], The primary function of
   754	this loss function is to ensure that the features extracted
   755	using query-guiding instance-aware prompts generated with
   756	query images (i.e., infrared images during training) of the
   757	same ID remain closer to the features extracted using selfguiding instance-aware prompts in feature space, compared
   758	to the features extracted using self-guiding instance-aware
   759	prompts generated with query images of the different ID.
   760	This prevents the query-guiding instance-aware prompts from
   761	learning redundant or misleading representations. The CEL
   762	loss can be written formally as follows,
   763	
   764	
   765	
   766	LCEL = max d ā s , āq,p − d ā s , āq,n + m, 0
   767	(15)
   768	
   769	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   770	
   771	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
   772	
   773	where a s , aq,p , aq,n are features extracted by self-guiding
   774	instance-aware prompts, query-guiding instance-aware
   775	prompts generated with query images of the same ID and
   776	different ID, respectively; ·¯ means a feature processed by BN
   777	layer; m is the margin.
   778	By optimizing the self-guiding instance-aware prompt generation process, the CEL loss ensures these prompts effectively
   779	capture identity-relevant and matching-useful information,
   780	improving the adaptability for matching with different query
   781	tasks.
   782	3) Cross-Modality Triplet Loss: Contrastive learning [55],
   783	[56] aims to enhance feature discrimination by pulling positive
   784	pairs closer and pushing negative pairs apart. Inspired by
   785	the idea of contrastive learning and widely used triplet loss,
   786	we employ the cross-modality triplet loss, which extends the
   787	traditional triplet loss to enhance identity discrimination across
   788	different modalities. Unlike standard triplet loss, where all
   789	samples come from the same modality, cross-modality triplet
   790	loss constructs triplets where the anchor and positive/negative
   791	samples originate from different modalities. Specifically, in our
   792	setup, infrared images serve as anchors, while the positive and
   793	negative samples are drawn from visible images. The crossmodality triplet loss can be written formally as follows,
   794	
   795	
   796	
   797	LCM−T RI = max d āir , āvis,p − d āir , āvis,n + m, 0
   798	(16)
   799	where air/vis are features of infrared/visible images; a p/n are
   800	features of images of positive/negative samples with the anchor
   801	images; ·¯ means a feature processed by BN layer; m is the
   802	margin.
   803	4) Overall Objective Function: We follow the training
   804	strategy of CLIP-ReID [50]. For the text-encoder training, we
   805	use the same objective function setting as CLIP-ReID. For the
   806	image-encoder training, we adopt a hybrid training strategy
   807	and hybrid objective function.
   808	For our MIP network with the MPL and SIPG modules, we
   809	utilize IAEL loss LIAEL and widely used cross-entropy loss
   810	LID , triplet loss LT RI , text cross-entropy loss Li2tce in CLIPReID, and MSEL loss in PMT [11] for training. The objective
   811	function can be defined as,
   812	L1 = LID + LT RI + Li2tce + 0.5 × L MS EL + α1 LIAEL
   813	
   814	(17)
   815	
   816	where α1 is the trade-off parameter, and we set α = 0.5 in
   817	practice; the IAEL loss LIAEL is only used after the 5-th epoch.
   818	For the model with the QIPG module, we utilize our IAEL
   819	loss LIAEL , CEL loss LCEL , cross-modality triplet loss LCM−T RI ,
   820	and widely used cross-entropy loss LID for training. The
   821	objective function can be defined as,
   822	L2 = LID + LCM−T RI + 0.5 × LIAEL + α2 LCEL
   823	
   824	(18)
   825	
   826	where α2 is the trade-off parameters, and we set α2 = 1.0 in
   827	practice; the IAEL loss LIAEL and CEL loss LCEL are only
   828	used after the 5-th epoch.
   829	IV. E XPERIMENT

codex
第六篇 MIP 的方法很长，当前读到了整体框架、模态提示、实例提示、提示融合和损失。前面还有一段理论分析被输出截断，我会补齐，因为它决定这篇到底是在卖“视角”还是卖“模块”。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf' - | nl -ba | sed -n '391,456p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   391	These studies have demonstrated the potential of visual prompt
   392	learning in adapting original models to other tasks, but there
   393	is still no method to introduce visual prompt learning into the
   394	VI ReID field. Wu et al. [19] firstly attempt to use prompt
   395	learning for the VI ReID task. In this work, we propose that
   396	the enhancement of the model’s adaptability and flexibility
   397	will improve the VI ReID performance, and we use specific
   398	prompts for different modality inputs and different instances
   399	to endow the model with sufficient flexibility.
   400	III. A PPROACH
   401	A. Preliminary
   402	Our model is based on the transformer-based CLIP-ReID
   403	[50] baseline equipped with commonly used VI ReID tricks,
   404	e.g. modality-balanced data sampler and data augmentations.
   405	For the text branch, we adopt the same setup with CLIP-ReID,
   406	with all the proposed modules applied solely to the image
   407	branch.
   408	In VI ReID, for an input image, we denote it as xm ∈
   409	C×H×W
   410	R
   411	, where the C, H, W denote channel dimension,
   412	height, width of the images, and m ∈ {vis, ir} denotes its
   413	modality flag. For each input image xm , we split it into a
   414	patch sequence sm ∈ Rl×C×b×b , where l denotes the length of
   415	the sequence and b denotes the size of the patch. Then, sm
   416	l×D
   417	is converted to embedding em
   418	by a linear projection
   419	0 ∈ R
   420	layer, where D denotes the embedding dimension. Thereafter, an extra learnable class token a0 will be concatenated
   421	m
   422	with the embedding em
   423	0 , and [a0 , e0 ] will be sent into the
   424	transformer encoder layers as the initial input before the
   425	first layer, where [·] indicates concatenation. Additionally, we
   426	N
   427	denote the transformer encoder layers as {Li }i=0
   428	, where N is
   429	
   430	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   431	
   432	8090
   433	
   434	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025
   435	
   436	Fig. 2. The overall framework of our proposed Modality-aware and Instance-aware Visual Prompts (MIP) network. Modality-aware prompts (produced by
   437	the MPL module), self-guiding instance-aware prompts (produced by the SIPG module), and query-guiding instance-aware are supplied into the backbone via
   438	different strategies. The ‘CE Loss’, ‘TRI Loss’, and ‘CM-TRI Loss’ is cross-entropy loss, triplet loss, and cross-modality triplet loss, respectively. The ‘CEL
   439	loss’ is our proposed Causality Enhancement Loss. © means concatenation operation; CLS denotes cls token. The ‘Query Instances’ are the images being
   440	matched, and the ‘Visual Embeddings of Query Instances’ are also obtained through the vision encoder on the left side of the figure.
   441	
   442	the number of transformer encoder layers. Subsequently, the
   443	feature extraction stage of the i-th layer can be described as
   444	the following,
   445	
   446	
   447	
   448	
   449	m
   450	(1)
   451	ai+1 , em
   452	i+1 = Li ai , ei
   453	where ai+1 and em
   454	i+1 denote the output class token and the
   455	output embedding of the i-th layer and i = 0, . . . , N − 1.
   456	B. Overall Framework

exec
/bin/zsh -lc "pdftotext -f 12 -l 15 'Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf' - | nl -ba | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
     2	
     3	8097
     4	
     5	TABLE VII
     6	T HE E FFECTS OF O UR D ESIGNED MPL AND SIPG M ODULES C OMPARED
     7	” M EANS
     8	W ITH G ENERAL P ROMPT-BASED A PPROACHES . T HE “
     9	U SING T WO S ETS OF G ENERAL P ROMPTS TO R EPLACE M ODALITYAWARE AND I NSTANCE -AWARE P ROMPTS , R ESPECTIVELY
    10	
    11	Fig. 5. Comparison of performance on our MIP method with different settings
    12	on trade-off parameter α1 , α2 in the overall objective function L1 , L2 .
    13	
    14	TABLE VI
    15	T HE C OMPARISONS OF M ODELS W ITH D IFFERENT P ROMPT L ENGTHS OF
    16	T WO K INDS OF P ROMPTS
    17	
    18	LCEL is relatively insensitive to these different values within a
    19	certain range.
    20	The above ablation experiment results demonstrate the stability and robustness of our method.
    21	3) Ablation About the Length of Prompts: We conduct
    22	ablation experiments on MIP w/o QIPG with different lengths
    23	of modality- and instance-aware prompts: 8, 16, 24, and 32.
    24	The goal is to assess how the length of the visual prompts
    25	affects model performance and whether there is an optimal
    26	length for our task.
    27	Our results show that varying the prompt length has a
    28	minimal effect on model performance. As seen in Table VI, the
    29	performance fluctuates slightly across different lengths, with
    30	some lengths yielding slightly higher or lower performance.
    31	However, the overall performance remains within a narrow
    32	range, and no significant trend is observed. This indicates
    33	that our framework is robust to variations in prompt length,
    34	suggesting that the model is capable of adapting to different
    35	prompt lengths without substantial degradation in retrieval
    36	performance.
    37	This robustness may result from the model’s ability to adapt
    38	to the prompt information, efficiently capturing discriminative
    39	features regardless of length. Therefore, we conclude that
    40	
    41	prompt length does not significantly affect overall performance, further validating the flexibility of our framework.
    42	4) Comparisons With General Prompt-Based Approaches:
    43	Several prompt-based approaches [13], [14], [15], [16], [17],
    44	[18] have demonstrated the ability of visual prompt learning in
    45	adapting original models to various tasks. To explore whether
    46	the advantages of MPL and IPG (taking SPIG as an example)
    47	solely depend on visual prompt learning, the modules designed
    48	in MIP are replaced with general visual prompt learning
    49	[13], results of which are shown in Table VII. (1) As shown
    50	in Table VII, Variant-2, i.e., baseline+general-prompts, only
    51	improves performance with an average increase of 0.51% mAP
    52	and 0.57% Rank-1 compared to Variant-1, i.e., the baseline. (2)
    53	In contrast, in Variant-3 and Variant-4, we replace the general
    54	prompts learning with our designed MPL module and SIPG
    55	module, introducing significant improvements to baseline, i.e.,
    56	2.66/2.37% mAP and 2.19/2.15% Rank-1 on average, respectively. This result illustrates the necessity to design specifically
    57	for modality and instance adaptation. (3) Furthermore, the
    58	performance of Variant-6 equipped with both MPL and SIPG
    59	is 4.15% higher in mAP and 3.98% higher in Rank-1 than
    60	baseline, which achieves more obvious advantages than that
    61	when only equipped with a single module, while Variant-5
    62	using two sets of general prompts only brings an average
    63	increase of 0.73% mAP and 0.71% Rank-1 compared to
    64	baseline.
    65	The above ablation experiments could denote that the
    66	improvement brought by MPL and IPG modules for VI
    67	ReID tasks is not solely dependent on the effectiveness of
    68	visual prompt learning. We illustrate the necessity of carefully designing modules to help the model adapt to different
    69	modalities and different instances.
    70	5) Ablation About the Product Strategy of Instance-Aware
    71	Prompts: Unlike our generation-based strategy, previous work
    72	[52] proposes a fusion-based instance-aware prompt learning
    73	pipeline. This pipeline generates specific weights for each
    74	instance, which are then used to fuse several prompt prototypes
    75	and create specific instance-aware prompts. To validate the
    76	necessity of our generation-based strategy, we implement a
    77	fusion-based SIPG module and conduct several experiments to
    78	compare it with our designed generation-based SIPG module.
    79	(1) As shown in Table VIII, Variant-7, which adds a fusionbased SIPG module to Variant-6 equipped with an MPL
    80	
    81	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
    82	
    83	8098
    84	
    85	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025
    86	
    87	TABLE VIII
    88	
    89	TABLE IX
    90	
    91	T HE C OMPARISONS OF G ENERATION -BASED AND F USION -BASED
    92	IPG M ODULES . ‘F US .’ M EANS ‘F USION - BASED ’; ‘G EN .’ M EANS
    93	‘G ENERATION - BASED ’. (TAKING THE SIPG M ODULE AS AN
    94	E XAMPLE )
    95	
    96	E FFECTS OF D IFFERENT C OMBINATION WAYS OF T WO K INDS OF P ROMPTS
    97	
    98	module, does not show additional improvement over Variant6, even decreases in some metrics. In contrast, Variant-8,
    99	which adds a generation-based SIPG module to Variant-6,
   100	demonstrates a significant improvement. After introducing the
   101	IAEL loss, comparing Variant-6, Variant-9, and Variant-10,
   102	we can find that, although the performance of Variable-9 with
   103	a fusion-based SIPG module increases with the help of the
   104	IAEL loss, Still significantly lower than Variant-10 using the
   105	generation-based strategy. (2) Variant-1 to Variant-5 show the
   106	results without the MPL module. We can find that the fusionbased SIPG module brings an average improvement of 1.15%
   107	mAP and 0.97% Rank-1 to the baseline when it does not
   108	work with the MPL module. However, its improvement on
   109	average is still 1.20% lower than the improvement brought
   110	by the generation-based SIPG module on average. The results
   111	are similar after introducing the IAEL loss. (3) The reason
   112	behind these results could be that, the awareness ability of
   113	fusion-based prompts may not be strong enough because of
   114	degenerating to the trivial solution. Consequently, when the
   115	two modules work together, the fusion-based SIPG module
   116	might duplicate the role of the MPL module.
   117	To intuitively explore this finding, we exhibit the t-SNE
   118	visualization results of instance-aware prompts produced by
   119	fusion-based and generation-based SIPG modules in Fig. 6.
   120	We randomly sample 448 images of 10 different persons. The
   121	different colors of dots mean different persons. As shown in
   122	Fig. 6, the generation-based prompts have obvious intra-person
   123	distances and clear boundaries, while the prompts generated
   124	by the fusion-based SIPG are distributed in a disordered and
   125	interleaving manner, which indicates that prompts produced by
   126	generation-based SIPG have more fine-grained instance-aware
   127	ability than prompts produced by fusion-based SIPG.
   128	6) Ablation About the Combination Ways of ModalityAware and Self-Guiding Instance-Aware Prompts: As shown
   129	in Table IX, we compare our proposed CPF module with two
   130	alternative methods for combining modality-aware and selfguiding instance-aware prompts, validating the effect of our
   131	CPF module.
   132	
   133	Fig. 6. The t-SNE visualizations results of prompts from fusion-based and
   134	generation-based SIPG modules. Different colors represent distinct identities.
   135	(a) Fusion-based SIPG prompts cluster closely, with less obvious boundaries
   136	between individuals, indicating weaker instance-aware ability. (b) Generationbased SIPG prompts show increased distances between individuals, reflecting
   137	stronger instance-aware ability, crucial for adaptation on diverse instances.
   138	
   139	Concatenation (Cat.): A straightforward way is to concatenate the two types of prompts along the sequence dimension,
   140	which achieves an average performance of 78.76% mAP and
   141	77.89% Rank-1.
   142	Projection (Proj.): Another common way is to add the two
   143	types of prompts together and apply a linear projection layer
   144	with the same input and output dimensions. This projection
   145	layer is designed to learn latent associations and mitigate
   146	potential conflicts between the prompts, leading to an average
   147	improvement of 0.29% mAP and 0.41% Rank-1 over the Cat.
   148	way.
   149	Customized Prompts Fusion (CPF): Our CPF module
   150	refines the modality-aware prompt using a feature-guided
   151	attention weighting mechanism before integrating it with the
   152	instance-aware prompt through a linear projection layer. This
   153	approach further improves performance, achieving an average
   154	gain of 0.83% mAP and 1.14% Rank-1 compared to the Cat.
   155	way.
   156	7) Ablation About the QIPG Module and Its CEL Loss Constraint: We conducted more detailed experiments to confirm
   157	the role of two components in the complete MIP model, i.e.
   158	MIP w/ QIPG, including the QIPG module and the CEL loss
   159	that constrains its learning.
   160	As shown in Table X, we observe that the inclusion of the
   161	QIPG module slightly improves performance compared to the
   162	MIP without QIPG. However, the addition of CEL loss further
   163	boosts performance. This demonstrates that the CEL loss plays
   164	a crucial role in refining the query-guiding instance-aware
   165	prompts, ensuring that they effectively capture identity-specific
   166	information while reducing irrelevant variations.
   167	
   168	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   169	
   170	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
   171	
   172	TABLE X
   173	E FFECTS OF QIPG M ODULE AND CEL L OSS FOR MIP W / QIPG
   174	
   175	8099
   176	
   177	TABLE XII
   178	T HE PARAMETER AND FLOP S TATISTICS OF THE BASELINE M ODEL
   179	AND I TS VARIANTS W ITH A DDED C OMPONENTS D URING THE I NFER ENCE . T HE R ELATIVE C HANGES ↑x.xx% A RE C OMPUTED W ITH
   180	R ESPECT TO THE BASELINE . T HE ‘*’ I NDICATES T HAT THE
   181	R ESULT I S E STIMATED BASED ON THE M ODEL F RAME WORK D ESCRIBED IN T HEIR PAPER
   182	
   183	TABLE XI
   184	T HE E XPERIMENTAL R ESULTS OF O UR MIP U NDER U NSUPERVISED S ETTING AND OTHER S TATE - OF - THE -A RT UVI R E ID M ETHODS
   185	
   186	This ablation experiment confirms that the QIPG module
   187	and CEL loss are effective in enhancing the model’s adaptability to specific query tasks and improving retrieval performance,
   188	especially the CEL loss playing a crucial role in enabling
   189	QIPG to function effectively.
   190	8) Ablation About the Effectiveness of MIP Under Unsupervised Setting: Unsupervised learning [87], [88] has gained
   191	significant attention in many areas due to its ability to
   192	eliminate the reliance on manually annotated labels while
   193	still learning discriminative representations. To evaluate the
   194	effectiveness of our method under unsupervised settings, we
   195	conduct additional experiments where the model is trained
   196	without identity labels. Specifically, we adopt a clusteringbased pseudo-labeling strategy, where features extracted from
   197	the model are clustered, and the resulting pseudo-labels are
   198	used for training.
   199	We then compared the performance of our method in
   200	this unsupervised setting with state-of-the-art Unsupervised
   201	VI ReID (UVI ReID) methods [81], [82], [83], [84], [85],
   202	[86]. As shown in Table XI, while our method does not
   203	surpass the best-performing UVI ReID approaches, it achieves
   204	comparable performance to most of them. This demonstrates
   205	that our framework remains effective even without supervised identity annotations, further validating its adaptability
   206	in unsupervised scenarios. These results also suggest that
   207	incorporating more advanced unsupervised learning techniques
   208	could be a promising direction for further improving VI ReID
   209	performance.
   210	9) Discussion on the Parameters and Computational Complexity: To better analyze the parameter scale and computational overhead of our proposed MIP and several components
   211	during inference, we use fvcore [89] and PyTorch-OpCounter
   212	[90] tool to summarize the parameter count and FLOP for
   213	the baseline model and the models with added components,
   214	
   215	as shown in Table XII. Due to the minimal parameter and
   216	computational overhead of the CPF module, variants 4 and
   217	5 enable CPF by default, and its overhead is not counted
   218	separately.
   219	As shown in Table XII, during the inference phase, our full
   220	model introduces an additional 16.50% in parameter count
   221	and 15.55 GFLOPs in computational complexity compared
   222	to the baseline. The introduction of MPL, SIPG, and QIPG
   223	contributes 1.04%, 8.89%, and 8.20% to the parameter count,
   224	and 0.94 GFLOPs, 12.56 GFLOPs, and 2.01 GFLOPs to the
   225	computational complexity, respectively.
   226	In terms of parameters, we discuss this from two aspects, the
   227	additionally introduced parameters and the overall parameter
   228	scale. The extra parameters introduced by our model are
   229	18.94% of our baseline and relatively small compared to the
   230	baseline. When compared to most existing methods, for example, DEEN [65] introduces an additional 24.82% parameters
   231	over its baseline and that of CSDN [71] even exceeds 100%
   232	of its baseline, our method is efficient in terms of parameter
   233	utilization. As for the overall parameter scale, firstly, our
   234	method is parameter-efficient and it’s fair to compare it to
   235	existing methods using ViT-B/16 as the backbone, e.g. CSDN.
   236	As for ResNet-based methods, as shown in Table I, most
   237	baselines have parameter sizes reaching 2 to 3 times that of
   238	the standard ResNet-50 backbone. Given the DEEN [65] as an
   239	example, as shown in Table XII, its baseline’s total parameter
   240	scale is already comparable to ViT-based methods, and the
   241	percentage of additional parameters introduced is also higher
   242	than that of our method. Therefore, comparing our method
   243	with most ResNet-based methods is also fair.
   244	In terms of computational complexity, it is important to
   245	note that, although our full model requires roughly twice the
   246	computational time of the baseline, the overall computational
   247	overhead remains very low (finishing processing a 256 × 128
   248	image in less than 1 ms. For large-scale image processing,
   249	such as processing 60,000 images (at the same level of scale
   250	with the SYSU-MM01 test set), the computation time for the
   251	
   252	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   253	
   254	8100
   255	
   256	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025
   257	
   258	Fig. 7. Attention map visualization for visible and infrared images. Each of case (a) and (b) corresponds to the same identity. The first row exhibits visible
   259	images, while the second displays infrared images. The second and seventh columns are the visualization results from the baseline network; the third and
   260	eighth columns are from Baseline+MPL; the fourth and ninth columns are from Baseline+IPG (SIPG); and the fifth and tenth columns are the visualization
   261	results from our proposed MIP (w/o QIPG). Compared to the baseline, the heatmaps produced by Baseline+MPL focus on more discriminative parts of
   262	the target’s body, showing improved modality adaptability, and allowing the model to better capture modality-consistent discriminative features. Meanwhile,
   263	Baseline+IPG strengthens instance-relevant feature extraction, leading to more precise attention on identity-relevant regions and expanding the focus to a
   264	broader range of discriminative body parts. When combining both components, MIP further refines the attention maps, concentrating more on discriminative
   265	regions while significantly reducing the focus on extraneous details. This refined attention demonstrates the model’s enhanced capability in identifying and
   266	distinguishing individuals across modalities.
   267	
   268	TABLE XIII
   269	C OMPARISONS OF THE E FFECTIVENESS OF THE SIPG AND SIPGS HALLOW S TRATEGIES
   270	
   271	baseline and MIP on one NVIDIA RTX 3090 GPU is about
   272	20.4s and 44.5s, respectively, which both remains within an
   273	acceptable range. Besides, we conduct the same statistical
   274	experiments on DEEN [65] based on its official implementation code, the inference computation time of the DEEN is
   275	33.71% higher than ours and even more than 2.5 times that
   276	of its baselines. Additionally, we can significantly reduce the
   277	computation time by generating self-guiding instance-aware
   278	prompts only once before the first encoder layer and allowing
   279	subsequent layers to update them independently, which is the
   280	SIPG-Shallow (SIPG-S) strategy. While this approach leads
   281	to a slight performance drop, it remains acceptable, as shown
   282	in Table XIII. In this case, the inference computation time of
   283	DEEN is 117.62% higher than ours.
   284	10) Focused Contents of the MPL Module: Fig. 7 showcases attention maps of infrared and visible images from
   285	the SYSU-MM01 dataset, providing deeper insights into the
   286	efficacy of our approach. Each panel, (a) and (b), features
   287	images of the same identity; the top row displays visible
   288	images, while the bottom row presents infrared images. The
   289	second and seventh columns are the visualization results
   290	
   291	from the baseline network; the third and eighth columns are
   292	from Baseline+MPL; the fourth and ninth columns are from
   293	Baseline+IPG (SIPG); and the fifth and tenth columns are the
   294	visualization results from our proposed MIP (w/o QIPG). As
   295	depicted in Fig. 7, compared to the baseline, the heatmaps
   296	produced by Baseline+MPL and Baseline+IPG both focus on
   297	more body parts of the person. Baseline+MPL focus on more
   298	discriminative parts of the target’s body, showing improved
   299	modality adaptability, and allowing the model to better capture modality-consistent discriminative features. Meanwhile,
   300	Baseline+IPG strengthens instance-relevant feature extraction,
   301	leading to more precise attention on identity-relevant regions
   302	and expanding the focus to a broader range of discriminative body parts. When combining both components, MIP
   303	further refines the attention maps, concentrating more on
   304	discriminative regions while significantly reducing the focus
   305	on extraneous details. This refined attention demonstrates the
   306	model’s enhanced capability in identifying and distinguishing
   307	individuals across modalities.
   308	V. C ONCLUSION
   309	In this work, we introduce a novel VI ReID method
   310	named the Modality-aware and Instance-aware Visual Prompts
   311	(MIP) network. This network leverages visual prompt learning
   312	specifically tailored for the VI ReID. Our design includes
   313	a Modality-aware Prompt Learning (MPL) module alongside two Instance-aware Prompt Generators: the Self-guiding
   314	Instance Prompt Generator (SIPG) and the Query-guiding
   315	Instance Prompt Generator (QIPG). These generators produce
   316	modality-aware and instance-aware prompts, directing the
   317	ReID model to efficiently adapt to various modalities, input
   318	instances, and current query tasks. Additionally, we introduce
   319	
   320	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   321	
   322	
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
     2	
     3	where a s , aq,p , aq,n are features extracted by self-guiding
     4	instance-aware prompts, query-guiding instance-aware
     5	prompts generated with query images of the same ID and
     6	different ID, respectively; ·¯ means a feature processed by BN
     7	layer; m is the margin.
     8	By optimizing the self-guiding instance-aware prompt generation process, the CEL loss ensures these prompts effectively
     9	capture identity-relevant and matching-useful information,
    10	improving the adaptability for matching with different query
    11	tasks.
    12	3) Cross-Modality Triplet Loss: Contrastive learning [55],
    13	[56] aims to enhance feature discrimination by pulling positive
    14	pairs closer and pushing negative pairs apart. Inspired by
    15	the idea of contrastive learning and widely used triplet loss,
    16	we employ the cross-modality triplet loss, which extends the
    17	traditional triplet loss to enhance identity discrimination across
    18	different modalities. Unlike standard triplet loss, where all
    19	samples come from the same modality, cross-modality triplet
    20	loss constructs triplets where the anchor and positive/negative
    21	samples originate from different modalities. Specifically, in our
    22	setup, infrared images serve as anchors, while the positive and
    23	negative samples are drawn from visible images. The crossmodality triplet loss can be written formally as follows,
    24	
    25	
    26	
    27	LCM−T RI = max d āir , āvis,p − d āir , āvis,n + m, 0
    28	(16)
    29	where air/vis are features of infrared/visible images; a p/n are
    30	features of images of positive/negative samples with the anchor
    31	images; ·¯ means a feature processed by BN layer; m is the
    32	margin.
    33	4) Overall Objective Function: We follow the training
    34	strategy of CLIP-ReID [50]. For the text-encoder training, we
    35	use the same objective function setting as CLIP-ReID. For the
    36	image-encoder training, we adopt a hybrid training strategy
    37	and hybrid objective function.
    38	For our MIP network with the MPL and SIPG modules, we
    39	utilize IAEL loss LIAEL and widely used cross-entropy loss
    40	LID , triplet loss LT RI , text cross-entropy loss Li2tce in CLIPReID, and MSEL loss in PMT [11] for training. The objective
    41	function can be defined as,
    42	L1 = LID + LT RI + Li2tce + 0.5 × L MS EL + α1 LIAEL
    43	
    44	(17)
    45	
    46	where α1 is the trade-off parameter, and we set α = 0.5 in
    47	practice; the IAEL loss LIAEL is only used after the 5-th epoch.
    48	For the model with the QIPG module, we utilize our IAEL
    49	loss LIAEL , CEL loss LCEL , cross-modality triplet loss LCM−T RI ,
    50	and widely used cross-entropy loss LID for training. The
    51	objective function can be defined as,
    52	L2 = LID + LCM−T RI + 0.5 × LIAEL + α2 LCEL
    53	
    54	(18)
    55	
    56	where α2 is the trade-off parameters, and we set α2 = 1.0 in
    57	practice; the IAEL loss LIAEL and CEL loss LCEL are only
    58	used after the 5-th epoch.
    59	IV. E XPERIMENT
    60	A. Datasets and Evaluation Protocols
    61	Three mainstream public person VI ReID datasets, SYSUMM01 [57], RedDB [72], LLCM [65], and a vehicle VI ReID
    62	dataset RGBN300 [73] are used for our experiments.
    63	
    64	8093
    65	
    66	SYSU-MM01 [57] consists of 491 persons whose 286,628
    67	visible images and 15,792 infrared images are captured by 4
    68	visible and 2 infrared cameras. The dataset is divided into the
    69	training set and the test set. There are 22,258 visible images
    70	and 11,909 infrared images of 395 persons in the training set,
    71	and the test set includes images of 96 persons whose 3,803
    72	infrared images are used as query images and several visible
    73	images are randomly selected as gallery images. According
    74	to the general setting, the composition of gallery images is
    75	decided by two search modes: All-Search (all images) and
    76	Indoor-Search (only images under indoor cameras), and two
    77	selection modes: Single-Shot (1 visible image per person) and
    78	Multi-Shot (10 visible images per person).
    79	RegDB [72] consists of 412 persons whose 4,120 visible
    80	and 4,120 infrared images are captured by one visible camera
    81	and one infrared camera, and there are 10 visible and 10
    82	infrared images for each person. The training set includes
    83	the images of 206 randomly chosen persons and the test set
    84	contains the images of the remaining 206 persons. In the test
    85	mode Visible to Infrared, the visible images are used as query
    86	images, and the infrared images are used as gallery images,
    87	and the arrangement is reversed in Infrared to Visible mode.
    88	LLCM [65] comprises 46,767 bounding boxes across 1,064
    89	identities, collected over an extended period under varying climate conditions and clothing styles. The training set
    90	includes 30,921 images of 713 individuals, with 16,946 images
    91	captured in the visible modality and 13,975 images in the
    92	infrared modality. The testing set contains 13,909 images of
    93	351 individuals, split between 8,680 images from the visible
    94	modality and 7,229 images from the infrared modality. Similar
    95	to RegDB, LLCM also includes two test modes, Infrared to
    96	Visible and Visible to Infrared.
    97	RGBN300 [73] comprises 50,125 image pairs, representing
    98	300 distinct vehicle identities in both RGB and infrared
    99	modalities. The dataset features a variable number of image
   100	pairs for each vehicle, ranging from 50 to 200, with an average
   101	of 6.7 camera views per vehicle. Out of the total, 150 vehicles
   102	(encompassing 25,200 image pairs) are randomly chosen to
   103	form the training set. The remaining 150 vehicles, with 24,925
   104	image pairs, are designated as the testing set. Within this
   105	testing set, 4,985 image pairs are randomly selected as the
   106	query set, while the remainder serves as the gallery set.
   107	Evaluation Protocols. We adopt the Cumulative Matching Characteristic curve (CMC) and mean Average Precision
   108	(mAP) as the evaluation metrics on the above datasets, to
   109	evaluate the performance of our proposed model. For CMC,
   110	we calculate the percentage of correctly retrieved images
   111	among top-1 results (Rank-1 accuracy) based on similarity.
   112	B. Implementation Details
   113	Our proposed MIP method is implemented on the Pytorch
   114	framework, and one NVIDIA RTX3090 GPU is used to
   115	execute experiments. We use a pre-trained CLIP [47] with
   116	ViT-B/16 [43] as the backbone. The size of images is resized
   117	to 256 × 128 for person VI ReID datasets and 256 × 256
   118	for the vehicle VI ReID dataset RGBN300, and the data
   119	augmentations, including random cropping, color jittering,
   120	random erasing, and grayscale, are applied to transform the
   121	
   122	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   123	
   124	8094
   125	
   126	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025
   127	
   128	TABLE I
   129	T HE E XPERIMENT R ESULTS OF O UR MIP AND OTHER S TATE - OF - THE -A RT M ETHODS U NDER VARIOUS T EST M ODES OF SYSU-MM01 AND R EG DB
   130	DATASETS . T HE M ETHODS U SING CLIP AS THE BACKBONE A RE S PECIALLY M ARKED (O UR M ETHOD I S N OT MARKED ); THE B OLD AND
   131	U NDERLINED F ONTS I NDICATE THE B EST AND S ECOND -B EST P ERFORMANCE , R ESPECTIVELY; ‘IR’ M EANS ‘I NFRARED ’; ‘VIS’ M EANS
   132	‘V ISIBLE ’; THE ‘*’ D ENOTES T HAT THE R ESULTS A RE R EPRODUCED BY U S BASED ON T HEIR PAPER AND R ELEASED C ODES ; ‘N×’
   133	I NDICATES T HAT I TS PARAMETER S CALE I S N T IMES T HAT OF THE S TANDARD V ERSION OF I TS BACKBONE .I N S UMMARY,
   134	O UR P ROPOSED MIP O UTPERFORMS M OST OF THE S TATE - OF - THE -A RT M ETHODS ON B OTH T WO M AINSTREAM
   135	DATASETS
   136	
   137	input images during training. In practice, we adopt a twostage training strategy. For the first stage, we train the model
   138	with the MPL, SIPG, and CFP modules. The batch size is
   139	set to 64 for person datasets and 32 for the RGBN300. We
   140	adopt the Adam optimizer to train the model for 60 epochs
   141	with an initial learning rate of 5.0 × 10−3 for the RegDB
   142	dataset and 5.0 × 10−4 for the other three datasets divided
   143	by 10 at the 18-th and 36-th epochs. For the second stage,
   144	we freeze the part trained in the first stage and only train the
   145	QIPG module and the additional transformer encoder layer.
   146	The batch size is set to 160 for person VI ReID datasets and 80
   147	for the RGBN300. The learning rate is initialized as 1.0×10−2
   148	for all datasets and divided by 10 at the 10-th, 25-th, 50-th
   149	epochs.
   150	Following the framework of Fig. 2, we maintain two sets
   151	of modality-aware prompts for each layer, a shared SIPG
   152	module for all layers, and an extra encoder layer with the
   153	QIPG module. In ablation studies 10), we provide more
   154	
   155	detailed statistics results and discussions on the parameters
   156	and computational complexity of these components.
   157	C. Comparison With State-of-the-Art Methods
   158	We compare our MIP with existing state-of-the-art methods
   159	for VI ReID, in mainstream datasets SYSU-MM01, RegDB,
   160	LLCM, and a vehicle VI ReID dataset RGBN300. The comparison results are reported in Table I, Table II, and Table III,
   161	where bold and underlined fonts indicate the best and secondbest performance, respectively.
   162	For the comparison results on the widely used SYSU-MM01
   163	and RegDB benchmarks, we have specifically indicated the
   164	backbone types and parameter scales for each method (with
   165	methods using CLIP-pretrained ViT-B/16 being separately
   166	noted). The parameter scale ‘N×’ indicates that the baseline
   167	parameter count is ‘N’ times that of the standard version of
   168	the declared backbone model. Unless the additional parameters
   169	
   170	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   171	
   172	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
   173	
   174	TABLE II
   175	T HE E XPERIMENT R ESULTS OF O UR MIP AND OTHER S TATE - OF - THE -A RT
   176	M ETHODS U NDER VARIOUS T EST M ODES OF THE LLCM DATASET.
   177	T HE B OLD AND U NDERLINED F ONTS I NDICATE THE B EST AND
   178	S ECOND -B EST P ERFORMANCE , RESPECTIVELY. ‘IR’ M EANS
   179	‘I NFRARED ’; ‘VIS’ M EANS ‘V ISIBLE ’. I N S UMMARY, O UR
   180	P ROPOSED MIP O UTPERFORMS OTHER S TATE - OF - THE A RT M ETHODS ON LLCM
   181	
   182	TABLE III
   183	T HE E XPERIMENT R ESULTS OF O UR MIP AND OTHER S TATE - OF - THE -A RT
   184	M ETHODS ON THE V EHICLE VI R E ID DATASET RGBN300. T HE B OLD
   185	AND U NDERLINED F ONTS I NDICATE THE B EST AND S ECOND -B EST
   186	P ERFORMANCE , RESPECTIVELY. I N S UMMARY, O UR P ROPOSED
   187	MIP O UTPERFORMS OTHER S TATE - OF - THE -A RT M ETHODS
   188	ON RGBN300
   189	
   190	introduced by a designed component or strategy are significantly large, we have not included them in the parameter count.
   191	As for the additional parameters introduced by components of
   192	our method, we have specifically summarized and discussed
   193	them in the ablation study.
   194	Notably, although some methods use ResNet50 as the backbone, most of them have designed baselines with parameter
   195	counts 2 to 3 times that of the standard ResNet50 backbone,
   196	whose parameter scales are already comparable to ViT-based
   197	methods. Additionally, it is worth noting that some methods
   198	may use the widely adopted AGW baseline, which has a
   199	parameter scale of ‘3×’, but without clearly specifying the
   200	baseline method used. Nevertheless, we have treated their
   201	parameter scale as ‘1×’.
   202	Overall, the excellent performance of our model on these
   203	four public datasets fully demonstrates the effectiveness of our
   204	proposed MIP method.
   205	1) SYSU-MM01: As shown in Table I, we can observe
   206	that our proposed MIP method shows the best results on
   207	
   208	8095
   209	
   210	most of the metrics under all SYSU-MM01 dataset modes.
   211	the best results on most of the metrics under all SYSUMM01 dataset modes. Compared with other CLIP-based
   212	methods CSDN, PMT (CLIP), CMTR (CLIP), ViT-based
   213	methods, PMT, CMTR, DFLN-ViT, SPOT, and recent CNNbased method DEEN, TMD, PMWGCN, MSALNet, ACD,
   214	CSC-Net, CycleTrans, SSRR, our MIP outperforms the stateof-the-art results by 3.22% mAP and 0.34% Rank-1 under
   215	the All-Search + Single-Shot mode. Under changeable AllSearch + Multi-Shot mode, MIP also achieves the best mAP
   216	results and comparable Rank-1 results. And under IndoorSearch mode, MIP also achieves the best or comparable results
   217	among all the SOTA methods.
   218	Notably, our MIP w/o QIPG method using only one-stage
   219	training similarly outperforms most of the existing methods in
   220	most of the metrics.
   221	2) RegDB: As shown in Table I, our proposed MIP method
   222	also achieves the best performance on RegDB. Notably,
   223	Method SSRR [70] achieves better performance than ours
   224	on the RegDB dataset. We believe this is because SSRR
   225	adopts PCB-like [3] local feature learning and local alignment techniques, which are highly effective for datasets like
   226	RegDB, where images from different modalities only differ in
   227	modality while remaining perfectly aligned in structure and
   228	spatial position. While our method does not incorporate such
   229	local alignment techniques, it still outperforms other existing
   230	methods on RegDB and achieves comparable performance
   231	with SOTA methods. Compared with the other top-performing
   232	methods, overall, our MIP significantly outperforms these
   233	methods with an average performance of 89.50%, which is
   234	1.80% higher than the averaged second-highest MSALNet
   235	method, and the MIP w/o QIPG method using only onestage training still outperforms MSALNet by an average
   236	performance of nearly 1%. In addition, MIP outperforms the
   237	CSDN method which also uses the CLIP backbone by an
   238	average performance of 3.33%.
   239	3) LLCM: As shown in Table II, our proposed MIP method
   240	achieves promising performance on LLCM. LLCM includes
   241	low-light scenes that pose a greater challenge to VI ReID task.
   242	On this dataset, MIP achieved the best performance in both
   243	two test modes, with an average 2.66% higher performance
   244	than the state-of-the-art method DEEN. Similarly, our MIP
   245	w/o QIPG method using only one-stage training achieves the
   246	second-highest performance after the full MIP w/ QIPG.
   247	4) RGBN300: As shown in Table III, we extended the evaluation of our MIP to the vehicle VI ReID dataset RGBN300, to
   248	further validate the broad validity of MIP in visible-infrared
   249	tasks, Our MIP shows the best results with 97.34% Rank1 and 78.35% mAP, which outperforms the state-of-the-art
   250	method GPFNet by an average performance of 6.20%. Such
   251	outstanding performance underscores the broad applicability
   252	of MIP in VI ReID tasks, effectively handling not only the
   253	person but also the vehicle re-identification challenges.
   254	D. Ablation Studies
   255	In this subsection, we conduct a series of ablation experiments to evaluate the effectiveness of our proposed MIP.
   256	We begin by validating the effects of each component on
   257	two mainstreaming benchmarks, and then conduct the ablation
   258	
   259	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   260	
   261	8096
   262	
   263	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025
   264	
   265	TABLE IV
   266	
   267	TABLE V
   268	
   269	T HE E FFECTS OF O UR P ROPOSED C OMPONENTS ON SYSU-MM01
   270	
   271	T HE E FFECTS OF O UR P ROPOSED C OMPONENTS ON R EG DB. ‘IR’ M EANS
   272	‘I NFRARED ’; ‘VIS’ M EANS ‘V ISIBLE ’
   273	
   274	experiments about the trade-off parameters and the length of
   275	prompts. Next, we compare our designed prompts learning
   276	modules with general prompt-based approaches. Subsequently,
   277	we discuss the necessity of the generation-based product strategy of instance-aware prompts. We also conduct the ablation
   278	about the combination ways of modality-aware and instanceaware prompts to validate the effects of CPF module. In
   279	addition, we conduct the ablation about the QIPG module
   280	and the CEL loss. Moreover, we conduct the ablation about
   281	the effects of our method under the unsupervised setting.
   282	Furthermore, we discuss the parameters and computational
   283	complexity of our method and some other existing methods.
   284	Finally, we present the CAM visualization results to further
   285	analyze the effects of our proposed model. Notably, all the
   286	ablation experiments are conducted on the SYSU-MM01
   287	dataset under Single-Shot mode, except when specifically
   288	stated.
   289	1) Effectiveness of Proposed Components: As shown in
   290	Table IV, to evaluate the effectiveness of our proposed MPL
   291	module, SIPG module, CPF module, IAEL loss, and QIPG
   292	module, we add these five components to the baseline gradually and evaluate the performances. (1) Compared with
   293	Variant-1, i.e., baseline model, Variant-2 is trained with an
   294	additional MPL module, which improves 2.66% mAP and
   295	2.19% Rank-1 on average under two search modes on SYSUMM01 dataset. (2) Variant-3 adopts baseline+SIPG as the
   296	model and improves the performance of the baseline model
   297	by 2.37% mAP and 2.15% Rank-1 on average on SYSUMM01. (3) Based on Variant-3, Variant-4 adds IAEL loss
   298	to promote the prompt generated by the SIPG module to be
   299	more customized to each input instance, achieving an average
   300	improvement of 2.84% mAP and 2.27% Rank-1 compared
   301	with the baseline. (4) Variant-5 equips the baseline with both
   302	MPL and SIPG modules, which brings a further improvement of 1.64% mAP and 1.80% Rank-1 on average than
   303	baseline+MPL and baseline+SIPG. This indicates that our
   304	designed modules can complement to promote performance
   305	improvement jointly. (5) Variant-6 achieves an additional
   306	average improvement of 1.04% in mAP and 0.53% in Rank-1
   307	compared to Variant-5, thanks to the enhancement from the
   308	IAEL loss. (6) Variant-7 introduces the CPF module to fuse
   309	the two kinds of prompts carefully and achieves an additional
   310	average improvement of 1.61% in mAP and 1.16% in Rank-1
   311	
   312	compared to Variant-5. (7) Variant-8, the full model of the
   313	first training stage, utilizes the IAEL loss and the CPF module
   314	together and achieves an additional average improvement of
   315	1.82% in mAP and 1.81% in Rank-1 compared to Variant5. (8) Finally, Variant-9, our final full model (MIP) that
   316	introduces the QIPG module based on Variant-8, achieves an
   317	additional average improvement of 1.28% in mAP and 1.20%
   318	in Rank-1.
   319	As shown in Table V, We also conduct the same ablation
   320	experiments on RegDB, and the addition of individual components similarly brings improvements to the model.
   321	From these experiments, we can find that the MPL module,
   322	SIPG module, CPF module, IAEL loss, and QIPG module
   323	are effective in adapting the model to different modalities and
   324	instances and enhancing its capability for VI ReID.
   325	2) Ablation About the Trade-off Parameters: For the optimization of hybrid objective functions composed of multiple
   326	losses, it is common to add weights to losses other than
   327	the main losses (such as LCE , LT RI ) as trade-off parameters
   328	to regulate the optimization process. Typically, a loss with
   329	a higher trade-off parameter will dominate the optimization
   330	process. We introduced weights α1 and α2 in two overall
   331	objective functions, respectively, to balance the losses. In this
   332	ablation study, we evaluated the impact of trade-off parameters
   333	under All-Search+Single-Shot mode on SYSU-MM01.
   334	As shown in Fig. 5, the trade-off parameters are set to
   335	different values {0, 0.5, 1.0, 2.0, 3.0, 5.0}. When α1 and α2
   336	are set to zero, the network is optimized solely through main
   337	losses like {LCE , LT RI } and {LCE , LCM−T RI , LIAEL }, respectively.
   338	We plotted the Rank-1 and mAP lines when parameters are
   339	set to zero as the referring baseline. (1) As shown in the first
   340	row, by introducing the trade-off parameter α1 greater than
   341	zero for LIAEL in L1 , we observed significant improvements
   342	in both Rank-1 and mAP performance, and the performance
   343	is relatively stable across different trade-off parameter values.
   344	Thus, the performance of IAEL loss LIAEL is not sensitive
   345	to these varying values, indicating its capability is not highly
   346	dependent on the specific process of multi-loss optimization.
   347	(2) As shown in the second row, introducing the trade-off
   348	parameter α2 greater than zero for LCEL in L2 also resulted in
   349	notable performance improvements and remained stable within
   350	the range {0.5, 1.0, 2.0}. Hence, the performance of CEL loss
   351	
   352	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.
   353	
   354	WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING
   355	
   356	8097
   357	
   358	TABLE VII
   359	T HE E FFECTS OF O UR D ESIGNED MPL AND SIPG M ODULES C OMPARED
   360	” M EANS

codex
MIP 的证据闭环很完整：不只做模块消融，还专门把“普通提示是否已经足够”“生成式实例提示是否必要”“两类提示会不会冲突”拆开验证。这类论文值得重点学它怎么把视角和证据绑定。
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
/bin/zsh -lc "pdftotext -f 12 -l 16 'Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf' - | nl -ba | sed -n '1,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	171:12
     2	
     3	X. Zhang et al.
     4	Table 2. Comparison with Other Domain Adaptation Methods
     5	
     6	Method
     7	
     8	3.7
     9	
    10	D+C+MS → M
    11	
    12	M+C+MS → D
    13	
    14	D+C+M → MS
    15	
    16	mAP
    17	
    18	R1
    19	
    20	R5
    21	
    22	mAP
    23	
    24	R1
    25	
    26	R5
    27	
    28	mAP
    29	
    30	R1
    31	
    32	R5
    33	
    34	BORT [4]
    35	
    36	79.3%
    37	
    38	92.1%
    39	
    40	95.9%
    41	
    42	65.1%
    43	
    44	79.5%
    45	
    46	88.9%
    47	
    48	24.3%
    49	
    50	50.6%
    51	
    52	62.5%
    53	
    54	DaDiL-E [19]
    55	
    56	82.6%
    57	
    58	92.8%
    59	
    60	97.5%
    61	
    62	64.9%
    63	
    64	79.0%
    65	
    66	88.8%
    67	
    68	30.1%
    69	
    70	58.4%
    71	
    72	65.0%
    73	
    74	VDD [25]
    75	
    76	85.7%
    77	
    78	94.5%
    79	
    80	97.0%
    81	
    82	68.3%
    83	
    84	81.6%
    85	
    86	89.9%
    87	
    88	33.4%
    89	
    90	62.5%
    91	
    92	73.9%
    93	
    94	MMT+GRL [5]
    95	
    96	77.1%
    97	
    98	90.4%
    99	
   100	96.8%
   101	
   102	64.3%
   103	
   104	77.6%
   105	
   106	88.1%
   107	
   108	22.6%
   109	
   110	46.3%
   111	
   112	59.8%
   113	
   114	MMT+DSBN [2]
   115	
   116	81.1%
   117	
   118	92.8%
   119	
   120	97.3%
   121	
   122	65.6%
   123	
   124	79.6%
   125	
   126	89.1%
   127	
   128	22.6%
   129	
   130	49.2%
   131	
   132	62.1%
   133	
   134	RDSBN [1]
   135	
   136	86.0%
   137	
   138	94.8%
   139	
   140	97.9%
   141	
   142	68.9%
   143	
   144	82.1%
   145	
   146	90.4%
   147	
   148	34.9%
   149	
   150	64.7%
   151	
   152	76.2%
   153	
   154	FFCCLS [24]
   155	
   156	86.2%
   157	
   158	95.1%
   159	
   160	97.9%
   161	
   162	69.6%
   163	
   164	83.5%
   165	
   166	91.3%
   167	
   168	35.8%
   169	
   170	65.3%
   171	
   172	77.3%
   173	
   174	Ours
   175	
   176	85.1%
   177	
   178	94.3%
   179	
   180	98.1%
   181	
   182	72.7%
   183	
   184	83.3%
   185	
   186	91.4%
   187	
   188	31.6%
   189	
   190	59.6%
   191	
   192	71.5%
   193	
   194	Comparison with State-of-the-Art Methods
   195	
   196	This article compares the proposed algorithm with the most advanced domain adaptation algorithm
   197	on three public datasets, and the comparison results are shown in Table 2 and Figure 4. Among
   198	them, the method that achieves the best results on the Market dataset uses residual network 101
   199	layers (ResNet101) as the backbone network, and its parameter volume reaches 42.5 M, which is
   200	much larger than the parameter volume of our model. Under the premise of ensuring lightweight,
   201	the mAP gap between our method and the most advanced method is only 1.1%. On the Duke
   202	dataset, our model achieves the best results compared with other advanced models, and the mAP
   203	is improved by 3.1% compared with the current most advanced model. The method that achieves
   204	the best results on the MSMT dataset is the same as the method on the Market1501, both of which
   205	use ResNet101 as the backbone network. Our method is much lower than the above method in
   206	Params and FLOPs, and the mAP gap is only 4.2%. Although the rectification domain-specific
   207	batch normalization (RDSBN) and variational domain disentanglement (VDD) methods are
   208	1.8–3.3% higher in mAP, they both use residual network 50 layers (ResNet50) as the backbone
   209	network, which is much higher than ours in terms of Params and FLOPs.
   210	To conclude, our framework demonstrates comparable or superior performance relative to
   211	advanced UDA approaches on the three public datasets while meeting lightweight requirements.
   212	This shows that the dual-branch lightweight network based on the whole and the part proposed in
   213	this article can learn effective knowledge from multiple source domains through SpML, thereby
   214	enhancing the reliability of pseudo-labels and improving the effect of domain adaptation.
   215	This experiment also uses a private dataset from a real application scenario to test the proposed
   216	method. Compared with public datasets, the environment in real scenes is more complex, and there
   217	are problems such as camera angle diversity and significant color differences, which increase the
   218	difficulty of processing. Therefore, this experiment contributes to a more thorough assessment
   219	of the model’s capabilities and applicability in real-world applications. Our evaluation employs
   220	DukeMTMC, Market1501, and CUHK03 datasets as source domains, with detailed results presented
   221	in Table 3. The results show that our method is 2.6% higher than other methods and achieves the
   222	best performance. Compared with supervised learning, our method still has an 18.9% mAP gap, but
   223	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   224	
   225	Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID
   226	
   227	171:13
   228	
   229	Fig. 4. Comparison of lightweight indicators.
   230	Table 3. Testing on Real Application Scenario Datasets
   231	
   232	Duke+CUHK+Market → Railway
   233	
   234	Method
   235	
   236	mAP
   237	
   238	R1
   239	
   240	R5
   241	
   242	R10
   243	
   244	Direct Transfer
   245	
   246	13.4%
   247	
   248	16.8%
   249	
   250	26.5%
   251	
   252	32.0%
   253	
   254	MMT+withsource
   255	
   256	18.6%
   257	
   258	20.4%
   259	
   260	31.5%
   261	
   262	36.6%
   263	
   264	MMT+M3L
   265	
   266	26.8%
   267	
   268	29.6%
   269	
   270	42.0%
   271	
   272	48.0%
   273	
   274	RDSBN
   275	
   276	28.6%
   277	
   278	30.2%
   279	
   280	40.3%
   281	
   282	45.4%
   283	
   284	Ours
   285	
   286	31.2%
   287	
   288	35.0%
   289	
   290	46.5%
   291	
   292	53.6%
   293	
   294	Supervised training
   295	ResNet50
   296	
   297	42.0%
   298	
   299	47.5%
   300	
   301	65.3%
   302	
   303	73.8%
   304	
   305	OSNet
   306	
   307	45.9%
   308	
   309	51.5%
   310	
   311	69.5%
   312	
   313	76.5%
   314	
   315	Ours
   316	
   317	50.1%
   318	
   319	56.1%
   320	
   321	71.3%
   322	
   323	78.2%
   324	
   325	the unsupervised method has achieved relatively impressive performance without relying on any
   326	labeled data, proving its important value in scenarios where labeled data is scarce.
   327	3.8
   328	
   329	Ablation
   330	
   331	This experiment sets up seven ablation experiments to verify the effectiveness of the proposed
   332	model and multi-source domain adaptation method. The dataset was set to Market, CUHK, and
   333	MSMT as source domains, and Duke as the target domain. As shown in Table 4, when only the
   334	network based on the whole feature branch is used for multi-source domain learning, the mAP
   335	index result obtained is 0.5% different from the result of using only the network based on the partial
   336	feature branch. It can be said that the learning ability of the two networks is similar under this
   337	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   338	
   339	171:14
   340	
   341	X. Zhang et al.
   342	Table 4. Ablation Experiment
   343	
   344	Method
   345	
   346	Dataset
   347	
   348	mAP
   349	
   350	R1
   351	
   352	R5
   353	
   354	R10
   355	
   356	SpML+W
   357	
   358	Duke
   359	
   360	50.8%
   361	
   362	70.0%
   363	
   364	81.5%
   365	
   366	85.2%
   367	
   368	SpML+P
   369	
   370	Duke
   371	
   372	50.3%
   373	
   374	69.0%
   375	
   376	80.7%
   377	
   378	84.6%
   379	
   380	SpML+W+P
   381	
   382	Duke
   383	
   384	53.4%
   385	
   386	72.0%
   387	
   388	82.7%
   389	
   390	86.0%
   391	
   392	MMT+W
   393	
   394	Duke
   395	
   396	64.3%
   397	
   398	77.9%
   399	
   400	87.4%
   401	
   402	89.8%
   403	
   404	MMT+SpML+W
   405	
   406	Duke
   407	
   408	69.5%
   409	
   410	82.0%
   411	
   412	90.0%
   413	
   414	92.4%
   415	
   416	MMT+SpML+P
   417	
   418	Duke
   419	
   420	72.1%
   421	
   422	83.5%
   423	
   424	91.5%
   425	
   426	93.7%
   427	
   428	MMT+SpML+W+P
   429	
   430	Duke
   431	
   432	72.7%
   433	
   434	83.3%
   435	
   436	91.4%
   437	
   438	93.7%
   439	
   440	Table 5. Effectiveness of MMT and SpML across Different
   441	Backbone Capacities
   442	
   443	Method
   444	
   445	Dataset
   446	
   447	mAP
   448	
   449	R1
   450	
   451	ResNet50+SpML
   452	
   453	Duke
   454	
   455	50.4%
   456	
   457	70.1%
   458	
   459	ResNet50+MMT
   460	
   461	Duke
   462	
   463	62.9%
   464	
   465	76.1%
   466	
   467	ResNet50+MMT+SpML
   468	
   469	Duke
   470	
   471	65.1%
   472	
   473	77.9%
   474	
   475	WPOSNet+SpML
   476	
   477	Duke
   478	
   479	53.4%
   480	
   481	72.0%
   482	
   483	WPOSNet+MMT
   484	
   485	Duke
   486	
   487	67.5%
   488	
   489	79.4%
   490	
   491	WPOSNet+MMT+SpML
   492	
   493	Duke
   494	
   495	72.7%
   496	
   497	83.3%
   498	
   499	dataset. However, when these two branches are used for feature extraction at the same time, the
   500	mAP index of the model can be improved by nearly 3%, which shows that the dual-branch model
   501	based on the whole and the local can obtain richer feature representation and better learning ability
   502	through multi-source domain learning.
   503	After multi-source domain learning, the model performs domain adaptation based on MMT. If
   504	the SpML method is not used and only pre-training is performed in multiple source domains, the
   505	mAP obtained by domain adaptation is 5.2% less than that after use. This verifies that SpML can
   506	better learn domain-invariant knowledge from multiple source domains and help learning in the
   507	domain adaptation stage. After further domain adaptation, the model performance of both wholebased and part-based single-branch networks has been significantly improved, among which the
   508	part-based single-branch network achieved better performance. After adopting dual branches, the
   509	domain adaptive ability of the model is further improved. Although the mAP improvement brought
   510	by the global branch is relatively small because fine-grained identity cues are mainly captured
   511	by local regions, the global branch remains indispensable since it provides holistic contextual
   512	information that compensates for missing or unreliable local details and thereby ensures robust
   513	feature representation across domain shifts. In summary, the multi-source domain lightweight
   514	dual-branch domain adaptation method proposed in this article can effectively learn multiple
   515	domain knowledge and effectively perform domain adaptation.
   516	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   517	
   518	Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID
   519	
   520	171:15
   521	
   522	Fig. 5. Heatmap visualization on the DukeMTMC dataset.
   523	
   524	To further verify the generality of the proposed MMT and SpML frameworks, we conduct
   525	additional ablation studies on backbones with different model capacities. In addition to the proposed
   526	lightweight WPOSNet, ResNet50 is adopted as a representative larger backbone, while keeping
   527	all training settings identical for a fair comparison. As reported in Table 5, both SpML and MMT
   528	consistently improve performance across different backbones. On ResNet50, SpML introduces a
   529	modest performance gain, while MMT yields a more substantial improvement by enhancing pseudolabel reliability during target-domain adaptation. When both strategies are combined, further
   530	performance gains are achieved, demonstrating their complementary effects. These observations
   531	indicate that the effectiveness of SpML and MMT is not restricted to a specific network architecture,
   532	but remains robust across backbones with different capacities, validating the backbone-agnostic of
   533	the proposed method.
   534	3.9
   535	
   536	Visualization
   537	
   538	This article uses GradCAM heatmap to visualize the features of some DukeMTMC data, as shown in
   539	Figure 5. In the C1 and C2 viewports of the first example, it can be found that the part-based model
   540	pays more attention to the backpack area than the whole-based model, while the whole-based model
   541	mostly pays attention to the head or feet. Similarly, the third and fourth rows are another pedestrian
   542	example, where the two models pay attention to roughly the same area, but the part-based model
   543	pays attention to a little more area. In general, the part-based model focuses on more parts of
   544	its features and can complement the whole-based features. Therefore, our method considers the
   545	characteristics of these two models at the same time, so it can obtain richer pedestrian features
   546	and improve the accuracy of Re-ID. The visualization results of our method on the DukeMTMC
   547	and Railway datasets are shown in Figure 6. Although there are a large number of pedestrian
   548	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   549	
   550	171:16
   551	
   552	X. Zhang et al.
   553	
   554	Fig. 6. Results visualization on DukeMTMC and Railway datasets.
   555	
   556	images with similar clothing and insufficient clarity in the Railway dataset. The visualization results
   557	demonstrate that our method can still make good matching results for most pedestrians.
   558	4
   559	
   560	Conclusions
   561	
   562	In this study, we present a lightweight multi-source domain adaptation person Re-ID method, which
   563	can more effectively utilize the source domain for target domain adaptation and remain lightweight.
   564	First, to meet the lightweight requirement and achieve efficient pedestrian recognition, this article
   565	proposes a dual-branch network, which effectively combines a whole-level feature branch and a
   566	part-level feature branch based on OSNet. Second, to fully utilize existing source domain knowledge,
   567	this article combines the pseudo-label-based domain adaptation method with the meta-learning
   568	method. Through SpML and style enhancement techniques, the source domains are set as tasks
   569	of different difficulties, and then the model is allowed to learn domain-invariant knowledge from
   570	multiple source domains from easy to difficult, thereby improving the quality of pseudo-labels
   571	in the domain adaptation stage. Finally, tests on multiple public datasets prove that our model
   572	can learn richer pedestrian features and learn effective knowledge from the source domain to
   573	improve domain adaptation performance. In addition, tests on real Re-ID application scenarios
   574	further demonstrate the practical application potential of this study and verify the effectiveness of
   575	the proposed algorithm.
   576	References
   577	[1] Zechen Bai, Zhigang Wang, Jian Wang, Di Hu, and Errui Ding. 2021. Unsupervised multi-source domain adaptation
   578	for person re-identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
   579	12914–12923.
   580	[2] Woong-Gi Chang, Tackgeun You, Seonguk Seo, Suha Kwak, and Bohyung Han. 2019. Domain-specific batch normalization for unsupervised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and
   581	Pattern Recognition, 7354–7362.
   582	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.
   583	
   584	
exec
/bin/zsh -lc "pdftotext -f 8 -l 11 'Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,460p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 12 -l 16 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '1,500p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
     2	
     3	sists of 287,628 visible and 15,792 infrared images, which are
     4	captured from 491 subjects. The training set includes 395 identities, while 96 identities are adopted for testing. Importantly,
     5	there is no overlap between the training and testing sets. The
     6	collected images are from six kinds of cameras, which include
     7	4 RGB cameras and 2 infrared cameras. Therefore, as shown
     8	in the Fig. 5 (a), there are significant variations in pedestrians’
     9	pose, viewpoint, and background. These pose considerable
    10	challenges for unsupervised clustering algorithms. Following
    11	[3], [15], the dataset can be divided into two different settings,
    12	i.e., all search and indoor search.
    13	RegDB. RegDB [42] is a relatively small-scale visibleinfrared dataset. This dataset includes 8,240 images from 412
    14	identities. Following [3], [15], a non-overlapping selection
    15	of 206 identities is employed for both training and testing
    16	purposes. Each identity consists of an equal distribution of
    17	10 visible and 10 thermal images. As displayed in Fig. 5
    18	(b), images are captured from consecutive frames, resulting
    19	in reduced variation among images of the same identity. This
    20	dataset contains ten protocols, and the final performance is
    21	taken from the overall average accuracy.
    22	Metrics. Cumulated Matching Characteristics (CMC),
    23	Mean Average Precision (mAP) and Mean Inverse Negative
    24	Penalty (MINP) [3] are employed to fairly compare the
    25	performance of our method with existing methods.
    26	
    27	8
    28	
    29	Fig. 6. The cluster numbers on the SYSU-MM01 dataset are displayed using
    30	the following experimental configurations: (a) VC, (b) VC+DNC, (c) MIE,
    31	and (d) MIE+DNC. The number of infrared clusters is represented by the
    32	red line, the number of visible clusters is depicted by the blue line, and the
    33	number of ground truth clusters is represented by the green line.
    34	
    35	B. Implementation Details
    36	The DMIC method is implemented on Pytorch platform,
    37	with two TITAN Xp GPU. We employ an effective network,
    38	i.e., AGW [3], as our backbone network. AGW is designed
    39	on ResNet-50 [43] with the differences that the first layer is
    40	replaced by two shallow modality-specific layers. Then, the
    41	backbone is initialized by the weights pretrained on ImageNet
    42	[44]. The whole training can be divided into intra-modality
    43	training for 50 epochs and inter-modality training for another
    44	50 epochs. In the training phase, without any special emphasis,
    45	the batch size is set to 64, and P and K are set to 4 and 16,
    46	respectively. Several image augment methods, such as random
    47	horizontal flipping, random erasing, and random cropping, are
    48	employed in training. The images are resized to 288 × 144.
    49	The learning rate is 3.5 × 10−4 , and it is reduced by a factor
    50	of ten every 20 epochs. During the testing phase, the features
    51	after global average pooling layer are used to calculate the
    52	cosine similarity for evaluation. DBSCAN [12] is employed
    53	to assign pseudo labels for unlabled data at the beginning of
    54	each epoch. For the SYSU-MM01 dataset, π2 , π1 , ϵ3 , ϵ2 , ϵ1 ,
    55	σn , σb , σk and k1 are set to 0.6, 0.3, 32, 18, 6, 0.5, 2, 3
    56	and 40 respectively. For the RegDB dataset, these values are
    57	configured as 0.3, 0.2, 18, 12, 6, 23 , 1.5, 2 and 38, respectively.
    58	The temperature factor τ is set to 0.05. The trade-off weights
    59	φ1 and φ2 for losses are set to 0.1 and 1.0.
    60	C. Comparison with State-of-the-art Methods
    61	In this subsection, we perform a comparative analysis, evaluating our unsupervised DMIC method against 16 state-of-theart supervised methods and 19 state-of-the-art unsupervised
    62	methods. The goal is to highlight the competitive performance
    63	
    64	Fig. 7. The visualization showcases the distribution of 10 randomly chosen
    65	identities from the SYSU-MM01 dataset. The experimental results depicted
    66	are obtained from two different configurations: (a) VC and (b) MIE+DNC.
    67	Different colors represent different identities. The ’o’ represents visible
    68	samples and the ’x’ represents infrared samples.
    69	
    70	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
    71	
    72	9
    73	
    74	Unsupervised
    75	
    76	Supervised
    77	
    78	TABLE I
    79	T HE PERFORMANCE (%) COMPARISON ON SYSU-MM01 DATASET. †INDICATES TRAINING WITH 128 BATCH SIZE .
    80	SYSU-MM01 Settings
    81	Methods
    82	Venue
    83	Zero-Padding [41]
    84	ICCV-17
    85	eBDTR [45]
    86	TIFS-19
    87	X-Modal [25]
    88	AAAI-20
    89	Hi-CMD [8]
    90	CVPR-20
    91	AGW [3]
    92	TPAMI-21
    93	DDAG [8]
    94	ECCV-20
    95	VCD+VML [50]
    96	CVPR-21
    97	CA [24]
    98	ICCV-21
    99	MPANet [24]
   100	CVPR-21
   101	MSO [51]
   102	MM-21
   103	MCLNet [23]
   104	ICCV-21
   105	SPOT [48]
   106	TIP-22
   107	DMiR [49]
   108	TCSVT-22
   109	FMCNet [52]
   110	CVPR-22
   111	MAUM [53]
   112	CVPR-22
   113	TMD [7]
   114	TMM-23
   115	
   116	Rank-1
   117	14.80
   118	27.82
   119	49.9
   120	34.9
   121	47.50
   122	54.75
   123	60.02
   124	69.88
   125	70.58
   126	58.70
   127	65.40
   128	65.34
   129	50.54
   130	66.34
   131	71.68
   132	68.81
   133	
   134	Rank-10
   135	54.12
   136	67.34
   137	89.8
   138	77.6
   139	84.39
   140	90.39
   141	94.18
   142	95.71
   143	96.21
   144	92.06
   145	93.33
   146	92.73
   147	88.12
   148	93.08
   149	
   150	All Search
   151	Rank-20
   152	71.33
   153	81.34
   154	96.0
   155	92.14
   156	95.81
   157	98.14
   158	98.46
   159	98.80
   160	97.14
   161	97.04
   162	94.84
   163	96.84
   164	
   165	mAP
   166	15.95
   167	58.42
   168	50.7
   169	35.9
   170	47.65
   171	53.02
   172	58.80
   173	66.89
   174	68.24
   175	56.42
   176	61.98
   177	62.25
   178	49.29
   179	62.51
   180	68.79
   181	63.96
   182	
   183	mINP
   184	35.30
   185	39.62
   186	53.61
   187	47.39
   188	48.86
   189	48.11
   190	
   191	Rank-1
   192	20.58
   193	32.46
   194	54.17
   195	61.02
   196	66.05
   197	76.26
   198	76.74
   199	63.09
   200	72.56
   201	69.42
   202	53.92
   203	68.15
   204	76.97
   205	76.31
   206	
   207	Indoor Search
   208	Rank-10 Rank-20
   209	68.38
   210	85.79
   211	77.42
   212	89.62
   213	91.14
   214	95.98
   215	94.06
   216	98.41
   217	96.59
   218	99.38
   219	97.88
   220	99.49
   221	98.21
   222	99.57
   223	96.61
   224	96.98
   225	99.20
   226	96.22
   227	99.12
   228	92.50
   229	97.09
   230	97.28
   231	98.91
   232	
   233	mAP
   234	26.92
   235	42.46
   236	62.97
   237	67.98
   238	72.98
   239	80.37
   240	80.95
   241	70.31
   242	76.58
   243	74.63
   244	62.49
   245	74.09
   246	81.94
   247	74.52
   248	
   249	mINP
   250	59.23
   251	62.61
   252	76.79
   253	72.10
   254	70.48
   255	65.05
   256	-
   257	
   258	SSG [58]
   259	
   260	ICCV-19
   261	
   262	2.32
   263	
   264	17.23
   265	
   266	28.88
   267	
   268	5.00
   269	
   270	-
   271	
   272	-
   273	
   274	-
   275	
   276	-
   277	
   278	-
   279	
   280	ECN [59]
   281	
   282	CVPR-19
   283	
   284	8.07
   285	
   286	32.49
   287	
   288	45.95
   289	
   290	12.68
   291	
   292	-
   293	
   294	-
   295	
   296	-
   297	
   298	-
   299	
   300	-
   301	
   302	-
   303	
   304	SPCL [29]
   305	
   306	NIPS-20
   307	
   308	18.37
   309	
   310	54.08
   311	
   312	69.02
   313	
   314	19.39
   315	
   316	10.99
   317	
   318	26.83
   319	
   320	68.31
   321	
   322	83.24
   323	
   324	36.42
   325	
   326	33.05
   327	
   328	MMT [54]
   329	CAP [55]
   330	IICS [60]
   331	
   332	ICLR-20
   333	AAAI-21
   334	CVPR-21
   335	
   336	21.47
   337	16.82
   338	14.39
   339	
   340	59.65
   341	47.6
   342	47.91
   343	
   344	73.29
   345	61.42
   346	62.32
   347	
   348	21.53
   349	15.71
   350	15.74
   351	
   352	11.50
   353	7.02
   354	8.41
   355	
   356	22.79
   357	24.57
   358	15.91
   359	
   360	63.18
   361	57.93
   362	54.20
   363	
   364	79.04
   365	72.74
   366	71.49
   367	
   368	31.50
   369	30.74
   370	24.87
   371	
   372	27.66
   373	26.15
   374	22.15
   375	
   376	ICE [35]
   377	
   378	ICCV-21
   379	
   380	20.54
   381	
   382	57.50
   383	
   384	70.89
   385	
   386	20.39
   387	
   388	10.24
   389	
   390	29.81
   391	
   392	69.41
   393	
   394	82.66
   395	
   396	38.35
   397	
   398	34.32
   399	
   400	Cluster-Contrast [30]
   401	PPLR [56]
   402	ISE [57]
   403	
   404	ACCV-22
   405	CVPR-22
   406	CVPR-22
   407	
   408	20.16
   409	12.58
   410	20.01
   411	
   412	59.27
   413	47.43
   414	57.45
   415	
   416	72.5
   417	62.69
   418	72.50
   419	
   420	22.00
   421	12.78
   422	18.93
   423	
   424	12.97
   425	4.85
   426	8.54
   427	
   428	23.33
   429	13.65
   430	14.22
   431	
   432	68.13
   433	52.66
   434	58.33
   435	
   436	82.66
   437	70.28
   438	75.32
   439	
   440	34.01
   441	22.19
   442	24.62
   443	
   444	30.88
   445	18.35
   446	21.74
   447	
   448	H2H [18]
   449	OTLA [20]
   450	ADCA [15]
   451	CHCR [17]
   452	TAA [19]
   453	PGMAL [16]
   454	CCLNet [61]
   455	GUR [37]
   456	DCCL [36]
   457	
   458	TIP-21
   459	ECCV-22
   460	MM-22

 succeeded in 0ms:
     1	Neural Networks 203 (2026) 109083
     2	
     3	Z.-H. Li et al.
     4	
     5	Table 7
     6	Comparison with the state-of-the-art methods on MSMT17. The best and
     7	second best unsupervised results are marked in bold and italic.
     8	
     9	ratio of cross/within-camera neighbors at the sample level, its eﬀectiveness is constrained by ﬁxed parameters (e.g., 𝑘𝑐1 = 25). Future work could
    10	develop an adaptive threshold mechanism to dynamically optimize the
    11	ratio between 𝑘𝑐1 and 𝑘𝑖1 during training.
    12	
    13	MSMT17
    14	
    15	Method
    16	
    17	mAP
    18	
    19	R1
    20	
    21	R5
    22	
    23	R10
    24	
    25	CRediT authorship contribution statement
    26	
    27	Supervised methods
    28	DG-Net (Zheng et al., 2019)
    29	TransReID (He et al., 2021)
    30	MCTN (Li et al., 2024)
    31	
    32	CVPR’19
    33	ICCV’21
    34	PR’24
    35	
    36	52.3
    37	69.4
    38	67.9
    39	
    40	77.2
    41	86.2
    42	85.6
    43	
    44	87.4
    45	–
    46	–
    47	
    48	90.5
    49	–
    50	–
    51	
    52	26.8
    53	35.4
    54	28.1
    55	36.5
    56	35.6
    57	
    58	53.7
    59	63.6
    60	55.5
    61	66.6
    62	60.9
    63	
    64	65.0
    65	75.5
    66	67.9
    67	75.3
    68	73.5
    69	
    70	69.8
    71	80.2
    72	–
    73	80.1
    74	78.2
    75	
    76	Zhi-Hua Li: Writing – review & editing, Writing – original draft,
    77	Visualization, Methodology, Investigation, Formal analysis; Xue-Yan
    78	Wang: Writing – review & editing, Writing – original draft, Methodology, Conceptualization; Si-Bao Chen: Writing – review & editing, Supervision, Investigation, Funding acquisition, Data curation; Chris H.Q.
    79	Ding: Writing – review & editing, Supervision, Formal analysis; Bin Luo:
    80	Writing – review & editing, Supervision, Formal analysis.
    81	
    82	Unsupervised Domain Adaptation methods
    83	SPCL (Ge et al., 2020b)
    84	IDM (Dai et al., 2021)
    85	MaskPre (Yin et al., 2022)
    86	CaCL (Lee et al., 2023)
    87	LRIMV (Li et al., 2023)
    88	
    89	NeurIPS’20
    90	ICCV’21
    91	PR’22
    92	ICCV’23
    93	TNNLS’24
    94	
    95	Declaration of competing interest
    96	
    97	Fully Unsupervised methods
    98	CAP (Wang et al., 2021)
    99	ICE (Chen et al., 2021)
   100	CCL (Dai et al., 2022)
   101	PPLR (Cho et al., 2022)
   102	ISE (Zhang et al., 2022)
   103	Lan et al. (2023)
   104	RTMem (Yin et al., 2023)
   105	DCMIP (Zou et al., 2023)
   106	AFC (Wang et al., 2024)
   107	DKD-MPL (Zhu et al., 2024)
   108	ACFL (Ji et al., 2024b)
   109	DSGL (Ji et al., 2024a)
   110	CAJ (Chen et al., 2024)
   111	CGMAL (Ran et al., 2025)
   112	DAMCL
   113	
   114	AAAI’21
   115	ICCV’21
   116	ACCV’22
   117	CVPR’22
   118	CVPR’22
   119	TIP’23
   120	TIP’23
   121	ICCV’23
   122	PR’24
   123	TMM’24
   124	PR’24
   125	TIP’24
   126	CVPR’24
   127	PR’25
   128	This paper
   129	
   130	36.9
   131	38.9
   132	31.2
   133	42.2
   134	37.0
   135	39.5
   136	38.5
   137	40.9
   138	40.7
   139	38.0
   140	45.4
   141	41.6
   142	44.3
   143	44.1
   144	60.0
   145	
   146	67.4
   147	70.2
   148	61.5
   149	73.3
   150	67.6
   151	67.9
   152	63.3
   153	69.3
   154	70.5
   155	68.1
   156	74.2
   157	70.9
   158	75.1
   159	69.7
   160	81.5
   161	
   162	78.0
   163	80.5
   164	71.8
   165	83.5
   166	77.5
   167	78.0
   168	75.4
   169	79.7
   170	81.2
   171	79.0
   172	84.1
   173	81.1
   174	84.3
   175	79.9
   176	89.9
   177	
   178	81.4
   179	84.4
   180	76.7
   181	86.5
   182	81.0
   183	81.6
   184	79.6
   185	83.6
   186	85.3
   187	81.7
   188	87.6
   189	84.6
   190	87.3
   191	83.2
   192	92.1
   193	
   194	The authors declare that they have no known competing ﬁnancial
   195	interests or personal relationships that could have appeared to inﬂuence
   196	the work reported in this paper.
   197	Acknowledgement
   198	This work was supported in part by NSFC Key Project of Joint
   199	Fund for Enterprise Innovation and Development (No. U24A20342)
   200	and National Natural Science Foundation of China (Nos. 62576006,
   201	61976004).
   202	References
   203	
   204	Table 8
   205	Comparison of computational cost and performance of diﬀerent methods on
   206	the MSMT17 dataset. All models are trained on a single RTX 3090 GPU.
   207	Method
   208	
   209	Training Time Inference Time Peak Memory mAP
   210	
   211	SpCL (Ge et al., 2020b)
   212	CCL (Dai et al., 2022)
   213	PPLR (Cho et al., 2022)
   214	RTMem (Yin et al., 2023)
   215	Lan et al. (Lan et al., 2023)
   216	DSGL (Ji et al., 2024a)
   217	CAJ (Chen et al., 2024)
   218	DAMCL
   219	
   220	4.73h
   221	4.65h
   222	6.39h
   223	7.47h
   224	8.53h
   225	9.35h
   226	8.46h
   227	13.03h
   228	
   229	6.23s
   230	6.23s
   231	6.21s
   232	6.23s
   233	6.12s
   234	6.21s
   235	6.23s
   236	6.19s
   237	
   238	16.3GB
   239	15.4GB
   240	17.3GB
   241	18.2GB
   242	16.9GB
   243	17.3GB
   244	16.8GB
   245	22.6GB
   246	
   247	Bengio, Y., Courville, A., & Vincent, P. (2013). Representation learning: A review and
   248	new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8),
   249	1798–1828.
   250	Cai, B., Wang, H., Yao, M., & Fu, X. (2025). Focus more on what? Guiding multi-task
   251	training for end-to-end person search. IEEE Transactions on Circuits and Systems for
   252	Video Technology, 35(7), 7266–7278. https://doi.org/10.1109/TCSVT.2025.3540089
   253	Chen, H., Lagadec, B., & Bremond, F. (2021). ICE: Inter-instance contrastive encoding
   254	for unsupervised person re-identiﬁcation. In Proceedings of the IEEE/CVF international
   255	conference on computer vision (pp. 14960–14969).
   256	Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In International conference on machine learning (pp. 1597–1607). PMLR.
   257	Chen, Y., Fan, Z., Chen, Z., & Zhu, Y. (2024). CA-jaccard: Camera-aware jaccard distance
   258	for person re-identiﬁcation. In Proceedings of the IEEE/CVF conference on computer vision
   259	and pattern recognition (pp. 17532–17541).
   260	Cheng, D., Zhou, J., Wang, N., & Gao, X. (2022). Hybrid dynamic contrast and probability
   261	distillation for unsupervised person Re-ID. IEEE Transactions on Image Processing, 31,
   262	3334–3346.
   263	Cho, Y., Kim, W. J., Hong, S., & Yoon, S.-E. (2022). Part-based pseudo label reﬁnement
   264	for unsupervised person re-identiﬁcation. In Proceedings of the IEEE/CVF conference on
   265	computer vision and pattern recognition (pp. 7308–7318).
   266	Dai, Y., Liu, J., Sun, Y., Tong, Z., Zhang, C., & Duan, L.-Y. (2021). IDM: An intermediate domain module for domain adaptive person Re-ID. In Proceedings of the IEEE/CVF
   267	international conference on computer vision (pp. 11864–11874).
   268	Dai, Z., Wang, G., Yuan, W., Zhu, S., & Tan, P. (2022). Cluster contrast for unsupervised
   269	person re-identiﬁcation. In Proceedings of the asian conference on computer vision (ACCV)
   270	(pp. 1142–1160).
   271	Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale
   272	hierarchical image database. In 2009 IEEE Conference on computer vision and pattern
   273	recognition (pp. 248–255). Ieee.
   274	Ester, M., Kriegel, H.-P., Sander, J., Xu, X. et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In KDD (pp. 226–231). (vol. 96).
   275	Ge, Y., Chen, D., & Li, H. (2020a). Mutual mean-teaching: Pseudo label reﬁnery for unsupervised domain adaptation on person re-identiﬁcation. arXiv:2001.01526.
   276	Ge, Y., Zhu, F., Chen, D., Zhao, R. et al. (2020b). Self-paced contrastive learning with
   277	hybrid memory for domain adaptive object Re-ID. Advances in Neural Information Processing Systems, 33, 11309–11321.
   278	Han, J., Li, Y.-L., & Wang, S. (2022a). Delving into probabilistic uncertainty for unsupervised domain adaptive person re-identiﬁcation. Proceedings of the AAAI Conference on
   279	Artiﬁcial Intelligence, 36(1), 790–798.
   280	Han, X., Yu, X., Li, G., Zhao, J., Pan, G., Ye, Q., Jiao, J., & Han, Z. (2022b). Rethinking sampling strategies for unsupervised person re-identiﬁcation. IEEE Transactions on
   281	Image Processing, 32, 29–42.
   282	He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised
   283	visual representation learning. In Proceedings of the IEEE/CVF conference on computer
   284	vision and pattern recognition (pp. 9729–9738).
   285	
   286	26.8
   287	31.2
   288	42.2
   289	38.5
   290	39.5
   291	41.6
   292	44.3
   293	60.0
   294	
   295	5. Conclusion and future work
   296	In this paper, we propose a novel dynamic adaptive multi-view contrastive learning (DAMCL) framework for unsupervised Re-ID. We improve the Jaccard distance calculated before clustering by utilizing camera information to compute a DACJ distance, which helps the clustering
   297	algorithm generate cleaner pseudo-labels. Furthermore, we introduce a
   298	DAP module to address the shortcomings of previous methods that adopt
   299	inaccurate cluster proxies. Finally, we propose a DAKD to generate soft
   300	labels that are more robust to pseudo-label noise. Extensive experiments
   301	demonstrate that our framework surpasses state-of-the-art methods on
   302	three datasets.
   303	Althongh our method has achieved amazing performance, there are
   304	still a few things that could be improved. For example, since our method
   305	employs multi-view feature extraction and utilizes four memory banks
   306	to store feature representations from multiple views, it poses signiﬁcant
   307	challenges in terms of memory consumption and computational eﬃciency. If extended to larger-scale unlabeled datasets in the future, our
   308	current experimental conﬁguration may become insuﬃcient. Therefore,
   309	designing a framework that is more eﬃcient in both time and space complexity becomes a critical task to explore. Additionally, while the current
   310	DACJ distance mitigates camera variations by dynamically adjusting the
   311	12
   312	
   313	Neural Networks 203 (2026) 109083
   314	
   315	Z.-H. Li et al.
   316	
   317	He, S., Luo, H., Wang, P., Wang, F., Li, H., & Jiang, W. (2021). TransReId: Transformerbased object re-identiﬁcation. In Proceedings of the IEEE/CVF international conference
   318	on computer vision (pp. 15013–15022).
   319	He, T., Shen, L., Guo, Y., Ding, G., & Guo, Z. (2022). Secret: Self-consistent pseudo label
   320	reﬁnement for unsupervised domain adaptive person re-identiﬁcation. In Proceedings
   321	of the AAAI conference on artiﬁcial intelligence (pp. 879–887). (vol. 36).
   322	Ji, H., Wang, L., Zhou, S., Tang, W., & Hua, G. (2024a). Disentangled sample guidance
   323	learning for unsupervised person re-identiﬁcation. IEEE Transactions on Image Processing, 33, 5144–5158.
   324	Ji, H., Wang, L., Zhou, S., Tang, W., Zheng, N., & Hua, G. (2024b). Transfer easy to hard:
   325	Adversarial contrastive feature learning for unsupervised person re-identiﬁcation. Pattern Recognition, 145, 109973.
   326	Kingma, D. P. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
   327	Lan, L., Teng, X., Zhang, J., Zhang, X., & Tao, D. (2023). Learning to puriﬁcation
   328	for unsupervised person re-identiﬁcation. IEEE Transactions on Image Processing, 32,
   329	3338–3353.
   330	Lee, G., Lee, S., Kim, D., Shin, Y., Yoon, Y., & Ham, B. (2023). Camera-driven representation learning for unsupervised domain adaptive person re-identiﬁcation. In Proceedings
   331	of the IEEE/CVF international conference on computer vision (pp. 11453–11462).
   332	Li, D., Wang, Z., Wang, J., Zhang, X., Ding, E., Wang, J., & Zhang, Z. (2022). Selfguided hard negative generation for unsupervised person re-identiﬁcation. In IJCAI
   333	(pp. 1067–1073).
   334	Li, S., Li, F., Li, J., Li, H., Zhang, B., Tao, D., & Gao, X. (2023). Logical relation inference
   335	and multiview information interaction for domain adaptation person re-identiﬁcation.
   336	IEEE Transactions on Neural Networks and Learning Systems, 35, 14770–14782.
   337	Li, Y., Miao, D., Zhang, H., Zhou, J., & Zhao, C. (2024). Multi-granularity cross transformer
   338	network for person re-identiﬁcation. Pattern Recognition, 150, 110362.
   339	Lloyd, S. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory, 28(2), 129–137.
   340	Ran, Z., Lu, X., Wei, X., & Liu, W. (2025). Camera-aware graph multi-domain adaptive
   341	learning for unsupervised person re-identiﬁcation. Pattern Recognition, 161, 111217.
   342	Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016). Performance measures
   343	and a data set for multi-target, multi-camera tracking. In European conference on computer vision (pp. 17–35). Springer.
   344	Tian, Y., Wang, Y., Krishnan, D., Tenenbaum, J. B., & Isola, P. (2020). Rethinking fewshot image classiﬁcation: A good embedding is all you need? In Computer vision–ECCV
   345	2020: 16th european conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part
   346	XIV 16 (pp. 266–282). Springer.
   347	Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine
   348	Learning Research, 9(11), 2579-2605.
   349	Wang, H., Chen, Y., Yao, M., Liu, W., Peng, J., & Fu, X. (2025). Tensor completion framework by graph reﬁnement for incomplete multi-view clustering. IEEE Transactions on
   350	Multimedia, 27, 9385–9398. https://doi.org/10.1109/TMM.2025.3613125
   351	Wang, M., Lai, B., Huang, J., Gong, X., & Hua, X.-S. (2021). Camera-aware proxies for
   352	unsupervised person re-identiﬁcation. In Proceedings of the AAAI conference on artiﬁcial
   353	intelligence (pp. 2764–2772). (vol. 35).
   354	Wang, X., & Qi, G.-J. (2022). Contrastive learning with stronger augmentations. IEEE
   355	Transactions on Pattern Analysis and Machine Intelligence, 45(5), 5549–5560.
   356	
   357	Wang, X., Sun, Z., Chehri, A., Jeon, G., & Song, Y. (2024). A novel attention-driven framework for unsupervised pedestrian re-identiﬁcation with clustering optimization. Pattern Recognition, 146, 110045.
   358	Wei, L., Zhang, S., Gao, W., & Tian, Q. (2018). Person transfer gan to bridge domain gap
   359	for person re-identiﬁcation. In Proceedings of the IEEE conference on computer vision and
   360	pattern recognition (pp. 79–88).
   361	Yao, M., Wang, H., Chen, Y., & Fu, X. (2025). Between/within view information completing for tensorial incomplete multi-view clustering. IEEE Transactions on Multimedia, 27,
   362	1538–1550. https://doi.org/10.1109/TMM.2024.3521771
   363	Ye, M., Lan, X., Leng, Q., & Shen, J. (2020). Cross-modality person re-identiﬁcation via
   364	modality-aware collaborative ensemble learning. IEEE Transactions on Image Processing,
   365	29, 9387–9399.
   366	Ye, M., Li, H., Du, B., Shen, J., Shao, L., & Hoi, S. C. H. (2021). Collaborative reﬁning
   367	for person re-identiﬁcation with label noise. IEEE Transactions on Image Processing, 31,
   368	379–391.
   369	Yin, J., Zhang, S., Xie, J., Ma, Z., & Guo, J. (2022). Unsupervised person re-identiﬁcation
   370	via simultaneous clustering and mask prediction. Pattern Recognition, 126, 108568.
   371	Yin, J., Zhang, X., Ma, Z., Guo, J., & Liu, Y. (2023). A real-time memory updating strategy
   372	for unsupervised person re-identiﬁcation. IEEE Transactions on Image Processing, 32,
   373	2309–2321.
   374	Yu, J., Peng, J., Li, K., & Wang, H. (2023). Hybrid partial-constrained learning with orthogonality regularization for unsupervised person re-identiﬁcation. Engineering Applications of Artiﬁcial Intelligence, 123, 106200.
   375	Yun, S., Park, J., Lee, K., & Shin, J. (2020). Regularizing class-wise predictions via selfknowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and
   376	pattern recognition (pp. 13876–13885).
   377	Zhang, M., Li, K., Ma, J., & Wang, X. (2024). Asymmetric double networks mutual teaching
   378	for unsupervised person re-identiﬁcation. Neural Networks, 169, 744–755.
   379	Zhang, X., Li, D., Wang, Z., Wang, J., Ding, E., Shi, J. Q., Zhang, Z., & Wang, J. (2022).
   380	Implicit sample extension for unsupervised person re-identiﬁcation. In Proceedings of
   381	the IEEE/CVF conference on computer vision and pattern recognition (pp. 7369–7378).
   382	Zheng, L., Shen, L., Tian, L., Wang, S., Wang, J., & Tian, Q. (2015). Scalable person reidentiﬁcation: A benchmark. In Proceedings of the IEEE international conference on computer vision (pp. 1116–1124).
   383	Zheng, Z., Yang, X., Yu, Z., Zheng, L., Yang, Y., & Kautz, J. (2019). Joint discriminative
   384	and generative learning for person re-identiﬁcation. In Proceedings of the IEEE/CVF
   385	conference on computer vision and pattern recognition (pp. 2138–2147).
   386	Zhong, Z., Zheng, L., Cao, D., & Li, S. (2017). Re-ranking person re-identiﬁcation with Kreciprocal encoding. In Proceedings of the IEEE conference on computer vision and pattern
   387	recognition (pp. 1318–1327).
   388	Zhu, W., Peng, B., & Yan, W. Q. (2024). Dual knowledge distillation on multiview pseudo
   389	labels for unsupervised person re-identiﬁcation. IEEE Transactions on Multimedia, 26,
   390	7359–7371.
   391	Zou, C., Chen, Z., Cui, Z., Liu, Y., & Zhang, C. (2023). Discrepant and multi-instance proxies
   392	for unsupervised person re-identiﬁcation. In Proceedings of the IEEE/CVF international
   393	conference on computer vision (pp. 11058–11068).
   394	
   395	13
   396	
   397	
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 11 -l 13 'Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
     2	
     3	11
     4	
     5	TABLE III
     6	A BLATION STUDIES FOR OBJECTIVE FUNCTIONS ON SYSU-MM01 AND R EG DB ( IN %).
     7	Components
     8	
     9	SYSU-MM01(All-search) SYSU-MM01(Indoor-search) RegDB(Visible to Infrared)
    10	
    11	Index CC-1 IC-1 CC-2 IC-2 Rank-1 Rank-10
    12	1
    13	
    14	✓
    15	
    16	2
    17	
    18	✓
    19	
    20	✓
    21	
    22	3
    23	
    24	✓
    25	
    26	✓
    27	
    28	✓
    29	
    30	4
    31	
    32	✓
    33	
    34	✓
    35	
    36	✓
    37	
    38	✓
    39	
    40	mAP
    41	
    42	Rank-1 Rank-10
    43	
    44	mAP
    45	
    46	Rank-1 Rank-10
    47	
    48	mAP
    49	
    50	32.47
    51	
    52	72.91
    53	
    54	30.6
    55	
    56	37.94
    57	
    58	79.92
    59	
    60	46.86
    61	
    62	31.17
    63	
    64	48.98
    65	
    66	32.78
    67	
    68	39.09
    69	
    70	82.21
    71	
    72	38.08
    73	
    74	47.79
    75	
    76	86.89
    77	
    78	56.14
    79	
    80	48.88
    81	
    82	65.97
    83	
    84	47.54
    85	
    86	59.94
    87	
    88	92.48
    89	
    90	57.49
    91	
    92	65.38
    93	
    94	96.54
    95	
    96	72.06
    97	
    98	85.23
    99	
   100	93.6
   101	
   102	80.92
   103	
   104	60.26
   105	
   106	92.67
   107	
   108	57.82
   109	
   110	65.83
   111	
   112	96.70
   113	
   114	72.55
   115	
   116	86.31
   117	
   118	94.18
   119	
   120	81.36
   121	
   122	TABLE IV
   123	A BLATION STUDIES FOR CLUSTERING ON SYSU-MM01 AND R EG DB ( IN %).
   124	Components
   125	
   126	SYSU-MM01(All-search) SYSU-MM01(Indoor-search) RegDB(Visible to Infrared)
   127	
   128	Index VC MIE DNC HMCL Rank-1 Rank-10
   129	1
   130	
   131	✓
   132	
   133	2
   134	
   135	✓
   136	
   137	✓
   138	
   139	3
   140	
   141	✓
   142	
   143	4
   144	
   145	✓
   146	
   147	✓
   148	
   149	mAP
   150	
   151	Rank-1 Rank-10
   152	
   153	mAP
   154	
   155	Rank-1 Rank-10
   156	
   157	mAP
   158	
   159	✓
   160	
   161	42.11
   162	
   163	82.41
   164	
   165	40.91
   166	
   167	46.09
   168	
   169	88.69
   170	
   171	55.79
   172	
   173	72.33
   174	
   175	85.34
   176	
   177	69.45
   178	
   179	✓
   180	
   181	43.44
   182	
   183	82.47
   184	
   185	43.01
   186	
   187	47.83
   188	
   189	89.44
   190	
   191	57.44
   192	
   193	78.20
   194	
   195	89.13
   196	
   197	74.15
   198	
   199	✓
   200	
   201	58.71
   202	
   203	90.21
   204	
   205	55.99
   206	
   207	62.45
   208	
   209	93.32
   210	
   211	68.95
   212	
   213	84.49
   214	
   215	92.71
   216	
   217	79.93
   218	
   219	✓
   220	
   221	60.26
   222	
   223	92.67
   224	
   225	57.82
   226	
   227	65.83
   228	
   229	96.70
   230	
   231	72.55
   232	
   233	86.31
   234	
   235	94.18
   236	
   237	81.36
   238	
   239	Query
   240	
   241	0.713
   242	
   243	0.700
   244	
   245	0.668
   246	
   247	0.639
   248	
   249	0.629
   250	
   251	0.582
   252	
   253	0.573
   254	
   255	0.495
   256	
   257	0.432
   258	
   259	0.563
   260	
   261	0.560
   262	
   263	0.557
   264	
   265	0.557
   266	
   267	0.482
   268	
   269	0.464
   270	
   271	0.451
   272	
   273	0.395
   274	
   275	0.390
   276	
   277	0.714
   278	
   279	0.518
   280	
   281	0.475
   282	
   283	0.430
   284	
   285	0.690
   286	
   287	0.586
   288	
   289	0.538
   290	
   291	0.512
   292	
   293	three major advantages: 1) Our approach effectively handles both cross-modality and cross-camera discrepancies at
   294	the clustering level, thereby bolstering the accuracy of label
   295	estimation and the robustness of the optimization objective.
   296	2) Adaptation of clustering objectives dynamically steers the
   297	network’s learning process from simplicity to complexity.
   298	Prioritizing reliable samples initially amplifies discriminative
   299	capabilities, while subsequently integrating cross-camera and
   300	cross-modal samples enhances generalization skills. Notably,
   301	our clustering method can be easily integrated into any
   302	unsupervised framework. 3) Cluster-level and instance-level
   303	objective functions can facilitate network learning to modalityinvariant feature representations. Overall, our method enhances
   304	the performance of USL-VI-ReID, rendering it a promising
   305	solution for real-world applications.
   306	
   307	Infrared to visible
   308	
   309	Visible to infrared
   310	Query
   311	
   312	0.512
   313	
   314	0.509
   315	
   316	0.501
   317	
   318	0.475
   319	
   320	0.474
   321	
   322	0.420
   323	
   324	0.573
   325	
   326	0.540
   327	
   328	0.540
   329	
   330	0.540
   331	
   332	0.536
   333	
   334	0.553
   335	
   336	0.563
   337	
   338	0.560
   339	
   340	0.557
   341	
   342	0.557
   343	
   344	0.553
   345	
   346	0.450
   347	
   348	0.449
   349	
   350	0.439
   351	
   352	0.432
   353	
   354	0.423
   355	
   356	0.389
   357	
   358	0.549
   359	
   360	0.523
   361	
   362	0.519
   363	
   364	0.519
   365	
   366	0.515
   367	
   368	0.434
   369	
   370	0.574
   371	
   372	0.570
   373	
   374	0.566
   375	
   376	0.558
   377	
   378	0.556
   379	
   380	(a)RegDB
   381	D. Various Properties Analysis of the DMIC Model
   382	In this subsection, we verify the effectiveness of each key
   383	module in our DMIC method on SYSU-MM01 and RegDB
   384	datasets. Ablation studies on objective functions and clustering
   385	strategy are conducted to exhaustively analyze the performance
   386	of the network.
   387	Ablation study for objective functions. As listed in Table
   388	III, we analyze the contribution of each objective function.
   389	CC-1 and IC-1 represent the cluster-level and instance-level
   390	losses during intra-modality training, while CC-2 and IC-2
   391	refer to the cluster-level and instance-level losses during intermodality training. When gradually superimposing the objective
   392	function IC-1, CC-2 and IC-2 on CC-1, the performance of the
   393	model can be effectively improved. The largest improvement
   394	lies in the introduction of CC-2, which nearly doubles the
   395	model performance on the Regdb dataset. CC-2 utilizes robust
   396	pseudo-labels assigned by the MIE and DNC, which are
   397	modality and camera-independent, to facilitate the learning of
   398	modality-camera invariant representations. Additionally, IC-1
   399	
   400	Query
   401	
   402	Query
   403	
   404	(b)SYSU-MM01
   405	Fig. 8. The top-5 retrieved results on (a) RegDB and (b) SYSU-MM01
   406	dataset are presented. Correct matchings are identified by green bounding
   407	boxes, wrong matchings by red bounding boxes, and query samples by yellow
   408	bounding boxes. The cosine similarity values are shown above the gallery
   409	images.
   410	
   411	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
   412	
   413	and IC-2 effectively refine the relationships between instances,
   414	which leads to performance improvements.
   415	Ablation study for clustering. Clustering is widely recognized as a critical component in unsupervised learning
   416	frameworks. As shown in Table IV, we analyze the effectiveness of MIE and DNC. VC refers to the vanilla clustering
   417	method employed in recent studies [15], [16], [19]. Results
   418	from experiment index 3 reveal a substantial improvement of
   419	approximately 10%-15% in mAP with the proposed MIE. By
   420	incorporating our MIE module, our unsupervised framework
   421	is able to generate more robust pseudo labels, leading to a
   422	significant enhancement in performance. From index 2 and 4,
   423	we can oberve that DNC can boost the performance of 2%5% in mAP. The dynamic strategy of DNC enables further
   424	refinement of the network’s optimization objective through
   425	clustering. By progressively transitioning from hard to easy
   426	objectives, we are able to systematically elevate the upper
   427	bound of the model’s performance. Overall, the enhancements
   428	in clustering achieved by our approach contribute to a significant boost in performance, with improvements ranging from
   429	14%-17% in mAP. This clearly illustrates the effectiveness of
   430	our association approach in handling cross-modal and crosscamera samples, which in turn offers fresh insights into the
   431	USL-VI-ReID task.
   432	Visualization Analysis. To further substantiate the efficacy of our method, we perform visualization experiments.
   433	Fig. 6 illustrates the cluster number evolution during training.
   434	Fig. 7 shows the distribution [62] of 10 randomly selected
   435	challenging identities from the SYSU-MM01 dataset. As
   436	displayed in Fig. 6, the cluster number of visible samples
   437	is prone to suffer from identity splitting problem. This is
   438	primarily attributed to the variations in color and lighting
   439	present in visible images, caused by disparities in RGB camera
   440	characteristics. The number of visible clusters in the VC
   441	method decreases to approximately 600 during intra-modality
   442	and inter-modality training. Introducing the MIE module helps
   443	alleviate the identity splitting problem, resulting in a combined
   444	number of visible clusters of around 550. However, the lack of
   445	refined clustering objectives limits the cross-modal and crosscamera generalizability, making cluster merging difficult. Fig.
   446	6 (d) illustrates that the collaboration between MIE and
   447	DNC is effective. This collaboration enhances the cross-modal
   448	and cross-camera generalizability of the model. It brings the
   449	number of visible and infrared clusters closer to the number
   450	of ground truth clusters. To be specific, we narrow down
   451	the eps from π2 to π1 to prioritize discriminability learning
   452	from reliable clusters in the first 50 intra-modality learning
   453	epochs. This process is repeated for an additional 10 intermodality learning epochs to further split solid single-camera
   454	clusters. During the last 40 inter-modality learning epochs,
   455	we widen eps from π1 to π2 to merge cross-modal and
   456	cross-camera instances into clusters, thereby facilitating the
   457	overall cross-modal and cross-camera generalizability of the
   458	model. As seen in Fig. 7 (a), we can observe that using VC
   459	results in a substantial separation of identities within nearly
   460	all clusters. This separation hampers the model’s classification
   461	capabilities and introduces noise, which adversely affect the
   462	model’s performance. However, by incorporating MIE and
   463	
   464	12
   465	
   466	DNC (see Fig. 7 (b)), we can learn compact representations
   467	for each cluster.
   468	Retrieved Results. As illustrated in Fig. 8, we display
   469	retrieved results on SYSU-MM01 and RegDB datasets under
   470	infrared to visible and visible to infrared modes. The results
   471	demonstrate that DMIC is effective in learning modalitycamera invariant features. Even in the presence of challenging
   472	and confusing samples, it is still possible to maintain a
   473	relatively high similarity among positive samples.
   474	V. C ONCLUSION
   475	This paper presents a novel Dynamic Modality-Camera
   476	Invariant Clustering (DMIC) method for USL-VI-ReID task.
   477	Specifically, Modality-Camera Invariant Expansion (MIE) integrates inter-modal and inter-camera distance coding, which
   478	effectively bridges cross-modality and cross-camera gaps at
   479	clustering-level. Dynamic Neighborhood Clustering (DNC)
   480	employs two dynamic search strategies. The dynamic search
   481	strategies help to refine the network’s optimization objective
   482	through clustering, from discriminability ability enhancement
   483	to cross-modal and cross-camera generalizability learning.
   484	Moreover, a Hybrid Modality Contrastive Learning (HMCL)
   485	is designed to optimize instance-level and cluster-level distributions. The memories for intra-modality and inter-modality
   486	training are updated by randomly selected samples, which help
   487	to fully explore modality-invariant representations in a realtime manner. Extensive experiments have demonstrated that
   488	our DMIC addresses the limitations present in current clustering approaches and significantly perform better than current
   489	state-of-the-art unsupervised methods. These findings provide
   490	valuable insights for the USL-VI-ReID task and further reduce
   491	the performance gap with supervised methods.
   492	R EFERENCES
   493	[1] Y. Ge, F. Zhu, D. Chen, R. Zhao, X. Wang and H. Li, “Structured
   494	Domain Adaptation With Online Relation Regularization for Unsupervised Person Re-ID,” in IEEE Trans. NNLS, vol. 35, no. 1, pp. 258–271,
   495	2024.
   496	[2] Q. Wu, J. Li, P. Dai, Q. Ye, L. Cao, Y. Wu and R. Ji, “Unsupervised
   497	Domain Adaptation on Person Reidentification via Dual-Level Asymmetric Mutual Learning,” in IEEE Trans. NNLS, 2023, Early Access.
   498	[3] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C. Hoi, “Deep
   499	learning for person re-identification: A survey and outlook,” IEEE
   500	Trans. PAMI, vol. 44, no. 6, pp. 2872–2893, 2021.
   501	[4] Y.-C. Chen, X. Zhu, W.-S. Zheng, and J.-H. Lai, “Person reidentification by camera correlation aware feature augmentation,” IEEE
   502	Trans. PAMI, vol. 40, no. 2, pp. 392–408, 2017.
   503	[5] H.-X. Yu, A. Wu, and W.-S. Zheng, “Unsupervised person reidentification by deep asymmetric metric embedding,” IEEE Trans.
   504	PAMI, vol. 42, no. 4, pp. 956–973, 2018.
   505	[6] M. Ye, X. Lan, Q. Leng, and J. Shen, “Cross-modality person reidentification via modality-aware collaborative ensemble learning,”
   506	IEEE Trans. IP, vol. 29, pp. 9387–9399, 2020.
   507	[7] Z. Lu, R. Lin, and H. Hu, “Tri-level modality-information disentanglement for visible-infrared person re-identification,” IEEE Trans. MM,
   508	2023, Early Access.
   509	[8] S. Choi, S. Lee, Y. Kim, T. Kim, and C. Kim, “Hi-cmd: Hierarchical cross-modality disentanglement for visible-infrared person reidentification,” in CVPR, 2020, pp. 10 257–10 266.
   510	[9] Z. Wang, Z. Wang, Y. Zheng, Y.-Y. Chuang, and S. Satoh, “Learning to reduce dual-level discrepancy for infrared-visible person reidentification,” in CVPR, 2019, pp. 618–626.
   511	[10] H. Fan, L. Zheng, C. Yan and Y. Yang, “Unsupervised person reidentification: Clustering and fine-tuning,” ACM Trans. OMM, vol. 14,
   512	no. 4, pp. 1–18, 2018.
   513	
   514	JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020
   515	
   516	[11] Z. He, M. Xue, Y. Du, Z. Zhao and F. Su, “Dynamic Clustering
   517	and Cluster Contrastive Learning for Unsupervised Person Re-Id With
   518	Feature Distribution Alignment,” ICASSP, 2024, pp. 3610–3614.
   519	[12] M. Ester, H. Kriegel, et al, ”A density-based algorithm for discovering
   520	clusters in large spatial databases with noise,” in KDD, 1996, pp. 226231.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 12 -l 16 'Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '1,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	325:12
     2	
     3	L. Dai et al.
     4	
     5	Table 1. Comparison of Our Method with the State-of-the-Art VI-ReID Methods on RegDB and
     6	SYSU-MM01; It contains SL, SSL, and USL VI-ReID Methods
     7	Settings
     8	
     9	RegDB
    10	
    11	SYSU-MM01
    12	
    13	Visible to Infrared
    14	Infrared to Visible
    15	All-Search
    16	Indoor-Search
    17	Type
    18	Method
    19	Venue Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP
    20	Zero-Pad [52]
    21	ICCV-17 17.75 18.90
    22	16.63 17.82
    23	14.80 15.95
    24	20.58 26.92
    25	AlignGAN [48]
    26	ICCV-19
    27	57.9
    28	53.6
    29	56.3
    30	53.4
    31	42.4
    32	40.7
    33	45.9
    34	54.3
    35	DDAG [64]
    36	ECCV-20 69.34 63.46 49.24 68.06 61.80 48.62 54.75 53.02 39.62 61.02 67.98 62.61
    37	AGW [65]
    38	TPAMI-21 70.05 66.37 50.19 70.49 65.90 51.24 47.50 47.65 35.30 54.17 62.97 59.23
    39	CA [63]
    40	ICCV-21 85.03 79.14 65.33 84.75 77.82 61.56 69.88 66.89 53.61 76.26 80.37 76.79
    41	SL
    42	FMCNet [69]
    43	CVPR-22 89.12 84.43
    44	88.38 83.86
    45	66.34 52.51
    46	68.15 74.09
    47	MPANet [53]
    48	CVPR-21
    49	82.8
    50	80.7
    51	83.7
    52	80.9
    53	70.58 68.24
    54	76.74 80.96
    55	MAUM [33]
    56	CVPR-22 87.87 85.09
    57	86.95 84.83
    58	71.68 68.79
    59	76.97 81.94
    60	PartMix [24]
    61	CVPR-23 84.93 82.52
    62	85.66 82.27
    63	77.78 74.62
    64	81.52 84.38
    65	DEEN [71]
    66	CVPR-23
    67	91.1
    68	85.1
    69	89.5
    70	83.4
    71	74.7
    72	71.8
    73	80.3
    74	83.3
    75	MAUM-50 [33]
    76	CVPR-22
    77	28.8
    78	36.1
    79	MAUM-100 [33]
    80	CVPR-22
    81	38.5
    82	39.2
    83	SSL
    84	OTLA [50]
    85	ECCV-22
    86	32.9
    87	29.7
    88	32.1
    89	28.6
    90	29.98 27.13
    91	29.8
    92	38.8
    93	TAA [56]
    94	TIP-23
    95	62.23 56.00 41.51 63.79 56.53 38.99 49.77 42.43 25.37 50.12 56.02 49.96
    96	DPIS [41]
    97	ICCV-23
    98	62.3
    99	53.2
   100	61.5
   101	52.7
   102	58.4
   103	55.6
   104	63.0
   105	70.0
   106	Cluster Contrast [8] ACCV-22 11.76 13.88 9.94 11.14 12.99 8.99 20.16 22.00 12.97 23.33 34.01 30.88
   107	ICE [1]
   108	ICCV-21 12.98 15.64 11.91 12.18 14.82 10.6 20.54 20.39 10.24 29.81 38.35 34.32
   109	H2H [28]
   110	TIP-21
   111	23.81 18.87
   112	30.15 29.40
   113	OTLA [50]
   114	ECCV-22
   115	32.9
   116	29.7
   117	32.1
   118	28.6
   119	29.98 27.13
   120	29.8
   121	38.8
   122	ADCA [59]
   123	MM-22
   124	67.20 64.05 52.67 68.48 63.81 49.62 45.51 42.73 28.29 50.60 59.11 55.17
   125	USL
   126	DFC [42]
   127	IPM-23
   128	38.88 38.11
   129	40.92 36.20
   130	44.12 28.36
   131	CHCR [37]
   132	TCSVT-23 68.18 63.75
   133	69.08 63.95
   134	47.72 45.34
   135	DOTLA [5]
   136	MM-23
   137	85.63 76.71 61.58 82.91 74.97 58.60 50.36 47.36 32.40 53.47 61.73 57.35
   138	MBCCM [4]
   139	MM-23
   140	83.79 77.87 65.04 82.82 76.74 61.73 53.14 48.16 32.41 55.21 61.98 57.13
   141	PGM [55]
   142	CVPR-23 69.48 65.41
   143	69.85 65.17
   144	57.27 51.78 34.96 56.23 62.74 58.13
   145	Ours
   146	89.68 84.32 72.79 89.04 83.38 70.56 53.89 51.58 37.93 58.25 65.11 60.85
   147	
   148	We report the Rank at r accuracy (%), mAP (%), and mINP (%). The best results are denoted in bold.
   149	
   150	Table 2. Comparison of Our Method with the State-of-the-Art VI-ReID Methods on LLCM
   151	Settings
   152	
   153	LLCM
   154	
   155	Infrared to Visible
   156	Visible to Infrared
   157	Type
   158	Method
   159	Venue
   160	Rank-1 Rank-10 Rank-20 mAP mINP Rank-1 Rank-10 Rank-20 mAP mINP
   161	DDAG [64]
   162	ECCV-20
   163	42.36
   164	72.69
   165	80.63
   166	48.97
   167	51.42
   168	81.45
   169	88.26
   170	38.77
   171	AGW [65]
   172	TPAMI-21 49.13
   173	79.06
   174	85.89
   175	55.80
   176	63.72
   177	88.66
   178	92.83
   179	47.21
   180	CA [63]
   181	ICCV-21
   182	49.86
   183	78.91
   184	85.83
   185	56.40
   186	63.73
   187	87.95
   188	92.41
   189	47.71
   190	SL
   191	MMN [73]
   192	MM-21
   193	50.14
   194	79.81
   195	87.27
   196	56.66
   197	63.97
   198	88.66
   199	93.05
   200	48.47
   201	MRCN [72]
   202	AAAI-23
   203	51.32
   204	80.10
   205	87.17
   206	57.74
   207	65.27
   208	88.11
   209	93.13
   210	49.45
   211	DART [60]
   212	CVPR-22
   213	52.97
   214	80.82
   215	87.05
   216	59.28
   217	65.33
   218	89.42
   219	93.33
   220	51.13
   221	DEEN [71]
   222	CVPR-23
   223	55.52
   224	83.88
   225	89.98
   226	62.07
   227	69.21
   228	90.95
   229	95.07
   230	55.52
   231	Cluster Contrast† [8] ACCV-22
   232	15.74
   233	41.83
   234	53.10
   235	21.81 19.43 17.98
   236	44.86
   237	55.83
   238	14.92 5.40
   239	OTLA† [50]
   240	ECCV-22
   241	26.45
   242	53.29
   243	63.10
   244	31.47 27.79 35.51
   245	63.83
   246	73.55
   247	22.15 4.63
   248	USL
   249	ADCA† [59]
   250	MM-22
   251	36.39
   252	66.44
   253	75.47
   254	42.73 39.17 32.52
   255	64.07
   256	73.73
   257	25.54 8.01
   258	PGM† [55]
   259	CVPR-23
   260	39.05
   261	67.38
   262	75.99
   263	44.72 40.85 36.07
   264	66.01
   265	75.19
   266	26.65 8.80
   267	Ours
   268	40.71
   269	69.45
   270	77.89 46.82 43.07 38.55
   271	67.21
   272	76.28 28.28 8.81
   273	
   274	The results of the SL methods are from DEEN [54] and we reproduced representative USL methods for comparison. The
   275	dagger symbols (†) denote the reproduced results of the methods. The best results are presented in bold and the hyphen
   276	(-) denotes that the results are not provided in the original papers.
   277	
   278	composition is more complex, presenting greater challenges, which results in generally lower
   279	model performance compared to other datasets. Nevertheless, our approach outperforms most
   280	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   281	
   282	DLLR for Unsupervised Visible-Infrared Person ReID
   283	
   284	325:13
   285	
   286	Table 3. Results of Ablation Study on SYSU-MM01 and RegDB
   287	Index
   288	1
   289	2
   290	3
   291	4
   292	5
   293	
   294	SYSU-MM01
   295	RegDB
   296	All-Search
   297	Indoor-Search
   298	Visible to Infrared
   299	Baseline CSM CRLR MM WMM Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP
   300	3
   301	37.58 36.66 23.53 44.93 52.65 47.91 44.38 42.99 32.74
   302	3
   303	3
   304	3
   305	48.09 47.07 33.97 50.26 59.31 55.43 87.39 82.65 71.93
   306	3
   307	3
   308	3
   309	3
   310	50.10 48.40 35.34 52.43 60.40 56.09 89.26 84.19 72.68
   311	3
   312	3
   313	3
   314	52.29 50.20 36.65 56.52 63.73 59.43 88.16 83.28 72.32
   315	3
   316	3
   317	3
   318	3
   319	53.89 51.58 37.93 58.25 65.11 60.85 89.68 84.32 72.79
   320	Approach
   321	
   322	“CSM” denotes the cluster similarity matching module for matching cross-modality samples. “CRLR” means the
   323	cluster relationship-based label refinement module. “WMM” represents the weighted modality-shared memory
   324	for dual-modality-shared learning. Rank at r accuracy (%), mAP (%), and mINP (%) are reported. The best results
   325	are denoted in bold.
   326	
   327	methods, achieving 46.82% mAP (Infrared to Visible) and 28.28% mAP (Visible to Infrared), further
   328	demonstrating the robustness and generalization capability of our framework.
   329	Furthermore, our method significantly outperforms existing unsupervised single-modality methods in terms of mAP, highlighting the critical role of cross-modality learning in USVI-ReID. Compared with H2H [28] and DOTLA [5], which necessitate pretraining on the annotated Market-1501
   330	[75] dataset, our method operates independently of any annotations, including camera labels.
   331	Comparison with Supervised Methods. We compare our proposed method with supervised VI-ReID
   332	methods. On SYSU-MM01, our approach achieves a Rank-1 accuracy of 53.89% for the All-Search
   333	mode and 58.25% for the Indoor-Search, demonstrating competitive performances to DDAG [64].
   334	Moreover, our method surpasses many supervised VI-ReID methods, including Zero-Pad [52],
   335	AlignGAN [48], and the strong baseline AGW [65]. Additionally, on RegDB (Visible to Infrared), our
   336	proposed method outperforms most supervised methods, achieving a Rank-1 accuracy of 89.68%.
   337	These results underscore the superiority of our method in the context of VI-ReID.
   338	Comparison with Semi-Supervised Methods. To further validate the effectiveness of our proposed
   339	method, we compared it with several SSL-VI-ReID methods on SYSU-MM01 and RegDB. Our method
   340	achieves an mAP that surpasses the state-of-the-art method DPIS [41] by 31.12% on RegDB (Visible
   341	to Infrared) and exceeds TAA [56] by 9.15% in terms of mAP on SYSU-MM01 (All-Search). The
   342	high-quality pseudo labels generated by our method enable the framework to attain competitive
   343	performance with SSL-VI-ReID methods, despite the availability of ground-truth labels for visible
   344	samples in SSL.
   345	4.4
   346	
   347	Ablation Study
   348	
   349	Our framework consists of three primary modules: CSM (Section 3.2), CRLR (Section 3.3), and
   350	MM, which includes the improved WMM (Section 3.4). The results of the ablation experiments
   351	conducted on SYSU-MM01 and RegDB are reported in Table 3.
   352	Baseline. The baseline model is trained by contrastive learning across two modalities independently, which also serves as the initial training phase (Section 3). As shown in 1st row, the baseline
   353	achieves 36.66% mAP on SYSU-MM01 (All-Search) and 42.99% mAP on RegDB (Visible to Infrared).
   354	These results indicate that solely relying on cluster contrastive learning within each modality is
   355	insufficient to bridge the gap between infrared and visible modalities.
   356	Effectiveness of CSM and MM. Since CSM and MM are interrelated modules, we present their
   357	combined effects in 2nd row. Comparing the results from 1st row and 2nd row, using both CSM and
   358	MM leads to a +10.51%/+10.41% and +43.01%/+39.66% Rank-1/mAP improvement on SYSU-MM01
   359	(All-Search) and RegDB (Visible to Infrared), respectively. With CSM, clusters from both modalities
   360	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   361	
   362	325:14
   363	
   364	L. Dai et al.
   365	Table 4. Effect of CRLR on Improving the Quality of Modality-Shared Labels
   366	(Visible to Infrared) and Visible Modality-Shared Labels (Infrared to Visible)
   367	Method
   368	Baseline
   369	Ours (training w/o CRLR)
   370	Ours (training w/ CRLR)
   371	
   372	Visible to Infrared
   373	Before CRLR After CRLR
   374	0.4567
   375	0.5076
   376	0.5984
   377	0.6138
   378	0.6385
   379	0.6527
   380	
   381	Infrared to Visible
   382	Before CRLR After CRLR
   383	0.5036
   384	0.5641
   385	0.6256
   386	0.6475
   387	0.6734
   388	0.7012
   389	
   390	Quality is measured on SYSU-MM01 by 𝑅𝑝𝑙𝑞 . For convenience, we remove the outlier
   391	samples while measuring. The best results are denoted in bold.
   392	
   393	are matched to generate modality-shared labels, facilitating contrastive learning. Meanwhile, MM
   394	effectively narrows the modality gap and extracts identity-related features. In contrastive learning
   395	with MM, the cross-modality differences are reduced through the cross-modality association
   396	provided by CSM. Consequently, the results on datasets are greatly improved. However, CSM
   397	will inevitably introduce some noise labels and correspondences, and MM fails to construct an
   398	appropriate memory bank. As a result, the efficiency of contrastive learning still requires further
   399	enhancement.
   400	Effectiveness of WMM. The results in 2nd row and 4th row show the effectiveness of WMM, which
   401	improves the performance of 4.20%/3.13% Rank-1/mAP on SYSU-MM01 (All-Search). Compared
   402	with MM, WMM constructs superior memory banks for clusters, allowing each sample to be learned
   403	differently based on its feature distribution. The weighted strategy employed in constructing the
   404	memory bank is more advantageous for cross-modality learning than simple averaging, as it
   405	enables samples, particularly hard ones, to engage more effectively in the cross-modality learning
   406	process [21]. On the other hand, we observe that WMM provides a more pronounced improvement
   407	on SYSU-MM01 compared to RegDB. This is because of the more complex shooting scenes in
   408	SYSU-MM01, which contain more difficult samples. Our WMM then initializes a memory bank
   409	which pays more attention to these samples, resulting in more enhancements on SYSU-MM01 than
   410	on RegDB.
   411	Effectiveness of CRLR. The results of using CRLR are in 3rd row and 5th row. Compared to
   412	CSM+WMM, CRLR achieves improvements of 1.60%/1.38% Rank-1/mAP on SYSU-MM01 (AllSearch). This enhancement can be attributed to CRLR’s ability to refine labels and mitigate the
   413	influence of noisy modality-shared labels, thereby enhancing the effectiveness of dual-modalityshared learning. To further illustrate the effect of CRLR in improving the quality of modality-shared
   414	labels, we present the 𝑅𝑝𝑙𝑞 [50] of modality-shared labels in Table 4. A higher 𝑅𝑝𝑙𝑞 indicates a
   415	higher quality of pseudo labels. By comparing label quality across modalities, we can assess both
   416	the quality of the modality-specific labels and the cross-modality correspondences. We present the
   417	label quality produced by the trained baseline, Ours (training w/o CRLR) and Ours (training w/
   418	CRLR), corresponding to 1st, 4th, and 5th rows in Table 3, respectively. Our method demonstrates the
   419	highest 𝑅𝑝𝑙𝑞 when CRLR is employed for training. Additionally, we compare the 𝑅𝑝𝑙𝑞 of modalityshared labels before CRLR and after CRLR, which indicate the pseudo labels before and after applying
   420	CRLR for label refinement, respectively. The results indicate that CRLR enhances the accuracy of
   421	cross-modality correspondences, even when the model is not explicitly trained with CRLR.
   422	4.5
   423	
   424	Parameters Analysis
   425	
   426	We analyze the sensitivity of key parameters of the framework: 𝛾 in Equation (8) and similarity
   427	threshold 𝜖 in CRLR. In the experiments, we tune the value of each parameter while fixing other
   428	parameters. The results are shown in Figure 4.
   429	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   430	
   431	DLLR for Unsupervised Visible-Infrared Person ReID
   432	
   433	325:15
   434	
   435	Fig. 4. The results of using different values of 𝛾 and 𝜖 on SYSU-MM01 and RegDB.
   436	
   437	Our method performs best at 𝛾 = 0.1. As 𝛾 increases, the weighted strategy becomes less
   438	influential in constructing the memory banks, leading to a degradation of 2.51% and 0.79% in
   439	Rank-1 accuracy on the SYSU-MM01 (All-Search) and RegDB (Visible to Infrared), respectively.
   440	This indicates the importance of assigning different weights to samples for hard sample mining
   441	in memory construction. Additionally, we find that the performance of our proposed method on
   442	SYSU-MM01 is more sensitive to 𝛾, with a performance fluctuation of 4.38% Rank-1, while on
   443	RegDB, the fluctuation is only 1.03%. Since 𝛾 determines how much our weighting strategy affects
   444	memory initialization, it plays a crucial role in addressing hard samples, which is the primary focus
   445	of WMM. As discussed in Section 4.4, the shooting scenes of SYSU-MM01 is more complex, resulting
   446	in more hard samples compared to RegDB. For these reasons, we set 𝛾 = 0.1 in our model to ensure
   447	a balanced initialization of memory banks, adequately representing identities while effectively
   448	handling hard samples.
   449	The parameter 𝜖 is crucial in determining whether two clusters belong to the same class. A large
   450	𝜖 causes CRLR to merge almost no clusters, and the cluster with a single identity will be divided.
   451	A small 𝜖 causes CRLR to merge clusters with different identities. Compared with SYSU-MM01, the
   452	clustering results of RegDB are better, and the variation of cluster similarity is smaller. Therefore,
   453	we set 𝜖 = 0.6 for SYSU-MM01 and set 𝜖 = 0.7 for RegDB.
   454	4.6
   455	
   456	t-distributed Stochastic Neighbor Embedding (t-SNE) Visualization
   457	
   458	As illustrated in Figure 5, we present the [47] visualization of 10 randomly selected identities from
   459	the SYSU-MM01 dataset. The baseline method demonstrates an initial capability to cluster samples
   460	of the same identity within a single modality. However, it struggles to reduce modality discrepancies,
   461	leading to the incorrect clustering of samples from different identities across modalities. We devise
   462	the CSM and MM modules to bridge the gap between features of the same identity. CSM and MM
   463	modules encourage cross-modality learning, but the boundaries of clusters are not clear due to the
   464	noise labels and correspondences. The designed CRLR refine the pseudo labels, thereby effectively
   465	clarifying these boundaries and further reducing the distances between features. Nonetheless, there
   466	are still hard samples being incorrectly matched. To further enhance the ability of the model to
   467	recognize hard samples, WMM is introduced which expands inter-class distances and improves
   468	intra-class compactness.
   469	4.7
   470	
   471	Modality Gap Visualization
   472	
   473	To analyze the modality gap between representations of identities, we visualize the distribution of
   474	Euclidean distance between 200,000 randomly selected positive sample pairs across modalities in
   475	Figure 6(a). As we progressively incorporate our designed modules, the distribution of distances
   476	for positive pairs shifts to the left, indicating an increased proportion of closer representations for
   477	the same identities. This demonstrates that our proposed modules effectively enhance the model’s
   478	ability to narrow the modality gap. Furthermore, we compare the distances between positive and
   479	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   480	
   481	325:16
   482	
   483	L. Dai et al.
   484	
   485	Fig. 5. The t-SNE [47] visualization of representations for 10 randomly selected identities. (a): Baseline.
   486	(b): Baseline+CSM+MM. (c): Baseline+CSM+CRLR+MM. (d): Baseline+CSM+CRLR+WMM. “” represents
   487	the infrared instances and “◦” represents the visible instances. Different identities are represented in different
   488	colors.
   489	
   490	Fig. 6. The distribution of the distance between 20,000 randomly selected pairs from SYSU-MM01 which are
   491	positive cross-modality. We present the results of experiments in which the proposed modules are gradually
   492	added to the framework in (a). The distribution of the distance between positive pairs and negative pairs of
   493	the baseline model and the proposed DLLR is compared in (b) and (c).
   494	
   495	negative pairs across modalities for both the baseline method and our proposed approach, as
   496	shown in Figure 6(b) and (c). Although the baseline method can roughly distinguish cross-modality
   497	positive and negative samples based on distance, there remains a substantial overlap in the distance
   498	distributions of these two categories. Therefore, a considerable number of cross-modality positive
   499	samples tend to be misclassified as negative samples. In contrast, our method significantly reduces
   500	this overlapping region, clarifying the distance boundary between positive and negative samples
   501	across modalities. This improvement illustrates the effectiveness of our approach in mitigating
   502	modality discrepancies.
   503	4.8
   504	
   505	Pseudo Classes Analysis
   506	
   507	The clustering algorithm is widely used in USVI-ReID. The quality of clustering has a great influence
   508	on the subsequent learning of the model. To further investigate the role of the proposed CRLR, we
   509	present the change in the number of pseudo classes during training in Figure 7.
   510	As illustrated in Figure 7(a) and (b), the number of pseudo classes gradually approaches the real
   511	number of classes during the training of our proposed method. Additionally, we draw Figure 7(c)
   512	and (d) to show the number of clusters added and merged by CRLR, respectively. As training
   513	progresses, the distance between the features becomes clearer and the clustering results become
   514	closer to the real, CRLR has less impact on split clusters. Meanwhile, it shows greater change
   515	of clusters compared to the separated. This is due to problem of many-to-one cross-modality
   516	matching caused by modality gap. In CRLR, we transfer the many-to-one into a one-to-one match
   517	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   518	
   519	
codex
第一篇 DLLR 后面的实验部分很有用：它没有只报 mAP，而是量化伪标签质量、看正负跨模态距离分布、看伪类数量变化。这正是“重定义是否正确”的证据链。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 16 -l 18 'Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	325:16
     2	
     3	L. Dai et al.
     4	
     5	Fig. 5. The t-SNE [47] visualization of representations for 10 randomly selected identities. (a): Baseline.
     6	(b): Baseline+CSM+MM. (c): Baseline+CSM+CRLR+MM. (d): Baseline+CSM+CRLR+WMM. “” represents
     7	the infrared instances and “◦” represents the visible instances. Different identities are represented in different
     8	colors.
     9	
    10	Fig. 6. The distribution of the distance between 20,000 randomly selected pairs from SYSU-MM01 which are
    11	positive cross-modality. We present the results of experiments in which the proposed modules are gradually
    12	added to the framework in (a). The distribution of the distance between positive pairs and negative pairs of
    13	the baseline model and the proposed DLLR is compared in (b) and (c).
    14	
    15	negative pairs across modalities for both the baseline method and our proposed approach, as
    16	shown in Figure 6(b) and (c). Although the baseline method can roughly distinguish cross-modality
    17	positive and negative samples based on distance, there remains a substantial overlap in the distance
    18	distributions of these two categories. Therefore, a considerable number of cross-modality positive
    19	samples tend to be misclassified as negative samples. In contrast, our method significantly reduces
    20	this overlapping region, clarifying the distance boundary between positive and negative samples
    21	across modalities. This improvement illustrates the effectiveness of our approach in mitigating
    22	modality discrepancies.
    23	4.8
    24	
    25	Pseudo Classes Analysis
    26	
    27	The clustering algorithm is widely used in USVI-ReID. The quality of clustering has a great influence
    28	on the subsequent learning of the model. To further investigate the role of the proposed CRLR, we
    29	present the change in the number of pseudo classes during training in Figure 7.
    30	As illustrated in Figure 7(a) and (b), the number of pseudo classes gradually approaches the real
    31	number of classes during the training of our proposed method. Additionally, we draw Figure 7(c)
    32	and (d) to show the number of clusters added and merged by CRLR, respectively. As training
    33	progresses, the distance between the features becomes clearer and the clustering results become
    34	closer to the real, CRLR has less impact on split clusters. Meanwhile, it shows greater change
    35	of clusters compared to the separated. This is due to problem of many-to-one cross-modality
    36	matching caused by modality gap. In CRLR, we transfer the many-to-one into a one-to-one match
    37	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
    38	
    39	DLLR for Unsupervised Visible-Infrared Person ReID
    40	
    41	325:17
    42	
    43	Fig. 7. We plot four figures to show the changes in cluster numbers during training: (a) the change of
    44	modality-specific pseudo classes during the baseline training process; (b) the changes in modality-specific
    45	pseudo-classes (before refinement) during training of our framework; (c) the number of clusters added by
    46	CRLR; (d) the number of clusters merged by CRLR.
    47	
    48	Fig. 8. Comparison with different methods on Adjusted Rand Index (ARI). The ARI is a similarity measure
    49	between two clusterings. We report ARI values of overall modality-shared pseudo labels, composed of visible
    50	and infrared pseudo labels.
    51	
    52	by separating and merging clusters. By using the separated labels and assigning them to another
    53	modality, we construct the modality-shared labels in each matching direction for training. This
    54	approach enables our method to achieve higher-quality pseudo labels.
    55	In Figure 8, we further present the Adjusted Rand Index (ARI) metric for the overall modalityshared pseudo labels of ours compared to other methods. The ARI measured the similarity between
    56	two clusterings and served as a metric for evaluating the reliability of cross-modality correspondences. The results indicate that our approach yields higher quality modality-shared pseudo labels
    57	than other methods, particularly with the integration of CRLR.
    58	5
    59	
    60	Discussion
    61	
    62	Our framework demonstrates good performance in promoting USVI-ReID for real-world applications. However, some limitations still need to be improved: (1) The clustering algorithm employed
    63	for pseudo label generation suffers from long running time and substantial resource requirements.
    64	(2) Noise labels introduced by the clustering algorithm persist throughout model training. Better
    65	label generation algorithms or label refinement algorithms are needed to enhance feature learning.
    66	(3) Using more generated data instead of real data can promote future research. However, the
    67	previous research focuses on using real data since the outcome of infrared image generation still
    68	awaits improvement. These limitations are also prevalent in existing USVI-ReID methods and
    69	represent crucial directions for future optimization.
    70	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
    71	
    72	325:18
    73	6
    74	
    75	L. Dai et al.
    76	
    77	Conclusion
    78	
    79	This article presents a novel DLLR framework for USVI-ReID. First, a CSM algorithm along with
    80	a CRLR algorithm is designed to obtain high-quality pseudo labels. Then, we propose the WMM
    81	for dual-modality-shared learning. The method is superior for refining pseudo labels with crossmodality relationships and building a suitable memory bank for cross-modality learning. Although
    82	the model has room for improvement since the training efficiency is affected by clustering and
    83	noise labels, extensive experiments demonstrate the importance and effectiveness of our work. In
    84	the future, we will explore generated data instead of real data and the application of the framework
    85	in other multi-modality image retrieval fields to facilitate research.
    86	References
    87	[1] Hao Chen, Benoit Lagadec, and Francois Bremond. 2021. ICE: Inter-instance contrastive encoding for unsupervised
    88	person re-identification. In ICCV . IEEE, 14960–14969.
    89	[2] Hao Chen, Yaohui Wang, Benoit Lagadec, Antitza Dantcheva, and Francois Bremond. 2021. Joint generative and
    90	contrastive learning for unsupervised person re-identification. In CVPR. IEEE, 2004–2013.
    91	[3] Zhong Chen, Zhizhong Zhang, Xin Tan, Yanyun Qu, and Yuan Xie. 2023. Unveiling the power of clip in unsupervised
    92	visible-infrared person re-identification. In ACM MM. ACM, 3667–3675.
    93	[4] De Cheng, Lingfeng He, Nannan Wang, Shizhou Zhang, Zhen Wang, and Xinbo Gao. 2023. Efficient bilateral crossmodality cluster matching for unsupervised visible-infrared person ReID. In ACM MM. ACM, 1325–1333.
    94	[5] De Cheng, Xiaojian Huang, Nannan Wang, Lingfeng He, Zhihui Li, and Xinbo Gao. 2023. Unsupervised visible-infrared
    95	person reid by collaborative learning with neighbor-guided label refinement. In ACM MM. ACM, 7085–7093.
    96	[6] Yoonki Cho, Woo Jae Kim, Seunghoon Hong, and Sung-Eui Yoon. 2022. Part-based pseudo label refinement for
    97	unsupervised person re-identification. In CVPR. IEEE, 7308–7318.
    98	[7] Seokeon Choi, Sumin Lee, Youngeun Kim, Taekyung Kim, and Changick Kim. 2020. Hi-CMD: Hierarchical crossmodality disentanglement for visible-infrared person re-identification. In CVPR. IEEE, 10257–10266.
    99	[8] Zuozhuo Dai, Guangyuan Wang, Weihao Yuan, Siyu Zhu, and Ping Tan. 2022. Cluster contrast for unsupervised
   100	person re-identification. In ACCV . Springer, 1142–1160.
   101	[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image
   102	database. In CVPR. IEEE, 248–255.
   103	[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
   104	Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An
   105	image is worth 16x16 words: Transformers for image recognition at scale. In ICLR.
   106	[11] Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996. A density-based algorithm for discovering
   107	clusters in large spatial databases with noise. In KDD, 226–231.
   108	[12] Jiawei Feng, Ancong Wu, and Wei-Shi Zheng. 2023. Shape-erased feature learning for visible-infrared person reidentification. In CVPR. IEEE, 22752–22761.
   109	[13] Chaoyou Fu, Yibo Hu, Xiang Wu, Hailin Shi, Tao Mei, and Ran He. 2021. CM-NAS: Cross-modality neural architecture
   110	search for visible-infrared person re-identification. In ICCV . IEEE, 11823–11832.
   111	[14] Yang Fu, Yunchao Wei, Guanshuo Wang, Yuqian Zhou, Honghui Shi, and Thomas S. Huang. 2019. Self-similarity
   112	grouping: A simple unsupervised cross domain adaptation approach for person re-identification. In ICCV . IEEE,
   113	6112–6121.
   114	[15] Yaroslav Ganin and Victor Lempitsky. 2015. Unsupervised domain adaptation by backpropagation. In ICML, 1180–1189.
   115	[16] Yajun Gao, Tengfei Liang, Yi Jin, Xiaoyan Gu, Wu Liu, Yidong Li, and Congyan Lang. 2021. MSO: Multi-feature space
   116	joint optimization network for RGB-infrared person re-identification. In ACM MM. ACM, 5257–5265.
   117	[17] Yixiao Ge, Dapeng Chen, and Hongsheng Li. 2020. Mutual mean-teaching: Pseudo label refinery for unsupervised
   118	domain adaptation on person re-identification. In ICLR.
   119	[18] Yixiao Ge, Feng Zhu, Dapeng Chen, Rui Zhao, and Hongsheng Li. 2020. Self-paced contrastive learning with hybrid
   120	memory for domain adaptive object Re-ID. In NIPS, 11309–11321.
   121	[19] Xin Hao, Sanyuan Zhao, Mang Ye, and Jianbing Shen. 2021. Cross-modality person re-identification via modality
   122	confusion and center aggregation. In ICCV . IEEE, 16403–16412.
   123	[20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In
   124	CVPR. IEEE, 770–778.
   125	[21] Ziqi He, Mengjia Xue, Yunhao Du, Zhicheng Zhao, and Fei Su. 2024. Dynamic clustering and cluster contrastive
   126	learning for unsupervised person re-id with feature distribution alignment. In ICASSP, 3610–3614.
   127	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.
   128	
   129	
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Neural Networks 203 (2026) 109083
     2	
     3	Z.-H. Li et al.
     4	
     5	DAKD loss can be expressed as:
     6	𝑑𝑎𝑘𝑑 = −
     7	
     8	||
     9	∑
    10	
    11	𝑦𝑖 ⋅ log 𝑦̂𝑖 ,
    12	
    13	Algorithm 2: Training process of the student model.
    14	Require: Unlabeled training data 𝑃 ; initialize the encoder 𝑓𝜃
    15	with ImageNet-pretrained ResNet-50; the teacher
    16	encoder 𝑓̃𝜃 trained on the unlabeled training data 𝑃
    17	using Algorithm 1; balancing factors 𝜇, 𝛼 and 𝛽 for
    18	Eqs. (1), (15) and (17); hyper-parameter 𝑘𝑐1 and 𝑘𝑖1 for
    19	Eq. (3); hyper-parameter 𝑘𝑐2 and 𝑘𝑖2 for Eq. (5);
    20	temperature hyper-parameter 𝜏 for Eqs. (9) and (11)
    21	// first stage
    22	{
    23	}
    24	Extract feature vector sets 𝐹̃𝑔𝑏 , 𝐹̃𝑢𝑝 , 𝐹̃𝑙𝑤 from 𝑃 by 𝑓̃𝜃 ;
    25	Compute dynamic adaptive camera Jaccard distance matrixs
    26	{
    27	}
    28	𝐷̃ 𝑔𝑏 , 𝐷̃ 𝑢𝑝 , 𝐷̃ 𝑙𝑤 with Eq. (6);
    29	{
    30	}
    31	Clustering 𝐹̃𝑔𝑏 , 𝐹̃𝑢𝑝 , 𝐹̃𝑙𝑤 from 𝑃 by 𝑓̃𝜃 into 𝐶 clusters with Eq.
    32	(1) and DBSCAN;
    33	Initialize four memory banks individually with DAP ;
    34	for 𝑖 in [1, 𝑛𝑢𝑚_𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠 × 2] do
    35	Sample 𝑃 × 𝐾 query images from 𝑃 ;
    36	Compute objective function with Eq. (17) ;
    37	end
    38	// second stage
    39	for 𝑛 in [1,num_epochs] do
    40	{
    41	}
    42	Extract feature vector sets 𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 from 𝑃 by 𝑓𝜃 ;
    43	Compute dynamic adaptive camera Jaccard distance
    44	{
    45	}
    46	matrixs 𝐷𝑔𝑏 , 𝐷𝑢𝑝 , 𝐷𝑙𝑤 with Eq. (6);
    47	{
    48	}
    49	Clustering 𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 into 𝐶 clusters with Eq. (1) and
    50	DBSCAN;
    51	Initialize four memory banks individually with DAP ;
    52	for 𝑖 in [1,num_iterations] do
    53	Sample 𝑃 × 𝐾 query images from 𝑃 ;
    54	Obtain soft pseudo-labels with Eq. (12) ;
    55	Compute objective function with Eq. (19) ;
    56	Update feature representations with Eqs. (8) and (10);
    57	end
    58	end
    59	
    60	(13)
    61	
    62	𝑖=1
    63	
    64	where || denotes the number of samples in the mini-batch . Finally,
    65	we introduce a regularization term to improve the stability of the features and ensure that the student model remains robust across various
    66	scenarios. This allows the DAKD loss to be optimized as follows:
    67	𝑑𝑎𝑘𝑑 = −
    68	
    69	||
    70	∑
    71	
    72	2
    73	‖ 𝑿̂ 𝑖
    74	𝑿𝑖 ‖
    75	‖
    76	𝑦𝑖 ⋅ log 𝑦̂𝑖 + ‖
    77	‖ ̂ − ‖𝑿 ‖ ‖ ,
    78	‖ ‖𝑿 𝑖 ‖
    79	𝑖 ‖2
    80	𝑖=1
    81	
    82	(14)
    83	
    84	where 𝑿̂ 𝑖 and 𝑿 𝑖 are the feature vectors of query 𝑥𝑖 in the teacher model
    85	and student model, respectively.
    86	3.5. Overall loss
    87	According to Eqs. (9) and (11), the cluster proxies loss 𝑡𝑒𝑎
    88	𝑐𝑝 and instance proxies loss 𝑖𝑝 are deﬁned as follows:
    89	𝑔𝑏
    90	𝑢𝑝
    91	𝑙𝑤
    92	𝑡𝑒𝑎
    93	𝑐𝑝 = (1 − 𝛼)𝑑𝑜𝑐𝑝 + 𝛼(𝑑𝑜𝑐𝑝 + 𝑑𝑜𝑐𝑝 ),
    94	
    95	(15)
    96	
    97	𝑖𝑝 = 𝑑𝑖𝑝 ,
    98	
    99	(16)
   100	
   101	where 𝛼 is the loss weight to balance the importance between global and
   102	local features. {𝑔𝑏
   103	, 𝑢𝑝
   104	, 𝑙𝑜
   105	} are the DOCP losses corresponding
   106	𝑑𝑜𝑐𝑝
   107	𝑑𝑜𝑐𝑝
   108	𝑑𝑜𝑐𝑝
   109	to the global, upper and lower views. Thus, the loss function of teacher
   110	model is:
   111	𝑡𝑒𝑎 = 𝑡𝑒𝑎
   112	𝑐𝑝 + 𝛽𝑖𝑝 ,
   113	
   114	(17)
   115	
   116	where 𝛽 controls the degree of two loss functions.
   117	For the student model, the 𝑠𝑡𝑢
   118	𝑐𝑝 is deﬁned as follows:
   119	𝑔𝑏
   120	𝑔𝑏
   121	𝑢𝑝
   122	𝑢𝑝
   123	𝑙𝑤
   124	𝑙𝑤
   125	𝑠𝑡𝑢
   126	𝑐𝑝 = (1 − 𝛼)(𝑑𝑜𝑐𝑝 + 𝑑𝑎𝑘𝑑 ) + 𝛼(𝑑𝑜𝑐𝑝 + 𝑑𝑎𝑘𝑑 + 𝑑𝑜𝑐𝑝 + 𝑑𝑎𝑘𝑑 ),
   127	
   128	(18)
   129	
   130	where {𝑔𝑏
   131	, 𝑢𝑝
   132	, 𝑙𝑜
   133	} are the DAKD losses as in Eq. (14) corre𝑑𝑎𝑘𝑑
   134	𝑑𝑎𝑘𝑑
   135	𝑑𝑎𝑘𝑑
   136	sponding to the global, upper and lower views. The instance proixes is
   137	not change. Thus, the loss function of student model is :
   138	𝑠𝑡𝑢 = 𝑠𝑡𝑢
   139	𝑐𝑝 + 𝛽𝑖𝑝 .
   140	
   141	Training Process of the Teacher Model: The detailed training process of the teacher model is shown in Algorithm 1. Given the unlabeled
   142	dataset 𝑃 and the encoder 𝑓𝜃 initialized with ResNet-50 parameters pretrained on ImageNet (Deng et al., 2009), the training procedure is as follows. In each epoch, the encoder 𝑓𝜃 is used to extract the global feature
   143	map set 𝑀𝑔𝑏 . These feature maps are then horizontally split into upper
   144	and lower parts, denoted as 𝑀𝑢𝑝 and 𝑀𝑙𝑤 , respectively. Subsequently,
   145	GEM pooling is applied to obtain the corresponding feature vector sets
   146	{𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 }. Based on the feature vector sets, we apply the DACJ
   147	module to compute the DACJ distances for the three views, denoted as
   148	{𝐷𝑔𝑏 , 𝐷𝑢𝑝 , 𝐷𝑙𝑤 }. Subsequently, pseudo-labels are generated by applying
   149	Eq. (1) in conjunction with the DBSCAN clustering algorithm. Finally,
   150	the DAP module is employed to initialize three cluster-level memory
   151	banks and one instance-level memory bank. Once the pseudo-labels and
   152	memory banks are prepared, model training begins. Speciﬁcally, in each
   153	iteration, a mini-batch of 𝑝 × 𝑘 query images is sampled from 𝑃 (where 𝑝
   154	represents the number of selected identities, and 𝑘 denotes the number
   155	of images per identity), and the model parameters are updated according to Eq. (17). Finally, the features stored in the memory banks are
   156	updated using Eqs. (8) and (10).
   157	Training Process of the Student Model: The detailed training process of the teacher model is shown in Algorithm 2. Apart from the unlabeled dataset 𝑃 and the encoder 𝑓𝜃 initialized with ResNet-50 parameters pretrained on ImageNet, a teacher model trained via Algorithm 1
   158	is also needed. The student model’s training procedure comprises two
   159	stages.
   160	
   161	(19)
   162	
   163	3.6. Training process
   164	
   165	Algorithm 1: Training process of the teacher model.
   166	Require: Unlabeled training dataset 𝑃 ; initialize the encoder 𝑓𝜃
   167	with ImageNet-pretrained ResNet-50; balancing
   168	factors 𝜇, 𝛼 and 𝛽 for Eqs. (1),(15) and (17);
   169	hyper-parameter 𝑘𝑐1 and 𝑘𝑖1 for Eq. (3);
   170	hyper-parameter 𝑘𝑐2 and 𝑘𝑖2 for Eq. (5); temperature
   171	hyper-parameter 𝜏 for Eqs. (9) and (11)
   172	for 𝑛 in [1,num_epochs] do
   173	{
   174	}
   175	Extract feature vector sets 𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 from 𝑃 by 𝑓𝜃 ;
   176	Compute dynamic adaptive camera Jaccard distance
   177	{
   178	}
   179	matrixs 𝐷𝑔𝑏 , 𝐷𝑢𝑝 , 𝐷𝑙𝑤 with Eq. (6);
   180	{
   181	}
   182	Clustering 𝐹𝑔𝑏 , 𝐹𝑢𝑝 , 𝐹𝑙𝑤 into 𝐶 clusters with Eq. (1) and
   183	DBSCAN;
   184	Initialize four memory banks individually with DAP ;
   185	for 𝑖 in [1,num_iterations] do
   186	Sample 𝑃 × 𝐾 query images from 𝑃 ;
   187	Compute objective function with Eq. (17) ;
   188	Update feature representations with Eqs. (8) and (10);
   189	end
   190	end
   191	
   192	8
   193	
   194	Neural Networks 203 (2026) 109083
   195	
   196	Z.-H. Li et al.
   197	
   198	Table 1
   199	Ablation studies on Market-1501, DukeMTMC-reID and MSMT17. Duke denotes DukeMTMC-reID and Base denotes baseline. The best results are marked
   200	in bold.
   201	
   202	In the ﬁrst stage, we directly utilize the trained teacher model to
   203	̃ 𝑔𝑏 . GEM poolencode the dataset 𝑃 into the global feature map set 𝑀
   204	ing followed by horizontal splitting is then applied to derive the corresponding feature vector sets {𝐹̃𝑔𝑏 , 𝐹̃𝑢𝑝 , 𝐹̃𝑙𝑤 }. Subsequently, the DACJ
   205	module is employed to compute the DACJ distances for the three views,
   206	denoted as {𝐷̃ 𝑔𝑏 , 𝐷̃ 𝑢𝑝 , 𝐷̃ 𝑙𝑤 }. Eq. (1) and the DBSCAN algorithm are then
   207	used to generate pseudo-labels and initialize the memory banks for each
   208	branch, adhering to the same procedure as in the teacher model’s training process. Then, we sample 𝑝 × 𝑘 query images from 𝑃 as a mini-batch
   209	to update the model’s parameters based on Eq. (17), without modifying the features stored in the memory bank. The key objective here is
   210	to utilize the stable cluster proxy and instance proxy from the teacher
   211	model’s memory banks, allowing the student model to acquire knowledge quickly and eﬀectively while mitigating the buildup of label noise
   212	during the initial training phase.
   213	In the second stage, the training process of the student model in each
   214	epoch is largely identical to that of the teacher model, with the primary
   215	diﬀerence being that we compute soft pseudo-labels for the samples and
   216	update the student model’s parameters according to Eqs. (12) and (19),
   217	respectively.
   218	
   219	Base
   220	
   221	✓
   222	✓
   223	✓
   224	✓
   225	✓
   226	✓
   227	
   228	DACJ
   229	
   230	DAP
   231	
   232	DAKD
   233	
   234	✓
   235	✓
   236	
   237	✓
   238	
   239	✓
   240	✓
   241	
   242	✓
   243	✓
   244	✓
   245	
   246	Market-1501
   247	
   248	Duke
   249	
   250	mAP
   251	
   252	R1
   253	
   254	mAP
   255	
   256	R1
   257	
   258	MSMT17
   259	mAP
   260	
   261	R1
   262	
   263	84.4
   264	85.7
   265	85.2
   266	86.9
   267	87.4
   268	87.8
   269	
   270	93.1
   271	94.1
   272	94.0
   273	94.4
   274	94.1
   275	95.1
   276	
   277	74.3
   278	76.7
   279	75.1
   280	76.9
   281	77.3
   282	78.5
   283	
   284	85.9
   285	87.8
   286	85.9
   287	86.5
   288	87.1
   289	87.1
   290	
   291	47.1
   292	51.5
   293	49.3
   294	52.6
   295	54.3
   296	60.0
   297	
   298	71.6
   299	76.0
   300	73.6
   301	75.8
   302	77.0
   303	81.5
   304	
   305	4.3. Ablation study
   306	We perform comprehensive ablation studies and develop multiple
   307	variants of the proposed method. The intermediate results for each component are evaluated and reported on three datasets. As a baseline, we
   308	adopt a method that employs Jaccard distance for clustering, where
   309	mean features serve as proxies and are updated using a momentumbased mechanism.
   310	
   311	4. Experiment
   312	
   313	4.3.1. Eﬀectiveness of DACJ
   314	As evidenced in Table 1, our DACJ module demonstrates substantial
   315	improvements over the baseline, achieving consistent performance gains
   316	across all datasets: +1.3% mAP/+1.0% R1 on Market-1501, +2.4%
   317	mAP/+1.9% R1 on DukeMTMC-reID, and +4.4% in both metrics on
   318	MSMT17. To precisely quantify the impact of DACJ, we conduct an
   319	ablation study by removing this module from the complete DAMCL
   320	framework (Base + DACJ + DAP + DAKD). The ablated model (Base
   321	+ DAP + DAKD) exhibits performance degradation across all three
   322	datasets, with mAP reductions of 0.5%, 1.2%, and 5.7% on Market-1501,
   323	DukeMTMC-reID, and MSMT17, respectively. This evidence conﬁrms
   324	that DACJ eﬀectively mitigates cross-camera discrepancies through dynamic sample-level camera variations assessment and adaptive distance
   325	metric calculation adjustment, thereby enhancing clustering reliability
   326	for pseudo-label generation.
   327	
   328	4.1. Datasets and evaluation protocols
   329	We conduct our evaluation using three widely recognized person ReID datasets: Market-1501 (Zheng et al., 2015), DukeMTMC-reID (Ristani et al., 2016), and MSMT17 (Wei et al., 2018). The Market-1501
   330	dataset consists of 32,668 images of 1501 unique pedestrians, captured
   331	across 6 diﬀerent cameras. DukeMTMC-reID includes 36,411 images of
   332	1404 pedestrians, recorded by 8 cameras at the Duke University campus. MSMT17, on the other hand, contains 126,441 images of 1041
   333	pedestrians, captured by 15 diﬀerent cameras. It is important to highlight that MSMT17 is not only the largest dataset in terms of the number of images but also provides greater environmental variability and
   334	a longer time span compared to the other two datasets. Consequently,
   335	MSMT17 poses more challenges in person re-identiﬁcation tasks due
   336	to its scale and complexity. We use two widely recognized evaluation
   337	metrics: mean Average Precision (mAP) and Cumulative Matching Characteristic (CMC). Speciﬁcally, we present the Rank-1 (R1), Rank-5 (R5),
   338	and Rank-10 (R10) accuracy values derived from the CMC curves in our
   339	analysis.
   340	
   341	4.3.2. Eﬀectiveness of DAP
   342	As detailed in Table 1, comparing to the baseline, mAP of Base +
   343	DAP is improved by 0.8% on Market-1501 and DukeMTMC-reID, 2.2%
   344	on MSMT17. This demonstrates that our DAP eﬀectively address the
   345	issues caused by traditional mean feature proxies and momentum update strategy in contrastive learning systems. This is especially crucial
   346	when the real-world data is not distributed in spherical clusters. This
   347	is achieved through enhanced proxy initialization methods and an improved proxy update strategy. And it has been demonstrated in Table 1
   348	that our optimal feature proxies outperform the mean feature proxies.
   349	In Table 2, we quantitatively analyze the performance variations between other proxy types and our approach, revealing that our method
   350	(Optimal) achieves the best performance. In Table 3, the ablation experiments reveal that the momentum update strategy yields suboptimal
   351	performance for updating the memory bank, particularly on the more
   352	complex MSMT17. This demonstrates that the momentum update strategy fails to preserve the intrinsic manifold structure of the data or maintain the integrity of optimal sample features, rendering it poorly suited
   353	to complex data distributions and leading to a signiﬁcant performance
   354	decline. To further validate the inseparability of the two submodules
   355	in DAP, we conduct ablation experiments as summarized in Table 4.
   356	When individually applied, each submodule provides only marginal improvements over the baseline: integrating DOCP yields a modest +0.5%
   357	mAP/+0.2% R1 gain on MSMT17, while DIP shows negligible performance variation (+0.1% mAP/±0.0% R1). However, their combined
   358	implementation achieves substantial performance enhancement, with
   359	improvements exceeding the arithmetic sum of individual contributions.
   360	
   361	4.2. Implementation details
   362	We utilize ResNet50 as our backbone model, initialized with parameters pre-trained on ImageNet. To adapt the architecture, we discard all
   363	sub-modules beyond the 4th layer. In their place, we introduce a generalized mean pooling (GEM) layer, followed by a batch normalization
   364	(BN) layer, which outputs 2048-dimensional feature embeddings.
   365	During the training process, each mini-batch consists of 128 images
   366	belonging to 16 pseudo classes for Market-1501 and DukeMTMC-reID,
   367	while for MSMT17, it comprises 256 images from 16 pseudo classes.
   368	All input images are resized to 256 × 128. Strong data augmentation
   369	includes random ﬂipping, padding by 10 pixels, random cropping, and
   370	random erasing. DBSCAN is utilized as the clustering algorithm, with the
   371	maximum distance between two samples to be considered neighbors set
   372	to 0.6 and the minimum number of neighbors set to 4. Adam (Kingma,
   373	2014) is chosen as the optimizer with a weight decay of 5e-4. The learning rate is initialized to 3.5e-4 and reduced by a factor of 0.1 every 20
   374	epochs. Training is conducted for a total of 50 epochs. We set the 𝜇 to
   375	0.15 in Eq. (1). The 𝑘 in Eq. (2) is set to 30. The 𝑘𝑐1 and 𝑘𝑖1 are set to 25
   376	and 5 respectively in Eq. (3). The 𝑘𝑐2 and 𝑘𝑖2 are set to 4 and 2 respectively in Eq. (5). The temperature hyper-parameter 𝜏 is set to 0.05. The
   377	𝛼 in Eqs. (15) and (18) are set to 0.2. The 𝛽 in Eqs. (19) and (17) is set
   378	to 1.2.
   379	9
   380	
   381	Neural Networks 203 (2026) 109083
   382	
   383	Z.-H. Li et al.
   384	
   385	Table 2
   386	Qualitative ablation studies of our dynamic adaptive optimal proxy with
   387	other diﬀerent proxies on MSMT17 dataset.
   388	proxy types
   389	Hardest
   390	
   391	mAP
   392	Easiest
   393	
   394	Random
   395	
   396	R1
   397	
   398	R5
   399	
   400	Table 4
   401	Ablation studies of the two submodules (DOCP, DIP) of
   402	the DAP on MSMT17.
   403	
   404	R10
   405	
   406	Base
   407	
   408	Optimal
   409	
   410	✓
   411	
   412	DOCP
   413	39.0
   414	48.1
   415	48.4
   416	49.3
   417	
   418	✓
   419	✓
   420	✓
   421	
   422	62.8
   423	73.3
   424	72.9
   425	73.6
   426	
   427	75.9
   428	83.9
   429	82.8
   430	83.6
   431	
   432	✓
   433	✓
   434	✓
   435	✓
   436	
   437	80.2
   438	87.5
   439	86.1
   440	87.1
   441	
   442	Table 3
   443	Ablation studies comparing the momentum update strategy with our
   444	proposed optimal update strategy, as deﬁned in Eqs. (8) and (10) on
   445	Market1501, DukeMTMC-reID and MSMT17.
   446	Update Strategies
   447	
   448	Momentum
   449	Optimal(ours)
   450	
   451	DAP
   452	
   453	Market-1501
   454	
   455	DukeMTMC-reID
   456	
   457	MSMT17
   458	
   459	mAP
   460	
   461	R1
   462	
   463	mAP
   464	
   465	R1
   466	
   467	mAP
   468	
   469	R1
   470	
   471	73.4
   472	87.8
   473	
   474	81.6
   475	95.1
   476	
   477	69.4
   478	78.5
   479	
   480	77.3
   481	87.1
   482	
   483	34.1
   484	60.0
   485	
   486	65.5
   487	81.5
   488	
   489	✓
   490	✓
   491	
   492	mAP
   493	
   494	R1
   495	
   496	R5
   497	
   498	R10
   499	
   500	47.1
   501	47.6
   502	47.2
   503	49.3
   504	
   505	71.6
   506	71.8
   507	71.6
   508	73.6
   509	
   510	82.6
   511	83.1
   512	82.8
   513	83.6
   514	
   515	86.4
   516	86.5
   517	86.2
   518	87.1
   519	
   520	DIP

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 10 -l 12 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Neural Networks 203 (2026) 109083
     2	
     3	Z.-H. Li et al.
     4	
     5	Table 2
     6	Qualitative ablation studies of our dynamic adaptive optimal proxy with
     7	other diﬀerent proxies on MSMT17 dataset.
     8	proxy types
     9	Hardest
    10	
    11	mAP
    12	Easiest
    13	
    14	Random
    15	
    16	R1
    17	
    18	R5
    19	
    20	Table 4
    21	Ablation studies of the two submodules (DOCP, DIP) of
    22	the DAP on MSMT17.
    23	
    24	R10
    25	
    26	Base
    27	
    28	Optimal
    29	
    30	✓
    31	
    32	DOCP
    33	39.0
    34	48.1
    35	48.4
    36	49.3
    37	
    38	✓
    39	✓
    40	✓
    41	
    42	62.8
    43	73.3
    44	72.9
    45	73.6
    46	
    47	75.9
    48	83.9
    49	82.8
    50	83.6
    51	
    52	✓
    53	✓
    54	✓
    55	✓
    56	
    57	80.2
    58	87.5
    59	86.1
    60	87.1
    61	
    62	Table 3
    63	Ablation studies comparing the momentum update strategy with our
    64	proposed optimal update strategy, as deﬁned in Eqs. (8) and (10) on
    65	Market1501, DukeMTMC-reID and MSMT17.
    66	Update Strategies
    67	
    68	Momentum
    69	Optimal(ours)
    70	
    71	DAP
    72	
    73	Market-1501
    74	
    75	DukeMTMC-reID
    76	
    77	MSMT17
    78	
    79	mAP
    80	
    81	R1
    82	
    83	mAP
    84	
    85	R1
    86	
    87	mAP
    88	
    89	R1
    90	
    91	73.4
    92	87.8
    93	
    94	81.6
    95	95.1
    96	
    97	69.4
    98	78.5
    99	
   100	77.3
   101	87.1
   102	
   103	34.1
   104	60.0
   105	
   106	65.5
   107	81.5
   108	
   109	✓
   110	✓
   111	
   112	mAP
   113	
   114	R1
   115	
   116	R5
   117	
   118	R10
   119	
   120	47.1
   121	47.6
   122	47.2
   123	49.3
   124	
   125	71.6
   126	71.8
   127	71.6
   128	73.6
   129	
   130	82.6
   131	83.1
   132	82.8
   133	83.6
   134	
   135	86.4
   136	86.5
   137	86.2
   138	87.1
   139	
   140	DIP
   141	
   142	✓
   143	✓
   144	
   145	Table 5
   146	Ablation studies of diﬀerent number of local split methods. Time represents the total duration of model training, measured in hours (h). The
   147	model is only trained on a single RTX 3090 GPU.
   148	Split Numbers
   149	
   150	Market-1501
   151	mAP R1
   152	
   153	Four
   154	Three
   155	Dual (ours)
   156	
   157	This empirically validates our theoretical analysis in Section 3.3, demonstrating that DOCP and DIP are mutually dependent components that
   158	function synergistically to optimize feature representation learning.
   159	
   160	DukeMTMC-reID
   161	Time mAP R1
   162	
   163	85.7 93.6 6.2
   164	87.4 94.4 4.0
   165	87.8 95.1 2.9
   166	
   167	MSMT17
   168	
   169	Time mAP R1
   170	
   171	75.3 86.1 7.1
   172	78.4 88.4 5.2
   173	78.5 87.1 3.4
   174	
   175	Time
   176	
   177	47.7 72.6 24.7
   178	53.2 76.8 16.4
   179	60.0 81.5 13.0
   180	
   181	are far apart in the feature space. With the proposed DAMCL, even the
   182	samples of the same person under diﬀerent cameras have signiﬁcantly
   183	reduced distances in the feature space and are clustered more compactly. This demonstrates that the DAMCL approach eﬀectively learns
   184	camera-invariant person feature representations, successfully mitigating
   185	the negative eﬀects of camera variations. As a result, the features of the
   186	same person captured by diﬀerent cameras become more similar, which
   187	in turn enhances the accuracy of person Re-ID.
   188	
   189	4.3.3. Eﬀectiveness of DAKD
   190	We combine DAKD with the baseline and the other two modules
   191	respectively. In Table 1, compared to the baseline, mAP/R1 of Base
   192	+ DAKD is improved by 2.5%/1.2% on Market-1501, 2.6%/0.6% on
   193	DukeMTMC-reID and 5.5%/4.2% on MSMT17. Base + DAKD + DAP
   194	increases mAP/R1 of Base + DAP by 2.2%/0.1% on Market1501,
   195	2.2%/1.2% on DukeMTMC-reID and 5.0%/3.4% on MSMT17. This suggests that, even in the absence of the DACJ to mitigate label noise,
   196	the DAKD method can still eﬀectively counteract the negative eﬀects
   197	of label noise during model training by producing high-quality soft labels. Furthermore, it is worth noting that compared to Market-1501 and
   198	DukeMTMC-reID, our DAKD provides the greatest improvement to the
   199	model’s performance on the more challenging MSMT17.
   200	
   201	4.4. Parameter analysis
   202	In multi-view frameworks, although the number of hyper-parameters
   203	is large, they are all necessary designs to address core issues such as
   204	camera variations and pseudo-label noise (e.g., 𝜇 balances multi-view
   205	weights, 𝑘𝑐1 ∕𝑘𝑐2 enhances the role of cross-camera samples, and 𝛼∕𝛽
   206	jointly optimizes global and local features). The parameters are logically
   207	interconnected, with their reasonable ranges validated through experiments (e.g., 𝜇 = 0.15, 𝑘𝑐1 = 25), and computational costs are controlled
   208	(e.g., 𝑘2 = 6) to ensure a balance between performance and eﬃciency.
   209	Systematic experiments demonstrate that the parameter design is targeted, synergistic, and data-driven, serving as a key to model robustness
   210	rather than a redundant drawback. In this section, we provide a comprehensive analysis of the model’s sensitivity to key hyper-parameters. Our
   211	focus is on understanding how these hyper-parameters inﬂuence both
   212	the model’s performance and its training eﬃciency.
   213	𝜇 in Eq. (1): The hyper-parameter 𝜇 is used to integrate the DACJs
   214	from three views. In Fig. 7(c), when 𝜇 is set to 0.15, the model achieves
   215	optimal performance on MSMT17 and DukeMTMC-reID, while a setting
   216	of 𝜇 at 0.2 yields the best performance on Market-1501. To balance
   217	across diﬀerent datasets, we set 𝜇 to 0.15 for all datasets.
   218	𝑘𝑖1 and 𝑘𝑐1 in Eq. (3): Following (Zhong et al., 2017), we set k in
   219	Eq. (2) to 30, such that 𝑘𝑖1 + 𝑘𝑐1 = 30. In Fig. 7(a), we observe that when
   220	𝑘𝑖1 ∕𝑘𝑐1 is 5/25, the performance is best. When 𝑘𝑖1 is less than 5 or greater
   221	than 5, model performance decreases. This indicates that selecting a
   222	larger 𝑘𝑐1 allows us to gather more informative positive samples across
   223	cameras, leading to a more accurate distance matrix that helps clustering
   224	algorithms generate cleaner pseudo-labels. Additionally, while ensuring
   225	𝑘𝑐1 is greater than 𝑘𝑖1 , it is important to choose an appropriate 𝑘𝑖1 to obtain rich cross-camera positive samples without discarding those from
   226	the same camera.
   227	𝑘𝑖2 and 𝑘𝑐2 in Eq. (5): According to Zhong et al. (2017), we limit the
   228	sum of 𝑘𝑖2 and 𝑘𝑐2 to 6. As shown in Fig. 7(b), the model shows enhanced
   229	performance with smaller 𝑘𝑖2 and larger 𝑘𝑐2 . However, performance declines when this balance is inverted. Based on these observations, we set
   230	
   231	4.3.4. Eﬀectiveness of multi-view
   232	Our DAMCL framework employs multi-view images (upper, lower,
   233	and global views) as inputs to address Re-ID challenges. This dual local view approach eﬀectively avoids neglecting inconspicuous yet crucial local cues that might be ignored by global feature only methods,
   234	enabling comprehensive extraction of subtle discriminative features. As
   235	systematically validated in our ablation studies with varying local views
   236	(Table 5), the dual local view conﬁguration achieves optimal eﬃciencyaccuracy equilibrium. The dual local views method achieves the highest mAP and Rank-1 accuracy on Market-1501 and MSMT17, and the
   237	highest mAP (78.5%) on DukeMTMC-reID. It is also more eﬃcient, requiring only 2.9, 3.4, and 13.0 h on Market-1501, DukeMTMC-reID, and
   238	MSMT17, respectively. In contrast, the three and four local views methods take 1.4× and 1.9× longer, with MSMT17 training times of 30.4 and
   239	40.7 h, nearly doubling our approach’s time. This signiﬁcant increase
   240	in time cost, with negligible performance gains, conﬁrms our design’s
   241	computational rationality Table 7.
   242	4.3.5. Clustering quality
   243	To visually demonstrate how our method reduces camera variations
   244	and mitigates pseudo-label noise, we applied t-SNE (Van der Maaten &
   245	Hinton, 2008) to randomly selected samples from 10 diﬀerent classes
   246	within the MSMT17 dataset. As illustrated in Fig. 6, the compactness
   247	of each class is notably enhanced in our DAMCL method when compared to the baseline. Moreover, due to the problems described in the
   248	Section 1, in the baseline, even if the samples belong to the same identity (same color), the samples from diﬀerent cameras (diﬀerent shapes)
   249	10
   250	
   251	Neural Networks 203 (2026) 109083
   252	
   253	Z.-H. Li et al.
   254	
   255	Table 6
   256	Comparison with the state-of-the-art methods on Market-1501, DukeMTMC-reID. The best and second best
   257	unsupervised results are marked in bold and italic.
   258	Market-1501
   259	
   260	Method
   261	
   262	DukeMTMC-reID
   263	
   264	mAP
   265	
   266	R1
   267	
   268	R5
   269	
   270	R10
   271	
   272	mAP
   273	
   274	R1
   275	
   276	R5
   277	
   278	R10
   279	
   280	86.0
   281	89.5
   282	92.7
   283	
   284	94.8
   285	95.2
   286	96.9
   287	
   288	–
   289	–
   290	–
   291	
   292	–
   293	–
   294	–
   295	
   296	74.8
   297	82.6
   298	83.5
   299	
   300	86.6
   301	90.7
   302	91.6
   303	
   304	–
   305	–
   306	–
   307	
   308	–
   309	–
   310	–
   311	
   312	NeurIPS’20
   313	ICCV’21
   314	PR’22
   315	ICCV’23
   316	TNNLS’24
   317	
   318	76.7
   319	82.8
   320	82.2
   321	84.7
   322	83.8
   323	
   324	90.3
   325	93.2
   326	92.7
   327	93.8
   328	93.1
   329	
   330	96.2
   331	97.5
   332	96.9
   333	97.7
   334	97.6
   335	
   336	97.7
   337	98.1
   338	98.3
   339	98.6
   340	98.5
   341	
   342	68.8
   343	70.5
   344	68.3
   345	–
   346	75.6
   347	
   348	82.9
   349	83.6
   350	80.9
   351	–
   352	85.5
   353	
   354	90.1
   355	91.5
   356	90.2
   357	–
   358	92.5
   359	
   360	92.5

codex
你这次列出的标题是 6 篇，不是 11 篇。我按这 6 篇拆；长文里方法超过前 8 页的，我补读了后续方法和关键消融页。

**1. Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID**

1. 触发观察：无监督 VI-ReID 不是简单缺标签，而是跨模态聚类会产生两类错。一类是同一身份被拆成多个簇，另一类是不同身份被错误合并。已有方法只做跨模态 cluster matching，却默认匹配关系可靠。作者还注意到 memory bank 用簇均值初始化，会被两模态样本数量不均和难样本拖偏。

2. 重定义动作：他们把问题从“怎么匹配可见光和红外簇”改成“跨模态匹配关系本身可以反过来诊断和修正伪标签”。关键词是 modality-shared pseudo labels、cluster relationship-based label refinement、weighted modality-shared memory、hard samples。这里的重点不是又做一个匹配算法，而是把匹配后的 many-to-one 关系解释成伪标签错误的信号。

3. 机制怎么长出来：先用 CSM 做双向 cluster similarity matching，得到跨模态伪标签。然后 CRLR 看哪些簇共同匹配到同一个异模态簇，如果同模态簇之间相似就合并；如果一个异模态簇被多个合并簇匹配，就用 k-means 把它切开。最后 WMM 不直接平均所有样本，而是先平衡两模态中心，再按样本到共享中心的关系给权重，让难样本在 memory 初始化里有更大作用。这个机制和重定义绑定较紧，因为每一步都在处理“伪标签关系不可靠”和“记忆中心不可靠”。

4. 证据闭环：它的关键不是只报涨点。消融里 baseline 到 CSM+MM 有大提升，CRLR 和 WMM 再分别增益；CRLR 还用伪标签质量指标证明 refinement 前后质量提高。t-SNE 显示跨模态簇边界变清，正负跨模态距离分布重叠变小，伪类数量逐渐接近真实类别数，ARI 也更高。这些证据直接服务于“伪标签关系能被修正”这个重定义。

5. reviewer 为什么买账：这篇卖的是“跨模态匹配关系可以用于 label refinement”的视角，加上一个完整训练框架。真正 novelty 不在某个复杂模块，而在把 cross-modality correspondence 从监督信号升级成诊断信号。审稿人会买账，是因为它把动机、机制和伪标签质量证据连起来了。

**2. Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification**

1. 触发观察：作者抓住无监督 ReID memory contrastive learning 的三个内部矛盾。第一，camera variation 会让同相机不同身份比跨相机同身份更近，Jaccard 聚类偏向错误邻居。第二，已有方法用 mean feature 当 cluster proxy，默认簇是球形，但 ReID 特征常是非球形流形，均值可能落在低密度空区。第三，one-hot hard pseudo label 对噪声不稳，会放大错误监督。

2. 重定义动作：他们没有泛泛说“伪标签有噪声”，而是把噪声拆成三个更具体的问题：距离度量不懂相机，代理点不符合 DBSCAN 的局部密度结构，监督目标太硬。关键词是 dynamic adaptive camera Jaccard、manifold、medoid proxy、dynamic adaptive proxies、soft pseudo labels。

3. 机制怎么长出来：DACJ 为每个样本估计跨相机和同相机邻居差异，动态增加跨相机正样本在 k-reciprocal 和 local query expansion 里的比例。DAP 用 medoid 代替 mean proxy，因为 medoid 是真实样本点，更贴合非球形流形；同时用 instance memory 补充簇内细粒度关系，并去掉动量均值更新。DAKD 再用 teacher-student 和 soft label 缓解 hard label 噪声。多视图的上半身、下半身、全局特征是补充判别信息。

4. 证据闭环：它的证据很对题。proxy 对比里 optimal medoid 优于 hardest、easiest、random；momentum update 和 optimal update 对比显示复杂 MSMT17 上差距很大；DOCP 和 DIP 单独作用小，合起来明显更强，证明两个代理层次互补。DACJ 的消融在 MSMT17 上尤其明显，说明相机差异越强越有用。t-SNE 显示同身份跨相机更紧，参数分析也围绕跨相机邻居比例展开。

5. reviewer 为什么买账：这篇卖的是“现有无监督 ReID 管线的几个默认假设错了”的视角。机制本身是多个组件，但每个组件都对应一个被重定义的假设错误。最有价值的是把“均值代理不适合 DBSCAN 发现的非球形簇”讲成了一个方法问题，而不是只说换个 proxy 涨点。

**3. Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification**

1. 触发观察：USL-VI-ReID 里，大家主要盯着 visible-infrared modality gap，但作者观察到 clustering 阶段还有 camera discrepancy。可见光图像受相机、光照、视角影响更大，导致同一身份被过度拆裂。已有 graph matching 或 optimal transport 如果建立在拆裂簇上，反而会放大 noisy labels。

2. 重定义动作：它把问题从“跨模态关联不准”改成“聚类过程被模态和相机信息同时污染”。关键词是 modality-camera invariant clustering、identity splitting、clustering-level discrepancy、from discriminability to generalization。这个表述比“减少模态差异”更具体，因为它把失败点锁定在聚类输入和聚类目标上。

3. 机制怎么长出来：MIE 在 k-reciprocal distance encoding 里均衡融合不同相机的信息，让 clustering embedding 不被同相机近邻主导。DNC 动态调 eps 和 k2，早期收缩半径排除噪声，先学可靠簇的判别性；后期扩大半径，把跨相机和跨模态正样本逐渐并进来。HMCL 先 intra-modality 再 inter-modality，用 cluster-level 和 instance-level memory 学模态相机不变表示。逻辑绑定很强，特别是 MIE 和 DNC 都直接服务于 identity splitting。

4. 证据闭环：关键证据是 clustering ablation 和 cluster number visualization。文本说明 MIE 带来约 10 到 15 个点 mAP，DNC 再带来 2 到 5 个点，合起来 14 到 17 个点。可视化里 visible cluster 数量更接近真实类别数，t-SNE 显示 MIE+DNC 后同身份跨模态更紧。objective ablation 里 inter-modality cluster loss 贡献最大，说明改好的跨模态伪标签确实在驱动训练。

5. reviewer 为什么买账：这篇卖的是视角，不只是模块。它把 USL-VI-ReID 的痛点从“模态对齐”推进到“模态和相机共同破坏聚类”。如果 reviewer 接受 identity splitting 是主要失败模式，MIE 和 DNC 就显得自然。它的 novelty 来源是把 camera-aware 思路放到 VI 全局聚类阶段，而不是只做相机代理损失。

**4. Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification**

1. 触发观察：AGPReID 的难点不只是 aerial 和 ground 视角差异，还包括大范围观测带来的尺度变化、背景冗余和无关区域。已有 VDT 能做 view decoupling，但仍然把大量 token 都送进 transformer，缺少对关键身份区域的动态选择。

2. 重定义动作：它把 aerial-ground ReID 从“视角差异问题”改写成“视角差异下的 token 冗余和关键区域选择问题”。关键词是 dynamic token selection、top-k informative tokens、compact long-range modeling、irrelevant tokens、view-heterogeneous ReID。

3. 机制怎么长出来：在 VDT 框架上保留 meta token 和 view token，用 identity、view classifier 和 orthogonal loss 做 view-related 与 view-unrelated disentanglement。新增 VTS，用轻量 attention 给每个 token 打分，取 top-K token，再用 Gumbel-Softmax 近似使选择可反传。最后选中的 token 和 class token 进入后续聚合。机制和重定义有一定绑定，但更多是把已有 token selection 思路迁移到 AGPReID。

4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。

5. reviewer 为什么买账：这篇更多卖机制迁移，而不是强视角。AGPReID 是新任务，VDT 是清晰强基线，在这个基线上加动态 token selection 有明确增益，reviewer 容易接受。真正 novelty 是“AGPReID 里显式选择关键 token”，但方法论上它不是最强的创新构造样本。

**5. Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning**

1. 触发观察：UDA ReID 常用目标域聚类和伪标签，但源域知识利用不足；DG 学泛化，却不直接适配目标域。现实部署还有边缘设备算力、隐私和低资源限制。另一方面，global-only feature 在遮挡、光照和低清场景下不够细。

2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。

3. 机制怎么长出来：WPOSNet 用 OSNet 的轻量深度可分离卷积，做 whole branch 和 part branch，垂直切分人体局部再拼接。SpML 把每个源域看成任务，用 SuA 扰动特征均值和方差形成不同难度，从无扰动、弱扰动到强扰动，模拟由易到难学习源域不变知识。最后用 MMT 在目标域做 hard pseudo label 和 EMA soft pseudo label 互教。逻辑上，SpML 与“多源泛化初始化”绑定较紧，WPOSNet 是另一个实用贡献，MMT 则是成熟框架复用。

4. 证据闭环：它用轻量指标和真实 Railway 数据补强 practical story。消融显示 whole 和 part 单独相近，合起来在 SpML 下 mAP 提升约 3 个点；MMT+SpML 比只用 MMT 高 5.2 个点，说明多源自步学习确实改善目标适配。ResNet50 和 WPOSNet 上都验证 SpML+MMT 有效，证明不是只靠某个骨干。GradCAM 显示 part branch 关注背包等局部，whole branch 补全上下文。

5. reviewer 为什么买账：这篇卖的是实用系统视角，机制新意不如 DMIC 或 MIP 紧。它的优势是把多源、轻量、目标适配和真实场景部署放在同一个问题框架里，并用 Params、FLOPs、Railway 数据支撑。reviewer 买账点是“现实约束下的 UDA ReID”，不是单个模块非常新。

**6. Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning**

1. 触发观察：VI-ReID 里很多方法都试图补齐或消除可见光和红外的信息差，比如生成、对齐、模态不变特征。作者反过来指出，完全桥接信息差很难，而且即使桥接，也不一定能稳定抓到身份判别线索。真正挑战是模型面对不同模态、不同实例、不同 query 时都要能灵活适配。

2. 重定义动作：它把“缩小 modality gap”重定义成“增强模型对 modality、input instance、query instance 的适应能力”。关键词是 model adaptability、flexibility、modality-aware prompts、instance-aware prompts、query-guiding prompts、adaptive feature space。这个转向很重要，因为它避开了“把两种模态硬映射到同一空间”的旧叙事。

3. 机制怎么长出来：MPL 为 visible 和 infrared 各维护一套 layer-wise prompts，让模型按模态适配。SIPG 从当前图像 embedding 生成 instance-aware prompts，解决不能为无限测试实例存独立 prompt 的问题。QIPG 从 query embedding 生成 query-guiding prompts，在额外 transformer 层里让特征提取适配当前匹配任务。CPF 用当前实例特征重加权 modality prompt，再和 instance prompt 融合，避免两类 prompt 冲突。IAEL 防止实例 prompt 退化成无差别提示，CEL 约束 query-guiding prompt 学到同身份相关而不是冗余信息。机制和重定义绑定很强，三类 prompt 分别对应三类适配对象。

4. 证据闭环：这篇的证据最像“方法论完整闭环”。它不只做组件消融，还专门验证普通 visual prompt 是否足够。结果显示 general prompts 只带来约 0.5 个点，而 MPL 和 SIPG 各自带来约 2.3 到 2.6 个点，二者合起来约 4 个点，证明提升不是“prompt tuning 本身”。它还比较 generation-based 和 fusion-based SIPG，t-SNE 显示 generation-based prompt 有更清楚的身份边界。CPF 优于直接拼接和投影，QIPG 加 CEL 进一步提升。CAM 显示 MPL、IPG 让模型更关注身体判别区域并减少无关细节。

5. reviewer 为什么买账：这篇主要卖视角，然后用机制兑现视角。真正 novelty 是把 VI-ReID 从“模态对齐”改写成“模型适配”，并把适配拆成模态、实例、query 三个轴。reviewer 容易买账，是因为它提前防住了几个质疑：普通 prompt 不够，实例 prompt 会退化，模态 prompt 和实例 prompt 会冲突，query prompt 需要约束。

**共同套路**

这批论文共同的构造方式是：先不说“我要加模块”，而是把 baseline 的失败解释成某个旧假设不成立，例如伪标签可靠、均值代理合理、只需模态对齐、token 越多越完整、单源初始化足够。然后把新方法做成这个新解释的直接后果，每个模块都对应一个失败来源。最能打动 reviewer 的不是最终涨点，而是有一组消融、伪标签质量、聚类数量、t-SNE、attention map 或真实场景测试，能证明“我重新定义的问题确实存在，而且我的机制正好在修它”。
tokens used
343,736
你这次列出的标题是 6 篇，不是 11 篇。我按这 6 篇拆；长文里方法超过前 8 页的，我补读了后续方法和关键消融页。

**1. Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID**

1. 触发观察：无监督 VI-ReID 不是简单缺标签，而是跨模态聚类会产生两类错。一类是同一身份被拆成多个簇，另一类是不同身份被错误合并。已有方法只做跨模态 cluster matching，却默认匹配关系可靠。作者还注意到 memory bank 用簇均值初始化，会被两模态样本数量不均和难样本拖偏。

2. 重定义动作：他们把问题从“怎么匹配可见光和红外簇”改成“跨模态匹配关系本身可以反过来诊断和修正伪标签”。关键词是 modality-shared pseudo labels、cluster relationship-based label refinement、weighted modality-shared memory、hard samples。这里的重点不是又做一个匹配算法，而是把匹配后的 many-to-one 关系解释成伪标签错误的信号。

3. 机制怎么长出来：先用 CSM 做双向 cluster similarity matching，得到跨模态伪标签。然后 CRLR 看哪些簇共同匹配到同一个异模态簇，如果同模态簇之间相似就合并；如果一个异模态簇被多个合并簇匹配，就用 k-means 把它切开。最后 WMM 不直接平均所有样本，而是先平衡两模态中心，再按样本到共享中心的关系给权重，让难样本在 memory 初始化里有更大作用。这个机制和重定义绑定较紧，因为每一步都在处理“伪标签关系不可靠”和“记忆中心不可靠”。

4. 证据闭环：它的关键不是只报涨点。消融里 baseline 到 CSM+MM 有大提升，CRLR 和 WMM 再分别增益；CRLR 还用伪标签质量指标证明 refinement 前后质量提高。t-SNE 显示跨模态簇边界变清，正负跨模态距离分布重叠变小，伪类数量逐渐接近真实类别数，ARI 也更高。这些证据直接服务于“伪标签关系能被修正”这个重定义。

5. reviewer 为什么买账：这篇卖的是“跨模态匹配关系可以用于 label refinement”的视角，加上一个完整训练框架。真正 novelty 不在某个复杂模块，而在把 cross-modality correspondence 从监督信号升级成诊断信号。审稿人会买账，是因为它把动机、机制和伪标签质量证据连起来了。

**2. Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification**

1. 触发观察：作者抓住无监督 ReID memory contrastive learning 的三个内部矛盾。第一，camera variation 会让同相机不同身份比跨相机同身份更近，Jaccard 聚类偏向错误邻居。第二，已有方法用 mean feature 当 cluster proxy，默认簇是球形，但 ReID 特征常是非球形流形，均值可能落在低密度空区。第三，one-hot hard pseudo label 对噪声不稳，会放大错误监督。

2. 重定义动作：他们没有泛泛说“伪标签有噪声”，而是把噪声拆成三个更具体的问题：距离度量不懂相机，代理点不符合 DBSCAN 的局部密度结构，监督目标太硬。关键词是 dynamic adaptive camera Jaccard、manifold、medoid proxy、dynamic adaptive proxies、soft pseudo labels。

3. 机制怎么长出来：DACJ 为每个样本估计跨相机和同相机邻居差异，动态增加跨相机正样本在 k-reciprocal 和 local query expansion 里的比例。DAP 用 medoid 代替 mean proxy，因为 medoid 是真实样本点，更贴合非球形流形；同时用 instance memory 补充簇内细粒度关系，并去掉动量均值更新。DAKD 再用 teacher-student 和 soft label 缓解 hard label 噪声。多视图的上半身、下半身、全局特征是补充判别信息。

4. 证据闭环：它的证据很对题。proxy 对比里 optimal medoid 优于 hardest、easiest、random；momentum update 和 optimal update 对比显示复杂 MSMT17 上差距很大；DOCP 和 DIP 单独作用小，合起来明显更强，证明两个代理层次互补。DACJ 的消融在 MSMT17 上尤其明显，说明相机差异越强越有用。t-SNE 显示同身份跨相机更紧，参数分析也围绕跨相机邻居比例展开。

5. reviewer 为什么买账：这篇卖的是“现有无监督 ReID 管线的几个默认假设错了”的视角。机制本身是多个组件，但每个组件都对应一个被重定义的假设错误。最有价值的是把“均值代理不适合 DBSCAN 发现的非球形簇”讲成了一个方法问题，而不是只说换个 proxy 涨点。

**3. Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification**

1. 触发观察：USL-VI-ReID 里，大家主要盯着 visible-infrared modality gap，但作者观察到 clustering 阶段还有 camera discrepancy。可见光图像受相机、光照、视角影响更大，导致同一身份被过度拆裂。已有 graph matching 或 optimal transport 如果建立在拆裂簇上，反而会放大 noisy labels。

2. 重定义动作：它把问题从“跨模态关联不准”改成“聚类过程被模态和相机信息同时污染”。关键词是 modality-camera invariant clustering、identity splitting、clustering-level discrepancy、from discriminability to generalization。这个表述比“减少模态差异”更具体，因为它把失败点锁定在聚类输入和聚类目标上。

3. 机制怎么长出来：MIE 在 k-reciprocal distance encoding 里均衡融合不同相机的信息，让 clustering embedding 不被同相机近邻主导。DNC 动态调 eps 和 k2，早期收缩半径排除噪声，先学可靠簇的判别性；后期扩大半径，把跨相机和跨模态正样本逐渐并进来。HMCL 先 intra-modality 再 inter-modality，用 cluster-level 和 instance-level memory 学模态相机不变表示。逻辑绑定很强，特别是 MIE 和 DNC 都直接服务于 identity splitting。

4. 证据闭环：关键证据是 clustering ablation 和 cluster number visualization。文本说明 MIE 带来约 10 到 15 个点 mAP，DNC 再带来 2 到 5 个点，合起来 14 到 17 个点。可视化里 visible cluster 数量更接近真实类别数，t-SNE 显示 MIE+DNC 后同身份跨模态更紧。objective ablation 里 inter-modality cluster loss 贡献最大，说明改好的跨模态伪标签确实在驱动训练。

5. reviewer 为什么买账：这篇卖的是视角，不只是模块。它把 USL-VI-ReID 的痛点从“模态对齐”推进到“模态和相机共同破坏聚类”。如果 reviewer 接受 identity splitting 是主要失败模式，MIE 和 DNC 就显得自然。它的 novelty 来源是把 camera-aware 思路放到 VI 全局聚类阶段，而不是只做相机代理损失。

**4. Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification**

1. 触发观察：AGPReID 的难点不只是 aerial 和 ground 视角差异，还包括大范围观测带来的尺度变化、背景冗余和无关区域。已有 VDT 能做 view decoupling，但仍然把大量 token 都送进 transformer，缺少对关键身份区域的动态选择。

2. 重定义动作：它把 aerial-ground ReID 从“视角差异问题”改写成“视角差异下的 token 冗余和关键区域选择问题”。关键词是 dynamic token selection、top-k informative tokens、compact long-range modeling、irrelevant tokens、view-heterogeneous ReID。

3. 机制怎么长出来：在 VDT 框架上保留 meta token 和 view token，用 identity、view classifier 和 orthogonal loss 做 view-related 与 view-unrelated disentanglement。新增 VTS，用轻量 attention 给每个 token 打分，取 top-K token，再用 Gumbel-Softmax 近似使选择可反传。最后选中的 token 和 class token 进入后续聚合。机制和重定义有一定绑定，但更多是把已有 token selection 思路迁移到 AGPReID。

4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。

5. reviewer 为什么买账：这篇更多卖机制迁移，而不是强视角。AGPReID 是新任务，VDT 是清晰强基线，在这个基线上加动态 token selection 有明确增益，reviewer 容易接受。真正 novelty 是“AGPReID 里显式选择关键 token”，但方法论上它不是最强的创新构造样本。

**5. Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning**

1. 触发观察：UDA ReID 常用目标域聚类和伪标签，但源域知识利用不足；DG 学泛化，却不直接适配目标域。现实部署还有边缘设备算力、隐私和低资源限制。另一方面，global-only feature 在遮挡、光照和低清场景下不够细。

2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。

3. 机制怎么长出来：WPOSNet 用 OSNet 的轻量深度可分离卷积，做 whole branch 和 part branch，垂直切分人体局部再拼接。SpML 把每个源域看成任务，用 SuA 扰动特征均值和方差形成不同难度，从无扰动、弱扰动到强扰动，模拟由易到难学习源域不变知识。最后用 MMT 在目标域做 hard pseudo label 和 EMA soft pseudo label 互教。逻辑上，SpML 与“多源泛化初始化”绑定较紧，WPOSNet 是另一个实用贡献，MMT 则是成熟框架复用。

4. 证据闭环：它用轻量指标和真实 Railway 数据补强 practical story。消融显示 whole 和 part 单独相近，合起来在 SpML 下 mAP 提升约 3 个点；MMT+SpML 比只用 MMT 高 5.2 个点，说明多源自步学习确实改善目标适配。ResNet50 和 WPOSNet 上都验证 SpML+MMT 有效，证明不是只靠某个骨干。GradCAM 显示 part branch 关注背包等局部，whole branch 补全上下文。

5. reviewer 为什么买账：这篇卖的是实用系统视角，机制新意不如 DMIC 或 MIP 紧。它的优势是把多源、轻量、目标适配和真实场景部署放在同一个问题框架里，并用 Params、FLOPs、Railway 数据支撑。reviewer 买账点是“现实约束下的 UDA ReID”，不是单个模块非常新。

**6. Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning**

1. 触发观察：VI-ReID 里很多方法都试图补齐或消除可见光和红外的信息差，比如生成、对齐、模态不变特征。作者反过来指出，完全桥接信息差很难，而且即使桥接，也不一定能稳定抓到身份判别线索。真正挑战是模型面对不同模态、不同实例、不同 query 时都要能灵活适配。

2. 重定义动作：它把“缩小 modality gap”重定义成“增强模型对 modality、input instance、query instance 的适应能力”。关键词是 model adaptability、flexibility、modality-aware prompts、instance-aware prompts、query-guiding prompts、adaptive feature space。这个转向很重要，因为它避开了“把两种模态硬映射到同一空间”的旧叙事。

3. 机制怎么长出来：MPL 为 visible 和 infrared 各维护一套 layer-wise prompts，让模型按模态适配。SIPG 从当前图像 embedding 生成 instance-aware prompts，解决不能为无限测试实例存独立 prompt 的问题。QIPG 从 query embedding 生成 query-guiding prompts，在额外 transformer 层里让特征提取适配当前匹配任务。CPF 用当前实例特征重加权 modality prompt，再和 instance prompt 融合，避免两类 prompt 冲突。IAEL 防止实例 prompt 退化成无差别提示，CEL 约束 query-guiding prompt 学到同身份相关而不是冗余信息。机制和重定义绑定很强，三类 prompt 分别对应三类适配对象。

4. 证据闭环：这篇的证据最像“方法论完整闭环”。它不只做组件消融，还专门验证普通 visual prompt 是否足够。结果显示 general prompts 只带来约 0.5 个点，而 MPL 和 SIPG 各自带来约 2.3 到 2.6 个点，二者合起来约 4 个点，证明提升不是“prompt tuning 本身”。它还比较 generation-based 和 fusion-based SIPG，t-SNE 显示 generation-based prompt 有更清楚的身份边界。CPF 优于直接拼接和投影，QIPG 加 CEL 进一步提升。CAM 显示 MPL、IPG 让模型更关注身体判别区域并减少无关细节。

5. reviewer 为什么买账：这篇主要卖视角，然后用机制兑现视角。真正 novelty 是把 VI-ReID 从“模态对齐”改写成“模型适配”，并把适配拆成模态、实例、query 三个轴。reviewer 容易买账，是因为它提前防住了几个质疑：普通 prompt 不够，实例 prompt 会退化，模态 prompt 和实例 prompt 会冲突，query prompt 需要约束。

**共同套路**

这批论文共同的构造方式是：先不说“我要加模块”，而是把 baseline 的失败解释成某个旧假设不成立，例如伪标签可靠、均值代理合理、只需模态对齐、token 越多越完整、单源初始化足够。然后把新方法做成这个新解释的直接后果，每个模块都对应一个失败来源。最能打动 reviewer 的不是最终涨点，而是有一组消融、伪标签质量、聚类数量、t-SNE、attention map 或真实场景测试，能证明“我重新定义的问题确实存在，而且我的机制正好在修它”。
