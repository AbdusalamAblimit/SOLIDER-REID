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
session id: 019ef835-d033-7ea3-b6b2-a003a7c1d727
--------
user
你是 ReID 创新方法论拆解员 18。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(18)==
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认当前目标和状态，再用 `pdftotext` 逐篇读方法部分。因为这是 PDF 阅读任务，我也会先看一下本地 PDF 技能的约束，避免漏掉必要的提取和核对步骤。
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
/bin/zsh -lc "sed -n '1,220p' ../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../CLAUDE.md: No such file or directory

exec
/bin/zsh -lc 'pwd && rg --files' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
manifest.csv
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
/bin/zsh -lc "sed -n '1,220p' ../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../GOAL.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
刚才 `STATUS.md` 读到了，但 `GOAL.md` 和 `CLAUDE.md` 的相对路径少上了一层。我先补读这两个文件，然后开始提取这 6 篇指定论文的方法部分。
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

exec
/bin/zsh -lc "sed -n '1,220p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,220p' ../../../AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
我已经对齐了当前目标，和用户这次要求一致，都是读文献和拆方法论，不启动实验。接下来我先定位每篇的方法章节边界，再读取对应正文，这样不会只停在摘要和引言。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|adapter|domain|mixture|parameter'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|mix|modality|paradigm'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:     1	MoDA: Mixture of Domain Adapters for Parameter-efficient
8:     8	The Domain Generalizable Re-identification (DG ReID) task has attracted significant attention in recent
9:     9	years, as a challenging task but closely aligned with practical applications. Mixture-of-experts (MoE)-based
10:    10	methods have been studied for DG ReID to exploit the discrepancies and inherent correlations between diverse
11:    11	domains. However, most of DG ReID methods, especially MoE-based methods, have to fully fine-tune a large
12:    12	amount of parameters, which are not always practical in real-world scenarios. Considering this problem, we
13:    13	propose a novel MoE-based DG ReID method, named Mixture of Domain Adapters (MoDA), which utilizes
14:    14	many expert adapters and a global adapter to help MoE-based method scale to a much larger model but in a
15:    15	more parameter-efficient way. Furthermore, we conduct our approach with the large-scale vision-language
17:    17	based on multimodal information. Extensive experiments verify the effectiveness of our method and show that
18:    18	MoDA achieves competitiveness with state-of-the-art DG ReID methods with much fewer tunable parameters.
19:    19	CCS Concepts: • Computing methodologies → Computer vision; Visual content-based indexing and
21:    21	Additional Key Words and Phrases: Generalizable Person Re-Identification, Domain Generalization, Parameterefficient Fine-tuning
23:    23	Yang Wang, Yixing Zhang, Xudie Ren, and Yuxin Deng. 2025. MoDA: Mixture of Domain Adapters for
24:    24	Parameter-efficient Generalizable Person Re-identification. ACM Trans. Multimedia Comput. Commun. Appl.
30:    30	Introduction
44:    44	Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
55:    55	There have been many works that can improve great performance on ReID benchmarks in the conventional scenario [8, 29, 32, 48]. However, when these methods are confronted with a completely
56:    56	unseen domain, the performance drops significantly. This phenomenon is commonly attributed
57:    57	to domain shift and domain conflict. When ReID comes to Domain Generalization (DG), the
58:    58	task becomes even more difficult. The model can only use images from source domains to optimize and is not allowed to access any target domain image during the training period. This task
60:    60	robustness when confronted with out-of-distribution data. To tackle this problem, some Domain
61:    61	Adaptive (DA) and Domain Generalizable ReID (DG ReID) methods are proposed. DA methods
62:    62	[6, 16, 42] can access a part of target domain data and then try to adapt the model which is already
63:    63	trained with source domain data to the target domain [36].
66:    66	In the context of DG ReID, the training and testing of models are performed on diverse domains,
67:    67	which are respectively referred to as the source domain and target domain. Many prior DG ReID
68:    68	methods [28, 44] only utilize one individual model and train the model on a hybrid dataset that
69:    69	consists of samples from different source domains. And then the model is directly tested on unseen target domains. These methods achieve good performance by extracting domain-invariant
71:    71	domains, which may provide more discriminative and complementary information to help generalize better. For this reason, Mixture-of-experts (MoE)-based methods [2, 36] have been studied
72:    72	for DG ReID. A common framework of MoE-based methods is to train domain-specific expert
73:    73	networks on each source domain and then these methods integrate multiple experts by calculating
74:    74	the relevance of the test sample and source domains to get one aggregated feature. The existing
75:    75	MoE-based DG ReID methods get better performance but have the common issue that the number
76:    76	of model parameters scales linearly with the number of source domains due to the increase of the
77:    77	number of experts as shown in Figure 1. Although there are some methods [36, 39] that try to
78:    78	minimize the number of expert parameters, there are still a large amount of trainable parameters
79:    79	need to be updated. All these existing methods have to fully fine-tune the entire model including the backbone and experts. And it is not practical and efficient enough in many real-world
81:    81	large, the number of parameters of classifier heads significantly increases. These MoE-based methods always need at least one classifier for each source domain and one global classifier to learn
82:    82	domain-invariant feature and to optimize the model. And it leads to a certain degree of parameter
84:    84	For the motivation to solve the issues mentioned above, we propose a novel DG ReID method
85:    85	which is parameter-efficient and can be scaled to a much larger backbone model, named Mixture
86:    86	of Domain Adapters (MoDA). We apply the large-scale vision-language model CLIP [27] and
87:    87	Adapter [10] in our method. CLIP is a powerful pretrained model which aligns texts and images
89:    89	representations against the challenge of domain shift. Moreover, the contrastive losses that CLIP
91:    91	loss, which means that it may substitute the ID loss of ReID to reduce the classifier head parameters.
93:    93	computer vision DG tasks. In addition, Adapter is a bottleneck module used for Parameterefficient Fine-tuning (PEFT) and will be used as domain-specific experts in MoDA. As Figure 2
94:    94	shows, different from prior CNN-based DG ReID methods, the adapters (experts) are inserted in
98:    98	MoDA for Parameter-efficient Generalizable Person Re-ID
102:   102	Fig. 1. Difference between prior MoE-based DG ReID methods and our more parameter-efficient method.
103:   103	Upper: Prior MoE-based DG methods need to optimize the whole model including the backbone and experts,
104:   104	parameters of which usually increase linearly with the increasing of the number of person IDs. Below: In our
105:   105	method, the backbone is frozen and only adapters, the parameters of which are not related with the person
108:   108	Fig. 2. We show the architecture of the adapter, ViT block with adapter, and our method MoDA block. The
109:   109	standard architectures of adapter and ViT block are shown within (a) and (b), respectively. In our proposed
110:   110	MoDA block, we set Adapters after the multi-head attention layer to implement an MoE architecture as (c)
111:   111	shows. An additional block-aware voting network is set to mix the adapters. Only the parameters of adapters
115:   115	integrate expert adapters in a more fine-grained way by generating aggregation weight for each
117:   117	parameters in the model network.
126:   126	model CLIP to tackle the DG problem in DG ReID. Our approach uniquely integrates CLIP to
127:   127	address domain shifts by aligning text and image features in a shared space, thereby enhancing
130:   130	methods, specifically Adapters, within the context of DG ReID. Adapters serve as domainspecific experts in our framework, demonstrating their efficacy in reducing the number of
131:   131	tunable parameters while maintaining performance.
132:   132	— Introduction of MoDA and Block-aware Voting Network. Based on the exploration of CLIP
134:   134	parameter efficiency while scaling MoE-based DG ReID methods to larger vision models.
136:   136	fine-grained integration of expert adapters across different blocks, optimizing the model’s
138:   138	— Empirical Validation of MoDA’s Effectiveness. Comprehensive experiments validate that MoDA
140:   140	methods in a more parameter-efficient manner. Our results underscore the effectiveness of
144:   144	Related Works
146:   146	DG Person ReID. Most of the existing ReID methods suffer from significant performance degradation
148:   148	generalization ability since it cannot access any data of target domains. To tackle the domain
149:   149	conflict problem we mentioned above, many DG ReID methods are proposed. DG ReID methods
150:   150	are required to be trained on accessible source domains and then tested on unseen target domains.
151:   151	The most prior approaches [1, 14, 28, 44] typically involve combining all source domains into a
153:   153	domain. These methods aim to learn a common feature space which works for all diverse domains.
156:   156	the DG scenario where no target domain data are accessible during training period. They learn
157:   157	domain-invariant features but tend to overlook domain-specific features and the correlations among
158:   158	them. (3) Normalization-based methods [1, 14], which is to investigate the statistical distribution
159:   159	discrepancies across different domains, for instance, by minimizing the domain gap through domain
160:   160	alignment. However, these methods may ignore the discriminative characteristics of each domain
163:   163	proposed method RaMoE integrates every domain-specific expert feature as one single aggregated
164:   164	feature according to the target domain’s inherent relevance w.r.t. source domains. Building upon
166:   166	domains. It also incorporates a global branch to combine domain-specific and domain-invariant
167:   167	representations. ACL [39] also integrates the design fashion of MoE into their method. However,
168:   168	the existing methods need to fully fine-tune the whole models with a large amount of parameters
169:   169	including the backbones and experts. Our method aims to mitigate this issue.
172:   172	MoDA for Parameter-efficient Generalizable Person Re-ID
183:   183	prompt-tuning method CoOp [50], is the first study to apply CLIP for ReID tasks. Some prior work
186:   186	PEFT. Conventional approaches usually fine-tune all the parameters (full fine-tuning) of pretrained models. Due to the significantly increasing overhead of full fine-tuning VLP models, more
187:   187	and more PEFT methods are proposed. These methods only fine-tune a small number of parameters
189:   189	parameters as much as possible with sacrificing a little bit performance or even outperforming
190:   190	their full fine-tuning counterparts. Many methods have been widely applied in NLP tasks for large
193:   193	during training. LoRA [11] tries to learn low-rank matrices to approximate parameter updates.
194:   194	Authors in [10] insert small bottleneck modules named adapters to each layer of the pretrained
195:   195	model and only fine-tune the adapters during training. Recently, parameter-efficient methods are
199:   199	uses adapters to adapt more image models to video tasks without full fine-tuning.
201:   201	on models with adapters, such as AdaMix [33] and MixDA [3], they actually employ distinct
202:   202	FC-Down and FC-Up layers within one single adapter structure and primarily utilize a randomly
203:   203	routing mechanism or simply generate weights to choose and integrate adapters. While we use a
205:   205	used for common NLP tasks and MixDA tries to inject domain knowledge into language models.
206:   206	Neither of these methods specifically address DG vision tasks, specifically exploitation of CLIP for
211:   211	Method
214:   214	In our work, we adopt a similar two-stages training approach as CLIP-ReID [17] as shown in Figure 3.
221:   221	information of each ID. And in our method, these tokens will be treated as domain prototypes for
222:   222	the second stage to compute the relevance w.r.t. different source domains. Notably, only tokens
232:   232	within each source domain are treated as specific prototypes of the corresponding domain, and they will
234:   234	pick one domain to simulate the unseen target domain, and the batch data will be sent to three branches to
235:   235	compute responding losses, respectively. As shown in the figure, we choose Domain 1 as the mimic target
236:   236	domain. The expert branch (a) and global branch (b) are used to optimize the expert adapter and the global
237:   237	adapter, respectively, and the branch (c) is used for the voting network. In the whole procedure, only the
238:   238	parameters of the expert adapters, the global adapter and the voting network are tunable. While all other
239:   239	parameters are frozen.
276:   276	In the second stage, we propose an MoE model with expert adapters and global adapters, which
277:   277	is to extract domain-specific features and domain-invariant features, respectively. Different from
282:   282	MoDA for Parameter-efficient Generalizable Person Re-ID
284:   284	CLIP-ReID, we adopt adapter [10] due to its simplicity and parameter-efficiency rather than full
286:   286	abundant parameters in ViT as well as the scalability limitation of MoE-based approaches on
287:   287	larger models. In addition, adapters also help prevent catastrophic forgetting from insufficient
289:   289	Figure 2(a) shows, adapter is a simple bottleneck module with two FC layers and one activation
290:   290	unit GELU. During training period, we can only access the datasets from source domains to train
291:   291	the DG model. And the optimized DG model will be test on unseen target domain directly. We
292:   292	𝐾 . And for the kth domain, there are
293:   293	assume there are K source domains, denoted as 𝐷 = {𝐷𝑘 }𝑘=1
297:   297	Domain-specific Adapters
299:   299	As well as the prior MoE-based methods for DG ReID, our method assigns an individual adapter as
300:   300	the domain expert of each source domain, denoted as 𝐴𝑘 standing for the domain-specific adapter
301:   301	of the kth domain. We aim for each expert to capture characteristic features of its corresponding
302:   302	source domain, which are unique but can provide complementary information different from other
303:   303	domains.
304:   304	In every ViT block, we set K adapters after the multi-head attention layer as Figure 3 shows.
305:   305	Expert Adapter. For a kth domain image from the nth ID, we denote it as 𝑥𝑛𝑘 . As Figure 3(a) shows,
306:   306	for each 𝑥𝑛𝑘 , we let it go through ViT blocks with only the corresponding domain adapter 𝐴𝑘 to
307:   307	obtain domain-specific feature 𝐹𝑥𝑘 . The computation of each expert block can be written as:
338:   338	where 𝑇𝑘 are just the ID-specific tokens in the corresponding kth domain of the current image.
339:   339	Notably, we find that with adapters, the usage of ID loss L𝑖𝑑 has little influence on model performance, which will be discussed in Section 4.4 and shown in Table 8. Intuitively, we speculate the
340:   340	reason might be that the image-to-text loss L𝑖2𝑡 and ID loss L𝑖𝑑 proposed by CLIP are both inherently cross-entropy losses, thus can provide approximate constraints and optimization capabilities.
341:   341	However, the additional classifier head leads to substantially more tunable parameters under the
342:   342	DG ReID experiment configuration. Therefore, to additionally enhance the parameter-efficiency of
344:   344	Triplet loss is still employed for expert adapter in our methodology:
350:   350	Meanwhile, 𝛼 denotes the margin parameter of the triplet loss due to [25].
357:   357	Thus, the loss for expert adapter is formulated as:
366:   366	where the superscript e which stands for expert is to distinguish from the losses for global adapter
368:   368	Global Adapter. Except for the domain-specific experts, we also maintain an individual global
369:   369	adapter to learn a common feature space to extract domain-invariant representations. Inspired by
370:   370	META [36], we similarly utilize a global adapter as the global branch to extract domain-invariant
371:   371	features. The extracted features are invariant to source domain shifts and more robust compared
372:   372	to any individual domain adapter. As Figure 3 Stage 2(b) branch shows, for each sample x from
374:   374	any domain, we pass it through the global adapter to obtain global feature 𝐹𝑥 . The computation of
375:   375	each global adapter block is actually the same as the expert adapter blocks. But we will combine all
376:   376	text features from different domains together to perform L𝑖2𝑡𝑐𝑒 . And the loss for global adapter is
386:   386	Notably, during the training process, the parameters of the ViT backbone are frozen and shared
387:   387	across different adapters, only the parameters of adapters need to be optimized.
392:   392	Now, we have both domain-specific and domain-invariant features, but the target domain is
393:   393	inaccessible during DG ReID training. To improve generalization on unseen domains, we propose a
394:   394	novel MoE-based approach specially for ViT with adapters to integrate domain-specific features
396:   396	In most existing MoE-based methods, they learn diverse experts for different source domains
397:   397	and then calculate the relevance between the test image which is from unseen target domains and
398:   398	source domains. Subsequently, as guided by the relevance, multiple source domain features are
399:   399	mixed to create a new aggregated feature. In prior CNN-based approaches, the standard voting
401:   401	with prototypes of different source domains. The class centers of domain are usually used as its
402:   402	prototypes. However, different from CNN structure with backbone plus expert heads, in our ViTbased method, to obtain the final features from k experts, we have to forward the whole ViT model
403:   403	k times. This is due to our domain adapters are positioned within each block (layer) of the model
406:   406	features from multiple domain-specific experts at the block level. The kind of block-wise approach,
407:   407	compared to the sample-wise approach, also makes the model conduct a more fine-grained feature
410:   410	for each identity. Additionally, tokens from each source domain will be treated as domain-specific
411:   411	prototypes for their corresponding domains. These prototypes will be utilized to describe the
412:   412	characteristics of different source domains.
414:   414	scenario where target domains are unseen. For a sample from the kth domain, we denote it as 𝑥 𝑘 .
416:   416	pass to blocks with the kth domain-specific adapters 𝐴𝑘 , then compute the expert loss L𝑒𝑥𝑝𝑒𝑟𝑡 .
417:   417	And in this iteration, the kth domain will be seemed as the unseen target domain for simulation,
420:   420	and the remaining 𝐾 − 1 domains {𝐷𝑖 }𝑖=1,𝑖≠𝑘
421:   421	will be seemed as source domains {𝐷𝑖𝑠 }𝑖=1

 succeeded in 0ms:
1:     1	Mix-Modality Person Re-Identification: A New and
2:     2	Practical Paradigm
12:    12	Current visible-infrared cross-modality person re-identification research has only focused on exploring the
13:    13	bi-modality mutual retrieval paradigm, and we propose a new and more practical mix-modality retrieval
14:    14	paradigm. Existing Visible-Infrared Person Re-Identification (VI-ReID) methods have achieved some
15:    15	results in the bi-modality mutual retrieval paradigm by learning the correspondence between visible and
16:    16	infrared modalities. However, significant performance degradation occurs due to the modality confusion
17:    17	problem when these methods are applied to the new mix-modality paradigm. Therefore, this article proposes a
18:    18	Mix-Modality Person Re-Identification (MM-ReID) task, explores the influence of modality mixing ratio
19:    19	on performance, and constructs mix-modality test sets for existing datasets according to the new mix-modality
20:    20	testing paradigm. To solve the modality confusion problem in MM-ReID, we propose a Cross-Identity
23:    23	samples with different identities while aggregating samples with the same modality and the same identity.
24:    24	Furthermore, we propose a Modality Bridge Similarity Optimization Strategy (MBSOS) to optimize
25:    25	the cross-modality similarity between the query and queried samples with the help of the similar bridge
41:    41	Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
52:    52	sample in the gallery. Extensive experiments demonstrate that compared to the original performance of
53:    53	existing cross-modality methods on MM-ReID, the addition of our CIDHL and MBSOS demonstrates a general
55:    55	CCS Concepts: • Computing methodologies → Artificial intelligence; Computer vision;
56:    56	Additional Key Words and Phrases: Cross-Modality Person Re-identification, Mix-Modality Paradigm, Metric
59:    59	Wei Liu, Xin Xu, Hua Chang, Xin Yuan, and Zheng Wang. 2025. Mix-Modality Person Re-Identification: A
60:    60	New and Practical Paradigm. ACM Trans. Multimedia Comput. Commun. Appl. 21, 4, Article 112 (March 2025),
66:    66	Introduction
70:    70	Single-Modality ReID (SM-ReID) methods focusing on visible image retrieval have achieved
76:    76	tackle the challenge of cross-modality image matching [9, 14, 16, 38].
79:    79	arising from modality differences. Despite considerable progress achieved in the bi-modality mutual
80:    80	retrieval paradigm, as illustrated in Figure 1(a) through learning potential correspondences between
84:    84	a collection of one modality type. This integration often leads to a mix of what seems to be
85:    85	straightforward SM-ReID tasks into the existing cross-modality framework. However, as indicated
86:    86	in Figure 1(c), this approach results in a marked performance degradation, primarily due to the
87:    87	“Modality Confusion” problem. This issue stems from identity-independent features such as colors
88:    88	being more similar within the same modality, which confuses the matching of cross-modality
89:    89	identity information. More specifically, the impact of modality confusion on the current approach
90:    90	of learning only visible-infrared cross-modality correspondences is huge, due to the fact that there
92:    92	same-modality correspondences, which is not taken into account by the existing approaches of
93:    93	VI-ReID. Under this influence, the natural similarity between samples of the same modality will
95:    95	identities of the same modality being smaller than the distance between samples of the same
97:    97	challenges, we introduce a new and practical Mix-Modality Re-ID (MM-ReID) task, creating
98:    98	mixed modality test sets for existing datasets and examining how the ratio of modality mixing
102:   102	Mix-Modality Person Re-Identification: A New and Practical Paradigm
106:   106	Fig. 1. (a) Existing bi-modality mutual retrieval test paradigms for VI-ReID use a visible probe image queried
107:   107	in an infrared image gallery or an infrared probe image queried in a visible image gallery. (b) Our proposed
108:   108	mix-modality testing paradigm for MM-ReID uses mix-modality probes to query in a mixed-modality gallery.
110:   110	modality. In the figure, blue/gray represents visible/infrared modality samples, respectively, and different
112:   112	information, the distance between samples of the same modality but different identities is closer than that
114:   114	retrieval and reduces the accuracy. MM-ReID, mix-modality person re-identification; VI-ReID, visible-infrared
117:   117	To address the modality confusion challenge in MM-ReID, we introduce the Cross-Identity
118:   118	Discrimination Harmonization Loss (CIDHL). This novel approach optimizes the sample
120:   120	samples of the same identity and modality towards their respective centers, while simultaneously
122:   122	those of different identities, irrespective of modality. This strategy mitigates the effects of modality
123:   123	confusion. Additionally, we propose a Modality Bridge Similarity Optimization Strategy
124:   124	(MBSOS), which optimizes the cross-modality distance metric by identifying a similar bridge
126:   126	the query and the queried samples. Extensive testing confirms that our methods—CIDHL and
127:   127	MBSOS—significantly enhance the model performance of existing cross-modality approaches,
128:   128	demonstrating their efficacy and adaptability in resolving modality confusion in MM-ReID.
130:   130	— Paradigm Contribution: We pioneer the MM-ReID task, introducing a novel testing paradigm
132:   132	— Empirical Contribution: We investigate the impact of varying modality mixing ratios on crossmodality model performance and conclude the unique challenges of “Modality Confusion”
140:   140	— Technical Contribution: We introduce the CIDHL and the MBSOS. These transferable methodologies effectively address modality confusion, as validated by extensive empirical experiments.
144:   144	Related Work
145:   145	Single-Modality Re-ID (SM-ReID)
149:   149	focus of existing approaches has been on addressing changes in pose, viewpoint, and occlusion
156:   156	was first proposed by Wu et al. [33], addresses the cross-modality matching of visible and infrared
163:   163	This bi-modality mutual search paradigm confronts not only the common challenges of SM-ReID,
166:   166	deep learning strategies have been devised to mitigate both inter- and intra-modality variations.
169:   169	to a refined approach that incorporates both set-level and instance-level alignment [27]. Zhao et al.
173:   173	learnable prototypes, using affinity information to refine cross-modality identification. Traditional
175:   175	method to enhance image diversity by swapping and erasing color channels randomly. For localized feature extraction, Wei et al. [31] utilized an adversarial learning framework with a flexible
177:   177	technology, Liang et al. [12] adopted this approach for cross-modality ReID tasks, focusing on
178:   178	extracting highly discriminative features through modality-aware enhancement. In terms of modality transformation, Wei et al. [32] proposed synthesizing new modalities by merging features
180:   180	generating intermediate modalities to minimize differences, with Zhong et al. [48] experimenting
185:   185	mix-modality images of specific pedestrians in a mix-modality gallery, rather than just a simple
188:   188	Mix-Modality Person Re-Identification: A New and Practical Paradigm
193:   193	different identities, blue/gray represents visible/infrared modality samples, while green and red lines represent
195:   195	between different identities (shapes); (b) VI-ReID needs to face the challenge of modality (color) differences in
196:   196	addition to identity differences; and (c) MM-ReID needs to face the challenge of modality confusions (similar
197:   197	modalities possessing a closer proximity) in addition to identity and modality differences.
199:   199	bi-modality mutual retrieval paradigm adopted by existing methods. Therefore, we propose an
200:   200	MM-ReID task according to the mix-modality paradigm.
203:   203	Methodology
205:   205	In this section, we elaborate on the novel MM-ReID paradigm and introduce two cornerstone
206:   206	methodologies: the CIDHL and the MBSOS. More specifically, Section 3.1 provides the task definition
207:   207	and formulation of our proposed MM-ReID together with an analysis of the unique modality
209:   209	design motivation for the method and the differences with existing methods, Section 3.3 introduces
219:   219	containing 𝑁 g gallery images. As depicted in Figure 1, under the traditional bi-modality mutual
220:   220	retrieval paradigm of VI-ReID, the sets Q and G each consist of images exclusively in one modality,
221:   221	either visible or infrared. Contrary to this, our proposed MM-ReID paradigm includes images in
227:   227	spectra, a process hampered by substantial differences in their imaging mechanisms. These modality differences represent significant obstacles to effective cross-modality matching. Furthermore,
229:   229	contends with “modality confusion,” where similarities irrelevant to identity, such as color consistency within a modality, can obscure vital identity-specific cues between the query (Q) and gallery
237:   237	Fig. 3. An illustration of our proposed CIDHL and MBSOS. The mix-modality data is fed into two feature
239:   239	between the centers of the cross-modality same identity samples is pulled closer together pushing away the
247:   247	Method Design Motivation
249:   249	To tackle the distinct modality confusion problem in the MM-ReID task, this article introduces
250:   250	two innovative methods: CIDHL and MBSOS. CIDHL specifically targets the confusion arising
251:   251	from similarity among same-modality samples by constraining these similarities in the metric
253:   253	modality when attempting to match cross-modality, same-identity positive samples. Moreover,
254:   254	to mitigate modality confusion effects during the testing phase, MBSOS optimizes the similarity
255:   255	measure between cross-modality samples by leveraging same-modality samples in the gallery
256:   256	as a bridge. This is a departure from traditional VI-ReID methods, which typically focus only
257:   257	on constraining or exploiting the similarity across modalities. Our approach aims to refine both
259:   259	posed by same-modality samples unique to the MM-ReID task.
264:   264	3.3.1 Feature Presentation Extractor. Our proposed methods, CIDHL and MBSOS, are designed to
265:   265	seamlessly integrate with the feature extraction frameworks of existing VI-ReID methods. This integration ensures that there is no need for modifications to the baseline feature extraction processes
266:   266	of these methods. As illustrated in Figure 3, we employ a generalized baseline feature representation
271:   271	varied modality inputs.
274:   274	Mix-Modality Person Re-Identification: A New and Practical Paradigm
280:   280	triplet hard loss [6]. This approach refines the basic triplet loss by dynamically selecting the most
338:   338	approach helps stabilize the training process by reducing the influence of outliers or mislabeled
340:   340	associated with that identity. This average is computed separately for each modality, ensuring that
343:   343	providing a more reliable basis for optimizing the model. This method not only mitigates the risk
365:   365	where 𝑣 i𝑗 and 𝑡 i𝑗 denote the jth image of ith identity within visible and infrared modality, while the
377:   377	centers of the same identity, regardless of the modality. This step ensures that the same identity
379:   379	— Maximize Inter-Identity Distances: Simultaneously, it is crucial to increase the distance between feature centers of different identities within the same modality and across different
381:   381	— Aggregate Intra-Modality Distances: To further refine the model’s effectiveness, distances
382:   382	between samples of the same modality and identity are aggregated towards their respective
383:   383	centers. This aggregation aids in minimizing the modality-specific variations within the same
386:   386	The CIDHL adopts a structured approach, containing two parts, the first of which is targeted:
387:   387	— Cross-Modality Attraction: Distances between centers of the same identity across modalities
389:   389	— Intra- and Inter-Modality Repulsion: Distances between centers of different identities, whether
390:   390	within the same modality or across modalities, are pushed apart to enhance discriminability.
433:   433	center of samples with the same modality and identity for solving the modality confusion problem

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|hairstyle|cloth|MSP|reid'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:     1	MSP-REID: HAIRSTYLE-ROBUST CLOTH-CHANGING PERSON RE-IDENTIFICATION
11:    11	ABSTRACT
12:    12	Cloth-Changing Person Re-Identification (CC-ReID) aims to
13:    13	match the same individual across cameras under varying clothing
14:    14	conditions. Existing approaches often remove apparel and focus on
15:    15	the head region to reduce clothing bias. However, treating the head
17:    17	over-reliance on volatile hairstyle cues, causing performance degradation under hairstyle changes. To address this issue, we propose the
18:    18	Mitigating Hairstyle Distraction and Structural Preservation (MSP)
19:    19	framework. Specifically, MSP introduces Hairstyle-Oriented Augmentation (HSOA), which generates intra-identity hairstyle diversity
20:    20	to reduce hairstyle dependence and enhance attention to stable facial and body cues. To prevent the loss of structural information,
21:    21	we design Cloth-Preserved Random Erasing (CPRE), which performs ratio-controlled erasing within clothing regions to suppress
24:    24	parsing-guided priors that highlight face and limb regions while suppressing hair features. Extensive experiments on multiple CC-ReID
25:    25	benchmarks demonstrate that MSP achieves state-of-the-art performance, providing a robust and practical solution for long-term person
27:    27	Index Terms— Cloth-Changing Person Re-Identification(CCReID), hairstyle augmentation, clothing-structure preservation,
29:    29	1. INTRODUCTION
33:    33	clothing typically remains unchanged [3]. However, models trained
34:    34	under such settings tend to overfit clothing appearance, causing
36:    36	or different people wear visually similar clothes. This limitation
37:    37	has motivated research on cloth-changing person re-identification
38:    38	(CC-ReID), which aims to ensure robust identity matching under
39:    39	clothing variations and better satisfies the requirements of long-term,
41:    41	Beyond clothing changes, factors such as hairstyle variations
43:    43	salient but identity-irrelevant features. Existing CC-ReID methods
44:    44	typically reduce clothing dependence through semantic mining, feature
47:    47	remains largely overlooked: the impact of hairstyle variations on
74:    74	(d) Clothing erasure
76:    76	Fig. 1. The Hairstyle Shortcut problem in CC-ReID. (a) Standard parsing merges face and hair together as "head"; (b) attention
78:    78	are robust to clothing changes but brittle to hairstyle variations; (d)
79:    79	conventional clothing erasure further removes structural cues.
80:    80	as identity-related [9, 14] (Fig. 1(a)), causing models to overemphasize
81:    81	head regions and become highly sensitive to hairstyle cues (Fig. 1(b)).
82:    82	This introduces a "hairstyle shortcut" whereby models mistakenly rely
83:    83	on hairstyle as the primary identity cue, degrading their generalization
84:    84	ability when hairstyles change (Fig. 1(c)).Furthermore, parsing-based
85:    85	clothing removal methods are often too aggressive [14], eliminating
86:    86	not only clothing pixels but also crucial structural cues—such as body
90:    90	Hairstyle Distraction and Structural Preservation (MSP) framework
91:    91	to simultaneously address these limitations of existing CC-ReID
92:    92	methods: the over-reliance on hairstyle cues and the loss of structural
93:    93	information caused by complete clothing removal. Specifically, we
94:    94	introduce Hairstyle-Oriented Augmentation (HSOA) to generate
95:    95	"same-identity, different-hairstyle" samples and align their features in
96:    96	the embedding space, explicitly decoupling hairstyle from identity representation. To preserve body geometry, we design Cloth-Preserved
98:    98	clothing pixels, suppressing texture bias while maintaining body
103:   103	(1) We pioneer MSP-ReID, a framework that explicitly addresses
104:   104	hairstyle-induced bias for the first time, leading to consistent improvement in robustness and performance in CC-ReID.
105:   105	(2) We present Hairstyle-Oriented Augmentation (HSOA) to
107:   107	HairStyle-Oriented Augmentation(HSOA)
300:   300	Different Hairstyle
302:   302	Clothing-Preserved Random Erasing (CPRE)
303:   303	Clothing
310:   310	Clothes
333:   333	Clothes
337:   337	Clothes
342:   342	Clothing-Preserved image
354:   354	Fig. 2. Overview of MSP-ReID. HSOA (blue dashed, top-right) synthesizes same-ID different-hairstyle images. CPRE (pink dashed,
355:   355	bottom-right) creates raw/erased pairs with a clothing keep ratio. RPA (purple, center-top) uses parsing masks to boost face/limbs and suppress
356:   356	hair. Green denotes the ID branch, pink denotes the clothes branch for adversarial regularization. Inference is RGB-only using the ID branch.
357:   357	decouple hairstyle cues from identity learning and propose Regionbased Parsing Attention (RPA) to focus representation on stable facial
359:   359	(3) We propose Cloth-Preserved Random Erasing (CPRE), which
361:   361	ratio of clothing pixels, preserving geometric information (body shape,
363:   363	(4) Extensive experiments confirm MSP’s robustness and effectiveness, achieving a new state-of-the-art for CC-ReID.
364:   364	2. METHODOLOGY
375:   375	The cropped head region x ⊙ Mhead and its mask are fed to HairFastGAN to synthesize three target hairstyles—short, medium,
382:   382	attributes such as clothing c and hairstyle h. We posit an underlying
396:   396	the clothing state, and H the hairstyle state. Equivalently, we aim for
398:   398	low intra-identity variance) while suppressing clothing/hairstyle cues.
399:   399	2.2. Hairstyle-Oriented Augmentation (HSOA)
401:   401	hairstyle augmentation with HairFastGAN [15]. For each training
410:   410	the identity label y and the clothing label c. The augmented set
412:   412	positive pairs of the same identity under different hairstyles. Notably,
414:   414	representations of the same identity across varying hairstyles and
415:   415	clothing while pushing apart those from different identities.
416:   416	2.3. Cloth-Preserved Random Erasing (CPRE)
417:   417	Although clothing contains strong visual cues, they are considered
418:   418	identity-unrelated features in our task. A common practice in existing
419:   419	methods is to remove this information entirely to compel the model to
420:   420	learn identity-related features. However, removing the entire clothing
422:   422	spatial context. To address this limitation, we propose Cloth-Preserved
423:   423	Random Erasing (CPRE), we design Cloth-Preserved Random Erasing
424:   424	(CPRE) to erase only within the clothing region, retaining a random
425:   425	proportion of clothing pixels, forcing the model to rely more on
426:   426	identity-related cues (face, limbs, shape). Let Mcloth ∈ {0, 1}H×W be
427:   427	the clothing mask (slightly dilated to cover boundary errors). Sample
430:   430	clothing
434:   434	(i.e., E Kr (i, j) Mcloth (i, j) = 1 = r) . The erased image is
438:   438	Methods
442:   442	xerase = (1−Mcloth )+Mcloth ⊙Kr
448:   448	⊙x+ Mcloth ⊙(1−Kr ) ⊙ϵ,
462:   462	Mcloth (i, j) = 0,
463:   463	Mcloth (i, j) = 1 ∧ Kr (i, j) = 1,
464:   464	Mcloth (i, j) = 1 ∧ Kr (i, j) = 0 .
472:   472	While CPRE reduces reliance on clothing features, hairstyle
473:   473	remains a prominent, identity-unrelated distractor. To mitigate this
476:   476	that emphasizes identity-related regions and minimizes attention to
477:   477	identity-unrelated features, such as hair, helping the model learn more
527:   527	Cloth-Changing
531:   531	Cloth-Changing Standard
559:   559	Instruct-ReID [23] CVPR’24 54.2
667:   667	clothes-adversarial loss adopted from our baseline CAL[11].
675:   675	3. EXPERIMENTS
677:   677	Datasets. We evaluate the proposed method on four mainstream
678:   678	Cloth-changing person Re-ID benchmarks: PRCC [17], LTCC [4],
679:   679	VC-Clothes [18], and LaST [19]. The first three are medium-scale
681:   681	Implementation details. We use CAL[11] as the baseline CC-ReID
682:   682	model in our experiments as it generally performed the best with our
686:   686	range is [0.1, 0.3] on PRCC, LaST and VC-Clothes, and [0.2, 0.5] on
689:   689	40). All experiments run on two NVIDIA RTX 3080 Ti GPUs. For
690:   690	PRCC/LTCC/VC-Clothes we sample 4 identities × 16 images per
693:   693	1.0 on VC-Clothes, and λatt = 1.0 performs best across all datasets.
696:   696	Precision (mAP), the two common metrics in Cloth-changing Re-ID.
697:   697	For all dataset, we evaluate our method under the standard setting and
698:   698	the cloth-changing setting following[8].
699:   699	3.2. Comparison with State-of-the-Art Methods
700:   700	Comparative results on PRCC and LTCC. Tables 1 reports comparisons on PRCC and LTCC with classic Re-ID methods (e.g.,
701:   701	HACNN [20], PCB [21], IANet [22]) and recent CC-ReID methods
702:   702	(e.g., AIM [12], RLQ [27], CAL [11]). On PRCC (cloth-changing),
703:   703	MSP-ReID shows a clear margin over the CAL baseline (about ten
707:   707	Table 2. Results on VC-Clothes under General, SC and CC protocols.
709:   709	Method
713:   713	Table 4. Ablation study of HSOA, CPRE, PRA on PRCC and VCClothes.
714:   714	Methods Baseline HSOA CPRE PRA
894:   894	VC-Clothes
915:   915	Methods
964:   964	saturates at the upper bound. On LTCC (cloth-changing), our
965:   965	method yields consistent gains over CAL, while on LTCC (standard)
967:   967	among RGB-only approaches.
968:   968	Comparative results on VC-Clothes. Table 2 compares with conventional baselines (MDLA [1], PCB [21], PS [28], FSAM [7],
969:   969	BSGA [29], CAL [11]) and recent CC-ReID lines (STL+ACL [30],
970:   970	DLCR [31]).under the General and SC protocols, MSP-ReID is
973:   973	reported Rank-1, while remaining RGB-only, methods with higher
978:   978	indicates that on the large-scale LaST (CC) benchmark, MSP-ReID
981:   981	RLQ—likely due to domain shift and inevitable hair/clothes boundary
985:   985	HSOA clearly changes performance trends. HSOA synthesizes sameID but different-hairstyle views, explicitly decoupling hairstyle from
987:   987	focus on identity-related cues (facial structure, exposed-skin geometry,
991:   991	a controllable clothing keep ratio, CPRE suppresses the dominance
992:   992	of large apparel regions while preserving partial clothing semantics.
1002:  1002	attention with identity-related anatomy reduces gradients attracted to
1003:  1003	hair/clothes boundaries and complements the above augmentations,
1008:  1008	data (hairstyle diversity), CPRE changes the appearance reliability
1011:  1011	yields the most stable cross-clothes/hairstyle retrieval.
1013:  1013	behaviors. Compared with the baseline, our method correctly retrieves
1014:  1014	the same identity under clothing and hairstyle changes, avoiding
1016:  1016	clear “hot face, cold hair/clothes” pattern—high responses on face,
1019:  1019	4. CONCLUSION
1020:  1020	In this paper, we proposed MSP-ReID, a unified framework that
1021:  1021	mitigates hairstyle-induced bias and preserves structural information
1022:  1022	for cloth-changing person re-identification. MSP-ReID introduces
1023:  1023	Hairstyle-Oriented Augmentation (HSOA) to generate same-identity
1024:  1024	images with diverse hairstyles, explicitly reducing reliance on hairstyle
1025:  1025	cues. Cloth-Preserved Random Erasing (CPRE) selectively removes
1026:  1026	pixels within clothing regions while retaining geometric and contextual
1029:  1029	experiments on PRCC, LTCC, VC-Clothes, and LaST confirm that
1030:  1030	MSP-ReID achieves state-of-the-art performance and exhibits strong
1031:  1031	robustness to both clothing and hairstyle changes, highlighting its
1032:  1032	potential for practical, long-term CC-ReID applications.
1043:  1043	and X. Xue, “Long-term cloth-changing person re-identification,”
1045:  1045	[5] Q. Wang, X. Qian, Y. Fu, and X. Xue, “Co-attention aligned mutual cross-attention for cloth-changing person re-identification,”
1050:  1050	hybrid clothing states,” arXiv preprint arXiv:2405.16600, 2024.
1052:  1052	shape-appearance mutual learning for cloth-changing person
1056:  1056	[8] Q. Wang, X. Qian, B. Li, X. Xue, and Y. Fu, “Exploring finegrained representation and recomposition for cloth-changing
1063:  1063	Saudagar, S. Mumtaz, and K. Muhammad, “Cloth-changing
1066:  1066	[11] X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, and X. Chen, “Clotheschanging person re-identification with rgb modality only,” in
1070:  1070	Causality inspired cloth-debiasing for cloth-changing person reidentification,” in Proceedings of the IEEE/CVF conference on
1072:  1072	[13] K. Han, S. Gong, Y. Huang, L. Wang, and T. Tan, “Clothingchange feature augmentation for person re-identification,” in
1076:  1076	consistency network for cloth-changing person re-identification,”
1080:  1080	approach,” Advances in Neural Information Processing Systems,
1086:  1086	contour sketch under moderate clothing change,” IEEE transactions on pattern analysis and machine intelligence, vol. 43,
1089:  1089	re-identification meets changing clothes,” in Proceedings of
1107:  1107	X. Chen, “Interaction-and-aggregation network for person reidentification,” in Proceedings of the IEEE/CVF conference on
1110:  1110	F. Zhu, R. Zhao, W. Ouyang et al., “Instruct-reid: A multipurpose person re-identification task with instructions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and
1114:  1114	clothes-changing person re-identification,” IEEE Transactions
1116:  1116	[25] W. Hu, C. Zhao, C. Gao, and Z. Wu, “Causal feature supervision decoupling: A novel method for clothes-changing person
1120:  1120	“Clothes-changing person re-identification with feasibility-aware
1124:  1124	task agnostic distillation for real world clothes changing reid,”
1127:  1127	pixel sampling for cloth-changing person re-identification,”
1129:  1129	[29] J. Mu, Y. Li, J. Li, and J. Yang, “Learning clothes-irrelevant
1130:  1130	cues for clothes-changing person re-identification.” in BMVC,
1133:  1133	the clothes-changing person re-identification with multiple loss
1139:  1139	diffusion for clothes-changing person re-id,” in 2025 IEEE/CVF
1153:  1153	[35] Y. Lu and M. Z. Jin, “Dual-branch network fused with twolevel attention mechanism for clothes-changing person reidentification,” International Journal of Web Services Research

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|mix|modality|paradigm'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3:     3	Mix-Modality Person Re-Identification: A New and Practical
4:     4	Paradigm
15:    15	Current visible-infrared cross-modality person re-identification research has only focused on exploring the
16:    16	bi-modality mutual retrieval paradigm, and we propose a new and more practical mix-modality retrieval
17:    17	paradigm. Existing Visible-Infrared person re-identification (VI-ReID) methods have achieved some results
18:    18	in the bi-modality mutual retrieval paradigm by learning the correspondence between visible and infrared
19:    19	modalities. However, significant performance degradation occurs due to the modality confusion problem when
20:    20	these methods are applied to the new mix-modality paradigm. Therefore, this paper proposes a Mix-Modality
21:    21	person re-identification (MM-ReID) task, explores the influence of modality mixing ratio on performance, and
22:    22	constructs mix-modality test sets for existing datasets according to the new mix-modality testing paradigm. To
23:    23	solve the modality confusion problem in MM-ReID, we propose a Cross-Identity Discrimination Harmonization
26:    26	aggregating samples with the same modality and the same identity. Furthermore, we propose a Modality
27:    27	Bridge Similarity Optimization Strategy (MBSOS) to optimize the cross-modality similarity between the
28:    28	query and queried samples with the help of the similar bridge sample in the gallery. Extensive experiments
29:    29	demonstrate that compared to the original performance of existing cross-modality methods on MM-ReID, the
31:    31	CCS Concepts: • Computing methodologies → Artificial intelligence; Computer vision;
32:    32	Additional Key Words and Phrases: Cross-Modality Person Re-identification, Mix-Modality Paradigm, Metric
35:    35	WEI LIU, XIN XU, HUA CHANG, XIN YUAN, and ZHENG WANG. 2024. Mix-Modality Person Re-Identification:
36:    36	A New and Practical Paradigm. J. ACM , (May 2024), 21 pages. https://doi.org/XXXXXXX.XXXXXXX
47:    47	Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
65:    65	Mix Modality Probe
67:    67	Mix Modality Gallery
86:    86	Visible modality sample
87:    87	Infrared modality sample
94:    94	Fig. 1. (a) Existing bi-modality mutual retrieval test paradigms for VI-ReID use a visible probe image queried
95:    95	in an infrared image gallery or an infrared probe image queried in a visible image gallery. (b) Our proposed
96:    96	mix-modality testing paradigm for MM-ReID uses mix-modality probes to query in a mixed-modality gallery.
98:    98	modality. In the figure, blue/gray represents visible/infrared modality samples, respectively, and different
100:   100	information, the distance between samples of the same modality but different identities is closer than that
106:   106	INTRODUCTION
109:   109	retrieve specific pedestrians across a network of non-overlapping cameras [11, 33, 37]. While SingleModality ReID (SM-ReID) methods focusing on visible image retrieval have achieved significant
115:   115	which aims to tackle the challenge of cross-modality image matching [9, 14, 16, 36].
118:   118	arising from modality differences. Despite considerable progress achieved in the bi-modality mutual
119:   119	retrieval paradigm, as illustrated in Fig 1 (a) through learning potential correspondences between
123:   123	of one modality type. This integration often leads to a mix of what seems to be straightforward
124:   124	SM-ReID tasks into the existing cross-modality framework. However, as indicated in Fig 1 (c), this
125:   125	approach results in a marked performance degradation, primarily due to the ‘Modality Confusion’
128:   128	Mix-Modality Person Re-Identification: A New and Practical Paradigm
133:   133	within the same modality, which confuses the matching of cross-modality identity information.
134:   134	More specifically, the impact of modality confusion on the current approach of learning only
135:   135	visible-infrared cross-modality correspondences is huge, due to the fact that there exists not only
136:   136	one correspondence from visible to infrared in the retrieval process, but also two same-modality
137:   137	correspondences, which is not taken into account by the existing approaches of VI-ReID. Under this
138:   138	influence, the natural similarity between samples of the same modality will disturb the perception of
139:   139	the algorithm, resulting in the distance between samples of different identities of the same modality
142:   142	practical Mix-Modality person re-identification (MM-ReID) task, creating mixed modality test sets
143:   143	for existing datasets and examining how the ratio of modality mixing affects retrieval performance.
144:   144	To address the modality confusion challenge in MM-ReID, we introduce the Cross-Identity
145:   145	Discrimination Harmonization Loss (CIDHL). This novel approach optimizes the sample distribution within a hyperspherical feature space. Specifically, CIDHL aggregates distances between
146:   146	samples of the same identity and modality towards their respective centers, while simultaneously
148:   148	those of different identities, irrespective of modality. This strategy mitigates the effects of modality
149:   149	confusion. Additionally, we propose a Modality Bridge Similarity Optimization Strategy (MBSOS),
150:   150	which optimizes the cross-modality distance metric by identifying a similar bridge sample in the
152:   152	queried samples. Extensive testing confirms that our methods—CIDHL and MBSOS—significantly
153:   153	enhance the model performance of existing cross-modality approaches, demonstrating their efficacy
154:   154	and adaptability in resolving modality confusion in MM-ReID.
156:   156	• Paradigm Contribution: We pioneer the Mix-Modality person re-identification (MMReID) task, introducing a novel testing paradigm that incorporates both visible and infrared
158:   158	• Empirical Contribution: We investigate the impact of varying modality mixing ratios
159:   159	on cross-modality model performance and conclude the unique challenges of ‘Modality
162:   162	Loss (CIDHL) and the Modality Bridge Similarity Optimization Strategy (MBSOS). These
163:   163	transferable methodologies effectively address modality confusion, as validated by extensive
164:   164	empirical experiments.
168:   168	RELATED WORK
169:   169	Single-Modality Person Re-Identification
171:   171	Single-Modality person re-identification (SM-ReID) seeks to identify and retrieve all images of a
174:   174	recent years. However, the predominant focus of existing approaches has been on addressing
188:   188	context, the Visible-Infrared person re-identification (VI-ReID) task, which was first proposed by
189:   189	Song et al. [31], addresses the cross-modality matching of visible and infrared images to overcome
195:   195	Visible-Infrared person re-identification (VI-ReID) aims to utilize visible images to retrieve corresponding infrared images or vice versa. This bi-modality mutual search paradigm confronts
196:   196	not only the common challenges of single-modality ReID, such as changes in background, viewpoint, and pose but also the substantial challenge caused by discrepancies between visible and
198:   198	been devised to mitigate both inter- and intra-modality variations. For instance, Wang et al. [26]
200:   200	Yet, they later recognized limitations in set-level alignment, leading to a refined approach that
205:   205	learnable prototypes, using affinity information to refine cross-modality identification. Traditional
207:   207	method to enhance image diversity by swapping and erasing color channels randomly. For localized
209:   209	Liang et al. [12] adopted this approach for cross-modality ReID tasks, focusing on extracting highly
210:   210	discriminative features through modality-aware enhancement. In terms of modality transformation,
211:   211	Wei et al. [30] proposed synthesizing new modalities by merging features from both RGB and
213:   213	modalities to minimize differences, with Zhong et al. [46] experimenting on converting infrared to
218:   218	mix-modality images of specific pedestrians in a mix-modality gallery, rather than just a simple
219:   219	bi-modality mutual retrieval paradigm adopted by existing methods. Therefore, we propose a
220:   220	Mix-Modality person re-identification (MM-ReID) task according to the mix-modality paradigm.
223:   223	METHODOLOGY
225:   225	In this section, we elaborate on the novel Mix-Modality person re-identification (MM-ReID) paradigm and introduce two cornerstone methodologies: the Cross-Identity Discrimination Harmonization
226:   226	Loss (CIDHL) and the Modality Bridge Similarity Optimization Strategy (MBSOS). More specifically, Section 3.1 provides the task definition and formulation of our proposed MM-ReID together
227:   227	with an analysis of the unique modality confusion problem that exists in MM-ReID with respect
228:   228	to VI-ReID, Section 3.2 describes the design motivation for the method and the differences with
229:   229	existing methods, Section 3.3 introduces the design idea and the detailed composition of the CIDHL
234:   234	Mix-Modality Person Re-Identification: A New and Practical Paradigm
269:   269	identities, blue/gray represents visible/infrared modality samples, while green and red lines represent correct
271:   271	different identities (shapes); (b) VI-ReID needs to face the challenge of modality (color) differences in addition
272:   272	to identity differences; and (c) MM-ReID needs to face the challenge of modality confusions (similar modalities
273:   273	possessing a closer proximity) in addition to identity and modality differences.
277:   277	Mix-Modality Person Re-identification
281:   281	containing 𝑁 g gallery images. As depicted in Figure 1, under the traditional bi-modality mutual
282:   282	retrieval paradigm of Visible-Infrared Re-Identification (VI-ReID), the sets Q and G each consist
283:   283	of images exclusively in one modality, either visible or infrared. Contrary to this, our proposed
284:   284	Mix-Modality Re-Identification (MM-ReID) paradigm includes images in both visible and infrared
286:   286	3.1.2 Challenge Analyze. As illustrated in Fig 2, the challenges inherent in Single-Modality ReID
291:   291	mechanisms. These modality differences represent significant obstacles to effective cross-modality
292:   292	matching. Furthermore, Mix-Modality ReID (MM-ReID) introduces additional complexities. Beyond
293:   293	the challenges faced in VI-ReID, MM-ReID contends with ’modality confusion,’ where similarities
294:   294	irrelevant to identity, such as color consistency within a modality, can obscure vital identityspecific cues between the query (Q) and gallery (G) sets, thereby complicating the differentiation
298:   298	Method Design Motivation
300:   300	To tackle the distinct modality confusion problem in the MM-ReID task, this paper introduces two
301:   301	innovative methods: Cross-Identity Discrimination Harmonization Loss (CIDHL) and Modality
303:   303	from similarity among same-modality samples by constraining these similarities in the metric
305:   305	modality when attempting to match cross-modality, same-identity positive samples. Moreover,
306:   306	to mitigate modality confusion effects during the testing phase, MBSOS optimizes the similarity
307:   307	measure between cross-modality samples by leveraging same-modality samples in the gallery
308:   308	as a bridge. This is a departure from traditional VI-ReID methods, which typically focus only
309:   309	on constraining or exploiting the similarity across modalities. Our approach aims to refine both
322:   322	Modality Bridge Similarity Optimize
351:   351	Fig. 3. An illustration of our proposed CIDHL and MBSOS. The mix-modality data is fed into two feature
353:   353	between the centers of the cross-modality same identity samples is pulled closer together pushing away the
360:   360	posed by same-modality samples unique to the MM-ReID task.
365:   365	3.3.1 Feature Presentation Extractor. Our proposed methods, Cross-Identity Discrimination
366:   366	Harmonization Loss (CIDHL) and Modality Bridge Similarity Optimization Strategy (MBSOS),
368:   368	methods. This integration ensures that there is no need for modifications to the baseline feature
369:   369	extraction processes of these methods. As illustrated in Figure 3, we employ a generalized baseline
374:   374	robust performance across varied modality inputs.
377:   377	triplet hard loss [6]. This approach refines the basic triplet loss by dynamically selecting the most
385:   385	Mix-Modality Person Re-Identification: A New and Practical Paradigm
440:   440	samples. This approach helps stabilize the training process by reducing the influence of outliers
442:   442	all samples associated with that identity. This average is computed separately for each modality,
445:   445	loss, thereby providing a more reliable basis for optimizing the model. This method not only
464:   464	where 𝑣 i𝑗 and 𝑡 i𝑗 denote the 𝑗 − 𝑡ℎ image of 𝑖 − 𝑡ℎ identity within visible and infrared modality,
469:   469	centers of the same identity, regardless of the modality. This step ensures that the same
472:   472	between feature centers of different identities within the same modality and across different
480:   480	• Aggregate intra-modality distances: To further refine the model’s effectiveness, distances
481:   481	between samples of the same modality and identity are aggregated towards their respective
482:   482	centers. This aggregation aids in minimizing the modality-specific variations within the same
485:   485	The CIDHL adopts a structured approach, containing two parts, the first of which is targeted:
486:   486	• Cross-modality attraction: Distances between centers of the same identity across modalities
488:   488	• Intra- and inter-modality repulsion: Distances between centers of different identities, whether
489:   489	within the same modality or across modalities, are pushed apart to enhance discriminability.
525:   525	center of samples with the same modality and identity for solving the modality confusion problem
556:   556	Modality Bridge Similarity Optimization Strategy
558:   558	The Modality Bridge Similarity Optimization Strategy (MBSOS) addresses the modality confusion
565:   565	gallery of similar modality to the query, to create a modality-consistent path. This method
566:   566	effectively reduces the impact of direct modality differences on the distance metric.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|meta|pairwise|relationship|distillation'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:     5	Meta Pairwise Relationship Distillation for Unsupervised
14:    14	Abstract
16:    16	Pairwise Similarity
25:    25	Pairwise Pseudo
44:    44	methods often rely on estimated pseudo labels via iterative clustering and classification, and they are unfortunately
47:    47	propose the Meta Pairwise Relationship Distillation (MPRD)
48:    48	method to estimate the pseudo labels of sample pairs for
55:    55	meta knowledge to judge the difference in the neighborhood
56:    56	structure between positive and negative sample pairs. Extensive experiments on Market-1501, DukeMTMC-reID and
57:    57	MSMT17 datasets show that our method outperforms the
58:    58	state-of-the-art approaches.
60:    60	1. Introduction
66:    66	Existing methods usually learn discriminative features in a
92:    92	Pairwise
103:   103	Pairwise Label
110:   110	Pairwise Neighborhoods
124:   124	(b) Our method
130:   130	Figure 1. Illustrations of two pseudo label estimation methods,
131:   131	in which (a) the traditional method directly take the pairwise similarity to estimate pseudo labels, while (b) our method takes the
132:   132	pairwise neighborhood structures to estimate pseudo labels. Each
137:   137	Recent unsupervised person Re-ID methods [13, 14, 6]
145:   145	discriminative feature learning as a pairwise relationship
166:   166	a graph structure is more suitable to estimate pairwise labels, as shown in Figure 1 (b), which exploits contextual
167:   167	information to deduce the correct pairwise pseudo label for
169:   169	In this paper, we propose the Meta Pairwise Relationship
170:   170	Distillation (MPRD) method for unsupervised person ReID. It comprises a Convolutional Neural Network (CNN)
172:   172	estimates the pseudo labels of sample pairs via the meta
177:   177	per-image feature and pairwise pseudo labels. At each iteration, the CNN extracts the current per-image feature, and
179:   179	and the previous features. Afterwards, the pairwise neighborhood structure is estimated by connecting every image with
184:   184	therefore, we exploit a small amount of labeled metadata to
188:   188	We evaluate our proposed method on Market-1501 [34],
192:   192	learning task as a pairwise relationship estimation problem, which avoids the error-prone step of estimating the
193:   193	number of clusters in most existing methods.
194:   194	2. We propose the MPRD method for unsupervised person
195:   195	Re-ID, which incorporates a dedicated GCN as the pairwise pseudo label generator in the training stage and it
199:   199	pseudo labels based on the pairwise neighborhood struc-
203:   203	2. Related Work
204:   204	Supervised Person Re-identification methods require
206:   206	Early methods usually extract a global feature representation per image for image retrieval [28, 18, 10]. In PersonNet [28], a small-scale convolutional filter captures the
215:   215	Unsupervised Person Re-identification methods relieve the requirement for the cost-prohibitive annotations,
216:   216	which include hand-crafted feature based methods [12, 34],
217:   217	unsupervised domain adaptation methods [7, 36, 37, 16, 9,
218:   218	11, 3, 31, 43] and fully unsupervised methods [13, 6, 27, 15,
222:   222	Methods based on unsupervised domain adaptation utilize prior knowledge on a source dataset with labels, and
233:   233	Some fully unsupervised methods are guided by pseudo
278:   278	Pairwise
279:   279	Relationship
287:   287	Figure 2. Overview of MPRD. An initialized backbone network extracts the feature of the training image. Then GCN infers the pairwise
288:   288	relationship between the features and their neighbors, which is used to train the CNN model.
290:   290	The most relevant existing method is MLCR [24], which
292:   292	MPRD differs from MLCR in two aspects. First, we reformulate the task as a pairwise relationship estimation problem; second, we design an effective GCN model to provide
300:   300	Loss function. Suppose the pairwise pseudo labels are
305:   305	3. Meta Pairwise Relationship Distillation
308:   308	training samples, the MPRD estimates the pairwise pseudo
310:   310	CNN learns discriminative features supervised by the pairwise pseudo labels generated by GCN; while the GCN estimates the pairwise pseudo labels based on CNN features.
326:   326	* Our method is compatible with various network backbones.
405:   405	Pairwise
420:   420	Relationship
461:   461	(a) Construct Pairwise Neighborhood Structures
469:   469	Figure 3. Illustration of the pairwise neighborhood structure and the training strategy of MPRD, where the red arrow means prohibit
470:   470	execution until training the CNN model after tth iteration, the blue line means the pairs’ relationship is not yet judged, the green line means
475:   475	unlabeled data. As shown in Figure 2 (b), it takes the pairwise neighborhood structures Gij = (Aij , Vij ) as inputs,
478:   478	NNk (xi ) = top(hM̃[i], M̃[j]i, k), the pairwise neighborhood structure can be constructed by connecting images xi
516:   516	layers extract features from the pairwise neighborhood structures, the graph aggregation layer aggregates the node-level
519:   519	Loss function. Our GCN takes the pairwise neighborhood structure as input, and outputs the likelihood of xi
570:   570	Generate pairwise neighborhood structures
590:   590	pseudo pairwise labels by GCN, we propose to generate
591:   591	an initial, accurate pairwise neighborhood structure with a
592:   592	tiny amount of labeled meta data Z = {zm }M
610:   610	training are unavailable, we resort to the GCN for pairwise
612:   612	pairwise neighborhood structures for GCN in three steps, as
615:   615	(3) the pairwise neighborhood structure Gij is obtained by
618:   618	pairwise label gij . To refine gij , we subsequently apply a
633:   633	2. For each image zi in the labeled meta dataset Z, we randomly draw two images of the same identity to generate
638:   638	construct a pairwise neighborhood structure Gij , and insert
640:   640	Parameter update. With the pairwise neighborhood
681:   681	and n0 /2 images from the labeled meta data and unlabeled
682:   682	data, respectively. Then we construct pairwise neighborhood
685:   685	G /∂ΘG from labeled meta data, and compute a linear
706:   706	represents the ratio of the number of pairwise neighborhood
707:   707	structure generated by metadata and generated by generated
711:   711	unlabeled data. At later iteration stages, the number of pairwise neighborhood structures with pseudo labels extracted
715:   715	5. Experiments
717:   717	Datasets. We evaluate our method on three standard largescale Person Re-ID datasets, including Market-1501 [34],
733:   733	We implement our method in PyTorch [17] with a single
752:   752	Methods
801:   801	Table 1. Performance with different pseudo label generation methods. “Super.” and “Single” are baselines representing performance
802:   802	upper and lower bounds, respectively. All methods have incorporate the same Binomial deviance loss. The “*” mark in “MPLP*”
811:   811	and λ1 = λ2 = 0.2. The small amount of labeled meta data
814:   814	labeled metadata is only used for training G. The value of k
820:   820	other pseudo label generation methods, including the KNN
822:   822	threshold. MPLP is proposed in [24], which predicts pseudolabels with high accuracy via similarity scores and cycle
827:   827	experiments, i.e., fully supervised re-ID with ground-truth
833:   833	its GCN trained purely on labeled meta data (i.e., without
841:   841	Methods
856:   856	and 39.3% mAP without CamStyle. These results demonstrate the effectiveness of the proposed MPRD, and show
867:   867	many false negative pairs. Based on these experiments, we
869:   869	Effect of small amount of labeled meta data. Since our
870:   870	approach introduces a small amount of labeled meta data to
872:   872	the competing MLCR method. For fair comparison, we let
873:   873	MLCR have access to the same amount of labeled meta data
875:   875	amounts of labeled meta data in different variants of MPRD.
876:   876	We let MLCR have access to the same labeled meta data
878:   878	whenever the input training data belong to the labeled meta
880:   880	extra small amount of labeled meta data, “MLCR(+5id)*”
881:   881	marginally outperforms its original version MLCR*, possibly due to the portion of such labeled meta dataset is too
908:   908	Table 2. Ablation study of the effect of meta data. “MLCR(+5id)*”
910:   910	same amount (5 identities) of meta data as extra supervision.
911:   911	“MPRD(0id)” represents an ablated version of our proposed MPRD
912:   912	with the labeled meta data-based jump start procedure completely
915:   915	with different amounts of labeled meta dataset, where “5id”
916:   916	means five labeled identities (meta data is only used for
925:   925	We evaluate the proposed MPRD on Market-1501 [34],
926:   926	DukeMTMC-reID [20] and MSMT17 [26] datasets. Although a small amount of labeled meta data are used to
927:   927	guide the training process of the GCN, our method also belongs to unsupervised Person Re-Identification because there
929:   929	data. The proposed method is compared against the state-ofthe-art unsupervised Person Re-ID methods: LOMO [12],
933:   933	Table 3 shows the results of the proposed method and
934:   934	state-of-the-art methods on Market-1501 and DukeMTMCreID. On Market-1501, our MPRD achieves 2.7% higher
939:   939	higher mAP. We also conduct experiments on MSMT17, and
942:   942	Of all the competing algorithms, MLCR is the most relevant one to our proposed MPRD. As is verified in the above
943:   943	results, MPRD outperforms it on Market-1501, DukeMTMCreID and MSMT17. We speculate that this performance advantage arise from the following aspects. Our proposed
948:   948	Methods
1041:  1041	Table 3. Unsupervised person re-ID performance with state-of-the-art methods on Market-1501 and DukeMTMC-reID datasets.
1043:  1043	Methods
1073:  1073	Table 4. Unsupervised person re-ID performance with state-of-theart methods on MSMT17 dataset.
1099:  1099	are resolved with the introduction of GCN. With Example2,
1104:  1104	6. Conclusion
1105:  1105	In this paper, we propose the MPRD method to address
1106:  1106	the unsupervised person Re-ID task. Unlike previous methods that estimate the pseudo labels through either iterative
1108:  1108	clustering or classification, it is unnecessary for our method
1110:  1110	proposed MPRD reformulates the unsupervised discriminative feature learning task into a pairwise relationship estimation problem. A GCN is used to estimate the pairwise relationship of sample pairs based on the graph structure among
1112:  1112	from input images according to these estimated pairwise
1113:  1113	relationship labels. Extensive experiments on Market-1501,
1115:  1115	effectiveness of the proposed method for the unsupervised

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf' - | nl -ba | rg -n -i 'abstract|introduction|method|approach|proposed|experiment|related|conclusion|memory|meta|shuffled'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
10:    10	Memory-augmented shuffled meta learning for visible–infrared person
34:    34	Meta learning
35:    35	Memory-augmentation
39:    39	ABSTRACT
43:    43	Existing methods often struggle to effectively address this issue, as they fail to capture global similarity metrics
45:    45	challenges, we introduce a novel approach called memory-augmented shuffled meta (MASM) learning. Our
46:    46	approach is distinguished by two key components: shuffled meta learning (SML) and memory meta learning
48:    48	from a wide range of data inputs. Meanwhile, MML leverages historical information stored in memory banks to
50:    50	but also empowers the model to learn comprehensive global meta metrics, significantly improving its ability
51:    51	to distinguish individuals across modalities. Extensive experiments on the RegDB and SYSU-MM01 datasets
52:    52	validate the effectiveness of our MASM method, demonstrating its superiority over several state-of-the-art
53:    53	approaches.
55:    55	1. Introduction
58:    58	person re-identification methods primarily rely on visible cameras,
72:    72	To address modal discrepancies in VIPR, metric learning is a promising approach, which does not require overly complex networks. Existing metric learning methods in VIPR include sample-based and centerbased approaches. The sample-based approaches, such as the triplet
77:    77	methods (Dai et al., 2018; Feng, Xu, Ji, & Wu, 2021; Liu, Ma, Xia, &
80:    80	sample-based methods involve sample mining or weighting techniques,
96:    96	In this paper, we introduce a memory-augmented shuffled meta
98:    98	adopting a meta-learning perspective lies in the way query-support
102:   102	meta-learning, we propose a shuffling strategy that maximizes data
104:   104	each training cycle. Furthermore, our memory-augmented mechanism
105:   105	utilizes historical feature representations stored in memory banks,
107:   107	comprehensive cross-modality metrics. The proposed MASM integrates
108:   108	shuffled meta learning (SML) and memory meta learning (MML).
109:   109	Specifically, the SML method enhances data utilization by constructing
123:   123	and support sets. Concurrently, the MML method captures long-term
124:   124	dependencies by leveraging historical data stored in memory banks,
126:   126	over time and facilitating the learning of comprehensive global meta
128:   128	usage but also empowers the model to learn comprehensive global meta
133:   133	Fig. 1. The illustration of metric learning methods. (a) The sample-based approach
134:   134	directly constraints the metric between the samples. (b) The center-based approach
135:   135	emphasizes intra-class compactness. (c) The proposed meta-based approach shuffles to
137:   137	meta cells.
139:   139	Center-based approaches (Can, Hong, Wei, & Mang, 2020; Liu, Tan,
141:   141	than sample-based methods through centralization technologies, as illustrated in Fig. 1(b). For example, improved center loss functions (Can
147:   147	However, center-based approaches underestimate complex intra-class
151:   151	In addition, both sample-based and center-based methods only perform metric learning on a fixed batch size of data in each calculation,
154:   154	adopt image augmentation methods (Jia, Zhong, Ye, Liu, & Huang,
161:   161	enhancement. However, these methods may introduce artifacts in the
163:   163	incorrect heat distributions in infrared images. Other approaches rely
169:   169	While these feature enhancement methods mitigate data insufficiency
177:   177	• We propose the shuffled meta learning (SML) method to enhance
179:   179	• We propose the memory meta learning (MML) method leverages
180:   180	historical data to enhance the model’s ability to learn global meta
182:   182	• Extensive experiments on two public datasets, RegDB (Nguyen,
184:   184	Gong, & Lai, 2017), demonstrate that our approach outperforms
185:   185	several state-of-the-art VIPR methods.
187:   187	surveys recent work related to our paper. Section 3 describes our
188:   188	method in detail. Section 4 presents experimental results and analysis to
189:   189	show the superiority of our method. Section 5 discusses the limitations
191:   191	2. Related work
193:   193	In visible–infrared person re-identification task, sample-based metric learning methods (Hu, Liu, Zeng, & Hu, 2022; Seokeon, Lee, Kim,
208:   208	(2023). Some methods (Alehdaghi, Josi, Cruz, & Granger, 2022; Li,
215:   215	realm of sample-based methods, it is important to note that many of
222:   222	Center-based method (Cai, Zhu, & Zhang, 2021; Feng, Wu, & Zheng,
238:   238	cross-modal proxies and samples. However, center-based approaches
242:   242	and nuances present within each class. Therefore, center-based methods
244:   244	across different modalities. Compared with center-based methods, the
245:   245	proposed shuffled meta learning (SML) method is designed to capture
249:   249	training cycle. This approach enhances data utilization and allows the
250:   250	model to learn from a broader range of intra-class variations, addressing some limitations of center-based methods that may overlook these
252:   252	in memory banks, SML is expected to improve the model’s ability to
259:   259	are model based meta learning (Munkhdalai & Yu, 2017; Santoro,
261:   261	based meta learning (Finn, Abbeel, & Levine, 2017; Nichol, Achiam,
262:   262	& Schulman, 2018). The classical model based meta learning method,
263:   263	namely, meta network (Munkhdalai & Yu, 2017), contains a meta
266:   266	well-known optimization based meta learning is the model-agnostic
267:   267	meta-learning (MAML) (Finn et al., 2017) method, which learns good
268:   268	initial parameters of the learner for a fast adaptation. Meta-learning
272:   272	There are also some meta-based methods to handle re-identification
274:   274	meta-learning strategy to simulate real train–test domain shifts. Zhang,
275:   275	Liu, Zhang and Zhang (2023) developed a self-paced meta-learning
276:   276	method that extends conventional one-stage meta-learning to a multistage training process, simulating human learning. These methods involve alternating meta-train and meta-test processes: the model is
277:   277	trained on source data during meta-training and validated on unseen
278:   278	data during meta-testing. In contrast, our method directly optimizes a
279:   279	metric space for VIPR by introducing a shuffling strategy that dynamically constructs query-support pairs while leveraging memory banks
280:   280	to preserve long-term feature relationships. Our mate-based method
283:   283	2.3. Memory bank designs
284:   284	Memory-based designing is an effective method to learn useful
286:   286	Wang, Zhang, Huang and Scott (2020) proposed a cross-batch memory
288:   288	et al. (2022) constructed a memory dictionary to take advantage of the
289:   289	meta learning validation set, which provides rich information in the
295:   295	of memory-based designs, we enhance our approach by integrating
296:   296	memory banks and meta learning techniques. This integration allows
297:   297	us to learn a global meta-metric that can effectively capture long-term
300:   300	In addition to memory mechanisms, existing works introduce temporal modeling to handle challenges in person re-identification. For
301:   301	example, MITML (Lin et al., 2022) employs long short-term memory (LSTM) layers to learn temporal representations from fixed-length
306:   306	context for a dedicated temporal memory module, which aggregates
310:   310	our MML method provides a more flexible and robust mechanism for
315:   315	2.2. Meta learning
316:   316	Meta learning, also known as learning to learn, is a learning
318:   318	learn new tasks quickly and effectively. This approach enables models
320:   320	strong generalization capabilities. There are some meta learning methods (Chen, Zhang, Lu, & Zhou, 2019; Vinyals, Blundell, Lillicrap,
324:   324	during the training process. Later, Chen et al. (2019) improved the setbased distance of meta learning by incorporating a hard sample mining
332:   332	3. Approach
334:   334	where 𝐿shuffled(𝑛) is designed in Eq. (4).
335:   335	𝐿shuffled(𝑛) (𝜃 , , ) = 𝐿→
338:   338	shuffled(𝑛)
339:   339	shuffled(𝑛)
341:   341	In this section, we begin by exploring the shuffled meta learning (SML) method. Next, we delve into the design of the memory
342:   342	meta learning (MML) method. Subsequently, we integrate SML and
343:   343	MML to develop the memory-augmented shuffled meta (MASM) learning approach. Finally, we present the comprehensive architecture for
345:   345	Notations: To facilitate a clearer understanding of our approach,
347:   347	the general meta learning framework (Chen et al., 2019), the VIPR task
350:   350	𝑝( ), the optimization objective of the general meta learning method
358:   358	shuffled(𝑛)
359:   359	shuffled(𝑛)
364:   364	By reformulating the meta-learning objective as in Eqs. (5) and
367:   367	meta-learning). This ensures that every sample serves as both query and
372:   372	aggregate gradients from all 𝐾 pershuffled(𝑛)
373:   373	shuffled(𝑛)
375:   375	to the single pairing in the general meta learning method, effectively
377:   377	In summary, our shuffled meta learning approach improves the
381:   381	combinations of query and support samples, our approach obtains the
387:   387	3.2. Memory meta learning
388:   388	Memory meta learning (MML) is a novel approach that leverages
389:   389	memory banks to store historical information, thus enhancing the
390:   390	learning process. The core idea is to maintain a memory of previously
391:   391	encountered tasks, which allows the model to draw upon past experiences to improve its performance on new and related tasks. This is
401:   401	 ,   ) denote a V2I meta metric loss and a I2V meta
417:   417	[𝐶]}, where 𝑐 ,𝑖 = {𝑥𝑐 ,𝑘 |𝑘 ∈ [𝐾], 𝑘 ≠ 𝑖} is a meta cell consisting of
421:   421	where 𝑐 ,𝑖 = {𝑦𝑐 ,𝑘 |𝑘 ∈ [𝐾], 𝑘 ≠ 𝑖} is the meta cell consisting of infrared
423:   423	From Eq. (2), it can be found that the general meta learning method
427:   427	features, we propose a shuffled meta learning method, which circularly
428:   428	divides query and support sets. The proposed shuffled meta metric loss
429:   429	𝐿shuffled(𝑛) is formulated as Eq. (3).
441:   441	shuffled(𝑛)
444:   444	modes in the VIPR task, i.e., visible-to-infrared (V2I) and infrared-tovisible (I2V) retrieval modes, the general meta learning method can be
453:   453	3.1. Shuffled meta learning
455:   455	𝜃 = ar g min E𝑛 ∼𝑝( ) [E,∼𝑛 [𝐿shuffled(𝑛) (𝜃 , , )]],
465:   465	shuffled(𝑛)
468:   468	set; 𝐿(𝑛) (𝜃 , , ) denotes the general meta metric loss of the sampled
489:   489	where  denotes a memory-based support set that contains critical
491:   491	memory meta (MM) loss, which is formulated in Eq. (8):
495:   495	shuffled(𝑛)
496:   496	shuffled(𝑛)
501:   501	memory banks, respectively. Here, 𝑀 = 𝐿×𝐶 ×𝐾, where 𝐿 denotes the
502:   502	number of subtasks stored in a memory bank, 𝐶 indicates the number of
507:   507	current subset are directly enqueued into the memory banks. This
510:   510	However, as the memory banks have a size limitation, once they reach
515:   515	Building on SML’s data-efficient framework, MML introduces memory banks  and  to store 𝑀 = 𝐿 × 𝐶 × 𝐾 historical features across
527:   527	Fig. 2. The framework of the proposed memory-augmented shuffled meta (MASM) learning method.
530:   530	subtask features are enqueued into memory banks, while outdated
532:   532	utilizes the most informative features. By leveraging these memory
535:   535	in the field of meta-learning by incorporating memory mechanisms
537:   537	approach not only enhances the model’s ability to generalize across
543:   543	According to the designed shuffled meta learning method introduced
544:   544	in Section 3.1, we shuffle to utilize 𝑖 (𝑖 ∈ [𝐾]), then, the V2I shuffled
545:   545	meta metric loss 𝐿→
546:   546	and the I2V shuffled meta metric loss 𝐿→
547:   547	shuffled
548:   548	shuffled
554:   554	shuffled
555:   555	shuffled
621:   621	𝐿MM(𝑛) denotes the memory meta (MM) loss function formulated in
622:   622	Eq. (8), which is responsible for learning global meta metrics by storing
623:   623	historical information. 𝐿CM(𝑛) represents the current meta (CM) loss
624:   624	function, which is in charge of optimizing meta metrics from the
632:   632	shuffled
651:   651	In the shuffled meta learning formulated in Eq. (3), the scale of
653:   653	subtask is limited. For that, we further extend Eq. (3) to a memoryaugmented shuffled meta (MASM) learning loss function by introducing
654:   654	the memory meta learning (MML). The MML leverages memory banks
665:   665	3.3. Memory-augmented shuffled meta learning
671:   671	shuffled
705:   705	the query and the meta cell. In the V2I retrieval mode, assume that 𝑥𝑐 ,𝑖
706:   706	is a randomly-selected query, the distance between query 𝑥𝑐 ,𝑖 and meta
741:   741	randomly-selected query 𝑦𝑐 ,𝑖 and the corresponding meta cell 𝑛,𝑖 is
780:   780	the stability of meta-learning, we propose shuffled meta regularization
837:   837	meta-metric.
843:   843	shuffled
844:   844	shuffled
870:   870	SMR. As shown in Fig. 2, the SMR works on the current meta learning
872:   872	different retrieval, improving the current meta learning.
873:   873	Our approach operates on the principle of pulling positive pairs
876:   876	advancement lies in reformulating this contrastive process from a metalearning perspective. We dynamically construct query-support sets using a shuffling strategy, which replaces traditional sample pairs or fixed
878:   878	gallery (corresponding to the support set in meta-learning) that match
879:   879	the identity of the query image. Furthermore, our memory mechanism
886:   886	4. Experiments and analysis
887:   887	We evaluate the proposed memory-augmented shuffled meta
888:   888	(MASM) learning method on two publicly available visible–infrared
925:   925	2022). This backbone is widely adopted in both classical methods
927:   927	approaches (e.g., Hua et al., 2025; Zhang, Du, Liu, Tu and Shu, 2025)
968:   968	(all-search mode), underscoring the effectiveness of meta-learning. The
970:   970	88.0% to 94.1% on RegDB, validating its role in stabilizing meta metric
972:   972	increase in mINP on SYSU-MM01, highlighting the benefits of leveraging historical meta metrics. Finally, applying the shuffling strategy to
975:   975	These results clearly demonstrate the effectiveness of our proposed
1009:  1009	methods (He et al., 2020; Luo et al., 2019; Ye, Shen et al., 2022), the
1049:  1049	discriminate learning through meta learning.
1052:  1052	To straightforwardly evaluate the role of units in the proposed
1053:  1053	memory-augmented shuffled meta (MASM) learning method, we implement ablation studies on RegDB and SYSU-MM01 datasets, as shown in
1055:  1055	training (Eq. (23)); CM refers to the current meta loss (Eq. (21) with
1057:  1057	CM without shuffled mate regularization (i.e., Eq. (21) with 𝜉 = 0); CM
1059:  1059	the memory meta loss (Eq. (8)); and MM (w/o Shuffling) denotes MM
1063:  1063	memory bank capacity, that is, the memory bank can store 𝐿 batch
1342:  1342	Fig. 5, the optimal memory bank capacities vary across different

codex
定位结果显示，6 个文件都能用 `pdftotext` 正常提取。两个 Mix-Modality 文件是一篇论文的 arXiv 版和 ACM 版，正文高度一致，但我会按两个文件分别核对，因为用户明确列了两个文件。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '364,675p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   364	2. METHODOLOGY
   365	
   366	image x, we utilize the human parser SCHP [16] to obtain a pixelwise semantic map P = SCHP(x) ∈ {1, . . . , K}H×W . From P
   367	we derive binary masks for face and hair and define the head mask as
   368	their union:
   369	Mface = 1[P ∈ Sface ] , Mhair = 1[P ∈ Shair ] ,
   370	Mhead = Mface ∨ Mhair = 1[P ∈ Sface ∪ Shair ] .
   371	
   372	(2)
   373	(3)
   374	
   375	The cropped head region x ⊙ Mhead and its mask are fed to HairFastGAN to synthesize three target hairstyles—short, medium,
   376	long—yielding heads ĥS , ĥM , ĥL that preserve facial structure while
   377	altering hair. Each synthesized head is seamlessly composited back:
   378	
   379	2.1. Problem Formulation
   380	Let x ∈ RH×W ×3 be a pedestrian image with identity label y.
   381	Besides identity, the appearance also contains time-varying nuisance
   382	attributes such as clothing c and hairstyle h. We posit an underlying
   383	data-generating distribution p(x, y, c, h) but only observe (x, y) during training, (c, h) are unannotated. Our goal is to learn an encoder
   384	fθ that maps x to an embedding z = fθ (x) which is (i) highly
   385	discriminative for identity and (ii) insensitive to (c, h). Formally, we
   386	seek a representation z that maximizes identity information while
   387	minimizing dependence on nuisance factors:
   388	
   389	
   390	
   391	max {I fθ (X), Y − λc I fθ (X), C − λh I fθ (X), H }, (1)
   392	θ
   393	
   394	where I(·, ·) denotes mutual information, i.e., the amount of information one variable reveals about the other (I(Z, C) = 0 implies Z and
   395	C are statistically independent). Here, Y denotes the identity label, C
   396	the clothing state, and H the hairstyle state. Equivalently, we aim for
   397	z to preserve identity-discriminative cues (large inter-identity margins,
   398	low intra-identity variance) while suppressing clothing/hairstyle cues.
   399	2.2. Hairstyle-Oriented Augmentation (HSOA)
   400	To explicitly break the shortcut "hair ≈ identity", we perform a
   401	hairstyle augmentation with HairFastGAN [15]. For each training
   402	
   403	x̃ℓ = Mhair ⊙ ĥℓ + (1 − Mhair ) ⊙ x,
   404	
   405	ℓ ∈ {S, M, L},
   406	
   407	(4)
   408	
   409	where ⊙ denotes element-wise multiplication. Each x̃ℓ inherits
   410	the identity label y and the clothing label c. The augmented set
   411	{x̃S , x̃M , x̃L } is sampled together with originals, creating abundant
   412	positive pairs of the same identity under different hairstyles. Notably,
   413	we leverage triplet loss to optimize the feature space, pulling closer
   414	representations of the same identity across varying hairstyles and
   415	clothing while pushing apart those from different identities.
   416	2.3. Cloth-Preserved Random Erasing (CPRE)
   417	Although clothing contains strong visual cues, they are considered
   418	identity-unrelated features in our task. A common practice in existing
   419	methods is to remove this information entirely to compel the model to
   420	learn identity-related features. However, removing the entire clothing
   421	region also discards useful information about body structure and
   422	spatial context. To address this limitation, we propose Cloth-Preserved
   423	Random Erasing (CPRE), we design Cloth-Preserved Random Erasing
   424	(CPRE) to erase only within the clothing region, retaining a random
   425	proportion of clothing pixels, forcing the model to rely more on
   426	identity-related cues (face, limbs, shape). Let Mcloth ∈ {0, 1}H×W be
   427	the clothing mask (slightly dilated to cover boundary errors). Sample
   428	a keep ratio r ∈ [rmin , rmax ] and draw Kr ∈ {0, 1}H×W inside the
   429	
   430	clothing
   431	a proportion r is preserved
   432	 region such that approximately
   433	
   434	(i.e., E Kr (i, j) Mcloth (i, j) = 1 = r) . The erased image is
   435	
   436	Table 1. Results on PRCC and LTCC. The best result is in bold, and
   437	the second-best result is underlined.
   438	Methods
   439	
   440	
   441	
   442	xerase = (1−Mcloth )+Mcloth ⊙Kr
   443	
   444	
   445	
   446	
   447	
   448	⊙x+ Mcloth ⊙(1−Kr ) ⊙ϵ,
   449	(5)
   450	
   451	where ϵ is a constant fill (zero).
   452	Pixel-wise image. Equivalently, for each pixel (i, j),
   453	
   454	
   455	xi,j ,
   456	xerase
   457	xi,j ,
   458	i,j =
   459	
   460	ϵ,
   461	
   462	Mcloth (i, j) = 0,
   463	Mcloth (i, j) = 1 ∧ Kr (i, j) = 1,
   464	Mcloth (i, j) = 1 ∧ Kr (i, j) = 0 .
   465	
   466	(6)
   467	
   468	image averaging. When CPRE is enabled, the Mix Decision module
   469	constructs input batches by the raw image x and erased image xerase
   470	at a 1:1 ratio.
   471	2.4. Region-based Parsing Attention (RPA)
   472	While CPRE reduces reliance on clothing features, hairstyle
   473	remains a prominent, identity-unrelated distractor. To mitigate this
   474	specific problem, we propose Region-based Parsing Attention (RPA),
   475	a lightweight attention mechanism that uses human parsing priors to guide the model’s focus.It generates a spatial attention map
   476	that emphasizes identity-related regions and minimizes attention to
   477	identity-unrelated features, such as hair, helping the model learn more
   478	robust identity representations.
   479	Backbone and ID head. A backbone B(·) produces a feature map
   480	F ∈ RC×H×W , a shallow ID head yields FID = ϕid (F ).
   481	Attention prediction and gating. Given FID , a 1 × 1 convolution
   482	predicts attention logits S ∈ R1×H×W :
   483	S = W ∗ FID + b,
   484	
   485	exp(Sij )
   486	∈ (0, 1),
   487	u,v exp(Suv )
   488	
   489	Âij = P
   490	
   491	(7)
   492	
   493	where W ∈ R1×C×1×1 . The gated ID features are
   494	FID ” = FID ⊙ Â,
   495	
   496	(8)
   497	
   498	with Â broadcast along channels, global average pooling (GAP) of
   499	FID ” is used for downstream losses. At test time, the RPA gate is
   500	disabled and the model uses the ungated FID .
   501	Parsing-guided attention loss. Given the parsing masks Mface , Mlimbs
   502	, Mhair , with
   503	Mface , Mlimbs , Mhair ∈ {0, 1}H×W ,
   504	we define the normalized positive target
   505	T+ =
   506	
   507	Mface + Mlimbs
   508	.
   509	⟨1, Mface + Mlimbs ⟩ + ε
   510	
   511	(9)
   512	
   513	We supervise Â toward T+ and penalize mass on hair:
   514	Latt = −⟨T+ , log Â⟩ + λneg
   515	
   516	⟨Â, Mhair ⟩
   517	.
   518	⟨1, Mhair ⟩ + ε
   519	
   520	When parsing masks are absent, this term is omitted.
   521	
   522	(10)
   523	
   524	Year
   525	
   526	PRCC
   527	Cloth-Changing
   528	Standard
   529	
   530	LTCC
   531	Cloth-Changing Standard
   532	
   533	R1
   534	
   535	mAP
   536	
   537	R1
   538	
   539	R1
   540	
   541	mAP
   542	
   543	R1
   544	
   545	mAP
   546	
   547	HACNN [20]
   548	CVPR ’18 21.8
   549	PCB [21]
   550	ECCV ’18 41.8
   551	IANet [22]
   552	CVPR ’19 46.3
   553	FSAM [7]
   554	CVPR ’21 54.5
   555	AIM [12]
   556	CVPR ’23 57.9
   557	CCFA [13]
   558	CVPR ’23 61.2
   559	Instruct-ReID [23] CVPR’24 54.2
   560	LIFTCAP [10]
   561	TVT’24
   562	54.3
   563	JIMGP [24]
   564	TMM’24 57.3
   565	CISupNet [25]
   566	ICASSP’25 58.3
   567	FAIM [26]
   568	TMM’25 59.8
   569	RLQ [27]
   570	arXiv’25 64.0
   571	
   572	38.7
   573	46.9
   574	58.3
   575	58.4
   576	52.3
   577	55.6
   578	65.8
   579	58.2
   580	62.5
   581	63.2
   582	
   583	82.5
   584	21.6
   585	99.8 97.0 23.5
   586	99.4 98.3 25.0
   587	98.8
   588	38.5
   589	100.0 99.9 40.6
   590	99.6 98.7 45.3
   591	100.0 99.8 37.0
   592	99.7 99.8 43.4
   593	100.0 99.8 41.5
   594	100.0 100.0 48.2
   595	100.0 99.8 46.4
   596	
   597	9.3
   598	10.0
   599	12.6
   600	16.2
   601	19.4
   602	22.1
   603	39.7
   604	18.2
   605	19.2
   606	27.5
   607	21.5
   608	
   609	60.2
   610	65.1
   611	63.7
   612	73.2
   613	76.3
   614	75.8
   615	75.8
   616	76.0
   617	79.5
   618	76.9
   619	
   620	26.7
   621	30.6
   622	31.0
   623	35.4
   624	41.1
   625	42.5
   626	52.0
   627	41.6
   628	53.4
   629	41.8
   630	
   631	CAL [11]
   632	ours
   633	
   634	55.8
   635	63.4
   636	
   637	100.0
   638	100.0
   639	
   640	18.0
   641	19.3
   642	
   643	74.2
   644	78.7
   645	
   646	40.8
   647	60.1
   648	
   649	CVPR ’22
   650	-
   651	
   652	55.2
   653	65.1
   654	
   655	mAP
   656	
   657	99.8
   658	99.1
   659	
   660	40.1
   661	41.6
   662	
   663	2.5. Objective
   664	We optimize a weighted sum of four terms. Here, Lid and Ltri
   665	are well-known identity classification and triplet losses, Latt is the
   666	parsing-guided attention loss defined in Sec. 2.4, and Lcal denotes the
   667	clothes-adversarial loss adopted from our baseline CAL[11].
   668	Ltotal = Lid + λtri Ltri + λatt Latt + λcal Lcal ,
   669	
   670	(11)
   671	
   672	where λtri , λatt , λcal balance the contributions.
   673	Inference. At test time, we only use the ungated FID branch, followed
   674	by ℓ2 normalization and cosine similarity for retrieval.
   675	3. EXPERIMENTS

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf' - | nl -ba | sed -n '332,886p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '305,715p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   332	3. Approach
   333	
   334	where 𝐿shuffled(𝑛) is designed in Eq. (4).
   335	𝐿shuffled(𝑛) (𝜃 , , ) = 𝐿→
   336	(𝜃 ,  ,   ) + 𝐿→
   337	(𝜃 ,  ,   ),
   338	shuffled(𝑛)
   339	shuffled(𝑛)
   340	
   341	In this section, we begin by exploring the shuffled meta learning (SML) method. Next, we delve into the design of the memory
   342	meta learning (MML) method. Subsequently, we integrate SML and
   343	MML to develop the memory-augmented shuffled meta (MASM) learning approach. Finally, we present the comprehensive architecture for
   344	visible–infrared person re-identification (VIPR) based on MASM.
   345	Notations: To facilitate a clearer understanding of our approach,
   346	we define the relevant notations used throughout this paper. Following
   347	the general meta learning framework (Chen et al., 2019), the VIPR task
   348	is decomposed into multiple subtasks, where the 𝑛th subtask is denoted
   349	as 𝑛 . Assuming that all subtasks are sampled from a task distribution
   350	𝑝( ), the optimization objective of the general meta learning method
   351	is formulated as Eq. (1):
   352	𝜃 = ar g min E𝑛 ∼𝑝( ) [E,∼𝑛 [𝐿(𝑛) (𝜃 , , )]],
   353	𝜃
   354	
   355	where 𝐿→
   356	(𝜃 ,  ,   ) and 𝐿→
   357	(𝜃 ,  ,   ) are calculated as
   358	shuffled(𝑛)
   359	shuffled(𝑛)
   360	in Eqs. (5) and (6).
   361	
   362	(6)
   363	
   364	By reformulating the meta-learning objective as in Eqs. (5) and
   365	(6), SML systematically cycles through all 𝐾 possible query-support
   366	combinations per identity (versus single pair sampling in conventional
   367	meta-learning). This ensures that every sample serves as both query and
   368	support data across different training cycles, increasing the usage rate
   369	from 1∕𝐾 to 100%. In addition, for each modality (visible/infrared), the
   370	loss terms 𝐿→
   371	and 𝐿→
   372	aggregate gradients from all 𝐾 pershuffled(𝑛)
   373	shuffled(𝑛)
   374	mutations. This creates 𝐾 2 implicit cross-modal interactions compared
   375	to the single pairing in the general meta learning method, effectively
   376	expanding the training diversity without additional data.
   377	In summary, our shuffled meta learning approach improves the
   378	data partitioning process without altering the original intention. It
   379	guarantees that the model is exposed to a wide range of query and
   380	support samples during each training cycle. By introducing various
   381	combinations of query and support samples, our approach obtains the
   382	comprehensive data utilization to enhance the discrimination of a VIPR
   383	model.
   384	
   385	(1)
   386	
   387	3.2. Memory meta learning
   388	Memory meta learning (MML) is a novel approach that leverages
   389	memory banks to store historical information, thus enhancing the
   390	learning process. The core idea is to maintain a memory of previously
   391	encountered tasks, which allows the model to draw upon past experiences to improve its performance on new and related tasks. This is
   392	particularly beneficial in scenarios where data is scarce or where tasks
   393	exhibit significant variability. As shown in Fig. 2, the formalization of
   394	MML is represented in Eq. (7):
   395	
   396	(2)
   397	
   398	where  = {𝑖 , 𝑖 }, 𝑖 ∈ [𝐾] and  = {𝑖 , 𝑖 }, 𝑖 ∈ [𝐾]; 𝐿→
   399	(𝜃 , 𝑖 ,
   400	(𝑛)
   401	 ,   ) denote a V2I meta metric loss and a I2V meta
   402	𝑖 ) and 𝐿→
   403	(𝜃
   404	,
   405	
   406	𝑖
   407	𝑖
   408	(𝑛)
   409	metric loss. To be more specific, assume that 𝐶 identities with 𝐾 visible
   410	images and 𝐾 infrared images sampled to train a subset, and a deep
   411	network is applied to extract from sampled visible images and infrared
   412	images to 𝑑-dimension features, i.e., visible features {𝑥𝑐 ,𝑘 ∈ R𝑑 |𝑐 ∈
   413	[𝐶]1 ; 𝑘 ∈ [𝐾]} and infrared features {𝑦𝑐 ,𝑘 ∈ R𝑑 |𝑐 ∈ [𝐶]; 𝑘 ∈ [𝐾]}. Then,
   414	we can construct query and support sets, as follows. For each identity,
   415	we select the 𝑖th visible sample as the query set 𝑖 = {𝑥𝑐 ,𝑖 |𝑐 ∈ [𝐶]},
   416	and the corresponding visible support set is obtained as 𝑖 = {𝑐 ,𝑖 |𝑐 ∈
   417	[𝐶]}, where 𝑐 ,𝑖 = {𝑥𝑐 ,𝑘 |𝑘 ∈ [𝐾], 𝑘 ≠ 𝑖} is a meta cell consisting of
   418	visible samples of the 𝑐th identity but without the 𝑖th sample. Similarly,
   419	we select the 𝑖th infrared sample as the infrared query set 𝑖 = {𝑦𝑐 ,𝑖 |𝑐 ∈
   420	[𝐶]} and the corresponding infrared support set 𝑖 = {𝑐 ,𝑖 |𝑐 ∈ [𝐶]},
   421	where 𝑐 ,𝑖 = {𝑦𝑐 ,𝑘 |𝑘 ∈ [𝐾], 𝑘 ≠ 𝑖} is the meta cell consisting of infrared
   422	samples of the 𝑐th identity but without the 𝑖th sample.
   423	From Eq. (2), it can be found that the general meta learning method
   424	only selects one visible sample and one infrared sample for each class
   425	to construct query sets, leading to limited query and support samples
   426	in each training cycle. From the purpose of a comprehensive data utilization to enhance the model’s ability of learning more discriminative
   427	features, we propose a shuffled meta learning method, which circularly
   428	divides query and support sets. The proposed shuffled meta metric loss
   429	𝐿shuffled(𝑛) is formulated as Eq. (3).
   430	
   431	1
   432	
   433	1 ∑ →
   434	𝐿
   435	(𝜃 , 𝑖 , 𝑖 ).
   436	𝐾 𝑖=1 (𝑛)
   437	𝐾
   438	
   439	𝐿→
   440	(𝜃 ,  ,   ) =
   441	shuffled(𝑛)
   442	
   443	As shown in Fig. 2, due to the fact that there are two retrieval
   444	modes in the VIPR task, i.e., visible-to-infrared (V2I) and infrared-tovisible (I2V) retrieval modes, the general meta learning method can be
   445	reformulated to optimize these two retrieval modes as Eq. (2).
   446	
   447	𝜃
   448	
   449	(5)
   450	
   451	and
   452	
   453	3.1. Shuffled meta learning
   454	
   455	𝜃 = ar g min E𝑛 ∼𝑝( ) [E,∼𝑛 [𝐿shuffled(𝑛) (𝜃 , , )]],
   456	
   457	1 ∑ →
   458	𝐿
   459	(𝜃 , 𝑖 , 𝑖 ),
   460	𝐾 𝑖=1 (𝑛)
   461	𝐾
   462	
   463	𝐿→
   464	(𝜃 ,  ,   ) =
   465	shuffled(𝑛)
   466	
   467	where E denotes an expectation calculation; 𝜃 denotes the learnable parameters of the network;  and  denote the query set and the support
   468	set; 𝐿(𝑛) (𝜃 , , ) denotes the general meta metric loss of the sampled
   469	subtask 𝑛 . Specifically, 𝐿(𝑛) (𝜃 , , ) can be calculated differently to
   470	adapt to various tasks. For example, in Chen et al. (2019), 𝐿(𝑛) (𝜃 , , )
   471	is derived from the triplet loss function for the re-identification task.
   472	In this paper, we design 𝐿(𝑛) (𝜃 , , ) for the VIPR task, taking modal
   473	discrepancies into consideration. More details are provided in the
   474	following subsection.
   475	
   476	(𝜃 , 𝑖 , 𝑖 ),
   477	𝐿(𝑛) (𝜃 , , ) = 𝐿→
   478	(𝜃 , 𝑖 , 𝑖 ) + 𝐿→
   479	(𝑛)
   480	(𝑛)
   481	
   482	(4)
   483	
   484	𝜃 = ar g min E𝑛 ∼𝑝( ) [E,∼𝑛 [𝐿MM(𝑛) (𝜃 , , )]],
   485	𝜃
   486	
   487	(7)
   488	
   489	where  denotes a memory-based support set that contains critical
   490	features from previous tasks. The loss function 𝐿MM represents the
   491	memory meta (MM) loss, which is formulated in Eq. (8):
   492	𝐿MM(𝑛) (𝜃 , , ) = 𝐿→
   493	(𝜃 ,  ,  ) + 𝐿→
   494	(𝜃 ,  ,  ),
   495	shuffled(𝑛)
   496	shuffled(𝑛)
   497	
   498	(8)
   499	
   500	where  ∈ R𝑀×𝑑 and  ∈ R𝑀×𝑑 represent the visible and infrared
   501	memory banks, respectively. Here, 𝑀 = 𝐿×𝐶 ×𝐾, where 𝐿 denotes the
   502	number of subtasks stored in a memory bank, 𝐶 indicates the number of
   503	identities, and 𝐾 represents the number of visible and infrared images
   504	sampled in each subtask. The dimension 𝑑 corresponds to the feature
   505	space of the stored representations.
   506	During the training of the top 𝐿 subtasks, the features from the
   507	current subset are directly enqueued into the memory banks. This
   508	process allows the model to maintain a dynamic repository of relevant
   509	features that can be accessed during subsequent training iterations.
   510	However, as the memory banks have a size limitation, once they reach
   511	capacity, the oldest subset of features is dequeued to make space for
   512	the newest subset. This mechanism ensures that the most relevant and
   513	recent information is prioritized, while still retaining a diverse set of
   514	historical data.
   515	Building on SML’s data-efficient framework, MML introduces memory banks  and  to store 𝑀 = 𝐿 × 𝐶 × 𝐾 historical features across
   516	modalities. This mechanism dynamically expands the effective training
   517	
   518	(3)
   519	
   520	𝑐 ∈ [𝐶] denote 𝑐 = 1, 2, … , 𝐶..
   521	4
   522	
   523	Neural Networks 191 (2025) 107812
   524	
   525	H. Wu et al.
   526	
   527	Fig. 2. The framework of the proposed memory-augmented shuffled meta (MASM) learning method.
   528	
   529	distribution by reusing historical data as synthetic support sets, circumventing the need for additional raw samples. During training, the top-𝐿
   530	subtask features are enqueued into memory banks, while outdated
   531	features are dequeued upon capacity limits, ensuring the model always
   532	utilizes the most informative features. By leveraging these memory
   533	banks, MML expands cross modal interaction from 𝐾 2 times on SML to
   534	𝐾 2 𝑡𝑖𝑚𝑒𝑠𝐿 times. Overall, MML represents a significant advancement
   535	in the field of meta-learning by incorporating memory mechanisms
   536	that allow for the retention and utilization of past experiences. This
   537	approach not only enhances the model’s ability to generalize across
   538	tasks but also provides a robust framework for tackling challenges
   539	associated with data scarcity and variability.
   540	
   541	function’s support set is historical features stored in memories. We take
   542	the CM loss function as an example to introduce the calculation process.
   543	According to the designed shuffled meta learning method introduced
   544	in Section 3.1, we shuffle to utilize 𝑖 (𝑖 ∈ [𝐾]), then, the V2I shuffled
   545	meta metric loss 𝐿→
   546	and the I2V shuffled meta metric loss 𝐿→
   547	shuffled
   548	shuffled
   549	are formulated as Eqs. (12) and (13):
   550	
   551	𝐿→
   552	(𝜃 , 𝑖 , 𝑖 ) + 𝐿→
   553	(𝜃 , 𝑖 , 𝑖 ).
   554	shuffled
   555	shuffled
   556	
   557	1 ∑ →
   558	𝐿
   559	(𝜃 , 𝑖 , 𝑖 ).
   560	𝐾 𝑖 focal
   561	
   562	(13)
   563	
   564	𝐿→
   565	(𝜃 , 𝑖 , 𝑖 ) =
   566	focal
   567	
   568	1 ∑
   569	(1 − 𝑝→
   570	)𝛾 ⋅ 𝑙𝑜𝑔(𝑝→
   571	),
   572	(𝑐 ,𝑐),𝑖
   573	(𝑐 ,𝑐),𝑖
   574	𝐶 𝑐=1
   575	
   576	(14)
   577	
   578	𝐿→
   579	(𝜃 , 𝑖 , 𝑖 ) =
   580	focal
   581	
   582	1 ∑
   583	(1 − 𝑝→
   584	)𝛾 ⋅ 𝑙𝑜𝑔(𝑝→
   585	),
   586	(𝑐 ,𝑐),𝑖
   587	(𝑐 ,𝑐),𝑖
   588	𝐶 𝑐=1
   589	
   590	(15)
   591	
   592	𝐶
   593	
   594	𝐶
   595	
   596	where 𝛾 > 0 is an adjustable parameter; (1 − 𝑝)𝛾 denotes the modulating
   597	factor; 𝑝→
   598	and 𝑝→
   599	represents probability distributions in V2I and
   600	(𝑐 ,𝑐),𝑖
   601	(𝑐 ,𝑐),𝑖
   602	
   603	(9)
   604	
   605	I2V retrieval modes, respectively. General formulations of probability distributions are calculated by the softmax function, as shown in
   606	Eqs. (16) to (17).
   607	
   608	(10)
   609	
   610	(
   611	)
   612	→
   613	exp −𝑑(𝑐
   614	,𝑛),𝑖
   615	
   616	𝑝→
   617	= ∑𝐶
   618	(𝑐 ,𝑛),𝑖
   619	
   620	where 𝜆 > 0 is a manual parameter to control the weight of 𝐿MM(𝑛) ;
   621	𝐿MM(𝑛) denotes the memory meta (MM) loss function formulated in
   622	Eq. (8), which is responsible for learning global meta metrics by storing
   623	historical information. 𝐿CM(𝑛) represents the current meta (CM) loss
   624	function, which is in charge of optimizing meta metrics from the
   625	current subtask; We simplify 𝐿MASM(𝑛) , 𝐿CM(𝑛) , and 𝐿MM(𝑛) to 𝐿MASM ,
   626	𝐿CM , and 𝐿MM .
   627	The CM loss is designed in Eq. (11).
   628	𝐿CM (𝜃 , , ) =
   629	
   630	𝐿→
   631	(𝜃 ,  ,   ) =
   632	shuffled
   633	
   634	Furthermore, to make the training model focus more on samples
   635	that are difficult to classify and reduce the weight of samples that are
   636	easy to classify, inspired by the focal loss function (Zhou et al., 2020),
   637	we concrete 𝐿→
   638	and 𝐿→
   639	as Eqs. (14) and (15), respectively.
   640	focal
   641	focal
   642	
   643	where MASM loss function 𝐿MASM(𝑛) (𝜃 , , , ) is further designed as
   644	Eq. (10).
   645	𝐿MASM(𝑛) (𝜃 , , , ) = 𝐿CM(𝑛) (𝜃 , , ) + 𝜆𝐿MM(𝑛) (𝜃 , , ),
   646	
   647	(12)
   648	
   649	𝐾
   650	
   651	In the shuffled meta learning formulated in Eq. (3), the scale of
   652	subsets sampled from the entire training set being available in each
   653	subtask is limited. For that, we further extend Eq. (3) to a memoryaugmented shuffled meta (MASM) learning loss function by introducing
   654	the memory meta learning (MML). The MML leverages memory banks
   655	to historical information, which acquires a more global similarity metrics learning, as shown in Fig. 2. The MASM loss function is designed
   656	in Eq. (9), as follows.
   657	𝜃
   658	
   659	1 ∑ →
   660	𝐿
   661	(𝜃 , 𝑖 , 𝑖 ),
   662	𝐾 𝑖 focal
   663	𝐾
   664	
   665	3.3. Memory-augmented shuffled meta learning
   666	
   667	𝜃 = ar g min E𝑛 ∼𝑝( ) [E,∼𝑛 [𝐿MASM(𝑛) (𝜃 , , , )]],
   668	
   669	𝐿→
   670	(𝜃 ,  ,   ) =
   671	shuffled
   672	
   673	→
   674	𝑗=1 exp(−𝑑(𝑐 ,𝑗),𝑖 )
   675	
   676	(
   677	)
   678	→
   679	exp −𝑑(𝑐
   680	,𝑛),𝑖
   681	
   682	𝑝→
   683	= ∑𝐶
   684	(𝑐 ,𝑛),𝑖
   685	
   686	→
   687	𝑗=1 exp(−𝑑(𝑐 ,𝑗),𝑖 )
   688	
   689	,
   690	
   691	(16)
   692	
   693	,
   694	
   695	(17)
   696	
   697	→
   698	→
   699	where 𝑐 ∈ [𝐶] and 𝑛 ∈ [𝐶]; 𝑑(𝑐
   700	and 𝑑(𝑐
   701	are the distance between
   702	,𝑛),𝑖
   703	,𝑛),𝑖
   704	
   705	the query and the meta cell. In the V2I retrieval mode, assume that 𝑥𝑐 ,𝑖
   706	is a randomly-selected query, the distance between query 𝑥𝑐 ,𝑖 and meta
   707	cell 𝑐 ,𝑖 can be calculated as Eq. (18):
   708	(
   709	)‖
   710	𝐾
   711	(
   712	) ‖
   713	∑
   714	‖
   715	‖
   716	1
   717	→
   718	
   719	‖
   720	‖ .
   721	𝑑(𝑐 ,𝑛),𝑖 = 𝐷 𝑥𝑐 ,𝑖 , 𝑛,𝑖 = ‖𝑥𝑐 ,𝑖 −
   722	𝑦
   723	(18)
   724	𝐾 − 1 𝑘=1,𝑘≠𝑖 𝑛,𝑘 ‖
   725	‖
   726	‖
   727	‖
   728	‖2
   729	
   730	(11)
   731	
   732	Comparing CM loss function in Eq. (11) and MM loss function in
   733	Eq. (8), one can see that the main difference is that the MM loss
   734	5
   735	
   736	Neural Networks 191 (2025) 107812
   737	
   738	H. Wu et al.
   739	
   740	Similarly, in the I2V retrieval mode, the distance between a
   741	randomly-selected query 𝑦𝑐 ,𝑖 and the corresponding meta cell 𝑛,𝑖 is
   742	calculated as Eq. (19).
   743	(
   744	)‖
   745	𝐾
   746	(
   747	) ‖
   748	∑
   749	‖
   750	‖
   751	1
   752	→
   753	
   754	‖
   755	𝑑(𝑐 ,𝑛),𝑖 = 𝐷 𝑦𝑐 ,𝑖 , 𝑛,𝑖 = ‖𝑦𝑐 ,𝑖 −
   756	𝑥𝑛,𝑘 ‖
   757	(19)
   758	‖ .
   759	𝐾
   760	−
   761	1
   762	‖
   763	‖
   764	𝑘=1,𝑘≠𝑖
   765	‖
   766	‖2
   767	
   768	The total loss function contains MASM and the traditional crossentropy (CE) loss functions, which is formulated as Eq. (22).
   769	𝐿 = 𝐿MASM + 𝐿CE ,
   770	
   771	where 𝐿MASM is the MASM loss defined in Eq. (9); 𝐿CE is the CE loss,
   772	which is formulated as Eq. (23).
   773	1 ∑∑
   774	𝑙(𝑓 , 𝑐)log(𝑝(𝑐|𝑓𝑖 )),
   775	𝑀 𝑖=1 𝑐=1 𝑖
   776	𝑀
   777	
   778	The CM loss function, when applied to a limited dataset, may
   779	introduce uncertainties due to inherent modality gaps. To enhance
   780	the stability of meta-learning, we propose shuffled meta regularization
   781	(SMR), which addresses the asymmetric predictions between visibleto-infrared and infrared-to-visible matching directions caused by these
   782	uncertainties, as formulated in Eq. (20).
   783	√
   784	√
   785	) (
   786	)]2
   787	√ 𝐶 𝐶 [(
   788	𝐾
   789	𝐾
   790	∑∑
   791	1 √
   792	1 ∑ →
   793	1 ∑ →
   794	(20)
   795	√
   796	𝑝
   797	𝑝
   798	𝐿SMR =
   799	−
   800	,
   801	𝐾 𝑖=1 (𝑐 ,𝑛),𝑖
   802	𝐾 𝑖=1 (𝑐 ,𝑛),𝑖
   803	𝐶 2 𝑐=1 𝑛=1
   804	
   805	CE = −
   806	
   807	𝐶
   808	
   809	(23)
   810	
   811	where 𝑓𝑖 represents the 𝑖th sample’s feature; 𝑝(𝑐|𝑓𝑖 ) represents the
   812	posterior probability of the 𝑖th sample 𝑓𝑖 belongs to the 𝑐th class 𝐶𝑐 ,
   813	which is calculated by the softmax function, as shown in Eq. (24).
   814	exp(𝜃𝑐𝖳 𝑓𝑖 )
   815	𝑝(𝑐|𝑓𝑖 ) = ∑𝐶
   816	,
   817	𝖳
   818	𝑗=1 exp(𝜃𝑗 𝑓𝑖 )
   819	
   820	(24)
   821	
   822	where 𝜃𝑗 ∈ R𝑑 denotes the learnable parameter for projecting to the
   823	𝑗th class 𝐶𝑗 ; 𝑙(𝑓𝑖 , 𝑐) is a label smooth indicator defined as Eq. (25).
   824	
   825	where 𝑝→
   826	and 𝑝→
   827	are defined in Eqs. (16) and (17). By enforcing
   828	(𝑐 ,𝑛),𝑖
   829	(𝑐 ,𝑛),𝑖
   830	consistency between the averaged cross-modal prediction probabilities
   831	∑ →
   832	1 ∑ →
   833	𝑝(𝑐 ,𝑛),𝑖 and 𝐾1
   834	𝑝(𝑐 ,𝑛),𝑖 across all 𝐶 classes, SMR implicitly aligns
   835	𝐾
   836	the latent feature spaces, enabling the model to learn a more stable
   837	meta-metric.
   838	Therefore, the complete CM loss function is reformulated in
   839	Eq. (21).
   840	𝐿CM (𝜃 , , ) = 𝐿→
   841	(𝜃 , 𝑖 , 𝑖 ) + 𝐿→
   842	(𝜃 , 𝑖 , 𝑖 ) + 𝜉 𝐿SMR
   843	shuffled
   844	shuffled
   845	
   846	(22)
   847	
   848	(𝐶 − 1)𝛿
   849	⎧
   850	, 𝑓𝑖 ∈ 𝐶th class,
   851	⎪1 −
   852	𝐶
   853	𝑙(𝑓𝑖 , 𝑐) = ⎨
   854	𝛿
   855	⎪ ,
   856	𝑓𝑖 ∉ 𝐶th class,
   857	⎩𝐶
   858	
   859	(25)
   860	
   861	where 𝛿 > 0 is a manually set parameter to control the label-smoothing
   862	degree, which is commonly set to 0.1 in practice. At last, following
   863	famous person re-identification baseline works (Luo, Gu, Liao, Lai, &
   864	Jiang, 2019; Ye, Shen et al., 2022), the MASM loss function is placed
   865	on the GeP layer, and the CE loss is assigned on the BN layer.
   866	
   867	(21)
   868	
   869	where 𝜉 > 0 is a manually-adjusted parameter to control the weight of
   870	SMR. As shown in Fig. 2, the SMR works on the current meta learning
   871	to restrict the variation of the distance metrics in different shuffle in
   872	different retrieval, improving the current meta learning.
   873	Our approach operates on the principle of pulling positive pairs
   874	closer together while pushing negative pairs apart in the embedding
   875	space, akin to conventional contrastive learning paradigms. The key
   876	advancement lies in reformulating this contrastive process from a metalearning perspective. We dynamically construct query-support sets using a shuffling strategy, which replaces traditional sample pairs or fixed
   877	cluster centers. This dynamic construction aligns more closely with reidentification tasks, where the objective is to retrieve images from a
   878	gallery (corresponding to the support set in meta-learning) that match
   879	the identity of the query image. Furthermore, our memory mechanism
   880	significantly enhances this paradigm by effectively leveraging historical
   881	query-support interactions. This mechanism allows the model to retain
   882	and utilize past experiences, enabling it to capture long-term feature
   883	relationships that are crucial for improving the accuracy of the retrieval
   884	process.
   885	
   886	4. Experiments and analysis

 succeeded in 0ms:
   305	3. Meta Pairwise Relationship Distillation
   306	Given an unlabeled dataset X = {xi }N
   307	i=1 , where xi denotes the ith input image, and N denotes the number of
   308	training samples, the MPRD estimates the pairwise pseudo
   309	labels for feature learning. As illustrated in Figure 2, the
   310	CNN learns discriminative features supervised by the pairwise pseudo labels generated by GCN; while the GCN estimates the pairwise pseudo labels based on CNN features.
   311	This interdependency is practically solved via alternating
   312	optimization of the GCN and the CNN.
   313	
   314	3.1. CNN
   315	Network backbone. The CNN module extracts discriminative features, which allows nearest neighbor search in the
   316	feature space. For simplicity, we adopt the backbone network
   317	in [8] as our CNN choice* , which consists of a feature extraction module and a feature memory module. In practice, the
   318	feature extraction module F extracts a d-dimensional feature
   319	F(xi ) from each input image xi , and then `2 -normalized by
   320	F̃(xi ) ← F(xi )/kF(xi )k2 , kF(xi )k2 indicates the norm
   321	of F(xi ), the feature memory M stores all the features of
   322	training images. The feature memory is updated at the tth
   323	iteration as follows.
   324	M(t) [i] ← γ (t) F̃(xi ) + (1 − γ (t) )M̃(t−1) [i],
   325	M̃(t) [i] ← M(t) [i]/kM(t) [i]k2 ,
   326	* Our method is compatible with various network backbones.
   327	
   328	(1)
   329	
   330	LF =
   331	
   332	n
   333	X
   334	1X δ
   335	ζ(α(λ1 − hF̃(xi ), M̃[j]i))
   336	n i=1 | Pi |
   337	xj ∈Pi
   338	
   339	(2)
   340	X
   341	1
   342	+
   343	ζ(α(hF̃(xi ), M̃[j]i − λ2 )),
   344	| N∗i |
   345	∗
   346	xj ∈Ni
   347	
   348	where h·, ·i represent inner product, since both F̃ (xi ) and
   349	M̃[j] are `2 -normalized unit vector, hF̃(xi ), M̃[j]i denote
   350	the cosine similarity between them, ζ(x) = log(1 + ex ), n
   351	is the batch size, | · | denotes the cardinality (number of elements), δ indicates the importance of positive pairs against
   352	negative pairs, λ1 and λ2 denote two margin parameters, and
   353	α is an amplification factor. Besides, Pi and Ni represent
   354	the positive pair list and the negative pair list, respectively.
   355	As |Ni ||Pi | in practice, we further focus on the hard negative pair list N∗i with fixed size r as follows.
   356	N∗i = {xj | xj ∈ top(hF̃(xi ), M̃[j]i, r), xj ∈ Ni }, (3)
   357	where top(·, r) represent the r largest samples. Therefore,
   358	N∗i contains the r closest negative samples to the query xi
   359	in the embedding feature space.
   360	After training the CNN, all positive pairs concentrate
   361	within a radius of λ1 ; while all negative pairs locate elsewhere with a distance of at least λ2 . Afterwards, a nearest
   362	neighbor searching algorithm can be applied to solve the
   363	person Re-ID problem.
   364	
   365	3643
   366	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:04 UTC from IEEE Xplore. Restrictions apply.
   367	
   368	Feature Memory
   369	
   370	(t )
   371	
   372	( t +1)
   373	
   374	Feature Memory
   375	
   376	(t )
   377	
   378	( t +1)
   379	
   380	( t +1)
   381	
   382	q
   383	
   384	i2
   385	
   386	i1
   387	
   388	Training F
   389	After t th Iter
   390	
   391	i3
   392	
   393	i1
   394	i2
   395	
   396	Pseudo Labels
   397	
   398	q
   399	q
   400	
   401	q
   402	
   403	i3
   404	NNk(q)
   405	Pairwise
   406	Neighborhood
   407	Structure
   408	
   409	NNk(i1)
   410	
   411	i1
   412	i2
   413	i3
   414	
   415	i1
   416	NNk(i2)
   417	
   418	i1
   419	New
   420	Relationship
   421	
   422	q
   423	q
   424	q
   425	
   426	i1
   427	i1
   428	
   429	(q, i2)
   430	
   431	i3
   432	q
   433	i1
   434	
   435	q
   436	i3
   437	NNk(i1)
   438	
   439	NNk(i3)
   440	
   441	NNk(i2)
   442	
   443	NNk(i3)
   444	
   445	Training g
   446	After t-th Iter
   447	
   448	(t )
   449	
   450	(q, i1)
   451	
   452	i3
   453	i2
   454	
   455	i2
   456	
   457	(q, i3)
   458	
   459	Du (i1, i3)
   460	
   461	(a) Construct Pairwise Neighborhood Structures
   462	
   463	Ds
   464	
   465	(i1, i2)
   466	
   467	(b) Training GCN Model
   468	
   469	Figure 3. Illustration of the pairwise neighborhood structure and the training strategy of MPRD, where the red arrow means prohibit
   470	execution until training the CNN model after tth iteration, the blue line means the pairs’ relationship is not yet judged, the green line means
   471	the pair is considered to have the positive label, and the red line indicates negative label.
   472	
   473	3.2. GCN
   474	Network backbone. The GCN estimates the pseudo labels of sample pairs, so as to guide the CNN training with
   475	unlabeled data. As shown in Figure 2 (b), it takes the pairwise neighborhood structures Gij = (Aij , Vij ) as inputs,
   476	where Aij denotes the adjacent matrix, and Vij indicates
   477	the node embedding. For image xi and each images xj in
   478	NNk (xi ) = top(hM̃[i], M̃[j]i, k), the pairwise neighborhood structure can be constructed by connecting images xi
   479	and xj with their neighbors. Therefore, the adjacent matrix
   480	of Gij can be defined as follows:
   481	
   482	1, xa ∈ NNk (xb )
   483	Aij (b, a) = Aij (a, b) =
   484	, (4)
   485	0, otherwise
   486	where b ∈ {i, j} denotes an image index in the extracted
   487	sample pair (xi , xj ). Besides, the node embedding of Gij
   488	can be achieved in two steps as follows. (1) We use the
   489	Double-Radius Node Labeling (DRNL) [32] to generate
   490	the position embedding of each node in Gij , which can
   491	distinguish nodes with different positions relative to sample
   492	pair (xi , xj ). (2) We concatenate the position embedding
   493	and the feature embedding of the nodes in Gij as Vij .
   494	The structure of our GCN is shown in Figure 2 (b), which
   495	consists of two graph convolutional layers, one graph aggregation layer and one multi-layer perception. In particular, the
   496	multi-layer perception (with its parameters denoted as Θm )
   497	contains two fully-connected layers, the graph aggregation
   498	layer (with its parameters denoted as Θa ) includes a maxpooling layer and a 1-D convolutional layer. The recursive
   499	
   500	function of our graph convolutional layers is,
   501	(l+1)
   502	
   503	Yij
   504	
   505	(l)
   506	
   507	(0)
   508	
   509	= σ(D−1 (Aij + I)Yij Θg(l) ), Yij = Vij , (5)
   510	
   511	where Θg(l) indicates the parameters of the lth layer, D is
   512	the Laplacian matrix of Gij , σ denotes ReLU as the activa(l)
   513	tion function, and Yij means the node-level embedding of
   514	th
   515	the l layer. In the training process, the graph convolutional
   516	layers extract features from the pairwise neighborhood structures, the graph aggregation layer aggregates the node-level
   517	features into the graph-level features, and the multi-layer
   518	perception estimates the pseudo labels of sample pairs.
   519	Loss function. Our GCN takes the pairwise neighborhood structure as input, and outputs the likelihood of xi
   520	and xj being of the same identity. Let G denotes the mapping function of our GCN, whose parameters are ΘG =
   521	{Θg(1) , · · · , Θg(L) , Θa , Θm }. To obtain this mapping function, we apply the Binary Cross Entropy (BCE) loss to supervise the training process:
   522	n0
   523	
   524	LG =
   525	
   526	−1 X 1
   527	n0 i=1 |Pig |
   528	
   529	X
   530	
   531	log(gij )+
   532	
   533	Gij ∈Pig
   534	
   535	X
   536	1
   537	(1 − log(gij )),
   538	i
   539	|Ng | G ∈Ni
   540	ij
   541	
   542	(6)
   543	
   544	g
   545	
   546	where gij denotes the prediction of Gij , n0 is the batch size,
   547	Pig is the set of positive samples, in which the sample Gis
   548	in Pig has the positive sample pair (xi , xs ), and Nig is the
   549	
   550	3644
   551	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:04 UTC from IEEE Xplore. Restrictions apply.
   552	
   553	(t)
   554	
   555	where ΘF denotes the parameters of our CNN model at the
   556	tth iteration, and β1 means the learning rate.
   557	
   558	Algorithm 1: Training MPRD.
   559	Input: Initial F, Initial G, Unlabeled data X,
   560	Labeled data Z, Feature memory M,
   561	Training epoch T .
   562	Output: Best CNN model F.
   563	1 Initial P = {Pi = {i}|1 < i < N };
   564	2 for t = 1, t ≤ T, t ++ do
   565	3
   566	for each xi in X do
   567	4
   568	Randomly select zj in Z;
   569	5
   570	Generate pairwise neighborhood structures
   571	and its labels for G;
   572	6
   573	Train G with parameters ΘG by Eq. (9);
   574	7
   575	end
   576	8
   577	for each xi in X do
   578	9
   579	Update M by Eq. (1);
   580	10
   581	Update Pi by Eq. (7);
   582	11
   583	Train F with parameters ΘF by Eq. (8);
   584	12
   585	end
   586	13 end
   587	
   588	4.2. Updating G with F Fixed
   589	Labels for GCN. To ensure the quality of the generated
   590	pseudo pairwise labels by GCN, we propose to generate
   591	an initial, accurate pairwise neighborhood structure with a
   592	tiny amount of labeled meta data Z = {zm }M
   593	m=1 , where M
   594	is the number of labeled identities (we set M = 5, which
   595	is approximately 0.5% of all data. The labeled data is reserved exclusively to jump start GCN training, and these
   596	annotations are never presented to CNN training.).
   597	After the initial jump start phase, the GCN is primarily
   598	trained with the remaining 99.5% unlabeled data. The label
   599	generation process is summarized as follows.
   600	
   601	negative ones. In practice, we obtain the labels of Gij in
   602	two ways to train our GCN model, which will be described
   603	in Section 4.2.
   604	
   605	4. Optimization
   606	Our CNN and GCN are optimized in an alternating manner, with the overall procedure summarized in Algorithm 1.
   607	
   608	4.1. Updating F with G Fixed
   609	Labels for CNN. Since the ground-truth labels for CNN
   610	training are unavailable, we resort to the GCN for pairwise
   611	pseudo labels. For each image pair (xi , xj ), we construct the
   612	pairwise neighborhood structures for GCN in three steps, as
   613	shown in Figure 3 (a): (1) CNN extracts the current feature
   614	F̃(xi ); (2) Memory feature M̃[i] is updated by Eq. (1); and
   615	(3) the pairwise neighborhood structure Gij is obtained by
   616	connecting image xi and the image xj in NNk (xi ) with their
   617	neighbors. Afterwards, Gij is fed into the GCN to predict
   618	pairwise label gij . To refine gij , we subsequently apply a
   619	binary filtering to improve the fidelity of Pi as follows.
   620	
   621	1. At the first iteration, for each xi in the unlabeled training
   622	set X, we generate positive pairs with different data augmentation techniques, including random Gaussian blur
   623	and grayscale conversion, to get its perturbed image x0i .
   624	We assume pairs such as (xi , x0i ) are positive pairs. At
   625	subsequent iterations, the cardinality of Pi gradually increases. If |Pi | > 1, we first randomly draw a sample xri
   626	from Pi . For every xrj ∈ Pi where ri 6= rj , we assign
   627	(xri , xrj ) with a pseudo positive pair label if and only if
   628	xrj ∈ NNk (xri ).
   629	Concurrently at the first iteration, we randomly draw
   630	different images to form negative pairs. At subsequent
   631	iterations, we randomly select |Pi | images from |N∗i | to
   632	generate negative pairs.
   633	2. For each image zi in the labeled meta dataset Z, we randomly draw two images of the same identity to generate
   634	one positive pair, and two images of different identities
   635	to generate one negative pair. Besides, each image xj in
   636	X can be paired with zi to generate one negative pair.
   637	After obtaining the sample pairs with pseudo labels, we
   638	construct a pairwise neighborhood structure Gij , and insert
   639	it into Pig or Nig , accordingly.
   640	Parameter update. With the pairwise neighborhood
   641	structures and pseudo labels ready, we apply the SGD algorithm to update the parameters of GCN as follows.
   642	(t)
   643	
   644	Pi = {xj | xj ∈ NNk (xi ), gij > µ},
   645	
   646	where µ is a predefined likelihood threshold that ensures
   647	xi and xj are of the same identity. Concurrently, the hard
   648	negative sample list N∗i is obtained according to Eq. (3).
   649	Parameter update. We apply the Stochastic Gradient
   650	Descent (SGD) algorithm to update the parameters of our
   651	CNN model, which can be formulated as follows,
   652	(t)
   653	
   654	(t−1)
   655	
   656	ΘF = ΘF
   657	
   658	− β1
   659	
   660	∂LF
   661	,
   662	∂ΘF Θ(t−1)
   663	
   664	(t−1)
   665	
   666	ΘG = ΘG
   667	
   668	(7)
   669	
   670	− β2
   671	
   672	∂LG
   673	,
   674	∂ΘG Θ(t−1)
   675	
   676	(9)
   677	
   678	G
   679	
   680	with β2 being the learning rate. In practice, we select n0 /2
   681	and n0 /2 images from the labeled meta data and unlabeled
   682	data, respectively. Then we construct pairwise neighborhood
   683	structures to calculate ∂LuG /∂ΘG from unlabeled data and
   684	∂Lm
   685	G /∂ΘG from labeled meta data, and compute a linear
   686	combination of them,
   687	∂LuG
   688	∂Lm
   689	∂LG
   690	=
   691	+ η (t) G ,
   692	∂ΘG
   693	∂ΘG
   694	∂ΘG
   695	
   696	(8)
   697	
   698	F
   699	
   700	(10)
   701	
   702	3645
   703	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:04 UTC from IEEE Xplore. Restrictions apply.
   704	
   705	where η (t) is an iteration-dependent weighting factor, which
   706	represents the ratio of the number of pairwise neighborhood
   707	structure generated by metadata and generated by generated
   708	by the unlabeled data in each mini-batch. It is monotonically
   709	decreases with iterations. At early stages of training, a larger
   710	η (t) value helps alleviate the influence of noisy labels in the
   711	unlabeled data. At later iteration stages, the number of pairwise neighborhood structures with pseudo labels extracted
   712	from the unlabeled portion increases, hence η (t) is set to be
   713	smaller to match this trend.
   714	
   715	5. Experiments

codex
前三篇的方法正文已经读完。现在继续读两个 Mix-Modality 版本和 MoDA，重点看它们怎样把“任务设定变化”或“工程约束变化”变成方法贡献。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | sed -n '211,980p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   211	Method
   212	Overview and Preliminary
   213	
   214	In our work, we adopt a similar two-stages training approach as CLIP-ReID [17] as shown in Figure 3.
   215	In the first training stage, to fully exploit the cross-modal description ability in CLIP, we utilize
   216	both text encoder T and image encoder I of CLIP. Similar to CLIP-ReID, we assign ID-specific
   217	tokens to each person. Specifically, the text descriptions fed into T are designed as “A photo of a
   218	[𝑋 ] 1 [𝑋 ] 2 [𝑋 ] 3 ...[𝑋 ] 𝑀 person,” where each [𝑋 ]𝑚 is a learnable text token with the same dimension
   219	as word embedding. By optimization of two contrastive losses inspired by CLIP (image-to-text loss
   220	L𝑖2𝑡 and modified text-to-image loss L𝑡 2𝑖 ), the trained ID-specific tokens can provide discriminative
   221	information of each ID. And in our method, these tokens will be treated as domain prototypes for
   222	the second stage to compute the relevance w.r.t. different source domains. Notably, only tokens
   223	[𝑋 ]𝑚 are optimized while the encoders are frozen. The loss L𝑖2𝑡 is formulated as, specifically,
   224	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   225	
   226	139:6
   227	
   228	Y. Wang et al.
   229	
   230	Fig. 3. The framework of MoDAs. Stage 1. Similar to CLIP-ReID, we assign ID-specific tokens to each person.
   231	The two encoders are frozen and only learnable tokens are optimized in this stage. Person ID-specific tokens
   232	within each source domain are treated as specific prototypes of the corresponding domain, and they will
   233	be used in the Stage 2 for the Voting Network to calculate relevance. Stage 2. For each batch, we randomly
   234	pick one domain to simulate the unseen target domain, and the batch data will be sent to three branches to
   235	compute responding losses, respectively. As shown in the figure, we choose Domain 1 as the mimic target
   236	domain. The expert branch (a) and global branch (b) are used to optimize the expert adapter and the global
   237	adapter, respectively, and the branch (c) is used for the voting network. In the whole procedure, only the
   238	parameters of the expert adapters, the global adapter and the voting network are tunable. While all other
   239	parameters are frozen.
   240	
   241	i denotes the index of the images within a batch with a batch-size B:
   242	𝑒𝑥𝑝 (h𝑉𝑖 ,𝑇𝑖 i)
   243	L𝑖2𝑡 (𝑖) = −𝑙𝑜𝑔 Í𝐵
   244	,
   245	𝑎=1 𝑒𝑥𝑝 (h𝑉𝑖 ,𝑇𝑎 i)
   246	
   247	(1)
   248	
   249	where 𝑉𝑖 and𝑇𝑖 are [CLS] tokens of image feature and text feature produced by I and T , respectively.
   250	And h𝑉𝑖 ,𝑇𝑖 i represents for inner product to compute similarities. For L𝑡 2𝑖 , different images in a
   251	batch probably belong to the same person, so one ID-specific token may have more than one
   252	positive image samples. Therefore, text-to-image loss is modified to:
   253	L𝑡 2𝑖 (𝑦𝑖 ) =
   254	
   255	−1
   256	|𝑃 (𝑦𝑖 )|
   257	
   258	Õ
   259	𝑝 ∈𝑃 (𝑦𝑖 )
   260	
   261	𝑒𝑥𝑝 h𝑉𝑝 ,𝑇𝑦𝑖 i
   262	𝑙𝑜𝑔 Í𝐵
   263	
   264	,
   265	
   266	𝑎=1 𝑒𝑥𝑝 (h𝑉𝑎 ,𝑇𝑦𝑖 i)
   267	
   268	(2)
   269	
   270	where 𝑃 (𝑦𝑖 ) is the set of indices of all positives for 𝑇𝑦𝑖 in the batch. The overall loss of the first
   271	training stage can be formulated as follows:
   272	L𝑠𝑡𝑎𝑔𝑒1 = L𝑖2𝑡 + L𝑡 2𝑖 .
   273	
   274	(3)
   275	
   276	In the second stage, we propose an MoE model with expert adapters and global adapters, which
   277	is to extract domain-specific features and domain-invariant features, respectively. Different from
   278	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   279	
   280	139:7
   281	
   282	MoDA for Parameter-efficient Generalizable Person Re-ID
   283	
   284	CLIP-ReID, we adopt adapter [10] due to its simplicity and parameter-efficiency rather than full
   285	fine-tuning the ViT. And this also concurrently alleviates the optimization challenge posed by
   286	abundant parameters in ViT as well as the scalability limitation of MoE-based approaches on
   287	larger models. In addition, adapters also help prevent catastrophic forgetting from insufficient
   288	full fine-tuning which can destroy the generalization of foundation models like CLIP [37]. As
   289	Figure 2(a) shows, adapter is a simple bottleneck module with two FC layers and one activation
   290	unit GELU. During training period, we can only access the datasets from source domains to train
   291	the DG model. And the optimized DG model will be test on unseen target domain directly. We
   292	𝐾 . And for the kth domain, there are
   293	assume there are K source domains, denoted as 𝐷 = {𝐷𝑘 }𝑘=1
   294	𝑀𝑘 identities and 𝑁𝑘 images. The subsequent sections extensively delineate the second stage.
   295	3.2
   296	
   297	Domain-specific Adapters
   298	
   299	As well as the prior MoE-based methods for DG ReID, our method assigns an individual adapter as
   300	the domain expert of each source domain, denoted as 𝐴𝑘 standing for the domain-specific adapter
   301	of the kth domain. We aim for each expert to capture characteristic features of its corresponding
   302	source domain, which are unique but can provide complementary information different from other
   303	domains.
   304	In every ViT block, we set K adapters after the multi-head attention layer as Figure 3 shows.
   305	Expert Adapter. For a kth domain image from the nth ID, we denote it as 𝑥𝑛𝑘 . As Figure 3(a) shows,
   306	for each 𝑥𝑛𝑘 , we let it go through ViT blocks with only the corresponding domain adapter 𝐴𝑘 to
   307	obtain domain-specific feature 𝐹𝑥𝑘 . The computation of each expert block can be written as:
   308	0
   309	
   310	𝑓𝑙𝑘 = 𝑓𝑙𝑘−1 + 𝐴𝑘 (𝑀𝐴(𝐿𝑁 1 (𝑓𝑙𝑘−1 )))
   311	0
   312	
   313	0
   314	
   315	𝑓𝑙𝑘 = 𝑓𝑙𝑘 + 𝑀𝐿𝑃 (𝐿𝑁 2 (𝑓𝑙𝑘 )),
   316	
   317	(4a)
   318	(4b)
   319	
   320	where 𝑓𝑙𝑘−1 and 𝑓𝑙𝑘 denote the input and output of the lth ViT block. And only the [CLS] token,
   321	rather than all tokens, of the output of the last layer is treated as the feature 𝐹𝑥𝑘 .
   322	Similar to CLIP-ReID, we use the ID-specific text features obtained in the first training stage to
   323	calculate the image-to-text cross-entropy L𝑖2𝑡𝑐𝑒 with label smoothing:
   324	L𝑖2𝑡𝑐𝑒 (𝑖) =
   325	
   326	𝑀𝑘
   327	Õ
   328	
   329	𝑒𝑥𝑝 (h𝑉𝑖 ,𝑇𝑘 i)
   330	−𝑞𝑘 𝑙𝑜𝑔 Í 𝑘
   331	,
   332	𝑀
   333	𝑘=1
   334	𝑎=1 𝑒𝑥𝑝 (h𝑉𝑖 ,𝑇𝑎 i)
   335	
   336	(5)
   337	
   338	where 𝑇𝑘 are just the ID-specific tokens in the corresponding kth domain of the current image.
   339	Notably, we find that with adapters, the usage of ID loss L𝑖𝑑 has little influence on model performance, which will be discussed in Section 4.4 and shown in Table 8. Intuitively, we speculate the
   340	reason might be that the image-to-text loss L𝑖2𝑡 and ID loss L𝑖𝑑 proposed by CLIP are both inherently cross-entropy losses, thus can provide approximate constraints and optimization capabilities.
   341	However, the additional classifier head leads to substantially more tunable parameters under the
   342	DG ReID experiment configuration. Therefore, to additionally enhance the parameter-efficiency of
   343	our model, we discard the ID loss which is widely used in ReID tasks.
   344	Triplet loss is still employed for expert adapter in our methodology:
   345	L𝑡𝑟𝑖𝑝𝑙𝑒𝑡 = 𝑚𝑎𝑥 (𝑑𝑝 − 𝑑𝑛 + 𝛼, 0),
   346	
   347	(6)
   348	
   349	where 𝑑𝑝 and 𝑑𝑛 represent the feature distances of positive pairs and negative pairs, respectively.
   350	Meanwhile, 𝛼 denotes the margin parameter of the triplet loss due to [25].
   351	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   352	
   353	139:8
   354	
   355	Y. Wang et al.
   356	
   357	Thus, the loss for expert adapter is formulated as:
   358	𝑒
   359	𝑒
   360	L𝑒𝑥𝑝𝑒𝑟𝑡 = L𝑖2𝑡𝑐𝑒
   361	+ L𝑡𝑟𝑖𝑝𝑙𝑒𝑡
   362	,
   363	
   364	(7)
   365	
   366	where the superscript e which stands for expert is to distinguish from the losses for global adapter
   367	that will be mentioned below.
   368	Global Adapter. Except for the domain-specific experts, we also maintain an individual global
   369	adapter to learn a common feature space to extract domain-invariant representations. Inspired by
   370	META [36], we similarly utilize a global adapter as the global branch to extract domain-invariant
   371	features. The extracted features are invariant to source domain shifts and more robust compared
   372	to any individual domain adapter. As Figure 3 Stage 2(b) branch shows, for each sample x from
   373	𝑔
   374	any domain, we pass it through the global adapter to obtain global feature 𝐹𝑥 . The computation of
   375	each global adapter block is actually the same as the expert adapter blocks. But we will combine all
   376	text features from different domains together to perform L𝑖2𝑡𝑐𝑒 . And the loss for global adapter is
   377	formulated as:
   378	𝑔
   379	
   380	𝑔
   381	
   382	L𝑔𝑙𝑜𝑏𝑎𝑙 = L𝑖2𝑡𝑐𝑒 + L𝑡𝑟𝑖𝑝𝑙𝑒𝑡 .
   383	
   384	(8)
   385	
   386	Notably, during the training process, the parameters of the ViT backbone are frozen and shared
   387	across different adapters, only the parameters of adapters need to be optimized.
   388	3.3
   389	
   390	MoDAs
   391	
   392	Now, we have both domain-specific and domain-invariant features, but the target domain is
   393	inaccessible during DG ReID training. To improve generalization on unseen domains, we propose a
   394	novel MoE-based approach specially for ViT with adapters to integrate domain-specific features
   395	and get aggregated representations which are more robust.
   396	In most existing MoE-based methods, they learn diverse experts for different source domains
   397	and then calculate the relevance between the test image which is from unseen target domains and
   398	source domains. Subsequently, as guided by the relevance, multiple source domain features are
   399	mixed to create a new aggregated feature. In prior CNN-based approaches, the standard voting
   400	network needs to generate query features from backbone feature maps and calculate relevance
   401	with prototypes of different source domains. The class centers of domain are usually used as its
   402	prototypes. However, different from CNN structure with backbone plus expert heads, in our ViTbased method, to obtain the final features from k experts, we have to forward the whole ViT model
   403	k times. This is due to our domain adapters are positioned within each block (layer) of the model
   404	architecture. This will sharply increase the computation for each sample. To mitigate this issue,
   405	we propose MoDA and a cross-modality block-aware voting network to enable the model to mix
   406	features from multiple domain-specific experts at the block level. The kind of block-wise approach,
   407	compared to the sample-wise approach, also makes the model conduct a more fine-grained feature
   408	mixing in each block rather than using one single weight for all blocks.
   409	As mentioned before, in the first stage, we follow CLIP-ReID [17] to generate ID-specific tokens
   410	for each identity. Additionally, tokens from each source domain will be treated as domain-specific
   411	prototypes for their corresponding domains. These prototypes will be utilized to describe the
   412	characteristics of different source domains.
   413	In the second stage, we adopt episode learning algorithm following [36] to simulate the evaluation
   414	scenario where target domains are unseen. For a sample from the kth domain, we denote it as 𝑥 𝑘 .
   415	We assume 𝑥 𝑘 represents the current input sample to the model during training. We firstly let it
   416	pass to blocks with the kth domain-specific adapters 𝐴𝑘 , then compute the expert loss L𝑒𝑥𝑝𝑒𝑟𝑡 .
   417	And in this iteration, the kth domain will be seemed as the unseen target domain for simulation,
   418	𝐾
   419	𝐾 −1 . Then,
   420	and the remaining 𝐾 − 1 domains {𝐷𝑖 }𝑖=1,𝑖≠𝑘
   421	will be seemed as source domains {𝐷𝑖𝑠 }𝑖=1
   422	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   423	
   424	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | sed -n '203,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   203	Methodology
   204	
   205	In this section, we elaborate on the novel MM-ReID paradigm and introduce two cornerstone
   206	methodologies: the CIDHL and the MBSOS. More specifically, Section 3.1 provides the task definition
   207	and formulation of our proposed MM-ReID together with an analysis of the unique modality
   208	confusion problem that exists in MM-ReID with respect to VI-ReID, Section 3.2 describes the
   209	design motivation for the method and the differences with existing methods, Section 3.3 introduces
   210	the design idea and the detailed composition of the CIDHL loss, and Section 3.4 gives a detailed
   211	description of the steps and algorithmic procedure of the MBSOS strategy.
   212	3.1
   213	
   214	MM Re-ID
   215	
   216	
   217	3.1.1 Task Definition and Formulation. In this study, we use Q = q𝑖 | 𝑖 = 1, 2, . . . , 𝑁 q to denote
   218	the query set, which consists of 𝑁 q probe images, and the gallery set as G = {g𝑖 | 𝑖 = 1, 2, . . . , 𝑁 g},
   219	containing 𝑁 g gallery images. As depicted in Figure 1, under the traditional bi-modality mutual
   220	retrieval paradigm of VI-ReID, the sets Q and G each consist of images exclusively in one modality,
   221	either visible or infrared. Contrary to this, our proposed MM-ReID paradigm includes images in
   222	both visible and infrared modalities within both Q and G, addressing a more realistic and complex
   223	retrieval scenario.
   224	3.1.2 Challenge Analyze. As illustrated in Figure 2, the challenges inherent in SM-ReID primarily stem from the need to accurately match pedestrian images across varied camera feeds, a task
   225	complicated by background noise, changes in viewpoint, and variations in pose. The complexity
   226	escalates in VI-ReID, where the task involves matching images across the visible and infrared
   227	spectra, a process hampered by substantial differences in their imaging mechanisms. These modality differences represent significant obstacles to effective cross-modality matching. Furthermore,
   228	MM-ReID introduces additional complexities. Beyond the challenges faced in VI-ReID, MM-ReID
   229	contends with “modality confusion,” where similarities irrelevant to identity, such as color consistency within a modality, can obscure vital identity-specific cues between the query (Q) and gallery
   230	(G) sets, thereby complicating the differentiation of individuals.
   231	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   232	
   233	112:6
   234	
   235	W. Liu et al.
   236	
   237	Fig. 3. An illustration of our proposed CIDHL and MBSOS. The mix-modality data is fed into two feature
   238	extractors with shared weights to extract the features, and under the constraints of CIDHL, the distance
   239	between the centers of the cross-modality same identity samples is pulled closer together pushing away the
   240	distance between the centers of the different identity samples of the same or different modalities, while at
   241	the same time pulling together the distance between the same identity sample point and the sample center.
   242	During the testing process, the extracted features are optimized by MBSOS to get the final shortest path
   243	with the help of bridge samples in the gallery set for obtaining the optimized distance metric 𝑑˜𝑖,𝑗 .
   244	
   245	3.2
   246	
   247	Method Design Motivation
   248	
   249	To tackle the distinct modality confusion problem in the MM-ReID task, this article introduces
   250	two innovative methods: CIDHL and MBSOS. CIDHL specifically targets the confusion arising
   251	from similarity among same-modality samples by constraining these similarities in the metric
   252	space. This adjustment reduces the risk of confusing different-identity samples within the same
   253	modality when attempting to match cross-modality, same-identity positive samples. Moreover,
   254	to mitigate modality confusion effects during the testing phase, MBSOS optimizes the similarity
   255	measure between cross-modality samples by leveraging same-modality samples in the gallery
   256	as a bridge. This is a departure from traditional VI-ReID methods, which typically focus only
   257	on constraining or exploiting the similarity across modalities. Our approach aims to refine both
   258	negative sample constraints and positive sample exploitation, specifically addressing the challenges
   259	posed by same-modality samples unique to the MM-ReID task.
   260	3.3
   261	
   262	CIDHL
   263	
   264	3.3.1 Feature Presentation Extractor. Our proposed methods, CIDHL and MBSOS, are designed to
   265	seamlessly integrate with the feature extraction frameworks of existing VI-ReID methods. This integration ensures that there is no need for modifications to the baseline feature extraction processes
   266	of these methods. As illustrated in Figure 3, we employ a generalized baseline feature representation
   267	extractor: a weight-shared, dual-branch convolutional neural network. This structure is adept at
   268	learning and extracting features from both visible and infrared images. The network’s effectiveness
   269	is further enhanced by the application of cross-entropy loss (𝐿id ) and our specifically developed
   270	CIDHL (𝐿CIDHL ), optimizing the feature extraction process and ensuring robust performance across
   271	varied modality inputs.
   272	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   273	
   274	Mix-Modality Person Re-Identification: A New and Practical Paradigm
   275	
   276	112:7
   277	
   278	3.3.2 Triplet Loss Review. Introduced initially by FaceNet [25], triplet loss has become a cornerstone in identifying and enhancing relationships within small batches of samples in various
   279	machine learning applications. A significant advancement in this area is the development of the
   280	triplet hard loss [6]. This approach refines the basic triplet loss by dynamically selecting the most
   281	challenging samples within a mini-batch to maximize the informative value derived during training.
   282	For a given mini-batch X, consisting of 𝑃 × 𝐾 images, where each of the P identities is represented
   283	by K images, triplet hard loss operates by selecting, for each anchor sample 𝑥 a , the nearest negative
   284	sample 𝑥 n and the farthest positive sample 𝑥 p . This selection strategy is designed to intensify the
   285	learning signals from the most informative triplets, involving the hardest positive and negative
   286	samples. The formulation of triplet hard loss, 𝐿𝑡ℎ , effectively leverages these critical relationships
   287	within the batch to optimize the discriminative capability of the model, and the specific formula
   288	𝐿𝑡ℎ can be expressed as follows:
   289	hardest positive
   290	
   291	𝐿𝑡ℎ (𝑋 ) =
   292	
   293	𝑃
   294	Õ 𝐾
   295	Õ
   296	
   297	z
   298	
   299	}|
   300	
   301	[𝑚 + max
   302	
   303	{
   304	
   305	𝑥𝑎𝑖 − 𝑥 𝑝𝑖
   306	
   307	𝑝=1...𝐾
   308	𝑖=1 𝑎=1
   309	
   310	2
   311	
   312	hardest negative
   313	
   314	z
   315	− min
   316	𝑗=1...𝑃
   317	𝑛=1...𝐾
   318	𝑗≠𝑖
   319	
   320	}|
   321	𝑥𝑎𝑖 − 𝑥𝑛𝑗
   322	
   323	{
   324	2
   325	
   326	] +,
   327	
   328	(1)
   329	
   330	where 𝑥 a𝑖 denotes the ath image of ith identity pedestrian within a batch, [𝑥] + stands for standard
   331	hinge loss, and 𝑥𝑎𝑖 − 𝑥 𝑝𝑖 denotes the Euclidean distance between 𝑥 a and 𝑥 p , m is the margin
   332	2
   333	parameter for controling distance.
   334	3.3.3 Cross-Identity Discrimination Harmonization. While the triplet hard loss provides a robust
   335	framework for learning from the most challenging samples, its effectiveness can be compromised
   336	by the presence of noisy data within the dataset. To counteract this issue, we have implemented a
   337	modification where the center of each identity is used as a substitute for individual samples. This
   338	approach helps stabilize the training process by reducing the influence of outliers or mislabeled
   339	instances. For each identity, the center is calculated by averaging the features of all samples
   340	associated with that identity. This average is computed separately for each modality, ensuring that
   341	the specific characteristics of visible and infrared images are accurately represented. The identity
   342	centers are then utilized in place of individual samples when calculating triplet hard loss, thereby
   343	providing a more reliable basis for optimizing the model. This method not only mitigates the risk
   344	of steering the learning process in the wrong direction due to noisy data but also enhances the
   345	overall robustness of the model, and the centers for identities of different modalities are computed
   346	as follows:
   347	𝐾
   348	
   349	𝑐 𝑖𝑣 =
   350	
   351	1 Õ 𝑖
   352	𝑣 ,
   353	𝐾 𝑗=1 𝑗
   354	
   355	𝑐𝑡𝑖 =
   356	
   357	1 Õ 𝑖
   358	𝑡,
   359	𝐾 𝑗=1 𝑗
   360	
   361	𝐾
   362	
   363	(2)
   364	
   365	where 𝑣 i𝑗 and 𝑡 i𝑗 denote the jth image of ith identity within visible and infrared modality, while the
   366	𝑐 𝑖v and 𝑐 𝑖t represent the center of ith identity within two modalities.
   367	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   368	
   369	112:8
   370	
   371	W. Liu et al.
   372	
   373	Metric Learning in CIDHL: Enhancing Identity Discrimination across Modalities. The fundamental
   374	principle of metric learning in CIDHL revolves around optimizing identity discrimination by
   375	manipulating distances in the feature space. The goal is to:
   376	— Minimize Intra-Identity Distances: The model aims to decrease the distance between feature
   377	centers of the same identity, regardless of the modality. This step ensures that the same identity
   378	is represented more cohesively across different imaging conditions.
   379	— Maximize Inter-Identity Distances: Simultaneously, it is crucial to increase the distance between feature centers of different identities within the same modality and across different
   380	modalities. This separation helps distinguish individuals clearly.
   381	— Aggregate Intra-Modality Distances: To further refine the model’s effectiveness, distances
   382	between samples of the same modality and identity are aggregated towards their respective
   383	centers. This aggregation aids in minimizing the modality-specific variations within the same
   384	identity, thus reducing the risk of boundary confusion where similar samples might be close
   385	to the decision boundaries.
   386	The CIDHL adopts a structured approach, containing two parts, the first of which is targeted:
   387	— Cross-Modality Attraction: Distances between centers of the same identity across modalities
   388	are brought closer to foster a uniform identity representation across different sensory inputs.
   389	— Intra- and Inter-Modality Repulsion: Distances between centers of different identities, whether
   390	within the same modality or across modalities, are pushed apart to enhance discriminability.
   391	The specific loss function of part one, denoted as 𝐿CID , is formulated to dynamically adjust
   392	distances, ensuring an optimal balance between attraction and repulsion forces within the hyperspherical identity space, the 𝐿𝑐𝑖𝑑 is calculated as:
   393	𝐿𝑐𝑖𝑑 =
   394	
   395	𝑃 
   396	Õ
   397	𝑚 + 𝑑 (𝐶 𝑖𝑣 − 𝐶𝑡𝑖 ) −
   398	
   399	min
   400	
   401	𝑛∈ {𝑣,𝑡 },𝑗≠𝑖
   402	
   403	𝑑 (𝐶 𝑖𝑣 − 𝐶𝑛𝑗 )
   404	
   405	𝑖=1
   406	𝑃
   407	Õ
   408	+
   409	𝑚 + 𝑑 (𝐶𝑡𝑖 − 𝐶 𝑖𝑣 ) −
   410	
   411	min
   412	
   413	𝑛∈ {𝑣,𝑡 },𝑗≠𝑖
   414	
   415	𝑖=1
   416	
   417	
   418	+
   419	
   420	𝑑 (𝐶𝑡𝑖 − 𝐶𝑛𝑗 )
   421	
   422	
   423	,
   424	
   425	(3)
   426	
   427	+
   428	
   429	
   430	
   431	where C consisting of all the visible center 𝑐 𝑖𝑣 | 𝑖 = 1, · · · , 𝑃 and 𝑐𝑡𝑖 | 𝑖 = 1, · · · , 𝑃 in the minibatch X. Both two terms in the loss aim to learn cross-identity discrimination. Based on this, we
   432	further propose discrimination harmonization loss 𝐿𝑑ℎ for controlling the distance from the sample
   433	center of samples with the same modality and identity for solving the modality confusion problem
   434	due to the similarity of boundary samples. The 𝐿𝑑ℎ is calculated as follows:
   435	
   436	
   437	𝑖
   438	𝑖
   439	𝑖
   440	𝑖
   441	𝐿𝑑ℎ = 𝑚 + max 𝑑 (𝐶 𝑣 − 𝑣 𝑗 ) − min 𝑑 (𝐶 𝑣 − 𝑡 𝑗 ) +
   442	1≤ 𝑗 ≤𝐾
   443	1≤ 𝑗 ≤𝐾
   444	
   445	+
   446	𝑚 + max 𝑑 (𝐶𝑡𝑖 − 𝑡 𝑖𝑗 ) − min 𝑑 (𝐶𝑡𝑖 − 𝑣 𝑖𝑗 ) .
   447	1≤ 𝑗 ≤𝐾
   448	
   449	1≤ 𝑗 ≤𝐾
   450	
   451	(4)
   452	
   453	The final CIDHL 𝐿𝐶𝐼 𝐷𝐻 𝐿 can be calculated as:
   454	𝐿𝐶𝐼 𝐷𝐻 𝐿 = 𝐿𝑐𝑖𝑑 + 𝛿𝐿𝑑ℎ ,
   455	where 𝛿 is the tradeoff parameter between 𝐿𝑐𝑖𝑑 and 𝐿𝑑ℎ .
   456	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   457	
   458	(5)
   459	
   460	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf' - | nl -ba | sed -n '223,780p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   223	METHODOLOGY
   224	
   225	In this section, we elaborate on the novel Mix-Modality person re-identification (MM-ReID) paradigm and introduce two cornerstone methodologies: the Cross-Identity Discrimination Harmonization
   226	Loss (CIDHL) and the Modality Bridge Similarity Optimization Strategy (MBSOS). More specifically, Section 3.1 provides the task definition and formulation of our proposed MM-ReID together
   227	with an analysis of the unique modality confusion problem that exists in MM-ReID with respect
   228	to VI-ReID, Section 3.2 describes the design motivation for the method and the differences with
   229	existing methods, Section 3.3 introduces the design idea and the detailed composition of the CIDHL
   230	loss, and Section 3.4 gives a detailed description of the steps and algorithmic procedure of the
   231	MBSOS strategy.
   232	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   233	
   234	Mix-Modality Person Re-Identification: A New and Practical Paradigm
   235	
   236	Probe
   237	
   238	Probe
   239	
   240	Gallery
   241	
   242	Probe
   243	
   244	Gallery
   245	
   246	Distance
   247	
   248	Distance
   249	
   250	(a)
   251	
   252	(b)
   253	
   254	5
   255	
   256	Probe
   257	
   258	Gallery
   259	
   260	Distance
   261	
   262	Gallery
   263	
   264	Distance
   265	
   266	(c)
   267	
   268	Fig. 2. An illustration of challenges in SM-ReID, VI-ReID, MM-ReID. Different geometries represent different
   269	identities, blue/gray represents visible/infrared modality samples, while green and red lines represent correct
   270	and incorrect matches. It can be seen that: (a) SM-ReID mainly faces the challenge of differences between
   271	different identities (shapes); (b) VI-ReID needs to face the challenge of modality (color) differences in addition
   272	to identity differences; and (c) MM-ReID needs to face the challenge of modality confusions (similar modalities
   273	possessing a closer proximity) in addition to identity and modality differences.
   274	
   275	3.1
   276	
   277	Mix-Modality Person Re-identification
   278	
   279	
   280	3.1.1 Task Definition and Formulation. In this study, we use Q = q𝑖 | 𝑖 = 1, 2, . . . , 𝑁 q to denote the query set, which consists of 𝑁 q probe images, and the gallery set as G = {g𝑖 | 𝑖 = 1, 2, . . . , 𝑁 g},
   281	containing 𝑁 g gallery images. As depicted in Figure 1, under the traditional bi-modality mutual
   282	retrieval paradigm of Visible-Infrared Re-Identification (VI-ReID), the sets Q and G each consist
   283	of images exclusively in one modality, either visible or infrared. Contrary to this, our proposed
   284	Mix-Modality Re-Identification (MM-ReID) paradigm includes images in both visible and infrared
   285	modalities within both Q and G, addressing a more realistic and complex retrieval scenario.
   286	3.1.2 Challenge Analyze. As illustrated in Fig 2, the challenges inherent in Single-Modality ReID
   287	(SM-ReID) primarily stem from the need to accurately match pedestrian images across varied camera
   288	feeds, a task complicated by background noise, changes in viewpoint, and variations in pose. The
   289	complexity escalates in Visible-Infrared ReID (VI-ReID), where the task involves matching images
   290	across the visible and infrared spectra, a process hampered by substantial differences in their imaging
   291	mechanisms. These modality differences represent significant obstacles to effective cross-modality
   292	matching. Furthermore, Mix-Modality ReID (MM-ReID) introduces additional complexities. Beyond
   293	the challenges faced in VI-ReID, MM-ReID contends with ’modality confusion,’ where similarities
   294	irrelevant to identity, such as color consistency within a modality, can obscure vital identityspecific cues between the query (Q) and gallery (G) sets, thereby complicating the differentiation
   295	of individuals.
   296	3.2
   297	
   298	Method Design Motivation
   299	
   300	To tackle the distinct modality confusion problem in the MM-ReID task, this paper introduces two
   301	innovative methods: Cross-Identity Discrimination Harmonization Loss (CIDHL) and Modality
   302	Bridge Similarity Optimization Strategy (MBSOS). CIDHL specifically targets the confusion arising
   303	from similarity among same-modality samples by constraining these similarities in the metric
   304	space. This adjustment reduces the risk of confusing different-identity samples within the same
   305	modality when attempting to match cross-modality, same-identity positive samples. Moreover,
   306	to mitigate modality confusion effects during the testing phase, MBSOS optimizes the similarity
   307	measure between cross-modality samples by leveraging same-modality samples in the gallery
   308	as a bridge. This is a departure from traditional VI-ReID methods, which typically focus only
   309	on constraining or exploiting the similarity across modalities. Our approach aims to refine both
   310	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   311	
   312	6
   313	
   314	WEI LIU, XIN XU, HUA CHANG, XIN YUAN, and ZHENG WANG
   315	
   316	Input
   317	
   318	Feature Extractor
   319	
   320	Cross-Identity Discrimination Harmonization
   321	
   322	Modality Bridge Similarity Optimize
   323	Probe 𝑞
   324	𝑑(𝑞 , 𝑔 )
   325	
   326	…
   327	
   328	𝑔
   329	𝑑(𝑔 , 𝑔 )
   330	𝑔
   331	𝑑(𝑞 , 𝑔 )
   332	
   333	Shared
   334	Weights
   335	
   336	Gallery
   337	
   338	…
   339	
   340	𝑑 , = 𝑑 𝑞 , 𝑔 + 𝑑 𝑔 , 𝑔 < 𝑑(𝑞 , 𝑔 )
   341	
   342	Class Center
   343	
   344	Pull Close
   345	
   346	Push Far
   347	
   348	𝑑(𝑞 , 𝑔 ) distance between 𝑞 and 𝑔
   349	𝑑 , shortest path between 𝑞 and 𝑔
   350	
   351	Fig. 3. An illustration of our proposed CIDHL and MBSOS. The mix-modality data is fed into two feature
   352	extractors with shared weights to extract the features, and under the constraints of CIDHL, the distance
   353	between the centers of the cross-modality same identity samples is pulled closer together pushing away the
   354	distance between the centers of the different identity samples of the same or different modalities, while at
   355	the same time pulling together the distance between the same identity sample point and the sample center.
   356	During the testing process, the extracted features are optimized by MBSOS to get the final shortest path
   357	with the help of bridge samples in the gallery set for obtaining the optimized distance metric 𝑑˜𝑖,𝑗 .
   358	
   359	negative sample constraints and positive sample exploitation, specifically addressing the challenges
   360	posed by same-modality samples unique to the MM-ReID task.
   361	3.3
   362	
   363	Cross-Identity Discrimination Harmonization Loss
   364	
   365	3.3.1 Feature Presentation Extractor. Our proposed methods, Cross-Identity Discrimination
   366	Harmonization Loss (CIDHL) and Modality Bridge Similarity Optimization Strategy (MBSOS),
   367	are designed to seamlessly integrate with the feature extraction frameworks of existing VI-ReID
   368	methods. This integration ensures that there is no need for modifications to the baseline feature
   369	extraction processes of these methods. As illustrated in Figure 3, we employ a generalized baseline
   370	feature representation extractor: a weight-shared, dual-branch convolutional neural network. This
   371	structure is adept at learning and extracting features from both visible and infrared images. The
   372	network’s effectiveness is further enhanced by the application of cross-entropy loss (𝐿id ) and our
   373	specifically developed CIDHL (𝐿CIDHL ), optimizing the feature extraction process and ensuring
   374	robust performance across varied modality inputs.
   375	3.3.2 Triplet Loss Review. Introduced initially by FaceNet [25], triplet loss has become a cornerstone in identifying and enhancing relationships within small batches of samples in various
   376	machine learning applications. A significant advancement in this area is the development of the
   377	triplet hard loss [6]. This approach refines the basic triplet loss by dynamically selecting the most
   378	challenging samples within a mini-batch to maximize the informative value derived during training.
   379	For a given mini-batch 𝑋 , consisting of 𝑃 × 𝐾 images, where each of the 𝑃 identities is represented
   380	by 𝐾 images, triplet hard loss operates by selecting, for each anchor sample 𝑥 a , the nearest negative
   381	sample 𝑥 n and the farthest positive sample 𝑥 p . This selection strategy is designed to intensify the
   382	learning signals from the most informative triplets, involving the hardest positive and negative
   383	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   384	
   385	Mix-Modality Person Re-Identification: A New and Practical Paradigm
   386	
   387	7
   388	
   389	samples. The formulation of triplet hard loss, 𝐿th , effectively leverages these critical relationships
   390	within the batch to optimize the discriminative capability of the model, and the specific formula
   391	𝐿th can be expressed as follows:
   392	hardest positive
   393	
   394	𝐿𝑡ℎ (𝑋 ) =
   395	
   396	𝑃
   397	∑︁ 𝐾
   398	∑︁
   399	
   400	z
   401	
   402	}|
   403	
   404	[𝑚 + max
   405	
   406	{
   407	
   408	𝑥𝑎𝑖 − 𝑥 𝑝𝑖
   409	
   410	𝑝=1...𝐾
   411	𝑖=1 𝑎=1
   412	
   413	2
   414	
   415	(1)
   416	
   417	hardest negative
   418	
   419	z
   420	
   421	}|
   422	
   423	{
   424	
   425	− min 𝑥𝑎𝑖 − 𝑥𝑛𝑗 ] +,
   426	𝑗=1...𝑃
   427	𝑛=1...𝐾
   428	𝑗≠𝑖
   429	
   430	2
   431	
   432	where 𝑥 a𝑖 denotes the 𝑎 − 𝑡ℎ image of 𝑖 − 𝑡ℎ identity pedestrian within a batch, [𝑥] + stands for
   433	standard hinge loss, and 𝑥𝑎𝑖 − 𝑥 𝑝𝑖 denotes the Euclidean distance between 𝑥 a and 𝑥 p , 𝑚 is the
   434	2
   435	margin parameter for controling distance.
   436	3.3.3 Cross-Identity Discrimination Harmonization . While the triplet hard loss provides
   437	a robust framework for learning from the most challenging samples, its effectiveness can be
   438	compromised by the presence of noisy data within the dataset. To counteract this issue, we have
   439	implemented a modification where the center of each identity is used as a substitute for individual
   440	samples. This approach helps stabilize the training process by reducing the influence of outliers
   441	or mislabeled instances. For each identity, the center is calculated by averaging the features of
   442	all samples associated with that identity. This average is computed separately for each modality,
   443	ensuring that the specific characteristics of visible and infrared images are accurately represented.
   444	The identity centers are then utilized in place of individual samples when calculating triplet hard
   445	loss, thereby providing a more reliable basis for optimizing the model. This method not only
   446	mitigates the risk of steering the learning process in the wrong direction due to noisy data but also
   447	enhances the overall robustness of the model, and the centers for identities of different modalities
   448	are computed as follows:
   449	𝐾
   450	
   451	𝑐 𝑖𝑣 =
   452	
   453	1 ∑︁ 𝑖
   454	𝑣 ,
   455	𝐾 𝑗=1 𝑗
   456	𝐾
   457	
   458	(2)
   459	
   460	1 ∑︁ 𝑖
   461	𝑐𝑡𝑖 =
   462	𝑡,
   463	𝐾 𝑗=1 𝑗
   464	where 𝑣 i𝑗 and 𝑡 i𝑗 denote the 𝑗 − 𝑡ℎ image of 𝑖 − 𝑡ℎ identity within visible and infrared modality,
   465	while the 𝑐 𝑖v and 𝑐 𝑖t represent the center of 𝑖 − 𝑡ℎ identity within two modalities.
   466	Metric Learning in CIDHL: Enhancing Identity Discrimination Across Modalities The fundamental principle of metric learning in CIDHL revolves around optimizing identity discrimination by
   467	manipulating distances in the feature space. The goal is to:
   468	• Minimize intra-identity distances: The model aims to decrease the distance between feature
   469	centers of the same identity, regardless of the modality. This step ensures that the same
   470	identity is represented more cohesively across different imaging conditions.
   471	• Maximize inter-identity distances: Simultaneously, it is crucial to increase the distance
   472	between feature centers of different identities within the same modality and across different
   473	modalities. This separation helps distinguish individuals clearly.
   474	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   475	
   476	8
   477	
   478	WEI LIU, XIN XU, HUA CHANG, XIN YUAN, and ZHENG WANG
   479	
   480	• Aggregate intra-modality distances: To further refine the model’s effectiveness, distances
   481	between samples of the same modality and identity are aggregated towards their respective
   482	centers. This aggregation aids in minimizing the modality-specific variations within the same
   483	identity, thus reducing the risk of boundary confusion where similar samples might be close
   484	to the decision boundaries.
   485	The CIDHL adopts a structured approach, containing two parts, the first of which is targeted:
   486	• Cross-modality attraction: Distances between centers of the same identity across modalities
   487	are brought closer to foster a uniform identity representation across different sensory inputs.
   488	• Intra- and inter-modality repulsion: Distances between centers of different identities, whether
   489	within the same modality or across modalities, are pushed apart to enhance discriminability.
   490	The specific loss function of part one, denoted as 𝐿CID , is formulated to dynamically adjust
   491	distances, ensuring an optimal balance between attraction and repulsion forces within the hyperspherical identity space, the 𝐿𝑐𝑖𝑑 is calculated as:
   492	
   493	𝑃 
   494	∑︁
   495	𝐿𝑐𝑖𝑑 =
   496	𝑚 + 𝑑 (𝐶 𝑖𝑣 − 𝐶𝑡𝑖 ) − min 𝑑 (𝐶 𝑖𝑣 − 𝐶𝑛𝑗 )
   497	𝑛∈ {𝑣,𝑡 },𝑗≠𝑖
   498	
   499	𝑖=1
   500	𝑃
   501	∑︁ 
   502	+
   503	𝑚 + 𝑑 (𝐶𝑡𝑖 − 𝐶 𝑖𝑣 ) −
   504	
   505	min
   506	
   507	𝑛∈ {𝑣,𝑡 },𝑗≠𝑖
   508	
   509	𝑖=1
   510	
   511	+
   512	
   513	𝑑 (𝐶𝑡𝑖 − 𝐶𝑛𝑗 )
   514	
   515	(3)
   516	
   517	
   518	,
   519	+
   520	
   521	
   522	
   523	where 𝐶 consisting of all the visible center 𝑐 𝑖𝑣 | 𝑖 = 1, · · · , 𝑃 and 𝑐𝑡𝑖 | 𝑖 = 1, · · · , 𝑃 in the minibatch 𝑋 . Both two terms in the loss aim to learn cross-identity discrimination. Based on this, we
   524	further propose discrimination harmonization loss 𝐿𝑑ℎ for controlling the distance from the sample
   525	center of samples with the same modality and identity for solving the modality confusion problem
   526	due to the similarity of boundary samples. The 𝐿𝑑ℎ is calculated as follows:
   527	
   528	
   529	𝑖
   530	𝑖
   531	𝑖
   532	𝑖
   533	𝐿𝑑ℎ = 𝑚 + max 𝑑 (𝐶 𝑣 − 𝑣 𝑗 ) − min 𝑑 (𝐶 𝑣 − 𝑡 𝑗 ) +
   534	1≤ 𝑗 ≤𝐾
   535	1≤ 𝑗 ≤𝐾
   536	(4)
   537	
   538	+
   539	𝑖
   540	𝑖
   541	𝑖
   542	𝑖
   543	𝑚 + max 𝑑 (𝐶𝑡 − 𝑡 𝑗 ) − min 𝑑 (𝐶𝑡 − 𝑣 𝑗 ) .
   544	1≤ 𝑗 ≤𝐾
   545	
   546	1≤ 𝑗 ≤𝐾
   547	
   548	The final CIDHL 𝐿𝐶𝐼 𝐷𝐻 𝐿 can be calculated as:
   549	𝐿𝐶𝐼 𝐷𝐻 𝐿 = 𝐿𝑐𝑖𝑑 + 𝛿𝐿𝑑ℎ ,
   550	
   551	(5)
   552	
   553	where 𝛿 is the tradeoff parameter between 𝐿𝑐𝑖𝑑 and 𝐿𝑑ℎ .
   554	3.4
   555	
   556	Modality Bridge Similarity Optimization Strategy
   557	
   558	The Modality Bridge Similarity Optimization Strategy (MBSOS) addresses the modality confusion
   559	issue by innovatively transforming the distance calculation process between a query sample and a
   560	gallery sample into a pathfinding problem. Here’s how it works:
   561	• Bridge Node Concept: Each query sample, 𝑞𝑖 ∈ Q, seeks the optimal path to a gallery sample,
   562	𝑔 𝑗 ∈ G, not by direct distance calculation but through intermediate "bridge nodes" within
   563	the gallery.
   564	• Pathfinding Optimization: The strategy uses these bridge nodes, which are samples in the
   565	gallery of similar modality to the query, to create a modality-consistent path. This method
   566	effectively reduces the impact of direct modality differences on the distance metric.
   567	• Distance Optimization: By navigating through these intermediate nodes, the system optimizes the apparent distance between 𝑞𝑖 and 𝑔 𝑗 . This optimization aims to provide a more
   568	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   569	
   570	
codex
刚才并行输出里 Mix 和 MoDA 后半段被截在页边附近。我继续从截断处往后读，补齐 MBSOS 和 MoDA 的块级投票、训练目标部分。
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
/bin/zsh -lc "pdftotext -f 1 -l 12 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | sed -n '453,900p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 12 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | sed -n '388,880p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   453	The final CIDHL 𝐿𝐶𝐼 𝐷𝐻 𝐿 can be calculated as:
   454	𝐿𝐶𝐼 𝐷𝐻 𝐿 = 𝐿𝑐𝑖𝑑 + 𝛿𝐿𝑑ℎ ,
   455	where 𝛿 is the tradeoff parameter between 𝐿𝑐𝑖𝑑 and 𝐿𝑑ℎ .
   456	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   457	
   458	(5)
   459	
   460	Mix-Modality Person Re-Identification: A New and Practical Paradigm
   461	3.4
   462	
   463	112:9
   464	
   465	MBSOS
   466	
   467	The MBSOS addresses the modality confusion issue by innovatively transforming the distance
   468	calculation process between a query sample and a gallery sample into a pathfinding problem. Here’s
   469	how it works:
   470	— Bridge Node Concept: Each query sample, 𝑞𝑖 ∈ Q, seeks the optimal path to a gallery sample,
   471	𝑔 𝑗 ∈ G, not by direct distance calculation but through intermediate “bridge nodes” within the
   472	gallery.
   473	— Pathfinding Optimization: The strategy uses these bridge nodes, which are samples in the
   474	gallery of similar modality to the query, to create a modality-consistent path. This method
   475	effectively reduces the impact of direct modality differences on the distance metric.
   476	— Distance Optimization: By navigating through these intermediate nodes, the system optimizes the apparent distance between 𝑞𝑖 and 𝑔 𝑗 . This optimization aims to provide a more
   477	accurate similarity measure by leveraging intra-modality consistency to guide cross-modality
   478	comparisons.
   479	The core objective of MBSOS is to enhance the robustness of cross-modality Re-ID by minimizing
   480	the distortive effects of modality differences on the identity-matching process. This approach not
   481	only streamlines the matching process but also significantly improves the precision of identity
   482	verification across different imaging conditions.
   483	3.4.1 Distance Map Construction. To obtain the relationship between sample points, we first
   484	calculate the distance 𝑑 (𝑞𝑖 , 𝑔𝑖 ) between query sample 𝑞𝑖 and gallery sample 𝑔𝑖 , as well as the distance
   485	𝑑 (𝑔𝑖 , 𝑔 𝑗 ) between gallery samples 𝑔𝑖 and 𝑔 𝑗 as follows:
   486	𝑑 (𝑞𝑖 , 𝑔 𝑗 ) = 𝑞𝑖 − 𝑔 𝑗 2
   487	𝑑 (𝑔 𝑗 , 𝑔𝑡 ) = 𝑞 𝑗 − 𝑔𝑡 2 ,
   488	
   489	(6)
   490	
   491	where(𝑖 = 1, 2, . . . , 𝑁𝑞 ), ( 𝑗 = 1, 2, . . . , 𝑁𝑔 ), and (𝑡 = 1, 2, . . . , 𝑁𝑔 ). Then, we construct a distance
   492	map M𝑄𝐺 (V, E) by 𝑑 (𝑞𝑖 , 𝑔𝑖 ) to represent the distances between the query set and gallery set,
   493	and another distance map M𝐺𝐺 (V, E) using 𝑑 (𝑔 𝑗 , 𝑔𝑡 ) to represent the distances between gallery
   494	samples in the gallery set. In M𝑄𝐺 and M𝐺𝐺 , each vertex V represents a pedestrian image, and
   495	each edge E represents the distance between two vertices. Considering that modality confusion is
   496	mainly caused by the fact that the same modality naturally has a closer distance, we propose an
   497	appropriate scaling adjustment for the distance of the edge E of the same modality to attenuate the
   498	natural effects of this imaging mechanism, the edges between query sample and gallery sample of
   499	the M𝑄𝐺 adjusted as follows:
   500	(
   501	𝜆d(qi, gj ) if ml(qi ) = ml(gj )
   502	𝑖𝑗
   503	Eqg =
   504	(7)
   505	d(qi, gj )
   506	otherwise,
   507	𝑖𝑗
   508	where Eqg
   509	is the adjusted distance between ith and jth vertex in M𝑄𝐺 . The ml(qi ) and ml(gj ) are
   510	the modality label of the 𝑞𝑖 and 𝑔 𝑗 . The edges in M𝐺𝐺 can be adjusted similarly as follows:
   511	(
   512	𝜆d(gj, gt ) if ml(gi ) = ml(gt )
   513	𝑗𝑡
   514	Egg =
   515	(8)
   516	d(gj, gt )
   517	otherwise.
   518	
   519	3.4.2 Modality Bridge Optimization. After constructing a distance graph M Q G , computing the
   520	nearest distance 𝑑˜𝑖 𝑗 from a given probe qi to a point gj in the gallery can be viewed as finding the
   521	shortest path from a probe node qi to a gallery node gj in the M Q G with the help of a bridge node
   522	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   523	
   524	112:10
   525	
   526	W. Liu et al.
   527	
   528	Algorithm 1: MBSOS
   529	
   530	𝑔𝑡 in M𝐺𝐺 . More specifically, a path ptij from a probe node qi to a gallery node gj in the M Q G can
   531	be represented as follows:
   532	𝑡𝑗
   533	𝑖𝑡
   534	𝑝𝑖𝑡 𝑗 = (𝑞𝑖 , 𝑔𝑡 , 𝑔 𝑗 ) = E𝑞𝑔
   535	+ E𝑔𝑔
   536	.
   537	
   538	(9)
   539	
   540	Then, the set of all paths between qi and gj is P = (𝑝𝑖1𝑗 , . . . , 𝑝𝑖𝑛𝑗 ), where 𝑛 = 𝑁𝐺 . Therefore, the
   541	shortest distance 𝑑˜𝑖 𝑗 between 𝑞𝑖 and 𝑔 𝑗 can be calculated as follows:
   542	n
   543	o
   544	𝑡𝑗
   545	𝑖𝑡
   546	𝑑˜𝑖 𝑗 = min P = min E𝑞𝑔
   547	+ E𝑔𝑔
   548	.
   549	1≤𝑡 ≤𝑁𝑔
   550	
   551	1≤𝑡 ≤𝑁𝑔
   552	
   553	(10)
   554	
   555	Finally, we obtain the optimized distance map M̃𝑄𝐺 constructed by all the shortest distance 𝑑˜𝑖 𝑗
   556	between the query nodes 𝑞𝑖 and 𝑔 𝑗 , which is optimized with the help of a bridge node 𝑔𝑡 . It is worth
   557	mentioning that in some cases, the direct distance from 𝑞𝑖 to 𝑔 𝑗 is the shortest distance without
   558	the help of any intermediate bridge node. The algorithmic procedure of MBSOS is described in
   559	Algorithm 1.
   560	4
   561	
   562	Experimental Design and Results
   563	
   564	In this section, we first present the adopted dataset, evaluation metrics, mix-modality paradigm,
   565	and method implementation details in Section 4.1. In Section 4.2, the original performance of the
   566	existing methods and the performance with the addition of our CIDHL and MBSOS are compared.
   567	In Section 4.3, we do ablation experiments on the hyperparameters 𝛿 and 𝜆. Finally, in Section 4.4,
   568	we show the visualization results of the AGW algorithm with the addition of our CIDHL after the
   569	t-distributed Stochastic Neighbor Embedding (t-SNE) dimensionality reduction.
   570	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   571	
   572	Mix-Modality Person Re-Identification: A New and Practical Paradigm
   573	
   574	112:11
   575	
   576	Table 1. The Comparison of Three Datasets
   577	
   578	4.1
   579	
   580	Datasets
   581	
   582	SYSU-MM01
   583	
   584	RegDB
   585	
   586	LLCM
   587	
   588	ID_number
   589	Train_ID_number
   590	Test_ID_number
   591	Visible image
   592	Infrared image
   593	Total image
   594	
   595	491
   596	395
   597	96
   598	287,628
   599	15,792
   600	303,420
   601	
   602	412
   603	206
   604	206
   605	4,120
   606	4,120
   607	8,240
   608	
   609	1,064
   610	713
   611	351
   612	25,626
   613	21,141
   614	46,767
   615	
   616	Datasets and Settings
   617	
   618	4.1.1 Datasets. As shown in Table 1, we employed three public datasets RegDB [24], SYSUMM01 [33], and LLCM [42]. The RegDB, SYSU-MM01, and LLCM datasets are prominent in the
   619	field of cross-modal Re-ID, utilizing dual-camera setups for data collection in visible and infrared
   620	modalities. RegDB includes 412 pedestrians with 10 visible and 10 thermal infrared images each,
   621	totaling 4,120 images per modality. It offers two testing modes: infrared image search using visible
   622	images and vice versa. SYSU-MM01, a larger dataset, contains 287,628 RGB and 15,792 IR images
   623	across 491 IDs, featuring operational modes like All-Search and Indoor-Search in both single-shot
   624	and multi-shot formats. The LLCM dataset, the largest to date for VI-ReID, captured using nine
   625	cameras, comprises 46,767 images across 1,064 identities, split roughly into training and testing sets
   626	with two search modes similar to RegDB. Each dataset facilitates rigorous evaluation in cross-modal
   627	analysis, maintaining its integrity in segmenting data for varied testing scenarios.
   628	4.1.2 Metrics. The cumulative matching characteristics [23] at Rank-1, Rank-5, Rank-10, mean
   629	average precision (mAP) and mean inverse negative penalty (mINP) [39] are adopted as
   630	evaluation metrics in this article.
   631	4.1.3 Mix-Modality Paradigm. We propose a new mix-modality paradigm by assigning the
   632	visible/infrared images of the same pedestrian in the test set to the query set and the gallery set,
   633	respectively, according to the modality mixing ratio. For example, if the modality mixing ratio is
   634	3 : 7, 3/10ths of the visible images and 7/10ths of the infrared images of the same pedestrian are
   635	constructed into the query set, and the remaining images are composed into the gallery set.
   636	As shown in Figure 4, under such a testing paradigm, we tested the performance of the AGW
   637	model on the RegDB, SYSU-MM01, and LLCM datasets in the Rank-1, mAP, and mINP metrics.
   638	It can be seen that the performance of the model decreases in all other metrics and mixing ratios,
   639	except for the Rank-1 metric, which shows an improvement in the RegDB and SYSU-MM01 datasets.
   640	We analyze that the reason for the performance improvement in these two datasets is caused by
   641	the presence of image pairs with the same identity and same modality that are extremely easy
   642	to match and that the decrease in the other metrics is due to modality confusion that leads to a
   643	decrease in the overall recognition ability of the model. It is worth mentioning that, in order to
   644	measure the overall performance of the model, we adopt the multi-shot all way of testing, i.e.,
   645	all the images of the target pedestrians will be included in the gallery set, and it can be seen that
   646	in this case, in the large-scale dataset, the model’s mINP metrics appear to be greatly degraded
   647	because the query samples are farther away from the hardest positive samples than the majority of
   648	the negative samples.
   649	4.1.4 Implementation Details. All experiments were conducted with the PyTorch framework and
   650	an RTX3090 GPU. Limited by the length and open source status of the algorithms, we conducted
   651	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   652	
   653	112:12
   654	
   655	W. Liu et al.
   656	
   657	Fig. 4. The effect of different modality mixing ratios with respect to the AGW model performance in Rank-1,
   658	mAP, and mINP metrics on the three datasets. It can be seen that a general performance degradation arises
   659	on the other metrics except for the Rank-1 metrics that rise on some of the datasets. The origin stands for the
   660	unmixed dataset, mix37 stands for the query set to the gallery set with a ratio of 3 : 7 for the visible images
   661	and infrared images, and so forth. mAP, mean average precision; mINP, mean inverse negative penalty.
   662	
   663	experiments with DART [36], LUPI [1], AGW [39], HCTL [13], and DEEN [42], continuing the
   664	basic settings of these methods for training. The m in Equations (1), (3), and (4) is set to 0.3, the 𝛿 in
   665	Equation (5) is set to 0.2, the 𝜆 in Equations (7) and (8) is set to 0.999.
   666	4.2
   667	
   668	Comparison with State-of-the-Arts
   669	
   670	We evaluate the performance of our proposed method on AGW, HCTL, DEEN, and three datasets,
   671	RegDB, SYSU-MM01, and LLCM in Rank-1, Rank-5, Rank-10, mAP, and mINP. Where Mode 1
   672	represents that we input the query set to the IR branch of the original model and the gallery set to
   673	the visible branch, and Mode 2 represents the opposite. The “𝑤/𝑜 mix” denotes the performance on
   674	the origin bi-modality mutual retrieval paradigm while others are on our mix-modality paradigm.
   675	The “+M,” “+C,” and “+C+M” denote the performance of the method added MBSOS, CIDHL, and
   676	both, respectively.
   677	Comparison on RegDB. As can be seen in Tables 2 and 3, a general improvement in performance
   678	occurs with the addition of our method. In Mode 1, the AGW algorithm achieves the best performance with the addition of CIDHL, showing “+6.09%,” “+2.19%,” “+1.01%,” “+8.84%,” and “+8.21%”
   679	improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. The addition of
   680	MBSOS also shows a performance improvement of “+3.67%” and “+11.89%” in the mAP and mINP
   681	metrics, which reflects the overall performance of the model, despite a slight decrease in the rank
   682	metric. In Mode 2, the AGW algorithm with the addition of both CIDHL and MBSOS achieves the
   683	best performance, showing “+11.42%,” “+9.52%,” “+8.02%,” “+2.46%,” and “+2.00%” improvements
   684	in Rank-1, Rank-5, Rank-10, mAP, and mINIP metrics, respectively. Unlike Model 1 the performance
   685	of the model with the addition of MBSOS improves on the Rank metric. The performance of other
   686	algorithms is similar to AGW.
   687	Comparison on SYSU-MM01. As can be seen in Tables 4 and 5, a general improvement in performance occurs with the addition of our method on SYSU-MM01, too. In Mode 1, the AGW algorithm
   688	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
   689	
   690	
 succeeded in 0ms:
   388	3.3
   389	
   390	MoDAs
   391	
   392	Now, we have both domain-specific and domain-invariant features, but the target domain is
   393	inaccessible during DG ReID training. To improve generalization on unseen domains, we propose a
   394	novel MoE-based approach specially for ViT with adapters to integrate domain-specific features
   395	and get aggregated representations which are more robust.
   396	In most existing MoE-based methods, they learn diverse experts for different source domains
   397	and then calculate the relevance between the test image which is from unseen target domains and
   398	source domains. Subsequently, as guided by the relevance, multiple source domain features are
   399	mixed to create a new aggregated feature. In prior CNN-based approaches, the standard voting
   400	network needs to generate query features from backbone feature maps and calculate relevance
   401	with prototypes of different source domains. The class centers of domain are usually used as its
   402	prototypes. However, different from CNN structure with backbone plus expert heads, in our ViTbased method, to obtain the final features from k experts, we have to forward the whole ViT model
   403	k times. This is due to our domain adapters are positioned within each block (layer) of the model
   404	architecture. This will sharply increase the computation for each sample. To mitigate this issue,
   405	we propose MoDA and a cross-modality block-aware voting network to enable the model to mix
   406	features from multiple domain-specific experts at the block level. The kind of block-wise approach,
   407	compared to the sample-wise approach, also makes the model conduct a more fine-grained feature
   408	mixing in each block rather than using one single weight for all blocks.
   409	As mentioned before, in the first stage, we follow CLIP-ReID [17] to generate ID-specific tokens
   410	for each identity. Additionally, tokens from each source domain will be treated as domain-specific
   411	prototypes for their corresponding domains. These prototypes will be utilized to describe the
   412	characteristics of different source domains.
   413	In the second stage, we adopt episode learning algorithm following [36] to simulate the evaluation
   414	scenario where target domains are unseen. For a sample from the kth domain, we denote it as 𝑥 𝑘 .
   415	We assume 𝑥 𝑘 represents the current input sample to the model during training. We firstly let it
   416	pass to blocks with the kth domain-specific adapters 𝐴𝑘 , then compute the expert loss L𝑒𝑥𝑝𝑒𝑟𝑡 .
   417	And in this iteration, the kth domain will be seemed as the unseen target domain for simulation,
   418	𝐾
   419	𝐾 −1 . Then,
   420	and the remaining 𝐾 − 1 domains {𝐷𝑖 }𝑖=1,𝑖≠𝑘
   421	will be seemed as source domains {𝐷𝑖𝑠 }𝑖=1
   422	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   423	
   424	MoDA for Parameter-efficient Generalizable Person Re-ID
   425	
   426	139:9
   427	
   428	𝐾
   429	𝑥 𝑘 is passed to the MoDA blocks only with the remaining 𝐾 − 1 expert adapters {𝐴𝑖 }𝑖=1,𝑖≠𝑘
   430	, as
   431	shown in Figure 3(c).
   432	For the lth MoDA block, we denote the input and output as 𝑧𝑙 −1 and 𝑧𝑙 . As shown in Figure 2(d),
   433	in each block, 𝑥 𝑘 will be fed into Layer Norm (LN) and Multi-head Attention (MA) first. And we
   434	pass the output of MA 𝑧𝑙0 to all 𝐾 − 1 expert adapters concurrently. Then we can get 𝐾 − 1 diverse
   435	𝐾
   436	intermediate features {ℎ𝑙𝑖 }𝑖=1,𝑖≠𝑘
   437	with characteristics from different source domains. The [CLS]
   438	0
   439	token of 𝑧𝑙 will be used for the voting network to generate a block-aware query feature 𝑞𝑙 . The
   440	block-aware voting network is simply implemented with a MLP architecture, and its parameters
   441	are shared across all the MoDA blocks.
   442	𝐾
   443	Next, we use the block query feature 𝑞𝑙 and domain prototypes {𝑝𝑖 }𝑖=1,𝑖≠𝑘
   444	to calculate the
   445	relevance w.r.t. each source domain. We use inner product (h·, ·i) to compute similarities here for
   446	its simplicity:
   447	𝑀
   448	
   449	𝑟𝑙𝑖 =
   450	
   451	𝑖
   452	1 Õ
   453	h𝑞𝑙 , 𝑝𝑚 i,
   454	𝑀𝑖 𝑚=1
   455	
   456	(9)
   457	
   458	where 𝑀𝑖 is the number of identities in ith source domain 𝐷𝑖𝑠 , and 𝑝𝑚 is the mth ID-specific
   459	prototypes in 𝐷𝑖𝑠 . According to these 𝐾 − 1 relevance scores, we aggregate the intermediate features
   460	𝐾
   461	{ℎ𝑖 }𝑖=1,𝑖≠𝑘
   462	into one single feature. This process can be formulated as:
   463	𝑎𝑔𝑔
   464	
   465	ℎ𝑙
   466	
   467	=
   468	
   469	𝐾
   470	−1
   471	Õ
   472	
   473	𝑠𝑜 𝑓 𝑡𝑚𝑎𝑥 (𝑟𝑙𝑖 ) · ℎ𝑙𝑖 ,
   474	
   475	(10)
   476	
   477	𝑖≠𝑘
   478	
   479	where 𝑠𝑜 𝑓 𝑡𝑚𝑎𝑥 (·) is used to normalize the relevance.
   480	And the following computation is exactly the same as the common ViT block with adapters:
   481	𝑧𝑙0 = 𝑧𝑙 −1 + ℎ𝑙
   482	
   483	(11a)
   484	
   485	𝑧𝑙 = 𝑧𝑙0 + 𝑀𝐿𝑃 (𝐿𝑁 2 (𝑧𝑙0 )).
   486	
   487	(11b)
   488	
   489	𝑎𝑔𝑔
   490	
   491	In addition, we find that treating the global adapter as another domain-specific adapter and
   492	incorporating it into the mixture can bring better performance. And the weight of intermediate
   493	𝑔𝑙𝑜𝑏𝑎𝑙
   494	feature of global adapter ℎ𝑙
   495	is fixed to 0.5, rather than being generated by the voting network.
   496	𝑎𝑔𝑔
   497	Finally, we take the [CLS] token of the last block as the aggregated feature 𝐹𝑥 of current
   498	sample x.
   499	To optimize the block-aware voting network and MoDA blocks, the following loss functions are
   500	adopted:
   501	First. Image-to-text cross-entropy loss L𝑖2𝑡𝑐𝑒 . This part is the same as the optimization of expert
   502	𝑎𝑔𝑔
   503	adapters. The objective is to ensure the aggregated feature 𝐹𝑥 to retain inherent discriminative
   504	capability.
   505	Second. We also expect the aggregated features to be as similar as possible to those derived
   506	by domain expert and stay in close proximity in the feature space. Thus, we set two more loss
   507	functions to impose enhanced constraints: one is a consistency loss L𝑐𝑜𝑛𝑠𝑖𝑠 , inspired by [36], and
   508	the other one is L𝑚𝑠𝑒 which plays the role as a regularization L2 loss, naive but effective. They can
   509	be formulated as:
   510	+
   511	+
   512	−
   513	−
   514	L𝑐𝑜𝑛𝑠𝑖𝑠 = [𝛼 1 + Γ𝑎𝑔𝑔
   515	− Γ𝑒𝑥𝑝𝑒𝑟𝑡
   516	] + + [𝛼 2 + Γ𝑎𝑔𝑔
   517	− Γ𝑒𝑥𝑝𝑒𝑟𝑡
   518	] +,
   519	
   520	(12)
   521	
   522	+ and Γ +
   523	𝑎𝑔𝑔 and 𝐹 𝑒𝑥𝑝𝑒𝑟𝑡
   524	where 𝛼 1 and 𝛼 2 are margins, Γ𝑎𝑔𝑔
   525	𝑒𝑥𝑝𝑒𝑟𝑡 are hardest positive distances of 𝐹
   526	− and Γ −
   527	𝑎𝑔𝑔 and 𝐹 𝑒𝑥𝑝𝑒𝑟𝑡 respectively, [𝑧]
   528	respectively, Γ𝑎𝑔𝑔
   529	+
   530	𝑒𝑥𝑝𝑒𝑟𝑡 are hardest negative distances of 𝐹
   531	
   532	equals to 𝑚𝑎𝑥 (𝑧, 0) [36]. As we expect the aggregated features to be as similar as possible to those
   533	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   534	
   535	139:10
   536	
   537	Y. Wang et al.
   538	
   539	Algorithm 1: Training Procedure of MoDA
   540	
   541	derived by domain expert, we utilize this loss function by minimizing the difference of the distances
   542	between them and their respective hardest positive/negative sample.
   543	And the L𝑚𝑠𝑒 loss is formulated as:
   544	1
   545	L𝑚𝑠𝑒 = k𝐹 𝑒𝑥𝑝𝑒𝑟𝑡 − 𝐹 𝑎𝑔𝑔 k 22 .
   546	2
   547	
   548	(13)
   549	
   550	The total loss can be eventually formulated as:
   551	L𝑎𝑔𝑔 = L𝑖2𝑡𝑐𝑒 + L𝑐𝑜𝑛𝑠𝑖𝑠 + 𝜆L𝑚𝑠𝑒 .
   552	
   553	(14)
   554	
   555	L = L𝑒𝑥𝑝𝑒𝑟𝑡 + L𝑔𝑙𝑜𝑏𝑎𝑙 + L𝑎𝑔𝑔 .
   556	
   557	(15)
   558	
   559	𝑎𝑔𝑔
   560	
   561	Algorithm 1 shows the overall training procedure. During test time, all K domain adapters will
   562	be used for voting network to produce the aggregated feature.
   563	4
   564	4.1
   565	
   566	Experiments
   567	Datasets and Evaluation Settings
   568	
   569	Datasets. We conduct experiments on several person ReID benchmarks following existing works
   570	[36, 39]: Market1501 [45], MSMT17 [34], CUHK02 [18], CUHK03 [19], CUHK-SYSU [35], PRID [9],
   571	GRID [24], VIPeR [5], and iLIDs [46]. For simplicity, we denote Market1501, MSMT17, CUHK02,
   572	CUHK03, CUHK-SYSU as M, MS, C2, C3, and CS in the following. For CUHK03, we use the “labeled”
   573	data and do not use the DukeMTMC [47] due to its privacy issues following [36, 39]. We conduct
   574	experiments on these nine public ReID datasets mentioned above. The details of these datasets are
   575	illustrated in Table 1.
   576	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   577	
   578	MoDA for Parameter-efficient Generalizable Person Re-ID
   579	
   580	139:11
   581	
   582	Table 1. The Details of Datasets
   583	
   584	Datasets
   585	Market1501(M)
   586	MSMT17(MS)
   587	CUHK02(C2)
   588	CUHK03(C3)
   589	CUHK-SYSU(CS)
   590	PRID
   591	GRID
   592	VIPeR
   593	iLIDs
   594	
   595	# IDs # Images # Cameras
   596	1,501
   597	32,217
   598	6
   599	4,101
   600	126,441
   601	15
   602	1,816
   603	7,264
   604	10
   605	1,467
   606	14,096
   607	2
   608	11,934
   609	34,574
   610	1
   611	749
   612	949
   613	2
   614	1,025
   615	1,275
   616	8
   617	632
   618	1,264
   619	2
   620	300
   621	4,515
   622	2
   623	
   624	The meaning of each column is as follows: the name of dataset, the
   625	quantity of unique person IDs, the total number of images in the
   626	dataset, and the number of distinct camera perspectives when constructing datasets.
   627	
   628	Table 2. The Details of Evaluation Protocols
   629	
   630	Protocols
   631	Protocol-1
   632	Protocol-2
   633	Protocol-3
   634	
   635	Training sets
   636	
   637	Testing sets
   638	PRID, GRID,
   639	Full-(M+C2+C3+CS)
   640	VIPeR, iLIDs
   641	M+MS+CS
   642	C3
   643	M+CS+C3
   644	MS
   645	MS+CS+C3
   646	M
   647	Full-(M+MS+CS)
   648	C3
   649	Full-(M+CS+C3)
   650	MS
   651	Full-(MS+CS+C3)
   652	M
   653	
   654	Evaluation Settings. The Mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) are used for evaluation. There are three testing protocols like the prior works
   655	[2, 36, 39] to evaluate the performance extensively.
   656	We show the details of the three evaluation protocols adopted in our experiments in Table 2. For
   657	protocol-1, model is trained with both the train and test images in M+C2+C3+CS datasets (denoted
   658	as Full-(M+C2+C3+CS)), and then tested on four small datasets: PRID, GRID, VIPeR, and iLIDs,
   659	respectively. Like [39], the results are evaluated on 10 repeated random splits of query and gallery
   660	sets. The average of results will be reported. For protocol-2 and protocol-3 we choose one domain
   661	from M+MS+CS+C3 for testing and the remaining three domains for training. Protocol-2 only
   662	uses the training data of source domains (denoted as M+MS+CS, M+CS+C3, and MS+CS+C3),
   663	while protocol-3 uses both training and testing data of source domains for training (denoted as
   664	Full-(M+MS+CS), Full-(M+CS+C3), and Full-(MS+CS+C3)).
   665	4.2
   666	
   667	Implementation Details
   668	
   669	We adopt the image encoder and text encoder of pretrained CLIP model as our backbone. For the
   670	image encoder, we use ViT-B/16, which contains 12 transformer layers with the hidden size of
   671	768 dimensions. The parameters of both the image encoder and the text encoder are frozen. The
   672	block voting network is implemented with a FC1-GELU-LN-FC2 architecture, where FC1 layer
   673	expands the dimensions of [CLS] token four times and the FC2 layer projects it back to the original
   674	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   675	
   676	139:12
   677	
   678	Y. Wang et al.
   679	Table 3. Comparison with SOTA Methods Under Protocol-1
   680	
   681	Source
   682	
   683	M+D+C2
   684	+C3+CS
   685	
   686	M+C2
   687	+C3+CS
   688	
   689	Method
   690	
   691	Reference
   692	
   693	DIMN [28]
   694	SNR [14]
   695	RaMoE [2]
   696	DMG-Net [39]
   697	QAConv50 [21]
   698	M3 L [44]
   699	MetaBIN [1]
   700	ACL [39]
   701	META [36]
   702	Baseline (ViT)
   703	Baseline (CLIP)
   704	MoDA (ours)
   705	
   706	CVPR 2019
   707	CVPR 2020
   708	CVPR 2021
   709	ECCV 2022
   710	ECCV 2020
   711	CVPR 2021
   712	CVPR 2021
   713	ECCV 2022
   714	ECCV 2022
   715	
   716	PRID
   717	mAP
   718	R1
   719	52.0 39.2
   720	66.5 52.1
   721	67.3 57.7
   722	68.4 60.6
   723	62.2 52.3
   724	64.3 53.1
   725	70.8 61.2
   726	73.4 63.0
   727	71.7 61.9
   728	61.1 50.0
   729	61.4 52.0
   730	71.2 61.6
   731	
   732	Target
   733	GRID
   734	VIPeR
   735	mAP
   736	R1
   737	mAP
   738	R1
   739	41.1 29.3 60.1 51.2
   740	47.7 40.2 61.3 52.9
   741	54.2 46.8 64.6 56.6
   742	56.6 51.0 60.4 53.9
   743	57.4 48.6 66.3 57.0
   744	55.0 44.4 66.2 57.5
   745	57.9 50.2 64.3 55.9
   746	65.7 55.2 75.1 66.4
   747	60.1 52.4 68.4 61.5
   748	42.2 43.7 62.6 55.4
   749	43.6 39.1 65.5 56.5
   750	52.3 49.9 73.4 65.4
   751	
   752	iLIDs
   753	mAP
   754	R1
   755	78.4 70.2
   756	89.9 84.1
   757	90.2 85.0
   758	83.9 79.3
   759	81.9 75.0
   760	81.5 74.0
   761	82.7 74.7
   762	86.5 81.8
   763	83.5 79.2
   764	77.2 71.7
   765	77.6 73.0
   766	87.0 82.7
   767	
   768	Average
   769	mAP
   770	R1
   771	57.9 47.5
   772	66.4 57.3
   773	69.1 61.5
   774	67.3 61.2
   775	67.0 58.2
   776	66.8 57.3
   777	68.9 60.5
   778	75.2 66.6
   779	70.9 63.8
   780	60.8 55.2
   781	62.0 55.2
   782	71.3 64.9
   783	
   784	# Tunable
   785	Params (M)
   786	60a
   787	60a
   788	63a
   789	60a
   790	60
   791	57
   792	57
   793	57a
   794	165
   795	107
   796	130
   797	19.1
   798	
   799	All the images in the source domains are used for training. Bold type indicates the best performance and underline
   800	denotes the second position in performance ranking among methods in comparison.
   801	a The number of tunable parameters is estimated based on the classification head parameters under the current protocol,
   802	otherwise comes from the actual results of running code.
   803	
   804	dimension. We resize the person image size to 256 × 128. For data augmentation, we perform
   805	random cropping, random flipping, color jittering, and auto augmentation [7] in the second stage.
   806	Similar to prior works [2, 36], we also discard random erasing (REA) which may affect the DG
   807	performance. We conduct all the experiments with PyTorch and the help of codebases: TransReID
   808	[8] and FastReID [7]. All the models are trained on RTX 3090 GPU.
   809	For the first training stage, we use the Adam optimizer with a learning rate initialized at 3.5 × 10−4
   810	and decayed by a cosine schedule. We train the learnable ID-specific tokens for 60 epochs and
   811	adopt the warmup strategy in the first 5 epochs. Notably, we use no augmentation in this stage.
   812	And the number of tokens [𝑋 ]𝑚 is set to 4 following CLIP-ReID [17].
   813	For the second training stage, the batch size is set to 64, including 16 identities and 4 images per
   814	identity. We use the Adam optimizer with a learning rate initialized at 3 × 10−4 . We train the model
   815	for 60 epochs and adopt warmup strategy in the first epochs. And the learning rate is decayed by
   816	0.1 at the 30th and 50th epochs. Only the adapters are optimized in this stage. The weight 𝜆 of L𝑚𝑠𝑒
   817	in L𝑎𝑔𝑔 is set to 0.1. The margins 𝛼 1, 𝛼 2 in L𝑐𝑜𝑛𝑠𝑖𝑠 are both set to 0.1.
   818	4.3
   819	
   820	Comparisons to the SOTA
   821	
   822	Comparison under Protocol-1. We compare our method with previous DG-ReID methods under
   823	Protocol-1, which are tested on four small datasets (PRID, GRID, VIPeR, and iLIDs). As shown in
   824	Table 3, our method could achieve competitive even better results with the SOTA methods in a
   825	more parameter-efficient way by fine-tuning much fewer parameters. And on average our method
   826	outperforms most SOTA methods. It is important to note that all the reported results were obtained
   827	without employing any post-processing techniques, such as re-ranking or query expansion [49].
   828	Due to the fact that most previous methods are based on CNN, in order to make a fairer comparison
   829	and demonstrate the effectiveness of our method, we have implemented two baseline methods
   830	based on ViT and CLIP. We combine all source domain datasets together into a single hybrid dataset
   831	for training. The implementation of the two transformer-based methods are inspired by TransReID
   832	[8] and CLIP-ReID [17]. Both of them are fully fine-tuned during training.
   833	Comparison under Protocol-2 and Protocol-3. We also compare our method with other methods
   834	under protocol-2 and protocol-3, as shown in Tables 4 and 5. Our method still maintains competitive
   835	performance within the top three, but it does not stand out as prominently as under protocol-1.
   836	The potential reasons for such results will be thoroughly analyzed and discussed in Section 4.5.
   837	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
   838	
   839	
exec
/bin/zsh -lc "pdftotext -f 1 -l 12 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf' - | nl -ba | sed -n '556,980p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   556	Modality Bridge Similarity Optimization Strategy
   557	
   558	The Modality Bridge Similarity Optimization Strategy (MBSOS) addresses the modality confusion
   559	issue by innovatively transforming the distance calculation process between a query sample and a
   560	gallery sample into a pathfinding problem. Here’s how it works:
   561	• Bridge Node Concept: Each query sample, 𝑞𝑖 ∈ Q, seeks the optimal path to a gallery sample,
   562	𝑔 𝑗 ∈ G, not by direct distance calculation but through intermediate "bridge nodes" within
   563	the gallery.
   564	• Pathfinding Optimization: The strategy uses these bridge nodes, which are samples in the
   565	gallery of similar modality to the query, to create a modality-consistent path. This method
   566	effectively reduces the impact of direct modality differences on the distance metric.
   567	• Distance Optimization: By navigating through these intermediate nodes, the system optimizes the apparent distance between 𝑞𝑖 and 𝑔 𝑗 . This optimization aims to provide a more
   568	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   569	
   570	Mix-Modality Person Re-Identification: A New and Practical Paradigm
   571	
   572	9
   573	
   574	accurate similarity measure by leveraging intra-modality consistency to guide cross-modality
   575	comparisons.
   576	The core objective of MBSOS is to enhance the robustness of cross-modality person re-identification
   577	by minimizing the distortive effects of modality differences on the identity-matching process. This
   578	approach not only streamlines the matching process but also significantly improves the precision
   579	of identity verification across different imaging conditions.
   580	3.4.1 Distance Map Construction. To obtain the relationship between sample points, we first
   581	calculate the distance 𝑑 (𝑞𝑖 , 𝑔𝑖 ) between query sample 𝑞𝑖 and gallery sample 𝑔𝑖 , as well as the distance
   582	𝑑 (𝑔𝑖 , 𝑔 𝑗 ) between gallery samples 𝑔𝑖 and 𝑔 𝑗 as follows:
   583	𝑑 (𝑞𝑖 , 𝑔 𝑗 ) = 𝑞𝑖 − 𝑔 𝑗 2
   584	𝑑 (𝑔 𝑗 , 𝑔𝑡 ) = 𝑞 𝑗 − 𝑔𝑡 2 ,
   585	
   586	(6)
   587	
   588	where(𝑖 = 1, 2, . . . , 𝑁𝑞 ) , ( 𝑗 = 1, 2, . . . , 𝑁𝑔 ) and (𝑡 = 1, 2, . . . , 𝑁𝑔 ). Then, we construct a distance
   589	map M𝑄𝐺 (V, E) by 𝑑 (𝑞𝑖 , 𝑔𝑖 ) to represent the distances between the query set and gallery set,
   590	and another distance map M𝐺𝐺 (V, E) using 𝑑 (𝑔 𝑗 , 𝑔𝑡 ) to represent the distances between gallery
   591	samples in the gallery set. In M𝑄𝐺 and M𝐺𝐺 , each vertex V represents a pedestrian image, and
   592	each edge E represents the distance between two vertices. Considering that modality confusion is
   593	mainly caused by the fact that the same modality naturally has a closer distance, we propose an
   594	appropriate scaling adjustment for the distance of the edge E of the same modality to attenuate the
   595	natural effects of this imaging mechanism, the edges between query sample and gallery sample of
   596	the M𝑄𝐺 adjusted as follows:
   597	(
   598	𝜆d(qi, gj ) if ml(qi ) = ml(gj )
   599	𝑖𝑗
   600	Eqg =
   601	(7)
   602	d(qi, gj )
   603	otherwise,
   604	𝑖𝑗
   605	where Eqg
   606	is the adjusted distance between 𝑖 − 𝑡ℎ and 𝑗 − 𝑡ℎ vertex in M𝑄𝐺 . The ml(qi ) and ml(gj )
   607	are the modality label of the 𝑞𝑖 and 𝑔 𝑗 . The edges in M𝐺𝐺 can be adjusted similarly as follows:
   608	(
   609	𝜆d(gj, gt ) if ml(gi ) = ml(gt )
   610	𝑗𝑡
   611	Egg =
   612	(8)
   613	d(gj, gt )
   614	otherwise.
   615	
   616	3.4.2 Modality Brige Optimization. After constructing a distance graph M Q G , computing the
   617	nearest distance 𝑑˜𝑖 𝑗 from a given probe qi to a point gj in the gallery can be viewed as finding the
   618	shortest path from a probe node qi to a gallery node gj in the M Q G with the help of a bridge node
   619	𝑔𝑡 in M𝐺𝐺 . More specifically, a path ptij from a probe node qi to a gallery node gj in the M Q G can
   620	be represented as follows:
   621	𝑡𝑗
   622	𝑖𝑡
   623	𝑝𝑖𝑡 𝑗 = (𝑞𝑖 , 𝑔𝑡 , 𝑔 𝑗 ) = E𝑞𝑔
   624	+ E𝑔𝑔
   625	.
   626	(9)
   627	Then, the set of all paths between qi and gj is P = (𝑝𝑖1𝑗 , . . . , 𝑝𝑖𝑛𝑗 ), where 𝑛 = 𝑁𝐺 . Therefore, the
   628	shortest distance 𝑑˜𝑖 𝑗 between 𝑞𝑖 and 𝑔 𝑗 can be calculated as follows:
   629	n
   630	o
   631	𝑡𝑗
   632	𝑖𝑡
   633	𝑑˜𝑖 𝑗 = min P = min E𝑞𝑔
   634	+ E𝑔𝑔
   635	.
   636	(10)
   637	1≤𝑡 ≤𝑁𝑔
   638	
   639	1≤𝑡 ≤𝑁𝑔
   640	
   641	Finally, we obtain the optimized distance map M̃𝑄𝐺 constructed by all the shortest distance 𝑑˜𝑖 𝑗
   642	between the query nodes 𝑞𝑖 and 𝑔 𝑗 , which is optimized with the help of a bridge node 𝑔𝑡 . It is worth
   643	mentioning that in some cases, the direct distance from 𝑞𝑖 to 𝑔 𝑗 is the shortest distance without
   644	the help of any intermediate bridge node. The algorithmic procedure of MBSOS is described in
   645	Algorithm 1.
   646	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   647	
   648	10
   649	
   650	WEI LIU, XIN XU, HUA CHANG, XIN YUAN, and ZHENG WANG
   651	
   652	Algorithm 1 modality Bridge Similarity Optimization Strategy
   653	Input: Query set Q, Gallery set G, Query-gallery distance map M𝑄𝐺 , gallery-gallery distance
   654	M𝐺𝐺 , modality label 𝑚𝑙
   655	Output: Optimized M̃𝑄𝐺
   656	1: for each 𝑞𝑖 , 𝑔 𝑗 in M𝑄𝐺 do
   657	2:
   658	if 𝑚𝑙 (𝑞𝑖 )=𝑚𝑙 (𝑔 𝑗 ) then
   659	𝑖𝑗
   660	according to Eq. 7.
   661	3:
   662	adjust E𝑞𝑔
   663	4:
   664	end if
   665	5: end for
   666	6: for each 𝑔 𝑗 , 𝑔𝑡 in M𝐺𝐺 do
   667	7:
   668	if 𝑚𝑙 (𝑔 𝑗 )=𝑚𝑙 (𝑔𝑡 ) then
   669	𝑗𝑡
   670	8:
   671	adjust E𝑔𝑔
   672	according to Eq. 8.
   673	9:
   674	end if
   675	10: end for
   676	11: for each 𝑞𝑖 in Q and 𝑔 𝑗 in G do
   677	12:
   678	for 𝑔𝑡 in G do
   679	13:
   680	Calculate 𝑝𝑖𝑡 𝑗 = 𝑑 (𝑞𝑖 , 𝑔𝑡 , 𝑔 𝑗 ) according to Eq. 9.
   681	14:
   682	end for
   683	15:
   684	Sort 𝑝𝑖𝑡 𝑗 for all 𝑔𝑡 .
   685	16:
   686	Calculate 𝑑˜𝑖 𝑗 according to Eq. 10.
   687	17: end for
   688	18: Construct M̃𝑄𝐺 by all the 𝑑˜𝑖 𝑗
   689	19: return M̃𝑄𝐺
   690	Table 1. The comparison of three datasets.
   691	
   692	Datasets
   693	SYSU-MM01
   694	ID_number
   695	491
   696	Train_ID_number
   697	395
   698	Test_ID_number
   699	96
   700	Visible image
   701	287,628
   702	Infrared image
   703	15,792
   704	Total image
   705	303,420
   706	4
   707	
   708	RegDB
   709	412
   710	206
   711	206
   712	4,120
   713	4,120
   714	8,240
   715	
   716	LLCM
   717	1,064
   718	713
   719	351
   720	25,626
   721	21,141
   722	46,767
   723	
   724	EXPERIMENTAL DESIGN AND RESULTS
   725	
   726	In this section we first present the adopted dataset, evaluation metrics, mix-modality paradigm,
   727	and method implementation details in Section 4.1. In Section 4.2 the original performance of the
   728	existing methods and the performance with the addition of our CIDHL and MBSOS are compared.
   729	In Section 4.3 we do ablation experiments on the hyperparameters 𝛿 and 𝜆. Finally, in Section 4.4
   730	we show the visualization results of the AGW algorithm with the addition of our CIDHL after the
   731	t-SNE dimensionality reduction.
   732	4.1
   733	
   734	Datasets and Settings
   735	
   736	4.1.1 Datasets. As shown in Table 1, we employed three public datasets RegDB [24], SYSUMM01 [31], and LLCM [40]. The RegDB, SYSU-MM01, and LLCM datasets are prominent in the field
   737	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   738	
   739	Mix-Modality Person Re-Identification: A New and Practical Paradigm
   740	
   741	90
   742	
   743	60
   744	
   745	50
   746	
   747	80
   748	
   749	50
   750	
   751	40
   752	
   753	70
   754	
   755	mINP
   756	
   757	60
   758	
   759	40
   760	
   761	30
   762	
   763	60
   764	
   765	30
   766	
   767	20
   768	
   769	50
   770	
   771	20
   772	
   773	10
   774	
   775	40
   776	
   777	10
   778	
   779	0
   780	origin mix37 mix46 mix55 mix64 mix73
   781	
   782	origin mix37 mix46 mix55 mix64 mix73
   783	
   784	100
   785	
   786	80
   787	
   788	70
   789	
   790	90
   791	
   792	70
   793	
   794	60
   795	
   796	80
   797	
   798	60
   799	
   800	70
   801	
   802	50
   803	
   804	60
   805	
   806	40
   807	
   808	50
   809	
   810	30
   811	
   812	40
   813	
   814	20
   815	origin mix37 mix46 mix55 mix64 mix73
   816	
   817	50
   818	mINP
   819	
   820	mAP
   821	
   822	Rank-1
   823	
   824	LLCM
   825	
   826	70
   827	
   828	origin mix37 mix46 mix55 mix64 mix73
   829	
   830	Mode2
   831	
   832	SYSU-MM01
   833	
   834	100
   835	
   836	mAP
   837	
   838	Mode1
   839	
   840	Rank-1
   841	
   842	RegDB
   843	
   844	11
   845	
   846	40
   847	30
   848	20
   849	10
   850	
   851	origin mix37 mix46 mix55 mix64 mix73
   852	
   853	0
   854	origin mix37 mix46 mix55 mix64 mix73
   855	
   856	Fig. 4. The effect of different modality mixing ratios with respect to the AGW model performance in Rank-1,
   857	mAP, and mINP metrics on the three datasets. It can be seen that a general performance degradation arises
   858	on the other metrics except for the Rank-1 metrics that rise on some of the datasets. The origin stands for the
   859	unmixed dataset, mix37 stands for the query set to the gallery set with a ratio of 3 : 7 for the visible images
   860	and infrared images, and so forth.
   861	
   862	of cross-modal person re-identification, utilizing dual-camera setups for data collection in visible
   863	and infrared modalities. RegDB includes 412 pedestrians with 10 visible and 10 thermal infrared
   864	images each, totaling 4, 120 images per modality. It offers two testing modes: infrared image search
   865	using visible images and vice versa. SYSU-MM01, a larger dataset, contains 287, 628 RGB and 15, 792
   866	IR images across 491 IDs, featuring operational modes like All-Search and Indoor-Search in both
   867	single-shot and multi-shot formats. The LLCM dataset, the largest to date for VI-ReID, captured
   868	using nine cameras, comprises 46, 767 images across 1, 064 identities, split roughly into training and
   869	testing sets with two search modes similar to RegDB. Each dataset facilitates rigorous evaluation
   870	in cross-modal analysis, maintaining its integrity in segmenting data for varied testing scenarios.
   871	4.1.2 Metrics. The cumulative matching characteristics (CMC) [23] at Rank-1, Rank-5, Rank-10,
   872	mean average precision (mAP) and mean inverse negative penalty (mINP) [37] are adopted as
   873	evaluation metrics in this paper.
   874	4.1.3 Mix-Modality Paradigm. We propose a new mix-modality paradigm by assigning the
   875	visible/infrared images of the same pedestrian in the test set to the query set and the gallery set,
   876	respectively, according to the modality mixing ratio. For example, if the modality mixing ratio is
   877	3 : 7, 3/10𝑡ℎ𝑠 of the visible images and 7/10𝑡ℎ𝑠 of the infrared images of the same pedestrian are
   878	constructed into the query set, and the remaining images are composed into the gallery set.
   879	As shown in Figure 3, under such a testing paradigm, we tested the performance of the AGW
   880	model on the RegDB, SYSU-MM01, and LLCM datasets in the Rank-1, mAP, and mINP metrics. It
   881	can be seen that the performance of the model decreases in all other metrics and mixing ratios,
   882	except for the Rank-1 metric, which shows an improvement in the RegDB and SYSU-MM01 datasets.
   883	We analyze that the reason for the performance improvement in these two datasets is caused by the
   884	presence of image pairs with the same identity and same modality that are extremely easy to match
   885	J. ACM, Vol. , No. , Article . Publication date: May 2024.
   886	
   887	12
   888	
   889	WEI LIU, XIN XU, HUA CHANG, XIN YUAN, and ZHENG WANG
   890	
   891	Table 2. Comparison to the state-of-the-art methods on RegDB datasets in mode 1. The bold result is the
   892	best of the same method, while the ↑is the method with performance improvement (the same as below).
   893	
   894	Method
   895	DART 𝑤/𝑜 mix
   896	DART
   897	AGW 𝑤/𝑜 mix
   898	AGW
   899	AGW + M
   900	AGW + C
   901	AGW + C + M
   902	HCTL 𝑤/𝑜 mix
   903	HCTL
   904	HCTL + M
   905	HCTL + C
   906	HCTL + C + M
   907	DEEN 𝑤/𝑜 mix
   908	DEEN
   909	DEEN + M
   910	DEEN + C
   911	DEEN + C + M
   912	
   913	Rank-1
   914	72.80
   915	98.52
   916	81.23
   917	89.34
   918	81.83
   919	95.43↑
   920	94.34↑
   921	88.90
   922	78.63
   923	77.93
   924	90.05↑
   925	78.29
   926	87.95
   927	97.55
   928	97.58↑
   929	97.82↑
   930	97.90↑
   931	
   932	Rank-5
   933	82.04
   934	99.74
   935	90.39
   936	96.21
   937	90.56
   938	98.40↑
   939	98.02↑
   940	94.47
   941	86.69
   942	87.48↑
   943	96.21↑
   944	86.76↑
   945	93.89
   946	99.33
   947	99.31
   948	99.35↑
   949	99.39↑
   950	
   951	Mode 1
   952	Rank-10
   953	86.74
   954	99.74
   955	93.93
   956	98.19
   957	94.44
   958	99.20↑
   959	99.03↑
   960	96.18
   961	90.30
   962	91.60↑
   963	98.32↑
   964	91.42↑
   965	96.24
   966	99.69
   967	99.71↑
   968	99.71↑
   969	99.72↑
   970	
   971	mAP
   972	68.35
   973	81.51
   974	74.81
   975	60.78
   976	64.45↑
   977	69.62↑
   978	68.42↑
   979	80.67
   980	59.35

codex
方法正文已经够支撑机制理解，但“证据闭环”需要看作者自己用什么消融或可视化证明观察成立。我继续从实验分析里抽关键消融，不取整篇结果表，只看能支撑问题重定义的部分。
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
/bin/zsh -lc "pdftotext -f 1 -l 10 'MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '675,1020p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   675	3. EXPERIMENTS
   676	3.1. Datasets and Protocols
   677	Datasets. We evaluate the proposed method on four mainstream
   678	Cloth-changing person Re-ID benchmarks: PRCC [17], LTCC [4],
   679	VC-Clothes [18], and LaST [19]. The first three are medium-scale
   680	datasets, while LaST is a large-scale dataset.
   681	Implementation details. We use CAL[11] as the baseline CC-ReID
   682	model in our experiments as it generally performed the best with our
   683	generated data across all datasets. We use ResNet-50 pre-trained on
   684	ImageNet with a maxavg global pooling head and BNNeck. Inputs
   685	are resized to 384 × 192. For random erasing, the optimal erase
   686	range is [0.1, 0.3] on PRCC, LaST and VC-Clothes, and [0.2, 0.5] on
   687	LTCC. We train for 60 epochs using Adam (lr 3.5 × 10−4 , weight
   688	decay 5 × 10−4 ) with a step scheduler (decay ×0.1 at epochs 20 and
   689	40). All experiments run on two NVIDIA RTX 3080 Ti GPUs. For
   690	PRCC/LTCC/VC-Clothes we sample 4 identities × 16 images per
   691	identity per GPU, for LaST we sample 2 identities × 16 images per
   692	identity per GPU. λcal performs best at 0.5 on PRCC/LaST/LTCC,
   693	1.0 on VC-Clothes, and λatt = 1.0 performs best across all datasets.
   694	Evaluation is performed every 5 epochs.
   695	Evaluation metrics. We report Rank-1 (R1) and mean Average
   696	Precision (mAP), the two common metrics in Cloth-changing Re-ID.
   697	For all dataset, we evaluate our method under the standard setting and
   698	the cloth-changing setting following[8].
   699	3.2. Comparison with State-of-the-Art Methods
   700	Comparative results on PRCC and LTCC. Tables 1 reports comparisons on PRCC and LTCC with classic Re-ID methods (e.g.,
   701	HACNN [20], PCB [21], IANet [22]) and recent CC-ReID methods
   702	(e.g., AIM [12], RLQ [27], CAL [11]). On PRCC (cloth-changing),
   703	MSP-ReID shows a clear margin over the CAL baseline (about ten
   704	points in Rank-1) and is on par with, or slightly better than, recent
   705	strong competitors such as RLQ. On PRCC (standard), performance
   706	
   707	Table 2. Results on VC-Clothes under General, SC and CC protocols.
   708	The best and second-best results are shown in bold and underlined.
   709	Method
   710	
   711	Year
   712	
   713	Table 4. Ablation study of HSOA, CPRE, PRA on PRCC and VCClothes.
   714	Methods Baseline HSOA CPRE PRA
   715	
   716	General
   717	R1
   718	mAP
   719	
   720	R1
   721	
   722	SC
   723	mAP
   724	
   725	R1
   726	
   727	CC
   728	mAP
   729	
   730	MDLA [1]
   731	PCB [21]
   732	PS [28]
   733	FSAM [7]
   734	BSGA [29]
   735	STL and ACL [30]
   736	PAH-Net[9]
   737	DLCR [31]
   738	
   739	ICCV’17
   740	ECCV’18
   741	SPL’21
   742	CVPR’21
   743	BMVC’22
   744	CSCWD’24
   745	IJON’24
   746	WACV’25
   747	
   748	88.9
   749	87.7
   750	93.1
   751	94.4
   752	94.2
   753	-
   754	
   755	76.8
   756	74.6
   757	84.9
   758	88.2
   759	90.2
   760	-
   761	
   762	94.3
   763	94.7
   764	94.7
   765	94.7
   766	94.9
   767	95.1
   768	95.3
   769	-
   770	
   771	93.9
   772	94.3
   773	92.9
   774	94.8
   775	94.4
   776	95.5
   777	95.2
   778	-
   779	
   780	59.2
   781	62.0
   782	82.4
   783	78.6
   784	84.5
   785	83.9
   786	86.4
   787	87.1
   788	
   789	60.8
   790	62.2
   791	80.3
   792	78.9
   793	84.3
   794	85.2
   795	86.1
   796	81.1
   797	
   798	CAL [11]
   799	Ours
   800	
   801	CVPR’22
   802	—
   803	
   804	92.9
   805	94.5
   806	
   807	87.2
   808	87.2
   809	
   810	95.1
   811	95.4
   812	
   813	95.3
   814	95.6
   815	
   816	81.4
   817	87.1
   818	
   819	81.7
   820	82.3
   821	
   822	1
   823	2
   824	3
   825	4
   826	5
   827	6
   828	7
   829	
   830	✓
   831	✓
   832	✓
   833	✓
   834	✓
   835	✓
   836	✓
   837	
   838	Ours
   839	
   840	✓
   841	
   842	mAP
   843	
   844	81.4
   845	84.9
   846	82.4
   847	82.2
   848	86.4
   849	86.2
   850	83.4
   851	
   852	81.7
   853	82.0
   854	81.9
   855	81.8
   856	82.2
   857	82.1
   858	82.1
   859	
   860	63.4
   861	
   862	87.1
   863	
   864	82.3
   865	
   866	✓
   867	✓
   868	
   869	✓
   870	
   871	✓
   872	
   873	65.1
   874	
   875	✓
   876	
   877	✓
   878	
   879	R1
   880	
   881	55.8
   882	59.6
   883	57.0
   884	57.8
   885	61.1
   886	62.1
   887	59.3
   888	
   889	✓
   890	
   891	✓
   892	✓
   893	
   894	VC-Clothes
   895	
   896	mAP
   897	
   898	55.2
   899	61.0
   900	56.7
   901	57.7
   902	63.1
   903	64.3
   904	59.5
   905	
   906	✓
   907	✓
   908	✓
   909	
   910	PRCC
   911	R1
   912	
   913	Table 3. Results on LaST (CC protocol). The best result is in bold,
   914	and the second-best result is underlined.
   915	Methods
   916	
   917	Year
   918	
   919	R1
   920	
   921	mAP
   922	
   923	OSNet [32]
   924	BoT [33]
   925	mAPLoss [19]
   926	MCL [34]
   927	Lu and Jin [35]
   928	IMS+GEP [24]
   929	RLQ [27]
   930	
   931	ICCV’19
   932	CVPR’19
   933	arXiv’21
   934	MM’22
   935	AIMS’23
   936	TMM’23
   937	arXiv’25
   938	
   939	63.8
   940	68.3
   941	69.9
   942	75.0
   943	68.9
   944	73.2
   945	77.9
   946	
   947	20.9
   948	25.3
   949	27.6
   950	22.7
   951	24.1
   952	29.8
   953	35.3
   954	
   955	CAL [11]
   956	ours
   957	
   958	CVPR’22 73.7
   959	75.4
   960	
   961	28.8
   962	30.6
   963	
   964	saturates at the upper bound. On LTCC (cloth-changing), our
   965	method yields consistent gains over CAL, while on LTCC (standard)
   966	it delivers the best retrieval quality in mAP and competitive Rank-1
   967	among RGB-only approaches.
   968	Comparative results on VC-Clothes. Table 2 compares with conventional baselines (MDLA [1], PCB [21], PS [28], FSAM [7],
   969	BSGA [29], CAL [11]) and recent CC-ReID lines (STL+ACL [30],
   970	DLCR [31]).under the General and SC protocols, MSP-ReID is
   971	competitive with the strongest counterparts. In the more challenging
   972	CC protocol, it clearly improves over CAL and matches the best
   973	reported Rank-1, while remaining RGB-only, methods with higher
   974	mAP typically rely on tailored designs or auxiliary signals.
   975	Comparative results on LaST. Table 3 presents results on the
   976	large-scale LaST (CC) benchmark against OSNet [32], BoT [33],
   977	mAPLoss [19], MCL [34], IMS+GEP [24], RLQ [27], and CAL [11].It
   978	indicates that on the large-scale LaST (CC) benchmark, MSP-ReID
   979	surpasses classic RGB baselines and its CAL baseline by a noticeable
   980	margin, leaving only a small gap to the current top-performing
   981	RLQ—likely due to domain shift and inevitable hair/clothes boundary
   982	noise at scale.
   983	3.3. Ablation Studies
   984	Effectiveness of HSOA. As shown in Tab. 4, removing or adding
   985	HSOA clearly changes performance trends. HSOA synthesizes sameID but different-hairstyle views, explicitly decoupling hairstyle from
   986	identity. This breaks the “hair shortcut” and drives the network to
   987	focus on identity-related cues (facial structure, exposed-skin geometry,
   988	limb proportions) without introducing any extra inputs at test time.
   989	Effectiveness of CPRE. Tab. 4 also indicates that CPRE steadily
   990	improves robustness. By constructing a raw/erased dual-view with
   991	a controllable clothing keep ratio, CPRE suppresses the dominance
   992	of large apparel regions while preserving partial clothing semantics.
   993	This controlled erasing regularizes training and yields more calibrated
   994	rankings, rather than overfitting to textures or colors.
   995	Effectiveness of RPA. Our Region-based Parsing Attention (RPA) is
   996	used only at training time to reweight regions—boosting face/limbs
   997	
   998	Fig. 3. Qualitative retrieval and attention. Query (top-left) and
   999	its feature map (bottom-left) are shown alongside Top-1∼Top-10
  1000	retrievals. Green boxes are correct matches and red boxes are false.
  1001	and suppressing hair. As reflected by the ablations in Tab. 4, aligning
  1002	attention with identity-related anatomy reduces gradients attracted to
  1003	hair/clothes boundaries and complements the above augmentations,
  1004	all while keeping inference RGB-only.
  1005	Module collaboration. The table further shows that single modules
  1006	are helpful but suboptimal. Pairwise combinations consistently
  1007	perform better, revealing clear complementarity: HSOA changes the
  1008	data (hairstyle diversity), CPRE changes the appearance reliability
  1009	(keep-ratio erasing), and RPA changes the feature selection (region
  1010	weighting). Using all three together fully exploits these effects and
  1011	yields the most stable cross-clothes/hairstyle retrieval.
  1012	Visualization of retrieval results. Fig. 3 qualitatively verifies these
  1013	behaviors. Compared with the baseline, our method correctly retrieves
  1014	the same identity under clothing and hairstyle changes, avoiding
  1015	confusion from similar colors or textures. The feature maps exhibit a
  1016	clear “hot face, cold hair/clothes” pattern—high responses on face,
  1017	shoulders, and limbs with suppressed activations on hair and large
  1018	apparel regions—consistent with HSOA, CPRE, and RPA.
  1019	4. CONCLUSION
  1020	In this paper, we proposed MSP-ReID, a unified framework that

exec
/bin/zsh -lc "pdftotext -f 1 -l 10 'Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '715,1115p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   715	5. Experiments
   716	5.1. Datasets and Evaluation Protocol
   717	Datasets. We evaluate our method on three standard largescale Person Re-ID datasets, including Market-1501 [34],
   718	DukeMTMC-reID [20], and MSMT17 [26]. Market-1501
   719	consists of 32,668 images of 1,501 identities captured by 6
   720	cameras, in which the training set comprises 12,936 images
   721	of 751 identities, and the test data set comprises 19,732 images of 750 identities. DukeMTMC-reID consists of 36,411
   722	images of 1,812 identities captured by 8 cameras, where the
   723	training dataset includes 16,522 images of 702 identities.
   724	MSMT17 is the largest Person Re-ID dataset, it includes
   725	126,411 images of 4,101 identities captured by 15 cameras,
   726	and its training set has 32,621 images of 1,041 identities,
   727	while its test dataset has 93,820 bounding boxes of 3,060
   728	identities.
   729	Evaluation Protocol. Following [34, 20, 26], we evaluate performance with the retrieval precision metric Cumulative Matching Characteristic (CMC) scores, and the recall
   730	metric Mean Average Precision (mAP).
   731	
   732	5.2. Implementation Details
   733	We implement our method in PyTorch [17] with a single
   734	NVIDIA GeForce GTX 1080Ti GPU. For the CNN part, we
   735	adopt ResNet50 [8], with the layers after pooling-5 removed,
   736	and a batch normalization layer appended. For an input
   737	image, F produces a 2048-dimensional feature. Similar
   738	to [24, 11], for the input image we use CamStyle [38] for
   739	data augmentation and resize it at 256 × 128, and then preprocess it with random crop, random rotation, random color
   740	jitter, and random erasing. For the perturbed image, we add
   741	random grayscale conversion and random Gaussian blur.
   742	In an alternating manner, we train the CNN F and the
   743	GCN G using SGD with a 0.9 momentum. The number of
   744	training epochs is set to 40. For F, the initial learning rate
   745	for the ResNet50 backbone is set to 0.01, and 0.1 for all
   746	other layers. The learning rate is reduced by a factor of 10
   747	after 20 epochs, and the training mini-batch size is 32. For G,
   748	the initial learning rate is 0.001, the likelihood threshold µ
   749	is 0.5, and the mini-batch size of 32. For the feature memory
   750	M, the updating rate γ (t) starts at 1 at the first epoch and
   751	
   752	Methods
   753	Super.
   754	KNN
   755	SS
   756	MPLP*
   757	MPRD w/o
   758	MPRD #
   759	MPRD
   760	Single
   761	
   762	Market-1501
   763	Rank-1 mAP
   764	87.0
   765	68.5
   766	72.7
   767	35.2
   768	72.8
   769	39.9
   770	80.0
   771	44.5
   772	75.8
   773	43.1
   774	73.0
   775	39.3
   776	83.0
   777	51.1
   778	46.1
   779	15.5
   780	
   781	DukeMTMC-reID
   782	Rank-1
   783	mAP
   784	75.6
   785	56.7
   786	59.4
   787	32.9
   788	60.0
   789	34.3
   790	64.6
   791	39.8
   792	61.1
   793	34.9
   794	57.4
   795	36.7
   796	67.4
   797	43.7
   798	38.3
   799	14.6
   800	
   801	Table 1. Performance with different pseudo label generation methods. “Super.” and “Single” are baselines representing performance
   802	upper and lower bounds, respectively. All methods have incorporate the same Binomial deviance loss. The “*” mark in “MPLP*”
   803	indicates this implementation is based on released code from the
   804	authors. “MPRD#” and “MPRD w/o” denote ablated MPRD with
   805	CamStyle data augmentation removed and its GCN trained without
   806	unlabeled data, respectively.
   807	
   808	linearly decreases to 0.5 at the 40th epoch. In the binomial
   809	deviance loss, the weight δ is fixed at 5 and r is the number
   810	of 1% negative samples as in [24]. Moreover, we set α = 4.0
   811	and λ1 = λ2 = 0.2. The small amount of labeled meta data
   812	involves five labeled identities that are randomly selected
   813	from the training data. Specifically, the small amount of
   814	labeled metadata is only used for training G. The value of k
   815	is the maximum between 8 and the number of images whose
   816	cosine similarity to the input image is larger than 0.6.
   817	
   818	5.3. Ablation Study
   819	Effectiveness of MPRD. We compare MPRD against
   820	other pseudo label generation methods, including the KNN
   821	search, cosine similarity score (denoted as “SS”), and selection by MPLP. SS selects positive samples with a similarity
   822	threshold. MPLP is proposed in [24], which predicts pseudolabels with high accuracy via similarity scores and cycle
   823	consistency. For KNN, we empirically set K = 8, where its
   824	performance peaks; for SS, we set the similarity threshold
   825	at 0.6; for MPLP, we incorporate it with the binomial deviance loss based on its released code by the authors. Under
   826	the same setting (Section 5.2), we also conduct 2 baseline
   827	experiments, i.e., fully supervised re-ID with ground-truth
   828	(denoted as “Super.” in Table 1); erroneously supervised
   829	re-ID with image index as labeles (denoted as “Single” in
   830	Table 1), which serve as the performance upper bound and
   831	lower bound, respectively. Additionally, two ablated variants of MPRD are compared, i.e., MPRD with CamStyle
   832	data augmentation removed (“MPRD#”), and MPRD with
   833	its GCN trained purely on labeled meta data (i.e., without
   834	vast majority of the unlabeled data, denoted as “MPRD w/o”
   835	in Table 1).
   836	According to Table 1, SS outperforms KNN, which could
   837	
   838	3646
   839	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:04 UTC from IEEE Xplore. Restrictions apply.
   840	
   841	Methods
   842	MLCR*
   843	MLCR(+5id)*
   844	MPRD(0id)
   845	MPRD
   846	
   847	Figure 4. Evaluation of different likelihood threshold µ in GCN.
   848	
   849	be attributed to its more flexible pseudo-labels. After incorporating the GCN model, MPRD achieves dramatic performance advantages over SS, i.e., with rank-1 accuracy
   850	increased from 72.8% to 83.0%, and mAP increased from
   851	39.9% to 51.1% on Market-1501. Comparing “MPRD w/o”
   852	and MPRD, we also verify the necessarity of incorporating
   853	the vast majority of unlabeled data in training G. On Market1501, MPRD achieves 7.2 and 8 percentage points advantage
   854	over “MPRD w/o” in Rank-1 accuracy and mAP, respectively. A similar trend also appears on DukeMTMC-reID.
   855	Additionally, “MPRD#” achieves 73.0% rank-1 accuracy
   856	and 39.3% mAP without CamStyle. These results demonstrate the effectiveness of the proposed MPRD, and show
   857	that CamStyle boosts the performance.
   858	Impact of the likelihood threshold µ. In Eq. (7), µ
   859	determines whether two images are of the same identity, and
   860	its sensitivity analysis is presented in Figure 4. Performance
   861	is analyzed with µ values from 0.1 to 0.9 at a step size
   862	of 0.1. We observe that both rank-1 accuracy and mAP
   863	metrics increase slowly and smoothly till approximately 0.5
   864	and abruptly drop afterwards. We speculate that extreme µ
   865	values degrade performance, i.e., too small µ values lead
   866	to many false positive pairs while too large µ values incur
   867	many false negative pairs. Based on these experiments, we
   868	empirically fix µ = 0.5.
   869	Effect of small amount of labeled meta data. Since our
   870	approach introduces a small amount of labeled meta data to
   871	jump start the training of GCN, we analyze its impacts on
   872	the competing MLCR method. For fair comparison, we let
   873	MLCR have access to the same amount of labeled meta data
   874	as extra supervision. We also compare the effect of different
   875	amounts of labeled meta data in different variants of MPRD.
   876	We let MLCR have access to the same labeled meta data
   877	by replacing the pseudo labels with the ground-truth labels,
   878	whenever the input training data belong to the labeled meta
   879	dataset. In Table 2, the upper part shows that with the
   880	extra small amount of labeled meta data, “MLCR(+5id)*”
   881	marginally outperforms its original version MLCR*, possibly due to the portion of such labeled meta dataset is too
   882	small (approximately 0.5% of all data).
   883	The bottom part of Table 2 compares MPRD variants
   884	
   885	market-1501
   886	Rank-1 mAP
   887	80.1
   888	44.7
   889	80.2
   890	45.0
   891	80.9
   892	46.8
   893	83.0
   894	51.1
   895	
   896	DukeMTMC-reID
   897	Rank-1
   898	mAP
   899	64.9
   900	40.6
   901	65.3
   902	40.9
   903	65.6
   904	40.1
   905	67.4
   906	43.7
   907	
   908	Table 2. Ablation study of the effect of meta data. “MLCR(+5id)*”
   909	denotes a modified MLCR that have extra access to the
   910	same amount (5 identities) of meta data as extra supervision.
   911	“MPRD(0id)” represents an ablated version of our proposed MPRD
   912	with the labeled meta data-based jump start procedure completely
   913	removed.
   914	
   915	with different amounts of labeled meta dataset, where “5id”
   916	means five labeled identities (meta data is only used for
   917	training the GCN). If this labeled data-based jump start
   918	portion is completely removed, “MPRD(0id)” suffers from
   919	only a small performance degradation, and still outperforms
   920	the competing “MPLR*” in Table 1. When the amount of
   921	labeled identities is 5, we observe that both rank-1 accuracy
   922	and mAP increase on Market-1501 and DukeMTMC-reID.
   923	
   924	5.4. Comparison with the State-of-the-Art
   925	We evaluate the proposed MPRD on Market-1501 [34],
   926	DukeMTMC-reID [20] and MSMT17 [26] datasets. Although a small amount of labeled meta data are used to
   927	guide the training process of the GCN, our method also belongs to unsupervised Person Re-Identification because there
   928	are only few labeled data are used to train GCN, and training the feature extraction CNN module only uses unlabeled
   929	data. The proposed method is compared against the state-ofthe-art unsupervised Person Re-ID methods: LOMO [12],
   930	BOW [34], BUC [13], DBC [6], and the recent TSSL [27],
   931	SSLR [15], MLCR [24], JVTC [11]. Table 3 and Table 4
   932	summarize the comparison.
   933	Table 3 shows the results of the proposed method and
   934	state-of-the-art methods on Market-1501 and DukeMTMCreID. On Market-1501, our MPRD achieves 2.7% higher
   935	rank-1 accuracy and 5.6% higher mAP than MLCR. Compared with JVTC, our MPRD achieves 10.1% higher rank1 accuracy and 9.3% higher mAP. On DukeMTMC-reID,
   936	our MPRD achieves 2.2% higher rank-1 accuracy and 3.5%
   937	higher mAP than MLCR. Compared with JVTC, our MPRD
   938	has a sight 0.2% lower rank-1 accuracy but achieves 1.5%
   939	higher mAP. We also conduct experiments on MSMT17, and
   940	the results are presented in Table 4. From the table, our
   941	MPRD achieves 37.7% rank-1 accuracy and 14.6% mAP.
   942	Of all the competing algorithms, MLCR is the most relevant one to our proposed MPRD. As is verified in the above
   943	results, MPRD outperforms it on Market-1501, DukeMTMCreID and MSMT17. We speculate that this performance advantage arise from the following aspects. Our proposed
   944	
   945	3647
   946	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:04 UTC from IEEE Xplore. Restrictions apply.
   947	
   948	Methods
   949	LOMO [12] (CVPR15)
   950	BOW [34] (ICCV15)
   951	BUC [13] (AAAI19)
   952	DBC [6] (BMVC19)
   953	TSSL [27] (AAAI20)
   954	SSLR [15] (CVPR20)
   955	MLCR [24] (CVPR20)
   956	JVTC [11] (ECCV20)
   957	MPRD
   958	
   959	Rank-1
   960	27.2
   961	35.8
   962	66.2
   963	69.2
   964	71.2
   965	71.7
   966	80.3
   967	72.9
   968	83.0
   969	
   970	Market-1501
   971	Rank-5 Rank-10
   972	41.6
   973	49.1
   974	52.4
   975	60.3
   976	79.6
   977	84.5
   978	83.0
   979	87.8
   980	83.8
   981	87.4
   982	89.4
   983	92.3
   984	84.2
   985	88.7
   986	91.3
   987	93.6
   988	
   989	mAP
   990	8.0
   991	14.8
   992	38.3
   993	41.3
   994	43.3
   995	37.8
   996	45.5
   997	41.8
   998	51.1
   999	
  1000	Rank-1
  1001	12.3
  1002	17.1
  1003	47.4
  1004	51.5
  1005	62.2
  1006	52.5
  1007	65.2
  1008	67.6
  1009	67.4
  1010	
  1011	DukeMTMC-reID
  1012	Rank-5 Rank-10
  1013	21.3
  1014	26.6
  1015	28.8
  1016	34.9
  1017	62.6
  1018	68.4
  1019	64.6
  1020	70.1
  1021	63.5
  1022	68.9
  1023	75.9
  1024	80.0
  1025	78.0
  1026	81.6
  1027	78.7
  1028	81.8
  1029	
  1030	mAP
  1031	4.8
  1032	8.3
  1033	27.5
  1034	30.0
  1035	38.5
  1036	28.6
  1037	40.2
  1038	42.2
  1039	43.7
  1040	
  1041	Table 3. Unsupervised person re-ID performance with state-of-the-art methods on Market-1501 and DukeMTMC-reID datasets.
  1042	
  1043	Methods
  1044	MLCR [24]
  1045	JVTC [11]
  1046	MPRD
  1047	
  1048	Rank-1
  1049	35.4
  1050	39.0
  1051	37.7
  1052	
  1053	MSMT17
  1054	Rank-5 Rank-10
  1055	44.8
  1056	49.8
  1057	50.9
  1058	56.8
  1059	51.3
  1060	57.1
  1061	
  1062	Example1
  1063	
  1064	Example2
  1065	
  1066	Example3
  1067	
  1068	mAP
  1069	11.2
  1070	15.1
  1071	14.6
  1072	
  1073	Table 4. Unsupervised person re-ID performance with state-of-theart methods on MSMT17 dataset.
  1074	
  1075	MPRD introduces the neighbor structure information between sample pairs via its GCN. Through iterative and alternating training, the GCN gradually learns and refines the
  1076	distinctions in neighborhood structure between positive and
  1077	negative sample pairs, and provides higher fidelity pseudosupervision for the CNN training. The alternating, collaborative training of the GCN and the CNN could be responsible
  1078	for the performance benefits.
  1079	
  1080	(a) Without GCN
  1081	
  1082	(b) With GCN
  1083	
  1084	Figure 5. T-SNE visualization of learned features on 100 identities
  1085	of Market-1501 training set. Points with the same color are of the
  1086	same identity. The distribution of the features learned (a) without
  1087	GCN, and (b) with the GCN.
  1088	
  1089	5.5. Qualitative Results
  1090	To intuitively understand the effectiveness of MPRD, we
  1091	visualize via t-SNE [23] the learned features on Market-1501
  1092	training set, without and with the GCN, as shown in Figure 5.
  1093	By comparing the two sets of learned features side-by-side,
  1094	after introducing the GCN, points of the same identity are
  1095	pulled closer to each other, as shown in Example3 where
  1096	yellow dots are more concentrated on the right. Challenging
  1097	cases (Example1 and Example2) where points of different
  1098	identities are embedded too close to each other without GCN
  1099	are resolved with the introduction of GCN. With Example2,
  1100	the magenta points, blue points, and cyan points are highly
  1101	proximate to one another in the embedding space without
  1102	GCN. On the contrary, they are well separated in the embedding space with GCN.
  1103	
  1104	6. Conclusion
  1105	In this paper, we propose the MPRD method to address
  1106	the unsupervised person Re-ID task. Unlike previous methods that estimate the pseudo labels through either iterative
  1107	
  1108	clustering or classification, it is unnecessary for our method
  1109	to determine the number of clusters in the training stage. The
  1110	proposed MPRD reformulates the unsupervised discriminative feature learning task into a pairwise relationship estimation problem. A GCN is used to estimate the pairwise relationship of sample pairs based on the graph structure among
  1111	the pairs’ neighbors. CNN learns the discriminative features
  1112	from input images according to these estimated pairwise
  1113	relationship labels. Extensive experiments on Market-1501,
  1114	DukeMTMC-reID, and MSMT17 datasets demonstrate the
  1115	effectiveness of the proposed method for the unsupervised

exec
/bin/zsh -lc "pdftotext -f 1 -l 12 'Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf' - | nl -ba | sed -n '886,1380p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   886	4. Experiments and analysis
   887	We evaluate the proposed memory-augmented shuffled meta
   888	(MASM) learning method on two publicly available visible–infrared
   889	person re-identification (VIPR) datasets, namely, RegDB (Nguyen et al.,
   890	2017) and SYSU-MM01 (Wu et al., 2017). The performance indicators
   891	are the cumulative matching characteristics (CMC) curve (Luo et al.,
   892	2020), the mean average precision (mAP) (Luo et al., 2019; Xie, Zhang,
   893	Xu, Zhu, & He, 2023; Zhu, Zeng et al., 2020), and the mean inverse
   894	negative penalty (mINP) (Ye, Shen et al., 2022; Zhu et al., 2023). The
   895	most important term in a cumulative matching characteristics (CMC)
   896	curve is the rank-1 identification rate, which is abbreviated as R1.
   897	4.1. Dataset
   898	RegDB (Nguyen et al., 2017) dataset is captured by visible and
   899	infrared cameras. Following (Huang et al., 2022; Ye et al., 2020; Ye,
   900	Shen and Shao, 2021), the result in RegDB is the average of 10 random
   901	trials. In each trial, the training subset randomly selects 206 subjects
   902	from RegDB, and the rest of RegDB is applied to construct a testing
   903	subset, and one subject has 10 visible images and 10 infrared images.
   904	As a result, in each trial, the training set contains 206 subjects of 2060
   905	visible images and 2060 infrared images. In the visible-to-infrared (V2I)
   906	retrieval mode, the query set has 206 subjects of 2060 visible images,
   907	and the gallery set includes 206 subjects of 2060 infrared images. In the
   908	infrared-to-visible (I2V) retrieval mode, the query set has 206 subjects
   909	of 2060 infrared images, and the gallery set includes 206 subjects of
   910	2060 visible images.
   911	SYSU-MM01 (Wu et al., 2017) dataset is a large-scale pedestrian
   912	dataset that is captured using four visible cameras and two infrared
   913	cameras in indoor and outdoor environments. There are two types of
   914	testing modes, called all-search testing mode and indoor-search testing
   915	mode. The all-search testing mode utilizes all visible cameras equipped
   916	indoors and outdoors to construct the gallery set, while the indoorsearch testing mode only uses the visible camera equipped indoors to
   917	construct the gallery set. Specifically, in the all-search testing mode,
   918	the probe set comprises 3803 infrared images of 96 subjects, and the
   919	gallery set comprises 30l visible images of the same 96 subjects. In the
   920	
   921	3.4. Complete VIPR architecture
   922	We use the two-branch ResNet50 (He, Zhang, Ren, & Sun, 2016) to
   923	construct the backbone, which is commonly used in many works (Liu
   924	et al., 2021; Ye, Chen et al., 2022; Ye et al., 2020; Ye, Shen et al.,
   925	2022). This backbone is widely adopted in both classical methods
   926	(e.g., Ye et al., 2020; Ye, Shen et al., 2022) and recent state-of-the-art
   927	approaches (e.g., Hua et al., 2025; Zhang, Du, Liu, Tu and Shu, 2025)
   928	for person re-identification. Using ResNet50 ensures fair and meaningful comparisons with prior work and allows us to isolate and highlight
   929	the contributions of our MASM framework without confounding factors
   930	introduced by different backbone designs. As shown in Fig. 2, among
   931	two branches, shallow layers, such as Stem, Layer1, and Layer2 layers
   932	have the same structure, but are independent to learn modal-specific
   933	features; deep layers, such as Layer3, Layer4, and generalized-mean
   934	pooling (GeP), and batch normalization (BN) layers have the same
   935	structure of the same parameters to learn modal-common features. In
   936	the testing phase, features of the GeP and BN layers are added to
   937	evaluate VIPR performance.
   938	6
   939	
   940	Neural Networks 191 (2025) 107812
   941	
   942	H. Wu et al.
   943	
   944	indoor-search testing mode, the same probe subset with 3803 infrared
   945	images of 96 subjects is used to search a different gallery consisting of
   946	112 visible images of the same 96 subjects.
   947	LLCM (Zhang & Wang, 2023) dataset utilizes 9 cameras deployed
   948	in low-light environments to capture the visible images in the daytime
   949	and infrared images at night. This dataset contains 46,767 images of
   950	1064 identities, encompassing various climate conditions and clothing
   951	styles. The training set contains 30,921 images of 713 identities, among
   952	them 16,946 images are from the visible modality and 13,975 images
   953	are from the infrared modality. The testing set contains 13,909 images
   954	of 351 identities, where 8680 images are from the visible modality
   955	and 7166 images are from the infrared modality. Both the V2I and
   956	I2V retrieval modes are used to evaluate the performance of the VIPR
   957	models. During the testing stage, for each camera, we randomly choose
   958	one image from the images of each identity to form the gallery set
   959	for evaluation the performance of the models. We repeat the above
   960	evaluation 10 times with random split of the gallery set and report the
   961	average performance.
   962	
   963	without the shuffling strategy. The ablation analyses are presented as
   964	follows.
   965	The combination of CE and CM (w/o SMR) significantly outperforms
   966	CE alone, improving Rank-1 accuracy (R1) from 59.0% to 88.0% on
   967	the RegDB (V2I retrieval) and from 56.5% to 67.1% on SYSU-MM01
   968	(all-search mode), underscoring the effectiveness of meta-learning. The
   969	inclusion of SMR further improves performance, with R1 rising from
   970	88.0% to 94.1% on RegDB, validating its role in stabilizing meta metric
   971	learning. The addition of MM yields incremental gains, such as a 1.6%
   972	increase in mINP on SYSU-MM01, highlighting the benefits of leveraging historical meta metrics. Finally, applying the shuffling strategy to
   973	both CM and MM maximizes overall performance, increasing R1 from
   974	89.3% to 94.1% on RegDB and from 63.6% to 68.6% on SYSU-MM01.
   975	These results clearly demonstrate the effectiveness of our proposed
   976	components and the designed shuffling strategy.
   977	4.4. Parameters sensitivity analysis
   978	In previous ablation studies, the effectiveness of CM and MM has
   979	been demonstrated, and in this section, we further analyze the parameters in CM and MM, as follows.
   980	
   981	4.2. Implementation details
   982	The software tools are Pytorch 1.7 (Paszke et al., 2019), CUDA 11.1,
   983	and python 3.8. The hardware device is one GeForce RTX 3090 GPU.
   984	We adopt Resnet-50 pre-trained on ImageNet as the backbone. The
   985	Adam (Kingma & Ba, 2015) is used for training with weight decays
   986	of 5 × 10−4 . Each mini-batch contains 4 classes, and each class consists
   987	of 5 visible images and 5 infrared images. There are 120 epochs for
   988	the training process. We use the warm-up (Luo et al., 2020) to adjust
   989	learning rates. When training the SYSU-MM01 and LLCM datasets, the
   990	learning rates are initialized at 2 × 10−5 and linearly warmed up to
   991	2 × 10−4 in the first 10 epochs. After warming up, the learning rates
   992	are maintained at 2 × 10−4 from the 11th to 30th epochs. Then, the
   993	initial learning rates are decayed by 10% every 20 epochs. That is, the
   994	learning rates are reduced to 2 × 10−5 between the 31st and 50th epochs,
   995	2 × 10−6 between the 51st and 70th epochs, and 2 × 10−7 between the
   996	71st and 90th epochs. Finally, learning rates are dropped to 2 × 10−8
   997	between the 91st and 120th epochs. When training the RegDB dataset,
   998	the learning rates are initialized at 6.5 × 10−5 and linearly warmed up
   999	to 6.5 × 10−4 in the first 10 epochs. After warming up, the learning
  1000	rates are maintained at 6.5 × 10−4 from the 11th to 30th epochs. Then,
  1001	the initial learning rates are decayed by 10% every 20 epochs. That is,
  1002	the learning rates are reduced to 6.5 × 10−5 between the 31st and 50th
  1003	epochs, 6.5 × 10−6 between the 51st and 70th epochs, and 6.5 × 10−7
  1004	between the 71st and 90th epochs. Finally, learning rates are dropped
  1005	to 6.5 × 10−8 between the 91st and 120th epochs. Person images are
  1006	resized to 144 × 288 pixels for training and testing, as done in Ye,
  1007	Chen et al. (2022), Ye et al. (2020), Ye, Shen et al. (2022) and Zhang,
  1008	Du et al. (2025). Then, following the famous baseline re-identification
  1009	methods (He et al., 2020; Luo et al., 2019; Ye, Shen et al., 2022), the
  1010	z-score normalization, random cropping, random erasing, and random
  1011	horizontal flip operations are combined for the data augmentation. The
  1012	probabilities of horizontal flip and random erasing operations are 0.5.
  1013	The Euclidean distance is applied as the similarity measure to rank
  1014	images for evaluating VIPR performance.
  1015	
  1016	4.4.1. Analysis of CM
  1017	The CM loss has two parameters. 𝛾 is used to adjust the modulating
  1018	factor for the balance of samples in the focal function; 𝜉 is used to
  1019	control the weight of SMR. We analyze the sensitivity of 𝛾 and 𝜉, as
  1020	follows.
  1021	• 𝛾 sensitivity. We set 𝜉 = 0 in Eq. (21) and 𝜆 = 0 in Eq. (10)
  1022	to eliminate the influence of the corresponding units in advance.
  1023	As shown in Fig. 3(a), on the RegDB dataset, when increasing 𝛾
  1024	from 1.0 to 2.0, R1 shows a trend of increasing first and then
  1025	decreasing. Specifically, when adjusting 𝛾 from 1.0 to 1.2, R1
  1026	increases from 87.5% to 88.0% in the visible-to-infrared retrieval
  1027	mode, and then when 𝛾 > 1.2, R1 decreases. On the SYSU-MM01
  1028	dataset, a similar trend can be found, as shown in Fig. 3(b).
  1029	The reason for this situation is that when 𝛾 is too small, the
  1030	model focuses more on easily classified samples and neglects
  1031	difficult ones, leading to a decrease in re-identification accuracy.
  1032	However, when 𝛾 is too large, the loss changes rapidly, which
  1033	makes optimization challenging.
  1034	• 𝜉 sensitivity. We set 𝜆 0 to exclude the impact of the MM loss
  1035	function and fix 𝛾 to 1.2 according to the analysis of 𝛾 sensitivity.
  1036	On the RegDB dataset, as shown in Fig. 4(a), R1 increases with
  1037	the increase of 𝜉. Specifically, in the V2I retrieval mode, R1
  1038	increases from 91.7% to 94.1% as 𝜉 increases from 0.5 to 2.0.
  1039	In the I2V retrieval mode, R1 increases from 89.9% to 93.0%
  1040	as 𝜉 increases from 0.5 to 2.5. On the SYSU-MM01 dataset, as
  1041	shown in Fig. 4(b), R1 achieves the best result at 𝜉 = 0.3, with
  1042	67.4% in the all-search testing mode and 73.3% in the indoorsearch testing mode. Therefore, comparing Figs. 4(a) and 4(b), it
  1043	can be observed that the optimal value of 𝜉 on the SYSU-MM01
  1044	dataset is smaller than that on the RegDB dataset. The reason
  1045	for this is that the SYSU-MM01 dataset has a larger scale and
  1046	a more complex appearance variation. Hence, a larger 𝜉 value
  1047	would cause the model to focus more on restricting consistency
  1048	in different retrieval modes and different shuffles, but neglect
  1049	discriminate learning through meta learning.
  1050	
  1051	4.3. Ablation studies
  1052	To straightforwardly evaluate the role of units in the proposed
  1053	memory-augmented shuffled meta (MASM) learning method, we implement ablation studies on RegDB and SYSU-MM01 datasets, as shown in
  1054	Tables 1 and 2, respectively. CE denotes the cross-entropy loss used for
  1055	training (Eq. (23)); CM refers to the current meta loss (Eq. (21) with
  1056	𝜉 = 2.0 on RegDB and 𝜉 = 0.3 on SYSU-MM01); CM (w/o SMR) denotes
  1057	CM without shuffled mate regularization (i.e., Eq. (21) with 𝜉 = 0); CM
  1058	(w/o Shuffling) refers to CM without the shuffling strategy; MM denotes
  1059	the memory meta loss (Eq. (8)); and MM (w/o Shuffling) denotes MM
  1060	
  1061	4.4.2. Analysis of MM
  1062	The MM loss function contains two parameters. 𝐿 represents the
  1063	memory bank capacity, that is, the memory bank can store 𝐿 batch
  1064	subsets; 𝜆 is the weight of 𝐿MM . To control variables, according to
  1065	above CM analysis, 𝛾 is fixed to 1.2 on both RegDB and SYSU-MM01
  1066	datasets, and 𝜉 is fixed to 2.0 on the RegDB dataset and 0.3 on the
  1067	SYSU-MM01 dataset. We analyze the sensitivity of 𝐿 and 𝜆 as follows.
  1068	7
  1069	
  1070	Neural Networks 191 (2025) 107812
  1071	
  1072	H. Wu et al.
  1073	Table 1
  1074	Ablation studies on the RegDB dataset to evaluate the role of CM, SMR, MM and the shuffling strategy (SS).
  1075	Loss functions
  1076	
  1077	Component
  1078	
  1079	CE
  1080	CE + CM (w/o SMR)
  1081	CE + CM
  1082	CE + CM + MM
  1083	CE + CM + MM (w/o Shuffling)
  1084	CE + CM (w/o Shuffling) + MM (w/o Shuffling)
  1085	
  1086	V2I
  1087	
  1088	I2V
  1089	
  1090	CE
  1091	
  1092	CM
  1093	
  1094	SS(CM)
  1095	
  1096	SMR
  1097	
  1098	MM
  1099	
  1100	SS(MM)
  1101	
  1102	R1
  1103	(%)
  1104	
  1105	mAP
  1106	(%)
  1107	
  1108	mINP
  1109	(%)
  1110	
  1111	R1
  1112	(%)
  1113	
  1114	mAP
  1115	(%)
  1116	
  1117	mINP
  1118	(%)
  1119	
  1120	!
  1121	!
  1122	!
  1123	!
  1124	!
  1125	!
  1126	
  1127	✗
  1128	!
  1129	!
  1130	!
  1131	!
  1132	!
  1133	
  1134	✗
  1135	!
  1136	!
  1137	!
  1138	!
  1139	✗
  1140	
  1141	✗
  1142	✗
  1143	!
  1144	!
  1145	!
  1146	!
  1147	
  1148	✗
  1149	✗
  1150	✗
  1151	!
  1152	!
  1153	!
  1154	
  1155	✗
  1156	✗
  1157	✗
  1158	!
  1159	✗
  1160	✗
  1161	
  1162	59.0
  1163	88.0
  1164	94.1
  1165	94.1
  1166	93.9
  1167	89.3
  1168	
  1169	52.5
  1170	82.1
  1171	88.3
  1172	88.6
  1173	88.2
  1174	84.3
  1175	
  1176	37.4
  1177	69.1
  1178	77.1
  1179	77.6
  1180	77.0
  1181	73.0
  1182	
  1183	55.3
  1184	86.1
  1185	92.5
  1186	92.8
  1187	92.8
  1188	88.3
  1189	
  1190	48.7
  1191	80.3
  1192	86.9
  1193	87.2
  1194	87.1
  1195	83.1
  1196	
  1197	33.2
  1198	65.5
  1199	73.7
  1200	74.3
  1201	74.0
  1202	69.8
  1203	
  1204	Table 2
  1205	Ablation studies on the SYSU-MM01 dataset to evaluate the role of CM, SMR, MM and the shuffling strategy (SS).
  1206	Loss functions
  1207	
  1208	Component
  1209	
  1210	CE
  1211	CE + CM (w/o SMR)
  1212	CE + CM
  1213	CE + CM + MM
  1214	CE + CM + MM (w/o Shuffling)
  1215	CE + CM (w/o Shuffling) + MM (w/o Shuffling)
  1216	
  1217	All-search
  1218	
  1219	Indoor-Search
  1220	
  1221	CE
  1222	
  1223	CM
  1224	
  1225	SS(CM)
  1226	
  1227	SMR
  1228	
  1229	MM
  1230	
  1231	SS(MM)
  1232	
  1233	R1
  1234	(%)
  1235	
  1236	mAP
  1237	(%)
  1238	
  1239	mINP
  1240	(%)
  1241	
  1242	R1
  1243	(%)
  1244	
  1245	mAP
  1246	(%)
  1247	
  1248	mINP
  1249	(%)
  1250	
  1251	!
  1252	!
  1253	!
  1254	!
  1255	!
  1256	!
  1257	
  1258	✗
  1259	!
  1260	!
  1261	!
  1262	!
  1263	!
  1264	
  1265	✗
  1266	!
  1267	!
  1268	!
  1269	!
  1270	✗
  1271	
  1272	✗
  1273	✗
  1274	!
  1275	!
  1276	!
  1277	!
  1278	
  1279	✗
  1280	✗
  1281	✗
  1282	!
  1283	!
  1284	!
  1285	
  1286	✗
  1287	✗
  1288	✗
  1289	!
  1290	✗
  1291	✗
  1292	
  1293	56.5
  1294	67.1
  1295	67.4
  1296	68.6
  1297	67.3
  1298	63.6
  1299	
  1300	53.1
  1301	64.2
  1302	64.1
  1303	65.5
  1304	65.5
  1305	60.6
  1306	
  1307	37.9
  1308	50.7
  1309	50.3
  1310	51.9
  1311	51.9
  1312	46.6
  1313	
  1314	61.3
  1315	72.1
  1316	73.0
  1317	72.9
  1318	72.9
  1319	66.8
  1320	
  1321	67.0
  1322	76.5
  1323	77.1
  1324	77.1
  1325	77.1
  1326	72.3
  1327	
  1328	62.1
  1329	72.2
  1330	72.9
  1331	73.0
  1332	73.0
  1333	68.2
  1334	
  1335	Fig. 3. The result of using different values of 𝛾 in Eq. (9) on (a) RegDB and (b) SYSU-MM01 dataset.
  1336	
  1337	Fig. 4. The result of using different values of 𝜉 in Eq. (21) on (a) RegDB and (b) SYSU-MM01 datasets.
  1338	
  1339	• 𝐿 sensitivity. As shown in the ablation study, MM plays an
  1340	auxiliary role, and we empirically set its weight to 0.01. Then,
  1341	we change 𝐿 to observe performance variations. As illustrated in
  1342	Fig. 5, the optimal memory bank capacities vary across different
  1343	datasets. More specifically, Fig. 5(a) shows that the highest values
  1344	of R1 and mAP are achieved at 𝐿 = 60, whereas Fig. 5(b) indicates
  1345	that the highest values of R1 and mAP are obtained at 𝐿 =
  1346	
  1347	40. This observation suggests that smaller datasets necessitate a
  1348	greater dependence on historical data in order to fully utilize the
  1349	available information.
  1350	• 𝜆 sensitivity. According to the analysis of 𝐿 sensitivity, we
  1351	remain 𝐿 at its optimal value (i.e., 𝐿 = 60 on the RegDB dataset
  1352	and 𝐿 = 40 on the SYSU-MM01 dataset) to further analyze 𝜆
  1353	sensitivity. As shown in Fig. 6, the performance changes on the
  1354	8
  1355	
  1356	Neural Networks 191 (2025) 107812
  1357	
  1358	H. Wu et al.
  1359	
  1360	Fig. 5. The results of using different memory bank capacities (i.e., 𝐿) (a) RegDB in the visible-to-infrared retrieval mode and (b) on SYSU-MM01 in the all-search testing mode.
  1361	
  1362	Fig. 6. The results of using different weights of 𝐿MM (i.e., 𝜆 in Eq. (9)) on (a) RegDB in the visible-to-infrared retrieval mode and on (b) SYSU-MM01 in the all-search testing
  1363	mode.
  1364	
  1365	SYSU-MM01 dataset are more significant than the performance
  1366	changes on the RegDB dataset when adjusting 𝜆. Specifically,
  1367	when changing 𝜆 from 0.01 to 0.5, mAP on the RegDB dataset
  1368	only decreases by 0.3% as shown in Fig. 6(a), but mAP on the
  1369	SYSU-MM01 dataset decreases by 2.7% as shown in Fig. 6(b).
  1370	We speculate that this is because the SYSU-MM01 dataset has
  1371	more complex data variations stored in the memory banks, so the
  1372	performance on the SYSU-MM01 dataset is more sensitive to 𝜆.
  1373	
  1374	retrieval of identity 0015, MASM retrieves more correct images, despite
  1375	significant appearance variations caused by lighting conditions. This
  1376	example illustrates that the model relies not only on local visual similarities but also captures more abstract, global relationships. This ability
  1377	originates from MML’s use of historical information, which enables
  1378	it to learn a meta-level similarity metric generalizable across diverse
  1379	contexts. Moreover, when combined with SML’s varied query-support
  1380	configurations, MASM is encouraged to develop comprehensive and

codex
这几篇的“证据闭环”有明显差异：有的靠消融表，有的靠新协议下性能坍塌，有的靠可视化。为了判断 reviewer 买的是视角还是模块，我还需要补读 Mix-Modality 的混合比例分析和 MoDA 的参数效率消融。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 18 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | rg -n -i 'mixing ratio|modality confusion|ablation|hyperparameter|visualization|t-SNE|MBSOS|CIDHL|\\+M|\\+C|Rank-1|mINP|performance degradation'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
16:    16	infrared modalities. However, significant performance degradation occurs due to the modality confusion
18:    18	Mix-Modality Person Re-Identification (MM-ReID) task, explores the influence of modality mixing ratio
20:    20	testing paradigm. To solve the modality confusion problem in MM-ReID, we propose a Cross-Identity
21:    21	Discrimination Harmonization Loss (CIDHL) adjusting the distribution of samples in the hyperspherical
24:    24	Furthermore, we propose a Modality Bridge Similarity Optimization Strategy (MBSOS) to optimize
53:    53	existing cross-modality methods on MM-ReID, the addition of our CIDHL and MBSOS demonstrates a general
86:    86	in Figure 1(c), this approach results in a marked performance degradation, primarily due to the
87:    87	“Modality Confusion” problem. This issue stems from identity-independent features such as colors
89:    89	identity information. More specifically, the impact of modality confusion on the current approach
117:   117	To address the modality confusion challenge in MM-ReID, we introduce the Cross-Identity
118:   118	Discrimination Harmonization Loss (CIDHL). This novel approach optimizes the sample
119:   119	distribution within a hyperspherical feature space. Specifically, CIDHL aggregates distances between
124:   124	(MBSOS), which optimizes the cross-modality distance metric by identifying a similar bridge
126:   126	the query and the queried samples. Extensive testing confirms that our methods—CIDHL and
127:   127	MBSOS—significantly enhance the model performance of existing cross-modality approaches,
128:   128	demonstrating their efficacy and adaptability in resolving modality confusion in MM-ReID.
132:   132	— Empirical Contribution: We investigate the impact of varying modality mixing ratios on crossmodality model performance and conclude the unique challenges of “Modality Confusion”
140:   140	— Technical Contribution: We introduce the CIDHL and the MBSOS. These transferable methodologies effectively address modality confusion, as validated by extensive empirical experiments.
196:   196	addition to identity differences; and (c) MM-ReID needs to face the challenge of modality confusions (similar
206:   206	methodologies: the CIDHL and the MBSOS. More specifically, Section 3.1 provides the task definition
210:   210	the design idea and the detailed composition of the CIDHL loss, and Section 3.4 gives a detailed
211:   211	description of the steps and algorithmic procedure of the MBSOS strategy.
229:   229	contends with “modality confusion,” where similarities irrelevant to identity, such as color consistency within a modality, can obscure vital identity-specific cues between the query (Q) and gallery
237:   237	Fig. 3. An illustration of our proposed CIDHL and MBSOS. The mix-modality data is fed into two feature
238:   238	extractors with shared weights to extract the features, and under the constraints of CIDHL, the distance
242:   242	During the testing process, the extracted features are optimized by MBSOS to get the final shortest path
249:   249	To tackle the distinct modality confusion problem in the MM-ReID task, this article introduces
250:   250	two innovative methods: CIDHL and MBSOS. CIDHL specifically targets the confusion arising
254:   254	to mitigate modality confusion effects during the testing phase, MBSOS optimizes the similarity
262:   262	CIDHL
264:   264	3.3.1 Feature Presentation Extractor. Our proposed methods, CIDHL and MBSOS, are designed to
270:   270	CIDHL (𝐿CIDHL ), optimizing the feature extraction process and ensuring robust performance across
373:   373	Metric Learning in CIDHL: Enhancing Identity Discrimination across Modalities. The fundamental
374:   374	principle of metric learning in CIDHL revolves around optimizing identity discrimination by
386:   386	The CIDHL adopts a structured approach, containing two parts, the first of which is targeted:
433:   433	center of samples with the same modality and identity for solving the modality confusion problem
453:   453	The final CIDHL 𝐿𝐶𝐼 𝐷𝐻 𝐿 can be calculated as:
465:   465	MBSOS
467:   467	The MBSOS addresses the modality confusion issue by innovatively transforming the distance
479:   479	The core objective of MBSOS is to enhance the robustness of cross-modality Re-ID by minimizing
495:   495	each edge E represents the distance between two vertices. Considering that modality confusion is
528:   528	Algorithm 1: MBSOS
558:   558	the help of any intermediate bridge node. The algorithmic procedure of MBSOS is described in
566:   566	existing methods and the performance with the addition of our CIDHL and MBSOS are compared.
567:   567	In Section 4.3, we do ablation experiments on the hyperparameters 𝛿 and 𝜆. Finally, in Section 4.4,
568:   568	we show the visualization results of the AGW algorithm with the addition of our CIDHL after the
569:   569	t-distributed Stochastic Neighbor Embedding (t-SNE) dimensionality reduction.
628:   628	4.1.2 Metrics. The cumulative matching characteristics [23] at Rank-1, Rank-5, Rank-10, mean
629:   629	average precision (mAP) and mean inverse negative penalty (mINP) [39] are adopted as
633:   633	respectively, according to the modality mixing ratio. For example, if the modality mixing ratio is
637:   637	model on the RegDB, SYSU-MM01, and LLCM datasets in the Rank-1, mAP, and mINP metrics.
638:   638	It can be seen that the performance of the model decreases in all other metrics and mixing ratios,
639:   639	except for the Rank-1 metric, which shows an improvement in the RegDB and SYSU-MM01 datasets.
642:   642	to match and that the decrease in the other metrics is due to modality confusion that leads to a
646:   646	in this case, in the large-scale dataset, the model’s mINP metrics appear to be greatly degraded
657:   657	Fig. 4. The effect of different modality mixing ratios with respect to the AGW model performance in Rank-1,
658:   658	mAP, and mINP metrics on the three datasets. It can be seen that a general performance degradation arises
659:   659	on the other metrics except for the Rank-1 metrics that rise on some of the datasets. The origin stands for the
661:   661	and infrared images, and so forth. mAP, mean average precision; mINP, mean inverse negative penalty.
671:   671	RegDB, SYSU-MM01, and LLCM in Rank-1, Rank-5, Rank-10, mAP, and mINP. Where Mode 1
675:   675	The “+M,” “+C,” and “+C+M” denote the performance of the method added MBSOS, CIDHL, and
678:   678	occurs with the addition of our method. In Mode 1, the AGW algorithm achieves the best performance with the addition of CIDHL, showing “+6.09%,” “+2.19%,” “+1.01%,” “+8.84%,” and “+8.21%”
679:   679	improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. The addition of
680:   680	MBSOS also shows a performance improvement of “+3.67%” and “+11.89%” in the mAP and mINP
682:   682	metric. In Mode 2, the AGW algorithm with the addition of both CIDHL and MBSOS achieves the
684:   684	in Rank-1, Rank-5, Rank-10, mAP, and mINIP metrics, respectively. Unlike Model 1 the performance
685:   685	of the model with the addition of MBSOS improves on the Rank metric. The performance of other
715:   715	Rank-1
754:   754	Rank-10
792:   792	mINP
835:   835	Rank-1
874:   874	Rank-10
912:   912	mINP
934:   934	achieves the best performance with the addition of both CIDHL and MBSOS, showing “+18.86%,”
935:   935	“+5.76%,” “+2.98%,” “+13.76%,” and “+1.33%” improvement in Rank-1, Rank-5, Rank-10, mAP, and
936:   936	mINP metrics, respectively. Adding CIDHL and MBSOS individually also shows performance improvements in most metrics. In Mode 2, the AGW algorithm with the addition of both CIDHL and
965:   965	Rank-1
1008:  1008	Rank-10
1050:  1050	mINP
1097:  1097	Rank-1
1140:  1140	Rank-10
1182:  1182	mINP
1231:  1231	Rank-1
1266:  1266	Rank-10
1300:  1300	mINP
1339:  1339	Rank-1
1374:  1374	Rank-10
1408:  1408	mINP
1428:  1428	MBSOS also achieves the best performance, showing “+17.72%,” “+4.36%,” “+1.95%,” “+16.24%,” and
1429:  1429	“+2.67%” improvements in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. Adding
1430:  1430	CIDHL and MBSOS individually also shows performance improvements in most metrics, too. The
1434:  1434	the best performance with the addition of both CIDHL and MBSOS, showing “+8.36%,” “+6.97%,”
1440:  1440	Table 8. Ablation study of 𝛿
1451:  1451	Rank-1
1470:  1470	Rank-10 mAP
1486:  1486	mINP
1497:  1497	Table 9. Ablation Study of 𝛿
1508:  1508	Rank-1
1527:  1527	Rank-10 mAP
1543:  1543	mINP
1554:  1554	“+6.29%” “+5.98%,” and “+0.31%” improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics,
1555:  1555	respectively. Adding CIDHL and MBSOS individually also shows performance improvements in
1556:  1556	most metrics. In Mode 2, the AGW algorithm with the addition of both CIDHL and MBSOS
1558:  1558	improvements in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. Adding CIDHL and
1559:  1559	MBSOS individually also shows performance improvements in most metrics, too. The performance
1563:  1563	Ablation Study
1565:  1565	We adopted the AGW method to perform our ablation experiments on the SYSU-MM01 dataset in
1566:  1566	Rank-1, Rank-5, Rank-10, and mAP metrics. We show the ablation experiments for the 𝛿 parameter
1569:  1569	As can be seen in Tables 8 and 9, in Mode 1, compared with the performance without our CIDHL,
1571:  1571	in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics (𝛿 is set to 0.3), and at most “+18.54%,” “+5.60%,”
1573:  1573	our CIDHL is added (𝛿 is set to 0.2). In Model 2, compared to the performance without our CIDHL,
1576:  1576	improvement (𝛿 is set to 0.2) in the four metrics of Rank-1, Rank-5, Rank-10, and mAP, respectively,
1583:  1583	Table 10. Ablation study of 𝜆
1594:  1594	Rank-1
1613:  1613	Rank-10
1622:  1622	mAP mINP
1633:  1633	Table 11. Ablation Study of 𝜆
1644:  1644	Rank-1
1663:  1663	Rank-10 mAP
1679:  1679	mINP
1690:  1690	with the addition of our CIDHL. Besides, in Mode 1, compared to only 𝐿𝑐𝑖𝑑 used (𝛿 is set to 0), the
1692:  1692	and “+2.14%” improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. In
1694:  1694	showing “+30.60%,” “+15.99%,” “+9.57%,” “+43.24%,” and “+4.00%” improvement in Rank-1, Rank-5,
1695:  1695	Rank-10, mAP, and mINP metrics, respectively.
1696:  1696	As can be seen in Tables 10 and 11, both in Mode 1 and Mode 2, the addition of our MBSOS
1701:  1701	Visualization
1703:  1703	As illustrated in Figure 5, we compared the t-SNE visualization results of the original AGW and the
1704:  1704	AGW with the addition of our CIDHL loss. After adding our CIDHL, it is clear from Figure 5(a)
1708:  1708	in the comparison experiments is a good proof of its effectiveness in solving the modality confusion
1716:  1716	Fig. 5. The t-SNE Visualization of only AGW (first row) and AGW addition with our CIDHL (second row).
1728:  1728	paradigm, we propose an MM-ReID task and explore the effect of different modality mixing ratios
1732:  1732	this problem as the modality confusion problem. To address this modality confusion problem, we
1733:  1733	propose a CIDHL and an MBSOS. The former pulls the centers of samples with the same identity
1738:  1738	transferability and ability to cope with modality confusion of our methods. The code and datasets

exec
/bin/zsh -lc "pdftotext -f 1 -l 18 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | rg -n -i 'ablation|parameter|adapter|global adapter|voting|ID loss|frozen|tunable|visual|baseline|Table 8|Protocol|performance|efficiency'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:     1	MoDA: Mixture of Domain Adapters for Parameter-efficient
12:    12	amount of parameters, which are not always practical in real-world scenarios. Considering this problem, we
13:    13	propose a novel MoE-based DG ReID method, named Mixture of Domain Adapters (MoDA), which utilizes
14:    14	many expert adapters and a global adapter to help MoE-based method scale to a much larger model but in a
15:    15	more parameter-efficient way. Furthermore, we conduct our approach with the large-scale vision-language
16:    16	pre-trained model CLIP, which exploits both visual and text encoders, to learn more robust representations
18:    18	MoDA achieves competitiveness with state-of-the-art DG ReID methods with much fewer tunable parameters.
19:    19	CCS Concepts: • Computing methodologies → Computer vision; Visual content-based indexing and
21:    21	Additional Key Words and Phrases: Generalizable Person Re-Identification, Domain Generalization, Parameterefficient Fine-tuning
23:    23	Yang Wang, Yixing Zhang, Xudie Ren, and Yuxin Deng. 2025. MoDA: Mixture of Domain Adapters for
24:    24	Parameter-efficient Generalizable Person Re-identification. ACM Trans. Multimedia Comput. Commun. Appl.
55:    55	There have been many works that can improve great performance on ReID benchmarks in the conventional scenario [8, 29, 32, 48]. However, when these methods are confronted with a completely
56:    56	unseen domain, the performance drops significantly. This phenomenon is commonly attributed
69:    69	consists of samples from different source domains. And then the model is directly tested on unseen target domains. These methods achieve good performance by extracting domain-invariant
75:    75	MoE-based DG ReID methods get better performance but have the common issue that the number
76:    76	of model parameters scales linearly with the number of source domains due to the increase of the
78:    78	minimize the number of expert parameters, there are still a large amount of trainable parameters
81:    81	large, the number of parameters of classifier heads significantly increases. These MoE-based methods always need at least one classifier for each source domain and one global classifier to learn
82:    82	domain-invariant feature and to optimize the model. And it leads to a certain degree of parameter
85:    85	which is parameter-efficient and can be scaled to a much larger backbone model, named Mixture
86:    86	of Domain Adapters (MoDA). We apply the large-scale vision-language model CLIP [27] and
87:    87	Adapter [10] in our method. CLIP is a powerful pretrained model which aligns texts and images
91:    91	loss, which means that it may substitute the ID loss of ReID to reduce the classifier head parameters.
92:    92	Some works [30] with the image encoder of CLIP have already shown good performance in other
93:    93	computer vision DG tasks. In addition, Adapter is a bottleneck module used for Parameterefficient Fine-tuning (PEFT) and will be used as domain-specific experts in MoDA. As Figure 2
94:    94	shows, different from prior CNN-based DG ReID methods, the adapters (experts) are inserted in
95:    95	each block of MoDA. Due to this architecture, we propose a Block-aware Voting Network to make
98:    98	MoDA for Parameter-efficient Generalizable Person Re-ID
102:   102	Fig. 1. Difference between prior MoE-based DG ReID methods and our more parameter-efficient method.
104:   104	parameters of which usually increase linearly with the increasing of the number of person IDs. Below: In our
105:   105	method, the backbone is frozen and only adapters, the parameters of which are not related with the person
108:   108	Fig. 2. We show the architecture of the adapter, ViT block with adapter, and our method MoDA block. The
109:   109	standard architectures of adapter and ViT block are shown within (a) and (b), respectively. In our proposed
110:   110	MoDA block, we set Adapters after the multi-head attention layer to implement an MoE architecture as (c)
111:   111	shows. An additional block-aware voting network is set to mix the adapters. Only the parameters of adapters
112:   112	and the voting network need to be updated during training.
115:   115	integrate expert adapters in a more fine-grained way by generating aggregation weight for each
117:   117	parameters in the model network.
129:   129	— Exploration of PEFT for DG ReID. We are also the first to investigate the performance of PEFT
130:   130	methods, specifically Adapters, within the context of DG ReID. Adapters serve as domainspecific experts in our framework, demonstrating their efficacy in reducing the number of
131:   131	tunable parameters while maintaining performance.
132:   132	— Introduction of MoDA and Block-aware Voting Network. Based on the exploration of CLIP
134:   134	parameter efficiency while scaling MoE-based DG ReID methods to larger vision models.
135:   135	Our framework introduces a cross-modality Block-aware Voting Network, which enables
136:   136	fine-grained integration of expert adapters across different blocks, optimizing the model’s
137:   137	performance.
139:   139	not only achieves competitive performance but also surpasses existing state-of-the-art (SOTA)
140:   140	methods in a more parameter-efficient manner. Our results underscore the effectiveness of
146:   146	DG Person ReID. Most of the existing ReID methods suffer from significant performance degradation
168:   168	the existing methods need to fully fine-tune the whole models with a large amount of parameters
172:   172	MoDA for Parameter-efficient Generalizable Person Re-ID
176:   176	Vision-language Pre-trained Model. Recently, more and more studies on Vision-language Pretraining (VLP) have shown that VLP can significantly improve the performance of many vision
185:   185	tasks. And we will explore the performance of CLIP in the DG ReID context based on CLIP-ReID.
186:   186	PEFT. Conventional approaches usually fine-tune all the parameters (full fine-tuning) of pretrained models. Due to the significantly increasing overhead of full fine-tuning VLP models, more
187:   187	and more PEFT methods are proposed. These methods only fine-tune a small number of parameters
188:   188	and freeze most of them. They aim to reduce model computation and the number of tunable
189:   189	parameters as much as possible with sacrificing a little bit performance or even outperforming
193:   193	during training. LoRA [11] tries to learn low-rank matrices to approximate parameter updates.
194:   194	Authors in [10] insert small bottleneck modules named adapters to each layer of the pretrained
195:   195	model and only fine-tune the adapters during training. Recently, parameter-efficient methods are
196:   196	also studied for computer vision tasks. Authors in [13] propose visual prompt tuning for large-scale
199:   199	uses adapters to adapt more image models to video tasks without full fine-tuning.
201:   201	on models with adapters, such as AdaMix [33] and MixDA [3], they actually employ distinct
202:   202	FC-Down and FC-Up layers within one single adapter structure and primarily utilize a randomly
203:   203	routing mechanism or simply generate weights to choose and integrate adapters. While we use a
204:   204	block-aware voting network by calculating the relevance with cross-modal features. AdaMix is
223:   223	[𝑋 ]𝑚 are optimized while the encoders are frozen. The loss L𝑖2𝑡 is formulated as, specifically,
231:   231	The two encoders are frozen and only learnable tokens are optimized in this stage. Person ID-specific tokens
233:   233	be used in the Stage 2 for the Voting Network to calculate relevance. Stage 2. For each batch, we randomly
236:   236	domain. The expert branch (a) and global branch (b) are used to optimize the expert adapter and the global
237:   237	adapter, respectively, and the branch (c) is used for the voting network. In the whole procedure, only the
238:   238	parameters of the expert adapters, the global adapter and the voting network are tunable. While all other
239:   239	parameters are frozen.
276:   276	In the second stage, we propose an MoE model with expert adapters and global adapters, which
282:   282	MoDA for Parameter-efficient Generalizable Person Re-ID
284:   284	CLIP-ReID, we adopt adapter [10] due to its simplicity and parameter-efficiency rather than full
286:   286	abundant parameters in ViT as well as the scalability limitation of MoE-based approaches on
287:   287	larger models. In addition, adapters also help prevent catastrophic forgetting from insufficient
289:   289	Figure 2(a) shows, adapter is a simple bottleneck module with two FC layers and one activation
297:   297	Domain-specific Adapters
299:   299	As well as the prior MoE-based methods for DG ReID, our method assigns an individual adapter as
300:   300	the domain expert of each source domain, denoted as 𝐴𝑘 standing for the domain-specific adapter
304:   304	In every ViT block, we set K adapters after the multi-head attention layer as Figure 3 shows.
305:   305	Expert Adapter. For a kth domain image from the nth ID, we denote it as 𝑥𝑛𝑘 . As Figure 3(a) shows,
306:   306	for each 𝑥𝑛𝑘 , we let it go through ViT blocks with only the corresponding domain adapter 𝐴𝑘 to
339:   339	Notably, we find that with adapters, the usage of ID loss L𝑖𝑑 has little influence on model performance, which will be discussed in Section 4.4 and shown in Table 8. Intuitively, we speculate the
340:   340	reason might be that the image-to-text loss L𝑖2𝑡 and ID loss L𝑖𝑑 proposed by CLIP are both inherently cross-entropy losses, thus can provide approximate constraints and optimization capabilities.
341:   341	However, the additional classifier head leads to substantially more tunable parameters under the
342:   342	DG ReID experiment configuration. Therefore, to additionally enhance the parameter-efficiency of
343:   343	our model, we discard the ID loss which is widely used in ReID tasks.
344:   344	Triplet loss is still employed for expert adapter in our methodology:
350:   350	Meanwhile, 𝛼 denotes the margin parameter of the triplet loss due to [25].
357:   357	Thus, the loss for expert adapter is formulated as:
366:   366	where the superscript e which stands for expert is to distinguish from the losses for global adapter
368:   368	Global Adapter. Except for the domain-specific experts, we also maintain an individual global
369:   369	adapter to learn a common feature space to extract domain-invariant representations. Inspired by
370:   370	META [36], we similarly utilize a global adapter as the global branch to extract domain-invariant
372:   372	to any individual domain adapter. As Figure 3 Stage 2(b) branch shows, for each sample x from
374:   374	any domain, we pass it through the global adapter to obtain global feature 𝐹𝑥 . The computation of
375:   375	each global adapter block is actually the same as the expert adapter blocks. But we will combine all
376:   376	text features from different domains together to perform L𝑖2𝑡𝑐𝑒 . And the loss for global adapter is
386:   386	Notably, during the training process, the parameters of the ViT backbone are frozen and shared
387:   387	across different adapters, only the parameters of adapters need to be optimized.
394:   394	novel MoE-based approach specially for ViT with adapters to integrate domain-specific features
399:   399	mixed to create a new aggregated feature. In prior CNN-based approaches, the standard voting
403:   403	k times. This is due to our domain adapters are positioned within each block (layer) of the model
405:   405	we propose MoDA and a cross-modality block-aware voting network to enable the model to mix
416:   416	pass to blocks with the kth domain-specific adapters 𝐴𝑘 , then compute the expert loss L𝑒𝑥𝑝𝑒𝑟𝑡 .
424:   424	MoDA for Parameter-efficient Generalizable Person Re-ID
429:   429	𝑥 𝑘 is passed to the MoDA blocks only with the remaining 𝐾 − 1 expert adapters {𝐴𝑖 }𝑖=1,𝑖≠𝑘
434:   434	pass the output of MA 𝑧𝑙0 to all 𝐾 − 1 expert adapters concurrently. Then we can get 𝐾 − 1 diverse
439:   439	token of 𝑧𝑙 will be used for the voting network to generate a block-aware query feature 𝑞𝑙 . The
440:   440	block-aware voting network is simply implemented with a MLP architecture, and its parameters
480:   480	And the following computation is exactly the same as the common ViT block with adapters:
491:   491	In addition, we find that treating the global adapter as another domain-specific adapter and
492:   492	incorporating it into the mixture can bring better performance. And the weight of intermediate
494:   494	feature of global adapter ℎ𝑙
495:   495	is fixed to 0.5, rather than being generated by the voting network.
499:   499	To optimize the block-aware voting network and MoDA blocks, the following loss functions are
503:   503	adapters. The objective is to ensure the aggregated feature 𝐹𝑥 to retain inherent discriminative
561:   561	Algorithm 1 shows the overall training procedure. During test time, all K domain adapters will
562:   562	be used for voting network to produce the aggregated feature.
578:   578	MoDA for Parameter-efficient Generalizable Person Re-ID
628:   628	Table 2. The Details of Evaluation Protocols
630:   630	Protocols
631:   631	Protocol-1
632:   632	Protocol-2
633:   633	Protocol-3
654:   654	Evaluation Settings. The Mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) are used for evaluation. There are three testing protocols like the prior works
655:   655	[2, 36, 39] to evaluate the performance extensively.
656:   656	We show the details of the three evaluation protocols adopted in our experiments in Table 2. For
657:   657	protocol-1, model is trained with both the train and test images in M+C2+C3+CS datasets (denoted
660:   660	sets. The average of results will be reported. For protocol-2 and protocol-3 we choose one domain
661:   661	from M+MS+CS+C3 for testing and the remaining three domains for training. Protocol-2 only
663:   663	while protocol-3 uses both training and testing data of source domains for training (denoted as
671:   671	768 dimensions. The parameters of both the image encoder and the text encoder are frozen. The
672:   672	block voting network is implemented with a FC1-GELU-LN-FC2 architecture, where FC1 layer
679:   679	Table 3. Comparison with SOTA Methods Under Protocol-1
702:   702	Baseline (ViT)
703:   703	Baseline (CLIP)
784:   784	# Tunable
799:   799	All the images in the source domains are used for training. Bold type indicates the best performance and underline
800:   800	denotes the second position in performance ranking among methods in comparison.
801:   801	a The number of tunable parameters is estimated based on the classification head parameters under the current protocol,
807:   807	performance. We conduct all the experiments with PyTorch and the help of codebases: TransReID
816:   816	0.1 at the 30th and 50th epochs. Only the adapters are optimized in this stage. The weight 𝜆 of L𝑚𝑠𝑒
822:   822	Comparison under Protocol-1. We compare our method with previous DG-ReID methods under
823:   823	Protocol-1, which are tested on four small datasets (PRID, GRID, VIPeR, and iLIDs). As shown in
825:   825	more parameter-efficient way by fine-tuning much fewer parameters. And on average our method
829:   829	and demonstrate the effectiveness of our method, we have implemented two baseline methods
833:   833	Comparison under Protocol-2 and Protocol-3. We also compare our method with other methods
834:   834	under protocol-2 and protocol-3, as shown in Tables 4 and 5. Our method still maintains competitive
835:   835	performance within the top three, but it does not stand out as prominently as under protocol-1.
841:   841	MoDA for Parameter-efficient Generalizable Person Re-ID
842:   842	Table 4. Comparison with SOTA Methods under Protocol-2
845:   845	Protocol-2
857:   857	Baseline (ViT)
858:   858	Baseline (CLIP)
959:   959	# Tunable
971:   971	Bold type indicates the best performance and underline denotes the second position and the third position in
972:   972	performance ranking among methods in comparison.
974:   974	Table 5. Comparison with SOTA Methods under Protocol-3
977:   977	Protocol-3
989:   989	Baseline (ViT)
990:   990	Baseline (CLIP)
1091:  1091	# Tunable
1103:  1103	Bold type indicates the best performance and underline denotes the second position and the third position in
1104:  1104	performance ranking among methods in comparison.
1108:  1108	Ablation Study
1110:  1110	The Effectiveness of Adapters and Different Mixture Methods. As shown in Table 6, we find that
1111:  1111	mixing the intermediate feature from the global adapter with that from MoE adapters within each
1113:  1113	characteristics, can lead to considerable performance gains compared to directly concatenating the
1114:  1114	features from both in the last layer. “MoE Adapters” means that we only mix the expert adapters by
1115:  1115	the voting network in every block without global adapters, and only concatenate the aggregated
1116:  1116	feature and global feature at last as we mentioned above. “MoE and Global Adapters” means that
1117:  1117	we mix the expert adapters and global adapter by the voting network in every block.
1118:  1118	The Effectiveness of Block-aware Voting Network. Additionally, we demonstrate the efficacy of the
1119:  1119	block-aware voting network in Table 6. For the case without the voting network, we compute the
1120:  1120	summation average, with the weight of each expert adapter set to 1/3.
1121:  1121	The Starting Layer Number of Voting Blocks. Considering that the first few layers of ViT may
1122:  1122	not have extracted semantically rich features yet, we conduct ablation studies to test which layer
1123:  1123	to start voting from works best. The results show that benefiting from the block-aware design,
1126:  1126	block-aware voting from the 4th layer yields the optimal performance. In the preceding layers, it is
1127:  1127	noteworthy that the weights of each adapter remain identical.
1133:  1133	Table 6. The Ablation Study of Effectiveness of Adapters,
1134:  1134	Different Mixture Methods and Voting Network on One of the
1135:  1135	Protocol-2 Experiments
1139:  1139	MoE Adapters
1143:  1143	w/o Global Adapter
1144:  1144	w/o Expert Adapters
1145:  1145	w/o Voting Network
1147:  1147	w/o Global Adapter
1148:  1148	w/o Expert Adapters
1149:  1149	w/o Voting Network
1172:  1172	W/o Global Adapter means that we only use the domain-specific feature for
1173:  1173	evaluation. W/o Expert Adapters means that we only use the domain-invariant
1175:  1175	the best performance among methods in comparison.
1177:  1177	Table 7. Ablation Study on the Starting Layer of Voting Blocks
1179:  1179	Starting Layer of Voting Network
1200:  1200	Bold type indicates the best performance among methods in comparison.
1202:  1202	Table 8. Ablation Study on the Effectiveness of Loss Functions
1258:  1258	# Tunable
1266:  1266	Bold type indicates the best performance among methods in comparison.
1268:  1268	The Effectiveness of Loss Functions. We also do the ablation study about the effectiveness of loss
1269:  1269	functions. The results are shown in Table 8. Experimental results show that applying triplet loss on
1270:  1270	the aggregated features hurts performance. The last three rows indicate that adding ID loss L𝐼 𝐷 and
1272:  1272	tunable parameters. And simply adding ID loss L𝐼 𝐷 and classification heads without fine-tuning
1273:  1273	the heads leads to inferior performance.
1278:  1278	MoDA for Parameter-efficient Generalizable Person Re-ID
1282:  1282	StrongBaseline [14]
1314:  1314	type indicates the best performance among methods in comparison.
1321:  1321	also exhibits outstanding performance in the single-source DG ReID task setting. In this task setting,
1322:  1322	due to having only one source domain, Voting Network becomes ineffective. However, we still
1323:  1323	retained three expert adapters, even though two of them do not correspond to specific source
1324:  1324	domains. During the training process, the weights of these two adapters are consistently set to 1/2
1325:  1325	each, while during testing, the weights of all three expert adapters are equal, fixed at 1/3. Due to
1336:  1336	Our method demonstrates outstanding performance in protocol-1 and single-source DG, but its
1337:  1337	performance is not as prominent in protocol-2 and protocol-3. The main difference is that MSMT17
1338:  1338	is included in the source domains for training most settings of protocol-2 and protocol-3. Notably,
1339:  1339	on M+CS+C3 → MS of both protocol-2 and protocol-3, where MSMT17 is the target domain, our
1340:  1340	method still achieves highly competitive performance compared to the SOTA methods. MSMT17
1342:  1342	three domains. Two reasons may lead to the less prominent performance of MoDA in protocol-2 and
1343:  1343	protocol-3: (1) The imbalance in the quantity of data in the source domain increases the difficulty of
1344:  1344	optimizing the expert adapters. (2) Parameter-efficient tuning methods like Adapter, with smaller
1345:  1345	model capacities, may potentially limit the performance of MoDA when dealing with large datasets.
1346:  1346	While, in situations with fewer sources, MoDA exhibits excellent performance. This could be a path
1350:  1350	Visualization of MoDA
1352:  1352	To better illustrate the superiority and robustness of the features generated by MoDA, we conducted t-SNE visualizations of both the ID-specific tokens and the final image features (Figure 4).
1362:  1362	Fig. 4. Visualization of person features and ID-specific tokens of Market1501 and CUHK03. Circles represent
1366:  1366	Fig. 5. Visualization of baseline and MoDA features of different domains. MoDA features are impact and
1367:  1367	have no significant domain gaps, while baseline features are intermixed across domains.
1375:  1375	the domain shift problem. However, as shown in Figure 5(a), the features learned by the baseline
1384:  1384	method employs Adapter-tuning in conjunction with CLIP to achieve parameter-efficient DG ReID
1386:  1386	block-aware voting network designed to fully leverage both the image and text encoders of CLIP
1389:  1389	results with markedly fewer tunable parameters.
1392:  1392	MoDA for Parameter-efficient Generalizable Person Re-ID
1396:  1396	As analysis in Section 4.6, our method is difficult to optimize the expert adapters for imbalance
1397:  1397	data in the source domains and the parameter-efficient Adapter, with smaller model capacities,
1398:  1398	may potentially limit the performance of MoDA when dealing with large datasets. While, in
1399:  1399	situations with fewer resources, MoDA exhibits excellent performance. This could be a direction
1401:  1401	technologies to balance the parameter efficiency and performance can mitigate above problem.
1402:  1402	Besides, we will try to utilize knowledge distillation to train a smaller model to replace the voting
1404:  1404	of starting layer on ReID performance w.r.t. domain generation.
1412:  1412	[3] Shizhe Diao, Tianyang Xu, Ruijia Xu, Jiawei Wang, and T. Zhang. 2023. Mixture-of-domain-adapters: Decoupling and
1430:  1430	Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In International Conference on
1435:  1435	Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In
1438:  1438	2022. Visual prompt tuning. arXiv:2203.12119. Retrieved from https://arxiv.org/abs/2203.12119
1441:  1441	[15] Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In
1464:  1464	Li. 2022. Frozen CLIP models are efficient video learners. In Proceedings of the European Conference on Computer
1471:  1471	[25] Hao Luo, Youzhi Gu, Xingyu Liao, Shenqi Lai, and Wei Jiang. 2019. Bag of tricks and a strong baseline for deep person
1477:  1477	Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision.
1492:  1492	for parameter-efficient model tuning. arXiv:2210.17451. Retrieved from https://arxiv.org/abs/2210.17451

codex
Mix-Modality 的证据链已经很清楚，是“先造新评测场景，再证明旧方法在新场景下系统掉点，再用两个针对性补丁恢复”。MoDA 我再补一段消融表文字，确认它是不是只卖参数量，还是也证明了块级投票和全局适配器必要。
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
/bin/zsh -lc "pdftotext -f 1 -l 18 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | sed -n '1563,1720p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1563	Ablation Study
  1564	
  1565	We adopted the AGW method to perform our ablation experiments on the SYSU-MM01 dataset in
  1566	Rank-1, Rank-5, Rank-10, and mAP metrics. We show the ablation experiments for the 𝛿 parameter
  1567	in Equation (5) and the 𝜆 parameter in Equations (7) and (8), and the effects of the addition of the
  1568	method are shown in the previous tables.
  1569	As can be seen in Tables 8 and 9, in Mode 1, compared with the performance without our CIDHL,
  1570	there are at least “+7.63%,” “+1.61%,” “+0.31%,” “+4.10%,” and “+0.76%” performance improvements
  1571	in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics (𝛿 is set to 0.3), and at most “+18.54%,” “+5.60%,”
  1572	“+2.85%,” “+14.01%,” and “+1.32%” performance improvements (𝛿 is set to 0.2), respectively, when
  1573	our CIDHL is added (𝛿 is set to 0.2). In Model 2, compared to the performance without our CIDHL,
  1574	there is at least “+11.49%,” “+1.70%,” “+0.36%,” “+6.24%,” and “+1.22%” performance improvement
  1575	(𝛿 is set to 0.3) and at most “+17.27%,” “+4.21%,” “+1.79%,” “+16.21%,” and “+2.49%” performance
  1576	improvement (𝛿 is set to 0.2) in the four metrics of Rank-1, Rank-5, Rank-10, and mAP, respectively,
  1577	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
  1578	
  1579	Mix-Modality Person Re-Identification: A New and Practical Paradigm
  1580	
  1581	112:17
  1582	
  1583	Table 10. Ablation study of 𝜆
  1584	
  1585	𝜆
  1586	0.99
  1587	0.999
  1588	0.9999
  1589	1
  1590	1.001
  1591	1.01
  1592	1.1
  1593	
  1594	Rank-1
  1595	70.73
  1596	70.26
  1597	70.22
  1598	70.22
  1599	69.90
  1600	70.12
  1601	62.89
  1602	
  1603	Rank-5
  1604	92.02
  1605	91.76
  1606	91.74
  1607	91.76
  1608	91.49
  1609	91.72
  1610	86.31
  1611	
  1612	Mode 1
  1613	Rank-10
  1614	96.09
  1615	96.01
  1616	95.99
  1617	95.99
  1618	98.86
  1619	95.95
  1620	92.54
  1621	
  1622	mAP mINP
  1623	34.76 1.96
  1624	34.78 1.97
  1625	34.78 1.97
  1626	34.78 1.97
  1627	34.78 1.96
  1628	34.75 1.98
  1629	31.58 1.94
  1630	
  1631	The bold is the best result on SYSU-MM01 dataset in mode 1.
  1632	
  1633	Table 11. Ablation Study of 𝜆
  1634	
  1635	𝜆
  1636	0.99
  1637	0.999
  1638	0.9999
  1639	1
  1640	1.001
  1641	1.01
  1642	1.1
  1643	
  1644	Rank-1
  1645	73.43
  1646	73.09
  1647	73.09
  1648	73.09
  1649	72.61
  1650	73.03
  1651	64.46
  1652	
  1653	Rank-5
  1654	94.18
  1655	93.94
  1656	93.94
  1657	93.91
  1658	93.47
  1659	93.88
  1660	85.95
  1661	
  1662	Mode 2
  1663	Rank-10 mAP
  1664	97.61
  1665	37.15
  1666	97.55
  1667	37.02
  1668	97.52
  1669	37.00
  1670	97.50
  1671	37.00
  1672	97.34
  1673	36.74
  1674	97.52
  1675	36.97
  1676	92.81
  1677	31.25
  1678	
  1679	mINP
  1680	2.83
  1681	2.83
  1682	2.83
  1683	2.83
  1684	2.83
  1685	2.82
  1686	2.61
  1687	
  1688	The bold is the best result on SYSU-MM01 dataset in mode 2.
  1689	
  1690	with the addition of our CIDHL. Besides, in Mode 1, compared to only 𝐿𝑐𝑖𝑑 used (𝛿 is set to 0), the
  1691	whole 𝐿𝐶𝐼 𝐷𝐻 𝐿 achieves the best performance, showing “+33.03%,” “+19.11%,” “+12.08%” “+42.37%,”
  1692	and “+2.14%” improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. In
  1693	Mode 2, compared to only 𝐿𝑐𝑖𝑑 used (𝛿 is set to 0), the whole 𝐿𝐶𝐼 𝐷𝐻 𝐿 achieves the best performance,
  1694	showing “+30.60%,” “+15.99%,” “+9.57%,” “+43.24%,” and “+4.00%” improvement in Rank-1, Rank-5,
  1695	Rank-10, mAP, and mINP metrics, respectively.
  1696	As can be seen in Tables 10 and 11, both in Mode 1 and Mode 2, the addition of our MBSOS
  1697	brings different effects with different distance adjustment ratios, and for most of the metrics, the
  1698	best results are achieved when 𝜆 is set to 0.99.
  1699	4.4
  1700	
  1701	Visualization
  1702	
  1703	As illustrated in Figure 5, we compared the t-SNE visualization results of the original AGW and the
  1704	AGW with the addition of our CIDHL loss. After adding our CIDHL, it is clear from Figure 5(a)
  1705	that the samples of the two modalities are well differentiated, and from Figure 5(b) and (c) that
  1706	the same-identity samples within each modality are more aggregated, and the different-identity
  1707	samples are easier to distinguish. The above phenomenon combined with the improved performance
  1708	in the comparison experiments is a good proof of its effectiveness in solving the modality confusion
  1709	problem.
  1710	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.
  1711	
  1712	112:18
  1713	
  1714	W. Liu et al.
  1715	
  1716	Fig. 5. The t-SNE Visualization of only AGW (first row) and AGW addition with our CIDHL (second row).
  1717	(a) Results of mix modality. (b) Results of visible modality. (b) Results of infrared modality. Different colors
  1718	represent different identities, dots for visible modal samples, and stars for infrared modal samples.
  1719	
  1720	5

exec
/bin/zsh -lc "pdftotext -f 1 -l 18 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | sed -n '1108,1280p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1108	Ablation Study
  1109	
  1110	The Effectiveness of Adapters and Different Mixture Methods. As shown in Table 6, we find that
  1111	mixing the intermediate feature from the global adapter with that from MoE adapters within each
  1112	block, which produces representations containing both domain-specific and domain-invariant
  1113	characteristics, can lead to considerable performance gains compared to directly concatenating the
  1114	features from both in the last layer. “MoE Adapters” means that we only mix the expert adapters by
  1115	the voting network in every block without global adapters, and only concatenate the aggregated
  1116	feature and global feature at last as we mentioned above. “MoE and Global Adapters” means that
  1117	we mix the expert adapters and global adapter by the voting network in every block.
  1118	The Effectiveness of Block-aware Voting Network. Additionally, we demonstrate the efficacy of the
  1119	block-aware voting network in Table 6. For the case without the voting network, we compute the
  1120	summation average, with the weight of each expert adapter set to 1/3.
  1121	The Starting Layer Number of Voting Blocks. Considering that the first few layers of ViT may
  1122	not have extracted semantically rich features yet, we conduct ablation studies to test which layer
  1123	to start voting from works best. The results show that benefiting from the block-aware design,
  1124	MoDA can also help model to mix experts in a more fine-grained way by generating block-aware
  1125	weight of aggregation. The results are presented in Table 7. The findings indicate that initiating
  1126	block-aware voting from the 4th layer yields the optimal performance. In the preceding layers, it is
  1127	noteworthy that the weights of each adapter remain identical.
  1128	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
  1129	
  1130	139:14
  1131	
  1132	Y. Wang et al.
  1133	Table 6. The Ablation Study of Effectiveness of Adapters,
  1134	Different Mixture Methods and Voting Network on One of the
  1135	Protocol-2 Experiments
  1136	
  1137	Method
  1138	
  1139	MoE Adapters
  1140	
  1141	MoE and Global
  1142	
  1143	w/o Global Adapter
  1144	w/o Expert Adapters
  1145	w/o Voting Network
  1146	MoDA
  1147	w/o Global Adapter
  1148	w/o Expert Adapters
  1149	w/o Voting Network
  1150	MoDA
  1151	
  1152	M+MS+CS → C3
  1153	mAP
  1154	Rank-1
  1155	28.6
  1156	28.6
  1157	29.2
  1158	28.9
  1159	28.9
  1160	29.0
  1161	29.6
  1162	29.8
  1163	31.8
  1164	31.9
  1165	31.9
  1166	31.9
  1167	30.7
  1168	32.1
  1169	33.6
  1170	34.1
  1171	
  1172	W/o Global Adapter means that we only use the domain-specific feature for
  1173	evaluation. W/o Expert Adapters means that we only use the domain-invariant
  1174	feature for evaluation. MoDA is to concatenate both of them. Bold type indicates
  1175	the best performance among methods in comparison.
  1176	
  1177	Table 7. Ablation Study on the Starting Layer of Voting Blocks
  1178	
  1179	Starting Layer of Voting Network
  1180	1st layer
  1181	4th layer
  1182	6th layer
  1183	10th layer
  1184	12th layer
  1185	
  1186	M+MS+CS → C3
  1187	mAP
  1188	Rank-1
  1189	31.2
  1190	32.6
  1191	33.6
  1192	34.1
  1193	32.2
  1194	32.9
  1195	30.8
  1196	31.8
  1197	30.8
  1198	32.0
  1199	
  1200	Bold type indicates the best performance among methods in comparison.
  1201	
  1202	Table 8. Ablation Study on the Effectiveness of Loss Functions
  1203	
  1204	L𝑐𝑜𝑛𝑠𝑖𝑠
  1205	Ø
  1206	Ø
  1207	Ø
  1208	Ø
  1209	
  1210	Loss Functions
  1211	L𝑎𝑔𝑔
  1212	L𝑒𝑥𝑝𝑒𝑟𝑡 / L𝑔𝑙𝑜𝑏𝑎𝑙
  1213	L𝑚𝑠𝑒
  1214	L𝑖2𝑡𝑐𝑒
  1215	L𝑡𝑟𝑖𝑝𝑙𝑒𝑡
  1216	L𝑡𝑟𝑖𝑝𝑙𝑒𝑡
  1217	L𝑖2𝑡𝑐𝑒
  1218	L𝐼 𝐷
  1219	Ø
  1220	Ø
  1221	Ø
  1222	Ø
  1223	Ø
  1224	Ø
  1225	Ø
  1226	Ø
  1227	Ø
  1228	Ø
  1229	Ø
  1230	Ø
  1231	Ø
  1232	Ø
  1233	Ø
  1234	Ø
  1235	Ø
  1236	Ø
  1237	Ø
  1238	Ø
  1239	Ø
  1240	Ø
  1241	Ø
  1242	-
  1243	
  1244	M+C3+CS
  1245	→ MS
  1246	mAP Rank-1
  1247	19.3
  1248	48.8
  1249	18.8
  1250	45.3
  1251	20.6
  1252	50.3
  1253	20.1
  1254	49.2
  1255	20.5
  1256	50.0
  1257	
  1258	# Tunable
  1259	Params (M)
  1260	15.5
  1261	15.5
  1262	38.1 (Tune the classifier heads)
  1263	15.5 (Freeze the classifier heads)
  1264	15.5
  1265	
  1266	Bold type indicates the best performance among methods in comparison.
  1267	
  1268	The Effectiveness of Loss Functions. We also do the ablation study about the effectiveness of loss
  1269	functions. The results are shown in Table 8. Experimental results show that applying triplet loss on
  1270	the aggregated features hurts performance. The last three rows indicate that adding ID loss L𝐼 𝐷 and
  1271	classification heads, and fine-tuning the heads lead to only marginal gains but substantially more
  1272	tunable parameters. And simply adding ID loss L𝐼 𝐷 and classification heads without fine-tuning
  1273	the heads leads to inferior performance.
  1274	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
  1275	
  1276	139:15
  1277	
  1278	MoDA for Parameter-efficient Generalizable Person Re-ID
  1279	Table 9. Evaluations on Single-Source DG ReID
  1280	

exec
/bin/zsh -lc "pdftotext -f 1 -l 18 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' - | nl -ba | sed -n '1350,1385p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1350	Visualization of MoDA
  1351	
  1352	To better illustrate the superiority and robustness of the features generated by MoDA, we conducted t-SNE visualizations of both the ID-specific tokens and the final image features (Figure 4).
  1353	Figure 4(a) and (b) presents the features obtained from the Market1501 and CUHK03 datasets.
  1354	Within each dataset, they generate distinctive features for each person ID. Moreover, since the
  1355	ID-specific tokens from different datasets are derived from the same shared text feature space, they
  1356	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.
  1357	
  1358	139:16
  1359	
  1360	Y. Wang et al.
  1361	
  1362	Fig. 4. Visualization of person features and ID-specific tokens of Market1501 and CUHK03. Circles represent
  1363	the image features of pedestrians, with different colors indicating different person IDs. Triangles correspond
  1364	to the ID-specific tokens of the respective person ID.
  1365	
  1366	Fig. 5. Visualization of baseline and MoDA features of different domains. MoDA features are impact and
  1367	have no significant domain gaps, while baseline features are intermixed across domains.
  1368	
  1369	inherently provide advantages in mitigating the domain shift problem, highlighting the superiority
  1370	of our cross-modal approach.
  1371	From the Figure 5(b), it can be observed that the features learned by MoDA retain the intrinsic
  1372	discriminative characteristics of each domain while avoiding a significant domain gap between
  1373	them. This indicates that there is a strong correlation among the domain feature and provides
  1374	not only discriminative but complementary information to help generalize better and mitigate
  1375	the domain shift problem. However, as shown in Figure 5(a), the features learned by the baseline
  1376	model, which trained with a single hybrid dataset combining all source domain datasets, are
  1377	almost intermixed across domains, lacking clear distinctions. This results in a loss of the intrinsic
  1378	information that each domain possesses.
  1379	5
  1380	
  1381	Conclusion
  1382	
  1383	In this study, we propose a novel MoE-based method for DG ReID, referred to as MoDAs. This
  1384	method employs Adapter-tuning in conjunction with CLIP to achieve parameter-efficient DG ReID
  1385	through the utilization of CLIP’s contrastive loss. Additionally, we introduce a cross-modality,

codex
已按本地 PDF 用 `pdftotext` 读了方法和关键实验分析。你列的是 6 个文件，其中 Mix-Modality 有 arXiv 版和 ACM 正式版，方法基本一致，但我按两篇分别拆。

**1. Memory-augmented shuffled meta learning for visible-infrared person re-identification**

1. **触发观察**：作者抓住的是现有 VI-ReID 度量学习的两个局限。样本级方法只看局部样本对，中心级方法又把类内复杂变化压成中心，都会低估跨模态同身份内部变化。更具体的是，普通 meta learning 只用一组 query-support，样本利用率是 `1/K`，而 ReID 检索本质上每张图都可能当 query，也可能当 gallery。
2. **重定义动作**：他们把“跨模态距离没学好”重写成“没有学到全局 meta metric”。关键词是 `shuffled meta learning`、`memory meta learning`、`global meta metrics`、`historical information`、`query-support`。旧问题是模态差异，新问题变成了 query-support 组织方式不足和 batch-local 度量不足。
3. **机制怎么长出来**：机制基本是顺着重定义推出来的。既然 ReID 是 query 查 support，就把每个身份的可见光和红外样本循环拆成 query/support，让每张图都轮流参与。既然单 batch 不够全局，就用 memory bank 存历史特征，把历史样本当扩展 support。SMR 再约束 V2I 和 I2V 的预测一致性。
4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
5. **reviewer 为什么买账**：这篇卖的是视角加度量组织，不是 backbone。novelty 来自把 ReID 检索过程解释成 meta query-support 度量学习，并用 shuffling 和 memory 把这个解释落地。模块本身不算惊艳，但逻辑闭合度还可以。

**2. Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification**

1. **触发观察**：无监督 ReID 常用聚类或分类式伪标签，但聚类要估类别数，且错误伪标签会滚雪球。作者观察到，单看两张图的相似度很脆弱，但看这对样本周围的邻域结构，能更稳地判断它们是不是同一身份。
2. **重定义动作**：他们把“给图片估伪身份标签”改写成“给样本对估 pairwise relationship”。关键词是 `pairwise pseudo label`、`pairwise neighborhood structure`、`relationship estimation`、`GCN pseudo label generator`。这一步很关键，因为它绕开了聚类类别数这个老难题。
3. **机制怎么长出来**：机制很自然。CNN 提图像特征并维护 feature memory，针对样本对构建 k 近邻图，GCN 输入这个 pairwise neighborhood structure，输出这对图是不是同身份。CNN 用 GCN 给的 pairwise label 训练，GCN 又用 CNN 的特征更新，两者交替优化。少量 labeled metadata 只用于启动 GCN，不直接训练 CNN。
4. **证据闭环**：最关键的消融是和 KNN、相似度阈值、MPLP 比。Market 上相似度阈值是 72.8 Rank-1、39.9 mAP，MPRD 到 83.0、51.1。去掉大量无标签数据训练 GCN 的版本只有 75.8、43.1，说明不是 metadata 本身撑起来的。t-SNE 里加 GCN 后同身份更聚、易混身份分开，直接对应“邻域关系比点对相似度更可靠”。
5. **reviewer 为什么买账**：这篇真正卖的是任务重写。GCN 是载体，核心 novelty 是把无监督 ReID 从聚类伪身份，改成图结构上的样本对关系估计。它的弱点是用了少量标注身份，严格无监督口径会被质疑，但作者用“只训练 GCN，不训练 CNN”来缓解。

**3. Mix-Modality Person Re-Identification，arXiv 版**

1. **触发观察**：现有 VI-ReID 默认是可见光 query 查红外 gallery，或反过来。但真实系统里 query 和 gallery 往往都是混合模态。旧方法一到混合模态场景就出现 `modality confusion`，同模态不同身份可能比跨模态同身份更近。
2. **重定义动作**：这篇是典型的“改问题定义”。它说大家以为 VI-ReID 只需要处理双模态互查，其实更实际的是 `Mix-Modality ReID`。关键词是 `new practical paradigm`、`mix-modality testing paradigm`、`modality mixing ratio`、`modality confusion`。
3. **机制怎么长出来**：机制和问题绑定很紧。既然同模态相似性会干扰身份，CIDHL 就用同身份跨模态中心拉近、不同身份中心推远、同模态同身份样本向中心聚合。既然测试时混合 gallery 中存在同模态桥，MBSOS 就把 query-gallery 直连距离改成经过 gallery bridge node 的最短路径问题。
4. **证据闭环**：核心证据是先构造 mix-modality 测试集，再看 AGW 等旧方法在不同 mixing ratio 下 mAP 和 mINP 普遍下降，说明新问题真实存在。之后在 AGW、HCTL、DEEN 等方法上加 CIDHL 和 MBSOS，大多指标恢复或提升。t-SNE 显示加 CIDHL 后同身份更聚、不同身份更分，支撑“modality confusion 被缓解”。
5. **reviewer 为什么买账**：主要买的是新范式，不是 CIDHL 或 MBSOS 本身。loss 和路径优化都不复杂，但它们服务于一个清楚的新评测场景。对 reviewer 来说，这篇的 novelty 来源是“旧协议不实用，新协议暴露新失败模式”。

**4. Mix-Modality Person Re-Identification，ACM TOMM 正式版**

1. **触发观察**：正式版进一步把观察说得更工程化，单模态 ReID 和传统 VI-ReID 都没有覆盖“query 集和 gallery 集同时混合可见光与红外”的情况。作者强调 Rank-1 有时会因为同模态易匹配而变好，但 mAP 和 mINP 下降，说明整体检索质量变差。
2. **重定义动作**：正式版把这个问题固定成 `MM-ReID`，并把混合比例变成可控变量。旧说法是“跨模态差异”，新说法是“同模态自然相似性会污染跨模态身份匹配”。这个措辞比单纯说 domain gap 更具体。
3. **机制怎么长出来**：CIDHL 对训练分布动手，要求同身份跨模态中心接近，不同身份无论同模态还是跨模态都远离，并控制同模态同身份样本到中心的距离。MBSOS 对测试距离动手，把 gallery 里的相似模态样本作为桥，重新计算 query 到 gallery 的最短路径距离。
4. **证据闭环**：正式版的证据链更完整。它先用 mixing ratio 展示旧方法在 RegDB、SYSU-MM01、LLCM 上的系统性退化，再用 `+C`、`+M`、`+C+M` 证明两个机制能迁移到多个基线。δ 和 λ 的消融证明两个超参不是任意选的。t-SNE 则证明 CIDHL 后身份结构更清楚。
5. **reviewer 为什么买账**：这篇正式版买账点仍然是问题定义。它不是说“我有一个更强损失”，而是说“你们一直评错了实际场景”。这类论文能发的关键，是先把协议变化讲成现实需求，再让机制看起来只是对新失败模式的必要修补。

**5. MoDA，Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification**

1. **触发观察**：DG ReID 里 MoE 方法有用，但通常要全量微调整个 backbone 和多个专家，参数量随源域数增长，分类头也随身份数膨胀。换成 CLIP 这类大模型后，全量微调更不现实，还可能破坏预训练泛化能力。
2. **重定义动作**：作者把“怎么做更强 DG ReID”重写成“怎么做参数高效、可扩展的 DG ReID”。关键词是 `parameter-efficient`、`frozen CLIP`、`domain adapters`、`global adapter`、`block-aware voting network`、`tunable parameters`。
3. **机制怎么长出来**：机制很顺。参数效率要求冻结 CLIP，只训 adapter。DG 要保留源域互补性，所以每个源域一个 domain-specific adapter，再加一个 global adapter。ViT 的 adapter 在每个 block 里，如果按传统专家投票要多次 forward，于是提出 block-aware voting，在每个 block 用 CLIP 学到的 ID-specific text tokens 作为 domain prototypes 来算权重。
4. **证据闭环**：关键证据不是单纯 SOTA，而是性能和参数量一起报。Protocol-1 中 MoDA 用 19.1M 可训练参数，和 57M 到 165M 参数量的方法竞争。Table 6 证明只用 global 或只用 expert 都不如组合，去掉 voting network 也变差。Table 7 证明从第 4 层开始 voting 最好。Table 8 说明加 ID 分类头只带来很小收益，却显著增加参数，支撑“不要分类头”的设计。
5. **reviewer 为什么买账**：这篇卖的是工程约束下的新视角。它不是发现了新的身份线索，而是把 DG ReID 和 PEFT、CLIP、MoE 组合成一个参数效率故事。reviewer 买账的前提是接受“同等或接近性能，少很多可训练参数”本身就是贡献。

**6. MSP-ReID，Hairstyle-Robust Cloth-Changing Person Re-Identification**

1. **触发观察**：CC-ReID 为了去衣服偏差，常把注意力推向 head 区域，但 human parsing 往往把 face 和 hair 合在 head 里。结果模型看似摆脱衣服，其实转而依赖 hairstyle。作者把这叫 `hairstyle shortcut`。同时，粗暴删除衣服区域会把身体结构和空间上下文也删掉。
2. **重定义动作**：它把“换衣 ReID 的核心是衣服偏差”改成“换衣 ReID 同时有 hairstyle shortcut 和 clothing structure loss”。关键词是 `hairstyle distraction`、`structural preservation`、`same-identity different-hairstyle`、`cloth-preserved random erasing`。
3. **机制怎么长出来**：机制是三件套。HSOA 用 HairFastGAN 合成同身份不同发型，强迫 embedding 不依赖发型。CPRE 只在衣服区域内按比例擦除，保留部分衣服像素和身体结构，而不是全删。RPA 用 parsing 监督 attention 强调 face 和 limbs，压低 hair。测试时只用 RGB 图像和 ID 分支。
4. **证据闭环**：它的证据主要是组件消融和可视化。Tab. 4 展示 HSOA、CPRE、RPA 单独和组合都有收益，三者一起最好。定性图显示 baseline 会被相似衣服或发型干扰，MSP 的 attention 呈现“脸和肢体高响应，头发和大衣服区域低响应”。不过从方法论角度看，它最缺的是一个真实 hairstyle-change 分组评测，现有证据更多证明模块涨点和注意力变化。
5. **reviewer 为什么买账**：如果买账，买的是“hairstyle shortcut”这个被忽略的 nuisance，而不是 HairFastGAN、random erasing、parsing attention 这些零件。它的 novelty 来源是把 CC-ReID 的失败从衣服扩展到发型，并指出旧方法的 head reliance 其实埋了新捷径。

**共同套路**

这批论文共同的套路是：先找一个旧范式没有显式处理的错位点，比如 batch-local 度量、聚类伪标签、纯双模态协议、全量微调成本、head 区域里的发型捷径，再把它命名成一个新问题。机制通常不复杂，但要和新问题强绑定，最后用“旧设定下的失败现象加针对性消融或可视化”证明不是单纯堆模块涨点。
tokens used
181,245
已按本地 PDF 用 `pdftotext` 读了方法和关键实验分析。你列的是 6 个文件，其中 Mix-Modality 有 arXiv 版和 ACM 正式版，方法基本一致，但我按两篇分别拆。

**1. Memory-augmented shuffled meta learning for visible-infrared person re-identification**

1. **触发观察**：作者抓住的是现有 VI-ReID 度量学习的两个局限。样本级方法只看局部样本对，中心级方法又把类内复杂变化压成中心，都会低估跨模态同身份内部变化。更具体的是，普通 meta learning 只用一组 query-support，样本利用率是 `1/K`，而 ReID 检索本质上每张图都可能当 query，也可能当 gallery。
2. **重定义动作**：他们把“跨模态距离没学好”重写成“没有学到全局 meta metric”。关键词是 `shuffled meta learning`、`memory meta learning`、`global meta metrics`、`historical information`、`query-support`。旧问题是模态差异，新问题变成了 query-support 组织方式不足和 batch-local 度量不足。
3. **机制怎么长出来**：机制基本是顺着重定义推出来的。既然 ReID 是 query 查 support，就把每个身份的可见光和红外样本循环拆成 query/support，让每张图都轮流参与。既然单 batch 不够全局，就用 memory bank 存历史特征，把历史样本当扩展 support。SMR 再约束 V2I 和 I2V 的预测一致性。
4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
5. **reviewer 为什么买账**：这篇卖的是视角加度量组织，不是 backbone。novelty 来自把 ReID 检索过程解释成 meta query-support 度量学习，并用 shuffling 和 memory 把这个解释落地。模块本身不算惊艳，但逻辑闭合度还可以。

**2. Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification**

1. **触发观察**：无监督 ReID 常用聚类或分类式伪标签，但聚类要估类别数，且错误伪标签会滚雪球。作者观察到，单看两张图的相似度很脆弱，但看这对样本周围的邻域结构，能更稳地判断它们是不是同一身份。
2. **重定义动作**：他们把“给图片估伪身份标签”改写成“给样本对估 pairwise relationship”。关键词是 `pairwise pseudo label`、`pairwise neighborhood structure`、`relationship estimation`、`GCN pseudo label generator`。这一步很关键，因为它绕开了聚类类别数这个老难题。
3. **机制怎么长出来**：机制很自然。CNN 提图像特征并维护 feature memory，针对样本对构建 k 近邻图，GCN 输入这个 pairwise neighborhood structure，输出这对图是不是同身份。CNN 用 GCN 给的 pairwise label 训练，GCN 又用 CNN 的特征更新，两者交替优化。少量 labeled metadata 只用于启动 GCN，不直接训练 CNN。
4. **证据闭环**：最关键的消融是和 KNN、相似度阈值、MPLP 比。Market 上相似度阈值是 72.8 Rank-1、39.9 mAP，MPRD 到 83.0、51.1。去掉大量无标签数据训练 GCN 的版本只有 75.8、43.1，说明不是 metadata 本身撑起来的。t-SNE 里加 GCN 后同身份更聚、易混身份分开，直接对应“邻域关系比点对相似度更可靠”。
5. **reviewer 为什么买账**：这篇真正卖的是任务重写。GCN 是载体，核心 novelty 是把无监督 ReID 从聚类伪身份，改成图结构上的样本对关系估计。它的弱点是用了少量标注身份，严格无监督口径会被质疑，但作者用“只训练 GCN，不训练 CNN”来缓解。

**3. Mix-Modality Person Re-Identification，arXiv 版**

1. **触发观察**：现有 VI-ReID 默认是可见光 query 查红外 gallery，或反过来。但真实系统里 query 和 gallery 往往都是混合模态。旧方法一到混合模态场景就出现 `modality confusion`，同模态不同身份可能比跨模态同身份更近。
2. **重定义动作**：这篇是典型的“改问题定义”。它说大家以为 VI-ReID 只需要处理双模态互查，其实更实际的是 `Mix-Modality ReID`。关键词是 `new practical paradigm`、`mix-modality testing paradigm`、`modality mixing ratio`、`modality confusion`。
3. **机制怎么长出来**：机制和问题绑定很紧。既然同模态相似性会干扰身份，CIDHL 就用同身份跨模态中心拉近、不同身份中心推远、同模态同身份样本向中心聚合。既然测试时混合 gallery 中存在同模态桥，MBSOS 就把 query-gallery 直连距离改成经过 gallery bridge node 的最短路径问题。
4. **证据闭环**：核心证据是先构造 mix-modality 测试集，再看 AGW 等旧方法在不同 mixing ratio 下 mAP 和 mINP 普遍下降，说明新问题真实存在。之后在 AGW、HCTL、DEEN 等方法上加 CIDHL 和 MBSOS，大多指标恢复或提升。t-SNE 显示加 CIDHL 后同身份更聚、不同身份更分，支撑“modality confusion 被缓解”。
5. **reviewer 为什么买账**：主要买的是新范式，不是 CIDHL 或 MBSOS 本身。loss 和路径优化都不复杂，但它们服务于一个清楚的新评测场景。对 reviewer 来说，这篇的 novelty 来源是“旧协议不实用，新协议暴露新失败模式”。

**4. Mix-Modality Person Re-Identification，ACM TOMM 正式版**

1. **触发观察**：正式版进一步把观察说得更工程化，单模态 ReID 和传统 VI-ReID 都没有覆盖“query 集和 gallery 集同时混合可见光与红外”的情况。作者强调 Rank-1 有时会因为同模态易匹配而变好，但 mAP 和 mINP 下降，说明整体检索质量变差。
2. **重定义动作**：正式版把这个问题固定成 `MM-ReID`，并把混合比例变成可控变量。旧说法是“跨模态差异”，新说法是“同模态自然相似性会污染跨模态身份匹配”。这个措辞比单纯说 domain gap 更具体。
3. **机制怎么长出来**：CIDHL 对训练分布动手，要求同身份跨模态中心接近，不同身份无论同模态还是跨模态都远离，并控制同模态同身份样本到中心的距离。MBSOS 对测试距离动手，把 gallery 里的相似模态样本作为桥，重新计算 query 到 gallery 的最短路径距离。
4. **证据闭环**：正式版的证据链更完整。它先用 mixing ratio 展示旧方法在 RegDB、SYSU-MM01、LLCM 上的系统性退化，再用 `+C`、`+M`、`+C+M` 证明两个机制能迁移到多个基线。δ 和 λ 的消融证明两个超参不是任意选的。t-SNE 则证明 CIDHL 后身份结构更清楚。
5. **reviewer 为什么买账**：这篇正式版买账点仍然是问题定义。它不是说“我有一个更强损失”，而是说“你们一直评错了实际场景”。这类论文能发的关键，是先把协议变化讲成现实需求，再让机制看起来只是对新失败模式的必要修补。

**5. MoDA，Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification**

1. **触发观察**：DG ReID 里 MoE 方法有用，但通常要全量微调整个 backbone 和多个专家，参数量随源域数增长，分类头也随身份数膨胀。换成 CLIP 这类大模型后，全量微调更不现实，还可能破坏预训练泛化能力。
2. **重定义动作**：作者把“怎么做更强 DG ReID”重写成“怎么做参数高效、可扩展的 DG ReID”。关键词是 `parameter-efficient`、`frozen CLIP`、`domain adapters`、`global adapter`、`block-aware voting network`、`tunable parameters`。
3. **机制怎么长出来**：机制很顺。参数效率要求冻结 CLIP，只训 adapter。DG 要保留源域互补性，所以每个源域一个 domain-specific adapter，再加一个 global adapter。ViT 的 adapter 在每个 block 里，如果按传统专家投票要多次 forward，于是提出 block-aware voting，在每个 block 用 CLIP 学到的 ID-specific text tokens 作为 domain prototypes 来算权重。
4. **证据闭环**：关键证据不是单纯 SOTA，而是性能和参数量一起报。Protocol-1 中 MoDA 用 19.1M 可训练参数，和 57M 到 165M 参数量的方法竞争。Table 6 证明只用 global 或只用 expert 都不如组合，去掉 voting network 也变差。Table 7 证明从第 4 层开始 voting 最好。Table 8 说明加 ID 分类头只带来很小收益，却显著增加参数，支撑“不要分类头”的设计。
5. **reviewer 为什么买账**：这篇卖的是工程约束下的新视角。它不是发现了新的身份线索，而是把 DG ReID 和 PEFT、CLIP、MoE 组合成一个参数效率故事。reviewer 买账的前提是接受“同等或接近性能，少很多可训练参数”本身就是贡献。

**6. MSP-ReID，Hairstyle-Robust Cloth-Changing Person Re-Identification**

1. **触发观察**：CC-ReID 为了去衣服偏差，常把注意力推向 head 区域，但 human parsing 往往把 face 和 hair 合在 head 里。结果模型看似摆脱衣服，其实转而依赖 hairstyle。作者把这叫 `hairstyle shortcut`。同时，粗暴删除衣服区域会把身体结构和空间上下文也删掉。
2. **重定义动作**：它把“换衣 ReID 的核心是衣服偏差”改成“换衣 ReID 同时有 hairstyle shortcut 和 clothing structure loss”。关键词是 `hairstyle distraction`、`structural preservation`、`same-identity different-hairstyle`、`cloth-preserved random erasing`。
3. **机制怎么长出来**：机制是三件套。HSOA 用 HairFastGAN 合成同身份不同发型，强迫 embedding 不依赖发型。CPRE 只在衣服区域内按比例擦除，保留部分衣服像素和身体结构，而不是全删。RPA 用 parsing 监督 attention 强调 face 和 limbs，压低 hair。测试时只用 RGB 图像和 ID 分支。
4. **证据闭环**：它的证据主要是组件消融和可视化。Tab. 4 展示 HSOA、CPRE、RPA 单独和组合都有收益，三者一起最好。定性图显示 baseline 会被相似衣服或发型干扰，MSP 的 attention 呈现“脸和肢体高响应，头发和大衣服区域低响应”。不过从方法论角度看，它最缺的是一个真实 hairstyle-change 分组评测，现有证据更多证明模块涨点和注意力变化。
5. **reviewer 为什么买账**：如果买账，买的是“hairstyle shortcut”这个被忽略的 nuisance，而不是 HairFastGAN、random erasing、parsing attention 这些零件。它的 novelty 来源是把 CC-ReID 的失败从衣服扩展到发型，并指出旧方法的 head reliance 其实埋了新捷径。

**共同套路**

这批论文共同的套路是：先找一个旧范式没有显式处理的错位点，比如 batch-local 度量、聚类伪标签、纯双模态协议、全量微调成本、head 区域里的发型捷径，再把它命名成一个新问题。机制通常不复杂，但要和新问题强绑定，最后用“旧设定下的失败现象加针对性消融或可视化”证明不是单纯堆模块涨点。
