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
session id: 019ef71f-bec2-7570-8ebd-80ca3b03b470
--------
user
你是 ReID 论文创新挖掘员 16/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(16/20):
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我会先按项目规则快速读取目标和状态文件，确认没有暂停或方向变更；随后只用 `pdftotext -l 3` 读取这 9 篇论文的摘要和引言。
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
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,260p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
目标文件确认没有暂停，当前任务就是从论文库里反推方向。我现在分批抽取每篇前 3 页，重点看它们怎样定义问题、搭机制和组织证据，而不是做普通综述。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
7140

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 5, MAY 2026

Self-Rectification Historical Consistency Learning
for Coupled Noisy Visible-Infrared
Person Re-Identification
Jiacheng Zhao , Yongxi Li , and Changsheng Xu , Fellow, IEEE

Abstract—Visible-infrared person re-identification (VI-ReID)
retrieves cross-modal identity matches between visible and
infrared images, offering significant value for round-the-clock
surveillance. Despite recent advances, challenges remain: the task
relies heavily on high-quality annotations, and factors such as
occlusion, viewpoint variations, and the inherent difficulty of
labeling infrared images inevitably introduce noisy annotations
(NA) into the dataset during large-scale dataset construction.
Moreover, coupled noisy labels in two modalities lead to noisy
correspondence (NC), further complicating the learning process.
Although prior research has achieved relatively stable results
in addressing the NA and NC problem for VI-ReID through
noise detection and robust loss functions, they still exhibit certain
limitations: 1) Underutilization of training data. Existing methods
often discard noisy samples to mitigate their negative impact,
overlooking their potential value. 2) Lack of historical relevance.
Unstable learning dynamics under noisy labels lead to inconsistent outputs, yet current approaches ignore the valuable historical
information embedded in these fluctuations. Focusing on these
challenges in VI-ReID, we propose Self-Rectification Historical
Consistency Learning (SRHCL) for VI-ReID, which consists
of noise detection, self-refined label rectification, and historical
consistency learning modules. Firstly, the noise detection module
calculates confidence weights for each sample by modeling the
model’s loss response, thereby mitigating the adverse impact
of noisy samples in subsequent training phases. Secondly, we
propose a self-refined label rectification module to rectify noisy
labels by reliable historical predictions, progressively collating the
training data at fixed intervals. Finally, we introduce cross-modal
contrastive learning and early learning regularization based on
momentum-updated memories to facilitate historical consistency
learning. Extensive experiments conducted on SYSU-MM01 and
RegDB datasets demonstrate the robustness and effectiveness of
our method across varying noisy ratios.
Received 2 May 2025; revised 18 November 2025; accepted 3 December
2025. Date of publication 11 December 2025; date of current version 7 May
2026. This work was supported in part by Guangdong Science and Technology
Program under Grant 2024B01015004, in part by Beijing Natural Science
Foundation under Grant L252032, and in part by the Joint Funds of the
National Natural Science Foundation of China under Grant U23A20387. This
article was recommended by Associate Editor L. Nie. (Corresponding author:
Changsheng Xu.)
Jiacheng Zhao is with the School of Information Science and
Technology, ShanghaiTech University, Shanghai 201210, China (e-mail:
zhaojch2022@shanghaitech.edu.cn).
Yongxi Li is with the State Key Laboratory of Multimodal Artificial
Intelligence Systems, Institute of Automation, Chinese Academy of Sciences,
Beijing 100190, China (e-mail: liyongxi@outlook.com).
Changsheng Xu is with the State Key Laboratory of Multimodal Artificial
Intelligence Systems, Institute of Automation, Chinese Academy of Sciences,
Beijing 100190, China, and also with the School of Artificial Intelligence,
University of Chinese Academy of Sciences, Beijing 100049, China (e-mail:
csxu@nlpr.ia.ac.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3642770

Index Terms—Visible-infrared person re-identification, noise
detection, contrastive learning, label rectification.

I. I NTRODUCTION

V

ISIBLE-INFRARED person re-identification (VI-ReID)
is a challenging cross-modal retrieval task that seeks to
match individuals across visible and infrared modalities from
a gallery of person images [1], [2], [3], [4], [5], [6]. Due to
the advantages of infrared cameras under low-light conditions,
this task has drawn increasing attention in real-life surveillance and security systems. Although learning from different
modalities allows models to uncover rich and diverse shared
semantics [7], [8], [9], the significant modality discrepancy
between visible and infrared images poses new challenges.
Numerous efforts [9], [10], [11] have been made to address
the cross-modal discrepancy between visible and infrared
images, aiming to enhance multi-modal learning and improve
performance in VI-ReID. ADCA [12] employed image augmentations and heterogeneous feature aggregation to narrow
the differences between modalities. Ye et al. [3] designed
channel augmentations to mitigate differences and establish
relationships of input channels. Kim et al. [13] proposed a
part-mix strategy that generates part-aware augmented samples
through the mixing of part-level descriptors. These augmentations provide an intermediary modality as joint inputs for
the model, enhancing generalization capabilities for VI-ReID.
For better feature alignment, DFLN-ViT [14] introduced crossmodal matching using part and location information, along
with modifications to the model structure. Recent studies
[15], [16] further researched this problem by incorporating
structural and shape information to extract modality-irrelevant
identity features to improve the robustness.
Despite advancements in VI-ReID, there remain several
challenging issues that impede the practical application of
existing methods. The success of these supervised VI-ReID
approaches relies heavily on high-quality data annotations that
are often resource-intensive and laborious to obtain, especially
when annotating infrared data. Meanwhile, the gap between
identities within a single modality is typically smaller than the
variation between the same identity across modalities. These
inherent characteristics significantly increase the difficulty of
obtaining accurate annotations. Consequently, the noisy annotations (NA) problem inevitably exists in collected cross-modal
data, complicating the learning process and causing performance degradation.Additionally, as a multi-modal retrieval

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

ZHAO et al.: SRHCL FOR COUPLED NOISY VISIBLE-INFRARED PERSON RE-IDENTIFICATION

7141

of the dataset. However, existing label correction methods
are typically limited to single-modal embedding spaces and
primarily designed for classification tasks. Consequently, these
approaches employed in prior works may not be directly
applicable to VI-ReID with coupled noisy labels.
Consequently, the primary challenges in learning with coupled noisy labels in VI-ReID can be summarized as follows:

Fig. 1. Noisy Annotations (NA) and Noisy Correspondence (NC). In the
figure, the shape of each sample represents its corresponding modality, while
the color differentiates the correct identity of each sample.

task, VI-ReID faces the challenge of noisy correspondence
(NC), which arises from coupled noisy labels. As shown in
Figure 1, cross-modal input image pairs are randomly sampled
and composed from each dataset according to their annotated
identity labels during the training phase, leading to mismatches
of cross-modal pairs. This exacerbates the complexity of the
learning process and poses further hurdles in achieving robust
performance.
In learning with noisy labels, previous approaches mainly
concentrated on precise noise detection to filter out noise
samples and mitigate their adverse effects. Existing methods
[17], [18], [19], [20] primarily treated noisy detection as a
classification task, relying on single-modal feature mining
to identify noisy samples. For learning with coupled noisy
labels in VI-ReID, DART [21] and LCNL [22] leveraged
the memorization effect [17], [23], [24] that DNNs tend to
fit clean data simple patterns during training at the early
stage. Consequently, the model tends to show reduced loss
for simple and clean samples, while simultaneously exhibiting
an increased loss for noisy samples. By utilizing a Gaussian
Mixture Model (GMM) [25] to model the loss distributions,
the model computes confidence weights for each sample and
can effectively split noise samples from the dataset. Building
upon this, their approach suppresses noisy samples through
loss reweighting while concentrating exclusively on clean
samples for subsequent training. However, this strategy may
lead to under-utilization of the dataset.
In the past decade, negative learning with complementary
labels [26], [27], [28] has been widely studied to cut data
annotation costs. For noisy correspondence in cross-modal
learning, works [29], [30], [31] further explore complementary
learning to address mismatched data underutilization. While
directly treating noisy labels as complementary labels still
causes suboptimal utilization. Notably, negative labels have far
less information than correct identity labels. Moreover, unlike
unsupervised or semi-supervised learning, noisy label learning
has unique potential: models can learn discriminative patterns
from clean samples via initial training, then gradually make
accurate identity predictions. Unfortunately, this potential is
overlooked in existing negative learning frameworks for noisy
samples.
In response to noisy data underutilization, researchers have
proposed label rectification methods. It serves to correct the
label of noisy samples, effectively transforming them into
clean samples, which enables the comprehensive utilization

• Underutilization of training data. Previous research
on VI-ReID with NA and NC mainly concentrated on
enhancing noise detection accuracy. Such studies often
train models exclusively on clean data to attain stable
outcomes. However, this strategy causes underutilization
of training data, ignoring the value of noisy samples.
Although label rectification can mitigate this issue by
rectifying noisy labels, current approaches are primarily
designed for single-modality classification tasks. Consequently, they fail to yield reliable outcomes in VI-ReID.
• Lack of historical relevance. In the context of learning
with noisy labels, the models are susceptible to overfitting
the noisy samples, leading to historical inconsistencies in
the model’s predictions and features. Existing methods
rely solely on noise detection results and model outputs at
the current timestep during training, while neglecting the
temporal correlations with earlier model outputs. These
correlations contain valuable historical information that
could facilitate more stable learning dynamics.
Inspired by these observations, we introduce a novel
approach called Self-Rectification Historical Consistency
Learning (SRHCL) to solve the problem of learning with
coupled noisy labels for VI-ReID. In detail, SRHCL consists
of three steps: noise detection, historical consistency learning,
and self-refined label rectification. Firstly, the noise detection
module utilizes the model’s memorization effect [32] and
fits the loss response by Gaussian Mixture Models (GMMs)
to calculate confidence weights for each sample. Secondly,
we adopt contrastive learning to SRHCL for robust multimodal learning within a shared feature space. Thirdly, based
on the stable predictions from the early training stage [33],
our method designs early learning regularization loss and
self-refined label rectification modules based on the historical
memory mechanism. This approach serves to stabilize the
training dynamics and restrains drastic fluctuations in the
model’s predictions, as shown in Figure 2 (b).
The key contributions in our paper are summarized as
follows:
• We present a novel three-step pipeline for robust learning
of VI-ReID despite the presence of noisy labels. It
prevents models from overfitting to noise in the dataset
and achieves progressive learning from simple and clean
samples to hard ones.
• We integrated the cross-modal contrastive learning to
address the modality gap in VI-ReID. Specifically, we
employ the confidence weight of each sample and force
the model to focus on sample pairs with clean labels for
robust cross-modal learning.
• Our method leverages historical memorization of models
as regularization. Through self-refined label rectification,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

7142

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 5, MAY 2026

mance [44]; Sun et al. [45] and Wang et al. [46] addressed
the issue by utilizing local detail information to segment
global features and extract multi-granularity features. They
enhance the models’ ability to distinguish identities with the
concatenated features. A group of works [47], [48], [49], [50],
[51], [52] also use image parsing and the attention mechanism
to extract fine-grained features and match corresponding body
parts, obtaining aggregated multi-grained features to improve
the model’s performance.
Nonetheless, in real-world scenarios, relying solely on a
single visible modality is inadequate, particularly in low-light
conditions where visibility is severely compromised.

B. Visible-Infrared Person Re-Identification

Fig. 2. (a) Previous methods for noisy labels rely on accurate noise detection
to compute sample confidence, but they overlook the value of noisy samples,
thus underutilizing available data. (b) Our method uses momentum-updated
memories (for historical relevance) to stably predict the labels of noisy
samples. By accurate label rectification, our approach enables the model to
undergo comprehensive training with the refined dataset.

our pipeline progressively corrects the labels of noisy
samples and restrains them in the early learning stage
to offer stable and accurate predictions.
• Our method exhibits robust performance on SYSUMM01 and RegDB datasets across a range of noise
ratios, achieving new SOTA in the realm of learning with
coupled noisy labels for VI-ReID. Extensive experiments
demonstrated the effectiveness of each module in our
approach.
II. R ELATED W ORK
A. Deep Person Re-Identification
Deep person re-identification focuses on retrieving images
of individuals with the same identity across cameras using
features extracted by the deep neural networks [34], [35]. The
key to this task is learning identity-aware concepts without
the influence of environmental factors. In recent years, deep
person re-identification has experienced significant progress,
driving advancements in this domain [5], [36], [37]. Global
representations of pedestrian images have been the primary
approach [38] when deep learning was introduced into the
field.
However, the person-identification task is characterized
by inter-identity similarities and intra-identity distinctions.
Similar to fine-grained image recognition [39], [40], objects
exhibit high similarity in overall appearance but differ in
subtle features. In person re-identification, details like clothing
and texture are indispensable in improving retrieval performance [41], [42], [43]. Therefore, fully understanding and
distinguishing the subtle visual differences between objects is
crucial for this task. To this end, input images are processed
through multiple branches for more refined retrieval perfor-

To overcome the limitation of visible images in varying
environments and conditions, images captured by infrared
cameras are integrated into person re-identification systems
for cross-modal learning [53]. This integration harnesses the
strengths of both modalities, ultimately enhancing model performance across a diverse range of scenarios.
The complementary nature of visible and infrared modalities has sparked increasing interest in cross-modality person
re-identification among researchers. To address the inherent modality discrepancies present in Visible-infrared ReIdentification (VI-ReID), a variety of methods [5], [54], [55],
[56], [57], [58] have been proposed, yielding notable advancements in performance on collected datasets. These methods
can be classified into two groups according to their purposes:
image-level augmentations and feature-level alignment. 1) The
image-level input of the intermediate modality can effectively
assist the model in capturing the connections between the
visible and infrared modalities. Data augmentations [3], [13]
and generative networks [59] have been employed to generate
intermediary modality images for joint training. This approach
encourages the model to learn modality-sharing identity information, bridging the gap between visible and thermal views. 2)
From the perspective of multi-modal feature alignment, metric
learning methods are employed to train the model to obtain
discriminative identity representations. Given the fine-grained
nature of ReID, relying solely on global features leads the
model to overlook cross-modal local information matching.
To address this issue, DFLN-ViT [14] introduced a crossmodal matching approach incorporating part and location
information to address this issue. Recent research [15], [16],
[60] considered structural and shape information, formulating
frameworks to extract disentangled and modality-irrelevant
features.
However, the effectiveness of these methods relies heavily
on precise data annotations, which are resource-intensive to
obtain. Moreover, pedestrian images in datasets are commonly
captured by cameras with varying parameters under real-life
scenarios, which might lead to occlusion and blurred images.
Incorrect annotations easily occur, particularly in the case
of infrared images, where color information is not available.
Consequently, solving the problem of learning with noisy
labels in VI-ReID is significant.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.


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
/bin/zsh -lc "pdftotext -l 3 'Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209289

Semantic Alignment and Hard Sample Retraining
for Visible-Infrared Person Re-Identification
1

Jingchen Ni1,* , Keyu Lyu1,* , Yu Guo2 , Chun Yuan1,†
Shenzhen International Graduate School, Tsinghua University, 2 CEC GienTech Technology Co.,Ltd.
*
{njc24, lvky24}@mails.tsinghua.edu.cn, † yuanc@sz.tsinghua.edu.cn

Abstract—Visible-Infrared Person Re-Identification (VI-ReID)
seeks to match individuals across different modalities. Recent
methods focus on discriminative feature extraction and hard
sample learning. However, they often suffer from semantic
misalignment due to horizontal partitioning in local feature extraction and overlook global hard samples in training. Moreover,
the widely used PK Sampler cannot ensure viewpoint balance and
diversity. To overcome these limitations, we propose the Semantic
Alignment and Hard Sample Retraining (SAHSR) framework.
This framework incorporates a Recurrent Semantic Aggregation
(RSA) module that progressively aggregates and aligns regional
semantics with the help of Modality Alignment loss. Besides, we
propose a Confidence-based Hard Sample Retraining (CHSR)
strategy that identifies and retrains hard samples to improve the
model’s robustness. Additionally, we introduce the ViewpointBalanced (VB) Sampler to guarantee a balanced distribution
of viewpoints. Extensive experiments on VI-ReID benchmarks
demonstrate the significant performance gains of our approach,
showing state-of-the-art performance. Code will be available.
Index Terms—Visible-Infrared Person Re-Identification, Semantic Alignment, Hard Sample Learning, Viewpoint Balance

I. I NTRODUCTION
Person re-identification (ReID) is a pedestrian retrieval
task that matches individuals across multiple non-overlapping
cameras, essential for tracking, security, and forensics in
video surveillance. While visible-spectrum ReID [1], [2] has
advanced under good lighting conditions, its performance
declines in low-light or nighttime scenarios due to limited
discriminative features. To address this, visible-infrared ReID
(VI-ReID) integrates infrared imagery, enabling recognition in
challenging lighting environments by extending ReID across
visible and infrared modalities. However, this introduces intermodality discrepancies caused by the inherent differences
between visible and infrared data.
These cross-modal gaps are further compounded by several
additional challenges. In particular, semantic misalignment
arises when corresponding body regions fail to spatially align
across modalities due to variations in pose, viewpoint, and
imaging characteristics. Existing methods [3]–[5] primarily
utilize horizontal partitioning for feature extraction, treating
patches independently and thereby overlooking broader contextual cues necessary for precise alignment. This exacerbates
semantic misalignment, and results in subtle yet critical mismatches that diminish the discriminative power of the learned
features, as illustrated in Figure 1(c).
* Equal Contribution. † Corresponding Author.

Fig. 1.
Framework Comparison: (a) Our method incorporates CHSR
Strategy, enhancing robustness by retraining on hard samples, unlike previous
methods that rely solely on loss calculation without additional retraining steps.
(b) Rectangle colors indicate different camera labels. The Viewpoint-Balanced
(VB) sampler ensures balanced sample distribution across camera views,
unlike the conventional PK sampler. (c) For feature extraction, our method
progressively aggregates local features using RNNs, effectively avoiding
semantic misalignment and ensuring better alignment of features across the
entire feature map.

Compounding these alignment issues is the underemphasis on globally challenging instances. Although triplet-based
losses [6], [7] are designed to emphasize hard samples, their
scope is generally confined to the mini-batch level and often neglects hard samples scattered throughout the dataset.
Without a more holistic mechanism to identify and revisit
these global complex cases, the model remains vulnerable to
similar ambiguities and fails to fully exploit the discriminative
potential of the entire dataset.
A further issue is viewpoint imbalance, where uneven
camera viewpoints bias the model’s representations. Although
PK samplers [6] include both visible and infrared samples in
each batch, they do not ensure balanced viewpoints, as shown
in Figure 1(b). This leads to overfitting common viewpoints
and limits generalization to rare ones.
Taken together, these considerations highlight the necessity
of methods that not only achieve more coherent semantic
alignment but also systematically emphasize globally challenging samples and ensure a balanced distribution of viewpoints.
Addressing these issues is critical for advancing VI-ReID toward more robust, discriminative, and generalizable solutions.
To address these challenges, we propose the Semantic

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.

Alignment and Hard Sample Retraining (SAHSR) framework,
comprising three components: the Recurrent Semantic Aggregation (RSA) module, the Confidence-based Hard Sample
Retraining (CHSR) strategy, and the Viewpoint-Balanced (VB)
Sampler.
The RSA module improves feature extraction by progressively aligning regional semantic features across the feature
map using a patch-based RNN, assisted by the Modality Alignment Loss to address semantic misalignment. The CHSR strategy selects hard samples globally based on confidence scores
and retrains them, enabling the model to better differentiate
between a sample and its hard negative samples. Meanwhile,
the VB Sampler balances sampling across cameras in both
modalities, reducing viewpoint discrepancies and improving
the model’s stability.
In summary, this paper makes the following contributions:
1) We introduce the SAHSR framework to simultaneously address semantic misalignment, hard sample learning, and viewpoint imbalance in VI-ReID. 2) We propose RSA, CHSR, and
VB components that respectively promote semantic alignment,
ensure thorough re-exposure to globally challenging samples,
and balance camera viewpoints. 3) Extensive experiments on
two standard VI-ReID benchmarks demonstrate that SAHSR
significantly outperforms state-of-the-art methods, confirming
its effectiveness and practical value.
II. R ELATED W ORK
Visible-Modality ReID. Person re-identification (ReID) in
the visible spectrum aims to match individuals across multiple non-overlapping cameras. Early approaches relied on
handcrafted features like SIFT [8] and metric learning [9],
which lacked robustness. The advent of deep learning, including CNN-based and transformer-based models [2], [10],
significantly enhanced discriminative representation learning.
However, these methods still struggle under poor illumination.
Visible-Infrared ReID. Visible-Infrared ReID (VI-ReID) incorporates infrared imagery to mitigate illumination issues,
facilitating recognition under varying lighting conditions. Approaches often focus on modality-invariant representations to
bridge the visible-infrared gap. For instance, Fang et al. [11]
and Zhang et al. [12] prioritize robust feature alignment, while
Ren et al. [13] use contrastive learning to reduce modality
discrepancies. Despite these advancements, existing VI-ReID
methods often overlook fine-grained semantic misalignments,
fail to address globally challenging samples, and do not ensure
balanced viewpoint distribution, highlighting the need for
more comprehensive solutions.
III. PROPOSED METHODS
The overall flow of SAHSR is shown in Figure 2. Given a
set of visible samples and infrared samples, the VB Sampler
is first used to sample P IDs, with each ID having K images,
forming a mini-batch of viewpoint-balanced samples (X m , Y )
where m ∈ {V, I} denotes the modality, with V representing
visible and I representing infrared, for training. After the
preprocessing step, a backbone network is exploited to extract

Fig. 2. The framework of the proposed SAHSR consists of three key
components: the VB Sampler, RSA Module, and CHSR Strategy.

a set of feature maps F m . Then the RSA module is utilized
to obtain global and local representations from the feature
maps for the subsequent ReID task. Finally, the CHSR strategy
selects hard samples based on confidence scores and retrains
them to further improve the model’s performance.
A. Recurrent Semantic Aggregation
The Recurrent Semantic Aggregation (RSA) module enhances feature extraction by progressively aggregating local
patch information via a BiLSTM. Unlike methods that simply
treat patches independently or rely on global attention, the
LSTM effectively models these patches as a short sequential
series, preserving their inherent spatial order. This ordersensitive modeling helps capture local dependencies—such
as the natural top-to-bottom arrangement of human body
parts—leading to more coherent and discriminative feature
representations. In contrast, Transformer-based approaches,
though powerful in global context modeling, may dilute such
local sequential cues, especially when dealing with a small
number of patches. By leveraging LSTM’s strength in handling
short sequences and local dependencies, RSA provides richer
contextual information that benefits the subsequent ReID tasks.
Visible samples are used to illustrate the process for ease of
explanation.
Specifically, after processing sample xV through the backbone, feature maps F V are obtained. To facilitate patchlevel analysis, the feature map is divided into n patches
by splitting it into a grid of size nh × nw , where nh and
nw represent the number of patches along the height and
width dimensions, respectively, satisfying n = nh × nw . Each
patch F V (k) corresponds to a spatially contiguous region of
the original feature map. For each patch, adaptive average
pooling (AAP) is applied to aggregate its spatial information
into a compact representation, yielding patch-wise features
f V (i) = avg(F V (i)), where f V (i) ∈ Rc .
Subsequently, a Bidirectional Long Short-Term Memory
(BiLSTM) network is employed to aggregate the local features, effectively capturing the interrelationships among them.
oV = BiLSTM(f V (1), f V (2), . . . , f V (n)),

(1)

where oV = [oV1 , oV2 , · · · , oVT ] ∈ RT ×2c represents the BiLSTM output at each time step, and T denotes the sequence
length. We utilize the output of the last time-step oVT to

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.

V
calculate the final aggregated local feature, fpart
= W (oVT ),
where W is the learnable parameters of the fully-connected
layers.
To capture global features, global average pooling (GAP) is
applied to F V . The resulting global features are concatenated
V
with fpart
to obtain fˆV as shown in (2).
V
fˆV = [fpart
, GAP(F V )],

(2)

where fˆV combines both global and local features. A
similar process is applied to the infrared modality to extract
fˆI from F I . These fused features, fˆV and fˆI , are then passed
through a dual BNNeck [11] for normalization. This step follows the calculation of the Modality Alignment Loss, ensuring
that the features from the visible and infrared modalities are
effectively aligned.
To further reduce the inter-modality discrepancy, we introduce a custom Modality Alignment Loss (Lma ), which
is designed to progressively minimize the Kullback-Leibler
(KL) divergence between the feature probability distributions
of the two modalities at each time step. Unlike static alignment strategies, this approach incorporates a time-dependent
weighting factor g(t), which increases linearly over time. This
design prioritizes stronger alignment constraints during the
later stages of the sequence, as we hypothesize that aligning
modalities at these stages is crucial for effective modality
integration.
Specifically, we first calculate the inter-modality distance
matrix DtV I using the Euclidean distance between oVt and oIt :
DtV I = ∥oVt − oIt ∥2 ,

DtIV = (DtV I )T

(3)

Next, we compute the probability distributions AVt I and
by applying the softmax function to the negative distance
AIV
t
matrices DtV I and DtIV , respectively. The resulting probability
distributions are then used to formulate the Modality Alignment Loss Lma , as shown in Equation (4).

Lma =

T
−1
X


g(t) KL(AVt I ∥ AIV
t ) ,

g(t) =

t=0

t
, (4)
T −1

where KL(AVt I ∥ AIV
t ) represents the KL divergence between
AVt I and AIV
,
and
g(t) is the linear weighting factor that
t
increases from 0 to 1 over the sequence. By assigning higher
weights to later steps, we enforce a stronger inter-modality
alignment when the local contextual cues are more consolidated. Ablation studies in Section IV-D show that this dynamic
weighting outperforms uniform weighting schemes, validating
our time-dependent alignment strategy.
B. Confidence-based Hard Sample Retraining
Previous research [6], [7], [19] has demonstrated that enhancing a model’s ability to discriminate hard samples significantly improves its overall performance. Unlike traditional
hard sample mining techniques that operate at the mini-batch
level, our Confidence-based Hard Sample Retraining (CHSR)
strategy leverages confidence scores to dynamically identify

hard samples across the entire dataset. This global focus
on challenging instances enhances the model’s generalization
capabilities.
To implement CHSR, we divide the training process into
standard training and retraining phases. During standard training, after the RSA module, feature vectors fˆm are generated
and fed into a classifier that produces confidence scores
S ∈ R(P ×K)×N , where P is the number of identities in a
batch, K is the number of images per identity, and N is the
total number of classes (i.e., the total number of IDs in the
training set).
We organize the batch such that the images of each ID
are grouped consecutively: the first K rows correspond to the
first ID, the next K rows to the second ID, and so forth. To
reference the confidence vector for the j-th image of ID i, we
define an indexing function:
r(i, j) = (i − 1) × K + j,

(5)

where i ∈ {1, . . . , P } and j ∈ {1, . . . , K}.
With this notation, S(r(i, j)) ∈ RN denotes the confidence
score vector for the j-th image of ID i. To obtain the mean
confidence score vector for ID i, we compute:
K

S(i) =

1 X
S(r(i, j)),
K j=1

(6)

where S(i) ∈ RN represents the mean confidence score over
the K images of the i-th ID.
Let c ∈ {1, . . . , N } index the classes. Then S(i)[c] denotes
the average probability that ID i is classified as class c. If
certain classes c ̸= i attain relatively high values in S(i),
this indicates that ID i is prone to be misclassified as these
classes, suggesting that images of these classes may serve as
hard negative samples for ID i. We define the set Cm (i) as the
m classes with the highest values of S(i)[c] (including c = i).
This set comprises ID i and its highly similar IDs, which will
be used for further retraining.
For each identity i in the standard training batch of P
identities, we define a Hard Sample Batch Hi , which contains
m × K̂ images. Specifically, for each ID i, we first select
all m IDs from the set Cm (i). Then, for each of these IDs,
we randomly select K̂ images. The collected images form the
batch Hi :
Hi = {xc,j | c ∈ Cm (i), j = 1, . . . , K̂}.

(7)

Since there are P identities in the standard training batch,
this process results in P distinct Hard Sample Batches
{H1 , H2 , . . . , HP }, each tailored to the corresponding identity’s challenging hard negative samples.
In the retraining phase, the P Hard Sample Batches are
concatenated into a unified batch and fed into the network.
To specifically enhance the model’s ability to distinguish each
identity from its hard negatives, we introduce two additional
loss functions: the Hard Batch Identity Loss (Lhid ) and
the Hard Batch Center Separation Loss (Lhcs ), which are
computed on Hi .

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10890656

Semantic Consistency And Integrity Network For
Cloth-changing Person Re-identification
Anqi Wang

Liyan Zhang*

School of Computer Science and Technology
Nanjing University of Aeronautics and Astronautics
Nanjing, China
aqwang@nuaa.edu.cn

School of Computer Science and Technology
Nanjing University of Aeronautics and Astronautics
Nanjing, China
zhangliyan@nuaa.edu.cn

Abstract—Cloth-changing Person Re-identification aims to retrieve target pedestrians across different cameras under clothingchanging scenarios. In recent years, many scholars have made
significant explorations in this field. However, existing methods
often overlook the semantic consistency and integrity of features.
To address this issue, we design a Semantic Consistency and
Integrity Network (SCI-Net) to learn semantically invariant
features and strip clothing bias from identity features while
maintaining their semantic integrity. The network consists of
three branches: clothing branch, raw image branch, and head
feature enhancement branch. Specifically, we first propose a
Head Soft Attention Generation Module to produce head soft
attention, thereby obtaining enhanced head features. Then, to
ensure that raw features can effectively learn invariant semantic
information from head-enhanced features, Semantic Consistency
Constraint is proposed to facilitate mutual learning between
the two branches. Finally, we leverage knowledge transfer to
enable clothing branch to perceive clothing bias entangled with
raw features and simulate causal intervention to quantify and
remove clothing bias. Experiments on the LTCC-ReID and PRCC
datasets demonstrate that our model outperforms other state-ofthe-art methods.
Index Terms—cloth-changing person re-identification, causal
intervention, semantic consistency.

I. I NTRODUCTION
Person re-identification aims to identify and retrieve the
same target pedestrian from videos or images recorded by
various cameras at different times and locations, which has
significant application value in fields such as smart security,
criminal investigation, and smart shopping [12], [19], [22],
[35], [37]. In recent years, person re-identification methods
has achieved significant progress [14], [17], [28] [4]. General
person re-identification algorithms mainly focus on short-term
Re-ID scenarios [11], [16], [29], with an impractical assumption that people would not change their clothes. Therefore,
they heavily rely on appearance features of the pedestrians,
especially salient colour and texture information of clothes.
However, in long-term Re-ID scenarios as shown in Fig. 1,
clothing may exhibit considerable variability over time .
Additionally, different people may wear the same or similar
clothing.
* Corresponding Author.
+ This work was supported in part by the National Natural Science
Foundation of China under Grant 62172212 in part by the Natural Science
Foundation of Jiangsu Province under Grant BK20230031.

Short-time ReID with
clothes consistency

Day1

Query

Clothes consistency result

Long-time ReID with
clothes inconsistency

Day2

Day3

Clothes inconsistency result

Fig. 1. The difference between the short-term Re-ID and the long-term ReID. In short-term Re-ID scenarios, people hardly change their clothes. While
in long-term Re-ID scenarios, clothing may exhibit considerable variability
over time and retrieval results should include both clothes consistency results
and clothes inconsistency results.

Recently, some scholars have also explored methods for
person re-identification under a significant but challenging ReID setting, Cloth-Changing Re-ID (CC-ReID) [2], [10], [25],
[27], [33], [3], [5]. Some existing CC-ReID methods use multimodality (information such as skeletons [25], radio signals [7],
faces [26], [31], gaits [6], [21], etc.) to model discriminative
biological features. However, these methods require additional
resources and models to capture multi-modality features. More
importantly, they ignore the semantic consistency of features,
which is critical for Re-ID.
Moreover, clothing bias entangled with identity features
can significantly affect the performance of CC-ReID models.
Although eliminating clothing bias is important, it is still not
clear how to eliminate clothing bias in the feature representation space. Some CC-ReID methods try directly covering up
clothing or simply ignoring it, e.g., the mask-based methods
[15], [36] and gait-based methods [6], [21]. Although these
methods are effective, they compromise semantic integrity.
To address the above issues, we design a Semantic Consistency and Integrity Network (SCI-Net). As shown in Fig. 2,
SCI-Net consists of three branches: a clothing branch, a raw
image branch, and a head feature enhancement branch. There
are two fundamental ideas of our method: 1) Promote model

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:40 UTC from IEEE Xplore. Restrictions apply.

focus on semantically invariant regions and thereby maintain semantic consistency features. 2) Eliminating clothing
bias while maintaining the semantic integrity of the features.
Firstly, to fully exploit discriminative features of raw images,
Head Soft Attention Generation Module (HSAGM) is designed
in the head feature enhancement branch. This module learns
to generate head soft attention from the body masks obtained
through human parsing by utilizing the proposed Body Part
Matching Loss. Secondly, Semantic Consistency Constraint
(SCC) is proposed which utilizes class activation maps and
saliency maps to achieve mutual learning of raw features
and head-enhanced features at high semantic level, enabling
semantic alignment between the two branches. Finally, to eliminate clothing bias, we design two loss functions: Knowledge
Transfer Loss and Causal Intervention Loss. The former loss
aims to enhance the perception of clothing features entangled
with pedestrian features in the raw image branch, while the
latter one focuses on quantifying effect of clothing bias and
eliminate it.
The main contributions of our work are as follows:
• We propose a Semantic Consistency and Integrity
Network (SCI-Net) for Clothing-Changing Person Reidentification (CC-ReID). The designed SCI-Net can extract semantically invariant identity-related features and
eliminate clothing bias without compromising the semantic integrity of pedestrian features.
• We design a Head Soft Attention Generation Module,
which utilizes body part matching loss and gets head soft
attention from body part masks. Additionally, Semantic
Consistency Constraint is proposed to facilitate mutual
learning between the raw image branch and the head feature enhancement branch, achieving semantic alignment
between the two branches.
• Knowledge transfer and causal intervention are applied
between raw image branch and clothing branch. Clothing
bias can be quantified and stripped away while the
semantic integrity of features can be maintained.
• Experiments on two public CC-ReID datasets, LTCCReID and PRCC, demonstrate that our proposed SCI-Net
outperforms the state-of-the-art CC-ReID methods.
II. M ETHODOLOGY
A. Overall Framework
As mentioned above, some existing CC-ReID models utilize
multi-modality information as auxiliary cues for retrieval and
bring a considerable amount of time and resource. Additionally, some methods compromises semantic integrity of
features to mitigate clothing bias. To better learn semantically
invariant and integrated features from pedestrian images, we
propose a Semantic Consistency and Integrity Network (SCINet). As illustrated in Fig. 2, SCI-Net consists of three
branches: a clothing branch, a original image branch, and a
head feature enhancement branch. First, we design a Head Soft
Attention Generation Module that utilizes body part matching
loss Lmatch to obtain head-enhanced features. Moreover, to

lead model’s attention to semantically invariant regions, we
introduce Semantic Consistency Constraint LSCC which can
promote mutual learning between raw image features Fraw
and head-enhanced features Fhead . In the clothing branch,
we extract local clothing features FC . Finally, we enhance
the original image branch’s perception of clothing features
through Knowledge Transfer Loss LKT and simulate causal
intervention between clothing branch and raw image branch,
quantifying the effect of clothing bias into causal relationship
loss LCI .
B. Head Soft Attention Generation Module
The Head Soft Attention Generation Module, as shown in
Fig. 2 a), takes raw image features Fraw and body parsing
results as inputs, utilizing body part matching loss Lmatch to
learn and obtain the head soft attention.
First, we generate human parsing results by SCHPNet [23].
Each parsing result contains 18 labels, including hair, face,
coat, skirt, left/right leg, left/right arm, and so on. We set
pixel values belonging to head region in the parsing results to
1, while assigning 0 to pixels in other regions, thus obtaining
head mask. Similarly, we generate masks for upper body, lower
body, and feet using the same method. In this way, we obtain
body part masks M ∈ RH×W ×K . Next, Fraw ∈ RH×W ×C
are input into a 1 × 1 convolution layer with a softmax
activation function to produce attention maps A ∈ RH×W ×K .
H, W and C represent height, width and number of channels,
respectively. K represents the number of body parts, which is
set to be 4, corresponding to head, upper body, lower body
and feet. For AK , value of a pixel denotes probability that
current pixel position belongs to the k-th body part. Finally,
body part matching loss Lmatch is proposed to optimize the
learning of body part attention maps A from body part masks
M , which can be defined as:
N K H−1 W −1
1 1 1 XX X X
Mk (h, w) log (Ak (h, w)) ,
N H W n=1
k=1 h=0 w=0
(1)
N represents batch size. MK represents value of k-th body
part mask at (h, w). And AK represents value of k-th attention
maps at (h, w). Furthermore, the learning process of body part
attention maps is supervised under identity loss (cross-entropy
loss). Thus, compared to the results from body parsing, body
part attention maps generated by this module are more relevant
to CC-ReID.

Lmatch = −

C. Semantic Consistensy Constraint
To enable the raw image branch to fully learn the headenhanced features and enhance the model’s attention to semantically invariant regions, a Semantic Consistency Constraint
(SCC) is designed, as illustrated in the figure Fig. 3.
Specifically, Fraw and Fhead are firstly processed by a
batch normalization layer. And then corresponding results are
input into a 1 × 1 convolution layer, and we can obtain two
class activation maps Graw and Ghead ∈ RI×H×W where I

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:40 UTC from IEEE Xplore. Restrictions apply.

a) Head Soft Attention Generation Module(HSAGM)

Clothes
Encoder

Clothes
Classifier

Conv1×1 + Softmax

Assist
Clothes
Classifier

0
0
0
0
0

ID-Clothes
Classifier

0
0
0
0

ID
Encoder

ID
Classifier

0
0
0
0
0
0
0

00
00
00
00
00
10
1

00
10
10
00
00
00
0

0
0
01
01
00
00
00
10
1

01
11
10
00
00
00
0

0
0
01
01
00
00
00
00
0

01
01
00
00
00
00

0
0
0
0

0
0
0

0
0
0
0
0
0

0

0

b) Causal Relationship and Causal Intervention

Causal Intervention

Feature

Feature

Human Parsing

ID
Classifier

HSAGM

Input

Output

Intervention

Output

Fig. 2. Overall framework of the proposed Semantic Consistensy and Integrity Network (SCI-Net). It consists of three branches: a raw image branch, a head
feature enhancement branch, and a clothing branch. Features of raw image branch are fed into the head soft attention generation module to generate head soft
attention by utilizing body part matching loss and enhance head features. The Semantic Consistency Constraint LSCC facilitates raw image branch to learn
semantically invariant identity-related features. Finally, we simulate causal intervention and eliminate clothing bias between raw image branch and clothing
branch under the supervision of the Knowledge Transfer Loss LKT and Causal Intervention Loss LCI .

AvgPool

1×1 Conv

MSE Loss

ID Label

AvgPool

1×1 Conv

Batch Norm

N
2
2 i
1 Xh
LSCC =
g − Eraw + g − Ehead
.
N n=1

ID Label

Batch Norm

represents the number of identities. Since the class activation
map represents the model’s attention to each region when
distinguishing pedestrians, we select maps corresponding to
ground truth identity label in the channel dimension. Then,
we compare pixel values of the two feature maps, selecting the maximum value at each pixel location to obtain a
more effective supervision signal. We denote this signal as
g ∈ RH×W . On the other hand, we apply average pooling on
Fraw and Fhead in the channel dimension to obtain saliency
maps Eraw and Ehead ∈ RH×W . Saliency maps essentially
indicate the focused areas of the network. To maintain the
semantic invariance of pedestrian features, we extend method
in [13] and impose a Semantic Consistency Constraint LSCC ,
which can be defined as:

MSE Loss

Fig. 3. Illustration of Semantic Consistency Constraint LSCC . ID Label
reperesents the true identity label.

and optimize it by using clothing classification loss. The
clothing bias is extracted by this classifier and transferred to
the clothing branch for further knowledge transfer. We can
calculate the KL distance from FC to Fraw as follows:

(2)

D. Causal Intervention
To tackle with CC-ReID task better, causal intervention is
introduced in this work to remove the clothing bias entangled
with pedestrian features. We first extract local clothing features
FC in the clothing branch through a clothes classifier δC by
adopting pyramid matching strategy [8]. To enable clothing
branch to perceive clothing bias entangled with raw features,
we use Kullback-Leibler (KL) divergence to fit the distribution
of clothing bias in pedestrian features. Specifically, we add
an assistant clothes classifier δA to the raw image branch

p̂C = exp (δC (F C )) ,

p̂raw = exp (δA (F raw )) ,

DKL (p̂C ∥p̂raw ) =

N
X
i=1

p̂m
C log

p̂m
C
.
p̂m
raw

(3)
(4)

Due to the asymmetry of KL divergence, we calculate
DKT (p̂raw ∥p̂C ) as well. And the total knowledge transfer loss
LKT is the sum of the two KL distance.
Following [30], the impact of clothing bias on final prediction results can be quantified by the difference between actual
prediction results and predictions obtained after intervention as
shown in Fig. 2 b), that is Yef f ect = YX,C − YC . To enhace
the representation of same-located bias, we perform fusion of

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:40 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Shape-centered representation learning for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
前 5 篇已经读到核心信息。它们共同的发法很清楚：把一个训练假设缺陷说成实际数据流问题，例如噪声标签、硬样本未回访、语义误对齐、形状估计不可靠、弱监督假负样本，而不是单纯堆模块。
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
/bin/zsh -lc "pdftotext -l 3 'Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


 succeeded in 0ms:
1

Spatial-Temporal Federated Learning for Lifelong
Person Re-identification on Distributed Edges

arXiv:2207.11759v2 [cs.LG] 11 Dec 2024

Lei Zhang, Guanyu Gao, and Huaizheng Zhang

Abstract—Data drift is a thorny challenge when deploying
person re-identification (ReID) models into real-world devices,
where the data distribution is significantly different from that
of the training environment and keeps changing. To tackle this
issue, we propose a federated spatial-temporal incremental learning approach, named FedSTIL, which leverages both lifelong
learning and federated learning to continuously optimize models
deployed on many distributed edge clients. Unlike previous efforts, FedSTIL aims to mine spatial-temporal correlations among
the knowledge learnt from different edge clients. Specifically,
the edge clients first periodically extract general representations
of drifted data to optimize their local models. Then, the learnt
knowledge from edge clients will be aggregated by centralized
parameter server, where the knowledge will be selectively and
attentively distilled from spatial- and temporal-dimension with
carefully designed mechanisms. Finally, the distilled informative
spatial-temporal knowledge will be sent back to correlated edge
clients to further improve the recognition accuracy of each edge
client with a lifelong learning method. Extensive experiments on a
mixture of five real-world datasets demonstrate that our method
outperforms others by nearly 4% in Rank-1 accuracy, while
reducing communication cost by 62%. All implementation codes
are publicly available on https://github.com/MSNLAB/FederatedLifelong-Person-ReID.
Index Terms—Federated learning, lifelong learning, person reidentification, spatial-temporal knowledge mining.

I. I NTRODUCTION

P

ERSON re-identification (ReID) aims to retrieve people
appearing at different locations and moments from the
over-lapped cameras. The deep learning-based approaches for
person ReID can achieve promising performance on popular
benchmarks [1], [2], which enables the applications of person
ReID in many computer vision-based applications, such as
urban analysis, suspect tracking, and city surveillance.
The deployment of person ReID in real-life still suffers
from many great challenges. One prevalent challenge is that
the recognition accuracy of the person ReID models will
decrease, with the changing of camera environments. This
is mainly because of the domain mismatch between the
training and deployment environments. Specifically, the person
ReID models are usually pre-trained on given datasets which
consist of images of a fixed set of person identities captured
L. Zhang and G.Y. Gao are with School of Computer Science and
Engineering, Nanjing University of Science and Technology, Nanjing
210094, China. Email: {lei.zhang, gygao}@njust.edu.cn. (Corresponding author: Guanyu Gao).
H.Z. Zhang is with School of Computer Science and Engineering, Nanyang
Technological University, Singapore. Email: huaizhen001@e.ntu.edu.sg.
Copyright © 20xx IEEE. Personal use of this material is permitted.
However, permission to use this material for any other purposes must be
obtained from the IEEE by sending an email to pubs-permissions@ieee.org.

𝑡𝑡0

𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪 1

𝑡𝑡1 𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪 𝟏𝟏

𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪 2

𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪 𝟐𝟐

𝑡𝑡2

𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪 3

𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪 𝟑𝟑

𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻𝑻
𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪

𝑺𝑺𝑺𝑺𝑺𝑺𝑺𝑺𝑺𝑺𝑺𝑺𝑺𝑺
𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪𝑪

Fig. 1. The spatial-temporal correlations among the data of different edge
clients. The image data captured by different cameras have spatial-temporal
correlations, and the edge clients can utilize the spatial-temporal knowledge
from others for federated learning to continuously improve their performances.

in specific camera environments. However, in the real-life
person ReID deployment, thousands of newly images that
are captured every moment often involve many new person
identities, which are unavailable at the model training stage.
Meanwhile, the camera environments are dynamic and everchanging due to influences brought by many reasons, such as
illumination changing and varying camera views. The domain
gap between the training and inference environment limits
the performance of person ReID in real-world deployment [3]
Some preliminary studies [4], [5] also simulated data drift by
training person ReID model on Market-1501 dataset [6] and
testing on MSMT17 dataset [5]. These studies observed that
model with 73.15% training mAP can only achieve 4.63% test
mAP, a reduction of almost 68% due to the domain changes.
Another challenge for person ReID is to preserve the privacy
of the person images [1]. The person images contain sensitive
private information, such as individuals’ identities, locations,
genders, ethnicity, and even facial features [7]. Sharing these
sensitive person images for model training and data analytics
is infeasible due to the potential risk of privacy leakage and
the expensive communication costs for data transmission. Besides, many EU/UK countries have issued privacy protection
regulations (e.g., GDPR [8]) to prohibit the centralization of
sensitive data from in-situ devices.
To address the domain drift for person ReID, some recent works (e.g., [3], [9]–[12]) adopted lifelong learning.
These works enable person ReID models to continuously
learn knowledge from new scenarios without forgetting previously learnt knowledge. HVIL [9] present a human-in-theloop paradigm for lifelong person ReID under interactive
manual feedback. AKA [3] and PTKP [12] generalised the

2

representation of lifelong person ReID for intra- and interdomains. GwFReID [11] alleviated forgetting under a classimbalance lifelong condition for person ReID. However, these
works require centralized training with the drifted data from
deployed devices to learn new knowledge, which also bring
data privacy concerns.
To alleviate privacy concerns, some recent works (e.g., [7],
[13]–[15]) adopted federated learning to jointly train models
on the edge clients. The federated learning-based approaches
enable the sensitive data to be utilized in situ [13]–[15],
while different edge clients can collaboratively update models
by aggregating their gradients or parameters [7]. FedDG
[15] generalized person ReID models to tackle cross-edge
domain mismatches in federated person ReID. FedPav [7] and
some other works [13], [14] adopted model aggregation and
distillation for the non-i.i.d federated person ReID. FedReID
[16] proposed an iterative client-server collaborative learning
to generalize person ReID models. SKA [14] proposed a selective knowledge aggregation method to transfer personalized
knowledge among different edge clients.
Prior works, however, considered the problems of continuously updating models and decentralized training models separately. They are still unable to support distributed edge clients
to continuously learn new knowledge while collaboratively
sharing their knowledge under privacy-preserving. Hence, we
first propose to judiciously combine both federated learning
and lifelong learning for person ReID. Moreover, we observe
that knowledge learnt from different locations and moments
has implicit spatial and temporal correlations [1]. As illustrated
in Fig. 1, pedestrians that appeared in the past often reappear
on other streets in the near future. We suppose the knowledge
learnt from one edge client may also be informative to other
neighbor clients shortly. However, previous works neglected
the spatial-temporal correlations for recognition knowledge
at different locations and moments. Therefore, they failed to
adaptively utilize the knowledge across spatial and temporal
spaces, and thus limit performance improvement.
We propose a Federated Spatial-Temporal Incremental
Learning framework, named FedSTIL, based on spatialtemporal knowledge integration for decentralized continuously
learning for person ReID. The edge clients utilize their arriving
drift data to optimize the local model for incremental domain
knowledge. Meanwhile, some general representations of the
drift data are periodically stored in local memory for future
rehearsal to alleviate the catastrophic forgetting. Next, the
parameter server integrates the incremental knowledge from
different edge clients based on the spatial-temporal correlation. Then, the parameter server delivers these informative
knowledge to the edge clients. Finally, the edge clients will
utilize both the integrated knowledge and previously learnt
knowledge to further improve the model with lifelong learning.
The main contributions of our paper are summarized as:
• Propose a federated lifelong person ReID framework,
which enables the distributed edge clients to continuously
learn incremental knowledge with collaboration.
• Design a spatial-temporal knowledge integration method
to transfer task-specific knowledge among edge models
to improve their performance.

Demonstrate the effectiveness of our framework via extensive experiments, ablation studies, and visualization.
• Release an open-source tool to facilitate the research for
federated lifelong person ReID.
The rests of this paper are organized as follows. Section II
introduces the related works, Section III presents the problem
definition and the system overview, Section IV illustrates the
learning methodology, Section V evaluates the performances
of our method, Section VI discusses some future directions,
and Section VII concludes the paper.
•

II. R ELATED W ORK
In this section, we first introduce some preliminaries of
person ReID, and then present the related works about person
ReID and federated lifelong learning.
A. Preliminary of Person ReID
Person ReID can retrieve a person from non-overlapped
camera views. The developments of deep neural network
and the large-scale person ReID datasets have significantly
improved the performances of person ReID in many visionbased applications [17]–[19]. Recently, many works [1], [17],
[20] focused on the data drift in real-world person ReID,
and revealed that there exist spatial-temporal correlations for
drifted data among different locations and moments. The
works that investigated the spatial-temporal correlations can be
categorized into two lines. One line of the works investigated
the temporal correlations of data drift, which mainly focus on
adopting domain adaptation [21], [22] and lifelong learning
[3], [9]–[11] to narrow domain mismatches across different
moments. The other line of works investigated the spatial
correlations of data drift, which mainly adopt knowledge
transfer [5], [7] and feature alignment [15], [23], [24] to
overcome the domain mismatches from different edges.
B. Lifelong Person ReID
In real-world person ReID deployments, camera environments and person characteristics are always different from that
of training data. To narrow the domain shifts for the training
and deployment stages, lifelong learning are preferred by
some recent works [3], [9]–[12]. Lifelong learning enables the
models to continuously learn from new domains or scenarios,
which has been widely adopted in many DNN-based serving
systems [25]–[28] to deal with the domain drift. The greatest
challenge for lifelong person ReID is catastrophic forgetting,
which requires ReID models to replay previous knowledge
while continually training on new task streams. HVIL [9]
introduced a human-in-the-loop incremental learning method,
which enables models to adaptively refresh and optimize parameters by human feedback on unrecognized person images.
AKA [3] proposed knowledge graph for lifelong person ReID,
which can preserve the knowledge from previous domains
while propagating learnt knowledge on unseen domains. PTKP
[12] cast lifelong person ReID as domain adaptation and
proposed a pseudo task knowledge preservation framework to
alleviate the domain gaps. GwFReID [11] proposed a classimbalance lifelong learning for person ReID to generalise
model representations for unseen domains.

3

Edge 1

Sampling

Adaptive Layer

Download

Rehearsal

Task

Time Axis

Prototypes

Extract Layer

Parameter Server
Upload

Edge 1

Distance

Edge 2
Edge N

…

Local Storage

…
Task Distance Space

Knowledge
Relevance

Model Aggregation

Edge N

Fig. 2. The architecture of FedSTIL for federated lifelong person ReID. The distributed edge clients continuously learn from both their local drift data and
the relevant spatial-temporal knowledge from other edge clients organized by the parameter server to improve recognition accuracy.

C. Federated Person ReID
With the growing data privacy concern, many person ReID
systems are changing to the decentralized or federated training
paradigm, where the private data are stored in isolated edges,
and edges can jointly update models, instead of centralising
private data for training. One challenge for federated person
ReID is knowledge interference due to the data and domain
heterogeneity across different edge clients. To address this
issue, FedDG [15] proposed domain and feature hallucinating techniques to train generalized person ReID models for
federated learning with domain heterogeneity. FedPav [7]
and some other works [13], [14] adopted model aggregation
and knowledge distillation to optimize the performances for
the non-i.i.d federated person ReID. FedReID [16] proposed
an iterative client-server collaborative learning to generalize
person ReID domains without sharing private data. SKA [14]
proposed a selective knowledge aggregation method to transfer
personalized knowledge among different edge clients. These
methods are successful, but they did not consider the spatialtemporal correlation for domain knowledge from different
edge locations and moments, which would limit the efficiency
of knowledge sharing. Hence, our method aims to capture taskspecific knowledge by integrating spatial-temporal knowledge
to improve performances.
D. Federated Lifelong Learning
Despite the rapid progresses of lifelong person ReID and
federated person ReID, few works studied lifelong person
ReID under the federated learning paradigm, which we term
as federated lifelong person ReID. Federated lifelong learning
[29], [30] can enable distributed models to collaboratively and
continuously learn incremental knowledge from new scenarios, without sharing their private data. To better explain the
federated lifelong setting, we summarize the main research
objectives in Table I, by comparing them with the lifelong
setting and federal setting. Specifically, federated lifelong person ReID has the merits of both lifelong and federated person
ReID. Besides, federated lifelong person ReID can also enable
both spatial- and temporal- knowledge transfer [1], which is
not simultaneously achievable for either the lifelong setting
or the federal setting. In this area, FedCurv [31] and CFeD

TABLE I
T HE COMPARISON FOR THREE DIFFERENT PERSON R E ID SETTINGS .
Person ReID Settings

Research Objectives

Lifelong
Continual Training [37]
Data / Domain Shifts [3]
Model Forgetting Alleviation [35]

Federal

✓
✓
✓

Decentralised Training [38]
Data / Domain Heterogeneity [15]
Privacy Protection [38]
Temporal Knowledge Transfer [1]
Spatial Knowledge Transfer [1]

Fed. + Life.
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

[32] mitigated the heterogeneity and forgetting by constraint
training for non-i.i.d federated lifelong learning. FCIL [33] addressed the class-imbalanced and class-incremental federated
lifelong learning by loss distillation. FedWeIT [34] proposed
sparse parameters to reduce the network cost for federated lifelong learning. These methods are successful, however, they are
not specific designed for the person ReID scenario, and leaves
a large room for improvement. For instance, FedCurv [31]
extended the regularization-based lifelong learning method
[35], [36] to the federated learning scenario, however, the
high communication cost for transferring the regularization
information is also an obstacle to apply it into the federated
lifelong person ReID. FedWeIT [34] utilized decomposed
layers to reduce the network consumption. It achieved great
performance for many incremental learning scenarios [37],
where task IDs are provided. However, it is not applicable
in the scenarios where task IDs are unknown, such as the
federated lifelong person ReID scenario. To the best of our
knowledge, we are one of the first works which studied
federated lifelong learning for person ReID.

III. S YSTEM D ESIGN
In this section, we first present the problem definition of the
federated lifelong person ReID, and then illustrate the system
overview and learning procedure. The main notations of the
paper are illustrated in Table II.


exec
/bin/zsh -lc "pdftotext -l 3 'SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
我只按要求读了每篇前 3 页。下面不是完整论文综述，而是从它们的发法里反推可用的新方向。

**逐篇提炼**
1. `SRHCL` 噪声 VI-ReID  
创新类型：问题重定义 + 新机制。把 VI-ReID 的标注噪声拆成单模态 noisy annotation 和跨模态 noisy correspondence。  
为什么能发：以前多数方法检测噪声后丢掉样本，它说噪声样本有历史预测价值，可以逐步修正。证据链是不同噪声率下的鲁棒性、标签修正、历史一致性消融。  
story 套路：不是“抗噪损失”，而是“跨模态耦合噪声下，历史预测比当前预测更可靠”。

2. `SAHSR` 语义对齐和困难样本重训  
创新类型：工程组合偏强，带一点机制新意。  
为什么能发：抓三个具体缺口：水平切块语义错位、mini-batch 难样本太局部、PK sampler 视角不平衡。证据链靠 RSA、全局 hard sample retraining、viewpoint-balanced sampler 分别消融。  
story 套路：把“训练采样和局部对齐不干净”讲成 VI-ReID 的系统性偏差来源。

3. `SCI-Net` 换衣 ReID 语义一致性与完整性  
创新类型：新机制，但已有换衣去衣服偏置框架痕迹重。  
为什么能发：不只说去衣服，而是说直接遮住衣服会破坏语义完整性，所以用头部增强、语义一致、衣服分支因果干预来剥离衣服偏置。  
story 套路：headline 是“去除衣服偏置，但不破坏行人语义完整性”。

4. `ScRL` 形状中心 VI-ReID  
创新类型：问题重定义 + 新机制。  
为什么能发：把 VI-ReID 从外观对齐转成“形状是天然跨模态稳定信息，但红外形状估计不准”。机制包括红外形状恢复、形状特征传播、形状引导外观增强。  
story 套路：不是“加 parsing”，而是“以形状为中心重建跨模态表征”。

5. `SRCA` 弱监督文本行人检索  
创新类型：新设定 + 新机制。  
为什么能发：没有身份标签时，一对一图文匹配会制造假负样本。它用软相似度关系和最优传输伪标签校准，缓解跨模态匹配歧义。  
story 套路：把“弱监督不是缺标签而已”，讲成“人检索中天然存在多对多局部相似”。

6. `FedSTIL` 联邦终身 ReID  
创新类型：新数据流设定 + 问题重定义。  
为什么能发：真实部署里模型在多个边缘端持续漂移，同时又不能集中数据。它把终身学习和联邦学习合成一个问题，并用时空相关性选择性迁移知识。  
story 套路：从 benchmark 训练转到真实部署数据流，卖点是“隐私约束下的持续适应”。

7. `STHF` 视频可见光-红外高频学习  
创新类型：新机制。  
为什么能发：现有中间模态要么生成成本高，要么丢高频，要么没有时间信息。它用 3D FFT 构造序列级高频中间模态，再分浅层细节补偿和深层语义细化。  
story 套路：跨模态差异不是只在语义层，频域里的时空高频才是稳定细节。

8. `FAViT` 视频 ReID 特征级增广 ViT  
创新类型：机制型工程组合。  
为什么能发：利用视频 ReID 两个事实，换背景身份不变，混入异常帧需要识别。用背景 token、空间特征增广、时间异常帧检测，让 ViT 学会前景和时序一致性。  
story 套路：不用加复杂网络，而是把视频 ReID 的不变性写进训练任务。

9. `SPCL` 文本行人检索语义多态和共性  
创新类型：问题重定义 + 新机制。  
为什么能发：它把图文检索的一对一对齐问题改写成语义多态和属性共性问题，同一 ID 内有弱对应，不同 ID 间有属性共性。证据链靠 ID 内多态对齐、ID-free 属性对齐、属性先验重建。  
story 套路：人检索不是一张图配一句话，而是“同一语义有多种表达，不同身份共享属性”。

**对我们最有用的规律**
这批能发的点，大多不是“模块更复杂”，而是抓住一个训练或数据流假设错误：标签不完全可靠、对应关系不是一对一、形状估计不可靠、困难样本不只在 batch 内、部署数据会持续漂移。我们要投 B 类，最好也这样写问题，而不是继续做普通 gating 或对齐模块。

**强创新点候选**
1. Headline：面向航拍到地面的 ReID，不再对齐 2D 外观，而是学习可见表面到 3D 人体规范面的对应关系。  
挂团队资产：aerial-ground + SMPL 3D 几何 + SOLIDER-Swin。  
区别：最像 `ScRL`，但它是 VI-ReID 的 2D parsing 形状恢复；我们做的是极端俯视和地面视角下的 3D mesh/UV 规范化，把“视角导致同一身体区域不可比”作为核心问题。  
cheap kill-switch：不用训练全模型，先在 CARGO 或 AG-ReID.v2 上跑现有 SMPL/pose，统计同 ID 跨视角的 3D 规范面局部特征距离是否显著小于异 ID；再做一个只加训练期 3D surface consistency 的 Tiny 小跑。如果零训练几何质量很差，或者简单蒸馏不改善跨视角 hard subset，就停。

2. Headline：航拍低清不是普通降分辨率，而是身份高频在视角和高度中系统性丢失，需要视角条件的频率补偿。  
挂团队资产：aerial-ground + pose 热图门控 + SOLIDER-Swin。  
区别：最像 `STHF`，但它是视频 VI-ReID 的 3D FFT 序列中间模态；我们做 aerial-ground 的高度和俯仰视角造成的频率不对称，用 pose 区域约束高频补偿，避免变成普通频域增强。  
cheap kill-switch：先做频谱诊断，比较同 ID 航拍和地面图在人体区域的高频能量差，检查它是否和检索错误强相关。再做一个训练期高频 dropout/restore 小实验，只看 aerial-ground 跨视角子集是否超过噪声线。

3. Headline：跨视角 ReID 的困难样本不是最相似负样本，而是 3D 姿态和可见表面造成的“几何假负样本”。  
挂团队资产：SMPL 3D 几何 + pose 热图门控 + aerial-ground。  
区别：最像 `SAHSR` 的 hard sample retraining，但它按分类置信度找全局难样本；我们按 3D 姿态、朝向、可见表面重叠度定义几何难度，专门解决航拍和地面视角下“看起来不一样的正样本”和“几何相似的负样本”。  
cheap kill-switch：离线算每对样本的 pose/SMPL 几何相似度，看 baseline 错误是否集中在几何相似负样本和几何不相似正样本。若错误不随几何难度分层，别做。

4. Headline：真实航拍地面系统里，相机和时间持续漂移，模型需要按空间时间相关性选择旧知识，而不是全量回放或统一蒸馏。  
挂团队资产：aerial-ground + 现有终身 ReID 基建 + SOLIDER-Swin。  
区别：最像 `FedSTIL`，但它是普通边缘 ReID 的联邦终身学习；我们可以收窄到 aerial-ground 场景，把无人机高度、路线、地面摄像头位置作为可解释的时空结构，主打跨视角部署漂移。  
cheap kill-switch：把 AG-ReID.v2 或 CARGO 按高度、摄像头、时间切成任务序列，测顺序微调遗忘是否和空间时间距离有关。若没有规律，只是普通 continual ReID，就不要硬写。
tokens used
72,569
我只按要求读了每篇前 3 页。下面不是完整论文综述，而是从它们的发法里反推可用的新方向。

**逐篇提炼**
1. `SRHCL` 噪声 VI-ReID  
创新类型：问题重定义 + 新机制。把 VI-ReID 的标注噪声拆成单模态 noisy annotation 和跨模态 noisy correspondence。  
为什么能发：以前多数方法检测噪声后丢掉样本，它说噪声样本有历史预测价值，可以逐步修正。证据链是不同噪声率下的鲁棒性、标签修正、历史一致性消融。  
story 套路：不是“抗噪损失”，而是“跨模态耦合噪声下，历史预测比当前预测更可靠”。

2. `SAHSR` 语义对齐和困难样本重训  
创新类型：工程组合偏强，带一点机制新意。  
为什么能发：抓三个具体缺口：水平切块语义错位、mini-batch 难样本太局部、PK sampler 视角不平衡。证据链靠 RSA、全局 hard sample retraining、viewpoint-balanced sampler 分别消融。  
story 套路：把“训练采样和局部对齐不干净”讲成 VI-ReID 的系统性偏差来源。

3. `SCI-Net` 换衣 ReID 语义一致性与完整性  
创新类型：新机制，但已有换衣去衣服偏置框架痕迹重。  
为什么能发：不只说去衣服，而是说直接遮住衣服会破坏语义完整性，所以用头部增强、语义一致、衣服分支因果干预来剥离衣服偏置。  
story 套路：headline 是“去除衣服偏置，但不破坏行人语义完整性”。

4. `ScRL` 形状中心 VI-ReID  
创新类型：问题重定义 + 新机制。  
为什么能发：把 VI-ReID 从外观对齐转成“形状是天然跨模态稳定信息，但红外形状估计不准”。机制包括红外形状恢复、形状特征传播、形状引导外观增强。  
story 套路：不是“加 parsing”，而是“以形状为中心重建跨模态表征”。

5. `SRCA` 弱监督文本行人检索  
创新类型：新设定 + 新机制。  
为什么能发：没有身份标签时，一对一图文匹配会制造假负样本。它用软相似度关系和最优传输伪标签校准，缓解跨模态匹配歧义。  
story 套路：把“弱监督不是缺标签而已”，讲成“人检索中天然存在多对多局部相似”。

6. `FedSTIL` 联邦终身 ReID  
创新类型：新数据流设定 + 问题重定义。  
为什么能发：真实部署里模型在多个边缘端持续漂移，同时又不能集中数据。它把终身学习和联邦学习合成一个问题，并用时空相关性选择性迁移知识。  
story 套路：从 benchmark 训练转到真实部署数据流，卖点是“隐私约束下的持续适应”。

7. `STHF` 视频可见光-红外高频学习  
创新类型：新机制。  
为什么能发：现有中间模态要么生成成本高，要么丢高频，要么没有时间信息。它用 3D FFT 构造序列级高频中间模态，再分浅层细节补偿和深层语义细化。  
story 套路：跨模态差异不是只在语义层，频域里的时空高频才是稳定细节。

8. `FAViT` 视频 ReID 特征级增广 ViT  
创新类型：机制型工程组合。  
为什么能发：利用视频 ReID 两个事实，换背景身份不变，混入异常帧需要识别。用背景 token、空间特征增广、时间异常帧检测，让 ViT 学会前景和时序一致性。  
story 套路：不用加复杂网络，而是把视频 ReID 的不变性写进训练任务。

9. `SPCL` 文本行人检索语义多态和共性  
创新类型：问题重定义 + 新机制。  
为什么能发：它把图文检索的一对一对齐问题改写成语义多态和属性共性问题，同一 ID 内有弱对应，不同 ID 间有属性共性。证据链靠 ID 内多态对齐、ID-free 属性对齐、属性先验重建。  
story 套路：人检索不是一张图配一句话，而是“同一语义有多种表达，不同身份共享属性”。

**对我们最有用的规律**
这批能发的点，大多不是“模块更复杂”，而是抓住一个训练或数据流假设错误：标签不完全可靠、对应关系不是一对一、形状估计不可靠、困难样本不只在 batch 内、部署数据会持续漂移。我们要投 B 类，最好也这样写问题，而不是继续做普通 gating 或对齐模块。

**强创新点候选**
1. Headline：面向航拍到地面的 ReID，不再对齐 2D 外观，而是学习可见表面到 3D 人体规范面的对应关系。  
挂团队资产：aerial-ground + SMPL 3D 几何 + SOLIDER-Swin。  
区别：最像 `ScRL`，但它是 VI-ReID 的 2D parsing 形状恢复；我们做的是极端俯视和地面视角下的 3D mesh/UV 规范化，把“视角导致同一身体区域不可比”作为核心问题。  
cheap kill-switch：不用训练全模型，先在 CARGO 或 AG-ReID.v2 上跑现有 SMPL/pose，统计同 ID 跨视角的 3D 规范面局部特征距离是否显著小于异 ID；再做一个只加训练期 3D surface consistency 的 Tiny 小跑。如果零训练几何质量很差，或者简单蒸馏不改善跨视角 hard subset，就停。

2. Headline：航拍低清不是普通降分辨率，而是身份高频在视角和高度中系统性丢失，需要视角条件的频率补偿。  
挂团队资产：aerial-ground + pose 热图门控 + SOLIDER-Swin。  
区别：最像 `STHF`，但它是视频 VI-ReID 的 3D FFT 序列中间模态；我们做 aerial-ground 的高度和俯仰视角造成的频率不对称，用 pose 区域约束高频补偿，避免变成普通频域增强。  
cheap kill-switch：先做频谱诊断，比较同 ID 航拍和地面图在人体区域的高频能量差，检查它是否和检索错误强相关。再做一个训练期高频 dropout/restore 小实验，只看 aerial-ground 跨视角子集是否超过噪声线。

3. Headline：跨视角 ReID 的困难样本不是最相似负样本，而是 3D 姿态和可见表面造成的“几何假负样本”。  
挂团队资产：SMPL 3D 几何 + pose 热图门控 + aerial-ground。  
区别：最像 `SAHSR` 的 hard sample retraining，但它按分类置信度找全局难样本；我们按 3D 姿态、朝向、可见表面重叠度定义几何难度，专门解决航拍和地面视角下“看起来不一样的正样本”和“几何相似的负样本”。  
cheap kill-switch：离线算每对样本的 pose/SMPL 几何相似度，看 baseline 错误是否集中在几何相似负样本和几何不相似正样本。若错误不随几何难度分层，别做。

4. Headline：真实航拍地面系统里，相机和时间持续漂移，模型需要按空间时间相关性选择旧知识，而不是全量回放或统一蒸馏。  
挂团队资产：aerial-ground + 现有终身 ReID 基建 + SOLIDER-Swin。  
区别：最像 `FedSTIL`，但它是普通边缘 ReID 的联邦终身学习；我们可以收窄到 aerial-ground 场景，把无人机高度、路线、地面摄像头位置作为可解释的时空结构，主打跨视角部署漂移。  
cheap kill-switch：把 AG-ReID.v2 或 CARGO 按高度、摄像头、时间切成任务序列，测顺序微调遗忘是否和空间时间距离有关。若没有规律，只是普通 continual ReID，就不要硬写。
