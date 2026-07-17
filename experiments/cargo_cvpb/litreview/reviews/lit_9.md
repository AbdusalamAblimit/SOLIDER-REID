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
session id: 019ef71f-8773-7553-b489-08cee1b314d8
--------
user
你是 ReID 论文创新挖掘员 9/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(9/20):
FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
Focusing on pedestrians like human for clothes changing person re-identification.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我会只读每篇前三页，按你给的三项提炼，不展开全文综述。先确认文件名和 `pdftotext` 可用性，再并行抽取文本。
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
/bin/zsh -lc "pwd && rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,120p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,160p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc "sed -n '1,120p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc 'which pdftotext && pdftotext -v' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/opt/homebrew/bin/pdftotext
pdftotext version 26.04.0
Copyright 2005-2026 The Poppler Developers - http://poppler.freedesktop.org
Copyright 1996-2011, 2022 Glyph & Cog, LLC

codex
目标文件都在当前目录，`pdftotext` 可用。我已经按项目规则确认当前目标仍是“读文献做差距分析，不开实验”，下面开始逐篇抽前三页。
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
/bin/zsh -lc "pdftotext -l 3 'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) | 978-1-6654-6946-3/22/$31.00 ©2022 IEEE | DOI: 10.1109/CVPR52688.2022.00720

FMCNet: Feature-Level Modality Compensation for Visible-Infrared Person
Re-Identification
Qiang Zhang1 , Changzhou Lai1 , Jianan Liu1 , Nianchang Huang1∗ , Jungong Han2
1
School of Mechano-Electronic Engineering, Xidian University, China
2
Computer Science Department, Aberystwyth University, U.K.
qzhang@xidian.edu.cn, {chzhlai,jianan liu,nchuang*}@stu.xidian.edu.cn, jungonghan77@gmail.com

Abstract
Visible
Modality-Specific
lity
Features

For Visible-Infrared person Re-IDentification (VI-ReID),
existing modality-specific information compensation based
models try to generate the images of missing modality from existing ones for reducing cross-modality discrepancy. However, because of the large modality discrepancy between visible and infrared images, the generated images usually have low qualities and introduce
much more interfering information (e.g., color inconsistency). This greatly degrades the subsequent VI-ReID
performance. Alternatively, we present a novel Featurelevel Modality Compensation Network (FMCNet) for VIReID in this paper, which aims to compensate the missing modality-specific information in the feature level rather
than in the image level, i.e., directly generating those missing modality-specific features of one modality from existing modality-shared features of the other modality. This
will enable our model to mainly generate some discriminative person related modality-specific features and discard those non-discriminative ones for benefiting VI-ReID.
For that, a single-modality feature decomposition module
is first designed to decompose single-modality features into
modality-specific ones and modality-shared ones. Then, a
feature-level modality compensation module is present to
generate those missing modality-specific features from existing modality-shared ones. Finally, a shared-specific feature fusion module is proposed to combine the existing and
generated features for VI-ReID. The effectiveness of our
proposed model is verified on two benchmark datasets.

1. Introduction
Person Re-IDentification (ReID) aims at matching the
given pedestrians from an image gallery taken by different cameras. Most existing ReID models focus on the
visible-visible image matching (i.e., VV-ReID). However,
* Equally corresponding authors.

978-1-6654-6946-3/22/$31.00 ©2022 IEEE
DOI 10.1109/CVPR52688.2022.00720

Modality-Shared
Features

Generating
VI-ReID
Network

VI-ReID
Network
Generating

Infrared
Modality-Specific
ality
Features

Missing Image Generation
(b)

(a)

Visible ModalitySpecific Features
Infrared
Modality-Specific
Features
Modality-Shared
Features

Matching

Generating

Infrared
Modality-Specific
Features
Visible ModalitySpecific Features

Generating

Our Model

Modality-Shared
Features

Our Model
(c)

Figure 1. Illustration of the differences between our model and
existing VI-ReID models. (a) Existing modality-shared feature
learning based models. (b) Existing image-level compensation
based models. (c) Our proposed feature-level compensation based
model.

these models may have poor performance when visible
cameras cannot well capture information, such as at night.
Compared with visible cameras, infrared cameras can still
capture clear images under those poor illumination conditions. Moreover, most cameras in modern surveillance systems support autoswitch between the visible and infrared
modes under different illumination conditions. Accordingly, Visible-Infrared ReID (i.e., VI-ReID) has raised more
and more attention recently.
The main challenge of VI-ReID lies in the modality discrepancy between the visible and infrared images. Meanwhile, it also surfers from large person variations, such as
viewpoints and postures. As shown in Fig. 1(a), most existing models [1–7] try to extract the discriminative modalityshared features for VI-ReID. Although great improvements
have been achieved, these models inevitably discard lots
of discriminative person-related modality-specific information, which may also benefit VI-ReID. Considering that,

7339

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.

some works [8, 9] propose the idea of modality-specific information compensation, which attempts to first generate
those missing modality-specific information from existing
modality and then jointly uses the generated and original
information for VI-ReID.
However, existing modality-specific information compensation based models usually achieve inferior results
compared with those modality-shared feature learning
based models. This may impute to the image-level compensation of existing models. That is, as shown in Fig.
1(b), existing models first generate the images of missing
modality from the images of existing modality and then extract discriminative person features from the paired images
for VI-ReID. However, it is very difficult to generate highquality images of one modality from another modality, due
to the large modality discrepancy between the visible and
infrared images. Especially, when generating visible images from infrared images, much more noisy information
(e.g., color inconsistency), instead of discriminative person features, will be introduced for VI-ReID. Besides, these
existing modality-specific information compensation based
models usually follow a two-stage structure and are not endto-end trainable, where the image generation sub-networks
and VI-ReID subnetworks are independent trained.
Actually, compared with the modality discrepancy between visible and infrared images, their features’ discrepancy has been reduced to some extents, since some common semantics information usually coexists in the unimodal
visible and infrared features. Therefore, the translation between visible data and infrared data in the feature level
may be easier than that in the image level. Meanwhile,
as discussed in some existing works [10–12], the singlemodality features (e.g., unimodal visible features or infrared features) can be decomposed into their own modalityspecific features and modality-shared features. The difficulties for cross-modality translation can be further reduced by generating those missing modality-specific features from existing modality-shared features rather than
from the whole single-modality features. More importantly, compared with image-level translation, the featurelevel translation allows us to flexibly control the generation
of those missing modality-specific features as our requirements by designing some dedicated loss functions. For example, we can only generate some discriminative personrelated modality-specific features and discard those nondiscriminative ones for benefiting VI-ReID.
Considering that, we will present a novel end-to-end
feature-level modality-specific information compensation
based model, i.e., the Feature-level Modality Compensation
Network (FMCNet), for VI-ReID in this paper. As shown
in Fig. 1(c), our proposed FMCNet aims to compensate
those missing modality-specific information in the feature
level rather than in the image level, i.e., directly generat-

ing those missing modality-specific features of one modality from existing modality-shared features of other modality. To this end, a Single-modality Feature Decomposition (SFD) module is first utilized to decompose the input
single-modality features into their own modality-specific
and modality-shared features, respectively. Meanwhile, a
modality decomposition loss is designed to facilitate the
decomposition of those single-modality features. Then, a
Feature-level Modality Compensation (FMC) module is designed to generate the missing modality-specific features of
one modality from the existing modality-shared ones of the
other modality for each sample image. Finally, a Sharedspecific Feature Fusion (SFF) module is designed to jointly
use the existing modality-shared and modality-specific features as well as the generated modality-specific features for
VI-ReID.
Similarly, cm-SSFT [13] also tries to simultaneously exploit those modality-shared and modality-specific features
for VI-ReID. It achieves shared-specific feature transfer by
modeling the affinities among different samples. Specially,
those missing modality-specific features in the cm-SSFT
are transfered from all the samples of the other modality in
the gallery. This may also introduce more modality-specific
information of other identities, thus easily leading to suboptimal results. Different from cm-SSFT, our proposed
model does not rely on other samples and is able to directly
and flexibly generate those missing modality-specific features from its own modality-shared features.
In summary, the main contributions of this work are as
follows:
(1) A novel FMCNet is presented, which proposes
feature-level rather than image-level modality-specific information compensation for VI-ReID. This enables our
model to focus on generating some required missing
modality-specific features (e.g., discriminative personrelated ones) for VI-ReID.
(2) Our proposed FMCNet provides an unified endto-end framework, achieving unimodal feature decomposion, modality-specific feature compensation and modality
shared-specific feature fusion for VI-ReID via the proposed
SFD, FMC and SFF modules, respectively.
(3) Our model significantly outperforms those imagelevel compensation based models and obtains competitive
and even better results than some state-of-the-art modalityshared feature learning based ones.

2. Related work
VV-ReID has been well studied for many years and
has achieved significant progress. Summarizing the vast
amount of existing works on VV-ReID is beyond the scope
of this paper and we refer those interested readers to
[14–16] for recent surveys.
Recently, VI-ReID has raised more and more attention

7340
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.

SFD Module
EV

FV

FMC Module

SFF Module

EVsp

Fsp ,V

Fsp ,V

DV  I

Fsp ,I

GV  I

Fsh,V

Fsp' ,I

V I
GAN

L

Real/Fake?

Fsp' ,I
V I
FC

L

V I
IC

L

C

Fsh,V

E sh

FI

I
E sp

D I V

Fsp ,V

LMD  LID

EI

Fsp ,I

LMC  LFID

Fsh,I

Fsh,I
G I V

Fsp' ,V

Real/Fake?

V
V
LIGAN
 LIFC
 LIICV

Fsp' ,V

C

Fsp ,I

Figure 2. Framework of the proposed Feature-level Modality Compensation Network (FMCNet).

due to its potential in real-life applications [5–7]. Most
existing VI-ReID models can be divided into two categories, i.e., modality-shared feature learning based ones and
modality-specific information compensation based ones.
Modality-shared feature learning based models aim to embed the features from different modalities into the same
feature space and reduce the cross-modality discrepancy
by using some feature-level constraints [1, 4, 7, 12, 17].
For example, [4] proposed a dual-path network to extract
modality-shared features from the input images by using
a shared network and designed a new bi-directional dualconstrained top-ranking loss to learn person discriminative
features for VI-ReID. Differently, modality-specific information compensation based models try to make up the missing modality-specific information from existing modalities
[8, 9, 18–20]. For example, [8] first designed two crossmodality image translation sub-networks to transfer an infrared image into its visible counterpart and transfered a
visible image to its infrared version, respectively. Then, a
ReID network was presented to reduce the appearance discrepancy by introducing some feature-level constraints.
In this paper, our proposed model follows the idea of
modality-specific information compensation. However, different from existing models which compensate missing
modality-specific information in the image level, our proposed model adopts the feature-level compensation.

3. Method
As shown in Fig. 2, the proposed model, i.e., Featurelevel Modality Compensation Network (FMCNet), mainly
consists of three parts, i.e., a Single-modality Feature Decomposition (SFD) module, a Feature-level Modality Com-

pensation (FMC) module and a Shared-specific Feature Fusion (SFF) module. Concretely, the proposed SFD module first extracts single-modality features from the input images and then decomposes them into their own modalityspecific and modality-shared ones. Then, the proposed
FMC module generates the missing (or compensated) visible (infrared) modality-specific features from those existing
decomposed infrared (visible) modality-shared features in
an adversarial way. Finally, the original modality-specific
features and modality-shared features as well as their compensated modality-specific features will be combined in the
proposed SFF module for VI-ReID. Details about these
modules will be discussed in the following contents.
Suppose that the training set (XV , XI ) contains P identities and each identity contains K samples. XV =
{xk,p
V , k = 1, .., K; p = 1, ..., P } denotes visible sample
images, and XI = {xk,p
I , k = 1, .., K; p = 1, ..., P } denotes infrared sample images.

3.1. SFD Module
As shown in Fig. 2, given the input visible images XV or
the infrared images XI , the proposed SFD module first extracts their single-modality features and then decomposes
those extracted single-modality visible (infrared) features
into their own modality-specific features and modalityshared features. Here, the ways of extracting and decomposing those single-modality visible and infrared features
are the same. Therefore, we take the input visible images
XV as the example to detail the corresponding process.
Specifically, the single-modality features FV are first extracted from XV by using a visible feature extraction subnetwork EV (∗). Then, a visible modality-specific feature

7341
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 192 (2025) 107946

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

GAE-Net: A gait-assisted enhancement network for video-based person
re-identiﬁcation
Minting Dai, Xi Yang

∗, Wenjiao Dong, Nannan Wang

State Key Laboratory of Integrated Services Networks, School of Telecommunications Engineering, Xidian University, Xi’an, 710071, China

a r t i c l e

i n f o

Keywords:
Video-based person re-identiﬁcation
Gait recognition
Knowledge distillation
Dynamic feature aggregation

a b s t r a c t
While video-based person re-identiﬁcation has garnered signiﬁcant attention and achieved substantial progress
in recent years, existing methods predominantly depend on appearance information, making them vulnerable to
changes in illumination and color. In contrast, the gait information concealed within walking postures exhibits
robustness against appearance changes, while gait sequences provide additional temporal cues to enhance the
temporal information in videos. Although some existing studies have utilized gait features to improve the robustness of Re-ID systems, the gap between gait and RGB data has not been adequately addressed. To bridge
this gap, we propose a Gait-Assisted Enhancement Network (GAE-Net) designed to concurrently learn both appearance features and complementary gait features from RGB video sequences. Speciﬁcally, GAE-Net consists of
two parts: Dynamic Two-stream Aggregation Network (DTA-Net) and Knowledge Distillation Fusion (KD-Fusion)
framework. Firstly, DTA-Net applies two branches to extract appearance features and gait features, respectively.
Moreover, the Dynamic Feature Aggregation (DFA) module is proposed to fuse gait and appearance features. Additionally, we propose Local Perception Complementary Distillation (LPCD) for Logit knowledge distillation. By
leveraging LPCD, robust dark knowledge from the multimodal model (DTA-Net) can be eﬀectively transferred to
enhance the robustness of the single-modal model (Re-ID network). Through collaborative eﬀorts between DTANet and LPCD, DGA-Net can acquire more comprehensive spatiotemporal representations. Extensive experiments
on MARS and LS-VID demonstrate that our proposed method signiﬁcantly outperforms other state-of-the-art
methods.

1. Introduction
Over the past decade, researchers have proposed numerous methods to address the highly challenging task of video-based person reidentiﬁcation (ReID). With the swift advancement in appearance feature
extraction techniques using Convolutional Neural Networks (CNNs),
video Re-ID aims to incorporate temporal information to achieve a more
robust spatio-temporal representation.
To aggregate temporal cues, existing works have made great eﬀorts
in temporal modeling. Taking inspiration from the ﬁeld of video recognition, some studies were dedicated to building temporal models by
utilizing two-stream network (Liu et al., 2017), recurrent neural network (RNN) (Chung et al., 2017), or 3D CNN (Liu et al., 2019b), etc.
These methods combine appearance static information (color, wearing,
etc.) with dynamic temporal representation (motion information, optical ﬂow cues, etc.) as the entire video representation. Despite adding
dynamic temporal representation as the supplementary information, the
Re-ID method still relies heavily on appearance features as the primary
discriminant representation. Therefore, it is easily disturbed by external

environmental factors, such as light changes, color changes, wearing
changes, etc.
Gait is a biological feature that embodies a persons’ unique walking posture and can be captured from a distance. As a biometric technology, gait recognition has made remarkable achievements in recent
years by extracting gait features from the input gait data to identify
target pedestrians. Gait recognition utilizes robust gait features, including skeletal joints and walking postures, as biometric information to
identify individuals. As a result, gait recognition is less aﬀected by
appearance variations. Earlier methods (Wu et al., 2017) represented
gait through static images, where all gait silhouettes were consolidated
into a single image or gait template for recognition purposes. However, this compression process often led to the loss of temporal dynamics and detailed spatial features. In recent years, many works (Fan
et al., 2020; Lin et al., 2021) have overcome the above shortcomings
by taking gait video sequences as input, thereby retaining more spatiotemporal features directly from the original gait sequence. However,
the lack of appearance information in gait data also limits its further
development.

∗ Corresponding author.

E-mail addresses: dmt@stu.xidian.edu.cn (M. Dai), yangx@xidian.edu.cn (X. Yang), dwj@stu.xidian.edu.cn (W. Dong), nnwang@xidian.edu.cn (N. Wang).
https://doi.org/10.1016/j.neunet.2025.107946
Received 30 May 2025; Received in revised form 14 July 2025; Accepted 1 August 2025
Available online 5 August 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 192 (2025) 107946

M. Dai et al.

method that enhances the perception and processing of local features
for transferring robust deep latent knowledge from multimodal fusion
models to unimodal appearance models.
We trained a binary segmentation model to generate human gait contour images based on the UNet (Ronneberger et al., 2015) architecture,
and take the obtained gait sequences as gait model input. In addition,
we propose LPCD to enhance the perception and processing of local features by employing multi-scale aggregation techniques, thereby extracting and preserving more reﬁned semantic knowledge. This approach effectively guides the student model(Re-ID network) in learning discriminative detailed features from the teacher model (DTA-Net) for similar
pedestrians, ultimately leading to a signiﬁcant enhancement in its performance. Through the combination of DTA-Net and LPCD, the GAE-Net
can achieve more robust video representation. Furthermore, the parameter reduction achieved via knowledge distillation not only improves the
eﬃciency of model inference but also signiﬁcantly enhances the overall
computational speed. The Single-modal model does not need to rely on
other types of data during the reasoning process and can complete tasks
by using only single-modal data.
In summary, our contributions are four folds:

Fig. 1. Comparison between RGB clips for video Re-ID and gait clips for gait
recognition. Although RGB sequences and gait sequences exhibit modality differences in terms of color, environmental background, and other factors, they
complement each other in terms of spatial information and temporal relationships.

•

We ﬁnd the shortcomings of video Re-ID and explore the feasibility
of the gait supplement feature: the Re-ID methods are easily disturbed due to their dependency on appearance features, while the
gait recognition method that is not aﬀected by appearance can provide supplementary spatiotemporal information.
• We propose a Dynamic Gait Assistance Network (DGA-Net), utilizing
Dynamic Feature Aggregation (DFA) to dynamically aggregate the
collected gait features and appearance features, thereby obtaining
more robust representations.
• We propose Local Perceptual Complementary Distillation (LPCD) to
overcome the limitations of classic logit distillation. The global logit
is decoupled into consistent and complementary local logit outputs,
aiming to mine and convey more abundant and explicit semantic
knowledge.
• The proposed GAE-Net demonstrates superior performance on two
widely used video-based person Re-ID benchmarks in contrast to the
many previous state-of-the-art.

Based on the above analysis, gait recognition can supplement the defects of person Re-ID while ensuring task consistency. Recent research
has explored the use of gait features to improve the reliability and
robustness of Re-ID systems (Liu et al., 2015) extracted feature representations by integrating gait and appearance features at both the
score and feature levels, fusing these features comprehensively. In contrast to the background removal methods, Tang et al. (2019) employed
a background suppression approach, assigning diﬀerential weights to
background and human body elements during image feature extraction (Zhao et al., 2023) constructed an appearance-gait dual-stream
network (AGNet), which simultaneously extracts appearance and gait
features from both RGB video clips and gait video sequences.However,
existing methods have introduced gait features for solution proposals,
but they have not completely bridged the gap between gait and visible data. Therefore, exploring better solutions that combine video-based
Re-ID and gait recognition holds signiﬁcant research value. As shown in
Fig. 1, there are modality diﬀerences between RGB sequences and gait
sequences, while they also exhibit modality complementarity. Specifically, gait features enhance the robustness of the Re-ID system with
its non-changeable gait features. At the same time, appearance features
can complement the appearance absence of gait features. In addition,
temporal information, as a common part, is better explored under the
joint fusion of the two representations. Therefore, we sought to explore
a novel framework that complementarily combines video Re-ID and gait
recognition.
Inspired by dynamic gait features, we propose a Gait-Assisted Enhancement Network (GAE-Net) for video Re-ID. Firstly, we design a Dynamic Two-stream Aggregation Network (DTA-Net) to simultaneously
capture both appearance and gait characteristics from RGB video frames
and gait data. In the feature extraction stage, DTA-Net employs the ReID network and gait network as separate branches to respectively extract appearance features and gait features. In the feature fusion stage,
Dynamic Attention Weighting (DAW) and Dynamic Weight Aggregation
(DWA) are designed to fuse gait and appearance features. With the combined eﬀect of the above modules, the DTA-Net can learn better appearance and gait complement features. Furthermore, we contend that existing knowledge distillation methods based on logit are not optimal because they only use global logit output, which includes various semantic
knowledge. This could impart ambiguous knowledge to the student, potentially leading its learning in the wrong direction. To overcome this issue, we propose a Local Perception Complementary Distillation (LPCD)

2. Related work
2.1. Person Re-ID
In recent years, image-based Re-ID has experienced remarkable advancements, propelled by the rapid progress in deep neural networks
(DNNs). Generally, current approaches can be classiﬁed into two primary categories: discriminative learning and metric learning. Discriminative learning focuses on exploring more robust and discriminative
features through better network architecture. The proposal of the idea
of constructing ResNet (He et al., 2016) with residual blocks made the
standard backbone network for the Re-ID task widely recognized. Further, Sun et al. (2018) and Wang et al. (2018) divided the feature map
and utilized ﬁne-grained spatial features to obtain more discriminant
information. However, the pursuit of highly discriminative features in
person re-identiﬁcation (Re-ID) methods leads to a signiﬁcant increase
in spatio-temporal complexity. To address this issue, Wang et al. (2024)
proposed a feature compression method that preserves Euclidean distance, thereby substantially reducing both computational and storage
costs. Liu et al. (2023c) proposed a knowledge- preserving continuous
person re-identiﬁcation model based on the GAT, which mitigates the
issue of catastrophic forgetting in continuous learning and enhances the
model’s generalization capability by leveraging a knowledge graph. In
addition, the Transformer structures (Zhang et al., 2021) were incorporated into the Re-ID task, resulting in enhanced performance. For example, Liu et al. (2023a) deeply integrates convolutional and Transformer
representations, leveraging spatial-temporal complementary learning to
2

Neural Networks 192 (2025) 107946

M. Dai et al.

better capture local and global dependencies. Similarly, some works attempt to unify Transformer and CNN architectures, such as Wang et al.
(2025b), which fuses their complementary characteristics to improve robustness under diverse conditions. Recently, large-scale language-image
pre-trained models, such as CLIP, have exhibited remarkable performance in various cross-modal retrieval tasks. Certain approaches have
started to leverage these models for person re-identiﬁcation tasks (Li
et al., 2023; Yu et al., 2025b), thereby enhancing the robustness of person re-identiﬁcation by utilizing high-level semantic features with improved generalization capabilities.
Video-based Re-ID is considered a generalization of image-based
Re-ID because video frames contain more spatio-temporal information
than static images. In video re-identiﬁcation tasks, temporal feature
learning is the core module of most algorithms. There are some works
aimed at integrating eﬀective spatiotemporal information into video
representations. Gao and Nevatia (2018) utilized a frame-level attention mechanism to learn spatiotemporal representation between frames,
thus reducing noise pollution, including issues like partial occlusion.
Subsequently, numerous methods began to divide the feature map spatially in order to explore local ﬁne-grained information. To explore local salient regional features, Li et al. (2018a), He et al. (2023) and
Zhang et al. (2022) applied a variety of spatial attention models to
learn attention weights between diﬀerent body parts. Additionally, Wu
(2024) leveraged relative states to extract robust features of the target
pedestrian and aggregate these features according to their completeness, thereby eﬀectively capturing the spatio-temporal characteristics
of pedestrians in video sequences. Moreover, Transformer-based models have been widely applied to model long-range dependencies. For
example, the Trigeminal Transformer framework (Liu et al., 2024a) extracts multi-view features from spatial, temporal, and visual branches,
enabling complementary learning across diﬀerent temporal granularities. To further improve video modeling, Mamba-style architectures
with state-space modeling ability are emerging as an alternative to traditional self-attention mechanisms. The CLIMB-ReID (Yu et al., 2025a)
framework integrates Mamba with CLIP features to simultaneously capture semantic alignment and temporal smoothness, demonstrating that
hybrid modeling yields superior results. However, although these existing methods have eﬀectively improved recognition performance by
modeling spatio-temporal features, they may still overlook individualspeciﬁc dynamic behavior patterns, and computational eﬃciency remains a critical concern.
Gait-assisted Re-ID is a re-identiﬁcation technology that integrates
gait features with appearance features, designed to overcome the limitations of traditional person re-identiﬁcation methods in scenarios involving clothing changes, occlusions, or illumination variations. Liu et al.
(2015) enhanced the pedestrian Re-ID network by integrating gait biometric features. Furthermore, Li et al. (2018b) introduced a progressive Re-ID approach that performs a hierarchical search within the feature space, which initially applies a rough ﬁlter using appearance characteristics, followed by a more detailed search utilizing gait information. Tang et al. (2019) used gait masks for progressive background
suppression in person Re-ID. Recently, Wu et al. (2022) designed a
two-stream network consisting of a provider stream (P-Stream) and a
receiver stream (R-Stream). The latter performs prior optimization on
foreground information, while the former guides the R-Stream to focus on foreground information and useful identity (ID)-related cues in
the background. Considering the temporal information inherent in gait,
Zhao et al. (2023) constructed an appearance-gait dual-stream network
(AGNet) to simultaneously extract appearance and gait characteristics
from both RGB video sequences and gait video sequences.

energy images (GEIs). However, during the synthesis process of GEIs, the
temporal information of the gait sequence may be lost. Therefore, some
researchers ﬁrst extract gait features on frame-level gait images rather
than GEIs, and then process them to generate gait templates. For instance, Chao et al. (2019) used 2D CNN to extract global features at the
frame level and then constructed gait templates using statistical functions. Zhang et al. (2019) partitioned human gait into various local segments and utilized several separate 2D CNNs to extract features from
each segment individually. However, these methods either lose temporal continuity or fail to fully capture the dynamic motion cues that
are critical for identity perception. Some researchers (Ma et al., 2023;
Wang et al., 2023, 2022) addressed this limitation by automatically establishing spatio-temporal feature representations of the dynamic parts
of the human body through a focus on walking’s dynamic characteristics and enhancements (dynamic mechanisms). Additionally, Dou et al.
(2023) proposed the GaitGCI framework, which employs counterfactual
intervention and causal reasoning to mitigate confounding factors’ inﬂuence by maximizing the likelihood diﬀerence between factual/counterfactual attention. Some researchers (Jin et al., 2022; Pan et al., 2023b)
introduced a two-stream network framework, which integrates gait features for person re-identiﬁcation. However, although these methods can
generate highly eﬀective representations, they imposed signiﬁcant computational overhead, making them less appropriate for deployment in
lightweight or end-to-end person re-identiﬁcation systems.
2.3. Knowledge distillation
Knowledge distillation aims to compress models while keeping the
network structure unchanged. Hinton et al. (2015) initially introduced a
technique for compressing large and complex models into more compact
and eﬃcient architectures, which is also the origin of knowledge distillation (KD). In order to make better use of the information from logits,
KD introduced hyperparameter temperature T. Subsequently, plenty of
works focused on better use of soft labels to deliver more knowledge.
The subsequent content explores two approaches: logit distillation (Chen
et al., 2021; Huang et al., 2022; Yang et al., 2023; Zhao et al., 2022) and
intermediate feature distillation (Shen & Xing, 2022; Yang et al., 2022;
Yue et al., 2020). Our work employs knowledge distillation to dynamically fuse gait and appearance features, which is more relevant to logit
distillation.
Logits Distillation: The core idea of Logits distillation is to utilize
the Logits output by the teacher model to guide the training process of
the student model. Recently, a lot of studies have begun to analyze the
soft labels of teacher models to tap the potential of logits distillation.
Furthermore, SRRL (Yang et al., 2023) aligned the output logits from
the teacher model and the features extracted by the student model after
they passed through the teacher’s linear layer. Compared to traditional
distillation, DIST (Huang et al., 2022) achieved more precise knowledge
transfer by capturing both the semantic similarity and contrast between
samples. Meanwhile, DKD (Zhao et al., 2022) decoupled logit into target and non-target classes and assigned them diﬀerent weights. In addition, Bai et al. (2023) proposed Guided Adversarial Contrast Distillation
(GACD) to transfer antagonistic robustness from teachers to students
through potential representation.
3. Methods
The Re-ID system pays too much attention to appearance features
and is disturbed by external factors, such as light changes, color changes,
etc. At the same time, gait recognition only learns the human contours
and temporal gait features without any appearance information, thus
limiting its further development. Therefore, Thus, we propose integrating the strengths of both approaches to combine visual appearance and
gait characteristics, aiming to develop a more resilient spatio-temporal
representation that can eﬀectively address the inevitable challenges. In
this paper, we propose a Gait-Assisted Enhancement Network (GAE-Net)

2.2. Gait recognition
Recently, many existing gait recognition algorithms have explored
convolutional neural networks (CNNs) to generate gait representation.
Shiraga et al. (2016) attempted to extract global gait features from gait
3


exec
/bin/zsh -lc "pdftotext -l 3 'Focusing on pedestrians like human for clothes changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 193 (2026) 107960

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Focusing on pedestrians like human for clothes changing person
re-identiﬁcation
Wenjie Pan a , Jianqing Zhu

a,∗, Xiaolin Cui b , Huanqiang Zeng a,c , Yibing Zhan

d

a College of Engineering, Huaqiao University, Quanzhou, 362021, Fujian, China
b

Technology and Communications Oﬃce, Xiamen Municipal Public Security Bureau, Xiamen, 361001, Fujian, China

c School of Optoelectronic and Communication Engineering, Xiamen University of Technology, Xiamen, 361024, Fujian, China
d

Yunnan United Vision Technology Co. Ltd, Kunming, 650500, Yunnan, China

a r t i c l e

i n f o

Keywords:
Re-identiﬁcation
Clothes changing
Human focus

a b s t r a c t
Current approaches focus mainly on the design of networks to learn key identity features from local body components for clothes-changing person re-identiﬁcation (CC-ReID). In this paper, we propose a humanoid focusinspired image augmentation (HFIA) method, which is intuitive image processing rather than a sophisticated
network architecture designed to enhance local nuances of pedestrian images. Based on pedestrian silhouettes,
we roughly divide a pedestrian image into ﬁve body components, that is, head-shoulder, upper left torso, upper
right torso, lower left torso, and lower right torso. The HFIA has two key designs to deal with these components:
the central emphasis strategy (CES) and the component continuity processing (CCP). For each component, leveraging the natural tendency of human visual attention towards central regions, the CES constructs an enlargement
grid, where the closer the center, the greater the enlargement. To maintain the continuity of assembly, the CCP
performs an overall alignment of component centers, that is, all components share the same normalized vertical
coordinate and the left and right torsos have mirrored horizontal coordinates. Furthermore, the CCP implements
a smoothing post-processing to uniformly erase the discontinuity between the head-shoulder, upper left torso,
and upper right torso. Experiments show the state-of-the-art performance of HFIA.

1. Introduction
Clothes changing person re-identiﬁcation (CC-ReID) (Li et al., 2023,
2024b; Lin et al., 2024; Liu et al., 2024; Yang et al., 2019) is an extremely challenging image retrieval task. It aims to ﬁnd pedestrians with
the same identity as a query, while those pedestrians have diﬀerent camera viewpoints and wear various clothes. The ensemble coding hypothesis in cognitive neuroscience (Michael et al., 2014) posits that humans
achieve identity recognition among similar objects by collectively activating multiple local characteristics. Inspired by the ensemble coding
hypothesis (Michael et al., 2014), it is worthwhile to explore a method
that enhances multiple local characteristics holistically. This might address the gap in data augmentation methods for existing CC-ReID.
Data augmentation methods for CC-ReID remain unexplored, with
current approaches primarily focusing on network architecture design.
We categorize these methods into three categories: local details learning methods (Huang et al., 2017; Sun et al., 2018; Zhao et al., 2017),
identity-related learning methods (Gao et al., 2022; Hong et al., 2021;
Jin et al., 2022; Li et al., 2021; Wang et al., 2022), and identity-irrelated

learning methods (Han et al., 2023; Huang et al., 2021; Xu et al., 2021;
Yang et al., 2023a). We posit that image-level learning (i.e., data augmentation) of local details can aid the network in discerning identityrelated from identity-irrelated information. This can further capture crucial identity information to diﬀerentiate pedestrians. As shown in Fig. 1,
it can be observed that focusing on local details makes it easier to understand the identity-irrelated and identity-related information within each
local region. Subsequently, by excluding identity-irrelated information
(such as hat and jacket) and then capturing identity-related information
(such as beard and pose) within the local region, identity recognition
can be achieved more eﬀectively. Therefore, it is worthwhile to investigate an image processing method to improve local details.
Identity-related learning methods typically involves recognizing
pose (Bansal et al., 2022; Gao et al., 2022; Hong et al., 2021; Jin
et al., 2022; Wang et al., 2022), 3D shape (Chen et al., 2021; Liu et al.,
2023), and facial information (Wan et al., 2020; Xue et al., 2018). Typically, a multi-branch network architecture is utilized, with one branch
dedicated to identiﬁcation, while the remaining branches focus on
understanding identity-related information. Identity-irrelated learning

∗ Corresponding author.

E-mail addresses: panwj@stu.hqu.edu.cn (W. Pan), jqzhu@hqu.edu.cn (J. Zhu), 294781673@qq.com (X. Cui), zeng0043@hqu.edu.cn (H. Zeng),
zybjy@mail.ustc.edu.cn (Y. Zhan).
https://doi.org/10.1016/j.neunet.2025.107960
Received 24 December 2024; Received in revised form 17 July 2025; Accepted 4 August 2025
Available online 11 August 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 193 (2026) 107960

W. Pan et al.

Fig. 1. A woman matches celebrities wearing diﬀerent clothes by focusing on local details in an image.

methods typically involves improving the understanding of clothes to
reduce its interference. The methods encompass the design of clothes
recognition modules (Huang et al., 2021, 2019; Xu et al., 2021; Yang
et al., 2023a), dressing simulation modules (Han et al., 2023; Yu et al.,
2020), and clothing adversarial loss (Gu et al., 2022). Both identityrelated and identity-irrelated learning methods excel in achieving pedestrian identiﬁcation with clothes changing. Furthermore, by enchancing
local details their capabilities could be improved.
Local details learning methods usually divide data into multiple body
parts and then use well-designed multi-branch networks to learn local
details of diﬀerent body parts. Considerations are made for the dividing
method, such as automatic (Huang et al., 2017; Zhao et al., 2017) or
manual (Huang et al., 2021, 2019; Sun et al., 2018; Wang et al., 2018), to
divide the data into several body parts of the same size or diﬀerent sizes.
In the design of multi-branch networks, weights for each branch are
manually assigned (Huang et al., 2021, 2019) or automatically learned
(Huang et al., 2017; Sun et al., 2018; Wang et al., 2018; Zhao et al.,
2017), and the identity-related or identity-irrelated learning strategies
discussed earlier are integrated. These methods are excellent, but focus
on network design without considering the use of data augmentation to
enhance local details.
Real-world variations, such as angle, color, or scale changes, serve as
basic data augmentation techniques across many tasks. With a diverse
range of basic augmentation methods available, some strategies (Cubuk
et al., 2019, 2020; Müller & Hutter, 2021) aim to automatically choose
the appropriate techniques from this pool. Although some methods (Pan
et al., 2023; Zhong et al., 2020, 2018) mitigate challenges such as occlusion (Zhong et al., 2020), lighting variations (Pan et al., 2023), and
view variations (Zhong et al., 2018) common in conventional ReID tasks,
CC-ReID presents additional complexities due to the clothes changing.
To overcome clothing variation in CC-ReID, Pos-Neg (Jia et al., 2022)
augments the training data by exchanging outﬁts between diﬀerent images from an identity-irrelated learning perspective. In addition to this
approach, we argue that local detail enhancement also holds great potential. Identity-irrelated clothes occupies most of the image, whereas

identity-related features like the face, gait, and posture are conﬁned to
small local regions. By applying data augmentation at the image level,
identity-related information can receive more attention, allowing subsequent networks to better learn identity cues embedded in local details.
In this paper, we propose a humanoid focus-inspired image augmentation (HFIA) method for CC-ReID. In contrast to methods that use
network architecture design to enhance local details, HFIA employs an
image-based strategy. The HFIA divides images into ﬁve body components based on pedestrian silhouettes: head-shoulder, upper left torso,
upper right torso, lower left torso, and lower right torso. HFIA comprises
two key designs: the central emphasis strategy (CES) and the component
continuity processing (CCP). The CES constructs an enlargement grid
to scale the image, with a greater proportion of data near the center
to simulate human visual attention focused on the central region. The
CCP aligns the CES used for diﬀerent body components to ensure that
all body components share a normalized vertical axis coordinate, while
the left and right body components use mirrored horizontal axis coordinates. Subsequently, the CCP applies a smoothing post-processing to
uniformly erase the discontinuities between the head-shoulder and upper left torso and upper right torso to produce a coordinated reassembled image. We evaluate HFIA on three public CC-ReID datasets (Gu
et al., 2022; Huang et al., 2019; Yang et al., 2019), and it turns out to
have state-of-the-art performance.
The contributions of the paper are summarized as follows.
•

Based on the characteristic that human vision mainly focuses on central regions, we propose a central emphasis strategy (CES) to enhance
local details in single regions by increasing the data proportion of
central areas.
• Based on the emsemble coding hypothesis in cognitive neuroscience,
we propose component continuity processing (CCP), which applies
CES to diﬀerent body regions according to human contours to
achieve multi-region local detail enhancement.
• By combining CES and CCP, we propose humanoid focus-inspired
image augmentation (HFIA), the ﬁrst local detail learning data
2

Neural Networks 193 (2026) 107960

W. Pan et al.

2.2. Local details learning

augmentation for CC-ReID tasks. Experiments show it achieves stateof-the-art performance on multiple CC-ReID benchmarks, and has
generalization in knowledge distillation and unsupervised learning.

The body part strategy (Hou et al., 2020; Suh et al., 2018; Wang
et al., 2018; Xu et al., 2018; Zhao et al., 2017) is a predominant approach in improving identity recognition by learning local details in
CC-ReID. It takes a single input data, splits it into several body parts,
and outputs these parts for diﬀerent models. Fixed-location splitting
strategies are commonly utilized. Wang et al. (2018) proposed a threebranch model which processes images with various levels of splitting-no
splitting, two parts, and three parts-for part-based learning. Some approaches (Huang et al., 2021, 2019) perform manual splitting at the image level by feeding these body parts into independent models, and adjusting their weights manually to control the inﬂuence of each parts. PCB
(Sun et al., 2018) employs body part strategy to process the output features of the backbone network for part alignment. Zhang et al. (2019a)
performed body part learning on a ﬁne-grained semantic level. Some
methods utilize learnable splitting strategies. DeepDiﬀ (Huang et al.,
2017) develops a multi-branch model that utilize two automated splitting strategies to produce overlapping body parts of diﬀerent heights.
Zhao et al. (2017) introduced SpindleNet, a method that employs a
tree-structured fusion of multi-stage features to generate parts of various sizes. These approaches employ diverse splitting strategies and intricate models to eﬀectively utilize the local details within body parts
with excellent results. However, there has been limited exploration of
data augmentation methods for enhancing local details.

The remainder of this paper is organized as follows. Section 2 surveys
recent work related to this paper. Section 3 describes our method in
detail. Section 4 presents experimental results and analysis to show the
superiority of our method. Section 5 concludes this paper.

2. Related works
2.1. Identity-related and identity-irrelated learning
It is a prevalent strategy to improve the perception of identity-related
information (Cui et al., 2023; Hu et al., 2024; Huang et al., 2024; Xie
et al., 2023; Zhao et al., 2022) in CC-ReID. Certain investigations bolster
the recognition of identity-related characteristics through pose-based
learning. GI-ReID (Jin et al., 2022) is a multi-branch model that integrates a ReID branch with a gait recognition branch. Similar to GI-ReID
(Jin et al., 2022), FSAM (Hong et al., 2021) is a multi-branch model
with its pose branch constructed through a pose clustering method.
Yang et al. (2019) devised a multi-branch network that incorporates
contour sketches for pose information and utilizes the polar coordinate
transformation to discern the angular details of various poses. 3DSL
(Chen et al., 2021) is a multi-branch model which aims to enhance
the multi-granularity pose perception through 3D reconstruction. 3D
pose features are incorporated in VITF (Bansal et al., 2022) into a transformer model, facilitating attention-driven pose learning. Several studies aim to enhance facial feature learning. Wan et al. (2020) constructed
a multi-branch network, where one branch handles faces through an
object detection model, and the other processes the full data. CCAN
(Xue et al., 2018) is a multi-branch network, which only learns the face
and body characteristics extracted from the object detection models.
Some researches focus on the information of the full body. Wang et al.
(2022) presented CAMC, a multi-branch model for acquiring knowledge
of pedestrian body shape using attention mechanisms. Yang et al. (2022)
presented SirNet, a multi-branch model for processing triplet samples
and modeling clusters of samples with the same identity.
Enhancing the learning of identity-irrelated information represents
another prevalent strategy (Davila et al., 2023; Pan et al., 2023; Xu et al.,
2021; Yang et al., 2023a; Zhu et al., 2023) in CC-ReID. Several methods
involve the construction of models to perceive identity-irrelated information. CASENet (Li et al., 2021) is introduced to leverage color information within and across images for improving clothes color perception.
In ReIDCaps (Huang et al., 2019), vector directions of the capsule network are utilized to encode clothes information. Huang et al. (2021)
developed a multi-branch network which employs unsupervised clustering to embed state awareness of clothes. AFDNet (Xu et al., 2021)
is proposed to process two samples concurrently from diﬀerent identities to enable clothes perception for both inter-class and intra-class
scenarios. Yang et al. (2023a) developed a multi-branch network to address feature bias due to clothes changing through causal intervention.
Certain researches involve clothes replacement simulation techniques.
CCFA (Han et al., 2023) employs adversarial learning in the feature
space to simulate clothes changes. BCNet (Yu et al., 2020) oﬀers multiple clothes templates at the image level for the same sample, enabling
clothes swapping simulation. Certain researches utilize loss functions to
improve the information perception of clothing. Gu et al. (2022) reduces
the sensitivity of the model to clothes through adversarial learning.
It is a prevailing method to enhance the learning of identity-related
or identity-irrelated information in CC-ReID, which has achieved remarkable results. Furthermore, information such as facial characteristics, gait, accessories, etc., is typically localized, suggesting that enhancing local details is advantageous for both identity-related and identityirrelated methods.

2.3. Data augmentation
Data augmentation improves the model’s ability to learn key information by transforming and perturbing images. For example, augmentation techniques such as random ﬂips, rotations, and perspectives
all facilitate the learning of position-invariant information. Similarly,
color jitter and grayscale transformations improve the learning of colorinvariant information. Some studies (Cubuk et al., 2019, 2020; Müller
& Hutter, 2021) propose automatic search strategies to identify taskspeciﬁc data augmentations from a diverse augmentation space. For
example, AutoAugment (Cubuk et al., 2019) employs reinforcement
learning to select multiple augmentations. RandAugment (Cubuk et al.,
2020) streamlines the search strategy through random selection. TrivialAugment (Müller & Hutter, 2021) makes a further simpliﬁcation by
randomly selecting individual augmentations. In object ReID, random
cropping deals with scale changes of objects in the camera view. Random erase (Zhong et al., 2020) simulates occlusion eﬀects. Furthermore,
random hybrid augmentation (Pan et al., 2023) enhances multi-spectral
perception under low-light conditions.
In CC-ReID, Pos-Neg (Jia et al., 2022) is an identity-irrelated data
augmentation method. It consists of two components: positive augmentation and negative augmentation. Before augmentation, a semantic segmentation model is required to generate masks, which are used to extract identity-irrelevant clothing information. In positive augmentation,
clothing regions from diﬀerent samples are exchanged to simulate clothing changes under the same identity. In negative augmentation, images
with variations in both clothing and pose are generated simultaneously.
Pos-Neg employs pose estimation heatmaps to ensure pose variation.
Although Pos-Neg heavily relies on semantic segmentation and pose estimation to produce high-quality augmented samples, resulting in considerable computational cost, this cost does not aﬀect inference, as it is
only incurred during training. Compared with Pos-Neg, our method focuses on local detail enhancement, operates on single images, and does
not require additional information or models.
Existing data augmentation methods are eﬀective, but none have explored the beneﬁts of enhancing local details for overall network learning. Since data augmentation is applied prior to network training, local
details emphasized at the image level can fully participate in the training
of all networks. Currently, the only data augmentation method designed
for CC-ReID is the identity-irrelated approach Pos-Neg (Jia et al., 2022).
3


codex
前三篇已经读完。它们共同的发稿套路很清楚：不是单纯堆模块，而是把“缺失信息补偿”“人类局部注意”“步态作为外部稳健模态”包装成任务缺口，再用较低推理成本或单模态蒸馏补可落地性。继续读下一组。
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
/bin/zsh -lc "pdftotext -l 3 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3990

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

GAReID: Grouped and Attentive High-Order
Representation Learning for
Person Re-Identification
Pingyu Wang , Fei Su , Member, IEEE, Zhicheng Zhao , Member, IEEE,
Yanyun Zhao , and Nikolaos V. Boulgouris , Senior Member, IEEE

Abstract— As person parts are frequently misaligned between
detected human boxes, an image representation that can handle
this part misalignment is required. In this work, we propose an
effective grouped attentive re-identification (GAReID) framework
to learn part-aligned and background robust representations
for person re-identification (ReID). Specifically, the GAReID
framework consists of grouped high-order pooling (GHOP) and
attentive high-order pooling (AHOP) layers, which generate highorder image and foreground features, respectively. In addition,
a novel grouped Kronecker product (GKP) is proposed to use
both channel group and shuffle strategies for high-order feature
compression, while promoting the representational capabilities
of compressed high-order features. We show that our method
derives from an interpretable motivation and elegantly reduces
part misalignments without using landmark detection or feature
partition. This article theoretically and experimentally demonstrates the superiority of the GAReID framework, achieving
state-of-the-art performance on various person ReID datasets.
Index Terms— Group shuffle, high-order pooling, Kronecker
product, part misalignments, person re-identification (ReID).

I. I NTRODUCTION

Fig. 1. Illustration of the part misalignment problem caused by camera views,
detection errors, body occlusions, and background clutters. Aligned part pairs
are connected with solid lines, while the misaligned part pairs are connected
with dashed lines. (a) Camera view. (b) Detection error. (c) Body occlusion.
(d) Background cluster.

P

ERSON re-identification (ReID) aims at matching person
images of the same person across nonoverlapping cameras. It plays an important role in various video surveillance
applications such as suspect tracking and missing elderly or
children retrieval. With the blooming of convolutional neural network (CNN), the current deep-feature-learning-based
methods [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
[12], [13], [14], [15], [16] have significantly outperformed a
variety of traditional feature-learning-based approaches [17],
[18], [19], [20], [21], [22]. However, the ReID task is far
Manuscript received 26 November 2020; revised 21 November 2021 and
27 May 2022; accepted 17 September 2022. Date of publication 5 October
2022; date of current version 1 March 2025. This work was supported by
the Chinese National Natural Science Foundation under Grant 62076033 and
Grant U1931202. (Corresponding author: Zhicheng Zhao.)
Pingyu Wang, Fei Su, Zhicheng Zhao, and Yanyun Zhao are with the Beijing
Key Laboratory of Network System and Network Culture, School of Artificial
Intelligence, Beijing University of Posts and Telecommunications, Beijing
100876, China (e-mail: applewangpingyu@bupt.edu.cn; sufei@bupt.edu.cn;
zhaozc@bupt.edu.cn; zyy@bupt.edu.cn).
Nikolaos V. Boulgouris is with the Department of Electronic and Computer
Engineering, Brunel University London, UB8 3PH Uxbridge, U.K. (e-mail:
nikolaos.boulgouris@brunel.ac.uk).
This article has supplementary downloadable material available at
https://doi.org/10.1109/TNNLS.2022.3209537, provided by the authors.
Digital Object Identifier 10.1109/TNNLS.2022.3209537

from being solved because of part misalignments caused by
camera views, detection errors, body occlusions, and background clutters. As shown in Fig. 1, part misalignments usually
change the spatial distribution of person appearances, which
might degenerate the distinctiveness and robustness of person
representations.
To mitigate part misalignments, prior ReID works have
broadly followed two main paradigms, i.e., part-based and
landmark-based methods. The part-based approaches [2], [5],
[12], [13], [14] partition the global person images/features into
a few fixed rigid parts and concentrate on local feature learning
so as to obviate the need for landmark detection. Nevertheless,
such coarse partition is unable to effectively align body parts
without considering fine-grained pose variations within each
part. For achieving fine-grained part alignments, the landmarkbased works [1], [6], [7], [8], [9], [10], [11], [23] use human
landmark annotations or landmark detection networks and
then learn part-aligned features from pose-normalized person
images. Although those works have boosted ReID performance, they introduce extra operations to the ReID system,

2162-237X © 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.

WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING

e.g., landmark detection and pose normalization. In addition,
those operations bring nonignorable space and time costs,
making it hard to train the ReID model.
In this work, we propose an effective grouped attentive
re-identification (GAReID) framework composed of two novel
pooling layers, i.e., grouped high-order pooling (GHOP) and
attentive high-order pooling (AHOP). As we know, compared
with the first-order function, the high-order function f (x) =
x n (n > 1, x ≥ 0) contributes to amplifying the discrepancies
between two dependent variables when two independent variables are fixed. Motivated by this amplification property of the
high-order function, the essential idea behind GAReID is to
compute high-order mapping of part similarities to enlarge the
similarity discrepancies between aligned and misaligned part
pairs. Specifically, GAReID is able to highlight aligned part
similarities and suppress misaligned part similarities. Since
the high-order feature similarity between a pair of person
images is equivalent to an average of high-order similarities
of both the aligned and misaligned part pairs, the highorder aligned similarities are likely to dominate the high-order
feature similarities. In this way, the part misalignment problem
is effectively alleviated without relying on landmark detection
or feature partition.
Although high-order features contribute to part alignments,
the dimension of high-order features increases exponentially,
which gravely impairs the applications of high-order models.
Therefore, we need to design an effective feature compression
method for high-order features. Inspired by the lightweight
networks [24], [25], the proposed GHOP layer adopts channel
group and shuffle strategies to compress the dimension of
high-order features. Specifically, input feature channels are
uniformly divided into different groups and then those groups
are shuffled to disperse the information across feature groups.
Subsequently, we propose grouped Kronecker product (GKP)
to use the Kronecker product for subfeatures in each original and shuffled group to excavate informative high-order
interactions. Since the Kronecker product increases feature
dimensions in each group, we obtain grouped high-order
features by conducting elementwise aggregation, which can
significantly improve the effectiveness of high-order features.
As background clutters may hinder part alignments, we put
forward an effective foreground attention module named adaptive foreground attention (AFA) to preserve foreground regions
and eliminate background areas. With the integration of the
GHOP layer and the AFA module, the proposed AHOP layer is
constructed to boost both part-aligned and background robust
representation learning.
In summary, this article makes the following contributions.
1) We analyze the cause of part misalignments and prove
that high-order mapping of part similarities facilitates
fine-grained part alignments in theory.
2) We propose an effective GAReID framework with two
novel pooling layers, i.e., GHOP and AHOP. The GHOP
layer aims at compressing high-order features, while the
AHOP layer focuses on eliminating background clutters.
3) The GAReID framework is able to learn both partaligned and background robust representations without

3991

relying on any landmark detection or feature partition,
making it highly generalizable to other unknown pose
and background variations.
4) The GAReID achieves state-of-the-art ReID performance on the Market1501 [26], CUHK03 [27],
DukeMTMC [28], and MSMT17 [29] datasets.
II. R ELATED W ORKS
A. Person Re-Identification
For relieving part misalignments, prior ReID works can
be roughly summarized into two streams, i.e., part-based and
landmark-based methods. The part-based works [2], [5], [8],
[9], [10], [12], [13], [14], [30], [31] usually use deep neural
networks for learning discriminative local features. As global
features learned from the full image intend to capture the
coarse-grained clues of appearance, the global feature maps
in [2], [12], and [14] are equally divided into multiple horizontal patches to exploit local details. Based on PCB [2],
[12], some following works, i.e., MGN [5], PyramidNet [13],
and HPM [14], extract both global and local person representations by dividing convolutional feature maps horizontally
into multigrained patches. To enhance the part alignment of
learned representations, the landmark-based works [1], [6], [7],
[8], [9], [10], [11], [23] consider extra landmark knowledge
for training ReID networks. For instance, the GAN-based
works [6], [7] use auxiliary landmark annotations to guide
the generative model [32] to synthesize pose-specific person
images and supervise the identity encoder model to mine
pose-aligned features. The two-stream networks [33], [34] are
applied in [3] to independently generate appearance and pose
representations which are fused to enable part alignments.
To achieve a more precise alignment, the fine-grained pixellevel person semantics predicted by DensePose [35] are used
in [11] as an additional regularizer to guide part-aligned
representation learning from the original images. To solve the
occluded person ReID problem, Occluded-ReID [36] incorporates the pose information to make the ReID model focus
on the body region only and filter noise features brought by
occlusions.
In general, these methods use either local feature partition
or additional landmark information to align person features.
However, the part-based ReID approaches only achieve the
coarse-grained part alignments without considering detailed
pose variations within each part. In addition, it is nontrivial to obtain landmark-labeled person images or landmark
detection networks in real-world circumstances. Therefore,
the landmark-based models might not generalize well to
new images with unseen pose variations. In this work,
GAReID heads from a totally disparate but effective idea
that emphasizes the similarity discrepancies between aligned
and misaligned part pairs via a high-order mapping function.
Furthermore, our method is able to automatically rectify part
misalignments without depending on landmark information
or feature partition. Besides, the GAReID framework can be
applied to the field of unsupervised person ReID [37] and
helps unsupervised person ReID models to select more reliable

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.

3992

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

Fig. 2. Overview of the proposed GAReID framework. It consists of three parts, i.e., a backbone network, a GHOP layer, and an AHOP layer. The backbone
network is input with person images to extract convolutional feature maps. Then, we use a series of 1 × 1 convolutional layers and BatchNorm layers to
n
n
produce multiple input feature maps, i.e., {X i }i=1
and {Zi }i=1
. Next, those feature maps are fed into the GHOP and AHOP layers to generate high-order
image and foreground features, respectively. Output features are supervised by triplet loss during training, while we concatenate the two features to compute
cosine similarities during testing.

neighborhoods for each person image. As a result, the proposed GAReID framework has increased practical significance
and application prospect.

the representational capabilities of compressed high-order
features.
C. Attention Mechanism

B. High-Order Statistics
High-order statistics has been widely studied in traditional machine learning due to its powerful representation
ability. Recently, the fine-grained visual classification task
[38], [39], [40] has shown that the integration of high-order
features with deep networks can bring promising performance
improvements. For person ReID, Ustinova et al. [41] propose
an architecture based on the deep bilinear convolutional network. Chen et al. [42] construct a mixed-order attention module to use both low-order and high-order statistics in attention
mechanism, so as to produce discriminative attention proposals. Although the two architectures lead to some performance
improvements, they are not explicitly concerned with part
alignments.
Although high-order features exhibit strong representational
capabilities, the dimension of high-order features exponentially increases, which hinders their applications in realworld problems. Recently, several works [39], [40], [43], [44]
seek various feature compression methods to learn compact high-order features. For example, both CBP [43] and
KP [39] adopt random feature projections [45], [46], [47], [48],
but introduce constant projection matrixes, resulting in
additional nonnegligible memory overheads. Besides, they
use fast Fourier transform (FFT) and inverse fast Fourier
transform (IFFT) to simplify convolution operations, but
it may be discommodious to achieve FFT and IFFT on
deep learning frameworks. DBT [40] adopts tensor partition to capture intragroup interactions, but ignores intergroup interactions. Moreover, since both CBP and DBT
are second-order modules, they are unable to learn higher
order (n ≥ 3) features, which might severely weaken the
generalization capabilities of trained models. In this work,
GAReID use both the channel group and shuffle strategies
to achieve high-order feature compression, while promoting

The attention mechanism, inspired by the human sensing
process, has been studied extensively in various computer
vision tasks [49]. Specifically, an attention mechanism aims
at emphasizing informative regions for image representations, while depreciating harmful ones (e.g., background and
occluded regions). Interestingly, this approach is also efficient
and effective for ReID [42], [50], [51], [52], [53] because
it can handle person misalignments and background clutters.
For instance, HACNN [50] jointly learns hard region-level
attention and soft pixel-level attention in a unified attention
block. Mancs [51] considers both the channelwise and spatialwise attention in a fully attentional block, where the channel
information is recalibrated and the spatial structure information is also preserved. In addition, SONA [52] introduces
a second-order nonlocal attention network to directly model
long-range relationships via second-order feature statistics.
As distinguished from previous attention methods, our AFA
module generates foreground attention masks according to the
l2 norms of all the spatial features. Interestingly, the AFA
module significantly contributes to discovering useful semantic
regions without introducing any learnable parameters. By combining the proposed GHOP layer and AFA module, we build
the AHOP layer to jointly relieve part misalignments and
background clutters.
N
N N
N
N
Theorem N
1: Suppose n u = u u · · · u and n v =
N N
v v · · · vNare two nth-order vectors generated by Kronecker product with two input vectors
u and
v, the similarity
N
N
of nth-order vectors is computed by ⟨ n u, n v⟩ = ⟨u, v⟩n .
Proof: See Proposition 2 in [54].
□
III. P ROPOSED M ETHOD
In this section, we first analyze theoretically the cause of
part misalignments. Then we introduce GHOP and AHOP in
the GAReID framework as shown in Fig. 2.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 165 (2025) 111591

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Generalizable person re-identification method using bi-stream interactive
learning with feature reconstruction
Feng Min, Yuhui Liu ∗, Yixin Mao
School of Computer Science and Engineering, Wuhan Institute of Technology, Hubei 430074, China

ARTICLE

INFO

Keywords:
Pedestrian re-identification
Correlation graph sampling
Sparsely focused
Correlation reconstruction

ABSTRACT
Recent studies have shown that metric learning and representation learning are two main methods to improve
the generalization ability of pedestrian re-identification models. However, their relationship has not been
fully explored. Unlike GANs’ emphasis on adversarial learning, our objective is to develop an interactive and
synergistic learning framework for them. To achieve this, we propose a generalized pedestrian re-identification
method using bi-stream interactive learning. One of the learning streams is the correlation graph sampler
(CGS) for metric learning, and the other learning stream is the global sparse attention network (GSANet) for
representation learning. We establish an intrinsic connection between these two learning streams. Unlike many
existing methods that have high memory and computation costs or lack learning ability, CGS provides a more
efficient and effective solution. CGS uses local sensitive hashing and feature metrics to construct the nearest
neighbor graph for all categories at the beginning of training, which ensures that each batch of training samples
contains randomly selected base categories and their nearest neighbor categories, providing strong similarity
and challenging learning examples. As CGS sampling performance is affected by the quality of the feature map,
we propose a global feature sparse reconstruction module to enhance the global self-correlation of the feature
map extracted by the backbone network. Additionally, we extensively evaluate our method on large-scale
datasets, including CUHK03, Market-1501, and MSMT17, and our method outperforms current state-of-the-art
methods. These results confirm the effectiveness of our method and demonstrate its potential in pedestrian
re-identification applications.

1. Introduction
Pedestrian re-identification (Re-ID), has received significant attention due to its practical value in intelligent security [1], video surveillance [2], and urban management [3]. Pedestrian re-identification
models have achieved promising performance on certain datasets, but
applying them to unknown scenarios with large amounts of data remains highly challenging [4]. Existing deep learning-based methods
mainly focus on single-image feature representation learning, lacking
adaptability to unseen scenes [5]. While these models perform well
within a single dataset, their cross-dataset testing results are often
unsatisfactory, highlighting a gap in practical application.
To address these challenges, research in generalized pedestrian
re-identification has gained momentum. Cross-dataset testing and generalization have become important research directions [6], with efforts made in direct cross-dataset evaluation for benchmarking performance [7]. However, the field still faces significant challenges in adapting well-trained models to unknown scenarios, necessitating further
research in cross-library testing and generalization.

One area of current research focuses in enhancing the generalization
capability of pedestrian re-identification algorithms is metric learning.
It aims to design training objectives with different sampling strategies
and loss functions. Batch samplers play a crucial role in deep metric
learning [8], yet there is limited research in this area. The PK sampler
(Fig. 1(a)) is a widely used random sampling method in pedestrian reidentification [9]. However, this sampler exhibits global randomness,
resulting in uniformly distributed sampled examples across the entire
dataset in small batches.
The PK sampler randomly selects p classes and then samples k
images for each class to construct a small batch of size n = p × k.
However, the global randomness of this method makes it challenging
to provide relevant information for efficient deep metric learning.
Additionally, the small batch size obtained from the PK sampler does
not consider the relationship between samples. When using incomplete
random sampling, it becomes necessary to consider the relationships
between classes. If incomplete random sampling is used, it is necessary
to consider the relationships between classes.

∗ Corresponding author.

E-mail addresses: fmin@wit.edu.cn (F. Min), liuyuhui@wit.edu.cn (Y. Liu), mlaycn@163.com (Y. Mao).
https://doi.org/10.1016/j.patcog.2025.111591
Received 20 February 2024; Received in revised form 26 December 2024; Accepted 9 March 2025
Available online 18 March 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 165 (2025) 111591

F. Min et al.

Fig. 1. Two different sampling methods are shown in (a) PK sampler and (b) the proposed CGS sampler. Different shapes indicate different classes, while different colors represent
different batches. CGS sampler always samples the nearest neighboring classes. (For interpretation of the references to color in this figure legend, the reader is referred to the
web version of this article.)

To address the issue of lacking correlation in sampled samples,
some studies have employed dataset-level representations based on
average class embeddings and clustering techniques [10]. However,
these methods may have suboptimal performance when dealing with
a large number of classes. Another approach by [11] introduced a
graph sampler based on metric learning, which partially addressed the
issue of global randomness in sampling. However, this sampler exhibits unstable performance across multiple training iterations and has
high computational complexity, limiting its application to large-scale
datasets.
To overcome these challenges, we propose a method called Correlation Graph Sampler (CGS), as shown in Fig. 1(b). It leverages hash
encoding to group samples based on their correlation, providing a
coarse classification. To obtain the most relevant training instances,
we further employ feature-adaptive matching to compute correlations
among samples with the same hash encoding. This allows us to identify
the top P classes that are most similar to each base class. Finally,
we construct mutually independent nearest neighbor graphs for each
class. By adopting this approach, our method efficiently selects the most
relevant training instances while reducing computational complexity.
Note that, We conducted separate evaluations of our CGS and the
latest Graph Sampler [11] on the Tesla V100. The results revealed
that the latest GS sampler required 4 s and 40 s for sampling calculations on the Market dataset and the MSMT(all) dataset, respectively.
In contrast, CGS required only 0.1 s and 1 s for sampling assessments on the Market dataset and the MSMT(all) dataset, respectively.
Encouragingly, when faced with datasets containing a greater number of identities, CGS demonstrated stable and outstanding sampling
performance, significantly reducing computational complexity.
Moreover, the feature-adaptive matching method in CGS sampling
strategy is correlated with the feature maps extracted by the backbone
network, which endows CGS with learnability. Specifically, the sampling performance of CGS improves as the feature maps extracted by
the backbone network improve during training iterations. Inspired by
the sampling principle of CGS, we identified the potential of improving
the performance of CGS by enhancing feature representation learning. Based on this, we propose a novel high-resolution flow network,
named Global Sparse Attention Network (GSANet), to reduce the loss of
spatial positional information in the process of feature representation
learning. We also design a new global relevance sparse reconstruction
module (GRSR) to reconstruct the pixel-level features’ auto-correlation
of the feature layer, which enhances the backbone network’s feature
representation learning capability.
Therefore, we propose a bi-stream interactive learning framework.
One of the learning streams is the correlation graph sampler (CGS)
for metric learning, and the other learning stream is the global sparse
attention network (GSANet) for representation learning. We establish
an intrinsic connection between these two learning streams. On one
hand, CGS provides challenging training instances to enhance the
representation learning capability of the backbone network. On the
other hand, the improvement in feature map quality extracted by
the backbone network facilitates the enhancement of CGS sampling
performance, thereby achieving the desired interactive learning effect.
Our approach establishes a mutually reinforcing relationship between

metric learning and representation learning, which contributes to its
uniqueness. Furthermore, the traditional triplet loss used in person reidentification aims to reduce prediction error by refining the distances
between positive and negative sample pairs, which only considers the
relative distances between positive and negative pairs, while ignoring
the positive sample pairs themselves. To address this, we propose
the Matching Triplet Loss, which focuses chiefly on the relationship
between the matching relevance of positive and negative sample pairs.
In summary, the main contributions of our paper are:
• We propose a learnable batch sampling method called Correlation Graph Sampler (CGS) to provide more challenging training
instances for network training, aiding the model in discriminative
learning.
• We design a high-resolution preservation network called Global
Sparse Attention Network (GSANet) and introduce a global relevance sparse reconstruction module based on sparse representation (GRSR) to achieve feature self-correlation reconstruction.
This attention is global in nature and aims to reduce the loss
of semantic and positional information during downsampling,
thereby enhancing the model’s representation learning capability.
• This paper proposes a matching triplet loss, which is more advantageous for the training and optimization of the model with
respect to difficult samples for metric learning in pedestrian
re-identification.
2. Related work
The deep learning-based pedestrian re-identification methods integrate two modules: feature representation learning and metric learning.
That is, the extraction of image features and the comparison of feature
vectors’ similarity are completed in one model. Since pedestrian images
captured under different camera parameters and shooting environments differ significantly in terms of background, lighting, resolution,
viewpoint, and posture, extracting discriminative features and designing effective feature matching algorithms are crucial for addressing this
problem. Current research suggests that explicit deep feature matching,
strong feature extraction capabilities of models, as well as large-scale
and diverse training data can significantly improve the generalization
ability of person re-identification.
2.1. Feature representation learning
One of the current hot research topics in pedestrian re-identification
is still to investigate a suitable backbone network for this task [12].
In the early stages, researchers attempted to modify the ResNet50
backbone structure commonly used in image classification scenarios.
In recent years, researchers have designed network structures that are
more applicable to Re-ID scenarios, such as multi-scale and fine-grained
structures. [13] proposed an unbiased bidirectional convolutional neural network architecture to learn unbiased spatiotemporal representations, achieving certain effectiveness. [14] proposed a multimodal
graph neural network (MMGN) to explore potential data correlations,
2

Pattern Recognition 165 (2025) 111591

F. Min et al.

achieving promising results. However, it is limited by the constraints
of camera positions.
Some researchers combined attention mechanisms with global feature representation learning to enhance representation learning [15].
Local feature representation learning utilizes local image regions to
learn aggregated features, making it more robust to predict local misaligned scenes for pedestrians [16]. For instance, [17] utilized a method
that integrates data augmentation with a multi-label assignment strategy to achieve semantic feature decoupling in the source domain,
while [18] proposed a unsupervised domain adaptive person reID
framework, which adapts to general scenarios by computing the correlation between unlabeled samples in a single iteration. Additionally, Jin
also introduced a style normalization and recovery module [19], which
exhibits excellent generalization. [20] proposed an adversarial domaininvariant representation learning network (ADIN) that explicitly learns
to separate identity-related features from challenging variables. [21]
proposed the PCB method, which evenly divides the pedestrian feature
map into 6 blocks, extracts features using convolution instead of fully
connected layers for each block, and then connects each block to
a classifier. They also proposed the RPP method, which adaptively
partitions the edges based on the content similarity of each block, but
ignores the correlation between adjacent local blocks, which may lead
to discriminative information loss.
Additionally, [22] introduced memory-based multi-source metalearning to generalize to invisible domains. [23] confirmed the benefit
of improving network feature extraction through pixel-level feature
reconstruction. [24] has developed a domain-specific adaptive framework to enhance the model’s generalization capability to unseen target domains. [13] designed a heterogeneous local graph attention
network (HLGAT) to model both the local intra-relations and interrelations within local graphs, as well as the local relationships among
different parts of pedestrian images. Therefore, it is valuable to explore and improve the feature extraction ability of the network to
enhance the recognition and generalization performance of pedestrian
re-identification algorithms. Based on the research results of the above
scholars, it can be concluded that the feature representation learning ability of the network can greatly enhance the recognition and
generalization ability of person re-identification

In the literature [10], all training classes are divided into subspaces
by clustering on the average class representation, and then sampling
small batches in each subspace. This approach requires a full forward
pass of all training data, and the clustering operation does not scale
easily to large-scale classes. The literature [27] proposed the Rankingbased Backward Compatible Learning (RBCL), which aims to address
the issue that distillation-based methods force new feature spaces to
simulate poor old feature spaces. In the literature [28], SmartMining was proposed to build an approximate nearest neighbor graph
for feature extraction for all training samples after a full forward
pass of the training data. However, such instance-level mining can
be computationally very expensive and even unattainable for complex
non-Euclidean metric layers. Batch samplers play an important role in
the sampling efficiency and the learning effectiveness of the model [8].
The well-known PK sampler is the most widely used random sampling
method for pedestrian re-identification [9]. This sampler has global
randomness, resulting in the sampled examples in small batches being
uniformly distributed over the entire dataset.
Moreover, [26] proposed a sampling strategy called group sampling,
to alleviate the negative impact of noisy pseudo-labels on unsupervised
person re-identification models. The strategy gradually enhances the
statistical stability of feature representation by adjusting the overfitting problem in the learning process. [11] proposed a graph sampler
based on metric learning, which partially addresses the issue of global
randomness in PK sampling. However, this sampling method overly
relies on the effectiveness of the features extracted by the backbone
network. As training progresses, the features extracted by the backbone
network may inevitably focus on irrelevant local features, resulting in
a significant decline in sampler performance. Additionally, this metric
method is global in nature, and significantly increases computational
complexity, neglecting the importance of interactive learning between
metric learning and representation learning. To address the randomness
issue in the training samples, we propose a method called Correlation
Graph Sampling (CGS), which explores the correlation between samples
and provides challenging training instances for the backbone network,
to enhance the model’s representation learning capability.
3. Method

2.2. Deep metric learning

Fig. 2 illustrates the overall training framework of our algorithm.
During the training process of the deep learning model, we first use
the CGS sampler to select example samples with similar characteristics
from the training set and provide them to the backbone network for
training. The backbone network is then used for feature map extraction, and the weight parameters of the backbone network are updated
through the triplet loss function to optimize the model. Since the
performance of the CGS sampler is influenced by the quality of features
extracted by the backbone network, updating the weight parameters of
the backbone network during training can improve the performance of
the CGS.
In Section 3.1, we will explain the implementation principles of the
CGS sampler, while the structure of the main network (GSANet) will be
discussed in detail in Section 3.2. We utilize the matching triplet loss
function as the optimization loss in this model, and further details on
the construction principles of this loss function will be provided in the
subsequent chapters.

Apart from feature representation learning, metric learning is also
an effective approach to improve the performance of pedestrian reidentification models. However, research on metric learning with largescale training data is still not sufficient. Taking face recognition as an
example, the future direction may be gradually learning from even
larger-scale data to achieve better performance. Although mini-batch
sampling plays a crucial role in deep metric learning, it has not been
widely explored. In addition to online hard example mining for minibatch data sampling, several methods have been proposed for mining
hard examples during the mini-batch data sampling process.
Small-batch samplers also play an important role in deep metric
learning, but there is still little research on them. In addition to online
hard example mining in small batches, several methods have been
proposed for hard example mining in the process of small-batch data
sampling. For example, [25] proposed a random class-based hard example mining method for deep metric learning. It uses learnable class
signatures to identify the nearest classes and further performs instancelevel refinement search in a subset of classes found in the first stage
for hard example mining. In addition, [26] proposed a simple yet
effective method called Group Sampling to mitigate the negative impact
of noisy pseudo-labels in unsupervised person re-identification models.
However, this approach requires significant computational resources.
However, these methods require learning classification parameters for
class mining, which is difficult to handle for large-scale classes and
complex non-Euclidean matchers.

3.1. Correlation graph sampling
The CGS sampler aims to improve the discriminative ability and
generalization of learning models by constructing a nearest neighbor graph for all classes at the beginning of each epoch, as shown
in Fig. 1(b). Specifically, the sampler selects a random class as the
anchor point and its k nearest neighbor classes, ensuring that each
class has k nearest neighbors. This generates batches of instances
3


exec
/bin/zsh -lc "pdftotext -l 3 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112760

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Global aggregated gradient-guided adversarial attacks for person
re-identiﬁcation
Zeze Tao
a

a , Hui Li

a , Jinjia Peng

a,b,∗, Huibing Wang

c

School of Cyber Security and Computer, Hebei University, Hebei, China

b Hebei Machine Vision Engineering Research Center, China
c

College of Information Science and Technology, Dalian Maritime University, Liaoning, China

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Adversarial examples
Transferability
Global aggregated gradient
Gradient consistency

a b s t r a c t
Driven by deep neural network technology, person re-identiﬁcation (Re-ID) has achieved remarkable advancements. Recent studies demonstrate that adversarial examples can readily deceive Re-ID systems, achieving remarkable attack performance with imperceptible perturbations. However, this paper ﬁnds that these works suﬀer
from low gradient consistency during initial attack phases, severely limiting their eﬀectiveness. To address this
challenge, we propose a Global Aggregated Gradient-guided Attack (GAGA) method to further enhance the transferability of attacking Re-ID systems. Speciﬁcally, this paper performs gradient pre-convergence before each iteration to obtain the global aggregated gradient. Furthermore, rather than directly employing the current gradient
for momentum accumulation, this paper further considers the global aggregated gradient to adaptively adjust
the current gradient, thus improving the gradient consistency. Experimental results show that GAGA achieves
the best transferability when compared with the state-of-the-art methods for attacking Re-ID. Furthermore, the
integration of GAGA with various input transformation attack methods can further boost the adversarial transferability. Code is available at https://github.com/ZezeTao/GAGA.

1. Introduction
Person re-identiﬁcation (Re-ID) [1,2] aims to recognize and retrieve
speciﬁc individuals from intelligent surveillance systems. With the continuous advancement and development of deep learning, Re-ID technology has achieved signiﬁcant breakthroughs and is widely applied
in various surveillance systems. However, research [3,4] has indicated
that deep neural networks (DNNs) are highly vulnerable to adversarial examples. These adversarial examples can mislead DNN models by
adding imperceptible perturbations to normal images. This vulnerability
of DNN models seriously threatens the security and reliability of Re-ID
systems. Therefore, it is extremely important to conduct comprehensive
research on adversarial attacks against Re-ID systems.
Existing adversarial attack techniques [3,5] have been predominantly developed for image classiﬁcation tasks, and these methods have
achieved remarkable attack performance. However, the methods related
to classiﬁcation attacks are not suitable for attacking Re-ID systems. This
is primarily because the Re-ID task is an open-set task. In this task setting, there are diﬀerences in the identity information contained in the
training set and the test set, whereas the query set and the gallery set
share identity information.

Recent studies [6–8] have shown that Re-ID models are vulnerable to
adversarial example attacks, and numerous white-box [6–8] adversarial
attack methods have been proposed, which assume that all parameter
information of the model is known. Nevertheless, these attack methods
have limitations in real-world scenarios. When the parameters of the
target Re-ID model are not accessible, researchers have shifted their focus to exploring transferable adversarial examples for black-box Re-ID
models. Unlike transfer-based black-box attacks in classiﬁcation tasks,
Re-ID is a retrieval task with a more complex scenario. This complexity
leads to insuﬃcient transferability of the generated adversarial examples, making it diﬃcult to eﬀectively test the robustness of real-world
Re-ID models.
In order to successfully attack the Re-ID system, numerous adversarial attack strategies have been put forward for generating adversarial person examples. These strategies include metric-based attack methods [6–9], pseudo label-based attack methods [8,10], color-based attack
methods [7], and universal perturbation-based attack methods [11,12].
Among the above methods, the metric-based method is the most eﬀective and widely concerned attack method, which utilizes a reference
feature to distort the distance between the targeted person image and
other similar person images. However, the existing work [6–8] ignores

∗ Corresponding author.

E-mail addresses: zeze@hbu.edu.cn (Z. Tao), lihui15794@hbu.edu.cn (H. Li), pengjinjia@hbu.edu.cn (J. Peng), huibing.wang@dlmu.edu.cn (H. Wang).
https://doi.org/10.1016/j.patcog.2025.112760
Received 7 May 2025; Received in revised form 6 October 2025; Accepted 16 November 2025
Available online 20 November 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112760

Z. Tao et al.

paradigm. At present, the practical deployment of Re-ID technology encounters a multitude of challenges. These encompass issues such as partial or complete occlusion of pedestrians, variations in viewpoints and
lighting conditions, the presence of highly similar appearances among
diﬀerent pedestrians, and the inherent diﬃculties associated with data
acquisition in real-world scenarios. In recent years, the advent of deep
learning has catalyzed a paradigm shift in Re-ID research, ushering
in revolutionary advancements. Researchers have diligently explored a
range of methodologies, including feature learning [13], metric learning
[14], and ranking optimization techniques [15], with the aim of elevating the performance of Re-ID systems. During the training phase, classiﬁcation loss [16] or triplet loss [17] is employed to optimize the neural
network, thereby enhancing the discriminative power of features. In the
inference phase, the similarity between the query image and the gallery
images is calculated using cosine distance or Euclidean distance. These
endeavors have culminated in notable progress and signiﬁcant breakthroughs, as evidenced by their successful application across multiple
publicly available datasets.

Fig. 1. Visualization of the gradient update at the 𝑡-th iteration, where 𝑔𝑡 represents the current gradient, 𝑔𝑡𝐴𝑔𝑔 denotes the global aggregated gradient, 𝑔𝑡𝐹 𝑖𝑛𝑎𝑙
stands for the ﬁnal gradient, and 𝑐𝑡 indicates the gradient consistency between
𝑔𝑡 and 𝑔𝑡𝐴𝑔𝑔 .

2.2. Adversarial attacks
Adversarial attacks are primarily classiﬁed into two major categories
based on their attack methodologies: white-box attacks [4] and blackbox attacks [5,18,19]. In white-box attacks, attackers have the ability
to acquire the internal architecture and parameter details of the victim
model. Conversely, in black-box attacks, attackers are unable to gain access to the speciﬁc particulars of the victim model. In real-world attack
scenarios, it is often the case that attackers encounter signiﬁcant diﬃculties in obtaining the detailed information of the target Re-ID model.
Consequently, they are compelled to conduct attacks in black-box settings. An eﬀective approach in such situations is to utilize the adversarial
samples generated by surrogate models to launch attacks against other
black-box models. This characteristic is vividly described as the transferability of adversarial samples. Nevertheless, while existing adversarial attacks demonstrate outstanding performance in white-box attacks,
their transferability is considerably low when it comes to attacking other
black-box models.
Adversarial attacks are primarily employed in the ﬁeld of image classiﬁcation, where imperceptible perturbations are introduced to genuine
images to deceive trained models, thereby facilitating robustness evaluation. Since Szegedy et al. [20] ﬁrst revealed the vulnerability of deep
neural networks (DNNs) to adversarial examples, researchers have developed various attack methodologies, including transfer-based attacks
[5,19,21], score-based attacks [22,23], and decision-based attacks [24].
Among these, transfer-based attacks have gained particular prominence
in real-world scenarios due to their black-box nature, requiring no access
to the target model’s internal information. Recent years have witnessed
signiﬁcant advancements in enhancing the transferability of adversarial
attacks, with most approaches building upon the Iterative Fast Gradient Sign Method (IFGSM) [4] framework. For instance, the MIFGSM [3]
method integrates momentum into IFGSM to produce more transferable
adversarial examples; the VMI [18] approach further improves transferability by minimizing gradient variance; and GRA [21] enhances performance through gradient alignment with neighborhood gradients. Meanwhile, BSR [19] disrupts attention heatmaps to boost attack eﬀectiveness. Recent works [5] has also demonstrated that placing adversarial
examples in ﬂat regions of the loss landscape can signiﬁcantly improve
their transferability. Additionally, recent work [25,26] has also been
dedicated to improving the transferability of attacks on vision-language
pre-training models.
These methods collectively operate by perturbing model outputs at
the logit level to maximize the deviation of predicted classes from their
ground-truth categories. Consequently, such approaches are only applicable to classiﬁcation tasks and cannot be directly applied to open-set
ranking task like Re-ID.

the consistency between the current gradient and the global aggregated
gradient and fails to make full use of the information of the global aggregated gradient. As depicted in Fig. 3, prior attack methods [7,8] exhibit
low consistency between the current gradient and the global aggregated
gradient during the initial attack phase, which undermines the eﬃcacy
of attacking Re-ID systems.
In this work, we propose a Global Aggregated Gradient-guided Attack (GAGA) method to address the low gradient consistency issue during the attack process. Unlike recent research that only considers the
current gradient information during each update process, our work further incorporates the information of globally aggregated gradients. As
depicted in Fig. 1, prior to each iteration, this paper initially executes
an internal loop (that is, performs the gradient pre-convergence operation), and then takes the average of all the gradients obtained in this
internal loop as the global aggregated gradient. Furthermore, this work
innovatively establishes a gradient consistency factor to extract latent
information from the globally aggregated gradient. This factor serves as
an eﬀective metric for quantifying the correlation between the current
gradient and the global aggregated gradient. In each iteration, GAGA
adaptively determines the update direction based on variations in the
gradient consistency factor.
To summarize, the principal contributions are presented as follows:
•

To the best of our knowledge, this work is the ﬁrst to reveal that
low gradient consistency can limit the attack performance on Re-ID
systems.
• To enhance gradient consistency, this paper proposes a novel adversarial attack method for Re-ID systems, termed Global Aggregated
Gradient Attack (GAGA), which is capable of generating highly transferable adversarial person images.
• Compared with the state-of-the-art methods for attacking Re-ID systems, the proposed GAGA exhibits the optimal attack performance.
Moreover, it can be combined with input transformation-based attack techniques to further improve transferability.
2. Related work
2.1. Person re-identiﬁcation
Person Re-identiﬁcation (Re-ID) stands as a pivotal research domain
within the ﬁeld of computer vision, with its primary objective centered on the precise retrieval of speciﬁc pedestrian images across nonoverlapping camera networks in surveillance environments. This task inherently represents a ﬁne-grained subset of the broader image retrieval
2

Pattern Recognition 172 (2026) 112760

Z. Tao et al.

2.3. Adversarial attacks on person re-identiﬁcation

the transferability of attacks against Re-ID systems. The overall framework of the GAGA attack on the Re-ID system is presented in Fig. 2.
The right half of the ﬁgure illustrates the process of generating one adversarial query from one clean query, whereas the left half displays the
retrieval results of both the clean query and the adversarial query. GAGA
primarily comprises two key components: 1) Global Aggregated Gradient: We introduce an inner loop before each update to obtain the global
aggregated gradient, which provides a global optimization direction. 2)
Gradient Consistency: We innovatively employ the global aggregated
gradient to guide the current gradient optimization process, thereby enhancing gradient consistency during the initial attack phase.

To eﬀectively attack person re-identiﬁcation (Re-ID) models, researchers have developed various adversarial strategies. Bai et al. [6] pioneered an adversarial metric attack that crafts misleading gallery samples by maximizing the feature distance between clean and perturbed
images. Wang et al. [27] developed physically realizable adversarial
patterns customized for clothing items to attack Re-ID systems, while
Wang et al. [28] employed a GAN-based approach to generate deceptive query images. The Multi-Expert [29] Adversarial Attack leveraged
contextual inconsistencies to fool DNN-based Re-ID systems, while the
Furthest-Negative Attack [9] employed hard sample mining and triplet
loss to optimize image feature movement and enhance attack performance. The Meta Attack [30] generated universal adversarial perturbations that eﬀectively deceived Re-ID models across multiple domains.
Meanwhile, the Local Transformations Attack [7] disrupted retrieval
performance by strategically altering image color distributions rather
than injecting random noise. ODFA [8] utilized feature-level adversarial gradients to drive feature representations in opposing directions. In
recent years, a substantial body of research [31–33] has emerged on
adversarial attacks targeting cross-modal Re-ID models. However, these
attack methods neglect gradient consistency, which consequently limits
their transferability when attacking Re-ID systems.

3.2.1. Global aggregated gradient
To obtain global gradient information, this paper introduces a preconvergence attack strategy, which incorporates an inner loop before
each iteration without exerting any substantive impact on the image
during this process. The attack process of the pre-convergence strategy
at the 𝑡-th iteration is illustrated in Fig. 1. The purple arrow represents
the gradient of the current point, while the blue arrow represents the
pre-gradient. The trajectory of the blue arrow from the initial point to
the predicted point constitutes the pre-convergence attack process. In
this paper, the average of all pre-gradients in the inner loop is deﬁned
as the global aggregated gradient.
Speciﬁcally, let 𝑥adv
denote the input at the 𝑡-th iteration. In this
𝑡
paper, an inner loop is introduced at the 𝑡-th iteration to compute the
global aggregated gradient. The expression for the predicted point 𝑥inner
𝑖
within the inner loop is formulated as follows:

3. Methodology
3.1. Problem deﬁnition

𝑥inner
= 𝑥𝑎𝑑𝑣
+ 𝛿𝑖inner + 𝑟𝑖 ,
𝑖
𝑡

In Re-ID systems, both query and gallery images are potential targets for adversarial attacks. However, in real-world scenarios, accessing gallery images is often challenging and resource-intensive, whereas
query images can be easily manipulated–either directly or during capture. Given this asymmetry, our work focuses on generating adversarial
queries to subvert Re-ID systems eﬀectively.
Given a clean query image 𝑥 and a true match gallery image 𝐺, let
𝑓 (𝑥) denote the embedding feature extracted from 𝑥 by the victim Re-ID
model. In an ideal Re-ID system, the cosine similarity score between the
features 𝑓 (𝑥) and 𝑓 (𝐺) should be as high as possible. To attack such a
system, the objective of adversarial attacks is to introduce an imperceptible perturbation 𝜖 to the clean query image 𝑥, generating an adversarial sample 𝑥adv = 𝑥 + 𝜖, such that the cosine similarity score between
the adversarial sample’s feature 𝑓 (𝑥adv ) and 𝑓 (𝐺) is as low as possible.
Since the cosine similarity score between the original query feature 𝑓 (𝑥)
and its opposite direction −𝑓 (𝑥) is the lowest, during the optimization
process, we push the feature of the adversarial sample 𝑓 (𝑥adv ) towards
the opposite direction of the initial sample’s feature −𝑓 (𝑥), and the optimized distance loss can be formulated as:
(
)
‖ ( 𝑎𝑑𝑣 )
‖2
𝐽 𝑥𝑎𝑑𝑣 , 𝑥 =
min
− (−𝑓 (𝑥))‖
‖𝑓 𝑥
‖2
‖𝑥𝑎𝑑𝑣 −𝑥‖ ≤𝜖 ‖
‖
‖∞
(1)
(
)
2
‖
‖
=
min
‖𝑓 𝑥𝑎𝑑𝑣 + 𝑓 (𝑥)‖ ,
‖2
‖𝑥𝑎𝑑𝑣 −𝑥‖ ≤𝜖 ‖
‖
‖∞

(2)

where 𝛿𝑖inner is the inner adversarial perturbation at the 𝑖-th iteration,
where 𝑖 = 1, 2, … , 𝑚, and 𝑚 denotes the number of iterations in the inner
loop. The initial condition is 𝛿1inner = 0. Additionally, 𝑟𝑖 is the random
[
]
[
]
noise satisfying 𝑟𝑖 ∼ 𝑈 −(𝛽 ⋅ 𝜖)𝑑 , (𝛽 ⋅ 𝜖)𝑑 , where 𝑈 𝑎𝑑 , 𝑏𝑑 represents the
uniform distribution, and 𝑑 is the dimension.
inner of the internal prediction point
Next, the current gradient 𝑔𝑖+1
inner
𝑥𝑖
is computed, and its expression is given as follows:
(
)
inner
𝑔𝑖
= ∇𝑥inner 𝐽 𝑥inner
,𝑥 .
(3)
𝑖
𝑖

inner by utilizSubsequently, we update the internal perturbation 𝛿𝑖+1
inner
ing the current internal gradient 𝑔𝑖+1 , and the formula is deﬁned as
follows:
(
(
))
inner
𝛿𝑖+1
= Clip𝜖𝛿 𝛿𝑖inner + 𝛼 ⋅ sign 𝑔𝑖inner ,
(4)

where 𝛼 denotes the step size, and the Clip(⋅) operation restricts the
perturbation amplitude under the 𝑙∞ norm.
Finally, by averaging all gradients within the internal loop, the global
Agg
aggregated gradient 𝑔𝑡+1 is obtained, and its speciﬁc calculation formula
is given as follows:
Agg

𝑔𝑡+1 =

where 𝐽 is the loss function, ‖ ⋅ ‖∞ represents the ∞-norm, and 𝜖 is the
perturbation bound.
In real-world adversarial attacks, attackers typically do not have direct access to the victim model. To circumvent this limitation, a surrogate model is employed to craft adversarial queries, which are subsequently transferred to the victim model. Consequently, the transferability of adversarial queries emerges as a pivotal factor determining the
success of such attacks, constituting the central focus of this study.

𝑚
𝑚
(
)
1 ∑ inner
1 ∑
𝑔𝑖
=
∇ inner 𝐽 𝑥inner
,𝑥 .
𝑖
𝑚 𝑖=1
𝑚 𝑖=1 𝑥𝑖

(5)

By means of the pre-convergence attack mechanism, we are able
to comprehensively capture the gradient information of all data points
along the pre-convergence path, thereby providing a global guiding direction for the optimization process.
3.2.2. Gradient consistency
Existing methods [6–8] can eﬀectively generate adversarial samples
that deceive Re-ID systems by leveraging gradient information. However, this paper reveals a signiﬁcant problem in the initial attack stage
(Fig. 3), from an experimental perspective: the current gradient direction shows nearly complete inconsistency with the global aggregated gradient direction. This inconsistency causes severe oscillations
in the update direction during early iterations, indicating substantial

3.2. Overall framework
This paper proposes a novel Global Aggregated Gradient Attack
method (GAGA), which aims to stabilize the optimization process by introducing global gradient information, thereby signiﬁcantly enhancing
3


codex
第二组读完。这里有两类值得反推：一类用“高阶相似性”绕开显式姿态对齐，一类把训练采样本身定义成可学习结构；攻击论文虽然不是正向 ReID 方法，但它给了一个可用的鲁棒性首验角度。继续读最后三篇。
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
/bin/zsh -lc "pdftotext -l 3 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

1

Harnessing Knowledge From Pretrained VLMs
for Unsupervised Person Search
Yanling Tian , Shanshan Zhang , Member, IEEE, Di Chen, and Jian Yang , Member, IEEE
Abstract—Person search is a unified task that includes the
subtasks of pedestrian detection and re-identification (re-ID).
It is expensive to label pedestrian bounding boxes and person
identities for training. Purely unsupervised (US) person search is
more practical for real-world situations. However, it is difficult
to obtain accurate pseudo-IDs from low-quality pseudo-boxes,
which brings a new challenge. To address this issue, we propose
FMUPS, a novel method that leverages semantic information to
produce reliable pseudo-labels. Semantic representations, particularly from vision-language models (VLMs), provide clear
and interpretable guidance, reducing noise caused by background disturbances during pseudo-label extraction. Despite
their advantages, VLM-generated pseudo-boxes often suffer from
poor alignment with person regions and misclassification of other
objects as people, adversely affecting the re-ID task. To overcome
these issues, we introduce an anti-bbox-noise re-ID loss that not
only alleviates the above localization and classification noises but
also helps to acquire effective re-ID features. In addition, we
propose a CLIP ID labeler, which exploits text–image alignment
capabilities of VLMs to generate pseudo-IDs based on our
predefined attributes and iteratively refines them using prior
knowledge of person search. The experimental results on two
typical benchmarks, CUHK-SYSU and PRW, demonstrate the
effectiveness of our method; in particular, we outperform some
previous fully and weakly supervised (WS) methods.
Index Terms—Anti-noise, person search, robust semantics,
unsupervised (US) learning, vision-language models (VLMs).

I. I NTRODUCTION

P

ERSON search is the task of simultaneously identifying
and localizing a target individual from a collection of
uncropped, realistic scene images. This task has gained significant attention due to its practical applications in public
safety, such as video surveillance [1]. Person search can be

Received 10 January 2025; revised 9 September 2025, 15 October 2025,
and 13 January 2026; accepted 17 April 2026. This work was supported in part
by the National Natural Science Foundation of China under Grant 62322602,
Grant U24A20330, and Grant 62361166670; and in part by the Natural
Science Foundation of Jiangsu Province, China, under Grant BK20230033.
(Corresponding authors: Shanshan Zhang; Jian Yang.)
Yanling Tian is with the PCA Laboratory, School of Computer Science and
Engineering, Nanjing University of Science and Technology, Nanjing 210094,
China, and also with the Graduate School of Information, Production and
Systems, Waseda University, Kitakyushu, Fukuoka 808-0135, Japan (e-mail:
yl.tian@njust.edu.cn).
Shanshan Zhang and Jian Yang are with the PCA Laboratory, School
of Computer Science and Engineering, Nanjing University of Science and
Technology, Nanjing 210094, China, and also with the PCA Laboratory,
School of Intelligence Science and Technology, Nanjing University, Nanjing
210093, China (e-mail: shanshan.zhang@njust.edu.cn; csjyang@njust.edu.cn).
Di Chen is with the PCA Laboratory, School of Computer Science and
Engineering, Nanjing University of Science and Technology, Nanjing 210094,
China (e-mail: dichen@njust.edu.cn).
This article has supplementary downloadable material available at
https://doi.org/10.1109/TNNLS.2026.3686858, provided by the authors.
Digital Object Identifier 10.1109/TNNLS.2026.3686858

divided into two subtasks: pedestrian detection and person
re-identification (re-ID) [2], [3], [4], [5], [6], [7], [8]. Thus,
typically, one needs both pedestrian bounding boxes and
identity labels for training a person search model in a fully
supervised (FS) way. However, it is too expensive to acquire
these annotations, which sometimes even raises privacy concerns. To mitigate the reliance on costly manual labels, weakly
supervised (WS) methods have been proposed [9], [10], [11],
which only use pedestrian bounding boxes for training. Yet,
the labeling cost is still high, especially in dense scenes, where
they are a large number of pedestrians to annotate.
In order to fully get rid of manual labels, in this work, we
propose a new setting of unsupervised (US) learning for person
search. First and foremost, we claim that US person search is
not a simple combination of US pedestrian detection [12], [13]
and US re-ID [14], [15]. The additional challenge mainly lies
in the fact that accurate re-ID pseudo-labels become more difficult to achieve due to the low-quality pseudo-boxes. Therefore,
our investigation focuses on generating high-quality pseudolabels, i.e., pseudo-boxes and pseudo-IDs, considering the
relationship between two subtasks. One straightforward way is
to generate pseudo-boxes and pseudo-IDs using state-of-theart pedestrian detectors (e.g., FeatComp [16]) and clustering
algorithms. However, on the one hand, these pseudo-boxes
often exhibit two primary types of noise: 1) localization noise,
where bounding boxes fail to accurately align with person
regions [Fig. 1(a)] and 2) classification noise, where nonperson
objects (e.g., a stroller and bicycle) are incorrectly classified
as persons [Fig. 1(b)]. On the other hand, traditional clustering
methods, such as DBSCAN, have shown limited effectiveness
when handling noisy pseudo-boxes, resulting in only 31.56
pp with respect tomean average precision (mAP) on CUHKSYSU [2]. These challenges become even more difficult to
address in complex scenes, where the background induces
severe disturbances.
Considering the above challenges, we are highly motivated
to use vision-language models (VLMs), in which the natural
language offers a complementary pathway, providing explicit
and unambiguous semantics that are helpful for pseudo-label
generation, against background noise. Specifically, on one
hand, VLMs serve as a bridge between visual features and
textual semantics, facilitating the integration of semanticlevel understanding into pseudo-label generation; on the other
hand, VLMs exhibit remarkable generalization capabilities
across various vision tasks [17], [18]. However, achieving
high-quality pseudo-labels directly through VLMs remains
nontrivial. For example, while SEEM-generated pseudo-boxes
demonstrate superior detection precision through semantic

2162-237X © 2026 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
2

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

and a CLIP ID labeler that aligns text and image features
to produce more accurate and reliable pseudo-IDs.
3) Experiments are conducted on PRW and CUHK-SYSU
datasets. Under the US setting, we establish new baseline
results, where our performance on PRW is competitively
close to and even surpasses some FS and WS methods.
These encouraging results would motivate more work in
this direction.
II. R ELATED W ORK
As this article develops person search methods using cues
from foundation models, we review relevant work from both
of the above perspectives.
A. Person Search

Fig. 1. Qualitative and quantitative analysis of boxes predicted by SEEM [19].
Green bboxes refer to the ground truth while red ones denote SEEM predicted
bbox. (a) and (b) Samples of low-quality localization and classification
mistakes. (c) Comparison of various foundation models (SEEM [19], SSA
[20], and CLIP+SAM1 ) and a state-of-the-art pedestrian detector (Featcomp
[16] pretrained on CityPersons dataset [21]).

prompts (e.g., “person” and “pedestrian”) [Fig. 1(c)], they
are still prone to localization and classification noises, which
negatively impact the quality of pseudo-IDs. To mitigate
these disturbances, we propose an anti-bbox-noise re-ID loss,
which drives the model to focus on the foreground person
regions and downweights those boxes with lower confidence
scores in the loss function, effectively reducing the impact of
poor localization and misclassification noise. In addition, we
propose a CLIP ID labeler, which is more robust to noisy
pseudo-boxes by making use of the alignment between image
and text features. Specifically, we first construct one sentence
consisting of predefined attributes for each ID. Subsequently,
each instance can be assigned as a pseudo-ID, whose text
embedding is closest to the instance’s visual features. The
primary pseudo-IDs are further refined by the prior knowledge
of person search, i.e., individuals in the same photo cannot
have identical IDs.
In summary, our contributions can be summarized as follows.
1) We for the first time propose a new setting of US
learning for person search and point out the unique
challenge over previous WS person search, i.e., the lack
of accurate person boxes brings great challenge to the
subsequent subtask of re-ID.
2) We introduce a novel framework leveraging robust
semantics of VLMs to generate and refine pseudolabels for person search. Specifically, we propose an
anti-bbox-noise re-ID loss to mitigate localization and
classification noise in SEEM-generated pseudo-boxes
1 Due to the lack of semantic information in SAM [19], [22], we use SAM
to obtain bboxes and CLIP [23] to remove the bboxes whose category is not
person in the CLIP+SAM method.

Person search has witnessed significant progress in recent
years, driven by its extensive potential applications. Existing works predominantly fall into two categories: two-stage
methods, which optimize pedestrian detection and re-ID tasks
separately [1], [24], [25], [26], [27], [28], [29], and one-stage
methods, which unify these subtasks by jointly optimizing
them in an end-to-end manner [2], [3], [4], [5], [6], [7], [8],
[9], [10], [11], [30], [31], [32], [33], [34], [35], [36], [37],
[38]. Broadly, current person search methods can be classified
into two distinct groups based on their reliance on the labels:
FS and WS methods.
1) Fully Supervised: This category leverages comprehensive
annotations, including pedestrian bounding boxes and identity
labels, to train models directly [3], [4], [6], [7], [8], [36],
[37], [38]. Xiao et al. [2] pioneer a unified framework that
integrates re-ID layers atop the Faster-RCNN detector [39].
Chen et al. [3] introduce a norm-aware embedding (NAE)
method to harmonize the divergent optimization goals of
detection and re-ID. In addition, recent approaches [36], [37]
incorporate transformers to extract more discriminative feature
representations, thereby enhancing performance.
2) Weakly Supervised: Due to the challenge of obtaining
identity labels, this group of approaches utilizes only bounding
box annotations for training [9], [10], [11]. Yan et al. [9]
exploit contextual information to derive discriminative features
for a robust US re-ID task. Wang et al. [11] introduce a
multiscale exemplar branch and devise a scale-invariant loss
to tackle scale variability issues.
3) Unsupervised: Unlike the aforementioned methods, US
person search eliminates the need for accurate annotations,
focusing on learning effective feature representations from
fully unlabeled datasets. Different from US domain-adaptive
person search methods [40], which rely on labeled sourcedomain data together with unlabeled target-domain images
for training, this setting does not assume the availability of
any labeled data from any domain. While no US methods
are specifically designed for person search under this strictly
label-free setting, various techniques [12], [13], [41], [42]
exist for its subtasks, including pedestrian detection [12], [13],
and re-ID [14]. Liu et al. [12] propose a US multiplane
detection (UMPD) method that removes the necessity for
pedestrian bounding box annotations through 2-D-3-D mapping. Han et al. [14] develop an innovative sampling strategy

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
TIAN et al.: HARNESSING KNOWLEDGE FROM PRETRAINED VLMs FOR US PERSON SEARCH

3

to refine the re-ID pseudo-label generation process. However,
the integration of pedestrian detection and re-ID in a US
setting presents unique challenges, particularly as detection
performance and the presence of noisy pseudo-boxes can
significantly influence the quality of person representations.
This aspect remains underexplored in existing literature. In
this work, we introduce a US setting to person search for the
first time, aiming to utilize cutting-edge foundation models to
address the challenge of US person search.
B. Foundation Models
Foundation models have rapidly advanced and been applied
across diverse domains, spanning interactive segmentation
models (SAM [22] and SEEM [19]), VLMs (CLIP [23]),
and human-centric approaches (SOLIDER [43], UniHCP [44],
and PATH [45]). Among these, SAM establishes a new
paradigm for segmentation by supporting diverse prompt
inputs, while SEEM further enhances its semantic understanding and prompting capabilities. CLIP has gained significant
attention due to its strong zero-shot recognition performance,
leading to its application in diverse vision tasks, including
person re-ID, crowd counting, and semantic segmentation.
Researchers have creatively adapted its cross-modal alignment
capabilities. For instance, Li et al. [18] develop CLIP-ReID,
which utilizes learnable text tokens for identity description,
while Liang et al. [17] reformulate crowd counting as an
image–text matching task. Human-centric models like UniHCP, PATH, and SOLIDER demonstrate the potential of
integrating diverse datasets to build comprehensive humanfocused models. These approaches not only provide robust
human representations but also enhance performance across
downstream visual tasks.
Beyond task-specific applications, researchers explore foundation models as sophisticated data engineering tools. SSA
[20], for instance, serves as an automated dense openvocabulary annotation engine, combining closed-set segmentation [46], open-vocabulary techniques [47], [48], and
intelligent class filtering [23]. Inspired by such innovations,
our work investigates foundation models’ potential in guiding
US person search.
III. M ETHOD
In this section, we begin with an overview of our one-stage
US person search method, termed FMUPS-S1, followed by an
explanation of the proposed CLIP ID labeler. Due to the noise
introduced by pseudo-boxes, we present in detail our antibbox-noise re-ID loss. In addition to FMUPS-S1, we present
another implementation: a two-stage approach, FMUPS-S2.
The details and individual components are described in
Section III-E.
A. Overview
The VLMs are used to provide high-quality pseudo-labels
for both detection and re-ID. As depicted in Fig. 2, our
FMUPS-S1 architecture includes two streams: a scene stream
and an instance stream. In the instance stream, each image
x is first passed through the frozen SEEM model, where we

Fig. 2. Overview of our one-stage method FMUPS-S1. The pipeline consists
of a scene stream (dashed lines) and an instance stream (solid lines).
Specifically, SEEM is employed for generating pseudo-boxes, while the
CLIP ID labeler is utilized to provide pseudo-IDs. Our novel anti-bboxnoise re-ID loss effectively reduces the negative effect of both classification
and localization noise by leveraging confidence scores ci and emphasizing
foreground information, respectively. RPN is the region proposal network [39].

obtain all person masks by utilizing the person-related prompts
(e.g., “person”). We then derive the pseudo-box y
bbi = ϕ(mi )
for each mask mi , where ϕ(·) denotes the function that computes the coordinates of the minimum bounding rectangle.
Subsequently, the image x is cropped according to y
bb to
b
b
b
extract box patches x = {x1 , . . . , xN }, which are then further
processed with corresponding masks to derive the foreground
patches xm = {x1m , . . . , xmN }, where the background regions are
filled with zero. Here, N is the number of bounding boxes
in an image. These patches, xm and xb , are subsequently
input into the image encoder to extract respective box features
f b = { f1b , . . . , fNb } and mask features f m = { f1m , . . . , fNm }. The
box features f b are then processed through the CLIP ID labeler
to generate and further refine pseudo-IDs. Concurrently, in
the scene stream, the entire image x is sent to the image
encoder shared with the instance stream, followed by an RPN,
a detection head, and a re-ID head. RPN aims to efficiently
generate high-quality proposals from the entire image that
are likely to contain pedestrians. The detection and re-ID
heads follow OIM [2]. After that, we obtain proposal features
f p = { f1p , . . . , f Jp } along with their associated confidence
scores, where J represents the number of predicted proposals.
In order to reduce the negative impact of noisy pseudo-boxes
on the performance of re-ID, we propose an anti-bbox-noise
re-ID loss. The use of these features, f b , f m , and f p , in this
loss allows the model to focus on the foreground information,
thereby mitigating localization noise, while the utilization
of confidence scores aids in diminishing classification noise
induced by pseudo-boxes. Further elaboration on the CLIP
ID labeler and anti-bbox-noise re-ID loss is provided in
Sections III-B and III-C.
B. CLIP ID Labeler
The CLIP ID labeler is designed to generate accurate
pseudo-IDs for each sample as shown in Fig. 3. It comprises

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5022

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

Heterogeneous Generative Tokens and
Distance-Aware Recovery Network for Occluded
Person Re-Identification
Zhihao Li , Huaxiang Zhang , Lei Zhu , Jiande Sun , and Li Liu

Abstract— In real-world surveillance scenarios, person
re-identification tasks are often seriously affected by occlusion
problems, which requires the model to be able to not
only extract powerful features, but also effectively recover
features when they are occluded. Although existing methods
disentangle visible human bodies by clustering semantic
information, they often damage discriminative appearance
due to the introduction of background noises. To solve this
problem, we propose Heterogeneous Generative Tokens and
Distance-aware Recovery (HGTDR) network, which aims to
effectively extract discriminative appearance and recover the
occluded body regions. HGTDR mainly contains two branches:
a holistic stream and a part stream. The holistic stream utilizes
ViT to capture the global context information and provide
stable global features by establishing long-range relationships.
In the part stream, we propose a Semantic Patch Generator
(SPG), which combines the local attention mechanism to
capture rich local semantics and further generate semantic
patches. Further, considering the discrimination score and
relevance score of semantic patches, we feed them into the
proposed Adaptive Heterogeneous Semantic Token Generator
(AHSTG) to gradually generate strong-response foreground and
weak-response background features. In addition, to complete
the features of occluded regions, the Distance-based Feature
Recovery (DFR) module is designed. The module calculates the
planar Euclidean distance of heterogeneous tokens and adaptively
allocates the corresponding weights to dynamically recover the
invisible bodies. Finally, we obtain discriminative and robust
person descriptors. Extensive experiments on several challenging
occluded, partial and holistic Re-ID datasets demonstrate that
our proposed HGTDR network achieves superior performance
and outperforms various state-of-the-art methods.
Index Terms— Occluded person re-identification, occlusion,
semantic token, feature recovery.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) aims to retrieve the
target pedestrian from a collection of images captured

Received 21 May 2024; revised 1 December 2024; accepted 14 December
2024. Date of publication 17 December 2024; date of current version 7 May
2025. This work was supported in part by the National Natural Science
Foundation of China under Grant 62176144 and Grant 62076153; in part by
the Major Fundamental Research Project of Shandong, China, under Grant
ZR2019ZD03, Grant ZR2024ZD08, and Grant ZR2024MF043; and in part
by Taishan Scholar Project of Shandong, China, under Grant ts20190924.
This article was recommended by Associate Editor B. Bao. (Corresponding
author: Li Liu.)
The authors are with the School of Information Science and Engineering, Shandong Normal University, Jinan 250358, China (e-mail:
liuli_790209@163.com).
Digital Object Identifier 10.1109/TCSVT.2024.3519312

by a series of non-overlapping distributed cameras. Due to
its practical significance in fields such as missing persons
tracking, smart cities, and intelligent security, person Re-ID
has always been a key research topic in the pattern recognition
field. In the early years, various person Re-ID researches [1],
[2], [3], [4], [5] have made significant progress. The majority
of previous general Re-ID works assume that there are no
occlusions in the images [6], [7], [8], [9], [10]. However,
in real-world scenarios, the captured person images often
contain occlusions such as vehicles, billboards, trees, luggage,
etc. It brings great frustration to the person Re-ID tasks. For
example, in Fig. 1(a), when a target person is blocked by a
billboard or car, previous models are unable to distinguish
between the target person and the obstacle. They might
retrieve incorrect person images containing similar billboards
or cars [11]. Therefore, occluded person Re-ID [6], [8], [9],
[12], [13], [14], [15], [16] is challenging and practical. There
are three major challenges in occluded person Re-ID tasks: 1)
Occlusion often introduces a variety of noise, which interferes
with feature extraction and matching; 2) Occluded images
always contain less discriminative information, and it is more
difficult to extract robust features of pedestrians, resulting in
wrong retrieval; 3) Occlusion may cause a lack of appearance
information, which leads to insufficient discriminative features
and misalignment of local semantics, as shown in Fig. 1(b).
To address occlusion problems, some researchers [1], [6],
[16], [17], [18] have shown that fine-grained local features are
effective for dealing with occlusion problems. For example,
Miao et al. [17] proposed a Re-ID framework, which extracts
visible pedestrian parts by integrating pose information. Ma et
al. [18] proposed a novel method, named Pirt, to obtain robust
feature representations by constructing groups of regions and
masks. Recently, Yan et al. [16] proposed an innovative
lightweight network (PRE-Net), which constructs more robust
local features through a reasonable segmentation strategy.
However, most of these methods rely on additional pose
estimators to locate the visible regions, which consumes a
lot of computing resources. To overcome these limitations,
researchers have begun to explore methods that do not rely on
external tools. In this context, ViT [19] has attracted attention
due to its strong performance in processing image sequences.
ViT can capture global and local context information through
its self-attention mechanism, which is crucial for handling
occlusion problems. Therefore, to better extract fine-grained
person features while reducing the reliance on auxiliary pose

1051-8215 © 2024 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.

LI et al.: HGTDR NETWORK FOR OCCLUDED PERSON RE-IDENTIFICATION

Fig. 1. (a) The performance of the previous methods severely decreases when
the person image is occluded. (b) The challenges of occluded person Re-ID.
(c) The motivation of our HGTDR: 1) The key to correctly identify the target
pedestrian lies in the discriminative local details. 2) Compared with previous
hand-crafted strip-based methods, which may introduce noise in the red box,
our method can adaptively extract heterogeneous person local features with
different scales and shapes. 3) How to effectively recover occluded person
features from visible body parts of the pedestrian.

estimators, many researchers have shifted their focus to the
ViT and proposed many transformer-based models [8], [20],
[21], [22]. Their main idea is to aggregate similar semantic
information into a predefined part token, thereby achieving disentangling of person image. However, most of the
methods rely solely on complex semantics for decoupling,
which inevitably introduces noise interference and hinders
the extraction of discriminative local features. Furthermore,
even though they can extract fine-grained features, the models
might lose robustness when the discriminative appearance is
occluded.
In view of the above limitations, some researchers explored
another occluded Re-ID method [10], [23], [24], [25], [26],
which aims to recover the invisible regions of pedestrians.
Some of them [24], [26] used generative adversarial networks
to produce the holistic image by restoring the occluded parts.
With the emergence of transformer models, some recent
advances [23], [25] have yielded promising solutions. They
primarily concentrate on constructing the feature set of the
retrieval results and then recovering the features of the
occluded probe from the k-nearest neighbor features within
this set. Recently, Wang et al. [10] proposed a Feature
Completion Transformer (FCFormer) to combat occlusion and
complete invisible body features. However, it relies on a specific data augmentation strategy. In addition, none of them take
into account the issue of neighbor weight for occlusion recovery, which leads to less robust recovered features. Alongside

5023

the above research ideas, it is of great application significance
to design a multi-effect network that can effectively extract
discriminative local features and dynamically recover invisible
regions.
In this paper, we investigate how to improve the extraction
ability of local semantics, adaptively disentangle discriminative body parts, and dynamically recover features of occluded
regions. This is because the key to identify the target pedestrian is to extract discriminative local details (as shown in
Fig. 1(c).1)), noise-free human bodies are more discriminative than the local features generated by rigid cutting (as
shown in Fig. 1(c).2)), and recovered person features are
more robust than occluded features (as shown in Fig. 1(c).3)).
At the same time, in order to effectively deal with the challenges in Fig. 1(b), we propose a multi-effect Heterogeneous
Generative Token and Distance-aware Recovery (HGTDR)
network. As shown in Fig. 2, HGTDR is a dual-stream
architecture, which mainly consists of a holistic stream and
a part stream. The holistic stream can provide aggregated
long-range global information to the part stream. In the
part stream, we focus on extracting local semantics and
constructing discriminative local features without introducing
any additional part tokens and auxiliary networks. Firstly,
a Semantic Patch Generator (SPG) is proposed, which uses
the spatial pooling and local attention to capture rich local
semantics and further obtain discriminative semantic patches.
Its design enables the network to extract finer and richer features from local regions, which provides basic local semantic
units for subsequent feature processing. Subsequently, based
on the semantic patches generated by SPG, we propose an
Adaptive Heterogeneous Semantic Token Generator (AHSTG)
to obtain identity-related fine-frained features. Guided by the
global features of the holistic stream, AHSTG calculates
the comprehensive scores by assessing the discrimination
and relevance between semantic patches generated by SPG.
The scores are used to adaptively generate strong-response
and weak-response heterogeneous tokens, which effectively
highlights the target person’s body components and suppresses occlusions. However, when in complex occlusion
scenarios, the heterogeneous tokens may struggle to achieve
excellent retrieval performance due to the lack of discriminative appearance features. To address this issue, we design
a Distance-based Feature Recovery (DFR) module, which
recovers the features of occluded regions by dynamically
allocating adaptive weights to neighboring features. Finally,
we obtain robust person features to handle the complex person
Re-ID tasks.
In summary, the main contributions of this paper can be
summarized as follows:
(1) A new Semantic Patch Generator (SPG) is proposed
to capture the local semantics of the image. It can enhance
the local extraction ability of the network and further obtain
discriminative semantic patches.
(2) We propose a flexible Adaptive Heterogeneous Semantic Token Generator (AHSTG), which takes into account
both discrimination and relevance scores to select salient
semantic patches. It can further help our network adaptively
generate heterogeneous tokens with different responses to

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.

5024

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

Fig. 2. The pipeline of our proposed HGTDR. The framework mainly consists of a holistic stream (Section III-B) and a part stream (Section III-C). Our
proposed SPG, AHSTG, and DFR modules are in the part stream. Here, the patch embedding layer and shallow transformer blocks act as base module and
are not expanded in detail anymore. ‘⊕’ represents the concatenation operation. ‘CLS’ represents the class token. In the holistic stream, the sequence below
CLS is the updated patch tokens by patch embeddings Pe . The blue arrows represent the holistic stream, the green arrows represent the part stream, and the
gray arrows represent the interaction of the two streams.

suppress occlusion. More importantly, it can flexibly generate
fine-grained tokens with different scales and shapes.
(3) A novel Distance-based Feature Recovery (DFR) module is designed to automatically mine implicit information
to recover the occluded body parts. Different from previous
methods, we consider the contribution to occlusion recovery
from the perspective of distance, and can adaptively recover
the discriminative person features of the occluded regions.
(4) Extensive experiments on two authoritative occluded
Re-ID datasets demonstrate the effectiveness and superiority
of our method. In addition, we confirmed that our method has
good generalization ability on the holistic Re-ID datasets.
II. R ELATED W ORK
A. General Person Re-ID
The purpose of person Re-ID is to match the target person
across time and space from the images captured by a set
of non-overlapping distributed cameras. In the early stages
of research, some studies [2], [3], [27], [28] mainly utilize
holistic person images to match the target person, which
mainly focus on two aspects: feature learning [2], [3], [14],
[29], [30] and attention-based [12], [28], [31], [32], [33],
[34]. For example, Zhou et al. [35] proposed an effective
Foreground Attention Neural Network (FANN) to enhance the
attention on the foreground and learn discriminative feature
representation for person Re-ID. To address the issue of
viewpoint misalignment, Zhang et al. [3] proposed a View
Confusion Feature Learning (VCFL) method to learn the
view-invariant features by using a view confusion learning
mechanism. In the attention-based methods, Zhang et al. [28]
proposed a plug-and-play Relational-aware Global Attention
(RGA) module to capture global context information for better
focusing identity-related regions. Recently, transformer [36]

has rapidly dominated the computer vision field with its
powerful Multi-head Self-Attention (MSA) mechanism. Some
researchers worked on transformer-based person Re-ID and
achieved remarkable performance. Although these methods
achieve good performance when dealing with the holistic
person Re-ID problem, they greatly ignore the existence of
occlusions.
B. Fine-Grained Feature Matching
Compared with the general person Re-ID, the person Re-ID
methods using fine-grained features [1], [7], [37], [38], [39],
[40] can effectively deal with the occluded Re-ID problems.
They employed a part-to-part matching strategy to retrieve the
target person by extracting fine local features. For example,
Sun et al. [1] proposed a Part-based Convolutional Baseline
(PCB) method combined with RPP to obtain fine-grained local
features, which further improves the performance of person
Re-ID. Tan et al. [40] proposed a Continuous Batch DropBlock
Network (CBDB-Net), which can capture pedestrian robust
fine-grained descriptors for person Re-ID tasks. Although
the performance improvement is significant, these methods
become inefficient when facing occlusion or different scales.
To solve the above problems, He et al. [37] proposed a Deep
Spatial feature Reconstruction (DSR) method to avoid the error
in matching images at different scales. Sun et al. [7] proposed a
Visibility-aware Part Model (VPM) to automatically recognize
visible human regions under self-supervision. However, these
methods usually require the manual definition of the scale
and shape of fine-grained features, which limits the scalability
of the model. In contrast, our proposed method is general
and flexible. It not only can adaptively construct fine-grained
heterogeneous features, but also can be applied to diverse
situations in the real world.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3315-6701-9/26/$31.00 ©2026 IEEE | DOI: 10.1109/ICASSP55912.2026.11464579

GSTNET: A GEOSPATIAL-TEMPORAL GRAPH NETWORK FOR GROUP PERSON
RE-IDENTIFICATION
Ping Hu1,2†

Jingyi Li1,2

Fating Hong3

Yangyuqi Peng1,2

Junhang Wu1,5

Ruimin Hu4,5

1

2

School of Computer Science and Technology, Xinjiang University, Xinjiang, China
Joint International Research Laboratory of Silk Road Multilingual Cognitive Computing, Xinjiang, China
3
CSE, The Hong Kong University of Science and Technology, Hong Kong, SAR
4
School of Cyber Science and Engineering, Wuhan University, Wuhan, China
5
Hubei Provincial Key Laboratory of Multimedia and Network Communication Engineering,
Wuhan University, Wuhan, China
ABSTRACT

Index Terms— Geospatial-temporal graph network, group person re-identification, reachability constraint
1. INTRODUCTION
Geospatial-temporal Group Person Re-identification (Gst-GReID)
can integrate geospatial constraint and spatio-temporal contextual
information. By modeling the appearance of groups across regions
and cameras, it enables accurate identity matching and retrieval in
complex scenarios. The existing methods for person re-identification
can be divided into two categories. The first category is Group
Re-identification (GReID). Early GReID studies focused mainly on
group-level feature extraction and matching, such as unsupervised
sparse transfer coding for cross-camera group matching [1]. The
DukeTMC Group and Road Group datasets advanced group person
re-identification by providing benchmark datasets [2]. To address
variations in group size and layout, some studies employed graph
modeling and feature aggregation to capture complex intra-group relationships [3, 4, 5]. Meanwhile, some works leveraged Transformers [6] to explore the advanced spatial reasoning of group layouts
and cross-modal designs [7, 8, 9, 10, 11, 12]. The second category
† Corresponding author Ping Hu: royalcat1982@gmail.com.

979-8-3315-6701-9/26/$31.00 ©2026 IEEE

Group B

Group B

Geospatial-temporal Group Person Re-identification (Gst-GReID)
can integrate geospatial constraint and spatio-temporal contextual information. By modeling the appearance of groups across
regions and cameras, it enables accurate identity matching and
retrieval in complex scenarios. The existing methods primarily
rely on static distribution of statistics across time-interval cameras and ignore geospatial-temporal reachability, resulting in poor
generalization over long time spans and across regions. Here, we
propose the Geospatial-temporal Graph Network (GstNet). GstNet has two core designs: the Geospatial-temporal Reachability
Module (GstRM) and the Gated Graph-MLP (GGM). GstRM imposes a geospatial-temporal reachability constraint on graph edges,
suppressing geospatial-temporal unreachable connections, thereby
mitigating static time-prior mismatch. GGM introduces channel
gating to achieve selective neighborhood aggregation over long time
spans and across regions, enhancing identity-discriminative representations and suppressing interfering samples. Extensive experimental results on the BRT and SYSU-Group datasets indicate that
our GstNet method outperforms existing state-of-the-art methods.

Group A

Unreachable (V > Vmax)

Previous Method

Group B

Reachable (V < Vmax)

GstNet

Fig. 1. Illustration of the proposed motivation. Unlike previous methods that overlook geospatial-temporal reachability constraint (e.g., traversing 20 km in 5 s is infeasible under the maximum traffic speed), we introduce such constraint to filter out groups
that are unreachable but have similar appearances.

is spatio-temporal modeling-based person re-identification. Some
studies have introduced static distribution of statistics across timeinterval cameras [13, 14]. There are also some works that adopted
3D CNN and attention mechanisms to characterize cross-frame temporal sequences and achieve multi-scale temporal modeling and spatiotemporal fusion [15, 16, 17, 18].
The accuracy of existing methods is limited by two major factors. First, most group person re-identification approaches remain
confined to group-level modeling of visual appearance features.
Over long time spans and across regions, such methods struggle to
leverage the geospatial-temporal reachability constraint. Second,
although spatio-temporal modeling methods introduce temporal
factors, they generally rely on a static distribution of statistics
across time-interval cameras. However, static modeling ignores
the geospatial-temporal reachability constraint, which is defined
as the speed of a group derived from distance-time relationships
must not exceed the maximum allowable speed. This often leads
to erroneous matches and degraded discriminative performance. As
illustrated in Fig. 1, previous methods overlook the reachability
constraint, namely the maximum feasible travel range of a group
within a given time interval. If a group traverses a long distance in
an extremely short time, resulting in a speed far beyond the normal
upper limit of traffic [19], the path should be deemed unreachable,
and the groups should be identified as the different. Conversely, if
movement occurs within a reasonable time and the speed remains
below the range of traffic, it should be considered reachable, and the
groups should be identified as the same.

9952

ICASSP 2026

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:22 UTC from IEEE Xplore. Restrictions apply.

Group Images with labels

GReID Graph Building

Geospatial-Temporal Feature Extraction

Distance Features

Feature
Fusion Matrix

...

GGM Layers

GstRM

GMNLayers
Layers
GMN

Temporal Features

Visual Feature Extraction

CE Loss
Ncontrast
Loss

Vision
Transformer

Visual Features

(a) Overview of Geospatial-temporal Graph Network (GstNet)
Clamped Speeds

Graph-MLP

Input

embedding layer

RELU

Output

Linear

Softmax
Linear

RELU

Linear

After Gate

Output
: Speed excess
: Penalty coefficient
: Gate span
: Speed limit

Conv1d
Linear

Linear

Norm

: Lower bound

Reachability Margin

Input

(b) Geospatial-temporal Reachability Module(GstRM)

(c) Gated Graph-MLP(GGM)

Fig. 2. Overall architecture of GstNet (a). Step 1: group images are encoded by a ViT to extract visual features, and distance/time
information is processed by GstRM (b) to produce a feature fusion matrix constrained by geospatial-temporal reachability. Step 2: a grouplevel graph is constructed by combining visual features with the feature fusion matrix derived from reachability modeling. Step 3: the graph
is propagated through GGM(c) for selective aggregation and optimized with joint cross-entropy loss and neighborhood contrastive loss.

To address this issue, we propose Geospatial-temporal Graph
Network (GstNet) for group person re-identification. This framework integrates geospatial-temporal information with group appearance features to enhance matching performance over long
time spans and across regions. GstNet comprises two core modules: Geospatial-temporal Reachability Module (GstRM), which
imposes geospatial-temporal reachability constraint during graph
construction by penalizing unreasonable connections; and Gated
Graph-MLP (GGM), which achieves selective neighborhood aggregation during feature propagation to suppress the mismatches
caused by the absence of reachability constraint. The contributions
of this paper are summarized as follows:
(1) To overcome mismatches over long time spans and across
regions, we propose GstNet, which integrates geospatial-temporal
reachability with group appearance features and reduces errors
through constrained graph construction and gated propagation.
(2) To address unreasonable connections during constrained
graph construction, we propose GstRM, which defines reachability
based on geospatial-temporal information and speed thresholds, and
penalizes edges that violate constraints to reduce incorrect matching.
(3) To reduce the interference of erroneous edges during the
propagation stage, GGM employs channel-wise gating and residual guidance to selectively propagate neighborhood information,
thereby reducing noise accumulation.

2. METHODOLOGY
2.1. Overall Framework
As illustrated in Fig. 2, the overall pipeline is as follows: first, group
images are fed to the visual branch to extract discriminative appearance features via a Vision Transformer (ViT) [20], while distance
and time attributes are extracted from the source data and sent to
GstRM to produce a reachability-constrained edge matrix. Next, a
graph is constructed with groups as nodes, and reachability penalties
are imposed on edges to effectively filter unreasonable connections.
Then, the graph is processed by GGM for gated feature propagation,
allowing for adaptive and selective aggregation of neighborhood information. Finally, the fused features are jointly optimized using
cross-entropy loss and the Ncontrast loss.
2.2. Geospatial-temporal Reachability Module (GstRM)
To address unreasonable edge connections caused by relying solely
on appearance similarity or static distribution of statistics across
time-interval cameras. For example, two groups that look similar
but are separated by tens of kilometers with only a few seconds
between timestamps, which is physically implausible. We propose the Geospatial-temporal Reachability Module (GstRM). This

9953
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:22 UTC from IEEE Xplore. Restrictions apply.

module introduces a speed constraint with a slack margin to dynamically modulate edge strengths, thereby suppressing geospatialtemporal infeasible links. We define the time difference and the
geographic distance. For any two group nodes (i, j), the time difference Tij = |ti − tj |, and the geographical distance Dij = D[i][j].
Based on the geographical distance Dij and the time difference Tij ,
we calculate the speed to measure the rationality of the edge in the
geospatial-temporal dimension:
vij = V(Dij , Tij ) ≜

Dij
Tij

(1)

When vij exceeds the preset physical speed limit vmax , the connection is deemed unreachable and is penalized in subsequent steps.
To further characterize this constraint more rigorously, we define a slack margin function that formally quantifies the reachability
margin under a given time interval:
Mij = M(Dij , Tij ; vmax ) ≜ vmax Tij − Dij

(2)

When Mij is positive, the group pair is considered reachable.
After obtaining the geospatial-temporal priors, we model
the matching similarity on edges. We feed (Dij , Tij ) into an
MLP to learn the baseline pairwise compatibility potential ϕij =
MLP(Dij , Tij ), where ϕij denotes the geospatial-temporal similarity potential. To modulate the baseline pairwise compatibility
potential with the slack margin, we design a gating factor:


Mij
γij = gmin + (gmax − gmin ) · σ
(3)
β
Where gmin and gmax denote the lower and upper bounds of the gating, β is a smoothing coefficient, and σ(·) is the sigmoid function.
When the slack margin is large, γij approaches the upper bound,
indicating higher confidence for this edge; when the slack margin
approaches zero, γij approaches the lower bound, thereby suppressing the weight of this edge.
For edges whose implied speed vij exceeds the upper bound
vmax , we further introduce a penalty term:

bij = −κ · ReLU vij − vmax
(4)
Here, κ denotes the penalty-strength hyperparameter that regulates
the magnitude of constraint violations. When the speed constraint
is violated, the penalty term suppresses the corresponding edge
strength , leading to a substantial attenuation of connectivity.
By jointly considering the baseline pairwise compatibility potential, the modulation factor, and the penalty term, the final edge
strength can be expressed as:

exp γij ϕij + bij

Eij = P
(5)
k∈N (i) exp γik ϕik + bik
The graph model G(F, E) is defined by a node feature matrix
F = [f1 , . . . , fN ] ∈ RN ×C and the edge strength matrix E =
[Eij ] ∈ RN ×N . Here, fi denotes the visual feature of image xi
extracted by the Vision Transformer (ViT), while Eij specifies the
connection strength between nodes i and j as formulated in Eq. 5.
2.3. Gated Graph-MLP (GGM)
In geospatial-temporal group re-identification (Gst-GReID), groups
that are adjacent in geographic location and time are not necessarily of the same identity; indiscriminate propagation of neighborhood

information can induce cross-identity interference. Meanwhile, irrelevant groups in the scene may be erroneously amplified during
propagation, degrading discriminability. To address this, we propose
Gated Graph-MLP (GGM), which augments Graph-MLP [21] with
a gating mechanism to enable selective propagation of neighborhood
information. Specifically, the input node features first pass through
a gated convolution to perform channel selection:

H = Conv1d(F) ⊙ σ(FW) + F
(6)
Where ⊙ denotes element-wise multiplication, F denotes the node
features, and H = [h1 ; . . . ; hN ], hi ∈ RC denotes the gated rep′
resentation of node i, and W ∈ RC×C is the weight matrix of a
linear transformation.
Subsequently, node representations are updated by incorporating
information from their neighbors:
X
h̃i = hi +
αij · hj
(7)
j

Here, αij is obtained from the edge strengths Eij in Eq. 5 by neighborhood normalization, with αij = softmaxj∈N (i) (Eij ). The
residual connection preserves the node’s own representation while
injecting more discriminative neighborhood information.
2.4. Loss Function
We jointly optimize the model by minimizing the node-level classification cross entropy loss and a ncontrast loss. To enhance the ability
of node embeddings to capture graph structure, we introduce a structural contrastive loss (ncontrast Loss). This loss exploits the structural information encoded in the edge feature tensor E ∈ RN ×N ×de .
For each node i, the positive set is defined as all nodes j such that
∥Eij ∥2 ≥ ϵ, which denotes the threshold for defining positive pairs.
The structural contrastive loss for node i is then defined as:

P
N
X
j∈Pi exp sim(h̃i , h̃j )/τ
log P
LNC = −
(8)
,
k̸=i exp sim(h̃i , h̃k )/τ
i=1
where h̃i denotes the final embedding of node i after gated propagation, τ is the temperature hyperparameter and ϵ is the threshold for
selecting positive samples.
Finally, the total loss can be written as follows:
Ltotal = LCE + λ · LNC

(9)

where λ is a balancing coefficient that controls the trade-off between
classification and structural contrastive objectives.
3. EXPERIMENTS
3.1. Experimental Setup
Datasets. We evaluate our approach on two group person reidentification datasets: BRT [22] and SYSU-Group [23]. The BRT
dataset was collected from the BRT lines of an urban public transit
system, covering 53 sampling stations, and contains 200 groups
with 3,210 group images. For the SYSU-Group dataset, we use
three indoor visible-light cameras (Cam1-Cam3), comprising 391
groups and 14,020 group images. Since only timestamp information
is available without inter-camera distance, we utilize temporal cues
alone for reachability modeling.
Implementation Details. We evaluate the effectiveness of the proposed approach using widely adopted retrieval metrics, including

9954
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:22 UTC from IEEE Xplore. Restrictions apply.


codex
只基于每篇前三页判断，重点放在能反推出什么可发的新问题和机制。

**逐篇提炼**

1. `FMCNet+`
创新类型：新机制，带一点问题重定义。  
为什么能发：它抓住了可见光和红外 ReID 里“图像级补全会引入噪声”这个缺口，把补偿从图像层改到特征层。证据链是特征分解、跨模态特征生成、共享特异融合，以及和图像生成式补偿方法对比。  
story 套路：不要生成另一种模态的假图像，而要生成对 ReID 有用的缺失模态特征。

2. `Focusing on pedestrians like human`
创新类型：新数据增强机制，偏工程组合，但 story 比较会讲。  
为什么能发：换衣 ReID 大多堆网络结构，它把缺口放在“训练图像本身没有强化局部细节”。用人体轮廓切五个区域，再做中心放大和连续性修复。证据链是多换衣数据集、消融、迁移到蒸馏和无监督。  
story 套路：像人一样看局部细节，用图像级增强让网络更容易学到衣服之外的身份线索。

3. `GAE-Net`
创新类型：新机制，特权模态蒸馏。  
为什么能发：视频 ReID 依赖外观，光照和衣服变化会伤害性能；步态稳健但和 RGB 有模态差距。它先用外观加步态双流学强教师，再把局部互补知识蒸馏给只用 RGB 的学生。  
story 套路：训练时借步态，测试时不增加输入，用稳健生物线索增强普通 ReID 表征。

4. `GAReID`
创新类型：新机制。  
为什么能发：部位错位通常靠固定切块或姿态检测，它提出高阶相似性会放大对齐部位、压低错位部位，并用分组 Kronecker 压缩解决维度爆炸。证据链有理论解释、模块消融和多个标准数据集。  
story 套路：不显式对齐人体部位，也能通过高阶相似性实现“隐式对齐”。

5. `Generalizable person re-identification`
创新类型：新训练机制。  
为什么能发：它把度量学习和表征学习的关系当作缺口。采样器不再随机取类，而是构建近邻类别图提供困难样本；表征网络又反过来提高采样质量。  
story 套路：让采样器和特征网络互相提升，训练批次本身成为可学习的泛化机制。

6. `Global aggregated gradient-guided adversarial attacks`
创新类型：新机制，安全评测方向。  
为什么能发：黑盒攻击 ReID 的迁移性差，它把原因归到攻击初期梯度一致性低。通过预收敛得到全局聚合梯度，再约束当前梯度更新。  
story 套路：ReID 攻击失败不是扰动不够强，而是优化方向太摇摆。

7. `GSTNET`
创新类型：问题重定义加新机制。  
为什么能发：群体 ReID 不能只看外观和静态时间先验，还要满足地理和时间上的物理可达性。它用速度上限约束图边，再用门控图传播抑制错误邻居。  
story 套路：视觉相似必须服从物理可达性，不可能出现的匹配先从图里删掉。

8. `Harnessing Knowledge From Pretrained VLMs`
创新类型：新设定加新机制。  
为什么能发：提出完全无监督 person search，不是无监督检测和无监督 ReID 的简单相加，因为伪框噪声会破坏伪身份。它用 VLM 生成语义伪框和伪 ID，再用抗框噪声损失稳住 ReID。  
story 套路：用基础模型把无标签监控场景转成可训练的 person search 数据，但核心在于处理伪框噪声。

9. `Heterogeneous Generative Tokens`
创新类型：新机制。  
为什么能发：遮挡 ReID 的固定条带和语义聚类容易引入背景噪声，已有恢复方法又没有认真处理邻居权重。它自适应生成不同尺度和形状的语义 token，再按距离恢复被遮挡部位。  
story 套路：遮挡时不是硬切人体，而是先找可见的判别语义，再按可靠距离补回缺失特征。

**给团队资产的强候选**

1. 一句话 headline：跨视角 ReID 不是只看视觉相似，而是要满足 3D 人体可见性一致性。  
挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
和本批最像工作的区别：最像 `GSTNET` 和 `HGTDR`。但 `GSTNET` 用地理时间可达性约束群体图，我们用 SMPL 表面、关节投影和视角可见性约束航拍和地面之间哪些身体区域应该能对应。`HGTDR` 是遮挡恢复，我们不是补遮挡，而是定义跨视角下“可比区域”和“不可比区域”。  
cheap kill-switch：不训练模型，先用现有 SMPL 或 2D pose 估计在 CARGO 或 AG-ReID.v2 上算可见性兼容分数。若同身份跨视角对的可见性兼容分数不能明显高于困难负样本，或者把该分数加到 SOLIDER 距离后 mAP 没有超过 0.4 的净增益，就先杀掉。

2. 一句话 headline：不要生成地面图或航拍图，而是在特征层补偿跨视角缺失的身份残差。  
挂靠资产：SOLIDER-Swin、aerial-ground、SMPL。  
和本批最像工作的区别：最像 `FMCNet+`。它补偿可见光和红外的模态特异特征，我们补偿航拍和地面视角下缺失的身体表面和局部身份残差。关键切开点是用 SMPL 可见性和姿态热图约束生成什么，不做无约束特征幻觉。  
cheap kill-switch：冻结 SOLIDER，只训一个小 MLP 预测跨视角残差。看同身份跨视角距离是否下降，同时异身份近邻是否不被拉近。若正负样本残差分布分不开，或者只是在所有样本上做平滑，直接杀掉。

3. 一句话 headline：用 3D 姿态和步态做训练期教师，蒸馏出测试时只需 RGB 的跨视角身份表征。  
挂靠资产：SMPL 3D 几何、pose 热图门控、SOLIDER-Swin，若数据有 tracklet 再挂 aerial-ground 视频。  
和本批最像工作的区别：最像 `GAE-Net`。它用 2D 步态轮廓辅助普通视频 ReID，我们用 SMPL 关节、体型和投影一致性作为跨视角教师，目标是解决航拍和地面视角差，不是普通 MARS/LS-VID 的外观鲁棒性。  
cheap kill-switch：先做零训练或浅层探针。单独用 SMPL 关节比例、投影姿态、步态统计做检索，如果它对同身份没有独立信号，或者和 SOLIDER late fusion 不能带来超过噪声线的增益，就不要开大训练。

4. 一句话 headline：训练批次不随机采样，而是按“跨视角物理可混淆”构造困难样本图。  
挂靠资产：aerial-ground、SMPL、SOLIDER-Swin。  
和本批最像工作的区别：最像 `Generalizable person re-identification`。它用特征近邻图做通用困难采样，我们把近邻定义成视觉相似、视角可达、3D 可见区域重叠的混合图，专门制造航拍和地面最容易错的训练批次。  
cheap kill-switch：不改网络，只替换 sampler 跑小规模训练。若同样 epoch 下没有比 PK sampler 更快拉开跨视角正负距离，或者增益只出现在训练集不出现在验证集，就判为普通采样工程，不作为主线。

我会优先押第 1 个和第 2 个。它们的问题层面更清楚，也最能把团队的 aerial-ground 和 SMPL 基建变成别人不容易复刻的机制。第 3 个有潜力，但依赖视频或稳定 SMPL 质量。第 4 个成本最低，适合作为辅助机制，不适合单独当 B 类主贡献。
tokens used
70,695
只基于每篇前三页判断，重点放在能反推出什么可发的新问题和机制。

**逐篇提炼**

1. `FMCNet+`
创新类型：新机制，带一点问题重定义。  
为什么能发：它抓住了可见光和红外 ReID 里“图像级补全会引入噪声”这个缺口，把补偿从图像层改到特征层。证据链是特征分解、跨模态特征生成、共享特异融合，以及和图像生成式补偿方法对比。  
story 套路：不要生成另一种模态的假图像，而要生成对 ReID 有用的缺失模态特征。

2. `Focusing on pedestrians like human`
创新类型：新数据增强机制，偏工程组合，但 story 比较会讲。  
为什么能发：换衣 ReID 大多堆网络结构，它把缺口放在“训练图像本身没有强化局部细节”。用人体轮廓切五个区域，再做中心放大和连续性修复。证据链是多换衣数据集、消融、迁移到蒸馏和无监督。  
story 套路：像人一样看局部细节，用图像级增强让网络更容易学到衣服之外的身份线索。

3. `GAE-Net`
创新类型：新机制，特权模态蒸馏。  
为什么能发：视频 ReID 依赖外观，光照和衣服变化会伤害性能；步态稳健但和 RGB 有模态差距。它先用外观加步态双流学强教师，再把局部互补知识蒸馏给只用 RGB 的学生。  
story 套路：训练时借步态，测试时不增加输入，用稳健生物线索增强普通 ReID 表征。

4. `GAReID`
创新类型：新机制。  
为什么能发：部位错位通常靠固定切块或姿态检测，它提出高阶相似性会放大对齐部位、压低错位部位，并用分组 Kronecker 压缩解决维度爆炸。证据链有理论解释、模块消融和多个标准数据集。  
story 套路：不显式对齐人体部位，也能通过高阶相似性实现“隐式对齐”。

5. `Generalizable person re-identification`
创新类型：新训练机制。  
为什么能发：它把度量学习和表征学习的关系当作缺口。采样器不再随机取类，而是构建近邻类别图提供困难样本；表征网络又反过来提高采样质量。  
story 套路：让采样器和特征网络互相提升，训练批次本身成为可学习的泛化机制。

6. `Global aggregated gradient-guided adversarial attacks`
创新类型：新机制，安全评测方向。  
为什么能发：黑盒攻击 ReID 的迁移性差，它把原因归到攻击初期梯度一致性低。通过预收敛得到全局聚合梯度，再约束当前梯度更新。  
story 套路：ReID 攻击失败不是扰动不够强，而是优化方向太摇摆。

7. `GSTNET`
创新类型：问题重定义加新机制。  
为什么能发：群体 ReID 不能只看外观和静态时间先验，还要满足地理和时间上的物理可达性。它用速度上限约束图边，再用门控图传播抑制错误邻居。  
story 套路：视觉相似必须服从物理可达性，不可能出现的匹配先从图里删掉。

8. `Harnessing Knowledge From Pretrained VLMs`
创新类型：新设定加新机制。  
为什么能发：提出完全无监督 person search，不是无监督检测和无监督 ReID 的简单相加，因为伪框噪声会破坏伪身份。它用 VLM 生成语义伪框和伪 ID，再用抗框噪声损失稳住 ReID。  
story 套路：用基础模型把无标签监控场景转成可训练的 person search 数据，但核心在于处理伪框噪声。

9. `Heterogeneous Generative Tokens`
创新类型：新机制。  
为什么能发：遮挡 ReID 的固定条带和语义聚类容易引入背景噪声，已有恢复方法又没有认真处理邻居权重。它自适应生成不同尺度和形状的语义 token，再按距离恢复被遮挡部位。  
story 套路：遮挡时不是硬切人体，而是先找可见的判别语义，再按可靠距离补回缺失特征。

**给团队资产的强候选**

1. 一句话 headline：跨视角 ReID 不是只看视觉相似，而是要满足 3D 人体可见性一致性。  
挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
和本批最像工作的区别：最像 `GSTNET` 和 `HGTDR`。但 `GSTNET` 用地理时间可达性约束群体图，我们用 SMPL 表面、关节投影和视角可见性约束航拍和地面之间哪些身体区域应该能对应。`HGTDR` 是遮挡恢复，我们不是补遮挡，而是定义跨视角下“可比区域”和“不可比区域”。  
cheap kill-switch：不训练模型，先用现有 SMPL 或 2D pose 估计在 CARGO 或 AG-ReID.v2 上算可见性兼容分数。若同身份跨视角对的可见性兼容分数不能明显高于困难负样本，或者把该分数加到 SOLIDER 距离后 mAP 没有超过 0.4 的净增益，就先杀掉。

2. 一句话 headline：不要生成地面图或航拍图，而是在特征层补偿跨视角缺失的身份残差。  
挂靠资产：SOLIDER-Swin、aerial-ground、SMPL。  
和本批最像工作的区别：最像 `FMCNet+`。它补偿可见光和红外的模态特异特征，我们补偿航拍和地面视角下缺失的身体表面和局部身份残差。关键切开点是用 SMPL 可见性和姿态热图约束生成什么，不做无约束特征幻觉。  
cheap kill-switch：冻结 SOLIDER，只训一个小 MLP 预测跨视角残差。看同身份跨视角距离是否下降，同时异身份近邻是否不被拉近。若正负样本残差分布分不开，或者只是在所有样本上做平滑，直接杀掉。

3. 一句话 headline：用 3D 姿态和步态做训练期教师，蒸馏出测试时只需 RGB 的跨视角身份表征。  
挂靠资产：SMPL 3D 几何、pose 热图门控、SOLIDER-Swin，若数据有 tracklet 再挂 aerial-ground 视频。  
和本批最像工作的区别：最像 `GAE-Net`。它用 2D 步态轮廓辅助普通视频 ReID，我们用 SMPL 关节、体型和投影一致性作为跨视角教师，目标是解决航拍和地面视角差，不是普通 MARS/LS-VID 的外观鲁棒性。  
cheap kill-switch：先做零训练或浅层探针。单独用 SMPL 关节比例、投影姿态、步态统计做检索，如果它对同身份没有独立信号，或者和 SOLIDER late fusion 不能带来超过噪声线的增益，就不要开大训练。

4. 一句话 headline：训练批次不随机采样，而是按“跨视角物理可混淆”构造困难样本图。  
挂靠资产：aerial-ground、SMPL、SOLIDER-Swin。  
和本批最像工作的区别：最像 `Generalizable person re-identification`。它用特征近邻图做通用困难采样，我们把近邻定义成视觉相似、视角可达、3D 可见区域重叠的混合图，专门制造航拍和地面最容易错的训练批次。  
cheap kill-switch：不改网络，只替换 sampler 跑小规模训练。若同样 epoch 下没有比 PK sampler 更快拉开跨视角正负距离，或者增益只出现在训练集不出现在验证集，就判为普通采样工程，不作为主线。

我会优先押第 1 个和第 2 个。它们的问题层面更清楚，也最能把团队的 aerial-ground 和 SMPL 基建变成别人不容易复刻的机制。第 3 个有潜力，但依赖视频或稳定 SMPL 质量。第 4 个成本最低，适合作为辅助机制，不适合单独当 B 类主贡献。
