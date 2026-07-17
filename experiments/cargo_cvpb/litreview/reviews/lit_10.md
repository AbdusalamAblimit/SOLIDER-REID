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
session id: 019ef71f-8f78-7402-9d70-289b178da60f
--------
user
你是 ReID 论文创新挖掘员 10/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(10/20):
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则确认目标和状态文件；如果当前沙箱只允许读论文目录，我会说明受限，然后直接读取这批 PDF 的前三页。
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
/bin/zsh -lc 'pwd && ls' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
目标文件和项目状态已经读到，当前任务就是文献差距分析，不开实验。我会分批读取每篇前三页，只抓摘要、引言里的问题定义、机制和证据组织方式。
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
/bin/zsh -lc "pdftotext -l 3 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 201 (2026) 108883

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Hierarchical fusion and local-aware transformer for occluded person
re-identiﬁcation
Haishun Du a,b,∗, Chuaner Huang c , Linbing Cao a , Jieru Li

a , Wenzhe Zhang a

a School of Artiﬁcial Intelligence, Henan University, Zhengzhou, 450046, China
b

International Joint Research Laboratory for Cooperative Vehicular Networks of Henan, Zhengzhou, 450046, China

c Patent Examination Cooperation (Henan) Center of the Patent Oﬃce, China National Intellectual Property Administration, Zhengzhou, 450046, China

a r t i c l e

i n f o

Keywords:
Occluded person re-identiﬁcation
Transformer
Feature fusion
Local-aware

a b s t r a c t
Occluded person re-identiﬁcation (ReID) is intended to address the problem of matching pedestrians when images of individuals are partially occluded. Recently, Transformer-based methods for occluded person ReID have
received considerable attention. However, although existing methods have achieved promising results, most of
them do not fully consider the varying contributions of diﬀerent patches to identity recognition, nor do they
suﬃciently emphasize the identity information in critical regions. Furthermore, those methods often lack suﬃcient capability to extract ﬁne-grained local features, making it diﬃcult to fully explore the identity information
embedded in various body parts. To resolve the mentioned problems, we propose a Hierarchical Fusion and
Local-aware Transformer (HFLAT) for occluded person ReID. Speciﬁcally, we ﬁrstly design a feature hierarchical
fusion module that hierarchizes and fuses the patch feature vector sequence according to the relative importance
of each patch to the global feature vector, thereby reinforcing the identity discriminative features of key regions.
We then design a feature separation module to distinguish foreground features from background features by employing patch-level saliency analysis, thereby mitigating the negative impact of backgrounds and occlusions on
the performance of the model. In addition, we design a local feature extraction module which restricts the range
of interactions between the features using a local-aware multi-head attention mechanism, increasing the model’s
ability to obtain ﬁne-grained local features. Experimental results on the Occluded-DukeMTMC, Occluded-ReID,
Market1501, and DukeMTMC-ReID datasets demonstrate that HFLAT reaches the current state-of-the-art performance for occluded person ReID. Speciﬁcally, on the Occluded-DukeMTMC and Occluded-ReID datasets, our
method achieves the Rank-1 accuracy of 79.6% and 89.8%, respectively, and the mAP of 64.7% and 84.9%,
respectively. On the Market1501 and DukeMTMC-ReID datasets, our method achieves the Rank-1 accuracy of
95.9% and 90.6%, respectively, and the mAP of 90.8% and 82.2%, respectively.

1. Introduction
Person re-identiﬁcation (ReID) tasks intend to identify and match
target individuals among multiple non-overlapping camera viewpoints.
Owing to the rapid growth of deep learning techniques, research about
person ReID (Luo et al., 2019; Qian et al., 2017; Zhou et al., 2019) has
made signiﬁcant progress. However, in practical situations, as illustrated
in Fig. 1, pedestrians often exhibit missing body parts due to occlusions
caused by objects such as trees and vehicles. Therefore, how to enhance
the performance of person ReID models when pedestrians are partially
occluded has become one of the core challenges.
In recent times, occluded person ReID has garnered heightened interest, and its key is extracting features from partially occluded images
to mitigate the impact of occlusions. To this end, various occluded per-

son ReID methods have been presented. Existing methods can be broadly
categorized into external auxiliary information-based methods Gao et al.
(2020), Huang et al. (2020), Miao et al. (2019) and Transformer-based
methods Li et al. (2021), Lin et al. (2024), Wang et al. (2022b). The external auxiliary information-based methods typically reduce the eﬀect of
occlusions on the discriminative capability of pedestrian features by locating un-occluded body regions with the help of auxiliary information,
such as human posture or body key points. The Transformer-based methods primarily leverage the global modelling strengths of Transformers,
integrating various attention mechanisms to extract more discriminative features. Although these methods have achieved some encouraging
results, most of them process diﬀerent image patches indiscriminately.
Not only they fail to quantify the diﬀerent contributions of diﬀerent image patches to identity recognition, but they also neglect to focus on

∗ Corresponding author.

E-mail address: jddhs@vip.henu.edu.cn (H. Du).
https://doi.org/10.1016/j.neunet.2026.108883
Received 8 July 2025; Received in revised form 9 March 2026; Accepted 19 March 2026
Available online 5 April 2026
0893-6080/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 201 (2026) 108883

H. Du et al.

Fig. 1. Examples of pedestrians being occluded.

•

key regions. Moreover, most of them fail to eﬀectively isolate invalid
or detrimental information contained in pedestrian images, thereby being unable to mitigate the adverse eﬀects of backgrounds and occlusions on the performance of their models. In particular, although most
of them possess a strong ability to capture global contextual information, their capacity for acquiring ﬁne-grained local features is largely
inadequate, preventing them from fully extracting the identity information contained in diﬀerent body parts of pedestrians.
To resolve the mentioned problems, we propose a hierarchical fusion and local-aware Transformer (HFLAT) for occluded person ReID.
First, we design a feature hierarchical fusion module (FHFM) to hierarchize and fuse the patch feature vector sequence based on their relative
importance to the global feature vector, enhancing the identity discriminative features of key regions. Subsequently, we design a feature separation module (FSM) to isolate foreground and background features using patch-level saliency analysis, thereby mitigating the negative eﬀects
of backgrounds and occlusions on the model’s performance. Additionally, we design a local feature extraction module (LFEM) that employs
a local-aware multi-attention mechanism to limit feature interactions,
improving the model’s ﬁne-grained local feature extraction capability.
The main contributions of this research work presented in this paper are
as follows:

We design a local feature extraction module (LFEM) to enhance the
ﬁne-grained local feature extraction capability of the model by limiting the range of interactions between the features using a local-aware
multi-head attention mechanism.
• We perform extensive experiments on four datasets to validate the
eﬃciency and advancement of the model.
2. Related works
As deep learning techniques progress rapidly, person ReID has
achieved remarkable results Luo et al. (2019), Qian et al. (2017), Zhou
et al. (2019). However, in practical scenarios, pedestrians in images are
often partially occluded by objects such as trees or vehicles, leading to
the loss of key body parts. To address this issue, scholars have presented
a series of methods for occluded person ReID, which can be grouped into
external auxiliary information-based methods Gao et al. (2020), Huang
et al. (2020), Miao et al. (2019) and Transformer-based methods Li et al.
(2021), Lin et al. (2024), Wang et al. (2022b).
The external auxiliary information-based methods primarily mitigate
the impact of occlusions on their models by utilizing auxiliary information such as human posture or key points to locate visible body parts.
For instance, Miao et al. (2019) proposed a pose-guided feature alignment (PGFA) method to align local features using human key-point information. Gao et al. (2020) proposed a pose-guided visible part matching (PVPM) method that employs a pose-guided attention mechanism
and a local visibility predictor to extract features from un-occluded
areas accurately. Wang et al. (2022a) proposed a key-point-aware occlusion suppression and semantic alignment (POS) method that aligns
features using human key points and employs a feature enhancement
strategy to compensate for missing feature regions. Cui et al. (2025)
proposed a pose-guided partial-attention network with batch information (PPBI), which eﬀectively reduces the negative impact of occlusions
on model performance by capturing the semantic relationships of key

•

We propose a hierarchical fusion and local-aware Transformer
(HFLAT) for occluded person ReID.
• We design a feature hierarchical fusion module (FHFM) that hierarchizes and fuses the patch feature vectors based on their relative
importance to the global feature vector, aiming to enhance the identity discriminative features of key regions.
• We design a feature separation module (FSM) that separates foreground and background features using patch-level saliency analysis
to mitigate the negative eﬀects of backgrounds and occlusions on the
model’s performance.
2

Neural Networks 201 (2026) 108883

H. Du et al.

points among diﬀerent samples within each batch. Huang et al. (2020)
proposed a human parsing based alignment with multi-task learning
(HPNet) model that enhances the features in the visible regions using body part masks generated by a human parsing task. Somers et al.
(2023) proposed a body part-based (BPreID) model that extracts local
features using a local attention mechanism guided by human parsed labels. Dou et al. (2024) proposed a decouple re-identiﬁcation and human
parsing (DROP) method, which decouples the features for person ReID
and human parsing tasks, mitigating the feature granularity conﬂict between the two tasks and improving their collaborative eﬀectiveness. Although the above methods based on external auxiliary information have
achieved some promising results, their performance may be severely degraded when human body poses or key points are inaccurately estimated
because they rely too much on the accuracy of the external auxiliary information.
The Transformer-based methods mainly utilize the advantages of
Transformer in global modelling, combined with the attention mechanisms to obtain more discriminative features. For instance, Li et al.
(2021) proposed a part-aware Transformer (PAT) by ﬁrst applying the
Transformer framework to occluded person ReID tasks, which improves
the representation ability of local features by employing a pixel context encoder and a local prototype decoder. Lin et al. (2024) proposed a multi-level relation-aware Transformer (MLRAT), which has
stronger feature extraction ability by mining the feature relationships at
patch and sample levels. Wang et al. (2022b) proposed a Transformerbased pose-guided feature disentangling (PFD) method, which eﬀectively mitigates the negative impact of occlusions by disentangling
features utilizing pose information and aligning them to un-occluded
regions. Yang et al. (2023) proposed a robust feature mining Transformer (RFMT) method, which combines residual Transformer layers
with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
enhancement strategy and a feature-complementary decoder to reconstruct occluded features based on neighboring un-occluded regions. Bian
et al. (2024) proposed a novel occlusion-aware feature recover (OAFR)
model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
critical pedestrian features using an uncertainty-aware self-attention
mechanism.
Although the above Transformer-based methods can address the
problems faced by occluded person ReID to some degree, most of them
fail to adequately account for the diﬀerent contributions of diﬀerent
image patches to identity recognition, nor do they prioritize the identity information contained in key regions. For instance, the methods
such as PAT (Li et al., 2021), PFD (Wang et al., 2022b), and RFMT
(Yang et al., 2023) fail to adequately account for the diﬀerent contributions of diﬀerent image patches to identity recognition, consequently lacking focus on key regions. Moreover, most of the existing
models exhibit insuﬃcient capability in extracting ﬁne-grained local
features, thereby failing to fully exploit the identity information embedded in diﬀerent body parts of pedestrians. For instance, the methods such as MLRAT (Lin et al., 2024) and RFMT (Yang et al., 2023)
do not perform ﬁne-grained local feature extraction, consequently failing to adequately mine the local information of pedestrians.they neither adequately consider the diﬀerence in the contribution of diﬀerent image patches to identity recognition, nor suﬃciently emphasize
the identity information of key regions. Additionally, most of their
ﬁne-grained local feature extraction capabilities are insuﬃcient to fully
explore the identity information embedded in diﬀerent body parts of
pedestrians.
Our work is also related to some works. For example, Eliwa et al.
(2024) proposed a framework that integrates Microsoft Azure cloud services with a permissioned blockchain network. After preprocessing and
anonymizing the CT images uploaded by patients via mobile terminals,

the framework stores the images in Azure Blob Storage, and realizes access control exclusive to authorized specialists through blockchain smart
contracts. Abd El-Hafeez et al. (2025) proposed a novel multi-scale attention model for the classiﬁcation of breast cancer histopathological
images, which achieves high-precision recognition by capturing discriminative features across multiple morphological scales in histopathological images. Eliwa and Abd El-Hafeez (2025c) proposed a robust deep
learning framework improved upon YOLOv11 for the multi-class classiﬁcation task of cervical cancer cells, which enhances the model’s classiﬁcation accuracy via an Attention-Guided Multi-Scale Feature Fusion
(AGMS-FF) module. Eliwa and Abd El-Hafeez (2025a) conducted a rigorous comparative evaluation of ﬁve ﬁne-tuned deep learning architectures for rice maturity classiﬁcation tasks, namely YOLOv11 enhanced
with an Attention-Guided Multi-Scale Feature Fusion (AGMS-FF) module, baseline YOLOv11, ResNet18, EﬃcientNet-B0, and MobileNetV3.
Their results verify the practical value of deep learning-based computer
vision systems in sustainable rice cultivation. Hassan et al. (2025b) proposed a novel DenseNet model integrated with attention mechanisms
and optimized by the Nadam algorithm, which enhances the focus on
pertinent features and thereby improves the model’s classiﬁcation accuracy under complex conditions. Eliwa and Abd El-Hafeez (2025b)
proposed an improved YOLOv11 architecture for the automated classiﬁcation of peripheral blood cells, which integrates a Dynamic CrossScale Context Aggregation (DCSCA) module. Through parallel convolution, dynamic attention, and cross-scale interaction, the module enables
multi-scale feature capture, scale recalibration, and context modeling,
thereby signiﬁcantly improving the model’s classiﬁcation accuracy for
blood cells. Hassan et al. (2025d) proposed a novel network architecture named Sign Neuro-Enhanced Dense Attention Network (SNDA),
which adopts the Nadam optimizer to achieve fast convergence of the
training process. Hassan et al. (2025a) presented a comprehensive review of research advances in the super-resolution ﬁeld from 2015 to
2025. They pointed out that the application of attention modules and
domain-speciﬁc architectures has eﬀectively improved super-resolution
performance, while also indicating that current research still suﬀers
from limitations such as insuﬃcient real-world scenario validation and
the lack of deployment considerations. Hassan et al. (2025c) proposed
a network architecture based on DenseNet121 and the Squeeze-andExcitation (SE) attention module for fruit image classiﬁcation, which
improves the model’s generalization ability through the application of
attention mechanisms, Nadam optimization, and other data augmentation methods. Eliwa and Abd El-Hafeez (2025) proposed a plant disease
detection and classiﬁcation method based on a modiﬁed YOLOv11 architecture, which improves the model’s performance in plant disease
detection and classiﬁcation through strategies such as a custom classiﬁcation head and dynamic validation set splitting. El-Sayed and El-Hafeez
(2012) proposed an improved edge detection method for grayscale images. Based on the Baljit and Amar algorithm, this method replaces
image derivative calculation with Shannon entropy, thereby reducing
CPU computation time while ensuring the robustness of edge detection
quality. Saabia et al. (2019) proposed a face recognition system, which
achieves eﬃcient face recognition through ﬁve sequential steps. These
steps include face detection combined with median ﬁlter preprocessing,
feature extraction using Gabor ﬁlters, dimensionality reduction via Principal Component Analysis (PCA), feature selection based on the Grey
Wolf Optimization (GWO) algorithm, and classiﬁcation using a k-NN
classiﬁer.

3. Proposed method
This section begins with an overview of the HFLAT framework,
then details the feature hierarchical fusion module (FHFM), the feature separation module (FSM), and the local feature extraction module (LFEM). Finally, the loss functions employed for model training are
described.
3


exec
/bin/zsh -lc "pdftotext -l 3 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 192 (2025) 107888

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Hierarchical knowledge-guided reasoning for text-based person
re-identiﬁcation
Ruigeng Zeng

a,b,c , Wentao Ma c,∗, Tongqing Zhou d , Shan Zhao e , Xinjun Mao c , Jie Liu a,b,c

a Laboratory of Digitizing Software for Frontier Equipment, National University of Defence Technology, Changsha, 410073, Hunan, China
b

National Key Laboratory of Parallel and Distributed Computing, National University of Defense Technology, Changsha, 410073, Hunan, China

c School of Information and Artiﬁcial Intelligence, Anhui Agricultural University, Hefei, 230036, Anhui, China
d
e

College of Computer Science and Technology, National University of Defense Technology, Changsha, 410073, Hunan, China
School of Computer and Information Engineering, Hefei University of Technology, Hefei, 230009, Anhui, China

a r t i c l e

i n f o

Keywords:
Text-image person re-identiﬁcation
Scene graph
Knowledge-guided reasoning

a b s t r a c t
Masked language modeling (MLM) has expanded the exploration of text-image person re-identiﬁcation (TIReID)
tasks from coarse-granularity to ﬁne-grained alignment. Whereas, we note that vanilla MLM picks random tokens for visual-to-token reasoning, which could fail the intention of semantic visual-textual alignment by indistinguishably focusing on all the sub-words. This work proposes to leverage the inherent hierarchical scene
graph knowledge in each text for guiding token masking and enhancing cross-modal representation in TIReID,
thus relieving the pitfall of blind visual-textual alignment. The proposed framework, Hierarchical KnowledgeGuided Reasoning (HKGR), parses object-level, attribute-level, and relation-level masking according to phrase
knowledge constructions and explicitly lets the training of a dedicated encoder focus on the visual-to-token reasoning of these highlighted tokens. In addition, we propose a Multi-Grained Semantic Alignment (MGA) module,
which leverages the token selection method and image-text similarity distribution constraint to further facilitate
the semantic alignment between image and text at both coarse-grained and ﬁne-grained levels. Experimental
results demonstrate that our HKGR framework achieves state-of-the-art (SoTA) performance on three public
benchmark datasets at all evaluation metrics. We believe that the knowledge-guided idea is beneﬁcial to other
multi-modal research communities, including cross-modal retrieval and visual question answering. Code is available at https://github.com/Ray-Zhen/HKGR.git.

1. Introduction
Text-image person re-identiﬁcation (TIReID) stands as a fundamental and long-standing task in person re-identiﬁcation, dedicated to
searching pedestrian images with the same identity according to text
descriptions. Given its potential applications in areas such as intelligent surveillance and city security (Li et al., 2019; Zhu et al., 2024),
this technique is progressively garnering widespread attention in both
academic and industrial communities. Despite its signiﬁcance, the task
remains highly challenging, as it demands ﬁne-grained visual-linguistic
alignment to bridge the inherent modality gap between images and text
modalities.
Towards this end, numerous TIReID approaches have been proposed,
which can be broadly categorized into two categories: The ﬁrst category,
global-matching methods, separately maps the visual global representation and textual global representation into a joint embedding space

to calculate the cross-modal similarity that enables cross-modal alignment (Saraﬁanos et al., 2019; Zhang & Lu, 2018; Zheng et al., 2020).
Nevertheless, it can be hard for such a compact representation to capture ﬁne-grained semantic details in texts and images. For example, as
is shown in Fig. 1, understanding the text descriptions involves complicated semantic reasoning regarding diﬀerent objects (‘lady’, ‘pant’,
‘shirt’), attributes (‘black’, ‘white’), and relations (‘hold’, ‘in front of’).
To avoid losing those details, the second category (Chen et al., 2022;
Ding et al., 2021; Wang et al., 2020b, 2022a; Yan et al., 2023a), localmatching methods, leverages detailed visual cues and individual textual words to represent images and texts, respectively, and performs
local semantic alignment to compute overall similarity. Particularly,
the Masked Language Modeling (MLM)-based paradigms (Bai et al.,
2023; Jiang & Ye, 2023; Zuo et al., 2023), which fall under the localmatching methods category, adopt local semantic reasoning to establish ﬁne-grained relationships between image and text representations,

∗ Corresponding author.

E-mail addresses: rgzeng@nudt.edu.cn (R. Zeng), wtma@ahau.edu.cn (W. Ma), zhoutongqing@nudt.edu.cn (T. Zhou), zhaoshan@hfut.edu.cn (S. Zhao),
xjmao@nudt.edu.cn (X. Mao), liujie@nudt.edu.cn (J. Liu).
https://doi.org/10.1016/j.neunet.2025.107888
Received 15 October 2024; Received in revised form 16 July 2025; Accepted 18 July 2025
Available online 19 July 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 192 (2025) 107888

R. Zeng et al.

object-attribute couplings (e.g., “white shirt”) and spatial relationships
(e.g., “in front of”), leading to fragmented textual representations and
an increased semantic gap between global textual descriptions and local
visual cues.
In light of the above analysis, to tackle those tricky challenges, we
propose a novel Hierarchical Knowledge-Guided Reasoning (dubbed
as HKGR) framework for TIReID tasks, which harnesses hierarchical
knowledge in scene graphs parsed from text to enhance the masking
and designs a Knowledge-Guided Reasoning (KGR) module for multilevel consistent alignment, as illustrated in Fig. 1(b). To be speciﬁc,
we decompose text descriptions into three hierarchical semantic levels
through scene graph construction, which are responsible for capturing
attributes, objects, and relations, respectively. Diﬀerent levels are not
independent, and their interactions explain what semantic roles they
play within the sentence, strengthening the conscious masking process
to mitigate semantic distortion caused by random masking (for Challenge 1). Moreover, for the KGR module, we propose object-level reasoning, attribute-level reasoning, and relation-level reasoning tasks, realizing multi-level cross-modal consistency alignment to narrow the semantic gap between image and text (for Challenge 2). Furthermore, we
also design a Multi-Grained Semantic Alignment (MGA) module. MGA
ﬁrst utilizes a token selection method to select multi-grained discriminative information both in visual and text tokens, and then it constrains
the multi-grained image-text similarity distributions for proper crossmodal alignments at both coarse-grained and ﬁne-grained levels. Our
contributions can be summarised as follows:
•

We propose a Hierarchical Knowledge-Guided Reasoning (HKGR)
framework, which decomposes text descriptions into three levels,
bolstering purposeful knowledge masking and alleviating semantic
bias stemming from random masking.
• To realize multi-level cross-modal consistency alignment and narrow
the semantic gap between image and text, we design a KnowledgeGuided Reasoning (KGR) module for better semantic coverage.
• We introduce a Multi-Grained Semantic Alignment (MGA) module
that employs a token selection method to select multi-grained discriminative information and perform proper cross-modal alignments
by multi-grained image-text similarity distributions.
• We evaluate our HKGR via the comparison with 18 SoTA baselines
and a series of ablation studies. The extensive experimental results
reveal that the HKGR can yield promising performance (a.k.a., R@1
metric of 75.21 %, 65.29 %, and 63.10 % on CUHK-PEDES, ICFGPEDS, and RSTPReid, respectively).

Fig. 1. Overview of the MLM-based paradigm and our HKGR paradigm.
(a) The MLM-based paradigm is trained by randomly masking and predicting sub-words based on the unmasked contextual texts and the paired image
patches. (b) Our HKGR proposes hierarchical knowledge from scene graphs to
enhance masking and adopt a knowledge-guided reasoning strategy, realizing
multi-level consistency alignment.

leading to notable improvements in TIReID performance. This motivates
us to investigate MLM in the context of TIReID tasks, aiming to exploit
the ﬁne-grained semantic interaction across images and texts.
In general, MLM-based paradigms rely on randomly masking and
predicting text sub-words to align contextual information with image
patches, as shown in Fig. 1(a). However, we argue that the vanilla
random masking strategy is often suboptimal for image-text alignment
in practical TIReID scenarios. This is primarily due to the complexity
of semantic concepts in textual descriptions. In many TIReID benchmarks (Ding et al., 2021; Li et al., 2017b; Zhu et al., 2021), it is common for multiple textual descriptions to refer to the same individual
from diﬀerent perspectives, further complicating the alignment process.
Therefore, the ﬁne-grained diﬀerences among text descriptions bring
challenges for image-text alignment in MLM-based TIReID paradigms:
1) The randomness of masked language: Given the complexity of semantic information in textual descriptions, vanilla MLM-based methods
typically select subword tokens at random for masking. However, this
approach tends to disproportionately target high-frequency or semantically peripheral tokens. Such indiscriminate masking introduces semantic noise, compelling the model to predict uninformative elements (e.g.,
function words or subword fragments) rather than semantically meaningful keywords or phrases that are critical for ﬁne-grained alignment.
Consequently, the learned visual-textual correspondences may become
diﬀuse and unstable–an issue particularly pronounced in TIReID, where
discriminative semantics (e.g., “red backpack,” “striped shirt”) are essential for accurate identity matching. 2) Ignoring phrase-level semantic
representation:
Existing MLM-based approaches primarily focus on isolated wordlevel predictions, overlooking multi-word phrases, such as objectattribute pairs or relational constructs, that convey richer, compositional semantics. As a result, these models often fail to capture critical

The rest of this paper is organized as follows. First, we brieﬂy review the related work in Section 2, and Section 3 introduces the design
of HKGR. Then we present the experimental settings and results in Section 4 and 5. Finally, conclusions are given in Section 6.
2. Related work
2.1. Text-image person re-identiﬁcation
TIReID, a sub-tasks of person re-identiﬁcation (ReID) (Pang et al.,
2024; Ye et al., 2021; Zheng et al., 2016) and image-text retrieval (Faghri et al., 2017; Qin et al., 2022; Zheng et al., 2020), was
ﬁrst introduced by Li et al. (2017b). It is a ﬁne-grained challenging task
due to its intra- and inter-modal variations. The existing methods can
be roughly categorized into the global-matching paradigm and the localmatching paradigm according to the alignment strategy.
For the global-matching paradigm, the eﬀorts primarily concentrated on designing model structures or objective functions to achieve
proper alignment between image and text modalities in the common
latent space. Early works (Li et al., 2017a,b) adopt CNN and LSTM
networks to extract image and text features and align these features
using an image-text contrastive loss in the shared latent space. Zheng
2

Neural Networks 192 (2025) 107888

R. Zeng et al.

et al. (2020) propose a Dual-Path method that employs a CNN structure for both image and text feature extraction to enable eﬀective endto-end ﬁne-tuning using an instance loss. Zhang and Lu (2018) design
a novel cross-modal projection matching loss to learn discriminative
cross-modal embeddings. Recent works (Gao et al., 2021; Li et al., 2022;
Shao et al., 2022) adopt Transformer-based feature extraction backbones to improve feature representation, achieving promising results.
The global-matching paradigm solely concentrates on global feature representation, overlooking distinctive local details, which may hinder ﬁnegrained cross-modal alignment learning.
The local-matching paradigm primarily concentrates on mining local cross-modal correspondence between image regions and words or
phrases. For instance, in the TIPCB model, Chen et al. (2022) utilize
a dual-path local alignment network to extract local visual and textual representations from horizontally segmented image patches. Subsequently, local representations are aligned adaptively with a multi-stage
cross-modal matching. Zhu et al. (2021) propose a DSSL model to extract and align body part information from images using a mutual exclusion constraint fusion mechanism. Additionally, some works (Ding et al.,
2021; Farooq et al., 2022; Shao et al., 2022; Yan et al., 2023a,b) focus on
mining local feature correspondences with attention mechanisms. For
example, Yan et al. (2023b) proposes an implicit local alignment module to aggregate pixel-level and word-level features, implicitly learning
the local ﬁne-grained correspondence between image-text modalities.
Yan et al. (2023a) designed an MGF module to extract discriminative
local information in each modality and devised CFR and FCD modules
to establish cross-grained and ﬁne-grained interactions between modalities. Another work (Bai et al., 2023; Jiang & Ye, 2023; Zuo et al., 2023)
introduced the MLM-based paradigm to learn relations between local
visual-textual tokens. Speciﬁcally, Jiang and Ye (2023) proposes an implicit relation reasoning module to predict the random masked text tokens based on image patches and unmasked surrounding text tokens,
aiming to align images and text representation. In the work of FLIP (Zuo
et al., 2023), attribute phrases are masked and predicted by combining
masked textual embeddings with global image embeddings to construct
correlations between images and texts.
However, the MLM-based paradigm only masks sub-words, which
may not capture semantic-rich words or phrases, neglecting detailed semantic feature representation and alignment. In this paper, we implement knowledge-guided reasoning by predicting hierarchical semantics
in scene graphs from text, aiming to guide detailed semantic alignment
across visual and text modalities.

information to be passed edge-wise and introduce metapaths to extract
speciﬁc semantic information along designated paths.
In this work, we make the ﬁrst attempt to explore the eﬀectiveness
of hierarchical scene graph knowledge in the TIReID.
2.3. Vision-language pre-training
Vision-language pre-training (VLP) seeks to establish semantic correlations between vision and language. Inspired by the success of pretraining paradigm in single-modal tasks, e.g., language pre-training
model BERT (Devlin et al., 2018) and vision pre-training model
ViT (Dosovitskiy et al., 2020), the VLP has garnered signiﬁcant attention (Chen et al., 2020; Dou et al., 2022; Jia et al., 2021; Kim et al., 2021;
Radford et al., 2021). Based on the model structure, existing VLP methods can be divided into two categories: single-tower and two-tower. The
single-tower models (Chen et al., 2020; Kim et al., 2021) concatenate
the visual and the language features together and then embed them into
a common space. While these models beneﬁt from their parameter efﬁciency due to the shared visual-language encoder, they incur higher
computational costs during the inference stage. The two-tower models
extract image and text features separately with distinct encoders. These
models exhibit fast retrieval speed due to the independent feature encoders.
As a representative VLP model, CLIP (Radford et al., 2021) exhibits
high-quality visual-language semantic representation capacity and has
been applied to various downstream multi-modal tasks, including textvideo retrieval (Fang et al., 2021; Luo et al., 2022; Ma et al., 2022)
and TIReID (Jiang & Ye, 2023; Yan et al., 2023a). Speciﬁcally, Luo
et al. (2022) are the ﬁrst to explore the transfer of CLIP knowledge
into video-text cross-modal retrieval and demonstrate that a large-scale
multi-modal pre-training model is beneﬁcial for video-text retrieval.
Ma et al. (2022) propose a novel end-to-end multi-grained contrastive
model, XCLIP, to capture correlations between cross-grained comparisons. Yan et al. (2023a) pioneer the integration of CLIP visual representations into TIReID, proposing a CLIP-driven method that achieves
ﬁne-grained cross-modal person re-identiﬁcation. To fully exploit CLIP’s
powerful capabilities, Jiang and Ye (2023) adopt both the visual and
language transformer encoder of CLIP to learn ﬁne-grained cross-modal
implicit local relations. In this paper, following the line of work (Jiang
& Ye, 2023), we also leverage the pretraining knowledge of CLIP for
the TIReID task.
3. The design of HKGR

2.2. Scene graph
The overall architecture of our HKGR is illustrated in Fig. 2. In what
follows, we ﬁrst introduce the image-text feature representation in Section 3.1, and then describe the scene graph construction in Section 3.2,
KGA module in Section 3.3, MGA module in Section 3.4, objective function and training strategy in Section 3.5, respectively.

Scene graph, which represents objects, attributes of objects, and relations between objects with a graph, was ﬁrst proposed by Johnson
et al. (2015). With the advancement of scene graph generation (Zellers
et al., 2018), scene graph knowledge has been extensively integrated
into multi-modal tasks (Yu et al., 2021), such as image captioning (Yang
et al., 2019; Yao et al., 2018), visual question answering (Hildebrandt
et al., 2020), and cross-modal retrieval (Duan et al., 2021; Guo et al.,
2020; Wang et al., 2020a).
Yu et al. propose an ERNIE-ViL (Yu et al., 2021), a vision-language
pre-training framework that integrates structured knowledge obtained
from scene graphs to learn ﬁne-grained semantic alignment across vision
and language, achieving promising performance on ﬁve cross-modal
downstream tasks. To model relations between objects in image captioning, Yao et al. (2018) presents a GCN-LSTM model to build graphs with
the detected objects from an image based on their spatial and semantic connections. In terms of cross-modal retrieval, Wang et al. (2020a)
introduce both visual scene graphs and textual scene graphs to characterize objects and their relations, optimizing the model by computing
object-level and relation-level similarities. To fully utilize the beneﬁts of
graph structure, Duan et al. (2021) adopt a heterogeneous scene graph
that combines a visual scene graph and textual scene graph to enable the

3.1. Image-text feature representation
Image feature representation. For image feature representation,
we follow previous studies (Han et al., 2021; Yan et al., 2023a), that
is, adopting the ViT initialized by pre-trained CLIP checkpoints as the
image encoder. Speciﬁcally, for an image 𝐼𝑖 ∈ ℝ𝑐×𝑤×ℎ , we ﬁrst evenly
split 𝐼𝑖 into 𝑛𝑣 = (ℎ × 𝑤)∕𝑝2 ﬁxed-dimension image patch sequences,
where 𝑝 represents the size of image patches. Then the split image
𝑛𝑣
patch sequence is ﬂattened into 1D token sequence {𝑓𝑖𝑣 }|𝑖=1
with a trainable fully connection layer. Finally, we concatenate the token sequence
with a learnable [CLS] token as the preﬁx and feed it into the transformer blocks of ViT. The output image features can be represented as
𝑓𝑖𝑣 = {𝑣𝑖𝑐𝑙𝑠 , 𝑣𝑖1 , 𝑣𝑖2 , …, 𝑣𝑖𝑛 }, where 𝑣𝑖𝑐𝑙𝑠 is the image-level global feature,
𝑣
{𝑣𝑖1 , 𝑣𝑖2 , …, 𝑣𝑖𝑛 } is the patch-level local features.
𝑣
Text feature representation. In terms of text feature representation, we adopt the text encoder of pre-trained CLIP as a text feature
3


exec
/bin/zsh -lc "pdftotext -l 3 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10889915

Hierarchical Proxy Learning for Cloth-Changing
Person Re-Identification
Chenyang Yu∗ , Xuehu Liu† , Ju Dai‡ , Pingping Zhang§ and Huchuan Lu∗§
∗ School of Information and Communication Engineering, Dalian University of Technology, Dalian, China
† School of Computer Science and Artificial Intelligence, Wuhan University of Technology, Wuhan, China
‡ Peng Cheng Laboratory, Shenzhen, China
§ School of Future Technology, School of Artifcial Intelligence, Dalian University of Technology, Dalian, China

Abstract—Cloth-Changing person Re-Identification (CC-ReID)
depends significantly on learning discriminative features under
the cloth-changing scenario. It is quite challenging due to
the large intra-person variance and small inter-person variance caused by clothes changing. To address these issues, in
this work we propose a Hierarchical Proxy Learning (HPL)
framework to extract clothes-irrelevant and person-invariant
features. Specifically, we employ person labels as the main proxy.
Instead of leveraging clothing labels as sub proxy, we further
propose a clustering-based automatic sub-proxy mining scheme.
More specifically, we first construct a person-aware Main Proxy
Learning (MPL) to improve the separability of different persons.
Then, a Sub Proxy Learning (SPL) is constructed to enhance the
intra-person compactness. Finally, a Sub-to-Main Proxy Learning
(S2MPL) is proposed to promote the cooperation between the
main proxies and sub proxies. In addition, to weed out the
negative effect of clothes, we propose a Sample Balance and
Diversity (SBD) module, which balances the number of sub
proxies in a mini-batch and utilizes semantic guidance to enrich
the diversity of clothes, simultaneously. Extensive experiments on
two public CC-ReID datasets demonstrate the superiority of our
proposed method over most state-of-the-art methods.
Index Terms—cloth-changing person re-identification, hierarchical proxy learning, sample balance, joint training.

I. I NTRODUCTION
Cloth-Changing person Re-Identification (CC-ReID) is a
long-term retrieval task, which aims at re-identifying target
persons across non-overlapping cameras. Compared with traditional ReID [1]–[6], CC-ReID [7], [8] is encountering more
realistic challenges. Despite being quite challenging, CC-ReID
is receiving more and more interest from researchers due to
its crucial role in more realistic scenario applications.
To address CC-ReID, previous methods [9]–[11] aim to
eliminate the impact of clothes, and extract the inherent
characteristics of pedestrians, such as 3D human shape, gait
information, contour sketches, etc. However, these inherent
characteristics are not as effective as appearance features,
leading to some performance deteriorations on the sameclothing ReID. Furthermore, various approaches rooted in metric learning [12], [13] and data augmentation techniques [14],
[15] have been introduced to tackle the CC-ReID problem.
In fact, a critical step in ReID is to design a good distance
metric [16]. As shown in Fig. 1 (a) and (b), due to the
This work was supported in part by the National Natural Science Foundation
of China (No. 62102208).

Dinter

P1

C1

P2

Dintra

(a) Instance-level Metric Learning

ipos Pull

i

MPL

 pneg

Push

 i ,1

Augmented Instances
Pull

i

Push

Pull

 i ,C

Dinter

Dintra

C3

C4

(b) Illustration of the Feature Distribution
Clothe 1

f p2,1

p

f p1,1

 p ,1
fˆi ,kc

Dinter

C2

Dintra

Dintra

Push

S2MPL

p
 p ,1

 p ,C

Clothe C

1
f pK,1 f p ,C

Pull

f p2,C

Sub Proxies

f pK,C

Raw Instances

 p ,C

SPL

Main Proxies

Generate Proxy

MPL

Main Proxy Learning

S2MPL

Sub-to-Main Proxy Learning

SPL

Sub Proxy Learning

(c) Hierarchical Proxy Learning

Fig. 1. Our motivations. (a) Geometry interpretation of instance-level metric
learning. (b) Illustration of the feature distribution of randomly selected
persons from CC-ReID datasets. (c) Geometry interpretation of the proposed
Hierarchical Proxy Learning (HPL). Different colored dots and shapes represent different persons and sub proxies identities, respectively.

large intra-person variance and small inter-person variance
caused by changing clothes, the instance-level triplet loss [17]
and contrastive loss [18], [19] cannot achieve satisfactory
performance. Recently, some works [20]–[22] perform ReID
by proxy-based metric learning. For example, Wang et al. [23]
propose intra-camera and inter-camera proxy contrastive learning. For CC-ReID, Gu et al. [24] design a clothes-based
adversarial loss to further pull the features with the same identity closer. Unfortunately, both of them focus on instance-toproxy interactions, and neglect inter-proxy relations. Different
from previous methods, as shown in Fig. 1 (c), we propose a
Hierarchical Proxy Learning (HPL) framework, which consists
of a Main Proxy Learning (MPL), a Sub Proxy Learning
(SPL) and a Sub-to-Main Proxy Learning (S2MPL). In MPL,
we first create main proxies for individuals, then bring the
proxies of the same person closer while distancing those of
different people, enhancing inter-person separability. In SPL,
we create sub proxies for each person and group instances
with different sub proxies to improve intra-person compactness. Unlike [24] which directly using clothing labels as sub
proxies, we propose a clustering-based automatic sub-proxy
mining scheme. In S2MPL, each sub-proxy acts as an anchor,
being pulled toward its corresponding main proxy and pushed
away from others, promoting inter-person diversity and intraperson compactness. Such a hierarchical structure contributes

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:35 UTC from IEEE Xplore. Restrictions apply.

Sub-proxy Mining Original
Clustering

C1

C2

Images

Pb

C2

Human Parsing

Pa

︙
Clustering
C1

Pb

Pa

SCT

SBS

C3

Classifier

Pb

Pa

𝐿𝐻𝑃𝐿
Pull
Pull

Pb

: Main Proxy

: Sub Proxy

Pull

Clothes Change

Augmented
Images

𝐿𝐼𝐷
Push

GAP+GMP

a

C4

ResNet-50

C1 P C2

C3

: Instances of
Different Sub
Proxies
Human Parsing
: A Pre-trained
Human Parsing
Network
: Pull

Part1: Sample Balancing and Diversity Strategy

Part2: Hierarchical Proxy Learning

: Push

Fig. 2. Illustration of the proposed framework.

to extracting person-invariant and clothes-irrelevant features.
As shown in Fig. 1 (c), when constructing a hierarchical
structure, if there is no assistance, the main proxy P1 in a minibatch will have no corresponding positive samples. What’s
more, if there is no constraint, the distribution of sub proxies
will be random. Meanwhile, due to the annotation limitation
in current CC-ReID datasets, it is highly possible for a person
who wears one clothes all the time. Considering the above
issues, we further propose a Sample Balance and Diversity
(SBD) module, which balances the number of sub proxies
in a mini-batch and utilizes semantic guidance to enrich
the diversity of clothes, simultaneously. Specifically, we first
explore a Sub-proxy Balanced Sampling (SBS) strategy taking
the balance and diversity of sub proxies into consideration,
which is more suitable for CC-ReID. Then, a Semanticguided Clothes Transfer (SCT) is proposed to enrich the
diversity of clothes, which utilizes a pre-trained human parsing
network [25] to guide clothing changing. Thanks to SCT, we
can get the main proxy positive samples corresponding to
pedestrians based on the augmented samples. Experimental
results demonstrate that our method significantly outperforms
most state-of-the-art works on two public CC-ReID datasets.
The contributions of our work can be summarized as: (1)
We propose an effective data processing module named SBD
for CC-ReID. (2) We propose a novel proxy-level metric
learning method with a hierarchical structure to extracting
person-invariant and clothes-irrelevant features. (3) Extensive
experiments demonstrate that our proposed method outperforms most state-of-the-art cloth-changing methods on two
widely-used CC-ReID datasets, i.e., PRCC and VC-Clothes.

clustering-based automatic sub-proxy mining scheme. Specifically, before each round of network training, we cluster all
Np
the feature representations {fnp }n=1
for each person p into Cp
clusters whose pseudo-labels are used as the sub-proxies. In
practice, we adopt the DBSCAN [28] method for clustering.
Sub-proxy Balanced Sampling. The sampling strategy [17]
in traditional ReID mainly considers the balance of different
persons but ignores the balance of different clothes. Intuitively,
it is useful to choose balanced sub-proxy in each batch for CCReID. Therefore, we propose a Sub-proxy Balanced Sampling
(SBS) strategy. We choose P persons in each mini-batch,
where C sub proxies per person and K images per sub proxy.
Our SBS strategy performs a balanced optimization of persons
and sub-proxy, thereby promoting the learning efficacy.
Semantic-guided Clothes Transfer. In CC-ReID datasets,
some persons may wear only one clothes all the time. Data
augmentation is an effective strategy to enrich the diversity of
training samples in CC-ReID. We propose a Semantic-guided
Clothes Transfer (SCT) to change clothes among different persons. Specifically, given one image xi ∈ {xkp,c }P,C,K
p=1,c=1,k=1 ,
we first randomly select another image xj with different person
and sub proxy in a mini-batch. Then, a pre-trained human
parsing network [25] is employed to obtain semantic masks
of xi and xj . Considering that the most common dressing
parts for persons are upper-clothes and pants, we perform SCT
based on the masks of upper-clothes and pants, respectively.
Given the upper-clothes masks mi and mj of two pedestrians,
we can transfer the upper-clothes of xj to xi ,

II. M ETHODS

where means the matrix multiplication. M ean(·) calculates
the average pixel value of the upper-clothes to address the variability in the clothing area of different persons. Reshape(·)
duplicates the pixel value to the same shape of the target image
xi . Similarly, we can change pants from xj to xi .
In one mini-batch, we get the corresponding augmented
image x̂ of each image x ∈ {xkp,c }P,C,K
p=1,c=1,k=1 through SCT.
Meanwhile, its person label remains unchanged while the
clothes label has been changed. Thus, our SCT can generate
more training samples for one person dressing in different
clothes, which enriches the diversity of samples for CC-ReID.

As illustrated in Fig. 2, our proposed framework mainly includes two components: Sample Balance and Diversity (SBD)
module and Hierarchical Proxy Learning (HPL). Detailed
descriptions are presented in the following sections.
A. Sample Balance and Diversity
Recent methods [24], [26], [27] utilize clothes labels to
improve the performance of CC-ReID. However, obtaining the
clothes labels requires a certain price. Inspired by the recent
success of unsupervised person ReID methods, we propose a

x̂i = xi

(1 − mi ) + Reshape(M ean(xj

mj ))

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:35 UTC from IEEE Xplore. Restrictions apply.

mi , (1)

B. Hierarchical Proxy Learning
xkp,c

For feature extraction, we feed the original images
and augmented images x̂kp,c into the ResNet-50 following a
GAP and a GMP to obtain the feature vectors f k , fˆk .
p,c

p,c

In our baseline method, we adopt the cross-entropy loss for
person classification. However, merely person classification is
hard to reduce the intra-person variance and increase the interperson variance. As shown in Fig. 1 (c), we propose a novel
Hierarchical Proxy Learning (HPL) framework, including a
Main Proxy Learning (MPL), a Sub Proxy Learning (SPL)
and a Sub-to-Main Proxy Learning (S2MPL).
Main Proxy Learning. As stated in [24], the instance-level
metric learning may lead to a sub-optimization for CC-ReID,
because it only mine the hard cases in a mini-batch and is
sensitive to noisy positives and negatives. Thus, to alleviate
this problem, we propose a Main Proxy Learning (MPL). The
illustration of our MPL is shown in the blue part of Fig. 1 (c).
Specifically, based on the features of persons, the main proxy
ρi can be constructed by:
ρi =

C,K
X

1
CK

k
fi,c
, i ∈ [1, P ].

(2)

c=1,k=1

The main proxy ρi can be seen as an anchor. Then, we
can obtain the corresponding positive main proxy ρpos
from
i
augmented samples by:
ρpos
=
i

1
CK

C,K
X

k
fˆi,c
, i ∈ [1, P ].

(3)

c=1,k=1

Afterward, the negative main proxy ρneg
which has a different
i
person label with the anchor can be defined as:
ρneg
=
p

1
CK

C,K
X

k
fp,c
, p 6= i, p ∈ [1, P ].

(4)

c=1,k=1

For one anchor, we have J = (P − 1) × 2 negative main
proxies. Thus, the loss of MPL can be defined as:
P

1 X
neg
LM P L =
α+D(ρi , ρpos
i )−minD(ρi , ρp ) + ,
P i=1

(5)

where D(, ) is the Euclidean distance, min represents the
minimized distances among negative pairs for obtaining the
hardest negative main proxies in the mini-batch. α is a margin
hyper-parameter and [·]+ represents the hinge loss. Different
from previous methods, our proposed MPL can suppress the
influence of noisy samples in feature optimization.
Sub Proxy Learning. Our MPL does not take the intraperson compactness into account. Thus, as shown in the green
part of Fig. 1 (c), we further propose the Sub Proxy Learning
(SPL) to resolve this problem. Specifically, thanks to SBS, we
can sample C sub proxies for the p-th person in a mini-batch
and construct sub proxy by:
K

δp,c =

1 X k
fp,c , c ∈ [1, C], p ∈ [1, P ].
K
k=1

(6)

In SPL, we constrain the feature learning to pull the different
sub proxies of the same person closer. Thus, the loss of our
SPL can be expressed as:
LSP L =

P X
C X
K
C
X
X

k
D(δp,c , fp,i
),

(7)

p=1 c=1 k=1 i=1,i6=c

where the first three summation items represent traversing
all samples obtained by the SBS strategy, and each sample
is regarded as an anchor. The last summation item aims to
calculate the distance between each sample and its proxy of instance samples with the same person but different sub proxies.
Considering that the mined sub proxies act as substitutes for
clothing labels. Under the constraint of SPL, the intra-person
variance caused by clothes transformation will be reduced.
Sub-to-Main Proxy Learning. To achieve the collaboration
between the main proxies and sub-proxies, we further propose a novel component called Sub-to-Main Proxy Learning
(S2MPL) to effectively address the inter-person diversity and
intra-person compactness. Specifically, as shown in the yellow
part of Fig. 1 (c), each sub proxy δi,c is treated as an anchor.
S2MPL pulls it towards the corresponding main proxy ρi , and
pushes it away from the others. The S2MPL is formulated as:
P

LS2M P L = −

C

exp(d(δi,c , ρi )/τ )
1 XX
log PP
,
P C i=1 c=1
p=1 exp(d(δi,c , ρp )/τ )

(8)

where d(, ) is the cosine distance and τ is a temperature
factor. It should be emphasized that, to ensure a reasonable
tolerance level of intra-person variances, we do not impose
strict constraints on the similarity between instances in the subproxy. Similarly, the augmented sample x̂ki,c generated from
the original sample xki,c is also not strictly enforced to be the
same as xki,c . The overall HPL loss is given by the combination
of all the above losses:
LHP L = LSP L + LM P L + LS2M P L .

(9)

C. Joint Training
In this paper, we also apply the cross-entropy loss LID on
the feature vectors to train the baseline network. Thus, the
total loss function Ltotal can be formulated as:
Ltotal = LID + λLHP L ,

(10)

where λ is a balanced parameter.
III. E XPERIMENTS
A. Datasets and Evaluation Protocols
We evaluate our approach on two benchmark datasets, i.e.,
PRCC [29] and VC-Clothes [30]. There are two test settings
in CC-ReID, namely cloth-changing setting (CC), and sameclothes setting (SC). To measure the performance, we adopt
the Cumulative Matching Characteristic (CMC) table and
mean Average Precision (mAP).

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:35 UTC from IEEE Xplore. Restrictions apply.


codex
第一批已经能看出一个规律：这些文章的卖点不是单纯堆模块，而是把“随机、平均、实例级”的训练信号改成有层级的信号。下一批我会重点看它们怎样把层级结构、人体解析、身份解耦包装成可验证机制。
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
/bin/zsh -lc "pdftotext -l 3 'HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Page 1 of 26

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

1

HOH-Net: High-Order Hierarchical Middle-Feature
Learning Network for Visible-Infrared Person
Re-Identification
Liuxiang Qiu, Si Chen, Senior Member, IEEE, Jing-Hao Xue, Senior Member, IEEE, Da-Han
Wang, Member, IEEE, Shunzhi Zhu, Yan Yan, Senior Member, IEEE

Abstract—Visible-infrared person re-identification (VI-ReID)
is a cross-modality retrieval task that aims to match images of
the same person across visible (VIS) and infrared (IR) modalities.
Existing VI-ReID methods ignore high-order structure information of features and struggle to learn a reliable common
feature space due to the modality discrepancy between VIS
and IR images. To alleviate the above issues, we propose a
novel high-order hierarchical middle-feature learning network
(HOH-Net) for VI-ReID. We introduce a high-order structure
learning (HSL) module to explore the high-order relationships of
short- and long-range feature nodes, for significantly mitigating
model collapse and effectively obtaining discriminative features.
We further develop a fine-coarse graph attention alignment
(FCGA) module, which efficiently aligns multi-modality feature
nodes from node-level and region-level perspectives, ensuring
reliable middle-feature representations. Moreover, we exploit
a hierarchical middle-feature agent learning (HMAL) loss to
hierarchically reduce the modality discrepancy at each stage
of the network by using the agents of middle features. The
proposed HMAL loss also exchanges detailed and semantic
information between low- and high-stage networks. Finally, we
introduce a modality-range identity-center contrastive (MRIC)
loss to minimize the distances between VIS, IR, and middle
features. Extensive experiments demonstrate that the proposed
HOH-Net yields state-of-the-art performance on the image-based
and video-based VI-ReID datasets. The code is available at:
https://github.com/Jaulaucoeng/HOS-Net.
Index Terms—Visible-infrared person re-identification, highorder structure, middle-feature learning.

I. I NTRODUCTION

P

ERSON re-identification (ReID) [1]–[3] has drawn more
and more attention in recent years because of its critical

This work was supported in part by the National Natural Science Foundation
of China (No. 62372388); the Natural Science Foundation of Xiamen (No.
3502Z202573073); the Unveiling and Leading Projects of Xiamen (No.
3502Z20241011); the Major Science and Technology Plan Project on the
Future Industry Fields of Xiamen City (No. 3502Z20241027); the Open
Project of the State Key Laboratory of Multimodal Artificial Intelligence
Systems (No. MAIS2024101). (Corresponding author: Si Chen.)
Liuxiang Qiu, Si Chen, Da-Han Wang, and Shunzhi Zhu are with the
Fujian Key Laboratory of Pattern Recognition and Image Understanding,
School of Computer and Information Engineering, Xiamen University of
Technology, Xiamen 361024, China, and Liuxiang Qiu is also with the
School of Informatics, Xiamen University, Xiamen 361005, China (email:
liuxiangqiu007@gmail.com; chensi@xmut.edu.cn; wangdh@xmut.edu.cn; szzhu@xmut.edu.cn).
Jing-Hao Xue is with the Department of Statistical Science, University
College London, London WC1E 6BT, UK (e-mail: jinghao.xue@ucl.ac.uk).
Yan Yan is with the School of Informatics, Xiamen University, Xiamen
361005, China (e-mail: yanyan@xmu.edu.cn).

role in security and surveillance. Visible-infrared person reidentification (VI-ReID) leverages both visible (VIS) and
infrared (IR) cameras to match pedestrian images across the
bright and low-light conditions. The VI-ReID methods not
only mitigate the problems of single-modality ReID (e.g.,
occlusion and posture deformation), but also need to handle
the modality discrepancy between VIS and IR images.
To bridge the modality gap, existing VI-ReID methods
can be classified as image-level and feature-level. Imagelevel methods [4], [5] often employ generative adversarial
networks (GANs [6]) to generate middle or new modality
images. For instance, to mitigate the modality discrepancy, Wei
et al. [4] introduced a reciprocal bidirectional framework that
generates the middle modality images from the latent space
by translating two opposite mappings between VIS and IR
modalities by the generative adversarial network. However,
GAN-based methods easily encounter issues such as color
inconsistency or the loss of image details, which make the
generated images less reliable for training and subsequent
retrieval.
Feature-level methods [1], [7]–[10] typically adopt a twostep learning process. First, these methods extract VIS and IR
feature maps using weight-specific sub-networks separately.
Subsequently, the weight-shared feature extraction projects
these modality-specific features into a common feature space.
For instance, Liang et al. [11] developed a pure Transformer
network to capture long-range information from different
modalities with the modality-aware enhancement loss. To
enhance feature representation, Zhang et al. [9] have attempted
to introduce the self-distillation to consistently focus on discriminative regions from the high-stage to the low-stage for
modality feature learning. The above feature-level methods
generally have three shortcomings. First, they often neglect
the high-order structural information of features, such as the
complex dependencies across feature nodes, which are crucial
for retrieving cross-modality images. Second, the traditional
methods extract person features from the low-stage to the highstage or enhance the feature representation by distillation from
the high-stage to the low-stage. Such strategies ignore the bidirectional interaction between the low-stage and the highstage and are thus hard to explore the detailed and semantic
features. Third, existing methods try to directly minimize
the distances between VIS and IR features, or generate the
auxiliary features from one or two modalities to mitigate the
modality discrepancy, but they still lack efficient alignment

IEEE Transactions on Circuits and Systems for Video Technology
PREPRINT SUBMITTED TO TCSVT, JULY 2025

2

Bi-directional Feature Enhancement
From low
to high

From high
to low
HSL
SLE

Stage j

Stage i

short- and longrange features
high-order
enhanced features

FC
G
A

VIS Image

Middle features
Cross-range cross-modality
features are aligned at
node-level and region-level.

H
Middle- M
feature A
agent
L

FC
G
A

Loss function

H
M MiddleA feature
L agent

HSL

IR Image
SLE

Stage j

Stage i

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

short- and longrange features
high-order
enhanced features

From high
to low

Bi-directional Feature Enhancement

From low
to high

Fig. 1. Illustration of the proposed method. Our HOH-Net aligns different
modalities and ranges of high-order enhanced features at node-level and
region-level simultaneously to generate the reliable middle-feature agents, and
leverages the bi-directional feature enhancement to hierarchically reduce the
modality discrepancy.

and full utilization between different modality features.
To address the above issues, we propose a novel highorder hierarchical middle-feature learning network (HOHNet), which is shown in Fig. 1. The HOH-Net is made up of
a high-order structure learning (HSL) module, a fine-coarse
graph attention alignment (FCGA) module, a hierarchical
middle-feature agent learning (HMAL) loss, and a modalityrange identity-center contrastive (MRIC) loss for VI-ReID.
The key innovation of our method lies in the novel formulation
of exploiting high-order structure information and hierarchical
middle-feature learning to learn a discriminative and reliable
common feature space, thereby significantly mitigating the
modality gap.
Specifically, given a VIS-IR image pair, the HSL module
captures the high-order relationships between the short-range
and long-range features that are extracted from the short- and
long-range feature extraction (SLE) module using a whitened
hypergraph. Instead of directly adding or concatenating features from different modalities and ranges, we design an
FCGA module that aligns these features appropriately and
effectively at node-level and region-level simultaneously to
achieve reliable middle features. Besides, we propose a HMAL
loss to address the modality gap hierarchically by utilizing
middle-feature agents and executing bi-directional interactions
between different stages to enhance feature representation.
Finally, we reduce the distances among VIS, IR, and middle
center features by an MRIC loss, thereby smoothing the
learning process of the common feature space between modalities. On the SYSU-MM01, RegDB, LLCM, and HITSZVCM datasets, our method achieves impressive 76.2%, 95.1%,
65.7%, and 74.8% in Rank-1, respectively.
The main contributions of our work are as follows:
•

Page 2 of 26

We propose an HSL module to learn high-order structure
information of both short and long-range features. Such

a novel way effectively models high-order relationships
across different feature nodes of each pedestrian image
and avoids the problem of model collapse.
• We design a lightweight yet effective FCGA module
that can refine the details of each high-order node-level
feature and perceive the semantic association of regionlevel features simultaneously to achieve reliable middle
features.
• An HMAL loss is designed to hierarchically reduce
the modality discrepancy at each stage network by the
middle-feature agents and perform the bi-directional feature enhancement between different stages to enhance the
detailed representation and the semantic relationship of
features.
• An MRIC loss is designed to minimize the distances
between VIS, IR, and middle features in the embedding space. This is beneficial to extracting discriminative
modality-shared pedestrian features.
This paper significantly extends our previous conference
work HOS-Net [12]. The limitations of our previous work
include the following: First, the computational cost of generating the middle features through graph attention is high
and did not make full use of the middle features. Second,
the previous method extracted modality-shared features from
the low stage to the high stage, ignoring the importance of
bi-directional interaction between different stages that can
enhance feature representation. The HOH-Net addresses these
limitations in two main ways. (1) We further develop a finecoarse graph attention alignment (FCGA) module to refine
the high-order node-level features and perceive the contextual
relationship between region-level features to achieve more
reliable middle features with less model complexity. (2) We
design an HMAL loss to mitigate modality discrepancy from
a hierarchical view by introducing the agents of the middle
features at each VIS and IR modality-shared feature extraction
stage. The proposed HMAL loss also enables the bi-directional
interaction of features between different stages, for obtaining
richer semantic and more detailed feature information than
the previous HOS-Net. In the experiments, we also provide
more comprehensive experimental evaluations, including comparative experiments, ablation studies, parameter analyses,
and visualization analyses. Compared to the previous HOSNet, the HOH-Net achieves lower computational cost and
superior retrieval accuracy than our previous work (the number
of parameters of the HOH-Net is reduced by 29.5%) and
the Rank-1 of our method is improved by 0.6%, 0.4%, and
0.8% on the three image-based VI-ReID datasets, i.e., SYSUMM01, RegDB, and LLCM, respectively. In addition, our
method can also be easily extended to the video-based VIReID field, and compared to the existing video-based methods,
our HOH-Net achieves the best 74.8% Rank-1 on the HITSZVCM dataset.
II. R ELATED W ORK
A. Visible-Infrared Person Re-Identification (VI-ReID)
VI-ReID methods can be divided into image-level and
feature-level methods to reduce the modality discrepancy. The

Page 3 of 26

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

image-level methods [4], [5], [13] often minimize the modality
gap by generating middle-modality images or new modality
images. Wang et al. [13] attempted to introduce a generative
adversarial network to generate new modality images from
VIS and IR modalities by jointly aligning the pixel-level and
feature-level features. Liu et al. [5] proposed a two-stage
modality enhancement network to perform the cross-modality
style translation and optimized the structures of images for
VI-ReID. Besides, Li et al. [14] leveraged the anaglyph data
of the pedestrian as the middle modality images to reduce the
modality gap. Du et al. [15] proposed a channel-blended transformation mechanism to confuse the VIS and IR information
and reduce the influence of modality-specific features, thereby
facilitating the learning of modality-shared features. However,
image-level methods easily encounter issues such as color
inconsistency or the loss of image details when generating
images by the generative adversarial network (GAN), which
is less reliable for training and subsequent visible-infrared
retrieval.
The feature-level methods seek to reduce the modality
discrepancy by mapping the features of different modalities
into a common feature space. A few methods [1], [8], [16]
leverage the weight-shared CNN or ViT as the backbone
to extract modality-shared features. Hybrid models of CNN
and Transformer [10], [17]–[20] can effectively extract shortrange and long-range features. For example, Zhao et al. [10]
enhanced the spatial-channel information of the pedestrian by
adopting the CNN-Transformer hybrid network. Chen et al.
[20] attempt to introduce the off-the-shelf key point extractors
(e.g., OpenPose [21]) to generate key point labels of person
images and achieve features based on the CNN-Transformer
hybrid network, aiming to learn modality-irrelevant features.
But the key point extractor may bring noisy labels, deteriorating the discriminability of final ReID features. However,
the above feature-level methods neglect the high-order structure information of features (i.e., the complex and diverse
relationships across features) that is important for VI-ReID.
To solve the above problem, our work introduces the highorder structure learning to obtain the high-order relationships
between the short- and long-range features and avoid the
model collapse by a whitened hypergraph.
To obtain a common feature space, a lot of feature-level
methods [1], [8], [22]–[24] employ the contrastive-based loss
that directly minimizes the distances between VIS and IR
features. However, it is not a trivial task to learn a reliable
common feature space due to the large modality discrepancy
between modalities. Different from these methods that tend
to minimize the distances between VIS and IR features directly, Zhang et al. [25] tried to generate diverse VIS or
IR embeddings for learning informative feature representations to mitigate the modality gap. Jiang et al. [26] adopted
the modality-level and instance-level alignments for learning robust modality compensation. Li et al. [27] introduced
the cross-modality semantic alignment to explore the intermodality correlation for eliminating the modality discrepancy.
However, they ignored the importance of fine-coarse alignment
for generating reliable middle features from different modalities and ranges to narrow the difference between VIS and IR

3

images. Different from these methods, our method generates
reliable hierarchical middle-feature agents via the fine-coarse
graph attention alignment, greatly promoting our method to
learn a discriminative and reliable common feature space.
In addition, to improve the discriminative ability of the
network, Yang et al. [28] designed a saliency response module that adopts the location attention mechanism to build
contextual connections between person features. Tian et al.
[29] adopted the variational self-distillation to fit the mutual
information between the input feature and its representation,
thus obtaining the multi-view information for VI-ReID. The
above methods follow low-to-high feature extraction, which
ignores the interaction between features at different stages. To
this end, the proposed HOH-Net performs the bi-directional
enhancement between different stages to enhance the detailed
representation and the semantic relationship of features. Moreover, we reduce the distances among VIS, IR, and middle
center features by a modality-range identity-center contrastive
loss, thereby smoothing the learning process of the common
feature space between ranges and modalities.
B. Graph Neural Network
Graph neural network (GNN) is a type of neural network
to process graph-structured data. Zhang et al. [30] adopted the
GNN to select correlated nodes for information aggregation,
thereby establishing the robust connection between the target
and the search regions. Zhang et al. [31] introduced the
GNN to perform the progressive relationship-mining for textto-image ReID. Contrasting with the vanilla graph models
that only allow connections between two nodes, Feng et al.
[32] proposed the novel hypergraph neural network (HGNN)
to represent high-order feature correlations by utilizing a
hypergraph structure. Wadhwa et al. [33] adopted the HGNN
to learn the complex relationship among the incomplete features for the image inpainting. Han et al. [34] utilized the
power of the hypergraph to encode image information and
update the hypergraph structure by the fuzzy c-means method
that can reduce the computational burden. Nevertheless, the
above methods that rely on the HGNN may easily suffer
from the model collapse (i.e., complex and diverse highorder correlations collapse to a single correlation) since the
small differences in the feature nodes of pedestrians and the
hyperedge can connect an arbitrary number of nodes. Different
from the above methods, this paper introduces the whitening
operation to HGNN, which can play the role of “scattering” on
the nodes of the hypergraph, thereby significantly alleviating
model collapse.
Besides, to establish the correspondence between feature
nodes, several methods [35]–[37] attempt to introduce the
graph attention network (GAT) to enhance the representation
of features. For instance, Dong et al. [35] fused the characteristics of CNN and GAT to discover feature connections for hyperspectral image classification. However, the above methods
consider the correspondence between feature nodes at nodelevel, and ignore the semantic connections between regionlevel features that can encapsulate the context of features. In
this work, we develop a fine-coarse graph attention alignment


exec
/bin/zsh -lc "pdftotext -l 3 'HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
第二批的共同点更明确：可发表的叙事通常先指出一个“已有监督或生成方式会破坏身份”的风险，再设计一个中间表征或辅助流来约束它。最后三篇我会看扩散生成、文本检索和合成数据这条线，重点找我们能借的强问题定义。
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
/bin/zsh -lc "pdftotext -l 3 'Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
清单里实际是 9 篇。我只读了每篇前三页，判断基于摘要和引言。

**逐篇提取**

1. **HFLAT 遮挡 ReID**
创新类型：工程组合偏新机制。  
为什么能发：抓住 Transformer 遮挡方法“patch 等权、背景遮挡未分离、局部细粒度弱”的 gap，用层级 patch 融合、显著性前景分离、局部注意力串证据。  
story：不是所有 patch 都同等重要，遮挡 ReID 要按身份贡献组织局部证据。

2. **HKGR 文本 ReID**
创新类型：新机制，带一点问题重定义。  
为什么能发：把随机 MLM 批成“盲目 token 对齐”，改成 scene graph 的 object、attribute、relation 层级 masking 和推理，证据链很清楚。  
story：从随机遮词，升级为知识引导的细粒度视觉语言推理。

3. **HPL 换衣 ReID**
创新类型：新机制。  
为什么能发：指出 instance-level metric 在换衣下被大类内差异和小类间差异破坏，用主身份 proxy、自动挖子 proxy、子到主 proxy 来重塑特征几何。  
story：换衣 ReID 不是拉近样本对，而是学习身份主簇和衣着子簇的层级关系。

4. **HOH-Net 可见光红外 ReID**
创新类型：新机制。  
为什么能发：不是直接压 VIS 和 IR 特征，而是用高阶结构、细粗图对齐和 middle-feature agent 逐层缓解模态差。  
story：跨模态对齐需要可靠中间表征，不能把两种模态硬拉到一起。

5. **HPRNet 换衣 ReID**
创新类型：工程组合偏新机制。  
为什么能发：批评粗糙换衣增广和重构会丢身份上下文，用 non-local 多尺度感知和 parsing reconstruction 保住局部相关与人体部件。  
story：衣服变化下，身份线索藏在非局部人体上下文里，不在粗暴去衣服里。

6. **IFD 换衣 ReID**
创新类型：新机制。  
为什么能发：把身体区域定义成“身份和衣服耦合区”，用 clothing-masked attention stream 教 RGB 主流关注身份区域，再用 clothing bias diminishing 抑制衣服偏置。  
story：先让无衣服图告诉模型身份在哪里，再让有衣服图别被衣服带偏。

7. **Identity-aware IR diffusion**
创新类型：新数据设定加新机制。  
为什么能发：VI-ReID 数据少，Stable Diffusion 不懂红外且生成不可控；用 LoRA 学红外、LLM text adapter 强化语义、可控生成保证同文本身份一致。  
story：让扩散模型真正理解红外，并生成可用于 ReID 的身份一致跨模态样本。

8. **EAIBC 文本检索**
创新类型：问题重定义加工程机制。  
为什么能发：把文本行人检索失败归因到 color over-reliance，用 RGB、灰度、高频、颜色四分支互学，逼模型均衡利用颜色、纹理、结构。  
story：文本检索不能只认颜色，要挖出颜色之外的全方位身份证据。

9. **InfinitePerson**
创新类型：新数据设定。  
为什么能发：合成 ReID 的 gap 不是只有规模，而是 3D 人物衣服纹理差、采集流程不可复现；用扩散生成 UV 纹理和子监控网络构造可扩展数据。  
story：高质量合成 ReID 要同时真实化人物纹理和监控采集过程。

**反推出来的强候选**

1. **三维可见证据预算的航拍地面 ReID**  
headline：把航拍地面 ReID 从二维 patch 匹配，改写为三维人体表面可见证据分配问题。  
挂资产：aerial-ground 数据、SMPL、pose 热图门控、SOLIDER-Swin。  
区别：HFLAT 只做二维 patch 重要性，HPRNet 和 IFD 依赖 parsing 或 masked image；我们用 SMPL 投影估计每个身体表面在航拍和地面视角下的可见性、尺度和可靠性。  
低成本杀停：不训练，先在 CARGO 或 AG-ReID.v2 上用 SOLIDER token 加 SMPL 可见性权重重算检索；如果错误和可见性、尺度、姿态置信度没有明显相关，且零训练加权不超过噪声线，就杀停。

2. **几何状态层级 proxy 的航拍地面 ReID**  
headline：同一身份不是一个单簇，而是由俯视、平视、背面、低清等几何状态子簇共同组成。  
挂资产：aerial-ground 数据、SMPL、SOLIDER-Swin。  
区别：HPL 的子 proxy 是换衣状态或聚类近似，我们的子 proxy 是由视角、SMPL 可见面、人体投影比例定义的几何状态，问题从换衣类内差异转到跨视角几何类内差异。  
低成本杀停：冻结 SOLIDER 特征，按 SMPL 可见性和框形态给训练样本分几何状态，看同身份跨状态距离是否显著大于同状态距离；再只训一个小投影头和 proxy loss，若没有稳定改善就停。

3. **身份一致的航拍地面几何反事实合成**  
headline：合成数据不追求数量，而是生成同一 SMPL 身份在航拍和地面视角下的成对反事实样本。  
挂资产：SMPL 基建、aerial-ground 设定、SOLIDER-Swin。  
区别：Identity-aware diffusion 解决 RGB 红外生成，InfinitePerson 解决通用合成数据质量；我们专门控制相机高度、俯仰角、低清、遮挡和同一身体几何，目标是航拍地面跨视角不变性。  
低成本杀停：先生成或渲染几十个身份的小集，只训练 adapter 或最后投影层；如果真实 AG 测试没有超过噪声线的提升，或者合成样本和真实样本特征域差过大，就不扩大。

4. **姿态不确定性感知的门控 ReID**  
headline：在航拍低清场景里，姿态不是可靠真值，而是带不确定性的身份证据。  
挂资产：pose 热图门控、SMPL、aerial-ground。  
区别：现有 pose/parsing 方法大多假设辅助信号可靠；我们的机制显式利用 heatmap entropy、SMPL 重投影误差和部位覆盖率来决定何时相信姿态，何时退回 SOLIDER 全局证据。  
低成本杀停：先统计姿态热图熵和检索错误的关系，再做零训练置信度门控；如果高熵样本并不更容易错，或门控没有提升，就杀停。

我会优先押第 1 个，第二优先是第 2 个。第 3 个有论文故事，但工程成本和生成域差风险更高。第 4 个适合作为第 1 个里的关键消融或备选机制，不建议单独硬撑成主线。
tokens used
82,492
清单里实际是 9 篇。我只读了每篇前三页，判断基于摘要和引言。

**逐篇提取**

1. **HFLAT 遮挡 ReID**
创新类型：工程组合偏新机制。  
为什么能发：抓住 Transformer 遮挡方法“patch 等权、背景遮挡未分离、局部细粒度弱”的 gap，用层级 patch 融合、显著性前景分离、局部注意力串证据。  
story：不是所有 patch 都同等重要，遮挡 ReID 要按身份贡献组织局部证据。

2. **HKGR 文本 ReID**
创新类型：新机制，带一点问题重定义。  
为什么能发：把随机 MLM 批成“盲目 token 对齐”，改成 scene graph 的 object、attribute、relation 层级 masking 和推理，证据链很清楚。  
story：从随机遮词，升级为知识引导的细粒度视觉语言推理。

3. **HPL 换衣 ReID**
创新类型：新机制。  
为什么能发：指出 instance-level metric 在换衣下被大类内差异和小类间差异破坏，用主身份 proxy、自动挖子 proxy、子到主 proxy 来重塑特征几何。  
story：换衣 ReID 不是拉近样本对，而是学习身份主簇和衣着子簇的层级关系。

4. **HOH-Net 可见光红外 ReID**
创新类型：新机制。  
为什么能发：不是直接压 VIS 和 IR 特征，而是用高阶结构、细粗图对齐和 middle-feature agent 逐层缓解模态差。  
story：跨模态对齐需要可靠中间表征，不能把两种模态硬拉到一起。

5. **HPRNet 换衣 ReID**
创新类型：工程组合偏新机制。  
为什么能发：批评粗糙换衣增广和重构会丢身份上下文，用 non-local 多尺度感知和 parsing reconstruction 保住局部相关与人体部件。  
story：衣服变化下，身份线索藏在非局部人体上下文里，不在粗暴去衣服里。

6. **IFD 换衣 ReID**
创新类型：新机制。  
为什么能发：把身体区域定义成“身份和衣服耦合区”，用 clothing-masked attention stream 教 RGB 主流关注身份区域，再用 clothing bias diminishing 抑制衣服偏置。  
story：先让无衣服图告诉模型身份在哪里，再让有衣服图别被衣服带偏。

7. **Identity-aware IR diffusion**
创新类型：新数据设定加新机制。  
为什么能发：VI-ReID 数据少，Stable Diffusion 不懂红外且生成不可控；用 LoRA 学红外、LLM text adapter 强化语义、可控生成保证同文本身份一致。  
story：让扩散模型真正理解红外，并生成可用于 ReID 的身份一致跨模态样本。

8. **EAIBC 文本检索**
创新类型：问题重定义加工程机制。  
为什么能发：把文本行人检索失败归因到 color over-reliance，用 RGB、灰度、高频、颜色四分支互学，逼模型均衡利用颜色、纹理、结构。  
story：文本检索不能只认颜色，要挖出颜色之外的全方位身份证据。

9. **InfinitePerson**
创新类型：新数据设定。  
为什么能发：合成 ReID 的 gap 不是只有规模，而是 3D 人物衣服纹理差、采集流程不可复现；用扩散生成 UV 纹理和子监控网络构造可扩展数据。  
story：高质量合成 ReID 要同时真实化人物纹理和监控采集过程。

**反推出来的强候选**

1. **三维可见证据预算的航拍地面 ReID**  
headline：把航拍地面 ReID 从二维 patch 匹配，改写为三维人体表面可见证据分配问题。  
挂资产：aerial-ground 数据、SMPL、pose 热图门控、SOLIDER-Swin。  
区别：HFLAT 只做二维 patch 重要性，HPRNet 和 IFD 依赖 parsing 或 masked image；我们用 SMPL 投影估计每个身体表面在航拍和地面视角下的可见性、尺度和可靠性。  
低成本杀停：不训练，先在 CARGO 或 AG-ReID.v2 上用 SOLIDER token 加 SMPL 可见性权重重算检索；如果错误和可见性、尺度、姿态置信度没有明显相关，且零训练加权不超过噪声线，就杀停。

2. **几何状态层级 proxy 的航拍地面 ReID**  
headline：同一身份不是一个单簇，而是由俯视、平视、背面、低清等几何状态子簇共同组成。  
挂资产：aerial-ground 数据、SMPL、SOLIDER-Swin。  
区别：HPL 的子 proxy 是换衣状态或聚类近似，我们的子 proxy 是由视角、SMPL 可见面、人体投影比例定义的几何状态，问题从换衣类内差异转到跨视角几何类内差异。  
低成本杀停：冻结 SOLIDER 特征，按 SMPL 可见性和框形态给训练样本分几何状态，看同身份跨状态距离是否显著大于同状态距离；再只训一个小投影头和 proxy loss，若没有稳定改善就停。

3. **身份一致的航拍地面几何反事实合成**  
headline：合成数据不追求数量，而是生成同一 SMPL 身份在航拍和地面视角下的成对反事实样本。  
挂资产：SMPL 基建、aerial-ground 设定、SOLIDER-Swin。  
区别：Identity-aware diffusion 解决 RGB 红外生成，InfinitePerson 解决通用合成数据质量；我们专门控制相机高度、俯仰角、低清、遮挡和同一身体几何，目标是航拍地面跨视角不变性。  
低成本杀停：先生成或渲染几十个身份的小集，只训练 adapter 或最后投影层；如果真实 AG 测试没有超过噪声线的提升，或者合成样本和真实样本特征域差过大，就不扩大。

4. **姿态不确定性感知的门控 ReID**  
headline：在航拍低清场景里，姿态不是可靠真值，而是带不确定性的身份证据。  
挂资产：pose 热图门控、SMPL、aerial-ground。  
区别：现有 pose/parsing 方法大多假设辅助信号可靠；我们的机制显式利用 heatmap entropy、SMPL 重投影误差和部位覆盖率来决定何时相信姿态，何时退回 SOLIDER 全局证据。  
低成本杀停：先统计姿态热图熵和检索错误的关系，再做零训练置信度门控；如果高熵样本并不更容易错，或门控没有提升，就杀停。

我会优先押第 1 个，第二优先是第 2 个。第 3 个有论文故事，但工程成本和生成域差风险更高。第 4 个适合作为第 1 个里的关键消融或备选机制，不建议单独硬撑成主线。
