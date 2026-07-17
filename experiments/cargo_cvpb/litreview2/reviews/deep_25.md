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
session id: 019ef836-0744-7841-b768-9ecd882b37ae
--------
user
你是 ReID 创新方法论拆解员 25。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(25)==
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则读目标和状态文件，再逐篇用 `pdftotext -f 1 -l 8` 抽方法部分。读完后如果某篇的“新问题”需要确认先例，我再用网页搜索补背景。
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
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
项目目标确认了：当前阶段就是读文献、做差距分析，不开实验。我现在开始逐篇抽第一页到第八页的正文，重点定位方法章节里作者怎样把观察转成问题定义和机制。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

1

Structured Distilled 3D Gait Fields for
Clothes-Changing Person Re-Identification in Videos
Yubin Wang , Huimin Yu , Member, IEEE, Haoji Hu , Member, IEEE, Yongsheng Gao , Senior Member, IEEE

Abstract—3D gait representation, intrinsically tied to identity,
is of great importance in tackling the challenge of video-based
clothes-changing person re-identification (VCCRe-ID), where individuals’ clothing appearances vary across video clips captured
over an extended duration. Most existing methods derive 3D
gait features from sequential 3D skeletons or parameters of a
predefined 3D human model (e.g. SMPL). However, the sparsity
of 3D skeletons and the low-dimensional nature of 3D human
model parameters not only limit their expressive power and
discriminative ability, but also impede alignment with higherdimensional appearance features. To address these issues, we
propose Structured Distilled 3D Gait Fields (SD-3DGF) to learn
better aligned, more expressive, and discriminative 3D gait
features for VCCRe-ID. SD-3DGF defines distilled feature fields
that evolve over time, allowing the simulation of 3D gait dynamics
from temporal 2D descriptors. Specifically, it densely transfers
frame-level shape knowledge from structured 3D space (i.e., 3D
mesh surface) to the 2D image domain via 3D-2D knowledge
distillation, and models local spatiotemporal correlations among
the distilled features to extract 3D gait features. This learning
paradigm makes 3D gait features more compatible with appearance features, as they are naturally aligned in the spatiotemporal
domain. To fully exploit this potential, we develop a novel
Modality-Representative Interactive Encoding (MRIE) module,
which facilitates efficient gait-appearance interaction and joint
encoding via global and local representative tokens. Experiments
on both VCCRe-ID and standard video-based Re-ID benchmarks
demonstrate the effectiveness of our method. The code is available
at https://github.com/yubinwang2021/SD-3DGF.
Index Terms—Video-based clothes-changing person reidentification, structured distilled 3D gait fields, 3D gait features,
modality-representative interactive encoding.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) involves re-identifying
a particular person across a series of disjoint cameras,
which plays a pivotal role in numerous practical applications
including intelligent surveillance, the retail industry, and security authentication [1], [2], [3]. After years of focused research,
current short-term Re-ID methods [4], [5], [6] have achieved
remarkable performance on public benchmarks, surpassing
human-level recognition, and have also exhibited promising
performance when extended to cross-modal retrieval, such
as leveraging additional text modalities [7], [8] or infrared

This work was supported in part by the Research Project of ZJULeague Research and Development Center, Zhejiang Laboratory under Grant
2019KD0AB01. (Corresponding authors: Huimin Yu; Yongsheng Gao.)
Yubin Wang, Huimin Yu and Haoji Hu are with the College of Information
Science and Electronic Engineering, Zhejiang University, Hangzhou, China (email: zjuwyb2021@zju.edu.cn; yhm2005@zju.edu.cn; haoji hu@zju.edu.cn).
Yongsheng Gao is with the Institute for Integrated and Intelligent Systems, Griffith University, Nathan, QLD4111, Australia (e-mail:
yongsheng.gao@griffith.edu.au).

3D vertex
embeddings

(a)

(b)

(c)

(d)
Distill

(e)
Fig. 1. Examples of different auxiliary modalities for gait modeling in
VCCRe-ID. (a) the input video tracklet, sampled from the CCVID dataset;
(b) 2D keypoints and silhouettes; (c) part segmentation maps; (d) SMPL
parameters; (e) dense correspondence embeddings, from the proposed SD3DGF framework.

modalities [9], [10]. However, these methods assume that
pedestrians appear briefly within the camera network, with
their clothing remaining unchanged. In real-world scenarios,
pedestrians may be re-captured by the camera after a long
period (a day, a month, or even a year). In such cases,
their clothing is likely altered due to various factors such as
temperature changes and shifts in personal preferences [11],
[12]. For this reason, short-term Re-ID methods, which overrely on clothing appearances, often fail in long-term settings.
To tackle this issue, researchers place increasing emphasis on clothes-changing person re-identification (CCRe-ID),
where the key challenge lies in extracting clothing-invariant
identity features [13]. A prominent line of work [14], [15],
[16] explores 2D and 3D body shape as soft biometrics, as
they provide personalized information [17]. However, both 2D
and 3D shape representations remain susceptible to clothing
variations. For example, an individual’s perceived body proportions can differ drastically when wearing a T-shirt versus
a thick down jacket.
To address this limitation, gait emerges as a promising
biometric cue, owing to its spatiotemporal dynamics and robustness to clothing variations [18], [19]. Using gait as an auxiliary modality for extracting clothing-invariant features fosters
growing interest in video-based clothes-changing person re-

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

identification (VCCRe-ID) [20], [21], [22], [23]. As shown
in Fig. 1(b) and (c), some VCCRe-ID approaches [20], [21]
learn 2D gait representations using sequential 2D keypoints,
silhouette sequences and part segmentation maps. However,
the learned 2D gait features may suffer from appearance ambiguity caused by viewpoint variations due to the lack of spatial
alignment. Another line of work focuses on training the gait
stream using 3D auxiliary modalities [22], [23]. For instance,
ASGL [22] extracts gait embeddings through spatiotemporal
graphs of 3D skeletons. In contrast, CMCL [23], as illustrated
in Fig. 1(d), models 3D gait dynamics using sequential SMPL
parameters [24]. However, both 3D skeletons and 3D human
model parameters are inherently low-dimensional and sparse,
exhibiting limited identity discriminability and poor distribution alignment with the high-dimensional appearance features.
Attempts to expand their dimensionality, such as mapping
them to the RGB feature space through a spatial transformation
network [18], still produce highly sparse features that are at
risk of losing key semantic information, thereby compromising
the reliability of gait-appearance cross-modal interaction.
In this paper, we propose Structured Distilled 3D Gait Fields
(SD-3DGF), a new 3D gait representation learning paradigm
for VCCRe-ID. As shown in Fig. 1(e), different from prior
3D gait-based methods, SD-3DGF performs dense, frameby-frame transfer of 3D shape knowledge from structured
surface meshes to the image domain via 3D-2D distillation,
with the distilled frame-level features forming a structured
3D gait field. More specifically, based on the surface-based
geodesic distances, we pre-train vertex embeddings for the
canonical SMPL mesh, and attach them to the reconstructed
sequential body meshes. Then, using the estimated camera
parameters, we project both the 3D vertices and randomly
sampled surface points onto the image plane frame by frame,
generating dense 2D-3D paired data to guide the training
of a lightweight frame-level feature extractor. The features
extracted by this extractor are spatially aligned with the
input frame, i.e., each pixel is mapped to a dedicated pixel
embedding. Theoretically, pixel embeddings associated with
the same 3D vertex across frames are inherently consistent,
and those corresponding to the neighboring 3D points should
also be similar. Thus, local spatiotemporal correlations can be
naturally modeled via similarity learning [25], facilitating the
extraction of 3D gait features that encode discriminative local
patterns. The proposed 3D gait learning paradigm represents
gait as a spatiotemporal field aligned with the video sequence,
enabling seamless interaction with clothing-invariant appearance features. Furthermore, the learned 3D gait features are
robust to viewpoint variations, as the extracted frame-level
features capture rich alignment information with the canonical
3D space.
To facilitate complementary information sharing between
gait and appearance modalities, we propose a novel ModalityRepresentative Interactive Encoding (MRIE) module. MRIE
first employs the Dual-Granularity Representative Token Generation (DGRT) strategy to produce, for each modality, a
global representative token summarizing holistic identity information and a set of local representative tokens preserving
region-level discriminative details. This process also reduces

2

Canonical SMPL Mesh

Canonical SMPL Mesh

3D vertex
coordinates

...

3D vertex
embeddings

𝓛𝒗𝒆𝒑

Graph Convolutions

Fig. 2. 3D vertex embeddings pre-training pipeline. The graph convolutional
network takes the canonical vertex coordinates as input and produces the 3D
vertex embeddings, constrained by geodesic distances and tangent directions
on the mesh surface.

the spatial dimensions of tokens, which significantly reduces
the computational cost. The Gait-Appearance Joint Encoding
(GAJE) submodule then models cross-modality interactions
between the dual-granularity tokens from the two modalities,
and outputs the final identity feature via adaptive integration.
Our contributions are summarized as follows:
• We propose Structured Distilled 3D Gait Fields (SD3DGF), a new 3D gait learning scheme that enables the
simulation of 3D gait dynamics from distilled feature
fields. To the best of our knowledge, this is the first
study to mine 3D gait features for VCCRe-ID from a
perspective of 3D-2D distillation.
• For the training of the proposed 3D gait fields, we develop
a comprehensive pipeline for generating dense paired 2D3D data, comprising 3D vertex embedding pretraining,
3D body motion capture, and dense 3D-2D projection
with both vertices and sampled surface points.
• We propose a novel Modality-Representative Interactive
Encoding (MRIE) module, which introduces global and
local representative tokens to enable efficient interaction
and joint encoding of 3D gait and clothing-invariant
appearance features.
• Extensive experiments on four video-based clotheschanging Re-ID benchmarks (VCCR, CCVID, CCVR, and CCV-S) and three standard video-based Re-ID
benchmarks (MARS, LS-VID, and iLIDS-VID) show that
our method not only delivers strong performance under
clothes-changing settings, but also maintains robustness
under standard video-based Re-ID settings.
II. R ELATED W ORKS
A. Video-Based Person Re-ID
Video-based person Re-ID (VRe-ID) has been extensively
investigated in recent years. The mainstream video-based ReID methods [26], [27], [28], [29], [30], [31], [32], [33], [34]
can be broadly categorized into two types: frame-level feature
aggregation methods and sequence-level spatiotemporal modeling methods. Frame-level feature aggregation methods [26],
[27], [28], [29], [30] first extract frame-level features, and
then aggregate them along the temporal dimension to obtain
a video-level representation. Zheng et al. [26] adopt a pooling
strategy to fuse frame-level features. Hou et al. [27] and Bai

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

et al. [28] leverage attention mechanisms to focus on salient
regions in frame-level features and capture temporal relations.
Yu et al. [29] and Su et al. [30] design a CLIP-based video
Re-ID framework to enhance the multi-modal understanding of
frame-level features. Sequence-level spatiotemporal modeling
methods [31], [32], [33] extract unified representations from
video sequences by jointly modeling spatial and temporal
information. Gu et al. [31] employ 3D Convolutional Neural
Networks (3D-CNNs) to uniformly encode spatiotemporal
information. Li et al. [33] leverage a dual-stream network
with I3D [35] as the backbone to learn spatiotemporal motion
and appearance features. Tang et al. [32] propose a multistage Transformer network to learn attribute- and identityrelevant features from raw video sequences. Leng et al. [34]
mine structured spatiotemporal relations in both Euclidean and
hyperbolic spaces using dynamic prompt graphs.
B. Clothes-Changing Person Re-ID from a Single Image
The key challenge of image-based clothes-changing person
Re-ID (CCRe-ID) is to learn clothing-invariant identity features from a single image. The existing literature mainly falls
into two categories: disentanglement-based methods [36], [37],
[38], [39], [40] and biometrics-based methods [15], [41], [42],
[43]. Disentanglement-based methods [36], [37], [38], [39],
[40] emphasize disentangling identity-related appearance features from clothing-related ones, without relying on additional
modalities. Xu et al. [36] and Xiong et al. [37] decouple
identity and clothing features through reconstructing person
images or parsing maps. Wang et al. [38] decouple attribute
features through clustering and recompose them for identity
modeling. Li et al. [39] and Liang et al. [40] use CLIP [44]
to disentangle identity-related and identity-irrelevant feature
subspaces guided by text descriptions.
Biometrics-based methods [15], [41], [42], [43] use auxiliary modalities to mine discriminative biometric cues. Shu
et al. [41] randomly replace clothing pixels guided by part
segmentation to learn clothing-invariant features. Jin et al. [42]
predict 2D gait dynamics from a single image to regulate the
appearance features. Chen [15] treat the 10-dimensional SMPL
shape parameters as 3D shape features and fuse them with
appearance features. Liu et al. [43] perform 3D clothed human
reconstruction to learn discriminative shape latent codes.
C. Clothes-Changing Person Re-ID in Videos
Video-based clothes-changing person Re-ID (VCCRe-ID)
exploits rich spatiotemporal cues in videos to mine clothinginvariant identity features. Gu et al. [45] introduce a reliable
and simple VCCRe-ID baseline, which achieves competitive
benchmark performance using only the RGB modality. The
mainstream VCCRe-ID approaches [20], [21], [23], [46], [47],
[48] place great emphasis on learning video-level canonical
shape or gait features as soft biometric cues. Chen et al. [20]
and Lu et al. [21] present a 2D gait and appearance fusion
network using keypoint heatmaps or silhouettes as additional
input. Wang et al. [46] propose a graph-based pipeline for
joint encoding of 2D gait and appearance features. Nguyen et
al. [22] adopt sequential 3D skeletons for gait modeling. Han

3

et al. [47] and Nguyen et al. [48] derive canonical 3D shape
features at the video level through temporal aggregation of
10-dimensional SMPL shape parameters. However, the lowdimensional SMPL shape parameters carry very limited information and exhibit weak discriminative capability. Moreover,
their feature space differs substantially from that of RGB
features. Nguyen et al. [23] take a further step by utilizing
SMPL parameters to guide the regulation of silhouette feature
maps via a 3D transformation network. However, directly mapping SMPL parameters into a high-dimensional visual space
produces sparse features and loses key 3D prior knowledge.
D. Dense Correspondence Embeddings
A core technique of our method is the learning of dense correspondence embeddings. Güler et al. [49] design a cascaded
network to estimate per-pixel UV coordinates and generate
IUV maps, which bridges the 2D and 3D domain. Wang
et al. [50] introduce a per-pixel vertex classification loss
for extracting pixel-wise surface embeddings. Liu et al. [51]
facilitate indoor parsing by distilling 3D scene embeddings
onto the image plane. Ianina et al. [52] explore scalable
correspondence embeddings across different 3D meshes. Hou
et al. [53] devise a Transformer-based refinement network
to generate fine-grained full-body correspondence embedding
maps. Hönig et al. [54] leverage diffusion models to learn
2D-3D correspondence mappings between images and point
clouds. Yang et al. [55] propose to learn video-level correspondence embeddings for animating deformable 3D objects.
Different from the prior works, we propose a novel method
using pre-trained 3D vertex embeddings and 3D-2D distillation
for learning dense correspondence embeddings.
III. M ETHODOLOGY
This section presents the overall architecture of the proposed method, detailing the SD-3DGF learning pipeline, the
MRIE module for efficient gait-appearance interaction, and the
formulation of training losses.
A. Overview
Fig. 3(a) depicts the overall architecture of our method and
illustrates the relationships among its modules. It adopts a
dual-stream structure comprising an appearance stream SA , a
gait stream SG , and the MRIE module. SA and SG operate in
parallel, while the MRIE module receives features from both
streams and yields the final identity representation. Given a
video tracklet T = {It | t = 1, 2, . . . , T } of a pedestrian
with T sampled frames, the appearance stream SA seeks
to extract clothing-invariant appearance features from T . It
employs a ResNet-50 backbone [56], using the first three
residual blocks to extract frame-level appearance features
A = {At | t = 1, 2, . . . , T }. The resulting feature set A is then
concatenated along the temporal dimension to form the initial
video-level appearance feature XA ∈ RT ×h×w×c , which is
further regularized by an orthogonal loss Lorg to achieve
clothing desensitization. The 3D gait stream SG , on the other
hand, is designed to learn expressive and discriminative 3D

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

4

Local Similarity Learning (LSL)

Video Tracklet 𝒯

Appearance Stream 𝓢𝑨

1×1×1 Conv

Cubic Window
𝑿𝑨

c

Cos. Sim.

𝒄𝜑

𝓐

ResNet-50

c
𝑿𝒅 1×1×1 Conv

Neighbors

𝓕’
𝓛𝒐𝒓𝒈

𝑿𝑮

(b)
Dual-Granularity Representative Token Generation (DGRT)
MHSA

𝓛𝒊𝒅

...

𝑺𝒍𝒐𝒄
𝑨

...

𝑺𝒍𝒐𝒄
𝑮

Patch Tokens

... 𝑻𝒑𝒂𝒕
𝑵𝒑

...

𝑹𝒍𝒐𝒄
𝟏

... 𝑹𝒍𝒐𝒄
𝑵𝒑

...

c
𝑹𝒈𝒍𝒃

𝑹𝒈𝒍𝒃

(c)

GAJE

𝓛𝒕𝒓𝒊

𝒈𝒍𝒃

𝑹𝑮

Patch Tokens

𝒈𝒍𝒃
𝑹𝑨

𝓛𝒈𝒇

𝒈𝒍𝒃
𝑹𝑨

෩
𝑺𝒍𝒐𝒄
𝑨

𝒈𝒍𝒃

MHCA

𝑹𝑮

c
LSL

𝑺𝒍𝒐𝒄
𝑮
MHCA

𝑿𝑮

ResNet-50

’

Gait Stream 𝓢𝑮

෩
𝑺𝒍𝒐𝒄
𝑮

𝓕

𝒈𝒍𝒃

𝑹𝑮

෩
𝑺𝒍𝒐𝒄
𝑨

෩
𝑺𝒍𝒐𝒄
𝑮

𝒈𝒍𝒃
෩
𝑹𝑨

AW ഥ
AW
ഥ 𝒍𝒐𝒄
𝑹𝒍𝒐𝒄
𝑹
𝑨
𝑮

MHCA

𝑺𝒍𝒐𝒄
𝑨

DGRT

MHCA

Gait-Appearance Joint Encoding (GAJE)

...

3D Patches

𝒇𝒊𝒅

𝒑𝒂𝒕

MHCA MHCA

...
𝑿𝑮

...

Flatten

Partition

Correspondence
Embeddings

DGRT

𝑻𝟏

MHCA

3D Patches

𝒈𝒍𝒃

𝑹𝑨

MHCA

...
𝑿𝑨

MRIE

Flatten

...

Partition

𝑺𝒍𝒐𝒄

MHSA

...

UNet-Based
Extractor
𝒇𝜽

𝑺𝒍𝒐𝒄

AW
𝒇𝒊𝒅
𝒈𝒍𝒃
෩
𝑹𝑮

Query
Key/Value

c Concatenation
Point-wise Add

(a)

(d)

Fig. 3. Overview of the proposed SD-3DGF. (a) The network includes an appearance stream and a gait stream, where the gait stream enhances local patterns
via LSL. The two streams interact and fuse to form the final identity representation via the MRIE module. (b) Local Similarity Learning (LSL) captures local
patterns by measuring the similarity between point-wise features and their neighbors. (c) Generating dual-granularity representative tokens via hierarchical
multi-head self-attention (MHSA). (d) Gait-Appearance Joint Encoding (GAJE) performs efficient cross-modal interaction via local and global representative
tokens, and generates the final identity representation through adaptive weighting (AW).

gait representations. It first employs a lightweight U-Netbased architecture to extract frame-level distilled features Z =
{Zt | t = 1, 2, . . . , T }, which approximate the distribution of
3D vertex embeddings and serve to construct the structured
distilled 3D gait field F. Subsequently, F is transformed by
a separate ResNet-50 backbone for spatiotemporal alignment
with the video-level appearance features XA . A local similarity
learning (LSL) mechanism is further employed to capture local
spatiotemporal correlations in the transformed 3D gait field,
yielding the 3D gait features XG .
Acting as a bridge between the two streams, the ModalityRepresentative Interactive Encoding (MRIE) module takes
video-level appearance features XA and 3D gait features XG
as input, and employs the Dual-Granularity Representative
Token Generation (DGRT) strategy to produce a global representative token Rglb and a set of local representative tokens
S loc for each modality. The Gait-Appearance Joint Encoding
(GAJE) submodule then efficiently models interactions between the representative tokens of both modalities, producing
the unified identity representation fid , which serves as the final
output of our model.
The following subsections provide a detailed exposition of
the proposed 3D gait representation, the MRIE module, and
the formulation of the training losses.
B. Structured Distilled 3D Gait Fields
Unlike prior methods that directly learn 3D gait features in
the sparse 3D space, we instead model them as a distilled 3D
gait field, which provides a dense representation. We further
discuss how to train the 3D gait fields and extract 3D gait

features with enhanced local awareness via local similarity
learning. To enable fair comparison with existing works, we
adopt the canonical SMPL model [24] with 6,890 vertices and
13,776 faces as the standard mesh representation.
1) Pre-training 3D Vertex Embeddings: Given the canonical SMPL model, we model it as a graph G = (V, A),
where V = {vi | i = 1, 2, . . . , N } ∈ RN is the vertex
set, A = {aij | i, j = 1, 2, . . . , N, i ̸= j} ∈ RN ×N is the
adjacency matrix, and N = 6890 represents the total number
of vertices. As illustrated in Fig. 2, graph residual convolutions are adopted to pre-train vertex embeddings, as they are
naturally well-suited for mesh models with explicit topology.
The detailed formulation of graph residual convolutions is as
follows:
X
(l+1)
(l)
(l)
(l)
(l)
xi
= σ(
w(xi , xj ) · xj ) + xi ,
(1)
j∈N (i)
(l)

(l+1)

where xi and xi
denote the features of vertex vi at
(l)
layers l and l+1, respectively, xj is the feature of a neighboring vertex vj at layer l, N (i) denotes the neighborhood
of vertex vi , w(·, ·) represents the weight between two graph
nodes, and σ(·) is the ReLU activation function [57].
The graph convolutional network (GCN) uses canonical
vertex coordinates as initial node features, consisting of four
graph residual layers, with graph normalization [58] applied
to all but the final layer to accelerate convergence. The node
features from the final layer are used as vertex embeddings.
We expect the embeddings of any two vertices to reflect
their geodesic distance on the mesh. To this end, we first
approximate geodesic distances by computing the shortest

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

spatiotemporal video domain. Formally, the structured distilled
3D gait field F is defined as:

Randomly
Sampled

3D Mesh
Recovery

Attach 3D
Vertex
Embeddings

5

𝒔

Project

F[t, x, y] = fθ (It )[x, y] ∈ RD ,
𝒗𝟏
𝒔

𝒗𝟐

𝒗𝟑

Fig. 4. Pipeline for generating dense 2D-3D paired data for a single frame.
The vertices of the recovered 3D human mesh, along with randomly sampled
surface points, are projected onto the image plane using camera parameters.
The vertex embeddings and interpolated surface point embeddings are used
as ground-truth for the corresponding pixel embeddings.

paths along the mesh surface between vertices using Dijkstra’s
algorithm, and compute the tangent vectors for each vertex
using the Trimesh 3D library [59]. We then normalize the
calculated geodesic distances to the range of (0, 1] via minmax scaling, and enforce the L2 distances between vertex
embeddings to match the corresponding geodesic distances:
d(evi , evj )
√
− g(vi , vj ) = 0,
(2)
D
where v and ev represent the vertex and its embedding
√
vector, D is the dimension of vertex embeddings, D is
a normalization factor to prevent distance imbalance from
dimensionality changes, d(·, ·) indicates the L2 distance, and
g(·, ·) defines the geodesic distance.
To encode the directional information of the structured 3D
mesh into the pre-trained vertex embeddings, we impose a
constraint ensuring that the angle between any two vertex embeddings matches the angle between their respective tangent
vectors on the mesh surface:
θemb (evi , evj ) − θtan (vi , vj ) = 0,

(3)

where θemb and θtan refer to the angles between the embedding vectors and the tangent vectors, respectively.
The training losses for the above two constraints will be
discussed in Section III-D. With these constraints, the pretrained vertex embeddings effectively preserve key inter-vetex
correlations and prior structural knowledge.
2) Formulating the Structured Distilled 3D Gait Fields:
The pre-trained 3D vertex embeddings E essentially define a
neural field:
fψ : R3 → RD ,
(4)
where ψ denotes the parameters of GCN, and D indicates the
embedding dimension.
A straightforward idea is to concatenate the embeddings
produced by fψ with the 3D vertex coordinates of the sequential 3D meshes, and then extract spatiotemporal point
cloud features as 3D gait features. However, the resulting
computational cost and inference time would be prohibitive.
Alternatively, we propose to build a structured distilled 3D
gait field (SD-3DGF) F that models 3D gait dynamics solely
from the frame-level features extracted from the video tracklet
T . Specifically, our goal is to transfer knowledge from 3D
vertex embeddings, which encode mesh-structured knowledge,
to frame-level features to learn 3D gait representations in the

(5)

where (x, y) refers to the pixel coordinates, t represents the
timestamp, fθ is a frame-level feature extractor, It ∈ RH×W ×3
denotes the t-th frame with height H and width W , and D
indicates the embedding dimension of 3D vertices.
The role of fθ is to extract pixel embeddings Zi = fθ (Ii ) ∈
RH×W ×D from frame Ii , which are distilled from the pretrained vertex embeddings E.
3) Constructing Dense Training Data and Spatial/Temporal
Constraints for SD-3DGF: We adopt a lightweight UNetbased architecture Eff-UNet [60] as the backbone of fθ . As
shown in Fig. 4, to prepare the data required for training
fθ , we begin by employing the body motion capture method
FrankMocap [61] to recover SMPL mesh sequences from the
input videos. The pre-trained vertex embeddings are assigned
to the mesh vertices and subsequently projected onto each
video frame using the estimated camera parameters. However,
this can only generate sparse paired 2D-3D training data, covering 15%-25% of the foreground pixels. Therefore, beyond
using mesh vertices alone, we randomly sample additional
points on each triangular face of the structured SMPL mesh,
and compute their embeddings via barycentric interpolation of
the triangle’s vertex embeddings. Formally, this process can be
expressed as:
es = b1 ev1 + b2 ev2 + b3 ev3 ,

b1 + b2 + b3 = 1,

(6)

where v1 , v2 , v3 denote the three vertices of the triangular face, and s is a sampled surface point on the face.
ev1 , ev2 , ev3 are the corresponding vertex embeddings, and
es is the embedding of s. (b1 , b2 , b3 ) are the barycentric
coordinates of s with respect to the triangle (v1 , v2 , v3 ), with
b1 = A(v2 , v3 , s)/A(v1 , v2 , v3 ), where A(·, ·, ·) denotes the
area of a triangle, and b2 and b3 are defined analogously.
These randomly sampled surface points are projected onto
the image plane following the same projection process as the
mesh vertices, thereby constructing dense 2D-3D paired data.
However, multiple mesh vertices or surface points may be
projected to the same pixel. To handle this, we adopt a zbuffer strategy that keeps only the vertex or surface point with
the smallest depth for each pixel, ensuring that the pixel is
assigned to the 3D point closest to the camera. Beyond the
dense 2D-3D paired data, we can further obtain pixel-pixel
correspondences (pi1 ,t1 , pi2 ,t2 ) by tracing the same 3D vertex
v across frames. Here, pi,t = (xi , yi , t) denotes the i-th pixel
in frame It with coordinates (xi , yi ). The pixels pi1 ,t1 and
pi2 ,t2 from frames It1 and It2 , respectively, correspond to the
same 3D vertex v.
Next, we impose three spatial or temporal constraints to
supervise the training of fθ for SD-3DGF:
First, for any frame t, the embedding of a foreground pixel
pi,t = (xi , yi , t) should align with that of its associated 3D
point qi (either a vertex v or a sampled surface point s):
Zt [xi , yi ] − eqi = ⃗0,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

(7)

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

6

where Zt = fθ (It ) ∈ RH×W ×D denotes the embedding map
of frame It produced by fθ , and eqi is the embedding vector
of the 3D point qi .
Second, for any frame t, the L2 distance between the
embeddings of two foreground pixels pi,t and pj,t (i ̸= j)
should match the L2 distance between the embeddings of their
corresponding 3D points qi and qj (either a vertex v or a
sampled surface point s):

neighborhood of each center point, resulting in 3×3×3−1 =
26 neighbors (see Fig. 3 (b)). This yields a channel dimension
of K = 26 for Xd . Additionally, cosine similarity is adopted as
the similarity function ϕ(·, ·) due to its parameter-free nature
and numerical stability.
To fuse the local similarity descriptor Xd with the gait
feature tensor F ′ , we first use a 1 × 1 × 1 3D convolution
to increase the channel dimension of Xd :

d(Zt [xi , yi ], Zt [xj , yj ]) − d(eqi , eqj ) = 0,

Xd′ = Conv3D1×1×1 [Xd ] ∈ RT ×h×w×c ,

(8)

where d(·, ·) denotes the L2 distance, and eqi , eqj are the
embeddings of the 3D points qi , qj , correspondingly.
Third, for any two distinct frames It1 and It2 (t1 ̸= t2 ), if a
foreground pixel pi,t1 from It1 and another pj,t2 from It2 map
to the same 3D vertex v (v is not a sampled surface point),
their embeddings should remain consistent:
Zt1 [xi , yi ] − Zt2 [xj , yj ] = ⃗0.

(9)

Focusing only on foreground pixels, we adopt an off-theshelf instance segmentation model, SOLOv2 [62], to generate
pedestrian masks. These masks remove background pixel embeddings from the distilled features Z = {Zt | t = 1, . . . , T },
where background pixels are set to zero and excluded from
loss computation. The structured distilled 3D gait field (SD3DGF) F ∈ RT ×H×W ×D is obtained by concatenating
Z along the temporal dimension. Consequently, F is fully
aligned with the input video in both spatial and temporal
dimensions, thus enabling strong compatibility between the
derived 3D gait features and video-level appearance features.
4) Modeling 3D Gait Features via Local Similarity: Fundamentally, the 3D gait field F consists of distilled frame-level
features. To learn discriminative representations, it is essential
to further model the underlying spatiotemporal correlations
within F. As the training of F enforces that similarities
between pixel embeddings reflect the geodesic distances of
their corresponding points on the 3D mesh, local similarity
learning (LSL) naturally serves as an effective choice for
capturing local spatiotemporal relations.
Considering the 3D gait field F ∈ RT ×H×W ×D as a
spatiotemporal feature tensor, we first set the input channel
of a ResNet-50 backbone to D and use its first three residual
blocks to transform the spatial resolution, producing a gait
feature tensor F ′ ∈ RT ×h×w×c with the same dimensionality
as the appearance features XA . Then, we define a local
similarity descriptor Xd ∈ RT ×h×w×K , where each element
Xd [t, x, y, k] measures the similarity between the central feature F ′ [t, x, y] and its k-th neighboring point. Here, K denotes
the total number of neighbors considered in the local similarity
computation. Specifically, the descriptor is formulated as:
Xd [t, x, y, k] = ϕ(F ′ [t, x, y], F ′ [t + δtk , x + δxk , y + δyk ]),
(10)
where ϕ(·, ·) denotes a similarity function, and (δtk , δxk , δyk )
represents the relative offset of the k-th neighboring point with
respect to the center point in the spatiotemporal feature space.
As a pairwise feature, the local similarity descriptor Xd
crucially depends on how the neighborhood is defined. We
use a cubic window with an edge length of 3 to determine the

′

(11)

where Xd′ denotes the local similarity descriptor after channel
expansion, c′ is set to 256, and Conv3D1×1×1 represents 1 ×
1 × 1 3D convolution operation.
Then, we concatenate Xd′ and F ′ along the channel dimension, and apply another 1 × 1 × 1 3D convolution for effective
feature integration, while ensuring that the generated 3D gait
features XG have the same channel dimension as F ′ :
XG = Conv3D1×1×1 [Xd′ , F ′ ] ∈ RT ×h×w×c ,

(12)

where [·, ·] indicates the concatenation of feature maps along
the channel dimension.
C. Modality Representative Interactive Encoding
For the appearance features XA and 3D gait features
XG , we seek to employ multi-head self-attention (MHSA) to
enhance intra-modal features, and multi-head cross-attention
(MHCA) to share information across modalities. However, full
space-time attention over all tokens is computationally expensive. Inspired by recent advances [63], [64] that hierarchically
reduce token size to accelerate self-attention, we propose
learning dual-granularity representative tokens with compact
spatial dimensions for both appearance and gait modalities,
enabling more efficient computation of MHSA and MHCA.
1) Generating Global and Local Representative Tokens:
Given the video-level appearance features XA ∈ RT ×h×w×c
and 3D gait features XG ∈ RT ×h×w×c , we first divide the
feature maps of each modality into a set of equally sized,
non-overlapping 3D patches. With a patch size of S = t′ ×
h′ × w′ , this partitioning produces Np = Tt′ × hh′ × ww′ =
T ×h×w
3D patches. Each 3D patch is then flattened along
S
the spatiotemporal dimensions and treated as a patch token,
denoted as Tipat ∈ RS×C , where i ∈ {1, 2, . . . , Np }.
To capture information at different levels of granularity and
achieve faster computation, we introduce a Dual-Granularity
Representative Token Generation (DGRT) strategy, as illustrated in Fig. 3(c). Specifically, a learnable local representative token Riloc ∈ R1×C is assigned to each patch token
Tipat . We then apply multi-head self-attention (MHSA) on the
concatenation of Tipat and Riloc , which aggregates key local
information into each local representative token Riloc :
T̂ipat = Concat(Tipat , Riloc ),

(13)

T̂ipat = MHSA(LN (T̂ipat )) + T̂ipat ,

(14)

where T̂ipat ∈ R(S+1)×C , Concat(·) represents the concatenation operation, MHSA(·) denotes the multi-head selfattention, and LN (·) denotes the layer normalization.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

Subsequently, all updated local representative tokens Riloc
are gathered to form the local representative token set S loc for
the current modality:
loc
S loc = {R1loc , R2loc , . . . , RN
}.
p

(15)

Furthermore, we concatenate all local representative tokens
in S loc and attach a learnable global representative token Rglb .
A second multi-head self-attention operation is then performed
to aggregate information from all local representative tokens
into the global one:
loc
R̂ = Concat(R1loc , R2loc , . . . , RN
, Rglb ),
p

(16)

R̂ = MHSA(LN (R̂)) + R̂,

(17)

where R̂ ∈ R(Np +1)×C , and Np denotes the number of local
representative tokens.
In this step, MHSA(·) is applied solely to summarize overall
modality information into Rglb , while the local representative
tokens in S loc are not updated with the latest results in R̂,
thereby preserving their critical local information.
Using the proposed dual-granularity token generation strategy, we obtain modality-specific global representative tokens
glb
glb
loc
loc
)
, SA
(RG
, RA
) and local representative token sets (SG
for both appearance and gait modalities.
2) Gait-Appearance Joint Encoding: As shown in Fig. 3(d),
we utilize reduced-size global and local representative tokens
glb
glb
loc
loc
)) for efficient joint encoding
, SA
(i.e., (RG
, RA
) and (SG
of gait and appearance modalities. Specifically, the multihead cross-attention (MHCA) mechanism is introduced to
perform cross-modal interaction between the dual-granularity
representative tokens. First, the local representative tokens of
each modality are updated via local-local and local-global
cross-modal interactions, as formulated below:
glb
loc
loc
loc
loc
loc
S̃A
= MHCA(LN (SA
), LN (SG
)) + MHCA(LN (SA
), LN (RG
)) + SA
,

(18)
glb
loc
loc
loc
loc
loc
), LN (RA
)) + SG
,
)) + MHCA(LN (SG
= MHCA(LN (SG
), LN (SA
S̃G

(19)
loc
loc
denote the updated local representative toand S̃A
where S̃G
ken sets of the 3D gait and appearance modalities, respectively,
and MHCA(·, ·) denotes multi-head cross-attention, with the
first argument specifying the source of queries and the second
specifying the source of keys and values.
Next, the global representative tokens are updated via
global-global and global-local cross-modal interactions, formulated as:
glb
glb
glb
glb
glb
loc
R̃A
= MHCA(LN (RA
), LN (RG
)) + MHCA(LN (RA
), LN (S̃G
)) + RA
,

(20)
glb
glb
glb
glb
glb
loc
R̃G
= MHCA(LN (RG
), LN (RA
)) + MHCA(LN (RG
), LN (S̃A
)) + RG
,

(21)
glb
glb
where R̃G
and R̃A
denote the updated global representative
tokens of the 3D gait and appearance modalities, respectively.
glb
glb
loc
loc
Finally, S̃G
, S̃A
, R̃G
, and R̃A
are adaptively fused
to form the identity representation. For each modality, the
local representative tokens are first adaptively aggregated via
attention:
loc
R̄m
=

Np
X
i=1

loc
αim R̃m,i
,

m ∈ {A, G},

(22)

7

loc
where m denotes the modality, R̄m
∈ R1×C is the aggregated
loc
local representation, R̃m,i is the i-th local representative token
loc
loc
in S̃m
, and αim is the learnable attention weight for R̃m,i
.
The final identity representation fid is then obtained by
glb
glb
loc
loc
adaptively weighting R̄A
, R̄G
, R̃A
and R̃G
:
glb
glb
loc
loc
fid = γ1 R̄A
+ γ2 R̄G
+ γ3 R̃A
+ γ4 R̃G
,

(23)

where γ1 , γ2 , γ3 , γ4 are learnable attention weights.

D. Loss Function
We present a detailed description of all loss functions
employed in our framework, including the vertex embedding
pre-training loss, the 3D gait field loss, the Re-ID loss, and
the overall training objective.
1) Vertex Embedding Pre-training Loss: As discussed in
Section III-B1, we enforce that the distances between the
vertex embeddings produced by the GCN match the corresponding geodesic distances on the mesh surface. The geodesic
loss Lgeo is defined as:
Lgeo =

1 X d(evi , evj )
( √
− g(vi , vj ))2 ,
|V|2
D

(24)

i,j∈V

where V denotes the set of vertices, evi and evj are the
embeddings of vertices vi and vj , d(·, ·) computes the L2
distance between two vertex embeddings,
D is the dimension
√
of the vertex embeddings, D serves as a normalization
factor, and g(·, ·) denotes the geodesic distance between two
vertices.
In addition, to preserve the directional information of the 3D
mesh, we constrain the angle between two vertex embeddings
to match the angle between their corresponding tangent vectors
on the mesh surface. The directional loss Ldir is defined as:
Ldir =

1 X
(θemb (evi , evj ) − θtan (vi , vj ))2 ,
|V|2

(25)

i,j∈V

where θemb and θtan denote the angles formed by the embedding vectors and the tangent vectors, respectively.
Thus, the vertex embedding pre-training loss Lvep for
training the GCN is given by:
Lvep = Lgeo + λLdir ,

(26)

where λ is a balancing hyperparameter that controls the
relative contribution of the directional loss.
2) 3D Gait Field Loss: According to Section III-B3, the
training of the 3D gait field F is guided by three consistency constraints. First, for each frame, the embedding of a
foreground pixel should align with that of its corresponding
3D point (either a vertex or a sampled surface point). Second,
within the same frame, the L2 distance between embeddings of
two foreground pixels should match that of their corresponding
3D points. Third, for foreground pixels from different frames
that correspond to the same 3D vertex, their embeddings

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

should remain consistent. The 3D gait field loss Lgf satisfying
the above three constraints is defined as follows:
Lgf =

N
T
1 X 1X 1 X
(
∥Zt [xi , yi ] − eqi ∥22
N n=1 T t=1 |St |

8

objective. The model is optimized using the Re-ID loss Lreid
and the 3D gait field loss Lgf , leading to the overall objective:
L = Lreid + η Lgf ,

pi,t ∈It

(31)

where the hyperparameter η balances the two loss terms.

T

X
1X 1
(d(Zt [xi , yi ], Zt [xj , yj ]) − d(eqi , eqj ))2
T t=1 |Pt |
pi,t ,pj,t ∈It
X
1
+ µ2
∥Zt1 [xi , yi ] − Zt2 [xj , yj ]∥22 ).
|Q|

+ µ1

pi,t1 ∈It1 ,pj,t2 ∈It2

(27)
Here, N represents the batch size. T denotes the number
of frames. St denotes the set of foreground pixels in frame t
that have corresponding 3D points. Pt denotes the set of foreground pixel pairs in frame t along with their corresponding
3D points. Q denotes the set of cross-frame pixel pairs that
correspond to the same 3D vertex. pi,t = (xi , yi , t) specifies
the coordinate (xi , yi ) of the i-th pixel in frame It . Zt is the
embedding map of frame It , and Zt [xi , yi ] is the embedding
vector of pi,t . qi and qj denote the 3D points corresponding
to pi,t and pj,t , respectively, with embeddings eqi and eqj .
The function d(·, ·) denotes the L2 distance between two
embedding vectors, and µ1 , µ2 are weighting parameters. The
three loss terms in Lgf correspond one-to-one to the three
consistency constraints described above.
3) Re-ID Loss: For the video-based appearance feature
XA ∈ RT ×h×w×c , which is obtained by aggregating framelevel features, we seek to reduce its sensitivity to clothing
via an orthogonality constraint. We first apply global average
pooling (GAP) on XA to obtain fA ∈ R1×c . Then, a clothing
classifier cφ , composed of a linear layer followed by a softmax
activation, uses fA to predict the clothing label for the video
sequence. The clothing classification loss Lc is defined as:
1 X
yn log(ŷn ),
N n=1

(28)

where N is the batch size; yn and ŷn denote the ground-truth
and predicted clothing labels for sample n, respectively.
During end-to-end training, the pre-trained clothing classifier cφ is kept frozen. Since its weight matrix Wc defines the
clothing directions, an orthogonal loss Lorg is applied to fA
to suppress its projections along these directions:
Lorg = −

N
X

1
∥fA Wc ∥2F ,
N n=1

(29)

where ∥ · ∥F denotes the Frobenius norm.
In addition, the final identity representation fid is supervised
using the commonly adopted ID loss Lid and triplet loss
Ltri [65] in the Re-ID literature. Accordingly, the Re-ID loss
Lreid is formulated as a weighted combination of the ID loss,
triplet loss, and the orthogonal loss:
Lreid = Lid + w1 Ltri + w2 Lorg ,

A. Datasets and Evaluation Protocols
We evaluate our method on four VCCRe-ID datasets, i.e.,
VCCR [47], CCVID [45], CCV-S [46], and CCV-R [46].
Among them, VCCR, CCVID, and CCV-R are real-world
datasets, and CCV-S is a synthetic dataset. All datasets are
evaluated under two test settings: the standard setting (SD)
and the clothes-changing setting (CC). Under the standard
setting (SD), the retrieval process includes all gallery samples,
covering both clothes-consistent and clothes-changing cases,
whereas under the clothes-changing setting (CC), gallery samples that share both the same clothing and identity labels with
the query are excluded. To further assess the generalization
ability of the proposed method, we conduct additional evaluations on three widely used video-based person Re-ID benchmarks, i.e., MARS [26], LS-VID [66], and iLIDS-VID [67],
which primarily consist of clothes-consistent scenarios. Following standard practice, we evaluate the overall performance
of the Re-ID model using the Rank-1 accuracy of the CMC
curve and mean average precision (mAP). In addition, we
introduce the Embedding Consistency Score (ECS) as a new
metric, which measures the quality of the extracted dense
correspondence embeddings, and is computed on the test set
of each dataset involved in the experiment. ECS is defined as:
ECS =

N

Lc = −

IV. E XPERIMENTS

(30)

where w1 and w2 are weighting coefficients.
4) Overall Training Objective: In the end-to-end training
of the full model, the 3D vertex embeddings and clothing
classifier are kept fixed, so the vertex embedding loss Lvep
and clothing classification loss Lc are excluded from the total

Nd
T
1
1 X 1 X 1 X
(
),
Nd n=1 T t=1 |St | p ∈I 1 + ∥Zt [xi , yi ] − evi ∥1
i,t

t

(32)

where Nd denotes the total number of video sequences, ∥ · ∥1
represents the L1 norm, and all other symbols are defined as
in the first term of Eq. 27. Note that in the test set, no surface
points are sampled, and only pixel-vertex pairs are used to
compute ECS.
B. Implementation Details
1) Pre-training Stage: During the pre-training stage of 3D
vertex embeddings, the node features in the final layer of the
GCN are set to 64 dimensions, i.e., the vertex embedding
dimension D = 64. This pre-training runs for 60 epochs using
the AdamW optimizer [68] at a learning rate of 2 × 10−5 , and
the hyperparameter λ in Lvep is set to 0.40. Then, we freeze
the 3D vertex embeddings output by the GCN and use them to
pre-train the correspondence embedding extractor fθ , enabling
it to gain basic performance before the end-to-end training.
From each triangular mesh face, we sample two surface points.
The extractor fθ follows the Eff-UNet [60] architecture, which
is a lightweight U-Net variant, and is trained with AdamW for
40 epochs at a learning rate of 5 × 10−5 . The hyperparameter
µ1 and µ2 in Lgf are set to 0.40, 0.25, respectively.
For the pre-training of the clothing classifier cφ , the output feature dimension matches the number of clothing label

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Spectrum-guided feature enhancement network for event person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112705

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Spectrum-guided feature enhancement network for event person
re-identiﬁcation
Hongchen Tan
a

a,∗, Yi Zhang

b , Xiuping Liu

b

Institute of Future Technology, Dalian University of Technology, 116024, Dalian, Dalian, China

b School of Mathematical Sciences, Dalian University of Technology, 116024, Dalian, China

a r t i c l e

i n f o

Keywords:
Event person re-ID
Spectrum attention mechanism
Consecutive patch dropout module
Event contextual semantics.

a b s t r a c t
As a novel biosensor, the event camera holds signiﬁcant potential, particularly regarding privacy preservation.
However, compared to traditional cameras, event streams often contain noise and possess extremely sparse semantics, posing a formidable challenge for event-based person re-identiﬁcation (event Re-ID). To address this, we
introduce a novel event person re-identiﬁcation network: the Spectrum-guided Feature Enhancement Network
(SFE-Net). The SFE-Net consists of two innovative components: the Multi-grain Spectrum Attention Mechanism
(MSAM) and the Consecutive Patch Dropout Module (CPDM). MSAM employs a fourier spectrum transform strategy to ﬁlter event noise, while also utilizing an event-guided multi-granularity attention strategy to enhance and
capture discriminative person semantics. CPDM employs a consecutive patch dropout strategy to generate multiple incomplete feature maps, encouraging the deep Re-ID model to equally perceive each eﬀective region of the
person’s body and capture robust person descriptors. Extensive experiments on Event Re-ID datasets demonstrate
that our SFE-Net achieves the best performance.

1. Introduction
Person re-identiﬁcation (Re-ID) is a technique that identiﬁes a speciﬁc person across non-overlapping camera systems, and it is widely
utilized in various surveillance applications. Currently, many outstanding RGB-based methods [1–3] have achieved good performance. However, traditional RGB cameras struggle to capture eﬀective person
semantics in high-exposure and low-light environments. Moreover, capturing too many detailed semantics also poses a risk of privacy leakage. As a novel biosensor, event cameras produce asynchronous data
streams that record the timestamp, location, and polarity of intensity
changes. They can still perceive and capture eﬀective person semantics
in the harsh lighting conditions mentioned above. Besides, the imaging
principle of event cameras means that they do not capture too many semantic details of persons, which also reduces the probability of privacy
leakage. Therefore, conducting person re-identiﬁcation tasks based on
event streams is a highly promising research endeavor.
From the perspective of data representation, in contrast to conventional RGB data, event data exhibit notably sparse semantic content and
incorporate substantial noise, as depicted in Fig. 1. This makes it challenging to directly adapt Re-ID technology from RGB-based to eventbased modalities. However, rich contextual semantics from RGB modalities remain critical for object perception and recognition. To address

this, Cao et al. [4] proposed combining RGB and event modalities for
long-sequence person re-identiﬁcation tasks. By leveraging the complementary strengths of both modalities, the performance of RE-ID was
signiﬁcantly improved. Nevertheless, such hybrid approaches inadvertently reintroduce facial portrait exposure risks due to their reliance
on RGB data. Additionally, long-sequence Re-ID requires substantially
more computational resources and imposes stricter data quality requirements. In this context, Ahmad et al. [5] pioneered person matching using
only event data. Their work focuses on preventing grayscale image reconstruction from event streams, thereby enhancing privacy protection.
However, their approach overlooks two key challenges: (1) extracting
high-quality person semantics from sparse event data, and (2) mitigating the impact of event noise. Similarly, Ahmad et al. [6] attempted to
enhance discriminative features by unfolding person contours in polar
coordinates. However, event-captured data inherently contains signiﬁcant noise and exhibits extremely sparse semantics. Consequently, the
resulting person contours remain fragmented, and the discriminative
power of the extracted descriptors remains limited. So, we try to alleviate two issues to capture high-quality person descriptors.
Event cameras capture changes in light intensity (or electromagnetic
waves) within a scene, generating event-based data streams. However,
during this process, these event streams often contain signiﬁcant noise
due to factors such as photon instability, the sensitivity threshold of

∗ Corresponding author.

E-mail addresses: thc123@dlut.edu.cn, tanhongchenphd@bjut.edu.cn (H. Tan), dlutzy@mail.dlut.edu.cn (Y. Zhang), xpliu@dlut.edu.cn (X. Liu).
https://doi.org/10.1016/j.patcog.2025.112705
Received 19 October 2024; Received in revised form 16 September 2025; Accepted 30 October 2025
Available online 2 November 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112705

H. Tan et al.

sensor hardware, and pixel-level defects (as shown in the right panel
of Fig. 1). Moreover, owing to the highly discrete nature of event semantics, eﬀective semantic information and noise exhibit strong coupling, making them diﬃcult to distinguish. As illustrated in Fig. 1,
a notable characteristic of this noise is that it predominantly manifests as isolated points. Consequently, in the frequency domain, such
noise components are highly concentrated in high-frequency regions.
Thus, we can utilize the spectral convolution theorem [7,8] to separate low-frequency components (representing general semantic information) from high-frequency components (corresponding to noise) in
the spectral domain. In addition to most noise manifesting as isolated
points, the sparsity of event semantics causes a substantial portion
of noise to also appear as semantic isolated points. Simply removing
high-frequency information at a single scale under these conditions inevitably leads to the loss of valuable semantic content. Multi-scale perception further enhances semantic interactions and fusion across regions of varying sizes, thereby stabilizing eﬀective semantics as much
as possible. Therefore, we propose integrating frequency-domain denoising with multi-scale receptive ﬁelds, enabling the model to simultaneously perform denoising and strengthen general semantics across
diﬀerent scales. However, spectral-domain denoising strategies are
prone to oversmoothing artifacts. Moreover, the inherent discreteness
of event semantics hinders the representation of high-quality semantic
information. To address this, we further leverage attention mechanisms
to enhance interactions and fusion among event semantics, thereby
strengthening semantic representation. Therefore, we propose a joint
spatial-spectral denoising and semantic interaction strategy that integrates semantic modeling across both domains.
In addition, deep models tend to concentrate solely on speciﬁc local key semantics and overlook the overall eﬀective semantics. This is
often attributed to the convolutional model being “lazy”. To address
this, the PCB strategy [9] was designed to guide deep models in extracting eﬀective semantics from small image patches. However, due
to the sparse nature of event semantics, accurately determining person
identities based solely on semantics from a single small patch remains
challenging. Moreover, semantic sparsity results in a very low proportion of eﬀective semantics within feature maps, which frequently causes
Dropout-based strategies [10–13] to focus on non-informative semantic
regions, thereby reducing their eﬀectiveness. To mitigate this issue, we
propose an inverse strategy to the PCB [9], which involves systematically erasing features to generate multiple incomplete feature representations. This approach encourages the deep model to balance its perception of eﬀective semantics across diﬀerent spatial regions.
To achieve the goal, we propose a novel event person reidentiﬁcation network, Spectrum-guided Feature Enhancement Network
(SFE-Net), to reduce the interference of noise on the event-streams and
fully explore the discriminative person semantics from the sparse eventstreams. The SFE Network incorporates two novel designs: the Multigrain Spectrum Attention Mechanism (MSAM) and the Consecutive
Patch Dropout Module (CPDM). In MSAM, we ﬁrst convert multi-scale
event semantics into the spectral domain and ﬁlter out high-frequency
noise information. Then, we attempt to enhance the interaction between
event semantics and strengthen the representation of person semantics,
which, to some extent, alleviates the shortcomings of smoothing caused
by the previous ﬁltering process. The second novel design, CPDM, involves a two-step process. First, we conduct a uniform partition on the
feature map, dividing it into smaller patches. Then, we independently
and continuously drop each patch from top to bottom on the feature
map. This operation results in multiple incomplete person feature maps.
By using these multiple incomplete feature maps, the Re-ID model is
driven to equally focus on every discriminative area. Since then, the ReID model captures discriminative semantics from a wider area rather
than getting trapped by a single feature area. This work makes the following contributions:
(i) We introduce the MSAM to eﬀectively ﬁlter noise from event
streams and enhance the representation of person semantics.

Fig. 1. Display of traditional RGB person samples versus event-based person
samples. Unlike RGB images, the semantic information encoded in event tensors is highly sparse and discrete; event cameras intentionally avoid capturing
redundant facial feature details. This characteristic makes them advantageous
for protecting person portraits.

(ii) We design the CPDM to encourage the event Re-ID model to
capture discriminative semantics from a broader area, rather than being
limited to a single feature area.
(iii) We propose the SFE-Net for the event person Re-ID task. Extensive experiments on many Event Re-ID datasets demonstrate that our
SFE-Net achieves the best performance.
2. Related work
2.1. Various person re-ID task
Person Re-ID, a pivotal technology for intelligent video surveillance,
has attracted substantial research attention. Early image-based Re-ID
methods, as demonstrated in studies such as [14], captured the holistic
semantic information of individuals by extracting global features and
employed specialized metric learning losses to enhance performance.
To enable the model to distinguish ﬁne-grained appearance details, recent approaches have adopted local patch partitioning strategies [15],
in which person images are divided into localized regions for detailed
semantic extraction.
Person re-identiﬁcation (Re-ID) faces diverse challenges in realworld scenarios, including occlusion, viewpoint variations, resolution
discrepancies, missing query images, and illumination changes; to address these issues, researchers have developed specialized methods such
as mask-aware mechanisms in [16,17] to mitigate partial occlusion effects, contrastive learning frameworks in [18] for viewpoint adaptation, and dual-resolution feature fusion strategies in [19,20] to bridge
resolution gaps, while alternative paradigms like text-aided retrieval
in [21,22] leverage textual descriptions for zero-shot person retrieval,
infrared modality fusion methods demonstrated in [23,24] signiﬁcantly
improve nighttime Re-ID performance, and disguise-robust approaches
including [25,26] extract gait patterns and appearance-invariant cues
to combat deliberate disguises.
With the growing prominence of Person Re-ID research, privacy
concerns have garnered increasing scrutiny, yet few methods [27–29]
speciﬁcally address this issue. Existing approaches include: 1) face blurring techniques [30] for identity anonymization; 2) person identity shift
(PIS) [31] that preserves image relationships while removing absolute
identity; and 3) cloud-based encryption schemes [27] enabling secure
matching on encrypted data. However, these methods fail to ensure
end-to-end privacy protection, as unauthorized camera access remains
a critical vulnerability. Event cameras, which capture essential motion
information without redundant details, oﬀer a promising privacypreserving alternative to RGB sensors. Their superior performance under
extreme lighting conditions [32,33] has spurred research on event-based
person vision. While some studies focus on reconstructing RGB images
from event data [34], innovative work [5,6] seeks to prevent such reconstruction to enhance privacy. Nevertheless, the sparse and discrete
2

Pattern Recognition 172 (2026) 112705

H. Tan et al.

3.1. Basic information for SFE-Net

nature of event data poses challenges for semantic perception-a limitation overlooked by current methods. To address this gap, we propose a
framework that extracts robust person descriptors from event semantics
while maintaining privacy protection.

The SFE-Net comprises four key components: a Backbone, an EventStream Anonymization Module (EAM), a Multi-grain Spectrum Attention Mechanism (MSAM), and a Consecutive Patch Dropout Module
(CPDM). Its architectural design, including the implementations of
MSAM and CPDM along with the data ﬂow, is illustrated in Fig. 2. This
section provides a concise overview of three fundamental components:
Input Event Representation, the Backbone, and EAM.
Input Event Representation. The output of an event camera is
an asynchronous event stream that encodes the location, time, and
polarity of the intensity changes (increase or decrease in intensity).
For fairness, we also adopt voxel grids of event data, following [5,6].
Spatiotemporal coordinates, 𝑥𝑘 , 𝑦𝑘 , 𝑡𝑘 , lie on a voxel grid such that
𝑥𝑘 ∈ {1, 2, … , 𝑊 }, 𝑦𝑘 ∈ {1, 2, … , 𝐻}, and 𝑡𝑘 ∈ {𝑡0 , 𝑡0 + △𝑡, … , 𝑡0 + 𝐵 △
𝑡}, where 𝑡0 is the ﬁrst time stamp, △𝑡 is the bin size, and 𝐵 is the number of temporal bins and 𝑊 , 𝐻 are the sensor width and height. So, like
many outstanding methods, the input of our SFE-Net is 𝑋𝑒 ∈ ℝ𝐵×𝑊 ×𝐻 .
Backbone. Our SFE-Net adopted the ResNet-50 pre-trained on ImageNet as the backbone network, to encode a person event-stream 𝑋𝑒 ∈
ℝ𝐵×𝑊 ×𝐻 . To get a larger size high-level feature tensor and enhance the
event information representation, we modify the basic structure of the
ResNet-50. We adopted two “ResBlock” to replace the whole “ResNet-50
Stage 4”. Therefore, we can get a larger feature tensor 𝑇 (𝑥) ∈ ℝ24×8×2048 .
Event-Stream Anonymization Module (EAM). Semantics in event
streams are sparse, which can protect the person’s privacy to a certain extent. However, deep image reconstruction methods [34] have
demonstrated impressive abilities in recovering gray images from event
streams, representing a potential threat to the privacy of event-based
vision applications. To this, [5] proposed the Event-Stream Anonymization Module (EAM) to modify the event streams to prevent the image
reconstruction techniques from converting events into intensity images
that can reveal privacy-sensitive information. At the same time, EAM
preserves useful spatial information for performing person Re-ID successfully. For more information about EAM refer to [5] or Appendix A.1.
The EAM takes a raw event-voxel 𝑋𝑒 ∈ ℝ𝐵×𝑊 ×𝐻 and output anonymized
event-voxel 𝑋̂ 𝑒 ∈ ℝ𝐵×𝑊 ×𝐻 . In our SFE-Net, we also adopt such privacy
protection strategy [5]. Unlike them, our SFE-Net focuses on how to
perceive and capture high-quality person semantics from event streams.

2.2. Event-based vision task
As an innovative sensor modality, event cameras distinguish themselves through their high dynamic range (HDR), microsecond-level temporal resolution, and inherent privacy-preserving characteristics. Leveraging these unique advantages, recent years have witnessed a surge in
research exploring event-based vision applications across diverse domains.
Recent advances in event-based vision systems have demonstrated
signiﬁcant potential for object perception tasks [35]. propose a groundbreaking MoE (Mixture of Experts)-based heat conduction algorithm
that achieves an optimal trade-oﬀ between detection accuracy and
computational eﬃciency [36]. pioneer the integration of asynchronous
event cameras into 3D object detection frameworks, leveraging their
high temporal resolution and low data bandwidth to enable real-time
3D perception. For temporal-aware segmentation tasks, [37] introduce
a continuous-time processing paradigm that recurrently updates temporal embeddings by jointly modeling historical and current event streams,
thereby supporting persistent object segmentation. In the domain of motion tracking, [38] develop a salient event blob detector that identiﬁes regions with consistent optical ﬂow through a novel Field of Active Flow Directions (FAFD) representation constructed from the Surface
of Active Events (SAE), achieving robust object tracking performance.
Multi-modal fusion approaches further enhance perception capabilities by combining event data with complementary modalities [36,39].
demonstrate performance improvements in perception tasks through
cross-modal calibration between event cameras and RGB/Radar sensors.
Speciﬁcally, [40] address the spatio-temporal resolution mismatch between RGB and event modalities by designing a dedicated fusion mechanism, which they validate on object tracking benchmarks. Similarly,
[41] propose a hierarchical alignment strategy for event-RGB fusion,
achieving state-of-the-art results in monocular facial expression recognition.
The imaging mechanism of event cameras inherently couples noise
with event data, posing challenges for semantic interpretation. Consequently, extensive research [7,42] has explored denoising methodologies to enhance event-based semantic representations. Notably,
frequency-domain approaches [8,43] ﬁlter high-frequency noise while
preserving low-frequency components, integrating denoising into perception pipelines. Inspired by these advances, we propose extending frequency-domain denoising to event-based Re-ID (RE-ID) tasks to
obtain high-ﬁdelity semantic embeddings. However, three limitations
arise: (1) event semantics exhibit extreme sparsity, making single-scale
ﬁltering prone to over-smoothing and detail loss; (2) asynchronous event
generation results in insuﬃcient contextual aggregation; (3) existing
methods [8,43] rely on ﬁxed receptive ﬁelds, limiting adaptation to
noise variations. Unlike prior works, our approach employs multi-scale
receptive ﬁelds for noise perception and enhances semantic interactions
through cross-scale attention mechanisms.

3.2. MSAM
As described in Intro. 1, the imaging principle of event cameras
results in a signiﬁcant amount of noise in the event streams. Although
[5,6] also adopt event-based models for Re-ID tasks, they neglect to
address the critical challenges of event semantic denoising and enhancement, which are essential for improving feature discriminability
in sparse event data scenarios. Therefore, denoising becomes an important prerequisite for obtaining high-quality semantic representations of
events. Consequently, extensive research [7,42,44] has explored denoising methodologies to enhance event-based semantic representations.
Notably, frequency-domain approaches [8,43] ﬁlter high-frequency
noise while preserving low-frequency components, integrating
denoising into perception pipelines. Inspired by these advances, we
propose extending frequency-domain denoising to event-based Re-ID
(RE-ID) tasks to obtain high-ﬁdelity semantic embeddings. However, the
sparsity of event semantics causes a substantial portion of noise to also
appear as semantic isolated points. Simply removing high-frequency
information at a single scale under these conditions inevitably leads to
the loss of valuable semantic content. Multi-scale perception further
enhances semantic interactions and fusion across regions of varying
sizes, thereby stabilizing eﬀective semantics as much as possible.
Therefore, we employ convolutions with diﬀerent scales to perceive
event semantics and attempt to ﬁlter high-frequency information of
event semantics from the perspective of diﬀerent receptive ﬁelds.
Furthermore, low-frequency information from diﬀerent receptive ﬁelds

3. Method
Structure of SFE-Net is shown in Fig. 2. The proposed SFE-Net is designed to mitigate noise interference in event-based data while comprehensively extracting discriminative person semantics from sparse event
streams. It integrates two innovative mechanisms: (1) the Multi-grain
Spectrum Attention Mechanism (MSAM), which ﬁlters noise and enhances semantic representations of person identities, and (2) the Consecutive Patch Dropout Module (CPDM), which forces the model to focus
uniformly on all discriminative regions, thereby improving the robustness of person descriptors.
3

Pattern Recognition 172 (2026) 112705

H. Tan et al.

Fig. 2. The SFE-Net mainly contains four components: Backbone, Event-Stream Anonymization Module (EAM), the proposed Multi-grain Spectrum Attention Mechanism (MSAM) and the proposed Consecutive Patch Dropout Module (CPDM).

also complements each other, jointly yielding high-quality semantic
representations. However, spectral-domain denoising strategies are
prone to oversmoothing artifacts. Moreover, the inherent discreteness
of event semantics hinders the representation of high-quality semantic
information. To address this, after performing multi-scale denoising,
we attempt to employ an attention mechanism for global semantic interaction to strengthen contextual semantic representations. To achieve
the goal, we propose a Multi-grain Spectrum Attention Mechanism
(MSAM).
The structure of MSAM is illustrated in the top part of Fig. 2. MSAM
performs three main steps: (i) Pre-Spectrum Augmentation is designed
to ﬁlter out noise from event information, (ii) Spatial Semantics Interaction Mechanism to enhance the person semantics representation, and
(iii) Post-Spectrum Augmentation to facilitate global semantic representation and further reﬁne event person semantics.
Pre-Spectrum Augmentation (Pre-SA). In the Pre-SA, we ﬁrst
utilize three types of convolutions with diﬀerent kernel sizes, namely
1 × 1, 3 × 3, and 5 × 5, to extract event semantics at various scales.
The large-scale receptive ﬁeld is employed to stabilize the representation of event semantics, reducing the likelihood of valid
semantics being misclassiﬁed as noise. Meanwhile, the small-scale receptive ﬁeld helps to decrease the possibility of noise being mistaken for valid semantics. Through information ﬁltering and processing, the semantics at diﬀerent scales mutually complement and
reinforce each other. As shown in Fig. 2, we ﬁrst transform the
image features into frequency-domain information through Fourier

transform. Subsequently, we utilize a Gaussian ﬁlter to remove
high-frequency information while retaining low-frequency information.
Next, we decompose the real and imaginary parts of the frequencydomain information and further employ 1 × 1 convolutions to process
this frequency-domain information. After that, we merge the real and
imaginary parts and apply Gaussian ﬁltering again to stabilize lowfrequency semantics and ﬁlter out high-frequency noise. Finally, we
concatenate the results of the two ﬁltering stages and proceed with the
inverse Fourier transform. Let 𝐹 ∈ ℝ𝐶×𝐻×𝑊 denotes the input features,
the Pre-Spectrum Augmentation (Pre-SA) is computed as:
̂ = 𝐶𝑜𝑛𝑣1×1 (𝜎(𝐾, 𝛾) ⊙ 𝛩𝐹 𝐹 𝑇 (𝐶𝑜𝑛𝑣𝑖×𝑖 (𝐹 ))),
𝐹𝑖×𝑖
𝛾 = 𝐹 𝐶(𝐺𝐴𝑃 (𝐶𝑜𝑛𝑣𝑖×𝑖 (𝐹 )))
∑
̂ + 𝐹𝑖×𝑖
̂ )].
𝐹 𝑃 𝑟𝑒 =
[𝛩𝐼𝐹 𝐹 𝑇 (𝜎(𝐾, 𝛾) ⊙ 𝐹𝑖×𝑖

(1)

𝑖∈{1,3,5}

where ⊙ denotes low-pass ﬁltering with adaptive Gaussian smoothed
ﬁlters 𝜎(𝐾, ⋅), which has the same spatial size as 𝐹 , and 𝐾 is the bandwidth. Since the frequency-domain Gaussian ﬁlter is a linear smoothing ﬁlter based on the Fourier transform, it achieves a smoothing effect in the spatial domain by applying Gaussian weighted attenuation
to the frequency components of an image in the frequency domain.
Its core characteristics include the eﬃciency of frequency-domain multiplication, smooth transitions without ringing artifacts, and eﬀective
suppression of high-frequency noise. In light of this, we employ Gaussian smoothed ﬁlters 𝜎(𝐾, ⋅) to ﬁlter out high-frequency noise in the
frequency domain. To make 𝜎(𝐾, ⋅) input-aware, we create an initial 2D
4

Pattern Recognition 172 (2026) 112705

H. Tan et al.

Gaussian map based on 𝐾, and apply Global Average Pooling (GAP) and
Fully Connection Linear layers (FC) on feature map 𝐹 to predict a scale
parameter 𝛾 to update the Gaussian map, i.e. 𝛾 = 𝐹 𝐶(𝐺𝐴𝑃 (𝐶𝑜𝑛𝑣𝑖×𝑖 (𝐹 ))).
Besides, 𝑖 × 𝑖, 𝑖 = 1, 3, 5 is the size of the convolution kernel. The 𝛩𝐹 𝐹 𝑇
is the fourier transform and 𝛩𝐼𝐹 𝐹 𝑇 is the inverse fourier transform.
As shown in Fig. 2, in the fourier transform process, semantics are
decomposed into Real and Imaginary parts. Real part is utilized to regulate the amplitude of the sine wave, while the Imaginary part serves
to adjust its phase. This enables the transformed sine wave to be converted back into the time domain. Following the ﬁltering process, adaptive adjustments to the amplitude and phase values are made through
convolution, ensuring that the adjusted semantics are more conducive to
person matching from a global perspective. To enhance noise ﬁltering
and stabilize low-frequency semantic information, we integrate residual connections with dual Gaussian ﬁltering. Besides, we employ joint
learning of the scaling factors 𝛾 for the two ﬁltering stages based on the
input features 𝐶𝑜𝑛𝑣𝑖×𝑖 (𝐹 ), enabling adaptive adjustment of the retention
ratio of low-frequency information.
Semantics Interaction Mechanism (SIM). After Pre-SA, although
ﬁltering out high-frequency components using frequency-domain strategies can eﬀectively remove noise, it is prone to causing blurred edges
and the loss of texture details. Moreover, the sparsity of event semantics results in weak interactions among semantics and insuﬃcient contextual semantic information. The contextual semantic is detrimental
to the model’s ability to perceive and discriminate semantics. In response, we attempt to employ an attention strategy to achieve interaction and fusion among event semantics, thereby enhancing the representation of event contextual semantics. We ﬁrst compute aﬃnities
coeﬃcients 𝑅 ∈ ℝ𝑁×𝑁 (𝑁 = 𝐻 × 𝑊 ) between diﬀerent event semantics
𝑓 𝑃 𝑟𝑒 ∈ ℝ𝐶 , 𝐹 𝑃 𝑟𝑒 = {𝑓𝑖𝑃 𝑟𝑒 ∈ ℝ𝐶 |𝑖 = 1, 2, ⋯ , 𝐻 × 𝑊 }. Then, according to
aﬃnity weight 𝑅 between event semantics 𝑓 𝑃 𝑟𝑒 , we inject global structure semantics into each event semantic vector, to achieve semantic interaction. Such Semantics Interaction Mechanism (SIM) is calculated as:

3.3. CPDM
Although the objective of MSAM is denoising and feature enhancement, deep models tend to concentrate solely on speciﬁc local key
semantics and overlook the overall eﬀective semantics. This is often
attributed to the convolutional model being “lazy”. To tackle this challenge, the PCB strategy [9] was introduced to guide deep models in extracting discriminative semantics from small image patches. However,
due to the sparse and fragmented nature of event semantics, relying
solely on single-patch semantics for accurate person identiﬁcation remains inadequate. This sparsity further manifests as a critically low
ratio of informative semantics in feature maps, causing Dropout-based
methods [11–13] to inadvertently focus on non-discriminative regions,
thereby diminishing their robustness. To address this limitation, we propose an inverse PCB strategy that systematically erases features to generate multiple incomplete representations. By doing so, we expect that
the deep model can equally perceive each eﬀective region of the feature
map and capture robust person descriptors. So, we propose a Consecutive Patch Dropout Module (CPDM).
In CPDM, (a) we divide the feature map 𝑇 (𝑥) ∈ ℝ24×8×2048 into 𝑚
equal part by row, (b) we obtain 𝑚 incomplete feature maps by erasing
each part, (c) we drive a deep model to capture discriminative semantics
from each incomplete feature map. Based on the feature tensor 𝑇 (𝑥), we
describe in detail the build details of the CPDM.
(i:) As shown in bottom part of Fig. 2, feature map 𝑇 (𝑥) is divided
to 𝑚 uniform patches. (ii:) The DropPatch-i, i = 1, 2, ⋯ , 𝑚 is designed to
drop the 𝑖 − 𝑡ℎ patch on the tensor 𝑇 (𝑥). As shown in bottom right part of
Fig. 2, the feature 𝑇 (𝑥) is divided to 6 uniform patches; the DropPatch-2
is used to drop the 2 − 𝑡ℎ patch on the tensor 𝑇 (𝑥). We can see that the
2-th patch on the 𝑇 (𝑥) is zeroed out. Since this, 𝑚 incomplete feature
tensors 𝑇 (𝑥)∗𝑖 ∈ ℝ24×8×2048 , i = 1, 2, ⋯ , 𝑚. (iii:) These 𝑚 incomplete feature tensors 𝑇 (𝑥)∗𝑖 ∈ ℝ24×8×2048 , i = 1, 2, ⋯ , 𝑚 are fed into the ResBlock
and Gobal Average Pooling (GAP) operation. Since this, we can gain 𝑚
incomplete person descriptors, i.e. 𝑓𝑖 ∈ ℝ512 , i = 1, 2, ⋯ , 𝑚, which is fed
into the loss functions: the Cross-Entropy Loss (CE) and the Triplet (Tri.)
Loss. Here, the ResBlock is composed of three Bottleneck Blocks [45].
Based on the design mentioned above, when compared to the person
feature map 𝑇 (𝑥), the 𝑖 − 𝑡ℎ patch feature region on 𝑇 (𝑥)∗𝑖 is missing.
During the training stage, the person matching metric will encourage
the deep model to capture discriminative semantics from the remaining areas on 𝑇 (𝑥)∗𝑖 . Our CPDM can produce multiple incomplete feature
maps 𝑇 (𝑥)∗𝑖 , i = 1, 2, ⋯ , 𝑚. By doing so, the deep Re-ID model is driven
to equally perceive each eﬀective region of the feature map and capture
robust person descriptors.

(

𝐹

𝑆𝐼𝑀

)
(𝑊 𝑞 ⋅ 𝐹 𝑃 𝑟𝑒 )𝑇 (𝑊 𝑘 ⋅ 𝐹 𝑃 𝑟𝑒 )
= 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥
(𝑊 𝑣 ⋅ 𝐹 𝑃 𝑟𝑒 ).
√
𝐶

MSAMs, namely MSAM-1, MSAM-2, and MSAM-3, in the shallow, middle, and deep sections in the ResNet-50.

(2)

where 𝑊 𝑞 , 𝑊 𝑘 , and 𝑊 𝑣 are projection functions. The 𝑊 𝑞 , 𝑊 𝑘 , and
𝑊 𝑣 are implemented as a 1 × 1 convolution with 𝐶 number of output
channels.
Post-Spectrum Augmentation (Post-SA). We also employ spectrum
augmentation to further reﬁne the event semantics. Filtering out highfrequency information and retaining low-frequency information can
eﬀectively denoise. Additionally, low-frequency information contains
general structure of an image, which is crucial for enhancing the model’s
discriminative ability. Therefore, we attempt to further strengthen lowfrequency semantics to enhance the representation of the overall event
semantics. We deﬁne this process as the Post-Spectrum Augmentation
(Post-SA), which is calculated as follows:

3.4. Loss function in SFE-Net
In our SFE-Net, we mainly adopt the Cross-Entropy Loss and the
Triplet Loss to train the whole model. Therefore, the total loss function
in the SFE-Net is given as follows:

𝐹̂ 𝑃 𝑜𝑠𝑡 = 𝐶𝑜𝑛𝑣1×1 (𝜎(𝐾, 𝜂) ⊙ 𝛩𝐹 𝐹 𝑇 (𝐹 𝑆𝐼𝑀 )), 𝜂 = 𝐹 𝐶(𝐺𝐴𝑃 (𝐹 𝑆𝐼𝑀 ))
𝐹 𝑃 𝑜𝑠𝑡 = 𝛩𝐼𝐹 𝐹 𝑇 (𝜎(𝐾, 𝜂) ⊙ 𝐹̂ 𝑃 𝑜𝑠𝑡 + 𝐹̂ 𝑃 𝑜𝑠𝑡 ) + 𝐹 𝑆𝐼𝑀 .

𝑓

𝑓

𝑝
𝐿𝐺 = 𝐿𝑇𝑝𝑟𝑖. + 𝐿𝐶𝐸
+𝜆

𝑚
∑
𝑖
[𝐿𝑓𝑇 𝑖𝑟𝑖. + 𝐿𝑓𝐶𝐸
] + 𝐿𝑆𝑡𝑟𝑢𝑐𝑡 + 𝐿𝑅𝑒𝑐 ,

(4)

𝑖=1

(3)
𝑓

𝑓

𝑝
Here Triplet Loss ( 𝐿𝑇𝑝𝑟𝑖. ) and Cross-Entropy Loss (𝐿𝐶𝐸
) are applied

Here, to make 𝜎(𝐾, ⋅) input-aware, we also apply pooling and linear
layers on feature map 𝐹 𝑆𝐼𝑀 to predict a scale parameter 𝜂 to update
the Gaussian map. Unlike Pre-SA, we merely utilize Post-SA to enhance
low-frequency linguistic representations. Therefore, the Post-SA model
adopts residual connections, which involves reintroducing 𝐹 𝑆𝐼𝑀 while
calculating 𝐹 𝑃 𝑜𝑠𝑡−𝑆𝐴 . Such residual operation further stabilizes and enhances the global semantics of the event person semantics. Since this,
the proposed MSAM can be utilized to establish both denoising and feature enhancement targets. As indicated in Fig. 2, we incorporate three

to the person descriptor 𝑓𝑝 . Similarly, the Triplet Loss (𝐿𝑓𝑇 𝑖𝑟𝑖. ) and

𝑖
Cross-Entropy Loss (𝐿𝑓𝐶𝐸
) are utilized for the 𝑖-th incomplete person
descriptor 𝑓𝑖 . In addition, 𝐿𝑆𝑡𝑟𝑢𝑐𝑡 + 𝐿𝑅𝑒𝑐 introduced by [5] is used to
modify the event stream to prevent the event information from being
successfully reconstructed into grayscale images. The parameter 𝜆 = 0.2
∑
𝑓𝑖
𝑓𝑖
is used to balance the loss term 𝑚
𝑖=1 [𝐿𝑇 𝑟𝑖. + 𝐿𝐶𝐸 ]. In the following experiments, we verify that when 𝑚 = 8, the SFE-Net achieves the best performance. Importantly, the CPDM is removed in the testing stage, and

5

Pattern Recognition 172 (2026) 112705

H. Tan et al.

we only adopted the person descriptor 𝑓𝑝 ∈ ℝ512 to conduct the person
matching task.

Therefore, when these models are applied to the event modality, RGBbased models ﬁnd it diﬃcult to perceive and capture eﬀective person
semantics amidst the numerous noises. Secondly, from the perspective
of semantic sparsity and density, although the representations of events
are similar to those of the RGB modality, the vast diﬀerence in semantic
richness between the two modalities makes it challenging for RGB models to perceive and capture suﬃcient contextual semantics from events.
Finally, currently, there are very few event-based REID models, and
most of them tend to focus on privacy protection research, neglecting
the exploration of semantics from events that are beneﬁcial for person
matching.
Besides, in Table 1, we also present the number of parameters for
diﬀerent methods. Our model has a relatively large number of parameters, which is primarily attributable to the convolution operation of Spectrum Augmentation and attention mechanism employed
in the MSAM. Secondly, the multi-drive models within the CPDM
uniformly perceive and capture eﬀective semantics. Furthermore, an
increase in the number of parameters is inevitable. This is because
ED-Net [6] and EA-Net [5] are based on a basic backbone network. To
enhance semantic representation and improve performance, it is necessary to design innovative mechanisms to guide the basic model in
perceiving and capturing high-quality semantic representations. Consequently, the number of parameters in the model will increase. Overall,
compared with the increase in the number of parameters, the performance of our SFE-Net has signiﬁcantly improved.

4. Experiment
Datasets. To evaluate the eﬀectiveness of our SEF-Net, we conducted
extensive experiments on Event-ReId, SoftBio (Event) and DukeMTMC
(Event) datasets. Event-ReId Currently, Event-ReId [5] is the ﬁrst
person re-identiﬁcation dataset captured from real event cameras. It
recorded from 33 person IDs, each person walking in the four eventcamera views. The Event-ReId dataset that contains 16𝐾 images from 33
IDs, the training set contains 22 IDs, the testing set contains 11 IDs. In the
testing set, for constructing the query set, we randomly select one voxelgrid per Id per camera from the testing set. SoftBio (Event) is a synthetic
event dataset meticulously crafted from the SoftBio dataset [46] through
open-source event simulator [47]. The SoftBio dataset comprises 152
identities and a total of 64, 472 frames collected with eight surveillance
cameras. The dataset is recorded in an uncontrolled environment, and
each identity may only appear in a subset of cameras, which collect
data under very diﬀerent viewpoints, with drastic changes in illumination and background. DukeMTMC (Event) is also a synthetic event
dataset meticulously crafted from the DukeMTMC-ReID [48]. It contains
16, 522 training images from 702 person IDs, 17, 661 gallery images from
another 702 person IDs, and 2, 228 query images from the same IDs as
query set. In addition to the viewpoint variations. DukeMTMC-reid main
challenges are occlusions and people at lower resolution
Setup and Implementation. We choose the time span for the spatiotemporal voxel grid T≈ 33.3ms for Event-ReId data to be synchronized
with the corresponding RGB frames. Following [5], we set the size of
temporal bin 𝐵 = 5 for the event voxel grid and during training, our
model resized the event voxel grid to 5 × 392 × 192. We use the Adam
optimizer [49] with 400 epochs. The base learning rate is initialized to
1𝑒 − 3 with a linear warm-up [50] in the ﬁrst 50 epochs, then decayed to
1𝑒 − 4 after 200 epochs, and further decayed to 1𝑒 − 5 after 300 epochs.
We set momentum 𝜇 = 0.9 and the weight decay to 5 × 10−4 . The implementation is based on PyTorch. The SFE-Net can be trained on one RTX
3090 in just 2.5 hours.
Evaluation Protocol. We employ two standard metrics as in most
person Re-ID approaches, namely the mean Average Precision (mAP)
and the cumulative matching curve (CMC) used for generating ranking
accuracy. We use Rank-1 and mAP to evaluate the eﬀectiveness of our
SFE-Net.

4.2. Contribution of each omponent in SFE-Net
We validate the eﬀectiveness of each component in our proposed
SFE-Net. The experimental results are documented in Table 2. We
added each of the diﬀerent components from SFE-Net into the
Baseline step by step to verify their eﬀectiveness: (i) We introduce the Multi-grain Spectrum Attention Mechanism (MSAM) into
Baseline, i.e., Baseline+MSAM. Compared with Baseline, Baseline+MSAM increases by 31.8 % on Rank-1 score and 20.9 % on
mAP score on the Event-ReId dataset, 28.5 𝑅𝑎𝑛𝑘 − 1𝑠𝑐𝑜𝑟𝑒𝑎𝑛𝑑13.8 core
on the SoftBio(Event) dataset, 22.1 𝑒𝑎𝑛𝑑10.3 DukeMTMC(Event)
dataset. (ii) We introduce the Consecutive Patch Dropout Module
(CPDM) into Baseline, i.e., Baseline+CPDM. Compared with Baseline, Baseline+CPDM increases by 28.4 𝑒𝑎𝑛𝑑18.5 Event-ReId dataset,
19.6 1 𝑠𝑐𝑜𝑟𝑒𝑎𝑛𝑑7.9 %𝑜𝑛𝑚𝐴𝑃 𝑠𝑐𝑜𝑟𝑒𝑜𝑛𝑡ℎ𝑒𝑆𝑜𝑓 𝑡𝐵𝑖𝑜 (𝐸𝑣𝑒𝑛𝑡)𝑑𝑎𝑡𝑎𝑠𝑒𝑡,15.7 % 𝑜𝑛
𝑅𝑎𝑛𝑘 − 1𝑠𝑐𝑜𝑟𝑒𝑎𝑛𝑑7.9 mAP score on the DukeMTMC(Event) dataset.
(iii) SFE-Net further clearly improves the performance over that of
Baseline+CPDM and Baseline+MSAM. This shows that each component contributes to the event person Re-ID task. (iv) While exploring
high-quality person descriptors, we still adopt the Event-Stream
Anonymization Module (EAM) [5] to prevent the event information
from being able to reconstruct grayscale images. To this end, we
attempt to remove the EAM to obtain SFE-Net∗ . Obviously, when EAM
is removed, the performance of SFE-Net∗ is further improved. Based on
EAM [5], SFE-Net can achieve both privacy protection and performance
enhancement simultaneously.

4.1. Comparison with SOTA methods
We compare our SFE-Net with several state-of-the-art (SOTA) event
re-id methods. Since Event Re-ID is a relatively new and challenging
visual task, there is limited work. Here, we compare the performance of
our SFE-Net with ED-Net [6] and EA-Net [5]. As shown in Table 1, when
compared to EA-Net [5], our SFE-Net has achieved an improvement in
the Rank-1 accuracy by 15.8 % and mAP by 8.9 % on the Event-ReId,
Rank-1 accuracy by 10.1 % and mAP by 8.3 % on the SoftBio (Event).
Compared to EA-Net [5], our SFE-Net has achieved an improvement in
the Rank-1 accuracy by 17.2 % and mAP by 12.0 % on the DukeMTMC
(Event). In addition, as shown in Table 1 with the method of †, we apply
the current classic and high-performance RGB-based Re-ID model to the
event RE-ID task. It is evident from Table 1 that these RGB-based Re-ID
models are not speciﬁc to event modalities and perform poorly on the
task of event Re-ID.
From the perspective of numerical precision performance, our SEFNet achieves the best results. This signiﬁcant achievement can largely
be attributed to the eﬀectiveness of the MSAM and CPDM models. From
the perspective of noise ﬁltering, unlike the event modality, the RGB
modality contains very little random noise, and its semantic information is dense. Correspondingly, as shown in Table 1, RE-ID models based
on the RGB modality do not take the issue of denoising into account.

4.3. Eﬀectiveness dion about MSAM
We focus on the performance of variants of the Multi-grain Spectrum
Attention Mechanism (MASM).
Firstly, we discuss the impact of the number or location of MASM
modules on model performance. As shown in Fig. 2, we introduce three
MASM modules in SFE-Net. Table 3 presents a series of ablation experiments to evaluate the eﬀect of the number and location of MASM
modules on model performance. Generally, it is evident that the contribution of MASM-1 to the model’s performance is greater than that
of MASM-2, and the contribution of MASM-2 is greater than that of
MASM-3. In shallow feature maps, the coupling between person-related
semantics and noise is relatively weak, so their information can be easily
6

Pattern Recognition 172 (2026) 112705

H. Tan et al.

Table 1
A9/A14: Comparison with many SOTA methods on Event-ReId, SoftBio (Event), and DukeMTMC (Event) datasets under
four evaluation metrics: Rank-1, mAP, Parameter(Param.), and FLOPs. †represents that the model was originally based
on the RGB modality, and we applied it to the event modality in this paper.
Methods

Backbone

Param.(M)

Event-ReId

FLOPs(G)

SoftBio (Event)

DukeMTMC (Event)

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

TransReID† [51]
PHA† [52]
BPBreID† [53]
UAL† [54]
ABD-Net† [55]
ED-Net [6]
EA-Net [5]

Transformer
Transformer
ResNet-50
ResNet-50
ResNet-50
ResNet-50
ResNet-50

100
100
24.4
30.5
69.17
24.07

19.3
19.3
6.9
9.0
14.1
7.02

33.6 %
39.6 %
28.6 %
33.6 %
55.3 %
57.9 %
59.2 %

13.8 %
25.5 %
17.8 %
17.6 %
21.4 %
34.9 %
36.1 %

25.4 %
44.7 %
22.5 %
28.9 %
36.8 %
∼ 47 %

12.9 %
29.6 %
10.7 %
16.5 %
15.3 %
∼ 24 %

22.7 %
23.9 %
18.1 %
23.2 %
32.6 %
24.6 %
-

8.7 %
20.8 %
9.4 %
11.5 %
10.9 %
15.5 %
-

SFE-Net

ResNet-50

40.7

14.6

75.0 %

45.0 %

57.1 %

32.3 %

41.8 %

27.5 %

Table 2
Results produced by combining diﬀerent components of the SFE-Net on Event-ReId, SoftBio (Event), and
DukeMTMC (Event) datasets.
Methods

Backbone

Reference

SoftBio (Event)

DukeMTMC (Event)

Rank-1

Event-ReId
mAP

Rank-1

mAP

Rank-1

mAP

Baseline(Base.)
Base.+MSAM
Base.+CPDM

ResNet-50
ResNet-50
ResNet-50

Ours
Ours
Ours

36.4 %
68.2 %
64.8 %

21.1 %
42.0 %
39.6 %

21.1 %
49.6 %
40.7 %

11.7 %
25.5 %
19.6 %

10.7 %
32.8 %
26.4 %

8.3 %
18.6 %
16.2 %

SFE-Net
SFE-Net w/o EAM

ResNet-50
ResNet-50

Ours
Ours

75.0 %
81.7 %

45.0 %
52.6 %

57.1 %
63.6 %

32.3 %
40.8 %

41.8 %
49.4 %

27.5 %
35.8 %

Table 3
Contribution of each MSAM or MSAM’s Variants to
event person Re-ID performance.
MSAM-1

MSAM-2

MSAM-3

Rank-1

mAP

✓

✓

✓

68.2

42.0

✓

✓
✓

61.5
64.9
65.3
60.4
56.6
51.7

37.4
38.7
39.2
35.6
32.2
30.6

36.4

21.1

✓
✓
✓

✓
✓
✓

Fig. 3. The changes in event feature maps after being acted upon by SA (Spectrum Augmentation) and SIM (Semantics Interaction Mechanism). It can be
clearly seen from the ﬁgure that SA can better enhance global semantics, enhance semantic coherence, and alleviate the discreteness of event semantics.
SIM further enriches the structured contextual semantics through interactions
between semantics.

mentation (Conv5+Pre.SA). Table 4 details the contribution of each
strategy at each scale to the model’s performance.
For three scale spectrum augmentation mechanisms, the discreteness and sparsity of event semantics make it diﬃcult to distinguish between valid semantics and noise, as they are intermingled.
Correspondingly, relying solely on a single-scale ﬁlter to remove
high-frequency noise can easily result in the loss of crucial semantic
information. Therefore, we attempt to process event semantics using
convolutions with diﬀerent receptive ﬁelds, thereby ﬁltering out highfrequency semantics under varying receptive ﬁelds. In this way, we can
comprehensively determine whether a certain event is noise by considering diﬀerent receptive ﬁelds; moreover, low-frequency semantics
at diﬀerent scales can complement and enhance each other. Thus, as
shown in Table 4, each spectrum augmentation mechanism contributes
to the performance of MSAM.
Besides, as shown in Fig. 3: we can clearly see that the event semantics are discrete and sparse on the feature map; when we use Spectrum
Augmentation operation (SA) on this basis, the event semantics become
more coherent and the global semantic representation is enhanced. Before applying Spectrum Augmentation operation (SA), there are numerous isolated noisy semantic points in the feature map. These noisy points
generally belong to high-frequency information. Therefore, after applying SA, the number of isolated semantic points in the feature map decreases, and the semantics become smoother and more continuous. Next,

separated. MASM-1 can eﬀectively ﬁlter out most of the noise, enabling
the subsequent deeper model to capture more discriminative person descriptors. Consequently, MASM-1 makes a greater contribution to the
model’s performance. As the convolutional model becomes deeper, the
gradual coupling between noise and semantics becomes more signiﬁcant, making it more diﬃcult to eﬀectively separate the noise. This diminishes the contribution of subsequent MASM modules to the model’s
performance. Therefore, as the model depth increases, the contribution
of MASM modules to the model’s performance decreases accordingly.
Secondly, we investigate the validity of each local strategy in
the MSAM module. To minimize interference between modules, we
solely incorporate MSAM-1 into the baseline model to assess the effectiveness of each individual strategy. As can be seen in Fig. 2,
MSAM comprises Pre-Spectrum Augmentation (Pre-SA), Spatial Interaction Enhancement (SIM), and Post-spectrum Augmentation (Post-SA).
MSAM encompasses three scale spectrum augmentation mechanisms:
Conv1×1 +Spectrum Augmentation (Conv1+Pre.SA), Conv3×3 +Spectrum Augmentation (Conv3+Pre.SA), and Conv5×5 +Spectrum Aug7

Pattern Recognition 172 (2026) 112705

H. Tan et al.

Table 4
Contribution of each strategy to performance in MASM.
MSAM-1 in SFE-Net ( Shown in Fig. 2)

Event-ReId

Conv1+Pre.SA

Conv3+Pre.SA

Conv5+Pre.SA

SIM

Post.SA

Rank-1

mAP

✓

✓

✓

✓

✓

60.4

35.6

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

54.8
56.3
55.6
52.9
55.0
51.3

30.2
32.2
32.8
27.4
30.1
30.5

36.4

21.1

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

Fig. 4. The capture of eﬀective semantics by Baseline and CPDM. The warmer the color, the higher the feature response, but the opposite is also true.

Table 5
Comparison with a series of dropout
strategies on the Event Re-ID dataset.
Method

Event Re-ID
Rank-1

mAP

SpatialDropout [11]
Dropout [10]
DropBlock [12]
Batch DropBlock [13]
PCB [9]

38.5
37.4
35.5
40.2
52.3

21.9
24.1
25.4
23.6
26.7

CPDM (Ours)

64.8

39.6

we use Spatial Interaction Enhancement (SIM), and the person semantics become more coherent and the contextual semantics are enhanced.
The attention mechanism requires facilitating interactions between each
local semantic element and all semantic elements. Correspondingly, connections can be established among diﬀerent semantics to enhance contextual semantic representation. Accordingly, we can clearly observe
from the ﬁgure that the semantic response of the person part becomes
increasingly stronger as the interaction information among semantics
accumulates. Correspondingly, as shown in Table 4, SIM indeed contributes to the performance of MSAM.
When it comes to Post-spectrum Augmentation (Post-SA), eliminating high-frequency information while preserving low-frequency information proves to be an eﬀective denoising strategy. Moreover, lowfrequency information encompasses the overall contour and general
structure of an image, playing a pivotal role in bolstering the model’s
discriminative capacity. Hence, as shown in Table 4, Post-SA also indeed contributes to the performance of MSAM. Above all, numerical
and visual experimental results validate that each strategy signiﬁcantly
improves the model’s performance.

Fig. 5. The inﬂuence of the ﬂuctuation of patch number 𝑚 on the performance
in Baseline+CPDM.

4.4. Discussion about CPDM and some dropout strategies
In this section, we ﬁrst explore the impact of the number of patches 𝑚
on the performance of the CPDM. In our CPDM implementation, the size
of 𝑇 (𝑥) is 24 × 8 × 2048. We present the performance of Baseline+CPDM,
when 𝑚 = 2, 3, 4, 6, 8, 12 with 24 %𝑚 = 0. The results are presented in
Fig. 5. When the value of 𝑚 is too small, it is equivalent to removing
a signiﬁcantly large semantic region. For event streams with extremely
sparse semantics, removing an excessively large semantic region will
severely impair the model’s perception of contextual semantics and its
ability to capture key semantics. Consequently, when 𝑚 is too small, the
model’s performance will be adversely aﬀected. As we can also clearly
8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 195 (2026) 108206

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Stochastic style perturbation modelling for visible-Infrared person
re-Identiﬁcation with severely modality imbalance
Haojie Liu a,b , Zhiyong Li a , Jianyang Gu a , Mingyu Wang a , Q. M Jonathan Wu c , Wei Jiang

b,∗

a The College of Control Science and Engineering, Zhejiang University, Hangzhou, 310027, China
b

The School of Computer Science and Technology, Zhejiang University of Water Resources and Electric Power, Hangzhou, 310018, China

c The department of Electrical and Computer Engineering, University of Windsor, Canada

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Data imbalance problems
Cross-modality learning

a b s t r a c t
In this paper, we address the challenge of data imbalance in visible-infrared person re-identiﬁcation (VI-ReID).
Previous studies often presuppose a uniform distribution of training data across various modalities, however,
due to constraints such as device limitations, privacy concerns, and operational conditions, gathering nightime
infrared modality data can be prohibitively expensive or even impossible. Consequently, the limited infrared (IR)
modality data tend to be overshadowed by the more plentiful visible (RGB) modality data during the training,
particularly in scenarios marked by signiﬁcant imbalance. To counter this issue, we introduce the Co-Modality
Balance Learning (CMBL) framework, designed to recalibrate the balance in cross-modality learning and enhance
the extraction of discriminative features. Initially, we design a Stochastic Style Perturbation (SSP) module that
dynamically generates IR modality samples within the deep feature space to emulate the characteristics of a
balanced dataset. Subsequently, we develop a cross-distribution alignment loss, which enables a reﬁned optimization of sparse modality features to improve their accuracy and robustness. Additionally, we propose the
novel Class-Aware Contrast Similarity Learning (CACS) strategy, which capitalizes on latent feature consistency
to boost intra-class compactness and inter-class separation. Our extensive empirical evaluations and ablation
studies on two publicly available cross-modality datasets under imbalanced conditions underscore the eﬃcacy
of our approach, showcasing its ability to adeptly navigate the complexities of data imbalance in VI-ReID.

1. Introduction
Person Re-identiﬁcation (ReID) aims to identify a query person from
non-overlapping camera networks (Leng et al., 2020; Liu et al., 2021a;
Ye et al., 2021b). Existing methods mainly exploit the appearance information captured by visible light cameras to associate the person
identities (Cheng et al., 2024; Huang et al., 2024). However, under
poor illumination, visible cameras cannot provide suﬃcient information, resulting in ineﬀective feature recognition. Wu et al. (2017) propose to employ infrared cameras alongside visible cameras to supplement the low-light information, thus giving rise to the Visible-Infrared
Re-Identiﬁcation (VI-ReID) task.
For VI-ReID, previous cross-modality methods have endeavored to
bridge the gap between modalities through the use of image generation techniques (Liu et al., 2025, 2023, 2022c; Pang et al., 2024; Wang
et al., 2019b; Zhong et al., 2021) and invariant feature learning strategies (Chen et al., 2021; Hu et al., 2022; Huang et al., 2022; Liu et al.,
2020), achieving remarkable results on public benchmarks. However, a
fundamental assumption underpinning the success of these networks is

the completeness of modality data. In practice, obtaining infrared images of people poses signiﬁcant challenges due to the limited number
of pedestrians active at night and the high cost associated with infrared
cameras. As a result, it is common to encounter scenarios where RGB
modality data is signiﬁcantly more abundant than in infrared modality data, challenging the eﬀectiveness of cross-modality approaches in
practical applications. To this end, we propose to address the data imbalance problem in this paper where certain infrared (IR) modality samples
are absent, whereas visible (RGB) modality data remains accessible. In
this case, the dataset contains signiﬁcantly fewer IR samples compared
to the number of RGB samples. This discrepancy in sample sizes not
only hampers the network’s ability to learn from IR data eﬀectively but
also reﬂects a broader issue in balancing the representation of diﬀerent
modalities within a dataset. A common strategy to address data imbalance involves adjusting the dataset’s balance through resampling at the
data level. This approach generally employs two classical resampling
techniques: undersampling (Gu et al., 2021; Yang & Ismail, 2022; Yulong et al., 2022) and oversampling (Chawla et al., 2002; Wang et al.,
2013; Yang et al., 2020). Undersampling is designed to restore balance

∗ Corresponding author.

E-mail address: jiangwei_zju@zju.edu.cn (W. Jiang).
https://doi.org/10.1016/j.neunet.2025.108206
Received 28 December 2024; Received in revised form 20 May 2025; Accepted 9 October 2025
Available online 14 October 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 195 (2026) 108206

H. Liu et al.

∙ We conduct comprehensive experiments to evaluate the eﬀectiveness of our method in the problems of imbalance VI-ReID. The experimental results demonstrate that our method surpasses the state-of-thearts by a large margin.
2. Related works
Visible-Infrared Re-ID Methods. Visible-infrared person reidentiﬁcation (VI-ReID) is proposed to improve the performance of
surveillance systems in low-light conditions during nighttime (Kim et al.,
2023; Wu et al., 2017; Ye et al., 2021b). Early works attempted to mitigate the cross-modality discrepancy at image level via generative adversarial networks (GANs) (Liu et al., 2023; Qi et al., 2024; Wang et al.,
2019a; Zhong et al., 2021). However, due to a lack of pair cross-modality
samples, GAN-based methods always suﬀer from uncontrollable noise
and ambiguity in identity information. Another line of shared feature
learning approaches (Du & Zhang, 2023; Feng et al., 2023; Liu et al.,
2021b; Ye et al., 2020a, 2018; Zhang et al., 2022a; Zhao et al., 2022)
aim to design various network structures or loss functions to bridge the
modality gap. One typical design choice is the utilization of two-stream
structures (Liu et al., 2025, 2021b; Ye et al., 2020a; Yu et al., 2025),
which involves the incorporation of modality-speciﬁc shallow layers and
shared deeper layers to learn rich feature representation. However, regardless of whether it is feature learning methods or GAN-based methods, their primary focus is on addressing the issue of modality gap, overlooking the challenge of data imbalance modality problem in VI-ReID.
In recent years, researchers have focused on alleviating the modality gap between infrared and visible light in video sequences. The pioneering work by Lin et al. (2022) introduced HITSZ-VCM, the ﬁrst
video-based visible-infrared person re-identiﬁcation dataset, along with
an adversarial learning framework for extracting modality-invariant features. While eﬀective in certain scenarios, this approach was found to
inadvertently discard potentially valuable modality-speciﬁc characteristics. To overcome this limitation, Li et al. (2023) developed the innovative IBAN (Intermediary-Based Alignment Network) model, which
employs anaglyph representations of pedestrian images as an intermediate modality. This architecture not only preserves distinctive modalityindependent features but also incorporates a bidirectional spatiotemporal aggregation module to eﬀectively exploit the rich spatial and temporal information inherent in video sequences.
Imbalanced Modality Learning. There exists two types of imbalanced scenario in multi-modal learning realm, eg. data imbalance and
optimization imbalance. To address the data imbalance problem, previous researchers introduce the re-sampling (Cui et al., 2019; Huang et al.,
2016; Wang et al., 2017) and re-weighting (Buda et al., 2018; Byrd &
Lipton, 2019; Cao et al., 2019) strategies to balance optimization proportion during the training process. More recently, MAUM (Liu et al.,
2022a) proposes to learn unidirectional metrics to adjust the modalityspeciﬁc augmentation against modality imbalance, which achieves the
satisfactory results. For another issue of optimization imbalance, Wang
et al. (2020) introduced gradient blending strategy to achieve optimal
blending of modalities. Du et al. (2021) exploited knowledge distillation
technique to distill knowledge from a well-trained uni-modal model.
These methods have potential to improve performance, however, they
necessitate the introduction of additional modules, complicating the
training process. In this paper, we tackle this issue from the feature normalization standpoint, addressing the imbalance problem by adaptively
perturbing features without the need for extra modules.

Fig. 1. Previous data-level oversampling methods face several issues: (a) The
presence of noise samples in the minority class. (b) The minority class spans
multiple subspaces. (c) The minority class exhibits high distribution density.

by eliminating some samples from the majority class, whereas oversampling aims to increase the dataset’s diversity by replicating existing minority class samples or creating new synthetic ones. Typically, synthetic
samples are produced by selecting a minority class sample at random
and performing an interpolation among its neighbors. However, this
straightforward application of local linear interpolation could lead to
generated samples deviating from the original data distribution. Additionally, as depicted in Fig. 1, the eﬀectiveness of basic data-level oversampling methods is signiﬁcantly aﬀected by the sample distribution,
constraining their utility in novel scenarios.
In this research, we extend the use of oversampling techniques into
cross-modality data imbalance scenario. Unlike previous approaches
that focus on data-level oversampling, we propose Stochastic Style Perturbation (SSP) to generate novel sparse modality samples at the feature
level. Our main idea is treating the infrared modality feature statistic as
a undeterministic point measured from the feature to synthetize novel
sparse modality instances. Speciﬁcally, we estimate the new style information range based on the variances of the mini-batch statistics in
an eﬃcient non-parametric manner. Subsequently, variants of feature
statistics are randomly sampled from this estimated range and utilized
to replace the original deterministic values. Due to the generated feature
statistics with diverse possibilities, the models can be trained on a more
varied distribution of infrared modality data and encode more robust
modality-balanced features.
In addition, to minimize modality discrepancies across both
modality-shared and SSP-enhanced features, we propose a crossdistribution alignment loss to optimize the distribution distance relationships in the modality level. Furthermore, we introduce the ClassAware Contrast Similarity (CACS) learning approach to regulate feature
relationships in the instance level. CACS calculates the aggregated class
centers for both visible and infrared modalities to reﬁne the similarity relationships among diﬀerent identities, which signiﬁcantly enhances the
angular discriminability of features and promote the learning of underlying interaction between diﬀerent modalities. Extensive experiments
were conducted to validate the eﬀectiveness of the proposed method
(Fig. 2).
The main contributions can be summarized as:
∙ We reveal a new problem for VI-ReID, termed imbalance VI-ReID.
In contrast to existing VI-ReID studies which only consider the modality
discrepancy issue, we consdier to tackle the scenario where data from
one modality is severely deﬁcient.
∙ To allow the neural network to work with highly imbalanced data,
we propose the SSP module that can dynamically estimate the potential
of instances within the deep feature space without any modiﬁcation on
its architecture, thus are applicable to a wide range of deep learning
models.
∙ To promote the balanced representation learning, we introduce a
cross-distribution alignment loss to optimize the distance relationships
in the global distribution level. In addition, we propose a novel CACS
learning paradigm in the class-instance level to explore the discriminative feature relationships.

3. Method
3.1. Stochastic style perturbation
3.1.1. Background
The conceptually approach to addressing imbalanced data through
oversampling involves randomly duplicating existing instances until
2

Neural Networks 195 (2026) 108206

H. Liu et al.

Fig. 2. This is a high-level overview of our CMBL framework, which uses a three-stream network for multi-modal feature encoding. SSP features are produced
from the IR features following conv layer1 in the IR stream subnetwork. Parameters of conv layers 2-4 in the three-stream network are shared between RGB and IR
streams, while being independent from the oversampling stream. The cross-distribution alignment module aligns heterogeneous features using MMD distance at the
global distribution level. The Class-Aware Contrast Similarity Learning (CACS) module enhances feature discrimination through intra- and inter-class similarity at a
nuanced identity level. Best viewed in color.

balanced class distributions are achieved. However, this method often
results in the minority class distribution being overly concentrated in
the areas where the original observations were located. Consequently,
models trained on data altered in this manner are at a higher risk of
overﬁtting. To mitigate this issue, data-level neighborhood-based methods (Chawla et al., 2002; Cui et al., 2019; Huang et al., 2016; Wang
et al., 2017) have been developed, among which the Synthetic Minority
Over-sampling Technique (SMOTE) (Chawla et al., 2002) is one of the
most prominent. The basic idea of SMOTE (Chawla et al., 2002) is to
identify the nearest neighbors within the same class for a given minority instance and then synthesizes new samples in the space between the
target instance and one of its neighbors. Speciﬁcally, suppose we have
a minority class sample (𝑥) and its (𝑘) nearest neighbors (𝑥1 , 𝑥2 , … , 𝑥𝑘 ).
The formula for generating a new synthetic sample by SMOTE (Chawla
et al., 2002) is as follows:
𝑥𝑛𝑒𝑤 = (𝑥 + 𝑟𝑎𝑛𝑑(0, 1) × (𝑥𝑖 − 𝑥)),

Fig. 3. Visualization in multi-modal scenarios for SMOTE (Chawla et al., 2002):
pre-synthesis (left) vs. post-synthesis (right). Due to the large modality discrepancy, the generated new samples always deviate from the original modality distribution (green dots). Best view in color.

age by mean and standard deviation 𝜇(𝑓 𝑡 ) ∈ ℝ𝐵×𝐶 , 𝜎(𝑓 𝑡 ) ∈ ℝ𝐵×𝐶 , as:

(1)

1 ∑∑ 𝑡
𝑓
,
𝐻𝑊 ℎ=1 𝑤=1 𝑏,𝑐,ℎ,𝑤
√
√
)2
√ 1 𝐻
∑ 𝑊
∑(
𝑡
𝑡
𝜎(𝑓 ) = √
𝑓𝑏,𝑐,ℎ,𝑤
− 𝜇(𝑓 𝑡 ) .
𝐻𝑊 ℎ=1 𝑤=1
𝐻

𝑊

𝜇(𝑓 𝑡 ) =

where 𝑟𝑎𝑛𝑑(0, 1) ranges between [0, 1] and is used to control the position
of the synthetic sample. Repeating the steps allows for the creation of
synthetic samples in quantities tailored to achieve dataset balance.
The foundational premise of SMOTE (Chawla et al., 2002) is that the
spaces between the nearest minority neighbors are viable for generating new instances. However, this assumption encounters challenges in
multi-modal scene. As illustrated in Fig. 3, due to large modality discrepancy, the nearest minority neighborhood is spread apart, leading to
the generation of synthetic samples that deviate from the original distribution. Moreover, given the varying degrees of modality discrepancy, it
is unclear what the optimal size of the neighborhood 𝑘 should be.

(2)

(3)

Here, 𝜇(𝑓 𝑡 ) and 𝜎(𝑓 𝑡 ) encapsulate the statistical characteristics of the
image, capturing its inherent style.
The normalized features that encode the semantic information are
represented as:
𝑓 =𝛾⊙

𝑓 𝑡 − 𝜇(𝑓 𝑡 )
+ 𝛽,
𝜎(𝑓 𝑡 )

(4)

where 𝛾 and 𝛽 represent the channel-wise aﬃne parameters. According
to the work Huang and Belongie (2017), Tang et al. (2021), modifying
the feature statistics (𝛾, 𝛽) can change the feature style of an image.
Then, to obtain reasonable values of 𝛾 and 𝛽, we ﬁrstly utilize the variance of 𝜇(𝑓 𝑡 ) and 𝜎(𝑓 𝑡 ) to estimate meaningful style vectors, denoted
as:

3.1.2. Feature-level stochastic style perturbation
Rather than depending on nearest neighbors for generating new samples at the data level, in this paper we explores the potential of a featurebased approach. Our objective diverges from SMOTE techniques: we
aim to directly generate IR features within the deep feature space. By
undertaking this approach, we aspire to transform the style of minority
modality samples while preserving their semantic information, ensuring
that the generated features do not stray from the original distribution.
Speciﬁcally, as shown in Fig. 4, given infrared features 𝑓 𝑡 ∈
ℝ𝐵×𝐶×𝐻×𝑊 where 𝐵, 𝐶, 𝐻, 𝑊 denoting the batchsize, dimension of
channel, height and width, respectively, we represents the style of im-

𝜎(𝜇(𝑓 𝑡 ))2 =

𝐵
𝐵
1 ∑
1 ∑
(𝜇(𝑓 𝑡 ) −
𝜇(𝑓 𝑡 ))2 ,
𝐵 𝑏=1
𝐵 𝑏=1

𝐵
𝐵
1 ∑
1 ∑
𝜎(𝜎(𝑓 )) =
(𝜎(𝑓 𝑡 ) −
𝜎(𝑓 𝑡 ))2 ,
𝐵 𝑏=1
𝐵 𝑏=1
𝑡 2

3

(5)

Neural Networks 195 (2026) 108206

H. Liu et al.

a RGB feature 𝑓𝑖𝑣 , IR feature 𝑓𝑖𝑡 and perturbed feature 𝑓𝑖𝑝 generated by
SSP, we ﬁrst calculate MMD (Gretton et al., 2012) to measure modality
discrepancy between visible and infrared domains:
𝐷 (𝑣, 𝑡) = ‖𝐄𝑣 [𝜙(𝕌𝑁
(𝑓 𝑣 ))] − 𝐄𝑡 [𝜙(𝕌𝑁
(𝑓 𝑡 ))]‖2 ,
𝑖=1 𝑖
𝑖=1 𝑖

Fig. 4. Illustration of the proposed stochastic style perturbation (SSP) layer. We
approximate the distribution of styles via disturbing the variance of the feature
statistics and randomly sample the scale parameter 𝛾𝑠𝑠𝑝 and shift parameter 𝛽𝑠𝑠𝑝
from the estimated style range to obtain the projected feature 𝑓 𝑝 .

𝐷 (𝑣, 𝑝) = ‖𝐄𝑣 [𝜙(𝕌𝑁
(𝑓 𝑣 ))] − 𝐄𝑝 [𝜙(𝕌𝑁
(𝑓 𝑝 ))]‖2 ,
𝑖=1 𝑖
𝑖=1 𝑖
where 𝜎(𝜇(𝑓 𝑡 ))2 and 𝜎(𝜎(𝑓 𝑡 ))2 denote the estimator of the mean and

𝛾𝑠𝑠𝑝 = Random[ 𝜎(𝑓 𝑡 ) − 𝜖𝜎 𝜎(𝜎(𝑓 𝑡 )), 𝜎(𝑓 𝑡 ) + 𝜖𝜎 𝜎(𝜎(𝑓 𝑡 )) ],

ancy (MMD) between RGB and perturbed features not only reﬁnes the
shared identity-speciﬁc information but also enables further optimization within the IR modality.
Then, we continue to compute the metric distance between 𝕌𝑁
(𝑓 𝑡 )
𝑖=1 𝑖
𝑝
and 𝕌𝑁
(𝑓
)
,
which
is
denoted
as:
𝑖=1 𝑖

(6)

𝐷 (𝑡, 𝑝) = ‖𝐄𝑣 [𝜙(𝕌𝑁
(𝑓 𝑡 ))] − 𝐄𝑝 [𝜙(𝕌𝑁
(𝑓 𝑝 ))]‖2 .
𝑖=1 𝑖
𝑖=1 𝑖

where 𝜖𝜇 and 𝜖𝜎 are hyper-parameters that are sampled from the standard Gaussian distribution, i.e. 𝜖𝜇 ∼  (0, 1), 𝜖𝜇 ∼  (0, 1). Random(.)
means the random sampling within the intervals.
Finally, we replace the 𝛽 and 𝛾 in Eq. (4) by 𝛽𝑠𝑠𝑝 and 𝛾𝑠𝑠𝑝 and the
generated SSP feature 𝑓 𝑝 is formulated as:
𝑓 𝑝 = 𝛾𝑠𝑠𝑝 ⊙

𝑓 𝑡 − 𝜇(𝑓 𝑡 )
+ 𝛽𝑠𝑠𝑝 .
𝜎(𝑓 𝑡 )

(9)

where the perturbed feature (𝑓𝑖𝑝 ) shares semantic information with the
IR feature (𝑓𝑖𝑡 ). As a result, optimizing the Maximum Mean Discrep-

standard deviation, respectively.
After obtaining the estimator for each channel, we remodel the feature statistics by introducing the Gaussian distribution and performing
random sampling within the speciﬁed style change interval:
𝛽𝑠𝑠𝑝 = Random[ 𝜇(𝑓 𝑡 ) − 𝜖𝜇 𝜎(𝜇(𝑓 𝑡 )), 𝜇(𝑓 𝑡 ) + 𝜖𝜇 𝜎(𝜇(𝑓 𝑡 )) ],

(8)

where 𝕌𝑁
𝑖 (.) is the set of corresponding features, 𝜙(.) is an implicit feature mapping function,  refers to Reproducing Kernel Hilbert Space
and 𝑁 is the number of images in the 𝑖th identity.
Previous methods (Jambigi et al., 2021; Wu et al., 2022) directly apply this constraint to align the RGB and IR distributions. However, in the
case of imbalance VI-ReID, where the quantity of IR images signiﬁcantly
trails behind that of RGB images, directly minimizing MMD across the
visible and infrared modalities would lead the infrared modality mindlessly lean towards visible modality instead of aligning their shared information, which limits the performance improvement.
In this paper, beneﬁting from oversampling of SSP features, we propose the cross-optimization of two distributions. Speciﬁcally, for the
RGB distribution 𝕌𝑁
(𝑓 𝑣 ), we additionally calculate the metric distance
𝑖=1 𝑖
𝑣 ) and 𝕌𝑁 (𝑓 𝑝 ), denoted as:
between 𝕌𝑁
(𝑓
𝑖=1 𝑖
𝑖=1 𝑖

(10)

Here, it should be emphasized that optimizing 𝐷 (𝑡, 𝑝) serves a dual
purpose: it reduces the discrepancy within the same modality and guarantees that the oversampled features are aligned with the original distribution.
Finally, the cross distribution alignment loss can be formulated as
follows:

(7)

𝑐𝑟𝑜𝑠𝑠 = 𝐷 (𝑣, 𝑡) + 𝐷 (𝑣, 𝑝) + 𝐷 (𝑡, 𝑝).

We position the SSP module after the CNN layer, notably at the second,
third, and fourth stages of the ResNet50 model. Given that SSP operates
without parameters, the computational overhead it introduces is minimal. For the inference phase, SSP is excluded from the feature extraction
process.
Discussion. Our proposed Stochastic Style Perturbation (SSP) draws
inspiration from ApaIN (Huang & Belongie, 2017), which achieves arbitrary image style transfer by substituting the feature statistics of a
content image with those from a style image. However, ApaIN (Huang
& Belongie, 2017) requires the target image as a reference, which does
not align with our task’s requirements. More recently, Zhou et al. introduced MixStyle (Zhou et al., 2023), a method that linearly combines the
statistics of two samples, and Li et al. (2022) designed DSU, which introduces disturbances to the original feature statistics. While these methods
have been successful in domain generalization as CNN training regularization strategies, their synthesized features remain closely related to
those of the original training images, diverging from the spirit of our
co-modlaity balance learning.
Unlike previous feature normalization methods, SSP focuses on dynamically altering feature statistics from a Gaussian distribution interval. This approach ensures the generation of new features that are meaningful and independent from the original styles in each batch, oﬀering
a distinct advantage in versatility and applicability to data imbalance
datasets.

(11)

3.3. Class-aware contrast similarity learning
Cross distribution alignment regularization mainly focus on the
alignment of global distributions, while ignoring the intra and inter class
correlations of features. By considering the relationships between different classes on a ﬁne-grained instance level, we impose the contrast
similarity constraint to the center of each classes to minimize intra-class
separation while maximizing inter-class separability. Speciﬁcally, given
the feature outputs {𝑓𝑖𝑣 , 𝑓𝑖𝑡 , 𝑓𝑖𝑝 } of the same identity, we ﬁrst calculate
the center of the visible modality 𝑐𝑖𝑣 and the aggregated infrared modality feature center 𝑐𝑖𝑡 :
𝑐𝑖𝑣 =

𝑁
𝑁
1 ∑ 𝑣 𝑡
1 ∑ 𝑡
(𝑓𝑖 ), 𝑐𝑖 =
(𝑓 + 𝑓𝑖𝑝 ),
𝑁 𝑖=1
2𝑁 𝑖=1 𝑖

(12)

Then, we conduct the contrast similarity learning from following two
aspects.
Intra-class Contrast. Intra-class contrast uses cross distribution selfto-self matching to achieve discriminative representations by maximizing self-similarity. To measure the degree of similarity between classes,
we use cosine distance to calculate the relation matrixs  𝑣𝑡 = 𝑐𝑖𝑣 (𝑐𝑖𝑡 )𝑇 ∈
ℝ𝑁×𝑁 , where 𝑇 means matrix transpose operation. Then, we employ
softmax operation to calculate its probability distribution matrix as:
 𝑣𝑡 = 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥( 𝑣𝑡 ),

3.2. Cross distribution alignment

(13)

where 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥(.) is row-wise softmax operation.
To maximize the matching probability of self-to-self, we further employ the cycle consistency strategy to achieve inverse relational matrix

To reduce the discrepancy between the diﬀerent distributions within
the shared space, we propose a cross distribution alignment loss. Given
4

Neural Networks 195 (2026) 108206

H. Liu et al.

Fig. 5. The prediction distribution of 100 IR testing samples of baseline (a) and CMBL (b) on the SYSU-MM01 dataset. The x-axis denotes the normalized logit
output, and x=0.5 represents the classiﬁcation boundary. The y-axis denotes the index of samples. Red and blue dots denote two diﬀerent classes.

 𝑡𝑣 = (𝑐 𝑡 )𝑇 𝑐 𝑣 . Similarly to Eq. (13), the probability matrix can be calculated as:
 𝑡𝑣 = 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥( 𝑡𝑣 ).

Finally, the class-aware contrast similarity loss is formulated by:
𝑐𝑙𝑎𝑠𝑠 = 𝜔1 𝑖𝑛𝑡𝑟𝑎 + 𝜔2 𝑖𝑛𝑡𝑒𝑟 ,

(14)

Here, the corresponding results  𝑣𝑡 and  𝑡𝑣 are regarded as matching
the same classes from visible to infrared modality and the same classes
from infrared to visible modality.
According to the of cycle association consistency, following the forward association  𝑣𝑡 and backward association  𝑡𝑣 , we further enhance
our intra-class contrast learning through self-to-self matching:
𝔻(𝑣, 𝑡) = diag( 𝑣𝑡  𝑡𝑣 ),

3.4. Training objective
We combine the following loss functions to make up the ﬁnal training
objective:
 = 𝐶𝐸 + 𝑐𝑟𝑜𝑠𝑠 + 𝑐𝑙𝑎𝑠𝑠 ,

(15)

(22)

where 𝐶𝐸 represents the cross entropy loss. The synergy among these
loss functions is critical for reﬁning the model’s ability to distinguish between classes, align distributions across modalities, and maintain generalization capabilities.

where diag(⋅) means to get the diagonal matrix. The intra-class contrast
loss then can be calculated as:
1 ∑
𝑖𝑛𝑡𝑟𝑎 =
(||𝔻(𝑣, 𝑡) ⊖ 𝕀||1 ),
(16)
𝑁
where ⊖ denotes the element-wise subtraction operation, ||.||1 denotes
the 𝐿1 norm and 𝕀 means the identity matrix, i.e.,
{
0,
if 𝑖 ≠ 𝑗
𝕀𝑖,𝑗 =
0 ≤ 𝑖, 𝑗 < 𝑁
(17)
1,
otherwise

3.5. Experiments
3.5.1. Datasets and evaluation metric
SYSU-MM01 (Wu et al., 2017) contains a total of 30,071 visible images and 15,792 infrared images of 491 person identities captured by 2
near-infrared cameras and 4 visible cameras. Following Wu et al. (2017),
we conduct experiments on both all search and indoor-search modes.
RegDB (Nguyen et al., 2017) contains 8240 images of 412 identities
captured by a pair of overlapping cameras. For testing, we randomly
divide the dataset into two parts and alternatively apply all RGB/IR
images as the gallery set.
LLCM (Zhang & Wang, 2023) comprises a total of 25,626 visiblelight and 21,141 infrared images collected from 1064 unique identities.
For training purposes, we utilize 16,946 visible and 13,975 infrared
images, with the remaining samples allocated for testing. The benchmark provides two distinct evaluation protocols: (1) visible-to-infrared
(VIS→IR) and (2) infrared-to-visible (IR→VIS) cross-modality retrieval
tasks.
Evaluation Protocol. To quantitatively evaluate the performance of
our proposed model, we adopt the Cumulated Matching Characteristics
(CMC) curve at Rank-1, Rank-10 and Rank-20, and the mean Average
Precision (mAP) as the evaluation metrics.
Modality Imbalanced Scenario. Due to the limited number of
pedestrians active at night and the high cost associated with infrared

Inter-class Contrast. Following the optimization of intra-class contrast, the similarity among identical targets is heightened, nearing a
value of 1 as much as possible. Next, our focus shifts to diminishing
the cross-similarity, with the goal of bringing the oﬀ-diagonal elements
of the similarity matrix as close to 0 as feasible.
Speciﬁcally, we ﬁrst subtract the diagonal matrix from the original
matrix to obtain the oﬀ-diagonal matrices:
̃ 𝑡) =  𝑣𝑡  𝑡𝑣 − 𝔻(𝑣, 𝑡)
𝔻(𝑣,

(21)

where 𝜔1 and 𝜔2 denote the weight parameters.

(18)

Since the oﬀ-diagonal matrixs are symmetric, the inter-class contrast
loss can be applied only to the lower triangular part, which is denoted
as:
1 ∑ ̃
𝑖𝑛𝑡𝑒𝑟 =
(|𝔻(𝑣, 𝑡) ⊙ 𝕋 ⊖ ℤ|1 ),
(19)
𝑁
where ℤ means the zero matrix and 𝕋 denotes a strict lower triangular
matrix, as:
{
0,
if 𝑖 ≤ 𝑗
𝐓𝑖,𝑗 =
0 ≤ 𝑖, 𝑗 < 𝑁
(20)
1,
otherwise
5

Neural Networks 195 (2026) 108206

H. Liu et al.

Table 1
Comparison with the state-of-the-art methods under modality-default setting on SYSU-MM01 and RegDB datasets.
SYSU-MM01 (RGB:IR = 2:1)
All-search

RegDB (RGB:IR = 1:1)
Indoor-Search

Visible to Infrared

Infrared to Visible

Method

Venue

R-1

R-10

R-20

mAP

R-1

R-10

R-20

mAP

R-1

R-10

R-20

mAP

R-1

R-10

R-20

mAP

Zero-Pad (Wu et al., 2017)
cmGAN (Dai et al., 2018)
eDBTR (Ye et al., 2019)
D2 RL (Wang et al., 2019b)
AGW (Ye et al., 2021b)
DOTLA (Cheng et al., 2023b)
MBCCM (Cheng et al., 2023a)
MULT (He et al., 2024)
MCLNet (Hao et al., 2021)
TSME (Liu et al., 2022b)
SFANet (Liu et al., 2021b)
CM-NAS (Liu et al., 2021b)
FMCNet (Zhang et al., 2022a)
DART (Yang et al., 2022)
PMT (Lu et al., 2023a)
CAJ (Ye et al., 2021a)
DSCNet (Zhang et al., 2022b)
TOPLight (Yu et al., 2023)
AGMNet (Liu et al., 2023)
MPANet (Wu et al., 2021)
MAUM (Liu et al., 2022a)
TMD (Lu et al., 2023b)
CMTR (Liang et al., 2023)
CAJ+ (Ye et al., 2023)
ACD (Pan et al., 2024)
AGPI (Alehdaghi et al., 2025)
DPMF (Lu et al., 2024)

ICCV-2017
IJCAI-2018
TIFS-2020
CVPR-2019
TPAMI-2021
ACMMM-2023
ACMMM-2023
IJCV-2024
ICCV-2021
TCSVT-2022
TNNLS-2023
ICCV-2021
CVPR-2022
CVPR-2022
AAAI-2023
ICCV-2021
TIFS-2022
CVPR-2023
JSTSP-2023
CVPR-2021
CVPR-2022
TMM-2023
TMM-2023
TPAMI-2023
TIFS-2024
TIFS-2025
TNNLS-2025

14.80
26.97
27.82
28.90
47.50
50.36
53.14
64.77
65.40
64.23
65.74
60.80
66.34
68.72
67.53
69.90
73.89
66.76
69.63
70.58
71.68
68.81
65.45
71.48
74.44
72.23
76.11

54.12
67.51
67.34
70.60
84.39
89.02
89.61
–
93.33
95.19
92.98
92.10
–
96.39
95.36
95.70
96.27
96.23
96.27
96.21
–
93.08
94.47
96.23
–
97.04
96.40

71.33
80.56
81.34
82.40
92.14
95.92
96.74
–
97.14
98.73
97.05
96.80
–
98.96
98.64
98.50
98.84
98.70
98.82
98.80
–
96.84
98.16
98.71
–
–
98.66

15.95
31.49
28.42
29.20
47.65
47.36
48.16
59.23
61.98
61.21
60.83
58.90
66.29
66.29
64.98
66.90
69.47
64.01
66.11
68.24
68.79
63.96
62.90
68.15
71.17
70.58
71.55

20.58
31.63
32.46
–
54.17
53.47
55.21
65.34
72.56
64.80
71.60
68.00
68.15
72.52
71.66
76.30
79.35
72.89
74.68
76.74
76.97
76.31
71.46
78.36
78.98
83.45
82.29

68.38
77.23
77.42
–
91.14
92.24
91.44
–
96.98
96.92
96.60
94.80
–
97.84
96.73
97.90
98.32
97.93
97.51
98.21
–
97.28
97.16
98.36
–
98.62
98.64

85.79
89.18
89.62
–
95.98
97.84
95.83
–
99.20
99.31
99.45
97.90
–
99.46
99.25
99.50
99.77
99.28
99.14
99.57
–
98.91
99.22
99.78
–
–
99.73

26.92
42.19
42.46
–
62.97
61.73
61.98
71.46
76.58
71.53
80.05
52.40
74.09
78.17
76.52
80.40
82.68
76.70
78.30
80.95
81.94
74.52
76.67
81.98
82.75
84.25
84.94

14.80
–
34.62
43.40
70.05
85.63
83.79
89.95
80.31
87.35
76.31
82.80
89.12
83.60
84.83
85.00
85.39
85.51
88.40
82.80
87.87
87.04
88.11
85.69
84.71
89.03
88.83

54.12
–
58.96
66.10
87.28
94.08
95.83
–
92.70
97.10
91.02
95.10
–
–
–
95.50
–
94.99
95.10
–
–
95.49
–
95.45
–
98.19
97.38

71.33
–
68.72
76.30
92.04
95.49
97.82
–
96.03
98.90
94.27
97.70
–
–
–
97.50
–
96.70
96.94
–
–
97.57
–
97.54
–
–
98.35

15.95
–
33.46
44.10
66.37
76.71
77.87
82.09
73.07
76.94
68.00
79.30
84.43
75.67
76.55
79.10
77.30
79.95
81.45
80.70
85.09
81.19
81.66
79.70
83.28
83.89
81.02

20.58
–
34.21
43.40
68.83
82.91
82.82
90.78
75.93
86.41
70.15
81.70
88.38
81.97
84.16
84.80
83.50
80.65
85.34
83.70
86.95
83.54
84.92
84.88
87.18
87.91
88.88

68.38
–
58.74
66.10
83.69
92.33
95.73
–
90.93
96.39
85.24
94.10
–
–
–
95.30
–
92.81
94.56
–
–
94.56
–
95.86
–
97.15
97.62

85.79
–
68.64
76.30
88.35
94.90
96.89
–
94.59
98.20
89.27
96.90
–
–
–
97.50
–
96.32
97.48
–
–
96.84
–
97.74
–
–
98.63

26.92
–
32.49
44.10
64.45
74.97
76.74
82.25
69.49
75.70
63.77
77.60
83.86
73.78
75.13
77.80
75.19
75.91
81.19
80.90
84.34
77.92
80.79
78.55
84.72
83.04
81.86

CMBL

This work

76.36 97.94 99.04 71.64 81.57

98.05

99.50

83.95

89.80 96.97

98.77 84.86 87.52

96.74

98.22

84.34

Table 2
Comparison with state-of-the-art methods in the context of imbalance settings on the SYSU-MM01 and RegDB datasets.
SYSU-MM01

RegDB

Ratio = 2:1

Imbalance Ratio = 4:1

Imbalance Ratio = 8:1

Ratio = 1:1

Imbalance Ratio = 2:1

Imbalance Ratio = 4:1

Method

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

AGW (Ye et al., 2021b)
MCLNet (Hao et al., 2021)
SFANet (Liu et al., 2021b)
PMT (Lu et al., 2023a)
AGMNet (Liu et al., 2023)
MMN (Zhang et al., 2021)
FMI (Tian et al., 2021)
MAUM− (Liu et al., 2022a)
MAUM (Liu et al., 2022a)
DSCNet (Zhang et al., 2022b)
DEEN (Zhang & Wang, 2023)

47.50
65.40
65.74
67.53
69.63
70.60
60.00
56.00
61.60
73.89
74.70

47.65
61.98
60.83
64.98
66.11
66.90
58.80
53.40
59.00
69.47
71.80

36.30(↓11.20)
51.88(↓13.52)
50.22(↓15.52)
45.89(↓21.64)
57.14(↓12.49)
57.30(↓13.30)
44.70(↓15.30)
40.30(↓15.70)
47.30(↓14.30)
50.22(↓23.67)
63.61(↓11.09)

35.54(↓12.11)
48.58(↓13.37)
44.27(↓16.46)
40.52(↓24.46)
54.30(↓11.81)
54.60(↓12.30)
45.60(↓13.20)
38.80(↓16.60)
46.90(↓12.10)
49.68(↓19.79)
61.67(↓10.13)

31.32(↓16.68)
47.67(↓17.73)
47.31(↓18.83)
42.66(↓24.87)
52.40(↓17.23)
54.70(↓15.90)
39.40(↓20.60)
36.10(↓19.90)
42.20(↓19.40)
47.39(↓26.50)
55.93(↓18.77)

32.24(↓15.41)
44.97(↓17.01)
43.02(↓17.81)
38.54(↓26.44)
50.10(↓16.01)
50.90(↓16.00)
41.50(↓17.30)
35.70(↓17.70)
45.20(↓13.80)
43.39(↓26.08)
52.94(↓18.86)

70.05
80.30
76.31
84.83
88.40
91.60
73.20
77.70
83.39
85.39
91.10

66.37
73.07
68.00
76.55
81.45
84.10
71.60
70.44
78.75
77.30
85.10

57.52(↓12.53)
56.94(↓23.36)
50.15(↓26.16)
64.16(↓20.67)
70.14(↓18.26)
74.50(↓17.10)
52.14(↓21.06)
57.24(↓20.46)
68.89(↓14.50)
63.50(↓21.89)
80.69(↓10.41)

52.70(↓13.67)
52.91(↓20.16)
45.24(↓22.76)
56.35(↓20.20)
65.56(↓15.89)
68.42(↓15.68)
48.66(↓22.94)
50.78(↓19.64)
64.97(↓13.78)
54.88(↓22.42)
73.16 (↓11.94)

38.54(↓31.51)
45.68(↓34.77)
46.22(↓30.09)
53.64(↓31.19)
61.74(↓26.66)
64.26(↓27.34)
42.46(↓30.94)
42.14(↓35.56)
54.94(↓28.45)
52.98(↓32.41)
63.11(↓27.99)

37.50(↓28.87)
42.61(↓30.46)
40.86(↓27.14)
48.26(↓28.29)
55.62(↓25.83)
56.88(↓27.22)
38.69(↓32.91)
36.77(↓33.67)
50.38(↓28.37)
47.49(↓29.81)
57.20(↓27.90)

CMBL (Our)

76.36

71.64

70.62(↓5.74)

65.72(↓5.92)

66.46(↓9.90)

62.73(↓8.91)

89.80

84.86

84.52(↓5.28)

78.82(↓6.04)

71.34(↓18.46)

64.64(↓20.22)

cameras, obtaining infrared images of people poses signiﬁcant challenges. Therefore, to further assess the performance of various methods
in real-world scenarios with modality imbalance, we intentionally omit
infrared images from the original dataset to create an extreme imbalance
condition. Speciﬁcally, within the SYSU-MM01 dataset, where the initial ratio of visible to infrared images stands at 2:1, for each identity, we
randomly remove some infrared modality images to establish ratios of
4:1 and 8:1, respectively. Similarly, for the RegDB dataset, which originally has a balanced ratio of 1:1 between visible and thermal images,
we adjust by excluding some infrared iamges for each identity to modify
the ratio to 2:1 and 4:1, respectively. This approach allows us to simulate and examine the resilience of diﬀerent methods under signiﬁcantly
skewed modality distributions.

Table 3
Comparison with the state-of-the-art methods on LLCM dataset. ‘-’ that the result
was not reported in the original publication.
LLCM
IR to VIS
Method

R-1

DDAG (Ye et al., 2020b) (ECCV22)
40.3
AGW (Ye et al., 2021b) (PAMI23)
43.6
LbA (Park et al., 2021) (ICCV21)
43.8
CAJ (Ye et al., 2021a) (ICCV21)
48.8
DART (Yang et al., 2022) (CVPR21)
52.2
MMN (Zhang et al., 2021) (ACM21)
52.5
DEEN (Zhang & Wang, 2023) (CVPR23) 54.9
MBCNet(Liu et al., 2024) (TCSS24)
55.4
HOS-Net (Qiu et al., 2024) (AAAI24)
56.4
CMBL

VIS to IR

R-10 R-20 mAP R-1

R-10 R-20 mAP

71.4
74.6
78.2
79.5
80.7
81.6
84.9
86.2
–

79.2
81.5
84.3
85.3
87.1
88.5
90.3
91.1
–

79.6
82.4
86.6
85.3
87.0
88.4
90.9
90.8
–

48.4
51.8
53.1
56.6
59.8
58.9
62.9
62.4
63.2

48.0
51.5
50.8
56.5
60.4
59.9
62.5
60.3
64.9

86.1
87.9
91.1
90.9
91.9
93.6
94.7
95.8
–

52.3
55.3
55.6
59.8
63.2
62.7
65.8
55.2
67.9

3.5.2. Implementation details
The method is implemented using PyTorch and trained on an 80GB
NVIDIA Tesla A100 GPU. We apply random channel exchangeable

58.1 89.2 92.0 64.2 66.3 92.6 96.4 68.2

6

Neural Networks 195 (2026) 108206

H. Liu et al.

augmentation, random cropping, random ﬂipping, and random erasing
methods for data augmentation. Following the approach in Ye et al.
(2021b), we schedule the learning rate with a warm-up strategy. We
utilize the SGD optimizer for optimization, with the momentum parameter set to 0.9.

et al., 2021b), and MAUM (Liu et al., 2022a), alongside our CMBL. Furthermore, we retrained additional methods (Lu et al., 2023a; Tian et al.,
2021; Ye et al., 2021b; Zhang et al., 2022b; Zhang & Wang, 2023; Zhang
et al., 2021) using their open-source codes to gather and report their results.
SYSU-MM01. As illustrated in Table 2, we observe a signiﬁcant decline in the performance of all methods as the imbalance ratio increases.
Speciﬁcally, the Rank-1 accuracy of DSCNet (Zhang et al., 2022b) drops
from 73.89 % to 50.22 % when the imbalance ratio reaches 4:1, deteriorating further to 47.39 % as the ratio escalates to 8:1. This trend underscores the substantial challenge posed by the imbalance modality issue
within VI-Re-ID tasks.
Furthermore, it’s noteworthy that at a 2:1 imbalance ratio, DEEN
(Zhang & Wang, 2023) marginally outperforms our CMBL by a mere
0.16 % in mAP. However, as the imbalance ratio widens to 4:1, our
CMBL begins to eclipse DEEN (Zhang & Wang, 2023) in performance.
This advantage becomes even more pronounced at an 8:1 imbalance
ratio, where CMBL surpasses DEEN (Zhang & Wang, 2023) by a significant margin of 10.53 % in Rank-1 accuracy and 9.79 % in mAP value.
These ﬁndings highlight the exceptional robustness of our method in
navigating through extreme modality imbalance scenarios, showcasing
its superior adaptability and eﬀectiveness in cross-modality person ReID.
RegDB. From Table 2, it is observed that in the initial balanced setting, our proposed method does not outperform the leading model DEEN
(Zhang & Wang, 2023), with a Rank-1 matching rate that is 1.30 % lower
and an mAP value that is 0.24 % lower. However, when the imbalance
ratio is adjusted to 2:1, our method demonstrates increased robustness,
outperforming DEEN (Zhang & Wang, 2023) by approximately 4.0 %
across two evaluation metrics. This further suggests that the robustness
of our proposed method is attributed to the comprehensive optimization
of sparse modalities.

3.5.3. Modality-default setting results
We compare our proposed method with numerous state-of-the-arts
on SYSU-MM01 and RegDB datasets under default setting. We focus on
using a single query and do not employ any post-processing techniques
such as re-ranking or part-based features. The results are reported in
Table 1.
SYSU-MM01. As demonstrated in Table 1, our proposed CMBL
framework establishes new state-of-the-art performance across both
evaluation protocols on the SYSU-MM01 dataset. Speciﬁcally, in the
challenging all-search mode, CMBL achieves remarkable Rank-1 accuracy of 76.36 % and mAP of 71.64 %, representing absolute improvements of +1.92 % and +0.47 % over the previous best method (DPMF
(Lu et al., 2024)). For the indoor-search mode, our method attains exceptional performance with 81.57 % Rank-1 accuracy and 83.95 % mAP,
maintaining consistent superiority. These signiﬁcant advancements not
only validate our approach’s eﬀectiveness in cross-modality alignment
but also demonstrate its enhanced discriminative capability for ﬁnegrained person re-identiﬁcation.
RegDB. In this section, We evaluate our model in the balanced
RegDB dataset, as illustrated in Table 1. Similar to the results obtained
on SYSU-MM01, our approach signiﬁcantly outperforms the existing
methods in both evaluation modes. In the visible-to-infrared search
mode, the proposed CMBL achieves a Rank-1 of 89.80 % and an mAP
of 84.86 %. In the infrared-to-visible search mode, the proposed CMBL
achieves a Rank-1 of 87.52 % and an mAP value of 84.34 %. These
experimental results demonstrate the outstanding performance of our
method, highlighting its ability to learn discriminative features and afﬁrming the eﬀectiveness of our concept of class contrast similarity learning.
However, it’s important to note that the improvements our method
oﬀers over other SOTA techniques appear marginal. This observation
could be partly attributed to the limited size of the RegDB dataset. Additionally, the balanced evaluation setting we employed may not fully
capture the nuances that underscore the advantages of our method. This
context suggests that while our approach is eﬀective, its full potential
might be more accurately gauged in a more varied or larger dataset
scenario.
LLCM. We compare We conduct comprehensive comparisons between our approach and state-of-the-art methods on the challenging LLCM dataset, with quantitative results summarized in Table 3.
The experimental results demonstrate the superior performance of our
proposed CMBL framework across all evaluation metrics. Speciﬁcally,
CMBL achieves remarkable Rank-1 accuracy of 58.1 % and mAP of
64.2 % in the more challenging IR-to-VIS modality, while attaining
66.3 % Rank-1 accuracy and 68.2 % mAP in the VIS-to-IR modality.
Notably, while the second-best performer HOS-Net (Qiu et al., 2024)
delivers competitive results through sophisticated multi-branch architectures and complex attention mechanisms, this comes with substantial
computational overhead during both training and inference. In contrast,
our CMBL framework achieves superior performance (average +2.4 %
improvement across metrics) with signiﬁcantly reduced model complexity, demonstrating both the eﬃciency and eﬀectiveness of our proposed
approach.

3.6. Ablation study
Eﬀectiveness of each component of CBML. This study conducts
an ablation analysis to assess the individual impacts of the various components within our CBML framework. During the experimental phase,
the general settings remained unchanged, except for the module under evaluation. As shown in Table 4, each component contributes to
performance improvement to varying extents. Beginning with the baseline, the direct integration of the SSP module into the model yields a
slight performance enhancement, although it falls short of being satisfactory. However, upon incorporating the cross-distribution alignment
loss (𝑐𝑟𝑜𝑠𝑠 ), a notable improvement in performance is observed (from
65.39 % to 72.05 % in Rank-1 accuracy). This highlights the eﬀectiveness of the proposed method in mitigating modality discrepancies and
data imbalance, enabling further optimization of the sparse modality.
Furthermore, the integration of CACS oﬀers additional performance enhancements. This improvement is attributed to the angular constraint
imposed by 𝑐𝑙𝑎𝑠𝑠 , which aims to minimize intra-class distances while
simultaneously maximizing inter-class separability.
Which stage to insert the SSP? The proposed SSP module is designed to be ﬂexibly integrated at any stage within the backbone network. Using a standard ResNet-50 model, which includes ﬁve residual
blocks denoted as Res 01234, we experiment with incorporating SSP at
diﬀerent layers. Speciﬁcally, Res 0 signiﬁes that SSP is applied after the
ﬁrst residual block, and Res 01 indicates SSP’s application after the ﬁrst
and second residual blocks. As illustrated in Table 5, inserting SSP from
layers 0 to 3 incrementally enhances model performance, with optimal
results observed at Res 123. This enhancement is attributed to the feature
statistics within Res 123, which are rich in modality-speciﬁc information.
Conversely, incorporating SSP after layer 4, as in ‘Res 1234’, leads to a
collapse in the training process. This phenomenon is likely because layer
4 is pivotal in capturing label-speciﬁc information, and destabilizing the

3.5.4. Modality-imbalance setting results
In this section, the performance of our proposed method is evaluated
across various modality-imbalance scenarios. Results are presented in
Table 2. For comparison, we replicated the following well-known methods: AGMNet (Liu et al., 2023), MCLNet (Hao et al., 2021), SFANet (Liu
7

Neural Networks 195 (2026) 108206

H. Liu et al.

Table 4
Eﬀectiveness of the proposed components on the SYSU-MM01 dataset under the all-search single-shot
mode.
Components
Base.
✓
✓
✓
✓
✓
✓

cross

SSP

CACS

✓
✓
✓
✓
✓

✓
✓
✓

✓

All-Search (Imbalance Ratio = 2:1)

All-Search (Imbalance Ratio = 8:1)

R-1

R-10

R-20

mAP

R-1

R-10

R-20

mAP

63.79
65.39
67.03
72.05
70.47
76.36

92.94
93.36
95.45
96.64
96.67
97.47

97.33
97.88
98.56
99.21
98.03
99.46

59.87
60.82
62.87
68.46
66.66
71.64

40.20
48.86
42.92
65.92
58.65
70.62

84.66
87.01
86.72
93.04
93.22
96.48

90.04
92.33
90.13
97.23
95.14
98.01

34.39
46.96
39.24
60.24
56.72
65.72

Table 5
Ablation study on where to apply SSP on the SYSU-MM01 dataset
under all-search mode.
Imbalance Ratio = 2:1

Imbalance Ratio = 8:1

Model

R-1

mAP

R-1

mAP

+SSP(Res0)
+SSP(Res01)
+SSP(Res012)
+SSP(Res0123)
+SSP(Res12)
+SSP(Res123)
+SSP(Res1234)

72.02
73.34
75.68
75.49
75.25
76.64
Nan

67.56
68.45
69.89
70.56
70.62
71.64
Nan

62.77
64.02
64.44
68.77
66.73
70.62
Nan

57.56
60.56
61.96
63.56
64.56
65.76
Nan

Fig. 6. Visualization in single-modal scenarios for SMOTE (Chawla et al., 2002)
and SSP: a comparative analysis with a randomly selected mini-batch of IR
modality samples (32 samples). Best viewed in color.

Table 6
The eﬀectiveness of cross-distribution alignment components on the SYSUMM01 dataset in both all-search and indoor-search mode.
Components
Index
1
2
3
4
5
6
7
8

𝐷 (𝑣, 𝑡)

𝐷 (𝑣, 𝑝)

All-Search

Indoor-Search

𝐷 (𝑡, 𝑝)

R-1

mAP

R-1

mAP

✓

65.39
69.22
69.03
66.54

60.82
65.77
65.87
62.48

74.86
78.22
78.92
75.39

77.96
79.44
79.24
78.15

✓
✓
✓

70.65
71.05
70.44
72.05

67.12
67.42
67.74
68.46

75.66
76.11
76.19
78.92

78.24
78.94
78.42
80.24

✓
✓

✓
✓
✓

✓
✓
✓

feature statistics at this juncture could disrupt the inherent structure of
the label space.
Eﬀectiveness of each component of 𝑐𝑟𝑜𝑠𝑠 . The cross-distribution
alignment loss 𝑐𝑟𝑜𝑠𝑠 comprises three components: the MMD between
RGB and IR features 𝐷 (𝑣, 𝑡), the MMD between RGB and perturbed
features 𝐷 (𝑣, 𝑝), and the MMD between IR and perturbed features
𝐷 (𝑡, 𝑝). To clarify the of each component on the overall performance,
we executed targeted experiments for each component, with the outcomes detailed in Table 6. The results demonstrate that every component contributes to performance enhancement, particularly 𝐷 (𝑣, 𝑡) and
𝐷 (𝑣, 𝑝). The signiﬁcance of 𝐷 (𝑣, 𝑡) and 𝐷 (𝑣, 𝑝) lies in their ability to
align the cross-modality distributions, thereby diminishing the modality
gap. In contrast, the signiﬁcance of 𝐷 (𝑡, 𝑝) lies in their ability to align
the intra-modality distributions, thereby improving the discriminability
of features. Notably, when all components are integrated for optimization, we observe a peak in performance, underscoring the individual and
collective eﬃcacy of each component.
Eﬀectiveness of each component of CACS. We gradually incorporate each component of CACS based on the benchmark to conduct
an ablation analysis. The results are presented in Table 7. It is evident
that each component consistently outperforms the baseline, enhancing
its performance. Speciﬁcally, employing only 𝑖𝑛𝑡𝑟𝑎 enhances the allsearch mode’s Rank-1 accuracy from 72.05 % to 73.45 % and the mAP
from 68.46 % to 68.85 %. Likewise, using 𝑖𝑛𝑡𝑒𝑟 alone boosts Rank-1 accuracy from 72.05 % to 74.79 % and mAP from 68.46 % to 69.21 %.

Fig. 7. Parameter analysis of 𝜔1 and 𝜔2 . Experiments are conducted under the
all-search mode with imbalance ratio = 2:1 on the SYSU-MMM01 dataset.
Table 7
Component analysis of Class-Aware Contrast Similarity (CACS).
𝑖𝑛𝑡𝑟𝑎 represents the intra-class contrast learning, 𝑖𝑛𝑡𝑒𝑟 denotes
inter-class contrast learning.
Imbalance Ratio = 2:1

Imbalance Ratio = 8:1

Methods

R-1

mAP

R-1

mAP

w/o CACS
+ 𝑖𝑛𝑡𝑟𝑎
+ 𝑖𝑛𝑡𝑒𝑟
+ 𝑖𝑛𝑡𝑟𝑎 + 𝑖𝑛𝑡𝑒𝑟

72.05
73.45
74.79
76.36

68.46
68.85
69.21
71.64

65.92
66.83
68.65
70.62

60.24
61.24
63.77
65.76

Notably, in contrast to intra-contrast learning, the inter-contrast learning contributes more to the performance improvement across diﬀerent
imbalance ratios. This highlights the eﬀectiveness of minimizing the
similarity across diﬀerent classes, enhancing the distinctiveness between
them.
8


codex
前三篇已经抽到方法主体，但有些输出被系统截断了。为了不丢方法链条，我会对被截断的论文再按方法章节做定向截取，同时继续读后三篇。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Text-Guided Cross-Modal Alignment with Attribute and
Contour Prototypes for Visible-Infrared Person
Re-Identification
YONG TAO and XINMING ZHANG, School of Computer Science and Technology, University of
Science and Technology of China, Hefei, China
Visible-infrared person re-identification (VI-ReID) aims to match pedestrian images captured under visible
and infrared modalities, which suffer from significant domain discrepancies. Existing approaches either
synthesize cross-modal images or learn modality-invariant representations, yet often encounter semantic
degradation or limited alignment capacity. Recent vision-language models leverage textual semantics for
modality bridging; however, CLIP-based frameworks typically rely on learnable token proxies with limited
expressiveness. In this article, we propose a novel semantic-driven framework that explicitly generates rich,
modality-agnostic textual descriptions from images as alignment cues. Specifically, we design a dual-branch
Text Semantic Generation Module that includes: (1) an Attribute-Aware text description Generation module
using prompt-based templates to capture modality-invariant identity cues, and (2) a Contour-Aware text
prompt Module that provides complementary structural information often missing in textual descriptions.
To reconcile semantic heterogeneity, a Text Re-definition Module (TRM) fuses instance-level and class-level
semantics into unified representations, enabling fine-grained alignment with image features. Furthermore, we
construct category-level textual prototypes as global semantic anchors to enhance cross-modal consistency.
Extensive experiments on two standard VI-ReID benchmarks demonstrate that our method achieves superior
performance, validating its effectiveness in semantic-guided modality alignment.
CCS Concepts: • Computing methodologies → Matching; Visual content-based indexing and retrieval;
Additional Key Words and Phrases: Person re-identification, Vision-Language Learning, infrared person
re-identification
ACM Reference format:
Yong Tao and Xinming Zhang. 2026. Text-Guided Cross-Modal Alignment with Attribute and Contour
Prototypes for Visible-Infrared Person Re-Identification. ACM Trans. Multimedia Comput. Commun. Appl. 22,
4, Article 113 (March 2026), 18 pages.
https://doi.org/10.1145/3796720

1

Introduction

Person re-identification (ReID) plays a crucial role in public safety and urban surveillance, as
it enables the tracking and localization of suspects by analyzing images captured from city-wide
surveillance cameras. Early ReID studies [23] mainly focused on pedestrian retrieval between
Authors’ Contact Information: Yong Tao, School of Computer Science and Technology, University of Science and Technology
of China, Hefei, China; e-mail: taoguandiu@163.com; Xinming Zhang (corresponding author), School of Computer Science
and Technology, University of Science and Technology of China, Hefei, China; e-mail: xinming@ustc.edu.cn.

This work is licensed under Creative Commons Attribution International 4.0.
© 2026 Copyright held by the owner/author(s).
ACM 1551-6865/2026/3-ART113
https://doi.org/10.1145/3796720
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

113:2

Y. Tao and X. Zhang

RGB images captured by daytime cameras. However, such settings are inadequate for crossmodal retrieval tasks involving RGB images from daytime cameras and infrared (IR) images
from nighttime cameras. The visible-infrared person re-identification (VI-ReID) task aims
to retrieve all images containing the same identity from a gallery of images captured under one
modality (e.g., nighttime IR), given a probe image captured under the opposite modality (e.g.,
daytime RGB). In recent years, numerous works [2, 3, 14, 25] have attempted to tackle this problem
by narrowing the modality gap between RGB and IR images, which is widely regarded as one of
the key challenges limiting the performance of cross-modal retrieval.
Mitigating the modality discrepancy between daytime RGB images and nighttime IR images
remains a fundamental challenge in visual identity recognition (VI-ReID). Existing approaches
can be broadly categorized into two paradigms. The first line of work leverages generation-based
methods [4, 18, 19], which learn a mapping between RGB and IR domains via image translation,
aiming to align different modalities within a shared distribution space. While effective to some
extent, these methods inevitably introduce generation artifacts. Specifically, generated images often
suffer from distributional shifts compared to real data, and their perceptual quality is generally
suboptimal, thereby limiting overall retrieval performance. The second category includes modalityinvariant representation learning approaches [5–7, 10], which aim to extract shared, identitydiscriminative features across modalities solely through visual supervision. Despite improvements
over generation-based methods, these approaches tend to overlook high-level semantic cues, as
they rely purely on image-level features. As a result, their ability to bridge the intrinsic modality
gap remains constrained [12]. Recent efforts have explored the integration of vision-language
models (VLMs), such as CLIP [15], into VI-ReID. These methods [1, 9, 20] attempt to leverage
implicit textual semantics derived from image features as auxiliary alignment signals. However,
most of these models rely on a limited number of learnable semantic tokens to represent latent
textual information. Such token representations lack sufficient semantic richness and generalization
capacity, thereby hindering their effectiveness in bridging modality gaps.
To overcome the aforementioned limitations, we propose a novel framework (Figure 1) that
incorporates an image-to-text generation paradigm to construct discriminative and modalityinvariant textual descriptions, thereby replacing traditional learnable token vectors. The proposed
framework consists of two complementary modules: (1) an attribute-aware text description
generation (ATG), which synthesizes identity-specific textual descriptions based on pre-defined
attribute templates (e.g., gender, clothing style) while deliberately omitting modality-sensitive
information such as color, and (2) a contour-aware text prompt (CTP) module, which captures
structural semantics (e.g., body shape, posture) to compensate for the lack of fine-grained contour
information typically absent in large-scale captioning models. These two modules jointly generate
semantic descriptions that are intentionally free from modality bias, serving as reliable anchors for
cross-modal alignment.
However, the ATG generates instance-level descriptions, while the CTP yields class-level contour
representations in the form of learnable textual vectors, making their direct integration nontrivial. To
bridge this gap, we introduce a text re-definition module (TRM) that serves two key functions:
(1) fusing attribute-aware and contour-aware representations into a unified and semantically
enriched textual prototype and (2) further narrowing the modality gap between generated textual
and visual features via adaptive refinement. The final class-level textual prototypes derived from
TRM are then adopted as shared semantic anchors to align both RGB and IR features within a
unified embedding space. This design enables effective modality-agnostic alignment, substantially
reducing cross-modal discrepancies and significantly enhancing retrieval performance on VI-ReID
benchmarks. Our main contributions are summarized as follows:

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID

113:3

Fig. 1. Comparison of different person VI-ReID frameworks. (a) While visual encoders primarily extract
modality-shared identity features, they often overlook valuable semantic cues provided by other modalities.
(b) “CSDN” [25], a CLIP-based VI-ReID variant, uses learnable feature vectors to represent textual semantics,
providing alignment cues for cross-modal matching. (c) Our proposed method introduces an image-to-text
generator to produce rich, modality-invariant textual descriptions for images from different modalities, which
serve as semantic cues for cross-modal alignment.

— Modality-invariant text semantic generation: We introduce a dual-branch text generation
framework that produces attribute-aware and contour-aware descriptions from pedestrian
images. The attribute branch employs modality-agnostic templates to encode identity-related
semantics, while the contour branch captures structural cues such as body shape and pose,
which are often overlooked in existing approaches.
— Unified text representation and fine-grained alignment: A TRM is proposed to integrate heterogeneous semantic descriptions into a unified representation. This unified text is further aligned
with visual features through a fine-grained interaction mechanism, effectively narrowing the
modality gap between RGB and IR modalities.
— Prototype-guided semantic alignment: We derive category-level textual prototypes from the
unified descriptions, which act as modality-shared semantic anchors to guide the alignment
of heterogeneous visual features in a prototype-centered manner.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

113:4

Y. Tao and X. Zhang

— Superior performance and generalization: Extensive experiments on two standard VI-ReID
benchmarks demonstrate that our method significantly outperforms existing approaches,
validating both its effectiveness and generalizability.
2

Related Work

2.1

VI-ReID

VI-ReID aims to match pedestrian images captured under different modalities, typically visible
light during the daytime and IR imaging at night. Given a query image from one modality (e.g.,
visible), the task is to retrieve corresponding pedestrian images from a gallery composed of another
modality (e.g., IR), often under varying viewpoints and illumination conditions.
Researchers broadly categorize existing approaches to VI-ReID into two main areas: unified
representation learning and modality-invariant feature learning. The first category seeks to transform heterogeneous data into a unified modality space, thus alleviating the modality gap at the
input level. For instance, Wang et al. [18] proposed a pixel-level alignment strategy to synthesize
virtual IR images from RGB inputs, allowing direct matching with real IR samples. While effective
in reducing modality discrepancy, generative models such as these may introduce visual artifacts
or result in the loss of fine-grained semantic cues due to imperfect translation.
The second category aims to extract modality-invariant features that are directly comparable
across modalities, bypassing the need for pixel-level synthesis. Representative work includes Ye et al.
[22], who proposed a dual-channel architecture to learn shared features using a common backbone,
together with a bi-directional center-constrained top-ranking loss that enhances cross-modal
discrimination. However, these methods often overlook the distributional biases that persist even in
modality-invariant features. Specifically, identity features from different modalities may still exhibit
inconsistent distribution patterns, which impairs class-level alignment in the embedding space.
To address this overlooked issue, recent studies have begun to explore distribution-level alignment. In this work, we propose a modality distribution bias loss, which explicitly regularizes the
intra-class feature distributions across modalities. This approach reduces modality-induced variance and promotes more consistent alignment, leading to improved performance in cross-modality
retrieval tasks.
2.2

Vision-Language Learning

Recent advances in vision-language pretraining [16], image–text retrieval [13], and video captioning
[17]. Inspired by these successes, increasing efforts have been devoted to incorporating VLMs into
person ReID, especially in text–image-based scenarios [8, 21, 27]. These methods typically aim to
enhance retrieval performance by introducing strong semantic alignment cues between text and
image modalities. However, directly applying CLIP to conventional image-based ReID presents
significant challenges, including the lack of suitable textual supervision and the substantial modality
gap between visual and language features. To overcome these limitations, CLIP-ReID [12] introduces
prompt learning [26], which transforms identity labels into learnable text embeddings that serve
as semantic supervision. These textual embeddings guide the visual encoder during training,
resulting in more discriminative features. TF-ReID [24] further enhances CLIP’s applicability to
video-based ReID by leveraging its memory module for capturing temporal semantics. In the
visible-infrared (VI-ReID) setting, most existing approaches follow the CLIP-ReID paradigm and
introduce specific modules for cross-modal alignment. For example, Wu et al. [9] simultaneously
optimize visible-infrared alignment (VIA) and visual-text reasoning (VTR), improving both
feature fusion and cross-modal semantic grounding. VIA enhances local–global feature correlation
via modality-guided fusion, while VTR leverages dual-layer reasoning and selective pooling to focus
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID

113:5

the model’s attention on semantically important regions. Despite these promising developments,
current works still primarily rely on implicit feature alignment and fail to exploit explicit semantic
descriptions derived from images fully. Furthermore, the inherent semantic gap between image and
text modalities remains underexplored. These challenges motivate our method, which introduces
structured textual representations that are both discriminative and modality-invariant, effectively
narrowing the semantic gap and improving VI-ReID performance.
3
3.1

Method
Framework Overview

𝑁 containing 𝑁 paired samples of visible (RGB) images 𝑥 𝑣 ,
Given a training dataset D = {𝑥𝑖𝑣 , 𝑥𝑖𝑟 , 𝑦𝑖 }𝑖=1
𝑖
𝑟
IR images 𝑥𝑖 , and their associated identity labels 𝑦𝑖 ∈ {1, . . . , 𝐶}, where 𝐶 denotes the total number
of identity classes, our framework introduces a text-enhanced cross-modal alignment strategy to
boost VI-ReID.
To exploit semantic priors, we first leverage a VLM to generate image-conditioned textual
𝑁 , where 𝑡 𝑣 and 𝑡 𝑟 describe the fine-grained appearance attributes of 𝑥 𝑣
descriptions T = {𝑡𝑖𝑣 , 𝑡𝑖𝑟 }𝑖=1
𝑖
𝑖
𝑖
𝑟
and 𝑥𝑖 , respectively. These descriptions capture detailed semantics such as clothing, carried objects,
or color hints, serving as auxiliary supervision to the visual encoders. We further observe that
LLM-generated descriptions implicitly reflect contour-aware cues, which are crucial for identity
alignment under modality variations. To explicitly model such structure, we propose a CTP module
that constructs contour-level textual descriptions T𝑠 = {𝑡𝑖𝑠 }𝐶𝑖=1 for each identity class. Unlike
instance-level descriptions T, these contour prompts emphasize shared structural patterns within
each identity class.
To effectively integrate both instance-level and class-level textual semantics, we introduce a TRM,
𝑁 .
which fuses 𝑡𝑖𝑣 , 𝑡𝑖𝑟 , and 𝑡 𝑦𝑠 𝑖 into unified, identity-sensitive textual representations T𝑜 = {𝑡𝑖𝑣𝑜 , 𝑡𝑖𝑟𝑜 }𝑖=1
This joint representation not only preserves detailed attribute cues but also inherits discriminative
identity patterns from Contour-aware prompts. Furthermore, we compute the Text Prototype
Discriminative Feature Set T𝑐 = {𝒇𝑖𝑡 }𝐶𝑖=1 by aggregating the class centers of the redefined textual
features. These prototypes serve as modality-invariant anchors that guide the alignment of visual
features across modalities. By enforcing cross-modal features to align with corresponding text
prototypes, we establish a semantic bridge between visible and IR domains, enabling robust identity
matching.

3.2

ATG Module

Pedestrian attributes serve as crucial semantic cues for cross-modal person ReID. To explicitly
incorporate such discriminative information, as illustrated in Figure 3, we design a structured
attribute-based text generation scheme that converts visual attributes into formalized textual
descriptions. Specifically, we construct a pedestrian attribute template 𝜏, grounded in established
attribute taxonomies, to standardize the descriptive process. The template is defined as:
“A {age} {gender} is {action}, wearing {upper clothes}, {lower-body clothes},
and {shoe type}. The {gender} has {short or long} hair, {yes or no} the person is
wearing glasses.”

To populate this template with image-specific content, we develop a parametrized question set
corresponding to each attribute slot. For instance, we design prompts such as: “Q1: The person is
young, a teenager, an adult, or elderly?” “Q2: The person is male or female?” “Q3: The person is
walking or standing?” and so on for clothing style, hair length, and accessories. We employ a visual
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

113:6

Y. Tao and X. Zhang

question-answering paradigm based on the BLIP model to extract the relevant attribute values.
For each input image, the BLIP model infers answers conditioned on both the image content and
our designed questions. The collected answers are then assembled into complete attribute-based
descriptions following the template 𝜏, yielding textual representations 𝑡𝑖𝑣 and 𝑡𝑖𝑟 for the 𝑖th visible and
IR images, respectively. This template-driven strategy brings several advantages. First, it enforces
a unified syntactic and semantic structure across all generated descriptions, which mitigates the
noise and diversity commonly observed in free-form language generation. Second, by avoiding
direct references to low-level color cues—which are often modality-specific and unreliable in IR
imagery—the generated descriptions focus on modality-invariant attributes, such as age, gender,
action, clothing type, and accessories. These carefully selected attributes offer stable semantic
anchors for visual–textual alignment across modalities.
𝑁 , which encodes
As a result, we construct an attribute-enhanced textual dataset T = {𝑡𝑖𝑣 , 𝑡𝑖𝑟 }𝑖=1
structured and fine-grained semantic information. This representation not only facilitates better
visual–textual alignment but also introduces human-interpretable and modality-agnostic cues into
the ReID process.
3.3

Contour-Aware Text Prompt Module

While the attribute-aware textual descriptions provide valuable cues for pedestrian recognition,
they inherently lack explicit contour-related semantics. Moreover, describing fine-grained sketch
information through natural language is non-trivial. To address this issue, as illustrated in Figure 4,
we draw inspiration from the CoOP [26] approach and aim to learn a set of class-discriminative
prompt tokens that can implicitly model pedestrian contour information.
Specifically, we employ the topologically aware human parsing framework SCHP [11] to decompose the input pedestrian image 𝑥𝑖 (where 𝑥𝑖 can be either 𝑥𝑖𝑣 or 𝑥𝑖𝑟 ) into 20 semantic categories,
including background, hair, upper clothes, dress, coat, pants, skirt, face, left arm, and others. To
isolate sketch-relevant structures (e.g., body shape and pose) while suppressing clothing-based
variations, we construct a binary mask M𝑠𝑖 by setting all clothing-related regions to 0 and all
remaining semantic regions to 1. Using this mask, we generate a sketch-style image 𝑥𝑖𝑠 by retaining
sketch-relevant pixels and replacing the rest with a constant background value:
𝑥𝑖𝑠 = M𝑠𝑖

𝑥𝑖 + (I − M𝑠𝑖 )

255,

(1)

where denotes the Hadamard (element-wise) product, and 𝑥𝑖𝑠 represents the sketch-style version
of the original image 𝑥𝑖 .
To enable textual modeling of sketch semantics, we adopt a learnable prompt tuning strategy
inspired by CLIP-ReID [4]. We define an identity-related textual template: 𝒕𝑖𝑠 = “A photo of a
[𝑋 1 ] [𝑋 2 ] [𝑋 3 ] . . . [𝑋𝑀 ] person”, where each [𝑋 ] 𝑖 is a learnable token and 𝑀 is the number of
tokens. These tokens are optimized to capture contour-aware identity semantics based on visual
guidance from 𝑥𝑖𝑠 .
Following CLIP-ReID training, we encode the image 𝑥𝑖𝑠 and the corresponding prompt t𝑠𝑖 using
the visual encoder 𝐸 𝑣 and the text encoder 𝐸𝑡 to obtain feature embeddings:
𝒇𝑖𝑠,𝑣 = 𝐸 𝑣 (𝑥𝑖𝑠 ),

𝒇𝑖𝑠,𝑡 = 𝐸𝑡 (𝒕𝑖𝑠 ).

(2)

To ensure that the learned prompt tokens t𝑠𝑖 are identity-discriminative, we employ a symmetric
image–text alignment objective consisting of an image-to-text loss and a text-to-image loss:
L𝑎𝑙𝑖𝑔𝑛 = L𝑖2𝑡 + L𝑡 2𝑖 ,

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

(3)

Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID

113:7


𝑁𝑏
exp 𝜓 f𝑖𝑠,𝑣 , f𝑖𝑠,𝑡
1 Õ
 
 ,
L𝑖2𝑡 = −
log Í
𝑁𝑏
𝑁𝑏 𝑖=1
exp 𝜓 f 𝑠,𝑣 , f 𝑠,𝑡

(4)


𝑁𝑏
exp 𝜓 f𝑖𝑠,𝑣 , f𝑖𝑠,𝑡
1 Õ
 
 ,
L𝑡 2𝑖 = −
log Í
𝑁𝑏
𝑁𝑏 𝑖=1
exp 𝜓 f 𝑠,𝑣 , f 𝑠,𝑡

(5)

𝑖

𝑗=1

𝑗

𝑗

𝑗=1

𝑖

where 𝜓 (·) denotes the similarity function (e.g., cosine similarity), and 𝑁𝑏 is the mini-batch size.
Through the above procedure, we obtain a set of Contour-aware textual features T𝑐 = {𝒇𝑖𝑡 }𝐶𝑖=1 ,
which serve as compact and discriminative contour-aware representations for cross-modal identity
alignment.
3.4

TRM

Although the instantiated attribute texts are generated per image and the Contour-aware prompts
are constructed at the class level, the inherent semantic gap between these two types of textual
descriptions limits their direct applicability in unified visual-language representation. To bridge this
gap, we introduce a TRM, aiming to integrate instance-level attribute descriptions and class-level
contour cues into a unified and discriminative textual feature space.
As illustrated in Figure 5, given an input image 𝑥𝑖 (either RGB or IR), we first extract its visual
representation 𝒇𝑖 using the visual encoder. In parallel, we feed its attribute-based textual description
𝒕𝑖 and the corresponding sketch-based class textual feature 𝒕𝑖𝑠 into a textual fusion network 𝐸𝑎
composed of three fully connected layers to obtain an initial fused textual representation:
(6)

𝒇𝑖𝑎 = 𝐸𝑎 ([𝒕𝑖 ; 𝑡𝑖𝑠 ]),

where [; ] denotes concatenation.
To further model cross-modal interaction, we introduce a Transformer-based decoder module 𝐸𝑞
composed of three layers of TransDecoder blocks. This module takes both the visual feature 𝒇𝑖 and
𝑞
the fused text 𝒇𝑖𝑎 as input to generate a redefined, image-aware textual representation 𝒇𝑖 :
𝒇𝑖 = TransDecoder(𝒇𝑖 , 𝒇𝑖𝑎 ),

(7)

𝑞

which is then refined via a residual connection:
𝑞

𝑞

(8)

𝑞

𝒇𝑖 ← 𝒇𝑖 + 𝛾𝒇𝑖 ,

where 𝛾 ∈ R𝐶 is a learnable scaling factor initialized to a small value (e.g., 10−4 ) to preserve the

semantic priors of the original textual input.
𝑞
To ensure that the redefined textual features 𝒇𝑖 are both semantically aligned with the visual content and discriminative with respect to pedestrian identity, we employ a combination of contrastive
and classification losses. Specifically, the training objective is:
𝑞

𝑞

𝑞

𝑞

(9)

L𝑎𝑙𝑖𝑔𝑛 = L𝑖2𝑡 + L𝑡 2𝑖 + L𝐼 𝐷 ,
where the image-to-text and text-to-image contrastive losses are defined as:
𝑞 
𝑁𝑏
exp 𝜓 𝒇𝑖 , 𝒇𝑖
1 Õ
𝑞
 

L𝑖2𝑡 = −
log Í
𝑞 ,
𝑁𝑏
𝑁𝑏 𝑖=1
exp 𝜓 𝒇 , 𝒇
𝑗=1

𝑖

(10)

𝑗

𝑞 
𝑁𝑏
exp 𝜓 𝒇𝑖 , 𝒇𝑖
1 Õ
𝑞
L𝑖2𝑡 = −
log Í𝑁
𝑞  ,
𝑏
𝑁𝑏 𝑖=1
𝑗=1 exp 𝜓 𝒇 𝑗 , 𝒇𝑖

(11)

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

113:8

Y. Tao and X. Zhang

and the identity classification loss is:
𝑁

𝑞

L𝐼 𝐷 = −

𝑏
1 Õ
𝑞
𝒒𝑖 log(𝑾𝑡 𝒇𝑖 ),
𝑁𝑏 𝑖=1

(12)

where 𝜓 (·) denotes a similarity metric (e.g., cosine similarity), 𝒒𝑖 is the one-hot label vector for
identity 𝑦𝑖 , and 𝑾𝑡 is the identity classifier.
The TRM module consists of four components: a Text Encoder, a Visual Encoder, a Fusion
Network, and an Interaction Network. The Text Encoder and Visual Encoder adopt the standard
CLIP Text Encoder and Visual Encoder, respectively. The Fusion Network is constructed with three
fully connected layers, with output dimensions of 1,024, 512, and 1,024, respectively. The Interaction
Network comprises three Transformer Decoder Blocks, each containing four sub-modules: Masked
Multi-Head Self-Attention, Cross-Attention, Feed-Forward, and LayerNorm. Within each block,
the multi-head attention consists of 8 heads, each with a dimension of 64; the feature dimension of
each token is 512, and the Feed-Forward layer produces a final output of dimension 1,024. During
training, we freeze the parameters of the visual encoder and text encoder and update only the
components of the fusion module 𝐸𝑎 , the interaction module 𝐸𝑞 , and the identity classifier 𝑾𝑡 .
𝑞
After convergence, the redefined feature 𝒇𝑖 serves as an enriched and interaction-aware textual
representation, effectively capturing both attribute-level and contour-level semantics for each image.
3.5

Text-Guided Attribute and Prototype Alignment Model

In this section, we present our proposed model, text-guided attribute and prototype alignment
(TAP-align). As illustrated in Figure 2, the proposed model adopts a dual-branch architecture
commonly used in VI-ReID tasks and employs the visual encoder from a pre-trained CLIP model as
its backbone. Specifically, the first four layers of the visual encoder are utilized to construct a parallel,
shallow structure, serving as modality-specific feature extractors, 𝐸 𝑣,𝑣 and 𝐸 𝑣,𝑖 , which extract
features from RGB and IR images, respectively. The remaining layers are shared and constitute the
encoder 𝐸, which is responsible for learning modality-invariant visual representations.
The model is trained using a joint optimization strategy that combines the identity classification loss and the weighted regularized triplet loss. The identity loss supervises the learning of
discriminative identity features. In contrast, the weighted regularized triplet loss assigns adaptive weights to positive and negative sample pairs, thereby enhancing the model’s capability to
extract modality-invariant features. This joint learning strategy proves particularly effective in
cross-modality person ReID scenarios, where RGB and IR images exhibit significant appearance
discrepancies.
To further improve cross-modality alignment, we explicitly mine potential modality-invariant
textual cues from images as semantic guidance. In particular, we introduce a Text Prototype
Discriminative Loss, which encourages visual features to align with class-level text prototypes that
carry identity-related semantic information. This strategy significantly enhances both semantic
consistency and discriminative capability in cross-modality matching. Finally, the overall joint
optimization objective is defined as:
𝑣𝑠
𝑟𝑠
L𝑡𝑜𝑡𝑎𝑙 = L𝑖𝑑 + L𝑤𝑟𝑡 + 𝜆1 L𝑖2𝑡𝑐𝑒
+ 𝜆2 L𝑖2𝑡𝑐𝑒
,

(13)

where identity loss L𝑖𝑑 , weighted regularized triplet loss L𝑤𝑟𝑡 , text prototype discriminative loss
𝑣𝑠 , and L𝑟𝑠
L𝑣2𝑡𝑐𝑒
𝑖2𝑡𝑐𝑒 are formally defined as :
𝑛

L𝑖𝑑 = −

𝑏
1 Õ
𝒒𝑖 log(𝑾𝑐 𝒇𝑖 ).
𝑛𝑏 𝑖=1

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

(14)


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-to-image Person Search based on Semantic Reorganization.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Computer Vision and Image Understanding 267 (2026) 104733

Contents lists available at ScienceDirect

Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu

Text-to-image Person Search based on Semantic Reorganization
Jielong He a , Feng Chen a,b ,∗, Xiwen Qu a , Yang Liu a
a
b

School of Computer Science and Technology, Anhui University of Technology, Anhui, Ma’anshan, 243032, China
Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education, Jilin University, Changchun, 130012, China

ARTICLE

INFO

Communicated by Shiliang Zhang
Keywords:
Cross-modal retrieval
Person re-identification
Data generation
Cross-modal re-ranking
Semantic alignment

ABSTRACT
Despite significant advancements in text-based person search (TBPS), challenges remain in acquiring large-scale
pedestrian data due to privacy concerns and the high cost of manual annotation. Additionally, substantial
modality gaps between textual and visual semantics lead to a degradation in retrieval performance. To
address these challenges, we propose a novel framework, i.e., text-to-image Person Search based on Semantic
Reorganization (PSSR). Specifically, PSSR consists of two key modules: a semantic reconstruction-based data
augmentation module, which enhances images through pedestrian structure and semantic information, and
utilizes a large multimodal language model to generate corresponding textual descriptions, thereby increasing
the diversity of the training data. Furthermore, during the testing phase, we introduce a cross-modal similarity
re-ranking mechanism that improves the feature representation by obtaining reliable image features for the
given text feature and fusing them into a whole. This approach alleviates the negative impact of modality
differences. Extensive experimental results demonstrate the significant effectiveness of our method across three
text-based person search datasets. Source code is at https://github.com/JielongHe/PSSR.

1. Introduction
Text-Based Person Search (TBPS) (Li et al., 2017; Wang et al., 2021)
is a cross-modal understanding task (Wu et al., 2024; Wang et al., 2023;
Yu et al., 2024; Gong et al., 2023) that retrieves pedestrian images
based on textual descriptions. Its goal is to establish a semantic link
between natural language and visual content, enabling the localization of fine-grained information in complex environments. Thanks to
breakthroughs in deep learning, object detection (Cheng et al., 2024),
and attention mechanisms (Zhou et al., 2022), traditional image-based
pedestrian retrieval (Feng et al., 2025b) has achieved great success,
but the performance of TBPS tasks has relatively declined. Unlike
other text–image retrieval tasks (Lülf et al., 2024), TBPS requires more
detailed textual descriptions of pedestrian images. As a result, data
annotation for TBPS is more complex than for text–image retrieval. And
due to privacy concerns, acquiring large-scale personnel image datasets
is challenging, leading to typically smaller TBPS datasets. Furthermore,
compared to traditional person re-identification (Chen et al., 2023; R.
Wang et al., 2022; Chen et al., 2021), TBPS faces significant modal
differences, making text-to-image retrieval more difficult than imageto-image retrieval. Therefore, effective measures must be taken to
augment TBPS datasets and mitigate the negative impact of modal
differences on retrieval performance.
In the TBPS task, due to data scarcity, some methods (Tan et al.,
2024) utilize extra large datasets such as LUPerson (Fu et al., 2021)

as image sources and employ Multi-modal Large Language Models
(MLLM) to generate textual descriptions. The generated image–text
pairs are used to train a pre-trained model, which is then fine-tuned
using transfer learning in the target domain. However, this approach
typically requires substantial computational resources for pre-training
and demands high-quality textual descriptions. To effectively augment
target domain data without relying on external datasets, Z. Li et al.
(2024) used large language models to diversify and rewrite text in the
existing TBPS datasets, achieving data augmentation. Z. Song et al.
(2024) leveraged publicly available clothing and accessory images from
the internet as references, as shown in Fig. 1(a), and used a diffusion
model to edit images from the original dataset, generating diverse
new images. They also modified the original textual descriptions using
the LLM model. While this approach achieved some success, it relies
on diffusion models, which have a complex training process, and the
generated images may differ from real images. To avoid excessive
computational costs, we propose a Semantic Restructuring-based Data
Augmentation (SRDA). This method divides the original images into
two groups according to gender information provided by the MLLM.
As shown in Fig. 1(b), horizontally splitting pedestrian images with
different IDs but the same gender and stitching the top and bottom parts
together to create a new image with a new ID. Unlike enhancement
strategies that rely on large-scale generative models to edit attributes
or reconstruct images (Z. Song et al., 2024), our SRDA does not attempt

∗ Corresponding author at: School of Computer Science and Technology, Anhui University of Technology, Anhui, Ma’anshan, 243032, China.

E-mail address: chenfeng@ahut.edu.cn (F. Chen).
https://doi.org/10.1016/j.cviu.2026.104733
Received 11 October 2025; Received in revised form 4 January 2026; Accepted 9 March 2026
Available online 10 March 2026
1077-3142/© 2026 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

Fig. 1. Comparison of different data augmentation methods. (a) Generate images using diffusion models based on attributes, while generating text based on the
LLM. (b) Image–text pair augmentation based on semantic information and pedestrian image structure information.

to generate visually natural new images. Instead, it focuses on the core
requirements of TBPS by explicitly reorganizing the structure of pedestrians (upper-body/lower-body semantic units) to create as diverse
and previously unseen images as possible from the original dataset,
thereby enhancing the model’s robustness. Diffusion models typically
perform high-cost iterative sampling in pixel space, with a focus on
the visual realism of the image. In contrast, our SRDA directly utilizes
semantic information and leverages the local structure of real images to
achieve combinatorial expansion. Its generation process requires no additional training and does not suffer from the cross-domain generation
quality fluctuations. Therefore, conceptually, SRDA is a ‘‘semanticbased structural combinatorial enhancement’’ rather than a generative
enhancement based on image reconstruction. Computationally, its cost
is much lower than that of diffusion models, and it does not rely on
large CLIP models for high-dimensional feature-driven image editing,
making it lightweight, stable, and controllable. Due to the lack of
consistency between the token positions in the text description and the
spatial information of the image, it is impossible to obtain an accurate
description of the generated image by splitting and reassembling the
two texts. Accordingly, we utilize the MLLM model to produce textual
descriptions for the reconstructed images, significantly improving the
accuracy of the textual descriptions.
To address the modality differences in the TBPS task, most methods
design improved cross-modal matching approaches. These approaches
align images and texts in a joint embedding space and use crossmodal matching loss functions to align image–text pairs. Common loss
functions include ranking loss and cross-modal projection matching
(CMPM) loss (Zhang and Lu, 2018). Compared to ranking loss, CMPM
loss does not require selecting specific triplets or adjusting margin parameters and demonstrates better stability across different batch sizes,
making it widely used in text-to-image person retrieval. However, due
to the variability in projection lengths, CMPM cannot precisely control
the projection probability distribution and struggles to focus on hard
negative samples when the model is updated. To address this, Jiang and
Ye (2023) proposed the SDM loss, which improves model performance
by minimizing the KL divergence between the normalized image–text
similarity score distribution and the normalized label distribution. To
better achieve semantic alignment across different modalities, several
works have made targeted improvements to the contrastive loss. Z. Lu

et al. (2025) introduced the anchor-based contrastive loss, which uses
features extracted from MLLM as anchors to align visual and text features more closely. Lu et al. (2024) proposed multimodality contrastive
learning, which facilitates the convergence of intra- and inter-modal
positive samples. Despite these advancements, these loss functions and
matching methods still do not fully address the modality differences
between text and visual data. On the other hand, some work attempts
to improve performance during the testing phase by re-ranking search
results (Zhong et al., 2017). Specifically, this method improves retrieval
performance by treating the target images (query) and the images to
be retrieved (gallery) as a whole through a metric based on reciprocal
nearest neighbors. However, the core assumption of traditional reranking relies on the ‘‘query and candidate samples sharing a stable
k-reciprocal neighborhood structure’’. This assumption holds true in
the same modality (e.g., traditional Re-ID image retrieval) but is inherently not satisfied in cross-modal (text–image) scenarios. Meanwhile,
comparisons of distance distributions across different modalities reveal
that the average cosine distance between image–image pairs is significantly smaller than that between text–image pairs, with the latter
even exhibiting an unbalanced distribution. This phenomenon suggests
that text lacks stable local structural support in feature space. These
empirical results clearly indicate that traditional re-ranking cannot
effectively utilize textual semantics in cross-modal retrieval. Based on
these observations, we propose a Cross-modal Similarity Re-ranking
Mechanism (CSRM) during the testing phase to build a more semantically reliable cross-modal neighborhood structure for text. Through
cross-modal mutual neighborhood selection and feature fusion (J. Song
et al., 2024), we retain only those neighbors that are consistent with
the text semantics and also representative in image space. We then use
the Jaccard distance to suppress noisy samples, generating more stable
new text query features. Compared to methods that improve model
architectures or loss functions, our CSRM module is activated during
the testing phase, improving the initial ranking results by finding crossmodal neighbors. The CSRM can be seen as a plug-and-play module,
which can be directly applied to the aforementioned methods to further
enhance performance.
It should be stressed that the SRDA and CSRM module are not two
independent, but rather complementary solutions proposed to address
two core issues of TBPS. Specifically, SRDA aims to alleviate the
2

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

problems of data scarcity and insufficient semantic coverage during the
training phase, while CSRM corrects cross-modal distribution discrepancies during the testing phase, thus forming an integrated framework
of ‘‘training enhancement + testing calibration’’. More specifically,
SRDA significantly enriches the attire combinations and upper–lower
body semantic variations required for cross-modal alignment through
structural-level semantic reconstruction, enabling the model to learn a
more comprehensive diversity during training. However, even with the
expanded training data, the alignment between the textual and visual
modalities remains challenging in feature space, which affects the ranking process during the testing phase due to modal discrepancies. CSRM
addresses this residual issue by leveraging cross-modal neighborhood
selection and Jaccard weight fusion, enabling the query text to adjust
its position in feature space with the help of the most similar image
features, thereby enhancing the stability and accuracy of the final
ranking. Therefore, the two modules are logically interconnected in
their design.
Our contribution can be summarized as follows:

2.2. Cross-modal alignment-based person search
In the field of cross-modal person search (Wang et al., 2021; Duan
et al., 2024; Y. Li et al., 2024), bridging the semantic gap between
text and visual representations remains a core challenge. Li et al.
(2017) pioneered this field by constructing a shared embedding space
and introducing a cross-modal alignment framework. Later studies implemented deep learning frameworks, incorporating VGG-LSTM dualstream networks to capture global visual representations and sequential
textual features, thereby establishing the groundwork for advanced
semantic alignment. Zheng et al. (2025) proposed a graph aggregation
alignment network that consists of two modules. The graph interaction fusion module leveraged graph representation learning to process
single-modality features, while guiding low-level multimodal features
for global fusion. And the information alignment module then refined
the results through secondary calibration. To tackle modal heterogeneity, several works have explored multi-granularity feature fusion. Feng
et al. (2025a) proposed a cross-modality multi-granularity transformer
framework that extracts coarse-to-fine feature representations and integrates discriminative information across multiple granularities. Shao
et al. (2022) introduced a BERT-Transformer collaborative architecture,
achieving phrase-level visual localization via multi-level attention networks. Yan et al. (2024) suggested that combining global and local
alignment modules enhances the semantic correspondence between
different modalities. Liu et al. (2023) utilized visual attention within
the global feature extraction network, enabling the model to selectively
concentrate on key local features in both images and text descriptions,
enhancing its focus on local details.
More recent studies have concentrated on resolving modal differences at a deeper level. S. Li et al. (2024) proposed an uncertaintyaware progressive learning framework to progressively eliminate
semantic biases between modalities. Liu et al. (2024) integrated causal
reasoning by separating visual features into causally connected elements and scene perturbations, establishing domain-invariant crossmodal links via counterfactual learning. Visual–language pre-training
models also show great promise, Jiang and Ye (2023) designed a
multi-granularity attention module based on the CLIP architecture to
reinforce intra-modal consistency and inter-modal topological correspondence. You et al. (2025) designed an effective framework that
extracts co-saliency features between images and text, enhancing the
variation of cross-modal discriminative features while reducing noise
interference. To tackle the semantic inconsistency in weak positive
pairs caused by viewpoint differences, Lu et al. (2024) extracted
cross-modal identity-invariant features using semantic-aligning optimal
transport and multimodal contrastive learning techniques. Moreover,
Z. Lu et al. (2025) utilized learnable prompts and MLLM interactive
learning to address the semantic variations in positive pairs and the appearance similarity in negative pairs, respectively. Beyond conventional
retrieval settings, Y. Lu et al. (2025) explored an interactive text-based
person search paradigm, where multimodal large language models are
leveraged to refine ambiguous queries through dialogue.
Despite progress, challenges in handling modal differences and
redundant information remain. We propose a cross-modal similarity
re-ranking mechanism in the testing phase. Specifically, we select
appropriate image features to represent the counterpart text, thereby
narrowing the modality gap. Unlike prior approaches, our method
refines cross-modal similarity estimation during the inference phase in
a training-free and model-agnostic manner, making it complementary
to existing cross-modal alignment frameworks.

• We utilize the MLLM model based on pedestrian image structural information to perform semantic-based data augmentation,
effectively enriching the diversity of the data.
• We propose a cross-modal similarity re-ranking mechanism,
which selects the most suitable image features to fuse with
the corresponding text feature, thereby mitigating the negative
impact of modality differences on retrieval performance.
• Our method outperforms existing state-of-the-art methods, achieving the mean Average Precision (mAP) of 73.34%, 48.11%
and 51.77% on the CUHK-PEDES, ICFG-PEDES and RSTPReid,
respectively.
2. Related works
Text-based person retrieval in this paper is closely related to data
augmentation and cross-modal alignment. We briefly review the literature in these two areas.
2.1. Data augmentation for person search
In Text-Based Person Search (TBPS), data augmentation is a key
technique for enhancing model robustness and generalization, gaining
significant attention in recent years. Traditional person re-identification
tasks typically employ image augmentation methods such as cropping,
flipping, and rotation to improve model generalization. However, generative augmentation strategies, including those based on traditional
methods and GANs (Zheng et al., 2017), often incur high training costs
with limited performance gains. Chen et al. (2020) proposed a selfsupervised learning-based data augmentation strategy that combines ID
and instance augmentation, but it lacks fine-grained semantic control.
In TBPS, both image augmentation and generation of semantically consistent image–text pairs are crucial. Z. Li et al. (2024) leveraged large language models to rewrite textual descriptions in existing
datasets, generating diverse text, though only addressing textual augmentation. The MALS dataset (Yang et al., 2023) employs diffusion
models and text calibration to generate large-scale synthetic data. Tan
et al. (2024) used the LUPerson dataset (Fu et al., 2021) and Multimodal Large Language Models (MLLM) to generate textual descriptions
for images, while Z. Song et al. (2024) proposed the Diverse Person framework, using a diffusion model to edit person attributes and
large language models for textual annotations, mitigating privacy and
annotation cost issues.
However, these methods depend on external attributes or images
and have limitations in fine-grained semantic control. In this paper, we
propose a data augmentation method based on semantic reorganization. Specifically, we generate high-quality images based on pedestrian
image structure and produce corresponding text by fine-tuning the
MLLM. During this process, we utilized MLLM to ensure cross-modal
semantic consistency of the generated data.

3. Method
To address the issue of insufficient diversity of training data and significant modal differences in TBPS tasks, as shown in Fig. 2, we propose
the PSSR framework. Firstly, we introduce semantic restructuring-based
data augmentation to get more diverse data. Secondly, all original and
3

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

Fig. 2. Overview of the proposed PSSR. The model is built upon a CLIP-based dual-stream network. We integrate a Semantic Restructuring-based Data
Augmentation (SRDA) into the framework to enhance the diversity of training data. Additionally, during the testing phase, we introduce a Cross-modal Similarity
Re-ranking Mechanism (CSRM) to effectively mitigate the negative impact of modality differences on retrieval performance.

images. Finally, the generated image–text pairs { ̂
𝐼, 𝑇̂ } are obtained,
expanding the dataset’s size and diversity, which is crucial for the
model’s performance.

augmented data are fed into a dual branch network for training. Finally,
a cross-modal similarity rearrangement mechanism in the testing phase
is proposed to address the modal differences. Next, these three parts are
described in detail.

3.2. Network structure
3.1. Semantic restructuring-based data augmentation
In order to better map images and text description to the common feature space, we designed a Clip-based dual-branch network
architecture. As shown in Fig. 2, The proposed network structure
consists of two unimodal encoders for image and text encoding, respectively. For a given image–text pair (𝐼, 𝑇 ), the image encoder first
processes the image 𝐼 to produce a series of visual representations
𝑣 , 𝑓 𝑣 , … , 𝑓 𝑣 }. Here, 𝑓 𝑣 denotes the global visual represen𝐹 𝑣 = {𝑓cls
𝑀
1
cls
tation, while 𝑓𝑖𝑣 (𝑖 = 1, … , 𝑀) denotes the local patch representation.
Similarly, the text 𝑇 undergoes processing by the text encoder, yielding
𝑡 , 𝑓 𝑡 }. Here, 𝑓 𝑡
a series of textual representations 𝐹 𝑡 = {𝑓1𝑡 , … , 𝑓𝑁
eos
eos
denotes the global textual representation.
During the model training phase, both original and generated data
will be used to update our model. Specifically, both the identification
loss (ID loss) and similar distribution matching loss (SDM loss) were
adopted. The ID loss improves the model’s accuracy in the identity
classification task. The SDM loss optimizes the alignment between
image and text representations by minimizing the discrepancy between
predicted matching probabilities and true matching labels. Given a
mini-batch of 𝑁 image–text pairs, the matching probability 𝑝𝑖,𝑗 is
calculated using the following softmax function:

In order to enhance the diversity of image–text pairs, we propose a
data augmentation method based on semantic information and pedestrian image structure information. Firstly, for the original pedestrian
images, a multimodal large language model (MLLM) is used to determine the gender of the pedestrian, and the original images are divided
into two groups based on gender, i.e., the male image group 𝐼male and
female image group 𝐼female . Secondly, we randomly select two different
IDs to combine within each gender group. Specifically, for each selected
ID pair (e.g., 𝐼𝐷1 and 𝐼𝐷2 ), we randomly some images with 𝐼𝐷1 to
provide the ‘‘upper body’’ and the same number of images with 𝐼𝐷2
to represent the ‘‘lower body’’. Afterward, we horizontally split the
upper body and lower body images to obtain new upper–lower body
image pairs, and then we concatenate them to generate new images
𝐼̂ = 𝑐𝑜𝑛𝑐𝑎𝑡(𝐼upper , 𝐼lower ), where 𝐼upper represents the selected upper
body image, and 𝐼lower indicates the selected lower body image. Since
the stitched images does not belong to any existing ID (class), we assign
it a completely new ID.
To ensure that the generated images hold reasonable and diverse
textual descriptions, we fine-tune the BLIP (Bootstrapping Language
Image Pretraining) model using the original image–text pairs. The BLIP
model generates textual descriptions based on image content. After
fine-tuning, it generates more semantically consistent descriptions with
respect to the generated images. For each augmented image, we take
the first three words of the corresponding upper body text as the
prompt, for example, ‘‘A man in’’, and input this prompt into the finetuned BLIP model to generate a new textual description. Specifically,
̂ and the corresponding upper body text be
let the generated image be 𝐼,
𝑇upper . We extract the first three words from 𝑇upper and combine them
into the prompt 𝑇𝑝𝑟𝑜 = {𝑤1 , 𝑤2 , 𝑤3 }. We then input the prompt into the
BLIP model to obtain the corresponding generated text 𝑇̂ :
̂
𝑇̂ = 𝐵𝐿𝐼𝑃 {𝑇𝑝𝑟𝑜 , 𝐼},

exp(sim(𝑓𝑖𝑣 , 𝑓𝑗𝑡 )∕𝜏)
𝑝𝑖,𝑗 = ∑𝑁
,
𝑣
𝑡
𝑘=1 exp(sim(𝑓𝑖 , 𝑓𝑘 )∕𝜏)

(2)

𝑇

𝑢 𝑣
where sim(𝑢, 𝑣) = ‖𝑢‖‖𝑣‖
represents the cosine similarity, and 𝜏 is a
temperature hyperparameter controlling the peak of the probability
distribution. The image-to-text SDM loss 𝐿𝑖2𝑡 is then computed using
the KL divergence between the predicted probabilities 𝑝𝑖,𝑗 and the true
matching probabilities 𝑞𝑖,𝑗 :

1 ∑∑
𝑝 log
𝑁 𝑖=1 𝑗=1 𝑖,𝑗
𝑁

𝐿𝑖2𝑡 =

𝑁

𝑦

(1)

where 𝑞𝑖,𝑗 = ∑𝑁 𝑖,𝑗

𝑘=1 𝑦𝑖,𝑘

This text 𝑇̂ serves as the textual description for the new image, further enhancing the diversity and semantic reliability of the generated

(

𝑝𝑖,𝑗
𝑞𝑖,𝑗 + 𝜖

)
,

(3)

represents the true matching probability, and 𝜖

is a small number to avoid numerical issues. Symmetrically, the textto-image SDM loss 𝐿𝑡2𝑖 is calculated by exchanging the image and text
4

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

representations. Finally, the bi-directional SDM loss is the sum of both
the image-to-text and text-to-image losses:

Subsequently, we use the Jaccard distance metric 𝐽 (⋅, ⋅) to compute the
similarity between the text feature 𝑡𝑚 and the image 𝑣𝑖 , defined as:

𝑠𝑑𝑚 = 𝐿𝑖2𝑡 + 𝐿𝑡2𝑖 .

𝐽 (𝑡𝑚 , 𝑣𝑖 ) = 1 −

(4)

To further reduce intra-modal differences, we introduce the intramodal triplet loss. This loss function promotes tight clustering of
features by minimizing the distance difference between positive and
negative samples within the same modality, thereby enhancing intramodal discrimination. Since the text is generated by a Multi-modal
Large Language Model (MLLM), it inevitably contains some noise,
which may cause the model to overfit to noise. Therefore, we use only
the image features to calculate the intra-modal triplet loss. Given a set
of visual feature representations for triplets (𝑓 𝑎𝑣 , 𝑓 𝑝𝑣 , 𝑓 𝑛𝑣 ), where 𝑓 𝑎𝑣
and 𝑓 𝑝𝑣 are positive sample pairs and 𝑓 𝑛𝑣 is a negative sample, the
triplet loss can be expressed as:
img_triplet =

𝑁 [
∑

(
)]
1
𝑎
𝑝
𝑎
𝑛
max sim(𝑓𝑖 𝑣 , 𝑓𝑖 𝑣 ) − sim(𝑓𝑖 𝑣 , 𝑓𝑖 𝑣 ) + 𝛿, 0 ,
𝑁 𝑖=1

𝑖

4. Experiments
4.1. Datasets and protocol
We evaluate our approach using three benchmark datasets. The
CUHK-PEDES (Li et al., 2017) dataset, a pioneer in text-based person retrieval, includes 34,054/68,108 images/sentences for 11,003
identities in the training set, with the validation/test set containing
3,078/3,074 images. The ICFG-PEDES (Ding et al., 2021) dataset consists of 54,522 images corresponding to 4102 individuals, with one caption per image. The training/testing set includes 34,674/19,848 image–
text pairs representing 3,102/1,000 identities. The RSTPReid (Zhu
et al., 2021) dataset contains 20,505 images of 4101 identities, with
two captions assigned to each image. The training/validation/testing
sets cover 3,701/200/200 identities, respectively. We adopt Rank-1 and
mean Average Precision (mAP) as evaluation metrics.

(6)

3.3. Cross-modal similarity re-ranking mechanism
During the testing phase, cross-modal matching faces the challenge
of modal gaps. In order to better solve this problem, we innovatively
propose a cross-modal re-rank method, whose core idea is to use
image features to represent text features and use the fused features for
retrieval. As shown in Fig. 2, we first calculate the similarity between
the query text feature 𝑡𝑚 and all gallery images {𝑣𝑖 }𝐺
. By applying the
1
k-nearest neighbor search, we rank the gallery images based on their
cosine similarity to the query, resulting in the following sorted image
set:

4.2. Implementation details
Our model is built upon the CLIP architecture. All input images are
resized to a fixed resolution of 384 × 128. The network is trained for
60 epochs using the Adam optimizer with an initial learning rate of
1 × 10−5 , which is decayed following a cosine annealing schedule to
promote stable convergence. A batch size of 64 is employed, and all
experiments are conducted on a single NVIDIA RTX 4090D GPU with
24 GB of memory.
The hyperparameters are configured as follows: 𝜏 = 0.02, 𝜖 = 1 ×
10−8 , and 𝛿 = 1.0 in Eqs. (2), (3), and (5), respectively. In the composite
loss function (Eq. (6)), the weighting factors are set to 𝛼 = 0.1, 𝛽 = 1.0,
and 𝛾 = 1.0. Furthermore, the neighborhood size 𝑘 in Eq. (7) is set to
4, and the momentum coefficient 𝜆 in Eq. (11) is set to 0.9.

(7)

Next, for each image 𝑣𝑖 ∈ 𝑁𝑘 (𝑡𝑚 ), we compute the cross-modal cosine
similarity between 𝑣𝑖 and all existing textual representations and obtain
the set of k-nearest texts of 𝑣𝑙 as 𝑁𝑘 (𝑣𝑖 ) = {𝑡1 , 𝑡2 , … , 𝑡𝑘 }. Therefore,
based on the bidirectional nearest-neighbor relation, we construct the
set of cross-modal k-reciprocal nearest images 𝑅𝑘 (𝑡𝑚 ) of query text
feature 𝑡𝑚 defined as:
𝑅𝑘 (𝑡𝑚 ) = {𝑣𝑖 ∣ (𝑣𝑖 ∈ 𝑁𝑘 (𝑡𝑚 )) ∩ (𝑡𝑚 ∈ 𝑁𝑘 (𝑣𝑖 ))}.

4.3. Comparison with the state-of-the-art methods

(8)

To validate the effectiveness of our proposed framework, we conducted a series of comparative experiments against several state-of-theart (SOTA) methods. The experimental results are summarized in Table
1.
On the CUHK-PEDES dataset, our method achieved a Rank-1 accuracy of 75.71% and an mAP of 73.34%, surpassing the best-performing
SOTA methods by 0.05% and 5.43%, respectively. On the ICFG-PEDES
dataset, our approach achieved a Rank-1 score of 65.91% and an mAP
of 48.11%, outperforming the SOTA methods by 0.15% and 8.47%,
respectively. Finally, on the RSTPReid dataset, our method achieved
a Rank-1 accuracy of 62.53% and an mAP of 51.77%, improving upon

In order to better fuse the text 𝑡𝑚 with its cross-modal k-reciprocal
nearest images, we introduce the Jaccard distance (Gong et al., 2024)
as a fusion weight. Specifically, for 𝑣𝑖 ∈ 𝑅𝑘 (𝑡𝑚 ), we compute the intramodal cosine similarity between it and all existing image features,
which yields the set of k-nearest neighbor of the images 𝑁𝑘 (𝑣𝑖 ). For
𝑣𝑗 ∈ 𝑁𝑘 (𝑣𝑖 ), we then compute the k-nearest neighbor set 𝑁𝑘 (𝑣𝑗 ), which
yields the intra-modal k-reciprocal nearest neighbor set 𝑅𝑘 (𝑣𝑖 ) of the
image 𝑣𝑖 , which is defined as:
𝑅𝑘 (𝑣𝑖 ) = {𝑣𝑗 ∣ (𝑣𝑗 ∈ 𝑁𝑘 (𝑣𝑖 )) ∩ (𝑣𝑖 ∈ 𝑁𝑘 (𝑣𝑗 ))}.

𝑘 𝑚

At this point, the fused text feature 𝑡𝑓𝑚𝑢𝑠𝑒𝑑 is used as a new query
text feature for image retrieval, which is able to capture cross-modal
inter-modal correlations more accurately than the original text feature
𝑡𝑚 .

(5)

where, 𝛼, 𝛽, and 𝛾 are hyperparameters used to balance various loss
components.

𝑁𝑘 (𝑡𝑚 ) = {𝑣1 , 𝑣2 , … , 𝑣𝑘 }.

(10)

On this basis, we enhance the fusion effect between textual features
𝑡𝑚 and cross-modal images by calculating the fusion weights for the
images within the image set 𝑅𝑘 (𝑡𝑚 ).
Additionally, during the fusion process, we consider the influence
of the query text feature 𝑡𝑚 itself. A weighting strategy is introduced,
where the ratio of 𝑡𝑚 is 𝜆, and (1 − 𝜆) represents the weight assigned
to the image features in 𝑅𝑘 (𝑡𝑚 ). Therefore, the final fused feature 𝑡𝑓𝑚𝑢𝑠𝑒𝑑
can be expressed as:
∑
(1 − 𝐽 (𝑡𝑚 , 𝑣𝑖 ))
𝑡𝑓𝑚𝑢𝑠𝑒𝑑 = 𝜆 ⋅ 𝑡𝑚 + (1 − 𝜆)
⋅ 𝑣𝑖 .
(11)
∑
𝑣𝑗 ∈𝑅𝑘 (𝑡𝑚 ) (1 − 𝐽 (𝑡𝑚 , 𝑣𝑗 ))
𝑣 ∈𝑅 (𝑡 )

where 𝑠𝑖𝑚(𝑢, 𝑣) denotes the cosine similarity between the vectors 𝑢 and
𝑣, and 𝛿 is a hyperparameter controlling the minimum spacing between
positive and negative samples. Since the augmented image shares half
of its content with the original image but has a different ID, the features
of the original ID image may act as strong negative samples. In this
case, to increase the distance between them in the feature space, the
model focuses more on the dissimilar parts, enabling it to extract the
required fine-grained features within a smaller scope.
The overall objective includes the Triplet loss, ID loss, and SDM loss:
 = 𝛼 ∗ img_triplet + 𝛽 ∗ 𝑖𝑑 + 𝛾 ∗ 𝑠𝑑𝑚 ,

|𝑅𝑘 (𝑡𝑚 ) ∩ 𝑅𝑘 (𝑣𝑖 )|
.
|𝑅𝑘 (𝑡𝑚 ) ∪ 𝑅𝑘 (𝑣𝑖 )|

(9)
5

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

Table 1
Comparison with the SOTA methods.
Method

Ref

CUHK-PEDES

ICFG-PEDES

Rank-1

mAP

Rank-1

mAP

RSTPReid
Rank-1

mAP

LBUL (Z. Wang et al., 2022)
AXM-Net (Farooq et al., 2022)
CTLPDG (Wu et al., 2023)
BDNet (Liu et al., 2023)
CFine (Yan et al., 2023)
VGSG (He et al., 2023)
IRRA (Jiang and Ye, 2023)
MGRL (Lv et al., 2024)
IRLT (Liu et al., 2024)
FSRL (Wang et al., 2024)
PLOT (Park et al., 2024)
IRRA+DP (Z. Song et al., 2024)
DM-Adapter (Liu et al., 2025)

MM22
MM22
TCSVT23
PR23
TIP23
TIP23
CVPR23
ICASSP24
AAAI24
ICMR24
ECCV24
AAAI24
AAAI25

64.04
64.44
69.47
66.27
69.57
71.38
73.38
73.91
74.46
74.65
75.28
75.66
72.17

–
58.73
60.52
57.04
–
67.91
66.13
67.28
–
67.49
–
66.58
64.33

–
–
57.69
57.31
60.83
63.05
63.46
63.87
64.72
64.01
65.76
65.61
62.64

–
–
36.07
–
–
–
38.06
39.12
–
39.64
–
39.14
36.50

45.55
–
–
–
50.55
–
60.20
–
61.49
60.20
61.80
62.48
60.00

–
–
–
–
–
–
47.17
–
–
47.38
–
48.86
47.37

PSSR (Ours)

–

75.71

73.34

65.91

48.11

62.53

51.77

Table 2
Ablation results for each component of PSSR.
Order

1
2
3
4

Components

CUHK-PEDES

ICFG-PEDES

RSTPReid

Baseline

SRDA

CSRM

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

✓
✓
✓
✓

–
✓
–
✓

–
–
✓
✓

70.79
74.35
72.02
75.71

64.69
69.63
68.13
73.34

61.30
64.85
62.20
65.91

37.68
45.31
44.91
48.11

59.20
61.00
60.31
62.53

46.65
48.18
49.68
51.77

the best SOTA results by 0.05% and 2.91%, respectively. These performance improvements can be attributed to the semantic-based data
augmentation technique employed in our approach, which enhances
the diversity of the training dataset. Unlike the ‘‘IRRA+DP’’ method,
which uses diffusion models to generate diverse images by editing the
original dataset, our approach leverages structural information from
pedestrian images. Although the ‘‘IRRA+DP’’ does generate more natural and realistic images, visual quality does not necessarily equate to
model training effectiveness. ‘‘IRRA+DP’’ relies on the diffusion model
for image editing, with a goal oriented towards producing visually
coherent appearances. However, there may be discrepancies between
the generated images and the textual semantics, and the diversity of
the generated results is limited. In contrast, our data augmentation
based on pedestrian structural information can provide more images
that may appear unrealistic or not exist in the real world. These
types of data effectively enrich the diversity of the original dataset,
thereby improving the model’s robustness. Moreover, in the testing
phase, we further introduce CSRM, which effectively mitigates the
impact of modality discrepancies, leading to a significant improvement
in overall retrieval performance. Therefore, while the visual results of
‘‘IRRA+DP’’ are more natural, from the perspective of the cross-modal
retrieval task objective, our method offers a more cost-effective way
to provide meaningful training gains and ultimately achieves better
performance.

mAP increased by 3.44%, 7.23%, and 3.03%, respectively. The full
model, which incorporates both SRDA and CSRM, exhibited the highest
performance, with mAP improvements of 8.65%, 10.43%, and 5.12%
compared to the baseline. From these results, it can be inferred that
SRDA enhances the diversity of the data through the MLLM model and
image structural information, thereby improving the model’s robustness. At the same time, CSRM effectively mitigates the negative impact
of modality discrepancies on retrieval performance by selecting appropriate image features to fuse with text features during the testing phase.
Furthermore, the combined use of both components leads to further
performance improvements, demonstrating their strong effectiveness
and complementarity.
4.4.2. Ablation studies for the triplet loss
To evaluate the impact of the two intra-modality triplet losses,
we conducted a comprehensive analysis on three different settings.
As shown in Table 3, ’ w/o 𝑖𝑚𝑔_𝑡𝑟𝑖𝑝𝑙𝑒𝑡 ’ indicates the removal of the
triplet loss within the image modality from our approach, while ’
with 𝑡𝑒𝑥𝑡_𝑡𝑟𝑖𝑝𝑙𝑒𝑡 ’ denotes the inclusion of triplet loss within the text
modality in our method. As shown in the table, removing 𝑖𝑚𝑔_𝑡𝑟𝑖𝑝𝑙𝑒𝑡
from our model (sequence 1) leads to a decrease in performance
across all datasets. Similarly, adding 𝑡𝑒𝑥𝑡_𝑡𝑟𝑖𝑝𝑙𝑒𝑡 (sequence 2) results in
a performance drop because the text generated by the MLLM model
contains certain noise, leading to overfitting. Therefore, we opt to use
only the triplet loss within the image modality, thereby optimizing the
CSRM re-ranking mechanism (sequence 3).

4.4. Ablation study

4.5. More discussions

4.4.1. Ablation studies for each component of pssr
We conducted ablation experiments to evaluate the effectiveness
of each component in the PSSR framework. Our baseline is based on
the CLIP model, optimized with ID loss and CMPM loss. Our method
consists of two components: Semantic Restructuring-based Data Augmentation (SRDA) and Cross-modal Similarity Re-ranking Mechanism
(CSRM). The items in ablation studies were independently trained,
rather than being sequentially fine-tuned based on previous models.
As shown in Table 2, the performance gradually improved with
the addition of SRDA and CSRM components. Specifically, when SRDA
was added alone, the model’s mAP on the CUHK-PEDES, ICFG-PEDES,
and RSTPReid datasets increased by 4.94%, 7.63%, and 1.53%, respectively. When CSRM was introduced individually into the baseline,

4.5.1. Comparison of the traditional re-ranking and our CSRM
In intra-modal retrieval tasks, performance can be improved by the
traditional re-ranking strategy (Zhong et al., 2017). However, in crossmodal retrieval tasks, significant modality discrepancies exist between
the image and text modalities, resulting in the traditional intra-modal
re-ranking method being unable to capture data from other modalities
when selecting the nearest neighbors. Accordingly, we propose the
Cross-modal Similarity Re-ranking Mechanism (CSRM) to select reliable
images and fuse them with the corresponding the text. This approach
effectively mitigates the negative impact of modality gaps. As shown
in Table 4, our method clearly surpasses the traditional re-ranking
strategy in cross-modal retrieval tasks.
6

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

Table 3
Ablation results for the triplet loss.
Order

1
2
3

Setting

 w/o 𝑖𝑚𝑔_𝑡𝑟𝑖𝑝𝑙𝑒𝑡
 with 𝑡𝑒𝑥𝑡_𝑡𝑟𝑖𝑝𝑙𝑒𝑡
PSSR

CUHK-PEDES

ICFG-PEDES

RSTPReid

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

75.07
75.45
75.71

72.03
72.95
73.34

65.10
65.35
65.91

46.72
47.53
48.11

61.70
62.05
62.53

50.22
51.15
51.77

Table 4
Comparison of traditional re-ranking and our CSRM.
Order

1
2
3

Setting

Baseline+SRDA
Baseline+SRDA+Traditional Re-ranking (Zhong et al., 2017)
Baseline+SRDA+CSRM

CUHK-PEDES

ICFG-PEDES

RSTPReid

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

74.35
68.68
75.71

69.63
67.76
73.34

64.85
64.28
65.91

45.31
47.29
48.11

61.00
56.40
62.53

48.18
49.14
51.77

Specifically, as shown in the second row, all three negative samples
reflect elements of the dark-blue jeans and laced up shoes mentioned
in the text description. The third and fifth column negative samples exhibit the muted plaid jacket, while the fourth column negative sample
reflects the wide open, long, blue jacket. Therefore, these samples are
ranked higher. As a result, the current method (including our PSSR)
achieves no more than 76% accuracy in the TBPS task. The difficulty
in further improving performance stems from modal discrepancies and
the incomplete nature of the textual descriptions.
To further investigate the impact of CSRM, we performed a T-SNE
visualization of the top-10 retrieved samples, as shown in Fig. 4. Here,
triangles represent text queries, circles represent image samples. The
same color indicates belonging to the same ID, while different colors
indicate belonging to different IDs. Without CSRM (Fig. 4(a)), the text
and image features belonging to the same identity are scattered in the
feature space, lacking clear clustering. In contrast, with CSRM (Fig.
4(b)), samples of the same identity form more compact clusters, illustrating that CSRM successfully mitigates cross-modal discrepancies and
strengthens semantic alignment across modalities. Overall, we can see
that CSRM really helps the model focus on the right text–image pairs.
This shows that CSRM effectively bridges the gap between modalities
and makes cross-modal retrieval both more reliable and more robust.
In the process of cross modal matching, some neighborhood images
may contain a small amount of noise, and there could even be cases
where the neighborhood images do not match the original query text.
However, our cross-modal re-ranking mechanism operates over all samples, and performance evaluation is conducted across the entire dataset.
As long as the negative impact of the noise is smaller than the positive
influence gained from fusing neighborhood image features, the overall
performance of the method still shows improvement. To demonstrate
that the introduction of a small amount of noise under the cross modal
re-ranking mechanism does not reduce the original performance, we
replaced the more effective Jaccard distance with a simple Cosine
distance and conducted comparative experiments. As shown in Table
5, ’Cosine distance’ showed a performance drop compared to ’Jaccard
distance’, but it still outperformed the baseline without cross-modal reranking. This outcome indicates that while the introduction of noise
can indeed affect performance, the carefully designed cross-modal reranking scheme is able to improve overall performance. To suppress
noise as much as possible, we have effectively mitigated it through a
carefully designed neighbor selection and weighted fusion mechanism.
Through deeper analysis, we believe that the reason why Jaccard distance is more effective than Cosine distance is as follows.
First, Cosine similarity has already been used as the basic measure
for cross-modal matching in the initial ranking phase. Therefore, if
Cosine distance were used again during the fusion phase, the evaluation would still be influenced by pointwise similarity, which cannot
provide additional structural information. In contrast, the Jaccard distance, based on the set intersection-over-union of the k-reciprocal

Fig. 3. Comparison of retrieval results with and without CSRM, where green
and red boxes indicate correct and incorrect matches, respectively. Results are
ordered by descending similarity.

4.5.2. Effectiveness of CSRM
To validate the effectiveness of CSRM mechanism, we conducted
comparative experiments with and without the CSRM module. As
shown in Fig. 3, when the CSRM module is removed, the retrieval
results contain a considerable number of incorrect matches (highlighted
in red boxes), indicating that the model fails to consistently select
images that best align with the semantic content of the text query. This
is mainly due to the cross-modal discrepancies that are not sufficiently
alleviated, causing the model to rely on noisy or irrelevant local
features for matching. In contrast, after integrating CSRM, the retrieval
accuracy is significantly improved, with a larger number of correct
matches among the top-5 candidates (highlighted in green boxes).
More specifically, when CSRM is not used (first row), the retrieval
results are relatively poor, but with CSRM (second row), the results
improve, and both positive samples are ranked at the top. This demonstrates that our CSRM improves the original retrieval results. Second,
regardless of whether CSRM is used, the top five samples appear to have
significant visual differences. This is because, although the five images
visually differ greatly, they are still quite similar to the text description.
7

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

Fig. 4. Comparison of T-SNE visualization on the CSRM. Triangles represent query text, circles represent gallery images, and colors represent IDs.

Fig. 5. Comparison of T-SNE visualization on the Baseline and our PSSR. Triangles represent text, circles represent images, and colors represent categories.

4.5.3. Analysis of image generation
Our image augmentation approach is based on pedestrian structural
and semantic information to achieve simple and efficient data generation. To obtain more diverse data, we ensure that the upper and
lower body in the generated results belong to the same gender, without
imposing further constraints. As a result, some of the generated data
may appear unnatural or less coordinated. In fact, a certain degree
of unnaturalness and lack of coordination can effectively enrich the
diversity of the training data, enabling the model to learn new patterns
that may be difficult to discern in the original dataset. Accordingly,
we introduce color differences and boundary gradients to measure the
intensity of artifacts in the generated images. We group the generated
data based on image quality (Zhou et al., 2025) into low artifact (LowA) and high artifact (High-A) groups, and train the model with each
group in combination with the original data. Taking CUHK-PEDES as
an example, the experimental results are shown in Table 6. The experimental results show that, even though high-artifact samples appear
visually uncoordinated, the model performance still sees a significant
improvement. Performance is slightly better when using images with
fewer artifacts. However, using both types of data together yields the
best performance. Therefore, visual artifacts in the generated data do
not degrade the performance of the original model; rather, the diversity
of the training data is key to improving model accuracy and robustness.
This also demonstrates that there is not a simple linear relationship
between the visual realism of generated images and model performance
gains. Additionally, in Fig. 6, we present multiple sets of generated
results. As shown in the third row, by combining pedestrians wearing
long sleeves and long pants (likely in spring) with pedestrians wearing
short sleeves and shorts (likely in summer), we generate images of
pedestrians wearing long sleeves and shorts. We believe this combination captures the intersection of different seasons, and based on this
mechanism, it is possible to achieve cross-fusion of information such
as posture and body type. Our data augmentation approach generates

Table 5
Performance comparison of CSRM using Jaccard distance and Cosine
distance.
Order

Setting

1
2
3

Baseline+SRDA
Baseline+SRDA + Cosine Distance
Baseline+SRDA + Jaccard Distance (PSSR)

CUHK-PEDES
Rank-1

mAP

74.35
74.93
75.71

69.63
72.17
73.34

Table 6
Performance comparison of SRDA using different generated images.
Order

Setting

1
2
3
4

Baseline
Baseline + Low-A
Baseline + High-A
Baseline + Low-A + High-A

CUHK-PEDES
Rank-1

mAP

70.79
73.25
72.83
74.35

64.69
67.95
67.67
69.63

neighborhoods, can reflect the consistency of samples in the local
neighborhood structure, rather than relying solely on the distance
between individual feature points. Given that cross-modal features generally exhibit distributional differences, the reliability of direct distance
measurements between text and image in local regions is limited.
Jaccard distance compensates for this structural discrepancy between
modalities, thereby enhancing the robustness of the fused features.
In order to see the overall effect more clearly, we conduct t-SNE
visualization analysis. As shown in Fig. 5, compared to the baseline,
our PSSR ensures better distribution of the data in the feature space.
Specifically, as indicated by the red elliptical box, the issue of different
categories of data being mixed together has been largely resolved.
8


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112341

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Texture-aware transformer with pose-patch mapping for occluded person
re-identiﬁcation
Dengwen Wang a , Guanyu Xing
a

b , Yanli Liu a,∗

College of Computer Science, Sichuan University, Chengdu, 610017, China

b School of Cyber Science and Engineering, Sichuan University, Chengdu, 610017, China

a r t i c l e

i n f o

Keywords:
Occluded person re-identiﬁcation
Pose-patch mapping
Texture aware decoder

a b s t r a c t
Occluded person re-identiﬁcation (re-ID) aims to retrieve the target person from occluded images captured by
diﬀerent cameras, where the challenges lie in identity loss caused by diﬀerent types of occlusion. To alleviate the
occlusion interference, some methods rely on external clues or generate more occlusion samples. However, these
methods fail to address the issues of pose misalignment under extreme occlusion and identity confusion caused
by non-target pedestrian occlusion. To solve these problems, we design a novel Texture-Aware Transformer
with Pose-Patch Mapping (TTPM), which does not require generating any occlusion samples. Speciﬁcally, a
Multi-patch Feature Encoder is proposed to encode discriminative features from inter patches and intra patches.
Afterwards, the Pose-Patch Mapping is designed to construct a positional mapping between poses and patches,
which highlights human patches and weakens the impact of occluded patches. Finally, to mitigate the non-target
pedestrian occlusion, a Texture-Aware Decoder is introduced to perceive texture features and leverage their
distinctiveness to enhance the representation of important regions. Extensive experiments show that our method
achieves state-of-the-art results on Occluded-Duke and Occluded-REID datasets.

1. Introduction
Person re-identiﬁcation (re-ID) aims to retrieve a target person from
multiple non-overlapping camera views. It provides critical technical
support for tracking suspects and ﬁnding lost people. Due to the impact
of imaging environments, the target person is often occluded by uncontrollable factors, including object occlusion (e.g., cars, billboards, and
trees) and non-target pedestrian occlusion, which often lead to significant loss of body information and increase the complexity of person
re-identiﬁcation. Therefore, re-identifying persons under complex occlusions in real-world scenarios poses a critical challenge for advancing
the practical applications of person re-identiﬁcation [1].
In recent years, researchers have designed various approaches to occluded person re-identiﬁcation from diﬀerent perspectives. These methods primarily fall into the following two categories: (1) Utilize extra
pre-trained pose estimators or human parsing models [2,3] to locate the
unoccluded regions of the human body. With additional semantic guidance, non-human features can be removed; (2) Generate more occlusion
samples through occlusion augmentation to enhance the perception of
occlusion, including randomly generating occlusion noise and manually
cropping occluding objects to paste onto the person images [4,5].

While the above methods perform well on holistic and objectoccluded person re-ID, it is still diﬃcult to maintain the same performance in cases of extreme object occlusion and non-target pedestrian
occlusion, as shown in Fig. 1(a). We attribute this performance degradation primarily to the following two reasons: (1) Under extreme object
occlusion, for some pose-based methods, owing to their mapping from
poses to global features, errors in local pose estimation can directly affect the eﬀectiveness of global features. In such cases, features highlighted by pose information usually introduce additional foreground
noise. (2) Under non-target pedestrian occlusion, the aforementioned
methods focus on separating occluded and non-occluded features, while
failing to fully capture distinctive and useful individual appearances
(e.g., backpacks, umbrellas, and clothes). As we know, when our human
eye re-identiﬁes a target person occluded by non-target pedestrians, the
perception of discriminative appearance often plays a decisive role in
the re-identiﬁcation process.
However, few studies simultaneously focuse on these two issues.
Some methods [6,7] directly use the output of pose estimation models as local masks without adaptive adjustment according to occlusion
conditions. For instance, Gao et al. [2] proposed a pose-guided visible part matching method, which estimates whether parts are occluded

∗ Corresponding author.

E-mail address: yanliliu@scu.edu.cn (Y. Liu).
https://doi.org/10.1016/j.patcog.2025.112341
Received 8 January 2025; Received in revised form 31 July 2025; Accepted 21 August 2025
Available online 24 August 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112341

D. Wang et al.

To implement these ideas, we propose a Texture-Aware Transformer
with Pose-Patch Mapping (TTPM) that relies solely on real data. It
is composed of three parts: a Multi-patch Feature Encoder (MFE), a
Pose-Patch Mapping (PPM), and a Texture-Aware Decoder (TAD). To
ensure that the patch features are rich in semantics while remaining
compact, MFE is proposed to encode features at two diﬀerent levels:
inter-patches and intra-patches. To eliminate occlusion semantics from
human features, PPM is developed to enhance pose-semantic awareness by calculating a similarity correlation matrix between patches and
poses, thereby weakening the representation of occluded patches. However, due to the presence of similar postures in non-target pedestrian
occlusions, PPM still cannot thoroughly eliminate the impact of pedestrian occlusion. To further address this issue, TAD uses the pose features
ﬁltered by PPM as constraints, and learns discriminative textures hierarchically through multi-head attention and cross-attention mechanisms.
Then, the discriminative textures are leveraged to enhance the eﬀective semantic representation of relevant regions, serving as a signiﬁcant
distinction for diﬀerent identities. As shown in Fig. 1(b), even under extreme object occlusion and non-target pedestrian occlusion, meaningful
regions still receive the appropriate attention.
In summary, the main contributions of this work are three-fold:

Fig. 1. Comparison between diﬀerent re-ID methods for two challenging cases:
extreme object occlusion and pedestrian occlusion. In Fig. (a), the (i) column
represents the original query image, the (ii) column shows the pose estimation
image, and the (iii) column denotes the human parsing image. In scenarios of
extreme object occlusion and non-target pedestrian occlusion, methods relying
solely on pose and human parsing inherently introduce occlusion noise, leading
to inaccurate feature representations. In contrast, Fig. (b) presents the attention
heatmaps of the proposed TTPM (on the right). Visualization results demonstrate
that TTPM is able to consistently focus on semantically discriminative regions,
even under these challenging occlusion cases.

•

A pose-patch mapping is proposed that constructs a direct mapping
from pose to patch, ensuring accurate alignment between the pose
and occluded pedestrians without any additional occlusion augmentation.
• A texture-aware decoder is designed that enhances the semantic representation of the eﬀective regions by hierarchically learning discriminative textures, serving as an important distinction for diﬀerent
identities.
• We conduct extensive experiments on six benchmark datasets across
three categories (i.e., occluded, holistic, and partial re-ID datasets)
and demonstrate that the proposed method outperforms 31 re-ID
methods. Especially compared to the latest method, MTIPE [11], it
shows improvements of mAP (+5.6 %) and Rank-1 (+7.0 %) on the
Occluded-Duke dataset.

through pose-guided attention and trains a visibility prediction model
in a self-supervised manner. Alternatively, some methods [8,9] utilize
customized attention mechanisms to extract specialized discriminative
semantics without pre-introducing human topological structures. For example, Li et al. [9] introduced a second-order attention mechanism to
model high-order semantic relationships among diﬀerent class tokens,
thereby enabling a more comprehensive capture of structural dependencies between human body parts. Despite their eﬀectiveness in speciﬁc
scenarios, pose-based methods struggle to generalize under extreme occlusions, while local attention-based methods often fail to guarantee the
extraction of reliable and holistic human appearance features.
In this paper, we attempt to provide a spatial alignment framework
for texture features through pose features, while texture features supplement pose features with ﬁne-grained discriminative clues. The combination of the two forms dual constraints of structure and detail. Speciﬁcally, instead of performing pose-to-global mapping, observing that the
re-ID based on Vision Transformer (Vit) [10] usually encodes the input
image into a set of patch sequences where diﬀerent patches vary in occlusion situations, we seek to establish a direct mapping from pose to
patch. Specially, we utilize smaller-scale patch mappings and pose-patch
correlation matrices to highlight human semantics in diﬀerent patches.
By calculating the similarity score between poses and patches, the negative impact caused by inaccurate pose estimation regions (i.e. regions
with lower similarity scores) can be eﬀectively reduced. Additionally,
since the occlusion situation of each patch is diﬀerent, constraining the
pose mapping of patches separately is beneﬁcial for guiding the model
to focus more on less-occluded areas. However, as non-target pedestrians can also be detected with pose skeletons, relying solely on pose
information is insuﬃcient to separate pedestrian occlusion from the target person. To tackle this issue, we attempt to leverage distinctive appearance as the primary basis for distinguishing target pedestrians from
non-targets. As the appearance of target and non-target pedestrians often exhibits notable texture variations, it is more feasible to leverage
these texture variations to enhance the representation of target regions
rather than directly identifying discriminative features from the overall
appearance. Therefore, we employ textures to augment the representation of discriminative appearance, thereby increasing the inter-class
distance between diﬀerent identities.

The remainder of this paper is structured as follows: Section 2 provides a brief review of related work on occluded re-ID. Section 3 introduces the pipeline of the proposed TTPM. Section 4 presents the experimental details, results, and analyses. Section 5 concludes the paper with
a summary.
2. Related work
2.1. Occluded person Re-ID
Existing occluded person re-ID methods can be roughly divided into
three streams, including occlusion augmentation-based methods, partbased methods, and external clue-based methods.
Occlusion augmentation-based methods enhance the adaptability of
the model to occlusion by simulating and generating more occlusion
samples. Dong et al. [4] improve the resistance of model to occlusion
noise by generating adversarial representations through random erasing
and feature perturbation. Bian et al. [5] develop a prior knowledgebased occlusion simulation approach, designing a novel OcclusionAware Feature Recovery (OAFR) network. This network leverages unoccluded gallery features to recover occluded query features. Chen et
al. [12] enhance occlusion robustness by simulating realistic, labeled occlusions during training and unifying occlusion types during inference
to reduce ambiguity and improve attention to visible body parts. The
aforementioned methods improve the re-ID performance of the model.
However, due to the overly simplistic and random strategy for generating occlusion, they may fail to maintain stable performance.
Part-based methods use a part-to-part region matching strategy to
solve the occlusion problem. For example, Somers et al. [13] propose
2

Pattern Recognition 172 (2026) 112341

D. Wang et al.

3.1. Overview

Body Part-Based Representation Learning ReID (BPBReID), which predicts part attention maps and generates body part features to leverage
identity and prior human topology information. Yan et al. [14] fully
leverage partial correlations to aggregate discriminative information between part features and enhance their signiﬁcance by suppressing occluded category features. Wang et al. [15] introduce jointly learning
part features and predicting part quality scores, automatically assigning
low scores to occluded body parts. Nevertheless, part-based methods
may not ensure the consistency of image matching in cases of severe
occlusion.
External clue-based methods mainly locate the non-occluded areas
of the human body through pose estimation or human parsing. PVPM
[2] utilizes pose information to achieve visibility matching of person
images and jointly learned discriminative features through attention to
mine the visibility of certain regions. Li et al. [3] propose a part-guided
representation method consisting of pose-invariant features and local
description features to infer pose-invariant representation through pose
estimation and normalization. Yuan et al. [11] employed a pose aggregation module to guide the learning of visible body parts and suppress occlusion noise by matching non-occluded regions based on pose-aligned
features. These methods can more accurately locate human body regions with the help of extra clues, but they cannot solve the problem of
occlusion caused by multiple pedestrians.

The overall network architecture of the proposed TTPM is outlined
in Fig. 2. Given a person image 𝑋 ∈ ℝ𝐻×𝑊 ×𝐶 , where 𝐻, 𝑊 , and 𝐶 are
the height, width, and channel dimensions of 𝑋, respectively, we feed
it into both the pose estimator HRnet [20] and the Vit [10] branches
to localize the target person accurately. In the pose estimation branch,
the estimator extracts 𝑀 landmarks from the input image 𝑋, each of
which corresponds to a keypoint heatmap {𝑦𝑖 |𝑖 = 1, 2, … , 𝑀 }. In the
Vit branch, the image 𝑋 is partitioned into 𝑁 non-overlapping patches
{𝑥𝑖 |𝑖 = 1, 2, … , 𝑁 }, which are mapped to 𝐷 dimensions through the linear projection function  (⋅) to generate a sequence of inputs 0 . Additionally, a learnable embedding token 𝑥𝑐𝑙𝑠 is added to the input sequence
to capture global representations of each patch. Similarly, following He
[17], we also incorporate learnable position embeddings  ∈ ℝ(𝑁+1)×𝐷
and camera viewpoint  ∈ ℝ(𝑁+1)×𝐷 as the crucial weights for the input sequence. The initial input sequence for the transformer encoder is
represented as:
[
( ) ( )
( )]
0 = 𝑥cls ;  𝑥1 ;  𝑥2 ; ⋯ ;  𝑥𝑁 +  + 𝜆
(1)
where 0 is the initial sequence input to transformer layer, and 𝜆 is a
hyperparameter used to balance the camera embedding.
Subsequently, the patch sequence 0 is fed into the MFE, which
encodes features at two levels: intra-patch and inter-patch, to explore
the relationships between diﬀerent body parts. The pose features 𝑦𝑖
and patch sequences 0 are input separately into PPM, constructing a
similarity mapping between poses and patches to avoid coarse-grained
interference under global mapping. Further details are available in
Sections 3.2 and 3.3.
Since transformer-based encoding and pose mapping focus more on
deep feature extraction and precise matching, they fail to address the
retrieval challenges caused by occlusions from non-target pedestrians.
To address this issue, we introduce a texture-aware decoder consisting
of a texture-extraction layer and a context-extraction layer. The former
is designed to capture the contextual feature of the original sequence
0 , while the latter applies cross-attention between the contextual features and patch features to extract distinctive textures. The structure of
TAD is presented in Section 3.4. To improve the re-ID accuracy of our
method, we employ encoder loss, decoder loss, and pose loss as supervision, which are introduced in Section 3.5.

2.2. Transformer-based person Re-ID
Transformer is a deep learning network widely known for its use of
attention mechanisms. Inspired by its excellent performance in natural
language processing, several works have attempted to apply it to computer vision. Carion et al. [16] ﬁrst introduce transformer into vision
tasks, proposing a Detection Transformer (DETR) model, which regards
object detection as a direct set prediction problem. Importantly, Dosovit
et al. [10] creatively present Vision Transformer (Vit) to process images
into patch sequences and demonstrate superior performance in traditional vision tasks.
Recently, some researchers have started to explore the application
of transformer in occluded person re-ID. For example, TransReID [17]
applies a pure transformer architecture based on Vit for re-identiﬁcation
and uses side information embedding to learn camera view clues, helping to mitigate feature bias caused by diﬀerent camera perspectives.
Wang et al. [18] employ transformer to occlusion augmentation and
propose a feature completion decoder that aggregates potential information to generate occluded features. Li et al. [9] introduced a secondorder attention mechanism to model high-order semantic relationships
among diﬀerent class tokens, thereby enabling a more comprehensive
capture of structural dependencies between human body parts. Wang et
al. [19] present a transformer-based Pose-guided Feature Disentangling
(PFD) approach, which selectively matches non-occluded parts using
pose information and automatically separates occluded semantics from
non-occluded semantics. In PFD, pose information is directly embedded into both global contextual features and local features. Although a
learnable semantic view is introduced to implicitly enhance discriminative features, it still fails to eﬀectively map pose positions to human
semantics.
In the paper, we present a transformer architecture with explicit
pose-patch mapping and introduce a texture-aware decoder to realize
eﬀective separation of occluded and non-occluded human semantics.

3.2. Multi-patch feature encoder
Under the transformer architecture, patches often have suﬃcient
global perception information but lack exploration of multi-level features within the patches. Therefore, we construct a multi-level patch
encoding to obtain semantics within patches and the interrelationships
between patches. As shown in Step 1 of Fig. 2, it includes two levels:
the intra-patch feature encoding and the inter-patch feature encoding.
First, the input embeddings 0 are split into 𝑘 groups:
⌊ ( )⌋
𝑙𝑒𝑛 0
(
)
𝑧 = 𝑠𝑝𝑙𝑖𝑡 0 [1 ∶], 𝑙 , 𝑙 =
(2)
𝑘
where 𝑠𝑝𝑙𝑖𝑡(⋅, ⋅) denotes the segmentation function, 𝑙 is the split length,
0 [1 ∶] is the input sequence 0 with 𝑥cls removed, and 𝑙𝑒𝑛(⋅) represents
the length of the current feature.
At the intra-patch level, patch features from the sequence 𝑧 =
[
]
𝑧0 , 𝑧1 , … , 𝑧𝑖 , … , 𝑧𝑘 and the embedding token 𝑥cls are separately input
into independent transformer layers. After the patch features and embedding token are concatenated, they are ﬂattened into three vectors
𝑄, 𝐾, 𝑉 ∈ ℝ𝑁×𝑑 , where 𝑑 denotes the embedding dimension. The attention weight calculation process for intra-patch feature encoding is as
follows:
(
)
𝐶𝑖 = 𝑐𝑎𝑡 𝑥𝑐𝑙𝑠 , 𝑧𝑖 ,
(3)
(
(
( )))
′
𝐼𝑖 = 𝜌 𝑓 𝑙𝑎𝑡𝑡𝑒𝑛 𝑛𝑜𝑟𝑚 𝐶𝑖
+ 𝐶𝑖 ,
(4)

3. Proposed method
This section details the proposed Texture-Aware Transformer with
Pose-Patch Mapping (TTPM). We ﬁrst present an overview of TTPM in
Section 3.1 and then introduce the motivation, objectives, and implementation of Multi-patch Feature Encoder (MFE), Pose-Patch Mapping
(PPM), and Texture-Aware Decoder (TAD) from Sections 3.2 to 3.4.
3

Pattern Recognition 172 (2026) 112341

D. Wang et al.

Fig. 2. The framework of Texture-Aware Transformer with Pose-Patch Mapping (TTPM). TTPM consists of three steps, including Step 1: the Multi-patch Feature
Encoder, which mines features from intra-patch and inter-patch levels. Step 2: the Pose-Patch Mapping, which separates pose semantics and occlusion semantics
through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.

(
( ))
𝐼𝑖 = 𝜗 𝑛𝑜𝑟𝑚 𝐼𝑖′ + 𝐼𝑖′ .

it better measures the feature distance under diverse scales and distributions, alleviating the inﬂuence of diﬀerent feature scales. The generic
formula for similarity calculation is as follows:
(
)
𝑆̃ 𝑥𝑖 , 𝑦𝑖 = 𝛾𝐷𝐌 − 𝜂𝑆𝐌
(8)

(5)

where 𝑐𝑎𝑡(⋅, ⋅) indicates concatenation along a speciﬁc channel. 𝜌(⋅)
includes softmax, linear variation, and element-wise multiplication.
𝑓 𝑙𝑎𝑡𝑡𝑒𝑛(⋅) denotes the ﬂattening operation. 𝑛𝑜𝑟𝑚(⋅) represents layer normalization. 𝜗(⋅) comprises mlp, gelu, and dropout operations.
At the inter-patch level, to further explore features within larger
patches and between patches, self-attention learning is implemented
again based on the previous level. Speciﬁcally, the intra-patch fea[
]
tures 𝐼 = 𝐼0 , 𝐼1 , … , 𝐼𝑖 , … , 𝐼𝑘 obtained from the previous level are regrouped by a factor of two to acquire larger patches. These patches are
concatenated with the embedding token 𝑥cls , which is deﬁned as:
(
(
))
𝐶𝑖′ = 𝑐𝑎𝑡 𝑥𝑐𝑙𝑠 , 𝑐𝑎𝑡 𝐼𝑖 , 𝐼𝑖+1
(6)

where 𝐷𝐌 and 𝑆𝐌 represent the Mahalanobis distance and Cosine similarity, respectively. 𝛾 and 𝜂 (𝛾, 𝜂 ≥ 0) are the related matrices weights.
The ﬁnal similarity model is determined from two perspectives: patch
distance and similarity.
To compensate for the limitations of linear transformations, we
extend the two linear projections {𝐔𝑥, 𝐔𝑦} to aﬃne transformations
{𝐀 𝑥 + 𝐚, 𝐀 𝑦 + 𝐛}. The aﬃne Mahalanobis distance 𝐷𝐌 in Eq. (8) can
be expressed as:
) (
)‖2
‖(
𝐷𝐌 = ‖ 𝐀 𝑥𝑖 + 𝐚 − 𝐁 𝑦𝑖 + 𝐛 ‖
‖
‖2
𝑥𝑖 ⎤
(9)
⎡
[
]
= 𝑥𝑇𝑖
𝑦𝑇𝑖
1 𝔻𝑀 ⎢ 𝑦𝑖 ⎥
⎢
⎥
⎣ 1 ⎦

Then, 𝐶𝑖′ is input into the same self-attention process described in
Eqs. (4), (5) to extract the inter-patch feature 𝐼̃𝑖 .
Note that, due to the information aggregation ability of tokens, the
ﬁnal encoded features are denoted by the token at the embedding position. The process can be expressed as follows:
[
]
𝐹𝑎 = 𝐼0 [∶, 0], 𝐼1 [∶, 0], … , 𝐼̃𝑀−1 [∶, 0], 𝐼̃𝑀 [∶, 0]
(7)

where ‖⋅‖22 represents the euclidean norm, which is the sum of the
squares of all elements in the current vector. The inputs for the similarity
calculation are the pose features 𝑦𝑖 and the patch features 𝑥𝑖 output by
MFE. To avoid introducing unnecessary scale changes in the initial stage,
the weight matrices 𝐀 and 𝐁 are initialized with random orthogonal
matrices, while the aﬃne transformation biases 𝐚 and 𝐛 are initialized
as the mean oﬀset of 𝑥𝑖 and 𝑦𝑖 . Thus, 𝔻𝑀 can be further expanded as:
(
)
𝐀𝑇 𝐀
−𝐀𝑇 𝐁
𝐀𝑇 𝐚 − 𝐛 ⎤
⎡
(
)
𝐁𝑇 𝐁
𝐁𝑇 𝐛 − 𝐚 ⎥
𝔻𝑀 = ⎢ −𝐁𝑇 𝐀
(10)
)
(
)
⎢( 𝑇
⎥
𝐛𝑇 − 𝐚𝑇 𝐁
‖𝐚 − 𝐛‖22 ⎦
⎣ 𝐚 − 𝐛𝑇 𝐀

where 𝐼𝑖 [∶, 0] represents the ﬁrst element in the second dimension, as
shown by the red square in Step 1 of Fig. 2. Due to the multi-level encoding process, the ﬁnal number of features is 𝑀, instead of 𝑁.
3.3. Pose-patch mapping
In real-world scenarios, the type, size, and position of occlusions result in varying occlusion situations across diﬀerent regions of the target person, which limits the positive impact of prior poses. Therefore,
we treat patch sequences and pose features as two diﬀerent modalities
{𝑥𝑖 , 𝑦𝑖 } and constrain the mapping process by calculating the similarity
matrix between poses and patches. To avoid the complex deformation
loss and mapping instability inherent in traditional linear similarity constraints, we introduce a new mapping algorithm based on mahalanobis
distance and cosine similarity. Compared to a single cosine similarity,

Similarly, the Cosine similarity calculation 𝑆𝐌 in Eq. (8) is expressed
using the aforementioned aﬃne transformation as:
⌢)
(⌢
⌢)𝑇 (⌢
𝑆𝐌 =  𝐀 𝑥𝑖 + 𝐚
 𝐁 𝑦𝑖 + 𝐛
=

4

[

𝑥𝑇𝑖

𝑦𝑇𝑖

1

]

⎡ 𝑥𝑖
𝕊𝑀 ⎢ 𝑦𝑖
⎢
⎣ 1

⎤
⎥
⎥
⎦

(11)

Pattern Recognition 172 (2026) 112341

D. Wang et al.
⌢

⌢

⌢

⌢

non-target pedestrians. Additionally, PPM relies on the accuracy of pose
localization and lacks the ability to perceive ﬁne-grained appearance
features. To address these problems, we develop a Texture-Aware De{
}
coder (TAD) for learning of the features 𝜌, 𝐹𝑠 , 𝐹𝑎 . As shown in Fig. 3,
diﬀerent from the traditional transformer architecture, our TAD comprises two types of layers: the context extraction layer is based on a
multi-head attention mechanism, which captures contextual dependencies of textures from diﬀerent perspectives through parallel multiple attention heads, and adaptively aggregates global semantic information.
The texture extraction layer utilizes a cross attention mechanism to establish a unidirectional guidance channel between texture and contextual features. By dynamically ﬁltering and enhancing key texture regions
based on contextual semantics and pose features, it enables the learning of more discriminative texture appearance features. The detailed
description is as follows.
On the left side of Fig. 3, the CELs have a multi-layer structure, which
(
( ) )
can be formally expressed as 𝐹𝑐𝑁 = 𝐶𝐸𝐿 ⋯ 𝐶𝐸𝐿 0 ⋯ . 0 is original sequence inputs in Eq. (1). Through multiple layers of CELs, local
and global dependencies in the sequence are calculated. And then, the
contextual features 𝐹𝑐𝑁 are used as the input of TELs to assist the mutual information attention from appearance to texture. Similar to CEL,
TEL also adopts a multi-layer structure, which can be represented as
(
(
) )
𝐹𝑡𝑖 = 𝑇 𝐸𝐿 𝜌, 𝐹𝑐𝑁 , ⋯ 𝑇 𝐸𝐿 𝜌, 𝐹𝑐𝑁 , 𝐹𝑠 ⋯ . The other two inputs, 𝜌 and 𝐹𝑠 ,
represent the learnable query prototype and the keypoint features after
PPM (in Eq. (18)), respectively. Speciﬁcally, 𝐹𝑠 is the initial input of
TEL as the initial value. After the CELs and TELs, TAD computes the
cosine similarity between the obtained texture-aware feature 𝐹𝑡 and the
patch appearance features 𝐹𝑎 to determine the corresponding feature
locations. TAD leverages the continuity of texture and contextual features to further augment the target features that are corrupted by other
pedestrian occlusions. The speciﬁc process can be formulated as:
( (
))
𝐼𝑛𝑑 = 𝜓𝑖𝑛𝑑 𝑆𝐌 𝐹𝑎 , 𝐹𝑡 ,
(19)

where  𝐀 𝑥𝑖 + 𝐚 and  𝐁 𝑦𝑖 + 𝐛 also belong to aﬃne transformation
{𝐴 𝑥 + 𝐚, 𝐁 𝑦 + 𝐛}, but they are described with diﬀerent symbols to dis⌢

⌢

tinguish the Mahalanobis distance. The weight matrices  𝐀 and  𝐁 are
initialized with random orthogonal matrices, while the aﬃne transfor⌢

⌢

mation biases 𝐚 and 𝐛 are initialized to 0 to eliminate absolute positional deviations between features. Furthermore, 𝕊𝑀 can be further
mathematically expanded as follows:
⎡
⎢ 𝟎
⎢ ⌢𝑇 ⌢
𝕊𝑀 = ⎢  𝐁  𝐀
⎢ ⌢𝑇2⌢
⎢ 𝐛 𝐀
⎣ 2

⌢𝑇 ⌢

𝐀𝐁
2

𝟎
⌢𝑇 ⌢

𝐚 𝐁
2

⌢𝑇 ⌢

𝐀 𝐛 ⎤
2 ⎥

⌢𝑇 ⌢ ⎥

𝐁 𝐚 ⎥
2 ⎥
⌢𝑇 ⌢⎥

(12)

𝐚 𝐛⎦

Therefore, we can derive the mathematical expression for 𝕊 in Eq. (8)
based on the Mahalanobis distance 𝐷𝐌 and Cosine similarity 𝑆𝐌 .
(
)
𝑆̃ 𝑥𝑖 , 𝑦𝑖 =𝛾𝐷𝐌 − 𝜂𝑆𝐌
𝐀
[
]⎡
= 𝑥𝑇𝑖 𝑦𝑇𝑖 1 ⎢ 𝐂𝑇
⎢ 𝑇
⎣ 𝐝

⎤⎡ 𝑥𝑖 ⎤
⎥⎢ 𝑦 ⎥
⎥⎢ 𝑖 ⎥
⎦⎣ 1 ⎦
(
)𝑇 (
)
2
2
𝑇
‖
‖
‖
‖
=‖𝐀𝑥𝑖 ‖ + ‖𝐁𝑦𝑖 ‖ + 2𝐝 𝑥𝑖 − 2 𝐂𝑥𝑖
𝐂𝑦𝑖 + 2𝐞𝑇 𝑦𝑖 + 𝑓
𝐂
𝐁
𝐞𝑇

𝐝
𝐞
𝑓

(13)

where
𝐀 = 𝛾𝐀𝑇 𝐀 ,
𝐁 = 𝛾𝐁𝑇 𝐁 ,
𝐂 = −𝛾𝐀𝑇 𝐁 − 𝜂

⌢𝑇 ⌢

𝐀𝐁
,
2
⌢𝑇 ⌢

𝐝 = 𝛾𝐀𝑇 (𝐚 − 𝐛) − 𝜂

𝐀 𝐛
,
2

(14)

⌢𝑇 ⌢

𝐁 𝐚
,
2
⌢
𝑇
⌢

𝐞 = 𝛾𝐁𝑇 (𝐛 − 𝐚) − 𝜂

𝐹𝑑𝑒 = 𝐹𝑎 [𝑖][𝐼𝑛𝑑[𝑖]] + 𝐹𝑠 [𝑖][𝑗].

𝑓 = 𝛾‖𝐚 − 𝐛‖22 − 𝜂 𝐚 𝐛 .

where 𝜓𝑖𝑛𝑑 denotes the index of the maximum value in the current matrix. 𝑆𝐌 represents the cosine similarity calculation. 𝑖 and 𝑗 represent
the batch index and patch index, respectively. 𝐹𝑑𝑒 is the decoding result
after texture augmentation. Next, we provide a detailed introduction to
the functions and implementation of CEL and TEL.

In the above formulas, 𝐀, 𝐁, 𝐂, 𝐝, 𝐞 and 𝑓 represent the parameters of the general mapping model. If there exists a certain set of
{𝐀, 𝐁, 𝐂, 𝐝, 𝐞, 𝑓 }, it means that there is at least one set of solutions 𝐷𝐌
and 𝑆𝐌 can represent the mapping matrix. That is, 𝑆𝐌 can be guaranteed to be decomposed by weighted Mahalanobis distance and Cosine
similarity.
(
)
As shown in Step 2 of Fig. 2, based on the mapping weights 𝑆̃ 𝑥𝑖 , 𝑦𝑖
{
}
in Eq. (8) of 𝑥𝑖 → 𝑦𝑖 , the 𝑖th activation features 𝑓𝑠𝑖 from pose 𝑦𝑖 to
patch 𝑥𝑖 can be calculated as:
(
(
))
𝑘 = 𝑎𝑟𝑔𝑚𝑖𝑛 1 − 𝓁 𝑆̃ 𝑥𝑖 , 𝑦𝑖 ,
(15)

3.4.1. Context extraction layer
In detail, the structure of the 𝑖th CEL is depicted in the top right
of Fig. 3. It performs Multi-Head Self-Attention (MHSA) computations
using residual connections. The calculation process is expressed as:
(
(
))
̃ 𝑖0 = 𝐼𝑁 𝑖−1
+ 𝑀𝐻𝑆𝐴 𝑖−1
, 𝑖−1
, 𝑖−1
(21)
0
0
0
0

𝑖

𝑓𝑠𝑖 = 𝐱 + 𝐲𝑘 .

(20)

(16)

where 𝑖−1
is the (𝑖 − 1)th context-extraction feature. 𝐼𝑁(⋅) represent
0
the instance normalization [21]. Then, 𝑖−1
ﬂows into the Multi-Layer
0
Perceptron (MLP) to capture more complex feature information.
(
( ))
𝐹𝑐𝑁 = 𝐼𝑁 ̃ 𝑖0 + 𝑀𝐿𝑃 ̃ 𝑖0
(22)

where 𝑘 is the position index with the highest similarity. {𝐱, 𝐲} represents the feature vectors of all {𝑥𝑖 , 𝑦𝑖 } pairs. 𝓁 represents the initialization heatmap conﬁdence after normalizing the pose features, and its
calculation process is as follows:
{
0 𝑐𝑖 < 𝜍
𝓁=
(𝑖 = 1, … , 𝑀).
(17)
1 𝑐𝑖 ≥ 𝜍

After 𝑁 layers of CELs, the contextual features 𝐹𝑐𝑁 can be output and
fed into each TEL for disentangled learning from appearance to texture.

where 𝑐𝑖 denotes the conﬁdence score of 𝑖th landmark. 𝜍 is the threshold
for ﬁltering high-conﬁdence and low-conﬁdence landmarks. Ultimately,
the keypoint features 𝐹𝑠 with high conﬁdence 𝓁, which have undergone
similarity mapping, are discribed as:
[
]
𝐹𝑠 = 𝑓𝑠1 , 𝑓𝑠2 , … , 𝑓𝑠𝑀 .
(18)

3.4.2. Texture extraction layer
To be speciﬁc, the structure of the 𝑖th TEL is shown in the bottom
right of Fig. 3. Firstly, TELs also relies on a residual structure and applies MHSA to capture key information in the features 𝐹𝑡𝑖−1 , where 𝑖 − 1
denotes the output of the (𝑖 − 1)th layer of TEL.
(
(
))
𝐹̃𝑡𝑖 = 𝐼𝑁 𝐹𝑡𝑖−1 + 𝑀𝐻𝑆𝐴 𝐹𝑡𝑖−1 , 𝐹𝑡𝑖−1 , 𝐹𝑡𝑖−1
(23)

3.4. Texture-aware decoder

Then, 𝜌, 𝐹𝑐𝑁 , and 𝐹̃𝑡𝑖 are input into Multi-Head Cross-Attention
(MHCA) to establish the correlation of three features. Speciﬁcally, we
use 𝜌 as the queries and 𝐹𝑐𝑁 as the keys to compute the relationships

The above process addresses object occlusion in person re-ID; however, it remains inadequate in addressing retrieval challenges caused by
5

Pattern Recognition 172 (2026) 112341

D. Wang et al.

Fig. 3. Structure of Texture-Aware Decoder (TAD). On the left is the overall framework of TAD, and on the right are the details of 𝑖th CEL and TEL.

mapping between pixels. Subsequently, with the assistance of pose mapping, 𝐹̃𝑡𝑖 is used as the value input into MHCA to achieve texture reﬁnement of 𝐹̂ 𝑖 .

deﬁned as:
𝑀
( ( ))
( )
( ))
1 ∑ ( ( ( 𝑖 ))
𝑑𝑒 = 𝑖𝑑  𝐹𝑑𝑒 + 𝑡𝑟𝑖 𝐹𝑑𝑒 +
  𝑓𝑠 + 𝑡𝑟𝑖 𝑓𝑠𝑖
𝑀 𝑖=1 𝑖𝑑

𝑡

(
(
))
𝐹̂𝑡𝑖 = 𝐼𝑁 𝐹̃𝑡𝑖 + 𝑀𝐻𝐶𝐴 𝜌, 𝐹𝑐𝑁 , 𝐹̃𝑡𝑖

(24)

(28)

In this way, 𝐹̂𝑡𝑖 gradually carries more texture information, promoting TELs to generate accurate and detailed texture features. Similarly,
at the end of 𝑖th TEL, there is a multi-layer perceptron, and the ﬁnal
texture capture process is as follows:
(
( ))
𝐹𝑡𝑖 = 𝐼𝑁 𝐹̂𝑡𝑖 + 𝑀𝐿𝑃 𝐹̂𝑡𝑖
(25)

where 𝑓𝑠𝑖 and 𝐹𝑑𝑒 indicate the pose-patch mapped feature (in Eq. (16))
and texture augmented feature (in Eq. (20)), respectively.
Pose Loss: Due to the accuracy of pose mapping aﬀects the optimization of the whole model, we add the pose loss constraint to the total
loss function. 𝑝 is introduced as one of the key factors to measure the
prediction error of the model, and it is deﬁned as follows:
⟨ 𝑖 𝑖⟩
𝐵
1 ∑ 𝑓𝑠 , 𝑓𝑙
𝑝 =
(29)
‖
𝐵 𝑖 ‖𝑓 𝑖 ‖‖
𝑓𝑙𝑖 ‖
‖ 𝑠 ‖‖
‖ ‖

After 𝑁 layers of TELs, the texture-aware feature 𝐹𝑡 is collected. In
Eq. (19), 𝐹𝑡 undergoes cosine similarity calculations (Eq. (19)) to locate
precise and ﬁne-grained textures.

where 𝐵 represents the batch size of training. ⟨⋅, ⋅⟩ indicates the inner
product. 𝑓𝑙𝑖 is the features of non-human body parts after pose mapping.
The motivation behind 𝑝 is that human body parts and non-human
body parts should not exhibit high similarity. If the similarity between
𝑓𝑠𝑖 and 𝑓𝑙𝑖 is high, 𝑝 increases, thereby constraining a more reasonable
mapping of the pose.

3.5. Loss function
The overall loss function is composed of the encoder loss, decoder
loss, and pose loss. In the training stage, 𝑡𝑜𝑡𝑎𝑙 jointly optimizes the
occlusion-aware and non-occlusion-aware of TTPM model, as described
in detail below:
𝑡𝑜𝑡𝑎𝑙 = 𝑒𝑛 + 𝑑𝑒 + 𝑝

4. Experiments
(26)
4.1. Datasets and evaluation metrics

where 𝑡𝑜𝑡𝑎𝑙 indicates the total loss function of TTPM. 𝑒𝑛 , 𝑑𝑒 , and 𝑝
indicate the encoder loss, the decoder loss, and the pose loss, respectively.
Encoder Loss: The encoder loss function is designed to optimize the
feature learning process of the encoder to obtain more discriminative
patch features. It mainly uses identity loss and triplet loss, deﬁned as:
𝐾
( ( ))
( ) 1 ∑
( ( ( ))
( ))
𝑒𝑛 = 𝑖𝑑  𝐹𝑎 + 𝑡𝑟𝑖 𝐹𝑎 +
  𝐼𝑖 + 𝑡𝑟𝑖 𝐼𝑖
𝐾 𝑖=1 𝑖𝑑

We conduct comprehensive experiments on six benchmark datasets,
including two occluded person re-ID datasets, two holistic re-ID datasets
and two partial re-ID datasets.
Occluded-Duke [7] is a subset of DukeMTMC-reID [22], which consists of 15,618 training images, 17,661 gallery images, and 2210 occluded query images. The proportion of occluded images in these sets is
9 %, 10 %, and 100 %, respectively.
Occluded-REID [1] is captured by the mobile camera on campus,
consisting of 2000 annotated images belonging to 200 identities. Each
identity has ﬁve full-body person images and ﬁve occluded person images.
Partial-REID [23] is a partial person re-ID dataset containing 600
images of 60 identities. Each identity has 5 partial images and 5 holistic
images. The partial images are used as query images, and the holistic
images as gallery images.

(27)

where (⋅) is the probability prediction function. 𝐼𝑖 and 𝐹𝑎 indicate
the intra-patch feature (in Eq. (5)) and global embedding feature (in
Eq. (7)), respectively.
Decoder Loss: To more eﬀectively guide the encoder in capturing ﬁnegrained texture semantics, the encoding loss 𝑑𝑒 is employed. It is used
to maximize the distance between identities of diﬀerent categories while
minimizing the distance between identities of the same category. 𝑑𝑒 is
6

Pattern Recognition 172 (2026) 112341

D. Wang et al.

Table 1
Quantitative Comparison of Our TTPM With State-of-the-art Methods on Occluded-Duke and Occluded-REID (%). The First Group:
CNN-based Person re-ID Methods. The Second Group: Vit-based
Person re-ID Methods. Bold indicates the Best Results.

Partial-iLIDS [24] contains 238 images of 119 identities captured
by cross cameras. Most of the images are collected in the airport, so the
lower part of the picture is often occluded by luggage.
Market-1501 [25] contains 32,668 images of 1501 identities captured by 6 camera viewpoints. The whole dataset is divided into
two parts: the training set and the testing set, with 751/12,936 and
750/19,732 identities/images, respectively.
DukeMTMC-reID [22] contains 36,411 images of 1812 identities
from 8 camera viewpoints. There are 16,522 images with 702 identities
for training, 17,661 images with 1110 identities for the gallery, and
other 2228 images with 702 identities for the query.
Evaluation metrics: Following the experimental protocol of re-ID,
the cumulative matching characteristic (CMC) and mean average precision (mAP) are adopted as the primary evaluation metrics. And all
experiments are conducted in the single query mode.
4.2. Implementation details
We adopt Vit [10] as the backbone to extract person features.
The initial weights are pre-trained on ImageNet-21K and ﬁnetuned on
ImageNet-1K. During the training stage, HRnet [20] is used as the pose
estimator, with the threshold 𝜍 set to 0.4. Furthermore, the number of
divided patches and the extracted landmarks are both set to 17, and the
hyper-parameter 𝜆 to 3. All training and testing images are resized to
256 × 128. The training images are augmented with random horizontal
ﬂipping, random cropping and random erasing as usual. The batch size is
set to 64 with 4 images per ID. The learning rate is initialized at 0.008
with cosine learning rate decay. Stochastic Gradient Descent (SGD) is
used as the optimizer, with momentum and weight decay initialized to
0.9 and 1e-4, respectively. All experiments are run in 300 epochs with
two RTX 3090 GPUs using the PyTorch toolbox.

Occluded-Duke

Occluded-REID

Rank-1

mAP

Rank-1

mAP

42.6
47.0
55.1
57.6
64.5
55.4
64.4
66.7
61.0
68.3

33.7
37.7
43.8
46.3
53.6
43.8
49.7
54.1
50.1
54.3

41.3
70.4
80.3
–
81.6
82.5
–
76.9
71.8
–

38.9
61.2
70.2
–
72.1
71.8
–
68.6
51.0
–

ICCV 21
AAAI 22
CVPR 22
TII 23
AAAI 23
TIP 24
AAAI 24
PR 25

66.4
67.7
68.1
62.8
70.0
71.8
68.6
66.4

59.2
60.1
56.4
54.9
62.2
62.2
57.4
57.8

70.2
79.8
86.3
80.4
83.0
82.6
86.8
–

67.3
81.3
79.3
76.1
76.8
78.2
81.3
–

–

73.4

63.4

86.8

82.2

Methods

Reference

PCB [26]
PVPM [2]
HOReID [6]
SORN [27]
PAT [8]
ASAN [28]
QPM [15]
BPBreID [13]
RTGAT [29]
PRE-Net [14]

ECCV 18
CVPR 20
CVPR 20
TCSVT 20
CVPR 21
TCSVT 21
TMM 22
WACV23
TIP 23
TCSVT 23

TransReID [17]
PFD [19]
FED [30]
SCAT [31]
SAP [32]
OAT [9]
SPT [33]
MTIPE [11]
TTPM(Ours)

Table 2
Quantitative Comparison of Our TTPM With State-of-the-art Methods
on Market-1501 and DukeMTMC-reID (%). The First Group: Holistic
Person re-ID Methods. The Second Group: Occluded Person re-ID Methods. Bold indicates the Best Results.
Methods

Reference

4.3. Comparison with the state-of-the-art
4.3.1. Results on occluded datasets
In occlusion scenarios, we compare our method with existing
approaches on two occlusion datasets, i.e., Occluded-Duke and
Occluded-REID. The compared methods include two categories: CNNbased person re-ID methods (the ﬁrst group) and Vit-based person reID methods (the second group). As shown in Table 1, the experimental results demonstrate that our TTPM achieves advanced performance
on both occlusion datasets. On the Occluded-Duke and Occluded-REID
datasets, our TTPM achieves 73.4 %/63.4 % and 86.8 %/82.2 % Rank1/mAP performance, respectively. Speciﬁcally, compared to the occlusion augmentation method FED [30], which requires additional computation to simulate occlusion data, our proposed method still shows
signiﬁcant improvements of mAP (+7.0 %) and Rank-1 (+5.3 %). On
the challenging Occluded-Duke dataset, TTPM achieves the best results compared to the state-of-the-art methods, outperforming the latest
methods SPT [33] and OAT [9] by at least 1.6 %/1.2 % Rank-1/mAP.
Note that, our TTPM does not require additional data augmentation
to generate simulated data. Instead, it only focuses on the target individual, which includes the human skeleton and ﬁne-grained texture
information, eﬀectively avoiding the negative impacts caused by object
occlusion and non-target pedestrian occlusion.

Market-1501

DukeMTMC-reID

Rank-1

mAP

Rank-1

mAP

PCB [26]
ISP [34]
TransReID [17]
HAT [35]
DAAF-BoT [36]
AAFormer [37]

ECCV 18
ECCV 20
ICCV 21
ACM MM 21
PR 22
TNNLS 23

93.8
95.3
95.2
95.6
95.1
95.4

81.6
88.6
88.9
89.5
87.9
88.0

69.2
89.6
90.7
90.4
87.9
90.1

83.3
80.0
82.0
81.4
77.9
80.9

HOReID [6]
PAT [8]
PFD [19]
ETND [4]
SCAT [31]
BPBreID [13]
RTGAT [29]
RGANet [38]
ADP [33]
FCFormer [18]
MTIPE [11]

CVPR 20
CVPR 21
AAAI 22
TCSVT 23
TII 23
WACV23
TIP 23
TIFS 23
AAAI 24
TMM 24
PR 25

94.2
95.4
95.5
95.3
95.1
95.1
95.3
95.5
95.6
95.0
95.2

84.9
88.0
89.6
87.2
88.0
87.0
88.2
89.8
89.5
86.8
89.0

86.9
88.8
90.6
88.5
89.3
89.6
89.1
–
91.2
89.7
90.1

75.6
78.2
82.2
77.9
79.8
78.3
80.2
–
83.1
78.8
80.8

TTPM (Ours)

–

95.6

90.0

91.4

83.5

achieving the best mAP (+0.3 %) compared to all person re-ID methods. Compared to the Occluded person re-ID methods, our method
achieved the best Rank-1 (91.4 %) and mAP (83.5 %) on DukeMTMCreID. Even compared to the recent state-of-the-art method FCFormer
[18], TTPM achieves enhancements on both mAP (3.2 %/4.7 %) and
Rank-1 (0.6 %/1.7 %) on Market-1501 and DukeMTMC-reID. The results
on holistic datasets indicate that TTPM not only eﬀectively mitigates
the adverse eﬀects of occlusion but also maintains strong robustness in
holistic person re-ID tasks.

4.3.2. Results on holistic datasets
Although our TTPM is proposed to address person re-ID in occlusion scenarios, it is also eﬀective for holistic person re-ID. As shown
in Table 2, to evaluate the performance of the proposed method
on holistic datasets, we conduct experiments on Market-1501 and
DukeMTMC-reID. Speciﬁcally, the compared methods are grouped into
two categories: holistic person re-ID methods (the ﬁrst group) and
occluded person re-ID methods (the second group). On the Market1501 datasets, our TTPM achieves comparable performance with 95.6 %
Rank-1 and 90.0 % mAP, respectively, ranking ﬁrst in Rank-1 and

4.3.3. Results on partial datasets
Unlike the previous two holistic datasets, the images in Partial-REID
and Partial-iLIDS appear as partially absent. The absence is generally
divided into two situations: artiﬁcial cropping and regular geometric occlusion, which presents a new challenge for methods that rely on pose
7

Pattern Recognition 172 (2026) 112341

D. Wang et al.

Table 3
Quantitative Comparison of Our TTPM With State-of-the-art Methods
on Partial-REID and Partial-iLIDS (%). The First Group: CNN-based Person re-ID Methods. The Second Group: Vit-based Person re-ID Methods.
Bold indicates the Best Results.
Partial-REID

Table 5
Contribution Analysis of TAD Components on OccludedDuke (%). TELs and CELs denote Texture extraction layers and Context extraction layers in TAD. Bold indicates
the Best Results.

Partial-iLIDS

Methods

Reference

Rank-1

Rank-3

Rank-1

Rank-3

PGFA [7]
PVPM [2]
HOReID [6]
SORN [27]
OAMN [12]
ASAN [28]
QPM [15]
PRE-Net [14]

ICCV 19
CVPR 20
CVPR 20
TCSVT 20
ICCV 21
TCSVT 21
TMM 22
TCSVT 23

68.0
75.3
85.3
76.7
86.0
55.4
77.3
86.0

80.0
86.0
91.0
84.3
–
43.8
85.7
91.3

69.1
–
72.6
79.8
77.3
82.5
81.7
78.2

80.9
–
86.4
86.6
–
88.3
88.0
87.4

TransReID [17]
PFD [19]
FED [30]
SCAT [31]
RGANet [38]

ICCV 21
AAAI 22
CVPR 22
TII 23
TIFS 24

81.3
67.6
83.1
76.3
87.2

88.0
74.3
–
85.3
93.5

73.1
69.7
–
73.9
77.0

84.9
80.7
–
86.6
87.6

TTPM(Ours)

–

87.4

93.6

77.3

88.5

Components
MFE

1
2
3
4
5

PPM

Occluded-Duke

Occluded-REID

TAD

Rank-1

mAP

Rank-1

mAP

✓

59.4
62.2
67.3
68.9
73.4

49.7
52.4
56.1
57.0
63.4

72.1
74.5
80.9
82.6
86.8

67.6
69.9
75.2
77.6
82.2

✓
✓
✓

✓
✓
✓

Rank-1

Rank-5

Rank-10

mAP

TAD w/o CELs
TAD w/o TELs
TAD

70.9
69.1
73.4

81.6
79.3
84.6

82.7
81.7
88.5

59.0
57.7
63.4

extracts only limited semantics within patches, which leads to a decrease
in both mAP and Rank-1.
4.4.2. Eﬀectiveness of the proposed PPM
We also test the eﬀects of PPM module on the performance of
TTPM. By comparing index-1 and index-3 in Table 4, we observe that
the model variant with PPM increases Rank-1/mAP by 7.9 %/6.4 % on
Occluded-Duke, which is the result under the inﬂuence of PPM alone.
Furthermore, through a comparative analysis of index-2 and index4, with the combined eﬀect of PPM and MFE, our model achieves
6.7 %/4.6 % and 8.1 %/7.7 % increases in Rank-1/mAP on two occluded
datasets, respectively. The results indicate that ensuring the eﬀectiveness of pose mapping enables the network to separate occlusion semantics from human semantics, thereby focusing on the feature regions of
the target person.
As shown in Fig. 4, we also visualize the attention heatmap of PPM
to provide a more intuitive analysis. The red box in the original image
(column i) indicates the eﬀective feature region of the target person,
which is also highlighted in the attention maps of the baseline (column
ii) and PPM (column iii). In usual object occlusion scenarios (ﬁrst row),
our PPM presents attention maps similar to the baseline, eﬀectively focusing on the valid feature regions. Even in extreme occlusion scenarios
(second row), the proposed PPM aligns its attention regions more accurately with the target areas, demonstrating the following advantages:
(1) Compared to the baseline that directly uses pose for global localization, TTPM focuses more on discriminative regions. (2) It achieves
eﬀective mapping between patches and poses, better highlighting the
visible body regions.

Table 4
Ablation Study on Occluded-Duke and Occluded-REID (%). MFE,
PPM, and TAD Respectively Denote Multi-patch Feature Encoder,
Pose-Patch Mapping and Texture-Aware Decoder. Bold indicates the
Best Results.
Index

Method

estimators. As shown in Table 3, our TTPM achieves 87.4 %/93.6 % and
77.3 %/88.5 % in terms of Rank-1/Rank-3 on Partial-REID and PartialiLIDS, respectively. The results indicate that although our method also
relies on pose estimators, the PPM module prevents feature interference from non-pose regions through its eﬀective mapping. Furthermore,
compared to the state-of-the-art method SCAT [31], our approach signiﬁcantly improves the performance of Rank-1 (+11.1 %) and Rank-3
(+8.3 %) on the Partial-REID dataset. The experimental results of Rank1 and Rank-3 validate that our TTPM also exhibits practical applicability
in partial scenarios.

4.4.3. Eﬀectiveness of the proposed TAD
Comprehensively, the eﬀectiveness analysis of TAD mainly includes
two parts: (1) the contribution of TAD to TTPM in Table 4, and (2)
the contribution of CELs and TELs to TAD in Table 5. As shown in
index-5 of Table 4, after incorporating TAD, consistent improvements in
Rank-1/mAP are achieved across two datasets. From index-4 to index5, when TAD is employed, the overall performance rises by 4.5 %/6.4 %
and 4.2 %/4.6 % in Rank-1/mAP, respectively. TAD leverages the continuity of texture features to further enhance the target features that
are corrupted by pedestrian occlusion. Intuitively, we visualize the retrieval results of TTPM. As shown in Fig. 5, under both object occlusion
and non-target pedestrian occlusion, our model demonstrates a notable
recognition capability. Compared to the retrieval results without the
texture-aware encoder (left column), TTPM accurately re-identiﬁes the
target person occluded by other pedestrians (the last two columns of
the third row on the right) with the aid of texture-aware encoder. Additionally, we present some failure cases, as shown in Fig. 6. Although
TAD demonstrates strong adaptability to various occlusion scenarios, it
fails to extract discriminative clues for correct retrieval when facing severe overlaps between targets and non-target individuals with similar
textures. This result indicates that the proposed TTPM still has limitations in relying on texture features. While these features enable accurate
recognition in general cases, they restrict retrieval accuracy in such special scenarios. This remains a key challenge to be addressed in the future
of occluded person re-identiﬁcation.

4.4. Ablation study
The ablation study results of Multi-patch Feature Encoder (MFE),
Pose-Patch Mapping (PPM), and Texture-Aware Decoder (TAD) are
shown in Table 4. To verify the eﬀectiveness of each component, we
use the backbone of TTPM alone as the baseline (index-1) and incrementally test the model variants: baseline+MFE (index-2), baseline+PPM
(index-3), baseline+MFE+PPM (index-4), and TTPM (index-5).
4.4.1. Eﬀectiveness of the proposed MFE
We ﬁrst remove all the proposed modules, retaining only the pure
transformer-based framework as a benchmark for comparison. The
pure transformer achieves a performance (index-1) of 59.4 %/49.7 %
Rank-1/mAP on Occluded-Duke. From index-1 to index-2, it can be seen
that after adding MFE, the Rank-1 and mAP of the model increased by
2.8 %/2.4 % and 2.7 %/2.3 % on two occluded datasets, respectively.
The Rank-1/mAP of index-2 validates that MFE is eﬀective in improving
the occluded re-ID performance by enhancing the diversity of features
extracted from inter-patches and intra-patches. The model without MFE
8


codex
后三篇的主线也读到了。接下来我会针对每篇用章节标题定位，从“Method/Methodology”到“Experiments”做二次截取，补齐刚才被截断的机制细节和消融证据。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | rg -n \""'^(III|IV|A'"\\.|B\\.|C\\.|D\\.|[0-9]+\\)|Methodology|Overview|Structured|Modality|Loss|Experiments|Datasets|Implementation)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Spectrum-guided feature enhancement network for event person re-identification.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|III|IV|Method|Experiment|MSAM|CPDM|Loss|Contribution|Discussion)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
140:2.1. Various person re-ID task
176:3.1. Basic information for SFE-Net
184:MSAM and CPDM along with the data ﬂow, is illustrated in Fig. 2. This
213:2.2. Event-based vision task
222:3D perception. For temporal-aware segmentation tasks, [37] introduce
245:3.2. MSAM
268:3. Method
273:3
328:4
356:3.3. CPDM
382:Loss. Here, the ResBlock is composed of three Bottleneck Blocks [45].
404:MSAMs, namely MSAM-1, MSAM-2, and MSAM-3, in the shallow, middle, and deep sections in the ResNet-50.
419:3.4. Loss function in SFE-Net
504:4. Experiment
531:3090 in just 2.5 hours.
538:4.2. Contribution of each omponent in SFE-Net
548:19.6 1 𝑠𝑐𝑜𝑟𝑒𝑎𝑛𝑑7.9 %𝑜𝑛𝑚𝐴𝑃 𝑠𝑐𝑜𝑟𝑒𝑜𝑛𝑡ℎ𝑒𝑆𝑜𝑓 𝑡𝐵𝑖𝑜 (𝐸𝑣𝑒𝑛𝑡)𝑑𝑎𝑡𝑎𝑠𝑒𝑡,15.7 % 𝑜𝑛
560:4.1. Comparison with SOTA methods
581:4.3. Eﬀectiveness dion about MSAM
601:Methods
645:24.4
646:30.5
647:69.17
648:24.07
650:19.3
651:19.3
652:6.9
653:9.0
654:14.1
655:7.02
657:33.6 %
658:39.6 %
659:28.6 %
660:33.6 %
661:55.3 %
662:57.9 %
663:59.2 %
665:13.8 %
666:25.5 %
667:17.8 %
668:17.6 %
669:21.4 %
670:34.9 %
671:36.1 %
673:25.4 %
674:44.7 %
675:22.5 %
676:28.9 %
677:36.8 %
680:12.9 %
681:29.6 %
682:10.7 %
683:16.5 %
684:15.3 %
687:22.7 %
688:23.9 %
689:18.1 %
690:23.2 %
691:32.6 %
692:24.6 %
695:8.7 %
696:20.8 %
697:9.4 %
698:11.5 %
699:10.9 %
700:15.5 %
707:40.7
709:14.6
711:75.0 %
713:45.0 %
715:57.1 %
717:32.3 %
719:41.8 %
721:27.5 %
726:Methods
761:36.4 %
762:68.2 %
763:64.8 %
765:21.1 %
766:42.0 %
767:39.6 %
769:21.1 %
770:49.6 %
771:40.7 %
773:11.7 %
774:25.5 %
775:19.6 %
777:10.7 %
778:32.8 %
779:26.4 %
781:8.3 %
782:18.6 %
783:16.2 %
794:75.0 %
795:81.7 %
797:45.0 %
798:52.6 %
800:57.1 %
801:63.6 %
803:32.3 %
804:40.8 %
806:41.8 %
807:49.4 %
809:27.5 %
810:35.8 %
813:Contribution of each MSAM or MSAM’s Variants to
815:MSAM-1
817:MSAM-2
819:MSAM-3
831:68.2
833:42.0
840:61.5
841:64.9
842:65.3
843:60.4
844:56.6
845:51.7
847:37.4
848:38.7
849:39.2
850:35.6
851:32.2
852:30.6
854:36.4
856:21.1
896:MSAM comprises Pre-Spectrum Augmentation (Pre-SA), Spatial Interaction Enhancement (SIM), and Post-spectrum Augmentation (Post-SA).
897:MSAM encompasses three scale spectrum augmentation mechanisms:
905:Contribution of each strategy to performance in MASM.
906:MSAM-1 in SFE-Net ( Shown in Fig. 2)
934:60.4
936:35.6
952:54.8
953:56.3
954:55.6
955:52.9
956:55.0
957:51.3
959:30.2
960:32.2
961:32.8
962:27.4
963:30.1
964:30.5
966:36.4
968:21.1
993:Method
1006:38.5
1007:37.4
1008:35.5
1009:40.2
1010:52.3
1012:21.9
1013:24.1
1014:25.4
1015:23.6
1016:26.7
1018:CPDM (Ours)
1020:64.8
1022:39.6
1039:4.4. Discussion about CPDM and some dropout strategies

 succeeded in 0ms:
8:Structured Distilled 3D Gait Fields for
32:Modality-Representative Interactive Encoding (MRIE) module,
210:A. Video-Based Person Re-ID
243:B. Clothes-Changing Person Re-ID from a Single Image
265:C. Clothes-Changing Person Re-ID in Videos
289:D. Dense Correspondence Embeddings
306:III. M ETHODOLOGY
310:A. Overview
594:B. Structured Distilled 3D Gait Fields
604:1) Pre-training 3D Vertex Embeddings: Given the canonical SMPL model, we model it as a graph G = (V, A),
712:2) Formulating the Structured Distilled 3D Gait Fields:
738:3) Constructing Dense Training Data and Spatial/Temporal
839:4) Modeling 3D Gait Features via Local Similarity: Fundamentally, the 3D gait field F consists of distilled frame-level
882:C. Modality Representative Interactive Encoding
892:1) Generating Global and Local Representative Tokens:
971:2) Gait-Appearance Joint Encoding: As shown in Fig. 3(d),
1127:D. Loss Function
1132:1) Vertex Embedding Pre-training Loss: As discussed in
1179:2) 3D Gait Field Loss: According to Section III-B3, the
1251:3) Re-ID Loss: For the video-based appearance feature
1288:A. Datasets and Evaluation Protocols
1313:IV. E XPERIMENTS
1318:4) Overall Training Objective: In the end-to-end training
1341:B. Implementation Details
1342:1) Pre-training Stage: During the pre-training stage of 3D

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Experiment|Stochastic|Cross|Class|Ablation|Which|Effectiveness)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
10:Stochastic style perturbation modelling for visible-Infrared person
30:Cross-modality learning
166:3. Method
167:3.1. Stochastic style perturbation
168:3.1.1. Background
258:3.1.2. Feature-level stochastic style perturbation
290:3
414:3.3. Class-aware contrast similarity learning
415:Cross distribution alignment regularization mainly focus on the
443:3.2. Cross distribution alignment
452:4
476:3.4. Training objective
506:3.5. Experiments
507:3.5.1. Datasets and evaluation metric
578:Method
670:14.80
671:26.97
672:27.82
673:28.90
674:47.50
675:50.36
676:53.14
677:64.77
678:65.40
679:64.23
680:65.74
681:60.80
682:66.34
683:68.72
684:67.53
685:69.90
686:73.89
687:66.76
688:69.63
689:70.58
690:71.68
691:68.81
692:65.45
693:71.48
694:74.44
695:72.23
696:76.11
698:54.12
699:67.51
700:67.34
701:70.60
702:84.39
703:89.02
704:89.61
706:93.33
707:95.19
708:92.98
709:92.10
711:96.39
712:95.36
713:95.70
714:96.27
715:96.23
716:96.27
717:96.21
719:93.08
720:94.47
721:96.23
723:97.04
724:96.40
726:71.33
727:80.56
728:81.34
729:82.40
730:92.14
731:95.92
732:96.74
734:97.14
735:98.73
736:97.05
737:96.80
739:98.96
740:98.64
741:98.50
742:98.84
743:98.70
744:98.82
745:98.80
747:96.84
748:98.16
749:98.71
752:98.66
754:15.95
755:31.49
756:28.42
757:29.20
758:47.65
759:47.36
760:48.16
761:59.23
762:61.98
763:61.21
764:60.83
765:58.90
766:66.29
767:66.29
768:64.98
769:66.90
770:69.47
771:64.01
772:66.11
773:68.24
774:68.79
775:63.96
776:62.90
777:68.15
778:71.17
779:70.58
780:71.55
782:20.58
783:31.63
784:32.46
786:54.17
787:53.47
788:55.21
789:65.34
790:72.56
791:64.80
792:71.60
793:68.00
794:68.15
795:72.52
796:71.66
797:76.30
798:79.35
799:72.89
800:74.68
801:76.74
802:76.97
803:76.31
804:71.46
805:78.36
806:78.98
807:83.45
808:82.29
810:68.38
811:77.23
812:77.42
814:91.14
815:92.24
816:91.44
818:96.98
819:96.92
820:96.60
821:94.80
823:97.84
824:96.73
825:97.90
826:98.32
827:97.93
828:97.51
829:98.21
831:97.28
832:97.16
833:98.36
835:98.62
836:98.64
838:85.79
839:89.18
840:89.62
842:95.98
843:97.84
844:95.83
846:99.20
847:99.31
848:99.45
849:97.90
851:99.46
852:99.25
853:99.50
854:99.77
855:99.28
856:99.14
857:99.57
859:98.91
860:99.22
861:99.78
864:99.73
866:26.92
867:42.19
868:42.46
870:62.97
871:61.73
872:61.98
873:71.46
874:76.58
875:71.53
876:80.05
877:52.40
878:74.09
879:78.17
880:76.52
881:80.40
882:82.68
883:76.70
884:78.30
885:80.95
886:81.94
887:74.52
888:76.67
889:81.98
890:82.75
891:84.25
892:84.94
894:14.80
896:34.62
897:43.40
898:70.05
899:85.63
900:83.79
901:89.95
902:80.31
903:87.35
904:76.31
905:82.80
906:89.12
907:83.60
908:84.83
909:85.00
910:85.39
911:85.51
912:88.40
913:82.80
914:87.87
915:87.04
916:88.11
917:85.69
918:84.71
919:89.03
920:88.83
922:54.12
924:58.96
925:66.10
926:87.28
927:94.08
928:95.83
930:92.70
931:97.10
932:91.02
933:95.10
937:95.50
939:94.99
940:95.10
943:95.49
945:95.45
947:98.19
948:97.38
950:71.33
952:68.72
953:76.30
954:92.04
955:95.49
956:97.82
958:96.03
959:98.90
960:94.27
961:97.70
965:97.50
967:96.70
968:96.94
971:97.57
973:97.54
976:98.35
978:15.95
980:33.46
981:44.10
982:66.37
983:76.71
984:77.87
985:82.09
986:73.07
987:76.94
988:68.00
989:79.30
990:84.43
991:75.67
992:76.55
993:79.10
994:77.30
995:79.95
996:81.45
997:80.70
998:85.09
999:81.19
1000:81.66
1001:79.70
1002:83.28
1003:83.89
1004:81.02
1006:20.58
1008:34.21
1009:43.40
1010:68.83
1011:82.91
1012:82.82
1013:90.78
1014:75.93
1015:86.41
1016:70.15
1017:81.70
1018:88.38
1019:81.97
1020:84.16
1021:84.80
1022:83.50
1023:80.65
1024:85.34
1025:83.70
1026:86.95
1027:83.54
1028:84.92
1029:84.88
1030:87.18
1031:87.91
1032:88.88
1034:68.38
1036:58.74
1037:66.10
1038:83.69
1039:92.33
1040:95.73
1042:90.93
1043:96.39
1044:85.24
1045:94.10
1049:95.30
1051:92.81
1052:94.56
1055:94.56
1057:95.86
1059:97.15
1060:97.62
1062:85.79
1064:68.64
1065:76.30
1066:88.35
1067:94.90
1068:96.89
1070:94.59
1071:98.20
1072:89.27
1073:96.90
1077:97.50
1079:96.32
1080:97.48
1083:96.84
1085:97.74
1088:98.63
1090:26.92
1092:32.49
1093:44.10
1094:64.45
1095:74.97
1096:76.74
1097:82.25
1098:69.49
1099:75.70
1100:63.77
1101:77.60
1102:83.86
1103:73.78
1104:75.13
1105:77.80
1106:75.19
1107:75.91
1108:81.19
1109:80.90
1110:84.34
1111:77.92
1112:80.79
1113:78.55
1114:84.72
1115:83.04
1116:81.86
1122:76.36 97.94 99.04 71.64 81.57
1124:98.05
1126:99.50
1128:83.95
1130:89.80 96.97
1132:98.77 84.86 87.52
1134:96.74
1136:98.22
1138:84.34
1158:Method
1196:47.50
1197:65.40
1198:65.74
1199:67.53
1200:69.63
1201:70.60
1202:60.00
1203:56.00
1204:61.60
1205:73.89
1206:74.70
1208:47.65
1209:61.98
1210:60.83
1211:64.98
1212:66.11
1213:66.90
1214:58.80
1215:53.40
1216:59.00
1217:69.47
1218:71.80
1220:36.30(↓11.20)
1221:51.88(↓13.52)
1222:50.22(↓15.52)
1223:45.89(↓21.64)
1224:57.14(↓12.49)
1225:57.30(↓13.30)
1226:44.70(↓15.30)
1227:40.30(↓15.70)
1228:47.30(↓14.30)
1229:50.22(↓23.67)
1230:63.61(↓11.09)
1232:35.54(↓12.11)
1233:48.58(↓13.37)
1234:44.27(↓16.46)
1235:40.52(↓24.46)
1236:54.30(↓11.81)
1237:54.60(↓12.30)
1238:45.60(↓13.20)
1239:38.80(↓16.60)
1240:46.90(↓12.10)
1241:49.68(↓19.79)
1242:61.67(↓10.13)
1244:31.32(↓16.68)
1245:47.67(↓17.73)
1246:47.31(↓18.83)
1247:42.66(↓24.87)
1248:52.40(↓17.23)
1249:54.70(↓15.90)
1250:39.40(↓20.60)
1251:36.10(↓19.90)
1252:42.20(↓19.40)
1253:47.39(↓26.50)
1254:55.93(↓18.77)
1256:32.24(↓15.41)
1257:44.97(↓17.01)
1258:43.02(↓17.81)
1259:38.54(↓26.44)
1260:50.10(↓16.01)
1261:50.90(↓16.00)
1262:41.50(↓17.30)
1263:35.70(↓17.70)
1264:45.20(↓13.80)
1265:43.39(↓26.08)
1266:52.94(↓18.86)
1268:70.05
1269:80.30
1270:76.31
1271:84.83
1272:88.40
1273:91.60
1274:73.20
1275:77.70
1276:83.39
1277:85.39
1278:91.10
1280:66.37
1281:73.07
1282:68.00
1283:76.55
1284:81.45
1285:84.10
1286:71.60
1287:70.44
1288:78.75
1289:77.30
1290:85.10
1292:57.52(↓12.53)
1293:56.94(↓23.36)
1294:50.15(↓26.16)
1295:64.16(↓20.67)
1296:70.14(↓18.26)
1297:74.50(↓17.10)
1298:52.14(↓21.06)
1299:57.24(↓20.46)
1300:68.89(↓14.50)
1301:63.50(↓21.89)
1302:80.69(↓10.41)
1304:52.70(↓13.67)
1305:52.91(↓20.16)
1306:45.24(↓22.76)
1307:56.35(↓20.20)
1308:65.56(↓15.89)
1309:68.42(↓15.68)
1310:48.66(↓22.94)
1311:50.78(↓19.64)
1312:64.97(↓13.78)
1313:54.88(↓22.42)
1314:73.16 (↓11.94)
1316:38.54(↓31.51)
1317:45.68(↓34.77)
1318:46.22(↓30.09)
1319:53.64(↓31.19)
1320:61.74(↓26.66)
1321:64.26(↓27.34)
1322:42.46(↓30.94)
1323:42.14(↓35.56)
1324:54.94(↓28.45)
1325:52.98(↓32.41)
1326:63.11(↓27.99)
1328:37.50(↓28.87)
1329:42.61(↓30.46)
1330:40.86(↓27.14)
1331:48.26(↓28.29)
1332:55.62(↓25.83)
1333:56.88(↓27.22)
1334:38.69(↓32.91)
1335:36.77(↓33.67)
1336:50.38(↓28.37)
1337:47.49(↓29.81)
1338:57.20(↓27.90)
1342:76.36
1344:71.64
1346:70.62(↓5.74)
1348:65.72(↓5.92)
1350:66.46(↓9.90)
1352:62.73(↓8.91)
1354:89.80
1356:84.86
1358:84.52(↓5.28)
1360:78.82(↓6.04)
1362:71.34(↓18.46)
1364:64.64(↓20.22)
1371:4:1 and 8:1, respectively. Similarly, for the RegDB dataset, which originally has a balanced ratio of 1:1 between visible and thermal images,
1381:Method
1386:40.3
1388:43.6
1390:43.8
1392:48.8
1394:52.2
1396:52.5
1399:55.4
1401:56.4
1410:71.4
1411:74.6
1412:78.2
1413:79.5
1414:80.7
1415:81.6
1416:84.9
1417:86.2
1420:79.2
1421:81.5
1422:84.3
1423:85.3
1424:87.1
1425:88.5
1426:90.3
1427:91.1
1430:79.6
1431:82.4
1432:86.6
1433:85.3
1434:87.0
1435:88.4
1436:90.9
1437:90.8
1440:48.4
1441:51.8
1442:53.1
1443:56.6
1444:59.8
1445:58.9
1446:62.9
1447:62.4
1448:63.2
1450:48.0
1451:51.5
1452:50.8
1453:56.5
1454:60.4
1455:59.9
1456:62.5
1457:60.3
1458:64.9
1460:86.1
1461:87.9
1462:91.1
1463:90.9
1464:91.9
1465:93.6
1466:94.7
1467:95.8
1470:52.3
1471:55.3
1472:55.6
1473:59.8
1474:63.2
1475:62.7
1476:65.8
1477:55.2
1478:67.9
1480:3.5.2. Implementation details
1484:58.1 89.2 92.0 64.2 66.3 92.6 96.4 68.2
1506:0.16 % in mAP. However, as the imbalance ratio widens to 4:1, our
1522:3.5.3. Modality-default setting results
1556:64.2 % in the more challenging IR-to-VIS modality, while attaining
1557:66.3 % Rank-1 accuracy and 68.2 % mAP in the VIS-to-IR modality.
1565:3.6. Ablation study
1572:65.39 % to 72.05 % in Rank-1 accuracy). This highlights the eﬀectiveness of the proposed method in mitigating modality discrepancies and
1577:Which stage to insert the SSP? The proposed SSP module is designed to be ﬂexibly integrated at any stage within the backbone network. Using a standard ResNet-50 model, which includes ﬁve residual
1587:4 is pivotal in capturing label-speciﬁc information, and destabilizing the
1589:3.5.4. Modality-imbalance setting results
1649:63.79
1650:65.39
1651:67.03
1652:72.05
1653:70.47
1654:76.36
1656:92.94
1657:93.36
1658:95.45
1659:96.64
1660:96.67
1661:97.47
1663:97.33
1664:97.88
1665:98.56
1666:99.21
1667:98.03
1668:99.46
1670:59.87
1671:60.82
1672:62.87
1673:68.46
1674:66.66
1675:71.64
1677:40.20
1678:48.86
1679:42.92
1680:65.92
1681:58.65
1682:70.62
1684:84.66
1685:87.01
1686:86.72
1687:93.04
1688:93.22
1689:96.48
1691:90.04
1692:92.33
1693:90.13
1694:97.23
1695:95.14
1696:98.01
1698:34.39
1699:46.96
1700:39.24
1701:60.24
1702:56.72
1703:65.72
1706:Ablation study on where to apply SSP on the SYSU-MM01 dataset
1730:72.02
1731:73.34
1732:75.68
1733:75.49
1734:75.25
1735:76.64
1738:67.56
1739:68.45
1740:69.89
1741:70.56
1742:70.62
1743:71.64
1746:62.77
1747:64.02
1748:64.44
1749:68.77
1750:66.73
1751:70.62
1754:57.56
1755:60.56
1756:61.96
1757:63.56
1758:64.56
1759:65.76
1772:3
1773:4
1799:65.39
1800:69.22
1801:69.03
1802:66.54
1804:60.82
1805:65.77
1806:65.87
1807:62.48
1809:74.86
1810:78.22
1811:78.92
1812:75.39
1814:77.96
1815:79.44
1816:79.24
1817:78.15
1823:70.65
1824:71.05
1825:70.44
1826:72.05
1828:67.12
1829:67.42
1830:67.74
1831:68.46
1833:75.66
1834:76.11
1835:76.19
1836:78.92
1838:78.24
1839:78.94
1840:78.42
1841:80.24
1884:Methods
1899:72.05
1900:73.45
1901:74.79
1902:76.36
1904:68.46
1905:68.85
1906:69.21
1907:71.64
1909:65.92
1910:66.83
1911:68.65
1912:70.62
1914:60.24
1915:61.24
1916:63.77
1917:65.76

codex
我定位到六篇的方法章节边界了。前三篇的方法里真正重要的是：事件 ReID 的频谱去噪加区域均衡、非均衡 VI-ReID 的特征级少数模态扩增、换衣视频 ReID 的 3D 到 2D 稠密步态蒸馏。现在我补方法原文细节，之后直接整理成逐篇方法论拆解。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | sed -n '306,1341p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
III. M ETHODOLOGY
This section presents the overall architecture of the proposed method, detailing the SD-3DGF learning pipeline, the
MRIE module for efficient gait-appearance interaction, and the
formulation of training losses.
A. Overview
Fig. 3(a) depicts the overall architecture of our method and
illustrates the relationships among its modules. It adopts a
dual-stream structure comprising an appearance stream SA , a
gait stream SG , and the MRIE module. SA and SG operate in
parallel, while the MRIE module receives features from both
streams and yields the final identity representation. Given a
video tracklet T = {It | t = 1, 2, . . . , T } of a pedestrian
with T sampled frames, the appearance stream SA seeks
to extract clothing-invariant appearance features from T . It
employs a ResNet-50 backbone [56], using the first three
residual blocks to extract frame-level appearance features
A = {At | t = 1, 2, . . . , T }. The resulting feature set A is then
concatenated along the temporal dimension to form the initial
video-level appearance feature XA ∈ RT ×h×w×c , which is
further regularized by an orthogonal loss Lorg to achieve
clothing desensitization. The 3D gait stream SG , on the other
hand, is designed to learn expressive and discriminative 3D

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

4

Local Similarity Learning (LSL)

Video Tracklet 𝒯

Appearance Stream 𝓢𝑨

1×1×1 Conv

Cubic Window
𝑿𝑨

c

Cos. Sim.

𝒄𝜑

𝓐

ResNet-50

c
𝑿𝒅 1×1×1 Conv

Neighbors

𝓕’
𝓛𝒐𝒓𝒈

𝑿𝑮

(b)
Dual-Granularity Representative Token Generation (DGRT)
MHSA

𝓛𝒊𝒅

...

𝑺𝒍𝒐𝒄
𝑨

...

𝑺𝒍𝒐𝒄
𝑮

Patch Tokens

... 𝑻𝒑𝒂𝒕
𝑵𝒑

...

𝑹𝒍𝒐𝒄
𝟏

... 𝑹𝒍𝒐𝒄
𝑵𝒑

...

c
𝑹𝒈𝒍𝒃

𝑹𝒈𝒍𝒃

(c)

GAJE

𝓛𝒕𝒓𝒊

𝒈𝒍𝒃

𝑹𝑮

Patch Tokens

𝒈𝒍𝒃
𝑹𝑨

𝓛𝒈𝒇

𝒈𝒍𝒃
𝑹𝑨

෩
𝑺𝒍𝒐𝒄
𝑨

𝒈𝒍𝒃

MHCA

𝑹𝑮

c
LSL

𝑺𝒍𝒐𝒄
𝑮
MHCA

𝑿𝑮

ResNet-50

’

Gait Stream 𝓢𝑮

෩
𝑺𝒍𝒐𝒄
𝑮

𝓕

𝒈𝒍𝒃

𝑹𝑮

෩
𝑺𝒍𝒐𝒄
𝑨

෩
𝑺𝒍𝒐𝒄
𝑮

𝒈𝒍𝒃
෩
𝑹𝑨

AW ഥ
AW
ഥ 𝒍𝒐𝒄
𝑹𝒍𝒐𝒄
𝑹
𝑨
𝑮

MHCA

𝑺𝒍𝒐𝒄
𝑨

DGRT

MHCA

Gait-Appearance Joint Encoding (GAJE)

...

3D Patches

𝒇𝒊𝒅

𝒑𝒂𝒕

MHCA MHCA

...
𝑿𝑮

...

Flatten

Partition

Correspondence
Embeddings

DGRT

𝑻𝟏

MHCA

3D Patches

𝒈𝒍𝒃

𝑹𝑨

MHCA

...
𝑿𝑨

MRIE

Flatten

...

Partition

𝑺𝒍𝒐𝒄

MHSA

...

UNet-Based
Extractor
𝒇𝜽

𝑺𝒍𝒐𝒄

AW
𝒇𝒊𝒅
𝒈𝒍𝒃
෩
𝑹𝑮

Query
Key/Value

c Concatenation
Point-wise Add

(a)

(d)

Fig. 3. Overview of the proposed SD-3DGF. (a) The network includes an appearance stream and a gait stream, where the gait stream enhances local patterns
via LSL. The two streams interact and fuse to form the final identity representation via the MRIE module. (b) Local Similarity Learning (LSL) captures local
patterns by measuring the similarity between point-wise features and their neighbors. (c) Generating dual-granularity representative tokens via hierarchical
multi-head self-attention (MHSA). (d) Gait-Appearance Joint Encoding (GAJE) performs efficient cross-modal interaction via local and global representative
tokens, and generates the final identity representation through adaptive weighting (AW).

gait representations. It first employs a lightweight U-Netbased architecture to extract frame-level distilled features Z =
{Zt | t = 1, 2, . . . , T }, which approximate the distribution of
3D vertex embeddings and serve to construct the structured
distilled 3D gait field F. Subsequently, F is transformed by
a separate ResNet-50 backbone for spatiotemporal alignment
with the video-level appearance features XA . A local similarity
learning (LSL) mechanism is further employed to capture local
spatiotemporal correlations in the transformed 3D gait field,
yielding the 3D gait features XG .
Acting as a bridge between the two streams, the ModalityRepresentative Interactive Encoding (MRIE) module takes
video-level appearance features XA and 3D gait features XG
as input, and employs the Dual-Granularity Representative
Token Generation (DGRT) strategy to produce a global representative token Rglb and a set of local representative tokens
S loc for each modality. The Gait-Appearance Joint Encoding
(GAJE) submodule then efficiently models interactions between the representative tokens of both modalities, producing
the unified identity representation fid , which serves as the final
output of our model.
The following subsections provide a detailed exposition of
the proposed 3D gait representation, the MRIE module, and
the formulation of the training losses.
B. Structured Distilled 3D Gait Fields
Unlike prior methods that directly learn 3D gait features in
the sparse 3D space, we instead model them as a distilled 3D
gait field, which provides a dense representation. We further
discuss how to train the 3D gait fields and extract 3D gait

features with enhanced local awareness via local similarity
learning. To enable fair comparison with existing works, we
adopt the canonical SMPL model [24] with 6,890 vertices and
13,776 faces as the standard mesh representation.
1) Pre-training 3D Vertex Embeddings: Given the canonical SMPL model, we model it as a graph G = (V, A),
where V = {vi | i = 1, 2, . . . , N } ∈ RN is the vertex
set, A = {aij | i, j = 1, 2, . . . , N, i ̸= j} ∈ RN ×N is the
adjacency matrix, and N = 6890 represents the total number
of vertices. As illustrated in Fig. 2, graph residual convolutions are adopted to pre-train vertex embeddings, as they are
naturally well-suited for mesh models with explicit topology.
The detailed formulation of graph residual convolutions is as
follows:
X
(l+1)
(l)
(l)
(l)
(l)
xi
= σ(
w(xi , xj ) · xj ) + xi ,
(1)
j∈N (i)
(l)

(l+1)

where xi and xi
denote the features of vertex vi at
(l)
layers l and l+1, respectively, xj is the feature of a neighboring vertex vj at layer l, N (i) denotes the neighborhood
of vertex vi , w(·, ·) represents the weight between two graph
nodes, and σ(·) is the ReLU activation function [57].
The graph convolutional network (GCN) uses canonical
vertex coordinates as initial node features, consisting of four
graph residual layers, with graph normalization [58] applied
to all but the final layer to accelerate convergence. The node
features from the final layer are used as vertex embeddings.
We expect the embeddings of any two vertices to reflect
their geodesic distance on the mesh. To this end, we first
approximate geodesic distances by computing the shortest

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

spatiotemporal video domain. Formally, the structured distilled
3D gait field F is defined as:

Randomly
Sampled

3D Mesh
Recovery

Attach 3D
Vertex
Embeddings

5

𝒔

Project

F[t, x, y] = fθ (It )[x, y] ∈ RD ,
𝒗𝟏
𝒔

𝒗𝟐

𝒗𝟑

Fig. 4. Pipeline for generating dense 2D-3D paired data for a single frame.
The vertices of the recovered 3D human mesh, along with randomly sampled
surface points, are projected onto the image plane using camera parameters.
The vertex embeddings and interpolated surface point embeddings are used
as ground-truth for the corresponding pixel embeddings.

paths along the mesh surface between vertices using Dijkstra’s
algorithm, and compute the tangent vectors for each vertex
using the Trimesh 3D library [59]. We then normalize the
calculated geodesic distances to the range of (0, 1] via minmax scaling, and enforce the L2 distances between vertex
embeddings to match the corresponding geodesic distances:
d(evi , evj )
√
− g(vi , vj ) = 0,
(2)
D
where v and ev represent the vertex and its embedding
√
vector, D is the dimension of vertex embeddings, D is
a normalization factor to prevent distance imbalance from
dimensionality changes, d(·, ·) indicates the L2 distance, and
g(·, ·) defines the geodesic distance.
To encode the directional information of the structured 3D
mesh into the pre-trained vertex embeddings, we impose a
constraint ensuring that the angle between any two vertex embeddings matches the angle between their respective tangent
vectors on the mesh surface:
θemb (evi , evj ) − θtan (vi , vj ) = 0,

(3)

where θemb and θtan refer to the angles between the embedding vectors and the tangent vectors, respectively.
The training losses for the above two constraints will be
discussed in Section III-D. With these constraints, the pretrained vertex embeddings effectively preserve key inter-vetex
correlations and prior structural knowledge.
2) Formulating the Structured Distilled 3D Gait Fields:
The pre-trained 3D vertex embeddings E essentially define a
neural field:
fψ : R3 → RD ,
(4)
where ψ denotes the parameters of GCN, and D indicates the
embedding dimension.
A straightforward idea is to concatenate the embeddings
produced by fψ with the 3D vertex coordinates of the sequential 3D meshes, and then extract spatiotemporal point
cloud features as 3D gait features. However, the resulting
computational cost and inference time would be prohibitive.
Alternatively, we propose to build a structured distilled 3D
gait field (SD-3DGF) F that models 3D gait dynamics solely
from the frame-level features extracted from the video tracklet
T . Specifically, our goal is to transfer knowledge from 3D
vertex embeddings, which encode mesh-structured knowledge,
to frame-level features to learn 3D gait representations in the

(5)

where (x, y) refers to the pixel coordinates, t represents the
timestamp, fθ is a frame-level feature extractor, It ∈ RH×W ×3
denotes the t-th frame with height H and width W , and D
indicates the embedding dimension of 3D vertices.
The role of fθ is to extract pixel embeddings Zi = fθ (Ii ) ∈
RH×W ×D from frame Ii , which are distilled from the pretrained vertex embeddings E.
3) Constructing Dense Training Data and Spatial/Temporal
Constraints for SD-3DGF: We adopt a lightweight UNetbased architecture Eff-UNet [60] as the backbone of fθ . As
shown in Fig. 4, to prepare the data required for training
fθ , we begin by employing the body motion capture method
FrankMocap [61] to recover SMPL mesh sequences from the
input videos. The pre-trained vertex embeddings are assigned
to the mesh vertices and subsequently projected onto each
video frame using the estimated camera parameters. However,
this can only generate sparse paired 2D-3D training data, covering 15%-25% of the foreground pixels. Therefore, beyond
using mesh vertices alone, we randomly sample additional
points on each triangular face of the structured SMPL mesh,
and compute their embeddings via barycentric interpolation of
the triangle’s vertex embeddings. Formally, this process can be
expressed as:
es = b1 ev1 + b2 ev2 + b3 ev3 ,

b1 + b2 + b3 = 1,

(6)

where v1 , v2 , v3 denote the three vertices of the triangular face, and s is a sampled surface point on the face.
ev1 , ev2 , ev3 are the corresponding vertex embeddings, and
es is the embedding of s. (b1 , b2 , b3 ) are the barycentric
coordinates of s with respect to the triangle (v1 , v2 , v3 ), with
b1 = A(v2 , v3 , s)/A(v1 , v2 , v3 ), where A(·, ·, ·) denotes the
area of a triangle, and b2 and b3 are defined analogously.
These randomly sampled surface points are projected onto
the image plane following the same projection process as the
mesh vertices, thereby constructing dense 2D-3D paired data.
However, multiple mesh vertices or surface points may be
projected to the same pixel. To handle this, we adopt a zbuffer strategy that keeps only the vertex or surface point with
the smallest depth for each pixel, ensuring that the pixel is
assigned to the 3D point closest to the camera. Beyond the
dense 2D-3D paired data, we can further obtain pixel-pixel
correspondences (pi1 ,t1 , pi2 ,t2 ) by tracing the same 3D vertex
v across frames. Here, pi,t = (xi , yi , t) denotes the i-th pixel
in frame It with coordinates (xi , yi ). The pixels pi1 ,t1 and
pi2 ,t2 from frames It1 and It2 , respectively, correspond to the
same 3D vertex v.
Next, we impose three spatial or temporal constraints to
supervise the training of fθ for SD-3DGF:
First, for any frame t, the embedding of a foreground pixel
pi,t = (xi , yi , t) should align with that of its associated 3D
point qi (either a vertex v or a sampled surface point s):
Zt [xi , yi ] − eqi = ⃗0,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

(7)

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

6

where Zt = fθ (It ) ∈ RH×W ×D denotes the embedding map
of frame It produced by fθ , and eqi is the embedding vector
of the 3D point qi .
Second, for any frame t, the L2 distance between the
embeddings of two foreground pixels pi,t and pj,t (i ̸= j)
should match the L2 distance between the embeddings of their
corresponding 3D points qi and qj (either a vertex v or a
sampled surface point s):

neighborhood of each center point, resulting in 3×3×3−1 =
26 neighbors (see Fig. 3 (b)). This yields a channel dimension
of K = 26 for Xd . Additionally, cosine similarity is adopted as
the similarity function ϕ(·, ·) due to its parameter-free nature
and numerical stability.
To fuse the local similarity descriptor Xd with the gait
feature tensor F ′ , we first use a 1 × 1 × 1 3D convolution
to increase the channel dimension of Xd :

d(Zt [xi , yi ], Zt [xj , yj ]) − d(eqi , eqj ) = 0,

Xd′ = Conv3D1×1×1 [Xd ] ∈ RT ×h×w×c ,

(8)

where d(·, ·) denotes the L2 distance, and eqi , eqj are the
embeddings of the 3D points qi , qj , correspondingly.
Third, for any two distinct frames It1 and It2 (t1 ̸= t2 ), if a
foreground pixel pi,t1 from It1 and another pj,t2 from It2 map
to the same 3D vertex v (v is not a sampled surface point),
their embeddings should remain consistent:
Zt1 [xi , yi ] − Zt2 [xj , yj ] = ⃗0.

(9)

Focusing only on foreground pixels, we adopt an off-theshelf instance segmentation model, SOLOv2 [62], to generate
pedestrian masks. These masks remove background pixel embeddings from the distilled features Z = {Zt | t = 1, . . . , T },
where background pixels are set to zero and excluded from
loss computation. The structured distilled 3D gait field (SD3DGF) F ∈ RT ×H×W ×D is obtained by concatenating
Z along the temporal dimension. Consequently, F is fully
aligned with the input video in both spatial and temporal
dimensions, thus enabling strong compatibility between the
derived 3D gait features and video-level appearance features.
4) Modeling 3D Gait Features via Local Similarity: Fundamentally, the 3D gait field F consists of distilled frame-level
features. To learn discriminative representations, it is essential
to further model the underlying spatiotemporal correlations
within F. As the training of F enforces that similarities
between pixel embeddings reflect the geodesic distances of
their corresponding points on the 3D mesh, local similarity
learning (LSL) naturally serves as an effective choice for
capturing local spatiotemporal relations.
Considering the 3D gait field F ∈ RT ×H×W ×D as a
spatiotemporal feature tensor, we first set the input channel
of a ResNet-50 backbone to D and use its first three residual
blocks to transform the spatial resolution, producing a gait
feature tensor F ′ ∈ RT ×h×w×c with the same dimensionality
as the appearance features XA . Then, we define a local
similarity descriptor Xd ∈ RT ×h×w×K , where each element
Xd [t, x, y, k] measures the similarity between the central feature F ′ [t, x, y] and its k-th neighboring point. Here, K denotes
the total number of neighbors considered in the local similarity
computation. Specifically, the descriptor is formulated as:
Xd [t, x, y, k] = ϕ(F ′ [t, x, y], F ′ [t + δtk , x + δxk , y + δyk ]),
(10)
where ϕ(·, ·) denotes a similarity function, and (δtk , δxk , δyk )
represents the relative offset of the k-th neighboring point with
respect to the center point in the spatiotemporal feature space.
As a pairwise feature, the local similarity descriptor Xd
crucially depends on how the neighborhood is defined. We
use a cubic window with an edge length of 3 to determine the

′

(11)

where Xd′ denotes the local similarity descriptor after channel
expansion, c′ is set to 256, and Conv3D1×1×1 represents 1 ×
1 × 1 3D convolution operation.
Then, we concatenate Xd′ and F ′ along the channel dimension, and apply another 1 × 1 × 1 3D convolution for effective
feature integration, while ensuring that the generated 3D gait
features XG have the same channel dimension as F ′ :
XG = Conv3D1×1×1 [Xd′ , F ′ ] ∈ RT ×h×w×c ,

(12)

where [·, ·] indicates the concatenation of feature maps along
the channel dimension.
C. Modality Representative Interactive Encoding
For the appearance features XA and 3D gait features
XG , we seek to employ multi-head self-attention (MHSA) to
enhance intra-modal features, and multi-head cross-attention
(MHCA) to share information across modalities. However, full
space-time attention over all tokens is computationally expensive. Inspired by recent advances [63], [64] that hierarchically
reduce token size to accelerate self-attention, we propose
learning dual-granularity representative tokens with compact
spatial dimensions for both appearance and gait modalities,
enabling more efficient computation of MHSA and MHCA.
1) Generating Global and Local Representative Tokens:
Given the video-level appearance features XA ∈ RT ×h×w×c
and 3D gait features XG ∈ RT ×h×w×c , we first divide the
feature maps of each modality into a set of equally sized,
non-overlapping 3D patches. With a patch size of S = t′ ×
h′ × w′ , this partitioning produces Np = Tt′ × hh′ × ww′ =
T ×h×w
3D patches. Each 3D patch is then flattened along
S
the spatiotemporal dimensions and treated as a patch token,
denoted as Tipat ∈ RS×C , where i ∈ {1, 2, . . . , Np }.
To capture information at different levels of granularity and
achieve faster computation, we introduce a Dual-Granularity
Representative Token Generation (DGRT) strategy, as illustrated in Fig. 3(c). Specifically, a learnable local representative token Riloc ∈ R1×C is assigned to each patch token
Tipat . We then apply multi-head self-attention (MHSA) on the
concatenation of Tipat and Riloc , which aggregates key local
information into each local representative token Riloc :
T̂ipat = Concat(Tipat , Riloc ),

(13)

T̂ipat = MHSA(LN (T̂ipat )) + T̂ipat ,

(14)

where T̂ipat ∈ R(S+1)×C , Concat(·) represents the concatenation operation, MHSA(·) denotes the multi-head selfattention, and LN (·) denotes the layer normalization.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

Subsequently, all updated local representative tokens Riloc
are gathered to form the local representative token set S loc for
the current modality:
loc
S loc = {R1loc , R2loc , . . . , RN
}.
p

(15)

Furthermore, we concatenate all local representative tokens
in S loc and attach a learnable global representative token Rglb .
A second multi-head self-attention operation is then performed
to aggregate information from all local representative tokens
into the global one:
loc
R̂ = Concat(R1loc , R2loc , . . . , RN
, Rglb ),
p

(16)

R̂ = MHSA(LN (R̂)) + R̂,

(17)

where R̂ ∈ R(Np +1)×C , and Np denotes the number of local
representative tokens.
In this step, MHSA(·) is applied solely to summarize overall
modality information into Rglb , while the local representative
tokens in S loc are not updated with the latest results in R̂,
thereby preserving their critical local information.
Using the proposed dual-granularity token generation strategy, we obtain modality-specific global representative tokens
glb
glb
loc
loc
)
, SA
(RG
, RA
) and local representative token sets (SG
for both appearance and gait modalities.
2) Gait-Appearance Joint Encoding: As shown in Fig. 3(d),
we utilize reduced-size global and local representative tokens
glb
glb
loc
loc
)) for efficient joint encoding
, SA
(i.e., (RG
, RA
) and (SG
of gait and appearance modalities. Specifically, the multihead cross-attention (MHCA) mechanism is introduced to
perform cross-modal interaction between the dual-granularity
representative tokens. First, the local representative tokens of
each modality are updated via local-local and local-global
cross-modal interactions, as formulated below:
glb
loc
loc
loc
loc
loc
S̃A
= MHCA(LN (SA
), LN (SG
)) + MHCA(LN (SA
), LN (RG
)) + SA
,

(18)
glb
loc
loc
loc
loc
loc
), LN (RA
)) + SG
,
)) + MHCA(LN (SG
= MHCA(LN (SG
), LN (SA
S̃G

(19)
loc
loc
denote the updated local representative toand S̃A
where S̃G
ken sets of the 3D gait and appearance modalities, respectively,
and MHCA(·, ·) denotes multi-head cross-attention, with the
first argument specifying the source of queries and the second
specifying the source of keys and values.
Next, the global representative tokens are updated via
global-global and global-local cross-modal interactions, formulated as:
glb
glb
glb
glb
glb
loc
R̃A
= MHCA(LN (RA
), LN (RG
)) + MHCA(LN (RA
), LN (S̃G
)) + RA
,

(20)
glb
glb
glb
glb
glb
loc
R̃G
= MHCA(LN (RG
), LN (RA
)) + MHCA(LN (RG
), LN (S̃A
)) + RG
,

(21)
glb
glb
where R̃G
and R̃A
denote the updated global representative
tokens of the 3D gait and appearance modalities, respectively.
glb
glb
loc
loc
Finally, S̃G
, S̃A
, R̃G
, and R̃A
are adaptively fused
to form the identity representation. For each modality, the
local representative tokens are first adaptively aggregated via
attention:
loc
R̄m
=

Np
X
i=1

loc
αim R̃m,i
,

m ∈ {A, G},

(22)

7

loc
where m denotes the modality, R̄m
∈ R1×C is the aggregated
loc
local representation, R̃m,i is the i-th local representative token
loc
loc
in S̃m
, and αim is the learnable attention weight for R̃m,i
.
The final identity representation fid is then obtained by
glb
glb
loc
loc
adaptively weighting R̄A
, R̄G
, R̃A
and R̃G
:
glb
glb
loc
loc
fid = γ1 R̄A
+ γ2 R̄G
+ γ3 R̃A
+ γ4 R̃G
,

(23)

where γ1 , γ2 , γ3 , γ4 are learnable attention weights.

D. Loss Function
We present a detailed description of all loss functions
employed in our framework, including the vertex embedding
pre-training loss, the 3D gait field loss, the Re-ID loss, and
the overall training objective.
1) Vertex Embedding Pre-training Loss: As discussed in
Section III-B1, we enforce that the distances between the
vertex embeddings produced by the GCN match the corresponding geodesic distances on the mesh surface. The geodesic
loss Lgeo is defined as:
Lgeo =

1 X d(evi , evj )
( √
− g(vi , vj ))2 ,
|V|2
D

(24)

i,j∈V

where V denotes the set of vertices, evi and evj are the
embeddings of vertices vi and vj , d(·, ·) computes the L2
distance between two vertex embeddings,
D is the dimension
√
of the vertex embeddings, D serves as a normalization
factor, and g(·, ·) denotes the geodesic distance between two
vertices.
In addition, to preserve the directional information of the 3D
mesh, we constrain the angle between two vertex embeddings
to match the angle between their corresponding tangent vectors
on the mesh surface. The directional loss Ldir is defined as:
Ldir =

1 X
(θemb (evi , evj ) − θtan (vi , vj ))2 ,
|V|2

(25)

i,j∈V

where θemb and θtan denote the angles formed by the embedding vectors and the tangent vectors, respectively.
Thus, the vertex embedding pre-training loss Lvep for
training the GCN is given by:
Lvep = Lgeo + λLdir ,

(26)

where λ is a balancing hyperparameter that controls the
relative contribution of the directional loss.
2) 3D Gait Field Loss: According to Section III-B3, the
training of the 3D gait field F is guided by three consistency constraints. First, for each frame, the embedding of a
foreground pixel should align with that of its corresponding
3D point (either a vertex or a sampled surface point). Second,
within the same frame, the L2 distance between embeddings of
two foreground pixels should match that of their corresponding
3D points. Third, for foreground pixels from different frames
that correspond to the same 3D vertex, their embeddings

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

should remain consistent. The 3D gait field loss Lgf satisfying
the above three constraints is defined as follows:
Lgf =

N
T
1 X 1X 1 X
(
∥Zt [xi , yi ] − eqi ∥22
N n=1 T t=1 |St |

8

objective. The model is optimized using the Re-ID loss Lreid
and the 3D gait field loss Lgf , leading to the overall objective:
L = Lreid + η Lgf ,

pi,t ∈It

(31)

where the hyperparameter η balances the two loss terms.

T

X
1X 1
(d(Zt [xi , yi ], Zt [xj , yj ]) − d(eqi , eqj ))2
T t=1 |Pt |
pi,t ,pj,t ∈It
X
1
+ µ2
∥Zt1 [xi , yi ] − Zt2 [xj , yj ]∥22 ).
|Q|

+ µ1

pi,t1 ∈It1 ,pj,t2 ∈It2

(27)
Here, N represents the batch size. T denotes the number
of frames. St denotes the set of foreground pixels in frame t
that have corresponding 3D points. Pt denotes the set of foreground pixel pairs in frame t along with their corresponding
3D points. Q denotes the set of cross-frame pixel pairs that
correspond to the same 3D vertex. pi,t = (xi , yi , t) specifies
the coordinate (xi , yi ) of the i-th pixel in frame It . Zt is the
embedding map of frame It , and Zt [xi , yi ] is the embedding
vector of pi,t . qi and qj denote the 3D points corresponding
to pi,t and pj,t , respectively, with embeddings eqi and eqj .
The function d(·, ·) denotes the L2 distance between two
embedding vectors, and µ1 , µ2 are weighting parameters. The
three loss terms in Lgf correspond one-to-one to the three
consistency constraints described above.
3) Re-ID Loss: For the video-based appearance feature
XA ∈ RT ×h×w×c , which is obtained by aggregating framelevel features, we seek to reduce its sensitivity to clothing
via an orthogonality constraint. We first apply global average
pooling (GAP) on XA to obtain fA ∈ R1×c . Then, a clothing
classifier cφ , composed of a linear layer followed by a softmax
activation, uses fA to predict the clothing label for the video
sequence. The clothing classification loss Lc is defined as:
1 X
yn log(ŷn ),
N n=1

(28)

where N is the batch size; yn and ŷn denote the ground-truth
and predicted clothing labels for sample n, respectively.
During end-to-end training, the pre-trained clothing classifier cφ is kept frozen. Since its weight matrix Wc defines the
clothing directions, an orthogonal loss Lorg is applied to fA
to suppress its projections along these directions:
Lorg = −

N
X

1
∥fA Wc ∥2F ,
N n=1

(29)

where ∥ · ∥F denotes the Frobenius norm.
In addition, the final identity representation fid is supervised
using the commonly adopted ID loss Lid and triplet loss
Ltri [65] in the Re-ID literature. Accordingly, the Re-ID loss
Lreid is formulated as a weighted combination of the ID loss,
triplet loss, and the orthogonal loss:
Lreid = Lid + w1 Ltri + w2 Lorg ,

A. Datasets and Evaluation Protocols
We evaluate our method on four VCCRe-ID datasets, i.e.,
VCCR [47], CCVID [45], CCV-S [46], and CCV-R [46].
Among them, VCCR, CCVID, and CCV-R are real-world
datasets, and CCV-S is a synthetic dataset. All datasets are
evaluated under two test settings: the standard setting (SD)
and the clothes-changing setting (CC). Under the standard
setting (SD), the retrieval process includes all gallery samples,
covering both clothes-consistent and clothes-changing cases,
whereas under the clothes-changing setting (CC), gallery samples that share both the same clothing and identity labels with
the query are excluded. To further assess the generalization
ability of the proposed method, we conduct additional evaluations on three widely used video-based person Re-ID benchmarks, i.e., MARS [26], LS-VID [66], and iLIDS-VID [67],
which primarily consist of clothes-consistent scenarios. Following standard practice, we evaluate the overall performance
of the Re-ID model using the Rank-1 accuracy of the CMC
curve and mean average precision (mAP). In addition, we
introduce the Embedding Consistency Score (ECS) as a new
metric, which measures the quality of the extracted dense
correspondence embeddings, and is computed on the test set
of each dataset involved in the experiment. ECS is defined as:
ECS =

N

Lc = −

IV. E XPERIMENTS

(30)

where w1 and w2 are weighting coefficients.
4) Overall Training Objective: In the end-to-end training
of the full model, the 3D vertex embeddings and clothing
classifier are kept fixed, so the vertex embedding loss Lvep
and clothing classification loss Lc are excluded from the total

Nd
T
1
1 X 1 X 1 X
(
),
Nd n=1 T t=1 |St | p ∈I 1 + ∥Zt [xi , yi ] − evi ∥1
i,t

t

(32)

where Nd denotes the total number of video sequences, ∥ · ∥1
represents the L1 norm, and all other symbols are defined as
in the first term of Eq. 27. Note that in the test set, no surface
points are sampled, and only pixel-vertex pairs are used to
compute ECS.
B. Implementation Details

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Spectrum-guided feature enhancement network for event person re-identification.pdf' - | sed -n '245,504p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3.2. MSAM
As described in Intro. 1, the imaging principle of event cameras
results in a signiﬁcant amount of noise in the event streams. Although
[5,6] also adopt event-based models for Re-ID tasks, they neglect to
address the critical challenges of event semantic denoising and enhancement, which are essential for improving feature discriminability
in sparse event data scenarios. Therefore, denoising becomes an important prerequisite for obtaining high-quality semantic representations of
events. Consequently, extensive research [7,42,44] has explored denoising methodologies to enhance event-based semantic representations.
Notably, frequency-domain approaches [8,43] ﬁlter high-frequency
noise while preserving low-frequency components, integrating
denoising into perception pipelines. Inspired by these advances, we
propose extending frequency-domain denoising to event-based Re-ID
(RE-ID) tasks to obtain high-ﬁdelity semantic embeddings. However, the
sparsity of event semantics causes a substantial portion of noise to also
appear as semantic isolated points. Simply removing high-frequency
information at a single scale under these conditions inevitably leads to
the loss of valuable semantic content. Multi-scale perception further
enhances semantic interactions and fusion across regions of varying
sizes, thereby stabilizing eﬀective semantics as much as possible.
Therefore, we employ convolutions with diﬀerent scales to perceive
event semantics and attempt to ﬁlter high-frequency information of
event semantics from the perspective of diﬀerent receptive ﬁelds.
Furthermore, low-frequency information from diﬀerent receptive ﬁelds

3. Method
Structure of SFE-Net is shown in Fig. 2. The proposed SFE-Net is designed to mitigate noise interference in event-based data while comprehensively extracting discriminative person semantics from sparse event
streams. It integrates two innovative mechanisms: (1) the Multi-grain
Spectrum Attention Mechanism (MSAM), which ﬁlters noise and enhances semantic representations of person identities, and (2) the Consecutive Patch Dropout Module (CPDM), which forces the model to focus
uniformly on all discriminative regions, thereby improving the robustness of person descriptors.
3

Pattern Recognition 172 (2026) 112705

H. Tan et al.

Fig. 2. The SFE-Net mainly contains four components: Backbone, Event-Stream Anonymization Module (EAM), the proposed Multi-grain Spectrum Attention Mechanism (MSAM) and the proposed Consecutive Patch Dropout Module (CPDM).

also complements each other, jointly yielding high-quality semantic
representations. However, spectral-domain denoising strategies are
prone to oversmoothing artifacts. Moreover, the inherent discreteness
of event semantics hinders the representation of high-quality semantic
information. To address this, after performing multi-scale denoising,
we attempt to employ an attention mechanism for global semantic interaction to strengthen contextual semantic representations. To achieve
the goal, we propose a Multi-grain Spectrum Attention Mechanism
(MSAM).
The structure of MSAM is illustrated in the top part of Fig. 2. MSAM
performs three main steps: (i) Pre-Spectrum Augmentation is designed
to ﬁlter out noise from event information, (ii) Spatial Semantics Interaction Mechanism to enhance the person semantics representation, and
(iii) Post-Spectrum Augmentation to facilitate global semantic representation and further reﬁne event person semantics.
Pre-Spectrum Augmentation (Pre-SA). In the Pre-SA, we ﬁrst
utilize three types of convolutions with diﬀerent kernel sizes, namely
1 × 1, 3 × 3, and 5 × 5, to extract event semantics at various scales.
The large-scale receptive ﬁeld is employed to stabilize the representation of event semantics, reducing the likelihood of valid
semantics being misclassiﬁed as noise. Meanwhile, the small-scale receptive ﬁeld helps to decrease the possibility of noise being mistaken for valid semantics. Through information ﬁltering and processing, the semantics at diﬀerent scales mutually complement and
reinforce each other. As shown in Fig. 2, we ﬁrst transform the
image features into frequency-domain information through Fourier

transform. Subsequently, we utilize a Gaussian ﬁlter to remove
high-frequency information while retaining low-frequency information.
Next, we decompose the real and imaginary parts of the frequencydomain information and further employ 1 × 1 convolutions to process
this frequency-domain information. After that, we merge the real and
imaginary parts and apply Gaussian ﬁltering again to stabilize lowfrequency semantics and ﬁlter out high-frequency noise. Finally, we
concatenate the results of the two ﬁltering stages and proceed with the
inverse Fourier transform. Let 𝐹 ∈ ℝ𝐶×𝐻×𝑊 denotes the input features,
the Pre-Spectrum Augmentation (Pre-SA) is computed as:
̂ = 𝐶𝑜𝑛𝑣1×1 (𝜎(𝐾, 𝛾) ⊙ 𝛩𝐹 𝐹 𝑇 (𝐶𝑜𝑛𝑣𝑖×𝑖 (𝐹 ))),
𝐹𝑖×𝑖
𝛾 = 𝐹 𝐶(𝐺𝐴𝑃 (𝐶𝑜𝑛𝑣𝑖×𝑖 (𝐹 )))
∑
̂ + 𝐹𝑖×𝑖
̂ )].
𝐹 𝑃 𝑟𝑒 =
[𝛩𝐼𝐹 𝐹 𝑇 (𝜎(𝐾, 𝛾) ⊙ 𝐹𝑖×𝑖

(1)

𝑖∈{1,3,5}

where ⊙ denotes low-pass ﬁltering with adaptive Gaussian smoothed
ﬁlters 𝜎(𝐾, ⋅), which has the same spatial size as 𝐹 , and 𝐾 is the bandwidth. Since the frequency-domain Gaussian ﬁlter is a linear smoothing ﬁlter based on the Fourier transform, it achieves a smoothing effect in the spatial domain by applying Gaussian weighted attenuation
to the frequency components of an image in the frequency domain.
Its core characteristics include the eﬃciency of frequency-domain multiplication, smooth transitions without ringing artifacts, and eﬀective
suppression of high-frequency noise. In light of this, we employ Gaussian smoothed ﬁlters 𝜎(𝐾, ⋅) to ﬁlter out high-frequency noise in the
frequency domain. To make 𝜎(𝐾, ⋅) input-aware, we create an initial 2D
4

Pattern Recognition 172 (2026) 112705

H. Tan et al.

Gaussian map based on 𝐾, and apply Global Average Pooling (GAP) and
Fully Connection Linear layers (FC) on feature map 𝐹 to predict a scale
parameter 𝛾 to update the Gaussian map, i.e. 𝛾 = 𝐹 𝐶(𝐺𝐴𝑃 (𝐶𝑜𝑛𝑣𝑖×𝑖 (𝐹 ))).
Besides, 𝑖 × 𝑖, 𝑖 = 1, 3, 5 is the size of the convolution kernel. The 𝛩𝐹 𝐹 𝑇
is the fourier transform and 𝛩𝐼𝐹 𝐹 𝑇 is the inverse fourier transform.
As shown in Fig. 2, in the fourier transform process, semantics are
decomposed into Real and Imaginary parts. Real part is utilized to regulate the amplitude of the sine wave, while the Imaginary part serves
to adjust its phase. This enables the transformed sine wave to be converted back into the time domain. Following the ﬁltering process, adaptive adjustments to the amplitude and phase values are made through
convolution, ensuring that the adjusted semantics are more conducive to
person matching from a global perspective. To enhance noise ﬁltering
and stabilize low-frequency semantic information, we integrate residual connections with dual Gaussian ﬁltering. Besides, we employ joint
learning of the scaling factors 𝛾 for the two ﬁltering stages based on the
input features 𝐶𝑜𝑛𝑣𝑖×𝑖 (𝐹 ), enabling adaptive adjustment of the retention
ratio of low-frequency information.
Semantics Interaction Mechanism (SIM). After Pre-SA, although
ﬁltering out high-frequency components using frequency-domain strategies can eﬀectively remove noise, it is prone to causing blurred edges
and the loss of texture details. Moreover, the sparsity of event semantics results in weak interactions among semantics and insuﬃcient contextual semantic information. The contextual semantic is detrimental
to the model’s ability to perceive and discriminate semantics. In response, we attempt to employ an attention strategy to achieve interaction and fusion among event semantics, thereby enhancing the representation of event contextual semantics. We ﬁrst compute aﬃnities
coeﬃcients 𝑅 ∈ ℝ𝑁×𝑁 (𝑁 = 𝐻 × 𝑊 ) between diﬀerent event semantics
𝑓 𝑃 𝑟𝑒 ∈ ℝ𝐶 , 𝐹 𝑃 𝑟𝑒 = {𝑓𝑖𝑃 𝑟𝑒 ∈ ℝ𝐶 |𝑖 = 1, 2, ⋯ , 𝐻 × 𝑊 }. Then, according to
aﬃnity weight 𝑅 between event semantics 𝑓 𝑃 𝑟𝑒 , we inject global structure semantics into each event semantic vector, to achieve semantic interaction. Such Semantics Interaction Mechanism (SIM) is calculated as:

3.3. CPDM
Although the objective of MSAM is denoising and feature enhancement, deep models tend to concentrate solely on speciﬁc local key
semantics and overlook the overall eﬀective semantics. This is often
attributed to the convolutional model being “lazy”. To tackle this challenge, the PCB strategy [9] was introduced to guide deep models in extracting discriminative semantics from small image patches. However,
due to the sparse and fragmented nature of event semantics, relying
solely on single-patch semantics for accurate person identiﬁcation remains inadequate. This sparsity further manifests as a critically low
ratio of informative semantics in feature maps, causing Dropout-based
methods [11–13] to inadvertently focus on non-discriminative regions,
thereby diminishing their robustness. To address this limitation, we propose an inverse PCB strategy that systematically erases features to generate multiple incomplete representations. By doing so, we expect that
the deep model can equally perceive each eﬀective region of the feature
map and capture robust person descriptors. So, we propose a Consecutive Patch Dropout Module (CPDM).
In CPDM, (a) we divide the feature map 𝑇 (𝑥) ∈ ℝ24×8×2048 into 𝑚
equal part by row, (b) we obtain 𝑚 incomplete feature maps by erasing
each part, (c) we drive a deep model to capture discriminative semantics
from each incomplete feature map. Based on the feature tensor 𝑇 (𝑥), we
describe in detail the build details of the CPDM.
(i:) As shown in bottom part of Fig. 2, feature map 𝑇 (𝑥) is divided
to 𝑚 uniform patches. (ii:) The DropPatch-i, i = 1, 2, ⋯ , 𝑚 is designed to
drop the 𝑖 − 𝑡ℎ patch on the tensor 𝑇 (𝑥). As shown in bottom right part of
Fig. 2, the feature 𝑇 (𝑥) is divided to 6 uniform patches; the DropPatch-2
is used to drop the 2 − 𝑡ℎ patch on the tensor 𝑇 (𝑥). We can see that the
2-th patch on the 𝑇 (𝑥) is zeroed out. Since this, 𝑚 incomplete feature
tensors 𝑇 (𝑥)∗𝑖 ∈ ℝ24×8×2048 , i = 1, 2, ⋯ , 𝑚. (iii:) These 𝑚 incomplete feature tensors 𝑇 (𝑥)∗𝑖 ∈ ℝ24×8×2048 , i = 1, 2, ⋯ , 𝑚 are fed into the ResBlock
and Gobal Average Pooling (GAP) operation. Since this, we can gain 𝑚
incomplete person descriptors, i.e. 𝑓𝑖 ∈ ℝ512 , i = 1, 2, ⋯ , 𝑚, which is fed
into the loss functions: the Cross-Entropy Loss (CE) and the Triplet (Tri.)
Loss. Here, the ResBlock is composed of three Bottleneck Blocks [45].
Based on the design mentioned above, when compared to the person
feature map 𝑇 (𝑥), the 𝑖 − 𝑡ℎ patch feature region on 𝑇 (𝑥)∗𝑖 is missing.
During the training stage, the person matching metric will encourage
the deep model to capture discriminative semantics from the remaining areas on 𝑇 (𝑥)∗𝑖 . Our CPDM can produce multiple incomplete feature
maps 𝑇 (𝑥)∗𝑖 , i = 1, 2, ⋯ , 𝑚. By doing so, the deep Re-ID model is driven
to equally perceive each eﬀective region of the feature map and capture
robust person descriptors.

(

𝐹

𝑆𝐼𝑀

)
(𝑊 𝑞 ⋅ 𝐹 𝑃 𝑟𝑒 )𝑇 (𝑊 𝑘 ⋅ 𝐹 𝑃 𝑟𝑒 )
= 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥
(𝑊 𝑣 ⋅ 𝐹 𝑃 𝑟𝑒 ).
√
𝐶

MSAMs, namely MSAM-1, MSAM-2, and MSAM-3, in the shallow, middle, and deep sections in the ResNet-50.

(2)

where 𝑊 𝑞 , 𝑊 𝑘 , and 𝑊 𝑣 are projection functions. The 𝑊 𝑞 , 𝑊 𝑘 , and
𝑊 𝑣 are implemented as a 1 × 1 convolution with 𝐶 number of output
channels.
Post-Spectrum Augmentation (Post-SA). We also employ spectrum
augmentation to further reﬁne the event semantics. Filtering out highfrequency information and retaining low-frequency information can
eﬀectively denoise. Additionally, low-frequency information contains
general structure of an image, which is crucial for enhancing the model’s
discriminative ability. Therefore, we attempt to further strengthen lowfrequency semantics to enhance the representation of the overall event
semantics. We deﬁne this process as the Post-Spectrum Augmentation
(Post-SA), which is calculated as follows:

3.4. Loss function in SFE-Net
In our SFE-Net, we mainly adopt the Cross-Entropy Loss and the
Triplet Loss to train the whole model. Therefore, the total loss function
in the SFE-Net is given as follows:

𝐹̂ 𝑃 𝑜𝑠𝑡 = 𝐶𝑜𝑛𝑣1×1 (𝜎(𝐾, 𝜂) ⊙ 𝛩𝐹 𝐹 𝑇 (𝐹 𝑆𝐼𝑀 )), 𝜂 = 𝐹 𝐶(𝐺𝐴𝑃 (𝐹 𝑆𝐼𝑀 ))
𝐹 𝑃 𝑜𝑠𝑡 = 𝛩𝐼𝐹 𝐹 𝑇 (𝜎(𝐾, 𝜂) ⊙ 𝐹̂ 𝑃 𝑜𝑠𝑡 + 𝐹̂ 𝑃 𝑜𝑠𝑡 ) + 𝐹 𝑆𝐼𝑀 .

𝑓

𝑓

𝑝
𝐿𝐺 = 𝐿𝑇𝑝𝑟𝑖. + 𝐿𝐶𝐸
+𝜆

𝑚
∑
𝑖
[𝐿𝑓𝑇 𝑖𝑟𝑖. + 𝐿𝑓𝐶𝐸
] + 𝐿𝑆𝑡𝑟𝑢𝑐𝑡 + 𝐿𝑅𝑒𝑐 ,

(4)

𝑖=1

(3)
𝑓

𝑓

𝑝
Here Triplet Loss ( 𝐿𝑇𝑝𝑟𝑖. ) and Cross-Entropy Loss (𝐿𝐶𝐸
) are applied

Here, to make 𝜎(𝐾, ⋅) input-aware, we also apply pooling and linear
layers on feature map 𝐹 𝑆𝐼𝑀 to predict a scale parameter 𝜂 to update
the Gaussian map. Unlike Pre-SA, we merely utilize Post-SA to enhance
low-frequency linguistic representations. Therefore, the Post-SA model
adopts residual connections, which involves reintroducing 𝐹 𝑆𝐼𝑀 while
calculating 𝐹 𝑃 𝑜𝑠𝑡−𝑆𝐴 . Such residual operation further stabilizes and enhances the global semantics of the event person semantics. Since this,
the proposed MSAM can be utilized to establish both denoising and feature enhancement targets. As indicated in Fig. 2, we incorporate three

to the person descriptor 𝑓𝑝 . Similarly, the Triplet Loss (𝐿𝑓𝑇 𝑖𝑟𝑖. ) and

𝑖
Cross-Entropy Loss (𝐿𝑓𝐶𝐸
) are utilized for the 𝑖-th incomplete person
descriptor 𝑓𝑖 . In addition, 𝐿𝑆𝑡𝑟𝑢𝑐𝑡 + 𝐿𝑅𝑒𝑐 introduced by [5] is used to
modify the event stream to prevent the event information from being
successfully reconstructed into grayscale images. The parameter 𝜆 = 0.2
∑
𝑓𝑖
𝑓𝑖
is used to balance the loss term 𝑚
𝑖=1 [𝐿𝑇 𝑟𝑖. + 𝐿𝐶𝐸 ]. In the following experiments, we verify that when 𝑚 = 8, the SFE-Net achieves the best performance. Importantly, the CPDM is removed in the testing stage, and

5

Pattern Recognition 172 (2026) 112705

H. Tan et al.

we only adopted the person descriptor 𝑓𝑝 ∈ ℝ512 to conduct the person
matching task.

Therefore, when these models are applied to the event modality, RGBbased models ﬁnd it diﬃcult to perceive and capture eﬀective person
semantics amidst the numerous noises. Secondly, from the perspective
of semantic sparsity and density, although the representations of events
are similar to those of the RGB modality, the vast diﬀerence in semantic
richness between the two modalities makes it challenging for RGB models to perceive and capture suﬃcient contextual semantics from events.
Finally, currently, there are very few event-based REID models, and
most of them tend to focus on privacy protection research, neglecting
the exploration of semantics from events that are beneﬁcial for person
matching.
Besides, in Table 1, we also present the number of parameters for
diﬀerent methods. Our model has a relatively large number of parameters, which is primarily attributable to the convolution operation of Spectrum Augmentation and attention mechanism employed
in the MSAM. Secondly, the multi-drive models within the CPDM
uniformly perceive and capture eﬀective semantics. Furthermore, an
increase in the number of parameters is inevitable. This is because
ED-Net [6] and EA-Net [5] are based on a basic backbone network. To
enhance semantic representation and improve performance, it is necessary to design innovative mechanisms to guide the basic model in
perceiving and capturing high-quality semantic representations. Consequently, the number of parameters in the model will increase. Overall,
compared with the increase in the number of parameters, the performance of our SFE-Net has signiﬁcantly improved.

4. Experiment

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf' - | sed -n '166,505p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3. Method
3.1. Stochastic style perturbation
3.1.1. Background
The conceptually approach to addressing imbalanced data through
oversampling involves randomly duplicating existing instances until
2

Neural Networks 195 (2026) 108206

H. Liu et al.

Fig. 2. This is a high-level overview of our CMBL framework, which uses a three-stream network for multi-modal feature encoding. SSP features are produced
from the IR features following conv layer1 in the IR stream subnetwork. Parameters of conv layers 2-4 in the three-stream network are shared between RGB and IR
streams, while being independent from the oversampling stream. The cross-distribution alignment module aligns heterogeneous features using MMD distance at the
global distribution level. The Class-Aware Contrast Similarity Learning (CACS) module enhances feature discrimination through intra- and inter-class similarity at a
nuanced identity level. Best viewed in color.

balanced class distributions are achieved. However, this method often
results in the minority class distribution being overly concentrated in
the areas where the original observations were located. Consequently,
models trained on data altered in this manner are at a higher risk of
overﬁtting. To mitigate this issue, data-level neighborhood-based methods (Chawla et al., 2002; Cui et al., 2019; Huang et al., 2016; Wang
et al., 2017) have been developed, among which the Synthetic Minority
Over-sampling Technique (SMOTE) (Chawla et al., 2002) is one of the
most prominent. The basic idea of SMOTE (Chawla et al., 2002) is to
identify the nearest neighbors within the same class for a given minority instance and then synthesizes new samples in the space between the
target instance and one of its neighbors. Speciﬁcally, suppose we have
a minority class sample (𝑥) and its (𝑘) nearest neighbors (𝑥1 , 𝑥2 , … , 𝑥𝑘 ).
The formula for generating a new synthetic sample by SMOTE (Chawla
et al., 2002) is as follows:
𝑥𝑛𝑒𝑤 = (𝑥 + 𝑟𝑎𝑛𝑑(0, 1) × (𝑥𝑖 − 𝑥)),

Fig. 3. Visualization in multi-modal scenarios for SMOTE (Chawla et al., 2002):
pre-synthesis (left) vs. post-synthesis (right). Due to the large modality discrepancy, the generated new samples always deviate from the original modality distribution (green dots). Best view in color.

age by mean and standard deviation 𝜇(𝑓 𝑡 ) ∈ ℝ𝐵×𝐶 , 𝜎(𝑓 𝑡 ) ∈ ℝ𝐵×𝐶 , as:

(1)

1 ∑∑ 𝑡
𝑓
,
𝐻𝑊 ℎ=1 𝑤=1 𝑏,𝑐,ℎ,𝑤
√
√
)2
√ 1 𝐻
∑ 𝑊
∑(
𝑡
𝑡
𝜎(𝑓 ) = √
𝑓𝑏,𝑐,ℎ,𝑤
− 𝜇(𝑓 𝑡 ) .
𝐻𝑊 ℎ=1 𝑤=1
𝐻

𝑊

𝜇(𝑓 𝑡 ) =

where 𝑟𝑎𝑛𝑑(0, 1) ranges between [0, 1] and is used to control the position
of the synthetic sample. Repeating the steps allows for the creation of
synthetic samples in quantities tailored to achieve dataset balance.
The foundational premise of SMOTE (Chawla et al., 2002) is that the
spaces between the nearest minority neighbors are viable for generating new instances. However, this assumption encounters challenges in
multi-modal scene. As illustrated in Fig. 3, due to large modality discrepancy, the nearest minority neighborhood is spread apart, leading to
the generation of synthetic samples that deviate from the original distribution. Moreover, given the varying degrees of modality discrepancy, it
is unclear what the optimal size of the neighborhood 𝑘 should be.

(2)

(3)

Here, 𝜇(𝑓 𝑡 ) and 𝜎(𝑓 𝑡 ) encapsulate the statistical characteristics of the
image, capturing its inherent style.
The normalized features that encode the semantic information are
represented as:
𝑓 =𝛾⊙

𝑓 𝑡 − 𝜇(𝑓 𝑡 )
+ 𝛽,
𝜎(𝑓 𝑡 )

(4)

where 𝛾 and 𝛽 represent the channel-wise aﬃne parameters. According
to the work Huang and Belongie (2017), Tang et al. (2021), modifying
the feature statistics (𝛾, 𝛽) can change the feature style of an image.
Then, to obtain reasonable values of 𝛾 and 𝛽, we ﬁrstly utilize the variance of 𝜇(𝑓 𝑡 ) and 𝜎(𝑓 𝑡 ) to estimate meaningful style vectors, denoted
as:

3.1.2. Feature-level stochastic style perturbation
Rather than depending on nearest neighbors for generating new samples at the data level, in this paper we explores the potential of a featurebased approach. Our objective diverges from SMOTE techniques: we
aim to directly generate IR features within the deep feature space. By
undertaking this approach, we aspire to transform the style of minority
modality samples while preserving their semantic information, ensuring
that the generated features do not stray from the original distribution.
Speciﬁcally, as shown in Fig. 4, given infrared features 𝑓 𝑡 ∈
ℝ𝐵×𝐶×𝐻×𝑊 where 𝐵, 𝐶, 𝐻, 𝑊 denoting the batchsize, dimension of
channel, height and width, respectively, we represents the style of im-

𝜎(𝜇(𝑓 𝑡 ))2 =

𝐵
𝐵
1 ∑
1 ∑
(𝜇(𝑓 𝑡 ) −
𝜇(𝑓 𝑡 ))2 ,
𝐵 𝑏=1
𝐵 𝑏=1

𝐵
𝐵
1 ∑
1 ∑
𝜎(𝜎(𝑓 )) =
(𝜎(𝑓 𝑡 ) −
𝜎(𝑓 𝑡 ))2 ,
𝐵 𝑏=1
𝐵 𝑏=1
𝑡 2

3

(5)

Neural Networks 195 (2026) 108206

H. Liu et al.

a RGB feature 𝑓𝑖𝑣 , IR feature 𝑓𝑖𝑡 and perturbed feature 𝑓𝑖𝑝 generated by
SSP, we ﬁrst calculate MMD (Gretton et al., 2012) to measure modality
discrepancy between visible and infrared domains:
𝐷 (𝑣, 𝑡) = ‖𝐄𝑣 [𝜙(𝕌𝑁
(𝑓 𝑣 ))] − 𝐄𝑡 [𝜙(𝕌𝑁
(𝑓 𝑡 ))]‖2 ,
𝑖=1 𝑖
𝑖=1 𝑖

Fig. 4. Illustration of the proposed stochastic style perturbation (SSP) layer. We
approximate the distribution of styles via disturbing the variance of the feature
statistics and randomly sample the scale parameter 𝛾𝑠𝑠𝑝 and shift parameter 𝛽𝑠𝑠𝑝
from the estimated style range to obtain the projected feature 𝑓 𝑝 .

𝐷 (𝑣, 𝑝) = ‖𝐄𝑣 [𝜙(𝕌𝑁
(𝑓 𝑣 ))] − 𝐄𝑝 [𝜙(𝕌𝑁
(𝑓 𝑝 ))]‖2 ,
𝑖=1 𝑖
𝑖=1 𝑖
where 𝜎(𝜇(𝑓 𝑡 ))2 and 𝜎(𝜎(𝑓 𝑡 ))2 denote the estimator of the mean and

𝛾𝑠𝑠𝑝 = Random[ 𝜎(𝑓 𝑡 ) − 𝜖𝜎 𝜎(𝜎(𝑓 𝑡 )), 𝜎(𝑓 𝑡 ) + 𝜖𝜎 𝜎(𝜎(𝑓 𝑡 )) ],

ancy (MMD) between RGB and perturbed features not only reﬁnes the
shared identity-speciﬁc information but also enables further optimization within the IR modality.
Then, we continue to compute the metric distance between 𝕌𝑁
(𝑓 𝑡 )
𝑖=1 𝑖
𝑝
and 𝕌𝑁
(𝑓
)
,
which
is
denoted
as:
𝑖=1 𝑖

(6)

𝐷 (𝑡, 𝑝) = ‖𝐄𝑣 [𝜙(𝕌𝑁
(𝑓 𝑡 ))] − 𝐄𝑝 [𝜙(𝕌𝑁
(𝑓 𝑝 ))]‖2 .
𝑖=1 𝑖
𝑖=1 𝑖

where 𝜖𝜇 and 𝜖𝜎 are hyper-parameters that are sampled from the standard Gaussian distribution, i.e. 𝜖𝜇 ∼  (0, 1), 𝜖𝜇 ∼  (0, 1). Random(.)
means the random sampling within the intervals.
Finally, we replace the 𝛽 and 𝛾 in Eq. (4) by 𝛽𝑠𝑠𝑝 and 𝛾𝑠𝑠𝑝 and the
generated SSP feature 𝑓 𝑝 is formulated as:
𝑓 𝑝 = 𝛾𝑠𝑠𝑝 ⊙

𝑓 𝑡 − 𝜇(𝑓 𝑡 )
+ 𝛽𝑠𝑠𝑝 .
𝜎(𝑓 𝑡 )

(9)

where the perturbed feature (𝑓𝑖𝑝 ) shares semantic information with the
IR feature (𝑓𝑖𝑡 ). As a result, optimizing the Maximum Mean Discrep-

standard deviation, respectively.
After obtaining the estimator for each channel, we remodel the feature statistics by introducing the Gaussian distribution and performing
random sampling within the speciﬁed style change interval:
𝛽𝑠𝑠𝑝 = Random[ 𝜇(𝑓 𝑡 ) − 𝜖𝜇 𝜎(𝜇(𝑓 𝑡 )), 𝜇(𝑓 𝑡 ) + 𝜖𝜇 𝜎(𝜇(𝑓 𝑡 )) ],

(8)

where 𝕌𝑁
𝑖 (.) is the set of corresponding features, 𝜙(.) is an implicit feature mapping function,  refers to Reproducing Kernel Hilbert Space
and 𝑁 is the number of images in the 𝑖th identity.
Previous methods (Jambigi et al., 2021; Wu et al., 2022) directly apply this constraint to align the RGB and IR distributions. However, in the
case of imbalance VI-ReID, where the quantity of IR images signiﬁcantly
trails behind that of RGB images, directly minimizing MMD across the
visible and infrared modalities would lead the infrared modality mindlessly lean towards visible modality instead of aligning their shared information, which limits the performance improvement.
In this paper, beneﬁting from oversampling of SSP features, we propose the cross-optimization of two distributions. Speciﬁcally, for the
RGB distribution 𝕌𝑁
(𝑓 𝑣 ), we additionally calculate the metric distance
𝑖=1 𝑖
𝑣 ) and 𝕌𝑁 (𝑓 𝑝 ), denoted as:
between 𝕌𝑁
(𝑓
𝑖=1 𝑖
𝑖=1 𝑖

(10)

Here, it should be emphasized that optimizing 𝐷 (𝑡, 𝑝) serves a dual
purpose: it reduces the discrepancy within the same modality and guarantees that the oversampled features are aligned with the original distribution.
Finally, the cross distribution alignment loss can be formulated as
follows:

(7)

𝑐𝑟𝑜𝑠𝑠 = 𝐷 (𝑣, 𝑡) + 𝐷 (𝑣, 𝑝) + 𝐷 (𝑡, 𝑝).

We position the SSP module after the CNN layer, notably at the second,
third, and fourth stages of the ResNet50 model. Given that SSP operates
without parameters, the computational overhead it introduces is minimal. For the inference phase, SSP is excluded from the feature extraction
process.
Discussion. Our proposed Stochastic Style Perturbation (SSP) draws
inspiration from ApaIN (Huang & Belongie, 2017), which achieves arbitrary image style transfer by substituting the feature statistics of a
content image with those from a style image. However, ApaIN (Huang
& Belongie, 2017) requires the target image as a reference, which does
not align with our task’s requirements. More recently, Zhou et al. introduced MixStyle (Zhou et al., 2023), a method that linearly combines the
statistics of two samples, and Li et al. (2022) designed DSU, which introduces disturbances to the original feature statistics. While these methods
have been successful in domain generalization as CNN training regularization strategies, their synthesized features remain closely related to
those of the original training images, diverging from the spirit of our
co-modlaity balance learning.
Unlike previous feature normalization methods, SSP focuses on dynamically altering feature statistics from a Gaussian distribution interval. This approach ensures the generation of new features that are meaningful and independent from the original styles in each batch, oﬀering
a distinct advantage in versatility and applicability to data imbalance
datasets.

(11)

3.3. Class-aware contrast similarity learning
Cross distribution alignment regularization mainly focus on the
alignment of global distributions, while ignoring the intra and inter class
correlations of features. By considering the relationships between different classes on a ﬁne-grained instance level, we impose the contrast
similarity constraint to the center of each classes to minimize intra-class
separation while maximizing inter-class separability. Speciﬁcally, given
the feature outputs {𝑓𝑖𝑣 , 𝑓𝑖𝑡 , 𝑓𝑖𝑝 } of the same identity, we ﬁrst calculate
the center of the visible modality 𝑐𝑖𝑣 and the aggregated infrared modality feature center 𝑐𝑖𝑡 :
𝑐𝑖𝑣 =

𝑁
𝑁
1 ∑ 𝑣 𝑡
1 ∑ 𝑡
(𝑓𝑖 ), 𝑐𝑖 =
(𝑓 + 𝑓𝑖𝑝 ),
𝑁 𝑖=1
2𝑁 𝑖=1 𝑖

(12)

Then, we conduct the contrast similarity learning from following two
aspects.
Intra-class Contrast. Intra-class contrast uses cross distribution selfto-self matching to achieve discriminative representations by maximizing self-similarity. To measure the degree of similarity between classes,
we use cosine distance to calculate the relation matrixs  𝑣𝑡 = 𝑐𝑖𝑣 (𝑐𝑖𝑡 )𝑇 ∈
ℝ𝑁×𝑁 , where 𝑇 means matrix transpose operation. Then, we employ
softmax operation to calculate its probability distribution matrix as:
 𝑣𝑡 = 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥( 𝑣𝑡 ),

3.2. Cross distribution alignment

(13)

where 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥(.) is row-wise softmax operation.
To maximize the matching probability of self-to-self, we further employ the cycle consistency strategy to achieve inverse relational matrix

To reduce the discrepancy between the diﬀerent distributions within
the shared space, we propose a cross distribution alignment loss. Given
4

Neural Networks 195 (2026) 108206

H. Liu et al.

Fig. 5. The prediction distribution of 100 IR testing samples of baseline (a) and CMBL (b) on the SYSU-MM01 dataset. The x-axis denotes the normalized logit
output, and x=0.5 represents the classiﬁcation boundary. The y-axis denotes the index of samples. Red and blue dots denote two diﬀerent classes.

 𝑡𝑣 = (𝑐 𝑡 )𝑇 𝑐 𝑣 . Similarly to Eq. (13), the probability matrix can be calculated as:
 𝑡𝑣 = 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥( 𝑡𝑣 ).

Finally, the class-aware contrast similarity loss is formulated by:
𝑐𝑙𝑎𝑠𝑠 = 𝜔1 𝑖𝑛𝑡𝑟𝑎 + 𝜔2 𝑖𝑛𝑡𝑒𝑟 ,

(14)

Here, the corresponding results  𝑣𝑡 and  𝑡𝑣 are regarded as matching
the same classes from visible to infrared modality and the same classes
from infrared to visible modality.
According to the of cycle association consistency, following the forward association  𝑣𝑡 and backward association  𝑡𝑣 , we further enhance
our intra-class contrast learning through self-to-self matching:
𝔻(𝑣, 𝑡) = diag( 𝑣𝑡  𝑡𝑣 ),

3.4. Training objective
We combine the following loss functions to make up the ﬁnal training
objective:
 = 𝐶𝐸 + 𝑐𝑟𝑜𝑠𝑠 + 𝑐𝑙𝑎𝑠𝑠 ,

(15)

(22)

where 𝐶𝐸 represents the cross entropy loss. The synergy among these
loss functions is critical for reﬁning the model’s ability to distinguish between classes, align distributions across modalities, and maintain generalization capabilities.

where diag(⋅) means to get the diagonal matrix. The intra-class contrast
loss then can be calculated as:
1 ∑
𝑖𝑛𝑡𝑟𝑎 =
(||𝔻(𝑣, 𝑡) ⊖ 𝕀||1 ),
(16)
𝑁
where ⊖ denotes the element-wise subtraction operation, ||.||1 denotes
the 𝐿1 norm and 𝕀 means the identity matrix, i.e.,
{
0,
if 𝑖 ≠ 𝑗
𝕀𝑖,𝑗 =
0 ≤ 𝑖, 𝑗 < 𝑁
(17)
1,
otherwise


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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Framework|ATG|Contour|TRM|Text-Guided|Experimental|Ablation|Prototype|Loss)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:Text-Guided Cross-Modal Alignment with Attribute and
2:Contour Prototypes for Visible-Infrared Person
26:Prototypes for Visible-Infrared Person Re-Identification. ACM Trans. Multimedia Comput. Commun. Appl. 22,
27:4, Article 113 (March 2026), 18 pages.
90:TRM are then adopted as shared semantic anchors to align both RGB and IR features within a
132:2.1
156:2.2
188:3
189:3.1
191:Method
192:Framework Overview
226:3.2
228:ATG Module
265:3.3
267:Contour-Aware Text Prompt Module
359:3.4
361:TRM
487:3.5
489:Text-Guided Attribute and Prototype Alignment Model

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-to-image Person Search based on Semantic Reorganization.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Semantic|Cross|Training|Ablation|Comparison|Analysis)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
22:Cross-modal retrieval
25:Cross-modal re-ranking
26:Semantic alignment
196:2.2. Cross-modal alignment-based person search
255:2.1. Data augmentation for person search
278:3. Method
282:3
302:3.2. Network structure
303:3.1. Semantic restructuring-based data augmentation
411:4
455:4. Experiments
456:4.1. Datasets and protocol
460:3,078/3,074 images. The ICFG-PEDES (Ding et al., 2021) dataset consists of 54,522 images corresponding to 4102 individuals, with one caption per image. The training/testing set includes 34,674/19,848 image–
469:3.3. Cross-modal similarity re-ranking mechanism
482:4.2. Implementation details
494:4, and the momentum coefficient 𝜆 in Eq. (11) is set to 0.9.
506:4.3. Comparison with the state-of-the-art methods
584:Comparison with the SOTA methods.
585:Method
634:64.04
635:64.44
636:69.47
637:66.27
638:69.57
639:71.38
640:73.38
641:73.91
642:74.46
643:74.65
644:75.28
645:75.66
646:72.17
649:58.73
650:60.52
651:57.04
653:67.91
654:66.13
655:67.28
657:67.49
659:66.58
660:64.33
664:57.69
665:57.31
666:60.83
667:63.05
668:63.46
669:63.87
670:64.72
671:64.01
672:65.76
673:65.61
674:62.64
678:36.07
682:38.06
683:39.12
685:39.64
687:39.14
688:36.50
690:45.55
694:50.55
696:60.20
698:61.49
699:60.20
700:61.80
701:62.48
702:60.00
710:47.17
713:47.38
715:48.86
716:47.37
722:75.71
724:73.34
726:65.91
728:48.11
730:62.53
732:51.77
735:Ablation results for each component of PSSR.
740:3
741:4
784:70.79
785:74.35
786:72.02
787:75.71
789:64.69
790:69.63
791:68.13
792:73.34
794:61.30
795:64.85
796:62.20
797:65.91
799:37.68
800:45.31
801:44.91
802:48.11
804:59.20
805:61.00
806:60.31
807:62.53
809:46.65
810:48.18
811:49.68
812:51.77
847:4.4.2. Ablation studies for the triplet loss
861:4.4. Ablation study
863:4.5. More discussions
865:4.4.1. Ablation studies for each component of pssr
877:4.5.1. Comparison of the traditional re-ranking and our CSRM
883:Cross-modal Similarity Re-ranking Mechanism (CSRM) to select reliable
895:Ablation results for the triplet loss.
900:3
926:75.07
927:75.45
928:75.71
930:72.03
931:72.95
932:73.34
934:65.10
935:65.35
936:65.91
938:46.72
939:47.53
940:48.11
942:61.70
943:62.05
944:62.53
946:50.22
947:51.15
948:51.77
951:Comparison of traditional re-ranking and our CSRM.
956:3
982:74.35
983:68.68
984:75.71
986:69.63
987:67.76
988:73.34
990:64.85
991:64.28
992:65.91
994:45.31
995:47.29
996:48.11
998:61.00
999:56.40
1000:62.53
1002:48.18
1003:49.14
1004:51.77
1021:4(b)), samples of the same identity form more compact clusters, illustrating that CSRM successfully mitigates cross-modal discrepancies and
1052:4.5.2. Effectiveness of CSRM
1080:4.5.3. Analysis of image generation
1119:3
1130:74.35
1131:74.93
1132:75.71
1134:69.63
1135:72.17
1136:73.34
1146:3
1147:4
1159:70.79
1160:73.25
1161:72.83
1162:74.35
1164:64.69
1165:67.95
1166:67.67
1167:69.63

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | rg -n \""'^(3|3'"\\.|4|4\\.|[0-9]+\\.[0-9]|Method|Overview|Multi|Pose|Texture|Loss|Ablation|Effectiveness|Comparison)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8:Texture-aware transformer with pose-patch mapping for occluded person
25:Pose-patch mapping
26:Texture aware decoder
35:Multi-patch Feature Encoder is proposed to encode discriminative features from inter patches and intra patches.
87:Pose-Patch Mapping (PPM), and a Texture-Aware Decoder (TAD). To
154:2.1. Occluded person Re-ID
174:3.1. Overview
231:2.2. Transformer-based person Re-ID
257:3.2. Multi-patch feature encoder
293:3. Proposed method
295:Pose-Patch Mapping (TTPM). We ﬁrst present an overview of TTPM in
298:3
400:3.3. Pose-patch mapping
417:4
626:3.4.1. Context extraction layer
684:3.4.2. Texture extraction layer
694:3.4. Texture-aware decoder
742:Pose Loss: Due to the accuracy of pose mapping aﬀects the optimization of the whole model, we add the pose loss constraint to the total
767:3.5. Loss function
774:4. Experiments
776:4.1. Datasets and evaluation metrics
837:4.2. Implementation details
849:0.9 and 1e-4, respectively. All experiments are run in 300 epochs with
864:42.6
865:47.0
866:55.1
867:57.6
868:64.5
869:55.4
870:64.4
871:66.7
872:61.0
873:68.3
875:33.7
876:37.7
877:43.8
878:46.3
879:53.6
880:43.8
881:49.7
882:54.1
883:50.1
884:54.3
886:41.3
887:70.4
888:80.3
890:81.6
891:82.5
893:76.9
894:71.8
897:38.9
898:61.2
899:70.2
901:72.1
902:71.8
904:68.6
905:51.0
917:66.4
918:67.7
919:68.1
920:62.8
921:70.0
922:71.8
923:68.6
924:66.4
926:59.2
927:60.1
928:56.4
929:54.9
930:62.2
931:62.2
932:57.4
933:57.8
935:70.2
936:79.8
937:86.3
938:80.4
939:83.0
940:82.6
941:86.8
944:67.3
945:81.3
946:79.3
947:76.1
948:76.8
949:78.2
950:81.3
955:73.4
957:63.4
959:86.8
961:82.2
963:Methods
1003:Methods
1007:4.3. Comparison with the state-of-the-art
1008:4.3.1. Results on occluded datasets
1048:93.8
1049:95.3
1050:95.2
1051:95.6
1052:95.1
1053:95.4
1055:81.6
1056:88.6
1057:88.9
1058:89.5
1059:87.9
1060:88.0
1062:69.2
1063:89.6
1064:90.7
1065:90.4
1066:87.9
1067:90.1
1069:83.3
1070:80.0
1071:82.0
1072:81.4
1073:77.9
1074:80.9
1100:94.2
1101:95.4
1102:95.5
1103:95.3
1104:95.1
1105:95.1
1106:95.3
1107:95.5
1108:95.6
1109:95.0
1110:95.2
1112:84.9
1113:88.0
1114:89.6
1115:87.2
1116:88.0
1117:87.0
1118:88.2
1119:89.8
1120:89.5
1121:86.8
1122:89.0
1124:86.9
1125:88.8
1126:90.6
1127:88.5
1128:89.3
1129:89.6
1130:89.1
1132:91.2
1133:89.7
1134:90.1
1136:75.6
1137:78.2
1138:82.2
1139:77.9
1140:79.8
1141:78.3
1142:80.2
1144:83.1
1145:78.8
1146:80.8
1152:95.6
1154:90.0
1156:91.4
1158:83.5
1168:4.3.2. Results on holistic datasets
1177:4.3.3. Results on partial datasets
1199:Methods
1229:68.0
1230:75.3
1231:85.3
1232:76.7
1233:86.0
1234:55.4
1235:77.3
1236:86.0
1238:80.0
1239:86.0
1240:91.0
1241:84.3
1243:43.8
1244:85.7
1245:91.3
1247:69.1
1249:72.6
1250:79.8
1251:77.3
1252:82.5
1253:81.7
1254:78.2
1256:80.9
1258:86.4
1259:86.6
1261:88.3
1262:88.0
1263:87.4
1277:81.3
1278:67.6
1279:83.1
1280:76.3
1281:87.2
1283:88.0
1284:74.3
1286:85.3
1287:93.5
1289:73.1
1290:69.7
1292:73.9
1293:77.0
1295:84.9
1296:80.7
1298:86.6
1299:87.6
1305:87.4
1307:93.6
1309:77.3
1311:88.5
1318:3
1319:4
1340:59.4
1341:62.2
1342:67.3
1343:68.9
1344:73.4
1346:49.7
1347:52.4
1348:56.1
1349:57.0
1350:63.4
1352:72.1
1353:74.5
1354:80.9
1355:82.6
1356:86.8
1358:67.6
1359:69.9
1360:75.2
1361:77.6
1362:82.2
1384:70.9
1385:69.1
1386:73.4
1388:81.6
1389:79.3
1390:84.6
1392:82.7
1393:81.7
1394:88.5
1396:59.0
1397:57.7
1398:63.4
1402:4.4.2. Eﬀectiveness of the proposed PPM
1408:6.7 %/4.6 % and 8.1 %/7.7 % increases in Rank-1/mAP on two occluded
1423:Ablation Study on Occluded-Duke and Occluded-REID (%). MFE,
1425:Pose-Patch Mapping and Texture-Aware Decoder. Bold indicates the
1429:Method
1432:77.3 %/88.5 % in terms of Rank-1/Rank-3 on Partial-REID and PartialiLIDS, respectively. The results indicate that although our method also
1438:4.4.3. Eﬀectiveness of the proposed TAD
1457:4.4. Ablation study
1459:Pose-Patch Mapping (PPM), and Texture-Aware Decoder (TAD) are
1463:4.4.1. Eﬀectiveness of the proposed MFE
1469:2.8 %/2.4 % and 2.7 %/2.3 % on two occluded datasets, respectively.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | sed -n '188,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3
3.1

Method
Framework Overview

𝑁 containing 𝑁 paired samples of visible (RGB) images 𝑥 𝑣 ,
Given a training dataset D = {𝑥𝑖𝑣 , 𝑥𝑖𝑟 , 𝑦𝑖 }𝑖=1
𝑖
𝑟
IR images 𝑥𝑖 , and their associated identity labels 𝑦𝑖 ∈ {1, . . . , 𝐶}, where 𝐶 denotes the total number
of identity classes, our framework introduces a text-enhanced cross-modal alignment strategy to
boost VI-ReID.
To exploit semantic priors, we first leverage a VLM to generate image-conditioned textual
𝑁 , where 𝑡 𝑣 and 𝑡 𝑟 describe the fine-grained appearance attributes of 𝑥 𝑣
descriptions T = {𝑡𝑖𝑣 , 𝑡𝑖𝑟 }𝑖=1
𝑖
𝑖
𝑖
𝑟
and 𝑥𝑖 , respectively. These descriptions capture detailed semantics such as clothing, carried objects,
or color hints, serving as auxiliary supervision to the visual encoders. We further observe that
LLM-generated descriptions implicitly reflect contour-aware cues, which are crucial for identity
alignment under modality variations. To explicitly model such structure, we propose a CTP module
that constructs contour-level textual descriptions T𝑠 = {𝑡𝑖𝑠 }𝐶𝑖=1 for each identity class. Unlike
instance-level descriptions T, these contour prompts emphasize shared structural patterns within
each identity class.
To effectively integrate both instance-level and class-level textual semantics, we introduce a TRM,
𝑁 .
which fuses 𝑡𝑖𝑣 , 𝑡𝑖𝑟 , and 𝑡 𝑦𝑠 𝑖 into unified, identity-sensitive textual representations T𝑜 = {𝑡𝑖𝑣𝑜 , 𝑡𝑖𝑟𝑜 }𝑖=1
This joint representation not only preserves detailed attribute cues but also inherits discriminative
identity patterns from Contour-aware prompts. Furthermore, we compute the Text Prototype
Discriminative Feature Set T𝑐 = {𝒇𝑖𝑡 }𝐶𝑖=1 by aggregating the class centers of the redefined textual
features. These prototypes serve as modality-invariant anchors that guide the alignment of visual
features across modalities. By enforcing cross-modal features to align with corresponding text
prototypes, we establish a semantic bridge between visible and IR domains, enabling robust identity
matching.

3.2

ATG Module

Pedestrian attributes serve as crucial semantic cues for cross-modal person ReID. To explicitly
incorporate such discriminative information, as illustrated in Figure 3, we design a structured
attribute-based text generation scheme that converts visual attributes into formalized textual
descriptions. Specifically, we construct a pedestrian attribute template 𝜏, grounded in established
attribute taxonomies, to standardize the descriptive process. The template is defined as:
“A {age} {gender} is {action}, wearing {upper clothes}, {lower-body clothes},
and {shoe type}. The {gender} has {short or long} hair, {yes or no} the person is
wearing glasses.”

To populate this template with image-specific content, we develop a parametrized question set
corresponding to each attribute slot. For instance, we design prompts such as: “Q1: The person is
young, a teenager, an adult, or elderly?” “Q2: The person is male or female?” “Q3: The person is
walking or standing?” and so on for clothing style, hair length, and accessories. We employ a visual
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

113:6

Y. Tao and X. Zhang

question-answering paradigm based on the BLIP model to extract the relevant attribute values.
For each input image, the BLIP model infers answers conditioned on both the image content and
our designed questions. The collected answers are then assembled into complete attribute-based
descriptions following the template 𝜏, yielding textual representations 𝑡𝑖𝑣 and 𝑡𝑖𝑟 for the 𝑖th visible and
IR images, respectively. This template-driven strategy brings several advantages. First, it enforces
a unified syntactic and semantic structure across all generated descriptions, which mitigates the
noise and diversity commonly observed in free-form language generation. Second, by avoiding
direct references to low-level color cues—which are often modality-specific and unreliable in IR
imagery—the generated descriptions focus on modality-invariant attributes, such as age, gender,
action, clothing type, and accessories. These carefully selected attributes offer stable semantic
anchors for visual–textual alignment across modalities.
𝑁 , which encodes
As a result, we construct an attribute-enhanced textual dataset T = {𝑡𝑖𝑣 , 𝑡𝑖𝑟 }𝑖=1
structured and fine-grained semantic information. This representation not only facilitates better
visual–textual alignment but also introduces human-interpretable and modality-agnostic cues into
the ReID process.
3.3

Contour-Aware Text Prompt Module

While the attribute-aware textual descriptions provide valuable cues for pedestrian recognition,
they inherently lack explicit contour-related semantics. Moreover, describing fine-grained sketch
information through natural language is non-trivial. To address this issue, as illustrated in Figure 4,
we draw inspiration from the CoOP [26] approach and aim to learn a set of class-discriminative
prompt tokens that can implicitly model pedestrian contour information.
Specifically, we employ the topologically aware human parsing framework SCHP [11] to decompose the input pedestrian image 𝑥𝑖 (where 𝑥𝑖 can be either 𝑥𝑖𝑣 or 𝑥𝑖𝑟 ) into 20 semantic categories,
including background, hair, upper clothes, dress, coat, pants, skirt, face, left arm, and others. To
isolate sketch-relevant structures (e.g., body shape and pose) while suppressing clothing-based
variations, we construct a binary mask M𝑠𝑖 by setting all clothing-related regions to 0 and all
remaining semantic regions to 1. Using this mask, we generate a sketch-style image 𝑥𝑖𝑠 by retaining
sketch-relevant pixels and replacing the rest with a constant background value:
𝑥𝑖𝑠 = M𝑠𝑖

𝑥𝑖 + (I − M𝑠𝑖 )

255,

(1)

where denotes the Hadamard (element-wise) product, and 𝑥𝑖𝑠 represents the sketch-style version
of the original image 𝑥𝑖 .
To enable textual modeling of sketch semantics, we adopt a learnable prompt tuning strategy
inspired by CLIP-ReID [4]. We define an identity-related textual template: 𝒕𝑖𝑠 = “A photo of a
[𝑋 1 ] [𝑋 2 ] [𝑋 3 ] . . . [𝑋𝑀 ] person”, where each [𝑋 ] 𝑖 is a learnable token and 𝑀 is the number of
tokens. These tokens are optimized to capture contour-aware identity semantics based on visual
guidance from 𝑥𝑖𝑠 .
Following CLIP-ReID training, we encode the image 𝑥𝑖𝑠 and the corresponding prompt t𝑠𝑖 using
the visual encoder 𝐸 𝑣 and the text encoder 𝐸𝑡 to obtain feature embeddings:
𝒇𝑖𝑠,𝑣 = 𝐸 𝑣 (𝑥𝑖𝑠 ),

𝒇𝑖𝑠,𝑡 = 𝐸𝑡 (𝒕𝑖𝑠 ).

(2)

To ensure that the learned prompt tokens t𝑠𝑖 are identity-discriminative, we employ a symmetric
image–text alignment objective consisting of an image-to-text loss and a text-to-image loss:
L𝑎𝑙𝑖𝑔𝑛 = L𝑖2𝑡 + L𝑡 2𝑖 ,

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

(3)

Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID

113:7


𝑁𝑏
exp 𝜓 f𝑖𝑠,𝑣 , f𝑖𝑠,𝑡
1 Õ
 
 ,
L𝑖2𝑡 = −
log Í
𝑁𝑏
𝑁𝑏 𝑖=1
exp 𝜓 f 𝑠,𝑣 , f 𝑠,𝑡

(4)


𝑁𝑏
exp 𝜓 f𝑖𝑠,𝑣 , f𝑖𝑠,𝑡
1 Õ
 
 ,
L𝑡 2𝑖 = −
log Í
𝑁𝑏
𝑁𝑏 𝑖=1
exp 𝜓 f 𝑠,𝑣 , f 𝑠,𝑡

(5)

𝑖

𝑗=1

𝑗

𝑗

𝑗=1

𝑖

where 𝜓 (·) denotes the similarity function (e.g., cosine similarity), and 𝑁𝑏 is the mini-batch size.
Through the above procedure, we obtain a set of Contour-aware textual features T𝑐 = {𝒇𝑖𝑡 }𝐶𝑖=1 ,
which serve as compact and discriminative contour-aware representations for cross-modal identity
alignment.
3.4

TRM

Although the instantiated attribute texts are generated per image and the Contour-aware prompts
are constructed at the class level, the inherent semantic gap between these two types of textual
descriptions limits their direct applicability in unified visual-language representation. To bridge this
gap, we introduce a TRM, aiming to integrate instance-level attribute descriptions and class-level
contour cues into a unified and discriminative textual feature space.
As illustrated in Figure 5, given an input image 𝑥𝑖 (either RGB or IR), we first extract its visual
representation 𝒇𝑖 using the visual encoder. In parallel, we feed its attribute-based textual description
𝒕𝑖 and the corresponding sketch-based class textual feature 𝒕𝑖𝑠 into a textual fusion network 𝐸𝑎
composed of three fully connected layers to obtain an initial fused textual representation:
(6)

𝒇𝑖𝑎 = 𝐸𝑎 ([𝒕𝑖 ; 𝑡𝑖𝑠 ]),

where [; ] denotes concatenation.
To further model cross-modal interaction, we introduce a Transformer-based decoder module 𝐸𝑞
composed of three layers of TransDecoder blocks. This module takes both the visual feature 𝒇𝑖 and
𝑞
the fused text 𝒇𝑖𝑎 as input to generate a redefined, image-aware textual representation 𝒇𝑖 :
𝒇𝑖 = TransDecoder(𝒇𝑖 , 𝒇𝑖𝑎 ),

(7)

𝑞

which is then refined via a residual connection:
𝑞

𝑞

(8)

𝑞

𝒇𝑖 ← 𝒇𝑖 + 𝛾𝒇𝑖 ,

where 𝛾 ∈ R𝐶 is a learnable scaling factor initialized to a small value (e.g., 10−4 ) to preserve the

semantic priors of the original textual input.
𝑞
To ensure that the redefined textual features 𝒇𝑖 are both semantically aligned with the visual content and discriminative with respect to pedestrian identity, we employ a combination of contrastive
and classification losses. Specifically, the training objective is:
𝑞

𝑞

𝑞

𝑞

(9)

L𝑎𝑙𝑖𝑔𝑛 = L𝑖2𝑡 + L𝑡 2𝑖 + L𝐼 𝐷 ,
where the image-to-text and text-to-image contrastive losses are defined as:
𝑞 
𝑁𝑏
exp 𝜓 𝒇𝑖 , 𝒇𝑖
1 Õ
𝑞
 

L𝑖2𝑡 = −
log Í
𝑞 ,
𝑁𝑏
𝑁𝑏 𝑖=1
exp 𝜓 𝒇 , 𝒇
𝑗=1

𝑖

(10)

𝑗

𝑞 
𝑁𝑏
exp 𝜓 𝒇𝑖 , 𝒇𝑖
1 Õ
𝑞
L𝑖2𝑡 = −
log Í𝑁
𝑞  ,
𝑏
𝑁𝑏 𝑖=1
𝑗=1 exp 𝜓 𝒇 𝑗 , 𝒇𝑖

(11)

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

113:8

Y. Tao and X. Zhang

and the identity classification loss is:
𝑁

𝑞

L𝐼 𝐷 = −

𝑏
1 Õ
𝑞
𝒒𝑖 log(𝑾𝑡 𝒇𝑖 ),
𝑁𝑏 𝑖=1

(12)

where 𝜓 (·) denotes a similarity metric (e.g., cosine similarity), 𝒒𝑖 is the one-hot label vector for
identity 𝑦𝑖 , and 𝑾𝑡 is the identity classifier.
The TRM module consists of four components: a Text Encoder, a Visual Encoder, a Fusion
Network, and an Interaction Network. The Text Encoder and Visual Encoder adopt the standard
CLIP Text Encoder and Visual Encoder, respectively. The Fusion Network is constructed with three
fully connected layers, with output dimensions of 1,024, 512, and 1,024, respectively. The Interaction
Network comprises three Transformer Decoder Blocks, each containing four sub-modules: Masked
Multi-Head Self-Attention, Cross-Attention, Feed-Forward, and LayerNorm. Within each block,
the multi-head attention consists of 8 heads, each with a dimension of 64; the feature dimension of
each token is 512, and the Feed-Forward layer produces a final output of dimension 1,024. During
training, we freeze the parameters of the visual encoder and text encoder and update only the
components of the fusion module 𝐸𝑎 , the interaction module 𝐸𝑞 , and the identity classifier 𝑾𝑡 .
𝑞
After convergence, the redefined feature 𝒇𝑖 serves as an enriched and interaction-aware textual
representation, effectively capturing both attribute-level and contour-level semantics for each image.
3.5

Text-Guided Attribute and Prototype Alignment Model

In this section, we present our proposed model, text-guided attribute and prototype alignment
(TAP-align). As illustrated in Figure 2, the proposed model adopts a dual-branch architecture
commonly used in VI-ReID tasks and employs the visual encoder from a pre-trained CLIP model as
its backbone. Specifically, the first four layers of the visual encoder are utilized to construct a parallel,
shallow structure, serving as modality-specific feature extractors, 𝐸 𝑣,𝑣 and 𝐸 𝑣,𝑖 , which extract
features from RGB and IR images, respectively. The remaining layers are shared and constitute the
encoder 𝐸, which is responsible for learning modality-invariant visual representations.
The model is trained using a joint optimization strategy that combines the identity classification loss and the weighted regularized triplet loss. The identity loss supervises the learning of
discriminative identity features. In contrast, the weighted regularized triplet loss assigns adaptive weights to positive and negative sample pairs, thereby enhancing the model’s capability to
extract modality-invariant features. This joint learning strategy proves particularly effective in
cross-modality person ReID scenarios, where RGB and IR images exhibit significant appearance
discrepancies.
To further improve cross-modality alignment, we explicitly mine potential modality-invariant
textual cues from images as semantic guidance. In particular, we introduce a Text Prototype
Discriminative Loss, which encourages visual features to align with class-level text prototypes that
carry identity-related semantic information. This strategy significantly enhances both semantic
consistency and discriminative capability in cross-modality matching. Finally, the overall joint
optimization objective is defined as:
𝑣𝑠
𝑟𝑠
L𝑡𝑜𝑡𝑎𝑙 = L𝑖𝑑 + L𝑤𝑟𝑡 + 𝜆1 L𝑖2𝑡𝑐𝑒
+ 𝜆2 L𝑖2𝑡𝑐𝑒
,

(13)

where identity loss L𝑖𝑑 , weighted regularized triplet loss L𝑤𝑟𝑡 , text prototype discriminative loss
𝑣𝑠 , and L𝑟𝑠
L𝑣2𝑡𝑐𝑒
𝑖2𝑡𝑐𝑒 are formally defined as :
𝑛

L𝑖𝑑 = −

𝑏
1 Õ
𝒒𝑖 log(𝑾𝑐 𝒇𝑖 ).
𝑛𝑏 𝑖=1

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

(14)


hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Text-to-image Person Search based on Semantic Reorganization.pdf' - | sed -n '278,505p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3. Method
To address the issue of insufficient diversity of training data and significant modal differences in TBPS tasks, as shown in Fig. 2, we propose
the PSSR framework. Firstly, we introduce semantic restructuring-based
data augmentation to get more diverse data. Secondly, all original and
3

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

Fig. 2. Overview of the proposed PSSR. The model is built upon a CLIP-based dual-stream network. We integrate a Semantic Restructuring-based Data
Augmentation (SRDA) into the framework to enhance the diversity of training data. Additionally, during the testing phase, we introduce a Cross-modal Similarity
Re-ranking Mechanism (CSRM) to effectively mitigate the negative impact of modality differences on retrieval performance.

images. Finally, the generated image–text pairs { ̂
𝐼, 𝑇̂ } are obtained,
expanding the dataset’s size and diversity, which is crucial for the
model’s performance.

augmented data are fed into a dual branch network for training. Finally,
a cross-modal similarity rearrangement mechanism in the testing phase
is proposed to address the modal differences. Next, these three parts are
described in detail.

3.2. Network structure
3.1. Semantic restructuring-based data augmentation
In order to better map images and text description to the common feature space, we designed a Clip-based dual-branch network
architecture. As shown in Fig. 2, The proposed network structure
consists of two unimodal encoders for image and text encoding, respectively. For a given image–text pair (𝐼, 𝑇 ), the image encoder first
processes the image 𝐼 to produce a series of visual representations
𝑣 , 𝑓 𝑣 , … , 𝑓 𝑣 }. Here, 𝑓 𝑣 denotes the global visual represen𝐹 𝑣 = {𝑓cls
𝑀
1
cls
tation, while 𝑓𝑖𝑣 (𝑖 = 1, … , 𝑀) denotes the local patch representation.
Similarly, the text 𝑇 undergoes processing by the text encoder, yielding
𝑡 , 𝑓 𝑡 }. Here, 𝑓 𝑡
a series of textual representations 𝐹 𝑡 = {𝑓1𝑡 , … , 𝑓𝑁
eos
eos
denotes the global textual representation.
During the model training phase, both original and generated data
will be used to update our model. Specifically, both the identification
loss (ID loss) and similar distribution matching loss (SDM loss) were
adopted. The ID loss improves the model’s accuracy in the identity
classification task. The SDM loss optimizes the alignment between
image and text representations by minimizing the discrepancy between
predicted matching probabilities and true matching labels. Given a
mini-batch of 𝑁 image–text pairs, the matching probability 𝑝𝑖,𝑗 is
calculated using the following softmax function:

In order to enhance the diversity of image–text pairs, we propose a
data augmentation method based on semantic information and pedestrian image structure information. Firstly, for the original pedestrian
images, a multimodal large language model (MLLM) is used to determine the gender of the pedestrian, and the original images are divided
into two groups based on gender, i.e., the male image group 𝐼male and
female image group 𝐼female . Secondly, we randomly select two different
IDs to combine within each gender group. Specifically, for each selected
ID pair (e.g., 𝐼𝐷1 and 𝐼𝐷2 ), we randomly some images with 𝐼𝐷1 to
provide the ‘‘upper body’’ and the same number of images with 𝐼𝐷2
to represent the ‘‘lower body’’. Afterward, we horizontally split the
upper body and lower body images to obtain new upper–lower body
image pairs, and then we concatenate them to generate new images
𝐼̂ = 𝑐𝑜𝑛𝑐𝑎𝑡(𝐼upper , 𝐼lower ), where 𝐼upper represents the selected upper
body image, and 𝐼lower indicates the selected lower body image. Since
the stitched images does not belong to any existing ID (class), we assign
it a completely new ID.
To ensure that the generated images hold reasonable and diverse
textual descriptions, we fine-tune the BLIP (Bootstrapping Language
Image Pretraining) model using the original image–text pairs. The BLIP
model generates textual descriptions based on image content. After
fine-tuning, it generates more semantically consistent descriptions with
respect to the generated images. For each augmented image, we take
the first three words of the corresponding upper body text as the
prompt, for example, ‘‘A man in’’, and input this prompt into the finetuned BLIP model to generate a new textual description. Specifically,
̂ and the corresponding upper body text be
let the generated image be 𝐼,
𝑇upper . We extract the first three words from 𝑇upper and combine them
into the prompt 𝑇𝑝𝑟𝑜 = {𝑤1 , 𝑤2 , 𝑤3 }. We then input the prompt into the
BLIP model to obtain the corresponding generated text 𝑇̂ :
̂
𝑇̂ = 𝐵𝐿𝐼𝑃 {𝑇𝑝𝑟𝑜 , 𝐼},

exp(sim(𝑓𝑖𝑣 , 𝑓𝑗𝑡 )∕𝜏)
𝑝𝑖,𝑗 = ∑𝑁
,
𝑣
𝑡
𝑘=1 exp(sim(𝑓𝑖 , 𝑓𝑘 )∕𝜏)

(2)

𝑇

𝑢 𝑣
where sim(𝑢, 𝑣) = ‖𝑢‖‖𝑣‖
represents the cosine similarity, and 𝜏 is a
temperature hyperparameter controlling the peak of the probability
distribution. The image-to-text SDM loss 𝐿𝑖2𝑡 is then computed using
the KL divergence between the predicted probabilities 𝑝𝑖,𝑗 and the true
matching probabilities 𝑞𝑖,𝑗 :

1 ∑∑
𝑝 log
𝑁 𝑖=1 𝑗=1 𝑖,𝑗
𝑁

𝐿𝑖2𝑡 =

𝑁

𝑦

(1)

where 𝑞𝑖,𝑗 = ∑𝑁 𝑖,𝑗

𝑘=1 𝑦𝑖,𝑘

This text 𝑇̂ serves as the textual description for the new image, further enhancing the diversity and semantic reliability of the generated

(

𝑝𝑖,𝑗
𝑞𝑖,𝑗 + 𝜖

)
,

(3)

represents the true matching probability, and 𝜖

is a small number to avoid numerical issues. Symmetrically, the textto-image SDM loss 𝐿𝑡2𝑖 is calculated by exchanging the image and text
4

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

representations. Finally, the bi-directional SDM loss is the sum of both
the image-to-text and text-to-image losses:

Subsequently, we use the Jaccard distance metric 𝐽 (⋅, ⋅) to compute the
similarity between the text feature 𝑡𝑚 and the image 𝑣𝑖 , defined as:

𝑠𝑑𝑚 = 𝐿𝑖2𝑡 + 𝐿𝑡2𝑖 .

𝐽 (𝑡𝑚 , 𝑣𝑖 ) = 1 −

(4)

To further reduce intra-modal differences, we introduce the intramodal triplet loss. This loss function promotes tight clustering of
features by minimizing the distance difference between positive and
negative samples within the same modality, thereby enhancing intramodal discrimination. Since the text is generated by a Multi-modal
Large Language Model (MLLM), it inevitably contains some noise,
which may cause the model to overfit to noise. Therefore, we use only
the image features to calculate the intra-modal triplet loss. Given a set
of visual feature representations for triplets (𝑓 𝑎𝑣 , 𝑓 𝑝𝑣 , 𝑓 𝑛𝑣 ), where 𝑓 𝑎𝑣
and 𝑓 𝑝𝑣 are positive sample pairs and 𝑓 𝑛𝑣 is a negative sample, the
triplet loss can be expressed as:
img_triplet =

𝑁 [
∑

(
)]
1
𝑎
𝑝
𝑎
𝑛
max sim(𝑓𝑖 𝑣 , 𝑓𝑖 𝑣 ) − sim(𝑓𝑖 𝑣 , 𝑓𝑖 𝑣 ) + 𝛿, 0 ,
𝑁 𝑖=1

𝑖

4. Experiments
4.1. Datasets and protocol
We evaluate our approach using three benchmark datasets. The
CUHK-PEDES (Li et al., 2017) dataset, a pioneer in text-based person retrieval, includes 34,054/68,108 images/sentences for 11,003
identities in the training set, with the validation/test set containing
3,078/3,074 images. The ICFG-PEDES (Ding et al., 2021) dataset consists of 54,522 images corresponding to 4102 individuals, with one caption per image. The training/testing set includes 34,674/19,848 image–
text pairs representing 3,102/1,000 identities. The RSTPReid (Zhu
et al., 2021) dataset contains 20,505 images of 4101 identities, with
two captions assigned to each image. The training/validation/testing
sets cover 3,701/200/200 identities, respectively. We adopt Rank-1 and
mean Average Precision (mAP) as evaluation metrics.

(6)

3.3. Cross-modal similarity re-ranking mechanism
During the testing phase, cross-modal matching faces the challenge
of modal gaps. In order to better solve this problem, we innovatively
propose a cross-modal re-rank method, whose core idea is to use
image features to represent text features and use the fused features for
retrieval. As shown in Fig. 2, we first calculate the similarity between
the query text feature 𝑡𝑚 and all gallery images {𝑣𝑖 }𝐺
. By applying the
1
k-nearest neighbor search, we rank the gallery images based on their
cosine similarity to the query, resulting in the following sorted image
set:

4.2. Implementation details
Our model is built upon the CLIP architecture. All input images are
resized to a fixed resolution of 384 × 128. The network is trained for
60 epochs using the Adam optimizer with an initial learning rate of
1 × 10−5 , which is decayed following a cosine annealing schedule to
promote stable convergence. A batch size of 64 is employed, and all
experiments are conducted on a single NVIDIA RTX 4090D GPU with
24 GB of memory.
The hyperparameters are configured as follows: 𝜏 = 0.02, 𝜖 = 1 ×
10−8 , and 𝛿 = 1.0 in Eqs. (2), (3), and (5), respectively. In the composite
loss function (Eq. (6)), the weighting factors are set to 𝛼 = 0.1, 𝛽 = 1.0,
and 𝛾 = 1.0. Furthermore, the neighborhood size 𝑘 in Eq. (7) is set to
4, and the momentum coefficient 𝜆 in Eq. (11) is set to 0.9.

(7)

Next, for each image 𝑣𝑖 ∈ 𝑁𝑘 (𝑡𝑚 ), we compute the cross-modal cosine
similarity between 𝑣𝑖 and all existing textual representations and obtain
the set of k-nearest texts of 𝑣𝑙 as 𝑁𝑘 (𝑣𝑖 ) = {𝑡1 , 𝑡2 , … , 𝑡𝑘 }. Therefore,
based on the bidirectional nearest-neighbor relation, we construct the
set of cross-modal k-reciprocal nearest images 𝑅𝑘 (𝑡𝑚 ) of query text
feature 𝑡𝑚 defined as:
𝑅𝑘 (𝑡𝑚 ) = {𝑣𝑖 ∣ (𝑣𝑖 ∈ 𝑁𝑘 (𝑡𝑚 )) ∩ (𝑡𝑚 ∈ 𝑁𝑘 (𝑣𝑖 ))}.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | sed -n '293,773p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3. Proposed method
This section details the proposed Texture-Aware Transformer with
Pose-Patch Mapping (TTPM). We ﬁrst present an overview of TTPM in
Section 3.1 and then introduce the motivation, objectives, and implementation of Multi-patch Feature Encoder (MFE), Pose-Patch Mapping
(PPM), and Texture-Aware Decoder (TAD) from Sections 3.2 to 3.4.
3

Pattern Recognition 172 (2026) 112341

D. Wang et al.

Fig. 2. The framework of Texture-Aware Transformer with Pose-Patch Mapping (TTPM). TTPM consists of three steps, including Step 1: the Multi-patch Feature
Encoder, which mines features from intra-patch and inter-patch levels. Step 2: the Pose-Patch Mapping, which separates pose semantics and occlusion semantics
through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.

(
( ))
𝐼𝑖 = 𝜗 𝑛𝑜𝑟𝑚 𝐼𝑖′ + 𝐼𝑖′ .

it better measures the feature distance under diverse scales and distributions, alleviating the inﬂuence of diﬀerent feature scales. The generic
formula for similarity calculation is as follows:
(
)
𝑆̃ 𝑥𝑖 , 𝑦𝑖 = 𝛾𝐷𝐌 − 𝜂𝑆𝐌
(8)

(5)

where 𝑐𝑎𝑡(⋅, ⋅) indicates concatenation along a speciﬁc channel. 𝜌(⋅)
includes softmax, linear variation, and element-wise multiplication.
𝑓 𝑙𝑎𝑡𝑡𝑒𝑛(⋅) denotes the ﬂattening operation. 𝑛𝑜𝑟𝑚(⋅) represents layer normalization. 𝜗(⋅) comprises mlp, gelu, and dropout operations.
At the inter-patch level, to further explore features within larger
patches and between patches, self-attention learning is implemented
again based on the previous level. Speciﬁcally, the intra-patch fea[
]
tures 𝐼 = 𝐼0 , 𝐼1 , … , 𝐼𝑖 , … , 𝐼𝑘 obtained from the previous level are regrouped by a factor of two to acquire larger patches. These patches are
concatenated with the embedding token 𝑥cls , which is deﬁned as:
(
(
))
𝐶𝑖′ = 𝑐𝑎𝑡 𝑥𝑐𝑙𝑠 , 𝑐𝑎𝑡 𝐼𝑖 , 𝐼𝑖+1
(6)

where 𝐷𝐌 and 𝑆𝐌 represent the Mahalanobis distance and Cosine similarity, respectively. 𝛾 and 𝜂 (𝛾, 𝜂 ≥ 0) are the related matrices weights.
The ﬁnal similarity model is determined from two perspectives: patch
distance and similarity.
To compensate for the limitations of linear transformations, we
extend the two linear projections {𝐔𝑥, 𝐔𝑦} to aﬃne transformations
{𝐀 𝑥 + 𝐚, 𝐀 𝑦 + 𝐛}. The aﬃne Mahalanobis distance 𝐷𝐌 in Eq. (8) can
be expressed as:
) (
)‖2
‖(
𝐷𝐌 = ‖ 𝐀 𝑥𝑖 + 𝐚 − 𝐁 𝑦𝑖 + 𝐛 ‖
‖
‖2
𝑥𝑖 ⎤
(9)
⎡
[
]
= 𝑥𝑇𝑖
𝑦𝑇𝑖
1 𝔻𝑀 ⎢ 𝑦𝑖 ⎥
⎢
⎥
⎣ 1 ⎦

Then, 𝐶𝑖′ is input into the same self-attention process described in
Eqs. (4), (5) to extract the inter-patch feature 𝐼̃𝑖 .
Note that, due to the information aggregation ability of tokens, the
ﬁnal encoded features are denoted by the token at the embedding position. The process can be expressed as follows:
[
]
𝐹𝑎 = 𝐼0 [∶, 0], 𝐼1 [∶, 0], … , 𝐼̃𝑀−1 [∶, 0], 𝐼̃𝑀 [∶, 0]
(7)

where ‖⋅‖22 represents the euclidean norm, which is the sum of the
squares of all elements in the current vector. The inputs for the similarity
calculation are the pose features 𝑦𝑖 and the patch features 𝑥𝑖 output by
MFE. To avoid introducing unnecessary scale changes in the initial stage,
the weight matrices 𝐀 and 𝐁 are initialized with random orthogonal
matrices, while the aﬃne transformation biases 𝐚 and 𝐛 are initialized
as the mean oﬀset of 𝑥𝑖 and 𝑦𝑖 . Thus, 𝔻𝑀 can be further expanded as:
(
)
𝐀𝑇 𝐀
−𝐀𝑇 𝐁
𝐀𝑇 𝐚 − 𝐛 ⎤
⎡
(
)
𝐁𝑇 𝐁
𝐁𝑇 𝐛 − 𝐚 ⎥
𝔻𝑀 = ⎢ −𝐁𝑇 𝐀
(10)
)
(
)
⎢( 𝑇
⎥
𝐛𝑇 − 𝐚𝑇 𝐁
‖𝐚 − 𝐛‖22 ⎦
⎣ 𝐚 − 𝐛𝑇 𝐀

where 𝐼𝑖 [∶, 0] represents the ﬁrst element in the second dimension, as
shown by the red square in Step 1 of Fig. 2. Due to the multi-level encoding process, the ﬁnal number of features is 𝑀, instead of 𝑁.
3.3. Pose-patch mapping
In real-world scenarios, the type, size, and position of occlusions result in varying occlusion situations across diﬀerent regions of the target person, which limits the positive impact of prior poses. Therefore,
we treat patch sequences and pose features as two diﬀerent modalities
{𝑥𝑖 , 𝑦𝑖 } and constrain the mapping process by calculating the similarity
matrix between poses and patches. To avoid the complex deformation
loss and mapping instability inherent in traditional linear similarity constraints, we introduce a new mapping algorithm based on mahalanobis
distance and cosine similarity. Compared to a single cosine similarity,

Similarly, the Cosine similarity calculation 𝑆𝐌 in Eq. (8) is expressed
using the aforementioned aﬃne transformation as:
⌢)
(⌢
⌢)𝑇 (⌢
𝑆𝐌 =  𝐀 𝑥𝑖 + 𝐚
 𝐁 𝑦𝑖 + 𝐛
=

4

[

𝑥𝑇𝑖

𝑦𝑇𝑖

1

]

⎡ 𝑥𝑖
𝕊𝑀 ⎢ 𝑦𝑖
⎢
⎣ 1

⎤
⎥
⎥
⎦

(11)

Pattern Recognition 172 (2026) 112341

D. Wang et al.
⌢

⌢

⌢

⌢

non-target pedestrians. Additionally, PPM relies on the accuracy of pose
localization and lacks the ability to perceive ﬁne-grained appearance
features. To address these problems, we develop a Texture-Aware De{
}
coder (TAD) for learning of the features 𝜌, 𝐹𝑠 , 𝐹𝑎 . As shown in Fig. 3,
diﬀerent from the traditional transformer architecture, our TAD comprises two types of layers: the context extraction layer is based on a
multi-head attention mechanism, which captures contextual dependencies of textures from diﬀerent perspectives through parallel multiple attention heads, and adaptively aggregates global semantic information.
The texture extraction layer utilizes a cross attention mechanism to establish a unidirectional guidance channel between texture and contextual features. By dynamically ﬁltering and enhancing key texture regions
based on contextual semantics and pose features, it enables the learning of more discriminative texture appearance features. The detailed
description is as follows.
On the left side of Fig. 3, the CELs have a multi-layer structure, which
(
( ) )
can be formally expressed as 𝐹𝑐𝑁 = 𝐶𝐸𝐿 ⋯ 𝐶𝐸𝐿 0 ⋯ . 0 is original sequence inputs in Eq. (1). Through multiple layers of CELs, local
and global dependencies in the sequence are calculated. And then, the
contextual features 𝐹𝑐𝑁 are used as the input of TELs to assist the mutual information attention from appearance to texture. Similar to CEL,
TEL also adopts a multi-layer structure, which can be represented as
(
(
) )
𝐹𝑡𝑖 = 𝑇 𝐸𝐿 𝜌, 𝐹𝑐𝑁 , ⋯ 𝑇 𝐸𝐿 𝜌, 𝐹𝑐𝑁 , 𝐹𝑠 ⋯ . The other two inputs, 𝜌 and 𝐹𝑠 ,
represent the learnable query prototype and the keypoint features after
PPM (in Eq. (18)), respectively. Speciﬁcally, 𝐹𝑠 is the initial input of
TEL as the initial value. After the CELs and TELs, TAD computes the
cosine similarity between the obtained texture-aware feature 𝐹𝑡 and the
patch appearance features 𝐹𝑎 to determine the corresponding feature
locations. TAD leverages the continuity of texture and contextual features to further augment the target features that are corrupted by other
pedestrian occlusions. The speciﬁc process can be formulated as:
( (
))
𝐼𝑛𝑑 = 𝜓𝑖𝑛𝑑 𝑆𝐌 𝐹𝑎 , 𝐹𝑡 ,
(19)

where  𝐀 𝑥𝑖 + 𝐚 and  𝐁 𝑦𝑖 + 𝐛 also belong to aﬃne transformation
{𝐴 𝑥 + 𝐚, 𝐁 𝑦 + 𝐛}, but they are described with diﬀerent symbols to dis⌢

⌢

tinguish the Mahalanobis distance. The weight matrices  𝐀 and  𝐁 are
initialized with random orthogonal matrices, while the aﬃne transfor⌢

⌢

mation biases 𝐚 and 𝐛 are initialized to 0 to eliminate absolute positional deviations between features. Furthermore, 𝕊𝑀 can be further
mathematically expanded as follows:
⎡
⎢ 𝟎
⎢ ⌢𝑇 ⌢
𝕊𝑀 = ⎢  𝐁  𝐀
⎢ ⌢𝑇2⌢
⎢ 𝐛 𝐀
⎣ 2

⌢𝑇 ⌢

𝐀𝐁
2

𝟎
⌢𝑇 ⌢

𝐚 𝐁
2

⌢𝑇 ⌢

𝐀 𝐛 ⎤
2 ⎥

⌢𝑇 ⌢ ⎥

𝐁 𝐚 ⎥
2 ⎥
⌢𝑇 ⌢⎥

(12)

𝐚 𝐛⎦

Therefore, we can derive the mathematical expression for 𝕊 in Eq. (8)
based on the Mahalanobis distance 𝐷𝐌 and Cosine similarity 𝑆𝐌 .
(
)
𝑆̃ 𝑥𝑖 , 𝑦𝑖 =𝛾𝐷𝐌 − 𝜂𝑆𝐌
𝐀
[
]⎡
= 𝑥𝑇𝑖 𝑦𝑇𝑖 1 ⎢ 𝐂𝑇
⎢ 𝑇
⎣ 𝐝

⎤⎡ 𝑥𝑖 ⎤
⎥⎢ 𝑦 ⎥
⎥⎢ 𝑖 ⎥
⎦⎣ 1 ⎦
(
)𝑇 (
)
2
2
𝑇
‖
‖
‖
‖
=‖𝐀𝑥𝑖 ‖ + ‖𝐁𝑦𝑖 ‖ + 2𝐝 𝑥𝑖 − 2 𝐂𝑥𝑖
𝐂𝑦𝑖 + 2𝐞𝑇 𝑦𝑖 + 𝑓
𝐂
𝐁
𝐞𝑇

𝐝
𝐞
𝑓

(13)

where
𝐀 = 𝛾𝐀𝑇 𝐀 ,
𝐁 = 𝛾𝐁𝑇 𝐁 ,
𝐂 = −𝛾𝐀𝑇 𝐁 − 𝜂

⌢𝑇 ⌢

𝐀𝐁
,
2
⌢𝑇 ⌢

𝐝 = 𝛾𝐀𝑇 (𝐚 − 𝐛) − 𝜂

𝐀 𝐛
,
2

(14)

⌢𝑇 ⌢

𝐁 𝐚
,
2
⌢
𝑇
⌢

𝐞 = 𝛾𝐁𝑇 (𝐛 − 𝐚) − 𝜂

𝐹𝑑𝑒 = 𝐹𝑎 [𝑖][𝐼𝑛𝑑[𝑖]] + 𝐹𝑠 [𝑖][𝑗].

𝑓 = 𝛾‖𝐚 − 𝐛‖22 − 𝜂 𝐚 𝐛 .

where 𝜓𝑖𝑛𝑑 denotes the index of the maximum value in the current matrix. 𝑆𝐌 represents the cosine similarity calculation. 𝑖 and 𝑗 represent
the batch index and patch index, respectively. 𝐹𝑑𝑒 is the decoding result
after texture augmentation. Next, we provide a detailed introduction to
the functions and implementation of CEL and TEL.

In the above formulas, 𝐀, 𝐁, 𝐂, 𝐝, 𝐞 and 𝑓 represent the parameters of the general mapping model. If there exists a certain set of
{𝐀, 𝐁, 𝐂, 𝐝, 𝐞, 𝑓 }, it means that there is at least one set of solutions 𝐷𝐌
and 𝑆𝐌 can represent the mapping matrix. That is, 𝑆𝐌 can be guaranteed to be decomposed by weighted Mahalanobis distance and Cosine
similarity.
(
)
As shown in Step 2 of Fig. 2, based on the mapping weights 𝑆̃ 𝑥𝑖 , 𝑦𝑖
{
}
in Eq. (8) of 𝑥𝑖 → 𝑦𝑖 , the 𝑖th activation features 𝑓𝑠𝑖 from pose 𝑦𝑖 to
patch 𝑥𝑖 can be calculated as:
(
(
))
𝑘 = 𝑎𝑟𝑔𝑚𝑖𝑛 1 − 𝓁 𝑆̃ 𝑥𝑖 , 𝑦𝑖 ,
(15)

3.4.1. Context extraction layer
In detail, the structure of the 𝑖th CEL is depicted in the top right
of Fig. 3. It performs Multi-Head Self-Attention (MHSA) computations
using residual connections. The calculation process is expressed as:
(
(
))
̃ 𝑖0 = 𝐼𝑁 𝑖−1
+ 𝑀𝐻𝑆𝐴 𝑖−1
, 𝑖−1
, 𝑖−1
(21)
0
0
0
0

𝑖

𝑓𝑠𝑖 = 𝐱 + 𝐲𝑘 .

(20)

(16)

where 𝑖−1
is the (𝑖 − 1)th context-extraction feature. 𝐼𝑁(⋅) represent
0
the instance normalization [21]. Then, 𝑖−1
ﬂows into the Multi-Layer
0
Perceptron (MLP) to capture more complex feature information.
(
( ))
𝐹𝑐𝑁 = 𝐼𝑁 ̃ 𝑖0 + 𝑀𝐿𝑃 ̃ 𝑖0
(22)

where 𝑘 is the position index with the highest similarity. {𝐱, 𝐲} represents the feature vectors of all {𝑥𝑖 , 𝑦𝑖 } pairs. 𝓁 represents the initialization heatmap conﬁdence after normalizing the pose features, and its
calculation process is as follows:
{
0 𝑐𝑖 < 𝜍
𝓁=
(𝑖 = 1, … , 𝑀).
(17)
1 𝑐𝑖 ≥ 𝜍

After 𝑁 layers of CELs, the contextual features 𝐹𝑐𝑁 can be output and
fed into each TEL for disentangled learning from appearance to texture.

where 𝑐𝑖 denotes the conﬁdence score of 𝑖th landmark. 𝜍 is the threshold
for ﬁltering high-conﬁdence and low-conﬁdence landmarks. Ultimately,
the keypoint features 𝐹𝑠 with high conﬁdence 𝓁, which have undergone
similarity mapping, are discribed as:
[
]
𝐹𝑠 = 𝑓𝑠1 , 𝑓𝑠2 , … , 𝑓𝑠𝑀 .
(18)

3.4.2. Texture extraction layer
To be speciﬁc, the structure of the 𝑖th TEL is shown in the bottom
right of Fig. 3. Firstly, TELs also relies on a residual structure and applies MHSA to capture key information in the features 𝐹𝑡𝑖−1 , where 𝑖 − 1
denotes the output of the (𝑖 − 1)th layer of TEL.
(
(
))
𝐹̃𝑡𝑖 = 𝐼𝑁 𝐹𝑡𝑖−1 + 𝑀𝐻𝑆𝐴 𝐹𝑡𝑖−1 , 𝐹𝑡𝑖−1 , 𝐹𝑡𝑖−1
(23)

3.4. Texture-aware decoder

Then, 𝜌, 𝐹𝑐𝑁 , and 𝐹̃𝑡𝑖 are input into Multi-Head Cross-Attention
(MHCA) to establish the correlation of three features. Speciﬁcally, we
use 𝜌 as the queries and 𝐹𝑐𝑁 as the keys to compute the relationships

The above process addresses object occlusion in person re-ID; however, it remains inadequate in addressing retrieval challenges caused by
5

Pattern Recognition 172 (2026) 112341

D. Wang et al.

Fig. 3. Structure of Texture-Aware Decoder (TAD). On the left is the overall framework of TAD, and on the right are the details of 𝑖th CEL and TEL.

mapping between pixels. Subsequently, with the assistance of pose mapping, 𝐹̃𝑡𝑖 is used as the value input into MHCA to achieve texture reﬁnement of 𝐹̂ 𝑖 .

deﬁned as:
𝑀
( ( ))
( )
( ))
1 ∑ ( ( ( 𝑖 ))
𝑑𝑒 = 𝑖𝑑  𝐹𝑑𝑒 + 𝑡𝑟𝑖 𝐹𝑑𝑒 +
  𝑓𝑠 + 𝑡𝑟𝑖 𝑓𝑠𝑖
𝑀 𝑖=1 𝑖𝑑

𝑡

(
(
))
𝐹̂𝑡𝑖 = 𝐼𝑁 𝐹̃𝑡𝑖 + 𝑀𝐻𝐶𝐴 𝜌, 𝐹𝑐𝑁 , 𝐹̃𝑡𝑖

(24)

(28)

In this way, 𝐹̂𝑡𝑖 gradually carries more texture information, promoting TELs to generate accurate and detailed texture features. Similarly,
at the end of 𝑖th TEL, there is a multi-layer perceptron, and the ﬁnal
texture capture process is as follows:
(
( ))
𝐹𝑡𝑖 = 𝐼𝑁 𝐹̂𝑡𝑖 + 𝑀𝐿𝑃 𝐹̂𝑡𝑖
(25)

where 𝑓𝑠𝑖 and 𝐹𝑑𝑒 indicate the pose-patch mapped feature (in Eq. (16))
and texture augmented feature (in Eq. (20)), respectively.
Pose Loss: Due to the accuracy of pose mapping aﬀects the optimization of the whole model, we add the pose loss constraint to the total
loss function. 𝑝 is introduced as one of the key factors to measure the
prediction error of the model, and it is deﬁned as follows:
⟨ 𝑖 𝑖⟩
𝐵
1 ∑ 𝑓𝑠 , 𝑓𝑙
𝑝 =
(29)
‖
𝐵 𝑖 ‖𝑓 𝑖 ‖‖
𝑓𝑙𝑖 ‖
‖ 𝑠 ‖‖
‖ ‖

After 𝑁 layers of TELs, the texture-aware feature 𝐹𝑡 is collected. In
Eq. (19), 𝐹𝑡 undergoes cosine similarity calculations (Eq. (19)) to locate
precise and ﬁne-grained textures.

where 𝐵 represents the batch size of training. ⟨⋅, ⋅⟩ indicates the inner
product. 𝑓𝑙𝑖 is the features of non-human body parts after pose mapping.
The motivation behind 𝑝 is that human body parts and non-human
body parts should not exhibit high similarity. If the similarity between
𝑓𝑠𝑖 and 𝑓𝑙𝑖 is high, 𝑝 increases, thereby constraining a more reasonable
mapping of the pose.

3.5. Loss function
The overall loss function is composed of the encoder loss, decoder
loss, and pose loss. In the training stage, 𝑡𝑜𝑡𝑎𝑙 jointly optimizes the
occlusion-aware and non-occlusion-aware of TTPM model, as described
in detail below:
𝑡𝑜𝑡𝑎𝑙 = 𝑒𝑛 + 𝑑𝑒 + 𝑝


codex
TTPM 的页面排版导致方法文本顺序有些乱，刚才截取漏了 MFE 的前半段。我会再补一次该论文从 3.1 到 3.3 的开头，避免把 pose-patch mapping 的逻辑链断掉。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' - | sed -n '174,400p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3.1. Overview

Body Part-Based Representation Learning ReID (BPBReID), which predicts part attention maps and generates body part features to leverage
identity and prior human topology information. Yan et al. [14] fully
leverage partial correlations to aggregate discriminative information between part features and enhance their signiﬁcance by suppressing occluded category features. Wang et al. [15] introduce jointly learning
part features and predicting part quality scores, automatically assigning
low scores to occluded body parts. Nevertheless, part-based methods
may not ensure the consistency of image matching in cases of severe
occlusion.
External clue-based methods mainly locate the non-occluded areas
of the human body through pose estimation or human parsing. PVPM
[2] utilizes pose information to achieve visibility matching of person
images and jointly learned discriminative features through attention to
mine the visibility of certain regions. Li et al. [3] propose a part-guided
representation method consisting of pose-invariant features and local
description features to infer pose-invariant representation through pose
estimation and normalization. Yuan et al. [11] employed a pose aggregation module to guide the learning of visible body parts and suppress occlusion noise by matching non-occluded regions based on pose-aligned
features. These methods can more accurately locate human body regions with the help of extra clues, but they cannot solve the problem of
occlusion caused by multiple pedestrians.

The overall network architecture of the proposed TTPM is outlined
in Fig. 2. Given a person image 𝑋 ∈ ℝ𝐻×𝑊 ×𝐶 , where 𝐻, 𝑊 , and 𝐶 are
the height, width, and channel dimensions of 𝑋, respectively, we feed
it into both the pose estimator HRnet [20] and the Vit [10] branches
to localize the target person accurately. In the pose estimation branch,
the estimator extracts 𝑀 landmarks from the input image 𝑋, each of
which corresponds to a keypoint heatmap {𝑦𝑖 |𝑖 = 1, 2, … , 𝑀 }. In the
Vit branch, the image 𝑋 is partitioned into 𝑁 non-overlapping patches
{𝑥𝑖 |𝑖 = 1, 2, … , 𝑁 }, which are mapped to 𝐷 dimensions through the linear projection function  (⋅) to generate a sequence of inputs 0 . Additionally, a learnable embedding token 𝑥𝑐𝑙𝑠 is added to the input sequence
to capture global representations of each patch. Similarly, following He
[17], we also incorporate learnable position embeddings  ∈ ℝ(𝑁+1)×𝐷
and camera viewpoint  ∈ ℝ(𝑁+1)×𝐷 as the crucial weights for the input sequence. The initial input sequence for the transformer encoder is
represented as:
[
( ) ( )
( )]
0 = 𝑥cls ;  𝑥1 ;  𝑥2 ; ⋯ ;  𝑥𝑁 +  + 𝜆
(1)
where 0 is the initial sequence input to transformer layer, and 𝜆 is a
hyperparameter used to balance the camera embedding.
Subsequently, the patch sequence 0 is fed into the MFE, which
encodes features at two levels: intra-patch and inter-patch, to explore
the relationships between diﬀerent body parts. The pose features 𝑦𝑖
and patch sequences 0 are input separately into PPM, constructing a
similarity mapping between poses and patches to avoid coarse-grained
interference under global mapping. Further details are available in
Sections 3.2 and 3.3.
Since transformer-based encoding and pose mapping focus more on
deep feature extraction and precise matching, they fail to address the
retrieval challenges caused by occlusions from non-target pedestrians.
To address this issue, we introduce a texture-aware decoder consisting
of a texture-extraction layer and a context-extraction layer. The former
is designed to capture the contextual feature of the original sequence
0 , while the latter applies cross-attention between the contextual features and patch features to extract distinctive textures. The structure of
TAD is presented in Section 3.4. To improve the re-ID accuracy of our
method, we employ encoder loss, decoder loss, and pose loss as supervision, which are introduced in Section 3.5.

2.2. Transformer-based person Re-ID
Transformer is a deep learning network widely known for its use of
attention mechanisms. Inspired by its excellent performance in natural
language processing, several works have attempted to apply it to computer vision. Carion et al. [16] ﬁrst introduce transformer into vision
tasks, proposing a Detection Transformer (DETR) model, which regards
object detection as a direct set prediction problem. Importantly, Dosovit
et al. [10] creatively present Vision Transformer (Vit) to process images
into patch sequences and demonstrate superior performance in traditional vision tasks.
Recently, some researchers have started to explore the application
of transformer in occluded person re-ID. For example, TransReID [17]
applies a pure transformer architecture based on Vit for re-identiﬁcation
and uses side information embedding to learn camera view clues, helping to mitigate feature bias caused by diﬀerent camera perspectives.
Wang et al. [18] employ transformer to occlusion augmentation and
propose a feature completion decoder that aggregates potential information to generate occluded features. Li et al. [9] introduced a secondorder attention mechanism to model high-order semantic relationships
among diﬀerent class tokens, thereby enabling a more comprehensive
capture of structural dependencies between human body parts. Wang et
al. [19] present a transformer-based Pose-guided Feature Disentangling
(PFD) approach, which selectively matches non-occluded parts using
pose information and automatically separates occluded semantics from
non-occluded semantics. In PFD, pose information is directly embedded into both global contextual features and local features. Although a
learnable semantic view is introduced to implicitly enhance discriminative features, it still fails to eﬀectively map pose positions to human
semantics.
In the paper, we present a transformer architecture with explicit
pose-patch mapping and introduce a texture-aware decoder to realize
eﬀective separation of occluded and non-occluded human semantics.

3.2. Multi-patch feature encoder
Under the transformer architecture, patches often have suﬃcient
global perception information but lack exploration of multi-level features within the patches. Therefore, we construct a multi-level patch
encoding to obtain semantics within patches and the interrelationships
between patches. As shown in Step 1 of Fig. 2, it includes two levels:
the intra-patch feature encoding and the inter-patch feature encoding.
First, the input embeddings 0 are split into 𝑘 groups:
⌊ ( )⌋
𝑙𝑒𝑛 0
(
)
𝑧 = 𝑠𝑝𝑙𝑖𝑡 0 [1 ∶], 𝑙 , 𝑙 =
(2)
𝑘
where 𝑠𝑝𝑙𝑖𝑡(⋅, ⋅) denotes the segmentation function, 𝑙 is the split length,
0 [1 ∶] is the input sequence 0 with 𝑥cls removed, and 𝑙𝑒𝑛(⋅) represents
the length of the current feature.
At the intra-patch level, patch features from the sequence 𝑧 =
[
]
𝑧0 , 𝑧1 , … , 𝑧𝑖 , … , 𝑧𝑘 and the embedding token 𝑥cls are separately input
into independent transformer layers. After the patch features and embedding token are concatenated, they are ﬂattened into three vectors
𝑄, 𝐾, 𝑉 ∈ ℝ𝑁×𝑑 , where 𝑑 denotes the embedding dimension. The attention weight calculation process for intra-patch feature encoding is as
follows:
(
)
𝐶𝑖 = 𝑐𝑎𝑡 𝑥𝑐𝑙𝑠 , 𝑧𝑖 ,
(3)
(
(
( )))
′
𝐼𝑖 = 𝜌 𝑓 𝑙𝑎𝑡𝑡𝑒𝑛 𝑛𝑜𝑟𝑚 𝐶𝑖
+ 𝐶𝑖 ,
(4)

3. Proposed method
This section details the proposed Texture-Aware Transformer with
Pose-Patch Mapping (TTPM). We ﬁrst present an overview of TTPM in
Section 3.1 and then introduce the motivation, objectives, and implementation of Multi-patch Feature Encoder (MFE), Pose-Patch Mapping
(PPM), and Texture-Aware Decoder (TAD) from Sections 3.2 to 3.4.
3

Pattern Recognition 172 (2026) 112341

D. Wang et al.

Fig. 2. The framework of Texture-Aware Transformer with Pose-Patch Mapping (TTPM). TTPM consists of three steps, including Step 1: the Multi-patch Feature
Encoder, which mines features from intra-patch and inter-patch levels. Step 2: the Pose-Patch Mapping, which separates pose semantics and occlusion semantics
through similarity matrix constraints. Step 3: the Texture-Aware Decoder, which employs textures to augment the representation of discriminative appearances.

(
( ))
𝐼𝑖 = 𝜗 𝑛𝑜𝑟𝑚 𝐼𝑖′ + 𝐼𝑖′ .

it better measures the feature distance under diverse scales and distributions, alleviating the inﬂuence of diﬀerent feature scales. The generic
formula for similarity calculation is as follows:
(
)
𝑆̃ 𝑥𝑖 , 𝑦𝑖 = 𝛾𝐷𝐌 − 𝜂𝑆𝐌
(8)

(5)

where 𝑐𝑎𝑡(⋅, ⋅) indicates concatenation along a speciﬁc channel. 𝜌(⋅)
includes softmax, linear variation, and element-wise multiplication.
𝑓 𝑙𝑎𝑡𝑡𝑒𝑛(⋅) denotes the ﬂattening operation. 𝑛𝑜𝑟𝑚(⋅) represents layer normalization. 𝜗(⋅) comprises mlp, gelu, and dropout operations.
At the inter-patch level, to further explore features within larger
patches and between patches, self-attention learning is implemented
again based on the previous level. Speciﬁcally, the intra-patch fea[
]
tures 𝐼 = 𝐼0 , 𝐼1 , … , 𝐼𝑖 , … , 𝐼𝑘 obtained from the previous level are regrouped by a factor of two to acquire larger patches. These patches are
concatenated with the embedding token 𝑥cls , which is deﬁned as:
(
(
))
𝐶𝑖′ = 𝑐𝑎𝑡 𝑥𝑐𝑙𝑠 , 𝑐𝑎𝑡 𝐼𝑖 , 𝐼𝑖+1
(6)

where 𝐷𝐌 and 𝑆𝐌 represent the Mahalanobis distance and Cosine similarity, respectively. 𝛾 and 𝜂 (𝛾, 𝜂 ≥ 0) are the related matrices weights.
The ﬁnal similarity model is determined from two perspectives: patch
distance and similarity.
To compensate for the limitations of linear transformations, we
extend the two linear projections {𝐔𝑥, 𝐔𝑦} to aﬃne transformations
{𝐀 𝑥 + 𝐚, 𝐀 𝑦 + 𝐛}. The aﬃne Mahalanobis distance 𝐷𝐌 in Eq. (8) can
be expressed as:
) (
)‖2
‖(
𝐷𝐌 = ‖ 𝐀 𝑥𝑖 + 𝐚 − 𝐁 𝑦𝑖 + 𝐛 ‖
‖
‖2
𝑥𝑖 ⎤
(9)
⎡
[
]
= 𝑥𝑇𝑖
𝑦𝑇𝑖
1 𝔻𝑀 ⎢ 𝑦𝑖 ⎥
⎢
⎥
⎣ 1 ⎦

Then, 𝐶𝑖′ is input into the same self-attention process described in
Eqs. (4), (5) to extract the inter-patch feature 𝐼̃𝑖 .
Note that, due to the information aggregation ability of tokens, the
ﬁnal encoded features are denoted by the token at the embedding position. The process can be expressed as follows:
[
]
𝐹𝑎 = 𝐼0 [∶, 0], 𝐼1 [∶, 0], … , 𝐼̃𝑀−1 [∶, 0], 𝐼̃𝑀 [∶, 0]
(7)

where ‖⋅‖22 represents the euclidean norm, which is the sum of the
squares of all elements in the current vector. The inputs for the similarity
calculation are the pose features 𝑦𝑖 and the patch features 𝑥𝑖 output by
MFE. To avoid introducing unnecessary scale changes in the initial stage,
the weight matrices 𝐀 and 𝐁 are initialized with random orthogonal
matrices, while the aﬃne transformation biases 𝐚 and 𝐛 are initialized
as the mean oﬀset of 𝑥𝑖 and 𝑦𝑖 . Thus, 𝔻𝑀 can be further expanded as:
(
)
𝐀𝑇 𝐀
−𝐀𝑇 𝐁
𝐀𝑇 𝐚 − 𝐛 ⎤
⎡
(
)
𝐁𝑇 𝐁
𝐁𝑇 𝐛 − 𝐚 ⎥
𝔻𝑀 = ⎢ −𝐁𝑇 𝐀
(10)
)
(
)
⎢( 𝑇
⎥
𝐛𝑇 − 𝐚𝑇 𝐁
‖𝐚 − 𝐛‖22 ⎦
⎣ 𝐚 − 𝐛𝑇 𝐀

where 𝐼𝑖 [∶, 0] represents the ﬁrst element in the second dimension, as
shown by the red square in Step 1 of Fig. 2. Due to the multi-level encoding process, the ﬁnal number of features is 𝑀, instead of 𝑁.
3.3. Pose-patch mapping

codex
证据闭环需要看实验消融，不只是方法描述。SFE、CMBL、PSSR、TTPM 的第一页到第八页已经含关键消融；SD-3DGF 和 TAP-align 的消融在后面，我会追加抽实验页来补足“为什么这个重定义成立”。
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
/bin/zsh -lc "pdfinfo 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' | sed -n '1,40p' && pdfinfo 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' | sed -n '1,40p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos
Subject:         IEEE Transactions on Circuits and Systems for Video Technology; ;PP;99;10.1109/TCSVT.2026.3691145
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.27; modified using iText® 7.1.12 ©2000-2020 iText Group NV (AGPL-version); modified using iText® Core 7.2.4 (AGPL version) ©2000-2022 iText Group NV
CreationDate:    Tue May  5 13:04:29 2026 CST
ModDate:         Thu May  7 06:25:56 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           16
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       12147166 bytes
Optimized:       no
PDF version:     1.5
Title:           Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification
Subject:         -  Computing methodologies  ->  Matching;Visual content-based indexing and retrieval;
Creator:         LaTeX with acmart 2025/08/27 v2.16 Typesetting articles for the Association for Computing Machinery and hyperref 2024-10-30 v7.01k Hypertext links for LaTeX
Producer:        LuaHBTeX, Version 1.18.1 (MiKTeX 24.4)
CreationDate:    Thu Mar 19 17:12:06 2026 CST
ModDate:         Thu Mar 19 17:12:06 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           18
Encrypted:       no
Page size:       486 x 720 pts
Page rot:        0
File size:       22113663 bytes
Optimized:       no
PDF version:     2.0

exec
/bin/zsh -lc "pdftotext -f 8 -l 14 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | rg -n \"(Ablation|SD-3DGF|MRIE|ECS|local similarity|LSL|appearance|gait|clothes-changing|Table|Fig\\.|Consistency|without|w/o|Impact|Effect|Contribution|ablation|representative|VCCR|CCVID|CCV|MARS|Rank|mAP)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
6:should remain consistent. The 3D gait field loss Lgf satisfying
20:and the 3D gait field loss Lgf , leading to the overall objective:
60:3) Re-ID Loss: For the video-based appearance feature
98:We evaluate our method on four VCCRe-ID datasets, i.e.,
99:VCCR [47], CCVID [45], CCV-S [46], and CCV-R [46].
100:Among them, VCCR, CCVID, and CCV-R are real-world
101:datasets, and CCV-S is a synthetic dataset. All datasets are
103:and the clothes-changing setting (CC). Under the standard
105:covering both clothes-consistent and clothes-changing cases,
106:whereas under the clothes-changing setting (CC), gallery samples that share both the same clothing and identity labels with
108:ability of the proposed method, we conduct additional evaluations on three widely used video-based person Re-ID benchmarks, i.e., MARS [26], LS-VID [66], and iLIDS-VID [67],
110:of the Re-ID model using the Rank-1 accuracy of the CMC
111:curve and mean average precision (mAP). In addition, we
112:introduce the Embedding Consistency Score (ECS) as a new
115:of each dataset involved in the experiment. ECS is defined as:
116:ECS =
149:compute ECS.
179:C OMPARATIVE R ESULTS (%) ON VCCR, CCVID, CCV-R, AND CCV-S († C ODE R EPRODUCED BY U S )
180:VCCR
187:CCVID
193:CCV-R
199:CCV-S
207:Rank-1
209:mAP
211:Rank-1
213:mAP
215:Rank-1
217:mAP
219:Rank-1
221:mAP
223:Rank-1
225:mAP
227:Rank-1
229:mAP
231:Rank-1
233:mAP
235:Rank-1
492:SD-3DGF(Ours)
625:MARS, LS-VID, and iLIDS-VID, where no clothing labels are
633:as frame-level appearance features, with the subsequent layers
635:video-level appearance features have a size of 8×16×8×1024.
636:In local similarity learning (LSL), the neighbor size K is
638:boundary of the gait field tensor F, the features of missing
640:calculation. In the process of dividing the gait and appearance
645:extractor fθ continues to be trained to optimize the 3D gait
659:mAP
662:C OMPARATIVE R ESULTS (%) ON MARS, LS-VID, AND I LIDS-VID (†
674:Rank-1
676:mAP
678:Rank-1
680:mAP
692:Rank-1
744:SD-3DGF(Ours)
787:and video-based clothes-changing Re-ID methods, such as
792:solely on the appearance stream of our model, i.e., the identity
793:representation is computed from the appearance features XA
795:1) Results on VCCRe-ID Datasets: As shown in Tab. I,
796:under the clothes-changing setting, our proposed SD-3DGF
798:VCCR, CCVID, CCV-R, and CCV-S datasets, achieving Rank1 improvements of 4.2%, 0.6%, 7.7%, and 3.7%, and mAP
800:image-based clothes-changing Re-ID methods perform poorly
801:on VCCRe-ID due to neglecting temporal information necessary for discriminative identity cues, video-based Re-ID
803:on appearance features, which limits their robustness and
817:A BLATION S TUDY OF K EY C OMPONENTS IN THE P ROPOSED SD-3DGF F RAMEWORK UNDER THE C LOTHES -C HANGING S ETTING (%)
824:MRIE
833:LSL
875:VCCR (CC)
876:Rank-1
887:mAP
898:CCVID (CC)
899:Rank-1
910:mAP
921:CCV-R (CC)
922:Rank-1
935:CCV-S (CC)
937:mAP
948:Rank-1
959:mAP
972:Fig. 5. Influences of the hyperparameters D, w2 , and η. The x-axis denotes the hyperparameter values, while the y-axis shows the Rank-1 and mAP
973:performance under the clothes-changing setting on the CCVID dataset.
977:VCCR E -ID M ETHODS WITH M ESH -BASED R EPRESENTATIONS AT
983:CCVID (CC)
985:CCV-R (CC)
987:CCV-S (CC)
989:Rank-1
991:mAP
993:Rank-1
995:mAP
997:Rank-1
999:mAP
1001:Rank-1
1003:mAP
1128:SD-3DGF (Ours)
1138:VCCR (CC)
1141:SD-3DGF (w/o. Lorg & Lgf )
1142:SD-3DGF (w/o. Lorg )
1143:SD-3DGF (w/o. Lgf )
1144:SD-3DGF (Full)
1150:CCVID (CC)
1152:CCV-R (CC)
1154:CCV-S (CC)
1156:Rank-1
1158:mAP
1160:Rank-1
1162:mAP
1164:Rank-1
1166:mAP
1168:Rank-1
1170:mAP
1212:VCCR (CC)
1216:correspondence embeddings for learning 3D gait features,
1218:D. Ablation Studies
1220:performance compared with VCCRe-ID methods.
1223:both clothes-changing and clothes-consistent samples, SD3DGF consistently outperforms all competitors on the VCCR,
1224:CCVID, CCV-R, and CCV-S datasets. Specifically, it achieves
1225:gains of 1.1%, 1.3%, 0.1%, and 2.8% in Rank-1 accuracy, and
1226:1.1%, 0.8%, 6.9%, and 9.0% in mAP. Furthermore, as shown
1228:MARS, LS-VID, and iLIDS-VID, where pedestrians undergo
1229:little or no clothing changes, most VCCRe-ID methods exhibit
1233:In this subsection, we conduct systematic ablation studies
1234:on VCCR, CCVID, CCV-R and CCV-S to investigate: (1) the
1237:(3) the effects of the 3D gait field loss and the orthogonal
1241:1) Effectiveness of GS, LSL, and MRIE: Tab. III reports
1243:computational complexity, and performance on VCCRe-ID
1244:datasets. For clarity, the notations used in Tab. III are defined as follows. AS and GS denote the appearance and
1245:gait streams, respectively. LSL stands for local similarity
1246:learning. MRIE refers to the Modality-Representative Inter-
1261:Rank-1 Retrieval Results
1263:VCCR
1265:CCVID
1267:CCV-R
1269:CCV-S
1271:Fig. 6. Visualization of the learned dense correspondence embeddings on VCCR, CCVID, CCV-R, and CCV-S, obtained via Principal Component Analysis
1279:corresponds to the Gait-Appearance Joint Encoding submodule in MRIE. When no representative token is generated,
1282:It is clear that single-stream models using only the appearance stream (Baseline) or the gait stream (Settings 2
1283:and 3) yield suboptimal performance. Adding LSL to the
1284:gait stream (Setting 2 → 3, 8 → 9) strengthens 3D gait
1286:only 3D gait fields from Eff-UNet [60] with average pooling for classification, whereas Setting 3 corresponds to the
1287:full gait stream (GS-F) with the subsequent convolutional
1288:blocks and LSL. We observe that the full gait stream (GSF) consistently outperforms the appearance stream on the four
1289:VCCRe-ID datasets. While simple concatenation of the two
1290:streams without feature interaction (Setting 4) brings only
1291:minor improvement, incorporating MRIE (Setting 4 → 9)
1292:leads to notable gains: Rank-1 improves by 8.8%, 10.7%,
1293:16.3%, and 7.5%, and mAP by 7.2%, 13.1%, 13.6%, and
1294:10.2% on VCCR, CCVID, CCV-R, and CCV-S, respectively,
1295:demonstrating MRIE’s effectiveness for core Re-ID modeling.
1297:in MRIE, we additionally introduce three control settings:
1298:Setting 5, 6, and 7. Generating global and local representative
1302:without performance loss. Replacing token-level pooling with
1305:computational burden, while GAJE makes the largest contribution to MRIE’s performance.
1307:on the results in Tab. III, the appearance stream, the full gait
1308:stream (GS-F), and the MRIE module have 10.7M, 16.6M, and
1310:13.3, 18.2, and 10.3 GFLOPs. Within the gait stream, the EffUNet-based correspondence embedding extractor adds only
1311:4.6M parameters and 3.5 GFLOPs, while the LSL module
1312:incurs 1.5M parameters and 1.2 GFLOPs. Most of the additional parameters and computational cost in the gait stream
1314:MRIE, DGRT requires 13.9M parameters and 8.5 GFLOPs,
1317:tokens, DGRT reduces 15.5 GFLOPs by generating dualgranularity representative tokens, which compress the spatial
1321:VCCRe-ID approaches in terms of storage and computational
1325:single appearance stream (Baseline), our method remains the
1347:CCVID (CC)
1349:CCV-R (CC)
1351:CCV-S (CC)
1353:Rank-1
1355:mAP
1357:Rank-1
1359:mAP
1361:Rank-1
1363:mAP
1365:Rank-1
1367:mAP
1441:SD-3DGF (Mask Face)
1442:SD-3DGF (Mask Lower Limbs)
1443:SD-3DGF (Full)
1479:VCCR (CC)
1483:CCVID (CC)
1485:CCV-R (CC)
1487:CCV-S (CC)
1489:Rank-1
1491:mAP
1493:Rank-1
1495:mAP
1497:Rank-1
1499:mAP
1501:Rank-1
1503:mAP
1613:SD-3DGF (Unperturbed)
1632:ECS ON THE T EST S ETS OF D IFFERENT V IDEO -BASED
1633:R E -ID/VCCR E -ID DATASETS AND THE G AIT S TREAM E VALUATION (%)
1637:Alignment Quality (ECS)
1649:Rank-1
1651:mAP
1684:VCCR
1685:CCVID
1686:CCV-R
1687:CCV-S
1715:3) Effects of the 3D Gait Field Loss and the Orthogonal
1716:Loss: To further examine the influence of the 3D gait field
1718:training stage, we perform ablation experiments by selectively
1723:larger performance degradation, reflecting that the appearance
1731:the gait stream. Keeping all other components of the model
1741:observed that using dense correspondence embeddings for gait
1744:VCCR (CC)
1753:the consistency constraints during 3D gait field training; w1
1756:relative contribution of the Re-ID loss and the 3D gait field
1757:loss. From Fig. 5(a), we observe that Rank-1 and mAP on
1758:the CCVID dataset peak when the embedding dimension D
1765:optimal values. Fig. 5(b) and (c) illustrate the effects of w2
1779:evaluated on multiple VCCRe-ID benchmarks. Specifically,
1838:Fig. 7. Visualization of the learned dense correspondence embeddings under
1839:challenging conditions in the CCV-R dataset.
1841:Fig. 8. t-SNE visualization of SMPL parameters, appearance features, 3D
1842:gait features, and fused ID features for 20 randomly selected identities from
1843:the CCV-R dataset.
1851:a 10% perturbation to the camera parameters leads to Rank-1
1852:drops of 9.9%, 11.6%, 21.5%, and 9.7%, and mAP drops of
1853:8.0%, 14.2%, 17.3%, and 14.9% on VCCR, CCVID, CCV-R,
1854:and CCV-S, respectively. This sensitivity arises because large
1863:and characteristics. For example, CCVID contains simple
1864:scenes with few occlusions, VCCR and CCV-R feature heavy
1865:occlusions and low resolution, and CCV-S consists entirely of
1867:embedding reliability on gait stream performance, we evaluate
1868:these embeddings on the test sets of both standard videobased Re-ID and VCCRe-ID datasets. Tab. VIII reports the
1869:Embedding Consistency Score (ECS) for the query, gallery,
1870:and full test sets, where ECS, as defined in Eq. 32, quantifies
1874:each dataset, we evaluate the performance of the full gait
1876:correlation between gait stream performance and embedding
1878:ECS, such as CCV-S and CCV-R, the gait stream maintains
1882:gait representations under challenging conditions.
1887:the appearance and gait streams, we conduct region-masking
1888:experiments on the single appearance stream (Baseline), the
1889:full gait stream (GS-F), and the overall model.
1890:According to Tab. IX, masking the face region during inference nearly disables the appearance stream, while masking
1891:the lower limbs significantly degrades the full gait stream
1893:This indicates that the gait stream is not solely dependent on
1906:Visualization results on VCCR, CCVID, CCV-R, and CCVS are shown in Fig. 6. The embeddings exhibit a smooth
1909:Fig. 7 presents visualizations on the CCV-R test set under
1930:Fig. 9. Class activation maps of the appearance stream and the gait stream,
1931:with the video tracklet sampled from the CCV-R dataset.
1938:We further employ t-SNE in Fig. 8 to visualize the feature
1939:distributions of 20 randomly selected identities from the CCVR dataset, including SMPL parameters, appearance features,
1940:3D gait features, and the fused ID features. The SMPL parameters appear largely disorganized and non-discriminative,
1942:and blurred inter-class boundaries, suggesting that appearance
1943:alone provides insufficient cues under the clothes-changing
1944:setting. In contrast, our 3D gait features form relatively more
1945:compact intra-class clusters than the appearance features, with
1947:remains limited. After fusing both streams via the MRIE
1949:class-wise aggregation and clearer inter-class separability, underscoring the strong complementarity between appearance
1950:and gait streams.
1952:activation maps for the appearance and gait streams in Fig. 9. It
1953:can be observed that the appearance stream primarily focuses
1954:on local regions less affected by clothing, reflecting the constraint imposed by the orthogonal loss, whereas the gait stream
1957:the gait stream primarily captures walking patterns, while also
1961:between the appearance and gait streams.
1963:In this paper, we introduce dense correspondence embeddings as a new representation to construct Structured Distilled 3D Gait Fields (SD-3DGF) for the VCCRe-ID task.
1964:Based on the constructed SD-3DGF, we show that extracting locally enhanced 3D gait features via local similarity
1967:(MRIE) module to enable efficient gait-appearance interaction
1968:and joint encoding. The MRIE module generates reducedsize dual-granularity representative tokens that act as a bridge
1976:VCCRe-ID task, but also generalizes well to standard videobased person Re-ID settings. A limitation of our method is that
2008:IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2015, pp. 1116–
2045:and Y. Qiao, “Cocas+: Large-scale clothes-changing person

exec
/bin/zsh -lc "pdftotext -f 8 -l 14 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | rg -n \"(Ablation|ablation|ATG|CTP|TRM|prototype|Prototype|attribute|contour|text|Text|SYSU|RegDB|Table|Figure|Fig\\.|Effect|w/o|without|modal|alignment|Rank|mAP|loss|TAP)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:and the identity classification loss is:
22:The TRM module consists of four components: a Text Encoder, a Visual Encoder, a Fusion
23:Network, and an Interaction Network. The Text Encoder and Visual Encoder adopt the standard
24:CLIP Text Encoder and Visual Encoder, respectively. The Fusion Network is constructed with three
30:training, we freeze the parameters of the visual encoder and text encoder and update only the
33:After convergence, the redefined feature 𝒇𝑖 serves as an enriched and interaction-aware textual
34:representation, effectively capturing both attribute-level and contour-level semantics for each image.
37:Text-Guided Attribute and Prototype Alignment Model
39:In this section, we present our proposed model, text-guided attribute and prototype alignment
40:(TAP-align). As illustrated in Figure 2, the proposed model adopts a dual-branch architecture
43:shallow structure, serving as modality-specific feature extractors, 𝐸 𝑣,𝑣 and 𝐸 𝑣,𝑖 , which extract
45:encoder 𝐸, which is responsible for learning modality-invariant visual representations.
46:The model is trained using a joint optimization strategy that combines the identity classification loss and the weighted regularized triplet loss. The identity loss supervises the learning of
47:discriminative identity features. In contrast, the weighted regularized triplet loss assigns adaptive weights to positive and negative sample pairs, thereby enhancing the model’s capability to
48:extract modality-invariant features. This joint learning strategy proves particularly effective in
49:cross-modality person ReID scenarios, where RGB and IR images exhibit significant appearance
51:To further improve cross-modality alignment, we explicitly mine potential modality-invariant
52:textual cues from images as semantic guidance. In particular, we introduce a Text Prototype
53:Discriminative Loss, which encourages visual features to align with class-level text prototypes that
55:consistency and discriminative capability in cross-modality matching. Finally, the overall joint
65:where identity loss L𝑖𝑑 , weighted regularized triplet loss L𝑤𝑟𝑡 , text prototype discriminative loss
82:Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID
86:Fig. 2. Our framework introduces a dual-branch text generation module to extract modality-invariant semantic
87:descriptions from pedestrian images. Specifically, an ATG constructs identity-relevant textual descriptions
88:based on predefined attribute templates. At the same time, a CTP module encodes structural contour cues at
89:the class level. These two complementary descriptions are fused by a TRM into unified textual prototypes,
92:Fig. 3. ATG module generates structured and modality-invariant pedestrian attribute descriptions using
94:for cross-modal recognition.
96:Here, 𝒇𝑖 can represent either 𝒇𝑖𝑣 or 𝒇𝑖𝑟 , which correspond to the features extracted by the modalityshared encoder 𝐸from RGB or IR images, respectively. 𝑛𝑏 denotes the batch size; 𝒒𝑖 is the one-hot
144:Fig. 4. CTP module learns identity-discriminative prompt tokens by leveraging human parsing (SCHP [11]) to
145:extract contour-relevant regions, enabling the generation of sketch-based textual representations for robust
146:cross-modal alignment.
148:Fig. 5. TRM merges attribute- and sketch-based texts with image features through fusion and transformer
149:interaction, producing discriminative, image-aware textual representations for each instance.
203:vectors, 𝒇𝑦𝑡𝑖 denotes the class-level textual prototype feature corresponding to the identity label 𝑦𝑖 ,
206:Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID 113:11
218:Here, 𝑛𝑞 denotes the number of textual features corresponding to the identity label 𝑦𝑖 .
223:We first construct rich and modality-invariant pedestrian semantic representations through a
224:Text Semantic Generation Module. This module generates two types of textual features: on the
225:one hand, a large language model is employed to produce attribute-descriptive texts based on
226:predefined attribute templates; on the other hand, a set of feature vectors characterizing pedestrian
227:contours is learned using the CoOp technique. Subsequently, we introduce a TRM, which integrates
228:the attribute-aware textual features with the contour-aware semantic features to form a newly
229:defined textual representation. Based on these redefined textual features, we compute class-wise
230:prototype representations, which serve as shared semantic cues to align the features extracted from
231:RGB and IR images via a dual-stream network toward the corresponding class prototypes. During
232:training, we jointly adopt the identity loss and the weighted regularized triplet loss to optimize the
233:modality-specific encoders 𝐸 𝑣,𝑣 and 𝐸 𝑣,𝑖 , the modality-shared encoder 𝐸, and the classifier 𝑾𝑐 . The
245:SYSU-MM01. This is currently one of the most demanding large-scale datasets designed specifically
246:for cross-modality person ReID, containing 2,87,628 visible images and 15,792 IR images from 491
256:RegDB. This is a compact VI-ReID dataset containing 8,240 images of 412 pedestrians. Each
257:individual has 10 visible and 10 IR images, all captured by a single camera for both modalities.
265:and mean average precision (mAP). The rank-K metric evaluates the percentage of successful
272:searches where at least one relevant individual is retrieved from the top-K outcomes. The mAP
276:as the backbone, employing two parallel shallow layers for modality-specific feature extraction,
277:while the remaining four shared deep convolutional blocks are used for modality-shared features. Input images are uniformly resized to 288 × 144, and common data augmentation strategies are applied, including random flipping, padding and cropping, channel augmentation, and
278:PartMix. In the first and second stages, we train the ATG Module and the CTP module, respectively, each for 60 epochs. The initial learning rate is set to 3 × 10−4 and decayed following a
279:cosine schedule. In the third stage, we train the TRM while freezing both the visual and text
292:datasets. As shown in Tables 1 and 2, we can observe the following: First, we compared several
293:traditional modality-invariant methods, such as AGW, CAJ, MPANET, DEEN, and SSRR. These
294:methods primarily rely on the relationships between images to extract modality-invariant features
295:without incorporating textual information from the images. We found that our method outperforms
297:the SYSU-MM01 dataset, our method improved the mAP by approximately 2.4% and the rank-1
300:improved the mAP by about 4.8% and the rank-1 score by approximately 5.8%. To further evaluate
301:the effectiveness and robustness of our method, we conducted tests on the RegDB dataset, comparing
303:method showed a decrease of approximately 3.3% and 5.0% in mAP, respectively. Additionally,
306:about 0.8%. These experimental results indicate that incorporating rich textual information can
308:modality-invariant features. Moreover, our method did not outperform other algorithms in certain
309:scenarios, suggesting that the introduced textual information might be subject to specific contextual
312:model-based methods, such as MIP, J-ReID, and CSDN. These methods leverage textual information
313:from images to mine modality-invariant features for identification. When compared to the bestperforming CSDN method, we found that on the SYSU-MM01 dataset, for both the ALL Search
315:and 0.7% in mAP and rank-1, respectively. On the RegDB dataset, for the Visible to IR and IR to
319:Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID 113:13
320:Table 1. Overall Performance Comparison on in SYSU-MM01
351:mAP
404:mAP
457:The performance is shown by rank-k accuracy (%) and mAP (%). The best results are in bold.
459:Table 2. Overall Performance Comparison on in RegDB
490:mAP
535:mAP
580:The performance is shown by rank-k accuracy (%) and mAP (%). The best results are in bold.
582:1.2% in mAP and rank-1, respectively. Overall, our method demonstrates superior performance
585:for constructing text prompts that better focus on pedestrian recognition information, further
586:enhancing performance. This suggests that richer semantic textual prompts can effectively guide
587:the visual encoder of large models, enabling it to better extract modality-invariant features of
589:Additionally, we observed from the table that some traditional modality-invariant methods,
590:such as MUN, perform better than the language model–based method MIP that introduces textual
591:prompts but still fall short compared to CSDN and J-ReID methods. We attribute this phenomenon
592:to the fact that the design of the textual prompts in the MIP method is not yet optimal, making it
595:cross-modal person ReID depends on generating effective textual prompts and designing suitable
602:Table 3. Ablation Study on SYSU-MM01
605:ATG CTP TRM
617:mAP
624:SYSU-MM01
642:Ablation Studies
645:of ablation experiments on SYSU-MM01. The rank-1, rank-5, and rank-10 mAP accuracies (%) are
646:reported in Table 3.
647:— Firstly, to further investigate the effectiveness of ATG, we compared the baseline method
648:with the baseline+ATG method. The ablation experiment in this section was implemented as
649:follows: we utilized the textual information generated from ATG to compute the category
650:prototypes of the text as alignment cues, guiding images from different modalities to align
651:towards these cues. As shown in the table, the textual information generated based on image
652:pedestrian recognition attributes serves as a new alignment cue, leading to better retrieval
653:performance compared to the baseline model. Specifically, mAP and rank-1 scores improved by
655:textual information based on image pedestrian attributes is effective.
656:— To further investigate the effectiveness of CTP, we conducted ablation studies by comparing
657:the baseline model with the baseline+CTP variant. In this configuration, the CTP module
658:extracts structural contour information from pedestrian images and generates Contour-aware
659:class-level textual features, which are employed as additional alignment cues to facilitate
660:modality-invariant feature learning. As reported in Table 3, the inclusion of CTP yields
661:consistent performance improvements, leading to gains of 0.88% in mAP and 0.99% in rank-1
662:accuracy. These results validate that contour-guided textual semantics provide meaningful
663:structural priors and enhance the cross-modal alignment capability of the model.
664:— To further validate the complementarity between semantic and structural alignment cues, we
665:evaluate the effectiveness of jointly applying the ATG and CTP modules. In this configuration,
666:class-level textual prototypes are independently generated by ATG and CTP, where ATG encodes high-level semantic attributes and CTP captures local structural contours of pedestrians.
667:These two sets of prototypes are averaged to obtain a unified textual representation, which
668:serves as a composite alignment reference across modalities. This simple yet effective fusion
669:integrates both semantic abstraction and structural detail, enabling more robust modalityinvariant feature learning. As presented in Table 3, the integration of ATG and CTP results
670:in further performance gains, yielding an mAP of 74.86% and rank-1 accuracy of 77.85%.
671:The improvements clearly indicate that attribute-aware and contour-aware prompts provide
672:complementary supervision, enhancing the cross-modal alignment capability of the model.
673:— Finally, we examine the contribution of the TRM by comparing the baseline+ATG+CTP
674:configuration with the full model that incorporates TRM. While ATG and CTP provide initial semantic and structural alignment cues, TRM is designed to further refine these textual

codex
TAP-align 的消融说明已经找到了：它把属性文本、轮廓文本、重定义模块分开验证。我要再取一次表 3 周边的具体文字和数值，方便最后不只写“有消融”，而能说明闭环证据是什么。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 14 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | sed -n '602,720p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Table 3. Ablation Study on SYSU-MM01

Components
ATG CTP TRM
3
3
3
4.3

3
3
3

3

mAP
72.25
73.28
73.13
74.86
76.21

SYSU-MM01
rank-1 rank-10
75.22
76.47
96.00
76.21
95.81
77.85
97.52
78.52
97.86

rank-20
97.72
97.45
98.63
99.23

Ablation Studies

To demonstrate the effectiveness of the modules in our proposed method, we conducted a series
of ablation experiments on SYSU-MM01. The rank-1, rank-5, and rank-10 mAP accuracies (%) are
reported in Table 3.
— Firstly, to further investigate the effectiveness of ATG, we compared the baseline method
with the baseline+ATG method. The ablation experiment in this section was implemented as
follows: we utilized the textual information generated from ATG to compute the category
prototypes of the text as alignment cues, guiding images from different modalities to align
towards these cues. As shown in the table, the textual information generated based on image
pedestrian recognition attributes serves as a new alignment cue, leading to better retrieval
performance compared to the baseline model. Specifically, mAP and rank-1 scores improved by
approximately 1.03% and 1.25%, respectively. This demonstrates that the method of generating
textual information based on image pedestrian attributes is effective.
— To further investigate the effectiveness of CTP, we conducted ablation studies by comparing
the baseline model with the baseline+CTP variant. In this configuration, the CTP module
extracts structural contour information from pedestrian images and generates Contour-aware
class-level textual features, which are employed as additional alignment cues to facilitate
modality-invariant feature learning. As reported in Table 3, the inclusion of CTP yields
consistent performance improvements, leading to gains of 0.88% in mAP and 0.99% in rank-1
accuracy. These results validate that contour-guided textual semantics provide meaningful
structural priors and enhance the cross-modal alignment capability of the model.
— To further validate the complementarity between semantic and structural alignment cues, we
evaluate the effectiveness of jointly applying the ATG and CTP modules. In this configuration,
class-level textual prototypes are independently generated by ATG and CTP, where ATG encodes high-level semantic attributes and CTP captures local structural contours of pedestrians.
These two sets of prototypes are averaged to obtain a unified textual representation, which
serves as a composite alignment reference across modalities. This simple yet effective fusion
integrates both semantic abstraction and structural detail, enabling more robust modalityinvariant feature learning. As presented in Table 3, the integration of ATG and CTP results
in further performance gains, yielding an mAP of 74.86% and rank-1 accuracy of 77.85%.
The improvements clearly indicate that attribute-aware and contour-aware prompts provide
complementary supervision, enhancing the cross-modal alignment capability of the model.
— Finally, we examine the contribution of the TRM by comparing the baseline+ATG+CTP
configuration with the full model that incorporates TRM. While ATG and CTP provide initial semantic and structural alignment cues, TRM is designed to further refine these textual
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.


exec
/bin/zsh -lc "pdftotext -f 8 -l 14 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | sed -n '817,1325p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
A BLATION S TUDY OF K EY C OMPONENTS IN THE P ROPOSED SD-3DGF F RAMEWORK UNDER THE C LOTHES -C HANGING S ETTING (%)
Settings

AS

GS

MRIE
Interaction
Fusion
FMHA
DGRT
PF
GAJE
-

LSL

1(Baseline)
2
3
4
5
6
7
8
9(Full)

-

Params
(M)

FLOPs
(G)

10.7
4.6
16.6
27.3
38.7
41.2
40.8
47.4
48.9

13.3
3.5
18.2
31.5
48.7
40.0
57.3
40.6
41.8

(a)

VCCR (CC)
Rank-1
49.7
45.9
50.2
51.0
55.6
55.9
59.4
56.5
59.8

mAP
40.2
35.8
40.9
41.4
45.5
46.1
48.3
45.5
48.6

CCVID (CC)
Rank-1
80.9
75.7
81.5
83.2
88.4
89.0
93.0
91.3
93.9

mAP
75.3
70.4
76.0
78.1
85.2
85.5
90.5
88.1
91.2

CCV-R (CC)
Rank-1
31.7
26.2
36.8
38.1
48.3
48.9
53.9
49.1
54.4

(b)

CCV-S (CC)

mAP
23.5
18.3
28.5
29.7
35.9
36.3
42.9
36.0
43.3

Rank-1
47.6
39.5
50.8
51.0
53.7
55.4
57.8
56.6
58.5

mAP
29.2
23.1
32.9
34.8
38.4
40.8
43.6
42.2
45.0

(c)

Fig. 5. Influences of the hyperparameters D, w2 , and η. The x-axis denotes the hyperparameter values, while the y-axis shows the Rank-1 and mAP
performance under the clothes-changing setting on the CCVID dataset.

TABLE IV
C OMPARISON OF S TORAGE AND C OMPUTATIONAL C OMPLEXITY OF
VCCR E -ID M ETHODS WITH M ESH -BASED R EPRESENTATIONS AT
I NFERENCE († C ODE R EPRODUCED BY U S )

TABLE VI
A BLATION S TUDY OF U SING D IFFERENT M ODALITIES FOR G AIT
M ODELING (%)
CCVID (CC)

CCV-R (CC)

CCV-S (CC)

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

2D-Aware

2D Keypoints
Silhouettes
Part Segm. Maps

50.9
51.7
52.1

41.6
42.9
43.0

83.3
89.7
90.9

79.1
88.1
89.4

43.8
49.3
50.2

33.0
37.4
39.2

48.5
52.0
54.8

32.9
37.9
40.0

3D-Aware

3D Skeletons
Point Clouds
SMPL Parameters
Dense Corr. Embed.

51.4
58.2
50.5
59.8

42.7
47.0
41.6
48.6

85.8
91.3
85.5
93.9

86.3
89.6
84.9
91.2

45.1
52.7
43.5
54.4

34.8
41.5
33.1
43.3

50.6
58.1
43.8
58.5

35.5
44.2
27.3
45.0

Categories

Categories

Methods

3D Prior Network

Baseline

Params (M)

FLOPs (G)

10.7

13.3

Mesh-Recovery-Based

3STA [47]†
SEMI [48]
CMCL [23]†

HMR [70]
HMR [70]
HMR [70]

75.6
64.2
161.9

60.13
54.52
122.3

Mesh-Distillation-Based

SD-3DGF (Ours)

Eff-UNet [60]

48.9

41.8

TABLE V
A BLATION S TUDY OF 3D G AIT F IELD L OSS AND O RTHOGONAL L OSS (%)
VCCR (CC)

Methods
SD-3DGF (w/o. Lorg & Lgf )
SD-3DGF (w/o. Lorg )
SD-3DGF (w/o. Lgf )
SD-3DGF (Full)

Modality

Storage & Computational Cost

CCVID (CC)

CCV-R (CC)

CCV-S (CC)

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

52.4
53.0
59.1
59.8

42.0
42.1
47.7
48.6

84.9
85.7
91.5
93.9

82.6
83.0
89.7
91.2

44.0
45.8
51.9
54.4

32.1
34.2
40.7
43.3

50.2
51.7
56.4
58.5

33.5
35.5
41.9
45.0

VCCR (CC)

achieves performance comparable to the state-of-the-art shortterm video Re-ID method HAMoBE, demonstrating its strong
generalization ability. This is attributed to the use of dense
correspondence embeddings for learning 3D gait features,
which remain robust in clothes-consistent scenarios.
D. Ablation Studies

performance compared with VCCRe-ID methods.
2) Generalization Ability in Clothes-Consistent Scenarios:
As shown in Tab. I, under the standard setting, which includes
both clothes-changing and clothes-consistent samples, SD3DGF consistently outperforms all competitors on the VCCR,
CCVID, CCV-R, and CCV-S datasets. Specifically, it achieves
gains of 1.1%, 1.3%, 0.1%, and 2.8% in Rank-1 accuracy, and
1.1%, 0.8%, 6.9%, and 9.0% in mAP. Furthermore, as shown
in Tab. II, for standard video-based Re-ID datasets such as
MARS, LS-VID, and iLIDS-VID, where pedestrians undergo
little or no clothing changes, most VCCRe-ID methods exhibit
noticeable performance degradation. A plausible explanation
is that desensitizing clothing features can inadvertently remove some reliable biometric cues. In contrast, our method

In this subsection, we conduct systematic ablation studies
on VCCR, CCVID, CCV-R and CCV-S to investigate: (1) the
effectiveness of key components in our proposed method; (2)
the storage and computational cost required by our model;
(3) the effects of the 3D gait field loss and the orthogonal
loss during end-to-end training; (4) the impacts of different
2D and 3D auxiliary modalities; (5) the influences of different
hyperparameters.
1) Effectiveness of GS, LSL, and MRIE: Tab. III reports
how different component combinations affect the storage,
computational complexity, and performance on VCCRe-ID
datasets. For clarity, the notations used in Tab. III are defined as follows. AS and GS denote the appearance and
gait streams, respectively. LSL stands for local similarity
learning. MRIE refers to the Modality-Representative Inter-

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

Query Tracklets

11

Rank-1 Retrieval Results

VCCR

CCVID

CCV-R

CCV-S

Fig. 6. Visualization of the learned dense correspondence embeddings on VCCR, CCVID, CCV-R, and CCV-S, obtained via Principal Component Analysis
(PCA) for dimensionality reduction. Query tracklets are shown on the left, with their corresponding rank-1 retrieval results from the gallery on the right.

active Encoding module. FMHA denotes full multi-head self/cross-attention across all tokens, while DGRT indicates the
Dual-Granularity Representative Token Generation strategy.
PF represents token-level pooling fusion, which first averages
all tokens from different modalities and then feeds the result
into an MLP to generate the final identity feature, and GAJE
corresponds to the Gait-Appearance Joint Encoding submodule in MRIE. When no representative token is generated,
GAJE is configured to perform MHCA over all tokens of both
modalities, followed by adaptive weighting.
It is clear that single-stream models using only the appearance stream (Baseline) or the gait stream (Settings 2
and 3) yield suboptimal performance. Adding LSL to the
gait stream (Setting 2 → 3, 8 → 9) strengthens 3D gait
features and improves Re-ID performance. Setting 2 uses
only 3D gait fields from Eff-UNet [60] with average pooling for classification, whereas Setting 3 corresponds to the
full gait stream (GS-F) with the subsequent convolutional
blocks and LSL. We observe that the full gait stream (GSF) consistently outperforms the appearance stream on the four
VCCRe-ID datasets. While simple concatenation of the two
streams without feature interaction (Setting 4) brings only
minor improvement, incorporating MRIE (Setting 4 → 9)
leads to notable gains: Rank-1 improves by 8.8%, 10.7%,
16.3%, and 7.5%, and mAP by 7.2%, 13.1%, 13.6%, and
10.2% on VCCR, CCVID, CCV-R, and CCV-S, respectively,
demonstrating MRIE’s effectiveness for core Re-ID modeling.
To evaluate the DGRT strategy and the GAJE submodule
in MRIE, we additionally introduce three control settings:
Setting 5, 6, and 7. Generating global and local representative
tokens to avoid full MHSA/MHCA (Setting 5 → 6, 7 →
9) adds few parameters while substantially reducing FLOPs,

without performance loss. Replacing token-level pooling with
GAJE for cross-modal fusion (Setting 5 → 7, 6 → 9) yields
significant performance gains, indicating that DGRT reduces
computational burden, while GAJE makes the largest contribution to MRIE’s performance.
2) Analysis of the Storage and Computational Cost: Based
on the results in Tab. III, the appearance stream, the full gait
stream (GS-F), and the MRIE module have 10.7M, 16.6M, and
21.6M parameters, respectively, with computational costs of
13.3, 18.2, and 10.3 GFLOPs. Within the gait stream, the EffUNet-based correspondence embedding extractor adds only
4.6M parameters and 3.5 GFLOPs, while the LSL module
incurs 1.5M parameters and 1.2 GFLOPs. Most of the additional parameters and computational cost in the gait stream
originate from the ResNet-50 convolutional blocks. Within
MRIE, DGRT requires 13.9M parameters and 8.5 GFLOPs,
whereas GAJE requires 7.7M parameters and 1.8 GFLOPs.
Compared to performing full MHSA and MHCA over all
tokens, DGRT reduces 15.5 GFLOPs by generating dualgranularity representative tokens, which compress the spatial
dimensions of the original tokens (Setting 7 → 9), with only
an 8.1M increase in parameters.
We further compare our method with existing mesh-based
VCCRe-ID approaches in terms of storage and computational
cost in Tab. IV. In our experiments, all mesh-recovery-based
methods adopt HMR [70] as the SMPL parameter estimator
during inference. Although introducing additional 3D modalities increases storage and computational cost compared to the
single appearance stream (Baseline), our method remains the

exec
/bin/zsh -lc "pdftotext -f 8 -l 14 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | sed -n '1715,1958p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3) Effects of the 3D Gait Field Loss and the Orthogonal
Loss: To further examine the influence of the 3D gait field
loss Lgf and the orthogonal loss Lorg in the end-to-end
training stage, we perform ablation experiments by selectively
removing each loss term. Specifically, removing Lgf indicates
that the correspondence embedding extractor fθ is entirely
frozen throughout the end-to-end optimization. As shown in
Tab. V, removing either Lorg or Lgf leads to a clear performance drop. Notably, removing Lorg leads to a significantly
larger performance degradation, reflecting that the appearance
stream in the dual-stream framework is highly sensitive to
its associated key constraint Lorg , which suppresses clothingrelated features. In contrast, removing Lgf (i.e., freezing the
pre-trained fθ ) results in a noticeable but limited performance
drop. This suggests that further optimizing fθ during end-toend training remains beneficial, yet the pre-trained fθ already
demonstrates strong transferability to downstream Re-ID tasks.
4) Choices of Different Auxiliary Modalities: We conduct
a systematic investigation of the auxiliary modalities used in
the gait stream. Keeping all other components of the model
fixed, we replace the dense correspondence embeddings with
a range of 2D modalities, including 2D keypoints, silhouettes,
and part segmentation maps, and 3D modalities such as 3D
skeletons, point clouds, and SMPL parameters, to examine
their effect on overall model performance. Specifically, the
SMPL parameters are transformed into feature maps via a 3D
spatial transformation network [18], while the point clouds are
first converted into sparse depth maps by projecting the SMPL
vertex coordinates and subsequently processed by ResNet50 to extract frame-level features. From Tab. VI, it can be
observed that using dense correspondence embeddings for gait
modeling leads to the most significant overall performance improvement. In contrast, employing 2D keypoints, 3D skeletons,

VCCR (CC)

or SMPL parameters yields inferior results, as the information
they provide is relatively sparse and insufficient for learning
highly discriminative identity cues.
5) Influences of Hyperparameters: In our model, seven key
hyperparameters are defined: the correspondence embedding
dimension D; λ in Eq. 26, which controls the relative weight
of the directional loss; µ1 and µ2 in Eq. 27, which regulate
the consistency constraints during 3D gait field training; w1
and w2 in Eq. 30, which balance the ID loss, the triplet loss,
and the orthogonal loss; and η in Eq. 31, which determines the
relative contribution of the Re-ID loss and the 3D gait field
loss. From Fig. 5(a), we observe that Rank-1 and mAP on
the CCVID dataset peak when the embedding dimension D
is 64. Increasing D further does not improve performance,
as higher-dimensional embeddings tend to be redundant or
non-discriminative, which can destabilize downstream Re-ID
training. For the loss-related hyperparameters, we initialize
each to 1.0 and vary them individually from 1.0 to 0.0
in steps of 0.05 while keeping the others fixed to identify
optimal values. Fig. 5(b) and (c) illustrate the effects of w2
and η, with optimal performance achieved at 0.10 and 0.90,
respectively. Notably, model performance is more sensitive to
w2 than to η, due to the strong orthogonal constraint. The
same hyperparameter settings yield optimal results across all
seven datasets, demonstrating cross-dataset robustness.
E. Further Analysis
1) Model Sensitivity to SMPL Parameter Errors in the
Training Data Generation Process: Our method leverages 3D2D distillation to eliminate the need for SMPL parameter
estimation at inference. However, during training, it is still
necessary to estimate SMPL parameters in order to project
the mesh vertices and generate dense 2D-3D paired data.
To assess our model’s sensitivity to errors in SMPL parameters during training, we introduce random perturbations of
varying magnitudes to the camera, pose, and shape parameters, degrading the paired training data. The model is then
evaluated on multiple VCCRe-ID benchmarks. Specifically,
perturbations of 1%, 5%, and 10% are applied to the original
parameter values, corresponding to small, moderate, and large
perturbations, respectively. Perturbations beyond 10% are not
considered, as they may violate inherent constraints (e.g., axisangle constraints) and cause reconstruction failures.
As summarized in Tab. VII, the results indicate that under
small perturbations, camera and shape parameters have minimal impact on performance, whereas pose perturbations cause

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

13

scattered

Low
Resolution

Part Missing

scattered

(a) SMPL parameters

(b) Appearance Stream (Baseline)
assembled

Under
Occlusion
Dense Corr. Embed. + Visible Mask

Mask
Unreliable
Poses

3D Mesh
Recovery

Dense Corr. Embed. + Visible Mask

assembled

3D Mesh
Recovery

assembled

scattered

(c) Gait Stream (Ours)

(d) ID Features (Ours)

Fig. 7. Visualization of the learned dense correspondence embeddings under
challenging conditions in the CCV-R dataset.

Fig. 8. t-SNE visualization of SMPL parameters, appearance features, 3D
gait features, and fused ID features for 20 randomly selected identities from
the CCV-R dataset.

a comparatively larger decline. This is due to perturbing all 72
dimensions of the pose parameters, where even minor changes
can markedly affect the underlying pose distribution. Under
moderate and large perturbations, the model is considerably
more sensitive to camera and pose parameters than to shape
parameters, with camera perturbations having the most pronounced effect. Compared to the unperturbed setting, applying
a 10% perturbation to the camera parameters leads to Rank-1
drops of 9.9%, 11.6%, 21.5%, and 9.7%, and mAP drops of
8.0%, 14.2%, 17.3%, and 14.9% on VCCR, CCVID, CCV-R,
and CCV-S, respectively. This sensitivity arises because large
changes in scale and translation (S&T) cause the projected 3D
vertices to become globally misaligned, severely disrupting the
2D-3D correspondences. Furthermore, large pose perturbations
may produce implausible poses; in practice, current 3D human
motion capture methods generally alleviate such errors by
employing adversarial training with data-driven priors.
2) Reliability of Dense Correspondence Embeddings at
Inference: Video-based Re-ID datasets vary in data quality
and characteristics. For example, CCVID contains simple
scenes with few occlusions, VCCR and CCV-R feature heavy
occlusions and low resolution, and CCV-S consists entirely of
synthetic data. To examine the impact of dense correspondence
embedding reliability on gait stream performance, we evaluate
these embeddings on the test sets of both standard videobased Re-ID and VCCRe-ID datasets. Tab. VIII reports the
Embedding Consistency Score (ECS) for the query, gallery,
and full test sets, where ECS, as defined in Eq. 32, quantifies
the alignment between the predicted dense correspondence
embeddings and ground-truth 3D vertex embeddings. Using
the predicted embeddings under the standard setting (SD) of
each dataset, we evaluate the performance of the full gait
stream (GS-F). The results in Tab. VIII indicate a positive
correlation between gait stream performance and embedding
reliability. Remarkably, even for datasets with relatively low
ECS, such as CCV-S and CCV-R, the gait stream maintains
reasonable performance, demonstrating not only its robustness
to imperfect 2D-3D alignment in low-quality video data but
also the notable discriminative capability of our proposed 3D
gait representations under challenging conditions.

3) Model Sensitivity to Different Body Regions: To investigate the model’s sensitivity to different body regions (e.g.,
the face and lower limbs) at inference, and to further probe
the underlying mechanisms and complementarity between
the appearance and gait streams, we conduct region-masking
experiments on the single appearance stream (Baseline), the
full gait stream (GS-F), and the overall model.
According to Tab. IX, masking the face region during inference nearly disables the appearance stream, while masking
the lower limbs significantly degrades the full gait stream
(GS-F), yet still preserves a reasonable level of performance.
This indicates that the gait stream is not solely dependent on
lower-limb dynamics, but can also capture informative identity
cues from other body regions. In contrast, although both types
of masking affect the dual-stream model, the performance
decrease is modest.
These results indicate that, within the dual-stream network,
one stream can effectively compensate for missing or corrupted cues in the other, illustrating their strong complementarity. In practical scenarios, facial regions are often unavailable
(e.g., due to back-facing poses), and lower limbs may be
missing or occluded in video frames. The complementarity of
the dual-stream features greatly enhances robustness in such
challenging cases.
4) Visualization: We apply Principal Component Analysis
(PCA) to project the dense correspondence embeddings extracted by fθ from H×W ×D to H×W ×3 for visualization.
Visualization results on VCCR, CCVID, CCV-R, and CCVS are shown in Fig. 6. The embeddings exhibit a smooth
and continuous distribution, indicating that fθ learns coherent
correspondences across the body surface.
Fig. 7 presents visualizations on the CCV-R test set under
challenging conditions, including low-resolution videos, missing body parts, and occlusions. It can be observed that the
dense correspondence embeddings still exhibit high similarity
within nearby body regions, while maintaining clear separability across distant regions, indicating that the correspondence
embedding extractor fθ remains reasonably stable. Moreover,
by restricting embeddings to visible regions, our method
automatically filters out unreliable information from occluded

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

Appearance Stream

Gait Stream

Fig. 9. Class activation maps of the appearance stream and the gait stream,
with the video tracklet sampled from the CCV-R dataset.

or missing body parts. This strategy not only preserves training data fidelity but also enhances inference robustness by
disregarding invisible 3D poses, which are often inaccurately
estimated and highly unstable in current mesh-recovery-based
methods that perform full-body reconstruction, thus mitigating
the impact of pose estimation errors.
We further employ t-SNE in Fig. 8 to visualize the feature
distributions of 20 randomly selected identities from the CCVR dataset, including SMPL parameters, appearance features,
3D gait features, and the fused ID features. The SMPL parameters appear largely disorganized and non-discriminative,
reflecting their limited utility for identity discrimination. Appearance features exhibit dispersed intra-class distributions
and blurred inter-class boundaries, suggesting that appearance
alone provides insufficient cues under the clothes-changing
setting. In contrast, our 3D gait features form relatively more
compact intra-class clusters than the appearance features, with
a few identities clearly clustered, but inter-class separability
remains limited. After fusing both streams via the MRIE
module, the resulting ID features display markedly improved
class-wise aggregation and clearer inter-class separability, underscoring the strong complementarity between appearance
and gait streams.
To examine spatial activation patterns, we visualize class
activation maps for the appearance and gait streams in Fig. 9. It
can be observed that the appearance stream primarily focuses
on local regions less affected by clothing, reflecting the constraint imposed by the orthogonal loss, whereas the gait stream
attends mainly to the lower-limb regions while also exhibiting
broad activations across the entire body. This indicates that
the gait stream primarily captures walking patterns, while also
encoding certain global discriminative identity cues. These

codex
SD-3DGF 的闭环很清楚：不仅有组件消融，还有“换掉辅助模态”“嵌入一致性”“t-SNE 和激活图”。我再补 TAP-align 的 TRM 消融结尾，然后开始组织最终答案。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 14 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | sed -n '674,725p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
configuration with the full model that incorporates TRM. While ATG and CTP provide initial semantic and structural alignment cues, TRM is designed to further refine these textual
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 14 -l 15 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | sed -n '1,120p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
113:14

Y. Tao and X. Zhang
Table 3. Ablation Study on SYSU-MM01

Components
ATG CTP TRM
3
3
3
4.3

3
3
3

3

mAP
72.25
73.28
73.13
74.86
76.21

SYSU-MM01
rank-1 rank-10
75.22
76.47
96.00
76.21
95.81
77.85
97.52
78.52
97.86

rank-20
97.72
97.45
98.63
99.23

Ablation Studies

To demonstrate the effectiveness of the modules in our proposed method, we conducted a series
of ablation experiments on SYSU-MM01. The rank-1, rank-5, and rank-10 mAP accuracies (%) are
reported in Table 3.
— Firstly, to further investigate the effectiveness of ATG, we compared the baseline method
with the baseline+ATG method. The ablation experiment in this section was implemented as
follows: we utilized the textual information generated from ATG to compute the category
prototypes of the text as alignment cues, guiding images from different modalities to align
towards these cues. As shown in the table, the textual information generated based on image
pedestrian recognition attributes serves as a new alignment cue, leading to better retrieval
performance compared to the baseline model. Specifically, mAP and rank-1 scores improved by
approximately 1.03% and 1.25%, respectively. This demonstrates that the method of generating
textual information based on image pedestrian attributes is effective.
— To further investigate the effectiveness of CTP, we conducted ablation studies by comparing
the baseline model with the baseline+CTP variant. In this configuration, the CTP module
extracts structural contour information from pedestrian images and generates Contour-aware
class-level textual features, which are employed as additional alignment cues to facilitate
modality-invariant feature learning. As reported in Table 3, the inclusion of CTP yields
consistent performance improvements, leading to gains of 0.88% in mAP and 0.99% in rank-1
accuracy. These results validate that contour-guided textual semantics provide meaningful
structural priors and enhance the cross-modal alignment capability of the model.
— To further validate the complementarity between semantic and structural alignment cues, we
evaluate the effectiveness of jointly applying the ATG and CTP modules. In this configuration,
class-level textual prototypes are independently generated by ATG and CTP, where ATG encodes high-level semantic attributes and CTP captures local structural contours of pedestrians.
These two sets of prototypes are averaged to obtain a unified textual representation, which
serves as a composite alignment reference across modalities. This simple yet effective fusion
integrates both semantic abstraction and structural detail, enabling more robust modalityinvariant feature learning. As presented in Table 3, the integration of ATG and CTP results
in further performance gains, yielding an mAP of 74.86% and rank-1 accuracy of 77.85%.
The improvements clearly indicate that attribute-aware and contour-aware prompts provide
complementary supervision, enhancing the cross-modal alignment capability of the model.
— Finally, we examine the contribution of the TRM by comparing the baseline+ATG+CTP
configuration with the full model that incorporates TRM. While ATG and CTP provide initial semantic and structural alignment cues, TRM is designed to further refine these textual
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID 113:15
Table 4. Analysis of Hyperparameter 𝜆1

𝜆1
0.1
0.15
0.2
0.25
0.3
0.35
0.4

mAP
75.87
76.21
75.46
74.88
74.57
73.97
73.85

rank-1
77.69
78.52
77.18
76.94
76.49
75.94
75.69

rank-5
96.83
97.86
96.78
96.21
96.17
95.91
95.58

rank-10
98.51
99.23

codex
TAP-align 的第 14 页跨页断在 TRM 说明处，表格已经给出了全模型从 74.86 mAP 到 76.21 mAP、rank-1 从 77.85 到 78.52 的提升。这个足够支撑 TRM 的闭环，我会以表格数值为准，不再扩大阅读范围。
codex
我按 PDF 正文读了方法和关键消融，没额外联网，因为这些论文自己的相关工作、方法和实验已经足够判断它们怎么构造 novelty。

**1. Spectrum-guided feature enhancement network for event person re-identification.pdf**

1. **触发观察**：事件相机有隐私和强光弱光优势，但事件流语义极稀疏、噪声又多。已有 event ReID 更偏隐私保护，忽略了“稀疏事件里怎么提可判别语义”这个核心问题。作者还抓住一个具体物理现象：事件噪声常表现为孤立点，在频域偏高频，但有效事件语义本身也可能是离散孤立点。

2. **重定义动作**：他们没有把问题讲成“把 RGB ReID 迁移到事件数据”，而是讲成“事件 ReID 的核心是 sparse semantic denoising and enhancement”。关键词是 sparse event semantics、event noise、frequency-domain denoising、multi-grain semantic interaction、event contextual semantics。这样一来，旧问题从“缺一个更强 ReID 网络”变成“事件流的语义和噪声强耦合，需要同时去噪和增强上下文”。

3. **机制怎么长出来**：MSAM 很自然地从这个重定义推出：既然噪声集中在高频，就用 Fourier 低通；既然单尺度会误删稀疏有效语义，就用 1×1、3×3、5×5 多尺度；既然频域滤波会过平滑，就接语义交互注意力补上下文。CPDM 则来自第二个观察：模型容易只盯少数局部有效点，所以连续擦掉不同横向 patch，迫使模型从剩余区域也能识别身份。机制和问题定义绑定较紧，但仍是“频域模块加训练正则”的组合型创新。

4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。

5. **reviewer 为什么买账**：这篇卖的是“事件 ReID 不只是隐私数据版 ReID，而是稀疏噪声语义建模问题”的视角，加上物理上能解释的频谱机制。novelty 主要来自任务特性绑定，不是单个模块多新。风险是 event ReID 数据集小，涨点很大容易被认为是场景窄，但审稿人会接受它的原因是问题定义、机制、可视化和消融能对上。

**2. Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf**

1. **触发观察**：VI-ReID 默认 RGB 和 IR 训练数据相对均衡，但现实里 IR 采集更贵、更少，夜间行人也少。已有方法主要解决 modality gap，却默认两边数据都够。作者还指出常规 SMOTE 这类少数类过采样在跨模态场景会失效，因为最近邻在强模态差异下会把合成样本推离真实 IR 分布。

2. **重定义动作**：他们把 VI-ReID 从“跨模态差异问题”重定义为“severely modality imbalance 下的 co-modality balance learning”。关键词是 imbalance VI-ReID、minority modality、feature-level oversampling、stochastic style perturbation、balanced representation learning。这个动作很关键，因为它新建了一个更贴近真实部署的协议，而不是继续在平衡 benchmark 上卷模块。

3. **机制怎么长出来**：既然 IR 少，直接复制图像会过拟合，SMOTE 又会偏离分布，那么就不在图像空间生成，而是在深层特征里扰动 IR 的 style statistics。SSP 用 IR 特征的均值和方差估计 style 范围，随机采样新的缩放和平移，生成语义不变但风格变化的 IR-like 特征。随后用 MMD 对齐 RGB、真实 IR、扰动 IR 三个分布，再用 CACS 做类中心层面的 intra-class 和 inter-class 约束。机制和重定义绑定很紧。

4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。

5. **reviewer 为什么买账**：这篇真正卖的是新问题设定，而不是 SSP 本身多复杂。SSP 是一个足够自然、低成本的机制，用来支撑“少数模态特征分布应该被扩展而不是被多数模态吞掉”这个视角。reviewer 买账点在于：它先证明原有 VI-ReID 假设不现实，再证明同一批 SOTA 在失衡下掉得厉害，最后用一个针对性机制补上。

**3. Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf**

1. **触发观察**：换衣视频 ReID 里外观不可靠，gait 是天然身份线索。问题是已有 2D gait 容易受视角影响，3D skeleton 和 SMPL 参数又太稀疏、低维，既不够判别，也和高维 appearance feature 难对齐。作者抓住的是“3D 信息有用，但现有 3D 表示形态不适合和 ReID 特征融合”。

2. **重定义动作**：他们没有说“再加一个 gait 分支”，而是把问题改成“如何把结构化 3D 身体知识蒸馏成与视频外观时空对齐的稠密 gait field”。关键词是 Structured Distilled 3D Gait Fields、3D-2D distillation、dense correspondence embeddings、spatiotemporal field、appearance-compatible 3D gait。旧问题从“用什么辅助模态”升级成“辅助模态的表示形态是否和 appearance 可交互”。

3. **机制怎么长出来**：机制几乎是从重定义里长出来的。先在 SMPL mesh 上预训练 vertex embeddings，让嵌入距离对应 geodesic distance 和 tangent direction；再用 3D mesh recovery 和相机参数把顶点、采样表面点投影到 2D 图像，训练 U-Net 输出 pixel-level dense correspondence embeddings；视频帧串起来就是 3D gait field。然后用 local similarity learning 建模局部时空邻域，最后用 MRIE 的全局和局部代表 token 让 gait 和 appearance 高效交互。这里不是随便拼模块，主线是“稀疏 3D 先验变成稠密、时空对齐、可融合的 ReID 表示”。

4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。

5. **reviewer 为什么买账**：这篇卖的是视角加表示范式。novelty 不在“用 gait”本身，而在“把 3D gait 从稀疏骨架或低维参数，重做成 2D 视频域稠密场”。这是比较像方法稿的创新构造：有明确旧表示缺陷，有自然机制，有替换模态消融和可视化支撑。缺点是工程重、依赖训练期 3D recovery，但作者通过推理期不需要 SMPL 来缓解这个风险。

**4. Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf**

1. **触发观察**：VI-ReID 里 RGB 和 IR 差异大，生成式方法会有伪影，纯视觉 invariant learning 缺高层语义。CLIP 类方法开始引入语言，但很多只是学几个 token proxy，语义表达不够丰富。另一个观察是：自然语言属性有用，但往往缺 contour、body shape、pose 这类结构信息；而颜色等低层线索在 IR 里又不可靠。

2. **重定义动作**：作者把问题讲成“跨模态对齐缺少显式、模态无关的 semantic anchor”，而不是“再做一个 CLIP prompt”。关键词是 attribute-aware text、contour-aware text prompt、text re-definition、category-level textual prototypes、modality-agnostic semantic anchors。这个重定义把文本从辅助描述变成“RGB 和 IR 共同对齐的类别原型”。

3. **机制怎么长出来**：ATG 用 BLIP 问答和固定模板生成属性文本，并刻意避开颜色，保留 age、gender、action、clothing type、hair、glasses 这些更跨模态的属性。CTP 用 SCHP 人体解析去掉衣服区域，保留 body shape 和 pose 相关区域，再用可学习 prompt 表示 contour-aware class text。TRM 负责把 instance-level attribute text 和 class-level contour prompt 融成 image-aware redefined text，最后做 class-level text prototype，指导 RGB 和 IR visual feature 对齐。机制和“属性语义加结构语义”的问题定义是匹配的，但整体偏组合式。

4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。

5. **reviewer 为什么买账**：这篇卖的主要是视角，不是模块精巧度。reviewer 买账点是它把“文本提示”拆成两类可解释语义：属性和轮廓，并用 prototype 把文本变成跨模态锚点。novelty 的来源是“显式语义原型替代弱 token proxy”，而不是 BLIP、SCHP、CLIP 或 Transformer decoder 本身。

**5. Text-to-image Person Search based on Semantic Reorganization.pdf**

1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。

2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。

3. **机制怎么长出来**：SRDA 很直接：先用 MLLM 判断性别，在同一性别内随机选两个 ID，把一个人的上半身和另一个人的下半身横向拼接，生成新 ID，再用微调 BLIP 为新图生成文本。它不追求自然，而追求结构组合多样性。CSRM 则在测试时为文本 query 找 cross-modal k-reciprocal 图像邻居，用 Jaccard 权重把可靠图像特征融合进文本特征，再检索。训练阶段用 CLIP 双流、ID loss、SDM loss 和 image triplet；不加 text triplet，因为 MLLM 文本有噪声。

4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。

5. **reviewer 为什么买账**：这篇卖的是两个务实视角：训练时不迷信 diffusion 真实感，测试时不照搬同模态 re-ranking。机制都很简单，但和观察贴得紧，消融也正好打在 reviewer 会怀疑的点上。novelty 来源不是复杂模型，而是对 TBPS 任务目标的重新理解。

**6. Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf**

1. **触发观察**：遮挡 ReID 里 pose、human parsing 和 occlusion augmentation 都常见，但作者指出两个更具体的失败场景：极端物体遮挡下，局部 pose 错误会污染全局特征；非目标行人遮挡下，遮挡者也有人体骨架，pose 无法区分目标和遮挡行人。作者还用了一个接近人类识别的观察：行人互相遮挡时，背包、伞、衣服纹理这类 distinctive appearance 往往决定识别。

2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。

3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。

4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。

5. **reviewer 为什么买账**：这篇卖的是“对遮挡失败类型的细分”加“机制一一对应”。如果只说 pose+texture，novelty 会弱；但作者先指出 pose 方法在极端遮挡和行人遮挡下分别为什么失效，再让 PPM 和 TAD 分别处理，reviewer 就更容易接受。真正 novelty 来自问题拆解，而不是 Transformer、pose 或 attention 本身。

**共同套路**

这批论文共同不是先发明一个模块，而是先把已有任务里的默认说法改窄、改具体：事件 ReID 不是 RGB 迁移，VI-ReID 不只是 modality gap，换衣 ReID 不只是加 gait，遮挡 ReID 不只是找可见部位。它们能成稿的关键是把观察变成一句新的失败机制，然后让方法的每个部件都对应这个失败机制，最后用“替换模态、破坏假设、组件拆分、可视化”证明视角本身是对的。
tokens used
175,790
我按 PDF 正文读了方法和关键消融，没额外联网，因为这些论文自己的相关工作、方法和实验已经足够判断它们怎么构造 novelty。

**1. Spectrum-guided feature enhancement network for event person re-identification.pdf**

1. **触发观察**：事件相机有隐私和强光弱光优势，但事件流语义极稀疏、噪声又多。已有 event ReID 更偏隐私保护，忽略了“稀疏事件里怎么提可判别语义”这个核心问题。作者还抓住一个具体物理现象：事件噪声常表现为孤立点，在频域偏高频，但有效事件语义本身也可能是离散孤立点。

2. **重定义动作**：他们没有把问题讲成“把 RGB ReID 迁移到事件数据”，而是讲成“事件 ReID 的核心是 sparse semantic denoising and enhancement”。关键词是 sparse event semantics、event noise、frequency-domain denoising、multi-grain semantic interaction、event contextual semantics。这样一来，旧问题从“缺一个更强 ReID 网络”变成“事件流的语义和噪声强耦合，需要同时去噪和增强上下文”。

3. **机制怎么长出来**：MSAM 很自然地从这个重定义推出：既然噪声集中在高频，就用 Fourier 低通；既然单尺度会误删稀疏有效语义，就用 1×1、3×3、5×5 多尺度；既然频域滤波会过平滑，就接语义交互注意力补上下文。CPDM 则来自第二个观察：模型容易只盯少数局部有效点，所以连续擦掉不同横向 patch，迫使模型从剩余区域也能识别身份。机制和问题定义绑定较紧，但仍是“频域模块加训练正则”的组合型创新。

4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。

5. **reviewer 为什么买账**：这篇卖的是“事件 ReID 不只是隐私数据版 ReID，而是稀疏噪声语义建模问题”的视角，加上物理上能解释的频谱机制。novelty 主要来自任务特性绑定，不是单个模块多新。风险是 event ReID 数据集小，涨点很大容易被认为是场景窄，但审稿人会接受它的原因是问题定义、机制、可视化和消融能对上。

**2. Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf**

1. **触发观察**：VI-ReID 默认 RGB 和 IR 训练数据相对均衡，但现实里 IR 采集更贵、更少，夜间行人也少。已有方法主要解决 modality gap，却默认两边数据都够。作者还指出常规 SMOTE 这类少数类过采样在跨模态场景会失效，因为最近邻在强模态差异下会把合成样本推离真实 IR 分布。

2. **重定义动作**：他们把 VI-ReID 从“跨模态差异问题”重定义为“severely modality imbalance 下的 co-modality balance learning”。关键词是 imbalance VI-ReID、minority modality、feature-level oversampling、stochastic style perturbation、balanced representation learning。这个动作很关键，因为它新建了一个更贴近真实部署的协议，而不是继续在平衡 benchmark 上卷模块。

3. **机制怎么长出来**：既然 IR 少，直接复制图像会过拟合，SMOTE 又会偏离分布，那么就不在图像空间生成，而是在深层特征里扰动 IR 的 style statistics。SSP 用 IR 特征的均值和方差估计 style 范围，随机采样新的缩放和平移，生成语义不变但风格变化的 IR-like 特征。随后用 MMD 对齐 RGB、真实 IR、扰动 IR 三个分布，再用 CACS 做类中心层面的 intra-class 和 inter-class 约束。机制和重定义绑定很紧。

4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。

5. **reviewer 为什么买账**：这篇真正卖的是新问题设定，而不是 SSP 本身多复杂。SSP 是一个足够自然、低成本的机制，用来支撑“少数模态特征分布应该被扩展而不是被多数模态吞掉”这个视角。reviewer 买账点在于：它先证明原有 VI-ReID 假设不现实，再证明同一批 SOTA 在失衡下掉得厉害，最后用一个针对性机制补上。

**3. Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf**

1. **触发观察**：换衣视频 ReID 里外观不可靠，gait 是天然身份线索。问题是已有 2D gait 容易受视角影响，3D skeleton 和 SMPL 参数又太稀疏、低维，既不够判别，也和高维 appearance feature 难对齐。作者抓住的是“3D 信息有用，但现有 3D 表示形态不适合和 ReID 特征融合”。

2. **重定义动作**：他们没有说“再加一个 gait 分支”，而是把问题改成“如何把结构化 3D 身体知识蒸馏成与视频外观时空对齐的稠密 gait field”。关键词是 Structured Distilled 3D Gait Fields、3D-2D distillation、dense correspondence embeddings、spatiotemporal field、appearance-compatible 3D gait。旧问题从“用什么辅助模态”升级成“辅助模态的表示形态是否和 appearance 可交互”。

3. **机制怎么长出来**：机制几乎是从重定义里长出来的。先在 SMPL mesh 上预训练 vertex embeddings，让嵌入距离对应 geodesic distance 和 tangent direction；再用 3D mesh recovery 和相机参数把顶点、采样表面点投影到 2D 图像，训练 U-Net 输出 pixel-level dense correspondence embeddings；视频帧串起来就是 3D gait field。然后用 local similarity learning 建模局部时空邻域，最后用 MRIE 的全局和局部代表 token 让 gait 和 appearance 高效交互。这里不是随便拼模块，主线是“稀疏 3D 先验变成稠密、时空对齐、可融合的 ReID 表示”。

4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。

5. **reviewer 为什么买账**：这篇卖的是视角加表示范式。novelty 不在“用 gait”本身，而在“把 3D gait 从稀疏骨架或低维参数，重做成 2D 视频域稠密场”。这是比较像方法稿的创新构造：有明确旧表示缺陷，有自然机制，有替换模态消融和可视化支撑。缺点是工程重、依赖训练期 3D recovery，但作者通过推理期不需要 SMPL 来缓解这个风险。

**4. Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf**

1. **触发观察**：VI-ReID 里 RGB 和 IR 差异大，生成式方法会有伪影，纯视觉 invariant learning 缺高层语义。CLIP 类方法开始引入语言，但很多只是学几个 token proxy，语义表达不够丰富。另一个观察是：自然语言属性有用，但往往缺 contour、body shape、pose 这类结构信息；而颜色等低层线索在 IR 里又不可靠。

2. **重定义动作**：作者把问题讲成“跨模态对齐缺少显式、模态无关的 semantic anchor”，而不是“再做一个 CLIP prompt”。关键词是 attribute-aware text、contour-aware text prompt、text re-definition、category-level textual prototypes、modality-agnostic semantic anchors。这个重定义把文本从辅助描述变成“RGB 和 IR 共同对齐的类别原型”。

3. **机制怎么长出来**：ATG 用 BLIP 问答和固定模板生成属性文本，并刻意避开颜色，保留 age、gender、action、clothing type、hair、glasses 这些更跨模态的属性。CTP 用 SCHP 人体解析去掉衣服区域，保留 body shape 和 pose 相关区域，再用可学习 prompt 表示 contour-aware class text。TRM 负责把 instance-level attribute text 和 class-level contour prompt 融成 image-aware redefined text，最后做 class-level text prototype，指导 RGB 和 IR visual feature 对齐。机制和“属性语义加结构语义”的问题定义是匹配的，但整体偏组合式。

4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。

5. **reviewer 为什么买账**：这篇卖的主要是视角，不是模块精巧度。reviewer 买账点是它把“文本提示”拆成两类可解释语义：属性和轮廓，并用 prototype 把文本变成跨模态锚点。novelty 的来源是“显式语义原型替代弱 token proxy”，而不是 BLIP、SCHP、CLIP 或 Transformer decoder 本身。

**5. Text-to-image Person Search based on Semantic Reorganization.pdf**

1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。

2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。

3. **机制怎么长出来**：SRDA 很直接：先用 MLLM 判断性别，在同一性别内随机选两个 ID，把一个人的上半身和另一个人的下半身横向拼接，生成新 ID，再用微调 BLIP 为新图生成文本。它不追求自然，而追求结构组合多样性。CSRM 则在测试时为文本 query 找 cross-modal k-reciprocal 图像邻居，用 Jaccard 权重把可靠图像特征融合进文本特征，再检索。训练阶段用 CLIP 双流、ID loss、SDM loss 和 image triplet；不加 text triplet，因为 MLLM 文本有噪声。

4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。

5. **reviewer 为什么买账**：这篇卖的是两个务实视角：训练时不迷信 diffusion 真实感，测试时不照搬同模态 re-ranking。机制都很简单，但和观察贴得紧，消融也正好打在 reviewer 会怀疑的点上。novelty 来源不是复杂模型，而是对 TBPS 任务目标的重新理解。

**6. Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf**

1. **触发观察**：遮挡 ReID 里 pose、human parsing 和 occlusion augmentation 都常见，但作者指出两个更具体的失败场景：极端物体遮挡下，局部 pose 错误会污染全局特征；非目标行人遮挡下，遮挡者也有人体骨架，pose 无法区分目标和遮挡行人。作者还用了一个接近人类识别的观察：行人互相遮挡时，背包、伞、衣服纹理这类 distinctive appearance 往往决定识别。

2. **重定义动作**：他们没有把问题讲成“遮挡区域检测不准”，而是拆成“pose misalignment under extreme occlusion”和“identity confusion caused by non-target pedestrian occlusion”。关键词是 pose-patch mapping、texture-aware decoder、non-target pedestrian occlusion、structure and detail dual constraints。这个拆解让方法天然有两个核心动作：结构上精确映射，细节上用纹理区分目标。

3. **机制怎么长出来**：MFE 先在 ViT patch 上做 intra-patch 和 inter-patch 编码，补足 patch 内外层级信息。PPM 不做 pose-to-global，而是把 pose 和 patch 当成两种模态，用 Mahalanobis distance 加 cosine similarity 建 pose-patch 相似矩阵，只保留高置信关键点对应的 patch，从而降低局部 pose 错误的全局污染。TAD 再用 context extraction layer 和 texture extraction layer，以 PPM 过滤后的 pose 特征为约束，通过 cross-attention 强化可区分纹理。pose loss 还约束人体和非人体映射特征不要太相似。机制和两个失败场景绑定得比较紧。

4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。

5. **reviewer 为什么买账**：这篇卖的是“对遮挡失败类型的细分”加“机制一一对应”。如果只说 pose+texture，novelty 会弱；但作者先指出 pose 方法在极端遮挡和行人遮挡下分别为什么失效，再让 PPM 和 TAD 分别处理，reviewer 就更容易接受。真正 novelty 来自问题拆解，而不是 Transformer、pose 或 attention 本身。

**共同套路**

这批论文共同不是先发明一个模块，而是先把已有任务里的默认说法改窄、改具体：事件 ReID 不是 RGB 迁移，VI-ReID 不只是 modality gap，换衣 ReID 不只是加 gait，遮挡 ReID 不只是找可见部位。它们能成稿的关键是把观察变成一句新的失败机制，然后让方法的每个部件都对应这个失败机制，最后用“替换模态、破坏假设、组件拆分、可视化”证明视角本身是对的。
