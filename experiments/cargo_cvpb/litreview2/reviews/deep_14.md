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
session id: 019ef835-b09a-77b1-ba26-267dc1d5d7cf
--------
user
你是 ReID 创新方法论拆解员 14。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(14)==
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我先按项目规则确认目标和论文文件，然后用 `pdftotext` 逐篇抽前 8 页的方法部分。这个任务不需要改文件，我会只读本地 PDF 和必要的公开背景信息。
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
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

exec
/bin/zsh -lc "rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
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

exec
/bin/zsh -lc 'ls -la' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
total 2234352
-rw-r--r--@   1 abdslm  staff   2138698 Jun  9 18:37 'Knowledge and experience' for visible-infrared person re-identification.pdf
drwxr-xr-x@ 170 abdslm  staff      5440 Jun 10 05:08 .
drwxr-xr-x@  58 abdslm  staff      1856 Jun 14 16:59 ..
-rw-r--r--@   1 abdslm  staff  16221269 Jun  9 16:56 3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2319861 Jun  9 21:18 4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   2230068 Jun  9 04:36 A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   6101317 Jun  9 20:10 A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   2087372 Jun  9 03:54 A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf
-rw-r--r--@   1 abdslm  staff   3807986 Jun  9 21:33 A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
-rw-r--r--@   1 abdslm  staff   3976252 Jun  9 20:34 A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   6081043 Jun  9 18:38 ADA framework for unsupervised domain adaptation person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   4335685 Jun  9 16:57 Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   5088061 Jun  9 16:59 Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3486878 Jun  9 18:40 Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   7881200 Jun  9 19:32 Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2566515 Jun  9 22:49 Adversarial perturbation and defense for generalizable person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   9047393 Jun  9 16:53 Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff    743161 Jun  9 03:54 Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2322567 Jun  9 03:53 Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3772906 Jun  9 22:26 Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
-rw-r--r--@   1 abdslm  staff   1867842 Jun  9 18:39 Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   7957363 Jun  9 18:40 Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3197573 Jun 10 02:34 CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3080803 Jun  9 03:57 CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
-rw-r--r--@   1 abdslm  staff   5743549 Jun  9 16:51 CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   2061436 Jun  9 16:58 CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2749258 Jun  9 22:42 CLIP-driven fine-grained mining for text-based person search.pdf
-rw-r--r--@   1 abdslm  staff   4418653 Jun  9 21:11 CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3451491 Jun  9 21:05 CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   6575728 Jun  9 17:06 CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2587966 Jun  9 21:57 CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  55682276 Jun 10 03:49 CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2946189 Jun  9 16:59 Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   1784093 Jun  9 19:25 Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
-rw-r--r--@   1 abdslm  staff    703089 Jun  9 05:17 Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   5899850 Jun  9 17:10 Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
-rw-r--r--@   1 abdslm  staff   1851108 Jun  9 22:04 Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff    962278 Jun  9 16:54 Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   4224547 Jun  9 16:54 Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
-rw-r--r--@   1 abdslm  staff   9515723 Jun 10 00:22 ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   2591562 Jun  9 18:39 Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
-rw-r--r--@   1 abdslm  staff   2924761 Jun  9 18:38 Confidence guided semi-supervised cross-modality person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   1175699 Jun  9 03:59 Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2147219 Jun  9 16:58 Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2132239 Jun  9 17:01 Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
-rw-r--r--@   1 abdslm  staff  32201166 Jun 10 04:05 Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
-rw-r--r--@   1 abdslm  staff   2130341 Jun  9 04:32 Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
-rw-r--r--@   1 abdslm  staff   2581535 Jun  9 18:38 Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
-rw-r--r--@   1 abdslm  staff    439812 Jun  9 05:17 Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
-rw-r--r--@   1 abdslm  staff   2764070 Jun 10 02:02 Cross-modality average precision optimization for visible thermal person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   2000295 Jun  9 17:12 CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   1172148 Jun  9 16:52 DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  19628719 Jun 10 03:37 DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
-rw-r--r--@   1 abdslm  staff   5137308 Jun  9 18:40 Deep intelligent technique for person Re-identification system in surveillance images.pdf
-rw-r--r--@   1 abdslm  staff   3866338 Jun  9 17:04 Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   9078515 Jun  9 17:03 Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3422783 Jun  9 17:13 Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   1399769 Jun  9 04:41 Distribution aligned semantics adaption for lifelong person re-identification.pdf
-rw-r--r--@   1 abdslm  staff    849090 Jun  9 04:34 Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2278771 Jun 10 00:29 DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
-rw-r--r--@   1 abdslm  staff  50760891 Jun  9 04:16 Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   4812496 Jun 10 03:43 Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
-rw-r--r--@   1 abdslm  staff   2299668 Jun  9 04:08 Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
-rw-r--r--@   1 abdslm  staff  10775208 Jun  9 04:35 Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   1802656 Jun  9 16:52 Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3557930 Jun 10 01:14 Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3813398 Jun  9 16:50 ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  28222740 Jun 10 03:59 Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
-rw-r--r--@   1 abdslm  staff   3685507 Jun  9 16:59 Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
-rw-r--r--@   1 abdslm  staff   1295640 Jun  9 03:55 Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   1795094 Jun  9 17:19 FDGReID - Federated Domain Generalization for Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   2088656 Jun  9 16:58 FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2610888 Jun  9 17:11 FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   1908749 Jun  9 16:52 False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   6889434 Jun  9 17:12 Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  11374207 Jun  9 23:50 Focusing on pedestrians like human for clothes changing person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   7046265 Jun 10 00:11 GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   6229550 Jun  9 17:12 GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2369425 Jun  9 16:54 GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3012089 Jun  9 19:51 Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
-rw-r--r--@   1 abdslm  staff   5759992 Jun  9 20:15 Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   9800028 Jun  9 05:02 HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   7622884 Jun  9 17:05 HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   5209950 Jun  9 17:14 Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
-rw-r--r--@   1 abdslm  staff   2661731 Jun  9 16:57 Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff    656161 Jun  9 16:53 Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  12380803 Jun 10 01:07 Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   6934377 Jun  9 23:35 Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
-rw-r--r--@   1 abdslm  staff    565909 Jun  9 03:58 Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   4500309 Jun  9 20:04 Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
-rw-r--r--@   1 abdslm  staff   4757438 Jun  9 17:13 Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
-rw-r--r--@   1 abdslm  staff   2867313 Jun  9 17:01 InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2378036 Jun  9 19:17 Instant pose extraction based on mask transformer for occluded person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3171668 Jun  9 16:52 Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
-rw-r--r--@   1 abdslm  staff   4656109 Jun  9 17:03 Internal-External Context Interaction Network for Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3881162 Jun  9 17:07 Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
-rw-r--r--@   1 abdslm  staff   3645169 Jun  9 16:55 Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  20386649 Jun 10 02:57 Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
-rw-r--r--@   1 abdslm  staff   2788783 Jun  9 19:44 Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   4367996 Jun 10 04:13 Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
-rw-r--r--@   1 abdslm  staff   8213628 Jun 10 00:58 Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
-rw-r--r--@   1 abdslm  staff   2222039 Jun  9 05:09 Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
-rw-r--r--@   1 abdslm  staff  19186822 Jun 10 03:03 Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2127244 Jun  9 03:59 MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2392069 Jun  9 17:01 Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2939259 Jun 10 00:05 Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   1800297 Jun  9 17:11 Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   5078550 Jun 10 02:41 Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
-rw-r--r--@   1 abdslm  staff   1923224 Jun  9 04:29 Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
-rw-r--r--@   1 abdslm  staff  13092735 Jun 10 03:10 MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff    994262 Jun  9 16:52 Multi Queue for Unsupervised Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   1364884 Jun 10 04:22 Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   1694532 Jun  9 16:54 Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2429090 Jun 10 01:27 Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   6621795 Jun  9 16:58 Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  46435711 Jun 10 02:30 Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2174615 Jun  9 23:43 Multi-feature balanced network for clothes-changing person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3279786 Jun  9 20:39 Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3455183 Jun  9 18:40 Multi-year long-term person re-identification using gait and HAR features.pdf
-rw-r--r--@   1 abdslm  staff   8010784 Jun  9 17:07 Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3124039 Jun  9 17:06 Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   1571573 Jun  9 20:23 Occluded person Re-Identification with noise injection.pdf
-rw-r--r--@   1 abdslm  staff  14486732 Jun  9 17:08 Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3270791 Jun 10 02:51 Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf
-rw-r--r--@   1 abdslm  staff  10664998 Jun  9 17:02 Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   4477107 Jun  9 17:05 Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   5316126 Jun  9 16:55 Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  12508701 Jun  9 18:38 Privacy preserving person re-identification via anonymizing diffusion model.pdf
-rw-r--r--@   1 abdslm  staff   9148464 Jun  9 17:10 Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
-rw-r--r--@   1 abdslm  staff    572740 Jun  9 03:58 Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   1808799 Jun  9 05:17 Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3532145 Jun  9 17:00 RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2996121 Jun  9 20:58 RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
-rw-r--r--@   1 abdslm  staff   1920912 Jun  9 16:50 Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff    379493 Jun  9 05:17 Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3584424 Jun  9 20:28 Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
-rw-r--r--@   1 abdslm  staff   6836471 Jun  9 17:02 SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
-rw-r--r--@   1 abdslm  staff   3667918 Jun  9 04:09 See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   4415212 Jun  9 17:07 Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   5665207 Jun  9 16:51 Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff    569971 Jun  9 16:53 Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
-rw-r--r--@   1 abdslm  staff   1772957 Jun  9 04:02 Shape-centered representation learning for visible-infrared person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   4249881 Jun 10 03:15 Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3154098 Jun  9 04:14 Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
-rw-r--r--@   1 abdslm  staff   4610796 Jun  9 17:09 Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   3506760 Jun  9 19:59 Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3375564 Jun  9 21:49 Spectrum-guided feature enhancement network for event person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3506716 Jun 10 00:47 Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
-rw-r--r--@   1 abdslm  staff  12147166 Jun  9 17:10 Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
-rw-r--r--@   1 abdslm  staff   5475373 Jun  9 17:13 TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
-rw-r--r--@   1 abdslm  staff   4657084 Jun  9 18:38 TP-LReID - Lifelong person re-identification using text prompts.pdf
-rw-r--r--@   1 abdslm  staff  22113663 Jun 10 03:54 Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2844163 Jun  9 22:33 Text-to-image Person Search based on Semantic Reorganization.pdf
-rw-r--r--@   1 abdslm  staff   9589032 Jun  9 18:40 Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   2596722 Jun  9 03:53 Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff   2329297 Jun  9 18:38 Two-stage knowledge distillation for visible-infrared person re-identification.pdf
-rw-r--r--@   1 abdslm  staff   3198767 Jun  9 22:11 UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
-rw-r--r--@   1 abdslm  staff   2417519 Jun  9 04:15 UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
-rw-r--r--@   1 abdslm  staff  19599969 Jun 10 02:24 Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
-rw-r--r--@   1 abdslm  staff   6453774 Jun  9 21:43 Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
-rw-r--r--@   1 abdslm  staff  92215394 Jun 10 02:15 Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
-rw-r--r--@   1 abdslm  staff   4366073 Jun  9 17:05 Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
-rw-r--r--@   1 abdslm  staff  71816003 Jun 10 03:22 Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
-rw-r--r--@   1 abdslm  staff  11317872 Jun  9 23:56 Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
-rw-r--r--@   1 abdslm  staff   1247654 Jun  9 16:53 Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff  21972539 Jun 10 02:46 Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
-rw-r--r--@   1 abdslm  staff   7673415 Jun  9 17:00 Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
-rw-r--r--@   1 abdslm  staff    932240 Jun 10 04:19 Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
-rw-r--r--@   1 abdslm  staff   1699143 Jun  9 03:59 When Large Vision-Language Models Meet Person Re-Identification.pdf
-rw-r--r--@   1 abdslm  staff    114226 Jun 10 05:08 manifest.csv

codex
本目录里没有 `GOAL.md`、`CLAUDE.md` 或 `STATUS.md`，只有论文库本身。我会以你这条消息给出的拆解目标作为当前目标继续。
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
exec
/bin/zsh -lc "test -f /Users/abdslm/Desktop/reid-paper/GOAL.md && sed -n '1,160p' /Users/abdslm/Desktop/reid-paper/GOAL.md || true" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "test -f /Users/abdslm/Desktop/reid-paper/CLAUDE.md && sed -n '1,200p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md || true" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "test -f /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md && sed -n '1,200p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md || true" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
我已经读到全局目标，当前任务和“读文献、做差距分析”的阶段一致。下一步先抽每篇前 8 页的目录式线索，确定方法部分标题和边界，再读对应段落。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|conclusion|geospatial|temporal|graph'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3:GSTNET: A GEOSPATIAL-TEMPORAL GRAPH NETWORK FOR GROUP PERSON
30:ABSTRACT
32:Index Terms— Geospatial-temporal graph network, group person re-identification, reachability constraint
33:1. INTRODUCTION
34:Geospatial-temporal Group Person Re-identification (Gst-GReID)
35:can integrate geospatial constraint and spatio-temporal contextual
38:complex scenarios. The existing methods for person re-identification
45:variations in group size and layout, some studies employed graph
56:Geospatial-temporal Group Person Re-identification (Gst-GReID)
57:can integrate geospatial constraint and spatio-temporal contextual information. By modeling the appearance of groups across
59:retrieval in complex scenarios. The existing methods primarily
60:rely on static distribution of statistics across time-interval cameras and ignore geospatial-temporal reachability, resulting in poor
62:propose the Geospatial-temporal Graph Network (GstNet). GstNet has two core designs: the Geospatial-temporal Reachability
63:Module (GstRM) and the Gated Graph-MLP (GGM). GstRM imposes a geospatial-temporal reachability constraint on graph edges,
64:suppressing geospatial-temporal unreachable connections, thereby
67:spans and across regions, enhancing identity-discriminative representations and suppressing interfering samples. Extensive experimental results on the BRT and SYSU-Group datasets indicate that
68:our GstNet method outperforms existing state-of-the-art methods.
74:Previous Method
82:Fig. 1. Illustration of the proposed motivation. Unlike previous methods that overlook geospatial-temporal reachability constraint (e.g., traversing 20 km in 5 s is infeasible under the maximum traffic speed), we introduce such constraint to filter out groups
85:is spatio-temporal modeling-based person re-identification. Some
87:3D CNN and attention mechanisms to characterize cross-frame temporal sequences and achieve multi-scale temporal modeling and spatiotemporal fusion [15, 16, 17, 18].
88:The accuracy of existing methods is limited by two major factors. First, most group person re-identification approaches remain
90:Over long time spans and across regions, such methods struggle to
91:leverage the geospatial-temporal reachability constraint. Second,
92:although spatio-temporal modeling methods introduce temporal
95:the geospatial-temporal reachability constraint, which is defined
99:illustrated in Fig. 1, previous methods overlook the reachability
117:GReID Graph Building
119:Geospatial-Temporal Feature Extraction
136:Temporal Features
149:(a) Overview of Geospatial-temporal Graph Network (GstNet)
152:Graph-MLP
192:(b) Geospatial-temporal Reachability Module(GstRM)
194:(c) Gated Graph-MLP(GGM)
197:information is processed by GstRM (b) to produce a feature fusion matrix constrained by geospatial-temporal reachability. Step 2: a grouplevel graph is constructed by combining visual features with the feature fusion matrix derived from reachability modeling. Step 3: the graph
200:To address this issue, we propose Geospatial-temporal Graph
201:Network (GstNet) for group person re-identification. This framework integrates geospatial-temporal information with group appearance features to enhance matching performance over long
202:time spans and across regions. GstNet comprises two core modules: Geospatial-temporal Reachability Module (GstRM), which
203:imposes geospatial-temporal reachability constraint during graph
205:Graph-MLP (GGM), which achieves selective neighborhood aggregation during feature propagation to suppress the mismatches
209:regions, we propose GstNet, which integrates geospatial-temporal
211:through constrained graph construction and gated propagation.
213:graph construction, we propose GstRM, which defines reachability
214:based on geospatial-temporal information and speed thresholds, and
220:2. METHODOLOGY
221:2.1. Overall Framework
226:graph is constructed with groups as nodes, and reachability penalties
228:Then, the graph is processed by GGM for gated feature propagation,
231:2.2. Geospatial-temporal Reachability Module (GstRM)
236:between timestamps, which is physically implausible. We propose the Geospatial-temporal Reachability Module (GstRM). This
241:module introduces a speed constraint with a slack margin to dynamically modulate edge strengths, thereby suppressing geospatialtemporal infeasible links. We define the time difference and the
242:geographic distance. For any two group nodes (i, j), the time difference Tij = |ti − tj |, and the geographical distance Dij = D[i][j].
243:Based on the geographical distance Dij and the time difference Tij ,
245:geospatial-temporal dimension:
261:After obtaining the geospatial-temporal priors, we model
264:MLP(Dij , Tij ), where ϕij denotes the geospatial-temporal similarity potential. To modulate the baseline pairwise compatibility
273:When the slack margin is large, γij approaches the upper bound,
275:approaches zero, γij approaches the lower bound, thereby suppressing the weight of this edge.
293:The graph model G(F, E) is defined by a node feature matrix
298:2.3. Gated Graph-MLP (GGM)
299:In geospatial-temporal group re-identification (Gst-GReID), groups
300:that are adjacent in geographic location and time are not necessarily of the same identity; indiscriminate propagation of neighborhood
304:Gated Graph-MLP (GGM), which augments Graph-MLP [21] with
328:of node embeddings to capture graph structure, we introduce a structural contrastive loss (ncontrast Loss). This loss exploits the structural information encoded in the edge feature tensor E ∈ RN ×N ×de .
352:3. EXPERIMENTS
353:3.1. Experimental Setup
354:Datasets. We evaluate our approach on two group person reidentification datasets: BRT [22] and SYSU-Group [23]. The BRT
360:is available without inter-camera distance, we utilize temporal cues
362:Implementation Details. We evaluate the effectiveness of the proposed approach using widely adopted retrieval metrics, including
370:Methods
462:Baseline(Graph-MLP) [21]
489:Table 2. Ablation Study results on BRT dataset.
518:(a) Baseline(Graph-MLP) (b) Baseline+GstRM
536:71.06% Rank-1 and 68.81% mAP, demonstrating their complementarity. Since the SYSU-Group dataset provides only temporal annotations without inter-camera distance information, a complete ablation of GstRM is infeasible; therefore, we only report the baseline
539:3.2. Comparison with State-of-the-Art Methods
544:spanning group-level context modeling (MACG [5]), classical graph
546:uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
548:with Rank-1 scores of 71.06% and 96.11%, surpassing the secondbest methods by 1.89% and 0.68%, respectively. Compared with
549:approaches that rely on static distribution of statistics across timeinterval cameras (e.g., St-ReID [13] and InSTD [14]), the proposed GstRM and GGM modules enable GstNet to model dynamic
550:geospatial-temporal reachability more effectively, thereby improving geospatial-temporal group person re-identification performance.
552:To evaluate the discriminative power of GstNet’s features, we visualize the feature distributions of the Baseline (Graph-MLP), Baseline+GstRM, and GstNet using t-SNE, as shown in Fig. 3. Different
560:3.3. Ablation Study
561:We conducted ablation studies on the BRT dataset, and the results
563:to 58.60% Rank-1 and 55.31% mAP, indicating that the geospatialtemporal reachability constraint suppresses spurious edges. With
566:4. CONCLUSION
567:In this paper, we present a novel approach GstNet. Unlike previous
568:methods that rely on static distribution of statistics across timeinterval cameras, GstNet jointly models the Geospatial-temporal
569:Reachability Module and the Gated Graph-Mlp, learning dynamic
570:geospatial-temporal reachability constraint and their interaction
574:with state-of-the-art methods.
581:Autonomous Region (Geospatial-temporal Feature Fusion for Group
596:[3] Z. Huang, Z. Wang, W. Hu, C. Lin, and S. Satoh, “Dotgnn: Domain-transferred graph neural network for group reidentification,” in Proceedings of the 27th ACM International
599:“Group re-identification with group context graph neural networks,” IEEE Transactions on Multimedia, vol. 23, pp. 2614–
602:“Learning multi-attention context graph for group-based reidentification,” IEEE Transactions on Pattern Analysis and
627:[13] G. Wang, J. Lai, P. Huang, and X. Xie, “Spatial-temporal person re-identification,” in Proceedings of the AAAI Conference
630:“Learning instance-level spatial-temporal patterns for person
638:Learning efficient spatial-temporal representation for video
643:“Multi-scale spatio-temporal feature adaptive aggregation for
646:temporal interaction for video-based person re-identification,”
656:“Graph-mlp: Node classification without message passing in
657:graph,” arXiv preprint arXiv:2106.04051, 2021.
664:with graph convolutional networks,” in International Conference on Learning Representations, 2017.
666:B. Yoshua, “Graph attention networks,” in International Conference on Learning Representations, 2018.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|vision|language|vlm|pseudo|knowledge'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
6:Harnessing Knowledge From Pretrained VLMs
9:Abstract—Person search is a unified task that includes the
14:to obtain accurate pseudo-IDs from low-quality pseudo-boxes,
16:FMUPS, a novel method that leverages semantic information to
17:produce reliable pseudo-labels. Semantic representations, particularly from vision-language models (VLMs), provide clear
18:and interpretable guidance, reducing noise caused by background disturbances during pseudo-label extraction. Despite
19:their advantages, VLM-generated pseudo-boxes often suffer from
26:capabilities of VLMs to generate pseudo-IDs based on our
28:knowledge of person search. The experimental results on two
30:effectiveness of our method; in particular, we outperform some
31:previous fully and weakly supervised (WS) methods.
33:unsupervised (US) learning, vision-language models (VLMs).
73:supervised (WS) methods have been proposed [9], [10], [11],
82:in the fact that accurate re-ID pseudo-labels become more difficult to achieve due to the low-quality pseudo-boxes. Therefore,
83:our investigation focuses on generating high-quality pseudolabels, i.e., pseudo-boxes and pseudo-IDs, considering the
85:to generate pseudo-boxes and pseudo-IDs using state-of-theart pedestrian detectors (e.g., FeatComp [16]) and clustering
86:algorithms. However, on the one hand, these pseudo-boxes
92:methods, such as DBSCAN, have shown limited effectiveness
93:when handling noisy pseudo-boxes, resulting in only 31.56
98:to use vision-language models (VLMs), in which the natural
99:language offers a complementary pathway, providing explicit
100:and unambiguous semantics that are helpful for pseudo-label
102:hand, VLMs serve as a bridge between visual features and
103:textual semantics, facilitating the integration of semanticlevel understanding into pseudo-label generation; on the other
104:hand, VLMs exhibit remarkable generalization capabilities
105:across various vision tasks [17], [18]. However, achieving
106:high-quality pseudo-labels directly through VLMs remains
107:nontrivial. For example, while SEEM-generated pseudo-boxes
121:to produce more accurate and reliable pseudo-IDs.
122:3) Experiments are conducted on PRW and CUHK-SYSU
125:close to and even surpasses some FS and WS methods.
129:As this article develops person search methods using cues
143:negatively impact the quality of pseudo-IDs. To mitigate
150:pseudo-boxes by making use of the alignment between image
153:each instance can be assigned as a pseudo-ID, whose text
155:primary pseudo-IDs are further refined by the prior knowledge
164:2) We introduce a novel framework leveraging robust
165:semantics of VLMs to generate and refine pseudolabels for person search. Specifically, we propose an
167:classification noise in SEEM-generated pseudo-boxes
170:person in the CLIP+SAM method.
174:methods, which optimize pedestrian detection and re-ID tasks
176:methods, which unify these subtasks by jointly optimizing
179:[38]. Broadly, current person search methods can be classified
181:FS and WS methods.
185:[37], [38]. Xiao et al. [2] pioneer a unified framework that
188:method to harmonize the divergent optimization goals of
189:detection and re-ID. In addition, recent approaches [36], [37]
193:identity labels, this group of approaches utilizes only bounding
199:3) Unsupervised: Unlike the aforementioned methods, US
203:person search methods [40], which rely on labeled sourcedomain data together with unlabeled target-domain images
205:any labeled data from any domain. While no US methods
210:detection (UMPD) method that removes the necessity for
216:TIAN et al.: HARNESSING KNOWLEDGE FROM PRETRAINED VLMs FOR US PERSON SEARCH
220:to refine the re-ID pseudo-label generation process. However,
223:performance and the presence of noisy pseudo-boxes can
232:models (SAM [22] and SEEM [19]), VLMs (CLIP [23]),
233:and human-centric approaches (SOLIDER [43], UniHCP [44],
238:leading to its application in diverse vision tasks, including
245:integrating diverse datasets to build comprehensive humanfocused models. These approaches not only provide robust
255:US person search method, termed FMUPS-S1, followed by an
256:explanation of the proposed CLIP ID labeler. Due to the noise
257:introduced by pseudo-boxes, we present in detail our antibbox-noise re-ID loss. In addition to FMUPS-S1, we present
258:another implementation: a two-stage approach, FMUPS-S2.
262:The VLMs are used to provide high-quality pseudo-labels
268:Fig. 2. Overview of our one-stage method FMUPS-S1. The pipeline consists
270:Specifically, SEEM is employed for generating pseudo-boxes, while the
271:CLIP ID labeler is utilized to provide pseudo-IDs. Our novel anti-bboxnoise re-ID loss effectively reduces the negative effect of both classification
275:obtain all person masks by utilizing the person-related prompts
276:(e.g., “person”). We then derive the pseudo-box y
292:to generate and further refine pseudo-IDs. Concurrently, in
301:In order to reduce the negative impact of noisy pseudo-boxes
307:induced by pseudo-boxes. Further elaboration on the CLIP
312:pseudo-IDs for each sample as shown in Fig. 3. It comprises
322:feature space. Text prompts are generated from predefined sets of attributes, and pseudo-IDs are assigned based on the highest similarity scores between text
323:embeddings D and I b . These pseudo-IDs are subsequently refined through a denoising procedure that utilizes prior knowledge.
326:leveraging the CLIP text–image matching principle to generate pseudo-IDs. These pseudo-IDs are subsequently refined
327:through the application of prior knowledge that each individual
329:of the generated pseudo-IDs.
361:3) Generation and Denoising of Pseudo-IDs: Consider a
365:pseudo-ID of a given box embedding Ii with the following
386:into the pseudo-IDs. Given the prior that “a person can only
387:appear once in a single image,” we can refine the pseudo-IDs
390:same pseudo-ID “d.” If si,d < s j,d , then I bj , which exhibits
391:the highest similarity, retains the pseudo-ID “d.” Conversely,
392:the pseudo-ID for Iib , which has a lower similarity score, is
395:are resolved (i.e., all pseudo-IDs are assigned unique values).
418:updated. Correspondingly, the pseudo-IDs are also updated
423:TIAN et al.: HARNESSING KNOWLEDGE FROM PRETRAINED VLMs FOR US PERSON SEARCH
427:Algorithm 1 Generation and Denoising of Pseudo-IDs inCLIP
435:correspond to the same pseudo-ID.
440:As shown in Fig. 1, pseudo-boxes can introduce two types
445:pseudo-boxes often introduce redundant background information around the target person. Consequently, we align the
448:background information. For classification noise, the pseudoboxes may contain objects other than people. Therefore, we
458:that share identical pseudo-labels y
536:E. Two-Stage Method
537:As discussed in Section II-A, person search methods can
538:be classified into one-stage and two-stage methods depending
540:framework. One-stage methods optimize both subtasks within
541:a single framework, while two-stage methods optimize them
554:in data usage across US, FS, and WS methods. Here, Bbox
561:Unless specified otherwise, our experiments use the default
564:search methods, we employ the mAP and Top-k scores.
567:Fig. 5. Overview of our two-stage method FMUPS-S2. The confidence
572:separately. Although one-stage methods are more efficient,
573:they generally exhibit lower performance compared to twostage methods [3], [7]. Therefore, in addition to the one-stage
574:method FMUPS-S1, we also provide a two-stage method,
580:and employs noisy pseudo-boxes for the subsequent reID stage.
597:metrics, and implementation details. We then provide a comprehensive analysis of our method, followed by a comparative
598:evaluation of state-of-the-art person search methods across
599:various experimental setups.
601:We propose a US person search method, i.e., FMUPS.
608:[51], with K set to 9. Our method is instantiated in two
609:variants: FMUPS-S1, a one-stage method, and FMUPS-S2,
610:a two-stage method. The FMUPS-S1 model consists of two
617:During training, all experiments are conducted on a single
629:In this section, we conduct several analytical experiments
630:to better understand our proposed method.
631:1) Comparison With Baselines: Our approach is the first
632:purely US method in the field of person search. Due to the
633:absence of directly comparable US methods, we establish
634:several baselines that leverage pseudo-labels generated by offthe-shelf methods for training, as follows.
637:Fig. 1(c)], to generate pseudo-boxes, and employ the
642:TIAN et al.: HARNESSING KNOWLEDGE FROM PRETRAINED VLMs FOR US PERSON SEARCH
654:the blue line represent the mAP values of different methods, while the bar
655:chart illustrates the Top-k performance of these methods, with k set to 1, 5,
661:method, to produce pseudo-IDs. In the following sections, this method will be referred to as the BASELINE.
663:pseudo-boxes for pedestrians are provided by a frozen
665:these pseudo-boxes and masks, along with the pseudoIDs extracted by DBSCAN, to train a person search
667:3) SEEM+SPCL: SPCL [53] is a conventional clustering method that has been demonstrated to outperform
669:widely used in recent WS person search methods [9],
672:We compare our FMUPS-S1 method with various baselines
676:highlights the effectiveness of our proposed approach. In addition, SEEM+DBSCAN shows a significant improvement of
678:our FMUPS-S1 (SEEM+CLIP ID labeler) method surpasses
680:These results indicate that VLMs, such as SEEM and CLIP,
682:utilizing VLMs, our framework achieves the shortest training
683:time and the least RAM among all methods. This efficiency
685:generates and denoises pseudo-IDs during training. In contrast, clustering methods like SPCL and DBSCAN require
689:Fig. 6, shows that both our one-stage method, FMUPS-S1, and
690:two-stage method, FMUPS-S2, outperform all baselines. The
692:These findings emphasize the robustness of our framework
695:2) Ablation Study on Anti-Bbox-Noise Re-ID Loss: The
712:of Lw f f in to our method. CUHK-SYSU shows similar
762:without prior knowledge (w/ prior versus w/o prior) of the CLIP ID labeler.
767:our method consistently surpasses the SEEM+SPCL baseline,
775:introduced by pseudo-boxes. As shown in Fig. 7(a), the performance without confidence scores is 20.97 pp with respect
780:a prior in the CLIP ID labeler is designed to denoise pseudoIDs and enhance their reliability. As illustrated in Fig. 7(b),
781:incorporating prior knowledge significantly improves the performance, with the mAP increasing from approximately 20.65
785:tomAP highlights the prior’s effectiveness in refining pseudoIDs and reducing ambiguities, further validating its critical role
786:in enhancing the overall framework.
788:a pivotal role in generating pseudo-IDs for training the re-ID
790:with different clustering methods, while maintaining other
793:SPCL [53], a widely used clustering method in WS person
806:9) Performance of Our Method at Different Crowd Densities: We analyze FMUPS-S1 under varying crowd densities

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|heterogeneous|generative|token|distance|recovery'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:Heterogeneous Generative Tokens and
6:Distance-Aware Recovery Network for Occluded
10:Abstract— In real-world surveillance scenarios, person
14:features when they are occluded. Although existing methods
17:due to the introduction of background noises. To solve this
18:problem, we propose Heterogeneous Generative Tokens and
19:Distance-aware Recovery (HGTDR) network, which aims to
30:proposed Adaptive Heterogeneous Semantic Token Generator
33:the features of occluded regions, the Distance-based Feature
34:Recovery (DFR) module is designed. The module calculates the
35:planar Euclidean distance of heterogeneous tokens and adaptively
38:person descriptors. Extensive experiments on several challenging
40:our proposed HGTDR network achieves superior performance
41:and outperforms various state-of-the-art methods.
43:semantic token, feature recovery.
93:Miao et al. [17] proposed a Re-ID framework, which extracts
95:al. [18] proposed a novel method, named Pirt, to obtain robust
97:masks. Recently, Yan et al. [16] proposed an innovative
100:However, most of these methods rely on additional pose
103:researchers have begun to explore methods that do not rely on
118:Fig. 1. (a) The performance of the previous methods severely decreases when
122:hand-crafted strip-based methods, which may introduce noise in the red box,
123:our method can adaptively extract heterogeneous person local features with
128:ViT and proposed many transformer-based models [8], [20],
130:information into a predefined part token, thereby achieving disentangling of person image. However, most of the
131:methods rely solely on complex semantics for decoupling,
138:another occluded Re-ID method [10], [23], [24], [25], [26],
140:Some of them [24], [26] used generative adversarial networks
147:this set. Recently, Wang et al. [10] proposed a Feature
150:into account the issue of neighbor weight for occlusion recovery, which leads to less robust recovered features. Alongside
164:At the same time, in order to effectively deal with the challenges in Fig. 1(b), we propose a multi-effect Heterogeneous
165:Generative Token and Distance-aware Recovery (HGTDR)
172:any additional part tokens and auxiliary networks. Firstly,
173:a Semantic Patch Generator (SPG) is proposed, which uses
179:Adaptive Heterogeneous Semantic Token Generator (AHSTG)
180:to obtain identity-related fine-frained features. Guided by the
185:and weak-response heterogeneous tokens, which effectively
187:scenarios, the heterogeneous tokens may struggle to achieve
189:a Distance-based Feature Recovery (DFR) module, which
196:(1) A new Semantic Patch Generator (SPG) is proposed
200:(2) We propose a flexible Adaptive Heterogeneous Semantic Token Generator (AHSTG), which takes into account
203:generate heterogeneous tokens with different responses to
211:Fig. 2. The pipeline of our proposed HGTDR. The framework mainly consists of a holistic stream (Section III-B) and a part stream (Section III-C). Our
212:proposed SPG, AHSTG, and DFR modules are in the part stream. Here, the patch embedding layer and shallow transformer blocks act as base module and
213:are not expanded in detail anymore. ‘⊕’ represents the concatenation operation. ‘CLS’ represents the class token. In the holistic stream, the sequence below
214:CLS is the updated patch tokens by patch embeddings Pe . The blue arrows represent the holistic stream, the green arrows represent the part stream, and the
218:fine-grained tokens with different scales and shapes.
219:(3) A novel Distance-based Feature Recovery (DFR) module is designed to automatically mine implicit information
221:methods, we consider the contribution to occlusion recovery
222:from the perspective of distance, and can adaptively recover
224:(4) Extensive experiments on two authoritative occluded
226:of our method. In addition, we confirmed that our method has
237:[34]. For example, Zhou et al. [35] proposed an effective
241:viewpoint misalignment, Zhang et al. [3] proposed a View
242:Confusion Feature Learning (VCFL) method to learn the
244:mechanism. In the attention-based methods, Zhang et al. [28]
245:proposed a plug-and-play Relational-aware Global Attention
247:focusing identity-related regions. Recently, transformer [36]
252:achieved remarkable performance. Although these methods
258:methods using fine-grained features [1], [7], [37], [38], [39],
262:Sun et al. [1] proposed a Part-based Convolutional Baseline
263:(PCB) method combined with RPP to obtain fine-grained local
265:Re-ID. Tan et al. [40] proposed a Continuous Batch DropBlock
268:the performance improvement is significant, these methods
270:To solve the above problems, He et al. [37] proposed a Deep
271:Spatial feature Reconstruction (DSR) method to avoid the error
272:in matching images at different scales. Sun et al. [7] proposed a
275:methods usually require the manual definition of the scale
277:of the model. In contrast, our proposed method is general
279:heterogeneous features, but also can be applied to diverse
293:al. [11] proposed a Pose-Guided Feature Alignment (PGFA)
296:al. [6] proposed a Pose-guided Visible Part Matching (PVPM)
297:method, which learns the visible local features guided by a
298:pose estimation model. Wang et al. [13] proposed a HighOrder Re-ID (HoReID) network that integrates the high-order
300:a graph network. However, these methods all require additional pre-trained networks, which may lead to domain bias
303:proposed several attention-based methods. For example, Ye et
304:al. [42] proposed a novel Dynamic Dual Attention Aggregation
305:(DDAG) learning method to extract discriminative partial
306:aggregation features. Xia et al. [43] proposed a novel Attention
309:attention. Li et al. [44] proposed a novel approach, called
311:body parts or regions. Although these methods have achieved
315:D. Feature Recovery
319:of the pedestrian. Some previous methods [24], [26] combine
320:the generative adversarial network to complete the human
322:matching. These approaches require massive images of the
326:Later, some recovery methods [10], [23], [25] based on feature
330:To fully leverage the benefits of global information interaction in the transformer, Xu et al. [25] proposed a Feature
331:Recovery Transformer (FRT) model, which learns the available
335:al. [10] proposed a new framework named Feature Completion Transformer (FCFormer), which enriches the occluded
344:However, many of the above methods either relied on additional pre-trained networks or failed to consider the effect of
345:neighborhood weight on recovery quality.
346:Different from the above methods, our proposed method
349:distance perception. Our work reasonably allocates weights
350:through the calculated distance, effectively simulates the real
353:method is able to escape the dependence on additional networks without sacrificing performance, while fully considering
355:recovery.
357:In this section, we introduce the proposed Heterogeneous
358:Generative Tokens and Distance-aware Recovery (HGTDR)
359:framework in detail. In Section III-A, we first give a concise
365:The proposed HGTDR network is mainly composed of a
381:K discriminative heterogeneous tokens with rich semantics.
382:These tokens are then fed into the decoder to enhance the
383:network’s robustness. We assess their distances and insert
384:a recovery token between them to dynamically recover the
405:we prepose a learnable global token E g into Pe and attach
415:3. The global token E g ∈ R1×D will be updated to the class
416:token (denoted as ‘CLS’ in Fig. 2 and Fig. 3) in subsequent
421:(one global class token f hg ) and discriminative local semantics
422:(N patch tokens). The class token is treated as a global feature
426:∈ R N ×D . Whereafter, following the methods [10] and [45],
497:Illustration of the proposed SPG. The black rectangular wire
500:cube ‘CLS’ represents the global aggregation token of the holistic stream in
503:distance between a and b. m represents the similarity margin.
519:Semantic Patch Generator (SPG), the Adaptive Heterogeneous
520:Semantic Token Generator (AHSTG), and the Distance-based
521:Feature Recovery module (DFR), as shown in Fig. 2. The SPG,
531:subsequent feature aggregation and recovery. The SPG is
547:Fig. 4. Illustration of the proposed AHSTG. Avgpooling represents the
548:adaptive global average pooling. ‘CLS’ represents the class token of the
555:represent the selected cluster centers in heterogeneous semantic tokens. The
556:number of heterogeneous tokens K in the figure is set to 4 for simplicity.
558:Fig. 5. Illustration of the proposed DFR. The index is from the AHSTG
560:feature (i.e. CLS token) of the holistic stream. After the Euclidean distance
563:the heterogeneous semantic tokens in the presence of occlusion (between the
564:red arrows), a recovery token is inserted to recover the person features. The
586:semantics should be highly correlated with each other, and
618:in Fig. 3, we merge the CLS token of the holistic stream
645:where C L S i ∈R1×D refers to the CLS token in the ith transformer layer at the current network depth, which is
673:2) Adaptive Heterogeneous Semantic Token Generator:
678:extracting identity-related local features is a core challenge.
679:Traditional strip-based methods often rely on hand-crafted
694:and highlights the identity-related regions. Specifically, we first
696:their importance. Since there is no global token for the part
707:DIS takes P̂ 2 , f cls and the class token as input. It uses the
726:feature. ‘CLS’ represents the class token of holistic stream at
732:identity-related patches and a lower value vice versa. Similarly,
776:we take their indexes as the heterogeneous tokens’ indexes
779:and values V into the transformer blocks to adaptively generate heterogeneous tokens. In the generation process, we also
782:generation of heterogeneous tokens. Formally,
795:operations such as FFN. Here, the identity-related patches are
799:with different shapes and sizes. For a heterogeneous token,
802:erated heterogeneous tokens f t (denoted as { f ti ∈ R1×D }i=1
809:3) Distance-Based Feature Recovery Module: Even though
813:In other words, the learned heterogeneous features still suffer
814:from the lack of pedestrian information. To address this problem, we propose a Distance-based Feature Recovery (DFR)
823:we introduce a distance factor into the DFR module to achieve
824:more accurate feature recovery. Our core idea is that when
834:related information to the occluded region, and thus are more
836:The distance-based contribution consideration makes the
838:occluded regions. To be specific, after obtaining the heterogeneous tokens f t and its indexes I , we project the index I
839:into a 2D space to calculate the Euclidean distance between
840:adjacent indexes, as shown in Fig. 5. If the distance is
843:regions, we subtly insert a recovery token between the two
844:heterogeneous tokens. The insertion of the recovery token not
848:region between the i-th token f ti (index: Ii ) and the (i + 1)th token f ti (index: Ii+1 ) is occluded, we insert a recovery
849:token f tr ∈R1×D between them with index I r = Random(Ii ,
850:Ii+1 ). Furthermore, we calculate the planar Euclidean distance
852:the contribution weights of heterogeneous tokens. The weights
862:where d(Ii , I r ) represents the planar Euclidean distance
863:function between Ii and I r . Since the greater the distance,
865:function to the negative values of the distance. It ensures a
871:heterogeneous token f t . Formally,
876:Then, we input the updated tokens f t and the inserted
877:recovery token f tr into the decoder to complete the missing
880:Following several transformer layers of the distance-based
881:recovery decoder, we acquire the ultimate completed features:
887:which is used to provide global semantics and guide the feature recovery. I nser t (a, b, c) are used to select the insertion
889:inputs are the updated tokens f t , the recovery token f tr and its
891:local features, we split the final recovery features and remove
895:, i.e. K heterogeneous
896:tokens and one recovery token). Additionally, we also apply
907:adjust the weight of a token in the loss function according to its
908:comprehensive score. Since these tokens have different scores,
910:a given token is small, the network should be less influenced

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|hierarchical|fusion|local-aware|local aware|transformer'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
10:Hierarchical fusion and local-aware transformer for occluded person
29:Transformer
30:Feature fusion
31:Local-aware
34:Occluded person re-identiﬁcation (ReID) is intended to address the problem of matching pedestrians when images of individuals are partially occluded. Recently, Transformer-based methods for occluded person ReID have
35:received considerable attention. However, although existing methods have achieved promising results, most of
37:suﬃciently emphasize the identity information in critical regions. Furthermore, those methods often lack suﬃcient capability to extract ﬁne-grained local features, making it diﬃcult to fully explore the identity information
38:embedded in various body parts. To resolve the mentioned problems, we propose a Hierarchical Fusion and
39:Local-aware Transformer (HFLAT) for occluded person ReID. Speciﬁcally, we ﬁrstly design a feature hierarchical
40:fusion module that hierarchizes and fuses the patch feature vector sequence according to the relative importance
44:of interactions between the features using a local-aware multi-head attention mechanism, increasing the model’s
45:ability to obtain ﬁne-grained local features. Experimental results on the Occluded-DukeMTMC, Occluded-ReID,
47:method achieves the Rank-1 accuracy of 79.6% and 89.8%, respectively, and the mAP of 64.7% and 84.9%,
48:respectively. On the Market1501 and DukeMTMC-ReID datasets, our method achieves the Rank-1 accuracy of
51:1. Introduction
64:son ReID methods have been presented. Existing methods can be broadly
65:categorized into external auxiliary information-based methods Gao et al.
66:(2020), Huang et al. (2020), Miao et al. (2019) and Transformer-based
67:methods Li et al. (2021), Lin et al. (2024), Wang et al. (2022b). The external auxiliary information-based methods typically reduce the eﬀect of
69:such as human posture or body key points. The Transformer-based methods primarily leverage the global modelling strengths of Transformers,
70:integrating various attention mechanisms to extract more discriminative features. Although these methods have achieved some encouraging
94:To resolve the mentioned problems, we propose a hierarchical fusion and local-aware Transformer (HFLAT) for occluded person ReID.
95:First, we design a feature hierarchical fusion module (FHFM) to hierarchize and fuse the patch feature vector sequence based on their relative
98:a local-aware multi-attention mechanism to limit feature interactions,
104:ﬁne-grained local feature extraction capability of the model by limiting the range of interactions between the features using a local-aware
106:• We perform extensive experiments on four datasets to validate the
108:2. Related works
114:a series of methods for occluded person ReID, which can be grouped into
115:external auxiliary information-based methods Gao et al. (2020), Huang
116:et al. (2020), Miao et al. (2019) and Transformer-based methods Li et al.
118:The external auxiliary information-based methods primarily mitigate
120:For instance, Miao et al. (2019) proposed a pose-guided feature alignment (PGFA) method to align local features using human key-point information. Gao et al. (2020) proposed a pose-guided visible part matching (PVPM) method that employs a pose-guided attention mechanism
122:areas accurately. Wang et al. (2022a) proposed a key-point-aware occlusion suppression and semantic alignment (POS) method that aligns
125:proposed a pose-guided partial-attention network with batch information (PPBI), which eﬀectively reduces the negative impact of occlusions
130:We propose a hierarchical fusion and local-aware Transformer
132:• We design a feature hierarchical fusion module (FHFM) that hierarchizes and fuses the patch feature vectors based on their relative
144:proposed a human parsing based alignment with multi-task learning
146:(2023) proposed a body part-based (BPreID) model that extracts local
147:features using a local attention mechanism guided by human parsed labels. Dou et al. (2024) proposed a decouple re-identiﬁcation and human
148:parsing (DROP) method, which decouples the features for person ReID
149:and human parsing tasks, mitigating the feature granularity conﬂict between the two tasks and improving their collaborative eﬀectiveness. Although the above methods based on external auxiliary information have
152:The Transformer-based methods mainly utilize the advantages of
153:Transformer in global modelling, combined with the attention mechanisms to obtain more discriminative features. For instance, Li et al.
154:(2021) proposed a part-aware Transformer (PAT) by ﬁrst applying the
155:Transformer framework to occluded person ReID tasks, which improves
156:the representation ability of local features by employing a pixel context encoder and a local prototype decoder. Lin et al. (2024) proposed a multi-level relation-aware Transformer (MLRAT), which has
158:patch and sample levels. Wang et al. (2022b) proposed a Transformerbased pose-guided feature disentangling (PFD) method, which eﬀectively mitigates the negative impact of occlusions by disentangling
160:regions. Yang et al. (2023) proposed a robust feature mining Transformer (RFMT) method, which combines residual Transformer layers
161:with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
163:et al. (2024) proposed a novel occlusion-aware feature recover (OAFR)
164:model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
167:Although the above Transformer-based methods can address the
170:image patches to identity recognition, nor do they prioritize the identity information contained in key regions. For instance, the methods
174:features, thereby failing to fully exploit the identity information embedded in diﬀerent body parts of pedestrians. For instance, the methods such as MLRAT (Lin et al., 2024) and RFMT (Yang et al., 2023)
180:Our work is also related to some works. For example, Eliwa et al.
181:(2024) proposed a framework that integrates Microsoft Azure cloud services with a permissioned blockchain network. After preprocessing and
184:the framework stores the images in Azure Blob Storage, and realizes access control exclusive to authorized specialists through blockchain smart
185:contracts. Abd El-Hafeez et al. (2025) proposed a novel multi-scale attention model for the classiﬁcation of breast cancer histopathological
186:images, which achieves high-precision recognition by capturing discriminative features across multiple morphological scales in histopathological images. Eliwa and Abd El-Hafeez (2025c) proposed a robust deep
187:learning framework improved upon YOLOv11 for the multi-class classiﬁcation task of cervical cancer cells, which enhances the model’s classiﬁcation accuracy via an Attention-Guided Multi-Scale Feature Fusion
189:with an Attention-Guided Multi-Scale Feature Fusion (AGMS-FF) module, baseline YOLOv11, ResNet18, EﬃcientNet-B0, and MobileNetV3.
191:vision systems in sustainable rice cultivation. Hassan et al. (2025b) proposed a novel DenseNet model integrated with attention mechanisms
194:proposed an improved YOLOv11 architecture for the automated classiﬁcation of peripheral blood cells, which integrates a Dynamic CrossScale Context Aggregation (DCSCA) module. Through parallel convolution, dynamic attention, and cross-scale interaction, the module enables
197:blood cells. Hassan et al. (2025d) proposed a novel network architecture named Sign Neuro-Enhanced Dense Attention Network (SNDA),
204:the lack of deployment considerations. Hassan et al. (2025c) proposed
207:attention mechanisms, Nadam optimization, and other data augmentation methods. Eliwa and Abd El-Hafeez (2025) proposed a plant disease
208:detection and classiﬁcation method based on a modiﬁed YOLOv11 architecture, which improves the model’s performance in plant disease
210:(2012) proposed an improved edge detection method for grayscale images. Based on the Baljit and Amar algorithm, this method replaces
213:quality. Saabia et al. (2019) proposed a face recognition system, which
220:3. Proposed method
221:This section begins with an overview of the HFLAT framework,
222:then details the feature hierarchical fusion module (FHFM), the feature separation module (FSM), and the local feature extraction module (LFEM). Finally, the loss functions employed for model training are
230:Fig. 2. Work ﬂowchart of HFLAT. FHFM denotes the feature hierarchical fusion module, FSM denotes the feature separation module, and LFEM denotes the local
235:3.2. Feature hierarchical fusion module
249:We feed 𝑍0 into 𝐿 Transformer layers to get an output feature sequence 𝑍𝐿 = [𝑧𝑐𝑙𝑠 ; 𝑧1 ; 𝑧2 ; … ; 𝑧𝑁 ] ∈ ℝ(𝑁+1)×𝐷 and an attention weight
252:hierarchical fusion module. In this module, we hierarchize and fuse
259:into the local feature extraction module, in which we insert four learnable local tokens into 𝑍𝑝𝑎𝑡𝑐ℎ and use the local-aware multi-head attention to limit the range of feature interactions, extracting four local feature vectors {𝑓𝑖𝐿 }4𝑖=1 . Lastly, inspired by Sun et al. (2018), Miao et al.
265:Most of current Transformer-based occluded person ReID methods
268:we design a feature hierarchical fusion module, as illustrated in Fig. 3,
305:Furthermore, we perform hierarchical fusion of these four levels of
334:Fig. 3. Work ﬂowchart of feature hierarchical fusion module.
344:obtain the fusion feature vector 𝑓𝑢 ∈ ℝ𝐷 .
362:while suppressing interfering features caused by backgrounds and occlusions. Compared to the methods such as PGFA (Miao et al., 2019),
367:methods such as FCFormer (Wang et al., 2024) and OAFR (Bian et al.,
369:to separate and enhance foreground features. This approach eﬀectively
408:Fig. 5. Work ﬂowcharts of (a) Local feature extraction module, (b) Local-aware multi-head attention and (c) Local-aware attention.
430:and use a local-aware multi-head attention in this module, as shown in
431:Fig. 5(b), where each attention head is a local-aware attention, as shown
433:our local-aware multi-head attention achieves an eﬀective interaction of
441:approach enables our model to learn discriminative local features
446:we process 𝑍 ′ using a local-aware multi-head attention mechanism.
532:of the concatenated result to 𝐷, obtaining the output of the local-aware
552:4. Experiments
599:experiments of this study, we strongly recommend employing the hyperparameter optimization method described in Shukla et al. (2025) to
645:4.2. Experimental settings
647:conduct experiments on an NVIDIA 3090 GPU. The backbone is a ViT
648:pretrained on ImageNet (Zhang et al., 2019). In our experiments, both
738:Experimental results(including mAP (%) and Rank-1 accuracy (%)) of
739:HFLAT and several SOTA occluded person ReID methods on the OccludedDukeMTMC and Occluded-ReID datasets. “-” indicates unavailable data,
741:Method
872:Experimental results(including mAP (%) and Rank-1 accuracy (%)) of HFLAT
873:and several SOTA occluded person ReID methods on the Market-1501 and
877:Method
1002:summarizes the experimental results of our model compared to several SOTA occluded person ReID methods on the Occluded-DukeMTMC
1003:and Occluded-ReID datasets. Among these methods, PGFA (Miao et al.,
1005:(Wang et al., 2022a) are occluded person ReID methods based on external auxiliary information, while TransReID (He et al., 2021), PAT
1010:et al., 2025) are Transformer-based occluded person ReID methods. As
1014:dataset. Compared with the external auxiliary information-based methods, HFLAT improves the mAP and Rank-1 accuracy on the OccludedDukeMTMC dataset by at least 10.7% and 14.6%, respectively, and the
1020:phases, we set three distinct random seeds for each experiment and use
1022:of each experiment. The values of key parameters, hardware and software requirements, and parameter count of our model are presented in
1024:4.3. Comparison with state-of-the-art methods
1025:To evaluate the eﬃcacy of HFLAT, we conduct comparative experiments against several state-of-the-art (SOTA) methods for occluded person ReID on the occlusion datasets Occluded-DukeMTMC and OccludedReID, as well as on the standard datasets Market-1501 and DukeMTMCReID.

codex
前四篇的方法边界已经找到了：GSTNet 是第 2 节，FMUPS 是第 3 节，HGTDR 是第 3 节，HFLAT 是第 3 节。接着我读剩下两篇，再回头把每篇的动机、话术、机制和证据链整理成方法论拆解。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|hierarchical|knowledge|reasoning|text|image|attribute|semantic'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
10:Hierarchical knowledge-guided reasoning for text-based person
33:Text-image person re-identiﬁcation
35:Knowledge-guided reasoning
38:Masked language modeling (MLM) has expanded the exploration of text-image person re-identiﬁcation (TIReID)
39:tasks from coarse-granularity to ﬁne-grained alignment. Whereas, we note that vanilla MLM picks random tokens for visual-to-token reasoning, which could fail the intention of semantic visual-textual alignment by indistinguishably focusing on all the sub-words. This work proposes to leverage the inherent hierarchical scene
40:graph knowledge in each text for guiding token masking and enhancing cross-modal representation in TIReID,
41:thus relieving the pitfall of blind visual-textual alignment. The proposed framework, Hierarchical KnowledgeGuided Reasoning (HKGR), parses object-level, attribute-level, and relation-level masking according to phrase
42:knowledge constructions and explicitly lets the training of a dedicated encoder focus on the visual-to-token reasoning of these highlighted tokens. In addition, we propose a Multi-Grained Semantic Alignment (MGA) module,
43:which leverages the token selection method and image-text similarity distribution constraint to further facilitate
44:the semantic alignment between image and text at both coarse-grained and ﬁne-grained levels. Experimental
45:results demonstrate that our HKGR framework achieves state-of-the-art (SoTA) performance on three public
46:benchmark datasets at all evaluation metrics. We believe that the knowledge-guided idea is beneﬁcial to other
49:1. Introduction
50:Text-image person re-identiﬁcation (TIReID) stands as a fundamental and long-standing task in person re-identiﬁcation, dedicated to
51:searching pedestrian images with the same identity according to text
56:alignment to bridge the inherent modality gap between images and text
58:Towards this end, numerous TIReID approaches have been proposed,
60:global-matching methods, separately maps the visual global representation and textual global representation into a joint embedding space
63:Nevertheless, it can be hard for such a compact representation to capture ﬁne-grained semantic details in texts and images. For example, as
64:is shown in Fig. 1, understanding the text descriptions involves complicated semantic reasoning regarding diﬀerent objects (‘lady’, ‘pant’,
65:‘shirt’), attributes (‘black’, ‘white’), and relations (‘hold’, ‘in front of’).
67:Ding et al., 2021; Wang et al., 2020b, 2022a; Yan et al., 2023a), localmatching methods, leverages detailed visual cues and individual textual words to represent images and texts, respectively, and performs
68:local semantic alignment to compute overall similarity. Particularly,
70:2023; Jiang & Ye, 2023; Zuo et al., 2023), which fall under the localmatching methods category, adopt local semantic reasoning to establish ﬁne-grained relationships between image and text representations,
79:0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
85:object-attribute couplings (e.g., “white shirt”) and spatial relationships
86:(e.g., “in front of”), leading to fragmented textual representations and
87:an increased semantic gap between global textual descriptions and local
90:propose a novel Hierarchical Knowledge-Guided Reasoning (dubbed
91:as HKGR) framework for TIReID tasks, which harnesses hierarchical
92:knowledge in scene graphs parsed from text to enhance the masking
93:and designs a Knowledge-Guided Reasoning (KGR) module for multilevel consistent alignment, as illustrated in Fig. 1(b). To be speciﬁc,
94:we decompose text descriptions into three hierarchical semantic levels
96:attributes, objects, and relations, respectively. Diﬀerent levels are not
97:independent, and their interactions explain what semantic roles they
99:to mitigate semantic distortion caused by random masking (for Challenge 1). Moreover, for the KGR module, we propose object-level reasoning, attribute-level reasoning, and relation-level reasoning tasks, realizing multi-level cross-modal consistency alignment to narrow the semantic gap between image and text (for Challenge 2). Furthermore, we
100:also design a Multi-Grained Semantic Alignment (MGA) module. MGA
101:ﬁrst utilizes a token selection method to select multi-grained discriminative information both in visual and text tokens, and then it constrains
102:the multi-grained image-text similarity distributions for proper crossmodal alignments at both coarse-grained and ﬁne-grained levels. Our
106:We propose a Hierarchical Knowledge-Guided Reasoning (HKGR)
107:framework, which decomposes text descriptions into three levels,
108:bolstering purposeful knowledge masking and alleviating semantic
111:the semantic gap between image and text, we design a KnowledgeGuided Reasoning (KGR) module for better semantic coverage.
112:• We introduce a Multi-Grained Semantic Alignment (MGA) module
113:that employs a token selection method to select multi-grained discriminative information and perform proper cross-modal alignments
114:by multi-grained image-text similarity distributions.
116:and a series of ablation studies. The extensive experimental results
121:(a) The MLM-based paradigm is trained by randomly masking and predicting sub-words based on the unmasked contextual texts and the paired image
122:patches. (b) Our HKGR proposes hierarchical knowledge from scene graphs to
123:enhance masking and adopt a knowledge-guided reasoning strategy, realizing
127:us to investigate MLM in the context of TIReID tasks, aiming to exploit
128:the ﬁne-grained semantic interaction across images and texts.
130:predicting text sub-words to align contextual information with image
132:random masking strategy is often suboptimal for image-text alignment
134:of semantic concepts in textual descriptions. In many TIReID benchmarks (Ding et al., 2021; Li et al., 2017b; Zhu et al., 2021), it is common for multiple textual descriptions to refer to the same individual
136:Therefore, the ﬁne-grained diﬀerences among text descriptions bring
137:challenges for image-text alignment in MLM-based TIReID paradigms:
138:1) The randomness of masked language: Given the complexity of semantic information in textual descriptions, vanilla MLM-based methods
140:approach tends to disproportionately target high-frequency or semantically peripheral tokens. Such indiscriminate masking introduces semantic noise, compelling the model to predict uninformative elements (e.g.,
141:function words or subword fragments) rather than semantically meaningful keywords or phrases that are critical for ﬁne-grained alignment.
142:Consequently, the learned visual-textual correspondences may become
144:discriminative semantics (e.g., “red backpack,” “striped shirt”) are essential for accurate identity matching. 2) Ignoring phrase-level semantic
146:Existing MLM-based approaches primarily focus on isolated wordlevel predictions, overlooking multi-word phrases, such as objectattribute pairs or relational constructs, that convey richer, compositional semantics. As a result, these models often fail to capture critical
148:The rest of this paper is organized as follows. First, we brieﬂy review the related work in Section 2, and Section 3 introduces the design
149:of HKGR. Then we present the experimental settings and results in Section 4 and 5. Finally, conclusions are given in Section 6.
150:2. Related work
151:2.1. Text-image person re-identiﬁcation
153:2024; Ye et al., 2021; Zheng et al., 2016) and image-text retrieval (Faghri et al., 2017; Qin et al., 2022; Zheng et al., 2020), was
155:due to its intra- and inter-modal variations. The existing methods can
158:proper alignment between image and text modalities in the common
160:networks to extract image and text features and align these features
161:using an image-text contrastive loss in the shared latent space. Zheng
168:et al. (2020) propose a Dual-Path method that employs a CNN structure for both image and text feature extraction to enable eﬀective endto-end ﬁne-tuning using an instance loss. Zhang and Lu (2018) design
173:The local-matching paradigm primarily concentrates on mining local cross-modal correspondence between image regions and words or
175:a dual-path local alignment network to extract local visual and textual representations from horizontally segmented image patches. Subsequently, local representations are aligned adaptively with a multi-stage
176:cross-modal matching. Zhu et al. (2021) propose a DSSL model to extract and align body part information from images using a mutual exclusion constraint fusion mechanism. Additionally, some works (Ding et al.,
180:the local ﬁne-grained correspondence between image-text modalities.
185:visual-textual tokens. Speciﬁcally, Jiang and Ye (2023) proposes an implicit relation reasoning module to predict the random masked text tokens based on image patches and unmasked surrounding text tokens,
186:aiming to align images and text representation. In the work of FLIP (Zuo
187:et al., 2023), attribute phrases are masked and predicted by combining
188:masked textual embeddings with global image embeddings to construct
189:correlations between images and texts.
191:may not capture semantic-rich words or phrases, neglecting detailed semantic feature representation and alignment. In this paper, we implement knowledge-guided reasoning by predicting hierarchical semantics
192:in scene graphs from text, aiming to guide detailed semantic alignment
193:across visual and text modalities.
196:speciﬁc semantic information along designated paths.
198:of hierarchical scene graph knowledge in the TIReID.
200:Vision-language pre-training (VLP) seeks to establish semantic correlations between vision and language. Inspired by the success of pretraining paradigm in single-modal tasks, e.g., language pre-training
203:Radford et al., 2021). Based on the model structure, existing VLP methods can be divided into two categories: single-tower and two-tower. The
208:extract image and text features separately with distinct encoders. These
211:high-quality visual-language semantic representation capacity and has
212:been applied to various downstream multi-modal tasks, including textvideo retrieval (Fang et al., 2021; Luo et al., 2022; Ma et al., 2022)
214:et al. (2022) are the ﬁrst to explore the transfer of CLIP knowledge
215:into video-text cross-modal retrieval and demonstrate that a large-scale
216:multi-modal pre-training model is beneﬁcial for video-text retrieval.
218:model, XCLIP, to capture correlations between cross-grained comparisons. Yan et al. (2023a) pioneer the integration of CLIP visual representations into TIReID, proposing a CLIP-driven method that achieves
223:& Ye, 2023), we also leverage the pretraining knowledge of CLIP for
229:follows, we ﬁrst introduce the image-text feature representation in Section 3.1, and then describe the scene graph construction in Section 3.2,
232:Scene graph, which represents objects, attributes of objects, and relations between objects with a graph, was ﬁrst proposed by Johnson
234:et al., 2018), scene graph knowledge has been extensively integrated
235:into multi-modal tasks (Yu et al., 2021), such as image captioning (Yang
240:pre-training framework that integrates structured knowledge obtained
241:from scene graphs to learn ﬁne-grained semantic alignment across vision
243:downstream tasks. To model relations between objects in image captioning, Yao et al. (2018) presents a GCN-LSTM model to build graphs with
244:the detected objects from an image based on their spatial and semantic connections. In terms of cross-modal retrieval, Wang et al. (2020a)
245:introduce both visual scene graphs and textual scene graphs to characterize objects and their relations, optimizing the model by computing
248:that combines a visual scene graph and textual scene graph to enable the
250:3.1. Image-text feature representation
251:Image feature representation. For image feature representation,
254:image encoder. Speciﬁcally, for an image 𝐼𝑖 ∈ ℝ𝑐×𝑤×ℎ , we ﬁrst evenly
255:split 𝐼𝑖 into 𝑛𝑣 = (ℎ × 𝑤)∕𝑝2 ﬁxed-dimension image patch sequences,
256:where 𝑝 represents the size of image patches. Then the split image
260:with a learnable [CLS] token as the preﬁx and feed it into the transformer blocks of ViT. The output image features can be represented as
261:𝑓𝑖𝑣 = {𝑣𝑖𝑐𝑙𝑠 , 𝑣𝑖1 , 𝑣𝑖2 , …, 𝑣𝑖𝑛 }, where 𝑣𝑖𝑐𝑙𝑠 is the image-level global feature,
265:Text feature representation. In terms of text feature representation, we adopt the text encoder of pre-trained CLIP as a text feature
272:Fig. 2. Overview of the proposed HKGR: given image-text pairs, the image is encoded into an embedding by the image encoder. Simultaneously, the text is parsed
273:into a scene graph, and then a knowledge-guided masking strategy is adopted to generate masked text. Finally, the masked text embedding obtained from the
274:text encoder and image embedding are fed into a cross-modal interaction encoder to perform object-level reasoning, attribute-level reasoning, and relation-level
275:reasoning.
289:extractor. Speciﬁcally, for the input text 𝑇𝑖 , we use the lower-cased byte
290:pair encoding (BPE) tokenizer to tokenize 𝑇𝑖 into a sequence of text tokens. We then add the speciﬁc token [SOS] at the start and [EOS] at the
292:transformer blocks of the text encoder to obtain the ﬁnal text features
296:Cross-modal interaction encoder. In order to construct the interaction between image and text modality, we adopt a transformer-based
297:encoder for image and text features fusion. As shown in Fig. 3, the CrossModal Interaction Encoder(CME) contains transformer blocks composed
299:Given a text 𝑇𝑖 , we mask a portion of the keywords/phrases derived
300:from the scene graph parsed from the text to obtain masked text 𝑇̂𝑖 . The
301:paired image 𝐼𝑖 and masked text 𝑇̂𝑖 are then fed into the image/text encoder to output image feature tokens 𝑓𝑖𝑣 and masked text feature tokens
302:𝑓̂𝑖𝑡 = {𝑡̂𝑖𝑠𝑜𝑠 , 𝑡̂𝑖1 , 𝑡̂𝑖2 , …, 𝑡̂𝑖𝑛 , 𝑡̂𝑖𝑒𝑜𝑠 }. Image feature tokens 𝑓𝑖𝑣 and masked text fea𝑡
306:denotes multi-head cross-attention, which adopts masked text feature
307:tokens 𝑓̂𝑖𝑡 as query and image feature tokens 𝑓𝑖𝑣 as key and value,
312:We use out-of-the-box scene graph generation methods (Anderson
313:et al., 2016) to obtain the scene graph. The method ﬁrst employs the
315:graph. Speciﬁcally, Given an input text 𝑇𝑖 , the scene graph can be represented as 𝐺(𝑇𝑖 ) = {𝑉 (𝑇𝑖 ), 𝐸(𝑇𝑖 )}, where 𝑉 (𝑇𝑖 ) = {𝑂, 𝐴, 𝑅} denote the
316:nodes set, which includes object nodes, attribute nodes, and relation
317:nodes. 𝐸(𝑇𝑖 ) = {𝑃 , 𝑄} is the edge set, where 𝑃 ⊆ 𝑂 × 𝐴 is the set of attribute pairs, and 𝑄 ⊆ 𝑂 × 𝑅 × 𝑂 is the set of relation triplet.
318:A typical example is illustrated in Fig. 2, where the green rectangles represent object nodes, the orange rectangles represent attribute
319:nodes, and the blue rectangles represent relation nodes. The black arrows indicate the edge of the attribute or relation associated with the
320:corresponding node. The scene graph parsed from the text is presented
327:3.3. Knowledge-guided reasoning
328:Utilizing hierarchical knowledge obtained from scene graphs,
339:Illustration of scene graph parsed from text.
340:text: 𝑇𝑖
352:attribute: 𝑃
367:in  of masked 𝑗-th attribute token, and 𝑦𝑗𝑘 is one-hot vocabulary indicator.
368:Relation-level reasoning. In general, relations depict actions or
370:similar attributes but varying relations. In the relation-level reasoning,
371:given the input image text pairs (𝐼𝑖 , 𝑇𝑖 ), we ﬁrst obtain all the relation triplet 𝑄 = {{𝑜1𝑖 , 𝑟𝑖 , 𝑜2𝑖 }|𝑖 = 1, 2, … , 𝑛𝑞 } from the scene graph of text,
374:triplets in the scene graph for masking and apply the same masking strategy described in object-level reasoning. The selected relation triplet can
377:node pairs {𝑜̂1𝑖 , 𝑜̂2𝑖 } and mask the sub-sequences of the text that correspond to the selected relation nodes {̂𝑟𝑖 }. At last, the relation masked
378:text 𝑇̂𝑖𝑟 and image 𝑇𝑖 are fed into the text/image encoder to generate
379:masked text features𝑓̂𝑖𝑟,𝑡 and image features 𝑓𝑖𝑣 .
380:In relation level reasoning, the goal is to recover the masked relation
382:masked text features𝑓̂𝑖𝑟,𝑡 and image features 𝑓𝑖𝑣 . The loss function can be
386:text, we construct a knowledge-guided reasoning module, including object-level reasoning, attribute-level reasoning, and relationlevel reasoning. These reasoning processes guide the model to
387:learn hierarchical semantic local alignments between image and text
389:Object-level reasoning. Objects are the fundamental elements in a
390:visual scene and thus play a crucial role in constructing semantic information. Through object-level reasoning, the model is learning to establish the object-level visual-text connection. Given the paired image 𝐼𝑖
391:and text 𝑇𝑖 , we ﬁrst obtain all the object nodes 𝑂 = {𝑜𝑖 |𝑖 = 1, 2, … , 𝑛𝑜 }
392:from the scene graph of text. We randomly select 30 % of object nodes
395:object masking by masking the sub-sequences in the text that correspond to the selected object nodes 𝑂̂ = {𝑜̂𝑖 |𝑖 = 1, 2, … , 𝑛̂ 𝑜 }. Finally, the
396:object masked text 𝑇̂𝑖𝑜 and image 𝑇𝑖 are fed into the image-text encoder
397:to obtain masked text features𝑓̂𝑖𝑜,𝑡 and image features 𝑓𝑖𝑣 .
398:In the object-level reasoning, our goal is to recover the masked
399:object nodes 𝑂̂ tokens based on the masked text features𝑓̂𝑜,𝑡
407:3.4. Multi-grained semantic alignment
408:Most existing CLIP-based TIReID methods only employ image-level
410:semantic alignment. However, focusing solely on coarse-grained semantic alignment may not capture the ﬁne-grained semantic interactions
411:between image and text modalities, thereby hindering retrieval performance gains. To tackle this limitation, we propose a Multi-Grained Semantic Alignment (MGA) module to mine discriminative feature representations and perform multi-grained cross-modal alignments. The MAG
412:module consists of coarse-grained semantic alignment and ﬁne-grained
413:semantic alignment.
414:For coarse-grained semantic alignment, we employ image-level global
415:feature 𝑣𝑖𝑐𝑙𝑠 from the image encoder and sentence-level global feature 𝑡𝑖𝑠𝑜𝑠
416:from the text encoder as coarse-grained feature representation. Then,
417:coarse-grained semantic alignment is carried out by minimizing the
418:KL divergence between global image-text feature similarity distribution
445:Attribute-level reasoning. Attributes describe speciﬁc characteristics of objects, such as color or size, which contribute to distinguishing
446:pedestrians with similar appearances but diﬀerent attributes.
447:Given the input image text pairs (𝐼𝑖 , 𝑇𝑖 ), we ﬁrst obtain all the attribute pairs 𝑃 = {{𝑜𝑖 , 𝑎𝑖 }|𝑖 = 1, 2, … , 𝑛𝑝 } from the scene graph of text,
448:where 𝑎𝑖 ∈ 𝐴 denotes the attribute nodes corresponding to the object
449:nodes. Similarly, we randomly select 30 % of the attribute pairs in the
451:in object-level reasoning. The selected attribute pairs are reorganized as
452:𝑃̂ = {{𝑜̂𝑖 , 𝑎̂𝑖 }|𝑖 = 1, 2, … , 𝑛̂ 𝑝 }. Then, we implement attribute masking by
453:masking the sub-sequences of the text that correspond to the selected
454:attribute nodes 𝑎̂𝑖 while keeping the associated object node 𝑜̂𝑖 . Finally,
455:the attribute masked text 𝑇̂𝑖𝑎 and image 𝑇𝑖 are fed into the text/image
456:encoder to generate masked text features𝑓̂𝑖𝑎,𝑡 and image features 𝑓𝑖𝑣 .
457:The attribute-level reasoning aims to recover the masked attribute
458:nodes {𝑎̂𝑖 |𝑖 = 1, 2, … , 𝑛̂ 𝑎 } tokens based on the masked text features𝑓̂𝑖𝑎,𝑡
459:and image features 𝑓𝑖𝑣 by minimizing the cross-entropy loss as follows:
475:and image features 𝑓𝑖𝑣 by minimizing the cross-entropy loss as
506:For ﬁne-grained semantic alignment, we observe that meaningless tokens can distract the model from capturing discriminative cues. Therefore, inspired by the previous works (Jiang & Ye, 2023; Qin et al., 2024;
508:both image and text modalities using correlation weights. In practice,
510:𝐴𝑣𝑖 ∈ ℝ(1+𝑛𝑣 )×(1+𝑛𝑣 ) from the last transformer block of the image encoder,
522:the token selection ratio. In terms of textual procedure, we can obtain
534:the selected local text features in a similar way as 𝑓̂𝑖𝑡 = {𝑡𝑖 𝑡 , 𝑡𝑖 𝑡 , …, 𝑡𝑖 𝑡 },
539:2021) dataset consists of 20,505 pedestrian images and 41,010 text descriptions with two captions per image. The average length of text descriptions is 23 words. The dataset is divided into three subsets, with
545:and textual token features 𝑓̂𝑖𝑡 , we perform linear feature transformation
549:and text embedding 𝑓𝑙𝑜𝑐
579:list, given a text description as the query. In addition, for a more thorough assessment, we also leverage the mean Average Precision (mAP) as
586:fully-connection layer. Finally, the ﬁne-grained semantic alignment is
587:performed by minimizing the KL divergence between the local image𝑡,𝑖
589:text feature similarity distributions 𝜓(𝑆(𝑓𝑙𝑜𝑐
591:)) and the label distribution 𝑃 (𝑌𝑖 ). The loss function for ﬁne-grained semantic alignment can
619:We initialized the image and text encoder with pre-trained weights
625:𝑅 of MGA is set to 0.3. For image data augmentation, we adopt random crop, random erasing, and random horizontal ﬂipping strategies.
626:For text data augmentation, we apply random masking, replacement,
627:and deletion strategies. All the image sizes are set to 384 × 128 instead
629:parameter 𝜏 to 0.02. We conduct our experiments on a single NVIDIA
635:To facilitate knowledge-guided reasoning, we adopt a multi-level
636:reasoning strategy that incorporates object-level, attribute-level, and
637:relation-level reasoning. Speciﬁcally, during each training epoch, we
638:sequentially and cyclically implement object-level reasoning, attributelevel reasoning, and relation-level reasoning until the model converges.
639:Following the previous work (Zheng et al., 2020), we also adopt an instance loss 𝑖𝑑 that classiﬁes each image and text into speciﬁc classes
640:based on their corresponding individual IDs, allowing the model to differentiate between image and text groups corresponding to distinct individuals. Above all, the ﬁnal loss function can be formulated as follows:
644:We compare our model with 2 classes of SoTA methods in TIReID:
646:under images (e.g., ResNet50) and the single-modality pre-training
647:model under texts (e.g., BERT), to encode the image and text respectively: Dual-Path (Zheng et al., 2020), CMPM/C (Zhang & Lu,
656:(e.g., CLIP) to encode images and text: CFine (Yan et al., 2023a), TextReID (Han et al., 2021), VGSG He et al. (2023), IRRA (Jiang & Ye,
662:where 𝑘𝑔𝑟 is knowledge-guided reasoning loss, which integrates 𝑜𝑏𝑗 ,
663:𝑎𝑡𝑡𝑟 , and 𝑟𝑒𝑙 follow the multi-level reasoning strategy.
664:4. Experiment settings and baselines
668:The CUHK-PEDES (Li et al., 2017b) dataset consists of 40,206 pedestrian images, 80,412 text descriptions and 13,003 individuals, with each
669:image corresponding to two textual captions. The average number of
670:words in each textual description is 23.5. In line with the oﬃcial data
672:person images and 68,108 text descriptions. The validation set includes
673:1000 identities with 3078 person images and 6158 text descriptions,
674:while the test set includes 1000 identities with 3074 person images and
675:6156 text descriptions. The ICFG-PEDES (Ding et al., 2021) is collected
676:from the MSMT17 (Wei et al., 2018) dataset, which includes 4102 individuals with 54,522 person images and 54,522 text descriptions. The
679:set. The training set consists of 31,002 identities with 34,674 images
680:and text descriptions, while the test set consists of 1000 identities with
681:19,848 person images and text descriptions. The RSTPReid (Zhu et al.,
683:5. Experimental results and analysis
684:In this section, We conduct extensive experiments to assess our HKGR
686:baselines. Speciﬁcally, our evaluation goals in the experiments are to answer three research questions (RQs): RQ1: How does the performance of
687:our HKGR compare to various SoTA baseline methods? RQ2: How do the
698:Performance comparisons with SoTA methods on the CUHK-PEDES dataset.
699:Method
703:Image Enc.
705:Text Enc.
724:TextReID (Han et al., 2021)
790:Text-CNN
930:Performance comparisons with SoTA methods on the ICFG-PEDES
932:Method
1016:2) The experimental results on the ICFG-PEDS dataset are shown
1017:in Table 3. Our proposed HKGR outperforms all SoTA methods on
1019:and R@10 accuracy, surpassing the SoTA method BiLMa (Fujii &
1020:Tarashima, 2023) by 1.46 %, 1.02 %, and 0.48 %, respectively. Compared with the recent strong baseline method IRRA (Jiang & Ye,
1021:2023), our proposed method achieves a performance gain of 1.83 %
1023:• 3) We also conduct our experiment on the newly-published RSTPReid dataset, as depicted in Table 4. Our proposal suppresses
1024:all SoTA methods in all metrics, reaching 63.10 %, 83.60 %, and
1026:with the recent state-of-the-art method BiLMa (Fujii & Tarashima,
1032:can be attributed to the eﬀectiveness of HKGR, which leads to competitive results.
1034:5.1. Comparison with SoTA methods (RQ1)
1035:We evaluate our HKGR on three public benchmark datasets and compare it with various SoTA methods, as shown in Tables 2–4. From the
1040:Performance comparisons with SoTA methods on the RSTPReid
1046:all SoTA methods. Speciﬁcally, our HKGR achieves 75.21 %, 90.22 %
1048:the representative SoTA method BiLMa (Fujii & Tarashima, 2023)
1050:achieves 1.83 % improvements of R@1 over the recent vanilla MLMbased model IRRA (Jiang & Ye, 2023). This can be attributed to
1051:our proposal providing detailed hierarchical semantic alignments,
1052:thereby reducing the semantic gap between image and text modalities.
1055:Method
1135:Ablation analysis on diﬀerent components of our HKGR on the CUHK-PEDES
1152:Methods
1251:5.2. Ablation study (RQ2)
1253:we conduct a series of ablation studies on the CUHK-PEDES dataset.
1256:Ablations on the diﬀerent components. The detailed results are
1269:coarse-grained and ﬁne-grained semantic alignment is more beneﬁcial
1271:loss (Zheng et al., 2020) to our proposed model (No.5 vs. No.8), the R@1
1273:Analysis of hierarchical consistency alignment. As aforementioned, the KGR module incorporates objective-level reasoning,
1274:attribute-level reasoning, and relation-level reasoning to establish ﬁnegrained connections between images and text. To further evaluate the
1275:eﬀectiveness of our proposal, we compare it with four diﬀerent knowledge reasoning strategies, including MLM-based reasoning, object-level
1276:reasoning, attribute-level reasoning, and relation-level reasoning. Note
1277:that we follow the same masking strategy for MLM-based reasoning as
1279:We can see that merely replacing random masking with a knowledge-
1286:a separate knowledge-based prediction strategy (No.4 vs No.1, No.2, and
1297:the 𝑅 to 0.3 for all the experiments. (2) As the value of 𝜏 increases, the
1300:experiments.
1302:We implement three qualitative experimental results, feature representation visualization, heatmap visualization, and retrieval visualization, to
1307:eﬀectiveness of our proposal in semantic correspondence of image and
1308:text, we use t-SNE to transform the high-dimensional image and text
1314:Comparisons between four diﬀerent knowledge reasoning
1318:Method
1332:MLM-based reasoning
1333:Object-level Reasoning
1334:Attribute-level Reasoning
1335:Relation-level Reasoning

exec
/bin/zsh -lc "for f in 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf'; do pdfinfo \""'$f" | rg '"'Title|Pages'; done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           GSTNET: A Geospatial-Temporal Graph Network for Group Person Re-Identification
Pages:           5
Title:           Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search
Pages:           12
Title:           Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification
Pages:           15
Title:           Hierarchical fusion and local-aware transformer for occluded person re-identification
Pages:           14
Title:           Hierarchical knowledge-guided reasoning for text-based person re-identification
Pages:           11
Title:           Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification
Pages:           5

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf' - | rg -n -i 'abstract|introduction|related|method|approach|proposed|framework|experiment|ablation|hierarchical|proxy|cloth|clothes|changing|identity|pseudo'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 200ms:
3:Hierarchical Proxy Learning for Cloth-Changing
11:Abstract—Cloth-Changing person Re-Identification (CC-ReID)
13:the cloth-changing scenario. It is quite challenging due to
14:the large intra-person variance and small inter-person variance caused by clothes changing. To address these issues, in
15:this work we propose a Hierarchical Proxy Learning (HPL)
16:framework to extract clothes-irrelevant and person-invariant
17:features. Specifically, we employ person labels as the main proxy.
18:Instead of leveraging clothing labels as sub proxy, we further
19:propose a clustering-based automatic sub-proxy mining scheme.
20:More specifically, we first construct a person-aware Main Proxy
22:Then, a Sub Proxy Learning (SPL) is constructed to enhance the
23:intra-person compactness. Finally, a Sub-to-Main Proxy Learning
24:(S2MPL) is proposed to promote the cooperation between the
26:negative effect of clothes, we propose a Sample Balance and
29:the diversity of clothes, simultaneously. Extensive experiments on
31:proposed method over most state-of-the-art methods.
32:Index Terms—cloth-changing person re-identification, hierarchical proxy learning, sample balance, joint training.
35:Cloth-Changing person Re-Identification (CC-ReID) is a
41:To address CC-ReID, previous methods [9]–[11] aim to
42:eliminate the impact of clothes, and extract the inherent
46:leading to some performance deteriorations on the sameclothing ReID. Furthermore, various approaches rooted in metric learning [12], [13] and data augmentation techniques [14],
97:Clothe 1
125:Clothe C
146:Generate Proxy
150:Main Proxy Learning
154:Sub-to-Main Proxy Learning
158:Sub Proxy Learning
160:(c) Hierarchical Proxy Learning
164:persons from CC-ReID datasets. (c) Geometry interpretation of the proposed
165:Hierarchical Proxy Learning (HPL). Different colored dots and shapes represent different persons and sub proxies identities, respectively.
168:caused by changing clothes, the instance-level triplet loss [17]
171:by proxy-based metric learning. For example, Wang et al. [23]
172:propose intra-camera and inter-camera proxy contrastive learning. For CC-ReID, Gu et al. [24] design a clothes-based
173:adversarial loss to further pull the features with the same identity closer. Unfortunately, both of them focus on instance-toproxy interactions, and neglect inter-proxy relations. Different
174:from previous methods, as shown in Fig. 1 (c), we propose a
175:Hierarchical Proxy Learning (HPL) framework, which consists
176:of a Main Proxy Learning (MPL), a Sub Proxy Learning
177:(SPL) and a Sub-to-Main Proxy Learning (S2MPL). In MPL,
182:with different sub proxies to improve intra-person compactness. Unlike [24] which directly using clothing labels as sub
183:proxies, we propose a clustering-based automatic sub-proxy
184:mining scheme. In S2MPL, each sub-proxy acts as an anchor,
185:being pulled toward its corresponding main proxy and pushed
186:away from others, promoting inter-person diversity and intraperson compactness. Such a hierarchical structure contributes
190:Sub-proxy Mining Original
233:: Main Proxy
235:: Sub Proxy
239:Clothes Change
270:Part2: Hierarchical Proxy Learning
274:Fig. 2. Illustration of the proposed framework.
276:to extracting person-invariant and clothes-irrelevant features.
277:As shown in Fig. 1 (c), when constructing a hierarchical
278:structure, if there is no assistance, the main proxy P1 in a minibatch will have no corresponding positive samples. What’s
282:who wears one clothes all the time. Considering the above
286:the diversity of clothes, simultaneously. Specifically, we first
287:explore a Sub-proxy Balanced Sampling (SBS) strategy taking
289:which is more suitable for CC-ReID. Then, a Semanticguided Clothes Transfer (SCT) is proposed to enrich the
290:diversity of clothes, which utilizes a pre-trained human parsing
291:network [25] to guide clothing changing. Thanks to SCT, we
292:can get the main proxy positive samples corresponding to
293:pedestrians based on the augmented samples. Experimental
294:results demonstrate that our method significantly outperforms
298:for CC-ReID. (2) We propose a novel proxy-level metric
299:learning method with a hierarchical structure to extracting
300:person-invariant and clothes-irrelevant features. (3) Extensive
301:experiments demonstrate that our proposed method outperforms most state-of-the-art cloth-changing methods on two
302:widely-used CC-ReID datasets, i.e., PRCC and VC-Clothes.
304:clustering-based automatic sub-proxy mining scheme. Specifically, before each round of network training, we cluster all
308:clusters whose pseudo-labels are used as the sub-proxies. In
309:practice, we adopt the DBSCAN [28] method for clustering.
310:Sub-proxy Balanced Sampling. The sampling strategy [17]
312:persons but ignores the balance of different clothes. Intuitively,
313:it is useful to choose balanced sub-proxy in each batch for CCReID. Therefore, we propose a Sub-proxy Balanced Sampling
315:where C sub proxies per person and K images per sub proxy.
317:and sub-proxy, thereby promoting the learning efficacy.
318:Semantic-guided Clothes Transfer. In CC-ReID datasets,
319:some persons may wear only one clothes all the time. Data
322:Clothes Transfer (SCT) to change clothes among different persons. Specifically, given one image xi ∈ {xkp,c }P,C,K
325:and sub proxy in a mini-batch. Then, a pre-trained human
328:parts for persons are upper-clothes and pants, we perform SCT
329:based on the masks of upper-clothes and pants, respectively.
330:Given the upper-clothes masks mi and mj of two pedestrians,
331:we can transfer the upper-clothes of xj to xi ,
336:the average pixel value of the upper-clothes to address the variability in the clothing area of different persons. Reshape(·)
343:clothes label has been changed. Thus, our SCT can generate
345:clothes, which enriches the diversity of samples for CC-ReID.
347:As illustrated in Fig. 2, our proposed framework mainly includes two components: Sample Balance and Diversity (SBD)
348:module and Hierarchical Proxy Learning (HPL). Detailed
351:Recent methods [24], [26], [27] utilize clothes labels to
353:clothes labels requires a certain price. Inspired by the recent
354:success of unsupervised person ReID methods, we propose a
366:B. Hierarchical Proxy Learning
376:In our baseline method, we adopt the cross-entropy loss for
379:Hierarchical Proxy Learning (HPL) framework, including a
380:Main Proxy Learning (MPL), a Sub Proxy Learning (SPL)
381:and a Sub-to-Main Proxy Learning (S2MPL).
382:Main Proxy Learning. As stated in [24], the instance-level
386:this problem, we propose a Main Proxy Learning (MPL). The
388:Specifically, based on the features of persons, the main proxy
406:The main proxy ρi can be seen as an anchor. Then, we
407:can obtain the corresponding positive main proxy ρpos
429:Afterward, the negative main proxy ρneg
468:from previous methods, our proposed MPL can suppress the
470:Sub Proxy Learning. Our MPL does not take the intraperson compactness into account. Thus, as shown in the green
471:part of Fig. 1 (c), we further propose the Sub Proxy Learning
474:and construct sub proxy by:
509:calculate the distance between each sample and its proxy of instance samples with the same person but different sub proxies.
511:clothing labels. Under the constraint of SPL, the intra-person
512:variance caused by clothes transformation will be reduced.
513:Sub-to-Main Proxy Learning. To achieve the collaboration
514:between the main proxies and sub-proxies, we further propose a novel component called Sub-to-Main Proxy Learning
517:part of Fig. 1 (c), each sub proxy δi,c is treated as an anchor.
518:S2MPL pulls it towards the corresponding main proxy ρi , and
538:strict constraints on the similarity between instances in the subproxy. Similarly, the augmented sample x̂ki,c generated from
557:We evaluate our approach on two benchmark datasets, i.e.,
558:PRCC [29] and VC-Clothes [30]. There are two test settings
559:in CC-ReID, namely cloth-changing setting (CC), and sameclothes setting (SC). To measure the performance, we adopt
566:C OMPARISON WITH OTHER METHODS ON PRCC.
568:Method
570:Clothes
628:T HE ABLATION STUDIES OF DIFFERENT COMPONENTS .
636:Method
646:C OMPARISON WITH OTHER METHODS ON VC- CLOTHES .
648:Clothes
667:VC-Clothes
702:the feature encoder. For automatic sub-proxy mining, we use
709:identity, and each sub proxy with K = 4 images. We train the
710:framework for 60 epochs in total by the Adam [39] optimizer.
716:C. Comparison with State-of-the-art Methods
717:We compared our method with various advanced CC-ReID
718:methods on PRCC and VC-Clothes. Experimental results are
719:reported in Tab. I and II. On the PRCC and VC-Clothes
720:datasets, our method achieves the best results of 74.3% and
722:RCSANet [22] utilizes a clothes-proxy loss that encourages
723:more consistency when a person wears the same clothes, and
724:pulls away different clothes. In contrast, the purpose of our
725:HPL loss is to pull the samples of different clothes with
726:the same identity together, which is similar to CAL [24].
728:proposed HPL don’t use any clothes information and achieves
730:above methods ignore the unbalanced distribution of training
732:VC-Clothes
756:T HE ABLATION STUDIES OF HPL ON PRCC.
760:Method
874:feature map of the baseline method (b) and our proposed method (c).
876:samples in a mini-batch. As a result, our method achieves
877:84.9% Rank-1 accuracy on VC-Clothes under the CC setting,
879:demonstrate the effectiveness and superiority of our method.
880:D. Ablation Study
882:experiments, and show the results in Tab. III. It can be observed that the proposed SBS strategy is superior to the baseline. And models leveraging SCT also achieved remarkable
884:the enrichment of training examples with clothes-changing. In
885:addition, the proposed HPL further improves the performance
886:of CC-ReID. A reasonable explanation is that a hierarchical
888:clothes-irrelevant features. What’s more, as shown in Tab. IV,
891:the proposed method will pay more attention to non-clothing
894:In this paper, we propose a novel Hierarchical Proxy
895:Learning (HPL) framework for CC-ReID. To get rid of the
896:dependence on clothing labels, we propose a clustering-based
897:automatic sub-proxy mining scheme. We further propose a
899:Meanwhile, we propose a Hierarchical Proxy Learning (HPL)
900:to extract clothes-irrelevant and person-invariant features. Extensive experiments demonstrate the superiority of our method.
913:framework for video-based person re-identification,” in ICASSP 20242024 IEEE International Conference on Acoustics, Speech and Signal
925:unified network for cloth-changing person re-identification,” IEEE
929:[9] J. Wu, H. Liu, W. Shi, H. Tang, and J. Guo, “Identity-sensitive
930:knowledge propagation for cloth-changing person re-identification,” in
934:the influence of clothing: Universal clothing attribute disentanglement
938:Z. Chen, and X.-S. Hua, “Cloth-changing person re-identification from a
942:[12] X. Li, S. Wei, J. Wang, Y. Du, and M. Ge, “Adaptive multi-proxy for
945:[13] Z. Yang, M. Bastan, X. Zhu, D. Gray, and D. Samaras, “Hierarchical
946:proxy-based loss for deep metric learning,” in Proceedings of the
968:[20] X. Li, B. Liu, Y. Lu, Q. Chu, and N. Yu, “Cloth-aware center cluster
969:loss for cloth-changing person re-identification,” in Proceedings of
976:[22] Y. Huang, Q. Wu, J. Xu, Y. Zhong, and Z. Zhang, “Clothing status
984:[24] X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, and X. Chen, “Clotheschanging person re-identification with rgb modality only,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
990:[27] J. Mu, Y. Li, J. Li, and J. Yang, “Learning clothes-irrelevant cues for
991:clothes-changing person re-identification,” in Proceedings of the British
997:sketch under moderate clothing change,” IEEE Transactions on Pattern
999:[30] F. Wan, Y. Wu, X. Qian, Y. Chen, and Y. Fu, “When person reidentification meets changing clothes,” in Proceedings of the IEEE/CVF
1009:[33] P. Hong, T. Wu, A. Wu, X. Han, and W.-S. Zheng, “Fine-grained shapeappearance mutual learning for cloth-changing person re-identification,”
1013:sampling for cloth-changing person re-identification,” IEEE Signal Processing Letters, vol. 28, pp. 1365–1369, 2021.
1014:[35] K. Zhu, H. Guo, Z. Liu, M. Tang, and J. Wang, “Identity-guided human
1021:[37] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in Proceedings of the
1027:[39] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”

codex
这几篇不是都按同一种结构写，尤其两篇长文的方法在第 3 节里跨了不少公式和图注。我现在按行号把方法段完整抽出来，重点看作者如何从“失败观察”推到模块，而不是只记模块名。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' - | nl -ba | sed -n '33,351p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    33	1. INTRODUCTION
    34	Geospatial-temporal Group Person Re-identification (Gst-GReID)
    35	can integrate geospatial constraint and spatio-temporal contextual
    36	information. By modeling the appearance of groups across regions
    37	and cameras, it enables accurate identity matching and retrieval in
    38	complex scenarios. The existing methods for person re-identification
    39	can be divided into two categories. The first category is Group
    40	Re-identification (GReID). Early GReID studies focused mainly on
    41	group-level feature extraction and matching, such as unsupervised
    42	sparse transfer coding for cross-camera group matching [1]. The
    43	DukeTMC Group and Road Group datasets advanced group person
    44	re-identification by providing benchmark datasets [2]. To address
    45	variations in group size and layout, some studies employed graph
    46	modeling and feature aggregation to capture complex intra-group relationships [3, 4, 5]. Meanwhile, some works leveraged Transformers [6] to explore the advanced spatial reasoning of group layouts
    47	and cross-modal designs [7, 8, 9, 10, 11, 12]. The second category
    48	† Corresponding author Ping Hu: royalcat1982@gmail.com.
    49	
    50	979-8-3315-6701-9/26/$31.00 ©2026 IEEE
    51	
    52	Group B
    53	
    54	Group B
    55	
    56	Geospatial-temporal Group Person Re-identification (Gst-GReID)
    57	can integrate geospatial constraint and spatio-temporal contextual information. By modeling the appearance of groups across
    58	regions and cameras, it enables accurate identity matching and
    59	retrieval in complex scenarios. The existing methods primarily
    60	rely on static distribution of statistics across time-interval cameras and ignore geospatial-temporal reachability, resulting in poor
    61	generalization over long time spans and across regions. Here, we
    62	propose the Geospatial-temporal Graph Network (GstNet). GstNet has two core designs: the Geospatial-temporal Reachability
    63	Module (GstRM) and the Gated Graph-MLP (GGM). GstRM imposes a geospatial-temporal reachability constraint on graph edges,
    64	suppressing geospatial-temporal unreachable connections, thereby
    65	mitigating static time-prior mismatch. GGM introduces channel
    66	gating to achieve selective neighborhood aggregation over long time
    67	spans and across regions, enhancing identity-discriminative representations and suppressing interfering samples. Extensive experimental results on the BRT and SYSU-Group datasets indicate that
    68	our GstNet method outperforms existing state-of-the-art methods.
    69	
    70	Group A
    71	
    72	Unreachable (V > Vmax)
    73	
    74	Previous Method
    75	
    76	Group B
    77	
    78	Reachable (V < Vmax)
    79	
    80	GstNet
    81	
    82	Fig. 1. Illustration of the proposed motivation. Unlike previous methods that overlook geospatial-temporal reachability constraint (e.g., traversing 20 km in 5 s is infeasible under the maximum traffic speed), we introduce such constraint to filter out groups
    83	that are unreachable but have similar appearances.
    84	
    85	is spatio-temporal modeling-based person re-identification. Some
    86	studies have introduced static distribution of statistics across timeinterval cameras [13, 14]. There are also some works that adopted
    87	3D CNN and attention mechanisms to characterize cross-frame temporal sequences and achieve multi-scale temporal modeling and spatiotemporal fusion [15, 16, 17, 18].
    88	The accuracy of existing methods is limited by two major factors. First, most group person re-identification approaches remain
    89	confined to group-level modeling of visual appearance features.
    90	Over long time spans and across regions, such methods struggle to
    91	leverage the geospatial-temporal reachability constraint. Second,
    92	although spatio-temporal modeling methods introduce temporal
    93	factors, they generally rely on a static distribution of statistics
    94	across time-interval cameras. However, static modeling ignores
    95	the geospatial-temporal reachability constraint, which is defined
    96	as the speed of a group derived from distance-time relationships
    97	must not exceed the maximum allowable speed. This often leads
    98	to erroneous matches and degraded discriminative performance. As
    99	illustrated in Fig. 1, previous methods overlook the reachability
   100	constraint, namely the maximum feasible travel range of a group
   101	within a given time interval. If a group traverses a long distance in
   102	an extremely short time, resulting in a speed far beyond the normal
   103	upper limit of traffic [19], the path should be deemed unreachable,
   104	and the groups should be identified as the different. Conversely, if
   105	movement occurs within a reasonable time and the speed remains
   106	below the range of traffic, it should be considered reachable, and the
   107	groups should be identified as the same.
   108	
   109	9952
   110	
   111	ICASSP 2026
   112	
   113	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:22 UTC from IEEE Xplore. Restrictions apply.
   114	
   115	Group Images with labels
   116	
   117	GReID Graph Building
   118	
   119	Geospatial-Temporal Feature Extraction
   120	
   121	Distance Features
   122	
   123	Feature
   124	Fusion Matrix
   125	
   126	...
   127	
   128	GGM Layers
   129	
   130	GstRM
   131	
   132	GMNLayers
   133	Layers
   134	GMN
   135	
   136	Temporal Features
   137	
   138	Visual Feature Extraction
   139	
   140	CE Loss
   141	Ncontrast
   142	Loss
   143	
   144	Vision
   145	Transformer
   146	
   147	Visual Features
   148	
   149	(a) Overview of Geospatial-temporal Graph Network (GstNet)
   150	Clamped Speeds
   151	
   152	Graph-MLP
   153	
   154	Input
   155	
   156	embedding layer
   157	
   158	RELU
   159	
   160	Output
   161	
   162	Linear
   163	
   164	Softmax
   165	Linear
   166	
   167	RELU
   168	
   169	Linear
   170	
   171	After Gate
   172	
   173	Output
   174	: Speed excess
   175	: Penalty coefficient
   176	: Gate span
   177	: Speed limit
   178	
   179	Conv1d
   180	Linear
   181	
   182	Linear
   183	
   184	Norm
   185	
   186	: Lower bound
   187	
   188	Reachability Margin
   189	
   190	Input
   191	
   192	(b) Geospatial-temporal Reachability Module(GstRM)
   193	
   194	(c) Gated Graph-MLP(GGM)
   195	
   196	Fig. 2. Overall architecture of GstNet (a). Step 1: group images are encoded by a ViT to extract visual features, and distance/time
   197	information is processed by GstRM (b) to produce a feature fusion matrix constrained by geospatial-temporal reachability. Step 2: a grouplevel graph is constructed by combining visual features with the feature fusion matrix derived from reachability modeling. Step 3: the graph
   198	is propagated through GGM(c) for selective aggregation and optimized with joint cross-entropy loss and neighborhood contrastive loss.
   199	
   200	To address this issue, we propose Geospatial-temporal Graph
   201	Network (GstNet) for group person re-identification. This framework integrates geospatial-temporal information with group appearance features to enhance matching performance over long
   202	time spans and across regions. GstNet comprises two core modules: Geospatial-temporal Reachability Module (GstRM), which
   203	imposes geospatial-temporal reachability constraint during graph
   204	construction by penalizing unreasonable connections; and Gated
   205	Graph-MLP (GGM), which achieves selective neighborhood aggregation during feature propagation to suppress the mismatches
   206	caused by the absence of reachability constraint. The contributions
   207	of this paper are summarized as follows:
   208	(1) To overcome mismatches over long time spans and across
   209	regions, we propose GstNet, which integrates geospatial-temporal
   210	reachability with group appearance features and reduces errors
   211	through constrained graph construction and gated propagation.
   212	(2) To address unreasonable connections during constrained
   213	graph construction, we propose GstRM, which defines reachability
   214	based on geospatial-temporal information and speed thresholds, and
   215	penalizes edges that violate constraints to reduce incorrect matching.
   216	(3) To reduce the interference of erroneous edges during the
   217	propagation stage, GGM employs channel-wise gating and residual guidance to selectively propagate neighborhood information,
   218	thereby reducing noise accumulation.
   219	
   220	2. METHODOLOGY
   221	2.1. Overall Framework
   222	As illustrated in Fig. 2, the overall pipeline is as follows: first, group
   223	images are fed to the visual branch to extract discriminative appearance features via a Vision Transformer (ViT) [20], while distance
   224	and time attributes are extracted from the source data and sent to
   225	GstRM to produce a reachability-constrained edge matrix. Next, a
   226	graph is constructed with groups as nodes, and reachability penalties
   227	are imposed on edges to effectively filter unreasonable connections.
   228	Then, the graph is processed by GGM for gated feature propagation,
   229	allowing for adaptive and selective aggregation of neighborhood information. Finally, the fused features are jointly optimized using
   230	cross-entropy loss and the Ncontrast loss.
   231	2.2. Geospatial-temporal Reachability Module (GstRM)
   232	To address unreasonable edge connections caused by relying solely
   233	on appearance similarity or static distribution of statistics across
   234	time-interval cameras. For example, two groups that look similar
   235	but are separated by tens of kilometers with only a few seconds
   236	between timestamps, which is physically implausible. We propose the Geospatial-temporal Reachability Module (GstRM). This
   237	
   238	9953
   239	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:22 UTC from IEEE Xplore. Restrictions apply.
   240	
   241	module introduces a speed constraint with a slack margin to dynamically modulate edge strengths, thereby suppressing geospatialtemporal infeasible links. We define the time difference and the
   242	geographic distance. For any two group nodes (i, j), the time difference Tij = |ti − tj |, and the geographical distance Dij = D[i][j].
   243	Based on the geographical distance Dij and the time difference Tij ,
   244	we calculate the speed to measure the rationality of the edge in the
   245	geospatial-temporal dimension:
   246	vij = V(Dij , Tij ) ≜
   247	
   248	Dij
   249	Tij
   250	
   251	(1)
   252	
   253	When vij exceeds the preset physical speed limit vmax , the connection is deemed unreachable and is penalized in subsequent steps.
   254	To further characterize this constraint more rigorously, we define a slack margin function that formally quantifies the reachability
   255	margin under a given time interval:
   256	Mij = M(Dij , Tij ; vmax ) ≜ vmax Tij − Dij
   257	
   258	(2)
   259	
   260	When Mij is positive, the group pair is considered reachable.
   261	After obtaining the geospatial-temporal priors, we model
   262	the matching similarity on edges. We feed (Dij , Tij ) into an
   263	MLP to learn the baseline pairwise compatibility potential ϕij =
   264	MLP(Dij , Tij ), where ϕij denotes the geospatial-temporal similarity potential. To modulate the baseline pairwise compatibility
   265	potential with the slack margin, we design a gating factor:
   266	
   267	
   268	Mij
   269	γij = gmin + (gmax − gmin ) · σ
   270	(3)
   271	β
   272	Where gmin and gmax denote the lower and upper bounds of the gating, β is a smoothing coefficient, and σ(·) is the sigmoid function.
   273	When the slack margin is large, γij approaches the upper bound,
   274	indicating higher confidence for this edge; when the slack margin
   275	approaches zero, γij approaches the lower bound, thereby suppressing the weight of this edge.
   276	For edges whose implied speed vij exceeds the upper bound
   277	vmax , we further introduce a penalty term:
   278	
   279	bij = −κ · ReLU vij − vmax
   280	(4)
   281	Here, κ denotes the penalty-strength hyperparameter that regulates
   282	the magnitude of constraint violations. When the speed constraint
   283	is violated, the penalty term suppresses the corresponding edge
   284	strength , leading to a substantial attenuation of connectivity.
   285	By jointly considering the baseline pairwise compatibility potential, the modulation factor, and the penalty term, the final edge
   286	strength can be expressed as:
   287	
   288	exp γij ϕij + bij
   289	
   290	Eij = P
   291	(5)
   292	k∈N (i) exp γik ϕik + bik
   293	The graph model G(F, E) is defined by a node feature matrix
   294	F = [f1 , . . . , fN ] ∈ RN ×C and the edge strength matrix E =
   295	[Eij ] ∈ RN ×N . Here, fi denotes the visual feature of image xi
   296	extracted by the Vision Transformer (ViT), while Eij specifies the
   297	connection strength between nodes i and j as formulated in Eq. 5.
   298	2.3. Gated Graph-MLP (GGM)
   299	In geospatial-temporal group re-identification (Gst-GReID), groups
   300	that are adjacent in geographic location and time are not necessarily of the same identity; indiscriminate propagation of neighborhood
   301	
   302	information can induce cross-identity interference. Meanwhile, irrelevant groups in the scene may be erroneously amplified during
   303	propagation, degrading discriminability. To address this, we propose
   304	Gated Graph-MLP (GGM), which augments Graph-MLP [21] with
   305	a gating mechanism to enable selective propagation of neighborhood
   306	information. Specifically, the input node features first pass through
   307	a gated convolution to perform channel selection:
   308	
   309	H = Conv1d(F) ⊙ σ(FW) + F
   310	(6)
   311	Where ⊙ denotes element-wise multiplication, F denotes the node
   312	features, and H = [h1 ; . . . ; hN ], hi ∈ RC denotes the gated rep′
   313	resentation of node i, and W ∈ RC×C is the weight matrix of a
   314	linear transformation.
   315	Subsequently, node representations are updated by incorporating
   316	information from their neighbors:
   317	X
   318	h̃i = hi +
   319	αij · hj
   320	(7)
   321	j
   322	
   323	Here, αij is obtained from the edge strengths Eij in Eq. 5 by neighborhood normalization, with αij = softmaxj∈N (i) (Eij ). The
   324	residual connection preserves the node’s own representation while
   325	injecting more discriminative neighborhood information.
   326	2.4. Loss Function
   327	We jointly optimize the model by minimizing the node-level classification cross entropy loss and a ncontrast loss. To enhance the ability
   328	of node embeddings to capture graph structure, we introduce a structural contrastive loss (ncontrast Loss). This loss exploits the structural information encoded in the edge feature tensor E ∈ RN ×N ×de .
   329	For each node i, the positive set is defined as all nodes j such that
   330	∥Eij ∥2 ≥ ϵ, which denotes the threshold for defining positive pairs.
   331	The structural contrastive loss for node i is then defined as:
   332	
   333	P
   334	N
   335	X
   336	j∈Pi exp sim(h̃i , h̃j )/τ
   337	log P
   338	LNC = −
   339	(8)
   340	,
   341	k̸=i exp sim(h̃i , h̃k )/τ
   342	i=1
   343	where h̃i denotes the final embedding of node i after gated propagation, τ is the temperature hyperparameter and ϵ is the threshold for
   344	selecting positive samples.
   345	Finally, the total loss can be written as follows:
   346	Ltotal = LCE + λ · LNC
   347	
   348	(9)
   349	
   350	where λ is a balancing coefficient that controls the trade-off between
   351	classification and structural contrastive objectives.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' - | nl -ba | sed -n '70,599p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    70	identity labels for training a person search model in a fully
    71	supervised (FS) way. However, it is too expensive to acquire
    72	these annotations, which sometimes even raises privacy concerns. To mitigate the reliance on costly manual labels, weakly
    73	supervised (WS) methods have been proposed [9], [10], [11],
    74	which only use pedestrian bounding boxes for training. Yet,
    75	the labeling cost is still high, especially in dense scenes, where
    76	they are a large number of pedestrians to annotate.
    77	In order to fully get rid of manual labels, in this work, we
    78	propose a new setting of unsupervised (US) learning for person
    79	search. First and foremost, we claim that US person search is
    80	not a simple combination of US pedestrian detection [12], [13]
    81	and US re-ID [14], [15]. The additional challenge mainly lies
    82	in the fact that accurate re-ID pseudo-labels become more difficult to achieve due to the low-quality pseudo-boxes. Therefore,
    83	our investigation focuses on generating high-quality pseudolabels, i.e., pseudo-boxes and pseudo-IDs, considering the
    84	relationship between two subtasks. One straightforward way is
    85	to generate pseudo-boxes and pseudo-IDs using state-of-theart pedestrian detectors (e.g., FeatComp [16]) and clustering
    86	algorithms. However, on the one hand, these pseudo-boxes
    87	often exhibit two primary types of noise: 1) localization noise,
    88	where bounding boxes fail to accurately align with person
    89	regions [Fig. 1(a)] and 2) classification noise, where nonperson
    90	objects (e.g., a stroller and bicycle) are incorrectly classified
    91	as persons [Fig. 1(b)]. On the other hand, traditional clustering
    92	methods, such as DBSCAN, have shown limited effectiveness
    93	when handling noisy pseudo-boxes, resulting in only 31.56
    94	pp with respect tomean average precision (mAP) on CUHKSYSU [2]. These challenges become even more difficult to
    95	address in complex scenes, where the background induces
    96	severe disturbances.
    97	Considering the above challenges, we are highly motivated
    98	to use vision-language models (VLMs), in which the natural
    99	language offers a complementary pathway, providing explicit
   100	and unambiguous semantics that are helpful for pseudo-label
   101	generation, against background noise. Specifically, on one
   102	hand, VLMs serve as a bridge between visual features and
   103	textual semantics, facilitating the integration of semanticlevel understanding into pseudo-label generation; on the other
   104	hand, VLMs exhibit remarkable generalization capabilities
   105	across various vision tasks [17], [18]. However, achieving
   106	high-quality pseudo-labels directly through VLMs remains
   107	nontrivial. For example, while SEEM-generated pseudo-boxes
   108	demonstrate superior detection precision through semantic
   109	
   110	2162-237X © 2026 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
   111	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   112	See https://www.ieee.org/publications/rights/index.html for more information.
   113	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.
   114	
   115	This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
   116	2
   117	
   118	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
   119	
   120	and a CLIP ID labeler that aligns text and image features
   121	to produce more accurate and reliable pseudo-IDs.
   122	3) Experiments are conducted on PRW and CUHK-SYSU
   123	datasets. Under the US setting, we establish new baseline
   124	results, where our performance on PRW is competitively
   125	close to and even surpasses some FS and WS methods.
   126	These encouraging results would motivate more work in
   127	this direction.
   128	II. R ELATED W ORK
   129	As this article develops person search methods using cues
   130	from foundation models, we review relevant work from both
   131	of the above perspectives.
   132	A. Person Search
   133	
   134	Fig. 1. Qualitative and quantitative analysis of boxes predicted by SEEM [19].
   135	Green bboxes refer to the ground truth while red ones denote SEEM predicted
   136	bbox. (a) and (b) Samples of low-quality localization and classification
   137	mistakes. (c) Comparison of various foundation models (SEEM [19], SSA
   138	[20], and CLIP+SAM1 ) and a state-of-the-art pedestrian detector (Featcomp
   139	[16] pretrained on CityPersons dataset [21]).
   140	
   141	prompts (e.g., “person” and “pedestrian”) [Fig. 1(c)], they
   142	are still prone to localization and classification noises, which
   143	negatively impact the quality of pseudo-IDs. To mitigate
   144	these disturbances, we propose an anti-bbox-noise re-ID loss,
   145	which drives the model to focus on the foreground person
   146	regions and downweights those boxes with lower confidence
   147	scores in the loss function, effectively reducing the impact of
   148	poor localization and misclassification noise. In addition, we
   149	propose a CLIP ID labeler, which is more robust to noisy
   150	pseudo-boxes by making use of the alignment between image
   151	and text features. Specifically, we first construct one sentence
   152	consisting of predefined attributes for each ID. Subsequently,
   153	each instance can be assigned as a pseudo-ID, whose text
   154	embedding is closest to the instance’s visual features. The
   155	primary pseudo-IDs are further refined by the prior knowledge
   156	of person search, i.e., individuals in the same photo cannot
   157	have identical IDs.
   158	In summary, our contributions can be summarized as follows.
   159	1) We for the first time propose a new setting of US
   160	learning for person search and point out the unique
   161	challenge over previous WS person search, i.e., the lack
   162	of accurate person boxes brings great challenge to the
   163	subsequent subtask of re-ID.
   164	2) We introduce a novel framework leveraging robust
   165	semantics of VLMs to generate and refine pseudolabels for person search. Specifically, we propose an
   166	anti-bbox-noise re-ID loss to mitigate localization and
   167	classification noise in SEEM-generated pseudo-boxes
   168	1 Due to the lack of semantic information in SAM [19], [22], we use SAM
   169	to obtain bboxes and CLIP [23] to remove the bboxes whose category is not
   170	person in the CLIP+SAM method.
   171	
   172	Person search has witnessed significant progress in recent
   173	years, driven by its extensive potential applications. Existing works predominantly fall into two categories: two-stage
   174	methods, which optimize pedestrian detection and re-ID tasks
   175	separately [1], [24], [25], [26], [27], [28], [29], and one-stage
   176	methods, which unify these subtasks by jointly optimizing
   177	them in an end-to-end manner [2], [3], [4], [5], [6], [7], [8],
   178	[9], [10], [11], [30], [31], [32], [33], [34], [35], [36], [37],
   179	[38]. Broadly, current person search methods can be classified
   180	into two distinct groups based on their reliance on the labels:
   181	FS and WS methods.
   182	1) Fully Supervised: This category leverages comprehensive
   183	annotations, including pedestrian bounding boxes and identity
   184	labels, to train models directly [3], [4], [6], [7], [8], [36],
   185	[37], [38]. Xiao et al. [2] pioneer a unified framework that
   186	integrates re-ID layers atop the Faster-RCNN detector [39].
   187	Chen et al. [3] introduce a norm-aware embedding (NAE)
   188	method to harmonize the divergent optimization goals of
   189	detection and re-ID. In addition, recent approaches [36], [37]
   190	incorporate transformers to extract more discriminative feature
   191	representations, thereby enhancing performance.
   192	2) Weakly Supervised: Due to the challenge of obtaining
   193	identity labels, this group of approaches utilizes only bounding
   194	box annotations for training [9], [10], [11]. Yan et al. [9]
   195	exploit contextual information to derive discriminative features
   196	for a robust US re-ID task. Wang et al. [11] introduce a
   197	multiscale exemplar branch and devise a scale-invariant loss
   198	to tackle scale variability issues.
   199	3) Unsupervised: Unlike the aforementioned methods, US
   200	person search eliminates the need for accurate annotations,
   201	focusing on learning effective feature representations from
   202	fully unlabeled datasets. Different from US domain-adaptive
   203	person search methods [40], which rely on labeled sourcedomain data together with unlabeled target-domain images
   204	for training, this setting does not assume the availability of
   205	any labeled data from any domain. While no US methods
   206	are specifically designed for person search under this strictly
   207	label-free setting, various techniques [12], [13], [41], [42]
   208	exist for its subtasks, including pedestrian detection [12], [13],
   209	and re-ID [14]. Liu et al. [12] propose a US multiplane
   210	detection (UMPD) method that removes the necessity for
   211	pedestrian bounding box annotations through 2-D-3-D mapping. Han et al. [14] develop an innovative sampling strategy
   212	
   213	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.
   214	
   215	This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
   216	TIAN et al.: HARNESSING KNOWLEDGE FROM PRETRAINED VLMs FOR US PERSON SEARCH
   217	
   218	3
   219	
   220	to refine the re-ID pseudo-label generation process. However,
   221	the integration of pedestrian detection and re-ID in a US
   222	setting presents unique challenges, particularly as detection
   223	performance and the presence of noisy pseudo-boxes can
   224	significantly influence the quality of person representations.
   225	This aspect remains underexplored in existing literature. In
   226	this work, we introduce a US setting to person search for the
   227	first time, aiming to utilize cutting-edge foundation models to
   228	address the challenge of US person search.
   229	B. Foundation Models
   230	Foundation models have rapidly advanced and been applied
   231	across diverse domains, spanning interactive segmentation
   232	models (SAM [22] and SEEM [19]), VLMs (CLIP [23]),
   233	and human-centric approaches (SOLIDER [43], UniHCP [44],
   234	and PATH [45]). Among these, SAM establishes a new
   235	paradigm for segmentation by supporting diverse prompt
   236	inputs, while SEEM further enhances its semantic understanding and prompting capabilities. CLIP has gained significant
   237	attention due to its strong zero-shot recognition performance,
   238	leading to its application in diverse vision tasks, including
   239	person re-ID, crowd counting, and semantic segmentation.
   240	Researchers have creatively adapted its cross-modal alignment
   241	capabilities. For instance, Li et al. [18] develop CLIP-ReID,
   242	which utilizes learnable text tokens for identity description,
   243	while Liang et al. [17] reformulate crowd counting as an
   244	image–text matching task. Human-centric models like UniHCP, PATH, and SOLIDER demonstrate the potential of
   245	integrating diverse datasets to build comprehensive humanfocused models. These approaches not only provide robust
   246	human representations but also enhance performance across
   247	downstream visual tasks.
   248	Beyond task-specific applications, researchers explore foundation models as sophisticated data engineering tools. SSA
   249	[20], for instance, serves as an automated dense openvocabulary annotation engine, combining closed-set segmentation [46], open-vocabulary techniques [47], [48], and
   250	intelligent class filtering [23]. Inspired by such innovations,
   251	our work investigates foundation models’ potential in guiding
   252	US person search.
   253	III. M ETHOD
   254	In this section, we begin with an overview of our one-stage
   255	US person search method, termed FMUPS-S1, followed by an
   256	explanation of the proposed CLIP ID labeler. Due to the noise
   257	introduced by pseudo-boxes, we present in detail our antibbox-noise re-ID loss. In addition to FMUPS-S1, we present
   258	another implementation: a two-stage approach, FMUPS-S2.
   259	The details and individual components are described in
   260	Section III-E.
   261	A. Overview
   262	The VLMs are used to provide high-quality pseudo-labels
   263	for both detection and re-ID. As depicted in Fig. 2, our
   264	FMUPS-S1 architecture includes two streams: a scene stream
   265	and an instance stream. In the instance stream, each image
   266	x is first passed through the frozen SEEM model, where we
   267	
   268	Fig. 2. Overview of our one-stage method FMUPS-S1. The pipeline consists
   269	of a scene stream (dashed lines) and an instance stream (solid lines).
   270	Specifically, SEEM is employed for generating pseudo-boxes, while the
   271	CLIP ID labeler is utilized to provide pseudo-IDs. Our novel anti-bboxnoise re-ID loss effectively reduces the negative effect of both classification
   272	and localization noise by leveraging confidence scores ci and emphasizing
   273	foreground information, respectively. RPN is the region proposal network [39].
   274	
   275	obtain all person masks by utilizing the person-related prompts
   276	(e.g., “person”). We then derive the pseudo-box y
   277	bbi = ϕ(mi )
   278	for each mask mi , where ϕ(·) denotes the function that computes the coordinates of the minimum bounding rectangle.
   279	Subsequently, the image x is cropped according to y
   280	bb to
   281	b
   282	b
   283	b
   284	extract box patches x = {x1 , . . . , xN }, which are then further
   285	processed with corresponding masks to derive the foreground
   286	patches xm = {x1m , . . . , xmN }, where the background regions are
   287	filled with zero. Here, N is the number of bounding boxes
   288	in an image. These patches, xm and xb , are subsequently
   289	input into the image encoder to extract respective box features
   290	f b = { f1b , . . . , fNb } and mask features f m = { f1m , . . . , fNm }. The
   291	box features f b are then processed through the CLIP ID labeler
   292	to generate and further refine pseudo-IDs. Concurrently, in
   293	the scene stream, the entire image x is sent to the image
   294	encoder shared with the instance stream, followed by an RPN,
   295	a detection head, and a re-ID head. RPN aims to efficiently
   296	generate high-quality proposals from the entire image that
   297	are likely to contain pedestrians. The detection and re-ID
   298	heads follow OIM [2]. After that, we obtain proposal features
   299	f p = { f1p , . . . , f Jp } along with their associated confidence
   300	scores, where J represents the number of predicted proposals.
   301	In order to reduce the negative impact of noisy pseudo-boxes
   302	on the performance of re-ID, we propose an anti-bbox-noise
   303	re-ID loss. The use of these features, f b , f m , and f p , in this
   304	loss allows the model to focus on the foreground information,
   305	thereby mitigating localization noise, while the utilization
   306	of confidence scores aids in diminishing classification noise
   307	induced by pseudo-boxes. Further elaboration on the CLIP
   308	ID labeler and anti-bbox-noise re-ID loss is provided in
   309	Sections III-B and III-C.
   310	B. CLIP ID Labeler
   311	The CLIP ID labeler is designed to generate accurate
   312	pseudo-IDs for each sample as shown in Fig. 3. It comprises
   313	
   314	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.
   315	
   316	This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
   317	4
   318	
   319	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
   320	
   321	Fig. 3. Pipeline of CLIP ID labeler. Projected box features I b are obtained through our pretrained CLIP projector, which maps the box features into the CLIP
   322	feature space. Text prompts are generated from predefined sets of attributes, and pseudo-IDs are assigned based on the highest similarity scores between text
   323	embeddings D and I b . These pseudo-IDs are subsequently refined through a denoising procedure that utilizes prior knowledge.
   324	
   325	a frozen CLIP text encoder and a trainable CLIP projector,
   326	leveraging the CLIP text–image matching principle to generate pseudo-IDs. These pseudo-IDs are subsequently refined
   327	through the application of prior knowledge that each individual
   328	appears only once per image, thereby enhancing the accuracy
   329	of the generated pseudo-IDs.
   330	1) Pretraining of the CLIP Projector: Since the pretraining
   331	parameters of the CLIP image encoder are not suitable for
   332	fine-grained task i.e., re-ID, we adopt a more suitable image
   333	encoder [43], which is pretrained on an unlabeled re-ID dataset
   334	[49]. However, this introduces a new challenge: the feature
   335	space produced by this image encoder is not aligned with that
   336	of the original CLIP image encoder. To resolve this problem
   337	and align the feature space, it is imperative to pretrain the CLIP
   338	projector Pθ . The optimization objective for this pretraining is
   339	defined as Lalign = ||Pθ ( f b ) − f clip ||2 , where f clip refers to the
   340	features obtained after image xb passes through the frozen
   341	CLIP image encoder; Lalign represents the mean squared error
   342	loss, which is strategically utilized to align the features f clip
   343	and f b .
   344	2) Design of Text Prompts: The CLIP model, not inherently designed for the person search task, necessitates an
   345	investigation into the customization of text prompts. Typically,
   346	various attribute terms, such as gender and clothing color,
   347	are employed to describe individuals. Operating under the
   348	assumption that individuals sharing similar attributes are likely
   349	the same person, we meticulously construct diverse attribute
   350	combinations to uniquely define and differentiate individuals.
   351	The text prompt is specifically crafted as follows: “This is a
   352	[A1 ] person with [A2 ], [A3 ] top, [A4 ], . . ., [Ai ], . . ., [AK ].” Here,
   353	“[Ai ]” represents the different selectable values for attribute
   354	Ai , and K denotes the total number of attribute types. These
   355	text prompts are subsequently processed by the CLIP text
   356	encoder to generate the text embedding D = [D1 , D2 , . . . , D M ],
   357	where D ∈ R M × C and M signifies the total number of text
   358	embeddings, i.e., the total number of person identities. C is
   359	the channel number of the embeddings.
   360	
   361	3) Generation and Denoising of Pseudo-IDs: Consider a
   362	scenario where we have a set of projected box embeddings
   363	I b ∈ RN × C derived from the CLIP projector, and text embeddings D ∈ R M × C obtained from the CLIP text encoder, where
   364	I b = [I1b , I2b , . . . , INb ] and each Iib = Pθ ( fib ). We can obtain the
   365	pseudo-ID of a given box embedding Ii with the following
   366	formulation:
   367	eid
   368	y
   369	(1)
   370	i = arg max si, j
   371	j
   372	
   373	where 0 < i ≤ N and 0 < j ≤ M; the term si, j is the element
   374	of the similarity matrix S ∈ RN × M , which is calculated as
   375	T
   376	
   377	
   378	
   379	(2)
   380	S = ξ I1b · DT , ξ I2b · DT , · · · , ξ INb · DT
   381	where Iib · DT refers to inner product of Iib and DT ; ξ(·) is the
   382	softmax function.
   383	However, a common challenge arises when different individuals possessing similar attributes are mistakenly identified as
   384	the same person, particularly when they are wearing identical
   385	uniforms within a single image. This scenario introduces noise
   386	into the pseudo-IDs. Given the prior that “a person can only
   387	appear once in a single image,” we can refine the pseudo-IDs
   388	to mitigate this noise. Specifically, within an image, consider
   389	two projected box embeddings Iib , I bj that are assigned the
   390	same pseudo-ID “d.” If si,d < s j,d , then I bj , which exhibits
   391	the highest similarity, retains the pseudo-ID “d.” Conversely,
   392	the pseudo-ID for Iib , which has a lower similarity score, is
   393	reassigned based on the highest similarity with the remaining
   394	IDs. This process continues until all conflicts within the image
   395	are resolved (i.e., all pseudo-IDs are assigned unique values).
   396	The corresponding equation can be expressed as
   397	8
   398	
   399	
   400	ˆ
   401	eid
   402	if si,eyidi = max sc,eyidi | c ∈ C
   403	<y
   404	i ,
   405	bid
   406	y
   407	(3)
   408	i =
   409	si,z , otherwise
   410	ˆ
   411	:arg max
   412	id
   413	z,z,e
   414	yi
   415	
   416	where C is the list of projected box embedding indices. As the
   417	model undergoes training, the image features are continuously
   418	updated. Correspondingly, the pseudo-IDs are also updated
   419	
   420	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.
   421	
   422	This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
   423	TIAN et al.: HARNESSING KNOWLEDGE FROM PRETRAINED VLMs FOR US PERSON SEARCH
   424	
   425	5
   426	
   427	Algorithm 1 Generation and Denoising of Pseudo-IDs inCLIP
   428	ID Labeler
   429	
   430	Fig. 4. Overview of our novel anti-bbox-noise re-ID loss. It incorporates two
   431	weighted foreground focusing losses: (a) Lw f f in in the instance stream and
   432	(b) Lw f f se in the scene stream. Both of them mitigate the negative effects
   433	of classification and localization noise by utilizing confidence scores ci and
   434	emphasizing foreground information. Features represented by identical colors
   435	correspond to the same pseudo-ID.
   436	
   437	iteratively. The complete process of generation and denoising
   438	is summed up in Algorithm 1.
   439	C. Anti-Bbox-Noise Re-ID Loss
   440	As shown in Fig. 1, pseudo-boxes can introduce two types
   441	of noise, i.e., localization noise and classification noise, both of
   442	which can detrimentally impact the subsequent re-ID task. To
   443	mitigate the negative effects of the above noises, we propose
   444	an anti-bbox-noise re-ID loss. For localization noise, these
   445	pseudo-boxes often introduce redundant background information around the target person. Consequently, we align the
   446	re-ID features (i.e., box features or proposal features) closely
   447	with the mask features to reduce the influence of unnecessary
   448	background information. For classification noise, the pseudoboxes may contain objects other than people. Therefore, we
   449	use confidence scores to assess the accuracy of each bounding
   450	box and dynamically adjust the weights for different instances,
   451	thereby minimizing the impact of misclassification.
   452	We first propose a weighted foreground focusing loss,
   453	Lw f f in , in instance stream as shown in Fig. 4. This loss
   454	dynamically drives the person search model to focus on the
   455	foreground information by aligning the box features f b with
   456	the mask features f m . In addition, it helps to acquire effective
   457	re-ID features by aligning mask features from different views
   458	that share identical pseudo-labels y
   459	bid and pushing away the
   460	features between different individuals and pulling the features
   461	of the same person together. The loss Lw f f in is written as
   462	i, j=Q
   463	
   464	Lw f f in =
   465	
   466	ˇ
   467	
   468	ˇ ˇ
   469	
   470	1 X b ˇˇ
   471	ci · ζ fib , f jm − δ ˇ + ˇζ fim , f jm − δˇ (4)
   472	Q2
   473	i, j=1
   474	
   475	where Q denotes the total number of the box and mask features
   476	within a single batch. δ = [b
   477	yid
   478	bidj ], where the Iverson bracket
   479	i =y
   480	indicator function [·] evaluates to 1 when fi and f j correspond
   481	to the same individual, i.e., y
   482	bid
   483	bidj , and 0 otherwise. The
   484	i = y
   485	function ζ(·) quantifies the cosine similarity. The confidence
   486	score cbi for the box feature fib is directly obtained from the
   487	detection head, expressed as cbi = Det(xib ). This score is then
   488	
   489	assigned to fib as a weighting factor, which serves to represent
   490	the confidence that the feature fib corresponds to a foreground
   491	object.
   492	In addition, a weighted foreground focusing loss, Lw f f se , is
   493	also introduced in the scene stream. This loss is distinct from
   494	Lw f f in in that the re-ID feature, which aligns closely with
   495	the mask features f m in Lw f f se , is the space-shifted proposal
   496	feature f p rather than the box feature f b . The loss Lw f f se is
   497	defined as
   498	Lw f f se =
   499	
   500	i, j=H
   501	
   502	ˇ ˇ
   503	 ˇ
   504	1 X p ˇˇ
   505	ci · ζ fip , f jm − δ ˇ + ˇζ fim , f jm − δˇ (5)
   506	2
   507	H
   508	i, j=1
   509	
   510	where H denotes the total number of proposal and mask
   511	features within a single batch.
   512	Therefore, the formulation of the anti-bbox-noise re-ID loss
   513	is presented as follows:
   514	Lreid =
   515	
   516	R
   517	X
   518	
   519	Lw(r)f f in + Lw(r)f f se
   520	
   521	(6)
   522	
   523	r=1
   524	
   525	where R is the total number of iterations during the entire
   526	training.
   527	D. Overall Training Objective
   528	The total loss function for our FMUPS-S1 is defined as
   529	follows:
   530	LFMUPS-S1 = Ldet + Lreid + Lcon
   531	(7)
   532	where Ldet is the detection loss based on Faster-RCNN [39].
   533	Lcon is a typical contrastive loss [50] toward to projected box
   534	features I b and projected mask features I m , where I m = Pθ ( f m ).
   535	Lreid is introduced in Section III-C.
   536	E. Two-Stage Method
   537	As discussed in Section II-A, person search methods can
   538	be classified into one-stage and two-stage methods depending
   539	on whether their subtasks are optimized within the same
   540	framework. One-stage methods optimize both subtasks within
   541	a single framework, while two-stage methods optimize them
   542	
   543	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.
   544	
   545	This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
   546	6
   547	
   548	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
   549	
   550	A. Datasets and Evaluation Metrics
   551	1) Datasets: Two widely used datasets for person search
   552	are PRW [1] and CUHK-SYSU [2]. Table I provides statistics
   553	on the training sets of these datasets, highlighting differences
   554	in data usage across US, FS, and WS methods. Here, Bbox
   555	denotes bounding box labels, and ID refers to identity labels.
   556	The PRW dataset includes a test set with 2057 queries and
   557	6112 gallery images. In contrast, the CUHK-SYSU dataset,
   558	a larger and more comprehensive benchmark, contains 2900
   559	query individuals and 6978 gallery images in its test subset.
   560	Unlike PRW, which uses all test images as a gallery, CUHKSYSU defines variable gallery sizes ranging from 50 to 4000.
   561	Unless specified otherwise, our experiments use the default
   562	gallery size of 100.
   563	2) Evaluation Metrics: To assess the performance of person
   564	search methods, we employ the mAP and Top-k scores.
   565	B. Implementation Details
   566	
   567	Fig. 5. Overview of our two-stage method FMUPS-S2. The confidence
   568	generator is designed to calculate a confidence score for each proposal
   569	based on its projected box features I b . (a) Detection stage. (b) Re-ID stage.
   570	(c) Confidence generator.
   571	
   572	separately. Although one-stage methods are more efficient,
   573	they generally exhibit lower performance compared to twostage methods [3], [7]. Therefore, in addition to the one-stage
   574	method FMUPS-S1, we also provide a two-stage method,
   575	termed FMUPS-S2. The overview of FMUPS-S2 is shown in
   576	Fig. 5. There are two differences between the architectures of
   577	FMUPS-S1 and FMUPS-S2.
   578	1) FMUPS-S2 only has the instance stream of Fig. 2. It
   579	utilizes the frozen SEEM as a detector without training
   580	and employs noisy pseudo-boxes for the subsequent reID stage.
   581	2) The way of getting confidence scores is different. Due
   582	to the absence of a detection head, we cannot obtain
   583	confidence scores from SEEM [19]. Instead, we leverage
   584	the generalization ability of CLIP [23] to obtain these
   585	scores. Specifically, as shown in Fig. 5(c), we generate
   586	the text embeddings D0 by constructing the text prompt
   587	“This is [not] a person.” The confidence score for
   588	the re-ID feature fib is then calculated by measuring
   589	the similarity between Iib and the text embeddings D0 .
   590	Therefore, the total loss for our FMUPS-S2 is
   591	LFMUPS-S2 = Lw f f in + Lcon .
   592	
   593	(8)
   594	
   595	IV. E XPERIMENTS
   596	In this section, we first introduce the datasets, evaluation
   597	metrics, and implementation details. We then provide a comprehensive analysis of our method, followed by a comparative
   598	evaluation of state-of-the-art person search methods across
   599	various experimental setups.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '90,920p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    90	To address occlusion problems, some researchers [1], [6],
    91	[16], [17], [18] have shown that fine-grained local features are
    92	effective for dealing with occlusion problems. For example,
    93	Miao et al. [17] proposed a Re-ID framework, which extracts
    94	visible pedestrian parts by integrating pose information. Ma et
    95	al. [18] proposed a novel method, named Pirt, to obtain robust
    96	feature representations by constructing groups of regions and
    97	masks. Recently, Yan et al. [16] proposed an innovative
    98	lightweight network (PRE-Net), which constructs more robust
    99	local features through a reasonable segmentation strategy.
   100	However, most of these methods rely on additional pose
   101	estimators to locate the visible regions, which consumes a
   102	lot of computing resources. To overcome these limitations,
   103	researchers have begun to explore methods that do not rely on
   104	external tools. In this context, ViT [19] has attracted attention
   105	due to its strong performance in processing image sequences.
   106	ViT can capture global and local context information through
   107	its self-attention mechanism, which is crucial for handling
   108	occlusion problems. Therefore, to better extract fine-grained
   109	person features while reducing the reliance on auxiliary pose
   110	
   111	1051-8215 © 2024 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
   112	and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   113	See https://www.ieee.org/publications/rights/index.html for more information.
   114	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   115	
   116	LI et al.: HGTDR NETWORK FOR OCCLUDED PERSON RE-IDENTIFICATION
   117	
   118	Fig. 1. (a) The performance of the previous methods severely decreases when
   119	the person image is occluded. (b) The challenges of occluded person Re-ID.
   120	(c) The motivation of our HGTDR: 1) The key to correctly identify the target
   121	pedestrian lies in the discriminative local details. 2) Compared with previous
   122	hand-crafted strip-based methods, which may introduce noise in the red box,
   123	our method can adaptively extract heterogeneous person local features with
   124	different scales and shapes. 3) How to effectively recover occluded person
   125	features from visible body parts of the pedestrian.
   126	
   127	estimators, many researchers have shifted their focus to the
   128	ViT and proposed many transformer-based models [8], [20],
   129	[21], [22]. Their main idea is to aggregate similar semantic
   130	information into a predefined part token, thereby achieving disentangling of person image. However, most of the
   131	methods rely solely on complex semantics for decoupling,
   132	which inevitably introduces noise interference and hinders
   133	the extraction of discriminative local features. Furthermore,
   134	even though they can extract fine-grained features, the models
   135	might lose robustness when the discriminative appearance is
   136	occluded.
   137	In view of the above limitations, some researchers explored
   138	another occluded Re-ID method [10], [23], [24], [25], [26],
   139	which aims to recover the invisible regions of pedestrians.
   140	Some of them [24], [26] used generative adversarial networks
   141	to produce the holistic image by restoring the occluded parts.
   142	With the emergence of transformer models, some recent
   143	advances [23], [25] have yielded promising solutions. They
   144	primarily concentrate on constructing the feature set of the
   145	retrieval results and then recovering the features of the
   146	occluded probe from the k-nearest neighbor features within
   147	this set. Recently, Wang et al. [10] proposed a Feature
   148	Completion Transformer (FCFormer) to combat occlusion and
   149	complete invisible body features. However, it relies on a specific data augmentation strategy. In addition, none of them take
   150	into account the issue of neighbor weight for occlusion recovery, which leads to less robust recovered features. Alongside
   151	
   152	5023
   153	
   154	the above research ideas, it is of great application significance
   155	to design a multi-effect network that can effectively extract
   156	discriminative local features and dynamically recover invisible
   157	regions.
   158	In this paper, we investigate how to improve the extraction
   159	ability of local semantics, adaptively disentangle discriminative body parts, and dynamically recover features of occluded
   160	regions. This is because the key to identify the target pedestrian is to extract discriminative local details (as shown in
   161	Fig. 1(c).1)), noise-free human bodies are more discriminative than the local features generated by rigid cutting (as
   162	shown in Fig. 1(c).2)), and recovered person features are
   163	more robust than occluded features (as shown in Fig. 1(c).3)).
   164	At the same time, in order to effectively deal with the challenges in Fig. 1(b), we propose a multi-effect Heterogeneous
   165	Generative Token and Distance-aware Recovery (HGTDR)
   166	network. As shown in Fig. 2, HGTDR is a dual-stream
   167	architecture, which mainly consists of a holistic stream and
   168	a part stream. The holistic stream can provide aggregated
   169	long-range global information to the part stream. In the
   170	part stream, we focus on extracting local semantics and
   171	constructing discriminative local features without introducing
   172	any additional part tokens and auxiliary networks. Firstly,
   173	a Semantic Patch Generator (SPG) is proposed, which uses
   174	the spatial pooling and local attention to capture rich local
   175	semantics and further obtain discriminative semantic patches.
   176	Its design enables the network to extract finer and richer features from local regions, which provides basic local semantic
   177	units for subsequent feature processing. Subsequently, based
   178	on the semantic patches generated by SPG, we propose an
   179	Adaptive Heterogeneous Semantic Token Generator (AHSTG)
   180	to obtain identity-related fine-frained features. Guided by the
   181	global features of the holistic stream, AHSTG calculates
   182	the comprehensive scores by assessing the discrimination
   183	and relevance between semantic patches generated by SPG.
   184	The scores are used to adaptively generate strong-response
   185	and weak-response heterogeneous tokens, which effectively
   186	highlights the target person’s body components and suppresses occlusions. However, when in complex occlusion
   187	scenarios, the heterogeneous tokens may struggle to achieve
   188	excellent retrieval performance due to the lack of discriminative appearance features. To address this issue, we design
   189	a Distance-based Feature Recovery (DFR) module, which
   190	recovers the features of occluded regions by dynamically
   191	allocating adaptive weights to neighboring features. Finally,
   192	we obtain robust person features to handle the complex person
   193	Re-ID tasks.
   194	In summary, the main contributions of this paper can be
   195	summarized as follows:
   196	(1) A new Semantic Patch Generator (SPG) is proposed
   197	to capture the local semantics of the image. It can enhance
   198	the local extraction ability of the network and further obtain
   199	discriminative semantic patches.
   200	(2) We propose a flexible Adaptive Heterogeneous Semantic Token Generator (AHSTG), which takes into account
   201	both discrimination and relevance scores to select salient
   202	semantic patches. It can further help our network adaptively
   203	generate heterogeneous tokens with different responses to
   204	
   205	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   206	
   207	5024
   208	
   209	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025
   210	
   211	Fig. 2. The pipeline of our proposed HGTDR. The framework mainly consists of a holistic stream (Section III-B) and a part stream (Section III-C). Our
   212	proposed SPG, AHSTG, and DFR modules are in the part stream. Here, the patch embedding layer and shallow transformer blocks act as base module and
   213	are not expanded in detail anymore. ‘⊕’ represents the concatenation operation. ‘CLS’ represents the class token. In the holistic stream, the sequence below
   214	CLS is the updated patch tokens by patch embeddings Pe . The blue arrows represent the holistic stream, the green arrows represent the part stream, and the
   215	gray arrows represent the interaction of the two streams.
   216	
   217	suppress occlusion. More importantly, it can flexibly generate
   218	fine-grained tokens with different scales and shapes.
   219	(3) A novel Distance-based Feature Recovery (DFR) module is designed to automatically mine implicit information
   220	to recover the occluded body parts. Different from previous
   221	methods, we consider the contribution to occlusion recovery
   222	from the perspective of distance, and can adaptively recover
   223	the discriminative person features of the occluded regions.
   224	(4) Extensive experiments on two authoritative occluded
   225	Re-ID datasets demonstrate the effectiveness and superiority
   226	of our method. In addition, we confirmed that our method has
   227	good generalization ability on the holistic Re-ID datasets.
   228	II. R ELATED W ORK
   229	A. General Person Re-ID
   230	The purpose of person Re-ID is to match the target person
   231	across time and space from the images captured by a set
   232	of non-overlapping distributed cameras. In the early stages
   233	of research, some studies [2], [3], [27], [28] mainly utilize
   234	holistic person images to match the target person, which
   235	mainly focus on two aspects: feature learning [2], [3], [14],
   236	[29], [30] and attention-based [12], [28], [31], [32], [33],
   237	[34]. For example, Zhou et al. [35] proposed an effective
   238	Foreground Attention Neural Network (FANN) to enhance the
   239	attention on the foreground and learn discriminative feature
   240	representation for person Re-ID. To address the issue of
   241	viewpoint misalignment, Zhang et al. [3] proposed a View
   242	Confusion Feature Learning (VCFL) method to learn the
   243	view-invariant features by using a view confusion learning
   244	mechanism. In the attention-based methods, Zhang et al. [28]
   245	proposed a plug-and-play Relational-aware Global Attention
   246	(RGA) module to capture global context information for better
   247	focusing identity-related regions. Recently, transformer [36]
   248	
   249	has rapidly dominated the computer vision field with its
   250	powerful Multi-head Self-Attention (MSA) mechanism. Some
   251	researchers worked on transformer-based person Re-ID and
   252	achieved remarkable performance. Although these methods
   253	achieve good performance when dealing with the holistic
   254	person Re-ID problem, they greatly ignore the existence of
   255	occlusions.
   256	B. Fine-Grained Feature Matching
   257	Compared with the general person Re-ID, the person Re-ID
   258	methods using fine-grained features [1], [7], [37], [38], [39],
   259	[40] can effectively deal with the occluded Re-ID problems.
   260	They employed a part-to-part matching strategy to retrieve the
   261	target person by extracting fine local features. For example,
   262	Sun et al. [1] proposed a Part-based Convolutional Baseline
   263	(PCB) method combined with RPP to obtain fine-grained local
   264	features, which further improves the performance of person
   265	Re-ID. Tan et al. [40] proposed a Continuous Batch DropBlock
   266	Network (CBDB-Net), which can capture pedestrian robust
   267	fine-grained descriptors for person Re-ID tasks. Although
   268	the performance improvement is significant, these methods
   269	become inefficient when facing occlusion or different scales.
   270	To solve the above problems, He et al. [37] proposed a Deep
   271	Spatial feature Reconstruction (DSR) method to avoid the error
   272	in matching images at different scales. Sun et al. [7] proposed a
   273	Visibility-aware Part Model (VPM) to automatically recognize
   274	visible human regions under self-supervision. However, these
   275	methods usually require the manual definition of the scale
   276	and shape of fine-grained features, which limits the scalability
   277	of the model. In contrast, our proposed method is general
   278	and flexible. It not only can adaptively construct fine-grained
   279	heterogeneous features, but also can be applied to diverse
   280	situations in the real world.
   281	
   282	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   283	
   284	LI et al.: HGTDR NETWORK FOR OCCLUDED PERSON RE-IDENTIFICATION
   285	
   286	C. Occluded Person Re-ID
   287	The occluded Person Re-ID task is challenging due to the
   288	lack of information and spatial misalignment in the occluded
   289	scenarios. To deal with these problems, previous studies [6],
   290	[11], [13], [17], [32], [33], [41] focus on auxiliary cues such as
   291	pose keypoints, graph networks, and spatial masks to weaken
   292	the interference of occlusion noise. For instance, Miao et
   293	al. [11] proposed a Pose-Guided Feature Alignment (PGFA)
   294	network that uses a cumbersome pose estimation model to
   295	guide the network to focus on non-occluded body areas. Gao et
   296	al. [6] proposed a Pose-guided Visible Part Matching (PVPM)
   297	method, which learns the visible local features guided by a
   298	pose estimation model. Wang et al. [13] proposed a HighOrder Re-ID (HoReID) network that integrates the high-order
   299	relationship and topological structure of person features using
   300	a graph network. However, these methods all require additional pre-trained networks, which may lead to domain bias
   301	between person Re-ID datasets and the pre-trained datasets.
   302	To get rid of the external auxiliary networks, some researchers
   303	proposed several attention-based methods. For example, Ye et
   304	al. [42] proposed a novel Dynamic Dual Attention Aggregation
   305	(DDAG) learning method to extract discriminative partial
   306	aggregation features. Xia et al. [43] proposed a novel Attention
   307	Disturbance and Dual-Path Constraint Network (ADP) based
   308	on transformer, which enhances the generalization network of
   309	attention. Li et al. [44] proposed a novel approach, called
   310	OAT, to capture higher-order relationships between different
   311	body parts or regions. Although these methods have achieved
   312	significant performance improvement in dealing with occluded
   313	Re-ID, they only consider the saliency of features and ignore
   314	the correlation with pedestrian identity.
   315	D. Feature Recovery
   316	To address the challenge of missing features and alleviate
   317	the interference of occlusion noise, some works [10], [23],
   318	[24], [25], [26] are devoted to recovering the occluded part
   319	of the pedestrian. Some previous methods [24], [26] combine
   320	the generative adversarial network to complete the human
   321	occluded parts, and then carry out feature extraction and
   322	matching. These approaches require massive images of the
   323	same person to train the network to recover the occluded
   324	parts well. However, it is impossible for the person Re-ID
   325	tasks to achieve this goal due to the limitation of datasets.
   326	Later, some recovery methods [10], [23], [25] based on feature
   327	transmission emerged. For example, Hou and colleagues [23]
   328	introduced a lightweight Region Feature Completion (RFC)
   329	block as a solution for addressing the issue of image occlusion.
   330	To fully leverage the benefits of global information interaction in the transformer, Xu et al. [25] proposed a Feature
   331	Recovery Transformer (FRT) model, which learns the available
   332	information from the k-nearest neighbors in the gallery to
   333	recover the invisible features of the probe. Subsequently,
   334	the recovered features are re-matched. Recently, Wang et
   335	al. [10] proposed a new framework named Feature Completion Transformer (FCFormer), which enriches the occluded
   336	samples in the training set by simulating real and diverse
   337	occlusion situations, thereby improving the robustness of the
   338	
   339	5025
   340	
   341	network. Then, the FCFormer aggregated possible information
   342	from the self-generated occlusion features in a self-supervised
   343	manner to complement the features of the occluded regions.
   344	However, many of the above methods either relied on additional pre-trained networks or failed to consider the effect of
   345	neighborhood weight on recovery quality.
   346	Different from the above methods, our proposed method
   347	eliminates the need for a complex external network, and
   348	employs a dynamic weight allocation strategy based on
   349	distance perception. Our work reasonably allocates weights
   350	through the calculated distance, effectively simulates the real
   351	occlusion situation, and dynamically recovers the occluded
   352	regions. Through the novel weight allocation mechanism, our
   353	method is able to escape the dependence on additional networks without sacrificing performance, while fully considering
   354	the importance of neighborhood features in the occlusion
   355	recovery.
   356	III. P ROPOSED M ETHOD
   357	In this section, we introduce the proposed Heterogeneous
   358	Generative Tokens and Distance-aware Recovery (HGTDR)
   359	framework in detail. In Section III-A, we first give a concise
   360	overview of our network. Next, the structure of the holistic
   361	stream is introduced in detail in Section III-B, and the components of the part stream are highlighted in Section III-C.
   362	Finally, the objective loss function and optimization strategy
   363	are described in Section III-D.
   364	A. Overview
   365	The proposed HGTDR network is mainly composed of a
   366	holistic stream and a part stream, as shown in Fig. 2. Initially,
   367	we use a basic transformer model to extract the low-level
   368	semantics and establish long-range global relationships from
   369	person images. Subsequently, the output features are fed
   370	into the holistic stream and the part stream. In the holistic
   371	stream, we employ the ViT-B/16 [19] to capture contextual
   372	information and obtain stable global features. While in the
   373	part stream, the output features are initially divided into
   374	multiple windows, and spatial pooling is performed on each
   375	window to obtain clustering centers. They aggregate relevant
   376	information from both the original features and the global
   377	features of the holistic stream to generate semantic patches.
   378	Next, the semantic patches are input into the scoring network
   379	to select the most obvious patches and their indexes are
   380	recorded. The selected patches are used to adaptively generate
   381	K discriminative heterogeneous tokens with rich semantics.
   382	These tokens are then fed into the decoder to enhance the
   383	network’s robustness. We assess their distances and insert
   384	a recovery token between them to dynamically recover the
   385	missing features. Thus, we can obtain robust and discriminative local features. Finally, the local and global features work
   386	together to achieve excellent retrieval performance.
   387	B. Holistic Stream
   388	Thanks to the long-range modeling capability of transformer, the research on transformers in the field of person
   389	
   390	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   391	
   392	5026
   393	
   394	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025
   395	
   396	Re-ID has sparked a boom. Due to the outstanding performance of transformers, we adopt the ViT-B/16 [19] to extract
   397	robust global features of pedestrians. Specifically, let x ∈
   398	R H ×W ×C represent a person image, where H , W and C are
   399	the height, width and channel dimensions respectively. First,
   400	x is cut into N patches using a 2D convolution with a sliding
   401	window (step=11). Then, we obtain a flattened sequence of
   402	patch embeddings (denoted as Pe ∈R N ×D ) by flattening the
   403	patches and performing linear projection operations, where D
   404	is the embedding dimension. Following the TransReID [20],
   405	we prepose a learnable global token E g into Pe and attach
   406	a learnable position embedding E p and a trainable camera
   407	embedding E c . Thus, we obtain a sequence of patch embeddings as our input:
   408	Pin = C(E g , Pe ) + E p + λc E c
   409	
   410	(1)
   411	
   412	where C represents the concatenation operation and λc is
   413	a balance parameter used to control the camera embedding
   414	weight. Following the setting of TransReID [20], we set λc to
   415	3. The global token E g ∈ R1×D will be updated to the class
   416	token (denoted as ‘CLS’ in Fig. 2 and Fig. 3) in subsequent
   417	layers.
   418	Next, the patch embeddings Pin will be processed by several
   419	transformer layers. In this way, we obtain the output feature f
   420	∈ R(N +1)×D , which contains rich global context information
   421	(one global class token f hg ) and discriminative local semantics
   422	(N patch tokens). The class token is treated as a global feature
   423	for the final person matching. To obtain fine-grained local
   424	features for assisting the person retrieval, we split f into a
   425	global feature f hg ∈ R1×D and the image patch features f patch
   426	∈ R N ×D . Whereafter, following the methods [10] and [45],
   427	we divide f patch into M parts, each of which is concatenated
   428	with f hg . To this end, local features can preserve useful global
   429	semantics. Finally, we feed the M local features into the last
   430	transformer layer to further obtain the discriminative local
   431	M
   432	features f hp (denoted as { f ihp }i=1
   433	).
   434	1) Holistic Stream Loss Function: To train our holistic
   435	stream and ensure that its final output can learn discriminative
   436	information, we apply cross-entropy loss as identity loss Lid
   437	h
   438	and triplet loss Ltri
   439	h as auxiliary loss for the global feature f hg
   440	and the local feature f hp . Thus, the loss of the holistic stream
   441	is defined as follows:
   442	M
   443	
   444	Lid
   445	h = − log Phg −
   446	
   447	1 X
   448	i
   449	log Php
   450	,
   451	M
   452	
   453	(2)
   454	
   455	i=1
   456	
   457	h
   458	p
   459	n
   460	Ltri
   461	h = < f hg , f hg > − < f hg , f hg > + m]+
   462	M
   463	
   464	+
   465	
   466	1 Xh
   467	i
   468	ip
   469	i
   470	in
   471	< f hp
   472	, f hp
   473	> − < f hp
   474	, f hp
   475	> + m]+ ,
   476	M
   477	i=1
   478	
   479	(3)
   480	tri
   481	Lh = Lid
   482	h + Lh ,
   483	
   484	(4)
   485	
   486	i represent the prediction probabilities of
   487	where Phg and Php
   488	i , respectively. f p has the same
   489	the classifier for f hg and f hp
   490	hg
   491	n
   492	identity as f hg , while f hg is different from f hg . The same
   493	i . The function < a, b > calculates the Euclidean
   494	goes for f hp
   495	
   496	Fig. 3.
   497	Illustration of the proposed SPG. The black rectangular wire
   498	frame represents the local window, whose size is w. The colorful rectangular
   499	blocks represent the generated cluster centers within the windows. The orange
   500	cube ‘CLS’ represents the global aggregation token of the holistic stream in
   501	corresponding depth.
   502	
   503	distance between a and b. m represents the similarity margin.
   504	[.]+ indicates the max(0, .) function.
   505	C. Part Stream
   506	Although the holistic stream can provide good features for
   507	the general person Re-ID task, countering occlusion is an
   508	inevitable challenge in actual surveillance scenarios. These
   509	occlusions not only introduce noise interference, but also often
   510	cause the model to ignore key semantic information in the
   511	invisible regions. In addition, in occluded scenarios, the discriminative information for identifying occluded pedestrians
   512	often depends on the robust body parts. Therefore, in order to
   513	effectively resist the interference of occlusion clutter, extract
   514	robust local features, and recover the features of the invisible
   515	regions, we propose a part stream to tackle these challenges.
   516	We use the Swin-Transformer [46] as our backbone to balance
   517	the accuracy and efficiency in the part stream encoder.
   518	The part stream is mainly composed of three parts: the
   519	Semantic Patch Generator (SPG), the Adaptive Heterogeneous
   520	Semantic Token Generator (AHSTG), and the Distance-based
   521	Feature Recovery module (DFR), as shown in Fig. 2. The SPG,
   522	AHSTG and DFR are illustrated in Fig. 3, Fig. 4 and Fig. 5,
   523	respectively. Next, we describe the three modules in detail.
   524	1) Semantic Patch Generator: In person Re-ID tasks, the
   525	key to distinguish different pedestrians is to capture discriminative local details. These details, such as the texture, color
   526	of clothing and shape of body parts, are important clues for
   527	recognizing the identity of the target pedestrian. To effectively
   528	extract such subtle but critical information, we specifically
   529	design a SPG module, which aims to mine deeper and
   530	emphasize these local details, and lay the foundation for
   531	subsequent feature aggregation and recovery. The SPG is
   532	shown in Fig. 3, is to improve the ability of the model to
   533	deal with discriminative local details.
   534	Compared with the global self-attention, the advantage of
   535	the local self-attention is that it can efficiently process the
   536	local region of the image. We use it to focus on a specific
   537	area of the person image to mine local semantics. For the
   538	image patches X p (each patch size is p× p) output by the
   539	base module, we perform the spatial pooling operation [47] on
   540	them. This operation can expand the local receptive field and
   541	enable our model to capture richer local context information,
   542	
   543	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   544	
   545	LI et al.: HGTDR NETWORK FOR OCCLUDED PERSON RE-IDENTIFICATION
   546	
   547	Fig. 4. Illustration of the proposed AHSTG. Avgpooling represents the
   548	adaptive global average pooling. ‘CLS’ represents the class token of the
   549	holistic stream. The input to the scoring network consists of three parts:
   550	p
   551	semantic patches P̂ 2 , f cls and CLS. ‘DIS’ and ‘RIS’ denote the Discrimination Indicator Score and Relevance Indicator Score, respectively. They are
   552	weighted and summed to give the comprehensive score s p . The red dots
   553	indicate the locations of the cluster centers selected based on DIS, and then
   554	obtain the semantic patches as cluster centers. The red rectangular boxes
   555	represent the selected cluster centers in heterogeneous semantic tokens. The
   556	number of heterogeneous tokens K in the figure is set to 4 for simplicity.
   557	
   558	Fig. 5. Illustration of the proposed DFR. The index is from the AHSTG
   559	module, which is the same as the index in Fig. 4. f hg represents the global
   560	feature (i.e. CLS token) of the holistic stream. After the Euclidean distance
   561	computing, green bidirectional arrows indicate that the region of this index
   562	interval is visible, while red ones indicate that there is occlusion. Between
   563	the heterogeneous semantic tokens in the presence of occlusion (between the
   564	red arrows), a recovery token is inserted to recover the person features. The
   565	‘Split’ operation removes the global feature.
   566	
   567	which further improves the discriminative ability of features.
   568	Specifically, as shown in Fig. 3, a feature map X p with spatial
   569	resolution H ×W is partitioned uniformly using a window
   570	of size w×w. In this way, we can obtain Nw (H/w×W/w)
   571	local windows, where each window applies local attention to
   572	dynamically generate a cluster center. To integrate the rich
   573	local semantics in each window, we set w> p. In addition,
   574	through the spatial pooling operation, our model can aggregate the semantics of important patches in the window and
   575	improve the separation between cluster centers. Consequently,
   576	′
   577	we obtain Nw cluster centers S ∈R Nw ×D , where D ′ = D/2.
   578	The detailed procedure of spatial pooling can be understood
   579	in ST-ViT [47]. The design allows the model to perform a
   580	weighted sum of image patches within each window.
   581	In person Re-ID tasks, the high correlation between image
   582	patches is the key to extract discriminative features. To take
   583	full advantage of this property, we introduce an innovative
   584	global optimization strategy that aims to enhance the semantic expressiveness of local features from the perspective of
   585	the global receptive field. Intuitively, patches with similar
   586	semantics should be highly correlated with each other, and
   587	each image patch should have significance in extracting
   588	discriminative features. Specifically, we adopt a Multi-head
   589	Self-Attention mechanism, which can dynamically aggregate
   590	image patches with similar semantics to enhance the model’s
   591	
   592	5027
   593	
   594	understanding and expression of local features. Therefore, our
   595	model can capture the complex relationships between image
   596	′
   597	patches and generate initial semantic patches P̂ 1 ∈R Nw ×D .
   598	This process not only enhances the sensitivity of our model
   599	to local details, but also improves the discriminative ability
   600	of the features through the integration of global information.
   601	Formally, the generation process of P̂ 1 can be expressed as
   602	follows:
   603	P̂ 1 = F F N (M S A(S, X p , X p )).
   604	
   605	(5)
   606	
   607	where M S A and F F N are abbreviations for Multi-head SelfAttention and FeedForward Neural network, respectively, the
   608	same below. The three inputs of M S A correspond to queries,
   609	keys, and values. Through the spatial pooling and M S A layer,
   610	the produced initial semantic patches P̂ 1 can capture the rich
   611	information of the person image from two levels: global and
   612	local, respectively.
   613	Since none of the above processing is guided by global
   614	features, these initial semantic patches lack certain global
   615	information. To compensate for the lack of global information in the P̂ 1 and strengthen the association with the
   616	holistic stream, we use several transformer layers to gradually
   617	enable P̂ 1 to learn global semantics. Specifically, as shown
   618	in Fig. 3, we merge the CLS token of the holistic stream
   619	with semantic patches, which enables the semantic patches
   620	to inherit the global context information. Then, we project the
   621	merged semantic patches into queries. Next, we concatenate
   622	the semantic patches with the original image patches X p and
   623	use them as keys and values for information transmission. The
   624	queries, keys and values are fed into the transformer layers to
   625	update the semantic patches. Through this design strategy, the
   626	transformer layers can dynamically adjust the information flow
   627	between the semantic patches and the original image patches,
   628	and realize the deep aggregation of semantics. The update
   629	′
   630	process of semantic patches P̂ 2 ∈R Nw ×D is as follows:
   631	P̂12 = F F N (M S A( P̂ 1 +C L S 1 , C( P̂ 1 , X p ), C( P̂ 1 , X p )),
   632	P̂22 = F F N (M S A( P̂12 +C L S 2 , C( P̂12 , X p ), C( P̂12 , X p )),
   633	··· ,
   634	2
   635	2
   636	2
   637	P̂i2 = F F N (M S A( P̂i−1
   638	+C L S i , C( P̂i−1
   639	,X p ), C( P̂i−1
   640	, X p )),
   641	··· .
   642	(6)
   643	′
   644	
   645	where C L S i ∈R1×D refers to the CLS token in the ith transformer layer at the current network depth, which is
   646	derived from the holistic stream. Note that the reason the
   647	dimension of C L S i is not D is that the dimension reduction
   648	has been done to unify its computation. We omit this operation
   649	in the figure. C L S i guides semantic patches to aggregate
   650	global context semantics.
   651	After N3 layers of processing, we obtain the updated semantic
   652	′
   653	patches P̂ 2 ∈R Nw ×D in the last layer. Throughout the update
   654	iteration process, the global aggregation features of some one
   655	layer are initially used as queries to guide semantic patches
   656	to aggregate global information. Subsequently, the semantic
   657	patches containing specific global information are employed as
   658	keys and values to progressively enhance the clustering effect
   659	
   660	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   661	
   662	5028
   663	
   664	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025
   665	
   666	of the semantic patches. In this way, each transformer layer
   667	provides further global semantics for the semantic patches,
   668	so that P̂ 2 can express the person local features more richly
   669	and precisely. Finally, in order to make P̂ 2 applicable to the
   670	following AHSTG processing, we upsample the dimension of
   671	P̂ 2 to D and obtain P̂ 2 ∈R Nw ×D . Likewise, we omit this
   672	operation in the figure.
   673	2) Adaptive Heterogeneous Semantic Token Generator:
   674	Although semantic patches P̂ 2 can capture the information
   675	of neighboring patches, some patches may include irrelevant
   676	parts of pedestrians such as occlusions or background noises.
   677	When dealing with the occluded person Re-ID task, adaptively
   678	extracting identity-related local features is a core challenge.
   679	Traditional strip-based methods often rely on hand-crafted
   680	rules when extracting local features, which introduces background noises and limits the adaptability of the model to
   681	the features with different scales and shapes. To overcome
   682	this limitation, we propose an AHSTG, as shown in Fig. 4.
   683	It can flexibly extract local features with different scales and
   684	shapes to effectively suppress the interference of background
   685	clutter and highlight the target person’s features. AHSTG
   686	uses the disentangling advantage of transformer based on
   687	semantic information, and considers the dual influence of
   688	holistic stream and part stream. This design not only enhances
   689	the ability of our model to extract local features, but also
   690	significantly improves the processing flexibility of person
   691	features of different scales and sizes. By adaptively adjusting
   692	the attention weight of different semantic patches, AHSTG
   693	effectively suppresses the interference of background noises
   694	and highlights the identity-related regions. Specifically, we first
   695	feed semantic patches P̂ 2 into the scoring network to estimate
   696	their importance. Since there is no global token for the part
   697	stream, we perform global average pooling on P̂ 2 in order
   698	p
   699	to obtain the global feature representation f cls ∈R1×D . The
   700	comprehensive score s consists of two components: the Discrimination Indicator Score (DIS) and the Relevance Indicator
   701	Score (RIS). DIS is introduced to quantify the importance of
   702	patches in each sample. It is expected to assign high values
   703	to those regions with high discrimination, such as critical
   704	parts of pedestrians or possible occlusions (e.g., billboards,
   705	etc.), while assigning low scores to background clutter. The
   706	p
   707	DIS takes P̂ 2 , f cls and the class token as input. It uses the
   708	self-attention to score each patch. To illustrate, let’s consider
   709	the i-th semantic patch. Its discrimination indicator score is
   710	calculated as follows:
   711	p
   712	
   713	di = αdih + (1 − α)di ,
   714	with dih = Atten(C L S, pi ),
   715	p
   716	
   717	p
   718	
   719	di = Atten( f cls , pi ),
   720	p
   721	
   722	(7)
   723	
   724	where dih and di represent the contribution of the holistic stream and part stream to IDS, respectively. α is the
   725	hyper-parameter that weighs them. Atten represents the selfattention, which calculates the importance of pi to the global
   726	feature. ‘CLS’ represents the class token of holistic stream at
   727	the current network depth.
   728	For the relevance indicator score (RIS), its primary purpose
   729	is to capture the affinity of the semantic patch to each
   730	
   731	identity. It is expected that RIS allocates a higher value to
   732	identity-related patches and a lower value vice versa. Similarly,
   733	the RIS of the i-th semantic patch is calculated as follows:
   734	p
   735	
   736	ri = αrih + (1 − α)ri ,
   737	ex p (ϕ (C L S, pi ))
   738	,
   739	with rih = P N
   740	w
   741	pi ))
   742	i=1 ex p (ϕ (C L S,
   743	
   744	p
   745	ex p ϕ( f cls , pi )
   746	p
   747	ri = P N
   748	
   749	p
   750	w
   751	i=1 ex p ϕ( f cls , pi )
   752	
   753	(8)
   754	
   755	p
   756	
   757	where rih and ri represent the contribution of the holistic
   758	stream and the part stream to RIS, respectively. α is the same
   759	as in Eq. (7). Nw is the number of semantic patches. The
   760	cosine similarity is defined by the function ϕ, which indicates
   761	the relevance of the semantic patch with the person identity.
   762	When a certain patch is heavily occluded, a small score is
   763	allocated for it to reduce its impact on the network.
   764	Finally, the discrimination indicator score di and the relevance indicator score ri are summed together, and then
   765	normalized using the softmax function to obtain the comprep
   766	hensive score si . The score indicates the extent to which the
   767	semantic patch contributes to the final pedestrian retrieval task.
   768	It not only combines the contributions of the two streams, but
   769	also considers them comprehensively in terms of discrimination and relevance, respectively. Thus, the scoring mechanism
   770	can improve the robustness of the model against occlusion
   771	noise and enhance the ability to recognize the target person.
   772	Based on the comprehensive scores obtained above, we further filter out the most discriminative semantic patches to
   773	construct a more effective feature representation. According to
   774	the discrimination scores, we select K patches with the highest
   775	response as the cluster centers, as shown in Fig. 4. Next,
   776	we take their indexes as the heterogeneous tokens’ indexes
   777	(while retaining the location information), denoted as I . Then,
   778	we input the K patches as queries Q, and the P̂ 2 as keys K
   779	and values V into the transformer blocks to adaptively generate heterogeneous tokens. In the generation process, we also
   780	consider the comprehensive scores s p to give more importance
   781	to discriminative patches, which contribute significantly to the
   782	generation of heterogeneous tokens. Formally,
   783	QKT
   784	Attention( Q, K , V ) = so f tmax( p
   785	+ s p )V
   786	dqk
   787	
   788	(9)
   789	
   790	where dqk is a scaling factor used for regularization. s p =
   791	p
   792	{si |i = 1, 2, . . . , Nw } represent the final comprehensive
   793	scores of semantic patches. Since the core of the transformer
   794	is self-attention, we only display the attention part and omit
   795	operations such as FFN. Here, the identity-related patches are
   796	allocated higher comprehensive scores, while those containing
   797	background or obstacles are allocated lower ones. Finally,
   798	after AHSTG processing, we obtain fine-grained features
   799	with different shapes and sizes. For a heterogeneous token,
   800	if its aggregated patches all have low comprehensive scores,
   801	we classify it as occlusion or background. In this way, the genK )
   802	erated heterogeneous tokens f t (denoted as { f ti ∈ R1×D }i=1
   803	can adaptively suppress interference from occlusion and facilitate the network in learning discriminative person features.
   804	
   805	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   806	
   807	LI et al.: HGTDR NETWORK FOR OCCLUDED PERSON RE-IDENTIFICATION
   808	
   809	3) Distance-Based Feature Recovery Module: Even though
   810	SPG and AHSTG modules can deal with local detail features
   811	and suppress the interference of occlusions well, the problem
   812	of missing key features due to occlusion is still unsolved.
   813	In other words, the learned heterogeneous features still suffer
   814	from the lack of pedestrian information. To address this problem, we propose a Distance-based Feature Recovery (DFR)
   815	module to recover the features of the occluded body parts,
   816	as shown in Fig. 5.
   817	The Masked Auto-Encoders (MAE) [48] states that the
   818	masked features can be predicted by the surrounding
   819	unmasked features. Inspired by MAE, we consider that the
   820	occluded body regions can also be recovered from the adjacent
   821	visible regions. However, different from MAE recovering
   822	the masked features through unconstrained self-attention,
   823	we introduce a distance factor into the DFR module to achieve
   824	more accurate feature recovery. Our core idea is that when
   825	recovering the occluded region, neighbor features closer to
   826	the occluded region should be assigned higher weights, while
   827	neighbor features farther away from the occluded region
   828	should have relatively small contributions. This is because,
   829	in an image, pixels or feature points that are close together
   830	usually have higher spatial correlation. This means that they
   831	are more likely to contain similar information or to be part
   832	of the same object. Therefore, when a region is occluded, its
   833	nearby feature points are more likely to contain similar or
   834	related information to the occluded region, and thus are more
   835	helpful for recovering the features of the occluded region.
   836	The distance-based contribution consideration makes the
   837	DFR module more accurate and effective in recovering the
   838	occluded regions. To be specific, after obtaining the heterogeneous tokens f t and its indexes I , we project the index I
   839	into a 2D space to calculate the Euclidean distance between
   840	adjacent indexes, as shown in Fig. 5. If the distance is
   841	large, it is inferred that the region between the two indexes
   842	may be obscured. To recover the features of these occluded
   843	regions, we subtly insert a recovery token between the two
   844	heterogeneous tokens. The insertion of the recovery token not
   845	only fills the blank of the missing feature, but also increases
   846	the adaptability and flexibility of our model to the occluded
   847	regions by randomly selecting the index. For example, if the
   848	region between the i-th token f ti (index: Ii ) and the (i + 1)th token f ti (index: Ii+1 ) is occluded, we insert a recovery
   849	token f tr ∈R1×D between them with index I r = Random(Ii ,
   850	Ii+1 ). Furthermore, we calculate the planar Euclidean distance
   851	between I r and each element in index I , and further determine
   852	the contribution weights of heterogeneous tokens. The weights
   853	are denoted by Wt = {Wti |i = 1, 2, . . . , K }. For the ith weight, it can be represented by the following softmax
   854	function:
   855	ex p(−d(Ii , I r ))
   856	Wti = P K
   857	r
   858	i=1 ex p(−d(Ii , I ))
   859	
   860	(10)
   861	
   862	where d(Ii , I r ) represents the planar Euclidean distance
   863	function between Ii and I r . Since the greater the distance,
   864	the smaller the weight should be, we apply the softmax
   865	function to the negative values of the distance. It ensures a
   866	reasonable distribution of the weights. Once we obtain the
   867	
   868	5029
   869	
   870	contribution weight Wt , we utilize it to weight and update the
   871	heterogeneous token f t . Formally,
   872	f ti = (1 + Wti ) f ti
   873	
   874	(11)
   875	
   876	Then, we input the updated tokens f t and the inserted
   877	recovery token f tr into the decoder to complete the missing
   878	features. The decoder, in comparison to the foregoing encoder,
   879	has a less complex network structure with fewer parameters.
   880	Following several transformer layers of the distance-based
   881	recovery decoder, we acquire the ultimate completed features:
   882	f c = Layer s(C( f hg , I nser t ( f t , f tr , I r )))
   883	
   884	(12)
   885	
   886	where Layer s(.) represents the processing of multiple transformer layers. f hg is the global feature of the holistic stream,
   887	which is used to provide global semantics and guide the feature recovery. I nser t (a, b, c) are used to select the insertion
   888	position of b according to its index c, and insert b into a. Its
   889	inputs are the updated tokens f t , the recovery token f tr and its
   890	index I r . Since the part stream aims to extract discriminative
   891	local features, we split the final recovery features and remove
   892	the global feature to obtain the final output of the part stream
   893	K +1
   894	fr p (denoted as { frip ∈ R1×D }i=1
   895	, i.e. K heterogeneous
   896	tokens and one recovery token). Additionally, we also apply
   897	the above operations on fr p to compute their comprehensive
   898	scores s t = {sit |i = 1, 2, . . . , K + 1}.
   899	4) Part Stream Loss Function: To train our part stream and
   900	improve the local feature extraction capability of HGTDR,
   901	we introduce an innovative loss function, weight-based ID loss
   902	−id ), on the final output f . The core idea of LW −id is to
   903	(LW
   904	rp
   905	p
   906	p
   907	adjust the weight of a token in the loss function according to its
   908	comprehensive score. Since these tokens have different scores,
   909	they contribute differently to the network. When the score of
   910	a given token is small, the network should be less influenced
   911	by it, and vice versa. It helps to reduce the interference of
   912	noise and irrelevant information. In addition, we also use the
   913	triplet loss Ltri
   914	p to further optimize the discriminability of
   915	features. Finally, the objective function of the part stream can
   916	be expressed as follows:
   917	K +1
   918	
   919	−id
   920	LW

codex
HGTDR 的核心链条已经比较清楚：先把“条带部位会引入噪声”改写成“需要异质尺度的语义 token”，再把“遮挡导致缺失”改写成“按空间距离恢复缺失 token”。我再补齐它的损失部分，同时抽 HFLAT、HKGR、HPL 的方法段。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '356,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   356	III. P ROPOSED M ETHOD
   357	In this section, we introduce the proposed Heterogeneous
   358	Generative Tokens and Distance-aware Recovery (HGTDR)
   359	framework in detail. In Section III-A, we first give a concise
   360	overview of our network. Next, the structure of the holistic
   361	stream is introduced in detail in Section III-B, and the components of the part stream are highlighted in Section III-C.
   362	Finally, the objective loss function and optimization strategy
   363	are described in Section III-D.
   364	A. Overview
   365	The proposed HGTDR network is mainly composed of a
   366	holistic stream and a part stream, as shown in Fig. 2. Initially,
   367	we use a basic transformer model to extract the low-level
   368	semantics and establish long-range global relationships from
   369	person images. Subsequently, the output features are fed
   370	into the holistic stream and the part stream. In the holistic
   371	stream, we employ the ViT-B/16 [19] to capture contextual
   372	information and obtain stable global features. While in the
   373	part stream, the output features are initially divided into
   374	multiple windows, and spatial pooling is performed on each
   375	window to obtain clustering centers. They aggregate relevant
   376	information from both the original features and the global
   377	features of the holistic stream to generate semantic patches.
   378	Next, the semantic patches are input into the scoring network
   379	to select the most obvious patches and their indexes are
   380	recorded. The selected patches are used to adaptively generate
   381	K discriminative heterogeneous tokens with rich semantics.
   382	These tokens are then fed into the decoder to enhance the
   383	network’s robustness. We assess their distances and insert
   384	a recovery token between them to dynamically recover the
   385	missing features. Thus, we can obtain robust and discriminative local features. Finally, the local and global features work
   386	together to achieve excellent retrieval performance.
   387	B. Holistic Stream
   388	Thanks to the long-range modeling capability of transformer, the research on transformers in the field of person
   389	
   390	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   391	
   392	5026
   393	
   394	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025
   395	
   396	Re-ID has sparked a boom. Due to the outstanding performance of transformers, we adopt the ViT-B/16 [19] to extract
   397	robust global features of pedestrians. Specifically, let x ∈
   398	R H ×W ×C represent a person image, where H , W and C are
   399	the height, width and channel dimensions respectively. First,
   400	x is cut into N patches using a 2D convolution with a sliding
   401	window (step=11). Then, we obtain a flattened sequence of
   402	patch embeddings (denoted as Pe ∈R N ×D ) by flattening the
   403	patches and performing linear projection operations, where D
   404	is the embedding dimension. Following the TransReID [20],
   405	we prepose a learnable global token E g into Pe and attach
   406	a learnable position embedding E p and a trainable camera
   407	embedding E c . Thus, we obtain a sequence of patch embeddings as our input:
   408	Pin = C(E g , Pe ) + E p + λc E c
   409	
   410	(1)
   411	
   412	where C represents the concatenation operation and λc is
   413	a balance parameter used to control the camera embedding
   414	weight. Following the setting of TransReID [20], we set λc to
   415	3. The global token E g ∈ R1×D will be updated to the class
   416	token (denoted as ‘CLS’ in Fig. 2 and Fig. 3) in subsequent
   417	layers.
   418	Next, the patch embeddings Pin will be processed by several
   419	transformer layers. In this way, we obtain the output feature f
   420	∈ R(N +1)×D , which contains rich global context information
   421	(one global class token f hg ) and discriminative local semantics
   422	(N patch tokens). The class token is treated as a global feature
   423	for the final person matching. To obtain fine-grained local
   424	features for assisting the person retrieval, we split f into a
   425	global feature f hg ∈ R1×D and the image patch features f patch
   426	∈ R N ×D . Whereafter, following the methods [10] and [45],
   427	we divide f patch into M parts, each of which is concatenated
   428	with f hg . To this end, local features can preserve useful global
   429	semantics. Finally, we feed the M local features into the last
   430	transformer layer to further obtain the discriminative local
   431	M
   432	features f hp (denoted as { f ihp }i=1
   433	).
   434	1) Holistic Stream Loss Function: To train our holistic
   435	stream and ensure that its final output can learn discriminative
   436	information, we apply cross-entropy loss as identity loss Lid
   437	h
   438	and triplet loss Ltri
   439	h as auxiliary loss for the global feature f hg
   440	and the local feature f hp . Thus, the loss of the holistic stream
   441	is defined as follows:
   442	M
   443	
   444	Lid
   445	h = − log Phg −
   446	
   447	1 X
   448	i
   449	log Php
   450	,
   451	M
   452	
   453	(2)
   454	
   455	i=1
   456	
   457	h
   458	p
   459	n
   460	Ltri
   461	h = < f hg , f hg > − < f hg , f hg > + m]+
   462	M
   463	
   464	+
   465	
   466	1 Xh
   467	i
   468	ip
   469	i
   470	in
   471	< f hp
   472	, f hp
   473	> − < f hp
   474	, f hp
   475	> + m]+ ,
   476	M
   477	i=1
   478	
   479	(3)
   480	tri
   481	Lh = Lid
   482	h + Lh ,
   483	
   484	(4)
   485	
   486	i represent the prediction probabilities of
   487	where Phg and Php
   488	i , respectively. f p has the same
   489	the classifier for f hg and f hp
   490	hg
   491	n
   492	identity as f hg , while f hg is different from f hg . The same
   493	i . The function < a, b > calculates the Euclidean
   494	goes for f hp
   495	
   496	Fig. 3.
   497	Illustration of the proposed SPG. The black rectangular wire
   498	frame represents the local window, whose size is w. The colorful rectangular
   499	blocks represent the generated cluster centers within the windows. The orange
   500	cube ‘CLS’ represents the global aggregation token of the holistic stream in
   501	corresponding depth.
   502	
   503	distance between a and b. m represents the similarity margin.
   504	[.]+ indicates the max(0, .) function.
   505	C. Part Stream
   506	Although the holistic stream can provide good features for
   507	the general person Re-ID task, countering occlusion is an
   508	inevitable challenge in actual surveillance scenarios. These
   509	occlusions not only introduce noise interference, but also often
   510	cause the model to ignore key semantic information in the
   511	invisible regions. In addition, in occluded scenarios, the discriminative information for identifying occluded pedestrians
   512	often depends on the robust body parts. Therefore, in order to
   513	effectively resist the interference of occlusion clutter, extract
   514	robust local features, and recover the features of the invisible
   515	regions, we propose a part stream to tackle these challenges.
   516	We use the Swin-Transformer [46] as our backbone to balance
   517	the accuracy and efficiency in the part stream encoder.
   518	The part stream is mainly composed of three parts: the
   519	Semantic Patch Generator (SPG), the Adaptive Heterogeneous
   520	Semantic Token Generator (AHSTG), and the Distance-based
   521	Feature Recovery module (DFR), as shown in Fig. 2. The SPG,
   522	AHSTG and DFR are illustrated in Fig. 3, Fig. 4 and Fig. 5,
   523	respectively. Next, we describe the three modules in detail.
   524	1) Semantic Patch Generator: In person Re-ID tasks, the
   525	key to distinguish different pedestrians is to capture discriminative local details. These details, such as the texture, color
   526	of clothing and shape of body parts, are important clues for
   527	recognizing the identity of the target pedestrian. To effectively
   528	extract such subtle but critical information, we specifically
   529	design a SPG module, which aims to mine deeper and
   530	emphasize these local details, and lay the foundation for
   531	subsequent feature aggregation and recovery. The SPG is
   532	shown in Fig. 3, is to improve the ability of the model to
   533	deal with discriminative local details.
   534	Compared with the global self-attention, the advantage of
   535	the local self-attention is that it can efficiently process the
   536	local region of the image. We use it to focus on a specific
   537	area of the person image to mine local semantics. For the
   538	image patches X p (each patch size is p× p) output by the
   539	base module, we perform the spatial pooling operation [47] on
   540	them. This operation can expand the local receptive field and
   541	enable our model to capture richer local context information,
   542	
   543	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   544	
   545	LI et al.: HGTDR NETWORK FOR OCCLUDED PERSON RE-IDENTIFICATION
   546	
   547	Fig. 4. Illustration of the proposed AHSTG. Avgpooling represents the
   548	adaptive global average pooling. ‘CLS’ represents the class token of the
   549	holistic stream. The input to the scoring network consists of three parts:
   550	p
   551	semantic patches P̂ 2 , f cls and CLS. ‘DIS’ and ‘RIS’ denote the Discrimination Indicator Score and Relevance Indicator Score, respectively. They are
   552	weighted and summed to give the comprehensive score s p . The red dots
   553	indicate the locations of the cluster centers selected based on DIS, and then
   554	obtain the semantic patches as cluster centers. The red rectangular boxes
   555	represent the selected cluster centers in heterogeneous semantic tokens. The
   556	number of heterogeneous tokens K in the figure is set to 4 for simplicity.
   557	
   558	Fig. 5. Illustration of the proposed DFR. The index is from the AHSTG
   559	module, which is the same as the index in Fig. 4. f hg represents the global
   560	feature (i.e. CLS token) of the holistic stream. After the Euclidean distance

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '899,990p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   899	4) Part Stream Loss Function: To train our part stream and
   900	improve the local feature extraction capability of HGTDR,
   901	we introduce an innovative loss function, weight-based ID loss
   902	−id ), on the final output f . The core idea of LW −id is to
   903	(LW
   904	rp
   905	p
   906	p
   907	adjust the weight of a token in the loss function according to its
   908	comprehensive score. Since these tokens have different scores,
   909	they contribute differently to the network. When the score of
   910	a given token is small, the network should be less influenced
   911	by it, and vice versa. It helps to reduce the interference of
   912	noise and irrelevant information. In addition, we also use the
   913	triplet loss Ltri
   914	p to further optimize the discriminability of
   915	features. Finally, the objective function of the part stream can
   916	be expressed as follows:
   917	K +1
   918	
   919	−id
   920	LW
   921	=−
   922	p
   923	
   924	Ltri
   925	p =
   926	
   927	1 X t
   928	si log Pri p ,
   929	K +1
   930	
   931	1
   932	K +1
   933	
   934	i=1
   935	K
   936	+1 h
   937	X
   938	
   939	p
   940	
   941	(13)
   942	n
   943	
   944	< frip , frip > − < frip , frip > + m]+ ,
   945	
   946	i=1
   947	
   948	(14)
   949	−id
   950	L p = LW
   951	+ Ltri
   952	p
   953	p ,
   954	
   955	(15)
   956	
   957	where Pri p is the predicted probability of the classifier for
   958	frip , and sit is the comprehensive score of frip . In this way, the
   959	output features combine the discrimination and relevance indicators, enabling the network to adaptively adjust to complex
   960	occluded scenes.
   961	D. Objective Function and Optimization
   962	In the training stage, we use cross-entropy loss and triplet
   963	loss for the global and local features of both the holistic stream
   964	
   965	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:56:57 UTC from IEEE Xplore. Restrictions apply.
   966	
   967	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '49,552p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    49	95.9% and 90.6%, respectively, and the mAP of 90.8% and 82.2%, respectively.
    50	
    51	1. Introduction
    52	Person re-identiﬁcation (ReID) tasks intend to identify and match
    53	target individuals among multiple non-overlapping camera viewpoints.
    54	Owing to the rapid growth of deep learning techniques, research about
    55	person ReID (Luo et al., 2019; Qian et al., 2017; Zhou et al., 2019) has
    56	made signiﬁcant progress. However, in practical situations, as illustrated
    57	in Fig. 1, pedestrians often exhibit missing body parts due to occlusions
    58	caused by objects such as trees and vehicles. Therefore, how to enhance
    59	the performance of person ReID models when pedestrians are partially
    60	occluded has become one of the core challenges.
    61	In recent times, occluded person ReID has garnered heightened interest, and its key is extracting features from partially occluded images
    62	to mitigate the impact of occlusions. To this end, various occluded per-
    63	
    64	son ReID methods have been presented. Existing methods can be broadly
    65	categorized into external auxiliary information-based methods Gao et al.
    66	(2020), Huang et al. (2020), Miao et al. (2019) and Transformer-based
    67	methods Li et al. (2021), Lin et al. (2024), Wang et al. (2022b). The external auxiliary information-based methods typically reduce the eﬀect of
    68	occlusions on the discriminative capability of pedestrian features by locating un-occluded body regions with the help of auxiliary information,
    69	such as human posture or body key points. The Transformer-based methods primarily leverage the global modelling strengths of Transformers,
    70	integrating various attention mechanisms to extract more discriminative features. Although these methods have achieved some encouraging
    71	results, most of them process diﬀerent image patches indiscriminately.
    72	Not only they fail to quantify the diﬀerent contributions of diﬀerent image patches to identity recognition, but they also neglect to focus on
    73	
    74	∗ Corresponding author.
    75	
    76	E-mail address: jddhs@vip.henu.edu.cn (H. Du).
    77	https://doi.org/10.1016/j.neunet.2026.108883
    78	Received 8 July 2025; Received in revised form 9 March 2026; Accepted 19 March 2026
    79	Available online 5 April 2026
    80	0893-6080/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    81	
    82	Neural Networks 201 (2026) 108883
    83	
    84	H. Du et al.
    85	
    86	Fig. 1. Examples of pedestrians being occluded.
    87	
    88	•
    89	
    90	key regions. Moreover, most of them fail to eﬀectively isolate invalid
    91	or detrimental information contained in pedestrian images, thereby being unable to mitigate the adverse eﬀects of backgrounds and occlusions on the performance of their models. In particular, although most
    92	of them possess a strong ability to capture global contextual information, their capacity for acquiring ﬁne-grained local features is largely
    93	inadequate, preventing them from fully extracting the identity information contained in diﬀerent body parts of pedestrians.
    94	To resolve the mentioned problems, we propose a hierarchical fusion and local-aware Transformer (HFLAT) for occluded person ReID.
    95	First, we design a feature hierarchical fusion module (FHFM) to hierarchize and fuse the patch feature vector sequence based on their relative
    96	importance to the global feature vector, enhancing the identity discriminative features of key regions. Subsequently, we design a feature separation module (FSM) to isolate foreground and background features using patch-level saliency analysis, thereby mitigating the negative eﬀects
    97	of backgrounds and occlusions on the model’s performance. Additionally, we design a local feature extraction module (LFEM) that employs
    98	a local-aware multi-attention mechanism to limit feature interactions,
    99	improving the model’s ﬁne-grained local feature extraction capability.
   100	The main contributions of this research work presented in this paper are
   101	as follows:
   102	
   103	We design a local feature extraction module (LFEM) to enhance the
   104	ﬁne-grained local feature extraction capability of the model by limiting the range of interactions between the features using a local-aware
   105	multi-head attention mechanism.
   106	• We perform extensive experiments on four datasets to validate the
   107	eﬃciency and advancement of the model.
   108	2. Related works
   109	As deep learning techniques progress rapidly, person ReID has
   110	achieved remarkable results Luo et al. (2019), Qian et al. (2017), Zhou
   111	et al. (2019). However, in practical scenarios, pedestrians in images are
   112	often partially occluded by objects such as trees or vehicles, leading to
   113	the loss of key body parts. To address this issue, scholars have presented
   114	a series of methods for occluded person ReID, which can be grouped into
   115	external auxiliary information-based methods Gao et al. (2020), Huang
   116	et al. (2020), Miao et al. (2019) and Transformer-based methods Li et al.
   117	(2021), Lin et al. (2024), Wang et al. (2022b).
   118	The external auxiliary information-based methods primarily mitigate
   119	the impact of occlusions on their models by utilizing auxiliary information such as human posture or key points to locate visible body parts.
   120	For instance, Miao et al. (2019) proposed a pose-guided feature alignment (PGFA) method to align local features using human key-point information. Gao et al. (2020) proposed a pose-guided visible part matching (PVPM) method that employs a pose-guided attention mechanism
   121	and a local visibility predictor to extract features from un-occluded
   122	areas accurately. Wang et al. (2022a) proposed a key-point-aware occlusion suppression and semantic alignment (POS) method that aligns
   123	features using human key points and employs a feature enhancement
   124	strategy to compensate for missing feature regions. Cui et al. (2025)
   125	proposed a pose-guided partial-attention network with batch information (PPBI), which eﬀectively reduces the negative impact of occlusions
   126	on model performance by capturing the semantic relationships of key
   127	
   128	•
   129	
   130	We propose a hierarchical fusion and local-aware Transformer
   131	(HFLAT) for occluded person ReID.
   132	• We design a feature hierarchical fusion module (FHFM) that hierarchizes and fuses the patch feature vectors based on their relative
   133	importance to the global feature vector, aiming to enhance the identity discriminative features of key regions.
   134	• We design a feature separation module (FSM) that separates foreground and background features using patch-level saliency analysis
   135	to mitigate the negative eﬀects of backgrounds and occlusions on the
   136	model’s performance.
   137	2
   138	
   139	Neural Networks 201 (2026) 108883
   140	
   141	H. Du et al.
   142	
   143	points among diﬀerent samples within each batch. Huang et al. (2020)
   144	proposed a human parsing based alignment with multi-task learning
   145	(HPNet) model that enhances the features in the visible regions using body part masks generated by a human parsing task. Somers et al.
   146	(2023) proposed a body part-based (BPreID) model that extracts local
   147	features using a local attention mechanism guided by human parsed labels. Dou et al. (2024) proposed a decouple re-identiﬁcation and human
   148	parsing (DROP) method, which decouples the features for person ReID
   149	and human parsing tasks, mitigating the feature granularity conﬂict between the two tasks and improving their collaborative eﬀectiveness. Although the above methods based on external auxiliary information have
   150	achieved some promising results, their performance may be severely degraded when human body poses or key points are inaccurately estimated
   151	because they rely too much on the accuracy of the external auxiliary information.
   152	The Transformer-based methods mainly utilize the advantages of
   153	Transformer in global modelling, combined with the attention mechanisms to obtain more discriminative features. For instance, Li et al.
   154	(2021) proposed a part-aware Transformer (PAT) by ﬁrst applying the
   155	Transformer framework to occluded person ReID tasks, which improves
   156	the representation ability of local features by employing a pixel context encoder and a local prototype decoder. Lin et al. (2024) proposed a multi-level relation-aware Transformer (MLRAT), which has
   157	stronger feature extraction ability by mining the feature relationships at
   158	patch and sample levels. Wang et al. (2022b) proposed a Transformerbased pose-guided feature disentangling (PFD) method, which eﬀectively mitigates the negative impact of occlusions by disentangling
   159	features utilizing pose information and aligning them to un-occluded
   160	regions. Yang et al. (2023) proposed a robust feature mining Transformer (RFMT) method, which combines residual Transformer layers
   161	with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
   162	enhancement strategy and a feature-complementary decoder to reconstruct occluded features based on neighboring un-occluded regions. Bian
   163	et al. (2024) proposed a novel occlusion-aware feature recover (OAFR)
   164	model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
   165	critical pedestrian features using an uncertainty-aware self-attention
   166	mechanism.
   167	Although the above Transformer-based methods can address the
   168	problems faced by occluded person ReID to some degree, most of them
   169	fail to adequately account for the diﬀerent contributions of diﬀerent
   170	image patches to identity recognition, nor do they prioritize the identity information contained in key regions. For instance, the methods
   171	such as PAT (Li et al., 2021), PFD (Wang et al., 2022b), and RFMT
   172	(Yang et al., 2023) fail to adequately account for the diﬀerent contributions of diﬀerent image patches to identity recognition, consequently lacking focus on key regions. Moreover, most of the existing
   173	models exhibit insuﬃcient capability in extracting ﬁne-grained local
   174	features, thereby failing to fully exploit the identity information embedded in diﬀerent body parts of pedestrians. For instance, the methods such as MLRAT (Lin et al., 2024) and RFMT (Yang et al., 2023)
   175	do not perform ﬁne-grained local feature extraction, consequently failing to adequately mine the local information of pedestrians.they neither adequately consider the diﬀerence in the contribution of diﬀerent image patches to identity recognition, nor suﬃciently emphasize
   176	the identity information of key regions. Additionally, most of their
   177	ﬁne-grained local feature extraction capabilities are insuﬃcient to fully
   178	explore the identity information embedded in diﬀerent body parts of
   179	pedestrians.
   180	Our work is also related to some works. For example, Eliwa et al.
   181	(2024) proposed a framework that integrates Microsoft Azure cloud services with a permissioned blockchain network. After preprocessing and
   182	anonymizing the CT images uploaded by patients via mobile terminals,
   183	
   184	the framework stores the images in Azure Blob Storage, and realizes access control exclusive to authorized specialists through blockchain smart
   185	contracts. Abd El-Hafeez et al. (2025) proposed a novel multi-scale attention model for the classiﬁcation of breast cancer histopathological
   186	images, which achieves high-precision recognition by capturing discriminative features across multiple morphological scales in histopathological images. Eliwa and Abd El-Hafeez (2025c) proposed a robust deep
   187	learning framework improved upon YOLOv11 for the multi-class classiﬁcation task of cervical cancer cells, which enhances the model’s classiﬁcation accuracy via an Attention-Guided Multi-Scale Feature Fusion
   188	(AGMS-FF) module. Eliwa and Abd El-Hafeez (2025a) conducted a rigorous comparative evaluation of ﬁve ﬁne-tuned deep learning architectures for rice maturity classiﬁcation tasks, namely YOLOv11 enhanced
   189	with an Attention-Guided Multi-Scale Feature Fusion (AGMS-FF) module, baseline YOLOv11, ResNet18, EﬃcientNet-B0, and MobileNetV3.
   190	Their results verify the practical value of deep learning-based computer
   191	vision systems in sustainable rice cultivation. Hassan et al. (2025b) proposed a novel DenseNet model integrated with attention mechanisms
   192	and optimized by the Nadam algorithm, which enhances the focus on
   193	pertinent features and thereby improves the model’s classiﬁcation accuracy under complex conditions. Eliwa and Abd El-Hafeez (2025b)
   194	proposed an improved YOLOv11 architecture for the automated classiﬁcation of peripheral blood cells, which integrates a Dynamic CrossScale Context Aggregation (DCSCA) module. Through parallel convolution, dynamic attention, and cross-scale interaction, the module enables
   195	multi-scale feature capture, scale recalibration, and context modeling,
   196	thereby signiﬁcantly improving the model’s classiﬁcation accuracy for
   197	blood cells. Hassan et al. (2025d) proposed a novel network architecture named Sign Neuro-Enhanced Dense Attention Network (SNDA),
   198	which adopts the Nadam optimizer to achieve fast convergence of the
   199	training process. Hassan et al. (2025a) presented a comprehensive review of research advances in the super-resolution ﬁeld from 2015 to
   200	2025. They pointed out that the application of attention modules and
   201	domain-speciﬁc architectures has eﬀectively improved super-resolution
   202	performance, while also indicating that current research still suﬀers
   203	from limitations such as insuﬃcient real-world scenario validation and
   204	the lack of deployment considerations. Hassan et al. (2025c) proposed
   205	a network architecture based on DenseNet121 and the Squeeze-andExcitation (SE) attention module for fruit image classiﬁcation, which
   206	improves the model’s generalization ability through the application of
   207	attention mechanisms, Nadam optimization, and other data augmentation methods. Eliwa and Abd El-Hafeez (2025) proposed a plant disease
   208	detection and classiﬁcation method based on a modiﬁed YOLOv11 architecture, which improves the model’s performance in plant disease
   209	detection and classiﬁcation through strategies such as a custom classiﬁcation head and dynamic validation set splitting. El-Sayed and El-Hafeez
   210	(2012) proposed an improved edge detection method for grayscale images. Based on the Baljit and Amar algorithm, this method replaces
   211	image derivative calculation with Shannon entropy, thereby reducing
   212	CPU computation time while ensuring the robustness of edge detection
   213	quality. Saabia et al. (2019) proposed a face recognition system, which
   214	achieves eﬃcient face recognition through ﬁve sequential steps. These
   215	steps include face detection combined with median ﬁlter preprocessing,
   216	feature extraction using Gabor ﬁlters, dimensionality reduction via Principal Component Analysis (PCA), feature selection based on the Grey
   217	Wolf Optimization (GWO) algorithm, and classiﬁcation using a k-NN
   218	classiﬁer.
   219	
   220	3. Proposed method
   221	This section begins with an overview of the HFLAT framework,
   222	then details the feature hierarchical fusion module (FHFM), the feature separation module (FSM), and the local feature extraction module (LFEM). Finally, the loss functions employed for model training are
   223	described.
   224	3
   225	
   226	Neural Networks 201 (2026) 108883
   227	
   228	H. Du et al.
   229	
   230	Fig. 2. Work ﬂowchart of HFLAT. FHFM denotes the feature hierarchical fusion module, FSM denotes the feature separation module, and LFEM denotes the local
   231	feature extraction module.
   232	
   233	3.1. Overall
   234	
   235	3.2. Feature hierarchical fusion module
   236	
   237	Our model utilizes the ViT as its backbone network. Speciﬁcally,
   238	given an image 𝑋 ∈ ℝ𝐶×𝐻×𝑊 , where 𝐶, 𝐻, and 𝑊 denote the number of channels, height, and width of the image, respectively. The work
   239	ﬂowchart of HFLAT is shown in Fig. 2. First, we use a 𝑃 × 𝑃 sliding
   240	window with stride 𝑆 to slice 𝑋 via overlapping sampling, obtaining 𝑁
   241	ﬁxed-size patches {𝑥𝑖 ∈ ℝ𝐶×𝑃 ×𝑃 |𝑖 = 1, 2, ⋯ , 𝑁}, where 𝑁 = 𝑁𝐻 × 𝑁𝑊 ,
   242	𝑁𝐻 = ⌊(𝐻 + 𝑆 − 𝑃 )∕𝑆⌋, 𝑁𝑊 = ⌊(𝑊 + 𝑆 − 𝑃 )∕𝑆⌋ and ⌊⋅⌋ denotes the
   243	ﬂoor operation. We then ﬂatten each patch into a vector and use a linear projection to map that vector into the 𝐷-dimensional feature space,
   244	yielding a sequence 𝑍 ∈ ℝ𝑁×𝐷 containing 𝑁 embeddings. Furthermore,
   245	we introduce a learnable class token 𝑧𝑐𝑙𝑠 ∈ ℝ1×𝐷 into 𝑍 to capture the
   246	global information in the pedestrian image. Meanwhile, we add the
   247	learnable positional embeddings 𝐸𝑝𝑜𝑠 ∈ ℝ(𝑁+1)×𝐷 and camera embeddings 𝐸𝑐𝑎𝑚 ∈ ℝ(𝑁+1)×𝐷 to 𝑍 in order to preserve the image patches’ spatial position information and camera information. Finally, we obtain the
   248	input sequence 𝑍0 = [𝑧𝑐𝑙𝑠 ; 𝑧1 ; 𝑧2 ; ⋯ ; 𝑧𝑁 ] + 𝐸𝑝𝑜𝑠 + 𝐸𝑐𝑎𝑚 .
   249	We feed 𝑍0 into 𝐿 Transformer layers to get an output feature sequence 𝑍𝐿 = [𝑧𝑐𝑙𝑠 ; 𝑧1 ; 𝑧2 ; … ; 𝑧𝑁 ] ∈ ℝ(𝑁+1)×𝐷 and an attention weight
   250	matrix 𝑀𝐿 ∈ ℝ(1+𝑁)×(1+𝑁) . For 𝑍𝐿 , we ﬁrst split it into a global feature vector 𝑧𝑐𝑙𝑠 ∈ ℝ𝐷 and a patch feature vector sequence 𝑍𝑝𝑎𝑡𝑐ℎ =
   251	[𝑧1 ; 𝑧2 ; ⋯ ; 𝑧𝑁 ] ∈ ℝ𝑁×𝐷 . Then, we feed 𝑍𝑝𝑎𝑡𝑐ℎ and 𝑀𝐿 into the feature
   252	hierarchical fusion module. In this module, we hierarchize and fuse
   253	the patch feature vectors in 𝑍𝑝𝑎𝑡𝑐ℎ based on the relative importance of
   254	each patch to the global feature vector, thus obtaining the fused feature vector 𝑓𝑢 . Meanwhile, we feed 𝑍𝑝𝑎𝑡𝑐ℎ into the feature separation
   255	module to separate the foreground features and background features
   256	and to obtain a more discriminative foreground feature vector 𝑓𝑓 𝑔 by
   257	calculating the probability that each patch belongs to the foreground
   258	employing patch-level saliency analysis. Additionally, we feed 𝑍𝑝𝑎𝑡𝑐ℎ
   259	into the local feature extraction module, in which we insert four learnable local tokens into 𝑍𝑝𝑎𝑡𝑐ℎ and use the local-aware multi-head attention to limit the range of feature interactions, extracting four local feature vectors {𝑓𝑖𝐿 }4𝑖=1 . Lastly, inspired by Sun et al. (2018), Miao et al.
   260	(2019), and Xu et al. (2022), we concatenate the global feature vector 𝑧𝑐𝑙𝑠 , the fused feature vector 𝑓𝑢 , the foreground feature vector 𝑓𝑓 𝑔 ,
   261	and the local feature vectors {𝑓𝑖𝐿 }4𝑖=1 . The concatenated feature vector is then processed through a fully connected layer for information
   262	interaction and dimensionality reduction, yielding the ﬁnal pedestrian
   263	representation 𝑓 .
   264	
   265	Most of current Transformer-based occluded person ReID methods
   266	fail to adequately account for the diﬀerent contributions of diﬀerent
   267	image patches to identity recognition, nor do they prioritize the identiﬁcation information contained in key regions. To resolve these issues,
   268	we design a feature hierarchical fusion module, as illustrated in Fig. 3,
   269	which divides the patch feature vector sequence into four diﬀerent levels
   270	of sub-sequences according to the relative importance of each patch to
   271	the global feature vector, and fuses these sub-sequences to enhance the
   272	identity discriminative features of key regions. That is to say, FHFM directs greater attention towards key regions, thereby enhancing the identity information they contain by dynamically stratifying image patch
   273	feature vectors based on their relative importance to the global feature
   274	vector.
   275	Speciﬁcally, given a patch feature vector sequence 𝑍𝑝𝑎𝑡𝑐ℎ =
   276	[𝑧1 ; 𝑧2 ; ⋯ ; 𝑧𝑁 ] and an attention weight matrix 𝑀𝐿 ∈ ℝ(1+𝑁)×(1+𝑁) , we
   277	ﬁrst deﬁne a relative importance vector 𝐼 ∈ ℝ𝑁 to measure the degree
   278	of signiﬁcance of each patch in relation to the global feature vector 𝑧𝑐𝑙𝑠 :
   279	𝑀cls,𝑖 × ‖𝑧𝑖 ‖2
   280	𝐼𝑖 = ∑𝑁
   281	𝑗=1 𝑀cls,𝑗 × ‖𝑧𝑗 ‖2
   282	
   283	(𝑖 = 1, 2, ⋯ , 𝑁),
   284	
   285	(1)
   286	
   287	where 𝑀𝑐𝑙𝑠,𝑖 denotes the attention weight of the 𝑖-th patch relative
   288	to the class token in 𝑀𝐿 . Then, we sort the 𝑁 patch feature vectors
   289	𝑧1 , 𝑧2 , ⋯ , 𝑧𝑁 according to 𝐼 in descending order of relative importance,
   290	and divide the sorted sequence into the ﬁrst-level feature vector sub𝑁
   291	
   292	𝑁
   293	
   294	sequence 𝐹𝑡𝑜𝑝 ∈ ℝ 4 ×𝐷 , the second-level sub-sequence 𝐹𝑚𝑒𝑑 ∈ ℝ 4 ×𝐷 ,
   295	the third-level sub-sequence 𝐹𝑚𝑖𝑑
   296	
   297	𝑁
   298	∈ ℝ 4 ×𝐷 ,
   299	
   300	and the fourth-level sub-
   301	
   302	𝑁
   303	
   304	sequence 𝐹𝑙𝑜𝑤 ∈ ℝ 4 ×𝐷 in equal portions.
   305	Furthermore, we perform hierarchical fusion of these four levels of
   306	feature vector sub-sequences in the following manner to obtain the fused
   307	feature sequence 𝐹𝑓 𝑢𝑠 ∈ ℝ𝑁×𝐷 :
   308	⎧ 𝐹1 = 𝐹𝑡𝑜𝑝
   309	⎪
   310	⎪ 𝐹2 = 𝐹1 + 𝐹𝑚𝑒𝑑
   311	⎪
   312	.
   313	⎨ 𝐹3 = 𝐹2 + 𝐹𝑚𝑖𝑑
   314	⎪
   315	𝐹
   316	=
   317	𝐹
   318	+
   319	𝐹
   320	3
   321	𝑙𝑜𝑤
   322	⎪ 4
   323	⎪𝐹
   324	⎩ 𝑓 𝑢𝑠 = [𝐹1 ; 𝐹2 ; 𝐹3 ; 𝐹4 ]
   325	
   326	4
   327	
   328	(2)
   329	
   330	Neural Networks 201 (2026) 108883
   331	
   332	H. Du et al.
   333	
   334	Fig. 3. Work ﬂowchart of feature hierarchical fusion module.
   335	
   336	Fig. 4. Work ﬂowchart of feature separation module.
   337	
   338	Speciﬁcally, given a patch feature vector sequence 𝑍𝑝𝑎𝑡𝑐ℎ =
   339	[𝑧1 ; 𝑧2 ; ⋯ ; 𝑧𝑁 ] ∈ ℝ𝑁×𝐷 , we ﬁrst process it using a multi-layer perceptron
   340	(MLP) that includes three fully connected layers, three layer normalization (LN) operations, and two GELU activation functions to obtain the
   341	feature vectors 𝑍̃ ∈ ℝ𝑁×𝐷 :
   342	
   343	Finally, we apply average pooling to 𝐹𝑓 𝑢𝑠 along the token dimension to
   344	obtain the fusion feature vector 𝑓𝑢 ∈ ℝ𝐷 .
   345	
   346	3.3. Feature separation module
   347	
   348	(
   349	( (
   350	(
   351	)
   352	))
   353	)
   354	𝑍̃ = LN GELU LN GELU LN(𝑍patch 𝑊1 + 𝑏1 ) 𝑊2 + 𝑏2 𝑊3 + 𝑏3 ,
   355	
   356	In the occluded person ReID tasks, foreground features are crucial for
   357	recognizing individuals. To eﬀectively extract the foreground features
   358	of images, as illustrated in Fig. 4, we design a feature separation module. This module employs a patch-level saliency analysis to calculate the
   359	probability that each patch belongs to the foreground, thereby achieving
   360	eﬀective separation of foreground and background features. The core
   361	objective of this module is to enhance the discriminative ability of pedestrian representations by extracting and enhancing foreground features
   362	while suppressing interfering features caused by backgrounds and occlusions. Compared to the methods such as PGFA (Miao et al., 2019),
   363	PVPM (Gao et al., 2020), and POS (Wang et al., 2022a) that rely on
   364	external auxiliary information for extracting features from un-occluded
   365	regions, this module operates independently of external auxiliary information. This eliminates the negative impact of inaccuracies in human
   366	pose or key-point estimation on the model’s performance. Unlike the
   367	methods such as FCFormer (Wang et al., 2024) and OAFR (Bian et al.,
   368	2024), which reconstruct occluded features based on those from unoccluded regions, this module employs a patch-level saliency analysis
   369	to separate and enhance foreground features. This approach eﬀectively
   370	mitigates potential errors introduced during feature reconstruction that
   371	could compromise the model’s performance.
   372	
   373	𝐷
   374	
   375	𝐷
   376	
   377	𝐷
   378	
   379	(3)
   380	
   381	𝐷
   382	
   383	where 𝑊1 ∈ ℝ𝐷× 2 , 𝑊2 ∈ ℝ 2 × 2 , and 𝑊3 ∈ ℝ 2 ×𝐷 denote the parameters associated with the fully connected layers 𝐹 𝐶1, 𝐹 𝐶2, and 𝐹 𝐶3,
   384	respectively, 𝑏1 , 𝑏2 , and 𝑏3 denote the corresponding bias vectors.
   385	Then, we reshape 𝑍̃ into a feature tensor 𝐹 ∈ ℝ𝑁𝐻 ×𝑁𝑊 ×𝐷 . Here,
   386	𝑁𝐻 × 𝑁𝑊 = 𝑁, and the deﬁnitions of 𝑁𝐻 and 𝑁𝑊 are provided in Subsection 3.1. Further, we process 𝐹 with a 3 × 3 convolutional layer, a
   387	batch normalization (BN) layer, and apply a sigmoid function to obtain
   388	the saliency map 𝑎 ∈ ℝ𝑁𝐻 ×𝑁𝑊 , where 𝑎𝑖,𝑗 denotes the probability that
   389	the patch at position (𝑖, 𝑗) is part of the foreground. Furthermore, we utilize 𝑎 to separate 𝐹 into the foreground feature tensor 𝐹𝑓 𝑔 ∈ ℝ𝑁𝐻 ×𝑁𝑊 ×𝐷
   390	and the background feature tensor 𝐹𝑏𝑔 ∈ ℝ𝑁𝐻 ×𝑁𝑊 ×𝐷 . The process mentioned above can be formalized as:
   391	⎧ 𝑎 = 𝜎(BN(Conv(𝐹𝑟𝑒𝑠 )))
   392	⎪
   393	,
   394	⎨𝐹𝑓 𝑔 = 𝐹𝑟𝑒𝑠 × 𝑎
   395	⎪
   396	⎩ 𝐹𝑏𝑔 = 𝐹𝑟𝑒𝑠 × (1 − 𝑎)
   397	
   398	(4)
   399	
   400	where 𝜎(⋅) denotes the sigmoid activation function, Conv(⋅) denotes a
   401	3 × 3 convolutional layer.
   402	5
   403	
   404	Neural Networks 201 (2026) 108883
   405	
   406	H. Du et al.
   407	
   408	Fig. 5. Work ﬂowcharts of (a) Local feature extraction module, (b) Local-aware multi-head attention and (c) Local-aware attention.
   409	
   410	For each attention head, we ﬁrst linearly transform 𝑍 ′ into the query
   411	𝑄, key 𝐾, and value 𝑉 , respectively:
   412	
   413	Finally, we apply the global average pooling to 𝐹𝑓 𝑔 and 𝐹𝑏𝑔 to obtain the foreground feature vector 𝑓𝑓 𝑔 ∈ ℝ𝐷 and the background feature
   414	vector 𝑓𝑏𝑔 ∈ ℝ𝐷 , respectively.
   415	
   416	′
   417	
   418	⎧ 𝑄 = 𝑍 𝑊𝑞 = [𝑞1 ; 𝑞2 ; ⋯ ; 𝑞𝑁 ; 𝑞𝑝1 ; 𝑞𝑝2 ; 𝑞𝑝3 ; 𝑞𝑝4 ]
   419	⎪
   420	′
   421	⎨𝐾 = 𝑍 𝑊𝑘 = [𝑘1 ; 𝑘2 ; ⋯ ; 𝑘𝑁 ; 𝑘𝑝1 ; 𝑘𝑝2 ; 𝑘𝑝3 ; 𝑘𝑝4 ] ,
   422	⎪
   423	⎩ 𝑉 = 𝑍 ′ 𝑊𝑣 = [𝑣1 ; 𝑣2 ; ⋯ ; 𝑣𝑁 ; 𝑣𝑝1 ; 𝑣𝑝2 ; 𝑣𝑝3 ; 𝑣𝑝4 ]
   424	
   425	3.4. Local feature extraction module
   426	Diﬀerent regions of a pedestrian image generally carry diﬀerent information. For example, the upper part often contains head information, the mid-upper part often contains upper-garment information, the
   427	mid-lower part often contains pants information, and the lower part
   428	often contains shoe information. To obtain ﬁne-grained local features
   429	in diﬀerent parts of a pedestrian image, we design a local feature extraction module, as shown in Fig. 5(a). In particular, we also design
   430	and use a local-aware multi-head attention in this module, as shown in
   431	Fig. 5(b), where each attention head is a local-aware attention, as shown
   432	in Fig. 5(c). Diﬀerent from the conventional multi-head self-attention,
   433	our local-aware multi-head attention achieves an eﬀective interaction of
   434	features from a same local region by strictly restricting the interaction
   435	scope of feature vectors, which ensures the module’s ability to capture
   436	ﬁne-grained local features.
   437	Speciﬁcally, given a patch feature sequence 𝑍𝑝𝑎𝑡𝑐ℎ =
   438	[𝑧1 ; 𝑧2 ; ⋯ ; 𝑧𝑁 ] ∈ ℝ𝑁×𝐷 , we ﬁrst append four learnable local tokens to its end to capture local features of the top-to-bottom four
   439	height-equally divided parts of a pedestrian image. Then, we initialize
   440	the four learnable local tokens with a zero vector. This initialization
   441	approach enables our model to learn discriminative local features
   442	directly from the data without relying on external auxiliary information
   443	(such as predeﬁned human poses or key-points). Speciﬁcally, the
   444	feature vector sequence after adding the four local tokens can be
   445	represented as 𝑍 ′ = [𝑧1 ; 𝑧2 ; ⋯ ; 𝑧𝑁 ; 𝑧𝑝1 ; 𝑧𝑝2 ; 𝑧𝑝3 ; 𝑧𝑝4 ] ∈ ℝ(𝑁+4)×𝐷 . Finally,
   446	we process 𝑍 ′ using a local-aware multi-head attention mechanism.
   447	
   448	(5)
   449	
   450	where 𝑊𝑞 ∈ ℝ𝐷×𝐷 , 𝑊𝑘 ∈ ℝ𝐷×𝐷 , and 𝑊𝑣 ∈ ℝ𝐷×𝐷 denote the linear
   451	transformation matrices. Then, we deﬁne a masking matrix Mask ∈
   452	ℝ(𝑁+4)×(𝑁+4) to ensure that only the patch feature vectors and a local
   453	token belonging to the same region can interact with each other for
   454	information. In particular, the elements in Mask are assigned values according to the following rules:
   455	a) When 1 ≤ 𝑖 ≤ 𝑁 and 1 ≤ 𝑗 ≤ 𝑁, if patches 𝑖 and 𝑗 belong to a same
   456	region, Mask(𝑖, 𝑗) is set to 1; otherwise 0.
   457	b) When 1 ≤ 𝑖 ≤ 𝑁 and 𝑁 + 1 ≤ 𝑗 ≤ 𝑁 + 4, if patch 𝑖 and local token
   458	𝑗 belong to a same region, Mask(𝑖, 𝑗) is set to 1; otherwise 0.
   459	c) When 𝑁 + 1 ≤ 𝑖 ≤ 𝑁 + 4 and 1 ≤ 𝑗 ≤ 𝑁, if local token 𝑖 and patch
   460	𝑗 belong to a same region, Mask(𝑖, 𝑗) is set to 1; otherwise 0.
   461	d) When 𝑁 + 1 ≤ 𝑖 ≤ 𝑁 + 4 and 𝑁 + 1 ≤ 𝑗 ≤ 𝑁 + 4, if 𝑖 = 𝑗, Mask(𝑖, 𝑗)
   462	is set to 1; otherwise 0.
   463	Obviously, the masking matrix Mask we deﬁne strictly restricts the
   464	interaction scope of feature vectors. That is, information exchange is
   465	permitted only between image patch feature vectors within a same local region, between image patch feature vectors and their corresponding
   466	local tokens, and between local tokens belonging to a same local region.
   467	Cross-region feature interactions are entirely prohibited. This design
   468	primarily leverages the structural characteristics of pedestrian images,
   469	wherein the discriminative ability of ﬁne-grained local features (such as
   470	garment textures and accessory details) manifests predominantly within
   471	speciﬁc semantic regions (e.g., the textural correlation between cuﬀs
   472	6
   473	
   474	Neural Networks 201 (2026) 108883
   475	
   476	H. Du et al.
   477	
   478	and the body of a garment, or the detailed correspondence between
   479	shoulders and a backpack).
   480	Further, we calculate the output 𝑍̃ ∈ ℝ(𝑁+4)×𝐷 of the attention head
   481	using the following equation:
   482	(
   483	(
   484	)
   485	)
   486	𝑄𝐾 𝑇
   487	𝑍̃ = SoftMax √
   488	⊙ Mask 𝑉 ,
   489	(6)
   490	𝐷
   491	
   492	a foreground-background contrastive loss as follows:
   493	+
   494	⎧
   495	𝑁
   496	exp(sim(𝑓𝑓𝑖 𝑔 , 𝑓𝑓𝑖 𝑔 )∕𝜏)
   497	1 ∑
   498	⎪
   499	𝐿
   500	=
   501	−
   502	log
   503	∑𝑁
   504	⎪ 𝑓𝑔
   505	𝑗
   506	𝑖
   507	𝑁 𝑖=1
   508	⎪
   509	𝑗=1 exp(sim(𝑓𝑓 𝑔 , 𝑓𝑓 𝑔 )∕𝜏)
   510	⎪
   511	𝑖 )∕𝜏)
   512	𝑁
   513	,
   514	exp(1 − sim(𝑓𝑓𝑖 𝑔 , 𝑓𝑏𝑔
   515	⎨
   516	1 ∑
   517	log ∑𝑁
   518	⎪ 𝐿𝑓 𝑏 = −
   519	𝑗
   520	𝑗
   521	𝑁
   522	⎪
   523	𝑖=1
   524	𝑗=1 exp(1 − sim(𝑓𝑓 𝑔 , 𝑓𝑏𝑔 )∕𝜏)
   525	⎪
   526	⎪ 𝐿𝑐𝑡 = 𝐿𝑓 𝑔 + 𝐿𝑓 𝑏
   527	⎩
   528	
   529	where ⊙ denotes the matrix Hadamard product.
   530	Furthermore, we concatenate the outputs of all the attention heads
   531	and execute a linear transformation aimed at decreasing the dimension
   532	of the concatenated result to 𝐷, obtaining the output of the local-aware
   533	multi-head attention 𝑍̌ ∈ ℝ(𝑁+4)×𝐷 . Then, as shown in Fig. 5(a), we
   534	conduct the following operations to derive the feature vector sequence
   535	𝑍̂ ∈ ℝ(𝑁+4)×𝐷 :
   536	{
   537	𝑍𝑚 = LayerNorm(𝑍 ′ + ̌
   538	𝑍)
   539	,
   540	(7)
   541	̂
   542	𝑍 = LayerNorm(𝑍𝑚 + FeedForward(𝑍𝑚 ))
   543	
   544	where 𝑁 represents the quantity of training samples per mini-batch,
   545	sim(⋅) means the cosine similarity, 𝑓𝑓𝑖 𝑔 (𝑓𝑓𝑗 𝑔 ) means the 𝑖-th(𝑗-th) sam+
   546	
   547	ple’s foreground feature vector, 𝑓𝑓𝑖 𝑔 means the foreground feature vec𝑖 (𝑓 𝑗 ) means
   548	tor of one sample having the same ID as the 𝑖-th sample, 𝑓𝑏𝑔
   549	𝑏𝑔
   550	the 𝑖-th(𝑗-th) sample’s background feature vector, and 𝜏 means the distribution control factor.
   551	
   552	4. Experiments

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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' - | nl -ba | sed -n '250,664p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   250	3.1. Image-text feature representation
   251	Image feature representation. For image feature representation,
   252	we follow previous studies (Han et al., 2021; Yan et al., 2023a), that
   253	is, adopting the ViT initialized by pre-trained CLIP checkpoints as the
   254	image encoder. Speciﬁcally, for an image 𝐼𝑖 ∈ ℝ𝑐×𝑤×ℎ , we ﬁrst evenly
   255	split 𝐼𝑖 into 𝑛𝑣 = (ℎ × 𝑤)∕𝑝2 ﬁxed-dimension image patch sequences,
   256	where 𝑝 represents the size of image patches. Then the split image
   257	𝑛𝑣
   258	patch sequence is ﬂattened into 1D token sequence {𝑓𝑖𝑣 }|𝑖=1
   259	with a trainable fully connection layer. Finally, we concatenate the token sequence
   260	with a learnable [CLS] token as the preﬁx and feed it into the transformer blocks of ViT. The output image features can be represented as
   261	𝑓𝑖𝑣 = {𝑣𝑖𝑐𝑙𝑠 , 𝑣𝑖1 , 𝑣𝑖2 , …, 𝑣𝑖𝑛 }, where 𝑣𝑖𝑐𝑙𝑠 is the image-level global feature,
   262	𝑣
   263	{𝑣𝑖1 , 𝑣𝑖2 , …, 𝑣𝑖𝑛 } is the patch-level local features.
   264	𝑣
   265	Text feature representation. In terms of text feature representation, we adopt the text encoder of pre-trained CLIP as a text feature
   266	3
   267	
   268	Neural Networks 192 (2025) 107888
   269	
   270	R. Zeng et al.
   271	
   272	Fig. 2. Overview of the proposed HKGR: given image-text pairs, the image is encoded into an embedding by the image encoder. Simultaneously, the text is parsed
   273	into a scene graph, and then a knowledge-guided masking strategy is adopted to generate masked text. Finally, the masked text embedding obtained from the
   274	text encoder and image embedding are fed into a cross-modal interaction encoder to perform object-level reasoning, attribute-level reasoning, and relation-level
   275	reasoning.
   276	
   277	a logit vector {𝑝𝑗,𝑘
   278	𝑖 } for predicting the masked keywords/phrases. This
   279	can be formulated as follows:
   280	̂𝑡 𝑣 𝑣
   281	{𝑝𝑗,𝑘
   282	𝑖 } = 𝑀𝐿𝑃 (𝑇 𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚𝑒𝑟(𝑀𝐶𝐴(𝑓𝑖 , 𝑓𝑖 , 𝑓𝑖 )))
   283	(1)
   284	𝑡
   285	𝑣
   286	̂
   287	= 𝐶𝑀𝐸(𝑓𝑖 , 𝑓𝑖 )
   288	
   289	extractor. Speciﬁcally, for the input text 𝑇𝑖 , we use the lower-cased byte
   290	pair encoding (BPE) tokenizer to tokenize 𝑇𝑖 into a sequence of text tokens. We then add the speciﬁc token [SOS] at the start and [EOS] at the
   291	end of the token sequence. Finally, the token sequence is fed into the
   292	transformer blocks of the text encoder to obtain the ﬁnal text features
   293	𝑓𝑖𝑡 = {𝑡𝑖𝑠𝑜𝑠 , 𝑡𝑖1 , 𝑡𝑖2 , …, 𝑡𝑖𝑛 , 𝑡𝑖𝑒𝑜𝑠 }, where 𝑡𝑖𝑠𝑜𝑠 serve as sentence-level global fea𝑡
   294	ture of 𝑇𝑖 , and {𝑡𝑖𝑠𝑜𝑠 , 𝑡𝑖1 , 𝑡𝑖2 , …, 𝑡𝑖𝑛 } represent the word-level local features.
   295	𝑡
   296	Cross-modal interaction encoder. In order to construct the interaction between image and text modality, we adopt a transformer-based
   297	encoder for image and text features fusion. As shown in Fig. 3, the CrossModal Interaction Encoder(CME) contains transformer blocks composed
   298	of 4-layer self-attention layers with a feed-forward layer and a crossattention layer, and a prediction head composed of a multi-layer perceptron (MLP) with 2 hidden layers.
   299	Given a text 𝑇𝑖 , we mask a portion of the keywords/phrases derived
   300	from the scene graph parsed from the text to obtain masked text 𝑇̂𝑖 . The
   301	paired image 𝐼𝑖 and masked text 𝑇̂𝑖 are then fed into the image/text encoder to output image feature tokens 𝑓𝑖𝑣 and masked text feature tokens
   302	𝑓̂𝑖𝑡 = {𝑡̂𝑖𝑠𝑜𝑠 , 𝑡̂𝑖1 , 𝑡̂𝑖2 , …, 𝑡̂𝑖𝑛 , 𝑡̂𝑖𝑒𝑜𝑠 }. Image feature tokens 𝑓𝑖𝑣 and masked text fea𝑡
   303	ture tokens 𝑓̂𝑡 are concatenated and fed into the Transformer blocks. The
   304	
   305	Where, 𝑗 ∈ [1, 𝑛𝑡 ], 𝑘 ∈ [1, ||], || is the length of vocabulary , 𝑀𝐶𝐴(⋅)
   306	denotes multi-head cross-attention, which adopts masked text feature
   307	tokens 𝑓̂𝑖𝑡 as query and image feature tokens 𝑓𝑖𝑣 as key and value,
   308	𝑇 𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚𝑒𝑟(⋅) represents the Transformer block, 𝑀𝐿𝑃 (⋅) denotes the
   309	multi-layer perceptron, 𝐶𝑀𝐸(⋅) is the cross-modal interaction encoder
   310	module.
   311	3.2. Scene-graph construction
   312	We use out-of-the-box scene graph generation methods (Anderson
   313	et al., 2016) to obtain the scene graph. The method ﬁrst employs the
   314	Stanford Scene Graph Parser (Schuster et al., 2015) to convert the caption into a dependency tree structure. It then applies nine simple linguistic rules to transform the tree into a scene graph, and ﬁnally generates a set of tuples that represent the connection logic of the scene
   315	graph. Speciﬁcally, Given an input text 𝑇𝑖 , the scene graph can be represented as 𝐺(𝑇𝑖 ) = {𝑉 (𝑇𝑖 ), 𝐸(𝑇𝑖 )}, where 𝑉 (𝑇𝑖 ) = {𝑂, 𝐴, 𝑅} denote the
   316	nodes set, which includes object nodes, attribute nodes, and relation
   317	nodes. 𝐸(𝑇𝑖 ) = {𝑃 , 𝑄} is the edge set, where 𝑃 ⊆ 𝑂 × 𝐴 is the set of attribute pairs, and 𝑄 ⊆ 𝑂 × 𝑅 × 𝑂 is the set of relation triplet.
   318	A typical example is illustrated in Fig. 2, where the green rectangles represent object nodes, the orange rectangles represent attribute
   319	nodes, and the blue rectangles represent relation nodes. The black arrows indicate the edge of the attribute or relation associated with the
   320	corresponding node. The scene graph parsed from the text is presented
   321	in Table 1.
   322	
   323	𝑖
   324	
   325	resulting tokens are then further processed by an MLP head to produce
   326	
   327	3.3. Knowledge-guided reasoning
   328	Utilizing hierarchical knowledge obtained from scene graphs,
   329	HKGR learns the multi-level cross-modal alignments. As shown
   330	
   331	Fig. 3. Illustration of our cross-modal interaction encoder.
   332	4
   333	
   334	Neural Networks 192 (2025) 107888
   335	
   336	R. Zeng et al.
   337	
   338	Table 1
   339	Illustration of scene graph parsed from text.
   340	text: 𝑇𝑖
   341	
   342	A lady with black hair is
   343	standing in front of a fence,
   344	she is wearing a white shirt
   345	with black pants.
   346	
   347	object: 𝑂
   348	
   349	{(lady), (hair), (fence),
   350	(shirt), (pant)}
   351	
   352	attribute: 𝑃
   353	
   354	{(lady,a), (hair,black),
   355	(fence), (shirt,white),
   356	(pant,black)}
   357	
   358	relation: 𝑄
   359	
   360	{(lady,in front of, fence),
   361	(lady, waring, shirt), (lady,
   362	with, hair)}
   363	
   364	̂𝑎,𝑡 𝑣
   365	where 𝑝𝑗,𝑘
   366	𝑖,𝑎 = 𝐶𝑀𝐸(𝑓𝑖 , 𝑓𝑖 ) is predicted token probability of 𝑘-th word
   367	in  of masked 𝑗-th attribute token, and 𝑦𝑗𝑘 is one-hot vocabulary indicator.
   368	Relation-level reasoning. In general, relations depict actions or
   369	spatial relations between the objects, distinguishing the pedestrians with
   370	similar attributes but varying relations. In the relation-level reasoning,
   371	given the input image text pairs (𝐼𝑖 , 𝑇𝑖 ), we ﬁrst obtain all the relation triplet 𝑄 = {{𝑜1𝑖 , 𝑟𝑖 , 𝑜2𝑖 }|𝑖 = 1, 2, … , 𝑛𝑞 } from the scene graph of text,
   372	where 𝑟𝑖 ∈ 𝑅 denotes the relation nodes corresponding to the object
   373	nodes pairs{𝑜1𝑖 , 𝑜2𝑖 }. Similarly, we randomly select 30 % of the relation
   374	triplets in the scene graph for masking and apply the same masking strategy described in object-level reasoning. The selected relation triplet can
   375	be represented as 𝑄̂ = {{𝑜̂1𝑖 , 𝑟̂𝑖 , 𝑜̂2𝑖 }|𝑖 = 1, 2, … , 𝑛̂ 𝑞 }. Then, we perform the
   376	masking strategy of the relation triplet. We keep the associated object
   377	node pairs {𝑜̂1𝑖 , 𝑜̂2𝑖 } and mask the sub-sequences of the text that correspond to the selected relation nodes {̂𝑟𝑖 }. At last, the relation masked
   378	text 𝑇̂𝑖𝑟 and image 𝑇𝑖 are fed into the text/image encoder to generate
   379	masked text features𝑓̂𝑖𝑟,𝑡 and image features 𝑓𝑖𝑣 .
   380	In relation level reasoning, the goal is to recover the masked relation
   381	tokens {̂𝑟𝑖 } based on the corresponding object pairs {𝑜̂1𝑖 , 𝑜̂2𝑖 }, as well as
   382	masked text features𝑓̂𝑖𝑟,𝑡 and image features 𝑓𝑖𝑣 . The loss function can be
   383	formulated as follows:
   384	
   385	in Fig. 2, according to the scene graph constructed from the
   386	text, we construct a knowledge-guided reasoning module, including object-level reasoning, attribute-level reasoning, and relationlevel reasoning. These reasoning processes guide the model to
   387	learn hierarchical semantic local alignments between image and text
   388	modalities.
   389	Object-level reasoning. Objects are the fundamental elements in a
   390	visual scene and thus play a crucial role in constructing semantic information. Through object-level reasoning, the model is learning to establish the object-level visual-text connection. Given the paired image 𝐼𝑖
   391	and text 𝑇𝑖 , we ﬁrst obtain all the object nodes 𝑂 = {𝑜𝑖 |𝑖 = 1, 2, … , 𝑛𝑜 }
   392	from the scene graph of text. We randomly select 30 % of object nodes
   393	for masking. Among the selected nodes, we replace 80 % of the nodes
   394	with the special token [MASK], 10 % of the nodes with random tokens, and 10 % of the nodes remain unchanged. Then, we implement
   395	object masking by masking the sub-sequences in the text that correspond to the selected object nodes 𝑂̂ = {𝑜̂𝑖 |𝑖 = 1, 2, … , 𝑛̂ 𝑜 }. Finally, the
   396	object masked text 𝑇̂𝑖𝑜 and image 𝑇𝑖 are fed into the image-text encoder
   397	to obtain masked text features𝑓̂𝑖𝑜,𝑡 and image features 𝑓𝑖𝑣 .
   398	In the object-level reasoning, our goal is to recover the masked
   399	object nodes 𝑂̂ tokens based on the masked text features𝑓̂𝑜,𝑡
   400	
   401	𝑟𝑒𝑙 = −
   402	
   403	̂𝑟,𝑡 𝑣
   404	where 𝑝𝑗,𝑘
   405	𝑖,𝑟 = 𝐶𝑀𝐸(𝑓𝑖 , 𝑓𝑖 ) is predicted token probability of 𝑘-th word
   406	in  of masked 𝑗-th relation token, and 𝑦𝑗𝑘 is one-hot vocabulary indicator.
   407	3.4. Multi-grained semantic alignment
   408	Most existing CLIP-based TIReID methods only employ image-level
   409	and sentence-level global features to attain coarse-grained cross-modal
   410	semantic alignment. However, focusing solely on coarse-grained semantic alignment may not capture the ﬁne-grained semantic interactions
   411	between image and text modalities, thereby hindering retrieval performance gains. To tackle this limitation, we propose a Multi-Grained Semantic Alignment (MGA) module to mine discriminative feature representations and perform multi-grained cross-modal alignments. The MAG
   412	module consists of coarse-grained semantic alignment and ﬁne-grained
   413	semantic alignment.
   414	For coarse-grained semantic alignment, we employ image-level global
   415	feature 𝑣𝑖𝑐𝑙𝑠 from the image encoder and sentence-level global feature 𝑡𝑖𝑠𝑜𝑠
   416	from the text encoder as coarse-grained feature representation. Then,
   417	coarse-grained semantic alignment is carried out by minimizing the
   418	KL divergence between global image-text feature similarity distribution
   419	𝜓(𝑆(𝑓𝑖𝑣 , 𝑓𝑖𝑡 )) and the label distribution 𝑃 (𝑌𝑖 ). The loss function can be
   420	formulated as:
   421	
   422	𝑖
   423	
   424	exp 𝑝𝑗,𝑘
   425	∑ ∑ 𝑗
   426	𝑖,𝑜
   427	1
   428	𝑦𝑘 log ∑||
   429	̂
   430	|𝑂|||
   431	exp 𝑝𝑗,𝑙
   432	𝑗∈| ̂
   433	𝑂| 𝑘∈||
   434	𝑖,𝑜
   435	
   436	(2)
   437	
   438	𝑙=1
   439	
   440	̂𝑜,𝑡 𝑣
   441	where  denotes the set of vocabulary, 𝑝𝑗,𝑘
   442	𝑖,𝑜 = 𝐶𝑀𝐸(𝑓𝑖 , 𝑓𝑖 ) is predicted
   443	token probability of 𝑘-th word in  of masked 𝑗-th object token, and 𝑦𝑗𝑘
   444	is one-hot vocabulary indicator.
   445	Attribute-level reasoning. Attributes describe speciﬁc characteristics of objects, such as color or size, which contribute to distinguishing
   446	pedestrians with similar appearances but diﬀerent attributes.
   447	Given the input image text pairs (𝐼𝑖 , 𝑇𝑖 ), we ﬁrst obtain all the attribute pairs 𝑃 = {{𝑜𝑖 , 𝑎𝑖 }|𝑖 = 1, 2, … , 𝑛𝑝 } from the scene graph of text,
   448	where 𝑎𝑖 ∈ 𝐴 denotes the attribute nodes corresponding to the object
   449	nodes. Similarly, we randomly select 30 % of the attribute pairs in the
   450	scene graph for masking and apply the same masking strategy described
   451	in object-level reasoning. The selected attribute pairs are reorganized as
   452	𝑃̂ = {{𝑜̂𝑖 , 𝑎̂𝑖 }|𝑖 = 1, 2, … , 𝑛̂ 𝑝 }. Then, we implement attribute masking by
   453	masking the sub-sequences of the text that correspond to the selected
   454	attribute nodes 𝑎̂𝑖 while keeping the associated object node 𝑜̂𝑖 . Finally,
   455	the attribute masked text 𝑇̂𝑖𝑎 and image 𝑇𝑖 are fed into the text/image
   456	encoder to generate masked text features𝑓̂𝑖𝑎,𝑡 and image features 𝑓𝑖𝑣 .
   457	The attribute-level reasoning aims to recover the masked attribute
   458	nodes {𝑎̂𝑖 |𝑖 = 1, 2, … , 𝑛̂ 𝑎 } tokens based on the masked text features𝑓̂𝑖𝑎,𝑡
   459	and image features 𝑓𝑖𝑣 by minimizing the cross-entropy loss as follows:
   460	𝑎𝑡𝑡𝑟 = −
   461	
   462	exp 𝑝𝑗,𝑘
   463	∑ ∑ 𝑗
   464	𝑖,𝑎
   465	1
   466	𝑦𝑘 log ∑||
   467	|𝑃̂ ||| 𝑗∈|𝑃̂ | 𝑘∈||
   468	exp 𝑝𝑗,𝑘
   469	𝑖,𝑎
   470	
   471	(4)
   472	
   473	𝑘=1
   474	
   475	and image features 𝑓𝑖𝑣 by minimizing the cross-entropy loss as
   476	follows:
   477	𝑜𝑏𝑗 = −
   478	
   479	exp 𝑝𝑗,𝑘
   480	∑ ∑ 𝑗
   481	𝑖,𝑟
   482	1
   483	𝑦𝑘 log ∑||
   484	̂
   485	|𝑄|||
   486	exp 𝑝𝑗,𝑙
   487	𝑗∈| ̂
   488	𝑄| 𝑘∈||
   489	𝑖,𝑟
   490	
   491	𝑐𝑔
   492	𝑚𝑔𝑎 =
   493	
   494	𝑁
   495	∑
   496	
   497	( ( (
   498	))
   499	)
   500	𝐾𝐿 𝜓 𝑆 𝑓𝑖𝑣 , 𝑓𝑖𝑡 ∕𝜏, 𝑃 (𝑌𝑖 )
   501	
   502	(5)
   503	
   504	𝑖=1
   505	
   506	For ﬁne-grained semantic alignment, we observe that meaningless tokens can distract the model from capturing discriminative cues. Therefore, inspired by the previous works (Jiang & Ye, 2023; Qin et al., 2024;
   507	Yan et al., 2023a), we ﬁrst select informative local token features from
   508	both image and text modalities using correlation weights. In practice,
   509	take the visual process for example, we can obtain the attention map
   510	𝐴𝑣𝑖 ∈ ℝ(1+𝑛𝑣 )×(1+𝑛𝑣 ) from the last transformer block of the image encoder,
   511	which reﬂects the importance cores between 1 + 𝑛𝑣 tokens (including
   512	[CLS] token in the ﬁrst place). Then the correlation weight between
   513	[CLS] token and local tokens 𝑎𝑣𝑖 = 𝐴𝑣𝑖 [0, 1 ∶] ∈ ℝ𝑛𝑣 are used to select
   514	the top-K informative local token features 𝑓̂𝑖𝑣 = {𝑣𝑖𝑘𝑣 , 𝑣𝑖𝑘𝑣 , …, 𝑣𝑖𝑘𝑣 }, where
   515	1
   516	
   517	2
   518	
   519	𝑛
   520	
   521	𝑘𝑣𝑛 = 𝑅 × 𝑛𝑣 denotes the indices of the selected local tokens and 𝑅 is
   522	the token selection ratio. In terms of textual procedure, we can obtain
   523	
   524	(3)
   525	
   526	𝑘=1
   527	
   528	5
   529	
   530	Neural Networks 192 (2025) 107888
   531	
   532	R. Zeng et al.
   533	
   534	the selected local text features in a similar way as 𝑓̂𝑖𝑡 = {𝑡𝑖 𝑡 , 𝑡𝑖 𝑡 , …, 𝑡𝑖 𝑡 },
   535	𝑘1
   536	
   537	𝑘2
   538	
   539	2021) dataset consists of 20,505 pedestrian images and 41,010 text descriptions with two captions per image. The average length of text descriptions is 23 words. The dataset is divided into three subsets, with
   540	3701 identities for training, 200 identities for validation, and 200 identities for testing.
   541	
   542	𝑘𝑛
   543	
   544	where 𝑘𝑡𝑛 = 𝑅 × 𝑛𝑡 . Finally, for the selected local visual token features 𝑓̂𝑖𝑣
   545	and textual token features 𝑓̂𝑖𝑡 , we perform linear feature transformation
   546	and max-pooling aggregation to obtain the ﬁnal ﬁne-grained local im𝑣,𝑖
   547	𝑡,𝑖
   548	age embedding 𝑓𝑙𝑜𝑐
   549	and text embedding 𝑓𝑙𝑜𝑐
   550	, which can be formulated
   551	as follows:
   552	( (
   553	(
   554	(
   555	( )))
   556	( )))
   557	𝑡,𝑖
   558	𝑓𝑙𝑜𝑐
   559	= 𝑀𝑎𝑥𝑃 𝑜𝑜𝑙 𝜎 𝐵𝑁 𝐹 𝐶 𝐹 𝐶 𝑓̂𝑖𝑡
   560	+ 𝐹 𝐶 𝑓̂𝑖𝑡
   561	(6)
   562	(
   563	(
   564	(
   565	(
   566	))
   567	(
   568	)))
   569	𝑓 𝑣,𝑖 = 𝑀𝑎𝑥𝑃 𝑜𝑜𝑙 𝜎 𝐵𝑁 𝐹 𝐶 𝐹 𝐶(𝑓̂𝑣 ) + 𝐹 𝐶 𝑓̂𝑣
   570	𝑖
   571	
   572	𝑙𝑜𝑐
   573	
   574	4.2. Evaluation metrics
   575	For performance evaluation, we adopt extensively used metrics,
   576	namely, Recall at K, to measure our model. Following the settings of
   577	prior works, we adopt K=1, 5, and 10, denoted as R@1, R@5, and
   578	R@10, respectively. The R@K indicator deﬁnes the probability of obtaining correct retrieval results within the top-K positions of the ranking
   579	list, given a text description as the query. In addition, for a more thorough assessment, we also leverage the mean Average Precision (mAP) as
   580	an additional retrieval criterion. Higher values of R@K and mAP mean
   581	better performance.
   582	
   583	𝑖
   584	
   585	where 𝑀𝑎𝑥𝑃 𝑜𝑜𝑙(⋅) is the max-pooling function, 𝜎(⋅) is the ReLU activation function, 𝐵𝑁(⋅) is the batch normalization, and 𝐹 𝐶(⋅) denotes the
   586	fully-connection layer. Finally, the ﬁne-grained semantic alignment is
   587	performed by minimizing the KL divergence between the local image𝑡,𝑖
   588	𝑣,𝑖
   589	text feature similarity distributions 𝜓(𝑆(𝑓𝑙𝑜𝑐
   590	, 𝑓𝑙𝑜𝑐
   591	)) and the label distribution 𝑃 (𝑌𝑖 ). The loss function for ﬁne-grained semantic alignment can
   592	be formulated as follows:
   593	𝑔
   594	𝑓𝑚𝑔𝑎
   595	=
   596	
   597	𝑁
   598	∑
   599	
   600	( ( ( 𝑡,𝑖 𝑣,𝑖 ))
   601	)
   602	𝐾𝐿 𝜓 𝑆 𝑓𝑙𝑜𝑐
   603	, 𝑓𝑙𝑜𝑐 ∕𝜏, 𝑃 (𝑌𝑖 )
   604	
   605	(7)
   606	
   607	4.3. Implementation details
   608	
   609	𝑖=1
   610	
   611	Where 𝑁 and 𝜏 represent the mini-batch size and temperature parameter, respectively. 𝜓(⋅) denotes the softmax operation, and 𝑆(⋅) denotes
   612	the cosine similarity function.
   613	The ﬁnal loss function of the MGA is deﬁned as:
   614	𝑔
   615	𝑚𝑔𝑎 = 𝑓𝑚𝑔𝑎
   616	+ 𝑐𝑔
   617	𝑚𝑔𝑎
   618	
   619	We initialized the image and text encoder with pre-trained weights
   620	from CLIP-ViT-B/16, while the multimodal interaction encoder is randomly initialized. For training, we adopt the Adam (Kingma & Ba, 2014)
   621	optimizer with an initial learning rate 1 × 10−5 and cosine learning rate
   622	decay. Our model is trained for a total of 60 epochs, beginning with
   623	5 epochs for warming up, during which the initial learning rate is set
   624	to 1 × 10−6 and linearly increased to 1 × 10−5 . The token selection ratio
   625	𝑅 of MGA is set to 0.3. For image data augmentation, we adopt random crop, random erasing, and random horizontal ﬂipping strategies.
   626	For text data augmentation, we apply random masking, replacement,
   627	and deletion strategies. All the image sizes are set to 384 × 128 instead
   628	of 224 × 224. We set the training batch size to 64 and the temperature
   629	parameter 𝜏 to 0.02. We conduct our experiments on a single NVIDIA
   630	Tesla A100 GPU with 64GB of memory.
   631	
   632	(8)
   633	
   634	3.5. Objective function and training strategy
   635	To facilitate knowledge-guided reasoning, we adopt a multi-level
   636	reasoning strategy that incorporates object-level, attribute-level, and
   637	relation-level reasoning. Speciﬁcally, during each training epoch, we
   638	sequentially and cyclically implement object-level reasoning, attributelevel reasoning, and relation-level reasoning until the model converges.
   639	Following the previous work (Zheng et al., 2020), we also adopt an instance loss 𝑖𝑑 that classiﬁes each image and text into speciﬁc classes
   640	based on their corresponding individual IDs, allowing the model to differentiate between image and text groups corresponding to distinct individuals. Above all, the ﬁnal loss function can be formulated as follows:
   641	 = 𝑚𝑔𝑎 + 𝑖𝑑 + 𝑘𝑔𝑟
   642	
   643	4.4. Baselines
   644	We compare our model with 2 classes of SoTA methods in TIReID:
   645	1) CLIP-Free models. Utilizing the single-modality pre-training model
   646	under images (e.g., ResNet50) and the single-modality pre-training
   647	model under texts (e.g., BERT), to encode the image and text respectively: Dual-Path (Zheng et al., 2020), CMPM/C (Zhang & Lu,
   648	2018), TIMAM (Saraﬁanos et al., 2019), ViTAA (Wang et al., 2020b),
   649	NAFS Gao et al. (2021), DSSL (Zhu et al., 2021), SSAN Ding et al.
   650	(2021), LapsCore (Wu et al., 2021), LBUL (Wang et al., 2022b), SAF (Li
   651	et al., 2022), TIPCB (Chen et al., 2022), CAIBC (Wang et al., 2022a),
   652	AXM-Net (Farooq et al., 2022), LGUR (Shao et al., 2022), IVT (Shu
   653	et al., 2022), CAPL (Niu et al., 2024), PBSL (Shen et al., 2023), and
   654	UniPT Shao et al. (2023).
   655	2) CLIP-based models. Leveraging the multi-modal pre-training model
   656	(e.g., CLIP) to encode images and text: CFine (Yan et al., 2023a), TextReID (Han et al., 2021), VGSG He et al. (2023), IRRA (Jiang & Ye,
   657	2023), BiLMa (Fujii & Tarashima, 2023) and TBPS-CLIP (Cao et al.,
   658	2024).
   659	
   660	(9)
   661	
   662	where 𝑘𝑔𝑟 is knowledge-guided reasoning loss, which integrates 𝑜𝑏𝑗 ,
   663	𝑎𝑡𝑡𝑟 , and 𝑟𝑒𝑙 follow the multi-level reasoning strategy.
   664	4. Experiment settings and baselines

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '1020,1350p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1020	phases, we set three distinct random seeds for each experiment and use
  1021	the average of the results of three independent trials as the ﬁnal result
  1022	of each experiment. The values of key parameters, hardware and software requirements, and parameter count of our model are presented in
  1023	Table 2.
  1024	4.3. Comparison with state-of-the-art methods
  1025	To evaluate the eﬃcacy of HFLAT, we conduct comparative experiments against several state-of-the-art (SOTA) methods for occluded person ReID on the occlusion datasets Occluded-DukeMTMC and OccludedReID, as well as on the standard datasets Market-1501 and DukeMTMCReID.
  1026	8
  1027	
  1028	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '35,555p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    35	Cloth-Changing person Re-Identification (CC-ReID) is a
    36	long-term retrieval task, which aims at re-identifying target
    37	persons across non-overlapping cameras. Compared with traditional ReID [1]–[6], CC-ReID [7], [8] is encountering more
    38	realistic challenges. Despite being quite challenging, CC-ReID
    39	is receiving more and more interest from researchers due to
    40	its crucial role in more realistic scenario applications.
    41	To address CC-ReID, previous methods [9]–[11] aim to
    42	eliminate the impact of clothes, and extract the inherent
    43	characteristics of pedestrians, such as 3D human shape, gait
    44	information, contour sketches, etc. However, these inherent
    45	characteristics are not as effective as appearance features,
    46	leading to some performance deteriorations on the sameclothing ReID. Furthermore, various approaches rooted in metric learning [12], [13] and data augmentation techniques [14],
    47	[15] have been introduced to tackle the CC-ReID problem.
    48	In fact, a critical step in ReID is to design a good distance
    49	metric [16]. As shown in Fig. 1 (a) and (b), due to the
    50	This work was supported in part by the National Natural Science Foundation
    51	of China (No. 62102208).
    52	
    53	Dinter
    54	
    55	P1
    56	
    57	C1
    58	
    59	P2
    60	
    61	Dintra
    62	
    63	(a) Instance-level Metric Learning
    64	
    65	ipos Pull
    66	
    67	i
    68	
    69	MPL
    70	
    71	 pneg
    72	
    73	Push
    74	
    75	 i ,1
    76	
    77	Augmented Instances
    78	Pull
    79	
    80	i
    81	
    82	Push
    83	
    84	Pull
    85	
    86	 i ,C
    87	
    88	Dinter
    89	
    90	Dintra
    91	
    92	C3
    93	
    94	C4
    95	
    96	(b) Illustration of the Feature Distribution
    97	Clothe 1
    98	
    99	f p2,1
   100	
   101	p
   102	
   103	f p1,1
   104	
   105	 p ,1
   106	fˆi ,kc
   107	
   108	Dinter
   109	
   110	C2
   111	
   112	Dintra
   113	
   114	Dintra
   115	
   116	Push
   117	
   118	S2MPL
   119	
   120	p
   121	 p ,1
   122	
   123	 p ,C
   124	
   125	Clothe C
   126	
   127	1
   128	f pK,1 f p ,C
   129	
   130	Pull
   131	
   132	f p2,C
   133	
   134	Sub Proxies
   135	
   136	f pK,C
   137	
   138	Raw Instances
   139	
   140	 p ,C
   141	
   142	SPL
   143	
   144	Main Proxies
   145	
   146	Generate Proxy
   147	
   148	MPL
   149	
   150	Main Proxy Learning
   151	
   152	S2MPL
   153	
   154	Sub-to-Main Proxy Learning
   155	
   156	SPL
   157	
   158	Sub Proxy Learning
   159	
   160	(c) Hierarchical Proxy Learning
   161	
   162	Fig. 1. Our motivations. (a) Geometry interpretation of instance-level metric
   163	learning. (b) Illustration of the feature distribution of randomly selected
   164	persons from CC-ReID datasets. (c) Geometry interpretation of the proposed
   165	Hierarchical Proxy Learning (HPL). Different colored dots and shapes represent different persons and sub proxies identities, respectively.
   166	
   167	large intra-person variance and small inter-person variance
   168	caused by changing clothes, the instance-level triplet loss [17]
   169	and contrastive loss [18], [19] cannot achieve satisfactory
   170	performance. Recently, some works [20]–[22] perform ReID
   171	by proxy-based metric learning. For example, Wang et al. [23]
   172	propose intra-camera and inter-camera proxy contrastive learning. For CC-ReID, Gu et al. [24] design a clothes-based
   173	adversarial loss to further pull the features with the same identity closer. Unfortunately, both of them focus on instance-toproxy interactions, and neglect inter-proxy relations. Different
   174	from previous methods, as shown in Fig. 1 (c), we propose a
   175	Hierarchical Proxy Learning (HPL) framework, which consists
   176	of a Main Proxy Learning (MPL), a Sub Proxy Learning
   177	(SPL) and a Sub-to-Main Proxy Learning (S2MPL). In MPL,
   178	we first create main proxies for individuals, then bring the
   179	proxies of the same person closer while distancing those of
   180	different people, enhancing inter-person separability. In SPL,
   181	we create sub proxies for each person and group instances
   182	with different sub proxies to improve intra-person compactness. Unlike [24] which directly using clothing labels as sub
   183	proxies, we propose a clustering-based automatic sub-proxy
   184	mining scheme. In S2MPL, each sub-proxy acts as an anchor,
   185	being pulled toward its corresponding main proxy and pushed
   186	away from others, promoting inter-person diversity and intraperson compactness. Such a hierarchical structure contributes
   187	
   188	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:35 UTC from IEEE Xplore. Restrictions apply.
   189	
   190	Sub-proxy Mining Original
   191	Clustering
   192	
   193	C1
   194	
   195	C2
   196	
   197	Images
   198	
   199	Pb
   200	
   201	C2
   202	
   203	Human Parsing
   204	
   205	Pa
   206	
   207	︙
   208	Clustering
   209	C1
   210	
   211	Pb
   212	
   213	Pa
   214	
   215	SCT
   216	
   217	SBS
   218	
   219	C3
   220	
   221	Classifier
   222	
   223	Pb
   224	
   225	Pa
   226	
   227	𝐿𝐻𝑃𝐿
   228	Pull
   229	Pull
   230	
   231	Pb
   232	
   233	: Main Proxy
   234	
   235	: Sub Proxy
   236	
   237	Pull
   238	
   239	Clothes Change
   240	
   241	Augmented
   242	Images
   243	
   244	𝐿𝐼𝐷
   245	Push
   246	
   247	GAP+GMP
   248	
   249	a
   250	
   251	C4
   252	
   253	ResNet-50
   254	
   255	C1 P C2
   256	
   257	C3
   258	
   259	: Instances of
   260	Different Sub
   261	Proxies
   262	Human Parsing
   263	: A Pre-trained
   264	Human Parsing
   265	Network
   266	: Pull
   267	
   268	Part1: Sample Balancing and Diversity Strategy
   269	
   270	Part2: Hierarchical Proxy Learning
   271	
   272	: Push
   273	
   274	Fig. 2. Illustration of the proposed framework.
   275	
   276	to extracting person-invariant and clothes-irrelevant features.
   277	As shown in Fig. 1 (c), when constructing a hierarchical
   278	structure, if there is no assistance, the main proxy P1 in a minibatch will have no corresponding positive samples. What’s
   279	more, if there is no constraint, the distribution of sub proxies
   280	will be random. Meanwhile, due to the annotation limitation
   281	in current CC-ReID datasets, it is highly possible for a person
   282	who wears one clothes all the time. Considering the above
   283	issues, we further propose a Sample Balance and Diversity
   284	(SBD) module, which balances the number of sub proxies
   285	in a mini-batch and utilizes semantic guidance to enrich
   286	the diversity of clothes, simultaneously. Specifically, we first
   287	explore a Sub-proxy Balanced Sampling (SBS) strategy taking
   288	the balance and diversity of sub proxies into consideration,
   289	which is more suitable for CC-ReID. Then, a Semanticguided Clothes Transfer (SCT) is proposed to enrich the
   290	diversity of clothes, which utilizes a pre-trained human parsing
   291	network [25] to guide clothing changing. Thanks to SCT, we
   292	can get the main proxy positive samples corresponding to
   293	pedestrians based on the augmented samples. Experimental
   294	results demonstrate that our method significantly outperforms
   295	most state-of-the-art works on two public CC-ReID datasets.
   296	The contributions of our work can be summarized as: (1)
   297	We propose an effective data processing module named SBD
   298	for CC-ReID. (2) We propose a novel proxy-level metric
   299	learning method with a hierarchical structure to extracting
   300	person-invariant and clothes-irrelevant features. (3) Extensive
   301	experiments demonstrate that our proposed method outperforms most state-of-the-art cloth-changing methods on two
   302	widely-used CC-ReID datasets, i.e., PRCC and VC-Clothes.
   303	
   304	clustering-based automatic sub-proxy mining scheme. Specifically, before each round of network training, we cluster all
   305	Np
   306	the feature representations {fnp }n=1
   307	for each person p into Cp
   308	clusters whose pseudo-labels are used as the sub-proxies. In
   309	practice, we adopt the DBSCAN [28] method for clustering.
   310	Sub-proxy Balanced Sampling. The sampling strategy [17]
   311	in traditional ReID mainly considers the balance of different
   312	persons but ignores the balance of different clothes. Intuitively,
   313	it is useful to choose balanced sub-proxy in each batch for CCReID. Therefore, we propose a Sub-proxy Balanced Sampling
   314	(SBS) strategy. We choose P persons in each mini-batch,
   315	where C sub proxies per person and K images per sub proxy.
   316	Our SBS strategy performs a balanced optimization of persons
   317	and sub-proxy, thereby promoting the learning efficacy.
   318	Semantic-guided Clothes Transfer. In CC-ReID datasets,
   319	some persons may wear only one clothes all the time. Data
   320	augmentation is an effective strategy to enrich the diversity of
   321	training samples in CC-ReID. We propose a Semantic-guided
   322	Clothes Transfer (SCT) to change clothes among different persons. Specifically, given one image xi ∈ {xkp,c }P,C,K
   323	p=1,c=1,k=1 ,
   324	we first randomly select another image xj with different person
   325	and sub proxy in a mini-batch. Then, a pre-trained human
   326	parsing network [25] is employed to obtain semantic masks
   327	of xi and xj . Considering that the most common dressing
   328	parts for persons are upper-clothes and pants, we perform SCT
   329	based on the masks of upper-clothes and pants, respectively.
   330	Given the upper-clothes masks mi and mj of two pedestrians,
   331	we can transfer the upper-clothes of xj to xi ,
   332	
   333	II. M ETHODS
   334	
   335	where means the matrix multiplication. M ean(·) calculates
   336	the average pixel value of the upper-clothes to address the variability in the clothing area of different persons. Reshape(·)
   337	duplicates the pixel value to the same shape of the target image
   338	xi . Similarly, we can change pants from xj to xi .
   339	In one mini-batch, we get the corresponding augmented
   340	image x̂ of each image x ∈ {xkp,c }P,C,K
   341	p=1,c=1,k=1 through SCT.
   342	Meanwhile, its person label remains unchanged while the
   343	clothes label has been changed. Thus, our SCT can generate
   344	more training samples for one person dressing in different
   345	clothes, which enriches the diversity of samples for CC-ReID.
   346	
   347	As illustrated in Fig. 2, our proposed framework mainly includes two components: Sample Balance and Diversity (SBD)
   348	module and Hierarchical Proxy Learning (HPL). Detailed
   349	descriptions are presented in the following sections.
   350	A. Sample Balance and Diversity
   351	Recent methods [24], [26], [27] utilize clothes labels to
   352	improve the performance of CC-ReID. However, obtaining the
   353	clothes labels requires a certain price. Inspired by the recent
   354	success of unsupervised person ReID methods, we propose a
   355	
   356	x̂i = xi
   357	
   358	(1 − mi ) + Reshape(M ean(xj
   359	
   360	mj ))
   361	
   362	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:35 UTC from IEEE Xplore. Restrictions apply.
   363	
   364	mi , (1)
   365	
   366	B. Hierarchical Proxy Learning
   367	xkp,c
   368	
   369	For feature extraction, we feed the original images
   370	and augmented images x̂kp,c into the ResNet-50 following a
   371	GAP and a GMP to obtain the feature vectors f k , fˆk .
   372	p,c
   373	
   374	p,c
   375	
   376	In our baseline method, we adopt the cross-entropy loss for
   377	person classification. However, merely person classification is
   378	hard to reduce the intra-person variance and increase the interperson variance. As shown in Fig. 1 (c), we propose a novel
   379	Hierarchical Proxy Learning (HPL) framework, including a
   380	Main Proxy Learning (MPL), a Sub Proxy Learning (SPL)
   381	and a Sub-to-Main Proxy Learning (S2MPL).
   382	Main Proxy Learning. As stated in [24], the instance-level
   383	metric learning may lead to a sub-optimization for CC-ReID,
   384	because it only mine the hard cases in a mini-batch and is
   385	sensitive to noisy positives and negatives. Thus, to alleviate
   386	this problem, we propose a Main Proxy Learning (MPL). The
   387	illustration of our MPL is shown in the blue part of Fig. 1 (c).
   388	Specifically, based on the features of persons, the main proxy
   389	ρi can be constructed by:
   390	ρi =
   391	
   392	C,K
   393	X
   394	
   395	1
   396	CK
   397	
   398	k
   399	fi,c
   400	, i ∈ [1, P ].
   401	
   402	(2)
   403	
   404	c=1,k=1
   405	
   406	The main proxy ρi can be seen as an anchor. Then, we
   407	can obtain the corresponding positive main proxy ρpos
   408	from
   409	i
   410	augmented samples by:
   411	ρpos
   412	=
   413	i
   414	
   415	1
   416	CK
   417	
   418	C,K
   419	X
   420	
   421	k
   422	fˆi,c
   423	, i ∈ [1, P ].
   424	
   425	(3)
   426	
   427	c=1,k=1
   428	
   429	Afterward, the negative main proxy ρneg
   430	which has a different
   431	i
   432	person label with the anchor can be defined as:
   433	ρneg
   434	=
   435	p
   436	
   437	1
   438	CK
   439	
   440	C,K
   441	X
   442	
   443	k
   444	fp,c
   445	, p 6= i, p ∈ [1, P ].
   446	
   447	(4)
   448	
   449	c=1,k=1
   450	
   451	For one anchor, we have J = (P − 1) × 2 negative main
   452	proxies. Thus, the loss of MPL can be defined as:
   453	P
   454	
   455	1 X
   456	neg
   457	LM P L =
   458	α+D(ρi , ρpos
   459	i )−minD(ρi , ρp ) + ,
   460	P i=1
   461	
   462	(5)
   463	
   464	where D(, ) is the Euclidean distance, min represents the
   465	minimized distances among negative pairs for obtaining the
   466	hardest negative main proxies in the mini-batch. α is a margin
   467	hyper-parameter and [·]+ represents the hinge loss. Different
   468	from previous methods, our proposed MPL can suppress the
   469	influence of noisy samples in feature optimization.
   470	Sub Proxy Learning. Our MPL does not take the intraperson compactness into account. Thus, as shown in the green
   471	part of Fig. 1 (c), we further propose the Sub Proxy Learning
   472	(SPL) to resolve this problem. Specifically, thanks to SBS, we
   473	can sample C sub proxies for the p-th person in a mini-batch
   474	and construct sub proxy by:
   475	K
   476	
   477	δp,c =
   478	
   479	1 X k
   480	fp,c , c ∈ [1, C], p ∈ [1, P ].
   481	K
   482	k=1
   483	
   484	(6)
   485	
   486	In SPL, we constrain the feature learning to pull the different
   487	sub proxies of the same person closer. Thus, the loss of our
   488	SPL can be expressed as:
   489	LSP L =
   490	
   491	P X
   492	C X
   493	K
   494	C
   495	X
   496	X
   497	
   498	k
   499	D(δp,c , fp,i
   500	),
   501	
   502	(7)
   503	
   504	p=1 c=1 k=1 i=1,i6=c
   505	
   506	where the first three summation items represent traversing
   507	all samples obtained by the SBS strategy, and each sample
   508	is regarded as an anchor. The last summation item aims to
   509	calculate the distance between each sample and its proxy of instance samples with the same person but different sub proxies.
   510	Considering that the mined sub proxies act as substitutes for
   511	clothing labels. Under the constraint of SPL, the intra-person
   512	variance caused by clothes transformation will be reduced.
   513	Sub-to-Main Proxy Learning. To achieve the collaboration
   514	between the main proxies and sub-proxies, we further propose a novel component called Sub-to-Main Proxy Learning
   515	(S2MPL) to effectively address the inter-person diversity and
   516	intra-person compactness. Specifically, as shown in the yellow
   517	part of Fig. 1 (c), each sub proxy δi,c is treated as an anchor.
   518	S2MPL pulls it towards the corresponding main proxy ρi , and
   519	pushes it away from the others. The S2MPL is formulated as:
   520	P
   521	
   522	LS2M P L = −
   523	
   524	C
   525	
   526	exp(d(δi,c , ρi )/τ )
   527	1 XX
   528	log PP
   529	,
   530	P C i=1 c=1
   531	p=1 exp(d(δi,c , ρp )/τ )
   532	
   533	(8)
   534	
   535	where d(, ) is the cosine distance and τ is a temperature
   536	factor. It should be emphasized that, to ensure a reasonable
   537	tolerance level of intra-person variances, we do not impose
   538	strict constraints on the similarity between instances in the subproxy. Similarly, the augmented sample x̂ki,c generated from
   539	the original sample xki,c is also not strictly enforced to be the
   540	same as xki,c . The overall HPL loss is given by the combination
   541	of all the above losses:
   542	LHP L = LSP L + LM P L + LS2M P L .
   543	
   544	(9)
   545	
   546	C. Joint Training
   547	In this paper, we also apply the cross-entropy loss LID on
   548	the feature vectors to train the baseline network. Thus, the
   549	total loss function Ltotal can be formulated as:
   550	Ltotal = LID + λLHP L ,
   551	
   552	(10)
   553	
   554	where λ is a balanced parameter.
   555	III. E XPERIMENTS

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext 'GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf' - | nl -ba | sed -n '480,565p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   480	97.71%
   481	99.72%
   482	
   483	99.51%
   484	99.86%
   485	
   486	81.30%
   487	94.29%
   488	
   489	Table 2. Ablation Study results on BRT dataset.
   490	GstRM
   491	
   492	GGM
   493	
   494	Rank-1
   495	
   496	mAP
   497	
   498	×
   499	✓
   500	×
   501	✓
   502	
   503	×
   504	×
   505	✓
   506	✓
   507	
   508	46.49%
   509	58.60%
   510	64.71%
   511	71.06%
   512	
   513	44.97%
   514	55.31%
   515	61.67%
   516	68.81%
   517	
   518	(a) Baseline(Graph-MLP) (b) Baseline+GstRM
   519	
   520	(C) GstNet
   521	
   522	(d) Identities
   523	
   524	Fig. 3. Visualization of the feature distributions with t-SNE. Different colors represent different identities.
   525	Rank-1, Rank-5, Rank-10 accuracy, and mean Average Precision
   526	(mAP). Our model is implemented in PyTorch and trained on an
   527	NVIDIA RTX 4090 GPU. Group-level features are extracted with
   528	a pretrained ViT-B/16, and all images are resized to 224 × 224.
   529	Training is conducted for 1,000 epochs with Adam (lr=0.001, weight
   530	decay=1 × 10−4 ). In GstRM, we set vmax = 27 km/h, gating range
   531	[0.7, 1.3], penalty κ = 6, and smoothing coefficient β = 2. The loss
   532	combines cross-entropy and neighborhood contrastive terms with
   533	τ = 2 and a balance λ = 0.2.
   534	
   535	mAP, showing that gated propagation alleviates neighborhood confusion. Combining both modules achieves the best performance of
   536	71.06% Rank-1 and 68.81% mAP, demonstrating their complementarity. Since the SYSU-Group dataset provides only temporal annotations without inter-camera distance information, a complete ablation of GstRM is infeasible; therefore, we only report the baseline
   537	and GstNet results in Table 1.
   538	
   539	3.2. Comparison with State-of-the-Art Methods
   540	
   541	3.4. Visualization
   542	
   543	In this study, we evaluate GstNet against representative baselines
   544	spanning group-level context modeling (MACG [5]), classical graph
   545	neural networks (GCN [24], GAT [25]), static distribution of statistics across time-interval cameras (St-ReID [13], InSTD [14]), and
   546	uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
   547	As shown, GstNet achieves superior performance on both datasets,
   548	with Rank-1 scores of 71.06% and 96.11%, surpassing the secondbest methods by 1.89% and 0.68%, respectively. Compared with
   549	approaches that rely on static distribution of statistics across timeinterval cameras (e.g., St-ReID [13] and InSTD [14]), the proposed GstRM and GGM modules enable GstNet to model dynamic
   550	geospatial-temporal reachability more effectively, thereby improving geospatial-temporal group person re-identification performance.
   551	
   552	To evaluate the discriminative power of GstNet’s features, we visualize the feature distributions of the Baseline (Graph-MLP), Baseline+GstRM, and GstNet using t-SNE, as shown in Fig. 3. Different
   553	colors denote different group identities. Compared with the other
   554	variants, GstNet achieves tighter intra-class clustering and clearer
   555	inter-class separation, reflecting superior feature separability. While
   556	the baseline (a) exhibits substantial overlap and Baseline+GstRM
   557	(b) shows only partial separation, GstNet (c) clusters samples of the
   558	same identity and separates different identities more distinctly.
   559	
   560	3.3. Ablation Study
   561	We conducted ablation studies on the BRT dataset, and the results
   562	are presented in Table 2. The baseline achieves only 46.49% Rank1 and 44.97% mAP. With GstRM added, the performance improves
   563	to 58.60% Rank-1 and 55.31% mAP, indicating that the geospatialtemporal reachability constraint suppresses spurious edges. With
   564	GGM added, the results increase to 64.71% Rank-1 and 61.67%
   565	

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' - | nl -ba | sed -n '629,790p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   629	In this section, we conduct several analytical experiments
   630	to better understand our proposed method.
   631	1) Comparison With Baselines: Our approach is the first
   632	purely US method in the field of person search. Due to the
   633	absence of directly comparable US methods, we establish
   634	several baselines that leverage pseudo-labels generated by offthe-shelf methods for training, as follows.
   635	1) FeatComp+DBSCAN: We utilize FeatComp [16],
   636	a state-of-the-art pedestrian detector [illustrated in
   637	Fig. 1(c)], to generate pseudo-boxes, and employ the
   638	
   639	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.
   640	
   641	This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
   642	TIAN et al.: HARNESSING KNOWLEDGE FROM PRETRAINED VLMs FOR US PERSON SEARCH
   643	
   644	7
   645	
   646	TABLE I
   647	S TATISTICS OF THE T RAINING S ETS FOR THE T WO DATASETS
   648	
   649	TABLE II
   650	P ERFORMANCE OF O UR M ETHOD AND D IFFERENT
   651	BASELINES ON CUHK-SYSU DATASET
   652	
   653	Fig. 6. Comparison with different baselines on PRW dataset. The points on
   654	the blue line represent the mAP values of different methods, while the bar
   655	chart illustrates the Top-k performance of these methods, with k set to 1, 5,
   656	and 10.
   657	TABLE III
   658	I MPACT OF L OSSES IN A NTI -B BOX -N OISE R E -ID L OSS
   659	
   660	DBSCAN clustering algorithm [52], a well-established
   661	method, to produce pseudo-IDs. In the following sections, this method will be referred to as the BASELINE.
   662	2) SEEM+DBSCAN: The masks and the corresponding
   663	pseudo-boxes for pedestrians are provided by a frozen
   664	interactive segmentation model called SEEM. We use
   665	these pseudo-boxes and masks, along with the pseudoIDs extracted by DBSCAN, to train a person search
   666	model.
   667	3) SEEM+SPCL: SPCL [53] is a conventional clustering method that has been demonstrated to outperform
   668	certain clustering techniques, such as DBSCAN. It is
   669	widely used in recent WS person search methods [9],
   670	[10]. Therefore, based on SEEM+DBSCAN, we replace
   671	DBSCAN with SPCL while keeping others unchanged.
   672	We compare our FMUPS-S1 method with various baselines
   673	on the CUHK-SYSU dataset. As shown in Table II, our
   674	FMUPS-S1 outperforms all other baselines, achieving 71.68
   675	pp with respect toand 73.24 pp with respect toTop-1. This
   676	highlights the effectiveness of our proposed approach. In addition, SEEM+DBSCAN shows a significant improvement of
   677	11.59 pp in mAP compared to the BASELINE. Furthermore,
   678	our FMUPS-S1 (SEEM+CLIP ID labeler) method surpasses
   679	SEEM+SPCL by a substantial margin of 25.89 pp in mAP.
   680	These results indicate that VLMs, such as SEEM and CLIP,
   681	effectively enhance the person search task. Notably, despite
   682	utilizing VLMs, our framework achieves the shortest training
   683	time and the least RAM among all methods. This efficiency
   684	is attributed to the CLIP ID labeler, which dynamically
   685	generates and denoises pseudo-IDs during training. In contrast, clustering methods like SPCL and DBSCAN require
   686	additional computation to extract and cluster features before
   687	each epoch.
   688	Similarly, the performance on the PRW dataset, illustrated in
   689	Fig. 6, shows that both our one-stage method, FMUPS-S1, and
   690	two-stage method, FMUPS-S2, outperform all baselines. The
   691	performance improves steadily as the k of Top-k increases.
   692	These findings emphasize the robustness of our framework
   693	across multiple datasets and evaluation metrics.
   694	
   695	2) Ablation Study on Anti-Bbox-Noise Re-ID Loss: The
   696	anti-box-noise re-ID loss, denoted as Lreid , comprises two
   697	components: the weighted foreground focusing loss in the
   698	instance stream, Lw f f in , and the weighted foreground focusing loss in the scene stream, Lw f f se . We provide empirical
   699	evidence of their impacts on the PRW dataset in Table III as
   700	follows.
   701	1) Training FMUPS-S1 solely with Lw f f in results in
   702	significantly inferior performance, with 3.06 pp with
   703	respect tomAP and 23.95 pp with respect toTop-1. This
   704	underperformance is due to the fact that Lw f f in only
   705	works on the instance stream and therefore does not
   706	optimize the re-ID task within the scene stream.
   707	2) Incorporating Lw f f se into the training of FMUPS-S1
   708	significantly enhances the results, achieving 21.56 pp
   709	with respect tomAP and 74.72 pp with respect toTop-1.
   710	This demonstrates the effectiveness of Lw f f se . Moreover, the margin of ∼1 pp (from 20.74 pp in row 2 to
   711	21.56 pp in row 3) underscores the positive contribution
   712	of Lw f f in to our method. CUHK-SYSU shows similar
   713	results.
   714	3) Effect of Different Number of Attributes in Text Prompt
   715	on Performance: To further analyze the role of textual
   716	attributes, we group the nine annotated attributes into four
   717	semantic categories: 1) colors {upper clothing color, lower
   718	clothing color}; 2) garment details {sleeve length, lower clothing length, lower clothing type}; 3) accessories {hat presence,
   719	bag presence and type}; and 4) personal characteristics {gender,
   720	hair length}. We then examine the impact of excluding each
   721	group on FMUPS-S1 performance. As shown in Table IV,
   722	removing any group reduces performance, with the largest
   723	drop occurring when clothing colors are removed, indicating
   724	that color cues are particularly important for distinguishing
   725	identities.
   726	4) Performance of Generalization to Different Datasets:
   727	In addition to PRW and CUHK-SYSU, we further evaluate
   728	
   729	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.
   730	
   731	This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
   732	8
   733	
   734	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
   735	
   736	TABLE IV
   737	
   738	TABLE VI
   739	
   740	E FFECT OF D IFFERENT G ROUPS OF P ERSON ATTRIBUTES ON
   741	FMUPS-S1 P ERFORMANCE ON CUHK-SYSU DATASET
   742	
   743	C OMPARISON OF M ETHODS FOR O BTAINING P SEUDO -ID S
   744	
   745	TABLE V
   746	
   747	C OMPARISON OF C OMPUTATIONAL R ESOURCE AND I NFERENCE S PEED
   748	B ETWEEN FMUPS-S1 AND FMUPS-S2 ON CUHK-SYSU DATASET
   749	
   750	TABLE VII
   751	P ERFORMANCE OF G ENERALIZATION TO D IFFERENT DATASETS . A→B: A
   752	R EFERS TO THE P ERFORMANCE OF SEEM+SPCL [53], W HILE B I S
   753	T HAT OF O UR FMUPS-S1 (SEEM+CLIP ID L ABELER )
   754	
   755	TABLE VIII
   756	P ERFORMANCE OF FMUPS-S1 ON CUHK-SYSU DATASET
   757	U NDER D IFFERENT C ROWD D ENSITIES
   758	
   759	Fig. 7. Performance comparison of FMUPS-S1 under different settings on
   760	the PRW dataset. (a) Impact of incorporating confidence scores (w/ ci versus
   761	w/o ci ) on mAP and Top-1 accuracy. (b) Compares the performance with and
   762	without prior knowledge (w/ prior versus w/o prior) of the CLIP ID labeler.
   763	The bars represent mAP, while the lines represent Top-1 accuracy.
   764	
   765	FMUPS-S1 on a larger and more diverse dataset, PoseTrack21 [54], which includes occlusion cases where queries
   766	may contain multiple individuals. As shown in Table V,
   767	our method consistently surpasses the SEEM+SPCL baseline,
   768	achieving 5.16, 25.89, and 13.02 pp gains in mAP on PRW,
   769	CUHK-SYSU, and PoseTrack21, respectively, along with clear
   770	improvements in Top-1 accuracy. These results demonstrate
   771	the robustness and generalization ability of FMUPS-S1 across
   772	diverse scenarios.
   773	5) Impact of Confidence Scores: Confidence scores are
   774	employed to dynamically mitigate the classification noise
   775	introduced by pseudo-boxes. As shown in Fig. 7(a), the performance without confidence scores is 20.97 pp with respect
   776	tomAP, which is a drop of approximately 0.6 pp compared
   777	to configurations that include confidence scores. This demonstrates that confidence scores are beneficial for bounding box
   778	denoising and underscores the effective leveraging of highconfidence predictions for better model performance.
   779	6) Impact of Prior in CLIP ID Labeler: The integration of
   780	a prior in the CLIP ID labeler is designed to denoise pseudoIDs and enhance their reliability. As illustrated in Fig. 7(b),
   781	incorporating prior knowledge significantly improves the performance, with the mAP increasing from approximately 20.65
   782	
   783	to 21.56 pp and Top-1 accuracy rising from 72.94 to 74.72 pp.
   784	The observed margin of approximately 1 pp with respect
   785	tomAP highlights the prior’s effectiveness in refining pseudoIDs and reducing ambiguities, further validating its critical role
   786	in enhancing the overall framework.
   787	7) Impact of CLIP ID Labeler: The CLIP ID labeler plays
   788	a pivotal role in generating pseudo-IDs for training the re-ID
   789	subtask. To assess its impact, we substitute the CLIP ID labeler
   790	with different clustering methods, while maintaining other

exec
/bin/zsh -lc "pdftotext 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | rg -n -i 'ablation|SPG|AHSTG|DFR|Distance|Visualization|heat|response|weight'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
6:     6	Distance-Aware Recovery Network for Occluded
19:    19	Distance-aware Recovery (HGTDR) network, which aims to
26:    26	(SPG), which combines the local attention mechanism to
31:    31	(AHSTG) to gradually generate strong-response foreground and
32:    32	weak-response background features. In addition, to complete
33:    33	the features of occluded regions, the Distance-based Feature
34:    34	Recovery (DFR) module is designed. The module calculates the
35:    35	planar Euclidean distance of heterogeneous tokens and adaptively
36:    36	allocates the corresponding weights to dynamically recover the
98:    98	lightweight network (PRE-Net), which constructs more robust
150:   150	into account the issue of neighbor weight for occlusion recovery, which leads to less robust recovered features. Alongside
165:   165	Generative Token and Distance-aware Recovery (HGTDR)
173:   173	a Semantic Patch Generator (SPG) is proposed, which uses
178:   178	on the semantic patches generated by SPG, we propose an
179:   179	Adaptive Heterogeneous Semantic Token Generator (AHSTG)
181:   181	global features of the holistic stream, AHSTG calculates
183:   183	and relevance between semantic patches generated by SPG.
184:   184	The scores are used to adaptively generate strong-response
185:   185	and weak-response heterogeneous tokens, which effectively
189:   189	a Distance-based Feature Recovery (DFR) module, which
191:   191	allocating adaptive weights to neighboring features. Finally,
196:   196	(1) A new Semantic Patch Generator (SPG) is proposed
200:   200	(2) We propose a flexible Adaptive Heterogeneous Semantic Token Generator (AHSTG), which takes into account
203:   203	generate heterogeneous tokens with different responses to
212:   212	proposed SPG, AHSTG, and DFR modules are in the part stream. Here, the patch embedding layer and shallow transformer blocks act as base module and
219:   219	(3) A novel Distance-based Feature Recovery (DFR) module is designed to automatically mine implicit information
222:   222	from the perspective of distance, and can adaptively recover
328:   328	introduced a lightweight Region Feature Completion (RFC)
345:   345	neighborhood weight on recovery quality.
348:   348	employs a dynamic weight allocation strategy based on
349:   349	distance perception. Our work reasonably allocates weights
350:   350	through the calculated distance, effectively simulates the real
352:   352	regions. Through the novel weight allocation mechanism, our
358:   358	Generative Tokens and Distance-aware Recovery (HGTDR)
383:   383	network’s robustness. We assess their distances and insert
414:   414	weight. Following the setting of TransReID [20], we set λc to
497:   497	Illustration of the proposed SPG. The black rectangular wire
503:   503	distance between a and b. m represents the similarity margin.
519:   519	Semantic Patch Generator (SPG), the Adaptive Heterogeneous
520:   520	Semantic Token Generator (AHSTG), and the Distance-based
521:   521	Feature Recovery module (DFR), as shown in Fig. 2. The SPG,
522:   522	AHSTG and DFR are illustrated in Fig. 3, Fig. 4 and Fig. 5,
529:   529	design a SPG module, which aims to mine deeper and
531:   531	subsequent feature aggregation and recovery. The SPG is
547:   547	Fig. 4. Illustration of the proposed AHSTG. Avgpooling represents the
552:   552	weighted and summed to give the comprehensive score s p . The red dots
558:   558	Fig. 5. Illustration of the proposed DFR. The index is from the AHSTG
560:   560	feature (i.e. CLS token) of the holistic stream. After the Euclidean distance
580:   580	weighted sum of image patches within each window.
670:   670	following AHSTG processing, we upsample the dimension of
682:   682	this limitation, we propose an AHSTG, as shown in Fig. 4.
685:   685	clutter and highlight the target person’s features. AHSTG
692:   692	the attention weight of different semantic patches, AHSTG
775:   775	response as the cluster centers, as shown in Fig. 4. Next,
798:   798	after AHSTG processing, we obtain fine-grained features
809:   809	3) Distance-Based Feature Recovery Module: Even though
810:   810	SPG and AHSTG modules can deal with local detail features
814:   814	from the lack of pedestrian information. To address this problem, we propose a Distance-based Feature Recovery (DFR)
823:   823	we introduce a distance factor into the DFR module to achieve
826:   826	the occluded region should be assigned higher weights, while
836:   836	The distance-based contribution consideration makes the
837:   837	DFR module more accurate and effective in recovering the
839:   839	into a 2D space to calculate the Euclidean distance between
840:   840	adjacent indexes, as shown in Fig. 5. If the distance is
850:   850	Ii+1 ). Furthermore, we calculate the planar Euclidean distance
852:   852	the contribution weights of heterogeneous tokens. The weights
853:   853	are denoted by Wt = {Wti |i = 1, 2, . . . , K }. For the ith weight, it can be represented by the following softmax
862:   862	where d(Ii , I r ) represents the planar Euclidean distance
863:   863	function between Ii and I r . Since the greater the distance,
864:   864	the smaller the weight should be, we apply the softmax
865:   865	function to the negative values of the distance. It ensures a
866:   866	reasonable distribution of the weights. Once we obtain the
870:   870	contribution weight Wt , we utilize it to weight and update the
880:   880	Following several transformer layers of the distance-based
901:   901	we introduce an innovative loss function, weight-based ID loss
907:   907	adjust the weight of a token in the loss function according to its
1115:  1115	processing technique (SPG), flexible fine-grained feature generation scheme (AHSTG), and adaptive distance-based feature
1116:  1116	recovery method (DFR). The application of these techniques
1185:  1185	D. Ablation Study
1187:  1187	components (SPG, AHSTG and DFR) in our HGTDR and
1189:  1189	we conduct comprehensive ablation experiments on the
1191:  1191	different settings of the ablation study along with the quantitative results. Following methods [10] and [45], the index-1
1195:  1195	1) Effectiveness of the SPG Module: As evidenced by
1202:  1202	network without SPG fails to attain optimal performance.
1203:  1203	By integrating the proposed SPG module onto index-5, the
1206:  1206	SPG can improve the ability to extract the local semantics.
1207:  1207	2) Effectiveness of the AHSTG Module: From index-1 and
1208:  1208	index-3, we can observe that the introduction of AHSTG
1210:  1210	Specifically, when a single AHSTG module is added to the
1212:  1212	and 3.5% on mAP. It demonstrates that our AHSTG module
1215:  1215	that the collaboration between the SPG and AHSTG modules further enhances the network’s ability to extract local
1223:  1223	AHSTG can fully utilize the local semantics processed by SPG
1225:  1225	3) Effectiveness of the DFR Module: From index-4 and
1228:  1228	DFR plays an important role in our HGTDR. Without the
1229:  1229	proposed DFR, the index-4 only achieves 68.2% Rank-1 and
1232:  1232	Considering the distance factor, we allocate a larger weight to
1234:  1234	weight to features that are farther away, thereby achieving
1236:  1236	the effectiveness of DFR is highly significant and it has
1240:  1240	in AHSTG, we also conduct thorough ablation studies on
1254:  1254	of layers in the AHSTG module N3 , the hyper-parameter
1324:  1324	stream, resulting in a weakening of the SPG of the part stream
1331:  1331	parameters, λ1 and λ2 , for controlling the weights of the
1366:  1366	G. Visualization
1373:  1373	noises. In contrast, the network after adding the SPG module
1374:  1374	(Index-2) can capture fine-grained semantic features. Furthermore, the network (Index-3) after introducing the AHSTG
1375:  1375	module can focus on the visible strong-response body regions,
1377:  1377	complete network integrated with the DFR module can focus
1379:  1379	response. Due to the introduction of our DFR module to
1382:  1382	relies on high-response human bodies and recovered person
1387:  1387	Fig. 9. Visualization of activation heatmaps of our proposed modules and
1395:  1395	lightweight design and high efficiency. To further explore the
1397:  1397	practicality, we conduct ablation experiments to analyze the
1405:  1405	In order to further reduce the model size, the DFR module is
1407:  1407	two, resulting in a lightweight version (denoted as HGTDR§).
1409:  1409	DFR merely causes a slight degradation in the performance
1412:  1412	In contrast, the lightweight version of HGTDR (HGTDR§)
1422:  1422	stream includes a Semantic Patch Generator (SPG), an Adaptive Heterogeneous Semantic Token Generator (AHSTG),
1423:  1423	and a Distance-based Feature Recovery module (DFR). The
1424:  1424	SPG extracts rich semantic information to generate semantic
1425:  1425	patches. The AHSTG considers the discrimination and relevance factors of semantic patches, and adaptively generates
1426:  1426	the strong-response body regions and the weak-response background regions. The DFR module leverages the information
1427:  1427	of heterogeneous tokens and calculates the distances between
1428:  1428	them. Based on the distances, we determine the presence
1429:  1429	of occlusion and assign the corresponding weight around
1594:  1594	probabilistic relative distance comparison,” in Proc. IEEE/CVF Conf.

exec
/bin/zsh -lc "pdftotext 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | rg -n -i 'ablation|FHFM|FSM|LFEM|feature hierarchical|feature separation|local feature|visual|heat|t-sne|attention'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
35:    35	received considerable attention. However, although existing methods have achieved promising results, most of
37:    37	suﬃciently emphasize the identity information in critical regions. Furthermore, those methods often lack suﬃcient capability to extract ﬁne-grained local features, making it diﬃcult to fully explore the identity information
39:    39	Local-aware Transformer (HFLAT) for occluded person ReID. Speciﬁcally, we ﬁrstly design a feature hierarchical
42:    42	We then design a feature separation module to distinguish foreground features from background features by employing patch-level saliency analysis, thereby mitigating the negative impact of backgrounds and occlusions on
43:    43	the performance of the model. In addition, we design a local feature extraction module which restricts the range
44:    44	of interactions between the features using a local-aware multi-head attention mechanism, increasing the model’s
45:    45	ability to obtain ﬁne-grained local features. Experimental results on the Occluded-DukeMTMC, Occluded-ReID,
70:    70	integrating various attention mechanisms to extract more discriminative features. Although these methods have achieved some encouraging
92:    92	of them possess a strong ability to capture global contextual information, their capacity for acquiring ﬁne-grained local features is largely
95:    95	First, we design a feature hierarchical fusion module (FHFM) to hierarchize and fuse the patch feature vector sequence based on their relative
96:    96	importance to the global feature vector, enhancing the identity discriminative features of key regions. Subsequently, we design a feature separation module (FSM) to isolate foreground and background features using patch-level saliency analysis, thereby mitigating the negative eﬀects
97:    97	of backgrounds and occlusions on the model’s performance. Additionally, we design a local feature extraction module (LFEM) that employs
98:    98	a local-aware multi-attention mechanism to limit feature interactions,
99:    99	improving the model’s ﬁne-grained local feature extraction capability.
103:   103	We design a local feature extraction module (LFEM) to enhance the
104:   104	ﬁne-grained local feature extraction capability of the model by limiting the range of interactions between the features using a local-aware
105:   105	multi-head attention mechanism.
120:   120	For instance, Miao et al. (2019) proposed a pose-guided feature alignment (PGFA) method to align local features using human key-point information. Gao et al. (2020) proposed a pose-guided visible part matching (PVPM) method that employs a pose-guided attention mechanism
125:   125	proposed a pose-guided partial-attention network with batch information (PPBI), which eﬀectively reduces the negative impact of occlusions
132:   132	• We design a feature hierarchical fusion module (FHFM) that hierarchizes and fuses the patch feature vectors based on their relative
134:   134	• We design a feature separation module (FSM) that separates foreground and background features using patch-level saliency analysis
147:   147	features using a local attention mechanism guided by human parsed labels. Dou et al. (2024) proposed a decouple re-identiﬁcation and human
153:   153	Transformer in global modelling, combined with the attention mechanisms to obtain more discriminative features. For instance, Li et al.
156:   156	the representation ability of local features by employing a pixel context encoder and a local prototype decoder. Lin et al. (2024) proposed a multi-level relation-aware Transformer (MLRAT), which has
161:   161	with a global attention mechanism, improving their model’s robustness in complex contexts. Wang et al. (2024) proposed a feature completion Transformer (FCFormer), which employs an occlusion instance
164:   164	model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
165:   165	critical pedestrian features using an uncertainty-aware self-attention
175:   175	do not perform ﬁne-grained local feature extraction, consequently failing to adequately mine the local information of pedestrians.they neither adequately consider the diﬀerence in the contribution of diﬀerent image patches to identity recognition, nor suﬃciently emphasize
177:   177	ﬁne-grained local feature extraction capabilities are insuﬃcient to fully
185:   185	contracts. Abd El-Hafeez et al. (2025) proposed a novel multi-scale attention model for the classiﬁcation of breast cancer histopathological
187:   187	learning framework improved upon YOLOv11 for the multi-class classiﬁcation task of cervical cancer cells, which enhances the model’s classiﬁcation accuracy via an Attention-Guided Multi-Scale Feature Fusion
189:   189	with an Attention-Guided Multi-Scale Feature Fusion (AGMS-FF) module, baseline YOLOv11, ResNet18, EﬃcientNet-B0, and MobileNetV3.
191:   191	vision systems in sustainable rice cultivation. Hassan et al. (2025b) proposed a novel DenseNet model integrated with attention mechanisms
194:   194	proposed an improved YOLOv11 architecture for the automated classiﬁcation of peripheral blood cells, which integrates a Dynamic CrossScale Context Aggregation (DCSCA) module. Through parallel convolution, dynamic attention, and cross-scale interaction, the module enables
197:   197	blood cells. Hassan et al. (2025d) proposed a novel network architecture named Sign Neuro-Enhanced Dense Attention Network (SNDA),
200:   200	2025. They pointed out that the application of attention modules and
205:   205	a network architecture based on DenseNet121 and the Squeeze-andExcitation (SE) attention module for fruit image classiﬁcation, which
207:   207	attention mechanisms, Nadam optimization, and other data augmentation methods. Eliwa and Abd El-Hafeez (2025) proposed a plant disease
222:   222	then details the feature hierarchical fusion module (FHFM), the feature separation module (FSM), and the local feature extraction module (LFEM). Finally, the loss functions employed for model training are
230:   230	Fig. 2. Work ﬂowchart of HFLAT. FHFM denotes the feature hierarchical fusion module, FSM denotes the feature separation module, and LFEM denotes the local
235:   235	3.2. Feature hierarchical fusion module
249:   249	We feed 𝑍0 into 𝐿 Transformer layers to get an output feature sequence 𝑍𝐿 = [𝑧𝑐𝑙𝑠 ; 𝑧1 ; 𝑧2 ; … ; 𝑧𝑁 ] ∈ ℝ(𝑁+1)×𝐷 and an attention weight
254:   254	each patch to the global feature vector, thus obtaining the fused feature vector 𝑓𝑢 . Meanwhile, we feed 𝑍𝑝𝑎𝑡𝑐ℎ into the feature separation
259:   259	into the local feature extraction module, in which we insert four learnable local tokens into 𝑍𝑝𝑎𝑡𝑐ℎ and use the local-aware multi-head attention to limit the range of feature interactions, extracting four local feature vectors {𝑓𝑖𝐿 }4𝑖=1 . Lastly, inspired by Sun et al. (2018), Miao et al.
261:   261	and the local feature vectors {𝑓𝑖𝐿 }4𝑖=1 . The concatenated feature vector is then processed through a fully connected layer for information
268:   268	we design a feature hierarchical fusion module, as illustrated in Fig. 3,
272:   272	identity discriminative features of key regions. That is to say, FHFM directs greater attention towards key regions, thereby enhancing the identity information they contain by dynamically stratifying image patch
276:   276	[𝑧1 ; 𝑧2 ; ⋯ ; 𝑧𝑁 ] and an attention weight matrix 𝑀𝐿 ∈ ℝ(1+𝑁)×(1+𝑁) , we
287:   287	where 𝑀𝑐𝑙𝑠,𝑖 denotes the attention weight of the 𝑖-th patch relative
334:   334	Fig. 3. Work ﬂowchart of feature hierarchical fusion module.
336:   336	Fig. 4. Work ﬂowchart of feature separation module.
346:   346	3.3. Feature separation module
358:   358	of images, as illustrated in Fig. 4, we design a feature separation module. This module employs a patch-level saliency analysis to calculate the
408:   408	Fig. 5. Work ﬂowcharts of (a) Local feature extraction module, (b) Local-aware multi-head attention and (c) Local-aware attention.
410:   410	For each attention head, we ﬁrst linearly transform 𝑍 ′ into the query
425:   425	3.4. Local feature extraction module
428:   428	often contains shoe information. To obtain ﬁne-grained local features
429:   429	in diﬀerent parts of a pedestrian image, we design a local feature extraction module, as shown in Fig. 5(a). In particular, we also design
430:   430	and use a local-aware multi-head attention in this module, as shown in
431:   431	Fig. 5(b), where each attention head is a local-aware attention, as shown
432:   432	in Fig. 5(c). Diﬀerent from the conventional multi-head self-attention,
433:   433	our local-aware multi-head attention achieves an eﬀective interaction of
436:   436	ﬁne-grained local features.
438:   438	[𝑧1 ; 𝑧2 ; ⋯ ; 𝑧𝑁 ] ∈ ℝ𝑁×𝐷 , we ﬁrst append four learnable local tokens to its end to capture local features of the top-to-bottom four
441:   441	approach enables our model to learn discriminative local features
446:   446	we process 𝑍 ′ using a local-aware multi-head attention mechanism.
469:   469	wherein the discriminative ability of ﬁne-grained local features (such as
480:   480	Further, we calculate the output 𝑍̃ ∈ ℝ(𝑁+4)×𝐷 of the attention head
530:   530	Furthermore, we concatenate the outputs of all the attention heads
533:   533	multi-head attention 𝑍̌ ∈ ℝ(𝑁+4)×𝐷 . Then, as shown in Fig. 5(a), we
558:   558	[𝑧̂ 1 ; 𝑧̂ 2 ; ⋯ ; 𝑧̂ 𝑁 ; 𝑧̂ 𝑝1 ; 𝑧̂ 𝑝2 ; 𝑧̂ 𝑝3 ; 𝑧̂ 𝑝4 ] and use them as the ﬁnal local feature
640:   640	vector 𝑓𝑓 𝑔 and background feature vector 𝑓𝑏𝑔 output by the feature separation module should be signiﬁcantly diﬀerent. To ensure a high degree
722:   722	Number of levels in patch sequence segmentation in FHFM
1040:  1040	the margin for improvement in Rank-1 accuracy will only become increasingly smaller. Second, the three core modules of our model-FHFM,
1041:  1041	FSM, and LFEM-are designed to enhance identity information in critical regions, suppress occlusion interference, and extract ﬁne-grained
1067:  1067	85.8%, 95.1%, and 88.4%, respectively. This indicates that FHFM enhances the discriminability of pedestrian features by hierarchically fusing image patch feature vectors.
1068:  1068	Eﬀectiveness of the Feature Separation Module (FSM): As
1070:  1070	DukeMTMC-ReID dataset, compared to the Baseline, the mAP of Baseline + FSM improves from 48.8%, 77.8%, and 75.9% to 52.1%, 78.7%,
1074:  1074	FSM improves from 87.7% to 88.0%, and its Rank-1 accuracy remains
1075:  1075	unchanged. This demonstrates that FSM eﬀectively reduces the interference of backgrounds and occlusions by separating foreground features
1077:  1077	Eﬀectiveness of the Local Feature Extraction Module (LFEM): As
1079:  1079	+ LFEM achieves improvements in the mAP from 48.8%, 77.8%, 87.7%,
1082:  1082	73.0%, 83.5%, 95.0%, and 87.7%, respectively. This suggests that LFEM
1085:  1085	Occluded-DukeMTMC dataset, Baseline + FHFM + FSM achieves the
1086:  1086	mAP and Rank-1 accuracy of 61.6% and 76.7%, respectively, signiﬁcantly outperforming Baseline + FHFM and Baseline + FSM; Baseline
1087:  1087	+ FHFM + LFEM achieves the mAP and Rank-1 accuracy of 62.5%
1088:  1088	and 76.8%, respectively, markedly outperforming Baseline + FHFM
1089:  1089	and Baseline + LFEM; Baseline + FSM + LFEM achieves the mAP
1090:  1090	and Rank-1 accuracy of 59.2% and 74.5%, respectively, markedly outperforming Baseline + FSM and Baseline + LFEM. On the OccludedReID dataset, Baseline + FHFM + FSM achieves the mAP and Rank-1
1092:  1092	Baseline + FHFM and Baseline + FSM; Baseline + FHFM + LFEM
1093:  1093	achieves the mAP and Rank-1 accuracy of 83.9% and 88.5%, respectively, signiﬁcantly outperforming Baseline + FHFM and Baseline +
1094:  1094	LFEM; Baseline + FSM + LFEM achieves the mAP and Rank-1 accuracy of 82.4% and 86.7%, respectively, markedly outperforming Baseline + FSM and Baseline + LFEM. On the Market-1501 dataset, Baseline + FHFM + FSM achieves the mAP and Rank-1 accuracy of 89.3%
1095:  1095	and 95.2%, respectively, markedly outperforming Baseline + FHFM
1096:  1096	and Baseline + FSM; Baseline + FHFM + LFEM achieves the mAP
1098:  1098	outperforming Baseline + FHFM and Baseline + LFEM; Baseline +
1099:  1099	FSM + LFEM achieves the mAP and Rank-1 accuracy of 89.9% and
1100:  1100	95.3%, respectively, markedly outperforming Baseline + FSM and Baseline + LFEM. On the DukeMTMC-ReID dataset, Baseline + FHFM +
1101:  1101	FSM achieves the mAP and Rank-1 accuracy of 81.3% and 90.5%, respectively, markedly outperforming Baseline + FHFM and Baseline +
1102:  1102	FSM; Baseline + FHFM + LFEM achieves the mAP and Rank-1 accuracy
1104:  1104	+ FHFM and Baseline + LFEM; Baseline + FSM + LFEM achieves the
1106:  1106	outperforming Baseline + FSM and Baseline + LFEM. These experimental results demonstrate that every pairwise combination of the three
1108:  1108	Baseline + FHFM + FSM + LFEM achieves the best performance
1110:  1110	On the Occluded-ReID dataset, Baseline + FHFM + FSM + LFEM
1113:  1113	+ FHFM + FSM + LFEM also achieves the best performance with the
1116:  1116	4.4. Ablation studies and hyperparameter analysis
1117:  1117	4.4.1. Ablation studies
1119:  1119	performance, we perform ablation studies on the Occluded-DukeMTMC,
1124:  1124	construct ﬁve models: Baseline + FHFM, Baseline + FSM, Baseline +
1125:  1125	LFEM, Baseline + FHFM + FSM, and Baseline + FHFM + FSM +
1126:  1126	LFEM, where, FHFM, FSM, and LFEM represent the feature hierarchy
1127:  1127	fusion module, the feature separation module, and the local feature extraction module, respectively. Table 5 presents the outcomes of the ablation experiments. In particular, the results from the original TransReID
1129:  1129	Eﬀectiveness of the Feature Hierarchy Fusion Module (FHFM):
1132:  1132	the mAP of Baseline + FHFM improves from 48.8%, 77.8%, 87.7%, and
1140:  1140	Results of ablation experiments (including mAP (%) and Rank-1 accuracy (%)) on the Occluded-DukeMTMC,
1145:  1145	Baseline + FHFM
1146:  1146	Baseline + FSM
1147:  1147	Baseline + LFEM
1148:  1148	Baseline + FHFM + FSM
1149:  1149	Baseline + FHFM + LFEM
1150:  1150	Baseline + FSM + LFEM
1151:  1151	Baseline + FHFM + FSM + LFEM
1251:  1251	number of levels in patch sequence segmentation in FHFM on the OccludedDukeMTMC, and Occluded-ReID datasets.
1318:  1318	the number of levels in patch sequence segmentation in FHFM on the
1336:  1336	Occluded-DukeMTMC dataset, Baseline + FHFM + FSM + LFEM
1404:  1404	shoes. These elements constitute the key semantic information for person re-identiﬁcation. If the number of levels is less than 4, FHFM will inevitably force the merging of adjacent semantic regions (such as the upper body with the lower body, or the lower body with the feet), thereby
1405:  1405	failing to accurately capture the distinctive features of diﬀerent key regions. If the number of levels exceeds 4, FHFM will inevitably split a
1411:  1411	To validate the impact of the number of levels in patch sequence segmentation in FHFM on the performance of our model, we conduct ablation experiments on the Occluded-DukeMTMC, Occluded-ReID, Market1501, and DukeMTMC-ReID datasets. Speciﬁcally, we establish ﬁve distinct patch sequence segmentation approaches, i.e., no segmentation,
1560:  1560	focuses on the feature separation. To achieve accurate pedestrian reidentiﬁcation, our model not only employs the ID loss and the triplet
1568:  1568	model to excessively focus on the feature separation, thereby suppressing FHFM’s ability to extract identity information from key regions and LFEM’s ability to capture ﬁne-grained local details. When
1570:  1570	insuﬃcient weight, preventing FSM from eﬀectively separating foreground features and background features. Consequently, the foreground
1571:  1571	features output by FSM contain excessive background interference,
1599:  1599	Fig. 7. Feature visualization of the three modules.
1601:  1601	Fig. 8. Visualization of retrieval results.
1603:  1603	4.6. Visualization of results
1606:  1606	To visualize the eﬀectiveness of HFLAT more clearly, we present the
1614:  1614	To demonstrate the eﬀectiveness of HFLAT clearly, we visualize the
1615:  1615	features of FHFM, FSM, and LFEM in Fig. 7. As shown in Fig. 7, FSM
1617:  1617	FHFM can focus well on key regions, and LFEM can capture ﬁne-grained
1618:  1618	local features eﬀectively. Nevertheless, it is also evident from Fig. 7 that,
1641:  1641	Cui, J., Chen, Y., Deng, B., Liu, G., Wang, Z., & Li, Y. (2025). PPBI: Pose-guided partialattention network with batch information for occluded person re-identiﬁcation. Sensors, 25(3), 757.
1669:  1669	Densenet model with attention mechanisms for robust date fruit image classiﬁcation. International Journal of Computational Intelligence Systems, 18(1), 228–261. https:
1672:  1672	Sustainable deep vision systems for date fruit quality assessment using attentionenhanced deep learning models. Frontiers in Plant Science, 16, 1521508–1521526.
1684:  1684	Jia, M., Sun, Y., Zhai, Y., Cheng, X., Yang, Y., & Li, Y. (2023). Semi-attention partition for
1706:  1706	feature hierarchical fusion module to hierarchize and fuse the image
1708:  1708	features of key regions. Then, we use a feature separation module to
1710:  1710	negative impact of backgrounds and occlusions on the model’s performance. Additionally, we use a local feature extraction module to capture the ﬁne-grained local features from diﬀerent regions of pedestrian
1721:  1721	although the ablation experimental results on the four datasets demonstrate the eﬀectiveness of LFEM, its equal height-based image division
1723:  1723	(e.g., sitting or crouching). We will design an adaptive pedestrian image segmentation strategy based on pose-aware attention mechanisms to
1731:  1731	Visualization, Validation.

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
/bin/zsh -lc "pdftotext 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '1366,1387p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext 'Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '1185,1240p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1366	G. Visualization
  1367	To further demonstrate the effectiveness of our proposed
  1368	modules and HGTDR network, we utilize Grad-CAM [62] to
  1369	ablate the effectiveness of our proposed modules and visualize
  1370	the experimental results. As shown in Fig. 9, it can be seen that
  1371	the baseline (Index-1) has a global receptive field and is unable
  1372	to focus on the pedestrians, thus introducing background
  1373	noises. In contrast, the network after adding the SPG module
  1374	(Index-2) can capture fine-grained semantic features. Furthermore, the network (Index-3) after introducing the AHSTG
  1375	module can focus on the visible strong-response body regions,
  1376	while giving little attention to obstacles. At the last, the
  1377	complete network integrated with the DFR module can focus
  1378	more on the visible human bodies and suppress the occlusion
  1379	response. Due to the introduction of our DFR module to
  1380	recover invisible areas, it is inevitable that some attention is
  1381	also paid to the obstacles near the pedestrians. However, compared with these regions, retrieving target pedestrians mainly
  1382	relies on high-response human bodies and recovered person
  1383	features. These visualized results demonstrate the effectiveness
  1384	of our proposed modules and HGTDR.
  1385	V. C ONCLUSION
  1386	
  1387	Fig. 9. Visualization of activation heatmaps of our proposed modules and

 succeeded in 0ms:
  1185	D. Ablation Study
  1186	In order to demonstrate the effectiveness of the three
  1187	components (SPG, AHSTG and DFR) in our HGTDR and
  1188	the two indicators (DIS and RIS) in the scoring network,
  1189	we conduct comprehensive ablation experiments on the
  1190	Occluded-DukeMTMC. As shown in Table IV, we show the
  1191	different settings of the ablation study along with the quantitative results. Following methods [10] and [45], the index-1
  1192	utilizes the ViT-B/16 [19] as our baseline, with overlapping
  1193	patches, and is optimized using both general cross-entropy and
  1194	triplet loss.
  1195	1) Effectiveness of the SPG Module: As evidenced by
  1196	index-1 and index-2, the performance is significantly improved
  1197	when the semantic patch generator is incorporated. The performance of index-2 shows an improvement of +2.2% in
  1198	Rank-1 accuracy and +2.1% in mAP compared to the baseline. It demonstrates that the introduction of spatial windows
  1199	enables the network to enhance the representation capacity of
  1200	patch features and achieve notable performance improvement.
  1201	Moreover, from index-5 and index-6, it is apparent that the
  1202	network without SPG fails to attain optimal performance.
  1203	By integrating the proposed SPG module onto index-5, the
  1204	model’s performance can be boosted by +2.1% and +1.5%
  1205	in Rank-1 accuracy and mAP, respectively. It shows that our
  1206	SPG can improve the ability to extract the local semantics.
  1207	2) Effectiveness of the AHSTG Module: From index-1 and
  1208	index-3, we can observe that the introduction of AHSTG
  1209	module can significantly improve the network’s performance.
  1210	Specifically, when a single AHSTG module is added to the
  1211	baseline, it outperforms the baseline by 4.4% on Rank-1
  1212	and 3.5% on mAP. It demonstrates that our AHSTG module
  1213	can effectively extract discriminative regions and reduce the
  1214	occlusion interference. From index-3 and index-4, we discover
  1215	that the collaboration between the SPG and AHSTG modules further enhances the network’s ability to extract local
  1216	semantics and adaptively resist noise and clutter. In particular,
  1217	
  1218	P ERFORMANCE A NALYSIS OF D IFFERENT C OMPREHENSIVE S CORES . (DIS
  1219	AND RIS R EPRESENT THE D ISCRIMINATION I NDICATOR S CORE AND
  1220	R ELEVANCE I NDICATOR S CORE , R ESPECTIVELY )
  1221	
  1222	index-4 achieves a performance increase of +5.7% on Rank1 and +6.0% on mAP compared to index-2. It indicates that
  1223	AHSTG can fully utilize the local semantics processed by SPG
  1224	to handle complex occlusion scenes.
  1225	3) Effectiveness of the DFR Module: From index-4 and
  1226	index-6, we can see that the fully equipped network HGTDR
  1227	can achieve optimal performance, which proves that effective
  1228	DFR plays an important role in our HGTDR. Without the
  1229	proposed DFR, the index-4 only achieves 68.2% Rank-1 and
  1230	58.9% mAP, whereas the holistic model (HGTDR) achieves
  1231	optimal performance with 71.8% Rank-1 and 62.3% mAP.
  1232	Considering the distance factor, we allocate a larger weight to
  1233	adjacent features closer to the occluded region, and a smaller
  1234	weight to features that are farther away, thereby achieving
  1235	adaptive dynamic recovery of the occlusion part. Additionally,
  1236	the effectiveness of DFR is highly significant and it has
  1237	practical implications for recovering occluded features.
  1238	4) Effectiveness of the two Indicator Scores: To further
  1239	validate the impact of the two indicators of the scoring network
  1240	in AHSTG, we also conduct thorough ablation studies on

exec
/bin/zsh -lc "pdftotext 'Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '628,895p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   628	T HE ABLATION STUDIES OF DIFFERENT COMPONENTS .
   629	Components
   630	Baseline
   631	+ SBS
   632	+ SCT
   633	+ HPL
   634	
   635	Components
   636	Method
   637	1
   638	2
   639	3
   640	4
   641	5
   642	6
   643	7
   644	8
   645	
   646	C OMPARISON WITH OTHER METHODS ON VC- CLOTHES .
   647	
   648	Clothes
   649	label
   650	
   651	PCB [31]
   652	ISP [35]
   653	FSAM [33]
   654	3DSL [26]
   655	BSGA [27]
   656	CAL [24]
   657	HPL(Ours)
   658	
   659	×
   660	×
   661	×
   662	X
   663	X
   664	X
   665	×
   666	
   667	VC-Clothes
   668	CC
   669	SC
   670	mAP Rank-1
   671	mAP
   672	Rank-1
   673	62.2
   674	94.3
   675	62.0
   676	94.7
   677	72.1
   678	94.7
   679	72.0
   680	94.5
   681	78.9
   682	94.8
   683	78.6
   684	94.7
   685	81.2
   686	79.9
   687	84.3
   688	94.4
   689	84.5
   690	94.9
   691	81.7
   692	81.4
   693	95.1
   694	95.3
   695	84.7
   696	95.3
   697	84.9
   698	95.3
   699	
   700	B. Implementation Details
   701	We use the ResNet-50 [36] pre-trained on ImageNet [37] as
   702	the feature encoder. For automatic sub-proxy mining, we use
   703	DBSCAN [28] for clustering, and set the minimum number of
   704	samples for each cluster to 4 and the scanning radius to 0.8.
   705	During training, we adopt random flipping, random cropping
   706	and random erasing [38] for data augmentation. The input
   707	images are resized to 384 × 192. For the SBS strategy, we
   708	sample P = 8 identities, C = 2 different sub proxies for each
   709	identity, and each sub proxy with K = 4 images. We train the
   710	framework for 60 epochs in total by the Adam [39] optimizer.
   711	The initial learning rate is 0.00035 and is divided by 10 at
   712	every 20 epochs. As for the hyper-parameter, α is set to be
   713	0.3, τ is set to be 1/16 and λ is set to be 0.1. We use the
   714	original features during testing, and the cosine similarity is
   715	employed as the distance metric for ranking.
   716	C. Comparison with State-of-the-art Methods
   717	We compared our method with various advanced CC-ReID
   718	methods on PRCC and VC-Clothes. Experimental results are
   719	reported in Tab. I and II. On the PRCC and VC-Clothes
   720	datasets, our method achieves the best results of 74.3% and
   721	84.9% in Rank-1 accuracy under the CC setting, respectively.
   722	RCSANet [22] utilizes a clothes-proxy loss that encourages
   723	more consistency when a person wears the same clothes, and
   724	pulls away different clothes. In contrast, the purpose of our
   725	HPL loss is to pull the samples of different clothes with
   726	the same identity together, which is similar to CAL [24].
   727	Moreover, unlike CAL [24], BSGA [27] and 3DSL [26], the
   728	proposed HPL don’t use any clothes information and achieves
   729	better performance with more limited information. And all the
   730	above methods ignore the unbalanced distribution of training
   731	
   732	VC-Clothes
   733	CC
   734	SC
   735	mAP Rank-1
   736	mAP
   737	Rank-1
   738	74.0
   739	94.8
   740	73.9
   741	94.7
   742	75.8
   743	94.7
   744	75.7
   745	94.9
   746	77.6
   747	94.7
   748	77.9
   749	94.7
   750	84.7
   751	95.3
   752	84.9
   753	95.3
   754	
   755	TABLE IV
   756	T HE ABLATION STUDIES OF HPL ON PRCC.
   757	
   758	TABLE II
   759	
   760	Method
   761	
   762	PRCC
   763	CC
   764	SC
   765	mAP Rank-1
   766	mAP
   767	Rank-1
   768	57.2
   769	99.6
   770	57.0
   771	100
   772	58.5
   773	99.6
   774	59.3
   775	100
   776	63.9
   777	98.6
   778	68.6
   779	100
   780	69.0
   781	98.5
   782	74.3
   783	99.8
   784	
   785	(a)
   786	
   787	(b)
   788	
   789	SPL
   790	
   791	MPL
   792	
   793	S2MPL
   794	
   795	×
   796	X
   797	×
   798	×
   799	X
   800	X
   801	×
   802	X
   803	
   804	×
   805	×
   806	X
   807	×
   808	×
   809	X
   810	X
   811	X
   812	
   813	×
   814	×
   815	×
   816	X
   817	X
   818	×
   819	X
   820	X
   821	
   822	(c)
   823	
   824	(a)
   825	
   826	PRCC
   827	CC
   828	SC
   829	Rank-1 mAP
   830	Rank-1 mAP
   831	68.6
   832	100
   833	63.9
   834	98.6
   835	72.4
   836	99.5
   837	67.7
   838	97.2
   839	71.3
   840	99.8
   841	67.5
   842	98.2
   843	71.5
   844	99.9
   845	67.4
   846	98.3
   847	73.5
   848	99.5
   849	68.4
   850	97.6
   851	73.9
   852	99.6
   853	68.6
   854	97.4
   855	72.7
   856	99.9
   857	67.9
   858	98.6
   859	74.3
   860	99.8
   861	69.0
   862	98.5
   863	(b)
   864	
   865	(c)
   866	
   867	(a)
   868	
   869	(b)
   870	
   871	(c)
   872	
   873	Fig. 3. The visualization of feature maps on PRCC. (a) original image; the
   874	feature map of the baseline method (b) and our proposed method (c).
   875	
   876	samples in a mini-batch. As a result, our method achieves
   877	84.9% Rank-1 accuracy on VC-Clothes under the CC setting,
   878	which surpasses CAL by 3.5%. These comparisons fully
   879	demonstrate the effectiveness and superiority of our method.
   880	D. Ablation Study
   881	To verify the impact of each component, we conduct several
   882	experiments, and show the results in Tab. III. It can be observed that the proposed SBS strategy is superior to the baseline. And models leveraging SCT also achieved remarkable
   883	performance improvements. The reason can be attributed to
   884	the enrichment of training examples with clothes-changing. In
   885	addition, the proposed HPL further improves the performance
   886	of CC-ReID. A reasonable explanation is that a hierarchical
   887	structure is contributed to extracting person-invariant and
   888	clothes-irrelevant features. What’s more, as shown in Tab. IV,
   889	we further verify the effectiveness of each component in HPL
   890	on PRCC. Fig. 3 also clearly shows that, with the help of HPL,
   891	the proposed method will pay more attention to non-clothing
   892	regions, e.g., head, face and arms.
   893	IV. C ONCLUSION
   894	In this paper, we propose a novel Hierarchical Proxy
   895	Learning (HPL) framework for CC-ReID. To get rid of the

exec
/bin/zsh -lc "pdftotext 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' - | nl -ba | sed -n '1251,1338p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1251	5.2. Ablation study (RQ2)
  1252	To fully demonstrate the impact of diﬀerent components in HKGR,
  1253	we conduct a series of ablation studies on the CUHK-PEDES dataset.
  1254	Speciﬁcally, we ﬁrst evaluate the eﬀectiveness of diﬀerent components
  1255	of our proposal. Then, comparisons between distinct alignment strategies are discussed.
  1256	Ablations on the diﬀerent components. The detailed results are
  1257	shown in Table 5. The No.0 shows the result of the Baseline model,
  1258	which is initialized with CLIP pre-trained weights and ﬁne-tuned with
  1259	InfoNCE loss (van den et al., 2018) without adding any components
  1260	or further feature embedding. (1)The eﬃcacy of KGR is demonstrated
  1261	through the ablated results comparison between No.0 vs No.3, No.2 vs
  1262	No.5, and No.4 vs No.8. The comparison between No.0 and No.3 reveals
  1263	that, compared to the Baseline, our exclusively adopted KGR module improves the R@1 accuracy by 0.55 %. Moreover, as shown in results No.5
  1264	and No.8, the combination of the KGR module can further achieve performance gains by 0.44 % and 0.93 %, respectively. These results justify that KGR can eﬀectively improve the cross-modal representation.
  1265	(2)Moreover, the ablated results comparison between No.0 vs No.2 and
  1266	No.3 vs No.5 reveals the eﬃcacy of the MGA module, which achieves
  1267	signiﬁcant performance with 3.70 % and 3.59 %, respectively. From the
  1268	comparison between No.6 and No.7, we can ﬁnd that combining the
  1269	coarse-grained and ﬁne-grained semantic alignment is more beneﬁcial
  1270	for improving retrieval performance than using them separately. (3)Besides, we also investigate the eﬀectiveness of ID loss. When adding ID
  1271	loss (Zheng et al., 2020) to our proposed model (No.5 vs. No.8), the R@1
  1272	accuracy further improved from 75.03 % to 75.21 %.
  1273	Analysis of hierarchical consistency alignment. As aforementioned, the KGR module incorporates objective-level reasoning,
  1274	attribute-level reasoning, and relation-level reasoning to establish ﬁnegrained connections between images and text. To further evaluate the
  1275	eﬀectiveness of our proposal, we compare it with four diﬀerent knowledge reasoning strategies, including MLM-based reasoning, object-level
  1276	reasoning, attribute-level reasoning, and relation-level reasoning. Note
  1277	that we follow the same masking strategy for MLM-based reasoning as
  1278	detailed in IRRA (Jiang & Ye, 2023). Results are presented in Table 6.
  1279	We can see that merely replacing random masking with a knowledge-
  1280	
  1281	Fig. 4. Parametric analysis of the hyper-parameters of 𝑅 and 𝜏 on CUHK-PEDES
  1282	dataset, reporting performance using the R@1 and mAP metrics.
  1283	
  1284	based prediction strategy (No.0 vs No.1, No.2, and No.3) can improve
  1285	performance. Furthermore, when we utilize the KGR module instead of
  1286	a separate knowledge-based prediction strategy (No.4 vs No.1, No.2, and
  1287	No.3), the model’s performance is further improved by 0.44 %, 0.52 %,
  1288	and 0.59 %, respectively.
  1289	5.3. Parametric sensitivity analysis (RQ3)
  1290	To answer RQ2, we further investigate the sensitivity of the hyperparameters 𝑅 and 𝜏 by analyzing their impact on model performance
  1291	across a range of values on the CUHK-PEDES dataset. 𝑅 is the token
  1292	selection ratio that controls the number of selected local tokens, as described in Section 3.4. 𝜏 is the temperature factor to regulates the difﬁculty of mining hard negatives in the contrastive loss, as deﬁned in
  1293	Eqs. (5) and (7). From Fig. 4 we can see that: (1) Selecting either an excessively high or low value for R can lead to suboptimal performance. A
  1294	relatively small 𝑅 may result in signiﬁcant information loss, yielding inadequate feature representations. In contrast, an too large 𝑅 is likely to
  1295	include numerous irrelevant or insigniﬁcant features, thereby diluting
  1296	the discriminative power of the learned representations. Thus, we set
  1297	the 𝑅 to 0.3 for all the experiments. (2) As the value of 𝜏 increases, the
  1298	retrieval performance ﬁrst increases and then decreases. The increase
  1299	of 𝜏 will gradually decrease the hardness of hard negative mining, leading to suboptimal performance. Therefore, we set 𝜏 to 0.01 in all our
  1300	experiments.
  1301	5.4. Qualitative results (RQ4)
  1302	We implement three qualitative experimental results, feature representation visualization, heatmap visualization, and retrieval visualization, to
  1303	visually illustrate the eﬀectiveness of our HKGR. Note that the Baseline model mentioned in this subsection refers to models initialized
  1304	with CLIP pre-trained weights and ﬁne-tuned using the original InfoNCE
  1305	loss (van den et al., 2018).
  1306	For feature representation visualization: To visually demonstrate the
  1307	eﬀectiveness of our proposal in semantic correspondence of image and
  1308	text, we use t-SNE to transform the high-dimensional image and text
  1309	feature representations from the common latent space into a 2D plane.
  1310	As shown in Fig. 5(a), it can be observed that the inter-modality feature
  1311	representations extracted from the pre-trained backbone of CLIP in the
  1312	
  1313	Table 6
  1314	Comparisons between four diﬀerent knowledge reasoning
  1315	strategies of our HKGR on the CUHK-PEDES dataset.
  1316	No.
  1317	
  1318	Method
  1319	
  1320	R@1
  1321	
  1322	R@5
  1323	
  1324	R@10
  1325	
  1326	0
  1327	1
  1328	2
  1329	3
  1330	4
  1331	
  1332	MLM-based reasoning
  1333	Object-level Reasoning
  1334	Attribute-level Reasoning
  1335	Relation-level Reasoning
  1336	Full HKGR
  1337	
  1338	75.03

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
exec
/bin/zsh -lc "pdftotext 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '1116,1165p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1116	4.4. Ablation studies and hyperparameter analysis
  1117	4.4.1. Ablation studies
  1118	To assess the contribution of each module in HFLAT to its overall
  1119	performance, we perform ablation studies on the Occluded-DukeMTMC,
  1120	Occlded-ReID, Market-1501, and DukeMTMC-ReID datasets. We use
  1121	TransReID (He et al., 2021) without the patch embedding module as
  1122	the baseline, and sequentially add our designed modules to the baseline
  1123	to assess their contribution to the model’s performance. Speciﬁcally, we
  1124	construct ﬁve models: Baseline + FHFM, Baseline + FSM, Baseline +
  1125	LFEM, Baseline + FHFM + FSM, and Baseline + FHFM + FSM +
  1126	LFEM, where, FHFM, FSM, and LFEM represent the feature hierarchy
  1127	fusion module, the feature separation module, and the local feature extraction module, respectively. Table 5 presents the outcomes of the ablation experiments. In particular, the results from the original TransReID
  1128	(He et al., 2021) are presented in Tables 3 and 4.
  1129	Eﬀectiveness of the Feature Hierarchy Fusion Module (FHFM):
  1130	As shown in Table 5, on the Occluded-DukeMTMC, Occlded-ReID,
  1131	Market-1501 and DukeMTMC-ReID datasets, compared to the Baseline,
  1132	the mAP of Baseline + FHFM improves from 48.8%, 77.8%, 87.7%, and
  1133	75.9% to 58.6%, 80.6%, 88.6%, and 79.6%, respectively, and its Rank9
  1134	
  1135	Neural Networks 201 (2026) 108883
  1136	
  1137	H. Du et al.
  1138	
  1139	Table 5
  1140	Results of ablation experiments (including mAP (%) and Rank-1 accuracy (%)) on the Occluded-DukeMTMC,
  1141	Occluded-ReID, Market-1501 and DukeMTMC-ReID datasets.
  1142	Model
  1143	
  1144	Baseline
  1145	Baseline + FHFM
  1146	Baseline + FSM
  1147	Baseline + LFEM
  1148	Baseline + FHFM + FSM
  1149	Baseline + FHFM + LFEM
  1150	Baseline + FSM + LFEM
  1151	Baseline + FHFM + FSM + LFEM
  1152	
  1153	Occluded-DukeMTMC
  1154	
  1155	Occluded-ReID
  1156	
  1157	Market1501
  1158	
  1159	DukeMTMC-ReID
  1160	
  1161	mAP
  1162	
  1163	Rank-1
  1164	
  1165	mAP

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf' - | nl -ba | sed -n '787,830p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   787	7) Impact of CLIP ID Labeler: The CLIP ID labeler plays
   788	a pivotal role in generating pseudo-IDs for training the re-ID
   789	subtask. To assess its impact, we substitute the CLIP ID labeler
   790	with different clustering methods, while maintaining other
   791	components unchanged. As depicted in Table VI, the CLIP ID
   792	labeler achieves an mAP of 21.56 pp, whereas replacing it with
   793	SPCL [53], a widely used clustering method in WS person
   794	search, results in a significant performance drop of 5.16/25.89
   795	pp with respect tomAP on PRW/CUHK-SYSU dataset. These
   796	results demonstrate the effectiveness of our CLIP ID labeler.
   797	8) Computational Resource and Inference Speed of Our
   798	Two Variants: We further compare the computational efficiency of FMUPS-S1 and FMUPS-S2 on the CUHK-SYSU
   799	dataset. As presented in Table VII, FMUPS-S1 demonstrates
   800	a faster inference speed (256 ms) with lower FLOPs (353.59
   801	GFLOPs) but requires more GPU memory (5.9 G). In contrast,
   802	FMUPS-S2 exhibits higher computational cost and slower
   803	inference (534 ms, 726.78 GFLOPs) while being significantly
   804	more memory-efficient (2.1 G). This tradeoff makes FMUPSS1 suitable for real-time inference and FMUPS-S2 preferable
   805	in memory-constrained settings.
   806	9) Performance of Our Method at Different Crowd Densities: We analyze FMUPS-S1 under varying crowd densities
   807	on the CUHK-SYSU dataset, where density is defined as
   808	
   809	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:57 UTC from IEEE Xplore. Restrictions apply.
   810	
   811	This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
   812	TIAN et al.: HARNESSING KNOWLEDGE FROM PRETRAINED VLMs FOR US PERSON SEARCH
   813	
   814	9
   815	
   816	TABLE IX
   817	C OMPARISON W ITH O NE -S TAGE M ETHODS
   818	
   819	Fig. 8. Comparison of the performance degradation between SEEM+SPCL
   820	[53] and our FMUPS-S1 under varying noise conditions. The degradation is
   821	calculated relative to each method’s own performance at 0% noise.
   822	
   823	the number of people per image. Table VIII summarizes the
   824	results across different density ranges. Performance decreases
   825	as density increases, with mAP dropping from 79.27 pp at low
   826	density ([1, 2)) to 57.54 pp at very high density (20+). This
   827	indicates that crowded scenes remain a challenging and largely
   828	unsolved problem for person search, pointing to a promising
   829	direction for future research.
   830	10) Robustness of Our Method to Noise: We evaluate the

exec
/bin/zsh -lc "pdftotext 'Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '1599,1625p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1599	Fig. 7. Feature visualization of the three modules.
  1600	
  1601	Fig. 8. Visualization of retrieval results.
  1602	
  1603	4.6. Visualization of results
  1604	
  1605	work, we will design a region-guided feature extraction strategy to enhance the performance of our model.
  1606	To visualize the eﬀectiveness of HFLAT more clearly, we present the
  1607	retrieval results for several query images from the Occluded-DukeMTMC
  1608	dataset in Fig. 8. In Fig. 8, the query images in rows 1–4, 5–8, and 9–12
  1609	are occluded by billboards, vehicles, and pedestrians, respectively. The
  1610	image on the leftmost of each row is the query image, while the remaining ten images are the retrieved results sorted in descending order of
  1611	similarity to the query image. Green numbers and green bounding boxes
  1612	indicate the accurate retrieval results, while red numbers and red bounding boxes denote the inaccurate retrieval results. As shown in Fig. 8,
  1613	
  1614	To demonstrate the eﬀectiveness of HFLAT clearly, we visualize the
  1615	features of FHFM, FSM, and LFEM in Fig. 7. As shown in Fig. 7, FSM
  1616	can separate foreground features and background features eﬀectively,
  1617	FHFM can focus well on key regions, and LFEM can capture ﬁne-grained
  1618	local features eﬀectively. Nevertheless, it is also evident from Fig. 7 that,
  1619	when the occluding object is another pedestrian, all the three modules
  1620	exhibit a tendency to focus on the interfering pedestrian. This may prevent our model from extracting eﬀective discriminative features from
  1621	the small, unobstructed body regions of the target pedestrian. In future
  1622	12
  1623	
  1624	Neural Networks 201 (2026) 108883
  1625	

exec
/bin/zsh -lc "pdftotext 'Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf' - | nl -ba | sed -n '1301,1365p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1301	5.4. Qualitative results (RQ4)
  1302	We implement three qualitative experimental results, feature representation visualization, heatmap visualization, and retrieval visualization, to
  1303	visually illustrate the eﬀectiveness of our HKGR. Note that the Baseline model mentioned in this subsection refers to models initialized
  1304	with CLIP pre-trained weights and ﬁne-tuned using the original InfoNCE
  1305	loss (van den et al., 2018).
  1306	For feature representation visualization: To visually demonstrate the
  1307	eﬀectiveness of our proposal in semantic correspondence of image and
  1308	text, we use t-SNE to transform the high-dimensional image and text
  1309	feature representations from the common latent space into a 2D plane.
  1310	As shown in Fig. 5(a), it can be observed that the inter-modality feature
  1311	representations extracted from the pre-trained backbone of CLIP in the
  1312	
  1313	Table 6
  1314	Comparisons between four diﬀerent knowledge reasoning
  1315	strategies of our HKGR on the CUHK-PEDES dataset.
  1316	No.
  1317	
  1318	Method
  1319	
  1320	R@1
  1321	
  1322	R@5
  1323	
  1324	R@10
  1325	
  1326	0
  1327	1
  1328	2
  1329	3
  1330	4
  1331	
  1332	MLM-based reasoning
  1333	Object-level Reasoning
  1334	Attribute-level Reasoning
  1335	Relation-level Reasoning
  1336	Full HKGR
  1337	
  1338	75.03
  1339	74.77
  1340	74.69
  1341	74.62
  1342	75.21
  1343	
  1344	90.09
  1345	90.09
  1346	90.31
  1347	90.01
  1348	90.22
  1349	
  1350	94.18
  1351	94.25
  1352	94.20
  1353	94.08
  1354	94.31
  1355	8
  1356	
  1357	Neural Networks 192 (2025) 107888
  1358	
  1359	R. Zeng et al.
  1360	
  1361	Fig. 5. Visualization of the semantic common space for 15 randomly selected
  1362	pedestrian embeddings (including both image and text modalities) from the
  1363	CUHK-PEDES dataset by using the t-SNE (Van der Maaten & Hinton, 2008).
  1364	The triangles represent text modality embeddings, while the circles represent
  1365	image modality embeddings. The same color indicates the data from the same

codex
你列出来的是 6 篇，我按这 6 篇拆。下面只讲“观察如何被包装成创新”，不讲怎么照搬模块。

**1. GSTNET**
1. 触发观察：作者抓住一个很具体的反常识点，两个群体外观相似，但如果 5 秒跨 20 公里，物理上不可能是同一群人。旧方法只用群体外观、图关系或静态时间统计，没管“能不能到达”。
2. 重定义动作：它把群体重识别从“外观加时空先验匹配”重定义为“受地理时间可达性约束的图匹配”。关键词是 `reachability constraint`、`speed limit`、`unreachable connections`、`static time-prior mismatch`。
3. 机制怎么长出来：机制几乎是自然推出的。既然问题是不可达边，就用距离和时间算速度，用速度上限和松弛边界惩罚图边；既然图传播会放大错边，就加门控图传播，只让可信邻居传播。
4. 证据闭环：关键不是只报涨点，而是拆开证明两个判断。BRT 上 Graph-MLP 基线 46.49 Rank-1、44.97 mAP，加 GstRM 到 58.60、55.31，说明可达性约束确实过滤了伪边；加 GGM 到 64.71、61.67，说明传播门控也有用；两者合到 71.06、68.81。t-SNE 进一步显示同身份更紧、异身份更分开。
5. 审稿人为什么买账：这篇卖的主要是视角。它把一个常识性物理约束变成 ReID 图边定义，机制并不复杂，但和问题重定义绑定很紧，审稿人容易相信“这个变量以前确实没被动态建模”。

**2. Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search**
1. 触发观察：作者先说无监督 person search 不是“无监督检测加无监督 ReID”的简单拼接。真正卡点是检测出来的伪框有定位噪声和类别噪声，伪框一脏，后面的伪身份聚类就塌。
2. 重定义动作：它把问题从“没有标注”重定义成“带伪框噪声的伪标签生成”。关键词是 `pseudo-box noise`、`pseudo-ID denoising`、`localization noise`、`classification noise`、`prior knowledge`。
3. 机制怎么长出来：机制和重定义绑定很紧。SEEM 负责给伪框和 mask，CLIP 用属性文本给伪身份，单图中同一个人不能出现两次作为去冲突先验；定位噪声用 mask 前景特征约束，类别噪声用置信度降权。
4. 证据闭环：它没有只和监督方法比，而是专门搭 FeatComp+DBSCAN、SEEM+DBSCAN、SEEM+SPCL 这些伪标签基线。FMUPS-S1 在 CUHK-SYSU 比 SEEM+SPCL 高 25.89 mAP，说明不是 SEEM 自己带来的；置信度、先验、CLIP ID labeler 都有单独消融，替换成聚类会明显掉点。
5. 审稿人为什么买账：这篇卖的是新设定加问题拆解。VLM 本身不是新零件，但作者把 VLM 从“特征提取器”改成“无监督 person search 的伪标签工程师”，再用噪声类型把每个组件解释清楚。

**3. HGTDR**
1. 触发观察：作者看到两类旧路都有问题。姿态、解析等外部工具贵且有域偏差；条带或固定 part token 又会把背景和遮挡混进来。另一方面，特征恢复方法虽然补遮挡，但没考虑邻近可见区域对恢复的贡献权重。
2. 重定义动作：它把遮挡 ReID 从“找可见部位”重定义成“生成异质语义 token，并按空间距离恢复缺失 token”。关键词是 `heterogeneous tokens`、`strong-response`、`weak-response`、`distance-aware recovery`。
3. 机制怎么长出来：SPG 先产生局部语义 patch，AHSTG 用区分性和身份相关性打分，选择不同形状和尺度的异质 token；如果 token 间空间距离大，就认为中间可能缺失，插入 recovery token，并让近邻 token 权重大、远邻权重小。前半段逻辑很顺，DFR 的距离假设更像一个合理启发。
4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
5. 审稿人为什么买账：这篇卖的是机制组合，不是一个特别新的问题视角。它的可取之处是把“固定部位不可靠”和“恢复要有邻近权重”连成一条完整流水线，审稿人买的是每个模块都对应一个可见失败点。

**4. HFLAT**
1. 触发观察：作者抓的是 Transformer 遮挡 ReID 的通病，patch 被近似一视同仁处理，没有量化不同 patch 对身份的贡献，也缺少足够强的局部细节抽取能力。
2. 重定义动作：它把遮挡问题从“遮挡区域要恢复”改写为“patch 要分层、前景背景要分离、局部交互要受限”。关键词是 `relative importance`、`patch-level saliency`、`feature hierarchical fusion`、`local-aware attention`。
3. 机制怎么长出来：FHFM 根据 class token 注意力和 patch 范数给 patch 排重要性，分四层融合；FSM 用 saliency map 分前景和背景；LFEM 加四个局部 token，并用 mask 限制只在同一人体区域内交互。逻辑能对上观察，但三模块有些“全都要”，绑定程度中等。
4. 证据闭环：消融比较了 Baseline、单模块、双模块和完整模型。比如 Occluded-DukeMTMC 上 Baseline mAP 48.8，加 FHFM 到 58.6；FSM、LFEM 也分别带来收益，双模块继续涨，三者完整最好。可视化显示 FHFM 关注关键区域，FSM 分离前景背景，LFEM 捕捉局部细节，同时作者也承认遇到行人遮挡行人时模块会被干扰者吸引。
5. 审稿人为什么买账：这篇主要卖工程完整性。视角不算尖锐，但“重要性分层、前景分离、局部受限交互”三个动作都能被消融和可视化支撑，所以容易作为一篇稳健的模块型遮挡稿成立。

**5. HKGR**
1. 触发观察：作者发现 TIReID 里直接套 MLM 有问题。随机 mask subword 可能遮到功能词或碎片词，训练目标偏离“图文语义对齐”；而行人描述真正有用的是对象、属性和关系，比如“白色上衣”“站在栅栏前”。
2. 重定义动作：它把文本图像对齐从“随机词预测”重定义为“由场景图知识引导的层级语义推理”。关键词是 `blind visual-textual alignment`、`hierarchical scene graph knowledge`、`object-level`、`attribute-level`、`relation-level reasoning`。
3. 机制怎么长出来：机制非常自然。先把文本解析成对象、属性、关系三层场景图，再分别 mask 对象词、属性词、关系词，让图像和剩余文本共同预测被 mask 的语义；再用 MGA 同时做全局句图对齐和局部 token 对齐。
4. 证据闭环：消融把 KGR、MGA、ID loss 拆开。MGA 带来的收益最大，KGR 在多个组合里有稳定增益；还专门比较 MLM、对象推理、属性推理、关系推理和完整 HKGR，完整模型最好。t-SNE、热图、检索可视化用来证明它不是只涨点，而是图文语义空间更靠近。
5. 审稿人为什么买账：这篇卖的是视角。它没有说“我又加了一个语言模块”，而是说“随机 MLM 对行人文本是不合适的，文本本身有层级知识”。这个重定义比模块本身更像贡献。

**6. Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification**
1. 触发观察：作者抓住换衣 ReID 的几何矛盾，同一人换衣导致类内距离大，不同人穿相似衣服导致类间距离小。实例级 triplet 或 contrastive 容易被噪声正负样本带偏；直接用形体、步态等固有特征又会损伤同衣场景。
2. 重定义动作：它把“去衣服特征”重定义成“建立身份主代理和衣服状态子代理之间的层级几何关系”。关键词是 `main proxy`、`sub proxy`、`sub-to-main proxy`、`automatic sub-proxy mining`、`sample balance and diversity`。
3. 机制怎么长出来：先按身份构造 main proxy，再在每个身份内部用 DBSCAN 挖 sub-proxy，近似衣服状态但不需要衣服标签；SBS 保证 batch 里身份和子代理均衡；SCT 用解析引导换衣增广补足一个人只有一套衣服的问题；MPL、SPL、S2MPL 分别约束身份间分离、身份内紧凑、子代理向主代理协作。
4. 证据闭环：组件消融显示 SBS、SCT、HPL 逐步提升，HPL 内部的 SPL、MPL、S2MPL 也分别有效。PRCC 换衣设置达到 74.3 Rank-1，VC-Clothes 换衣设置达到 84.9 Rank-1。特征图显示方法更关注头、脸、手臂等非衣服区域，这正好回应“衣服捷径被压下去”。
5. 审稿人为什么买账：这篇卖的是机制和几何视角的结合。真正的 novelty 不是衣服迁移，而是把“换衣造成的类内多峰分布”说成主代理和子代理的层级结构，并且不依赖人工衣服标签。

共同套路很清楚：先抓一个具体失败点，不说“模型表达不够”，而是把失败点命名成一个更窄的新问题，例如可达性、伪框噪声、随机 MLM、层级代理。然后机制不是凭空堆模块，而是让每个模块对应一个被重新定义后的子矛盾。最后证据闭环也不是只报 SOTA，而是做“旧假设错在哪里、新定义对应的模块是否单独有效、可视化是否符合新解释”这三件事。
tokens used
189,766
你列出来的是 6 篇，我按这 6 篇拆。下面只讲“观察如何被包装成创新”，不讲怎么照搬模块。

**1. GSTNET**
1. 触发观察：作者抓住一个很具体的反常识点，两个群体外观相似，但如果 5 秒跨 20 公里，物理上不可能是同一群人。旧方法只用群体外观、图关系或静态时间统计，没管“能不能到达”。
2. 重定义动作：它把群体重识别从“外观加时空先验匹配”重定义为“受地理时间可达性约束的图匹配”。关键词是 `reachability constraint`、`speed limit`、`unreachable connections`、`static time-prior mismatch`。
3. 机制怎么长出来：机制几乎是自然推出的。既然问题是不可达边，就用距离和时间算速度，用速度上限和松弛边界惩罚图边；既然图传播会放大错边，就加门控图传播，只让可信邻居传播。
4. 证据闭环：关键不是只报涨点，而是拆开证明两个判断。BRT 上 Graph-MLP 基线 46.49 Rank-1、44.97 mAP，加 GstRM 到 58.60、55.31，说明可达性约束确实过滤了伪边；加 GGM 到 64.71、61.67，说明传播门控也有用；两者合到 71.06、68.81。t-SNE 进一步显示同身份更紧、异身份更分开。
5. 审稿人为什么买账：这篇卖的主要是视角。它把一个常识性物理约束变成 ReID 图边定义，机制并不复杂，但和问题重定义绑定很紧，审稿人容易相信“这个变量以前确实没被动态建模”。

**2. Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search**
1. 触发观察：作者先说无监督 person search 不是“无监督检测加无监督 ReID”的简单拼接。真正卡点是检测出来的伪框有定位噪声和类别噪声，伪框一脏，后面的伪身份聚类就塌。
2. 重定义动作：它把问题从“没有标注”重定义成“带伪框噪声的伪标签生成”。关键词是 `pseudo-box noise`、`pseudo-ID denoising`、`localization noise`、`classification noise`、`prior knowledge`。
3. 机制怎么长出来：机制和重定义绑定很紧。SEEM 负责给伪框和 mask，CLIP 用属性文本给伪身份，单图中同一个人不能出现两次作为去冲突先验；定位噪声用 mask 前景特征约束，类别噪声用置信度降权。
4. 证据闭环：它没有只和监督方法比，而是专门搭 FeatComp+DBSCAN、SEEM+DBSCAN、SEEM+SPCL 这些伪标签基线。FMUPS-S1 在 CUHK-SYSU 比 SEEM+SPCL 高 25.89 mAP，说明不是 SEEM 自己带来的；置信度、先验、CLIP ID labeler 都有单独消融，替换成聚类会明显掉点。
5. 审稿人为什么买账：这篇卖的是新设定加问题拆解。VLM 本身不是新零件，但作者把 VLM 从“特征提取器”改成“无监督 person search 的伪标签工程师”，再用噪声类型把每个组件解释清楚。

**3. HGTDR**
1. 触发观察：作者看到两类旧路都有问题。姿态、解析等外部工具贵且有域偏差；条带或固定 part token 又会把背景和遮挡混进来。另一方面，特征恢复方法虽然补遮挡，但没考虑邻近可见区域对恢复的贡献权重。
2. 重定义动作：它把遮挡 ReID 从“找可见部位”重定义成“生成异质语义 token，并按空间距离恢复缺失 token”。关键词是 `heterogeneous tokens`、`strong-response`、`weak-response`、`distance-aware recovery`。
3. 机制怎么长出来：SPG 先产生局部语义 patch，AHSTG 用区分性和身份相关性打分，选择不同形状和尺度的异质 token；如果 token 间空间距离大，就认为中间可能缺失，插入 recovery token，并让近邻 token 权重大、远邻权重小。前半段逻辑很顺，DFR 的距离假设更像一个合理启发。
4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
5. 审稿人为什么买账：这篇卖的是机制组合，不是一个特别新的问题视角。它的可取之处是把“固定部位不可靠”和“恢复要有邻近权重”连成一条完整流水线，审稿人买的是每个模块都对应一个可见失败点。

**4. HFLAT**
1. 触发观察：作者抓的是 Transformer 遮挡 ReID 的通病，patch 被近似一视同仁处理，没有量化不同 patch 对身份的贡献，也缺少足够强的局部细节抽取能力。
2. 重定义动作：它把遮挡问题从“遮挡区域要恢复”改写为“patch 要分层、前景背景要分离、局部交互要受限”。关键词是 `relative importance`、`patch-level saliency`、`feature hierarchical fusion`、`local-aware attention`。
3. 机制怎么长出来：FHFM 根据 class token 注意力和 patch 范数给 patch 排重要性，分四层融合；FSM 用 saliency map 分前景和背景；LFEM 加四个局部 token，并用 mask 限制只在同一人体区域内交互。逻辑能对上观察，但三模块有些“全都要”，绑定程度中等。
4. 证据闭环：消融比较了 Baseline、单模块、双模块和完整模型。比如 Occluded-DukeMTMC 上 Baseline mAP 48.8，加 FHFM 到 58.6；FSM、LFEM 也分别带来收益，双模块继续涨，三者完整最好。可视化显示 FHFM 关注关键区域，FSM 分离前景背景，LFEM 捕捉局部细节，同时作者也承认遇到行人遮挡行人时模块会被干扰者吸引。
5. 审稿人为什么买账：这篇主要卖工程完整性。视角不算尖锐，但“重要性分层、前景分离、局部受限交互”三个动作都能被消融和可视化支撑，所以容易作为一篇稳健的模块型遮挡稿成立。

**5. HKGR**
1. 触发观察：作者发现 TIReID 里直接套 MLM 有问题。随机 mask subword 可能遮到功能词或碎片词，训练目标偏离“图文语义对齐”；而行人描述真正有用的是对象、属性和关系，比如“白色上衣”“站在栅栏前”。
2. 重定义动作：它把文本图像对齐从“随机词预测”重定义为“由场景图知识引导的层级语义推理”。关键词是 `blind visual-textual alignment`、`hierarchical scene graph knowledge`、`object-level`、`attribute-level`、`relation-level reasoning`。
3. 机制怎么长出来：机制非常自然。先把文本解析成对象、属性、关系三层场景图，再分别 mask 对象词、属性词、关系词，让图像和剩余文本共同预测被 mask 的语义；再用 MGA 同时做全局句图对齐和局部 token 对齐。
4. 证据闭环：消融把 KGR、MGA、ID loss 拆开。MGA 带来的收益最大，KGR 在多个组合里有稳定增益；还专门比较 MLM、对象推理、属性推理、关系推理和完整 HKGR，完整模型最好。t-SNE、热图、检索可视化用来证明它不是只涨点，而是图文语义空间更靠近。
5. 审稿人为什么买账：这篇卖的是视角。它没有说“我又加了一个语言模块”，而是说“随机 MLM 对行人文本是不合适的，文本本身有层级知识”。这个重定义比模块本身更像贡献。

**6. Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification**
1. 触发观察：作者抓住换衣 ReID 的几何矛盾，同一人换衣导致类内距离大，不同人穿相似衣服导致类间距离小。实例级 triplet 或 contrastive 容易被噪声正负样本带偏；直接用形体、步态等固有特征又会损伤同衣场景。
2. 重定义动作：它把“去衣服特征”重定义成“建立身份主代理和衣服状态子代理之间的层级几何关系”。关键词是 `main proxy`、`sub proxy`、`sub-to-main proxy`、`automatic sub-proxy mining`、`sample balance and diversity`。
3. 机制怎么长出来：先按身份构造 main proxy，再在每个身份内部用 DBSCAN 挖 sub-proxy，近似衣服状态但不需要衣服标签；SBS 保证 batch 里身份和子代理均衡；SCT 用解析引导换衣增广补足一个人只有一套衣服的问题；MPL、SPL、S2MPL 分别约束身份间分离、身份内紧凑、子代理向主代理协作。
4. 证据闭环：组件消融显示 SBS、SCT、HPL 逐步提升，HPL 内部的 SPL、MPL、S2MPL 也分别有效。PRCC 换衣设置达到 74.3 Rank-1，VC-Clothes 换衣设置达到 84.9 Rank-1。特征图显示方法更关注头、脸、手臂等非衣服区域，这正好回应“衣服捷径被压下去”。
5. 审稿人为什么买账：这篇卖的是机制和几何视角的结合。真正的 novelty 不是衣服迁移，而是把“换衣造成的类内多峰分布”说成主代理和子代理的层级结构，并且不依赖人工衣服标签。

共同套路很清楚：先抓一个具体失败点，不说“模型表达不够”，而是把失败点命名成一个更窄的新问题，例如可达性、伪框噪声、随机 MLM、层级代理。然后机制不是凭空堆模块，而是让每个模块对应一个被重新定义后的子矛盾。最后证据闭环也不是只报 SOTA，而是做“旧假设错在哪里、新定义对应的模块是否单独有效、可视化是否符合新解释”这三件事。
