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
session id: 019ef835-7945-7661-998b-89b9415d5bbd
--------
user
你是 ReID 创新方法论拆解员 7。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(7)==
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Confidence guided semi-supervised cross-modality person re-identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我先按项目规则尝试确认当前目标和目录内容，然后逐篇抽取前 8 页正文，重点定位 method/approach，而不是只看摘要。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'STATUS.md' -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "find . -maxdepth 1 -type f -iname '*.pdf' -print | sort" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
./'Knowledge and experience' for visible-infrared person re-identification.pdf
./3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
./4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
./A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
./A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf
./A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
./A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
./ADA framework for unsupervised domain adaptation person re-identification.pdf
./Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
./Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
./Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf
./Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
./Adversarial perturbation and defense for generalizable person re-identification.pdf
./Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
./Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
./Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
./Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
./Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf
./Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
./CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
./CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
./CLIP-driven fine-grained mining for text-based person search.pdf
./CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
./CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
./CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
./CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
./CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
./Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
./Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
./Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
./Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
./Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
./Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
./ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
./Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
./Confidence guided semi-supervised cross-modality person re-identification.pdf
./Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
./Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
./Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
./Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
./Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
./Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
./Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
./Cross-modality average precision optimization for visible thermal person re-identification.pdf
./CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
./DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
./DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
./Deep intelligent technique for person Re-identification system in surveillance images.pdf
./Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
./Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
./Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
./Distribution aligned semantics adaption for lifelong person re-identification.pdf
./Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
./DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
./Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
./Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
./Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
./Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
./Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
./Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
./Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
./Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
./Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
./FDGReID - Federated Domain Generalization for Person Re-identification.pdf
./FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
./FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
./False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
./Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
./Focusing on pedestrians like human for clothes changing person re-identification.pdf
./GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
./GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
./GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
./Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
./Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
./HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
./HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
./Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
./Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
./Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
./Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
./Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
./Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
./Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
./Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
./InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
./Instant pose extraction based on mask transformer for occluded person re-identification.pdf
./Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
./Internal-External Context Interaction Network for Person Re-Identification.pdf
./Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
./Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
./Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
./Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
./Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
./Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
./Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
./MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
./Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
./Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
./Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
./Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
./Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
./MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
./Multi Queue for Unsupervised Person Re-identification.pdf
./Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
./Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
./Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
./Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
./Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
./Multi-feature balanced network for clothes-changing person re-identification.pdf
./Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
./Multi-year long-term person re-identification using gait and HAR features.pdf
./Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
./Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf
./Occluded person Re-Identification with noise injection.pdf
./Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf
./Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf
./Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
./Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
./Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf
./Privacy preserving person re-identification via anonymizing diffusion model.pdf
./Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
./Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
./Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
./RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
./RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
./Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
./Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
./Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
./SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
./See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
./Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
./Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
./Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
./Shape-centered representation learning for visible-infrared person re-identification.pdf
./Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
./Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
./Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
./Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
./Spectrum-guided feature enhancement network for event person re-identification.pdf
./Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
./Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
./TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
./TP-LReID - Lifelong person re-identification using text prompts.pdf
./Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
./Text-to-image Person Search based on Semantic Reorganization.pdf
./Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
./Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
./Two-stage knowledge distillation for visible-infrared person re-identification.pdf
./UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
./UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
./Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
./Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
./Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
./Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
./Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
./Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
./Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf
./Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
./Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
./Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
./When Large Vision-Language Models Meet Person Re-Identification.pdf

codex
目录里能看到用户列出的 6 篇 PDF，而不是 7 篇；我先按这 6 篇逐篇做，最后会明确说明数量差异。现在先抽标题结构，确认每篇方法部分在哪几页结束。
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
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.?\\s+[A-Z]|[A-Z][A-Z ]{5,}|Method|METHOD|Approach|APPROACH|Experiments|EXPERIMENTS|Conclusion|CONCLUSION)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
4:1 School of Computer Science, Fudan University, Shanghai, China
5:2 School of Automation, Northwestern Polytechnical University, Xi’an, China
6:3 School of Data Science, Fudan University, Shanghai, China
12:Abstract—Cloth-changing person re-identification aims at recognizing
31:I. I NTRODUCTION
176:II. M ETHODOLOGY
268:TABLE I
269:C OMPARISON OF OUR METHOD WITH STATE - OF - THE - ART METHODS ON PRCC
270:AND LTCC. M ETHODS IN THE GRAY REGION USE EXTRA GROUND - TRUTH
271:CLOTHING LABELS FOR TRAINING . “ SKETCH ”, “ SIL .”, AND “ POSE ” REPRESENT
272:CONTOUR SKETCHES , SILHOUETTES , AND HUMAN POSES , RESPECTIVELY. M ETHODS
273:MARKED WITH “∗” INVOLVE MULTIPLE TRAINING STAGES FOR EXTRA AUXILIARY
274:NETWORKS . “S TANDARD ” AND “C LOTH -C HANGING ” MEAN THE STANDARD AND
275:CLOTH - CHANGING SETTINGS , RESPECTIVELY. T HE BEST RESULTS ARE SHOWN IN
278:Methods
492:III. E XPERIMENTS
585:TABLE IV
587:C OMPARISONS RESULTS ON C ELEB - RE ID. “ POSE ” REPRESENTS HUMAN POSES .
588:M ETHODS MARKED WITH “†” ADOPT D ENSE N ET-121 AS THE BACKBONE .
590:A BLATION OF THE CROSS - PARALLEL SEMANTICS COLLABORATION
591:FRAMEWORK . W E REPORT THE RESULTS ON PRCC AND LTCC UNDER THE
592:CLOTH - CHANGING SETTING . “SMR-C-S (SMR-S-C)” DENOTES SEQUENTIALLY
593:USING SMR-C AND SMR-S (SMR-S AND SMR-C) IN ONE BRANCH .
595:Methods
660:Methods
742:TABLE III
743:A BLATION STUDIES OF THE SMR MODULE . W E REPORT THE RESULTS ON PRCC
744:AND LTCC UNDER THE CLOTH - CHANGING SETTING . “L OCAL .” AND “R EFINE .”
745:DENOTE LOCAL SEMANTICS MINING AND SEMANTICS REFINEMENT IN SMR.
748:Methods
858:IV. C ONCLUSION
945:IEEE Transactions on Circuits and Systems for Video Technology, 2023.
968:IEEE Transactions on Image Processing, vol. 32, pp. 2985–2999, 2023.

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.?\\s+[A-Z]|[A-Z][A-Z ]{5,}|Method|METHOD|Approach|APPROACH|Experiments|EXPERIMENTS|Conclusion|CONCLUSION)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
49:1. Introduction
154:2. Related work
222:3. Proposed method
469:4. Experiments
708:Methods
715:Methods
956:Methods
1037:Methods
1145:Methods

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.?\\s+[A-Z]|[A-Z][A-Z ]{5,}|Method|METHOD|Approach|APPROACH|Experiments|EXPERIMENTS|Conclusion|CONCLUSION)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
9:Abstract— Video-based person re-identification (Re-ID) aims at
36:I. I NTRODUCTION
37:ERSON re-identification (Re-ID) [1], which attempts
98:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
160:II. R ELATED W ORK
259:TABLE I
260:S UMMARY OF S YMBOLS AND D EFINITIONS
277:III. P ROPOSED M ETHOD
317:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
439:LC A R E = Lce (W c z q ),
536:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
593:TABLE II
594:P ROPERTIES OF THE V IDEO -BASED Re-ID DATASETS
600:1 X
618:IV. E XPERIMENTS
675:LSTRL [51], STMN [50], and STMP [23], attention-based:
688:TABLE III
689:C OMPARISON W ITH S TATE - OF - THE -A RT A PPROACHES ON LS-VID.
690:W E D IVIDE T HESE A PPROACHES I NTO S EVERAL G ROUPS BASED
691:ON T HEIR U TILIZATIONS OF T EMPORAL C LUES . OF R EPRESENTS
692:O PTICAL F LOW, 3D R EPRESENTS 3D C ONVOLUTIONAL
693:N EURAL N ETWORKS , RNN S R EPRESENTS R ECURRENT
694:N EURAL N ETWORKS , AND AM R EPRESENTS ATTENTION
695:M ECHANISM . B OLD VALUES I NDICATE THE B EST
696:P ERFORMANCE . B OTH R ANK 1 AND
787:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
789:TABLE IV
790:C OMPARISON W ITH S TATE - OF - THE -A RT M ETHODS ON DukeMTMC-VideoReID AND MARS. T HE A PPROACHES A RE S EPARATED I NTO S EVERAL
791:G ROUPS . RNN S R EPRESENTS R ECURRENT N EURAL N ETWORKS , OF R EPRESENTS O PTICAL F LOW, TP R EPRESENTS T EMPORAL P OOLING ,
793:M ECHANISM . B OLD VALUES I NDICATE THE B EST P ERFORMANCE . R ANK 20, R ANK 5, R ANK 1, AND mAP ACCURACY A RE L ISTED
795:TABLE V
796:A BLATION S TUDIES ON LS-VID, MARS, AND D UKE -V IDEO . CSSA1−4 D ENOTE F OUR VARIANTS OF THE P ROPOSED CSSA F RAMEWORK .
797:FACTORS IN C OLUMN ‘S ETTING ’ D ENOTE W HETHER THE E XPERIMENT I S P ERFORMED W ITH THE C ORRESPONDING M ODULE .
798:B OLD VALUES I NDICATE THE B EST P ERFORMANCE . R ANK 5, R ANK 1 AND mAP ACCURACY A RE L ISTED

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Confidence guided semi-supervised cross-modality person re-identification.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.?\\s+[A-Z]|[A-Z][A-Z ]{5,}|Method|METHOD|Approach|APPROACH|Experiments|EXPERIMENTS|Conclusion|CONCLUSION)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.?\\s+[A-Z]|[A-Z][A-Z ]{5,}|Method|METHOD|Approach|APPROACH|Experiments|EXPERIMENTS|Conclusion|CONCLUSION)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025
8:Abstract— Corruption-invariant
42:I. I NTRODUCTION
46:ORRUPTION-INVARIANT person re-identification
50:2024. Date of publication 2 October 2024; date of current version 13 February
51:2025. This work was supported in part by the Talent Introduction Program for
186:II. R ELATED W ORK
197:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025
314:III. M ETHOD
383:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025
398:1 X
430:1 X (1 − ω)esim(Z n ,Z n )/τ
499:1 X
551:Z B B = µ(Z Cl , Z Cor , t) ≜
569:Z B F G = D F V G (g(µ, σ, ε))
603:Z B B , defined as
604:L K L ( pB F G , pB B )
631:1 X
654:IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025
668:TABLE I
669:C ORRUPTION U SED IN DATA R ECONSTRUCTION
671:TABLE II
672:F OUR D IFFERENT E VALUATION P ROTOCOLS . “▲” D ENOTES
673:C ORRUPTED S ET, “∇” D ENOTES C LEAN S ET
703:IV. E XPERIMENTS
759:1 The source codes are publicly
774:TABLE III
775:P ERFORMANCE C OMPARISON ON THE M ARKET-1501 DATASET. T HE B EST R ESULTS A RE H IGHLIGHTED IN B OLD . * I NDICATES THE BASELINE
777:TABLE IV
778:P ERFORMANCE C OMPARISON ON THE CUHK-03 DATASET

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.?\\s+[A-Z]|[A-Z][A-Z ]{5,}|Method|METHOD|Approach|APPROACH|Experiments|EXPERIMENTS|Conclusion|CONCLUSION)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
13:ARTICLE
22:ABSTRACT
39:1. Introduction
148:2. Related work
214:3. Approach
623:Methods
633:4. Experiments
743:DBSCAN are discarded during the training process. As for the setting of
899:DBSCAN. The results are shown in Figs. 3 and 4, where the first row
1030:Methods
1078:CASCL [59]
1087:CACHE [64]

 succeeded in 123ms:
23:ARTICLE
33:ABSTRACT
49:1. Introduction
87:2. Related work
234:3. Proposed method
416:3 Initialize sets  and , 𝑖 = 1, 𝑗 = 1.
417:4 Perform DBSCAN clustering on labeled data and unlabeled data
419:5 Compute matching confidence 𝛥 via Eq. (2)
429:2 Output: Optimized feature extraction model 𝑓
431:3 Initialize model parameters 𝜃
475:13 Count the number of matched image pairs in  for each cluster
477:14 Sort all cluster pairs by the matched numbers in descending
479:15 Compute clustering confidence 𝜌 via Eq. (4) on labeled data
480:16 Calculate the number of selected cluster pairs 𝑘 = ⌈𝜌 × 𝑛⌉,
482:17 Return top 𝑘 cluster pairs in 
811:4. Experiments
853:2. Specifically, in the Visible to Infrared search mode, there is a
908:Methods
1306:Methods
1632:Methods

codex
这些 PDF 的结构都能在前 8 页内定位到方法段：大多数是第 3 节，少数是第 II 节或 Approach。接下来我会读取方法本身和少量实验消融段，用来判断它们怎么把观察变成创新叙事。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '31,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    31	I. I NTRODUCTION
    32	Person Re-IDentification (Re-ID) aims to recognize individuals
    33	across different cameras and times. With the growing demand for
    34	surveillance applications and the resurgence of deep learning, significant efforts have been devoted to the advancement of person ReID [1]–[6]. Existing person Re-ID models are principally studied
    35	in the short-term scenario [7], where the clothing of the same
    36	person remains consistent. Consequently, the learned features heavily
    37	rely on clothing appearances, rendering the models ineffective when
    38	individuals change their clothes or wear similar clothes as others. As
    39	a result, there is a rising interest in addressing the cloth-changing
    40	challenge in long-term real-world scenarios [8]–[11].
    41	In this paper, we address the cloth-changing challenge without
    42	relying on any auxiliary module [8]–[10], [12], [13] or extra annotation [14]–[16]. Interestingly, we observe that two commonly
    43	used pooling strategies, namely average and max poolings, exhibit
    44	distinct characteristics in information integration. As depicted in
    45	Fig. 1, average pooling effectively preserves image content information while smoothing out some details, but max pooling excels at
    46	capturing salient information within the content. Motivated by these
    47	findings, we expect to leverage them to enhance the model’s ability
    48	in content understanding and salient information capture, which are
    49	both crucial for effective cloth-changing person Re-ID. On the one
    50	hand, the person Re-ID model is required to comprehend the content
    51	This work was supported in part by the National Natural Science Foundation
    52	of China under Grant 62176061 and Grant 62406252, in part by Shanghai
    53	Technology Development and Entrepreneurship Platform for Neuromorphic
    54	and AI SoC, and in part by the Shanghai Research and Innovation Functional
    55	Program under Grant 17DZ2260900.
    56	† Co-corresponding authors.
    57	
    58	Original
    59	images
    60	
    61	Average
    62	pooling
    63	
    64	Max
    65	pooling
    66	
    67	Fig. 1. Comparison of applying different poolings on pedestrian images.
    68	Average pooling can better preserve image content while smoothing details,
    69	while max pooling is better at capturing salient information in the content.
    70	Incorporating both of them is expected to learn abundant semantics, and thus
    71	improve the discriminative ability of Re-ID models.
    72	
    73	of pedestrian images and perceive clothing information to extract
    74	identity characteristics adaptively to tackle cloth-changing cases. On
    75	the other hand, salient identity cues play a vital role in mitigating confusion from varying clothes and stabilizing learning features robust to
    76	clothing variations. However, as shown in Fig. 1, employing pooling
    77	in the pixel space of original images is susceptible to interference
    78	from some identity-irrelevant information, such as background and
    79	illumination, which are unrelated to pedestrian identity.
    80	To this end, we exploit both pooling strategies to facilitate feature
    81	learning in the high-level semantic space with the guidance of identity. Specifically, we propose the Semantics Mining and Refinement
    82	(SMR) module. It guides the model to perceive and extract identityrelated semantics, including pedestrian body shape, faces, etc. Then,
    83	the learned semantic features facilitate model learning adaptively. To
    84	avoid interference from identity-irrelevant information and improve
    85	the robustness of the Re-ID model in the cloth-changing scenario, we
    86	propose to mine local semantics and guide the mining process with
    87	carefully designed identity supervision. By integrating the SMR module into the backbone seamlessly, we present the Content and Salient
    88	Semantics Collaboration (CSSC) framework capable of sequential
    89	and parallel semantic interaction and refinement. Both content and
    90	salient semantics play pivotal roles in recognizing and distinguishing
    91	pedestrians with interference from varying clothing appearances.
    92	Contributions. We summarize key contributions as follows.
    93	(1) We propose the novel Content and Salient Semantics Collaboration (CSSC) framework incorporating effective cross-parallel semantic interaction. It effectively addresses the cloth-changing challenge in
    94	
    95	SMR-S
    96	
    97	SMR-C
    98	
    99	" 𝒄𝒔
   100	"
   101	𝑭
   102	
   103	"𝒄
   104	𝑭
   105	𝑭
   106	
   107	SMR-S
   108	
   109	Conv
   110	Block
   111	
   112	𝓛𝒄𝒔𝒔𝒄
   113	𝑭𝒄𝒔𝒔𝒄
   114	
   115	SMR-C
   116	
   117	Backbone
   118	"
   119	" 𝒔𝒄
   120	𝑭
   121	
   122	"𝒔
   123	𝑭
   124	
   125	Element-wise
   126	Addition
   127	
   128	Semantics Mining and Refinement (SMR) Module
   129	
   130	Element-wise
   131	Multiplication
   132	
   133	Conv
   134	Block
   135	
   136	Sigmoid
   137	Function
   138	
   139	Horizontal
   140	Division
   141	
   142	C
   143	
   144	𝓛𝒊𝒅
   145	
   146	FC
   147	Semantics Mining
   148	
   149	𝓛𝒕𝒓𝒊
   150	
   151	ReLU
   152	
   153	C
   154	
   155	FC
   156	
   157	Semantics Refinement
   158	
   159	Concatenation
   160	Operation
   161	Average or Max
   162	Pooling
   163	
   164	Fig. 2. Framework of our method. The Semantics Mining and Refinement (SMR) module learns identity-related semantics without relying on any auxiliaries.
   165	SMR modules equipped with average pooling (SMR-C) and max pooling (SMR-S) learn content and salient semantics effectively, respectively. Taking advantage
   166	of SMR, our framework interacts and refines both semantics sequentially and parallelly to promote cloth-changing person Re-ID.
   167	
   168	person Re-ID without relying on unpractical and inflexible auxiliary
   169	modalities or extra annotations.
   170	(2) We propose the Semantics Mining and Refinement (SMR)
   171	module, which applies the distinct pooling characteristics to learn
   172	and utilize robust identity-related content and salient semantics in a
   173	unified manner.
   174	(3) Extensive experiments on three cloth-changing person ReID benchmarks demonstrate the effectiveness and superiority of our
   175	proposed method.
   176	II. M ETHODOLOGY
   177	A. Problem Formulation
   178	Given a training dataset D = {xi , yi }N
   179	i=1 containing totally N
   180	images and N p identities, where xi and yi represent the i-th images
   181	and its corresponding identity label, the Re-ID model G aims to
   182	map the person image xi into a discriminative feature representation
   183	Fi = G (xi ). Subsequently, a pooling layer is then applied to
   184	Fi ∈ Rh×w×d , where h and w are the height and width of the feature
   185	representation, yielding the feature vector fi ∈ Rd . The model G is
   186	required to tolerate clothing changes and extract the robust feature fi ,
   187	so that the similarity between features of pedestrians with the same
   188	identity is larger than that of pedestrians with different identities,
   189	regardless of whether they change clothes or not. In the subsequent
   190	discussion, unless otherwise specified, we omit the subscript i of Fi
   191	and fi to simplify notation.
   192	B. Semantics Mining and Refinement Module
   193	To enhance the discriminative ability of the model in the clothchanging scenario, we leverage both identity-related content and
   194	salient semantics with our designed Semantics Mining and Refinement (SMR) module. In the following, we first illustrate the
   195	learning of identity-related content semantics which help comprehend
   196	pedestrian images and extract identity characteristics adaptively to
   197	tackle cloth-changing cases.
   198	
   199	Global content semantics mining. With F extracted by the backbone, one convolutional block is appended, resulting in F c ∈
   200	Rh×w×d . Then, we obtain the average pooled feature,
   201	f c = AvgPool(F c ) ∈ Rd ,
   202	
   203	(1)
   204	
   205	where AvgPool denotes the global average pooling operation, which
   206	effectively integrates content semantics in feature maps.
   207	Local content semantics mining. In addition to mining global
   208	content semantics, the semantics of local regions are also crucial.
   209	Therefore, we horizontally divide F c into different parts and use one
   210	convolutional layer for dimensionality reduction, resulting in several
   211	c P
   212	local content semantic features {f lp }p=1 . However, in the clothchanging scenario, the reliability of local regions is questionable.
   213	Due to clothing changes and clothing similarity between pedestrians,
   214	if carried out as previous works [3], [17], local semantics can mislead
   215	model learning. To leverage local content semantics effectively, we
   216	concatenate the global and local content semantic features. Formally,
   217	c
   218	c
   219	c
   220	fˆc = [f c ; f l1 ; f l2 ; · · · ; f lP ],
   221	
   222	(2)
   223	
   224	where [·; ·] denotes concatenation in the feature dimension. To ensure
   225	the mined content semantics are identity-related and discriminative,
   226	widely-used Re-ID losses are used as follows:
   227	Lsmr c (F c ) = Lid (fˆc ) + Ltri (f c ),
   228	
   229	(3)
   230	
   231	where Lid and Ltri denote widely used identity classification
   232	loss [18] and triplet loss [19], respectively. Note that Lid includes
   233	an identity classifier implemented as one fully connected layer.
   234	Additonally, the identity label of fˆc (i.e., y) is required in Lid and
   235	Ltri for identity supervision, and we omit it for notation simplicity.
   236	Content semantics refinement. To improve the discriminative ability
   237	of the model, the mined identity-related global content semantics are
   238	also used to refine F c in an adaptive manner. F c is recalibrated along
   239	the channel dimension using the learned semantics in f c . Formally,
   240	Fec = σ(W2 ϕ(W1 f c )) ⊗ F c ,
   241	
   242	(4)
   243	
   244	where ϕ denotes the ReLU activation function, σ denotes the Sigmoid
   245	d
   246	function, and ⊗ denotes element-wise multiplication. W1 ∈ R r ×d
   247	
   248	d
   249	
   250	and W2 ∈ Rd× r are the weights of two Fully-Connected (FC)
   251	layers, where the first one compresses the channel dimension to
   252	achieve information bottleneck. r is set to 16 following [20]. The final
   253	Sigmoid function acts as a self-attention mechanism across channels,
   254	generating scores to refine the content semantics of pedestrian features adaptively. The ∼ notation represents our proposed semantics
   255	mining and refinement operation applied to F .
   256	Salient semantics mining and refinement. Salient identity cues
   257	are helpful to mitigate confusion from varying clothes and stabilize
   258	learning features robust to clothing variation. Following a similar
   259	design for learning identity-related content semantics, identity-related
   260	salient semantics can be mined with our SMR module by simply
   261	replacing average pooling with max pooling. We denote the extracted
   262	salient semantic feature as F s , and its max pooled feature f s and
   263	locally augmented feature fˆs are supervised by the loss Lsmr s (F s )
   264	(similar to Eq. 3) for identity-related salient semantics mining.
   265	Finally, refined salient semantic feature Fes can be obtained.
   266	C. Semantics Collaboration Framework
   267	
   268	TABLE I
   269	C OMPARISON OF OUR METHOD WITH STATE - OF - THE - ART METHODS ON PRCC
   270	AND LTCC. M ETHODS IN THE GRAY REGION USE EXTRA GROUND - TRUTH
   271	CLOTHING LABELS FOR TRAINING . “ SKETCH ”, “ SIL .”, AND “ POSE ” REPRESENT
   272	CONTOUR SKETCHES , SILHOUETTES , AND HUMAN POSES , RESPECTIVELY. M ETHODS
   273	MARKED WITH “∗” INVOLVE MULTIPLE TRAINING STAGES FOR EXTRA AUXILIARY
   274	NETWORKS . “S TANDARD ” AND “C LOTH -C HANGING ” MEAN THE STANDARD AND
   275	CLOTH - CHANGING SETTINGS , RESPECTIVELY. T HE BEST RESULTS ARE SHOWN IN
   276	BOLD .
   277	PRCC
   278	Methods
   279	
   280	Modality
   281	
   282	Cloth-Changing
   283	
   284	LTCC
   285	Standard
   286	
   287	Cloth-Changing
   288	
   289	Standard
   290	
   291	Rank-1
   292	
   293	mAP
   294	
   295	Rank-1
   296	
   297	mAP
   298	
   299	Rank-1
   300	
   301	mAP
   302	
   303	Rank-1
   304	
   305	mAP
   306	
   307	UCAD [25]
   308	CAL [14]
   309	DCR-ReID [24]
   310	AIM [15]
   311	CCFA∗ [16]
   312	
   313	RGB+sil.
   314	RGB
   315	RGB+sil.+sketch
   316	RGB
   317	RGB
   318	
   319	45.3
   320	55.2
   321	57.2
   322	57.9
   323	61.2
   324	
   325	55.8
   326	57.4
   327	58.3
   328	58.4
   329	
   330	96.5
   331	100
   332	100
   333	100
   334	99.6
   335	
   336	99.8
   337	99.7
   338	99.9
   339	98.7
   340	
   341	32.5
   342	40.1
   343	41.1
   344	40.6
   345	45.3
   346	
   347	15.1
   348	18.0
   349	20.4
   350	19.1
   351	22.1
   352	
   353	74.4
   354	74.2
   355	76.1
   356	76.3
   357	75.8
   358	
   359	34.8
   360	40.8
   361	42.3
   362	41.1
   363	42.5
   364	
   365	HA-CNN [29]
   366	PCB [3]
   367	IANet [2]
   368	TransReID [30]
   369	RCSANet∗ [31]
   370	ACID [32]
   371	FSAM [12]
   372	GI-ReID∗ [13]
   373	CAMC [10]
   374	
   375	RGB
   376	RGB
   377	RGB
   378	RGB
   379	RGB
   380	RGB
   381	RGB+pose+sil.
   382	RGB+sil.
   383	RGB+pose
   384	
   385	21.8
   386	41.8
   387	46.3
   388	46.6
   389	50.2
   390	55.4
   391	54.5
   392	33.3
   393	-
   394	
   395	38.7
   396	45.9
   397	44.8
   398	48.6
   399	66.1
   400	-
   401	
   402	82.5
   403	99.8
   404	99.4
   405	100
   406	100
   407	99.1
   408	98.8
   409	80.0
   410	-
   411	
   412	97.0
   413	98.3
   414	99.0
   415	97.2
   416	99.0
   417	-
   418	
   419	21.6
   420	23.5
   421	25.0
   422	34.4
   423	29.1
   424	38.5
   425	23.7
   426	36.0
   427	
   428	9.3
   429	10.0
   430	12.6
   431	17.1
   432	14.5
   433	16.2
   434	10.4
   435	15.4
   436	
   437	60.2
   438	65.1
   439	63.7
   440	70.4
   441	65.1
   442	73.2
   443	63.2
   444	73.2
   445	
   446	26.7
   447	30.6
   448	31.0
   449	37.0
   450	30.6
   451	35.4
   452	29.4
   453	35.3
   454	
   455	CSSC (Ours)
   456	
   457	RGB
   458	
   459	65.5
   460	
   461	63.0
   462	
   463	100
   464	
   465	99.1
   466	
   467	43.6
   468	
   469	18.6
   470	
   471	78.1
   472	
   473	40.2
   474	
   475	The SMR modules with average pooling and max pooling, denoted
   476	as SMR-C and SMR-S, are used for content and salient semantics
   477	mining and refinement, respectively. As illustrated in Fig. 2, SMR-C
   478	and SMR-S are alternatively employed in different orders within two
   479	branches. On the one hand, the refined content semantics Fec facilitate
   480	e
   481	mining salient semantics and result in Fecs . The losses involved in
   482	the first branch are formulated as follows:
   483	
   484	can lead to the overfitting problem and contribute to suboptimal
   485	performance. Therefore, we empirically find it better to introduce
   486	Ltri for model optimization after the first learning rate decay. During
   487	inference, the comprehensive semantics feature vector f cssc is used
   488	to compute the cosine distance between person images for retrieval.
   489	
   490	Lbranch1 = Lsmr c (F c ) + Lsmr s (Fecs ).
   491	
   492	III. E XPERIMENTS
   493	
   494	(5)
   495	
   496	On the other hand, the refined salient semantics Fes aid in mining
   497	e
   498	content semantics and result in Fesc . The losses involved in the second
   499	branch are formulated as follows:
   500	Lbranch2 = Lsmr s (F s ) + Lsmr c (Fesc ).
   501	
   502	(6)
   503	
   504	The cross-parallel interaction framework takes full advantage of
   505	mined content and salient semantics, promoting the robustness of
   506	Re-ID models in the cloth-changing scenario.
   507	Comprehensive semantics learning. The two features from the two
   508	branches specialized in content and salient semantics are fused to a
   509	comprehensive semantics feature as follows:
   510	e
   511	e
   512	F cssc = Conv(Fecs ⊕ Fesc ),
   513	
   514	(7)
   515	
   516	where ⊕ denotes element-wise addition, and Conv denotes a convolutional block. We take the max-pooled F cssc as the final Re-ID
   517	identity representation, which is supervised by commonly used Re-ID
   518	losses. Formally,
   519	f cssc = MaxPool(F cssc ) ∈ Rd ,
   520	
   521	(8)
   522	
   523	cssc
   524	
   525	(9)
   526	
   527	Lcssc = Lid (f
   528	
   529	) + Ltri (f
   530	
   531	cssc
   532	
   533	),
   534	
   535	where MaxPool denotes the global max pooling operation.
   536	
   537	Datasets. To demonstrate the effectiveness of our method, we evaluate it on three widely-used cloth-changing person Re-ID datasets,
   538	i.e., PRCC [9], LTCC [8], and Celeb-reID [21].
   539	Implementation details. Following previous works, we adopt
   540	ResNet-50 [22] pre-trained on ImageNet [23] and apply our SMR
   541	modules with P = 8 in the conv5 layer by reusing the three readymade convolutional blocks. Following [8], [14], [15], [24], [25], the
   542	input images are resized to 384 × 192. The batch size is set to 32.
   543	Random horizontal flipping, padding, random cropping, and random
   544	erasing [26] are used for data augmentation. Adam optimizer [27]
   545	with weight decay of 5×10−4 is adopted for 120 epochs. The learning
   546	rate linearly increases from 3 × 10−5 to 3 × 10−4 in the first 10
   547	epochs and decreases by a factor of 10 at the 30th and 60th epochs.
   548	All experiments are conducted on one NVIDIA GeForce GTX 1080
   549	Ti with 11GB of memory.
   550	Evaluation metrics. For evaluation, we adopt standard metrics
   551	as in most person Re-ID literature, namely Cumulative Matching
   552	Characteristic (CMC) curves and mean Average Precision (mAP). For
   553	LTCC and PRCC, we evaluate our method under both the standard
   554	setting and the cloth-changing setting following [28].
   555	B. Comparison with State-of-the-Art Methods
   556	
   557	D. Training and Inference
   558	Both content and salient semantics are mined and refined with
   559	the help of our proposed SMR module, and effectively collaborate to
   560	promote person Re-ID in our proposed CSSC framework. The overall
   561	loss is computed as follows:
   562	L = Lbranch1 + Lbranch2 + Lcssc .
   563	
   564	A. Experimental Settings
   565	
   566	(10)
   567	
   568	In the cloth-changing scenario, the appearance of pedestrians
   569	can vary significantly due to different clothing. Introducing Ltri
   570	prematurely when the Re-ID model is not robust to clothing variations
   571	
   572	Results on PRCC and LTCC. In Tab. I, we compare with advanced
   573	methods designed for conventional person Re-ID [2], [3], [29], [30]
   574	and methods tailored for cloth-changing person Re-ID using auxiliary
   575	modalities [10], [12], [13] and using DG-Net [32], [35]. CSSC
   576	with only RGB modality shows significant superiority over them,
   577	without relying on extra annotations or auxiliary modalities. Recently,
   578	some works have proposed to utilize ground-truth clothing labels to
   579	mitigate the impact of clothes [14]–[16], and some also use auxiliary
   580	modalities [24], [25], [36]. CSSC achieves state-of-the-art results on
   581	PRCC and exhibits competitive performance with them on LTCC. It
   582	
   583	TABLE II
   584	
   585	TABLE IV
   586	
   587	C OMPARISONS RESULTS ON C ELEB - RE ID. “ POSE ” REPRESENTS HUMAN POSES .
   588	M ETHODS MARKED WITH “†” ADOPT D ENSE N ET-121 AS THE BACKBONE .
   589	
   590	A BLATION OF THE CROSS - PARALLEL SEMANTICS COLLABORATION
   591	FRAMEWORK . W E REPORT THE RESULTS ON PRCC AND LTCC UNDER THE
   592	CLOTH - CHANGING SETTING . “SMR-C-S (SMR-S-C)” DENOTES SEQUENTIALLY
   593	USING SMR-C AND SMR-S (SMR-S AND SMR-C) IN ONE BRANCH .
   594	
   595	Methods
   596	
   597	Modality
   598	
   599	Rank-1
   600	
   601	Rank-5
   602	
   603	mAP
   604	
   605	PCB [3]
   606	MGN [17]
   607	CESD [8]
   608	ReIDCaps [21]
   609	IS-GANKL [33]
   610	RCSANet† [31]
   611	SirNet† [34]
   612	CAMC [10]
   613	
   614	RGB
   615	RGB
   616	RGB+pose
   617	RGB
   618	RGB
   619	RGB
   620	RGB
   621	RGB+pose
   622	
   623	37.1
   624	49.0
   625	50.9
   626	51.2
   627	54.5
   628	55.6
   629	56.0
   630	57.5
   631	
   632	57.0
   633	64.9
   634	66.3
   635	65.4
   636	70.3
   637	71.5
   638	
   639	8.2
   640	10.8
   641	9.8
   642	9.8
   643	12.8
   644	11.9
   645	14.2
   646	12.3
   647	
   648	CSSC (Ours)
   649	
   650	RGB
   651	
   652	64.5
   653	
   654	78.1
   655	
   656	17.3
   657	
   658	Branch 1
   659	
   660	Methods
   661	SMR-C
   662	1
   663	2
   664	3
   665	4
   666	5
   667	6
   668	
   669	SMR-S
   670	
   671	Branch 2
   672	
   673	SMR-C-S
   674	
   675	SMR-S-C
   676	
   677	SMR-S
   678	
   679	SMR-S-C
   680	
   681	✓
   682	✓
   683	✓
   684	
   685	✓
   686	✓
   687	✓
   688	✓
   689	
   690	Ours
   691	
   692	✓
   693	
   694	PRCC
   695	
   696	LTCC
   697	
   698	Rank-1
   699	
   700	mAP
   701	
   702	Rank-1
   703	
   704	mAP
   705	
   706	57.8
   707	61.7
   708	60.3
   709	63.7
   710	62.0
   711	63.6
   712	
   713	55.8
   714	58.3
   715	58.8
   716	60.3
   717	58.7
   718	60.9
   719	
   720	39.3
   721	40.6
   722	41.1
   723	41.6
   724	41.8
   725	41.3
   726	
   727	16.4
   728	17.1
   729	17.8
   730	18.1
   731	17.4
   732	18.2
   733	
   734	65.5
   735	
   736	63.0
   737	
   738	43.6
   739	
   740	18.6
   741	
   742	TABLE III
   743	A BLATION STUDIES OF THE SMR MODULE . W E REPORT THE RESULTS ON PRCC
   744	AND LTCC UNDER THE CLOTH - CHANGING SETTING . “L OCAL .” AND “R EFINE .”
   745	DENOTE LOCAL SEMANTICS MINING AND SEMANTICS REFINEMENT IN SMR.
   746	PRCC
   747	
   748	Methods
   749	
   750	(a)
   751	
   752	Query
   753	
   754	Rank-1 -------------------------------------------------------------------------------------------------------> Rank-10
   755	
   756	LTCC
   757	
   758	Rank-1
   759	
   760	mAP

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf' - | nl -ba | sed -n '49,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    49	1. Introduction
    50	
    51	achieved by ‖𝑥‖2 , which is the modulus of the one-dimensional vector
    52	after the feature ﬂattening of the image). It is highly consistent with our
    53	cognition because in the classiﬁcation task, no matter what kind of data
    54	distribution (such as long-tail distribution, etc.), the modulus length
    55	of the image feature vectors will be compressed during the training
    56	process.
    57	Based on this assumption, we conducted experiments on the Re-ID
    58	task and face recognition to observe the relationship between feature
    59	magnitude and accuracy. Fig. 1(a) and (b) show the variation of the
    60	average magnitude and accuracy of images in the VGG2FP dataset with
    61	respect to the training epochs under our baseline method [4]; (c) and (d)
    62	show the variation of the average magnitude and retrieval accuracy of
    63	images in the Market-1501 dataset with respect to the training epochs
    64	under our baseline method [5]. It can be seen that with the progress
    65	of training, the epoch increases continuously, the magnitudes of images
    66	decrease, and the classiﬁcation accuracy improves. This conﬁrms that
    67	the connotation behind the picture is the feature magnitude. Therefore,
    68	we claim that classiﬁcation accuracy is related to the feature magnitudes
    69	of the model outputs in Re-ID and face recognition.
    70	
    71	Face recognition technology has been widely applied in the real
    72	world because facial features are the best indicator of a person. Because
    73	sometimes the cameras cannot capture the front side pedestrians, person Re-Identiﬁcation (Re-ID) [1,2] technology is a signiﬁcant auxiliary
    74	to capture pedestrians through diﬀerent cameras. Although the Re-ID
    75	task, as we know it, is more similar to the image retrieval task, the representative features are extracted from the original image data during
    76	the training process. Then the ID or attribute of the pedestrian is used
    77	as the training label to train the model, which is also transformed into
    78	a ﬁne-grained classiﬁcation task.
    79	For a face picture, the facial features have ﬁxed relative positions,
    80	and we can use tools like MTCNN [3] to easily align the face. However, pictures of persons show various poses and diﬀerent views since
    81	their arms and legs are not ﬁxed. In addition, people walk in diﬀerent
    82	directions at various angles, which also makes pictures of persons differently. What is the connotation underlying these pictures? Our answer
    83	is the feature vector magnitude (for a feature vector 𝑥, its magnitude is
    84	
    85	∗ Corresponding author.
    86	
    87	E-mail address: yangx@xidian.edu.cn (X. Yang).
    88	https://doi.org/10.1016/j.patcog.2025.112443
    89	Received 6 June 2024; Received in revised form 9 September 2025; Accepted 10 September 2025
    90	Available online 13 September 2025
    91	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    92	
    93	Pattern Recognition 172 (2026) 112443
    94	
    95	X. Yang et al.
    96	
    97	Fig. 1. The variation trends of feature magnitude and accuracy during the training process. (a) The average feature magnitude of the images during the training
    98	process on the VGG2FP dataset with naive softmax loss. (b) Accuracy change curve during the training process on the VGG2FP dataset with naive softmax loss. (c)
    99	The average feature magnitude of the images during the training process on the Market-1501 dataset with naive softmax loss. (d) Rank-1 change curve during the
   100	training process on the Market-1501 dataset with naive softmax loss.
   101	
   102	In the Re-ID task’s dataset, there is a small gap in the number
   103	of pictures of each type of pedestrian, so researchers often use Naive
   104	softmax as the internal calculation method of cross entropy classiﬁcation loss. In the data set of face recognition tasks, there is a particular gap in the number of images of each type of face, and the data
   105	balance ability is relatively poor, so L2 normalization is commonly
   106	used by researchers to resist balance. But in any case, both of these
   107	loss calculation methods are to reduce the length of the characteristic mode. We can therefore claim that in face recognition and ReID tasks, the feature magnitudes should be fully utilized to train the
   108	model.
   109	Furthermore, Fig. 1(d) shows the relationship between classiﬁcation accuracy (rank-1) and epochs in the Market-1501 dataset under our baseline method [5]. With the increase of epochs, the overall accuracy is improved. But we can see that in the ﬁrst 30 epochs,
   110	there is a decline in rank-1. At the same time, when rank-1 decreases, image feature magnitude increases in the reverse direction.
   111	Therefore, we believe that the strong inverse correlation between feature magnitude and classiﬁcation accuracy further validates the importance of the magnitude of feature vectors in image classiﬁcation.
   112	However, we also found that the compression process of feature magnitude experiences unstable phases, which can result in the magnitude not being reduced to an appropriate size at certain stages,
   113	thereby aﬀecting classiﬁcation performance. Therefore, it is particularly important to design a classiﬁer that can stably reduce the
   114	magnitude of feature vectors in face recognition and person Re-ID
   115	tasks.
   116	In the training process of neural networks, the weight parameters are
   117	updated by calculating the gradient of the loss function. Based on our
   118	analysis (details in Section 3.4), the tangential gradients of the Naive
   119	Softmax loss and L2 Softmax loss usually ﬂuctuate. In other words, the
   120	tangential gradient is not monotonically decreasing. This means that the
   121	Naive Softmax loss could sometimes provide larger gradients in later iterations, leading the network to a local minimum. To avoid this prob-
   122	
   123	lem, we need a loss function that can provide an ideal monotonically
   124	decreasing tangential gradient.
   125	In this paper, we propose a novel softmax loss formulation named
   126	Condense loss, which takes the advantage of feature and weight vector magnitudes in Re-ID tasks to give a monotonically decreasing coeﬃcient of a tangential gradient. Diﬀerent from the Naive and L2 normalized softmax losses which are in inner product formulation between
   127	normalized or un-normalized feature and weight, it takes the negative
   128	Euclidean distance as the formulation. When updating weight parameters, it provides negative radial gradients, hence, the feature vector magnitudes keep reducing during the training process. Furthermore, Condense loss provides a larger tangential gradient at the beginning of the
   129	training process, and a smaller tangential gradient at the end, with the
   130	change in feature vector magnitudes. After training, both the magnitude
   131	of the feature vectors and the angles between the feature vectors and the
   132	class centers decrease, so we name this loss function “Condense Loss”.
   133	And we also introduce a margin parameter into Condense loss to enlarge
   134	the inter-class distances. The mathematical analysis will be given in
   135	Section 3.
   136	Our contributions are summarized as follows:
   137	•
   138	
   139	We propose Condense loss with negative Euclidean distance formulation for Re-ID tasks, which utilizes the vector magnitudes to provide
   140	an adaptive gradient during the training phase.
   141	• We introduce a margin parameter into Condense loss to enlarge the
   142	inter-class distances, which helps the model learn more discriminative features.
   143	• We give the mathematical analysis to show why Condense loss can
   144	work and prove its superiority in Re-ID tasks when compared with
   145	other softmax loss variants.
   146	• We conduct extensive experiments on two Re-ID benchmarks and
   147	face recognition tasks, which demonstrates the eﬀectiveness of Condense loss. When compared with the state-of-the-art methods, Condense loss achieves competitive results.
   148	2
   149	
   150	Pattern Recognition 172 (2026) 112443
   151	
   152	X. Yang et al.
   153	
   154	2. Related work
   155	
   156	batches, and it has been widely adopted by many Re-ID methods. Chen
   157	et al. [8] improved triplet loss by introducing a hard negative sample.
   158	Therefore, we devote ourselves to researching classiﬁcation loss.
   159	
   160	2.1. Person Re-identiﬁcation
   161	
   162	2.2. Modiﬁed softmax losses
   163	
   164	Because Re-ID is a complex feature learning task, many works focus
   165	on learning and extracting highly discriminative features from pictures
   166	or videos. The methods based on metric learning such as Contrastive
   167	loss [6], Triplet loss [7] and Quadruplet loss [8] have been developed
   168	quite mature.
   169	Part-level feature learning frameworks are pioneering methods that
   170	signiﬁcantly improve Re-ID performance. Wu et al. [9] introduced a
   171	deep attention-based comparator module to learn relative parts of paired
   172	pictures and produce relevant features to improve the part-based Re-ID
   173	methods. In the meantime, the attention mechanism has been proved
   174	to be beneﬁcial for Re-ID tasks since it makes the network focus on the
   175	human body. Zhang et al. [10] introduced a relation-aware global attention module to learn the global feature of person pictures. Rao et
   176	al. [11] proposed to learn the attention with counterfactual causality,
   177	which provides a tool to measure the attention quality and a powerful supervisory signal to guide the learning process. Li et al. [1] introduced MCTN, which integrates multi-granularity feature learning with
   178	pyramidal cross attention mechanisms to enhance local-global feature
   179	relationships.
   180	With the further development of the Re-ID task, more and more researchers have switched from the CNN framework to the Transformer
   181	framework in terms of network architecture for feature extraction. He et
   182	al. [12] proposed the ﬁrst feature extraction work that fully adopts the
   183	Transformer framework. The side information embeddings (SIE) are introduced to mitigate feature bias towards camera/view variations; This
   184	work strongly supports Re-ID subtasks. Based on this framework, Luo et
   185	al. [13] and Chen et al. [14] proposed diﬀerent self-supervision modes
   186	to provide additional prior knowledge support for Re-ID tasks, thereby
   187	improving the recognition accuracy from another perspective. A viewdecoupled transformer is developed by Zhang et al. [15] to decouple
   188	view and identity features for aerial-ground person Re-ID through hierarchical separation mechanisms.
   189	Recently, the unsupervised person Re-Id task with more robust generalization performance has been further studied. Zheng et al. [16] proposed a label transfer algorithm that simultaneously uses pseudo labels
   190	to train the data while reﬁning the pseudo labels as an online clustering algorithm. To address cross-camera domain variations, Ran et al.
   191	[17] developed CGMAL, which leverages heterogeneous graph learning and adversarial training to conduct multi-domain feature transfer
   192	across camera domains. Meanwhile, Chen et al. [18] tackled the reliability issue of Jaccard distance by proposing CA-Jaccard distance, where
   193	camera-aware neighbor selection and query expansion mechanisms are
   194	designed to mitigate the negative impact of camera variation on distance
   195	metric computation.
   196	Nowadays, Re-ID tasks still face a few problems in realistic scenarios, such as cross-resolution problems, cross-modality problems, etc. Li
   197	et al. [19] designed a discriminative semi-coupled projective dictionary model to learn multiple pairs of dictionaries and multiple mapping functions in variational resolution gap scenarios to address the
   198	cross-resolution problems. Peng et al. [20] introduced MRLReID, which
   199	employs resolution estimation as an auxiliary task to enhance crossresolution person re-identiﬁcation through multi-task learning. Wei
   200	et al. [21] proposed a novel syncretic modality collaborative learning (SMCL) model to bridge the cross-modality gap. Multi-granularity
   201	frameworks (DMANet and CM2GT) are proposed by Cheng et al. [22]
   202	and Feng et al. [23] respectively to tackle cross-modality discrepancies,
   203	which achieved advanced results.
   204	However, few methods focus on designing speciﬁc losses for Re-ID
   205	tasks, most of which are metric learning losses rather than classiﬁcation losses. Siamese loss and triplet loss are two common metric learning losses; however, directly using them cannot achieve satisfactory results. Hermans et al. [24] proposed a PK sampling strategy to learn hard
   206	
   207	The modiﬁed softmax losses are usually used in face recognition
   208	tasks. The angular margin in the form of multiplication was ﬁrst proposed in L-Softmax [25], and Normface [26] was a pioneer work that
   209	normalizes the feature and weight of the naive softmax loss. Shortly
   210	afterwards A-Softmax (Sphereface) [27] combined two advantages of LSoftmax loss and Normface, normalizing the weight and giving an angular margin in the form of multiplication. AM-Softmax [28] and CosFace
   211	[29] normalized both feature and weight and changed the multiplication angular margin as minus cosine margin. Soon after, ArcFace [4]
   212	changed the minus cosine margin of CosFace to an additive angular margin. More recently, ElasticFace [30] replaced ﬁxed margins with adaptive elastic margins that adjust based on class separability. X2-Softmax
   213	[32] further introduced quadratic function-based adaptive margins that
   214	automatically scale with inter-class angles.
   215	There are two methods in Re-ID tasks that replicate the success of
   216	Normface. SPhereReID [34] ﬁrst normalized the feature and weight in
   217	Re-ID tasks and HSME [35] was proposed to solve the cross modality
   218	Re-ID problems. Recently, Circle Loss [36] uniﬁed the softmax loss and
   219	triplet loss, and thus achieved great success in many areas. Notably,
   220	changing the negative integer factors added in the process of backpropagation in Circle loss can be converted into all kinds of softmax loss,
   221	which is highly innovative with similar complexity.
   222	3. Proposed method
   223	In this section, we ﬁrst introduce the proposed Condense loss in detail. And a mathematical analysis comparison with other classiﬁcation
   224	losses is presented to show the superiority of Condense loss.
   225	3.1. Condense loss
   226	The naive softmax loss is the most widely used classiﬁcation loss in
   227	Re-ID tasks. It is usually combined with cross-entropy loss, which can
   228	be presented as:
   229	1 ∑
   230	𝑒𝑓𝑖,𝑦𝑖
   231	𝑙𝑜𝑔 ∑𝐾
   232	,
   233	𝑁 𝑖=1
   234	𝑒𝑓𝑖,𝑘
   235	𝑁
   236	
   237	𝐿𝐶𝐸 = −
   238	
   239	𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 + 𝑏𝑘 ,
   240	
   241	(1)
   242	
   243	𝑘=1
   244	
   245	where 𝑥𝑖 ∈ ℝ𝑙 denotes the 𝑙 dim feature of the 𝑖th sample, 𝑦𝑖 denotes
   246	the one-hot label coding of the 𝑖th sample, 𝑊𝑘 denotes the 𝑘th column
   247	of weight 𝑊 ∈ ℝ𝑙×𝐾 and 𝑏𝑘 is the bias parameter. 𝑁 and 𝐾 denote the
   248	size of the mini-batch and the number of classes in the training dataset,
   249	respectively. The formulation of 𝑓𝑖,𝑘 is the classiﬁcation function. For
   250	simplicity, we ﬁx the bias 𝑏 as 0, and thus 𝑓𝑖,𝑘 can be rewritten as:
   251	‖‖ ‖
   252	𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 = ‖
   253	‖𝑊𝑘 ‖‖𝑥𝑖 ‖cos𝜃𝑘 ,
   254	
   255	(2)
   256	
   257	where 𝜃𝑘 is the angle between the weight 𝑊𝑘 and feature 𝑥𝑖 . L2 normalized softmax losses usually normalize 𝑊 or 𝑥 and set margin parameters in 𝑓𝑖,𝑘 function to give clear boundaries between diﬀerent classes.
   258	Normface normalizes both 𝑊 and 𝑥, and sets a scale parameter 𝑠 to converge the network; its formulation is 𝑓𝑛𝑜𝑟𝑚𝑓 𝑎𝑐𝑒 = 𝑠 cos𝜃𝑘 . Althongh classiﬁcation functions 𝑓𝑖,𝑘 focus exclusively on classiﬁcation, the feature
   259	extracted during the testing phase may not be discriminative enough.
   260	To solve this problem, Sphereface and CosFace set margins to 𝑓𝑖,𝑘 ; their
   261	formulations are 𝑓𝑠𝑝ℎ𝑒𝑟𝑒𝑓 𝑎𝑐𝑒 = 𝑠 cos(𝑚𝜃𝑘 ) and 𝑓𝑐𝑜𝑠𝑓 𝑎𝑐𝑒 = 𝑠 (cos𝜃𝑘 − 𝑚), respectively.
   262	However, all these functions are inner-product based; we show their
   263	gradients which are updating directions of class weight 𝑊𝑦𝑖 in Fig. 2(a).
   264	The gradient direction of 𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 is the same with vector 𝑥⃗𝑖 and the
   265	⃗ 𝑦 (the lengh of 𝑊
   266	⃗𝑦
   267	gradient direction of 𝑓𝑖,𝑘 = 𝑠 cos𝜃𝑘 is vertical to 𝑊
   268	𝑖
   269	
   270	3
   271	
   272	𝑖
   273	
   274	Pattern Recognition 172 (2026) 112443
   275	
   276	X. Yang et al.
   277	
   278	Fig. 2. (a) The updated gradient directions of loss functions. Blue, green and red lines represent the gradient directions of 𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 , 𝑓𝑖,𝑘 = 𝑠 cos𝜃𝑘 and the proposed
   279	Condense loss, respectively. The light gray dashed line serves as a reference, which includes the update directions under the above three functions, while the dark
   280	gray arrow indicates the target direction of parameter updates during model optimization. (b) The geometry illustration of margin in the Condense loss. 𝑚 denotes
   281	the margin parameter. For interpretation of the references to color in this ﬁgure legend, the reader is referred to the web version of this article.
   282	
   283	has no eﬀect on cos𝜃𝑘 ). Since our aim is to pull 𝑊𝑦𝑖 closer to 𝑥𝑖 , why not
   284	⃗ 𝑦 )? Naturally, we come up with a
   285	give it a gradient direction in (⃗
   286	𝑥𝑖 − 𝑊
   287	
   288	3.2. Condense loss in Re-ID
   289	
   290	new 𝑓 formulation, which can be presented as:
   291	
   292	The feature and weight vector magnitudes are beneﬁcial to training
   293	Re-ID models, so we propose Condense loss which can utilize them to
   294	provide monotonically decreasing tangential gradients when training a
   295	Re-ID model.
   296	Since Condense loss does not care what backbone is adopted, we
   297	only talk about how to compute Condense loss here. Usually, we input the training pictures into CNNs to learn the features. For instance,
   298	ResNet50 is the most widely used backbone. In most conditions, the last
   299	fully-connected layer of ResNet50 will be removed and replaced with
   300	a global average pooling (GAP) layer and a linear layer. To compute
   301	the Condense loss, we remove the linear layer and only use the features
   302	extracted from the GAP layer. As shown in Fig. 3, supposing the size of
   303	a mini-batch feature is (𝑏𝑠, 𝑑𝑖𝑚), we initialize a weight tensor with size
   304	(𝑛𝑢𝑚, 𝑑𝑖𝑚) by nn.Parameter(). By computing the negative Euclidean distance between the feature tensor and the weight tensor, a scores tensor
   305	with size (𝑏𝑠, 𝑛𝑢𝑚) is achieved. With the assistance of the label tensor,
   306	we initialize a margin tensor with size (𝑏𝑠, 𝑛𝑢𝑚). By adding the negative
   307	margin tensor, we can ﬁnally achieve Condense scores tensor, which
   308	will be fed into the cross-entropy loss.
   309	Further, since the Condense loss is a classiﬁcation loss, it is capable of
   310	being jointly trained with some other metric learning losses, e.g., triplet
   311	loss, which can further improve the Re-ID performance.
   312	
   313	𝑖
   314	
   315	𝑓𝑖,𝑘 = −(𝑊𝑘 − 𝑥𝑖 )2 ,
   316	
   317	(3)
   318	
   319	and we initially deﬁne the Condense loss as:
   320	1 ∑
   321	𝑒−(𝑊𝑦𝑖 −𝑥𝑖 )
   322	𝑙𝑜𝑔 ∑𝐾
   323	.
   324	2
   325	𝑁 𝑖=1
   326	𝑒−(𝑊𝑘 −𝑥𝑖 )
   327	𝑁
   328	
   329	𝐿𝐶𝑜𝑛𝑑𝑒𝑛𝑠𝑒 = −
   330	
   331	2
   332	
   333	(4)
   334	
   335	𝑘=1
   336	
   337	Note that Fig. 2(a) only illustrates the direction of parameter up⃗ 𝑦 and 𝑥⃗𝑖 . The change
   338	dates, whose objective is to align the direction of 𝑊
   339	𝑖
   340	in feature magnitude is quantitatively analyzed in Section 3.4.
   341	As mentioned before, 𝐿𝐶𝑜𝑛𝑑𝑒𝑛𝑠𝑒 cannot make the network learn
   342	enough discriminative features because it only cares about the classiﬁcation accuracy. Inspired by the cosine margin in L2 normalized softmax
   343	losses, we introduce a Euclidean margin to tackle this problem.
   344	As shown in Fig. 2(b), imagine a binary classiﬁcation (Class 𝐶𝑖 ,
   345	𝑖 = 1, 2) task, there are two weight vectors, i.e., 𝑊1 and 𝑊2 , and the
   346	learned feature vector 𝑉 belongs to 𝐶2 . To correctly classify 𝑉 to 𝐶2 , the
   347	Condense loss needs to make −(𝑊2 − 𝑉 )2 > −(𝑊1 − 𝑉 )2 . We can make
   348	′
   349	this more diﬃcult by setting a margin as −(𝑊2 − 𝑉 )2 − 𝑚 > −(𝑊1 − 𝑉 )2 .
   350	It is easy to understand that if we want to keep the inequality true,
   351	′
   352	the value of −(𝑊2 − 𝑉 )2 should be larger than before, which means the
   353	′
   354	
   355	value of 𝑊2 − 𝑉 will be smaller. We artiﬁcially increase the diﬃculty of
   356	updating the weight 𝑊1 gradient by increasing the distance between 𝑊2
   357	and 𝑉 ; hence, the Condense loss makes the features more discriminative
   358	by introducing a margin.
   359	Finally, we deﬁne the margin version of Condense loss as:
   360	1 ∑
   361	𝑒−(𝑊𝑦𝑖 −𝑥𝑖 ) −𝑚
   362	𝑙𝑜𝑔
   363	,
   364	∑
   365	2
   366	−(𝑊𝑦𝑖 −𝑥𝑖 )2 −𝑚
   367	𝑁 𝑖=1
   368	𝑒
   369	+ 𝐾 𝑒−(𝑊𝑘 −𝑥𝑖 )
   370	𝑁
   371	
   372	𝐿𝐶𝑜𝑛𝑑𝑒𝑛𝑠𝑒 = −
   373	
   374	3.3. Qualitative analysis
   375	We give a qualitative analysis of softmax function 𝑓 , i.e., Naive softmax function 𝑓𝑛𝑎𝑖𝑣𝑒 , L2 normalized softmax function 𝑓𝑛𝑜𝑟𝑚 and Condense
   376	function 𝑓𝐶𝑜𝑛 , which are re-written as:
   377	
   378	2
   379	
   380	(5)
   381	
   382	𝑘≠𝑦𝑖
   383	
   384	where 𝑚 is the margin parameter.
   385	
   386	‖‖ ‖
   387	𝑓𝑛𝑎𝑖𝑣𝑒 = ‖
   388	‖𝑊𝑘 ‖‖𝑥𝑖 ‖cos𝜃𝑘 ,
   389	
   390	(6)
   391	
   392	𝑓𝑛𝑜𝑟𝑚 = 𝑐𝑜𝑠𝜃𝑘 ,
   393	
   394	(7)
   395	
   396	‖‖ ‖
   397	‖ ‖2 ‖ ‖2
   398	𝑓𝐶𝑜𝑛 = 2‖
   399	‖𝑊𝑘 ‖‖𝑥𝑖 ‖cos𝜃𝑘 − ‖𝑊𝑘 ‖ − ‖𝑥𝑖 ‖ ,
   400	
   401	(8)
   402	
   403	Fig. 3. The ﬂowchart of computing Condense loss in Re-ID tasks. Images are extracted from the feature extraction network. 𝑏𝑠 denotes the batch size, 𝑑𝑖𝑚 denotes
   404	the feature dimension, 𝑛𝑢𝑚 denotes the number of classes and 𝑚 denotes the margin parameter (the 𝑚 is positive in the ﬁgure but we need −𝑚 to make features
   405	more discriminative). ⊖ denotes the operation of computing negative Euclidean distance obtaining 𝑓 = −(𝑊 − 𝑥)2 and ⊕ denotes the add operation obtaining
   406	𝑓 = −(𝑊 − 𝑥)2 − 𝑚. Colors mean diﬀerent values and white means the value is 0.
   407	4
   408	
   409	Pattern Recognition 172 (2026) 112443
   410	
   411	X. Yang et al.
   412	
   413	𝜕𝑓𝐶𝑜𝑛
   414	‖ ‖ ‖
   415	because of the initialize manner of weight, ‖
   416	‖𝑊𝑖 ‖ ≪ ‖𝑥𝑖 ‖, 𝑅( 𝜕𝑥⃗ ) is in
   417	
   418	For simplicity, we omit the margin parameter. To optimize the crossentropy loss, the network needs to make softmax functions 𝑓 larger.
   419	The naive softmax function has two choices, 1) increase the value of
   420	‖𝑊𝑘 ‖‖𝑥𝑖 ‖ ↑, 2) increase the value of 𝑐𝑜𝑠𝜃𝑘 ↑ (i.e., decrease the value of
   421	‖ ‖‖ ‖
   422	𝜃𝑘 ↓). Hence, the naive softmax function sometimes would make the network learn larger feature vector magnitude instead of reducing angles.
   423	The L2 normalized softmax function can solve this problem by making the network focus on optimizing angles with weight and feature
   424	normalization. In contrast, the Condense function fully utilizes the vector magnitudes. To increase the value of 𝑓𝐶𝑜𝑛𝑑𝑒𝑛𝑠𝑒 , the network would
   425	‖
   426	‖ ‖
   427	‖ ‖‖ ‖
   428	seek to reduce ‖
   429	‖𝑊𝑘 ‖ ↓ and ‖𝑥𝑖 ‖↓, which means the value of ‖𝑊𝑘 ‖‖𝑥𝑖 ‖ ↓
   430	would also decline. Consequently, Condense loss reinforces the network
   431	to learn smaller angle 𝜃𝑘 ↓.
   432	
   433	𝜕𝐿𝑡
   434	
   435	𝜕 𝑊𝑖
   436	
   437	that during the initial stages of training, the Condense loss can provide
   438	larger tangential gradients, facilitating faster parameter updates. As the
   439	network approaches convergence, the reduction in tangential gradients
   440	leads to smaller update magnitudes, thereby enhancing the stability of
   441	the training process. This mechanism ensures that after rapid learning
   442	in the early stages, the network can achieve smooth parameter updates
   443	as it nears convergence. Thus, Condense loss ideally provides monotonically decreasing tangential gradients. Note that, because we usually
   444	use mini-batch strategy to optimize the model, the gradients won’t decrease monotonously. However, Condense loss makes the model converge faster.
   445	
   446	3.4. Quantitative analysis
   447	We give a quantitative analysis of the naive softmax function and
   448	L2 normalized softmax function. The naive softmax functions and the
   449	⃗ 𝑖 , 𝑥⃗𝑖 can be written as:
   450	gradients of 𝑊
   451	𝜕𝑓𝑛𝑎𝑖𝑣𝑒
   452	
   453	‖‖ ‖
   454	𝑓𝑛𝑎𝑖𝑣𝑒 = ‖
   455	‖𝑊𝑘 ‖‖𝑥𝑖 ‖cos𝜃𝑘 ,
   456	
   457	⃗𝑖
   458	𝜕𝑊
   459	
   460	𝜕𝑓𝑛𝑎𝑖𝑣𝑒
   461	
   462	= 𝑥⃗𝑖 ,
   463	
   464	𝜕 𝑥⃗𝑖
   465	
   466	⃗ 𝑖,
   467	=𝑊
   468	
   469	4. Experiments
   470	
   471	(9)
   472	
   473	4.1. Datasets and evaluation orotocol
   474	
   475	since the network is to learn the weight parameter, we decompose
   476	𝜕𝑓
   477	‖
   478	̂
   479	the 𝑛𝑎𝑖𝑣𝑒
   480	into radial and tangential directions, i.e., ‖
   481	‖𝑥𝑖 ‖𝑐𝑜𝑠𝜃 𝑊𝑖 and
   482	⃗
   483	
   484	Re-ID. We utilize Market-1501 [38] and MSMT17 [39] for Re-ID
   485	tasks. Market-1501 is collected by 6 cameras and contains 1,501 pedestrians; there are 12,936 pictures of 751 pedestrians used for training, 19,732 pictures of 750 pedestrians used for testing. MSMT17 is
   486	the largest dataset in the Re-ID community which contains 4,101 and
   487	126,411 pictures captured with 15 cameras.
   488	Face Veriﬁcation. We have also done some experiments on face
   489	veriﬁcation tasks to demonstrate the eﬀectiveness of Condense loss.
   490	We train the model with MS1MV2 [4] which is a semi-automatic reﬁned version of the MS-Celeb-1M dataset [40]. For evaluation, we adopt
   491	LFW [41], AgeDB30 [43], CFP-FP [44], CALFW [45], CPLFW [46] and
   492	VGG2FP [47].
   493	Evaluation Protocol. We select the cumulative matching characteristic (CMC) and mean Average Precision (mAP) to evaluate the performance of Re-ID tasks. The oﬃcial protocol of face veriﬁcation tasks is
   494	adopted.
   495	
   496	𝜕 𝑊𝑖
   497	
   498	‖𝑥𝑖 ‖(𝑥̂𝑖 − 𝑐𝑜𝑠𝜃 𝑊̂ 𝑖 ) (𝑊̂ 𝑖 and 𝑥̂𝑖 are unit vectors of 𝑊
   499	⃗ 𝑖 and 𝑥⃗𝑖 , respectively).
   500	‖ ‖
   501	𝜕𝑓𝑛𝑎𝑖𝑣𝑒
   502	‖
   503	Similarly, the radial direction of 𝜕𝑥⃗
   504	is ‖𝑊𝑖 ‖
   505	‖𝑐𝑜𝑠𝜃 𝑥̂ 𝑖 , because cos 𝜃
   506	𝑖
   507	
   508	ranges from (−1, 1), the magnitude of 𝑥⃗𝑖 would increase or decrease
   509	based on the value of cos 𝜃, which imports volatility into the training
   510	‖
   511	̂
   512	process when optimizing the ‖
   513	‖𝑥𝑖 ‖𝑐𝑜𝑠𝜃 𝑊𝑖 .
   514	The L2 normalized softmax functions and their gradients can be written as:
   515	𝑓𝑛𝑜𝑟𝑚 = 𝑐𝑜𝑠𝜃𝑘 ,
   516	
   517	𝜕𝑓𝑛𝑜𝑟𝑚
   518	⃗𝑖
   519	𝜕𝑊
   520	𝜕𝑓𝑛𝑜𝑟𝑚

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Confidence guided semi-supervised cross-modality person re-identification.pdf' - | nl -ba | sed -n '49,850p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 144ms:
    49	1. Introduction
    50	With the development of public video surveillance systems, modern
    51	devices increasingly operate in dual modes: capturing visible-light images during daytime and automatically switching to the infrared mode
    52	at night. This technological evolution has propelled visible-infrared
    53	person re-identification (VI-ReID) as a critical research frontier, aiming
    54	to match pedestrian images across heterogeneous modalities. Specifically, VI-ReID seeks to retrieve the same identity from infrared/visible
    55	galleries given a visible/infrared query image. Compared with conventional single-modality ReID, this cross-modality matching faces greater
    56	challenges due to significant modality discrepancies in image characteristics. Moreover, the need of large amount of labeled cross-modality
    57	data further compounds these difficulties, as manual annotation of
    58	paired visible-infrared identities remains labor-intensive and costly.
    59	In recent years, numerous methods have been proposed for the VIReID task. These methods are broadly divided into supervised and unsupervised VI-ReID, based on the utilization of labeled cross-modality
    60	
    61	training data. Supervised VI-ReID aims to improve the performance
    62	of cross-modality matching by learning discriminative features from
    63	extensive labeled cross-modality datasets. For example, FMCNet [1]
    64	alleviates inter-modal divergence by enhancing feature discrimination
    65	through feature-level modality compensation. In [2], the modalityspecific memory network is proposed to overcome the loss of discriminative modality-specific information, further reducing the modality
    66	discrepancy. In [3], Zhang et al. proposed a dual-semantic consistency learning network, which can alleviate the modality difference
    67	by reducing the channel consistency. In [4], Chen et al. proposed a
    68	consistency-driven feature scoring and regularization network (CFSRNet) for VI-ReID to enhance the training stability. While effective, these
    69	methods require extensive cross-modality annotations. Compared with
    70	the single-modality person ReID, it is more expensive and difficult to
    71	annotate such a large number of cross-modality samples for VI-ReID.
    72	This largely limits the scalability of supervised methods in real-world
    73	scenarios.
    74	
    75	∗ Corresponding author.
    76	
    77	E-mail address: xpchen@henu.edu.cn (X. Chen).
    78	https://doi.org/10.1016/j.patcog.2025.111669
    79	Received 14 January 2024; Received in revised form 31 March 2025; Accepted 31 March 2025
    80	Available online 8 April 2025
    81	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    82	
    83	Pattern Recognition 165 (2025) 111669
    84	
    85	X. Zhu et al.
    86	
    87	2. Related work
    88	
    89	Different from supervised VI-ReID methods, unsupervised VI-ReID
    90	trains the model by mining the potential information of unlabeled
    91	samples. For example, Liang et al. [5] proposed a two-stage learning
    92	approach, termed H2H, including unsupervised intra-modality selflearning stage and intra-modality heterogeneous learning stage. However, H2H is complex and must use the extra Market-1501 dataset as
    93	labeled RGB samples for pre-training, which reduces the scalability of
    94	this method in actual deployment. To alleviate the difficulties in generating reliable cross-modality labels and learn robust modality-invariant
    95	features without any annotations, an Augmented Dual-Contrastive Aggregation (ADCA) learning framework is proposed in [6], which utilizes an enhanced double contrast aggregation module to capture the
    96	complementary information on visible-infrared pedestrian features.
    97	The aforementioned unsupervised methods can partially mitigate
    98	the issue of data labeling costs. However, the performance of these
    99	unsupervised methods still shows a significant gap compared to that of
   100	supervised VI-ReID methods. For example, on SYSU-MM01 dataset, the
   101	Rank-1 accuracy of the top-performing supervised method is 80.83%
   102	[7], while the performance of the most advanced unsupervised method
   103	is only 45.51% [6]. The significant performance gap may be caused by
   104	the inherent large inter-modality discrepancy in the unlabeled crossmodality data itself. Furthermore, the process of generating crossmodality data with pseudo-labels usually lacks the guidance of real
   105	labels, resulting in the limited reliability of the generated pseudo-labels,
   106	which affects the performance of the learned model.
   107	As a compromise between unsupervised and supervised learning
   108	paradigms, semi-supervised learning strategy not only reduces dependence on labeled data but also effectively leverages limited annotated
   109	samples to guide the exploitation of unlabeled data, thereby improving
   110	model reliability. In recent years, this methodology has demonstrated
   111	remarkable success in single-modality person re-identification tasks.
   112	Huang et al. [8] developed the Multi-pseudo Regularized Label (MpRL)
   113	mechanism to enhance pseudo-label quality by assigning appropriate virtual labels to generated data samples. Furthermore, in order
   114	to reduce minimize the annotation cost, Hu et al. [9] proposed a
   115	Multi-grained Active Semi-Supervised learning framework (MASS) incorporating a cluster-scatter procedure and purification loss for optimized label efficiency. Chen et al. [10] proposed a semi-supervised
   116	ReID framework, which leverages reliable sample selection to improve
   117	unlabeled data utilization.
   118	However, the aforementioned methods assume that all images originate from visible modality, and thus cannot be directly employed
   119	to effectively tackle the problem of VI-ReID due to the significant
   120	divergence between the visible and infrared modalities. To the best of
   121	our knowledge, limited research has explored the application of semisupervised learning methodologies for the task of cross-modality person
   122	re-identification.
   123	Based on the above analyses, this paper aims to minimize the
   124	dependence on labeled data and alleviate the performance degradation
   125	caused by the absence of labeled samples. To this end, we incorporate
   126	semi-supervised learning techniques into the VI-ReID task and propose
   127	a semi-supervised cross-modality person re-identification approach. By
   128	leveraging the generated pseudo-labeled data, our method enables
   129	the model to learn modality-irrelevant features by jointly optimizing
   130	cluster-level pseudo-labeled features with true labeled features.
   131	The main contributions of our approach are as follows.
   132	(1) We design a Confidence guided Cross-modality Pseudo label
   133	Generation (CCPG) module, which leverages the supervised information to compute the cross-modality matching confidence and the
   134	clustering confidence of the current model, and then uses the obtained
   135	confidence values to generate more reliable pseudo labels for the next
   136	iteration.
   137	(2) We design a Semi-supervised Cross-modality Discriminant
   138	F eature Learning (SCDFL) module, which can improve the discriminability of features gradually by making use of nearest-neighbor image
   139	pairs sampled from both the labeled and pseudo-labeled data.
   140	(3) We conduct extensive experiments on the SYSU-MM01 and
   141	RegDB datasets. The results demonstrate the effectiveness of our
   142	proposed method.
   143	
   144	2.1. Supervised cross-modality person ReID
   145	Recent years have witnessed significant progress in Visible-Infrared
   146	ReID (VI-ReID), with various approaches proposed to address modality
   147	discrepancy. Wu et al. [11] established the SYSU-MM01 benchmark dataset, and proposed a zero-padding strategy for learning
   148	modality-shared features in deep networks. Subsequent efforts explored
   149	constrained feature learning: Hao et al. [12] designed dual-stream
   150	networks with hypersphere manifold embedding to reduce intra- and
   151	inter-modality divergence, while Ye et al. [13] employed bi-directional
   152	dual-constrained top-ranking loss for modality alignment. Adversarial learning emerged as another promising direction in VI-ReID. The
   153	cmGAN framework [14] leveraged generative adversarial networks
   154	to minimize cross-modality divergence, while Hi-CMD [15] further
   155	disentangled identity-related and modality-specific features through
   156	adversarial disentanglement. In addition, Wu et al. [16] proposed joint
   157	modality-pattern alignment to uncover subtle cross-modality differences, and Ling et al. [17] developed a cross-modality average precision
   158	method to optimize the cross-modality overall ranking order.
   159	Despite these advances, existing supervised methods heavily depend on extensive cross-modality identity labels, significantly limiting
   160	their adaptability to new scenarios. Our work addresses this critical
   161	limitation by proposing a semi-supervised framework that achieves
   162	competitive performance with limited labeled data, enabling practical
   163	deployment in real-world environments.
   164	
   165	2.2. Unsupervised visible-infrared person ReID
   166	Unsupervised Visible-Infrared Person ReID (USVI-ReID) is proposed
   167	to cope with the cases characterized by a scarcity of annotated data
   168	labels. The USVI-ReID task was first addressed in [5] by introducing
   169	a two-stage method called H2H, which involves homogeneous and
   170	heterogeneous learning techniques. Wang et al. [18] accomplished
   171	the process of visible-infrared matching by minimizing the distance
   172	between the feature sets of visible images and infrared images. The approach in [6] overcomes the defect of pre-training in H2H and proposes
   173	an enhanced contrastive aggregation learning method for USVI-ReID.
   174	In [19], Wu et al. proposed a method based on step-wise graph matching and alternating learning, achieving relatively good performance. Ji
   175	et al. [20] designed an adversarial contrastive feature learning framework for USVI-ReID, aiming to enhance the discriminative capability
   176	of features. Yang et al. [21] proposed a transformer-based shallowdeep collaborative learning framework, which integrates shallow patch
   177	embeddings with deep features to address modality gaps in VI-ReID.
   178	In [22], Yu et al. introduced a controllable diffusion framework that
   179	generates diverse, identity-aware infrared person images to enhance
   180	VI-ReID without extra annotations.
   181	Although the above unsupervised methods have achieved very encouraging performance by only using unlabeled data, their performance
   182	still shows a large gap compared with the supervised learning methods.
   183	The major reason can be attributed to the lack of guidance from labeled
   184	data, which hinders the reliability of the trained model.
   185	
   186	2.3. Semi-supervised person ReID
   187	This section reviews semi-supervised person re-identification,
   188	covering both single-modality and cross-modality approaches.
   189	2
   190	
   191	Pattern Recognition 165 (2025) 111669
   192	
   193	X. Zhu et al.
   194	
   195	of generated noisy pseudo-labels. Different from DPIS, our approach
   196	leverages the labeled data to enhance the reliability of generated crossmodality cluster pairs, which are then used to reduce the differences
   197	between the infrared and visible modalities.
   198	In summary, both our approach and DPIS aim to alleviate the
   199	dependency on labeled data in cross-modality person re-identification
   200	through the use of semi-supervised techniques. However, there are
   201	significant differences in the solutions and settings employed.
   202	
   203	2.3.1. Semi-supervised single-modality person re-identification
   204	In recent years, many semi-supervised person re-identification
   205	methods have been proposed. Some representative works are as follows. [23] first proposed a semi-supervised coupled dictionary learning
   206	method that integrates labeled and unlabeled data to acquire shared
   207	discriminant dictionaries and re-configurable sparse encoders via dictionary decomposition techniques. Han et al. [24] proposed using
   208	CycleGAN to generate unlabeled samples from labeled training data,
   209	and innovatively learned the pseudo relationship between labeled and
   210	unlabeled training data. Furthermore, [25] designed an end-to-end
   211	multiple branch network, which optimizes the model through identity
   212	classification. Li et al. [26] proposed a semi-supervised region metric
   213	learning method that employs label propagation and cross-person distribution alignment to handle imbalanced unlabeled data and identify
   214	positive neighbors. Wu et al. [27] proposed a joint training method
   215	and a progressive sampling strategy which can optimize the model
   216	through labeled data (or pseudo-labeled) and unlabeled data. The
   217	above semi-supervised methods leverage a small amount of labeled data
   218	and a large amount of unlabeled data for training, and have achieved
   219	interesting results on single-modality person ReID task. However, these
   220	semi-supervised methods cannot be directly employed to solve the
   221	cross-modality person ReID due to the large discrepancy between different modalities. Our approach differs from existing single-modality
   222	methods in the following two folds.
   223	(1) Existing single-modality methods usually do not take the modality differences into consideration during the feature extraction phase. In
   224	contrast, our feature extraction model employs a dual-stream network,
   225	which can reduce the influence of cross-modality discrepancy on the
   226	feature extraction effectively.
   227	(2) Regarding the utilization of unlabeled information, existing
   228	methods generally generate pseudo-labels through clustering algorithms without considering the impact of modality differences on
   229	clustering results. In this paper, we design a Confidence Guided Crossmodality Pseudo Label Generation module, which can generate the
   230	cross-modality matching confidence and the clustering confidence.
   231	The generated confidences are used to enhance the reliability of
   232	cross-modality clustering pairs.
   233	
   234	3. Proposed method
   235	3.1. Problem formulation and overview
   236	In this paper, we propose a confidence-aware pseudo labels
   237	optimization-based semi-supervised cross-modality feature learning approach for VI-ReID. Our approach, as illustrated in Fig. 1, mainly
   238	consists of two modules: confidence guided cross-modality pseudo label
   239	generation module and semi-supervised cross-modality discriminant
   240	feature learning module. Details of each module can be found in Sections 3.2 and 3.3. In both modules, the feature extraction blocks share
   241	the same structure, consisting of modality-specific shallow layers and
   242	modality-shared layers. The modality-specific layers for each modality
   243	can capture modality-specific information. The modality-shared layers
   244	can alleviate the divergence between the two heterogeneous modalities
   245	by learning a multi-modality shareable space. The training process is
   246	divided into the following two stages.
   247	Stage 1. Training a feature extraction model with strong representation ability. Specifically, the labeled and unlabeled images are
   248	first sent to the feature extraction module and the features of each
   249	sample are obtained. Based on the obtained features, the unlabeled
   250	data is clustered using the DBSCAN algorithm. Then, the cluster-center
   251	features for each cluster are computed and stored in the static memory
   252	bank. Finally, the loss functions 𝑠𝑖𝑑 (on labeled images) and 𝑠𝑡𝑎 (on
   253	unlabeled images) are employed to optimize the feature extraction
   254	model.
   255	Stage 2. Further optimizing the feature extraction model by
   256	introducing the confidence-guided cross-modality pseudo label
   257	generation module and the dynamic memory bank. Specifically,
   258	the labeled visible and infrared features obtained from feature extraction module are fed into the adaptive confidence generation block,
   259	to compute the cross-modality matching confidence and the clustering
   260	confidence. Then, the unlabeled images are clustered, and the pseudolabeled infrared and visible cluster pairs can be generated according to
   261	the cross-modality matching confidence and the clustering confidence.
   262	Next, the generated pseudo-labeled cross-modality cluster pairs are
   263	used to update the dynamic memory bank. In this way, the differences
   264	between cross-modality clusters of the same pedestrian can be reduced
   265	gradually. Finally, the static memory bank and the updated dynamic
   266	memory bank are used for semi-supervised cross-modality discriminant
   267	feature learning. In this stage, the loss functions 𝑠𝑖𝑑 , 𝑠𝑡𝑎 and 𝑑𝑦𝑛 are
   268	employed to optimize the feature extraction model. The flow of our
   269	proposed approach is depicted in Algorithm 1.
   270	
   271	2.3.2. Semi-supervised cross-modality person re-identification
   272	Research on semi-supervised learning in cross-modality person reidentification remains in its early stages. To our knowledge, there is
   273	only one relevant article, i.e., the literature [28], which introduces
   274	Dual Pseudo-label Interactive Self-training (DPIS) for this task. DPIS
   275	addresses both reliable pseudo-label generation and noisy pseudo-label
   276	calibration. Our method differs from DPIS in three key aspects.
   277	(1) The semi-supervised settings in our paper differ from those
   278	in DPIS. In DPIS, labels of different rates for each identity on both
   279	modalities are utilized in the bi-semi-supervised setting. However, our
   280	approach selects different rates of pedestrians, and leverages the labels
   281	of images of these pedestrians in the semi-supervised setting. Taking
   282	the RegDB dataset as an example, the original training set consists of
   283	206 identities, with each identity having ten visible images and ten
   284	infrared images. Under the setting of 10% labeled data, DPIS utilizes
   285	the labels of 10% images for each identity (i.e., one visible image
   286	and one infrared image for each individual). In contrast, our approach
   287	utilizes the labeled images of about twenty-one pedestrian individuals
   288	(206×10% ≈ 21), leaving the remaining individuals’ images as unlabeled
   289	data.
   290	(2) The assumptions on the relationship between unlabeled and labeled images are different. Our method does not impose any restrictions
   291	on the label space between unlabeled and labeled images, whereas DPIS
   292	assumes that both unlabeled and labeled images have the same label
   293	space (i.e., the labeled and unlabeled images are from the same group
   294	of pedestrians).
   295	(3) The goal of utilizing the small amount of labeled data is
   296	different. DPIS utilizes labeled data to mitigate the negative impact
   297	
   298	3.2. Confidence guided cross-modality pseudo label generation
   299	To facilitate the description of our
   300	approach, we first}introduced
   301	{
   302	the notation used in this paper. Let 𝑋𝑟 = 𝑥𝑟1 , 𝑥𝑟2 , … , 𝑥𝑟𝑀
   303	and 𝑋𝑣 =
   304	1
   305	{
   306	}
   307	𝑣
   308	𝑣
   309	𝑣
   310	𝑥1 , 𝑥2 , … , 𝑥𝑀 separately denote the features of labeled infrared and
   311	2
   312	{
   313	}
   314	{
   315	}
   316	visible images. Let 𝑈𝑟 = 𝑢𝑟1 , 𝑢𝑟2 , … , 𝑢𝑟𝑁
   317	and 𝑈𝑣 = 𝑢𝑣1 , 𝑢𝑣2 , … , 𝑢𝑣𝑁
   318	1
   319	
   320	2
   321	
   322	represent the features of unlabeled infrared and visible images, respectively. Here, 𝑀1 , 𝑀2 , 𝑁1 and 𝑁2 are the numbers of items in the
   323	corresponding sets, respectively. Let 𝐶𝑢𝑟 and 𝐶𝑢𝑣 (𝐶𝑙𝑟 and 𝐶𝑙𝑣 ) represent
   324	the clustering results of unlabeled (labeled) infrared and visible images,
   325	𝑟 and 𝐶 𝑣 (𝐶 𝑟 and 𝐶 𝑣 ) be the 𝑖th clusters in 𝐶 𝑟 and
   326	respectively. Let 𝐶𝑢,𝑖
   327	𝑢
   328	𝑢,𝑖
   329	𝑙,𝑖
   330	𝑙,𝑖
   331	𝐶𝑢𝑣 (𝐶𝑙𝑟 and 𝐶𝑙𝑣 ), respectively.
   332	3
   333	
   334	Pattern Recognition 165 (2025) 111669
   335	
   336	X. Zhu et al.
   337	
   338	Fig. 1. Overall framework of our proposed approach. The blocks marked with a red star in the bottom right corner are only used for the training of stage 2.
   339	
   340	where 𝑥𝑟𝑜 and 𝑥𝑣𝑡 represent labeled infrared and visible data,
   341	respectively. 𝑎𝑣𝑔 (⋅) denotes the mean function.
   342	Finally, the cross-modality matching confidence is used as a dynamic threshold to select more reliable unlabeled cross-modality image
   343	pairs, as shown in Eq. (3).
   344	{(
   345	}
   346	)|
   347	(
   348	)
   349	 = 𝑢𝑣𝑎 , 𝑢𝑟𝑏 |𝑠𝑖𝑚 𝑢𝑣𝑎 , 𝑢𝑟𝑏 ≥ 𝛥, 𝑢𝑣𝑎 ∈ 𝑈𝑣 , 𝑢𝑟𝑏 ∈ 𝑈𝑟
   350	(3)
   351	|
   352	
   353	To enhance the reliability of generated pseudo-labeled crossmodality image pairs, we propose leveraging a limited number of
   354	labeled samples to guide the generation process. Specifically, we design
   355	two metrics to evaluate the reliability of the current model: crossmodality matching confidence and clustering confidence. Details of
   356	these two metrics are as follows.
   357	
   358	where (𝑢𝑣𝑎 , 𝑢𝑟𝑏 ) represents a pseudo-labeled cross-modality image pair
   359	obtained using the DBSCAN algorithm, with 𝑎 and 𝑏 denoting their respective pseudo-labels. Based on the pseudo labels (𝐶𝑢𝑟 and 𝐶𝑢𝑣 obtained
   360	by clustering) of each image in , we can count the number of matched
   361	image pairs between any two cross-modality clusters. Then, we rank
   362	the cross-modality cluster pairs in descending order according to the
   363	number of matched image pairs.
   364	The dynamic threshold proposed in this paper aims to leverage labeled data for selecting high-quality pseudo-label sample pairs. Existing
   365	studies typically employ a fixed threshold of 0.5 to determine crossmodality sample category consistency. However, experiments reveal
   366	that factors such as background and pose variations can cause the
   367	cosine similarity between samples of different categories to exceed 0.5,
   368	highlighting the limitations of the fixed threshold method. To address
   369	this issue, in each training epoch, we dynamically generate thresholds
   370	based on the cosine similarity distribution of labeled cross-modality
   371	data. This strategy adapts to the sample similarity distribution, reduces
   372	noise interference, and fully utilizes limited annotation information,
   373	thereby making pseudo-label selection more accurate and reliable.
   374	To better investigate the effect of the dynamic threshold, we
   375	experimentally analyzed the number of cross-modality sample pairs
   376	
   377	3.2.1. Cross-modality matching confidence computation
   378	The cross-modality matching confidence is designed to assess the
   379	overall similarities between different modalities. During each training
   380	epoch, we initially utilize the updated feature extraction networks
   381	𝑓𝜃𝑟 and 𝑓𝜃𝑣 to extract features from labeled samples. These extracted
   382	features are denoted as 𝑋𝑟 and 𝑋𝑣 . Subsequently, we calculate the
   383	cosine similarity matrix between 𝑋𝑟 and 𝑋𝑣 . Finally, the confidence
   384	coefficient is determined by computing the average cosine similarity
   385	among all truly matching image pairs. The confidence coefficient allows
   386	us to determine with greater certainty whether unlabeled data samples
   387	from different modalities belong to the same identity.
   388	First, we compute the pairwise similarity for each cross-modality
   389	image pair ⟨𝑝, 𝑞⟩ (labeled or unlabeled) via Eq. (1).
   390	𝑠𝑖𝑚 (𝑝, 𝑞) =
   391	
   392	𝑝⋅𝑞
   393	‖𝑝‖ × ‖𝑞‖
   394	
   395	(1)
   396	
   397	Then, the cross-modality matching confidence (denoted as 𝛥) can be
   398	obtained by averaging the similarity scores of all labeled cross-modality
   399	image pairs with high similarity, as shown in Eq. (2).
   400	({
   401	})
   402	𝛥 = 𝑎𝑣𝑔 𝑠𝑖𝑚(𝑥𝑣𝑡 , 𝑥𝑟𝑜 )|𝑥𝑣𝑡 ∈ 𝑋𝑣 , 𝑥𝑟𝑜 ∈ 𝑋𝑟 , 𝑠𝑖𝑚(𝑥𝑣𝑡 , 𝑥𝑟𝑜 ) > 0.5
   403	(2)
   404	4
   405	
   406	Pattern Recognition 165 (2025) 111669
   407	
   408	X. Zhu et al.
   409	
   410	Algorithm 1: Confidence Guided Semi-supervised Feature
   411	Learning
   412	
   413	Algorithm 2: Reliable Cross-Modality Pseudo Label Generation
   414	Input: Labeled data (𝑋𝑟 ,𝑋𝑣 ), unlabeled data (𝑈𝑟 ,𝑈𝑣 )
   415	Output: Cross-modality pseudo-labeled cluster pairs
   416	3 Initialize sets  and , 𝑖 = 1, 𝑗 = 1.
   417	4 Perform DBSCAN clustering on labeled data and unlabeled data
   418	independently.
   419	5 Compute matching confidence 𝛥 via Eq. (2)
   420	6 while 𝑖 ≤ 𝑁1 do
   421	7
   422	while 𝑗 ≤ 𝑁2 do
   423	8
   424	Add ⟨𝑢𝑟𝑖 , 𝑢𝑣𝑗 ⟩ to candidate set  via Eq. (3)
   425	1
   426	
   427	Input: Labeled data, unlabeled data
   428	∗
   429	2 Output: Optimized feature extraction model 𝑓
   430	𝜃
   431	3 Initialize model parameters 𝜃
   432	4 𝑒𝑝𝑜𝑐ℎ ← 1
   433	5 while 𝑒𝑝𝑜𝑐ℎ ≤ max_epochs do
   434	6
   435	Initialize static memory bank 𝑠 and dynamic memory
   436	bank 𝑑
   437	7
   438	Extract features of labeled and unlabeled data, i.e., 𝑋𝑟 , 𝑋𝑣 ,
   439	𝑈𝑟 and 𝑈𝑣
   440	8
   441	Apply DBSCAN clustering on unlabeled data (𝑈𝑟 and 𝑈𝑣 )
   442	9
   443	Compute the cluster center of each cluster via Eq. (6) and
   444	store in 𝑠
   445	10
   446	if 𝑒𝑝𝑜𝑐ℎ ≤ ⌊max_epochs∕2⌋ then
   447	11
   448	Update 𝜃 by minimizing 𝑠𝑖𝑑 + 𝑠𝑡𝑎 (Eqs. (9), (10))
   449	12
   450	else
   451	13
   452	Compute the cross-modality matching confidence 𝛥 via
   453	Eq. (2)
   454	14
   455	Calculate the clustering confidence via Eq. (4)
   456	15
   457	Update 𝑑 with reliable pseudo-labels (see
   458	Algorithm 2)
   459	16
   460	Optimize 𝜃 by minimizing the loss function  (Eq. (12))
   461	17
   462	end
   463	18
   464	𝑒𝑝𝑜𝑐ℎ ← 𝑒𝑝𝑜𝑐ℎ + 1
   465	19 end
   466	1
   467	
   468	2
   469	
   470	𝑗 =𝑗+1
   471	end
   472	11
   473	𝑖=𝑖+1
   474	12 end
   475	13 Count the number of matched image pairs in  for each cluster
   476	pair in unlabeled data
   477	14 Sort all cluster pairs by the matched numbers in descending
   478	order, denoted as 
   479	15 Compute clustering confidence 𝜌 via Eq. (4) on labeled data
   480	16 Calculate the number of selected cluster pairs 𝑘 = ⌈𝜌 × 𝑛⌉,
   481	where 𝑛 is size of 
   482	17 Return top 𝑘 cluster pairs in 
   483	9
   484	
   485	10
   486	
   487	the minimum average distance between the 𝑡𝑡ℎ instance and samples
   488	belonging to distinct classes in . The clustering confidence serves
   489	as a dynamic selection ratio, guiding the selection of more reliable
   490	cross-modality clusters for updating the memory bank.
   491	
   492	filtered out by the dynamic threshold during training. The experimental
   493	results demonstrate that, on the SYSU-MM01 dataset, the dynamic
   494	threshold filters approximately 20% more unreliable pseudo-labeled
   495	cross-modality sample pairs compared to using a fixed cosine similarity
   496	threshold, while on the RegDB dataset, it filters about 1% more. This
   497	quantity discrepancy primarily stems from the greater complexity of
   498	image types and scenes in SYSU-MM01, which introduces more noise
   499	between samples, thus necessitating stricter filtering. Additionally,
   500	the number of filtered sample pairs dynamically changes throughout
   501	training: as model accuracy improves, the number of filtered samples gradually decreases. This phenomenon indicates that the dynamic
   502	threshold can adapt to the model’s learning state, effectively balancing
   503	sample utilization and noise suppression.
   504	
   505	3.2.3. Reliable cross-modality pseudo label generation
   506	Based on the cross-modality matching confidence computation module and clustering confidence calculation framework, we propose a twostep strategy to generate high-quality cross-modality pseudo-labeled
   507	pairs from unlabeled data.
   508	Step 1. Cross-Modality Matching Confidence based Filtering.
   509	Based on the labeled cross-modality data, we can compute the crossmodality matching confidence 𝛥 by Eq. (2). Then, for each pair of
   510	unlabeled cross-modality images ⟨𝑝, 𝑞⟩, we calculate the pairwise cosine
   511	similarity 𝑠𝑖𝑚(𝑝, 𝑞) by Eq. (1). Next, image pairs satisfying 𝑠𝑖𝑚(𝑝, 𝑞) ≥ 𝛥
   512	are filtered and used as the candidate data for generating pseudo labels.
   513	Step 2. Clustering Confidence Guided Pseudo Label Selection.
   514	Given the candidate high-similarity cross-modality image pairs, we
   515	count the number of matched image pairs between each cross-modality
   516	cluster pair, and use the number as the reliability score for the corresponding cluster pair. Then, we sort cluster pairs by the reliability
   517	score in descending order, and denote the sorted cluster pair list by
   518	 = {1 , 2 , … , 𝑛 }. Here, 𝑖 represents the 𝑖th cross-modality cluster pair. Next, we compute the clustering confidence (denoted by 𝜌)
   519	using labeled data clusters by Eq. (4). Finally, the top ⌈𝜌 × 𝑛⌉ cluster
   520	pairs with highest confidence in  are selected as the pseudo-labeled
   521	cross-modality data for updating the dynamic memory bank.
   522	The overall algorithm is summarized in Algorithm 2. This twostep generation mechanism effectively utilizes both the image-level
   523	and cluster-level confidence to improve the reliability of the generated
   524	pseudo labels.
   525	
   526	3.2.2. Clustering confidence computation
   527	In practical scenarios, certain clusters may contain some incorrectly grouped samples, which can adversely affect feature learning.
   528	To mitigate this issue and improve pseudo-label reliability, we propose
   529	a cluster filtering mechanism based on a novel clustering confidence
   530	metric. This metric enables selective retention of high-confidence
   531	cross-modality clusters from the ranked candidates.
   532	Following the theoretical foundation of silhouette analysis [29],
   533	we design our confidence measure by adapting the classic silhouette
   534	coefficient. The cross-modality clustering confidence is defined as:
   535	(
   536	)
   537	1 𝑆(𝑋𝑣 , 𝑌̂𝑣 ) 𝑆(𝑋𝑟 , 𝑌̂𝑟 )
   538	𝜌=
   539	+
   540	,
   541	(4)
   542	2 𝑆(𝑋𝑣 , 𝑌𝑣 ) 𝑆(𝑋𝑟 , 𝑌𝑟 )
   543	where 𝑋𝑣 and 𝑋𝑟 represent visible and infrared feature sets respectively. 𝑌𝑚 and 𝑌̂𝑚 denote the ground-truth and pseudo-label sets for
   544	modality 𝑚 ∈ {𝑣, 𝑟}, respectively. The silhouette score function 𝑆(⋅, ⋅)
   545	is formulated as:
   546	𝑑min (𝑥𝑡 ) − 𝑑avg (𝑥𝑡 )
   547	1 ∑
   548	𝑆(, ) =
   549	,
   550	(5)
   551	|| 𝑥 ∈ max{𝑑min (𝑥𝑡 ), 𝑑avg (𝑥𝑡 )}
   552	
   553	3.2.4. Discussion of pseudo label generation strategy
   554	Pseudo-label generation plays an important role in both semisupervised and unsupervised VI-ReID methods. While sharing the core
   555	concept of generating pseudo-labels via clustering with other unsupervised VI-ReID methods, our proposed confidence guided cross-modality
   556	pseudo-label generation module offers a distinct approach to improving pseudo-label reliability. Existing techniques enhance pseudo-label
   557	quality in various ways. For example, [30] utilizes a two-component
   558	
   559	𝑡
   560	
   561	Here, 𝑑avg (⋅) denotes the average distance between the 𝑡𝑡ℎ sample and all
   562	other samples sharing the same class label in , while 𝑑min (⋅) represents
   563	5
   564	
   565	Pattern Recognition 165 (2025) 111669
   566	
   567	X. Zhu et al.
   568	
   569	as 𝑋 ∈ 𝑅𝐾1 ×𝑑 . Here, 𝐾1 and 𝑑 represent the number of clusters for
   570	unlabeled infrared samples and the dimension of features, respectively.
   571	Then, cosine similarities are calculated between all samples in the subdataset, which are then used to construct the distance matrix between
   572	all clusters (𝑑𝑖𝑠𝑡 ∈ 𝑅𝐾1 ×𝐾1 ).
   573	(Then,
   574	) based on the obtained distance (matrix,
   575	) we construct a set
   576	𝑟
   577	𝑟 . N 𝐶𝑟
   578	N 𝐶𝑢,𝑖
   579	for each infrared cluster 𝐶𝑢,𝑖
   580	consists of the cur𝑢,𝑖
   581	
   582	Gaussian mixture model to estimate label confidence scores, [31] enhances cross-modality label credibility through heterogeneous transfer
   583	and homogeneous structure smoothing, and [32] introduces a noisy
   584	pseudo-label calibration module to rectify label noise. In contrast
   585	to these methods, our approach introduces a novel dual-confidence
   586	mechanism that simultaneously evaluates cross-modality matching confidence and clustering confidence. This innovative strategy effectively
   587	filters out unreliable pseudo-labels, thereby ensuring the generation of
   588	more reliable pseudo-labels for subsequent model training.
   589	
   590	rent cluster’s top 𝑝 − 1 nearest classes. Here 𝑝 is the number of
   591	clusters appeared in each mini batch. The constructed graph can be
   592	{
   593	}
   594	denoted
   595	as 𝐺 = (𝑉 }
   596	, 𝐸), where 𝑉 = 𝑐 ||𝑐 = 1, 2, … , 𝐾1 and 𝐸 =
   597	{(
   598	)|
   599	( )
   600	𝑐𝐼 , 𝑐𝐽 |𝑐𝐽 ∈ N 𝑐𝐼 . Each cluster represents a single node of the
   601	|
   602	graph 𝐺.
   603	𝑟 and its 𝑝 − 1 nearest clusters
   604	Finally, by combining each cluster 𝐶𝑢,𝑖
   605	{
   606	}⋃ (
   607	)
   608	𝑟
   609	𝑟
   610	from the graph 𝐺, we can obtain a new set 𝐴𝑖 = 𝐶𝑢,𝑖
   611	N 𝐶𝑢,𝑖
   612	in
   613	⋃𝑖=𝐾1
   614	each mini-batch. Then, 𝐴 = 𝑖=1 𝐴𝑖 . We randomly select 𝑘 samples
   615	for each cluster in 𝐴, generating a mini-batch of 𝐵 = 𝑝 × 𝑘 samples for
   616	the training of current epoch.
   617	For the unlabeled visible images, labeled infrared and visible images, the mini-batch construction processes are the same as that of
   618	unlabeled infrared images. Note that the true labels are utilized for the
   619	sampling of labeled infrared and visible images.
   620	
   621	3.3. Semi-supervised cross-modality discriminant feature learning
   622	The semi-supervised cross-modality discriminant feature learning
   623	module aims to make full use of the high-quality pseudo-labeled data
   624	and labeled cross-modality image pairs. In this module, we first design
   625	a nearest neighbor class based sampler (NNC-sampler) that selects
   626	samples with high similarity for training. This provides rich information
   627	and challenging samples for feature learning, further improving the
   628	model’s robustness. In addition, we design two memory banks: static
   629	memory bank and dynamic memory bank. The static memory bank
   630	is composed of the cluster-center features of unlabeled data, and the
   631	dynamic memory bank consists of the dynamic cluster-center features
   632	updated by using the generated pseudo-labeled cross-modality clusters.
   633	The class-center losses of static and dynamic memory banks are calculated separately to improve the intra-class compactness and inter-class
   634	separability of samples. Details of the static and dynamic memory banks
   635	are as follows.
   636	Static Memory Bank. The static memory bank (denoted as 𝑠 )
   637	stores the centroid of each cluster derived from unlabeled data. For the
   638	𝑖th cluster of unlabeled visible/infrared images, the cluster centroid is
   639	computed by Eq. (6).
   640	∑
   641	1
   642	𝜙𝑚
   643	𝑢𝑚
   644	(6)
   645	𝑖 = | 𝑚|
   646	𝑘
   647	𝑚
   648	|𝐶𝑢,𝑖 | 𝑢𝑚𝑘 ∈𝐶𝑢,𝑖
   649	|
   650	|
   651	
   652	3.4. Loss function
   653	(1) The loss for unlabeled samples
   654	Let 𝑞 𝑟 and 𝑞 𝑣 represent the features of an unlabeled infrared image and an unlabeled visible image, respectively. In the optimization
   655	process, the ClusterNCE [33] is employed to compute the loss for each
   656	unlabeled image.
   657	For each image 𝑞 𝑚 , 𝑚 ∈ {𝑣, 𝑟}, the loss function consists of two parts:
   658	𝑚
   659	(i) The ClusterNCE loss 𝑚
   660	𝑠𝑡𝑎 between 𝑞 and the cluster-center features
   661	in the static memory bank; (ii) The ClusterNCE loss 𝑚
   662	between 𝑞 𝑚
   663	𝑑𝑦𝑛
   664	and the infrared cluster-center features in the dynamic memory bank.
   665	𝑚
   666	Both 𝑚
   667	𝑠𝑡𝑎 and 𝑑𝑦𝑛 are defined as Eq. (8).
   668	(
   669	)
   670	exp 𝑞 𝑚 ⋅ 𝜑𝑚,𝑏
   671	+ ∕𝜏
   672	𝑚
   673	(8)
   674	(
   675	)
   676	𝑏 = − log ∑ 𝐾
   677	𝑚
   678	exp 𝑞 𝑚 · 𝜑𝑚,𝑏
   679	∕𝜏
   680	𝑘
   681	𝑘=0
   682	
   683	where 𝑚 denotes the modality, 𝑚 ∈ {𝑣, 𝑟}. 𝑢𝑚
   684	represents the 𝑘th sample
   685	𝑘
   686	𝑚 . |⋅| denotes the number of samples in a cluster. Based on the
   687	in 𝐶𝑢,𝑖
   688	static memory bank, the loss 𝑟𝑠𝑡𝑎 (described in Section 3.4) is used to
   689	reduce the intra-modality divergence.
   690	Dynamic Memory Bank. The dynamic memory bank (denoted as
   691	𝑑 ) is first initialized with the static memory bank, and then updated
   692	the generated pseudo-labeled cross-modality cluster pairs.
   693	⟨ using ⟩
   694	𝑣 , 𝐶𝑟
   695	Let 𝐶𝑢,𝑖
   696	𝑢,𝑗 represents a pseudo-labeled cross-modality cluster pair,
   697	
   698	where 𝑏 ∈ {𝑠𝑡𝑎, 𝑑𝑦𝑛} denotes the type of memory bank (static/dynamic). 𝜑𝑚,𝑏
   699	+ represents the cluster centroid features corresponding to
   700	the cluster containing query 𝑞 𝑚 in memory bank 𝑏. 𝜑𝑚,𝑏
   701	denotes the
   702	𝑘
   703	centroid of the 𝑘th cluster for modality 𝑚 in memory bank 𝑏. 𝐾𝑚
   704	represents the number of clusters for the unlabeled images of modality
   705	𝑚. 𝜏 is a hyperparameter, which is set to 0.1 in this paper.
   706	
   707	𝑣 means the 𝑖th cluster of unlabeled visible images, and 𝐶 𝑟
   708	where 𝐶𝑢,𝑖
   709	𝑢,𝑗
   710	denotes the 𝑗th cluster of unlabeled infrared images. We update the
   711	dynamic memory bank for visible modality by integrating features
   712	from the infrared modality into the visible feature memory using a
   713	momentum updating strategy as shown in Eq. (7).
   714	
   715	𝜓𝑖𝑣 ← 𝛽𝜓𝑖𝑣 + (1 − 𝛽) 𝜓𝑗𝑟
   716	
   717	(2) Loss for Labeled Samples
   718	For labeled samples, we calculate the cross entropy loss between the
   719	predicted labels and the ground truth labels, as shown in Eq. (9).
   720	(
   721	𝑛
   722	𝑚
   723	(
   724	))
   725	| (
   726	)) 1 ∑
   727	1∑
   728	| (
   729	𝑠𝑖𝑑 = −
   730	log 𝑃 y𝑣𝑖 |𝐶 x𝑣𝑖 |𝜃
   731	−
   732	log 𝑃 y𝑟𝑗 ||𝐶 x𝑟𝑗 |𝜃
   733	(9)
   734	|
   735	𝑛 𝑖=1
   736	𝑚 𝑗=1
   737	|
   738	
   739	(7)
   740	
   741	𝑣 and 𝐶 𝑟 ,
   742	where 𝜓𝑖𝑣 and 𝜓𝑗𝑟 represent the dynamic centers of clusters 𝐶𝑢,𝑖
   743	𝑢,𝑗
   744	
   745	respectively. 𝛽 is a balancing factor, which is set to 0.1 in our paper.
   746	By using Eq. (7), the center of the 𝑖th visible cluster becomes closer to
   747	that of the 𝑗th infrared cluster. The dynamic memory bank can guide
   748	the feature extraction model to learn more similar features for samples
   749	within the cross-modality cluster pair, thereby further mitigating the
   750	discrepancy between different modalities to some extent.
   751	Nearest Neighbor Class based Sampler. To fully utilize the sample
   752	data and enhance the robustness of the model, we design a nearest
   753	neighbor class based sampler(NNC-sampler) for both the unlabeled
   754	and labeled samples. Here, we present the construction process of a
   755	nearest neighbor class-based sampler on unlabeled infrared samples as
   756	an illustrative example.
   757	For unlabeled infrared samples, we randomly select one sample
   758	from each cluster to construct a sub-dataset. Then, the features of samples in the sub-dataset are extracted with the current network, denoted
   759	
   760	where 𝑛 and 𝑚 represent the number of labeled visible and infrared
   761	images in the current mini-batch, respectively. y𝑣𝑖 and y𝑟𝑗 respectively
   762	(
   763	)
   764	represent the corresponding true labels of x𝑣𝑖 and x𝑟𝑗 , 𝐶 x𝑣𝑖 |𝜃 and
   765	(
   766	)
   767	𝐶 x𝑟𝑗 |𝜃 respectively denote the predicted labels for x𝑣𝑖 and x𝑟𝑗 through
   768	the classifier 𝐶(⋅).
   769	For clarity, we define 𝑠𝑡𝑎 and 𝑑𝑦𝑛 as follows.
   770	𝑠𝑡𝑎 = 𝑣𝑠𝑡𝑎 + 𝐿𝑟𝑠𝑡𝑎
   771	
   772	(10)
   773	
   774	𝑑𝑦𝑛 = 𝑣𝑑𝑦𝑛 + 𝐿𝑟𝑑𝑦𝑛
   775	
   776	(11)
   777	
   778	The total loss function of our approach is defined as Eq. (12).
   779	 = 𝑠𝑖𝑑 + 𝜆𝑠𝑡𝑎 + (1 − 𝜆)𝑑𝑦𝑛
   780	6
   781	
   782	(12)
   783	
   784	Pattern Recognition 165 (2025) 111669
   785	
   786	X. Zhu et al.
   787	
   788	where 𝜆 is the weight coefficient. 𝑠𝑖𝑑 aims to improve the discriminability of the model by using the labeled information, and guides the
   789	model to make better use of the unlabeled data. 𝑠𝑡𝑎 facilitates alleviating the intra-modality divergence in the model’s learning process. 𝑑𝑦𝑛
   790	helps to reduce the discrepancy between different modalities.
   791	During inference, we only utilize the well-trained feature extraction
   792	model, without other components of the proposed framework. Specifically, query images and gallery images from two modalities are fed
   793	into the feature extraction module to extract features. Then, we perform
   794	inference based on the distance between the extracted features.
   795	
   796	4.3. Comparison with state-of-the-art methods
   797	In this section, we compare our approach with state-of-the-art
   798	supervised and unsupervised VI-ReID methods to demonstrate its effectiveness. Tables 1 and 2 report the corresponding experimental results
   799	on the SYSU-MM01 and RegDB datasets, respectively. In both tables,
   800	SVI-ReID, USVI-ReID, and SSVI-ReID separately denote supervised,
   801	unsupervised, and semi-supervised VI-ReID settings.
   802	4.3.1. Comparison with supervised VI-ReID methods
   803	From Table 1, we can observe that our semi-supervised crossmodality approach surpasses some supervised methods on SYSU-MM01
   804	dataset, such as FMCNet [1], MPANet [16], Zero-Padding [11],
   805	HSME [12], AlignGAN [36], and DART [45]. Moreover, from the results
   806	in Table 2, we can see that our semi-supervised approach achieves very
   807	competing or even better results compared with the state-of-the-art
   808	methods on RegDB dataset. These results are encouraging, indicating that semi-supervised cross-modality ReID holds great potential in
   809	approximating the performance of supervised VI-ReID.
   810	
   811	4. Experiments
   812	4.1. Datasets and evaluation protocol
   813	We evaluate our proposed approach on two cross-modality person
   814	re-ID datasets: SYSU-MM01 and RegDB.
   815	The SYSU-MM01 [11] dataset comprises 491 individuals captured by a combination of four visible cameras and two near-infrared
   816	cameras, encompassing both indoor and outdoor environments. The
   817	training and testing sets consist of 395 and 96 identities, respectively.
   818	The former contains 22,258 visible images and 11,909 infrared images,
   819	while the latter includes 3803 infrared query images. Our method is
   820	evaluated under two testing modes: all-search mode, with a gallery set
   821	composed of all visible images; indoor-search mode, with a gallery set
   822	consisting of indoor visible camera images.
   823	The RegDB [34] dataset is a compact cross-modality dataset consisting of images captured by both visible and infrared cameras. It
   824	comprises 4120 images from 412 individuals, with 206 identities used
   825	for training and the remaining 206 identities reserved for testing
   826	purposes. Each individual is represented by 10 visible images and
   827	10 thermal images. Following established protocols, we evaluate the
   828	dataset in both visible-to-infrared mode (using visible images as queries
   829	and infrared images as the gallery) and infrared-to-visible mode (with
   830	the opposite configuration).
   831	Following the protocols in [11], our approach is executed on each
   832	dataset 10 times, and the average performance is reported. Moreover, we use the Cumulative Matching Characteristic (CMC) and mean
   833	Average Precision (mAP) as evaluation metrics.
   834	
   835	4.3.2. Comparison with unsupervised VI-ReID methods
   836	In this experiment, our proposed approach is compared with
   837	the state-of-the-art unsupervised VI-ReID works, including H2H [5],
   838	OTLA [18], ADCA [6] and ACCL [19].
   839	Table 1 shows the experimental results on SYSU-MM01. Under the all-search mode, our proposed method surpasses all the
   840	best unsupervised models, with Rank-1 and mAP gains being
   841	2.56%(=59.83%–57.27%) and 1.34% (=53.12%–51.78%), respectively. Under the indoor-search mode, the performance gains of our
   842	approach are 5.27%(=61.50%–56.23%) and 1.09% (=63.83%–62.74%)
   843	in terms of Rank-1 and mAP, respectively. Note that the results of
   844	our approach are obtained with a batch size of 128 due to the video
   845	memory limitation of our computer’s graphics card (those of ADCA and
   846	ACCL are obtained with the batch size of 256). According to research
   847	in the Cluster-Contrast [33] (the framework used in this paper), batch
   848	size has a significant impact on the performance. Therefore, there is
   849	still some potential room for improvement in the performance of our
   850	approach.

codex
其中一篇抽取时被命令输出截断了，我会单独按更窄的行号补读缺失的方法段，避免只凭不完整文本判断。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '742,850p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   742	TABLE III
   743	A BLATION STUDIES OF THE SMR MODULE . W E REPORT THE RESULTS ON PRCC
   744	AND LTCC UNDER THE CLOTH - CHANGING SETTING . “L OCAL .” AND “R EFINE .”
   745	DENOTE LOCAL SEMANTICS MINING AND SEMANTICS REFINEMENT IN SMR.
   746	PRCC
   747	
   748	Methods
   749	
   750	(a)
   751	
   752	Query
   753	
   754	Rank-1 -------------------------------------------------------------------------------------------------------> Rank-10
   755	
   756	LTCC
   757	
   758	Rank-1
   759	
   760	mAP
   761	
   762	Rank-1
   763	
   764	mAP
   765	
   766	Ours w/o SMR
   767	Ours w/o Local.
   768	Ours w/o Refine.
   769	
   770	57.8
   771	63.1
   772	62.8
   773	
   774	55.8
   775	60.5
   776	60.2
   777	
   778	39.3
   779	42.1
   780	42.1
   781	
   782	16.4
   783	17.7
   784	18.2
   785	
   786	Ours
   787	
   788	65.5
   789	
   790	63.0
   791	
   792	43.6
   793	
   794	18.6
   795	
   796	is worth noting that some advanced methods [14], [15], [24], [31]
   797	also incorporate both average and max poolings, but they simply
   798	concatenate the two pooled features at the end. The advantages
   799	of both poolings are not exerted. However, CSSC using abundant
   800	semantics shows great advantages over them, without relying on
   801	extra annotations or auxiliary information. Our SMR module can
   802	be seamlessly integrated into the backbone without imposing a
   803	significant computational burden. The total number of parameters on
   804	LTCC is 54.3M for CSSC, while 141.2M for AIM [15] and 62.0M
   805	for CAMC [10]. The training FLOPs on LTCC are 12.2G for CSSC,
   806	while 18.5G for AIM [15] and 19.7G for CAMC [10].
   807	Results on Celeb-reID. As shown in Tab. II, all competitors achieve
   808	relatively poor performance, even those using stronger DenseNet121 [31], [34], [37] as the backbone. Despite the utilization of
   809	other modalities by advanced methods [8], [10], CSSC excels them
   810	without bells and whistles. It is worth noting that methods relying
   811	on manually annotated clothing labels, such as CAL [14], AIM [15],
   812	and CCFA [16], cannot work on Celeb-reID since the clothing annotations are not available. In contrast, our CSSC achieves remarkable
   813	performance without relying on extra impractical annotations.
   814	C. Ablation Studies
   815	Effectiveness of the SMR module. As shown in Tab. III, there is a
   816	substantial performance decrease without using our SMR module.
   817	The results show the undeniable importance of SMR. Additionally, both local semantics mining and semantics refinement designs
   818	contribute to performance improvement, further demonstrating the
   819	effectiveness of our proposed SMR.
   820	Effectiveness of our cross-parallel semantics collaboration design.
   821	To demonstrate that the great success of our method comes from our
   822	semantics collaboration design, rather than simply introducing multiple branches or several Re-ID losses, we ablate each branch by trying
   823	different combinations of SMR in Tab. IV. Either content or salient
   824	semantics (Methods 2 and 3) can improve the capability of the Re-ID
   825	model, but only one kind of semantics is suboptimal. Collaborating
   826	the two semantics in parallel (Method 4) or serial (Methods 5 and
   827	
   828	————————————————— —————————
   829	
   830	(b)
   831	
   832	Query
   833	
   834	Rank-1 -------------------------------------------------------------------------------------------------------> Rank-10
   835	
   836	Fig. 3. Visualization of top-10 retrieval results. For each query image,
   837	the first and the second rows are the ordered matching results obtained by
   838	using the baseline ResNet-50 and our proposed CSSC, respectively. Images
   839	with green and red borders indicate correct and wrong matching results,
   840	respectively. The results are obtained in the cloth-changing setting on LTCC.
   841	
   842	6) bring limited improvement, while our cross-parallel collaboration
   843	framework exerts the potential of both semantics.
   844	Visualization of retrieval results. To intuitively demonstrate the
   845	effect of abundant semantics in our proposed method to promote
   846	cloth-changing person Re-ID, we compare the retrieval results of
   847	CSSC and the baseline model in Fig. 3. As shown in the firstrow results of (a) and (b), when people change their clothes, the
   848	baseline model is unable to identify pedestrians correctly with the
   849	interference of similar visual appearances, e.g., similar colors, and
   850	clothing textures. However, the pedestrian images with different

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Confidence guided semi-supervised cross-modality person re-identification.pdf' - | nl -ba | sed -n '298,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Confidence guided semi-supervised cross-modality person re-identification.pdf' - | nl -ba | sed -n '621,795p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 101ms:
 succeeded in 113ms:
   621	3.3. Semi-supervised cross-modality discriminant feature learning
   622	The semi-supervised cross-modality discriminant feature learning
   623	module aims to make full use of the high-quality pseudo-labeled data
   624	and labeled cross-modality image pairs. In this module, we first design
   625	a nearest neighbor class based sampler (NNC-sampler) that selects
   626	samples with high similarity for training. This provides rich information
   627	and challenging samples for feature learning, further improving the
   628	model’s robustness. In addition, we design two memory banks: static
   629	memory bank and dynamic memory bank. The static memory bank
   630	is composed of the cluster-center features of unlabeled data, and the
   631	dynamic memory bank consists of the dynamic cluster-center features
   632	updated by using the generated pseudo-labeled cross-modality clusters.
   633	The class-center losses of static and dynamic memory banks are calculated separately to improve the intra-class compactness and inter-class
   634	separability of samples. Details of the static and dynamic memory banks
   635	are as follows.
   636	Static Memory Bank. The static memory bank (denoted as 𝑠 )
   637	stores the centroid of each cluster derived from unlabeled data. For the
   638	𝑖th cluster of unlabeled visible/infrared images, the cluster centroid is
   639	computed by Eq. (6).
   640	∑
   641	1
   642	𝜙𝑚
   643	𝑢𝑚
   644	(6)
   645	𝑖 = | 𝑚|
   646	𝑘
   647	𝑚
   648	|𝐶𝑢,𝑖 | 𝑢𝑚𝑘 ∈𝐶𝑢,𝑖
   649	|
   650	|
   651	
   652	3.4. Loss function
   653	(1) The loss for unlabeled samples
   654	Let 𝑞 𝑟 and 𝑞 𝑣 represent the features of an unlabeled infrared image and an unlabeled visible image, respectively. In the optimization
   655	process, the ClusterNCE [33] is employed to compute the loss for each
   656	unlabeled image.
   657	For each image 𝑞 𝑚 , 𝑚 ∈ {𝑣, 𝑟}, the loss function consists of two parts:
   658	𝑚
   659	(i) The ClusterNCE loss 𝑚
   660	𝑠𝑡𝑎 between 𝑞 and the cluster-center features
   661	in the static memory bank; (ii) The ClusterNCE loss 𝑚
   662	between 𝑞 𝑚
   663	𝑑𝑦𝑛
   664	and the infrared cluster-center features in the dynamic memory bank.
   665	𝑚
   666	Both 𝑚
   667	𝑠𝑡𝑎 and 𝑑𝑦𝑛 are defined as Eq. (8).
   668	(
   669	)
   670	exp 𝑞 𝑚 ⋅ 𝜑𝑚,𝑏
   671	+ ∕𝜏
   672	𝑚
   673	(8)
   674	(
   675	)
   676	𝑏 = − log ∑ 𝐾
   677	𝑚
   678	exp 𝑞 𝑚 · 𝜑𝑚,𝑏
   679	∕𝜏
   680	𝑘
   681	𝑘=0
   682	
   683	where 𝑚 denotes the modality, 𝑚 ∈ {𝑣, 𝑟}. 𝑢𝑚
   684	represents the 𝑘th sample
   685	𝑘
   686	𝑚 . |⋅| denotes the number of samples in a cluster. Based on the
   687	in 𝐶𝑢,𝑖
   688	static memory bank, the loss 𝑟𝑠𝑡𝑎 (described in Section 3.4) is used to
   689	reduce the intra-modality divergence.
   690	Dynamic Memory Bank. The dynamic memory bank (denoted as
   691	𝑑 ) is first initialized with the static memory bank, and then updated
   692	the generated pseudo-labeled cross-modality cluster pairs.
   693	⟨ using ⟩
   694	𝑣 , 𝐶𝑟
   695	Let 𝐶𝑢,𝑖
   696	𝑢,𝑗 represents a pseudo-labeled cross-modality cluster pair,
   697	
   698	where 𝑏 ∈ {𝑠𝑡𝑎, 𝑑𝑦𝑛} denotes the type of memory bank (static/dynamic). 𝜑𝑚,𝑏
   699	+ represents the cluster centroid features corresponding to
   700	the cluster containing query 𝑞 𝑚 in memory bank 𝑏. 𝜑𝑚,𝑏
   701	denotes the
   702	𝑘
   703	centroid of the 𝑘th cluster for modality 𝑚 in memory bank 𝑏. 𝐾𝑚
   704	represents the number of clusters for the unlabeled images of modality
   705	𝑚. 𝜏 is a hyperparameter, which is set to 0.1 in this paper.
   706	
   707	𝑣 means the 𝑖th cluster of unlabeled visible images, and 𝐶 𝑟
   708	where 𝐶𝑢,𝑖
   709	𝑢,𝑗
   710	denotes the 𝑗th cluster of unlabeled infrared images. We update the
   711	dynamic memory bank for visible modality by integrating features
   712	from the infrared modality into the visible feature memory using a
   713	momentum updating strategy as shown in Eq. (7).
   714	
   715	𝜓𝑖𝑣 ← 𝛽𝜓𝑖𝑣 + (1 − 𝛽) 𝜓𝑗𝑟
   716	
   717	(2) Loss for Labeled Samples
   718	For labeled samples, we calculate the cross entropy loss between the
   719	predicted labels and the ground truth labels, as shown in Eq. (9).
   720	(
   721	𝑛
   722	𝑚
   723	(
   724	))
   725	| (
   726	)) 1 ∑
   727	1∑
   728	| (
   729	𝑠𝑖𝑑 = −
   730	log 𝑃 y𝑣𝑖 |𝐶 x𝑣𝑖 |𝜃
   731	−
   732	log 𝑃 y𝑟𝑗 ||𝐶 x𝑟𝑗 |𝜃
   733	(9)
   734	|
   735	𝑛 𝑖=1
   736	𝑚 𝑗=1
   737	|
   738	
   739	(7)
   740	
   741	𝑣 and 𝐶 𝑟 ,
   742	where 𝜓𝑖𝑣 and 𝜓𝑗𝑟 represent the dynamic centers of clusters 𝐶𝑢,𝑖
   743	𝑢,𝑗
   744	
   745	respectively. 𝛽 is a balancing factor, which is set to 0.1 in our paper.
   746	By using Eq. (7), the center of the 𝑖th visible cluster becomes closer to
   747	that of the 𝑗th infrared cluster. The dynamic memory bank can guide
   748	the feature extraction model to learn more similar features for samples
   749	within the cross-modality cluster pair, thereby further mitigating the
   750	discrepancy between different modalities to some extent.
   751	Nearest Neighbor Class based Sampler. To fully utilize the sample
   752	data and enhance the robustness of the model, we design a nearest
   753	neighbor class based sampler(NNC-sampler) for both the unlabeled
   754	and labeled samples. Here, we present the construction process of a
   755	nearest neighbor class-based sampler on unlabeled infrared samples as
   756	an illustrative example.
   757	For unlabeled infrared samples, we randomly select one sample
   758	from each cluster to construct a sub-dataset. Then, the features of samples in the sub-dataset are extracted with the current network, denoted
   759	
   760	where 𝑛 and 𝑚 represent the number of labeled visible and infrared
   761	images in the current mini-batch, respectively. y𝑣𝑖 and y𝑟𝑗 respectively
   762	(
   763	)
   764	represent the corresponding true labels of x𝑣𝑖 and x𝑟𝑗 , 𝐶 x𝑣𝑖 |𝜃 and
   765	(
   766	)
   767	𝐶 x𝑟𝑗 |𝜃 respectively denote the predicted labels for x𝑣𝑖 and x𝑟𝑗 through
   768	the classifier 𝐶(⋅).
   769	For clarity, we define 𝑠𝑡𝑎 and 𝑑𝑦𝑛 as follows.
   770	𝑠𝑡𝑎 = 𝑣𝑠𝑡𝑎 + 𝐿𝑟𝑠𝑡𝑎
   771	
   772	(10)
   773	
   774	𝑑𝑦𝑛 = 𝑣𝑑𝑦𝑛 + 𝐿𝑟𝑑𝑦𝑛
   775	
   776	(11)
   777	
   778	The total loss function of our approach is defined as Eq. (12).
   779	 = 𝑠𝑖𝑑 + 𝜆𝑠𝑡𝑎 + (1 − 𝜆)𝑑𝑦𝑛
   780	6
   781	
   782	(12)
   783	
   784	Pattern Recognition 165 (2025) 111669
   785	
   786	X. Zhu et al.
   787	
   788	where 𝜆 is the weight coefficient. 𝑠𝑖𝑑 aims to improve the discriminability of the model by using the labeled information, and guides the
   789	model to make better use of the unlabeled data. 𝑠𝑡𝑎 facilitates alleviating the intra-modality divergence in the model’s learning process. 𝑑𝑦𝑛
   790	helps to reduce the discrepancy between different modalities.
   791	During inference, we only utilize the well-trained feature extraction
   792	model, without other components of the proposed framework. Specifically, query images and gallery images from two modalities are fed
   793	into the feature extraction module to extract features. Then, we perform
   794	inference based on the distance between the extracted features.
   795	

codex
我已经拿到前三篇的方法链条和关键消融。接下来补读后三篇，里面分别是视频语义自对齐、污染鲁棒性和跨域异质伪标签，叙事结构会不太一样。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf' - | nl -ba | sed -n '36,640p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    36	I. I NTRODUCTION
    37	ERSON re-identification (Re-ID) [1], which attempts
    38	to identify a particular pedestrian across multiple nonoverlapping cameras, has attracted increasing attention
    39	because of its applications in monitoring, such as cross-scene
    40	crowd counting, criminal investigation, and finding people of
    41	interest (e.g., criminals or lost kids). Thanks to the significant
    42	progress in deep learning, person Re-ID has accomplished
    43	impressive performance. However, Re-ID remains a challenging undertaking owing to the presence of miscellaneous
    44	occlusions (e.g., vehicles, trees, and other pedestrians), complex camera environments, illumination changes, etc.
    45	
    46	P
    47	
    48	Received 8 May 2024; revised 6 November 2024; accepted 17 January 2025.
    49	Date of publication 20 January 2025; date of current version 6 June 2025. This
    50	work was supported in part by the National Natural Science Foundation of
    51	China under Grant 62271143 and in part by the Big Data Computing Center
    52	of Southeast University. This article was recommended by Associate Editor
    53	A. Iosifidis. (Corresponding author: Xiaobo Lu.)
    54	The authors are with the School of Automation, Southeast University,
    55	Nanjing 210096, China, and also with the Key Laboratory of Measurement
    56	and Control of Complex Systems of Engineering, Ministry of Education,
    57	Nanjing 210096, China (e-mail: zhidan_r@163.com; 230198548@seu.edu.cn;
    58	xblu2013@126.com; wx1204@seu.edu.cn; liu_wei@seu.edu.cn).
    59	Digital Object Identifier 10.1109/TCSVT.2025.3531883
    60	
    61	Fig. 1. Instances of video-based Re-ID handicaps. As illustrated by the
    62	images boxed in red rectangles, person appearance is influenced by (a-b)
    63	pedestrian detection errors and (c) occlusions.
    64	
    65	Generally, existing work on person Re-ID can be summarized into two classifications, i.e., image-based Re-ID [2], [3]
    66	and video-based Re-ID [4], [5]. For the former, the gallery and
    67	query samples are images, while for the latter, the gallery and
    68	query samples are represented by video clips. Compared with
    69	individual images, video data with multiple frames contains
    70	richer appearance and temporal information. Therefore, we are
    71	motivated to study the topic of person Re-ID in the video
    72	setting.
    73	The primary issue of video-based person Re-ID lies in
    74	aggregating discriminative information from several frames in
    75	video sequences. Most existing approaches employ a global
    76	strategy, learning a global-level representation of the pedestrian target as a single embedding vector. However, the features
    77	learned from the entire bounding boxes might contain misleading information due to occluding objects and background
    78	clutter. On the other hand, global feature representation learning may ignore fine-grained information that contains essential
    79	cues for recognizing identities.
    80	Some recent work has shown promising results in solving
    81	the Re-ID problem by generating several local embedding
    82	vectors, i.e., one for each component of the target object.
    83	Nonetheless, as shown in Fig. 1, learning such part-based
    84	representations is inherently difficult due to the ubiquitous
    85	misalignment problem, which is frequently caused by inaccurate human detection, partial occlusions, camera viewpoint
    86	changes, or body posture variations. In order to alleviate the
    87	misalignment issue, previous solutions are generally divided
    88	into two categories: hard stripe-based [6] and external cuesbased [7]. The first group leverages uniform partitioning
    89	to generate the horizontal stripe parts for extracting local
    90	
    91	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
    92	and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
    93	See https://www.ieee.org/publications/rights/index.html for more information.
    94	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.
    95	
    96	5308
    97	
    98	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
    99	
   100	feature representations. Inevitably, the stripe-based partition
   101	is sensitive to large background clutter and is too coarse to
   102	achieve precise semantic alignment. The second group relies
   103	on external detectors, such as pose estimation [8] and body
   104	parsing [9], to locate human parts for semantic and spatial
   105	alignment. This strategy results in a significant computational
   106	overhead and is limited by complicated occlusions. More
   107	crucially, the distinctive individual accessories (e.g., backpacks
   108	and handbags), which are the potentially important contextual
   109	clues for pedestrian recognition, are misinterpreted as background by these pre-trained external models.
   110	In this work, we propose a novel video-based person
   111	Re-ID framework, termed Context-Aided Semantic-Aware
   112	Self-Alignment (CSSA), which can locate different body
   113	components and achieve pixel-level semantic alignment without prior human topology information. CSSA jointly learns
   114	pixel-level part-aligned representations and semantic-aligned
   115	global-level representations in an end-to-end manner. In particular, we design a classifier trained with human body
   116	pseudo-labels to achieve part estimation. For image frames in
   117	the input video clips, we first separate the entire foreground
   118	from the background based on the reasonable hypothesis that
   119	the feature maps exhibit a stronger response in the foreground
   120	than in the background [10], [11]. To ensure semantic consistency, we select a fixed number of video sequences for
   121	each identity and cluster their foreground pixels. The cluster
   122	IDs are employed for human body parts as pseudo-labels.
   123	We iteratively generate pseudo-labels to supervise the semantic
   124	estimation. The confidence maps of the classifier are used to
   125	obtain the part-aligned representations.
   126	For background clutters, especially occlusions, we further
   127	propose the Context-Aided Region Enhancement (CARE)
   128	module, which aims to highlight responses from relevant
   129	regions and restrain noise interference responses. Existing
   130	spatial attention modules do not fully utilize the temporal
   131	context knowledge in the whole video sequence. By contrast,
   132	CARE integrates global spatio-temporal information into a
   133	video-level feature and adopts a concise autoencoder network
   134	to suppress noise. Then, a cosine similarity matrix is computed
   135	to highlight macro-visual patterns associated with the target
   136	pedestrian.
   137	In terms of global-level representation learning, we propose the Semantic-Aware Global Feature Alignment (SGFA)
   138	scheme for generating pair-wise semantic-aligned global features. Our key observation is that the contribution of the
   139	same semantic part is inconsistent across various video clips.
   140	Accordingly, SGFA calculates the adjustment weight based
   141	on the confidence map of each semantic part and obtains the
   142	video-level global features through the weighted aggregation
   143	of the feature maps.
   144	In summary, the main contributions of this article are as
   145	follows:
   146	• We propose the CSSA framework that addresses appearance misalignment in video person Re-ID while learning
   147	pixel-level part-aligned representations and semanticaligned global-level representations in an end-to-end
   148	manner. CSSA can locate different body parts and
   149	achieve pixel-level semantic alignment without prior
   150	human topology information.
   151	• We propose a novel Context-Aided Region Enhancement (CARE) module, which can efficiently highlight
   152	
   153	macro-visual patterns associated with the target pedestrian and suppress noise caused by factors like background clutters and occlusions.
   154	• A Semantic-Aware Global Feature Alignment (SGFA)
   155	method is proposed for generating pair-wise semanticaligned global representations, which play an essential
   156	role in both the training and inference phases.
   157	• The proposed CSSA achieves superior performance on
   158	three video-based person Re-ID benchmarks compared
   159	with current state-of-the-art approaches.
   160	II. R ELATED W ORK
   161	A. Video-Based Person Re-ID Models
   162	Recently, video-based Re-ID has attracted widespread attention from researchers in academia and industry [12]. Compared
   163	with image data, video data consisting of multiple frames
   164	contain richer temporal and spatial information, which can be
   165	leveraged to alleviate many issues such as motion blurs and
   166	occlusions, thereby achieving more robust Re-ID [13]. This
   167	part reviews some closely related work on video-based Re-ID.
   168	Some researchers use optical flow for encoding the
   169	short-term temporal cues between adjacent image frames [14],
   170	[15], [16]. Chung et al. [14] design a two-stream siamese
   171	convolutional neural network for processing spatial content
   172	and optical flow information. Simonyan et al. [15] process
   173	optical flow and color features concurrently for the following
   174	post-fusion of their separate classification scores. Feichtenhofer et al. [16] exploit motion information based on an input
   175	stack of several vertical and horizontal optical flow frames.
   176	Temporal pooling is extensively utilized for aggregating
   177	features across different time stamps [17]. Wu et al. [18]
   178	design the feature embedding function as an end-to-end CNN
   179	model with temporal average pooling. Zheng et al. [19] adopt
   180	mean/max pooling for generating video-level representations.
   181	In [20], a global average pooling layer is employed for transforming the spatial feature maps from different spatial regions.
   182	Temporal pooling demonstrates intriguing efficacy, but also
   183	results in a significant loss of discriminative information.
   184	Recurrent Neural Network (RNN) is employed for representation construction in video person Re-ID. In order to
   185	learn the sequence-level feature representations, Yan et al. [21]
   186	introduce the long-short term memory model for learning the
   187	frame-wise human region information from all time stamps.
   188	In [22], attentive temporal pooling is simultaneously learning with spatial pooling layers and recurrent-convolutional
   189	networks, guiding the proposed ASTPN to achieve efficient
   190	temporal information extraction. Liu et al. [23] leverage a
   191	refinement recurrent unit for restoring the missing information
   192	and suppressing the noisy part of the current frame feature
   193	representations by referring to previous frames. Due to the
   194	complicated structure, RNN is difficult to train and introduces
   195	a large number of parameters, making it show certain limitations in modeling temporal cues for Re-ID tasks.
   196	3D convolution is expert in encoding the relative position
   197	and handling local temporal relations. In [24], 3D convolution
   198	models temporal information while maintaining the appearance representation quality. Li et al. [25] employ a multi-scale
   199	3D convolutional layer implemented using multiple parallel
   200	temporal convolution kernels with distinct temporal ranges.
   201	Qiu et al. [26] disassemble the 3D convolutional filters
   202	
   203	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.
   204	
   205	RAN et al.: CONTEXT-AIDED SEMANTIC-AWARE SELF-ALIGNMENT FOR VIDEO-BASED PERSON RE-IDENTIFICATION
   206	
   207	into temporal and spatial parts for mining relations from
   208	videos. Tran et al. [27] build a homogeneous architecture with
   209	3D convolution kernels that models appearance and motion
   210	simultaneously.
   211	Graph-based models have been applied to person Re-ID.
   212	Yao et al. [28] design the graph network to represent the
   213	semantic information of local regions. In [29], GPNet leverages a graph model to mine multi-granularity knowledge.
   214	In order to enhance information propagation across features,
   215	Wu et al. [30] propose a graph-based representation learning strategy to establish associations between node features.
   216	Shi et al. [31] construct the graph-based network for learning
   217	higher-order information about pedestrian attributes.
   218	Some works [12], [13], [32] use attention mechanism to
   219	exploit spatiotemporal information. Hou et al. [13] utilize
   220	diverse attention models to learn fine-grained visual information and long-range dependencies. In [12], SINet constructs the
   221	salient-to-broad module to gradually expand the attention area
   222	of adjacent image frames. Wu et al. [32] design the attention
   223	module to enhance the features of target objects. However, all
   224	these methods are sensitive to appearance misalignments that
   225	generally exists across video frames.
   226	B. Part-Based Person Re-ID Models
   227	Part-level features provide fine granularity for describing the images. Part-based approaches learn part aggregated
   228	feature representations, making it robust against misalignment [1]. According to the method of obtaining body part
   229	locations, previous studies can be roughly summarized into
   230	three categories:
   231	1) Rigid Stripe-Based Methods: Some work directly partitions each bounding box into stiff vertical or horizontal streaks
   232	for local-level representation learning. Wang et al. [33] adopt
   233	a multi-branch deep network architecture, which has branches
   234	for obtaining local feature representations from horizontal
   235	stripes with different granularities. To capture discriminative
   236	information at different spatial scales, Zheng et al. [34]
   237	introduce horizontal stripes of various sizes that can integrate progressive cues. Yan et al. [6] design a partitioning
   238	method that separates the the entire feature map into seven
   239	specific parts using the symmetrical information belonging to
   240	the human body structure and the degree of discriminating
   241	knowledge offered by various areas. However, the stripe-based
   242	partition is helpless against background noise and semantic
   243	misalignment.
   244	2) Auto-Localization Based Methods: Some research utilizes regional selection sub-networks in order to automatically
   245	locate the body parts. Li et al. [35] leverage spatial transformer networks with novel spatial constraints for learning and
   246	localizing deformable object parts. Yao et al. [36] employ a
   247	local loss network that automatically generates a bounding box
   248	for each part. Shi et al. [37] design a reinforcement learning
   249	scheme to determine the number and sizes of horizontal
   250	streaks. However, these methods either generate a constant
   251	number of body parts or cannot handle occluded regions.
   252	Besides, the positioning of body parts is still rough and has a
   253	lot of overlap.
   254	3) Extra Semantic-Based Methods: Many approaches
   255	exploit additional semantics in terms of posture to identify object parts, which provides aligned part features.
   256	
   257	5309
   258	
   259	TABLE I
   260	S UMMARY OF S YMBOLS AND D EFINITIONS
   261	
   262	Zhang et al. [38] adopt an off-the-shelf DensePose model
   263	to deform the original image input into 24 semantically
   264	densely aligned part images in the UV space. To aggregate
   265	pixel-wise features weighted by a series of confidence maps,
   266	Guo et al. [39] use the state-of-the-art body parsing framework CE2P for predicting the semantic human part masks.
   267	In [40], a pre-trained pose estimator is introduced to locate
   268	human landmarks and guide robust representation fabrication.
   269	Miao et al. [41] employ a trained human posture estimation
   270	method to supply visible key-point landmarks, which indicate
   271	informative and reliable areas for both gallery and probe
   272	images. There are two key drawbacks of these approaches:
   273	First, they are not capable of identifying individual items
   274	such as backpacks and handbags that are critical for person
   275	Re-ID. Second, the off-the-shelf models are vulnerable to various Re-ID scenarios and cannot correct semantic estimation
   276	mistakes during training.
   277	III. P ROPOSED M ETHOD
   278	We propose a novel CSSA framework for video-based
   279	person Re-ID. It learns pixel-level part-aligned representations
   280	and semantic-aligned global-level representations in an endto-end manner. In this part, we first provide a description of
   281	the proposed CSSA framework. Next, the specifics of two
   282	core processes, i.e., CARE and SGFA, are clarified. Finally,
   283	we present the overall objective function of our framework.
   284	A. The Framework
   285	The overview of the proposed CSSA framework is illustrated in Fig. 2. Table I summarizes some key symbol
   286	q q
   287	q
   288	definitions. Given a video segment I q = {I0 , I1 , . . . , IT −1 }
   289	comprising T sampled frames, we utilize a backbone model to
   290	q
   291	q
   292	q
   293	get the intermediate feature maps X q = {X 0 , X 1 , . . . , X T −1 }.
   294	q
   295	Here X t ∈ RC×H ×W , where W, H, C, t are the width, height,
   296	channel size, and index of the video frame, respectively.
   297	Then, X q is fed into the CARE module for obtaining the
   298	q
   299	q
   300	q
   301	feature maps F q = {F0 , F1 , . . . , FT −1 }. CARE can efficiently highlight macro-visual patterns associated with the
   302	target pedestrian and suppress noise caused by factors like
   303	background clutters and occlusions.
   304	We represent the homogeneous semantic part of a video clip
   305	with all pixel representations belonging to that part. To achieve
   306	pixel-level semantic estimation, a part classifier is designed
   307	to localize distinct parts from each video frame according
   308	to F q . Suppose there are one background category and N p
   309	semantic categories in total. The classifier needs to generate
   310	N p + 1 confidence maps of separate categories for each video
   311	snippet. Then, the representation for the n-th semantic part can
   312	
   313	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.
   314	
   315	5310
   316	
   317	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
   318	
   319	Fig. 2. Overview of the proposed context-aided semantic-aware self-alignment (CSSA) framework. CSSA iteratively performs part classifier training and
   320	semantic-aligned representation learning during the training phase. The context-aided region enhancement (CARE) module is designed to efficiently highlight
   321	macro-visual patterns associated with the target pedestrian and suppress noise caused by factors like background clutters and occlusions. The semantic-aware
   322	global feature alignment (SGFA) method is proposed for generating pair-wise semantic-aligned global features.
   323	
   324	be formulated as follows:
   325	q
   326	q
   327	E n = G A P(F q ◦ Mn ),
   328	
   329	n ∈ {1, . . . , N p },
   330	
   331	(1)
   332	
   333	where ◦ refers to element-wise multiplication, G A P represents
   334	q
   335	global average pooling, and Mn ∈ RT ×H ×W means the
   336	n-th confidence map. For the clear exposition, we leverage
   337	q
   338	Mn (t, h, w) to represent the pixel confidence (t, h, w) of
   339	the semantic category n. Ideally, for the background and
   340	q
   341	occlusion parts, ∀(t,h,w) Mn (t, h, w) = 0 should be satisfied.
   342	The whole representation of local parts for training can be
   343	q
   344	q
   345	q
   346	expressed as [E 1 , E 2 , . . . , E N p ] ([·] denotes the concatenation
   347	operation). Next, the generated confidence maps are used
   348	to guide SGFA for obtaining semantic-aligned global-level
   349	representations that play a crucial role in both the training
   350	and test phases.
   351	In order to get rid of the dependence on prior human
   352	topology information, we adopt a pseudo-label-based strategy
   353	for the semantic part classifier training. Specifically, the feature
   354	maps F q are grouped into background or foreground according
   355	to the normalized L 2 -norm of pixels [10]. For Nt video
   356	sequences of each specific pedestrian, we perform a clustering
   357	algorithm with the fixed cluster number N p . We assign the
   358	
   359	labels {1, . . . , N p } to the semantic parts according to their
   360	average position from bottom to top, and the background label
   361	is assigned to 0. The pseudo-labels are then used to construct
   362	a cross-entropy loss L pc for supervising the training of the
   363	classifier [42].
   364	During the testing phase, the final distance between a gallery
   365	sample g and a query q is as follows:
   366	PN p n n n
   367	g
   368	dqg + n=1
   369	(vq · vg )dqg
   370	all
   371	(2)
   372	dqg =
   373	PN p  n n  ,
   374	1 + n=1 vq · vg
   375	n represents the cosine distance between the n-th part
   376	where dqg
   377	g
   378	feature pair, dqg is the distance of the global-level features,
   379	n
   380	n
   381	vq , vg represent the binary visibility scores for the n-th part
   382	of q and g, with 1/0 corresponding to visible/invisible parts.
   383	The binary visibility scores are computed by
   384	
   385	
   386	 1, ∃(t, h, w) ∈
   387	n
   388	v =
   389	(3)
   390	{(t, h, w)|argmaxi Mi (t, h, w) = n}
   391	
   392	 0, other wise
   393	
   394	where n ∈ {1, . . . , N p }.
   395	
   396	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.
   397	
   398	RAN et al.: CONTEXT-AIDED SEMANTIC-AWARE SELF-ALIGNMENT FOR VIDEO-BASED PERSON RE-IDENTIFICATION
   399	
   400	5311
   401	
   402	Fig. 3. Model architecture of CSSA in the testing stage. The final distance is calculated using a part-to-part matching strategy. The identical semantic part
   403	contributes differently to the distance across various video sequences. For conciseness, CSSA is represented here with N p = 5.
   404	
   405	B. Context-Aided Region Enhancement
   406	We propose CARE to highlight the responses of relevant
   407	regions to the primary object in the video clip and suppress
   408	responses triggered by disturbances. Existing spatial enhancement models [43] typically consider only the content of
   409	each frame, frequently overlooking the temporal information.
   410	However, such models may fail to accurately determine which
   411	regions are informative without temporal context. For instance,
   412	as shown in Fig. 1, the identification information of the main
   413	target object in some frames is incomplete when disturbance
   414	between different targets occurs. Observation shows that the
   415	majority of images in a video clip are unaffected by disturbance, which enlightens us to infer valuable regions for each
   416	frame by means of global temporal context in the video clip.
   417	The architecture of CARE is illustrated in the upper part
   418	of Fig. 2. First, we leverage temporal average pooling and
   419	spatial Generalized-Mean pooling [44] to aggregate the output
   420	feature map of the backbone network into a video-level embedding z q . Next, in order to mitigate the impact of noise, we build
   421	an encoder-decoder network for treating z q . Specifically, the
   422	dimension of z q is first squeezed via a convolutional block
   423	of the form Conv-LeakyReLU, then restored to the original
   424	dimensions by another convolution block. Formally, the operation process is as follows:
   425	z q = δ(W e z q ),
   426	zbq = δ(W d z q ),
   427	
   428	(4)
   429	C
   430	r ×C
   431	
   432	where δ denotes the ReLU [43] function, W e ∈ R
   433	,
   434	C
   435	W d ∈ RC× r represent the parameters of the two convolutional
   436	blocks, z q , zbq refer to the output of the two blocks, and r is
   437	the reduction ratio. Importantly, we impose a cross-entropy
   438	loss with identity labels on z q as follows:
   439	LC A R E = Lce (W c z q ),
   440	
   441	where P q (t, h, w) denotes the cosine similarity between zbq
   442	and X q (t, h, w), ∥·∥ is characterized as the L 2 norm of the
   443	feature vector. Moreover, we introduce a residual connection
   444	with weight for the intermediate feature maps X q ,
   445	F q (t, h, w) = X q (t, h, w) + P q (t, h, w)X q (t, h, w),
   446	t ∈ {1, . . . , T − 1}, w ∈ {0, . . . , W − 1},
   447	h ∈ {0, . . . , H − 1},
   448	(7)
   449	where F q (t, h, w) stands for the output feature vector at
   450	spatial position (t, h, w).
   451	C. Semantic-Aware Global Feature Alignment
   452	With pixel-level semantic estimation, our proposed framework combines the whole body feature and local part
   453	representations. Most existing methods [2], [4], [46] simply
   454	extract global features based on the entire region of each
   455	image frame. However, as illustrated in Fig. 3, pedestrian
   456	appearance is affected by occlusions and pedestrian detection
   457	errors, which results in the identical semantic parts contributing differently to the global-level feature representations
   458	across various video sequences. Accordingly, we propose the
   459	Semantic-Aware Global Feature Alignment (SGFA) method to
   460	generate pair-wise semantic-aligned global features with the
   461	help of the confidence maps.
   462	The lower part of Fig. 2 illustrates the pipeline of SGFA. For
   463	two video segments I q , I g in a mini-batch, we first utilize the
   464	CARE module to get their feature maps F q and F g . In order
   465	to quantify the contribution of each semantic part, we design
   466	a semantic score according to the results of part estimation:
   467	q
   468	
   469	(5)
   470	
   471	where W c refers to the parameters of the classification layers
   472	and Lce denotes the cross-entropy loss function. In this way,
   473	the generated feature zbq removes noise while retaining identity
   474	information. Typically, the features with analogous semantics
   475	have higher cosine similarity [45]. Accordingly, we obtain
   476	the cross-pixel cosine similarities between zbq and each pixel
   477	in X q :
   478	t ∈ {1, . . . , T − 1},
   479	T
   480	zbq X q (t, h, w)
   481	q
   482	P (t, h, w) = q
   483	, w ∈ {0, . . . , W − 1},
   484	zb ∥X q (t, h, w)∥
   485	h ∈ {0, . . . , H − 1},
   486	(6)
   487	
   488	q
   489	
   490	Jn = {(t, h, w)|Mn (t, h, w) > 0.5},
   491	X q
   492	q
   493	sn =
   494	Mn (t, h, w), n ∈ {1, . . . , N p },
   495	
   496	(8)
   497	
   498	q
   499	(t,h,w)∈Jn
   500	
   501	g
   502	
   503	g
   504	
   505	Jn = {(t, h, w)|Mn (t, h, w) > 0.5},
   506	X g
   507	g
   508	sn =
   509	Mn (t, h, w), n ∈ {1, . . . , N p },
   510	
   511	(9)
   512	
   513	g
   514	(t,h,w)∈Jn
   515	
   516	q
   517	
   518	g
   519	
   520	where sn , sn indicate the n-th part semantic scores for
   521	q
   522	g
   523	F q and F g . Jn , Jn denote the point sets with confidence
   524	q
   525	g
   526	higher than 0.5 in Mn and Mn , respectively. In the next
   527	step, we leverage the confidence maps to generate the mask
   528	{U1 , . . . , U N p } about each semantic part. Un (t, h, w) is set
   529	to 1 if argmaxi Mi (t, h, w) = n or 0 for other parts. Note
   530	that we omit the superscript of the mask. In this case,
   531	
   532	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.
   533	
   534	5312
   535	
   536	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
   537	
   538	the final pair-wise semantic-aligned global features can be
   539	formulated as:
   540	PN p q g
   541	q
   542	(sn · sn )G A P(F q ◦ Un )
   543	q
   544	,
   545	(10)
   546	E g = n=1 P N
   547	q
   548	g
   549	p
   550	(s
   551	·
   552	s
   553	)
   554	i
   555	i=1 i
   556	g
   557	Eg =
   558	q
   559	
   560	PN p
   561	
   562	q
   563	g
   564	g
   565	g
   566	n=1 (sn · sn )G A P(F ◦ Un )
   567	,
   568	PN p q g
   569	)
   570	·
   571	s
   572	(s
   573	i
   574	i=1 i
   575	
   576	(11)
   577	
   578	g
   579	
   580	where Un , Un are the n-th part masks for Iq and Iq .
   581	D. Objective Function
   582	In the training phase, we construct two joint losses,
   583	denoted as L pt and Lgl , which are enforced on concatenated part features and pair-wise semantic-aligned global
   584	features, respectively. Following [47], each basic joint loss
   585	consists of a triplet loss [48] and a cross-entropy loss.
   586	Notably, the cross-entropy loss in Lgl can be formulated as
   587	follows:
   588	N
   589	
   590	Lce
   591	gl =
   592	
   593	TABLE II
   594	P ROPERTIES OF THE V IDEO -BASED Re-ID DATASETS
   595	
   596	N
   597	
   598	v X
   599	v
   600	1 X
   601	g
   602	q
   603	(Lce (W g E g ) + Lce (W g E g )),
   604	Nv Nv
   605	
   606	(12)
   607	
   608	q=1 g=1
   609	
   610	where Nv represents the number of video clips in a mini-batch,
   611	and W g is the parameters regarding the classification layers.
   612	The overall objective function Ltotal utilized to optimize the
   613	framework is formulated as follows:
   614	Ltotal = L pt + Lgl + L pc + LC A R E .
   615	
   616	(13)
   617	
   618	IV. E XPERIMENTS
   619	A. Datasets and Evaluation Metrics
   620	We evaluate our proposed approach on three challenging
   621	video-based Re-ID datasets, i.e., MARS [19], DukeMTMCVideoReID (Duke-Video) [18], and LS-VID [17]. The statistics of these datasets are summarized in Table II.
   622	MARS [19] is a large-scale benchmark dataset for video
   623	Re-ID. It contains 17,503 tracklets with 1,261 identities
   624	captured by the six cameras. In addition, there are 3,248 distracter tracklets in MARS, which makes it more realistic
   625	and challenging. All video clips in the dataset are generated using the Deformable Part Model (DPM) detector
   626	and Generalized Maximum Multi Clique Problem (GMMCP)
   627	tracker.
   628	DukeMTMC-VideoReID [18] is another widely-used benchmark dataset, which includes 4,832 tracklets associated
   629	with 1,812 identities. Each tracklet contains 168 frames on
   630	average, and the bounding boxes of pedestrians are annotated
   631	manually. Videos in this dataset are captured by eight cameras.
   632	DukeMTMC-VideoReID is divided into a training set and a
   633	test set: the former consists of 2,196 tracklets of 702 identities,
   634	while the latter contains 3,338 tracklets of the remaining
   635	702 identities.
   636	LS-VID [17] is obtained by Faster R-CNN. It consists of
   637	14,943 tracklets of 3,772 identities, captured using twelve
   638	outdoor cameras and three indoor cameras. LS-VID contains
   639	many challenging scenarios, such as drastic illumination variation and indoor/outdoor cameras, so it can be considered as
   640	the closest dataset to real-world environments.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf' - | nl -ba | sed -n '42,735p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    42	I. I NTRODUCTION
    43	
    44	C
    45	
    46	ORRUPTION-INVARIANT person re-identification
    47	(CI-ReID) [1] aims at re-identifying a target person
    48	of interest even when image corruptions occur. In addition
    49	Received 18 May 2024; revised 1 September 2024; accepted 25 September
    50	2024. Date of publication 2 October 2024; date of current version 13 February
    51	2025. This work was supported in part by the Talent Introduction Program for
    52	Youth Innovation Teams of Shandong Province, in part by the National Natural
    53	Science Foundation of China under Grant 62202280, and in part by Shandong
    54	Provincial Natural Science Foundation under Grant ZR2021QF017. This
    55	article was recommended by Associate Editor H.-C. Shih. (Corresponding
    56	authors: Peng Zhang; Caifeng Shan.)
    57	Xinyu Zhang is with the College of Electrical Engineering and Automation,
    58	Shandong University of Science and Technology, Qingdao 266590, China
    59	(e-mail: zhangxinyu@sdust.edu.cn).
    60	Peng Zhang is with the College of Computer Science and Engineering,
    61	Shandong University of Science and Technology, Qingdao 266590, China
    62	(e-mail: pengzhang_skd@sdust.edu.cn).
    63	Caifeng Shan is with the College of Electrical Engineering and Automation,
    64	Shandong University of Science and Technology, Qingdao 266590, China, and
    65	also with the State Key Laboratory for Novel Software Technology and the
    66	School of Intelligence Science and Technology, Nanjing University, Nanjing
    67	210023, China (e-mail: caifeng.shan@gmail.com).
    68	Digital Object Identifier 10.1109/TCSVT.2024.3472122
    69	
    70	to perturbations coming from factors such as illumination
    71	difference, deformable poses and occlusion in traditional
    72	ReID, CI-ReID involves extra noises induced by objective
    73	factors in real-world deployment, such as villainous weather,
    74	digital noise, imaging blur and image compression, etc. These
    75	factors result in issues like blurred edge, occluded feature,
    76	color jittering, etc., which cause corrupted image quality and
    77	contaminate intrinsic pedestrian characteristics. Therefore,
    78	CI-ReID confuses existing ReID models and yields inferior
    79	identification performance. This limits the scalability of the
    80	existing ReID methods to the wild scene.
    81	Though CI-ReID is valuable for practical deployment, it has
    82	been overlooked by the mainstream ReID research for a long
    83	time. Until 2021, Chen et al. [1] pioneered the research that
    84	established benchmarks and built a baseline for CI-ReID.
    85	Aided by some tricks, i.e., local-based augmentation [2],
    86	consistency identity loss and inference before BNNeck [3], the
    87	baseline improved robustness for inference with corruptions.
    88	In spite of this, such primary work is still far from satisfaction
    89	due to significant dataset bias [4] between training phase and
    90	inference phase. In another word, the baseline model trained
    91	on clean/un-corrupted data generalizes poorly to corrupted
    92	scene due to the large distribution gap.
    93	To better address the above issue, we carefully investigate
    94	genesis of image corruptions and its impact on data distribution. As illustrated in [5], a total of 20 typical types of
    95	corruptions could degrade person ReID performance in realworld deployment. These corruptions can be categorized into
    96	four classes according to their causes, i.e., noise, blur, weather,
    97	and digital. Fig. 1(a) shows examples of these different types
    98	of corruptions. It can be observed that these corruptions
    99	severely lower the image quality which further contaminates intrinsic pedestrian characteristics in monitoring scene.
   100	We further visualize the distribution of pedestrian images with
   101	different corruptions using t-SNE [6] in Fig. 1(b). From the
   102	figure, we can see that these corruptions bias seriously data
   103	distribution, which cause samples affected by the same type of
   104	corruption to be clustered together, while there exists a certain
   105	gap among samples affected by different types of corruptions.
   106	This suggest CI-ReID can be regarded as pedestrian retrieval
   107	between heterogeneous data, where each type of image corruption can be seen as an image modality. This inspires us to
   108	address CI-ReID from the perspective of multi-modal feature
   109	learning, and learn identity-related discriminative features in
   110	a modality-unified semantic subspace.
   111	Inspired by the insights mentioned above, this paper introduces a pedestrian semantic alignment framework for CI-ReID
   112	that aligns distribution of clean and corrupted pedestrian
   113	
   114	1051-8215 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
   115	See https://www.ieee.org/publications/rights/index.html for more information.
   116	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.
   117	
   118	ZHANG et al.: CORRUPTION-INVARIANT PERSON RE-IDENTIFICATION VIA COARSE-TO-FINE FEATURE ALIGNMENT
   119	
   120	1085
   121	
   122	Fig. 1. An illustration of samples in CI-ReID. (a) The top two rows list different types of image corruptions in real-world deployment, and the bottom row
   123	shows image degradation under different levels of corruptions; (b) the distribution of corrupted samples using t-SNE [6].
   124	
   125	samples in the semantic space. To ease the feature alignment, we adopt a coarse-to-fine feature alignment strategy.
   126	Specifically, a Dual-stream Feature Alignment (DFA) model
   127	is proposed to achieve coarse alignment, which includes a
   128	siamese Teacher Feature Extraction (TFE) network and a
   129	Student Feature Extraction (SFE) network. It is motivated
   130	by the idea of knowledge distillation [7] that the pre-trained
   131	TFE on clean dataset supervises the further training of the
   132	pre-trained SFE network on the corrupted dataset. This idea
   133	regards one clean sample of the target pedestrian as the
   134	exemplar and forces semantics of its corresponding corrupted
   135	samples to be close with the exemplar. This helps SFE restore
   136	pedestrian features from corrupted images and further benefits
   137	person ReID. Particularly, we propose a weighted contrastive
   138	loss, and the weight is positively correlated to the distance
   139	between samples, which adaptly imposes a larger penalty to
   140	challenging samples, such as samples with severe corruptions
   141	in Level 5 (Lv5) as shown in Fig. 1(a).
   142	Recent studies [8] suggests the strict modality alignment
   143	in latent space can lead to loss of mode-specific information,
   144	resulting in sub-optimal performance. However, by expanding
   145	the lower bound of the approximate alignment, more useful information can be retained, and better alignment can
   146	be obtained. In another aspect, it is difficult to completely
   147	restore features from corrupted samples from the perspective
   148	of information-theoretic argument. Therefore, we propose to
   149	relax the regularization of strict alignment and construct the
   150	latent structure between modalities. Motivated by the Brownian bridge principle [9], [10], this paper introduces the Bridge
   151	Feature Generation (BFG) module to generate the intermediate features between modalities. By filling the intermediate
   152	features between clean and corrupt modalities, a seamless
   153	semantic structure is constructed. The BFG extends the range
   154	of corrupt modality in the semantic space and achieving fine
   155	alignment. It is worth noting that we adapt the re-parameter
   156	trick into BFG to generate intermediate features in stochastic
   157	path of Brownian motion. Benefiting from modality transition,
   158	
   159	the BFG expands intra-person cluster, which enables the
   160	proposed method to be more capable of achieving person ReID
   161	under corruption scene.
   162	In summary, the main contributions of this work are threefold:
   163	• Based on our insights on corrupt images, we propose to
   164	formulate CI-ReID as a modal alignment problem, aiming
   165	to recover pedestrian features from corrupted images
   166	by aligning clean and corrupt modalities. Additionally,
   167	we introduce a novel modality alignment framework for
   168	this problem.
   169	• We adopt a coarse-to-fine feature alignment strategy to
   170	optimize the framework, utilizing a novel dual-stream
   171	feature alignment module to achieve coarse alignment
   172	by aligning features of corrupted samples to their corresponding clean features under teacher-student paradigm.
   173	On this basis, a novel bridge feature generation module
   174	is introduced to construct the latent structure between
   175	modalities for fine alignment.
   176	• Extensive experiments are conducted on three benchmark
   177	datasets, e.g., Market-1501 [11], CUHK-03 [12], and
   178	MSMT-17 [13]. Under four different evaluation settings,
   179	the proposed method outperforms the state-of-the-art
   180	ReID methods with a large margin.
   181	The remaining parts of this paper are organized as follows.
   182	In Section II, we review works most closely related to this
   183	paper. In Section III, we introduced the proposed method
   184	in detail. Section IV presents all the experimental content.
   185	Section V summarizes the paper.
   186	II. R ELATED W ORK
   187	A. Person Re-Identification (Re-ID)
   188	In the earlier years, ReID was considered as a sub-problem
   189	of cross-camera tracking until Gheissari et al. [14] first defined
   190	ReID as an independent topic. ReID can be classified into
   191	close-world problem and open-world one. With the saturation
   192	
   193	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.
   194	
   195	1086
   196	
   197	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025
   198	
   199	of performance in closed-world scenarios, the focus of ReID
   200	research has gradually shifted to open-world scenarios. For
   201	example, Zhou et al. [15] proposed the concept of omni-scale
   202	learning to tackle the need for attention to features at different
   203	scales in ReID. Li et al. [16] developed a Diverse and
   204	Compact Transformer to learn discriminative representations.
   205	Due to various challenges in open-world scenarios, such as
   206	diverse data collection methods, uncontrollability of pedestrians, and environmental factors, pedestrian re-identification
   207	faces many different challenges [1]. To address these issues
   208	in open scenarios, Pang et al. [17] proposed a cross-modality
   209	hierarchical clustering and refinement method to address the
   210	task of cross-modal pedestrian image retrieval. Jin et al. [18]
   211	addressed the issue of occlusion in pedestrian re-identification
   212	by calculating the differences between occluded and nonoccluded pedestrians. We refer [19] for more related works.
   213	However, the study of corrupted images in the field of ReID
   214	is lacking attention. Image corruption will cause edge blur,
   215	feature occlusion, and color jitter, which will confuse existing
   216	ReID models. The study of corrupted images, considered
   217	as heterogeneous data compared to clean images, is also
   218	overlooked in the field of ReID. Therefore, this paper conducts
   219	research on CI-ReID, hoping to obtain a model with higher
   220	recognition accuracy in corrupted scenes.
   221	B. Corrupted Image Recognition
   222	In the real world, the obtained images are often affected
   223	by various disturbing factors such as noise, blur, weather
   224	conditions, or data corruption, which severely degrades image
   225	quality [20], [21] and yields superior downstream performance [1]. Research on corrupted images in the field of
   226	computer vision has a long history [22], [23]. However, since
   227	it is challenging to obtain certain types of corrupted images
   228	from the real world, synthetic methods have been employed
   229	to create datasets for corrupted images, using image quality assessment methods [20], [24] for validation. Currently,
   230	datasets such as CIFAR-10-C, CIFAR-100-C, and ImageNet-C
   231	[5] have been established for corrupted image recognition,
   232	which has drawn widespread interest in the study of corrupted images. Corruption benchmarks have been proposed in
   233	various domains, including object detection [25], pose estimation [26], and medical imaging [27]. Additionally, benchmarks
   234	for corruption-invariant person re-identification [1] have also
   235	been introduced, aiming to enhance the feature representation
   236	capability and improve the model’s generalization performance
   237	through representation learning and metric learning methods.
   238	However, the results of this work are still far from satisfaction due to significant dataset bias between training phase
   239	and inference phase. To mitigate the impact of dataset bias,
   240	Wei et al. [28] introduced a novel exponential loss that
   241	adaptively assigns weights based on the similarity of sample
   242	pairs to address the cross-modal retrieval problem. Li et al.
   243	[29] generated low-quality images through data augmentation
   244	and utilized a realistic UIESR network for reconstruction,
   245	enabling the model to enhance image texture and correct
   246	color bias. Different from them, in this work we diversify the
   247	inputs during the training phase through data augmentation,
   248	
   249	introduce weighted contrastive loss, and extract the most
   250	essential features through knowledge distillation [7], [30] to
   251	obtain a recognition model that is more robust to diverse
   252	corrupted data.
   253	C. Teacher-Student Learning
   254	Teacher-student learning regime [31] is widely utilized for knowledge distillation (KD) [7]. It builds
   255	knowledge correspondence between models to perform
   256	model compression and distillation. Inspired by KD,
   257	Tarvainen et al. [32] introduced a semi-supervised Mean
   258	Teacher model to enhance the model’s generalization capability. It forces outputs from teacher and students consistent
   259	by adding different perturbations to them. The teacher network reflects the Student’s state over time. In addition,
   260	Zhang et al. [33] developed a teacher-student learning strategy
   261	to mitigate the distribution discrepancies between imbalanced
   262	classes by constraining the predicted probabilities to achieve
   263	alignment between the classes. Zhang et al. [34] proposed a
   264	cross-resolution relational contrastive distillation approach for
   265	low-quality image recognition through cross-quality knowledge transfer, achieving feature alignment between images of
   266	different qualities. This work is also inspired by KD, but it is
   267	built on quite another formulation. In this work, the teacher
   268	acts as an exemplar regularizer that is pre-trained on clean
   269	samples. It regularizes the latent feature structure constructing
   270	from multi-source corruptions with the help of our Bridging Feature Generation module. Under this motivation, our
   271	proposed method preserves identity information as much as
   272	possible though various severe corruption happens.
   273	D. Modality Alignment
   274	The purpose of cross-modal alignment is to explore the correlations between sub-elements of multi-modal data. Modality
   275	alignment is widely used in multi-modal tasks, and the main
   276	methods include explicit alignment [35] and implicit alignment [36], [37]. If a model’s optimization objective is to
   277	maximize the alignment of sub-elements of multi-modal data,
   278	it is referred to as explicit alignment. Explicit alignment
   279	includes unsupervised [38], [39] and supervised [40], [41]
   280	methods. Unsupervised methods do not have annotations,
   281	and during the process of aligning sub-elements, the model
   282	needs to learn similarity measures and alignment methods
   283	simultaneously. Supervised methods involve annotations and
   284	enable the training of models to learn similarity measures.
   285	If the final optimization objective of the model is not an alignment task, and the alignment process is only an intermediate
   286	(or implicit) step, it is referred to as implicit alignment. Some
   287	implicit alignment methods [42] require manually constructing
   288	mappings between modalities. Other alignment methods [43]
   289	are based on attention mechanisms, which compute attention
   290	weight matrices between sub-elements of two modalities,
   291	implicitly measuring the degree of correlation between crossmodal sub-elements. For more in-depth information, we refer
   292	to [44]. In 2023, Jiang et al. [8] aligned text and image features
   293	by constructing implicit structure between modalities. This
   294	inspired us to achieve fine alignment by filling bridge features
   295	
   296	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.
   297	
   298	ZHANG et al.: CORRUPTION-INVARIANT PERSON RE-IDENTIFICATION VIA COARSE-TO-FINE FEATURE ALIGNMENT
   299	
   300	1087
   301	
   302	Fig. 2. (Viewed in color) An illustration of the proposed Coarse-to-Fine Feature Alignment Transformer. The network structure consists of one DFA module
   303	(i.e., siamese teacher-student feature extraction networks) for coarse feature alignment and one BFG module for fine feature alignment. The feature extraction
   304	network of the DFA module uses a traditional Vision Transformer (ViT) [45] structure, with the teacher model parameters fixed during training. The BFG
   305	module uses the Feature Variational Generator (FVG) to generate intermediate features, and utilizes Kullback-Leibler (KL) divergence and mean squared error
   306	to constrain the feature filling into the target region.
   307	
   308	on top of the coarse alignment using siamese teacher-student
   309	feature extraction networks.
   310	
   311	Particularly, the data generation process for the corrupted
   312	dataset DCor = {X iCor |i = 1, 2, . . . , N } is defined as follows:
   313	
   314	III. M ETHOD
   315	
   316	X iCor = G(X i , c, l)
   317	
   318	As shown in Fig. 2, our proposed Coarse-to-Fine Feature
   319	Alignment Transformer (CFAT) includes a Dual-stream
   320	Feature Alignment (DFA) module, i.e., the siamese
   321	teacher-student feature extraction networks, and a Bridge
   322	Feature Generation (BFG) module. Given a clean/uncorrupted ReID dataset D = {X i |i = 1, 2, . . . , N }, where
   323	X i represents the i-th image in the dataset, N represents
   324	the number of images in the dataset. The Teacher Feature
   325	Extraction (TFE) network and Student Feature Extraction
   326	(SFE) network are both pre-trained, with the difference
   327	being that TFE is pre-trained on D while SFE is pre-trained
   328	on the D after corrupted augmentation. Subsequently, TFE
   329	is frozen to supervise the further training of SFE. During
   330	training, our CFAT takes a coarse-to-fine alignment strategy
   331	in the latent semantic space. In the coarse alignment phase,
   332	contrastive loss is adopted to align features of corrupted
   333	augmentations to that of their corresponding clean sample.
   334	In the fine alignment phase, the BFG is further introduced to
   335	generate intermediate features between clean exemplar and its
   336	corresponding corrupted augmentations. On top of contrastive
   337	loss, the intermediate augmentations help extend the modality
   338	range by constructing a reasonable modality structure instead
   339	of forming strict feature alignment in the latent space, which
   340	further boosts CI-ReID feature learning. The following gives
   341	details of our CFAT.
   342	A. Data Augmentation via Random Corruption
   343	Due to the difficulty of collecting data with various corruptions, there are still no corrupted ReID datasets available.
   344	Following [5], this paper simulates different corruptions and
   345	generates corresponding corrupted datasets from existing ReID
   346	datasets, such as Market-1501, CUHK-03, and MSMT-17.
   347	
   348	(1)
   349	
   350	where X iCor is the corresponding corrupted sample of X i ,
   351	G(·, ·, ·) denotes the corruption generator function, c ∈
   352	{1, . . . , P} and l ∈ {1, . . . , Q} are the corruption type and
   353	corruption level to generate X iCor . In our work, we defined
   354	P = 20 types of corruption in real-world deployment with
   355	Q = 5 different corruption levels. Notably, the type and
   356	level of corruptions are randomly set with an equal probability
   357	during corrupted data augmentation unless specified. Different
   358	from [5], this paper considers the random corruption process
   359	as a kind of data augmentation and takes it into SFE network
   360	optimization. This significantly improves model robustness for
   361	corruptions.
   362	B. Coarse Alignment via Knowledge Distillation
   363	To mitigate the semantic misalignment in the feature space
   364	caused by corruption contamination, we propose a Dual-stream
   365	Feature Alignment (DFA) module for CI-ReID. The proposed DFA is built on the concept of knowledge distillation,
   366	which consists of siamese teacher-student feature extraction
   367	networks with information correspondence. In particular, the
   368	Teacher Feature Extraction (TFE) network is pre-trained on
   369	un-corrupted samples, which provides rich and reliable semantic information as supervision. It guides the Student Feature
   370	Extraction (SFE) network to obtain features on corrupted data
   371	that are semantically consistent with their clean counterpart.
   372	Such alignment enables SFE to restore rich pedestrian semantics though severe corruptions occur, ultimately generating
   373	robust pedestrian representation.
   374	As shown in Fig. 2, we adopt Vision Transformer (ViT)
   375	[45] as backbone for both TFE and SFE because of its strong
   376	representation ability. Given an input pedestrian image X ∈
   377	{D, DCor }, it is first cut into n patches and then flattened into
   378	
   379	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.
   380	
   381	1088
   382	
   383	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025
   384	
   385	vectors for feature encoding by Transformer Encoder (TE),
   386	which is defined as
   387	Z cls = C L S(L N (T E((xcls ; x1 ; x2 ; . . . ; xn ) + E pos ))) (2)
   388	where T E(·) denotes the transformer encoder, L N (·) is a
   389	linear layer, C L S(·) denotes feature extraction operation on
   390	class token, xcls is the class token, xn represents the embedding
   391	of the n-th patch, and E pos represents the position embedding.
   392	During training, we treat features of clean data as exemplars
   393	to supervise feature learning of corrupted data. Thus, we freeze
   394	pre-trained parameters of TFE while optimizing SFE on corrupted samples. To achieve the alignment, contrastive loss [46]
   395	is adopted to force semantics of corrupted augmentations and
   396	their clean counterpart consistent, which is defined as
   397	esim(Z n ,Z n )/τ
   398	1 X
   399	P
   400	Cl Cor
   401	2N
   402	N
   403	esim(Z n ,Z k )/τ
   404	N
   405	
   406	L Con = −
   407	
   408	Cl
   409	
   410	n=1
   411	
   412	Cor
   413	
   414	(3)
   415	
   416	k=1; k̸ =n
   417	
   418	where Z nCl and Z nCor denotes the feature embedding of
   419	n-th clean image and its corrupted augmentation, sim(·, ·)
   420	is the cosine similarity function. τ is a hyper-parameter that
   421	controls the model’s discrimination against negative samples,
   422	and we empirically set τ = 0.5 unless specified. However,
   423	such contrastive loss treats all samples equally, which is
   424	insufficient for addressing hard samples with severe corruption
   425	degree. To address the problem, we propose a Weighted
   426	Contrastive Loss (WCL) instead of L Con . Inspired by [19],
   427	our WCL aims to adaptively adjust penalties for challenging
   428	samples, which is realized by adding an adaptive weight ω.
   429	Thus, the improved WCL can be defined as
   430	1 X (1 − ω)esim(Z n ,Z n )/τ
   431	P2N
   432	Cl Cor
   433	N
   434	esim(Z n ,Z k )/τ
   435	N
   436	
   437	LWCL = −
   438	
   439	Cl
   440	
   441	n=1
   442	
   443	Cor
   444	
   445	(4)
   446	
   447	k=1; k̸ =n
   448	
   449	where the adaptive weight ω is obtained by calculating the
   450	cosine similarity of two inputs. As ω is negatively correlated to
   451	the similarity between samples, WCL ensures a larger penalty
   452	is enforced to the challenging samples and results in better
   453	alignment results.
   454	In addition, to enhance feature discriminability of corrupted
   455	data for pedestrian retrieval, the triplet loss is utilized on top
   456	of the SFE, defined as
   457	LTr =
   458	
   459	N h
   460	X
   461	
   462	p
   463	
   464	2
   465	
   466	2
   467	
   468	a
   469	a
   470	n
   471	Z Cor
   472	− Z Cor 2 − Z Cor
   473	− Z Cor
   474	+α
   475	2
   476	
   477	i=1
   478	
   479	i
   480	+
   481	
   482	(5)
   483	
   484	p
   485	
   486	a , Z
   487	n
   488	where Z Cor
   489	Cor and Z Cor denote anchor, positive and
   490	negative samples, respectively. α is the margin, which is
   491	empirically set to 0.3. Following the ReID regime [11], [47],
   492	[48], we adopted an identity loss to achieve classification,
   493	defined as
   494	N
   495	
   496	LID = −
   497	
   498	b
   499	1 X
   500	qi log( pi )
   501	Nb
   502	
   503	(6)
   504	
   505	i=1
   506	
   507	where Nb is the number of samples in a mini-batch, qi and
   508	pi denote the ground truth and predicted results, respectively.
   509	
   510	C. Fine Alignment via Brownian Bridge
   511	As shown in Fig. 3, the coarse alignment mentioned
   512	above, which benefits from the contrastive loss, helps alleviate the effect of corruption contamination. However, recent
   513	studies [8], [49] indicate that strict alignment achieved by
   514	contrastive loss might result in sub-optimal performance for
   515	downstream tasks. In another aspect, the perfect alignment
   516	is intractable because of the information loss of corrupted
   517	data. Therefore, we propose to relax the regularization of
   518	strict alignment and construct the latent structure between
   519	modalities. By extending the modal range through such
   520	latent structure, the seamless integration between modalities
   521	is achieved. To this end, this paper proposes the Bridging
   522	Feature Generation (BFG) module. As shown in Fig. 3, this
   523	module utilizes the principle of the Brownian bridge [9], [10]
   524	to establish a region of the Brownian bridge path between the
   525	two modalities. By using this Brownian bridge structure as
   526	guidance, we fill in intermediate features between the clean
   527	and corrupted modalities, achieving seamless transformation
   528	between modalities. This seamless transformation makes the
   529	fusion of modalities smoother and helps the model retain more
   530	identity-related discriminative features during the modality
   531	alignment process.
   532	The Brownian bridge is a particular type of Brownian
   533	motion that defines a random path (referred to as a bridge)
   534	between a pair of fixed starting and ending points (corresponding to the clean-image features and the corrupted-image
   535	features extracted by the DFA module). The structure of the
   536	Brownian bridge is illustrated in Fig. 2. According to the
   537	properties of the Brownian bridge, it follows a conditional
   538	Gaussian distribution, defined as
   539	p(Z Mid | Z Cl , Z Cor ) = N (Z Mid ;
   540	µ(Z Cl , Z Cor , t) , t (l - t)I )
   541	
   542	(7)
   543	
   544	where Z Mid is the sampled intermediate feature, Z Cl is the
   545	clean-image feature, Z Cor is the corrupted-image feature,
   546	and t is a hyper-parameter randomly chosen from [0, 1].
   547	This structure consists of two important components, i.e.,
   548	the Brownian bridge path and the sampling generation of
   549	intermediate features. Based on the Gaussian distribution in
   550	Eq. 7, a Brownian bridge path [8] is defined as
   551	Z B B = µ(Z Cl , Z Cor , t) ≜
   552	
   553	t Z Cl + (1 − t)Z Cor
   554	∥ t Z Cl + (1 − t)Z Cor ∥
   555	
   556	(8)
   557	
   558	where t affects the position of the intermediate feature on the
   559	Brownian bridge path. The established path features serve as
   560	the exemplar to supervise intermediate feature Z B B generation
   561	as shown in Fig. 2.
   562	Particularly, we borrow idea from [8] and [10] to design
   563	the intermediate feature sampling structure, i.e., generating in
   564	the informative latent space. Based on the principle, we proposed a Feature Variational Generation (FVG) module that
   565	integrates re-parameter sampling into the generation process.
   566	Fig. 4 illustrates our proposed FVG that samples intermediate
   567	features from latent distribution of Z Cor , i.e.,
   568	µ, σ = E F V G (Z Cor )
   569	Z B F G = D F V G (g(µ, σ, ε))
   570	
   571	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.
   572	
   573	(9)
   574	(10)
   575	
   576	ZHANG et al.: CORRUPTION-INVARIANT PERSON RE-IDENTIFICATION VIA COARSE-TO-FINE FEATURE ALIGNMENT
   577	
   578	1089
   579	
   580	Fig. 3. Comparison of two different alignment methods. We categorize modal features into common features (i.e., corruption-invariant features) and modal
   581	unique features (i.e., clean-image features susceptible to corruption and corrupted-image features introduced by corruption). Using only DFA for coarse
   582	alignment, the contrastive loss will encourage the features in the corrupted modality to match those in the clean modality and lead to the loss of modal unique
   583	features. DFA+BFG expands the bound of the corrupted modality and achieves seamless modal alignment by filling intermediate features. This smoother
   584	alignment method preserves more discriminative parts of modal unique features for identifying pedestrian identities.
   585	
   586	Fig. 4. The Feature Variational Generation module parses the distribution
   587	parameters of the input through the encoder, then adds randomly sampled
   588	values, and utilizes the re-parameter trick to pass them to the decoder to
   589	obtain generated features.
   590	
   591	where E F V G and D F V G denote the encoder and decoder of
   592	the FVG, g(·, ·, ·) is the sampling function that is realized
   593	by the re-parameter trick, i.e., g(µ, σ, ε) = µ + σ × ε, µ
   594	and σ is the distribution parameters learned from Z Cor , ε is
   595	random noise sampling from a standard Gaussian distribution,
   596	i.e., ε ∼ N (0, 1). During the training process, the FVG
   597	structure takes the randomly corrupted features Z Cor and the
   598	Brownian bridge path feature Z B B as inputs and generates the
   599	intermediate feature Z B F G , as well as the mean and variance
   600	µ B F G , σ B F G , µ B B , σ B B of Z B F G and Z B B , as shown in
   601	Eq. 9 and 10. To bring the distributions of the two more
   602	similar, we minimize KL divergence between Z B F G and
   603	Z B B , defined as
   604	L K L ( pB F G , pB B )
   605	= K L( p B F G ∥ p B B )
   606	σB B
   607	1
   608	1
   609	)+
   610	(σ B2 F G + (µ B F G − µ B B )2 ) −
   611	= log(
   612	2
   613	σB F G
   614	2
   615	2σ B B
   616	(11)
   617	
   618	where p B F G = N (µ B F G , σ B2 F G ) represents the Gaussian
   619	distribution form of the intermediate feature Z B F G , and
   620	p B B = N (µ B B , σ B2 B ) represents the Gaussian distribution
   621	form of the Brownian bridge path feature Z B B .
   622	To constrain the generated intermediate feature Z B F G to
   623	the Brownian bridge path region, we enforce a Mean Square
   624	Error (MSE) loss between Z B F G and its corresponding exemplar Z B B . Ultimately, the total loss for intermediate feature
   625	generation via Brownian bridge is defined as
   626	L Br = L M S E (Z B F G , Z B B ) + L K L ( p B F G , p B B )
   627	N
   628	
   629	=
   630	
   631	1 X
   632	j
   633	j
   634	∥Z B F G − Z B B ∥2 + K L( p B F G ∥ p B B )
   635	N
   636	j=1
   637	
   638	(12)
   639	By optimizing Eq. 12, the BFG module generates intermediate
   640	pseudo samples to bridge clean exemplar and its corresponding
   641	corrupted variants. Such manner achieves alignment by constructing semantic structure rather than enforcing alignment,
   642	and yields superior performance.
   643	D. Overall Loss Function
   644	We adopt a coarse-to-fine alignment strategy for model
   645	optimization. In the coarse alignment stage, we enforce different modalities of corruptions align to their corresponding
   646	clean exemplar with a weighted contrastive loss, i.e., Eq. 4.
   647	In the fine alignment stage, a Brownian feature generation
   648	module together with intermediate feature generation loss, i.e.,
   649	
   650	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.
   651	
   652	1090
   653	
   654	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025
   655	
   656	Eq. 12, is utilized to construct seamless semantic structure
   657	for modal range expansion, addressing the issue of potential
   658	information loss in the coarse alignment stage. To improve
   659	identity discriminative ability, an identity loss and a triplet loss
   660	are also used. It is worth noting that the identity loss and triplet
   661	loss in the coarse alignment stage are calculated solely from
   662	corrupted samples, while those in the fine alignment stage,
   663	they incorporate the results of intermediate pseudo samples in
   664	the calculation. Thus, the overall optimization loss is defined
   665	as
   666	L T otal = L T r + L I D + λ ∗ L W C L + (1 − λ) ∗ L Br
   667	
   668	TABLE I
   669	C ORRUPTION U SED IN DATA R ECONSTRUCTION
   670	
   671	TABLE II
   672	F OUR D IFFERENT E VALUATION P ROTOCOLS . “▲” D ENOTES
   673	C ORRUPTED S ET, “∇” D ENOTES C LEAN S ET
   674	
   675	(13)
   676	
   677	where λ ∈ {0, 1} denotes the optimization indicator, and 1 for
   678	coarse stage, 0 otherwise. We summarize the proposed method
   679	in Algorithm 1.
   680	Algorithm 1 The Proposed CFAT Method
   681	Input: dataset D = {X i }; corruption simulator parameters c,
   682	l; patch size P; epochs
   683	Output: CFAT model
   684	1: Generating corrupted dataset DCor according to Eq. 1;
   685	2: Model initialization;
   686	3: Pretrain DFA model and freeze TFE parameters;
   687	4: while not convergence or (epoch < epochs+1): do
   688	5:
   689	Extract features Z Cl from TFE and Z Cor from SFE,
   690	respectively;
   691	// Stage 1. Coarse alignment.
   692	6:
   693	Optimize SFE using Eq. 13;
   694	// Stage 2. Fine alignment.
   695	7:
   696	Get Z B B and Z B F G from BFG;
   697	8:
   698	Get µ B B , µ B F G , σ B B , σ B F G according to Eq. 9;
   699	9:
   700	Optimize SFE according to Eq. 13.
   701	10: end while
   702	
   703	IV. E XPERIMENTS
   704	In this section, we conducted extensive experiments on
   705	three benchmark datasets to evaluate the effectiveness of the
   706	proposed method.
   707	A. Datasets
   708	1) Market-1501 [11]: The Market-1501 dataset contains
   709	32,668 images of 1,501 pedestrian identities collected
   710	from six cameras on the campus of Tsinghua University.
   711	The commonly used partitioning of this dataset includes
   712	751 pedestrian data for the training set and the remaining
   713	750 pedestrian data for the testing set, with 3,368 images
   714	in the query set and 19,732 images in the gallery set.
   715	2) CUHK-03 [12]: CUHK-03 was collected using five
   716	cameras on the Chinese University of Hong Kong campus and consists of 1,467 pedestrian identities. For fair
   717	comparison, we adopted the most commonly used partitioning method in CUHK-03 [50], dividing the dataset
   718	into 767 pedestrian data for training and 700 pedestrian
   719	data for testing.
   720	3) MSMT-17 [13]: MSMT-17 is a large-scale dataset.
   721	It was collected from 12 outdoor and 3 indoor cameras,
   722	
   723	providing a realistic environment. The dataset includes
   724	4,101 pedestrian identities. There are 1,041 identities in
   725	the training set and 3,060 in the testing set.
   726	In our experiments, we augmented the corresponding corrupted datasets from the above three benchmarks by corruption
   727	scene simulation. Considering the generation mechanisms of
   728	corruptions, these corruptions are roughly categorized into
   729	four classes, i.e., noise, blur, weather and digital. In total,
   730	we generated 20 different types of corruption scenes as listed
   731	in Table I. Notably, our work also randomly set five different
   732	intensity levels of corruption severity for complicated corruption scenarios.
   733	B. Evaluation Protocols and Metrics
   734	1) Evaluation Protocols: Following [1], we set up four
   735	different evaluation configurations, i.e., Clean Eval, Corrupt

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf' - | nl -ba | sed -n '39,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    39	1. Introduction
    40	Person re-identification (ReID) [1–3] aims to match a target pedestrian captured from non-overlapping cameras at different time or different locations, which is of great significance for automatic surveillance,
    41	pedestrian behavior analysis and smart city [4–7]. The single-domain
    42	person ReID technique is where both the training and test samples
    43	are from the same domain. Though the method achieves impressive
    44	performance, it significantly degrades when directly applied to an
    45	unseen domain due to the inevitable domain gap [8–10]. To reduce the
    46	domain gap, cross-domain person ReID is proposed to simultaneously
    47	exploit source domain samples with identity labels and target domain
    48	samples without label annotations, so that the deep model is well
    49	generalized to the target domain.
    50	The clustering-based method is one kind of representative methods
    51	for cross-domain person ReID [10–15], and it mainly contains three
    52	steps: (1) using labeled source domain samples to obtain pre-training
    53	models; (2) assigning pseudo labels to unlabeled target domain samples using clustering algorithms, such as 𝐾-means and DBSCAN; (3)
    54	fine-tuning models/ learning features with the pseudo labels. Steps
    55	(2) and (3) are iteratively performed to ameliorate each other until
    56	
    57	convergence. The key stage for cross-domain person ReID is the pseudo
    58	label assignment. The existing methods [10,12,13] only assign one kind
    59	of pseudo labels using clustering algorithms based on the similarities
    60	between samples. However, it is difficult to describe each unlabeled
    61	target domain sample accurately by assigning only one kind of pseudo
    62	labels for iterative optimization because these samples are captured
    63	from unconstrained environments, where the pedestrian appearances
    64	are easily influenced by many environmental factors such as occlusions,
    65	illuminations, viewpoints, and poses, etc., leading to large intra-class
    66	and small inter-class variances [16,17]. Hence, learning one kind of
    67	label results in inaccurate pseudo labels for the target domain samples
    68	as shown in Fig. 1(a).
    69	In this paper, we propose a novel label learning method named
    70	Heterogeneous Pseudo Labels (HPL) for cross-domain person ReID,
    71	which could overcome large intra-class and small inter-class variances
    72	between pedestrian images in the target domain. As shown in Fig.
    73	1(b), HPL simultaneously learns three different kinds of pseudo labels
    74	for iterative optimization to obtain a completed description of each
    75	unlabeled target domain sample, and the three kinds of pseudo labels
    76	include fine-grained labels, coarse-grained labels, and instance labels.
    77	
    78	∗ Corresponding author.
    79	
    80	E-mail address: shuangliu.tjnu@gmail.com (S. Liu).
    81	https://doi.org/10.1016/j.patcog.2025.111702
    82	Received 16 September 2024; Received in revised form 5 April 2025; Accepted 8 April 2025
    83	Available online 19 April 2025
    84	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    85	
    86	Pattern Recognition 166 (2025) 111702
    87	
    88	Z. Zhang et al.
    89	
    90	Fig. 1. (a) The most existing methods only learn one kind of pseudo label for iterative optimization, making it difficult to describe each target domain sample accurately. (b) The
    91	proposed method learns heterogeneous pseudo labels, such as fine-grained label, coarse-grained label, and instance label, for iterative optimization to describe each target domain
    92	sample more comprehensively from different aspects.
    93	
    94	• We propose Heterogeneous Pseudo Labels which could overcome large intra-class and small inter-class variances in the crossdomain person ReID, where fine-grained labels, coarse-grained
    95	labels, and instance labels are designed to describe unlabeled
    96	target domain samples from three different perspectives. Furthermore, to improve the quality of pseudo labels, we propose the
    97	Pseudo Labels Constraint to maintain the consistency between the
    98	heterogeneous labels.
    99	• We propose the Confidence Contrastive Loss to re-weight the similarity of sample pairs by considering the confidence coefficient so
   100	as to reduce the influence of the samples with noisy labels in the
   101	process of contrastive learning.
   102	• Extensive experiments on four cross-domain tasks demonstrate
   103	the superiority of the proposed method, and it achieves a new
   104	state-of-the-art performance.
   105	
   106	The fine-grained labels are derived from a sufficient number of cluster
   107	centers generated by the clustering algorithm, which could capture the
   108	subtle distinctions among different identities. Meanwhile, the coarsegrained labels are formed by fewer and broader clusters, which could
   109	reduce the influence of noise by avoiding fine divisions of the feature
   110	space. Moreover, the instance labels are learned by the relationships
   111	between the samples, providing insights into the individual sample
   112	characteristics and their similarities or differences with other samples.
   113	In a word, the three kinds of pseudo labels describe the same target
   114	domain sample from different perspectives, and using all three kinds
   115	of pseudo labels makes the best of their own advantages in order to
   116	describe target domain samples more comprehensively.
   117	To improve the quality of generated pseudo labels, we propose the
   118	Pseudo Labels Constraint (PLC) to regulate the process of label learning.
   119	To this end, we enforce the instance labels to be consistent with the
   120	fine-grained labels and the coarse-grained labels, because the three
   121	kinds of pseudo labels describe the same target domain sample. But
   122	these pseudo labels are heterogeneous, that is, they distribute in different feature spaces. Correspondingly, we design the label transformation
   123	strategy so that the three kinds of pseudo labels could be constrained
   124	after transformation.
   125	Recently, contrastive learning combined with the memory bank
   126	is adopted to learn discriminative features for cross-domain person
   127	ReID [13,14,18]. Specifically, for each sample (anchor) of the target
   128	domain in contrastive learning, the positive and negative samples are
   129	selected from the memory bank according to pseudo labels. However,
   130	most existing contrastive learning neglects the influence of noisy labels
   131	which are inevitable in the process of pseudo labels generation. To
   132	overcome this limitation, we propose the Confidence Contrastive Loss
   133	(CCL) to consider the sample confidence in the learning process. To this
   134	end, we first compute the cluster center of target domain samples with
   135	the same identity in the memory bank. Then, we utilize the similarity
   136	between the cluster centers of fine-grained labels and memory bank as
   137	the confidence coefficient. After obtaining the confidence coefficient,
   138	it is used to re-weight the similarity of sample pairs in order to rectify
   139	the optimization process of contrastive learning.
   140	The main contributions of the proposed method are summarized as
   141	follows:
   142	
   143	The rest of the paper is organized as follows. Section 2 gives a revisit
   144	of related work, Section 3 details the proposed method including Heterogeneous Pseudo Labels, Pseudo Labels Constraint, and Confidence
   145	Contrastive Loss, Section 4 shows extensive experiments to verify the
   146	advantages of the proposed method, and finally, we make a conclusion
   147	and discussion in Section 5.
   148	2. Related work
   149	Cross-domain image learning is crucial in practical applications,
   150	because it equips models with strong generalization capabilities when
   151	training and test image distributions do not overlap, mitigating performance degradation due to distributional shifts. Recently, cross-domain
   152	image learning has gained popularity across various tasks, including
   153	image classification [19,20], palmprint recognition [21–23], person reidentification [15,17,24,25], and so on. In this paper, we focus on
   154	cross-domain person re-identification, which is mainly categorized into
   155	three, i.e., clustering-based methods, domain transfer based methods
   156	and memory bank based methods.
   157	2.1. Clustering-based methods
   158	Generating pseudo labels is a crucial stage in cross-domain person
   159	ReID, and the quality of the generated pseudo labels determines its
   160	2
   161	
   162	Pattern Recognition 166 (2025) 111702
   163	
   164	Z. Zhang et al.
   165	
   166	performance. The pseudo labels are generally classified into two kinds,
   167	i.e., hard labels and soft labels. Some methods utilize clustering algorithms to obtain the hard (one-hot) labels [14,26–28]. For example,
   168	both Fan et al. [26] and Zheng et al. [14] apply the clustering algorithms once or several times to generate the hard labels, so that the
   169	model can be trained on the target domain in a supervised way. In
   170	order to adapt better to the feature distribution of the target domain,
   171	Fu et al. [28] propose to assign multi-scale hard labels by exploiting
   172	the global and local features.
   173	In order to reduce the overfitting of the deep model, some methods [24,29,30] design different strategies to generate the soft labels.
   174	Zhong et al. [24,29] propose to learn soft labels by exploiting the stored
   175	averaged features in an exemplar memory module so as to alleviate the
   176	intra-domain variations in the target domain. Yu et al. [30] learn the
   177	soft multilabel by introducing the labeled auxiliary domain as reference
   178	persons in the unlabeled target domain. Furthermore, the soft labels
   179	generated from the deep network are beneficial in combination with
   180	pseudo labels or sample denoising [10,13,31,32]. Ge et al. [10] propose
   181	Mutual Mean-Teaching (MMT) to refine the soft labels generated by
   182	peer networks using mutual learning. Zheng et al. [13] propose an
   183	Uncertainty-guided Noise Resilient Network (UNRN) that estimates the
   184	reliability of the pseudo labels and softly exploits them to alleviate the
   185	negative effects of noise-prone samples.
   186	However, these methods only assign one kind of pseudo labels to unlabeled target domain samples, which cannot describe these unlabeled
   187	samples accurately. Different from the above-mentioned methods, the
   188	proposed method is to gradually approximate the real label distribution
   189	of the target domain through learning Heterogeneous Pseudo Labels,
   190	where we utilize different clustering algorithms to generate three different kinds of pseudo labels. Furthermore, we propose the Pseudo Labels
   191	Constraint to regulate the process of label learning in order to improve
   192	the labels quality. Hence, the proposed method could describe target
   193	domain samples more comprehensively and accurately.
   194	
   195	Ge et al. [18] and Zheng et al. [13] combine contrastive learning
   196	with the memory bank to jointly optimize positive and negative pairs
   197	for a query/anchor sample, thereby enhancing discrimination of the
   198	representations. Sun et al. [43] and Luo et al. [25] focus on the learning
   199	process of the deep model based on the memory bank, so as to improve
   200	the performance of cross-domain person ReID.
   201	Furthermore, the noisy samples in contrastive learning results in
   202	the performance degradation of the person ReID model, and some
   203	methods [13,44,45] alleviate this problem by weighting the samples
   204	for person ReID. For example, Zheng et al. [13] and Han et al. [45]
   205	present to assign different weights to target domain samples based on
   206	the uncertainty, thereby improving the discriminative ability of the
   207	cross-domain person ReID model.
   208	Different from these existing methods, we propose the Confidence
   209	Contrastive Loss combined with the memory bank to obtain accurate
   210	similarity of sample pairs by considering the sample confidence, where
   211	the well-designed confidence coefficient is beneficial to adjusting the
   212	similarity of sample pairs adaptively, so as to alleviate the influence of
   213	noisy labels.
   214	3. Approach
   215	In this section, we first present an overview of the proposed method
   216	and then describe the major parts, i.e., Heterogeneous Pseudo Labels,
   217	Pseudo Labels Constraint, and Confidence Contrastive Loss in detail.
   218	3.1. Overview
   219	The cross-domain person ReID model is trained by the labeled
   220	source domain and the unlabeled target domain. Formally, the samples
   221	and the ground-truth labels in the source domain are denoted as D𝑠 =
   222	𝑛𝑠
   223	𝑛𝑠
   224	{𝑥𝑠𝑖 |𝑖=1
   225	} and {𝑦𝑠𝑖 |𝑖=1
   226	} ∈ [1, … , 𝐾𝑠 ], where 𝑛𝑠 and 𝐾𝑠 denote the number
   227	of samples and identities in the source domain, respectively. Similarly,
   228	𝑛𝑡
   229	D𝑡 = {𝑥𝑡𝑖 |𝑖=1
   230	} represents 𝑛𝑡 samples in the target domain and there is no
   231	ground-truth labels in the target domain.
   232	As illustrated in Fig. 2, we adopt ResNet50 [46] as the backbone
   233	to build a mean teacher network and a student network [47]. We
   234	follow the general pipeline of clustering-based cross-domain person
   235	ReID methods [12–14,26], in which they include three steps (model
   236	pre-training, pseudo label assignment, and model fine-tuning), and the
   237	last two steps are performed iteratively until convergence.
   238	In the stage of the model pre-training, we exploit the labeled source
   239	domain samples to train a ReID model and it is utilized to initialize
   240	the mean teacher network 𝐹 (⋅) and the student network 𝐺(⋅). In the
   241	process of clustering, we extract the features of target domain samples
   242	𝑓𝑖𝑡 = 𝐹 (𝑥𝑡𝑖 ) (𝑖 = 1, … , 𝑛𝑡 ) from the mean teacher network. Then the finegrained labels and the coarse-grained labels are generated by clustering
   243	these features. Meanwhile, the instance labels are learned using all the
   244	target domain features and the source domain centers in the memory
   245	bank. During the model fine-tuning, the three kinds of pseudo labels
   246	are simultaneously utilized to train the student network. In particular,
   247	the student network is updated in a real-time manner, and the mean
   248	teacher network is updated by exponential moving average (EMA) of
   249	parameters of the student network.
   250	
   251	2.2. Domain transfer based methods
   252	One of the challenges of the cross-domain person ReID task is to
   253	align the data distributions between the source domain and the target
   254	domain, and some methods [33–38] utilize the Generative Adversarial
   255	Networks (GANs) with the source samples to generate specific images
   256	or features of the target domain to align the data distributions of
   257	different domains. For example, Deng et al. [33] and Wei et al. [34]
   258	transfer the labeled source domain images to the style of the target
   259	domain while preserving the identity labels so as to learn the information from the target domain. Zhong et al. [35] and Zhou et al. [38]
   260	transfer the images in the target domain to multiple camera styles for
   261	domain adaptation. Since the images or features generated are always
   262	different from those of the real scene, the GANs-based approaches are
   263	still imperfect.
   264	2.3. Memory bank based methods
   265	The memory bank is utilized to enhance the ability of feature representation and has been widely applied in the fields of unsupervised
   266	and self-supervised learning, especially in contrastive learning [39–41].
   267	The core idea of the memory bank is to store a large number of feature
   268	representations and provide abundant negative samples for contrastive
   269	learning, where it could keep up-to-date representations using the
   270	specific update mechanism. In the field of cross-domain person ReID,
   271	Zhong et al. [29] apply a memory bank for the first time, and propose
   272	to treat each target sample as an individual identity and store them in
   273	an exemplar memory module to adapt to the intra-domain variations
   274	in the target domain.
   275	Recently, a variety of strategies [13,18,25,42,43] based on contrastive learning and memory bank have been proposed to improve
   276	the performance of cross-domain person ReID models. For example,
   277	
   278	3.2. Heterogeneous pseudo labels
   279	Most existing methods have achieved impressive performance by
   280	assigning one kind of pseudo labels to the unlabeled target domain
   281	samples for cross-domain person ReID. However, only assigning one
   282	kind of pseudo labels is difficult to describe unlabeled target domain
   283	samples accurately because they are easily affected by complex environments. To overcome this limitation, we propose to apply three
   284	kinds of pseudo labels, i.e., fine-grained labels, coarse-grained labels
   285	and instance labels, for cross-domain person ReID.
   286	3
   287	
   288	Pattern Recognition 166 (2025) 111702
   289	
   290	Z. Zhang et al.
   291	
   292	Fig. 2. The overall framework of the proposed method for cross-domain person ReID. The orange and green slots in the memory bank represent source domain centers and target
   293	domain features, respectively.
   294	
   295	Fine-grained Labels. We first extract the features 𝑓𝑖𝑡 ∈ R𝐷 (𝑖 =
   296	1, … , 𝑛𝑡 ) from the mean teacher network, and then we obtain the
   297	cluster centers 𝐶 1 ∈ R𝐷×𝐾1 using DBSCAN [48], where 𝐾1 is the
   298	number of cluster centers. We design the fine-grained labels 𝑌 1 =
   299	[𝑦11 , … , 𝑦1𝑘 , … , 𝑦1𝐾 ] ∈ R𝐾1 as the hard labels (one-hot vectors), where
   300	1
   301	𝐾1 is the number of identities when using the fine-grained labels and
   302	it is equal to the cluster center number of 𝐶 1 . Here, there is only one
   303	non-zero entry in 𝑌 1 , and this non-zero entry is equal to 1. We find the
   304	cluster center in 𝐶 1 which is the nearest to 𝑓𝑖𝑡 , and the index of this
   305	nearest cluster center is the 𝑘th class if the non-zero entry in 𝑌 1 is 𝑦1𝑘 .
   306	After obtaining the fine-grained labels, we utilize the target ID loss to
   307	optimize the student network:
   308	
   309	process, we first generate the fine-grained labels using DBSCAN. Then,
   310	we obtain the coarse-grained labels by clustering the remaining samples
   311	after discarding the outliers by DBSCAN in order to describe the target
   312	domain samples from fine and coarse perspectives.
   313	Instance Labels. We construct the memory bank 𝑀 ∈ R𝐷×𝐾3 and
   314	𝐾3 =  + 𝐾𝑠 is the number of slots in the memory bank, where 
   315	and 𝐾𝑠 are the number of the target domain features in the memory
   316	bank and the source domain centers in the memory bank, respectively.
   317	Specifically, the target domain features are extracted from the student
   318	network, and they are updated in the way of the queue [14,40]. The
   319	source domain centers are the mean vectors of the source domain features extracted from the mean teacher network, and they are updated
   320	in the form of key–value pairs [29]. Here, we learn the instance labels
   321	𝑌 3 = [𝑦31 , … , 𝑦3𝑘 , … , 𝑦3𝐾 ] ∈ R𝐾3 by calculating the similarity between 𝑔𝑖𝑡
   322	3
   323	and 𝑀. The 𝑘th entry of the instance label is formulated as:
   324	
   325	𝑁𝑡
   326	
   327	𝐿1𝑖𝑑 =
   328	
   329	1 ∑
   330	𝐿 (𝑃 (𝑔𝑖𝑡 ), 𝑌 1 ),
   331	𝑁𝑡 𝑖=1 𝑐𝑒
   332	
   333	(1)
   334	
   335	where 𝑁𝑡 is the mini-batch size of the target domain samples, 𝐿𝑐𝑒 is
   336	the cross-entropy loss, and 𝑃 (𝑔𝑖𝑡 ) is the predicted probability that the
   337	features 𝑔𝑖𝑡 = 𝐺(𝑥𝑡𝑖 ) ∈ R𝐷 extracted from the student network belongs
   338	to the fine-grained label 𝑌 1 .
   339	Coarse-grained Labels. We generate the coarse-grained labels using the features 𝑓𝑖𝑡 ∈ R𝐷 (𝑖 = 1, … , 𝑛𝑡 ) based on the 𝐾-means algorithm,
   340	and obtain the cluster centers 𝐶 2 ∈ R𝐷×𝐾2 , where 𝐾2 (𝐾2 < 𝐾1 ) is the
   341	number of identities when using the coarse-grained labels. We design
   342	the coarse-grained labels 𝑌 2 = [𝑦21 , … , 𝑦2𝑘 , … , 𝑦2𝐾 ] ∈ R𝐾2 as the soft
   343	2
   344	labels by computing the similarity between the cluster centers 𝐶 2 and
   345	the features 𝑔𝑖𝑡 = 𝐺(𝑥𝑡𝑖 ) ∈ R𝐷 (𝑖 = 1, … , 𝑛𝑡 ). The 𝑘th entry of the
   346	coarse-grained label is:
   347	exp((𝐶𝑘2 )𝑇 𝑔𝑖𝑡 )
   348	𝑦2𝑘 = ∑𝐾
   349	,
   350	2
   351	exp((𝐶𝑗2 )𝑇 𝑔𝑖𝑡 )
   352	𝑗=1
   353	
   354	exp(𝑀𝑘𝑇 𝑔𝑖𝑡 )
   355	𝑦3𝑘 = ∑𝐾
   356	,
   357	3
   358	exp(𝑀𝑗𝑇 𝑔𝑖𝑡 )
   359	𝑗=1
   360	
   361	(3)
   362	
   363	where 𝑀𝑘 denotes the 𝑘th slot of the memory bank. In particular, we
   364	design the instance label 𝑌 3 as the hard labels (one-hot vectors). We
   365	employ the instance labels to optimize the student network, and similar
   366	to Eq. (1), the target ID loss is denoted as 𝐿3𝑖𝑑 . The instance labels
   367	generated based on the memory bank could reflect the features over
   368	past epochs.
   369	The fine-grained labels could subtly describe the identities of target
   370	domain samples due to the sufficient number of clustering centers,
   371	offering a nuanced view of each identity. This level of detail ensures
   372	that differences among identities are captured, thereby providing a
   373	comprehensive understanding of the target domain. In contrast, the
   374	coarse-grained labels contain fewer noisy labels because of the rough
   375	division of the feature space. By simplifying the feature space, coarsegrained labels reduce the potential for misclassification and noise,
   376	making them more robust against outliers and noises. Meanwhile,
   377	learning the instance labels helps to build relationships in the sample
   378	level, providing insights into the individual sample characteristics and
   379	their similarities or differences with other samples. Hence, the three
   380	different kinds of pseudo labels have their advantages.
   381	Furthermore, under different evaluation criteria, different pseudo
   382	labels have different performances. For example, if the evaluation
   383	criterion is the pedestrian features with different identities clustered
   384	in different clusters, the order of clustering performance (𝑃 ) for the
   385	three kinds of pseudo labels is: 𝑃 3 > 𝑃 1 > 𝑃 2 , and the superscripts
   386	
   387	(2)
   388	
   389	where 𝐶𝑘2 is the 𝑘th column of 𝐶 2 . Then, we apply the coarse-grained
   390	labels to optimize the student network, and similar to Eq. (1), the target
   391	ID loss for the coarse-grained labels is denoted as 𝐿2𝑖𝑑 .
   392	The fine-grained labels and the coarse-grained labels are defined
   393	based on the number of the clusters obtained by clustering. If the
   394	number of clusters is close to the ground-truth cluster number of the
   395	target domain, these clusters are treated as the fine-grained labels. If
   396	the number of clusters is less than the ground-truth cluster number
   397	of the target domain, the clusters are defined as the coarse-grained
   398	labels. DBSCAN defines the clusters by the density, and for a sample,
   399	if the density around it is not sufficient to classify it as a member of
   400	the cluster, that sample is considered an outlier. During the training
   401	4
   402	
   403	Pattern Recognition 166 (2025) 111702
   404	
   405	Z. Zhang et al.
   406	
   407	1, 2 and 3 represent the fine-grained labels, the coarse-grained labels
   408	and the instance labels, respectively. If the evaluation criterion is the
   409	pedestrian features with the same identity clustered in the same cluster,
   410	the order of clustering performance for the three kinds of pseudo labels
   411	is 𝑃 2 > 𝑃 1 > 𝑃 3 . In a word, reasonable use of three different kinds
   412	of pseudo labels in HPL, i.e., fine-grained labels, coarse-grained labels,
   413	and instance labels, makes the best of their respective advantages in
   414	order to describe the target domain samples more comprehensively,
   415	which could improve the ReID performance.
   416	Finally, considering the three kinds of pseudo labels simultaneously,
   417	the overall ID loss of HPL for the target domain samples is defined as:
   418	𝑖𝑑 = 𝐿1𝑖𝑑 + 𝜆1 𝐿2𝑖𝑑 + 𝜆2 𝐿3𝑖𝑑 ,
   419	
   420	3.4. Confidence contrastive loss
   421	Recently, the contrastive learning [13,14,18] is widely applied to
   422	cross-domain person ReID because it could learn discriminative features. The Circle Loss (CL) [49] is a representative contrastive learning
   423	algorithm which is proven to perform well on the person ReID task.
   424	The CL selects the positive samples and the negative samples from the
   425	memory bank according to the pseudo labels for cross-domain person
   426	ReID. However, the noisy labels are inevitable in the process of pseudo
   427	labels generation. In order to address this issue, we propose CCL to
   428	consider the sample confidence in the process of contrastive learning,
   429	and it is defined as:
   430	
   431	(4)
   432	
   433	𝑁−
   434	
   435	𝐶𝐶𝐿 =
   436	
   437	where 𝜆1 and 𝜆2 serve as the trade-off parameters.
   438	
   439	𝑁𝑡
   440	𝑖
   441	∑
   442	𝑠−
   443	1 ∑
   444	− 𝑖𝑘
   445	log[1 +
   446	exp(𝛾𝛼𝑖𝑘
   447	(
   448	− 𝑚))
   449	𝑁𝑡 𝑖=1
   450	𝑤𝑖
   451	𝑘=1
   452	𝑁+
   453	
   454	𝑖
   455	∑
   456	
   457	3.3. Pseudo labels constraint
   458	
   459	𝑗=1
   460	
   461	The three kinds of pseudo labels describe the same target domain
   462	sample from different views, and therefore there exists consistency
   463	among them. Hence, we propose PLC to regulate the process of label
   464	learning using the label consistency. We enforce the instance labels to
   465	be consistent with the fine-grained labels and the coarse-grained labels,
   466	respectively. However, the three kinds of pseudo labels are heterogeneous and distribute in different feature spaces. Correspondingly, we
   467	design the label transformation strategy so that these pseudo labels
   468	could be constrained after transformation. To enforce the instance
   469	labels to be consistent with the fine-grained labels, we change the
   470	instance label 𝑌 3 to 𝑌̃ 3 = [𝑦̃31 , … , 𝑦̃3𝑘 , … , 𝑦̃3𝐾 ] using the label transforma1
   471	tion strategy which considers the similarity between the cluster centers
   472	of the fine-grained label 𝐶 1 ∈ R𝐷×𝐾1 and the slot of the memory bank
   473	𝑀 corresponding to the index of the non-zero entry in 𝑌 3 for each
   474	target domain sample. The 𝑘th entry of 𝑌̃ 3 is formulated as:
   475	𝑦̃3𝑘 = 𝑀𝑎𝑇 𝐶𝑘1 ,
   476	
   477	where 𝑠−
   478	is the cosine similarity between the 𝑖th target domain sample
   479	𝑖𝑘
   480	and the 𝑘th negative sample, 𝑠+
   481	𝑖𝑗 is the cosine similarity between the
   482	𝑖th target domain sample and the 𝑗th positive sample, 𝑁𝑖− and 𝑁𝑖+
   483	denote the number of negative and positive sample pairs for the 𝑖th
   484	target domain sample respectively, and 𝛾 is the scale factor. Here,
   485	− = [𝑚 + 𝑠− ∕𝑤 ] and 𝛼 + = [1 + 𝑚 − 𝑤 𝑠+ ] are non-negative weighting
   486	𝛼𝑖𝑘
   487	𝑖 +
   488	𝑖 𝑖𝑗 +
   489	𝑖𝑗
   490	𝑖𝑘
   491	factors of 𝑠−
   492	and 𝑠+
   493	𝑖𝑗 respectively, where 𝑚 is the margin and [⋅]+ is the
   494	𝑖𝑘
   495	‘‘cut-off at zero’’ operation.
   496	More importantly, 𝑤𝑖 represents the confidence coefficient for the
   497	𝑖th sample, and it could reflect the confidence of the sample. Since
   498	averaging the samples within the same cluster could mitigate the
   499	influence of noisy labels, we compute the confidence coefficient 𝑤𝑖 by
   500	considering the interaction between the cluster centers 𝐶 𝑚 ∈ R𝐷×𝐾1 of
   501	the target domain samples in the memory bank and the cluster centers
   502	𝐶 1 ∈ R𝐷×𝐾1 of the fine-grained labels. Hence, it is defined as:
   503	
   504	(5)
   505	
   506	𝐾1
   507	
   508	𝑤𝑖 =
   509	
   510	where 𝑀𝑎 denotes the 𝑎th slot of the memory bank, 𝑎 indicates the
   511	index of the largest value in 𝑌 3 , and 𝐶𝑘1 is the 𝑘th cluster center of
   512	the fine-grained labels. Similarly, we transfer the instance labels to
   513	𝑌̂ 3 = [𝑦̂31 , … , 𝑦̂3𝑘 , … , 𝑦̂3𝐾 ] in order to compare with the coarse-grained
   514	2
   515	labels using the label transformation strategy, and the 𝑘th entry of 𝑌̂ 3
   516	(6)
   517	
   518	where 𝐶𝑘2 is the 𝑘th cluster center of the coarse-grained labels. Hence,
   519	PLC is defined as:
   520	𝑟𝑒𝑔 =
   521	
   522	1 ̃3
   523	[‖𝑌 − 𝑌 1 ‖22 + ‖𝑌̂ 3 − 𝑌 2 ‖22 ],
   524	2
   525	
   526	1 ∑
   527	1
   528	,
   529	𝐾1 𝑘=1 1 + 𝑒−(𝐶𝑘𝑚 )𝑇 𝐶 1 (𝑖)
   530	
   531	(9)
   532	
   533	where 𝐶𝑘𝑚 is the 𝑘th column of 𝐶 𝑚 , and 𝐶 1 (𝑖) is the cluster center of the
   534	fine-grained label corresponding to 𝑔𝑖𝑡 .
   535	In Eq. (9), we utilize the interaction between the cluster centers to
   536	derive the confidence coefficient. This confidence coefficient is then
   537	applied to Eq. (8), enabling the adaptive adjustment of the similarity of
   538	sample pairs, thereby effectively mitigating the interference of noise.
   539	In other words, CCL amplifies the influence of sample pairs with
   540	higher confidence, while diminishes the influence of sample pairs with
   541	lower confidence. This adjustment helps to rectify the optimization
   542	process of contrastive learning, making it more robust to noisy labels.
   543	The re-weighting mechanism in CCL ensures that the learning process prioritizes more reliable samples, thereby improving the overall
   544	performance of the model in cross-domain person ReID task.
   545	
   546	is formulated as:
   547	𝑦̂3𝑘 = 𝑀𝑎𝑇 𝐶𝑘2 ,
   548	
   549	(8)
   550	
   551	exp(−𝛾𝛼𝑖𝑗+ (𝑤𝑖 𝑠+
   552	𝑖𝑗 − 1 + 𝑚))],
   553	
   554	(7)
   555	
   556	where ‖ ⋅ ‖2 denotes the 𝐿2 norm, and 𝑌̃ 3 and 𝑌̂ 3 are 𝐿2 -normalized.
   557	The need for the Pseudo Labels Constraint (PLC) arises from the
   558	challenges inherent in generating pseudo labels of cross-domain person
   559	ReID task. That is, different kinds of pseudo labels inherently share
   560	consistency and dependency relationships despite of their different
   561	perspectives for each target domain sample. However, due to the
   562	influence of noise, these pseudo labels may lead to the conflicting
   563	description of the same target domain sample. Hence, PLC is introduced
   564	to address this limitation by enforcing consistency among different
   565	kinds of pseudo labels, namely, the fine-grained labels, the coarsegrained labels, and the instance labels, so as to improve the quality
   566	of pseudo labels.
   567	Specifically, PLC ensures that the fine-grained labels, the coarsegrained labels, and the instance labels are consistent in their description
   568	for the same target domain sample. The label transformation strategy
   569	is designed so that different kinds of pseudo labels can be constrained
   570	to align after transformation. This alignment reduces the conflicting
   571	information, so as to generate more accurate and reliable pseudo labels.
   572	
   573	3.5. Optimization
   574	Similar to [10,13], we also employ the triplet loss to optimize the
   575	network:
   576	𝑇 𝑅𝐼 = −
   577	
   578	𝑁𝑡
   579	exp(𝑝+
   580	1 ∑
   581	𝑖 )
   582	log
   583	,
   584	−)
   585	𝑁𝑡 𝑖=1
   586	exp(𝑝+
   587	)
   588	+
   589	exp(𝑝
   590	𝑖
   591	𝑖
   592	
   593	(10)
   594	
   595	where 𝑝+
   596	𝑖 is the cosine similarity between the 𝑖th target domain sample
   597	and the sample with the same label that is the farthest away from the
   598	𝑖th target domain sample, and 𝑝−
   599	𝑖 is the cosine similarity between the
   600	𝑖th target domain sample and the sample with different labels that is
   601	the closest to the 𝑖th target domain sample.
   602	In a word, the total loss of the proposed method is formulated as:
   603	𝑡𝑜𝑡𝑎𝑙 = 𝑖𝑑 + 𝑇 𝑅𝐼 + 𝐶𝐶𝐿 + 𝜆𝑟𝑒𝑔 𝑟𝑒𝑔 ,
   604	5
   605	
   606	(11)
   607	
   608	Pattern Recognition 166 (2025) 111702
   609	
   610	Z. Zhang et al.
   611	
   612	Table 1
   613	The results (%) of ablation studies. F: fine-grained labels, C: coarse-grained labels, I:
   614	instance labels.
   615	
   616	where 𝜆𝑟𝑒𝑔 is the weight parameter to control the importance of PLC. In
   617	the end, the student network is updated by 𝑡𝑜𝑡𝑎𝑙 , and the parameters
   618	of the mean teacher network are updated by EMA of parameters
   619	from the student network. Correspondingly, EMA is implemented in a
   620	momentum-based manner:
   621	𝜃̂ 𝑙 = 𝜑𝜃̂ 𝑙−1 + (1 − 𝜑)𝜃 𝑙 ,
   622	
   623	Methods
   624	
   625	(12)
   626	
   627	where 𝜃̂ 𝑙 is the parameters of the mean teacher network in the 𝑙
   628	iteration, 𝜃̂ 𝑙−1 is the parameters of the mean teacher network in the
   629	𝑙 − 1 iteration, 𝜃 𝑙 is the parameters of the student network in the 𝑙
   630	iteration, 𝜑 = min((1 − 1∕𝑙), 𝜑0 ), and 𝜑0 is the higher bound of 𝜑
   631	and empirically set to 0.999. This momentum-based integration could
   632	improve the robustness of the mean teacher network.
   633	4. Experiments
   634	In this section, we first introduce the datasets and evaluation protocols, and present the implementation details. Then, we conduct the
   635	ablation studies to verify the effectiveness of the key components of our
   636	method, and compare our method with the state-of-the-art approaches
   637	on four cross-domain tasks. Afterwards, we visualize our results for
   638	intuitive verification. Finally, we analyze the influence of important
   639	hyper-parameters for the proposed method.
   640	
   641	a
   642	
   643	MSMT17 → Market
   644	
   645	Market → MSMT17
   646	
   647	mAP
   648	
   649	R1
   650	
   651	mAP
   652	
   653	R1
   654	
   655	BS+F
   656	BS+C
   657	BS+I
   658	
   659	76.7
   660	51.2
   661	74.3
   662	
   663	91.0
   664	70.6
   665	89.1
   666	
   667	28.3
   668	12.4
   669	25.9
   670	
   671	53.4
   672	33.5
   673	51.8
   674	
   675	BS+F+C
   676	BS+F+I
   677	BS+C+I
   678	
   679	78.6
   680	79.4
   681	77.9
   682	
   683	91.5
   684	92.3
   685	91.2
   686	
   687	29.6
   688	30.8
   689	27.5
   690	
   691	54.6
   692	56.7
   693	53.9
   694	
   695	BS+HPL
   696	BS+HPL+PLC
   697	
   698	80.3
   699	82.5
   700	
   701	92.5
   702	92.9
   703	
   704	31.7
   705	33.3
   706	
   707	59.1
   708	61.4
   709	
   710	BS+HPL+PLC+CL
   711	BS+HPL+PLC+CCLa
   712	BS+HPL+PLC+CCL
   713	Ours
   714	
   715	84.2
   716	84.5
   717	86.0
   718	87.2
   719	
   720	93.6
   721	93.7
   722	94.3
   723	95.0
   724	
   725	34.7
   726	34.5
   727	35.2
   728	36.6
   729	
   730	63.6
   731	63.4
   732	64.9
   733	65.8
   734	
   735	Median operator for the cluster centers in Eq. (9).
   736	
   737	Jaccard distance [57] with 𝑘-reciprocal nearest neighbors is used to
   738	cluster where 𝑘 = 30, the maximum distance between neighbors 𝑒𝑝𝑠
   739	is set to 0.6, the minimum number of neighbors for a sample to be
   740	considered as the center instance is set to 4, and the algorithm utilizes
   741	the Nearest Neighbors module to compute pointwise distances and find
   742	nearest neighbors. Furthermore, the un-clustered samples generated by
   743	DBSCAN are discarded during the training process. As for the setting of
   744	𝐾-means, the number of identities for the coarse-grained labels 𝐾2 is set
   745	to 120. Moreover, all the features are 𝐿2 -normalized in the experiments.
   746	In the test stage, the output of the mean teacher network is treated
   747	as the representation of pedestrian image to calculate the cosine similarity.
   748	
   749	4.1. Datasets and evaluation protocols
   750	We evaluate the proposed method on two benchmark datasets
   751	for person ReID, i.e., Market-1501 (Market) [50], and MSMT17 [34].
   752	Market-1501 [50] consists of 32,668 annotated images of 1501 identities captured by 6 cameras, where is captured by at least 2 cameras,
   753	and it contains 12,936 training images of 751 identities and 19,732 test
   754	images of 750 identities. The pedestrian detection rectangles of 3368
   755	query images are drawn manually, while the pedestrian detection rectangles in the gallery are detected by the deformable part model [51].
   756	MSMT17 [34] consists of 126,441 annotated images of 4101 identities
   757	captured by 15 cameras, where it is a large-scale dataset closer to
   758	the real scene because of covering multiple scenes and multiple time
   759	periods, and it contains 32,621 images of 1041 identities for training
   760	and 93,820 images of 3060 identities in the test stage. We also evaluate

codex
后三篇的方法段已经读完。为了回答“证据闭环”，我还需要从前 8 页里补一段关键消融表和可视化说明，重点看作者是否证明了自己的重定义，而不是只报总榜涨点。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf' - | nl -ba | sed -n '520,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   520	𝜕𝑓𝑛𝑜𝑟𝑚
   521	𝜕 𝑥⃗𝑖
   522	
   523	=
   524	
   525	1
   526	(𝑥̂ − 𝑐𝑜𝑠𝜃 𝑊̂ 𝑖 ),
   527	‖𝑊𝑖 ‖ 𝑖
   528	‖ ‖
   529	
   530	=
   531	
   532	1
   533	(𝑊̂ − 𝑐𝑜𝑠𝜃 𝑥̂𝑖 ),
   534	‖𝑥𝑖 ‖ 𝑖
   535	‖ ‖
   536	
   537	(10)
   538	
   539	because all features are normalized, they only have tangential gradients.
   540	𝜕𝐿𝑡
   541	
   542	For 𝑊𝑖𝑡+1 = 𝑊𝑖𝑡 − 𝜇 𝜕𝑊𝐶𝐸𝑡 [37] (𝑡 is the training iteration and 𝜇 is the
   543	𝑖
   544	
   545	𝑖
   546	
   547	the negative direction of 𝑥⃗𝑖 . Because 𝑥𝑡+1
   548	= 𝑥𝑡 + 𝛼 𝜕𝑥𝐶𝐸
   549	[26] (𝛼 is a pa𝑡
   550	𝑖
   551	𝑖
   552	‖
   553	‖
   554	rameter) during the training process, we can say that ‖𝑥𝑖 ‖ keeps decreasing during the training
   555	( process.
   556	) As the feature magnitudes decrease, the
   557	𝜕𝑓𝐶𝑜𝑛
   558	tangential gradient 𝑇
   559	also decreases accordingly. This indicates
   560	⃗
   561	
   562	4.2. Implement details
   563	
   564	𝜕𝐿𝑡
   565	
   566	learning rate), the magnitude of 𝑊𝑖 is not monotonous because the 𝜕𝑊𝐶𝐸𝑡
   567	𝑖
   568	
   569	may be positive or negative. Both the Naive and L2 softmax losses cannot
   570	provide monotonically decreasing tangential gradients.
   571	⃗ 𝑖 , 𝑥⃗𝑖 can be written as:
   572	The Condense loss and the gradients of 𝑊
   573	
   574	In this paper, we use two RTX TITAN with Ubuntu 18.04 to train the
   575	models, all experiments without special indication are conducted under
   576	Pytorch Framework, the Pytorch version is 1.0.
   577	For Re-ID tasks, we adopt the Bag of Tricks [5] to implement the
   578	Condense loss. ResNet50 is selected as the CNN backbone. The batch
   579	size is 64, and the size of the picture is 256 × 128. Adam is used as
   580	the optimizer during the training process, and the initial learning rate
   581	is 0.0003. We use 160 epochs to train models with triplet loss; its steps
   582	are set as (40, 90, 110, 130). At each step, the learning rate will be shrunk
   583	by 10 times. And we utilize the PK strategy for sampling when training
   584	triplet loss. In each batch, we select 8 people with 8 diﬀerent pictures of
   585	themselves. Because the Condense loss converges quickly and overﬁts on
   586	all datasets, the weight decay should be set as 5 × 10−3 . For other losses,
   587	the weight decay is set as 5 × 10−4 . We also use Pytorch to train face
   588	veriﬁcation models. All hyper-parameters are the same with the settings
   589	in ArcFace [4]. Because MS1MV2 is very large, hence the weight decay
   590	needn’t increase.
   591	
   592	‖‖ ‖
   593	‖ ‖
   594	‖ ‖
   595	𝑓𝐶𝑜𝑛 = −(𝑊𝑘 − 𝑥𝑖 )2 = 2‖
   596	‖𝑊𝑘 ‖‖𝑥𝑖 ‖cos𝜃𝑘 − ‖𝑊𝑘 ‖ − ‖𝑥𝑖 ‖ ,
   597	2
   598	
   599	𝜕𝑓𝐶𝑜𝑛
   600	⃗𝑖
   601	𝜕𝑊
   602	
   603	⃗ 𝑖 ),
   604	= 2(⃗
   605	𝑥𝑖 − 𝑊
   606	
   607	𝜕𝑓𝐶𝑜𝑛
   608	𝜕 𝑥⃗𝑖
   609	
   610	2
   611	
   612	(11)
   613	
   614	⃗ 𝑖 − 𝑥⃗𝑖 ).
   615	= 2(𝑊
   616	
   617	𝜕𝑓
   618	For 𝐶𝑜𝑛
   619	we can achieve its radial gradient by:
   620	⃗𝑖
   621	𝜕𝑊
   622	
   623	𝑅(
   624	
   625	𝜕𝑓𝐶𝑜𝑛
   626	⃗𝑖
   627	𝜕𝑊
   628	
   629	‖
   630	‖ ‖
   631	̂
   632	⃗
   633	̂ ‖ ‖
   634	) = 2‖
   635	‖𝑥𝑖 ‖𝑐𝑜𝑠𝜃 𝑊𝑖 − 2𝑊𝑖 = 2𝑊𝑖 (‖𝑥𝑖 ‖𝑐𝑜𝑠𝜃 − ‖𝑊𝑖 ‖).
   636	
   637	(12)
   638	
   639	Consequently, its tangential gradient is expressed as:
   640	(
   641	)
   642	(
   643	)
   644	𝜕𝑓𝐶𝑜𝑛
   645	𝜕𝑓
   646	𝜕𝑓𝐶𝑜𝑛
   647	‖
   648	̂
   649	𝑇
   650	= 𝐶𝑜𝑛 − 𝑅
   651	=‖
   652	‖2𝑥𝑖 ‖(𝑥̂𝑖 − 𝑐𝑜𝑠𝜃 𝑊𝑖 ).
   653	⃗𝑖
   654	⃗𝑖
   655	⃗𝑖
   656	𝜕𝑊
   657	𝜕𝑊
   658	𝜕𝑊
   659	Symmetrically, the radial and tangential gradients of
   660	follows:
   661	(
   662	)
   663	(
   664	)
   665	𝜕𝑓𝐶𝑜𝑛
   666	‖
   667	‖ ‖
   668	𝑅
   669	= 2𝑥̂ 𝑖 ‖
   670	‖𝑊𝑖 ‖𝑐𝑜𝑠𝜃 − ‖𝑥𝑖 ‖ ,
   671	𝜕 𝑥⃗𝑖
   672	(
   673	)
   674	(
   675	)
   676	𝜕𝑓𝐶𝑜𝑛
   677	‖ ̂
   678	𝑇
   679	=‖
   680	‖2𝑊𝑖 ‖ 𝑊𝑖 − 𝑐𝑜𝑠𝜃 𝑥̂𝑖 ,
   681	𝜕 𝑥⃗
   682	
   683	(13)
   684	𝜕𝑓𝐶𝑜𝑛𝑑
   685	𝜕 𝑥⃗𝑖
   686	
   687	are as
   688	4.3. Comparison with other losses
   689	Table 1 reports the experimental results of diﬀerent losses on Market1501 and MSMT17 datasets. When NormFace, CosFace and ArcFace are
   690	compared with the naive softmax loss, we doubt whether L2 softmax
   691	losses are suitable for Re-ID tasks because all results are worse than the
   692	
   693	(14)
   694	
   695	𝑖
   696	
   697	5
   698	
   699	Pattern Recognition 172 (2026) 112443
   700	
   701	X. Yang et al.
   702	
   703	Table 1
   704	The Rank-1, Rank-5 and mAP (%) results of diﬀerent losses on Market-1501
   705	and MSMT17, respectively. ∗ means the result of AM-Softmax is from Circle loss
   706	[36]. 𝑤𝑜∕𝑚 means the margin parameter in the Condense loss is not considered
   707	and 𝑤∕𝑚 means the margin parameter is used for computing loss.
   708	Methods
   709	
   710	Market-1501
   711	
   712	Table 2
   713	The Rank-1, Rank-5, Rank-10 and mAP (%) results for diﬀerent methods on
   714	Market-1501 and MSMT17 datasets.
   715	Methods
   716	
   717	MSMT17
   718	
   719	rank-1
   720	
   721	rank-5
   722	
   723	mAP
   724	
   725	rank-1
   726	
   727	rank-5
   728	
   729	mAP
   730	
   731	Naive Softmax
   732	Naive Softmax (𝑤𝑑 5 × 10−3 )
   733	NormFace
   734	CosFace
   735	ArcFace
   736	AM-Softmax∗
   737	
   738	92.8
   739	93.2
   740	84.7
   741	91.2
   742	91.6
   743	92.4
   744	
   745	97.8
   746	97.3
   747	93.7
   748	96.6
   749	97.4
   750	-
   751	
   752	82.6
   753	83.7
   754	63.5
   755	77.5
   756	77.5
   757	83.8
   758	
   759	73.8
   760	73.7

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf' - | nl -ba | sed -n '612,930p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   612	Table 1
   613	The results (%) of ablation studies. F: fine-grained labels, C: coarse-grained labels, I:
   614	instance labels.
   615	
   616	where 𝜆𝑟𝑒𝑔 is the weight parameter to control the importance of PLC. In
   617	the end, the student network is updated by 𝑡𝑜𝑡𝑎𝑙 , and the parameters
   618	of the mean teacher network are updated by EMA of parameters
   619	from the student network. Correspondingly, EMA is implemented in a
   620	momentum-based manner:
   621	𝜃̂ 𝑙 = 𝜑𝜃̂ 𝑙−1 + (1 − 𝜑)𝜃 𝑙 ,
   622	
   623	Methods
   624	
   625	(12)
   626	
   627	where 𝜃̂ 𝑙 is the parameters of the mean teacher network in the 𝑙
   628	iteration, 𝜃̂ 𝑙−1 is the parameters of the mean teacher network in the
   629	𝑙 − 1 iteration, 𝜃 𝑙 is the parameters of the student network in the 𝑙
   630	iteration, 𝜑 = min((1 − 1∕𝑙), 𝜑0 ), and 𝜑0 is the higher bound of 𝜑
   631	and empirically set to 0.999. This momentum-based integration could
   632	improve the robustness of the mean teacher network.
   633	4. Experiments
   634	In this section, we first introduce the datasets and evaluation protocols, and present the implementation details. Then, we conduct the
   635	ablation studies to verify the effectiveness of the key components of our
   636	method, and compare our method with the state-of-the-art approaches
   637	on four cross-domain tasks. Afterwards, we visualize our results for
   638	intuitive verification. Finally, we analyze the influence of important
   639	hyper-parameters for the proposed method.
   640	
   641	a
   642	
   643	MSMT17 → Market
   644	
   645	Market → MSMT17
   646	
   647	mAP
   648	
   649	R1
   650	
   651	mAP
   652	
   653	R1
   654	
   655	BS+F
   656	BS+C
   657	BS+I
   658	
   659	76.7
   660	51.2
   661	74.3
   662	
   663	91.0
   664	70.6
   665	89.1
   666	
   667	28.3
   668	12.4
   669	25.9
   670	
   671	53.4
   672	33.5
   673	51.8
   674	
   675	BS+F+C
   676	BS+F+I
   677	BS+C+I
   678	
   679	78.6
   680	79.4
   681	77.9
   682	
   683	91.5
   684	92.3
   685	91.2
   686	
   687	29.6
   688	30.8
   689	27.5
   690	
   691	54.6
   692	56.7
   693	53.9
   694	
   695	BS+HPL
   696	BS+HPL+PLC
   697	
   698	80.3
   699	82.5
   700	
   701	92.5
   702	92.9
   703	
   704	31.7
   705	33.3
   706	
   707	59.1
   708	61.4
   709	
   710	BS+HPL+PLC+CL
   711	BS+HPL+PLC+CCLa
   712	BS+HPL+PLC+CCL
   713	Ours
   714	
   715	84.2
   716	84.5
   717	86.0
   718	87.2
   719	
   720	93.6
   721	93.7
   722	94.3
   723	95.0
   724	
   725	34.7
   726	34.5
   727	35.2
   728	36.6
   729	
   730	63.6
   731	63.4
   732	64.9
   733	65.8
   734	
   735	Median operator for the cluster centers in Eq. (9).
   736	
   737	Jaccard distance [57] with 𝑘-reciprocal nearest neighbors is used to
   738	cluster where 𝑘 = 30, the maximum distance between neighbors 𝑒𝑝𝑠
   739	is set to 0.6, the minimum number of neighbors for a sample to be
   740	considered as the center instance is set to 4, and the algorithm utilizes
   741	the Nearest Neighbors module to compute pointwise distances and find
   742	nearest neighbors. Furthermore, the un-clustered samples generated by
   743	DBSCAN are discarded during the training process. As for the setting of
   744	𝐾-means, the number of identities for the coarse-grained labels 𝐾2 is set
   745	to 120. Moreover, all the features are 𝐿2 -normalized in the experiments.
   746	In the test stage, the output of the mean teacher network is treated
   747	as the representation of pedestrian image to calculate the cosine similarity.
   748	
   749	4.1. Datasets and evaluation protocols
   750	We evaluate the proposed method on two benchmark datasets
   751	for person ReID, i.e., Market-1501 (Market) [50], and MSMT17 [34].
   752	Market-1501 [50] consists of 32,668 annotated images of 1501 identities captured by 6 cameras, where is captured by at least 2 cameras,
   753	and it contains 12,936 training images of 751 identities and 19,732 test
   754	images of 750 identities. The pedestrian detection rectangles of 3368
   755	query images are drawn manually, while the pedestrian detection rectangles in the gallery are detected by the deformable part model [51].
   756	MSMT17 [34] consists of 126,441 annotated images of 4101 identities
   757	captured by 15 cameras, where it is a large-scale dataset closer to
   758	the real scene because of covering multiple scenes and multiple time
   759	periods, and it contains 32,621 images of 1041 identities for training
   760	and 93,820 images of 3060 identities in the test stage. We also evaluate
   761	the proposed method on synthetic dataset, i.e., PersonX [52], which
   762	includes 9840 training images.
   763	The evaluation protocols are the mean average precision (mAP) and
   764	the Cumulated Matching Characteristics (CMC) at Rank-1 (R1), Rank-5
   765	(R5) and Rank-10 (R10) accuracies. The post-processing methods are
   766	not used for inference, i.e., re-ranking or multi-query fusion.
   767	
   768	4.3. Ablation studies
   769	In this subsection, we conduct ablation studies to investigate the
   770	effectiveness of each component in the proposed method.
   771	Effectiveness of each component in our method. The results of
   772	ablation studies are listed in Table 1, where F, C and I denote the
   773	fine-grained labels, the coarse-grained labels and the instance labels,
   774	respectively. Here, the baseline (BS+F) consists of the mean teacher
   775	network and the student network optimized by the target ID loss
   776	and the triplet loss with the fine-grained labels, and the symbol †
   777	indicates that we adopt the median operator to compute the cluster
   778	centers in Eq. (9). BS+F is similar to the baseline of many state-ofthe-art methods [13,14,58]. From the table, we can draw the following
   779	conclusions.
   780	Firstly, reasonable use of multiple kinds of pseudo labels boosts the
   781	performance of cross-domain person ReID. For example, two kinds of
   782	pseudo labels (BS+F+C, BS+F+I, and BS+C+I) are better than one kind
   783	(BS+F, BS+C, and BS+I), and the proposed BS+HPL which applies three
   784	kinds of pseudo labels obtains better results than BS+F+C, BS+F+I,
   785	and BS+C+I, respectively. It is because simultaneously learning three
   786	different kinds of pseudo labels could make full use of their own
   787	advantages so as to improve the discrimination of pedestrian features.
   788	Secondly, compared BS+HPL+PLC with BS+HPL, PLC further improves
   789	the performance, which verifies it could enhance the quality of the
   790	pseudo labels. Thirdly, BS+HPL+PLC+CCL achieves better results than
   791	BS+HPL+PLC+CL, where CCL denotes the Eq. (8) introducing the
   792	confidence coefficient implemented by Eq. (9), and CL indicates the
   793	circle loss [49]. It is because the confidence coefficient re-weights the
   794	similarity of sample pairs in order to reduce the influence of noisy
   795	labels.
   796	Furthermore, BS+HPL+PLC+CCL† represents that the cluster centers in Eq. (9) are computed using median instead of mean. From the
   797	table, the performance of BS+HPL+PLC+CCL outperforms BS+HPL+
   798	
   799	4.2. Implementation details
   800	In the experiments, we adopt ResNet50 as the backbone network,
   801	where it is initialized by the pre-trained weights of ImageNet [53].
   802	All pedestrian images are resized to 256 × 128 before feeding into the
   803	network, and the training images are augmented by random horizontal
   804	flipping, random erasing, random cropping, padding [54], and the
   805	mixup strategy [55]. We employ the Adam optimizer [56] with the
   806	weight decay of 0.0005 to optimize the network.
   807	In the stage of the source domain pre-training, the batch size is set
   808	to 64 which includes 16 source domain identities, and each identity
   809	contains 4 images. The initial learning rate is set to 0.00035 and is
   810	decreased to 1∕10 of its previous value on the 40th and 70th epoch in
   811	the total 80 epochs.
   812	In the stage of the target domain fine-tuning, the batch size is set to
   813	128 which includes 16 target domain identities and 16 source domain
   814	identities, and each identity contains 4 images. The number of epochs
   815	is set to 80, and the learning rate is fixed at 0.00035. We set 𝑚 = 0.25,
   816	𝛾 = 128 in Eq. (8). The number of the target domain features in the
   817	memory bank  is set to 8192. For DBSCAN, similar to [13,18], the
   818	6
   819	
   820	Pattern Recognition 166 (2025) 111702
   821	
   822	Z. Zhang et al.
   823	
   824	Fig. 3. The first row corresponds to the clustering choices of the first row in Table 2, and the second row corresponds to the clustering choices of the second row in Table 2.
   825	Table 2
   826	The results (%) of different clustering choices for the fine-grained labels and the coarsegrained labels. Here, 𝐾 and 𝐷 denote 𝐾-means algorithm and DBSCAN, respectively.
   827	The tuple indicates the clustering algorithms of the fine-grained labels and the coarsegrained labels, respectively.
   828	Choices
   829	
   830	(K, K)
   831	(K, D)
   832	(D, K)
   833	(D, D)
   834	
   835	MSMT17 → Market
   836	
   837	Table 3
   838	The results (%) of different forms of the pseudo labels. The triplet represents the label
   839	forms of fine-grained labels, coarse-grained labels, and instance labels, respectively. ℎ
   840	and 𝑠 denote hard and soft, respectively.
   841	Forms
   842	
   843	Market → MSMT17
   844	
   845	mAP
   846	
   847	R1
   848	
   849	mAP
   850	
   851	R1
   852	
   853	83.8
   854	85.5
   855	87.2
   856	84.7
   857	
   858	93.2
   859	94.0
   860	95.0
   861	93.5
   862	
   863	32.7
   864	34.9
   865	36.6
   866	34.1
   867	
   868	61.2
   869	64.2
   870	65.8
   871	63.8
   872	
   873	(h, h, h)
   874	(h, h, s)
   875	(h, s, h)
   876	(h, s, s)
   877	(s, h, h)
   878	(s, h, s)
   879	(s, s, h)
   880	(s, s, s)
   881	
   882	PLC+CCL† , which shows that the mean operation is more effective than
   883	the median operation when calculating the cluster centers in Eq. (9).
   884	It is because the average of pedestrian features is more holistic than
   885	the median of pedestrian features. Finally, the classification loss for the
   886	source domain is introduced on the basis of BS+HPL+PLC+CCL in order
   887	to limit the forgetting effect on the source domain, denoted as Ours,
   888	and Ours achieves a new state-of-the-art performance of cross-domain
   889	person ReID.
   890	Clustering choices for the fine-grained labels and the coarsegrained labels. Since DBSCAN and 𝐾-means clustering algorithms are
   891	commonly used for cross-domain person ReID, we test their performance for generating the fine-grained labels and the coarse-grained
   892	labels, and the results are listed in Table 2. To ensure that the results
   893	in each row of Table 2 are obtained with the optimal parameters, for
   894	each choice, we study the influence of the parameters of the clustering
   895	algorithms on MSMT17→Market, such as the number of cluster centers
   896	in the 𝐾-means algorithm and two parameters (the maximum distance
   897	between neighbors 𝑒𝑝𝑠 and the minimum number of neighbors for
   898	a sample to be considered as the center instance ‘‘min_samples’’) in
   899	DBSCAN. The results are shown in Figs. 3 and 4, where the first row
   900	and the second row in Fig. 3 correspond to the clustering choices of
   901	the first row and the second row in Table 2, and the first row and the
   902	second row in Fig. 4 correspond to the clustering choices of the third
   903	row and the fourth row in Table 2.
   904	From Table 2 and Fig. 4(a), we can see that the performance is
   905	best when using DBSCAN for the fine-grained labels with 𝑒𝑝𝑠 = 0.6
   906	and min_samples = 4. It is because the original target domain samples
   907	contain noises and DBSCAN can detect the noises. Furthermore, the
   908	performance is better when using the 𝐾-means algorithm for the coarsegrained labels, where the coarse-grained labels are generated from
   909	noise-free samples. From Fig. 4(b), the results are best when 𝐾2 is
   910	
   911	MSMT17 → Market
   912	
   913	Market → MSMT17
   914	
   915	mAP
   916	
   917	R1
   918	
   919	mAP
   920	
   921	R1
   922	
   923	84.5
   924	85.8
   925	87.2
   926	84.7
   927	77.4
   928	79.0
   929	74.5
   930	74.8

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf' - | nl -ba | sed -n '733,1080p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   733	B. Evaluation Protocols and Metrics
   734	1) Evaluation Protocols: Following [1], we set up four
   735	different evaluation configurations, i.e., Clean Eval, Corrupt
   736	Eval, Corrupt Query and Corrupt Gallery, according to whether
   737	the samples in gallery/query set are corrupted. As shown in
   738	Table II, Clean Eval refers to the configuration that both
   739	the query set and gallery set include only clean samples.
   740	In contrast, Corrupt Eval indicates that both the query set
   741	and gallery set are pre-processed with random corruption
   742	operation. To comprehensively evaluate the robustness of our
   743	model, two additional evaluation settings are adopted, that is,
   744	Corrupt Query includes corrupted query set and clean gallery
   745	set while the counterpart Corrupt Gallery includes clean query
   746	set and corrupted gallery set. Generally, corrupting the query
   747	leads to a more challenging sample ranking (lower Rank-1),
   748	while corrupting the gallery provides the model with more
   749	hard samples (lower mINP) [1].
   750	2) Evaluation Metrics: Following the previous ReID methods [51], [52], we adopted the widely-used metrics Cumulative
   751	Match Characteristics of Rank-1 accuracy (R-1) [53], mean
   752	Average Precision (mAP) [11] and mean Inverse Negative
   753	Penalty (mINP) [19] to evaluate the effectiveness of the
   754	proposed method. We reported all the metrics on the above
   755	four evaluation protocols.
   756	C. Implementation Details
   757	The proposed method1 is implemented using Pytorch and
   758	trained on a 48G NVIDIA Quadro RTX8000 GPU. In all
   759	1 The source codes are publicly
   760	sdustzxy/CFAT for research purpose.
   761	
   762	available
   763	
   764	at
   765	
   766	https://github.com/
   767	
   768	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.
   769	
   770	ZHANG et al.: CORRUPTION-INVARIANT PERSON RE-IDENTIFICATION VIA COARSE-TO-FINE FEATURE ALIGNMENT
   771	
   772	1091
   773	
   774	TABLE III
   775	P ERFORMANCE C OMPARISON ON THE M ARKET-1501 DATASET. T HE B EST R ESULTS A RE H IGHLIGHTED IN B OLD . * I NDICATES THE BASELINE
   776	
   777	TABLE IV
   778	P ERFORMANCE C OMPARISON ON THE CUHK-03 DATASET
   779	
   780	experiments, we chose ViT-B as the backbone model, which
   781	consists of 12 transformer layers. The input image is resized
   782	to 256 × 128 and equally cut into 16 × 16 patches for
   783	embedding. In addition, we randomly chose either weak or
   784	strong augmentations with a probability of 0.5. The weak augmentation includes some common strategies such as flipping,
   785	cropping, padding, etc. The strong augmentation randomly
   786	incorporates one or more types of corruption in Table I on
   787	top of weak augmentation. It is worth noting that both the
   788	corruption type and severity are randomly selected with an
   789	equal probability during training.
   790	During training, we utilized the SGD optimizer to train
   791	our model, and the CosineAnnealing Decay with Warmup
   792	strategy was adopted to adjust the learning rate. In particular,
   793	we set the base learning rate to 0.001 and performed warmup
   794	in the first five epochs. The batch size is set to 64 and the
   795	parameters are optimized in a total of 120 epochs. To ease the
   796	alignment across different corruptions, we adopted a coarseto-fine strategy to optimize the proposed model. In the coarse
   797	alignment stage, we optimized the SFE sub-network under
   798	knowledge distillation paradigm by freezing TFE parameters.
   799	In the fine alignment stage, the SFE is further fine-tuned to
   800	construct semantic structure across clean and corrupted data
   801	with a bridge feature generation module. It is worth noting that
   802	to ensure our model does not overfit the test set, we follow the
   803	settings of classical ReID methods [1], [54], [55], leaving a
   804	validation set for hyper-parameter tuning. Moreover, random
   805	corruptions are also applied to the testing set. It ensures intraclass variations and avoids perfect overlap between training
   806	and testing.
   807	D. Comparison With State-of-the-Art Methods
   808	In this section, we compare our proposed method with the
   809	state-of-the-art (SOTA) methods 1) classical ReID methods,
   810	
   811	such as Deep Person ReID (DP-ReID) [15], DP-ReID with
   812	tricks (DP+Tricks) [3] and Faster-ReID [56]; 2) ReID methods
   813	for challenging scenarios, such as PFD [54], DC-Form [16],
   814	and SP-Loss [55]; and 3) Robust ReID for corrupted scene,
   815	i.e., CIL-ReID model [1]. However, the original CIL-ReID is
   816	trained on purely clean images and evaluated on corrupted
   817	images. To be compared fairly, we re-trained all methods
   818	on the generated corrupted dataset and reported the results
   819	of CIL-ReID (dubbed as CIL-ReID∗ ) as baseline. To reduce
   820	the impact of randomness on the evaluation metrics, all
   821	experiments were averaged over ten runs. Table III-V compare
   822	results on Market-1501, CUHK-03 and MSMT-17 datasets,
   823	respectively.
   824	1) Performance Comparison on Market-1501 Dataset: As
   825	shown in Table III, compared to the baseline, our proposed
   826	method shows an increase of 14.48% in mINP metric (from
   827	18.48% to 32.96%), an increase of 13.38% in mAP metric
   828	(from 56.36% to 69.74%), and an increase of 9.82% in
   829	Rank-1 metric (from 75.86% to 85.68%) under the Corrupt
   830	Eval protocol. It shows consistently improvement trend under
   831	the Corrupt Query and Corrupt Gallery protocols. In addition,
   832	our proposed method also outperforms SOTA ReID methods
   833	and ReID methods for challenging scenarios with a large margin. This demonstrates effectiveness of our proposed method
   834	for complicated corrupted scenes. However, under the Clean
   835	Eval protocol, the model’s recognition accuracy shows a slight
   836	decrease in clean scenes compared to the baseline. We suppose it is because the alignment among different corruption
   837	modals causes information loss compared with the baseline.
   838	Such alignment slightly sacrifices the model preference to
   839	specific identity information within clean samples, but benefits
   840	from exploiting corruption-invariant features. Therefore, the
   841	proposed methods significantly boosts retrieval performance
   842	on complicated scenes with various corruptions.
   843	
   844	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.
   845	
   846	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf' - | nl -ba | sed -n '789,980p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   789	TABLE IV
   790	C OMPARISON W ITH S TATE - OF - THE -A RT M ETHODS ON DukeMTMC-VideoReID AND MARS. T HE A PPROACHES A RE S EPARATED I NTO S EVERAL
   791	G ROUPS . RNN S R EPRESENTS R ECURRENT N EURAL N ETWORKS , OF R EPRESENTS O PTICAL F LOW, TP R EPRESENTS T EMPORAL P OOLING ,
   792	3D R EPRESENTS 3D C ONVOLUTIONAL N EURAL N ETWORKS , GM R EPRESENTS G RAPH M ODELS , AND AM R EPRESENTS ATTENTION
   793	M ECHANISM . B OLD VALUES I NDICATE THE B EST P ERFORMANCE . R ANK 20, R ANK 5, R ANK 1, AND mAP ACCURACY A RE L ISTED
   794	
   795	TABLE V
   796	A BLATION S TUDIES ON LS-VID, MARS, AND D UKE -V IDEO . CSSA1−4 D ENOTE F OUR VARIANTS OF THE P ROPOSED CSSA F RAMEWORK .
   797	FACTORS IN C OLUMN ‘S ETTING ’ D ENOTE W HETHER THE E XPERIMENT I S P ERFORMED W ITH THE C ORRESPONDING M ODULE .
   798	B OLD VALUES I NDICATE THE B EST P ERFORMANCE . R ANK 5, R ANK 1 AND mAP ACCURACY A RE L ISTED
   799	
   800	1) The Influence of the Reduction Ratio r : r determines
   801	the size of the intermediate features in CARE. Fig. 5(a)
   802	gives some comparisons. We can observe that as the hyperparameters increase, the performance first demonstrates an
   803	escalating trend. When r = 4, the proposed model reaches
   804	peak performance. As r increases further, the performance
   805	showed a gradual decrease. Actually, large parameters will lose
   806	too much identification information, and small parameters will
   807	cause a heavy computational burden.
   808	
   809	2) The Impact of the Semantic Part Number N p : Intuitively, the number of clustering categories N p regulates the
   810	semantic part granularity. Quantitative ablation experiments
   811	are performed to explicitly figure out the most appropriate N p . Experimental results are indicated in Fig. 5(b). It is
   812	shown that the proposed CSSA is sufficiently resilient to
   813	various N p . Moreover, we also find that N p = 6 is the
   814	optimal value. Therefore, N p is constantly assigned to 6 in
   815	this work.
   816	
   817	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.
   818	
   819	
codex
Condense loss 的方法段在 PDF 抽取里有一点排版交错，但关键论证完整：先观察范数随训练和准确率反向变化，再把 softmax 的内积形式改写成欧氏距离形式。我会继续补它的表格尾部，确认消融对比是不是支撑这个叙事。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf' - | nl -ba | sed -n '844,910p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   844	Visualization of retrieval results. To intuitively demonstrate the
   845	effect of abundant semantics in our proposed method to promote
   846	cloth-changing person Re-ID, we compare the retrieval results of
   847	CSSC and the baseline model in Fig. 3. As shown in the firstrow results of (a) and (b), when people change their clothes, the
   848	baseline model is unable to identify pedestrians correctly with the
   849	interference of similar visual appearances, e.g., similar colors, and
   850	clothing textures. However, the pedestrian images with different
   851	clothes are correctly retrieved as the top results for our method. As
   852	observed from the second-row results of (a) and (b), identity-related
   853	semantics in global and local like body and part shapes excavated
   854	by our proposed SMR module help re-identify the same pedestrian
   855	despite changing clothes. Our proposed CSSC is better at capturing
   856	content and salient semantics, such as discriminative shorts and shoes
   857	in (b), to get good retrieval results.
   858	IV. C ONCLUSION
   859	In this paper, we address the cloth-changing challenge and propose
   860	a novel Content and Salient Semantics Collaboration (CSSC) framework with effective semantic interaction. In the framework, we propose the Semantics Mining and Refinement (SMR) module equipped
   861	with distinct pooling strategies to mine and leverage identity-related
   862	content semantics and identity-related salient semantics. Extensive
   863	experiments demonstrate the effectiveness and superiority of our
   864	method. We hope this paper can inspire more research to advance
   865	cloth-changing person Re-ID without relying on unpractical and
   866	inflexible auxiliary modalities or extra annotations.
   867	
   868	R EFERENCES
   869	[1] X. Qian, Y. Fu, Y.-G. Jiang, T. Xiang, and X. Xue, “Multi-scale deep
   870	learning architectures for person re-identification,” in Proceedings of the
   871	IEEE international conference on computer vision, 2017, pp. 5399–5408.
   872	[2] R. Hou, B. Ma, H. Chang, X. Gu, S. Shan, and X. Chen, “Interactionand-aggregation network for person re-identification,” in Proceedings of
   873	the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
   874	2019, pp. 9317–9326.
   875	[3] Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang, “Beyond part models:
   876	Person retrieval with refined part pooling (and a strong convolutional
   877	baseline),” in Proceedings of the European Conference on Computer
   878	Vision, 2018, pp. 480–496.
   879	[4] Q. Wang, X. Qian, B. Li, Y. Fu, and X. Xue, “Rethinking person
   880	re-identification from a projection-on-prototypes perspective,” arXiv
   881	preprint arXiv:2308.10717, 2023.
   882	[5] N. Meng, Q. Wang, B. Li, and X. Xue, “Unleashing the potential of
   883	tracklets for unsupervised video person re-identification,” arXiv preprint
   884	arXiv:2406.14261, 2024.
   885	[6] Q. Wang, X. Qian, B. Li, and X. Xue, “Distribution aligned semantics adaption for lifelong person re-identification,” arXiv preprint
   886	arXiv:2405.19695, 2024.
   887	[7] Q. Wang, B. Li, and X. Xue, “When large vision-language models meet
   888	person re-identification,” arXiv preprint arXiv:2411.18111, 2024.
   889	[8] X. Qian, W. Wang, L. Zhang, F. Zhu, Y. Fu, T. Xiang, Y.-G. Jiang,
   890	and X. Xue, “Long-term cloth-changing person re-identification,” in
   891	Proceedings of the Asian Conference on Computer Vision, 2020, pp.
   892	71–88.
   893	[9] Q. Yang, A. Wu, and W.-S. Zheng, “Person re-identification by contour
   894	sketch under moderate clothing change,” IEEE Transactions on Pattern
   895	Analysis and Machine Intelligence, vol. 43, no. 6, pp. 2029–2046, 2019.
   896	[10] Q. Wang, X. Qian, Y. Fu, and X. Xue, “Co-attention aligned mutual
   897	cross-attention for cloth-changing person re-identification,” in Proceedings of the Asian Conference on Computer Vision, 2022, pp. 2270–2288.
   898	[11] Q. Wang, X. Qian, B. Li, Y. Fu, and X. Xue, “Image-text-image
   899	knowledge transferring for lifelong person re-identification with hybrid
   900	clothing states,” arXiv preprint arXiv:2405.16600, 2024.
   901	[12] P. Hong, T. Wu, A. Wu, X. Han, and W.-S. Zheng, “Fine-grained shapeappearance mutual learning for cloth-changing person re-identification,”
   902	in Proceedings of the IEEE/CVF Conference on Computer Vision and
   903	Pattern Recognition, 2021, pp. 10 513–10 522.
   904	[13] X. Jin, T. He, K. Zheng, Z. Yin, X. Shen, Z. Huang, R. Feng, J. Huang,
   905	Z. Chen, and X.-S. Hua, “Cloth-changing person re-identification from a
   906	single image with gait prediction and regularization,” in Proceedings of
   907	the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
   908	2022, pp. 14 278–14 287.
   909	[14] X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, and X. Chen, “Clotheschanging person re-identification with rgb modality only,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
   910	Recognition, 2022, pp. 1060–1069.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf' - | nl -ba | sed -n '731,900p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   731	Naive Softmax
   732	Naive Softmax (𝑤𝑑 5 × 10−3 )
   733	NormFace
   734	CosFace
   735	ArcFace
   736	AM-Softmax∗
   737	
   738	92.8
   739	93.2
   740	84.7
   741	91.2
   742	91.6
   743	92.4
   744	
   745	97.8
   746	97.3
   747	93.7
   748	96.6
   749	97.4
   750	-
   751	
   752	82.6
   753	83.7
   754	63.5
   755	77.5
   756	77.5
   757	83.8
   758	
   759	73.8
   760	73.7
   761	57.4
   762	61.3
   763	62.3
   764	75.6
   765	
   766	85.3
   767	85.5
   768	70.9
   769	75.3
   770	75.8
   771	-
   772	
   773	50.3
   774	49.9
   775	29.5
   776	33.4
   777	35.1
   778	49.3
   779	
   780	Condense (𝑤𝑜∕𝑚)
   781	Condense (𝑤∕𝑚)
   782	
   783	94.6
   784	94.7
   785	
   786	98.3
   787	98.3
   788	
   789	86.1
   790	86.1
   791	
   792	76.3
   793	78.5
   794	
   795	86.9
   796	88.2
   797	
   798	51.8
   799	54.0
   800	
   801	Batch Normalization
   802	Layer Normalization
   803	L2 Normalization
   804	Ours
   805	
   806	Market-1501
   807	
   808	MSMT17
   809	
   810	rank-1
   811	
   812	rank-5
   813	
   814	rank-10
   815	
   816	mAP
   817	
   818	rank-1
   819	
   820	rank-5
   821	
   822	rank-10
   823	
   824	mAP
   825	
   826	92.8
   827	
   828	97.8
   829	
   830	98.6
   831	
   832	82.6
   833	
   834	73.8
   835	
   836	85.3
   837	
   838	89.7
   839	
   840	50.3
   841	
   842	94.3
   843	
   844	97.9
   845	
   846	98.9
   847	
   848	86.3
   849	
   850	75.4
   851	
   852	86.5
   853	
   854	89.9
   855	
   856	50.4
   857	
   858	73.3
   859	
   860	87.6
   861	
   862	91.9
   863	
   864	54.6
   865	
   866	53.2
   867	
   868	74.9
   869	
   870	80.4
   871	
   872	32.0
   873	
   874	94.7
   875	
   876	98.3
   877	
   878	98.9
   879	
   880	86.1
   881	
   882	78.5
   883	
   884	88.2
   885	
   886	90.9
   887	
   888	54.0
   889	
   890	4.4. Comparison with regularization/normalization
   891	To verify that Condense loss not only achieves compression of the
   892	feature magnitudes but also yields better results, we conduct a comparative experiment between regularization/normalization methods and
   893	Condense loss (as shown in Table 2). The experimental results show
   894	that while combining Batch Normalization or Layer Normalization with
   895	softmax loss achieves relatively good accuracy, their performance remains unsatisfactory on the challenging dataset MSMT17. The results
   896	for L2 normalization are even worse. In contrast, our Condense loss signiﬁcantly outperforms other methods on both datasets, demonstrating
   897	its ability to eﬀectively stabilize the model’s update process while compressing feature magnitudes, thereby signiﬁcantly improving accuracy.
   898	
   899	naive softmax loss. We refer to the results reported by Sun et al. [36]
   900	(AM-Softmax is the same with CosFace), it can achieve slight improvement, i.e., mAP 1.2 % improvement on Market-1501 and rank-1 1.8 %

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Confidence guided semi-supervised cross-modality person re-identification.pdf' - | nl -ba | sed -n '850,1100p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   850	approach.
   851	The proposed approach demonstrates significant improvements in
   852	Rank-1 and mAP accuracy on the RegDB dataset, as show in Table
   853	2. Specifically, in the Visible to Infrared search mode, there is a
   854	remarkable improvement of 19.88%(=89.36%–69.48%) in Rank-1 accuracy and an impressive enhancement of 18.76%(=84.17%–65.41%)
   855	in mAP. Similarly, under the Infrared to Visible search mode,
   856	our method achieves substantial performance gains with a boost
   857	of 19.26%(=89.11%–69.85%) in Rank-1 accuracy and a notable
   858	improvement of 16.32%(=81.49%–65.17%) in mAP accuracy.
   859	
   860	4.2. Implementation details
   861	Our proposed approach is implemented using the PyTorch library.
   862	The settings of the two-stream model adhere to AGW [35]. We use a
   863	pre-trained ResNet50, trained on ImageNet, as the shared layers in the
   864	feature extraction module. In the training stage, 30% of the training
   865	images are labeled, while the remaining 70% are unlabeled. During
   866	inference, we compute cosine similarity using feature maps obtained
   867	from the GeM pooling layer. At the start of each training epoch, pseudolabels are generated for both modalities using the DBSCAN algorithm.
   868	In each mini-batch, 16 visible images and 16 infrared images from
   869	8 identities are selected from NNC-sampler for training labeled and
   870	unlabeled samples respectively. The batch size is set to 128.
   871	In experiments, all the images are resized to 288 × 144. During
   872	the training process, we apply random cropping, random horizontal
   873	flipping, and random erasing techniques to augment the images. Our
   874	model is optimized using the Adam optimizer with an initial learning
   875	rate of 0.00035. To enhance convergence, the learning rate is reduced
   876	by a factor of 0.1 every 20 epochs throughout the total duration of
   877	100 epochs. In the first stage of semi-supervised training, we train
   878	our model for 50 epochs exclusively. Subsequently, in the remaining 50 epochs, we introduce both pseudo-labeled cross-modality data
   879	generation module and pseudo-labeled cross-modality data refinement
   880	module.
   881	
   882	4.3.3. Comparison with DPIS
   883	To further validate the effectiveness of our proposed approach,
   884	we conducted a comparative analysis with the latest semi-supervised
   885	cross-modality method DPIS [28] under the same settings in this experiment. As analyzed in Section 2.3.2, the semi-supervised setting
   886	adopted by our approach is different from that of DPIS. Consequently,
   887	we modified our approach to adapt to the experimental settings of
   888	DPIS. Our modified approach is termed as CGSFL-DPIS. Table 3 reports
   889	the experimental results of CGSFL-DPIS and DPIS on SYSU-MM01 and
   890	RegDB datasets under three different settings: 10%, 25%, and 50%
   891	labeled data.
   892	From Table 3, we can make the following observations: (1) In the
   893	Visible to Infrared search mode on the RegDB dataset, our method
   894	achieves improvements of 29.54%, 9.45%, and 2.13% in terms of rank1 accuracy when utilizing labeled data rates of 10%, 25%, and 50%,
   895	respectively. Similarly, for the Infrared-to-Visible setting, our approach
   896	improves the rank-1 accuracy by 29.78%, 11.35%, and 3.98% under
   897	the 10%, 25%, and 50% labeled data rates, respectively. (2) On the
   898	7
   899	
   900	Pattern Recognition 165 (2025) 111669
   901	
   902	X. Zhu et al.
   903	
   904	Table 1
   905	Comparison with the state-of-the-art methods on SYSU-MM01 dataset.
   906	Settings
   907	
   908	Methods
   909	
   910	Venue
   911	
   912	All-search
   913	
   914	Indoor-search
   915	
   916	r1 (%)
   917	
   918	r10 (%)
   919	
   920	r20 (%)
   921	
   922	mAP (%)
   923	
   924	mINP (%)
   925	
   926	r1 (%)
   927	
   928	r10 (%)
   929	
   930	r20 (%)
   931	
   932	mAP (%)
   933	
   934	mINP (%)
   935	
   936	SVI-ReID
   937	
   938	Zero-Padding [11]
   939	HSME [12]
   940	AlignGAN [36]
   941	X-Modal [37]
   942	Hi-CMD [15]
   943	cm-SSFT [38]
   944	AGW [35]
   945	DDAG [39]
   946	VCD+VML [40]
   947	CA [41]
   948	MPANet [16]
   949	MSO [42]
   950	AGM [43]
   951	MCLNet [44]
   952	FMCNet [1]
   953	DART [45]
   954	
   955	ICCV-17
   956	AAAI-19
   957	ICCV-19
   958	AAAI-20
   959	CVPR-20
   960	CVPR-20
   961	TPAMI-22
   962	ECCV-20
   963	CVPR-21
   964	ICCV-21
   965	CVPR-21
   966	MM-21
   967	MM-21
   968	ICCV-21
   969	CVPR-22
   970	CVPR-22
   971	
   972	14.80
   973	20.68
   974	42.40
   975	49.90
   976	34.90
   977	47.7
   978	47.50
   979	54.75
   980	60.02
   981	69.88
   982	70.58
   983	58.70
   984	69.63
   985	65.40
   986	66.34
   987	68.72
   988	
   989	54.12
   990	32.74
   991	85.00
   992	89.80
   993	77.60
   994	–
   995	84.39
   996	90.39
   997	94.18
   998	95.71
   999	96.21
  1000	92.06
  1001	96.27
  1002	93.33
  1003	–
  1004	96.36
  1005	
  1006	71.33
  1007	77.95
  1008	93.70
  1009	96.00
  1010	–
  1011	–
  1012	92.14
  1013	95.81
  1014	98.14
  1015	98.46
  1016	98.80
  1017	–
  1018	98.82
  1019	97.14
  1020	–
  1021	98.96
  1022	
  1023	15.95
  1024	23.12
  1025	40.70
  1026	50.70
  1027	35.90
  1028	54.1
  1029	47.65
  1030	53.02
  1031	58.80
  1032	66.89
  1033	68.24
  1034	56.42
  1035	66.11
  1036	61.98
  1037	62.51
  1038	66.29
  1039	
  1040	–
  1041	–
  1042	–
  1043	–
  1044	–
  1045	–
  1046	35.30
  1047	39.62
  1048	–
  1049	53.61
  1050	–
  1051	–
  1052	52.24
  1053	47.39
  1054	–
  1055	53.26
  1056	
  1057	20.58
  1058	–
  1059	45.90
  1060	–
  1061	–
  1062	–
  1063	54.17
  1064	61.02
  1065	66.05
  1066	76.26
  1067	76.74
  1068	63.09
  1069	74.68
  1070	72.56
  1071	68.15
  1072	72.52
  1073	
  1074	68.38
  1075	–
  1076	87.60
  1077	–
  1078	–
  1079	–
  1080	91.14
  1081	94.06
  1082	96.59
  1083	97.88
  1084	98.21
  1085	96.61
  1086	97.51
  1087	96.98
  1088	–
  1089	97.84
  1090	
  1091	85.79
  1092	–
  1093	94.40
  1094	–
  1095	–
  1096	–
  1097	95.98
  1098	98.41
  1099	99.38
  1100	99.49

codex
Confidence-guided 这篇前 8 页主要给了方法和对 SOTA/DPIS 的对比，关键组件消融可能排在后文，但我会先在前 8 页里检索“ablation/effectiveness”等词，能读到多少就据实分析。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Confidence guided semi-supervised cross-modality person re-identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|Effectiveness|effectiveness|confidence|threshold|CCPG|SCDFL|dynamic|filter|pseudo|DPIS|Table 3|Table 4|Table 5|Table 6\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
39:    39	data. Specifically, the CGSFL framework consists of two key modules, including confidence guided crossmodality pseudo label generation module and semi-supervised cross-modality discriminant feature learning
40:    40	module. The former calculates the confidence of cross-modality matching and clustering, using them to enhance
41:    41	the reliability of generated cross-modality pseudo labels. The latter designs dynamic and static memory
43:    43	present in labeled and pseudo-labeled data, thereby ensuring the discriminative capability of the learned
104:   104	the inherent large inter-modality discrepancy in the unlabeled crossmodality data itself. Furthermore, the process of generating crossmodality data with pseudo-labels usually lacks the guidance of real
105:   105	labels, resulting in the limited reliability of the generated pseudo-labels,
112:   112	Huang et al. [8] developed the Multi-pseudo Regularized Label (MpRL)
113:   113	mechanism to enhance pseudo-label quality by assigning appropriate virtual labels to generated data samples. Furthermore, in order
128:   128	leveraging the generated pseudo-labeled data, our method enables
130:   130	cluster-level pseudo-labeled features with true labeled features.
133:   133	Generation (CCPG) module, which leverages the supervised information to compute the cross-modality matching confidence and the
134:   134	clustering confidence of the current model, and then uses the obtained
135:   135	confidence values to generate more reliable pseudo labels for the next
138:   138	F eature Learning (SCDFL) module, which can improve the discriminability of features gradually by making use of nearest-neighbor image
139:   139	pairs sampled from both the labeled and pseudo-labeled data.
141:   141	RegDB datasets. The results demonstrate the effectiveness of our
195:   195	of generated noisy pseudo-labels. Different from DPIS, our approach
198:   198	In summary, both our approach and DPIS aim to alleviate the
209:   209	and innovatively learned the pseudo relationship between labeled and
216:   216	through labeled data (or pseudo-labeled) and unlabeled data. The
228:   228	methods generally generate pseudo-labels through clustering algorithms without considering the impact of modality differences on
230:   230	cross-modality matching confidence and the clustering confidence.
231:   231	The generated confidences are used to enhance the reliability of
236:   236	In this paper, we propose a confidence-aware pseudo labels
238:   238	consists of two modules: confidence guided cross-modality pseudo label
256:   256	introducing the confidence-guided cross-modality pseudo label
257:   257	generation module and the dynamic memory bank. Specifically,
258:   258	the labeled visible and infrared features obtained from feature extraction module are fed into the adaptive confidence generation block,
259:   259	to compute the cross-modality matching confidence and the clustering
260:   260	confidence. Then, the unlabeled images are clustered, and the pseudolabeled infrared and visible cluster pairs can be generated according to
261:   261	the cross-modality matching confidence and the clustering confidence.
262:   262	Next, the generated pseudo-labeled cross-modality cluster pairs are
263:   263	used to update the dynamic memory bank. In this way, the differences
265:   265	gradually. Finally, the static memory bank and the updated dynamic
274:   274	Dual Pseudo-label Interactive Self-training (DPIS) for this task. DPIS
275:   275	addresses both reliable pseudo-label generation and noisy pseudo-label
276:   276	calibration. Our method differs from DPIS in three key aspects.
278:   278	in DPIS. In DPIS, labels of different rates for each identity on both
284:   284	infrared images. Under the setting of 10% labeled data, DPIS utilizes
291:   291	on the label space between unlabeled and labeled images, whereas DPIS
296:   296	different. DPIS utilizes labeled data to mitigate the negative impact
298:   298	3.2. Confidence guided cross-modality pseudo label generation
342:   342	Finally, the cross-modality matching confidence is used as a dynamic threshold to select more reliable unlabeled cross-modality image
353:   353	To enhance the reliability of generated pseudo-labeled crossmodality image pairs, we propose leveraging a limited number of
355:   355	two metrics to evaluate the reliability of the current model: crossmodality matching confidence and clustering confidence. Details of
358:   358	where (𝑢𝑣𝑎 , 𝑢𝑟𝑏 ) represents a pseudo-labeled cross-modality image pair
359:   359	obtained using the DBSCAN algorithm, with 𝑎 and 𝑏 denoting their respective pseudo-labels. Based on the pseudo labels (𝐶𝑢𝑟 and 𝐶𝑢𝑣 obtained
364:   364	The dynamic threshold proposed in this paper aims to leverage labeled data for selecting high-quality pseudo-label sample pairs. Existing
365:   365	studies typically employ a fixed threshold of 0.5 to determine crossmodality sample category consistency. However, experiments reveal
368:   368	highlighting the limitations of the fixed threshold method. To address
369:   369	this issue, in each training epoch, we dynamically generate thresholds
373:   373	thereby making pseudo-label selection more accurate and reliable.
374:   374	To better investigate the effect of the dynamic threshold, we
377:   377	3.2.1. Cross-modality matching confidence computation
378:   378	The cross-modality matching confidence is designed to assess the
383:   383	cosine similarity matrix between 𝑋𝑟 and 𝑋𝑣 . Finally, the confidence
385:   385	among all truly matching image pairs. The confidence coefficient allows
397:   397	Then, the cross-modality matching confidence (denoted as 𝛥) can be
415:   415	Output: Cross-modality pseudo-labeled cluster pairs
419:   419	5 Compute matching confidence 𝛥 via Eq. (2)
435:   435	Initialize static memory bank 𝑠 and dynamic memory
452:   452	Compute the cross-modality matching confidence 𝛥 via
455:   455	Calculate the clustering confidence via Eq. (4)
457:   457	Update 𝑑 with reliable pseudo-labels (see
479:   479	15 Compute clustering confidence 𝜌 via Eq. (4) on labeled data
488:   488	belonging to distinct classes in . The clustering confidence serves
489:   489	as a dynamic selection ratio, guiding the selection of more reliable
492:   492	filtered out by the dynamic threshold during training. The experimental
493:   493	results demonstrate that, on the SYSU-MM01 dataset, the dynamic
494:   494	threshold filters approximately 20% more unreliable pseudo-labeled
496:   496	threshold, while on the RegDB dataset, it filters about 1% more. This
499:   499	between samples, thus necessitating stricter filtering. Additionally,
500:   500	the number of filtered sample pairs dynamically changes throughout
501:   501	training: as model accuracy improves, the number of filtered samples gradually decreases. This phenomenon indicates that the dynamic
502:   502	threshold can adapt to the model’s learning state, effectively balancing
505:   505	3.2.3. Reliable cross-modality pseudo label generation
506:   506	Based on the cross-modality matching confidence computation module and clustering confidence calculation framework, we propose a twostep strategy to generate high-quality cross-modality pseudo-labeled
509:   509	Based on the labeled cross-modality data, we can compute the crossmodality matching confidence 𝛥 by Eq. (2). Then, for each pair of
512:   512	are filtered and used as the candidate data for generating pseudo labels.
518:   518	 = {1 , 2 , … , 𝑛 }. Here, 𝑖 represents the 𝑖th cross-modality cluster pair. Next, we compute the clustering confidence (denoted by 𝜌)
520:   520	pairs with highest confidence in  are selected as the pseudo-labeled
521:   521	cross-modality data for updating the dynamic memory bank.
523:   523	and cluster-level confidence to improve the reliability of the generated
524:   524	pseudo labels.
526:   526	3.2.2. Clustering confidence computation
528:   528	To mitigate this issue and improve pseudo-label reliability, we propose
529:   529	a cluster filtering mechanism based on a novel clustering confidence
530:   530	metric. This metric enables selective retention of high-confidence
533:   533	we design our confidence measure by adapting the classic silhouette
534:   534	coefficient. The cross-modality clustering confidence is defined as:
543:   543	where 𝑋𝑣 and 𝑋𝑟 represent visible and infrared feature sets respectively. 𝑌𝑚 and 𝑌̂𝑚 denote the ground-truth and pseudo-label sets for
553:   553	3.2.4. Discussion of pseudo label generation strategy
555:   555	concept of generating pseudo-labels via clustering with other unsupervised VI-ReID methods, our proposed confidence guided cross-modality
556:   556	pseudo-label generation module offers a distinct approach to improving pseudo-label reliability. Existing techniques enhance pseudo-label
582:   582	Gaussian mixture model to estimate label confidence scores, [31] enhances cross-modality label credibility through heterogeneous transfer
584:   584	pseudo-label calibration module to rectify label noise. In contrast
585:   585	to these methods, our approach introduces a novel dual-confidence
586:   586	mechanism that simultaneously evaluates cross-modality matching confidence and clustering confidence. This innovative strategy effectively
587:   587	filters out unreliable pseudo-labels, thereby ensuring the generation of
588:   588	more reliable pseudo-labels for subsequent model training.
623:   623	module aims to make full use of the high-quality pseudo-labeled data
629:   629	memory bank and dynamic memory bank. The static memory bank
631:   631	dynamic memory bank consists of the dynamic cluster-center features
632:   632	updated by using the generated pseudo-labeled cross-modality clusters.
633:   633	The class-center losses of static and dynamic memory banks are calculated separately to improve the intra-class compactness and inter-class
634:   634	separability of samples. Details of the static and dynamic memory banks
664:   664	and the infrared cluster-center features in the dynamic memory bank.
690:   690	Dynamic Memory Bank. The dynamic memory bank (denoted as
692:   692	the generated pseudo-labeled cross-modality cluster pairs.
696:   696	𝑢,𝑗 represents a pseudo-labeled cross-modality cluster pair,
698:   698	where 𝑏 ∈ {𝑠𝑡𝑎, 𝑑𝑦𝑛} denotes the type of memory bank (static/dynamic). 𝜑𝑚,𝑏
711:   711	dynamic memory bank for visible modality by integrating features
742:   742	where 𝜓𝑖𝑣 and 𝜓𝑗𝑟 represent the dynamic centers of clusters 𝐶𝑢,𝑖
747:   747	that of the 𝑗th infrared cluster. The dynamic memory bank can guide
798:   798	supervised and unsupervised VI-ReID methods to demonstrate its effectiveness. Tables 1 and 2 report the corresponding experimental results
867:   867	from the GeM pooling layer. At the start of each training epoch, pseudolabels are generated for both modalities using the DBSCAN algorithm.
878:   878	our model for 50 epochs exclusively. Subsequently, in the remaining 50 epochs, we introduce both pseudo-labeled cross-modality data
879:   879	generation module and pseudo-labeled cross-modality data refinement
882:   882	4.3.3. Comparison with DPIS
883:   883	To further validate the effectiveness of our proposed approach,
885:   885	cross-modality method DPIS [28] under the same settings in this experiment. As analyzed in Section 2.3.2, the semi-supervised setting
886:   886	adopted by our approach is different from that of DPIS. Consequently,
888:   888	DPIS. Our modified approach is termed as CGSFL-DPIS. Table 3 reports
889:   889	the experimental results of CGSFL-DPIS and DPIS on SYSU-MM01 and
892:   892	From Table 3, we can make the following observations: (1) In the
1628:  1628	Table 3
1629:  1629	Comparisons with DPIS method on SYSU-MM01 and RegDB in terms of r1 (%) and mAP (%).
1666:  1666	DPIS
1667:  1667	CGSFL-DPIS
1698:  1698	DPIS
1699:  1699	CGSFL-DPIS
1730:  1730	DPIS
1731:  1731	CGSFL-DPIS

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf' - | nl -ba | rg -n \"magnitude|gradient|Fig\\. 1|Table 1|Table 2|regularization|normalization|Condense|Naive|NormFace|ArcFace|overfit|monoton\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8:     8	Condense loss: Exploiting vector magnitude during person Re-identiﬁcation
34:    34	Tangential gradient
36:    36	The magnitudes of features and weights signiﬁcantly aﬀect the gradients during the training process. L2 normalized softmax losses (such as NormFace, CosFace, ArcFace, etc.) and Naive softmax losses both reduce the
37:    37	magnitudes of image features in the training process and achieve good results in face recognition and person
38:    38	re-identiﬁcation tasks, respectively. In this paper, we fully utilize the feature vector magnitudes and propose
39:    39	Condense loss for Re-ID tasks, which replaces the inner production of Naive softmax loss with the negative Euclidean distance. Condense loss generates negative radial gradients when updating weight parameters to push all
40:    40	features compacter. Because the coeﬃcients of tangential gradients (the tangential component of the gradients)
41:    41	are related to feature magnitudes, it ideally provides monotonically decreasing tangential gradients, resulting
43:    43	parameter into Condense loss to enlarge inter-class distances and thus help the model learn more discriminative
45:    45	Re-ID tasks to prove the corresponding conclusion. The experimental results demonstrate that the Condense loss
59:    59	magnitude and accuracy. Fig. 1(a) and (b) show the variation of the
60:    60	average magnitude and accuracy of images in the VGG2FP dataset with
62:    62	show the variation of the average magnitude and retrieval accuracy of
65:    65	of training, the epoch increases continuously, the magnitudes of images
67:    67	the connotation behind the picture is the feature magnitude. Therefore,
68:    68	we claim that classiﬁcation accuracy is related to the feature magnitudes
83:    83	is the feature vector magnitude (for a feature vector 𝑥, its magnitude is
97:    97	Fig. 1. The variation trends of feature magnitude and accuracy during the training process. (a) The average feature magnitude of the images during the training
99:    99	The average feature magnitude of the images during the training process on the Market-1501 dataset with naive softmax loss. (d) Rank-1 change curve during the
103:   103	of pictures of each type of pedestrian, so researchers often use Naive
105:   105	balance ability is relatively poor, so L2 normalization is commonly
107:   107	loss calculation methods are to reduce the length of the characteristic mode. We can therefore claim that in face recognition and ReID tasks, the feature magnitudes should be fully utilized to train the
109:   109	Furthermore, Fig. 1(d) shows the relationship between classiﬁcation accuracy (rank-1) and epochs in the Market-1501 dataset under our baseline method [5]. With the increase of epochs, the overall accuracy is improved. But we can see that in the ﬁrst 30 epochs,
110:   110	there is a decline in rank-1. At the same time, when rank-1 decreases, image feature magnitude increases in the reverse direction.
111:   111	Therefore, we believe that the strong inverse correlation between feature magnitude and classiﬁcation accuracy further validates the importance of the magnitude of feature vectors in image classiﬁcation.
112:   112	However, we also found that the compression process of feature magnitude experiences unstable phases, which can result in the magnitude not being reduced to an appropriate size at certain stages,
114:   114	magnitude of feature vectors in face recognition and person Re-ID
117:   117	updated by calculating the gradient of the loss function. Based on our
118:   118	analysis (details in Section 3.4), the tangential gradients of the Naive
120:   120	tangential gradient is not monotonically decreasing. This means that the
121:   121	Naive Softmax loss could sometimes provide larger gradients in later iterations, leading the network to a local minimum. To avoid this prob-
123:   123	lem, we need a loss function that can provide an ideal monotonically
124:   124	decreasing tangential gradient.
126:   126	Condense loss, which takes the advantage of feature and weight vector magnitudes in Re-ID tasks to give a monotonically decreasing coeﬃcient of a tangential gradient. Diﬀerent from the Naive and L2 normalized softmax losses which are in inner product formulation between
128:   128	Euclidean distance as the formulation. When updating weight parameters, it provides negative radial gradients, hence, the feature vector magnitudes keep reducing during the training process. Furthermore, Condense loss provides a larger tangential gradient at the beginning of the
129:   129	training process, and a smaller tangential gradient at the end, with the
130:   130	change in feature vector magnitudes. After training, both the magnitude
132:   132	class centers decrease, so we name this loss function “Condense Loss”.
133:   133	And we also introduce a margin parameter into Condense loss to enlarge
139:   139	We propose Condense loss with negative Euclidean distance formulation for Re-ID tasks, which utilizes the vector magnitudes to provide
140:   140	an adaptive gradient during the training phase.
141:   141	• We introduce a margin parameter into Condense loss to enlarge the
143:   143	• We give the mathematical analysis to show why Condense loss can
147:   147	face recognition tasks, which demonstrates the eﬀectiveness of Condense loss. When compared with the state-of-the-art methods, Condense loss achieves competitive results.
211:   211	[29] normalized both feature and weight and changed the multiplication angular margin as minus cosine margin. Soon after, ArcFace [4]
223:   223	In this section, we ﬁrst introduce the proposed Condense loss in detail. And a mathematical analysis comparison with other classiﬁcation
224:   224	losses is presented to show the superiority of Condense loss.
225:   225	3.1. Condense loss
263:   263	gradients which are updating directions of class weight 𝑊𝑦𝑖 in Fig. 2(a).
264:   264	The gradient direction of 𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 is the same with vector 𝑥⃗𝑖 and the
267:   267	gradient direction of 𝑓𝑖,𝑘 = 𝑠 cos𝜃𝑘 is vertical to 𝑊
278:   278	Fig. 2. (a) The updated gradient directions of loss functions. Blue, green and red lines represent the gradient directions of 𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 , 𝑓𝑖,𝑘 = 𝑠 cos𝜃𝑘 and the proposed
279:   279	Condense loss, respectively. The light gray dashed line serves as a reference, which includes the update directions under the above three functions, while the dark
280:   280	gray arrow indicates the target direction of parameter updates during model optimization. (b) The geometry illustration of margin in the Condense loss. 𝑚 denotes
285:   285	give it a gradient direction in (⃗
288:   288	3.2. Condense loss in Re-ID
292:   292	The feature and weight vector magnitudes are beneﬁcial to training
293:   293	Re-ID models, so we propose Condense loss which can utilize them to
294:   294	provide monotonically decreasing tangential gradients when training a
296:   296	Since Condense loss does not care what backbone is adopted, we
297:   297	only talk about how to compute Condense loss here. Usually, we input the training pictures into CNNs to learn the features. For instance,
301:   301	the Condense loss, we remove the linear layer and only use the features
307:   307	margin tensor, we can ﬁnally achieve Condense scores tensor, which
309:   309	Further, since the Condense loss is a classiﬁcation loss, it is capable of
319:   319	and we initially deﬁne the Condense loss as:
340:   340	in feature magnitude is quantitatively analyzed in Section 3.4.
347:   347	Condense loss needs to make −(𝑊2 − 𝑉 )2 > −(𝑊1 − 𝑉 )2 . We can make
356:   356	updating the weight 𝑊1 gradient by increasing the distance between 𝑊2
357:   357	and 𝑉 ; hence, the Condense loss makes the features more discriminative
359:   359	Finally, we deﬁne the margin version of Condense loss as:
375:   375	We give a qualitative analysis of softmax function 𝑓 , i.e., Naive softmax function 𝑓𝑛𝑎𝑖𝑣𝑒 , L2 normalized softmax function 𝑓𝑛𝑜𝑟𝑚 and Condense
403:   403	Fig. 3. The ﬂowchart of computing Condense loss in Re-ID tasks. Images are extracted from the feature extraction network. 𝑏𝑠 denotes the batch size, 𝑑𝑖𝑚 denotes
422:   422	𝜃𝑘 ↓). Hence, the naive softmax function sometimes would make the network learn larger feature vector magnitude instead of reducing angles.
424:   424	normalization. In contrast, the Condense function fully utilizes the vector magnitudes. To increase the value of 𝑓𝐶𝑜𝑛𝑑𝑒𝑛𝑠𝑒 , the network would
430:   430	would also decline. Consequently, Condense loss reinforces the network
437:   437	that during the initial stages of training, the Condense loss can provide
438:   438	larger tangential gradients, facilitating faster parameter updates. As the
439:   439	network approaches convergence, the reduction in tangential gradients
440:   440	leads to smaller update magnitudes, thereby enhancing the stability of
443:   443	as it nears convergence. Thus, Condense loss ideally provides monotonically decreasing tangential gradients. Note that, because we usually
444:   444	use mini-batch strategy to optimize the model, the gradients won’t decrease monotonously. However, Condense loss makes the model converge faster.
450:   450	gradients of 𝑊
489:   489	veriﬁcation tasks to demonstrate the eﬀectiveness of Condense loss.
508:   508	ranges from (−1, 1), the magnitude of 𝑥⃗𝑖 would increase or decrease
514:   514	The L2 normalized softmax functions and their gradients can be written as:
539:   539	because all features are normalized, they only have tangential gradients.
556:   556	) As the feature magnitudes decrease, the
558:   558	tangential gradient 𝑇
566:   566	learning rate), the magnitude of 𝑊𝑖 is not monotonous because the 𝜕𝑊𝐶𝐸𝑡
569:   569	may be positive or negative. Both the Naive and L2 softmax losses cannot
570:   570	provide monotonically decreasing tangential gradients.
572:   572	The Condense loss and the gradients of 𝑊
578:   578	Condense loss. ResNet50 is selected as the CNN backbone. The batch
585:   585	themselves. Because the Condense loss converges quickly and overﬁts on
589:   589	in ArcFace [4]. Because MS1MV2 is very large, hence the weight decay
619:   619	we can achieve its radial gradient by:
639:   639	Consequently, its tangential gradient is expressed as:
659:   659	Symmetrically, the radial and tangential gradients of
689:   689	Table 1 reports the experimental results of diﬀerent losses on Market1501 and MSMT17 datasets. When NormFace, CosFace and ArcFace are
703:   703	Table 1
706:   706	[36]. 𝑤𝑜∕𝑚 means the margin parameter in the Condense loss is not considered
712:   712	Table 2
731:   731	Naive Softmax
732:   732	Naive Softmax (𝑤𝑑 5 × 10−3 )
733:   733	NormFace
735:   735	ArcFace
780:   780	Condense (𝑤𝑜∕𝑚)
781:   781	Condense (𝑤∕𝑚)
890:   890	4.4. Comparison with regularization/normalization
891:   891	To verify that Condense loss not only achieves compression of the
892:   892	feature magnitudes but also yields better results, we conduct a comparative experiment between regularization/normalization methods and
893:   893	Condense loss (as shown in Table 2). The experimental results show
896:   896	for L2 normalization are even worse. In contrast, our Condense loss signiﬁcantly outperforms other methods on both datasets, demonstrating
897:   897	its ability to eﬀectively stabilize the model’s update process while compressing feature magnitudes, thereby signiﬁcantly improving accuracy.
902:   902	of CosFace and ArcFace is diﬃcult to decide on diﬀerent datasets.
903:   903	In contrast, the Condense loss is easy to train and is more eﬀective
907:   907	MSMT17 datasets. When we utilize the margin parameter, the performance is further improved. For example, the Condense loss gains rank1/mAP 2.2 %/2.2 % on MSMT17. The improvement on Market-1501 is
911:   911	slight. Hence, the improvement of the Condense loss is not led by a
912:   912	higher weight decay value. The results show that the Condense loss
916:   916	on Market-1501 in the appendix. During the ﬁrst 25 epochs, the Condense loss has a lower performance than the naive softmax loss. However, it converges faster and achieves its best result at 65 epoch, while
917:   917	the naive softmax loss achieves its best result at 85 epoch. This phenomenon demonstrates that our Condense loss provides more reasonable gradients to help the model converge faster and needs fewer epochs
921:   921	The margin parameter in Condense loss is designed to enhance the
924:   924	of margin-based softmax functions such as ArcFace and CosFace, this
947:   947	indicates that the margin parameter in the Condense loss is not considered, while “w/m” indicates that the margin parameter is used in the
960:   960	early training process, compared with Naive softmax loss, our Condense
965:   965	faster and ﬁnds the best value under the current gradient update earlier.
967:   967	Naive Softmax
968:   968	Condense (𝑤𝑜∕𝑚)
969:   969	Condense (𝑤∕𝑚)
1006:  1006	We compare the computational eﬃciency of the Condense loss and
1007:  1007	Naive Softmax on the Market-1501 dataset. Speciﬁcally, we conduct
1009:  1009	the Naive Softmax, the classiﬁer consists of a single linear layer (including matrix multiplication and bias addition). In contrast, our method
1011:  1011	for Condense with margin). The experimental results indicate that, compared to Naive Softmax, our method does not increase computational
1023:  1023	Condense loss can be combined with the triplet loss to further improve
1024:  1024	Re-ID performance. The experimental results are shown in Table 3. Compared with the Naive softmax + triplet loss, Condense loss can achieve
1027:  1027	triplet loss. After considering the margin parameter, the Condense loss
1029:  1029	1.6 %/1.0 %. The results demonstrate that Condense loss and its margin parameter are still eﬀective when combined with the triplet loss.
1035:  1035	Condense loss is not considered and 𝑤∕𝑚 means the margin parameter is used
1039:  1039	Naive Softmax
1041:  1041	Condense
1044:  1044	Condense (𝑤∕𝑚)
1049:  1049	We compare the Condense loss with the latest Re-ID methods, which
1053:  1053	when compared with the Condense loss (without triplet loss), SphereReID and Circle loss respectively performed rank-1/mAP 0.3 %/2.5 %,
1054:  1054	3.9 %/7.7 % lower on Market-1501. Meanwhile, the triplet loss can further improve the performance of Condense loss. Hence, Condense loss
1057:  1057	attention modules in Re-ID networks which means their network architectures were more complex than Condense loss which only use the
1138:  1138	for replacing the Naive softmax loss with our Condense loss.
1231:  1231	APNET+Condense
1247:  1247	MSINet+Condense
1261:  1261	Condense
1262:  1262	Condense+Triplet(ﬁne-tune)
1278:  1278	vit-stride+Condense
1294:  1294	deit-stride+Condense
1322:  1322	a complex baseline). RGA-SC performed slightly better than Condense
1336:  1336	since triplet loss is widely applied in almost all methods, Condense loss
1343:  1343	Previously, we have applied Condense loss with triplet loss in the
1344:  1344	simple framework of CNN to conﬁrm the competitiveness of our Condense loss. Further, we replace the Naive softmax loss with Condense
1346:  1346	using ResNet as the baseline to extract features. Condense loss achieves
1348:  1348	Naive Softmax loss. However, there is mAP 0.9 % decrease on Market1501. This is because the original accuracy is already suﬃciently high,
1351:  1351	1501 dataset is relatively easy, and using Condense loss may lead to the
1355:  1355	observed that in our replication, Condense loss has improved across all
1356:  1356	metrics. Under the framework of Transformer, whether baseline is vitbase or deit-base, Condense loss achieves the best accuracy of any rank1/mAP on the Market-1501 and MSMT17 datasets. Both the experimental data of APNET, MSINet and Trans-ReID [12] are reproduced by us
1359:  1359	ansReID). Therefore, only replacing the Naive softmax loss with Condense loss has achieved certain improvements. As a result, Condense loss
1362:  1362	We also performe experiments on face veriﬁcation tasks using Naive
1363:  1363	softmax, CosFace and ArcFace for comparison, to demonstrate the eﬀectiveness of Condense loss. The oﬃcial protocol of face veriﬁcation tasks
1365:  1365	training datasets and training strategies kept the same as ArcFace, and
1370:  1370	The face veriﬁcation results are shown in Table 6. The oﬃcial ArcFace implementation uses MxNet, and the results reported in [4] are

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf' - | nl -ba | rg -n \"Ablation|ablation|Effectiveness|effectiveness|t-SNE|visual|Visualization|DFA|BFG|WCL|coarse|fine|Clean Eval|Corrupt Eval|Table VI|Table VII|alignment\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
14:    14	pedestrian characteristics and cause semantic misalignment
16:    16	a coarse-to-fine semantic alignment framework that learns
18:    18	from the perspective of multi-modal feature alignment. In this
23:    23	clean samples in a knowledge distillation manner in the coarse
24:    24	alignment stage, i.e., a teacher network distils identity-related
27:    27	samples. To avoid information loss of the strict alignment,
28:    28	we propose to integrate a Bridge Feature Generation (BFG)
30:    30	among modalities in the fine alignment stage. This enables
31:    31	seamless alignment of the same identity between corrupted and
33:    33	To evaluate the effectiveness of the proposed method, extensive
39:    39	corrupt scenes, multi-modal alignment, intermediate feature
100:   100	We further visualize the distribution of pedestrian images with
101:   101	different corruptions using t-SNE [6] in Fig. 1(b). From the
111:   111	Inspired by the insights mentioned above, this paper introduces a pedestrian semantic alignment framework for CI-ReID
123:   123	shows image degradation under different levels of corruptions; (b) the distribution of corrupted samples using t-SNE [6].
125:   125	samples in the semantic space. To ease the feature alignment, we adopt a coarse-to-fine feature alignment strategy.
126:   126	Specifically, a Dual-stream Feature Alignment (DFA) model
127:   127	is proposed to achieve coarse alignment, which includes a
142:   142	Recent studies [8] suggests the strict modality alignment
145:   145	the lower bound of the approximate alignment, more useful information can be retained, and better alignment can
149:   149	relax the regularization of strict alignment and construct the
151:   151	Feature Generation (BFG) module to generate the intermediate features between modalities. By filling the intermediate
153:   153	semantic structure is constructed. The BFG extends the range
154:   154	of corrupt modality in the semantic space and achieving fine
155:   155	alignment. It is worth noting that we adapt the re-parameter
156:   156	trick into BFG to generate intermediate features in stochastic
159:   159	the BFG expands intra-person cluster, which enables the
164:   164	formulate CI-ReID as a modal alignment problem, aiming
167:   167	we introduce a novel modality alignment framework for
169:   169	• We adopt a coarse-to-fine feature alignment strategy to
171:   171	feature alignment module to achieve coarse alignment
175:   175	modalities for fine alignment.
189:   189	of cross-camera tracking until Gheissari et al. [14] first defined
209:   209	hierarchical clustering and refinement method to address the
263:   263	alignment between the classes. Zhang et al. [34] proposed a
265:   265	low-quality image recognition through cross-quality knowledge transfer, achieving feature alignment between images of
274:   274	The purpose of cross-modal alignment is to explore the correlations between sub-elements of multi-modal data. Modality
275:   275	alignment is widely used in multi-modal tasks, and the main
276:   276	methods include explicit alignment [35] and implicit alignment [36], [37]. If a model’s optimization objective is to
277:   277	maximize the alignment of sub-elements of multi-modal data,
278:   278	it is referred to as explicit alignment. Explicit alignment
282:   282	needs to learn similarity measures and alignment methods
285:   285	If the final optimization objective of the model is not an alignment task, and the alignment process is only an intermediate
286:   286	(or implicit) step, it is referred to as implicit alignment. Some
287:   287	implicit alignment methods [42] require manually constructing
288:   288	mappings between modalities. Other alignment methods [43]
294:   294	inspired us to achieve fine alignment by filling bridge features
302:   302	Fig. 2. (Viewed in color) An illustration of the proposed Coarse-to-Fine Feature Alignment Transformer. The network structure consists of one DFA module
303:   303	(i.e., siamese teacher-student feature extraction networks) for coarse feature alignment and one BFG module for fine feature alignment. The feature extraction
304:   304	network of the DFA module uses a traditional Vision Transformer (ViT) [45] structure, with the teacher model parameters fixed during training. The BFG
308:   308	on top of the coarse alignment using siamese teacher-student
312:   312	dataset DCor = {X iCor |i = 1, 2, . . . , N } is defined as follows:
320:   320	Feature Alignment (DFA) module, i.e., the siamese
322:   322	Feature Generation (BFG) module. Given a clean/uncorrupted ReID dataset D = {X i |i = 1, 2, . . . , N }, where
330:   330	training, our CFAT takes a coarse-to-fine alignment strategy
331:   331	in the latent semantic space. In the coarse alignment phase,
334:   334	In the fine alignment phase, the BFG is further introduced to
339:   339	of forming strict feature alignment in the latent space, which
353:   353	corruption level to generate X iCor . In our work, we defined
363:   363	To mitigate the semantic misalignment in the feature space
365:   365	Feature Alignment (DFA) module for CI-ReID. The proposed DFA is built on the concept of knowledge distillation,
372:   372	Such alignment enables SFE to restore rich pedestrian semantics though severe corruptions occur, ultimately generating
386:   386	which is defined as
394:   394	pre-trained parameters of TFE while optimizing SFE on corrupted samples. To achieve the alignment, contrastive loss [46]
396:   396	their clean counterpart consistent, which is defined as
426:   426	Contrastive Loss (WCL) instead of L Con . Inspired by [19],
427:   427	our WCL aims to adaptively adjust penalties for challenging
429:   429	Thus, the improved WCL can be defined as
437:   437	LWCL = −
451:   451	the similarity between samples, WCL ensures a larger penalty
453:   453	alignment results.
456:   456	of the SFE, defined as
493:   493	defined as
511:   511	As shown in Fig. 3, the coarse alignment mentioned
513:   513	studies [8], [49] indicate that strict alignment achieved by
515:   515	downstream tasks. In another aspect, the perfect alignment
518:   518	strict alignment and construct the latent structure between
522:   522	Feature Generation (BFG) module. As shown in Fig. 3, this
531:   531	alignment process.
533:   533	motion that defines a random path (referred to as a bridge)
535:   535	features extracted by the DFA module). The structure of the
538:   538	Gaussian distribution, defined as
550:   550	Eq. 7, a Brownian bridge path [8] is defined as
580:   580	Fig. 3. Comparison of two different alignment methods. We categorize modal features into common features (i.e., corruption-invariant features) and modal
581:   581	unique features (i.e., clean-image features susceptible to corruption and corrupted-image features introduced by corruption). Using only DFA for coarse
582:   582	alignment, the contrastive loss will encourage the features in the corrupted modality to match those in the clean modality and lead to the loss of modal unique
583:   583	features. DFA+BFG expands the bound of the corrupted modality and achieves seamless modal alignment by filling intermediate features. This smoother
584:   584	alignment method preserves more discriminative parts of modal unique features for identifying pedestrian identities.
603:   603	Z B B , defined as
625:   625	generation via Brownian bridge is defined as
639:   639	By optimizing Eq. 12, the BFG module generates intermediate
641:   641	corrupted variants. Such manner achieves alignment by constructing semantic structure rather than enforcing alignment,
644:   644	We adopt a coarse-to-fine alignment strategy for model
645:   645	optimization. In the coarse alignment stage, we enforce different modalities of corruptions align to their corresponding
647:   647	In the fine alignment stage, a Brownian feature generation
658:   658	information loss in the coarse alignment stage. To improve
661:   661	loss in the coarse alignment stage are calculated solely from
662:   662	corrupted samples, while those in the fine alignment stage,
664:   664	the calculation. Thus, the overall optimization loss is defined
678:   678	coarse stage, 0 otherwise. We summarize the proposed method
686:   686	3: Pretrain DFA model and freeze TFE parameters;
691:   691	// Stage 1. Coarse alignment.
694:   694	// Stage 2. Fine alignment.
696:   696	Get Z B B and Z B F G from BFG;
705:   705	three benchmark datasets to evaluate the effectiveness of the
735:   735	different evaluation configurations, i.e., Clean Eval, Corrupt
738:   738	Table II, Clean Eval refers to the configuration that both
740:   740	In contrast, Corrupt Eval indicates that both the query set
753:   753	Penalty (mINP) [19] to evaluate the effectiveness of the
796:   796	alignment across different corruptions, we adopted a coarseto-fine strategy to optimize the proposed model. In the coarse
797:   797	alignment stage, we optimized the SFE sub-network under
799:   799	In the fine alignment stage, the SFE is further fine-tuned to
833:   833	and ReID methods for challenging scenarios with a large margin. This demonstrates effectiveness of our proposed method
836:   836	decrease in clean scenes compared to the baseline. We suppose it is because the alignment among different corruption
838:   838	Such alignment slightly sacrifices the model preference to

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|Effectiveness|effectiveness|visual|Visualization|Table V|CSSA1|CARE|SGFA|N_p|semantic|confidence|alignment\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
10:    10	associating the video sequences of the identical person across multiple cameras. The ubiquitous appearance misalignment poses a
11:    11	major obstacle for video person Re-ID. Existing alignment-based
12:    12	methods generally rely on off-the-shelf semantic parsing models
18:    18	representations and semantic-aligned global-level representations
22:    22	semantic alignment without extra human topology semantics.
23:    23	Second, a Context-Aided Region Enhancement (CARE) module is
24:    24	proposed to efficiently highlight macro-visual patterns associated
27:    27	a Semantic-Aware Global Feature Alignment (SGFA) method
28:    28	for generating pair-wise semantic-aligned global representations,
31:    31	benchmarks indicate the superiority and effectiveness of the
33:    33	Index Terms— Video-based person re-identification, semanticaware feature alignment, context-aided region enhancement,
85:    85	misalignment problem, which is frequently caused by inaccurate human detection, partial occlusions, camera viewpoint
87:    87	misalignment issue, previous solutions are generally divided
102:   102	achieve precise semantic alignment. The second group relies
104:   104	parsing [9], to locate human parts for semantic and spatial
105:   105	alignment. This strategy results in a significant computational
113:   113	components and achieve pixel-level semantic alignment without prior human topology information. CSSA jointly learns
114:   114	pixel-level part-aligned representations and semantic-aligned
120:   120	than in the background [10], [11]. To ensure semantic consistency, we select a fixed number of video sequences for
123:   123	We iteratively generate pseudo-labels to supervise the semantic
124:   124	estimation. The confidence maps of the classifier are used to
127:   127	propose the Context-Aided Region Enhancement (CARE)
132:   132	CARE integrates global spatio-temporal information into a
135:   135	to highlight macro-visual patterns associated with the target
137:   137	In terms of global-level representation learning, we propose the Semantic-Aware Global Feature Alignment (SGFA)
138:   138	scheme for generating pair-wise semantic-aligned global features. Our key observation is that the contribution of the
139:   139	same semantic part is inconsistent across various video clips.
140:   140	Accordingly, SGFA calculates the adjustment weight based
141:   141	on the confidence map of each semantic part and obtains the
146:   146	• We propose the CSSA framework that addresses appearance misalignment in video person Re-ID while learning
147:   147	pixel-level part-aligned representations and semanticaligned global-level representations in an end-to-end
149:   149	achieve pixel-level semantic alignment without prior
151:   151	• We propose a novel Context-Aided Region Enhancement (CARE) module, which can efficiently highlight
153:   153	macro-visual patterns associated with the target pedestrian and suppress noise caused by factors like background clutters and occlusions.
154:   154	• A Semantic-Aware Global Feature Alignment (SGFA)
155:   155	method is proposed for generating pair-wise semanticaligned global representations, which play an essential
213:   213	semantic information of local regions. In [29], GPNet leverages a graph model to mine multi-granularity knowledge.
220:   220	diverse attention models to learn fine-grained visual information and long-range dependencies. In [12], SINet constructs the
224:   224	these methods are sensitive to appearance misalignments that
228:   228	feature representations, making it robust against misalignment [1]. According to the method of obtaining body part
242:   242	partition is helpless against background noise and semantic
243:   243	misalignment.
255:   255	exploit additional semantics in terms of posture to identify object parts, which provides aligned part features.
263:   263	to deform the original image input into 24 semantically
265:   265	pixel-wise features weighted by a series of confidence maps,
266:   266	Guo et al. [39] use the state-of-the-art body parsing framework CE2P for predicting the semantic human part masks.
275:   275	Re-ID. Second, the off-the-shelf models are vulnerable to various Re-ID scenarios and cannot correct semantic estimation
280:   280	and semantic-aligned global-level representations in an endto-end manner. In this part, we first provide a description of
282:   282	core processes, i.e., CARE and SGFA, are clarified. Finally,
297:   297	Then, X q is fed into the CARE module for obtaining the
301:   301	feature maps F q = {F0 , F1 , . . . , FT −1 }. CARE can efficiently highlight macro-visual patterns associated with the
304:   304	We represent the homogeneous semantic part of a video clip
306:   306	pixel-level semantic estimation, a part classifier is designed
309:   309	semantic categories in total. The classifier needs to generate
310:   310	N p + 1 confidence maps of separate categories for each video
311:   311	snippet. Then, the representation for the n-th semantic part can
319:   319	Fig. 2. Overview of the proposed context-aided semantic-aware self-alignment (CSSA) framework. CSSA iteratively performs part classifier training and
320:   320	semantic-aligned representation learning during the training phase. The context-aided region enhancement (CARE) module is designed to efficiently highlight
321:   321	macro-visual patterns associated with the target pedestrian and suppress noise caused by factors like background clutters and occlusions. The semantic-aware
322:   322	global feature alignment (SGFA) method is proposed for generating pair-wise semantic-aligned global features.
336:   336	n-th confidence map. For the clear exposition, we leverage
338:   338	Mn (t, h, w) to represent the pixel confidence (t, h, w) of
339:   339	the semantic category n. Ideally, for the background and
347:   347	operation). Next, the generated confidence maps are used
348:   348	to guide SGFA for obtaining semantic-aligned global-level
353:   353	for the semantic part classifier training. Specifically, the feature
359:   359	labels {1, . . . , N p } to the semantic parts according to their
402:   402	Fig. 3. Model architecture of CSSA in the testing stage. The final distance is calculated using a part-to-part matching strategy. The identical semantic part
406:   406	We propose CARE to highlight the responses of relevant
417:   417	The architecture of CARE is illustrated in the upper part
452:   452	With pixel-level semantic estimation, our proposed framework combines the whole body feature and local part
457:   457	errors, which results in the identical semantic parts contributing differently to the global-level feature representations
459:   459	Semantic-Aware Global Feature Alignment (SGFA) method to
460:   460	generate pair-wise semantic-aligned global features with the
461:   461	help of the confidence maps.
462:   462	The lower part of Fig. 2 illustrates the pipeline of SGFA. For
464:   464	CARE module to get their feature maps F q and F g . In order
465:   465	to quantify the contribution of each semantic part, we design
466:   466	a semantic score according to the results of part estimation:
474:   474	information. Typically, the features with analogous semantics
520:   520	where sn , sn indicate the n-th part semantic scores for
523:   523	F q and F g . Jn , Jn denote the point sets with confidence
527:   527	step, we leverage the confidence maps to generate the mask
528:   528	{U1 , . . . , U N p } about each semantic part. Un (t, h, w) is set
538:   538	the final pair-wise semantic-aligned global features can be
583:   583	denoted as L pt and Lgl , which are enforced on concatenated part features and pair-wise semantic-aligned global
651:   651	to construct the semantic part classifier. Each video frame
661:   661	1/10 every 40 epochs. The semantic part number, i.e., N p ,
712:   712	learns the pixel-level semantics without any prior human topology information. Notably, our proposed method outperforms
720:   720	semantic-aligned global-level representations, which are not
721:   721	fully exploited in existing studies. Moreover, CARE integrates
727:   727	D. Ablation Studies
728:   728	In this section, we execute a series of ablation studies on
730:   730	The experimental results in Table V indicate that the proposed
732:   732	1) Effectiveness of the Semantic-Aligned Part-Level Representation Learning: In Table V, ‘Baseline’ denotes a model
735:   735	the variant as the degraded CSSA without CARE and SGFA.
736:   736	We represent the variant as ‘CSSA1 ’ for simplification.
737:   737	As shown in Table V, CSSA1 significantly increases the
739:   739	CSSA1 outperforms the mAP of ‘Baseline’ by 9.5%, 9.8%,
742:   742	the semantic-aligned part-level learning framework considerably benefits the video Re-ID task.
743:   743	2) Effectiveness of the CARE Module: Compared with
744:   744	CSSA1 , the CARE module brings consistent performance
746:   746	considering only the CARE over the CSSA1 is 5.6% for mAP
747:   747	and 7.1% for Rank1 accuracy. Similarly, CARE brings 4.4%
748:   748	mAP and 5.5% mAP gains over the CSSA1 on MARS and
750:   750	to highlight macro-visual patterns associated with the target
753:   753	3) Effectiveness of the SGFA Method: For ‘CSSA3 ’
754:   754	in Table V, we remove the CARE module from the proposed
755:   755	full model (‘CSSA4 ’). Clearly, adding SGFA can boost the
758:   758	semantic-aligned global features for each video sequence.
759:   759	4) Combination of CARE and SGFA: Note that the ‘CSSA1 ’
760:   760	model using only CARE achieves comparable or even better
761:   761	improvement than when using only SGFA. Moreover, when
762:   762	CARE and SGFA are combined into a CSSA model, extra
763:   763	advantages can be realized. It indicates that they are complementary to each other. Furthermore, we visualize several
767:   767	Conversely, with the help of CARE and SGFA, CSSA4 can
771:   771	image-based person Re-ID framework. As shown in Table VI,
796:   796	A BLATION S TUDIES ON LS-VID, MARS, AND D UKE -V IDEO . CSSA1−4 D ENOTE F OUR VARIANTS OF THE P ROPOSED CSSA F RAMEWORK .
801:   801	the size of the intermediate features in CARE. Fig. 5(a)
810:   810	semantic part granularity. Quantitative ablation experiments

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf' - | nl -ba | sed -n '727,770p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   727	D. Ablation Studies
   728	In this section, we execute a series of ablation studies on
   729	person Re-ID datasets to verify the effect of each component.
   730	The experimental results in Table V indicate that the proposed
   731	CSSA is robust and effective in various experimental settings.
   732	1) Effectiveness of the Semantic-Aligned Part-Level Representation Learning: In Table V, ‘Baseline’ denotes a model
   733	that utilizes uniform partitioning on the conv-layer to learn
   734	part-level representations. For fair comparison, we construct
   735	the variant as the degraded CSSA without CARE and SGFA.
   736	We represent the variant as ‘CSSA1 ’ for simplification.
   737	As shown in Table V, CSSA1 significantly increases the
   738	performance compared to the baseline model. Particularly,
   739	CSSA1 outperforms the mAP of ‘Baseline’ by 9.5%, 9.8%,
   740	and 4.6%, as well as Rank1 accuracy by 3.6%, 4.3%, and 4.3%
   741	on the three databases, respectively. The result validates that
   742	the semantic-aligned part-level learning framework considerably benefits the video Re-ID task.
   743	2) Effectiveness of the CARE Module: Compared with
   744	CSSA1 , the CARE module brings consistent performance
   745	promotion. On LS-VID, the margin of improvement while
   746	considering only the CARE over the CSSA1 is 5.6% for mAP
   747	and 7.1% for Rank1 accuracy. Similarly, CARE brings 4.4%
   748	mAP and 5.5% mAP gains over the CSSA1 on MARS and
   749	DukeMTMC-VideoReID. These results prove that it is vital
   750	to highlight macro-visual patterns associated with the target
   751	person and suppress noise caused by factors like background
   752	clutters and occlusions.
   753	3) Effectiveness of the SGFA Method: For ‘CSSA3 ’
   754	in Table V, we remove the CARE module from the proposed
   755	full model (‘CSSA4 ’). Clearly, adding SGFA can boost the
   756	Rank1 accuracy from 78.6% to 83.7% on LS-VID. The
   757	improvement demonstrates that the value of learning pair-wise
   758	semantic-aligned global features for each video sequence.
   759	4) Combination of CARE and SGFA: Note that the ‘CSSA1 ’
   760	model using only CARE achieves comparable or even better
   761	improvement than when using only SGFA. Moreover, when
   762	CARE and SGFA are combined into a CSSA model, extra
   763	advantages can be realized. It indicates that they are complementary to each other. Furthermore, we visualize several
   764	retrieval results for the baseline and CSSA4 . As demonstrated
   765	in Fig. 4, it is hard for the baseline to identify target
   766	objects that are affected by detection errors and occlusions.
   767	Conversely, with the help of CARE and SGFA, CSSA4 can
   768	distinguish them successfully.
   769	5) Comparison With Image-Based Person Re-ID Methods:
   770	When T is set to 1, CSSA can be easily degenerated into an

codex
现在材料足够写逐篇拆解了。还剩一个小补充：semi-supervised 那篇的 DPIS 对比表在前 8 页靠后，我补读具体表头和结论，防止把“证据闭环”说虚。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Confidence guided semi-supervised cross-modality person re-identification.pdf' - | nl -ba | sed -n '1628,1755p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1628	Table 3
  1629	Comparisons with DPIS method on SYSU-MM01 and RegDB in terms of r1 (%) and mAP (%).
  1630	Rate
  1631	
  1632	Methods
  1633	
  1634	Venue
  1635	
  1636	SYSU-MM01
  1637	
  1638	RegDB
  1639	
  1640	All-search
  1641	
  1642	Indoor-search
  1643	
  1644	Visible2infrared
  1645	
  1646	Infrared2visible
  1647	
  1648	r1 (%)
  1649	
  1650	mAP (%)
  1651	
  1652	r1 (%)
  1653	
  1654	mAP (%)
  1655	
  1656	r1 (%)
  1657	
  1658	mAP (%)
  1659	
  1660	r1 (%)
  1661	
  1662	mAP (%)
  1663	
  1664	10%
  1665	
  1666	DPIS
  1667	CGSFL-DPIS
  1668	
  1669	ICCV-2023
  1670	–
  1671	
  1672	57.70
  1673	53.17
  1674	
  1675	55.00
  1676	48.50
  1677	
  1678	62.86
  1679	60.17
  1680	
  1681	69.84
  1682	65.05
  1683	
  1684	54.95
  1685	84.49
  1686	
  1687	50.79
  1688	75.33
  1689	
  1690	55.05
  1691	84.83
  1692	
  1693	49.64
  1694	76.26
  1695	
  1696	25%
  1697	
  1698	DPIS
  1699	CGSFL-DPIS
  1700	
  1701	ICCV-2023
  1702	–
  1703	
  1704	65.20
  1705	58.51
  1706	
  1707	61.75
  1708	54.71
  1709	
  1710	70.50
  1711	63.07
  1712	
  1713	75.55
  1714	68.37
  1715	
  1716	77.28
  1717	86.73
  1718	
  1719	67.74
  1720	78.01
  1721	
  1722	75.34
  1723	86.69
  1724	
  1725	66.27
  1726	79.34
  1727	
  1728	50%
  1729	
  1730	DPIS
  1731	CGSFL-DPIS
  1732	
  1733	ICCV-2023
  1734	–
  1735	
  1736	67.89
  1737	62.78
  1738	
  1739	64.16
  1740	56.54
  1741	
  1742	71.78
  1743	64.78
  1744	
  1745	76.60
  1746	70.05
  1747	
  1748	85.58
  1749	87.71
  1750	
  1751	76.73
  1752	79.29
  1753	
  1754	84.41
  1755	88.39

web search: 
web search: "Condense loss" "person Re-identification"
web search: 
web search: "Corruption-Invariant Person Re-Identification" "Coarse-to-Fine Feature Alignment"
web search: 
web search: Corruption Invariant Person Re Identification Coarse to Fine Feature Alignment
codex
我按 `pdftotext -f 1 -l 8 '文件名' -` 读了你列出的论文。你标题里写“7”，但实际清单只有 6 篇；下面按这 6 篇完整拆。

**1. Condense loss**

1. 触发观察：作者先观察训练过程中“特征向量范数”和准确率的关系。随着训练推进，特征范数下降，准确率上升；Market-1501 前 30 个 epoch 里 rank-1 下降时，特征范数反而上升。这个观察把一个很普通的 loss 设计问题，变成了“训练动态里范数压缩不稳定”的问题。

2. 重定义动作：大家通常把 softmax、NormFace、ArcFace 这类 loss 理解为“角度间隔”或“分类边界”问题，作者改说：ReID 训练里被忽略的是 feature magnitude，它不仅不是要被归一化掉的副产品，反而控制梯度行为。关键词是 vector magnitude、negative radial gradient、tangential gradient、monotonically decreasing。

3. 机制怎么长出来：如果目标是稳定压缩范数，并让后期梯度变小，那么内积分类器就不够自然，因为内积既可增大范数也可减小角度。于是作者把分类分数从 `W^T x` 改成负欧氏距离 `-(W-x)^2`，让梯度方向直接指向类中心，同时通过范数影响切向梯度。margin 只是进一步拉大类间距离。机制和重定义绑定较紧。

4. 证据闭环：闭环不是只靠涨点，而是三层证据：Fig.1 显示范数与准确率的反向趋势；Section 3.4 做 Naive/L2/Condense 的梯度分解；Table 1 比 NormFace、CosFace、ArcFace、Naive Softmax，Table 2 再排除“只是 weight decay 或 normalization 起作用”的解释。这个证据链是在证明“范数和梯度动态值得被建模”。

5. reviewer 为什么买账：卖点主要是视角，不是复杂模块。它把“ReID loss 还能怎么改”从常见的角度间隔，转成训练过程中的范数利用和梯度单调性。缺点是“图像内涵就是范数”这个说法有点过，但数学推导和对照实验把它撑住了。

**2. Confidence guided semi-supervised cross-modality person re-identification**

1. 触发观察：监督 VI-ReID 标注贵，纯无监督 VI-ReID 又和监督方法差距大。关键失败点不是模型不够强，而是跨模态伪标签不可靠。作者还指出固定相似度阈值 0.5 会被背景、姿态等因素误导，不同身份也可能超过阈值。

2. 重定义动作：它把问题从“怎么做更强的跨模态特征学习”改成“少量真标签如何作为当前模型可靠性的标尺，指导无标签跨模态伪标签生成”。关键词是 confidence guided、cross-modality matching confidence、clustering confidence、semi-supervised cross-modality discriminant feature learning。

3. 机制怎么长出来：既然核心是可靠性，就自然要有两个置信度：图像对层面的跨模态匹配置信度，用有标注可见光/红外真匹配对动态估计阈值；簇层面的 clustering confidence，用类似轮廓系数的指标决定保留多少跨模态簇对。之后用静态 memory bank 保持模态内结构，用动态 memory bank 拉近可靠跨模态簇。

4. 证据闭环：前 8 页里最像闭环的是动态阈值分析：SYSU-MM01 上比固定阈值多过滤约 20% 不可靠伪标签对，RegDB 约多过滤 1%，且随着模型变好，被过滤数量下降。这证明“阈值应随模型状态变化”这个重定义。另一个证据是和 DPIS 在 10%、25%、50% 标签率下对齐设置比较，说明不是只靠更宽松设定赢。

5. reviewer 为什么买账：这篇卖的是“半监督跨模态”这个实际设定和“置信度校准伪标签”的视角。模块本身偏工程组合，但每个部件都服务于伪标签可靠性，所以叙事成立。它的投稿价值来自实际问题设定，而不是某个单独模块多新。

**3. Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification**

1. 触发观察：换衣 ReID 里，普通模型过度依赖衣服；很多方法靠姿态、轮廓、衣服标签等额外信息。作者的具体观察很小：average pooling 更保留整体内容，max pooling 更抓显著区域。这个观察被拿来解释 RGB 图像内部其实已有可用语义。

2. 重定义动作：大家以为换衣 ReID 需要“去衣服”或借助外部模态，作者改说：不需要额外模态，RGB 特征里已经有 identity-related content semantics 和 identity-related salient semantics，只是过去没有把两者分开挖掘和协同。关键词是 content semantics、salient semantics、abundant semantics、without auxiliaries。

3. 机制怎么长出来：average pooling 对应 content semantics，max pooling 对应 salient semantics，这一步很自然。SMR 用全局和局部语义挖掘，再用身份监督保证语义和身份相关，用 refinement 做通道重标定。CSSC 的 cross-parallel collaboration 让 content 先帮助 salient，再让 salient 反过来帮助 content。前半段机制自然，cross-parallel 有一定架构设计味道，但能接住叙事。

4. 证据闭环：Table III 显示去掉 SMR 后 PRCC cloth-changing 从 65.5/63.0 掉到 57.8/55.8，local mining 和 refinement 各自也有贡献。Table IV 证明只用 content、只用 salient、简单串联或简单并联都不如 cross-parallel collaboration。Fig.3 检索可视化说明 baseline 被相似颜色和衣服纹理误导，而 CSSC 能靠身体、局部形状、鞋裤等语义找回。

5. reviewer 为什么买账：这篇卖的是视角加简洁性。创新不在 average/max pooling 本身，而在把两个老操作命名成两类身份语义，并用换衣 ReID 的痛点证明它们互补。对 B 类稿很有参考价值：小观察，只要能重定义成清晰语义，再配干净消融，也能成立。

**4. Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification**

1. 触发观察：视频 ReID 的全局特征会混入遮挡、检测框错误、背景；局部 part 特征又有对齐问题。硬条带太粗，外部 pose/parsing 计算重，还会把背包、手提包这类有身份价值的上下文当背景。

2. 重定义动作：它把视频 ReID 从“时间聚合问题”改成“无需外部人体拓扑的语义自对齐问题”。作者强调的不是再做 attention，而是用视频自身上下文学习 pixel-level part-aligned representation 和 semantic-aligned global representation。关键词是 self-alignment、context-aided、semantic-aware、without prior human topology。

3. 机制怎么长出来：先用特征响应强弱分前景/背景，再对同一身份若干视频的前景像素聚类，生成部件伪标签，训练 part classifier。CARE 用整段视频的全局时空上下文增强目标区域、压制噪声。SGFA 根据语义部件置信图给全局特征加权，因为同一部件在不同视频中贡献不同。每个机制都对应一个前面的观察，绑定很紧。

4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。

5. reviewer 为什么买账：这篇卖的是完整视角：外部语义不可靠，视频内部可以自生成语义对齐。模块复杂，但每个模块都解释一个具体失败点，所以不像堆模块。novelty 主要来自“self-alignment without human topology”的问题重定义。

**5. Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment**

1. 触发观察：现实部署有噪声、模糊、天气、压缩等 corruption，普通 ReID 干净训练、污染测试会崩。作者关键观察是 t-SNE：不同 corruption 类型会形成各自分布，样本按 corruption 聚在一起。这让 corruption 不再只是数据增强噪声，而像不同模态。

2. 重定义动作：大家以为 CI-ReID 是鲁棒增强或低质量恢复问题，作者改说：CI-ReID 是 clean modality 和 multiple corrupted modalities 之间的语义对齐问题。进一步，严格对齐会丢失模态独有信息，所以要 coarse-to-fine alignment。关键词是 corruption as modality、modality-unified semantic subspace、coarse-to-fine feature alignment、bridge feature generation。

3. 机制怎么长出来：如果 clean 是可靠 exemplar，corrupted 是异质模态，那么 teacher-student 蒸馏很自然：冻结 clean teacher，用 corrupted student 向 clean feature 对齐。严重 corruption 是 hard sample，所以加 weighted contrastive loss。又因为严格对齐会丢信息，所以用 Brownian bridge 生成 clean 与 corrupted 之间的中间特征，填出连续语义结构。机制和重定义绑定很强。

4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。

5. reviewer 为什么买账：真正 novelty 是“把 corruption 看成模态”以及“严格对齐不够，要桥接中间结构”。BFG 本身是借来的思想，但放在 corruption-as-modality 的框架里就有了合理性。它是典型的现实问题重定义型稿子。

**6. Cross-domain person re-identification via learning Heterogeneous Pseudo Labels**

1. 触发观察：跨域 ReID 依赖目标域伪标签，但现有聚类方法通常只给一种伪标签。目标域有遮挡、光照、视角、姿态，导致类内大、类间小；一种伪标签很难准确描述每个样本。另一个观察是 contrastive learning 选正负样本时会被 noisy labels 误导。

2. 重定义动作：作者把问题从“怎么生成更准的单一伪标签”改成“一个目标域样本需要多个异质伪标签共同描述”。关键词是 Heterogeneous Pseudo Labels、fine-grained label、coarse-grained label、instance label、Pseudo Labels Constraint、Confidence Contrastive Loss。

3. 机制怎么长出来：fine-grained label 用 DBSCAN，捕捉细身份差异；coarse-grained label 用 K-means，减少细分带来的噪声；instance label 来自 memory bank，保留样本级关系和历史特征。既然三种标签描述同一样本，就需要 PLC 把异质标签变换到可比较空间并约束一致。既然 contrastive learning 受噪声影响，就用 CCL 根据置信度重权正负样本相似度。

4. 证据闭环：Table 1 很强。单标签里 F、C、I 各有表现，两两组合普遍强于单标签，HPL 三标签又强于两标签；加 PLC 从 80.3/92.5 提到 82.5/92.9，加 CCL 比普通 CL 更好，最终 Ours 达到 87.2 mAP、95.0 Rank-1。Table 2 还证明 DBSCAN 做 fine、K-means 做 coarse 最优，Table 3 讨论 hard/soft 标签形式。这些消融直接证明“异质标签互补”这个重定义。

5. reviewer 为什么买账：卖点是伪标签视角，不是 DBSCAN 或 K-means。它把已有工具组织成“细粒度、粗粒度、实例级”三种互补描述，再用一致性和置信度把故事闭合。它比单纯“伪标签去噪”更像一个可投稿的问题定义。

**共同套路**

这批论文共同套路是：先抓一个很具体的失败现象或反直觉观察，然后把它改名成一个更高层的问题，比如范数动态、伪标签可靠性、内部语义协同、自语义对齐、corruption 模态、异质标签描述。机制不是凭空堆出来，而是让每个组件对应重定义里的一个词，最后用消融或可视化证明“这个新问题定义是真的”，而不是只证明模型涨点。

外部补查只用于确认 CSSC 背景链接，主要判断来自本地 PDF 前 8 页文本： https://arxiv.org/abs/2405.16597
tokens used
145,025
我按 `pdftotext -f 1 -l 8 '文件名' -` 读了你列出的论文。你标题里写“7”，但实际清单只有 6 篇；下面按这 6 篇完整拆。

**1. Condense loss**

1. 触发观察：作者先观察训练过程中“特征向量范数”和准确率的关系。随着训练推进，特征范数下降，准确率上升；Market-1501 前 30 个 epoch 里 rank-1 下降时，特征范数反而上升。这个观察把一个很普通的 loss 设计问题，变成了“训练动态里范数压缩不稳定”的问题。

2. 重定义动作：大家通常把 softmax、NormFace、ArcFace 这类 loss 理解为“角度间隔”或“分类边界”问题，作者改说：ReID 训练里被忽略的是 feature magnitude，它不仅不是要被归一化掉的副产品，反而控制梯度行为。关键词是 vector magnitude、negative radial gradient、tangential gradient、monotonically decreasing。

3. 机制怎么长出来：如果目标是稳定压缩范数，并让后期梯度变小，那么内积分类器就不够自然，因为内积既可增大范数也可减小角度。于是作者把分类分数从 `W^T x` 改成负欧氏距离 `-(W-x)^2`，让梯度方向直接指向类中心，同时通过范数影响切向梯度。margin 只是进一步拉大类间距离。机制和重定义绑定较紧。

4. 证据闭环：闭环不是只靠涨点，而是三层证据：Fig.1 显示范数与准确率的反向趋势；Section 3.4 做 Naive/L2/Condense 的梯度分解；Table 1 比 NormFace、CosFace、ArcFace、Naive Softmax，Table 2 再排除“只是 weight decay 或 normalization 起作用”的解释。这个证据链是在证明“范数和梯度动态值得被建模”。

5. reviewer 为什么买账：卖点主要是视角，不是复杂模块。它把“ReID loss 还能怎么改”从常见的角度间隔，转成训练过程中的范数利用和梯度单调性。缺点是“图像内涵就是范数”这个说法有点过，但数学推导和对照实验把它撑住了。

**2. Confidence guided semi-supervised cross-modality person re-identification**

1. 触发观察：监督 VI-ReID 标注贵，纯无监督 VI-ReID 又和监督方法差距大。关键失败点不是模型不够强，而是跨模态伪标签不可靠。作者还指出固定相似度阈值 0.5 会被背景、姿态等因素误导，不同身份也可能超过阈值。

2. 重定义动作：它把问题从“怎么做更强的跨模态特征学习”改成“少量真标签如何作为当前模型可靠性的标尺，指导无标签跨模态伪标签生成”。关键词是 confidence guided、cross-modality matching confidence、clustering confidence、semi-supervised cross-modality discriminant feature learning。

3. 机制怎么长出来：既然核心是可靠性，就自然要有两个置信度：图像对层面的跨模态匹配置信度，用有标注可见光/红外真匹配对动态估计阈值；簇层面的 clustering confidence，用类似轮廓系数的指标决定保留多少跨模态簇对。之后用静态 memory bank 保持模态内结构，用动态 memory bank 拉近可靠跨模态簇。

4. 证据闭环：前 8 页里最像闭环的是动态阈值分析：SYSU-MM01 上比固定阈值多过滤约 20% 不可靠伪标签对，RegDB 约多过滤 1%，且随着模型变好，被过滤数量下降。这证明“阈值应随模型状态变化”这个重定义。另一个证据是和 DPIS 在 10%、25%、50% 标签率下对齐设置比较，说明不是只靠更宽松设定赢。

5. reviewer 为什么买账：这篇卖的是“半监督跨模态”这个实际设定和“置信度校准伪标签”的视角。模块本身偏工程组合，但每个部件都服务于伪标签可靠性，所以叙事成立。它的投稿价值来自实际问题设定，而不是某个单独模块多新。

**3. Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification**

1. 触发观察：换衣 ReID 里，普通模型过度依赖衣服；很多方法靠姿态、轮廓、衣服标签等额外信息。作者的具体观察很小：average pooling 更保留整体内容，max pooling 更抓显著区域。这个观察被拿来解释 RGB 图像内部其实已有可用语义。

2. 重定义动作：大家以为换衣 ReID 需要“去衣服”或借助外部模态，作者改说：不需要额外模态，RGB 特征里已经有 identity-related content semantics 和 identity-related salient semantics，只是过去没有把两者分开挖掘和协同。关键词是 content semantics、salient semantics、abundant semantics、without auxiliaries。

3. 机制怎么长出来：average pooling 对应 content semantics，max pooling 对应 salient semantics，这一步很自然。SMR 用全局和局部语义挖掘，再用身份监督保证语义和身份相关，用 refinement 做通道重标定。CSSC 的 cross-parallel collaboration 让 content 先帮助 salient，再让 salient 反过来帮助 content。前半段机制自然，cross-parallel 有一定架构设计味道，但能接住叙事。

4. 证据闭环：Table III 显示去掉 SMR 后 PRCC cloth-changing 从 65.5/63.0 掉到 57.8/55.8，local mining 和 refinement 各自也有贡献。Table IV 证明只用 content、只用 salient、简单串联或简单并联都不如 cross-parallel collaboration。Fig.3 检索可视化说明 baseline 被相似颜色和衣服纹理误导，而 CSSC 能靠身体、局部形状、鞋裤等语义找回。

5. reviewer 为什么买账：这篇卖的是视角加简洁性。创新不在 average/max pooling 本身，而在把两个老操作命名成两类身份语义，并用换衣 ReID 的痛点证明它们互补。对 B 类稿很有参考价值：小观察，只要能重定义成清晰语义，再配干净消融，也能成立。

**4. Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification**

1. 触发观察：视频 ReID 的全局特征会混入遮挡、检测框错误、背景；局部 part 特征又有对齐问题。硬条带太粗，外部 pose/parsing 计算重，还会把背包、手提包这类有身份价值的上下文当背景。

2. 重定义动作：它把视频 ReID 从“时间聚合问题”改成“无需外部人体拓扑的语义自对齐问题”。作者强调的不是再做 attention，而是用视频自身上下文学习 pixel-level part-aligned representation 和 semantic-aligned global representation。关键词是 self-alignment、context-aided、semantic-aware、without prior human topology。

3. 机制怎么长出来：先用特征响应强弱分前景/背景，再对同一身份若干视频的前景像素聚类，生成部件伪标签，训练 part classifier。CARE 用整段视频的全局时空上下文增强目标区域、压制噪声。SGFA 根据语义部件置信图给全局特征加权，因为同一部件在不同视频中贡献不同。每个机制都对应一个前面的观察，绑定很紧。

4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。

5. reviewer 为什么买账：这篇卖的是完整视角：外部语义不可靠，视频内部可以自生成语义对齐。模块复杂，但每个模块都解释一个具体失败点，所以不像堆模块。novelty 主要来自“self-alignment without human topology”的问题重定义。

**5. Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment**

1. 触发观察：现实部署有噪声、模糊、天气、压缩等 corruption，普通 ReID 干净训练、污染测试会崩。作者关键观察是 t-SNE：不同 corruption 类型会形成各自分布，样本按 corruption 聚在一起。这让 corruption 不再只是数据增强噪声，而像不同模态。

2. 重定义动作：大家以为 CI-ReID 是鲁棒增强或低质量恢复问题，作者改说：CI-ReID 是 clean modality 和 multiple corrupted modalities 之间的语义对齐问题。进一步，严格对齐会丢失模态独有信息，所以要 coarse-to-fine alignment。关键词是 corruption as modality、modality-unified semantic subspace、coarse-to-fine feature alignment、bridge feature generation。

3. 机制怎么长出来：如果 clean 是可靠 exemplar，corrupted 是异质模态，那么 teacher-student 蒸馏很自然：冻结 clean teacher，用 corrupted student 向 clean feature 对齐。严重 corruption 是 hard sample，所以加 weighted contrastive loss。又因为严格对齐会丢信息，所以用 Brownian bridge 生成 clean 与 corrupted 之间的中间特征，填出连续语义结构。机制和重定义绑定很强。

4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。

5. reviewer 为什么买账：真正 novelty 是“把 corruption 看成模态”以及“严格对齐不够，要桥接中间结构”。BFG 本身是借来的思想，但放在 corruption-as-modality 的框架里就有了合理性。它是典型的现实问题重定义型稿子。

**6. Cross-domain person re-identification via learning Heterogeneous Pseudo Labels**

1. 触发观察：跨域 ReID 依赖目标域伪标签，但现有聚类方法通常只给一种伪标签。目标域有遮挡、光照、视角、姿态，导致类内大、类间小；一种伪标签很难准确描述每个样本。另一个观察是 contrastive learning 选正负样本时会被 noisy labels 误导。

2. 重定义动作：作者把问题从“怎么生成更准的单一伪标签”改成“一个目标域样本需要多个异质伪标签共同描述”。关键词是 Heterogeneous Pseudo Labels、fine-grained label、coarse-grained label、instance label、Pseudo Labels Constraint、Confidence Contrastive Loss。

3. 机制怎么长出来：fine-grained label 用 DBSCAN，捕捉细身份差异；coarse-grained label 用 K-means，减少细分带来的噪声；instance label 来自 memory bank，保留样本级关系和历史特征。既然三种标签描述同一样本，就需要 PLC 把异质标签变换到可比较空间并约束一致。既然 contrastive learning 受噪声影响，就用 CCL 根据置信度重权正负样本相似度。

4. 证据闭环：Table 1 很强。单标签里 F、C、I 各有表现，两两组合普遍强于单标签，HPL 三标签又强于两标签；加 PLC 从 80.3/92.5 提到 82.5/92.9，加 CCL 比普通 CL 更好，最终 Ours 达到 87.2 mAP、95.0 Rank-1。Table 2 还证明 DBSCAN 做 fine、K-means 做 coarse 最优，Table 3 讨论 hard/soft 标签形式。这些消融直接证明“异质标签互补”这个重定义。

5. reviewer 为什么买账：卖点是伪标签视角，不是 DBSCAN 或 K-means。它把已有工具组织成“细粒度、粗粒度、实例级”三种互补描述，再用一致性和置信度把故事闭合。它比单纯“伪标签去噪”更像一个可投稿的问题定义。

**共同套路**

这批论文共同套路是：先抓一个很具体的失败现象或反直觉观察，然后把它改名成一个更高层的问题，比如范数动态、伪标签可靠性、内部语义协同、自语义对齐、corruption 模态、异质标签描述。机制不是凭空堆出来，而是让每个组件对应重定义里的一个词，最后用消融或可视化证明“这个新问题定义是真的”，而不是只证明模型涨点。

外部补查只用于确认 CSSC 背景链接，主要判断来自本地 PDF 前 8 页文本： https://arxiv.org/abs/2405.16597
