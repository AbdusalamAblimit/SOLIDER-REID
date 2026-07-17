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
session id: 019ef835-a8ca-7ca3-bf9f-5dea1eeeba12
--------
user
你是 ReID 创新方法论拆解员 13。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(13)==
FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
Focusing on pedestrians like human for clothes changing person re-identification.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认当前目标和状态文件，然后只读这批 PDF 的前 8 页文本来拆方法论。这个任务不需要写文件，也不需要 Python。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
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

hook: PreToolUse Completed
hook: PreToolUse Completed
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

codex
当前目录只列出了论文库内文件，没有直接看到项目目标文件。我会尝试从上级工作区读取 `GOAL.md`、`CLAUDE.md` 和 `experiments/STATUS.md`，如果沙箱限制了父目录读取，就按用户这条任务继续。
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

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,240p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
我已经对齐了 `GOAL.md` 和当前状态：现在确实处在读文献、拆别人如何构造创新的阶段。下面先提取每篇前 8 页的章节结构，确认方法部分在文本里的位置，再逐篇读方法。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "for f in \\
'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' \\
'Focusing on pedestrians like human for clothes changing person re-identification.pdf' \\
'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' \\
'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' \\
'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' \\
'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf'; do
  printf '\\n===== %s =====\\n' \""'$f"
  pdftotext -f 1 -l 8 "$f" - | rg -n "''^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|VI\\.|VII\\.|1\\.|2\\.|3\\.|4\\.|5\\.|6\\.|[0-9]+\\s+[A-Z]|[A-Z][A-Za-z -]+"'$|A'"\\. |B\\. |C\\. |D\\. |E\\. |F\\. )\" | head -n 80
done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 2740ms:
3:Contents lists available at ScienceDirect
5:Neural Networks
8:Full Length Article
10:Focusing on pedestrians like human for clothes changing person
34:Clothes changing
35:Human focus
49:1. Introduction
69:Identity-related learning methods typically involves recognizing
97:Local details learning methods usually divide data into multiple body
157:2.2. Local details learning
182:2. Related works
183:2.1. Identity-related and identity-irrelated learning
184:It is a prevalent strategy to improve the perception of identity-related
206:Enhancing the learning of identity-irrelated information represents
220:It is a prevailing method to enhance the learning of identity-related
223:2.3. Data augmentation
273:3. Proposed method
315:3.1. Ensemble coding hypothesis
348:3.2. Central emphasis strategy
440:The proposed enlargement grid aims to increase the proportion of the
466:3.3. Component continuity processing
484:This involves erasing random regions between the head-shoulder component and the components of the left upper torso and the right upper
510:CCP can emphasize several regions where identity-related information is
513:3.4. Humanoid focus-inspired image augmentation
662:4.2. Setup
681:4. Experiments
690:4.1. Datasets
702:Methods
704:Reference
706:Type
742:LD
743:LD
744:IR
745:II
746:II
747:II
748:IR
749:IR
751:IR
752:IR
753:IR
754:II
798:HFIA-ZUT
799:HFIA
801:Ours
802:Ours
805:LD
830:4.3. Comparison with state-of-the-art methods
831:We compare our HFIA with the state-of-the-art methods on three
844:5.5 % in R1 (60.4 % vs 54.9 %).
864:4.4. Comparison with data augmentation
865:We compare our HFIA with data augmentation methods on PRCC
875:4.5. Generalization ability of HFIA
876:Local details learning is an important component of various ReID
885:4.5.1. Knowledge distillation

===== GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf =====
3:Contents lists available at ScienceDirect
5:Neural Networks
8:Full Length Article
24:Gait recognition
25:Knowledge distillation
26:Dynamic feature aggregation
43:1. Introduction
54:Re-ID method still relies heavily on appearance features as the primary
145:2. Related work
146:2.1. Person Re-ID
174:Video-based Re-ID is considered a generalization of image-based
175:Re-ID because video frames contain more spatio-temporal information
219:2.3. Knowledge distillation
220:Knowledge distillation aims to compress models while keeping the
243:3. Methods
244:The Re-ID system pays too much attention to appearance features
253:2.2. Gait recognition
267:3.2. Dynamic two-stream aggregation network (DTA-Net)
279:3.1. Overview
281:3.2.1. Re-ID network
332:3.2.2. Gait network
389:3.3. Local perception complementary distillation (LPCD)
412:3.2.3. Dynamic feature aggregation (DFA)
518:3.4. Model learning
581:4. Experiments
582:4.1. Datasets and evaluation protocol
588:We assess the performance of the proposed DTA-Net and GAE-Net
599:Dataset
601:MARS
603:LSVID
605:Resolution
606:Identities
607:Tracklets
608:Cameras
609:Evaluation
665:4.2. Implementation details
666:The entire framework is implemented using PyTorch and is based on
696:4.4. Ablation study
697:A set of ablation experiments are conducted to evaluate the impact
709:4.3. Comparison with state-of-the-arts
721:MARS
723:Method
761:Baseline
1013:Methods
1017:Runtime
1038:4.67
1039:4.24
1040:5.12
1041:6.33
1088:1.0
1090:1.5
1092:2.0
1094:4.0
1115:1.0
1119:2.0
1123:4.0
1126:LSVID
1136:1.0. 𝛽 is utilized to regulate the contribution of NCKD (Normalized Cross
1145:Components

===== GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf =====
6:Representation Learning for
7:Person Re-Identification
11:Abstract— As person parts are frequently misaligned between
30:I. I NTRODUCTION
49:27 May 2022; accepted 17 September 2022. Date of publication 5 October
61:This article has supplementary downloadable material available at
149:II. R ELATED W ORKS
150:A. Person Re-Identification
185:GAReID heads from a totally disparate but effective idea
215:C. Attention Mechanism
217:B. High-Order Statistics
218:High-order statistics has been widely studied in traditional machine learning due to its powerful representation
227:Although high-order features exhibit strong representational
240:GAReID use both the channel group and shuffle strategies
264:N N
267:Theorem N
269:N N
278:III. P ROPOSED M ETHOD
291:B. Grouped High-Order Pooling
296:A. Part Misalignment
305:1 X
306:1 X
309:Vp
324:1 X
325:1 X
328:Vp
407:1 XO
408:1 XO
421:1 XO
478:Ijx
481:Ijx
505:Ij
509:Ijx
543:1 XO
551:Since multiple input features provide informative semantic
562:1 X 1 O
563:X px
566:X npx
578:C. Attentive High-Order Pooling
643:Mi
658:1 X
668:It is worth noting that this AHOP layer can be viewed
672:D. Overall Loss Function
682:Xh
716:IV. D ISCUSSION
717:A. Feature Visualization
737:AHOP layer is able to remove the background regions and
741:B. Similarity Visualization
775:C. Landmark Visualization
801:AHOP layer certifies that foreground region mining is conducive to highlighting the semantic correspondences of person
803:D. Attention Visualization

===== Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf =====
3:Contents lists available at ScienceDirect
5:Pattern Recognition
8:Generalizable person re-identification method using bi-stream interactive
13:ARTICLE
15:INFO
18:Pedestrian re-identification
19:Correlation graph sampling
20:Sparsely focused
21:Correlation reconstruction
23:ABSTRACT
24:Recent studies have shown that metric learning and representation learning are two main methods to improve
42:1. Introduction
54:One area of current research focuses in enhancing the generalization
56:It aims to design training objectives with different sampling strategies
62:The PK sampler randomly selects p classes and then samples k
82:F. Min et al.
134:Our approach establishes a mutually reinforcing relationship between
149:This attention is global in nature and aims to reduce the loss
155:2. Related work
165:2.1. Feature representation learning
166:One of the current hot research topics in pedestrian re-identification
178:F. Min et al.
183:Local feature representation learning utilizes local image regions to
220:The well-known PK sampler is the most widely used random sampling
241:3. Method
243:2.2. Deep metric learning
269:Small-batch samplers also play an important role in deep metric
283:3.1. Correlation graph sampling
284:The CGS sampler aims to improve the discriminative ability and
293:F. Min et al.
335:To ensure that each batch of training samples is correlated and
349:3.1.2. Nearest neighbor graph node construction
371:3.1.1. Hash bucket allocation of samples
372:The principles of hashed bucket allocation involve using the Locally
398:F. Min et al.
436:The newly obtained discriminative model from the previous round
451:3.3. Global relevance sparse reconstruction
452:The convolution operation in the feature extraction network can
535:3.2. Global sparse attention network
568:F. Min et al.
598:3.5. Loss function
612:3.4. Multi-scale feature fusion
623:The integration of feature layers with different resolutions in the
633:F. Min et al.
668:4.3. Comparison to the state of the art
693:1.6%, respectively. For MSMT17 → Market-1501, the Rank-1 and mAP
707:The experimental results confirm the effectiveness of the method in
722:4. Experiment
723:4.1. Experimental details
734:4.2. Datasets
750:F. Min et al.
753:Method
755:Venue
757:Training
783:Multi
784:Multi
814:GSANet-CGS
826:Ours
895:5.9
907:1.7
922:GSANet-CGS
934:Ours
1024:GSANet-CGS
1030:Ours
1078:Method
1080:Training
1084:Market
1095:Experimental results on the impact of CGS and GRSR modules on generality with
1097:Backbone
1099:CGS
1101:GRSR
1103:Training
1107:Market
1119:MSMT
1120:MSMT
1121:MSMT
1141:MSMT
1142:MSMT
1143:MSMT
1163:MSMT

===== Global aggregated gradient-guided adversarial attacks for person re-identification.pdf =====
3:Contents lists available at ScienceDirect
5:Pattern Recognition
8:Global aggregated gradient-guided adversarial attacks for person
10:Zeze Tao
34:Adversarial examples
35:Transferability
36:Global aggregated gradient
37:Gradient consistency
48:1. Introduction
110:2.2. Adversarial attacks
136:These methods collectively operate by perturbing model outputs at
167:2. Related work
168:2.1. Person re-identiﬁcation
177:2.3. Adversarial attacks on person re-identiﬁcation
202:3.2.1. Global aggregated gradient
220:3. Methodology
221:3.1. Problem deﬁnition
309:Agg
312:Agg
340:3.2.2. Gradient consistency
346:3.2. Overall framework
347:This paper proposes a novel Global Aggregated Gradient Attack
358:This work aims to improve the consistency between the current gradient and the globally aggregated gradient during the initial attack
361:Agg
369:The core idea of the weighted gradient is to dynamically adjust the
371:Agg
375:Agg
417:Agg
433:The primary objective of our work is to enhance gradient consistency
479:These models are trained using cross-entropy loss and optimized with
521:Agg
559:We will separately validate the superiority of the GAGA method in both
571:The primary goal of improving gradient consistency at the early
575:4.2. Comparison with state-of-the-art methods
576:4.2.1. White-Box attacks on re-ID models
592:4. Experiments
593:4.1. Experimental setup
605:Dataset
609:Method
619:Average
637:Before attack
648:3.86
649:1.28
651:2.67
652:5.97
653:1.37
658:2.81
661:1.89
662:4.24
663:1.20
668:3.33
671:1.66
672:6.15
673:1.07
678:2.27
681:1.37
682:4.38
688:4.07
689:1.07
691:2.58
693:1.25
698:3.03
701:2.00
702:5.14
703:1.18
708:3.75
711:2.30
712:6.52
713:1.23
718:2.70
721:1.75
722:4.59
723:1.12
727:Before attack
738:1.48
742:1.54
758:1.16
762:1.42
778:1.23
782:1.16

codex
章节结构已经出来了，六篇的方法段都在前 8 页内。接下来我会重点读 `3. Method` 或同名部分，同时看实验里的关键消融位置，因为你的问题要的是“观察到问题，如何重定义，如何闭环证明”。
codex
我先从第一篇开始按“引言里的失败叙事，加方法里的机制，实验里的归因证据”三块读。方法本身会用前 8 页文本，消融如果落在后面页面，我会额外读对应页来补证据闭环。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '35,172p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    35	1. Introduction
    36	Person Re-IDentification (ReID) aims at matching the
    37	given pedestrians from an image gallery taken by different cameras. Most existing ReID models focus on the
    38	visible-visible image matching (i.e., VV-ReID). However,
    39	* Equally corresponding authors.
    40	
    41	978-1-6654-6946-3/22/$31.00 ©2022 IEEE
    42	DOI 10.1109/CVPR52688.2022.00720
    43	
    44	Modality-Shared
    45	Features
    46	
    47	Generating
    48	VI-ReID
    49	Network
    50	
    51	VI-ReID
    52	Network
    53	Generating
    54	
    55	Infrared
    56	Modality-Specific
    57	ality
    58	Features
    59	
    60	Missing Image Generation
    61	(b)
    62	
    63	(a)
    64	
    65	Visible ModalitySpecific Features
    66	Infrared
    67	Modality-Specific
    68	Features
    69	Modality-Shared
    70	Features
    71	
    72	Matching
    73	
    74	Generating
    75	
    76	Infrared
    77	Modality-Specific
    78	Features
    79	Visible ModalitySpecific Features
    80	
    81	Generating
    82	
    83	Our Model
    84	
    85	Modality-Shared
    86	Features
    87	
    88	Our Model
    89	(c)
    90	
    91	Figure 1. Illustration of the differences between our model and
    92	existing VI-ReID models. (a) Existing modality-shared feature
    93	learning based models. (b) Existing image-level compensation
    94	based models. (c) Our proposed feature-level compensation based
    95	model.
    96	
    97	these models may have poor performance when visible
    98	cameras cannot well capture information, such as at night.
    99	Compared with visible cameras, infrared cameras can still
   100	capture clear images under those poor illumination conditions. Moreover, most cameras in modern surveillance systems support autoswitch between the visible and infrared
   101	modes under different illumination conditions. Accordingly, Visible-Infrared ReID (i.e., VI-ReID) has raised more
   102	and more attention recently.
   103	The main challenge of VI-ReID lies in the modality discrepancy between the visible and infrared images. Meanwhile, it also surfers from large person variations, such as
   104	viewpoints and postures. As shown in Fig. 1(a), most existing models [1–7] try to extract the discriminative modalityshared features for VI-ReID. Although great improvements
   105	have been achieved, these models inevitably discard lots
   106	of discriminative person-related modality-specific information, which may also benefit VI-ReID. Considering that,
   107	
   108	7339
   109	
   110	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.
   111	
   112	some works [8, 9] propose the idea of modality-specific information compensation, which attempts to first generate
   113	those missing modality-specific information from existing
   114	modality and then jointly uses the generated and original
   115	information for VI-ReID.
   116	However, existing modality-specific information compensation based models usually achieve inferior results
   117	compared with those modality-shared feature learning
   118	based models. This may impute to the image-level compensation of existing models. That is, as shown in Fig.
   119	1(b), existing models first generate the images of missing
   120	modality from the images of existing modality and then extract discriminative person features from the paired images
   121	for VI-ReID. However, it is very difficult to generate highquality images of one modality from another modality, due
   122	to the large modality discrepancy between the visible and
   123	infrared images. Especially, when generating visible images from infrared images, much more noisy information
   124	(e.g., color inconsistency), instead of discriminative person features, will be introduced for VI-ReID. Besides, these
   125	existing modality-specific information compensation based
   126	models usually follow a two-stage structure and are not endto-end trainable, where the image generation sub-networks
   127	and VI-ReID subnetworks are independent trained.
   128	Actually, compared with the modality discrepancy between visible and infrared images, their features’ discrepancy has been reduced to some extents, since some common semantics information usually coexists in the unimodal
   129	visible and infrared features. Therefore, the translation between visible data and infrared data in the feature level
   130	may be easier than that in the image level. Meanwhile,
   131	as discussed in some existing works [10–12], the singlemodality features (e.g., unimodal visible features or infrared features) can be decomposed into their own modalityspecific features and modality-shared features. The difficulties for cross-modality translation can be further reduced by generating those missing modality-specific features from existing modality-shared features rather than
   132	from the whole single-modality features. More importantly, compared with image-level translation, the featurelevel translation allows us to flexibly control the generation
   133	of those missing modality-specific features as our requirements by designing some dedicated loss functions. For example, we can only generate some discriminative personrelated modality-specific features and discard those nondiscriminative ones for benefiting VI-ReID.
   134	Considering that, we will present a novel end-to-end
   135	feature-level modality-specific information compensation
   136	based model, i.e., the Feature-level Modality Compensation
   137	Network (FMCNet), for VI-ReID in this paper. As shown
   138	in Fig. 1(c), our proposed FMCNet aims to compensate
   139	those missing modality-specific information in the feature
   140	level rather than in the image level, i.e., directly generat-
   141	
   142	ing those missing modality-specific features of one modality from existing modality-shared features of other modality. To this end, a Single-modality Feature Decomposition (SFD) module is first utilized to decompose the input
   143	single-modality features into their own modality-specific
   144	and modality-shared features, respectively. Meanwhile, a
   145	modality decomposition loss is designed to facilitate the
   146	decomposition of those single-modality features. Then, a
   147	Feature-level Modality Compensation (FMC) module is designed to generate the missing modality-specific features of
   148	one modality from the existing modality-shared ones of the
   149	other modality for each sample image. Finally, a Sharedspecific Feature Fusion (SFF) module is designed to jointly
   150	use the existing modality-shared and modality-specific features as well as the generated modality-specific features for
   151	VI-ReID.
   152	Similarly, cm-SSFT [13] also tries to simultaneously exploit those modality-shared and modality-specific features
   153	for VI-ReID. It achieves shared-specific feature transfer by
   154	modeling the affinities among different samples. Specially,
   155	those missing modality-specific features in the cm-SSFT
   156	are transfered from all the samples of the other modality in
   157	the gallery. This may also introduce more modality-specific
   158	information of other identities, thus easily leading to suboptimal results. Different from cm-SSFT, our proposed
   159	model does not rely on other samples and is able to directly
   160	and flexibly generate those missing modality-specific features from its own modality-shared features.
   161	In summary, the main contributions of this work are as
   162	follows:
   163	(1) A novel FMCNet is presented, which proposes
   164	feature-level rather than image-level modality-specific information compensation for VI-ReID. This enables our
   165	model to focus on generating some required missing
   166	modality-specific features (e.g., discriminative personrelated ones) for VI-ReID.
   167	(2) Our proposed FMCNet provides an unified endto-end framework, achieving unimodal feature decomposion, modality-specific feature compensation and modality
   168	shared-specific feature fusion for VI-ReID via the proposed
   169	SFD, FMC and SFF modules, respectively.
   170	(3) Our model significantly outperforms those imagelevel compensation based models and obtains competitive
   171	and even better results than some state-of-the-art modalityshared feature learning based ones.
   172	

hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '290,608p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   290	3. Method
   291	As shown in Fig. 2, the proposed model, i.e., Featurelevel Modality Compensation Network (FMCNet), mainly
   292	consists of three parts, i.e., a Single-modality Feature Decomposition (SFD) module, a Feature-level Modality Com-
   293	
   294	pensation (FMC) module and a Shared-specific Feature Fusion (SFF) module. Concretely, the proposed SFD module first extracts single-modality features from the input images and then decomposes them into their own modalityspecific and modality-shared ones. Then, the proposed
   295	FMC module generates the missing (or compensated) visible (infrared) modality-specific features from those existing
   296	decomposed infrared (visible) modality-shared features in
   297	an adversarial way. Finally, the original modality-specific
   298	features and modality-shared features as well as their compensated modality-specific features will be combined in the
   299	proposed SFF module for VI-ReID. Details about these
   300	modules will be discussed in the following contents.
   301	Suppose that the training set (XV , XI ) contains P identities and each identity contains K samples. XV =
   302	{xk,p
   303	V , k = 1, .., K; p = 1, ..., P } denotes visible sample
   304	images, and XI = {xk,p
   305	I , k = 1, .., K; p = 1, ..., P } denotes infrared sample images.
   306	
   307	3.1. SFD Module
   308	As shown in Fig. 2, given the input visible images XV or
   309	the infrared images XI , the proposed SFD module first extracts their single-modality features and then decomposes
   310	those extracted single-modality visible (infrared) features
   311	into their own modality-specific features and modalityshared features. Here, the ways of extracting and decomposing those single-modality visible and infrared features
   312	are the same. Therefore, we take the input visible images
   313	XV as the example to detail the corresponding process.
   314	Specifically, the single-modality features FV are first extracted from XV by using a visible feature extraction subnetwork EV (∗). Then, a visible modality-specific feature
   315	
   316	7341
   317	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.
   318	
   319	ID-1
   320	ID-2
   321	
   322	l3
   323	
   324	ID-1
   325	
   326	The decorrelation loss (Ldc ) aims to push the modalityspecific features away from the modality-shared features.
   327	For that, it first computes the modality-specific feature center as well as the modality-shared feature center for each
   328	identity by
   329	
   330	ID-2
   331	l1
   332	
   333	l4
   334	l2
   335	
   336	Modality-shared
   337	Features
   338	Visible and infrared modality-shared features
   339	
   340	Modality-specific
   341	Features
   342	Feature center
   343	
   344	Visible or infrared modality-specific features
   345	
   346	  \small \begin {aligned} C_{sp, m}^{p} &=\frac {1}{K}\sum \limits _{k=1}^{K}\mathbf {F}_{sp,m}^{k,p}, C_{sh, m}^{p} =\frac {1}{K}\sum \limits _{k=1}^{K}\mathbf {F}_{sh,m}^{k,p}. \end {aligned} 
   347	
   348	(3)
   349	
   350	Figure 3. Illustration of the proposed MD loss.
   351	p
   352	Here, Csp,m
   353	denotes the center of the p-th identity’s
   354	p
   355	modality-specific features. Similarly, Csh,m
   356	denotes the
   357	center of the p-th identity’s modality-shared features. Then,
   358	to push the modality-specific features away from the
   359	modality-shared features, it constraints that the maximum
   360	distances among different modality-specific feature centers
   361	(e.g., l1 in Fig. 3) should be smaller than the minimum
   362	distances from the modality-specific feature centers to the
   363	modality-shared feature centers (e.g., l2 in Fig. 3), i.e.,
   364	
   365	extraction sub-network EVsp (∗) and a modality-shared feature extraction sub-network Esh (∗) are performed on FV to
   366	decompose them into their corresponding visible modalityspecific features Fsp,V and visible modality-shared features
   367	Fsh,V , respectively, i.e.,
   368	  \small \begin {aligned} &\mathbf {F}_{V} = \operatorname {E}_{V} (X_{V} ), \mathbf {F}_{sp,V}=\operatorname {E}_{sp}^V (\mathbf {F}_{V} ),\mathbf {F}_{sh,V} =\operatorname {E}_{sh} (\mathbf {F}_{V}). \end {aligned} 
   369	(1)
   370	
   371	Finally, a specific visible identity classifier PVsp (∗) is performed on Fsp,V to predict the corresponding identity score
   372	Ssp,V . Meanwhile, a shared identity classifier Psh (∗) is
   373	performed on Fsh,V , which outputs their predicted identity
   374	score Ssh,V . Mathematically, these processes can be expressed by
   375	  \small \begin {aligned} &S_{sp,V} = \operatorname {P}_{sp}^V (\mathbf {F}_{sp,V} ), S_{sh,V} =\operatorname {P}_{sh} (\mathbf {F}_{sh,V}). \end {aligned} 
   376	
   377	 \label {dc} \small \begin {aligned} L_{dc} =\sum \limits _{p=1}^{P}&\left (\max \left (\max \limits _{d}\Vert C_{sp, V}^{p}-C_{sp,V}^{d}\Vert _{2}-\right . \right .\\& \left .\min \limits _{j}\Vert C_{sp, V}^{p} -C_{sh, V}^{j}\Vert _{2}+\rho _{1},0\right ) +\\& \left . \max \left (\max \limits _{d}\Vert C_{sp, I}^{p}-C_{sp, I}^{d}\Vert _{2}- \right .\right .\\&\left .\left . \min \limits _{j}\Vert C_{sp, I}^{p}-C_{sh, I}^{j}\Vert _{2}+\rho _{1} ,0 \right )\right ). \end {aligned} 
   378	
   379	(4)
   380	
   381	(2)
   382	
   383	Similarly, we may obtain the single-modality features
   384	FI , infrared modality-specific features Fsp,I and infrared
   385	modality-shared features Fsh,I from XI by using the
   386	EI (∗), EIsp (∗) and Esh (∗), respectively. The corresponding identity scores Ssp,I and Ssh,I are thus obtained by using the specific infrared identity classifier PIsp (∗) and the
   387	shared identity classifier Psh (∗), respectively.
   388	Here, EV (∗) and EI (∗) follow the same structure with
   389	the first three blocks of ResNet-50 [21]. Similarly, EVsp (∗)
   390	and EIsp (∗) follow the same structure with the last two
   391	blocks of ResNet-50 and further attach an extra global average pooling layer, respectively. Moreover, these subnetworks’ parameters are independent to each other. Esh (∗)
   392	has the same network sturcture with the EVsp (∗). However,
   393	its parameters are shared for single-modality visible and infrared features to extract their modality-shared features.
   394	Loss function: To facilitate decomposing the singlemodality features Fm (m ∈ {V, I}) into modality-specific
   395	features Fsp,m and modality-shared features Fsh,m , a novel
   396	Modality Decomposition (MD) loss is further designed. As
   397	shown in Fig. 3, MD loss aims to separate modality-shared
   398	features away from those modality-specific features, and
   399	meanwhile makes those decomposed modality-specific features and modality-shared features identity-discriminable.
   400	Therefore, the proposed MD loss consists of three items, including a decorrelation loss (Ldc ), a modality-specific feature separation loss (Lsps ) and a modality-shared feature
   401	separation loss (Lshs ).
   402	
   403	Here, d, j = 1, 2, ..., P . ρ1 denotes the corresponding margin and is empirically set to 1.
   404	As shown in the right part of Fig. 3, the modality-specific
   405	feature separation loss (Lsps ) tries to separate the decomposed modality-specific features according to their identities. To this end, it enlarges the distances among the visible (infrared) modality-specific feature centers of different
   406	identities (e.g., l1 in Fig. 3), i.e.,
   407	 \label {sps} \small \begin {aligned} L_{sps} =&\sum \limits _{p=1}^{P}\left (\max \left (\rho _{2}-\min \limits _{j\neq p}\Vert C_{sp, V}^{p}-C_{sp, V}^{j} \Vert _{2},0 \right ) \right . \\ & \left .+ \max \left (\rho _{2}-\min \limits _{d\neq p}\Vert C_{sp, I}^{p}-C_{sp, I}^{d} \Vert _{2} ,0\right )\right ). \end {aligned} 
   408	(5)
   409	
   410	Here, j, d = 1, 2, .., P . ρ2 denotes the corresponding margin, which is empirically set to 0.7.
   411	As shown in the left part of Fig. 3, the modalityshared feature separation loss (Lshs ) tries to simultaneously
   412	make the decomposed modality-shared features be identitydistinguishable and modality-invariant. To this end, Lshs
   413	tries to shrink the distances between the visible modalityshared feature centers and the infrared modality-shared feature centers from the same identities (e.g., l4 in Fig. 3), and
   414	meanwhile enlarge the distances between the visible (infrared) modality-shared feature centers and the both visible
   415	
   416	7342
   417	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.
   418	
   419	(e.g., Fig. 4(b)). This means that, with the collaboration of
   420	the designed network structure and the MD loss, the input
   421	single-modality features will be successfully decomposed
   422	into the modality-shared ones and modality-specific ones.
   423	(a)
   424	
   425	3.2. FMC Module
   426	
   427	(b)
   428	
   429	Figure 4. Distributions of the modality-shared features and
   430	modality-specific features. (a) FMCNet without MD loss. (b) FMCNet with MD loss.
   431	
   432	and infrared modality-shared feature centers from different
   433	identities (e.g., l3 in Fig. 3), i.e.,
   434	 \label {shs} \small \begin {aligned} \small &L_{shs} =\sum \limits _{p=1}^{P}\left (\alpha \Vert C_{sh,V}^{p}-C_{sh,I}^{p} \Vert _{2} +\max \left (\rho _{3}- \right .\right .\\& \left .\left . \min \limits _{j\neq p}\Vert C_{sh,V}^{p} -C_{sh,m}^{j}\Vert _{2},0\right ) + \max \left (\rho _{3}- \min \limits _{d\neq p}\Vert C_{sh,I}^{p}- \right . \right .\\& \left . \left . C_{sh,m}^{d}\Vert _{2} ,0\right )\right ). \end {aligned} 
   435	
   436	(6)
   437	
   438	Here, j, d = 1, 2, ..., P . ρ3 denotes the corresponding margin and is also set to 0.7. α is a predefined tradeoff parameters to balance the different losses and is set to 2.
   439	Accordingly, the proposed MD loss is totally expressed
   440	by
   441	  \small \begin {aligned}\label {MD} L_{MD} &=L_{shs}+\lambda _{1}L_{dc}+\lambda _{2}L_{sps}, \end {aligned} 
   442	
   443	(7)
   444	
   445	where λ1 and λ2 are the predefined tradeoff parameters to
   446	balance different losses and are both set to 0.5.
   447	Besides, an identity classification (ID) loss is employed
   448	to facilitate extracting those person-related features and discarding those background information, i.e.,
   449	  \small \begin {aligned} L_{ID} &= L_{CE}(S_{sh,V}, Y_V) + L_{CE}(S_{sh,I}, Y_I) \\ & + L_{CE}(S_{sp,V}, Y_V) + L_{CE}(S_{sp,I}, Y_I), \end {aligned} 
   450	
   451	(8)
   452	
   453	where YV and YI denote the ground truths. Here, the ID
   454	loss is constructed by using cross-entropy loss LCE , i.e.,
   455	  \small \begin {aligned} L_{CE}(X, Y) &=-\frac {1}{N}\sum \limits _{i=1}^{N}y_{i} log(x_i), \end {aligned} 
   456	
   457	(9)
   458	
   459	where, X = {x1 , ..., xN } and Y = {y1 , ..., yN }. xi denotes the predicted classification score for the i-th sample,
   460	and yi denotes the corresponding ground truth. Here, N is
   461	the total numbers of samples contained in X. Therefore, the
   462	totall loss LSF D for training the SFD module is
   463	  \small \begin {aligned} L_{SFD} &= L_{MD}+ L_{ID}. \end {aligned} 
   464	
   465	(10)
   466	
   467	As shown in Fig. 4(a), without using the proposed MD
   468	loss, the modality-shared and modality-specific features of
   469	different identities are mixed together. While, by virtue of
   470	the proposed MD loss, those modality-shared and modalityspecific features are effectively separated from each other
   471	
   472	As discussed in the earlier part of this section, the
   473	next step in FMCNet is to directly generate those missing modality-specific information in the feature level rather
   474	than image level via the proposed FMC module. Meanwhile, as shown in Fig. 2, the process of generating the
   475	missing infrared modality-specific features F′ sp,I from the
   476	existing visible modality-shared features Fsh,V is similar to
   477	that of generating the missing visible modality-specific features F′ sp,V from the existing infrared modality-shared features Fsh,I . We take the process of generating F′ sp,I from
   478	Fsh,V as an example to detail our proposed FMC module.
   479	Specifically, the proposed FMC module consists of a
   480	feature-level generator GV −I (∗) and a feature-level modality discriminator DV −I (∗). The visible modality-shared
   481	features Fsh,V are first fed into the feature-level generator
   482	GV −I (∗) to generate the missing (or compensated) infrared
   483	modality-specific features F′ sp,I , i.e.,
   484	  \small \begin {aligned} \mathbf {F'}_{sp,I} &=\operatorname {G}_{V-I}(\mathbf {F}_{sh,V} ). \end {aligned} 
   485	
   486	(11)
   487	
   488	Here, the feature-level generator GV −I (∗) is constructed by
   489	using three stacked fully connected layers.
   490	Then, given the generated infrared modality-specific
   491	features F′ sp,I and the existing real infrared modalityspecific features Fsp,I , the feature-level modality discriminator DV −I (∗) aims to accurately distinguish the two types
   492	of modality-specific features. It is constructed by using one
   493	layer fully connected layer stacked with a Sigmoid function
   494	and outputs a classification score St for distinguishing the
   495	two types of features. The higher value of St indicates that
   496	the input features are more likely to be corresponding real
   497	infrared modality-specific features.
   498	The feature-level generator GV −I (∗) and the featurelevel modality discriminator DV −I (∗) are trained in an adversarial way. Concretely, GV −I (∗) tries to fool the discriminators DV −I (∗) by generating the missing infrared
   499	modality-specific features that approximate real infrared
   500	modality-specific features as closely as possible. While, the
   501	discriminators DV −I (∗) tries to distinguish the generated
   502	modality-specific features and the real ones as accurately
   503	as possible. Accordingly, the generated infrared modalityspecific features F′ sp,I will be eventually close to the real
   504	ones Fsp,I . Mathematically, the adversarial loss is defined
   505	by
   506	 \label {lgan} \small \begin {aligned} \min \limits _{\operatorname {G}_{V-I}}\max \limits _{\operatorname {D}_{V-I}}&L_{GAN}^{V-I} = \frac {1}{PK}\sum \limits _{p=1}^{P}\sum \limits _{k=1}^{K} \left ( log\left (\operatorname {D}_{V-I}(\mathbf {F}_{sp,I}^{k,p})\right ) \right .\\ &\left .+ log\left (1 -\operatorname {D}_{V-I}(\operatorname {G}_{V-I}(\mathbf {F}_{sh,V}^{k,p}))\right ) \right ). \end {aligned} 
   507	(12)
   508	
   509	7343
   510	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.
   511	
   512	F′ sp,I serve as the auxiliary information. Therefore, we first
   513	combine Fsp,V and F′ sp,I to obtain fused modality-specific
   514	features Ff u,V via a weighted fusion way, i.e.,
   515	  \small \begin {aligned} \mathbf {F}_{fu,V}&= \omega _{1}\mathbf {F}_{sp,V} + \omega _{2}\mathbf {F'}_{sp,I}. \end {aligned} 
   516	
   517	Figure 5. Distributions of the existing modality-specific features,
   518	modality-shared features and those generated modality-specific
   519	features of two modalities.
   520	
   521	Besides Eq. (12), the generator GV −I (∗) is also super−I
   522	vised by a feature consistency loss LVF C
   523	and an identity
   524	V −I
   525	V −I
   526	consistency loss LIC . LF C aims to make the generated
   527	features be close to the infrared modality-specific feature’s
   528	centers of the same identities, which is expressed by
   529	  \small \begin {aligned} L^{V-I}_{FC} &=\frac {1}{PK}\sum \limits _{p=1}^{P}\sum \limits _{k=1}^{K}\Vert \mathbf {F'}_{sp,I}^{k,p}-{C}_{sp,I}^{p}\Vert _{1}, \end {aligned} 
   530	
   531	(13)
   532	
   533	where ∥ ∗ ∥1 denotes the l1 -norm of a vector or matrix.
   534	While, LVIC−I enforces the generated features to be discriminative for person identification, i.e.,
   535	  \small \begin {aligned} L^{V-I}_{IC}&=L_{CE}(S'_{sp,I}, Y_I), \end {aligned} 
   536	
   537	(14)
   538	
   539	′
   540	denotes the set of the predicted identity scores
   541	where Ssp,I
   542	by feeding F′ sp,I into the specific infrared classifier PIsp (∗).
   543	Similarly, given the infrared modality-shared features
   544	Fsh,I , their corresponding visible modality-specific features F′ sp,V can be obtained in the same way by using
   545	a feature-level generator GI−V (∗) and a feature-level discriminator DI−V (∗).
   546	Fig. 5 shows that the distributions of those visible (infrared) modality-specific features generated by FMC module are very close to those of existing visible (infrared)
   547	modality-specific features. Moreover, both the existing
   548	and the generated modality-specific features are identitydiscriminable. This means that, by virtue of the proposed
   549	FMC module, the missing modality-specific information
   550	will be effectively compensated in the feature level.
   551	
   552	Here, ω1 and ω1 are weights for Fsp,V and F′ sp,I , respectively, which are also learnable parameters.
   553	Then, the modality-shared features Fsh,V and the fused
   554	modality-specific features Ff u,V are concatenated to obtain
   555	the final fused person features Ff p,V of the visible images,
   556	i.e.,
   557	  \small \begin {aligned} \mathbf {F}_{fp,V} &= \operatorname {Cat}(\mathbf {F}_{sh,V} , \mathbf {F}_{fu,V} ), \end {aligned} 
   558	
   559	(16)
   560	
   561	where Cat(∗) denotes the concatenation operation. The
   562	corresponding identity score Sf p,V is thus obtained by feeding the fused features Ff p,V into a shared identity classifier
   563	Pf p (∗).
   564	Similarly, the final fused person features Ff p,I of the
   565	infrared images are obtained by fusing Fsh,I , Fsp,I and
   566	F′ sp,V in the same way. The corresponding identity score
   567	Sf p,I is thus obtained by feeding Ff p,I into the shared identity classifier Pf p (∗).
   568	Loss function: Similar to that in Eq. (6), a crossmodality center (MC) loss is also employed to make the
   569	learned person features Ff p,V and Ff p,I be discriminative
   570	and modality-invariant, i.e.,
   571	  \small \begin {aligned}\label {MC} L_{MC} &=\sum \limits _{p=1}^{P} \left ( \beta \Vert C_{fp,V}^{p}-C_{fp,I}^{p} \Vert _{2} \right . \\ & \left . +\max \left (\rho _{4}-\min \limits _{j\neq p}\Vert C_{fp,V}^{p}-C_{fp,m}^{j}\Vert _{2},0\right ) \right . \\ & \left . +\max \left (\rho _{4}-\min \limits _{d\neq p}\Vert C_{fp,I}^{p}-C_{fp,m}^{d}\Vert _{2} ,0\right )\right ). \end {aligned} 
   572	(17)
   573	
   574	1
   575	Here, j, d = 1, 2, ..., P , and Cfpp,m = K
   576	
   577	K
   578	P
   579	k=1
   580	
   581	Fk,p
   582	f p,m . ρ4
   583	
   584	denotes the corresponding margin and β is the predefined
   585	tradeoff parameter to balance different losses, which are set
   586	to 0.7 and 2, respectively. Besides, the identity classification
   587	loss is also performed on the fused features by
   588	  \small \begin {aligned} L_{FID} &= L_{CE}(S_{fp,V}, Y_V) + L_{CE}(S_{fp,I}, Y_I). \end {aligned} 
   589	
   590	3.3. SFF Module
   591	After that, a Shared-specific Feature Fusion (SFF) module is further designed to mine the original modalityspecific features and modality-shared features as well as
   592	those generated modality-specific features for VI-ReID.
   593	Here, we take the fusion of the visible modality-shared features Fsh,V , the visible modality-specific features Fsp,V
   594	and the generated infrared modality-specific features F′ sp,I
   595	as an example to describe its steps.
   596	In our proposed SFF module, the modality-shared features Fsh,V are considered as the primary information for
   597	VI-ReID, while those modality-specific features Fsp,V and
   598	
   599	(15)
   600	
   601	(18)
   602	
   603	Therefore, total loss for training our proposed SFF module
   604	is
   605	 \label {fu} \small \begin {aligned} L_{SFF}&= L_{MC} + L_{FID}. \end {aligned} 
   606	
   607	(19)
   608	

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdfinfo 'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' | sed -n '1,40p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           FMCNet: Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification
Subject:         2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR);2022; ; ;10.1109/CVPR52688.2022.00720
Producer:        OpenPDF 1.0.0-SNAPSHOT; modified using iText® 7.1.1 ©2000-2018 iText Group NV (AGPL-version)
CreationDate:    Wed Jul 27 04:43:47 2022 CST
ModDate:         Wed Sep 21 22:27:31 2022 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           10
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       2610888 bytes
Optimized:       no
PDF version:     1.6

codex
第一篇的核心比较清楚：它不是简单说“跨模态差异大”，而是把失败点收窄到“图像级补偿把噪声带进来，特征级补偿更可控”。我会继续补它的消融证据，然后读下一篇。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 10 'FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Setting
     2	
     3	Rank-1
     4	
     5	mAP
     6	
     7	Base
     8	Base+SFD
     9	Base+SFD+FMC
    10	Base+SFD+FMC+SFF
    11	
    12	57.09
    13	63.16
    14	65.50
    15	66.34
    16	
    17	53.11
    18	58.83
    19	62.32
    20	62.51
    21	
    22	Table 2. Evaluation of each component in our proposed model.
    23	Setting
    24	
    25	Rank-1
    26	
    27	mAP
    28	
    29	FMCNet+LID
    30	FMCNet+LID +Lshs
    31	FMCNet+LID +Lshs +Lsps
    32	FMCNet+LID +Lshs +Lsps +Ldc
    33	
    34	58.68
    35	63.96
    36	64.68
    37	66.34
    38	
    39	54.33
    40	58.64
    41	59.34
    42	62.51
    43	
    44	Table 3. Effectiveness of different items in our proposed MD loss.
    45	Setting
    46	
    47	Rank-1
    48	
    49	mAP
    50	
    51	w/o FMC
    52	FMC+LIC +LF C
    53	FMC+LGAN
    54	FMC+LGAN +LIC
    55	FMC+LGAN +LIC +LF C
    56	
    57	63.16
    58	57.78
    59	62.56
    60	65.61
    61	66.34
    62	
    63	58.83
    64	55.56
    65	58.40
    66	62.17
    67	62.51
    68	
    69	Table 4. Evaluation results of different losses in FMC module.
    70	
    71	4.4. Ablation Study
    72	In this subsection, we evaluate each component of our
    73	proposed model on SYSU-MM01 dataset.
    74	Effectiveness of each module: As shown in Table 2,
    75	we first remove SFD and FMC from our model as the
    76	‘Base’, which just consists of EV (∗), EI (∗) and Esh (∗) in
    77	Fig. 2. Moreover, ‘Base’ is trained by only using ID loss.
    78	‘Base+SFD’ denotes the model that employs the proposed
    79	SFD module, and is jointly trained by using MD loss and
    80	ID loss. As well, ‘Base+SFD’ only uses the decomposed
    81	modality-shared features for VI-ReID. ‘Base+SFD+FMC’
    82	further employs the proposed FMC module for VI-ReID,
    83	where the existing modality-shared and modality-specific
    84	features and the generated modality-specific features are
    85	simply concatenated. ‘Base+SFD+FMC+SFF’ then attaches the proposed SFF module as the final model.
    86	It can be seen that, compared with ‘Base’, ‘Base+SFD’
    87	can significantly increase the performance. This indicates that the modality-shared features are well separated
    88	from the unimodal features via SFD, which greatly reduces the modality discrepancy between visible and infrared images, thus benefiting VI-ReID. The results of
    89	‘Base+SFD+FMC’ indicates that the modality-specific features generated by using FMC contain much more discriminative person-related information for VI-ReID. Finally, the
    90	results of ‘Base+SFD+FMC+SFF’ shows that the proper
    91	
    92	exploitation of those existing and generated features further
    93	boosts the performance of VI-ReID.
    94	Verifying the effectiveness of each item in the proposed MD loss: As shown in Table 3, the modality-shared
    95	feature separation loss Lshs can significantly improve our
    96	model’s performance. This indicates that the Lshs can
    97	effectively reduce the modality discrepancy between the
    98	modality-shared visible and infrared features. Similarly,
    99	the modality-specific feature separation loss Lsps can further boost our model’s performance, by separating diffierent modality-specific features from each other according
   100	to their identities. Moreover, the decorrelation loss Ldc
   101	also increases our model’s performance. This means that
   102	the modality-shared and modality-specific features are well
   103	separated from each other with Ldc , which further benefits
   104	the subsequent missing modality-specific feature compensation and VI-ReID.
   105	Verifying the effectiveness of different loss functions
   106	in the proposed FMC module: In Table 4, ‘w/o FMC’
   107	means only using the decomposed modality-shared features
   108	for VI-ReID. ‘FMC’ means using FMC module to compensate missing modality-specific features. ‘FMC+LIC +LF C ’
   109	degrades rather than increases the performance. This indicates that, without using the adversarial loss, the missing modality-specific features cannot be well generated,
   110	thus leading to performance drops. On the contrary, with
   111	the adversarial loss (i.e., ‘FMC+LGAN ’), FMC ensures the
   112	similarity between the generated modality-specific features
   113	and real ones. With the proposed identity consistency loss
   114	LIC and feature consistency loss LF C , the discriminability of those generated modality-specific features is further
   115	enhanced, thus leading to performance improvements.
   116	
   117	5. Conclusion
   118	In this paper, our proposed FMCNet invests the featurelevel rather than image-level modality-specific information
   119	compensation for VI-ReID, which is achieved by using the
   120	proposed SFD, FMC and SFF modules. Compared with
   121	that in image level, the proposed modality-specific information compensation in feature level avoids the introduction
   122	of interfering information, and meanwhile is able to flexibly generate more discriminative person-related modalityspecific features, thus effectively boosting the performance
   123	of VI-ReID. The experimental results demonstrate that
   124	our approach significantly outperforms existing image-level
   125	modality-specific information compensation based models.
   126	Moreover, it even achieves better results than some SOTA
   127	modality-shared feature learning based models.
   128	Limitation and Societal Impact:
   129	The missing
   130	modality-specific features compensated by our model still
   131	lack spatial structure information, which may be futher beneficial for VI-ReID. Moreover, all used datasets are publicly
   132	available and involve no ethical issues.
   133	
   134	7346
   135	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.
   136	
   137	[14] M. Ye, J. Shen, G. Lin, T. Xiang, and S. Hoi, “Deep learning
   138	for person re-identification: A survey and outlook,” IEEE
   139	Transactions on Pattern Analysis and Machine Intelligence,
   140	vol. PP, no. 99, pp. 1–1, 2021. 2
   141	
   142	References
   143	[1] A. Wu, W.-S. Zheng, H.-X. Yu, S. Gong, and J. Lai, “Rgbinfrared cross-modality person re-identification,” in Proceedings of the IEEE International Conference on Computer
   144	Vision, 2017, pp. 5380–5389. 1, 3, 6
   145	
   146	[15] Q. Leng, M. Ye, and Q. Tian, “A survey of open-world person re-identification,” IEEE Transactions on Circuits and
   147	Systems for Video Technology, pp. 1092–1108, 2019. 2
   148	
   149	[2] M. Ye, Z. Wang, X. Lan, and P. C. Yuen, “Visible thermal
   150	person re-identification via dual-constrained top-ranking,” in
   151	Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, 2018, pp. 1092–1099. 1
   152	
   153	[16] D. Wu, S. J. Zheng, X. P. Zhang, C. A. Yuan, F. Cheng,
   154	Y. Zhao, Y. J. Lin, Z. Q. Zhao, Y. L. Jiang, and D. S. Huang,
   155	“Deep learning-based methods for person re-identification:
   156	A comprehensive review,” Neurocomputing, vol. 337, no.
   157	APR.14, pp. 354–371, 2019. 2
   158	
   159	[3] Y. Hao, N. Wang, L. Jie, and X. Gao, “HSME: Hypersphere manifold embedding for visible thermal person reidentification,” pp. 8385–8392, 2019. 1
   160	
   161	[17] M. Ye, J. Shen, and L. Shao, “Visible-infrared person reidentification via homogeneous augmented tri-modal learning,” IEEE Transactions on Information Forensics and Security, vol. 16, pp. 728–739, 2020. 3, 7
   162	
   163	[4] M. Ye, X. Lan, Z. Wang, and P. C. Yuen, “Bi-directional
   164	center-constrained top-ranking for visible thermal person reidentification,” IEEE Transactions on Information Forensics
   165	and Security, vol. 15, pp. 407–419, 2020. 1, 3
   166	[5] Z. Wei, X. Yang, N. Wang, and X. Gao, “Flexible body
   167	partition-based adversarial learning for visible infrared person re-identification.” IEEE Transactions on Neural Networks and Learning Systems, 2021. 1, 3
   168	
   169	[18] G. Wang, T. Zhang, J. Cheng, S. Liu, Y. Yang, and Z. Hou,
   170	“Rgb-infrared cross-modality person re-identification via
   171	joint pixel and feature alignment,” in Proceedings of the
   172	IEEE International Conference on Computer Vision, 2019,
   173	pp. 3623–3632. 3, 7
   174	
   175	[6] J. Sun, Y. Li, H. Chen, Y. Peng, X. Zhu, and J. Zhu, “Visibleinfrared cross-modality person re-identification based on
   176	whole-individual training,” Neurocomputing, vol. 440, pp.
   177	1–11, 2021. 1, 3
   178	
   179	[19] Z. Zhang, S. Jiang, C. Huang, Y. Li, and R. Y. Da Xu, “Rgbir cross-modality person reid based on teacher-student gan
   180	model,” arXiv preprint arXiv:2007.07452, 2020. 3, 7
   181	[20] Y. Yang, T. Zhang, J. Cheng, Z. Hou, P. Tiwari, H. M. Pandey
   182	et al., “Cross-modality paired-images generation and augmentation for RGB-Infrared person re-identification,” Neural Networks, vol. 128, pp. 294–304, 2020. 3
   183	
   184	[7] H. Liu, S. Ma, D. Xia, and S. Li, “Sfanet: A spectrum-aware
   185	feature augmentation network for visible-infrared person reidentification,” arXiv preprint arXiv:2102.12137, 2021. 1, 3,
   186	7
   187	
   188	[21] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning
   189	for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp.
   190	770–778. 4
   191	
   192	[8] Z. Wang, Z. Wang, Y. Zheng, Y. Y. Chuang, and S. Satoh,
   193	“Learning to reduce dual-level discrepancy for infraredvisible person re-identification,” in Proceedings of the IEEE
   194	Conference on Computer Vision and Pattern Recognition,
   195	2019, pp. 618–626. 2, 3, 7
   196	
   197	[22] D. Li, X. Wei, X. Hong, and Y. Gong, “Infrared-visible
   198	cross-modal person re-identification with an x modality,” in
   199	The Thirty-Fourth AAAI Conference on Artificial Intelligence
   200	(AAAI-20), 2020, pp. 4610–4617. 7
   201	
   202	[9] X. Fan, W. Jiang, H. Luo, and W. Mao, “Modality-transfer
   203	generative adversarial network and dual-level unified latent
   204	representation for visible thermal person re-identification,”
   205	The Visual Computer, pp. 1–16, 2020. 2, 3, 7
   206	
   207	[23] X. Hu and Y. Zhou, “Cross-modality person reid with maximum intra-class triplet loss,” in Pattern Recognition and
   208	Computer Vision, 2020, pp. 557–568. 7
   209	
   210	[10] S. Choi, S. Lee, Y. Kim, T. Kim, and C. Kim, “Hi-CMD:
   211	Hierarchical cross-modality disentanglement for visibleinfrared person re-identification,” pp. 10 257–10 266, 2020.
   212	2
   213	
   214	[24] X. Zhong, T. Lu, W. Huang, M. Ye, X. Jia, and C.-W. Lin,
   215	“Grayscale enhancement colorization network for visibleinfrared person re-identification,” IEEE Transactions on Circuits and Systems for Video Technology, pp. 1–1, 2021. 7
   216	
   217	[11] K. Kansal, A. Subramanyam, Z. Wang, and S. Satoh, “Sdl:
   218	Spectrum-disentangled representation learning for visibleinfrared person re-identification,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 10, pp.
   219	3422–3432, 2020. 2
   220	
   221	[25] X. Tian, Z. Zhang, S. Lin, Y. Qu, Y. Xie, and L. Ma,
   222	“Farewell to mutual information variational distiilation for
   223	cross-modal person re-identification,” in Computer Vision
   224	and Pattern Recognition, 2021, pp. 1522–1531. 7
   225	
   226	[12] Z. Feng, J. Lai, and X. Xie, “Learning modality-specific
   227	representations for visible-infrared person re-identification,”
   228	IEEE Transactions on Image Processing, vol. 29, pp. 579–
   229	590, 2019. 2, 3
   230	
   231	[26] H. Liu, Y. Chai, X. Tan, D. Li, and X. Zhou, “Strong but
   232	simple baseline with dual-granularity triplet loss for visiblethermal person re-identification,” IEEE Signal Processing
   233	Letters, vol. 28, pp. 653–657, 2021. 7
   234	
   235	[13] Y. Lu, Y. Wu, B. Liu, T. Zhang, B. Li, Q. Chu, and
   236	N. Yu, “Cross-modality person re-identification with sharedspecific transfer,” in Proceedings of the IEEE Conference on
   237	Computer Vision and Pattern Recognition, 2020, pp. 13 379–
   238	13 389. 2, 7
   239	
   240	[27] Y. Chen, L. Wan, Z. Li, Q. Jing, and Z. Sun, “Neural feature
   241	search for rgb-infrared person re-identification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and
   242	Pattern Recognition, 2021, pp. 587–597. 7
   243	
   244	7347
   245	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:18 UTC from IEEE Xplore. Restrictions apply.
   246	
   247	[28] C. Fu, Y. Hu, X. Wu, H. Shi, T. Mei, and R. He,
   248	“Cm-nas: Rethinking cross-modality neural architectures
   249	for visible-infrared person re-identification,” ArXiv, vol.
   250	abs/2101.08467, 2021. 7
   251	[29] Z. Sun, Y. Zhu, S. Song, J. Hou, S. Du, and Y. Song, “The
   252	multi-layer constrained loss for cross-modality person reidentification,” Proceedings of the International Conference
   253	on Artificial Intelligence and Signal Processing, pp. 1–6,
   254	2020. 7
   255	[30] G. Wang, T. Zhang, Y. Yang, J. Cheng, J. Chang, X. Liang,
   256	and Z. Hou, “Cross-modality paired-images generation for
   257	rgb-infrared person re-identification,” in Proceedings of
   258	the AAAI Conference on Artificial Intelligence, 2020, pp.
   259	12 144–12 151. 7
   260	[31] N. Dat, H. Hong, K. Ki, and P. Kang, “Person recognition

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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Focusing on pedestrians like human for clothes changing person re-identification.pdf' - | nl -ba | sed -n '49,181p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    49	1. Introduction
    50	Clothes changing person re-identiﬁcation (CC-ReID) (Li et al., 2023,
    51	2024b; Lin et al., 2024; Liu et al., 2024; Yang et al., 2019) is an extremely challenging image retrieval task. It aims to ﬁnd pedestrians with
    52	the same identity as a query, while those pedestrians have diﬀerent camera viewpoints and wear various clothes. The ensemble coding hypothesis in cognitive neuroscience (Michael et al., 2014) posits that humans
    53	achieve identity recognition among similar objects by collectively activating multiple local characteristics. Inspired by the ensemble coding
    54	hypothesis (Michael et al., 2014), it is worthwhile to explore a method
    55	that enhances multiple local characteristics holistically. This might address the gap in data augmentation methods for existing CC-ReID.
    56	Data augmentation methods for CC-ReID remain unexplored, with
    57	current approaches primarily focusing on network architecture design.
    58	We categorize these methods into three categories: local details learning methods (Huang et al., 2017; Sun et al., 2018; Zhao et al., 2017),
    59	identity-related learning methods (Gao et al., 2022; Hong et al., 2021;
    60	Jin et al., 2022; Li et al., 2021; Wang et al., 2022), and identity-irrelated
    61	
    62	learning methods (Han et al., 2023; Huang et al., 2021; Xu et al., 2021;
    63	Yang et al., 2023a). We posit that image-level learning (i.e., data augmentation) of local details can aid the network in discerning identityrelated from identity-irrelated information. This can further capture crucial identity information to diﬀerentiate pedestrians. As shown in Fig. 1,
    64	it can be observed that focusing on local details makes it easier to understand the identity-irrelated and identity-related information within each
    65	local region. Subsequently, by excluding identity-irrelated information
    66	(such as hat and jacket) and then capturing identity-related information
    67	(such as beard and pose) within the local region, identity recognition
    68	can be achieved more eﬀectively. Therefore, it is worthwhile to investigate an image processing method to improve local details.
    69	Identity-related learning methods typically involves recognizing
    70	pose (Bansal et al., 2022; Gao et al., 2022; Hong et al., 2021; Jin
    71	et al., 2022; Wang et al., 2022), 3D shape (Chen et al., 2021; Liu et al.,
    72	2023), and facial information (Wan et al., 2020; Xue et al., 2018). Typically, a multi-branch network architecture is utilized, with one branch
    73	dedicated to identiﬁcation, while the remaining branches focus on
    74	understanding identity-related information. Identity-irrelated learning
    75	
    76	∗ Corresponding author.
    77	
    78	E-mail addresses: panwj@stu.hqu.edu.cn (W. Pan), jqzhu@hqu.edu.cn (J. Zhu), 294781673@qq.com (X. Cui), zeng0043@hqu.edu.cn (H. Zeng),
    79	zybjy@mail.ustc.edu.cn (Y. Zhan).
    80	https://doi.org/10.1016/j.neunet.2025.107960
    81	Received 24 December 2024; Received in revised form 17 July 2025; Accepted 4 August 2025
    82	Available online 11 August 2025
    83	0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    84	
    85	Neural Networks 193 (2026) 107960
    86	
    87	W. Pan et al.
    88	
    89	Fig. 1. A woman matches celebrities wearing diﬀerent clothes by focusing on local details in an image.
    90	
    91	methods typically involves improving the understanding of clothes to
    92	reduce its interference. The methods encompass the design of clothes
    93	recognition modules (Huang et al., 2021, 2019; Xu et al., 2021; Yang
    94	et al., 2023a), dressing simulation modules (Han et al., 2023; Yu et al.,
    95	2020), and clothing adversarial loss (Gu et al., 2022). Both identityrelated and identity-irrelated learning methods excel in achieving pedestrian identiﬁcation with clothes changing. Furthermore, by enchancing
    96	local details their capabilities could be improved.
    97	Local details learning methods usually divide data into multiple body
    98	parts and then use well-designed multi-branch networks to learn local
    99	details of diﬀerent body parts. Considerations are made for the dividing
   100	method, such as automatic (Huang et al., 2017; Zhao et al., 2017) or
   101	manual (Huang et al., 2021, 2019; Sun et al., 2018; Wang et al., 2018), to
   102	divide the data into several body parts of the same size or diﬀerent sizes.
   103	In the design of multi-branch networks, weights for each branch are
   104	manually assigned (Huang et al., 2021, 2019) or automatically learned
   105	(Huang et al., 2017; Sun et al., 2018; Wang et al., 2018; Zhao et al.,
   106	2017), and the identity-related or identity-irrelated learning strategies
   107	discussed earlier are integrated. These methods are excellent, but focus
   108	on network design without considering the use of data augmentation to
   109	enhance local details.
   110	Real-world variations, such as angle, color, or scale changes, serve as
   111	basic data augmentation techniques across many tasks. With a diverse
   112	range of basic augmentation methods available, some strategies (Cubuk
   113	et al., 2019, 2020; Müller & Hutter, 2021) aim to automatically choose
   114	the appropriate techniques from this pool. Although some methods (Pan
   115	et al., 2023; Zhong et al., 2020, 2018) mitigate challenges such as occlusion (Zhong et al., 2020), lighting variations (Pan et al., 2023), and
   116	view variations (Zhong et al., 2018) common in conventional ReID tasks,
   117	CC-ReID presents additional complexities due to the clothes changing.
   118	To overcome clothing variation in CC-ReID, Pos-Neg (Jia et al., 2022)
   119	augments the training data by exchanging outﬁts between diﬀerent images from an identity-irrelated learning perspective. In addition to this
   120	approach, we argue that local detail enhancement also holds great potential. Identity-irrelated clothes occupies most of the image, whereas
   121	
   122	identity-related features like the face, gait, and posture are conﬁned to
   123	small local regions. By applying data augmentation at the image level,
   124	identity-related information can receive more attention, allowing subsequent networks to better learn identity cues embedded in local details.
   125	In this paper, we propose a humanoid focus-inspired image augmentation (HFIA) method for CC-ReID. In contrast to methods that use
   126	network architecture design to enhance local details, HFIA employs an
   127	image-based strategy. The HFIA divides images into ﬁve body components based on pedestrian silhouettes: head-shoulder, upper left torso,
   128	upper right torso, lower left torso, and lower right torso. HFIA comprises
   129	two key designs: the central emphasis strategy (CES) and the component
   130	continuity processing (CCP). The CES constructs an enlargement grid
   131	to scale the image, with a greater proportion of data near the center
   132	to simulate human visual attention focused on the central region. The
   133	CCP aligns the CES used for diﬀerent body components to ensure that
   134	all body components share a normalized vertical axis coordinate, while
   135	the left and right body components use mirrored horizontal axis coordinates. Subsequently, the CCP applies a smoothing post-processing to
   136	uniformly erase the discontinuities between the head-shoulder and upper left torso and upper right torso to produce a coordinated reassembled image. We evaluate HFIA on three public CC-ReID datasets (Gu
   137	et al., 2022; Huang et al., 2019; Yang et al., 2019), and it turns out to
   138	have state-of-the-art performance.
   139	The contributions of the paper are summarized as follows.
   140	•
   141	
   142	Based on the characteristic that human vision mainly focuses on central regions, we propose a central emphasis strategy (CES) to enhance
   143	local details in single regions by increasing the data proportion of
   144	central areas.
   145	• Based on the emsemble coding hypothesis in cognitive neuroscience,
   146	we propose component continuity processing (CCP), which applies
   147	CES to diﬀerent body regions according to human contours to
   148	achieve multi-region local detail enhancement.
   149	• By combining CES and CCP, we propose humanoid focus-inspired
   150	image augmentation (HFIA), the ﬁrst local detail learning data
   151	2
   152	
   153	Neural Networks 193 (2026) 107960
   154	
   155	W. Pan et al.
   156	
   157	2.2. Local details learning
   158	
   159	augmentation for CC-ReID tasks. Experiments show it achieves stateof-the-art performance on multiple CC-ReID benchmarks, and has
   160	generalization in knowledge distillation and unsupervised learning.
   161	
   162	The body part strategy (Hou et al., 2020; Suh et al., 2018; Wang
   163	et al., 2018; Xu et al., 2018; Zhao et al., 2017) is a predominant approach in improving identity recognition by learning local details in
   164	CC-ReID. It takes a single input data, splits it into several body parts,
   165	and outputs these parts for diﬀerent models. Fixed-location splitting
   166	strategies are commonly utilized. Wang et al. (2018) proposed a threebranch model which processes images with various levels of splitting-no
   167	splitting, two parts, and three parts-for part-based learning. Some approaches (Huang et al., 2021, 2019) perform manual splitting at the image level by feeding these body parts into independent models, and adjusting their weights manually to control the inﬂuence of each parts. PCB
   168	(Sun et al., 2018) employs body part strategy to process the output features of the backbone network for part alignment. Zhang et al. (2019a)
   169	performed body part learning on a ﬁne-grained semantic level. Some
   170	methods utilize learnable splitting strategies. DeepDiﬀ (Huang et al.,
   171	2017) develops a multi-branch model that utilize two automated splitting strategies to produce overlapping body parts of diﬀerent heights.
   172	Zhao et al. (2017) introduced SpindleNet, a method that employs a
   173	tree-structured fusion of multi-stage features to generate parts of various sizes. These approaches employ diverse splitting strategies and intricate models to eﬀectively utilize the local details within body parts
   174	with excellent results. However, there has been limited exploration of
   175	data augmentation methods for enhancing local details.
   176	
   177	The remainder of this paper is organized as follows. Section 2 surveys
   178	recent work related to this paper. Section 3 describes our method in
   179	detail. Section 4 presents experimental results and analysis to show the
   180	superiority of our method. Section 5 concludes this paper.
   181	

exec
/bin/zsh -lc "pdfinfo 'Focusing on pedestrians like human for clothes changing person re-identification.pdf' | sed -n '1,35p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Focusing on pedestrians like human for clothes changing person re-identification
Subject:         Neural Networks, 193 (2026) 107960. doi:10.1016/j.neunet.2025.107960
Keywords:        Re-identification,Clothes changing,Human focus
Author:          Wenjie Pan
Creator:         Elsevier
Producer:        Acrobat Distiller 8.1.0 (Windows)
CreationDate:    Thu Dec 11 14:38:26 2025 CST
ModDate:         Thu Dec 11 15:18:54 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           14
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       11374207 bytes
Optimized:       yes
PDF version:     1.7

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Focusing on pedestrians like human for clothes changing person re-identification.pdf' - | nl -ba | sed -n '273,662p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   273	3. Proposed method
   274	The proposed humanoid focus-inspired image augmentation (HFIA)
   275	is a data augmentation method for CC-ReID. HFIA enhances details
   276	in the central region using the central emphasis strategy (CES) and
   277	achieves targeted enhancement of multiple pedestrian characteristics
   278	using the component continuity processing (CCP).
   279	
   280	|𝑖|
   281	
   282	𝑓𝑦 + 𝑖⋅8
   283	
   284	3
   285	
   286	𝐻𝑖𝑏 ← 𝐻 𝑝 ⋅ (
   287	
   288	4
   289	
   290	𝑊𝑖𝑏 ← 𝑊 𝑝 ⋅ (
   291	
   292	2
   293	|𝑖|
   294	𝑓𝑥 + 𝑖⋅8
   295	
   296	⋅ 𝛿(𝑖 ≠ 0) + 28 ⋅ 𝛿(𝑖 = 0))
   297	
   298	2
   299	
   300	⋅ 𝛿(𝑖 ≠ 0) + 28 ⋅ 𝛿(𝑖 = 0))
   301	
   302	// Step 2: Divide and enlarge blocks.
   303	{𝑠0 , 𝑠1 , 𝑠2 } ← {2.0, 1.5, 1.0}
   304	7 𝑏ℎ ← 0
   305	8 for 𝑖 ← −2 to 2 do
   306	9
   307	𝑏𝑤 ← 0
   308	10
   309	for 𝑗 ← −2 to 2 do
   310	11
   311	𝑥𝑏𝑖𝑗 ← 𝑥𝑝 [∶, 𝑏ℎ ∶ 𝑏ℎ + 𝐻𝑖𝑏 , 𝑏𝑤 ∶ 𝑏𝑤 + 𝑊𝑗𝑏 ]
   312	5
   313	6
   314	
   315	3.1. Ensemble coding hypothesis
   316	The ensemble coding hypothesis (Michael et al., 2014) suggests that
   317	“object recognition results from activation across complex feature detectors.” For example, in recognizing granny, some detectors are activated
   318	by body shape, some by hair color, and some by facial features. Recognition is not due to one unit but to the collective activation of many
   319	units. We believe there are two key aspects of the ensemble coding hypothesis. The ﬁrst is the activation of local features, based on which we
   320	design CES to emphasize local regions. The second is the collective activation of multiple units, upon which we develop CCP to simultaneously
   321	emphasize multiple regions.
   322	
   323	12
   324	
   325	𝑦𝑓𝑖𝑗 ← Resize(𝑥𝑏𝑖𝑗 , 𝐻𝑖𝑏 ⋅ 𝑠|𝑖| , 𝑊𝑗𝑏 ⋅ 𝑠|𝑗| )
   326	
   327	13
   328	
   329	𝑏𝑤 ← 𝑏𝑤 + 𝑊𝑗𝑏
   330	
   331	14
   332	
   333	𝑏ℎ ← 𝑏ℎ + 𝐻𝑖𝑏
   334	
   335	15
   336	
   337	// Step 3: Reassemble and reduce the body component.
   338	
   339	16
   340	
   341	𝑦𝑝 ← Reassemble({𝑦𝑓𝑖𝑗 })
   342	
   343	𝑦𝑝 ← Resize(𝑦𝑝 , 𝐻 𝑝 , 𝑊 𝑝 )
   344	𝑝
   345	18 return 𝑦
   346	17
   347	
   348	3.2. Central emphasis strategy
   349	Fig. 2 illustrates the central emphasis strategy (CES) utilized in the
   350	HFIA, focusing on the central area of a body component. For a given
   351	𝑝
   352	𝑝
   353	body component 𝑥𝑝 ∈ ℝ3×𝐻 ×𝑊 (𝐻 𝑝 and 𝑊 𝑝 denote the height and
   354	𝑝
   355	width of 𝑥 , respectively) and a normalized reference point (𝑓𝑥 , 𝑓𝑦 ). The
   356	algorithm of the CES is described as Algorithm 1, detailed as follows.
   357	Firstly, the CES generates an enlargment grid with 5 × 5 grids. At
   358	the beginning of the generation, a central grid is ﬁrst determined. The
   359	
   360	height and width of the central grid are proportional to 2∕8 of 𝑥𝑝 , with
   361	its geometric center aligned with the given reference point (𝑓𝑥 , 𝑓𝑦 ). Subsequently, areas to the left, right, above, and below the central grid are
   362	evenly divided into two sections, forming 5 × 5 grids, termed the enlargement grid. Taking the left side as an example, the widths of grids
   363	in the ﬁrst and second columns adjacent to the central grid are identi4
   364	
   365	Neural Networks 193 (2026) 107960
   366	
   367	W. Pan et al.
   368	
   369	Algorithm 2: Component continuity processing.
   370	Input: Image tensor 𝑥 ∈ ℝ3×𝐻×𝑊 .
   371	Output: Focused image tensor 𝑦 ∈ ℝ3×𝐻×𝑊 .
   372	1 // Step 1: Split body components.
   373	𝑝
   374	2 𝑥 ← 𝑥[∶, 0 ∶ 2∕8 ⋅ 𝐻, 0 ∶ 𝑊 ]
   375	0
   376	𝑝
   377	3 𝑥 ← 𝑥[∶, 2∕8 ⋅ 𝐻 ∶ 5∕8 ⋅ 𝐻, 0 ∶ 𝑊 ∕2]
   378	1
   379	𝑝
   380	4 𝑥 ← 𝑥[∶, 2∕8 ⋅ 𝐻 ∶ 5∕8 ⋅ 𝐻, 𝑊 ∕2 ∶ 𝑊 ]
   381	2
   382	𝑝
   383	5 𝑥 ← 𝑥[∶, 5∕8 ⋅ 𝐻 ∶ 𝐻, 0 ∶ 𝑊 ∕2]
   384	3
   385	𝑝
   386	6 𝑥 ← 𝑥[∶, 5∕8 ⋅ 𝐻 ∶ 𝐻, 𝑊 ∕2 ∶ 𝑊 ]
   387	4
   388	7 // Step 2: Determine the reference point and apply the CESs.
   389	8 repeat
   390	9
   391	𝑓𝑥 ←  (1∕2, (1∕8)2 )
   392	10
   393	𝑓𝑦 ←  (1∕2, (1∕8)2 )
   394	11 until 1∕8 ≤ 𝑓𝑥 ≤ 7∕8 and 1∕8 ≤ 𝑓𝑦 ≤ 7∕8;
   395	12 for 𝑘 ← 0 to 4 do
   396	13
   397	if k%2 = 0 then
   398	14
   399	𝑦𝑝𝑘 ← CES(𝑥𝑝𝑘 , (𝑓𝑥 , 𝑓𝑦 ))
   400	15
   401	
   402	else
   403	𝑦𝑝𝑘 ← CES(𝑥𝑝𝑘 , (1 − 𝑓𝑥 , 𝑓𝑦 ))
   404	
   405	16
   406	
   407	Fig. 3. Comparison of (a) CES without enlargement grid with (b) CES with
   408	enlargement grid. We mark the discontinuity by a red circle.
   409	
   410	// Step 3: Reassemble and smoothing post-processing.
   411	𝑦 ← Reassemble({𝑦𝑝𝑘 })
   412	𝑝
   413	𝑝
   414	19 ℎ𝑡 ← Random(𝐻 − 𝐻∕16, 𝐻 )
   415	0
   416	0
   417	𝑝
   418	𝑝
   419	20 ℎ𝑏 ← Random(𝐻 , 𝐻 + 𝐻∕16)
   420	0
   421	0
   422	21 {𝜎0 , 𝜎1 , 𝜎2 } ← {0.485, 0.456, 0.406}
   423	22 for 𝑘 ← 0 to 2 do
   424	23
   425	𝑦[𝑘, ℎ𝑡 ∶ ℎ𝑏 , ∶] ← 𝜎𝑘
   426	17
   427	18
   428	
   429	cal. Furthermore, the sum of widths in the ﬁrst columns on the left and
   430	right sides is equal to 3∕8 of 𝑥𝑝 , and likewise for the second columns.
   431	Secondly, based on the enlargement grid, the CES divides 𝑥𝑝 into 25
   432	blocks and enlarges the edges of each block. Regarding the degree of
   433	enlargement, the CES deﬁnes a set of enlargement degrees {𝑠0 , 𝑠1 , 𝑠2 }
   434	and enlarges the edges of the grid accordingly based on these enlargement degrees. Regarding the position of enlargement, the enlargement
   435	degree for the width/height of the grids in the same column/row as
   436	the central grid is 𝑠0 , while for the grids in the ﬁrst row/column away
   437	from the center, it is 𝑠1 , and for the second row/column, it is 𝑠2 . Ultimately, any two adjacent blocks will have the same enlargement degree
   438	on their adjacent edges, ensuring seamless reassembling after enlargement. Finally, the 25 enlarged blocks are reassembled and reduced to
   439	the size of 𝑥𝑝 to obtain a focused body component 𝑦𝑝 .
   440	The proposed enlargement grid aims to increase the proportion of the
   441	central area. We present an alternative method to illustrate the beneﬁts
   442	of our enlargement grid, as shown in Fig. 3(a). This method divides the
   443	image into three regions, with the region near the center experiencing
   444	a higher enlargement degree, and the enlarged images are overlapped.
   445	However, due to diﬀerent oﬀsets of the enlarged images in diﬀerent
   446	regions, this method results in signiﬁcant discontinuities in local areas,
   447	leading to misinterpretations of the human body structure by the model.
   448	To address this issue, it is necessary to ensure that the data in the same
   449	row (column) maintain the same oﬀset before and after enlargement. We
   450	ﬁnd that if the data in the same row (column) have the same enlargement degree along the vertical (horizontal) axis, the relative positions
   451	of the row (column) data along the vertical (horizontal) axis remain unchanged. As illustrated in Fig. 3(b), based on this ﬁnding, we divide the
   452	image into ﬁve rows and ﬁve columns, ensuring that the data in the same
   453	row/column have the same enlargement degree, ultimately preserving
   454	the relative positions before and after enlargement.
   455	Considering that human visual attention primarily focuses on a central region, we propose the CES to increase the proportion of data in
   456	the central area. This indirectly reallocates the learning resources of the
   457	network, with resources originally allocated to the outer regions being
   458	redistributed to the central area, ultimately making it easier to understand local details in the central region. Nonetheless, CC-ReID is characterized by substantial misleading cues such as clothes, in which case
   459	it is imperative to identify a region that is more related to pedestrian
   460	identity.
   461	
   462	24
   463	
   464	return 𝑦
   465	
   466	3.3. Component continuity processing
   467	Fig. 4 depicts the component continuity processing (CCP) employed
   468	by HFIA to focus on multiple body components in pedestrian image tensor 𝑥 ∈ ℝ3×𝐻×𝑊 , where 𝐻 and 𝑊 denote the height and width of 𝑥,
   469	respectively. The algorithm for the CCP, presented in Algorithm 2, is
   470	detailed below.
   471	First, the CCP divides the image into ﬁve body components, namely
   472	head-shoulder, upper left torso, upper right torso, lower left torso, and
   473	lower right torso. Speciﬁcally, the head-shoulder region occupies the
   474	topmost part of the image, measuring 2∕8 ⋅ 𝐻 in height and spanning
   475	the entire width, denoted by 𝑊 . The remaining image area is evenly divided among the other four body components, each having a height of
   476	3∕8 ⋅ 𝐻 and a width of 1∕2 ⋅ 𝑊 . Second, the CCP employs Gaussian distribution functions  to determine normalized coordinates, known as
   477	the reference point (𝑓𝑥 , 𝑓𝑦 ), which tends to be positioned closer to the
   478	center of the body component. Subsequently, ﬁve CESs are generated
   479	and applied to the ﬁve body components respectively. In particular, the
   480	reference points used for the CESs corresponding to the left and right
   481	body components are adjusted to achieve mirroring. Finally, the CCP
   482	reassembles the ﬁve body components processed by the CES to their initial positions. Subsequently, the CCP applies smoothing post-processing
   483	to the reassembled image utilizing the mean values of normalization.
   484	This involves erasing random regions between the head-shoulder component and the components of the left upper torso and the right upper
   485	torso using {0.485, 0.456, 0.406}.
   486	According to the ensemble coding hypothesis in cognitive neuroscience (Michael et al., 2014), humans identify similar objects by activating multiple local characteristics collectively. Therefore, we investigate the regions in the CC-ReID images (Gu et al., 2022; Huang et al.,
   487	2019; Yang et al., 2019) that may contribute to identity recognition and
   488	applied two priors to the CCP. First, most pedestrian images are captured
   489	in standing posture, with the head at the top of the image. Therefore,
   490	5
   491	
   492	Neural Networks 193 (2026) 107960
   493	
   494	W. Pan et al.
   495	
   496	Fig. 4. Visualization of the component continuity processing (CCP) computation.
   497	
   498	Furthermore, given the bilateral symmetry of the human body, mirrored
   499	reference points were assigned to the left and right body components.
   500	Ultimately, all reference points for body components have the same vertical oﬀset, with those on the left body components mirroring those on
   501	the right, as illustrated in Fig. 5(b). Nevertheless, due to the uneven
   502	widths of the head-shoulder body component compared to the upper
   503	left and upper right torsos, the CCP cannot align them in a uniform and
   504	horizontal manner. Therefore, inspired by the attention mask of SwinTransformer (Liu et al., 2021), we devise a smoothing post-processing
   505	method to erase random regions with mean values of normalization,
   506	as depicted in Fig. 5(c). Finally, we align the four torso body components on the computation level and alleviate the discontinuity between
   507	the head-shoulder and upper left torso, and the upper right torso at the
   508	data level, so as to achieve coordinated reassembly.
   509	In summary, by means of the prior information from CC-ReID, the
   510	CCP can emphasize several regions where identity-related information is
   511	more frequent, thus enhancing the local details of these identity-related
   512	areas.
   513	3.4. Humanoid focus-inspired image augmentation
   514	
   515	Fig. 5. Results of the HFIA under three strategies: (a) directly reassembling,
   516	(b) aligning body components by applying correlative reference points to ﬁve
   517	CESs, (c) smoothing post-processing. We mark the reference points of the CES,
   518	emphasizing the two locations of distinctly discontinuous data in the directly
   519	reassembling strategy.
   520	
   521	Following the conventional data augmentation method and our humanoid focus-inspired image augmentation (HFIA), the images are then
   522	inputted into the baseline model. The construction of baseline is illustrated in Fig. 6. Initially, the image is passed through the backbone
   523	network, speciﬁcally ResNet50 (He et al., 2016). After passing through
   524	a global pooling layer and a batch normalization (BN) layer (Ioﬀe &
   525	Szegedy, 2015; Luo et al., 2019), the features 𝑔 𝑏 are fed to both an identity classiﬁer and a clothes classiﬁer, culminating in loss calculation.
   526	The features 𝑔 𝑘 extracted by the identity classiﬁer undergo identity loss
   527	computation. Similarly, the features 𝑔 𝑐 extracted by the clothing classiﬁer undergo both clothes classiﬁcation loss and clothes adversarial loss
   528	computations. In this paper, the identiﬁcation loss adopts label smoothing cross-entropy loss 𝐿𝑆 (see Eq. (1)), the clothes classiﬁcation loss
   529	adopts cross-entropy loss 𝐶𝐸 (see Eq. (2)), and the clothes adversarial
   530	loss adopts clothes-based adversarial loss 𝐶𝐴𝐿 (Gu et al., 2022) (see
   531	Eq. (3)).
   532	
   533	we divide the body into ﬁve components based on the human skeleton structure: head-shoulder, upper left torso, upper right torso, lower
   534	left torso, and lower right torso. Second, within each body component,
   535	the areas containing identity-related information are mainly distributed
   536	around the component’s center. For example, the center of the headshoulder component contains facial information, while centers of the
   537	other four torso components contain posture-related information (such
   538	as joints and gait). Thus, we utilize Gaussian distribution functions to
   539	select the position of reference points, ensuring that they fall within
   540	the component centers more frequently, termed the pedestrian-oriented
   541	conﬁguration.
   542	After CES processing, the size of ﬁve body components remains unchanged, allowing the CCP to directly reassemble the ﬁve body components. However, if irrelated reference points are used for the ﬁve body
   543	components, discontinuities may occur at the connections, as shown in
   544	Fig. 5(a). These discontinuities arise from diﬀerent oﬀsets of diﬀerent
   545	body components after processing, which can be addressed through collaborative enlargement of adjacent data, as discussed in Section 3.2.
   546	
   547	𝑘
   548	
   549	𝐿𝑆 = −
   550	
   551	𝑁
   552	∑
   553	
   554	𝑤𝐿𝑆 (𝑖) ⋅ log(𝑝(𝑖|𝑔 𝑘 ))
   555	
   556	𝑖=1
   557	
   558	𝑤𝐿𝑆 (𝑖) =
   559	6
   560	
   561	{
   562	
   563	1 − 𝜖,
   564	
   565	𝑖 ∈ 𝑆𝑘
   566	
   567	𝜖
   568	,
   569	𝑁𝑘
   570	
   571	𝑖 ∉ 𝑆𝑘
   572	
   573	(1)
   574	
   575	Neural Networks 193 (2026) 107960
   576	
   577	W. Pan et al.
   578	
   579	Fig. 6. The architecture of baseline.
   580	
   581	𝐶𝐸 = −
   582	
   583	𝑁𝑐
   584	∑
   585	
   586	from three diﬀerent camera perspectives: camera A, B and C. While
   587	individuals wear identical clothes on cameras A and B, they are photographed in distinct rooms. Conversely, in camera C, individuals wear
   588	diﬀerent clothes, the images are captured on separate occasions. The
   589	PRCC dataset is randomly split into three subsets: a training set (150
   590	identities with 17,896 images), a validation set (150 identities with
   591	5002 images), and a testing set (71 identities with 10,800 images). For
   592	clothes changing evaluation, in the testing set, 3384 images captured
   593	from camera A are allocated as a gallery set and 3543 images captured
   594	from camera C are allocated as a query set.
   595	Celeb-ReID (Huang et al., 2019) is an image-based dataset consisting
   596	of 1052 identities with 34,185 images collected from the Internet. All
   597	images depict celebrities photographed on public streets. These celebrities are of diverse nationalities and age groups, with 53.14 % of males
   598	and 46.86 % of females. On average, each person wears a diﬀerent outﬁt
   599	in more than 70 % of the images. Celeb-ReID is partitioned into a training set and a testing set. The training set comprises 632 identities with
   600	20,208 images. The testing set includes a query set (420 identities with
   601	2972 images) and a gallery set (420 identities with 11,006 images).
   602	CCVID (Gu et al., 2022) is a video-based dataset comprising 226
   603	identities, 2856 tracklets, and 347,833 images. Derived from the FVG
   604	gait recognition dataset (Zhang et al., 2019b), CCVID encompasses 2856
   605	sequences, varying in length from 27 to 410 frames, with an average
   606	duration of 122 frames per sequence. The dataset provides ﬁne-grained
   607	clothing labels, including tops, bottoms, and shoes. A total of 75 identities are assigned to the training set, while the remaining 151 identities form the testing set. Within the testing set, 834 sequences serve as
   608	queries, while the remaining 1074 sequences comprise the gallery set.
   609	
   610	𝑤𝐶𝐸 (𝑖) ⋅ log(𝑝(𝑖|𝑔 𝑐 ))
   611	
   612	𝑖=1
   613	
   614	𝑤𝐶𝐸 (𝑖) =
   615	
   616	{
   617	1 − 𝜖,
   618	
   619	𝑖 ∈ 𝑆𝑐
   620	
   621	0,
   622	
   623	𝑖 ∉ 𝑆𝑐
   624	
   625	(2)
   626	
   627	𝑐
   628	
   629	𝐶𝐴𝐿 = −
   630	
   631	𝑁
   632	∑
   633	
   634	𝑤𝐶𝐴𝐿 (𝑖) ⋅ log(𝑝(𝑖|𝑔 𝑐 ))
   635	
   636	𝑖=1
   637	
   638	⎧1 − 𝜖 + 𝜖 𝑐 , 𝑖 ∈ 𝑆 𝑐 and 𝑖 ∈ 𝑆 𝑘
   639	𝑁
   640	⎪
   641	𝑤𝐶𝐴𝐿 (𝑖) = ⎨ 𝜖𝑐 ,
   642	𝑖 ∉ 𝑆 𝑐 and 𝑖 ∈ 𝑆 𝑘
   643	𝑁
   644	⎪
   645	0,
   646	𝑖 ∉ 𝑆𝑘
   647	⎩
   648	
   649	(3)
   650	
   651	where 𝑁 𝑘 and 𝑁 𝑐 denotes the number of identiﬁcation and clothes, respectively; 𝑔 𝑘 and 𝑔 𝑐 are the identity classiﬁer’s feature and clothes classiﬁer’s feature of the sample, respectively; 𝜖 is a smoothing parameter
   652	which conﬁgured to 0.1 in this paper; 𝑆 𝑘 and 𝑆 𝑐 are the truth identiﬁcation label and the truth cloth label of the sample, respectively.
   653	HFIA, as a data augmentation method, has three advantages compared to the existing designs for network in CC-ReID. First, by enhancing local details in images, the HFIA enables each network structure
   654	to learn pedestrian information embedded in these details. Second, the
   655	HFIA is only activated during training, ensuring it imposes no computational burden during inference, thus eﬀectively aiding practical deployment. Finally, the HFIA is the ﬁrst local detail data augmentation
   656	method speciﬁcally designed for CC-ReID, which may boost the development of more augmentation techniques.
   657	HFIA, as an image-only data augmentation technique, is applicable
   658	to most computer vision tasks. Its local detail enhancement is implemented by amplifying central data, resulting in data shifts. In localization tasks (e.g., object detection and semantic segmentation), such shifts
   659	lead to mismatches between data and spatial labels, consequently misguiding model training. Accordingly, HFIA is unsuitable for localization
   660	tasks.
   661	
   662	4.2. Setup

codex
第二篇把“换衣 ReID”从常见的去衣服、学身份分支，重讲成“局部细节学习也可以在图像层面做，而且训练期增强就够”。我会补它的增强对比、泛化和消融，这些决定它是不是只卖了一个花哨增强。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 13 'Focusing on pedestrians like human for clothes changing person re-identification.pdf' - | nl -ba | rg -n \"4\\.4|4\\.5|4\\.6|Ablation|ablation|Comparison with data augmentation|Generalization|CES|CCP|HFIA|Table|Fig|Visualization|analysis\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2-     2	
3-     3	W. Pan et al.
4-     4	
5:     5	Table 1
6:     6	Comparison of HFIA with SOTA methods on PRCC (Yang et al., 2019), Celeb-ReID (Huang et al., 2019), and CCVID (Gu et al., 2022). Here, LD, IR, and
7-     7	II denote local details learning method, identity-related learning method, and identity-irrelated learning method, respectively. The metric is R1.
8-     8	Methods
9-     9	
--
75-    75	
76-    76	37.1
77-    77	N/A
78:    78	54.4
79-    79	52.1
80-    80	N/A
81-    81	51.6
--
101-   101	N/A
102-   102	85.4
103-   103	
104:   104	HFIA-ZUT
105:   105	HFIA
106-   106	
107-   107	Ours
108-   108	Ours
--
124-   124	between the query set and gallery set data determines the relevance of
125-   125	their identities.
126-   126	
127:   127	dataset PRCC (Yang et al., 2019), our HFIA, solely focusing on local
128-   128	detail learning, outperforms AIM (Yang et al., 2023a), a hybrid method
129-   129	combining local detail learning with identity-irrelated learning, by 2.5 %
130-   130	in the R1 metric (60.4 % vs 57.9 %).
131:   131	In summary, HFIA achieved state-of-the-art performance on three
132-   132	public CC-ReID datasets, validating its eﬀectiveness. Furthermore, the
133:   133	superiority of HFIA suggests that enhancing local details at the image level remains eﬀective and comparable to local feature-level detail
134-   134	learning methods.
135-   135	
136-   136	4.3. Comparison with state-of-the-art methods
137:   137	We compare our HFIA with the state-of-the-art methods on three
138-   138	public CC-ReID datasets: PRCC (Yang et al., 2019), Celeb-ReID (Huang
139:   139	et al., 2019), and CCVID (Gu et al., 2022). As shown in Table 1, our HFIA
140-   140	performs better than other state-of-the-art methods. For example, based
141-   141	on the image-based dataset PRCC (Yang et al., 2019), our approach outperforms IRM (He et al., 2023) (published in CVPR 2024). Speciﬁcally,
142-   142	it achieves a superiority of 6.2 % in the R1 metric (60.4 % vs 54.2 %).
143-   143	Based on the video-based dataset CCVID (Gu et al., 2022), our approach
144-   144	outperforms SEMI (Nguyen et al., 2024) (published in WACV 2024).
145-   145	Speciﬁcally, it outperforms by 1.7 % in terms of the R1 metric (84.2 %
146:   146	vs 82.5 %). The comparison results demonstrate that our HFIA exhibits
147:   147	excellent performance in the CC-ReID task. Moreover, HFIA achieves
148-   148	superior performance compared to existing augmentation strategies for
149-   149	CC-ReID. On PRCC, it exceeds the Pos-Neg (Jia et al., 2022) baseline by
150-   150	5.5 % in R1 (60.4 % vs 54.9 %).
151-   151	To more comprehensively validate the eﬀectiveness of our method,
152:   152	ablation studies are conducted on the latest state-of-the-art method, ZUT
153:   153	(Pan et al., 2025) (published in AAAI 2025). All experimental conﬁgurations are kept consistent with those of ZUT. As shown in Table 1, it can
154:   154	be observed that HFIA improves the performance of ZUT across multiple
155:   155	datasets. For example, on PRCC, the introduction of HFIA improves ZUT
156:   156	by 0.7 % in R1. On Celeb-ReID, HFIA increases ZUT by 0.5 % in R1. On
157:   157	CCVID, HFIA enhances ZUT by 0.9 % in R1. These results demonstrate
158:   158	the eﬀectiveness of HFIA under diﬀerent frameworks.
159-   159	These comparative ﬁndings provide additional insights. Firstly, local
160-   160	detail learning methods indirectly enhance the model’s grasp of information, reaching performance levels similar to that of directly learning
161-   161	identity-related or identity-irrelated information. On the image-based
162:   162	Celeb-ReID dataset (Huang et al., 2019), our HFIA outperforms the
163-   163	identity-irrelated learning method AFDNet (Xu et al., 2021) by 3.1 %
164-   164	in R1 (55.2 % vs 52.1 %). Compared to identity-related learning method
165:   165	ACID (Yang et al., 2023b), our HFIA wins it by 2.7 % in R1 (55.2 %
166-   166	vs 52.5 %). Secondly, image-level methods for learning local details exhibit comparable eﬃcacy to feature-level approaches. For example, on
167:   167	the video-based dataset CCVID (Gu et al., 2022), our HFIA, employing image-level local detail processing, outperforms the feature-level local detail method TCLNet (Hou et al., 2020) by 3.5 % in the R1 metric (84.2 % vs 80.7 %). Lastly, pure local detail learning methods still
168-   168	hold signiﬁcantly untapped potential. For example, on the image-based
169-   169	
170:   170	4.4. Comparison with data augmentation
171:   171	We compare our HFIA with data augmentation methods on PRCC
172-   172	dataset (Yang et al., 2019), including random cropping, random vertical ﬂip, random rotation, random perspective, color jitter, AutoAugment (Cubuk et al., 2019), RandAugment (Cubuk et al., 2020), and TrivialAugment (Müller & Hutter, 2021).
173:   173	The experimental results are illustrated in Fig. 7. It can be observed
174:   174	that our HFIA achieved the best performance, surpassing AutoAugment
175-   175	(Cubuk et al., 2019), the second-ranked method, by 0.3 % in the R1
176-   176	metric, demonstrating the eﬀectiveness of our approach. Furthermore,
177-   177	several data augmentation techniques have adversely aﬀected performance, particularly random cropping (reduced R1 by 4.9 %) and random rotation (reduced R1 by 10.0 %). This may be attributed to the fact
178-   178	that crucial identity information in CC-ReID resides in a few localized
179-   179	regions (such as the head and gait), and random cropping and rotation
180-   180	lead to the loss of image information. This indirectly underscores the necessity of designing data augmentation methods tailored for CC-ReID.
181:   181	4.5. Generalization ability of HFIA
182-   182	Local details learning is an important component of various ReID
183-   183	subtasks. For example, in knowledge distillation, it is worth exploring which details the teacher model focuses on more, and in unsupervised learning, which details are more important. Therefore, we explore
184:   184	the generalization capability of HFIA on other ReID subtasks, including knowledge distillation and unsupervised learning. All experiments
185:   185	are conducted by integrating HFIA as a data augmentation method into
186-   186	the open-source codebases of D3still (Xie et al., 2024) and FCM (Li
187-   187	et al., 2024a). The datasets includes MSMT17 (Wei et al., 2018) and
188-   188	Market1501 (Zheng et al., 2015). MSMT17 (Wei et al., 2018) contains
189-   189	126,441 images of 4101 identities. Market1501 (Zheng et al., 2015)
190-   190	comprises 32,668 images of 1501 identities.
191:   191	4.5.1. Knowledge distillation
192-   192	The experiments are conducted based on the D3still (Xie et al., 2024),
193-   193	with conﬁgurations consistent with the source code of D3Still. Speciﬁ8
194-   194	
--
196-   196	
197-   197	W. Pan et al.
198-   198	
199:   199	Fig. 7. Performance comparison between our HFIA and data augmentation methods on PRCC (Yang et al., 2019). Here, RandCrop denotes random cropping;
200-   200	RandVertFlip denotes random vertical ﬂip; RandRotation denotes random rotation; RandPerspective denotes random perspective; ColorJitter denotes color jitter;
201-   201	AutoAug denotes AutoAugment (Cubuk et al., 2019); RandAug denotes RandAugment (Cubuk et al., 2020); TriAug denotes TrivialAugment (Müller & Hutter, 2021).
202:   202	Table 2
203-   203	Comparison with SOTA knowledge distillation methods on the MSMT17 (Wei et al., 2018) dataset. The metric is R1.
204-   204	Methods
205-   205	
--
225-   225	
226-   226	Baseline
227-   227	
228:   228	HFIA
229-   229	
230-   230	Baseline
231-   231	
232:   232	HFIA
233-   233	
234-   234	56.9
235-   235	57.2
--
259-   259	49.3
260-   260	54.0
261-   261	
262:   262	Table 3
263-   263	Comparison with SOTA unsupervised learning methods on the MSMT17 (Wei et al., 2018) and Market1501
264-   264	(Zheng et al., 2015) dataset.
265-   265	Methods
--
267-   267	Reference
268-   268	
269-   269	FCM (Li et al., 2024a)
270:   270	HFIA-FCM
271-   271	
272-   272	AAAI 2024
273-   273	Ours
--
301-   301	a multiplicative factor of 0.1. Data augmentation include random horizontal ﬂipping and random erasing. The training images are resized to
302-   302	256 × 128, padded with 10 pixels, and then randomly cropped back to
303-   303	256 × 128. The test images are resized to 256 × 128.
304:   304	The experimental results are shown in Table 3. It can be seen that
305:   305	HFIA achieves performance gains across two datasets. For example, on
306:   306	MSMT17 (Wei et al., 2018), HFIA brings a 1.6 % improvement in R1
307-   307	over the baseline. Similarly, on Market1501 (Zheng et al., 2015), the
308:   308	introduction of HFIA improves the baseline by 0.5 % in R1.
309-   309	
310-   310	cally, ResNet101 (He et al., 2016) is used as the teacher model, while
311-   311	ResNet18 (He et al., 2016) and MobileNetV3-small (Howard et al., 2019)
--
316-   316	factor of 0.1. Data augmentation include random horizontal ﬂipping
317-   317	and random erasing. Teacher images are resized to 320 × 160, padded
318-   318	with 8 pixels, and then randomly cropped back to 320 × 160. Student images are resized to 160 × 80, padded with 4 pixels, and then randomly
319:   319	cropped back to 160 × 80. The student network is trained using crossentropy and triplet loss. HFIA is applied for data augmentation in the
320-   320	teacher network.
321:   321	The experimental results are shown in Table 2. It can be seen that
322:   322	HFIA achieves performance gains across two student networks and six
323-   323	distillation methods. For example, when ResNet18 (He et al., 2016) is
324-   324	used as the student network and D3still (Xie et al., 2024) is adopted as
325:   325	the distillation method, the introduction of HFIA improves the baseline
326-   326	by 1.2 % in R1. Similarly, with MobileNetV3-small (Howard et al., 2019)
327-   327	as the student network and CSD (Wu et al., 2022) as the distillation
328:   328	method, HFIA brings a 1.2 % improvement in R1 over the baseline.
329-   329	
330:   330	4.6. Ablation study
331:   331	We conduct ablation studies of HFIA on PRCC (Yang et al., 2019),
332-   332	Celeb-ReID (Huang et al., 2019), and CCVID (Gu et al., 2022), as shown
333:   333	in Fig. 8. It is observed that HFIA achieves performance improvement
334-   334	across all three datasets. For example, on the image-based dataset PRCC
335:   335	(Yang et al., 2019), the introduction of HFIA leads to a boost of 1.4 %
336:   336	in R1. On the video-based dataset CCVID (Gu et al., 2022), the integration of HFIA results in a 2.1 % increase in R1. Moreover, based
337:   337	on the ablation studies of CES and CCP, HFIA is a method with wellcoordinated components. Each individual component performs worse
338-   338	when used alone than when combined. We believe this is because each
339-   339	component was originally designed to work in conjunction with the
340:   340	other. HFIA without CCP can only focus on a single region (lacking
341:   341	pedestrian-oriented priors when the selection is random), while HFIA
342:   342	without CES completely loses its ability to focus.
343-   343	
344:   344	4.5.2. Unsupervised learning
345-   345	The experiments are conducted based on the FCM (Li et al., 2024a),
346-   346	with conﬁgurations consistent with the source code of FCM. Speciﬁcally,
347-   347	ViT (Dosovitskiy et al., 2021) is used as the backbone. Training is performed for 50 epochs, with a batch size of 256. The SGD optimizer is
--
351-   351	
352-   352	W. Pan et al.
353-   353	
354:   354	Fig. 8. Ablation studies of the HFIA on PRCC (Yang et al., 2019), Celeb-ReID (Huang et al., 2019), and CCVID (Gu et al., 2022).
355-   355	
356:   356	Fig. 9. Comparison of HFIA and baseline in ranking gallery retrieval results and visualization (Selvaraju et al., 2017) of each image from model output feature map.
357-   357	Here, gallery images highlighted in green and red denote right retrieval and wrong retrieval, respectively.
358-   358	
359:   359	We conduct a qualitative analysis of HFIA in PRCC (Yang et al.,
360:   360	2019), as shown in Fig. 9. The retrieval rank of gallery images demonstrates the great performance of HFIA in identity recognition. First, HFIA
361-   361	can correctly recognize pedestrians in similar clothing. For example, in
362:   362	Fig. 9(a1), both query and gallery images depict individuals wearing
363:   363	black tops and light-colored shorts. Our HFIA correctly retrieves all images, while the baseline incorrectly retrieves ﬁve images. Second, HFIA
364-   364	can mitigate the interference caused by signiﬁcant changes in clothing
365:   365	styles. For example, in Fig. 9(a2), the query image shows a person wearing a black top, while the gallery images show individuals wearing white
366:   366	tops. Our HFIA successfully retrieves images of the same person wearing
367-   367	a white top, whereas the baseline incorrectly retrieves another person
368-   368	wearing a black top.
369-   369	The visualization of feature maps showcases the characteristics of
370:   370	HFIA in feature learning. HFIA is more dedicated to learning identityrelated regions. For example, in the case of the query in Fig. 9(a2), the
371-   371	activation region of the baseline is present in the background region,
372:   372	while HFIA conﬁnes it to the pedestrian region. HFIA achieves targeted
373-   373	
374:   374	focus capabilities. In the majority of feature visualizations in Fig. 9, the
375:   375	activation regions in HFIA are more reﬁned compared to those in the
376-   376	baseline, consistently focusing on identity-related regions.
377:   377	In summary, by emulating human focus ability, HFIA improves the
378-   378	capability of the model for information comprehension. This helps to improve the model’s discrimination between identity-related and identityirrelated information.
379-   379	4.7. Analysis
380:   380	In this section, we conduct a quantitative analysis of HFIA. This analysis encompassed the eﬀect of the body components, the inﬂuence of
381-   381	data continuity, the aﬀect of pedestrian-oriented conﬁguration, and the
382-   382	impact of enlargement degree.
383-   383	4.7.1. Eﬀect of body components
384:   384	By means of HFIA, an image is split into ﬁve body components to
385-   385	attain realistic single-sample multi-region focus. We evaluate the eﬀec10
386-   386	
387-   387	Neural Networks 193 (2026) 107960
388-   388	
389-   389	W. Pan et al.
390-   390	
391:   391	Fig. 12. Ablation studies of pedestrian-oriented conﬁguration on PRCC (Yang
392:   392	et al., 2019). Here, HFIA w/o pedestrian-oriented denotes the HFIA without
393-   393	pedestrian-oriented conﬁguration (i.e., using Gaussian distribution functions to
394-   394	make reference point fall within the component centers more frequently).
395-   395	
396:   396	Fig. 10. Ablation studies of body components on PRCC (Yang et al., 2019).
397:   397	Here, HFIA w/o body components denotes HFIA without splitting image into
398-   398	ﬁve body components.
399-   399	
400:   400	to ﬁve CESs, HFIA exhibits higher performance, including a 0.1 %
401-   401	increase in mAP and a 1.7 % increase in R1. This may be due to
402:   402	the fact that ﬁve coordinated CESs allow the CCP to smoothly align
403-   403	body components, thus avoiding information deviations within connections.
404:   404	• Smoothing post-processing within the CCP. Fig. 5(c) shows the
405-   405	visual result of the smoothing post-processing strategy. As shown
406:   406	in Fig. 11, integration of the smoothing post-processing strategy within the HFIA results in a performance improvement, including a 3.5 % increase in mAP and a 5.2 % increase in R1.
407-   407	This improvement may be attributed to the smoothing postprocessing strategy, which smoothes discontinuous regions at the
408-   408	data level and avoids signiﬁcant information deviations within local
409-   409	regions.
410:   410	Fig. 11. Ablation studies of three strategy for data continuity on PRCC (Yang
411:   411	et al., 2019). Here, HFIA w/o enlargement grid denotes the HFIA without enlargement grid; HFIA w/o smoothing denotes the HFIA without smoothing postprocessing strategy; HFIA w/o aligning denotes the HFIA without align body
412:   412	components by applying correlative reference points to ﬁve CESs.
413-   413	
414:   414	In summary, data discontinuity leads to substantial information deviations within local regions. HFIA mitigates this issue through three
415-   415	stability-enhancing strategies at both the data and computation levels,
416-   416	achieving a stable enhancement of local details. This ensures that the focused images facilitate the subsequent learning of both identity-related
417-   417	and identity-irrelated details.
418-   418	
419:   419	tiveness of this strategy, as depicted in Fig. 10. The results clearly show
420-   420	a performance enhancement when body components are applied. Notably, there is a 0.8 % increase in mAP and a 1.0 % increase in R1. There
421-   421	are two reasons for the result. First, focusing on details across multiple
422-   422	regions can help the model activate multiple characteristics collectively.
423-   423	Second, body components can eﬀectively encapsulate identity information of pedestrians.
424-   424	
425-   425	4.7.3. Aﬀect of pedestrian-oriented
426:   426	The pedestrian-oriented conﬁguration makes HFIA more inclined to
427:   427	focus on pedestrian regions. Illustrated in Fig. 12, the adoption of this
428-   428	conﬁguration leads to a performance boost, including a 3.0 % increase
429-   429	in mAP and 5.0 % increase in R1. The result indicates that information
430-   430	closer to the center of body components is good at encapsulating pedestrian identity.
431-   431	
432-   432	4.7.2. Inﬂuence of data continuity
433:   433	Data continuity in local regions is a critical aspect of the HFIA.
434:   434	To achieve this, the HFIA utilizes three key strategies: (a) enlargement grid within the CES, (b) aligning body components within the
435:   435	CCP, and (c) smoothing post-processing within the CCP. The results of
436:   436	these strategies are presented in Fig. 11, and the following is a detailed
437:   437	analysis.
438-   438	
439-   439	4.7.4. Impact of enlargement degree
440:   440	The enlargement degree denotes strength of HFIA when focusing
441-   441	on local details. We conducte a comparison of four sets of diﬀerent
442:   442	enlargement degrees, and the results are shown in Fig. 13. It’s evident that the most stable enlargement degrees (i.e. 𝑆 = {1.5, 1.25, 1.0})
443-   443	yielded the most favorable results, surpassing the second-best by 4.8 %
444-   444	in terms of mAP and 5.6 % in terms of R1. Furthermore, with an increase in the enlargement degrees, the model’s performance exhibits
445-   445	a gradual decline. Notably, the strongest enlargement degrees (i.e.
446-   446	𝑆 = {4.0, 2.0, 1.0}) yields the lowest performance, reaching only 51.6 %
447-   447	mAP and 50.2 % R1. We speculate that the increase in enlargement
448-   448	degrees leads to information silos. Speciﬁcally, in the initial design,
449:   449	the HFIA recognizes pedestrian identity through ﬁve body components. However, as shown in Fig. 13, with the increase in enlargement degrees, the focused local details gradually dominate the image.
450:   450	Ultimately, the HFIA relies solely on ﬁve local details to recognize
451-   451	pedestrian identity, overlooking potential global features such as body
452-   452	shape.
453-   453	
454:   454	Enlargement grid within the CES. Fig. 3(a) and (b) present the
455:   455	visual results of HFIA without and with enlargement grid, respectively. As shown in Fig. 11, employing enlargement grid enhances
456:   456	the performance of the HFIA, resulting in a 2.3 % increase in mAP
457-   457	and a 3.0 % increase in R1. This improvement may be attributed
458-   458	to the information loss and discontinuity in local details caused
459:   459	by HFIA without enlargement grid. When employed with enlargement grid, the HFIA focuses on local details in a smooth and stable
460-   460	manner.
461:   461	• Aligning body components within the CCP. Fig. 5(b) presents the
462:   462	visual result of aligning body components. As shown in Fig. 11, when
463-   463	aligned body components by applying correlative reference points
464-   464	•
465-   465	
--
469-   469	
470-   470	W. Pan et al.
471-   471	
472:   472	Fig. 13. Results of four sets of enlargement degrees on PRCC (Yang et al., 2019). Here, 𝑆 = {𝑠0 , 𝑠1 , 𝑠2 } denotes the set of enlargement degrees.
473-   473	
474-   474	5. Conclusion
475-   475	
--
484-   484	
485-   485	To mitigate the challenges posed by clothes changing in clothes
486-   486	changing person re-identiﬁcation (CC-ReID), we propose a humanoid
487:   487	focus-inspired image augmentation (HFIA) to enhance the recognition
488-   488	of local details. Our inspiration stems from human ability to focus on
489:   489	multiple regions of pedestrians collectively. Speciﬁcally, HFIA divides
490-   490	the image into ﬁve body components: head-shoulder, left upper torso,
491:   491	right upper torso, left lower torso, and right lower torso. Then, employing a centeral emphasis strategy (CES), it adjusts the size of each component’s regions, enlarging those near the center and shrinking those
492-   492	farther away. Subsequently, through component continuity processing
493:   493	(CCP), it ensures the coordination of the CES for all components, with
494:   494	shared normalized vertical coordinates among them, and mirrored horizontal coordinates for the left and right torsos. Moreover, the CCP applies smoothing post-processing to erase discontinuities between the
495:   495	head-shoulder component and the left and right upper torso components. Experimental results demonstrate that our HFIA surpasses the
496-   496	state-of-the-art methods across three CC-ReID datasets.
497-   497	
498-   498	References
--
535-   535	for the content of the published article.
536-   536	
537-   537	CRediT authorship contribution statement
538:   538	Wenjie Pan: Writing – original draft, Visualization, Conceptualization; Jianqing Zhu: Writing – review & editing, Supervision, Funding
539-   539	acquisition; Xiaolin Cui: Writing – review & editing; Huanqiang Zeng:
540-   540	Writing – review & editing, Supervision; Yibing Zhan: Writing – review
541-   541	& editing.
--
679-   679	person identiﬁcation. In Proceedings of the IEEE conference on computer vision and pattern
680-   680	recognition workshops (pp. 2112–2120). Utah, USA.
681-   681	Yang, Q., Wu, A., & Zheng, W. S. (2019). Person re-identiﬁcation by contour sketch under
682:   682	moderate clothing change. IEEE transactions on pattern analysis and machine intelligence,
683-   683	43(6), 2029–2046.
684-   684	Yang, S., Kang, B., & Lee, Y. (2022). Sampling agnostic feature representation for
685-   685	long-term person re-identiﬁcation. IEEE Transactions on Image Processing, 31, 6412

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
/bin/zsh -lc "pdfinfo 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' | sed -n '1,35p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           GAE-Net: A gait-assisted enhancement network for video-based person re-identification
Subject:         Neural Networks, 192 (2025) 107946. doi:10.1016/j.neunet.2025.107946
Keywords:        Video-based person re-identification,Gait recognition,Knowledge distillation,Dynamic feature aggregation
Author:          Minting Dai
Creator:         Elsevier
Producer:        Acrobat Distiller 8.1.0 (Windows)
CreationDate:    Sat Nov  8 08:16:26 2025 CST
ModDate:         Sat Nov  8 08:28:10 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           12
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       7046265 bytes
Optimized:       no
PDF version:     1.7

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | sed -n '43,145p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    43	1. Introduction
    44	Over the past decade, researchers have proposed numerous methods to address the highly challenging task of video-based person reidentiﬁcation (ReID). With the swift advancement in appearance feature
    45	extraction techniques using Convolutional Neural Networks (CNNs),
    46	video Re-ID aims to incorporate temporal information to achieve a more
    47	robust spatio-temporal representation.
    48	To aggregate temporal cues, existing works have made great eﬀorts
    49	in temporal modeling. Taking inspiration from the ﬁeld of video recognition, some studies were dedicated to building temporal models by
    50	utilizing two-stream network (Liu et al., 2017), recurrent neural network (RNN) (Chung et al., 2017), or 3D CNN (Liu et al., 2019b), etc.
    51	These methods combine appearance static information (color, wearing,
    52	etc.) with dynamic temporal representation (motion information, optical ﬂow cues, etc.) as the entire video representation. Despite adding
    53	dynamic temporal representation as the supplementary information, the
    54	Re-ID method still relies heavily on appearance features as the primary
    55	discriminant representation. Therefore, it is easily disturbed by external
    56	
    57	environmental factors, such as light changes, color changes, wearing
    58	changes, etc.
    59	Gait is a biological feature that embodies a persons’ unique walking posture and can be captured from a distance. As a biometric technology, gait recognition has made remarkable achievements in recent
    60	years by extracting gait features from the input gait data to identify
    61	target pedestrians. Gait recognition utilizes robust gait features, including skeletal joints and walking postures, as biometric information to
    62	identify individuals. As a result, gait recognition is less aﬀected by
    63	appearance variations. Earlier methods (Wu et al., 2017) represented
    64	gait through static images, where all gait silhouettes were consolidated
    65	into a single image or gait template for recognition purposes. However, this compression process often led to the loss of temporal dynamics and detailed spatial features. In recent years, many works (Fan
    66	et al., 2020; Lin et al., 2021) have overcome the above shortcomings
    67	by taking gait video sequences as input, thereby retaining more spatiotemporal features directly from the original gait sequence. However,
    68	the lack of appearance information in gait data also limits its further
    69	development.
    70	
    71	∗ Corresponding author.
    72	
    73	E-mail addresses: dmt@stu.xidian.edu.cn (M. Dai), yangx@xidian.edu.cn (X. Yang), dwj@stu.xidian.edu.cn (W. Dong), nnwang@xidian.edu.cn (N. Wang).
    74	https://doi.org/10.1016/j.neunet.2025.107946
    75	Received 30 May 2025; Received in revised form 14 July 2025; Accepted 1 August 2025
    76	Available online 5 August 2025
    77	0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    78	
    79	Neural Networks 192 (2025) 107946
    80	
    81	M. Dai et al.
    82	
    83	method that enhances the perception and processing of local features
    84	for transferring robust deep latent knowledge from multimodal fusion
    85	models to unimodal appearance models.
    86	We trained a binary segmentation model to generate human gait contour images based on the UNet (Ronneberger et al., 2015) architecture,
    87	and take the obtained gait sequences as gait model input. In addition,
    88	we propose LPCD to enhance the perception and processing of local features by employing multi-scale aggregation techniques, thereby extracting and preserving more reﬁned semantic knowledge. This approach effectively guides the student model(Re-ID network) in learning discriminative detailed features from the teacher model (DTA-Net) for similar
    89	pedestrians, ultimately leading to a signiﬁcant enhancement in its performance. Through the combination of DTA-Net and LPCD, the GAE-Net
    90	can achieve more robust video representation. Furthermore, the parameter reduction achieved via knowledge distillation not only improves the
    91	eﬃciency of model inference but also signiﬁcantly enhances the overall
    92	computational speed. The Single-modal model does not need to rely on
    93	other types of data during the reasoning process and can complete tasks
    94	by using only single-modal data.
    95	In summary, our contributions are four folds:
    96	
    97	Fig. 1. Comparison between RGB clips for video Re-ID and gait clips for gait
    98	recognition. Although RGB sequences and gait sequences exhibit modality differences in terms of color, environmental background, and other factors, they
    99	complement each other in terms of spatial information and temporal relationships.
   100	
   101	•
   102	
   103	We ﬁnd the shortcomings of video Re-ID and explore the feasibility
   104	of the gait supplement feature: the Re-ID methods are easily disturbed due to their dependency on appearance features, while the
   105	gait recognition method that is not aﬀected by appearance can provide supplementary spatiotemporal information.
   106	• We propose a Dynamic Gait Assistance Network (DGA-Net), utilizing
   107	Dynamic Feature Aggregation (DFA) to dynamically aggregate the
   108	collected gait features and appearance features, thereby obtaining
   109	more robust representations.
   110	• We propose Local Perceptual Complementary Distillation (LPCD) to
   111	overcome the limitations of classic logit distillation. The global logit
   112	is decoupled into consistent and complementary local logit outputs,
   113	aiming to mine and convey more abundant and explicit semantic
   114	knowledge.
   115	• The proposed GAE-Net demonstrates superior performance on two
   116	widely used video-based person Re-ID benchmarks in contrast to the
   117	many previous state-of-the-art.
   118	
   119	Based on the above analysis, gait recognition can supplement the defects of person Re-ID while ensuring task consistency. Recent research
   120	has explored the use of gait features to improve the reliability and
   121	robustness of Re-ID systems (Liu et al., 2015) extracted feature representations by integrating gait and appearance features at both the
   122	score and feature levels, fusing these features comprehensively. In contrast to the background removal methods, Tang et al. (2019) employed
   123	a background suppression approach, assigning diﬀerential weights to
   124	background and human body elements during image feature extraction (Zhao et al., 2023) constructed an appearance-gait dual-stream
   125	network (AGNet), which simultaneously extracts appearance and gait
   126	features from both RGB video clips and gait video sequences.However,
   127	existing methods have introduced gait features for solution proposals,
   128	but they have not completely bridged the gap between gait and visible data. Therefore, exploring better solutions that combine video-based
   129	Re-ID and gait recognition holds signiﬁcant research value. As shown in
   130	Fig. 1, there are modality diﬀerences between RGB sequences and gait
   131	sequences, while they also exhibit modality complementarity. Specifically, gait features enhance the robustness of the Re-ID system with
   132	its non-changeable gait features. At the same time, appearance features
   133	can complement the appearance absence of gait features. In addition,
   134	temporal information, as a common part, is better explored under the
   135	joint fusion of the two representations. Therefore, we sought to explore
   136	a novel framework that complementarily combines video Re-ID and gait
   137	recognition.
   138	Inspired by dynamic gait features, we propose a Gait-Assisted Enhancement Network (GAE-Net) for video Re-ID. Firstly, we design a Dynamic Two-stream Aggregation Network (DTA-Net) to simultaneously
   139	capture both appearance and gait characteristics from RGB video frames
   140	and gait data. In the feature extraction stage, DTA-Net employs the ReID network and gait network as separate branches to respectively extract appearance features and gait features. In the feature fusion stage,
   141	Dynamic Attention Weighting (DAW) and Dynamic Weight Aggregation
   142	(DWA) are designed to fuse gait and appearance features. With the combined eﬀect of the above modules, the DTA-Net can learn better appearance and gait complement features. Furthermore, we contend that existing knowledge distillation methods based on logit are not optimal because they only use global logit output, which includes various semantic
   143	knowledge. This could impart ambiguous knowledge to the student, potentially leading its learning in the wrong direction. To overcome this issue, we propose a Local Perception Complementary Distillation (LPCD)
   144	
   145	2. Related work

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | sed -n '243,580p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   243	3. Methods
   244	The Re-ID system pays too much attention to appearance features
   245	and is disturbed by external factors, such as light changes, color changes,
   246	etc. At the same time, gait recognition only learns the human contours
   247	and temporal gait features without any appearance information, thus
   248	limiting its further development. Therefore, Thus, we propose integrating the strengths of both approaches to combine visual appearance and
   249	gait characteristics, aiming to develop a more resilient spatio-temporal
   250	representation that can eﬀectively address the inevitable challenges. In
   251	this paper, we propose a Gait-Assisted Enhancement Network (GAE-Net)
   252	
   253	2.2. Gait recognition
   254	Recently, many existing gait recognition algorithms have explored
   255	convolutional neural networks (CNNs) to generate gait representation.
   256	Shiraga et al. (2016) attempted to extract global gait features from gait
   257	3
   258	
   259	Neural Networks 192 (2025) 107946
   260	
   261	M. Dai et al.
   262	
   263	Fig. 2. The overall architecture of our GAE-Net. It integrates appearance and gait cues via a dual-branch structure (DTA-Net), where the Dynamic Feature Aggregation
   264	(DFA) module fuses both features. A lightweight Re-ID branch is guided by the DTA-Net through knowledge distillation (LPCD), enabling robust spatio-temporal
   265	representation.
   266	
   267	3.2. Dynamic two-stream aggregation network (DTA-Net)
   268	
   269	to dynamically learn appearance feature and gait feature. First, we design the Dynamic Two-Stream Aggregation Network (DTA-Net) to fuse
   270	the two modal features. In addition, we propose LPCD to overcome the
   271	limitations of classic logit distillation which decouples the global logit
   272	into consistent and complementary local logit outputs, aiming to mine
   273	and convey more abundant and explicit semantic knowledge.
   274	
   275	In this part, we provide a detailed overview of the proposed DTANet. As discussed previously, DTA-Net is composed of two main components: the appearance stream and the gait stream. We design a Dynamic
   276	Feature Aggregation (DFA) module to explore the shared and complementary features between the two modalities and thus provide a more
   277	discriminative representation.
   278	
   279	3.1. Overview
   280	
   281	3.2.1. Re-ID network
   282	For the appearance branch of DTA-Net, we use the common video
   283	Re-ID baseline as the Re-ID network (as shown in Fig. 2(a)). Speciﬁcally,
   284	for the video clip input 𝑉𝐼 = 𝐼1 , 𝐼2 , … , 𝐼𝑇 , we feed the 𝑇 frames into the
   285	appearance feature extractor CNN backbone (ResNet50). As a result, we
   286	obtain a frame-level feature collection 𝑓 𝐼 ∈ ℝ𝐶×𝑇 ×𝐻×𝑊 . Then, according to the practice of most existing methods, temporal average pooling
   287	(TAP) and global average pooling (GAP) are applied to the feature map.
   288	Finally, a video-level appearance representation is obtained through the
   289	Re-ID network, denoted as:
   290	
   291	Fig. 2 illustrates the architecture of our GAE-Net. Unlike singlemodal approaches, our objective is to explore an integrated spatiotemporal representation by fusing the appearance features and the
   292	gait features. Therefore, we propose a Gait-Assisted Enhancement Network (GAE-Net), which consists of the DTA-Net and the DGA-Net.
   293	In practice, DTA-Net employs the Re-ID branch and the gait branch
   294	to extract the appearance features and the gait features, respectively.
   295	During feature Aggregation, the Dynamic Feature Aggregation (DFA)
   296	module is designed to fuse the appearance features and the gait features. With DTA-Net as the teacher model and Re-ID network as the
   297	student model, we construct the LPCD. By implicit feature fusion
   298	through knowledge distillation, GAE-Net can achieve more robust video
   299	representation.
   300	Formally, given a temporal input sequence 𝑆(𝐼, 𝐺), where 𝐼 and 𝐺
   301	represent the appearance image sequence and the gait image sequence,
   302	respectively. By convention, we ﬁrst adopt RRS (Li et al., 2018a) to
   303	sample 𝑇 frames to establish the appearance video clip 𝑉𝐼 and the gait
   304	sequence clip 𝑉𝐺 , where 𝑉𝐼 = 𝐼1 , 𝐼2 , … , 𝐼𝑇 represents the input to the
   305	Re-ID network and 𝑉𝐺 = 𝐺1 , 𝐺2 , … , 𝐺𝑇 indicates the input to the gait
   306	network. Then, the appearance video clip 𝑉𝐼 and the gait sequence clip
   307	𝑉𝐺 are fed into the two-branch backbone network to separately extract
   308	the appearance features and the gait features. In addition, the Dynamic
   309	Feature Aggregation (DFA) module is designed to transform features
   310	from two modalities into a uniﬁed feature space, where they can be dynamically combined to produce an integrated representation of a person.
   311	Finally, we adopt Local Perceptual Complementary Distillation (LPCD)
   312	to transfer the fusion feature to a single modality network. With the
   313	help of GAE-Net, appearance feature and gait feature are dynamically
   314	integrated and incredibly transferred into the appearance network.
   315	
   316	𝑓𝐼 = 𝐼(𝑉𝐼 ) = 𝑇 𝐴𝑃 (𝐺𝐴𝑃 (𝐹 (𝑉𝐼 )))
   317	=
   318	
   319	𝑇
   320	𝐻 𝐻
   321	1 ∑ 1 ∑∑ 𝐼
   322	𝑓 ,
   323	𝑇 𝑡=1 𝐻𝑊 ℎ=1 𝑤=1
   324	
   325	(1)
   326	
   327	where 𝑓𝐼 , 𝐹 (𝑉𝐼 ), and 𝐼(𝑉𝐼 ) represent the appearance representation, the
   328	feature extractor and the Re-ID network, respectively. In order to aggregate spatial information, feature map 𝑓 𝐼 ∈ ℝ𝐶×𝑇 ×𝐻×𝑊 are aggregated
   329	into feature vectors by GAP. To integrate temporal information, TAP
   330	is utilized to combine frame-level features into an overall video-level
   331	representation.
   332	3.2.2. Gait network
   333	We utilize the gait branch to capture gait information (as illustrated
   334	in Fig. 2 (b)) to augment appearance features and explore temporal information. Similarly, the gait branch follows an identical procedure for
   335	extracting gait features: it performs feature aggregation following the
   336	extraction process. In practice, we adopt the GaitGL (Lin et al., 2021)
   337	network as the gait network. During the feature extraction phase, GaitGL
   338	4
   339	
   340	Neural Networks 192 (2025) 107946
   341	
   342	M. Dai et al.
   343	
   344	to obtain an aggregation representation. The DWA can be deﬁned as:
   345	𝑓𝑚 = 𝑀𝐿𝑃 (𝑓𝑝 ),
   346	𝑓ℎ = 𝑀𝐿𝑃 (𝑓𝑝 ),
   347	
   348	(4)
   349	
   350	𝑓𝑎 = 𝑓𝑚 @𝑟𝑒𝑠ℎ𝑎𝑝𝑒(𝑓ℎ ),
   351	where 𝑓𝑚 , 𝑓ℎ , @, 𝑟𝑒𝑠ℎ𝑎𝑝𝑒, 𝑓𝑎 denote the intermediate representation and
   352	the high-dimensional representation, the Matrix multiplication, reshape
   353	operation and the aggregation representaion, respectively.
   354	To take full advantage of the performance beneﬁts of the DAW and
   355	the DWA, we add 𝑓𝑤 and 𝑓𝑎 point by point and an MLP is used to get
   356	the ﬁnal fusion representation 𝑓𝑓 𝑢𝑠𝑒 , i.e,
   357	𝑓𝑓 𝑢𝑠𝑒 = 𝑀𝐿𝑃 (𝑓𝑤 + 𝑓𝑎 ).
   358	
   359	Fig. 3. The design details of Dynamic Feature Aggregation (DFA). DFA explores
   360	two eﬀective feature fusion schemes: Dynamic Attention Weighting (DAW) and
   361	Dynamic Weight Aggregation (DWA). By applying these two modules, DTA-Net
   362	ﬁnally obtained a robust video representation.
   363	
   364	In conclusion, to eﬀectively integrate the advantages of appearance
   365	features and gait features, we propose a Dynamic Two-stream Aggregation Network (DTA-Net). Among them, the DAW reweights channelwise features to provide guidance on the importance of global characteristics, while the DWA introduces dynamic ﬁne-grained aggregation to
   366	enhance feature representation and temporal adaptability. As complementary pathways, these two modules work collaboratively to achieve
   367	more eﬀective and adaptive feature fusion, thereby addressing existing
   368	challenges in both video-based person re-identiﬁcation and gait recognition.
   369	
   370	inputs the corresponding gait clip 𝑉𝐺 = 𝐺1 , 𝐺2 , … , 𝐺𝑇 into the backbone
   371	extractor to obtain frame-level features. Moreover, GaitGL then applies
   372	Generalized Mean Pooling (GeM Radenović et al. (2018)) and TMP to
   373	aggregate features into video-scale gait representation. The gait network
   374	can be expressed as:
   375	𝑓𝐺 = 𝐺(𝑉𝐺 ) = 𝐺𝑒𝑀(𝑇 𝑀𝑃 (𝐹 (𝑉𝐺 )))
   376	1 ∑
   377	𝑇 𝑡=1
   378	𝑇
   379	
   380	=
   381	
   382	[(
   383	( )𝑝 )] 1𝑝
   384	𝐹𝑎𝑣𝑔 𝑓 𝐺
   385	,
   386	
   387	(2)
   388	
   389	3.3. Local perception complementary distillation (LPCD)
   390	In recent years, knowledge distillation (KD) has been extensively applied in various scenarios, including model compression, accelerated inference, and performance enhancement of smaller models. Fundamentally, knowledge distillation enables the transfer of implicit knowledge
   391	from the teacher model to the student model. We consider the current logit-based approaches to be suboptimal due to their exclusive reliance on global logits output, which combines diverse semantic knowledge. This may overlook intricate characteristics of similar pedestrian
   392	samples, such as hairstyles, backpack straps, and glasses, potentially
   393	transmitting ambiguous information and misleading students in their
   394	learning process. Therefore, we propose Local Perception Complementary Distillation (LPCD), which transfers the robust deep dark knowledge from the multimodal fusion model to the single-modal appearance
   395	model by enhancing the perception and processing of local features.
   396	Speciﬁcally, LPCD decomposes the global logits output into multiple local logits outputs, allowing the student model to learn ﬁnegrained logical knowledge from the teacher model. Furthermore, the
   397	extracted knowledge can be categorized into two types: consistent logical knowledge and complementary logical knowledge. Consistent logical knowledge primarily conveys the semantic information of samples, aiding the student model in understanding their core features.
   398	In contrast, complementary logical knowledge captures the ambiguity or uncertainty within samples, thereby enhancing the model’s
   399	discriminative capability for complex cases. By placing greater emphasis on the complementarity component, LPCD directs the student
   400	model to concentrate more on similar pedestrian samples, thus boosting its ability to distinguish between diﬀerent instances. Additionally,
   401	model parameter compression through knowledge distillation signiﬁcantly improves the model’s inference eﬃciency. Notably, the singlemodal model does not require additional modal data participation while
   402	inferencing.
   403	Overall Framework: After passing through the DTA-Net, we combine appearance features and gait features to obtain the fusion-modal
   404	representation. To transfer the multi-modal representation to the singlemodal representation, we design the LPCD (as shown in Fig. 4) based on
   405	knowledge distillation. Before distilling, it is essential to ensure that the
   406	teacher model has been trained. The parameters of the teacher model
   407	were frozen to guarantee that the knowledge and performance of the
   408	teacher model remained stable throughout the training process, and the
   409	
   410	where 𝑓𝐺 , 𝐹 (𝑉𝐺 ), 𝐺(𝑉𝐺 ) represent the gait representation, the gait feature extractor, the gait network, respectively. In particular, 𝑝 > 0 is a
   411	learnable parameter. The value of 𝑝 here is selected as 6.5.
   412	3.2.3. Dynamic feature aggregation (DFA)
   413	After processing through the Re-ID branch and the gait branch, we
   414	obtain the appearance feature 𝑓𝐼 and the gait feature 𝑓𝐺 . As mentioned
   415	earlier, these two features are complementary. To achieve a more robust spatio-temporal representation, we should integrate the strengths
   416	of both features. To better aggregate the appearance representation
   417	and the gait representation, we propose a Dynamic Feature Aggregation (DFA) module (as shown in Fig. 3). Speciﬁcally, the proposed DFA
   418	module comprises two parallel components, Dynamic Attention Weighting (DAW) and Dynamic Weight Aggregation (DWA). Before entering
   419	the features into the DFA module, we concatenate the 𝑓𝐼 and 𝑓𝐺 as
   420	𝑓𝑝 = [𝑓𝐼 , 𝑓𝐺 ] as the joint person feature.
   421	Dynamic Attention Weighting (DAW): In DAW, the attention
   422	mechanism is employed to assign weights to the joint features, thereby
   423	enhancing the discriminative power of the feature representation. In
   424	practice, the joint features 𝑓𝑝 are ﬁrst projected to the intermediate feature by a multi-layer perceptron (MLP). Next, the softmax function is
   425	applied to determine the weights for the intermediate representation,
   426	resulting in the derivation of the weighting coeﬃcients. Finally, the
   427	weight coeﬃcient and the intermediate features are combined to obtain the weighted representation. The DAW can be deﬁned as:
   428	𝑓𝑚 = 𝑀𝐿𝑃 (𝑓𝑝 ),
   429	𝑓𝑤 = 𝑓𝑚 ∗ 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(𝑓𝑚 ),
   430	
   431	(5)
   432	
   433	(3)
   434	
   435	where 𝑓𝑚 , 𝑓𝑤 , ∗ denote the intermediate representation, the weighted
   436	representation and the dot product, respectively.
   437	Dynamic Weight Aggregation (DWA): As for DWA, dynamic MLP
   438	is applied to dynamically aggregate the feature weight to mine a more
   439	robust representation. Speciﬁcally, the joint features 𝑓𝑝 are projected
   440	to intermediate features 𝑓𝑚 ∈ ℝ𝐶𝑚 by a low-dimensional MLP. At the
   441	same time, it projects to high-dimensional features 𝑓ℎ ∈ ℝ𝐶ℎ by another
   442	high-dimensional MLP. Here, we consider high-dimensional features to
   443	be dynamic fusion feature. Moreover, the high-dimensional features are
   444	reshaped to 𝐶ℎ = (𝐶𝑚 , 𝑛) and combined with the intermediate features
   445	5
   446	
   447	Neural Networks 192 (2025) 107946
   448	
   449	M. Dai et al.
   450	
   451	DKD refers to representing the traditional KD loss as a weighted combination of two parts: one related to the target category (TCKD) and the
   452	other related to the non-target category (NCKD). TCKD focuses on the
   453	knowledge transfer related to the target category, while NCKD focuses
   454	on the relationship modeling between non-target categories.
   455	𝐷𝐾𝐷 = 𝛼𝑇 𝐶𝐾𝐷 + 𝛽𝑁𝐶𝐾𝐷,
   456	
   457	(9)
   458	
   459	where 𝛼 and 𝛽 represent the TCKD coeﬃcient and the NCKD coeﬃcient
   460	respectively.
   461	Then iterate through all scales 𝐦 ∈ 𝐌 and the corresponding regions
   462	𝐧 ∈ 𝐍𝐦 to obtain the ﬁnal LPCD loss,
   463	∑ ∑
   464	𝐿𝐿𝑃 𝐶𝐷 =
   465	𝐷(𝑚, 𝑛).
   466	(10)
   467	𝑚∈𝑀 𝑛∈𝑁𝑚
   468	
   469	Furthermore, LPCD further partitions the decoupled logit knowledge
   470	into two components: consistent logit knowledge and complementary
   471	logit knowledge. Consistent logit knowledge involves local logit outputs
   472	that align with the category of the global logit output, thereby providing multi-scale insights speciﬁc to that particular category. In contrast,
   473	complementary logit knowledge encompasses local logit outputs associated with categories diﬀerent from that of the global logit output. This
   474	approach preserves sample uncertainty and stops the student network
   475	from overﬁtting on ambiguous samples. By capturing consistent logit
   476	knowledge, the model can better understand the nuances within a single category across various scales. Meanwhile, leveraging complementary logit knowledge ensures that the diversity of potential categories
   477	is preserved, promoting a more robust and generalized learning process
   478	for the student network. We introduce hyper-parameter 𝛾 to amplify the
   479	weighting of the complementary logit knowledge component, the LPCD
   480	loss can be rewritten as follows,
   481	
   482	Fig. 4. Illustration of our LPCD. LPCD introduces a method to extract multiscale logit knowledge using multi-scale pooling techniques, enabling the student model to acquire detailed and clear semantic information from the teacher
   483	model.
   484	
   485	fused representation of the teacher model as a soft label to constrain the
   486	training process of the student model. In practice, the trained DTA-Net
   487	serves as the teacher model, whereas the Re-ID network functions as the
   488	student model.
   489	LPCD Details: The logit output maps of the teacher model and the
   490	student model are represented as 𝐿𝑇 and 𝐿𝑆 respectively. Multi-scale
   491	pooling applies average pooling across multiple scales to generate logit
   492	outputs corresponding to various regions of the input image. This approach facilitates the retention of ﬁne-grained knowledge with clear
   493	semantics in the student model, as compared to traditional knowledge
   494	distillation methods that solely focus on global logit knowledge. Then, a
   495	knowledge distillation pipeline is established for the logit output of each
   496	scale. This means that during the training process, the student model
   497	not only needs to learn the overall output of the teacher model but also
   498	needs to capture the local features of the teacher model at diﬀerent levels. Finally, by introducing information weighting to adjust its weight
   499	to increase the emphasis on the local logit distillation loss.
   500	Speciﬁcally, multi-scale pooling systematically divides and aggregates information from diﬀerent scales and employs average pooling
   501	to consolidate the logical information within
   502	( each
   503	) (cell.)For a given scale
   504	m, the logit output map is divided intoƵ
   505	
   506	ℎ𝑇
   507	𝑚
   508	
   509	×
   510	
   511	𝑤𝑇
   512	𝑚
   513	
   514	𝐿𝐿𝑃 𝐶𝐷 = 𝐷con + 𝛾𝐷com ,
   515	
   516	where 𝐷con and 𝐷com represents the combined loss of consistent and
   517	complementary logit knowledge respectively.
   518	3.4. Model learning
   519	Training Phase: In the implementation process, we compute both the
   520	cross-entropy loss and the triplet loss for the fused representation 𝑓𝑓 𝑢𝑠𝑒 .
   521	The total loss is obtained by summing up all individual losses. The model
   522	is supervised by:
   523	𝐿𝑡 = 𝐿𝑐𝑒 + 𝐿𝑡𝑟𝑖 ,
   524	
   525	∑
   526	
   527	𝐿𝑆 (𝑗, 𝑘)
   528	
   529	𝑗,𝑘∈(𝑚,𝑛)
   530	
   531	𝑚2
   532	
   533	,
   534	
   535	(12)
   536	
   537	where 𝐿𝑡 , 𝐿𝑐𝑒 and 𝐿𝑡𝑟𝑖 indicate the total loss during training, the cross
   538	entropy loss and the triplet loss.
   539	Distillation Phase: Given that GAE-Net comprises two distinct stages,
   540	we employ separate loss functions for model training and model distillation. In the model distillation phase, we compute the cross-entropy
   541	loss, triplet loss, and LPCD loss individually. The overall loss function is
   542	achieved by:
   543	
   544	regions, where
   545	
   546	each region has dimensions of 𝑚 × 𝑚. Average pooling is then applied
   547	to the Logits within each region in order to obtain local logit outputs at
   548	that particular scale,
   549	∑ 𝐿𝑇 (𝑗, 𝑘)
   550	𝜋𝑇 (𝑚, 𝑛) =
   551	,
   552	(6)
   553	𝑚2
   554	𝑗,𝑘∈(𝑚,𝑛)
   555	𝜋𝑆 (𝑚, 𝑛) =
   556	
   557	(11)
   558	
   559	𝐿𝑑 = 𝐿𝑐𝑒 + 𝐿𝑡𝑟𝑖 + 𝐿𝐿𝑃 𝐶𝐷 ,
   560	
   561	(13)
   562	
   563	where 𝐿𝑑 , 𝐿𝑐𝑒 , 𝐿𝑡𝑟𝑖 , 𝐿𝐿𝑃 𝐶𝐷 indicate the total loss of distillation, the cross
   564	entropy loss, the triplet loss and the LPCD loss.
   565	Testing Phase: It is crucial to assess the integration of multi-modal features in DTA-Net to validate its eﬀectiveness and feasibility. In addition,
   566	we need to test the single-modal feature of the GAE-Net to demonstrate
   567	the eﬀectiveness of LPCD.
   568	
   569	(7)
   570	
   571	where (𝑚, 𝑛) denotes the spatial bins of the cell at a speciﬁc scale, (𝑗, 𝑘)
   572	represents the coordinates of the logit output in (𝑚, 𝑛).
   573	For each scale, the distillation loss (𝑚, 𝑛) is computed based on the
   574	local logit outputs,
   575	( (
   576	) (
   577	))
   578	(8)
   579	(𝑚, 𝑛) =  𝜎 𝜋𝑇 (𝑚, 𝑛) , 𝜎 𝜋𝑆 (𝑚, 𝑛) ,
   580	

codex
第三篇的结构是典型“两阶段卖法”：先把 gait 作为训练期或教师期的互补信息源，再用蒸馏把多模态能力压回单模态推理。关键证据要看它是否分别证明了融合教师和局部蒸馏，而不是只证明多了一个 gait 分支。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 7 -l 12 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | rg -n \"4\\.4|Ablation|ablation|DTA|DFA|DAW|DWA|LPCD|DGA|GAE|Comparison|Table|Fig|visual|parameter|Runtime|Effect|Baseline|teacher|student|gamma|scale\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2-     2	
3-     3	M. Dai et al.
4-     4	
5:     5	Table 1
6-     6	The statistics of two datasets for video Re-ID.
7-     7	Dataset
8-     8	
--
28-    28	15
29-    29	CMC & mAP
30-    30	
31:    31	Results on MARS: As illustrated in Table 2, the Rank-1 accuracy of
32:    32	GAE-Net proposed is 91.1 % on MARS, and the mAP result is 87.7 %,
33-    33	which outperform previous approaches and represent a substantial improvement over several state-of-the-art methods. It is crucial to emphasize that the two-stream structure (Simonyan & Zisserman, 2014) remains a valuable and eﬃcient approach for multimodal parallel fusion.
34-    34	At present, several eﬀorts have produced commendable results: STRF
35:    35	(Aich et al., 2021) obtained 86.1 % mAP and 90.3 % Rank-1 using spatiotemporal representation decomposition; CTL (Liu et al., 2021a) integrates essential features with graph convolution to build a multi-scale
36-    36	mAP. This approach achieves an mAP of 86.7 % and a Rank-1 accuracy
37-    37	of 91.4 % on MARS. However, they do not incorporate gait information
38-    38	so performance may be impaired by appearance noise. AGNet (Zhao
39-    39	et al., 2023) integrated appearance features and gait features via the attention mechanism, yet it failed to address the substantial variations in
40-    40	gait. In contrast, our method signiﬁcantly reduces the adverse eﬀects of
41:    41	gait changes across diverse scenarios through DFA. In addition, SSN3D
42-    42	(Jiang et al., 2021) addressed the issue of temporal misalignment via 3D
43-    43	convolution, while SGMN (Chen et al., 2022) enhanced robustness by
44-    44	extracting ﬁne-grained information from each frame. However, both approaches may place excessive reliance on complex temporal modeling.
45-    45	In contrast, our method improves local feature processing and extracts
46:    46	information across multiple scales, thereby demonstrating superior performance in handling challenges associated with similar target pedestrians.
47:    47	Results on LSVID: As illustrated in Table 2, GAE-Net outperforms
48-    48	the majority of existing methods, achieving state-of-the-art precision.
49-    49	Overall, our model attains 75.1 % mAP and 84.6 % Rank-1 accuracy on
50-    50	LSVID, surpassing many state-of-the-arts approaches. By analysis, we believe that the quality of the gait mask extracted from the LSVID dataset
--
52-    52	addition, BiCnet-TKS (Hou et al., 2021) utilized both CNN-based and
53-    53	attention-based architectures, ultimately achieving 74.6 % Rank-1 and
54-    54	75.1 % mAP.
55:    55	In conclusion, GAE-Net demonstrates commendable performance on
56-    56	the two datasets mentioned above. In contrast to the current methods,
57-    57	the proposed approach integrates gait information to investigate more
58-    58	robust temporal and spatial representations. Moreover, we apply knowledge distillation to establish a dark knowledge transfer channel, so as
--
66-    66	video clips and featuring 1261 unique identities, along with an additional 3248 distractor terms. LSVID collected 14,943 sequence samples
67-    67	from 3772 individuals, with each sequence containing an average of
68-    68	200 frames. The structures of the two benchmarks for video Re-ID are
69:    69	detailed in Table 1.
70-    70	Evaluation Protocol: We adhere to prior standards by utilizing the
71:    71	Cumulative Matching Characteristic (𝐶𝑀𝐶) curve and the mean Average Precision (𝑚𝐴𝑃 ) to assess the performance of our proposed GAE-Net
72-    72	on the aforementioned two datasets.
73-    73	4.2. Implementation details
74-    74	The entire framework is implemented using PyTorch and is based on
--
94-    94	adjusted to a resolution of 256×128. However, each gait image of the
95-    95	sequence is adjusted to 64 × 44 to accommodate GaitGL, and the input
96-    96	gait sequence is enhanced by using random cropping, and horizontal
97:    97	ﬂipping. To optimize the model, we employ the Adam optimizer to adjust the parameters, applying a weight decay of 0.0005. Furthermore,
98-    98	the training process spans 250 epochs in total, starting with an initial
99-    99	learning rate of 0.00035. During training, the learning rate is decreased
100-   100	by a factor of 10 every 40 epochs.
101:   101	Testing Phase: During the test stage, we test the fusion mode representation (DTA-Net) and single mode representation (GAE-Net) on the
102-   102	testset respectively.
103-   103	
104:   104	4.4. Ablation study
105:   105	A set of ablation experiments are conducted to evaluate the impact
106-   106	of each individual component.
107:   107	Eﬀects of Key Modules: As presented in Table 3, it is evident that
108:   108	GAE-Net enhances the results on MARS, thereby validating its eﬀectiveness. Speciﬁcally, the Re-ID branch achieved an 84.3 % mAP result,
109-   109	while the gait branch achieved a 10.7 % mAP result on MARS. Further,
110:   110	DTA-Net combined with two branches achieves 90.7 % Rank-1 accuracy and 85.8 % mAP on MARS. In addition, GAE-Net performs best after knowledge distillation, achieving a mAP result of 87.7 % on MARS.
111:   111	The ablation results clearly indicate the eﬀectiveness of each branch.
112:   112	Furthermore, branch fusion improves performance and veriﬁes the effectiveness of its fusion module DFA. Finally, after knowledge distillation, the single-modal model can also learn gait information to improve
113:   113	its performance. Furthermore, in comparison with the teacher network
114:   114	DTA-Net, the number of parameters in the student network GAE-Net
115:   115	has been reduced from 164.1M to 24.8M after applying LPCD, representing a reduction of 84.8 %, which signiﬁcantly outperforms exciting
116-   116	
117:   117	4.3. Comparison with state-of-the-arts
118-   118	We compare our method with the current state-of-the-art methods.
119:   119	As summarized in Table 2, the results are presented across two widely
120:   120	used datasets. Our GAE-Net achieves the highest mAP on MARS. Notably, GAE-Net also demonstrates competitive performance on LSVID.
121-   121	7
122-   122	
123-   123	Neural Networks 192 (2025) 107946
124-   124	
125-   125	M. Dai et al.
126-   126	
127:   127	Table 2
128:   128	Comparison with state-of-the-arts methods on two datasets: MARS and LSVID.
129-   129	MARS
130-   130	
131-   131	Method
--
166-   166	TMT (Liu et al., 2024b)
167-   167	MFANet (Zhu et al., 2025)
168-   168	TAE-ViT (Wang et al., 2025a)
169:   169	Baseline
170:   170	DTA-Net (Ours)
171:   171	GAE-Net (Ours)
172-   172	
173-   173	𝑅@1
174-   174	
--
224-   224	65.3
225-   225	88.6
226-   226	82.3
227:   227	84.4
228-   228	86.3
229-   229	87.0
230-   230	84.9
--
416-   416	94.3
417-   417	94.6
418-   418	
419:   419	Table 3
420:   420	Eﬀect of diﬀerernt modules in GAE-Net on MARS.
421-   421	Methods
422-   422	
423-   423	Param(M)
424-   424	
425:   425	Runtime
426-   426	(h)
427-   427	
428-   428	mAP
--
476-   476	temporal information from TCKD is progressively integrated, improving
477-   477	overall performance. When 𝛼 reaches 1.0, the balance between NCKD
478-   478	and TCKD is optimized, achieving peak performance.
479:   479	Eﬀects of Key Components in DFA: As previously discussed, we
480:   480	design a Dynamic Feature Aggregation (DFA) module to explore the
481-   481	shared and complementary features between these two modalities of
482-   482	gait and ReID, thereby providing a more discriminative representation.
483:   483	The DFA module is composed of two submodules: DAW and DWA. These
484-   484	two sub-modules are indispensable and play a critical role in feature
485-   485	aggregation, with neither being able to function eﬀectively without the
486:   486	other. To evaluate the impact of these key components within the DFA,
487:   487	we performed ablation studies. The results in Table 5 demonstrate that
488-   488	our proposed architecture is highly eﬀective.
489-   489	
490:   490	Table 4
491:   491	Eﬀects of hyper parameter 𝛼 and 𝛽 on MARS.
492-   492	𝛽
493-   493	
494-   494	0.5
--
539-   539	factors such as camera viewpoint, environment, and walking style. This
540-   540	highlights the importance of combining gait with appearance features
541-   541	to mitigate the limitations of single-modal representations.
542:   542	Eﬀects of Hyper Parameter 𝛼 AND 𝛽 : As demonstrated in Table 4,
543-   543	the optimal performance is achieved when 𝛼=1.0 and 𝛽 are both set to
544-   544	1.0. 𝛽 is utilized to regulate the contribution of NCKD (Normalized Cross
545-   545	Knowledge Distillation) to the total loss. When 𝛽 equals 1.0, NCKD’s
546-   546	positive impact on the model is maximized, resulting in optimal perfor-
547-   547	
548:   548	Table 5
549:   549	Eﬀect of key components of DTA-Net on MARS.
550-   550	
551-   551	8
552-   552	
--
589-   589	
590-   590	M. Dai et al.
591-   591	
592:   592	Table 6
593:   593	Comparison of Gait Branches on MARS.
594-   594	
595:   595	Table 9
596:   596	Comparison of hyper parameter T on MARS.
597-   597	
598-   598	Methods
599-   599	
--
643-   643	97.2
644-   644	97.0
645-   645	
646:   646	Table 10
647:   647	Eﬀect of hyper parameter 𝛾 of LPCD on MARS..
648-   648	
649:   649	Table 7
650:   650	Eﬀect of hyper parameter 𝑝 of DTA-Net on MARS.
651-   651	𝑝
652-   652	
653-   653	2.0
--
688-   688	
689-   689	84.7
690-   690	
691:   691	84.4
692-   692	
693-   693	mAP
694-   694	
--
704-   704	
705-   705	87.1
706-   706	
707:   707	Table 11
708:   708	Comparison of hyper parameter 𝑀 on MARS..
709-   709	
710-   710	Eﬀects of Gait Branch 𝐺𝑎𝑖𝑡𝐺𝐿 : We employ various gait recognition
711-   711	networks as distinct gait branches, and the experimental results demonstrate that GaitGl signiﬁcantly outperforms other methods. As shown in
712:   712	Table 6, compared to GaitBase (Fan et al., 2023), which is limited to
713-   713	basic feature extraction, GaitGL captures both global and local features,
714-   714	thereby providing a more comprehensive gait representation. While DyGait (Wang et al., 2023) focuses on time series information, GaitGL models both spatial and temporal dimensions, making it especially adept at
715-   715	handling gait variations across diﬀerent resolutions. This explains the
716-   716	superior performance of GaitGL.
717:   717	Eﬀects of Hyper Parameter 𝑝 : To validate the impact of hyperparameter 𝑝 in DTA-Net, we set diﬀerent values of 𝑝 in order to achieve
718-   718	the best performance. When 𝑝=1, GeM pooling is equivalent to average pooling, while as 𝑝 approaches inﬁnity, it approximates maximum
719-   719	pooling. By selecting an intermediate value of 𝑝 between 1 and inﬁnity,
720-   720	we strike a balance between the advantages of average and maximum
721-   721	pooling, preserving global information while capturing local details. As
722:   722	demonstrated in Table 7, the optimal performance is achieved when 𝑝 is
723-   723	set to 6.5, which enables GeM pooling to eﬀectively capture ﬁne spatial
724-   724	information through block processing, which proves beneﬁcial for tasks
725-   725	such as gait recognition that require precise localization.
726:   726	Eﬀects of LPCD in GAE-Net: Based on knowledge distillation, our
727:   727	LPCD is supervised by KD loss. To evaluate the impact of various knowledge distillation loss functions on the model performance, we designed
728:   728	and conducted a series of ablation studies. In the experiments, we compared a variety of common knowledge distillation loss functions, including but not limited to KD, WSLD, and some improved contrastive
729:   729	learning losses. The results in Table 8 show that our LPCD is the most
730:   730	eﬀective. Meanwhile, other KD losses also have certain eﬀects. We believe that LPCD performs better when learning complex multi-modal
731-   731	representations.
732:   732	Eﬀects of Hyper Parameter T: The parameter T is the distillation
733:   733	temperature in LPCD. As illustrated in Table 9, the optimal performance
734-   734	is achieved when T is set to 4. The results of 91.1 % Rank-1, and 87.7 %
735:   735	mAP underscore the eﬀectiveness of this parameter setting for T. We
736:   736	analyze the ablation results and conclude that deep features could be
737-   737	mined by increasing the distillation temperature.
738-   738	Eﬀects of Hyper Parameter 𝛾: To investigate the impact of the
739:   739	trade-oﬀ hyper-parameter 𝛾 in LPCD, we conduct ablation experiments
740:   740	by varying 𝛾 from 1.0 to 6.0. As shown in Table 10, the model achieves
741-   741	
742-   742	mAP
743-   743	
--
803-   803	between appearance-guided and gait-guided supervision. When 𝛾 is too
804-   804	small or too large, the performance drops slightly. This suggests that either underemphasizing or overemphasizing the gait-guided knowledge
805-   805	distillation may undermine the overall representation quality.
806:   806	Eﬀects of diﬀerent decoupled scales: To assess the impact of diﬀerent decoupled scales 𝑀 in our LPCD framework, we established various
807:   807	scale sets 𝑀. The results, shown in Table 11, indicate that the choice
808:   808	of scales signiﬁcantly aﬀects the model’s performance. The best performance is achieved when 𝑀 = {1, 2, 4}, with an mAP of 87.7 %. The result
809:   809	conﬁrm that the introduction of multiple decoupled scales enhances the
810:   810	our GAENet’s ability to capture both global and local feature details,
811-   811	making it more robust in handling complex scenarios in person ReID
812-   812	tasks.
813-   813	4.5. Visulization
814:   814	Visualization of Retrieval Results: Fig. 5 illustrates the person retrieval results comparing the baseline and the GAE-Net. Although there
815:   815	is misalignment occlusion in the query image, our proposed GAE-Net
816-   816	prefers to rank the matching image higher in the results than the baseline. As a result, the accuracy of the matching results of the corresponding person is improved. Conversely, the baseline model is disturbed by
817:   817	noise sequences, resulting in mismatching. In summary, GAE-Net fuses
818-   818	gait information, thus overcoming the misalignment, and producing better re-idenﬁcation results.
819:   819	Visualization of t-SNE: In Fig. 6, we compare the visualization results of t-SNE between Baseline (Fig. 6 (a)) and GAE-Net (Fig. 6 (b)).
820-   820	Speciﬁcally, we performed t-SNE on 45 randomly selected samples from
821:   821	10 diﬀerent IDs. Compared to the baseline, GAE-Net shows that samples from the same ID are more closely grouped together, indicating
822:   822	that GAE-Net is better able to integrate appearance features and gait
823-   823	features, resulting in discriminative features for video-based person reidentiﬁcation.
824-   824	Visualization of Activation Maps: The activation maps presented
825:   825	in Fig. 7 conﬁrm that GAE-Net is more eﬀective in suppressing noise
826-   826	caused by occlusion frames and enhancing discriminative representation
827:   827	compared to baseline models. These results indicate that GAE-Net successfully leverages gait features to compensate for the limitations of appearance features, especially in scenarios involving occlusions. In comparison to the baseline, GAE-Net demonstrates superior performance in
828-   828	reducing noise from occlusion frames while emphasizing discriminative
829-   829	regions, thereby improving feature representations and enhancing robustness in complex environments.
830-   830	
831:   831	Table 8
832:   832	Comparison of KD-lOSS on MARS.
833-   833	Methods
834-   834	
835-   835	Methods
--
840-   840	
841-   841	M. Dai et al.
842-   842	
843:   843	Fig. 5. Visualization of retrieval results for baseline and our GAE-Net. In each scenario, the ﬁrst column presents a sample frame from the query sequence, whereas
844-   844	the following ten columns showcase the top-10 matches from the gallery set. Note that search results outlined in green are accurately re-identiﬁed.
845-   845	
846:   846	As the core of DTA-Net, the Dynamic Feature Aggregation (DFA) module
847:   847	integrates the features of the two branches by exploring two featureweighted fusion methods (Dynamic Attention Weighting (DAW) and
848:   848	Dynamic Weight Aggregation (DWA)). Considering that multi-modal
849:   849	information can be exploited to enhance single-modal models, we propose local perception complementary distillation (LPCD) based on logit
850:   850	knowledge distillation. With the assistance of LPCD, single-modal models are enabled to acquire more discriminative and complementary details of similar pedestrians from multimodal representations, thereby enhancing their performance. As conﬁrmed from extensive experiments,
851:   851	our DGA-Net outperforms several state-of-the-art.
852:   852	Limitations and Future Work: Although GAE-Net successfully incorporates gait cues for video-based person Re-ID, it remains dependent
853-   853	on appearance consistency and has not been tested on datasets involving
854-   854	clothing changes. In future work, we aim to focus on exploring dynamic
855-   855	gait cues that are robust to variations in appearance, thereby enhancing
856-   856	the reliability of person Re-ID under varying clothing conditions.
857-   857	
858:   858	Fig. 6. The t-SNE visualisation of feature distribution between baseline and
859:   859	GAE-Net.
860-   860	
861-   861	CRediT authorship contribution statement
862-   862	Minting Dai: Writing – original draft, Software, Methodology; Xi
--
866-   866	Funding acquisition.
867-   867	
868-   868	Declaration of competing interest
869:   869	Fig. 7. Visualization of the class activation heatmaps. The ﬁrst row shows the
870-   870	original video sequence over a series of consecutive times. The second and third
871:   871	rows represent the channel activation plots for baseline and GAE-Net, respectively. Considering the conﬁdentiality of the person’s information, the face of
872-   872	the person in the image is obscured.
873-   873	
874-   874	The authors declare that they have no known competing ﬁnancial
--
891-   891	bottleneck, whereas the gait information can resist the interference of
892-   892	appearance and supplement the spatiotemporal information. Based on
893-   893	complementarity and mutual assistance, we then propose a Gait-Assisted
894:   894	Enhancement Network (GAE-Net) to obtain the fusion feature dynamically. In the proposed GAE-Net, we design a dynamic two-stream aggregate network (DTA-Net) to obtain a more comprehensive representation.
895-   895	10
896-   896	
897-   897	Neural Networks 192 (2025) 107946
--
931-   931	1655–1668.
932-   932	Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for
933-   933	biomedical image segmentation. In MICCAI (pp.234–241).
934:   934	Shen, Z., & Xing, E. (2022). A fast knowledge distillation framework for visual recognition.
935-   935	In ECCV,9351 (pp. 673–690).
936-   936	Shiraga, K., Makihara, Y., Muramatsu, D., Echigo, T., & Yagi, Y. (2016). GeiNet: Viewinvariant gait recognition using a convolutional neural network. In ICB (pp. 1–8).
937-   937	Simonyan, K., & Zisserman, A. (2014). Two-stream convolutional networks for action
--
979-   979	Spatio-temporal representation factorization for video-based person re-identiﬁcation.
980-   980	In ICCV (pp. 152–162).
981-   981	Bai, T., Zhao, J., & Wen, B. (2023). Guided adversarial contrastive distillation for robust
982:   982	students. IEEE Transactions on Information Forensics and Security, 19, 9643–9655.
983-   983	Carreira, J., & Zisserman, A. (2017, July). Quo vadis, action recognition? A new model
984-   984	and the kinetics dataset. In CVPR (pp. 6299–6308).
985-   985	Chao, H., He, Y., Zhang, J., & Feng, J. (2019, July). Gaitset: Regarding gait as a set for
--
988-   988	Transactions on Circuits and Systems for Video Technology, 32(9), 6100–6112.
989-   989	Chen, G., Rao, Y., Lu, J., & Zhou, J. (2020). Temporal coherence or temporal motion:
990-   990	Which is more critical for video-based person re-identiﬁcation? In ECCV (pp. 660–676).
991:   991	Chen, H., Guo, T., Xu, C., Li, W., Xu, C., Xu, C., & Wang, Y. (2021). Learning student
992-   992	networks in the wild. In CVPR (pp. 6424–6433).
993-   993	Chung, D., Tahboub, K., & Delp, E. J. (2017). A two stream siamese convolutional neural
994-   994	network for person re-identiﬁcation. In ICCV (pp. 1983–1991).
--
1000-  1000	Fan, C., Peng, Y., Cao, C., Liu, X., Hou, S., Chi, J., Huang, Y., Li, Q., & He, Z. (2020).
1001-  1001	GaitPart: Temporal part-based model for gait recognition. In CVPR (pp. 14213–
1002-  1002	14221).
1003:  1003	Fu, Y., Wang, X., Wei, Y., & Huang, T. (2019). STA: Spatial-temporal attention for largescale video-based person re-identiﬁcation. In AAAI (pp. 8287–8294).
1004-  1004	Gao, J., & Nevatia, R. (2018). Revisiting temporal modeling for video-based person reid.
1005-  1005	arXiv preprint arXiv:1805.02104.
1006-  1006	Gu, X., Chang, H., Ma, B., Zhang, H., & Chen, X. (2020). Appearance-preserving 3D convolution for video-based person re-identiﬁcation. In ECCV (pp. 228–243).
--
1016-  1016	Hou, R., Chang, H., Ma, B., Shan, S., & Chen, X. (2020). Temporal complementary learning
1017-  1017	for video person re-identiﬁcation. In ECCV (pp. 388–405).
1018-  1018	Huang, T., You, S., Wang, F., Qian, C., & Xu, C. (2022). Knowledge distillation from a
1019:  1019	stronger teacher. In NIPS (pp. 33716–33727).
1020-  1020	Jiang, X., Qiao, Y., Yan, J., & Li, Q. (2021). Ssn3D: Self-separated network to align parts
1021-  1021	for 3D convolution in video person re-identiﬁcation. In AAAI (pp. 1691–1699).
1022-  1022	Jin, X., He, T., Zheng, K., Yin, Z., Shen, X., Huang, Z., & Feng, R. (2022). Cloth-changing
1023-  1023	person re-identiﬁcation from a single image with gait prediction and regularization.
1024-  1024	In CVPR (pp. 14278–14287).
1025-  1025	Li, J., Wang, J., Tian, Q., Gao, W., & Zhang, S. (2019). Global-local temporal representations for video person re-identiﬁcation. In ICCV (pp. 3958–3967).
1026:  1026	Li, J., Zhang, S., & Huang, T. (2020). Multi-scale temporal cues learning for video person
1027-  1027	re-identiﬁcation. IEEE Transactions on Image Processing, 29, 4461–4473.
1028-  1028	Li, S., Bak, S., Carr, P., & Wang, X. (2018a). Diversity regularized spatiotemporal attention
1029-  1029	for video-based person re-identiﬁcation. In CVPR (pp. 369–378).
--
1072-  1072	Zhao, Y., Wang, X., Yu, X., Liu, C., & Gao, Y. (2023). Gait-assisted video person retrieval.
1073-  1073	IEEE Transactions on Circuits and Systems for Video Technology, 33(2), 897–908.
1074-  1074	Zheng, L., Bie, Z., Sun, Y., Wang, J., Su, C., Wang, S., & Tian, Q. (2016). Mars: A video
1075:  1075	benchmark for large-scale person re-identiﬁcation. In ECCV (pp. 868–884).
1076-  1076	Zhu, Z., Chen, S., Qi, H., Li, G.,& Gao, X. (2025). Multi-granular inter-frame relation exploration and global residual embedding for video-based person re-identiﬁcation. Signal
1077-  1077	Processing: Image Communication, 117240.
1078-  1078	
1079-  1079	Xi Yang (M’15-SM’24) received the B.Eng. degree in electronic information engineering and the Ph.D. degree in pattern recognition and intelligence system from Xidian University, Xi’an, China, in 2010 and 2015, respectively. From 2013
1080:  1080	to 2014, she was a visiting Ph.D. student with the Department
1081-  1081	of Computer Science, University of Texas at San Antonio, San
1082-  1082	Antonio, TX, USA. In 2015, she joined the State Key Laboratory of Integrated Services Networks, School of Telecommunications Engineering, Xidian University, where she is currently a Professor in communications and information systems. She has published over 60 articles in refereed journals
1083-  1083	and proceedings, including IEEE T-TIP, T-NNLS, T-CYB, TGRS, CVPR, ICCV, ACM MM, etc. Her current research interests include image/video processing, computer vision, and machine learning. yangx@xidian.edu.cn

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 10 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | sed -n '300,470p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   300	
   301	Param(M)
   302	
   303	Runtime
   304	(h)
   305	
   306	mAP
   307	
   308	𝑅@1
   309	
   310	𝑅@5
   311	
   312	𝑅@10
   313	
   314	𝑅𝑒 − 𝐼𝐷 𝑏𝑟𝑎𝑛𝑐ℎ
   315	𝐺𝑎𝑖𝑡 𝑏𝑟𝑎𝑛𝑐ℎ
   316	𝐷𝑇 𝐴 − 𝑁𝑒𝑡
   317	𝐺𝐴𝐸 − 𝑁𝑒𝑡
   318	
   319	26.7
   320	23.5
   321	164.1
   322	24.8
   323	
   324	4.67
   325	4.24
   326	5.12
   327	6.33
   328	
   329	84.3
   330	10.7
   331	85.8
   332	87.7
   333	
   334	89.6
   335	17.7
   336	90.7
   337	91.1
   338	
   339	96.7
   340	34.1
   341	96.9
   342	97.2
   343	
   344	97.5
   345	43.1
   346	98.1
   347	97.6
   348	
   349	mance enhancement. However, as 𝛽 continues to increase beyond 1.0,
   350	the excessive weight of NCKD may cause overﬁtting or interfere with
   351	TCKD, ultimately degrading performance. When 𝛼 is set to 0, the model
   352	relies solely on NCKD and loses the temporal information provided by
   353	TCKD, leading to diminished performance. As 𝛼 increases gradually, the
   354	temporal information from TCKD is progressively integrated, improving
   355	overall performance. When 𝛼 reaches 1.0, the balance between NCKD
   356	and TCKD is optimized, achieving peak performance.
   357	Eﬀects of Key Components in DFA: As previously discussed, we
   358	design a Dynamic Feature Aggregation (DFA) module to explore the
   359	shared and complementary features between these two modalities of
   360	gait and ReID, thereby providing a more discriminative representation.
   361	The DFA module is composed of two submodules: DAW and DWA. These
   362	two sub-modules are indispensable and play a critical role in feature
   363	aggregation, with neither being able to function eﬀectively without the
   364	other. To evaluate the impact of these key components within the DFA,
   365	we performed ablation studies. The results in Table 5 demonstrate that
   366	our proposed architecture is highly eﬀective.
   367	
   368	Table 4
   369	Eﬀects of hyper parameter 𝛼 and 𝛽 on MARS.
   370	𝛽
   371	
   372	0.5
   373	
   374	1.0
   375	
   376	1.5
   377	
   378	2.0
   379	
   380	4.0
   381	
   382	8.0
   383	
   384	mAP
   385	𝛼
   386	mAP
   387	
   388	87.1
   389	0.0
   390	87.2
   391	
   392	87.7
   393	0.2
   394	87.1
   395	
   396	87.2
   397	0.5
   398	87.3
   399	
   400	86.8
   401	1.0
   402	87.7
   403	
   404	86.5
   405	2.0
   406	87.2
   407	
   408	86.4
   409	4.0
   410	87.0
   411	
   412	LSVID
   413	
   414	mAP
   415	
   416	state-of-the-art methods. It is worth noting that the relatively low performance of the gait branch alone is due to the sensitivity of gait features to
   417	factors such as camera viewpoint, environment, and walking style. This
   418	highlights the importance of combining gait with appearance features
   419	to mitigate the limitations of single-modal representations.
   420	Eﬀects of Hyper Parameter 𝛼 AND 𝛽 : As demonstrated in Table 4,
   421	the optimal performance is achieved when 𝛼=1.0 and 𝛽 are both set to
   422	1.0. 𝛽 is utilized to regulate the contribution of NCKD (Normalized Cross
   423	Knowledge Distillation) to the total loss. When 𝛽 equals 1.0, NCKD’s
   424	positive impact on the model is maximized, resulting in optimal perfor-
   425	
   426	Table 5
   427	Eﬀect of key components of DTA-Net on MARS.
   428	
   429	8
   430	
   431	Components
   432	
   433	mAP
   434	
   435	𝑅@1
   436	
   437	𝑅@5
   438	
   439	𝑅@10
   440	
   441	𝐵𝑎𝑠𝑒𝑙𝑖𝑛𝑒
   442	+𝐷𝐴𝑊
   443	+𝐷𝑊 𝐴
   444	+𝐷𝐹 𝐴
   445	
   446	84.5
   447	85.2
   448	85.5
   449	85.8
   450	
   451	89.5
   452	90.3
   453	90.5
   454	90.7
   455	
   456	96.7
   457	97.0
   458	97.1
   459	96.9
   460	
   461	97.5
   462	97.8
   463	98.0
   464	98.1
   465	
   466	Neural Networks 192 (2025) 107946
   467	
   468	M. Dai et al.
   469	
   470	Table 6

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 10 'GAE-Net - A gait-assisted enhancement network for video-based person-reidentification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
I/O Error: Couldn't open file 'GAE-Net - A gait-assisted enhancement network for video-based person-reidentification.pdf': No such file or directory.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 10 'GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Neural Networks 192 (2025) 107946
     2	
     3	M. Dai et al.
     4	
     5	Table 6
     6	Comparison of Gait Branches on MARS.
     7	
     8	Table 9
     9	Comparison of hyper parameter T on MARS.
    10	
    11	Methods
    12	
    13	mAP
    14	
    15	𝑅@1
    16	
    17	𝑅@5
    18	
    19	Methods
    20	
    21	mAP
    22	
    23	𝑅@1
    24	
    25	𝑅@5
    26	
    27	𝐺𝑎𝑖𝑡𝐵𝑎𝑠𝑒 (Fan et al., 2023)
    28	𝐷𝑦𝐺𝑎𝑖𝑡 (Wang et al., 2023)
    29	𝐺𝑎𝑖𝑡𝐺𝐿(𝑜𝑢𝑟𝑠)
    30	
    31	77.2
    32	75.8
    33	87.7
    34	
    35	82.0
    36	83.5
    37	91.1
    38	
    39	90.9
    40	93.4
    41	97.2
    42	
    43	𝑇 =1
    44	𝑇 =4
    45	𝑇 = 10
    46	
    47	86.0
    48	87.7
    49	85.8
    50	
    51	91.0
    52	91.1
    53	90.8
    54	
    55	96.3
    56	97.2
    57	97.0
    58	
    59	Table 10
    60	Eﬀect of hyper parameter 𝛾 of LPCD on MARS..
    61	
    62	Table 7
    63	Eﬀect of hyper parameter 𝑝 of DTA-Net on MARS.
    64	𝑝
    65	
    66	2.0
    67	
    68	3.0
    69	
    70	5.5
    71	
    72	6.5
    73	
    74	7.5
    75	
    76	8.5
    77	
    78	𝛾
    79	
    80	1.0
    81	
    82	2.0
    83	
    84	3.0
    85	
    86	4.0
    87	
    88	5.0
    89	
    90	6.0
    91	
    92	mAP
    93	
    94	84.1
    95	
    96	84.3
    97	
    98	84.5
    99	
   100	85.8
   101	
   102	84.7
   103	
   104	84.4
   105	
   106	mAP
   107	
   108	87.4
   109	
   110	87.3
   111	
   112	87.7
   113	
   114	86.9
   115	
   116	87.3
   117	
   118	87.1
   119	
   120	Table 11
   121	Comparison of hyper parameter 𝑀 on MARS..
   122	
   123	Eﬀects of Gait Branch 𝐺𝑎𝑖𝑡𝐺𝐿 : We employ various gait recognition
   124	networks as distinct gait branches, and the experimental results demonstrate that GaitGl signiﬁcantly outperforms other methods. As shown in
   125	Table 6, compared to GaitBase (Fan et al., 2023), which is limited to
   126	basic feature extraction, GaitGL captures both global and local features,
   127	thereby providing a more comprehensive gait representation. While DyGait (Wang et al., 2023) focuses on time series information, GaitGL models both spatial and temporal dimensions, making it especially adept at
   128	handling gait variations across diﬀerent resolutions. This explains the
   129	superior performance of GaitGL.
   130	Eﬀects of Hyper Parameter 𝑝 : To validate the impact of hyperparameter 𝑝 in DTA-Net, we set diﬀerent values of 𝑝 in order to achieve
   131	the best performance. When 𝑝=1, GeM pooling is equivalent to average pooling, while as 𝑝 approaches inﬁnity, it approximates maximum
   132	pooling. By selecting an intermediate value of 𝑝 between 1 and inﬁnity,
   133	we strike a balance between the advantages of average and maximum
   134	pooling, preserving global information while capturing local details. As
   135	demonstrated in Table 7, the optimal performance is achieved when 𝑝 is
   136	set to 6.5, which enables GeM pooling to eﬀectively capture ﬁne spatial
   137	information through block processing, which proves beneﬁcial for tasks
   138	such as gait recognition that require precise localization.
   139	Eﬀects of LPCD in GAE-Net: Based on knowledge distillation, our
   140	LPCD is supervised by KD loss. To evaluate the impact of various knowledge distillation loss functions on the model performance, we designed
   141	and conducted a series of ablation studies. In the experiments, we compared a variety of common knowledge distillation loss functions, including but not limited to KD, WSLD, and some improved contrastive
   142	learning losses. The results in Table 8 show that our LPCD is the most
   143	eﬀective. Meanwhile, other KD losses also have certain eﬀects. We believe that LPCD performs better when learning complex multi-modal
   144	representations.
   145	Eﬀects of Hyper Parameter T: The parameter T is the distillation
   146	temperature in LPCD. As illustrated in Table 9, the optimal performance
   147	is achieved when T is set to 4. The results of 91.1 % Rank-1, and 87.7 %
   148	mAP underscore the eﬀectiveness of this parameter setting for T. We
   149	analyze the ablation results and conclude that deep features could be
   150	mined by increasing the distillation temperature.
   151	Eﬀects of Hyper Parameter 𝛾: To investigate the impact of the
   152	trade-oﬀ hyper-parameter 𝛾 in LPCD, we conduct ablation experiments
   153	by varying 𝛾 from 1.0 to 6.0. As shown in Table 10, the model achieves
   154	
   155	mAP
   156	
   157	𝑅@1
   158	
   159	𝑅@5
   160	
   161	𝐵𝑎𝑠𝑒𝑙𝑖𝑛𝑒
   162	𝐷𝑇 𝐴 − 𝑁𝑒𝑡
   163	𝐾𝐷
   164	𝑊 𝑆𝐿𝐷
   165	𝑁𝐾𝐷
   166	𝐷𝐼𝑆𝑇
   167	𝐿𝑃 𝐶𝐷
   168	
   169	84.3
   170	85.8
   171	85.8
   172	85.9
   173	85.8
   174	85.9
   175	87.7
   176	
   177	89.6
   178	90.7
   179	90.7
   180	90.7
   181	90.7
   182	90.8
   183	91.1
   184	
   185	96.7
   186	96.9
   187	96.7
   188	97.3
   189	96.7
   190	97.0
   191	97.2
   192	
   193	mAP
   194	
   195	𝑅@1
   196	
   197	𝑅@5
   198	
   199	𝑀={1}
   200	𝑀={1,2}
   201	𝑀={1,2,4}
   202	
   203	85.8
   204	86.4
   205	87.7
   206	
   207	90.7
   208	90.9
   209	91.1
   210	
   211	96.8
   212	97.3
   213	97.2
   214	
   215	the best result (87.7 % mAP) when 𝛾 = 3.0, indicating an optimal balance
   216	between appearance-guided and gait-guided supervision. When 𝛾 is too
   217	small or too large, the performance drops slightly. This suggests that either underemphasizing or overemphasizing the gait-guided knowledge
   218	distillation may undermine the overall representation quality.
   219	Eﬀects of diﬀerent decoupled scales: To assess the impact of diﬀerent decoupled scales 𝑀 in our LPCD framework, we established various
   220	scale sets 𝑀. The results, shown in Table 11, indicate that the choice
   221	of scales signiﬁcantly aﬀects the model’s performance. The best performance is achieved when 𝑀 = {1, 2, 4}, with an mAP of 87.7 %. The result
   222	conﬁrm that the introduction of multiple decoupled scales enhances the
   223	our GAENet’s ability to capture both global and local feature details,
   224	making it more robust in handling complex scenarios in person ReID
   225	tasks.
   226	4.5. Visulization
   227	Visualization of Retrieval Results: Fig. 5 illustrates the person retrieval results comparing the baseline and the GAE-Net. Although there
   228	is misalignment occlusion in the query image, our proposed GAE-Net
   229	prefers to rank the matching image higher in the results than the baseline. As a result, the accuracy of the matching results of the corresponding person is improved. Conversely, the baseline model is disturbed by
   230	noise sequences, resulting in mismatching. In summary, GAE-Net fuses
   231	gait information, thus overcoming the misalignment, and producing better re-idenﬁcation results.
   232	Visualization of t-SNE: In Fig. 6, we compare the visualization results of t-SNE between Baseline (Fig. 6 (a)) and GAE-Net (Fig. 6 (b)).
   233	Speciﬁcally, we performed t-SNE on 45 randomly selected samples from
   234	10 diﬀerent IDs. Compared to the baseline, GAE-Net shows that samples from the same ID are more closely grouped together, indicating
   235	that GAE-Net is better able to integrate appearance features and gait
   236	features, resulting in discriminative features for video-based person reidentiﬁcation.
   237	Visualization of Activation Maps: The activation maps presented
   238	in Fig. 7 conﬁrm that GAE-Net is more eﬀective in suppressing noise
   239	caused by occlusion frames and enhancing discriminative representation
   240	compared to baseline models. These results indicate that GAE-Net successfully leverages gait features to compensate for the limitations of appearance features, especially in scenarios involving occlusions. In comparison to the baseline, GAE-Net demonstrates superior performance in
   241	reducing noise from occlusion frames while emphasizing discriminative
   242	regions, thereby improving feature representations and enhancing robustness in complex environments.
   243	
   244	Table 8
   245	Comparison of KD-lOSS on MARS.
   246	Methods
   247	
   248	Methods
   249	
   250	9
   251	
   252	Neural Networks 192 (2025) 107946
   253	
   254	M. Dai et al.
   255	
   256	Fig. 5. Visualization of retrieval results for baseline and our GAE-Net. In each scenario, the ﬁrst column presents a sample frame from the query sequence, whereas
   257	the following ten columns showcase the top-10 matches from the gallery set. Note that search results outlined in green are accurately re-identiﬁed.
   258	
   259	As the core of DTA-Net, the Dynamic Feature Aggregation (DFA) module
   260	integrates the features of the two branches by exploring two featureweighted fusion methods (Dynamic Attention Weighting (DAW) and

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
/bin/zsh -lc "pdfinfo 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' | sed -n '1,35p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           GAReID: Grouped and Attentive High-Order Representation Learning for Person Re-Identification
Subject:         IEEE Transactions on Neural Networks and Learning Systems;2025;36;3;10.1109/TNNLS.2022.3209537
Creator:         Aspose Ltd.
Producer:        Aspose.Pdf for .NET 8.3.0; modified using iText® Core 7.2.4 (AGPL version) ©2000-2022 iText Group NV
CreationDate:    Sun Feb 16 22:15:48 2025 CST
ModDate:         Fri Feb 28 07:25:34 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           15
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       6229550 bytes
Optimized:       no
PDF version:     1.4

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '30,148p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '278,683p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 400ms:
    30	I. I NTRODUCTION
    31	
    32	Fig. 1. Illustration of the part misalignment problem caused by camera views,
    33	detection errors, body occlusions, and background clutters. Aligned part pairs
    34	are connected with solid lines, while the misaligned part pairs are connected
    35	with dashed lines. (a) Camera view. (b) Detection error. (c) Body occlusion.
    36	(d) Background cluster.
    37	
    38	P
    39	
    40	ERSON re-identification (ReID) aims at matching person
    41	images of the same person across nonoverlapping cameras. It plays an important role in various video surveillance
    42	applications such as suspect tracking and missing elderly or
    43	children retrieval. With the blooming of convolutional neural network (CNN), the current deep-feature-learning-based
    44	methods [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
    45	[12], [13], [14], [15], [16] have significantly outperformed a
    46	variety of traditional feature-learning-based approaches [17],
    47	[18], [19], [20], [21], [22]. However, the ReID task is far
    48	Manuscript received 26 November 2020; revised 21 November 2021 and
    49	27 May 2022; accepted 17 September 2022. Date of publication 5 October
    50	2022; date of current version 1 March 2025. This work was supported by
    51	the Chinese National Natural Science Foundation under Grant 62076033 and
    52	Grant U1931202. (Corresponding author: Zhicheng Zhao.)
    53	Pingyu Wang, Fei Su, Zhicheng Zhao, and Yanyun Zhao are with the Beijing
    54	Key Laboratory of Network System and Network Culture, School of Artificial
    55	Intelligence, Beijing University of Posts and Telecommunications, Beijing
    56	100876, China (e-mail: applewangpingyu@bupt.edu.cn; sufei@bupt.edu.cn;
    57	zhaozc@bupt.edu.cn; zyy@bupt.edu.cn).
    58	Nikolaos V. Boulgouris is with the Department of Electronic and Computer
    59	Engineering, Brunel University London, UB8 3PH Uxbridge, U.K. (e-mail:
    60	nikolaos.boulgouris@brunel.ac.uk).
    61	This article has supplementary downloadable material available at
    62	https://doi.org/10.1109/TNNLS.2022.3209537, provided by the authors.
    63	Digital Object Identifier 10.1109/TNNLS.2022.3209537
    64	
    65	from being solved because of part misalignments caused by
    66	camera views, detection errors, body occlusions, and background clutters. As shown in Fig. 1, part misalignments usually
    67	change the spatial distribution of person appearances, which
    68	might degenerate the distinctiveness and robustness of person
    69	representations.
    70	To mitigate part misalignments, prior ReID works have
    71	broadly followed two main paradigms, i.e., part-based and
    72	landmark-based methods. The part-based approaches [2], [5],
    73	[12], [13], [14] partition the global person images/features into
    74	a few fixed rigid parts and concentrate on local feature learning
    75	so as to obviate the need for landmark detection. Nevertheless,
    76	such coarse partition is unable to effectively align body parts
    77	without considering fine-grained pose variations within each
    78	part. For achieving fine-grained part alignments, the landmarkbased works [1], [6], [7], [8], [9], [10], [11], [23] use human
    79	landmark annotations or landmark detection networks and
    80	then learn part-aligned features from pose-normalized person
    81	images. Although those works have boosted ReID performance, they introduce extra operations to the ReID system,
    82	
    83	2162-237X © 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
    84	See https://www.ieee.org/publications/rights/index.html for more information.
    85	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
    86	
    87	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
    88	
    89	e.g., landmark detection and pose normalization. In addition,
    90	those operations bring nonignorable space and time costs,
    91	making it hard to train the ReID model.
    92	In this work, we propose an effective grouped attentive
    93	re-identification (GAReID) framework composed of two novel
    94	pooling layers, i.e., grouped high-order pooling (GHOP) and
    95	attentive high-order pooling (AHOP). As we know, compared
    96	with the first-order function, the high-order function f (x) =
    97	x n (n > 1, x ≥ 0) contributes to amplifying the discrepancies
    98	between two dependent variables when two independent variables are fixed. Motivated by this amplification property of the
    99	high-order function, the essential idea behind GAReID is to
   100	compute high-order mapping of part similarities to enlarge the
   101	similarity discrepancies between aligned and misaligned part
   102	pairs. Specifically, GAReID is able to highlight aligned part
   103	similarities and suppress misaligned part similarities. Since
   104	the high-order feature similarity between a pair of person
   105	images is equivalent to an average of high-order similarities
   106	of both the aligned and misaligned part pairs, the highorder aligned similarities are likely to dominate the high-order
   107	feature similarities. In this way, the part misalignment problem
   108	is effectively alleviated without relying on landmark detection
   109	or feature partition.
   110	Although high-order features contribute to part alignments,
   111	the dimension of high-order features increases exponentially,
   112	which gravely impairs the applications of high-order models.
   113	Therefore, we need to design an effective feature compression
   114	method for high-order features. Inspired by the lightweight
   115	networks [24], [25], the proposed GHOP layer adopts channel
   116	group and shuffle strategies to compress the dimension of
   117	high-order features. Specifically, input feature channels are
   118	uniformly divided into different groups and then those groups
   119	are shuffled to disperse the information across feature groups.
   120	Subsequently, we propose grouped Kronecker product (GKP)
   121	to use the Kronecker product for subfeatures in each original and shuffled group to excavate informative high-order
   122	interactions. Since the Kronecker product increases feature
   123	dimensions in each group, we obtain grouped high-order
   124	features by conducting elementwise aggregation, which can
   125	significantly improve the effectiveness of high-order features.
   126	As background clutters may hinder part alignments, we put
   127	forward an effective foreground attention module named adaptive foreground attention (AFA) to preserve foreground regions
   128	and eliminate background areas. With the integration of the
   129	GHOP layer and the AFA module, the proposed AHOP layer is
   130	constructed to boost both part-aligned and background robust
   131	representation learning.
   132	In summary, this article makes the following contributions.
   133	1) We analyze the cause of part misalignments and prove
   134	that high-order mapping of part similarities facilitates
   135	fine-grained part alignments in theory.
   136	2) We propose an effective GAReID framework with two
   137	novel pooling layers, i.e., GHOP and AHOP. The GHOP
   138	layer aims at compressing high-order features, while the
   139	AHOP layer focuses on eliminating background clutters.
   140	3) The GAReID framework is able to learn both partaligned and background robust representations without
   141	
   142	3991
   143	
   144	relying on any landmark detection or feature partition,
   145	making it highly generalizable to other unknown pose
   146	and background variations.
   147	4) The GAReID achieves state-of-the-art ReID performance on the Market1501 [26], CUHK03 [27],
   148	DukeMTMC [28], and MSMT17 [29] datasets.

 succeeded in 398ms:
   278	III. P ROPOSED M ETHOD
   279	In this section, we first analyze theoretically the cause of
   280	part misalignments. Then we introduce GHOP and AHOP in
   281	the GAReID framework as shown in Fig. 2.
   282	
   283	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
   284	
   285	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
   286	
   287	3993
   288	
   289	objects. As a result, the background descriptors may bring an
   290	objectionable bias to the aggregated similarities in (2).
   291	B. Grouped High-Order Pooling
   292	
   293	Fig. 3. Illustration of the first-order, second-order, and third-order functions. High-order function contributes to enlarging the difference in dependent variables when the difference in independent variables is fixed, i.e.,
   294	1y3 > 1y2 > 1y1 .
   295	
   296	A. Part Misalignment
   297	In this part, we give a theoretical analysis for the cause of
   298	part misalignments in the person ReID task. Given two input
   299	person images I u and I v from the same class, we use CNNs
   300	to extract two convolutional feature maps U, V ∈ RC×H ×W ,
   301	where C, H , and W denote the channel, height, and width
   302	dimension, respectively. Then the two feature maps are pooled
   303	by a global average pooling (GAP) layer [55] to obtain the
   304	corresponding person descriptors as follows:
   305	1 X
   306	1 X
   307	u=
   308	U pu , v =
   309	Vp
   310	(1)
   311	|S| p ∈S
   312	|S| p ∈S v
   313	v
   314	
   315	u
   316	
   317	where U pu , V pv ∈ RC are two part descriptors at positions pu
   318	and pv , respectively. The set S = {1, 2, . . . , H W } is the set
   319	of all the spatial positions and |S| = H W is its cardinality.
   320	Here, we use the inner product between u and v to measure
   321	the similarity of the two person images
   322	*
   323	+
   324	1 X
   325	1 X
   326	Sim(I u , I v ) =
   327	Up ,
   328	Vp
   329	|S| p ∈S u |S| p ∈S v
   330	u
   331	v
   332	X
   333	1
   334	U pu , V pv
   335	(2)
   336	=
   337	|S|2 p , p ∈S
   338	u
   339	
   340	v
   341	
   342	where ⟨u, v⟩ denotes the inner product between u and v. The
   343	similarity of u and v can be interpreted as an average of part
   344	similarities between |S|2 part pairs.
   345	However, such coarse similarity aggregation may degenerate into a suboptimal solution, which can be attributed to
   346	two major reasons. The first reason is associated with the
   347	imbalanced quantity distribution between about |S| aligned
   348	and |S|(|S| − 1) misaligned body part pairs. Since the number
   349	of the misaligned pairs (shoulder ↔ hand) is quadratically
   350	larger than the number of the aligned ones (hand ↔ hand),
   351	the similarities of the aligned part pairs may be overwhelmed
   352	by the misaligned part pairs, which might exacerbate the part
   353	misalignment problem to some extent. The second reason is
   354	related to the nonperson part descriptors containing various
   355	background clutters. This problem is particularly apparent
   356	when person bodies are partially occluded by other nonperson
   357	
   358	1) High-Order Representation: As illustrated in Fig. 1, the
   359	aligned parts usually contain identical semantics, while the
   360	misaligned parts have dissimilar semantics, so the aligned
   361	part similarities are likely to be larger than the misaligned
   362	part similarities. However, recent works are unable to exploit
   363	this prior knowledge efficiently, so similarity discrepancies
   364	between aligned and misaligned part pairs may not be sharp
   365	and easy to distinguish. As indicated in Fig. 3, the high-order
   366	function f (x) = x n (n > 1, x ≥ 0) contributes to enlarging the
   367	similarity discrepancies between aligned and misaligned bodypart pairs. Note that we need to add a ReLU layer after input
   368	features to ensure the part similarity is always nonnegative.
   369	By taking this high-order function into (2), a high-order
   370	similarity is defined as
   371	X
   372	1
   373	n
   374	U pu , V pv
   375	(3)
   376	Sim(I u , I v ; n) =
   377	2
   378	|S| p , p ∈S
   379	u
   380	
   381	v
   382	
   383	where ⟨u, v⟩ represents the nth-order part similarity between
   384	parts u and v. As the order n increases, the aligned part
   385	similarities will dominate the aggregated similarity in (3).
   386	Therefore, the high-order mapping function is beneficial to
   387	solve the part misalignment problem without the requirement
   388	of auxiliary landmark knowledge.
   389	According to Theorem 1, the similarity of high-order features is equivalent to high-order mapping of the first-order
   390	similarity. Subsequently, we reformulate (3) to simplify the
   391	computation of high-order similarities
   392	
   393	X O
   394	O
   395	1
   396	Sim(I u , I v ; n) =
   397	U
   398	,
   399	V
   400	pu
   401	pv
   402	|S|2 p , p ∈S n
   403	n
   404	u v
   405	*
   406	+
   407	1 XO
   408	1 XO
   409	=
   410	U pu ,
   411	V pv . (4)
   412	|S| p ∈S n
   413	|S| p ∈S n
   414	n
   415	
   416	v
   417	
   418	u
   419	
   420	Hence, a high-order representation is defined as
   421	1 XO
   422	x=
   423	X px .
   424	|S| p ∈S n
   425	
   426	(5)
   427	
   428	x
   429	
   430	N
   431	
   432	Since the Kronecker product allows all the elements of feature vectors to interact with each other, the high-order features
   433	exhibit strong representational capabilities. Notwithstanding,
   434	the dimension of high-order features increases exponentially,
   435	leading to very high memory consumption O(C n ) and computational complexity O(C n ). Therefore, an effective feature
   436	compression approach is needed to project high-order features
   437	onto a lower dimensional space.
   438	2) High-Order Compression: Motivated by lightweight network design [24], [25], we propose a novel GKP to compress
   439	high-order features using channel group and shuffle strategies.
   440	As shown in Fig. 4, the input feature channels are uniformly divided into G groups which are then shuffled to help
   441	information dispersion across feature groups. Then, we use
   442	the conventional Kronecker product for subfeatures in each
   443	original and shuffled group, which contributes to encoding
   444	
   445	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
   446	
   447	3994
   448	
   449	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
   450	
   451	Fig. 4. Toy example of GKP nG x with n = 2, G = 3, and C = 9. “G1,” “G2,” and “G3” represent the first, second, and third groups, respectively. The
   452	left two vectors are split into three groups and then the two subvectors of each group are aggregated by the conventional Kronecker product to produce a
   453	second-order vector with a length 9. The right two vectors are the group-shuffled versions of the left two vectors and then the three second-order vectors are
   454	generated by the same process as the left two vectors. Finally, all six second-order vectors are fused by elementwise summation to produce a second-order
   455	vector with a length 9.
   456	N
   457	
   458	both intragroup and intergroup high-order interactions. Since
   459	the Kronecker product increases feature dimensions in each
   460	group, we further compress high-order features by conducting
   461	elementwise aggregation. This can significantly improve the
   462	effectiveness of high-order features. Mathematically, the nthorder GKP operation is formulated as
   463	
   464	x, n = 1
   465	
   466	
   467	
   468	G
   469	
   470	X
   471	
   472	O
   473	O
   474	
   475	
   476	b
   477	
   478	Ijx
   479	I j x, n = 2
   480	I jx +b
   481	Ijx
   482	
   483	G
   484	
   485	O
   486	j=1
   487	!
   488	!
   489	x=
   490	G
   491	G
   492	G
   493	X
   494	
   495	O
   496	O
   497	O
   498	O
   499	n
   500	
   501	
   502	b
   503	b
   504	
   505	Ij
   506	x
   507	Ijx + Ij
   508	x
   509	Ijx
   510	
   511	
   512	
   513	n−1
   514	n−1
   515	
   516	j=1
   517	
   518	
   519	n>2
   520	(6)
   521	where I j ∈ R(C/G)×C is a block matrix and I =
   522	[I 1 ; I 2 ; . . . ; I G ] is an identity matrix. b
   523	I ∈ RC×C is the
   524	shuffled
   525	√ version of the identity matrix I. Note that we set
   526	G = C to keep high-order feature dimension unchanged.
   527	In this way, the proposed GKP has much lower time com2.5
   528	plexity O(nC 2.5 ) and space complexity
   529	√ O(nC ) than the
   530	conventional √
   531	Kronecker product. If C is not an integer,
   532	we set G = ⌈ C⌉ and an extra subfeature with a length G 2 −C
   533	is generated by randomly sampling elements from the input
   534	feature. Then, we concatenate the input feature and sampled
   535	subfeature to produce a new feature with length G 2 . This fused
   536	feature is used to generate a high-order feature with length G 2 .
   537	Then we randomly discard G 2 −C elements from the highorder feature to reduce the feature length to C.
   538	3) High-Order Pooling: By applying the GKP into (5), the
   539	proposed GHOP layer is defined as
   540	x=
   541	
   542	G
   543	1 XO
   544	X px .
   545	|S| p ∈S n
   546	
   547	(7)
   548	
   549	x
   550	
   551	Since multiple input features provide informative semantic
   552	characteristics of person poses, the high-order interactions
   553	among multiple input features are able to enhance the generalization ability of the GAReID model. For exploiting those
   554	high-order interactions, we extend the GHOP layer by reformulating (7) with multiple input features
   555	x=
   556	
   557	G
   558	G
   559	G
   560	O
   561	O
   562	1 X 1 O
   563	X px
   564	X 2px
   565	···
   566	X npx
   567	|S| p ∈S
   568	x
   569	
   570	(8)
   571	
   572	where G denotes the second-order GKP with n = 2. It is
   573	worth noting that this GHOP layer can be viewed as the
   574	high-order fusion method of multiple input features, which
   575	contributes to mining much richer information than the firstorder method such as channel concatenation.
   576	N
   577	
   578	C. Attentive High-Order Pooling
   579	1) Foreground Attention: Since aligned background similarities might introduce noise to the similarity aggregation
   580	of (2), the background knowledge should be excluded from
   581	person features. Recent studies [56] have found that the largest
   582	feature norms appear above target objects in a classification
   583	model pretrained on ImageNet. Our goal is to bootstrap on
   584	this phenomenon to highlight the foreground regions without
   585	explicitly introducing the learnable parameters. To this end,
   586	we design an attention module named AFA to produce a binary
   587	mask over spatial locations with using the l2 -norm of spatial
   588	features. Formally, given a feature map Z ∈ RC×H ×W , we first
   589	generate a feature map T ∈ R H ×W by operating the l2 norm
   590	for features as
   591	T p = Zp 2
   592	
   593	(9)
   594	
   595	where T p denotes the response score of T at the position p.
   596	To mine the foreground parts, we sample the positions where
   597	the response value is larger than an adaptive threshold. In this
   598	way, we produce a foreground position set as
   599	
   600	S F = p|T p > εT avg
   601	(10)
   602	where T avg denotes the average response of T and ε = 0.4 is
   603	a hyperparameter controlling the activation threshold. Subsequently, the attention mask M ∈ R H ×W is formed as
   604	M p = αI( p ∈ S F ) + βI( p ∈
   605	/ SF ) ∀ p ∈ S
   606	
   607	(11)
   608	
   609	where M p denotes the attention score of M at position p.
   610	The indicator function I(·) returns 1 if the input condition is
   611	true; otherwise, it returns 0. In our experiments, we set the
   612	foreground and background attention values as α = 1.0 and
   613	β = 0.3, respectively.
   614	2) Ensemble Attention: However, a single attention mask
   615	may not locate the foreground regions accurately because of
   616	diverse variations from person images. Inspired by ensemble
   617	learning, we adopt an elementwise average to fuse multiple
   618	
   619	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
   620	
   621	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
   622	
   623	3995
   624	
   625	Fig. 5. Visualization of feature maps extracted from the first-order (n = 1), second-order (n = 2), and third-order (n = 3) GHOP and AHOP layers.
   626	Following SIFTFlow [57], we use principal component analysis (PCA) to compress all the part descriptors into three-dimensional vectors and then rescale the
   627	vector values into the range of [0, 255] to represent the three color channels of RGB images. In the visualized feature maps, the same color implies that the
   628	n
   629	n
   630	part descriptors are similar, whereas different colors indicate that the part descriptors are dissimilar. Notably, {X i }i=1
   631	and {Zi }i=1
   632	represent the input feature
   633	maps of the nth-order GHOP and AHOP layers, respectively, while X and Z represent the output feature maps of the nth-order GHOP and AHOP layers,
   634	respectively. (a) GHOP. (b) AHOP.
   635	
   636	attention masks generated from input feature maps in (8) as
   637	M=
   638	
   639	n
   640	X
   641	
   642	1
   643	Mi
   644	n i=1
   645	
   646	(12)
   647	
   648	where M i denotes the attention mask of the ith input feature
   649	map Zi . With the combination of the GHOP and AFA layers,
   650	the proposed AHOP layer is defined as follows:
   651	z=
   652	
   653	G
   654	G
   655	O
   656	O
   657	
   658	1 X
   659	M pz Z 1pz
   660	···
   661	M pz Z npz .
   662	|S| p ∈S
   663	
   664	(13)
   665	
   666	z
   667	
   668	It is worth noting that this AHOP layer can be viewed
   669	as an attention fusion method, which aggregates different
   670	attention masks to refine the segmentation of the foreground
   671	and background regions.
   672	D. Overall Loss Function
   673	To train the GAReID framework, we use the triplet loss [58]
   674	to learn discriminative high-order features. We define x a ,
   675	x p , and x n as the anchor, positive, and negative high-order
   676	features from the GHOP layer, respectively, while z a , z p ,
   677	and z n represent the anchor, positive, and negative high-order
   678	features from the AHOP layer, respectively. The triplet loss
   679	aims at separating the positive pair from the negative one by
   680	a similarity margin m. The triplet loss is defined as
   681	i
   682	Xh
   683	ea ,e

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '672,815p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 388ms:
   672	D. Overall Loss Function
   673	To train the GAReID framework, we use the triplet loss [58]
   674	to learn discriminative high-order features. We define x a ,
   675	x p , and x n as the anchor, positive, and negative high-order
   676	features from the GHOP layer, respectively, while z a , z p ,
   677	and z n represent the anchor, positive, and negative high-order
   678	features from the AHOP layer, respectively. The triplet loss
   679	aims at separating the positive pair from the negative one by
   680	a similarity margin m. The triplet loss is defined as
   681	i
   682	Xh
   683	ea ,e
   684	ea ,e
   685	Lt =
   686	x
   687	xn − x
   688	xp + m
   689	+
   690	
   691	a, p,n
   692	
   693	h
   694	
   695	+ e
   696	z a ,e
   697	zn − e
   698	z a ,e
   699	zp + m
   700	
   701	i
   702	
   703	(14)
   704	+
   705	
   706	ea , x
   707	ep , and x
   708	en are the
   709	where m is set as m = 0.2. The vectors x
   710	l2 normalized features of x a , x p , and x n , respectively, while
   711	e
   712	za , e
   713	z p , and e
   714	z n are the l2 normalized features of z a , z p , and
   715	z n , respectively.
   716	IV. D ISCUSSION
   717	A. Feature Visualization
   718	In this part, considering the collaborative effect of highorder interactions among multiple feature maps in (8) and (13),
   719	
   720	we give a microscopic interpretation from the perspective
   721	of feature visualization, which shows a strong justification
   722	of our method. To some extent, it also reveals the reason
   723	why high-order interactions in the GHOP and AHOP layers
   724	contribute significantly to part-aligned and background robust
   725	representation learning.
   726	As exemplified in Fig. 5, one can observe that the first-order
   727	input feature maps mainly encode the semantics of various
   728	body parts, including heads, hands, shoulders, and legs, and
   729	their corresponding colors differ depending on their spatial
   730	positions. Furthermore, the part descriptors with the same positions from different input feature maps are shown in different
   731	colors due to the diversity of multiple input feature maps.
   732	In Fig. 5(a), the high-order output feature maps concentrate
   733	on encoding the discriminative body parts (e.g., heads, shoulders, and legs) to represent person identities, while the loworder output feature maps focus on capturing coarse-grained
   734	appearance information. Hence, high-order interactions from
   735	the GHOP layer are beneficial as they enhance pose invariance
   736	within the learned person features. In Fig. 5(b), the proposed
   737	AHOP layer is able to remove the background regions and
   738	retain the foreground areas of the input feature maps. This
   739	contributes to high-order background-invariant representation
   740	learning.
   741	B. Similarity Visualization
   742	Based on high-order similarity aggregation in (3), we provide another macroscopic explanation from high-order feature
   743	similarities. In a sense, it also furnishes a valuable angle for the
   744	understanding of the relationship between high-order feature
   745	similarities and fine-grained part alignments.
   746	As illustrated in Fig. 6, the maximum part similarity of
   747	the high-order features is clearly larger than the similarity
   748	of the low-order features, while the minimum part similarity
   749	remains largely unchanged for all the orders. In addition,
   750	the number of misaligned part pairs with prominent similarities consistently decreases along with the increase in
   751	feature order. Compared with the GHOP layer, the AHOP
   752	layer distinctly reduces the similarities of background part
   753	
   754	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
   755	
   756	3996
   757	
   758	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
   759	
   760	Fig. 6.
   761	Part similarity visualization of H 2 W 2 part pairs. Given a pair of images from the same person, we extract a pair of output feature maps
   762	from the first-order (n = 1), second-order (n = 2), and third-order (n = 3) GHOP/AHOP layers. Two feature maps are individually normalized by
   763	dividing the l2 norms of spatially pooled features. Finally, the similarity matrix is calculated by the inner product of all H 2 W 2 part pairs. Note that
   764	“max” and “min” denote the maximal and minimal part similarities, respectively. (a) GHOP (n = 1), (max/min) = ((1.9 × 10−3 )/(−6.0 × 10−5 )).
   765	(b) GHOP (n = 2), (max/min) = ((3.4 × 10−3 )/(−1.3 × 10−4 )). (c) GHOP (n = 3), (max/min) = ((1.3 × 10−2 )/(−3.4 × 10−4 )). (d) AHOP
   766	(n = 1), (max/min) = ((5.3 × 10−3 )/(−2.0 × 10−4 )). (e) AHOP (n = 2), (max/min) = ((8.7 × 10−3 )/(−1.8 × 10−4 )). (f) AHOP (n = 3),
   767	(max/min) = ((2.7 × 10−2 )/(−4.8 × 10−4 )).
   768	
   769	pairs, which reinforces the background robust representation
   770	learning. Moreover, the increase amplitude of the maximum
   771	part similarity in the AHOP layer is evidently larger than the
   772	similarity of the GHOP layer with the same increase in feature
   773	orders. This observation indicates that the background removal
   774	alleviates the part misalignment problem.
   775	C. Landmark Visualization
   776	As suggested in prior works [8], [9], [10], the semantic
   777	knowledge of person landmarks is likely to remain unchanged,
   778	even when drastic pose variations have taken place. Besides,
   779	person pose variations mainly reflect the landmark distribution
   780	of person images. Therefore, to analyze the effectiveness of
   781	part alignments, it is worth exploring the high-order semantic
   782	interactions between different landmark pairs.
   783	Given a pair of images I u and I v of the same person,
   784	we extract a pair of output feature maps U and V from the
   785	first-order, second-order, and third-order GHOP/AHOP layers.
   786	Then, we adopt an existing OpenPose [34] to detect 16 body
   787	landmarks for the two person images. To extract landmark
   788	descriptors, we upsample the output feature maps with the
   789	cubic interpolation to have the same size as the input images
   790	and then the landmark features are acquired from the resized
   791	feature maps according to landmark positions. Finally, the
   792	cosine similarities of 16 × 16 landmark pairs from the two
   793	images are computed to form a similarity confusion matrix.
   794	
   795	The results shown in Fig. 7 well demonstrate that high-order
   796	part features can successfully learn landmark correspondences
   797	between the two images without using landmark annotations.
   798	Specifically, compared with low-order features, high-order
   799	features are able to significantly enlarge the similarity discrepancies between aligned and misaligned landmark pairs.
   800	More interestingly, the comparison between the GHOP and
   801	AHOP layer certifies that foreground region mining is conducive to highlighting the semantic correspondences of person
   802	landmarks.
   803	D. Attention Visualization
   804	In this part, we provide an intuitive interpretation by visualizing foreground attention masks to study the impact of
   805	the proposed AFA module. The visualized results demonstrate
   806	the superiority of the proposed ensemble attention strategy
   807	for the person ReID task. To some degree, the interpretation
   808	also clarifies the reason why learning foreground-based features is more helpful to part alignments than learning imagebased features.
   809	Given an input person image I x , we extract individual
   810	n
   811	attention masks {M i }i=1
   812	and ensemble attention masks M
   813	from the first-order (n = 1), second-order (n = 2), and thirdorder (n = 3) AHOP layers. To better visualize the spatial
   814	relationships between confidence maps and body parts, the
   815	low-resolution attention mask is upsampled using the cubic

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '803,850p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 392ms:
   803	D. Attention Visualization
   804	In this part, we provide an intuitive interpretation by visualizing foreground attention masks to study the impact of
   805	the proposed AFA module. The visualized results demonstrate
   806	the superiority of the proposed ensemble attention strategy
   807	for the person ReID task. To some degree, the interpretation
   808	also clarifies the reason why learning foreground-based features is more helpful to part alignments than learning imagebased features.
   809	Given an input person image I x , we extract individual
   810	n
   811	attention masks {M i }i=1
   812	and ensemble attention masks M
   813	from the first-order (n = 1), second-order (n = 2), and thirdorder (n = 3) AHOP layers. To better visualize the spatial
   814	relationships between confidence maps and body parts, the
   815	low-resolution attention mask is upsampled using the cubic
   816	
   817	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
   818	
   819	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
   820	
   821	3997
   822	
   823	Fig. 7. Landmark similarity visualization of a pair of images from the same person. We use OpenPose [34] to detect 16 body landmarks, and then 16 landmark
   824	features are extracted from the first-order (n = 1), second-order (n = 2), and third-order (n = 3) GHOP/AHOP layers. Then, the cosine similarities of
   825	16 × 16 landmark pairs from the two images are computed to form a similarity confusion matrix. (a) GHOP (n = 1). (b) GHOP (n = 2). (c) GHOP (n = 3).
   826	(d) AHOP (n = 1). (e) AHOP (n = 2). (f) AHOP (n = 3).
   827	
   828	Fig. 8. Attention mask visualization of the three attention methods. In each visualized attention, the red areas refer to the foreground regions, while the
   829	n
   830	purple ones are associated with the background clutters. We extract individual attention masks {M i }i=1
   831	and ensemble attention masks M from the first-order
   832	(n = 1), second-order (n = 2), and third-order (n = 3) AHOP layers. “l2 Norm,” “Avg,” and “Max” represent that the l2 norms, average values, and maximal
   833	values along the channel dimension are used to generate attention masks, respectively.
   834	
   835	interpolation to have the same size as I x . Then we merge both
   836	the attention masks and person images by alpha blending. For
   837	interpreting the effectiveness of the proposed AFA method,
   838	we analyze the three attention generation methods, including
   839	“l2 Norm,” “Avg,” and “Max.” In particular, “l2 Norm,” “Avg,”
   840	and “Max” represent that the l2 norms, average values, and
   841	maximal values along the channel dimension are used to
   842	generate attention masks, respectively. As seen in Fig. 8, “Avg”
   843	performs the worst among the three attention generation methods because it is unable to finely discriminate the foreground
   844	regions from the background ones. In other words, “Avg”
   845	
   846	mixes up the foreground and background knowledge, which
   847	may hinder the background robust representation learning.
   848	On the whole, both “l2 Norm” and “Max” can successfully capture the foreground regions and eliminate the background areas without using person segmentation annotations.
   849	Compared with “Max,” “l2 Norm” performs foreground detection with a more fine-grained manner. For example, when the
   850	order factor n = 2 or n = 3, “Max” is unable to detect the

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|GHOP|AHOP|AFA|high-order|order|Table|Market|MSMT|Duke|components|Visualization|epsilon|group|shuffle\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3-     3	3997
4-     4	
5-     5	Fig. 7. Landmark similarity visualization of a pair of images from the same person. We use OpenPose [34] to detect 16 body landmarks, and then 16 landmark
6:     6	features are extracted from the first-order (n = 1), second-order (n = 2), and third-order (n = 3) GHOP/AHOP layers. Then, the cosine similarities of
7:     7	16 × 16 landmark pairs from the two images are computed to form a similarity confusion matrix. (a) GHOP (n = 1). (b) GHOP (n = 2). (c) GHOP (n = 3).
8:     8	(d) AHOP (n = 1). (e) AHOP (n = 2). (f) AHOP (n = 3).
9-     9	
10-    10	Fig. 8. Attention mask visualization of the three attention methods. In each visualized attention, the red areas refer to the foreground regions, while the
11-    11	n
12-    12	purple ones are associated with the background clutters. We extract individual attention masks {M i }i=1
13:    13	and ensemble attention masks M from the first-order
14:    14	(n = 1), second-order (n = 2), and third-order (n = 3) AHOP layers. “l2 Norm,” “Avg,” and “Max” represent that the l2 norms, average values, and maximal
15-    15	values along the channel dimension are used to generate attention masks, respectively.
16-    16	
17-    17	interpolation to have the same size as I x . Then we merge both
18-    18	the attention masks and person images by alpha blending. For
19:    19	interpreting the effectiveness of the proposed AFA method,
20-    20	we analyze the three attention generation methods, including
21-    21	“l2 Norm,” “Avg,” and “Max.” In particular, “l2 Norm,” “Avg,”
22-    22	and “Max” represent that the l2 norms, average values, and
--
29-    29	may hinder the background robust representation learning.
30-    30	On the whole, both “l2 Norm” and “Max” can successfully capture the foreground regions and eliminate the background areas without using person segmentation annotations.
31-    31	Compared with “Max,” “l2 Norm” performs foreground detection with a more fine-grained manner. For example, when the
32:    32	order factor n = 2 or n = 3, “Max” is unable to detect the
33-    33	foreground regions of person legs, while “l2 Norm” is capable
34-    34	of avoiding a few residual background problems. Furthermore,
35:    35	as the order n increases, the foreground attention quality of
36-    36	
37-    37	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
38-    38	
--
48-    48	preserve discriminative foreground regions and remove hard
49-    49	background areas.
50-    50	E. Similarity Attention
51:    51	To analyze the impact of the proposed AHOP layer,
52:    52	we reformulate high-order similarity between the two images
53-    53	I u and I v as
54-    54	Sim(I u , I v ; n) = u, v
55-    55	X
--
76-    76	
77-    77	where M u and M v represent the foreground attention maps
78-    78	of the two images I u and I v , respectively. In addition,
79:    79	(M upu M vpv )n can be viewed as the nth-order similarity attention between pu and pv . To illustrate the effectiveness of
80-    80	foreground attention, we consider four part pair cases, i.e.,
81-    81	foreground–foreground (FF), foreground–background (FB),
82-    82	background–foreground (BF), and background–background
83-    83	(BB). If the body part pair belongs to the FF case, then
84:    84	M upu M vpv = 1 always holds and its high-order similarity
85-    85	attention keeps unchanged as follows:
86-    86	n
87-    87	lim M upu M vpv = 1.
--
89-    89	n→∞
90-    90	
91-    91	If the part pair belongs to FB, BF, or BB, then M upu M vpv < 1
92:    92	always holds and its high-order similarity attention dramatically decreases as follows:
93-    93	n
94-    94	lim M upu M vpv = 0.
95-    95	(17)
96-    96	n→∞
97-    97	
98:    98	To sum up, the high-order similarity attention contributes to
99-    99	reducing the part similarity of FB, BF, and BB pairs, while
100:   100	maintaining the similarity of FF pairs. As the order factor
101-   101	n → ∞, person similarity is equivalent to an average of the
102-   102	similarities of aligned foreground part pairs, resulting in both
103-   103	part-aligned and background robust person ReID.
104-   104	F. Gradient Optimization
105:   105	Finally, to assess the collaborative impact of high-order
106-   106	features on metric learning, we provide another theoretical
107-   107	analysis based on the gradient optimization for the triplet loss.
108:   108	To simplify the following analysis, we ignore the l2 normalization for high-order features. If we suppose that the
109:   109	high-order features are directly aggregated by the Kronecker
110-   110	product, the triplet loss is formulated as
111-   111	h
112-   112	i
--
134-   134	
135-   135	p
136-   136	
137:   137	where n o is the order coefficient of attentive high-order features. In addition, Z apa denotes the part feature vector of the
138-   138	anchor feature map Z a at the position pa , while M apa refers
139-   139	to the attention value of the anchor attention mask M a at
140-   140	the position pa . In the same way, a similar definition is also
--
257-   257	ap
258-   258	When n o ≥ 2, the two weight coefficients W an
259-   259	pa pn and W pa p p
260:   260	can be viewed as attentive (n o −1)th-order feature similarities.
261:   261	As the order n o increases, the gradients of aligned part
262-   262	descriptors are highlighted over those of the misaligned parts.
263-   263	In this case, the gradient term ∂L/∂ Z apa pushes the anchor
264-   264	descriptor Z apa close to the aligned positive part descriptors
--
285-   285	p
286-   286	as W an
287-   287	pa pn = M pa M pn and W pa p p = M pa M p p , respectively.
288:   288	Therefore, the weight coefficient of the first-order AHOP layer
289-   289	is equivalent to the product of two attention values. Although
290-   290	the attention product encodes the relationships between a pair
291-   291	of spatial positions, there is no guarantee that the attention
--
294-   294	large attention values, the aligned (leg ↔ leg and hand ↔
295-   295	hand) and misaligned (leg ↔ hand) part pairs might have
296-   296	similar values of attention product. Thus, the gradient terms of
297:   297	the first-order model contain both the aligned and misaligned
298-   298	part descriptors, which might generate even totally erroneous
299-   299	gradient directions for backpropagation optimization.
300-   300	According to the above analysis, the weight coefficient can
301-   301	be treated as a regularization term to regulate the gradient
302:   302	direction, which explains well the reason why high-order
303-   303	features enhance the generalization ability of the ReID model.
304-   304	In summary, by considering the collaborative effort of all
305-   305	the gradient terms, we could understand better the working
--
309-   309	properties within features.
310-   310	V. E XPERIMENTS
311-   311	A. Dataset
312:   312	1) Market1501 [26]: It contains 32 668 images of 1501 persons captured by six camera views. The whole dataset is
313-   313	divided into a training set containing 12 936 images of 751 persons and a testing set containing 19 732 images of 750 persons.
314-   314	For each person in the testing set, we select one image from
315-   315	each camera as a query image, forming 3368 queries following
--
319-   319	(Detected). We use the settings of both labeled and detected
320-   320	person images on the splits in [71], where 767 and 700 persons
321-   321	are used for training and testing, respectively.
322:   322	3) DukeMTMC [28]: It contains 36 411 images of 1812 persons captured by eight cameras, where only 1404 persons
323-   323	appeared in more than two cameras. The other 408 persons are
324-   324	regarded as distractors. The training set contains 16 522 images
325-   325	of 702 persons, while the testing set contains 2228 query
326-   326	images of 702 persons and 17 661 gallery images.
327:   327	4) MSMT17 [29]: It contains manually annotated 126 441
328-   328	bounding boxes of 4101 persons, which is currently the
329-   329	largest person ReID dataset. All the images are captured by
330-   330	the 15-camera network deployed in campus, which contains
--
376-   376	E5-2620 v4 at 2.10 GHz CPUs, four GeForce GTX 1080 Ti
377-   377	GPUs, and 128-GB RAM.
378-   378	C. Comparison With State-of-the-Art Methods
379:   379	In Table I, we compare the proposed GAReID with the
380-   380	current state-of-the-art methods on the four person ReID
381-   381	datasets. From the results, we can see that our method
382-   382	achieves the best ReID performance on each dataset. Specifically, the proposed GAReID based on ResNet50 outperforms
383-   383	the previous best performed SAN [63] by 4.16% in mAP
384-   384	on CUHK03-D. Although our method performs closely to
385:   385	ISP [68] on the Market1501 and DukeMTMC datasets, our
386-   386	method can achieve a slightly higher accuracy in a very simple
387-   387	yet effective way. This is because GAReID performs superior
388-   388	
--
393-   393	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
394-   394	
395-   395	TABLE I
396:   396	C OMPARISON W ITH THE S TATE - OF - THE -A RT M ETHODS ON THE M ARKET 1501 [26], CUHK03 [27], D UKE MTMC [28], AND MSMT17 [29] DATASETS .
397-   397	CUHK03-L AND CUHK03-D U SE L ABELED AND D ETECTED B OUNDING B OXES TO C ROP P ERSON I MAGES ON CUHK03, R ESPECTIVELY. T WO
398-   398	BASELINE M ODELS BASED ON R ES N ET 50/101 [59] A RE T RAINED W ITH T RIPLET L OSS , AND G LOBAL F EATURES
399-   399	A RE E XTRACTED F ROM THE GAP L AYER TO P ERFORM R E ID E VALUATION
400-   400	
401:   401	Fig. 9. Ablation studies on the Market1501 [26] and DukeMTMC [28]
402:   402	datasets. (a) Analyzing the impact of the order n. (b) Comparing different
403:   403	order fusion strategies, “(1, 2)” means that the first-order and second-order
404-   404	features are fused by channel concatenation.
405-   405	
406-   406	part alignment with only identity labels, while other methods
407-   407	require landmark annotations or body partition during the
408-   408	training and testing phases. Compared with other datasets, the
409:   409	MSMT17 dataset presents the following challenges: 1) large
410-   410	number of person identities, bounding boxes and cameras;
411-   411	2) complex scenes and backgrounds; and 3) multiple time
412-   412	slots with severe lighting changes. Although all the compared
413:   413	methods achieve lower accuracies on MSMT17 than other
414-   414	datasets, the proposed GAReID is the best performing method,
415-   415	outperforming the second best method by 1.23% for mAP.
416-   416	This clearly demonstrates that GAReID achieves a satisfactory
417-   417	generalization on the large-scale dataset.
418:   418	D. Ablation Study
419:   419	1) Feature Order: We first study the impact of the order of
420:   420	high-order features. As seen in Fig. 9(a), we can observe two
421-   421	
422:   422	interesting phenomena. First, a higher feature order benefits
423:   423	person ReID performance. The mAP scores of Market1501
424:   424	and DukeMTMC datasets increase consistently until they reach
425:   425	a stable performance. For example, the third-order feature
426:   426	(n = 3) outperforms the first-order feature (n = 1) by
427:   427	3.57% and 7.99% in terms of mAP on the Market1501
428:   428	and DukeMTMC datasets, respectively. Second, increasing
429:   429	the order (n > 3) makes a limited contribution to mAP
430-   430	improvement compared with n = 3. To some extent, this is
431:   431	because the third-order pooling layer has largely eliminated
432-   432	part misalignments. Therefore, there is little room for further
433-   433	part alignment improvements. To sum up, we recommend
434-   434	n = 3 for GAReID as it strikes a satisfactory balance between
435-   435	the computational efficiency and ReID performance.
436:   436	2) Order Fusion: We explore the effectiveness of order
437:   437	fusion by averaging features from different orders. Two interesting observations can be made in Fig. 9(b). First, compared
438:   438	with low-order features [n = (1, 2)], fusing high-order features
439-   439	[n = (2, 3)] always benefits person ReID performance. The
440:   440	main reason is that high-order features help reduce the person
441:   441	part misalignment problem. Second, compared with singleorder features (n = 3), mixed-order features [n = (1, 2, 3)]
442-   442	may significantly degrade ReID accuracies. To some extent,
443:   443	this is because fusing too many low-order features is unable
444-   444	to highlight the discriminative information.
445-   445	3) Attention Generation: We compare the performance of
446:   446	different attention generation methods on the Market1501 and
447:   447	DukeMTMC datasets. The results in Table II show that the
448-   448	“l2 Norm” consistently achieves superior mAP scores than
449-   449	other attention methods. This suggests that “l2 Norm” is
450-   450	more suitable to mine foreground regions than other methods.
--
464-   464	ATTENTION M ASKS , R ESPECTIVELY. N OTE T HAT A LL
465-   465	THE M ODELS U SE R ES N ET 50 AS THE BACKBONE
466-   466	
467:   467	Fig. 10. Ablation studies on the Market1501, CUHK03, and DukeMTMC
468-   468	datasets. (a) Analyzing different network architectures. (b) Analyzing different
469-   469	pooling layers.
470-   470	
471-   471	TABLE III
472-   472	A BLATION S TUDIES OF D IFFERENT M ODULES ON THE M ARKET 1501,
473:   473	CUHK03, D UKE MTMC, AND MSMT17 DATASETS . “HOP,” “MF,”
474-   474	“GS,” AND “EA” R EPRESENT H IGH -O RDER P OOLING , M ULTIPLE
475-   475	F EATURE I NPUT, G ROUP S HUFFLE , AND E NSEMBLE
476-   476	ATTENTION , R ESPECTIVELY. N OTE T HAT A LL
--
486-   486	are aggregated by the Kronecker product, while the single
487-   487	feature input denotes that multiple duplicates of the single
488-   488	feature are aggregated by the Kronecker product. From the
489:   489	results in Table III, it can be observed that multiple feature
490-   490	fusion performs better than the single feature input on the three
491-   491	datasets. The major reason is that multiple features are able to
492-   492	bring richer pose knowledge than the single feature, resulting
493:   493	in a very strong high-order representational capability for the
494-   494	ReID models.
495:   495	5) Group Shuffle: Since the channel group strategy is crucial to high-order feature compression, we need to explore
496:   496	the impact of the group shuffle strategy on enhancing the
497:   497	generalization capability of the ReID models. From Table III,
498:   498	we can observe that the group shuffle strategy consistently
499-   499	improves ReID performance with a significant margin on
500:   500	the three datasets. This is because the group shuffle strategy
501:   501	encodes the intergroup interactions, which are beneficial to
502:   502	enrich the information of compressed high-order features.
503-   503	6) Ensemble Attention: In this part, we investigate the
504-   504	impact of ensemble attention on background robust feature
505-   505	
506-   506	learning. We also design independent attention masks for
507-   507	different input features to eliminate the background regions.
508:   508	From the results reported in Table III, we note that ensemble
509-   509	attention achieves significant ReID performance improvements
510-   510	over independent attention. This observation indicates that
511-   511	ensemble attention, by integrating multiple attention masks,
512-   512	can reduce the influence of background clutters more effectively and generate better background robust features.
513-   513	7) Network Architecture: As shown in Fig. 2, we use
514-   514	different BN layers for different input branches. The impact of
515:   515	the BN layer on high-order feature learning can be observed
516-   516	in Fig. 10(a). As seen, using the BN layer achieves significant improvements on the three datasets, which justifies that
517:   517	normalized features contribute to learning discriminative highorder representations. As mentioned in Section III-B, we need
518-   518	to add a ReLU layer after input features to ensure that part
519-   519	similarity is always nonnegative. Therefore, it is worthwhile
520-   520	to examine whether GAReID can perform satisfactory part
--
524-   524	ReLU causes “dead” neurons when their activation values are
525-   525	negative. In other words, ReLU restricts the distribution of feature maps to a nonnegative space and ignores the information
526-   526	of negative neurons. This might impair the representational
527:   527	capability of high-order features. In line with the findings of
528-   528	the above analysis, we use the BN layer instead of the ReLU
529-   529	layer as default.
530:   530	8) Pooling Layer: In this part, we investigate the contributions of the GHOP and AHOP layers on part-aligned
531-   531	representation learning. In Fig. 10(b), the results show that the
532:   532	AHOP layer consistently achieves superior mAP scores than
533:   533	the GHOP layer. This phenomenon indicates that foregroundbased features are more suitable than image-based features
534-   534	in part alignment tasks. To show the effectiveness of joint
535-   535	learning of the two features, we concatenate them along the
536-   536	feature dimension to obtain complete person representations.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 11 -l 12 'GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf' - | nl -ba | sed -n '1,220p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	4000
     2	
     3	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
     4	
     5	TABLE I
     6	C OMPARISON W ITH THE S TATE - OF - THE -A RT M ETHODS ON THE M ARKET 1501 [26], CUHK03 [27], D UKE MTMC [28], AND MSMT17 [29] DATASETS .
     7	CUHK03-L AND CUHK03-D U SE L ABELED AND D ETECTED B OUNDING B OXES TO C ROP P ERSON I MAGES ON CUHK03, R ESPECTIVELY. T WO
     8	BASELINE M ODELS BASED ON R ES N ET 50/101 [59] A RE T RAINED W ITH T RIPLET L OSS , AND G LOBAL F EATURES
     9	A RE E XTRACTED F ROM THE GAP L AYER TO P ERFORM R E ID E VALUATION
    10	
    11	Fig. 9. Ablation studies on the Market1501 [26] and DukeMTMC [28]
    12	datasets. (a) Analyzing the impact of the order n. (b) Comparing different
    13	order fusion strategies, “(1, 2)” means that the first-order and second-order
    14	features are fused by channel concatenation.
    15	
    16	part alignment with only identity labels, while other methods
    17	require landmark annotations or body partition during the
    18	training and testing phases. Compared with other datasets, the
    19	MSMT17 dataset presents the following challenges: 1) large
    20	number of person identities, bounding boxes and cameras;
    21	2) complex scenes and backgrounds; and 3) multiple time
    22	slots with severe lighting changes. Although all the compared
    23	methods achieve lower accuracies on MSMT17 than other
    24	datasets, the proposed GAReID is the best performing method,
    25	outperforming the second best method by 1.23% for mAP.
    26	This clearly demonstrates that GAReID achieves a satisfactory
    27	generalization on the large-scale dataset.
    28	D. Ablation Study
    29	1) Feature Order: We first study the impact of the order of
    30	high-order features. As seen in Fig. 9(a), we can observe two
    31	
    32	interesting phenomena. First, a higher feature order benefits
    33	person ReID performance. The mAP scores of Market1501
    34	and DukeMTMC datasets increase consistently until they reach
    35	a stable performance. For example, the third-order feature
    36	(n = 3) outperforms the first-order feature (n = 1) by
    37	3.57% and 7.99% in terms of mAP on the Market1501
    38	and DukeMTMC datasets, respectively. Second, increasing
    39	the order (n > 3) makes a limited contribution to mAP
    40	improvement compared with n = 3. To some extent, this is
    41	because the third-order pooling layer has largely eliminated
    42	part misalignments. Therefore, there is little room for further
    43	part alignment improvements. To sum up, we recommend
    44	n = 3 for GAReID as it strikes a satisfactory balance between
    45	the computational efficiency and ReID performance.
    46	2) Order Fusion: We explore the effectiveness of order
    47	fusion by averaging features from different orders. Two interesting observations can be made in Fig. 9(b). First, compared
    48	with low-order features [n = (1, 2)], fusing high-order features
    49	[n = (2, 3)] always benefits person ReID performance. The
    50	main reason is that high-order features help reduce the person
    51	part misalignment problem. Second, compared with singleorder features (n = 3), mixed-order features [n = (1, 2, 3)]
    52	may significantly degrade ReID accuracies. To some extent,
    53	this is because fusing too many low-order features is unable
    54	to highlight the discriminative information.
    55	3) Attention Generation: We compare the performance of
    56	different attention generation methods on the Market1501 and
    57	DukeMTMC datasets. The results in Table II show that the
    58	“l2 Norm” consistently achieves superior mAP scores than
    59	other attention methods. This suggests that “l2 Norm” is
    60	more suitable to mine foreground regions than other methods.
    61	Moreover, we observe that “Avg” achieves the worst mAP
    62	
    63	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
    64	
    65	WANG et al.: GAReID: GROUPED AND ATTENTIVE HIGH-ORDER REPRESENTATION LEARNING
    66	
    67	4001
    68	
    69	TABLE II
    70	C OMPARATIVE E XPERIMENTS U SING D IFFERENT ATTENTION
    71	M ECHANISM M ETHODS . “l2 N ORM ,” “AVG ,” AND “M AX ” R EPRESENT
    72	T HAT THE l2 N ORMS , AVERAGE VALUES , AND M AXIMAL VALUES
    73	A LONG THE C HANNEL D IMENSION A RE U SED TO G ENERATE
    74	ATTENTION M ASKS , R ESPECTIVELY. N OTE T HAT A LL
    75	THE M ODELS U SE R ES N ET 50 AS THE BACKBONE
    76	
    77	Fig. 10. Ablation studies on the Market1501, CUHK03, and DukeMTMC
    78	datasets. (a) Analyzing different network architectures. (b) Analyzing different
    79	pooling layers.
    80	
    81	TABLE III
    82	A BLATION S TUDIES OF D IFFERENT M ODULES ON THE M ARKET 1501,
    83	CUHK03, D UKE MTMC, AND MSMT17 DATASETS . “HOP,” “MF,”
    84	“GS,” AND “EA” R EPRESENT H IGH -O RDER P OOLING , M ULTIPLE
    85	F EATURE I NPUT, G ROUP S HUFFLE , AND E NSEMBLE
    86	ATTENTION , R ESPECTIVELY. N OTE T HAT A LL
    87	THE M ODELS U SE R ES N ET 50
    88	AS THE BACKBONE
    89	
    90	score than the model without using attention. As illustrated in
    91	Fig. 8, the main reason is that “Avg” can be viewed as a lowpass filter which removes some discriminative information.
    92	In view of performance and efficiency, we adopt the “l2 Norm”
    93	to generate foreground attention masks in this work.
    94	4) Multiple Feature Fusion: In this part, we examine the
    95	effectiveness of multiple feature fusion in (8) and (13). Specifically, multiple feature fusion represents that multiple features
    96	are aggregated by the Kronecker product, while the single
    97	feature input denotes that multiple duplicates of the single
    98	feature are aggregated by the Kronecker product. From the
    99	results in Table III, it can be observed that multiple feature
   100	fusion performs better than the single feature input on the three
   101	datasets. The major reason is that multiple features are able to
   102	bring richer pose knowledge than the single feature, resulting
   103	in a very strong high-order representational capability for the
   104	ReID models.
   105	5) Group Shuffle: Since the channel group strategy is crucial to high-order feature compression, we need to explore
   106	the impact of the group shuffle strategy on enhancing the
   107	generalization capability of the ReID models. From Table III,
   108	we can observe that the group shuffle strategy consistently
   109	improves ReID performance with a significant margin on
   110	the three datasets. This is because the group shuffle strategy
   111	encodes the intergroup interactions, which are beneficial to
   112	enrich the information of compressed high-order features.
   113	6) Ensemble Attention: In this part, we investigate the
   114	impact of ensemble attention on background robust feature
   115	
   116	learning. We also design independent attention masks for
   117	different input features to eliminate the background regions.
   118	From the results reported in Table III, we note that ensemble
   119	attention achieves significant ReID performance improvements
   120	over independent attention. This observation indicates that
   121	ensemble attention, by integrating multiple attention masks,
   122	can reduce the influence of background clutters more effectively and generate better background robust features.
   123	7) Network Architecture: As shown in Fig. 2, we use
   124	different BN layers for different input branches. The impact of
   125	the BN layer on high-order feature learning can be observed
   126	in Fig. 10(a). As seen, using the BN layer achieves significant improvements on the three datasets, which justifies that
   127	normalized features contribute to learning discriminative highorder representations. As mentioned in Section III-B, we need
   128	to add a ReLU layer after input features to ensure that part
   129	similarity is always nonnegative. Therefore, it is worthwhile
   130	to examine whether GAReID can perform satisfactory part
   131	alignments without ReLU. From the results in Fig. 10(a),
   132	we observe that GAReID without ReLU achieves superior
   133	performance than the architecture with ReLU. This is because
   134	ReLU causes “dead” neurons when their activation values are
   135	negative. In other words, ReLU restricts the distribution of feature maps to a nonnegative space and ignores the information
   136	of negative neurons. This might impair the representational
   137	capability of high-order features. In line with the findings of
   138	the above analysis, we use the BN layer instead of the ReLU
   139	layer as default.
   140	8) Pooling Layer: In this part, we investigate the contributions of the GHOP and AHOP layers on part-aligned
   141	representation learning. In Fig. 10(b), the results show that the
   142	AHOP layer consistently achieves superior mAP scores than
   143	the GHOP layer. This phenomenon indicates that foregroundbased features are more suitable than image-based features
   144	in part alignment tasks. To show the effectiveness of joint
   145	learning of the two features, we concatenate them along the
   146	feature dimension to obtain complete person representations.
   147	We observe that the leveraging of these two features significantly outperforms either of them on the three datasets.
   148	In other words, the foreground-based features have some
   149	advantages over the image-based ones, but they are complementary to each other. Therefore, our proposed system
   150	
   151	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:36 UTC from IEEE Xplore. Restrictions apply.
   152	
   153	
codex
第四篇的可学点在于：作者没有说“我要做更强的 part 模块”，而是用一个数学解释把普通全局池化的问题变成“错位 part pair 数量压过对齐 pair”，于是高阶相似度自然成为机制。这个“问题重定义到公式里”的力度比单纯模块拼接强。
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
/bin/zsh -lc "pdfinfo 'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' | sed -n '1,35p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction
Subject:         Pattern Recognition, 165 (2025) 111591. doi:10.1016/j.patcog.2025.111591
Keywords:        Pedestrian re-identification,Correlation graph sampling,Sparsely focused,Correlation reconstruction
Author:          Feng Min
Creator:         Elsevier
Producer:        Acrobat Distiller 8.1.0 (Windows)
CreationDate:    Wed Apr 23 12:01:06 2025 CST
ModDate:         Wed Apr 23 12:30:59 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           12
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       3012089 bytes
Optimized:       no
PDF version:     1.7

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' - | nl -ba | sed -n '42,154p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    42	1. Introduction
    43	Pedestrian re-identification (Re-ID), has received significant attention due to its practical value in intelligent security [1], video surveillance [2], and urban management [3]. Pedestrian re-identification
    44	models have achieved promising performance on certain datasets, but
    45	applying them to unknown scenarios with large amounts of data remains highly challenging [4]. Existing deep learning-based methods
    46	mainly focus on single-image feature representation learning, lacking
    47	adaptability to unseen scenes [5]. While these models perform well
    48	within a single dataset, their cross-dataset testing results are often
    49	unsatisfactory, highlighting a gap in practical application.
    50	To address these challenges, research in generalized pedestrian
    51	re-identification has gained momentum. Cross-dataset testing and generalization have become important research directions [6], with efforts made in direct cross-dataset evaluation for benchmarking performance [7]. However, the field still faces significant challenges in adapting well-trained models to unknown scenarios, necessitating further
    52	research in cross-library testing and generalization.
    53	
    54	One area of current research focuses in enhancing the generalization
    55	capability of pedestrian re-identification algorithms is metric learning.
    56	It aims to design training objectives with different sampling strategies
    57	and loss functions. Batch samplers play a crucial role in deep metric
    58	learning [8], yet there is limited research in this area. The PK sampler
    59	(Fig. 1(a)) is a widely used random sampling method in pedestrian reidentification [9]. However, this sampler exhibits global randomness,
    60	resulting in uniformly distributed sampled examples across the entire
    61	dataset in small batches.
    62	The PK sampler randomly selects p classes and then samples k
    63	images for each class to construct a small batch of size n = p × k.
    64	However, the global randomness of this method makes it challenging
    65	to provide relevant information for efficient deep metric learning.
    66	Additionally, the small batch size obtained from the PK sampler does
    67	not consider the relationship between samples. When using incomplete
    68	random sampling, it becomes necessary to consider the relationships
    69	between classes. If incomplete random sampling is used, it is necessary
    70	to consider the relationships between classes.
    71	
    72	∗ Corresponding author.
    73	
    74	E-mail addresses: fmin@wit.edu.cn (F. Min), liuyuhui@wit.edu.cn (Y. Liu), mlaycn@163.com (Y. Mao).
    75	https://doi.org/10.1016/j.patcog.2025.111591
    76	Received 20 February 2024; Received in revised form 26 December 2024; Accepted 9 March 2025
    77	Available online 18 March 2025
    78	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    79	
    80	Pattern Recognition 165 (2025) 111591
    81	
    82	F. Min et al.
    83	
    84	Fig. 1. Two different sampling methods are shown in (a) PK sampler and (b) the proposed CGS sampler. Different shapes indicate different classes, while different colors represent
    85	different batches. CGS sampler always samples the nearest neighboring classes. (For interpretation of the references to color in this figure legend, the reader is referred to the
    86	web version of this article.)
    87	
    88	To address the issue of lacking correlation in sampled samples,
    89	some studies have employed dataset-level representations based on
    90	average class embeddings and clustering techniques [10]. However,
    91	these methods may have suboptimal performance when dealing with
    92	a large number of classes. Another approach by [11] introduced a
    93	graph sampler based on metric learning, which partially addressed the
    94	issue of global randomness in sampling. However, this sampler exhibits unstable performance across multiple training iterations and has
    95	high computational complexity, limiting its application to large-scale
    96	datasets.
    97	To overcome these challenges, we propose a method called Correlation Graph Sampler (CGS), as shown in Fig. 1(b). It leverages hash
    98	encoding to group samples based on their correlation, providing a
    99	coarse classification. To obtain the most relevant training instances,
   100	we further employ feature-adaptive matching to compute correlations
   101	among samples with the same hash encoding. This allows us to identify
   102	the top P classes that are most similar to each base class. Finally,
   103	we construct mutually independent nearest neighbor graphs for each
   104	class. By adopting this approach, our method efficiently selects the most
   105	relevant training instances while reducing computational complexity.
   106	Note that, We conducted separate evaluations of our CGS and the
   107	latest Graph Sampler [11] on the Tesla V100. The results revealed
   108	that the latest GS sampler required 4 s and 40 s for sampling calculations on the Market dataset and the MSMT(all) dataset, respectively.
   109	In contrast, CGS required only 0.1 s and 1 s for sampling assessments on the Market dataset and the MSMT(all) dataset, respectively.
   110	Encouragingly, when faced with datasets containing a greater number of identities, CGS demonstrated stable and outstanding sampling
   111	performance, significantly reducing computational complexity.
   112	Moreover, the feature-adaptive matching method in CGS sampling
   113	strategy is correlated with the feature maps extracted by the backbone
   114	network, which endows CGS with learnability. Specifically, the sampling performance of CGS improves as the feature maps extracted by
   115	the backbone network improve during training iterations. Inspired by
   116	the sampling principle of CGS, we identified the potential of improving
   117	the performance of CGS by enhancing feature representation learning. Based on this, we propose a novel high-resolution flow network,
   118	named Global Sparse Attention Network (GSANet), to reduce the loss of
   119	spatial positional information in the process of feature representation
   120	learning. We also design a new global relevance sparse reconstruction
   121	module (GRSR) to reconstruct the pixel-level features’ auto-correlation
   122	of the feature layer, which enhances the backbone network’s feature
   123	representation learning capability.
   124	Therefore, we propose a bi-stream interactive learning framework.
   125	One of the learning streams is the correlation graph sampler (CGS)
   126	for metric learning, and the other learning stream is the global sparse
   127	attention network (GSANet) for representation learning. We establish
   128	an intrinsic connection between these two learning streams. On one
   129	hand, CGS provides challenging training instances to enhance the
   130	representation learning capability of the backbone network. On the
   131	other hand, the improvement in feature map quality extracted by
   132	the backbone network facilitates the enhancement of CGS sampling
   133	performance, thereby achieving the desired interactive learning effect.
   134	Our approach establishes a mutually reinforcing relationship between
   135	
   136	metric learning and representation learning, which contributes to its
   137	uniqueness. Furthermore, the traditional triplet loss used in person reidentification aims to reduce prediction error by refining the distances
   138	between positive and negative sample pairs, which only considers the
   139	relative distances between positive and negative pairs, while ignoring
   140	the positive sample pairs themselves. To address this, we propose
   141	the Matching Triplet Loss, which focuses chiefly on the relationship
   142	between the matching relevance of positive and negative sample pairs.
   143	In summary, the main contributions of our paper are:
   144	• We propose a learnable batch sampling method called Correlation Graph Sampler (CGS) to provide more challenging training
   145	instances for network training, aiding the model in discriminative
   146	learning.
   147	• We design a high-resolution preservation network called Global
   148	Sparse Attention Network (GSANet) and introduce a global relevance sparse reconstruction module based on sparse representation (GRSR) to achieve feature self-correlation reconstruction.
   149	This attention is global in nature and aims to reduce the loss
   150	of semantic and positional information during downsampling,
   151	thereby enhancing the model’s representation learning capability.
   152	• This paper proposes a matching triplet loss, which is more advantageous for the training and optimization of the model with
   153	respect to difficult samples for metric learning in pedestrian
   154	re-identification.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' - | nl -ba | sed -n '241,668p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   241	3. Method
   242	
   243	2.2. Deep metric learning
   244	
   245	Fig. 2 illustrates the overall training framework of our algorithm.
   246	During the training process of the deep learning model, we first use
   247	the CGS sampler to select example samples with similar characteristics
   248	from the training set and provide them to the backbone network for
   249	training. The backbone network is then used for feature map extraction, and the weight parameters of the backbone network are updated
   250	through the triplet loss function to optimize the model. Since the
   251	performance of the CGS sampler is influenced by the quality of features
   252	extracted by the backbone network, updating the weight parameters of
   253	the backbone network during training can improve the performance of
   254	the CGS.
   255	In Section 3.1, we will explain the implementation principles of the
   256	CGS sampler, while the structure of the main network (GSANet) will be
   257	discussed in detail in Section 3.2. We utilize the matching triplet loss
   258	function as the optimization loss in this model, and further details on
   259	the construction principles of this loss function will be provided in the
   260	subsequent chapters.
   261	
   262	Apart from feature representation learning, metric learning is also
   263	an effective approach to improve the performance of pedestrian reidentification models. However, research on metric learning with largescale training data is still not sufficient. Taking face recognition as an
   264	example, the future direction may be gradually learning from even
   265	larger-scale data to achieve better performance. Although mini-batch
   266	sampling plays a crucial role in deep metric learning, it has not been
   267	widely explored. In addition to online hard example mining for minibatch data sampling, several methods have been proposed for mining
   268	hard examples during the mini-batch data sampling process.
   269	Small-batch samplers also play an important role in deep metric
   270	learning, but there is still little research on them. In addition to online
   271	hard example mining in small batches, several methods have been
   272	proposed for hard example mining in the process of small-batch data
   273	sampling. For example, [25] proposed a random class-based hard example mining method for deep metric learning. It uses learnable class
   274	signatures to identify the nearest classes and further performs instancelevel refinement search in a subset of classes found in the first stage
   275	for hard example mining. In addition, [26] proposed a simple yet
   276	effective method called Group Sampling to mitigate the negative impact
   277	of noisy pseudo-labels in unsupervised person re-identification models.
   278	However, this approach requires significant computational resources.
   279	However, these methods require learning classification parameters for
   280	class mining, which is difficult to handle for large-scale classes and
   281	complex non-Euclidean matchers.
   282	
   283	3.1. Correlation graph sampling
   284	The CGS sampler aims to improve the discriminative ability and
   285	generalization of learning models by constructing a nearest neighbor graph for all classes at the beginning of each epoch, as shown
   286	in Fig. 1(b). Specifically, the sampler selects a random class as the
   287	anchor point and its k nearest neighbor classes, ensuring that each
   288	class has k nearest neighbors. This generates batches of instances
   289	3
   290	
   291	Pattern Recognition 165 (2025) 111591
   292	
   293	F. Min et al.
   294	
   295	Fig. 2. Training framework of the proposed pedestrian re-identification algorithm. The CGS sampler is used to provide correlation samples, CNN is the feature extraction network,
   296	and the triplet loss function is used for supervised training.
   297	𝑝
   298	
   299	where ‖𝑝 𝑖‖ in Eq. (1) is to convert 𝑝𝑖 to a unit column vector. To ensure
   300	‖ 𝑖 ‖2
   301	
   302	the spatial distance property of the original features is preserved, a
   303	Gaussian random mapping matrix is used for hash bucket allocation.
   304	The matrix 𝐴 is a 𝑏 × 𝑐 matrix, where 𝑏 represents the number of
   305	hash buckets and 𝑐 represents the number of channels of the input
   306	feature vector. 𝐴 is a Gaussian random mapping matrix, meaning it
   307	is a random matrix that satisfies a Gaussian distribution. Using this
   308	matrix, the hidden hash bucket assigned to 𝑝𝑖 can be defined as shown
   309	in Eq. (2):
   310	( )
   311	( )
   312	ℎ 𝑝𝑖 = arg max 𝑝̂𝑖 ,
   313	(2)
   314	
   315	Fig. 3. Schematic of Spherical-LSH algorithm.
   316	
   317	where ℎ(𝑝𝑖 ) = 1, 2, … , 𝑏, the function of Eq. (2) is to select the index
   318	corresponding to the largest element in the target column vector 𝑝̂𝑖
   319	as the hash bucket encoding of 𝑝𝑖 . In practice, Spherical LSH encodes
   320	all hash buckets for all categories simultaneously using batch matrix
   321	multiplication, resulting in negligible additional computation. After the
   322	hash bucket allocation of all elements, the focus bucket of the feature
   323	vector 𝑝𝑖 can be represented by the index set, as shown in Eq. (3):
   324	{
   325	( )
   326	( )}
   327	𝜆𝑖 = 𝑗 ∣ ℎ 𝑝𝑗 = ℎ 𝑝𝑖 ,
   328	(3)
   329	
   330	that are primarily similar, providing informative and challenging examples for discriminative learning. Similar to face recognition loss
   331	functions, which enhance the discriminative ability of learning models
   332	by emphasizing sample correlation, the CGS sampler also emphasizes
   333	nearest neighbor classes, potentially further improving discrimination
   334	and generalization.
   335	To ensure that each batch of training samples is correlated and
   336	facilitates discriminative learning, the CGS sampler constructs a nearest
   337	neighbor relation graph for all classes prior to training. The graph’s
   338	nearest neighbor relation is established through two steps: (1) allocating samples from different categories to different hash buckets based on
   339	their spatial distance attribute, which groups related classes in the same
   340	hash buckets; (2) using the feature maps adaptive matching method to
   341	measure the similarity relationship between the selected base class and
   342	other classes in the same hash bucket, resulting in the acquisition of
   343	the graph node nearest neighbor relationship for all classes.
   344	
   345	this approach enables the initial hash coding assignment of the training
   346	dataset based on relevance, with the parameter 𝑏 set to 64 in this
   347	paper. As a result, it divides the samples of the categories with higher
   348	relevance into the same hash bucket.
   349	3.1.2. Nearest neighbor graph node construction
   350	If we want to construct the nearest neighbor graph as shown in Fig.
   351	1(b), it is not enough to simply divide the dataset by relevance using
   352	the LSH function. In this paper, we utilize the feature maps adaptive
   353	matching method to measure the correlation between the selected base
   354	class and other classes with the same hash encoding, and then obtain
   355	the graph’s nearest neighbor relationship for all classes. The matching
   356	method is illustrated in Fig. 4.
   357	To match the query graph and registration graph, the first step is
   358	to input them into the backbone network to obtain their feature maps,
   359	which are then normalized. Next, a fixed-sized local square is extracted
   360	at each position of the feature maps of the query graph to serve as the
   361	convolution kernel. This is an adaptive convolution kernel of the query
   362	graph, whose parameters are constructed in real-time from the feature
   363	maps of the query graph. Unlike a fixed trained convolution kernel, the
   364	convolution kernel constructed in real-time can achieve better results
   365	when used for convolution on the feature maps of the registration
   366	graph, which can be regarded as template matching. Then, the local
   367	matching with the maximum response can be obtained through global
   368	maximum pooling. In this way, the cross-convolution of the two images
   369	yields the correlation weights between the two feature maps, allowing
   370	
   371	3.1.1. Hash bucket allocation of samples
   372	The principles of hashed bucket allocation involve using the Locally
   373	Sensitive Hashing (Spherical-LSH) algorithm to hash code the training
   374	set by category, thereby allocating similar categories to the same hash
   375	bucket as much as possible. The LSH function is a specific type of hash
   376	function that can preserve the spatial distance attribute between the
   377	original data effectively. To put it simply, if we have three points 𝛼1 ,
   378	𝛼2 , and 𝛼3 , where 𝛼1 , 𝛼2 are very close to each other while 𝛼1 , 𝛼3 are
   379	far apart, a hash function Hash(.) belongs to the LSH function if the
   380	probability of collision between Hash(𝛼1 ) and Hash(𝛼2 ) is much higher
   381	than that between Hash(𝛼1 ) and Hash(𝛼3 ), as shown in Fig. 3.
   382	The feature vector of sample category 𝑝𝑖 is converted to a column
   383	vector, and then the hash bucket allocation is performed based on the
   384	Spherical LSH algorithm. The operational principle of this allocation is
   385	shown in Eq. (1).
   386	(
   387	)
   388	𝑝𝑖
   389	𝑝̂𝑖 = 𝐴
   390	,
   391	(1)
   392	‖𝑝𝑖 ‖
   393	‖ ‖2
   394	4
   395	
   396	Pattern Recognition 165 (2025) 111591
   397	
   398	F. Min et al.
   399	
   400	denoted by Layer(𝑠, 𝑓 ), using 𝑠 as the index of different stages and 𝑓 as
   401	the index of the corresponding sub-stream of the corresponding stage,
   402	so that 𝑠 = 1 and 𝑓 = 1 in the first stage, the number of channels
   403	Chs(𝑠, 𝑓 ) and the input resolution Size(𝑠, 𝑓 ) of the sth stage Layer(𝑠, 𝑓 )
   404	are shown in Eq. (5).
   405	{
   406	Chs(𝑠, 𝑓 ) = 2𝑠−1 Chs(1, 1)
   407	,
   408	(5)
   409	1
   410	Size(𝑠, 𝑓 ) = 2𝑠−1
   411	Size(1, 1)
   412	Each scale of the feature stream in Fig. 5 is called a Stage, and each
   413	part of the feature stream containing only same-resolution interactions
   414	is called a Block, with only each Block in Stage 1 containing a global
   415	relevance sparse reconstruction (GRSR) module.
   416	Unlike conventional network models, such as ResNet [30] and
   417	VGGNet [31], GSANet maintains the high-resolution representation
   418	throughout the process. The network has the following key features:
   419	(1) it connects the high-resolution to low-resolution convolutional
   420	data streams in parallel and processes feature extraction from sameresolution modules using a residual module (Residual), where the
   421	Residual module is a structure consisting of 2 layers of 2D convolutional
   422	layers with a convolutional kernel size of 3 and the residuals of the
   423	original input feature layers; (2) the GRSR module is utilized to globally
   424	explore correlated pixel-level features in the feature map and perform
   425	feature sparse reconstruction to enhance the network’s capability to
   426	learn global correlation features. This results in a more semantically
   427	rich and accurate feature representation that preserves spatial location
   428	information. The network structure is illustrated in Fig. 5.
   429	
   430	Fig. 4. Principle of feature map similarity matching.
   431	
   432	for the determination of the ordering relation of the first p nearest
   433	neighbor points of each base class of the same hash bucket.
   434	To summarize, at the beginning of each training round, CGS uses
   435	the Spherical LSH algorithm to assign hash buckets to training samples.
   436	The newly obtained discriminative model from the previous round
   437	of training is used to feedback into this round for re-evaluating the
   438	similarity between categories within each hash bucket. This process
   439	also involves constructing a graph of all classes, which can be utilized
   440	for information relevance sampling. Through this method, we aim to
   441	achieve a continuously optimizable and learnable model. Specifically,
   442	assuming there are 𝑧 total classes to be trained, each class in 𝑍 is
   443	assigned a hash bucket using the Spherical LSH algorithm, and all
   444	selected base class samples are used to compute weight correlation with
   445	other classes belonging to the same hash bucket. For each class 𝑧, the
   446	top 𝑝 − 1 nearest neighbor classes can be retrieved, as shown in Eq. (4).
   447	{
   448	}
   449	𝜂(𝑧) = 𝑧𝑖 ∣ 𝑖 = 1, 2 … , 𝑝 − 1 ,
   450	
   451	3.3. Global relevance sparse reconstruction
   452	The convolution operation in the feature extraction network can
   453	extract significant features in response to the convolution kernel, but
   454	these features may lack global autocorrelation [29,30]. In other words,
   455	the correlation between pixel-level features in a single feature layer
   456	may not be effectively explored. Moreover, for low-resolution images
   457	with limited information, multiple convolution layers may result in
   458	discriminative feature loss, which is not desirable for feature extraction. To address these issues, this paper proposes a global relevance
   459	sparse reconstruction module (GRSR) for mining the autocorrelation of
   460	pixel-level features in feature maps. The GRSR module can effectively
   461	preserve the original semantic information of feature layers and utilize
   462	the autocorrelation of pixel-level feature information in the same feature layer for feature reconstruction. This approach can greatly reduce
   463	the loss of effective information and enhance the discriminative feature
   464	extraction ability and generalization potential of the network model.
   465	The objective function of GRSR can be expressed as shown in Eq. (6),
   466	as described in the paper.
   467	(
   468	)
   469	𝑛
   470	∑
   471	𝑆 𝑥𝑖 , 𝑥 𝑗
   472	( )
   473	y𝑖 =
   474	(6)
   475	(
   476	) 𝑔 𝑥𝑗 ,
   477	∑𝑛
   478	𝑗=1
   479	̂ 𝑆 𝑥𝑖 , 𝑥𝑗̂
   480	𝑗=1
   481	
   482	(4)
   483	
   484	where 𝑝 in Eq. (4) denotes the number of classes loaded in each small
   485	batch. 𝜂(𝑧) means that the base class is the top 𝑝 − 1 nearest neighbor
   486	class of 𝑧. Accordingly, a graph 𝐺 = (𝑅, 𝐸), is constructed, where
   487	𝑅 = {𝑧 ∣ 1, 2, 3 … , 𝑍} is the set of base classes, i.e. each base class
   488	{(
   489	)
   490	}
   491	can be considered as a node. 𝐸 =
   492	𝑧, 𝑧𝑖 ∣ 𝑧𝑖 ∈ 𝜂(𝑧) denotes the
   493	edges of the nearest neighbor graph constructed with 𝑧 as the base
   494	class. Finally, using class 𝑧 as the base class, the first 𝑝 − 1 neighbor
   495	classes are retrieved in 𝐺. Together with the base class, the set 𝑈 =
   496	{
   497	(
   498	)
   499	}
   500	{𝑐} ∪ 𝑐𝑖 ∣ 𝑐, 𝑐𝑖 ∈ 𝐸 is obtained. For each class in 𝑈 , we randomly
   501	sample 𝑘 instances from that class to create a small batch of 𝑏 =
   502	𝑝 × 𝑘 training samples. Unlike other small-batch sampling methods,
   503	the CGS sampler always performs 𝑍 iterations per cycle of the small
   504	batch, regardless of the values of 𝑏, 𝑝, and 𝑘. The hash bucket hash
   505	assignments are computed using the feature relevance measure only
   506	once per epoch. Extensive experimental evaluations have confirmed
   507	that this sampling method is effective in providing strongly correlated
   508	and challenging training examples, which improve the discriminative
   509	feature representation learning ability and generalization performance
   510	of the model.
   511	
   512	where 𝑥𝑖 , 𝑥𝑗 , 𝑥𝑗̂ in Eq. (6) denote the pixel-level features located at
   513	positions 𝑖, 𝑗, 𝑗̂ of feature layer 𝑋, where 𝑆 (. , .) denotes the Gaussian
   514	projection kernel function that measures the correlation of two pixel
   515	features, 𝑔 (.) is the feature transformation function that maps the
   516	original graph features using the obtained weights.
   517	Therefore, y𝑖 is the reconstructed target result, but it may not
   518	accurately capture autocorrelation because it aggregates information
   519	from all locations and requires global matching calculations, which
   520	can increase computational resource usage. To mitigate this issue,
   521	sparse constraints
   522	complexity.
   523	[ ( can
   524	) be imposed
   525	( )]to reduce computational
   526	[ (
   527	)
   528	(
   529	)]
   530	Suppose 𝐷 = 𝑔 𝑥1 , … … , 𝑔 𝑥𝑛 and 𝑎𝑖 = 𝑆 𝑥𝑖 , 𝑥1 , … ., 𝑆 𝑥𝑖 , 𝑥𝑛 ,
   531	then the objective function of the above Eq. (6) can be transformed into
   532	a sparse expression, 𝑦𝑖 = 𝐷𝑎𝑖 . By restricting the number of non-zero
   533	elements of 𝑎𝑖 to a constant 𝑘, Eq. (6) can be derived as Eq. (7).
   534	
   535	3.2. Global sparse attention network
   536	According to current research, good feature extraction capability
   537	and assisted feature representation learning are effective in improving the generalization potential of pedestrian re-identification models.
   538	Building on the work of Sun et al. [29], this paper argues that preserving high-resolution spatial location representation during backbone
   539	network feature extraction can reduce the loss of spatial location
   540	representation information of sample features, which is important for
   541	improving the feature representation learning ability of pedestrian reidentification models. To address this issue, this paper proposes a global
   542	sparse attention network (GSANet), as shown in Fig. 5. The input image
   543	size was set to 256 × 192, GSANet can be divided into four phases
   544	
   545	‖𝑎𝑖 ‖ ≤ 𝑘
   546	S.t
   547	‖ ‖0
   548	(
   549	)
   550	∑
   551	𝑆 𝑥𝑖 , 𝑥 𝑗
   552	( ),
   553	=
   554	(
   555	) 𝑔 𝑥𝑗
   556	∑
   557	̂𝑗∈𝛿𝑖 𝑆 𝑥𝑖 , 𝑥𝑗̂
   558	𝑗∈𝛿𝑖
   559	
   560	𝑦𝑖 = 𝐷𝑎𝑖
   561	
   562	5
   563	
   564	(7)
   565	
   566	Pattern Recognition 165 (2025) 111591
   567	
   568	F. Min et al.
   569	
   570	Fig. 5. GSANet network structure diagram. The network consists of 4 stages and 4 blocks, and the feature streams in the same stage have the same number of channels. Only
   571	the Stage 1 feature stream contains Global Relevance Sparse Reconstruction (GRSR).
   572	
   573	Fig. 6. GRSR module reconfiguration schematic.
   574	
   575	where 𝛿𝑖 in Eq. (7) is the index set of non-zero elements of 𝑎𝑖 , i.e. 𝛿𝑖 =
   576	{
   577	}
   578	𝑗 ∣ 𝑎𝑖 [𝑗] ≠ 0 , 𝛿𝑖 is the search range of 𝑥𝑖 . In the case of sparse attention, elements with zero coefficients are ignored, leading to a significant reduction in computational cost. To ensure that the elements 𝛿𝑖
   579	in Eq. (7) remain sparse and are incorporated into the most relevant
   580	global features, GRSR employs the Spherical Local Sensitive Hashing
   581	(Spherical-LSH) algorithm, which utilizes the Gaussian random mapping method by Eq. (1). This algorithm assigns relevant pixel-level
   582	features to the same hash bucket and irrelevant features to different
   583	hash buckets, resulting in sparse elements in 𝛿𝑖 The details of this
   584	process are not discussed here. After completing the hash-coding of all
   585	pixel-level features, the feature reconstruction principle is illustrated in
   586	Eq. (7).
   587	Finally, the reconstructed features are fused with the original features, and the dimensionality of the fused features remains the same.
   588	By using the method described above, each pixel-level feature in the
   589	feature layer can be reconstructed with the features associated with
   590	it in that layer, which helps to enhance the global associative feature
   591	learning of the backbone network. For instance, for a pixel-level feature
   592	𝑥𝑖 the steps of GRSR module feature reconstruction are illustrated in
   593	Fig. 6.
   594	
   595	resolutions to interact with each other multiple times, the resulting feature representation is more semantically rich, and the spatial location
   596	information is more accurately represented.
   597	
   598	3.5. Loss function
   599	
   600	Considering the correlation between the small batch training samples provided by the CGS sampler, we choose to construct a triple-based
   601	loss function for the sorting learning problem in small batches. To
   602	design the learning loss function, the more commonly used triplet loss
   603	function is mainly designed to minimize the distance between positive
   604	sample pairs and maximize the distance between negative sample pairs.
   605	However, this paper mainly focuses on the relative relationship between the matching correlation degrees of positive and negative sample
   606	pairs, and proposes a matching triplet loss for metric learning. The
   607	principle of this method is shown in Eq. (8), where, 𝜀(., .) represents
   608	feature extractor, 𝜙(., .), is used to measure feature similarity, 𝑚 is the
   609	margin, 𝑝 represents the number of classes loaded in each batch, and 𝑘
   610	represents the random sampling of 𝑘 images for each class.
   611	
   612	3.4. Multi-scale feature fusion
   613	
   614	Eq. (8) can be divided into two parts. One part is the relative loss of
   615	feature matching similarity between positive sample pairs and negative
   616	sample pairs. During training, this part optimizes the feature matching
   617	similarity value of positive sample pairs in the direction of larger
   618	values, while the feature similarity value between negative sample pairs
   619	is optimized in the direction of smaller values. However, optimizing the
   620	feature similarity value between negative sample pairs in the desired
   621	direction can be difficult with this loss function alone, as it is greatly
   622	
   623	The integration of feature layers with different resolutions in the
   624	GSANet network is illustrated in Fig. 7. The high-resolution feature
   625	layer employs 2D convolution to match the channel numbers, whereas
   626	the low-resolution feature layer generates a new high-resolution representation by performing upsampling and feature fusion.
   627	After passing through each Block module, the input features in
   628	GSANet undergo multi-resolution feature fusion. By allowing multiple
   629	6
   630	
   631	Pattern Recognition 165 (2025) 111591
   632	
   633	F. Min et al.
   634	
   635	Fig. 7. Schematic diagram of feature layer fusion method. Right legend: strided 3 × 3(stride = 2, padding = 1), up samp. 1 × 1 = nearest neighbor up-sampling following a 1 × 1
   636	convolution, Arrow: Features are fused in an additive manner.
   637	
   638	influenced by 𝑚.
   639	𝑙(𝑋) =
   640	
   641	𝑝
   642	𝑘
   643	∑∑
   644	
   645	( ( ) ( ))
   646	[𝑚 − min 𝜙 𝜀 𝜔𝑎𝑖 , 𝜀 𝜔𝑡𝑖 .
   647	𝑡=1…𝑘
   648	
   649	𝑖=1 𝑎=1
   650	
   651	( ( ) ( ))
   652	+ max 𝜙 𝜀 𝜔𝑎𝑖 , 𝜀 𝜔𝑛𝑗 ]
   653	𝑗=1…𝑝
   654	𝑗≠𝑖
   655	𝑛=1…𝑘
   656	
   657	+
   658	
   659	training on the training subset of one dataset and evaluating on the test
   660	subset of another datasets. Rank-1 and average accuracy (mAP) were
   661	employed as performance evaluation metrics to assess the performance
   662	and generalization potential of the training model.
   663	
   664	𝑝 ∑
   665	𝑘
   666	∑
   667	
   668	4.3. Comparison to the state of the art

codex
第五篇有明显“相互促进”的包装：采样器要好特征，特征网络又需要相关难样本，作者把这两个常见方向绑成一个闭环。它的关键不在某个注意力模块，而在“采样和表征不是两件事”的重定义。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 6 -l 12 'Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf' - | nl -ba | rg -n \"Ablation|ablation|CGS|GRSR|GSANet|matching triplet|triplet|generalization|Visualization|Table|Fig|Sampler|PK|Graph|sparse|feature reconstruction|Market|MSMT|improvement|impact\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2-     2	
3-     3	F. Min et al.
4-     4	
5:     5	Fig. 5. GSANet network structure diagram. The network consists of 4 stages and 4 blocks, and the feature streams in the same stage have the same number of channels. Only
6:     6	the Stage 1 feature stream contains Global Relevance Sparse Reconstruction (GRSR).
7-     7	
8:     8	Fig. 6. GRSR module reconfiguration schematic.
9-     9	
10-    10	where 𝛿𝑖 in Eq. (7) is the index set of non-zero elements of 𝑎𝑖 , i.e. 𝛿𝑖 =
11-    11	{
12-    12	}
13:    13	𝑗 ∣ 𝑎𝑖 [𝑗] ≠ 0 , 𝛿𝑖 is the search range of 𝑥𝑖 . In the case of sparse attention, elements with zero coefficients are ignored, leading to a significant reduction in computational cost. To ensure that the elements 𝛿𝑖
14:    14	in Eq. (7) remain sparse and are incorporated into the most relevant
15:    15	global features, GRSR employs the Spherical Local Sensitive Hashing
16-    16	(Spherical-LSH) algorithm, which utilizes the Gaussian random mapping method by Eq. (1). This algorithm assigns relevant pixel-level
17-    17	features to the same hash bucket and irrelevant features to different
18:    18	hash buckets, resulting in sparse elements in 𝛿𝑖 The details of this
19-    19	process are not discussed here. After completing the hash-coding of all
20:    20	pixel-level features, the feature reconstruction principle is illustrated in
21-    21	Eq. (7).
22-    22	Finally, the reconstructed features are fused with the original features, and the dimensionality of the fused features remains the same.
23-    23	By using the method described above, each pixel-level feature in the
24-    24	feature layer can be reconstructed with the features associated with
25-    25	it in that layer, which helps to enhance the global associative feature
26-    26	learning of the backbone network. For instance, for a pixel-level feature
27:    27	𝑥𝑖 the steps of GRSR module feature reconstruction are illustrated in
28:    28	Fig. 6.
29-    29	
30-    30	resolutions to interact with each other multiple times, the resulting feature representation is more semantically rich, and the spatial location
31-    31	information is more accurately represented.
32-    32	
33-    33	3.5. Loss function
34-    34	
35:    35	Considering the correlation between the small batch training samples provided by the CGS sampler, we choose to construct a triple-based
36-    36	loss function for the sorting learning problem in small batches. To
37:    37	design the learning loss function, the more commonly used triplet loss
38-    38	function is mainly designed to minimize the distance between positive
39-    39	sample pairs and maximize the distance between negative sample pairs.
40-    40	However, this paper mainly focuses on the relative relationship between the matching correlation degrees of positive and negative sample
41:    41	pairs, and proposes a matching triplet loss for metric learning. The
42-    42	principle of this method is shown in Eq. (8), where, 𝜀(., .) represents
43-    43	feature extractor, 𝜙(., .), is used to measure feature similarity, 𝑚 is the
44-    44	margin, 𝑝 represents the number of classes loaded in each batch, and 𝑘
--
56-    56	direction can be difficult with this loss function alone, as it is greatly
57-    57	
58-    58	The integration of feature layers with different resolutions in the
59:    59	GSANet network is illustrated in Fig. 7. The high-resolution feature
60-    60	layer employs 2D convolution to match the channel numbers, whereas
61-    61	the low-resolution feature layer generates a new high-resolution representation by performing upsampling and feature fusion.
62-    62	After passing through each Block module, the input features in
63:    63	GSANet undergo multi-resolution feature fusion. By allowing multiple
64-    64	6
65-    65	
66-    66	Pattern Recognition 165 (2025) 111591
67-    67	
68-    68	F. Min et al.
69-    69	
70:    70	Fig. 7. Schematic diagram of feature layer fusion method. Right legend: strided 3 × 3(stride = 2, padding = 1), up samp. 1 × 1 = nearest neighbor up-sampling following a 1 × 1
71-    71	convolution, Arrow: Features are fused in an additive manner.
72-    72	
73-    73	influenced by 𝑚.
--
94-    94	training on the training subset of one dataset and evaluating on the test
95-    95	subset of another datasets. Rank-1 and average accuracy (mAP) were
96-    96	employed as performance evaluation metrics to assess the performance
97:    97	and generalization potential of the training model.
98-    98	
99-    99	𝑝 ∑
100-   100	𝑘
--
107-   107	( ( ) ( ))
108-   108	[ max 𝜙 𝜀 𝜔𝑎𝑖 , 𝜀 𝜔𝑛𝑗 ],
109-   109	
110:   110	Table 1 presents a comparison between the method proposed in this
111-   111	paper and the current state-of-the-art person re-identification methods.
112:   112	For the experiments, the MSMT17 and Market-1501 datasets were
113-   113	used as separate training samples, and after the model training, all
114:   114	three datasets (MSMT17, Market-1501, and CUHK03) were used for
115:   115	testing. Among them, MSMT17 was divided into two training modes,
116:   116	one using all images of the dataset for training without considering the subsets, denoted as MSMT17(all) in the table. The comparison includes recently published person re-identification methods that
117-   117	were evaluated on cross-datasets, including SNR [19], ADIN [20],
118-   118	M3 L [22], OSNet-IBN [45], OSNet-AIN [41], QAConv [38], CBN [39],
119:   119	MuDeep [46], QAConv-GS [11], MDA [44] and DMN [36]. Table 1
120:   120	shows that GSANet-CGS significantly improves the previous SOTA. For
121:   121	example, in the case of Market-1501 → CUHK03 cross-dataset, the
122-   122	Rank-1 and mAP improved by 3.2% and 2.9%, respectively, compared
123:   123	to QAConv-GS. For Market-1501 → MSMT17, the Rank-1 and mAP
124:   124	improved by 2.3% and 1.9%, respectively. In the case of MSMT17
125:   125	(all) → Market-1501, the Rank-1 and mAP improvements were 3.2%
126:   126	and 1.8%, respectively. Compared to MDA [44], for the Market-1501
127:   127	→ MSMT17 cross-dataset, the performance improved by 1.7% and
128:   128	1.6%, respectively. For MSMT17 → Market-1501, the Rank-1 and mAP
129:   129	improvements were 3.2% and 1.8%, respectively.
130-   130	Note that, M3 L [19] and DMN [36] employ distinct evaluation protocols, making direct comparisons of their results challenging. Specifically, both M3 L and DMN are trained on three datasets: CUHK03,
131:   131	Market-1501, DukeMTMC-reID, and MSMT17, with one dataset reserved for testing purposes. M3 L achieves remarkable results on
132-   132	CUHK03-NP, although a direct comparison to our results is not feasible,
133-   133	as it surpasses all of our results, including those obtained from training
134:   134	on the entire MSMT17 image set. However, on Market-1501, our
135:   135	proposed method trained on MSMT17 outperforms DMN trained on
136:   136	MSMT17 by 3.8% in terms of Rank-1 accuracy, while the mean Average
137:   137	Precision (mAP) scores are comparable. Furthermore, on MSMT17, our
138:   138	proposed method trained on Market-1501 significantly outperforms
139-   139	DMN with a 1.0% increase in Rank-1 accuracy. These findings are
140-   140	encouraging, considering that our training datasets are subsets of those
141-   141	used by M3 L and DMN.
142-   142	The experimental results confirm the effectiveness of the method in
143:   143	this paper, and the GSANet-CGS algorithm shows good performance in
144-   144	
145-   145	𝑗=1…𝑝
146-   146	𝑖=1 𝑎=1
--
153-   153	negative sample pairs is optimized in the direction of smaller values as
154-   154	much as possible, thereby achieving better optimization of the network
155-   155	model parameters and effectively improving recognition performance
156:   156	and generalization potential.
157-   157	4. Experiment
158-   158	4.1. Experimental details
159:   159	In this study, the GSANet was utilized as the feature extraction
160-   160	network, and the fused high-resolution stream was employed as the
161-   161	effective feature map. The input image size was set to 256 × 192,
162-   162	and several common datas augmentation techniques were utilized,
--
164-   164	batch size was set to 64, and the network was trained using the SGD
165-   165	optimizer, with a minimum learning rate of 0.0002 and a maximum
166-   166	learning rate of 0.001. The maximum number of training epochs was
167:   167	120, and the triplet loss function parameters were set to 𝑚 = 16, and
168-   168	the number of random sampling images per class 𝑘 = 2.
169-   169	4.2. Datasets
170-   170	In this paper, three large-scale person re-identification datasets,
171:   171	CUHK03 [32], Market-1501 [33], and MSMT17 [34], were utilized for
172:   172	the experiments. The Market-1501 dataset contains 32,668 images from
173-   173	1501 identities captured by six cameras. The training subset includes
174-   174	12,936 images from 751 identities, while the test subset includes
175:   175	19,732 images from 750 identities. The MSMT17 dataset comprises
176-   176	4101 identities and 126,441 images taken from 15 cameras. It is split
177-   177	into a training set with 32,621 images from 1041 identities and a
178-   178	test set with 3010 identities and 93,820 images. The CUHK03 dataset
--
183-   183	Pattern Recognition 165 (2025) 111591
184-   184	
185-   185	F. Min et al.
186:   186	Table 1
187-   187	Comparison of cross-evaluation results % for datasets with Frontier algorithms, with ‘‘–’’ indicating not reported or not applicable.
188-   188	Method
189-   189	
--
193-   193	
194-   194	CUHK03-NP
195-   195	
196:   196	Market-1501
197-   197	
198:   198	MSMT17
199-   199	
200-   200	Rank-1
201-   201	
--
246-   246	OG-NET [43]
247-   247	MDA [44]
248-   248	QAConv-GS [11]
249:   249	GSANet-CGS
250-   250	
251-   251	PR’23
252-   252	TPAMI’20
--
260-   260	CVPR’22
261-   261	Ours
262-   262	
263:   263	Market-1501
264:   264	Market-1501
265:   265	Market-1501
266:   266	Market-1501
267:   267	Market-1501
268:   268	Market-1501
269:   269	Market-1501
270:   270	Market-1501
271:   271	Market-1501
272:   272	Market-1501
273:   273	Market-1501
274-   274	
275-   275	–
276-   276	10.3
--
354-   354	OG-NET [43]
355-   355	QAConv-GS [11]
356-   356	MDA [44]
357:   357	GSANet-CGS
358-   358	
359-   359	ECCV’18
360-   360	PR’23
--
368-   368	CVPR’22
369-   369	Ours
370-   370	
371:   371	MSMT17
372:   372	MSMT17
373:   373	MSMT17
374:   374	MSMT17
375:   375	MSMT17
376:   376	MSMT17
377:   377	MSMT17
378:   378	MSMT17
379:   379	MSMT17
380:   380	MSMT17
381:   381	MSMT17
382-   382	
383-   383	–
384-   384	22.8
--
456-   456	QAConv [38]
457-   457	OSNet-AIN [41]
458-   458	QAConv-GS [11]
459:   459	GSANet-CGS
460-   460	
461-   461	CVPR’19
462-   462	ECCV’20
--
464-   464	CVPR’22
465-   465	Ours
466-   466	
467:   467	MSMT17(all)
468:   468	MSMT17(all)
469:   469	MSMT17(all)
470:   470	MSMT17(all)
471:   471	MSMT17(all)
472-   472	
473-   473	–
474-   474	25.3
--
506-   506	–
507-   507	–
508-   508	
509:   509	Table 2
510-   510	Performance comparison of different sampling methods, where ‘‘–’’ indicates not
511:   511	reported or not applicable (%). R1 stands for Rank-1, and MS-all represents MSMT
512-   512	(all).
513-   513	Method
514-   514	
--
516-   516	
517-   517	CUHK03-NP
518-   518	
519:   519	Market
520-   520	
521-   521	R1
522-   522	
--
526-   526	
527-   527	mAP
528-   528	
529:   529	Table 3
530:   530	Experimental results on the impact of CGS and GRSR modules on generality with
531-   531	different backbones (%). R1 stands for Rank-1.
532-   532	Backbone
533-   533	
534:   534	CGS
535-   535	
536:   536	GRSR
537-   537	
538-   538	Training
539-   539	
540-   540	CUHK03-NP
541-   541	
542:   542	Market
543-   543	
544-   544	R1
545-   545	
--
551-   551	
552-   552	3
553-   553	
554:   554	MSMT
555:   555	MSMT
556:   556	MSMT
557-   557	
558-   558	14.9
559-   559	18.6
--
573-   573	
574-   574	3
575-   575	
576:   576	MSMT
577:   577	MSMT
578:   578	MSMT
579-   579	
580-   580	17.3
581-   581	21.4
--
595-   595	
596-   596	3
597-   597	
598:   598	MSMT
599:   599	MSMT
600:   600	MSMT
601-   601	
602-   602	18.5
603-   603	22.7
--
615-   615	49.8
616-   616	51.5
617-   617	
618:   618	MSMT17
619-   619	R1
620-   620	
621-   621	mAP
622-   622	
623-   623	RS
624:   624	PK
625-   625	Cluster
626:   626	CGS
627-   627	
628:   628	Market
629:   629	Market
630:   630	Market
631:   631	Market
632-   632	
633-   633	17.5
634-   634	18.3
--
659-   659	19.1
660-   660	
661-   661	RS
662:   662	PK
663-   663	Cluster
664:   664	CGS
665-   665	
666:   666	MSMT
667:   667	MSMT
668:   668	MSMT
669:   669	MSMT
670-   670	
671-   671	18.1
672-   672	18.9
--
697-   697	–
698-   698	
699-   699	RS
700:   700	PK
701-   701	Cluster
702:   702	CGS
703-   703	
704-   704	MS-all
705-   705	MS-all
--
738-   738	
739-   739	HRNet
740-   740	
741:   741	GSANet
742-   742	
743-   743	3
744-   744	3
--
747-   747	3
748-   748	3
749-   749	
750:   750	that random shuffle and PK sampling perform the worst, as they are
751-   751	completely random and unable to provide informative and challenging
752-   752	samples for discriminative learning. The performance of the approach
753-   753	using spectral clustering is generally improved, with a slight increase
754-   754	in performance due to the small batch of informative samples in each
755:   755	cluster. The CGS sampler improves by 3.0% and 2.8% across dataset
756:   756	with Market-1501 → CUHK03 for Rank-1 and mAP, respectively, compared to Cluster. With Market-1501 → MSMT17, they improved by
757:   757	3.9% and 2.9%, respectively. With MSMT17 (all) → Market-1501, the
758:   758	improvements in Rank-1 and mAP were 4.2% and 3.4% respectively In
759:   759	MSMT17 (all) → CUHK03, the improvements were 3.5% in Rank-1 and
760-   760	2.4% in mAP. The above comparison of experimental results confirms
761-   761	the potential of batch samplers that help improve the discrimination
762:   762	and generalization ability of learning models by exploring correlations
763:   763	among datas. However, this improvement is not sufficient. In contrast,
764:   764	the CGS sampler provides more informative instances for discriminative
765-   765	learning, and thus outperforms the other three methods.
766-   766	
767-   767	both the test results of the training sample library and the cross-library
768:   768	test results, reflecting that the algorithm has strong distinguishing feature extraction. The GSANet-CGS shows good performance in both the
769-   769	training sample library and cross-library test results, which reflects that
770-   770	the algorithm has strong differentiated feature extraction and strong
771:   771	model generalization ability.
772:   772	4.4. Ablation study
773-   773	4.4.1. Comparison of different sampling methods
774:   774	Table 2 shows the comparison of the effects of four small-batch
775:   775	sampling methods on model performance when using the same GSANet
776-   776	feature extraction network and loss function. These methods include
777:   777	PK sampling, random shuffle, Cluster [10] sampling, and the proposed
778:   778	CGS method. Due to the limitation of k-means clustering which does
779-   779	not support non-Euclidean metrics, we use spectral clustering instead
780:   780	for Cluster. From the experimental results in Table 2, we can observe
781-   781	
782:   782	4.4.2. The effect of GSANet, CGS and GRSR on generality
783:   783	To validate the effectiveness of CGS and GRSR, we performed
784:   784	experiments using HRNet [29], ResNet [30], and GSANet as backbone
785-   785	8
786-   786	
787-   787	Pattern Recognition 165 (2025) 111591
788-   788	
789-   789	F. Min et al.
790:   790	Table 4
791:   791	Comparison of loss functions. Lifted: Lifted loss. Triplet: Hard triplet loss. M-Triplet
792:   792	matching triplet loss.
793-   793	Method
794-   794	
795-   795	Training
796-   796	
797-   797	CUHK03-NP
798-   798	
799:   799	Market
800-   800	
801-   801	R1
802-   802	
--
810-   810	
811-   811	mAP
812-   812	
813:   813	Table 5
814:   814	Comparison of model complexity and time cost when training in Market1501 dataset.
815-   815	
816:   816	MSMT
817-   817	
818-   818	Lifted
819-   819	Triplet
820-   820	M-Triplet
821-   821	
822:   822	Market
823:   823	Market
824:   824	Market
825-   825	
826-   826	21.2
827-   827	21.8
--
851-   851	Triplet
852-   852	M-Triplet
853-   853	
854:   854	MSMT
855:   855	MSMT
856:   856	MSMT
857-   857	
858-   858	23.5
859-   859	23.9
--
892-   892	ResNet50
893-   893	CBN [49]
894-   894	QAConv-GS [31]
895:   895	GSANet
896:   896	GSANet+GRSR
897:   897	GSANet+GRSR+CGS
898-   898	
899:   899	Market
900:   900	Market
901:   901	Market
902:   902	Market
903:   903	Market
904:   904	Market
905-   905	
906-   906	5.37
907-   907	6.35
--
924-   924	103
925-   925	104
926-   926	
927:   927	networks. CGS and GRSR were employed as optional modules. The
928:   928	model was trained on the MSMT17 dataset and evaluated on the Market
929:   929	and CUHK03 datasets. The experimental results are shown in Table 3. It
930:   930	can be observed that the complete model (B+CGS+GRSR) achieves the
931-   931	best performance and outperforms the backbone model (B) alone. This
932:   932	demonstrates the effectiveness of CGS and GRSR in facilitating discriminative learning of the model. In the complete model (B+CGS+GRSR)
933:   933	with GSANet as the backbone, compared to ResNet, the Rank-1 and
934:   934	mAP are improved by 4.0% and 3.5%, respectively, on the MSMT17
935:   935	→ CUHK03-NP task. On the MSMT17 → Market-1501 task, the Rank1 and mAP are improved by 5.2% and 4.4%, respectively. Compared
936:   936	to HRNet, on the MSMT17 → CUHK03-NP task, the Rank-1 and mAP
937:   937	are improved by 1.4% and 0.4%, respectively. On the MSMT17 →
938:   938	Market-1501 task, the Rank-1 and mAP are improved by 1.4% and
939:   939	0.6%, respectively. These results demonstrate that GSANet exhibits
940:   940	better generalization potential in the field of person re-identification
941-   941	compared to ResNet and HRNet.
942-   942	4.4.3. Different loss functions
943:   943	The proposed network is compatible with existing optimization objectives, including lifted loss [47], triplet loss [48], and matching triplet
944:   944	loss. We evaluate the impact of these losses on the model performance
945:   945	using GSANet as the backbone network. Additional results are reported
946:   946	in Table 4. The results show that different metric learning losses have
947:   947	varying effects on the model. It should be noted that the proposed CGS
948:   948	sampler demonstrates the effectiveness of the matching triplet loss by
949-   949	itself with K = 2. If K exceeds 2, it may result in excessively stringent
950-   950	training conditions, making convergence challenging.
951-   951	4.5. Comparison of model complexity and time cost
952:   952	Fig. 8. Average Rank-1 and average mAP (%) performance with (a) different batch
953:   953	sizes, and (b) different margin parameters, trained on MSMT17.
954-   954	
955-   955	In addition to accuracy, we also compared the complexity and time
956:   956	cost of the models. The comparison results are shown in Table 5. It
957:   957	is worth noting that all the data in Table 5 were obtained from tests
958-   958	conducted on Tesla V100. During the experiments, we observed convergence of the loss function within 100 training iterations, indicating
959-   959	the stability of the training process. The compared models include the
960:   960	backbone network (ResNet50), CBN, QAConv-GS, and GSANet. Among
961-   961	them, CBN, QAConv, and QAConv-GS use ResNet50 as the backbone.
962-   962	We obtained the GFLOPs (giga floating-point operations per second)
963:   963	and average training time (s/epoch) on the Market1501 dataset. It can
964-   964	be seen that our model has the highest complexity and longest training
965-   965	time. This is primarily attributed to two factors: the high-resolution
966:   966	flow representation maintained by GSANet and the additional computational cost introduced by the CGS sampler and GRSR. However, these
967-   967	trade-offs are justified as they result in improved accuracy.
968-   968	Furthermore, we evaluated the sampling time of the latest GS
969:   969	sampler and our CGS sampler on a Tesla V100. We found that the
970:   970	latest GS sampler takes 4 s for sampling computation on the Market
971:   971	dataset and 40 s on the MSMT (all) dataset. When facing datasets with
972-   972	more identities, GS consumes a significant amount of computational
973:   973	resources. Encouragingly, CGS not only exhibits stable and excellent
974-   974	sampling performance but also requires only 0.1 s and 1 s for sampling
975:   975	evaluation on the Market dataset and MSMT (all) dataset, respectively,
976-   976	significantly reducing the computational complexity.
977-   977	
978-   978	4.6. Parameter analysis
979-   979	
980:   980	Fig. 8 depicts the performance of the method proposed in this paper
981-   981	under different conditions of batch size (B) and correlation metric
982:   982	margin parameter (𝑚). The model training is conducted on MSMT17
983:   983	and the cross-datasets performance is evaluated on Market-1501 and
984-   984	CUHK03. Average Rank-1 and average mAP are used as performance
985:   985	evaluation metrics for Market-1501 and CUHK03, respectively. It can
986:   986	be observed from Fig. 8(a) that Rank-1 and mAP increase within a
987-   987	certain range of batch size B and the model performance reaches
988:   988	saturation at B = 64. On the other hand, Fig. 8(b) shows the effect of the
989-   989	correlation metric margin parameter 𝑚 on the algorithm’s performance.
990-   990	The model performance improves with the increase of 𝑚 and achieves
991-   991	the best result at 𝑚 = 16. However, the performance decreases significantly when 𝑚 exceeds 64 due to the large correlation metric margin
--
997-   997	
998-   998	F. Min et al.
999-   999	
1000:  1000	Fig. 9. (a) Example groups of Graph Sampling on Market-1501 (left) and MSMT17 (right). (b) Example groups of CGS sampler, nearest neighbor classes generated during training
1001:  1001	on Market-1501 (left) and MSMT17 (right). In each group, the top left image is the base class and the other images are the closest nearest neighbor classes.
1002-  1002	
1003:  1003	CGS sampler has limitations. As it provides small batches of samples
1004-  1004	with strong correlation between them, its performance may not be
1005:  1005	as good as the PK sampler for tasks such as classification and detection. Furthermore, the network model requires a significant amount
1006-  1006	of computational resources and storage space when processing data,
1007:  1007	and there is still considerable room for improvement in terms of time
1008-  1008	complexity. Moreover, the mainstream training samples are limited
1009-  1009	by the angles from which they were captured. Moving forward, it is
1010-  1010	essential to consider how to construct models that are more adaptable
1011-  1011	to generalized scenarios, in order to apply them to a wider range of
1012-  1012	practical situations.
1013-  1013	
1014:  1014	4.7. Visualization of CGS sampling
1015:  1015	Fig. 9 shows some examples of the nearest neighbor classes generated by the CGS sampler. It can be seen that the CGS sampler does find
1016-  1016	training examples with some similarity for the feature extraction network. The method is able to identify similar clothing types, colors, and
1017-  1017	other similar local similarity features, and the learning of these associative instances contributes significantly to the discriminative learning
1018-  1018	ability of the model.
1019-  1019	5. Conclusion
1020:  1020	The randomness in the popular PK sampler is not conducive to
1021-  1021	learning discriminative features in person re-identification models. To
1022-  1022	address this, we propose a new batch sampler called the Correlation
1023:  1023	Graph Sampler (CGS), which constructs nearest neighbor graphs of
1024-  1024	all classes to sample informative and challenging examples for model
1025-  1025	training. This approach helps the network learn discriminative models
1026:  1026	with better performance, improving the robustness and generalization potential of the model. Additionally, we propose the GSANet
1027-  1027	for person re-identification, which reduces semantic information loss
1028-  1028	by preserving high-resolution streams and achieves pixel-level feature
1029:  1029	autocorrelation reconstruction within the feature layer using the GRSR
1030-  1030	module. This approach can fully exploit the global relevance of the
1031:  1031	feature layer and influence the perceptual field of the training network to process information. GSANet shows good feature extraction
1032:  1032	capability, which is crucial for the sampling effect of the CGS sampler
1033-  1033	since the quality of the extracted feature maps affects the quality of the
1034-  1034	constructed nearest neighbor graph. However, we also found that the
1035-  1035	
1036-  1036	CRediT authorship contribution statement
1037:  1037	Feng Min: Writing – original draft, Visualization, Validation,
1038-  1038	Methodology, Funding acquisition. Yuhui Liu: Writing – review
1039-  1039	& editing, Validation, Methodology. Yixin Mao: Writing – review
1040:  1040	& editing, Writing – original draft, Visualization, Validation,
1041-  1041	Conceptualization.
1042-  1042	
1043-  1043	Declaration of competing interest
--
1060-  1060	[9] F. Schroff, D. Kalenichenko, J. Philbin, Facenet: A unified embedding for face
1061-  1061	recognition and clustering, in: Proc. IEEE Conf. Comput. Vis. Pattern Recognit.,
1062-  1062	CVPR, 2015, pp. 815–823.
1063:  1063	[10] C. Wang, X. Zhang, X. Lan, How to train triplet networks with 100k identities? in:
1064-  1064	Proc. IEEE Int. Conf. Comput. Vis. Workshops, ICCVW, 2017, pp. 1907–1915.
1065:  1065	[11] S. Liao, L. Shao, Graph sampling based deep metric learning for generalizable
1066-  1066	person re-identification, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.,
1067-  1067	CVPR, 2022, pp. 7359–7368.
1068-  1068	[12] Y. Wang, Z. Chen, F. Wu, G. Wang, Person re-identification with cascaded
--
1095-  1095	[22] Y. Zhao, Z. Zhong, F. Yang, Z. Luo, Y. Lin, S. Li, N. Sebe, Learning to generalize
1096-  1096	unseen domains via memory-based multi-source meta-learning for person reidentification, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., CVPR,
1097-  1097	2021, pp. 6277–6286.
1098:  1098	[23] Y. Mei, Y. Fan, Y. Zhou, Image super-resolution with non-local sparse attention,
1099-  1099	in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., CVPR, 2021, pp.
1100-  1100	3517–3526.
1101-  1101	[24] L. Qi, J. Liu, L. Wang, Y. Shi, X. Geng, Unsupervised generalizable multi-source
--
1239-  1239	[47] H. Oh Song, Y. Xiang, S. Jegelka, S. Savarese, Deep metric learning via
1240-  1240	lifted structured feature embedding, in: Proc. IEEE Conf. Comput. Vis. Pattern
1241-  1241	Recognit., CVPR, 2016, pp. 4004–4012.
1242:  1242	[48] A. Hermans, L. Beyer, B. Leibe, In defense of the triplet loss for person
1243-  1243	re-identification, 2017, arXiv preprint arXiv:1703.07737.
1244-  1244	[49] X. Ning, K. Gong, W. Li, L. Zhang, X. Bai, S. Tian, Feature refinement and filter
1245-  1245	network for person re-identification, IEEE Trans. Circuits Syst. Video Technol.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | sed -n '48,167p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    48	1. Introduction
    49	Person re-identiﬁcation (Re-ID) [1,2] aims to recognize and retrieve
    50	speciﬁc individuals from intelligent surveillance systems. With the continuous advancement and development of deep learning, Re-ID technology has achieved signiﬁcant breakthroughs and is widely applied
    51	in various surveillance systems. However, research [3,4] has indicated
    52	that deep neural networks (DNNs) are highly vulnerable to adversarial examples. These adversarial examples can mislead DNN models by
    53	adding imperceptible perturbations to normal images. This vulnerability
    54	of DNN models seriously threatens the security and reliability of Re-ID
    55	systems. Therefore, it is extremely important to conduct comprehensive
    56	research on adversarial attacks against Re-ID systems.
    57	Existing adversarial attack techniques [3,5] have been predominantly developed for image classiﬁcation tasks, and these methods have
    58	achieved remarkable attack performance. However, the methods related
    59	to classiﬁcation attacks are not suitable for attacking Re-ID systems. This
    60	is primarily because the Re-ID task is an open-set task. In this task setting, there are diﬀerences in the identity information contained in the
    61	training set and the test set, whereas the query set and the gallery set
    62	share identity information.
    63	
    64	Recent studies [6–8] have shown that Re-ID models are vulnerable to
    65	adversarial example attacks, and numerous white-box [6–8] adversarial
    66	attack methods have been proposed, which assume that all parameter
    67	information of the model is known. Nevertheless, these attack methods
    68	have limitations in real-world scenarios. When the parameters of the
    69	target Re-ID model are not accessible, researchers have shifted their focus to exploring transferable adversarial examples for black-box Re-ID
    70	models. Unlike transfer-based black-box attacks in classiﬁcation tasks,
    71	Re-ID is a retrieval task with a more complex scenario. This complexity
    72	leads to insuﬃcient transferability of the generated adversarial examples, making it diﬃcult to eﬀectively test the robustness of real-world
    73	Re-ID models.
    74	In order to successfully attack the Re-ID system, numerous adversarial attack strategies have been put forward for generating adversarial person examples. These strategies include metric-based attack methods [6–9], pseudo label-based attack methods [8,10], color-based attack
    75	methods [7], and universal perturbation-based attack methods [11,12].
    76	Among the above methods, the metric-based method is the most eﬀective and widely concerned attack method, which utilizes a reference
    77	feature to distort the distance between the targeted person image and
    78	other similar person images. However, the existing work [6–8] ignores
    79	
    80	∗ Corresponding author.
    81	
    82	E-mail addresses: zeze@hbu.edu.cn (Z. Tao), lihui15794@hbu.edu.cn (H. Li), pengjinjia@hbu.edu.cn (J. Peng), huibing.wang@dlmu.edu.cn (H. Wang).
    83	https://doi.org/10.1016/j.patcog.2025.112760
    84	Received 7 May 2025; Received in revised form 6 October 2025; Accepted 16 November 2025
    85	Available online 20 November 2025
    86	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    87	
    88	Pattern Recognition 172 (2026) 112760
    89	
    90	Z. Tao et al.
    91	
    92	paradigm. At present, the practical deployment of Re-ID technology encounters a multitude of challenges. These encompass issues such as partial or complete occlusion of pedestrians, variations in viewpoints and
    93	lighting conditions, the presence of highly similar appearances among
    94	diﬀerent pedestrians, and the inherent diﬃculties associated with data
    95	acquisition in real-world scenarios. In recent years, the advent of deep
    96	learning has catalyzed a paradigm shift in Re-ID research, ushering
    97	in revolutionary advancements. Researchers have diligently explored a
    98	range of methodologies, including feature learning [13], metric learning
    99	[14], and ranking optimization techniques [15], with the aim of elevating the performance of Re-ID systems. During the training phase, classiﬁcation loss [16] or triplet loss [17] is employed to optimize the neural
   100	network, thereby enhancing the discriminative power of features. In the
   101	inference phase, the similarity between the query image and the gallery
   102	images is calculated using cosine distance or Euclidean distance. These
   103	endeavors have culminated in notable progress and signiﬁcant breakthroughs, as evidenced by their successful application across multiple
   104	publicly available datasets.
   105	
   106	Fig. 1. Visualization of the gradient update at the 𝑡-th iteration, where 𝑔𝑡 represents the current gradient, 𝑔𝑡𝐴𝑔𝑔 denotes the global aggregated gradient, 𝑔𝑡𝐹 𝑖𝑛𝑎𝑙
   107	stands for the ﬁnal gradient, and 𝑐𝑡 indicates the gradient consistency between
   108	𝑔𝑡 and 𝑔𝑡𝐴𝑔𝑔 .
   109	
   110	2.2. Adversarial attacks
   111	Adversarial attacks are primarily classiﬁed into two major categories
   112	based on their attack methodologies: white-box attacks [4] and blackbox attacks [5,18,19]. In white-box attacks, attackers have the ability
   113	to acquire the internal architecture and parameter details of the victim
   114	model. Conversely, in black-box attacks, attackers are unable to gain access to the speciﬁc particulars of the victim model. In real-world attack
   115	scenarios, it is often the case that attackers encounter signiﬁcant diﬃculties in obtaining the detailed information of the target Re-ID model.
   116	Consequently, they are compelled to conduct attacks in black-box settings. An eﬀective approach in such situations is to utilize the adversarial
   117	samples generated by surrogate models to launch attacks against other
   118	black-box models. This characteristic is vividly described as the transferability of adversarial samples. Nevertheless, while existing adversarial attacks demonstrate outstanding performance in white-box attacks,
   119	their transferability is considerably low when it comes to attacking other
   120	black-box models.
   121	Adversarial attacks are primarily employed in the ﬁeld of image classiﬁcation, where imperceptible perturbations are introduced to genuine
   122	images to deceive trained models, thereby facilitating robustness evaluation. Since Szegedy et al. [20] ﬁrst revealed the vulnerability of deep
   123	neural networks (DNNs) to adversarial examples, researchers have developed various attack methodologies, including transfer-based attacks
   124	[5,19,21], score-based attacks [22,23], and decision-based attacks [24].
   125	Among these, transfer-based attacks have gained particular prominence
   126	in real-world scenarios due to their black-box nature, requiring no access
   127	to the target model’s internal information. Recent years have witnessed
   128	signiﬁcant advancements in enhancing the transferability of adversarial
   129	attacks, with most approaches building upon the Iterative Fast Gradient Sign Method (IFGSM) [4] framework. For instance, the MIFGSM [3]
   130	method integrates momentum into IFGSM to produce more transferable
   131	adversarial examples; the VMI [18] approach further improves transferability by minimizing gradient variance; and GRA [21] enhances performance through gradient alignment with neighborhood gradients. Meanwhile, BSR [19] disrupts attention heatmaps to boost attack eﬀectiveness. Recent works [5] has also demonstrated that placing adversarial
   132	examples in ﬂat regions of the loss landscape can signiﬁcantly improve
   133	their transferability. Additionally, recent work [25,26] has also been
   134	dedicated to improving the transferability of attacks on vision-language
   135	pre-training models.
   136	These methods collectively operate by perturbing model outputs at
   137	the logit level to maximize the deviation of predicted classes from their
   138	ground-truth categories. Consequently, such approaches are only applicable to classiﬁcation tasks and cannot be directly applied to open-set
   139	ranking task like Re-ID.
   140	
   141	the consistency between the current gradient and the global aggregated
   142	gradient and fails to make full use of the information of the global aggregated gradient. As depicted in Fig. 3, prior attack methods [7,8] exhibit
   143	low consistency between the current gradient and the global aggregated
   144	gradient during the initial attack phase, which undermines the eﬃcacy
   145	of attacking Re-ID systems.
   146	In this work, we propose a Global Aggregated Gradient-guided Attack (GAGA) method to address the low gradient consistency issue during the attack process. Unlike recent research that only considers the
   147	current gradient information during each update process, our work further incorporates the information of globally aggregated gradients. As
   148	depicted in Fig. 1, prior to each iteration, this paper initially executes
   149	an internal loop (that is, performs the gradient pre-convergence operation), and then takes the average of all the gradients obtained in this
   150	internal loop as the global aggregated gradient. Furthermore, this work
   151	innovatively establishes a gradient consistency factor to extract latent
   152	information from the globally aggregated gradient. This factor serves as
   153	an eﬀective metric for quantifying the correlation between the current
   154	gradient and the global aggregated gradient. In each iteration, GAGA
   155	adaptively determines the update direction based on variations in the
   156	gradient consistency factor.
   157	To summarize, the principal contributions are presented as follows:
   158	•
   159	
   160	To the best of our knowledge, this work is the ﬁrst to reveal that
   161	low gradient consistency can limit the attack performance on Re-ID
   162	systems.
   163	• To enhance gradient consistency, this paper proposes a novel adversarial attack method for Re-ID systems, termed Global Aggregated
   164	Gradient Attack (GAGA), which is capable of generating highly transferable adversarial person images.
   165	• Compared with the state-of-the-art methods for attacking Re-ID systems, the proposed GAGA exhibits the optimal attack performance.
   166	Moreover, it can be combined with input transformation-based attack techniques to further improve transferability.
   167	2. Related work

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | sed -n '220,570p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   220	3. Methodology
   221	3.1. Problem deﬁnition
   222	
   223	𝑥inner
   224	= 𝑥𝑎𝑑𝑣
   225	+ 𝛿𝑖inner + 𝑟𝑖 ,
   226	𝑖
   227	𝑡
   228	
   229	In Re-ID systems, both query and gallery images are potential targets for adversarial attacks. However, in real-world scenarios, accessing gallery images is often challenging and resource-intensive, whereas
   230	query images can be easily manipulated–either directly or during capture. Given this asymmetry, our work focuses on generating adversarial
   231	queries to subvert Re-ID systems eﬀectively.
   232	Given a clean query image 𝑥 and a true match gallery image 𝐺, let
   233	𝑓 (𝑥) denote the embedding feature extracted from 𝑥 by the victim Re-ID
   234	model. In an ideal Re-ID system, the cosine similarity score between the
   235	features 𝑓 (𝑥) and 𝑓 (𝐺) should be as high as possible. To attack such a
   236	system, the objective of adversarial attacks is to introduce an imperceptible perturbation 𝜖 to the clean query image 𝑥, generating an adversarial sample 𝑥adv = 𝑥 + 𝜖, such that the cosine similarity score between
   237	the adversarial sample’s feature 𝑓 (𝑥adv ) and 𝑓 (𝐺) is as low as possible.
   238	Since the cosine similarity score between the original query feature 𝑓 (𝑥)
   239	and its opposite direction −𝑓 (𝑥) is the lowest, during the optimization
   240	process, we push the feature of the adversarial sample 𝑓 (𝑥adv ) towards
   241	the opposite direction of the initial sample’s feature −𝑓 (𝑥), and the optimized distance loss can be formulated as:
   242	(
   243	)
   244	‖ ( 𝑎𝑑𝑣 )
   245	‖2
   246	𝐽 𝑥𝑎𝑑𝑣 , 𝑥 =
   247	min
   248	− (−𝑓 (𝑥))‖
   249	‖𝑓 𝑥
   250	‖2
   251	‖𝑥𝑎𝑑𝑣 −𝑥‖ ≤𝜖 ‖
   252	‖
   253	‖∞
   254	(1)
   255	(
   256	)
   257	2
   258	‖
   259	‖
   260	=
   261	min
   262	‖𝑓 𝑥𝑎𝑑𝑣 + 𝑓 (𝑥)‖ ,
   263	‖2
   264	‖𝑥𝑎𝑑𝑣 −𝑥‖ ≤𝜖 ‖
   265	‖
   266	‖∞
   267	
   268	(2)
   269	
   270	where 𝛿𝑖inner is the inner adversarial perturbation at the 𝑖-th iteration,
   271	where 𝑖 = 1, 2, … , 𝑚, and 𝑚 denotes the number of iterations in the inner
   272	loop. The initial condition is 𝛿1inner = 0. Additionally, 𝑟𝑖 is the random
   273	[
   274	]
   275	[
   276	]
   277	noise satisfying 𝑟𝑖 ∼ 𝑈 −(𝛽 ⋅ 𝜖)𝑑 , (𝛽 ⋅ 𝜖)𝑑 , where 𝑈 𝑎𝑑 , 𝑏𝑑 represents the
   278	uniform distribution, and 𝑑 is the dimension.
   279	inner of the internal prediction point
   280	Next, the current gradient 𝑔𝑖+1
   281	inner
   282	𝑥𝑖
   283	is computed, and its expression is given as follows:
   284	(
   285	)
   286	inner
   287	𝑔𝑖
   288	= ∇𝑥inner 𝐽 𝑥inner
   289	,𝑥 .
   290	(3)
   291	𝑖
   292	𝑖
   293	
   294	inner by utilizSubsequently, we update the internal perturbation 𝛿𝑖+1
   295	inner
   296	ing the current internal gradient 𝑔𝑖+1 , and the formula is deﬁned as
   297	follows:
   298	(
   299	(
   300	))
   301	inner
   302	𝛿𝑖+1
   303	= Clip𝜖𝛿 𝛿𝑖inner + 𝛼 ⋅ sign 𝑔𝑖inner ,
   304	(4)
   305	
   306	where 𝛼 denotes the step size, and the Clip(⋅) operation restricts the
   307	perturbation amplitude under the 𝑙∞ norm.
   308	Finally, by averaging all gradients within the internal loop, the global
   309	Agg
   310	aggregated gradient 𝑔𝑡+1 is obtained, and its speciﬁc calculation formula
   311	is given as follows:
   312	Agg
   313	
   314	𝑔𝑡+1 =
   315	
   316	where 𝐽 is the loss function, ‖ ⋅ ‖∞ represents the ∞-norm, and 𝜖 is the
   317	perturbation bound.
   318	In real-world adversarial attacks, attackers typically do not have direct access to the victim model. To circumvent this limitation, a surrogate model is employed to craft adversarial queries, which are subsequently transferred to the victim model. Consequently, the transferability of adversarial queries emerges as a pivotal factor determining the
   319	success of such attacks, constituting the central focus of this study.
   320	
   321	𝑚
   322	𝑚
   323	(
   324	)
   325	1 ∑ inner
   326	1 ∑
   327	𝑔𝑖
   328	=
   329	∇ inner 𝐽 𝑥inner
   330	,𝑥 .
   331	𝑖
   332	𝑚 𝑖=1
   333	𝑚 𝑖=1 𝑥𝑖
   334	
   335	(5)
   336	
   337	By means of the pre-convergence attack mechanism, we are able
   338	to comprehensively capture the gradient information of all data points
   339	along the pre-convergence path, thereby providing a global guiding direction for the optimization process.
   340	3.2.2. Gradient consistency
   341	Existing methods [6–8] can eﬀectively generate adversarial samples
   342	that deceive Re-ID systems by leveraging gradient information. However, this paper reveals a signiﬁcant problem in the initial attack stage
   343	(Fig. 3), from an experimental perspective: the current gradient direction shows nearly complete inconsistency with the global aggregated gradient direction. This inconsistency causes severe oscillations
   344	in the update direction during early iterations, indicating substantial
   345	
   346	3.2. Overall framework
   347	This paper proposes a novel Global Aggregated Gradient Attack
   348	method (GAGA), which aims to stabilize the optimization process by introducing global gradient information, thereby signiﬁcantly enhancing
   349	3
   350	
   351	Pattern Recognition 172 (2026) 112760
   352	
   353	Z. Tao et al.
   354	
   355	Fig. 2. Framework of the proposed GAGA method for attacking Re-ID systems. The adversarial query generation is shown on the right, with the retrieval procedure
   356	presented on the left.
   357	
   358	This work aims to improve the consistency between the current gradient and the globally aggregated gradient during the initial attack
   359	phase. We therefore adjust the current gradient via globally aggregated
   360	gradient information in a new manner, yielding the weighted gradient:
   361	Agg
   362	
   363	𝑤
   364	𝑔𝑡+1
   365	= 𝑐𝑡+1 ⋅ 𝑔̂𝑡+1 + (1 − 𝑐𝑡+1 ) ⋅ 𝑔𝑡+1 .
   366	
   367	(7)
   368	
   369	The core idea of the weighted gradient is to dynamically adjust the
   370	weights between the current gradient and the globally aggregated gradient through the gradient consistency factor. The globally aggregated
   371	Agg
   372	gradient 𝑔𝑡+1 combines gradient information from multiple future points
   373	and can be viewed as an auxiliary correction term. When the current graAgg
   374	dient 𝑔̂𝑡+1 exhibits high consistency with 𝑔𝑡+1 , we assign a larger weight
   375	Agg
   376	
   377	to 𝑔̂𝑡+1 and a smaller weight to 𝑔𝑡+1 , as 𝑔̂𝑡+1 requires minimal correction
   378	in this scenario. Conversely, if their consistency is low, we tend to asAgg
   379	sign a larger weight to 𝑔𝑡+1 , placing greater trust in it, since it represents
   380	an average of multiple predicted points along the pre-convergence path
   381	rather than being based on a single input.
   382	Next, the momentum term 𝑔𝑡+1 is updated via the weighted gradient
   383	𝑤 :
   384	𝑔𝑡+1
   385	
   386	Fig. 3. Gradient consistency between the current gradient and the global aggregated gradient. Previous methods exhibit signiﬁcantly low consistency during
   387	initial iterations.
   388	
   389	𝑤
   390	𝑔𝑡+1
   391	,
   392	(8)
   393	‖ 𝑤 ‖
   394	‖𝑔𝑡+1 ‖
   395	‖
   396	‖1
   397	where 𝜇 denotes the decay factor for momentum accumulation.
   398	Finally, the adversarial queries 𝑥𝑎𝑑𝑣
   399	are updated iteratively as fol𝑡+1
   400	lows:
   401	{
   402	(
   403	)}
   404	𝑥𝑎𝑑𝑣
   405	= Clip 𝑥𝑎𝑑𝑣
   406	+ 𝛼 ⋅ sign 𝑔𝑡+1 ,
   407	(9)
   408	𝑡
   409	𝑡+1
   410	
   411	𝑔𝑡+1 = 𝜇 ⋅ 𝑔𝑡 +
   412	randomness in the initial attack direction. We identify both the low gradient consistency and high randomness during the attack process as key
   413	factors contributing to poor transferability against Re-ID systems.
   414	To address the aforementioned issues, this paper proposes a gradient consistency factor, which quantitatively measures the similarity between the current gradient and the globally aggregated gradient during
   415	the attack process. At the t-th iteration, the gradient consistency factor
   416	𝑐𝑡+1 can be expressed as:
   417	Agg
   418	⎛
   419	⎞
   420	𝑔̂𝑡+1 ⋅ 𝑔𝑡+1
   421	⎟,
   422	𝑐𝑡+1 = max ⎜0,
   423	Agg ‖
   424	⎜ ‖𝑔̂ ‖ ⋅ ‖
   425	𝑔𝑡+1 ‖ ⎟⎠
   426	⎝ ‖ 𝑡+1 ‖2 ‖
   427	‖
   428	‖2
   429	
   430	where 𝐶𝑙𝑖𝑝(⋅) denotes the pixel-wise projection operation that conﬁnes
   431	image values to a predeﬁned constraint range, and 𝛼 represents the ﬁxed
   432	step size.
   433	The primary objective of our work is to enhance gradient consistency
   434	during the initial iterations, preventing violent ﬂuctuations in update directions during the early attack phase. This represents a key distinction
   435	between our approach and previous methods. The Global Aggregated
   436	Gradient Attack (GAGA) procedure is summarized in Algorithm 1.
   437	Why does GAGA enhance gradient consistency during early attack stages, and how does this improved consistency boost adversarial transferability?
   438	
   439	(6)
   440	
   441	where, 𝑐𝑡+1 ∈ [0, 1] measures the directional consistency between gradients. A higher 𝑐𝑡+1 indicates stronger alignment between the current
   442	(
   443	)
   444	gradient and the globally aggregated gradient. 𝑔̂𝑡+1 = ∇𝑥𝑎𝑑𝑣 𝐽 𝑥𝑎𝑑𝑣
   445	𝑡 , 𝑥 deAgg
   446	
   447	𝑡
   448	
   449	notes the current gradient, while 𝑔𝑡+1 is the aggregated gradient derived
   450	from Eq. 5.
   451	4
   452	
   453	Pattern Recognition 172 (2026) 112760
   454	
   455	Z. Tao et al.
   456	
   457	ditional 3368 query images for evaluation. In contrast, MSMT17 dataset
   458	[35] is recognized for its large-scale and diverse nature, with 126,441
   459	images captured by 15 cameras. The training set consists of 32,621 images from 1041 identities, while the test set comprises 11,659 probe
   460	images (from 3060 identities) and 82,161 gallery images. Owing to its
   461	unprecedented scale and real-world variability, MSMT17 is regarded as
   462	one of the most challenging Re-ID benchmarks to date.
   463	Evaluation Metric. Rank-K and mean Average Precision (mAP) are
   464	two standard evaluation metrics in person Re-ID. Rank-K measures the
   465	probability that the true match appears within the top-K retrieved candidates, reﬂecting the system’s retrieval accuracy at a given rank threshold
   466	(e.g., Rank-1, Rank-5). The mAP quantiﬁes the overall retrieval performance by computing the mean of Average Precision (AP) scores across
   467	all query instances, where AP accounts for both precision and recall at
   468	each rank position. In attacks on Re-ID tasks, lower Rank-K and mAP
   469	values indicate better attack performance.
   470	Re-ID Models. This study trains three baseline backbone networks
   471	and three state-of-the-art backbone networks without employing any
   472	performance-enhancing techniques. The baseline models use ResNet50
   473	(Re-50) [36], DenseNet121 (De-121) [37], and HrNet18 (Hr-18) [38]
   474	as backbone networks. As there are many advanced Re-ID backbone
   475	networks available, it is not practical to evaluate all of them. Therefore,
   476	this paper focuses on replicating three representative models that have
   477	achieved the latest state-of-the-art performance: ConvNext (Conv) [39],
   478	Swin-Transformer (Swin) [40], and Swinv2-Transformer (Swinv2) [41].
   479	These models are trained using cross-entropy loss and optimized with
   480	Adam for 60 epochs, with a batch size of 64.
   481	Baselines. To evaluate the eﬀectiveness of the proposed method,
   482	this paper compares several latest attack methods for Re-ID, and three
   483	advanced transfer-based adversarial attacks, including LTA [7], ODFA
   484	[8], MI [6], IAAR [42], GRA [21], TPA [5], and BSR [19]. Additionally, to further validate the eﬀectiveness of the proposed method, this
   485	paper integrates GAGA with various input transformations-based attack
   486	methods, including SI [43], TI [44], Admix [45], and SSA [46].
   487	Attack Settings. This paper conducts extensive experimental comparisons under various attack settings, including diﬀerent types of attack
   488	settings such as white-box settings, black-box settings, and ensemblemodel settings. In this paper, the value of the adversarial perturbation
   489	is set to 𝜖 = 16∕255, and the number of iterations 𝑇 = 10. Additionally,
   490	the step size is set to 𝛼 = 𝜖∕𝑇 , and the decay factor 𝜇 is set to 1. For MI
   491	[6], ODFA [8], LTA [7], GRA [21], TPA [5] and BSR [19], this paper
   492	follows the oﬃcial settings described in the corresponding papers. In
   493	GAGA, the inner iteration number 𝑚 is set to 5, and the noise boundary
   494	𝛽 is set to 1.
   495	
   496	Algorithm 1 Global Aggregated Gradient Attack (GAGA).
   497	Input: A surrogate Re-ID model 𝑓 ; a query image 𝑥 and loss function 𝐽
   498	Input: The magnitude of perturbation 𝜀; the step size 𝛼; the number of
   499	iteration 𝑇 ; the decay factor 𝜇; the noise boundary 𝛽 and the number
   500	of inner iterations 𝑚
   501	Output: An adversarial query 𝑥𝑎𝑑𝑣
   502	1: 𝑔0 = 0; 𝑥𝑎𝑑𝑣
   503	=𝑥
   504	0
   505	2: for 𝑡 = 0 → 𝑇 − 1 do
   506	(
   507	)
   508	3:
   509	Calculate the gradient 𝑔̂𝑡+1 = ∇𝑥𝑎𝑑𝑣 𝐽 𝑥𝑎𝑑𝑣
   510	𝑡 ,𝑥
   511	𝑡
   512	
   513	4:
   514	5:
   515	6:
   516	7:
   517	
   518	𝑔𝑡+1 = 𝜇 ⋅ 𝑔𝑡 +
   519	8:
   520	
   521	Agg
   522	
   523	Calculate the global aggregated gradient 𝑔𝑡+1 by Eq. (5)
   524	Update the gradient consistency factor 𝑐𝑡+1 by Eq. (6)
   525	𝑤 by Eq. (7)
   526	Calculate the weighted gradient 𝑔𝑡+1
   527	𝑤
   528	Update the momentum accumulation 𝑔𝑡+1 with 𝑔𝑡+1
   529	𝑤
   530	𝑔𝑡+1
   531	‖ 𝑤 ‖
   532	‖𝑔𝑡+1 ‖
   533	‖
   534	‖1
   535	
   536	(10)
   537	
   538	Update adversarial query 𝑥𝑎𝑑𝑣
   539	by applying the sign of gradient
   540	𝑡+1
   541	{ 𝑎𝑑𝑣
   542	(
   543	)}
   544	𝑎𝑑𝑣
   545	𝑥𝑡+1 = Clip 𝑥𝑡 + 𝛼 ⋅ sign 𝑔𝑡+1 ,
   546	(11)
   547	
   548	9: end for
   549	10: 𝑥𝑎𝑑𝑣 = 𝑥𝑎𝑑𝑣
   550	𝑇
   551	11: return 𝑥𝑎𝑑𝑣
   552	
   553	As shown in Fig. 3, previous methods [7,8] exhibit low consistency
   554	between the current gradient and the global aggregated gradient during the initial attack phase. Only after several rounds of forward gradient convergence does the gradient consistency signiﬁcantly improve
   555	and stabilize. Consequently, the update direction in the initial phase of
   556	the attack exhibits high randomness, leading to instability in the update
   557	direction and, thus, limited transferability. To tackle this issue, we utilize the information from the globally aggregated gradient to stabilize
   558	the update direction of the current gradient, thereby enhancing transferability. Our study aims to accelerate the convergence process, eﬀectively mitigating the instability of attack directions in the initial phase.
   559	We will separately validate the superiority of the GAGA method in both
   560	the early and late stages of the attack.
   561	In the initial stage, the direction of the current gradient diﬀers significantly from that of the global aggregated gradient, resulting in a small
   562	gradient consistency factor 𝑐𝑡+1 value. According to the weight allocation mechanism in Eq. 7, the current gradient is assigned a low weight
   563	coeﬃcient, while the global aggregated gradient dominates the update.
   564	Consequently, the GAGA method guides the current gradient to align
   565	with the global aggregated gradient during early iterations, rapidly improving gradient consistency and achieving early convergence.
   566	In the later stage, the current gradient and the global aggregated gradient achieve high consistency, leading to a large gradient consistency
   567	factor 𝑐𝑡+1 value. Under these conditions, the weight of the global aggregated gradient automatically decreases, while the current gradient
   568	gains a larger update weight. Since the gradient direction has stabilized
   569	by this stage, the system primarily relies on the current gradient for
   570	ﬁne-tuning, ensuring stable convergence in later iterations.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdfinfo 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' | sed -n '1,35p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Global aggregated gradient-guided adversarial attacks for person re-identification
Subject:         Pattern Recognition, 172 (2026) 112760. doi:10.1016/j.patcog.2025.112760
Keywords:        Person re-identification,Adversarial examples,Transferability,Global aggregated gradient,Gradient consistency
Author:          Zeze Tao
Creator:         Elsevier
Producer:        Acrobat Distiller 8.1.0 (Windows)
CreationDate:    Mon Dec  8 19:42:44 2025 CST
ModDate:         Mon Dec  8 19:43:03 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           12
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       5759992 bytes
Optimized:       no
PDF version:     1.7

codex
第六篇虽然是攻击论文，但它的创新构造很值得学：不是泛泛说“黑盒迁移差”，而是定义了一个可测的中间变量“梯度一致性”，再让算法直接优化这个中间变量。下面补它的验证部分。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|gradient consistency|consistency|global aggregated|GAGA|Table|Fig|iterations|early|late|transfer|black-box|white-box|Visualization|parameter|inner|m |beta|Input transformation|ensemble\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2-     2	
3-     3	Z. Tao et al.
4-     4	
5:     5	Table 7
6-     6	Comparison of Rank-1 (%) and mAP (%) with the state-of-the-art methods on SYSU-MM01 and
7-     7	RegDB datasets. Best attacks are marked in bold.
8-     8	SYSU-MM01
--
18-    18	BSR [19]
19-    19	TPA [5]
20-    20	IAAR [42]
21:    21	GAGA (Ours)
22-    22	
23-    23	RegDB
24-    24	Indoor-search
--
123-   123	14.30
124-   124	9.25
125-   125	
126:   126	Table 8
127-   127	Attack Performance (%) and Runtime Analysis. We choose Re-50 as the surrogate model and evaluate the runtime
128:   128	on an NVIDIA RTX3090, * indicates the white-box model.
129-   129	Method
130-   130	
131-   131	MI [6]
--
134-   134	GRA [21]
135-   135	BSR [19]
136-   136	TPA [5]
137:   137	GAGA (Ours)
138-   138	
139-   139	Market-1501
140-   140	
--
240-   240	1589.36
241-   241	967.38
242-   242	
243:   243	two key ﬁndings: First, black-box attack performance is notably weaker
244:   244	than white-box attacks. This phenomenon occurs because architectural
245-   245	diﬀerences between models make adversarial samples generated for one
246-   246	model less eﬀective against others. Second, there are signiﬁcant robustness variations among diﬀerent proxy models. For example, when using
247:   247	GAGA to attack the Swinv2 model on the MSMT17 dataset, the Rank-1
248-   248	scores diﬀer by 5.14 % between proxy models Re-50 and Hr-18, while
249-   249	the mAP performance gap reaches 4.80 %.
250:   250	To further validate the eﬀectiveness of the GAGA method, we present
251:   251	both black-box and white-box results for all models in Table 9 on the
252:   252	Market-1501 dataset. The experimental results demonstrate that GAGA
253-   253	achieves the best attack performance.
254-   254	
255-   255	input transformation techniques. To ensure the reliability of experimental results, we adopt a rigorous comparative experimental design: ﬁrst
256:   256	incorporating all input transformation strategies into the ODFA [8] baseline method, then conducting systematic testing under standard blackbox attack scenarios. As shown in Table 4, the experimental results
257:   257	demonstrate that the GAGA method achieves outstanding performance
258:   258	improvement when combined with diﬀerent input transformation techniques. Notably, when GAGA is integrated with SSA [46], the Rank1 recognition rate dramatically drops from 30.92 % to 14.64 %. This
259-   259	breakthrough achievement not only signiﬁcantly surpasses the baseline
260-   260	performance but also strongly conﬁrms the crucial role of input diversity
261:   261	strategies in enhancing the transferability of adversarial examples.
262-   262	
263:   263	4.2.3. Attacking diﬀerent re-ID models in ensemble-model setting
264:   264	Lin et al. [43] demonstrated that employing a multi-model ensemble strategy for generating adversarial examples can signiﬁcantly improve the transfer success rate of attacks. Therefore, this paper conducts
265:   265	comparative experiments under the ensemble-model setting, using three
266:   266	standard trained models (Re-50, De-121, and Hr-18) as the ensemble.
267:   267	The corresponding experimental results are recorded in Table 3. The results show that compared with state-of-the-art attack methods, our proposed GAGA method achieves the most promising attack performance in
268:   268	the ensemble-model setting. Speciﬁcally, when adversarial queries are
269:   269	generated based on the ensemble of Re-50, De-121, and Hr-18, GAGA
270-   270	signiﬁcantly outperforms ODFA [8] with substantially improved attack
271:   271	eﬀectiveness. This ﬁnding further conﬁrms that the low consistency of
272:   272	gradients limits the transferability of attacks.
273-   273	
274-   274	4.4. Attack defense models
275:   275	To evaluate the eﬀectiveness of the proposed GAGA method, this paper analyzes the attack performance of GAGA against multiple advanced
276-   276	defense models on both Market-1501 and MSMT17 datasets. The tested
277:   277	defense mechanisms include: random resizing and padding (R&P) [47],
278-   278	feature distillation (FD) [48], bit depth reduction (Bit-Red) [49], as well
279:   279	as JPEG compression (JPEG) [50]. The experimental results are presented in Table 5. Under black-box attack scenarios, GAGA demonstrates
280:   280	signiﬁcantly superior performance compared to other state-of-the-art attack algorithms. For instance, on Market-1501 dataset, GAGA achieves
281-   281	an average Rank-1 score of 22.92 % against the four defense models. The
282-   282	second-best performing TPA [5] method only reaches 35.55 % Rank-1
283:   283	accuracy. GAGA outperforms BSR [19] by a notable margin of 17.96 %.
284:   284	This phenomenon demonstrates that the GAGA method exhibits remarkable attack eﬃcacy in bypassing various defense models, posing a substantial threat to state-of-the-art defense mechanisms.
285-   285	
286-   286	4.3. Integrated with transformation-based attacks
287:   287	Recent studies have demonstrated that gradient-based adversarial attacks can signiﬁcantly enhance transferability by incorporating diverse
288-   288	input transformation strategies. State-of-the-art methods such as SSA
289-   289	[46], SI [43], Admix [45], and TI [44] have shown remarkable eﬀectiveness in boosting attack performance. Building upon these advances, our
290:   290	work systematically integrates the proposed GAGA method with these
291-   291	
292-   292	4.5. Attacking cross-Modal re-ID models
293:   293	To further validate the eﬀectiveness of the GAGA method, we conducted experiments on attacking Cross-Modal Re-ID models. For the
294-   294	SYSU-MM01 [51] dataset, infrared images serve as the query set, and
295-   295	8
296-   296	
--
298-   298	
299-   299	Z. Tao et al.
300-   300	
301:   301	Table 9
302:   302	Performance (%) of diﬀerent attack methods on Re-ID models, *indicates the white-box model being attacked. Lower is better for the attack.
303-   303	Best attacks are marked in bold.
304-   304	Re-50
305-   305	
--
350-   350	BSR [19]
351-   351	TPA [5]
352-   352	IAAR [42]
353:   353	GAGA (Ours)
354-   354	
355-   355	3.86*
356-   356	1.28*
--
469-   469	BSR [19]
470-   470	TPA [5]
471-   471	IAAR [42]
472:   472	GAGA (Ours)
473-   473	
474-   474	42.22
475-   475	26.36
--
588-   588	BSR [19]
589-   589	TPA [5]
590-   590	IAAR [42]
591:   591	GAGA (Ours)
592-   592	
593-   593	30.25
594-   594	13.58
--
707-   707	BSR [19]
708-   708	TPA [5]
709-   709	IAAR [42]
710:   710	GAGA (Ours)
711-   711	
712-   712	25.46
713-   713	11.05
--
826-   826	BSR [19]
827-   827	TPA [5]
828-   828	IAAR [42]
829:   829	GAGA (Ours)
830-   830	
831-   831	42.16
832-   832	25.81
--
945-   945	BSR [19]
946-   946	TPA [5]
947-   947	IAAR [42]
948:   948	GAGA (Ours)
949-   949	
950-   950	40.16
951-   951	24.16
--
1057-  1057	
1058-  1058	visible light images act as the gallery set. The search modes include allsearch and indoor-search. The RegDB [52] dataset mainly focuses on the
1059-  1059	performance of cross-modal bidirectional retrieval, that is, the retrieval
1060:  1060	capabilities from the visible light modality to the infrared modality and
1061:  1061	from the infrared modality to the visible light modality. We chose the
1062:  1062	powerful baseline model AGW [1], which is a Re-50 pretrained on ImageNet, as the backbone network. The experimental results in Table 7
1063:  1063	demonstrate that our proposed GAGA method can also achieve strong
1064-  1064	attack performance on cross-modal Re-ID datasets.
1065-  1065	
1066-  1066	ferent attack methods. The adversarial examples are generated on
1067-  1067	the Re-50 model. The results indicate that ODFA has the shortest
1068-  1068	time for generating adversarial examples, but its attack success rate
1069-  1069	is very low. Secondly, TPA and PGN have relatively long runtimes.
1070:  1070	Although GAGA achieves state-of-the-art attack performance, its runtime is not the shortest. Given that the primary objective of this paper is to enhance adversarial transferability, in the future, we will investigate ways to reduce computational costs while maintaining high
1071:  1071	transferability.
1072-  1072	
1073-  1073	4.6. Eﬃciency analysis
1074-  1074	
1075:  1075	4.7. Ablation study and hyper-parameter analysis
1076-  1076	
1077:  1077	Since the GAGA method proposed in this paper requires an internal loop to obtain the globally aggregated gradient during each
1078-  1078	update, we conducted an eﬃciency analysis experiment. The results,
1079:  1079	as shown in Table 8, present the performance and runtime of dif-
1080-  1080	
1081:  1081	In this session, this paper conducts three ablation studies: (1) The
1082:  1082	hyper-parameters for the inner iteration number 𝑚; (2) The hyperparameters for the noise boundary 𝛽; (3) The inﬂuence of diﬀerent update directions on experimental results.
1083-  1083	9
1084-  1084	
1085-  1085	Pattern Recognition 172 (2026) 112760
1086-  1086	
1087-  1087	Z. Tao et al.
1088-  1088	
1089:  1089	Fig. 4. The mAP (%) on four black-box models with various hyper-parameters 𝑚 or 𝛽. The adversarial queries are generated by GAGA on Re-50. Lower is better for
1090-  1090	the attack.
1091-  1091	
1092:  1092	Fig. 5. Retrieval results of clean queries and adversarial queries generated by the proposed GAGA method.
1093-  1093	
1094:  1094	The inner iteration number. We investigate the impact of the inner iteration number on experimental results (with the noise boundary 𝛽
1095:  1095	ﬁxed at 1). As shown in Fig. 4, when 𝑚 increases from 1, the mAP values
1096-  1096	of all four models progressively decrease, indicating improved attack
1097-  1097	eﬀectiveness. At 𝑚 = 20, the models achieve their lowest mAP values,
1098-  1098	which corresponds to optimal attack performance. When 𝑚 exceeds 20,
1099-  1099	the mAP values of Conv, Swin, and Swinv2 gradually increase, suggesting degraded attack performance. However, for De-121, the mAP values
1100-  1100	stabilize. To reduce computational costs, this paper sets the number of
1101:  1101	inner iterations to 5.
1102:  1102	The noise boundary. To examine the inﬂuence of the noise boundary 𝛽 on the results, this work conducts ablation experiments to analyze
1103:  1103	this parameter (with the nner iteration number 𝑚 ﬁxed at 5). As shown
1104:  1104	in Fig. 4, when 𝛽 = 0, all four models achieve their highest mAP values, indicating the poorest attack performance. As 𝛽 increases, the mAP
1105-  1105	values of the four models gradually decrease, demonstrating progres-
1106-  1106	
1107-  1107	sively enhanced attack eﬀectiveness. At 𝛽 = 4, the mAP values reach
--
1110-  1110	capability. To ensure fair comparison with prior methods, we set 𝛽 = 1
1111-  1111	in this work.
1112-  1112	Update direction. To investigate the inﬂuence of diﬀerent update
1113:  1113	directions on the results, this study conducts ablation experiments by
1114-  1114	respectively using the current gradient (gradient), globally aggregated
1115-  1115	gradient (GAG), and the weight gradient (WG) employed in this paper
1116:  1116	as the update directions. The experimental results are shown in Table 6.
1117-  1117	We ﬁnd that when the weight gradient is used as the update direction,
1118-  1118	the attack performance is the best. In contrast, the performance is the
1119-  1119	poorest when using the current gradient as the update direction. The results further demonstrate that the weighting method enables us to strike
--
1125-  1125	
1126-  1126	Z. Tao et al.
1127-  1127	
1128:  1128	4.8. Visualization analysis
1129-  1129	
1130:  1130	[5] M. Fan, X. Li, C. Chen, W. Zhou, Y. Li, Transferability bound theory: exploring relationship between adversarial transferability and ﬂatness, in: Proceedings of the
1131-  1131	Advances in Neural Information Processing Systems, 2024.
1132-  1132	[6] S. Bai, Y. Li, Y. Zhou, Q. Li, P.H.S. Torr, Adversarial metric attack and defense for
1133-  1133	person re-identiﬁcation, in: IEEE Transactions on Pattern Analysis and Machine Intelligence, 43, IEEE, 2020, pp. 2119–2126.
--
1155-  1155	reidentiﬁcation, in: ACM Transactions on Multimedia Computing, Communications,
1156-  1156	and Applications (TOMM), 14, ACM New York, NY, USA, 2017, pp. 1–20.
1157-  1157	[17] E. Ristani, C. Tomasi, Features for multi-target multi-camera tracking and reidentiﬁcation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 6036–6046.
1158:  1158	[18] X. Wang, K. He, Enhancing the transferability of adversarial attacks through variance
1159-  1159	tuning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
1160-  1160	Recognition, 2021, pp. 1924–1933.
1161:  1161	[19] K. Wang, X. He, W. Wang, X. Wang, Boosting adversarial transferability by block
1162-  1162	shuﬄe and rotation, in: Proceedings of the IEEE/CVF International Conference on
1163-  1163	Computer Vision, 2024.
1164-  1164	[20] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I.J. Goodfellow, R. Fergus,
1165-  1165	Intriguing properties of neural networks, in: Corr, abs/1312.6199, 2014.
1166:  1166	[21] H. Zhu, Y. Ren, X. Sui, L. Yang, W. Jiang, Boosting adversarial transferability via
1167-  1167	gradient relevance attack, in: Proceedings of the IEEE/CVF International Conference
1168-  1168	on Computer Vision, 2023, pp. 4741–4750.
1169:  1169	[22] S. Cheng, Y. Dong, T. Pang, H. Su, J. Zhu, Improving black-box adversarial attacks
1170:  1170	with a transfer-based prior, Adv. Neural Inf. Process. Syst. 32 (2019).
1171:  1171	[23] Y. Li, L. Li, L. Wang, T. Zhang, B. Gong, Nattack: learning the distributions of adversarial examples for an improved black-box attack on deep neural networks, in:
1172-  1172	International Conference on Machine Learning, PMLR, 2019, pp. 3866–3876.
1173-  1173	[24] X. Wang, Z. Zhang, K. Tong, D. Gong, K. He, Z. Li, W. Liu, Triangle attack: a queryeﬃcient decision-based adversarial attack, in: European Conference on Computer
1174-  1174	Vision, Springer, 2022, pp. 156–174.
1175-  1175	[25] D. Lu, Z. Wang, T. Wang, W. Guan, H. Gao, F. Zheng, Set-level guidance attack:
1176:  1176	boosting adversarial transferability of vision-language pre-training models, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023,
1177-  1177	pp. 102–111.
1178-  1178	[26] X. Jia, S. Gao, Q. Guo, S. Qin, K. Ma, Y. Huang, Y. Liu, I. Tsang, X. Cao, SemanticAligned adversarial evolution triangle for high-Transferability vision-Language attack, IEEE Trans. Pattern Anal. Mach. Intell. (01) (2025) 1–18.
1179-  1179	[27] Z. Wang, S. Zheng, M. Song, Q. Wang, A. Rahimpour, H. Qi, Advpattern: physicalworld attacks on deep person re-identiﬁcation via adversarially transformable patterns, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 8341–8350.
1180-  1180	[28] H. Wang, G. Wang, Y. Li, D. Zhang, L. Lin, Transferable, controllable, and inconspicuous adversarial attacks on person re-identiﬁcation with deep mis-ranking, in:
1181-  1181	Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 342–351.
1182-  1182	[29] X. Wang, S. Li, M. Liu, Y. Wang, A.K. Roy-Chowdhury, Multi-expert adversarial
1183:  1183	attack detection in person re-identiﬁcation using context inconsistency, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021,
1184-  1184	pp. 15097–15107.
1185-  1185	[30] F. Yang, Z. Zhong, H. Liu, Z. Wang, Z. Luo, S. Li, N. Sebe, S. Satoh, Learning
1186-  1186	to attack real-world models for person re-identiﬁcation via virtual-guided metalearning, in: Proceedings of the AAAI Conference on Artiﬁcial Intelligence, 35, 2021,
--
1188-  1188	[31] X. Yang, D. Cheng, N. Wang, X. Gao, et al., Feature-level adversarial attacks and
1189-  1189	ranking disruption for visible-infrared person re-identiﬁcation, Adv. Neural Inf. Process. Syst. 37 (2024) 135043–135061.
1190-  1190	[32] Y. Gong, Q. Zeng, D. Xu, Z. Wang, M. Jiang, Cross-modality attack boosted by
1191:  1191	gradient-evolutionary multiform optimization, arXiv preprint arXiv:2409.17977
1192-  1192	(2024).
1193-  1193	
1194-  1194	We present the retrieval results for clean queries and adversarial
1195:  1195	queries, as shown in Fig. 5. The perturbation rate 𝜖 is ﬁxed at 8. The
1196:  1196	ﬁrst row displays the clean query, while rows 2–4 show randomly selected adversarial queries from the dataset. In the ﬁgure, green boxes
1197:  1197	indicate correct results, and red boxes mark incorrect results. From the
1198:  1198	Fig. 5, it can be observed that the retrieval performance for the original
1199-  1199	query is excellent. However, when adversarial queries are used, the top
1200:  1200	10 retrieval results are all incorrect matches, exhibiting signiﬁcant differences from the query image. This also demonstrates that adversarial
1201-  1201	queries can successfully mislead the Re-ID model, leading to irrelevant
1202-  1202	ranking results.
1203-  1203	5. Conclusion
1204:  1204	In this work, we propose a novel Global Aggregated Gradient Attack (GAGA) method for Re-ID systems to address the low gradient consistency issue during initial attack phases. Speciﬁcally, our approach
1205-  1205	performs gradient pre-convergence before each update to obtain the
1206:  1206	global aggregated gradient. Additionally, we design a gradient consistency factor based on the relationship between the global aggregated
1207:  1207	gradient and the current gradient to enhance gradient consistency. Experimental results demonstrate that GAGA outperforms state-of-the-art
1208:  1208	attack methods by a signiﬁcant margin. However, GAGA requires an inner loop before each update, resulting in higher computational costs.
1209-  1209	In future work, we will investigate approaches to reduce computational
1210-  1210	overhead while maintaining high attack eﬀectiveness.
1211-  1211	CRediT authorship contribution statement
1212-  1212	Zeze Tao: Writing – original draft, Methodology, Conceptualization;
1213:  1213	Hui Li: Visualization, Validation; Jinjia Peng: Writing – review & editing, Validation, Supervision, Project administration; Huibing Wang:
1214-  1214	Methodology, Funding acquisition.
1215-  1215	Data availability
1216-  1216	The data used in this study are publicly available.
--
1220-  1220	the work reported in this paper.
1221-  1221	Acknowledgments
1222-  1222	This work was supported by Basic Research Project of Shijiazhuang
1223:  1223	Municipal Universities in Hebei Province (241791387A); Interdisciplinary Research Program of Hebei University (DXK202404); National Natural Science Foundation of China (62002041, 62501226,
1224-  1224	62576067); Dalian Science and Technology Bureau (2022JJ12GX019).
1225-  1225	References
1226-  1226	[1] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, S.C.H. Hoi, Deep learning for person reidentiﬁcation: a survey and outlook, in: IEEE Transactions on Pattern As and Machine Intelligence, 44, IEEE, 2021, pp. 2872–2893.
--
1242-  1242	Neural Information Processing Systems, 2024.
1243-  1243	[34] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, Q. Tian, Scalable person reidentiﬁcation: a benchmark, in: Proceedings of the IEEE International Conference
1244-  1244	on Computer Vision, 2015, pp. 1116–1124.
1245:  1245	[35] L. Wei, S. Zhang, W. Gao, Q. Tian, Person transfer gan to bridge domain gap for
1246-  1246	person re-identiﬁcation, in: Proceedings of the IEEE Conference on Computer Vision
1247-  1247	and Pattern Recognition, 2018, pp. 79–88.
1248-  1248	[36] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in:
--
1267-  1267	
1268-  1268	[43] J. Lin, C. Song, K. He, L. Wang, J. Hopcroft, Nesterov accelerated gradient and scale
1269-  1269	invariance for adversarial attacks, in: arXiv: Learning, 2019.
1270:  1270	[44] Y. Dong, T. Pang, H. Su, J. Zhu, Evading defenses to transferable adversarial examples by translation-invariant attacks, in: Proceedings of the IEEE/CVF Conference
1271-  1271	on Computer Vision and Pattern Recognition, 2019, pp. 4312–4321.
1272:  1272	[45] X. Wang, X. He, J. Wang, K. He, Admix: enhancing the transferability of adversarial attacks, in: Proceedings of the IEEE/CVF International Conference on Computer
1273-  1273	Vision, 2021, pp. 16158–16167.
1274-  1274	[46] Y. Long, Q. Zhang, B. Zeng, L. Gao, X. Liu, J. Zhang, J. Song, Frequency domain
1275-  1275	model augmentation for adversarial attack, in: Computer Vision–ECCV 2022: 17th
--
1281-  1281	IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, 2019,
1282-  1282	pp. 860–868.
1283-  1283	[49] W. Xu, D. Evans, Y. Qi, Feature squeezing: detecting adversarial examples in deep
1284:  1284	neural networks, in: Proceedings 2018 Network and Distributed System Security
1285-  1285	Symposium, Internet Society, 2018.
1286-  1286	[50] C. Guo, M. Rana, M. Cisse, L. van der Maaten, Countering adversarial images using
1287-  1287	input transformations, in: International Conference on Learning Representations,
--
1289-  1289	[51] A. Wu, W.-S. Zheng, H.-X. Yu, S. Gong, J. Lai, RGB-Infrared cross-modality person
1290-  1290	re-identiﬁcation, in: Proceedings of the IEEE International Conference on Computer
1291-  1291	Vision, 2017, pp. 5380–5389.
1292:  1292	[52] D.T. Nguyen, H.G. Hong, K.W. Kim, K.R. Park, Person recognition system based on
1293:  1293	a combination of body images from visible light and thermal cameras, Sensors 17
1294-  1294	(3) (2017) 605.
1295-  1295	
1296-  1296	12

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 10 -l 10 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | sed -n '1,180p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Pattern Recognition 172 (2026) 112760
     2	
     3	Z. Tao et al.
     4	
     5	Fig. 4. The mAP (%) on four black-box models with various hyper-parameters 𝑚 or 𝛽. The adversarial queries are generated by GAGA on Re-50. Lower is better for
     6	the attack.
     7	
     8	Fig. 5. Retrieval results of clean queries and adversarial queries generated by the proposed GAGA method.
     9	
    10	The inner iteration number. We investigate the impact of the inner iteration number on experimental results (with the noise boundary 𝛽
    11	ﬁxed at 1). As shown in Fig. 4, when 𝑚 increases from 1, the mAP values
    12	of all four models progressively decrease, indicating improved attack
    13	eﬀectiveness. At 𝑚 = 20, the models achieve their lowest mAP values,
    14	which corresponds to optimal attack performance. When 𝑚 exceeds 20,
    15	the mAP values of Conv, Swin, and Swinv2 gradually increase, suggesting degraded attack performance. However, for De-121, the mAP values
    16	stabilize. To reduce computational costs, this paper sets the number of
    17	inner iterations to 5.
    18	The noise boundary. To examine the inﬂuence of the noise boundary 𝛽 on the results, this work conducts ablation experiments to analyze
    19	this parameter (with the nner iteration number 𝑚 ﬁxed at 5). As shown
    20	in Fig. 4, when 𝛽 = 0, all four models achieve their highest mAP values, indicating the poorest attack performance. As 𝛽 increases, the mAP
    21	values of the four models gradually decrease, demonstrating progres-
    22	
    23	sively enhanced attack eﬀectiveness. At 𝛽 = 4, the mAP values reach
    24	their minimum, signifying optimal attack performance. When 𝛽 exceeds
    25	4, the mAP values of all models begin to rise, reﬂecting degraded attack
    26	capability. To ensure fair comparison with prior methods, we set 𝛽 = 1
    27	in this work.
    28	Update direction. To investigate the inﬂuence of diﬀerent update
    29	directions on the results, this study conducts ablation experiments by
    30	respectively using the current gradient (gradient), globally aggregated
    31	gradient (GAG), and the weight gradient (WG) employed in this paper
    32	as the update directions. The experimental results are shown in Table 6.
    33	We ﬁnd that when the weight gradient is used as the update direction,
    34	the attack performance is the best. In contrast, the performance is the
    35	poorest when using the current gradient as the update direction. The results further demonstrate that the weighting method enables us to strike
    36	a balance between global and local gradients, allowing us to leverage
    37	the advantages of both simultaneously.
    38	10
    39	
    40	
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 9 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | tail -n 220" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   570	
   571	44.18
   572	30.26
   573	22.26
   574	30.21
   575	31.86
   576	26.68
   577	32.09
   578	9.35
   579	
   580	33.29
   581	22.46
   582	16.85
   583	26.03
   584	24.48
   585	19.04
   586	15.93
   587	7.33
   588	
   589	37.25
   590	18.26
   591	21.45
   592	24.88
   593	21.06
   594	17.85
   595	22.49
   596	5.31
   597	
   598	23.31
   599	12.58
   600	16.78
   601	16.77
   602	14.69
   603	11.89
   604	17.28
   605	4.14
   606	
   607	1.25*
   608	0.22*
   609	0.26*
   610	1.31*
   611	5.12*
   612	1.11*
   613	0.32*
   614	0.12*
   615	
   616	1.12*
   617	0.27*
   618	0.33*
   619	1.15*
   620	3.84*
   621	0.95*
   622	0.36*
   623	0.13*
   624	
   625	29.28
   626	13.54
   627	15.96
   628	20.77
   629	17.23
   630	15.06
   631	18.69
   632	3.80
   633	
   634	22.00
   635	11.12
   636	12.96
   637	13.92
   638	12.51
   639	11.36
   640	13.85
   641	3.24
   642	
   643	Swinv2
   644	
   645	MI [6]
   646	LTA [7]
   647	ODFA [8]
   648	GRA [21]
   649	BSR [19]
   650	TPA [5]
   651	IAAR [42]
   652	GAGA (Ours)
   653	
   654	40.16
   655	24.16
   656	24.17
   657	27.16
   658	29.12
   659	24.23
   660	28.91
   661	7.36
   662	
   663	28.33
   664	18.77
   665	17.26
   666	22.56
   667	22.39
   668	18.06
   669	23.16
   670	5.91
   671	
   672	42.36
   673	26.54
   674	26.13
   675	28.85
   676	27.51
   677	23.69
   678	28.04
   679	6.95
   680	
   681	29.55
   682	19.38
   683	18.27
   684	21.03
   685	21.52
   686	16.36
   687	22.96
   688	5.66
   689	
   690	46.21
   691	30.30
   692	29.16
   693	32.25
   694	31.28
   695	26.19
   696	32.27
   697	9.56
   698	
   699	33.36
   700	22.48
   701	26.11
   702	24.36
   703	23.61
   704	18.47
   705	25.94
   706	6.99
   707	
   708	17.66
   709	3.59
   710	5.96
   711	12.63
   712	8.56
   713	8.69
   714	9.69
   715	1.01
   716	
   717	11.54
   718	4.79
   719	4.36
   720	5.34
   721	4.89
   722	4.11
   723	5.79
   724	1.01
   725	
   726	40.26
   727	21.58
   728	24.96
   729	27.85
   730	25.06
   731	21.36
   732	26.33
   733	7.42
   734	
   735	27.89
   736	16.20
   737	20.26
   738	20.76
   739	17.65
   740	15.69
   741	22.06
   742	6.12
   743	
   744	0.98*
   745	0.25*
   746	0.28*
   747	0.86*
   748	5.23*
   749	0.50*
   750	0.15*
   751	0.09*
   752	
   753	0.89*
   754	0.31*
   755	0.34*
   756	0.84*
   757	3.56*
   758	0.55*
   759	0.21*
   760	0.13*
   761	
   762	visible light images act as the gallery set. The search modes include allsearch and indoor-search. The RegDB [52] dataset mainly focuses on the
   763	performance of cross-modal bidirectional retrieval, that is, the retrieval
   764	capabilities from the visible light modality to the infrared modality and
   765	from the infrared modality to the visible light modality. We chose the
   766	powerful baseline model AGW [1], which is a Re-50 pretrained on ImageNet, as the backbone network. The experimental results in Table 7
   767	demonstrate that our proposed GAGA method can also achieve strong
   768	attack performance on cross-modal Re-ID datasets.
   769	
   770	ferent attack methods. The adversarial examples are generated on
   771	the Re-50 model. The results indicate that ODFA has the shortest
   772	time for generating adversarial examples, but its attack success rate
   773	is very low. Secondly, TPA and PGN have relatively long runtimes.
   774	Although GAGA achieves state-of-the-art attack performance, its runtime is not the shortest. Given that the primary objective of this paper is to enhance adversarial transferability, in the future, we will investigate ways to reduce computational costs while maintaining high
   775	transferability.
   776	
   777	4.6. Eﬃciency analysis
   778	
   779	4.7. Ablation study and hyper-parameter analysis
   780	
   781	Since the GAGA method proposed in this paper requires an internal loop to obtain the globally aggregated gradient during each
   782	update, we conducted an eﬃciency analysis experiment. The results,
   783	as shown in Table 8, present the performance and runtime of dif-
   784	
   785	In this session, this paper conducts three ablation studies: (1) The
   786	hyper-parameters for the inner iteration number 𝑚; (2) The hyperparameters for the noise boundary 𝛽; (3) The inﬂuence of diﬀerent update directions on experimental results.
   787	9
   788	
   789	
codex
你这条消息实际列了 6 篇，我按这 6 篇逐篇拆。

**1. FMCNet+**
1. **触发观察**：VI-ReID 里大家都在学 modality-shared feature，但这样会丢掉一部分有判别力的 modality-specific 信息。已有补偿方法走图像生成，RGB 和 IR 互转太难，尤其 IR 生成可见光会带来颜色不一致等噪声，所以补偿线反而不如 shared feature 线。
2. **重定义动作**：作者把问题从“跨模态差异大”改成“缺失的 modality-specific 信息应该补偿，但补偿层级错了”。关键词是 feature-level modality compensation、modality-shared/specific decomposition、discriminative person-related modality-specific features。
3. **机制怎么长出来**：既然图像级补偿噪声大，就先用 SFD 把单模态特征拆成 shared 和 specific，再用 FMC 从对方 shared feature 生成缺失 specific feature，最后用 SFF 融合 shared、原 specific、生成 specific。逻辑绑定很紧，模块基本是从“特征级补偿比图像级更可控”自然推出的。
4. **证据闭环**：关键消融是 Base 57.09/53.11，加 SFD 到 63.16/58.83，加 FMC 到 65.50/62.32，加 SFF 到 66.34/62.51。MD loss 的三项逐步提升，FMC 里没有 adversarial loss 时反而掉点，说明不是“随便生成特征”有效，而是要让生成 specific feature 接近真实 specific feature 且保持身份判别。特征分布图也证明 shared/specific 被分开，生成特征贴近真实特征。
5. **reviewer 为什么买账**：卖的是视角，不只是模块。它把一条表现不佳的旧路线，图像级模态补偿，诊断为“补偿层级错了”，再给出端到端特征级补偿。这比单纯做一个跨模态融合模块更像创新。

**2. Focusing on pedestrians like human**
1. **触发观察**：换衣 ReID 方法多在网络结构上做身份相关信息、衣服无关信息、局部细节分支，但数据增强几乎没被系统做。衣服占图像大部分，脸、体态、步态等身份线索只在小局部，普通随机裁剪、旋转还可能破坏这些小线索。
2. **重定义动作**：作者把换衣问题从“去掉衣服干扰”重讲成“训练图像应该像人一样同时关注多个局部身份线索”。关键词是 human focus、ensemble coding hypothesis、image-level local detail learning、central emphasis、component continuity。
3. **机制怎么长出来**：CES 用 5×5 网格放大 body component 中央区域，CCP 把人体分成头肩、左右上身、左右下身五块，并用共享纵向偏移、左右镜像横向偏移保证连续，再用 smoothing 处理拼接断裂。机制和叙事有绑定，但具体网格、平滑是工程化选择，不如 FMCNet 和 GAReID 那么“公式自然推出”。
4. **证据闭环**：它和普通增强比较，HFIA 优于 AutoAugment 等，随机裁剪和旋转明显伤害性能，支撑“换衣需要定制增强”。消融里 CES 和 CCP 单独都不如组合，body components、pedestrian-oriented 配置、enlargement grid、smoothing 都分别带来提升。可视化显示 HFIA 更集中在行人和身份相关区域，检索例子也展示它不再被相似衣服强干扰。
5. **reviewer 为什么买账**：主要卖视角。它不是新 backbone，而是把“局部细节学习”从 feature branch 搬到 image augmentation，并且训练期使用、测试无开销。新意来自“换衣 ReID 的局部细节增强可以在图像层完成”，不是 CES 这个小操作本身。

**3. GAE-Net**
1. **触发观察**：视频 ReID 虽然有时序建模，但仍强依赖外观，容易被光照、颜色、穿着变化干扰。gait 对外观变化更鲁棒，但单独 gait 缺少外观信息，已有 gait-assisted 方法也没有充分解决 RGB 和 gait 的互补融合。
2. **重定义动作**：作者把视频 ReID 从“更强时序聚合”重定义为“外观和步态两个互补模态的训练期协同，再蒸馏回单模态部署”。关键词是 gait-assisted enhancement、dynamic two-stream aggregation、privileged multimodal teacher、local perceptual complementary distillation。
3. **机制怎么长出来**：DTA-Net 用 RGB ReID 分支和 GaitGL 分支形成多模态教师，DFA 用 DAW 和 DWA 动态融合外观与步态。然后 LPCD 把教师知识蒸馏给 RGB-only 学生，并把全局 logit 拆成多尺度局部 logit，区分 consistent 和 complementary knowledge。机制逻辑较紧，因为“训练期有 gait，测试期只用 RGB”自然导向蒸馏。
4. **证据闭环**：ReID branch 是 84.3 mAP，gait branch 单独只有 10.7，DTA-Net 到 85.8，GAE-Net 经 LPCD 到 87.7，同时参数从 164.1M 降到 24.8M。DAW、DWA、完整 DFA 逐步提升。LPCD 明显优于普通 KD、WSLD、NKD、DIST，多尺度集合 {1,2,4} 最好。检索、t-SNE 和激活图证明 gait 信息帮助压制遮挡和噪声。
5. **reviewer 为什么买账**：卖的是“特权信息蒸馏”的视角。真正 novelty 不是又加一个 gait 分支，而是让 gait 成为训练期教师，并通过局部互补知识把多模态能力压进单模态模型，部署时不用额外 gait 输入。

**4. GAReID**
1. **触发观察**：part misalignment 来自拍摄视角、检测误差、遮挡和背景。part-based 方法划分太粗，landmark-based 方法需要检测器和额外代价。作者进一步指出，GAP 后两张图的相似度其实是所有 part pair 相似度的平均，其中 misaligned pair 数量远多于 aligned pair，背景 pair 也会污染平均。
2. **重定义动作**：它把“人体部位没对齐”重定义为“相似度聚合中，对齐 part 的贡献被大量错配 part 稀释”。关键词是 high-order part similarity、grouped high-order pooling、attentive high-order pooling、landmark-free alignment。
3. **机制怎么长出来**：如果 aligned part 的相似度天然更高，那么高阶函数会放大高相似 pair 和低相似 pair 的差距。于是 GHOP 用高阶 Kronecker 交互做无 landmark 对齐，再用 group 和 shuffle 压缩维度。背景会污染相似度，所以 AHOP 用基于特征范数的前景 attention 抑制背景。这个逻辑非常紧。
4. **证据闭环**：最强证据不是单纯涨点，而是 Fig. 6 和 Fig. 7 的相似度可视化。高阶后 aligned landmark pair 更突出，misaligned pair 的高相似减少。n=3 相比 n=1 在 Market mAP 提升 3.57，在 Duke 提升 7.99，n>3 后收益饱和，符合“错位已被大幅缓解”的解释。AHOP 优于 GHOP，说明前景过滤确实补上了背景污染问题。
5. **reviewer 为什么买账**：这篇卖的是视角加机制。它没有借助姿态或人体解析，却用一个相似度分解公式解释为什么 misalignment 会发生，再让高阶映射成为自然解法。novelty 的核心是问题数学化，而不是高阶池化这个名词。

**5. Generalizable ReID with bi-stream interactive learning**
1. **触发观察**：ReID 跨数据集泛化差。PK sampler 全局随机，小 batch 内样本关系弱，难以提供有效 hard samples。已有 graph sampler 又慢，数据规模大时不稳定。作者还注意到，采样质量依赖 backbone 特征，而 backbone 又依赖采样到的训练样本。
2. **重定义动作**：作者把泛化问题从“换一个更强网络或损失”重定义为“metric learning sampling 和 representation learning 之间缺少交互”。关键词是 bi-stream interactive learning、correlation graph sampler、learnable batch sampling、feature reconstruction。
3. **机制怎么长出来**：CGS 每个 epoch 用 Spherical-LSH 先把相近类别放进 bucket，再用 feature-map adaptive matching 找邻近类别，构造相关 hard batch。GSANet 保持高分辨率流，GRSR 用稀疏全局相关像素重构特征，提升特征质量。特征更好会让下一轮 CGS 更好，CGS 更好又提供更有价值样本。这个“互相增强”的逻辑成立，但模块数量较多，绑定不如 GAReID 干净。
4. **证据闭环**：采样对比显示 CGS 好于 random、PK、cluster，并且比已有 GS 快很多，Market 上采样 0.1 秒对 4 秒，MSMT(all) 上 1 秒对 40 秒。CGS 相比 Cluster 在 Market 到 CUHK03 上提升约 3.0 Rank-1 和 2.8 mAP，在 Market 到 MSMT17 上提升约 3.9 和 2.9。CGS、GRSR、GSANet 的组合消融显示完整模型最好。采样可视化证明 CGS 找到衣服颜色、局部外观相近的邻近类别。
5. **reviewer 为什么买账**：卖的是系统机制。单看每个部件都不算特别新，但“采样器和特征网络闭环互动”给了它论文结构。风险是东西较多，reviewer 会要求每个部件隔离清楚，否则容易被看成堆模块。

**6. Global aggregated gradient-guided adversarial attacks**
1. **触发观察**：ReID 是 open-set ranking，分类攻击不能直接搬。已有 ReID 攻击黑盒迁移差，作者观察到早期迭代里 current gradient 和 global aggregated gradient 一致性很低，更新方向随机震荡，导致迁移性不足。
2. **重定义动作**：作者把“黑盒攻击迁移差”重定义为“早期攻击阶段的梯度一致性不足”。关键词是 global aggregated gradient、gradient consistency、pre-convergence、weighted gradient。
3. **机制怎么长出来**：每次外层更新前做 inner loop，沿预收敛路径采样多个梯度并平均成 global aggregated gradient。再计算当前梯度和全局聚合梯度的余弦一致性，低一致时更信任全局梯度，高一致时更信任当前梯度。机制几乎直接从重定义推出。
4. **证据闭环**：Fig. 3 先证明旧方法早期 gradient consistency 低。update direction 消融比较 current gradient、GAG 和 weighted gradient，weighted gradient 最好。inner iteration 数 m 和噪声边界 beta 的实验展示存在合理区间，不是随便加算力。黑盒、白盒、ensemble、防御模型、跨模态 ReID 都验证有效，检索可视化显示 adversarial query 会让 top-10 全错。
5. **reviewer 为什么买账**：卖的是可测中间变量。它不是说“我们攻击更强”，而是提出“迁移性差的原因是早期梯度一致性低”，再用一个很直接的算法修这个变量。novelty 的来源是诊断变量，而不是攻击 loss 本身。

这批论文共同套路很明确：先抓一个具体失败点，不是泛泛说任务难，再把失败点命名成一个可操作的新问题。然后机制要尽量从这个新问题自然长出来，最后用消融或可视化证明那个中间问题确实被修了，而不是只报 SOTA 表。对我们做 B 类 ReID 方法稿来说，最值得学的是找“可测的中间变量”，例如补偿层级、错位 pair 占比、采样相关性、梯度一致性，而不是先想一个模块名字。
tokens used
260,734
你这条消息实际列了 6 篇，我按这 6 篇逐篇拆。

**1. FMCNet+**
1. **触发观察**：VI-ReID 里大家都在学 modality-shared feature，但这样会丢掉一部分有判别力的 modality-specific 信息。已有补偿方法走图像生成，RGB 和 IR 互转太难，尤其 IR 生成可见光会带来颜色不一致等噪声，所以补偿线反而不如 shared feature 线。
2. **重定义动作**：作者把问题从“跨模态差异大”改成“缺失的 modality-specific 信息应该补偿，但补偿层级错了”。关键词是 feature-level modality compensation、modality-shared/specific decomposition、discriminative person-related modality-specific features。
3. **机制怎么长出来**：既然图像级补偿噪声大，就先用 SFD 把单模态特征拆成 shared 和 specific，再用 FMC 从对方 shared feature 生成缺失 specific feature，最后用 SFF 融合 shared、原 specific、生成 specific。逻辑绑定很紧，模块基本是从“特征级补偿比图像级更可控”自然推出的。
4. **证据闭环**：关键消融是 Base 57.09/53.11，加 SFD 到 63.16/58.83，加 FMC 到 65.50/62.32，加 SFF 到 66.34/62.51。MD loss 的三项逐步提升，FMC 里没有 adversarial loss 时反而掉点，说明不是“随便生成特征”有效，而是要让生成 specific feature 接近真实 specific feature 且保持身份判别。特征分布图也证明 shared/specific 被分开，生成特征贴近真实特征。
5. **reviewer 为什么买账**：卖的是视角，不只是模块。它把一条表现不佳的旧路线，图像级模态补偿，诊断为“补偿层级错了”，再给出端到端特征级补偿。这比单纯做一个跨模态融合模块更像创新。

**2. Focusing on pedestrians like human**
1. **触发观察**：换衣 ReID 方法多在网络结构上做身份相关信息、衣服无关信息、局部细节分支，但数据增强几乎没被系统做。衣服占图像大部分，脸、体态、步态等身份线索只在小局部，普通随机裁剪、旋转还可能破坏这些小线索。
2. **重定义动作**：作者把换衣问题从“去掉衣服干扰”重讲成“训练图像应该像人一样同时关注多个局部身份线索”。关键词是 human focus、ensemble coding hypothesis、image-level local detail learning、central emphasis、component continuity。
3. **机制怎么长出来**：CES 用 5×5 网格放大 body component 中央区域，CCP 把人体分成头肩、左右上身、左右下身五块，并用共享纵向偏移、左右镜像横向偏移保证连续，再用 smoothing 处理拼接断裂。机制和叙事有绑定，但具体网格、平滑是工程化选择，不如 FMCNet 和 GAReID 那么“公式自然推出”。
4. **证据闭环**：它和普通增强比较，HFIA 优于 AutoAugment 等，随机裁剪和旋转明显伤害性能，支撑“换衣需要定制增强”。消融里 CES 和 CCP 单独都不如组合，body components、pedestrian-oriented 配置、enlargement grid、smoothing 都分别带来提升。可视化显示 HFIA 更集中在行人和身份相关区域，检索例子也展示它不再被相似衣服强干扰。
5. **reviewer 为什么买账**：主要卖视角。它不是新 backbone，而是把“局部细节学习”从 feature branch 搬到 image augmentation，并且训练期使用、测试无开销。新意来自“换衣 ReID 的局部细节增强可以在图像层完成”，不是 CES 这个小操作本身。

**3. GAE-Net**
1. **触发观察**：视频 ReID 虽然有时序建模，但仍强依赖外观，容易被光照、颜色、穿着变化干扰。gait 对外观变化更鲁棒，但单独 gait 缺少外观信息，已有 gait-assisted 方法也没有充分解决 RGB 和 gait 的互补融合。
2. **重定义动作**：作者把视频 ReID 从“更强时序聚合”重定义为“外观和步态两个互补模态的训练期协同，再蒸馏回单模态部署”。关键词是 gait-assisted enhancement、dynamic two-stream aggregation、privileged multimodal teacher、local perceptual complementary distillation。
3. **机制怎么长出来**：DTA-Net 用 RGB ReID 分支和 GaitGL 分支形成多模态教师，DFA 用 DAW 和 DWA 动态融合外观与步态。然后 LPCD 把教师知识蒸馏给 RGB-only 学生，并把全局 logit 拆成多尺度局部 logit，区分 consistent 和 complementary knowledge。机制逻辑较紧，因为“训练期有 gait，测试期只用 RGB”自然导向蒸馏。
4. **证据闭环**：ReID branch 是 84.3 mAP，gait branch 单独只有 10.7，DTA-Net 到 85.8，GAE-Net 经 LPCD 到 87.7，同时参数从 164.1M 降到 24.8M。DAW、DWA、完整 DFA 逐步提升。LPCD 明显优于普通 KD、WSLD、NKD、DIST，多尺度集合 {1,2,4} 最好。检索、t-SNE 和激活图证明 gait 信息帮助压制遮挡和噪声。
5. **reviewer 为什么买账**：卖的是“特权信息蒸馏”的视角。真正 novelty 不是又加一个 gait 分支，而是让 gait 成为训练期教师，并通过局部互补知识把多模态能力压进单模态模型，部署时不用额外 gait 输入。

**4. GAReID**
1. **触发观察**：part misalignment 来自拍摄视角、检测误差、遮挡和背景。part-based 方法划分太粗，landmark-based 方法需要检测器和额外代价。作者进一步指出，GAP 后两张图的相似度其实是所有 part pair 相似度的平均，其中 misaligned pair 数量远多于 aligned pair，背景 pair 也会污染平均。
2. **重定义动作**：它把“人体部位没对齐”重定义为“相似度聚合中，对齐 part 的贡献被大量错配 part 稀释”。关键词是 high-order part similarity、grouped high-order pooling、attentive high-order pooling、landmark-free alignment。
3. **机制怎么长出来**：如果 aligned part 的相似度天然更高，那么高阶函数会放大高相似 pair 和低相似 pair 的差距。于是 GHOP 用高阶 Kronecker 交互做无 landmark 对齐，再用 group 和 shuffle 压缩维度。背景会污染相似度，所以 AHOP 用基于特征范数的前景 attention 抑制背景。这个逻辑非常紧。
4. **证据闭环**：最强证据不是单纯涨点，而是 Fig. 6 和 Fig. 7 的相似度可视化。高阶后 aligned landmark pair 更突出，misaligned pair 的高相似减少。n=3 相比 n=1 在 Market mAP 提升 3.57，在 Duke 提升 7.99，n>3 后收益饱和，符合“错位已被大幅缓解”的解释。AHOP 优于 GHOP，说明前景过滤确实补上了背景污染问题。
5. **reviewer 为什么买账**：这篇卖的是视角加机制。它没有借助姿态或人体解析，却用一个相似度分解公式解释为什么 misalignment 会发生，再让高阶映射成为自然解法。novelty 的核心是问题数学化，而不是高阶池化这个名词。

**5. Generalizable ReID with bi-stream interactive learning**
1. **触发观察**：ReID 跨数据集泛化差。PK sampler 全局随机，小 batch 内样本关系弱，难以提供有效 hard samples。已有 graph sampler 又慢，数据规模大时不稳定。作者还注意到，采样质量依赖 backbone 特征，而 backbone 又依赖采样到的训练样本。
2. **重定义动作**：作者把泛化问题从“换一个更强网络或损失”重定义为“metric learning sampling 和 representation learning 之间缺少交互”。关键词是 bi-stream interactive learning、correlation graph sampler、learnable batch sampling、feature reconstruction。
3. **机制怎么长出来**：CGS 每个 epoch 用 Spherical-LSH 先把相近类别放进 bucket，再用 feature-map adaptive matching 找邻近类别，构造相关 hard batch。GSANet 保持高分辨率流，GRSR 用稀疏全局相关像素重构特征，提升特征质量。特征更好会让下一轮 CGS 更好，CGS 更好又提供更有价值样本。这个“互相增强”的逻辑成立，但模块数量较多，绑定不如 GAReID 干净。
4. **证据闭环**：采样对比显示 CGS 好于 random、PK、cluster，并且比已有 GS 快很多，Market 上采样 0.1 秒对 4 秒，MSMT(all) 上 1 秒对 40 秒。CGS 相比 Cluster 在 Market 到 CUHK03 上提升约 3.0 Rank-1 和 2.8 mAP，在 Market 到 MSMT17 上提升约 3.9 和 2.9。CGS、GRSR、GSANet 的组合消融显示完整模型最好。采样可视化证明 CGS 找到衣服颜色、局部外观相近的邻近类别。
5. **reviewer 为什么买账**：卖的是系统机制。单看每个部件都不算特别新，但“采样器和特征网络闭环互动”给了它论文结构。风险是东西较多，reviewer 会要求每个部件隔离清楚，否则容易被看成堆模块。

**6. Global aggregated gradient-guided adversarial attacks**
1. **触发观察**：ReID 是 open-set ranking，分类攻击不能直接搬。已有 ReID 攻击黑盒迁移差，作者观察到早期迭代里 current gradient 和 global aggregated gradient 一致性很低，更新方向随机震荡，导致迁移性不足。
2. **重定义动作**：作者把“黑盒攻击迁移差”重定义为“早期攻击阶段的梯度一致性不足”。关键词是 global aggregated gradient、gradient consistency、pre-convergence、weighted gradient。
3. **机制怎么长出来**：每次外层更新前做 inner loop，沿预收敛路径采样多个梯度并平均成 global aggregated gradient。再计算当前梯度和全局聚合梯度的余弦一致性，低一致时更信任全局梯度，高一致时更信任当前梯度。机制几乎直接从重定义推出。
4. **证据闭环**：Fig. 3 先证明旧方法早期 gradient consistency 低。update direction 消融比较 current gradient、GAG 和 weighted gradient，weighted gradient 最好。inner iteration 数 m 和噪声边界 beta 的实验展示存在合理区间，不是随便加算力。黑盒、白盒、ensemble、防御模型、跨模态 ReID 都验证有效，检索可视化显示 adversarial query 会让 top-10 全错。
5. **reviewer 为什么买账**：卖的是可测中间变量。它不是说“我们攻击更强”，而是提出“迁移性差的原因是早期梯度一致性低”，再用一个很直接的算法修这个变量。novelty 的来源是诊断变量，而不是攻击 loss 本身。

这批论文共同套路很明确：先抓一个具体失败点，不是泛泛说任务难，再把失败点命名成一个可操作的新问题。然后机制要尽量从这个新问题自然长出来，最后用消融或可视化证明那个中间问题确实被修了，而不是只报 SOTA 表。对我们做 B 类 ReID 方法稿来说，最值得学的是找“可测的中间变量”，例如补偿层级、错位 pair 占比、采样相关性、梯度一致性，而不是先想一个模块名字。
