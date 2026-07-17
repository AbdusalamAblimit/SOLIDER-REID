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
session id: 019ef835-e80d-73b2-8477-d674d530b7c9
--------
user
你是 ReID 创新方法论拆解员 21。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(21)==
Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf
Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf
Privacy preserving person re-identification via anonymizing diffusion model.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我先按项目规则确认本轮上下文，然后逐篇抽取前 8 页的方法部分。重点会放在“观察如何被包装成问题”和“证据如何闭环”，不按模块清单复述。
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
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'find .. -name GOAL.md -maxdepth 6 -print 2>/dev/null | head -20' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
exec
/bin/zsh -lc 'find .. -name CLAUDE.md -maxdepth 6 -print 2>/dev/null | head -20' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "find .. -name STATUS.md -path '*/experiments/STATUS.md' -maxdepth 6 -print 2>/dev/null | head -20" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
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
当前沙箱只暴露了论文目录，父级的 `GOAL.md`、`CLAUDE.md` 和 `experiments/STATUS.md` 读不到；我会按你贴出的规则执行。目录里目标 PDF 都在，下面先定位每篇的方法标题和页内结构。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf' - | rg -n \""'^('"\\s*[0-9IVX]+\\.?\\s+)?(METHOD|Method|METHODOLOGY|Methodology|Approach|APPROACH|Proposed|Framework|Experiments|Experiment|RELATED|Related|III\\.|3\\.|4\\.)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
152:Related Work
241:4.1
244:Framework Overview
279:4.2
307:4.3
329:4.4

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf' - | rg -n \""'^('"\\s*[0-9IVX]+\\.?\\s+)?(METHOD|Method|METHODOLOGY|Methodology|Approach|APPROACH|Proposed|Framework|Experiments|Experiment|RELATED|Related|III\\.|3\\.|4\\.)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
265:III. O UR A PPROACH

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf' - | rg -n \""'^('"\\s*[0-9IVX]+\\.?\\s+)?(METHOD|Method|METHODOLOGY|Methodology|Approach|APPROACH|Proposed|Framework|Experiments|Experiment|RELATED|Related|III\\.|3\\.|4\\.)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
30:Experimental results on several popular datasets prove that our
317:III. M ETHODOLOGY

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Privacy preserving person re-identification via anonymizing diffusion model.pdf' - | rg -n \""'^('"\\s*[0-9IVX]+\\.?\\s+)?(METHOD|Method|METHODOLOGY|Methodology|Approach|APPROACH|Proposed|Framework|Experiments|Experiment|RELATED|Related|III\\.|3\\.|4\\.)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
127:Experiments on public datasets demonstrate that ADM achieves
142:2. Related work
187:3. Methodology
225:3.2. Anonymizing diﬀusion model
241:3.2.1. 2-stage ﬁne-tuning
255:3.1. Image-text pair generation
355:3.2.2. Image generation
370:3.2.3. ADM griddle
405:3.3. ID separation degree (ISD)
459:Methods
509:4. Experiment
510:4.1. Implementation details
530:Methods
621:4.2. Comparison with state-of-the-arts
622:4.2.1. Experimental settings
636:4.4. Ablation studies
638:4.2.2. Experimental results
652:4.4.1. Comparison with base model
659:4.4.2. Eﬀectiveness of diﬀerent components
675:4.3. Results on benchmark datasets

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | rg -n \""'^('"\\s*[0-9IVX]+\\.?\\s+)?(METHOD|Method|METHODOLOGY|Methodology|Approach|APPROACH|Proposed|Framework|Experiments|Experiment|RELATED|Related|III\\.|3\\.|4\\.)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
270:III. M ETHODOLOGY

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf' - | rg -n \""'^('"\\s*[0-9IVX]+\\.?\\s+)?(METHOD|Method|METHODOLOGY|Methodology|Approach|APPROACH|Proposed|Framework|Experiments|Experiment|RELATED|Related|III\\.|3\\.|4\\.)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
368:III. M ETHOD

codex
方法区已经定位：有几篇是 IEEE 风格的 `III. Methodology/Approach`，两篇是普通编号的小节。我接下来直接抽方法主体，并顺手看它们在实验部分用什么消融或可视化闭合论证。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf' - | nl -ba | sed -n '220,380p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   220	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   221	
   222	26:6
   223	
   224	C. Wang et al.
   225	
   226	Fig. 3. The same image pair has different metric values under different illumination, and there exists a
   227	minimum metric distance under a certain lighting condition.
   228	
   229	Fig. 4. Optimal illumination differs for different image pairs, and the optimal illumination distance between
   230	dissimilar (fig7) is also much larger than that between similar (figs1–6).
   231	
   232	The analysis results of several different object groups are shown in Figures 3 and 4, from which
   233	the following two conclusions can be observed: (1) Image pairs have different metric values under
   234	different lighting conditions, and there exists a certain lighting condition in which the distance
   235	of the image pair is minimized, i.e., the optimal lighting. (2) Not all image pairs are optimally
   236	illuminated at a light condition of 1. The optimal illumination differs for different image pairs.
   237	Therefore, replacing the distance under the original lighting conditions with the distance between
   238	image pairs under the optimal lighting conditions can more effectively express the semantic distance
   239	of the image pairs.
   240	4
   241	4.1
   242	
   243	The Proposed Method
   244	Framework Overview
   245	
   246	As shown in Figure 5, our UIFE framework consists of four modules. (1) Lighting normalization
   247	module, mainly composed of a generator G, which is used to convert images captured under
   248	different lighting conditions into images under normal lighting conditions. (2) Pre-defined lighting
   249	transformation module, which is composed of a series of generators. Each generator 𝐺𝑖 is used to
   250	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   251	
   252	Optimal Illumination Distance Metrics for Person Re-Identification
   253	
   254	26:7
   255	
   256	Fig. 5. Our proposed framework for optimal illumination distance metric. The framework includes illumination normalization, pre-defined illumination transformation, pedestrian feature extraction, and optimal
   257	illumination distance replacement.
   258	
   259	transform images captured under normal lighting conditions into images under a specific predefined lighting condition. (3) Feature extraction module, which mainly consists of two pre-trained
   260	models. One is the EH model pre-trained under normal lighting conditions, and the other is the
   261	EL model pre-trained under complex lighting conditions. They are used to extract features from
   262	images captured under normal lighting conditions and images captured under pre-defined lighting
   263	conditions, respectively. (4) Optimal lighting distance replacement module, which replaces the
   264	distance matrix formed by image pairs under normal lighting conditions with the distance matrix
   265	formed under pre-defined lighting conditions. The two matrices are concatenated along the channel
   266	dimension and the minimum value is selected along the channel dimension to form the optimal
   267	distance matrix, which is used for sorting and retrieval.
   268	During the inference stage, all GANs and feature extractors are pre-trained. Given a gallery
   269	library composed of images captured under different lighting conditions, they are input to the
   270	Lighting Normalization module, where a pre-trained model G is used to transform the images to
   271	normal lighting conditions. The resulting images under normal lighting conditions are then input
   272	to the Pre-defined Lighting Transformation module to generate images under different lighting
   273	conditions. The images under different lighting conditions, as well as the images under normal
   274	lighting conditions and the query images, are input to the feature extraction module, where their
   275	features are extracted and a distance calculation is performed to form a series of metric matrices.
   276	These metric matrices are then input to the Optimal Illumination Distance Replacement module,
   277	where the minimum value for each image pair is selected along the channels to form the optimal
   278	metric matrix 𝑇 ∗ . This matrix is used for retrieval ranking.
   279	4.2
   280	
   281	Lighting Normalization
   282	
   283	Previous studies [11, 13, 24] have suggested that the issue of data samples taken from different cameras, perspectives, and scenes can be considered a problem of different domains, and
   284	cross-domain approaches can be utilized to address this problem. These methods have also been
   285	demonstrated to be effective. Similarly, the problem of person Re-ID in complex lighting conditions can be seen as a problem of samples from different lighting domains due to variations in
   286	lighting conditions. Therefore, GAN can be used to unify images taken under different lighting
   287	conditions to common lighting. Given a gallery image dataset 𝑋 ∗ consisting of images taken under
   288	different lighting conditions, feeding 𝑋 ∗ into the generator G results in obtaining images X under
   289	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   290	
   291	26:8
   292	
   293	C. Wang et al.
   294	
   295	normal lighting conditions:
   296	𝑋 = 𝐺 (𝑋 ∗ ),
   297	
   298	(1)
   299	
   300	where the gallery image dataset 𝑋 ∗ contains 𝑛 images, which are captured under 𝐿 different
   301	lighting conditions. Regarding the lighting normalization process, the GAN used in the lighting
   302	normalization conversion module adopts the network architecture from CycleGAN [35]. To train a
   303	GAN, the original images in the (Market-1501, DukeMTMC-ReID) training dataset are considered
   304	as labels, and they are transformed using gamma correction into images under different lighting
   305	conditions, which are considered as inputs. The GAN loss [7] is used to train the model, and a
   306	pre-trained generator G is obtained as the final result.
   307	4.3
   308	
   309	Pre-Defined Lighting Transformation
   310	
   311	This module is composed of a series of generators 𝐺 ∗ = {𝐺 1 ...𝐺𝑖 ...𝐺 𝐿 }, which are used to transform
   312	retrieval set images 𝑋 and a query set images 𝑌 under normal lighting conditions to specified
   313	lighting conditions. Specifically, given a series of pre-defined lighting conditions 𝐴 = {𝑎 1 ...𝑎𝑖 ...𝑎𝐿 },
   314	the generator 𝐺𝑖 is used to transform 𝑋 and 𝑌 to lighting condition 𝑎𝑖 , resulting in 𝑋𝑖 and 𝑌𝑖 . This
   315	can be formalized as
   316	𝑋𝑖 = 𝐺𝑖 (𝑋 )
   317	
   318	(2)
   319	
   320	𝑌𝑖 = 𝐺𝑖 (𝑌 ),
   321	
   322	(3)
   323	
   324	where 𝑋𝑖 and 𝑌𝑖 represent the gallery and query image data under lighting condition 𝑎𝑖 , respectively. Given m images in 𝑌𝑖 , a series of gallery images under pre-defined lighting conditions
   325	𝑄 = {𝑋, 𝑋 1, 𝑋 2 ...𝑋𝑖 ...𝑋𝐿 } and query images 𝐸 = {𝑌, 𝑌1, 𝑌2 ...𝑌𝑖 ...𝑌𝐿 } can be obtained using the predefined lighting transformation module with the pre-sequenced 𝐺 ∗ models. For the training of each
   326	𝐺𝑖 , the original 𝑋 and images from the training dataset are used as inputs to the 𝐺𝑖 model, while
   327	the images from the training dataset after gamma correction to the images under lighting condition 𝑎𝑖 are used as labels. Finally, the GAN loss [7] is used for training, resulting in a pre-trained
   328	generator 𝐺 ∗ .
   329	4.4
   330	
   331	Pedestrian Feature Extraction
   332	
   333	This module is to extract features from images under different lighting conditions. Firstly, a feature
   334	extractor EH trained under normal lighting conditions is used to extract features from 𝑋 and 𝑌 ,
   335	resulting in 𝐹𝑋 and 𝐹𝑌 . Then, the Euclidean distance between 𝐹𝑋 and 𝐹𝑌 is calculated to obtain the
   336	distance measurement matrix 𝑑. Next, a pre-trained model EL under complex lighting conditions
   337	is used to extract features from 𝑋𝑖 and 𝑌𝑖 , resulting in 𝐹𝑋𝑖 and 𝐹𝑌𝑖 , and the Euclidean distance
   338	between them is calculated to obtain 𝑑𝑖 . This process can be formalized as follows:
   339	𝐹𝑋 = 𝐸𝐻 (𝑋 ),
   340	
   341	𝐹𝑌 = 𝐸𝐻 (𝑌 ),
   342	
   343	𝑑 = 𝐷 (𝐹𝑋, 𝐹𝑌 ),
   344	𝐹𝑋𝑖 = 𝐸𝐿(𝑥𝑖 ),
   345	
   346	𝐹𝑌𝑖 = 𝐸𝐿(𝑌𝑖 ),
   347	
   348	𝑑𝑖 = 𝐷 (𝐹𝑋𝑖 , 𝐹𝑌𝑖 ),
   349	
   350	(4)
   351	(5)
   352	(6)
   353	(7)
   354	
   355	where 𝐹𝑋 ⊂ 𝑅𝑛×2,048 , 𝐹𝑋𝑖 ⊂ 𝑅𝑛×2,048 are the features of the gallery images under normal lighting
   356	conditions and under lighting condition 𝑎𝑖 , respectively, and 𝐹𝑌 ⊂ 𝑅𝑚×2,048 , 𝐹𝑌𝑖 ⊂ 𝑅𝑚×2,048 are
   357	the features of the query images under normal lighting conditions and under lighting condition
   358	𝑎𝑖 , respectively. 𝐷 represents the Euclidean distance calculation between the two features, and
   359	𝑑 ⊂ 𝑅𝑚×𝑛 and 𝑑𝑖 ⊂ 𝑅𝑚×𝑛 represent the Euclidean distance measurement matrix between images
   360	under normal lighting conditions and under lighting condition 𝑎𝑖 . Therefore, this model can obtain
   361	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   362	
   363	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '250,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   250	multi-view framework that transfers complementary identity
   251	cues to single-image representations [21].
   252	Although these methods have advanced occlusion-aware
   253	representation learning or cross-modality feature modeling,
   254	they remain insufficient for occluded VI-ReID. This task
   255	must simultaneously address missing identity cues caused
   256	by occlusion and feature inconsistency caused by modality
   257	variation, resulting in more severe cross-modality misalignment. Moreover, single-modality completion cannot exploit
   258	complementary cues from another modality, while existing
   259	VI-ReID methods generally emphasize modality alignment
   260	rather than explicit recovery of missing identity information.
   261	To address these issues, our OCCNet jointly performs intramodality and cross-modality feature completion, producing
   262	
   263	more complete and discriminative representations for robust
   264	matching under occlusion.
   265	III. O UR A PPROACH
   266	In this section, we provide a detailed introduction to the proposed Occlusion-aware Cross-modality Completion Network
   267	(OCCNet). We first present the Graph-guided Feature Completion Transformer (GFCT), which performs token-level feature
   268	completion with graph-guided attention. Then, we describe the
   269	Semantic-aware Region Completion Module (SRCM), which
   270	extends the completion to region-level through cross-modality
   271	alignment. Finally, the Multi-distribution Consistency Alignment (MCA) loss is introduced to enforce consistency among
   272	different completion paths.
   273	A. Overview
   274	Fig. 2 illustrates the overall architecture of the proposed
   275	OCCNet. Given a pair of visible and near-infrared images,
   276	a shared ResNet50 [22], pre-trained on ImageNet [23], is
   277	vis
   278	employed as the backbone to extract feature maps Xcnn
   279	
   280	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
   281	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   282	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   283	
   284	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   285	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
   286	
   287	4
   288	
   289	nir
   290	and Xcnn
   291	from the two modalities. Although the backbone
   292	parameters are shared, the extracted feature maps correspond
   293	to different input modalities, namely the visible and nearinfrared branches.
   294	The pre-trained ImageNet mainly provides a strong initialization for learning general visual patterns, while the backbone
   295	is subsequently fine-tuned on the VI-ReID training data containing both VIS and NIR images, allowing the network to
   296	adapt its weights to modality-specific characteristics during
   297	training. As a result, the shared backbone captures both
   298	modality-invariant identity cues and modality-dependent feature variations. Based on these representations, the proposed
   299	completion modules further refine the features to enhance
   300	cross-modality consistency and robustness under occlusion.
   301	The proposed Graph-guided Feature Completion Transformer (GFCT) then exploits semantic correlations among
   302	tokens to complete occluded regions, thus generating the
   303	structure-enhanced token-level features X̃ vis and X̃ nir . Based
   304	on these representations, the proposed Semantic-aware Region
   305	Completion Module (SRCM) aggregates tokens into semantically consistent regions and performs adaptive cross-modality
   306	alignment, producing refined region-level features X̂ vis and
   307	X̂ nir . To ensure semantic coherence between different completion paths, the Multi-distribution Consistency Alignment
   308	(MCA) loss is introduced to align token-level and region-level
   309	features within a unified distributional space. Through this
   310	process, the comprehensive and discriminative features Ovis
   311	and Onir are obtained for subsequent person re-identification.
   312	Given the structural similarity between modalities, we
   313	describe the process using the visible modality for clarity.
   314	
   315	B. Graph-guided Feature Completion Transformer
   316	Occlusion inevitably leads to partial feature loss and semantic discontinuity, which seriously damages the integrity
   317	and discriminability of pedestrian representation. Therefore,
   318	we propose a Graph-guided Feature Completion Transformer
   319	(GFCT) that embeds semantic structural priors into the completion process. Unlike conventional vision Transformers that
   320	rely on positional embeddings while ignoring structural relationships, GFCT dynamically builds a semantic adjacency
   321	matrix, explicitly modeling token correlations to enhance
   322	representation robustness under occlusion.
   323	Let the convolutional feature map extracted from ResNet50
   324	vis
   325	be denoted as Xcnn
   326	∈ RC×H×W . It is first reshaped into a
   327	sequence of token embeddings X vis ∈ RN ×D , where N is the
   328	total number of tokens, and D denotes the feature dimension.
   329	The separation of X vis into occluded Xovis and non-occluded
   330	Xcvis components shown in Fig. 2 is for illustration only and
   331	is not explicitly required during training or inference.
   332	To embed explicit semantic relationships into the attention process, we dynamically construct a semantic graph
   333	represented by an adjacency matrix Av ∈ RN ×N . The
   334	semantic graph here is constructed from latent correlations
   335	among learned region-level features, rather than from explicit
   336	semantic parsing annotations or predefined body-part labels.
   337	Specifically, each node corresponds to a region representation
   338	aggregated from the feature map, and the graph edges encode
   339	
   340	X vis
   341	
   342	element-wise
   343	multiplication
   344	Add&Norm
   345	
   346	attention-weighted
   347	multiplication
   348	
   349	Feedforward Network
   350	Add&Norm
   351	
   352	Sum
   353	
   354	Softmax
   355	
   356	A
   357	
   358	Scaling
   359	
   360	Qv
   361	
   362	Product
   363	
   364	Linear
   365	
   366	Kv
   367	
   368	Vv
   369	
   370	Linear
   371	
   372	Linear
   373	
   374	semantic similarity
   375	computation
   376	
   377	X vis
   378	
   379	Fig. 3: Illustration of Adaptive Graph-guided Attention module.
   380	
   381	similarity relationships between these region features in the
   382	learned feature space. Each entry Avij captures the semantic
   383	similarity between token i and token j, computed as:
   384	exp(ϕ(Xivis ) · ϕ(Xjvis )/τ )
   385	Avij = PN
   386	,
   387	vis
   388	vis
   389	k=1 exp(ϕ(Xi ) · ϕ(Xk )/τ )
   390	
   391	(1)
   392	
   393	where ϕ(·) represents a linear embedding projection, and τ
   394	is a temperature factor controlling the sharpness of semantic
   395	similarity distribution. Although occlusions may affect certain
   396	tokens, the similarity graph is computed over all tokens, and
   397	contextual tokens provide reliable semantic cues that guide the
   398	completion process. Since semantic similarity is estimated in a
   399	global feature space, the graph structure is determined by the
   400	overall contextual coherence of the feature map rather than by
   401	a few locally corrupted responses. Therefore, occluded tokens
   402	are not repaired only by their own unreliable features, but
   403	by aggregating information from multiple semantically related
   404	tokens across the entire image.
   405	To explicitly encode semantic structure into token completion, we propose the Adaptive Graph-guided Attention (AGA)
   406	in GFCT, which utilizes the dynamically computed semantic
   407	adjacency matrix Av to directly guide and restrict the selfattention interactions among tokens:
   408	
   409	
   410	v
   411	v ⊤
   412	Av
   413	ij ·(Qi (Kj ) )
   414	√
   415	exp
   416	N
   417	X
   418	dk
   419	
   420	 Vjv ,
   421	Attaga
   422	=
   423	(2)
   424	v
   425	i
   426	PN
   427	v )⊤ )
   428	Aim ·(Qv
   429	(Km
   430	i
   431	√
   432	exp
   433	j=1
   434	m=1
   435	d
   436	k
   437	
   438	v
   439	
   440	v
   441	
   442	v
   443	
   444	where Q , K , and V are the standard query, key, and value
   445	projections of X vis , and dk is the key dimension. Av ∈ RN ×N
   446	explicitly restricts attention propagation to semantically relevant token pairs, thus enhancing the structural consistency of
   447	occluded feature completion. The term Qvi (Kjv )⊤ measures the
   448	semantic similarity between tokens i and j. This mechanism
   449	significantly enhances the structural consistency and improves
   450	the effectiveness of feature completion in occluded regions.
   451	
   452	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
   453	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   454	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   455	
   456	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   457	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
   458	
   459	5
   460	
   461	Therefore, during inference, the completion of occluded regions can be implicitly achieved through contextual aggregation from semantically related visible tokens.
   462	In Fig. 3, by explicitly integrating the semantic structure into
   463	attention computation, AGA generates structurally enhanced
   464	token-level features X̃ vis = Attaga ∈ RN ×D , which encodes structure-aware token representations guided by semantic graph priors. Such a design ensures tokens from occluded
   465	regions are completed by exploiting semantic dependencies
   466	learned from non-occluded contextual regions, significantly
   467	improving the structural consistency and occlusion robustness
   468	of the learned feature representation within each modality.
   469	Since the semantic graph constrains attention propagation to
   470	semantically correlated tokens, GFCT can model multiple
   471	disjoint occluded regions in a more disentangled manner by
   472	relying on their corresponding contextual cues, and subsequent
   473	SRCM further refines the representation using complementary
   474	cross-modality information at the region level.
   475	
   476	C. Semantic-aware Region Completion Module
   477	While GFCT performs token-level completion within a
   478	single modality, it is insufficient to recover coarse-grained
   479	semantic integrity under cross-modality occlusions. To address this, we propose the Semantic-aware Region Completion Module (SRCM), which leverages complementary crossmodality information to extend the completion process to
   480	the region-level. SRCM integrates two key components: a
   481	Graph Context Perception (GCP) mechanism that embeds
   482	hierarchical structural cues into regional representations, and
   483	a Cross-modality Soft Querying (CSQ) strategy that retrieves
   484	semantically aligned features from the other modality, enabling
   485	robust bidirectional region reconstruction.
   486	Although SRCM takes the token features generated by
   487	GFCT as input, it is not a purely passive downstream stage.
   488	The initial region prototypes only provide a coarse semantic
   489	grouping of tokens, while the subsequent GCP further refines
   490	the region representations through hierarchical structural propagation. By aggregating contextual information from neighboring regions, GCP helps smooth local noise and improves the
   491	stability of region formation under occlusion. Moreover, the
   492	CSQ module introduces complementary cues from the other
   493	modality, which further compensates for unreliable region
   494	features when one modality is severely degraded. Therefore,
   495	SRCM serves as a refinement stage rather than a simple
   496	continuation of the previous module, reducing the risk of
   497	directly propagating noisy token-level errors.
   498	vis
   499	]∈
   500	Given token embeddings X̃ vis = [X1vis , X2vis , . . . , XN
   501	N ×D
   502	R
   503	from GFCT, we first compute pairwise cosine similarity to measure semantic affinities:
   504	Sij =
   505	
   506	Xivis (Xjvis )⊤
   507	,
   508	∥Xivis ∥2 · ∥Xjvis ∥2
   509	
   510	(3)
   511	
   512	where Sij denotes the semantic similarity between the i-th and
   513	j-th token.
   514	
   515	To obtain region-level prototypes, tokens with the top-K
   516	global similarity scores are selected as region centers:
   517	
   518	
   519	N
   520	X
   521	C = {c1 , c2 , . . . , cK } = top-K 
   522	Sij  ,
   523	(4)
   524	j=1
   525	
   526	where C denotes the index set of selected region prototypes.
   527	Each token is assigned to its most semantically related prototype to form regions:
   528	
   529	
   530	vis
   531	vis
   532	k = arg max Sjc , k = 1, . . . , K, (5)
   533	Gk = Xj
   534	c∈C
   535	
   536	where Gvis
   537	is the token group assigned to the k-th semantic
   538	k
   539	region. Although the region prototypes are selected based
   540	on global similarity, the token embeddings used here are
   541	already enhanced by the preceding GFCT, which encodes
   542	contextual structural information through graph-guided attention. Therefore, tokens belonging to the same semantic body
   543	region tend to exhibit more consistent similarity patterns.
   544	Furthermore, the subsequent GCP and CSQ further refine the
   545	region representations through structure propagation and crossmodality querying, which alleviates the impact of imperfect
   546	initial grouping.
   547	Then, the feature representation for each region is computed
   548	as the average of its constituent token features:
   549	X
   550	1
   551	Xjvis .
   552	(6)
   553	Rkvis = vis
   554	|Gk | vis
   555	Xj
   556	
   557	∈Gk
   558	
   559	Importantly, the region index does not imply a fixed semantic correspondence across modalities. Instead, cross-modality
   560	interactions are guided by feature similarity between region

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '300,600p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   300	Reference [27] integrated SEA and CA to capture attention
   301	in three dimensions, while HOA [51] and RGA [52] have
   302	developed attention modules that leverage complex, highorder statistical information to capture global structural details,
   303	thereby generating more discriminatively effective attention
   304	proposals. Beyond these CNN-based mechanisms, AGCL [53]
   305	introduces a graph-based attention strategy for partial Re-ID,
   306	where body keypoints and attributes are used to guide identityconsistent representation learning through structured message
   307	passing. While AGCL relies on auxiliary annotations and is
   308	designed for partial-body inputs, it shares our goal of refining
   309	semantic features for identity alignment—achieved in our case
   310	through a fully unsupervised, attention-driven module without
   311	external supervision. In this paper, we introduce the Pyramid
   312	Kernel Global Context block to refine features at every spatial
   313	position, effectively managing the balance between spatialchannel attention and computational demands. This module
   314	adeptly extracts identity-distinctive semantic information for
   315	person Re-ID tasks by incorporating spatial insights to adjust
   316	channel attention throughout the feature map, thereby enhancing performance.
   317	III. M ETHODOLOGY
   318	As depicted in Fig. 1, the proposed OPMCN framework
   319	includes three important parts, the Hardest Camera Proxy
   320	
   321	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   322	
   323	2098
   324	
   325	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   326	
   327	Fig. 1. The framework of the proposed Optimal Proxy Mining Contrastive Network (OPMCN). The feature embedding is extracted from the Pyramid Kernel
   328	Attention (PKA) Network with Pyramid Kernel Global Context (PKGC) blocks.
   329	
   330	Mining (HCPM), False Negative Proxies Mining (FNPM),
   331	and the Pyramid Kernel Attention (PKA) Network with the
   332	proposed Pyramid Kernel Global Context (PKGC) block.
   333	A. Preliminary
   334	N
   335	Given the unlabeled target dataset D = {xi }i=1
   336	and CNN
   337	backbone model F(·|θ), the embedding vector fi = F(xi |θ) ∈ Rd
   338	is extracted as the visual embedding, where N, xi , θ, d denote
   339	the number of instances, the i-th instances, the weights of the
   340	network, and the dimension of the embedding vector, respectively. The target of unsupervised person Re-Identification is
   341	to train the model F(·|θ) to get the robust and discriminative
   342	feature embedding. In each epoch, the clustering method
   343	DBSCAN [28] is adopted to cluster the training instances
   344	into Y classes based on the calculated Jaccard distance [54].
   345	By leveraging the camera ID, the camera-agnostic clustering
   346	results are split into camera-aware proxies [6] in proxybased contrastive training. At the beginning of each epoch,
   347	training instances are categorized into inliers and outliers.
   348	N0
   349	The inliers are then reassembled into set Q = {qi , ỹi }i=1
   350	and
   351	used for training, while the isolated instances are discarded
   352	in the training process to enhance the overall reliability of
   353	the clustering results, where N 0 denotes the number of inliers,
   354	ỹi = {1, . . . , Y} is the corresponding pseudo label for qi .
   355	The state-of-the-art unsupervised person Re-ID methods [6],
   356	[19] have utilized contrastive learning for optimizing the training. For each query feature q in the batch size, it is necessary to
   357	identify a positive sample and a sufficient number of negative
   358	samples to enhance the model’s ability to differentiate between
   359	similar and dissimilar semantic information [55]. To conserve
   360	space needed for storing all instances and to avoid misleading
   361	errors caused by idiosyncrasies within a class, many studies
   362	[27] initialize the representation of each class with the average
   363	
   364	features of its instances. As shown in the memory initialization
   365	part of Fig. 1, these representations are stored in a proxylevel memory, which facilitates the retrieval during contrastive
   366	learning for loss function computation. Specifically, clustering
   367	is performed on all instance features extracted from different
   368	cameras. Within each cluster, instances are further divided
   369	by camera ID. Each subset—composed of instances from the
   370	same cluster and the same camera—is used to compute a
   371	representation, referred to as a proxy.
   372	The proxy-level memory stores the central features p j
   373	calculated from all instance features within a proxy,
   374	1 X
   375	pj =
   376	qi ,
   377	(1)
   378	|P j | q ∈P
   379	i
   380	
   381	j
   382	
   383	where P j denotes the set of instance features of the j-th proxy.
   384	Many methods naturally apply the InfoNCE loss [56] for the
   385	training procedure,
   386	P
   387	exp(q · p+
   388	i /τ)
   389	,
   390	(2)
   391	L = −log PY 0
   392	j=1 exp(q · p j /τ)
   393	where p+
   394	i is the selected positive proxies corresponding to q,
   395	Y 0 is the total number of proxies, and τ is the temperature
   396	factor that controls the concentration of the similarity data
   397	distribution. In this paper, all the retrieved proxies in the
   398	corresponding cluster of q and k-nearest negative proxies
   399	are selected as contrastive pairs (i.e. positive samples and
   400	negative samples) with q [6]. The k-nearest negative proxies
   401	are selected by hard negative mining strategy [46] and its
   402	efficacy is verified by [6] and [46]. To obtain the optimal
   403	feature representation in memory, the corresponding proxy p
   404	is updated by query feature q in each batch as follows:
   405	p ← µp + (1 − µ)q,
   406	
   407	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   408	
   409	(3)
   410	
   411	CAO et al.: OPTIMAL PROXY MINING CONTRASTIVE NETWORK FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   412	
   413	2099
   414	
   415	where µ ∈ [0, 1] is the updating rate, a larger µ value
   416	implies retaining more prior information. Proxy-level memory
   417	solves the intra-class update inconsistency problem by storing
   418	central features instead of instance features, which is adopted
   419	as the baseline in the proposed OPMCN framework. In the
   420	contrastive learning stage, both the proposed Hardest Camera
   421	Proxy Mining (HCPM) and False Negative Proxies Mining
   422	(FNPM) branches operate on the proxy-level memory. They
   423	use the similarity between the query features and the stored
   424	proxies to select positive and negative pairs under different mining strategies, enabling more effective representation
   425	learning.
   426	B. Hardest Camera Proxy Mining
   427	For a given query, based on the results of clustering, several
   428	positive proxies and negative proxies can be identified. Prior
   429	methods have opted for different approaches: some [19] chose
   430	the corresponding proxy as the positive sample of the query,
   431	others [6] treated all intra-cluster proxies as positive samples
   432	with the query, while some of the proxies similar to the
   433	query but from different clusters were designated as negative
   434	samples. Positive and negative samples are each paired with
   435	the query to form contrastive pairs. Although these methods
   436	have achieved commendable results, they exhibit pronounced
   437	deficiencies. If only the proxy to which the query belongs
   438	is selected, the similarity between the corresponding proxy
   439	and the query can be excessively high. This proximity hinders
   440	effective optimization since the contrastive learning loss function, intended to push the query towards the positive sample,
   441	finds little room for improvement, relying inadequately on the
   442	differences from negative proxies. Treating all proxies within
   443	a cluster as positive samples has the advantage of uniformly
   444	learning the semantic distinctions between the query and the
   445	different proxies within the cluster. However, this approach not
   446	only increases computational complexity but also the learning
   447	of an excessive number of positive samples, which does not
   448	effectively aid the model in clearly discerning the optimal
   449	direction for semantic information enhancement. To solve the
   450	issues mentioned above and further reduce intra-class variance,
   451	we propose the Hardest Camera Proxy Mining (HCPM), which
   452	consists of two parts: the selection of the hardest camera proxy
   453	and the hard negative proxies corresponding to the hardest
   454	camera.
   455	For each query q, all proxies that belong to the same
   456	cluster as q are considered positive proxies P + , while those
   457	proxies that do not belong to the same cluster as the query are
   458	designated as negative proxies P − . Regarding positive proxies,
   459	the representation of the proxy to which q belongs typically
   460	exhibits the highest similarity to q. However, due to the
   461	inability of the backbone to perfectly extract all discriminative
   462	semantic information of the human body during training, some
   463	positive proxies may have relatively lower similarity to the
   464	query, performed as challenging intra-cluster variations. We
   465	define the hardest camera for query q as the camera whose
   466	positive proxy has the lowest similarity to q among all cameras
   467	within the same cluster. The proxy from this camera is referred
   468	to as the hardest positive proxy (HPP) and is used for mining
   469	harder positive examples during training. An illustration of
   470	
   471	Fig. 2. The contrastive pairs for the proposed hardest camera proxy mining
   472	(HCPM) intuitively highlight the distinctions between the hardest positive
   473	and regular positives, as well as hard negatives and regular negatives. In
   474	this example, the hardest camera is indicated in bold red, with each point
   475	representing a proxy, and each proxy displaying an instance.
   476	
   477	this process is provided in Fig. 2, where query and proxy
   478	features are visualized on a unit sphere. In this example, the
   479	hardest positive proxy (from camera ID 3) is highlighted in
   480	red, showing the largest angular distance to the query among
   481	all positive proxies.
   482	In terms of the collection of negative samples, this
   483	paper proposes a camera-constrained hard negative mining
   484	(CCHNM) strategy targeted at negative samples. After identifying the hardest camera, we first collect negative proxies from
   485	the hardest camera. Subsequently, we choose the top-K1 hard
   486	negative proxies (HNPs), which exhibit the highest similarity
   487	to the query, to serve as the negative samples. To summarize,
   488	the contrastive learning loss for the HCPM is defined as,
   489	LHCPM = −log
   490	
   491	exp(q · p+ /τ)
   492	,
   493	PK1
   494	exp(q · p+ /τ) + i=1
   495	exp(q · p−i /τ)
   496	
   497	(4)
   498	
   499	K1
   500	where p+ denotes the HPP, {p−i }i=1
   501	denotes the top-K1 selected
   502	HNPs, which respectively constitute the positive sample and
   503	negative samples with q. Among them, p+ is chosen by,
   504	
   505	p+ ← argmin < q, p j >, p j ∈ P + ,
   506	
   507	(5)
   508	
   509	K1
   510	= argmaxK1 {< q, p− >: p− ∈ Ph− }
   511	{p−i }i=1
   512	
   513	(6)
   514	
   515	where < · > denotes the operation that computing similarities
   516	between two representation vectors, and Ph− is defined as the
   517	subset of P − where each proxy from the hardest camera.
   518	In the proposed HCPM module, the HPP is selected as the
   519	sole prototype of the corresponding cluster, representing the
   520	positive sample. However, as illustrated in Fig. 2, the hardest
   521	positive exhibits significant differences from the query both
   522	in terms of critical human body information and background
   523	expression. The original HNM strategy does not effectively
   524	aid unsupervised training. The HNPs selected by the proposed
   525	CCHNM strategy share similar backgrounds with the HPP,
   526	particularly capturing similar angles of the human subjects.
   527	Given that the contrastive learning loss function aims to
   528	increase the similarity between the query and the positive
   529	
   530	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   531	
   532	2100
   533	
   534	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   535	
   536	Fig. 3. Example of query-based proxy selection for positive samples and
   537	negative samples in the FNPM module. The proxies with the same camera
   538	ID with the query are denoted as “Cor-cam proxies”, which are removed.
   539	
   540	samples while reducing the similarity between the query and
   541	the negative samples, CCHNM effectively assists the model
   542	in understanding the non-semantic information of background
   543	and lighting conditions, thereby accurately focusing the model
   544	on the human aspects to learn the most discriminative semantic
   545	information.
   546	C. False Negative Proxies Mining
   547	Although the HCPM module assists the model in learning
   548	the discriminative information, the training of contrastive
   549	learning still encounters interference from noise, particularly
   550	in the collection of negative samples. When the model exhibits
   551	suboptimal capabilities in extracting semantic information,
   552	false negative proxies (FNPs) and false positive proxies (FPPs)
   553	are inevitably generated during the clustering phase, both of
   554	which adversely affect the training process. To strategically
   555	tackle this problem, this paper has introduced a False Negative
   556	Proxies Mining (FNPM) loss.
   557	The FNPM proposed in this paper possesses dual capabilities: 1) it efficiently identifies unequivocal false negative
   558	proxies (FNPs) within the pool of negative proxies for use as
   559	positive samples, which inevitably enhances training performance; 2) the FNPM incorporates a filtering mechanism for
   560	false positive proxies (FPPs) when choosing positive samples.
   561	Initially, by calculating the similarity scores between the
   562	query q and all proxies P, we generate a query-based descending ranking list as shown in Fig. 3. Prior to differentiating
   563	among proxies, those sharing the same camera ID as the query
   564	are excluded, which are denoted as Pcor (In Fig. 3, these are
   565	referred to as “Cor-cam proxies”). So Pcor and the set of
   566	proxies Pexcl after excluding Pcor are noted:
   567	Pcor = {p ∈ P : camID(p) = camID(q)},
   568	
   569	(7)
   570	
   571	Pexcl = P \ Pcor ,
   572	
   573	(8)
   574	
   575	where camID(·) denotes the camera ID of a instance or proxy.
   576	The ranking list R is created based on descending similarity
   577	scores, excluding proxies with the same camera ID:
   578	R = arg sort < q, p >, p ∈ Pexcl ,
   579	
   580	(9)
   581	
   582	As depicted in Fig. 3, instances or proxies captured by the
   583	same camera typically exhibit more similar backgrounds and
   584	
   585	viewpoints of the human body, resulting in proxies p ∈ Pcor
   586	often possessing a higher similarity to q. However, this situation presents a dual challenge: for positive proxies, it increases
   587	the likelihood of selecting false positive proxies (indicated by
   588	red solid boxes), and for hard negative proxies, p ∈ Pcor tends
   589	to rank higher, thus obscuring the discovery of truly effective
   590	false negative proxies.
   591	After the removal of Pcor , the top-K2 proxies are paired with
   592	q to form positive samples set P + . Subsequently, the top-K3
   593	proxies, chosen from the remaining proxies after removing the
   594	selected K2 proxies, form negative samples set P − with q. By
   595	this means, we define the contrastive learning loss as follows:
   596	P
   597	+
   598	+ S (q, pi )
   599	1
   600	p+

codex
已读到三篇的核心方法：这一组里有明显的两类套路，一类把“缺失信息”重说成“可补全分布”，另一类把已有检索距离或代理样本选择重说成“要先找最优条件”。我继续补完后半段方法和实验闭环。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf' - | nl -ba | sed -n '360,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   360	under normal lighting conditions and under lighting condition 𝑎𝑖 . Therefore, this model can obtain
   361	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   362	
   363	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '560,920p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '590,920p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   560	interactions are guided by feature similarity between region
   561	representations rather than their indices. Even when the clustered regions are not perfectly aligned across modalities, the
   562	proposed soft-querying mechanism can still retrieve complementary information from the most semantically relevant regions, which helps alleviate the risk of rigid region mismatch.
   563	To enhance structural robustness across modalities, as
   564	shown in Fig. 4, a GCP mechanism is proposed to aggregate
   565	region-level representations from both visible and infrared
   566	modalities via hierarchical graph convolutions. It enables the
   567	model to capture semantically aligned, long-range interactions
   568	beyond modality boundaries.
   569	Let the set of region features from both modalities be
   570	vis
   571	nir
   572	concatenated as Rmix = [R1vis , . . . , RK
   573	; R1nir , . . . , RK
   574	] ∈
   575	2K×D
   576	R
   577	, where the first K rows correspond to visible modality
   578	and the remaining K rows correspond to infrared. To construct
   579	the semantic graph, the adjacency matrix is defined based
   580	on the pairwise cosine similarity between region features in
   581	Rmix , so that the edge strength reflects the semantic affinity
   582	between two region nodes. In this way, semantically related
   583	regions are assigned stronger connections, while weakly related regions contribute less to graph propagation. Based on
   584	this similarity matrix, the corresponding normalized symmetric Laplacian L ∈ R2K×2K is constructed to model both
   585	intra-modality and inter-modality structural relations. This
   586	similarity-based graph construction is more suitable than a
   587	predefined spatial topology, since the goal of GCP is to capture
   588	semantic dependencies across visible and infrared regions.
   589	
   590	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
   591	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   592	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   593	
   594	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   595	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
   596	
   597	6
   598	
   599	where expand(Ω(p) ) ∈ R2K×dp denotes the expanded perceptual weights aligned to the shape of Z (p) , enabling independent
   600	weighting across feature channels. The fused features are then
   601	split back into modality-specific sets:
   602	
   603	mix
   604	2 K d 0
   605	
   606	2 K d 2
   607	
   608	2 K d1
   609	
   610	mix
   611	Rvis
   612	[1 : K],
   613	g = Rg
   614	
   615	Z (1)
   616	
   617	Z (0)
   618	
   619	 0 ( Z (0) )
   620	
   621	 2 (Z ( 2) )
   622	
   623	Order-wise Softmax Weighter
   624	
   625	 (0)
   626	
   627	 (1)
   628	
   629	mix
   630	g
   631	
   632	K ( d0 + d1 + d 2 )
   633	
   634	 (2)
   635	
   636	K ( d0 + d1 + d 2 )
   637	
   638	vis
   639	g
   640	
   641	(12)
   642	
   643	which serve as structurally enriched region representations for
   644	subsequent bidirectional querying. Through GCP, each region
   645	perceives both intra- and inter-modality semantic structures at
   646	multiple receptive field scales, forming a robust foundation for
   647	region-level feature reconstruction under occlusion.
   648	The CSQ strategy is further designed to retrieve semantic cues from the complementary modality, facilitating cross-modality feature alignment. Let Rnir
   649	=
   650	g
   651	nir
   652	nir
   653	nir
   654	{Rg1
   655	, Rg2
   656	, . . . , RgK
   657	} denote a cross-modality region memvis
   658	ory bank for CSQ, allowing each visible region Rgk
   659	∈ Rvis
   660	g
   661	to retrieve semantic structure from the bank via soft querying.
   662	Specifically, we compute the attention weights as:
   663	
   664	√ 
   665	vis
   666	nir
   667	exp Φ(Rgk
   668	) · Ψ(Rgj
   669	)/ d
   670	
   671	(13)
   672	αkj = P
   673	√ ,
   674	K
   675	vis ) · Ψ(Rnir )/ d
   676	exp
   677	Φ(R
   678	gt
   679	t=1
   680	gk
   681	
   682	Z (2)
   683	
   684	 1 ( Z (1) )
   685	
   686	Rnir
   687	= Rmix
   688	[K + 1 : 2K],
   689	g
   690	g
   691	
   692	nir
   693	g
   694	
   695	Fig. 4: Illustration of Graph Context Perception module.
   696	Rknir∗ =
   697	
   698	K
   699	X
   700	
   701	nir
   702	αkj · Υ(Rgj
   703	),
   704	
   705	(14)
   706	
   707	j=1
   708	
   709	For each graph order p ∈ {0, 1, 2}, the corresponding graph
   710	convolutional representation is computed as:
   711	
   712	p
   713	Z (p) = σ (L) Rmix Wz ,
   714	(7)
   715	where Wz ∈ RD×dp is a shared learnable projection matrix,
   716	σ(·) denotes the ReLU activation, and Z (p) ∈ R2K×dp
   717	captures features perceived under the p-th order graph neighborhood.
   718	To unify the feature dimensions and capture nonlinear
   719	interactions across different orders, we apply a set of orderspecific transformation functions ψp (·), each instantiated as a
   720	shallow MLP:
   721	n
   722	o
   723	S = ψ0 (Z (0) ), ψ1 (Z (1) ), ψ2 (Z (2) ) ∈ R2K×3 ,
   724	(8)
   725	where ψp (·) : R2K×dp → R2K×1 compresses the corresponding order’s representation to a scalar importance score per
   726	node. These scores are normalized across the three orders via
   727	a softmax operation:
   728	n
   729	o
   730	Ω = softmax(S) = ω (0) , ω (1) , ω (2) ,
   731	(9)
   732	o
   733	n
   734	(p)
   735	(p)
   736	(p)
   737	(10)
   738	ω (p) = ω1 , ω2 , · · · , ω2K ,
   739	where each row of Ω reflects relative contribution
   740	order
   741	P2 of each
   742	(p)
   743	for the corresponding i-th node, satisfying p=0 ωi = 1.
   744	Finally, the multi-order representations are fused through a
   745	weighted concatenation mechanism:
   746	Rmix
   747	=
   748	g
   749	
   750	2
   751	X
   752	p=0
   753	
   754	expand(Ω(p) ) · Z (p) ,
   755	
   756	(11)
   757	
   758	where Φ(·), Ψ(·), and Υ(·) are learnable linear projections
   759	for query, key, and value transformation, respectively. αkj
   760	denotes the normalized attention weight assigned to the jth infrared region when reconstructing the k-th visible region,
   761	and the denominator performs softmax normalization over all
   762	K candidate infrared regions.
   763	To reconstruct modality-aligned features, the retrieved
   764	cross-modality representation Rknir∗ is projected into the visible feature space using a two-layer MLP:
   765	vis
   766	Rck
   767	= fθ (Rknir∗ ) = W2 (ReLU(W1 Rknir∗ + b1 )) + b2 . (15)
   768	
   769	Subsequently, to preserve modality-specific identity, the
   770	original visible region feature Rvis is fused with the reconvis
   771	vis
   772	vis
   773	, Rc2
   774	, . . . , RcK
   775	structed feature Rvis
   776	= {Rc1
   777	} as:
   778	c
   779	R̂vis = Fuse(Rvis , Rvis
   780	c ),
   781	
   782	(16)
   783	
   784	where Fuse(·) denotes concatenation followed by a linear
   785	transformation.
   786	Finally, all enhanced region features are concatenated to
   787	generate the complete region-level visible representation:
   788	vis
   789	X̂ vis = [R̂1vis ; R̂2vis ; . . . ; R̂K
   790	] ∈ RK×D .
   791	
   792	(17)
   793	
   794	In summary, SRCM enhances region-level feature completion by clustering semantic tokens, embedding structural priors
   795	through GCP, and leveraging cross-modality alignment via
   796	CSQ. This design enables robust, structure-aware reconstruction under severe occlusions and modality discrepancies.
   797	
   798	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
   799	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   800	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   801	
   802	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   803	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
   804	
   805	7
   806	
   807	D. Multi-distribution Consistency Alignment Loss
   808	While GFCT and SRCM provide token-level and regionlevel completed features respectively, the resulting representations may exhibit inconsistencies due to incomplete supervision or modality gap. To promote semantic coherence
   809	and structural consistency among different completion paths,
   810	we introduce the Multi-distribution Consistency Alignment
   811	(MCA) loss. This loss function aligns two types of completed
   812	features with a unified pseudo-holistic reference through symmetric distribution matching.
   813	To obtain a unified pseudo-holistic reference that reflects
   814	common semantics across multiple completion paths, we compute the average of the token-level and region-level features
   815	produced by GFCT and SRCM, respectively. Specifically, the
   816	reference is defined as the element-wise mean of X̃ vis and
   817	X̂ vis , which denote the completed visible features from two
   818	completion pathways, respectively. This averaging operation
   819	ensures that both local details and semantic structural cues
   820	contribute equally to the reference representation, while maintaining alignment in feature dimensionality.
   821	We then project all feature types into a probability space
   822	using softmax normalization over the feature dimension:
   823	vis
   824	Ptoken
   825	= softmax(X̃ vis ),
   826	vis
   827	Pregion
   828	= softmax(X̂ vis ),
   829	
   830	(18)
   831	
   832	vis
   833	Pholistic
   834	= softmax(H vis ).
   835	
   836	where X̃ vis and X̂ vis are the completed token- and regionlevel features, and H vis is the pseudo-holistic reference.
   837	To align these representations, the MCA loss computes pairwise symmetric Kullback–Leibler (KL) divergence between
   838	the distributions:
   839	vis
   840	vis
   841	vis
   842	vis
   843	Lmca = DKL (Ptoken
   844	∥Pregion
   845	) + DKL (Pregion
   846	∥Ptoken
   847	)
   848	vis
   849	vis
   850	vis
   851	vis
   852	+ DKL (Ptoken
   853	∥Pholistic
   854	) + DKL (Pregion
   855	∥Pholistic
   856	),
   857	(19)
   858	where each divergence term is defined as:
   859	X
   860	DKL (P |Q) =
   861	Pi · (log(Pi + ε) − log(Qi + ε)) , (20)
   862	i
   863	
   864	where ε is a small positive constant introduced to prevent
   865	numerical instability. By reducing feature inconsistency and
   866	promoting structural coherence, MCA enables the learning
   867	of more discriminative and semantically complete representations, especially in the presence of severe occlusions or
   868	cross-modality variance. Fig. 2 explicitly annotates the three
   869	alignment terms in MCA, which correspond directly to the
   870	token–region, token–holistic, and region–holistic alignment
   871	terms in Eq. (19), respectively.
   872	To jointly optimize semantic completeness, structural consistency, and identity discriminability, we employ a combination of three loss functions: the identity loss Lid , the tripletcenter loss Ltc , and the proposed MCA loss Lmca . The overall
   873	training objective is defined as:
   874	Ltotal = Lid + αLtc + λLmca ,
   875	
   876	(21)
   877	
   878	where α and λ are balancing coefficients that control the
   879	relative contributions of the triplet-center loss and the MCA
   880	loss, respectively.
   881	
   882	IV. O CCLUDED VI-R E ID DATASETS
   883	
   884	Occluded-SYSU-MM01
   885	
   886	Occluded-RegDB
   887	
   888	Fig. 5: Examples from the our constructed Occluded-SYSUMM01 and Occluded-RegDB dataset.
   889	Most existing approaches to modeling occlusion in ReID
   890	rely on random block erasing or cropped patches [6], [24]–
   891	[28], which results in limited semantic information and lacks
   892	spatial consistency. Therefore, we establish two occlusionaware benchmarks, Occluded-SYSU-MM01 and OccludedRegDB, based on their original versions [29], [30]. As shown
   893	in Fig. 5, these datasets introduce semantically plausible
   894	occlusions into pedestrian images.
   895	Instead of inserting occlusions arbitrarily, the occlusion
   896	generation is modeled as a spatially constrained process. In
   897	surveillance scenes, many occlusions naturally appear near the
   898	ground plane, while smaller portable items may occur across a
   899	wider spatial region. Based on this observation, occlusions are
   900	placed according to spatial compatibility with the pedestrian
   901	layout. As illustrated in Fig. 6, given a visible image X vis ,
   902	an occlusion is resized and pasted at a location l based on
   903	its spatial prior to produce ovis
   904	γ,l with a scale factor γ, where
   905	mγ,l denotes the corresponding binary mask. The resulting
   906	occluded image is denoted by Xavis , and a small horizontal
   907	offset is further added to avoid overly rigid layouts and to
   908	simulate variations in viewpoint.
   909	The occlusions are cropped from COCO [31], OccludedDuke [32], and Occluded-ReID [33], by a pre-trained Mask RCNN [34]. It contains 1010 manually selected object instances
   910	across 21 categories such as car, umbrella, and billboard,
   911	each annotated with spatial priors (e.g., bottom-aligned or
   912	freely placed), as shown in Fig. 7. To maintain semantic
   913	consistency, the occlusions are filtered by the environment
   914	label of the source camera, so that outdoor views preferentially
   915	use outdoor-compatible occlusions, while indoor views avoid
   916	semantically inconsistent objects.
   917	Although SYSU-MM01 and RegDB are both widely used
   918	in the VI-ReID setting, their infrared imagery is not identical
   919	in appearance. Occluded-SYSU-MM01 is closer to a nearinfrared (NIR) imaging setting, whereas Occluded-RegDB
   920	corresponds to thermal infrared imagery that typically exhibits

 succeeded in 0ms:
   590	false negative proxies.
   591	After the removal of Pcor , the top-K2 proxies are paired with
   592	q to form positive samples set P + . Subsequently, the top-K3
   593	proxies, chosen from the remaining proxies after removing the
   594	selected K2 proxies, form negative samples set P − with q. By
   595	this means, we define the contrastive learning loss as follows:
   596	P
   597	+
   598	+ S (q, pi )
   599	1
   600	p+
   601	i ∈P
   602	,
   603	LFNPM = − + log P
   604	P
   605	|P |
   606	S (q, p+
   607	S (q, p−j )
   608	i )+
   609	+
   610	p+
   611	i ∈P
   612	
   613	(10)
   614	
   615	p−j ∈P −
   616	
   617	in which S (q, p) = exp(q· p/τ), where |P + | = K2 is the number
   618	of selected positive samples.
   619	During each training iteration, a batch is formed by selecting
   620	P×K instances, drawing K instances from each of P randomly
   621	chosen proxies. Initially, it is inevitable that within the K
   622	queries from a given proxy, some may exhibit relatively
   623	weaker semantic expressions. This can impair the effectiveness
   624	of a query-based ranking list. To mitigate the adverse impact
   625	of such individual queries, we have devised an anti-noise
   626	ranking criterion. This method is straightforward yet effective:
   627	it involves summing the similarity ranking lists of queries
   628	from the same proxy with those from all other proxies,
   629	thereby generating a ranking list with enhanced robustness.
   630	The contrastive loss functions in the HCPM module and
   631	FNPM module mentioned above consistently optimize the
   632	Optimal Proxy Mining Contrast (OPMC).
   633	
   634	D. Pyramid Kernel Global Context Block
   635	Although the proposed OPMC learning has significantly
   636	aided the model in grasping the semantic information essential
   637	for person Re-Identification and has been effective in reducing
   638	intra-class variance while increasing inter-class variance, they
   639	still fall short in fully focusing the network’s attention on
   640	the individual. Consequently, we have strategically designed a
   641	Pyramid Kernel Global Context (PKGC) block modified from
   642	[50] to establish a Pyramid Kernel Attention (PKA) network
   643	for addressing this issue.
   644	For an input feature map F ∈ RC×H×W , we compute the
   645	global context modeling with a Pyramid kernel (PK) convolution WPK and softmax function to obtain the attention weights.
   646	Through the Softmax operation, the model can emphasize
   647	or suppress certain features based on the computed weights,
   648	enabling the network to focus on the features most critical to
   649	the current task. Then a two-layer bottleneck incorporating
   650	an intermediate layer normalization is utilized to capture
   651	channel-wise dependencies, where r denotes the bottleneck
   652	ratio. Hadamard addition is applied as the fusion function
   653	to aggregate the global context features to the feature of
   654	
   655	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   656	
   657	CAO et al.: OPTIMAL PROXY MINING CONTRASTIVE NETWORK FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   658	
   659	each channel. The detailed architecture of the PKGC block
   660	is formulated in Eq. 11.
   661	0 0
   662	11
   663	HW
   664	W1 x j
   665	X
   666	e
   667	z = x + WPK ReLU @LN @W2
   668	PHW W x xAA , (11)
   669	1 m
   670	m=1 e
   671	j=1
   672	where x and z denotes the input and output of the PKGC block,
   673	j enumerated all possible positions, W1 and W2 are weights of
   674	two 1 × 1 convolution in two-layer bottleneck, LN denotes the
   675	layer normalization.
   676	In terms of the PK convolution before computing the
   677	softmax, the kernel size of the convolution is variant depending
   678	on the size of the feature map after each stage in CNN
   679	backbone [31] (see PKA Network sugfigure of Fig. 1).
   680	The PK convolution compresses the channel dimension and
   681	reduces computational complexity, providing a suitable feature representation for the subsequent softmax-based spatial
   682	global attention weighting. In Re-ID tasks, identity-distinctive
   683	features are usually not very small spatially (referring to
   684	H × W dimensions), so for the shallower stages where feature
   685	maps are larger, using a larger kernel size helps cover more
   686	extensive areas. The proposed PK convolution facilitates the
   687	comprehensive learning of discriminative features and contextual relationships, aiding the network in grasping large-scale
   688	visual patterns. In deeper stages, where feature maps are
   689	smaller and primarily provide semantical rather than purely
   690	visual information, smaller kernel sizes are used to capture
   691	crucial semantical details. Therefore, the proposed PKGC
   692	block not only enhances overall performance in tasks with
   693	complex backgrounds and rich details but also maximizes
   694	feature extraction and contextual understanding at each stage,
   695	thus optimizing the model’s generalization capabilities.
   696	E. Optimization
   697	This paper employs an iterative process where it alternates
   698	between clustering part and updating part, i.e., both the modified backbone network and the memory, using the HCPM and
   699	the FNPM losses. The entire loss used for model optimization
   700	is
   701	L = λ × LHCPM + (1 − λ) × LFNPM .
   702	(12)
   703	To enable a better understanding of the proposed Optimal
   704	Proxy Mining Contrastive Network (OPMCN), we summarize
   705	the overall procedure in Algorithm 1.
   706	IV. E XPERIMENTS
   707	A. Experimental Settings
   708	1) Datasets: The proposed method is evaluated on four
   709	person Re-ID datasets: Market-1501 [32], MSMT17 [33],
   710	PersonX [34], and VeRi-776 [57], we adopt the official settings
   711	to split them for training and testing.
   712	• Market-1501 dataset comprises 32,668 images captured
   713	by 6 cameras on the Tsinghua campus. It is divided into
   714	a training set with 12,936 images of 751 identities and a
   715	testing set with 19,732 images of 750 identities.
   716	• MSMT17, recognized as the most challenging dataset,
   717	features 126,442 images from 15 cameras. It includes
   718	
   719	2101
   720	
   721	Algorithm 1 Optimization Process of Optimal Proxy Mining
   722	Contrastive Network
   723	N
   724	Require: Training set D = {xi }i=1
   725	; Backbone model
   726	F(xi |θ) with PKGC blocks, the epoch number
   727	num epochs, the number of batches num batch
   728	temperature τ for Eq. (3) and Eq. (6), update rate µ for
   729	Eq. (2).
   730	Ensure: Optimized encoder F(xi |θ).
   731	1: for epoch in [1, num epoch] do
   732	2:
   733	Extract fi through F(xi |θ);
   734	3:
   735	Calculate Jaccard distance with the extracted fi , cluster
   736	DT into C clusters to get pseudo label dataset Q and
   737	initialize the proxy-level memory bank M.
   738	4:
   739	for b=1 to num batch do
   740	5:
   741	Sample each mini-batch B = {qi , ỹi }
   742	6:
   743	Calculate HCPM contrastive loss by Eg. (4);
   744	7:
   745	Calculate FNPM contrastive loss by Eg. (10);
   746	8:
   747	Update the model F by back-propagation;
   748	9:
   749	Update M.
   750	10:
   751	end for
   752	11: end for
   753	
   754	32,621 training images of 1,041 identities and 93,821 test
   755	images of 3,060 identities.
   756	• PersonX provides 45,792 synthetic images of 1,266 identities under 6 camera views, created using Unity [58].
   757	It utilizes 9,840 images for training 410 identities, while
   758	for testing, it employs 5,136 images as query images and
   759	30,816 images as gallery images for 856 identities.
   760	• The VeRi-776 dataset contains 51,003 images recorded by
   761	20 different cameras, representing 775 unique vehicles.
   762	For evaluation, 37,746 images of 575 vehicles are designated for training, 1,678 images for queries, and 11,579
   763	images as gallery images during testing.
   764	For performance evaluation, the commonly used mean Average Precision (mAP) and Cumulative Matching Characteristic
   765	(CMC) are adopted as the metrics. The CMC metric is reported
   766	via Rank-1, Rank-5, and Rank-10.
   767	2) Implementation Details: The modified ResNet-50 [31] is
   768	utilized as the backbone network, which replaces the final fully
   769	connected layer with a generalized mean pooling [59] and a
   770	Batch Normalization. The network is pre-trained on ImageNet
   771	[60]. The L2 norm is used when computing distance in the
   772	clustering process and testing. We resize each instance from
   773	three person Re-ID datasets to 256 × 128 and each vehicle
   774	instance from Veri-776 to 224 × 224.
   775	The hyper-parameters involved in our model are empirically
   776	set as follows. The memory updating rate µ, temperature factor
   777	τ are set as 0.2 and 0.07, respectively. The number of negative
   778	samples in HCPM module, i.e. K1 is fixed to 20. The number
   779	of positive samples and negative pairs in the FNPM module,
   780	i.e. K2 and K3 are set to 2 and 30 for both Market-1501
   781	and MSMT17, 5 and 50 for both PersonX and VeRi-776. We
   782	calculate Jaccard distance [54] and employ DBSCAN [28] for
   783	clustering, the thresholds and neighborhood eps are set as 0.5
   784	and 4. In terms of the PKGC block, we set the kernel of 4
   785	
   786	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   787	
   788	2102
   789	
   790	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   791	
   792	TABLE I
   793	A BLATION S TUDY ON THE C OMPONENTS OF THE P ROPOSED M ODEL . I N PARTICULAR , HCPM R EFERS TO H ARDEST C AMERA P ROXY M INING M ODULE ;
   794	FNPM D ENOTES FALSE N EGATIVE P ROXIES M INING M ODULE ; PKGC M EANS P YRAMID K ERNEL G LOBAL C ONTEXT B LOCK
   795	
   796	blocks after each stage as {7, 5, 3, 1}, and ratio r is set to 4
   797	following [50].
   798	Additionally, for training settings and devices, random
   799	flipping, cropping, and erasing for data augmentation are
   800	employed during the training process. The ADAM [61] with
   801	β1 = 0.9, β2 = 0.999, and weight decay of 5e-4 is adopted as
   802	the optimizer. The learning rate is initially set to 3.5e-4 with
   803	a warm-up in the first 10 epochs and is divided by 10 after
   804	20 epochs. The number of total training epochs is set to 50
   805	and the number of iterations in each epoch is set to 400. The
   806	proposed method is implemented by PyTorch Toolbox [62]
   807	and experiments on a single GeForce GTX 3090 GPU and
   808	Intel Core i7-12700 processor 6.3 GHz.
   809	
   810	TABLE II
   811	A BLATION S TUDY ON D IFFERENT C HOICE OF P OSITIVE S AMPLE AND THE
   812	E FFECTIVENESS OF O UR C AMERA C ONSTRAINED H ARD N EGATIVE
   813	M INING (CCHNM) ON M ARKET-1501 AND MSMT17
   814	
   815	TABLE III
   816	A BLATION S TUDY ON D IFFERENT VARIANTS . “ANR” D ENOTES THE
   817	O PERATION OF A PPLYING A NTI -N OISE R ANKING
   818	IN THE FNPM M ODULE
   819	
   820	B. Ablation Study
   821	To confirm the effectiveness of different components in the
   822	proposed OPMC framework, a series of ablation experiments
   823	is demonstrated on Market-1501 [32], MSMT17 [33], VeRi776 [57] datasets. The method [6] applied all positive proxies
   824	as positive samples is adopted as the baseline model. The
   825	ablation results are listed in Table I. It should be noted that
   826	(a) uses the baseline loss strictly defined in Section iii. A
   827	(Eq. 2), whereas (b)-(f) are based on a unified training setup
   828	consistent with the full OPMCN framework, ensuring comparability across ablation variants. Compared to the baseline,
   829	the performance of the proposed full model is remarkably
   830	improved on three datasets. It improves the baseline by a large
   831	marge of 5.0% in mAP and 2.9% in Rank-1 on Market-1501,
   832	17.6% in mAP and 17.9% in Rank-1 on MSMT17, and 10.8%
   833	and 8.1% in Rank-1 on VeRi-776.
   834	1) Effectiveness of Hardest Camera Proxy Mining: The
   835	proposed HCPM aims at learning semantic information while
   836	reducing intra-class variance. By comparing the results of
   837	variant (a) and variant (b) in Table I, the effectiveness of the
   838	proposed HCPM module is obvious. Within HCPM, variant
   839	(b) promotes the performance over baseline with 2.5% in
   840	mAP and 1.8% in Rank-1 on Market-1501, 10.6% in mAP
   841	and 13.2% in Rank-1 on MSMT17, and 9.2% in mAP and
   842	2.0% in Rank-1 on VeRi-776, respectively. In the proposed
   843	HCPM module, the hard positive proxy (HPP) is selected as
   844	the positive sample when computing Eq. (4), and cameraconstrained hard negative mining (CCHNM) is utilized to
   845	collect the special hard negative proxies as the negative
   846	samples.
   847	To further verify the effectiveness of applying HPP in the
   848	HCPM module, we adopt several choices when selecting the
   849	
   850	positive sample. As shown in Table II, the results for 1)
   851	“Cor”: the corresponding proxy is used as a positive sample
   852	for contrastive loss Eq. (4). 2) “All”: all the proxies belong
   853	to the corresponding cluster are used as positive samples
   854	for q when computing the loss. 3) “HPP”: the hard positive
   855	proxy is chosen as the positive sample. In this comparison
   856	group, we set K1 = 50 (follow [6]) as the hyperparameter
   857	for hard negative proxy mining to fairly compare the methods
   858	that utilize different proxies as positive samples. The results
   859	also demonstrate that, under identical experimental conditions,
   860	merely selecting the corresponding proxy as the positive
   861	sample does not sufficiently enhance the unsupervised training.
   862	Furthermore, to prove the superiority of the proposed
   863	CCHNM in unsupervised learning, the comparison between
   864	experimental groups (c) and (d) in Table II is shown to
   865	demonstrate that using the proposed CCHNM allows “HPP”
   866	to achieve superior performance enhancements over “ALL”,
   867	further substantiating the efficacy of HCPM learning. This
   868	experiment confirms the central premise of HCPM: proxies
   869	from the hardest camera settings offer optimal semantic differentiation for unsupervised re-ID.
   870	2) Effectiveness of False Negative Proxies Mining: From
   871	variant (b) vs. variant (c) in Table I, we observe that the models
   872	additionally integrated with the FNPM module can boost the
   873	performance by a considerable margin of 0.7% in mAP and
   874	
   875	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   876	
   877	
codex
前三篇的消融线索也已经出现了，尤其是 OPMCN 把“最难相机代理”和“假负代理”拆成了能单独验证的命题。下面读剩下三篇的方法区。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '250,720p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   250	properties of translation, scale, distortion invariance, as well
   251	as the hierarchical structures found in CNN convolution
   252	operations. Addressing this limitation, [30] combines a CNN
   253	backbone to extract hierarchical features and harnesses the
   254	global interaction characteristics of Transformers to aggregate features of different scales from a global perspective.
   255	Another notable approach, PAT [31], employs CNNs as a
   256	backbone model and incorporates a transformer encoder to
   257	extract additional features. It utilizes cross-attention with a
   258	group of learnable part prototypes to selectively mask specific
   259	features. Similarly, our proposed method also leverages the
   260	fusion of CNN and transformer encoder. However, what sets
   261	our method apart is that our method is the incorporation of
   262	local skeletal regions of the human body, which allows for
   263	the construction of a comprehensive set of local features.
   264	Furthermore, our method utilizes the local region weight map
   265	derived from the joint mask to regulate the cross-attention
   266	fusion process between the backbone network’s features and
   267	the local features. As a result, we achieve enhanced discriminative local features and alignment of high-level semantic
   268	features.
   269	
   270	III. M ETHODOLOGY
   271	In this section, we present the implementation details of our
   272	PSCR method for occluded person re-id, as shown in Fig. 2.
   273	An overview of the proposed method is introduced in Section
   274	III-A. Each component of our method is then described in
   275	detail in Sections III-B to Section III-D.
   276	
   277	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   278	
   279	GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION
   280	
   281	8601
   282	
   283	A. Framework
   284	Structurally, the proposed PSCR framework comprises three
   285	key components: the VARA module, the SAM module, and the
   286	SMCAF module.
   287	1) VARA Model: The VARA module constitutes the foundational layer of the PSCR framework, with its core objective
   288	being to suppress occlusion interference and enhance the
   289	model’s focus on visible regions through dual attention mechanisms. This module comprises two critical sub-components:
   290	Foreground Perception Channel Attention Sub-module:
   291	Establishes foreground-correlated attention mechanisms based
   292	on keypoint heatmaps. It implements channel-level occlusion
   293	suppression through multi-level weighted feature mapping,
   294	assigning high weights to visible regions and low weights to
   295	occluded areas.
   296	Key Feature Space Cross-Attention Fusion Sub-module:
   297	Constructs a context-aware mechanism in the spatial domain
   298	without channel compression, dynamically adjusting feature
   299	representations to enhance the salience of visible body parts
   300	while diminishing the impact of occluded regions.
   301	By implementing soft fusion operations on intermediate layer features, VARA effectively differentiates between
   302	occluded and visible features, alleviating the problem of
   303	occlusion information diffusion caused by expanding receptive
   304	fields in the backbone network. The clean and precise feature
   305	representations generated by VARA lay a solid foundation for
   306	subsequent modules.
   307	2) SAM Model: Building upon the clean feature representations provided by VARA, the SAM module achieves
   308	fine-grained localization of pedestrian regions using pose
   309	information. It employs a semantics-driven partitioning strategy to construct feature radiation weight masks centered
   310	around skeletal joints, based on spatial pose relationships and
   311	confidence features of keypoints.
   312	Through enhanced precision in local feature extraction,
   313	SAM provides richer semantic information for subsequent
   314	cross-attention fusion. This module bridges VARA’s global
   315	representations with local detail extraction, ensuring simultaneous capture of pedestrians’ holistic morphology and
   316	microscopic characteristics.
   317	3) SMCAF Model: As the final feature integration layer
   318	of the framework, the SMCAF module dynamically captures semantic correlations between VARA’s global features
   319	and SAM’s local features through cross-modal attention
   320	mechanisms. Its fusion process employs a spatial attention
   321	mask-guided strategy that strengthens feature responses in critical regions while suppressing occlusion-induced uncertainties.
   322	The high-level semantic alignment fusion mechanism
   323	enables SMCAF to effectively enhance local feature expressiveness and eliminate occlusion noise. The resulting composite feature representation combines global discriminability
   324	with local specificity, providing robust foundations for similarity computation in re-identification tasks.
   325	B. Visible Appearance Region Attention
   326	To enhance the attention of the backbone network towards
   327	visible areas, mainstream methods often use pose features
   328	
   329	Fig. 3. The detailed internal structure of proposed VARA module.
   330	
   331	directly fused with the output features of the backbone network
   332	[8], [10], [12]. However, the receptive field of spatial units
   333	in the feature map will continuously expand as the network
   334	grows deeper, causing non-occluded unit receptive fields in
   335	the output layer of the backbone network to cover occluded
   336	areas, leading to occlusion information being transmitted to
   337	originally non-occluded areas and making it difficult for the
   338	algorithm to decouple occlusion information.
   339	To address this, we establish a VARA guided by pose
   340	semantic information, and embeds it into ResNet-50. VARA
   341	is embedded in different layers of the ResNet-50 backbone network, constructing a pose attention guided soft
   342	fusion mechanism, driving the backbone network to learn
   343	the differences between occlusion information and pedestrian
   344	information in the intermediate layer and adaptively suppressing the expression of occlusion information. Compared with
   345	the conventional methods based on pose estimation, VARA
   346	can effectively alleviate the problem of occlusion information
   347	diffusion produced by the expansion of receptive fields through
   348	a multi-layer soft fusion mechanism.
   349	Given that pose keypoint information presents dependable
   350	spatial positional relationships within localized regions of
   351	the human body, we adopt the strategy of channel attention
   352	followed by spatial attention, which helps suppress identityunrelated channels and enhances the reliability of spatial-level
   353	feature fusion. As illustrated in Fig. 3, the backbone network
   354	in our architecture establishes a sequential connection between
   355	the Conv.B2 and Conv.B3 layers, each linked to a respective
   356	VARA module. Fig. 3 provides an explicit depiction of the
   357	intricate internal structure of the VARA module. The comprehensive description of the structure and principles governing
   358	the two distinct sub-modules is presented in the subsequent
   359	sections.
   360	1) Foreground-Aware Channel Attention Sub-Module: The
   361	interference caused by background or occlusion information
   362	poses a significant challenge to pedestrian re-identification.
   363	Consequently, effectively suppressing such interference has
   364	become a crucial research topic in the field. State-of-the-art
   365	methods have explored the utilization of attention mechanisms
   366	to enhance the focus of the model on the visible regions
   367	of pedestrians [12], [28], [32]. However, traditional attention
   368	mechanisms typically use global average pooling (GAP) to
   369	compress the spatial dimension of the feature map into a
   370	
   371	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   372	
   373	8602
   374	
   375	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025
   376	
   377	channel signal vector, where the weights of the feature distributed in the spatial dimension are equal. Obviously, the
   378	representation of channel importance obtained by GAP is not
   379	an optimal choice [4], [8]. If the influence of each spatial pixel
   380	is deemed same, the attention mechanism may be interfered
   381	by cluttered background or occlusion information, resulting
   382	in an inability to provide accurate information that facilitates independent modeling between channels. To address this
   383	limitation, in VARA, we design a foreground-aware channel
   384	attention sub-module, whose core is to first encode the features
   385	of the foreground area using the keypoint heat map, calculate
   386	the attention scores of each channel related to the foreground,
   387	and then perform channel weighting on the input feature map
   388	of VARA to suppress the feature responses of channels related
   389	to occlusion and background noise.
   390	Specifically, a ResNet-50 network pre-trained on the ImageNet dataset is implemented as the CNN backbone and is
   391	slightly modified by removing the average pooling layer and
   392	fully connected layer from the end, resulting in a feature
   393	map F ∈ RH×W×C with spatial information, where H, W,
   394	and C represent the height, width, and number of channels,
   395	respectively [18]. The internal structure of the ResNet-50
   396	network can be divided into five parts, defined as conv1 and
   397	conv.B1, conv.B2, conv.B3, and conv.B4, each consisting of
   398	Bottleneck blocks, with 3, 4, 6, and 3 blocks in the latter 4
   399	parts, respectively, and each block containing 3 convolutional
   400	layers.
   401	The feature map inputted to VARA is defined as F ∈
   402	RHin ×Win ×Cin , where Hin ,Win and Cin represent the height,
   403	width, and number of channels of the inputted feature map,
   404	respectively. The inputted heatmap containing 14 key points
   405	is defined as Fk ∈ RHin ×Win ×1 , where the spatial size of the
   406	key point heatmap is consistent with the spatial size of a
   407	single channel in the inputted feature map. First, as shown
   408	in Fig. 3, the nonzero region of the keypoint heatmap Fk
   409	is globally average pooled down-sampled to attain a channel
   410	signal vector Vc ∈ R1×1×Cin of features that correspond to the
   411	nonzero region, which can be represented as:
   412	Vc = AvgPool(Fin
   413	
   414	Fk )
   415	
   416	(1)
   417	
   418	where, denotes the element-wise product of feature maps,
   419	and AvgPool(·) denotes a global average pooling operation.
   420	Second, two different convolution layers {Cin /r, 1 × 1, 1}and
   421	{Cin , 1 × 1, 1} are connected, and ReLU and Sigmoid activation functions are added after each convolution layer, to
   422	implement a Squeeze-and-Excitation-like [33] attention mechanism. The computation process of the channel attention vector
   423	Ac ∈ R(1×1×Cin ) is described as:
   424	Ac = sigmoid(W2 relu(W1 Vc ))
   425	
   426	(2)
   427	
   428	where, W1 ∈ R((Cin /r)×Cin ) and W2 ∈ R(Cin ×(Cin /r)) represent the
   429	weight parameters of the two convolution layers, respectively.
   430	The dimension reduction ratio r is set to 8. Finally, the
   431	generated channel score vector Ac acts on the input feature
   432	map Fin in a product-wise manner, producing a channelweighted feature map Fd ∈ RHin ×Win ×Cin . Compared to directly
   433	using global average pooling on the feature map, our method
   434	utilizes prior semantic knowledge of key points to filter out
   435	
   436	irrelevant information distributed in space. Driven by the channel attention mechanism, different convolution kernels (each
   437	corresponding to a channel of feature map) of the convolution
   438	layer can make adaptive decisions on the importance of the
   439	extracted patterns, thereby improving the model’s ability to
   440	perceive outlier information.
   441	2) Key Feature Spaces Fusion Sub-Module: The
   442	foreground-aware sub-module filters out channel information
   443	to facilitate feature extraction. Similarly, a Key Feature
   444	Space Cross-Attention Fusion Sub-Module is purposefully
   445	designed to enhance the model’s attention towards the
   446	visible contextual area at a spatial level, without resorting
   447	to channel compression. This architectural design enables
   448	further improvements in the input feature maps and promotes
   449	enhanced performance.
   450	Firstly, the corresponding feature vectors are extracted based
   451	on the heatmaps of 14 key points [34], which are organized
   452	into a sparse feature map F s ∈ R1×14×Cin . The output of the
   453	previous foreground aware sub-module, Fd ∈ RHin ×Win ×Cin ,
   454	weighted by attention, is viewed as a dense feature map.
   455	Secondly, to achieve linear transformation of vector features
   456	and reduce the computational complexity in the attention calculation process, Fd is channels-reduced by a 1×1 convolution
   457	layer and flattened in the spatial dimension, obtaining the
   458	input Qd ∈ RHin ×Win ×Cin /t of cross-attention mechanism. Similarly, F s undergoes the aforementioned independent operations
   459	twice to obtain V s ∈ R14×Cin /t and K s ∈ R14×Cin /t , where
   460	Qd , V s and K s represent the query vector, value vector, and
   461	key vector, respectively, and t denotes the dimensional decay
   462	ratio. The cross-attention mechanism exploits the correlation
   463	between sparse and dense features, effectively integrating
   464	the importance representation of different local features for
   465	each pixel region, thereby enhancing the representation of
   466	pedestrian appearance features in the original feature map.
   467	Finally, the channel number of the output feature map of
   468	the Cross-Attention Module is restored to Cin by a 1 × 1
   469	convolutional layer, and the size is resized to Hin × Win × Cin ,
   470	followed by a residual connection with the feature map Fd .
   471	The output feature Fout is then input into the next convolutional
   472	layer of the ResNet network.
   473	C. Skeletal Area Modeling
   474	1) Skeleton-Based Mask Generation: Despite the construction of a pose attention-guided soft fusion mechanism
   475	in the VARA module, which enables adaptive suppression of occlusion information, this alone is insufficient for
   476	occluded pedestrian re-identification. To further enhance feature discriminability, it is necessary to achieve more precise
   477	localization of pedestrian regions.
   478	To address this, most methods [5], [8], [13], [14] use
   479	keypoint heatmaps or semantic parsing results as masks of
   480	the human body parts to localize and extract local features
   481	of pedestrians. Nevertheless, there are certain limitations to
   482	consider. Firstly, there exists a discrepancy between the radiation range of pedestrian keypoint heatmaps and the actual
   483	pedestrian appearance area, thereby leading to incomplete
   484	feature extraction using this approach. For instance, in the
   485	ViTPose model [34], the radiation range of pedestrian keypoint
   486	
   487	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   488	
   489	GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION
   490	
   491	8603
   492	
   493	Fig. 4. Examples of key point heat map for occluded person images.
   494	
   495	Fig. 6. Schematic diagram of Skeletal Area Modeling.
   496	
   497	levels of points A and B are δA and δB , respectively. The weight
   498	mask M of a particular local skeletal region can be represented
   499	as:
   500	Mi j = exp−
   501	
   502	((i−c x )2 +( j−cy )2 )
   503	r2
   504	
   505	I(δA , δB )
   506	
   507	−γ(sin θ)2
   508	
   509	Fig. 5. Schematic diagram of predefined skeleton regions.
   510	
   511	heatmaps fails to cover the entirety of the pedestrian appearance area, losing some extractable appearance information
   512	between adjacent keypoints and introducing some occlusion
   513	interference, as depicted in Fig. 4. Secondly, the accuracy of
   514	the semantic mask, employed in the alternative method, can
   515	be easily compromised by low image resolution. Therefore,
   516	a novel SAM strategy is proposed, which utilizes keypoint
   517	coordinates and confidence to model the characteristics of the
   518	skeletal area between keypoints, generates a soft mask with
   519	pixel confidence scores for skeleton, and leverages it to extract
   520	local features of the skeleton.
   521	Firstly, the skeletal regions of the pedestrian need to be
   522	predefined. Based on the establishment of human keypoints in
   523	the CrowdPose dataset [35], the following pedestrian skeletal
   524	regions are defined: upper arm, lower arm, upper leg, lower
   525	leg, torso, and head. Considering that the torso occupies a
   526	larger proportion of the body and there is a risk of it being
   527	occluded, four sub-regions are further delineated based on the
   528	retained torso: by selecting the midpoints of the four edges
   529	that make up the torso and connecting them, the torso is
   530	divided into four sub-regions, namely upper-left torso, upperright torso, lower-left torso, and lower-right torso. In this way,
   531	14 skeletal regions can be obtained, as shown in Fig. 5. Each
   532	sub-region of the torso is composed of the midpoint of its
   533	upper and lower edges and the midpoints of its left and right
   534	edges, and the confidence score of the midpoint is calculated
   535	by linear interpolation, while the other skeletal regions are
   536	composed of the corresponding two keypoints.
   537	Secondly, each predefined skeletal region is independently
   538	modeled to obtain weight masks specific to each region. As
   539	shown in Fig. 6, the key points at both ends of a certain
   540	skeletal region are defined as points A and B, point C(i, j)
   541	is an arbitrary position point in the image plane space, point
   542	O(c x , cy ) is the midpoint between A and B, and the confidence
   543	
   544	r = α ∗ exp
   545	lAO
   546	(
   547	(δA + δB )/2, (δA + δB )/2 < T
   548	I(δA , δB ) =
   549	(δA + δB )/4, (δA + δB )/2 ≥ q T
   550	
   551	(3)
   552	(4)
   553	(5)
   554	
   555	where θ represents the angle between line AB and line CO,
   556	and lAO is the length of line AO. r indicates the radiation
   557	scale. α and γ refers to the expansion coefficient of overall
   558	radiation scale and the transverse (perpendicular to the direction of line segment AB) compression coefficient of radiation
   559	scale, respectively. The confidence threshold denoted as T in
   560	this context is established at 0.7. A higher value of the α
   561	corresponds to a larger expansion scale for the mask, while
   562	a larger γ value leads to increased compression perpendicular
   563	to the direction of the line segment AB. Consequently, the
   564	skeletal mask tends to exhibit a narrow and elongated region.
   565	I(·, ·) indicates the confidence level of the current local skeletal
   566	area. Following the above approach, modeling is conducted
   567	on 14 predefined local skeletal regions. The resulting mask
   568	image is expressed as M ∈ R4H×4W×14 . It is important to
   569	acknowledge that the fine details within the masked region can
   570	only be accurately represented at higher resolutions. Therefore,
   571	the spatial dimensions of the mask are set to 4H ×4W to align
   572	with the spatial layer dimensions of the Conv.B1 layer’s output
   573	feature map.
   574	This adaptive method of generating skeletal masks can
   575	effectively suppress responses from obstructed or background
   576	areas, and achieve a smooth transition between responses from
   577	human and non-human regions, which in turn leads to more
   578	complete semantic extraction of local features.
   579	2) Local Feature Extraction Based on Multi-Layer Fusion:
   580	Once the local feature mask is obtained, a direct approach to
   581	extracting the corresponding local features for each skeletal
   582	region would involve multiplying the mask with the feature
   583	map output from the backbone network [16], [36], [37].
   584	However, this method poses certain inherent risks. This is
   585	primarily due to the fact that the fine details within the
   586	masked region can only be accurately represented at higher
   587	resolutions, resulting in a larger mask size compared to the
   588	output size of the backbone network. Upsampling the feature
   589	
   590	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   591	
   592	8604
   593	
   594	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025
   595	
   596	map to match the mask’s size leads to reduced similarity
   597	between pixel features corresponding to the local mask region
   598	and a diminished amount of detailed information. Conversely,
   599	downsampling the mask image to align with the output size
   600	of the backbone network results in a coarse representation
   601	of the local masked region due to size compression. Consequently, the effectiveness of posture cues experiences a
   602	significant decline. To address the challenges of balancing
   603	local feature detail expression and mask feature map size,
   604	this paper leverages a multi-layer fusion approach, inspired
   605	by reference [38]. Specifically, the output feature maps of
   606	Conv.B1, VARA1, and Conv.B4 layers are fused to obtain
   607	the feature map F 0 ∈ R4H×4W×C . The fused feature map
   608	integrates high-level semantic information and shallow-level
   609	detail information, making up for the deficiency caused by
   610	directly upsampling the output feature map of the backbone
   611	network.
   612	Subsequently, local feature extraction is performed. As
   613	described in this section, the predefined skeletal regions are
   614	modeled to extract the corresponding mask feature map M ∈
   615	R4H×4W×14 . Each layer’s mask Mi ∈ R4H×4W×1 is multiplied
   616	with the corresponding pixel of feature map F 0 , resulting in
   617	skeletal region local feature map Fi∗ ∈ R4H×4W×C . The output
   618	is then subjected to global average pooling to obtain a local
   619	feature vector fi∗ ∈ R1×1×C , where C represents the number
   620	of channels. This process can be described as follows:
   621	fi∗ = pool(F 0 Mi ), (i = 1, . . . , 14)
   622	
   623	(6)
   624	
   625	where pool(·) denotes the global average pooling operation.
   626	In the mask modeling process, the coordinates and confidence
   627	of the key points are considered. Therefore, when a skeletal
   628	region is occluded, the corresponding mask pixels of occluded
   629	parts have lower responses, effectively suppressing the representation of occlusion information in the local feature.
   630	D. Skeleton Modulated Cross-Attention Fusion
   631	Although the generation of skeletal local features considers
   632	the confidence of human key points and their spatial radiation
   633	characteristics, the skeletal modeling process lacks trainable
   634	parameters and relies solely on the spatial location information
   635	of key points provided by the human pose estimation model.
   636	As a result, it suffers from limited flexibility. Additionally, the
   637	skeletal mask solely provides visibility information, offering
   638	little insight into the strength and discriminative qualities of
   639	internal pixel features within the defined range. Consequently,
   640	the discriminability of the extracted skeletal local features
   641	remains unexplored.
   642	To address these challenges, this section proposes the
   643	SMCF. This module incorporates the decoder part of the
   644	Transformer and employs cross-attention mechanisms to
   645	dynamically capture the semantic correlation between the
   646	output features of the backbone network and the skeletal local
   647	features. Since a skeletal local feature corresponds to an area
   648	containing multiple spatial pixels, the alignment and fusion
   649	process can be simplified as a “one-skeleton-to-many-pixels”
   650	problem. This module offers a novel solution to enhance the
   651	flexibility and discriminability of skeletal local features.
   652	
   653	Fig. 7. Architecture of the SMCF module.
   654	
   655	Specifically, as illustrated in Fig. 7, the SMCF module
   656	comprises a multi-head self-attention and a Skeleton Modulated Cross-Attention (SMC). The module takes as input the
   657	flattened feature map G ∈ RHW×C from the backbone network,
   658	the skeletal local feature f ∗ ∈ R14×C , and the attention weight
   659	map Fw∗ ∈ R14×HW , generated by the skeletal mask. As the
   660	dimensions of the input features G and f ∗ are large, they are
   661	first passed through 1 × 1 convolutional layers to reduce their
   662	0
   663	0
   664	dimensions, resulting in Gin ∈ RHW×C and fin∗ ∈ R14×C , where
   665	0
   666	C signifies the number of filters in the 1 × 1 convolutional
   667	layer.
   668	The feature map Gin is initially passed through a multi-head
   669	self-attention module, where the self-attention mechanism
   670	conducts a global analysis on the feature map, exploring
   671	the interrelationships between features (such as the symmetric structural relationships in pedestrians or the contrasting
   672	relationships between upper-body and lower-body features).
   673	0
   674	Subsequently, the skeletal local feature fin∗ ∈ R14×C serves
   675	as the query matrix and is input into the SMC sub-module.
   676	Additionally, the key matrix and value matrix are provided by
   677	Gm . The computation process of the SMC sub-module can be
   678	represented as follows:
   679	fm∗ = CrossAtt( fin∗ , Gm , Fw )
   680	 ∗ ∗
   681	
   682	( fin W1 )(Gm W2∗ )
   683	∗
   684	= so f tmax
   685	F
   686	√
   687	w (G m W3 )
   688	dk
   689	0
   690	
   691	0
   692	
   693	0
   694	
   695	0
   696	
   697	(7)
   698	(8)
   699	0
   700	
   701	0
   702	
   703	where W1∗ ∈ R(C ×C ) ,W2∗ ∈ R(C ×C ) and W3∗ ∈ R(C ×C )
   704	respectively represent the weights of the three fully connected layers. dw denotes the scaling factor whose purpose
   705	is to avoid the problem of gradient disappearance arising
   706	from the softmax when the attention map elements are too
   707	large. The scale factor, denoted as dw , is introduced to prevent
   708	the gradient vanishing problem caused by excessively large
   709	attention weights during Softmax operation. Besides, in the
   710	fusion process, it is crucial to avoid the focus of a specific
   711	patch in a query on visually irrelevant regions. Inspired by
   712	Gao et al. [39], the attention weight map Fw is element-wise
   713	multiplied with the cross-attention map generated in SMC,
   714	imposing constraints on the fusion process at the local spatial
   715	level.
   716	During the alignment fusion of each skeletal feature, only
   717	the relevant regions corresponding to the mask are attended
   718	
   719	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   720	

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Privacy preserving person re-identification via anonymizing diffusion model.pdf' - | nl -ba | sed -n '170,470p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   170	models is that it performs diﬀusion in the latent space of images, achieving higher computational eﬃciency. In addition, Stable Diﬀusion introduces conditional control. It uses the text encoder of CLIP to generate
   171	prompts’ latent embeddings as conditions to guide image generation.
   172	
   173	2.2. Privacy-preserving methods
   174	Traditional privacy-preserving methods focus on blurring, pixelating, or adding noise to images, which may aﬀect the ReID performance
   175	to some extent. Dietlmeier et al.[5] show that blurring faces has little impact on the performance of ReID systems, and propose an anonymous ReID dataset DAA [5] with blurred faces. However, the face only
   176	occupies a small part of the pedestrian image, and privacy attackers
   177	can still retrieve the target person through other major features such
   178	as clothing, posture, background, etc. Zhang et al.[22] propose a reversible anonymous framework based on joint learning. They adapt desensitized images generated by conventional methods as the initial supervision to generate anonymous images for privacy protection, and the
   179	original images can be recovered for ReID research. PIS [23] uses images from other identities in the dataset to weakly encrypt the original
   180	image, generating anonymized images with relatively consistent identities. Diﬀerential privacy [24,25] and secure computation [26,27] have
   181	also been used for privacy protection, yet they introduce relatively high
   182	computational overhead [28]. Some synthetic virtual datasets [29–31]
   183	have better privacy protection eﬀects while the large domain gap results in poor generalizability. Recently, event cameras have also been
   184	used for privacy-preserving in ReID [32,33], but their application is not
   185	widespread.
   186	
   187	3. Methodology
   188	The goal of ADM is to generate new IDs that are anonymized and
   189	ReID usable. In the following, we describe the main components of our
   190	method: Image-text Pair generation (3.1), Anonymizing Diﬀusion Model
   191	(3.2), and ID separation degree (3.3).
   192	3
   193	
   194	Pattern Recognition 174 (2026) 112939
   195	
   196	R. Wang et al.
   197	
   198	to obtain the caption for 𝑥𝑖 :
   199	
   200	Algorithm 1 Prompt griddle.
   201	Iter(𝑃𝑎𝑙𝑙 ∶ 𝑙𝑖𝑠𝑡, 𝑃𝑜𝑟𝑔𝑛 ∶ 𝑙𝑖𝑠𝑡, 𝑛 ∶ 𝑖𝑛𝑡)
   202	𝑐ℎ𝑒𝑐𝑘𝑢 , 𝑐ℎ𝑒𝑐𝑘𝑙 , 𝑃𝑠𝑒𝑙𝑒𝑐𝑡 , 𝑏𝑎𝑐𝑘𝑢𝑝 ← empty list
   203	for 𝑖 = 0 to 𝑙𝑒𝑛(𝑃𝑎𝑙𝑙 )
   204	if 𝑝𝑖 ∉ 𝑃𝑜𝑟𝑔𝑛
   205	if 𝑝𝑖𝑢 ∉ 𝑐ℎ𝑒𝑐𝑘𝑢 ς 𝑝𝑖𝑙 ∉ 𝑐ℎ𝑒𝑐𝑘𝑙
   206	Add 𝑝𝑖 to 𝑃𝑠𝑒𝑙𝑒𝑐𝑡
   207	Add 𝑝𝑖𝑢 to 𝑐ℎ𝑒𝑐𝑘𝑢
   208	Add 𝑝𝑖𝑙 to 𝑐ℎ𝑒𝑐𝑘𝑙
   209	if 𝑙𝑒𝑛(𝑃𝑠𝑒𝑙𝑒𝑐𝑡 ) >= 𝑛
   210	return 𝑃𝑠𝑒𝑙𝑒𝑐𝑡 , 𝑏𝑎𝑐𝑘𝑢𝑝
   211	else Add 𝑝𝑖 to 𝑏𝑎𝑐𝑘𝑢𝑝
   212	return 𝑃𝑠𝑒𝑙𝑒𝑐𝑡 , 𝑏𝑎𝑐𝑘𝑢𝑝
   213	
   214	𝐶𝑎𝑝𝑡𝑖𝑜𝑛 =A < 𝑔𝑒𝑛𝑑𝑒𝑟 > 𝑤𝑖𝑡ℎ < ℎ𝑎𝑖𝑟 > is wearing
   215	
   216	Griddle(𝑃𝑎𝑙𝑙 ∶ 𝑙𝑖𝑠𝑡, 𝑃𝑜𝑟𝑔𝑛 ∶ 𝑙𝑖𝑠𝑡, 𝑛 ∶ 𝑖𝑛𝑡)
   217	𝑃 ← 𝑃𝑎𝑙𝑙 , 𝑃𝑟𝑒𝑚𝑎𝑖𝑛 ← empty list, 𝑛𝑢𝑚 ← 𝑛
   218	while 𝑙𝑒𝑛(𝑃𝑟𝑒𝑚𝑎𝑖𝑛 ) < 𝑛
   219	𝑃𝑠𝑒𝑙𝑒𝑐𝑡 , 𝑏𝑎𝑐𝑘𝑢𝑝 = 𝐼𝑡𝑒𝑟(𝑃 , 𝑃𝑜𝑟𝑔𝑛 , 𝑛𝑢𝑚)
   220	Add 𝑃𝑠𝑒𝑙𝑒𝑐𝑡 to 𝑃𝑟𝑒𝑚𝑎𝑖𝑛
   221	𝑃 = 𝑏𝑎𝑐𝑘𝑢𝑝
   222	𝑛𝑢𝑚 = 𝑛𝑢𝑚 − 𝑙𝑒𝑛(𝑃𝑠𝑒𝑙𝑒𝑐𝑡 )
   223	return 𝑃𝑟𝑒𝑚𝑎𝑖𝑛
   224	
   225	3.2. Anonymizing diﬀusion model
   226	
   227	< 𝑢𝑝𝑝𝑒𝑟_𝑤𝑒𝑎𝑟_𝑐𝑜𝑙𝑜𝑟 > < 𝑢𝑝𝑝𝑒𝑟_𝑤𝑒𝑎𝑟 > and
   228	
   229	The above img2text model can generate captions suitable for ReID. The
   230	format of generated caption is consistent. Diﬀerent components of the
   231	caption accurately describe the appearance of each part of the pedestrian, facilitating the construction of new captions through mixing and
   232	matching, which will guide the model to generate anonymous images.
   233	Note that diﬀerent from T2I-ReID, we only use a ﬁxed Template 2
   234	for new prompt generation, and do not select optional attributes like
   235	< 𝑏𝑎𝑔 >, < ℎ𝑎𝑡 >, < 𝑣𝑒ℎ𝑖𝑐𝑙𝑒 > to focus on identities.
   236	
   237	As shown in Fig. 3, we ﬁne-tune SD in 2 stages. We utilize the entire
   238	image-text pairs in Stage 1 and a small subset in Stage 2 to ﬁne-tune
   239	the U-Net, with other parts frozen. To improve the intra-identity consistency, we add ID (class) Embedding to the time step embedding. The
   240	particular method is as follows.
   241	3.2.1. 2-stage ﬁne-tuning
   242	Stage 1. ReID model is sensitive to domain shift and intra-identity consistency. To ensure robust performance, it is crucial to bridge the gap
   243	between synthetic training data and real-world test data. The objective of stage 1 is to align the style of generated images with the original datasets and to cultivate the initial capability to produce realistic,
   244	discriminative pedestrian images based on textual prompts. We accomplish this by ﬁne-tuning the pre-trained Stable-Diﬀusion-v-1.5 model on
   245	a comprehensive set of real image-text pairs from Section 3.1. Unlike
   246	typical synthetic data generation, Stable-Diﬀusion-v-1.5 is adept at creating high-resolution images but lacks ReID-speciﬁc traits such as low
   247	resolution and varied lighting and backgrounds. Fine-tuning on authentic images enables the model to assimilate these distinctive ReID styles.
   248	Moreover, training on image-text pairs allows the model to associate
   249	textual descriptions with the visual attributes and features of pedestrians.
   250	Through the ﬁrst stage of ﬁne-tuning, the SD model assimilates the
   251	style and captures basic text-conditioning abilities without overﬁtting
   252	to speciﬁc identities, laying the foundation for subsequent controlled
   253	generation tailored to the ReID task.
   254	
   255	3.1. Image-text pair generation
   256	Previous work solely relying on noise to generate new images cannot
   257	ensure consistency between identities, which is crucial for ReID training. Therefore, we use text to control identity consistency. We adopt
   258	an automatic img2text generation approach since manually annotating
   259	image-text pairs is time and labor-costly. However, directly using CLIP
   260	to generate captions for ReID images has limitations as it struggles to
   261	produce ﬁne-grained, consistent attribute descriptions necessary to effectively control the image generation process for ReID tasks. Inspired by
   262	T2I-ReID [41], we implemented img2text generation using the DivideConquer-Combine approach.
   263	Divide. Based on the experience from T2I-ReID, we initialize a set of
   264	attribute description terms:
   265	𝐴𝑡𝑡 ∈ { < 𝑔𝑒𝑛𝑑𝑒𝑟 >, < ℎ𝑎𝑖𝑟 >,
   266	< 𝑢𝑝𝑝𝑒𝑟_𝑤𝑒𝑎𝑟_𝑐𝑜𝑙𝑜𝑟 >, < 𝑢𝑝𝑝𝑒𝑟_𝑤𝑒𝑎𝑟 >,
   267	
   268	(1)
   269	
   270	< 𝑙𝑜𝑤𝑒𝑟_𝑤𝑒𝑎𝑟_𝑐𝑜𝑙𝑜𝑟 >, < 𝑙𝑜𝑤𝑒𝑟_𝑤𝑒𝑎𝑟 >}
   271	
   272	Stage 2. After Stage 1, SD can not accurately interpret prompt terms
   273	due to overlapping attributes between identities in the full dataset. This
   274	might harm the consistency of one generated ID since the appearance of
   275	certain part will shift due to vague instruction. To tackle this problem,
   276	attributes in the prompt should be bound to certain visual patterns in
   277	the images.
   278	Stage 2 addresses this by ﬁne-tuning SD on small selected subsets.
   279	Speciﬁcally, we divide all image-text pairs into many small portions
   280	while ensuring the proportions of a certain attribute in each portion
   281	do not exceed a threshold 𝐴𝑡𝑡. 𝐴𝑡𝑡 is selected by experiments to achieve
   282	a maximum trade-oﬀ between intra-ID consistency and inter-ID variety
   283	of anonymized dataset. Thus overlapped attributes are limited so that
   284	each prompt can accurately points to certain appearance. SD is ﬁnetuned on each subset and generates anonymized images according to
   285	Section 3.2.2 separately. Various anonymized images can be obtained
   286	from diﬀerent subsets.
   287	
   288	Each attribute 𝐴𝑡𝑡 is a set of description terms corresponding to it, e.g. <
   289	𝑔𝑒𝑛𝑑𝑒𝑟 > contains the terms 𝑚𝑎𝑛 and 𝑤𝑜𝑚𝑎𝑛. These attributes cover the
   290	basic appearance of pedestrians and can provide suﬃcient information
   291	to train a ReID model.
   292	Conquer. Denote the set of samples in the original dataset as 𝑋 =
   293	{ 𝑖
   294	}
   295	𝑥 |𝑖 ∈ (0, 𝑛) . We populate template 𝑇 𝑒𝑚𝑝𝑙𝑎𝑡𝑒1 with the attributes mentioned earlier to obtain the following set of sentences:
   296	𝑇𝐴𝑡𝑡 = {𝑡𝑗𝐴𝑡𝑡 |𝑡𝑗𝐴𝑡𝑡 = “A photo of a phrase person”,
   297	where phrase ∈ 𝐴𝑡𝑡}
   298	
   299	(4)
   300	
   301	< 𝑙𝑜𝑤𝑒𝑟_𝑤𝑒𝑎𝑟_𝑐𝑜𝑙𝑜𝑟 > < 𝑙𝑜𝑤𝑒𝑟_𝑤𝑒𝑎𝑟 > .
   302	
   303	(2)
   304	
   305	𝑗
   306	Encoding 𝑥𝑖 and 𝑡𝑗𝐴𝑡𝑡 through CLIP gives 𝑓𝑥𝑖 and 𝑓𝐴𝑡𝑡
   307	. We calculate the
   308	𝑗
   309	cosine distance between 𝑓𝑥𝑖 and each 𝑓𝐴𝑡𝑡 , and select the attribute term
   310	with the maximum value, which can be denoted as:
   311	(
   312	)
   313	𝑗
   314	𝑝𝑖𝐴𝑡𝑡 = arg max cos 𝑓𝑥𝑖 , 𝑓𝐴𝑡𝑡
   315	(3)
   316	
   317	Class Embedding. When training SD models, we initialize a unique learnable class embedding 𝑒𝑚𝑏𝑖𝑐𝑙𝑠 for ID 𝑖𝑑 𝑖 in the training set. We replace the
   318	original time step embedding 𝑒𝑚𝑏𝑡𝑖𝑚𝑒 of SD with 𝑒𝑚𝑏 = 𝑒𝑚𝑏𝑖𝑐𝑙𝑠 + 𝑒𝑚𝑏𝑡𝑖𝑚𝑒 .
   319	The vector 𝑒𝑚𝑏 is expanded to match the spatial dimensions of the feature map in U-Net through replication, then the broadcasted embedding
   320	
   321	𝑗
   322	
   323	Combine. Each sample 𝑥𝑖 after the Conquer phase will obtain a set of description terms: 𝑝𝑖𝐴𝑡𝑡 . We populate these description terms into 𝑇 𝑒𝑚𝑝𝑙𝑎𝑡𝑒2
   324	4
   325	
   326	Pattern Recognition 174 (2026) 112939
   327	
   328	R. Wang et al.
   329	
   330	Fig. 3. An overview of anonymizing diﬀusion model. Given a real public dataset, ADM ﬁrst trains a basic Stable Diﬀusion model on the entire dataset in Stage 1. In
   331	Stage 2, the real dataset is partitioned into multiple subsets, with each subset containing 10 IDs. The base model is ﬁne-tuned separately on each subset (producing
   332	ADM_1, ADM_2, …, ADM_n as diﬀerent ﬁne-tuned instances). Within each subset, new prompts are constructed according to Algorithm 1, and anonymized images
   333	are generated using the corresponding ﬁne-tuned model instance and new prompts.
   334	
   335	Suppose the subset selected in Section 3.2.1 Stage 2 contains 𝑚 diﬀerent IDs 𝑥1 , 𝑥2 , … , 𝑥𝑚 , where the attribute group of each 𝑥𝑖 (𝑖 = 1, 2, … , 𝑚)
   336	is 𝑝𝑖 = (𝑝𝑖ℎ , 𝑝𝑖𝑢 , 𝑝𝑖𝑙 ). Theoretically, we can combine these 𝑚 attribute groups
   337	into 𝑚3 prompts. Let the set of all 𝑚3 prompts be 𝑃𝑎𝑙𝑙 and 𝑚 prompts of
   338	selected IDs be 𝑃𝑜𝑟𝑔𝑛 . We use Algorithm 1 to select prompts 𝑃𝑟𝑒𝑚𝑎𝑖𝑛 from
   339	𝑃𝑎𝑙𝑙 . The algorithm ensures all prompts in 𝑃𝑟𝑒𝑚𝑎𝑖𝑛 have almost diﬀerent attributes, preventing diﬀerent generated IDs from sharing similar
   340	prompts. Thus the diversity of the generated dataset is further ensured.
   341	We ﬁll the attribute groups generated by the above algorithm into
   342	the template used in Section 3.1 as prompts for the ﬁne-tuned SD model.
   343	This allows the model to generate a novel dataset with new IDs. By
   344	feeding these prompts into the ﬁne-tuned SD model, we can gain additional synthetic pedestrian images never seen during training. The new
   345	dataset thus achieves the goals of reconstructing identity features for
   346	ReID while avoiding memorizing sensitive private training data details,
   347	fulﬁlling both data utility and privacy preservation.
   348	
   349	is added channel-wise to intermediate feature maps in U-Net’s residual
   350	blocks. 𝑒𝑚𝑏𝑖𝑐𝑙𝑠 will learn to capture unique feature of a certain pedestrian
   351	𝑖𝑑𝑖 during training. Compared to using 𝑒𝑚𝑏𝑡𝑖𝑚𝑒 alone, our method can
   352	provide more continuously consistent contextual information, thereby
   353	mitigating consistency issues caused by random noise. This additional
   354	conditioning helps produce even more consistent results.
   355	3.2.2. Image generation
   356	In this section, we will generate prompts for new IDs. It is intuitive
   357	to use existing prompts, but this may cause a lack of diversity and privacy protection ability. This is because directly using the original text
   358	prompts may easily associate the generated images to existing IDs since
   359	we ﬁne-tuned SD using image-text pairs. Therefore, we need to use text
   360	prompts not appearing in the subset we selected in Section 3.2.1 Stage
   361	2 to generate new IDs. We propose a method to automatically construct
   362	richer and more discriminative prompts. We separate the attributes in
   363	Section 3.1 into three groups:
   364	⎧
   365	⎪
   366	⎨
   367	⎪
   368	⎩
   369	
   370	3.2.3. ADM griddle
   371	Through the former process, we have obtained a novel dataset with
   372	reasonable intra-identity consistency by controlling generation with text
   373	prompts. However, inconsistency can still occur due to noise introduced
   374	
   375	𝑝ℎ =< 𝑔𝑒𝑛𝑑𝑒𝑟 > + < ℎ𝑎𝑖𝑟 >
   376	𝑝𝑢 =< 𝑢𝑝𝑝𝑒𝑟_𝑤𝑒𝑎𝑟_𝑐𝑜𝑙𝑜𝑟 > + < 𝑢𝑝𝑝𝑒𝑟_𝑤𝑒𝑎𝑟 >
   377	
   378	(5)
   379	
   380	𝑝𝑙 =< 𝑙𝑜𝑤𝑒𝑟_𝑤𝑒𝑎𝑟_𝑐𝑜𝑙𝑜𝑟 > + < 𝑙𝑜𝑤𝑒𝑟_𝑤𝑒𝑎𝑟 >
   381	5
   382	
   383	Pattern Recognition 174 (2026) 112939
   384	
   385	R. Wang et al.
   386	
   387	Fig. 4. A demo of generated images on Market-1501, CUHK-SYSU and MSMT17. In each row, the 4 images on the right are generated from the 3 images on the left.
   388	
   389	from the original dataset and ﬁne-tuning process. To ensure the potential
   390	ReID performance of the generated dataset is not adversely aﬀected,
   391	we need further screening to guarantee consistency within each ID. We
   392	propose a new method called ADM Griddle to ﬁlter inconsistent samples
   393	{
   394	}
   395	{
   396	}
   397	for a given ID. Let 𝑥1 , 𝑥2 , … , 𝑥𝑛 and 𝐹 1 , 𝐹 2 , … , 𝐹 𝑛 denote samples
   398	and their features of a speciﬁc ID. We specify a hyperparameter 𝜖, and
   399	the set of samples within a distance of 𝜖 from 𝑥𝑖 is 𝑆(𝑥𝑖 ). We choose the
   400	ID with the largest set,
   401	𝑘 = arg max |𝑆(𝑥𝑖 )|.
   402	𝑖∈(0,𝑛)
   403	
   404	so that the total amount of ﬁltered images meets the experimental requirements.
   405	3.3. ID separation degree (ISD)
   406	Generated identities should be mutually independent from real ones.
   407	Thus, we can measure their similarity by the distance between feature
   408	representations of generated and real identity images. Anonymized features should be as far as possible from real identity features in the feature
   409	space to present distinct visual characteristics and achieve anonymization. To characterize the visual diﬀerence between samples, we propose
   410	a new metric called ID separation degree (ISD). Given a real dataset
   411	with 𝑛 identities denoted as 𝑖𝑑 1 , 𝑖𝑑 2 , … , 𝑖𝑑 𝑛 , and the number of samples
   412	
   413	(6)
   414	
   415	ADM Griddle removes outlier samples that do not closely match the
   416	densest cluster of samples for each identity, thereby enhancing the intraidentity consistency of the ﬁltered dataset. 𝜖 can be carefully selected
   417	6
   418	
   419	Pattern Recognition 174 (2026) 112939
   420	
   421	R. Wang et al.
   422	
   423	Fig. 5. Visualization of retrieval results on the generated dataset attacked by real CUHK-SYSU IDs.
   424	
   425	Fig. 6. Cosine distance distribution. Orgn refers to the distance between training images and the feature centroid of their belonging IDs. ADM refers to the distance
   426	between generated images and the nearest real ID feature centroids.
   427	
   428	belonging to each 𝑖𝑑 𝑖 is 𝑚𝑖 . The 𝑘-th sample of the 𝑖-th identity in the
   429	feature space is 𝑓𝑘𝑖 . The centroid of the 𝑖-th identity is:
   430	1 ∑ 𝑖
   431	𝑐𝑖 = 𝑖
   432	𝑓 .
   433	(7)
   434	𝑚 𝑘 𝑘
   435	
   436	Table 1
   437	Compare with other methods on Market-1501. The R1, R-5, R-10 indicats the metric Rank-1, Rank-5, Rank10 respectively. Here, the ReID models are trained on
   438	synthesized or mixed datasets. 1x/2x data denote that
   439	the amount of generated IDs and images are the same
   440	as/twice that of the real training set.
   441	
   442	The generated data contains 𝑚 samples with features 𝑓𝑗 , where 𝑗 ∈
   443	[1, 𝑚]. ISD is deﬁned as:
   444	[
   445	]
   446	‖𝑐 𝑖 × 𝑓𝑗 ‖
   447	1 ∑
   448	ISD =
   449	min 1 −
   450	.
   451	(8)
   452	𝑚 𝑗 𝑖
   453	‖𝑐 𝑖 ‖ ⋅ ‖𝑓𝑗 ‖
   454	ISD uses the cosine distances between generated samples and the centroid of real identities. A higher ISD indicates lower similarity between
   455	real and generated data in the feature space on average, hence stronger
   456	privacy protection by preventing identity association. ISD eﬀectively
   457	evaluates the degree of anonymity in privacy-preserving ReID methods.
   458	
   459	Methods
   460	
   461	R-1
   462	
   463	R-5
   464	
   465	R-10
   466	
   467	mAP
   468	
   469	PersonX Sub1 [30]
   470	PersonX Sub4 [30]

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '340,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   340	T HE N OTATIONS D EFINED IN T HIS PAPER
   341	
   342	learning. In reality, however, the informative content contained
   343	in each local feature varies.
   344	To capture more comprehensive local feature clues, we
   345	propose a novel approach for combining features, as shown
   346	in Figure 1. Using a ResNet-50-based backbone encoder, we
   347	first extract the global feature Fg , then divide it horizontally
   348	into two parts to obtain the local features Ftop and Fbot .
   349	These local features are then fused using adaptive weights,
   350	leading to the generation of combined features, as expressed in
   351	Eq. (1).
   352	Fcombined = w1 ∗ Ftop + w2 ∗ Fbot
   353	(1)
   354	where w1 and w2 are trainable weight parameters that are
   355	updated during each step of back-propagation, with the constraint that w1 + w2 = 1. Subsequently, the obtained combined
   356	features will be employed in the subsequent module.
   357	A. GMM Features Denoising
   358	
   359	To reduce the pseudo-label noise under the fully unsupervised setting, we design two modules, GFD and MKD.
   360	The GFD module utilizes the Gaussian Mixture Model and
   361	Gaussian assumption distribution to denoise low-confidence
   362	features within the same pseudo-class, thus reducing the
   363	impact of label noise in the same pseudo-class and improving
   364	the reliability of pseudo-labels. The MKD module employs a
   365	refined teacher model to guide the student model, effectively
   366	reducing pseudo-label noise and enhancing semantic consistency. Extensive experiments demonstrate that the proposed
   367	modules are highly effective for USL ReID.
   368	III. M ETHOD
   369	This section presents the comprehensive framework of partbased feature complementary denoising, illustrated in Figure 2.
   370	To ensure that readers fully comprehend the framework we
   371	propose, we first introduce the mathematical notations (as
   372	shown in Table I). We then proceed to describe the relevant
   373	modules and their corresponding loss functions.
   374	Despite significant progress in USL ReID methods [26],
   375	[27] that focus on local features, these methods typically
   376	assume that all local features contribute equally to feature
   377	
   378	In existing USL ReID methods [20], [29], [41], [50], the
   379	use of pseudo-labels has become a mainstream approach.
   380	However, the presence of noise in pseudo-labels is inevitable,
   381	posing challenges for the model in capturing accurate feature
   382	representations and ultimately affecting the entire learning
   383	process. Most studies attempt to suppress pseudo-label noise
   384	by designing complex loss functions or feature purification
   385	techniques, such as LP [42] and PPLR [26]. However, they
   386	overlook the label noise within each pseudo-label class. This
   387	noise can cause samples from different true categories to
   388	be grouped under the same pseudo-label, leading to a shift
   389	in the class center away from the correct feature mean and
   390	even violating the assumption of intra-class compactness. As
   391	a result, these methods miss a critical opportunity to mitigate
   392	pseudo-label noise during the learning process.
   393	To tackle this problem, we propose the GFD module, as
   394	shown in Figure 2, which first uses a Gaussian Mixture
   395	Model [51] to distinguish between high-confidence and lowconfidence features, and then denoises the low-confidence
   396	features by sampling from the Gaussian assumption distribution using the high-confidence features. Specifically, in the
   397	GFD module, the first step is to calculate the recognition loss
   398	LCE of the sample xi .
   399	i
   400	)yi
   401	LCE {xi , yi } = −logσ(Fcombined
   402	
   403	(2)
   404	
   405	where xi indicates the i-th sample, yi represents the pseudolabel of the sample xi , σ denotes the softmax function, and
   406	Fcombined is the combined feature learned from x. We then calculate the confidence of each sample using a Gaussian Mixture
   407	Model. First, we need to know the probability of observing xi
   408	under the Gaussian distribution of yi : N (LCE {xi , yi } | µyi , Σyi ).
   409	The specific formula is as follows:
   410	N (LCE | µyi , Σyi )
   411	
   412	
   413	1
   414	1
   415	> −1
   416	exp − (LCE − µyi ) Σk (LCE − µyi )
   417	(3)
   418	=
   419	(2π)d/2 |Σyi |1/2
   420	2
   421	where d signifies the dimensionality of the distribution associated with yi , µyi and Σyi represent the mean vector and
   422	the covariance matrix, respectively. Subsequently, we need to
   423	
   424	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   425	
   426	TIAN et al.: PART-BASED FEATURE COMPLEMENTARY DENOISING FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   427	
   428	2141
   429	
   430	Fig. 2. The overview of our proposed PFCD framework, which comprises three components: (a) GMM Features Denoising: Denoising low-confidence
   431	samples within the same pseudo-class through a Gaussian Mixture Model (GMM) to improve overall accuracy, both globally and locally. At the global level,
   432	low-confidence features within the same pseudo-class are denoised, while at the local level, low-confidence part-based features undergo the same denoising
   433	process. It is essential to highlight that this module is not executed in the first epoch. (b) Partial-features Fusion and Contrastive Scheme: The fusion of local
   434	and global features for clustering, along with the introduction of PFCL loss. In addition to global features, yet the local features are retained as a separate
   435	clustering memory bank to mitigate the impact of inherent biases of global features on clustering results. Except for the first epoch, the module uses the global
   436	and local features that have been denoised by the GFD module. (c) Modular Knowledge Distillation: We leverage the well-trained teacher model to mentor
   437	student model, updating it by combining the Knowledge Distillation Loss (KDL) with the PFCL Loss. This approach helps to alleviate the pseudo-label noise
   438	for the student model.
   439	
   440	comprehend the joint probability of xi under the Gaussian
   441	distribution of yi , as shown in Eq. (4), along with the total
   442	probability pall (xi ). The total probability of xi is the sum of
   443	the joint probabilities across all K Gaussian distributions, as
   444	shown in Eq. (5):
   445	p(xi ) = πyi N (LCE | µyi , Σyi )
   446	pall (xi ) =
   447	
   448	K
   449	X
   450	
   451	π j N (LCE | µ j , Σ j )
   452	
   453	(4)
   454	(5)
   455	
   456	j=1
   457	
   458	where π j denotes the mixing coefficientPof the Gaussian distriK
   459	bution, which satisfies the conditions: k=1
   460	πk = 1andπk ≥ 0,
   461	and j represents j-th Gaussian distribution. Finally, we calculate the confidence of xi using Bayes’ theorem [52]:
   462	p(xi )
   463	γ xi ,yi =
   464	pall (xi )
   465	πy N (LCE | µyi , Σyi )
   466	= PK i
   467	= αi
   468	(6)
   469	j=1 π j N (LCE | µ j , Σ j )
   470	where αi is the confidence value of the sample xi , LCE is the
   471	recognition loss of xi .
   472	After obtaining the confidence values α, we utilize the
   473	confidence threshold δ to categorize features with confidence
   474	values above the threshold into a high-confidence feature set
   475	AH , while those below the threshold are classified into a lowconfidence feature set AL .
   476	
   477	To better address the impact of noise on pseudo-labels
   478	[24], we introduce a Gaussian distribution assumption to
   479	obtain more robust features. We follow the hypothesis that
   480	the features of samples with the same pseudo-label follow the
   481	Gaussian distribution. By using samples that are above the
   482	confidence threshold δ under the same pseudo-label, we calculate their Gaussian prior distribution N (µβj , Σβj ). The formula
   483	in detail is as follows:
   484	PN j i
   485	i=1 F β
   486	j
   487	µβ =
   488	(7)
   489	Nj
   490	Nj
   491	
   492	Σβj =
   493	
   494	1 X i
   495	(Fβ − µβj )(Fβi − µβj )T
   496	Nj
   497	
   498	(8)
   499	
   500	i=1
   501	
   502	where N j denotes the number of samples in the j-th pseudolabel class that are above the confidence threshold δ, Fβi
   503	represents the feature of the i-th sample in the current part
   504	β, and β ∈ {g, t, b} represents the global, upper half part, and
   505	the lower half part.
   506	In the end, we randomly sample from the Gaussian prior
   507	distribution to generate auxiliary denoised features fβj , which
   508	are then used to denoise the low-confidence samples Fβi . The
   509	overall formula is as follows:
   510	(
   511	Fβi
   512	i f xi ∈ {AH }
   513	i
   514	F̃β =
   515	(9)
   516	(1 − αi ) ∗ Fβi + αi ∗ fβj i f xi ∈ {AL }
   517	
   518	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   519	
   520	2142
   521	
   522	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   523	
   524	where Fβi and F̃βi indicate the original features and the features
   525	after denoising through the GFD module, respectively. In this
   526	method, a Gaussian Mixture Model is first utilized to calculate
   527	the confidence values αi . Subsequently, the threshold δ is
   528	employed to flexibly control the scope of feature denoising,
   529	aiming to preserve as much useful information from the
   530	original features as possible throughout the denoising process.
   531	It should be noted that this module will not be active during
   532	the first epoch.
   533	
   534	Algorithm 1 PFCD Framework Training for Teacher Model
   535	
   536	B. Partial-Features Fusion and Contrastive Scheme
   537	In recent years, USL ReID research has increasingly focused
   538	on the use of local features. In contrast, global features
   539	often only capture the most salient feature clues in appearance, potentially overlooking some equally important but
   540	less conspicuous local features, as demonstrated by methods
   541	such as SPCL [9] and CCL [20]. However, current existing
   542	methods generally assume that the importance of each local
   543	feature block is the same. Instead, we believe that each local
   544	feature block contains distinct feature clues. Therefore,
   545	the feature extraction method is improved by dynamically
   546	adjusting weights to fuse different local feature blocks.
   547	Through the combination of CF and GFD, the denoising
   548	of global and local features is collaboratively achieved. The
   549	denoised features are formed as F̃β .
   550	F̃βi = {F̃β1 , F̃β2 , F̃β3 , · · · , F̃βN }
   551	
   552	(10)
   553	
   554	where β ∈ {g, t, b} represents the global, upper half part, and
   555	the lower half part.
   556	In USL ReID methods, the DBSCAN [40] clustering algorithm has more advantages compared to K-means [39]. It does
   557	not necessitate the pre-specification of the number of clusters,
   558	making it more suitable for this task. In recent years, with the
   559	introduction of SPCL and CCL, more and more USL ReID
   560	tasks have begun to adopt the DBSCAN algorithm. Therefore,
   561	as demonstrated in Algorithms 1 and 2, our method also
   562	uses the same clustering algorithm. Through the DBSCAN
   563	clustering algorithm, the pseudo-labels are generated.
   564	Given the propensity of global feature-based pseudo-labels
   565	to erroneously group visually similar yet distinct images into
   566	the same cluster, we diverge from traditional global feature
   567	clustering. Instead, we employ a weighted fusion of global
   568	and local features for clustering, as depicted in Figure 2.
   569	This involves utilizing the denoised feature set F̃βi to compute the Jaccard distance matrix dβ , where β ∈ {g, t, b},
   570	and subsequently, the fused Jaccard distance matrix d̂, as
   571	detailed in Eq. (11). This method ensures a more nuanced
   572	and accurate clustering that accounts for both global and local
   573	characteristics of the images.
   574	d̂ = (1 − λ1 )dg + λ1 (dt + db )
   575	
   576	(11)
   577	
   578	where λ1 is the balancing factor. Perform clustering on the
   579	merged Jaccard distance matrix d̂ to generate pseudo-labels,
   580	thereby reducing the bias brought about by relying solely on
   581	global features. It should be noted that in the first epoch, the
   582	Jaccard distance matrix is calculated using the feature set Fβi
   583	that has not been denoised.
   584	
   585	Additionally, inspired by the previous work [20], we adopt
   586	cluster-level contrastive learning to avoid inconsistency issues
   587	during the memory update process. Specifically, as shown in
   588	Algorithm 1, we first construct three memory banks, enabling
   589	the model to capture more nuanced information during the
   590	learning process. After obtaining the pseudo-label, we initialize the cluster centers in memory banks with the corresponding
   591	cluster centroids. The specific formula is as follows:
   592	1 X β
   593	φβk =
   594	F̃i
   595	(12)
   596	|Ck |
   597	i∈Ck
   598	
   599	where |Ck | denotes the total number of instances in cluster
   600	k, β ∈ {g, t, b} represents the global, upper half part, and the
   601	lower half part, and F̃i indicates the i-th feature vector in Ck .
   602	The three memory banks independently calculate the cluster
   603	centroids. Based on these centroids, we calculate the clusterlevel clustering loss for the β-th part as follows:
   604	Lβclu = − log
   605	
   606	exp(F̃qβ ∗ φβ+ /τ)
   607	ΣCk=0 exp(F̃qβ ∗ φβk /τ)
   608	
   609	(13)
   610	
   611	where F̃qβ indicates the feature vector of the query instance q
   612	in its respective part β, φβ+ denotes the cluster centroid feature
   613	vector of the cluster to which the query instance q belongs,
   614	and φβk represents the cluster centroid feature vector of the k-th
   615	cluster stored in β memory bank.
   616	To enhance the ability of our method to capture more local
   617	information, we design the PFCL (Partial-features Fusion and
   618	
   619	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   620	
   621	TIAN et al.: PART-BASED FEATURE COMPLEMENTARY DENOISING FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   622	
   623	Contrastive Learning) loss. Firstly, we need to compute the
   624	overall cluster-level clustering loss, which is given by the
   625	following formula:
   626	Lclu = (1 − λ2 )Lgclu + λ2 (Ltclu + Lbclu )
   627	
   628	2143
   629	
   630	Algorithm 2 PFCD Framework Training for Student Model
   631	
   632	(14)
   633	
   634	where λ2 is the balancing factor. Subsequently, we update the
   635	cluster memory bank iteratively using momentum, with the
   636	query features extracted from the current mini-batch, as shown
   637	in Eq. (15):
   638	φβk = mφβk + (1 − m)F̃qβ
   639	(15)
   640	where m is the momentum updating parameter, k denotes the
   641	cluster where the feature vector of the current query resides.
   642	Next, to further improve the robustness of the model, we
   643	introduce a soft triplet loss. For each person, there will be
   644	many positive and negative samples. We introduce cosine
   645	similarity to identify the nearest neighbors as positive samples
   646	S p , and the farthest neighbors as negative samples S n . The
   647	detailed formula is presented as follows:
   648	β
   649	N
   650	exp S ip
   651	1 X
   652	β
   653	(16)
   654	log
   655	Ltri = −
   656	β
   657	β
   658	N
   659	exp S i + exp S i
   660	i=1
   661	
   662	p
   663	
   664	n
   665	
   666	where β ∈ {g, t, b} represents the global, upper half part,
   667	and lower half part, and N refers to the number of samples
   668	in the current mini-batch. By separately calculating the soft
   669	triplet loss for both global and local features and combining
   670	these losses as shown in Eq. (17), the model is able to learn
   671	more comprehensive feature representations. Meanwhile, it
   672	can capture discriminative feature information from multiple
   673	levels, thereby enhancing the overall robustness of the model.
   674	Ltri = (1 − λ3 )Lgtri + λ3 (Lttri + Lbtri )
   675	
   676	(17)
   677	
   678	where λ3 is the balancing factor. Finally, our PFCL loss is
   679	expressed as:
   680	LPFCL = (1 − η1 )Lclu + η1 Ltri
   681	
   682	(18)
   683	
   684	where η1 represents the hyper-parameter. With the PFCL loss,
   685	we effectively combine local features with global features.
   686	However, in the early stages of training, due to significant
   687	label noise in the pseudo-labels, it is challenging to eliminate
   688	noise using only the GFD module. Therefore, we introduce
   689	the MKD (Modular Knowledge Distillation) module to further
   690	enhance noise suppression.
   691	C. Modular Knowledge Distillation
   692	Traditional teacher models [8], [27] are typically initialized
   693	with parameters pre-trained on ImageNet, potentially leading
   694	to significant label noise during the initial training phase. In
   695	addition, teacher networks that are updated online have certain constraints regarding feature representation. To overcome
   696	these issues, we employ the GFD and PFCS modules that
   697	we previously designed to train the teacher model, as shown
   698	in Algorithm 1, resulting in a fully trained off-line teacher
   699	model. This model is subsequently utilized to retrain the
   700	student model, enabling the student model to rapidly assimilate
   701	knowledge from the teacher model at the onset of training. As
   702	demonstrated in Algorithm 2, this strategy facilitates the early
   703	
   704	generation of precise pseudo-labels. Thus, we construct the
   705	Knowledge Distillation Loss (KDL) as detailed below:
   706	LβKDL =
   707	
   708	F̃qβ
   709	F̃qβ
   710	
   711	−
   712	
   713	bqβ
   714	F
   715	bqβ
   716	F
   717	
   718	2
   719	
   720	(19)
   721	2
   722	
   723	LKDL = LgKDL + LtKDL + LbKDL
   724	
   725	(20)
   726	
   727	bq is the feature vector of the query instance q in
   728	where F
   729	teacher model, F̃ denotes the feature after denoising by the
   730	GFD module, β ∈ {g, t, b} represents the global, upper half
   731	part, and lower half part. Ultimately, the final Knowledge
   732	Distillation Loss (KDL) is obtained by summing the losses
   733	calculated from three distinct parts. The overall loss function
   734	for the student model is as follows:
   735	Lall = LPFCL + (1 − η1 )(η2 LKDL )
   736	
   737	(21)
   738	
   739	where η2 denotes the hyper-parameter. By combining the use
   740	of the above modules, our model maximizes its denoising
   741	performance. This not only effectively leverages local finegrained clues, but also generates more accurate pseudo-labels
   742	through the denoising modules.
   743	
   744	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   745	
   746	2144
   747	
   748	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   749	
   750	TABLE II
   751	T HE DATASETS U SED FOR T RAINING AND T ESTING
   752	
   753	IV. E XPERIMENT
   754	A. Datasets and Evaluation Protocol
   755	Datasets: We use four benchmark datasets, including
   756	Market-1501 [53], DukeMTMC-reID [54], PersonX [55], and
   757	MSMT17 [56], to measure the effectiveness of our method.
   758	These datasets are widely used in existing works, and detailed
   759	statistics are summarized in Table II.
   760	Evaluation Metrics: Following existing USL ReID research,

codex
后三篇方法主体已经拿到大半。PSCR 的方法还差交叉注意力融合后的输出和优化，ADM 的方法区里已经直接给了隐私证据指标和可视化线索。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '716,900p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   716	During the alignment fusion of each skeletal feature, only
   717	the relevant regions corresponding to the mask are attended
   718	
   719	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   720	
   721	GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION
   722	
   723	to, enabling the network to learn accurate alignment of highlevel semantic features and enhancing the discriminability of
   724	the original skeletal local features. Finally, after undergoing residual connections, a normalization layer, and a fully
   725	connected layer, fm∗ gives rise to the fused skeletal local
   726	features fout ∗.
   727	∗
   728	During matching stage, the local features fout
   729	and the
   730	global feature fg obtained from the backbone are concatenated
   731	together for occluded re-id.
   732	
   733	8605
   734	
   735	TABLE I
   736	C OMPARISON W ITH THE S TATE - OF - THE -A RT M ETHODS ON O CCLUDED D UKE MTMC (%)
   737	
   738	IV. E XPERIMENTS
   739	In this section, for the occlusion problem, we demonstrate
   740	the effectiveness of the PSCR method by comparing SOTA
   741	methods and conducting extensive ablation experiments.
   742	A. Implementation Details
   743	The proposed method is implemented on the PyTorch
   744	framework, utilizing a single A100 (40G) GPU for both training and testing. We employ a pre-trained ResNet50 network as
   745	the backbone network, trained on the ImageNet dataset, with
   746	its fully connected layer removed. Pose information, including 14 keypoints’ location coordinates, confidence scores,
   747	and keypoint heatmaps, is generated using the advanced
   748	pedestrian pose estimation model ViTPose [34]. To expedite
   749	re-identification model training, the pose information is generated offline.
   750	During the training phase, the input images are uniformly
   751	resized to a resolution of 384 × 128. Triplet loss and ID loss
   752	are employed as training objectives [4]. For the triplet loss,
   753	we select a hard positive and a hard negative sample for each
   754	image within the mini-batch, obtained through PK sampling,
   755	in order to form triplets. The margin for the triplet loss is set
   756	to 0.3. For the ID loss, it is employed to calculate the loss
   757	specifically for the local branch, while both the ID loss and
   758	triplet loss are combined to calculate the loss for the global
   759	branch. The entire training process follows an end-to-end
   760	approach, with the learning rate initialized to 3.5 × 10−4 and
   761	subsequently decayed at the 40th, 70th, and 100th iterations to
   762	values of 3.5 × 10−5 , 3.5 × 10−6 , and 3.5 × 10−7 , respectively.
   763	Model convergence is achieved after 150 iterations. In the
   764	inference phase of the model, all the local features and global
   765	features output by the network are concatenated along the
   766	channel direction to obtain a one-dimensional feature vector,
   767	which represents the pedestrian’s re-identification feature.
   768	
   769	TABLE II
   770	C OMPARISON W ITH S TATE - OF - THE -A RTS ON THE M ARKET-1501 AND
   771	D UKE MTMC DATASETS
   772	
   773	B. Comparison With the State-of-the-Art Methods
   774	In order to conduct an objective evaluation of the proposed
   775	method, a comparative analysis is performed with stateof-the-art occluded pedestrian re-identification techniques
   776	using the extensive Occluded-DukeMTMC dataset, OccludedReID dataset, as well as the well-established Market-1501
   777	and DukeMTMC-ReID pedestrian re-identification datasets.
   778	The comparative evaluation is presented in Table I and
   779	Table II. The selected comparison methods are representative approaches in the field of pedestrian re-identification,
   780	encompassing prominent techniques from the past three years.
   781	The experimental results presented in this study are directly
   782	
   783	sourced from the original published research papers in reputable journals or conferences.
   784	1) Result on the Occluded Pedestrian Re-Identification
   785	Datasets: In Table I, the first and second groups illustrate
   786	
   787	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   788	
   789	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Privacy preserving person re-identification via anonymizing diffusion model.pdf' - | nl -ba | sed -n '470,700p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   470	PersonX Sub4 [30]
   471	CamStyle [43]
   472	TRS [44]
   473	K. Kansal et al.[45]
   474	ADM(1 x data)
   475	ADM(2 x data)
   476	
   477	34.8
   478	48.6
   479	58.8
   480	64.6
   481	77.6
   482	67.4
   483	72.1
   484	
   485	46.1
   486	61.4
   487	78.2
   488	74.9
   489	–
   490	83.8
   491	87.2
   492	
   493	51.6
   494	67.2
   495	84.3
   496	79.5
   497	–
   498	88.4
   499	91.4
   500	
   501	17.2
   502	29.9
   503	27.4
   504	44.7
   505	57.7
   506	45.3
   507	50.6
   508	
   509	4. Experiment
   510	4.1. Implementation details
   511	To analyze the eﬀectiveness of our proposed method, we conduct
   512	thorough experiments on commonly used ReID datasets. The remainder
   513	of this section presents the implementation details of our framework
   514	(4.1), comparison with sotas (4.2), results on benchmark datasets (4.3),
   515	ablation studies (4.4), and discussion (4.5).
   516	
   517	SD Fine-tuning We use Stable-Diﬀusion-v1-5 for ﬁne-tuning. The
   518	batch size is set to 48. To optimize the model, we use Adam optimizer
   519	with a learning rate of 10−4 and weight decay of 10−2 . The training state
   520	takes 100 epochs in Stage 1, and 50 epochs in Stage 2. In Stage 2, within
   521	7
   522	
   523	Pattern Recognition 174 (2026) 112939
   524	
   525	R. Wang et al.
   526	
   527	Table 2
   528	Compare with other methods on Market-1501 and MSMT17. Here, the
   529	ReID models are trained exclusively on the real dataset.
   530	Methods
   531	
   532	SecureReID [46]
   533	PixelFade [47]
   534	ADM
   535	
   536	Market-1501
   537	
   538	intra-identity consistency. To thoroughly evaluate the quality of generated samples, FID [48] scores are also provided. The higher 𝑇𝑔 , 𝑇𝑢 , and
   539	lower FID indicate the stronger data utility.
   540	To evaluate the privacy protection level of the generated dataset,
   541	we use the metric 𝑇𝑝 and ISD. From the perspective of a privacy attack, the attacker determines whether a certain person exists in the generated dataset by using the query image to retrieve the correct target
   542	images from the generated ReID dataset. Privacy leakage occurs when
   543	the related images are successfully retrieved. To measure the privacypreserving capability during an attack through neural networks., we use
   544	the metric 𝑇𝑝 corresponding to the accuracy and mAP of the retrieval
   545	results. Speciﬁcally, the label of a certain ID in the generated gallery
   546	is speciﬁed as its corresponding IDs’ labels in original real datasets.
   547	This means that the accuracy and mAP rise when any image of the corresponding IDs is in the call-back list. Furthermore, as mentioned in
   548	Section 3.3, we use ISD to measure the visual discrimination between
   549	generated and original data. The original (Orgn) of ISD denotes the average cosine distance between features of each original training sample
   550	and its ID’s centroid. The ADM of ISD denotes the average cosine distance between the features of each generated training sample and the
   551	centroid of its closest ID in the real training set. The improvement (Impv)
   552	of ISD is calculated as Impv = (ADM − Orgn)∕Orgn, indicating the relative enhancement of privacy by ADM. The lower 𝑇𝑝 and higher Impv of
   553	ISD indicate the stronger privacy protection capability.
   554	The measurement settings of 𝑇𝑔 , 𝑇𝑢 and 𝑇𝑝 are shown in Table 4.
   555	Note that for a fair comparison, the scale of the generated training set is
   556	consistent with the training set in the real dataset. The results are shown
   557	in Table 3. ADM performs well in both 𝑇𝑔 and 𝑇𝑢 , aligned with the result in Section 4.2.2. The result of 𝑇𝑝 is signiﬁcantly lower than normal
   558	levels, which means that the ReID model fails to retrieve relevant IDs
   559	from the gallery protected by ADM. The high Impv of ISD demonstrates
   560	that there is distinct visual discrimination between generated data and
   561	original ones, which is aligned with the result of 𝑇𝑝 . The result of FID
   562	is slightly higher than the mainstream generative model. However, considering the goal of ADM is to anonymize the original IDs, the result is
   563	reasonable.
   564	
   565	MSMT17
   566	
   567	R-1
   568	
   569	R-5
   570	
   571	R-10
   572	
   573	mAP
   574	
   575	R-1
   576	
   577	R-5
   578	
   579	R-10
   580	
   581	mAP
   582	
   583	91.7
   584	94.2
   585	57.3
   586	
   587	96.8
   588	–
   589	81.9
   590	
   591	–
   592	–
   593	87.9
   594	
   595	78.2
   596	85.2
   597	29.4
   598	
   599	48.1
   600	62.7
   601	44.8
   602	
   603	62.7
   604	–
   605	70.0
   606	
   607	–
   608	–
   609	77.5
   610	
   611	27.6
   612	43.1
   613	21.3
   614	
   615	each subset, there are 10 IDs, 25 images for each ID, and we set the
   616	portion of overlapped keywords 𝐴𝑡𝑡 to 0.4.
   617	ReID Training We use Resnet50 as the backbone. The batch size
   618	is set to 64. To optimize the model, we use Adam optimizer with base
   619	learning rate 3.5 × 10−4 and weight decay of 5 × 10−4 . The total training
   620	state takes 120 epochs.
   621	4.2. Comparison with state-of-the-arts
   622	4.2.1. Experimental settings
   623	We compare our proposed ADM with other privacy-protected methods in two settings. Under the ﬁrst setting, the Reid models are trained
   624	on the generated (or mixed) datasets, which the most works follow. Under the second setting, the Reid models are trained on the real datasets.
   625	The ﬁrst setting is more in line with the paradigm we proposed, in
   626	which the real datasets are protected from achievement of the attackers.
   627	In this scenario, each comparative method is trained and tested on the
   628	Market-1501 dataset following the experimental settings speciﬁed in its
   629	original paper. Speciﬁcally, for PersonX, both the query and gallery sets
   630	consist of generated images, while for other methods, both the query
   631	and gallery sets are composed of real images.
   632	The second setting is typically adopted by encryption-based methods. We compare our method with two such approaches on both Market1501 and MSMT17 datasets, following their original experimental protocols. Speciﬁcally, in this scenario, both query and gallery sets consist
   633	of decrypted/generated images for these two encryption-based methods
   634	as well as our ADM during testing.
   635	
   636	4.4. Ablation studies
   637	
   638	4.2.2. Experimental results
   639	The results of the ﬁrst and second setting are reported respectively
   640	in Tables 1 and 2. As shown in Table 1, ADM far exceeds PersonX Sub,
   641	CamStyle and almost outperforms TRS. However, K. Kansal et al gains
   642	better accuracy and mAP than ADM. It is worth noticing that ADM can
   643	generate more IDs and images than real training sets. Doubling the training data results in a 6.9 % improvement on Rank-1 and an 11.6 % improvement on mAP. By providing much more training data for the ReID
   644	model than normal datasets, ADM can further improve its ReID performance. As shown in Table 2, ADM achieves notably lower performance
   645	compared to the other two methods on both Market-1501 and MSMT17.
   646	This is because SecureReID and PixelFade are encryption-based methods where the test images are decrypted and restored to their original
   647	form, retaining substantial pedestrian features from the original images.
   648	While this undoubtedly contributes to their superior ReID performance,
   649	it overlooks privacy risks, especially in scenarios where attackers gain
   650	access to decryption keys.
   651	
   652	4.4.1. Comparison with base model
   653	To provide an intuitive comparison between ADM and the baseline
   654	ReID model in terms of ReID performance, we conduct an experiment
   655	on Market-1501 under Setting 1 described in Section 4.2.1, comparing
   656	ResNet50Mid with ADM. As shown in Table 5, relative to ResNet50Mid,
   657	ADM yields an 8.2 % drop in Rank-1 accuracy and a 44.6 % drop in mAP,
   658	representing the cost paid to achieve privacy protection.
   659	4.4.2. Eﬀectiveness of diﬀerent components
   660	We compare four experiments: (a) Diﬀerent ﬁne-tuning strategies (b)
   661	With/Without class embedding (c) Using diﬀerent data ﬁltering methods (d) Complete ADM:2-Stage ﬁne-tuning + class embedding + ADM
   662	griddle. The results are shown in Table 6.
   663	(a) vs (d): Single-stage ﬁne-tuning achieves great privacy protection
   664	but poor utility. The ﬁrst stage overﬁts duplicated attributes without
   665	precise semantics, which means prompts can not control the image generation process steadily and the appearance of generated images may
   666	be biased from prompts, leading to poor intra-identity consistency. The
   667	second stage lacks style from real data and has a nonnegligible domain
   668	gap with real data, causing a lack of reality.
   669	(b) vs (d): Adaptation of class embedding signiﬁcantly improves utility with minor privacy loss. Class embedding can provide better control
   670	of generated images with the same ID, thus enhancing intra-identity
   671	consistency, which is fundamental in real-world applications.
   672	(c) vs (d): Random griddle leads to notably lower utility than centroid/ADM griddle, which performs similarly. The result indicates that
   673	a carefully designed griddle can ﬁlter out outlier samples of each ID and
   674	
   675	4.3. Results on benchmark datasets
   676	To further quantitatively evaluate ADM, we conducted experiments
   677	on three benchmark datasets and evaluated utility and privacy capabilities using more metrics. To evaluate the utility of the generated dataset,
   678	inspired by [23], we use two metrics, 𝑇𝑔 and 𝑇𝑢 . 𝑇𝑔 reﬂects the utility
   679	of generated data on training usable ReID models targeting real scenarios, which is the same metric in Table 1. 𝑇𝑢 measures discriminability within the generated data, considering inter-identity variability and
   680	8
   681	
   682	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '920,1100p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   920	corresponds to thermal infrared imagery that typically exhibits
   921	smoother and less textured responses. To maintain a consistent
   922	dataset construction strategy across benchmarks, we adopt a
   923	unified occlusion synthesis framework while enabling slight
   924	adjustments in transformation strength to accommodate the
   925	different characteristics of the two infrared modalities.
   926	For near-infrared or thermal infrared modalities, as illustrated in Fig. 6, the visible occlusion ovis
   927	γ,l is first converted to
   928	a grayscale representation by G(·) and then processed by an
   929	intensity normalization operator N (·) to reduce illumination
   930	
   931	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
   932	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   933	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   934	
   935	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   936	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
   937	
   938	8
   939	
   940	V. EXPERIMENTS
   941	X avis
   942	
   943	Outdoor
   944	
   945	X vis
   946	Outdoor
   947	
   948	ovis,l
   949	
   950	Bottomaligned
   951	
   952	m ,l
   953	
   954	Grayscale
   955	
   956	Intensity
   957	normalization
   958	
   959	Adaptive
   960	selection
   961	
   962	Thermal
   963	infrared
   964	
   965	Contrast
   966	compression
   967	
   968	Low-frequency
   969	filtering
   970	
   971	X anir
   972	
   973	Similar process
   974	
   975	X nir
   976	
   977	otir,l
   978	
   979	on,irl
   980	Near
   981	infrared
   982	
   983	Fig. 6: Illustration of occluded image generation process for
   984	Occluded VI-ReID datasets.
   985	
   986	VIS
   987	Modality
   988	
   989	NIR
   990	Modality
   991	
   992	A. Datasets
   993	1) Holistic Datasets: SYSU-MM01 [29] is a large-scale
   994	VI-ReID dataset, containing 30,071 visible and 15,792 nearinfrared images of 491 identities. Images are captured by six
   995	cameras, including four visible-light and two near-infrared
   996	modalities, under both indoor and outdoor environments. The
   997	dataset follows a standard training/test split, where 395 identities are used for training and 96 for evaluation. RegDB [30]
   998	consists of 8,240 paired visible and infrared images of 412
   999	identities. Each identity has one visible and one infrared image
  1000	per viewpoint. The dataset supports two evaluation modes:
  1001	visible-to-infrared (V - I) and infrared-to-visible (I - V), with
  1002	206 identities for training and the remaining 206 for testing.
  1003	2) Occluded Datasets: Occluded-SYSU-MM01 and
  1004	Occluded-RegDB are extended versions of the above
  1005	datasets, constructed following the occlusion protocol
  1006	introduced in Section IV. Both datasets preserve the structure,
  1007	identity labels, and modality settings of the original datasets,
  1008	which are used to evaluate the robustness of our model under
  1009	structured occlusions. These semantically rich occlusions
  1010	significantly increase the difficulty of the matching task and
  1011	more realistically reflect occluded pedestrians in surveillance
  1012	environments.
  1013	B. Evaluation Metrics and Implementation Details
  1014	
  1015	Fig. 7: Examples of the occlusions from visible (VIS) and
  1016	near-infrared (NIR) modalities.
  1017	
  1018	bias. A contrast compression transformation C(·) is subsequently applied to regularize the dynamic range, followed by
  1019	a low-frequency filtering operation L(·) to suppress excessive
  1020	texture details and produce smoother structural responses.
  1021	Considering the different imaging characteristics of nearinfrared and thermal sensors, relatively mild compression and
  1022	filtering are adopted for near-infrared imagery to preserve
  1023	reflective structural cues, whereas stronger texture suppression
  1024	is applied for thermal infrared imagery to produce more
  1025	homogeneous responses. The above transformation can be
  1026	summarized as
  1027	
  1028	om = Lm Cm (N (G(ovis ))) ,
  1029	
  1030	(22)
  1031	
  1032	where m ∈ {nir, tir} denotes the infrared modality. This
  1033	process generates infrared-compatible occlusions that match
  1034	the appearance statistics of each modality while preserving
  1035	the original occlusion geometry.
  1036	All occluded images are generated offline using a fixed
  1037	protocol [6], [24], [33]: half of all training images are occluded, all query images are occluded, and all gallery images
  1038	remain unoccluded. This configuration preserves the original
  1039	dataset structure, identity labels, and evaluation partitioning.
  1040	The resulting Occluded-SYSU-MM01 and Occluded-RegDB
  1041	datasets provide a robust and semantically consistent benchmark for evaluating VI-ReID under partial occlusion.
  1042	
  1043	The performance of the Occluded VI-ReID task is assessed
  1044	using the Cumulative Matching Characteristics (CMC) curve
  1045	and mean Average Precision (mAP). The CMC curve (Rankk) measures top-k accuracy, while mAP calculates average
  1046	precision across recall rates. All experiments are conducted
  1047	on an NVIDIA 4090 GPU using PyTorch. Input images are
  1048	resized to 288 × 144 pixels, with random cropping, horizontal
  1049	flipping, and channel augmentation for data enhancement. Network optimization is performed using the stochastic gradient
  1050	descent (SGD) optimizer.
  1051	C. Ablation Study
  1052	1) Effect of Each Component: To verify the effectiveness
  1053	and complementarity of each module, we conduct ablation experiments on Occluded-SYSU-MM01 and Occluded-RegDB
  1054	datasets, where “B” denotes the baseline with ResNet50, “G”
  1055	refers to the Graph-guided Feature Completion Transformer
  1056	(GFCT), “S” represents the Semantic-aware Region Completion Module (SRCM), and “M” denotes the Multi-distribution
  1057	Consistency Alignment (MCA) loss.
  1058	In Table I, we evaluate all module combinations to analyze
  1059	their independent and joint effects. Starting from the baseline
  1060	(Index 1), introducing G (Index 2) improves the Rank-1
  1061	accuracy in the all-search mode from 58.32% to 60.47%
  1062	(+2.15%), demonstrating the effectiveness of structure-aware
  1063	token completion. Applying S alone (Index 3) also increases
  1064	Rank-1 to 60.14% (+1.82%), while introducing M alone (Index
  1065	4) achieves 59.44% (+1.12%), indicating that each module
  1066	independently contributes to performance improvement. When
  1067	combining modules, performance further improves. The G+S
  1068	
  1069	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
  1070	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
  1071	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
  1072	
  1073	
codex
OCCNet 的第 8 页已经进入消融，说明作者用模块组合消融支撑“补全路径是互补的”。照这个标准，我会把其他几篇的关键实验段也补看一下，不只停在框架描述。
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
/bin/zsh -lc "pdftotext -f 9 -l 12 'Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Optimal Illumination Distance Metrics for Person Re-Identification
     2	
     3	26:9
     4	
     5	a series of distance measurement matrices 𝑑, 𝑑 1 ...𝑑𝑖 ...𝑑𝐿 , under pre-defined lighting conditions. The
     6	feature extractor EH is trained using the cross-entropy loss function on a portion of the original
     7	train dataset. In contrast, EL is trained on the training dataset with gamma correction applied.
     8	4.5
     9	
    10	Optimal Illumination Distance Replacement
    11	
    12	Concatenate the distance matrices formed by normal lighting 𝑑 and the distance matrices formed
    13	under pre-defined lighting conditions 𝑑 1, 𝑑 2 ...𝑑𝐿 on the channels to create the matrix 𝑇 . Then, use the
    14	minimum operation on the channels to select the optimal distance under the best lighting condition
    15	for different image pairs as the final distance in the image pairs, forming the optimal distance
    16	matrix 𝑇 ∗ . This is used for final sorting and retrieval. This process can be formally expressed as
    17	𝑇 = [𝑑 ⊕ 𝑑 1 ... ⊕ 𝑑𝐿 ]
    18	
    19	(8)
    20	
    21	𝑇 ∗ = 𝑚𝑖𝑛(𝑇 ),
    22	
    23	(9)
    24	
    25	where 𝑇 ⊂ 𝑅𝑚×𝑛×𝐿 represents the matrix obtained by concatenating all the distance matrices on
    26	the channels, ⊕ represents the concatenation operation, min represents the minimum operation,
    27	𝑇 ∗ ⊂ 𝑅𝑚×𝑛 represents the optimal distance matrix used for final sorting and retrieval.
    28	5
    29	5.1
    30	
    31	Experiment
    32	Experimental Settings
    33	
    34	Dataset. Following the settings in reference [29], we conducted experiments on the synthetic
    35	sophisticated illumination datasets Market-1501-IA and Duke-IA.
    36	(1) Market-1501-IA: This dataset was synthesized based on the Market-1501 [30] dataset, where
    37	the training and query sets of Market-1501-IA are as same as training and query sets of Market-1501,
    38	respectively. For the gallery part of Market-1501-IA, each image from the Market-1501 gallery,
    39	which was randomly selected from 13 lighting scales (0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2.1, 2.5,
    40	2.9, 3.3), and a random gamma adjustment was applied to each channel of the original image to
    41	generate images suitable for different lighting conditions. In total, Market-1501-IA contains 751
    42	identities with 12,936 training images, 750 different identities with 3,368 query images, and 19,732
    43	gallery images.
    44	(2) Duke-IA: This dataset was also synthesized using the same method on the DukeMTMC-ReID
    45	dataset [33], resulting in a dataset containing 702 identities with 16,522 training images, 702 different
    46	identities with 2,228 query images, and 17,661 gallery images.
    47	Evaluation Metrics. The retrieval library contains multiple images of the same pedestrian, so
    48	in order to effectively compare the performance of the proposed method, we use the standard
    49	cumulative matching characteristics and average precision as evaluation metrics.
    50	Baseline Methods. Our proposed method is compared with a number of advanced methods,
    51	including (1) traditional Re-ID methods: IDE*, PCB, BOT, CBN [36], and AGW, (2) methods that can
    52	be transformed to the same lighting conditions using GAN: Camstyle.DG-Net [32], IS-GAN [6],
    53	baseline+IER methods.
    54	Implementation Details. All of our experiments were conducted on Tesla p40 GPU in Pytorch.
    55	We employed the generator and discriminator from CycleGAN for GAN, where the generator
    56	has nine layers of blocks. During training, Adam optimizer was used with a batch size of 32. The
    57	learning rate for the GAN generator and discriminator was set to 2e-4, while the learning rate for
    58	the association network was set to 2e-3. A total of 60 epochs were trained, and the learning rate was
    59	updated with a decay rate of 1e-3 every 20 epochs. For GAN training, we used gamma correction to
    60	produce images that adapt to different lighting conditions. The input was the light-adapted image,
    61	and the label was the normal image. All images were scaled to 256 × 128 × 3, and random cropping
    62	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
    63	
    64	26:10
    65	
    66	C. Wang et al.
    67	Table 1. Overall Performance Comparison on Market-IA and Duke-IA
    68	Methods
    69	
    70	IDE [18]
    71	PCB [17]
    72	Camstyle [34]
    73	BOT(baseline) [14]
    74	BOT(all trick) [14]
    75	DG-Net [32]
    76	CBN [36]
    77	IS-GAN [26]
    78	AGW [25]
    79	IDE+IER
    80	IE+IER+MFOR
    81	BOT+GeM
    82	BOT+GeM+IER
    83	BOT+GeM+IER+MTOR
    84	BOT+GeM+GAN
    85	OIDM(BOT+GeM)
    86	OIDM(AGW)
    87	
    88	mAP
    89	27.1
    90	33.1
    91	37.8
    92	29.9
    93	43.7
    94	49.0
    95	51.8
    96	45.9
    97	51.5
    98	58.5
    99	63.4
   100	33.9
   101	67.4
   102	72.1
   103	75.0
   104	76.2
   105	76.9
   106	
   107	Market-1501-IA
   108	Rank-1 Rank-5 Rank-10
   109	69.7
   110	84.9
   111	89.3
   112	79.1
   113	90.0
   114	92.4
   115	76.0
   116	89.4
   117	92.4
   118	70.8
   119	86.8
   120	90.7
   121	86.1
   122	94.5
   123	96.4
   124	83.0
   125	95.1
   126	96.8
   127	84.4
   128	94.1
   129	96.2
   130	87.4
   131	95.7
   132	96.5
   133	90.0
   134	95.9
   135	97.4
   136	82.7
   137	93.1
   138	95.7
   139	85.2
   140	94.4
   141	96.4
   142	74.2
   143	88.5
   144	92.2
   145	85.4
   146	94.7
   147	96.9
   148	89.2
   149	96.1
   150	97.7
   151	90.3
   152	96.3
   153	97.7
   154	90.4
   155	96.5
   156	97.8
   157	90.8
   158	96.7
   159	98.1
   160	
   161	mAP
   162	25.0
   163	36.5
   164	38.8
   165	29.6
   166	45.3
   167	51.8
   168	50.9
   169	48.0
   170	50.4
   171	44.8
   172	50.2
   173	35.4
   174	58.7
   175	63.2
   176	62.9
   177	64.3
   178	64.5
   179	
   180	Duke-IA
   181	Rank-1 Rank-5
   182	63.2
   183	77.5
   184	76.3
   185	87.6
   186	69.7
   187	82.3
   188	69.0
   189	83.4
   190	81.6
   191	91.5
   192	83.5
   193	92.1
   194	78.2
   195	88.4
   196	84.3
   197	92.1
   198	86.0
   199	93.3
   200	68.9
   201	81.4
   202	73.3
   203	85.0
   204	73.0
   205	85.9
   206	77.0
   207	88.9
   208	81.7
   209	90.0
   210	81.4
   211	90.8
   212	81.9
   213	90.3
   214	86.7
   215	93.8
   216	
   217	Rank-10
   218	82.2
   219	90.5
   220	.85.9
   221	87.7
   222	93.8
   223	94.0
   224	91.2
   225	94.4
   226	95.1
   227	85.7
   228	89.0
   229	89.5
   230	92.5
   231	92.8
   232	93.2
   233	93.1
   234	95.7
   235	
   236	The best results are in bold.
   237	
   238	and horizontal flipping were applied to our method. The learning rate was set to 3e-4, and Adam
   239	optimizer was used. We trained the Re-ID model for 60 epochs.
   240	5.2
   241	
   242	Performance Evaluation
   243	
   244	The comparison results are shown in Tables 1 and 2. From the table, we can observe that our
   245	proposed framework UIFE achieved the best experimental performance on the mAP metric on both
   246	datasets over the baselines, indicating that our model is effective in meeting the requirements of
   247	Re-ID tasks by matching as many correct pedestrian images as possible. Nevertheless, our method
   248	did not outperform the AGW algorithm in terms of the Rank-K metric on the Duke-IA dataset.
   249	Moreover, the following observations can be made from the table:
   250	The Effectiveness of Lighting Normalization. By comparing the experimental results of IDE+IER
   251	and IDE on both datasets, we found that converting images taken under different lighting conditions
   252	to the same lighting conditions can effectively improve pedestrian retrieval tasks, proving that
   253	standardizing images to the same lighting conditions has some validity. Additionally, our proposed
   254	UIFE method uses BOT+GeM as the feature extractor, which outperforms the state-of-the-art
   255	BOT+GeM+IER method. This indicates the effectiveness of our proposed framework.
   256	The Effectiveness of Lighting Transformation. By comparing the BOT+GeM+IER and BOT+GeM
   257	+GAN methods on both datasets, we found that the direct use of BOT+GeM+GAN was better,
   258	which demonstrates that directly inputting images taken under different lighting conditions into
   259	the same GAN network can effectively convert these images to normal lighting conditions, avoiding
   260	the selection of incorrect transformation networks due to lighting estimation errors and resulting

exec
/bin/zsh -lc "pdftotext -f 9 -l 12 'Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
     2	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
     3	
     4	9
     5	
     6	TABLE I: Comprehensive ablation study on the OccludedSYSU-MM01 and Occluded-RegDB datasets.
     7	Occluded-SYSU-MM01
     8	Occluded-RegDB
     9	Index B G S M All-search Indoor-search
    10	V-I
    11	I-V
    12	R1 mAP R1 mAP R1 mAP R1 mAP
    13	1
    14	58.32 56.69 61.22 66.29 70.22 49.93 69.73 47.28
    15	2
    16	60.47 59.88 62.34 66.42 72.89 50.64 71.40 48.38
    17	3
    18	60.14 59.43 62.08 66.30 71.58 50.46 70.94 48.19
    19	4
    20	59.44 58.33 61.67 66.31 71.26 50.19 70.35 47.95
    21	5
    22	62.63 60.29 63.47 67.34 73.69 51.15 72.25 50.44
    23	6
    24	61.49 60.15 62.78 67.09 73.15 50.96 71.97 49.87
    25	7
    26	61.35 60.03 62.45 66.98 72.67 50.58 71.64 49.69
    27	8
    28	63.82 62.62 65.57 69.88 75.63 54.76 74.68 52.84
    29	
    30	"%%%
    31	""%%
    32	"%"%
    33	"%%"
    34	"""%
    35	""%"
    36	"%""
    37	""""
    38	
    39	TABLE III: Comparison of occlusion generation strategies.
    40	Property
    41	Occlusion shape
    42	
    43	Random Erasing
    44	Ours
    45	Rectangular Object-aware irregular
    46	Person, Vehicle, Plant,
    47	Occlusion types
    48	1 (uniform mask)
    49	Wall, Bag, Chair, etc.
    50	Semantic consistency
    51	×
    52	✓
    53	Boundary awareness
    54	×
    55	✓
    56	Realistic simulation
    57	Low
    58	High
    59	Cross-modality consistency
    60	×
    61	✓
    62	
    63	TABLE II: Comparison of different consistency alignment
    64	strategies on the two occluded datasets.
    65	Occ-SYSU-MM01
    66	Occ-RegDB
    67	All-search Indoor-search
    68	V-I
    69	I-V
    70	R1 mAP R1 mAP R1 mAP R1 mAP
    71	Without MCA
    72	62.63 60.29 63.47 67.34 73.69 51.15 72.25 50.44
    73	Direct mutual alignment 62.81 60.77 64.12 68.23 73.94 52.76 72.71 51.04
    74	MCA (Ours)
    75	63.82 62.62 65.57 69.88 75.63 54.76 74.68 52.84
    76	Strategy
    77	
    78	configuration (Index 5) boosts Rank-1 to 62.63%, outperforming both G-only and S-only variants by more than 2%,
    79	which confirms their complementary effects. Removing either
    80	G (Index 7, 61.35%) or S (Index 6, 61.49%) leads to noticeable
    81	degradation compared with Index 5, verifying that neither
    82	module is redundant. Finally, integrating all three modules
    83	(Index 8) achieves the best results, reaching 63.82% and
    84	65.57% Rank-1 accuracy in two modes, respectively.
    85	A similar trend is observed on the Occluded-RegDB dataset,
    86	as shown in Table I. Starting from the baseline (Index 1),
    87	introducing G (Index 2) improves the Rank-1 accuracy from
    88	70.22% to 72.89% (+2.67%) in the V - I setting, demonstrating the effectiveness of graph-guided structural completion.
    89	Applying S alone (Index 3) and M alone (Index 4) also
    90	yield consistent gains over the baseline, increasing Rank1 to 71.58% (+1.36%) and 71.26% (+1.04%), respectively.
    91	When combining modules, performance further improves.
    92	The G+S configuration (Index 5) raises Rank-1 to 73.69%,
    93	outperforming both single-module variants, while removing
    94	either G (Index 7, 72.67%) or S (Index 6, 73.15%) results in
    95	inferior performance compared with Index 5, indicating nonredundant contributions. The full model (Index 8) achieves the
    96	best results, reaching 75.63% and 74.68% Rank-1 accuracy in
    97	the V - I and I - V settings, respectively, which corresponds to
    98	overall gains of +5.41% and +4.95% over the baseline. These
    99	consistent improvements across datasets confirm that G, S, and
   100	M collaboratively enhance robustness under occlusion.
   101	These results indicate that SRCM independently contributes
   102	to performance improvement, while the combination of GFCT
   103	consistently provides additional gains. GFCT enhances global
   104	structural token relationships, which in turn strengthens the
   105	effectiveness of region-level semantic refinement in SRCM.
   106	Thus, the two stages are complementary and mutually reinforcing, rather than one dominating the overall improvement.
   107	2) Effect of MCA Alignment Strategy: To further validate
   108	the design of MCA, we compare different consistency align-
   109	
   110	Ours Occlusion Generation
   111	
   112	Random Erasing
   113	
   114	Fig. 8: Visual comparison of our occlusion generation strategy
   115	and standard random erasing.
   116	
   117	ment strategies on the two occluded datasets, as shown in
   118	Table II. Direct mutual alignment improves the performance
   119	over the variant without MCA, confirming that cross-branch
   120	consistency learning is beneficial. For example, on OccludedSYSU-MM01 (all-search), the performance increases from
   121	62.63%/60.29% to 62.81%/60.77% in Rank-1/mAP, and on
   122	Occluded-RegDB (V - I), it improves from 73.69%/51.15% to
   123	73.94%/52.76%. However, MCA achieves the best results in
   124	all settings, further improving performance to 63.82%/62.62%
   125	on Occluded-SYSU-MM01 and 75.63%/54.76% on OccludedRegDB. These results indicate that aligning both branches
   126	to a pseudo-holistic reference is more effective than directly
   127	forcing them to match each other, since the mean-reference
   128	target provides a more balanced and symmetric supervision
   129	signal, reducing the risk that noisy predictions from one branch
   130	directly bias the other during optimization.
   131	3) Comparison with Random Erasing: Random Erasing
   132	(RE) is a commonly used data augmentation strategy that
   133	applies stochastic rectangular masking during training. While
   134	effective for improving general robustness, RE introduces
   135	uniform rectangular regions without semantic awareness or
   136	structural constraints. In contrast, our occlusion generation
   137	strategy is designed to simulate realistic object-level occlusion
   138	commonly observed in surveillance scenarios. Specifically, the
   139	simulated occlusions are derived from real object instances
   140	and overlaid onto pedestrian images while preserving natural
   141	object boundaries and contextual consistency.
   142	As shown in Table III, the two strategies differ in occlusion
   143	shape, diversity of occlusion types, semantic consistency,
   144	boundary awareness, and cross-modality alignment. Fig. 8
   145	further illustrates that RE produces simple rectangular masks,
   146	whereas our approach generates irregular, object-aware occlusion patterns aligned with natural object contours. These
   147	differences indicate that our strategy aims to improve realism
   148	and structural coherence in occlusion simulation, rather than
   149	merely introducing stochastic masking.
   150	
   151	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
   152	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   153	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   154	
   155	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   156	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
   157	
   158	10
   159	
   160	TABLE IV: Comparison on Occluded-SYSU-MM01 dataset
   161	with complexity analysis.
   162	
   163	Indoor-search
   164	
   165	All-search
   166	65.00
   167	
   168	Rank-1
   169	
   170	Rank-1
   171	
   172	mAP
   173	
   174	mAP
   175	
   176	71.00
   177	
   178	Method
   179	
   180	Occ-SYSU-MM01
   181	Complexity
   182	R1
   183	mAP
   184	Params (M) FLOPs (G) Time (ms)
   185	PMT [35]
   186	50.75
   187	51.14
   188	86.11
   189	18.88
   190	1.05
   191	OCMF [6]
   192	54.66
   193	53.10
   194	53.79
   195	15.26
   196	0.66
   197	56.27
   198	54.43
   199	89.04
   200	29.43
   201	1.98
   202	DEEN [36]
   203	CSDN [37]
   204	60.24
   205	59.16
   206	84.01
   207	16.82
   208	1.10
   209	B
   210	58.32
   211	56.69
   212	31.49
   213	11.69
   214	0.53
   215	B+G
   216	60.47
   217	59.88
   218	52.63
   219	16.39
   220	0.81
   221	B + G + M 61.49
   222	60.15
   223	52.63
   224	16.39
   225	0.81
   226	B+S
   227	60.14
   228	59.43
   229	47.80
   230	14.57
   231	0.75
   232	61.35
   233	60.03
   234	47.80
   235	14.57
   236	0.75
   237	B+S+M
   238	B+G+S
   239	62.63
   240	60.29
   241	68.94
   242	19.27
   243	1.09
   244	OCCNet(Ours) 63.82
   245	62.62
   246	68.94
   247	19.27
   248	1.09
   249	
   250	63.82
   251	
   252	64.00
   253	
   254	69.88
   255	
   256	63.34
   257	
   258	63.00
   259	
   260	63.21

exec
/bin/zsh -lc "pdftotext -f 9 -l 12 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '1,320p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	8606
     2	
     3	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025
     4	
     5	occluded pedestrian re-identification models based on pedestrian parsers (or semantic segmentation) and pose estimation,
     6	respectively. The third group represents methods based on
     7	Transformers. The results demonstrate that the proposed PSCR
     8	model achieves an R-1 accuracy of 73.4% and a mAP of 63.1%
     9	on the Occluded-DukeMTMC dataset, surpassing the listed
    10	representative occluded pedestrian re-identification methods.
    11	Compared to the first group of methods, the proposed PSCR
    12	model achieves state-of-the-art performance in both R-1 accuracy and mAP metrics. This demonstrates the superiority of
    13	the skeleton mask generation approach based on pose keypoints over semantic masks produced by image parsers. The
    14	enhanced robustness can be attributed to the following factors:
    15	1) Image parsers exhibit sensitivity to low-resolution inputs
    16	and severe occlusions due to their reliance on fine-grained
    17	perception, whereas pose estimation models leverage highlevel semantic features of local regions. 2) The confidence
    18	information derived from pose estimation enables reasonable
    19	inference of occluded body parts, stabilizing the modeling
    20	of skeletal semantic regions and yielding more robust local
    21	features despite potential spatial prediction errors in keypoints.
    22	For ResNet-based methods in the second group, PSCR
    23	significantly outperforms PGFA [8] and PAFM [36] in R-1
    24	accuracy. Compared to RFCnet, PSCR achieves a 7% improvement in R-1 accuracy and a 6.6% increase in mAP. While
    25	RFCnet requires both pose encoding and additional semantic
    26	segmentation masks for feature extraction, PSCR simplifies
    27	the pipeline by modeling visible regions solely through pose
    28	information. Furthermore, conventional pose-guided methods overly depend on external models by introducing pose
    29	information post-backbone. In contrast, the proposed VARA
    30	module synergistically integrates attention mechanisms with
    31	the backbone network, providing occlusion-aware cues during feature extraction through soft fusion, thereby mitigating
    32	excessive reliance on external semantic models.
    33	When compared to Transformer-based methods in the third
    34	group, PSCR-Vit achieves a 1.5% R-1 accuracy advantage
    35	on the DukeMTMC-ReID dataset. Notably, it surpasses poseintegrated Transformer approaches like MTIPE [40] and
    36	KARR [41] by at least +5.7% in R-1 and +3.5% in mAP.
    37	The limitations of MTIPE [40] stem from its coarse mask
    38	generation via the Pose Aggregation Module (PAM), while
    39	KARR’s [41] strategy of learning features beyond heatmaps
    40	by eliminating keypoints lacks spatial guidance. In contrast,
    41	PSCR enhances localization precision through mathematical
    42	fusion of confidence features to expand inter-keypoint coverage areas. Additionally, its robustness to image resolution
    43	variations makes it particularly effective for handling occlusions and appearance changes in pedestrian re-identification
    44	tasks.
    45	2) Result on the Complete Pedestrian Re-Identification
    46	Datasets: Table II presents an evaluation of the proposed
    47	method and several mainstream approaches on comprehensive datasets for pedestrian re-identification. The collected
    48	methods are categorized into two main groups: complete
    49	pedestrian re-identification methods and occluded pedestrian
    50	re-identification methods utilizing pose estimation. The results
    51	demonstrate that the proposed method achieves impressive R-1
    52	
    53	TABLE III
    54	A BLATION E XPERIMENTS ON T HREE O CCLUDED DATASETS , O CCLUDED D UKE MTMC, PARTIAL -R E ID, AND PARTIAL - I LIDS DATASETS
    55	
    56	accuracies of 96.1% and 91.2% on the two datasets, respectively, surpassing the performances of nearly all occluded
    57	pedestrian re-identification models utilizing pose estimation.
    58	In unconcealed environments, the confidence level of the
    59	pose estimation results is higher. The outstanding performance of the proposed method can be attributed to its
    60	precise localization of local pedestrian regions, comprehensive
    61	extraction of appearance features, and avoidance of the loss
    62	of discriminative appearance features. When compared to
    63	the pose estimation-based SRNet [44], the proposed PSCR
    64	method exhibits significant advantages on the DukeMTMCReID dataset, yielding a remarkable improvement of 3.9% in
    65	R-1 accuracy and 6.7% in mAP. Although our PSCR method
    66	does not outperform task-specific models designed exclusively
    67	for complete pedestrian re-identification tasks (such as PFD
    68	[11]), it achieves competitive performance. Moreover, the
    69	proposed SAM strategy, serving as a decoupled local semantic
    70	positioning algorithm, can be integrated with these cuttingedge approaches to further enhance the model’s performance.
    71	In summary, the proposed PSCR method demonstrates strong
    72	generalization capabilities in the context of complete pedestrian scenes.
    73	C. Ablation Experiment
    74	(1) Module validity analysis Firstly, in order to verify the
    75	effectiveness of various modules in our method, this section conducts ablation studies on the Occluded-DukeMTMC,
    76	Partial-ReID, and Partial-iLIDS datasets, and the experimental
    77	results are shown in Table III. “Baseline” refers to the method
    78	of learning pedestrian local features using only the multiplication of keypoint heatmaps and the output feature maps
    79	of backbone (Res-50). VARA, SAM, and SMCF respectively
    80	represent the appearance region attention module, the skeletal
    81	region modeling module, and the cross-attention fusion module modulated by skeletons.
    82	The results presented in the first two rows of Table III
    83	demonstrate that incorporating VARA into the baseline network leads to a notable enhancement in the model’s R-1
    84	accuracy by 2.9% and mAP by 2% on Occluded-DukeMTMC.
    85	These outcomes highlight the beneficial impact of integrating
    86	keypoint information into the intermediate layers of the network, effectively improving the perception of occlusions and
    87	discriminative features associated with pedestrians. Furthermore, the substantial improvement in R-1 accuracy observed
    88	in the third set of experimental results, surpassing the baseline
    89	
    90	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
    91	
    92	GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION
    93	
    94	TABLE IV
    95	P ERFORMANCE R ESULTS ON VARIOUS BACKBONE M ETHODS
    96	
    97	model by 5.8%, provides supporting evidence that the proposed SAM not only generates comprehensive masks for local
    98	areas of pedestrians but also more elaborately considers the
    99	appearance shape and keypoint confidence of each local area,
   100	thereby avoiding the introduction of occlusion information
   101	and background noise. On the contrary, relying solely on
   102	keypoint masks for extracting pedestrian local features may
   103	result in the loss of valuable appearance information. In similar
   104	fashion, on the Partial-ReID and Partial-iLIDS datasets, the
   105	combination of the baseline model and SAM demonstrates
   106	significant improvements in the R-1 metric, with an increment
   107	of 5.2% and 3.2% respectively when compared to the baseline
   108	alone.
   109	Notably, the experimental results in the fourth group
   110	highlight that the incorporation of VARA and SAM modules further enhances the model’s performance. This can be
   111	attributed to the VARA module’s ability to partially suppress
   112	occlusion information in the output feature maps of the
   113	ResNet backbone network, while the precise localization of
   114	skeletal regions by SAM facilitates the extraction of more
   115	discriminative local features. Moreover, the latter two groups
   116	of experiments provide further evidence that the utilization
   117	of skeletal masks and the associated constraints on local
   118	areas facilitate the fusion of high-resolution and semantically
   119	stronger features within a specified spatial range. This fusion
   120	process aids in aligning the semantics of local features and
   121	ultimately improves their robustness.
   122	D. Further Analysis of VARA
   123	1) The Location of the VARA Module in Backbone Network:
   124	Furthermore, in order to investigate the impact of the VARA
   125	module in the middle layers of the backbone network on the
   126	performance of the model, a variant of the VARA network
   127	is validated on the Occluded-DukeMTMC dataset, as shown
   128	in Table IV. “P” indicates that the Conv.X layer and VARA
   129	
   130	8607
   131	
   132	module are connected in parallel, while “S” indicates that they
   133	are connected in series (Conv.X layer first, VARA module
   134	second). “DenseV” denotes that the VARA module is applied
   135	to all Conv.X layers, i.e., in a dense connection mode. “PSCR
   136	w/o V” acts as the baseline network obtained by removing the
   137	VARA module from the proposed method for this experimental
   138	study.
   139	Table IV This study conducts an in-depth analysis of the
   140	performance of pedestrian re-identification models from three
   141	aspects: the positioning of the VARA module, connection
   142	patterns, and input sizes. First, the influence of the VARA
   143	module’s position on model performance significantly relies
   144	on the feature extraction layers it is embedded within.
   145	When ResNet is employed as the backbone network with
   146	an input size of 256 × 128, applying the VARA module
   147	solely to the low-level layer Conv.B1 results in a marginal
   148	improvement in R-1 accuracy (+0.4%) under the serial connection mode, while a slight degradation (−0.1%) is observed
   149	under the parallel connection mode. Similarly, applying the
   150	VARA module exclusively to the high-level layer Conv.B4
   151	yields limited performance gains. In contrast, applying the
   152	VARA module specifically to the mid-level layer Conv.B3
   153	demonstrates a significant performance enhancement, with the
   154	serial and parallel connection modes improving R-1 accuracy
   155	by 2.2% and 1.9%, respectively. This can be attributed to
   156	the fact that low-level layers lack robust semantic features,
   157	making it challenging for the VARA module to effectively
   158	extract identity-related cues. Meanwhile, the high-level layers
   159	suffer from low feature resolution, resulting in coarse keypoint
   160	masks that diminish the effectiveness of pose cues.
   161	By increasing the input resolution (e.g., to 384 × 128 and
   162	384 × 256), the feature resolution of the mid- and high-level
   163	layers is enhanced, allowing certain information to become
   164	clearer and better preserved. Consequently, the model’s performance improves under higher-resolution inputs. However,
   165	this does not imply that performance will consistently improve
   166	with larger input sizes. For example, R-1 accuracy does not
   167	show further improvement when comparing 384 × 128 to
   168	384×256. This may be attributed to the relatively low original
   169	resolution of pedestrian datasets, where even upscaling fails
   170	to adequately reveal discriminative features, thereby limiting
   171	further performance gains.
   172	Furthermore, integrating the VARA module into both
   173	Conv.B2 and Conv.B3 optimizes performance by effectively
   174	leveraging the informative features of the intermediate layers.
   175	This approach preserves detailed information, enabling the
   176	model to capture pedestrian characteristics without significant
   177	information loss. However, densely embedding the VARA
   178	module increases GPU resource consumption. Under limited
   179	hardware resources, performance degradation may occur due
   180	to insufficient batch sizes.
   181	Similar to ResNet, the performance of the VARA module
   182	within the ViT backbone network also exhibits analogous
   183	characteristics. Integrating the VARA module into low-level
   184	and high-level layers provides less performance improvement
   185	compared to its integration into mid-level layers. However,
   186	overall, the combination of the ViT backbone with the VARA
   187	module outperforms the combination of ResNet with the
   188	
   189	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   190	
   191	8608
   192	
   193	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025
   194	
   195	Fig. 9. The impact of different settings of parameters α and γ.
   196	
   197	Fig. 8. The performance comparison of Fd and Fin for Key feature spaces
   198	fusion.
   199	
   200	VARA module. This indicates that ViT, through its selfattention mechanism, captures global contextual information,
   201	and when combined with the VARA module, it can more
   202	effectively extract detailed pedestrian features.
   203	2) The Performance Comparison of Fd and Fin for Key
   204	Feature Spaces Fusion in VARA: In the Key Feature Spaces
   205	Fusion submodule of VARA, we choose to utilize the feature
   206	map Fd , which has been processed through the foregroundaware submodule for channel weighting, instead of the original
   207	feature map Fin . This decision is primarily based on the fact
   208	that Fd can more effectively focus on pedestrian features
   209	while reducing interference from background and occlusion
   210	information (Fig. 8). The original feature map Fin contains
   211	rich background information, which, although beneficial for
   212	enhancing global contextual understanding, can be detrimental
   213	in pedestrian re-identification tasks, leading to challenges in
   214	accurately locating and identifying pedestrians.
   215	Through the foreground-aware submodule, we achieve
   216	channel weighting for Fin , a process that emphasizes pedestrian features and suppresses redundant information related
   217	to the background. This weighting results in the generated
   218	feature map Fd being more precise in representing pedestrian
   219	characteristics, thereby aiding the model in more effectively
   220	capturing important features relevant to pedestrians during
   221	cross-attention computations.
   222	Specifically, experimental results indicate that utilizing Fd
   223	as input improved key metrics such as Rank-1 identification
   224	rate (by 0.4%) and mAP (by 0.3%) compared to directly using
   225	Fin . This improvement validates the design choice of employing channel-weighted feature maps, effectively demonstrating
   226	its efficacy in enhancing pedestrian recognition accuracy.
   227	E. Further Analysis of SAM
   228	1) The Settings of Parameters α and γ: Fig. 9 depicts the
   229	impact of different settings of parameters α and γ in the SAM
   230	model on the coverage area of the mask (where α refers to
   231	the expansion coefficient of the overall radiation scale, and γ
   232	represents the transverse compression coefficient of the radiation scale). A larger α value indicates a larger radiation scale,
   233	
   234	Fig. 10. The impact of different settings of parameters α and γ on the
   235	Occluded-DukeMTMC.
   236	
   237	while a larger γ value implies greater transverse compression
   238	between nodes, resulting in a narrower and elongated focus
   239	area. The results reveal that as the α value gradually increases
   240	with fixed γ value of 0.3, the model achieves a richer and
   241	more comprehensive focus on the pedestrian region. When
   242	α is set to 1, the model attains the optimal level of focus
   243	on the visible pedestrian area. This indicates that in order
   244	to achieve better coverage of the pedestrian region, within a
   245	certain range of compression, it is necessary to maximize the
   246	radiation range of the model, enabling coverage of pedestrian
   247	areas beyond key points. Furthermore, when α is fixed at 1, a
   248	γ value within the range of [0.1-0.3] results in a clearer focus
   249	on the pedestrian region. However, as γ continues to increase,
   250	the focus area becomes blurred. This demonstrates that setting
   251	an appropriate transverse expansion coefficient between key
   252	points can enhance the quality of the model’s focus area.
   253	However, excessive compression leads to elongated regions
   254	between nodes, causing some information overlap and loss.
   255	Moreover, Fig. 10 illustrates the impact of different parameter
   256	settings on the R-1 and mAP. The results indicate that the
   257	algorithm performs optimally when α is set to 1 and γ is set
   258	to 0.3. Therefore, in this study, α is set to 1, and γ is set
   259	to 0.3.
   260	2) The Settings of the Confidence Threshold Value T:
   261	To further analyze the confidence threshold value of T, we
   262	utilized ResNet-50 as the backbone and provided visualizations for varying threshold values (Fig. 11 and Fig. 12). The
   263	visualizations indicate that when T < 0.7, the mask contains
   264	a substantial amount of occlusion information. Specifically,
   265	in the range of T ∈ [0.3, 0.5], the mask primarily reflects
   266	
   267	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   268	
   269	GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION
   270	
   271	Fig. 11. The visualization of different settings of parameters T .
   272	
   273	8609
   274	
   275	Fig. 13. Comparison of the three radiation scales on the Occluded
   276	DUKEMTMC.
   277	
   278	Fig. 14. Visualization results of the mask of local skeletal areas.
   279	Fig. 12. The impact of different settings of parameters T .
   280	TABLE V
   281	C OMPARISON OF THE T HREE R ADIATION S CALES
   282	
   283	an accurate representation of pose information. Due to the
   284	relatively low value of T , certain low-confidence occluded
   285	areas are not effectively filtered out, resulting in inadequate
   286	perception of occluded parts and leading to a plateau in the
   287	algorithm’s performance (with Rank-1 and mAP stabilizing).
   288	As T continues to increase, specifically within the range of
   289	[0.6, 0.95], the mask’s ability to perceive occlusions enhances
   290	continuously. Some occluded areas are detected, and the
   291	algorithm mitigates their disruptive influence. At T = 0.7,
   292	the algorithm achieves optimal performance. However, as T
   293	increases further to the range of [0.8, 0.95], certain pedestrian
   294	parts are mistakenly classified as occluded areas, resulting in a
   295	gradual reduction of the mask area and significantly impacting
   296	the algorithm’s performance.
   297	Therefore, to better localize non-occluded regions of pedestrians while maintaining high performance of the algorithm,
   298	we select T = 0.7.
   299	3) Analysis of Variants of the Radiation Range γ: To
   300	analyze the effectiveness of the radiation range, this study
   301	conducted three radiation range experiments (settings in
   302	Table V) using three distinct methods. The methods progress
   303	from M1 , which uses a linear relationship for simple, direct
   304	analysis of stable datasets with minimal variation. M2 introduces an exponential function, improving responsiveness to
   305	
   306	directional changes and identifying rapid trend shifts, especially in dynamic data. M3 employs the square of cosine
   307	similarity, refining directional sensitivity and helping resolve
   308	details in occluded pedestrian scenes. This progression demonstrates the methods’ adaptability and enhanced performance
   309	across varying conditions.
   310	From the visualization in the Table V, it can be observed
   311	that M1 has relatively low accuracy and precision (R-1 70.4%,
   312	mAP 60.7%); M2 outperforms M1 in terms of accuracy and
   313	the localization of the visible regions of pedestrians(R-1 +
   314	0.2%); and M3 demonstrates the best performance (R-1 70.9%,
   315	mAP 61.1%) in complex occlusion scenarios, offering a more
   316	precise capture of the details of the radiation range.
   317	F. Visualization Analysis
   318	1) Visualization Analysis of Skeletal Region: To better
   319	demonstrate the superiority of the proposed method, this
   320	section shows the visualization effect of the mask images

exec
/bin/zsh -lc "pdftotext -f 9 -l 12 'Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '1,320p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	TIAN et al.: PART-BASED FEATURE COMPLEMENTARY DENOISING FOR UNSUPERVISED PERSON RE-IDENTIFICATION
     2	
     3	2145
     4	
     5	TABLE III
     6	C OMPARISON OF D IFFERENT M ETHODS ON M ARKET-1501, D UKE MTMC- RE ID AND MSMT17 DATASETS
     7	
     8	Fig. 3. T-SNE visualization of features extracted on the Market-1501 dataset under different model settings. This figure shows the feature distributions of
     9	different methods using T-SNE. (A) Baseline method (CCL); (B) Our method without MKD; (C) Our method without GFD; (D) Our complete method. Each
    10	color represents an identity, and compactness reflects feature consistency. The baseline exhibits large intra-class variation and inter-class overlap. Removing
    11	MKD or GFD respectively leads to less compact clusters, confirming the importance of both modules. Our full method achieves the most discriminative and
    12	compact feature distribution.
    13	TABLE IV
    14	TABLE V
    15	C OMPARISON S OTA M ETHODS ON PERSON X DATASET
    16	
    17	A BLATION S TUDY ON M ARKET-1501 AND D UKE MTMC- RE ID DATASETS
    18	
    19	Compared to the baseline, PFCS improves the mAP by
    20	2.0% on Market-1501 and 1.2% on DukeMTMC-reID,
    21	
    22	indicating that incorporating local features helps capture more
    23	fine-grained cues and mitigates feature bias. The third row
    24	
    25	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
    26	
    27	2146
    28	
    29	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
    30	
    31	Fig. 4. Visualization comparison with the baseline and our method on Top-5 retrieved images on Market1501. The images with green boxes mean correct
    32	identity, and the red boxes are wrong.
    33	
    34	Fig. 5. Influence of different hyper-parameters η1 , η2 on Market-1501 and DukeMTMC-reID datasets. The left y-axis denotes the range of Rank-1, and the
    35	right y-axis indicates the range of mAP.
    36	
    37	demonstrates that integrating the MKD module into the baseline leads to mAP improvements of 2.5% and 2.0% on
    38	Market-1501 and DukeMTMC-reID, respectively, verifying
    39	the effectiveness of teacher-student guidance in enhancing
    40	feature robustness. The fourth row presents the result of
    41	incorporating the GFD module into the baseline. By using
    42	the Gaussian distribution assumption to reduce label noise
    43	below the confidence threshold, the mAP on the Market1501 and DukeMTMC-reID datasets increases by 2.9% and
    44	1.7%, respectively. The fifth row shows the effect of jointly
    45	applying PFCS and GFD, achieving an mAP increase of 3.5%
    46	on Market-1501 and 2.7% on DukeMTMC-reID compared to
    47	the baseline. The sixth row reflects the result of combining
    48	the baseline with PFCS, GFD, and MKD. This combination achieves an improvement of 5.1% in terms of mAP
    49	on the Market-1501 dataset and 4.2% on the DukeMTMCreID dataset over the baseline. The results indicate that the
    50	synergistic effect of these three modules yields optimal performance. The introduction of the teacher model further alleviates
    51	label noise in the early training stages, enhancing the overall
    52	performance.
    53	
    54	2) The influence of hyper-parameters: Figure 5 illustrates the variations in performance of our method on the
    55	Market-1501 [53] and DukeMTMC-reID [54] datasets as the
    56	hyper-parameters η1 and η2 take on different values. Specifically, on the Market-1501 dataset, changes in η1 show stable
    57	performance, indicating the robustness of our method to this
    58	hyper-parameter. In contrast, when η2 deviates from 0.7–either
    59	increasing or decreasing—there is a slight performance drop.
    60	This indicates that an inappropriate η2 value can lead to
    61	either insufficient or excessive guidance from the teacher
    62	model, negatively impacting the generalization of feature
    63	representation. On the DukeMTMC-reID dataset, as shown
    64	in Figures 5c and 5d, the impact of η1 and η2 follows a
    65	similarly stable trend as observed on Market-1501, further
    66	demonstrating the robustness of our method to variations in
    67	these hyper-parameters. Based on these findings, we set η1
    68	to 0.5 and η2 to 0.7, achieving optimal performance across
    69	both datasets. These settings are subsequently applied to other
    70	datasets as well.
    71	3) The influence of balancing factors: Balancing factors
    72	adjust the weight ratios of global and local features, enabling
    73	
    74	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
    75	
    76	TIAN et al.: PART-BASED FEATURE COMPLEMENTARY DENOISING FOR UNSUPERVISED PERSON RE-IDENTIFICATION
    77	
    78	2147
    79	
    80	Fig. 6. Influence of different balancing factors λ1 , λ2 and λ3 on Market-1501 and DukeMTMC-reID datasets. The left y-axis denotes the range of Rank-1,
    81	and the right y-axis indicates the range of mAP.
    82	TABLE VI
    83	I NFLUENCE OF D IFFERENT M OMENTUM PARAMETER m ON M ARKET-1501
    84	AND D UKE MTMC- RE ID DATASETS
    85	
    86	Fig. 7. Influence of different confidence threshold δ on Market-1501 and
    87	DukeMTMC-reID datasets. The left y-axis denotes the range of Rank-1, and
    88	the right y-axis indicates the range of mAP.
    89	
    90	our model to learn the salient features within global characteristics while also exploring discriminative local clues.
    91	Figures 6a, 6b, and 6c illustrate the performance of our
    92	method under different values of the balancing factors λ1 , λ2 ,
    93	and λ3 in the Market-1501 dataset. It can be observed that
    94	when λ2 is set to 0.1, the mAP increases, but the Rank-1
    95	decreases. To achieve the optimal overall performance, we set
    96	λ1 , λ2 , and λ3 to 0.2, 0.2, and 0.3, respectively. Similarly,
    97	on the DukeMTMC-reID dataset, as shown in Figures 6d,
    98	6e, and 6f, the model achieves the best performance when
    99	λ1 and λ2 are set to 0.2, and λ3 is set to 0.3. This indicates
   100	that appropriate balancing factor settings can effectively allocate weights between global and local features, enabling the
   101	model to extract salient global characteristics while exploring
   102	discriminative local clues. For the sake of consistency and
   103	simplicity, we apply these optimal settings (λ1 = 0.2, λ2 = 0.2,
   104	λ3 = 0.3) to all datasets.
   105	
   106	4) The influence of momentum parameter: The momentum
   107	parameter [25] determines the update rate, and the smaller
   108	the momentum parameter, the faster the memory bank is
   109	updated. We explore the impact of different momentum values
   110	on Market-1501 and DukeMTMC-reID datasets, as illustrated
   111	in Table VI. Notably, the best results are produced when
   112	the momentum parameter is set to 0.3. When the momentum
   113	parameter deviates from 0.3, either increasing or decreasing,
   114	the accuracy on the datasets shows a slight decline. Therefore,
   115	for simplicity and to maintain optimal performance, we set
   116	the momentum coefficient m to 0.3 across all datasets in our
   117	experiments.
   118	5) The influence of confidence threshold: Figure 7 illustrates the influence of varying values of the confidence
   119	threshold δ on the performance of model for the Market-1501
   120	and DukeMTMC-reID datasets. Specifically, for both datasets,
   121	as the confidence threshold δ increases, the mAP gradually
   122	
   123	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   124	
   125	2148
   126	
   127	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   128	
   129	TABLE VII
   130	
   131	TABLE IX
   132	
   133	P ERFORMANCE C OMPARISON U NDER D IFFERENT PART-BASED D IVISION
   134	S TRATEGIES ON M ARKET-1501 AND D UKE MTMC-R EID
   135	
   136	P ERFORMANCE C OMPARISON U SING P SEUDO -L ABELS G ENERATED BY
   137	D IFFERENT C LUSTERING A LGORITHMS ON M ARKET-1501
   138	
   139	TABLE VIII
   140	T HE T IME AND S PACE C OMPLEXITY ON M ARKET-1501
   141	
   142	declines, with the optimal value achieved when δ is set to 0.05.
   143	This indicates that denoising samples with low confidence can
   144	effectively enhance the performance of model. However, if
   145	the denoising range is expanded indiscriminately, it may lead
   146	to a decrease in performance instead. Therefore, to maintain
   147	optimal performance, we set the confidence threshold δ to 0.05
   148	across all datasets in our experiments.
   149	6) The impact of part division strategy: Table VII summarizes the performance of our method under different
   150	part-based feature division strategies on the Market-1501 and
   151	DukeMTMC-reID datasets. Specifically, we compare splitting
   152	the input image into 2, 3, and 4 horizontal parts. The results
   153	show that using 2 parts achieves the best performance, with an
   154	mAP of 87.7% and Rank-1 of 94.7% on Market-1501. Increasing the number of parts does not further improve performance
   155	and may introduce unnecessary noise or redundancy in the
   156	local features.
   157	7) The time and space complexity: The Table VIII summarizes the training time and model complexity for the
   158	baseline, teacher, and student models on the Market-1501
   159	dataset. The baseline model demonstrates lower computational
   160	requirements, with 4.09 MMac FLOPs, 23.51M parameters,
   161	and a training time of 1h35m17s. In contrast, our proposed
   162	method, which includes both the teacher and student models,
   163	exhibits higher computational complexity, with 8.16 MMac
   164	FLOPs and 23.72M parameters for both models. The training
   165	times for the teacher and student models are 2h19m40s and
   166	2h43m38s, respectively.
   167	Compared to the baseline, the increase in model complexity
   168	is primarily attributed to the design of the CF module in the
   169	model, which is used to obtain combined features. In addition,
   170	the increased training time in the teacher model mainly stems
   171	from GFD and PFCS module. For the student model, the
   172	extended training time is primarily due to the use of the pretrained teacher model to guide the learning process of student
   173	during training. Although the teacher assistance in training the
   174	student model introduces additional computational overhead
   175	and prolongs the training time, significant improvements in
   176	both mAP and Rank-1 are achieved compared to the baseline
   177	
   178	model. Therefore, the proposed method further enhances the
   179	accuracy and robustness of pseudo-labels, demonstrating that
   180	the trade-off between training efficiency and improved learning
   181	capability is well justified.
   182	8) The impact of different clustering algorithms: To
   183	evaluate the impact of different clustering algorithms on
   184	pseudo-label quality, we compare two representative clustering methods commonly used in USL ReID: K-Means and
   185	DBSCAN. As shown in Table IX, the model trained with
   186	pseudo-labels generated by K-Means performs significantly
   187	worse than the one using DBSCAN, with both mAP and Rank1 showing large declines. This suggests that the pseudo-labels
   188	produced by K-Means are of lower quality, while DBSCAN
   189	is more effective in generating reliable labels, making it more
   190	suitable for USL ReID. This also reflects the prevailing trend
   191	in USL ReID in recent years, where DBSCAN is often adopted
   192	instead of K-Means for pseudo-label generation.
   193	V. C ONCLUSION
   194	In this paper, we design a feature fusion denoising method
   195	based on local features, especially for USL ReID. The method
   196	includes three meticulously designed modules: PFCS, GFD,
   197	and MKD, along with CF for obtaining combined features.
   198	The CF and PFCS module, based on global features, provide a
   199	richer set of local fine-grained clues, reducing the inherent bias
   200	of global features. The GFD module uses a Gaussian Mixture
   201	Model to denoise low-confidence samples, thereby mitigating
   202	the impact of pseudo-label noise. Lastly, the MKD module
   203	employs a well-trained teacher model to guide the student
   204	model, enhancing its discriminative ability during the initial
   205	training phase and further improving the denoising effect. Our
   206	method demonstrates superior performance across four complex ReID datasets, significantly exceeding the performance
   207	of most current methods. In the future, we will explore more
   208	denoising methods and fusion strategies to enhance model
   209	discriminative capability, extending these approaches to other
   210	specific practical applications and other related research areas
   211	such as smart surveillance systems and unsupervised crossmodal ReID.
   212	R EFERENCES
   213	[1]
   214	[2]
   215	[3]
   216	
   217	M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C. H. Hoi, “Deep
   218	learning for person re-identification: A survey and outlook,” IEEE Trans.
   219	Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 2872–2893, Jun. 2022.
   220	J. Mao, Y. Yao, Z. Sun, X. Huang, F. Shen, and H.-T. Shen, “Attention
   221	map guided transformer pruning for occluded person re-identification on
   222	edge device,” IEEE Trans. Multimedia, vol. 25, pp. 1592–1599, 2023.
   223	X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, and X. Chen,
   224	“Clothes-changing person re-identification with RGB modality only,”
   225	in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Jun. 2022,
   226	pp. 1060–1069.
   227	
   228	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   229	
   230	
codex
证据部分现在比较完整：PFCD 和 PSCR 都有可视化加模块消融，不只是最终表格涨点。OIDM 还需要补完“最优光照距离替换”的验证段，尤其看它有没有证明不是单纯 GAN 变亮。
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
/bin/zsh -lc "pdftotext -f 12 -l 15 'Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	26:12
     2	
     3	C. Wang et al.
     4	
     5	Fig. 6. GAN converts images under different lighting to images under normal lighting.
     6	
     7	Fig. 7. Pedestrian search top-10 figure. Red rectangular boxes represent incorrect search results.
     8	
     9	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
    10	
    11	Optimal Illumination Distance Metrics for Person Re-Identification
    12	
    13	26:13
    14	
    15	Table 3. Combination of Different Single Light Conditions
    16	and Normal Light Conditions Metric Distances, Where mAP
    17	Represents the Value Under Single Light Conditions, and
    18	mAP(+Std) Represents the Result Under the Combination of
    19	Normal Light and Single Light Conditions Metric Matrix
    20	
    21	Illumination
    22	0.3
    23	0.4
    24	0.5
    25	0.6
    26	0.8
    27	1.2
    28	1.5
    29	1.8
    30	2.1
    31	2.5
    32	2.9
    33	3.3
    34	
    35	Market-1501
    36	Duke
    37	mAP mAP(+Std) mAP mAP(+Std)
    38	72.6
    39	75.7
    40	62.9
    41	64.2
    42	73.5
    43	75.8
    44	63.3
    45	64.2
    46	73.8
    47	75.9
    48	63.4
    49	64.2
    50	74.0
    51	63.4
    52	87.2
    53	64.3
    54	74.2
    55	75.9
    56	63.2
    57	64.3
    58	74.2
    59	75.9
    60	63.0
    61	64.2
    62	73.9
    63	75.9
    64	62.8
    65	64.2
    66	73.5
    67	75.8
    68	62.5
    69	64.2
    70	72.9
    71	75.8
    72	62.0
    73	64.1
    74	71.4
    75	75.8
    76	60.9
    77	64.0
    78	69.0
    79	75.5
    80	58.7
    81	63.4
    82	65.1
    83	75.1
    84	55.2
    85	91.9
    86	
    87	also found that the high-illumination part appears to be more useful than the low-illumination part,
    88	which in a way suggests that the optimal illumination may be more favorable for high illumination
    89	than for low illumination, i.e., high illumination would be less useful for image loss than low
    90	illumination. Also the data under bad lighting conditions (2.5, 2.9, 3.3) can have a relatively small
    91	effect on the original distance matrix, for example, by comparing the mAP metric of 75.0 under
    92	BOT+GeM+GAN conditions. For example, by comparing the mAP metric under BOT+GeM+GAN
    93	conditions, the mAP metric is 75.0, while the mAP metric under the combination of 3.3 and normal
    94	lighting conditions is only 75.1, which shows that the data under bad lighting conditions cannot
    95	effectively replace the distance metric formed under normal lighting conditions. Secondly, we found
    96	that the combination of single light and ideal light does not appear to be very sensitive to light
    97	conditions in the mAP metrics, for example, there is basically no change in the mAP between light
    98	(0.3–2.5) on the Duke dataset, and also on the Market-1501 dataset, and we also found that the
    99	effect of the mAP appears to be better when the light changes closer to normal light. This indicates
   100	that the distribution of optimal illumination for most images is not in the extreme case, but tends
   101	to fluctuate within a relatively small range of normal illumination. Finally, we believe that when it
   102	is difficult to find a clear pattern of optimal lighting, a pre-defined series of lighting methods can
   103	effectively improve the index of mAP, compared to the current index also proposed.
   104	Optimal Illumination Selection vs. Completely Random Selection, Partially Random Selection, and
   105	Fixed Value Selection. To verify the effectiveness of using optimal illumination substitution, we
   106	compared it with completely random substitution, partially random substitution, and fixed-value
   107	substitution. Specifically: (1) Completely random selection, in which we randomly selected the
   108	metric distance of each image pair under 13 lighting conditions as its final distance in the test phase
   109	for forming the final metric matrix. (2) Partially random selection, in the test phase, the distance
   110	under normal lighting conditions is used as a reference, and some images are randomly selected to
   111	replace the distance under a certain lighting condition, which is used to form the final metric matrix.
   112	(3) Fixed value selection, we selected the light conditions closer to the normal light conditions (1.2)
   113	and the low light conditions (3.3) and high light conditions (0.3) as the most three fixed values,
   114	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   115	
   116	26:14
   117	
   118	C. Wang et al.
   119	Table 4. Optimal Illumination Selection vs. Completely Random Selection,
   120	Partially Random Selection, and Fixed Value Selection
   121	
   122	Conditions
   123	mAP
   124	Completely random selection 63.2
   125	Partially random selection
   126	63.8
   127	Fixed value (0.3)
   128	63.9
   129	Fixed value (1.2)
   130	64.1
   131	Fixed value (3.3)
   132	62.6
   133	Optimal illumination selection 64.3
   134	
   135	Rank-1 Rank-5 Rank-10
   136	81.1
   137	90.0
   138	93.0
   139	81.2
   140	90.0
   141	93.1
   142	81.2
   143	90.1
   144	93.2
   145	81.4
   146	90.2
   147	93.0
   148	80.9
   149	89.9
   150	92.9
   151	81.9
   152	90.3
   153	93.1
   154	
   155	and in the testing phase, the distance of the image pair under the normal light conditions was
   156	used as the reference, and the distance of the image pair under the fixed light bar was compared
   157	with the distance under the normal light conditions, and the smallest distance was selected as the
   158	distance of the image pair of the distance, which is used to form the final metric matrix. We finally
   159	conducted the experiments on Market-1501.
   160	As shown in Table 4, firstly, from the table, we can see that the optimal illumination replacement
   161	method achieved the best experimental results, providing direct evidence of the effectiveness of
   162	optimal illumination replacement. Secondly, we found that partial random selection methods were
   163	better than completely random selection, but not as good as the fixed value (1.2) replacement
   164	method. At the same time, they were better than the high illumination replacement with a fixed
   165	value of 0.3 and the low illumination replacement with a fixed value of 3.3. This means that under
   166	normal illumination conditions, the semantic distance between the majority of image pairs can
   167	be effectively expressed, with only a small number of image pairs having less accurate semantic
   168	expression. Moreover, the probability of these image pairs with less accurate semantic expression
   169	occurring near normal illumination conditions is higher, while the probability of them occurring
   170	under low illumination conditions or high illumination conditions is smaller. On the other hand,
   171	the semantic distance expression of image pairs under low illumination conditions is the poorest,
   172	as a large amount of detailed information in the images is lost in low illumination conditions.
   173	Optimal Illumination Replacement Analysis. In order to further analyze the effectiveness of the
   174	optimal illumination method, we used replacement and effective replacement as evaluation metrics.
   175	In terms of replacement, we compared the image pairs replaced by the optimal illumination method
   176	with the total number of image pairs, using the metric of the ratio of replaced pairs to the benchmark
   177	pairs formed by the illumination normalization method. Effective replacement refers to the ratio
   178	of the number of image pairs that moved closer to the correct sample’s previous position after
   179	replacement to the total number of replacements, compared to the benchmark pairs.
   180	From Figure 8, we can observe that the proportion of replacement by the optimal illumination
   181	method varies between the two datasets. The replacement ratio is lower on Market-1501 compared
   182	to DukeReID, while the effective replacement ratio is higher on Market-1501. This suggests that,
   183	despite the higher replacement ratio in DukeReID, the method’s ability to find truly optimal
   184	illumination conditions is limited. However, the overall effective replacement rate remains relatively
   185	low for both datasets. This phenomenon may be due to the limitations of the pre-defined choices
   186	of illumination conditions and the number of pre-defined illuminations. The optimal illumination
   187	found may not be the true optimal illumination, but rather the minimum value under the current
   188	pre-defined illumination conditions. As a result, the accuracy of finding the optimal illumination
   189	conditions is compromised, leading to a lower effective replacement rate and limited improvement
   190	in performance.
   191	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   192	
   193	Optimal Illumination Distance Metrics for Person Re-Identification
   194	
   195	26:15
   196	
   197	Fig. 8. Percentage statistics of replacement and effective replacement on Market-1501 and DukeMTMC-ReID
   198	datasets.
   199	
   200	Figure 8 highlights the contrast between the two datasets in terms of their replacement behaviors.
   201	DukeReID shows a notably higher replacement ratio but with fewer effective replacements. This
   202	indicates that, while more image pairs are being replaced under the optimal illumination strategy,
   203	fewer of these replacements are truly effective in improving the matching results. On the other
   204	hand, Market-1501 demonstrates a lower replacement ratio but with a higher effective replacement
   205	rate, indicating that the replacements made on this dataset are more impactful in improving person
   206	Re-ID performance.
   207	This discrepancy might be attributed to the nature of the datasets. DukeReID, with its more
   208	complex lighting conditions, likely presents more opportunities for sub-optimal illumination
   209	conditions, which the pre-defined set of illuminations cannot fully address. Meanwhile, Market1501, being a slightly simpler dataset in terms of lighting variability, allows the method to make
   210	more accurate replacements. Nevertheless, it is clear that the current method, while beneficial in
   211	certain cases, is still limited by the pre-defined illumination conditions. The optimal illumination
   212	may not always be found due to the fixed number of lighting conditions provided. As a result, the
   213	method is less effective at finding the best matches for more challenging image pairs in DukeReID,
   214	leading to lower overall performance.
   215	5.4
   216	
   217	Generalize to Unseen Light Scenarios
   218	
   219	To investigate the generalization of our proposed method on unseen lighting conditions, we
   220	constructed the Market-IA-Unseen and Duke-IA-Unseen datasets. The train and query datasets
   221	are the same as Market-1501 and Duke datasets, but the gallery dataset is different. We divided
   222	the range of 0.3–3.3 into 301 intervals with a step size of 0.01 to generate 301 different lighting
   223	conditions. For each image in the gallery dataset, we randomly selected one lighting condition from
   224	the 301 conditions and generated a new image. Therefore, the gallery database includes images
   225	with 301 different lighting conditions.
   226	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   227	
   228	
exec
/bin/zsh -lc "pdftotext -f 12 -l 15 'Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	2106
     2	
     3	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
     4	
     5	TABLE VI
     6	C OMPARISON B ETWEEN THE P ROPOSED M ETHOD AND S TATE - OF - THE -A RT A LGORITHMS . T HE R ESULTS ON F OUR TARGET P ERSON R E -ID DATASETS ,
     7	M ARKET-1501 [32], MSMT17 [33], P ERSON X [34], AND V E R I -776 [57]. T RANSFER D ENOTES U NSUPERVISED D OMAIN A DAPTATION
     8	M ETHODS , W HICH L EVERAGE OTHER L ABELED S OURCE DATASETS . N ONE R EPRESENTS P URELY U NSUPERVISED M ETHODS , W HICH
     9	A RE T RAINED W ITHOUT A NY L ABELS . AGNOSTIC R EFERS TO C AMERA -AGNOSTIC M ETHODS T HAT O MIT C AMERA ID S , W HILE
    10	AWARE D ENOTES C AMERA -AWARE M ETHODS T HAT A DOPT C AMERA ID S FOR T RAINING
    11	
    12	TABLE VII
    13	C OMPARISON OF THE D IFFERENT K ERNEL S IZE S ETTINGS OF PKGC ON
    14	M ARKET-1501 AND MSMT17
    15	
    16	λ = 0, the balanced contrastive loss is completely depending
    17	on LFNPM and when λ = 1, it relies on the LHCPM . The performance trends on both the Market-1501 and MSMT17 datasets
    18	exhibited similar patterns. Initially, due to the propensity of
    19	the FNPM module to erroneously select proxies as positive
    20	samples, training often resulted in underfitting and ineffective
    21	convergence when the λ parameter was low (i.e. when λ < 0.2
    22	on Market-1501 and λ < 0.3 on MSMT17). On both datasets,
    23	optimal and consistently superior results were achieved when
    24	λ is equal to 0.6, surpassing results obtained by training
    25	solely with the LHCPM . This demonstrates the effectiveness
    26	of combining training methods and underscores the necessity
    27	for LHCPM to play a dominant role in the training process.
    28	5) The Sensitivity of PKGC Kernel Setting: As shown in
    29	Table VII, we present the comparison of kernel sizes in the
    30	proposed PKGC block, which corresponds to the kernel sizes
    31	of the convolutional layers prior to the softmax operation
    32	across four distinct stages in the backbone network [31]
    33	architecture (refer to the CNN Backbone in Fig. 1), with
    34	“S-1” representing stage 1, and so forth. The results of
    35	static settings reveal that using larger kernels enhances the
    36	
    37	integration of global contextual information, as reflected by
    38	the improved performance from 1 × 1 to 3 × 3 configurations.
    39	Static settings show that increasing kernel size enhances global
    40	context modeling, and the proposed pyramid arrangement further improves performance by capturing multi-scale semantic
    41	information across stages. Compared with the static settings,
    42	the pyramid design achieves the largest gains, while all tested
    43	configurations show positive improvements over the baseline.
    44	This demonstrates that PKGC is not overly sensitive to kernel
    45	choices and remains robust across different settings.
    46	D. Comparison to State-of-The-Art Methods
    47	This section evaluates the proposed method against recent
    48	unsupervised techniques, categorized into two groups: (1)
    49	unsupervised domain adaptation methods; (2) purely unsupervised methods. Table VI displays the results of these methods
    50	across four datasets. 1) Unsupervised Domain Adaptation
    51	Method Comparison: This analysis includes comparisons with
    52	various unsupervised domain adaptation methods such as ECN
    53	[10], MMCL [45], Ad-cluster [11], MMT [12], SpCL [5],
    54	TAL-MIRN [13], IDM [14], CaCL [15], and PDA [16]. Our
    55	method outperforms these methods in terms of Rank-1 and
    56	mAP scores on all datasets, even without the use of external
    57	labeled data. For instance, our method shows improvements
    58	over the next best, CaCL, by 1.6% in mAP and 0.6% in
    59	Rank-1 on Market-1501, and by 9.6% in mAP and 8.6% in
    60	Rank-1 on MSMT17. Additionally, our method shows gains
    61	of at least 2.3% in mAP and 5.0% in mAP on the vehicle
    62	Re-ID dataset Veri-776, demonstrating the model’s robust
    63	generalization capabilities. 2) Purely Unsupervised Method
    64	Comparison: Our method is benchmarked against completely
    65	
    66	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
    67	
    68	CAO et al.: OPTIMAL PROXY MINING CONTRASTIVE NETWORK FOR UNSUPERVISED PERSON RE-IDENTIFICATION
    69	
    70	unsupervised person Re-ID techniques such as BUC [17], SSL
    71	[18], MMCL [45], JVTC [29], SpCL [5], CAP [6], ICE [19],
    72	IICS [20], PPLR [8], CCL [7], STS [36], STDA [65], and
    73	DHCCN [9]. According to the results shown in Table VI,
    74	our model outperforms all competitors. Notable improvements
    75	include at least 0.7% mAP and 0.3% Rank-1 on Market-1501,
    76	9.7% mAP and 10.7% Rank-1 on MSMT17, 3.2% mAP and
    77	1.7% Rank-1 on PersonX, and 2.7% mAP and 2.3% Rank-1 on
    78	VeRi-776. Particularly, our CCL model consistently exceeds
    79	all other camera-aware methods listed in Table VI,(e.g., ECN,
    80	Ad-cluster, TAL-MIRN, SSL, CACL, JVTC, ICE, IICS, and
    81	CAP), further affirming the effectiveness of the proposed
    82	OPMCN.
    83	V. C ONCLUSION
    84	In the presented paper, we present the Optimal Proxy
    85	Mining Contrastive Network (OPMCN), a novel framework
    86	tailored for unsupervised person re-identification (Re-ID).
    87	OPMCN incorporates two innovative contrastive loss functions, namely, Hardest Camera Proxy Mining (HCPM) and
    88	False Negative Proxies Mining (FNPM). HCPM leverages
    89	assumed accurate pseudo labels to select challenging proxies from the toughest camera settings, focusing on the
    90	most distinctive semantic features within clusters to mitigate
    91	background variations and emphasize ID-specific attributes.
    92	Conversely, FNPM combats the degradation caused by pseudo
    93	label noise through an anti-noise similarity ranking, refining
    94	contrastive pair selection for contrast enhancement. Additionally, the introduction of the Pyramid Kernel Global Context
    95	(PKGC) block within OPMCN aids in the robust processing
    96	of spatial information and the seamless integration of global
    97	features, significantly boosting the network’s ability to discern
    98	identity-specific characteristics across varying scales and adapt
    99	to environmental changes. Collectively, these methodologies
   100	profoundly augment the efficacy of unsupervised training.
   101	Since the inherent noise challenges exist in unsupervised
   102	learning, the performance discrepancy between unsupervised
   103	and supervised models persists. Future work should focus on
   104	developing more robust mechanisms to enhance the precision
   105	of pseudo label generation and utilization.
   106	R EFERENCES
   107	[1]
   108	[2]
   109	[3]
   110	
   111	[4]
   112	
   113	[5]
   114	[6]
   115	[7]
   116	
   117	Z. Zheng, L. Zheng, and Y. Yang, “Pedestrian alignment network for
   118	large-scale person re-identification,” IEEE Trans. Circuits Syst. Video
   119	Technol., vol. 29, no. 10, pp. 3037–3045, Oct. 2019.
   120	X. Ning, K. Gong, W. Li, L. Zhang, X. Bai, and S. Tian, “Feature
   121	refinement and filter network for person re-identification,” IEEE Trans.
   122	Circuits Syst. Video Technol., vol. 31, no. 9, pp. 3391–3402, Sep. 2021.
   123	Y. Huang, J. Xu, Q. Wu, Y. Zhong, P. Zhang, and Z. Zhang, “Beyond
   124	scalar neuron: Adopting vector-neuron capsules for long-term person
   125	re-identification,” IEEE Trans. Circuits Syst. Video Technol., vol. 30,
   126	no. 10, pp. 3459–3471, Oct. 2020.
   127	P. Wang, Z. Zhao, F. Su, X. Zu, and N. V. Boulgouris, “HOReID:
   128	Deep high-order mapping enhances pose alignment for person reidentification,” IEEE Trans. Image Process., vol. 30, pp. 2908–2922,
   129	2021.
   130	Y. Ge, F. Zhu, D. Chen, and R. Zhao, “Self-paced contrastive learning
   131	with hybrid memory for domain adaptive object re-ID,” in Proc. NIPS,
   132	2020, pp. 11309–11321.
   133	M. Wang, B. Lai, J. Huang, X. Gong, and X.-S. Hua, “Camera-aware
   134	proxies for unsupervised person re-identification,” in Proc. AAAI Conf.
   135	Artif. Intell., May 2021, vol. 35, no. 4, pp. 2764–2772.
   136	Z. Dai, G. Wang, W. Yuan, S. Zhu, and P. Tan, “Cluster contrast for
   137	unsupervised person re-identification,” in Proc. Asian Conf. Comput.
   138	Vis. (ACCV), 2022, pp. 1142–1160.
   139	
   140	[8]
   141	
   142	2107
   143	
   144	Y. Cho, W. J. Kim, S. Hong, and S.-E. Yoon, “Part-based pseudo
   145	label refinement for unsupervised person re-identification,” in Proc.
   146	IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022,
   147	pp. 7308–7318.
   148	[9] Y. Li, W. Tang, S. Wang, S. Qian, and C. Xu, “Distribution-guided
   149	hierarchical calibration contrastive network for unsupervised person reidentification,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 8,
   150	pp. 7149–7164, Aug. 2024.
   151	[10] Z. Zhong, L. Zheng, Z. Luo, S. Li, and Y. Yang, “Invariance matters:
   152	Exemplar memory for domain adaptive person re-identification,” in
   153	Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
   154	2019, pp. 598–607.
   155	[11] Y. Zhai et al., “AD-Cluster: Augmented discriminative clustering for
   156	domain adaptive person re-identification,” in Proc. IEEE/CVF Conf.
   157	Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020, pp. 9021–9030.
   158	[12] Y. Ge, D. Chen, and H. Li, “Mutual mean-teaching: Pseudo label refinery
   159	for unsupervised domain adaptation on person re-identification,” 2020,
   160	arXiv:2001.01526.
   161	[13] H. Li, N. Dong, Z. Yu, D. Tao, and G. Qi, “Triple adversarial learning
   162	and multi-view imaginative reasoning for unsupervised domain adaptation person re-identification,” IEEE Trans. Circuits Syst. Video Technol.,
   163	vol. 32, no. 5, pp. 2814–2830, May 2022.
   164	[14] Y. Dai, J. Liu, Y. Sun, Z. Tong, C. Zhang, and L.-Y. Duan, “IDM: An
   165	intermediate domain module for domain adaptive person re-ID,” in Proc.
   166	IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 11864–11874.
   167	[15] M. Li, C.-G. Li, and J. Guo, “Cluster-guided asymmetric contrastive
   168	learning for unsupervised person re-identification,” IEEE Trans. Image
   169	Process., vol. 31, pp. 3606–3617, 2022.
   170	[16] T. Isobe, D. Li, L. Tian, W. Chen, Y. Shan, and S. Wang, “Towards
   171	discriminative representation learning for unsupervised person reidentification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct.
   172	2021, pp. 8526–8536.
   173	[17] Y. Lin, X. Dong, L. Zheng, Y. Yan, and Y. Yang, “A bottom-up clustering
   174	approach to unsupervised person re-identification,” in Proc. AAAI Conf.
   175	Artif. Intell., 2019, vol. 33, no. 1, pp. 8738–8745.
   176	[18] Y. Lin, L. Xie, Y. Wu, C. Yan, and Q. Tian, “Unsupervised person reidentification via softened similarity learning,” in Proc. IEEE/CVF Conf.
   177	Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020, pp. 3390–3399.
   178	[19] H. Chen, B. Lagadec, and F. Bremond, “ICE: Inter-instance contrastive
   179	encoding for unsupervised person re-identification,” in Proc. IEEE/CVF
   180	Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 14960–14969.
   181	[20] S. Xuan and S. Zhang, “Intra-inter camera similarity for unsupervised
   182	person re-identification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
   183	Recognit. (CVPR), Jun. 2021, pp. 11926–11935.
   184	[21] Y. He, X. Wei, X. Hong, W. Shi, and Y. Gong, “Multi-target multicamera tracking by tracklet-to-target assignment,” IEEE Trans. Image
   185	Process., vol. 29, pp. 5191–5205, 2020.
   186	[22] S. Jain et al., “Spatula: Efficient cross-camera video analytics on large
   187	camera networks,” in Proc. IEEE/ACM Symp. Edge Comput., Nov. 2020,
   188	pp. 110–124.
   189	[23] L. Chen, H. Ai, R. Chen, Z. Zhuang, and S. Liu, “Cross-view tracking
   190	for multi-human 3D pose estimation at over 100 FPS,” in Proc.
   191	IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020,
   192	pp. 3276–3285.
   193	[24] J. Dong et al., “Fast and robust multi-person 3D pose estimation and
   194	tracking from multiple views,” IEEE Trans. Pattern Anal. Mach. Intell.,
   195	vol. 44, no. 10, pp. 6981–6992, Oct. 2022.
   196	[25] M. Bilal, A. Khan, M. U. Karim Khan, and C.-M. Kyung, “A lowcomplexity pedestrian detection framework for smart video surveillance
   197	systems,” IEEE Trans. Circuits Syst. Video Technol., vol. 27, no. 10,
   198	pp. 2260–2273, Oct. 2017.
   199	[26] B.-H. Chen, L.-F. Shi, and X. Ke, “A robust moving object detection
   200	in multi-scenario big data for video surveillance,” IEEE Trans. Circuits
   201	Syst. Video Technol., vol. 29, no. 4, pp. 982–995, Apr. 2019.
   202	[27] G. Zhang, H. Zhang, W. Lin, A. K. Chandran, and X. Jing, “Camera
   203	contrast learning for unsupervised person re-identification,” IEEE Trans.
   204	Circuits Syst. Video Technol., vol. 33, no. 8, pp. 4096–4107, Aug. 2023.
   205	[28] A. Ram, S. Jalal, A. S. Jalal, and M. Kumar, “A density based algorithm
   206	for discovering density varied clusters in large spatial databases,” Int.
   207	J. Comput. Appl., vol. 3, no. 6, pp. 1–4, Jun. 2010.
   208	[29] J. Li and S. Zhang, “Joint visual and temporal consistency for unsupervised domain adaptive person re-identification,” in Proc. Eur. Conf.
   209	Comput. Vis. Cham, Switzerland: Springer, 2020, pp. 483–499.
   210	[30] D. Zheng, J. Xiao, M. Sun, H. Bai, and J. Hou, “Plausible proxy
   211	mining with credibility for unsupervised person re-identification,” IEEE
   212	Trans. Circuits Syst. Video Technol., vol. 33, no. 7, pp. 3308–3318,
   213	Jul. 2023.
   214	
   215	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   216	
   217	2108
   218	
   219	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   220	
   221	[31] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
   222	image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.
   223	(CVPR), Jun. 2016, pp. 770–778.
   224	[32] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian, “Scalable
   225	person re-identification: A benchmark,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Dec. 2015, pp. 1116–1124.
   226	[33] L. Wei, S. Zhang, W. Gao, and Q. Tian, “Person transfer GAN to
   227	bridge domain gap for person re-identification,” in Proc. IEEE/CVF
   228	Conf. Comput. Vis. Pattern Recognit., Jun. 2018, pp. 79–88.
   229	[34] X. Sun and L. Zheng, “Dissecting person re-identification from the
   230	viewpoint of viewpoint,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
   231	Recognit. (CVPR), Jun. 2019, pp. 608–617.
   232	[35] Z. Hu, Y. Sun, Y. Yang, and J. Zhou, “Divide-and-regroup clustering
   233	for domain adaptive person re-identification,” in Proc. AAAI Conf. Artif.
   234	Intell., 2022, vol. 36, no. 1, pp. 980–988.
   235	[36] T. Liu, Y. Lin, and B. Du, “Unsupervised person re-identification
   236	with stochastic training strategy,” IEEE Trans. Image Process., vol. 31,
   237	pp. 4240–4250, 2022.
   238	[37] K. Zeng, M. Ning, Y. Wang, and Y. Guo, “Hierarchical clustering
   239	with hard-batch triplet loss for person re-identification,” in Proc.
   240	IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020,
   241	pp. 13657–13665.
   242	[38] Y. Lin, X. Guo, Z. Wang, and B. Du, “Privacy-protected person reidentification via virtual samples,” IEEE Trans. Inf. Forensics Security,
   243	vol. 18, pp. 5495–5505, 2023.
   244	[39] Y. Movshovitz-Attias, A. Toshev, T. K. Leung, S. Ioffe, and S. Singh,
   245	“No fuss distance metric learning using proxies,” in Proc. IEEE Int.
   246	Conf. Comput. Vis. (ICCV), Oct. 2017, pp. 360–368.
   247	[40] X. Yao et al., “PCL: Proxy-based contrastive learning for domain
   248	generalization,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
   249	(CVPR), Jun. 2022, pp. 7097–7107.
   250	[41] T. Zhao, X. Guo, Y. Lin, and B. Du, “MixIR: Mixing input and
   251	representations for contrastive learning,” IEEE Trans. Neural Netw.
   252	Learn. Syst., vol. 36, no. 5, pp. 8255–8264, May 2025.
   253	[42] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum
   254	contrast for unsupervised visual representation learning,” in Proc.
   255	IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020,
   256	pp. 9729–9738.
   257	[43] J. Deng, J. Guo, J. Yang, A. Lattas, and S. Zafeiriou, “Variational
   258	prototype learning for deep face recognition,” in Proc. IEEE/CVF
   259	Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2021,
   260	pp. 11906–11915.
   261	[44] J. Li, P. Zhou, C. Xiong, and S. C. H. Hoi, “Prototypical
   262	contrastive learning of unsupervised representations,” 2020,
   263	arXiv:2005.04966.
   264	[45] D. Wang and S. Zhang, “Unsupervised person re-identification via multilabel classification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
   265	Recognit. (CVPR), Jun. 2020, pp. 10981–10990.
   266	[46] J. Robinson, C.-Y. Chuang, S. Sra, and S. Jegelka, “Contrastive learning
   267	with hard negative samples,” 2020, arXiv:2010.04592.
   268	[47] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,”
   269	in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jul. 2018,
   270	pp. 7132–7141.
   271	[48] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional
   272	block attention module,” in Proc. Eur. Conf. Comput. Vis., Sep. 2018,
   273	pp. 3–19.
   274	[49] Q. Hou, D. Zhou, and J. Feng, “Coordinate attention for efficient
   275	mobile network design,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
   276	Recognit. (CVPR), Jun. 2021, pp. 13713–13722.
   277	[50] Y. Cao, J. Xu, S. Lin, F. Wei, and H. Hu, “Global context networks,”
   278	IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 6, pp. 6881–6895,
   279	Jan. 2020.
   280	[51] B. Chen, W. Deng, and J. Hu, “Mixed high-order attention network
   281	for person re-identification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis.
   282	(ICCV), Oct. 2019, pp. 371–381.
   283	[52] Z. Zhang, C. Lan, W. Zeng, X. Jin, and Z. Chen, “Relation-aware global
   284	attention for person re-identification,” in Proc. IEEE/CVF Conf. Comput.
   285	Vis. Pattern Recognit. (CVPR), Jun. 2020, pp. 3186–3195.
   286	[53] H. Zhang et al., “Attribute-guided collaborative learning for partial person re-identification,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45,
   287	no. 12, pp. 14144–14160, Dec. 2023.
   288	[54] Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking person reidentification with K-reciprocal encoding,” in Proc. IEEE Conf. Comput.
   289	Vis. Pattern Recognit., Jul. 2017, pp. 1318–1327.
   290	[55] F. Wang and H. Liu, “Understanding the behaviour of contrastive loss,”
   291	in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
   292	2021, pp. 2495–2504.
   293	
   294	[56] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with
   295	contrastive predictive coding,” 2018, arXiv:1807.03748.
   296	[57] X. Liu, W. Liu, T. Mei, and H. Ma, “A deep learning-based approach
   297	to progressive vehicle re-identification for urban surveillance,” in
   298	Proc. Eur. Conf. Comput. Vis. Cham, Switzerland: Springer, 2016,
   299	pp. 869–884.
   300	[58] J. Riccitiello, “John Riccitiello sets out to identify the engine of
   301	growth for unity technologies,” VentureBeat, San Francisco, CA,
   302	USA, Tech. Rep., Oct. 2014. Accessed: Sep. 2, 2025. [Online].
   303	Available: https://venturebeat.com/2014/10/23/john-riccitiello-setsoutto-identify-the-engine-of-growth-for-unity-technologies-interview/
   304	[59] F. Radenovic, G. Tolias, and O. Chum, “Fine-tuning CNN image
   305	retrieval with no human annotation,” IEEE Trans. Pattern Anal. Mach.
   306	Intell., vol. 41, no. 7, pp. 1655–1668, Jul. 2019.
   307	[60] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet:
   308	A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput.
   309	Vis. Pattern Recognit., Jun. 2009, pp. 248–255.
   310	[61] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
   311	2014, arXiv:1412.6980.
   312	[62] A. Paszke et al., “PyTorch: An imperative style, high-performance deep
   313	learning library,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS),
   314	2019, pp. 8024–8035.
   315	[63] L. van der Maaten and G. E. Hinton, “Visualizing data using t-SNE,”
   316	J. Mach. Learn. Res., vol. 9, no. 86, pp. 2579–2605, 2008.
   317	[64] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and
   318	D. Batra, “Grad-CAM: Visual explanations from deep networks via
   319	gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis.
   320	(ICCV), Oct. 2017, pp. 618–626.
   321	[65] Q. He, Z. Wang, Z. Zheng, and H. Hu, “Spatial and temporal dualattention for unsupervised person re-identification,” IEEE Trans. Intell.
   322	Transp. Syst., vol. 25, no. 2, pp. 1953–1965, Feb. 2024.
   323	
   324	Ge Cao (Member, IEEE) received the bachelor’s
   325	degree in vehicle engineering from the School of
   326	Automotive Engineering, Shanghai University of
   327	Engineering Science, Shanghai, in 2019. He is currently pursuing the combined M.S. and Ph.D. degree
   328	with the Department of Electrical, Electronic and
   329	Computer Engineering, University of Ulsan, Ulsan,
   330	South Korea. His current research interests include
   331	computer vision and deep learning, focusing on
   332	object re-identification, human pose estimation, and
   333	multiple people tracking.
   334	
   335	Qing Tang (Member, IEEE) received the Ph.D.
   336	degree in electrical and computer engineering from
   337	the Graduate School of Electrical Engineering, University of Ulsan, South Korea, in 2022. She is
   338	currently a Senior Researcher with the Data Analysis Team, InterX, a leading AI solution provider
   339	for the manufacturing sector in South Korea.
   340	Her research focuses on intelligent manufacturing
   341	systems, advanced surveillance, and transportation
   342	systems, with a particular emphasis on the application of computer vision and machine learning
   343	technologies.
   344	
   345	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   346	
   347	CAO et al.: OPTIMAL PROXY MINING CONTRASTIVE NETWORK FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   348	
   349	Xuan-Thuy Vo (Member, IEEE) received the
   350	B.S. degree in electrical and electronic engineering from the University of Science and
   351	Technology—University of Da Nang, Da Nang,
   352	Vietnam, in 2018, and the Ph.D. degree from the
   353	Department of Electrical, Electronic and Computer
   354	Engineering, University of Ulsan, South Korea, in
   355	2024. He is currently with the Intelligent Systems
   356	Laboratory, University of Ulsan, as a Post-Doctoral
   357	Researcher. His current research interests include
   358	computer vision and deep learning, with a focus on
   359	efficient vision architectures, object detection, object segmentation, multiple
   360	people tracking, human action recognition, and multimodal large language

exec
/bin/zsh -lc "pdftotext -f 12 -l 15 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION
     2	
     3	Fig. 11. The visualization of different settings of parameters T .
     4	
     5	8609
     6	
     7	Fig. 13. Comparison of the three radiation scales on the Occluded
     8	DUKEMTMC.
     9	
    10	Fig. 14. Visualization results of the mask of local skeletal areas.
    11	Fig. 12. The impact of different settings of parameters T .
    12	TABLE V
    13	C OMPARISON OF THE T HREE R ADIATION S CALES
    14	
    15	an accurate representation of pose information. Due to the
    16	relatively low value of T , certain low-confidence occluded
    17	areas are not effectively filtered out, resulting in inadequate
    18	perception of occluded parts and leading to a plateau in the
    19	algorithm’s performance (with Rank-1 and mAP stabilizing).
    20	As T continues to increase, specifically within the range of
    21	[0.6, 0.95], the mask’s ability to perceive occlusions enhances
    22	continuously. Some occluded areas are detected, and the
    23	algorithm mitigates their disruptive influence. At T = 0.7,
    24	the algorithm achieves optimal performance. However, as T
    25	increases further to the range of [0.8, 0.95], certain pedestrian
    26	parts are mistakenly classified as occluded areas, resulting in a
    27	gradual reduction of the mask area and significantly impacting
    28	the algorithm’s performance.
    29	Therefore, to better localize non-occluded regions of pedestrians while maintaining high performance of the algorithm,
    30	we select T = 0.7.
    31	3) Analysis of Variants of the Radiation Range γ: To
    32	analyze the effectiveness of the radiation range, this study
    33	conducted three radiation range experiments (settings in
    34	Table V) using three distinct methods. The methods progress
    35	from M1 , which uses a linear relationship for simple, direct
    36	analysis of stable datasets with minimal variation. M2 introduces an exponential function, improving responsiveness to
    37	
    38	directional changes and identifying rapid trend shifts, especially in dynamic data. M3 employs the square of cosine
    39	similarity, refining directional sensitivity and helping resolve
    40	details in occluded pedestrian scenes. This progression demonstrates the methods’ adaptability and enhanced performance
    41	across varying conditions.
    42	From the visualization in the Table V, it can be observed
    43	that M1 has relatively low accuracy and precision (R-1 70.4%,
    44	mAP 60.7%); M2 outperforms M1 in terms of accuracy and
    45	the localization of the visible regions of pedestrians(R-1 +
    46	0.2%); and M3 demonstrates the best performance (R-1 70.9%,
    47	mAP 61.1%) in complex occlusion scenarios, offering a more
    48	precise capture of the details of the radiation range.
    49	F. Visualization Analysis
    50	1) Visualization Analysis of Skeletal Region: To better
    51	demonstrate the superiority of the proposed method, this
    52	section shows the visualization effect of the mask images
    53	output by SAM in the local feature extraction stage. As
    54	shown in Fig. 14, Each set of three images represents: the
    55	visualization results of the keypoints outputted by the ViTPose
    56	pose estimation model [34], the keypoints heatmap outputted
    57	by the ViTPose, and the skeletal mask image outputted by our
    58	SAM module, respectively.
    59	The results in Fig. 14 demonstrate the effectiveness of the
    60	SAM module in accurately localizing pedestrian appearance
    61	regions. Unlike traditional keypoint heatmaps, our method
    62	enhances focus on visible regions while effectively suppressing
    63	responses to occlusions and background areas. This capability
    64	not only compensates for information loss between keypoints
    65	in heatmaps generated by pose estimation methods but also
    66	
    67	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
    68	
    69	8610
    70	
    71	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025
    72	
    73	Fig. 15. The performance of SAM under different occlusion scenarios.
    74	
    75	achieves fine-grained partitioning of the pedestrian’s trunk
    76	regions.
    77	In occlusion scenarios, a key challenge is the potential
    78	unreliability of keypoint coordinates predicted by pose estimation models. For example, in test sample (3), where the
    79	lower half of a pedestrian is obscured by a car, even humans
    80	may struggle to accurately determine the true positions of the
    81	knee joints and ankles. This limitation reduces the effectiveness of local features extracted by existing methods relying
    82	on keypoint heatmaps. However, the confidence scores of
    83	keypoints predicted by pose estimation models are based on
    84	high-level semantic features of the local region, which reflect
    85	the visibility of the pedestrian’s appearance in that area.
    86	These confidence scores provide valuable cues for perceiving
    87	occlusion interference. By leveraging this reliable information,
    88	the SAM module assigns visibility attributes to each local
    89	skeletal region through threshold filtering, thereby improving
    90	robustness in occlusion scenarios.
    91	Furthermore, the results from test samples (4) and (8)
    92	highlight the stability of our method. Despite variations in the
    93	quality of pose estimation results, our approach consistently
    94	generates stable mask outputs. This indicates that the SAM
    95	module has relatively low dependence on external models,
    96	making it more adaptable to diverse and challenging scenarios.
    97	2) Evaluation the Performance of SAM Under Different
    98	Occlusion Scenarios: To evaluate the performance of SAM
    99	under different occlusion scenarios, we present its performance
   100	under no occlusion, light occlusion, and heavy occlusion conditions (Fig. 15 (a)-(c)). The experimental results demonstrate
   101	that under no occlusion conditions (a), the SAM model can
   102	accurately locate and cover the overall appearance region of
   103	pedestrians. Under light occlusion conditions (b), although
   104	the positions of some key points are not precise and some
   105	key points may be located within the occluded regions, the
   106	SAM model can still effectively cover the visible regions
   107	of pedestrians through comprehensive confidence judgment,
   108	exhibiting relatively strong feature robustness.
   109	Furthermore, in heavily occluded scenarios (c), the SAM
   110	model still demonstrates a certain level of robustness. Despite
   111	significant occlusion (e.g., the first images in (c)), the SAM
   112	model can effectively locate the visible regions of pedestrians.
   113	This also indicates that SAM’s method of constructing a
   114	mask model by integrating node information and confidence
   115	is effective in handling common occlusion scenarios and can
   116	better retrieve the visible regions of pedestrians.
   117	
   118	Fig. 16. The Rank-5 performance of our method under different occlusion
   119	scenarios.
   120	
   121	However, when the occlusion area exceeds approximately
   122	80% (e.g., the last two images in (c)), the model’s performance
   123	is significantly affected (such situations are relatively rare,
   124	even for humans it is difficult to make judgments). The
   125	results show that while the SAM model can still locate the
   126	visible regions of pedestrians, the performance of pedestrian
   127	re-identification still has room for improvement due to the
   128	severe lack of information.
   129	To further observe the impact of occlusion, we visualize the
   130	pedestrian ranking results under heavy occlusion conditions
   131	in Fig. 16 (Rank-5, green boxes indicate correct results, and
   132	red boxes indicate incorrect results). The results indicate that
   133	in general occlusion scenarios, the SAM model can handle
   134	occlusion issues relatively well, and the Rank-1 results are
   135	usually able to correctly identify pedestrians (Fig. 16 up).
   136	However, in extremely severe occlusion scenarios (e.g., when
   137	the occlusion area exceeds 80%), the model’s performance
   138	drops significantly, relying only on the remaining 20% of
   139	information for pedestrian identification, leading to a substantial decline in performance (Fig. 16 down).
   140	In summary, the SAM model, through region modeling
   141	and comprehensive confidence judgment, can to some extent
   142	mitigate the issue that keypoint heatmaps cannot fully cover
   143	the appearance regions of pedestrians. It also exhibits relatively
   144	strong robustness under occlusion conditions. However, in
   145	extremely severe occlusion scenarios, the model’s performance
   146	still requires further improvement. Future research will focus
   147	on optimizing the model to address more complex occlusion
   148	scenarios, thereby enhancing the accuracy and robustness of
   149	pedestrian re-identification.
   150	3) Visualization Analysis of Attention Region: Fig. 17 illustrates the heat map generated by our method outlined in
   151	this paper. Through examination of the results, it becomes
   152	apparent that our method exhibits a remarkable ability to
   153	precisely determine the visible areas of pedestrians, even
   154	when confronted with substantial occlusion. This outcome
   155	
   156	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   157	
   158	GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION
   159	
   160	8611
   161	
   162	method exhibits superior complexity control in model structure
   163	and inference processes, enabling efficient performance in
   164	resource-constrained environments.
   165	V. C ONCLUSION
   166	
   167	Fig. 17. The visualization analysis of attention region by the proposed method.
   168	TABLE VI
   169	C OMPARISON OF D IFFERENT M ETHODS ON O CCLUDED -D UKE DATASETS
   170	
   171	serves as compelling evidence supporting the effectiveness of
   172	the proposed approach in effectively distinguishing between
   173	occluded backgrounds and pedestrian regions.
   174	G. Comparisons of Model Complexity
   175	To further demonstrate the practicality of the proposed
   176	method, we conduct an analysis of the model complexity and
   177	compare ours with seven state-of-the-art methods. The results
   178	are illustrated in Table VI (Ours-Res and Ours-ViT denote the
   179	backbones based on ResNet and ViT, respectively.).
   180	In the context of the ResNet backbone, the Ours-Res method
   181	significantly outperforms other approaches in several aspects.
   182	Firstly, it has a parameter count of only 83.10M and FLOPs of
   183	15.18G, both of which are lower than those of PGFA (115.5M
   184	and 25.9G) and HOReID (163.1M and 22.6G), demonstrating
   185	superior model efficiency. In terms of inference time, OursRes achieves 58.8 ms/img, which is considerably lower than
   186	PGFA’s 488.6 ms/img and HOReID’s 358.3 ms/img. Although
   187	Ours-Res utilizes external pose information similarly to PGFA,
   188	HOReID, and RFCnet, the feature extraction and fusion
   189	mechanisms proposed in this paper are relatively simplified,
   190	avoiding unnecessary complexity. Notably, the SAM module
   191	in our method employs arithmetic model mapping rather than
   192	the complex iterative inference used by other methods. This
   193	design choice enables the model to maintain high performance
   194	while reducing computational complexity. When utilizing a
   195	ViT backbone, although our FLOPs performance is not the
   196	best, we still demonstrate significant advantages in terms of
   197	parameter count and inference time. Our model excels in
   198	both Rank-1 and mAP metrics, particularly on the OccludedDuke dataset, where it achieves a Rank-1 of 73.4% and
   199	an mAP of 63.1%. This indicates that despite a slightly
   200	higher computational overhead, our approach can still deliver
   201	better performance in practical applications. Overall, our
   202	
   203	To address the challenge of coupling occlusion information and pedestrian features in the feature maps, this paper
   204	presents an occluded person re-identification algorithm based
   205	on pose-guided cross-attention feature fusion. The proposed
   206	method consists of three modules: the VARA module, the
   207	SAM module, and the SMCF module. Firstly, the VARA
   208	is introduced to guide the backbone network in learning
   209	the differences between occlusion information and pedestrian
   210	features at intermediate layers, suppressing occlusion interference, and mitigating the problem of occlusion information
   211	diffusion caused by enlarged receptive fields. Secondly, leveraging the advantages of keypoint confidence, SAM generates
   212	fine-grained masks for local skeletal regions to extract comprehensive local features. Compared to methods that extract
   213	local features using keypoint heatmaps, our approach considers
   214	the locally ignored regions between keypoints, resulting in
   215	more complete extraction of appearance information. Finally,
   216	under the constraint of spatial attention masks, the feature
   217	maps obtained from the previous two steps and the local
   218	features are fused through cross-attention, generating enhanced
   219	local features with aligned high-level semantic information.
   220	Experimental results on datasets for occluded and complete
   221	person re-identification demonstrate significant performance
   222	improvement of the proposed algorithm compared to stateof-the-art methods.
   223	R EFERENCES
   224	[1]
   225	
   226	E. Ning, C. Wang, H. Zhang, X. Ning, and P. Tiwari, “Occluded person
   227	re-identification with deep learning: A survey and perspectives,” Expert
   228	Syst. Appl., vol. 239, Apr. 2024, Art. no. 122419.
   229	[2] H. Hu et al., “PersonMAE: Person re-identification pre-training
   230	with masked AutoEncoders,” IEEE Trans. Multimedia, vol. 26,
   231	pp. 10029–10040, 2024.
   232	[3] H. Ma, C. Zhang, Y. Zhang, Z. Li, Z. Wang, and C. Wei, “A review on
   233	video person re-identification based on deep learning,” Neurocomputing,
   234	vol. 609, Dec. 2024, Art. no. 128479.
   235	[4] S. Yao, K. Pan, T. Wang, Z. Zheng, J. Jin, and C. Hu,
   236	“AAGNet: Attribute-aware graph-based network for occluded pedestrian re-identification,” IEEE Trans. Consum. Electron., vol. 70, no. 4,
   237	pp. 6580–6588, Nov. 2024.
   238	[5] G. Yan, Z. Wang, S. Geng, Y. Yu, and Y. Guo, “Part-based representation enhancement for occluded person re-identification,” IEEE Trans.
   239	Circuits Syst. Video Technol., vol. 33, no. 8, pp. 4217–4231, Aug. 2023.
   240	[6] Y. Ye et al., “Dynamic feature pruning and consolidation for occluded
   241	person re-identification,” in Proc. AAAI Conf. Artif. Intell., Mar. 2024,
   242	vol. 38, no. 7, pp. 6684–6692.
   243	[7] Z. Wang, H. Huang, A. Zheng, C. Li, and R. He, “Parallel augmentation
   244	and dual enhancement for occluded person re-identification,” in Proc.
   245	IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), Apr. 2024,
   246	pp. 3590–3594.
   247	[8] J. Miao, Y. Wu, P. Liu, Y. Ding, and Y. Yang, “Pose-guided feature
   248	alignment for occluded person re-identification,” in Proc. IEEE/CVF Int.
   249	Conf. Comput. Vis. (ICCV), Oct. 2019, pp. 542–551.
   250	[9] J. Cui, Y. Chen, B. Deng, G. Liu, Z. Wang, and Y. Li, “PPBI: Poseguided partial-attention network with batch information for occluded
   251	person re-identification,” Sensors, vol. 25, no. 3, p. 757, Jan. 2025.
   252	[10] S. Gao, J. Wang, H. Lu, and Z. Liu, “Pose-guided visible part matching
   253	for occluded person ReID,” in Proc. IEEE/CVF Conf. Comput. Vis.
   254	Pattern Recognit., Jun. 2020, pp. 11744–11752.
   255	
   256	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   257	
   258	8612
   259	
   260	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025
   261	
   262	[11] T. Wang, H. Liu, P. Song, T. Guo, and W. Shi, “Pose-guided feature disentangling for occluded person re-identification based on transformer,”
   263	in Proc. AAAI Conf. Artif. Intell., 2022, pp. 2540–2549.
   264	[12] A. Khatun, S. Denman, S. Sridharan, and C. Fookes, “Pose-driven
   265	attention-guided image generation for person re-identification,” Pattern
   266	Recognit., vol. 137, May 2023, Art. no. 109246.
   267	[13] H. Zheng et al., “Cascade transformer reasoning embedded by uncertainty for occluded person re-identification,” IEEE Trans. Biometrics,
   268	Behav., Identity Sci., vol. 6, no. 2, pp. 219–229, Apr. 2024.
   269	[14] G. Zhang, Y. Yang, Y. Zheng, G. Martin, and R. Wang, “Maskaware hierarchical aggregation transformer for occluded person reidentification,” IEEE Trans. Circuits Syst. Video Technol., early access,
   270	Jan. 17, 2025, doi: 10.1109/TCSVT.2025.3531142.
   271	[15] X. Zhang, Y. Yan, J.-H. Xue, Y. Hua, and H. Wang, “Semantic-aware
   272	occlusion-robust network for occluded person re-identification,” IEEE
   273	Trans. Circuits Syst. Video Technol., vol. 31, no. 7, pp. 2764–2778, Jul.
   274	2021.
   275	[16] Q. Yang, P. Wang, Z. Fang, and Q. Lu, “Focus on the visible regions:
   276	Semantic-guided alignment model for occluded person re-identification,”
   277	Sensors, vol. 20, no. 16, p. 4431, Aug. 2020.
   278	[17] Z. Li, H. Zhang, L. Zhu, J. Sun, and L. Liu, “Heterogeneous generative
   279	tokens and distance-aware recovery network for occluded person reidentification,” IEEE Trans. Circuits Syst. Video Technol., early access,
   280	Dec. 17, 2025, doi: 10.1109/TCSVT.2024.3519312.
   281	[18] Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang, “Beyond part
   282	models: Person retrieval with refined part pooling (and a strong
   283	convolutional baseline),” in Proc. Eur. Conf. Comput. Vis. (ECCV),
   284	2018, pp. 480–496.
   285	[19] Y. Xiahou, N. Li, and X. Li, “Identity consistency multi-viewpoint generative aggregation for person re-identification,” IEEE Trans. Circuits
   286	Syst. Video Technol., vol. 34, no. 3, pp. 1441–1455, Mar. 2024.
   287	[20] S. He, H. Luo, P. Wang, F. Wang, H. Li, and W. Jiang, “TransReID:
   288	Transformer-based object re-identification,” in Proc. IEEE/CVF Int.
   289	Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 15013–15022.
   290	[21] Y. Zhao, S. Zhu, D. Wang, and Z. Liang, “Short range correlation transformer for occluded person re-identification,” Neural Comput. Appl.,
   291	vol. 34, no. 20, pp. 17633–17645, Oct. 2022.
   292	[22] J. Li, S. Zhang, Q. Tian, M. Wang, and W. Gao, “Pose-guided representation learning for person re-identification,” IEEE Trans. Pattern Anal.
   293	Mach. Intell., vol. 44, no. 2, pp. 622–635, Feb. 2022.
   294	[23] G. Wang et al., “High-order information matters: Learning relation and topology for occluded person re-identification,” in Proc.
   295	IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020,
   296	pp. 6448–6457.
   297	[24] R. Hou, B. Ma, H. Chang, X. Gu, S. Shan, and X. Chen,
   298	“Feature completion for occluded person re-identification,” IEEE
   299	Trans. Pattern Anal. Mach. Intell., vol. 44, no. 9, pp. 4894–4912,
   300	Sep. 2022.
   301	[25] S. Dou, C. Zhao, X. Jiang, S. Zhang, W.-S. Zheng, and W. Zuo, “Human
   302	co-parsing guided alignment for occluded person re-identification,”
   303	IEEE Trans. Image Process., vol. 32, pp. 458–470, 2023.
   304	[26] C. Hu, Y. Chen, L. Guo, L. Tao, Z. Tie, and W. Ke, “Pose-guided
   305	node and trajectory construction transformer for occluded person reidentification,” J. Electron. Imag., vol. 33, no. 4, Jul. 2024, Art. no.
   306	043021.
   307	[27] V. Somers, A. Alahi, and C. D. Vleeschouwer, “Keypoint promptable
   308	re-identification,” in Proc. Eur. Conf. Comput. Vis. Cham, Switzerland:
   309	Springer, Nov. 2024, pp. 216–233.
   310	[28] Y. Lu, M. Jiang, Z. Liu, and X. Mu, “Dual-branch adaptive attention
   311	transformer for occluded person re-identification,” Image Vis. Comput.,
   312	vol. 131, Mar. 2023, Art. no. 104633.
   313	[29] K. Zhu et al., “AAformer: Auto-aligned transformer for person reidentification,” 2021, arXiv:2104.00921.
   314	[30] G. Zhang, P. Zhang, J. Qi, and H. Lu, “HAT: Hierarchical aggregation
   315	transformers for person re-identification,” in Proc. 29th ACM Int. Conf.
   316	Multimedia, Oct. 2021, pp. 516–525.
   317	[31] Y. Li, J. He, T. Zhang, X. Liu, Y. Zhang, and F. Wu, “Diverse part discovery: Occluded person re-identification with part-aware transformer,”
   318	in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
   319	2021, pp. 2898–2907.
   320	[32] H. Tan, X. Liu, B. Yin, and X. Li, “MHSA-Net: Multihead self-attention
   321	network for occluded person re-identification,” IEEE Trans. Neural
   322	Netw. Learn. Syst., vol. 34, no. 11, pp. 8210–8224, Nov. 2023, doi:
   323	10.1109/TNNLS.2022.3144163.
   324	[33] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in
   325	Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Jun. 2018,
   326	pp. 7132–7141.
   327	
   328	[34] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, “ViTPose: Simple vision
   329	transformer baselines for human pose estimation,” in Proc. Adv. Neural
   330	Inf. Process. Syst., vol. 35. Red Hook, NY, USA: Curran Associates,
   331	2022, pp. 38571–38584.
   332	[35] J. Li, C. Wang, H. Zhu, Y. Mao, H.-S. Fang, and C. Lu, “CrowdPose:
   333	Efficient crowded scenes pose estimation and a new benchmark,” in
   334	Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
   335	2019, pp. 10855–10864.
   336	[36] J. Yang, C. Zhang, Y. Tang, and Z. Li, “PAFM: Pose-drive attention
   337	fusion mechanism for occluded person re-identification,” Neural Comput. Appl., vol. 34, no. 10, pp. 8241–8252, May 2022.
   338	[37] X. Ren, D. Zhang, and X. Bao, “Semantic-guided shared feature
   339	alignment for occluded person re-IDentification,” in Proc. Asian Conf.
   340	Mach. Learn., Sep. 2020, pp. 17–32.
   341	[38] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie,
   342	“Feature pyramid networks for object detection,” in Proc. IEEE Conf.
   343	Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017, pp. 936–944.
   344	[39] P. Gao, M. Zheng, X. Wang, J. Dai, and H. Li, “Fast convergence of
   345	DETR with spatially modulated co-attention,” in Proc. IEEE/CVF Int.
   346	Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 3621–3630.
   347	[40] T.-T. Yuan, Q.-L. Shu, S.-B. Chen, L.-L. Huang, and B. Luo, “Instant
   348	pose extraction based on mask transformer for occluded person reidentification,” Pattern Recognit., vol. 159, Mar. 2025, Art. no. 111082.
   349	[41] S. Kim, S. Kang, H. Choi, S. S. Kim, and K. Seo, “Keypoint aware
   350	robust representation for transformer-based re-identification of occluded
   351	person,” IEEE Signal Process. Lett., vol. 30, pp. 65–69, 2023.
   352	[42] Z. Ma, Y. Zhao, and J. Li, “Pose-guided inter-and intra-part relational
   353	transformer for occluded person re-identification,” in Proc. IEEE/CVF
   354	Conf. Comput. Vis. pattern Recognit., Jun. 2021, pp. 1487–1496.
   355	[43] S. Wang, B. Huang, H. Li, G. Qi, D. Tao, and Z. Yu, “Key point-aware
   356	occlusion suppression and semantic alignment for occluded person reidentification,” Inf. Sci., vol. 606, pp. 669–687, Aug. 2022.
   357	[44] H. Wang, X. Chen, and C. Liu, “Pose-guided part matching network via
   358	shrinking and reweighting for occluded person re-identification,” Image
   359	Vis. Comput., vol. 111, Jul. 2021, Art. no. 104186.
   360	[45] H. Gao, C. Hu, G. Han, J. Mao, W. Huang, and Q. Guan, “Pointlevel feature learning based on vision transformer for occluded person

codex
OIDM 的关键闭环找到了：他们专门把“最优光照选择”拿去和随机、固定光照替换对比，还统计有效替换比例，这就是在证明视角本身。OPMCN 还缺几段具体消融，我补一下。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 10 -l 11 'Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '1,420p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	2104
     2	
     3	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
     4	
     5	Fig. 5. T-SNE [63] visualization of 20 random identities on Market-1501 between the baseline (Left) and the proposed OPMCN (Right). Different numbers
     6	and colors represent different identities.
     7	
     8	Fig. 7. Grad-CAM [64] visualization of feature maps extracted by different
     9	models in Table V.
    10	
    11	Fig. 6. Qualitative visualization of top-10 retrieval results under different
    12	configurations on the Market-1501. Each row shows the results for the same
    13	query image (leftmost), with correct matches outlined in green and incorrect
    14	ones in red.
    15	
    16	PKGC enhances the quality of identity-relevant features at the
    17	backbone level.
    18	Additionally, to further validate the effectiveness of the
    19	PKGC block, we also compare it with some classic attention modules including squeeze-and-excitation attention (SEA)
    20	module [47], convolutional block attention module (CBAM)
    21	[48], coordinate attention (CA) module [49], and global context (GC) block [50]. According to the comparison results
    22	listed in Table V, our PKGC block outperforms the other four
    23	attention modules Market-1501 dataset, which proves that the
    24	PKGC block can better extract the important discrimination
    25	
    26	information for Re-ID. Fig. 7 illustrates the feature maps of
    27	some instances from Market-1501, which are produced by
    28	different models in Table V. It is obvious that our PKGC
    29	block can better locate the body regions of interest than the
    30	other attention modules, which also indicates that the proposed
    31	PKGC can capture ID-related information more effectively.
    32	In order to gain more insight into the PKGC block, we
    33	provide one more qualitative visualization of the learned feature distributions using a modified t-SNE plot with embedded
    34	images, as shown in Fig. 8. Since PKGC operates at the
    35	backbone level to refine feature representations, conventional
    36	t-SNE visualizations that only plot sample positions are insufficient to reflect its full effect. In this visualization, each
    37	image is displayed at its embedding location; border colors
    38	indicate identity clusters, and camera IDs are annotated in the
    39	top-right corner of each sample. From Fig. 8, it is evident
    40	
    41	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
    42	
    43	CAO et al.: OPTIMAL PROXY MINING CONTRASTIVE NETWORK FOR UNSUPERVISED PERSON RE-IDENTIFICATION
    44	
    45	2105
    46	
    47	Fig. 10. The performance evaluation with different values of K2 .
    48	Fig. 8. T-SNE [63] visualization of the proposed OPMCN and the framework
    49	without the proposed PKGC block.
    50	
    51	Fig. 11. The performance evaluation with different values of K3 .
    52	Fig. 9. The performance evaluation with different values of K1 .
    53	
    54	that PKGC contributes to better feature alignment across
    55	camera views. Samples of the same identity—especially those
    56	captured under different cameras—are grouped more tightly,
    57	while inter-identity clusters become more clearly separated. In
    58	particular, we highlight two hard identity clusters (blue-circled
    59	region) where background similarity previously caused visual
    60	confusion. Without PKGC, these identities are mixed in the
    61	embedding space, while the visual gap between them increases
    62	noticeably after adding the proposed PKGC block, confirming
    63	that PKGC strengthens identity-invariant representation and
    64	alleviates cross-camera ambiguity.
    65	C. Parameter Analysis
    66	In this subsection, we analyze the sensitivity of the hyperparameters involved in the whole work. Here, we conduct
    67	experiments on Market-1501 and MSMT17 to investigate the
    68	sensitivity of the remaining hyper-parameters, which include
    69	the number of negative samples (K1 in HCPM module), the
    70	number of positive samples (K2 in FNPM module), the number
    71	of negative samples (K3 in FNPM module), and the balancing
    72	weight λ in computing the entire loss.
    73	1) The Sensitivity of K1 : K1 is the number of negative
    74	samples mined in the HCPM module. Fig. 9 presents the
    75	performance of HCPM when K1 varies from 10 to 100. We
    76	observe that the performance goes up when the number of
    77	hard negatives increases from 10 to 20. But the performance
    78	gradually drops when more negative proxies are taken into
    79	consideration. It indicates that easy negatives may hamper
    80	contrastive learning. Focusing on a small number of the most
    81	informative negative proxies from the corresponding camera
    82	ID helps the model to better discriminate confusing instances.
    83	
    84	Fig. 12. The performance evaluation with different values of λ.
    85	
    86	2) The Sensitivity of K2 : K2 is the number of positive samples when computing LFNPM . Fig. 10 presents the performance
    87	of “HCPM”+“FNPM” when K2 varies from 1 to 6. We see
    88	that the performance increases first and then degenerates. If
    89	K2 is too small, not enough false negative proxies will be
    90	recalled. Conversely, if the selection of K2 is excessively large,
    91	the quality of positive samples selection cannot be assured,
    92	potentially leading to an influx of true negative proxies, which
    93	would subsequently diminish the effectiveness of the model.
    94	3) The Sensitivity of K3 : K3 is the number of negative
    95	samples in FNPM module. Fig. 11 indicates the performance
    96	when K3 varies from 10 to 100. Similar to the trend of K1 ,
    97	we found that setting K3 to 30 yielded the best results across
    98	the majority of datasets.
    99	4) Parameter Analysis on Balancing Weight λ: The combination of the HCPM module and the FNPM module is
   100	implemented via the sum of two contrastive losses. To validate
   101	the effectiveness of this combination way, the performance
   102	varying along with the change of the balancing weight λ
   103	between LHCPM and LFNPM is presented in Fig. 12. When
   104	
   105	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   106	
   107	
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 10 'Privacy preserving person re-identification via anonymizing diffusion model.pdf' - | nl -ba | sed -n '1,320p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Pattern Recognition 174 (2026) 112939
     2	
     3	R. Wang et al.
     4	
     5	Table 3
     6	Results on diﬀerent datasets. High 𝑇𝑔 , 𝑇𝑢 and low FID indicate great utility capability. Low 𝑇𝑝 and high ISD indicate great privacy-preserving
     7	capability. 𝐼𝑚𝑝𝑣 means the percentage improved.
     8	𝑇𝑔
     9	
    10	Datasets
    11	
    12	Market-1501
    13	MSMT17
    14	CUHK-SYSU
    15	
    16	𝑇𝑢
    17	
    18	𝑇𝑝
    19	
    20	R-5
    21	
    22	R-10
    23	
    24	mAP
    25	
    26	R-1
    27	
    28	R-5
    29	
    30	R-10
    31	
    32	mAP
    33	
    34	R-1
    35	
    36	R-5
    37	
    38	R-10
    39	
    40	mAP
    41	
    42	Orgn
    43	
    44	ADM
    45	
    46	Impv
    47	
    48	67.4
    49	18.0
    50	72.0
    51	
    52	83.8
    53	30.7
    54	83.6
    55	
    56	88.4
    57	37.2
    58	86.5
    59	
    60	45.3
    61	8.0
    62	68.7
    63	
    64	94.8
    65	49.7
    66	94.8
    67	
    68	97.9
    69	74.7
    70	98.1
    71	
    72	98.7
    73	82.0
    74	98.7
    75	
    76	72.5
    77	23.7
    78	96.2
    79	
    80	57.1
    81	5.2
    82	21.7
    83	
    84	68.4
    85	16.2
    86	31.9
    87	
    88	73.6
    89	24.9
    90	37.1
    91	
    92	19.4
    93	2.2
    94	6.6
    95	
    96	0.15
    97	0.18
    98	0.13
    99	
   100	0.50
   101	0.58
   102	0.63
   103	
   104	2.33
   105	2.22
   106	3.85
   107	
   108	Metrics
   109	
   110	Training
   111	
   112	Query
   113	
   114	Gallery
   115	
   116	𝑇𝑔
   117	𝑇𝑢
   118	𝑇𝑝
   119	
   120	G
   121	G
   122	R
   123	
   124	R
   125	G
   126	R
   127	
   128	R
   129	G
   130	G
   131	
   132	Methods
   133	
   134	R-1
   135	
   136	mAP
   137	
   138	Resnet50Mid [49]
   139	ADM
   140	
   141	75.6
   142	67.4
   143	
   144	89.9
   145	45.3
   146	
   147	4.4.5. Demo of attack defense
   148	To visually present the privacy-preserving capability of ADM, we
   149	visualize the retrieval result during attacks. We use real query images to
   150	retrieve generated images protected by ADM. As shown in Fig. 5, images
   151	retrieved by the ReID model are not relevant to query IDs. Considering
   152	the anonymized dataset generated using ADM, there is no ID that looks
   153	the same as the query image, because the anonymized images generated
   154	using the query image in Stage 2 has diﬀerent visual features with the
   155	query image, thus the ReID model failed to retrieve images related to
   156	the query ID, and the privacy attack is eﬀectively resisted.
   157	
   158	Table 6
   159	Ablation study on diﬀerent components.
   160	𝑇𝑔
   161	
   162	Methods
   163	
   164	(c)
   165	(d)
   166	
   167	only Stagel
   168	only Stage2
   169	w/o class embedding
   170	random
   171	centroid
   172	ADM (full model)
   173	
   174	4.4.6. Cosine distance distribution
   175	To gain deeper insights into the privacy protection diﬀerences among
   176	datasets, we analyze the distribution of (1) the distance between training images and the feature centroid of their belonging IDs, and (2) the
   177	distance between generated images and the nearest real ID feature centroids. As shown in Fig. 6, the distances between generated images and
   178	real ID centroids are signiﬁcantly larger. This means that features of
   179	generated new IDs are distinctly diﬀerent from real IDs, thereby achieving better privacy protection. Notably, the mean of ADM distribution is
   180	ISD. A more right-shifted ADM distribution indicates greater ISD.
   181	
   182	ISD
   183	
   184	mAP
   185	
   186	R-1
   187	
   188	16.97
   189	31.08
   190	38.76
   191	37.26
   192	45.14
   193	45.29
   194	
   195	38.42
   196	53.18
   197	61.37
   198	61.97
   199	67.99
   200	67.37
   201	
   202	31.2
   203	35.7
   204	29.8
   205	
   206	We conduct ADM on CUHK-SYSU, Market1501 and MSMT17 dataset
   207	to obtain CUHK-SYSU-A, Market1501-A, MSMT17-A, three privacycompliant counterparts of public benchmarks. The three datasets contain a total of 5000 IDs and 150,000 images. Enough data enables researchers to conduct ReID model training without concerning privacy
   208	issues.
   209	
   210	Table 5
   211	ReID performance comparison between the baseline ResNet50Mid
   212	and ADM on Market-1501.
   213	
   214	(b)
   215	
   216	FID
   217	
   218	R-1
   219	
   220	Table 4
   221	The measurement settings of diﬀerent
   222	metrics. G and R denote generated and
   223	real datasets respectively.
   224	
   225	(a)
   226	
   227	ISD
   228	
   229	0.57
   230	0.56
   231	0.52
   232	0.50
   233	0.49
   234	0.50
   235	
   236	Table 7
   237	Eﬀects of 𝑛 in Algorithm 1 𝑚 = 3.
   238	
   239	4.5. Discussion
   240	
   241	n
   242	
   243	𝑇𝑢 (𝑅 − 1)
   244	
   245	n
   246	
   247	𝑇𝑢 (𝑅 − 1)
   248	
   249	n
   250	
   251	𝑇𝑢 (𝑅 − 1)
   252	
   253	3
   254	
   255	93.4
   256	
   257	5
   258	
   259	96.12
   260	
   261	8
   262	
   263	92.9
   264	
   265	The comprehensive experimental evaluation presented above reveals
   266	important insights about the privacy-utility trade-oﬀ inherent in our
   267	approach. This section discusses these ﬁndings in depth, comparing
   268	ADM with existing methods and identifying areas for future improvement. Compared with the majority of existing privacy-preserving methods for ReID, ADM achieves competitive ReID performance. However,
   269	ADM does not reach the current state-of-the-art level and exhibits notable performance gaps compared to encryption-based methods. This
   270	performance trade-oﬀ is primarily a consequence of achieving complete erasure of privacy information, which necessitates sacriﬁcing some
   271	ReID performance. Nevertheless, it is worth noting that ADM can progressively improve ReID model performance by generating increasingly
   272	large volumes of data, which to some extent mitigates this limitation.
   273	It is observed that the consistency of certain generated identiﬁers is
   274	inadequate and necessitates further optimization. This may be attributed
   275	to the generally low image quality (particularly low resolution) of the
   276	real datasets used for training and poor sample consistency, especially
   277	in Market-1501. Unfortunately, due to time and resource constraints, we
   278	are unable to conduct further investigation into this issue. Future work
   279	could potentially address this problem by further optimizing hyperparameters or adopting more powerful Stable Diﬀusion models.
   280	For example, the hyperparameter 𝐴𝑡𝑡 in Stage 2 regulates the proportion of attributes allocated within each subset, with an empirical
   281	conﬁguration of 𝐴𝑡𝑡 = 0.4. This conﬁguration achieves an equilibrium
   282	
   283	further ensure intra-identity consistency. In the main experiment, we
   284	chose the ADM griddle as it gives better results.
   285	4.4.3. Eﬀects of 𝑛 in Algorithm 1
   286	The ablation study of Algorithm 1 with diﬀerent 𝑛 are shown in
   287	Table 7. An excessively large value of 𝑛 can lead to a degradation in the
   288	diversity of generated identities, increasing their similarity to authentic
   289	ones. Conversely, a minimal 𝑛 fails to optimize the use of existing genuine data, potentially leading to an ineﬃcient exploitation of valuable
   290	information resources.
   291	4.4.4. Demo of ADM dataset
   292	To visually present the anonymization capability of ADM, we present
   293	some demos of the generated IDs in Fig. 4. The generated IDs share similar styles with the original dataset. Furthermore, they have a very low
   294	visual association with the related original IDs, indicating that attackers would not be able to retrieve private information of the original IDs
   295	from the generated ones. While erased privacy information, the generated samples maintain great details, thus achieving satisfying data authenticity.
   296	9
   297	
   298	Pattern Recognition 174 (2026) 112939
   299	
   300	R. Wang et al.
   301	
   302	between intra-ID consistency and inter-ID diversity by constraining attribute overlap, thereby facilitating precise semantic control over visual
   303	feature generation. Excessive 𝐴𝑡𝑡 values engender attribute ambiguity
   304	and compromise the ﬁdelity of text-to-image synthesis; conversely, excessively low values diminish data utilization eﬃciency. Consequently,
   305	ﬁne-grained optimization of the 𝐴𝑡𝑡 parameter is instrumental in reconciling this inherent trade-oﬀ between competing objectives.
   306	
   307	[7] H. Wang, G. Wang, Y. Li, D. Zhang, L. Lin, Transferable, controllable, and inconspicuous adversarial attacks on person re-identiﬁcation with deep mis-ranking, in:
   308	Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 342–351.
   309	[8] S. Lu, et al., Discriminator-free generative adversarial attack, in: Proceedings of the
   310	29th ACM International Conference on Multimedia, 2021, pp. 1544–1552.
   311	[9] L. Wang, et al., Attack is the best defense: towards preemptive-protection person
   312	re-identiﬁcation, in: Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 550–559.
   313	[10] R. Rombach, et al., High-resolution image synthesis with latent diﬀusion models, in:
   314	Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10684–10695.
   315	[11] L. Zhu, T. Chen, D. Ji, J. Ye, J. Liu, Not every patch is needed: toward a
   316	more eﬃcient and eﬀective backbone for video-based person re-identiﬁcation,
   317	IEEE Trans. Image Process. 34 (2025) 785–800. https://api.semanticscholar.org/
   318	CorpusID:275931800.
   319	[12] M. Ye, et al., Collaborative reﬁning for person re-identiﬁcation with label noise,
   320	IEEE Trans. Image Process. 31 (2021) 379–391.

codex
OPMCN 的关键可视化也找到了：t-SNE、Top-10 检索和 Grad-CAM 都用来证明代理选择和 PKGC 的语义焦点更对。还差模块消融第一段的表格解释，补齐后就可以整理逐篇方法论拆解。
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
/bin/zsh -lc "pdftotext -f 9 -l 10 'Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '1,380p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	CAO et al.: OPTIMAL PROXY MINING CONTRASTIVE NETWORK FOR UNSUPERVISED PERSON RE-IDENTIFICATION
     2	
     3	2103
     4	
     5	TABLE IV
     6	T HE T RAINING T IME C OMPARISON AND C ORRESPONDING P ERFORMANCE
     7	FOR D IFFERENT C ONTRIBUTIONS ON M ARKET-1501
     8	
     9	TABLE V
    10	C OMPUTATIONAL C OMPLEXITY AND RUNTIME C OMPARISON AND
    11	C ORRESPONDING P ERFORMANCE FOR D IFFERENT BACKBONE
    12	M ETHODS ON M ARKET-1501
    13	Fig. 4. Illustration of the architecture of the proposed pyramid kernel global
    14	context block. The feature maps are shown as feature dimensions, e.g., C×H×
    15	W denotes the number of channels, height, and width, ⊕ denotes Hadamard
    16	addition, and ⊗ denotes matrix multiplication. The attention maps (heatmaps)
    17	of input examples are visualized to demonstrate the effectiveness of “Softmax”
    18	operation.
    19	
    20	0.4% in Rank-1 on Market-1501, 2.6% in mAP and 1.8% in
    21	Rank-1 on MSMT17, and 1.0% in mAP and 0.7% in Rank1 on VeRi-776. It is worth noting that FNPM is designed to
    22	operate on top of the structured proxy relations provided by
    23	HCPM. Applying FNPM without prior hard positive/negative
    24	mining leads to training instability and performance degradation, hence its independent evaluation is not considered.
    25	In the proposed FNPM module, the proxies Pcor sharing the
    26	same camera ID as the query are excluded before generating
    27	the query-based descending ranking list. To further verify the
    28	effectiveness of excluding the Pcor and anti-noise ranking
    29	(ANR) scheme, we provide the experiment groups whether
    30	exclude the Pcor and ANR in Table III. Experimental results
    31	from two datasets demonstrate that ANR markedly improved
    32	training performance, with further enhancements observed
    33	following the excluding of Pcor .
    34	3) Effectiveness of Pyramid Kernel Global Context Block:
    35	The proposed PKGC block aims to extract the most discriminative parts from the spatial parts. It can be observed from
    36	the comparison results of variant (a) vs. variant (d), variant (c)
    37	vs. variant (e), and variant (c) vs. variant (f) in Table I, which
    38	demonstrate that the proposed PKGC block plays a positive
    39	role on all datasets. Regardless of the unsupervised training mechanism employed, the incorporation of the proposed
    40	PKGC block consistently leads to significant enhancements in
    41	performance.
    42	4) Computational Cost and Efficiency: To provide a comprehensive analysis of the computational overhead introduced
    43	by each proposed module, we report performance and efficiency from two perspectives: overall training time (Table IV)
    44	and model complexity during inference (Table V). As shown
    45	in Table IV, we compare the total training time and corresponding performance across different combinations of the
    46	proposed contributions. It can be observed that while adding
    47	HCPM, FNPM, and PKGC successively leads to incremental improvements in mAP and Rank-1 accuracy, the total
    48	
    49	training time increases only marginally (from 1h03m to 1h20m
    50	per epoch, demonstrating the practical feasibility of each
    51	module. In Table V, we further analyze the computational
    52	complexity for models with different attention-based backbone
    53	variants. Since HCPM and FNPM do not alter the backbone structure, they do not impact computational complexity
    54	or inference time, and thus are not included in this table.
    55	Notably, in this table, “OPMC” denotes the combination of
    56	“Baseline+HCPM+FNPM” in Table IV. Among all configurations, OPMC+PKGC achieves the best performance while
    57	introducing only a small increase in FLOPs (+0.036 GB) and
    58	inference latency (+6.5 ms), compared to other widely used
    59	attention modules like CA, CBAM, and GC. This confirms that
    60	our multi-scale PKGC design offers the best trade-off between
    61	accuracy and efficiency among all considered variants.
    62	5) Qualitative Analysis: To evaluate the effectiveness of the
    63	proposed components, the t-SNE technique [63] is utilized to
    64	visualize the feature representations learned by both the baseline and the proposed OPMCN for 20 IDs. Fig. 5 illustrates
    65	that the proposed methods attain a reduced intra-ID variance
    66	and an enhanced inter-ID variance, notably among visually
    67	similar instances such as those from IDs #12, #14, #17, #6,
    68	and #15.
    69	To further demonstrate the contribution of the proposed
    70	modules, we present a qualitative comparison of Rank-10
    71	retrieval results on the Market-1501 dataset under three configurations: full OPMCN model; OPMCN without PKGC;
    72	OPMCN without both PKGC and FNPM as shown in Fig. 6.
    73	In this visualization, correctly retrieved gallery instances (i.e.,
    74	same identity as the query) are outlined in green, while
    75	incorrect retrievals are outlined in red. With FNPM enabled,
    76	the retrieval results contain fewer visually misleading false
    77	positives (e.g., different persons with similar clothing), indicating that FNPM improves the robustness of proxy-based
    78	contrastive learning. In addition, incorporating PKGC leads
    79	to further improvements, especially in handling viewpoint
    80	changes and background interference. The retrieved samples
    81	show clearer semantic alignment in fine-grained attributes such
    82	as bag type, clothing texture, and body posture, suggesting that
    83	
    84	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
    85	
    86	2104
    87	
    88	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
    89	
    90	Fig. 5. T-SNE [63] visualization of 20 random identities on Market-1501 between the baseline (Left) and the proposed OPMCN (Right). Different numbers
    91	and colors represent different identities.
    92	
    93	Fig. 7. Grad-CAM [64] visualization of feature maps extracted by different
    94	models in Table V.
    95	
    96	Fig. 6. Qualitative visualization of top-10 retrieval results under different
    97	configurations on the Market-1501. Each row shows the results for the same
    98	query image (leftmost), with correct matches outlined in green and incorrect
    99	ones in red.
   100	
   101	PKGC enhances the quality of identity-relevant features at the
   102	backbone level.
   103	Additionally, to further validate the effectiveness of the
   104	PKGC block, we also compare it with some classic attention modules including squeeze-and-excitation attention (SEA)
   105	module [47], convolutional block attention module (CBAM)
   106	[48], coordinate attention (CA) module [49], and global context (GC) block [50]. According to the comparison results
   107	listed in Table V, our PKGC block outperforms the other four
   108	attention modules Market-1501 dataset, which proves that the
   109	PKGC block can better extract the important discrimination
   110	
   111	information for Re-ID. Fig. 7 illustrates the feature maps of
   112	some instances from Market-1501, which are produced by
   113	different models in Table V. It is obvious that our PKGC
   114	block can better locate the body regions of interest than the
   115	other attention modules, which also indicates that the proposed
   116	PKGC can capture ID-related information more effectively.
   117	In order to gain more insight into the PKGC block, we
   118	provide one more qualitative visualization of the learned feature distributions using a modified t-SNE plot with embedded
   119	images, as shown in Fig. 8. Since PKGC operates at the
   120	backbone level to refine feature representations, conventional
   121	t-SNE visualizations that only plot sample positions are insufficient to reflect its full effect. In this visualization, each
   122	image is displayed at its embedding location; border colors
   123	indicate identity clusters, and camera IDs are annotated in the
   124	top-right corner of each sample. From Fig. 8, it is evident
   125	
   126	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   127	
   128	
exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf' - | nl -ba | sed -n '1,240p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Optimal Illumination Distance Metrics for Person
     2	Re-Identification in Complex Lighting Conditions
     3	CHAO WANG, ZHONGYUAN WANG, RUIMIN HU, and XIAOCHEN WANG, National
     4	Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University,
     5	Wuhan, China
     6	WEN ZHOU, Nanjing University of Finance and Economics, Nanjing, China
     7	Person re-identification is extensively applied in public security and surveillance. However, environmental factors like time and location often lead to varying lighting conditions in captured pedestrian images, significantly
     8	impacting identification accuracy. Current approaches mitigate this issue through lighting transformation
     9	techniques, aiming to normalize images to a standard lighting condition for consistent person re-identification
    10	results. Yet, these methods overlook the fact that different content may hold distinct identification values
    11	under diverse lighting conditions. To address this, we conducted an analysis on the identification distance
    12	between images of the same or different pedestrians under pre-defined lighting conditions. From this analysis,
    13	we introduce the concept of optimal lighting: a condition where the distance between image pairs is minimized
    14	compared to other lighting scenarios. We propose utilizing this optimal lighting distance in the image retrieval
    15	process for final ranking. Our study, validated on synthetic datasets Market-IA and Duke-IA, demonstrates
    16	that optimal lighting is independent of image texture information. Each image pair exhibits a unique optimal
    17	lighting, yet consistently shows a minimum distance value.
    18	CCS Concepts: • Computing methodologies → Scene understanding; Computer Vision; Activity
    19	recognition and understanding; • Information systems → Data mining;
    20	Additional Key Words and Phrases: Person re-identification, Complex Lighting, Optimal Illumination Distance
    21	ACM Reference format:
    22	Chao Wang, Zhongyuan Wang, Ruimin Hu, Xiaochen Wang, and Wen Zhou. 2025. Optimal Illumination
    23	Distance Metrics for Person Re-Identification in Complex Lighting Conditions. ACM Trans. Multimedia Comput.
    24	Commun. Appl. 21, 1, Article 26 (January 2025), 18 pages.
    25	https://doi.org/10.1145/3700771
    26	
    27	This work was supported by the National Nature Science Foundation of China (Nos. U22A2035, 62371350, 62071339,
    28	U1903214, and 2042023kf1033).
    29	Authors’ Contact Information: Chao Wang, National Engineering Research Center for Multimedia Software, School of
    30	Computer Science, Wuhan University, Wuhan, China; e-mail: snipercwang@whu.edu.cn; Zhongyuan Wang (corresponding
    31	author), National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University,
    32	Wuhan, China; e-mail: wzy_hope@163.com; Ruimin Hu (corresponding author), National Engineering Research Center for
    33	Multimedia Software, School of Computer Science, Wuhan University, Wuhan, China; e-mail: hrm@whu.edu.cn; Xiaochen
    34	Wang, National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University,
    35	Wuhan, China; e-mail: clowang@163.com; Wen Zhou, Nanjing University of Finance and Economics, Nanjing, China;
    36	e-mail: zw_academic 08@163.com.
    37	Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
    38	provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
    39	full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
    40	Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
    41	prior specific permission and/or a fee. Request permissions from permissions@acm.org.
    42	© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
    43	ACM 1551-6865/2025/1-ART26
    44	https://doi.org/10.1145/3700771
    45	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
    46	
    47	26:2
    48	1
    49	
    50	C. Wang et al.
    51	
    52	Introduction
    53	
    54	Person Re-Identification (Re-ID) [1, 3, 5], which is widely utilized in public safety and security
    55	domains, retrieves images of the same pedestrian from different non-overlapping camera regions
    56	based on a given pedestrian image. Although existing Re-ID tasks perform exceptionally well in
    57	pedestrian recognition under slightly changing lighting conditions, real-world scenarios present
    58	significant challenges due to complex lighting variations. Pedestrians may appear at different times
    59	and locations in different cameras, leading to diverse illumination information in the captured
    60	images. Such illumination variations pose a major obstacle to recognition performance [28]. Consequently, there is an urgent need to develop Re-ID techniques that are highly robust to lighting
    61	variations and specifically tailored for real-world applications.
    62	The article investigates the task of person Re-ID in complex lighting scenarios, particularly for
    63	pedestrians. The task, as illustrated in Figure 1, can be defined as follows: Given an image of a
    64	pedestrian taken under unknown lighting conditions, the objective is to retrieve all images of the
    65	same pedestrian from a gallery consisting of images captured by cameras located in non-overlapping
    66	areas with varying lighting conditions.
    67	Compared to traditional Re-ID tasks, Re-ID under complex lighting scenarios presents additional
    68	challenges due to lighting noise. Illumination changes can alter the color representation of pedestrian
    69	images, making it unreliable to use color as a distinguishing feature. Furthermore, significant
    70	lighting differences between images can cause the intra-class distance (distance between images
    71	of the same pedestrian) to be larger than the inter-class distance (distance between images of
    72	different pedestrians). To address this, researchers have attempted to extract pedestrian features
    73	that remain stable under varying lighting conditions [2]. For example, Kviatkovsky et al. note that
    74	while color is susceptible to lighting changes, transforming the color space can mitigate this effect.
    75	They propose a multi-dimensional histogram method to describe image content [12]. Yu et al. argue
    76	that color distribution is a multi-modal structure that can remain consistent under different lighting
    77	conditions and suggest using shape content to describe this structure as an identification feature
    78	[27]. However, these shallow, content-based semantic features are only effective when there are
    79	significant differences in clothing color and type. They are less effective when pedestrians wear
    80	similar clothing.
    81	To enable effective pedestrian recognition, deep semantic representation features are crucial.
    82	Based on the Retinex theory [10], Zeng et al. decouple the essential features from pedestrian
    83	images, independent of lighting conditions, and combine them with lighting-independent features
    84	for feature matching [28].
    85	While this approach can separate the lighting information from the pedestrian features that
    86	eliminate the impact of lighting changes, it may also reduce the discriminative features of pedestrians. Zhang et al. [29] employ lighting transformation techniques to transform images taken under
    87	different lighting conditions into images with the same lighting, and then use traditional pedestrian
    88	recognition methods to extract more identifiable features. However, this approach assumes that
    89	all images can perfectly express semantic features under normalized lighting conditions, which
    90	may not be valid because different image contents have distinct identification values under various
    91	lighting conditions. This assumption may lead to additional losses introduced by uniform lighting
    92	conditions.
    93	As shown in Figure 2, existing approaches to solving the task of person Re-ID in complex illumination conditions can be broadly classified into two categories: (1) methods based on illuminationinvariant features and (2) methods based on illumination unification. The former often leads to a
    94	reduction in the distinctiveness of pedestrian features, while the latter introduces additional errors
    95	due to the process of normalizing lighting conditions. To address these limitations, we conducted
    96	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
    97	
    98	Optimal Illumination Distance Metrics for Person Re-Identification
    99	
   100	26:3
   101	
   102	Fig. 1. Given a pedestrian retrieval image with unknown illumination, from a retrieval gallery consisting of
   103	images taken by cameras in non-overlapping areas under different illumination conditions, retrieve all images
   104	containing the same pedestrian.
   105	
   106	Fig. 2. Existing approaches for person Re-ID in complex lighting: (1) illumination invariance feature-based
   107	methods and (2) illumination unification methods. Our method selects the optimal lighting condition for
   108	person Re-ID.
   109	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   110	
   111	26:4
   112	
   113	C. Wang et al.
   114	
   115	an analysis of the recognition performance of different image pairs (either of the same or different
   116	pedestrians) under varying lighting conditions. Our findings reveal that the recognition distance
   117	between image pairs fluctuates based on the lighting conditions, with each pair having a unique
   118	minimum distance and an optimal lighting condition for recognition. Based on this observation, we
   119	introduce the concept of optimal lighting: There exists a specific lighting condition under which the
   120	distance between image pairs is minimized compared to other lighting conditions. Therefore, during
   121	the inference stage, the distance between image pairs under their optimal lighting conditions can
   122	be used to improve ranking and retrieval accuracy.
   123	Specifically, we propose a novel framework called the Unified Illumination Feature Extraction
   124	(UIFE). The basic idea is shown in Figure 2 as our Method shows. First, a Generative Adversarial
   125	Network (GAN) is used to transform images under different lighting conditions into images
   126	under same lighting conditions (which is normal lighting condition: the light conditions of the
   127	original Market-1501 and DukeMTMC-ReID datasets). Next, Using the pre-trained feature extractor
   128	network, we extract the image features from the gallery and the query, respectively, and then
   129	calculate the Euclidean distance between each image feature in the query and each image feature
   130	in the gallery to form the base metric matrix. Then, different GANs are used to transform the
   131	images under normal lighting conditions into images under different lighting conditions. We use
   132	a pre-trained model under complex lighting conditions to extract the features of these images,
   133	and calculate the distance between each image feature in query and each image feature in gallery
   134	under the same lighting conditions separately, thus forming a series of metric matrices under
   135	different lighting conditions. Finally, these distance matrices are then concatenated along the
   136	channel axis, where each channel represents the metric distance of the image pair under different
   137	lighting conditions, and the minimum value along the channels is selected to form the final distance
   138	matrix for ranking. We evaluated our proposed method on two synthetic datasets, Market-1501-IA
   139	and Duke-IA, and achieved a certain improvement in performance.
   140	Our contributions can be summarized in threefold:
   141	— We analyzed the similarity relationship of image pairs under different lighting conditions,
   142	and the differences between similarity relationships. (1) Image pairs have different distances
   143	under different lighting conditions, but there is a unique minimum distance. (2) The minimum
   144	distance of each image pair is located in different lighting conditions, and complex multiinvariant situation, the pattern is difficult to find.
   145	— We introduce the concept of optimal illumination: For any given pair of pedestrian images,
   146	there exists a specific illumination condition under which the distance between the images is
   147	minimized compared to their distances under other lighting conditions.
   148	— We propose a new pedestrian recognition framework, UIFE, and conduct extensive experiments
   149	to validate the effectiveness of our approach.
   150	2
   151	
   152	Related Work
   153	
   154	2.1 Person Re-ID Under Constant Illumination
   155	Benefiting from the significant advance of deep learning, Re-ID has achieved remarkable improvement [25]. However, it is still challenged by real-world variations such as pose changes [16, 32],
   156	viewpoint changes, and occlusions [8, 9, 31]. To address these challenges, Cheng et al. [3] proposed
   157	a novel model training regularization method called Inter-Task Association critics. This method
   158	discovers potential associative knowledge between image Super-Resolution (SR) and person
   159	Re-ID and utilizes them as additional constraints during model training to enhance the compatibility
   160	between SR models and person Re-ID in high-resolution image space. Additionally, Wang et al. [22]
   161	proposed a method to identify and align features by learning higher-order relations and topological
   162	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 1, Article 26. Publication date: January 2025.
   163	
   164	
exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Privacy preserving person re-identification via anonymizing diffusion model.pdf' - | nl -ba | sed -n '1,170p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Pattern Recognition 174 (2026) 112939
     2	
     3	Contents lists available at ScienceDirect
     4	
     5	Pattern Recognition
     6	journal homepage: www.elsevier.com/locate/pr
     7	
     8	Privacy preserving person re-identiﬁcation via anonymizing diﬀusion
     9	model
    10	Rui Wang
    11	a
    12	
    13	a,1 , Ruikang Lin
    14	
    15	b,1 , Jiabing Yang
    16	
    17	a , Hao Ni
    18	
    19	b , Jingkuan Song
    20	
    21	c,∗
    22	
    23	Yingcai Honor College, University of Electronic Science and Technology of China, Chengdu, 611731, Sichuan, China
    24	
    25	b School of Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu, 611731, Sichuan, China
    26	c
    27	
    28	Shanghai Innovation Institute, Shanghai, 200231, Shanghai, China
    29	
    30	a r t i c l e
    31	Keywords:
    32	Person re-identiﬁcation
    33	Privacy protection
    34	Diﬀusion
    35	
    36	i n f o
    37	
    38	a b s t r a c t
    39	Existing public person re-identiﬁcation (ReID) datasets face signiﬁcant privacy challenges that raise critical
    40	ethical and legal concerns, particularly regarding the collection, and utilization of pedestrian imagery. Those
    41	datasets contain sensitive biometric information of real pedestrians, and privacy attributes(e.g. identity, age,
    42	location) may be deduced by attackers. To address this issue, numerous anonymization methods have been
    43	proposed to reconstruct pedestrian images. However, the privacy protection capability of these methods is limited due to incomplete removal of private information. In this paper, we propose an anonymizing diﬀusion
    44	model (ADM), a novel generation-based privacy protection approach for ReID. Through 2-stage ﬁne-tuning
    45	of diﬀusion model and a novel prompt construction algorithm, ADM is able to maintain the usability of the
    46	generated data while preventing the real ID from being retrieved by either neural networks or human observers. Building on ADM, we release three privacy-compliant ReID benchmarks-Market1501-A, MSMT17-A,
    47	and CUHK-SYSU-A-which systematically replace original pedestrian identities with synthetic yet discriminative counterparts. These datasets retain re-ID utility (e.g., posture, clothing styles) while eliminating biometric
    48	identiﬁers, oﬀering the ﬁrst standardized anonymized alternatives to their non-compliant originals. Furthermore, we introduce a new metric called ID separation degree (ISD) to measure the visual diﬀerence between
    49	real and generated samples. To the best of our knowledge, this is the ﬁrst exploration of utilizing diﬀusion
    50	models to generate new IDs for privacy protection ReID. We release an open-source implementation ofADM at
    51	https://github.com/Physics-wr/Anonymizing-Diﬀusion-Model.
    52	
    53	1. Introduction
    54	Person re-identiﬁcation (ReID) is to detect and track the same target
    55	person across diﬀerent camera views based on image features. However,
    56	the collection of training data for ReID tasks faces severe privacy leakage
    57	issues. Those data often contain high-resolution images captured in public spaces without explicit informed consent from individuals. A primary
    58	issue lies in the irreversible exposure of sensitive biometric information,
    59	including facial features, gait patterns, and clothing details, which can
    60	be exploited for unauthorized identity tracking or linkage to auxiliary
    61	personal data. Furthermore, the open accessibility of these datasets exacerbates potential misuse, as malicious actors could repurpose them
    62	to develop surveillance tools or infer private attributes (e.g., health status, location history) through attacks. For example, as shown in Fig. 1,
    63	as long as a photo of these pedestrians is obtained from social media,
    64	
    65	attackers can retrieve other images of pedestrians from public datasets
    66	for more privacy information, such as their whereabouts. Mainstream
    67	ReID datasets such as Market-1501 [1], DukeMTMC-reID [2], CUHKSYSU [3] and MSMT17 [4] are mostly collected from public scenarios
    68	such as malls and campuses, but whether the collection of these datasets
    69	has fully considered the participants’ right to information and portrait
    70	remains in doubt.
    71	Although these datasets have made important contributions to the
    72	advancement of ReID research, insuﬃcient privacy protection hinders
    73	the adoption of the technology in real application environments. To address this issue, some existing works on privacy protection for ReID
    74	attempt to reconstruct datasets using image encryption or adversarial attacks. Image encryption aims to selectively blur sensitive regions [5], add noise to the image [5], or encrypt [6] before releasing
    75	datasets. The images obtained by encryption-based methods require a
    76	
    77	∗
    78	
    79	Corresponding author.
    80	E-mail addresses: wangrui604@foxmail.com (R. Wang), linrk.proton@gmail.com (R. Lin), jiabingyang01@gmail.com (J. Yang), haoni0812@gmail.com (H. Ni),
    81	jingkuansong@tongji.edu.cn (J. Song).
    82	1
    83	Contributed equally to this work.
    84	https://doi.org/10.1016/j.patcog.2025.112939
    85	Received 26 May 2025; Received in revised form 4 November 2025; Accepted 14 December 2025
    86	Available online 17 December 2025
    87	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    88	
    89	Pattern Recognition 174 (2026) 112939
    90	
    91	R. Wang et al.
    92	
    93	Fig. 1. Most benchmark datasets aggregate high-resolution pedestrian images captured across multiple non-overlapping surveillance cameras, often with detailed
    94	metadata such as camera IDs, sequence numbers, and timestamps. This granular spatiotemporal information enables adversaries to reconstruct individual movement
    95	patterns by cross-referencing camera deployment locations and timestamps, thereby exposing sensitive trajectories and daily routines of real-world pedestrians.
    96	
    97	complex decryption process before being utilized by ReID models. Adversarial attack approaches [7–9] add carefully crafted perturbations
    98	to samples to reduce their recognizability during retrieval. However,
    99	to ensure data usability, most of these methods do not change the
   100	IDs of pedestrians. For example, adversarial attack methods only add
   101	slight perturbations to the images, allowing the true IDs to still be retrieved by humans or other ReID models. Essentially, these anonymization methods hide private information rather than eliminate it. Therefore, these methods still have privacy leakage risks. Despite those efforts to construct privacy-preserving datasets, the inherent trade-oﬀ
   102	between data utility and anonymization eﬃcacy remains unresolved,
   103	often leading to suboptimal model performance or residual identity
   104	leakage.
   105	To eliminate the privacy information of the original data while preserving data utility, we propose an anonymizing diﬀusion model (ADM),
   106	a novel generation-based privacy protection approach for ReID. ADM
   107	aims to anonymize ReID datasets by generating visually distinct new
   108	IDs, preventing being retrieved by either neural networks or human observers. To this end, ADM utilizes Stable Diﬀusion (SD) [10] to generate anonymous images with new IDs through text prompts. As shown
   109	in Fig. 2, compared with previous methods that edit real images, the
   110	generated images originate from noise, so private information is eliminated at its source without relying on potentially reversible alterations.
   111	To enable diﬀusion model to produce synthetic ReID dataset, we ﬁrst
   112	ﬁne-tune it on the entire dataset. To achieve ﬁne-grained control over
   113	pedestrian visual attributes, a second stage ﬁne-tuning on subsets is
   114	conducted to facilitate intra-id consistency.A new anonymous dataset
   115	is generated based on novel text prompts constructed through transformation of original image captions. Finally, some outlier may exist among speciﬁc generated ID, which hinders data utility. ADM ﬁlters outlier samples from the generated dataset through the proposed
   116	griddle.
   117	Additionally, to quantitatively measure the visual diﬀerence between original and generated samples, we introduce a new metric called
   118	ID separation degree (ISD). ISD measures the anonymization capability of a privacy protection method based on the feature distance in the
   119	embedding space. A higher ISD indicates greater dissimilarity between
   120	the real and generated samples, and thus enhanced privacy protection
   121	level.
   122	To analyze the eﬀectiveness of our proposed method, we conduct
   123	thorough experiments on commonly used ReID datasets and compare
   124	them with other privacy-preserving methods. The results show that ADM
   125	
   126	achieves both competitive utility capability and privacy-preserving capability.
   127	Experiments on public datasets demonstrate that ADM achieves
   128	excellent privacy protection performance while maintaining competitive ReID accuracy. Based on ADM, we obtain three privacy-compliant
   129	derivatives of widely adopted benchmarks Market1501, MSMT17 and
   130	CUHK-SYSU, enabling researchers to develop robust re-ID models without compromising individual privacy. The contributions of our work can
   131	be concluded as:
   132	•
   133	
   134	We introduce the ﬁrst set of anonymized ReID datasets (Market1501A, MSMT17-A, CUHK-SYSU-A) that eliminate personally identiﬁable
   135	information while retaining ReID eﬃcacy, providing ethically compliant alternatives to mainstream benchmarks.
   136	• Through 2-stage ﬁne-tuning and novel prompt construction algorithm, ADM is able to synthetize ReID dataset with a trade-oﬀ between data utility and privacy protection capability. We also propose ISD as a new metric to evaluate privacy protection eﬃcacy of
   137	the entire synthetic dataset.
   138	• We conduct experiments on benchmark datasets to demonstrate
   139	ADM can eﬀectively remove identity associations between real and
   140	generated data, with superior performance in preventing retrieving
   141	private identities from public datasets, while maintaining competitive ReID accuracy.
   142	2. Related work
   143	2.1. Person re-identiﬁcation
   144	Person re-identiﬁcation (ReID) is an important person retrieval
   145	task, which aims to retrieve a person of interest across multiple nonoverlapping camera views. It has wide applications in video surveillance, intelligent security, and other ﬁelds. In recent years, many deep
   146	learning-based methods have been proposed to solve this problem. Traditionally, CNNs have served as the dominant backbone [11–14] for extracting discriminative features from pedestrian images. Beyond CNNbased approaches, Transformer architectures [15–17] have emerged as
   147	a powerful alternative in recent years. Metrics-based learning methods are commonly used to calculate the similarity between person images [18–20]. Chen et al.[21] apply GAN to ReID to solve the domain adaptation problems. However, person ReID still faces severe privacy issues. For example, the widely used dataset DukeMTMC-reID
   148	2
   149	
   150	Pattern Recognition 174 (2026) 112939
   151	
   152	R. Wang et al.
   153	
   154	Fig. 2. The previous methods protect ID privacy through attacking, encryption, etc. These methods fail to erase the correlation with the original ID, leaving private
   155	information that can still be retrieved through human observation or decryption. ADM (ours) protects ID privacy by generating new IDs to replace the original ones.
   156	
   157	2.3. CLIP and stable diﬀusion
   158	
   159	[2] was withdrawn due to privacy issues. Most existing ReID models rarely consider privacy protection problems, and some methods
   160	proposed have diﬃculty in balancing privacy protection and model
   161	performance.
   162	
   163	CLIP [34], as a pre-trained model based on a transformer architecture, has been widely applied to various multimodal tasks [35–38].
   164	Through contrastive learning on a huge dataset, CLIP builds semantic
   165	connections between text and images to achieve mutual understanding between images and text. ClipCap [39] uses CLIP encodings as image caption preﬁxes, and ﬁne-tunes a language model GPT2 [40] to
   166	generate image captions through a simple mapping network. Shao et
   167	al.[41] utilize a divide-conquer-combine strategy to generate text descriptions of images through the CLIP paradigm, constructing a largescale text-labeled person dataset “LUPerson-T”. encoder of CLIP to generate prompts’ latent embeddings as conditions to guide image generation. Diﬀusion models [42] are generative models that include a process
   168	of forward noise injection diﬀusion and a reverse process of denoising
   169	reconstruction. An improvement of Stable Diﬀusion [10] over Diﬀusion
   170	models is that it performs diﬀusion in the latent space of images, achieving higher computational eﬃciency. In addition, Stable Diﬀusion introduces conditional control. It uses the text encoder of CLIP to generate

codex
现在进入整理前的最后一步：我补读每篇引言里作者如何提出“失败现象”。方法论拆解最容易误判的地方就在这里，因为很多创新不是模块新，而是把旧失败重新命名成一个可解决的问题。
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
/bin/zsh -lc "pdftotext -f 1 -l 4 'Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
     2	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
     3	
     4	1
     5	
     6	Occlusion-aware Cross-modality Completion
     7	Network for Occluded Visible-Infrared Person
     8	Re-Identification
     9	Jian Yu, Yujian Feng, Shuai You, Zhengjun Jing, Fei Wu, Yimu Ji, Xiao-yuan Jing
    10	
    11	Abstract—Occlusion is a key challenge in real-world person reidentification (ReID), as partial occlusions often remove critical
    12	identity cues. Most existing methods focus on feature extraction
    13	for visible regions while ignoring the holistic contextual information of pedestrians, leading to incomplete feature representations.
    14	This issue is particularly severe in visible-infrared ReID (VIReID), where spatial inconsistencies of occluded regions and
    15	modality-specific feature degradation jointly exacerbate feature
    16	misalignment, but has received little attention in previous studies. Therefore, we propose an Occlusion-Aware Cross-Modality
    17	Completion Network (OCCNet) to explicitly complete missing
    18	identity cues, which integrates three complementary components.
    19	The Graph-guided Feature Completion Transformer (GFCT)
    20	embeds a dynamically constructed semantic graph into selfattention to complete occluded tokens within each modality,
    21	while the Semantic-Aware Region Completion Module (SRCM)
    22	aggregates tokens into coherent regions and adaptively transfers complementary cues across modalities. To ensure semantic
    23	coherence, a Multi-Distribution Consistency Alignment (MCA)
    24	loss is employed to regularize token- and region-level completions
    25	toward a unified distribution. By integrating these components,
    26	OCCNet recovers missing identity information at multiple levels
    27	and mitigates modality gaps caused by occlusion. To evaluate its
    28	effectiveness under realistic occlusion scenarios, we first construct
    29	two realistic occluded benchmarks, Occluded-SYSU-MM01 and
    30	Occluded-RegDB, on which extensive experiments demonstrate
    31	that our OCCNet outperforms state-of-the-art methods.
    32	Index
    33	Terms—Occluded
    34	visible-infrared
    35	person
    36	reidentification, Cross-modality feature completion, Distribution
    37	alignment, Occluded VI-ReID datasets.
    38	
    39	Woman
    40	Black backpack
    41	Short hair
    42	...
    43	
    44	Single
    45	Modality
    46	
    47	Original
    48	Image
    49	
    50	Pose
    51	Estimation
    52	
    53	Human
    54	Parsing
    55	
    56	Instance
    57	Segmentation
    58	
    59	Attribute
    60	Annotations
    61	
    62	(a)
    63	
    64	Vis
    65	Modality
    66	
    67	Intra-modality
    68	Complete
    69	
    70	Matching
    71	
    72	Complete
    73	Region
    74	Inter-modality
    75	Complete
    76	
    77	Complete
    78	Region
    79	
    80	NIS
    81	Modality
    82	
    83	Matching
    84	Intra-modality
    85	Complete
    86	
    87	(b)
    88	
    89	Fig. 1: The difference between the existing methods and
    90	our approach. (a) Existing methods. (b) Our Occlusion-aware
    91	Cross-modality Completion Network (OCCNet).
    92	
    93	I. I NTRODUCTION
    94	Visible-infrared person re-identification (VI-ReID) plays a
    95	critical role in intelligent surveillance systems by matching
    96	the same individual across visible and near-infrared cameras.
    97	Although numerous VI-ReID methods [1]–[5] have been proposed in recent years, most of them assume full body visibility. In real-world scenarios, however, occlusions (e.g., signs,
    98	vehicles or other pedestrians) frequently occur in crowded
    99	environments such as railway stations, schools, and streets,
   100	leading to severe degradation of recognition performance.
   101	This motivates the study of occluded visible-infrared person
   102	Jian Yu, Yujian Feng, and Zhengjun Jing are with the School of Computer Engineering,
   103	Jiangsu University of Technology, Changzhou, China (email: yujian 928@163.com;
   104	fengyujian 904@163.com; jzjing@jsut.edu.cn).
   105	Shuai You and Xiao-Yuan Jing are with the School of Computer, Wuhan University,
   106	Wuhan, China (email: youshuai 666@163.com; jingxy 2000@126.com).
   107	Fei Wu is with the College of Automation, Nanjing University of Posts and Telecommunications, Nanjing, China (email: wufei 8888@126.com).
   108	Yimu Ji is with the School of Computer Science and Technology, Nanjing University
   109	of Posts and Telecommunications, Nanjing, China (email: jiym@njupt.edu.cn).
   110	Yujian Feng is the corresponding author.
   111	
   112	re-identification (Occluded VI-ReID), which addresses crossmodality matching under partial occlusion. However, to date,
   113	only a single work has explicitly explored this problem [6].
   114	Local detail preservation and cross-modality alignment are
   115	both important in complex multi-modality scenarios [7]–[9].
   116	These requirements are particularly critical for Occluded VIReID, which presents two main challenges. (1) Occlusion
   117	leads to the loss of critical identity cues such as facial features, clothing textures, or distinctive accessories, resulting in
   118	incomplete and less discriminative pedestrian representations;
   119	(2) Occlusion significantly exacerbates modality discrepancies,
   120	mainly caused by two factors: On the one hand, the spatial
   121	locations of occluded regions often differ across modalities,
   122	resulting in positional misalignment. On the other hand,
   123	occlusion affects each modality in different ways, leading
   124	to inconsistent feature degradation. For example, in visible
   125	light images, occlusions usually obscure visual cues such as
   126	color and texture, while in infrared images, occlusions often
   127	
   128	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
   129	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   130	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   131	
   132	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   133	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
   134	
   135	2
   136	
   137	mask structural information such as contours and shapes.
   138	This asymmetry significantly increases the difficulty of crossmodality feature alignment.
   139	As illustrated in Fig. 1(a), most existing occluded ReID
   140	methods [10]–[13] are developed from a single-modality perspective. They typically handle occlusion by discarding the
   141	occluded parts and extracting features only from the visible body regions, often relying on auxiliary cues such as
   142	pose estimation, human parsing, instance segmentation, or
   143	attribute annotations. While effective to some extent, these
   144	approaches inevitably neglect the holistic contextual semantics
   145	of pedestrians, such as the overall silhouette, the swing range
   146	of the arms, and the movement posture of the legs, which
   147	are crucial for discriminative alignment. As a result, the
   148	learned representations remain incomplete and less robust
   149	under severe occlusions. This limitation also exists in the only
   150	study on Occluded VI-ReID, OCMF [6], and becomes even
   151	more critical in the cross-modality scenario, where the visible
   152	regions in different modalities are often spatially inconsistent.
   153	Such asymmetry easily leads to severe feature misalignment
   154	and degraded retrieval performance.
   155	In contrast, as shown in Fig. 1(b), our method addresses
   156	occlusion from a completion perspective rather than a discarding one. Specifically, we propose to fully exploit both intramodality contextual cues and inter-modality complementary
   157	information to complete the occluded regions. By jointly completing missing features within modality and across modalities,
   158	our framework generates semantically holistic and structurally
   159	consistent representations, thus enabling more robust and
   160	discriminative cross-modality matching under occlusion.
   161	To implement this framework, we propose an OcclusionAware Cross-Modality Completion Network (OCCNet), which
   162	explicitly completes missing identity cues by integrating
   163	token-level and region-level completion with distribution consistency constraints. Specifically, OCCNet is composed of
   164	three complementary modules. First, the proposed Graphguided Feature Completion Transformer (GFCT) incorporates
   165	dynamically constructed semantic graphs into self-attention to
   166	complete occluded tokens, ensuring that completed features
   167	are structurally consistent with surrounding contexts within the
   168	same modality. Second, a Semantic-Aware Region Completion
   169	Module (SRCM) is designed to aggregate tokens into semantic
   170	regions and performs adaptive cross-modality completion,
   171	allowing complementary cues from another modality to fill
   172	missing information and alleviate spatial misalignment.
   173	To further ensure semantic coherence across different completion pathways, we proposed a Multi-Distribution Consistency Alignment (MCA) loss to align token- and regionlevel completions with a pseudo-holistic reference through
   174	symmetric distribution matching. By jointly modeling these
   175	three components, OCCNet produces semantically holistic and
   176	structurally consistent representations, significantly improving
   177	robustness against occlusion in cross-modality matching.
   178	The main contributions of this paper can be summarized as
   179	follows:
   180	(1) A Graph-guided Feature Completion Transformer
   181	(GFCT) is proposed to embed dynamic semantic graphs into
   182	
   183	self-attention, constraining interactions to semantically relevant pairs and enabling robust token-level feature completion.
   184	(2) A Semantic-aware Region Completion module (SRCM)
   185	is proposed to cluster tokens into semantic regions and perform
   186	adaptive cross-modality feature completion, effectively alleviating spatial misalignment and modality-specific degradation.
   187	(3) A Multi-Distribution Consistency Alignment (MCA)
   188	loss is designed to align token- and region-level completions
   189	with a unified pseudo-holistic reference via symmetric KL
   190	divergence, improving multi-path semantic coherence.
   191	(4) Two new benchmarks, Occluded-SYSU-MM01 and
   192	Occluded-RegDB, are built with realistic occlusions, which
   193	are the first real occluded VI-ReID datasets. Extensive experiments show that our method achieves state-of-the-art performance in cross-modality occlusion scenarios.
   194	II. R ELATED W ORK
   195	A. Visible-infrared Person Re-identification
   196	Visible-infrared person re-identification (VI-ReID) aims to
   197	match pedestrian identities across visible and infrared cameras. Its main challenges lie in the inter-modality discrepancy caused by different imaging mechanisms and the intramodality variation caused by viewpoint changes.
   198	To address these challenges, many VI-ReID methods have
   199	been proposed from the perspective of modality alignment.
   200	For example, Zhang et al. [1] investigated the real-world label
   201	noise problem in VI-ReID and proposed a robust hybrid loss to
   202	improve model robustness under instance-dependent and realworld noisy annotations. Pang et al. [2] proposed a hierarchical
   203	clustering and refinement framework for unsupervised VIReID. Wu et al. [14] developed a modality- and instance-aware
   204	adaptation framework to alleviate cross-modality discrepancy.
   205	Liang et al. [4] introduced a multi-level joint alignment method
   206	to reduce modality discrepancy in image and feature spaces.
   207	Zhao et al. [15] proposed a historical consistency learning
   208	strategy to correct noisy labels in VI-ReID.
   209	However, the effectiveness of these methods relies on
   210	obtaining complete, unobstructed images. When pedestrians
   211	are occluded by trees, buildings, vehicles, text, and other
   212	factors, these methods fail to effectively extract discriminative
   213	features. Therefore, this paper proposes an Occlusion-aware
   214	Cross-modality Completion Network (OCCNet) to complete
   215	the features of pedestrians in occluded areas, thus improving
   216	cross-modality matching performance.
   217	B. Occluded Person Re-identification
   218	In real-world scenarios, pedestrians frequently encounter
   219	occlusion from various obstacles, necessitating the matching
   220	of occluded pedestrians with those in the gallery, defining
   221	the task of Occluded Person ReID [16]–[21]. The traditional
   222	methods focus on single-modality pedestrian matching for
   223	visible light/RGB images. For example, Wang et al. [16]
   224	proposed a text-guided hierarchical context blending network
   225	for occluded person re-identification, which integrates textual
   226	information with visual features to enhance contextual representation under occlusion. Chen et al. [17] introduced a
   227	hierarchical token-aware reconstruction framework to alleviate
   228	
   229	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:58 UTC from IEEE Xplore. Restrictions apply.
   230	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   231	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   232	
   233	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   234	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3688708
   235	
   236	3
   237	
   238	Fig. 2: Overview of the proposed OCCNet. Modality-specific features are first extracted by a ResNet50 backbone and then
   239	enhanced through two completion stages: the Graph-guided Feature Completion Transformer (GFCT) for token-level completion
   240	and the Semantic-Aware Region Completion Module (SRCM) for cross-modality region-level completion. The network is trained
   241	with a total loss that enforces consistency between token- and region-level completions, resulting in discriminative features for
   242	robust cross-modality person retrieval.
   243	
   244	modality discrepancy in VI-ReID. Qiu et al. [18] proposed a
   245	hierarchical middle-feature learning network to build a more
   246	reliable common feature space for VI-ReID. Zhang et al. [19]
   247	developed a mask-aware hierarchical transformer to improve
   248	visible-part localization and feature completion under occlusion. Dong et al. [20] addressed Occluded Person ReID from
   249	an adversarial defense perspective, while further introduced a
   250	multi-view framework that transfers complementary identity
   251	cues to single-image representations [21].
   252	Although these methods have advanced occlusion-aware
   253	representation learning or cross-modality feature modeling,
   254	they remain insufficient for occluded VI-ReID. This task
   255	must simultaneously address missing identity cues caused
   256	by occlusion and feature inconsistency caused by modality
   257	variation, resulting in more severe cross-modality misalignment. Moreover, single-modality completion cannot exploit
   258	complementary cues from another modality, while existing
   259	VI-ReID methods generally emphasize modality alignment
   260	rather than explicit recovery of missing identity information.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '1,300p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
     2	
     3	2095
     4	
     5	Optimal Proxy Mining Contrastive Network for
     6	Unsupervised Person Re-Identification
     7	Ge Cao , Member, IEEE, Qing Tang , Member, IEEE, Xuan-Thuy Vo , Member, IEEE,
     8	Adri Priadana , Member, IEEE, and Kang-Hyun Jo , Senior Member, IEEE
     9	Abstract—Unsupervised person re-identification (Re-ID) performance enhancement hinges on extracting the most informative
    10	features from unlabeled person datasets. In recent approaches,
    11	proxy-based contrastive learning with awareness of camera labels
    12	has been adopted for model training, thereby achieving highly
    13	promising results. However, inappropriate selections of contrastive pairs can significantly degrade the performance of these
    14	models. To address this issue, we propose the Optimal Proxy Mining Contrastive Network (OPMCN), a novel framework designed
    15	to strategically optimize the selection of proxies for positive and
    16	negative pair formation, thus enhancing the efficacy of contrastive
    17	training. The OPMCN framework proposes two specific contrastive losses: Hardest Camera Proxy Mining (HCPM) and False
    18	Negative Proxies Mining (FNPM), each essential for enhancing
    19	model performance in unsupervised settings. The HCPM loss
    20	targets proxies from the most challenging cameras to maximize
    21	semantic differences between pairs while ensuring minimal background shifts. In contrast, the FNPM loss counters noise in
    22	pseudo labels by prioritizing similarity rankings over clustering
    23	results to effectively identify and correct false negatives among
    24	proxies. Moreover, we have developed the Pyramid Kernel Global
    25	Context (PKGC) block, which employs an attention mechanism
    26	that focuses on identity-invariant semantic cues in instances.
    27	This module utilizes optimally sized convolutional kernels to
    28	enhance identity recognition consistency across camera-based
    29	variations, thereby improving the precision of feature extraction.
    30	Experimental results on several popular datasets prove that our
    31	work surpasses existing unsupervised person Re-ID approaches
    32	to a remarkable extent.
    33	Index Terms—Unsupervised person re-identification, contrastive learning, attention mechanism.
    34	
    35	I. I NTRODUCTION
    36	
    37	P
    38	
    39	ERSON re-identification (Re-ID) aims at precisely identifying the target pedestrian from a vast pool of instances. It
    40	serves as a cornerstone in non-overlapping cross-camera video
    41	surveillance systems, offering essential support for various
    42	downstream computer vision tasks by enabling person ID
    43	tracking. Over the past few years, deep learning-based person
    44	Re-ID methods [1], [2], [3], [4] have become highly dependent
    45	on the representational quality of the training datasets, causing
    46	Received 10 October 2024; revised 12 May 2025; accepted 24 August 2025.
    47	Date of publication 27 August 2025; date of current version 5 February 2026.
    48	This result was supported by the “Regional Innovation System & Education
    49	(RISE)” through the Ulsan RISE Center, funded by the Ministry of Education
    50	(MOE) and the Ulsan Metropolitan City, Republic of Korea. (2025-RISE07-001). This article was recommended by Associate Editor Y. S. Rawat.
    51	(Corresponding author: Kang-Hyun Jo.)
    52	Ge Cao, Xuan-Thuy Vo, Adri Priadana, and Kang-Hyun Jo are with the
    53	Department of Electrical, Electronic and Computer Engineering, University
    54	of Ulsan, Ulsan 44610, Republic of Korea (e-mail: acejo@ulsan.ac.kr).
    55	Qing Tang is with the Department of Research and Development center,
    56	INTERX, Ulsan 44542, Republic of Korea.
    57	Digital Object Identifier 10.1109/TCSVT.2025.3603199
    58	
    59	a dramatic decline in the test performance of these models
    60	when deployed in real-world surveillance environments. To
    61	combat this decline, it is essential to collect data from real
    62	scenarios; however, manual annotation of such data is both
    63	time-consuming and labor-intensive.
    64	To mitigate this issue, unsupervised learning offers a viable
    65	solution that enhances testing performance without the ponderous manual annotations. Furthermore, with ongoing research
    66	in person Re-ID under unsupervised training scheme, numerous approaches [5], [6], [7], [8], [9] have shown progressively
    67	improved outcomes on public datasets, steadily approaching
    68	the efficacy of supervised training models. Unsupervised person Re-ID is divided into unsupervised domain adaptation
    69	[10], [11], [12], [13], [14], [15], [16] and purely unsupervised
    70	methods [17], [18], [19], [20] based on the use of labeled data
    71	from other domains. The latter, not relying on any annotated
    72	labels, presents greater challenges and offers increased practical value. These methods underpin foundational technologies
    73	for cross-camera tracking [21], [22] and pose estimation
    74	[23], [24] and are integral to developing intelligent video
    75	surveillance and smart traffic systems [25], [26]. In this paper,
    76	we concentrate on developing purely unsupervised methods to
    77	enhance flexibility in person Re-ID.
    78	Recent works indicate that pseudo label based methods
    79	[5], [6], [27] facilitate the stable and efficient training for
    80	unsupervised training for person Re-Identification, while typically adhering to the following pipeline: (1) obtain the feature
    81	representation through a designed extractor; (2) generate the
    82	pseudo labels by clustering methods (i.e. DBSCAN [28]); (3)
    83	optimize the extractor with a set of loss functions applying
    84	contrastive learning. As training progresses, the model incrementally masters the semantic features of analogous instances
    85	and more adeptly articulates the characteristics of these
    86	instances.
    87	With the aim of achieving notable accuracy, prior works
    88	[6], [11], [13], [15], [18], [20], [29] have utilized camera
    89	labels to further split identity clusters into multiple proxies,
    90	enhancing the granularity and effectiveness of person Re-ID
    91	models through what is known as proxy-based contrastive
    92	learning [6], [19], [20], [30]. Among these, CAP [6] has
    93	played a pivotal role by identifying the high intra-camera
    94	similarity of instances captured by the same camera, based
    95	on inherent characteristics. It proposed camera-aware proxies
    96	and designed both intra-camera and inter-camera contrastive
    97	learning strategies to improve ID-discrimination within and
    98	across cameras. CAP’s innovative approach has significantly
    99	influenced many subsequent proxy-based contrastive learning
   100	methods, cementing its importance in advancing the field.
   101	
   102	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
   103	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   104	See https://www.ieee.org/publications/rights/index.html for more information.
   105	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   106	
   107	2096
   108	
   109	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   110	
   111	Although the existing methods have achieved promising performance in unsupervised person Re-ID, they are still unable
   112	to resolve the following issues:
   113	• Misaligned contrastive pairs: The core of contrastive
   114	learning in purely unsupervised person Re-ID lies in
   115	selectively choosing appropriate positive and negative
   116	samples to aid semantic distinction, a process where nonselective methods may fail to yield optimal results due to
   117	inherent variability and noise in pseudo labels. Such noise
   118	often misrepresents the ID-distinctive feature of instances,
   119	leading to misalignment that introduce false negatives and
   120	undermine learning by reinforcing semantical noise. The
   121	failure to accurately adjust both positive and negative
   122	instances in contrastive learning significantly limits the
   123	effectiveness and generalization of the model, a challenge
   124	that current methods rarely address.
   125	• Insufficient global information: In the Re-ID task,
   126	instances are often captured from cluttered environments, leading models to inadvertently neglect identitydistinctive features and erroneously focus on background
   127	noise. The capability of convolution neural networks
   128	(CNNs), including architectures like ResNet [31], to
   129	extract complex or subtle features is limited. These
   130	networks tend to prioritize local feature extraction, inadequately addressing the integration of global contextual
   131	information. Consequently, current methodologies are
   132	insufficiently robust in extracting pertinent information
   133	for effective model performance.
   134	To address the challenges outlined, this paper introduces the
   135	Optimal Proxy Mining Contrastive Network (OPMCN) for the
   136	unsupervised person Re-ID task. Within OPMCN, we develop
   137	two specialized contrastive loss functions: the Hardest Camera
   138	Proxy Mining (HCPM) and the False Negative Proxies Mining
   139	(FNPM). HCPM operates under the assumption that pseudo
   140	labels are accurate, selecting proxies from the hardest camera.
   141	These proxies, although in the same cluster as the query,
   142	exhibit the largest semantic distinctions, thus providing rich
   143	discriminative information. This method effectively reduces
   144	background variations, allowing the model to concentrate on
   145	ID-distinctive features. Conversely, FNPM directly addresses
   146	the performance degradation due to the inevitable noise in
   147	pseudo labels by introducing an anti-noise similarity ranking
   148	system that refines the selection of contrastive pairs. Together,
   149	HCPM and FNPM identify optimal proxies, substantially
   150	enhancing model training. Subsequently, the paper presents
   151	the Pyramid Kernel Global Context (PKGC) block, which
   152	supports the backbone Pyramid Kernel Attention (PKA) network by enhancing key features and facilitating the integration
   153	of global information. This module pyramidally adjusts the
   154	processing of spatial information according to the feature map
   155	size, effectively managing different scales of data and aiding
   156	the network in recognizing identity-distinctive features while
   157	adapting to background changes.
   158	In summary, the main contributions of our model are as
   159	follows:
   160	• We introduced the Hardest Camera Proxy Mining
   161	(HCPM) approach to mine proxies from the most
   162	
   163	challenging cameras, leveraging the richest discriminative
   164	information within each cluster to focus the model on
   165	learning identity-invariant features while reducing background noise.
   166	• The False Negative Proxies Mining (FNPM) method
   167	counters the adverse effects of noisy pseudo labels, which
   168	mines potential false negatives and refines the selection
   169	of contrastive pairs to improve overall accuracy and
   170	robustness.
   171	• The PKGC block is proposed to enhance key features
   172	by pyramidally adjusting kernel sizes, enabling the backbone network to better integrate global information and
   173	extract identity-distinctive features. This design improves
   174	adaptability and recognition accuracy.
   175	• Experiments conducted across multiple large-scale Re-ID
   176	datasets confirm that the proposed method significantly
   177	enhances the performance of purely unsupervised person
   178	Re-ID, surpassing other approaches by a considerable
   179	margin.
   180	II. R ELATED W ORKS
   181	A. Unsupervised Person Re-Identification
   182	With the rapidly increasing demand for social security and
   183	public services, video surveillance systems, as the cornerstone
   184	of smart cities, have attracted widespread attention [21], [22],
   185	[23], [24], [25], [26]. For the person Re-ID system, obtaining
   186	instances of pedestrians is not difficult, but the challenge lies
   187	in the fact that modern industry is incapable of affording
   188	the cost of manually labeling a large number of pedestrian
   189	instances. Fortunately, the emergence of some large-scale
   190	datasets [32], [33], [34] has shown the possibility of not
   191	relying entirely on annotated instances for researchers, i.e.,
   192	the unsupervised person Re-ID. The unsupervised Re-ID is
   193	divided into unsupervised domain adaptation (UDA) methods
   194	[5], [10], [11], [12], [13], [14], [15], [16] and purely unsupervised methods [6], [7], [8], [17], [18], [19], [20], based on
   195	whether it relies on annotated data beyond the target unlabeled
   196	domain.
   197	Existing methods commonly apply the pseudo label based
   198	pipeline [5], [6], [27] for training and achieving notable
   199	performance. By utilizing camera factors for unsupervised
   200	learning, prior works have focused on different aspects of
   201	model optimization. Although almost all methods directly
   202	apply DBSCAN [28] and K-means as the clustering algorithm,
   203	some methods [17], [35], [36] proposed new clustering methods to generate better pseudo label results. BUC [17] and
   204	HCT [37] implement learning in a camera-agnostic manner,
   205	effectively capturing similarity within identities but often overlooking variance within the same identity due to changes in
   206	camera angles. Conversely, DARC [35] segments the target
   207	domain into distinct camera-specific groups to enhance global
   208	clustering. Similarly, STS [36] generates a camera offset
   209	matrix alongside a unified distance matrix to mitigate camerainduced variances. Alternatively, several approaches [6], [19],
   210	[20] bifurcate the Re-ID process into intra- and inter-camera
   211	learning phases, allowing the discriminative capabilities honed
   212	within individual cameras to bolster identity matching across
   213	
   214	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:02:14 UTC from IEEE Xplore. Restrictions apply.
   215	
   216	CAO et al.: OPTIMAL PROXY MINING CONTRASTIVE NETWORK FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   217	
   218	different cameras. Notably, CAP [6] introduces camera-aware
   219	proxies to address intra-ID variance, pursuing unsupervised
   220	learning from both intra- and inter-camera viewpoints. Additionally, IICS [20] establishes proxy memories based on
   221	separate clustering outcomes for each camera, facilitating the
   222	computation of intra- and inter-camera similarities to inform
   223	model training. Beyond these real-image-based approaches,
   224	Privacy-Protected Re-ID [38] introduces a novel paradigm
   225	where no real person images are used during training, relying
   226	instead on synthetic samples generated from privacy-irrelevant
   227	cues. However, these methods do not systematically select
   228	positive and negative samples for contrastive learning while
   229	incorporating camera information. This paper introduces the
   230	optimal proxy mining contrastive method to address this issue
   231	and provides an analysis and results of various positive and
   232	negative sample selection methods.
   233	B. Metric Learning With Proxies
   234	Metric learning is crucial for person Re-ID and various
   235	other vision tasks. The Proxy-NCA [39] method introduced
   236	the use of proxies to represent groups of instances, enhancing
   237	the contextual information available for similarity calculations.
   238	Subsequent research [6], [19], [20] has confirmed the value
   239	of proxies in unsupervised person Re-ID, employing proxybased strategies to divide each cluster into multiple proxies
   240	based on camera labels. This division helps better capture local
   241	structures, addressing the substantial inter-camera variability
   242	inherent in the Re-ID task. This study further refines this
   243	approach by partitioning clusters into varying numbers of
   244	proxies according to camera views. Additionally, it introduces
   245	two strategic contrastive learning loss functions designed to
   246	optimize unsupervised training.
   247	C. Contrastive Learning and Hard Negative Mining
   248	Contrastive learning (CL), a subset of metric learning
   249	techniques, focuses on assessing sample similarities and has
   250	recently gained popularity in unsupervised representation
   251	learning [40], [41], [42]. To improve the robustness of learned
   252	representations, recent methods such as MixIR [41] enforce
   253	invariance by aligning the prediction of mixed augmented
   254	views with aggregated features, encouraging consistency under
   255	stronger intra-instance variations. Key contrastive losses like
   256	InfoNCE [42] are structured non-parametrically, designed to
   257	attract positive samples and repel negative ones. Originally
   258	centered on individual instances, these losses have evolved to
   259	encompass prototypes [43] and proxies [44]. Moreover, CL
   260	has been extensively adopted for both Unsupervised Domain
   261	Adaptation (UDA) [5], [10], [45] and purely unsupervised
   262	person Re-ID tasks [6], [8], [19], [20]. For example, SpCL [5]
   263	integrates a hybrid memory, facilitating contrastive learning
   264	across various levels: source-domain classes, target-domain
   265	clusters, and individual unclustered instances. Meanwhile,
   266	CAP [6] employs a proxy-level memory bank to enable both
   267	intra- and inter-camera contrastive learning at the proxy level,
   268	whereas ICE [19] enhances both SpCL and CAP by incorporating instance-level contrastive learning elements. This
   269	revision avoids direct repetition, restructures sentences for
   270	
   271	2097
   272	
   273	clarity, and refines the description of how contrastive learning
   274	has been applied in different research contexts.
   275	In contrast, this paper not only applies CL in unsupervised
   276	training but also utilizes a hard negative mining (HNM) algorithm for selecting suitable negative samples. A key challenge
   277	in CL is the selection of positive and negative samples for
   278	effective comparison. Despite the introduction of several hard
   279	negative mining strategies [46], this issue continues to be
   280	unresolved in the field. CAP [6] applied this scheme when
   281	computing the inter-camera contrastive learning and proved
   282	the performance of HNM but lacked deep exploration. In
   283	this work, we apply HNM with camera information on two
   284	contrastive loss functions, which demonstrates notable performance in unsupervised training.
   285	D. Attention-Based Person Re-ID
   286	The attention mechanism focuses on identifying salient
   287	regions within images, which is crucial for extracting discriminative information from visual data. Various attention-based
   288	models [47], [48], [49], [50] enhance feature extraction by
   289	prioritizing spatial and channel-wise information that is crucial
   290	for recognizing contextual and detailed aspects within images.
   291	SEA [47] enhances specific feature activations by adaptively reweighting channel weights based on their importance.
   292	CBAM [48] applies attention sequentially across channels and
   293	spatial dimensions to focus on relevant features selectively.
   294	CA [49] encodes positional information to capture long-range
   295	dependencies, enhancing attention across spatial features. GC
   296	[50] utilizes global context to enrich local features, aggregating
   297	comprehensive global information into a compact descriptor.
   298	In person Re-ID field, prior works [27], [51], [52] also
   299	attempt to improve feature extraction by attention mechanism.
   300	Reference [27] integrated SEA and CA to capture attention

exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '1,340p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
     2	
     3	2137
     4	
     5	Part-Based Feature Complementary Denoising for
     6	Unsupervised Person Re-Identification
     7	Qing Tian , Bin Wang, Xiang Liu , Jiashuo Shen, Keyang Cheng , Member, IEEE, Weihua Ou,
     8	and Zhen Lei , Fellow, IEEE
     9	Abstract—Unsupervised Person re-identification (ReID) aims
    10	to automatically capture and match images of the same person across different camera viewpoints without any manual
    11	annotations. Current methods primarily generate pseudo-labels
    12	by clustering global features and employ contrastive learning
    13	strategies for training. Despite the promising advancements made
    14	by these approaches, effectively addressing the inherent bias of
    15	global features and mitigating the impact of pseudo-label noise
    16	remains an unresolved issue. To tackle this challenge, we propose
    17	a part-based features complementary denoising method (PFCD).
    18	Specifically, we design the combined features (CF) module and
    19	the partial-features fusion and contrastive scheme (PFCS), which
    20	capture fine-grained clues from a local perspective and combine
    21	global and local features for clustering with consistent pseudolabel assignment, thereby achieving a complementarity between
    22	global and local features. Furthermore, to diminish the influence
    23	of pseudo-label noise on the model, we design the GMM features
    24	denoising (GFD) module, which employs a Gaussian Mixture
    25	Model to categorize features within each pseudo-class based
    26	on confidence levels and performs denoising on low-confidence
    27	features. Lastly, we construct a modular knowledge distillation (MKD) to enhance feature representation capabilities and
    28	effectively reduce pseudo-label noise. In addition, our method
    29	is confirmed to be effective through extensive experiments on
    30	four challenging ReID datasets, remarkably surpassing numerous
    31	state-of-the-art methods. Code has been made available at https://
    32	github.com/xfltdzzz/PFCS ReID
    33	
    34	Received 17 February 2025; revised 3 June 2025 and 12 July 2025;
    35	accepted 8 September 2025. Date of publication 12 September 2025; date
    36	of current version 5 February 2026. This work was supported in part by the
    37	National Natural Science Foundation of China under Grant 62176128, in part
    38	by the Basic Research Program of Jiangsu under Grant BK20231143, in part
    39	by the Fundamental Research Funds for the Central Universities under Grant
    40	NJ2023032, in part by the Project Funded by the Priority Academic Program
    41	Development of Jiangsu Higher Education Institutions (PAPD) Fund, and in
    42	part by the 333 High-Level Talent Project of Jiangsu Province. This article
    43	was recommended by Associate Editor M. Devanne. (Corresponding author:
    44	Qing Tian.)
    45	Qing Tian is with the School of Software, Nanjing University of Information
    46	Science and Technology, Nanjing 210044, China, also with Wuxi Institute
    47	of Technology, Nanjing University of Information Science and Technology,
    48	Wuxi 214000, China, and also with the MIIT Key Laboratory of Pattern
    49	Analysis and Machine Intelligence, Nanjing University of Aeronautics and
    50	Astronautics, Nanjing 211106, China (e-mail: tianqing@nuist.edu.cn).
    51	Bin Wang, Xiang Liu, and Jiashuo Shen are with the School of
    52	Software, Nanjing University of Information Science and Technology, Nanjing 210044, China (e-mail: wangbin@nuist.edu.cn; liuxiang@nuist.edu.cn;
    53	shenjiashuo@nuist.edu.cn).
    54	Keyang Cheng is with the School of Computer Science and Communication Engineering, Jiangsu University, Zhenjiang 212013, China (e-mail:
    55	kycheng@ujs.edu.cn).
    56	Weihua Ou is with the School of Big Data and Computer Science, Guizhou
    57	Normal University, Guiyang 550025, China (e-mail: ouweihua@gznu.edu.cn).
    58	Zhen Lei is with the National Laboratory of Pattern Recognition, Institute
    59	of Automation, Chinese Academy of Sciences, Beijing 100190, China (e-mail:
    60	zhen.lei@ia.ac.cn).
    61	Digital Object Identifier 10.1109/TCSVT.2025.3609570
    62	
    63	Index Terms—Combined feature, knowledge distillation, contrastive scheme, feature denoising, unsupervised person reidentification.
    64	
    65	I. I NTRODUCTION
    66	
    67	P
    68	
    69	ERSON re-identification (ReID) strives to recognize the
    70	identical pedestrian across monitoring cameras [1]. Currently, ReID is frequently employed in practical scenarios
    71	such as intelligent security and protection and pedestrian
    72	analysis. Nevertheless, ReID faces significant challenges due
    73	to variability in appearance caused by factors such as posture, clothing, occlusion, and changes in body shape [2],
    74	[3], [4]. As a result, researchers have been paying growing
    75	attention to tackling these challenges present in ReID in
    76	recent times.
    77	According to the modeling paradigm of current ReID works,
    78	they can be implemented through both supervised [5], [6], [7]
    79	and unsupervised learning [8], [9] methodologies. While the
    80	former usually surpasses the latter in performance, annotating
    81	data with labels is typically time-consuming and high-cost,
    82	which limits the applicability and scalability of supervised
    83	learning-based methods. To this end, growing attention is paid
    84	to the unsupervised modeling paradigm.
    85	Unsupervised learning-induced ReID can be further grouped
    86	into unsupervised domain adaptation (UDA) ReID [10], [11]
    87	and fully unsupervised learning (USL) ReID [12], [13],
    88	[14]. For UDA ReID methods [9], [15], [16], [17], they
    89	usually involve pre-training a model on a labeled source
    90	domain and subsequently fine-tuning it on an unlabeled target
    91	domain. In contrast, USL ReID methods [18], [19] do not
    92	require supervised annotation knowledge from the domains
    93	while learning directly from unlabeled datasets. Although
    94	the USL ReID approaches enjoy better scalability to large
    95	data applications, their performance seriously relies on the
    96	quality of the predicted pseudo-labels, thus requiring more
    97	sophisticated training strategies to improve the accuracy of
    98	pseudo-labels.
    99	The majority of USL ReID methods [20], [21], [22], [23]
   100	have achieved significant advancements by leveraging clustering algorithms to generate pseudo-labels [24], which are
   101	subsequently utilized to train the models. These approaches
   102	presume that images of the same person are more similar,
   103	making them more likely to be clustered together within the
   104	same cluster. In addition, contrastive learning has shown great
   105	promise in USL ReID tasks, in which the memory bank [25] is
   106	commonly employed in contrastive learning to store previous
   107	feature representations, which is particularly beneficial for
   108	
   109	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
   110	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   111	See https://www.ieee.org/publications/rights/index.html for more information.
   112	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   113	
   114	2138
   115	
   116	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   117	
   118	this domain. By providing a larger pool of negative samples
   119	for comparison, the memory bank enhances the effectiveness
   120	of contrastive learning and consequently alleviates feature
   121	instability issues caused by small batch sizes.
   122	However, despite the great progress of above USL ReID
   123	methods, they still face a critical limitation of discriminating
   124	ability when relying solely on global feature representations. It
   125	is because that compared to local feature representations, the
   126	global representations are inferior in characterizing the person
   127	body details. We empirically analyzed this issue through
   128	evaluations and found that nearly all of related works primarily
   129	depend on global features to capture and thus recognize people
   130	identities. When people of different identities have highly
   131	similar salient features in global visual appearance, relying
   132	solely on global features may tend to incur representationbias and thus result in the ReID classifier failing to distinguish
   133	these identities. Therefore, local features, by providing additional fine-grained discriminative information, can effectively
   134	compensate for the deficiencies of global features. Although a
   135	small number of current works [26], [27] also focus on local
   136	features, they treat these feature representations independently,
   137	for example, by assigning pseudo-labels to these local features
   138	separately or clustering them individually. In this manner, the
   139	complementarities between global and local feature representations are overlooked and, unfortunately, remain unexplored.
   140	To tackle the issues mentioned above, we propose a novel
   141	USL Re-ID framework consisting of three synergistic modules: PFCS, GFD, and MKD, along with the CF module.
   142	The CF module integrates local and global features to capture
   143	both dominant semantic appearances and fine-grained body
   144	details, providing more robust and discriminative combined
   145	features that indirectly enhance the denoising capability of
   146	the GFD module. Building on this, the PFCS module innovates the clustering process by computing separate distance
   147	matrices for global and local features and fusing them by
   148	weights, effectively mitigating the feature bias inherent in
   149	traditional global-only clustering methods. To further address
   150	the pseudo-label noise common in USL Re-ID, we introduce
   151	two complementary denoising mechanisms. The GFD module
   152	applies a Gaussian Mixture Model on the combined features
   153	to estimate confidence levels and suppress noisy samples
   154	within the same pseudo-class. Meanwhile, the MKD module
   155	strengthens early-stage robustness by employing an external
   156	teacher network to guide the student network through knowledge distillation, thereby reducing noise impact and enhancing
   157	representation learning.
   158	It is worth pointing out that although our method is built
   159	upon the contrastive learning framework proposed in CCL
   160	[20], it fundamentally differs from global-only iterative training methods such as RTMem [28] and CCL itself. Specifically,
   161	we specially design fine-grained local features and assign
   162	them to independent memory banks, allowing these features to
   163	actively participate in both the clustering and contrastive learning processes. This joint participation continually enhances
   164	the discriminative capability of model. Moreover, while
   165	PPLR [26] also utilizes local features, its underlying design
   166	philosophy is substantially different. Unlike our approach
   167	which incorporates the designed local features directly into
   168	
   169	clustering, PPLR focuses on improving pseudo-label quality
   170	through intra-image consistency analysis. Notably, PPLR does
   171	not adopt a contrastive learning framework.
   172	Compared with recent denoising-oriented methods such as
   173	DCCT [29] and DKD-MPL [30], our approach also exhibits
   174	distinct differences in both motivation and model-design.
   175	DKD-MPL introduces a tri-view pseudo-labeling scheme and
   176	employs dual knowledge distillation modules to perform selfrefinement via consistency filtering and soft-label supervision.
   177	DCCT, on the other hand, adopts a dual-network co-teaching
   178	framework where pseudo-labels are generated from two separate clustering processes, and training is performed only on
   179	consistent, high-confidence samples shared by both networks.
   180	In contrast, our method introduces two synergistic modules,
   181	GFD and MKD, to jointly mitigate the impact of noisy pseudo
   182	labels. Unlike DKD-MPL and DCCT, our framework neither
   183	relies on generating additional pseudo labels nor requires
   184	multiple clustering processes. Specifically, the GFD module
   185	performs explicit denoising at the feature level by modeling
   186	the distribution of intra-class samples with a Gaussian Mixture
   187	Model. In parallel, the MKD module adopts an independently
   188	designed teacher network that provides consistent and structured guidance throughout the entire training process, thereby
   189	enhancing the stability and robustness of representation
   190	learning.
   191	These design choices collectively improve pseudo-label
   192	quality and enable a more robust training pipeline under fully
   193	unsupervised settings. Based on the above motivations and
   194	designs, our main contributions can be summarized as follows:
   195	• We design a novel feature fusion method. First, during
   196	the feature extraction stage, after the global features are
   197	extracted using a ResNet-based backbone network, the
   198	features are divided and fused using adaptive weights.
   199	In the clustering stage, we fuse the Jaccard distance
   200	matrices of both global and local features with different
   201	weights. This method not only effectively captures the
   202	most prominent clues in the global features but also
   203	leverages the local features to complement the missing
   204	fine-grained information in the global features, achieving
   205	a complementary fusion of global and local features.
   206	• An innovative method for pseudo-labels denoising is
   207	presented, employing Gaussian Mixture Model (GMM)
   208	to identify high-confidence and low-confidence features.
   209	This method leverages Gaussian distribution assumption to denoise low-confidence features within the same
   210	pseudo-class. Additionally, knowledge from the teacher
   211	model is harnessed to further reduce the noise in pseudolabels, ensuring the optimization of the denoising effect.
   212	• Comprehensive experiments conducted on the Market1501, DukeMTMC-ReID, MSMT17, and PersonX
   213	datasets have shown that our method surpasses existing
   214	state-of-the-art USL ReID methods.
   215	II. R ELATED W ORK
   216	A. Unsupervised Person ReID
   217	The current unsupervised person ReID methods can be
   218	broadly classified into two types: Unsupervised Domain
   219	
   220	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   221	
   222	TIAN et al.: PART-BASED FEATURE COMPLEMENTARY DENOISING FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   223	
   224	Adaptation (UDA) ReID [31], [32], [33] and Fully Unsupervised Learning (USL) ReID [18], [34], [35], [36]. UDA ReID
   225	[37], [38] transfers knowledge from the source domain to the
   226	target domain, assuming that the discrepancies between the
   227	two domains are minimal. In contrast, USL ReID is trained
   228	directly on the target dataset, making it more scalable than
   229	UDA ReID. Thus, we focus on USL ReID, as it aligns more
   230	closely with our research objectives.
   231	To fully leverage unlabeled datasets in USL ReID, existing clustering algorithms are usually used to generate
   232	pseudo-labels, such as K-means [39] and DBSCAN [40]. Subsequently, the pseudo-labels generated are used in conjunction
   233	with the unlabeled dataset for iterative training. Therefore, how
   234	to boost the precision of pseudo-labels is the central issue
   235	in USL ReID. Early methods introduced various strategies
   236	to address this challenge. BUC [12] introduced a bottom-up
   237	clustering method that utilized the inherent diversity between
   238	identities and the similarity within each identity to obtain
   239	more discriminative features. PLM [13] presented a progressive learning method based on multi-scale fusion networks,
   240	enabling inference on large datasets without annotations.
   241	MMT [8] learned feature representations through the joint
   242	supervision of offline refined hard pseudo-labels and online
   243	refined soft pseudo-labels, which helped hinder the gradual
   244	accumulation of label noise. ISE [41] proposed an implicit
   245	sample extension method that enhanced the effectiveness of
   246	the clustering module by generating support samples.
   247	Building upon these earlier works, the state-of-the-art USL
   248	ReID methods [9], [30], [42] primarily utilized memorybased contrastive learning frameworks. Specifically, SPCL [9]
   249	introduced a self-paced approach, progressively utilizing more
   250	dependable clustering to train the hybrid memory and learning objectives. CCL [20] introduced a brand-new clustering
   251	contrastive learning approach that was built on a cluster-level
   252	clustering memory bank to solve the problem of inconsistency.
   253	Furthermore, various modules for selection or improvement
   254	have been proposed to boost the accuracy of USL ReID.
   255	For instance, DHCCN [43] improved the performance of
   256	USL ReID by introducing local clues and cross-granularity
   257	consistency constraints. Wu et al. [44] constructed patch proxy
   258	classes as initial supervision, they proposed the method for
   259	assigning pseudo-labels to samples using pairwise gradientguided similarity separation.
   260	In this work, to enhance representation learning and improve
   261	model performance, we focus on generating more precise
   262	pseudo-labels. Specifically, we combine global and local features, and leverage a Gaussian Mixture Model along with
   263	teacher knowledge to reduce pseudo-label noise. The following
   264	parts will discuss these techniques and related works in detail.
   265	B. Part-Based Approaches for Person ReID
   266	Most deep learning-based ReID methods [9], [20], [29],
   267	[36], primarily rely on global features of individuals, which
   268	may lead to the neglect of important information contained
   269	in local features. In recent years, some studies have focused
   270	on leveraging discriminative local features, aiming to improve
   271	retrieval accuracy by incorporating these local parts. Currently,
   272	the most widely used methods for part-based ReID involve
   273	
   274	2139
   275	
   276	directly using predefined stripes. This approach assumes that
   277	person images are vertically aligned, does not require additional pre-trained models, and offers good scalability.
   278	For example, MGN [45], SSG [27], and PPLR [26] have
   279	all utilized local features to further refine the ReID model.
   280	Different from USL ReID works, MGN, which was designed
   281	for supervised ReID, could easily learn more discriminative
   282	feature representations given the presence of labels. Similar
   283	to our approach, SSG and PPLR were both employed for
   284	USL ReID. SSG [27] generated separate pseudo-labels for
   285	each group of global and local features without employing
   286	any fusion strategy, potentially overlooking the consistency
   287	between the different branches. PPLR [26] proposed a cross
   288	agreement score to obtain reliable complementary knowledge
   289	in both global and local features, but exclusively used global
   290	features for clustering, neglecting the impact of local clues
   291	during the clustering process.
   292	In contrast to these methods, our approach enhances global
   293	features by complementing them with local features through
   294	the CF and PFCS modules. We further integrate global and
   295	local features for clustering, ensuring the same pseudo-labels
   296	are assigned to both and utilizing their respective memory
   297	banks to enrich their feature representations. This synergistic
   298	fusion of global and local features allows the model to
   299	better capture fine-grained details while maintaining overall
   300	consistency, thus enhancing its discriminative power.
   301	C. Person ReID With Noisy Labels
   302	Over the past few years, an increasing number of studies
   303	have focused on minimizing the impact of noisy samples
   304	to enhance the reliability of pseudo-labels. Existing methods
   305	primarily aim to filter out the interference from low-confidence
   306	samples to obtain more accurate features, based on the
   307	assumption that low-confidence samples typically incur greater
   308	losses than high-confidence ones. For instance, Arazo et al.
   309	[46] fitted the Beta mixture model to distinguish between highconfidence and low-confidence samples, while DivideMix [47]
   310	used the Gaussian mixture model instead of the Beta mixture
   311	model, offering a more flexible distribution. Additionally,
   312	strategies employing knowledge distillation to reduce label
   313	noise have also been widely studied. MMT [8] proposed a
   314	collaborative training strategy that used the temporal averaging
   315	model of each network to generate soft pseudo-labels for
   316	supervising other networks, aiming to prevent the amplification
   317	of training errors. HDCPD [48] aimed to match the statistical
   318	distributions between the network and the teacher model,
   319	updating parameters via the Exponential Moving Average
   320	(EMA) method. While these methods have made significant
   321	strides in mitigating label noise, substantial label noise may
   322	still remain during the initial training phase, as most models
   323	are initialized with parameters pre-trained on ImageNet. Moreover, NRNet [49] proposed a dual-network framework that
   324	uses networks of different depths (ResNet-50 and ResNet-152)
   325	to extract complementary features and decouples clustering
   326	from training to reduce noise accumulation. While effective for
   327	unsupervised domain adaptation, it relies on a labeled source
   328	domain and is therefore not applicable to fully unsupervised
   329	person re-identification.
   330	
   331	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:05:08 UTC from IEEE Xplore. Restrictions apply.
   332	
   333	2140
   334	
   335	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   336	
   337	Fig. 1. CF Module: By employing adaptive weights to fuse Ftop with Fbot ,
   338	combined features are formed.
   339	TABLE I
   340	T HE N OTATIONS D EFINED IN T HIS PAPER

exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	8598
     2	
     3	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025
     4	
     5	Pose-Skeleton Guided Cross-Attention
     6	Representation Fusion for Occluded Pedestrian
     7	Re-Identification
     8	Shuze Geng , Yifan Liu, Zijin Wang, Gang Yan , Yang Yu , and Yingchun Guo , Member, IEEE
     9	Abstract—Most methods address occluded pedestrian ReIdentification (Re-ID) by employing external auxiliary models
    10	in the feature output stage of the backbone network to locate
    11	visible appearance areas. Nevertheless, these approaches suffer
    12	from issues such as occlusion information diffusion and imprecise
    13	masks generated by external models, indicating the need for
    14	further exploration in the decoupling of pedestrian features
    15	from occlusion information. In light of these challenges, we
    16	propose an innovative algorithm called Pose-Skeleton guided
    17	Cross-attention Representation fusion (PSCR) method. Firstly,
    18	we introduce the Visible Appearance Region Attention (VARA)
    19	model designed to leverage pose information for guiding the
    20	backbone network in effectively distinguishing between occlusion
    21	information and pedestrian features at the intermediate layer. By
    22	employing a suppression strategy, the model is able to effectively
    23	suppress occlusion interference and alleviate the diffusion of
    24	occlusion information. Next, to achieve precise localization of
    25	pedestrian-specific semantic regions, a groundbreaking Skeletal
    26	Area Modeling (SAM) is proposed. Leveraging the principles of
    27	mathematical modeling and capitalizing on the efficacy of human
    28	keypoint confidence, this module generates finely-grained masks
    29	for local skeleton regions and extracts an exhaustive set of local
    30	features. Lastly, under the constraints imposed by spatial attention masks, a cross-attention mechanism is employed to fuse the
    31	features acquired from the previous two steps with local features.
    32	This fusion process results in the generation of enhanced local
    33	features that seamlessly integrate aligning high-level semantic
    34	information. Extensive experimentation demonstrates that the
    35	proposed algorithm exhibits notable performance advancements
    36	when compared to existing methodologies.
    37	Index Terms—Occluded, re-identification, pose-skeleton, crossattention model.
    38	
    39	P
    40	
    41	I. I NTRODUCTION
    42	ERSON re-identification (re-id) aims to identify individuals across different camera views [1]. However,
    43	
    44	Received 3 March 2025; accepted 27 March 2025. Date of publication
    45	31 March 2025; date of current version 8 September 2025. This work was
    46	supported in part by the National Natural Science Foundation of China under
    47	Grant 61806071 and Grant 62102129; in part by Tianjin Municipal Education
    48	Commission Research Plan Project under Grant 2022KJ110; in part by the
    49	Sci-Tech Research Projects of Colleges and Universities in Hebei Province,
    50	China, under Grant QN2019207; in part by the Natural Science Foundation
    51	of Hebei Province (General Program), China, under Grant F2019202381
    52	and Grant 2019202464; and in part by the Key Project of Tianjin University of Technology and Education under Grant KRKC012106. This article
    53	was recommended by Associate Editor Z. Ding. (Corresponding author:
    54	Shuze Geng.)
    55	Shuze Geng and Yifan Liu are with the School of Information Technology
    56	and Engineering, Tianjin University of Technology and Education, Tianjin
    57	300350, China (e-mail: gengshuze@tute.edu.cn).
    58	Zijin Wang, Gang Yan, Yang Yu, and Yingchun Guo are with the School
    59	of Artificial Intelligence, Hebei University of Technology, Tianjin 300401,
    60	China.
    61	Digital Object Identifier 10.1109/TCSVT.2025.3556250
    62	
    63	challenges such as lighting changes, pedestrian occlusion,
    64	and background variations make cross-scene matching highly
    65	difficult [2], [3]. Among these challenges, occlusion stands out
    66	as a key issue with a significant impact [4], [5], [6], [7].
    67	The central problem in pedestrian re-identification under
    68	occlusion is to decouple pedestrian appearance information
    69	from occlusion objects in images and create an effective representation for matching. Current approaches often incorporate
    70	advanced pedestrian pose estimation models or part parsing
    71	models to assist in extracting features from non-occluded
    72	regions of pedestrians. Typical methods include PGFA [8],
    73	PGPAN [9], PVPM [10] and PFD [11], and so on. While these
    74	methods do employ semantic and spatial position information
    75	from pedestrian pose keypoints to achieve pedestrian region
    76	localization and enhance re-identification performance to some
    77	degree [12], they still exhibit two limitations. One major
    78	limitation is that these methods only perform visible feature
    79	extraction at the output of the backbone network, thus failing
    80	to effectively separate occlusion information. As the network
    81	deepens, the receptive field of spatial units in the feature map
    82	expands, causing non-occluded units to cover occluded regions
    83	and transmit occlusion information to originally un-occluded
    84	areas. Another drawback is the adoption of pedestrian feature
    85	extraction techniques, such as mask filtering or semantic alignment, on the feature map generated by the backbone network.
    86	This approach restricts the backbone network’s ability to
    87	perceive and make decisions based on occlusion information.
    88	It selectively updates a subset of the network’s neurons,
    89	impairing the learning capacity of neurons responsible for
    90	occlusion and background information. These issues make
    91	it challenging to decouple pedestrian features and occlusion
    92	information. Additionally, the pedestrian masks generated by
    93	external models are not detailed enough, resulting in some
    94	information loss and occlusion interference. From Fig. 1, it
    95	can be observed that the utilization of conventional approaches
    96	based on pose or part parsing methods fails to effectively
    97	decouple pedestrian regions from occluded regions.
    98	To address the aforementioned issues, this paper proposes
    99	a new Pose-Skeleton guided Cross-attention Representation
   100	fusion (PSCR) algorithm,as shown in Fig. 2. Building upon
   101	the feature fusion paradigm, our proposed method synergistically integrates pose information with the backbone
   102	network and a dedicated local feature extraction network.
   103	This novel integration empowers the network with enhanced
   104	perceptual capabilities and bolstered decision-making prowess
   105	specifically tailored for the challenge of decoupling pedestrian
   106	
   107	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
   108	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   109	See https://www.ieee.org/publications/rights/index.html for more information.
   110	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   111	
   112	GENG et al.: POSE-SKELETON GUIDED CROSS-ATTENTION REPRESENTATION FUSION
   113	
   114	8599
   115	
   116	work features and local features. This approach not only
   117	enhances discriminative local features but also aligns
   118	high-level semantic features.
   119	The remainder of this paper is structured as follows: Section II
   120	provides a comprehensive review of related works in the field.
   121	In Section III, we present the architecture of the proposed
   122	framework and elaborate on the implementation details. Experimental results and a thorough analysis of the effectiveness of
   123	our method are presented in Section IV. Finally, Section V
   124	concludes our work by summarizing the key findings and
   125	contributions.
   126	II. R ELATED W ORK
   127	Fig. 1. Illustration of the Original Image (a) and pedestrian attention regions
   128	obtained using traditional pose or parsing-based methods (b), as well as our
   129	method (c).
   130	
   131	features and occlusion information. Specifically, an innovative
   132	Visible Appearance Region Attention (VARA) module is constructed, employing a multi-layer pose information soft fusion
   133	technique to facilitate the intermediate layer of the backbone
   134	network in effectively learning occlusion information. This
   135	approach attempts to resolve the issue of occlusion information
   136	diffusion, employing a suppression method instead of relying
   137	solely on pose estimation to filter high-level features [8],
   138	[12], [13], [14]. Furthermore, a pioneering Skeletal Area
   139	Modeling (SAM) technique is developed, leveraging human
   140	keypoint information to generate pre-defined joint masks in
   141	a mathematically modeled manner. By comparison, these
   142	masks outperform those generated by semantic segmentation
   143	[15], [16], [17] or pose points since they provide pixellevel visibility confidence and exhibit greater stability when
   144	confronted with low-resolution pedestrian images. Lastly, a
   145	Skeleton Modulated Cross-Attention Fusion (SMCF) module
   146	is established to explore the interplay between local features
   147	derived from the pedestrian skeleton and features outputted by
   148	the backbone network. Through the fusion of local contextual
   149	information, this module effectively learns pedestrian retrieval
   150	features, enabling enhanced retrieval performance.
   151	The main contributions of our work can be summarized as
   152	follows:
   153	• Firstly, VARA module is proposed that is organically
   154	integrated with the backbone network, which guides
   155	the backbone network to perceive pedestrian appearance
   156	information through the foreground-perceiving channel
   157	attention mechanism (pedestrian area) and the crossattention fusion mechanism of key feature spaces, and
   158	improves the model’s perception and discrimination abilities for occlusion information.
   159	• Secondly, the SAM technique is developed to construct
   160	a comprehensive set of local features by capturing local
   161	skeleton regions of the human body. This compensates for
   162	potential information loss in appearance regions between
   163	key points extracted from the heat map generated by the
   164	pose estimation model.
   165	• Thirdly, the SMCF module utilizes the local region
   166	weight map provided by the joint mask to constrain
   167	the cross-attention fusion process between backbone net-
   168	
   169	Currently, existing occluded pedestrian re-identification
   170	methods are primarily categorized into two research
   171	paradigms: methods utilizing external pedestrian models (e.g.,
   172	pose estimation and semantic segmentation) [5], [15], [16],
   173	[17], [18], [19], and approaches leveraging Vision Transformer
   174	(ViT) architectures [20], [21].
   175	A. Pedestrian Re-Identification Method Based on External
   176	Semantic Clues
   177	Currently, methods that employ external semantic clues have
   178	gained prominence in the field of re-ID. These approaches
   179	often utilize external pedestrian models, such as pedestrian
   180	parsing, pose estimation, or foreground segmentation (pedestrian area), to locate the visible area of pedestrians [22],
   181	[23], [24], [25], [26], [27]. For instance, in [8], a posture
   182	information branch is added to the PCB network. The heat
   183	maps of key points are treated as masks for the pedestrian
   184	key areas, which are then applied to the output feature map
   185	of the re-identification network to extract the pedestrian’s
   186	appearance features. Similarly, to emphasize the local regions
   187	of pedestrians, study [11] introduces a Pose-guided Feature
   188	Disentangling (PFD) method based on transformers. It leverages pose information to disentangle semantic components
   189	and selectively match non-occluded parts. To obtain complete
   190	pedestrian features under occlusion, Hou et al. [24] simultaneously introduce a foreground extractor and a pose estimation
   191	model. The pose estimation result guides the structural segmentation of the feature map, while the foreground extractor
   192	filters out occlusion information. Spatial correlation between
   193	local features is then used to reconstruct the appearance
   194	features lost in the occluded areas. With a similar motivation,
   195	Human Co-Parsing Guided Alignment (HCGA) entails the
   196	amalgamation of a co-parsing network and a guided alignment
   197	module, leading to the extraction of crucial foreground and
   198	human part features for occluded person ReID [25].
   199	The aforementioned approaches have partially addressed the
   200	issue of occlusion on re-identification models. However, these
   201	methods typically incorporate modules post feature extraction
   202	by the backbone network (e.g., CNN or Transformer). This
   203	approach presents a challenge as occlusion information may
   204	spread to previously un-occluded areas due to the expanding
   205	pixel receptive field in the backbone network. Consequently,
   206	the higher or output layers struggle to filter out the diffused
   207	interference information. In contrast, our proposed method
   208	
   209	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:55 UTC from IEEE Xplore. Restrictions apply.
   210	
   211	8600
   212	
   213	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 9, SEPTEMBER 2025
   214	
   215	Fig. 2. The overall architecture of the proposed PSCR network. Our method contains VARA module(Section III-B), SAM module (Section III-C), SMCF
   216	module (Section III-D). Module VARA: Focusing on visible light area and suppressing occlusion interference in the backbone; Module SAM: Achieving more
   217	precise localization and extracting local features of pedestrians; Module SMCF: Incorporating Cross-Attention Mechanism to dynamically capture semantic
   218	∗ and the global feature f obtained from
   219	correlations between features for local feature alignment and fusion. During matching stage, the local features fout
   220	g
   221	the backbone are concatenated together for occluded re-id.
   222	
   223	introduces the VARA module and leverages multi-layer pose
   224	information soft fusion to guide the backbone network in
   225	learning occlusion information at intermediate layers. Moreover, we introduce a novel SAM module to address the loss
   226	of pedestrian appearance information in the pose heatmap.
   227	This effectively mitigates the decoupling problem between
   228	pedestrian information and occlusion information within our
   229	method, elevating its performance.
   230	B. Methods Based on Transformer
   231	The initial introduction of Transformer in the Re-ID field
   232	was proposed by [20] as TransReID. However, TransReID
   233	primarily focuses on global feature learning, neglecting the
   234	correlation between local features. Thus, in literature [21], a
   235	spatial slice strategy is proposed to incorporate the spatiality
   236	of input images into the image patch sequence, enhancing the
   237	short-distance correlation among patches. Similarly, Lu [28]
   238	presents a dual-branch Transformer network that operates in an
   239	end-to-end manner. The first branch is a global branch based
   240	on transformers, responsible for extracting global features.
   241	The second branch, known as the local branch, utilizes the
   242	Selective Token Attention (STA) module, employing multiheaded self-attention to efficiently extract unique tokens for
   243	obtaining local features.
   244	Moreover, an auto-aligned Transformer model is proposed in [29] which avoids the use of a fixed partition to
   245	acquire local features. Instead, it utilizes clustering for patch
   246	allocation and introduces part encoding to simultaneously
   247	adaptively learn and align local features. However, despite the
   248	discriminative power of Transformers, they lack the desired
   249	
   250	properties of translation, scale, distortion invariance, as well
   251	as the hierarchical structures found in CNN convolution
   252	operations. Addressing this limitation, [30] combines a CNN
   253	backbone to extract hierarchical features and harnesses the
   254	global interaction characteristics of Transformers to aggregate features of different scales from a global perspective.
   255	Another notable approach, PAT [31], employs CNNs as a
   256	backbone model and incorporates a transformer encoder to
   257	extract additional features. It utilizes cross-attention with a
   258	group of learnable part prototypes to selectively mask specific
   259	features. Similarly, our proposed method also leverages the
   260	fusion of CNN and transformer encoder. However, what sets

codex
我按每篇的 `method/approach` 和必要的实验消融段读，不按摘要概括。下面只拆“创新是怎么构造出来的”。

**1. Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification**

1. **触发观察**：已有 VI-ReID 大多默认行人完整可见；已有遮挡 ReID 又多是单模态，做法是定位可见区域、丢掉遮挡区域。作者抓住的现象是：在 VI-ReID 里，遮挡不只是少了一块可见信息，还会让可见光和红外的缺失位置、退化形式不同，导致跨模态对齐更坏。

2. **重定义动作**：他们把问题从“如何关注可见区域”改写成“如何补全缺失身份线索”。关键词是 `completion perspective rather than discarding`、`cross-modality feature completion`、`token-level completion`、`region-level completion`、`multi-distribution consistency`。这一步很关键，它把旧问题里的“遮挡鲁棒特征提取”升级成“遮挡 VI-ReID 的跨模态补全”。

3. **机制怎么长出来**：机制基本顺着重定义推出。既然是缺失补全，就先在单模态内部用语义图引导 token 互相补，再在区域级从另一模态软查询互补信息，最后用 MCA 让 token 补全、region 补全和伪整体分布一致。GFCT、SRCM、MCA 不是孤立模块，逻辑链是“局部缺失、跨模态不一致、多路径补全需一致”。

4. **证据闭环**：关键证据不是单个 SOTA 表，而是模块组合消融。GFCT、SRCM、MCA 单独加都涨，GFCT+SRCM 比单独更好，完整组合最好；MCA 又和直接互相对齐对比，证明“对齐到伪整体参考”比简单两路互拉更合理。另一个闭环是他们构造了语义遮挡数据集，并和随机擦除对比，说明他们讨论的是更真实的遮挡，而不是普通增强。

5. **reviewer 为什么买账**：买账点主要是视角和任务定义，不是图注意力或 KL 本身多新。它把“遮挡 VI-ReID”讲成一个尚未充分处理的交叉问题：缺失身份线索和跨模态错位同时发生。因此模块虽然常见，但被一个清楚的问题链条串住了。

**2. Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions**

1. **触发观察**：传统复杂光照 ReID 要么学光照不变特征，要么把图片统一到标准光照。作者先做了一个很具体的观察：同一图像对在不同光照下距离不同，而且每个图像对都有自己的最小距离光照；最优光照不一定是正常光照。

2. **重定义动作**：他们把光照从“要消除的干扰”改写成“检索距离里的可选择条件”。关键词是 `optimal illumination`、`optimal illumination distance`、`distance replacement`、`each image pair has a unique optimal lighting`。旧问题是做光照归一化，新问题是为每个图像对选择最能表达语义距离的光照状态。

3. **机制怎么长出来**：机制非常直接。先用 GAN 把图库归一到正常光照，再用一组预定义光照生成器产生多种光照版本；对每种光照都提特征并算距离矩阵，最后在距离矩阵通道上取最小值作为最终检索距离。这个方法简单，但和观察绑定很紧：既然每对图像存在最小距离，就枚举光照并取最小距离。

4. **证据闭环**：最关键的是三类对比：随机替换、部分随机替换、固定光照替换，都不如最优光照选择。作者还统计 replacement 和 effective replacement，承认并分析“并非每次替换都有效”，这反而让论证更可信。图像对距离随光照变化的分析图，是支撑重定义的第一证据。

5. **reviewer 为什么买账**：这篇卖的是视角，不是模型复杂度。它的 novelty 来自把“统一光照”反过来讲成“统一光照可能丢失最优语义距离”，然后给出一个可验证的距离替换框架。弱点也明显：枚举预定义光照很粗，但观察足够具体。

**3. Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification**

1. **触发观察**：无监督 ReID 的代理对比学习依赖伪标签和 camera-aware proxy，但正负样本选错会严重伤训练。作者抓住两个反直觉点：对应 proxy 太容易，优化信号弱；所有同簇 proxy 都当正样本又太杂，会让方向不清。另一个观察是，同相机 proxy 往往背景和视角相似，会干扰假负样本挖掘。

2. **重定义动作**：他们把问题从“如何生成更好伪标签”改成“如何选择最有训练价值的 proxy 对”。关键词是 `optimal proxy mining`、`hardest camera proxy`、`false negative proxies mining`、`misaligned contrastive pairs`、`camera-constrained hard negative mining`、`anti-noise ranking`。这让创新点落在训练信号选择，而不是聚类算法本身。

3. **机制怎么长出来**：HCPM 假设当前伪标签基本可信，从同簇里找和 query 最不相似的相机 proxy 当 hardest positive，再从这个 hardest camera 里找 hard negatives，控制背景和视角因素。FNPM 则承认伪标签有噪声，先排除同相机 proxy，再按相似度排序挖潜在假负样本，并用同 proxy 多 query 排名求和做抗噪。PKGC 是额外的特征增强，用多尺度全局上下文让网络看身份相关区域。

4. **证据闭环**：关键消融是正样本选择策略对比：corresponding proxy、all proxies、hardest positive proxy，以及是否使用 CCHNM。FNPM 部分又验证了排除同相机 proxy 和 anti-noise ranking 的作用。可视化包括 t-SNE、Top-10 检索和 Grad-CAM，用来证明不是只涨点，而是类内更紧、类间更开、注意力更落在人身上。

5. **reviewer 为什么买账**：真正买账的是“proxy 不是越多越好，也不是越近越好，而是要按相机和噪声结构选最有信息的对比对象”。PKGC 相对像常规增强，novelty 主要来自 proxy mining 视角。它把无监督 ReID 的噪声问题讲成了“对比对构造错误”的问题。

**4. Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification**

1. **触发观察**：USL ReID 大多用全局特征聚类生成伪标签，但全局特征会偏向最显眼外观，遇到衣服颜色相似、局部细节不同的人容易混。已有局部特征方法又常把局部特征单独聚类或单独监督，忽略全局和局部的互补。作者还指出一个更细的问题：伪类别内部也有噪声，会把类中心拉偏。

2. **重定义动作**：他们把问题从“伪标签有噪声”改成“伪标签噪声来自全局特征偏置和伪类内部低置信特征”。关键词是 `part-based feature complementary denoising`、`global-local complementarity`、`within pseudo-class noise`、`low-confidence feature denoising`、`modular knowledge distillation`。

3. **机制怎么长出来**：如果全局有偏，就把上半身、下半身局部特征和全局特征加权融合，并把全局、局部 Jaccard 距离融合后再聚类，保证伪标签来自互补视角。若伪类内部有低置信噪声，就用 GMM 根据识别损失估计置信度，再用高置信样本估计高斯先验，对低置信特征做特征级去噪。MKD 则用训练好的 teacher 带 student，缓解早期伪标签不稳。

4. **证据闭环**：它的闭环比较完整。PFCS、GFD、MKD 分别消融，三者组合最好；t-SNE 展示 baseline、去掉 MKD、去掉 GFD、完整模型的聚类紧致度差异；Top-5 检索可视化显示错误匹配减少。阈值 δ 的实验也重要，说明“只去噪低置信样本”这个定义是对的，扩大去噪范围反而会掉。

5. **reviewer 为什么买账**：这篇卖的是机制和训练流程，但背后的视角也清楚：局部特征不是另一个分支，而是提高伪标签质量和去噪可靠性的补充证据。它不是发明一个新零件，而是把“局部互补、类内去噪、教师稳定”包装成一个一致的无监督噪声治理框架。

**5. Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification**

1. **触发观察**：现有遮挡 ReID 常在 backbone 输出后用姿态或解析模型找可见区域。作者抓住一个更深的失败点：到输出层时感受野已经扩大，遮挡信息已经扩散到原本未遮挡区域，后处理式过滤太晚了。同时，关键点热图只覆盖点附近，关键点之间的身体区域会丢信息。

2. **重定义动作**：他们把问题从“如何在输出特征上找可见区域”改成“如何在中间层阻断遮挡信息扩散，并用骨架区域补足关键点之间的外观”。关键词是 `occlusion information diffusion`、`decouple pedestrian features and occlusion information`、`visible appearance region attention`、`skeletal area modeling`、`skeleton-modulated cross-attention`。

3. **机制怎么长出来**：VARA 插在 ResNet 中间层，用姿态热图做前景通道注意力和空间交叉注意力，让 backbone 早一点学习遮挡和行人的区别。SAM 不直接拿关键点热图当 mask，而是根据两端关键点坐标和置信度，把手臂、腿、躯干等 14 个骨架区域建成软 mask。SMCF 再让骨架局部特征作为 query，与 backbone 特征做受 mask 约束的交叉注意力融合。

4. **证据闭环**：最关键消融是 VARA、SAM、SMCF 分开加。VARA 证明中间层抑制有效，SAM 证明骨架区域比点热图更完整。进一步分析 VARA 插入位置，发现中层最好，这正好支撑“输出层太晚、低层语义不够、高层分辨率太低”的重定义。SAM 的 α、γ、T 可视化和不同遮挡程度可视化，也在证明骨架区域建模确实覆盖可见身体而抑制遮挡。

5. **reviewer 为什么买账**：这篇真正卖的是“遮挡信息扩散”这个视角。姿态、mask、cross-attention 都不新，但“为什么不能只在输出层用姿态过滤”说得清楚，机制也自然落到中间层和骨架区域建模上。它的可视化证据和参数分析让这个视角比较容易被接受。

**6. Privacy preserving person re-identification via anonymizing diffusion model**

1. **触发观察**：公开 ReID 数据集有隐私风险，已有模糊、加噪、加密、对抗扰动方法多是在原图上“隐藏身份”，并没有消除真实身份关联，人或模型仍可能找回原 ID。纯合成数据隐私好，但和真实 ReID 域差距大；只靠噪声生成又难保证同一合成 ID 的一致性。

2. **重定义动作**：他们把隐私保护从“改造真实图片”改成“用新合成 ID 替换真实 ID，同时保留 ReID 训练效用”。关键词是 `new IDs`、`anonymizing diffusion model`、`privacy-compliant benchmarks`、`ID separation degree`、`privacy-utility trade-off`、`eliminate at source`。这一步把任务从图像脱敏变成了数据集替代。

3. **机制怎么长出来**：既然要新 ID 又要可训练，就用文本控制身份属性，用扩散模型生成。Stage 1 在全数据上微调 Stable Diffusion 学 ReID 风格；Stage 2 在小子集上微调，限制属性重叠，让 prompt 更稳定地绑定视觉模式；class embedding 提高同一合成 ID 内一致性；prompt griddle 重组头发、上装、下装属性生成未出现的新 prompt；ADM griddle 过滤每个 ID 的离群样本。ISD 则量化合成 ID 和真实 ID 的分离程度。

4. **证据闭环**：证据不是只看 ReID 准确率，而是同时看 utility 和 privacy。`Tg/Tu` 证明生成数据还能训练和区分，`Tp` 和攻击检索 demo 证明真实 query 难以找回对应生成 ID，ISD 和余弦距离分布证明生成特征远离真实 ID 中心。消融则验证两阶段微调、class embedding、griddle 过滤分别服务于风格、ID 一致性和样本稳定性。

5. **reviewer 为什么买账**：这篇卖的是视角和基准范式，不是 Stable Diffusion 本身。它把隐私 ReID 从“降低可识别性”改成“替换身份实体”，并且提出 ISD 这种能讨论隐私强度的指标。reviewer 买的是问题定义、数据集替代价值和隐私-效用闭环。

**共同套路**

这批论文的共同套路是：先找一个旧范式默认成立的假设，然后证明它在某个更真实场景里不成立。比如“遮挡就看可见区域”“光照应统一”“proxy 越近越正”“全局特征能代表伪类”“输出层再处理遮挡也来得及”“匿名就是改原图”。

能发的关键不是模块堆得多，而是把观察改写成一个新变量：补全路径、最优光照、最优代理、全局-局部互补、遮挡扩散、新 ID 替代。机制只要顺着这个新变量自然长出来，再用一组能直接反驳旧假设的消融或可视化闭环，reviewer 就更容易买账。
tokens used
195,517
我按每篇的 `method/approach` 和必要的实验消融段读，不按摘要概括。下面只拆“创新是怎么构造出来的”。

**1. Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification**

1. **触发观察**：已有 VI-ReID 大多默认行人完整可见；已有遮挡 ReID 又多是单模态，做法是定位可见区域、丢掉遮挡区域。作者抓住的现象是：在 VI-ReID 里，遮挡不只是少了一块可见信息，还会让可见光和红外的缺失位置、退化形式不同，导致跨模态对齐更坏。

2. **重定义动作**：他们把问题从“如何关注可见区域”改写成“如何补全缺失身份线索”。关键词是 `completion perspective rather than discarding`、`cross-modality feature completion`、`token-level completion`、`region-level completion`、`multi-distribution consistency`。这一步很关键，它把旧问题里的“遮挡鲁棒特征提取”升级成“遮挡 VI-ReID 的跨模态补全”。

3. **机制怎么长出来**：机制基本顺着重定义推出。既然是缺失补全，就先在单模态内部用语义图引导 token 互相补，再在区域级从另一模态软查询互补信息，最后用 MCA 让 token 补全、region 补全和伪整体分布一致。GFCT、SRCM、MCA 不是孤立模块，逻辑链是“局部缺失、跨模态不一致、多路径补全需一致”。

4. **证据闭环**：关键证据不是单个 SOTA 表，而是模块组合消融。GFCT、SRCM、MCA 单独加都涨，GFCT+SRCM 比单独更好，完整组合最好；MCA 又和直接互相对齐对比，证明“对齐到伪整体参考”比简单两路互拉更合理。另一个闭环是他们构造了语义遮挡数据集，并和随机擦除对比，说明他们讨论的是更真实的遮挡，而不是普通增强。

5. **reviewer 为什么买账**：买账点主要是视角和任务定义，不是图注意力或 KL 本身多新。它把“遮挡 VI-ReID”讲成一个尚未充分处理的交叉问题：缺失身份线索和跨模态错位同时发生。因此模块虽然常见，但被一个清楚的问题链条串住了。

**2. Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions**

1. **触发观察**：传统复杂光照 ReID 要么学光照不变特征，要么把图片统一到标准光照。作者先做了一个很具体的观察：同一图像对在不同光照下距离不同，而且每个图像对都有自己的最小距离光照；最优光照不一定是正常光照。

2. **重定义动作**：他们把光照从“要消除的干扰”改写成“检索距离里的可选择条件”。关键词是 `optimal illumination`、`optimal illumination distance`、`distance replacement`、`each image pair has a unique optimal lighting`。旧问题是做光照归一化，新问题是为每个图像对选择最能表达语义距离的光照状态。

3. **机制怎么长出来**：机制非常直接。先用 GAN 把图库归一到正常光照，再用一组预定义光照生成器产生多种光照版本；对每种光照都提特征并算距离矩阵，最后在距离矩阵通道上取最小值作为最终检索距离。这个方法简单，但和观察绑定很紧：既然每对图像存在最小距离，就枚举光照并取最小距离。

4. **证据闭环**：最关键的是三类对比：随机替换、部分随机替换、固定光照替换，都不如最优光照选择。作者还统计 replacement 和 effective replacement，承认并分析“并非每次替换都有效”，这反而让论证更可信。图像对距离随光照变化的分析图，是支撑重定义的第一证据。

5. **reviewer 为什么买账**：这篇卖的是视角，不是模型复杂度。它的 novelty 来自把“统一光照”反过来讲成“统一光照可能丢失最优语义距离”，然后给出一个可验证的距离替换框架。弱点也明显：枚举预定义光照很粗，但观察足够具体。

**3. Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification**

1. **触发观察**：无监督 ReID 的代理对比学习依赖伪标签和 camera-aware proxy，但正负样本选错会严重伤训练。作者抓住两个反直觉点：对应 proxy 太容易，优化信号弱；所有同簇 proxy 都当正样本又太杂，会让方向不清。另一个观察是，同相机 proxy 往往背景和视角相似，会干扰假负样本挖掘。

2. **重定义动作**：他们把问题从“如何生成更好伪标签”改成“如何选择最有训练价值的 proxy 对”。关键词是 `optimal proxy mining`、`hardest camera proxy`、`false negative proxies mining`、`misaligned contrastive pairs`、`camera-constrained hard negative mining`、`anti-noise ranking`。这让创新点落在训练信号选择，而不是聚类算法本身。

3. **机制怎么长出来**：HCPM 假设当前伪标签基本可信，从同簇里找和 query 最不相似的相机 proxy 当 hardest positive，再从这个 hardest camera 里找 hard negatives，控制背景和视角因素。FNPM 则承认伪标签有噪声，先排除同相机 proxy，再按相似度排序挖潜在假负样本，并用同 proxy 多 query 排名求和做抗噪。PKGC 是额外的特征增强，用多尺度全局上下文让网络看身份相关区域。

4. **证据闭环**：关键消融是正样本选择策略对比：corresponding proxy、all proxies、hardest positive proxy，以及是否使用 CCHNM。FNPM 部分又验证了排除同相机 proxy 和 anti-noise ranking 的作用。可视化包括 t-SNE、Top-10 检索和 Grad-CAM，用来证明不是只涨点，而是类内更紧、类间更开、注意力更落在人身上。

5. **reviewer 为什么买账**：真正买账的是“proxy 不是越多越好，也不是越近越好，而是要按相机和噪声结构选最有信息的对比对象”。PKGC 相对像常规增强，novelty 主要来自 proxy mining 视角。它把无监督 ReID 的噪声问题讲成了“对比对构造错误”的问题。

**4. Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification**

1. **触发观察**：USL ReID 大多用全局特征聚类生成伪标签，但全局特征会偏向最显眼外观，遇到衣服颜色相似、局部细节不同的人容易混。已有局部特征方法又常把局部特征单独聚类或单独监督，忽略全局和局部的互补。作者还指出一个更细的问题：伪类别内部也有噪声，会把类中心拉偏。

2. **重定义动作**：他们把问题从“伪标签有噪声”改成“伪标签噪声来自全局特征偏置和伪类内部低置信特征”。关键词是 `part-based feature complementary denoising`、`global-local complementarity`、`within pseudo-class noise`、`low-confidence feature denoising`、`modular knowledge distillation`。

3. **机制怎么长出来**：如果全局有偏，就把上半身、下半身局部特征和全局特征加权融合，并把全局、局部 Jaccard 距离融合后再聚类，保证伪标签来自互补视角。若伪类内部有低置信噪声，就用 GMM 根据识别损失估计置信度，再用高置信样本估计高斯先验，对低置信特征做特征级去噪。MKD 则用训练好的 teacher 带 student，缓解早期伪标签不稳。

4. **证据闭环**：它的闭环比较完整。PFCS、GFD、MKD 分别消融，三者组合最好；t-SNE 展示 baseline、去掉 MKD、去掉 GFD、完整模型的聚类紧致度差异；Top-5 检索可视化显示错误匹配减少。阈值 δ 的实验也重要，说明“只去噪低置信样本”这个定义是对的，扩大去噪范围反而会掉。

5. **reviewer 为什么买账**：这篇卖的是机制和训练流程，但背后的视角也清楚：局部特征不是另一个分支，而是提高伪标签质量和去噪可靠性的补充证据。它不是发明一个新零件，而是把“局部互补、类内去噪、教师稳定”包装成一个一致的无监督噪声治理框架。

**5. Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification**

1. **触发观察**：现有遮挡 ReID 常在 backbone 输出后用姿态或解析模型找可见区域。作者抓住一个更深的失败点：到输出层时感受野已经扩大，遮挡信息已经扩散到原本未遮挡区域，后处理式过滤太晚了。同时，关键点热图只覆盖点附近，关键点之间的身体区域会丢信息。

2. **重定义动作**：他们把问题从“如何在输出特征上找可见区域”改成“如何在中间层阻断遮挡信息扩散，并用骨架区域补足关键点之间的外观”。关键词是 `occlusion information diffusion`、`decouple pedestrian features and occlusion information`、`visible appearance region attention`、`skeletal area modeling`、`skeleton-modulated cross-attention`。

3. **机制怎么长出来**：VARA 插在 ResNet 中间层，用姿态热图做前景通道注意力和空间交叉注意力，让 backbone 早一点学习遮挡和行人的区别。SAM 不直接拿关键点热图当 mask，而是根据两端关键点坐标和置信度，把手臂、腿、躯干等 14 个骨架区域建成软 mask。SMCF 再让骨架局部特征作为 query，与 backbone 特征做受 mask 约束的交叉注意力融合。

4. **证据闭环**：最关键消融是 VARA、SAM、SMCF 分开加。VARA 证明中间层抑制有效，SAM 证明骨架区域比点热图更完整。进一步分析 VARA 插入位置，发现中层最好，这正好支撑“输出层太晚、低层语义不够、高层分辨率太低”的重定义。SAM 的 α、γ、T 可视化和不同遮挡程度可视化，也在证明骨架区域建模确实覆盖可见身体而抑制遮挡。

5. **reviewer 为什么买账**：这篇真正卖的是“遮挡信息扩散”这个视角。姿态、mask、cross-attention 都不新，但“为什么不能只在输出层用姿态过滤”说得清楚，机制也自然落到中间层和骨架区域建模上。它的可视化证据和参数分析让这个视角比较容易被接受。

**6. Privacy preserving person re-identification via anonymizing diffusion model**

1. **触发观察**：公开 ReID 数据集有隐私风险，已有模糊、加噪、加密、对抗扰动方法多是在原图上“隐藏身份”，并没有消除真实身份关联，人或模型仍可能找回原 ID。纯合成数据隐私好，但和真实 ReID 域差距大；只靠噪声生成又难保证同一合成 ID 的一致性。

2. **重定义动作**：他们把隐私保护从“改造真实图片”改成“用新合成 ID 替换真实 ID，同时保留 ReID 训练效用”。关键词是 `new IDs`、`anonymizing diffusion model`、`privacy-compliant benchmarks`、`ID separation degree`、`privacy-utility trade-off`、`eliminate at source`。这一步把任务从图像脱敏变成了数据集替代。

3. **机制怎么长出来**：既然要新 ID 又要可训练，就用文本控制身份属性，用扩散模型生成。Stage 1 在全数据上微调 Stable Diffusion 学 ReID 风格；Stage 2 在小子集上微调，限制属性重叠，让 prompt 更稳定地绑定视觉模式；class embedding 提高同一合成 ID 内一致性；prompt griddle 重组头发、上装、下装属性生成未出现的新 prompt；ADM griddle 过滤每个 ID 的离群样本。ISD 则量化合成 ID 和真实 ID 的分离程度。

4. **证据闭环**：证据不是只看 ReID 准确率，而是同时看 utility 和 privacy。`Tg/Tu` 证明生成数据还能训练和区分，`Tp` 和攻击检索 demo 证明真实 query 难以找回对应生成 ID，ISD 和余弦距离分布证明生成特征远离真实 ID 中心。消融则验证两阶段微调、class embedding、griddle 过滤分别服务于风格、ID 一致性和样本稳定性。

5. **reviewer 为什么买账**：这篇卖的是视角和基准范式，不是 Stable Diffusion 本身。它把隐私 ReID 从“降低可识别性”改成“替换身份实体”，并且提出 ISD 这种能讨论隐私强度的指标。reviewer 买的是问题定义、数据集替代价值和隐私-效用闭环。

**共同套路**

这批论文的共同套路是：先找一个旧范式默认成立的假设，然后证明它在某个更真实场景里不成立。比如“遮挡就看可见区域”“光照应统一”“proxy 越近越正”“全局特征能代表伪类”“输出层再处理遮挡也来得及”“匿名就是改原图”。

能发的关键不是模块堆得多，而是把观察改写成一个新变量：补全路径、最优光照、最优代理、全局-局部互补、遮挡扩散、新 ID 替代。机制只要顺着这个新变量自然长出来，再用一组能直接反驳旧假设的消融或可视化闭环，reviewer 就更容易买账。
