Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef97f-1309-7cf1-8d2b-9a69b55991a8
--------
user
我们要做 ReID **效率**方法 Rank-Instability Adaptive ReID。idea: ReID 默认所有 query 用 full 网络推理(uniform compute); 应 **rank-stability conditioned**——用 cheap(早层/低分辨率)特征算的 top-k 排名 vs full 排名的**不一致度(rank instability)** 当路由信号: 稳定 query 早退(cheap 够), 不稳定走 full。隐藏变量 = cheap-vs-full rank disagreement(不是一个静态 difficulty score)。效率-精度 Pareto。

联网查确切先例 + 撞车, 逐类判它们是否已用 **cheap-vs-full rank disagreement** 当路由信号(而非静态 difficulty/置信度): (a)CFPER / query-difficulty-adaptive compute ReID; (b)early-exit / dynamic-inference / adaptive-depth / multi-exit person ReID; (c)cascade / coarse-to-fine / hierarchical retrieval ReID; (d)anytime ReID; (e)hardness/uncertainty-aware ReID; (f)通用 retrieval 的 adaptive re-ranking budget。**关键问: rank-instability(cheap-full 排名不一致)当信号, 相对'又一个 difficulty-adaptive(CFPER)'有没有独立新意? 还是会被归类成置信度/难度路由?** 撞了说撞哪+杀伤; 没撞说切口。诚实, verdict(撞车/存活)+证据链接+信心1-10。
codex
我先按仓库接手要求快速读当前实验文档，再做联网文献/代码检索。这个任务核心是判“cheap-vs-full 排名不一致”是否已被 ReID 或检索效率工作拿来做路由信号，所以我会按你列的类别逐类取证。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/innovation_brainstorm.md: No such file or directory

 exited 1 in 0ms:
sed: experiments/decisions.md: No such file or directory

 exited 1 in 0ms:
sed: experiments/results.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/paper_materials/story.md: No such file or directory

codex
当前工作目录下没有 `experiments/...` 这一层，说明我们可能已经在某个子实验目录或文献目录里。我先定位仓库结构和这些文档的真实位置，然后继续检索。
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2

exec
/bin/zsh -lc ls in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
B_CONTAINMENT_DESIGN.md
GOPL_KILLSWITCH_DESIGN.md
HUBNESS_ANALYSIS.md
HUBNESS_KILLSWITCH_DESIGN.md
OSAC_KILLSWITCH_DESIGN.md
SYNTHESIS_METHODOLOGY.md
all_papers.txt
analyses
batches
cvpb_containment_full.log
cvpb_containment_killswitch_design.md
debate
hubness_debate.sh
launch_30deep.sh
novelty_hubness.md
novelty_osac.md
novelty_rankinstab.md
ondisk_pivot.md
pivot
reassess
reassess.sh
reassess2
reassess2.sh
reviews
strategic_pivot.sh
validate
validate_candidates.sh
video_feasibility.md

exec
/bin/zsh -lc "rg --files -g 'results.md' -g 'decisions.md' -g 'innovation_brainstorm.md' -g 'story.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "find .. -maxdepth 4 \\( -name results.md -o -name decisions.md -o -name innovation_brainstorm.md -o -name story.md \\)" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
 succeeded in 0ms:
codex
这个目录本身像是文献/novelty 工作区，不含主实验文档；受当前只读沙箱限制，我不能跳出工作区补主实验文档或写入新记录。这里已经有 `novelty_rankinstab.md`，我会先读它避免重复已有判断，再联网补证据。
exec
/bin/zsh -lc "sed -n '1,260p' novelty_rankinstab.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef97f-1309-7cf1-8d2b-9a69b55991a8
--------
user
我们要做 ReID **效率**方法 Rank-Instability Adaptive ReID。idea: ReID 默认所有 query 用 full 网络推理(uniform compute); 应 **rank-stability conditioned**——用 cheap(早层/低分辨率)特征算的 top-k 排名 vs full 排名的**不一致度(rank instability)** 当路由信号: 稳定 query 早退(cheap 够), 不稳定走 full。隐藏变量 = cheap-vs-full rank disagreement(不是一个静态 difficulty score)。效率-精度 Pareto。

联网查确切先例 + 撞车, 逐类判它们是否已用 **cheap-vs-full rank disagreement** 当路由信号(而非静态 difficulty/置信度): (a)CFPER / query-difficulty-adaptive compute ReID; (b)early-exit / dynamic-inference / adaptive-depth / multi-exit person ReID; (c)cascade / coarse-to-fine / hierarchical retrieval ReID; (d)anytime ReID; (e)hardness/uncertainty-aware ReID; (f)通用 retrieval 的 adaptive re-ranking budget。**关键问: rank-instability(cheap-full 排名不一致)当信号, 相对'又一个 difficulty-adaptive(CFPER)'有没有独立新意? 还是会被归类成置信度/难度路由?** 撞了说撞哪+杀伤; 没撞说切口。诚实, verdict(撞车/存活)+证据链接+信心1-10。
codex
我先按仓库接手要求快速读当前实验文档，再做联网文献/代码检索。这个任务核心是判“cheap-vs-full 排名不一致”是否已被 ReID 或检索效率工作拿来做路由信号，所以我会按你列的类别逐类取证。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/innovation_brainstorm.md: No such file or directory

 exited 1 in 0ms:
sed: experiments/decisions.md: No such file or directory

 exited 1 in 0ms:
sed: experiments/results.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
sed: experiments/paper_materials/story.md: No such file or directory

codex
当前工作目录下没有 `experiments/...` 这一层，说明我们可能已经在某个子实验目录或文献目录里。我先定位仓库结构和这些文档的真实位置，然后继续检索。
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2

exec
/bin/zsh -lc ls in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
B_CONTAINMENT_DESIGN.md
GOPL_KILLSWITCH_DESIGN.md
HUBNESS_ANALYSIS.md
HUBNESS_KILLSWITCH_DESIGN.md
OSAC_KILLSWITCH_DESIGN.md
SYNTHESIS_METHODOLOGY.md
all_papers.txt
analyses
batches
cvpb_containment_full.log
cvpb_containment_killswitch_design.md
debate
hubness_debate.sh
launch_30deep.sh
novelty_hubness.md
novelty_osac.md
novelty_rankinstab.md
ondisk_pivot.md
pivot
reassess
reassess.sh
reassess2
reassess2.sh
reviews
strategic_pivot.sh
validate
validate_candidates.sh
video_feasibility.md

exec
/bin/zsh -lc "rg --files -g 'results.md' -g 'decisions.md' -g 'innovation_brainstorm.md' -g 'story.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "find .. -maxdepth 4 \\( -name results.md -o -name decisions.md -o -name innovation_brainstorm.md -o -name story.md \\)" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 exited 1 in 0ms:
 succeeded in 0ms:
codex
这个目录本身像是文献/novelty 工作区，不含主实验文档；受当前只读沙箱限制，我不能跳出工作区补主实验文档或写入新记录。这里已经有 `novelty_rankinstab.md`，我会先读它避免重复已有判断，再联网补证据。

exec
/bin/zsh -lc "sed -n '1,220p' all_papers.txt" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
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
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Confidence guided semi-supervised cross-modality person re-identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
Deep intelligent technique for person Re-identification system in surveillance images.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
Distribution aligned semantics adaption for lifelong person re-identification.pdf
Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
FDGReID - Federated Domain Generalization for Person Re-identification.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
Focusing on pedestrians like human for clothes changing person re-identification.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Multi Queue for Unsupervised Person Re-identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
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
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Two-stage knowledge distillation for visible-infrared person re-identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf

exec
/bin/zsh -lc 'rg -n "Rank|rank|instability|adaptive|early|exit|anytime|CFPER|cascade|re-ranking|rerank|difficulty|uncertainty|budget" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
./novelty_rankinstab.md:14:我们要做 ReID **效率**方法 Rank-Instability Adaptive ReID。idea: ReID 默认所有 query 用 full 网络推理(uniform compute); 应 **rank-stability conditioned**——用 cheap(早层/低分辨率)特征算的 top-k 排名 vs full 排名的**不一致度(rank instability)** 当路由信号: 稳定 query 早退(cheap 够), 不稳定走 full。隐藏变量 = cheap-vs-full rank disagreement(不是一个静态 difficulty score)。效率-精度 Pareto。
./novelty_rankinstab.md:16:联网查确切先例 + 撞车, 逐类判它们是否已用 **cheap-vs-full rank disagreement** 当路由信号(而非静态 difficulty/置信度): (a)CFPER / query-difficulty-adaptive compute ReID; (b)early-exit / dynamic-inference / adaptive-depth / multi-exit person ReID; (c)cascade / coarse-to-fine / hierarchical retrieval ReID; (d)anytime ReID; (e)hardness/uncertainty-aware ReID; (f)通用 retrieval 的 adaptive re-ranking budget。**关键问: rank-instability(cheap-full 排名不一致)当信号, 相对'又一个 difficulty-adaptive(CFPER)'有没有独立新意? 还是会被归类成置信度/难度路由?** 撞了说撞哪+杀伤; 没撞说切口。诚实, verdict(撞车/存活)+证据链接+信心1-10。
./novelty_rankinstab.md:25: exited 1 in 0ms:
./novelty_rankinstab.md:28: exited 1 in 0ms:
./novelty_rankinstab.md:31: exited 1 in 0ms:
./novelty_rankinstab.md:36: exited 1 in 0ms:
./novelty_rankinstab.md:65:novelty_rankinstab.md
./novelty_rankinstab.md:82: exited 1 in 0ms:
./novelty_rankinstab.md:85:这个目录本身像是文献/novelty 工作区，不含主实验文档；受当前只读沙箱限制，我不能跳出工作区补主实验文档或写入新记录。这里已经有 `novelty_rankinstab.md`，我会先读它避免重复已有判断，再联网补证据。
./HUBNESS_ANALYSIS.md:41:- vs k-reciprocal/CA-Jaccard(test-time re-rank): 我们是诊断变量定义, 不声称更好的 re-rank。
./HUBNESS_ANALYSIS.md:42:- vs HAL/NeighborRetr(cross-modal hubness-aware training): 图文检索非 person ReID; 且我们坐实 person ReID 训练端 remedy 被 re-rank 吞, 不重复它们的 training-time claim。
./HUBNESS_ANALYSIS.md:43:- vs hard-negative mining: H_neg 是全局误吸附(对很多身份), 非 anchor-local difficulty。
./HUBNESS_ANALYSIS.md:48:- 与 re-ranking 的互补性叙事(hub-fix R1 在 Market 赢 k-reciprocal +1.13 vs −0.12, 是 rank-repair 互补轴)。
./HUBNESS_ANALYSIS.md:52:强 ReID 的残差失败有清晰的 gallery-topology 结构(many-to-one 负向吸附), M(q) 干净解释 AP 误差(rho+0.60, 控代理后仍在, perm-p0.001), 越难的集越明显。但其 remedy 落在 k-reciprocal/camera-aware re-ranking 已覆盖的空间——**这是一个干净的诊断, 不是一个新方法**。
./launch_30deep.sh:6:cd "$LIB" || exit 1
./reviews/deep_3.md:146:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_3.md:183:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_3.md:191:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_3.md:203:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_3.md:281:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_3.md:297:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_3.md:302:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_3.md:357:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_3.md:483:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_3.md:512:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_3.md:515:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_3.md:516:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_3.md:593:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_3.md:597:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_3.md:611:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_3.md:617:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_3.md:708:427:   427	horizontal flipping, cropping, and erasing [30] are used. In the crossmodal alignment stage, the Adam optimizer is used with a base learning rate of 3.5e-04 and a warm-up learning rate that increases linearly
./reviews/deep_3.md:714:487:   487	achieving 55.1% mAP and 63.7% Rank-1, compared to 53.5% mAP
./reviews/deep_3.md:855:860:   860	rank-1 accuracy of 76.8% and a mAP of 74.6%, in the IndoorSearch mode, BDLF achieved a rank-1 accuracy of 84.2%
./reviews/deep_3.md:856:862:   862	achieved a rank-1 accuracy of 94.4% and a mAP of 90.1%
./reviews/deep_3.md:939:805:   805	Comprehensive comparison of person ReID methods across different modalities and datasets. Methods are evaluated using Rank-1 accuracy and mean Average
./reviews/deep_3.md:1137:1200:  1200	described in Section 3.2.1. As reported in He et al. (2021), the Baseline achieved ∼ 94% Rank-1 score (see Table 5) on the Market1501
./reviews/deep_3.md:1185:    30	often struggle with convergence instability and mode collapse.
./reviews/deep_3.md:1607:   128	been successfully applied in the V2V person ReID. For the network, early
./reviews/deep_3.md:2781:   448	model indeed adaptively selects effective feature conducive to person
./reviews/deep_3.md:3101:   226	based on the complexity of the attribute being described.
./reviews/deep_3.md:3302:   427	horizontal flipping, cropping, and erasing [30] are used. In the crossmodal alignment stage, the Adam optimizer is used with a base learning rate of 3.5e-04 and a warm-up learning rate that increases linearly
./reviews/deep_3.md:3312:   437	the ViT-B/16 backbone (86M parameters). Crucially, during inference, AG-ReID only requires the image encoder (ViT-B/16), resulting in identical computational complexity and model size as CLIPReID and other ViT-based methods. The attribute encoder is exclusively used during training and does not affect deployment efficiency.
./reviews/deep_3.md:3343:   468	Following common practices, we employ the cumulative matching characteristics (CMC) [21] at Rank-1 (R1) and the mean average
./reviews/deep_3.md:3356:   481	the effectiveness of fully utilizing the fine-grained semantic information within the pre-trained model without requiring extra annotation data. Furthermore, applying standard k-reciprocal re-ranking
./reviews/deep_3.md:3359:   484	re-ranking results are provided in supplementary materials.
./reviews/deep_3.md:3362:   487	achieving 55.1% mAP and 63.7% Rank-1, compared to 53.5% mAP
./reviews/deep_3.md:3470:   595	and 61.0% Rank-1 for CLIP-ReID (RN50), demonstrating the effectiveness of our approach across different architectures.
./reviews/deep_3.md:3575:   700	align with the data distribution encountered during the model’s extensive pre-training. When combined with k-reciprocal re-ranking,
./reviews/deep_3.md:4224:   415	reduce computational complexity. Therefore, we feed the extracted detail embeddings into a cross-attention-based transformer to facilitate cross-modality reasoning and information
./reviews/deep_3.md:4669:   860	rank-1 accuracy of 76.8% and a mAP of 74.6%, in the IndoorSearch mode, BDLF achieved a rank-1 accuracy of 84.2%
./reviews/deep_3.md:4671:   862	achieved a rank-1 accuracy of 94.4% and a mAP of 90.1%
./reviews/deep_3.md:4672:   863	for the VIS to IR search, and attained a rank-1 accuracy of
./reviews/deep_3.md:4679:   870	a rank-1 accuracy of 67.0% and a mAP of 68.9% in VIS to
./reviews/deep_3.md:4680:   871	IR search, as well as a rank-1 accuracy of 58.1% and a mAP
./reviews/deep_3.md:4734:   925	accuracy and computational complexity when the number of
./reviews/deep_3.md:4839:   925	Generalization veriﬁcation on multiple baselines. In Wilcoxon signed-rank test,
./reviews/deep_3.md:4844:   930	Rank1 mAP
./reviews/deep_3.md:4846:   932	Rank1 mAP
./reviews/deep_3.md:4916:  1002	Wilcoxon signed-rank test
./reviews/deep_3.md:4940:  1026	Rank1
./reviews/deep_3.md:4951:  1037	Rank1
./reviews/deep_3.md:4957:  1043	evaluation metrics, we report mAP and Rank1 of CMC. The mAP is
./reviews/deep_3.md:4993:  1079	For CMC, we compute the top-K accuracy Acc𝑖 @𝐾 for each query sample, and deﬁne rank K as following:
./reviews/deep_3.md:5017:  1103	we perform Wilcoxon signed-rank test on the metric of Rank1. Specifically, we repeated the experiment ﬁve times and used the results of
./reviews/deep_3.md:5018:  1104	Rank1 to perform the Wilcoxon signed-rank test. We compare the model
./reviews/deep_3.md:5024:  1110	Bootstrap CI calculation method to compute the 95 % CI for Rank-1 accuracy or mAP on a test set. As can be seen in Table 4, for TSE and CMG,
./reviews/deep_3.md:5025:  1111	our method achieves 72.23 % Rank1 accuracy (95 % CI [71.84, 72.62])
./reviews/deep_3.md:5026:  1112	and 71.08 % Rank1 accuracy (95 % CI [70.80, 71.36]), respectively.
./reviews/deep_3.md:5067:    29	re-ranking strategies. Our approach introduces explainability to 3D person ReID through the visualization of
./reviews/deep_3.md:5071:    33	FusionTexReIDNet achieves state-of-the-art performance across various scenarios, with Rank-1 accuracies of
./reviews/deep_3.md:5072:    34	98.5% and 89.7% Rank-1 on benchmark datasets, while providing interpretable results through its explainable
./reviews/deep_3.md:5129:    91	free from viewpoint-induced distortions, enabling better understanding of model decisions and facilitating performance improvements through explainable re-ranking strategies.
./reviews/deep_3.md:5132:    94	scenarios, with our method reaching 98.5% Rank-1 accuracy
./reviews/deep_3.md:5213:   175	7.7% improvement in Rank-1 accuracy over state-of-the-art 3D
./reviews/deep_3.md:5228:   190	Intersection-Alignment Score (IAS) from IoU between attention maps and visible clothes masks, plus activation alignment scores. Three re-ranking methods (KNN,
./reviews/deep_3.md:5287:   249	both streams through weighted combination and multiple re-ranking
./reviews/deep_3.md:5290:   252	re-ranking using the Intersection-Alignment Score (IAS). A detailed
./reviews/deep_3.md:5342:   304	KNN and attribute-based re-ranking
./reviews/deep_3.md:5343:   305	Explainable re-ranking
./reviews/deep_3.md:5357:   319	- Compute IAS value for explainable re-ranking
./reviews/deep_3.md:5378:   340	directly improves performance through explainable re-ranking. This
./reviews/deep_3.md:5407:   369	point cloud stream, allowing our network to adaptively balance their
./reviews/deep_3.md:5456:   418	ReID shows query with rank-1 (incorrect, red) and rank-7 (correct, green)
./reviews/deep_3.md:5477:   439	(low IAS), the explainable re-ranking mechanism adjusts the similarity
./reviews/deep_3.md:5509:   471	re-ranking component, which further enhances the explainability and
./reviews/deep_3.md:5510:   472	performance of 3D person ReID. The proposed re-ranking method
./reviews/deep_3.md:5513:   475	fine-grained patterns, the re-ranking component can effectively capture
./reviews/deep_3.md:5516:   478	The explainable re-ranking component leverages two key scores: the
./reviews/deep_3.md:5691:   653	score into the re-ranking process, we refine the distance matrix to
./reviews/deep_3.md:5715:   677	The proposed explainable re-ranking component enhances the 3D
./reviews/deep_3.md:5738:   700	Evaluation Metrics: We employ two widely used evaluation metrics for person ReID: rank-k accuracy of cumulative matching features
./reviews/deep_3.md:5740:   702	et al., 2015). Rank-k represents the probability that the true match
./reviews/deep_3.md:5741:   703	appears in the top-k retrieval results, with Rank-1 being the most
./reviews/deep_3.md:5747:   709	parameters (#params) to provide insights into the model complexity.
./reviews/deep_3.md:5751:   713	ShuffleNetV2 and MobileNetV2 achieve moderate results (e.g., ShuffleNetV2 with CE+Circle: 79.8% Rank-1 on Market-1501), recent advanced architectures like DC-Former and TransReID demonstrate strong
./reviews/deep_3.md:5752:   714	performance (DC-Former: 96.0% Rank-1, 90.5% mAP on Market-1501)
./reviews/deep_3.md:5755:   717	baseline architectures like DGCNN and PointNet++ show limited performance (DGCNN: 28.9% Rank-1 on Market-1501). More recent methods like PointReIDNet achieve better results (90.6% Rank-1, 75.3%
./reviews/deep_3.md:5757:   719	achieves 91.2% Rank-1 and 76.9% mAP on Market-1501, demonstrating competitive performance in this category.
./reviews/deep_3.md:5758:   720	When using Texformer-based point clouds (pc† ), our FusionTexReIDNet architecture shows substantial improvements. The baseline configuration achieves 92.9% Rank-1 and 79.9% mAP on Market-1501,
./reviews/deep_3.md:5759:   721	significantly outperforming OG-Net Deep’s 85.2% Rank-1 and 66.7%
./reviews/deep_3.md:5760:   722	mAP. This represents an improvement of 7.7% in Rank-1 accuracy
./reviews/deep_3.md:5763:   725	Impact of Optimization Techniques. As shown in Table 4, various optimization strategies can significantly enhance the baseline performance. Our framework incorporates three key optimization techniques: KNN re-ranking (1 ), attribute-based re-ranking (2 ), and optimal weight fusion (3 ). When all optimization techniques are applied
./reviews/deep_3.md:5764:   726	(FusionTexReIDNet1,2,3 with pc† ), our method achieves 98.5% Rank-1
./reviews/deep_3.md:5767:   729	FusionTexReIDNet1,2,3 achieving 89.7% Rank-1 accuracy on the challenging AG-ReID.v2 dataset. The superior performance validates our
./reviews/deep_3.md:5777:   739	90.5% Rank-1 and 86.6% mAP, while on the larger AG-ReID.v2 dataset,
./reviews/deep_3.md:5778:   740	FusionTexReIDNet attains 89.7% Rank-1 and 87.0% mAP. These results
./reviews/deep_3.md:6126:   304	KNN and attribute-based re-ranking
./reviews/deep_3.md:6127:   305	Explainable re-ranking
./reviews/deep_3.md:6141:   319	- Compute IAS value for explainable re-ranking
./reviews/deep_3.md:6162:   340	directly improves performance through explainable re-ranking. This
./reviews/deep_3.md:6191:   369	point cloud stream, allowing our network to adaptively balance their
./reviews/deep_3.md:6247:  1200	described in Section 3.2.1. As reported in He et al. (2021), the Baseline achieved ∼ 94% Rank-1 score (see Table 5) on the Market1501
./reviews/deep_3.md:6260:  1213	can effectively modify the appearance feature of persons. We also can clearly observe that the metric perturbation loss 𝐿𝑀 𝑃 Eq. (3) plays a very important role in this synthesis
./reviews/deep_3.md:6264:  1217	increases the difficulty of sample matching.
./reviews/deep_3.md:6267:  1220	metric relationship between samples, significantly increasing the difficulty of
./reviews/deep_3.md:6285:  1238	can clearly see that the MPG-Net can effectively modify the appearance
./reviews/deep_3.md:6286:  1239	feature of persons. Firstly, from Fig. 6, we can clearly observe that the
./reviews/deep_3.md:6291:  1244	increases the difficulty of sample matching. Specifically, (1) the basic
./reviews/deep_3.md:6296:  1249	increased sample difficulty. Of course, Fig. 6 can also verify that the
./reviews/deep_3.md:6302:  1255	In all, MPG-Net can indeed increase the difficulty of sample matching, and can ensure that main content does not change. These hard
./reviews/deep_3.md:6764:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reviews/deep_3.md:6819:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reviews/deep_8.md:129:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_8.md:142:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_8.md:159:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_8.md:162:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_8.md:171:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_8.md:203:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_8.md:233:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_8.md:342:that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
./reviews/deep_8.md:352:pairs. A-SDM adaptively pulls the unmatched positive pairs closer in the joint embedding space, thereby
./reviews/deep_8.md:430:pairs. Within the joint embedding space, the A-SDM module adaptively pulls positive pairs closer
./reviews/deep_8.md:431:together. In cases of mismatched positive pairs, the A-SDM module adaptively regulates the pulling
./reviews/deep_8.md:442:distinction between positive and negative samples. (b) Our adaptive global matching method builds on a
./reviews/deep_8.md:443:discriminative boundary and concentrates on unmatched positive samples, adaptively pulling them closer to
./reviews/deep_8.md:460:embedding space. It adaptively adjusts to narrow the distance between mismatched positive
./reviews/deep_8.md:472:a joint embedding space. Schroff et al. [39] proposed a triplet ranking loss to enforce a margin
./reviews/deep_8.md:476:ability to capture detailed semantic correspondences. To address this limitation, early local matching
./reviews/deep_8.md:548:(b) A-SDM obtains the adaptive weight through the similarity score of unmatched positive pairs, adaptively
./reviews/deep_8.md:552:𝑣 is linearly mapped into the joint image-text embedding
./reviews/deep_8.md:564:linearly mapped into the joint image-text representation space, generating a compressed global
./reviews/deep_8.md:571:Adopted from IRRA [21], we introduce a novel A-SDM module, which aims to adaptively pull the
./reviews/deep_8.md:601:image for 𝑇𝑖 at rank-𝑘, where 𝑘 > 1. Different from IRRA [21], we propose to derive an adaptive
./reviews/deep_8.md:615:pairs while allowing the model to focus adaptively on harder and misaligned pairs. Conversely,
./reviews/deep_8.md:795:triplet ranking loss [39]:
./reviews/deep_8.md:1629:evaluated using Rank-k accuracy (R@k, k=1, 5, 10).
./reviews/deep_8.md:2143:Computational Complexity: We compare the computational cost and inference time of CoRL with classic methods in Table 5. TIPCB and CFine build fine-grained correspondences by learning local features, reducing retrieval efficiency due to pairwise similarity calculations. IRRA and
./reviews/deep_8.md:2221:ages with the same identity as the given query text are ranked
./reviews/deep_8.md:2245:Table 5: Computational complexity comparison with several stateof-the-art methods on CUHK-PEDES.
./reviews/deep_8.md:2329:uncertainty-based learning for text-based person retrieval.
./reviews/deep_8.md:2357:that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
./reviews/deep_8.md:2367:pairs. A-SDM adaptively pulls the unmatched positive pairs closer in the joint embedding space, thereby
./reviews/deep_8.md:2441:A-SDM module adaptively pulls positive pairs closer together. In cases of mismatched positive pairs,
./reviews/deep_8.md:2442:the A-SDM module adaptively regulates the pulling force based on their relative distance within
./reviews/deep_8.md:2475:(b) Our adaptive global matching method.
./reviews/deep_8.md:2479:enhancing the distinction between positive and negative samples. (b) Our adaptive global matching method
./reviews/deep_8.md:2480:builds on a discriminative boundary and concentrates on unmatched positive samples, adaptively pulling
./reviews/deep_8.md:2492:• We present an adaptive similarity distribution matching module aimed at precisely aligning
./reviews/deep_8.md:2493:image-text pairs in a shared embedding space. It adaptively adjusts to narrow the distance
./reviews/deep_8.md:2508:a joint embedding space. Schroff et al. [39] proposed a triplet ranking loss to enforce a margin
./reviews/deep_8.md:2512:ability to capture detailed semantic correspondences. To address this limitation, early local matching
./reviews/deep_8.md:2645:obtains the adaptive weight through the similarity score of unmatched positive pairs, adaptively pulling
./reviews/deep_8.md:2650:𝑣 is linearly mapped into the joint image-text
./reviews/deep_8.md:2661:𝑡 , is linearly mapped into the joint image-text representation space, generating a compressed
./reviews/deep_8.md:2676:module, which aims to adaptively pull the unmatched positive image-text pairs into a shared
./reviews/deep_8.md:2696:image for 𝑇𝑖 at rank-𝑘, where 𝑘 > 1. Different from IRRA [21], we propose to derive an adaptive
./reviews/deep_8.md:2710:pairs while allowing the model to focus adaptively on harder and misaligned pairs. Conversely,
./reviews/deep_8.md:2973:importance of high-ranking ones, due to failing to consider the overall ranking order within a batch. To
./reviews/deep_8.md:2975:directly optimizes the cross-modality overall ranking order in VT-ReID. Unlike the recently introduced Smooth
./reviews/deep_8.md:2976:Average Precision (Smooth-AP), which primarily corrects misordered samples at high ranks, CAP specifically
./reviews/deep_8.md:2980:jointly optimizing these two aspects, we can effectively improve the cross-modality overall ranking order.
./reviews/deep_8.md:2982:Modality Alignment (DMA), which reduces the cross-modality discrepancy by adaptively adjusting the weights
./reviews/deep_8.md:3035:overall ranking order. (b) Original Smooth-AP will prioritize optimizing the ranking
./reviews/deep_8.md:3039:ranking orders. (For interpretation of the references to color in this figure legend, the
./reviews/deep_8.md:3042:an over-focus on optimizing low-ranking (located at the rear of the
./reviews/deep_8.md:3043:ranking list with lower similarity) positive samples at the expense
./reviews/deep_8.md:3044:of high-ranking (located at the front of the ranking list with higher
./reviews/deep_8.md:3045:similarity) ones, due to the failure to consider the overall ranking
./reviews/deep_8.md:3052:overall ranking order. However, the original Smooth-AP fails to address
./reviews/deep_8.md:3053:the main challenge of cross-modality in VT-ReID. Specifically, SmoothAP prioritizes correcting misordered samples at high ranks [8], while
./reviews/deep_8.md:3055:optimizing the ranking of intra-modality instances (high rank 𝑝1, 𝑝2)
./reviews/deep_8.md:3056:while disregarding the ranking of inter-modality instances (low rank
./reviews/deep_8.md:3060:optimizes the cross-modality overall ranking order within a batch.
./reviews/deep_8.md:3065:other and jointly optimize the cross-modality overall ranking order.
./reviews/deep_8.md:3083:(CAP) to directly optimize the cross-modality overall ranking orders, which explicitly handle the large cross-modality discrepancy
./reviews/deep_8.md:3151:𝐿CAP can directly optimize the cross-modality overall ranking orders in
./reviews/deep_8.md:3156:Existing metric learning methods will over-focus on optimizing lowranking positive samples at the expense of high-ranking ones due to
./reviews/deep_8.md:3157:a lack of consideration of the overall ranking orders. Moreover, these
./reviews/deep_8.md:3161:the cross-modality overall ranking orders. We next introduce CAP in
./reviews/deep_8.md:3164:the query and the sample 𝑖. The ranking of sample 𝑖 in any set 𝑆 can
./reviews/deep_8.md:3215:Precision’’, which allows us to directly optimize the global ranking
./reviews/deep_8.md:3236:So the original Smooth-AP will prioritize optimizing the ranking of
./reviews/deep_8.md:3237:intra-modality instances while neglecting to optimize the ranking of
./reviews/deep_8.md:3297:overall ranking orders (a → c). These two aspects jointly optimize the model and gain
./reviews/deep_8.md:3540:retrieval performance. For CMC, we report the rank-1 (R1), rank-10
./reviews/deep_8.md:3541:(R10), and rank-20 (R20) accuracies.
./reviews/deep_8.md:3602:Specifically, for the all-search mode, we achieve rank-1 = 75.05% and
./reviews/deep_8.md:3603:mAP = 71.72% for the single-shot setting, and, rank-1 = 78.94% and
./reviews/deep_8.md:3605:we achieve rank-1 = 83.92% and mAP = 85.44% for the single-shot
./reviews/deep_8.md:3606:setting, and, rank-1 = 87.77% and mAP = 77.88% for the multishot setting. Compared to the best competitor (MAUM [67]), we show
./reviews/deep_8.md:3608:by 3.37% and 6.96% in rank-1 accuracy for single-shot setting under
./reviews/deep_8.md:3611:reported in Table 2. It is clear that our method outperforms the state-ofthe-art methods by a large margin in both evaluation settings. Specifically, we obtain rank-1 = 93.93% and mAP = 89.17% for the ‘‘Visible
./reviews/deep_8.md:3612:to Thermal’’ setting, and, rank-1 = 92.09% and mAP = 87.19% for
./reviews/deep_8.md:3614:(GLMC [10]), our method is higher than it by 2.09% in rank-1 accuracy
./reviews/deep_8.md:3616:by 0.97% in rank-1 accuracy and 8.16% in mAP accuracy on the
./reviews/deep_8.md:3639:74], our CAP can directly optimize the overall ranking orders and
./reviews/deep_8.md:4445:4.4. Hyperparameter and complexity analysis
./reviews/deep_8.md:5011:Computational complexity analysis. we compare the computational complexity of our model with the competitive method DEEN
./reviews/deep_8.md:5063:and visual tokens enables adaptive reweighting of image representations, effectively suppressing background
./reviews/deep_8.md:5107:perspectives thus facilitates the construction of more comprehensive and adaptive intelligent
./reviews/deep_8.md:5113:frontal or side views of pedestrians, where clothing and appearance features are clearly visible. In
./reviews/deep_8.md:5209:discrepancies. Additionally, it proposes an iterative adaptive batch construction strategy to mitigate
./reviews/deep_8.md:5249:self-calibration and adaptive prompting. This method refines the view-invariant features extracted
./reviews/deep_8.md:5438:Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
./reviews/deep_8.md:5519:overall ranking orders (a → c). These two aspects jointly optimize the model and gain
./reviews/deep_8.md:6059:rank-1 mAP rank-1 mAP rank-1 mAP
./reviews/deep_8.md:6186:increases linearly from 3×10−6 to 3×10−4 during the first 10 epochs and
./reviews/deep_8.md:6204:Matching Precision (rank-k) and Mean Average Precision (mAP) to
./reviews/deep_8.md:6205:evaluate the model’s performance. Rank-k represents the probability of
./reviews/deep_8.md:6225:achieve rank-1 accuracies of 68.6% and 52.0%, and mAP of 22.7%
./reviews/deep_8.md:6248:rank-1rank-10mAP rank-1rank-10mAP
./reviews/deep_8.md:6340:rank-1
./reviews/deep_8.md:6342:rank-10
./reviews/deep_8.md:6411:4.4% and 18.6% in terms of rank-1 and mAP on the Celeb-Light dataset,
./reviews/deep_8.md:6415:by 22.6% in terms of mAP and 28.5% in terms of rank-1 accuracy. On
./reviews/deep_8.md:6417:by 8.4% in rank-1 accuracy, demonstrating a considerable advantage.
./reviews/deep_8.md:6420:proposed CSGN performs excellently across all environments. Specifically, our algorithm achieves 100.0% rank-1 accuracy and 99.4% mAP
./reviews/deep_8.md:6421:when the clothing remains unchanged, and 63.2% rank-1 accuracy, and
./reviews/deep_8.md:6429:where we add MDG, we observe an improvement of 2.2% in rank-1
./reviews/deep_8.md:6430:accuracy and 2.3% in mAP. When we further incorporate the VSS module in stage 2 of CSGN, performance improves by 3.0% in both rank-1
./reviews/deep_8.md:6432:to a significant performance boost of 14.5% in rank-1 accuracy and
./reviews/deep_8.md:6452:rank-1
./reviews/deep_8.md:6454:rank-10
./reviews/deep_8.md:6460:rank-1
./reviews/deep_8.md:6462:rank-10
./reviews/deep_8.md:6517:rank-1
./reviews/deep_8.md:6519:rank-10
./reviews/deep_8.md:6640:algorithm achieves the best performance on Celeb, Celeb-Light, and VCClothes datasets, with a rank-1 accuracy of 64.6%,56.4%, and 92.7%
./reviews/deep_8.md:6650:with CLIP alone, with improvements of 2.2%, 3.0%, and 2.3% in rank1, rank-10, and mAP on the PRCC dataset, respectively. Additionally, in
./reviews/deep_8.md:6653:CoOP by 2.2% in rank-1 accuracy and 2.3% in mAP. Similarly, when
./reviews/deep_8.md:6655:rank-1 accuracy and 1.0% in mAP. These results highlight the superiority of our method. It can be demonstrated that our MDG generates
./reviews/deep_8.md:6663:in rank-1 and 14.0% in mAP. Similarly, when comparing VSS with
./reviews/deep_8.md:6664:CLIP(ViT-B-16), our VSS outperforms it by 5.9% in rank-1 and 5.4%
./reviews/deep_8.md:6674:our HSF loss, our method outperforms 14.5% and 11.2% in rank-1 and
./reviews/deep_8.md:6676:outperforms 10.6% and 3.4% in rank-1 and mAP. Comparing line 3
./reviews/deep_8.md:6677:with our HSF loss, our method outperforms 12.2% and 4.4% in rank-1
./reviews/deep_8.md:6826:triplet ranking loss [39]:
./reviews/deep_8.md:6885:Evaluation Metrics. To gauge retrieval quality, we primarily report Rank-K results (K = 1, 5, 10),
./reviews/deep_8.md:6887:Average Precision (mAP) is adopted to summarize ranking accuracy over all query outcomes. In
./reviews/deep_8.md:6962:Rank-1 and 66.66% mAP, while surpassing IRRA by 0.74% in Rank-5 and 0.41% in Rank-10. When
./reviews/deep_8.md:6964:achieves Rank-5 accuracy exceeding 95% with the HAM-based backbone. Notably, FMFA with
./reviews/deep_8.md:6965:NAM-based backbone attains 91.33% in Rank-5, outperforming IRRA with the HAM-based backbone
./reviews/deep_8.md:6968:competitive performance, attaining 61.05% Rank-1, 83.85% Rank-5, 89.80% Rank-10, and 48.22%
./reviews/deep_8.md:6969:mAP, respectively, outperforming IRRA by 1.55% in Rank-1 and 2.05% in Rank-5. When adopting
./reviews/deep_8.md:6971:IRRA by 0.45% in Rank-1 with both the NAM-based and HAM-based backbones. Notably, FMFA
./reviews/deep_8.md:6972:achieves Rank-5 accuracy higher than 88% with the HAM-based backbone.
./reviews/deep_8.md:7032:Rank-1
./reviews/deep_8.md:7034:Rank-5
./reviews/deep_8.md:7036:Rank-10
./reviews/deep_8.md:7179:Rank-1 Rank-5
./reviews/deep_8.md:7237:Rank-10
./reviews/deep_8.md:7278:the leading results across all metrics, attaining 64.29% Rank-1 and 39.43% mAP. Compared with
./reviews/deep_8.md:7279:IRRA, FMFA shows a notable improvement of 0.81% Rank-1 and 1.23% mAP, which is meaningful
./reviews/deep_8.md:7288:Rank-1 Rank-5
./reviews/deep_8.md:7349:Rank-10
./reviews/deep_8.md:7412:Rank-1
./reviews/deep_8.md:7419:Rank-5 Rank-10
./reviews/deep_8.md:7435:Rank-1
./reviews/deep_8.md:7442:Rank-5 Rank-10
./reviews/deep_8.md:7458:Rank-1
./reviews/deep_8.md:7465:Rank-5 Rank-10
./reviews/deep_8.md:7484:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./reviews/deep_8.md:7498:on three widely used datasets. Table 6 summarizes the Rank-1/5/10 accuracies (%) together with
./reviews/deep_8.md:8022:evaluated using Rank-k accuracy (R@k, k=1, 5, 10).
./reviews/deep_8.md:8180:ranking loss [39]:
./reviews/deep_8.md:8235:Evaluation Metrics. To gauge retrieval quality, we primarily report Rank-K results (K = 1, 5,
./reviews/deep_8.md:8237:mean Average Precision (mAP) is adopted to summarize ranking accuracy over all query outcomes.
./reviews/deep_8.md:8310:74.16% Rank-1 and 66.66% mAP, while surpassing IRRA by 0.74% in Rank-5 and 0.41% in Rank-10.
./reviews/deep_8.md:8312:and achieves Rank-5 accuracy exceeding 95% with the HAM-based backbone. Notably, FMFA
./reviews/deep_8.md:8313:with NAM-based backbone attains 91.33% in Rank-5, outperforming IRRA with the HAM-based
./reviews/deep_8.md:8317:competitive performance, attaining 61.05% Rank-1, 83.85% Rank-5, 89.80% Rank-10, and 48.22%
./reviews/deep_8.md:8318:mAP, respectively, outperforming IRRA by 1.55% in Rank-1 and 2.05% in Rank-5. When adopting
./reviews/deep_8.md:8379:Rank-1
./reviews/deep_8.md:8381:Rank-5
./reviews/deep_8.md:8383:Rank-10
./reviews/deep_8.md:8522:Rank-1 Rank-5
./reviews/deep_8.md:8580:Rank-10
./reviews/deep_8.md:8618:IRRA by 0.45% in Rank-1 with both the NAM-based and HAM-based backbones. Notably, FMFA
./reviews/deep_8.md:8619:achieves Rank-5 accuracy higher than 88% with the HAM-based backbone.
./reviews/deep_8.md:8622:the leading results across all metrics, attaining 64.29% Rank-1 and 39.43% mAP. Compared with
./reviews/deep_8.md:8631:Rank-1 Rank-5
./reviews/deep_8.md:8692:Rank-10
./reviews/deep_8.md:8729:IRRA, FMFA shows a notable improvement of 0.81% Rank-1 and 1.23% mAP, which is meaningful
./reviews/deep_8.md:8731:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./reviews/deep_8.md:8745:on three widely used datasets. Table 6 summarizes the Rank-1/5/10 accuracies (%) together with
./reviews/deep_8.md:8749:Table 6, replacing A-SDM with SDM results in a reduction of Rank-1 accuracy by 0.59%, 0.78%, and
./reviews/deep_8.md:8754:replacing the A-SDM module with the SDM module results in 0.43% and 0.48% decrease in Rank-1
./reviews/deep_8.md:8755:and Rank-5 on the CUHK-PEDES dataset, respectively, as well as a 1.55% drop in Rank-5 and a
./reviews/deep_8.md:8808:Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
./reviews/deep_8.md:8881:Evaluation Metrics. We adopt Rank-1 accuracy and Mean Average Precision (mAP) as the
./reviews/deep_8.md:9092:augmentation or re-ranking strategies are applied.
./reviews/deep_8.md:9104:protocols. These results clearly demonstrate the effectiveness of CVAF in modeling cross-view
./reviews/deep_8.md:9285:CoCoOp is instance-adaptive, making it more robust to class migration.
./reviews/deep_8.md:9327:is then layer-normalized and linearly projected into the cross-modal
./reviews/deep_8.md:9699:rank-1 mAP rank-1 mAP rank-1 mAP
./reviews/deep_8.md:9826:increases linearly from 3×10−6 to 3×10−4 during the first 10 epochs and
./reviews/deep_8.md:9844:Matching Precision (rank-k) and Mean Average Precision (mAP) to
./reviews/deep_8.md:9845:evaluate the model’s performance. Rank-k represents the probability of
./reviews/deep_8.md:9865:achieve rank-1 accuracies of 68.6% and 52.0%, and mAP of 22.7%
./reviews/deep_8.md:9909:Rank-1 Rank-5
./reviews/deep_8.md:9970:Rank-10
./reviews/deep_8.md:10033:Rank-1
./reviews/deep_8.md:10040:Rank-5 Rank-10
./reviews/deep_8.md:10056:Rank-1
./reviews/deep_8.md:10063:Rank-5 Rank-10
./reviews/deep_8.md:10079:Rank-1
./reviews/deep_8.md:10086:Rank-5 Rank-10
./reviews/deep_8.md:10105:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./reviews/deep_8.md:10119:on three widely used datasets. Table 6 summarizes the Rank-1/5/10 accuracies (%) together with
./reviews/deep_8.md:10129:reduction of Rank-1 accuracy by 0.59%, 0.78%, and 0.75% across the three datasets, and also causes
./reviews/deep_8.md:10134:module results in 0.43% and 0.48% decrease in Rank-1 and Rank-5 on the CUHK-PEDES dataset,
./reviews/deep_8.md:10135:respectively, as well as a 1.55% drop in Rank-5 and a 0.58% decline in mAP on the RSTPReid dataset.
./reviews/deep_8.md:10141:drop of 0.26% and 0.21% in Rank-5 and Rank-10 on the CUHK-PEDES dataset, and a more significant
./reviews/deep_8.md:10142:decline of 2.40% and 1.10% in Rank-5 and Rank-10, along with a 0.53% decrease in mAP on the
./reviews/deep_8.md:10143:RSTPReid dataset. However, EFA causes a 0.11% drop in Rank-5 on ICFG-PEDES, suggesting that
./reviews/deep_8.md:10150:and 0.95% drops in Rank-1 on CUHK-PEDES, ICFG-PEDES, and RSTPReid, and causes a 1.23% drop
./reviews/deep_8.md:10162:evaluated metrics, suggesting that the adaptive pull force on unmatched positive pairs is optimal.
./reviews/deep_8.md:10195:Rank-1
./reviews/deep_8.md:10203:Rank-5 Rank-10
./reviews/deep_8.md:10215:Rank-1
./reviews/deep_8.md:10223:Rank-5
./reviews/deep_8.md:10230:Rank-10
./reviews/deep_8.md:10249:Rank-1
./reviews/deep_8.md:10253:Rank-5
./reviews/deep_8.md:10257:Rank-10
./reviews/deep_8.md:10268:ensures a positive lower bound for the adaptive weight, preventing it from becoming zero when a
./reviews/deep_8.md:10301:on RSTPReid to 50 s vs. 91 s on ICFG-PEDES. These comparisons clearly demonstrate that FMFA
./reviews/deep_8.md:10310:positive pairs and adaptively pulls the positive pairs closer. More comparisons of the top-5 retrieved
./reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
./reviews/deep_8.md:10587:model achieves notable performance improvements on AG-ReID: +1.68% Rank-1 and +0.59% mAP
./reviews/deep_8.md:10588:under the A → G setting, and +2.81% Rank-1 and +1.18% mAP under the G → A setting. Similarly,
./reviews/deep_8.md:10589:on the CARGP dataset, the model gains +2.28% Rank-1 and +1.18% mAP in the ALL setting, and
./reviews/deep_8.md:10590:+3.01% Rank-1 and +2.69% mAP in the A ↔ G setting. These consistent improvements demonstrate
./reviews/deep_8.md:10668:Table 6. Model Complexity Comparison of Baseline CLIP-ReID and CVAF
./reviews/deep_8.md:10699:Analysis of Model Complexity. To evaluate the computational cost of the proposed CVAF module,
./reviews/deep_8.md:10702:As shown in Table 6, the CVAF module introduces only minor increases in model complexity: For
./reviews/deep_8.md:10728:rank-1
./reviews/deep_8.md:10730:rank-10
./reviews/deep_8.md:10736:rank-1
./reviews/deep_8.md:10738:rank-10
./reviews/deep_8.md:10793:rank-1
./reviews/deep_8.md:10795:rank-10
./reviews/deep_8.md:10916:algorithm achieves the best performance on Celeb, Celeb-Light, and VCClothes datasets, with a rank-1 accuracy of 64.6%,56.4%, and 92.7%
./reviews/deep_8.md:10926:with CLIP alone, with improvements of 2.2%, 3.0%, and 2.3% in rank1, rank-10, and mAP on the PRCC dataset, respectively. Additionally, in
./reviews/deep_8.md:10929:CoOP by 2.2% in rank-1 accuracy and 2.3% in mAP. Similarly, when
./reviews/deep_8.md:10931:rank-1 accuracy and 1.0% in mAP. These results highlight the superiority of our method. It can be demonstrated that our MDG generates
./reviews/deep_8.md:10939:in rank-1 and 14.0% in mAP. Similarly, when comparing VSS with
./reviews/deep_8.md:10940:CLIP(ViT-B-16), our VSS outperforms it by 5.9% in rank-1 and 5.4%
./reviews/deep_8.md:10950:our HSF loss, our method outperforms 14.5% and 11.2% in rank-1 and
./reviews/deep_8.md:10952:outperforms 10.6% and 3.4% in rank-1 and mAP. Comparing line 3
./reviews/deep_8.md:10953:with our HSF loss, our method outperforms 12.2% and 4.4% in rank-1
./reviews/deep_8.md:11053:Liu, Fangyi, Ye, Mang, Du, Bo, 2023. Dual level adaptive weighting for cloth-changing
./reviews/deep_8.md:11688:Computational complexity analysis. we compare the computational complexity of our model with the competitive method DEEN
./reviews/deep_8.md:11724:see that the top-10 retrieval ranking list of baseline contains many
./reviews/deep_8.md:11726:the ranking list with more positive samples ranked in the top positions.
./reviews/deep_8.md:11772:This paper presents a novel approach called Cross-modality Average Precision (CAP) for directly optimizing the overall ranking order in visible-thermal person re-identification (VT-ReID) tasks, where
./reviews/deep_8.md:11906:Computational complexity analysis.
./reviews/deep_8.md:11959:learn ranking loss surrogates, in: CVPR, 2019.
./reviews/deep_8.md:11960:[33] K. He, F. Cakir, S.A. Bargal, S. Sclaroff, Hashing as tie-aware learning to rank,
./reviews/deep_8.md:11963:Optimizing rank-based metrics with blackbox differentiation, in: CVPR, 2020.
./reviews/deep_8.md:11964:[35] F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff, Deep metric learning to rank, in:
./reviews/deep_8.md:11975:[40] Y. Ling, Z. Luo, Y. Lin, S. Li, A multi-constraint similarity learning with adaptive
./reviews/deep_8.md:11995:dual-constrained top-ranking, in: IJCAI, 2018.
./reviews/deep_8.md:12032:[3] M. Ye, X. Lan, Z. Wang, P.C. Yuen, Bi-directional center-constrained top-ranking
./reviews/deep_8.md:12127:2. 重定义动作：作者把问题从“学一个跨模态嵌入空间”改成“直接优化跨模态 AP”。关键词是 `Cross-modality Average Precision`、`CAPV`、`CAPT`、`overall ranking order`、`cross-modality discrepancy`。
./reviews/deep_8.md:12174:2. 重定义动作：作者把问题从“学一个跨模态嵌入空间”改成“直接优化跨模态 AP”。关键词是 `Cross-modality Average Precision`、`CAPV`、`CAPT`、`overall ranking order`、`cross-modality discrepancy`。
./reviews/deep_7.md:73:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_7.md:79:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_7.md:93:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_7.md:173:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_7.md:181:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_7.md:189:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_7.md:222:./4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_7.md:239:./Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_7.md:242:./CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_7.md:251:./Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_7.md:283:./Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_7.md:313:./Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_7.md:637:    74	identity characteristics adaptively to tackle cloth-changing cases. On
./reviews/deep_7.md:646:    83	the learned semantic features facilitate model learning adaptively. To
./reviews/deep_7.md:759:   196	pedestrian images and extract identity characteristics adaptively to
./reviews/deep_7.md:801:   238	also used to refine F c in an adaptive manner. F c is recalibrated along
./reviews/deep_7.md:817:   254	generating scores to refine the content semantics of pedestrian features adaptively. The ∼ notation represents our proposed semantics
./reviews/deep_7.md:854:   291	Rank-1
./reviews/deep_7.md:858:   295	Rank-1
./reviews/deep_7.md:862:   299	Rank-1
./reviews/deep_7.md:866:   303	Rank-1
./reviews/deep_7.md:1109:   546	rate linearly increases from 3 × 10−5 to 3 × 10−4 in the first 10
./reviews/deep_7.md:1162:   599	Rank-1
./reviews/deep_7.md:1164:   601	Rank-5
./reviews/deep_7.md:1261:   698	Rank-1
./reviews/deep_7.md:1265:   702	Rank-1
./reviews/deep_7.md:1317:   754	Rank-1 -------------------------------------------------------------------------------------------------------> Rank-10
./reviews/deep_7.md:1321:   758	Rank-1
./reviews/deep_7.md:1378:    99	The average feature magnitude of the images during the training process on the Market-1501 dataset with naive softmax loss. (d) Rank-1 change curve during the
./reviews/deep_7.md:1388:   109	Furthermore, Fig. 1(d) shows the relationship between classiﬁcation accuracy (rank-1) and epochs in the Market-1501 dataset under our baseline method [5]. With the increase of epochs, the overall accuracy is improved. But we can see that in the ﬁrst 30 epochs,
./reviews/deep_7.md:1389:   110	there is a decline in rank-1. At the same time, when rank-1 decreases, image feature magnitude increases in the reverse direction.
./reviews/deep_7.md:1419:   140	an adaptive gradient during the training phase.
./reviews/deep_7.md:1491:   212	changed the minus cosine margin of CosFace to an additive angular margin. More recently, ElasticFace [30] replaced ﬁxed margins with adaptive elastic margins that adjust based on class separability. X2-Softmax
./reviews/deep_7.md:1492:   213	[32] further introduced quadratic function-based adaptive margins that
./reviews/deep_7.md:1500:   221	which is highly innovative with similar complexity.
./reviews/deep_7.md:1721:   442	in the early stages, the network can achieve smooth parameter updates
./reviews/deep_7.md:1856:   101	Rank-1 accuracy of the top-performing supervised method is 80.83%
./reviews/deep_7.md:1907:   152	dual-constrained top-ranking loss for modality alignment. Adversarial learning emerged as another promising direction in VI-ReID. The
./reviews/deep_7.md:1913:   158	method to optimize the cross-modality overall ranking order.
./reviews/deep_7.md:2013:   258	the labeled visible and infrared features obtained from feature extraction module are fed into the adaptive confidence generation block,
./reviews/deep_7.md:2027:   272	Research on semi-supervised learning in cross-modality person reidentification remains in its early stages. To our knowledge, there is
./reviews/deep_7.md:2116:   361	image pairs between any two cross-modality clusters. Then, we rank
./reviews/deep_7.md:2252:   497	quantity discrepancy primarily stems from the greater complexity of
./reviews/deep_7.md:2286:   531	cross-modality clusters from the ranked candidates.
./reviews/deep_7.md:2595:   840	best unsupervised models, with Rank-1 and mAP gains being
./reviews/deep_7.md:2598:   843	in terms of Rank-1 and mAP, respectively. Note that the results of
./reviews/deep_7.md:2636:   754	Rank-1 -------------------------------------------------------------------------------------------------------> Rank-10
./reviews/deep_7.md:2640:   758	Rank-1
./reviews/deep_7.md:2644:   762	Rank-1
./reviews/deep_7.md:2716:   834	Rank-1 -------------------------------------------------------------------------------------------------------> Rank-10
./reviews/deep_7.md:3741:   241	adaptively assigns weights based on the similarity of sample
./reviews/deep_7.md:3843:   343	Due to the difficulty of collecting data with various corruptions, there are still no corrupted ReID datasets available.
./reviews/deep_7.md:3927:   427	our WCL aims to adaptively adjust penalties for challenging
./reviews/deep_7.md:3928:   428	samples, which is realized by adding an adaptive weight ω.
./reviews/deep_7.md:3949:   449	where the adaptive weight ω is obtained by calculating the
./reviews/deep_7.md:4407:   206	the uncertainty, thereby improving the discriminative ability of the
./reviews/deep_7.md:4413:   212	similarity of sample pairs adaptively, so as to alleviate the influence of
./reviews/deep_7.md:4738:   537	applied to Eq. (8), enabling the adaptive adjustment of the similarity of
./reviews/deep_7.md:5168:   704	The Rank-1, Rank-5 and mAP (%) results of diﬀerent losses on Market-1501
./reviews/deep_7.md:5177:   713	The Rank-1, Rank-5, Rank-10 and mAP (%) results for diﬀerent methods on
./reviews/deep_7.md:5183:   719	rank-1
./reviews/deep_7.md:5185:   721	rank-5
./reviews/deep_7.md:5189:   725	rank-1
./reviews/deep_7.md:5191:   727	rank-5
./reviews/deep_7.md:5381:   764	the Cumulated Matching Characteristics (CMC) at Rank-1 (R1), Rank-5
./reviews/deep_7.md:5382:   765	(R5) and Rank-10 (R10) accuracies. The post-processing methods are
./reviews/deep_7.md:5383:   766	not used for inference, i.e., re-ranking or multi-query fusion.
./reviews/deep_7.md:5566:   747	leads to a more challenging sample ranking (lower Rank-1),
./reviews/deep_7.md:5570:   751	Match Characteristics of Rank-1 accuracy (R-1) [53], mean
./reviews/deep_7.md:5648:   829	Rank-1 metric (from 75.86% to 85.68%) under the Corrupt
./reviews/deep_7.md:5867:   810	rank-1
./reviews/deep_7.md:5869:   812	rank-5
./reviews/deep_7.md:5871:   814	rank-10
./reviews/deep_7.md:5875:   818	rank-1
./reviews/deep_7.md:5877:   820	rank-5
./reviews/deep_7.md:5879:   822	rank-10
./reviews/deep_7.md:5957:   900	(AM-Softmax is the same with CosFace), it can achieve slight improvement, i.e., mAP 1.2 % improvement on Market-1501 and rank-1 1.8 %
./reviews/deep_7.md:5964:   852	Rank-1 and mAP accuracy on the RegDB dataset, as show in Table
./reviews/deep_7.md:5966:   854	remarkable improvement of 19.88%(=89.36%–69.48%) in Rank-1 accuracy and an impressive enhancement of 18.76%(=84.17%–65.41%)
./reviews/deep_7.md:5969:   857	of 19.26%(=89.11%–69.85%) in Rank-1 accuracy and a notable
./reviews/deep_7.md:6006:   894	achieves improvements of 29.54%, 9.45%, and 2.13% in terms of rank1 accuracy when utilizing labeled data rates of 10%, 25%, and 50%,
./reviews/deep_7.md:6008:   896	improves the rank-1 accuracy by 29.78%, 11.35%, and 3.98% under
./reviews/deep_7.md:6262:258:   258	the labeled visible and infrared features obtained from feature extraction module are fed into the adaptive confidence generation block,
./reviews/deep_7.md:6390:99:    99	The average feature magnitude of the images during the training process on the Market-1501 dataset with naive softmax loss. (d) Rank-1 change curve during the
./reviews/deep_7.md:6394:109:   109	Furthermore, Fig. 1(d) shows the relationship between classiﬁcation accuracy (rank-1) and epochs in the Market-1501 dataset under our baseline method [5]. With the increase of epochs, the overall accuracy is improved. But we can see that in the ﬁrst 30 epochs,
./reviews/deep_7.md:6395:110:   110	there is a decline in rank-1. At the same time, when rank-1 decreases, image feature magnitude increases in the reverse direction.
./reviews/deep_7.md:6412:140:   140	an adaptive gradient during the training phase.
./reviews/deep_7.md:6488:907:   907	MSMT17 datasets. When we utilize the margin parameter, the performance is further improved. For example, the Condense loss gains rank1/mAP 2.2 %/2.2 % on MSMT17. The improvement on Market-1501 is
./reviews/deep_7.md:6496:960:   960	early training process, compared with Naive softmax loss, our Condense
./reviews/deep_7.md:6514:1053:  1053	when compared with the Condense loss (without triplet loss), SphereReID and Circle loss respectively performed rank-1/mAP 0.3 %/2.5 %,
./reviews/deep_7.md:6532:1356:  1356	metrics. Under the framework of Transformer, whether baseline is vitbase or deit-base, Condense loss achieves the best accuracy of any rank1/mAP on the Market-1501 and MSMT17 datasets. Both the experimental data of APNET, MSINet and Trans-ReID [12] are reproduced by us
./reviews/deep_7.md:6608:427:   427	our WCL aims to adaptively adjust penalties for challenging
./reviews/deep_7.md:6764:747:   747	and 7.1% for Rank1 accuracy. Similarly, CARE brings 4.4%
./reviews/deep_7.md:6769:755:   755	full model (‘CSSA4 ’). Clearly, adding SGFA can boost the
./reviews/deep_7.md:6802:   740	and 4.6%, as well as Rank1 accuracy by 3.6%, 4.3%, and 4.3%
./reviews/deep_7.md:6809:   747	and 7.1% for Rank1 accuracy. Similarly, CARE brings 4.4%
./reviews/deep_7.md:6817:   755	full model (‘CSSA4 ’). Clearly, adding SGFA can boost the
./reviews/deep_7.md:6818:   756	Rank1 accuracy from 78.6% to 83.7% on LS-VID. The
./reviews/deep_7.md:6983:1. 触发观察：作者先观察训练过程中“特征向量范数”和准确率的关系。随着训练推进，特征范数下降，准确率上升；Market-1501 前 30 个 epoch 里 rank-1 下降时，特征范数反而上升。这个观察把一个很普通的 loss 设计问题，变成了“训练动态里范数压缩不稳定”的问题。
./reviews/deep_7.md:7025:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
./reviews/deep_7.md:7037:4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。
./reviews/deep_7.md:7049:4. 证据闭环：Table 1 很强。单标签里 F、C、I 各有表现，两两组合普遍强于单标签，HPL 三标签又强于两标签；加 PLC 从 80.3/92.5 提到 82.5/92.9，加 CCL 比普通 CL 更好，最终 Ours 达到 87.2 mAP、95.0 Rank-1。Table 2 还证明 DBSCAN 做 fine、K-means 做 coarse 最优，Table 3 讨论 hard/soft 标签形式。这些消融直接证明“异质标签互补”这个重定义。
./reviews/deep_7.md:7064:1. 触发观察：作者先观察训练过程中“特征向量范数”和准确率的关系。随着训练推进，特征范数下降，准确率上升；Market-1501 前 30 个 epoch 里 rank-1 下降时，特征范数反而上升。这个观察把一个很普通的 loss 设计问题，变成了“训练动态里范数压缩不稳定”的问题。
./reviews/deep_7.md:7106:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
./reviews/deep_7.md:7118:4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。
./reviews/deep_7.md:7130:4. 证据闭环：Table 1 很强。单标签里 F、C、I 各有表现，两两组合普遍强于单标签，HPL 三标签又强于两标签；加 PLC 从 80.3/92.5 提到 82.5/92.9，加 CCL 比普通 CL 更好，最终 Ours 达到 87.2 mAP、95.0 Rank-1。Table 2 还证明 DBSCAN 做 fine、K-means 做 coarse 最优，Table 3 讨论 hard/soft 标签形式。这些消融直接证明“异质标签互补”这个重定义。
./reviews/deep_13.md:74:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_13.md:82:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_13.md:90:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_13.md:134:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_13.md:140:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_13.md:154:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_13.md:222:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_13.md:239:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_13.md:242:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_13.md:251:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_13.md:283:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_13.md:313:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_13.md:461:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_13.md:589:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_13.md:618:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_13.md:621:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_13.md:622:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_13.md:699:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_13.md:703:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_13.md:717:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_13.md:723:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_13.md:1019:693:1.6%, respectively. For MSMT17 → Market-1501, the Rank-1 and mAP
./reviews/deep_13.md:1088:571:The primary goal of improving gradient consistency at the early
./reviews/deep_13.md:1652:     3	Rank-1
./reviews/deep_13.md:1674:    25	Rank-1
./reviews/deep_13.md:1696:    47	Rank-1
./reviews/deep_13.md:1799:   150	person re-identification via dual-constrained top-ranking,” in
./reviews/deep_13.md:1813:   164	center-constrained top-ranking for visible thermal person reidentification,” IEEE Transactions on Information Forensics
./reviews/deep_13.md:1994:   117	CC-ReID presents additional complexities due to the clothes changing.
./reviews/deep_13.md:2566:175-   175	(Cubuk et al., 2019), the second-ranked method, by 0.3 % in the R1
./reviews/deep_13.md:2676:356:   356	Fig. 9. Comparison of HFIA and baseline in ranking gallery retrieval results and visualization (Selvaraju et al., 2017) of each image from model output feature map.
./reviews/deep_13.md:2680:360:   360	2019), as shown in Fig. 9. The retrieval rank of gallery images demonstrates the great performance of HFIA in identity recognition. First, HFIA
./reviews/deep_13.md:2739:419:   419	tiveness of this strategy, as depicted in Fig. 10. The results clearly show
./reviews/deep_13.md:3099:   367	more eﬀective and adaptive feature fusion, thereby addressing existing
./reviews/deep_13.md:3130:   398	In contrast, complementary logical knowledge captures the ambiguity or uncertainty within samples, thereby enhancing the model’s
./reviews/deep_13.md:3206:   474	approach preserves sample uncertainty and stops the student network
./reviews/deep_13.md:3334:31:    31	Results on MARS: As illustrated in Table 2, the Rank-1 accuracy of
./reviews/deep_13.md:3338:35:    35	(Aich et al., 2021) obtained 86.1 % mAP and 90.3 % Rank-1 using spatiotemporal representation decomposition; CTL (Liu et al., 2021a) integrates essential features with graph convolution to build a multi-scale
./reviews/deep_13.md:3339:36-    36	mAP. This approach achieves an mAP of 86.7 % and a Rank-1 accuracy
./reviews/deep_13.md:3352:49-    49	Overall, our model attains 75.1 % mAP and 84.6 % Rank-1 accuracy on
./reviews/deep_13.md:3356:53-    53	attention-based architectures, ultimately achieving 74.6 % Rank-1 and
./reviews/deep_13.md:3389:110:   110	DTA-Net combined with two branches achieves 90.7 % Rank-1 accuracy and 85.8 % mAP on MARS. In addition, GAE-Net performs best after knowledge distillation, achieving a mAP result of 87.7 % on MARS.
./reviews/deep_13.md:3390:111:   111	The ablation results clearly indicate the eﬀectiveness of each branch.
./reviews/deep_13.md:3541:734-   734	is achieved when T is set to 4. The results of 91.1 % Rank-1, and 87.7 %
./reviews/deep_13.md:3565:816-   816	prefers to rank the matching image higher in the results than the baseline. As a result, the accuracy of the matching results of the corresponding person is improved. Conversely, the baseline model is disturbed by
./reviews/deep_13.md:3873: exited 1 in 0ms:
./reviews/deep_13.md:4029:   147	is achieved when T is set to 4. The results of 91.1 % Rank-1, and 87.7 %
./reviews/deep_13.md:4111:   229	prefers to rank the matching image higher in the results than the baseline. As a result, the accuracy of the matching results of the corresponding person is improved. Conversely, the baseline model is disturbed by
./reviews/deep_13.md:4282:   127	forward an effective foreground attention module named adaptive foreground attention (AFA) to preserve foreground regions
./reviews/deep_13.md:4463:   435	leading to very high memory consumption O(C n ) and computational complexity O(C n ). Therefore, an effective feature
./reviews/deep_13.md:4556:   528	plexity O(nC 2.5 ) and space complexity
./reviews/deep_13.md:4625:   597	the response value is larger than an adaptive threshold. In this
./reviews/deep_13.md:4795:   747	the high-order features is clearly larger than the similarity
./reviews/deep_13.md:5103:416-   416	This clearly demonstrates that GAReID achieves a satisfactory
./reviews/deep_13.md:5235:    26	This clearly demonstrates that GAReID achieves a satisfactory
./reviews/deep_13.md:5459:    95	high computational complexity, limiting its application to large-scale
./reviews/deep_13.md:5464:   100	we further employ feature-adaptive matching to compute correlations
./reviews/deep_13.md:5469:   105	relevant training instances while reducing computational complexity.
./reviews/deep_13.md:5475:   111	performance, significantly reducing computational complexity.
./reviews/deep_13.md:5476:   112	Moreover, the feature-adaptive matching method in CGS sampling
./reviews/deep_13.md:5622:   340	hash buckets; (2) using the feature maps adaptive matching method to
./reviews/deep_13.md:5634:   352	the LSH function. In this paper, we utilize the feature maps adaptive
./reviews/deep_13.md:5643:   361	convolution kernel. This is an adaptive convolution kernel of the query
./reviews/deep_13.md:5804:   522	complexity.
./reviews/deep_13.md:5942:   660	subset of another datasets. Rank-1 and average accuracy (mAP) were
./reviews/deep_13.md:6025:95-    95	subset of another datasets. Rank-1 and average accuracy (mAP) were
./reviews/deep_13.md:6047:122-   122	Rank-1 and mAP improved by 3.2% and 2.9%, respectively, compared
./reviews/deep_13.md:6048:123:   123	to QAConv-GS. For Market-1501 → MSMT17, the Rank-1 and mAP
./reviews/deep_13.md:6050:125:   125	(all) → Market-1501, the Rank-1 and mAP improvements were 3.2%
./reviews/deep_13.md:6053:128:   128	1.6%, respectively. For MSMT17 → Market-1501, the Rank-1 and mAP
./reviews/deep_13.md:6061:136:   136	MSMT17 by 3.8% in terms of Rank-1 accuracy, while the mean Average
./reviews/deep_13.md:6064:139-   139	DMN with a 1.0% increase in Rank-1 accuracy. These findings are
./reviews/deep_13.md:6115:200-   200	Rank-1
./reviews/deep_13.md:6195:511:   511	reported or not applicable (%). R1 stands for Rank-1, and MS-all represents MSMT
./reviews/deep_13.md:6213:531-   531	different backbones (%). R1 stands for Rank-1.
./reviews/deep_13.md:6322:756:   756	with Market-1501 → CUHK03 for Rank-1 and mAP, respectively, compared to Cluster. With Market-1501 → MSMT17, they improved by
./reviews/deep_13.md:6324:758:   758	improvements in Rank-1 and mAP were 4.2% and 3.4% respectively In
./reviews/deep_13.md:6325:759:   759	MSMT17 (all) → CUHK03, the improvements were 3.5% in Rank-1 and
./reviews/deep_13.md:6374:814:   814	Comparison of model complexity and time cost when training in Market1501 dataset.
./reviews/deep_13.md:6425:933:   933	with GSANet as the backbone, compared to ResNet, the Rank-1 and
./reviews/deep_13.md:6427:935:   935	→ CUHK03-NP task. On the MSMT17 → Market-1501 task, the Rank1 and mAP are improved by 5.2% and 4.4%, respectively. Compared
./reviews/deep_13.md:6428:936:   936	to HRNet, on the MSMT17 → CUHK03-NP task, the Rank-1 and mAP
./reviews/deep_13.md:6430:938:   938	Market-1501 task, the Rank-1 and mAP are improved by 1.4% and
./reviews/deep_13.md:6443:951-   951	4.5. Comparison of model complexity and time cost
./reviews/deep_13.md:6444:952:   952	Fig. 8. Average Rank-1 and average mAP (%) performance with (a) different batch
./reviews/deep_13.md:6447:955-   955	In addition to accuracy, we also compared the complexity and time
./reviews/deep_13.md:6456:964-   964	be seen that our model has the highest complexity and longest training
./reviews/deep_13.md:6468:976-   976	significantly reducing the computational complexity.
./reviews/deep_13.md:6476:984-   984	CUHK03. Average Rank-1 and average mAP are used as performance
./reviews/deep_13.md:6478:986:   986	be observed from Fig. 8(a) that Rank-1 and mAP increase within a
./reviews/deep_13.md:6496:1008-  1008	complexity. Moreover, the mainstream training samples are limited
./reviews/deep_13.md:6541:1068-  1068	[12] Y. Wang, Z. Chen, F. Wu, G. Wang, Person re-identification with cascaded
./reviews/deep_13.md:6589:    71	Re-ID is a retrieval task with a more complex scenario. This complexity
./reviews/deep_13.md:6617:    99	[14], and ranking optimization techniques [15], with the aim of elevating the performance of Re-ID systems. During the training phase, classiﬁcation loss [16] or triplet loss [17] is employed to optimize the neural
./reviews/deep_13.md:6657:   139	ranking task like Re-ID.
./reviews/deep_13.md:6673:   155	adaptively determines the update direction based on variations in the
./reviews/deep_13.md:6817:   343	(Fig. 3), from an experimental perspective: the current gradient direction shows nearly complete inconsistency with the global aggregated gradient direction. This inconsistency causes severe oscillations
./reviews/deep_13.md:6818:   344	in the update direction during early iterations, indicating substantial
./reviews/deep_13.md:6908:   434	during the initial iterations, preventing violent ﬂuctuations in update directions during the early attack phase. This represents a key distinction
./reviews/deep_13.md:6911:   437	Why does GAGA enhance gradient consistency during early attack stages, and how does this improved consistency boost adversarial transferability?
./reviews/deep_13.md:6937:   463	Evaluation Metric. Rank-K and mean Average Precision (mAP) are
./reviews/deep_13.md:6938:   464	two standard evaluation metrics in person Re-ID. Rank-K measures the
./reviews/deep_13.md:6939:   465	probability that the true match appears within the top-K retrieved candidates, reﬂecting the system’s retrieval accuracy at a given rank threshold
./reviews/deep_13.md:6940:   466	(e.g., Rank-1, Rank-5). The mAP quantiﬁes the overall retrieval performance by computing the mean of Average Precision (AP) scores across
./reviews/deep_13.md:6942:   468	each rank position. In attacks on Re-ID tasks, lower Rank-K and mAP
./reviews/deep_13.md:7030:   556	the attack exhibits high randomness, leading to instability in the update
./reviews/deep_13.md:7032:   558	the update direction of the current gradient, thereby enhancing transferability. Our study aims to accelerate the convergence process, eﬀectively mitigating the instability of attack directions in the initial phase.
./reviews/deep_13.md:7034:   560	the early and late stages of the attack.
./reviews/deep_13.md:7039:   565	with the global aggregated gradient during early iterations, rapidly improving gradient consistency and achieving early convergence.
./reviews/deep_13.md:7083:/bin/zsh -lc "pdftotext -f 8 -l 12 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|gradient consistency|consistency|global aggregated|GAGA|Table|Fig|iterations|early|late|transfer|black-box|white-box|Visualization|parameter|inner|m |beta|Input transformation|ensemble\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_13.md:7089:6-     6	Comparison of Rank-1 (%) and mAP (%) with the state-of-the-art methods on SYSU-MM01 and
./reviews/deep_13.md:7126:247:   247	GAGA to attack the Swinv2 model on the MSMT17 dataset, the Rank-1
./reviews/deep_13.md:7137:258:   258	improvement when combined with diﬀerent input transformation techniques. Notably, when GAGA is integrated with SSA [46], the Rank1 recognition rate dramatically drops from 30.92 % to 14.64 %. This
./reviews/deep_13.md:7160:281-   281	an average Rank-1 score of 22.92 % against the four defense models. The
./reviews/deep_13.md:7161:282-   282	second-best performing TPA [5] method only reaches 35.55 % Rank-1
./reviews/deep_13.md:7332:1180-  1180	[28] H. Wang, G. Wang, Y. Li, D. Zhang, L. Lin, Transferable, controllable, and inconspicuous adversarial attacks on person re-identiﬁcation with deep mis-ranking, in:
./reviews/deep_13.md:7341:1189-  1189	ranking disruption for visible-infrared person re-identiﬁcation, Adv. Neural Inf. Process. Syst. 37 (2024) 135043–135061.
./reviews/deep_13.md:7354:1202-  1202	ranking results.
./reviews/deep_13.md:7721:3. **机制怎么长出来**：CGS 每个 epoch 用 Spherical-LSH 先把相近类别放进 bucket，再用 feature-map adaptive matching 找邻近类别，构造相关 hard batch。GSANet 保持高分辨率流，GRSR 用稀疏全局相关像素重构特征，提升特征质量。特征更好会让下一轮 CGS 更好，CGS 更好又提供更有价值样本。这个“互相增强”的逻辑成立，但模块数量较多，绑定不如 GAReID 干净。
./reviews/deep_13.md:7722:4. **证据闭环**：采样对比显示 CGS 好于 random、PK、cluster，并且比已有 GS 快很多，Market 上采样 0.1 秒对 4 秒，MSMT(all) 上 1 秒对 40 秒。CGS 相比 Cluster 在 Market 到 CUHK03 上提升约 3.0 Rank-1 和 2.8 mAP，在 Market 到 MSMT17 上提升约 3.9 和 2.9。CGS、GRSR、GSANet 的组合消融显示完整模型最好。采样可视化证明 CGS 找到衣服颜色、局部外观相近的邻近类别。
./reviews/deep_13.md:7726:1. **触发观察**：ReID 是 open-set ranking，分类攻击不能直接搬。已有 ReID 攻击黑盒迁移差，作者观察到早期迭代里 current gradient 和 global aggregated gradient 一致性很低，更新方向随机震荡，导致迁移性不足。
./reviews/deep_13.md:7768:3. **机制怎么长出来**：CGS 每个 epoch 用 Spherical-LSH 先把相近类别放进 bucket，再用 feature-map adaptive matching 找邻近类别，构造相关 hard batch。GSANet 保持高分辨率流，GRSR 用稀疏全局相关像素重构特征，提升特征质量。特征更好会让下一轮 CGS 更好，CGS 更好又提供更有价值样本。这个“互相增强”的逻辑成立，但模块数量较多，绑定不如 GAReID 干净。
./reviews/deep_13.md:7769:4. **证据闭环**：采样对比显示 CGS 好于 random、PK、cluster，并且比已有 GS 快很多，Market 上采样 0.1 秒对 4 秒，MSMT(all) 上 1 秒对 40 秒。CGS 相比 Cluster 在 Market 到 CUHK03 上提升约 3.0 Rank-1 和 2.8 mAP，在 Market 到 MSMT17 上提升约 3.9 和 2.9。CGS、GRSR、GSANet 的组合消融显示完整模型最好。采样可视化证明 CGS 找到衣服颜色、局部外观相近的邻近类别。
./reviews/deep_13.md:7773:1. **触发观察**：ReID 是 open-set ranking，分类攻击不能直接搬。已有 ReID 攻击黑盒迁移差，作者观察到早期迭代里 current gradient 和 global aggregated gradient 一致性很低，更新方向随机震荡，导致迁移性不足。
./reviews/deep_17.md:140:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_17.md:152: exited 1 in 0ms:
./reviews/deep_17.md:157: exited 1 in 0ms:
./reviews/deep_17.md:188:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_17.md:217:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_17.md:220:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_17.md:221:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_17.md:298:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_17.md:302:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_17.md:316:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_17.md:322:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_17.md:456:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_17.md:561:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_17.md:578:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_17.md:581:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_17.md:590:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_17.md:622:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_17.md:652:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_17.md:868:   256	single-stream structure requires early fusion of the two modalities, which cannot be directly
./reviews/deep_17.md:888:   276	are then linearly projected through linear projection layer, with bit-set padding and an extra image
./reviews/deep_17.md:1093:   359	knowledge acquired from prior domains, facilitating adaptive parameter
./reviews/deep_17.md:1290:   524	the dataset’s complexity, featuring signiﬁcant illumination variations,
./reviews/deep_17.md:1386:    50	adopt Mean Average Precision (mAP) and Cumulative Matching Characteristics at Rank-1 and
./reviews/deep_17.md:1387:    51	Rank-5 for performance evaluation on downstream Re-ID task.
./reviews/deep_17.md:1407:    71	Rank-1 ↑
./reviews/deep_17.md:1411:    75	Rank-1 ↑
./reviews/deep_17.md:1458:   122	MMM Improves MMET Especially in Rank-1 Accuracy. According to Table 1, while MMET already
./reviews/deep_17.md:1460:   124	in Rank-1 accuracy is +8.1% and +9.6%, respectively; the improvement in mAP is +2.1% and +4.7%,
./reviews/deep_17.md:1461:   125	respectively. The improvement in Rank-1 is larger than in mAP accuracy. In fact, Rank-1 accuracy
./reviews/deep_17.md:1495:   159	mAP ↑ Rank-1 ↑ Rank-5 ↑
./reviews/deep_17.md:1539:   203	Rank-1
./reviews/deep_17.md:1543:   207	Rank-1
./reviews/deep_17.md:1547:   211	Rank-1
./reviews/deep_17.md:2412:106:   106	to propose an adaptive knowledge accumulation strategy, encompassing both knowledge representation and knowledge manipulation aspects. However, existing methods and tasks primarily focus
./reviews/deep_17.md:2450:226:   226	framework to adaptively learn modality-sharable features for VIReID. Typical methods[9, 12, 50, 51, 64] adopt a two-stream network,
./reviews/deep_17.md:2465:259:   259	[17] adopts an adaptive prompt generator to generate appropriate
./reviews/deep_17.md:2472:282:   282	the task identifier 𝑡 as the supervise information to linearly transform the image tokens into the adaptive prompts 𝑃𝐼 , which can
./reviews/deep_17.md:2485:409:   409	that directly optimizing the prompt 𝑃𝐾 could result in instability
./reviews/deep_17.md:2494:496:   496	as the prompt pool expands dynamically. To address these challenges, we propose an instance-aware adaptive prompting strategy,
./reviews/deep_17.md:2583:   171	a query-key mechanism using query tokens to adaptively
./reviews/deep_17.md:2630:   218	were insufficient for addressing the complexities of real-world
./reviews/deep_17.md:2638:   226	framework to adaptively learn modality-sharable features for VIReID. Typical methods[9, 12, 50, 51, 64] adopt a two-stream network,
./reviews/deep_17.md:2671:   259	[17] adopts an adaptive prompt generator to generate appropriate
./reviews/deep_17.md:2694:   282	the task identifier 𝑡 as the supervise information to linearly transform the image tokens into the adaptive prompts 𝑃𝐼 , which can
./reviews/deep_17.md:2821:   409	that directly optimizing the prompt 𝑃𝐾 could result in instability
./reviews/deep_17.md:2910:   495	is unavailable .Secondly, the memory footprint increases linearly
./reviews/deep_17.md:2911:   496	as the prompt pool expands dynamically. To address these challenges, we propose an instance-aware adaptive prompting strategy,
./reviews/deep_17.md:3020:   605	For the metric of CMC, we calculated the proportion of the number of correctly retrieved images using the Rank-1 accuracy (R-1).
./reviews/deep_17.md:3183:289:   289	and Transformers inevitably increases model complexity and computational costs. Additionally,
./reviews/deep_17.md:3216:   289	and Transformers inevitably increases model complexity and computational costs. Additionally,
./reviews/deep_17.md:3272:   345	adaptively emphasized by utilizing these selection values to reweight them:
./reviews/deep_17.md:3575:   278	on the SYSU-MM01 dataset. LAReViT achieves 76.71% Rank-1 and 72.95% mAP in the All-search
./reviews/deep_17.md:3589:   292	Rank-1 Rank-10 Rank-20 mAP Rank-1 Rank-10 Rank-20 mAP
./reviews/deep_17.md:3862:   558	Rank-k accuracy (%) and mAP (%) are reported. The highest value of each evaluation metrics is underlined.
./reviews/deep_17.md:3864:   560	mode. In addition, in the Indoor-search mode, LAReViT achieves 84.22% Rank-1 and 86.26% mAP.
./reviews/deep_17.md:3868:   564	in terms of Rank-n accuracy and mAP. Our method outperforms DGRNet [26] that utilizes dynamic
./reviews/deep_17.md:3869:   565	weighted gradient reversal by 5.18% in Rank-1 accuracy and 4.91% in mAP. Additionally, compared
./reviews/deep_17.md:3871:   567	the proposed LAReViT achieves a performance gain of 6.12% in Rank-1 accuracy and 5.46% in mAP.
./reviews/deep_17.md:3876:   572	Rank-1 accuracy and 2.39% in mAP. While TRR considered the correlation between modalities
./reviews/deep_17.md:3882:   578	in terms of Rank-n accuracy and mAP. These analyses evidence the effectiveness of our proposed
./reviews/deep_17.md:4077:   349	mask patch is flattened and linearly projected, resulting in
./reviews/deep_17.md:4102:   374	transformer-based baselines also face the difficulty of capturing fine-grained information. In other words, because of the
./reviews/deep_17.md:4424:   693	73.3%/85.8% Rank-1 accuracy and a 62.3%/79.5% mAP on
./reviews/deep_17.md:4451:   720	Rank-1 accuracy and a 88.2%/81.2% mAP on the Market1501
./reviews/deep_17.md:4456:   725	we can see that our approach attains a Rank-1 accuracy
./reviews/deep_17.md:4504:   773	3.5% enhancement in Rank-1 accuracy and a 3.8% enhancement in mAP, indicating that MIE can avoid being affected
./reviews/deep_17.md:4507:   776	HFA module, resulting in a 5.6% enhancement in Rank-1
./reviews/deep_17.md:4512:   781	3.6% enhancement in Rank-1 accuracy and a 3.8% enhancement in mAP, indicating that the DFC can successfully learn
./reviews/deep_17.md:4525:   794	MAHATMA with a 71.9% Rank-1 accuracy and a 61.4% mAP.
./reviews/deep_17.md:4528:   797	a Rank-1 accuracy of 73.3% and a mAP of 62.3%, which
./reviews/deep_17.md:4626:    97	computational complexity increases substantially as the number of tasks
./reviews/deep_17.md:4821:   152	ViT. The former adaptively recalibrates channel-wise feature responses by explicitly modeling
./reviews/deep_17.md:5028:   167	to produce competitive results on large-scale datasets. The main reason is that these early works
./reviews/deep_17.md:5044:   183	the Re-ID model by ranking each positive sample ahead of the negative samples according to the
./reviews/deep_17.md:5047:   186	multimodal representation learning or optimizing the order of positive samples with low rank
./reviews/deep_17.md:5072:   211	an adaptive part division model to better extract local features for person Re-ID. In addition, Xiang
./reviews/deep_17.md:5169:968:   968	shown in Table 3, the Rank-1 diﬀerences between training and testing sets remain consistently small across all datasets. The maximum observed gap does not exceed 5%, indicating that the model does not rely
./reviews/deep_17.md:5250:855:   855	Figure 4: Incremental average Rank-1 after each training
./reviews/deep_17.md:5330:773:   773	3.5% enhancement in Rank-1 accuracy and a 3.8% enhancement in mAP, indicating that MIE can avoid being affected
./reviews/deep_17.md:5332:776:   776	HFA module, resulting in a 5.6% enhancement in Rank-1
./reviews/deep_17.md:5335:781:   781	3.6% enhancement in Rank-1 accuracy and a 3.8% enhancement in mAP, indicating that the DFC can successfully learn
./reviews/deep_17.md:5360:881:   881	and Rank-1 are both optimal. When λ surpasses 1, the DFC
./reviews/deep_17.md:5459:633:   633	on the SYSU-MM01 dataset. LAReViT achieves 76.71% Rank-1 and 72.95% mAP in the All-search
./reviews/deep_17.md:5462:915:   915	mode. In addition, in the Indoor-search mode, LAReViT achieves 84.22% Rank-1 and 86.26% mAP.
./reviews/deep_17.md:5464:922:   922	the proposed LAReViT achieves a performance gain of 6.12% in Rank-1 accuracy and 5.46% in mAP.
./reviews/deep_17.md:5466:933:   933	in terms of Rank-n accuracy and mAP. These analyses evidence the effectiveness of our proposed
./reviews/deep_17.md:5470:1118:  1118	research on the RegDB dataset. In the V-I mode, LAReViT achieves a Rank-1 accuracy of 90.4% and
./reviews/deep_17.md:5471:1119:  1119	an mAP of 84.7%. In the I-V mode, LAReViT achieves a Rank-1 accuracy of 90.5% and an mAP of
./reviews/deep_17.md:5472:1123:  1123	proposed LAReViT in the V-I search mode. However, in the I-V search mode, their Rank-1 and
./reviews/deep_17.md:5515:1440:  1440	4.4.6 Analysis of Parameter 𝜆 of Loss Function. The Rank-1 accuracy and mAP results of LAReViT
./reviews/deep_17.md:5735:   855	Figure 4: Incremental average Rank-1 after each training
./reviews/deep_17.md:5820:   940	frequencies shown in the right-hand plot becomes significantly regular after training. Samples from different domains can adaptively
./reviews/deep_17.md:5822:   942	designed query-key mechanism enables the model to adaptively
./reviews/deep_17.md:5939:  1285	Rank-1
./reviews/deep_17.md:5948:  1294	Rank-10
./reviews/deep_17.md:5963:  1285	Rank-1
./reviews/deep_17.md:5972:  1294	Rank-10
./reviews/deep_17.md:5981:  1303	Rank-20
./reviews/deep_17.md:6206:1602:  1602	seen domains before the current training domain) to measure the antiforgetting ability of models. Both 𝑠̄ and 𝑚̄ are measured with Rank-1
./reviews/deep_17.md:6234:1785:  1785	Rank-1 and 4.45 mAP under the IR to VIS test mode (in Table 3),
./reviews/deep_17.md:6237:1790:  1790	our method (Base.+CDCN) achieves an elevation of 3.8 Rank-1 and
./reviews/deep_17.md:6250:1822:  1822	the VIS to IR test mode (in Table 2), and further accomplishes increases of 4.17 Rank-1 and 3.07 mAP when tested under the IR to VIS
./reviews/deep_17.md:6306:  1555	Rank-1 and mAP, respectively, in 𝑠̄ when compared to LL methods,
./reviews/deep_17.md:6307:  1556	while outperforming the Lifelong ReID methods by 6.95 Rank-1 and
./reviews/deep_17.md:6309:  1558	achieves improvements of 6.62 Rank-1 and 4.73 mAP over the average
./reviews/deep_17.md:6316:  1565	achieving increases of 8.63 Rank-1 and 7.82 mAP, respectively, in ̄
./reviews/deep_17.md:6319:  1568	them with enhancements of 8.41 Rank-1 and 7.05 mAP. While we
./reviews/deep_17.md:6320:  1569	achieve 6.54 Rank-1 and 4.52 mAP increases under the IR to VIS test
./reviews/deep_17.md:6366:  1779	including an increase of 15.00 Rank-1 and 15.24 mAP for RegDB, 6.12
./reviews/deep_17.md:6367:  1780	Rank-1 and 5.34 mAP for SYSU-MM01, 4.1 Rank-1 and 3.57 mAP for
./reviews/deep_17.md:6368:  1781	LLCM, and 3.89 Rank-1 and 3.02 mAP for HITSZ-VCM. Meanwhile,
./reviews/deep_17.md:6370:  1783	average adaptation performance ̄𝑠, realizing a boost of 6.67 Rank-1
./reviews/deep_17.md:6372:  1785	Rank-1 and 4.45 mAP under the IR to VIS test mode (in Table 3),
./reviews/deep_17.md:6377:  1790	our method (Base.+CDCN) achieves an elevation of 3.8 Rank-1 and
./reviews/deep_17.md:6379:  1792	of 2.83 Rank-1 and 1.64 mAP when tested under the inverse IR to
./reviews/deep_17.md:6408:  1821	method yields enhancements of 4.76 Rank-1 and 4.43 mAP under
./reviews/deep_17.md:6409:  1822	the VIS to IR test mode (in Table 2), and further accomplishes increases of 4.17 Rank-1 and 3.07 mAP when tested under the IR to VIS
./reviews/deep_17.md:6486:4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。
./reviews/deep_17.md:6563:4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。
./reviews/deep_19.md:129:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_19.md:143:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_19.md:160:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_19.md:163:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_19.md:172:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_19.md:204:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_19.md:234:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_19.md:316:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_19.md:364:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_19.md:372:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_19.md:436:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_19.md:444:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_19.md:455:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_19.md:552:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_19.md:581:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_19.md:584:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_19.md:585:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_19.md:662:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_19.md:666:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_19.md:680:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_19.md:686:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_19.md:717:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_19.md:937:establish early methodologies for viewpoint-invariant feature extraction, emphasizing handcrafted descriptors robust to pose variations and illumination changes. The ﬁeld underwent a paradigm shift
./reviews/deep_19.md:976:introduces alignment noise. This underscores the need for adaptive
./reviews/deep_19.md:1314:Rank-1
./reviews/deep_19.md:1353:Rank-1
./reviews/deep_19.md:1415:Rank-1
./reviews/deep_19.md:1443:Rank-1
./reviews/deep_19.md:1457:Rank-1
./reviews/deep_19.md:1471:Rank-1
./reviews/deep_19.md:1529:Rank-1 in CMC measures the probability of correctly identifying the
./reviews/deep_19.md:1530:target ID within the Rank-1 results, and mAP quantiﬁes overall retrieval precision by averaging precision across recall thresholds.
./reviews/deep_19.md:1538:Rank-1
./reviews/deep_19.md:1582:For the PRCC dataset, our method demonstrates exceptional performance in the same-clothes setting, achieving a perfect Rank-1 of
./reviews/deep_19.md:1584:our method attains a Rank-1 of 71.6% and an mAP of 67.6%, surpassing the next best method (CCAF) by 1.2% in Rank-1 and 3.9%
./reviews/deep_19.md:1586:On the LTCC dataset, our method achieves a Rank-1 of 85.2% and
./reviews/deep_19.md:1588:setting, our method attains a Rank-1 of 52.3% and an mAP of 29.9%,
./reviews/deep_19.md:1590:Rank-1 accuracy.
./reviews/deep_19.md:1593:method reaches 69.7% Rank-1 accuracy and 29.5% mAP, surpassing
./reviews/deep_19.md:1676:On the PRCC dataset, our method achieves a Rank-1 accuracy
./reviews/deep_19.md:1679:advancements, with Rank-1 accuracy surging by 9.4% (52.3% vs.
./reviews/deep_19.md:1681:the VC-Clothes dataset, our approach raises Rank-1 accuracy from
./reviews/deep_19.md:1683:The most striking gains emerge on DeepChange, where Rank-1
./reviews/deep_19.md:1686:75.6% Rank-1 and 55.8% mAP.
./reviews/deep_19.md:1703:dataset, Rank-1 for CC increases by 2.0% (from 61.4% to 63.4%).
./reviews/deep_19.md:1704:On the LTCC dataset, Rank-1 for CC improves by 2.0% (from 42.9%
./reviews/deep_19.md:1707:performance is further enhanced: on the PRCC dataset, Rank-1 for
./reviews/deep_19.md:1758:Rank-1 (%)
./reviews/deep_19.md:1773:dataset, Rank-1 for CC improves to 46.9%, and mAP increases to
./reviews/deep_19.md:1775:Incorporating the PIE module into the baseline also yields signiﬁcant gains: on the PRCC dataset, Rank-1 for CC reaches 66.1%, and
./reviews/deep_19.md:1776:mAP improves to 63.4%; on the LTCC dataset, Rank-1 for CC is
./reviews/deep_19.md:1778:Combining all three components achieves the best overall performance: on the PRCC dataset, Rank-1 accuracy for CC reaches
./reviews/deep_19.md:1779:71.6%, and mAP climbs to 67.6%; on the LTCC dataset, Rank-1 accuracy for CC hits 52.3%, and mAP reaches 29.9%. These results
./reviews/deep_19.md:1786:features with a Rank-1 accuracy of 71.6% compared to 68.8%. On
./reviews/deep_19.md:1787:the LTCC dataset, the superiority of local features is even more pronounced. For SC scenarios, local features yield 85.2% Rank-1 and
./reviews/deep_19.md:1789:advantage with a Rank-1 accuracy of 52.3%, compared to 46.4% for
./reviews/deep_19.md:1794:Comparison with different numbers of captions. Table 6 compares performance with varying caption numbers on PRCC CC settings. Optimal results (65.9% Rank-1) are achieved using two captions, outperforming single-caption (63.4%) and multi-caption conﬁgurations (≤ 64.2%). This demonstrates that two captions optimally balance diversity and relevance: they mitigate single-model
./reviews/deep_19.md:1822:Rank-1(%)
./reviews/deep_19.md:1841:during inference, achieving 14.12 ms per image, signiﬁcantly outperforming GI-ReID, AIM, and MADE. Notably, conventional approaches consistently reveal an inherent speed-accuracy compromise—illustrated by MADE attaining 67.5% Rank-1 accuracy at
./reviews/deep_19.md:1854:and our approach. Each row presents the ranking for a query image
./reviews/deep_19.md:1855:(leftmost), followed by Rank-1 to Rank-10 retrieved images from left to
./reviews/deep_19.md:2248:especially on low difficulty datasets. Therefore, in this paper,
./reviews/deep_19.md:2250:difficulty and suppress model overfitting. Following Eq.1, the
./reviews/deep_19.md:2697:be ranked in terms of difficulty as follows: Market1501 <
./reviews/deep_19.md:2700:on difficult samples. This adaptive adjustment allows the
./reviews/deep_19.md:2726:domain adaptive re-ID[J]. arXiv preprint arXiv:2008.10313, 2020.
./reviews/deep_19.md:2733:clustering for domain adaptive person re-identification[C]//Proceedings
./reviews/deep_19.md:2737:hybrid memory for domain adaptive object re-id[J]. Advances in neural
./reviews/deep_19.md:2744:domain adaptive person re-id[C]//Proceedings of the AAAI conference
./reviews/deep_19.md:2747:adaptive person re-identification[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 5310-5319.
./reviews/deep_19.md:2895:the proposed approach, achieving a Rank-1/mAP accuracy of 44.6%/22.7%, 58.3%/57.9%, 87.2%/84.0%.
./reviews/deep_19.md:3519:Comparison of Rank-k and mAP Performance with State-Of-The-Art (SOTA) Methods in LTCC and PRCC. The best and second-best results are highlighted in bold and underline.
./reviews/deep_19.md:3534:Rank1
./reviews/deep_19.md:3538:Rank1
./reviews/deep_19.md:3542:Rank1
./reviews/deep_19.md:3546:Rank1
./reviews/deep_19.md:3740:Comparison of Rank-k and mAP Performance with State-Of-The-Art (SOTA) Methods in VC-Clothes. The best and second-best results are
./reviews/deep_19.md:3749:Rank1
./reviews/deep_19.md:3753:Rank1
./reviews/deep_19.md:3843:45.3% of Rank-1 through feature augmentation learning to augment
./reviews/deep_19.md:3845:shows improvements in both Rank-1 and mAP compared to the current
./reviews/deep_19.md:3847:Rank-1 in cloth-changing. As it can be seen in Table 1, our method
./reviews/deep_19.md:3849:respectively on rank-1 and increases by 3.7% and 4.3% compared with
./reviews/deep_19.md:3856:achieve 99.8% and 98.2% accuracy on Rank-1, respectively. In the
./reviews/deep_19.md:3858:2018) by 16.5% and 19.2% in Rank-1 and mAP, respectively. Comparing the disentanglement methods, our method surpasses the CAL (Gu
./reviews/deep_19.md:3859:et al., 2022) by 3.1% and 2.1% in Rank-1 and mAP, respectively. Our
./reviews/deep_19.md:3871:accuracy on Rank-1, respectively. However, they performed poorly in
./reviews/deep_19.md:3873:into account. Compared with the ISP (Zhu et al., 2020), Rank-1 is
./reviews/deep_19.md:3897:Rank1
./reviews/deep_19.md:3901:Rank1
./reviews/deep_19.md:3944:Rank1
./reviews/deep_19.md:3948:Rank1
./reviews/deep_19.md:3972:method achieves at least 2% increase over baseline in both Rank-1
./reviews/deep_19.md:3974:method, our method improves by 1.1% on Rank-1 in same-clothes.
./reviews/deep_19.md:3975:Compared to the best competitor CCFA (Han et al., 2023), Rank-1 and
./reviews/deep_19.md:3988:et al., 2020) show great competitiveness, achieving Rank-1 and mAP
./reviews/deep_19.md:3990:setting, our method achieves 87.2% and 84.0% respectively on Rank-1
./reviews/deep_19.md:3993:et al., 2023), our method is improved by 4.5% and 13.7% respectively on Rank-1 and mAP. Compared with the GI-ReID (Jin et al.,
./reviews/deep_19.md:3994:2022) method based on gait, Rank-1 and mAP are improved by 22.7%
./reviews/deep_19.md:4016:setting, Rank-1 and mAP are increased by 1.0% and 1.4% respectively,
./reviews/deep_19.md:4022:appearance information. In the mask branch, Rank-1 and mAP are increased by 2.5% and 1.3% respectively in the general setting, indicating
./reviews/deep_19.md:4038:branch. It increases Rank-1 and mAP to 42.1% and 18.7% respectively,
./reviews/deep_19.md:4081:Rank-1 and 4.0%/18.3% in mAP on PRCC and LTCC under
./reviews/deep_19.md:4525:PRCC [9] and LTCC [4], using Rank-1 accuracy and mean
./reviews/deep_19.md:4613:LTCC dataset, the camera-adaptive inference strategy from
./reviews/deep_19.md:4616:The overall experimental results are presented in Table.1. It can be clearly observed that our proposed method
./reviews/deep_19.md:4619:[24] by 3.2% in Rank-1 and 4.0% in mAP; on LTCC, it surpasses DIFIF by 13.0% in Rank-1 and 18.3% in mAP. We
./reviews/deep_19.md:4623:at the attribute level improves Rank-1/mAP by 4.2%/3.5%,
./reviews/deep_19.md:4625:Identity-Relevant Attributes (A P ). Enhancing identityrelevant attributes at the attribute level improves Rank-1/mAP
./reviews/deep_19.md:4629:Rank-1 by 2.0%, validating multi-granularity textual disentanglement. Masked Branch Disentanglement (M LB).
./reviews/deep_19.md:4631:reduces Rank-1 by 0.6%, indicating residual clothing interference. Hyperparameter Impact. As shown in Fig.2,
./reviews/deep_19.md:4632:mAP remains stable while Rank-1 peaks at a = 0.1. Performance degrades when prompt length exceeds 30, suggesting
./reviews/deep_19.md:5183:providing this early version to give early visibility of the article. Please note that Elsevier’s sharing
./reviews/deep_19.md:5202:features by adaptively capturing multi-granularity temporal clues. Specifically, we adaptively segment the video features
./reviews/deep_19.md:5204:at the same level are considered as graph nodes, which are then adaptively connected based on feature similarity.
./reviews/deep_19.md:5205:By dynamically aggregating features from neighboring nodes in the same graph, DHGN is able to adaptively capture
./reviews/deep_19.md:5208:similarity matrix. Extensive experiments on four benchmarks clearly demonstrate the effectiveness of the proposed
./reviews/deep_19.md:5240:in other frames. Therefore, processing these parts separately and adaptively capturing temporal clues can model
./reviews/deep_19.md:5245:graph neural networks to model fine-grained temporal features. While these methods can adaptively extract temporal clues of different lengths, they cannot dynamically
./reviews/deep_19.md:5253:content and adaptively models the temporal features of
./reviews/deep_19.md:5258:GC layer adaptively captures temporal clues for each body
./reviews/deep_19.md:5263:are then clustered based on feature similarity into adaptive
./reviews/deep_19.md:5267:only mutually similar nodes are connected. This strategy yields a similarity-adaptive and dynamically updated
./reviews/deep_19.md:5273:(GNNs) to adaptively model the corresponding temporal features. Moreover, most of the approaches use only
./reviews/deep_19.md:5277:different body parts. DHGN can adaptively aggregate
./reviews/deep_19.md:5296:can cope with non-Euclidean data and adaptively capture temporal
./reviews/deep_19.md:5321:• We propose a novel framework, DHGN, for videobased person re-identification. DHGN adaptively
./reviews/deep_19.md:5327:and cannot adaptively capture temporal clues of different lengths. Transformer-based approaches have demonstrated promising potential due to their powerful learning
./reviews/deep_19.md:5335:a revised ranking list by using the top-k nearest neighbors from the initial ranking list as new queries. Although
./reviews/deep_19.md:5338:attempted to address the re-ranking problem by utilizing
./reviews/deep_19.md:5341:al. [27] proposed the k-mutual-neighbor method, which revises the initial ranking list by calculating the Jaccard distance between samples. Recently, Cheb-GR [28] replaced
./reviews/deep_19.md:5342:traditional k-nearest neighbor search with graph convolutions for re-ranking, significantly reducing computational
./reviews/deep_19.md:5358:can extract fine-grained pedestrian features, they are unable to adaptively segment features based on occlusion
./reviews/deep_19.md:5359:versus body part, nor can they adaptively capture temporal clues. In addition, although these methods divide
./reviews/deep_19.md:5362:a hierarchical adaptive graph construction strategy. Instead of using fixed horizontal strips as nodes, we form
./reviews/deep_19.md:5373:layer is inserted at the end of ResNet to process the extracted features and adaptively capture global temporal
./reviews/deep_19.md:5381:The objective of the GC layer is to adaptively capture
./reviews/deep_19.md:5389:2.3. Re-ranking for Re-ID
./reviews/deep_19.md:5390:Re-ranking aims to modify the original sequence of
./reviews/deep_19.md:5391:rankings through an automated and unsupervised approach. Currently, various Re-ranking methods have been
./reviews/deep_19.md:5400:at stages 2 and 3 of the ResNet to adaptively extract multi-granularity features, which are then reconstructed and passed to the next stage.
./reviews/deep_19.md:5405:a semantically consistent manner, we construct p partconsistent graphs according to the adaptive grouping results from horizontal partition and clustering. Each graph
./reviews/deep_19.md:5419:adaptive edges. Two nodes are connected only when they
./reviews/deep_19.md:5423:updated, resulting in an adaptive graph structure. Mathematically, the edge set is defined as
./reviews/deep_19.md:5431:adaptive semantic regions.
./reviews/deep_19.md:5450:Graph propagation. After constructing the partconsistent adaptive graphs, we perform graph propagation
./reviews/deep_19.md:5572:two branches. By utilizing the generated attention weights, the features of each branch are adaptively reweighted and fused.
./reviews/deep_19.md:5588:Graph construction. Similarly, we adaptively capture global temporal clues by constructing graph and using
./reviews/deep_19.md:5592:represents a node. Finally, we adaptively connect these
./reviews/deep_19.md:5798:method is 89.1% and the Rank-1 accuracy is 91.2%. In
./reviews/deep_19.md:5873:at Rank-1 (R1), Rank-5 (R5), and Rank-10 (R10). To ensure a fair
./reviews/deep_19.md:5947:Rank-1
./reviews/deep_19.md:5949:Rank-5
./reviews/deep_19.md:5951:Rank-10
./reviews/deep_19.md:5955:Rank-1
./reviews/deep_19.md:5957:Rank-5
./reviews/deep_19.md:5959:Rank-10
./reviews/deep_19.md:6113:Rank-1
./reviews/deep_19.md:6115:Rank-5
./reviews/deep_19.md:6117:Rank-10
./reviews/deep_19.md:6184:a mean average precision (mAP) of 90.5% and a Rank-1 accuracy of
./reviews/deep_19.md:6185:95.9% on Market-1501, along with a mAP of 58.9% and a Rank-1
./reviews/deep_19.md:6189:method attains mAP values of 90.6% and 47.0%, as well as Rank-1
./reviews/deep_19.md:6207:Rank-1
./reviews/deep_19.md:6209:Rank-5
./reviews/deep_19.md:6211:Rank-10
./reviews/deep_19.md:6299:that 𝑐𝑙𝑏 improved the mAP by 0.3% and 2.0%, and Rank-1 by 0.2%
./reviews/deep_19.md:6305:in mAP, and 0.4% and 1.3% in Rank-1 on the Market1501 and MSMT17
./reviews/deep_19.md:6359:Rank-1
./reviews/deep_19.md:6363:Rank-1
./reviews/deep_19.md:6391:Fig. 4. Retrieval rank-list visualization: green indicates correct recognition, and red indicates incorrect recognition.
./reviews/deep_19.md:6565:method is 89.1% and the Rank-1 accuracy is 91.2%. In
./reviews/deep_19.md:6577:the mAP and Rank-1 metrics of our method are 83.4%
./reviews/deep_19.md:6578:and 90.2%, respectively, both of which achieve optimal results. On the iLIDS-VID dataset, DHGN achieves a Rank1 accuracy of 94.0%, outperforming other ResNet50-based
./reviews/deep_19.md:6583:our method achieves 97.8% Rank-1 accuracy, which is significantly better than other SOTA methods.
./reviews/deep_19.md:6599:results, while an excessively high number of divisions increases the difficulty of clustering.
./reviews/deep_19.md:6616:degrades when K1 > 3. This is due to the difficulty of
./reviews/deep_19.md:6618:adaptive capture of temporal clues.
./reviews/deep_19.md:6629:the graph convolutional layer. “+GFC ” denotes the adaptive capture of global temporal clues using a GFC layer after the baseline extraction of sequence features. It can be
./reviews/deep_19.md:6634:adaptively capture temporal clues at different hierarchical
./reviews/deep_19.md:6637:on the Rank-1 metrics is less pronounced. This may be
./reviews/deep_19.md:6752:mAP Rank-1
./reviews/deep_19.md:6788:mAP Rank-1
./reviews/deep_19.md:6833:mAP Rank-1
./reviews/deep_19.md:6858:Rank-1
./reviews/deep_19.md:6878:Rank-1
./reviews/deep_19.md:6895:mAP Rank-1
./reviews/deep_19.md:6917:mAP Rank-1
./reviews/deep_19.md:6930:mAP Rank-1
./reviews/deep_19.md:6984:mAP Rank-1
./reviews/deep_19.md:7009:mAP Rank-1
./reviews/deep_19.md:7067:mAP Rank-1
./reviews/deep_19.md:7084:mAP Rank-1
./reviews/deep_19.md:7109:on the MARS dataset, the Rank-1 metric shows a slight
./reviews/deep_19.md:7133:mAP Rank-1 mAP Rank-1
./reviews/deep_19.md:7208:mAP Rank-1
./reviews/deep_19.md:7223:mAP Rank-1
./reviews/deep_19.md:7238:Node similarity visualization. To better demonstrate the ability of the proposed method to adaptively
./reviews/deep_19.md:7250:Comparison with other re-ranking methods. Table 7 reports the comparison between different re-ranking
./reviews/deep_19.md:7254:significantly increased computational overhead. In contrast, the proposed SWIM re-ranking strategy achieves the
./reviews/deep_19.md:7257:re-identification re-ranking.
./reviews/deep_19.md:7258:Table 7: Comparison with other re-ranking methods.
./reviews/deep_19.md:7279:mAP Rank-1
./reviews/deep_19.md:7294:mAP Rank-1
./reviews/deep_19.md:7312:mAP Rank-1 mAP Rank-1
./reviews/deep_19.md:7484:existing methods in temporal feature extraction, we propose DHGN to adaptively capture temporal clues at each
./reviews/deep_19.md:7487:feature updating to adaptively capture temporal clues at
./reviews/deep_19.md:7504:exploring for unsupervised domain adaptive person re-id. IEEE
./reviews/deep_19.md:7548:Rank1
./reviews/deep_19.md:7552:Rank1
./reviews/deep_19.md:7595:Rank1
./reviews/deep_19.md:7599:Rank1
./reviews/deep_19.md:7623:method achieves at least 2% increase over baseline in both Rank-1
./reviews/deep_19.md:7625:method, our method improves by 1.1% on Rank-1 in same-clothes.
./reviews/deep_19.md:7626:Compared to the best competitor CCFA (Han et al., 2023), Rank-1 and
./reviews/deep_19.md:7639:et al., 2020) show great competitiveness, achieving Rank-1 and mAP
./reviews/deep_19.md:7641:setting, our method achieves 87.2% and 84.0% respectively on Rank-1
./reviews/deep_19.md:7644:et al., 2023), our method is improved by 4.5% and 13.7% respectively on Rank-1 and mAP. Compared with the GI-ReID (Jin et al.,
./reviews/deep_19.md:7645:2022) method based on gait, Rank-1 and mAP are improved by 22.7%
./reviews/deep_19.md:7667:setting, Rank-1 and mAP are increased by 1.0% and 1.4% respectively,
./reviews/deep_19.md:7673:appearance information. In the mask branch, Rank-1 and mAP are increased by 2.5% and 1.3% respectively in the general setting, indicating
./reviews/deep_19.md:7689:branch. It increases Rank-1 and mAP to 42.1% and 18.7% respectively,
./reviews/deep_19.md:7699:Fig. 6. The impact of different hyper-parameters 𝛼 and 𝛾 on Rank-1 accuracy in the LTCC dataset.
./reviews/deep_19.md:7701:Fig. 7. The visualization of the retrieval ranking results of baseline and our proposed model on LTCC and VC-Clothes datasets. The first column is the query image. The images
./reviews/deep_19.md:7720:as depicted in Fig. 7. The results indicate that the ranking results
./reviews/deep_19.md:7850:Fu, H., Zhang, K., & Wang, J. (2024). An adaptive self-correction joint training
./reviews/deep_19.md:7996:Rank-1
./reviews/deep_19.md:8000:Rank-1
./reviews/deep_19.md:8028:Fig. 4. Retrieval rank-list visualization: green indicates correct recognition, and red indicates incorrect recognition.
./reviews/deep_19.md:8085:As shown in the retrieval rank-list in Fig. 4, the MCCAN method
./reviews/deep_19.md:8140:1.3% in mAP, and 0.1% and 0.5% in Rank-1 on the Market1501 and
./reviews/deep_19.md:8147:datasets, as shown in Table 6. In the early training stage, the model
./reviews/deep_19.md:8165:distribution visualization (Fig. 3(b)), retrieval rank-list visualization
./reviews/deep_19.md:8232:Rank-1
./reviews/deep_19.md:8236:Rank-1
./reviews/deep_19.md:8367:domain adaptive person re-identification, in: Computer Vision–ECCV 2020: 16th
./reviews/deep_19.md:8377:adaptive person re-identification, in: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIV 16,
./reviews/deep_19.md:8403:the extraction strategy according to the semantic complexity of input
./reviews/deep_19.md:8407:and construct a dynamically adaptive multi-granular feature alignment mechanism to further improve the generalization performance of
./reviews/deep_19.md:8446:token and the class token, and ranks them to obtain a Rank-list. We
./reviews/deep_19.md:8447:select the top K ranks to form the Top-K list 𝐾 :
./reviews/deep_19.md:8450:𝑑𝑜𝑡𝑖 = 𝑥𝑖 ⊤ 𝑐𝑙𝑠1 ,  = rank {𝑑𝑜𝑡𝑖 }𝑛𝑖=1 ,  = 𝜉(),
./reviews/deep_19.md:8546:the dot product 𝑑𝑜𝑡𝑖 with the corresponding 𝑥𝑖 , and obtain a re-ranked
./reviews/deep_19.md:8547:Rank-list 𝑋 𝑗 based on the computed dot products:
./reviews/deep_19.md:8598:𝜉(⋅) is the mapping from the dot product list to the feature list. The reranked list 𝑋 𝑗 essentially represents the semantic similarity between
./reviews/deep_19.md:8600:Branch 2; the higher the ranking, the greater the semantic consistency
./reviews/deep_19.md:8604:features, this study needs to introduce the global-feature-based ranked
./reviews/deep_19.md:8605:list  in Eq. (1) for assistance. For all features in 𝑋 𝑗 , we select the topranked feature 𝑝𝑗𝑐1 and the highest-ranked feature in the  list, denoted
./reviews/deep_19.md:8631:where 𝑇 𝑜𝑝(⋅) denotes the operation of selecting the highest-ranked
./reviews/deep_19.md:8640:Fig. 2 clearly illustrates the overall computation pipeline of the multigranularity constraint loss. First, the patch tokens in Branch 1 are
./reviews/deep_19.md:8641:guided for re-ranking by the part-based fine-grained features from
./reviews/deep_19.md:8642:Branch 2 (Eq. (10)), and the top-ranked feature is denoted as 𝑝𝑗𝑐1 .
./reviews/deep_19.md:8644:feature-based ranking (Eq. (1)), the top-ranked feature is indexed and
./reviews/deep_19.md:8777:positive features set to reduce computational complexity:
./reviews/deep_19.md:8911:at Rank-1 (R1), Rank-5 (R5), and Rank-10 (R10). To ensure a fair
./reviews/deep_19.md:8993:4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。
./reviews/deep_19.md:9001:2. 重定义动作：旧问题是“怎么建模视频时间特征”；新问题是“怎么按身体部位自适应选择不同长度的时间线索”。关键词是 dynamic hierarchical graph、adaptive multi-granularity temporal clues、part-consistent graph、mutual nearest neighbor、similarity weighted inference。
./reviews/deep_19.md:9070:4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。
./reviews/deep_19.md:9078:2. 重定义动作：旧问题是“怎么建模视频时间特征”；新问题是“怎么按身体部位自适应选择不同长度的时间线索”。关键词是 dynamic hierarchical graph、adaptive multi-granularity temporal clues、part-consistent graph、mutual nearest neighbor、similarity weighted inference。
./reviews/deep_11.md:28:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_11.md:127:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_11.md:142:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_11.md:159:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_11.md:162:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_11.md:171:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_11.md:203:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_11.md:233:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_11.md:394:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_11.md:423:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_11.md:426:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_11.md:427:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_11.md:504:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_11.md:508:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_11.md:522:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_11.md:528:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_11.md:621:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_11.md:758:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_11.md:833:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_11.md:981:219:These tokens are then ranked based on their importance
./reviews/deep_11.md:1207:/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | rg -n \""'^(Abstract|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[A-Z][A-Za-z ]+"'$|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*)"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_11.md:1215:128:We propose a novel dynamic adaptive camera Jaccard distance
./reviews/deep_11.md:1224:314:3.2. Dynamic adaptive camera jaccard
./reviews/deep_11.md:1225:459:3.3. Dynamic adaptive proxies
./reviews/deep_11.md:1227:609:3.4. Dynamic adaptive knowledge distillation
./reviews/deep_11.md:1228:646:Compute dynamic adaptive camera Jaccard distance matrixs
./reviews/deep_11.md:1229:664:Compute dynamic adaptive camera Jaccard distance
./reviews/deep_11.md:1233:797:Compute dynamic adaptive camera Jaccard distance
./reviews/deep_11.md:1294:/bin/zsh -lc "for f in 'Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf' 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' 'Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf' 'Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf' 'Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf' 'Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf'; do printf '%s: ' \""'$f"; pdfinfo "$f" | rg '"'"'^Pages:'"'; done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_11.md:1297:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf: Pages:           13
./reviews/deep_11.md:1423:   162	features in the target domain. Some uncertainty-based ReID methods [74, 76] have been proposed
./reviews/deep_11.md:1425:   164	of uncertainty in pseudo labels is challenging in the unsupervised scenario, thereby complicating
./reviews/deep_11.md:1693:/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '44,830p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_11.md:1746:    95	This enables comprehensive exploitation of global information to mitigate within-cluster variations and adaptively reﬁne clustering outcomes.
./reviews/deep_11.md:1747:    96	Thirdly, we introduce a novel dynamic adaptive knowledge distillation
./reviews/deep_11.md:1761:   110	inherent complexity indicates that the data points are actually distributed on a manifold rather than strictly conforming to the traditional
./reviews/deep_11.md:1779:   128	We propose a novel dynamic adaptive camera Jaccard distance
./reviews/deep_11.md:1781:   130	characteristics into similarity measurement. By adaptively adjusting
./reviews/deep_11.md:1785:   134	• We propose a novel dynamic adaptive proxies (DAP) module that
./reviews/deep_11.md:1788:   137	• We propose a dynamic adaptive knowledge distillation (DAKD) module that generates high-quality soft labels. This design enables adaptive adjustment of knowledge transfer intensity across diﬀerent samples, eﬀectively suppressing error propagation from noisy pseudolabels.
./reviews/deep_11.md:1791:   140	contrastive learning framework called dynamic adaptive multi-view
./reviews/deep_11.md:1793:   142	Camera Jaccard (DACJ) distance to dynamically estimate camera variations per training epoch and perform adaptive variation reduction. Crucially, when camera variations for a sample exceed a threshold, DACJ
./reviews/deep_11.md:1812:   161	incorporating camera information, eﬀectively mitigating camera variations to improve pseudo-label generation and re-ranking accuracy. However, this method rely on static parameter settings, which constrain
./reviews/deep_11.md:1860:   209	both inter-view and intra-view information and employs an adaptive
./reviews/deep_11.md:1953:   302	Dynamic adaptive camera k-reciprocal nearest neighbors. For
./reviews/deep_11.md:1955:   304	adaptive camera k-reciprocal nearest neighbors(DAC-KRNNs) of 𝑥𝑖 :
./reviews/deep_11.md:1965:   314	3.2. Dynamic adaptive camera jaccard
./reviews/deep_11.md:1983:   332	Dynamic adaptive camera variations estimation. To reduce the
./reviews/deep_11.md:2006:   355	Dynamic adaptive camera local query expansion. Due to camera variations, a large number of within-camera negative samples are
./reviews/deep_11.md:2014:   363	proxy is a deliberate design choice to better cope with the complexity
./reviews/deep_11.md:2049:   398	to these negative samples. So we propose a dynamic adaptive camera
./reviews/deep_11.md:2110:   459	3.3. Dynamic adaptive proxies
./reviews/deep_11.md:2220:   569	adaptive knowledge distillation on the student model, a process that
./reviews/deep_11.md:2223:   572	Since the initialized student model exhibits poor performance on person Re-ID tasks, its early pseudo-labels tend to contain substantial noise,
./reviews/deep_11.md:2232:   581	in the early stages of training. During the second training stage, the student model is used to generate soft pseudo-labels and update the model
./reviews/deep_11.md:2260:   609	3.4. Dynamic adaptive knowledge distillation
./reviews/deep_11.md:2297:   646	Compute dynamic adaptive camera Jaccard distance matrixs
./reviews/deep_11.md:2315:   664	Compute dynamic adaptive camera Jaccard distance
./reviews/deep_11.md:2448:   797	Compute dynamic adaptive camera Jaccard distance
./reviews/deep_11.md:2767:   288	where 𝑅𝑖ℎ𝑎𝑟𝑑 denotes the rank position of the hardest match, and |𝐺𝑖 | indicates the total number of
./reviews/deep_11.md:2831:   352	Venue Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP
./reviews/deep_11.md:2902:/bin/zsh -lc "pdftotext -f 1 -l 8 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '250,458p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_11.md:2956:   302	Dynamic adaptive camera k-reciprocal nearest neighbors. For
./reviews/deep_11.md:2958:   304	adaptive camera k-reciprocal nearest neighbors(DAC-KRNNs) of 𝑥𝑖 :
./reviews/deep_11.md:2968:   314	3.2. Dynamic adaptive camera jaccard
./reviews/deep_11.md:2986:   332	Dynamic adaptive camera variations estimation. To reduce the
./reviews/deep_11.md:3009:   355	Dynamic adaptive camera local query expansion. Due to camera variations, a large number of within-camera negative samples are
./reviews/deep_11.md:3017:   363	proxy is a deliberate design choice to better cope with the complexity
./reviews/deep_11.md:3052:   398	to these negative samples. So we propose a dynamic adaptive camera
./reviews/deep_11.md:3139:    55	Tang et al. [10] introduce adaptive context-aware selection to
./reviews/deep_11.md:3190:   106	method adaptively selects the most critical tokens based
./reviews/deep_11.md:3303:   219	These tokens are then ranked based on their importance
./reviews/deep_11.md:3436:   123	early optimization of model discriminability and gradually extending generalization to different cameras and
./reviews/deep_11.md:3693:   380	discriminatory power. Additionally, during the early stages
./reviews/deep_11.md:3705:   392	foremost, we begin with an assumption: during the early stages
./reviews/deep_11.md:4100:   787	during the early stages of model training, accumulates progressively, thereby negatively impacting the final efficacy of
./reviews/deep_11.md:4161:   281	matching characteristic at Rank1, mean Average Precision
./reviews/deep_11.md:4172:   292	performance. For example, DTST surpasses the mAP/Rank1/mINP of the baseline by 1.18%/3.13%/0.43% on the aerialto-ground (A↔G) protocol of CARGO. Besides, DTST also
./reviews/deep_11.md:4174:   294	Specifically, our proposed DTST exceeds VDT on mAP/Rank1/mINP by 1.51%/1.60%/2.00% on the ALL of AG-ReID.
./reviews/deep_11.md:4181:   301	our approach adaptively selects the most critical tokens using
./reviews/deep_11.md:4191:   311	by 0.57% for the A→G Rank-1 protocol and 1.04% for the
./reviews/deep_11.md:4192:   312	G→A Rank-1 protocol. This consistent improvement suggests
./reviews/deep_11.md:4258:   378	model in several specific retrieval patterns. Rank1, mAP, and mINP are reported (%). The best performance is shown in bold.
./reviews/deep_11.md:4285:   405	Rank1
./reviews/deep_11.md:4291:   411	Rank1
./reviews/deep_11.md:4297:   417	Rank1
./reviews/deep_11.md:4424:   544	Rank1
./reviews/deep_11.md:4443:   563	Rank1
./reviews/deep_11.md:4462:   582	dataset. Rank1, mMAP, and mINP are reported(%). Best in
./reviews/deep_11.md:4470:   590	Rank1
./reviews/deep_11.md:4474:   594	represent the performance in two specific patterns. Rank1,
./reviews/deep_11.md:4485:   605	Rank1
./reviews/deep_11.md:4495:   615	improvement in rank-1 accuracy and 1.34% increase in mAP
./reviews/deep_11.md:4504:   624	rank-1 accuracy and 0.93% drop in mAP. This suggests that
./reviews/deep_11.md:4512:   632	Performance is assessed through Rank1, mAP, and mINP(%),
./reviews/deep_11.md:4551:   671	Rank1
./reviews/deep_11.md:4576:   696	could increase the model’s complexity without corresponding
./reviews/deep_11.md:4584:   704	metrics, i.e. Rank-1 accuracy, mAP, and mINP. Specifically,
./reviews/deep_11.md:4595:   715	2 tokens results in a decrease of 1.25% in rank-1 accuracy,
./reviews/deep_11.md:4601:   721	the number of tokens is 3, model-3 achieves a higher Rank1 accuracy at 46.88%, but both mAP and mINP decrease.
./reviews/deep_11.md:4701:   147	practical applications, due to the difficulty of data annotation, more and more researchers have
./reviews/deep_11.md:4831:   277	SE-Net concentrates on refining channels within a single branch while the AG focuses on adaptively
./reviews/deep_11.md:4833:   279	global average pooling operation, adaptively assigns importance weights to features across varying
./reviews/deep_11.md:4895:   341	noise perturbation is used to enhance the style of each task, which is called style uncertainty
./reviews/deep_11.md:4959:   405	By setting different standard deviations, source tasks of different difficulty levels can be obtained.
./reviews/deep_11.md:5066:    53	By setting different standard deviations, source tasks of different difficulty levels can be obtained.
./reviews/deep_11.md:5335:   322	progressively increasing difficulty. In the initial stage, the Gaussian noise intensities m1 and m2 are
./reviews/deep_11.md:5342:   329	For the experimental evaluation, mean Average Precision (mAP), Rank 1/5/10, and lightweight
./reviews/deep_11.md:5597:   125	different instance inputs and adaptively recognize their discriminative clues.
./reviews/deep_11.md:5621:   149	discriminative information in an adaptive feature space and
./reviews/deep_11.md:5648:   176	discriminative information in an adaptive feature space and overcoming outof-distribution samples, thereby improving identification performance.
./reviews/deep_11.md:5669:   197	to adaptively integrate modality-aware and instance-aware
./reviews/deep_11.md:5733:   261	on the parameters and computational complexity.
./reviews/deep_11.md:5933:   461	divergence and adaptively recognize the discriminative clues
./reviews/deep_11.md:6034:   562	prompting modules, capable of adaptively generating instanceaware prompts according to the instance, including the current
./reviews/deep_11.md:6134:   662	adaptively reweighting the modality-aware prompts based on
./reviews/deep_11.md:6138:   666	where adaptive weighting enhances generalization by reducing
./reviews/deep_11.md:6169:   697	prompts, the CPF module adaptively reweights the modalityaware prompt based on the instance, ensuring that modality
./reviews/deep_11.md:6205:   733	to force the generated prompts being instance-adaptive. The
./reviews/deep_11.md:6441:    52	and 0.57% Rank-1 compared to Variant-1, i.e., the baseline. (2)
./reviews/deep_11.md:6445:    56	2.66/2.37% mAP and 2.19/2.15% Rank-1 on average, respectively. This result illustrates the necessity to design specifically
./reviews/deep_11.md:6448:    59	is 4.15% higher in mAP and 3.98% higher in Rank-1 than
./reviews/deep_11.md:6452:    63	increase of 0.73% mAP and 0.71% Rank-1 compared to
./reviews/deep_11.md:6496:   107	mAP and 0.97% Rank-1 to the baseline when it does not
./reviews/deep_11.md:6530:   141	77.89% Rank-1.
./reviews/deep_11.md:6536:   147	improvement of 0.29% mAP and 0.41% Rank-1 over the Cat.
./reviews/deep_11.md:6543:   154	gain of 0.83% mAP and 1.14% Rank-1 compared to the Cat.
./reviews/deep_11.md:6599:   210	9) Discussion on the Parameters and Computational Complexity: To better analyze the parameter scale and computational overhead of our proposed MIP and several components
./reviews/deep_11.md:6610:   221	and 15.55 GFLOPs in computational complexity compared
./reviews/deep_11.md:6614:   225	computational complexity, respectively.
./reviews/deep_11.md:6633:   244	In terms of computational complexity, it is important to
./reviews/deep_11.md:6825:   111	among top-1 results (Rank-1 accuracy) based on similarity.
./reviews/deep_11.md:6870:   156	and computational complexity of these components.
./reviews/deep_11.md:6913:   199	parameter scale of ‘3×’, but without clearly specifying the
./reviews/deep_11.md:6928:   214	CSC-Net, CycleTrans, SSRR, our MIP outperforms the stateof-the-art results by 3.22% mAP and 0.34% Rank-1 under
./reviews/deep_11.md:6930:   216	results and comparable Rank-1 results. And under IndoorSearch mode, MIP also achieves the best or comparable results
./reviews/deep_11.md:6950:   236	performance of nearly 1%. In addition, MIP outperforms the
./reviews/deep_11.md:6963:   249	tasks, Our MIP shows the best results with 97.34% Rank1 and 78.35% mAP, which outperforms the state-of-the-art
./reviews/deep_11.md:6997:   283	complexity of our method and some other existing methods.
./reviews/deep_11.md:7009:   295	2.19% Rank-1 on average under two search modes on SYSUMM01 dataset. (2) Variant-3 adopts baseline+SIPG as the
./reviews/deep_11.md:7011:   297	by 2.37% mAP and 2.15% Rank-1 on average on SYSUMM01. (3) Based on Variant-3, Variant-4 adds IAEL loss
./reviews/deep_11.md:7014:   300	improvement of 2.84% mAP and 2.27% Rank-1 compared
./reviews/deep_11.md:7016:   302	MPL and SIPG modules, which brings a further improvement of 1.64% mAP and 1.80% Rank-1 on average than
./reviews/deep_11.md:7020:   306	average improvement of 1.04% in mAP and 0.53% in Rank-1
./reviews/deep_11.md:7024:   310	average improvement of 1.61% in mAP and 1.16% in Rank-1
./reviews/deep_11.md:7029:   315	1.82% in mAP and 1.81% in Rank-1 compared to Variant5. (8) Finally, Variant-9, our final full model (MIP) that
./reviews/deep_11.md:7032:   318	in Rank-1.
./reviews/deep_11.md:7052:   338	We plotted the Rank-1 and mAP lines when parameters are
./reviews/deep_11.md:7056:   342	in both Rank-1 and mAP performance, and the performance
./reviews/deep_11.md:7310:   218	difficulty of processing. Therefore, this experiment contributes to a more thorough assessment
./reviews/deep_11.md:7592:   500	mAP index of the model can be improved by nearly 3%, which shows that the dual-branch model
./reviews/deep_11.md:7601:   509	domain adaptive ability of the model is further improved. Although the mAP improvement brought
./reviews/deep_11.md:7680:/bin/zsh -lc "pdftotext -f 12 -l 16 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '1,500p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_11.md:7797:   116	Rank-1
./reviews/deep_11.md:7815:   134	Rank-10
./reviews/deep_11.md:7832:   151	Rank-20
./reviews/deep_11.md:7872:   191	Rank-1
./reviews/deep_11.md:7889:   208	Rank-10 Rank-20
./reviews/deep_11.md:8153:    10	develop an adaptive threshold mechanism to dynamically optimize the
./reviews/deep_11.md:8409:   266	Dai, Y., Liu, J., Sun, Y., Tong, Z., Zhang, C., & Duan, L.-Y. (2021). IDM: An intermediate domain module for domain adaptive person Re-ID. In Proceedings of the IEEE/CVF
./reviews/deep_11.md:8420:   277	hybrid memory for domain adaptive object Re-ID. Advances in Neural Information Processing Systems, 33, 11309–11321.
./reviews/deep_11.md:8421:   278	Han, J., Li, Y.-L., & Wang, S. (2022a). Delving into probabilistic uncertainty for unsupervised domain adaptive person re-identiﬁcation. Proceedings of the AAAI Conference on
./reviews/deep_11.md:8439:   296	In this paper, we propose a novel dynamic adaptive multi-view contrastive learning (DAMCL) framework for unsupervised Re-ID. We improve the Jaccard distance calculated before clustering by utilizing camera information to compute a DACJ distance, which helps the clustering
./reviews/deep_11.md:8452:   309	designing a framework that is more eﬃcient in both time and space complexity becomes a critical task to explore. Additionally, while the current
./reviews/deep_11.md:8463:   320	reﬁnement for unsupervised domain adaptive person re-identiﬁcation. In Proceedings
./reviews/deep_11.md:8473:   330	Lee, G., Lee, S., Kim, D., Shin, Y., Yoon, Y., & Ham, B. (2023). Camera-driven representation learning for unsupervised domain adaptive person re-identiﬁcation. In Proceedings
./reviews/deep_11.md:8483:   340	Ran, Z., Lu, X., Wei, X., & Liu, W. (2025). Camera-aware graph multi-domain adaptive
./reviews/deep_11.md:8529:   386	Zhong, Z., Zheng, L., Cao, D., & Li, S. (2017). Re-ranking person re-identiﬁcation with Kreciprocal encoding. In Proceedings of the IEEE conference on computer vision and pattern
./reviews/deep_11.md:8558:    11	Index CC-1 IC-1 CC-2 IC-2 Rank-1 Rank-10
./reviews/deep_11.md:8589:    42	Rank-1 Rank-10
./reviews/deep_11.md:8593:    46	Rank-1 Rank-10
./reviews/deep_11.md:8675:   128	Index VC MIE DNC HMCL Rank-1 Rank-10
./reviews/deep_11.md:8698:   151	Rank-1 Rank-10
./reviews/deep_11.md:8702:   155	Rank-1 Rank-10
./reviews/deep_11.md:8844:   297	network’s learning process from simplicity to complexity.
./reviews/deep_11.md:8941:   394	lies in the introduction of CC-2, which nearly doubles the
./reviews/deep_11.md:8976:   429	14%-17% in mAP. This clearly illustrates the effectiveness of
./reviews/deep_11.md:9006:   459	results in a substantial separation of identities within nearly
./reviews/deep_11.md:9094:    19	Venue Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP
./reviews/deep_11.md:9223:   148	We report the Rank at r accuracy (%), mAP (%), and mINP (%). The best results are denoted in bold.
./reviews/deep_11.md:9235:   160	Rank-1 Rank-10 Rank-20 mAP mINP Rank-1 Rank-10 Rank-20 mAP mINP
./reviews/deep_11.md:9374:   299	Baseline CSM CRLR MM WMM Rank-1 mAP mINP Rank-1 mAP mINP Rank-1 mAP mINP
./reviews/deep_11.md:9399:   324	for dual-modality-shared learning. Rank at r accuracy (%), mAP (%), and mINP (%) are reported. The best results
./reviews/deep_11.md:9407:   332	methods. On SYSU-MM01, our approach achieves a Rank-1 accuracy of 53.89% for the All-Search
./reviews/deep_11.md:9411:   336	proposed method outperforms most supervised methods, achieving a Rank-1 accuracy of 89.68%.
./reviews/deep_11.md:9433:   358	MM leads to a +10.51%/+10.41% and +43.01%/+39.66% Rank-1/mAP improvement on SYSU-MM01
./reviews/deep_11.md:9476:   401	improves the performance of 4.20%/3.13% Rank-1/mAP on SYSU-MM01 (All-Search). Compared
./reviews/deep_11.md:9487:   412	CSM+WMM, CRLR achieves improvements of 1.60%/1.38% Rank-1/mAP on SYSU-MM01 (AllSearch). This enhancement can be attributed to CRLR’s ability to refine labels and mitigate the
./reviews/deep_11.md:9514:   439	Rank-1 accuracy on the SYSU-MM01 (All-Search) and RegDB (Visible to Infrared), respectively.
./reviews/deep_11.md:9517:   442	SYSU-MM01 is more sensitive to 𝛾, with a performance fluctuation of 4.38% Rank-1, while on
./reviews/deep_11.md:9725:   120	memory for domain adaptive object Re-ID. In NIPS, 11309–11321.
./reviews/deep_11.md:9738:/bin/zsh -lc "pdftotext -f 8 -l 12 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_11.md:9764:    25	Compute dynamic adaptive camera Jaccard distance matrixs
./reviews/deep_11.md:9782:    43	Compute dynamic adaptive camera Jaccard distance
./reviews/deep_11.md:9915:   176	Compute dynamic adaptive camera Jaccard distance
./reviews/deep_11.md:10063:   324	that DACJ eﬀectively mitigates cross-camera discrepancies through dynamic sample-level camera variations assessment and adaptive distance
./reviews/deep_11.md:10075:   336	to its scale and complexity. We use two widely recognized evaluation
./reviews/deep_11.md:10076:   337	metrics: mean Average Precision (mAP) and Cumulative Matching Characteristic (CMC). Speciﬁcally, we present the Rank-1 (R1), Rank-5 (R5),
./reviews/deep_11.md:10077:   338	and Rank-10 (R10) accuracy values derived from the CMC curves in our
./reviews/deep_11.md:10125:   386	Qualitative ablation studies of our dynamic adaptive optimal proxy with
./reviews/deep_11.md:10266:/bin/zsh -lc "pdftotext -f 10 -l 12 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' - | nl -ba | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_11.md:10273:     6	Qualitative ablation studies of our dynamic adaptive optimal proxy with
./reviews/deep_11.md:10503:   236	(Table 5), the dual local view conﬁguration achieves optimal eﬃciencyaccuracy equilibrium. The dual local views method achieves the highest mAP and Rank-1 accuracy on Market-1501 and MSMT17, and the
./reviews/deep_11.md:10506:   239	40.7 h, nearly doubling our approach’s time. This signiﬁcant increase
./reviews/deep_11.md:10644:**2. Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification**
./reviews/deep_11.md:10648:2. 重定义动作：他们没有泛泛说“伪标签有噪声”，而是把噪声拆成三个更具体的问题：距离度量不懂相机，代理点不符合 DBSCAN 的局部密度结构，监督目标太硬。关键词是 dynamic adaptive camera Jaccard、manifold、medoid proxy、dynamic adaptive proxies、soft pseudo labels。
./reviews/deep_11.md:10676:4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。
./reviews/deep_11.md:10684:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reviews/deep_11.md:10696:2. 重定义动作：它把“缩小 modality gap”重定义成“增强模型对 modality、input instance、query instance 的适应能力”。关键词是 model adaptability、flexibility、modality-aware prompts、instance-aware prompts、query-guiding prompts、adaptive feature space。这个转向很重要，因为它避开了“把两种模态硬映射到同一空间”的旧叙事。
./reviews/deep_11.md:10723:**2. Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification**
./reviews/deep_11.md:10727:2. 重定义动作：他们没有泛泛说“伪标签有噪声”，而是把噪声拆成三个更具体的问题：距离度量不懂相机，代理点不符合 DBSCAN 的局部密度结构，监督目标太硬。关键词是 dynamic adaptive camera Jaccard、manifold、medoid proxy、dynamic adaptive proxies、soft pseudo labels。
./reviews/deep_11.md:10755:4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。
./reviews/deep_11.md:10763:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reviews/deep_11.md:10775:2. 重定义动作：它把“缩小 modality gap”重定义成“增强模型对 modality、input instance、query instance 的适应能力”。关键词是 model adaptability、flexibility、modality-aware prompts、instance-aware prompts、query-guiding prompts、adaptive feature space。这个转向很重要，因为它避开了“把两种模态硬映射到同一空间”的旧叙事。
./reviews/deep_28.md:67: exited 1 in 0ms:
./reviews/deep_28.md:150:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_28.md:186:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_28.md:215:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_28.md:218:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_28.md:219:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_28.md:296:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_28.md:300:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_28.md:314:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_28.md:320:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_28.md:386: exited 1 in 0ms:
./reviews/deep_28.md:392:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_28.md:409:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_28.md:412:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_28.md:421:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_28.md:453:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_28.md:483:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_28.md:643:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_28.md:887:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_28.md:962:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_28.md:1619:   293	Table 1: Comparisons with SOTA methods on SYSU-MM01 and RegDB datasets. We report Rank at 𝑟 accuracy (%) and mAP (%).
./reviews/deep_28.md:1811:   485	Rank-1
./reviews/deep_28.md:1829:   503	Rank-1
./reviews/deep_28.md:1976:    83	matching and similarity ranking. However, in the modality-shared feature space, identity information is often intertwined with camera noise and modality-sensitive redundancy, resulting in the
./reviews/deep_28.md:2054:   161	Network that leverages adversarial training and adaptive dynamic weights to reduce cross-modality
./reviews/deep_28.md:2075:   182	Existing decoupling methods do not fully consider the complex relationships between modalityshared features and other redundant features, which are not simply linearly combined. As a result,
./reviews/deep_28.md:2135:   242	We are inspired by SE-Net [13] and hope to use channel masks 𝑀 to decouple features 𝐹 and adaptively separate camera-related features 𝐹𝑐𝑟 and camera-irrelated information 𝐹𝑐𝑖 . From Equations
./reviews/deep_28.md:2250:   357	To guide the channel mask 𝑀 to adaptively extract identity information from 𝐹˜𝑐𝑖 , we propose
./reviews/deep_28.md:2497:   425	LoRA. LoRA [21] incorporates low-rank trainable matrices into transformer layers with the aim of providing an
./reviews/deep_28.md:2499:   427	matrix W ∈ Rd×k , LoRA can update it with a low-rank
./reviews/deep_28.md:2770:   698	through a residual connection to adapt the output information of MLP and MHA. LoRA [21] incorporates low-rank
./reviews/deep_28.md:2795:   723	attention of MHA sublayer, which complements the taskspecific information and steers model to focus on crucial TPRspecific content. (2) Low-rank matrices are designed in weight
./reviews/deep_28.md:3585:   184	also adds to the complexity and challenge of VI-ReID
./reviews/deep_28.md:3733:   268	Rank-1
./reviews/deep_28.md:3737:   272	Rank-1
./reviews/deep_28.md:3741:   276	Rank-1
./reviews/deep_28.md:3745:   280	Rank-1
./reviews/deep_28.md:3766:    39	Fig. 1 (a). Due to the difficulty of learning rich pedestrian semantics
./reviews/deep_28.md:3880:   153	of LVLMs to enhance visual features adaptively. Our experiments
./reviews/deep_28.md:3895:   441	Evaluation metrics. We follow the common practices to adopt Cumulative Matching Characteristics (CMC) at Rank-1 and mean Average Precision (mAP) for performance evaluation.
./reviews/deep_28.md:3903:   449	adopted, with the warmup strategy that linearly increases the learning rate from 3 × 10−5 to 3 × 10−4 in the first 10 epochs. We train
./reviews/deep_28.md:3926:   472	Rank-1
./reviews/deep_28.md:3930:   476	Rank-1
./reviews/deep_28.md:3976:   522	Rank-1
./reviews/deep_28.md:3980:   526	Rank-1
./reviews/deep_28.md:4004:   550	achieves an mAP of 82.8% and a Rank-1 accuracy of 92.2%, surpassing previous advanced methods. On the CUHK03 dataset,
./reviews/deep_28.md:4041:   587	Rank-1 --------------------------------------------------------------------------------------> Rank-8
./reviews/deep_28.md:4047:   593	Rank-1 --------------------------------------------------------------------------------------> Rank-8
./reviews/deep_28.md:4162:   404	Due to the complexity and variability of the patterns of
./reviews/deep_28.md:4306:   548	confidence ranking. Thus we have the loss function:
./reviews/deep_28.md:4346:   588	the advantages of early learning [84], the empirical and
./reviews/deep_28.md:4378:   620	labelled samples. Since distinguishing the two becomes nearly
./reviews/deep_28.md:4399:   641	components, balanced by three adaptive hyper-parameters
./reviews/deep_28.md:4400:   642	β1 , β2 and β3 . The adaptive hyper-parameters are computed
./reviews/deep_28.md:4888:671:results under all-search mode for analysis. Compared to the baseline network, MPT improves 5.45% on Rank-1 and 5.88% on mAP.
./reviews/deep_28.md:4923:   268	Rank-1
./reviews/deep_28.md:4927:   272	Rank-1
./reviews/deep_28.md:4931:   276	Rank-1
./reviews/deep_28.md:4935:   280	Rank-1
./reviews/deep_28.md:5147:320:models. LoRA [21] as a representative method for reparameterized fine-tuning methods, utilizes low-rank decomposition
./reviews/deep_28.md:5159:425:LoRA. LoRA [21] incorporates low-rank trainable matrices into transformer layers with the aim of providing an
./reviews/deep_28.md:5160:427:matrix W ∈ Rd×k , LoRA can update it with a low-rank
./reviews/deep_28.md:5187:698:through a residual connection to adapt the output information of MLP and MHA. LoRA [21] incorporates low-rank
./reviews/deep_28.md:5236:1174:rank of LoRA r is set to 32, while for RSTPReid, due to its
./reviews/deep_28.md:5247:1249:model parameters, indicating lower computational complexity.
./reviews/deep_28.md:5272:1312:across all three TPR datasets to balance performance and efficiency. For LoRA, its rank r is adjusted according to dataset
./reviews/deep_28.md:5308:1848:cost of an dramatic increase in model size and trainable parameters. Therefore, UP-Person with SDM offers an optimal tradeoff between complexity and performance without bells and
./reviews/deep_28.md:5377:83:matching and similarity ranking. However, in the modality-shared feature space, identity information is often intertwined with camera noise and modality-sensitive redundancy, resulting in the
./reviews/deep_28.md:5406:161:Network that leverages adversarial training and adaptive dynamic weights to reduce cross-modality
./reviews/deep_28.md:5415:182:Existing decoupling methods do not fully consider the complex relationships between modalityshared features and other redundant features, which are not simply linearly combined. As a result,
./reviews/deep_28.md:5439:242:We are inspired by SE-Net [13] and hope to use channel masks 𝑀 to decouple features 𝐹 and adaptively separate camera-related features 𝐹𝑐𝑟 and camera-irrelated information 𝐹𝑐𝑖 . From Equations
./reviews/deep_28.md:5479:718:the model’s Rank values and mAP showed significant improvements when the adversarial decoupling loss was applied. This indicates that relying solely on the channel masking mentioned in
./reviews/deep_28.md:5536:16:often overlook the complexities of instance-dependent and realworld noise, creating a disconnect from the practical applications
./reviews/deep_28.md:5597:300:complexity of real-world noise. Recognizing this gap, our
./reviews/deep_28.md:5699:1025:more misleading. On the Rank-1 metric, real-world noise
./reviews/deep_28.md:5749:  1260	UP-Person ranks among the top two in multiple metrics
./reviews/deep_28.md:5752:  1263	computational efficiency, storage efficiency, model complexity,
./reviews/deep_28.md:5801:  1312	across all three TPR datasets to balance performance and efficiency. For LoRA, its rank r is adjusted according to dataset
./reviews/deep_28.md:5803:  1314	datasets, use a rank of 32 to ensure sufficient representational
./reviews/deep_28.md:5805:  1316	benefits from a reduced rank of 16 to prevent overfitting while
./reviews/deep_28.md:6102:   718	the model’s Rank values and mAP showed significant improvements when the adversarial decoupling loss was applied. This indicates that relying solely on the channel masking mentioned in
./reviews/deep_28.md:6241:   671	results under all-search mode for analysis. Compared to the baseline network, MPT improves 5.45% on Rank-1 and 5.88% on mAP.
./reviews/deep_28.md:6269:   953	of 6.79% in Rank-1 is observed, attributable to the powerful
./reviews/deep_28.md:6271:   955	row, our method achieves a 7.23% increase in Rank-1, a
./reviews/deep_28.md:6306:   990	of L O N , there is a 0.44% increase in Rank-1, a 1.53% increase
./reviews/deep_28.md:6320:  1004	parameters β1 , β2 and β3 , we adopted a strategy that adaptively
./reviews/deep_28.md:6341:  1025	more misleading. On the Rank-1 metric, real-world noise
./reviews/deep_28.md:6538:  1842	overall Rank metrics with ID loss are lower than those of SDM
./reviews/deep_28.md:6544:  1848	cost of an dramatic increase in model size and trainable parameters. Therefore, UP-Person with SDM offers an optimal tradeoff between complexity and performance without bells and
./reviews/deep_28.md:6655:   937	of 0.73%, 0.37%, and 0.32% in the metrics of Rank-1, mean
./reviews/deep_28.md:6661:   943	metrics of Rank-1, mAP, and mINP, respectively, compared to
./reviews/deep_28.md:6689:   758	size and consequently lower complexity, RegDB serves as an
./reviews/deep_28.md:6770:   839	primarily focuses on rank accuracy, evaluating how effectively
./reviews/deep_28.md:6771:   840	a model ranks the correct match within a list of potential
./reviews/deep_28.md:6811:   880	and 5.76% on Rank-1, Rank-10, Rank-20, mAP and mINP,
./reviews/deep_28.md:6821:   890	improves the performance on the metrics Rank-1, mAP, and
./reviews/deep_28.md:6837:   906	there is a 7.23% increase in Rank-1 accuracy, a 6.94%
./reviews/deep_28.md:6841:   910	due to the reduced complexity of the task, the Rank-20
./reviews/deep_28.md:6844:   913	class within the top 20 results. Additionally, the Rank-1, mAP,
./reviews/deep_28.md:6902:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./reviews/deep_28.md:6916:机制几乎是从三类噪声定义直接长出来的。先用 co-training 和 warm-up 利用 early learning，小损失样本更可能干净。再用 GMM 拟合单样本 CE loss，把样本分成 clean、indistinguishably noisy、obviously noisy。干净样本用 CE，因为标签可信；难分辨噪声用 GCE，降低噪声样本的大损失偏置；明显噪声直接不信原标签，用分类置信度构造软伪标签。检索侧还把样本对按置信度划成 clean/noisy correspondence，修正 triplet 里的正负关系，形成 robust retrieval loss。最后把识别损失和检索损失混成 RHL。
./reviews/deep_28.md:6919:这篇的证据闭环很强，因为它先造了 SYSU-Real，报告真实噪声比例是 23.79%，并且红外内部噪声比例略高，这直接支撑“真实噪声值得研究”。然后它在 random noise、IDN 和 SYSU-Real 三种条件下都测，证明不是只吃某一种噪声。RHL 消融里，加入处理 indistinguishable noisy subset 的 GCE 带来最大提升，完整方法在 SYSU-Real 上比 DART 提升 7.23% Rank-1、6.94% mAP 和 7.48% mINP，正好支持“三类样本要分治”。可视化也服务于重定义：不同噪声类型的退化柱状图说明 IDN 和真实噪声比随机噪声更伤；相似度矩阵差异图显示 RHL 提高类内相似、降低类间相似；t-SNE 显示比 DART 聚类更清楚。
./reviews/deep_28.md:6992:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./reviews/deep_28.md:7006:机制几乎是从三类噪声定义直接长出来的。先用 co-training 和 warm-up 利用 early learning，小损失样本更可能干净。再用 GMM 拟合单样本 CE loss，把样本分成 clean、indistinguishably noisy、obviously noisy。干净样本用 CE，因为标签可信；难分辨噪声用 GCE，降低噪声样本的大损失偏置；明显噪声直接不信原标签，用分类置信度构造软伪标签。检索侧还把样本对按置信度划成 clean/noisy correspondence，修正 triplet 里的正负关系，形成 robust retrieval loss。最后把识别损失和检索损失混成 RHL。
./reviews/deep_28.md:7009:这篇的证据闭环很强，因为它先造了 SYSU-Real，报告真实噪声比例是 23.79%，并且红外内部噪声比例略高，这直接支撑“真实噪声值得研究”。然后它在 random noise、IDN 和 SYSU-Real 三种条件下都测，证明不是只吃某一种噪声。RHL 消融里，加入处理 indistinguishable noisy subset 的 GCE 带来最大提升，完整方法在 SYSU-Real 上比 DART 提升 7.23% Rank-1、6.94% mAP 和 7.48% mINP，正好支持“三类样本要分治”。可视化也服务于重定义：不同噪声类型的退化柱状图说明 IDN 和真实噪声比随机噪声更伤；相似度矩阵差异图显示 RHL 提高类内相似、降低类间相似；t-SNE 显示比 DART 聚类更清楚。
./reviews/deep_15.md:123:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_15.md:140:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_15.md:173:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_15.md:181:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_15.md:224:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_15.md:230:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_15.md:244:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_15.md:610:spatial attention due to the difficulty of learning effective
./reviews/deep_15.md:703:two frequently-used metrics rank-1 and mAP to perform the
./reviews/deep_15.md:753:rank-1
./reviews/deep_15.md:754:mAP rank-1
./reviews/deep_15.md:799:rank-1
./reviews/deep_15.md:801:rank-1
./reviews/deep_15.md:854:rank-1
./reviews/deep_15.md:868:rank-1
./reviews/deep_15.md:884:superior performance with 19.0% / 4.1% absolute improvements in rank-1 on LTCC/PRCC of CC Mode, illustrating the
./reviews/deep_15.md:887:the SC mode of PRCC, the 99.0% rank-1 of IFD is close to
./reviews/deep_15.md:1258:multiscale perception schemes have been proven to significantly improve the accuracy of CC-ReID (Rank-01 and mAP)
./reviews/deep_15.md:1355:task. DLAW [3] incorporated an adaptive cloth-changed region
./reviews/deep_15.md:1838:complexity.
./reviews/deep_15.md:1864:as evaluation metrics, with the rank metric being demonstrated
./reviews/deep_15.md:1865:by Rank@1.
./reviews/deep_15.md:1940:improvement in Rank@1 and mAP for the LTCC and PRCC
./reviews/deep_15.md:1943:difficulty of garment variation. For example, PRCC places
./reviews/deep_15.md:1960:4) performance of different feature scaling ratios, 5) computational complexity of the model, and 6) visualization results
./reviews/deep_15.md:1992:complexity of the feature alignment process can be reduced.
./reviews/deep_15.md:2787:models, which are insufficient to fully simulate real pedestrians, and the difficulty in reproducing the process of collecting
./reviews/deep_15.md:2847:object instances [17]. The early incorporation of texture information into Re-ID tasks was primarily aimed at mitigating
./reviews/deep_15.md:2970:to the MSMT17 test set show that both Rank-1 and mAP
./reviews/deep_15.md:3212:specific emphasis on Rank-1 accuracy, Rank-5 accuracy, and
./reviews/deep_15.md:3295:characters with distinct clothing textures. Figure 8 clearly
./reviews/deep_15.md:3297:accuracy of Rank-1 and mAP consistently increases on all
./reviews/deep_15.md:3324:networks, the accuracy of Rank-1 and mAP continues to
./reviews/deep_15.md:3328:accuracy of Rank-1 and mAP consistently exhibits a consistent
./reviews/deep_15.md:3434:person retrieval task is still in its early stages of development.
./reviews/deep_15.md:3467:to retrieval failures. In certain cases, an image may still rank
./reviews/deep_15.md:3586:ranking model based on the learning to rank framework.
./reviews/deep_15.md:3588:features to construct the ranking model. To be more precise,
./reviews/deep_15.md:3664:a Gumbel attention module, a hierarchical adaptive matching
./reviews/deep_15.md:4027:ranking loss and ID loss, which are combined to form the
./reviews/deep_15.md:4039:whereas the triplet ranking loss is applied to vl / tl , vl /
./reviews/deep_15.md:4070:evaluated based on the Rank-k accuracy metric. Specifically,
./reviews/deep_15.md:4072:the test set are ranked based on their similarities with the
./reviews/deep_15.md:4074:image of the targeted pedestrian is ranked within the top-k.
./reviews/deep_15.md:4075:Rank-k accuracy stands for the probability of finding at least
./reviews/deep_15.md:4078:is conducted by reporting the Rank-1/5/10 accuracies for all
./reviews/deep_15.md:4117:retrieval. The Rank-1, Rank-5, and Rank-10 accuracies are
./reviews/deep_15.md:4131:0.37% and 0.95%, 0.35% 0.40% under the Rank-1/5/10
./reviews/deep_15.md:4346:Considering the complexity of manually annotating data, unsupervised
./reviews/deep_15.md:4381:the learned features. Recently, to alleviate the complexity and difficulties associated with manual annotation, several unsupervised VI-ReID
./reviews/deep_15.md:4401:latent space and lower the computational complexity. For an image
./reviews/deep_15.md:4418:which updates all the parameters of the pre-trained model, Low-Rank
./reviews/deep_15.md:4420:a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
./reviews/deep_15.md:4720:mode and indoor-search mode. In particular, we achieve 64.95% Rank1 and 63.59% mAP for all-search, and 71.57% Rank-1 and 77.44% mAP
./reviews/deep_15.md:4723:by 2.87% Rank-1 and 1.95% mAP in term of all-search, and enhance
./reviews/deep_15.md:4724:RPNR by 2.71% Rank-1 and 1.71% mAP in term of indoor-search. Our
./reviews/deep_15.md:4736:excellent results on RegDB. To be specific, our method demonstrates superiority over both baselines, with 88.88% Rank-1 and 82.41% mAP for
./reviews/deep_15.md:4737:Vis-to-IR, and 87.52% Rank-1 and 81.18% mAP for IR-to-Vis. Note that
./reviews/deep_15.md:4802:Rank-1
./reviews/deep_15.md:4804:Rank-10
./reviews/deep_15.md:4806:Rank-20
./reviews/deep_15.md:4810:Rank-1
./reviews/deep_15.md:4812:Rank-10
./reviews/deep_15.md:4814:Rank-20
./reviews/deep_15.md:5041:Rank-1
./reviews/deep_15.md:5045:Rank-1
./reviews/deep_15.md:5244:Rank-1
./reviews/deep_15.md:5246:Rank-10
./reviews/deep_15.md:5248:Rank-20
./reviews/deep_15.md:5389:   262	to the MSMT17 test set show that both Rank-1 and mAP
./reviews/deep_15.md:5631:   504	specific emphasis on Rank-1 accuracy, Rank-5 accuracy, and
./reviews/deep_15.md:5780:   268	task. DLAW [3] incorporated an adaptive cloth-changed region
./reviews/deep_15.md:6315:   498	mode and indoor-search mode. In particular, we achieve 64.95% Rank1 and 63.59% mAP for all-search, and 71.57% Rank-1 and 77.44% mAP
./reviews/deep_15.md:6318:   501	by 2.87% Rank-1 and 1.95% mAP in term of all-search, and enhance
./reviews/deep_15.md:6319:   502	RPNR by 2.71% Rank-1 and 1.71% mAP in term of indoor-search. Our
./reviews/deep_15.md:6331:   514	excellent results on RegDB. To be specific, our method demonstrates superiority over both baselines, with 88.88% Rank-1 and 82.41% mAP for
./reviews/deep_15.md:6332:   515	Vis-to-IR, and 87.52% Rank-1 and 81.18% mAP for IR-to-Vis. Note that
./reviews/deep_15.md:6644:   289	spatial attention due to the difficulty of learning effective
./reviews/deep_15.md:6934:   329	65.7%, and 74.8% in Rank-1, respectively.
./reviews/deep_15.md:6969:   364	reliable middle features with less model complexity. (2) We
./reviews/deep_15.md:6981:   376	the Rank-1 of our method is improved by 0.6%, 0.4%, and
./reviews/deep_15.md:6984:   379	our HOH-Net achieves the best 74.8% Rank-1 on the HITSZVCM dataset.
./reviews/deep_15.md:7406:   382	two frequently-used metrics rank-1 and mAP to perform the
./reviews/deep_15.md:7456:   432	rank-1
./reviews/deep_15.md:7457:   433	mAP rank-1
./reviews/deep_15.md:7502:   478	rank-1
./reviews/deep_15.md:7504:   480	rank-1
./reviews/deep_15.md:7616:    68	person retrieval task is still in its early stages of development.
./reviews/deep_15.md:7649:   101	to retrieval failures. In certain cases, an image may still rank
./reviews/deep_15.md:7768:   220	ranking model based on the learning to rank framework.
./reviews/deep_15.md:7770:   222	features to construct the ranking model. To be more precise,
./reviews/deep_15.md:7846:   298	a Gumbel attention module, a hierarchical adaptive matching
./reviews/deep_15.md:8476:  1163	difficulty, we introduce a whitening operation to project the
./reviews/deep_15.md:8541:   587	characters with distinct clothing textures. Figure 8 clearly
./reviews/deep_15.md:8543:   589	accuracy of Rank-1 and mAP consistently increases on all
./reviews/deep_15.md:8570:   616	networks, the accuracy of Rank-1 and mAP continues to
./reviews/deep_15.md:8574:   620	accuracy of Rank-1 and mAP consistently exhibits a consistent
./reviews/deep_15.md:8752:   661	ranking loss and ID loss, which are combined to form the
./reviews/deep_15.md:8764:   673	whereas the triplet ranking loss is applied to vl / tl , vl /
./reviews/deep_15.md:8795:   704	evaluated based on the Rank-k accuracy metric. Specifically,
./reviews/deep_15.md:8797:   706	the test set are ranked based on their similarities with the
./reviews/deep_15.md:8799:   708	image of the targeted pedestrian is ranked within the top-k.
./reviews/deep_15.md:8800:   709	Rank-k accuracy stands for the probability of finding at least
./reviews/deep_15.md:8803:   712	is conducted by reporting the Rank-1/5/10 accuracies for all
./reviews/deep_15.md:8842:   751	retrieval. The Rank-1, Rank-5, and Rank-10 accuracies are
./reviews/deep_15.md:8856:   765	0.37% and 0.95%, 0.35% 0.40% under the Rank-1/5/10
./reviews/deep_15.md:9322:  1586	complexity of the feature alignment process can be reduced.
./reviews/deep_15.md:9533:    79	models, which are insufficient to fully simulate real pedestrians, and the difficulty in reproducing the process of collecting
./reviews/deep_15.md:9593:   139	object instances [17]. The early incorporation of texture information into Re-ID tasks was primarily aimed at mitigating
./reviews/deep_15.md:10411:   124	Considering the complexity of manually annotating data, unsupervised
./reviews/deep_15.md:10446:   159	the learned features. Recently, to alleviate the complexity and difficulties associated with manual annotation, several unsupervised VI-ReID
./reviews/deep_15.md:10466:   179	latent space and lower the computational complexity. For an image
./reviews/deep_15.md:10483:   196	which updates all the parameters of the pre-trained model, Low-Rank
./reviews/deep_15.md:10485:   198	a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
./reviews/deep_15.md:10668:   563	superior performance with 19.0% / 4.1% absolute improvements in rank-1 on LTCC/PRCC of CC Mode, illustrating the
./reviews/deep_15.md:10671:   566	the SC mode of PRCC, the 99.0% rank-1 of IFD is close to
./reviews/deep_15.md:10863:   171	multiscale perception schemes have been proven to significantly improve the accuracy of CC-ReID (Rank-01 and mAP)
./reviews/deep_15.md:11221:   777	as evaluation metrics, with the rank metric being demonstrated
./reviews/deep_15.md:11222:   778	by Rank@1.
./reviews/deep_15.md:11279:   873	4) performance of different feature scaling ratios, 5) computational complexity of the model, and 6) visualization results
./reviews/deep_15.md:11303:   897	Fig. 3. Influence of parameters α and β on the LTCC dataset: (a) for fixed β, results (Rank@1 and mAP) of adjusting α; (b) results corresponding to variable
./reviews/deep_15.md:11306:   900	Fig. 4. Influence of parameters α and β on the PRCC dataset: (a) for fixed β, results (Rank@1 and mAP) of adjusting α; (b) results corresponding to variable
./reviews/deep_15.md:11365:   971	original input features to reduce the computational complexity
./reviews/deep_15.md:11369:   975	optimal performance at Rank@1 for the LTCC dataset was
./reviews/deep_15.md:11649:  2771	the HOH-Net outperforms the previous it by 1.2% Rank-1 for
./reviews/deep_15.md:11659:  2788	reduced by 29.5%, and the Rank-1 of the VIS to IR search
./reviews/deep_15.md:11677:  2806	improved accuracy by 0.8% in Rank-1 on the SYSU-MM01
./reviews/deep_15.md:11684:  2824	representation. Compared with #5, the Rank-1 performance of
./reviews/deep_15.md:11694:  2834	computational complexity (+6.3M) and has a higher number
./reviews/deep_15.md:11700:  2840	slightly increase complexity, they also significantly improve
./reviews/deep_15.md:11726:  2982	w/o CB” by 1.1% and 0.7% in Rank-1 on the SYSU-MM01
./reviews/deep_15.md:11746:  3002	method outperforms them by 1.0% and 0.6% in Rank-1 on
./reviews/deep_15.md:11803:  3081	Rank-1 of “+FCGA w/o FCGA w/o LHM AL ” and “+FGA w/o
./reviews/deep_15.md:11806:  3084	w/o LHM AL ”), it achieves 74.1% in Rank-1 on the SYSUMM01 dataset, compared with the “Baseline+SLE+HSL”
./reviews/deep_15.md:11816:  3094	higher Rank-1 than the method without middle-feature agents
./reviews/deep_15.md:11859:  3243	Rank-1 when applied to all the weight-shared feature extraction network stages (i.e., Stages 1-4 of the backbone). This
./reviews/deep_15.md:11886:  3289	the complete MRIC loss yields a 0.7% Rank-1 improvement
./reviews/deep_15.md:12230:   504	specific emphasis on Rank-1 accuracy, Rank-5 accuracy, and
./reviews/deep_15.md:12289:   587	characters with distinct clothing textures. Figure 8 clearly
./reviews/deep_15.md:12291:   589	accuracy of Rank-1 and mAP consistently increases on all
./reviews/deep_15.md:12318:   616	networks, the accuracy of Rank-1 and mAP continues to
./reviews/deep_15.md:12322:   620	accuracy of Rank-1 and mAP consistently exhibits a consistent
./reviews/deep_15.md:12502:   101	to retrieval failures. In certain cases, an image may still rank
./reviews/deep_15.md:12821:   661	ranking loss and ID loss, which are combined to form the
./reviews/deep_15.md:12833:   673	whereas the triplet ranking loss is applied to vl / tl , vl /
./reviews/deep_15.md:12842:   709	Rank-k accuracy stands for the probability of finding at least
./reviews/deep_15.md:12845:   712	is conducted by reporting the Rank-1/5/10 accuracies for all
./reviews/deep_15.md:12880:   751	retrieval. The Rank-1, Rank-5, and Rank-10 accuracies are
./reviews/deep_15.md:12894:   765	0.37% and 0.95%, 0.35% 0.40% under the Rank-1/5/10
./reviews/deep_15.md:13113:  1121	quite a few untargeted images ranked high in the result lists
./reviews/deep_15.md:13120:  1128	the wrong local parts, it still ranks high. On the contrary,
./reviews/deep_15.md:13127:  1135	ones ranked high in the result lists of EAIBC are also still
./reviews/deep_15.md:13135:  1143	the queries are ranked higher than the untargeted ones as the
./reviews/deep_15.md:13258:   196	which updates all the parameters of the pre-trained model, Low-Rank
./reviews/deep_15.md:13260:   198	a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
./reviews/deep_15.md:13478:   514	excellent results on RegDB. To be specific, our method demonstrates superiority over both baselines, with 88.88% Rank-1 and 82.41% mAP for
./reviews/deep_15.md:13479:   515	Vis-to-IR, and 87.52% Rank-1 and 81.18% mAP for IR-to-Vis. Note that
./reviews/deep_15.md:13604:  1026	Rank-20
./reviews/deep_15.md:13747:  1250	Lora: Low-rank adaptation of large language models, 2021, arXiv preprint arXiv:
./reviews/deep_18.md:135:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_18.md:147: exited 1 in 0ms:
./reviews/deep_18.md:176:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_18.md:182:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_18.md:196:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_18.md:277:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_18.md:285:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_18.md:293:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_18.md:325: exited 1 in 0ms:
./reviews/deep_18.md:356:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_18.md:385:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_18.md:388:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_18.md:389:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_18.md:466:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_18.md:470:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_18.md:484:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_18.md:490:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_18.md:624:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_18.md:763:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_18.md:838:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_18.md:1013:76:    76	of model parameters scales linearly with the number of source domains due to the increase of the
./reviews/deep_18.md:1030:104:   104	parameters of which usually increase linearly with the increasing of the number of person IDs. Below: In our
./reviews/deep_18.md:1071:193:   193	during training. LoRA [11] tries to learn low-rank matrices to approximate parameter updates.
./reviews/deep_18.md:1406:973:   973	reported Rank-1, while remaining RGB-only, methods with higher
./reviews/deep_18.md:1409:985:   985	HSOA clearly changes performance trends. HSOA synthesizes sameID but different-hairstyle views, explicitly decoupling hairstyle from
./reviews/deep_18.md:1560:292:   292	matching. Furthermore, Mix-Modality ReID (MM-ReID) introduces additional complexities. Beyond
./reviews/deep_18.md:1928:975:   975	These results clearly demonstrate the effectiveness of our proposed
./reviews/deep_18.md:3235:   709	decreases with iterations. At early stages of training, a larger
./reviews/deep_18.md:3499:   225	complicated by background noise, changes in viewpoint, and variations in pose. The complexity
./reviews/deep_18.md:3502:   228	MM-ReID introduces additional complexities. Beyond the challenges faced in VI-ReID, MM-ReID
./reviews/deep_18.md:3654:   380	modalities. This separation helps distinguish individuals clearly.
./reviews/deep_18.md:3804:   289	complexity escalates in Visible-Infrared ReID (VI-ReID), where the task involves matching images
./reviews/deep_18.md:3807:   292	matching. Furthermore, Mix-Modality ReID (MM-ReID) introduces additional complexities. Beyond
./reviews/deep_18.md:3988:   473	modalities. This separation helps distinguish individuals clearly.
./reviews/deep_18.md:4280:   628	4.1.2 Metrics. The cumulative matching characteristics [23] at Rank-1, Rank-5, Rank-10, mean
./reviews/deep_18.md:4289:   637	model on the RegDB, SYSU-MM01, and LLCM datasets in the Rank-1, mAP, and mINP metrics.
./reviews/deep_18.md:4291:   639	except for the Rank-1 metric, which shows an improvement in the RegDB and SYSU-MM01 datasets.
./reviews/deep_18.md:4309:   657	Fig. 4. The effect of different modality mixing ratios with respect to the AGW model performance in Rank-1,
./reviews/deep_18.md:4311:   659	on the other metrics except for the Rank-1 metrics that rise on some of the datasets. The origin stands for the
./reviews/deep_18.md:4323:   671	RegDB, SYSU-MM01, and LLCM in Rank-1, Rank-5, Rank-10, mAP, and mINP. Where Mode 1
./reviews/deep_18.md:4331:   679	improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. The addition of
./reviews/deep_18.md:4333:   681	metrics, which reflects the overall performance of the model, despite a slight decrease in the rank
./reviews/deep_18.md:4336:   684	in Rank-1, Rank-5, Rank-10, mAP, and mINIP metrics, respectively. Unlike Model 1 the performance
./reviews/deep_18.md:4337:   685	of the model with the addition of MBSOS improves on the Rank metric. The performance of other
./reviews/deep_18.md:4756:   800	denotes the second position in performance ranking among methods in comparison.
./reviews/deep_18.md:4783:   827	without employing any post-processing techniques, such as re-ranking or query expansion [49].
./reviews/deep_18.md:5065:   822	Rank-1
./reviews/deep_18.md:5083:   840	Rank-1
./reviews/deep_18.md:5099:   856	Fig. 4. The effect of different modality mixing ratios with respect to the AGW model performance in Rank-1,
./reviews/deep_18.md:5101:   858	on the other metrics except for the Rank-1 metrics that rise on some of the datasets. The origin stands for the
./reviews/deep_18.md:5114:   871	4.1.2 Metrics. The cumulative matching characteristics (CMC) [23] at Rank-1, Rank-5, Rank-10,
./reviews/deep_18.md:5123:   880	model on the RegDB, SYSU-MM01, and LLCM datasets in the Rank-1, mAP, and mINP metrics. It
./reviews/deep_18.md:5125:   882	except for the Rank-1 metric, which shows an improvement in the RegDB and SYSU-MM01 datasets.
./reviews/deep_18.md:5156:   913	Rank-1
./reviews/deep_18.md:5175:   932	Rank-5
./reviews/deep_18.md:5195:   952	Rank-10
./reviews/deep_18.md:5262:   695	Evaluation metrics. We report Rank-1 (R1) and mean Average
./reviews/deep_18.md:5271:   704	points in Rank-1) and is on par with, or slightly better than, recent
./reviews/deep_18.md:5533:   966	it delivers the best retrieval quality in mAP and competitive Rank-1
./reviews/deep_18.md:5539:   972	CC protocol, it clearly improves over CAL and matches the best
./reviews/deep_18.md:5540:   973	reported Rank-1, while remaining RGB-only, methods with higher
./reviews/deep_18.md:5552:   985	HSOA clearly changes performance trends. HSOA synthesizes sameID but different-hairstyle views, explicitly decoupling hairstyle from
./reviews/deep_18.md:5561:   994	rankings, rather than overfitting to textures or colors.
./reviews/deep_18.md:5640:   763	Rank-1 mAP
./reviews/deep_18.md:5659:   782	Rank-1
./reviews/deep_18.md:5685:   808	linearly decreases to 0.5 at the 40th epoch. In the binomial
./reviews/deep_18.md:5726:   849	be attributed to its more flexible pseudo-labels. After incorporating the GCN model, MPRD achieves dramatic performance advantages over SS, i.e., with rank-1 accuracy
./reviews/deep_18.md:5731:   854	over “MPRD w/o” in Rank-1 accuracy and mAP, respectively. A similar trend also appears on DukeMTMC-reID.
./reviews/deep_18.md:5732:   855	Additionally, “MPRD#” achieves 73.0% rank-1 accuracy
./reviews/deep_18.md:5739:   862	of 0.1. We observe that both rank-1 accuracy and mAP
./reviews/deep_18.md:5763:   886	Rank-1 mAP
./reviews/deep_18.md:5774:   897	Rank-1
./reviews/deep_18.md:5798:   921	labeled identities is 5, we observe that both rank-1 accuracy
./reviews/deep_18.md:5812:   935	rank-1 accuracy and 5.6% higher mAP than MLCR. Compared with JVTC, our MPRD achieves 10.1% higher rank1 accuracy and 9.3% higher mAP. On DukeMTMC-reID,
./reviews/deep_18.md:5813:   936	our MPRD achieves 2.2% higher rank-1 accuracy and 3.5%
./reviews/deep_18.md:5815:   938	has a sight 0.2% lower rank-1 accuracy but achieves 1.5%
./reviews/deep_18.md:5818:   941	MPRD achieves 37.7% rank-1 accuracy and 14.6% mAP.
./reviews/deep_18.md:5836:   959	Rank-1
./reviews/deep_18.md:5848:   971	Rank-5 Rank-10
./reviews/deep_18.md:5877:  1000	Rank-1
./reviews/deep_18.md:5889:  1012	Rank-5 Rank-10
./reviews/deep_18.md:5925:  1048	Rank-1
./reviews/deep_18.md:5931:  1054	Rank-5 Rank-10
./reviews/deep_18.md:6007:   896	curve is the rank-1 identification rate, which is abbreviated as R1.
./reviews/deep_18.md:6077:   966	CE alone, improving Rank-1 accuracy (R1) from 59.0% to 88.0% on
./reviews/deep_18.md:6086:   975	These results clearly demonstrate the effectiveness of our proposed
./reviews/deep_18.md:6101:   990	learning rates are initialized at 2 × 10−5 and linearly warmed up to
./reviews/deep_18.md:6109:   998	the learning rates are initialized at 6.5 × 10−5 and linearly warmed up
./reviews/deep_18.md:6124:  1013	The Euclidean distance is applied as the similarity measure to rank
./reviews/deep_18.md:6504:/bin/zsh -lc "pdftotext -f 1 -l 18 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | rg -n -i 'mixing ratio|modality confusion|ablation|hyperparameter|visualization|t-SNE|MBSOS|CIDHL|\\+M|\\+C|Rank-1|mINP|performance degradation'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_18.md:6553:628:   628	4.1.2 Metrics. The cumulative matching characteristics [23] at Rank-1, Rank-5, Rank-10, mean
./reviews/deep_18.md:6556:637:   637	model on the RegDB, SYSU-MM01, and LLCM datasets in the Rank-1, mAP, and mINP metrics.
./reviews/deep_18.md:6558:639:   639	except for the Rank-1 metric, which shows an improvement in the RegDB and SYSU-MM01 datasets.
./reviews/deep_18.md:6561:657:   657	Fig. 4. The effect of different modality mixing ratios with respect to the AGW model performance in Rank-1,
./reviews/deep_18.md:6563:659:   659	on the other metrics except for the Rank-1 metrics that rise on some of the datasets. The origin stands for the
./reviews/deep_18.md:6565:671:   671	RegDB, SYSU-MM01, and LLCM in Rank-1, Rank-5, Rank-10, mAP, and mINP. Where Mode 1
./reviews/deep_18.md:6568:679:   679	improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. The addition of
./reviews/deep_18.md:6571:684:   684	in Rank-1, Rank-5, Rank-10, mAP, and mINIP metrics, respectively. Unlike Model 1 the performance
./reviews/deep_18.md:6572:685:   685	of the model with the addition of MBSOS improves on the Rank metric. The performance of other
./reviews/deep_18.md:6573:715:   715	Rank-1
./reviews/deep_18.md:6574:754:   754	Rank-10
./reviews/deep_18.md:6576:835:   835	Rank-1
./reviews/deep_18.md:6577:874:   874	Rank-10
./reviews/deep_18.md:6580:935:   935	“+5.76%,” “+2.98%,” “+13.76%,” and “+1.33%” improvement in Rank-1, Rank-5, Rank-10, mAP, and
./reviews/deep_18.md:6582:965:   965	Rank-1
./reviews/deep_18.md:6583:1008:  1008	Rank-10
./reviews/deep_18.md:6585:1097:  1097	Rank-1
./reviews/deep_18.md:6586:1140:  1140	Rank-10
./reviews/deep_18.md:6588:1231:  1231	Rank-1
./reviews/deep_18.md:6589:1266:  1266	Rank-10
./reviews/deep_18.md:6591:1339:  1339	Rank-1
./reviews/deep_18.md:6592:1374:  1374	Rank-10
./reviews/deep_18.md:6595:1429:  1429	“+2.67%” improvements in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. Adding
./reviews/deep_18.md:6599:1451:  1451	Rank-1
./reviews/deep_18.md:6600:1470:  1470	Rank-10 mAP
./reviews/deep_18.md:6603:1508:  1508	Rank-1
./reviews/deep_18.md:6604:1527:  1527	Rank-10 mAP
./reviews/deep_18.md:6606:1554:  1554	“+6.29%” “+5.98%,” and “+0.31%” improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics,
./reviews/deep_18.md:6609:1558:  1558	improvements in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. Adding CIDHL and
./reviews/deep_18.md:6613:1566:  1566	Rank-1, Rank-5, Rank-10, and mAP metrics. We show the ablation experiments for the 𝛿 parameter
./reviews/deep_18.md:6615:1571:  1571	in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics (𝛿 is set to 0.3), and at most “+18.54%,” “+5.60%,”
./reviews/deep_18.md:6617:1576:  1576	improvement (𝛿 is set to 0.2) in the four metrics of Rank-1, Rank-5, Rank-10, and mAP, respectively,
./reviews/deep_18.md:6619:1594:  1594	Rank-1
./reviews/deep_18.md:6620:1613:  1613	Rank-10
./reviews/deep_18.md:6623:1644:  1644	Rank-1
./reviews/deep_18.md:6624:1663:  1663	Rank-10 mAP
./reviews/deep_18.md:6627:1692:  1692	and “+2.14%” improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. In
./reviews/deep_18.md:6628:1694:  1694	showing “+30.60%,” “+15.99%,” “+9.57%,” “+43.24%,” and “+4.00%” improvement in Rank-1, Rank-5,
./reviews/deep_18.md:6629:1695:  1695	Rank-10, mAP, and mINP metrics, respectively.
./reviews/deep_18.md:6659:76:    76	of model parameters scales linearly with the number of source domains due to the increase of the
./reviews/deep_18.md:6673:104:   104	parameters of which usually increase linearly with the increasing of the number of person IDs. Below: In our
./reviews/deep_18.md:6701:193:   193	during training. LoRA [11] tries to learn low-rank matrices to approximate parameter updates.
./reviews/deep_18.md:6786:800:   800	denotes the second position in performance ranking among methods in comparison.
./reviews/deep_18.md:6804:972:   972	performance ranking among methods in comparison.
./reviews/deep_18.md:6811:1104:  1104	performance ranking among methods in comparison.
./reviews/deep_18.md:6866:1343:  1343	protocol-3: (1) The imbalance in the quantity of data in the source domain increases the difficulty of
./reviews/deep_18.md:6917:  1566	Rank-1, Rank-5, Rank-10, and mAP metrics. We show the ablation experiments for the 𝛿 parameter
./reviews/deep_18.md:6922:  1571	in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics (𝛿 is set to 0.3), and at most “+18.54%,” “+5.60%,”
./reviews/deep_18.md:6927:  1576	improvement (𝛿 is set to 0.2) in the four metrics of Rank-1, Rank-5, Rank-10, and mAP, respectively,
./reviews/deep_18.md:6945:  1594	Rank-1
./reviews/deep_18.md:6954:  1603	Rank-5
./reviews/deep_18.md:6964:  1613	Rank-10
./reviews/deep_18.md:6995:  1644	Rank-1
./reviews/deep_18.md:7004:  1653	Rank-5
./reviews/deep_18.md:7014:  1663	Rank-10 mAP
./reviews/deep_18.md:7043:  1692	and “+2.14%” improvement in Rank-1, Rank-5, Rank-10, mAP, and mINP metrics, respectively. In
./reviews/deep_18.md:7045:  1694	showing “+30.60%,” “+15.99%,” “+9.57%,” “+43.24%,” and “+4.00%” improvement in Rank-1, Rank-5,
./reviews/deep_18.md:7046:  1695	Rank-10, mAP, and mINP metrics, respectively.
./reviews/deep_18.md:7122:  1154	Rank-1
./reviews/deep_18.md:7156:  1188	Rank-1
./reviews/deep_18.md:7214:  1246	mAP Rank-1
./reviews/deep_18.md:7298:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
./reviews/deep_18.md:7306:4. **证据闭环**：最关键的消融是和 KNN、相似度阈值、MPLP 比。Market 上相似度阈值是 72.8 Rank-1、39.9 mAP，MPRD 到 83.0、51.1。去掉大量无标签数据训练 GCN 的版本只有 75.8、43.1，说明不是 metadata 本身撑起来的。t-SNE 里加 GCN 后同身份更聚、易混身份分开，直接对应“邻域关系比点对相似度更可靠”。
./reviews/deep_18.md:7319:1. **触发观察**：正式版进一步把观察说得更工程化，单模态 ReID 和传统 VI-ReID 都没有覆盖“query 集和 gallery 集同时混合可见光与红外”的情况。作者强调 Rank-1 有时会因为同模态易匹配而变好，但 mAP 和 mINP 下降，说明整体检索质量变差。
./reviews/deep_18.md:7353:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
./reviews/deep_18.md:7361:4. **证据闭环**：最关键的消融是和 KNN、相似度阈值、MPLP 比。Market 上相似度阈值是 72.8 Rank-1、39.9 mAP，MPRD 到 83.0、51.1。去掉大量无标签数据训练 GCN 的版本只有 75.8、43.1，说明不是 metadata 本身撑起来的。t-SNE 里加 GCN 后同身份更聚、易混身份分开，直接对应“邻域关系比点对相似度更可靠”。
./reviews/deep_18.md:7374:1. **触发观察**：正式版进一步把观察说得更工程化，单模态 ReID 和传统 VI-ReID 都没有覆盖“query 集和 gallery 集同时混合可见光与红外”的情况。作者强调 Rank-1 有时会因为同模态易匹配而变好，但 mAP 和 mINP 下降，说明整体检索质量变差。
./novelty_osac.md:16:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'seen-ID 过坍缩定义成 open-set ReID 失败变量 + 训练期抗坍缩保 unseen 迁移': (a)Neural Collapse(Papyan 2020)用到 ReID / open-set / metric learning 没; (b)dimensional collapse / rank collapse in SSL/contrastive(Jing et al, Hua et al); (c)VICReg/Barlow Twins/whitening/decorrelation 泛化零件; (d)ProNet / prototype-projection / classifier-reuse ReID; (e)generalizable/open-set/DG ReID 里有没有'过坍缩伤 unseen 迁移'诊断; (f)feature whitening / group whitening ReID(AMS / SNR / IN 类); (g)spectral/rank regularization metric learning; (h)open-set recognition 里的 feature collapse。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./novelty_osac.md:28:web search: "domain generalization" "person re-identification" "rank" "feature"
./novelty_osac.md:36:web search: "spectral regularization" "deep metric learning" rank collapse
./novelty_osac.md:42:web search: "rank regularization" "embedding" "metric learning"
./novelty_osac.md:110:| (g) spectral/rank regularization metric learning | **局部撞车** | OLE 是 deep metric/classification 的 low-rank orthogonal embedding，方向甚至是主动类内低秩；CLOP 用 orthonormal prototypes 防 contrastive collapse；Top-PC removal 在 NLP embedding 后处理也有先例。没撞到“ReID 训练期 spectral floor + Top-PC Dropout 保 unseen 证据”。证据：[OLÉ](https://arxiv.org/abs/1712.01727)、[CLOP](https://arxiv.org/abs/2403.18699)、[All-but-the-Top](https://arxiv.org/abs/1702.01417) | 7/10 |
./novelty_osac.md:117:1. baseline 训练末期出现 effective rank 下降、谱尾能量下降、top-PC 与 classifier/prototype subspace 对齐增强，并且这些量和 unseen mAP/R1 负相关。  
./novelty_osac.md:118:2. k-reciprocal/re-ranking 只能重排邻域，不能恢复 embedding 谱维度；OSAC 不靠 test-time trick。  
./novelty_osac.md:119:3. OSAC 胜过 VICReg-like variance/cov、GW/whitening、random dropout、early stopping、label smoothing、ProNet/ProNet-like prototype feature 等强对照。
./novelty_osac.md:132:| (g) spectral/rank regularization metric learning | **局部撞车** | OLE 是 deep metric/classification 的 low-rank orthogonal embedding，方向甚至是主动类内低秩；CLOP 用 orthonormal prototypes 防 contrastive collapse；Top-PC removal 在 NLP embedding 后处理也有先例。没撞到“ReID 训练期 spectral floor + Top-PC Dropout 保 unseen 证据”。证据：[OLÉ](https://arxiv.org/abs/1712.01727)、[CLOP](https://arxiv.org/abs/2403.18699)、[All-but-the-Top](https://arxiv.org/abs/1702.01417) | 7/10 |
./novelty_osac.md:139:1. baseline 训练末期出现 effective rank 下降、谱尾能量下降、top-PC 与 classifier/prototype subspace 对齐增强，并且这些量和 unseen mAP/R1 负相关。  
./novelty_osac.md:140:2. k-reciprocal/re-ranking 只能重排邻域，不能恢复 embedding 谱维度；OSAC 不靠 test-time trick。  
./novelty_osac.md:141:3. OSAC 胜过 VICReg-like variance/cov、GW/whitening、random dropout、early stopping、label smoothing、ProNet/ProNet-like prototype feature 等强对照。
./batches/bap:4:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./ondisk_pivot.md:14:一个 ReID 团队连续 4 个 frozen-image 方向被零训练 kill-switch 干净否掉(B 航拍不确定性包含=前提错 / GOPL SMPL正样本可靠性=occlusion-count廉价代理 / Gallery Hubness=诊断真但remedy被k-reciprocal完全占 / OSAC谱过坍缩=前提倒置训练反而升rank)。**frozen-image-retrieval 的隐藏变量都被成熟 test-time 工具(k-reciprocal/camera-aware re-ranking)碾压或证伪, 这条路穷尽。** 唯一真发现=gallery 负向 in-degree hubness(M(q) 解释 AP 误差 rho+0.60, analysis 级非方法)。
./ondisk_pivot.md:23: exited 1 in 0ms:
./ondisk_pivot.md:30: exited 1 in 0ms:
./ondisk_pivot.md:39: exited 1 in 0ms:
./ondisk_pivot.md:44: exited 1 in 0ms:
./ondisk_pivot.md:118:web search: "RSTPReid" "2024" "text-to-image person re-identification" "Rank-1"
./ondisk_pivot.md:120:web search: "RSTPReID" "Rank-1" "2024"
./ondisk_pivot.md:136:联网结果已经给出一个重要边界：RSTPReid 本身不是空地，IRRA/RDE/APTM/CADA/ICL 都已经覆盖了 CLIP 微调、噪声对应、属性预训练、局部双向匹配和 MLLM rerank。下一步我把这个边界和你们本地四个 kill-switch 记录对齐，筛一个仍有“问题重定义”空间的方向。
./ondisk_pivot.md:153:一个 ReID 团队连续 4 个 frozen-image 方向被零训练 kill-switch 干净否掉(B 航拍不确定性包含=前提错 / GOPL SMPL正样本可靠性=occlusion-count廉价代理 / Gallery Hubness=诊断真但remedy被k-reciprocal完全占 / OSAC谱过坍缩=前提倒置训练反而升rank)。**frozen-image-retrieval 的隐藏变量都被成熟 test-time 工具(k-reciprocal/camera-aware re-ranking)碾压或证伪, 这条路穷尽。** 唯一真发现=gallery 负向 in-degree hubness(M(q) 解释 AP 误差 rho+0.60, analysis 级非方法)。
./ondisk_pivot.md:162: exited 1 in 0ms:
./ondisk_pivot.md:169: exited 1 in 0ms:
./ondisk_pivot.md:178: exited 1 in 0ms:
./ondisk_pivot.md:183: exited 1 in 0ms:
./ondisk_pivot.md:257:web search: "RSTPReid" "2024" "text-to-image person re-identification" "Rank-1"
./ondisk_pivot.md:259:web search: "RSTPReID" "Rank-1" "2024"
./ondisk_pivot.md:275:联网结果已经给出一个重要边界：RSTPReid 本身不是空地，IRRA/RDE/APTM/CADA/ICL 都已经覆盖了 CLIP 微调、噪声对应、属性预训练、局部双向匹配和 MLLM rerank。下一步我把这个边界和你们本地四个 kill-switch 记录对齐，筛一个仍有“问题重定义”空间的方向。
./ondisk_pivot.md:295:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'gallery 负向 in-degree 定义成 ReID 失败主变量 + 训练端 anti-hub': (a)hubness in 高维检索(Radovanovic hubness / CSLS / Mutual Proximity / local scaling)用到 ReID 没; (b)k-reciprocal re-ranking / neighbor-based re-ranking ReID; (c)hard-negative mining / anti-hub margin / proxy metric learning; (d)Pose2ID / feature centralization / neighbor aggregation ReID; (e)graph/relation ReID(SGGNN等); (f)long-tail / popularity-bias / false-match-rate retrieval。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./ondisk_pivot.md:311:web search: person re-identification re-ranking k-reciprocal encoding CVPR 2017
./ondisk_pivot.md:379:| (b) ReID k-reciprocal / neighbor re-ranking | **存活，但需强对照** | 这类已经把 ReID 当邻域拓扑问题处理，但在**测试后处理**改距离/排名，不是训练单 embedding；也不是把少数 gallery 的跨 ID 负 in-degree 当失败主变量。 | k-reciprocal 用 reciprocal NN、Jaccard distance、local query expansion；ECN / PSE 也是 unsupervised re-ranking。链接：[Zhong k-reciprocal](https://arxiv.org/abs/1701.08398), [PSE/ECN](https://arxiv.org/abs/1711.10378) | 8 |
./ondisk_pivot.md:380:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./ondisk_pivot.md:381:| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
./ondisk_pivot.md:389:- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
./ondisk_pivot.md:390:- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。
./ondisk_pivot.md:399:| (b) ReID k-reciprocal / neighbor re-ranking | **存活，但需强对照** | 这类已经把 ReID 当邻域拓扑问题处理，但在**测试后处理**改距离/排名，不是训练单 embedding；也不是把少数 gallery 的跨 ID 负 in-degree 当失败主变量。 | k-reciprocal 用 reciprocal NN、Jaccard distance、local query expansion；ECN / PSE 也是 unsupervised re-ranking。链接：[Zhong k-reciprocal](https://arxiv.org/abs/1701.08398), [PSE/ECN](https://arxiv.org/abs/1711.10378) | 8 |
./ondisk_pivot.md:400:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./ondisk_pivot.md:401:| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
./ondisk_pivot.md:409:- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
./ondisk_pivot.md:410:- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。
./ondisk_pivot.md:430:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'seen-ID 过坍缩定义成 open-set ReID 失败变量 + 训练期抗坍缩保 unseen 迁移': (a)Neural Collapse(Papyan 2020)用到 ReID / open-set / metric learning 没; (b)dimensional collapse / rank collapse in SSL/contrastive(Jing et al, Hua et al); (c)VICReg/Barlow Twins/whitening/decorrelation 泛化零件; (d)ProNet / prototype-projection / classifier-reuse ReID; (e)generalizable/open-set/DG ReID 里有没有'过坍缩伤 unseen 迁移'诊断; (f)feature whitening / group whitening ReID(AMS / SNR / IN 类); (g)spectral/rank regularization metric learning; (h)open-set recognition 里的 feature collapse。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./ondisk_pivot.md:442:web search: "domain generalization" "person re-identification" "rank" "feature"
./ondisk_pivot.md:450:web search: "spectral regularization" "deep metric learning" rank collapse
./ondisk_pivot.md:456:web search: "rank regularization" "embedding" "metric learning"
./ondisk_pivot.md:524:| (g) spectral/rank regularization metric learning | **局部撞车** | OLE 是 deep metric/classification 的 low-rank orthogonal embedding，方向甚至是主动类内低秩；CLOP 用 orthonormal prototypes 防 contrastive collapse；Top-PC removal 在 NLP embedding 后处理也有先例。没撞到“ReID 训练期 spectral floor + Top-PC Dropout 保 unseen 证据”。证据：[OLÉ](https://arxiv.org/abs/1712.01727)、[CLOP](https://arxiv.org/abs/2403.18699)、[All-but-the-Top](https://arxiv.org/abs/1702.01417) | 7/10 |
./ondisk_pivot.md:531:1. baseline 训练末期出现 effective rank 下降、谱尾能量下降、top-PC 与 classifier/prototype subspace 对齐增强，并且这些量和 unseen mAP/R1 负相关。  
./ondisk_pivot.md:532:2. k-reciprocal/re-ranking 只能重排邻域，不能恢复 embedding 谱维度；OSAC 不靠 test-time trick。  
./ondisk_pivot.md:533:3. OSAC 胜过 VICReg-like variance/cov、GW/whitening、random dropout、early stopping、label smoothing、ProNet/ProNet-like prototype feature 等强对照。
./ondisk_pivot.md:546:| (g) spectral/rank regularization metric learning | **局部撞车** | OLE 是 deep metric/classification 的 low-rank orthogonal embedding，方向甚至是主动类内低秩；CLOP 用 orthonormal prototypes 防 contrastive collapse；Top-PC removal 在 NLP embedding 后处理也有先例。没撞到“ReID 训练期 spectral floor + Top-PC Dropout 保 unseen 证据”。证据：[OLÉ](https://arxiv.org/abs/1712.01727)、[CLOP](https://arxiv.org/abs/2403.18699)、[All-but-the-Top](https://arxiv.org/abs/1702.01417) | 7/10 |
./ondisk_pivot.md:553:1. baseline 训练末期出现 effective rank 下降、谱尾能量下降、top-PC 与 classifier/prototype subspace 对齐增强，并且这些量和 unseen mAP/R1 负相关。  
./ondisk_pivot.md:554:2. k-reciprocal/re-ranking 只能重排邻域，不能恢复 embedding 谱维度；OSAC 不靠 test-time trick。  
./ondisk_pivot.md:555:3. OSAC 胜过 VICReg-like variance/cov、GW/whitening、random dropout、early stopping、label smoothing、ProNet/ProNet-like prototype feature 等强对照。
./ondisk_pivot.md:562:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./ondisk_pivot.md:579:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./ondisk_pivot.md:580:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./ondisk_pivot.md:586:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./ondisk_pivot.md:621:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./ondisk_pivot.md:653:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./ondisk_pivot.md:846:./ondisk_pivot.md:14:一个 ReID 团队连续 4 个 frozen-image 方向被零训练 kill-switch 干净否掉(B 航拍不确定性包含=前提错 / GOPL SMPL正样本可靠性=occlusion-count廉价代理 / Gallery Hubness=诊断真但remedy被k-reciprocal完全占 / OSAC谱过坍缩=前提倒置训练反而升rank)。**frozen-image-retrieval 的隐藏变量都被成熟 test-time 工具(k-reciprocal/camera-aware re-ranking)碾压或证伪, 这条路穷尽。** 唯一真发现=gallery 负向 in-degree hubness(M(q) 解释 AP 误差 rho+0.60, analysis 级非方法)。
./ondisk_pivot.md:861:./ondisk_pivot.md:118:web search: "RSTPReid" "2024" "text-to-image person re-identification" "Rank-1"
./ondisk_pivot.md:862:./ondisk_pivot.md:120:web search: "RSTPReID" "Rank-1" "2024"
./ondisk_pivot.md:867:./ondisk_pivot.md:136:联网结果已经给出一个重要边界：RSTPReid 本身不是空地，IRRA/RDE/APTM/CADA/ICL 都已经覆盖了 CLIP 微调、噪声对应、属性预训练、局部双向匹配和 MLLM rerank。下一步我把这个边界和你们本地四个 kill-switch 记录对齐，筛一个仍有“问题重定义”空间的方向。
./ondisk_pivot.md:868:./ondisk_pivot.md:153:一个 ReID 团队连续 4 个 frozen-image 方向被零训练 kill-switch 干净否掉(B 航拍不确定性包含=前提错 / GOPL SMPL正样本可靠性=occlusion-count廉价代理 / Gallery Hubness=诊断真但remedy被k-reciprocal完全占 / OSAC谱过坍缩=前提倒置训练反而升rank)。**frozen-image-retrieval 的隐藏变量都被成熟 test-time 工具(k-reciprocal/camera-aware re-ranking)碾压或证伪, 这条路穷尽。** 唯一真发现=gallery 负向 in-degree hubness(M(q) 解释 AP 误差 rho+0.60, analysis 级非方法)。
./ondisk_pivot.md:883:./ondisk_pivot.md:257:web search: "RSTPReid" "2024" "text-to-image person re-identification" "Rank-1"
./ondisk_pivot.md:884:./ondisk_pivot.md:259:web search: "RSTPReID" "Rank-1" "2024"
./ondisk_pivot.md:889:./ondisk_pivot.md:275:联网结果已经给出一个重要边界：RSTPReid 本身不是空地，IRRA/RDE/APTM/CADA/ICL 都已经覆盖了 CLIP 微调、噪声对应、属性预训练、局部双向匹配和 MLLM rerank。下一步我把这个边界和你们本地四个 kill-switch 记录对齐，筛一个仍有“问题重定义”空间的方向。
./ondisk_pivot.md:891:./ondisk_pivot.md:295:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'gallery 负向 in-degree 定义成 ReID 失败主变量 + 训练端 anti-hub': (a)hubness in 高维检索(Radovanovic hubness / CSLS / Mutual Proximity / local scaling)用到 ReID 没; (b)k-reciprocal re-ranking / neighbor-based re-ranking ReID; (c)hard-negative mining / anti-hub margin / proxy metric learning; (d)Pose2ID / feature centralization / neighbor aggregation ReID; (e)graph/relation ReID(SGGNN等); (f)long-tail / popularity-bias / false-match-rate retrieval。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./ondisk_pivot.md:904:./ondisk_pivot.md:380:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./ondisk_pivot.md:909:./ondisk_pivot.md:400:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./ondisk_pivot.md:912:./ondisk_pivot.md:430:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'seen-ID 过坍缩定义成 open-set ReID 失败变量 + 训练期抗坍缩保 unseen 迁移': (a)Neural Collapse(Papyan 2020)用到 ReID / open-set / metric learning 没; (b)dimensional collapse / rank collapse in SSL/contrastive(Jing et al, Hua et al); (c)VICReg/Barlow Twins/whitening/decorrelation 泛化零件; (d)ProNet / prototype-projection / classifier-reuse ReID; (e)generalizable/open-set/DG ReID 里有没有'过坍缩伤 unseen 迁移'诊断; (f)feature whitening / group whitening ReID(AMS / SNR / IN 类); (g)spectral/rank regularization metric learning; (h)open-set recognition 里的 feature collapse。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./ondisk_pivot.md:929:./HUBNESS_KILLSWITCH_DESIGN.md:8:- vs k-reciprocal re-ranking: 它用 reciprocal neighbor 做 test-time ranking, 不定义 hubness 为失败变量, 不做训练端 anti-hub。
./ondisk_pivot.md:947:./analyses/deep_12.txt:28:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./ondisk_pivot.md:1078:./reviews/deep_23.md:3912:body parts and carried items at the pixel level. Moreover, uses paired image and text encoders to learn a shared embedsince person ReID is inherently formulated as a ranking ding space via contrastive learning, benefiting applications such
./ondisk_pivot.md:1362:./reviews/deep_28.md:5236:1174:rank of LoRA r is set to 32, while for RSTPReid, due to its
./ondisk_pivot.md:1423:./reviews/deep_28.md:6902:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./ondisk_pivot.md:1433:./reviews/deep_28.md:6992:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./ondisk_pivot.md:1457:./reviews/deep_8.md:342:that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
./ondisk_pivot.md:1488:./reviews/deep_8.md:552:𝑣 is linearly mapped into the joint image-text embedding
./ondisk_pivot.md:1491:./reviews/deep_8.md:564:linearly mapped into the joint image-text representation space, generating a compressed global
./ondisk_pivot.md:1493:./reviews/deep_8.md:571:Adopted from IRRA [21], we introduce a novel A-SDM module, which aims to adaptively pull the
./ondisk_pivot.md:1499:./reviews/deep_8.md:601:image for 𝑇𝑖 at rank-𝑘, where 𝑘 > 1. Different from IRRA [21], we propose to derive an adaptive
./ondisk_pivot.md:1653:./reviews/deep_8.md:2143:Computational Complexity: We compare the computational cost and inference time of CoRL with classic methods in Table 5. TIPCB and CFine build fine-grained correspondences by learning local features, reducing retrieval efficiency due to pairwise similarity calculations. IRRA and
./ondisk_pivot.md:1657:./reviews/deep_8.md:2221:ages with the same identity as the given query text are ranked
./ondisk_pivot.md:1673:./reviews/deep_8.md:2329:uncertainty-based learning for text-based person retrieval.
./ondisk_pivot.md:1678:./reviews/deep_8.md:2357:that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
./ondisk_pivot.md:1700:./reviews/deep_8.md:2493:image-text pairs in a shared embedding space. It adaptively adjusts to narrow the distance
./ondisk_pivot.md:1712:./reviews/deep_8.md:2650:𝑣 is linearly mapped into the joint image-text
./ondisk_pivot.md:1715:./reviews/deep_8.md:2661:𝑡 , is linearly mapped into the joint image-text representation space, generating a compressed
./ondisk_pivot.md:1718:./reviews/deep_8.md:2676:module, which aims to adaptively pull the unmatched positive image-text pairs into a shared
./ondisk_pivot.md:1723:./reviews/deep_8.md:2696:image for 𝑇𝑖 at rank-𝑘, where 𝑘 > 1. Different from IRRA [21], we propose to derive an adaptive
./ondisk_pivot.md:1808:./reviews/deep_8.md:5438:Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
./ondisk_pivot.md:1858:./reviews/deep_8.md:6962:Rank-1 and 66.66% mAP, while surpassing IRRA by 0.74% in Rank-5 and 0.41% in Rank-10. When
./ondisk_pivot.md:1859:./reviews/deep_8.md:6965:NAM-based backbone attains 91.33% in Rank-5, outperforming IRRA with the HAM-based backbone
./ondisk_pivot.md:1861:./reviews/deep_8.md:6969:mAP, respectively, outperforming IRRA by 1.55% in Rank-1 and 2.05% in Rank-5. When adopting
./ondisk_pivot.md:1862:./reviews/deep_8.md:6971:IRRA by 0.45% in Rank-1 with both the NAM-based and HAM-based backbones. Notably, FMFA
./ondisk_pivot.md:1877:./reviews/deep_8.md:7279:IRRA, FMFA shows a notable improvement of 0.81% Rank-1 and 1.23% mAP, which is meaningful
./ondisk_pivot.md:1884:./reviews/deep_8.md:7484:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./ondisk_pivot.md:1989:./reviews/deep_8.md:8310:74.16% Rank-1 and 66.66% mAP, while surpassing IRRA by 0.74% in Rank-5 and 0.41% in Rank-10.
./ondisk_pivot.md:1990:./reviews/deep_8.md:8313:with NAM-based backbone attains 91.33% in Rank-5, outperforming IRRA with the HAM-based
./ondisk_pivot.md:1992:./reviews/deep_8.md:8318:mAP, respectively, outperforming IRRA by 1.55% in Rank-1 and 2.05% in Rank-5. When adopting
./ondisk_pivot.md:2008:./reviews/deep_8.md:8618:IRRA by 0.45% in Rank-1 with both the NAM-based and HAM-based backbones. Notably, FMFA
./ondisk_pivot.md:2014:./reviews/deep_8.md:8729:IRRA, FMFA shows a notable improvement of 0.81% Rank-1 and 1.23% mAP, which is meaningful
./ondisk_pivot.md:2015:./reviews/deep_8.md:8731:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./ondisk_pivot.md:2024:./reviews/deep_8.md:8808:Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
./ondisk_pivot.md:2071:./reviews/deep_8.md:10105:yields slight gains, outperforming IRRA by 0.24% and 0.16% in Rank-1 with the NAM-based and
./ondisk_pivot.md:2073:./reviews/deep_8.md:10135:respectively, as well as a 1.55% drop in Rank-5 and a 0.58% decline in mAP on the RSTPReid dataset.
./ondisk_pivot.md:2074:./reviews/deep_8.md:10143:RSTPReid dataset. However, EFA causes a 0.11% drop in Rank-5 on ICFG-PEDES, suggesting that
./ondisk_pivot.md:2075:./reviews/deep_8.md:10150:and 0.95% drops in Rank-1 on CUHK-PEDES, ICFG-PEDES, and RSTPReid, and causes a 1.23% drop
./ondisk_pivot.md:2077:./reviews/deep_8.md:10301:on RSTPReid to 50 s vs. 91 s on ICFG-PEDES. These comparisons clearly demonstrate that FMFA
./ondisk_pivot.md:2193:./reviews/deep_13.md:7083:/bin/zsh -lc "pdftotext -f 8 -l 12 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|gradient consistency|consistency|global aggregated|GAGA|Table|Fig|iterations|early|late|transfer|black-box|white-box|Visualization|parameter|inner|m |beta|Input transformation|ensemble\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./ondisk_pivot.md:2276:./reviews/deep_18.md:5561:   994	rankings, rather than overfitting to textures or colors.
./ondisk_pivot.md:2281:./reviews/deep_18.md:6504:/bin/zsh -lc "pdftotext -f 1 -l 18 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' - | nl -ba | rg -n -i 'mixing ratio|modality confusion|ablation|hyperparameter|visualization|t-SNE|MBSOS|CIDHL|\\+M|\\+C|Rank-1|mINP|performance degradation'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./ondisk_pivot.md:2347:./reviews/deep_25.md:3014:local semantic element and all semantic elements. Correspondingly, connections can be established among diﬀerent semantics to enhance contextual semantic representation. Accordingly, we can clearly observe
./ondisk_pivot.md:2369:./reviews/deep_25.md:5066:and visual features via adaptive refinement. The final class-level textual prototypes derived from
./ondisk_pivot.md:2551:./reviews/deep_25.md:7165:multi-head attention mechanism, which captures contextual dependencies of textures from diﬀerent perspectives through parallel multiple attention heads, and adaptively aggregates global semantic information.
./ondisk_pivot.md:2571:./reviews/deep_25.md:8151:and 4.2 %/4.6 % in Rank-1/mAP, respectively. TAD leverages the continuity of texture features to further enhance the target features that
./ondisk_pivot.md:2662:./reviews/deep_25.md:12029:multi-head attention mechanism, which captures contextual dependencies of textures from diﬀerent perspectives through parallel multiple attention heads, and adaptively aggregates global semantic information.
./ondisk_pivot.md:2689:./reviews/deep_25.md:12642:/bin/zsh -lc "pdftotext -f 8 -l 14 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' - | rg -n \"(Ablation|SD-3DGF|MRIE|ECS|local similarity|LSL|appearance|gait|clothes-changing|Table|Fig\\.|Consistency|without|w/o|Impact|Effect|Contribution|ablation|representative|VCCR|CCVID|CCV|MARS|Rank|mAP)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./ondisk_pivot.md:2690:./reviews/deep_25.md:12906:/bin/zsh -lc "pdftotext -f 8 -l 14 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | rg -n \"(Ablation|ablation|ATG|CTP|TRM|prototype|Prototype|attribute|contour|text|Text|SYSU|RegDB|Table|Figure|Fig\\.|Effect|w/o|without|modal|alignment|Rank|mAP|loss|TAP)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./ondisk_pivot.md:2759:./reviews/deep_25.md:14077:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./ondisk_pivot.md:2761:./reviews/deep_25.md:14083:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./ondisk_pivot.md:2762:./reviews/deep_25.md:14085:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./ondisk_pivot.md:2764:./reviews/deep_25.md:14089:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./ondisk_pivot.md:2771:./reviews/deep_25.md:14156:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./ondisk_pivot.md:2773:./reviews/deep_25.md:14162:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./ondisk_pivot.md:2774:./reviews/deep_25.md:14164:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./ondisk_pivot.md:2776:./reviews/deep_25.md:14168:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./ondisk_pivot.md:2783:./debate/d_2.md:25:角色=**蓝队(降级 Hubness)**。为'别投训练, 降级转 r_2 备胎 Rank-Instability'辩护: D2 是铁证——k-reciprocal/camera **免费 test-time** 就在 mAP 上赢, 一个训练版要 beat camera-aware k-reciprocal 是高 bar 且小 headroom(Market 饱和)。用联网查: (a)hubness-aware training / anti-hub margin 历史上是否真能 beat 强 re-ranking(还是总被 k-reciprocal 这类盖过); (b)R1 赢 mAP 输是不是只是把 ranking 重排没真增加判别信息; (c)团队反复投不确定训练的教训。论证'诊断真≠方法能发', 该把诊断当一个 observation 写进别的稿/换 r_2。信心 1-10。
./ondisk_pivot.md:2789:./debate/d_2.md:77:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./ondisk_pivot.md:2790:./debate/d_2.md:79:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./ondisk_pivot.md:2792:./debate/d_2.md:84:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./ondisk_pivot.md:2793:./debate/d_2.md:92:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./ondisk_pivot.md:2796:./debate/d_2.md:117:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./ondisk_pivot.md:2797:./debate/d_2.md:119:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./ondisk_pivot.md:2799:./debate/d_2.md:124:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./ondisk_pivot.md:2800:./debate/d_2.md:132:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./ondisk_pivot.md:2885:./reviews/deep_19.md:4629:Rank-1 by 2.0%, validating multi-granularity textual disentanglement. Masked Branch Disentanglement (M LB).
./ondisk_pivot.md:3000:./reviews/deep_14.md:1426:138:1) The randomness of masked language: Given the complexity of semantic information in textual descriptions, vanilla MLM-based methods
./ondisk_pivot.md:3007:./reviews/deep_14.md:1443:175:a dual-path local alignment network to extract local visual and textual representations from horizontally segmented image patches. Subsequently, local representations are aligned adaptively with a multi-stage
./ondisk_pivot.md:3301:./reviews/deep_22.md:842:incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
./ondisk_pivot.md:3302:./reviews/deep_22.md:861:way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
./ondisk_pivot.md:3303:./reviews/deep_22.md:880:learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
./ondisk_pivot.md:3323:./reviews/deep_22.md:1018:texts with low uncertainty is primarily determined by the
./ondisk_pivot.md:3324:./reviews/deep_22.md:1020:while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./ondisk_pivot.md:3347:./reviews/deep_22.md:1336:introduces uncertainty by randomly masking textual tokens.
./ondisk_pivot.md:3353:./reviews/deep_22.md:1386:ICFG-PEDES [21], RSTPReid [54] and adopt the Rank-k and
./ondisk_pivot.md:3358:./reviews/deep_22.md:1451:(MLLMs) for diverse textual generation, PDA improves Rank1 by 1.25% and mAP by 0.01%. When compared with MUM,
./ondisk_pivot.md:3428:./reviews/deep_22.md:6160:between intra-modality image-text pairs. As shown in Table 5, the direct alignment (1) reduces the Rank-1 accuracy
./ondisk_pivot.md:3529:./reviews/deep_22.md:7327:1292:  1292	between intra-modality image-text pairs. As shown in Table 5, the direct alignment (1) reduces the Rank-1 accuracy
./ondisk_pivot.md:3573:./reviews/deep_22.md:7911:   392	texts with low uncertainty is primarily determined by the
./ondisk_pivot.md:3574:./reviews/deep_22.md:7913:   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./ondisk_pivot.md:3597:./reviews/deep_22.md:8229:   710	introduces uncertainty by randomly masking textual tokens.
./ondisk_pivot.md:3603:./reviews/deep_22.md:8279:   760	ICFG-PEDES [21], RSTPReid [54] and adopt the Rank-k and
./ondisk_pivot.md:3650:./reviews/deep_22.md:9247:/bin/zsh -lc "pdftotext 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | rg -n \"Ablation|TABLE|Distribution|Visualization|Parameter|containment|CMC|IMC|DRM|masked|uncertainty|Domain Generalization|Robustness\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./ondisk_pivot.md:3656:./reviews/deep_22.md:9286:216:   216	incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
./ondisk_pivot.md:3657:./reviews/deep_22.md:9290:235:   235	way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
./ondisk_pivot.md:3658:./reviews/deep_22.md:9294:254:   254	learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
./ondisk_pivot.md:3659:./reviews/deep_22.md:9306:392:   392	texts with low uncertainty is primarily determined by the
./ondisk_pivot.md:3660:./reviews/deep_22.md:9307:394:   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./ondisk_pivot.md:3668:./reviews/deep_22.md:9341:710:   710	introduces uncertainty by randomly masking textual tokens.
./ondisk_pivot.md:3675:./reviews/deep_22.md:9408:1156:  1156	proves that modeling visual-textual uncertainty through probabilistic distributions is a more efficient and powerful paradigm
./ondisk_pivot.md:3677:./reviews/deep_22.md:9424:1312:  1312	uncertainty modeling and semantic alignment for text-to-image person
./ondisk_pivot.md:3678:./reviews/deep_22.md:9425:1464:  1464	uncertainty-based learning for text-based person retrieval,” in Proc.
./ondisk_pivot.md:3717:./reviews/deep_22.md:9884:  1189	Similarly, compared with the second-ranked method APTM
./ondisk_pivot.md:3760:./reviews/deep_22.md:11267:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./ondisk_pivot.md:3764:./reviews/deep_22.md:11315:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./ondisk_pivot.md:3770:./reviews/deep_22.md:11344:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./ondisk_pivot.md:3774:./reviews/deep_22.md:11392:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./ondisk_pivot.md:3783:./OSAC_KILLSWITCH_DESIGN.md:34:**通过标准:** 过坍缩轨迹明确(ep后期 rank↓) + AP error ~ top-PC energy partial 显著 + ABTT 降 hubness 涨 raw + **D1 随机PC不如top-PC + D2 ABTT 在 k-reciprocal 之后仍有残余增益 + D3 控代理后仍在**。
./ondisk_pivot.md:3784:./OSAC_KILLSWITCH_DESIGN.md:35:→ 全过(尤其 D2 残余) = 过坍缩是 k-reciprocal 拿不到的真 training-side 失败变量 → 单训练 kill-switch(强 baseline 加 OSAC: spectral floor + Top-PC Dropout, 测试仍单 embedding; ep30/60 看 effective rank↑/top-PC hub correlation↓/raw +0.5 稳; 成功线 raw +0.8~1.0 且 k-reciprocal/camera 后仍 +0.3 残余)。
./ondisk_pivot.md:3787:./debate/clean/d_2.txt:1:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./ondisk_pivot.md:3788:./debate/clean/d_2.txt:3:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./ondisk_pivot.md:3790:./debate/clean/d_2.txt:8:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./ondisk_pivot.md:3791:./debate/clean/d_2.txt:16:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./ondisk_pivot.md:3809:./hubness_debate.sh:20:ROLES[2]="角色=**蓝队(降级 Hubness)**。为'别投训练, 降级转 r_2 备胎 Rank-Instability'辩护: D2 是铁证——k-reciprocal/camera **免费 test-time** 就在 mAP 上赢, 一个训练版要 beat camera-aware k-reciprocal 是高 bar 且小 headroom(Market 饱和)。用联网查: (a)hubness-aware training / anti-hub margin 历史上是否真能 beat 强 re-ranking(还是总被 k-reciprocal 这类盖过); (b)R1 赢 mAP 输是不是只是把 ranking 重排没真增加判别信息; (c)团队反复投不确定训练的教训。论证'诊断真≠方法能发', 该把诊断当一个 observation 写进别的稿/换 r_2。信心 1-10。"
./ondisk_pivot.md:3812:./debate/clean/d_1.txt:6:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./ondisk_pivot.md:3828:./debate/d_1.md:84:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./ondisk_pivot.md:3832:./debate/d_1.md:118:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./ondisk_pivot.md:3847:./analyses/deep_28.txt:49:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./ondisk_pivot.md:4005:./reassess/r_2.md:6440:结论：我只推荐 **1 个主线 + 1 个备胎**。不要再救 SMPL / 遮挡 / 航拍-地面；也不要把“相机偏差”当主线，DART³ 已经把 camera bias + distance-aware TTA 写得很近，CA-Jaccard 也占了 camera-aware neighbor/re-ranking 这块。([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2311.10605))
./ondisk_pivot.md:4007:./reassess/r_2.md:6467:- k-reciprocal re-ranking 已经用 reciprocal neighbor 做 test-time ranking，但它不是把 **gallery in-degree hubness** 定义成强 ReID 的失败变量，也不是训练端 anti-hub。([arxiv.org](https://arxiv.org/abs/1701.08398))
./ondisk_pivot.md:4012:./reassess/r_2.md:6501:结论：我只推荐 **1 个主线 + 1 个备胎**。不要再救 SMPL / 遮挡 / 航拍-地面；也不要把“相机偏差”当主线，DART³ 已经把 camera bias + distance-aware TTA 写得很近，CA-Jaccard 也占了 camera-aware neighbor/re-ranking 这块。([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2311.10605))
./ondisk_pivot.md:4014:./reassess/r_2.md:6528:- k-reciprocal re-ranking 已经用 reciprocal neighbor 做 test-time ranking，但它不是把 **gallery in-degree hubness** 定义成强 ReID 的失败变量，也不是训练端 anti-hub。([arxiv.org](https://arxiv.org/abs/1701.08398))
./ondisk_pivot.md:4266:./reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./ondisk_pivot.md:4269:./reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./ondisk_pivot.md:4402:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./ondisk_pivot.md:4412:./reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
./ondisk_pivot.md:4413:./reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
./ondisk_pivot.md:4421:./reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
./ondisk_pivot.md:4461:./reassess/r_3.md:5568:./reviews/deep_16.md:2471:/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|diffusion|inpainting|re-ranking|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./ondisk_pivot.md:4534:./reassess/r_3.md:7347:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./ondisk_pivot.md:4557:./reassess/r_3.md:7579:./validate/v_2.md:14462:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./ondisk_pivot.md:4558:./reassess/r_3.md:7583:./validate/v_2.md:14516:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./ondisk_pivot.md:4652:./reassess/r_3.md:7999:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./ondisk_pivot.md:4660:./reassess/r_3.md:8057:./pivot/p_2.md:700:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./ondisk_pivot.md:4719:./reassess/r_3.md:8298:./pivot/p_2.md:8427:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./ondisk_pivot.md:4738:./reassess/r_3.md:8684:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./ondisk_pivot.md:4740:./reassess/r_3.md:8690:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./ondisk_pivot.md:4741:./reassess/r_3.md:8692:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./ondisk_pivot.md:4743:./reassess/r_3.md:8696:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./ondisk_pivot.md:4799:./reassess/r_3.md:9264:validate/v_3.md:3005:../litreview2/analyses/ALL_ANALYSES.txt:1462:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./ondisk_pivot.md:4822:./reassess/r_3.md:9608:reviews/deep_3.md:6764:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./ondisk_pivot.md:4823:./reassess/r_3.md:9611:reviews/deep_3.md:6819:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./ondisk_pivot.md:4828:./reassess/r_3.md:9666:analyses/deep_3.txt:46:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./ondisk_pivot.md:4863:./reassess/r_3.md:10048:analyses/ALL_ANALYSES.txt:1462:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./ondisk_pivot.md:4888:./reassess/r_3.md:10821:reviews/deep_16.md:2471:/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|diffusion|inpainting|re-ranking|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./ondisk_pivot.md:5269:- 不撞 SeCap：SeCap 是 adaptive prompt + local feature refinement。SHSM 反而反对局部细节主导。来源：[SeCap arXiv](https://arxiv.org/abs/2503.06965)
./ondisk_pivot.md:5337:- 不撞 SeCap：SeCap 是 adaptive prompt + local feature refinement。SHSM 反而反对局部细节主导。来源：[SeCap arXiv](https://arxiv.org/abs/2503.06965)
./ondisk_pivot.md:5387:- 泄漏存在性：`cos(f_M, f_D)-cos(f_T, f_D)`、donor pid 进入 top-k 的比例、donor rank 是否显著提前。
./ondisk_pivot.md:5438:- 泄漏存在性：`cos(f_M, f_D)-cos(f_T, f_D)`、donor pid 进入 top-k 的比例、donor rank 是否显著提前。
./ondisk_pivot.md:5478: exited 1 in 0ms:
./ondisk_pivot.md:5644:./analyses/deep_18.txt:29:1. **触发观察**：正式版进一步把观察说得更工程化，单模态 ReID 和传统 VI-ReID 都没有覆盖“query 集和 gallery 集同时混合可见光与红外”的情况。作者强调 Rank-1 有时会因为同模态易匹配而变好，但 mAP 和 mINP 下降，说明整体检索质量变差。
./ondisk_pivot.md:5656:./analyses/deep_11.txt:47:4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。
./ondisk_pivot.md:5660:./analyses/deep_22.txt:11:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./ondisk_pivot.md:5710: exited 1 in 0ms:
./ondisk_pivot.md:5939:- 测试仍是单 text embedding 查 gallery image embedding，不用 MLLM、不用 re-ranking。
./ondisk_pivot.md:5978:- 测试仍是单 text embedding 查 gallery image embedding，不用 MLLM、不用 re-ranking。
./reassess2.sh:10:**meta-learning(关键)**: 团队frozen-feature+现有数据的隐藏变量候选, 反复被成熟ReID test-time工具(k-reciprocal/camera-aware re-ranking/hard-negative)碾压或证伪。ReID field成熟, 明显的retrieval-side/topology-side隐藏变量都被现成后处理占了。
./all_papers.txt:3:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./all_papers.txt:20:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./all_papers.txt:21:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./all_papers.txt:27:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./all_papers.txt:62:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./all_papers.txt:94:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./batches/bak:2:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_16.md:30:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_16.md:68: exited 1 in 0ms:
./reviews/deep_16.md:99:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_16.md:128:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_16.md:131:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_16.md:132:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_16.md:209:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_16.md:213:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_16.md:227:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_16.md:233:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_16.md:297: exited 1 in 0ms:
./reviews/deep_16.md:302: exited 1 in 0ms:
./reviews/deep_16.md:383:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_16.md:465:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_16.md:606:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_16.md:681:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_16.md:801:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_16.md:818:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_16.md:821:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_16.md:830:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_16.md:862:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_16.md:892:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_16.md:991:22:feedback to refine the sketch person retrieval ranking without
./reviews/deep_16.md:997:28:explicit sketch-text compositionality, thus achieving feedbackguided ranking refinement. Extensive experiments underscore
./reviews/deep_16.md:1014:75:person retrieval ranking without text training. Figure 1 (c)
./reviews/deep_16.md:1019:87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
./reviews/deep_16.md:1034:105:feedback with sketch queries for ranking refinement.
./reviews/deep_16.md:1046:131:through modality early-fusion (i.e., representing sketch as a
./reviews/deep_16.md:1076:293:refine the retrieval ranking, users’ text feedback tw can be integrated with the sketch’s pseudo-word token sw through context and feedback guidance.
./reviews/deep_16.md:1109:473:In order to refine the retrieval ranking results, users can provide any textual feedback tw for interactive communication.
./reviews/deep_16.md:1135:576:Fig. 3. A typical interactive person retrieval process of our InteractReID. Top-10 initial and refined ranking results on the Market-Sketch-1K dataset are
./reviews/deep_16.md:1140:616:early-fusion that achieves explicit sketch-text compositionality
./reviews/deep_16.md:1145:622:on Market-Sketch-1K for retrieval ranking refinement.
./reviews/deep_16.md:1312:603:Rank-1/mAP of this method achieves 66.5% Rank-1 accuracy and
./reviews/deep_16.md:1339:930:+1.0% Rank1 accuracy on DukeMTMC-ReID. We also experimented
./reviews/deep_16.md:1341:933:method achieves 80.4% Rank-1 accuracy and 61.1% mAP on MSMT17.
./reviews/deep_16.md:1352:1018:[12] C. Zhao, Z. Qu, X. Jiang, Y. Tu, X. Bai, Content-adaptive auto-occlusion network
./reviews/deep_16.md:1559:   249	Ranking
./reviews/deep_16.md:1565:   255	Ranking
./reviews/deep_16.md:1603:   293	refine the retrieval ranking, users’ text feedback tw can be integrated with the sketch’s pseudo-word token sw through context and feedback guidance.
./reviews/deep_16.md:1781:   471	generating the initial retrieval ranking results, which are then
./reviews/deep_16.md:1783:   473	In order to refine the retrieval ranking results, users can provide any textual feedback tw for interactive communication.
./reviews/deep_16.md:1790:   480	retrieval ranking by comparing the cosine similarity between
./reviews/deep_16.md:2374:   590	2) Batch Hard Triplet Loss: To improve the ranking performance of the network and to increase the differences between
./reviews/deep_16.md:2471:/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|diffusion|inpainting|re-ranking|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_16.md:2474:7:Re-Identification With Hybrid Re-Ranking
./reviews/deep_16.md:2484:26:known at test time, we introduce a Hybrid Re-Ranking (HRR)
./reviews/deep_16.md:2485:28:re-ranking by leveraging centroid-based identity anchors to
./reviews/deep_16.md:2486:29:refine k-reciprocal re-ranking, thus boosting retrieval precision
./reviews/deep_16.md:2508:103:Second, we propose a novel Hybrid Re-Ranking scheme
./reviews/deep_16.md:2509:117:re-ranking to centroid-curated candidate sets, thereby reducing
./reviews/deep_16.md:2510:119:re-ranking as a centroid-guided refinement over the manifold
./reviews/deep_16.md:2519:138:• We propose a Hybrid Re-Ranking (HRR) algorithm
./reviews/deep_16.md:2520:140:re-ranking process, improving retrieval precision and
./reviews/deep_16.md:2525:152:use adaptive position extraction with occlusion weighting, while
./reviews/deep_16.md:2536:179:for re-ranking [53] improve ranking quality by exploiting local
./reviews/deep_16.md:2546:209:with our latent Diffusion based De-Occluder (DDO) and Hybrid Re-Ranking (HRR) formulation.
./reviews/deep_16.md:2556:326:training. Novel algorithm to restrict re-ranking
./reviews/deep_16.md:2558:346:re-ranking
./reviews/deep_16.md:2563:360:reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reviews/deep_16.md:2565:364:problem inherent in standard re-ranking when gallery sets
./reviews/deep_16.md:2569:383:Re-Ranking (HRR) as follows: given a gallery set G with recover F̂clean .
./reviews/deep_16.md:2570:391:for k-reciprocal re-ranking, rather than relying solely on noisy
./reviews/deep_16.md:2599:772:C. Hybrid Re-Ranking (HRR)
./reviews/deep_16.md:2600:773:Standard re-ranking methods [51] compute pairwise distances
./reviews/deep_16.md:2601:776:We propose Hybrid Re-Ranking (HRR), combining robust
./reviews/deep_16.md:2602:778:By restricting re-ranking strictly to semantically relevant
./reviews/deep_16.md:2604:788:for re-ranking. Lines between query and centroids are labeled
./reviews/deep_16.md:2605:792:Hybrid Re-Ranking (HRR).
./reviews/deep_16.md:2606:795:for Jaccard-based re-ranking.
./reviews/deep_16.md:2607:799:compared to standard re-ranking approaches.
./reviews/deep_16.md:2608:812:Geometrically, this procedure restricts re-ranking to identity
./reviews/deep_16.md:2609:832:Standard re-ranking computes the Jaccard distance dJ (q, g)
./reviews/deep_16.md:2610:857:We construct Cq = {A, B, C} and restrict re-ranking to gallery
./reviews/deep_16.md:2611:900:Re-ranking can be interpreted as transductive inference over a
./reviews/deep_16.md:2612:902:In standard re-ranking, the graph includes all gallery samples,
./reviews/deep_16.md:2615:925:latent Diffusion-based De-Occluder (DDO) and Hybrid ReRanking (HRR) scheme in robustly improving accuracy under
./reviews/deep_16.md:2616:947:Stage 2: Local Re-ranking within Filtered Gallery:
./reviews/deep_16.md:2617:984:inter-identity noise, yielding a more robust re-ranking graph and
./reviews/deep_16.md:2618:986:more semantically meaningful subgraphs, improving re-ranking
./reviews/deep_16.md:2619:989:re-ranking in this manner.
./reviews/deep_16.md:2623:1106:prototype-guided re-ranking to enhance retrieval accuracy further confirm that HRR effectively leverages identity-level
./reviews/deep_16.md:2625:1113:we introduce our second contribution—Hybrid Re-Ranking benchmark ReID backbones, specifically TransReID [27] and
./reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reviews/deep_16.md:2827:136:(1) To the best of our knowledge, it is an early work on combining multi-granularity features based on Transformer structure in the
./reviews/deep_16.md:2949:1953:69.79% and 73.41% in Rank-1 accuracy. These significant improvements demonstrate the effectiveness of each module in our approach.
./reviews/deep_16.md:2950:1960:in Rank-1/mAP when our approach includes three-granularity features. In Table 10, similar improvements have been achieved on the
./reviews/deep_16.md:2956:1984:than using only global-level features in Rank-1 accuracy. This improvement is largely attributed to the fact that global- and patch-granularity
./reviews/deep_16.md:2969:/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | nl -ba | sed -n '348,990p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_16.md:2973:   350	preliminary distance matrix, which the HRR module subsequently reranks to produce the final distance matrix for retrieval.
./reviews/deep_16.md:2983:   360	reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reviews/deep_16.md:2987:   364	problem inherent in standard re-ranking when gallery sets
./reviews/deep_16.md:3006:   383	Re-Ranking (HRR) as follows: given a gallery set G with recover F̂clean .
./reviews/deep_16.md:3014:   391	for k-reciprocal re-ranking, rather than relying solely on noisy
./reviews/deep_16.md:3018:   395	ranking is applied to refine top matches. Together, DDO and fidelity.
./reviews/deep_16.md:3395:   772	C. Hybrid Re-Ranking (HRR)
./reviews/deep_16.md:3396:   773	Standard re-ranking methods [51] compute pairwise distances
./reviews/deep_16.md:3399:   776	We propose Hybrid Re-Ranking (HRR), combining robust
./reviews/deep_16.md:3401:   778	By restricting re-ranking strictly to semantically relevant
./reviews/deep_16.md:3411:   788	for re-ranking. Lines between query and centroids are labeled
./reviews/deep_16.md:3415:   792	Hybrid Re-Ranking (HRR).
./reviews/deep_16.md:3418:   795	for Jaccard-based re-ranking.
./reviews/deep_16.md:3422:   799	compared to standard re-ranking approaches.
./reviews/deep_16.md:3435:   812	Geometrically, this procedure restricts re-ranking to identity
./reviews/deep_16.md:3455:   832	Standard re-ranking computes the Jaccard distance dJ (q, g)
./reviews/deep_16.md:3480:   857	We construct Cq = {A, B, C} and restrict re-ranking to gallery
./reviews/deep_16.md:3523:   900	Re-ranking can be interpreted as transductive inference over a
./reviews/deep_16.md:3525:   902	In standard re-ranking, the graph includes all gallery samples,
./reviews/deep_16.md:3544:   921	drerank (q, g) = (1 − λr ) dorig (q, g) + λr · dJ (q, g)
./reviews/deep_16.md:3548:   925	latent Diffusion-based De-Occluder (DDO) and Hybrid ReRanking (HRR) scheme in robustly improving accuracy under
./reviews/deep_16.md:3567:   944	where Top-N denotes the operator that ranks all gallery
./reviews/deep_16.md:3570:   947	Stage 2: Local Re-ranking within Filtered Gallery:
./reviews/deep_16.md:3573:   950	to the top-selected identities. We then compute the re-ranked
./reviews/deep_16.md:3607:   984	inter-identity noise, yielding a more robust re-ranking graph and
./reviews/deep_16.md:3609:   986	more semantically meaningful subgraphs, improving re-ranking
./reviews/deep_16.md:3612:   989	re-ranking in this manner.
./reviews/deep_16.md:3841:   512	|Rank(Avc ) ≤ γ · ˇΩvc ˇ},
./reviews/deep_16.md:3850:   521	Ω = { f |Rank(A ) ≤ γ · Ω },
./reviews/deep_16.md:3859:   530	where Rank (·) is a rank function that refers to the ranking of
./reviews/deep_16.md:3862:   533	based on the ranking of feature scores Avc and Arl , while
./reviews/deep_16.md:3974:   645	noisy labels into the cost matrix, increasing the difficulty
./reviews/deep_16.md:4858:/bin/zsh -lc "pdftotext 'Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf' - | rg -n -i \"ablation|effectiveness|visual|typical interactive|pseudo-word|converting|feedback|ranking refinement|Table|Fig\\. 3|Does\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_16.md:4861:22:feedback to refine the sketch person retrieval ranking without
./reviews/deep_16.md:4865:28:explicit sketch-text compositionality, thus achieving feedbackguided ranking refinement. Extensive experiments underscore
./reviews/deep_16.md:4873:87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
./reviews/deep_16.md:4877:105:feedback with sketch queries for ranking refinement.
./reviews/deep_16.md:4894:293:refine the retrieval ranking, users’ text feedback tw can be integrated with the sketch’s pseudo-word token sw through context and feedback guidance.
./reviews/deep_16.md:4907:473:In order to refine the retrieval ranking results, users can provide any textual feedback tw for interactive communication.
./reviews/deep_16.md:4916:576:Fig. 3. A typical interactive person retrieval process of our InteractReID. Top-10 initial and refined ranking results on the Market-Sketch-1K dataset are
./reviews/deep_16.md:4922:622:on Market-Sketch-1K for retrieval ranking refinement.
./reviews/deep_16.md:4945:714:domain, InteractReID can achieve retrieval ranking refinement.
./reviews/deep_16.md:4947:719:retrieval ranking refinement. Concretely, Task-oriented Knowledge Adaptation is first conducted for CLIP’s pre-trained
./reviews/deep_16.md:5176:771:outperforms IEINet by 0.2% in Rank-1 accuracy using more
./reviews/deep_16.md:5185:811:different sites. The Rank-1 accuracy and mAP of IEINet reach
./reviews/deep_16.md:5217:889:the Rank-1 accuracy and mAP of “Baseline+IEI-S (w/o
./reviews/deep_16.md:5222:895:“Baseline+IEI-C (w/o SAP)” are inferior to “Baseline+IEIS” and “Baseline+IEI-C” in Rank-1 accuracy and mAP,
./reviews/deep_16.md:5236:925:IEI-S module outperforms the SA module by 0.7% Rank-1
./reviews/deep_16.md:5239:939:complexity of the SA module, the IEI-S module and the
./reviews/deep_16.md:5274:1048:Fig. 7. Performance comparison of distinct values of λ. (a) The Rank-1 accuracy and mAP on Market1501. (b) The Rank-1 accuracy and mAP on CUHK03Labeled.
./reviews/deep_16.md:5278:1055:of Rank-1 accuracy of IEINet on Market1501 and CUHK03Labeled datasets are similar. As the value of λ increases,
./reviews/deep_16.md:5328:   895	Rank-1
./reviews/deep_16.md:5363:   930	+1.0% Rank1 accuracy on DukeMTMC-ReID. We also experimented
./reviews/deep_16.md:5366:   933	method achieves 80.4% Rank-1 accuracy and 61.1% mAP on MSMT17.
./reviews/deep_16.md:5451:  1018	[12] C. Zhao, Z. Qu, X. Jiang, Y. Tu, X. Bai, Content-adaptive auto-occlusion network
./reviews/deep_16.md:5533:   678	demonstrate that the complexity of adaptation datasets is
./reviews/deep_16.md:5565:   710	structures. When the initial ranking results are returned, our
./reviews/deep_16.md:5569:   714	domain, InteractReID can achieve retrieval ranking refinement.
./reviews/deep_16.md:5631:   889	the Rank-1 accuracy and mAP of “Baseline+IEI-S (w/o
./reviews/deep_16.md:5637:   895	“Baseline+IEI-C (w/o SAP)” are inferior to “Baseline+IEIS” and “Baseline+IEI-C” in Rank-1 accuracy and mAP,
./reviews/deep_16.md:5667:   925	IEI-S module outperforms the SA module by 0.7% Rank-1
./reviews/deep_16.md:5668:   926	accuracy and 0.9% Rank-1 accuracy on the Market1501 and
./reviews/deep_16.md:5679:   937	4) Comparison of Computational Complexity: Taking the
./reviews/deep_16.md:5681:   939	complexity of the SA module, the IEI-S module and the
./reviews/deep_16.md:5690:   948	both have a value of 3. The complexity of the SA module is
./reviews/deep_16.md:5693:   951	a complexity of O(HW), which is linear to the number of
./reviews/deep_16.md:5708:   966	the computational complexity while remaining competitive
./reviews/deep_16.md:5764:  1022	0.4%+/2.0%+ improvement in Rank-1 accuracy/mAP on these
./reviews/deep_16.md:5774:  1032	Lrd )” on both datasets, especially in Rank-1 accuracy, which
./reviews/deep_16.md:5790:  1048	Fig. 7. Performance comparison of distinct values of λ. (a) The Rank-1 accuracy and mAP on Market1501. (b) The Rank-1 accuracy and mAP on CUHK03Labeled.
./reviews/deep_16.md:5797:  1055	of Rank-1 accuracy of IEINet on Market1501 and CUHK03Labeled datasets are similar. As the value of λ increases,
./reviews/deep_16.md:5841:/bin/zsh -lc "pdftotext 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"ablation|effectiveness|visual|visualization|DDO|HRR|inpainting|re-ranking|feature-space|clean|occluded|Table|Fig\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_16.md:5844:7:Re-Identification With Hybrid Re-Ranking
./reviews/deep_16.md:5848:26:known at test time, we introduce a Hybrid Re-Ranking (HRR)
./reviews/deep_16.md:5850:28:re-ranking by leveraging centroid-based identity anchors to
./reviews/deep_16.md:5851:29:refine k-reciprocal re-ranking, thus boosting retrieval precision
./reviews/deep_16.md:5870:103:Second, we propose a novel Hybrid Re-Ranking scheme
./reviews/deep_16.md:5873:117:re-ranking to centroid-curated candidate sets, thereby reducing
./reviews/deep_16.md:5874:119:re-ranking as a centroid-guided refinement over the manifold
./reviews/deep_16.md:5876:126:Average Precision (mAP) and Cumulative Matching Characteristic (CMC) Rank-1 scores, validating the effectiveness and
./reviews/deep_16.md:5880:138:• We propose a Hybrid Re-Ranking (HRR) algorithm
./reviews/deep_16.md:5881:140:re-ranking process, improving retrieval precision and
./reviews/deep_16.md:5890:179:for re-ranking [53] improve ranking quality by exploiting local
./reviews/deep_16.md:5896:209:with our latent Diffusion based De-Occluder (DDO) and Hybrid Re-Ranking (HRR) formulation.
./reviews/deep_16.md:5906:326:training. Novel algorithm to restrict re-ranking
./reviews/deep_16.md:5910:346:re-ranking
./reviews/deep_16.md:5913:350:preliminary distance matrix, which the HRR module subsequently reranks to produce the final distance matrix for retrieval.
./reviews/deep_16.md:5918:360:reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reviews/deep_16.md:5920:364:problem inherent in standard re-ranking when gallery sets
./reviews/deep_16.md:5927:383:Re-Ranking (HRR) as follows: given a gallery set G with recover F̂clean .
./reviews/deep_16.md:5929:391:for k-reciprocal re-ranking, rather than relying solely on noisy
./reviews/deep_16.md:5931:395:ranking is applied to refine top matches. Together, DDO and fidelity.
./reviews/deep_16.md:5993:772:C. Hybrid Re-Ranking (HRR)
./reviews/deep_16.md:5994:773:Standard re-ranking methods [51] compute pairwise distances
./reviews/deep_16.md:5995:776:We propose Hybrid Re-Ranking (HRR), combining robust
./reviews/deep_16.md:5996:778:By restricting re-ranking strictly to semantically relevant
./reviews/deep_16.md:5999:788:for re-ranking. Lines between query and centroids are labeled
./reviews/deep_16.md:6001:792:Hybrid Re-Ranking (HRR).
./reviews/deep_16.md:6002:795:for Jaccard-based re-ranking.
./reviews/deep_16.md:6005:799:compared to standard re-ranking approaches.
./reviews/deep_16.md:6007:812:Geometrically, this procedure restricts re-ranking to identity
./reviews/deep_16.md:6008:832:Standard re-ranking computes the Jaccard distance dJ (q, g)
./reviews/deep_16.md:6010:857:We construct Cq = {A, B, C} and restrict re-ranking to gallery
./reviews/deep_16.md:6012:900:Re-ranking can be interpreted as transductive inference over a
./reviews/deep_16.md:6013:902:In standard re-ranking, the graph includes all gallery samples,
./reviews/deep_16.md:6016:925:latent Diffusion-based De-Occluder (DDO) and Hybrid ReRanking (HRR) scheme in robustly improving accuracy under
./reviews/deep_16.md:6017:947:Stage 2: Local Re-ranking within Filtered Gallery:
./reviews/deep_16.md:6019:984:inter-identity noise, yielding a more robust re-ranking graph and
./reviews/deep_16.md:6020:986:more semantically meaningful subgraphs, improving re-ranking
./reviews/deep_16.md:6021:989:re-ranking in this manner.
./reviews/deep_16.md:6029:1106:prototype-guided re-ranking to enhance retrieval accuracy further confirm that HRR effectively leverages identity-level
./reviews/deep_16.md:6032:1113:we introduce our second contribution—Hybrid Re-Ranking benchmark ReID backbones, specifically TransReID [27] and
./reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reviews/deep_16.md:6040:1130:(CMC) Rank-1 accuracy and mean Average Precision (mAP). mitigates domain-specific camera biases, allowing DDO to
./reviews/deep_16.md:6041:1132:the ranking quality and overall retrieval effectiveness of our holistic analysis of the computational overhead with subsequent
./reviews/deep_16.md:6081:2381:accuracy by effectively guiding the re-ranking process, thereby
./reviews/deep_16.md:6085:2387:the structure of the centroid-based hybrid re-ranking approach.
./reviews/deep_16.md:6086:2391:on re-ranking within this subset GN . A larger gallery implies
./reviews/deep_16.md:6089:2398:relevant positive samples before re-ranking.
./reviews/deep_16.md:6090:2399:with DDO, (d) the Rank-1 retrieval without DDO, and (e)
./reviews/deep_16.md:6091:2401:the Rank-1 retrieval with DDO. Green and red boxes indicate
./reviews/deep_16.md:6093:2412:contrast, DDO-enhanced features (Fclean vs. F ) maintain ranking performance. Figure 7 (c) shows that larger galleries
./reviews/deep_16.md:6096:2420:cases, the ViT-extracted features are passed through DDO filtering and local re-ranking under occlusion. An actionable
./reviews/deep_16.md:6099:2424:gallery centroids during re-ranking. As shown in Table VI, E. Limitations
./reviews/deep_16.md:6105:2440:re-ranking across five ReID datasets. A linear trend is fitted, and Spearman correlation is reported to quantify the dependency.
./reviews/deep_16.md:6113:2492:TABLE VIII: Cross-dataset evaluation. Each cell shows mAP and Rank-1 (%) for models trained on the specified dataset and
./reviews/deep_16.md:6130:2678:92.3% to 90.1%) and Table VIII on Partial-ReID (Rank-1
./reviews/deep_16.md:6142:2757:Fig. 8: Retrieval Performance (mAP) vs top-N centroid filtering novel Hybrid Re-Ranking (HRR) scheme that successfully
./reviews/deep_16.md:6143:2758:for different datasets during the hybrid re-ranking.
./reviews/deep_16.md:6145:2762:and the re-ranking strategy. Additionally, the proposed diffusionLet each gallery feature gi for identity c be expressed as: based feature inpainting framework extends naturally to other
./reviews/deep_16.md:6153:2837:where the Rank-1 retrieval drops from 58.3% (without HRR) to
./reviews/deep_16.md:6166:3013:[51] Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking person reidentification with k-reciprocal encoding,” in Proceedings of the IEEE
./reviews/deep_16.md:6167:3027:for re-ranking person re-identification,” IEEE Transactions on Image
./reviews/deep_16.md:6214:136:(1) To the best of our knowledge, it is an early work on combining multi-granularity features based on Transformer structure in the
./reviews/deep_16.md:6314:1953:69.79% and 73.41% in Rank-1 accuracy. These significant improvements demonstrate the effectiveness of each module in our approach.
./reviews/deep_16.md:6316:1960:in Rank-1/mAP when our approach includes three-granularity features. In Table 10, similar improvements have been achieved on the
./reviews/deep_16.md:6326:1984:than using only global-level features in Rank-1 accuracy. This improvement is largely attributed to the fact that global- and patch-granularity
./reviews/deep_16.md:6353:2527:our approach shows notable leads of 3.12% and 6.97% in Rank-1
./reviews/deep_16.md:6363:3366:dataset scale and computational complexity, our method uses a multigranularity feature extraction algorithm based on the original dataset
./reviews/deep_16.md:6374:3394:As shown in Fig. 7, we visualize the Rank-10 search results for five
./reviews/deep_16.md:6480:645:noisy labels into the cost matrix, increasing the difficulty
./reviews/deep_16.md:6532:1103:extractor. As shown in Table IV, compared to LADCL , the performance is improved to 3% in Rank-1 and 1% in mAP on the
./reviews/deep_16.md:6551:1184:presented in Fig. 5. Additionally, the ranking list for the
./reviews/deep_16.md:6561:1221:3) Ranking Visualization: We conduct the top-5 retrieval
./reviews/deep_16.md:6568:1419:[52] K. Okamoto, W. Chen, and X.-Y. Li, “Ranking of closeness centrality
./reviews/deep_16.md:6585:/bin/zsh -lc "pdftotext 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | nl -ba | sed -n '1138,1162p;2188,2240p;2380,2441p;2658,2684p;2735,2739p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_16.md:6666:  2381	accuracy by effectively guiding the re-ranking process, thereby
./reviews/deep_16.md:6672:  2387	the structure of the centroid-based hybrid re-ranking approach.
./reviews/deep_16.md:6675:  2390	most similar to a query feature q. The final ranking is based
./reviews/deep_16.md:6676:  2391	on re-ranking within this subset GN . A larger gallery implies
./reviews/deep_16.md:6681:  2396	ranking. Increasing N improves the likelihood of including
./reviews/deep_16.md:6683:  2398	relevant positive samples before re-ranking.
./reviews/deep_16.md:6684:  2399	with DDO, (d) the Rank-1 retrieval without DDO, and (e)
./reviews/deep_16.md:6686:  2401	the Rank-1 retrieval with DDO. Green and red boxes indicate
./reviews/deep_16.md:6691:  2406	then approximated as: E[mAP] ∝ E[rank-quality(PN )] · rN .
./reviews/deep_16.md:6697:  2412	contrast, DDO-enhanced features (Fclean vs. F ) maintain ranking performance. Figure 7 (c) shows that larger galleries
./reviews/deep_16.md:6705:  2420	cases, the ViT-extracted features are passed through DDO filtering and local re-ranking under occlusion. An actionable
./reviews/deep_16.md:6709:  2424	gallery centroids during re-ranking. As shown in Table VI, E. Limitations
./reviews/deep_16.md:6725:  2440	re-ranking across five ReID datasets. A linear trend is fitted, and Spearman correlation is reported to quantify the dependency.
./reviews/deep_16.md:6747:  2678	92.3% to 90.1%) and Table VIII on Partial-ReID (Rank-1
./reviews/deep_16.md:6917:  1506	than baseline in Rank-1 accuracy on two modes, respectively. This
./reviews/deep_16.md:7364:  1953	69.79% and 73.41% in Rank-1 accuracy. These significant improvements demonstrate the effectiveness of each module in our approach.
./reviews/deep_16.md:7371:  1960	in Rank-1/mAP when our approach includes three-granularity features. In Table 10, similar improvements have been achieved on the
./reviews/deep_16.md:7395:  1984	than using only global-level features in Rank-1 accuracy. This improvement is largely attributed to the fact that global- and patch-granularity
./reviews/deep_16.md:7414:  3394	As shown in Fig. 7, we visualize the Rank-10 search results for five
./reviews/deep_16.md:7467:  1089	As a result, the method achieves 36.18% and 45.00% in rank-1
./reviews/deep_16.md:7473:  1095	performance of 6% and 7% in Rank-1 on SYSU-MM01 and
./reviews/deep_16.md:7481:  1103	extractor. As shown in Table IV, compared to LADCL , the performance is improved to 3% in Rank-1 and 1% in mAP on the
./reviews/deep_16.md:7491:  1113	two modalities, improving performance by 3% in Rank-1 and
./reviews/deep_16.md:7501:  1123	SYSU-MM01, demonstrating an increase of 17% in Rank-1
./reviews/deep_16.md:7562:  1184	presented in Fig. 5. Additionally, the ranking list for the
./reviews/deep_16.md:7599:  1221	3) Ranking Visualization: We conduct the top-5 retrieval
./reviews/deep_16.md:7600:  1222	ranking of some example queries on SYSU-MM01 with the
./reviews/deep_16.md:7645:4. 证据闭环：消融非常贴问题定义。CCE 在 SYSU-MM01 和 RegDB 上 Rank-1 分别约 +6 和 +7；BCLM 约 +3 Rank-1 和 +3 mAP；GPRA 加 LDRHL 在 SYSU-MM01 上带来约 +17 Rank-1 和 +14 mAP。t-SNE 也不是泛泛展示聚类更好，而是专门显示已有方法虽然模态靠近了，但噪声标签还在，而 HCCL 缓解了这个问题。
./reviews/deep_16.md:7653:4. 证据闭环：MGFE 在 SYSU-MM01 两种模式下 Rank-1 分别约 +6.26 和 +4.47；MGFT 和 CNC 继续提升；三粒度组合明显优于单粒度或两粒度。热图也服务于视角，global 关注大区域，patch 关注轮廓和局部，pixel 捕捉姿态和手臂动作，组合后覆盖更完整语义。
./reviews/deep_16.md:7698:4. 证据闭环：消融非常贴问题定义。CCE 在 SYSU-MM01 和 RegDB 上 Rank-1 分别约 +6 和 +7；BCLM 约 +3 Rank-1 和 +3 mAP；GPRA 加 LDRHL 在 SYSU-MM01 上带来约 +17 Rank-1 和 +14 mAP。t-SNE 也不是泛泛展示聚类更好，而是专门显示已有方法虽然模态靠近了，但噪声标签还在，而 HCCL 缓解了这个问题。
./reviews/deep_16.md:7706:4. 证据闭环：MGFE 在 SYSU-MM01 两种模式下 Rank-1 分别约 +6.26 和 +4.47；MGFT 和 CNC 继续提升；三粒度组合明显优于单粒度或两粒度。热图也服务于视角，global 关注大区域，patch 关注轮廓和局部，pixel 捕捉姿态和手臂动作，组合后覆盖更完整语义。
./reviews/deep_6.md:48:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_6.md:81:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_6.md:89:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_6.md:132:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_6.md:138:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_6.md:152:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_6.md:299:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_6.md:311: exited 1 in 0ms:
./reviews/deep_6.md:361:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_6.md:489:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_6.md:518:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_6.md:521:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_6.md:522:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_6.md:599:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_6.md:603:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_6.md:617:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_6.md:623:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_6.md:775:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_6.md:850:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_6.md:1022:    33	previous methods, achieving a 6% increase in Rank-1 and a 4.4%
./reviews/deep_6.md:1114:   375	in Rank-1 and mAP, respectively. And it also outperforms the
./reviews/deep_6.md:1115:   376	previous state-of-the-art method by 3.3% and 1.9% on Rank-1
./reviews/deep_6.md:1116:   378	method improves the Rank-1 and mAP by 1.5% and 0.2%,
./reviews/deep_6.md:1117:   379	respectively, compared to the baseline method in the clothchanging setting. In the standard setting, Rank-1 and mAP
./reviews/deep_6.md:1125:   390	rank-1 and mAP increasing by 0.6% and 0.7%, respectively,
./reviews/deep_6.md:1159:   472	and X. Xie, “Learning resolution-adaptive representations for crossresolution person re-identification,” IEEE Transactions on Image Processing, 2023.
./reviews/deep_6.md:1183:   577	[34] F. Liu, M. Ye, and B. Du, “Dual level adaptive weighting for clothchanging person re-identification,” IEEE Transactions on Image Processing, 2023.
./reviews/deep_6.md:1227:    29	enhancement module with adaptive graph construction and
./reviews/deep_6.md:1230:    35	3.5% improvement in Rank-1 accuracy and 2.8% in mAP
./reviews/deep_6.md:1274:   181	reducing camera background bias through optimized clustering and adaptive filtering.
./reviews/deep_6.md:1310:   262	These loss functions are jointly optimized with adaptive weight
./reviews/deep_6.md:1316:   276	position) and adaptive graph construction (right, where node
./reviews/deep_6.md:1330:   314	a global feature memory bank; right demonstrates adaptive graph construction based on local density. Throughout both stages, clustering evolves from
./reviews/deep_6.md:1350:   396	an adaptive k-nearest neighbor approach with label-aware
./reviews/deep_6.md:1377:   635	adaptive feature fusion network with a residual structure to
./reviews/deep_6.md:1402:   726	Based on the memory bank, we propose an adaptive
./reviews/deep_6.md:1404:   736	projection matrix, and S is the similarity matrix. The adaptive
./reviews/deep_6.md:1407:   751	neighbors, where ki is adaptively determined by:
./reviews/deep_6.md:1409:   759	0.5 (regulation parameter). This adaptive mechanism reduces
./reviews/deep_6.md:1411:   762	As shown in Figure 2(c), our adaptive algorithm adjusts the
./reviews/deep_6.md:1420:   823	innovation lies in the adaptive post-processing strategy that
./reviews/deep_6.md:1422:   843	labels, and c j is the camera ID. We propose an adaptive
./reviews/deep_6.md:1424:   865	through adaptive probability decay, allowing the model to
./reviews/deep_6.md:1440:   904	CMAG achieves 73.3% Rank-1 and 77.7% mAP, substantially outperforming the latest methods TokenMatcher (69.0%,
./reviews/deep_6.md:1656:  2097	Table 2. Note that none of the methods in this experiment use reranking. Our proposed MCST outperforms existing SOTA methods,
./reviews/deep_6.md:1658:  2103	improving the average recognition rate. As shown in Table 3, MCST outperforms SOTA methods on the SYSU-MM01 dataset. With re-ranking,
./reviews/deep_6.md:1659:  2106	approaches. Notably, it improves Rank-1 accuracy by 9.1% and mAP
./reviews/deep_6.md:1725:    97	Noise-aware Memory Updating (NMU) strategy adaptively re-weights
./reviews/deep_6.md:1750:   168	label purification by adaptively adjusting the pseudo-label distribution.
./reviews/deep_6.md:1816:   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
./reviews/deep_6.md:1822:   658	epochs, linearly increasing the learning rate from 3.5×10−6 to 3.5×10−4 .
./reviews/deep_6.md:1882:  1296	94.79% and 95.33% Rank-1 for VIS-to-IR and IR-to-VIS modes, respectively, averaging over 1% improvement across all metrics. Furthermore,
./reviews/deep_6.md:2236:  1173	Notably, Vehicle-Diff achieves a 0.6% increase in Rank1 accuracy over VehicleX, whose Rank-1 is already high
./reviews/deep_6.md:2240:  1177	97.68% Rank-1 and 83.79% mAP, which surpasses CLIPReID [64] of 97.40% Rank-1 and 83.30% mAP. Similarly, for
./reviews/deep_6.md:2242:  1179	VehicleX by 4.17% on Rank-1 and 4.26% on Rank-5 (see
./reviews/deep_6.md:2254:  1265	method with both our baseline and VehicleX. The ranking
./reviews/deep_6.md:2356:  1632	and R. Chellappa, “A dual-path model with adaptive attention for vehicle re-identification,” in Proceedings of the IEEE/CVF International
./reviews/deep_6.md:2381:   199	leveraged sketch abstraction levels combined with diﬀerentiable ranking losses to improve cross-style alignment stability. The latter introduce
./reviews/deep_6.md:2421:   285	Dynamic Lighting Modiﬁer. The core of the dynamic light modiﬁer is to adaptively modify the lighting based on the average brightness
./reviews/deep_6.md:2423:   287	to dark or overexposed environments, which can adaptively adjust the
./reviews/deep_6.md:2446:   373	To ensure attribute fairness between diﬀerent modalities, we propose a attribute compensation and reﬁned module (ACRM) to adaptively
./reviews/deep_6.md:2462:   431	a normalization term b. Speciﬁcally, g serves as a channel-wise gating mechanism, adaptively regulating the contribution of compensation features so that channels correlated with the missing attributes
./reviews/deep_6.md:2463:   432	(e.g., color or texture) are enhanced while irrelevant channels are suppressed. Meanwhile, b acts as a residual normalization term: the constant 1 guarantees preservation of original sketch features, while the additional adaptive bias ensures numerical stability and balanced feature
./reviews/deep_6.md:2467:   459	that can introduce instability near boundaries. Consequently, YUV provides a more robust and linear basis for integrating contour and color,
./reviews/deep_6.md:2480:   518	This optimization process can adaptively adjust the importance of
./reviews/deep_6.md:2520:   702	the complete ACRM (AC+RF) achieves the best performance. Specifically, on the Tri-RSTPReid and Tri-PKU-Sketch datasets, Rank-1 accuracies reach 58.72 % and 73.00 % for the Text-RGB task, 65.50 %
./reviews/deep_6.md:2528:   711	As shown in Table 3, introducing the cross-modal structure regularization loss 𝑐𝑚 in the Sketch-RGB task, the Rank-1 accuracy increases
./reviews/deep_6.md:2531:   715	Rank-1 accuracy of 78.00 %, 93.50 % under the “Text + Sketch" query
./reviews/deep_6.md:2546:   749	3.10 % in Rank-1 and Rank-5 accuracy, respectively. Likewise, Table 4
./reviews/deep_6.md:2550:   759	matching, such as diverse style modeling, uncertainty-aware feature
./reviews/deep_6.md:2552:   762	Therefore, we plan to explore more adaptive and task-aware feature
./reviews/deep_6.md:2563:   777	queries, with Rank-1 accuracy gains of 1.54 % and 3.80 % on the TriRSTPReid and Tri-PKU-Sketch datasets, respectively. Under sketchbased queries, the auxiliary color sketch modality achieves Rank-1 accuracies of 70.24 % and 91.00 % on Tri-RSTPReid and Tri-PKU-Sketch,
./reviews/deep_6.md:2565:   779	counterpart by 0.95 % and 3.30 % in Rank-1 accuracy on two datasets,
./reviews/deep_6.md:2567:   789	Ablation study about each component on three multi-modal datasets. Rank (R) at k accuracy (%), mAP (%), and mINP (%) are reported.
./reviews/deep_6.md:3205:   673	environments, lacking real-world complexities such as occlusion, clothing changes, and other appearance variations. To address these challenges, we have developed a new V-I ReID dataset, CMG-P, which incorporates more complex scenarios. As illustrated in Table 1, CMG-P
./reviews/deep_6.md:3665:   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
./reviews/deep_6.md:3709:   658	epochs, linearly increasing the learning rate from 3.5×10−6 to 3.5×10−4 .
./reviews/deep_6.md:3783:   237	the complexity of unsupervised cross-modal person reidentification. Stage 1 (50 epochs) establishes robust feature
./reviews/deep_6.md:3808:   262	These loss functions are jointly optimized with adaptive weight
./reviews/deep_6.md:3822:   276	position) and adaptive graph construction (right, where node
./reviews/deep_6.md:3860:   314	a global feature memory bank; right demonstrates adaptive graph construction based on local density. Throughout both stages, clustering evolves from
./reviews/deep_6.md:3942:   396	an adaptive k-nearest neighbor approach with label-aware
./reviews/deep_6.md:4146:   600	through asymmetric feature projection, reducing computational complexity while enhancing modality interaction
./reviews/deep_6.md:4181:   635	adaptive feature fusion network with a residual structure to
./reviews/deep_6.md:4272:   726	Based on the memory bank, we propose an adaptive
./reviews/deep_6.md:4282:   736	projection matrix, and S is the similarity matrix. The adaptive
./reviews/deep_6.md:4297:   751	neighbors, where ki is adaptively determined by:
./reviews/deep_6.md:4305:   759	0.5 (regulation parameter). This adaptive mechanism reduces
./reviews/deep_6.md:4308:   762	As shown in Figure 2(c), our adaptive algorithm adjusts the
./reviews/deep_6.md:4369:   823	innovation lies in the adaptive post-processing strategy that
./reviews/deep_6.md:4389:   843	labels, and c j is the camera ID. We propose an adaptive
./reviews/deep_6.md:4411:   865	through adaptive probability decay, allowing the model to
./reviews/deep_6.md:4918:   373	To ensure attribute fairness between diﬀerent modalities, we propose a attribute compensation and reﬁned module (ACRM) to adaptively
./reviews/deep_6.md:4976:   431	a normalization term b. Speciﬁcally, g serves as a channel-wise gating mechanism, adaptively regulating the contribution of compensation features so that channels correlated with the missing attributes
./reviews/deep_6.md:4977:   432	(e.g., color or texture) are enhanced while irrelevant channels are suppressed. Meanwhile, b acts as a residual normalization term: the constant 1 guarantees preservation of original sketch features, while the additional adaptive bias ensures numerical stability and balanced feature
./reviews/deep_6.md:5004:   459	that can introduce instability near boundaries. Consequently, YUV provides a more robust and linear basis for integrating contour and color,
./reviews/deep_6.md:5063:   518	This optimization process can adaptively adjust the importance of
./reviews/deep_6.md:5931:  1733	Rank-1, 2.69% in mAP, and 2.85% in mINP compared to the baseline.
./reviews/deep_6.md:6026:  1828	drop (e.g., −1.06% Rank-1 and −1.12% mAP). This empirical evidence
./reviews/deep_6.md:6124:  1926	choice correlates with dataset difficulty. The challenging LLCM dataset
./reviews/deep_6.md:6141:  1943	cascade dependency between CPC and OTPM, and the coarse-to-fine
./reviews/deep_6.md:6187:    25	to 58.36% Rank-1 and 52.93%mAP in all-search mode, indicating that the ViT architecture effectively captures long-range
./reviews/deep_6.md:6189:    28	cross-modal attention mechanism (Row 4) improved performance to 62.54% Rank-1 and 56.98%mAP (all-search). When
./reviews/deep_6.md:6198:    61	improved to 63.18% Rank-1 and 59.91%mAP, demonstrating synergistic effects between components. This validates
./reviews/deep_6.md:6201:    66	64.77% Rank-1 and 59.18%mAP (all-search), demonstrating the unique value of circular topological structure in
./reviews/deep_6.md:6215:   100	best performance: 68.52% Rank-1 and 64.0%mAP, demonstrating the important role of global feature memory banks and
./reviews/deep_6.md:6250:   375	in Rank-1 and mAP, respectively. And it also outperforms the
./reviews/deep_6.md:6251:   376	previous state-of-the-art method by 3.3% and 1.9% on Rank-1
./reviews/deep_6.md:6253:   378	method improves the Rank-1 and mAP by 1.5% and 0.2%,
./reviews/deep_6.md:6254:   379	respectively, compared to the baseline method in the clothchanging setting. In the standard setting, Rank-1 and mAP
./reviews/deep_6.md:6263:   388	model achieves 55.2% on Rank-1 and 55.8% on mAP. When
./reviews/deep_6.md:6265:   390	rank-1 and mAP increasing by 0.6% and 0.7%, respectively,
./reviews/deep_6.md:6276:   401	Rank-1 and mAP. This represents a significant improvement,
./reviews/deep_6.md:6282:   407	their retrieval rank results. The results in Fig. 3 indicate
./reviews/deep_6.md:6317:  1202	Rank-1
./reviews/deep_6.md:6319:  1204	Rank-2
./reviews/deep_6.md:6321:  1206	Rank-3
./reviews/deep_6.md:6323:  1208	Rank-4
./reviews/deep_6.md:6325:  1210	Rank-5
./reviews/deep_6.md:6380:  1265	method with both our baseline and VehicleX. The ranking
./reviews/deep_6.md:6413:  1298	Rank-1
./reviews/deep_6.md:6489:  1374	Rank-1
./reviews/deep_6.md:6493:  1378	Rank-5
./reviews/deep_6.md:6506:  1391	top-5 of the ranking list, surpassing the same model trained
./reviews/deep_6.md:6532:  1285	4.4. On the model complexity
./reviews/deep_6.md:6602:4. 证据闭环：消融显示 DCAM、CATS、GSMEM、CARC 逐步涨，完整模型在 SYSU-MM01 到 68.52 Rank-1 和 64.0 mAP。更贴近重定义的是三类图：超参 p 证明闭环长度有效，检索可视化说明 partial modules 仍处理不好大姿态差，t-SNE 和距离分布显示跨模态聚合和类间分离逐步改善。但如果严审，“拓扑”本身是否比普通图传播更关键，还需要更强的 cycle-only 对照。
./reviews/deep_6.md:6681:4. 证据闭环：消融显示 DCAM、CATS、GSMEM、CARC 逐步涨，完整模型在 SYSU-MM01 到 68.52 Rank-1 和 64.0 mAP。更贴近重定义的是三类图：超参 p 证明闭环长度有效，检索可视化说明 partial modules 仍处理不好大姿态差，t-SNE 和距离分布显示跨模态聚合和类间分离逐步改善。但如果严审，“拓扑”本身是否比普通图传播更关键，还需要更强的 cycle-only 对照。
./reviews/deep_26.md:45: exited 1 in 0ms:
./reviews/deep_26.md:56:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_26.md:73:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_26.md:76:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_26.md:85:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_26.md:117:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_26.md:147:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_26.md:292:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_26.md:422:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_26.md:451:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_26.md:454:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_26.md:455:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_26.md:532:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_26.md:536:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_26.md:550:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_26.md:556:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_26.md:905:the loss weight by task difficulty, training speed, and priority [3, 10, 18, 28]. Our HDW method groups tasks based
./reviews/deep_26.md:1168:a comprehensive assessment of model performance and explore anytime retrieval. For each identity, we selected three
./reviews/deep_26.md:1453:tasks have different levels of difficulty and learning curves.
./reviews/deep_26.md:1485:Rank-1
./reviews/deep_26.md:1489:Rank-1
./reviews/deep_26.md:1493:Rank-1
./reviews/deep_26.md:1497:Rank-1
./reviews/deep_26.md:1501:Rank-1
./reviews/deep_26.md:1505:Rank-1
./reviews/deep_26.md:1587:Table 1. Cross-domain generalization experiments in different datasets. Rank-1 accuracy (%) is reported.
./reviews/deep_26.md:1611:Rank-1
./reviews/deep_26.md:1759:Evaluation Protocols. The Rank-k matching accuracy
./reviews/deep_26.md:1918:Table 3. Ablation study on AT-USTC. Rank-1 (R1) and mAP accuracy (%) are reported.
./reviews/deep_26.md:2045:and LTCC datasets. Rank-1 (R1) and mAP accuracy (%) are reported.
./reviews/deep_26.md:2063:model trained on MSMT17/DeepChange/LLCM by an average of 10.90% / 13.69% / 14.36% in Rank-1 accuracy and
./reviews/deep_26.md:2189:roughly divided into the following three categories: imagelevel early fusion [14], [15], feature-level late fusion [16],
./reviews/deep_26.md:2276:IMMI module forms a mask that adaptively reflects intermodal
./reviews/deep_26.md:2373:enhancement module for the image-level fusion of different modalities. However, they simply concatenate or linearly
./reviews/deep_26.md:2394:introduced by He et al. [17], is a low-rank fusion network
./reviews/deep_26.md:2412:enabling adaptive interactions between modalities. IMFF utilizes the location attention mechanism to achieve contextual
./reviews/deep_26.md:2424:ranking list, where the same pedestrian should be at the front
./reviews/deep_26.md:2576:networks. In addition, the similarity-based adaptive interactive fusion enables the fused features to better maintain
./reviews/deep_26.md:2784:(CMCs) at RankR (R = 1, 5, and 10) (in %) and mean average
./reviews/deep_26.md:2838:exhibiting an improvement of at least 1.32% over other stateof-the-art methods and achieves a Rank1 accuracy of 54.43%,
./reviews/deep_26.md:2841:mAP of 67.38% and a Rank1 accuracy of 86.05%, showing
./reviews/deep_26.md:3322:SM is influenced by self-distillation [29], where valuable information is distilled at early epochs and then used to enhance the training
./reviews/deep_26.md:3476:As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
./reviews/deep_26.md:3502:To clearly outline our experiments, we first describe the experimental
./reviews/deep_26.md:3636:Rank-1
./reviews/deep_26.md:3638:Rank-10
./reviews/deep_26.md:3640:Rank-20
./reviews/deep_26.md:3644:Rank-1
./reviews/deep_26.md:3646:Rank-10
./reviews/deep_26.md:3648:Rank-20
./reviews/deep_26.md:3989:impressive results with 91.1% Rank-1 accuracy and 81.7% mAP in the
./reviews/deep_26.md:3990:Visible2thermal mode, and 89.9% Rank-1 accuracy and 80.5% mAP
./reviews/deep_26.md:3992:LCNL [50], exceeding its Rank-1 accuracy by 5.5% and mAP by 3.0%
./reviews/deep_26.md:3993:in the Visible2thermal mode, and surpassing it by 5.9% in Rank-1 and
./reviews/deep_26.md:3998:model with 50% labeled data achieves 86.8% Rank-1 and 77.3% mAP
./reviews/deep_26.md:3999:in the Visible2thermal mode, and 82.1% Rank-1 and 75.2% mAP in the
./reviews/deep_26.md:4006:Our method achieves impressive results, recording a Rank-1 accuracy of 76.6% and an mAP of 73.0%. Thanks to the proposed SI, SM,
./reviews/deep_26.md:4009:21.8% in Rank-1 accuracy and 20.0% in mAP without adding extra network parameters. Furthermore, our method improved Rank-1 by 1.9%
./reviews/deep_26.md:4015:gains impressive results with Rank-1 accuracy of 58.2%, 68.5%, and
./reviews/deep_26.md:4019:it by 5.4% in Rank-1 accuracy with 50% labeled samples. Furthermore,
./reviews/deep_26.md:4030:65.1% Rank-1 accuracy and 62.3% mAP under full-supervised setting
./reviews/deep_26.md:4031:and achieves 43.0% Rank-1 accuracy and 37.8% mAP under the weaksupervised setting. Next, we present the results after integrating the
./reviews/deep_26.md:4034:improves to 66.3% Rank-1 accuracy and 62.9% mAP under the fullsupervised setting. Besides, the performance reaches 48.3% Rank-1
./reviews/deep_26.md:4036:a significant enhancement, achieving 71.9% Rank-1 and 68.2% mAP
./reviews/deep_26.md:4054:Rank-1
./reviews/deep_26.md:4058:Rank-1
./reviews/deep_26.md:4207:Comparisons with six advanced methods under the weak-supervised setting, and all methods are evaluated using Rank-1 (%) and mAP (%). All results
./reviews/deep_26.md:4231:Rank-1
./reviews/deep_26.md:4235:Rank-1
./reviews/deep_26.md:4239:Rank-1
./reviews/deep_26.md:4243:Rank-1
./reviews/deep_26.md:4541:Rank-1
./reviews/deep_26.md:4543:Rank-10
./reviews/deep_26.md:4545:Rank-20
./reviews/deep_26.md:4549:Rank-1
./reviews/deep_26.md:4551:Rank-10
./reviews/deep_26.md:4553:Rank-20
./reviews/deep_26.md:4635:reaches 53.5% Rank-1 and 48.6% mAP. These improvements highlight
./reviews/deep_26.md:4638:Table 4), our TSKD gains 74.7% Rank-1 and 71.3% mAP under the fullsupervised setting, and achieves 51.4% Rank-1 and 47.2% mAP under
./reviews/deep_26.md:4644:achieves 74.9% Rank-1 accuracy and 72.1% mAP under full-supervised
./reviews/deep_26.md:4651:Fig. 4. Rank-1 and mAP with different mimic learning schemes on the SYSU-MM01 dataset.
./reviews/deep_26.md:4653:setting and achieves 56.0% Rank-1 accuracy and 51.7% mAP under
./reviews/deep_26.md:4656:TSKD achieves 76.6% Rank-1 and 73.0% mAP under the full-supervised
./reviews/deep_26.md:4657:setting, surpassing the baseline by substantial margins of 11.5% Rank-1
./reviews/deep_26.md:4659:58.2% Rank-1 and 56.3% mAP, surpassing the baseline by 15.2% in
./reviews/deep_26.md:4660:Rank-1 and 18.5% in mAP.
./reviews/deep_26.md:4662:Table 4) leads to a clear drop in performance compared to the configuration with all components (Index 6 in Table 4). Specifically, Rank-1
./reviews/deep_26.md:4668:performance. For instance, under the weak-supervised setting, Rank-1
./reviews/deep_26.md:4697:Analysis of mutual distillation. To clearly show the effectiveness
./reviews/deep_26.md:4718:of the two-stage feature alignment strategy, we show the Rank-1 and
./reviews/deep_26.md:4877:query-adaptive convolution [12] and self-distillation [39]. In addition,
./reviews/deep_26.md:5432:The performance of the model on datasets is assessed using two evaluation metrics: mean average precision (mAP) and Rank-1. Both Rank1
./reviews/deep_26.md:5596:Rank1(↑)
./reviews/deep_26.md:5600:Rank1(↑)
./reviews/deep_26.md:5604:Rank1(↑)
./reviews/deep_26.md:5608:Rank1(↑)
./reviews/deep_26.md:5612:Rank1(↑)
./reviews/deep_26.md:5959:achieves best average performance in terms of mAP and Rank1 compared with other competitive models of Lifelong, Common, Generalized
./reviews/deep_26.md:5961:ReID by 1.67 % and 1.82 % according to average mAP and Rank1. Note
./reviews/deep_26.md:5985:future domains in terms of mAP and Rank1, where 𝐵𝑎𝑠𝑒𝑙𝑖𝑛𝑒 + 𝑃 𝑇 + 𝐴𝐹
./reviews/deep_26.md:6032:Year mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑)
./reviews/deep_26.md:6667:Rank1(↑)
./reviews/deep_26.md:6671:Rank1(↑)
./reviews/deep_26.md:6675:Rank1(↑)
./reviews/deep_26.md:6679:Rank1(↑)
./reviews/deep_26.md:7020:The performance of the model on datasets is assessed using two evaluation metrics: mean average precision (mAP) and Rank-1. Both Rank1
./reviews/deep_26.md:7184:Rank1(↑)
./reviews/deep_26.md:7188:Rank1(↑)
./reviews/deep_26.md:7192:Rank1(↑)
./reviews/deep_26.md:7196:Rank1(↑)
./reviews/deep_26.md:7200:Rank1(↑)
./reviews/deep_26.md:7547:achieves best average performance in terms of mAP and Rank1 compared with other competitive models of Lifelong, Common, Generalized
./reviews/deep_26.md:7549:ReID by 1.67 % and 1.82 % according to average mAP and Rank1. Note
./reviews/deep_26.md:7573:future domains in terms of mAP and Rank1, where 𝐵𝑎𝑠𝑒𝑙𝑖𝑛𝑒 + 𝑃 𝑇 + 𝐴𝐹
./reviews/deep_26.md:7620:Year mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑)
./reviews/deep_26.md:8255:Rank1(↑)
./reviews/deep_26.md:8259:Rank1(↑)
./reviews/deep_26.md:8263:Rank1(↑)
./reviews/deep_26.md:8267:Rank1(↑)
./reviews/deep_26.md:8488:As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
./reviews/deep_26.md:8514:To clearly outline our experiments, we first describe the experimental
./reviews/deep_26.md:8648:Rank-1
./reviews/deep_26.md:8650:Rank-10
./reviews/deep_26.md:8652:Rank-20
./reviews/deep_26.md:8656:Rank-1
./reviews/deep_26.md:8658:Rank-10
./reviews/deep_26.md:8660:Rank-20
./reviews/deep_26.md:9001:impressive results with 91.1% Rank-1 accuracy and 81.7% mAP in the
./reviews/deep_26.md:9002:Visible2thermal mode, and 89.9% Rank-1 accuracy and 80.5% mAP
./reviews/deep_26.md:9004:LCNL [50], exceeding its Rank-1 accuracy by 5.5% and mAP by 3.0%
./reviews/deep_26.md:9005:in the Visible2thermal mode, and surpassing it by 5.9% in Rank-1 and
./reviews/deep_26.md:9010:model with 50% labeled data achieves 86.8% Rank-1 and 77.3% mAP
./reviews/deep_26.md:9011:in the Visible2thermal mode, and 82.1% Rank-1 and 75.2% mAP in the
./reviews/deep_26.md:9018:Our method achieves impressive results, recording a Rank-1 accuracy of 76.6% and an mAP of 73.0%. Thanks to the proposed SI, SM,
./reviews/deep_26.md:9021:21.8% in Rank-1 accuracy and 20.0% in mAP without adding extra network parameters. Furthermore, our method improved Rank-1 by 1.9%
./reviews/deep_26.md:9027:gains impressive results with Rank-1 accuracy of 58.2%, 68.5%, and
./reviews/deep_26.md:9031:it by 5.4% in Rank-1 accuracy with 50% labeled samples. Furthermore,
./reviews/deep_26.md:9042:65.1% Rank-1 accuracy and 62.3% mAP under full-supervised setting
./reviews/deep_26.md:9043:and achieves 43.0% Rank-1 accuracy and 37.8% mAP under the weaksupervised setting. Next, we present the results after integrating the
./reviews/deep_26.md:9046:improves to 66.3% Rank-1 accuracy and 62.9% mAP under the fullsupervised setting. Besides, the performance reaches 48.3% Rank-1
./reviews/deep_26.md:9048:a significant enhancement, achieving 71.9% Rank-1 and 68.2% mAP
./reviews/deep_26.md:9066:Rank-1
./reviews/deep_26.md:9070:Rank-1
./reviews/deep_26.md:9219:Comparisons with six advanced methods under the weak-supervised setting, and all methods are evaluated using Rank-1 (%) and mAP (%). All results
./reviews/deep_26.md:9243:Rank-1
./reviews/deep_26.md:9247:Rank-1
./reviews/deep_26.md:9251:Rank-1
./reviews/deep_26.md:9255:Rank-1
./reviews/deep_26.md:9553:Rank-1
./reviews/deep_26.md:9555:Rank-10
./reviews/deep_26.md:9557:Rank-20
./reviews/deep_26.md:9561:Rank-1
./reviews/deep_26.md:9563:Rank-10
./reviews/deep_26.md:9565:Rank-20
./reviews/deep_26.md:9647:reaches 53.5% Rank-1 and 48.6% mAP. These improvements highlight
./reviews/deep_26.md:9650:Table 4), our TSKD gains 74.7% Rank-1 and 71.3% mAP under the fullsupervised setting, and achieves 51.4% Rank-1 and 47.2% mAP under
./reviews/deep_26.md:9656:achieves 74.9% Rank-1 accuracy and 72.1% mAP under full-supervised
./reviews/deep_26.md:9663:Fig. 4. Rank-1 and mAP with different mimic learning schemes on the SYSU-MM01 dataset.
./reviews/deep_26.md:9665:setting and achieves 56.0% Rank-1 accuracy and 51.7% mAP under
./reviews/deep_26.md:9668:TSKD achieves 76.6% Rank-1 and 73.0% mAP under the full-supervised
./reviews/deep_26.md:9669:setting, surpassing the baseline by substantial margins of 11.5% Rank-1
./reviews/deep_26.md:9671:58.2% Rank-1 and 56.3% mAP, surpassing the baseline by 15.2% in
./reviews/deep_26.md:9672:Rank-1 and 18.5% in mAP.
./reviews/deep_26.md:9674:Table 4) leads to a clear drop in performance compared to the configuration with all components (Index 6 in Table 4). Specifically, Rank-1
./reviews/deep_26.md:9680:performance. For instance, under the weak-supervised setting, Rank-1
./reviews/deep_26.md:9709:Analysis of mutual distillation. To clearly show the effectiveness
./reviews/deep_26.md:9730:of the two-stage feature alignment strategy, we show the Rank-1 and
./reviews/deep_26.md:9953:Fig. 3. Developments in UAV-based person ReID from early 2000s.
./reviews/deep_26.md:9978:Early 2000s: The early 2000s saw the widespread adoption of UAVs
./reviews/deep_26.md:10024:75.5% rank-1 accuracy, 72.7% mAP on DRHIT01;
./reviews/deep_26.md:10054:80.72% mAP, 88.77% Rank-1 in aerial-to-CCTV; Effective
./reviews/deep_26.md:10064:79% Rank-1 with FL vs. 65% local on AHU-UAV;
./reviews/deep_26.md:10130:Mid-2000s to Early 2010s: From the mid-2000s to the early 2010s,
./reviews/deep_26.md:10297:et al., 2016). This increased difficulty can be attributed to several
./reviews/deep_26.md:10776:Additional Key Words and Phrases: Person re-identification, generative adversarial network, illuminationadaptive
./reviews/deep_26.md:11403:illumination-adaptive datasets for training and evaluating our proposed method. Gamma correction
./reviews/deep_26.md:11416:Evaluation Metrics. Following [17], we assess the performance of our algorithm using rank-K
./reviews/deep_26.md:11417:(K = 1, 5, 10) and Mean Average Precision (mAP). The rank-K metric evaluates the percentage of
./reviews/deep_26.md:11457:rank-1
./reviews/deep_26.md:11458:rank-5
./reviews/deep_26.md:11484:rank-10
./reviews/deep_26.md:11512:rank-1
./reviews/deep_26.md:11527:rank-5
./reviews/deep_26.md:11541:rank-10
./reviews/deep_26.md:11621:rank-1
./reviews/deep_26.md:11622:rank-5
./reviews/deep_26.md:11634:rank-10
./reviews/deep_26.md:11643:a decrease of approximately 2.4% in rank-1, 1.4% in rank-5, and 0.6% in rank-10. The potential
./reviews/deep_26.md:11648:results on both datasets. Especially on Duke-AI, it further increased rank-1, rank-5, and rank-10
./reviews/deep_26.md:11656:ablation experiments on Market1501-AI. The rank-1, rank-5, and rank-10 mAP accuracies (%) are
./reviews/deep_26.md:11660:resulting in inferior performance compared to AGW, the rank-1 accuracy decreased by 0.4%;
./reviews/deep_26.md:11664:that both modules contribute to improved retrieval performance, with a rank-1 accuracy
./reviews/deep_26.md:11668:— However, AGW + DFSC outperformed AGW + LFCM by an additional 0.2% in rank-1 accuracy,
./reviews/deep_26.md:11673:enhancement, DLFC (AGW + DFSC+LFCM) achieved a rank-1 accuracy improvement of 0.4%
./reviews/deep_26.md:11734:284:80.72% mAP, 88.77% Rank-1 in aerial-to-CCTV; Effective
./reviews/deep_26.md:11772:1041:The above challenges highlight the complexities involved in the data
./reviews/deep_26.md:11808:1452:benchmark for this task. The rank-1 accuracy and mAP scores were similar across both datasets, indicating that existing short-term ReID techniques are relatively robust to the challenges posed by UAV-based person ReID. Specifically, NVAN achieved a Rank-1 accuracy of 83.64% on
./reviews/deep_26.md:11815:1503:and a Rank-1 accuracy of 49.79%, outperforming the other approaches.
./reviews/deep_26.md:11816:1506:resolutions introduce additional challenges. However, despite the improvements brought by SP, the absolute mAP and Rank-1 accuracies
./reviews/deep_26.md:11819:1518:Rank-1 accuracy of 88.77% in the aerial-to-CCTV setting. This demonstrates the effectiveness of combining a Vision Transformer backbone
./reviews/deep_26.md:11935:2438:Zhong, Z., Zheng, L., Cao, D., Li, S., 2017. Re-ranking person re-identification with
./reviews/deep_26.md:11976:- UAV Shaking: Strong winds cause UAV instability, resulting in
./reviews/deep_26.md:11978:and quality of the data, adding complexity to the identification task (Li
./reviews/deep_26.md:11996:introduced additional complexity due to posture variability, clothing diversity, and frequent occlusions between individuals, which negatively
./reviews/deep_26.md:12004:The above challenges highlight the complexities involved in the data
./reviews/deep_26.md:12014:Ensemble of Local Features (ELF) - RankSVM - BR-SVM KISS
./reviews/deep_26.md:12092:of local features (ELF) combined with various machine learning algorithms, including RankSVM (Zhang et al., 2016), BR-SVM (Tsujitani
./reviews/deep_26.md:12094:to improve ReID accuracy by focusing on local features, ranking-based
./reviews/deep_26.md:12123:while Faster R-CNN and Cascade R-CNN progressively refine object proposals. Sparse R-CNN reduces computational complexity using sparse
./reviews/deep_26.md:12192:Rank-1 accuracy. ReID is evaluated as a retrieval task, where the
./reviews/deep_26.md:12195:Rank-1 accuracy.
./reviews/deep_26.md:12196:The Rank-1 accuracy is calculated as:
./reviews/deep_26.md:12197:# correct matches at rank-1
./reviews/deep_26.md:12212:is positioned at the top of the ranked list of potential matches, effectively representing the system’s precision in immediate identification. Zhong et al. (2017).
./reviews/deep_26.md:12229:Rank-1
./reviews/deep_26.md:12240:RankSVM
./reviews/deep_26.md:12399:performance with an mAP of 47%, outperforming more complex methods like RankSVM (Zhang et al., 2016) and KISS (Tao et al., 2016),
./reviews/deep_26.md:12415:benchmark for this task. The rank-1 accuracy and mAP scores were similar across both datasets, indicating that existing short-term ReID techniques are relatively robust to the challenges posed by UAV-based person ReID. Specifically, NVAN achieved a Rank-1 accuracy of 83.64% on
./reviews/deep_26.md:12418:et al., 2019) performed well, achieving Rank-1 accuracies of 74.5% and
./reviews/deep_26.md:12424:features, yielded a Rank-1 accuracy of only 49.88%, substantially lower
./reviews/deep_26.md:12426:the increased difficulty of reidentifying individuals in different sessions
./reviews/deep_26.md:12446:rank-1 accuracy over using Triplet loss (Hermans et al., 2017) alone.
./reviews/deep_26.md:12457:components, achieves an mAP of 80.72% and a Rank-1 accuracy of
./reviews/deep_26.md:12459:of 3.69% in mAP and 3.37% in Rank-1 accuracy compared to using the
./reviews/deep_26.md:12466:and a Rank-1 accuracy of 49.79%, outperforming the other approaches.
./reviews/deep_26.md:12469:resolutions introduce additional challenges. However, despite the improvements brought by SP, the absolute mAP and Rank-1 accuracies
./reviews/deep_26.md:12470:remain below 50%, which highlights the difficulty of this new ReID
./reviews/deep_26.md:12481:Rank-1 accuracy of 88.77% in the aerial-to-CCTV setting. This demonstrates the effectiveness of combining a Vision Transformer backbone
./reviews/deep_26.md:12484:AHU-UAV dataset, achieving a Rank-1 accuracy of 0.79 when using
./reviews/deep_26.md:12488:other datasets, state-of-the-art ReID methods achieve Rank-1 accuracies
./reviews/deep_26.md:12501:Table 4 shows the Rank-1 accuracy results in the AHU-UAV dataset
./reviews/deep_26.md:12504:without federated learning, the Rank-1 accuracy is 0.65. Training the
./reviews/deep_26.md:12507:the accuracy of Rank-1 to 0.79. Model training with knowledge distillation (KD) (Gou et al., 2021), after FL, the global model is compressed
./reviews/deep_26.md:12508:using knowledge distillation. The Rank-1 accuracy is 0.69.
./reviews/deep_26.md:12524:and Rank-1 accuracies between 60%–65%. Specifically, Tricks obtains
./reviews/deep_26.md:12525:the highest mAP at 63.41%, while DG-Net achieves the best Rank-1
./reviews/deep_26.md:12548:Precision) and CMC-k (Cumulative Matching Characteristic at rank-k),
./reviews/deep_26.md:12549:with a focus on rank-1 accuracy. CMC-k measures the probability that
./reviews/deep_26.md:12550:a correct match for a query appears within the top-k ranked results,
./reviews/deep_26.md:12552:at rank-1, which indicates the likelihood that the correct match is the
./reviews/deep_26.md:12553:top-ranked result (Wang et al., 2007). The EP+EVA method, which
./reviews/deep_26.md:12656:overall mean average precision (mAP) and Rank-1 accuracy scores remain below 50%, highlighting the continued difficulty of re-identifying
./reviews/deep_26.md:12731:views, and objects obstructing the vision. By highlighting the integration of these complexities into the dataset, a thorough assessment of
./reviews/deep_26.md:12741:can effectively handle the complexities of real-world scenarios. Furthermore, the establishment of standardized evaluation metrics and
./reviews/deep_26.md:12918:in mAP and 11.94% in Rank1 accuracy. Linter is favorable to
./reviews/deep_26.md:12921:introduction of the loss Lintra , the accuracy of mAP and Rank1
./reviews/deep_26.md:13007:2. 重定义动作：作者把 ReID 从单一任务重定义为 Anytime ReID，用“时间点”和“时间间隔”划出六个场景。关键词是 anytime retrieval、multi-scenario retrieval、scenario-specific cues、inter-scenario interference。
./reviews/deep_26.md:13011:4. 证据闭环：数据集 AT-USTC 本身证明旧数据集缺口，21 个月、RGB 和红外、多衣服、多季节支撑“anytime”成立。模型证据包括六场景分别评测、跨数据集泛化、MS-ReID、MoAE、HDW 消融，以及和其他 MoE 方法对比，证明不是单纯多头分类。
./reviews/deep_26.md:13084:2. 重定义动作：作者把 ReID 从单一任务重定义为 Anytime ReID，用“时间点”和“时间间隔”划出六个场景。关键词是 anytime retrieval、multi-scenario retrieval、scenario-specific cues、inter-scenario interference。
./reviews/deep_26.md:13088:4. 证据闭环：数据集 AT-USTC 本身证明旧数据集缺口，21 个月、RGB 和红外、多衣服、多季节支撑“anytime”成立。模型证据包括六场景分别评测、跨数据集泛化、MS-ReID、MoAE、HDW 消融，以及和其他 MoE 方法对比，证明不是单纯多头分类。
./reviews/deep_23.md:57:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_23.md:74:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_23.md:77:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_23.md:86:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_23.md:118:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_23.md:148:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_23.md:288:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_23.md:416:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_23.md:445:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_23.md:448:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_23.md:449:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_23.md:526:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_23.md:530:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_23.md:544:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_23.md:550:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_23.md:674:which achieves rank-1 accuracies of 76.0%, 93.2%, 92.0%, 95.6%, and 91.1% respectively. Compared with
./reviews/deep_23.md:675:KPR (ECCV 2024), ETND (TCSVT 2024) and TTPM (PR 2025), RMPSNet improves the rank-1 performance on
./reviews/deep_23.md:878:complexity.
./reviews/deep_23.md:1597:for 10 epochs, during which the learning rate linearly increases from
./reviews/deep_23.md:1612:Rank-1 and mean Average Precision (mAP). Therefore, we also strictly
./reviews/deep_23.md:1614:is conducted under a single-query setting without using re-ranking
./reviews/deep_23.md:1624:achieved a Rank-1 accuracy of 95.6% and an mAP of 89.7% on Market1501, slightly inferior to the current best method THCB, which has
./reviews/deep_23.md:1625:a Rank-1 accuracy of 96.2% and an mAP of 90.6%. On DukeMTMCReID, RMPSNet attains a Rank-1 accuracy of 91.1% and an mAP of
./reviews/deep_23.md:1637:Occluded-Duke, RMPSNet achieves a Rank-1 accuracy of 76.0% and an
./reviews/deep_23.md:1639:Rank-1 accuracy and 1.6% in mAP. Compared to ProFD, which achieves
./reviews/deep_23.md:1640:70.8% in Rank-1 accuracy and 62.8% in mAP, RMPSNet shows a notable improvement of 5.2% and 2.2%, respectively. This improvement
./reviews/deep_23.md:1651:a Rank-1 accuracy of 93.2% and an mAP of 89.8%, surpassing ProFD by
./reviews/deep_23.md:1652:2.1% in Rank-1 accuracy and 1.3% in mAP. Compared to other strong
./reviews/deep_23.md:1654:approximately 4.0% to 9.6% in Rank-1 accuracy and around 4.1% to
./reviews/deep_23.md:1659:Partial-ReID, RMPSNet achieves a Rank-1 accuracy of 92.0% and an
./reviews/deep_23.md:1660:mAP of 86.9%, outperforming PADE by 2.7% in Rank-1 accuracy and
./reviews/deep_23.md:1677:clearly demonstrate its strong potential as a practical and versatile
./reviews/deep_23.md:1789:presented in Table 2, the baseline model, achieves a Rank-1 accuracy
./reviews/deep_23.md:1791:Rank-1 accuracy and 87.2% in mAP on Occluded-ReID. When the
./reviews/deep_23.md:1793:is called ‘Baseline+DMPA’, significant improvements have been observed. Specifically, on Occluded-Duke, the Rank-1 accuracy improves
./reviews/deep_23.md:1794:by 1.5% to reach 68.6%, and the mAP increases by 1.4%, achieving 60.9%. On Occluded-ReID, the Rank-1 accuracy rises by 2.2%
./reviews/deep_23.md:1803:a Rank-1 accuracy drop of 1.8% and an mAP decrease of 1.0% on
./reviews/deep_23.md:1805:0.3% in Rank-1 accuracy and 0.6% in mAP. This consistent performance degradation highlights the essential role of DMPA in enhancing
./reviews/deep_23.md:1817:augmentation [9], achieves a Rank-1 accuracy of 67.1% and an mAP
./reviews/deep_23.md:1818:of 59.5% on Occluded-Duke, as well as 90.5% Rank-1 and 87.2%
./reviews/deep_23.md:1819:mAP on Occluded-ReID. When adding the RPE module to the baseline, as was noted in ‘Baseline+RPE’, we observe substantial performance improvements. On Occluded-Duke, the Rank-1 accuracy increases by 6.1% to reach 73.2%, and the mAP improves by 4.2%
./reviews/deep_23.md:1825:the effectiveness of the MDO module, (4) complexity analysis, (5)
./reviews/deep_23.md:1849:Complexity analysis of different occluded person ReID methods. ‘FLOPs’ represents the number of floating-point operations, and ‘Params’ represents the
./reviews/deep_23.md:1915:and 1.2% in mAP on Occluded-Duke, and decreases of 0.2% in Rank-1
./reviews/deep_23.md:1925:to 63.7%. On Occluded-ReID, the Rank-1 accuracy rises by 1.6% to
./reviews/deep_23.md:1937:Rank-1 accuracy and 1.8% in mAP on Occluded-Duke. Similarly, on
./reviews/deep_23.md:1938:Occluded-ReID, excluding RPE leads to a reduction of 0.5% in Rank1 accuracy and 0.8% in mAP. These results highlight the important
./reviews/deep_23.md:1946:5.4. Complexity analysis
./reviews/deep_23.md:1948:ReID methods, we compare their computational complexity and model
./reviews/deep_23.md:1953:provide a better trade-off between complexity and accuracy but still
./reviews/deep_23.md:1957:Rank-1 accuracy of 76.0%. These results demonstrate that our design
./reviews/deep_23.md:1958:effectively balances model complexity and recognition performance.
./reviews/deep_23.md:1960:without incurring excessive redundancy. RMPSNet justifies its moderate increase in complexity by providing significant gains in occlusion
./reviews/deep_23.md:1968:models may remain preferable in highly resource-constrained environments. The practical impact of the increased complexity is a trade-off
./reviews/deep_23.md:1979:dataset, the Rank-1 accuracy and mAP remain unchanged, but there
./reviews/deep_23.md:1981:Rank-1 accuracy increases to 92.5% and the mAP rises to 88.6%. These
./reviews/deep_23.md:1987:observed. On Occluded-Duke, the Rank-1 accuracy improves to 71.0%,
./reviews/deep_23.md:1989:of 3.9% in Rank-1 accuracy and 2.4% in mAP over the Baseline. On
./reviews/deep_23.md:1990:Occluded-ReID, the Rank-1 accuracy reaches 92.7%, and the mAP
./reviews/deep_23.md:1997:from RMPSNet leads to a performance drop of 2.7% in Rank-1 accuracy
./reviews/deep_23.md:3027:scene complexity and stronger degradations. Among existing methods,
./reviews/deep_23.md:3030:consistently improves their performance. Notably, MDCD w/ CLIPReID achieves the best mAP and Rank-1 scores under degraded conditions, demonstrating the effectiveness and generality of the proposed
./reviews/deep_23.md:3393:complexity.
./reviews/deep_23.md:3862:positives to rank ahead of negatives based on distance, and
./reviews/deep_23.md:3912:body parts and carried items at the pixel level. Moreover, uses paired image and text encoders to learn a shared embedsince person ReID is inherently formulated as a ranking ding space via contrastive learning, benefiting applications such
./reviews/deep_23.md:4565:Rank-k in CMC measures the probability of the correct match
./reviews/deep_23.md:4575:Rank-1
./reviews/deep_23.md:4608:hidden size 768; a linear projection reduces the image feature approximately 2.0% in both Rank-1 accuracy and mAP across
./reviews/deep_23.md:4629:approaches (ISP [24], CLIP-ReID [14], and Instruct-ReID [15]) PRCC, where Rank-1 drops by 4.6% and mAP decreases from
./reviews/deep_23.md:4642:improves Rank-1 by 3.8% and mAP by 2.7% over CLIP- excessive emphasis on clothing-relevant representations during
./reviews/deep_23.md:4650:On PRCC, SCI surpasses AIM by 3.5% in Rank-1 accuracy the latent space distributions at different stages of the baseline
./reviews/deep_23.md:5270:5) Visualization of Feature Similarity: In Fig. 7, we visualize 42.1% in Rank-1 accuracy and 18.6% in mAP, as depicted
./reviews/deep_23.md:6260:the original feature map. For each patch, adaptive average
./reviews/deep_23.md:6301:weighting factor g(t), which increases linearly over time. This
./reviews/deep_23.md:6462:Rank-1
./reviews/deep_23.md:6466:Rank-1
./reviews/deep_23.md:6470:Rank-1
./reviews/deep_23.md:6474:Rank-1
./reviews/deep_23.md:6611:datasets, SYSU-MM01 [22] and RegDB [23], using Rank-1
./reviews/deep_23.md:6626:VB Sampler ensures diversity and balance in the viewpoints [18], we use re-ranking [24] for evaluation.
./reviews/deep_23.md:6658:Rank-1
./reviews/deep_23.md:6724:Rank-1
./reviews/deep_23.md:6785:the BiLSTM setup consistently yields higher Rank-1 and
./reviews/deep_23.md:6796:(Avg). In contrast, the Decrease strategy adopts a linearly
./reviews/deep_23.md:6800:early-stage alignment. The Avg strategy, on the other hand,
./reviews/deep_23.md:6849:Zha, “Modality-adaptive mixup and invariant decomposition for rgbinfrared person re-identification,” in Proceedings of the AAAI conference
./reviews/deep_23.md:6890:[16] Yiyuan Zhang, Sanyuan Zhao, Yuhao Kang, and Jianbing Shen, “Modality synergy complement learning with cascaded aggregation for visibleinfrared person re-identification,” in European Conference on Computer
./reviews/deep_23.md:6915:[24] Zhun Zhong, Liang Zheng, Donglin Cao, and Shaozi Li, “Re-ranking
./reviews/deep_23.md:6944:occlusion, viewpoint variations, and the inherent difficulty of
./reviews/deep_23.md:6967:contrastive learning and early learning regularization based on
./reviews/deep_23.md:7032:inherent characteristics significantly increase the difficulty of
./reviews/deep_23.md:7061:of cross-modal pairs. This exacerbates the complexity of the
./reviews/deep_23.md:7072:fit clean data simple patterns during training at the early
./reviews/deep_23.md:7130:on the stable predictions from the early training stage [33],
./reviews/deep_23.md:7131:our method designs early learning regularization loss and
./reviews/deep_23.md:7180:samples and restrains them in the early learning stage
./reviews/deep_23.md:7308:with early learning regularization. Finally, section III-C outlines our self-refined label rectification strategy, detailing how
./reviews/deep_23.md:7367:Consistency Learning. Our method enhances the robustness of cross-modal learning by integrating contrastive learning with early learning regularization. This
./reviews/deep_23.md:7399:undergo an “early learning” phase, where they tend to fit clean
./reviews/deep_23.md:7401:early training stage, the model learns those general patterns
./reviews/deep_23.md:7689:ambiguous samples is suppressed in the early phase and thus
./reviews/deep_23.md:7797:• LCNL [22]: This paper proposes an adaptive quadruple
./reviews/deep_23.md:7800:against noisy correspondence by adaptively dividing the
./reviews/deep_23.md:8282:Rank-1 mAP
./reviews/deep_23.md:8283:Rank-1
./reviews/deep_23.md:8351:Rank-1 accuracy and mean average precision (mAP) are
./reviews/deep_23.md:8369:Rank-1
./reviews/deep_23.md:8371:Rank-1
./reviews/deep_23.md:8403:4.2% Rank-1 and 1.7% mAP. Although Rank-1 accuracy under
./reviews/deep_23.md:8406:the SOTA method IANet [18] by 4.2% Rank-1 and 5.1% mAP
./reviews/deep_23.md:8409:Rank-1 under general setting.
./reviews/deep_23.md:8442:Rank-1 mAP
./reviews/deep_23.md:8686:• LCNL [22]: This paper proposes an adaptive quadruple
./reviews/deep_23.md:8689:against noisy correspondence by adaptively dividing the
./reviews/deep_23.md:8704:and error accumulation. It consists of robust adaptive
./reviews/deep_23.md:8708:VI-ReID. It proposes adaptive calibration and division
./reviews/deep_23.md:8722:final results achieve 66.03% mAP, 69.08% Rank-1 accuracy,
./reviews/deep_23.md:8724:73.48% Rank-1 accuracy, and 75.42% mINP for indoor-search
./reviews/deep_23.md:8734:mAP, 3.41% Rank-1 accuracy, and 3.25% mINP for allsearch mode, and 2.96% mAP, 3.40% Rank-1 accuracy, and
./reviews/deep_23.md:8798:soft identification loss; “+Lcon ” represents the model training along with contrastive learning; “+Lelr ” indicates early
./reviews/deep_23.md:8808:samples by notably improving Rank-1 accuracy and mAP.
./reviews/deep_23.md:8890:Fig. 7. Rank-10 retrieval result on SYSU-MM01 evaluation set. Both DART and our method SRHCL are trained under 20% noise.
./reviews/deep_23.md:8901:retrieval results of our pipeline, we visualize the Rank-10
./reviews/deep_23.md:8908:annotations, where an incorrect identity, despite being dissimilar to the query, appears in the rank-10 retrieval results. It
./reviews/deep_23.md:8975:2. **重定义动作**：它把噪声样本从“需要丢掉的坏样本”改写成“可以被历史预测逐步纠正的潜在有效样本”。同时把训练波动从坏现象改写成信息来源。关键词是 `self-rectification`、`historical consistency`、`early learning`、`momentum-updated memories`、`coupled noisy labels`。
./reviews/deep_23.md:8977:4. **证据闭环**：核心消融是 baseline、加跨模态对比、加 early learning regularization、再加 label rectification 的逐步对比。可视化更关键：不同噪声率下标签准确率随 rectification 逐步上升，loss 分布里噪声样本被转成低损失干净样本，和 DART 对比显示它不是单纯抗噪，而是在回收训练数据。
./reviews/deep_23.md:9025:2. **重定义动作**：它把噪声样本从“需要丢掉的坏样本”改写成“可以被历史预测逐步纠正的潜在有效样本”。同时把训练波动从坏现象改写成信息来源。关键词是 `self-rectification`、`historical consistency`、`early learning`、`momentum-updated memories`、`coupled noisy labels`。
./reviews/deep_23.md:9027:4. **证据闭环**：核心消融是 baseline、加跨模态对比、加 early learning regularization、再加 label rectification 的逐步对比。可视化更关键：不同噪声率下标签准确率随 rectification 逐步上升，loss 分布里噪声样本被转成低损失干净样本，和 DART 对比显示它不是单纯抗噪，而是在回收训练数据。
./batches/baa:3:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_9.md:54:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_9.md:71:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_9.md:74:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_9.md:83:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_9.md:115:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_9.md:145:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_9.md:309:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_9.md:391:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_9.md:530:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_9.md:605:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_9.md:750:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_9.md:779:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_9.md:782:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_9.md:783:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_9.md:860:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_9.md:864:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_9.md:878:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_9.md:884:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_9.md:975:154:methods, e.g., +1.88% Rank-1 on SYSU-MM01 and +1.1%
./reviews/deep_9.md:1085:28:method, achieving 61.06% Rank-1 and 67.75% mAP in the
./reviews/deep_9.md:1125:   367	weights to adaptively adjust the contribution of each pattern
./reviews/deep_9.md:1971:   113	tation complexity, increased cost, and increased time consumption. A
./reviews/deep_9.md:1977:   119	complexity.
./reviews/deep_9.md:1991:   133	increasing the complexity score in identifying the persons who have
./reviews/deep_9.md:2057:   199	mechanism so the adaptive identification of relevant features within
./reviews/deep_9.md:2119:   261	Both rank accuracy
./reviews/deep_9.md:2178:   320	Complexity,
./reviews/deep_9.md:2216:   358	adaptive and defensive behavior of the horned Lizard. This optimization
./reviews/deep_9.md:2219:   361	lizard’s adaptive color changing, it adapts its search strategy dynami­
./reviews/deep_9.md:2222:   364	adaptive mechanism based on iteration progress and solution quality
./reviews/deep_9.md:2228:   370	extraction coupled with HLO’s adaptive exploration, HLGFS provides a
./reviews/deep_9.md:2311:   453	preprocessing phase reduces the identification complexity and thereby
./reviews/deep_9.md:2350:   492	between − 1 and 1, ε indicates the ranking variable that attains the value
./reviews/deep_9.md:2375:   517	ranking the extracted features with the other images. This process has
./reviews/deep_9.md:2473:    94	the complexity of the task. Existing image-based unsupervised
./reviews/deep_9.md:2570:   191	to adaptively learn different feature distributions. However,
./reviews/deep_9.md:2601:   222	adaptive framework to balance the information volume of
./reviews/deep_9.md:2669:   290	Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
./reviews/deep_9.md:3078:   113	enables modality interaction during the adaptive aggregation of temporal information, thereby
./reviews/deep_9.md:3496:    49	tracklets, while the gallery contains 4,844 tracklets. Notably, nearly 30% of the pedestrians are
./reviews/deep_9.md:3504:    57	[31]. The CMC curve, depicted as rank-k, assesses the top-k result accuracy. On the other hand,
./reviews/deep_9.md:3507:    60	metric, which represents the average of each evaluation metric such as rank-k and mAP. It is used
./reviews/deep_9.md:3552:   105	Rank-1 Rank-5 Rank-10
./reviews/deep_9.md:3592:   145	mAP Rank-1
./reviews/deep_9.md:3609:   162	Rank-5 Rank-10
./reviews/deep_9.md:3682:   235	evaluation metrics. Specifically, in the infrared-to-visible task, our Rank-1, Rank-5, and Rank10 accuracies reached 65.18%, 79.12%, and 84.60%, respectively, surpassing all listed methods.
./reviews/deep_9.md:3684:   237	improvement in Rank-1, Rank-5, and Rank-10 accuracy, respectively. In the visible-to-infrared
./reviews/deep_9.md:3685:   238	task, our Rank-5 and Rank-10 results reached 81.72% and 84.19%, respectively, outperforming the
./reviews/deep_9.md:3702:    30	LLCM, surpassing existing methods by significant margins without resorting to reranking or additional labeled data. Code is
./reviews/deep_9.md:3740:    68	gap through input-space alignment. However, generative models suffer from training instability and detail degradation, while
./reviews/deep_9.md:3849:   177	Ranking Regularization (CMRR) loss. The CMSR loss is
./reviews/deep_9.md:3857:   185	diversity of cross-modality features. It achieves this by maximizing the inconsistency in ranking among features extracted
./reviews/deep_9.md:3859:   187	challenge posed by non-differentiable ranking, a differentiable
./reviews/deep_9.md:3860:   188	Spearman rank function is incorporated.
./reviews/deep_9.md:3872:   200	maximizing the inconsistency in ranking among features
./reviews/deep_9.md:3908:   236	complexity and large number of parameters also limit their
./reviews/deep_9.md:4025:   353	Ranking Regularization (CMRR) loss are specifically designed
./reviews/deep_9.md:4096:   424	Enhancer (FADE) modules. (d, e) Illustrations of the key principles behind our novel Cross-Modality Soft Retrieval (CMSR) and Cross-Modality Ranking
./reviews/deep_9.md:4113:   441	[18] exhibit prohibitive quadratic computational complexity,
./reviews/deep_9.md:4124:   452	reduction in both parameter count and computational complexity. Specifically, the matrix X̃L contains non-zero values
./reviews/deep_9.md:4528:   856	2) Cross-Modality Ranking Regularization Loss: To
./reviews/deep_9.md:4530:   858	module, we propose a Cross-Modality Ranking Regularization
./reviews/deep_9.md:4531:   859	(CMRR) loss based on differentiable Spearman rank
./reviews/deep_9.md:4532:   860	correlation, starting from the inconsistency in the ranking of
./reviews/deep_9.md:4561:   889	can obtain the ranking list of features within the set:
./reviews/deep_9.md:4599:   927	Based on the ranking lists of two different sets, the Spearman
./reviews/deep_9.md:4600:   928	rank correlation coefficient can be defined as:
./reviews/deep_9.md:4634:   962	and the smaller this value is, the more inconsistent the ranking
./reviews/deep_9.md:4635:   963	lists of the two sets are, thus achieving the goal of ranking
./reviews/deep_9.md:4637:   965	the softrank operation [51] to obtain the differentiable crossmodality ranking regularization loss:
./reviews/deep_9.md:4662:   990	softrank
./reviews/deep_9.md:4671:   999	ranking regularization loss is:
./reviews/deep_9.md:4769:   353	Ranking Regularization (CMRR) loss are specifically designed
./reviews/deep_9.md:4840:   424	Enhancer (FADE) modules. (d, e) Illustrations of the key principles behind our novel Cross-Modality Soft Retrieval (CMSR) and Cross-Modality Ranking
./reviews/deep_9.md:4857:   441	[18] exhibit prohibitive quadratic computational complexity,
./reviews/deep_9.md:4868:   452	reduction in both parameter count and computational complexity. Specifically, the matrix X̃L contains non-zero values
./reviews/deep_9.md:4988:290:Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
./reviews/deep_9.md:5043:749:in Table IV, we present the experimental results on ICFGPEDES, which achieves excellent performance, with a Rank-1
./reviews/deep_9.md:5046:756:database. As shown in Table IV, the key metrics of Rank-1
./reviews/deep_9.md:5109:991:— The introduced CIM module provides an 8.97% improvement in Rank-1 accuracy and approximately a 6.7% boost in mAP over the baseline. This demonstrates its effectiveness in enhancing
./reviews/deep_9.md:5111:997:results clearly highlight the effectiveness of the two proposed modules.
./reviews/deep_9.md:5117:1026:Fig. 4. The Rank-5 results for certain queries using our approach in the infrared-visible and visible-infrared
./reviews/deep_9.md:5128:1066:Fig. 7. Comparison of performance with varying values of S in the FDM for both infrared-visible and visibleinfrared modes on the HITSZ-VCM dataset, measured by Rank-1 accuracy.
./reviews/deep_9.md:5149:25:Ranking Regularization (CMRR) loss enhances feature diversity
./reviews/deep_9.md:5191:185:diversity of cross-modality features. It achieves this by maximizing the inconsistency in ranking among features extracted
./reviews/deep_9.md:5198:236:complexity and large number of parameters also limit their
./reviews/deep_9.md:5250:452:reduction in both parameter count and computational complexity. Specifically, the matrix X̃L contains non-zero values
./reviews/deep_9.md:5287:   687	Rank-1, respectively. The use of cycle construction, i.e., +Lr ec
./reviews/deep_9.md:5289:   689	e.g., +4.56% on Rank-1 compared to “+D M M.” Meanwhile,
./reviews/deep_9.md:5298:   698	can improve the baseline by up to +12.97% Rank-1, strongly
./reviews/deep_9.md:5319:   719	our DMM, e.g., −3.52% Rank-1 and −3.69% mAP. Overall,
./reviews/deep_9.md:5342:   742	+4.57% on Rank-1 and +5.46% on mAP. Under the setting
./reviews/deep_9.md:5344:   744	expanded. For instance, the SOTA performance on Singleshot Rank-1 and Multi-shot Rank-1 is improved by +9.99%
./reviews/deep_9.md:5348:   748	All-Search Single-Shot, e.g., +1.88% on Rank-1 and +0.82%
./reviews/deep_9.md:5355:   755	and +1.9% Rank-1 on two settings. When taking the same
./reviews/deep_9.md:5372:   772	3.23% in Rank-1. Especially when the value is too large,
./reviews/deep_9.md:5402:   802	that the proposed CycleTrans significantly improves the performance, i.e., +17.59% and +18.33% on Rank-1 and mAP,
./reviews/deep_9.md:5405:   805	CycleTrans* achieves +1.88% and +0.82% on Rank-1 and
./reviews/deep_9.md:5485:   872	+1.88% Rank-1 and +1.1% Rank-1 on SYSU-MM01 and
./reviews/deep_9.md:5723:   623	different ranks and shows reliability in fetching relevant matches. This
./reviews/deep_9.md:5724:   624	ensures robust ranking and matching performance. Similar trends are
./reviews/deep_9.md:5725:   625	also noted in various datasets where Rank-1 to Rank-5 accuracies are
./reviews/deep_9.md:5727:   627	perform well on datasets with different complexities and may have
./reviews/deep_9.md:5731:   631	accuracy is validated for images ranked 1 to 5.
./reviews/deep_9.md:5737:   637	SSLA [13], EFAN [19], SA [20], AAT [21], APS [22], Hybrid Ranking
./reviews/deep_9.md:5744:   644	correct matching image is ranked within the top n retrieved images from
./reviews/deep_9.md:5755:   655	Rank-1 accuracy indicates the proportion of queries where the model
./reviews/deep_9.md:5830:   730	Rank 1
./reviews/deep_9.md:5832:   732	Rank 2
./reviews/deep_9.md:5834:   734	Rank 3
./reviews/deep_9.md:5836:   736	Rank 4
./reviews/deep_9.md:5838:   738	Rank 5
./reviews/deep_9.md:6245:   742	unsupervised methods, with a Rank-1 accuracy of 50.20%
./reviews/deep_9.md:6252:   749	in Table IV, we present the experimental results on ICFGPEDES, which achieves excellent performance, with a Rank-1
./reviews/deep_9.md:6254:   751	our method surpasses it by 10.29% in Rank-1 accuracy and
./reviews/deep_9.md:6259:   756	database. As shown in Table IV, the key metrics of Rank-1
./reviews/deep_9.md:6343:   991	— The introduced CIM module provides an 8.97% improvement in Rank-1 accuracy and approximately a 6.7% boost in mAP over the baseline. This demonstrates its effectiveness in enhancing
./reviews/deep_9.md:6345:   993	— The designed FDM method enhances Rank-1 accuracy by 8.16% and mAP by 4.51%, highlighting
./reviews/deep_9.md:6348:   996	a 9.6% improvement in Rank-1 accuracy and a 7.13% boost in mAP over the baseline. These
./reviews/deep_9.md:6349:   997	results clearly highlight the effectiveness of the two proposed modules.
./reviews/deep_9.md:6355:  1003	outcomes in both infrared-visible and visible-infrared modes. Figure 4 illustrates the Rank-5 retrieval
./reviews/deep_9.md:6378:  1026	Fig. 4. The Rank-5 results for certain queries using our approach in the infrared-visible and visible-infrared
./reviews/deep_9.md:6406:  1054	increase in the value of S. This clearly demonstrates the positive impact of higher-dimensional
./reviews/deep_9.md:6445:   929	Rank-1
./reviews/deep_9.md:6452:   936	Rank-5 Rank-10 Rank-20 mAP
./reviews/deep_9.md:6470:   954	Rank-1
./reviews/deep_9.md:6477:   961	Rank-5 Rank-10 Rank-20 mAP
./reviews/deep_9.md:6519:25:Ranking Regularization (CMRR) loss enhances feature diversity
./reviews/deep_9.md:6562:177:Ranking Regularization (CMRR) loss. The CMSR loss is
./reviews/deep_9.md:6600:353:Ranking Regularization (CMRR) loss are specifically designed
./reviews/deep_9.md:6616:424:Enhancer (FADE) modules. (d, e) Illustrations of the key principles behind our novel Cross-Modality Soft Retrieval (CMSR) and Cross-Modality Ranking
./reviews/deep_9.md:6656:859:(CMRR) loss based on differentiable Spearman rank
./reviews/deep_9.md:6674:1110:(Rank-1 94.82%, the best among frequency domain methods)
./reviews/deep_9.md:6858:  1286	6) Discussion on the Parameters and Computational Complexity: Table IV provides a detailed comparison of our
./reviews/deep_9.md:6860:  1288	efficiency (parameters and computational load) and performance metrics (Rank-1 and mAP) on the SYSU-MM01 and
./reviews/deep_9.md:6865:  1293	the SYSU-MM01 dataset, MFENet surpasses AGW’s Rank-1
./reviews/deep_9.md:6902:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./reviews/deep_9.md:6930:4. 证据闭环：HITSZ-VCM 消融显示 CIM 单独带来 Rank-1 和 mAP 大提升，FDM 单独也提升，两者组合最好。t-SNE 显示同身份跨模态更聚、不同身份更分开，Rank-5 可视化展示视频检索结果。参数分析说明帧数增加和 FDM 维度变化会影响性能。
./reviews/deep_9.md:6949:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./reviews/deep_9.md:6977:4. 证据闭环：HITSZ-VCM 消融显示 CIM 单独带来 Rank-1 和 mAP 大提升，FDM 单独也提升，两者组合最好。t-SNE 显示同身份跨模态更聚、不同身份更分开，Rank-5 可视化展示视频检索结果。参数分析说明帧数增加和 FDM 维度变化会影响性能。
./batches/bae:3:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./pivot/p_1.md:133:- 不撞 SeCap：SeCap 是 adaptive prompt + local feature refinement。SHSM 反而反对局部细节主导。来源：[SeCap arXiv](https://arxiv.org/abs/2503.06965)
./pivot/p_1.md:201:- 不撞 SeCap：SeCap 是 adaptive prompt + local feature refinement。SHSM 反而反对局部细节主导。来源：[SeCap arXiv](https://arxiv.org/abs/2503.06965)
./reviews/deep_21.md:67:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_21.md:84:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_21.md:87:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_21.md:96:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_21.md:128:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_21.md:158:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_21.md:390:   278	metric matrix 𝑇 ∗ . This matrix is used for retrieval ranking.
./reviews/deep_21.md:534:   305	Completion Module (SRCM) aggregates tokens into semantically consistent regions and performs adaptive cross-modality
./reviews/deep_21.md:940:   446	only increases computational complexity but also the learning
./reviews/deep_21.md:942:   448	effectively aid the model in clearly discerning the optimal
./reviews/deep_21.md:1056:   562	query q and all proxies P, we generate a query-based descending ranking list as shown in Fig. 3. Prior to differentiating
./reviews/deep_21.md:1070:   576	The ranking list R is created based on descending similarity
./reviews/deep_21.md:1083:   589	to rank higher, thus obscuring the discovery of truly effective
./reviews/deep_21.md:1427:   865	numerical instability. By reducing feature inconsistency and
./reviews/deep_21.md:1519:   624	of a query-based ranking list. To mitigate the adverse impact
./reviews/deep_21.md:1521:   626	ranking criterion. This method is straightforward yet effective:
./reviews/deep_21.md:1522:   627	it involves summing the similarity ranking lists of queries
./reviews/deep_21.md:1524:   629	thereby generating a ranking list with enhanced robustness.
./reviews/deep_21.md:1576:   681	reduces computational complexity, providing a suitable feature representation for the subsequent softmax-based spatial
./reviews/deep_21.md:1661:   766	via Rank-1, Rank-5, and Rank-10.
./reviews/deep_21.md:1726:   831	marge of 5.0% in mAP and 2.9% in Rank-1 on Market-1501,
./reviews/deep_21.md:1727:   832	17.6% in mAP and 17.9% in Rank-1 on MSMT17, and 10.8%
./reviews/deep_21.md:1728:   833	and 8.1% in Rank-1 on VeRi-776.
./reviews/deep_21.md:1735:   840	mAP and 1.8% in Rank-1 on Market-1501, 10.6% in mAP
./reviews/deep_21.md:1736:   841	and 13.2% in Rank-1 on MSMT17, and 9.2% in mAP and
./reviews/deep_21.md:1737:   842	2.0% in Rank-1 on VeRi-776, respectively. In the proposed
./reviews/deep_21.md:1884:   344	information in the intermediate layer and adaptively suppressing the expression of occlusion information. Compared with
./reviews/deep_21.md:1978:   438	layer can make adaptive decisions on the importance of the
./reviews/deep_21.md:1996:   456	and reduce the computational complexity in the attention calculation process, Fd is channels-reduced by a 1×1 convolution
./reviews/deep_21.md:2015:   475	in the VARA module, which enables adaptive suppression of occlusion information, this alone is insufficient for
./reviews/deep_21.md:2114:   574	This adaptive method of generating skeletal masks can
./reviews/deep_21.md:2532:   437	Compare with other methods on Market-1501. The R1, R-5, R-10 indicats the metric Rank-1, Rank-5, Rank10 respectively. Here, the ReID models are trained on
./reviews/deep_21.md:2579:   349	These local features are then fused using adaptive weights,
./reviews/deep_21.md:2916:   686	However, in the early stages of training, due to significant
./reviews/deep_21.md:2932:   702	demonstrated in Algorithm 2, this strategy facilitates the early
./reviews/deep_21.md:3259:   643	generate more IDs and images than real training sets. Doubling the training data results in a 6.9 % improvement on Rank-1 and an 11.6 % improvement on mAP. By providing much more training data for the ReID
./reviews/deep_21.md:3273:   657	ADM yields an 8.2 % drop in Rank-1 accuracy and a 44.6 % drop in mAP,
./reviews/deep_21.md:3392:  1010	significantly increase the difficulty of the matching task and
./reviews/deep_21.md:3427:  1045	and mean Average Precision (mAP). The CMC curve (Rankk) measures top-k accuracy, while mAP calculates average
./reviews/deep_21.md:3442:  1060	(Index 1), introducing G (Index 2) improves the Rank-1
./reviews/deep_21.md:3446:  1064	Rank-1 to 60.14% (+1.82%), while introducing M alone (Index
./reviews/deep_21.md:3584:   108	Rank-1 Rank-5 Rank-10
./reviews/deep_21.md:3657:   181	Rank-1 Rank-5
./reviews/deep_21.md:3693:   217	Rank-10
./reviews/deep_21.md:3724:   248	did not outperform the AGW algorithm in terms of the Rank-K metric on the Duke-IA dataset.
./reviews/deep_21.md:3818:    78	configuration (Index 5) boosts Rank-1 to 62.63%, outperforming both G-only and S-only variants by more than 2%,
./reviews/deep_21.md:3824:    84	65.57% Rank-1 accuracy in two modes, respectively.
./reviews/deep_21.md:3827:    87	introducing G (Index 2) improves the Rank-1 accuracy from
./reviews/deep_21.md:3830:    90	yield consistent gains over the baseline, increasing Rank1 to 71.58% (+1.36%) and 71.26% (+1.04%), respectively.
./reviews/deep_21.md:3832:    92	The G+S configuration (Index 5) raises Rank-1 to 73.69%,
./reviews/deep_21.md:3836:    96	best results, reaching 75.63% and 74.68% Rank-1 accuracy in
./reviews/deep_21.md:3861:   121	62.63%/60.29% to 62.81%/60.77% in Rank-1/mAP, and on
./reviews/deep_21.md:3901:   161	with complexity analysis.
./reviews/deep_21.md:3908:   168	Rank-1
./reviews/deep_21.md:3910:   170	Rank-1
./reviews/deep_21.md:3921:   181	Complexity
./reviews/deep_21.md:4060:    56	accuracies of 96.1% and 91.2% on the two datasets, respectively, surpassing the performances of nearly all occluded
./reviews/deep_21.md:4227:   223	as input improved key metrics such as Rank-1 identification
./reviews/deep_21.md:4291:   287	algorithm’s performance (with Rank-1 and mAP stabilizing).
./reviews/deep_21.md:4362:    34	Fig. 5. Influence of different hyper-parameters η1 , η2 on Market-1501 and DukeMTMC-reID datasets. The left y-axis denotes the range of Rank-1, and the
./reviews/deep_21.md:4379:    51	label noise in the early training stages, enhancing the overall
./reviews/deep_21.md:4408:    80	Fig. 6. Influence of different balancing factors λ1 , λ2 and λ3 on Market-1501 and DukeMTMC-reID datasets. The left y-axis denotes the range of Rank-1,
./reviews/deep_21.md:4415:    87	DukeMTMC-reID datasets. The left y-axis denotes the range of Rank-1, and
./reviews/deep_21.md:4422:    94	when λ2 is set to 0.1, the mAP increases, but the Rank-1
./reviews/deep_21.md:4482:   154	mAP of 87.7% and Rank-1 of 94.7% on Market-1501. Increasing the number of parts does not further improve performance
./reviews/deep_21.md:4485:   157	7) The time and space complexity: The Table VIII summarizes the training time and model complexity for the
./reviews/deep_21.md:4491:   163	exhibits higher computational complexity, with 8.16 MMac
./reviews/deep_21.md:4495:   167	Compared to the baseline, the increase in model complexity
./reviews/deep_21.md:4504:   176	both mAP and Rank-1 are achieved compared to the baseline
./reviews/deep_21.md:4515:   187	worse than the one using DBSCAN, with both mAP and Rank1 showing large declines. This suggests that the pseudo-labels
./reviews/deep_21.md:4710:   135	Rank-1 Rank-5 Rank-10
./reviews/deep_21.md:4861:    55	method outperforms these methods in terms of Rank-1 and
./reviews/deep_21.md:4865:    59	Rank-1 on Market-1501, and by 9.6% in mAP and 8.6% in
./reviews/deep_21.md:4866:    60	Rank-1 on MSMT17. Additionally, our method shows gains
./reviews/deep_21.md:4881:    75	include at least 0.7% mAP and 0.3% Rank-1 on Market-1501,
./reviews/deep_21.md:4882:    76	9.7% mAP and 10.7% Rank-1 on MSMT17, 3.2% mAP and
./reviews/deep_21.md:4883:    77	1.7% Rank-1 on PersonX, and 2.7% mAP and 2.3% Rank-1 on
./reviews/deep_21.md:4899:    93	label noise through an anti-noise similarity ranking, refining
./reviews/deep_21.md:4937:   131	with hybrid memory for domain adaptive object re-ID,” in Proc. NIPS,
./reviews/deep_21.md:4958:   152	Exemplar memory for domain adaptive person re-identification,” in
./reviews/deep_21.md:4962:   156	domain adaptive person re-identification,” in Proc. IEEE/CVF Conf.
./reviews/deep_21.md:4971:   165	intermediate domain module for domain adaptive person re-ID,” in Proc.
./reviews/deep_21.md:5002:   196	[25] M. Bilal, A. Khan, M. U. Karim Khan, and C.-M. Kyung, “A lowcomplexity pedestrian detection framework for smart video surveillance
./reviews/deep_21.md:5014:   208	[29] J. Li and S. Zhang, “Joint visual and temporal consistency for unsupervised domain adaptive person re-identification,” in Proc. Eur. Conf.
./reviews/deep_21.md:5039:   233	for domain adaptive person re-identification,” in Proc. AAAI Conf. Artif.
./reviews/deep_21.md:5094:   288	[54] Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking person reidentification with K-reciprocal encoding,” in Proc. IEEE Conf. Comput.
./reviews/deep_21.md:5189:    19	algorithm’s performance (with Rank-1 and mAP stabilizing).
./reviews/deep_21.md:5288:   118	Fig. 16. The Rank-5 performance of our method under different occlusion
./reviews/deep_21.md:5300:   130	pedestrian ranking results under heavy occlusion conditions
./reviews/deep_21.md:5301:   131	in Fig. 16 (Rank-5, green boxes indicate correct results, and
./reviews/deep_21.md:5304:   134	occlusion issues relatively well, and the Rank-1 results are
./reviews/deep_21.md:5332:   162	method exhibits superior complexity control in model structure
./reviews/deep_21.md:5344:   174	G. Comparisons of Model Complexity
./reviews/deep_21.md:5346:   176	method, we conduct an analysis of the model complexity and
./reviews/deep_21.md:5360:   190	avoiding unnecessary complexity. Notably, the SAM module
./reviews/deep_21.md:5364:   194	while reducing computational complexity. When utilizing a
./reviews/deep_21.md:5368:   198	both Rank-1 and mAP metrics, particularly on the OccludedDuke dataset, where it achieves a Rank-1 of 73.4% and
./reviews/deep_21.md:5437:   267	[13] H. Zheng et al., “Cascade transformer reasoning embedded by uncertainty for occluded person re-identification,” IEEE Trans. Biometrics,
./reviews/deep_21.md:5439:   269	[14] G. Zhang, Y. Yang, Y. Zheng, G. Martin, and R. Wang, “Maskaware hierarchical aggregation transformer for occluded person reidentification,” IEEE Trans. Circuits Syst. Video Technol., early access,
./reviews/deep_21.md:5449:   279	tokens and distance-aware recovery network for occluded person reidentification,” IEEE Trans. Circuits Syst. Video Technol., early access,
./reviews/deep_21.md:5480:   310	[28] Y. Lu, M. Jiang, Z. Liu, and X. Mu, “Dual-branch adaptive attention
./reviews/deep_21.md:5597:    57	while inter-identity clusters become more clearly separated. In
./reviews/deep_21.md:5961:   307	[7] H. Wang, G. Wang, Y. Li, D. Zhang, L. Lin, Transferable, controllable, and inconspicuous adversarial attacks on person re-identiﬁcation with deep mis-ranking, in:
./reviews/deep_21.md:6012:    20	0.4% in Rank-1 on Market-1501, 2.6% in mAP and 1.8% in
./reviews/deep_21.md:6013:    21	Rank-1 on MSMT17, and 1.0% in mAP and 0.7% in Rank1 on VeRi-776. It is worth noting that FNPM is designed to
./reviews/deep_21.md:6016:    24	mining leads to training instability and performance degradation, hence its independent evaluation is not considered.
./reviews/deep_21.md:6019:    27	the query-based descending ranking list. To further verify the
./reviews/deep_21.md:6020:    28	effectiveness of excluding the Pcor and anti-noise ranking
./reviews/deep_21.md:6036:    44	and model complexity during inference (Table V). As shown
./reviews/deep_21.md:6039:    47	HCPM, FNPM, and PKGC successively leads to incremental improvements in mAP and Rank-1 accuracy, the total
./reviews/deep_21.md:6044:    52	complexity for models with different attention-based backbone
./reviews/deep_21.md:6045:    53	variants. Since HCPM and FNPM do not alter the backbone structure, they do not impact computational complexity
./reviews/deep_21.md:6062:    70	modules, we present a qualitative comparison of Rank-10
./reviews/deep_21.md:6138:    15	process for final ranking. Our study, validated on synthetic datasets Market-IA and Duke-IA, demonstrates
./reviews/deep_21.md:6245:   122	be used to improve ranking and retrieval accuracy.
./reviews/deep_21.md:6261:   138	matrix for ranking. We evaluated our proposed method on two synthetic datasets, Market-1501-IA
./reviews/deep_21.md:6500:    22	aggregates tokens into coherent regions and adaptively transfers complementary cues across modalities. To ensure semantic
./reviews/deep_21.md:6616:   138	This asymmetry significantly increases the difficulty of crossmodality feature alignment.
./reviews/deep_21.md:6648:   170	regions and performs adaptive cross-modality completion,
./reviews/deep_21.md:6664:   186	adaptive cross-modality feature completion, effectively alleviating spatial misalignment and modality-specific degradation.
./reviews/deep_21.md:6768:    22	pseudo labels by prioritizing similarity rankings over clustering
./reviews/deep_21.md:6893:   147	pseudo labels by introducing an anti-noise similarity ranking
./reviews/deep_21.md:7037:   291	SEA [47] enhances specific feature activations by adaptively reweighting channel weights based on their importance.
./reviews/deep_21.md:7171:   121	instability issues caused by small batch sizes.
./reviews/deep_21.md:7178:   128	evaluations and found that nearly all of related works primarily
./reviews/deep_21.md:7205:   155	strengthens early-stage robustness by employing an external
./reviews/deep_21.md:7248:   198	features are divided and fused using adaptive weights.
./reviews/deep_21.md:7387:   337	Fig. 1. CF Module: By employing adaptive weights to fuse Ftop with Fbot ,
./reviews/deep_21.md:7641:   247	adaptively learn and align local features. However, despite the
./reviews/deep_21.md:7687:2. **重定义动作**：他们把问题从“如何生成更好伪标签”改成“如何选择最有训练价值的 proxy 对”。关键词是 `optimal proxy mining`、`hardest camera proxy`、`false negative proxies mining`、`misaligned contrastive pairs`、`camera-constrained hard negative mining`、`anti-noise ranking`。这让创新点落在训练信号选择，而不是聚类算法本身。
./reviews/deep_21.md:7691:4. **证据闭环**：关键消融是正样本选择策略对比：corresponding proxy、all proxies、hardest positive proxy，以及是否使用 CCHNM。FNPM 部分又验证了排除同相机 proxy 和 anti-noise ranking 的作用。可视化包括 t-SNE、Top-10 检索和 Grad-CAM，用来证明不是只涨点，而是类内更紧、类间更开、注意力更落在人身上。
./reviews/deep_21.md:7768:2. **重定义动作**：他们把问题从“如何生成更好伪标签”改成“如何选择最有训练价值的 proxy 对”。关键词是 `optimal proxy mining`、`hardest camera proxy`、`false negative proxies mining`、`misaligned contrastive pairs`、`camera-constrained hard negative mining`、`anti-noise ranking`。这让创新点落在训练信号选择，而不是聚类算法本身。
./reviews/deep_21.md:7772:4. **证据闭环**：关键消融是正样本选择策略对比：corresponding proxy、all proxies、hardest positive proxy，以及是否使用 CCHNM。FNPM 部分又验证了排除同相机 proxy 和 anti-noise ranking 的作用。可视化包括 t-SNE、Top-10 检索和 Grad-CAM，用来证明不是只涨点，而是类内更紧、类间更开、注意力更落在人身上。
./reviews/deep_1.md:29:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_1.md:119:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_1.md:159:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_1.md:165:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_1.md:179:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_1.md:259:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_1.md:267:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_1.md:275:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_1.md:682:/bin/zsh -lc 'pdftotext -f 1 -l 8 "4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_1.md:693:817:47.6% and 46.9%, and their corresponding improvements can reach 10.0% (mAP) and 22.5% (rank-1).
./reviews/deep_1.md:937:1026:84.96% in Rank-1 and 86.85% in mAP. On the RegDB dataset, 4SNet
./reviews/deep_1.md:1040:814:93.1% Rank-1 accuracy, outperforming all the mentioned
./reviews/deep_1.md:1409:1198:85.5 % and a Rank-1 score of 86.0 % on CUHK 03-Labeled labeled by a
./reviews/deep_1.md:2274:It achieves this by adaptively fusing the pillar-based features
./reviews/deep_1.md:2284:CMF module adaptively determines relevant regions for fusion
./reviews/deep_1.md:2692:model to adaptively select where and what features should
./reviews/deep_1.md:2723:reaches cross modality alignment with adaptive fusion. The
./reviews/deep_1.md:2866:/bin/zsh -lc "pdftotext -f 1 -l 8 \"4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf\" - | sed -n '35,570p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_1.md:2873:Nevertheless, how to adaptively capture modality-specific frequency characteristics and effectively integrate
./reviews/deep_1.md:2875:presents a novel and efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet) for the VI-ReID
./reviews/deep_1.md:2896:have exacerbated the difficulty and challenges of this modeling task,
./reviews/deep_1.md:2918:the complexity of their underlying learning algorithms adds to the
./reviews/deep_1.md:2961:early-stage feature extraction, which justifies the use of AFF for input-adaptive
./reviews/deep_1.md:2977:efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet),
./reviews/deep_1.md:2991:using spatial attention. These bases are then adaptively combined
./reviews/deep_1.md:3040:to perform modality-specific and input-adaptive frequency filtering,
./reviews/deep_1.md:3061:modality to lower the difficulty of modal recognition. CSDN [29] and
./reviews/deep_1.md:3115:frequency-based methods by performing modality-specific and inputadaptive filtering at the early feature extraction stage. In this way, the
./reviews/deep_1.md:3193:Dynamic Kernel Synthesis. The AFF module synthesizes an adaptive
./reviews/deep_1.md:3210:to adaptively generate both the channel reweighting vector 𝐰𝑐 and the
./reviews/deep_1.md:3341:network to adaptively emphasize the most discriminative spectrum
./reviews/deep_1.md:3410:/bin/zsh -lc "pdftotext \"4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf\" - | sed -n '1320,1435p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_1.md:3412:Rank-1
./reviews/deep_1.md:3462:results clearly demonstrate the effectiveness of the proposed modules.
./reviews/deep_1.md:3498:Rank-1
./reviews/deep_1.md:3502:Rank-1
./reviews/deep_1.md:3838:1410:complexity of the knowledge module is higher than that of the reasoning module because the knowledge module has more layers [See Section 4.5.3 and 4.5.4]. We will discuss this drawbacks in Section 4.6.
./reviews/deep_1.md:3895:1663-blocks to 4 in our ﬁnal model. Besides, the computational complexity of
./reviews/deep_1.md:3923:/bin/zsh -lc "pdftotext \"4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf\" - | rg -n -C 2 \"visual|Visualization|t-SNE|frequency|band|Fig\\. 6|Fig\\. 7|attention\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_1.md:3928:40:Nevertheless, how to adaptively capture modality-specific frequency characteristics and effectively integrate
./reviews/deep_1.md:3930:42-presents a novel and efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet) for the VI-ReID
./reviews/deep_1.md:3974:128-early-stage feature extraction, which justifies the use of AFF for input-adaptive
./reviews/deep_1.md:4001:158:using spatial attention. These bases are then adaptively combined
./reviews/deep_1.md:4043:207:to perform modality-specific and input-adaptive frequency filtering,
./reviews/deep_1.md:4068:282:frequency-based methods by performing modality-specific and inputadaptive filtering at the early feature extraction stage. In this way, the
./reviews/deep_1.md:4114:360-Dynamic Kernel Synthesis. The AFF module synthesizes an adaptive
./reviews/deep_1.md:4131:377-to adaptively generate both the channel reweighting vector 𝐰𝑐 and the
./reviews/deep_1.md:4208:508-network to adaptively emphasize the most discriminative spectrum
./reviews/deep_1.md:4295:1029-Infrared mode, and 94.31% Rank-1 and 90.61% mAP in the Infrared to
./reviews/deep_1.md:4307:1298-RegDB and competitive Rank-10 performance on LLCM, it is slightly
./reviews/deep_1.md:4308:1299-lower than some recent methods on certain Rank-1 metrics. One possible reason, as supported by the ablation study in Table 5, is that SSM
./reviews/deep_1.md:4312:1303-finer-grained local detail modeling may achieve slightly better Rank-1
./reviews/deep_1.md:4329:1370-results clearly demonstrate the effectiveness of the proposed modules.
./reviews/deep_1.md:4401:1504:bounding boxes, while incorrect ones are marked in red. The visualization clearly underscores the superiority of our approach. The
./reviews/deep_1.md:4402:1505-baseline model frequently fails to rank the correct match first and
./reviews/deep_1.md:4430:1570-0.60 improves Rank-1 accuracy and mAP, suggesting that extending
./reviews/deep_1.md:4436:1576-This paper has presented 4SNet, a Spatial and Spectrum Selfadaptive Synergy Network, which addresses the VI-ReID task from
./reviews/deep_1.md:4473:1866:[40] W. Sun, et al., Visible-infrared person re-identification via adaptive frequency
./reviews/deep_1.md:4600:595-model to adaptively select where and what features should
./reviews/deep_1.md:4625:626-reaches cross modality alignment with adaptive fusion. The
./reviews/deep_1.md:4749:814-93.1% Rank-1 accuracy, outperforming all the mentioned
./reviews/deep_1.md:4753:818-Rank-1. Besides, SDCL [28] exploit event data to enhance
./reviews/deep_1.md:4757:822-mAP and Rank-1 accuracy, respectively.
./reviews/deep_1.md:4760:825-on Rank-5 accuracy when compare to the HASI [27] and
./reviews/deep_1.md:4763:829-methods leveraging other modality information on Rank-1
./reviews/deep_1.md:4767:833-MFA [79] reach the best Rank-1 accuracy of 96.6% and
./reviews/deep_1.md:4775:841-Rank-1 accuracy among existing SOTA methods. Specifically,
./reviews/deep_1.md:4777:843-of 0.2% in mAP and 0.3% in Rank-1 accuracy, respectively.
./reviews/deep_1.md:4808:882-in Table III, compared with Index-1 and Index-2, Index2 obtains 0.7% and 0.9% increases on mAP and Rank-1
./reviews/deep_1.md:4812:886:Rank-1 accuracy. It demonstrates that as a new and discriminative pedestrian characteristic extracted from SSTI module,
./reviews/deep_1.md:4817:891-Rank-1 accuracy, respectively, which shows the significance of
./reviews/deep_1.md:4873:951-direct concatenation by 2.1% mAP and 2.2% Rank-1 accuracy,
./reviews/deep_1.md:4875:953-concatenation by 1.3% mAP and 1.5% Rank-1 accuracy.
./reviews/deep_1.md:4877:955-Rank-1 accuracy gains compare to cross-attention mechanism
./reviews/deep_1.md:4880:958:2.3% Rank-1 accuracy. This shows that while using crossattention mechanism can improve the performance, the CMF
./reviews/deep_1.md:4887:967-+0.6% mAP and +0.4% Rank-1, +0.3% mAP and +0.2%
./reviews/deep_1.md:4888:968-Rank-1 and +0.2% mAP and +0.3% Rank-1 on three strategies (the model without fusion, the cross-attention and the
./reviews/deep_1.md:4907:987-on mAP and increase 0.1% on Rank-1 accuracy with only
./reviews/deep_1.md:4911:991-module by 0.3% mAP and 0.3% Rank-1 accuracy, but exhibits
./reviews/deep_1.md:4912:992-a reduction of 0.3% mAP and 0.8% Rank-1 accuracy compare
./reviews/deep_1.md:4937:1019:SSTI module by 0.4% mAP and 0.8% Rank-1 accuracy, and
./reviews/deep_1.md:4938:1020-0.6% mAP and 1.1% Rank-1 accuracy, respectively, which
./reviews/deep_1.md:4993:1088:from Rank-1 to Rank-10 by leveraging the CMF module to
./reviews/deep_1.md:5065:1164:a Cross-modal Fusion (CMF) module is designed to adaptively
./reviews/deep_1.md:5140:1394:[65] Y. Rong, T. Shiratori, and H. Joo, “FrankMocap: A monocular 3D
./reviews/deep_1.md:5195:in Table III, compared with Index-1 and Index-2, Index2 obtains 0.7% and 0.9% increases on mAP and Rank-1
./reviews/deep_1.md:5199:Rank-1 accuracy. It demonstrates that as a new and discriminative pedestrian characteristic extracted from SSTI module,
./reviews/deep_1.md:5204:Rank-1 accuracy, respectively, which shows the significance of
./reviews/deep_1.md:5212:2) Computation Complexity Analysis: We analyse model
./reviews/deep_1.md:5213:efficiency using parameter count (space complexity) and
./reviews/deep_1.md:5214:FLOPs (time complexity), common in video Re-ID. Note
./reviews/deep_1.md:5264:direct concatenation by 2.1% mAP and 2.2% Rank-1 accuracy,
./reviews/deep_1.md:5266:concatenation by 1.3% mAP and 1.5% Rank-1 accuracy.
./reviews/deep_1.md:5268:Rank-1 accuracy gains compare to cross-attention mechanism
./reviews/deep_1.md:5271:2.3% Rank-1 accuracy. This shows that while using crossattention mechanism can improve the performance, the CMF
./reviews/deep_1.md:5280:+0.6% mAP and +0.4% Rank-1, +0.3% mAP and +0.2%
./reviews/deep_1.md:5281:Rank-1 and +0.2% mAP and +0.3% Rank-1 on three strategies (the model without fusion, the cross-attention and the
./reviews/deep_1.md:5300:on mAP and increase 0.1% on Rank-1 accuracy with only
./reviews/deep_1.md:5304:module by 0.3% mAP and 0.3% Rank-1 accuracy, but exhibits
./reviews/deep_1.md:5305:a reduction of 0.3% mAP and 0.8% Rank-1 accuracy compare
./reviews/deep_1.md:5332:SSTI module by 0.4% mAP and 0.8% Rank-1 accuracy, and
./reviews/deep_1.md:5333:0.6% mAP and 1.1% Rank-1 accuracy, respectively, which
./reviews/deep_1.md:5383:[29] P. Wu, L. Wang, S. Zhou, G. Hua, and C. Sun, “Temporal correlation            [52] C.-T. Liu, C.-W. Wu, Y.-C. Frank Wang, and S.-Y. Chien, “Spatially and
./reviews/deep_1.md:5441:[65] Y. Rong, T. Shiratori, and H. Joo, “FrankMocap: A monocular 3D                    A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput.
./reviews/deep_1.md:5537:reaches cross modality alignment with adaptive fusion. The                     expressed as:
./reviews/deep_1.md:5609:                                                                              93.1% Rank-1 accuracy, outperforming all the mentioned
./reviews/deep_1.md:5619:                                                                              Rank-1. Besides, SDCL [28] exploit event data to enhance
./reviews/deep_1.md:5627:                                                                              mAP and Rank-1 accuracy, respectively.
./reviews/deep_1.md:5635:                                                                              on Rank-5 accuracy when compare to the HASI [27] and
./reviews/deep_1.md:5639:                                                                              method attains 94.7% Rank-1 accuracy which corresponds
./reviews/deep_1.md:5643:                                                                              methods leveraging other modality information on Rank-1
./reviews/deep_1.md:5648:   Our method is implemented based on PyTorch platform                        MFA [79] reach the best Rank-1 accuracy of 96.6% and
./reviews/deep_1.md:5656:cloud followed by [68]. For the Spatial-Temporal Aggregation                  Rank-1 accuracy among existing SOTA methods. Specifically,
./reviews/deep_1.md:5658:pyramid architecture as proposed in [21], leveraging its ability              of 0.2% in mAP and 0.3% in Rank-1 accuracy, respectively.
./reviews/deep_1.md:5677:competitive performance compare to other SOTA methods.                        Rank-1 accuracy. It demonstrates that as a new and discrim-
./reviews/deep_1.md:5683:trian information in image while preserving 3D information.                   Rank-1 accuracy, respectively, which shows the significance of
./reviews/deep_1.md:5692:when compared to the other multi-modal technique [30], our                       2) Computation Complexity Analysis: We analyse model
./reviews/deep_1.md:5693:method exhibits advantages in both model parameter and                        efficiency using parameter count (space complexity) and
./reviews/deep_1.md:5694:overall performance.                                                          FLOPs (time complexity), common in video Re-ID. Note
./reviews/deep_1.md:5705:2 obtains 0.7% and 0.9% increases on mAP and Rank-1                           parameters and 4.8G FLOPs compared to the baseline, it
./reviews/deep_1.md:5786:precision (mAP) (rank-1), respectively. When compared with the
./reviews/deep_1.md:5789:the PRCC, LTCC, Celeb, and NKUP datasets in terms of rank-1
./reviews/deep_1.md:5824:acquisition period becomes longer, the complexity of clothing
./reviews/deep_1.md:5918:mean average precision (mAP) and rank-1.
./reviews/deep_1.md:5984:they proposed an adaptive threshold-guided masked graph
./reviews/deep_1.md:6023:latter was utilized to adaptively distill the ID-relevant shape
./reviews/deep_1.md:6482:        47.6% and 46.9%, and their corresponding improve-                                     its rank-1 can obtain the first place. The reason for
./reviews/deep_1.md:6483:        ments can reach 10.0% (mAP) and 22.5% (rank-1).                                       this is that these specially designed methods mainly
./reviews/deep_1.md:6484:        Similarly, the mAP and rank-1 accuracies of the SAVS                                  focus on body shape or contour sketch, and the complex
./reviews/deep_1.md:6486:        respectively, but the corresponding mAP and rank-1 of                                 fully explored or are ignored, but in the SAVS, the
./reviews/deep_1.md:6489:        (rank-1). Thus, the SAVS can significantly outperform                                 semantic information, and the negative influences of the
./reviews/deep_1.md:6504:        dataset, where the mAP and rank-1 can reach 15.8%                                     we will compare SAVS with the MGN and HOR-
./reviews/deep_1.md:6507:        (mAP) and 2.9% (rank-1), respectively. Similarly, when                                improvements of 17.4% (mAP) and 24.9% (rank-1)
./reviews/deep_1.md:6509:        used, the mAP and rank-1 on the LTCC dataset are                                      dataset, SAVS exhibits a 10.5% improvement in mAP
./reviews/deep_1.md:6510:        16.1% and 42.3%, and the improvements of the SAVS                                     and a 16.9% improvement in rank-1 compared to MGN.
./reviews/deep_1.md:6545:     dataset is used, the rank-1 accuracies of ResNet50,
./reviews/deep_1.md:6552:     11.1%, and 21.3%, respectively, and the corresponding                            datasets are utilized, and the evaluation metrics of (a) mAP and (b) rank-1
./reviews/deep_1.md:6576:A. Effectiveness of the HSA Module                                                          and rank-1 accuracies of the original image are 15.2%
./reviews/deep_1.md:6583:the SAVS, in our experiments, it is also used as the baseline,                              Similarly, when the Celeb-reID dataset is used, the rank-
./reviews/deep_1.md:6614:        used, the rank-1 accuracies of “O + F,” “O,” and “F”
./reviews/deep_1.md:6627:        and NKUP datasets are used, the rank-1 accuracies of
./reviews/deep_1.md:6629:        datasets, the rank-1 accuracies of “O + B” can obtain
./reviews/deep_1.md:6634:        information is very helpful for feature extraction. In                                 5.4% (mAP) and 6.8% (rank-1). On the PRCC dataset,
./reviews/deep_1.md:6635:        addition, the background information is somewhat useful                                the rank-1 accuracies of “+HSA + VCS” and “+HSA”
./reviews/deep_1.md:6656:       reID dataset is used, the mAP and rank-1 accuracies                                     mAP (rank-1) accuracies of the baseline, “+HSA,” and
./reviews/deep_1.md:6658:       mAP and rank-1 accuracies of “+HSA” are 15.9%                                           and 32.5% (71.2%), respectively, whose performance
./reviews/deep_1.md:6694:     mAP (rank-1) accuracies of the “ResNet50 + VCS” and
./reviews/deep_1.md:6712:     LTCC dataset, the mAP (rank-1) accuracies of “Swin-                              tative visualization of the retrieval results. In the following,
./reviews/deep_1.md:6763:72-the PRCC, LTCC, Celeb, and NKUP datasets in terms of rank-1
./reviews/deep_1.md:6776:107-acquisition period becomes longer, the complexity of clothing
./reviews/deep_1.md:7185:823-(rank-1). Thus, the SAVS can significantly outperform
./reviews/deep_1.md:7194:848-its rank-1 can obtain the first place. The reason for
./reviews/deep_1.md:7245:947-datasets are utilized, and the evaluation metrics of (a) mAP and (b) rank-1
./reviews/deep_1.md:7276:1014-and NKUP datasets are used, the rank-1 accuracies of
./reviews/deep_1.md:7278:1016-datasets, the rank-1 accuracies of “O + B” can obtain
./reviews/deep_1.md:7300:1038:of the “+HSA” module. For example, when the CelebreID dataset is used, the mAP and rank-1 accuracies
./reviews/deep_1.md:7302:1040:mAP and rank-1 accuracies of “+HSA” are 15.9%
./reviews/deep_1.md:7315:1053-5.4% (mAP) and 6.8% (rank-1). On the PRCC dataset,
./reviews/deep_1.md:7316:1054:the rank-1 accuracies of “+HSA + VCS” and “+HSA”
./reviews/deep_1.md:7336:1074:mAP (rank-1) accuracies of the baseline, “+HSA,” and
./reviews/deep_1.md:7367:1106:mAP (rank-1) accuracies of the “ResNet50 + VCS” and
./reviews/deep_1.md:7383:1122:LTCC dataset, the mAP (rank-1) accuracies of “SwinT + VCS” and Pixel Sampling are 22.8% (54.9%)
./reviews/deep_1.md:7518:1276:since the visual appearance of the cloth-changing person drastically changes, the baseline still has difficulty
./reviews/deep_1.md:7561:1328-and rank-1 accuracy, and more discriminative and robust
./reviews/deep_1.md:7731:extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
./reviews/deep_1.md:7904:Importance ranking set of surrounding area
./reviews/deep_1.md:7929:placement and fusion process using the semantic diﬀerence map. To illustrate this process more clearly, we provide intermediate examples for each sub-stage in the
./reviews/deep_1.md:8308:makes the samples belonging to diﬀerent identities are pushed further                                                                     mAp    Rank-1     mAp       Rank-1
./reviews/deep_1.md:8346:form the test set, which contains 702 diﬀerent pedestrian identities.                                                              mAp        Rank-1      mAp       Rank-1
./reviews/deep_1.md:8395:                                                                                                                                   mAP      Rank-1
./reviews/deep_1.md:8428:ods [12]. Given the structured and semantically guided nature of our               petitive recognition rate of 96.9 % Rank-1 accuracy and 91.8 % mAP
./reviews/deep_1.md:8430:ducing eﬀective interference and preserving identity-discriminative fea-           much more diﬃcult, our FOSENet still achieves 90.8 % Rank-1 accuracy
./reviews/deep_1.md:8437:ket1501, CUHK 03, DukeMTMCreID and Occluded-DukeMTMC, and                          ENet achieves a mAP score of 82.3 % and a Rank-1 score of 85.0 % on
./reviews/deep_1.md:8439:sure a fair comparison and eliminate confounding factors, our FOSENet              85.5 % and a Rank-1 score of 86.0 % on CUHK 03-Labeled labeled by a
./reviews/deep_1.md:8444:techniques (e.g., random rotation and random cropping), and most of                and achieves a mAP score of 61.5 % and a Rank-1 score of 68.7 %. Al-
./reviews/deep_1.md:8445:the compared methods, including ours, are based on CNN architectures.              though FOSENet can not achieve a best Rank-1 score, the mAP score
./reviews/deep_1.md:8468:                 Ablation study on diﬀerent modules of the proposed                            Mean and standard deviation of CMC Rank-1 ac-
./reviews/deep_1.md:8473:                  Model                                                                          Dataset                 mAP (%)       Rank-1 (%)
./reviews/deep_1.md:8474:                                                   mAP     Rank-1
./reviews/deep_1.md:8492:proposed method. In Table 5, the Baseline denotes the underlying back-             achieves 87.2 % mAP accuracy and 95.0 % Rank-1, which signiﬁ-
./reviews/deep_1.md:8507:mAP accuracy and 95.5 % Rank-1 accuracy can be reached. Finally, by                   ferent distances and widening the interclass distances, which helps the
./reviews/deep_1.md:8508:using all components, FOSENet can obtain 96.9 % Rank-1 accuracy and                   model to extract the features eﬃciently.
./reviews/deep_1.md:8513:Rank-1 accuracy on four benchmark datasets and the standard devia-
./reviews/deep_1.md:8541:vide the recognition accuracy data (mAP,Rank-1∼10) of FOSENet in 120
./reviews/deep_1.md:8545:rapidly in the early part of the training epoch. The accuracy stabilised
./reviews/deep_1.md:8561:formed Rank-1 and mAP evaluations on the Market-1501 and Occluded-                    5. Conclusion
./reviews/deep_1.md:8563:It is observed that mAP and Rank-1 reach the best when the parame-                        In this paper, a semantic-guided occlusion simulation based local
./reviews/deep_1.md:8591:computational cost and training complexity due to their multi-stage                                    226–235.
./reviews/deep_1.md:8600:lightweight and adaptive occlusion simulation strategies and domain                                    ceedings of the IEEE/CVF International Conference on Computer Vision, 2021,
./reviews/deep_1.md:8767:127:extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
./reviews/deep_1.md:8913:300-Importance ranking set of surrounding area
./reviews/deep_1.md:8933:325:placement and fusion process using the semantic diﬀerence map. To illustrate this process more clearly, we provide intermediate examples for each sub-stage in the
./reviews/deep_1.md:9117:1198-85.5 % and a Rank-1 score of 86.0 % on CUHK 03-Labeled labeled by a
./reviews/deep_1.md:9122:1203-and achieves a mAP score of 61.5 % and a Rank-1 score of 68.7 %. Although FOSENet can not achieve a best Rank-1 score, the mAP score
./reviews/deep_1.md:9147:1239:Mean and standard deviation of CMC Rank-1 accuracy and mAP (%) across ﬁve independent trials on four public Re-ID datasets (Market-1501,
./reviews/deep_1.md:9175:1291:achieves 87.2 % mAP accuracy and 95.0 % Rank-1, which signiﬁcantly outperforms the baseline by incorporating the semantic based
./reviews/deep_1.md:9188:1309-mAP accuracy and 95.5 % Rank-1 accuracy can be reached. Finally, by
./reviews/deep_1.md:9189:1310:using all components, FOSENet can obtain 96.9 % Rank-1 accuracy and
./reviews/deep_1.md:9192:1313-Rank-1 accuracy on four benchmark datasets and the standard deviations are within ±0.3. The experiment results indicate that the proposed
./reviews/deep_1.md:9200:1322-4) Convergence of Accuracy Over Epochs. As shown in Fig. 6 . We provide the recognition accuracy data (mAP,Rank-1∼10) of FOSENet in 120
./reviews/deep_1.md:9202:1333-It is observed that mAP and Rank-1 reach the best when the parameter 𝜆 reaches 0.8, and then show a diﬀerent degree of decline on both
./reviews/deep_1.md:9235:1368-computational cost and training complexity due to their multi-stage
./reviews/deep_1.md:9241:1374:lightweight and adaptive occlusion simulation strategies and domain
./reviews/deep_1.md:9463:timeline for the best view). (b) Parameter sizes (Millions (M)), computational complexity (Giga Floating Point Operations (GFLOPs)), and
./reviews/deep_1.md:9464:KS20 Rank-1 accuracy of state-of-the-art deep learning methods for SRID (Red: Sequence-based models; Green: Graph-based models).
./reviews/deep_1.md:9551:Rank-1
./reviews/deep_1.md:9881:skeleton mining mechanism to adaptively infer importance of
./reviews/deep_1.md:10026:Generally low complexity.
./reviews/deep_1.md:10343:172-KS20 Rank-1 accuracy of state-of-the-art deep learning methods for SRID (Red: Sequence-based models; Green: Graph-based models).
./reviews/deep_1.md:10461:1191-Evaluation Metrics and Protocols. In SRID, the performance is typically evaluated based on several mertics, including Cumulative Matching Characteristics (CMC), Rank-1 accuracy (R1 ), Rank-5 accuracy, Rank-10 accuracy, and Mean
./reviews/deep_1.md:10881:**3. 4SNet - Spatial and Spectrum Self-adaptive Synergy Network**
./reviews/deep_1.md:10885:2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。
./reviews/deep_1.md:10925:4. **证据闭环**：它用方法表、数据集表、参数量和 Rank-1 对比、benchmark/protocol 汇总、挑战清单和 3DGait 医疗任务 case study 来证明 SRID 不是零散技巧，而是有共同问题、共同评价和外溢应用的方向。
./reviews/deep_1.md:10962:**3. 4SNet - Spatial and Spectrum Self-adaptive Synergy Network**
./reviews/deep_1.md:10966:2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。
./reviews/deep_1.md:11006:4. **证据闭环**：它用方法表、数据集表、参数量和 Rank-1 对比、benchmark/protocol 汇总、挑战清单和 3DGait 医疗任务 case study 来证明 SRID 不是零散技巧，而是有共同问题、共同评价和外溢应用的方向。
./reviews/deep_2.md:60:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_2.md:108:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_2.md:116:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_2.md:180:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_2.md:209:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_2.md:215:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_2.md:313:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_2.md:450:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_2.md:525:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_2.md:664:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_2.md:693:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_2.md:696:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_2.md:697:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_2.md:774:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_2.md:778:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_2.md:792:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_2.md:798:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_2.md:934:architecture level. (3) Scalable Design, if necessary, SAAB block can be ﬂexibly cascaded to construct taskspeciﬁc ADA framework. Experiments on three benchmarks validate ADA’s superiority: (1) Achieves state-of-theart performance across domains (e.g., 16.5 % mAP gain on CUHK03→Market-1501). (2) Demonstrates consistent
./reviews/deep_2.md:952:complexity. (2) Image style translation suﬀers from unstable optimization due to its dependency on generative adversarial networks (GANs).
./reviews/deep_2.md:993:requiring auxiliary annotations from either domain, reducing implementation complexity. Second, it eliminates dependence on GAN-generated
./reviews/deep_2.md:1008:quadratic spatial complexity 𝑂((𝐻𝑊 )2 ) (where H and W denote feature map height and width) within supervised learning frameworks.
./reviews/deep_2.md:1010:self-attention with reduced spatial complexity, and (2) unsupervised
./reviews/deep_2.md:1033:annotations (e.g., attribute labels), increasing implementation complexity and data acquisition costs.
./reviews/deep_2.md:1039:constraints: (1) The training instability characteristic of adversarial
./reviews/deep_2.md:1060:complexity due to dense matrix operations within backbone network. In
./reviews/deep_2.md:1101:Sparse Attention Eﬃciency: Implements Sparse complexity attention(<
./reviews/deep_2.md:1190:#The low−rank m a t r i x a p p r o x i m a t e s
./reviews/deep_2.md:1285:Fig. 3. The diagram of the SAP: the SAP achieves architectural ﬂexibility, enhances feature representation , and optimizes the complexity-accuracy tradeoﬀ.
./reviews/deep_2.md:1374:• (2) Computational Complexity Optimization: (I) Linear SelfAttention implements low-rank matrix approximating self-attention
./reviews/deep_2.md:1377:(II) Linear variants utilize low-rank matrices for attention matrix estimation.
./reviews/deep_2.md:1385:• (3) Complexity-Accuracy Tradeoﬀ: Maintains high accuracy while
./reviews/deep_2.md:1403:Cumulative Matching Characteristics (CMC) Rank-1 accuracy. Our implementation strictly uses only source domain labels and target domain
./reviews/deep_2.md:1512:at the expense of increased architectural and computational complexity
./reviews/deep_2.md:2374:4.5.1. Evaluation of parameters and computational complexity
./reviews/deep_2.md:2375:We quantitatively evaluate the parameters and computational complexity of the ADA framework. The detailed data is shown in the
./reviews/deep_2.md:3027:Based on the aforementioned classification, we propose a multiscale cross-modal composite loss that integrates contrastive learning, hard example mining, and attribute-adaptive weight adjustment.
./reviews/deep_2.md:3030:importance [11,13], the proposed weighting scheme adaptively adjusts
./reviews/deep_2.md:3045:and 𝑤𝑖𝑛𝑡𝑟𝑎 , respectively. Among them, 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 are treated as hyperparameters, while 𝑤𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑖𝑛𝑡𝑟𝑎 are adaptively determined through
./reviews/deep_2.md:3196:grows approximately linearly as (𝑛), where 𝑛 is the number of attributes. Secondly, the loss functions are designed to be attribute-aware
./reviews/deep_2.md:3217:conditions, reducing complexity; (2) To align with common practices in
./reviews/deep_2.md:3662:+4.91% in Rank-1, Rank-5, Rank-10, and mAP over the baseline
./reviews/deep_2.md:3715:those outlined in IRRA [36]. We employ Rank-𝑘 (𝑘 = 1,5,10) and
./reviews/deep_2.md:4024:while slightly trailing the most recent work [50] in Rank-1, it excels
./reviews/deep_2.md:4027:improvement of +10.45% and +8.98% on Rank-1 and mAP. The
./reviews/deep_2.md:4063:T2I in Rank-1 accuracy: prototype similarity is robust in a global,
./reviews/deep_2.md:4072:from the global ranking stability of prototypes while retaining the
./reviews/deep_2.md:4082:1.1%/1.24% in Rank-1 accuracy and 3.72%/5.56% in mAP over the
./reviews/deep_2.md:4083:RaSa model, and 0.99%/1.31% in Rank-1 and 3.38%/3.9% in mAP over
./reviews/deep_2.md:4085:improvements in Rank-1 and 3.79%/5.77% in mAP over IRRA, and
./reviews/deep_2.md:4086:0.68%/0.76% improvements in Rank-1 and 4.83%/5.42% in mAP over
./reviews/deep_2.md:4100:Fig. 5. Rank-1 accuracy of IRRA on three datasets under three inference modes during training.
./reviews/deep_2.md:4285:RaSa-rerank
./reviews/deep_2.md:4289:MARS-rerank
./reviews/deep_2.md:4304:IRRA-rerank
./reviews/deep_2.md:4308:RDE-rerank
./reviews/deep_2.md:4312:HAM-rerank
./reviews/deep_2.md:4341:rank 4.02% mAP on CUHK-PEDES and 6.53% mAP on ICFG-PEDES.
./reviews/deep_2.md:4342:HAM-PDE𝑀 surpasses third rank 0.39% Rank-1 on CUHK-PEDES, and
./reviews/deep_2.md:4345:We further evaluate k-reciprocal re-ranking [34], a standard postprocessing method in image-based person ReID. As shown in Table 1,
./reviews/deep_2.md:4351:or diffusion-based re-ranking strategy, but an identity-level, prototypebased prior that uses global anchors derived from the image gallery
./reviews/deep_2.md:4568:Fig. 7. Effect of the scaling factor 𝜆 and cluster number 𝑁 on the Rank-1 performance of HAM-PDE across three datasets.
./reviews/deep_2.md:4580:As shown in Figs. 7 and 8, Rank-1 and mAP follow similar trends with
./reviews/deep_2.md:4581:respect to 𝑁. Although the 𝑁 that maximizes Rank-1 does not always
./reviews/deep_2.md:4584:The larger fluctuation of Rank-1 on RSTPReid is likely due to
./reviews/deep_2.md:4593:as a good trade-off between effectiveness and complexity.
./reviews/deep_2.md:4598:Fig. 10 shows that Rank-1 is more variable, but larger 𝛽 generally
./reviews/deep_2.md:4642:(e.g., handbags, backpacks, etc.) while increasing the complexity
./reviews/deep_2.md:4647:adaptively extract distinguishing features of body parts, APE
./reviews/deep_2.md:4651:by adaptively calculating the occlusion weights for body parts.
./reviews/deep_2.md:4659:81.2% Rank-1 accuracy.
./reviews/deep_2.md:4690:pose uncertainty, appearance changes, and scene complexity
./reviews/deep_2.md:4722:network, which increases the overall complexity of the model.
./reviews/deep_2.md:4758:occlusion-adaptive local features that accurately reflect the
./reviews/deep_2.md:4763:The AOANet enables feature adaptive representation of
./reviews/deep_2.md:4767:obtains body regions that are adaptive to occlusion by
./reviews/deep_2.md:4771:a lower complexity.
./reviews/deep_2.md:4773:which can adaptively calculate occlusion weights. These
./reviews/deep_2.md:4851:procedure for obtaining the four adaptive local features. Adaptive Position Extractor (APE) is developed to determine the
./reviews/deep_2.md:4940:extract adaptive part locations, which are trained supervised
./reviews/deep_2.md:5264:evaluations and do not use re-ranking [51] in our experiments.
./reviews/deep_2.md:5301:MSMT17, achieving 74.8% mAP and 89.8% Rank-1 score.
./reviews/deep_2.md:5305:mAP and 88.4% Rank-1 accuracy. This shows that our method
./reviews/deep_2.md:5308:with a mAP of 87.0% and a Rank-1 score of 93.6%, surpassing
./reviews/deep_2.md:5311:and a Rank-1 score of 97.1%. Although the mAP of NFormer
./reviews/deep_2.md:5312:on this dataset is comparable to ours, our Rank-1 score is
./reviews/deep_2.md:5317:adaptive representation design in AOANet, which performs
./reviews/deep_2.md:5323:achieves mAP of 70.6% and Rank-1 score of 81.2%. On the
./reviews/deep_2.md:5326:for mAP and 91.7% for Rank-1 score. These results demonstrate that our model exhibits a high degree of robustness, even
./reviews/deep_2.md:5329:adaptively capture key information even in severe occlusion.
./reviews/deep_2.md:5356:Rank-1. A comparable scenario is observed in the matching
./reviews/deep_2.md:5394:and Rank-1 score show notable improvements, increasing by
./reviews/deep_2.md:5396:and Rank-1 score increasing by 18.4% and 16.9%, respectively. The APE has the capability to extract adaptive body
./reviews/deep_2.md:5409:1.3% in rank-1 score. The improvement is more pronounced
./reviews/deep_2.md:5411:and rank-1 score by 2.9%. This notable enhancement can
./reviews/deep_2.md:5605:critical improvements: adaptive pseudo-label division,
./reviews/deep_2.md:5749:pseudo-labels and adaptively select cleaner data. To be specific, for the c-th cluster in modality t, represented by a
./reviews/deep_2.md:5872:To address this limitation, we propose an adaptive pseudo-label
./reviews/deep_2.md:6455:APPD achieves 64.7% in Rank-1 and 60.5% in mAP, which
./reviews/deep_2.md:6456:surpasses MMM by 3.1% in Rank-1 and 2.6% in mAP. For
./reviews/deep_2.md:6459:70.6% in Rank-1 and 75.3% in mAP, exceeding MMM by a
./reviews/deep_2.md:6460:large margin with 6.2% in Rank-1 and 4.9% in mAP. Morever,
./reviews/deep_2.md:6461:the performance on RegDB achieves 91.1% in Rank-1 and
./reviews/deep_2.md:6463:in Rank-1 and 82.8% in mAP under the Thermal2Visible
./reviews/deep_2.md:6466:a 1.2% increase in average Rank-1 accuracy and a 1.0% boost
./reviews/deep_2.md:6521:and reduce the computational complexity of processing high-dimensional data in the self-attention mechanism.
./reviews/deep_2.md:6524:cloth-changing datasets clearly confirm the superior performance of 𝐴3 𝑃 𝐹 𝑁. The dataset and related code are
./reviews/deep_2.md:6561:Fig. 1. Visualization of the Top-8 ranking lists generated by MGN [3] on the MSMT17 and Celeb-reID datasets. Images with red boxes indicate incorrect matches.
./reviews/deep_2.md:6570:clearly observed that there are significant differences in the focus of
./reviews/deep_2.md:6612:[16] adopts loop generation to adaptively balance environment and
./reviews/deep_2.md:6658:Multi-Biometric Unified Network (MBUNet), which applies adaptive
./reviews/deep_2.md:6773:Although the adaptive weighted average method is very effective in
./reviews/deep_2.md:6862:in [24], which can significantly reduce the computational complexity
./reviews/deep_2.md:7003:scenes. In contrast, real image datasets often have the complexity of
./reviews/deep_2.md:7018:Considering the complexity of precipitation levels and changes in
./reviews/deep_2.md:7031:function for each output feature to adaptive weights. Initially, the
./reviews/deep_2.md:7049:Rank-1 accuracy and mean precision (mAP) in three test scenarios: (1)
./reviews/deep_2.md:7081:Rank-1
./reviews/deep_2.md:7085:Rank-1
./reviews/deep_2.md:7196:increases Rank-1 and mAP in cross-clothes scenarios by 10.6% and
./reviews/deep_2.md:7200:increases Rank-1 and mAP in cross-clothes scenarios by 4.9% and
./reviews/deep_2.md:7215:identities, which greatly increases the complexity of the model. In
./reviews/deep_2.md:7218:in Rank-1 and a 0.1% improvement in mAP but also effectively reduces
./reviews/deep_2.md:7221:complexity.
./reviews/deep_2.md:7282:Rank-1
./reviews/deep_2.md:7286:Rank-1
./reviews/deep_2.md:7382:Rank-1
./reviews/deep_2.md:7436:Rank-1 and mAP by 3.6% and 5.2% respectively in the cross-clothes
./reviews/deep_2.md:7444:conditions, such as incorporating adaptive image restoration techniques
./reviews/deep_2.md:7449:method MBUNet [21], our method improves Rank-1 by 1.5% and
./reviews/deep_2.md:7464:method, AIM’s Rank-1 is 0.2% and 4.2% higher on PRCC and LTCC,
./reviews/deep_2.md:7477:Rank-1 accuracy shows an enhancement of 18.4% and 11.1%, and
./reviews/deep_2.md:7488:results. Specifically, ours is 4.5% and 4.4% lower than SAVS in Rank-1
./reviews/deep_2.md:7499:method SAVS [23], our method improves Rank-1 and mAP by 0.4% and
./reviews/deep_2.md:7524:Rank-1
./reviews/deep_2.md:7528:Rank-1
./reviews/deep_2.md:7618:Rank-1
./reviews/deep_2.md:7622:Rank-1
./reviews/deep_2.md:7626:Rank-1
./reviews/deep_2.md:7630:Rank-1
./reviews/deep_2.md:7786:Rank-1
./reviews/deep_2.md:7790:Rank-1
./reviews/deep_2.md:7794:Rank-1
./reviews/deep_2.md:7798:Rank-1
./reviews/deep_2.md:7802:Rank-1
./reviews/deep_2.md:7868:accuracies of Rank-1 and mAP have increased by 4.3% and 1.4%
./reviews/deep_2.md:7873:further improved, especially the Rank-1 on NKUP and Celeb-reID-light
./reviews/deep_2.md:7877:When MLDC and LPAM are used simultaneously, the model’s performance of all datasets is significantly improved, especially on CelebreID-light, where the accuracies of Rank-1 and mAP are increased by
./reviews/deep_2.md:7886:all datasets has been further improved, especially on the VC-ClothesW&R dataset, where Rank-1 and mAP are increased by 1.0% and
./reviews/deep_2.md:7896:these two modules are used in combination, its matching rate in Ranks
./reviews/deep_2.md:7985:Multi-Biometric Unified Network (MBUNet), which applies adaptive
./reviews/deep_2.md:8100:Although the adaptive weighted average method is very effective in
./reviews/deep_2.md:8189:in [24], which can significantly reduce the computational complexity
./reviews/deep_2.md:8330:scenes. In contrast, real image datasets often have the complexity of
./reviews/deep_2.md:8345:Considering the complexity of precipitation levels and changes in
./reviews/deep_2.md:8358:function for each output feature to adaptive weights. Initially, the
./reviews/deep_2.md:8376:Rank-1 accuracy and mean precision (mAP) in three test scenarios: (1)
./reviews/deep_2.md:8762:Based on the aforementioned classification, we propose a multiscale cross-modal composite loss that integrates contrastive learning, hard example mining, and attribute-adaptive weight adjustment.
./reviews/deep_2.md:8765:importance [11,13], the proposed weighting scheme adaptively adjusts
./reviews/deep_2.md:8780:and 𝑤𝑖𝑛𝑡𝑟𝑎 , respectively. Among them, 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 are treated as hyperparameters, while 𝑤𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑖𝑛𝑡𝑟𝑎 are adaptively determined through
./reviews/deep_2.md:8931:grows approximately linearly as (𝑛), where 𝑛 is the number of attributes. Secondly, the loss functions are designed to be attribute-aware
./reviews/deep_2.md:8952:conditions, reducing complexity; (2) To align with common practices in
./reviews/deep_2.md:9061:Sparse Attention Eﬃciency: Implements Sparse complexity attention(<
./reviews/deep_2.md:9150:#The low−rank m a t r i x a p p r o x i m a t e s
./reviews/deep_2.md:9245:Fig. 3. The diagram of the SAP: the SAP achieves architectural ﬂexibility, enhances feature representation , and optimizes the complexity-accuracy tradeoﬀ.
./reviews/deep_2.md:9334:• (2) Computational Complexity Optimization: (I) Linear SelfAttention implements low-rank matrix approximating self-attention
./reviews/deep_2.md:9337:(II) Linear variants utilize low-rank matrices for attention matrix estimation.
./reviews/deep_2.md:9345:• (3) Complexity-Accuracy Tradeoﬀ: Maintains high accuracy while
./reviews/deep_2.md:9363:Cumulative Matching Characteristics (CMC) Rank-1 accuracy. Our implementation strictly uses only source domain labels and target domain
./reviews/deep_2.md:9472:at the expense of increased architectural and computational complexity
./reviews/deep_2.md:9701:matching and stabilize rankings.
./reviews/deep_2.md:9708:with QG-Similarity to yield enhanced ranking scores. The cascaded
./reviews/deep_2.md:9710:decreasing both prototype granularity and fusion strength, so that early
./reviews/deep_2.md:9713:make the pipeline self-adaptive across datasets, we further introduce
./reviews/deep_2.md:9719:2.3. Re-ranking for ReID
./reviews/deep_2.md:9720:ReID can be seen as a retrieval task, and re-ranking strategy is
./reviews/deep_2.md:9721:an effective post-processing method to improve performance in image retrieval. Therefore, re-ranking methods originally designed for
./reviews/deep_2.md:9723:and average query expansion (AQE) [30] enhance retrieval performance by refining the initial ranking based on neighborhood relationships, feature averaging, or decision boundary distances. In the context
./reviews/deep_2.md:9725:information [31], and orientation-based re-ranking [32], further optimize rankings. Additionally, methods incorporating graph-based models [33], k-reciprocal encoding [34], and expanded cross-neighborhood
./reviews/deep_2.md:9726:re-ranking [35] have shown promise in improving ranking accuracy.
./reviews/deep_2.md:9737:making conventional re-ranking operations unreliable: neighborhoodbased averaging may drift toward mismatched semantics, and graph
./reviews/deep_2.md:9742:refinement methods that directly revise pairwise similarities or ranking
./reviews/deep_2.md:9846:This adaptively estimated 𝑁 is then seamlessly integrated into the
./reviews/deep_2.md:9865:so that early stages use more fine-grained prototypes, while later stages
./reviews/deep_2.md:9889:with 𝜆 the initial fusion weight. This schedule lets early stages apply stronger corrections based on fine-grained prototypes, while later
./reviews/deep_2.md:9937:decaying their influence through 𝛽, PDE𝑀 improves retrieval robustness: early stages correct prominent mismatches using localized identity
./reviews/deep_2.md:9938:cues, and later stages stabilize rankings under broader identity consensus, which is particularly beneficial on large-scale or distributionshifted datasets.
./reviews/deep_2.md:9970:3.6. Analysis of prototype-based ranking robustness
./reviews/deep_2.md:9979:To analyze how identity prototypes influence ranking robustness,
./reviews/deep_2.md:9980:consider a query sample 𝑞𝑖 and two gallery samples: 𝑔𝑗 + , a positive sample, and 𝑔𝑘− , a negative sample. A desirable TI-ReID ranking satisfies
./reviews/deep_2.md:10020:Case 1: Correct initial ranking. If 𝑆𝑞𝑔 (𝑖𝑗 + ) > 𝑆𝑞𝑔 (𝑖𝑘− ), the prototype term usually enlarges the margin. When the prototypes are close
./reviews/deep_2.md:10028:to more stable rankings.
./reviews/deep_2.md:10047:Case 2: Incorrect initial ranking. If 𝑆𝑞𝑔 (𝑖𝑗 + ) < 𝑆𝑞𝑔 (𝑖𝑘− ), the prototype term provides a corrective signal. Since prototypes summarize
./reviews/deep_2.md:10055:and rectify a mis-ranked pair.
./reviews/deep_2.md:10057:ranking robustness in two complementary ways: (1) amplifying margins
./reviews/deep_2.md:10058:for initially correct rankings and (2) offering a chance to correct
./reviews/deep_2.md:10097:Precision (mAP) [40]. Higher Rank-𝑘 and mAP values indicate better
./reviews/deep_2.md:10127:T2I in Rank-1 accuracy: prototype similarity is robust in a global,
./reviews/deep_2.md:10136:from the global ranking stability of prototypes while retaining the
./reviews/deep_2.md:10146:1.1%/1.24% in Rank-1 accuracy and 3.72%/5.56% in mAP over the
./reviews/deep_2.md:10147:RaSa model, and 0.99%/1.31% in Rank-1 and 3.38%/3.9% in mAP over
./reviews/deep_2.md:10149:improvements in Rank-1 and 3.79%/5.77% in mAP over IRRA, and
./reviews/deep_2.md:10150:0.68%/0.76% improvements in Rank-1 and 4.83%/5.42% in mAP over
./reviews/deep_2.md:10464:while slightly trailing the most recent work [50] in Rank-1, it excels
./reviews/deep_2.md:10467:improvement of +10.45% and +8.98% on Rank-1 and mAP. The
./reviews/deep_2.md:10547:cause premature convergence while overly large margins (> 0.5) increase optimization difficulty and risk of overfitting to noisy samples.
./reviews/deep_2.md:10707:The last four rows of Tables 2 and 3 clearly show that incorporating
./reviews/deep_2.md:10725:complexity.
./reviews/deep_2.md:10733:contributions of two critical components. Table 8 investigates the efficacy of dynamic weight adjustment for 𝑤𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑖𝑛𝑡𝑟𝑎 . Activation of this dynamic weight mechanism yields consistent improvements across core retrieval metric Rank-1 and mean average precision (mAP). Table 9 quantifies the impact of integrating self-attention
./reviews/deep_2.md:10737:improvements in Rank-𝑘 and mAP metrics.
./reviews/deep_2.md:10840:Model parameter count and computational complexity comparison.
./reviews/deep_2.md:11121:Fig. 7. Effect of the scaling factor 𝜆 and cluster number 𝑁 on the Rank-1 performance of HAM-PDE across three datasets.
./reviews/deep_2.md:11133:As shown in Figs. 7 and 8, Rank-1 and mAP follow similar trends with
./reviews/deep_2.md:11134:respect to 𝑁. Although the 𝑁 that maximizes Rank-1 does not always
./reviews/deep_2.md:11137:The larger fluctuation of Rank-1 on RSTPReid is likely due to
./reviews/deep_2.md:11146:as a good trade-off between effectiveness and complexity.
./reviews/deep_2.md:11151:Fig. 10 shows that Rank-1 is more variable, but larger 𝛽 generally
./reviews/deep_2.md:11171:Fig. 10. Effect of decay factors 𝛼 and 𝛽 on Rank-1 performance of HAM-PDE𝑀 across three datasets.
./reviews/deep_2.md:11222:and evaluation settings, methods such as k-reciprocal re-ranking, RWR,
./reviews/deep_2.md:11265:Fig. 13. Impact of bandwidth ℎ on Rank-1 performance of HAM-PDE𝑀 across three datasets.
./reviews/deep_2.md:11333:k-reciprocal re-ranking, RWR [51], and CSLS [52] are less stable,
./reviews/deep_2.md:11356:4.7.2. Visualization of ranking robustness
./reviews/deep_2.md:11359:a larger proportion of positive samples are retrieved and ranked higher
./reviews/deep_2.md:11382:Rank-1
./reviews/deep_2.md:11391:Rerank
./reviews/deep_2.md:11425:Rerank
./reviews/deep_2.md:11459:Rerank
./reviews/deep_2.md:11491:refinement, resulting in overall improved ranking of positive samples and enhanced retrieval precision.
./reviews/deep_2.md:11493:Fig. 16. Example retrieval results for three queries on the ICFG-PEDES dataset. For each query, the top row displays the ranking produced by the HAM model,
./reviews/deep_2.md:11494:and the bottom row shows the ranking after applying PDE𝑀 . Correct matches are highlighted with green boxes. (For interpretation of the references to color in
./reviews/deep_2.md:11537:APPD achieves 64.7% in Rank-1 and 60.5% in mAP, which
./reviews/deep_2.md:11538:surpasses MMM by 3.1% in Rank-1 and 2.6% in mAP. For
./reviews/deep_2.md:11541:70.6% in Rank-1 and 75.3% in mAP, exceeding MMM by a
./reviews/deep_2.md:11542:large margin with 6.2% in Rank-1 and 4.9% in mAP. Morever,
./reviews/deep_2.md:11543:the performance on RegDB achieves 91.1% in Rank-1 and
./reviews/deep_2.md:11545:in Rank-1 and 82.8% in mAP under the Thermal2Visible
./reviews/deep_2.md:11548:a 1.2% increase in average Rank-1 accuracy and a 1.0% boost
./reviews/deep_2.md:11566:dataset, APPD achieves a Rank-1 accuracy of 71.4% and a
./reviews/deep_2.md:11570:achieving 94.4% in Rank-1 accuracy and 88.4% in mAP under
./reviews/deep_2.md:11571:the Visible2Thermal setting, and 94.0% in Rank-1 accuracy
./reviews/deep_2.md:11585:achieves the performance with 92.1% in Rank-1 and 84.8%
./reviews/deep_2.md:11586:under the Visible2Thermal setting and 91.3% in Rank-1 and
./reviews/deep_2.md:11589:a notable improvement of 3.1% in average Rank-1 and 4.5%
./reviews/deep_2.md:11592:4.2% improvement in Rank-1 and a 7.1% improvement in
./reviews/deep_2.md:11606:is designed to explicitly rectify noisy pseudo-labels and adaptively identify cleaner pseudo-labels, ensuring the generation
./reviews/deep_2.md:11647:in a 2%-3% improvement in Rank-1 accuracy on SYSUMM01, demonstrating its ability to effectively capture complex
./reviews/deep_2.md:11671:in Rank-1 accuracy and 16.8% in mAP (see Order 1 &
./reviews/deep_2.md:11684:an adaptive pseudo-label division strategy to identify cleaner
./reviews/deep_2.md:11689:average Rank-1 accuracy and 1.3% in mAP.
./reviews/deep_2.md:11781:while the subsequent columns display the corresponding top10 retrieval images ranked from left to right based on similarity
./reviews/deep_2.md:11783:borders signify incorrect ones. As we can see, APPD significantly enhances the ranking results, achieving more correctly
./reviews/deep_2.md:11852:and Rank-1 score show notable improvements, increasing by
./reviews/deep_2.md:11854:and Rank-1 score increasing by 18.4% and 16.9%, respectively. The APE has the capability to extract adaptive body
./reviews/deep_2.md:11867:1.3% in rank-1 score. The improvement is more pronounced
./reviews/deep_2.md:11869:and rank-1 score by 2.9%. This notable enhancement can
./reviews/deep_2.md:11905:designed to extract adaptive and robust human body region.
./reviews/deep_2.md:12125:[51] Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking person reidentification with K-reciprocal encoding,” in Proc. IEEE Conf.
./reviews/deep_2.md:12182:4.5.1. Evaluation of parameters and computational complexity
./reviews/deep_2.md:12183:We quantitatively evaluate the parameters and computational complexity of the ADA framework. The detailed data is shown in the
./reviews/deep_2.md:12328:mechanisms eﬀectively enhance domain adaptation capabilities. However, sparse self-attention exhibits three distinct advantages: (1) Reduced Computational Complexity: Achieves lower FLOPs compared to
./reviews/deep_2.md:12335:necessitates increased hyperparameter tuning complexity. (2) Static
./reviews/deep_2.md:12396:[27] Z. Zhong, L. Zheng, D. Cao, S. Li, Re-ranking person re-identiﬁcation with kreciprocal encoding, in: Proceedings of the IEEE Conference on Computer Vision
./reviews/deep_2.md:12407:[31] T. He, L. Shen, Y. Guo, G. Ding, Z. Guo, SECRET: self-consistent pseudo label reﬁnement for unsupervised domain adaptive person re-identiﬁcation, in: Proceedings of
./reviews/deep_2.md:12436:[6] Y. Zhu, Y. Zheng, J. Liu, Y. Li, Z.-j. Zha, Noise-resistance learning via multigranularity consistency for unsupervised domain adaptive person re-identiﬁcation,
./reviews/deep_2.md:12439:cluster adaptive balancing loss for unsupervised person re-identiﬁcation, Pattern
./reviews/deep_2.md:12450:[12] S. Lai, Z. Chai, X. Wei, Transformer meets part model: adaptive part division for
./reviews/deep_2.md:12465:complexity, 2020, arXiv:2006.04768
./reviews/deep_2.md:12512:Rank-1
./reviews/deep_2.md:12516:Rank-1
./reviews/deep_2.md:12606:Rank-1
./reviews/deep_2.md:12610:Rank-1
./reviews/deep_2.md:12614:Rank-1
./reviews/deep_2.md:12618:Rank-1
./reviews/deep_2.md:12774:Rank-1
./reviews/deep_2.md:12778:Rank-1
./reviews/deep_2.md:12782:Rank-1
./reviews/deep_2.md:12786:Rank-1
./reviews/deep_2.md:12790:Rank-1
./reviews/deep_2.md:12856:accuracies of Rank-1 and mAP have increased by 4.3% and 1.4%
./reviews/deep_2.md:12861:further improved, especially the Rank-1 on NKUP and Celeb-reID-light
./reviews/deep_2.md:12865:When MLDC and LPAM are used simultaneously, the model’s performance of all datasets is significantly improved, especially on CelebreID-light, where the accuracies of Rank-1 and mAP are increased by
./reviews/deep_2.md:12874:all datasets has been further improved, especially on the VC-ClothesW&R dataset, where Rank-1 and mAP are increased by 1.0% and
./reviews/deep_2.md:12884:these two modules are used in combination, its matching rate in Ranks
./reviews/deep_2.md:12948:Effectiveness of adaptive weights for loss functions. In order
./reviews/deep_2.md:12949:to explore the specific effects of adaptive weighting and fixed weight
./reviews/deep_2.md:12977:Rank-1
./reviews/deep_2.md:12981:Rank-1
./reviews/deep_2.md:12985:Rank-1
./reviews/deep_2.md:13038:Ablation study of manual weighting and adaptive weighting.
./reviews/deep_2.md:13051:Rank-1
./reviews/deep_2.md:13053:Rank-5
./reviews/deep_2.md:13105:Ours (adaptive weights)
./reviews/deep_2.md:13108:evident that the adaptive weights outperform the best manual weights
./reviews/deep_2.md:13109:by 1.3%, 1.6% and 0.6% on Rank-1, Rank-5 and mAP respectively.
./reviews/deep_2.md:13113:experimental results fully demonstrate that adaptive weighting has a
./reviews/deep_2.md:13237:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./reviews/deep_2.md:13269:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./reviews/deep_2.md:13273:4. **证据闭环**：关键证据很直接。APE 单独带来大幅提升，在 Occluded-Duke 上 mAP 和 Rank-1 分别涨 18.4 和 16.9；OAM 再带来 2.6 和 2.9。部位数实验说明 4 个部位最合理，太多会冗余，6 个会忽略左右区分。最有说服力的是可视化，APE 在姿态标签错误时仍能抽对区域，能避开伞遮住的头部，还能抓住姿态估计忽略的手提包。这是在证明“自适应区域优于姿态输出”，不是只证明最终分数涨。
./reviews/deep_2.md:13318:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./reviews/deep_2.md:13350:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./reviews/deep_2.md:13354:4. **证据闭环**：关键证据很直接。APE 单独带来大幅提升，在 Occluded-Duke 上 mAP 和 Rank-1 分别涨 18.4 和 16.9；OAM 再带来 2.6 和 2.9。部位数实验说明 4 个部位最合理，太多会冗余，6 个会忽略左右区分。最有说服力的是可视化，APE 在姿态标签错误时仍能抽对区域，能避开伞遮住的头部，还能抓住姿态估计忽略的手提包。这是在证明“自适应区域优于姿态输出”，不是只证明最终分数涨。
./analyses/deep_12.txt:10:**证据闭环：** 关键证据是组件消融：空间分支、时间分支、STIM、FA 逐步加上都有收益；再用 RCB、自注意力、交叉注意力对比 FEM/TIM；还有模型大小和 Rank-1 的效率图，说明不是靠堆参数。弱点是它没有特别强地证明“一半帧就足够”这个核心重定义，只是间接由效率和性能成立。
./analyses/deep_12.txt:28:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./pivot/p_3.md:39: exited 1 in 0ms:
./pivot/p_3.md:626:| 062-eq | PSG+GCN + LKU (eq_concat) | 60.7% | 71.2% | 84.1% | 87.4% | mAP+4.1%, R1+4.7% | ❌ 负面。vs 3-seed: -0.03%/-1.37%。Learned uncertainty 损害 R1 |
./pivot/p_3.md:646:| 078-eq | PSG+GCN+PAA+APG (eq) | 60.5% | 72.5% | 84.3% | 87.9% | mAP+3.9%, R1+6.0% | ❌ vs PAA: -1.1%/-1.7%。adaptive gate 负面 |
./pivot/p_3.md:718:| 157+rr | PLBOA+Re-ranking | 78.8% | 79.7% | 87.8% | 90.0% | 🟢🟢🟢🟢🟢 含 re-ranking |
./pivot/p_3.md:767:1. NFC/Re-ranking 等 test-time 方法所有 SOTA 都可以用，不算公平的对比
./pivot/p_3.md:1530:| 13 | 新协议贴部署 | severe modality imbalance; mix-modality; anytime |
./pivot/p_3.md:1535:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
./pivot/p_3.md:1980:### #3 Time-Conditioned 跨年ReID(anytime)— 任务重定义故事最完整
./pivot/p_3.md:1983:- 风险: DeepChange需签学术协议; AT-USTC可得性不确定。anytime刚被AT-ReID(2509.16635,IJCAI25)命名窗口刚开。
./pivot/p_3.md:1998:例: ①频率解耦(VI)→空地VIR视频; ②RGB/CLIP特权蒸馏→event-only(隐私); ③diffusion生成(SD-ReID)→anytime(time-counterfactual); ④middle-image(MTRL灰图)→contour/sketch=ScribbleBridge。
./batches/bad:2:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./batches/bad:3:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./pivot/novelty_gopl.md:16:用联网**尽力查确切先例+撞车**, 逐类判它们是否已做'用几何共同可见表面定same-ID正边可信度并据此重排训练关系': (a)hard-positive mining / sample re-weighting / positive reliability ReID; (b)noisy-correspondence / noisy-label ReID(positive不可靠); (c)curriculum/self-paced ReID(by difficulty); (d)pose/visibility/co-visibility-aware ReID; (e)SMPL/3D-geometry-guided ReID; (f)graph-sampler / relation ReID; (g)metric learning 里'positive不等价'(supcon变体/soft-positive)。撞了说清撞哪+杀伤; 没撞说清切开点。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./pivot/novelty_gopl.md:66:web search: curriculum learning person re-identification training difficulty positive pairs
./pivot/novelty_gopl.md:141:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./pivot/novelty_gopl.md:184:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./analyses/deep_13.txt:34:3. **机制怎么长出来**：CGS 每个 epoch 用 Spherical-LSH 先把相近类别放进 bucket，再用 feature-map adaptive matching 找邻近类别，构造相关 hard batch。GSANet 保持高分辨率流，GRSR 用稀疏全局相关像素重构特征，提升特征质量。特征更好会让下一轮 CGS 更好，CGS 更好又提供更有价值样本。这个“互相增强”的逻辑成立，但模块数量较多，绑定不如 GAReID 干净。
./analyses/deep_13.txt:35:4. **证据闭环**：采样对比显示 CGS 好于 random、PK、cluster，并且比已有 GS 快很多，Market 上采样 0.1 秒对 4 秒，MSMT(all) 上 1 秒对 40 秒。CGS 相比 Cluster 在 Market 到 CUHK03 上提升约 3.0 Rank-1 和 2.8 mAP，在 Market 到 MSMT17 上提升约 3.9 和 2.9。CGS、GRSR、GSANet 的组合消融显示完整模型最好。采样可视化证明 CGS 找到衣服颜色、局部外观相近的邻近类别。
./analyses/deep_13.txt:39:1. **触发观察**：ReID 是 open-set ranking，分类攻击不能直接搬。已有 ReID 攻击黑盒迁移差，作者观察到早期迭代里 current gradient 和 global aggregated gradient 一致性很低，更新方向随机震荡，导致迁移性不足。
./reviews/deep_27.md:50:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_27.md:67:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_27.md:70:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_27.md:79:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_27.md:111:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_27.md:141:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_27.md:274:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_27.md:431:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_27.md:460:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_27.md:463:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_27.md:464:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_27.md:541:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_27.md:545:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_27.md:559:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_27.md:565:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_27.md:666:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_27.md:741:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_27.md:888:5.2% in Rank-1 accuracy.
./reviews/deep_27.md:941:learning significantly increases the difficulty of model training. Consequently, in unsupervised
./reviews/deep_27.md:1047:introduces adaptive style rehearsal to dynamically model domain distributions, further enhancing
./reviews/deep_27.md:1444:method CLIP-ReID [24] underperforms the top CC re-ID method SCNet [20] by ∼8 pp w.r.t. Rank-1 under CC. This is because the SC re-ID
./reviews/deep_27.md:1596:Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
./reviews/deep_27.md:1824:Rank-1
./reviews/deep_27.md:1828:Rank-1
./reviews/deep_27.md:1965:most methods, the comparison is performed under GEN. We employ Cumulative Matching Characteristic Rank-1 accuracy and Mean Average
./reviews/deep_27.md:2006:Rank-1
./reviews/deep_27.md:2010:Rank-1
./reviews/deep_27.md:2082:existing top CC re-ID method, SC-Net [20], by 2.5 pp w.r.t. Rank-1 under CC. Due to the simplicity of the SC scenario on PRCC, the performance reached saturation.Our method achieves 100% Rank-1 accuracy,
./reviews/deep_27.md:2098:based adaptive scheduling.
./reviews/deep_27.md:2108:Rank-1
./reviews/deep_27.md:2112:Rank-1
./reviews/deep_27.md:2174:Rank-1
./reviews/deep_27.md:2217:Rank-1
./reviews/deep_27.md:2221:Rank-1
./reviews/deep_27.md:2300:Rank-1
./reviews/deep_27.md:2304:Rank-1
./reviews/deep_27.md:2331:Rank-1
./reviews/deep_27.md:2335:Rank-1
./reviews/deep_27.md:2372:the appropriate and adaptive introduction of new clothes data into the
./reviews/deep_27.md:2379:using DS, we obtain an improvement of 1.4 pp and 0.8 pp w.r.t. Rank-1
./reviews/deep_27.md:2381:improvement of 1.9 pp and 2.0 pp w.r.t. Rank-1 under CC on PRCC and
./reviews/deep_27.md:2382:LTCC, respectively; when adding both DW and DS, we obtain a signiﬁcant improvement of 5.8 pp and 5.9 pp w.r.t. Rank-1 under CC on PRCC
./reviews/deep_27.md:2406:Rank-1
./reviews/deep_27.md:2421:by 2.0 pp and 2.1 pp w.r.t. Rank-1 and mAP under CC. On two largescale datasets, LaST and DeepChange, our method outperforms the existing top CC re-ID method, MCSC, by 10.2 pp and 9.4 pp w.r.t. Rank-1
./reviews/deep_27.md:2456:Rank-1
./reviews/deep_27.md:2460:Rank-1
./reviews/deep_27.md:2508:Rank-1
./reviews/deep_27.md:2512:Rank-1
./reviews/deep_27.md:2544:Rank-1
./reviews/deep_27.md:2548:Rank-1
./reviews/deep_27.md:2829:the Mean Average Precision (mAP) and Rank-1 Accuracy (R1) for each dataset. The overall
./reviews/deep_27.md:2922:At the beginning of each epoch, DBSCAN [8] is applied on the re-ranked Jaccard distance [54]
./reviews/deep_27.md:2995:intra- and inter-camera relationships to improve global and local pseudo-labels. At the instance level, the aﬃnityguided mutual reﬁnement (AGMR) module computes aﬃnity scores between samples based on selected informative nodes, adaptively pulling reliable positive pairs closer while pushing negative ones apart. By integrating
./reviews/deep_27.md:3068:ﬁnal performance, particularly during the early training stages when
./reviews/deep_27.md:3070:in Fig. 1(a), the number of clusters generated by DBSCAN on Market1501 Zheng et al. (2015) and MSMT17 Wei et al. (2018) deviates substantially from the ground-truth identity count and ﬂuctuates throughout training, especially in the early stages. These ﬂuctuations highlight
./reviews/deep_27.md:3071:the instability of clustering results and further indicate the presence of
./reviews/deep_27.md:3156:the early training stages, limited feature extraction capability and ambiguous cluster boundaries result in substantial label noise. Moreover,
./reviews/deep_27.md:3198:divides images into patches and calculates the average distance between corresponding patches of sample pairs to adjust class ranking. LP
./reviews/deep_27.md:3351:adaptively pulled toward the centroids of its neighbors’ clusters. Although the original cluster remains the primary inﬂuence, its role in
./reviews/deep_27.md:3352:determining the pseudo-label is no longer absolute. As the model’s feature extraction capability improves, misclassiﬁed samples in the early
./reviews/deep_27.md:3387:on the corresponding upper and lower local features. Due to the incomplete semantic representation of local features, they tend to be less stable than global features. This instability makes inter-camera neighbor
./reviews/deep_27.md:3394:respectively. Their intersection, 𝑆𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝑘) serves as a ranked list of reliable intra-camera neighbors for sample 𝑥𝑖 , ordered by ascending feature distance. Following the same strategy as in the global reﬁnement,
./reviews/deep_27.md:3407:(𝑖, 𝑘), up to its last occurrence. The probability 𝑃𝑖𝑛𝑡𝑒𝑟 is determined by the distance rank of neighboring samples and the occurrence
./reviews/deep_27.md:3659:𝑥𝑖 itself to enforce strong self-supervision. The weights for its informative nodes are adaptively determined by their aﬃnity scores 𝑠𝑐𝑜𝑟𝑒(𝑗 ∣ 𝑥𝑖 ),
./reviews/deep_27.md:3709:its complexity and scale. It contains 126,441 images collected from 15
./reviews/deep_27.md:3712:comparison to Market-1501, MSMT17 exhibits signiﬁcantly higher complexity, characterized by greater environmental variability, increased
./reviews/deep_27.md:3716:Matching Characteristic (CMC) accuracy. mAP assesses the overall retrieval quality by integrating both precision and recall across all ranking
./reviews/deep_27.md:3717:positions, providing a comprehensive measure of the method’s eﬀectiveness. CMC quantiﬁes the probability of the correct match appearing within the top-ranked results. In this study, we report the Rank-1,
./reviews/deep_27.md:3718:Rank-5, and Rank-10 accuracies of the CMC metric to provide a detailed
./reviews/deep_27.md:3728:in cluster-level contrastive loss 𝐿𝐷𝐼𝐼𝑅 , the model is guided to adaptively
./reviews/deep_27.md:4011:and a 9.3 % increase in Rank-1 accuracy.
./reviews/deep_27.md:4018:information to alleviate label noise and camera bias. On the Market1501 dataset, our method achieves mAP and Rank-1 scores of 86.7 %
./reviews/deep_27.md:4020:and Rank-1 scores of 49.1 % and 75.9 %, signiﬁcantly surpassing the
./reviews/deep_27.md:4282:[32] provide fine-grained patches but increase complexity
./reviews/deep_27.md:4312:transformer parameters to control the content of the generated patches. This reduces computational complexity while
./reviews/deep_27.md:4423:should utilize units with the largest receptive field as early as
./reviews/deep_27.md:4456:the early layers. As shown in Fig. 4, it relies on a channelwise fast Fourier transform (FFT) [50] and has a receptive
./reviews/deep_27.md:4474:global features from the early layers, which is essential for
./reviews/deep_27.md:4770:where i, j ∈ [1, P]. The following loss is utilized to adaptively
./reviews/deep_27.md:4888:the Cumulative Matching Characteristics (CMC) for Rank-1,
./reviews/deep_27.md:4889:Rank-5, and Rank-10 accuracies, as well as mean Average
./reviews/deep_27.md:4890:Precision (mAP), without applying re-ranking.
./reviews/deep_27.md:4909:By adjusting their values, we evaluate the Rank-1 accuracy on
./reviews/deep_27.md:4924:Rank-1 and mAP accuracy improve with increasing P. When P
./reviews/deep_27.md:4944:scale, complexity, and diversity, making them ideal for assessing the effectiveness and robustness of our method. We use the
./reviews/deep_27.md:4949:boosting Rank-1/mAP accuracy by 6.3%/13.0% on Market1501 and 24.8%/20.5% on MSMT17 compared to the baseline.
./reviews/deep_27.md:5082:adaptively pulled toward the centroids of its neighbors’ clusters. Although the original cluster remains the primary inﬂuence, its role in
./reviews/deep_27.md:5083:determining the pseudo-label is no longer absolute. As the model’s feature extraction capability improves, misclassiﬁed samples in the early
./reviews/deep_27.md:5118:on the corresponding upper and lower local features. Due to the incomplete semantic representation of local features, they tend to be less stable than global features. This instability makes inter-camera neighbor
./reviews/deep_27.md:5125:respectively. Their intersection, 𝑆𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝑘) serves as a ranked list of reliable intra-camera neighbors for sample 𝑥𝑖 , ordered by ascending feature distance. Following the same strategy as in the global reﬁnement,
./reviews/deep_27.md:5138:(𝑖, 𝑘), up to its last occurrence. The probability 𝑃𝑖𝑛𝑡𝑒𝑟 is determined by the distance rank of neighboring samples and the occurrence
./reviews/deep_27.md:5469:compensate for the model’s weak discrimination ability in early stages, ultimately enhancing its generalization
./reviews/deep_27.md:6049:For all experiments, we adopt Rank-1, Rank-5, and Rank-10
./reviews/deep_27.md:6055:our proposed MLWM achieved 78.46% Rank-1 accuracy
./reviews/deep_27.md:6063:existing methods in terms of Rank-1 performance under
./reviews/deep_27.md:6065:a Rank-1 accuracy of 68.80%, surpassing the most recent
./reviews/deep_27.md:6072:“G” and “L” in “Type” denote the global-matching/local-matching method. Results are ordered based on the Rank-1 accuracy.
./reviews/deep_27.md:6101:Rank-1
./reviews/deep_27.md:6115:Rank-5
./reviews/deep_27.md:6116:Rank-10
./reviews/deep_27.md:6146:Rank-1
./reviews/deep_27.md:6155:Rank-5
./reviews/deep_27.md:6156:Rank-10
./reviews/deep_27.md:6175:Rank-1
./reviews/deep_27.md:6189:Rank-5
./reviews/deep_27.md:6190:Rank-10
./reviews/deep_27.md:6234:Rank-1
./reviews/deep_27.md:6242:Rank-5
./reviews/deep_27.md:6243:Rank-10
./reviews/deep_27.md:6255:local well-posedness. However, the method is slightly less effective in terms of Rank-5, Rank-10, and mAP. This method is
./reviews/deep_27.md:6256:slightly less effective in terms of Rank-5, Rank-10, and mAP,
./reviews/deep_27.md:6257:but we optimize the accuracy of the top-ranked result more,
./reviews/deep_27.md:6263:surpassing CADA by 1.3% and 0.61% in Rank-1 accuracy and
./reviews/deep_27.md:6266:To fully demonstrate the effectiveness of different components within the MLWM framework, we conducted experiments on RSTPReid, comparing the performance across Rank1, Rank-5, and Rank-10 metrics. The experimental results are
./reviews/deep_27.md:6271:+ 2.25%, 1.80% and 1.25% on Rank-1, Rank-5, and Rank-10
./reviews/deep_27.md:6287:The results clearly demonstrate that the MLWM method, based
./reviews/deep_27.md:6467:early stages when discriminative ability is limited. Furthermore, by integrating UPR with MBLM,
./reviews/deep_27.md:6955:Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
./reviews/deep_27.md:7183:Rank-1
./reviews/deep_27.md:7187:Rank-1
./reviews/deep_27.md:7324:most methods, the comparison is performed under GEN. We employ Cumulative Matching Characteristic Rank-1 accuracy and Mean Average
./reviews/deep_27.md:7365:Rank-1
./reviews/deep_27.md:7369:Rank-1
./reviews/deep_27.md:7441:existing top CC re-ID method, SC-Net [20], by 2.5 pp w.r.t. Rank-1 under CC. Due to the simplicity of the SC scenario on PRCC, the performance reached saturation.Our method achieves 100% Rank-1 accuracy,
./pivot/clean/novelty_hubness.txt:7:| (b) ReID k-reciprocal / neighbor re-ranking | **存活，但需强对照** | 这类已经把 ReID 当邻域拓扑问题处理，但在**测试后处理**改距离/排名，不是训练单 embedding；也不是把少数 gallery 的跨 ID 负 in-degree 当失败主变量。 | k-reciprocal 用 reciprocal NN、Jaccard distance、local query expansion；ECN / PSE 也是 unsupervised re-ranking。链接：[Zhong k-reciprocal](https://arxiv.org/abs/1701.08398), [PSE/ECN](https://arxiv.org/abs/1711.10378) | 8 |
./pivot/clean/novelty_hubness.txt:8:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./pivot/clean/novelty_hubness.txt:9:| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
./pivot/clean/novelty_hubness.txt:17:- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
./pivot/clean/novelty_hubness.txt:18:- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。
./reviews/deep_12.md:191:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_12.md:230:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_12.md:358:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_12.md:387:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_12.md:390:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_12.md:391:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_12.md:468:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_12.md:472:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_12.md:486:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_12.md:492:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_12.md:566:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_12.md:599:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_12.md:607:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_12.md:685:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_12.md:724:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_12.md:725:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_12.md:791:potential false negatives, while AFNS assigns adaptive weights
./reviews/deep_12.md:891:AFNS assigns adaptive weights, adjusted based on false negative confidence, to the similarities between the anchor and
./reviews/deep_12.md:1087:the selected local visual and textual token features are linearly
./reviews/deep_12.md:1212:in the early stages of training, the backbone model is insufficiently trained, leading to unreliable image-text features. As a
./reviews/deep_12.md:1220:loss that adaptively adjusts the weights of identified false
./reviews/deep_12.md:1257:detected false negative samples, ρ(·) denotes the adaptive
./reviews/deep_12.md:1258:weight function that is adaptively adjusted based on the false
./reviews/deep_12.md:1332:and a 5 epoch warm-up, where the learning rate is linearly
./reviews/deep_12.md:1512:outperforms FNE and FNA methods on all datasets, demonstrating the effectiveness of adaptive weights in mitigating the
./reviews/deep_12.md:1763:model with adaptive self-adversarial training,” in Proceedings of the
./reviews/deep_12.md:1776:[21] Zhiwei Zhao, Bin Liu, Yan Lu, et al., “Unifying multi-modal uncertainty modeling and semantic alignment for text-to-image person reidentification,” in Proceedings of the AAAI Conf. Artif. Intell., 2024,
./reviews/deep_12.md:1832:Notably, it reports 91.0% Rank-1 and 76.9% mAP on the
./reviews/deep_12.md:1937:Rank-1 and 76.9% mAP, on the MSMT17 benchmark, and
./reviews/deep_12.md:2369:Rank-1 accuracy and mAP of various methods across four
./reviews/deep_12.md:2440:Rank-1
./reviews/deep_12.md:2472:to state-of-the-arts methods on all four experimental benchmarks. Specially, we achieve 91.0% Rank-1 and 76.9% mAP
./reviews/deep_12.md:2475:Notably, all the improvements are achieved under the singlequery mode without re-ranking or other bells and whistles,
./reviews/deep_12.md:2502:Rank-1
./reviews/deep_12.md:2526:yields 0.5% gains of Rank-1 and 0.6% enhancement of mAP.
./reviews/deep_12.md:2535:Rank-1
./reviews/deep_12.md:2575:Rank-1
./reviews/deep_12.md:2597:Rank-1
./reviews/deep_12.md:2621:improvement in Rank-1 accuracy and a 0.3% increase in mAP.
./reviews/deep_12.md:2627:Rank-1 and 0.6% enhancement of mAP. This indicates that
./reviews/deep_12.md:2643:Rank-1
./reviews/deep_12.md:2666:does not necessarily translate to a higher Rank-1 accuracy
./reviews/deep_12.md:2990:due to its difficulty. In this section, we will introduce these
./reviews/deep_12.md:3043:adaptive interference removal framework to remove various
./reviews/deep_12.md:3048:mentioned above; they can achieve 97.2 and 96.9 on Rank-1
./reviews/deep_12.md:3090:development time and the task’s difficulty, the performance of
./reviews/deep_12.md:3094:only achieve 89.2 and 84.7 on the Rank-1 metric in the
./reviews/deep_12.md:3210:while maintaining low complexity.
./reviews/deep_12.md:3367:Table I shows the parameter and computational complexity
./reviews/deep_12.md:3379:complexity and needs more modalities. So, our MGA is
./reviews/deep_12.md:3405:quantity, computational complexity, and input modality as the
./reviews/deep_12.md:3530:Rank-k matching accuracy and the mean Average Precision
./reviews/deep_12.md:3531:(mAP) are used to evaluate performance. Rank-k matching
./reviews/deep_12.md:3533:the top-k ranked retrieved results. The mAP reflects the overall
./reviews/deep_12.md:3594:by 4.2% and 2.8% for the Rank-1 metric on the general and
./reviews/deep_12.md:3597:it by 4.8% and 6.1% for the Rank-1 metric on the general and
./reviews/deep_12.md:3632:∼ 3.6% and 2.64% ∼ 5.76% on Rank-1 with AP3D and
./reviews/deep_12.md:3669:(FEM) for the spatial branch to focus on enhancing spatial dependencies adaptively, and a Temporal Interaction Module (TIM)
./reviews/deep_12.md:3675:complexity.
./reviews/deep_12.md:3755:deformable attention [19] to adaptively extract crucial pedestrian information while reducing computational overhead. To
./reviews/deep_12.md:3975:Ft . This processing enables the FEM to adaptively focus on
./reviews/deep_12.md:4153:Rank-1 mAP
./reviews/deep_12.md:4204:Rank-1 Rank-5
./reviews/deep_12.md:4236:Rank-1
./reviews/deep_12.md:4237:Rank-5
./reviews/deep_12.md:4260:Fig. 4. Comparison of Rank-1 accuracy and model size between our ESTI
./reviews/deep_12.md:4327:Rank-1
./reviews/deep_12.md:4386:Rank-2
./reviews/deep_12.md:4388:Rank-3
./reviews/deep_12.md:4390:Rank-4
./reviews/deep_12.md:4397:Rank-5
./reviews/deep_12.md:4407:Query Feature Map Rank-1
./reviews/deep_12.md:4411:Rank-1
./reviews/deep_12.md:4431:highest Rank-1 accuracy (91.9%) on the MARS dataset and the
./reviews/deep_12.md:4432:highest Rank-5 accuracy (99.3%) on the iLIDS-VID dataset.
./reviews/deep_12.md:4434:best Rank-1 accuracy. While we achieve the second-best mAP
./reviews/deep_12.md:4435:on the MARS dataset and Rank-1 accuracy on iLIDS-VID, our
./reviews/deep_12.md:4437:which attains the best mAP on MARS and the best Rank-1
./reviews/deep_12.md:4442:Rank-1
./reviews/deep_12.md:4462:Rank-1
./reviews/deep_12.md:4470:and effectiveness, obtaining the highest Rank-1 accuracy while
./reviews/deep_12.md:4479:increases 1.1% mAP and 2.0% Rank-1 accuracy compare to
./reviews/deep_12.md:4481:mAP and 2.6% Rank-1 accuracy improvement. Combining
./reviews/deep_12.md:4483:2.9% Rank-1 accuracy increase over the baseline. We further
./reviews/deep_12.md:4486:module provides an additional 0.1% mAP and 0.3% Rank-1
./reviews/deep_12.md:4487:accuracy improvement. Incorporating the STIM boosts performance by 0.1% mAP and 0.2% Rank-1 accuracy. Eventually,
./reviews/deep_12.md:4490:and computational complexity (FLOPs) introduced by each
./reviews/deep_12.md:4503:in Rank-1 accuracy, and surpasses the addition with RCBs
./reviews/deep_12.md:4504:by 0.7% in mAP and 0.9% in Rank-1 accuracy. This shows
./reviews/deep_12.md:4506:branch adaptively extract distinctive spatial dependencies by
./reviews/deep_12.md:4514:by 0.9% in mAP and 1.2% in Rank-1 accuracy. Furthermore, it
./reviews/deep_12.md:4515:achieves a 0.4% mAP and 0.5% Rank-1 accuracy improvement
./reviews/deep_12.md:4523:0.2 Rank-1 accuracy demonstrate that the FEM enhances the
./reviews/deep_12.md:4535:ESTI consistently identifies the correct matches from Rank-1
./reviews/deep_12.md:4536:to Rank-5. This demonstrates the superiority of ESTI in capturing distinctive spatial-temporal features, leading to improved
./reviews/deep_12.md:4544:(FEM) and a Temporal Interaction Module (TIM) are designed for extract enhanced spatial features adaptively and
./reviews/deep_12.md:4597:“Person re-identification by video ranking,” in ECCV. Springer, 2014,
./reviews/deep_12.md:4666:complexity of real-world operational environments. Specifically, while training datasets
./reviews/deep_12.md:4697:Information Sharing (SIS): A mechanism that promotes stylistic diversity across federated clients by enabling early-layer style information exchange. This encourages models to
./reviews/deep_12.md:4961:Extrapolation mechanism that adaptively perturbs the statistical style descriptors of feature representations. Specifically, we define an extrapolation function E(µ, σ, α), where
./reviews/deep_12.md:5024:extrapolated statistics are then applied to transform the features via adaptive normalization.
./reviews/deep_12.md:5225:clusters in two modalities based on the similarity ranking of
./reviews/deep_12.md:5741:The total complexity of both the proposed adversarial learning method and the baseline is as follows:
./reviews/deep_12.md:5761:E. Analysis of Computational Complexity
./reviews/deep_12.md:5762:The complexity of the proposed adversarial learning Lao
./reviews/deep_12.md:5764:complexity of the backbone model that extracts a single picture
./reviews/deep_12.md:5766:instances. Then, the complexity of the backbone is used to
./reviews/deep_12.md:5777:computational complexity is as follows:
./reviews/deep_12.md:5788:floating-point operation, respectively, the computational complexity of the sampling process is
./reviews/deep_12.md:5795:The computational complexity of the adversarial optimization
./reviews/deep_12.md:5801:The computational complexity of the proposed adversarial
./reviews/deep_12.md:5811:complexity during the feature extraction. For this reason, the
./reviews/deep_12.md:5813:complexity, and the main computational complexity comes
./reviews/deep_12.md:6028:by 4.2% and 2.8% for the Rank-1 metric on the general and
./reviews/deep_12.md:6031:it by 4.8% and 6.1% for the Rank-1 metric on the general and
./reviews/deep_12.md:6066:∼ 3.6% and 2.64% ∼ 5.76% on Rank-1 with AP3D and
./reviews/deep_12.md:6104:0.84% on Rank-1 with AP3D and TCLNet, respectively.
./reviews/deep_12.md:6105:The parameters, computational complexity (MACs), and input
./reviews/deep_12.md:6172:performance on Rank-1 is only 91.73. Our best choice of
./reviews/deep_12.md:6185:Rank-1, which can demonstrate the effectiveness of our EPFM.
./reviews/deep_12.md:6268:complexity.
./reviews/deep_12.md:6270:the future. The limitation of our work is the complexity of the
./reviews/deep_12.md:6380:by video ranking,” in Computer Vision–(ECCV). Cham, Switzerland:
./reviews/deep_12.md:6464:[43] H. Tao, Q. Duan, and J. An, “An adaptive interference removal framework for video person re-identification,” IEEE Trans. Circuits Syst. Video
./reviews/deep_12.md:6596:extrapolated statistics are then applied to transform the features via adaptive normalization.
./reviews/deep_12.md:6639:well-suited for this task due to two main advantages: (1) its attention mechanism adaptively assigns importance to each node, which is crucial as different body regions contribute
./reviews/deep_12.md:6732:(SIS) mechanism is integrated into the early layers of the backbone to retain style-relevant
./reviews/deep_12.md:6805:offering viewpoint-invariant cues–adaptive fusion enables the model to leverage both
./reviews/deep_12.md:6816:To adaptively determine the relative importance of each feature type, we introduce a
./reviews/deep_12.md:6825:where ∥ denotes feature concatenation. The outputs wai and wsi represent the adaptive
./reviews/deep_12.md:7003:Precision (mAP) and Rank-1 accuracy. Performance results in this section are reported in
./reviews/deep_12.md:7004:the format (Rank-1 / mAP) for consistency across experiments.
./reviews/deep_12.md:7431:The total complexity of both the proposed adversarial learning method and the baseline is as follows:
./reviews/deep_12.md:7451:E. Analysis of Computational Complexity
./reviews/deep_12.md:7452:The complexity of the proposed adversarial learning Lao
./reviews/deep_12.md:7454:complexity of the backbone model that extracts a single picture
./reviews/deep_12.md:7456:instances. Then, the complexity of the backbone is used to
./reviews/deep_12.md:7467:computational complexity is as follows:
./reviews/deep_12.md:7478:floating-point operation, respectively, the computational complexity of the sampling process is
./reviews/deep_12.md:7485:The computational complexity of the adversarial optimization
./reviews/deep_12.md:7491:The computational complexity of the proposed adversarial
./reviews/deep_12.md:7501:complexity during the feature extraction. For this reason, the
./reviews/deep_12.md:7503:complexity, and the main computational complexity comes
./reviews/deep_12.md:7792:VIPeR, and iLIDs) and report the average of mean Average Precision (mAP) and Rank-1
./reviews/deep_12.md:7808:8.0% in mAP and 7.2% in Rank-1 accuracy. This superior performance demonstrates that
./reviews/deep_12.md:7826:Rank-1/mAP
./reviews/deep_12.md:7827:Rank-1/mAP
./reviews/deep_12.md:7828:Rank-1/mAP
./reviews/deep_12.md:7829:Rank-1/mAP
./reviews/deep_12.md:7830:Rank-1/mAP
./reviews/deep_12.md:7900:Rank-1 accuracy. This improvement is primarily attributed to the enlarged training data
./reviews/deep_12.md:7907:and +15.5% in Rank-1 accuracy over the previous state-of-the-art, representing the most
./reviews/deep_12.md:7912:a +5.5% improvement in Rank-1 accuracy. These results confirm that our method remains
./reviews/deep_12.md:7914:an additional +3.1% mAP and a notable +10.8% improvement in Rank-1 accuracy. Given
./reviews/deep_12.md:7936:Rank-1/mAP
./reviews/deep_12.md:7937:Rank-1/mAP Rank-1/mAP Rank-1/mAP
./reviews/deep_12.md:7938:Rank-1/mAP
./reviews/deep_12.md:7995:Rank-1 accuracy drops below 10%, and the average mAP remains around 30%. These
./reviews/deep_12.md:7999:achieves an average mAP of 13.6% and Rank-1 accuracy of 11.8% across all four test sets.
./reviews/deep_12.md:8007:notably, with a 4.9% drop in average mAP and an 8.5% decline in Rank-1 accuracy. This
./reviews/deep_12.md:8026:Rank-1/mAP
./reviews/deep_12.md:8027:Rank-1/mAP
./reviews/deep_12.md:8028:Rank-1/mAP
./reviews/deep_12.md:8029:Rank-1/mAP
./reviews/deep_12.md:8030:Rank-1/mAP
./reviews/deep_12.md:8117:Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization.
./reviews/deep_12.md:8132:Liao, S., & Shao, L. (2020). Interpretable and generalizable person re-identification with query-adaptive
./reviews/deep_12.md:8156:Xu, B., Liang, J., He, L., & Sun, Z. (2021). Mimic embedding via adaptive aggregation: Learning generalizable person re-identification.
./reviews/deep_12.md:8200:**证据闭环：** 关键证据是组件消融：空间分支、时间分支、STIM、FA 逐步加上都有收益；再用 RCB、自注意力、交叉注意力对比 FEM/TIM；还有模型大小和 Rank-1 的效率图，说明不是靠堆参数。弱点是它没有特别强地证明“一半帧就足够”这个核心重定义，只是间接由效率和性能成立。
./reviews/deep_12.md:8218:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./reviews/deep_12.md:8272:**证据闭环：** 关键证据是组件消融：空间分支、时间分支、STIM、FA 逐步加上都有收益；再用 RCB、自注意力、交叉注意力对比 FEM/TIM；还有模型大小和 Rank-1 的效率图，说明不是靠堆参数。弱点是它没有特别强地证明“一半帧就足够”这个核心重定义，只是间接由效率和性能成立。
./reviews/deep_12.md:8290:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./analyses/deep_11.txt:15:**2. Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification**
./analyses/deep_11.txt:19:2. 重定义动作：他们没有泛泛说“伪标签有噪声”，而是把噪声拆成三个更具体的问题：距离度量不懂相机，代理点不符合 DBSCAN 的局部密度结构，监督目标太硬。关键词是 dynamic adaptive camera Jaccard、manifold、medoid proxy、dynamic adaptive proxies、soft pseudo labels。
./analyses/deep_11.txt:47:4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。
./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./analyses/deep_11.txt:67:2. 重定义动作：它把“缩小 modality gap”重定义成“增强模型对 modality、input instance、query instance 的适应能力”。关键词是 model adaptability、flexibility、modality-aware prompts、instance-aware prompts、query-guiding prompts、adaptive feature space。这个转向很重要，因为它避开了“把两种模态硬映射到同一空间”的旧叙事。
./pivot/clean/ondisk_pivot.txt:14:- 测试仍是单 text embedding 查 gallery image embedding，不用 MLLM、不用 re-ranking。
./pivot/clean/video_feasibility.txt:47:核心叙事是“强 ReID 的残差失败不是 pairwise similarity，而是 gallery topology 的 many-to-one 吸附”。主结果放 `M(q)` 解释 AP error，rho +0.60；然后诚实写 remedy 被 k-reciprocal / re-ranking 吞掉，因此这是诊断贡献，不是训练方法贡献。
./reviews/deep_22.md:93:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_22.md:225:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_22.md:242:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_22.md:245:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_22.md:254:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_22.md:286:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_22.md:316:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_22.md:420:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_22.md:449:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_22.md:452:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_22.md:453:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_22.md:530:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_22.md:534:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_22.md:548:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_22.md:554:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_22.md:648:allowing the model to capture uncertainty in the representations. Second, Cross-Modal Containment (CMC) aligns the
./reviews/deep_22.md:713:while rejecting visually similar but semantically distinct negatives (Icn ). Second, even methods that introduce uncertainty
./reviews/deep_22.md:737:uncertainty but also models the semantic containment relationships between vision and language. As illustrated in Fig. 1(c),
./reviews/deep_22.md:745:captures modality-specific uncertainty. By learning compact
./reviews/deep_22.md:768:semantic scope, distinguishing it from prior uncertaintyaware methods that treat variance primarily as a noise
./reviews/deep_22.md:802:Gaussian Mixture Models and triplet ranking loss to better
./reviews/deep_22.md:826:ambiguity, semantic granularity, and distributional uncertainty
./reviews/deep_22.md:830:for representing cross-modal semantics under uncertainty.
./reviews/deep_22.md:832:Probabilistic distribution representations have been increasingly used to model uncertainty and improve robustness
./reviews/deep_22.md:835:encode feature embeddings with uncertainty estimates reflecting representation quality. In domain generalization, DSU [37]
./reviews/deep_22.md:838:pose embeddings as Gaussians to handle input uncertainty
./reviews/deep_22.md:842:incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
./reviews/deep_22.md:843:uncertainty via batch- and identity-level variances.
./reviews/deep_22.md:860:cascaded cross-attention alignment [48], providing a principled
./reviews/deep_22.md:861:way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
./reviews/deep_22.md:866:can also represent feature uncertainty. Student’s t [50] distribution has heavier tails, which can capture rare or extreme
./reviews/deep_22.md:869:[51] allows modeling multi-modal uncertainty, accommodating features with multiple plausible states; however, learning
./reviews/deep_22.md:880:learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
./reviews/deep_22.md:882:While prior work primarily focuses on uncertainty within
./reviews/deep_22.md:965:uncertainty when only the first two moments are known. Crucially, the mathematical closure and analytical tractability of
./reviews/deep_22.md:1018:texts with low uncertainty is primarily determined by the
./reviews/deep_22.md:1020:while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./reviews/deep_22.md:1047:uncertainty-aware approaches that treat variance merely as a
./reviews/deep_22.md:1059:where µvi and µt j denote the mean features, and σ2vi , σ2t j represent variances that capture feature uncertainty and semantic
./reviews/deep_22.md:1266:enhances robustness by ensuring masked distributions encompass unmasked ones, reflecting increased uncertainty in partial
./reviews/deep_22.md:1317:exhibit greater uncertainty, thereby naturally encompassing the
./reviews/deep_22.md:1336:introduces uncertainty by randomly masking textual tokens.
./reviews/deep_22.md:1386:ICFG-PEDES [21], RSTPReid [54] and adopt the Rank-k and
./reviews/deep_22.md:1419:2) Evaluation Metrics: We adopt the popular Rank-k (%)
./reviews/deep_22.md:1421:values 1, 5, and 10. Rank-k represents the probability that,
./reviews/deep_22.md:1425:comprehensively evaluate model performance, which complements Rank-k by measuring the overall ranking quality across
./reviews/deep_22.md:1439:the batch size is fixed at 64. We adopt an early training strategy
./reviews/deep_22.md:1447:CUHK PEDES dataset. PDA achieves a Rank-1 accuracy of
./reviews/deep_22.md:1451:(MLLMs) for diverse textual generation, PDA improves Rank1 by 1.25% and mAP by 0.01%. When compared with MUM,
./reviews/deep_22.md:1453:uncertainty in cross modal retrieval, PDA demonstrates a
./reviews/deep_22.md:1454:notable advantage of 3.82% in Rank-1 accuracy. Furthermore,
./reviews/deep_22.md:1474:across all key metrics, with a Rank-1 of 69.27% and an
./reviews/deep_22.md:1476:improvement in Rank-1 and a 0.53% gain in mAP. Against
./reviews/deep_22.md:1477:MUM, our model shows a 3.65% advantage in Rank-1 accuracy, indicating its superior ability to cope with uncertainty.
./reviews/deep_22.md:1480:0.76% in Rank-1 and 0.82% in mAP. These improvements
./reviews/deep_22.md:1487:terms of Rank-1 and mAP. Specifically, PDA surpasses NAM
./reviews/deep_22.md:1488:by 2.20% in Rank-1 and 0.77% in mAP, despite NAM’s strong
./reviews/deep_22.md:1490:slightly lower in Rank-5 and Rank-10, PDA demonstrates
./reviews/deep_22.md:1493:Rank-1 by 3.2% and mAP by 1.21%. Remarkably, the gap
./reviews/deep_22.md:1495:a Rank-1 improvement of 7.3%, highlighting PDA’s enhanced
./reviews/deep_22.md:1506:based on Swin B and BERT backbones, PDA shows improvements of 1.54% in Rank-1 and 2.65% in mAP, all while
./reviews/deep_22.md:1629:features through a cascaded amalgamation of modality cooperative complementary learning methods. Likewise, FIENet
./reviews/deep_22.md:1992:Rank-1
./reviews/deep_22.md:1994:Rank-1
./reviews/deep_22.md:2059:Rank-1
./reviews/deep_22.md:2144:Rank-1
./reviews/deep_22.md:2178:Rank-1
./reviews/deep_22.md:2209:in cross-modality person re-identification tasks. On the SYSUMM01 dataset, it achieves a rank-1 accuracy of 79.3% and
./reviews/deep_22.md:2210:mAP of 76.2% in the All-search mode, and 88.7% rank-1
./reviews/deep_22.md:2212:LLCM dataset, PDM achieves a rank-1 accuracy of 57.1%
./reviews/deep_22.md:2247:Rank-1
./reviews/deep_22.md:2251:Rank-1
./reviews/deep_22.md:2303:rank-1 accuracy and 67.3% mAP in the VIS-to-IR mode.
./reviews/deep_22.md:2307:higher rank-1 accuracy and 2.1% higher mAP. In the LLCM
./reviews/deep_22.md:2391:[9] Y. Zhang, S. Zhao, Y. Kang, and J. Shen, “Modality synergy complement learning with cascaded aggregation for visible-infrared person reidentification,” in European Conference on Computer Vision. Springer,
./reviews/deep_22.md:3266:[26] Kiran Misra, Tianying Ji, Andrew Segall, and Frank Bossen, “Video
./reviews/deep_22.md:3338:of mAP and rank-1 accuracy. Code is available at
./reviews/deep_22.md:3580:(θt−1 ), as shown by the dashed arrows in Fig. 1. This ensures adaptive evolution of model parameters while retaining
./reviews/deep_22.md:4334:dynamic parameter fusion, APE provides an adaptive and robust mechanism for lifelong learning, enabling the previous
./reviews/deep_22.md:4427:outperforms existing methods on this dataset, achieving a notably higher mAP and Rank-1 accuracy, which underscores
./reviews/deep_22.md:4529:and rank@1 accuracy (R@1) are used to evaluate the model
./reviews/deep_22.md:4757:improvements in mean Average Precision (mAP) and rank1 accuracy, along with enhanced generalization. These results highlight the potential of PKA as a robust, privacypreserving, and practical solution for LReID.
./reviews/deep_22.md:4804:Wang, Zicheng Liu, and Zheng-jun Zha. Lifelong unsupervised domain adaptive person re-identification with coordinated anti-forgetting and adaptation. In Proceedings
./reviews/deep_22.md:4845:Lifelong person reidentification via adaptive knowledge accumulation. In
./reviews/deep_22.md:5367:CLIP and CoOP to address this limitation, the uncertainty and
./reviews/deep_22.md:5702:our Rank-1 accuracy and mAP surpass those of the best
./reviews/deep_22.md:5710:Rank-1 accuracy exceeds that of CycleTrans by 1.9% (0.5%),
./reviews/deep_22.md:5720:compared to existing generative-based approaches. For instance, in the visible-to-infrared testing mode, RSBA outperforms TSME in Rank-1 accuracy by 7.9% and surpasses ACD
./reviews/deep_22.md:5724:CSDN, RSBA improves the Rank-1 and mAP by 2.1% and
./reviews/deep_22.md:5727:mode, RSBA achieves a Rank-1 accuracy and mAP that are
./reviews/deep_22.md:5729:of 0.9% in Rank-1 accuracy and 1.0% in mAP. These results
./reviews/deep_22.md:5748:complexity and inference time of our framework.
./reviews/deep_22.md:5803:We evaluate the effectiveness of each component in our proposed RSBA, with the results presented in Table 4. The Rank1 and mAP of Baseline (’0’) are 71.9% and 67.6% under the
./reviews/deep_22.md:5805:Effectiveness of ESFA. ESFA aims to introduce explicit semantics to guide the alignment of cross-modality visual features. As shown in Table 4, it improves the Rank-1 and mAP
./reviews/deep_22.md:5809:4, under the single-shot test mode, it improves the Rank-1 accuracy from 76.3% to 77.4%, which indicates that the align-
./reviews/deep_22.md:6108:Table 4, when it is equipped with ESFA, the Rank-1 accuracy
./reviews/deep_22.md:6136:Rank-1
./reviews/deep_22.md:6153:Rank-1
./reviews/deep_22.md:6160:between intra-modality image-text pairs. As shown in Table 5, the direct alignment (1) reduces the Rank-1 accuracy
./reviews/deep_22.md:6163:(2) improves the Rank-1 and mAP by 1.1% and 0.6%. This
./reviews/deep_22.md:6258:in Rank-1 accuracy and 2.7% in mAP under the single-shot
./reviews/deep_22.md:6302:with adaptive weighting for visible-thermal person reidentification. In IJCAI, pages 845–851, 2021.
./reviews/deep_22.md:6409:errors. Clearly, the interrelationship between the bag and other
./reviews/deep_22.md:6507:in the early stage of model training. We aggregate the initial
./reviews/deep_22.md:6763:different features is calculated and ranked. Afterward, it is
./reviews/deep_22.md:6909:that the Hc-Tri loss converges quickly in the early stages,
./reviews/deep_22.md:6940:where cos(·) is the cosine similarity, µ is the adaptive weight.
./reviews/deep_22.md:6968:network to be biased towards early learning samples and affect
./reviews/deep_22.md:6971:loss for coarse guidance in the early stages of training. When
./reviews/deep_22.md:7086:performance, achieving 69.29% in Rank-1, 86.05% in Rank5, and 65.02% in mAP. The following conclusion can be got:
./reviews/deep_22.md:7108:Rank-1, 84.51% in Rank-5, and 64.32% in mAP. This shows
./reviews/deep_22.md:7134:Rank-1 and 5.83% in Rank-5. After the introduction of the
./reviews/deep_22.md:7136:improved with 3.82% in Rank-1, 3.66% in Rank-5, and 4.3%
./reviews/deep_22.md:7178:22:    22	allowing the model to capture uncertainty in the representations. Second, Cross-Modal Containment (CMC) aligns the
./reviews/deep_22.md:7239:269:   269	(θt−1 ), as shown by the dashed arrows in Fig. 1. This ensures adaptive evolution of model parameters while retaining
./reviews/deep_22.md:7270:1023:  1023	dynamic parameter fusion, APE provides an adaptive and robust mechanism for lifelong learning, enabling the previous
./reviews/deep_22.md:7313:935:   935	We evaluate the effectiveness of each component in our proposed RSBA, with the results presented in Table 4. The Rank1 and mAP of Baseline (’0’) are 71.9% and 67.6% under the
./reviews/deep_22.md:7314:937:   937	Effectiveness of ESFA. ESFA aims to introduce explicit semantics to guide the alignment of cross-modality visual features. As shown in Table 4, it improves the Rank-1 and mAP
./reviews/deep_22.md:7324:1240:  1240	Table 4, when it is equipped with ESFA, the Rank-1 accuracy
./reviews/deep_22.md:7327:1292:  1292	between intra-modality image-text pairs. As shown in Table 5, the direct alignment (1) reduces the Rank-1 accuracy
./reviews/deep_22.md:7344:154:   154	(C2FL) strategy and the novel adaptive corner center (Acc)
./reviews/deep_22.md:7458:   499	CLIP and CoOP to address this limitation, the uncertainty and
./reviews/deep_22.md:7793:   834	our Rank-1 accuracy and mAP surpass those of the best
./reviews/deep_22.md:7801:   842	Rank-1 accuracy exceeds that of CycleTrans by 1.9% (0.5%),
./reviews/deep_22.md:7811:   852	compared to existing generative-based approaches. For instance, in the visible-to-infrared testing mode, RSBA outperforms TSME in Rank-1 accuracy by 7.9% and surpasses ACD
./reviews/deep_22.md:7815:   856	CSDN, RSBA improves the Rank-1 and mAP by 2.1% and
./reviews/deep_22.md:7818:   859	mode, RSBA achieves a Rank-1 accuracy and mAP that are
./reviews/deep_22.md:7820:   861	of 0.9% in Rank-1 accuracy and 1.0% in mAP. These results
./reviews/deep_22.md:7858:   339	uncertainty when only the first two moments are known. Crucially, the mathematical closure and analytical tractability of
./reviews/deep_22.md:7911:   392	texts with low uncertainty is primarily determined by the
./reviews/deep_22.md:7913:   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./reviews/deep_22.md:7940:   421	uncertainty-aware approaches that treat variance merely as a
./reviews/deep_22.md:7952:   433	where µvi and µt j denote the mean features, and σ2vi , σ2t j represent variances that capture feature uncertainty and semantic
./reviews/deep_22.md:8159:   640	enhances robustness by ensuring masked distributions encompass unmasked ones, reflecting increased uncertainty in partial
./reviews/deep_22.md:8210:   691	exhibit greater uncertainty, thereby naturally encompassing the
./reviews/deep_22.md:8229:   710	introducesrd mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./reassess2/x_2.md:5420:| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
./reassess2/x_2.md:5428:- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
./reassess2/x_2.md:5429:- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。
./reassess2/x_2.md:5438:| (b) ReID k-reciprocal / neighbor re-ranking | **存活，但需强对照** | 这类已经把 ReID 当邻域拓扑问题处理，但在**测试后处理**改距离/排名，不是训练单 embedding；也不是把少数 gallery 的跨 ID 负 in-degree 当失败主变量。 | k-reciprocal 用 reciprocal NN、Jaccard distance、local query expansion；ECN / PSE 也是 unsupervised re-ranking。链接：[Zhong k-reciprocal](https://arxiv.org/abs/1701.08398), [PSE/ECN](https://arxiv.org/abs/1711.10378) | 8 |
./reassess2/x_2.md:5439:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./reassess2/x_2.md:5440:| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
./reassess2/x_2.md:5448:- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
./reassess2/x_2.md:5449:- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。
./reassess2/x_2.md:5471:**meta-learning(关键)**: 团队frozen-feature+现有数据的隐藏变量候选, 反复被成熟ReID test-time工具(k-reciprocal/camera-aware re-ranking/hard-negative)碾压或证伪。ReID field成熟, 明显的retrieval-side/topology-side隐藏变量都被现成后处理占了。
./reassess2/x_2.md:5480: exited 1 in 0ms:
./reassess2/x_2.md:5489: exited 1 in 0ms:
./reassess2/x_2.md:5492: exited 1 in 0ms:
./reassess2/x_2.md:5495: exited 1 in 0ms:
./reassess2/x_2.md:5636:| 062-eq | PSG+GCN + LKU (eq_concat) | 60.7% | 71.2% | 84.1% | 87.4% | mAP+4.1%, R1+4.7% | ❌ 负面。vs 3-seed: -0.03%/-1.37%。Learned uncertainty 损害 R1 |
./reassess2/x_2.md:5656:| 078-eq | PSG+GCN+PAA+APG (eq) | 60.5% | 72.5% | 84.3% | 87.9% | mAP+3.9%, R1+6.0% | ❌ vs PAA: -1.1%/-1.7%。adaptive gate 负面 |
./reassess2/x_2.md:5697:**meta-learning(关键)**: 团队frozen-feature+现有数据的隐藏变量候选, 反复被成熟ReID test-time工具(k-reciprocal/camera-aware re-ranking/hard-negative)碾压或证伪。ReID field成熟, 明显的retrieval-side/topology-side隐藏变量都被现成后处理占了。
./reassess2/x_2.md:5706: exited 1 in 0ms:
./reassess2/x_2.md:5762:1. NFC/Re-ranking 等 test-time 方法所有 SOTA 都可以用，不算公平的对比
./reassess2/x_2.md:5969:### #3 Time-Conditioned 跨年ReID(anytime)— 任务重定义故事最完整
./reassess2/x_2.md:5972:- 风险: DeepChange需签学术协议; AT-USTC可得性不确定。anytime刚被AT-ReID(2509.16635,IJCAI25)命名窗口刚开。
./reassess2/x_2.md:5987:例: ①频率解耦(VI)→空地VIR视频; ②RGB/CLIP特权蒸馏→event-only(隐私); ③diffusion生成(SD-ReID)→anytime(time-counterfactual); ④middle-image(MTRL灰图)→contour/sketch=ScribbleBridge。
./reassess2/x_2.md:6250: exited 1 in 0ms:
./reassess2/x_2.md:6485:web search: person re-identification embedding anisotropy feature spectrum effective rank hubness
./reassess2/x_2.md:6499:web search: k-reciprocal encoding re-ranking person re-identification CVPR 2017 arxiv 1701.08398
./reassess2/x_2.md:6535:   - 提高 effective rank；
./reassess2/x_2.md:6549:   - effective rank；
./reassess2/x_2.md:6555:   - 训练后期 loss 继续降，但 effective rank 下降 / top-PC energy 上升；
./reassess2/x_2.md:6560:   - 若 rank/collapse 指标不能解释 AP residual，判死；
./reassess2/x_2.md:6567:  - effective rank 至少上升；
./reassess2/x_2.md:6607:   - 提高 effective rank；
./reassess2/x_2.md:6621:   - effective rank；
./reassess2/x_2.md:6627:   - 训练后期 loss 继续降，但 effective rank 下降 / top-PC energy 上升；
./reassess2/x_2.md:6632:   - 若 rank/collapse 指标不能解释 AP residual，判死；
./reassess2/x_2.md:6639:  - effective rank 至少上升；
./reviews/deep_20.md:60:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_20.md:77:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_20.md:80:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_20.md:89:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_20.md:121:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_20.md:151:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_20.md:327:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_20.md:405:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_20.md:480:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_20.md:622:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_20.md:750:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_20.md:779:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_20.md:782:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_20.md:783:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_20.md:860:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_20.md:864:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_20.md:878:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_20.md:884:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_20.md:975:**方向（用户决策#33=选项1）**：放开 SOLIDER 强主干，回常规 ImageNet 预训练主干(TransReID ViT-base、ImageNet 预训练 Swin-Tiny)刷标准遮挡 ReID mAP/rank1。可靠性/conformal 线归档(决策#28-#32 的负结果+诊断作分析素材)。
./reviews/deep_20.md:1065:   492	clearly, demonstrating the effectiveness of CB loss in improving feature
./reviews/deep_20.md:1088:   151	task complexity under certain conditions. This framework demonstrates
./reviews/deep_20.md:1144:  1295	surpasses the previous best method FMGNet [46] by 0.5% Rank-1
./reviews/deep_20.md:1145:  1301	best method, APD [47] by 2.0% Rank-1. In the labeled setting, we
./reviews/deep_20.md:1160:   177	model, we propose an adaptive-weight method to aggregate
./reviews/deep_20.md:1170:   295	a multi-scale query-adaptive convolution framework to encode
./reviews/deep_20.md:1235:   182	noise; however, our method introduces an adaptive, multi-scale fusion mechanism that allows
./reviews/deep_20.md:1237:   188	our approach enables more flexible, adaptive decision-making, dynamically refining features to
./reviews/deep_20.md:1266:   441	module dynamically generates convolution kernels and biases for adaptive processing of various
./reviews/deep_20.md:1371:   274	AGC component employs a dynamic convolution kernel that is adjusted adaptively on the basis
./reviews/deep_20.md:1435:   338	computational complexity and memory requirements as the input sequence length increases [13].
./reviews/deep_20.md:1476:   379	via adaptive pooling, which compresses the spatial dimension to 𝐾 2 . Moreover, to increase the
./reviews/deep_20.md:1477:   380	feature representation power and reduce the computational complexity, the input is then fed to the
./reviews/deep_20.md:1523:   426	where 𝑆𝑏 ∈ R𝐵×𝐺 × 2 is the generated adaptive bias weight and where 𝑏 ∈ R𝐺 × 2 is the initial
./reviews/deep_20.md:1538:   441	module dynamically generates convolution kernels and biases for adaptive processing of various
./reviews/deep_20.md:1896:   669	fa ]). Clearly, such intra-submodel optimization is unable to
./reviews/deep_20.md:2011:   784	learn adaptive weights of each submodel. Given the testing
./reviews/deep_20.md:2020:   793	where D(·, ·) represents the Euclidean distance. And its adaptive weight wat can be computed by:
./reviews/deep_20.md:2114:   168	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
./reviews/deep_20.md:2139:   317	are used to evaluate the performance of the method. In particular, CMC (rank-k matching accuracy)
./reviews/deep_20.md:2171:    23	on more fine-grained features through an adaptive convolutional generation structure with multiple
./reviews/deep_20.md:2274:   126	strategy and local averaging technique, this method enhances feature alignment and similaritybased feature refinement, similar in effect to the re-ranking [57] technique. Specifically, given a set
./reviews/deep_20.md:2316:   168	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
./reviews/deep_20.md:2465:   317	are used to evaluate the performance of the method. In particular, CMC (rank-k matching accuracy)
./reviews/deep_20.md:2499:   351	while our proposed MDFN performs on par with GGMMNet in Rank-1, we achieve a significant
./reviews/deep_20.md:2501:   353	TMAL, surpassing DEEN by 0.40% in Rank-1 and 4.42% in mAP, and DGRNet by 4.52% in Rank-1
./reviews/deep_20.md:2502:   354	and 2.26% in mAP. For the indoor-search mode, MDFN provides impressive results with a Rank-1
./reviews/deep_20.md:2503:   355	accuracy of 81.46% and an mAP of 85.45%. Compared with DEEN, MDFN improves Rank-1 by 1.16%
./reviews/deep_20.md:2504:   356	and mAP by 2.15%, and outperforms TRR by 0.14% and 2.23% in Rank-1 and mAP, respectively. These
./reviews/deep_20.md:2506:   358	over existing methods and provides the best performance in terms of Rank-1 accuracy and mAP
./reviews/deep_20.md:2512:   364	MDFN achieves a Rank-1 accuracy of 91.28% and an mAP of 92.15%. In the IR-to-VIS mode, the
./reviews/deep_20.md:2513:   365	Rank-1 accuracy is 90.03% and the mAP is 91.20%. Although the MDFN’s Rank-1 accuracy in both
./reviews/deep_20.md:2515:   367	performance gap may be attributed to the smaller scale and lower complexity of the RegDB dataset.
./reviews/deep_20.md:2585:   317	to capture and adaptively integrate hierarchical semantics
./reviews/deep_20.md:2947:   679	the prediction space, achieving adaptive alignment across
./reviews/deep_20.md:3065:    33	gain in same-year evaluations. The HAR-enhanced models also exhibit faster convergence and higher Rank-1
./reviews/deep_20.md:3512:   480	adds structural complexity to the network of relationships by connecting more location pairs per individual rather than inﬂating the dataset
./reviews/deep_20.md:3539:   507	complexity and cost: XS applies multiple scaling stages, S reduces the
./reviews/deep_20.md:3560:   528	and localized temporal-spatial dynamics. However, its added complexity does not always yield consistent improvements in real-world scenarios. GLN Phase 1 introduces a grouped latent representation strategy
./reviews/deep_20.md:3568:   536	GREW-which vary signiﬁcantly in terms of scale, environmental conditions, and overall complexity. OU-MVLP [15] is a large-scale constrained
./reviews/deep_20.md:3576:   544	as GLN Phase 2 or GaitBase on GREW can be computationally demanding due to the dataset’s scale and complexity. Consequently, model training choices are inﬂuenced by dataset characteristics and the architectural robustness and scalability of the gait backbones.
./reviews/deep_20.md:3629:   597	RP1_20 and RP2_23. Metrics reported as mAP / Rank-1 / Rank-5. Baseline
./reviews/deep_20.md:3904:   492	clearly, demonstrating the effectiveness of CB loss in improving feature
./reviews/deep_20.md:4043:   631	Fig. 5. Diagram of feature fusion, feature cascade and our proposed ODE-FR.
./reviews/deep_20.md:4051:   639	range corresponds to the layer that can be selected adaptively,
./reviews/deep_20.md:4169:   757	average precision (mAP) and CMC rank-1, rank-5, rank-10
./reviews/deep_20.md:4265:   150	as feature fusion and cascade, the residual structure can be
./reviews/deep_20.md:4361:   151	task complexity under certain conditions. This framework demonstrates
./reviews/deep_20.md:4391:   181	that reduces the complexity of occluded person ReID without requiring
./reviews/deep_20.md:4447:   237	highlights early noise-aware representation learning. SFE-Net [17] provides a complementary view on robustness under sparse and noisy
./reviews/deep_20.md:4695:   462	measures the uncertainty or complexity of the tasks. The mutual information between the task and the injected noise is given by
./reviews/deep_20.md:4716:   483	task by reducing uncertainty rather than introducing disturbance.
./reviews/deep_20.md:4775:  1295	surpasses the previous best method FMGNet [46] by 0.5% Rank-1
./reviews/deep_20.md:4777:  1297	achieves 96.1% Rank-1 accuracy, outperforming all existing state-ofthe-art methods. On DukeMTMC-ReID, we achieve comparable Rank-1
./reviews/deep_20.md:4781:  1301	best method, APD [47] by 2.0% Rank-1. In the labeled setting, we
./reviews/deep_20.md:4783:  1303	by 3.0% in Rank-1 accuracy and 2.8% in mAP.
./reviews/deep_20.md:4786:  1306	3.1% in both Rank-1 accuracy and mAP (see Table 7). Remarkably,
./reviews/deep_20.md:4787:  1307	our NIReID+R50 also outperforms TransReID [8] by 1.0% in Rank-1
./reviews/deep_20.md:4809:  1329	Rank-1
./reviews/deep_20.md:4811:  1331	Rank-5
./reviews/deep_20.md:4813:  1333	Rank-10
./reviews/deep_20.md:5017:    88	incorporates a local unit that uses adaptive group convolution (AGC) to capture local features,
./reviews/deep_20.md:5046:   117	— Extensive experiments demonstrate that our proposed MDFN outperforms the existing stateof-the-art VI-ReID methods overall in terms of the Rank-1 and mAP metrics on the commonly
./reviews/deep_20.md:5074:   102	background changes, further increasing task complexity.
./reviews/deep_20.md:5102:   130	of VI-ReID into two aspects: (i) difficulty in aligning lowand high-level semantics across modalities, and (ii) distribution discrepancies between VIS and IR features. To address
./reviews/deep_20.md:5105:   133	Augmentation (MIA) module. DLF captures and adaptively
./reviews/deep_20.md:5154:   177	model, we propose an adaptive-weight method to aggregate
./reviews/deep_20.md:5233:   274	AGC component employs a dynamic convolution kernel that is adjusted adaptively on the basis
./reviews/deep_20.md:5255:   613	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
./reviews/deep_20.md:5270:   829	Table 1. Comparison of Rank-1 (%) and mAP (%) Performances with the State-of-the-art Methods on
./reviews/deep_20.md:5272:  1160	DPE modules, capturing local and global features in a robust and adaptive manner. This multi-scale
./reviews/deep_20.md:5278:  1209	Fig. 6. Some Rank-10 retrieval results obtained by the baseline and the proposed MDFN on SYSU-MM01
./reviews/deep_20.md:5281:  1213	Baseline DFF DPE FRS Rank-1 mAP
./reviews/deep_20.md:5282:  1269	Retrieval results. To better reflect the superiority of our approach, we visualize the Rank-10
./reviews/deep_20.md:5288:  1283	baseline, DFF improves the Rank-1 accuracy by 5.50% and the mAP by 5.78%. DPE improves the
./reviews/deep_20.md:5293:  1292	shown in Table 2(b), L𝑀𝐻𝐶 improves the Rank-1 accuracy by 2.55% and mAP by 2.71%. L𝑂𝐺𝑀
./reviews/deep_20.md:5304:  1333	results are presented in Figure 8(a). Our DPE module achieves an improvement of 0.74% in Rank-1
./reviews/deep_20.md:5308:  1345	module is incorporated, the DPE module further achieves a 0.75% improvement in Rank-1 accuracy
./reviews/deep_20.md:5332:    33	gain in same-year evaluations. The HAR-enhanced models also exhibit faster convergence and higher Rank-1
./reviews/deep_20.md:5411:   544	as GLN Phase 2 or GaitBase on GREW can be computationally demanding due to the dataset’s scale and complexity. Consequently, model training choices are inﬂuenced by dataset characteristics and the architectural robustness and scalability of the gait backbones.
./reviews/deep_20.md:5441:  1275	insight into where HAR contributes most through early rank improvements (e.g., Rank-1 and Rank-5) or enhanced overall retrieval consistency.
./reviews/deep_20.md:5444:  1283	the HAR-enhanced models in the ﬁnal ranks. AlignedReID performs the
./reviews/deep_20.md:5446:  1290	weaker early-rank performance than the HAR models (29.4 % Rank-1)
./reviews/deep_20.md:5448:  1295	boosts early-rank retrieval performance while contributing to more stable performance across higher ranks. The performance gap between
./reviews/deep_20.md:5455:  1307	solid performance (35.3 % Rank-1), trails behind the HAR models at
./reviews/deep_20.md:5459:  1317	blue), adding HAR consistently improves performance across all metrics, particularly in mAP and Rank-1 accuracy. In the RP1_20 → RP2_23
./reviews/deep_20.md:5461:  1328	Rank-1 improves from 35.3 % to 52.9 % in the RP3_23 → RP1_20 direction. Other strong HAR models like SLF_8x8𝑁𝐿𝑁 , SLF_4x16𝑁𝐿𝑁 , and
./reviews/deep_20.md:5463:  1330	We also present CMC curves to analyze how HAR integration impacts person Re-ID performance. These curves visualize the rank-based
./reviews/deep_20.md:5468:  1348	are more substantial in these highly unconstrained, long-term scenarios. HAR-enhanced models oﬀer better Rank-1 and Rank-5 performance
./reviews/deep_20.md:5475:  1381	Short-term Re-ID using only gait features. Performance on same-year crosslocation evaluations (probe → gallery). Metrics reported as mAP / Rank-1 /
./reviews/deep_20.md:5481:  1414	mAP and 76.5 % Rank-1 in the reverse direction. When HAR is added,
./reviews/deep_20.md:5482:  1418	at 84.7 % mAP and 76.5 % Rank-1 in reverse. Other top-performing HAR
./reviews/deep_20.md:5494:  1594	100.0 %, indicating weaker retrieval consistency at higher ranks compared to HAR-enhanced models.
./reviews/deep_20.md:5496:  1599	While GaitGL_G maintains a strong baseline (76.5 % Rank-1), it converges more slowly than the HAR-fused models. AlignedReID again performs the weakest at early and mid ranks, achieving only 64.7 % Rank-1
./reviews/deep_20.md:5497:  1601	These results conﬁrm that HAR integration improves early retrieval
./reviews/deep_20.md:5504:  1757	insight into where HAR contributes most through early rank improve-
./reviews/deep_20.md:5508:  1762	gait and HAR embeddings. Cross-attention adaptively emphasizes behavioral cues conditioned on gait, while cosine similarity enforces alignment by maximizing directional closeness between modalities. In contrast, concatenation stacks the embeddings without explicit interaction,
./reviews/deep_20.md:5519:  1795	Our fused gait-HAR model outperformed appearance-based and gaitonly baselines, with +12 % mAP in long-term and +11.6 % in shortterm setups. HAR-enhanced models also improved early-rank retrieval,
./reviews/deep_20.md:5594:   956	Table III. Specifically, our method achieves 49.4% rank-1 and
./reviews/deep_20.md:5615:  1057	the SSP achieves improvements of 2.0% mAP and 1.9% rank1. The results validate that the collaborative learning among
./reviews/deep_20.md:5620:  1068	0.9% and 1.6% rank-1 respectively. And adding the PGD can
./reviews/deep_20.md:5670:  1272	large-scale domain as shown in Fig. 12(b) and stopping early
./reviews/deep_20.md:5682:  1391	[12] S. Liao and L. Shao, “Interpretable and generalizable person reidentification with query-adaptive convolution and temporal lifting,” in
./reviews/deep_20.md:5713:   133	Augmentation (MIA) module. DLF captures and adaptively
./reviews/deep_20.md:5793:  1447	on the baseline model. Specifically, the proposed DLF module (Index-2) improves the Rank-1 accuracy by 1.86% and
./reviews/deep_20.md:5797:  1454	between visible and infrared modalities. By individually introducing the Lhsm (Index-4) and Lesm (Index-5), Rank-1
./reviews/deep_20.md:5828:  1563	Fig. 8: The Rank-10 retrieval results obtained by the baseline
./reviews/deep_20.md:5857:    52	difficulty stems from dual occlusion patterns: (1) object-induced occlusion, where environmental elements (e.g., vehicles, buildings) obstruct
./reviews/deep_20.md:5863:    87	Fig. 1. Our noise injection enhances the Rank-1 accuracy on Occluded-Duke [3] of various vision backbones without additional manual annotations.
./reviews/deep_20.md:6002:  1272	the baseline achieves 61.3% Rank-1 accuracy, which is modestly enhanced to 62.3% with ENI alone (+1.0%), significantly improved to
./reviews/deep_20.md:6012:  1306	3.1% in both Rank-1 accuracy and mAP (see Table 7). Remarkably,
./reviews/deep_20.md:6180:   492	clearly, demonstrating the effectiveness of CB loss in improving feature
./reviews/deep_20.md:6205:   631	Fig. 5. Diagram of feature fusion, feature cascade and our proposed ODE-FR.
./reviews/deep_20.md:6237:   865	0.8% and 0.5% improvements on mAP and Rank-1, respectively. Finally, the ODE-FR effectively reconstructs the feature
./reviews/deep_20.md:6266:   953	ODEs [57], which is based on adaptive time steps, suffers from
./reviews/deep_20.md:6370:4. 证据闭环：Table III 做 DLF、MIA、HSM、ESM 的组件消融，Table IV 验证 DLF 和 MIA 放在哪些层最合适。还有 α 和 τ 的参数分析、VIS-IR cosine distance 分布、t-SNE、Grad-CAM 和 Rank-10 检索可视化。它的证据闭环是证明双空间不只是换名词，确实让跨模态分布更紧、注意区域更对。
./reviews/deep_20.md:6447:4. 证据闭环：Table III 做 DLF、MIA、HSM、ESM 的组件消融，Table IV 验证 DLF 和 MIA 放在哪些层最合适。还有 α 和 τ 的参数分析、VIS-IR cosine distance 分布、t-SNE、Grad-CAM 和 Rank-10 检索可视化。它的证据闭环是证明双空间不只是换名词，确实让跨模态分布更紧、注意区域更对。
./analyses/deep_2.txt:11:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./analyses/deep_2.txt:43:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./analyses/deep_2.txt:47:4. **证据闭环**：关键证据很直接。APE 单独带来大幅提升，在 Occluded-Duke 上 mAP 和 Rank-1 分别涨 18.4 和 16.9；OAM 再带来 2.6 和 2.9。部位数实验说明 4 个部位最合理，太多会冗余，6 个会忽略左右区分。最有说服力的是可视化，APE 在姿态标签错误时仍能抽对区域，能避开伞遮住的头部，还能抓住姿态估计忽略的手提包。这是在证明“自适应区域优于姿态输出”，不是只证明最终分数涨。
./analyses/deep_4.txt:15:**2. Bridging the gap: Learning adaptive knowledge transition for lifelong person re-identification**
./analyses/deep_4.txt:19:2. 重定义动作：他们把 catastrophic forgetting 重定义成 adaptive knowledge transition 问题。旧说法是 preserve previous knowledge，新说法是 control the degree and direction of knowledge distillation。关键词从 distillation、forgetting，换成 transition path、intermediate knowledge、adaptive transfer。这样一来，论文的中心不再是“再加一个蒸馏损失”，而是“旧知识和新知识之间需要一座可学习的桥”。
./analyses/deep_4.txt:27:**3. Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification**
./analyses/deep_4.txt:31:2. 重定义动作：他们把无监督 ReID 从“伪标签不准”重定义成“多相机子域之间缺少语义传播”。每个 camera 被定义成一个 domain，训练不只是 cluster contrastive learning，而是 camera-aware multi-domain adaptive learning。关键词是 camera-aware graph、multi-domain、semantic propagation、domain-invariant representation。
./analyses/deep_4.txt:71:4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。
./novelty_hubness.md:16:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'gallery 负向 in-degree 定义成 ReID 失败主变量 + 训练端 anti-hub': (a)hubness in 高维检索(Radovanovic hubness / CSLS / Mutual Proximity / local scaling)用到 ReID 没; (b)k-reciprocal re-ranking / neighbor-based re-ranking ReID; (c)hard-negative mining / anti-hub margin / proxy metric learning; (d)Pose2ID / feature centralization / neighbor aggregation ReID; (e)graph/relation ReID(SGGNN等); (f)long-tail / popularity-bias / false-match-rate retrieval。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./novelty_hubness.md:32:web search: person re-identification re-ranking k-reciprocal encoding CVPR 2017
./novelty_hubness.md:100:| (b) ReID k-reciprocal / neighbor re-ranking | **存活，但需强对照** | 这类已经把 ReID 当邻域拓扑问题处理，但在**测试后处理**改距离/排名，不是训练单 embedding；也不是把少数 gallery 的跨 ID 负 in-degree 当失败主变量。 | k-reciprocal 用 reciprocal NN、Jaccard distance、local query expansion；ECN / PSE 也是 unsupervised re-ranking。链接：[Zhong k-reciprocal](https://arxiv.org/abs/1701.08398), [PSE/ECN](https://arxiv.org/abs/1711.10378) | 8 |
./novelty_hubness.md:101:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./novelty_hubness.md:102:| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
./novelty_hubness.md:110:- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
./novelty_hubness.md:111:- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。
./novelty_hubness.md:120:| (b) ReID k-reciprocal / neighbor re-ranking | **存活，但需强对照** | 这类已经把 ReID 当邻域拓扑问题处理，但在**测试后处理**改距离/排名，不是训练单 embedding；也不是把少数 gallery 的跨 ID 负 in-degree 当失败主变量。 | k-reciprocal 用 reciprocal NN、Jaccard distance、local query expansion；ECN / PSE 也是 unsupervised re-ranking。链接：[Zhong k-reciprocal](https://arxiv.org/abs/1701.08398), [PSE/ECN](https://arxiv.org/abs/1711.10378) | 8 |
./novelty_hubness.md:121:| (c) hard-negative mining / anti-hub margin / proxy metric learning | **部分撞车，杀伤最大** | ReID hard mining/HAP2S/MS loss 主要按 anchor-local difficulty/相似度加权，不是全局误吸附 in-degree。**但 HAL 和 NeighborRetr 已经非常接近“训练端 hubness-aware loss + memory/global stats”**，只是任务是图文/跨模态检索，不是 person ReID。 | HAL 用 memory bank 和 kNN query 统计 hub，并在训练 loss 中加权；NeighborRetr 明确 bad/good hubs、centrality、memory bank、训练端缓解 hubness。链接：[HAL](https://arxiv.org/abs/1911.10097), [NeighborRetr](https://arxiv.org/abs/2503.10526), [HAP2S ReID](https://arxiv.org/abs/1807.11206), [MS Loss](https://arxiv.org/abs/1904.06627) | 9 |
./novelty_hubness.md:122:| (d) Pose2ID / feature centralization / neighbor aggregation ReID | **存活** | Pose2ID 是同身份 feature centralization：生成多姿态同 ID，或从邻域找潜在正样本做特征聚合。它处理的是 positive aggregation / noise reduction，不是“负向 gallery hub 吸附很多 query”。 | Pose2ID 摘要和方法写的是 training-free feature centralization、neighbor feature centralization、保留原分布以兼容 re-ranking。链接：[Pose2ID](https://arxiv.org/abs/2503.00938) | 8 |
./novelty_hubness.md:130:- **可存活切口**：必须把贡献压在 **ReID-specific gallery topology failure**：closed-set/gallery 中“少数图像成为跨身份 query 的误吸附点”，用 **cross-ID top-k in-degree / hub mass** 作为诊断变量和训练信号；测试仍是单 embedding，不做 CSLS/k-reciprocal/re-ranking。
./novelty_hubness.md:131:- **必须做的对照**：baseline triplet/CE、batch-hard/HAP2S 或 MS-style local hard negative、k-reciprocal 仅作为 post-hoc reference、HAL/NeighborRetr 思路的简化移植强对照。这样才能证明不是普通 hard negative，也不是 test-time re-ranking。
./analyses/deep_19.txt:59:4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。
./analyses/deep_19.txt:67:2. 重定义动作：旧问题是“怎么建模视频时间特征”；新问题是“怎么按身体部位自适应选择不同长度的时间线索”。关键词是 dynamic hierarchical graph、adaptive multi-granularity temporal clues、part-consistent graph、mutual nearest neighbor、similarity weighted inference。
./debate/d_2.md:16:**Gallery Hubness re-frame**: 强 ReID 失败不是 pairwise 相似度病, 而是少数 gallery 样本成为很多不同身份 query 的误吸附点(gallery 负向 in-degree / hub mass)=many-to-one 图库拓扑病。机制设想: 训练端 anti-hub embedding(memory bank 存 cross-ID in-degree, anti-hub margin, 负样本权重从'离anchor近'改'是否全局误吸附点', 测试仍单 embedding 不 re-rank)。
./debate/d_2.md:25:角色=**蓝队(降级 Hubness)**。为'别投训练, 降级转 r_2 备胎 Rank-Instability'辩护: D2 是铁证——k-reciprocal/camera **免费 test-time** 就在 mAP 上赢, 一个训练版要 beat camera-aware k-reciprocal 是高 bar 且小 headroom(Market 饱和)。用联网查: (a)hubness-aware training / anti-hub margin 历史上是否真能 beat 强 re-ranking(还是总被 k-reciprocal 这类盖过); (b)R1 赢 mAP 输是不是只是把 ranking 重排没真增加判别信息; (c)团队反复投不确定训练的教训。论证'诊断真≠方法能发', 该把诊断当一个 observation 写进别的稿/换 r_2。信心 1-10。
./debate/d_2.md:47:web search: TransReID Market1501 mAP rank1 re-ranking k-reciprocal
./debate/d_2.md:55:web search: person re-identification camera-aware re-ranking same camera penalty Market1501 mAP
./debate/d_2.md:73:web search: Market-1501 dataset mAP evaluates all correct matches person re-identification multiple ground truth rank-1 not enough
./debate/d_2.md:77:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./debate/d_2.md:79:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./debate/d_2.md:84:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./debate/d_2.md:88:   k-reciprocal re-ranking 本来就是 ReID 图库拓扑修正：无标注、无训练、对任意初始排序可用；原论文在 Market/MARS/CUHK03/PRW 上展示大 mAP 增益，例如 IDE(R)+KISSME 在 Market 上 mAP 从 49.05 到 63.63，论文自己强调 mAP 提升显著。你的 hub score 本质也是图库图结构惩罚，但零训练 mAP 被 k-reciprocal 覆盖，这直接把训练版的证明门槛抬高。  
./debate/d_2.md:92:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./debate/d_2.md:96:   ReID 的 mAP 看的是多正样本整体排序。hub penalty 让 R1 `+1.13`，但 mAP 只 `+0.31` 且输给 k-reciprocal/camera，说明它可能只是把少数 false-top1 hub 压下去，没有系统性把所有 positives 拉上来。训练版也许能扩大这个效应，但当前证据更像“rank head repair”，不是“embedding 判别力提升”。
./debate/d_2.md:112:- 主线切 Rank-Instability，因为它更可能解释“强 embedding 下排序不稳定/邻域拓扑变化”的方法空间，而不是在已被免费 re-ranking 覆盖的 hub penalty 上硬训。
./debate/d_2.md:117:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./debate/d_2.md:119:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./debate/d_2.md:124:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./debate/d_2.md:128:   k-reciprocal re-ranking 本来就是 ReID 图库拓扑修正：无标注、无训练、对任意初始排序可用；原论文在 Market/MARS/CUHK03/PRW 上展示大 mAP 增益，例如 IDE(R)+KISSME 在 Market 上 mAP 从 49.05 到 63.63，论文自己强调 mAP 提升显著。你的 hub score 本质也是图库图结构惩罚，但零训练 mAP 被 k-reciprocal 覆盖，这直接把训练版的证明门槛抬高。  
./debate/d_2.md:132:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./debate/d_2.md:136:   ReID 的 mAP 看的是多正样本整体排序。hub penalty 让 R1 `+1.13`，但 mAP 只 `+0.31` 且输给 k-reciprocal/camera，说明它可能只是把少数 false-top1 hub 压下去，没有系统性把所有 positives 拉上来。训练版也许能扩大这个效应，但当前证据更像“rank head repair”，不是“embedding 判别力提升”。
./debate/d_2.md:152:- 主线切 Rank-Instability，因为它更可能解释“强 embedding 下排序不稳定/邻域拓扑变化”的方法空间，而不是在已被免费 re-ranking 覆盖的 hub penalty 上硬训。
./analyses/ALL_ANALYSES.txt:29:**3. 4SNet - Spatial and Spectrum Self-adaptive Synergy Network**
./analyses/ALL_ANALYSES.txt:33:2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。
./analyses/ALL_ANALYSES.txt:73:4. **证据闭环**：它用方法表、数据集表、参数量和 Rank-1 对比、benchmark/protocol 汇总、挑战清单和 3DGait 医疗任务 case study 来证明 SRID 不是零散技巧，而是有共同问题、共同评价和外溢应用的方向。
./analyses/ALL_ANALYSES.txt:155:**2. Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification**
./analyses/ALL_ANALYSES.txt:159:2. 重定义动作：他们没有泛泛说“伪标签有噪声”，而是把噪声拆成三个更具体的问题：距离度量不懂相机，代理点不符合 DBSCAN 的局部密度结构，监督目标太硬。关键词是 dynamic adaptive camera Jaccard、manifold、medoid proxy、dynamic adaptive proxies、soft pseudo labels。
./analyses/ALL_ANALYSES.txt:187:4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。
./analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./analyses/ALL_ANALYSES.txt:207:2. 重定义动作：它把“缩小 modality gap”重定义成“增强模型对 modality、input instance、query instance 的适应能力”。关键词是 model adaptability、flexibility、modality-aware prompts、instance-aware prompts、query-guiding prompts、adaptive feature space。这个转向很重要，因为它避开了“把两种模态硬映射到同一空间”的旧叙事。
./analyses/ALL_ANALYSES.txt:229:**证据闭环：** 关键证据是组件消融：空间分支、时间分支、STIM、FA 逐步加上都有收益；再用 RCB、自注意力、交叉注意力对比 FEM/TIM；还有模型大小和 Rank-1 的效率图，说明不是靠堆参数。弱点是它没有特别强地证明“一半帧就足够”这个核心重定义，只是间接由效率和性能成立。
./analyses/ALL_ANALYSES.txt:247:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./analyses/ALL_ANALYSES.txt:325:3. **机制怎么长出来**：CGS 每个 epoch 用 Spherical-LSH 先把相近类别放进 bucket，再用 feature-map adaptive matching 找邻近类别，构造相关 hard batch。GSANet 保持高分辨率流，GRSR 用稀疏全局相关像素重构特征，提升特征质量。特征更好会让下一轮 CGS 更好，CGS 更好又提供更有价值样本。这个“互相增强”的逻辑成立，但模块数量较多，绑定不如 GAReID 干净。
./analyses/ALL_ANALYSES.txt:326:4. **证据闭环**：采样对比显示 CGS 好于 random、PK、cluster，并且比已有 GS 快很多，Market 上采样 0.1 秒对 4 秒，MSMT(all) 上 1 秒对 40 秒。CGS 相比 Cluster 在 Market 到 CUHK03 上提升约 3.0 Rank-1 和 2.8 mAP，在 Market 到 MSMT17 上提升约 3.9 和 2.9。CGS、GRSR、GSANet 的组合消融显示完整模型最好。采样可视化证明 CGS 找到衣服颜色、局部外观相近的邻近类别。
./analyses/ALL_ANALYSES.txt:330:1. **触发观察**：ReID 是 open-set ranking，分类攻击不能直接搬。已有 ReID 攻击黑盒迁移差，作者观察到早期迭代里 current gradient 和 global aggregated gradient 一致性很低，更新方向随机震荡，导致迁移性不足。
./analyses/ALL_ANALYSES.txt:345:4. 证据闭环：关键不是只报涨点，而是拆开证明两个判断。BRT 上 Graph-MLP 基线 46.49 Rank-1、44.97 mAP，加 GstRM 到 58.60、55.31，说明可达性约束确实过滤了伪边；加 GGM 到 64.71、61.67，说明传播门控也有用；两者合到 71.06、68.81。t-SNE 进一步显示同身份更紧、异身份更分开。
./analyses/ALL_ANALYSES.txt:359:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
./analyses/ALL_ANALYSES.txt:380:4. 证据闭环：组件消融显示 SBS、SCT、HPL 逐步提升，HPL 内部的 SPL、MPL、S2MPL 也分别有效。PRCC 换衣设置达到 74.3 Rank-1，VC-Clothes 换衣设置达到 84.9 Rank-1。特征图显示方法更关注头、脸、手臂等非衣服区域，这正好回应“衣服捷径被压下去”。
./analyses/ALL_ANALYSES.txt:472:4. 证据闭环：消融非常贴问题定义。CCE 在 SYSU-MM01 和 RegDB 上 Rank-1 分别约 +6 和 +7；BCLM 约 +3 Rank-1 和 +3 mAP；GPRA 加 LDRHL 在 SYSU-MM01 上带来约 +17 Rank-1 和 +14 mAP。t-SNE 也不是泛泛展示聚类更好，而是专门显示已有方法虽然模态靠近了，但噪声标签还在，而 HCCL 缓解了这个问题。
./analyses/ALL_ANALYSES.txt:480:4. 证据闭环：MGFE 在 SYSU-MM01 两种模式下 Rank-1 分别约 +6.26 和 +4.47；MGFT 和 CNC 继续提升；三粒度组合明显优于单粒度或两粒度。热图也服务于视角，global 关注大区域，patch 关注轮廓和局部，pixel 捕捉姿态和手臂动作，组合后覆盖更完整语义。
./analyses/ALL_ANALYSES.txt:556:4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。
./analyses/ALL_ANALYSES.txt:570:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
./analyses/ALL_ANALYSES.txt:578:4. **证据闭环**：最关键的消融是和 KNN、相似度阈值、MPLP 比。Market 上相似度阈值是 72.8 Rank-1、39.9 mAP，MPRD 到 83.0、51.1。去掉大量无标签数据训练 GCN 的版本只有 75.8、43.1，说明不是 metadata 本身撑起来的。t-SNE 里加 GCN 后同身份更聚、易混身份分开，直接对应“邻域关系比点对相似度更可靠”。
./analyses/ALL_ANALYSES.txt:591:1. **触发观察**：正式版进一步把观察说得更工程化，单模态 ReID 和传统 VI-ReID 都没有覆盖“query 集和 gallery 集同时混合可见光与红外”的情况。作者强调 Rank-1 有时会因为同模态易匹配而变好，但 mAP 和 mINP 下降，说明整体检索质量变差。
./analyses/ALL_ANALYSES.txt:676:4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。
./analyses/ALL_ANALYSES.txt:684:2. 重定义动作：旧问题是“怎么建模视频时间特征”；新问题是“怎么按身体部位自适应选择不同长度的时间线索”。关键词是 dynamic hierarchical graph、adaptive multi-granularity temporal clues、part-consistent graph、mutual nearest neighbor、similarity weighted inference。
./analyses/ALL_ANALYSES.txt:705:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./analyses/ALL_ANALYSES.txt:737:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./analyses/ALL_ANALYSES.txt:741:4. **证据闭环**：关键证据很直接。APE 单独带来大幅提升，在 Occluded-Duke 上 mAP 和 Rank-1 分别涨 18.4 和 16.9；OAM 再带来 2.6 和 2.9。部位数实验说明 4 个部位最合理，太多会冗余，6 个会忽略左右区分。最有说服力的是可视化，APE 在姿态标签错误时仍能抽对区域，能避开伞遮住的头部，还能抓住姿态估计忽略的手提包。这是在证明“自适应区域优于姿态输出”，不是只证明最终分数涨。
./analyses/ALL_ANALYSES.txt:822:4. 证据闭环：Table III 做 DLF、MIA、HSM、ESM 的组件消融，Table IV 验证 DLF 和 MIA 放在哪些层最合适。还有 α 和 τ 的参数分析、VIS-IR cosine distance 分布、t-SNE、Grad-CAM 和 Rank-10 检索可视化。它的证据闭环是证明双空间不只是换名词，确实让跨模态分布更紧、注意区域更对。
./analyses/ALL_ANALYSES.txt:883:2. **重定义动作**：他们把问题从“如何生成更好伪标签”改成“如何选择最有训练价值的 proxy 对”。关键词是 `optimal proxy mining`、`hardest camera proxy`、`false negative proxies mining`、`misaligned contrastive pairs`、`camera-constrained hard negative mining`、`anti-noise ranking`。这让创新点落在训练信号选择，而不是聚类算法本身。
./analyses/ALL_ANALYSES.txt:887:4. **证据闭环**：关键消融是正样本选择策略对比：corresponding proxy、all proxies、hardest positive proxy，以及是否使用 CCHNM。FNPM 部分又验证了排除同相机 proxy 和 anti-noise ranking 的作用。可视化包括 t-SNE、Top-10 检索和 Grad-CAM，用来证明不是只涨点，而是类内更紧、类间更开、注意力更落在人身上。
./analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./analyses/ALL_ANALYSES.txt:964:2. **重定义动作**：它把“灾难性遗忘”重写成“旧身份原型、新任务特征、更新后特征提取器之间的兼容问题”。关键词是 non-exemplar、prototype-guided knowledge propagation、representation bias、adaptive parameter evolution、compatibility。旧问题是保留旧知识，新说法是让旧知识在新表征空间里继续能区分。
./analyses/ALL_ANALYSES.txt:968:4. **证据闭环**：组件消融很对题。baseline 的 Seen-Avg 是 42.8 mAP/57.4 Rank-1，加入 PKP 到 53.4/66.0，加入 APE 到 54.5/67.8，两者一起到 55.4/68.7；Unseen-Avg 也从 50.7/45.0 到 62.3/55.3。t-SNE 显示 PKA 后类内更紧、类间更分开，证明它不是只调分类器，而是在改善检索表征空间。
./analyses/ALL_ANALYSES.txt:992:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./analyses/ALL_ANALYSES.txt:1004:4. **证据闭环**：IMRM 相比全局和局部基线都有提升，F&B-TCRMG 好于只用正序或倒序，CMRM 加上 IMRM 最好。训练证据也对题：Hc-Tri 在约第 20 次迭代就收敛到 0，Acc 和 C2FL 能继续提供监督，C2FL 达到 69.16 Rank-1、86.23 Rank-5、64.57 mAP。热力图和检索例子显示，在外观细节高度相似时，模型能利用“包和人的关系”区分身份。
./analyses/ALL_ANALYSES.txt:1036:2. **重定义动作**：它把噪声样本从“需要丢掉的坏样本”改写成“可以被历史预测逐步纠正的潜在有效样本”。同时把训练波动从坏现象改写成信息来源。关键词是 `self-rectification`、`historical consistency`、`early learning`、`momentum-updated memories`、`coupled noisy labels`。
./analyses/ALL_ANALYSES.txt:1038:4. **证据闭环**：核心消融是 baseline、加跨模态对比、加 early learning regularization、再加 label rectification 的逐步对比。可视化更关键：不同噪声率下标签准确率随 rectification 逐步上升，loss 分布里噪声样本被转成低损失干净样本，和 DART 对比显示它不是单纯抗噪，而是在回收训练数据。
./analyses/ALL_ANALYSES.txt:1108:4. 证据闭环：RSSPA 是主证据，CUHK 上 Rank-1 提升超过 3%，mAP 约 7%，mINP 约 11%。关系敏感对齐比传统单映射多 2.76 Rank-1、6.16 mAP、10.28 mINP。IBSPA、IFAA、APFR 继续带来增益。作者还比较了 RSSPA+IBSPA 和 SDM/CMPM，属性实验显示多粒度属性优于词级属性，属性数量过多反而伤害性能。
./analyses/ALL_ANALYSES.txt:1128:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./analyses/ALL_ANALYSES.txt:1140:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./analyses/ALL_ANALYSES.txt:1164:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./analyses/ALL_ANALYSES.txt:1170:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./analyses/ALL_ANALYSES.txt:1172:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./analyses/ALL_ANALYSES.txt:1176:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./analyses/ALL_ANALYSES.txt:1178:5. **reviewer 为什么买账**：这篇卖的是两个务实视角：训练时不迷信 diffusion 真实感，测试时不照搬同模态 re-ranking。机制都很简单，但和观察贴得紧，消融也正好打在 reviewer 会怀疑的点上。novelty 来源不是复杂模型，而是对 TBPS 任务目标的重新理解。
./analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./analyses/ALL_ANALYSES.txt:1215:2. 重定义动作：作者把 ReID 从单一任务重定义为 Anytime ReID，用“时间点”和“时间间隔”划出六个场景。关键词是 anytime retrieval、multi-scenario retrieval、scenario-specific cues、inter-scenario interference。
./analyses/ALL_ANALYSES.txt:1219:4. 证据闭环：数据集 AT-USTC 本身证明旧数据集缺口，21 个月、RGB 和红外、多衣服、多季节支撑“anytime”成立。模型证据包括六场景分别评测、跨数据集泛化、MS-ReID、MoAE、HDW 消融，以及和其他 MoE 方法对比，证明不是单纯多头分类。
./analyses/ALL_ANALYSES.txt:1375:它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。
./analyses/ALL_ANALYSES.txt:1389:机制几乎是从三类噪声定义直接长出来的。先用 co-training 和 warm-up 利用 early learning，小损失样本更可能干净。再用 GMM 拟合单样本 CE loss，把样本分成 clean、indistinguishably noisy、obviously noisy。干净样本用 CE，因为标签可信；难分辨噪声用 GCE，降低噪声样本的大损失偏置；明显噪声直接不信原标签，用分类置信度构造软伪标签。检索侧还把样本对按置信度划成 clean/noisy correspondence，修正 triplet 里的正负关系，形成 robust retrieval loss。最后把识别损失和检索损失混成 RHL。
./analyses/ALL_ANALYSES.txt:1392:这篇的证据闭环很强，因为它先造了 SYSU-Real，报告真实噪声比例是 23.79%，并且红外内部噪声比例略高，这直接支撑“真实噪声值得研究”。然后它在 random noise、IDN 和 SYSU-Real 三种条件下都测，证明不是只吃某一种噪声。RHL 消融里，加入处理 indistinguishable noisy subset 的 GCE 带来最大提升，完整方法在 SYSU-Real 上比 DART 提升 7.23% Rank-1、6.94% mAP 和 7.48% mINP，正好支持“三类样本要分治”。可视化也服务于重定义：不同噪声类型的退化柱状图说明 IDN 和真实噪声比随机噪声更伤；相似度矩阵差异图显示 RHL 提高类内相似、降低类间相似；t-SNE 显示比 DART 聚类更清楚。
./analyses/ALL_ANALYSES.txt:1462:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./analyses/ALL_ANALYSES.txt:1486:**2. Bridging the gap: Learning adaptive knowledge transition for lifelong person re-identification**
./analyses/ALL_ANALYSES.txt:1490:2. 重定义动作：他们把 catastrophic forgetting 重定义成 adaptive knowledge transition 问题。旧说法是 preserve previous knowledge，新说法是 control the degree and direction of knowledge distillation。关键词从 distillation、forgetting，换成 transition path、intermediate knowledge、adaptive transfer。这样一来，论文的中心不再是“再加一个蒸馏损失”，而是“旧知识和新知识之间需要一座可学习的桥”。
./analyses/ALL_ANALYSES.txt:1498:**3. Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification**
./analyses/ALL_ANALYSES.txt:1502:2. 重定义动作：他们把无监督 ReID 从“伪标签不准”重定义成“多相机子域之间缺少语义传播”。每个 camera 被定义成一个 domain，训练不只是 cluster contrastive learning，而是 camera-aware multi-domain adaptive learning。关键词是 camera-aware graph、multi-domain、semantic propagation、domain-invariant representation。
./analyses/ALL_ANALYSES.txt:1542:4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。
./analyses/ALL_ANALYSES.txt:1573:**CFPER**
./analyses/ALL_ANALYSES.txt:1576:2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
./analyses/ALL_ANALYSES.txt:1580:4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。
./analyses/ALL_ANALYSES.txt:1669:4. 证据闭环：消融显示 DCAM、CATS、GSMEM、CARC 逐步涨，完整模型在 SYSU-MM01 到 68.52 Rank-1 和 64.0 mAP。更贴近重定义的是三类图：超参 p 证明闭环长度有效，检索可视化说明 partial modules 仍处理不好大姿态差，t-SNE 和距离分布显示跨模态聚合和类间分离逐步改善。但如果严审，“拓扑”本身是否比普通图传播更关键，还需要更强的 cycle-only 对照。
./analyses/ALL_ANALYSES.txt:1706:1. 触发观察：作者先观察训练过程中“特征向量范数”和准确率的关系。随着训练推进，特征范数下降，准确率上升；Market-1501 前 30 个 epoch 里 rank-1 下降时，特征范数反而上升。这个观察把一个很普通的 loss 设计问题，变成了“训练动态里范数压缩不稳定”的问题。
./analyses/ALL_ANALYSES.txt:1748:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
./analyses/ALL_ANALYSES.txt:1760:4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。
./analyses/ALL_ANALYSES.txt:1772:4. 证据闭环：Table 1 很强。单标签里 F、C、I 各有表现，两两组合普遍强于单标签，HPL 三标签又强于两标签；加 PLC 从 80.3/92.5 提到 82.5/92.9，加 CCL 比普通 CL 更好，最终 Ours 达到 87.2 mAP、95.0 Rank-1。Table 2 还证明 DBSCAN 做 fine、K-means 做 coarse 最优，Table 3 讨论 hard/soft 标签形式。这些消融直接证明“异质标签互补”这个重定义。
./analyses/ALL_ANALYSES.txt:1808:2. 重定义动作：作者把问题从“学一个跨模态嵌入空间”改成“直接优化跨模态 AP”。关键词是 `Cross-modality Average Precision`、`CAPV`、`CAPT`、`overall ranking order`、`cross-modality discrepancy`。
./analyses/ALL_ANALYSES.txt:1836:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./analyses/ALL_ANALYSES.txt:1864:4. 证据闭环：HITSZ-VCM 消融显示 CIM 单独带来 Rank-1 和 mAP 大提升，FDM 单独也提升，两者组合最好。t-SNE 显示同身份跨模态更聚、不同身份更分开，Rank-5 可视化展示视频检索结果。参数分析说明帧数增加和 FDM 维度变化会影响性能。
./analyses/deep_25.txt:11:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./analyses/deep_25.txt:23:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./analyses/deep_25.txt:47:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./analyses/deep_25.txt:53:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./analyses/deep_25.txt:55:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./analyses/deep_25.txt:59:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./analyses/deep_25.txt:61:5. **reviewer 为什么买账**：这篇卖的是两个务实视角：训练时不迷信 diffusion 真实感，测试时不照搬同模态 re-ranking。机制都很简单，但和观察贴得紧，消融也正好打在 reviewer 会怀疑的点上。novelty 来源不是复杂模型，而是对 TBPS 任务目标的重新理解。
./analyses/deep_25.txt:71:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./debate/d_1.md:16:**Gallery Hubness re-frame**: 强 ReID 失败不是 pairwise 相似度病, 而是少数 gallery 样本成为很多不同身份 query 的误吸附点(gallery 负向 in-degree / hub mass)=many-to-one 图库拓扑病。机制设想: 训练端 anti-hub embedding(memory bank 存 cross-ID in-degree, anti-hub margin, 负样本权重从'离anchor近'改'是否全局误吸附点', 测试仍单 embedding 不 re-rank)。
./debate/d_1.md:27:我会先查一轮原始论文/官方仓库信息，把“训练端单向量”和“test-time re-ranking”的边界、HAL/NeighborRetr 的任务域、以及 Market/MSMT/Occluded-Duke 难度差异钉住，再给出红队判断和最小训练方案。
./debate/d_1.md:29:web search: k-reciprocal re-ranking person re-identification CVPR 2017 single query gallery post-processing
./debate/d_1.md:55:web search: TransReID github re-ranking Market1501 mAP Rank-1 rerank
./debate/d_1.md:57:web search: github TransReID re-ranking MSMT17 Market1501 performance
./debate/d_1.md:63:web search: 're-rank' in https://arxiv.org/abs/1903.07071
./debate/d_1.md:65:web search: SOLIDER person re-identification re-ranking results GitHub
./debate/d_1.md:84:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./debate/d_1.md:86:Market 的 D2 mAP 弱不能过度解读。你们 Market 已经 94.6 mAP，headroom 极低；而 MSMT17 原论文明确设计成更大、更真实、更难，有 15 camera、复杂光照/场景、4101 IDs/126441 boxes。([arxiv.org](https://arxiv.org/abs/1711.08565)) Occluded ReID 又天然有 missing/noisy information。([arxiv.org](https://arxiv.org/abs/2212.04712)) 一个 2025 ReID re-ranking 结果也显示，邻域/拓扑类后处理在 MSMT17、Occluded-DukeMTMC 的 R1 增益远大于 Market。([arxiv.org](https://arxiv.org/abs/2509.04050)) 这正支持“Market 看不出，难集会放大 hub 病”的判断。
./debate/d_1.md:100:5. inference 完全不变：只输出一个 L2-normalized embedding，不用 re-rank。
./debate/d_1.md:104:- 主通过：no-rerank `mAP >= +0.8` 且 `R1 >= +1.0`，或 R1 明显涨且 false-top1 top1% hub mass 下降 `>=20%`。
./debate/d_1.md:105:- 强通过：`anti-hub + k-reciprocal` 仍高于 `baseline + k-reciprocal`，证明不是和 re-ranking 抢同一块。
./debate/d_1.md:106:- 杀死：no-rerank `< +0.5 mAP` 且 `< +0.5 R1`，同时 hub mass / AP-error partial corr 基本不降。
./debate/d_1.md:118:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./debate/d_1.md:120:Market 的 D2 mAP 弱不能过度解读。你们 Market 已经 94.6 mAP，headroom 极低；而 MSMT17 原论文明确设计成更大、更真实、更难，有 15 camera、复杂光照/场景、4101 IDs/126441 boxes。([arxiv.org](https://arxiv.org/abs/1711.08565)) Occluded ReID 又天然有 missing/noisy information。([arxiv.org](https://arxiv.org/abs/2212.04712)) 一个 2025 ReID re-ranking 结果也显示，邻域/拓扑类后处理在 MSMT17、Occluded-DukeMTMC 的 R1 增益远大于 Market。([arxiv.org](https://arxiv.org/abs/2509.04050)) 这正支持“Market 看不出，难集会放大 hub 病”的判断。
./debate/d_1.md:134:5. inference 完全不变：只输出一个 L2-normalized embedding，不用 re-rank。
./debate/d_1.md:138:- 主通过：no-rerank `mAP >= +0.8` 且 `R1 >= +1.0`，或 R1 明显涨且 false-top1 top1% hub mass 下降 `>=20%`。
./debate/d_1.md:139:- 强通过：`anti-hub + k-reciprocal` 仍高于 `baseline + k-reciprocal`，证明不是和 re-ranking 抢同一块。
./debate/d_1.md:140:- 杀死：no-rerank `< +0.5 mAP` 且 `< +0.5 R1`，同时 hub mass / AP-error partial corr 基本不降。
./analyses/deep_3.txt:46:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reviews/deep_5.md:29:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_5.md:52:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./reviews/deep_5.md:85:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./reviews/deep_5.md:93:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./reviews/deep_5.md:149:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./reviews/deep_5.md:155:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./reviews/deep_5.md:169:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./reviews/deep_5.md:295:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./reviews/deep_5.md:391:rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。
./reviews/deep_5.md:466:假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。
./reviews/deep_5.md:605:3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
./reviews/deep_5.md:733:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./reviews/deep_5.md:762:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
./reviews/deep_5.md:765:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./reviews/deep_5.md:766:- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
./reviews/deep_5.md:843:**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。
./reviews/deep_5.md:847:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reviews/deep_5.md:861:用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。
./reviews/deep_5.md:867:- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
./reviews/deep_5.md:989:without violating privacy (DukeMTMC-ReID has been retracted due to privacy concerns). In particular, the complexity
./reviews/deep_5.md:1156:and a large-scale OmniReID benchmark as well as adaptive
./reviews/deep_5.md:1328:where R denotes Rank-1 evaluation and Df ine test is the test
./reviews/deep_5.md:1345:comparison of these datasets are shown in Tab. I. Subsequently, two frequently-used metrics Rank-1 and mAP are
./reviews/deep_5.md:1368:28.8% rank-1 on NKUP, and FIRe2 achieves 64.7% rank1 on PRCC, and 85.1% rank-1 and 85.0% mAP on VCClothes. The best performance of mAP on NKUP is obtained
./reviews/deep_5.md:1450:(+13.0%) rank-1 and 59.0% (+10.8%) mAP on PRCC, 83.3%
./reviews/deep_5.md:1451:(+10.2%) rank-1 and 83.1% (+8.8%) mAP on VC-Clothes,
./reviews/deep_5.md:1452:as well as 28.8% (+7.6%) rank-1 on NKUP. Furthermore,
./reviews/deep_5.md:1454:PRCC and NKUP, achieving 64.7% (+5.6%) rank-1, 57.7
./reviews/deep_5.md:1455:(+7.2%) mAP on PRCC, 26.4% (+4.0%) rank-1 and 17.8
./reviews/deep_5.md:1465:rank-1
./reviews/deep_5.md:1506:rank-1
./reviews/deep_5.md:1542:rank-1
./reviews/deep_5.md:1589:and 26.5 outfits per individual. To overcome the difficulty
./reviews/deep_5.md:1622:Bo Li, and Qi Tian, “Unrealperson: An adaptive pipeline towards
./reviews/deep_5.md:1953:Firstly, due to the complexity and sensitivity of surveillance data, manual data annotation is an
./reviews/deep_5.md:1954:unsustainable task. Faced with a vast amount of raw data, it is clearly impractical to adjust the
./reviews/deep_5.md:2110:/bin/zsh -lc "pdftotext -f 1 -l 8 'CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reviews/deep_5.md:2114:CFPER: Coarse-to-Fine Part-Experts Retrieval for
./reviews/deep_5.md:2132:observe that query difficulty varies: global features suffice for
./reviews/deep_5.md:2138:adaptively allocates resources based on query difficulty. For
./reviews/deep_5.md:2140:stage, and inference terminates early. For “hard” queries, part
./reviews/deep_5.md:2162:difficulty of queries, which may limit accuracy on challenging
./reviews/deep_5.md:2164:ones. In real-world applications, the retrieval difficulty of different queries varies: some query images exhibit distinct discriminative features, and global features alone are sufficient for
./reviews/deep_5.md:2181:Rank-1 (%)
./reviews/deep_5.md:2215:with increasing retrieval difficulty: Market-1501 (holistic) [8],
./reviews/deep_5.md:2218:features, and report mAP, Rank-1 and FLOPs in Table I. The
./reviews/deep_5.md:2220:improvements in mAP (1.1% and 1.0%) and Rank-1 (0.4% and
./reviews/deep_5.md:2222:improvements (1.9% mAP, 3.1% Rank-1) on the more challenging occluded dataset. This indicates that global features
./reviews/deep_5.md:2226:by the above observations, we propose a two-stage Coarse-toFine Dynamic Retrieval (CFDR) mechanism that adaptively
./reviews/deep_5.md:2227:allocates computational resources based on the difficulty of
./reviews/deep_5.md:2229:early-exit threshold is computed to evaluate query difficulty.
./reviews/deep_5.md:2231:retrieval in the coarse stage, and inference terminates early to
./reviews/deep_5.md:2243:Fig. 1. The framework of CFPER consists of two stages: coarse inference stage and fine inference stage. In the coarse stage, a global feature G and N patch
./reviews/deep_5.md:2244:embeddings Z are extracted. An early-exit decision is made based on the threshold ⌈N α⌉: if met, the query is considered “easy”, only global feature is used
./reviews/deep_5.md:2262:Coarse-to-Fine Part-Experts Retrieval model (CFPER).
./reviews/deep_5.md:2266:mechanism that balances computational efficiency and retrieval accuracy by adaptively allocating resources based
./reviews/deep_5.md:2267:on the query difficulty.
./reviews/deep_5.md:2274:• The proposed CFPER achieves competitive performance
./reviews/deep_5.md:2278:Part-Experts Retrieval (CFPER) in detail. An overview of
./reviews/deep_5.md:2279:CFPER is shown in Fig. 1.
./reviews/deep_5.md:2286:Early-exit strategy. In the coarse inference stage, we
./reviews/deep_5.md:2287:introduce an early-exit strategy to balance performance and
./reviews/deep_5.md:2288:efficiency. Traditional early-exit strategies [12] are typically
./reviews/deep_5.md:2291:making the classifier score-based exit metric unsuitable. To
./reviews/deep_5.md:2292:address this, we propose an early-exit strategy for person
./reviews/deep_5.md:2296:Fig. 2. Illustration of the early-exit strategy based on the first-order differences
./reviews/deep_5.md:2299:ReID, which determines whether to exit early by evaluating
./reviews/deep_5.md:2300:the query difficulty based on the first-order differences of
./reviews/deep_5.md:2302:To assess query difficulty, our intuition is to count the
./reviews/deep_5.md:2321:introduce an early-exit threshold α ∈ [0, 1]. If η ≥ ⌈N α⌉, the
./reviews/deep_5.md:2495:In the training stage, we set the early-exit threshold α = 1,
./reviews/deep_5.md:2514:(mAP) to evaluate the performance of ReID methods. To access the efficiency of our CFPER, we report FLOPs calculated
./reviews/deep_5.md:2526:model for 300 epochs. For a fair comparison, no re-ranking
./reviews/deep_5.md:2529:The early-exit threshold α is an important factor that
./reviews/deep_5.md:2531:methods. The CFDR uses the early-exit threshold α to assess
./reviews/deep_5.md:2532:the query difficulty and control the number of queries proceed
./reviews/deep_5.md:2536:α on Occluded-Duke, using Rank-1 and mAP as performance
./reviews/deep_5.md:2540:demonstrating significant computational savings. Rank-1 accuracy initially increases, indicating that global features can
./reviews/deep_5.md:2545:adaptive allocation of computational resources between “easy”
./reviews/deep_5.md:2553:BETWEEN P ERFORMANCE AND E FFICIENCY OF CFPER.
./reviews/deep_5.md:2560:Rank-1 (%)
./reviews/deep_5.md:2603:CFPER (α = 1)
./reviews/deep_5.md:2604:CFPER (α = 0.6)
./reviews/deep_5.md:2607:Rank-1
./reviews/deep_5.md:2641:Rank-1
./reviews/deep_5.md:2675:Rank-1 mAP
./reviews/deep_5.md:2727:Rank-1
./reviews/deep_5.md:2748:our CFPER (α = 0.6) achieves comparable performance with
./reviews/deep_5.md:2749:a Rank-1 accuracy of 68.5% and mAP of 58.4%, outperforming most of the compared methods on both metrics without
./reviews/deep_5.md:2752:CFPER (α = 0.6) achieves the highest Rank-1 accuracy
./reviews/deep_5.md:2753:on Market-1501 and the second-highest Rank-1 accuracy on
./reviews/deep_5.md:2755:CFPER achieves competitive mAP on both datasets. These
./reviews/deep_5.md:2756:results validate CFPER’s ability to adaptively handle queries
./reviews/deep_5.md:2757:with varying difficulty, ensuring both efficiency and accuracy.
./reviews/deep_5.md:2759:In this section, we conduct ablation studies on OccludedDuke to analyze the effectiveness of components of CFPER.
./reviews/deep_5.md:2781:+2.5% Rank-1 accuracy and +2.3% mAP improvements,
./reviews/deep_5.md:2783:shows that PE further improve performance by +4.8% Rank-1
./reviews/deep_5.md:2796:just 0.0123M parameters, greatly reducing model complexity
./reviews/deep_5.md:2803:dataset to demonstrate the effectiveness of our CFPER.
./reviews/deep_5.md:2805:shows “easy” and “hard” queries identified by CFPER, validating the soundness of our early-exit strategy in the CFDR
./reviews/deep_5.md:2837:In this paper, we propose a novel model CFPER to balance
./reviews/deep_5.md:2840:difficulty and tailoring the learning of visible body part representations, CFPER achieves competitive performance while
./reviews/deep_5.md:2841:significantly reducing computational costs. Extensive experiments demonstrate the effectiveness of CFPER.
./reviews/deep_5.md:3219:fine-grained channel-level feature modeling at the early stage of feature extraction, effectively mitigating the impact of channel imbalance
./reviews/deep_5.md:3221:This design allows CLFO to not only adaptively highlight informative channels but also maintain computational efficiency and training
./reviews/deep_5.md:3265:where the SE block implements global channel attention via adaptive
./reviews/deep_5.md:3297:feature optimization at an early stage of feature extraction, CLFO effectively mitigates the impact of modality-specific noise and imbalanced
./reviews/deep_5.md:3347:channels. It uses adaptive average pooling (AvgPool1×1 ) to process 𝑂,
./reviews/deep_5.md:3429:both channel and spatial importance in a cascaded manner, CLFR significantly improves the quality and effectiveness of features, providing
./reviews/deep_5.md:3489:Evaluation Metrics. We adopt widely used metrics in person reidentification tasks, including Rank-k accuracy and mean Average Precision (mAP). All results are reported as averages over 10 independent
./reviews/deep_5.md:3543:further processed through a dual-attention mechanism that adaptively
./reviews/deep_5.md:3561:performance among all compared methods. Specifically, under the allsearch mode in SYSU-MM01, CAFMNet achieves a Rank-1 accuracy
./reviews/deep_5.md:3563:achieves a Rank-1 accuracy of 84.95% and an mAP of 87.09%. These
./reviews/deep_5.md:3564:results clearly validate the effectiveness of the proposed approach.
./reviews/deep_5.md:3772:unique design of CLFO — its ability to adaptively refine channel-wise
./reviews/deep_5.md:3777:additional gain of +0.77% in Rank-1 and +0.85% in mAP, despite a relatively larger increase in computational complexity (FLOPs: +6.858G,
./reviews/deep_5.md:3784:and MDFO are activated, the model achieves 75.87% Rank-1 accuracy
./reviews/deep_5.md:3792:CAFMNet model achieves optimal performance: 77.49% Rank-1 accuracy and 74.19% mAP. In this configuration, CLFO provides the
./reviews/deep_5.md:3800:performance improvements stem from architectural innovation or simply from increased model complexity. As demonstrated in Table 3,
./reviews/deep_5.md:3807:and effectiveness of the CAFMNet network structure design and clearly
./reviews/deep_5.md:3893:Complexity
./reviews/deep_5.md:3901:As shown, CAFMNet achieves strong performance in both infrared-tovisible (IR-to-VIS) and visible-to-infrared (VIS-to-IR) settings. Specifically, in the VIS-to-IR setting, CAFMNet achieves a Rank-1 accuracy
./reviews/deep_5.md:3903:a Rank-1 accuracy of 57.58% and an mAP of 64.10%. These results
./reviews/deep_5.md:3913:CLFO alone improved Rank-1 accuracy from 69.85% to 72.11%, and
./reviews/deep_5.md:3933:Rank-1 (%) mAP (%) FLOPs (G) Params (M)
./reviews/deep_5.md:3943:Rank-1 (%)
./reviews/deep_5.md:4063:Complexity
./reviews/deep_5.md:4093:Complexity
./reviews/deep_5.md:4102:benchmark of 72.11% Rank-1 accuracy and 69.19% mAP. Introducing
./reviews/deep_5.md:4103:LGF, DSC, and SA without CA elevates the Rank-1 accuracy to 72.33%
./reviews/deep_5.md:4113:in Rank-1 accuracy and 0.11% in mAP. By analyzing global feature
./reviews/deep_5.md:4118:standard convolutions slightly improves performance to 72.88% Rank-1
./reviews/deep_5.md:4121:After removing SA, the Rank-1 accuracy drops to 72.60% and the
./reviews/deep_5.md:4124:module (CA + LGF + DSC + SA) configuration, the Rank-1 accuracy
./reviews/deep_5.md:4138:Rank-1 accuracy of 72.88% and an mAP of 70.04% with a moderate
./reviews/deep_5.md:4152:adaptive L-P, to highlight the impact of dynamic feature adjustment.
./reviews/deep_5.md:4153:As illustrated in Table 4, the baseline model achieves a Rank-1
./reviews/deep_5.md:4155:and L-P (without SE) improves Rank-1 to 71.51% and mAP to 68.71%,
./reviews/deep_5.md:4160:Adding the SE block further boosts Rank-1 by 0.25% and mAP by
./reviews/deep_5.md:4164:aligning with CLFO’s goal of channel-level feature optimization. Removing DSC in favor of standard convolutions increases Rank-1 to
./reviews/deep_5.md:4168:The ablation of RC results in a significant performance drop (Rank1: 68.78%, mAP: 65.67%), confirming its necessity in maintaining
./reviews/deep_5.md:4171:configurations, L-P outperforms F-P by 0.91% in Rank-1 and 0.57%
./reviews/deep_5.md:4172:in mAP. The learnable nature of L-P allows it to adaptively adjust the
./reviews/deep_5.md:4177:an optimal balance, reaching 72.11% Rank-1 and 69.19% mAP with a
./reviews/deep_5.md:4191:CFPER 的完整方法和消融已经读完，它的关键不是“部件专家”本身，而是先用难度分流把效率问题改写成检索流程问题。Channel-aware 这篇文本较长，中间被输出截断，我会再针对第 3 节补读。
./reviews/deep_5.md:4469:domain-adaptive (UDA) ReID and purely unsupervised learning (USL) ReID. The latter, pure unsupervised ReID, presents
./reviews/deep_5.md:5144:pedestrian features across different cameras) in the early
./reviews/deep_5.md:5291:rich contextual information during early feature extraction. We denote
./reviews/deep_5.md:5501:We adopt the popular Rank-k (k = 1, 5, 10) as the evaluation metric
./reviews/deep_5.md:5503:Penalty (mINP) for comprehensive evaluation. Higher Rank-k, mAP and
./reviews/deep_5.md:5557:2.65% Rank-1 accuracy drop on the three datasets, respectively. By
./reviews/deep_5.md:5558:introducing ABF into baseline, the method also results in a Rank-1
./reviews/deep_5.md:5567:proposed MoD and ABF into our baseline (No. 3), we observe a performance gain of 1.57%, 0.09%, and 1.55% in terms of Rank-1 accuracy
./reviews/deep_5.md:5584:re-ranking. The final similarity for text–image pairs is computed as
./reviews/deep_5.md:5665:Rank-1
./reviews/deep_5.md:5667:Rank-5
./reviews/deep_5.md:5669:Rank-10
./reviews/deep_5.md:5671:Rank-1
./reviews/deep_5.md:5673:Rank-5
./reviews/deep_5.md:5675:Rank-10
./reviews/deep_5.md:5677:Rank-1
./reviews/deep_5.md:5679:Rank-5
./reviews/deep_5.md:5681:Rank-10
./reviews/deep_5.md:5768:Rank-1
./reviews/deep_5.md:5811:improvements of 2.27%, 2.04%, and 3.8% in Rank-1 accuracy over
./reviews/deep_5.md:5825:3.22%, 2.37%, and 3.8% in Rank-1 accuracy over the baseline on the
./reviews/deep_5.md:5840:embeddings, which are then aligned with the text. In terms of Rank1 accuracy, the performance of the vanilla PCB drops by 1.32% and
./reviews/deep_5.md:5846:Rank-1 accuracy by 0.83% and 1.13%, respectively. This indicates
./reviews/deep_5.md:5862:4.3.1. Analysis of complexity and performance
./reviews/deep_5.md:5863:Table 2 presents the impact of different components on model performance, complexity, and training efficiency. Comparing the Global
./reviews/deep_5.md:5865:parameter count to 177 M, leads to a degradation in Rank-1 accuracy
./reviews/deep_5.md:5874:Rank-1 accuracy of 74.68%, outperforming the Baseline by 3.22%.
./reviews/deep_5.md:5876:169.1 samples/s due to increased computational complexity, the number of trainable parameters remains constant at 177 M. This demonstrates that CDFM significantly enhances feature representation capabilities with only marginal computational overhead and no additional
./reviews/deep_5.md:5891:ICFG-PEDES datasets in terms of Rank-1 accuracy.
./reviews/deep_5.md:5914:Rank-1
./reviews/deep_5.md:5916:Rank-5
./reviews/deep_5.md:5918:Rank-10
./reviews/deep_5.md:5920:Rank-1
./reviews/deep_5.md:5922:Rank-5
./reviews/deep_5.md:5924:Rank-10
./reviews/deep_5.md:5982:Rank-1
./reviews/deep_5.md:5984:Rank-5
./reviews/deep_5.md:5986:Rank-10
./reviews/deep_5.md:5988:Rank-1
./reviews/deep_5.md:5990:Rank-5
./reviews/deep_5.md:5992:Rank-10
./reviews/deep_5.md:6035:decoders (No. 1) results in a Rank-1 accuracy improvement of 2.53%
./reviews/deep_5.md:6040:cosine similarity relationships (No. 2). This approach improves Rank1 accuracy by 2.53% on CUHK-PEDES and 0.60% on ICFG-PEDES.
./reviews/deep_5.md:6044:Rank-1 accuracy improved significantly by 3.01% on CUHK-PEDES and
./reviews/deep_5.md:6050:layers (D) on Rank-1 accuracy. As shown in Fig. 5(a), Rank-1 accuracy
./reviews/deep_5.md:6051:fluctuates non-linearly when D ranges from 2 to 4. It peaks at 𝐷 = 5 on
./reviews/deep_5.md:6060:experiments to examine the impact of different values of K on Rank1 accuracy. As shown in Fig. 5(b), the performance improves as K
./reviews/deep_5.md:6065:on Rank-1 accuracy. As shown in Fig. 5(c), the accuracy trend is
./reviews/deep_5.md:6081:brings performance gains on the Rank-1 metrics, regardless of whether
./reviews/deep_5.md:6308:inability to adequately fit the data distribution, failing to capture the complexity and diversity of
./reviews/deep_5.md:6403:Rank-1 accuracy but maintains consistency in Mean Average Precision (mAP). This is because,
./reviews/deep_5.md:6408:both Rank-1 accuracy and mAP. Specifically, compared to the FedUReID [55] method, where all
./reviews/deep_5.md:6419:Rank@1 Rank@5 Rank@10
./reviews/deep_5.md:6474:mAP Rank@1 Rank@5 Rank@10
./reviews/deep_5.md:6538:Table 2. Rank@1 Results on Eight Datasets (%)
./reviews/deep_5.md:6619:Tables 2 and 3 show the Rank-1 and mAP results for FedUReID, FedUCC, and CCFL (ours)
./reviews/deep_5.md:6623:corresponding to the Rank-1 accuracy closest to the results reported by Zhuang et al. [55] as an
./reviews/deep_5.md:6625:our method achieves significant improvements across all datasets, with an average Rank-1 accuracy
./reviews/deep_5.md:6737:fine-grained channel-level feature modeling at the early stage of feature extraction, effectively mitigating the impact of channel imbalance
./reviews/deep_5.md:6739:This design allows CLFO to not only adaptively highlight informative channels but also maintain computational efficiency and training
./reviews/deep_5.md:6783:where the SE block implements global channel attention via adaptive
./reviews/deep_5.md:6815:feature optimization at an early stage of feature extraction, CLFO effectively mitigates the impact of modality-specific noise and imbalanced
./reviews/deep_5.md:6865:channels. It uses adaptive average pooling (AvgPool1×1 ) to process 𝑂,
./reviews/deep_5.md:6947:both channel and spatial importance in a cascaded manner, CLFR significantly improves the quality and effectiveness of features, providing
./reviews/deep_5.md:7007:Evaluation Metrics. We adopt widely used metrics in person reidentification tasks, including Rank-k accuracy and mean Average Precision (mAP). All results are reported as averages over 10 independent
./reviews/deep_5.md:7061:further processed through a dual-attention mechanism that adaptively
./reviews/deep_5.md:7079:performance among all compared methods. Specifically, under the allsearch mode in SYSU-MM01, CAFMNet achieves a Rank-1 accuracy
./reviews/deep_5.md:7081:achieves a Rank-1 accuracy of 84.95% and an mAP of 87.09%. These
./reviews/deep_5.md:7082:results clearly validate the effectiveness of the proposed approach.
./reviews/deep_5.md:7745:pedestrian features across different cameras) in the early
./reviews/deep_5.md:7800:In terms of evaluation metrics, we adopt cumulative matching characteristics (CMC) [44] including Rank-1, Rank-5, and
./reviews/deep_5.md:7801:Rank-10 as well as mean Average Precision (mAP).
./reviews/deep_5.md:8046:rich contextual information during early feature extraction. We denote
./reviews/deep_5.md:8256:We adopt the popular Rank-k (k = 1, 5, 10) as the evaluation metric
./reviews/deep_5.md:8258:Penalty (mINP) for comprehensive evaluation. Higher Rank-k, mAP and
./reviews/deep_5.md:8312:2.65% Rank-1 accuracy drop on the three datasets, respectively. By
./reviews/deep_5.md:8313:introducing ABF into baseline, the method also results in a Rank-1
./reviews/deep_5.md:8322:proposed MoD and ABF into our baseline (No. 3), we observe a performance gain of 1.57%, 0.09%, and 1.55% in terms of Rank-1 accuracy
./reviews/deep_5.md:8339:re-ranking. The final similarity for text–image pairs is computed as
./reviews/deep_5.md:8366:web search: "CFPER" "Coarse-to-Fine Part-Experts Retrieval"
./reviews/deep_5.md:8396:**CFPER**
./reviews/deep_5.md:8399:2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
./reviews/deep_5.md:8403:4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。
./reviews/deep_5.md:8470:**CFPER**
./reviews/deep_5.md:8473:2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
./reviews/deep_5.md:8477:4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。
./debate/clean/d_2.txt:1:结论：**不建议投第一次训练。Hubness 降级为 r_2 备胎，把诊断写成 observation；主线切到 Rank-Instability。信心 8/10。**
./debate/clean/d_2.txt:3:蓝队理由很硬：你的 D4 证明的是“Gallery negative hubness 是真实失败结构”，但 D2 证明的是“当前可落地方法入口不够强”。这两件事不能混在一起。零训练 `cos - λlog(1+H_k)` 在 Market 只给 `+0.31 mAP`，而 k-reciprocal `+1.26 mAP`、camera downweight `+0.67 mAP` 都是免费 test-time。训练版如果要成为 CCF-B 方法稿，不能只赢 raw cosine，至少要在 mAP 上打过这些便宜强基线，否则审稿人会说：你发现了 hubness，但有效解法还是 re-ranking/camera prior。
./debate/clean/d_2.txt:8:   HAL 确实在 text-image matching 里提出 hubness-aware loss，用 memory bank/global-local statistics 给 hubs 加权，并报告 MS-COCO/Flickr30k R@1 提升；但它比较的是 Sum/Max ranking loss，不是 person ReID 的 k-reciprocal/camera-aware re-ranking。NeighborRetr 也明确说跨模态里已有 HAL、Dual Softmax、QB-Norm、DBNorm 等路线，并且 HAL 这种“放大 hub loss”的策略可能惩罚 beneficial hubs，需要区分 good/bad hubs。来源：HAL 方法描述和结果、NeighborRetr related work。  
./debate/clean/d_2.txt:12:   k-reciprocal re-ranking 本来就是 ReID 图库拓扑修正：无标注、无训练、对任意初始排序可用；原论文在 Market/MARS/CUHK03/PRW 上展示大 mAP 增益，例如 IDE(R)+KISSME 在 Market 上 mAP 从 49.05 到 63.63，论文自己强调 mAP 提升显著。你的 hub score 本质也是图库图结构惩罚，但零训练 mAP 被 k-reciprocal 覆盖，这直接把训练版的证明门槛抬高。  
./debate/clean/d_2.txt:16:   近年的 CA-Jaccard 明确指出 camera variation 会让 Jaccard/re-ranking 邻域被 intra-camera samples 主导，带来 intra-camera negatives、排除 inter-camera positives；它用 camera-aware k-reciprocal neighbor 和 local query expansion 修正。你的 D2 里同相机降权直接赢 hub penalty，说明 hubness 可能仍被 camera topology 解释/覆盖一部分。  
./debate/clean/d_2.txt:20:   ReID 的 mAP 看的是多正样本整体排序。hub penalty 让 R1 `+1.13`，但 mAP 只 `+0.31` 且输给 k-reciprocal/camera，说明它可能只是把少数 false-top1 hub 压下去，没有系统性把所有 positives 拉上来。训练版也许能扩大这个效应，但当前证据更像“rank head repair”，不是“embedding 判别力提升”。
./debate/clean/d_2.txt:36:- 主线切 Rank-Instability，因为它更可能解释“强 embedding 下排序不稳定/邻域拓扑变化”的方法空间，而不是在已被免费 re-ranking 覆盖的 hub penalty 上硬训。
./debate/clean/d_1.txt:6:这不等于撞车。HAL 是 text-image matching，NeighborRetr 是 cross-modal retrieval；你们的切口是 **same-modality person ReID 的 gallery 负向 in-degree / many-query-to-one false attraction**。ReID 里的经典 k-reciprocal 是 test-time re-ranking：用 reciprocal neighbors 编 Jaccard，再和原距离融合，不产生部署用的单 embedding。([arxiv.org](https://arxiv.org/abs/1701.08398)) Person ReID 文献也长期把“学 embedding”和“再 re-rank”当互补步骤，PSE/ECN 明确同时贡献 learned representation 和 re-ranking。([arxiv.org](https://arxiv.org/abs/1711.10378)) 所以 anti-hub embedding 的价值不是赢过 k-reciprocal 的 mAP，而是得到一个 **不用 re-rank 的更好单向量**，再看能否与 re-rank 叠加。
./debate/clean/d_1.txt:8:Market 的 D2 mAP 弱不能过度解读。你们 Market 已经 94.6 mAP，headroom 极低；而 MSMT17 原论文明确设计成更大、更真实、更难，有 15 camera、复杂光照/场景、4101 IDs/126441 boxes。([arxiv.org](https://arxiv.org/abs/1711.08565)) Occluded ReID 又天然有 missing/noisy information。([arxiv.org](https://arxiv.org/abs/2212.04712)) 一个 2025 ReID re-ranking 结果也显示，邻域/拓扑类后处理在 MSMT17、Occluded-DukeMTMC 的 R1 增益远大于 Market。([arxiv.org](https://arxiv.org/abs/2509.04050)) 这正支持“Market 看不出，难集会放大 hub 病”的判断。
./debate/clean/d_1.txt:22:5. inference 完全不变：只输出一个 L2-normalized embedding，不用 re-rank。
./debate/clean/d_1.txt:26:- 主通过：no-rerank `mAP >= +0.8` 且 `R1 >= +1.0`，或 R1 明显涨且 false-top1 top1% hub mass 下降 `>=20%`。
./debate/clean/d_1.txt:27:- 强通过：`anti-hub + k-reciprocal` 仍高于 `baseline + k-reciprocal`，证明不是和 re-ranking 抢同一块。
./debate/clean/d_1.txt:28:- 杀死：no-rerank `< +0.5 mAP` 且 `< +0.5 R1`，同时 hub mass / AP-error partial corr 基本不降。
./validate/v_2.md:35: exited 1 in 0ms:
./validate/v_2.md:38: exited 1 in 0ms:
./validate/v_2.md:41: exited 1 in 0ms:
./validate/v_2.md:44: exited 1 in 0ms:
./validate/v_2.md:56: exited 1 in 0ms:
./validate/v_2.md:188:| 13 | 新协议贴部署 | severe modality imbalance; mix-modality; anytime |
./validate/v_2.md:193:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
./validate/v_2.md:263:**3. 4SNet - Spatial and Spectrum Self-adaptive Synergy Network**
./validate/v_2.md:267:2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。
./validate/v_2.md:307:4. **证据闭环**：它用方法表、数据集表、参数量和 Rank-1 对比、benchmark/protocol 汇总、挑战清单和 3DGait 医疗任务 case study 来证明 SRID 不是零散技巧，而是有共同问题、共同评价和外溢应用的方向。
./validate/v_2.md:389:**2. Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification**
./validate/v_2.md:393:2. 重定义动作：他们没有泛泛说“伪标签有噪声”，而是把噪声拆成三个更具体的问题：距离度量不懂相机，代理点不符合 DBSCAN 的局部密度结构，监督目标太硬。关键词是 dynamic adaptive camera Jaccard、manifold、medoid proxy、dynamic adaptive proxies、soft pseudo labels。
./validate/v_2.md:421:4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。
./validate/v_2.md:429:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:441:2. 重定义动作：它把“缩小 modality gap”重定义成“增强模型对 modality、input instance、query instance 的适应能力”。关键词是 model adaptability、flexibility、modality-aware prompts、instance-aware prompts、query-guiding prompts、adaptive feature space。这个转向很重要，因为它避开了“把两种模态硬映射到同一空间”的旧叙事。
./validate/v_2.md:457:/bin/zsh -lc 'rg -n "非对称|包含|probabil|uncertain|uncertainty|OT|optimal|PDA|MaxSim|ColBERT|Aligned|distribution|Gaussian|KL|variance|方差|因果|causal|align|对齐" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./validate/v_2.md:460:./reviews/deep_8.md:129:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:502:./reviews/deep_8.md:615:pairs while allowing the model to focus adaptively on harder and misaligned pairs. Conversely,
./validate/v_2.md:540:./reviews/deep_8.md:2329:uncertainty-based learning for text-based person retrieval.
./validate/v_2.md:561:./reviews/deep_8.md:2492:• We present an adaptive similarity distribution matching module aimed at precisely aligning
./validate/v_2.md:582:./reviews/deep_8.md:2710:pairs while allowing the model to focus adaptively on harder and misaligned pairs. Conversely,
./validate/v_2.md:649:./reviews/deep_8.md:6205:evaluate the model’s performance. Rank-k represents the probability of
./validate/v_2.md:694:./reviews/deep_8.md:9845:evaluate the model’s performance. Rank-k represents the probability of
./validate/v_2.md:696:./reviews/deep_8.md:10162:evaluated metrics, suggesting that the adaptive pull force on unmatched positive pairs is optimal.
./validate/v_2.md:715:./reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
./validate/v_2.md:769:./reviews/deep_19.md:129:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:778:./reviews/deep_19.md:552:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:780:./reviews/deep_19.md:584:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:788:./reviews/deep_19.md:976:introduces alignment noise. This underscores the need for adaptive
./validate/v_2.md:790:./reviews/deep_19.md:1529:Rank-1 in CMC measures the probability of correctly identifying the
./validate/v_2.md:791:./reviews/deep_19.md:1794:Comparison with different numbers of captions. Table 6 compares performance with varying caption numbers on PRCC CC settings. Optimal results (65.9% Rank-1) are achieved using two captions, outperforming single-caption (63.4%) and multi-caption conﬁgurations (≤ 64.2%). This demonstrates that two captions optimally balance diversity and relevance: they mitigate single-model
./validate/v_2.md:801:./reviews/deep_19.md:3519:Comparison of Rank-k and mAP Performance with State-Of-The-Art (SOTA) Methods in LTCC and PRCC. The best and second-best results are highlighted in bold and underline.
./validate/v_2.md:802:./reviews/deep_19.md:3740:Comparison of Rank-k and mAP Performance with State-Of-The-Art (SOTA) Methods in VC-Clothes. The best and second-best results are
./validate/v_2.md:837:./reviews/deep_19.md:6578:and 90.2%, respectively, both of which achieve optimal results. On the iLIDS-VID dataset, DHGN achieves a Rank1 accuracy of 94.0%, outperforming other ResNet50-based
./validate/v_2.md:838:./reviews/deep_19.md:6583:our method achieves 97.8% Rank-1 accuracy, which is significantly better than other SOTA methods.
./validate/v_2.md:856:./reviews/deep_19.md:8165:distribution visualization (Fig. 3(b)), retrieval rank-list visualization
./validate/v_2.md:877:./reviews/deep_19.md:8407:and construct a dynamically adaptive multi-granular feature alignment mechanism to further improve the generalization performance of
./validate/v_2.md:902:./reviews/deep_19.md:8993:4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。
./validate/v_2.md:910:./reviews/deep_19.md:9070:4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。
./validate/v_2.md:935:./reviews/deep_28.md:150:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:936:./reviews/deep_28.md:186:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:938:./reviews/deep_28.md:218:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:957:./reviews/deep_28.md:1619:   293	Table 1: Comparisons with SOTA methods on SYSU-MM01 and RegDB datasets. We report Rank at 𝑟 accuracy (%) and mAP (%).
./validate/v_2.md:1022:./reviews/deep_28.md:5308:1848:cost of an dramatic increase in model size and trainable parameters. Therefore, UP-Person with SDM offers an optimal tradeoff between complexity and performance without bells and
./validate/v_2.md:1067:./reviews/deep_28.md:6544:  1848	cost of an dramatic increase in model size and trainable parameters. Therefore, UP-Person with SDM offers an optimal tradeoff between complexity and performance without bells and
./validate/v_2.md:1090:./reviews/deep_27.md:431:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:1092:./reviews/deep_27.md:463:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:1102:./reviews/deep_27.md:1047:introduces adaptive style rehearsal to dynamically model domain distributions, further enhancing
./validate/v_2.md:1116:./reviews/deep_27.md:1596:Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
./validate/v_2.md:1147:./reviews/deep_27.md:3407:(𝑖, 𝑘), up to its last occurrence. The probability 𝑃𝑖𝑛𝑡𝑒𝑟 is determined by the distance rank of neighboring samples and the occurrence
./validate/v_2.md:1152:./reviews/deep_27.md:3717:positions, providing a comprehensive measure of the method’s eﬀectiveness. CMC quantiﬁes the probability of the correct match appearing within the top-ranked results. In this study, we report the Rank-1,
./validate/v_2.md:1202:./reviews/deep_27.md:5138:(𝑖, 𝑘), up to its last occurrence. The probability 𝑃𝑖𝑛𝑡𝑒𝑟 is determined by the distance rank of neighboring samples and the occurrence
./validate/v_2.md:1273:./reviews/deep_27.md:6955:Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
./validate/v_2.md:1300:./reviews/deep_23.md:416:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:1302:./reviews/deep_23.md:448:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:1456:./reviews/deep_23.md:4565:Rank-k in CMC measures the probability of the correct match
./validate/v_2.md:1460:./reviews/deep_23.md:4650:On PRCC, SCI surpasses AIM by 3.5% in Rank-1 accuracy the latent space distributions at different stages of the baseline
./validate/v_2.md:1557:./reviews/deep_23.md:6800:early-stage alignment. The Avg strategy, on the other hand,
./validate/v_2.md:1608:./reviews/deep_23.md:8406:the SOTA method IANet [18] by 4.2% Rank-1 and 5.1% mAP
./validate/v_2.md:1647:./reviews/deep_15.md:123:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:1665:./reviews/deep_15.md:1992:complexity of the feature alignment process can be reduced.
./validate/v_2.md:1712:./reviews/deep_15.md:4075:Rank-k accuracy stands for the probability of finding at least
./validate/v_2.md:1803:./reviews/deep_15.md:8800:   709	Rank-k accuracy stands for the probability of finding at least
./validate/v_2.md:1820:./reviews/deep_15.md:9322:  1586	complexity of the feature alignment process can be reduced.
./validate/v_2.md:1854:./reviews/deep_15.md:11369:   975	optimal performance at Rank@1 for the LTCC dataset was
./validate/v_2.md:1881:./reviews/deep_15.md:12842:   709	Rank-k accuracy stands for the probability of finding at least
./validate/v_2.md:1924:./reviews/deep_3.md:146:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:1930:./reviews/deep_3.md:483:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:1932:./reviews/deep_3.md:515:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:1935:./reviews/deep_3.md:708:427:   427	horizontal flipping, cropping, and erasing [30] are used. In the crossmodal alignment stage, the Adam optimizer is used with a base learning rate of 3.5e-04 and a warm-up learning rate that increases linearly
./validate/v_2.md:2001:./reviews/deep_3.md:3302:   427	horizontal flipping, cropping, and erasing [30] are used. In the crossmodal alignment stage, the Adam optimizer is used with a base learning rate of 3.5e-04 and a warm-up learning rate that increases linearly
./validate/v_2.md:2004:./reviews/deep_3.md:3575:   700	align with the data distribution encountered during the model’s extensive pre-training. When combined with k-reciprocal re-ranking,
./validate/v_2.md:2037:./reviews/deep_3.md:5228:   190	Intersection-Alignment Score (IAS) from IoU between attention maps and visible clothes masks, plus activation alignment scores. Three re-ranking methods (KNN,
./validate/v_2.md:2056:./reviews/deep_3.md:5740:   702	et al., 2015). Rank-k represents the probability that the true match
./validate/v_2.md:2057:./reviews/deep_3.md:5763:   725	Impact of Optimization Techniques. As shown in Table 4, various optimization strategies can significantly enhance the baseline performance. Our framework incorporates three key optimization techniques: KNN re-ranking (1 ), attribute-based re-ranking (2 ), and optimal weight fusion (3 ). When all optimization techniques are applied
./validate/v_2.md:2097:./reviews/deep_9.md:309:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:2100:./reviews/deep_9.md:750:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:2102:./reviews/deep_9.md:782:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:2130:./reviews/deep_9.md:2570:   191	to adaptively learn different feature distributions. However,
./validate/v_2.md:2136:./reviews/deep_9.md:2669:   290	Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
./validate/v_2.md:2169:./reviews/deep_9.md:3740:    68	gap through input-space alignment. However, generative models suffer from training instability and detail degradation, while
./validate/v_2.md:2178:./reviews/deep_9.md:4988:290:Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
./validate/v_2.md:2197:./reviews/deep_9.md:5344:   744	expanded. For instance, the SOTA performance on Singleshot Rank-1 and Multi-shot Rank-1 is improved by +9.99%
./validate/v_2.md:2222:./reviews/deep_9.md:6902:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./validate/v_2.md:2230:./reviews/deep_9.md:6949:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./validate/v_2.md:2237:./reviews/deep_12.md:191:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:2239:./reviews/deep_12.md:358:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:2241:./reviews/deep_12.md:390:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:2280:./reviews/deep_12.md:1776:[21] Zhiwei Zhao, Bin Liu, Yan Lu, et al., “Unifying multi-modal uncertainty modeling and semantic alignment for text-to-image person reidentification,” in Proceedings of the AAAI Conf. Artif. Intell., 2024,
./validate/v_2.md:2409:./reviews/deep_12.md:8218:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./validate/v_2.md:2419:./reviews/deep_12.md:8290:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./validate/v_2.md:2429:./reviews/deep_6.md:299:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:2431:./reviews/deep_6.md:489:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:2433:./reviews/deep_6.md:521:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:2453:./reviews/deep_6.md:1424:   865	through adaptive probability decay, allowing the model to
./validate/v_2.md:2471:./reviews/deep_6.md:1656:  2097	Table 2. Note that none of the methods in this experiment use reranking. Our proposed MCST outperforms existing SOTA methods,
./validate/v_2.md:2472:./reviews/deep_6.md:1658:  2103	improving the average recognition rate. As shown in Table 3, MCST outperforms SOTA methods on the SYSU-MM01 dataset. With re-ranking,
./validate/v_2.md:2482:./reviews/deep_6.md:1750:   168	label purification by adaptively adjusting the pseudo-label distribution.
./validate/v_2.md:2489:./reviews/deep_6.md:1816:   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
./validate/v_2.md:2503:./reviews/deep_6.md:2381:   199	leveraged sketch abstraction levels combined with diﬀerentiable ranking losses to improve cross-style alignment stability. The latter introduce
./validate/v_2.md:2522:./reviews/deep_6.md:2550:   759	matching, such as diverse style modeling, uncertainty-aware feature
./validate/v_2.md:2544:./reviews/deep_6.md:3665:   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
./validate/v_2.md:2568:./reviews/deep_6.md:4411:   865	through adaptive probability decay, allowing the model to
./validate/v_2.md:2614:./reviews/deep_6.md:6141:  1943	cascade dependency between CPC and OTPM, and the coarse-to-fine
./validate/v_2.md:2654:./reviews/deep_14.md:600:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:2656:./reviews/deep_14.md:632:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:2660:./reviews/deep_14.md:873:546:uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:2670:./reviews/deep_14.md:1289:164:model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:2692:./reviews/deep_14.md:1443:175:a dual-path local alignment network to extract local visual and textual representations from horizontally segmented image patches. Subsequently, local representations are aligned adaptively with a multi-stage
./validate/v_2.md:2749:./reviews/deep_14.md:3977:   164	model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:2750:./reviews/deep_14.md:3978:   165	critical pedestrian features using an uncertainty-aware self-attention
./validate/v_2.md:2770:./reviews/deep_14.md:4710:   578	R@10, respectively. The R@K indicator deﬁnes the probability of obtaining correct retrieval results within the top-K positions of the ranking
./validate/v_2.md:2784:./reviews/deep_14.md:5410:   546	uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:2791:./reviews/deep_14.md:5764:164:   164	model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:2792:./reviews/deep_14.md:5765:165:   165	critical pedestrian features using an uncertainty-aware self-attention
./validate/v_2.md:2795:./reviews/deep_14.md:5980:  1231	optimal performance with 71.8% Rank-1 and 62.3% mAP.
./validate/v_2.md:2820:./reviews/deep_13.md:589:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:2822:./reviews/deep_13.md:621:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:2871:./reviews/deep_13.md:3130:   398	In contrast, complementary logical knowledge captures the ambiguity or uncertainty within samples, thereby enhancing the model’s
./validate/v_2.md:2873:./reviews/deep_13.md:3206:   474	approach preserves sample uncertainty and stops the student network
./validate/v_2.md:2982:./reviews/deep_13.md:6939:   465	probability that the true match appears within the top-K retrieved candidates, reﬂecting the system’s retrieval accuracy at a given rank threshold
./validate/v_2.md:3003:./reviews/deep_16.md:99:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:3005:./reviews/deep_16.md:131:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:3009:./reviews/deep_16.md:383:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:3017:./reviews/deep_16.md:1019:87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
./validate/v_2.md:3102:./reviews/deep_16.md:4873:87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
./validate/v_2.md:3156:./reviews/deep_4.md:110:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:3158:./reviews/deep_4.md:142:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:3167:./reviews/deep_4.md:629:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:3171:./reviews/deep_4.md:801:each feature based on its uncertainty, highlighting the influence of more reliable features. As shown in Fig. 1(c), our
./validate/v_2.md:3172:./reviews/deep_4.md:938:uncertainty
./validate/v_2.md:3173:./reviews/deep_4.md:955:uncertainty
./validate/v_2.md:3176:./reviews/deep_4.md:1064:to each feature based on prediction uncertainty, allowing the
./validate/v_2.md:3177:./reviews/deep_4.md:1069:weight λm is determined according to the uncertainty of feature fm , which is quantified by the entropy of its prediction
./validate/v_2.md:3178:./reviews/deep_4.md:1070:distribution. The uncertainty Em is computed as follows:
./validate/v_2.md:3180:./reviews/deep_4.md:1104:individuals in the dataset. A lower Em reflects less uncertainty in the prediction, resulting in an increased weight for
./validate/v_2.md:3188:./reviews/deep_4.md:1777:all these methods by adaptively assigning weights to each feature based on the entropy of its prediction distribution.
./validate/v_2.md:3219:./reviews/deep_4.md:3961:effects. Notably, the trends in Rank-1 and mAP align, with the best
./validate/v_2.md:3221:./reviews/deep_4.md:4038:alignment and uncertainty estimation for visible-infrared person re-identification,
./validate/v_2.md:3339:./reviews/deep_4.md:7198:uncertainty replayer strategy, which focuses on storing samples with
./validate/v_2.md:3341:./reviews/deep_4.md:7218:model adaptive framework that adapts to the data distribution of new
./validate/v_2.md:3345:./reviews/deep_4.md:7401:(AKT) framework, which conducts the simulated distribution for facilitating the adaptive knowledge distillation. The framework of AKT can
./validate/v_2.md:3360:./reviews/deep_4.md:8843:(AKT) framework, which conducts the simulated distribution for facilitating the adaptive knowledge distillation. The framework of AKT can
./validate/v_2.md:3423:./reviews/deep_4.md:10818:to each feature based on prediction uncertainty, allowing the
./validate/v_2.md:3424:./reviews/deep_4.md:10823:weight λm is determined according to the uncertainty of feature fm , which is quantified by the entropy of its prediction
./validate/v_2.md:3425:./reviews/deep_4.md:10824:distribution. The uncertainty Em is computed as follows:
./validate/v_2.md:3427:./reviews/deep_4.md:10858:individuals in the dataset. A lower Em reflects less uncertainty in the prediction, resulting in an increased weight for
./validate/v_2.md:3448:./reviews/deep_4.md:11173:1015:60.5% Rank-1 and 59.4% mAP, which is close to the groundtruth performance. These results demonstrate that our causal
./validate/v_2.md:3472:./reviews/deep_4.md:11401:4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。
./validate/v_2.md:3478:./reviews/deep_4.md:11478:4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。
./validate/v_2.md:3487:./reviews/deep_2.md:664:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:3489:./reviews/deep_2.md:696:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:3586:./reviews/deep_2.md:3217:conditions, reducing complexity; (2) To align with common practices in
./validate/v_2.md:3614:./reviews/deep_2.md:4690:pose uncertainty, appearance changes, and scene complexity
./validate/v_2.md:3720:./reviews/deep_2.md:8952:conditions, reducing complexity; (2) To align with common practices in
./validate/v_2.md:3733:./reviews/deep_2.md:9938:cues, and later stages stabilize rankings under broader identity consensus, which is particularly beneficial on large-scale or distributionshifted datasets.
./validate/v_2.md:3830:./reviews/deep_2.md:13237:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./validate/v_2.md:3838:./reviews/deep_2.md:13269:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./validate/v_2.md:3845:./reviews/deep_2.md:13318:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./validate/v_2.md:3853:./reviews/deep_2.md:13350:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./validate/v_2.md:3867:./validate/v_1.md:30:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:3871:./validate/v_1.md:38:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:3872:./validate/v_1.md:40:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:3931:./reviews/deep_7.md:1907:   152	dual-constrained top-ranking loss for modality alignment. Adversarial learning emerged as another promising direction in VI-ReID. The
./validate/v_2.md:4030:./reviews/K-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:4038:./reviews/deep_22.md:11315:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./validate/v_2.md:4045:./reviews/deep_22.md:11344:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:4053:./reviews/deep_22.md:11392:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./validate/v_2.md:4068:./reviews/deep_26.md:422:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:4070:./reviews/deep_26.md:454:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:4132:./reviews/deep_26.md:3476:As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
./validate/v_2.md:4152:./reviews/deep_26.md:4718:of the two-stage feature alignment strategy, we show the Rank-1 and
./validate/v_2.md:4193:./reviews/deep_26.md:8488:As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
./validate/v_2.md:4213:./reviews/deep_26.md:9730:of the two-stage feature alignment strategy, we show the Rank-1 and
./validate/v_2.md:4251:./reviews/deep_26.md:12549:with a focus on rank-1 accuracy. CMC-k measures the probability that
./validate/v_2.md:4280:./analyses/deep_2.txt:11:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./validate/v_2.md:4288:./analyses/deep_2.txt:43:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./validate/v_2.md:4295:./reviews/deep_17.md:140:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:4296:./reviews/deep_17.md:188:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:4298:./reviews/deep_17.md:220:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:4381:./reviews/deep_17.md:5360:881:   881	and Rank-1 are both optimal. When λ surpasses 1, the DFC
./validate/v_2.md:4398:./reviews/deep_20.md:327:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:4401:./reviews/deep_20.md:750:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:4403:./reviews/deep_20.md:782:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:4441:./reviews/deep_20.md:2114:   168	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
./validate/v_2.md:4448:./reviews/deep_20.md:2274:   126	strategy and local averaging technique, this method enhances feature alignment and similaritybased feature refinement, similar in effect to the re-ranking [57] technique. Specifically, given a set
./validate/v_2.md:4450:./reviews/deep_20.md:2316:   168	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
./validate/v_2.md:4481:./reviews/deep_20.md:2947:   679	the prediction space, achieving adaptive alignment across
./validate/v_2.md:4524:./reviews/deep_20.md:4695:   462	measures the uncertainty or complexity of the tasks. The mutual information between the task and the injected noise is given by
./validate/v_2.md:4525:./reviews/deep_20.md:4716:   483	task by reducing uncertainty rather than introducing disturbance.
./validate/v_2.md:4537:./reviews/deep_20.md:5102:   130	of VI-ReID into two aspects: (i) difficulty in aligning lowand high-level semantics across modalities, and (ii) distribution discrepancies between VIS and IR features. To address
./validate/v_2.md:4548:./reviews/deep_20.md:5255:   613	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
./validate/v_2.md:4555:./reviews/deep_20.md:5444:  1283	the HAR-enhanced models in the ﬁnal ranks. AlignedReID performs the
./validate/v_2.md:4559:./reviews/deep_20.md:5496:  1599	While GaitGL_G maintains a strong baseline (76.5 % Rank-1), it converges more slowly than the HAR-fused models. AlignedReID again performs the weakest at early and mid ranks, achieving only 64.7 % Rank-1
./validate/v_2.md:4560:./reviews/deep_20.md:5508:  1762	gait and HAR embeddings. Cross-attention adaptively emphasizes behavioral cues conditioned on gait, while cosine similarity enforces alignment by maximizing directional closeness between modalities. In contrast, concatenation stacks the embeddings without explicit interaction,
./validate/v_2.md:4600:./reviews/deep_18.md:135:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:4605:./reviews/deep_18.md:356:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:4607:./reviews/deep_18.md:388:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:4697:./analyses/deep_1.txt:31:2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。
./validate/v_2.md:4706:./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:4734:./analyses/deep_9.txt:7:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./validate/v_2.md:4764:./reviews/deep_25.md:282:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:4766:./reviews/deep_25.md:314:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:4796:./reviews/deep_25.md:2310:where ⊙ denotes low-pass ﬁltering with adaptive Gaussian smoothed
./validate/v_2.md:4840:./reviews/deep_25.md:3446:not align with our task’s requirements. More recently, Zhou et al. introduced MixStyle (Zhou et al., 2023), a method that linearly combines the
./validate/v_2.md:4922:./reviews/deep_25.md:5735:More recent studies have concentrated on resolving modal differences at a deeper level. S. Li et al. (2024) proposed an uncertaintyaware progressive learning framework to progressively eliminate
./validate/v_2.md:4969:./reviews/deep_25.md:10265:where ⊙ denotes low-pass ﬁltering with adaptive Gaussian smoothed
./validate/v_2.md:5000:./reviews/deep_25.md:10689:not align with our task’s requirements. More recently, Zhou et al. introduced MixStyle (Zhou et al., 2023), a method that linearly combines the
./validate/v_2.md:5047:./reviews/deep_25.md:12906:/bin/zsh -lc "pdftotext -f 8 -l 14 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' - | rg -n \"(Ablation|ablation|ATG|CTP|TRM|prototype|Prototype|attribute|contour|text|Text|SYSU|RegDB|Table|Figure|Fig\\.|Effect|w/o|without|modal|alignment|Rank|mAP|loss|TAP)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./validate/v_2.md:5097:./reviews/deep_25.md:14029:TAP-align 的第 14 页跨页断在 TRM 说明处，表格已经给出了全模型从 74.86 mAP 到 76.21 mAP、rank-1 从 77.85 到 78.52 的提升。这个足够支撑 TRM 的闭环，我会以表格数值为准，不再扩大阅读范围。
./validate/v_2.md:5098:./reviews/deep_25.md:14041:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:5101:./reviews/deep_25.md:14053:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./validate/v_2.md:5106:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./validate/v_2.md:5110:./reviews/deep_25.md:14120:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:5113:./reviews/deep_25.md:14132:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./validate/v_2.md:5118:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./validate/v_2.md:5128:./analyses/deep_19.txt:59:4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。
./validate/v_2.md:5133:./analyses/deep_22.txt:11:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:5141:./analyses/deep_22.txt:59:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./validate/v_2.md:5160:./analyses/deep_4.txt:71:4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。
./validate/v_2.md:5162:./analyses/deep_25.txt:11:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:5165:./analyses/deep_25.txt:23:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./validate/v_2.md:5170:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./validate/v_2.md:5176:./reviews/deep_24.md:127:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:5183:./reviews/deep_24.md:766:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:5185:./reviews/deep_24.md:798:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:5203:./reviews/deep_24.md:3096:1151-accuracy of 82.4%, 85.4%, and 82.2% for the Rank-1, mAP, and mINP indicators, respectively. Furthermore, our method outperforms the suboptimal method by 0.9%(2.1%), 0.8%( 2.1%), and 7.9%
./validate/v_2.md:5204:./reviews/deep_24.md:3101:1156-achieved 91.8% and 85.3% accuracy in Rank-1 and mAP indicators, respectively, and has demonstrated superiority over the suboptimal SEFL approach by 0.7% in terms of Rank-1. Similarly, in
./validate/v_2.md:5205:./reviews/deep_24.md:3103:1358-the suboptimal CST approach by 0.7% in terms of Rank-1 in the “VIS to IR” mode. Furthermore,
./validate/v_2.md:5206:./reviews/deep_24.md:3105:1360-to VIS” (“VIS to IR”) mode, surpassing the suboptimal SEFL approach by 3.5% (3.1%) in Rank-1
./validate/v_2.md:5207:./reviews/deep_24.md:3166:1495:meaningful shape cues, leading to suboptimal performance (Rank-1: 74.6%, mAP: 71.6%). Setting 2
./validate/v_2.md:5212:./reviews/deep_24.md:3591:the suboptimal CST approach by 0.7% in terms of Rank-1 in the “VIS to IR” mode. Furthermore,
./validate/v_2.md:5213:./reviews/deep_24.md:3593:to VIS” (“VIS to IR”) mode, surpassing the suboptimal SEFL approach by 3.5% (3.1%) in Rank-1
./validate/v_2.md:5214:./reviews/deep_24.md:3728:meaningful shape cues, leading to suboptimal performance (Rank-1: 74.6%, mAP: 71.6%). Setting 2
./validate/v_2.md:5269:./reviews/deep_24.md:4792:text–image similarity. Rank-k is the probability of finding at least one matching person image in
./validate/v_2.md:5310:./reviews/deep_24.md:5393:715:text–image similarity. Rank-k is the probability of finding at least one matching person image in
./validate/v_2.md:5388:./reviews/deep_24.md:14164:further increases the difficulty of cross-modal alignment.
./validate/v_2.md:5396:./reviews/deep_24.md:14255:the constraint of IDs, we align features between different modalities with the same ID adaptively according to
./validate/v_2.md:5399:./reviews/deep_24.md:14264:by ID limitations. It adaptively establishes alignment relationships between various attributes and different images. Finally,
./validate/v_2.md:5438:./reviews/deep_24.md:14739:IDs to adaptively explore attribute alignment across different
./validate/v_2.md:5471:./reviews/deep_24.md:15398:IDs to adaptively explore attribute alignment across different
./validate/v_2.md:5482:./reviews/deep_24.md:15703:75-further increases the difficulty of cross-modal alignment.
./validate/v_2.md:5490:./reviews/deep_24.md:15786:166-the constraint of IDs, we align features between different modalities with the same ID adaptively according to
./validate/v_2.md:5493:./reviews/deep_24.md:15795:175:by ID limitations. It adaptively establishes alignment relationships between various attributes and different images. Finally,
./validate/v_2.md:5515:./reviews/deep_24.md:16026:650:IDs to adaptively explore attribute alignment across different
./validate/v_2.md:5592:./reviews/deep_24.md:17072:4. 证据闭环：RSSPA 是主证据，CUHK 上 Rank-1 提升超过 3%，mAP 约 7%，mINP 约 11%。关系敏感对齐比传统单映射多 2.76 Rank-1、6.16 mAP、10.28 mINP。IBSPA、IFAA、APFR 继续带来增益。作者还比较了 RSSPA+IBSPA 和 SDM/CMPM，属性实验显示多粒度属性优于词级属性，属性数量过多反而伤害性能。
./validate/v_2.md:5597:./reviews/deep_24.md:17129:4. 证据闭环：RSSPA 是主证据，CUHK 上 Rank-1 提升超过 3%，mAP 约 7%，mINP 约 11%。关系敏感对齐比传统单映射多 2.76 Rank-1、6.16 mAP、10.28 mINP。IBSPA、IFAA、APFR 继续带来增益。作者还比较了 RSSPA+IBSPA 和 SDM/CMPM，属性实验显示多粒度属性优于词级属性，属性数量过多反而伤害性能。
./validate/v_2.md:5603:./analyses/deep_21.txt:31:2. **重定义动作**：他们把问题从“如何生成更好伪标签”改成“如何选择最有训练价值的 proxy 对”。关键词是 `optimal proxy mining`、`hardest camera proxy`、`false negative proxies mining`、`misaligned contrastive pairs`、`camera-constrained hard negative mining`、`anti-noise ranking`。这让创新点落在训练信号选择，而不是聚类算法本身。
./validate/v_2.md:5607:./analyses/deep_7.txt:47:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
./validate/v_2.md:5611:./analyses/deep_7.txt:59:4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。
./validate/v_2.md:5619:./reviews/deep_5.md:295:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:5622:./reviews/deep_5.md:733:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:5624:./reviews/deep_5.md:765:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:5667:./reviews/deep_5.md:3792:CAFMNet model achieves optimal performance: 77.49% Rank-1 accuracy and 74.19% mAP. In this configuration, CLFO provides the
./validate/v_2.md:5671:./reviews/deep_5.md:4164:aligning with CLFO’s goal of channel-level feature optimization. Removing DSC in favor of standard convolutions increases Rank-1 to
./validate/v_2.md:5672:./reviews/deep_5.md:4177:an optimal balance, reaching 72.11% Rank-1 and 69.19% mAP with a
./validate/v_2.md:5711:./reviews/deep_5.md:5840:embeddings, which are then aligned with the text. In terms of Rank1 accuracy, the performance of the vanilla PCB drops by 1.32% and
./validate/v_2.md:5721:./reviews/deep_5.md:6308:inability to adequately fit the data distribution, failing to capture the complexity and diversity of
./validate/v_2.md:5803:./analyses/ALL_ANALYSES.txt:33:2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。
./validate/v_2.md:5821:./analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:5831:./analyses/ALL_ANALYSES.txt:247:**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。
./validate/v_2.md:5861:./analyses/ALL_ANALYSES.txt:676:4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。
./validate/v_2.md:5864:./analyses/ALL_ANALYSES.txt:705:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./validate/v_2.md:5872:./analyses/ALL_ANALYSES.txt:737:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./validate/v_2.md:5889:./analyses/ALL_ANALYSES.txt:883:2. **重定义动作**：他们把问题从“如何生成更好伪标签”改成“如何选择最有训练价值的 proxy 对”。关键词是 `optimal proxy mining`、`hardest camera proxy`、`false negative proxies mining`、`misaligned contrastive pairs`、`camera-constrained hard negative mining`、`anti-noise ranking`。这让创新点落在训练信号选择，而不是聚类算法本身。
./validate/v_2.md:5893:./analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:5901:./analyses/ALL_ANALYSES.txt:992:4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./validate/v_2.md:5920:./analyses/ALL_ANALYSES.txt:1108:4. 证据闭环：RSSPA 是主证据，CUHK 上 Rank-1 提升超过 3%，mAP 约 7%，mINP 约 11%。关系敏感对齐比传统单映射多 2.76 Rank-1、6.16 mAP、10.28 mINP。IBSPA、IFAA、APFR 继续带来增益。作者还比较了 RSSPA+IBSPA 和 SDM/CMPM，属性实验显示多粒度属性优于词级属性，属性数量过多反而伤害性能。
./validate/v_2.md:5921:./analyses/ALL_ANALYSES.txt:1128:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:5924:./analyses/ALL_ANALYSES.txt:1140:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./validate/v_2.md:5929:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./validate/v_2.md:5967:./analyses/ALL_ANALYSES.txt:1542:4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。
./validate/v_2.md:5986:./analyses/ALL_ANALYSES.txt:1748:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
./validate/v_2.md:5990:./analyses/ALL_ANALYSES.txt:1760:4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。
./validate/v_2.md:6010:./analyses/ALL_ANALYSES.txt:1836:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./validate/v_2.md:6022:./reviews/deep_10.md:420:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:6024:./reviews/deep_10.md:452:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:6081:./reviews/deep_10.md:2385:value of weights 𝜆 in an adaptive manner. Specifically, we first leverage the Wasserstein distance to measure the distribution discrepancy
./validate/v_2.md:6149:./reviews/deep_10.md:3852:parameters, resulting in increasing training costs and computation complexity. Besides, the image alignment approaches
./validate/v_2.md:6188:./reviews/deep_10.md:4648:adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
./validate/v_2.md:6302:./reviews/deep_10.md:7465:33:adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
./validate/v_2.md:6314:./reviews/deep_10.md:7573:1030:Our framework has two important components: DSH in early bottlenecks to bring about style variance, and the DED loss 𝐷𝐸𝐷 to enhance
./validate/v_2.md:6341:./reviews/deep_10.md:8945:modeling in baseline with causal modeling, the rank-1 accuracy and mAP of all-search
./validate/v_2.md:6421:./reviews/deep_1.md:119:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:6464:./reviews/deep_1.md:2723:reaches cross modality alignment with adaptive fusion. The
./validate/v_2.md:6545:./reviews/deep_1.md:4625:626-reaches cross modality alignment with adaptive fusion. The
./validate/v_2.md:6551:./reviews/deep_1.md:4775:841-Rank-1 accuracy among existing SOTA methods. Specifically,
./validate/v_2.md:6573:./reviews/deep_1.md:5537:reaches cross modality alignment with adaptive fusion. The                     expressed as:
./validate/v_2.md:6579:./reviews/deep_1.md:5656:cloud followed by [68]. For the Spatial-Temporal Aggregation                  Rank-1 accuracy among existing SOTA methods. Specifically,
./validate/v_2.md:6580:./reviews/deep_1.md:5677:competitive performance compare to other SOTA methods.                        Rank-1 accuracy. It demonstrates that as a new and discrim-
./validate/v_2.md:6630:./reviews/deep_1.md:10885:2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。
./validate/v_2.md:6639:./reviews/deep_1.md:10966:2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。
./validate/v_2.md:6644:./reviews/deep_11.md:127:- Charts, tables, and images must be sharp, aligned, and clearly labeled.
./validate/v_2.md:6649:./reviews/deep_11.md:394:**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。
./validate/v_2.md:6651:./reviews/deep_11.md:426:- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
./validate/v_2.md:6658:./reviews/deep_11.md:1423:   162	features in the target domain. Some uncertainty-based ReID methods [74, 76] have been proposed
./validate/v_2.md:6659:./reviews/deep_11.md:1425:   164	of uncertainty in pseudo labels is challenging in the unsupervised scenario, thereby complicating
./validate/v_2.md:6722:./reviews/deep_11.md:4895:   341	noise perturbation is used to enhance the style of each task, which is called style uncertainty
./validate/v_2.md:6737:./reviews/deep_11.md:5335:   322	progressively increasing difficulty. In the initial stage, the Gaussian noise intensities m1 and m2 are
./validate/v_2.md:6747:./reviews/deep_11.md:5648:   176	discriminative information in an adaptive feature space and overcoming outof-distribution samples, thereby improving identification performance.
./validate/v_2.md:6775:./reviews/deep_11.md:8421:   278	Han, J., Li, Y.-L., & Wang, S. (2022a). Delving into probabilistic uncertainty for unsupervised domain adaptive person re-identiﬁcation. Proceedings of the AAAI Conference on
./validate/v_2.md:6797:./reviews/deep_11.md:10125:   386	Qualitative ablation studies of our dynamic adaptive optimal proxy with
./validate/v_2.md:6799:./reviews/deep_11.md:10273:     6	Qualitative ablation studies of our dynamic adaptive optimal proxy with
./validate/v_2.md:6802:./reviews/deep_11.md:10503:   236	(Table 5), the dual local view conﬁguration achieves optimal eﬃciencyaccuracy equilibrium. The dual local views method achieves the highest mAP and Rank-1 accuracy on Market-1501 and MSMT17, and the
./validate/v_2.md:6806:./reviews/deep_11.md:10684:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:6814:./reviews/deep_11.md:10763:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:6829:/bin/zsh -lc "rg -n \"PDA|probabilistic|Probability|Probabil|distribution|variance|uncertainty|text.*image|文本分布|方差=|⊇|⊆|Optimal Transport|OT-ReID|Wasserstein|EMD|CM-EMD|G2DA\" all_papers.txt analyses reviews validate -g '*.txt' -g '*.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./validate/v_2.md:6836:analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:6875:reviews/deep_8.md:2329:uncertainty-based learning for text-based person retrieval.
./validate/v_2.md:6882:reviews/deep_8.md:2492:• We present an adaptive similarity distribution matching module aimed at precisely aligning
./validate/v_2.md:6917:reviews/deep_8.md:5438:Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
./validate/v_2.md:6945:reviews/deep_8.md:8808:Sigmoid activation function. This operation enables the text feature to adaptively generate channelwise attention over the image representation, allowing semantic guidance to modulate the visual
./validate/v_2.md:6969:reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
./validate/v_2.md:7016:validate/v_3.md:1790:./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:7019:validate/v_3.md:1928:../codex_review_ovli.txt:14:Review experiments/cargo_cvpb/afd_train.py, focus on the NEW --ovli OVLIHead (Opposite-View Late-Interaction Evidence Alignment): token proj (Conv2d 2048->256 on a layer4 feature-map hook, adaptive-pool to 8x4 grid, L2-norm per token), opposite-view supervised-contrastive loss score=alpha*cos(global)+(1-alpha)*sym_MaxSim(bidirectional mean-max), logsumexp multi-positive, tau0.05 alpha0.5, lambda warmup --ovli_warmup. 逐行查: bug/runtime/shape/device/AMP数值安全; ★proj新参数是否正确加进optimizer(list(model.parameters())+list(ovli.parameters())+assert自检, 这是关键); MaxSim/logsumexp数值(无NaN, empty-candidate用-1e4 floor); train/test对称(eval默认global-only, --ovli_rerank可选); --ovli off精确复现baseline; --ovp/--ovli互斥。用web search查novelty撞车: opposite-view/cross-view token-set late-interaction(ColBERT/MaxSim)作为training-time cross-view retrieval loss 在 aerial-ground 或 person ReID 的先例(对比CM-EMD/G2DA VI-ReID OT, CVFT geo-localization, DTST token selection, CMPC prototype)。输出 Verdict(approve/needs-attention)+findings(severity Critical/High/Medium/Low)。
./validate/v_2.md:7024:validate/v_3.md:2883:../litreview2/analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:7076:reviews/deep_19.md:8165:distribution visualization (Fig. 3(b)), retrieval rank-list visualization
./validate/v_2.md:7085:analyses/deep_25.txt:53:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./validate/v_2.md:7182:analyses/deep_22.txt:11:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:7206:reviews/deep_9.md:2570:   191	to adaptively learn different feature distributions. However,
./validate/v_2.md:7212:reviews/deep_9.md:2669:   290	Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
./validate/v_2.md:7235:reviews/deep_9.md:4988:290:Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
./validate/v_2.md:7262:analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:7271:analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:7276:analyses/ALL_ANALYSES.txt:1170:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./validate/v_2.md:7286:validate/v_1.md:30:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:7290:validate/v_1.md:38:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:7291:validate/v_1.md:40:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:7292:validate/v_1.md:42:web search: "person re-identification" "aleatoric uncertainty"
./validate/v_2.md:7293:validate/v_1.md:48:web search: "Robust person re-identification" "feature uncertainty"
./validate/v_2.md:7294:validate/v_1.md:50:web search: "feature uncertainty" "ReID" "Gaussian" "posterior"
./validate/v_2.md:7306:validate/v_2.md:429:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:7307:validate/v_2.md:457:/bin/zsh -lc 'rg -n "非对称|包含|probabil|uncertain|uncertainty|OT|optimal|PDA|MaxSim|ColBERT|Aligned|distribution|Gaussian|KL|variance|方差|因果|causal|align|对齐" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./validate/v_2.md:7319:validate/v_2.md:540:./reviews/deep_8.md:2329:uncertainty-based learning for text-based person retrieval.
./validate/v_2.md:7322:validate/v_2.md:561:./reviews/deep_8.md:2492:• We present an adaptive similarity distribution matching module aimed at precisely aligning
./validate/v_2.md:7360:validate/v_2.md:715:./reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
./validate/v_2.md:7394:validate/v_2.md:856:./reviews/deep_19.md:8165:distribution visualization (Fig. 3(b)), retrieval rank-list visualization
./validate/v_2.md:7481:validate/v_2.md:1102:./reviews/deep_27.md:1047:introduces adaptive style rehearsal to dynamically model domain distributions, further enhancing
./validate/v_2.md:7492:validate/v_2.md:1116:./reviews/deep_27.md:1596:Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
./validate/v_2.md:7559:validate/v_2.md:1273:./reviews/deep_27.md:6955:Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
./validate/v_2.md:7639:validate/v_2.md:1460:./reviews/deep_23.md:4650:On PRCC, SCI surpasses AIM by 3.5% in Rank-1 accuracy the latent space distributions at different stages of the baseline
./validate/v_2.md:7759:validate/v_2.md:2004:./reviews/deep_3.md:3575:   700	align with the data distribution encountered during the model’s extensive pre-training. When combined with k-reciprocal re-ranking,
./validate/v_2.md:7778:validate/v_2.md:2130:./reviews/deep_9.md:2570:   191	to adaptively learn different feature distributions. However,
./validate/v_2.md:7783:validate/v_2.md:2136:./reviews/deep_9.md:2669:   290	Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
./validate/v_2.md:7797:validate/v_2.md:2178:./reviews/deep_9.md:4988:290:Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
./validate/v_2.md:7825:validate/v_2.md:2280:./reviews/deep_12.md:1776:[21] Zhiwei Zhao, Bin Liu, Yan Lu, et al., “Unifying multi-modal uncertainty modeling and semantic alignment for text-to-image person reidentification,” in Proceedings of the AAAI Conf. Artif. Intell., 2024,
./validate/v_2.md:7891:validate/v_2.md:2482:./reviews/deep_6.md:1750:   168	label purification by adaptively adjusting the pseudo-label distribution.
./validate/v_2.md:7894:validate/v_2.md:2489:./reviews/deep_6.md:1816:   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
./validate/v_2.md:7903:validate/v_2.md:2522:./reviews/deep_6.md:2550:   759	matching, such as diverse style modeling, uncertainty-aware feature
./validate/v_2.md:7910:validate/v_2.md:2544:./reviews/deep_6.md:3665:   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
./validate/v_2.md:7938:validate/v_2.md:2660:./reviews/deep_14.md:873:546:uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:7942:validate/v_2.md:2670:./reviews/deep_14.md:1289:164:model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:7947:validate/v_2.md:2692:./reviews/deep_14.md:1443:175:a dual-path local alignment network to extract local visual and textual representations from horizontally segmented image patches. Subsequently, local representations are aligned adaptively with a multi-stage
./validate/v_2.md:7962:validate/v_2.md:2749:./reviews/deep_14.md:3977:   164	model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:7963:validate/v_2.md:2750:./reviews/deep_14.md:3978:   165	critical pedestrian features using an uncertainty-aware self-attention
./validate/v_2.md:7977:validate/v_2.md:2784:./reviews/deep_14.md:5410:   546	uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:7982:validate/v_2.md:2791:./reviews/deep_14.md:5764:164:   164	model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:7983:validate/v_2.md:2792:./reviews/deep_14.md:5765:165:   165	critical pedestrian features using an uncertainty-aware self-attention
./validate/v_2.md:7994:validate/v_2.md:2871:./reviews/deep_13.md:3130:   398	In contrast, complementary logical knowledge captures the ambiguity or uncertainty within samples, thereby enhancing the model’s
./validate/v_2.md:7995:validate/v_2.md:2873:./reviews/deep_13.md:3206:   474	approach preserves sample uncertainty and stops the student network
./validate/v_2.md:8013:validate/v_2.md:3017:./reviews/deep_16.md:1019:87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
./validate/v_2.md:8032:validate/v_2.md:3102:./reviews/deep_16.md:4873:87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
./validate/v_2.md:8057:validate/v_2.md:3171:./reviews/deep_4.md:801:each feature based on its uncertainty, highlighting the influence of more reliable features. As shown in Fig. 1(c), our
./validate/v_2.md:8058:validate/v_2.md:3172:./reviews/deep_4.md:938:uncertainty
./validate/v_2.md:8059:validate/v_2.md:3173:./reviews/deep_4.md:955:uncertainty
./validate/v_2.md:8061:validate/v_2.md:3176:./reviews/deep_4.md:1064:to each feature based on prediction uncertainty, allowing the
./validate/v_2.md:8062:validate/v_2.md:3177:./reviews/deep_4.md:1069:weight λm is determined according to the uncertainty of feature fm , which is quantified by the entropy of its prediction
./validate/v_2.md:8063:validate/v_2.md:3178:./reviews/deep_4.md:1070:distribution. The uncertainty Em is computed as follows:
./validate/v_2.md:8064:validate/v_2.md:3180:./reviews/deep_4.md:1104:individuals in the dataset. A lower Em reflects less uncertainty in the prediction, resulting in an increased weight for
./validate/v_2.md:8066:validate/v_2.md:3188:./reviews/deep_4.md:1777:all these methods by adaptively assigning weights to each feature based on the entropy of its prediction distribution.
./validate/v_2.md:8079:validate/v_2.md:3221:./reviews/deep_4.md:4038:alignment and uncertainty estimation for visible-infrared person re-identification,
./validate/v_2.md:8108:validate/v_2.md:3339:./reviews/deep_4.md:7198:uncertainty replayer strategy, which focuses on storing samples with
./validate/v_2.md:8109:validate/v_2.md:3341:./reviews/deep_4.md:7218:model adaptive framework that adapts to the data distribution of new
./validate/v_2.md:8112:validate/v_2.md:3345:./reviews/deep_4.md:7401:(AKT) framework, which conducts the simulated distribution for facilitating the adaptive knowledge distillation. The framework of AKT can
./validate/v_2.md:8121:validate/v_2.md:3360:./reviews/deep_4.md:8843:(AKT) framework, which conducts the simulated distribution for facilitating the adaptive knowledge distillation. The framework of AKT can
./validate/v_2.md:8141:validate/v_2.md:3423:./reviews/deep_4.md:10818:to each feature based on prediction uncertainty, allowing the
./validate/v_2.md:8142:validate/v_2.md:3424:./reviews/deep_4.md:10823:weight λm is determined according to the uncertainty of feature fm , which is quantified by the entropy of its prediction
./validate/v_2.md:8143:validate/v_2.md:3425:./reviews/deep_4.md:10824:distribution. The uncertainty Em is computed as follows:
./validate/v_2.md:8144:validate/v_2.md:3427:./reviews/deep_4.md:10858:individuals in the dataset. A lower Em reflects less uncertainty in the prediction, resulting in an increased weight for
./validate/v_2.md:8168:validate/v_2.md:3614:./reviews/deep_2.md:4690:pose uncertainty, appearance changes, and scene complexity
./validate/v_2.md:8197:validate/v_2.md:3733:./reviews/deep_2.md:9938:cues, and later stages stabilize rankings under broader identity consensus, which is particularly beneficial on large-scale or distributionshifted datasets.
./validate/v_2.md:8233:validate/v_2.md:3867:./validate/v_1.md:30:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:8237:validate/v_2.md:3871:./validate/v_1.md:38:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:8238:validate/v_2.md:3872:./validate/v_1.md:40:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:8269:validate/v_2.md:4030:./reviews/K-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:8273:validate/v_2.md:4045:./reviews/deep_22.md:11344:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:8301:validate/v_2.md:4132:./reviews/deep_26.md:3476:As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
./validate/v_2.md:8332:validate/v_2.md:4193:./reviews/deep_26.md:8488:As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
./validate/v_2.md:8439:validate/v_2.md:4524:./reviews/deep_20.md:4695:   462	measures the uncertainty or complexity of the tasks. The mutual information between the task and the injected noise is given by
./validate/v_2.md:8440:validate/v_2.md:4525:./reviews/deep_20.md:4716:   483	task by reducing uncertainty rather than introducing disturbance.
./validate/v_2.md:8447:validate/v_2.md:4537:./reviews/deep_20.md:5102:   130	of VI-ReID into two aspects: (i) difficulty in aligning lowand high-level semantics across modalities, and (ii) distribution discrepancies between VIS and IR features. To address
./validate/v_2.md:8483:validate/v_2.md:4706:./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:8554:validate/v_2.md:4922:./reviews/deep_25.md:5735:More recent studies have concentrated on resolving modal differences at a deeper level. S. Li et al. (2024) proposed an uncertaintyaware progressive learning framework to progressively eliminate
./validate/v_2.md:8614:validate/v_2.md:5133:./analyses/deep_22.txt:11:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:8636:validate/v_2.md:5269:./reviews/deep_24.md:4792:text–image similarity. Rank-k is the probability of finding at least one matching person image in
./validate/v_2.md:8645:validate/v_2.md:5310:./reviews/deep_24.md:5393:715:text–image similarity. Rank-k is the probability of finding at least one matching person image in
./validate/v_2.md:8755:validate/v_2.md:5721:./reviews/deep_5.md:6308:inability to adequately fit the data distribution, failing to capture the complexity and diversity of
./validate/v_2.md:8769:validate/v_2.md:5821:./analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:8775:validate/v_2.md:5893:./analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:8807:validate/v_2.md:6081:./reviews/deep_10.md:2385:value of weights 𝜆 in an adaptive manner. Specifically, we first leverage the Wasserstein distance to measure the distribution discrepancy
./validate/v_2.md:8827:validate/v_2.md:6188:./reviews/deep_10.md:4648:adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
./validate/v_2.md:8885:validate/v_2.md:6302:./reviews/deep_10.md:7465:33:adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
./validate/v_2.md:8895:validate/v_2.md:6314:./reviews/deep_10.md:7573:1030:Our framework has two important components: DSH in early bottlenecks to bring about style variance, and the DED loss 𝐷𝐸𝐷 to enhance
./validate/v_2.md:8960:validate/v_2.md:6658:./reviews/deep_11.md:1423:   162	features in the target domain. Some uncertainty-based ReID methods [74, 76] have been proposed
./validate/v_2.md:8961:validate/v_2.md:6659:./reviews/deep_11.md:1425:   164	of uncertainty in pseudo labels is challenging in the unsupervised scenario, thereby complicating
./validate/v_2.md:8993:validate/v_2.md:6722:./reviews/deep_11.md:4895:   341	noise perturbation is used to enhance the style of each task, which is called style uncertainty
./validate/v_2.md:9012:validate/v_2.md:6747:./reviews/deep_11.md:5648:   176	discriminative information in an adaptive feature space and overcoming outof-distribution samples, thereby improving identification performance.
./validate/v_2.md:9017:validate/v_2.md:6775:./reviews/deep_11.md:8421:   278	Han, J., Li, Y.-L., & Wang, S. (2022a). Delving into probabilistic uncertainty for unsupervised domain adaptive person re-identiﬁcation. Proceedings of the AAAI Conference on
./validate/v_2.md:9031:validate/v_2.md:6806:./reviews/deep_11.md:10684:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:9032:validate/v_2.md:6814:./reviews/deep_11.md:10763:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:9156:reviews/deep_23.md:4650:On PRCC, SCI surpasses AIM by 3.5% in Rank-1 accuracy the latent space distributions at different stages of the baseline
./validate/v_2.md:9242:reviews/deep_12.md:1776:[21] Zhiwei Zhao, Bin Liu, Yan Lu, et al., “Unifying multi-modal uncertainty modeling and semantic alignment for text-to-image person reidentification,” in Proceedings of the AAAI Conf. Artif. Intell., 2024,
./validate/v_2.md:9343:reviews/deep_21.md:5437:   267	[13] H. Zheng et al., “Cascade transformer reasoning embedded by uncertainty for occluded person re-identification,” IEEE Trans. Biometrics,
./validate/v_2.md:9434:reviews/deep_5.md:5584:re-ranking. The final similarity for text–image pairs is computed as
./validate/v_2.md:9440:reviews/deep_5.md:6308:inability to adequately fit the data distribution, failing to capture the complexity and diversity of
./validate/v_2.md:9465:reviews/deep_5.md:8339:re-ranking. The final similarity for text–image pairs is computed as
./validate/v_2.md:9480:reviews/deep_27.md:1047:introduces adaptive style rehearsal to dynamically model domain distributions, further enhancing
./validate/v_2.md:9491:reviews/deep_27.md:1596:Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
./validate/v_2.md:9570:reviews/deep_27.md:6955:Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
./validate/v_2.md:9587:reviews/deep_13.md:3130:   398	In contrast, complementary logical knowledge captures the ambiguity or uncertainty within samples, thereby enhancing the model’s
./validate/v_2.md:9588:reviews/deep_13.md:3206:   474	approach preserves sample uncertainty and stops the student network
./validate/v_2.md:9676:reviews/deep_10.md:2385:value of weights 𝜆 in an adaptive manner. Specifically, we first leverage the Wasserstein distance to measure the distribution discrepancy
./validate/v_2.md:9705:reviews/deep_10.md:4648:adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
./validate/v_2.md:9771:reviews/deep_10.md:7465:33:adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
./validate/v_2.md:9781:reviews/deep_10.md:7573:1030:Our framework has two important components: DSH in early bottlenecks to bring about style variance, and the DED loss 𝐷𝐸𝐷 to enhance
./validate/v_2.md:9846:reviews/deep_7.md:4407:   206	the uncertainty, thereby improving the discriminative ability of the
./validate/v_2.md:9882:reviews/deep_24.md:4791:the specified query text to retrieve its corresponding person in the image library. Rank images by
./validate/v_2.md:9883:reviews/deep_24.md:4792:text–image similarity. Rank-k is the probability of finding at least one matching person image in
./validate/v_2.md:9901:reviews/deep_24.md:5392:714-the specified query text to retrieve its corresponding person in the image library. Rank images by
./validate/v_2.md:9902:reviews/deep_24.md:5393:715:text–image similarity. Rank-k is the probability of finding at least one matching person image in
./validate/v_2.md:10067:reviews/deep_2.md:4690:pose uncertainty, appearance changes, and scene complexity
./validate/v_2.md:10116:reviews/deep_2.md:9938:cues, and later stages stabilize rankings under broader identity consensus, which is particularly beneficial on large-scale or distributionshifted datasets.
./validate/v_2.md:10166:reviews/deep_4.md:801:each feature based on its uncertainty, highlighting the influence of more reliable features. As shown in Fig. 1(c), our
./validate/v_2.md:10170:reviews/deep_4.md:938:uncertainty
./validate/v_2.md:10171:reviews/deep_4.md:955:uncertainty
./validate/v_2.md:10173:reviews/deep_4.md:1064:to each feature based on prediction uncertainty, allowing the
./validate/v_2.md:10174:reviews/deep_4.md:1069:weight λm is determined according to the uncertainty of feature fm , which is quantified by the entropy of its prediction
./validate/v_2.md:10175:reviews/deep_4.md:1070:distribution. The uncertainty Em is computed as follows:
./validate/v_2.md:10176:reviews/deep_4.md:1104:individuals in the dataset. A lower Em reflects less uncertainty in the prediction, resulting in an increased weight for
./validate/v_2.md:10179:reviews/deep_4.md:1777:all these methods by adaptively assigning weights to each feature based on the entropy of its prediction distribution.
./validate/v_2.md:10193:reviews/deep_4.md:4038:alignment and uncertainty estimation for visible-infrared person re-identification,
./validate/v_2.md:10223:reviews/deep_4.md:7198:uncertainty replayer strategy, which focuses on storing samples with
./validate/v_2.md:10224:reviews/deep_4.md:7218:model adaptive framework that adapts to the data distribution of new
./validate/v_2.md:10227:reviews/deep_4.md:7401:(AKT) framework, which conducts the simulated distribution for facilitating the adaptive knowledge distillation. The framework of AKT can
./validate/v_2.md:10236:reviews/deep_4.md:8843:(AKT) framework, which conducts the simulated distribution for facilitating the adaptive knowledge distillation. The framework of AKT can
./validate/v_2.md:10256:reviews/deep_4.md:10818:to each feature based on prediction uncertainty, allowing the
./validate/v_2.md:10257:reviews/deep_4.md:10823:weight λm is determined according to the uncertainty of feature fm , which is quantified by the entropy of its prediction
./validate/v_2.md:10258:reviews/deep_4.md:10824:distribution. The uncertainty Em is computed as follows:
./validate/v_2.md:10259:reviews/deep_4.md:10858:individuals in the dataset. A lower Em reflects less uncertainty in the prediction, resulting in an increased weight for
./validate/v_2.md:10297:reviews/deep_3.md:3575:   700	align with the data distribution encountered during the model’s extensive pre-training. When combined with k-reciprocal re-ranking,
./validate/v_2.md:10316:reviews/deep_11.md:1423:   162	features in the target domain. Some uncertainty-based ReID methods [74, 76] have been proposed
./validate/v_2.md:10317:reviews/deep_11.md:1425:   164	of uncertainty in pseudo labels is challenging in the unsupervised scenario, thereby complicating
./validate/v_2.md:10349:reviews/deep_11.md:4895:   341	noise perturbation is used to enhance the style of each task, which is called style uncertainty
./validate/v_2.md:10368:reviews/deep_11.md:5648:   176	discriminative information in an adaptive feature space and overcoming outof-distribution samples, thereby improving identification performance.
./validate/v_2.md:10373:reviews/deep_11.md:8421:   278	Han, J., Li, Y.-L., & Wang, S. (2022a). Delving into probabilistic uncertainty for unsupervised domain adaptive person re-identiﬁcation. Proceedings of the AAAI Conference on
./validate/v_2.md:10387:reviews/deep_11.md:10684:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:10388:reviews/deep_11.md:10763:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./validate/v_2.md:10400:reviews/deep_14.md:873:546:uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:10405:reviews/deep_14.md:1289:164:model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:10415:reviews/deep_14.md:1443:175:a dual-path local alignment network to extract local visual and textual representations from horizontally segmented image patches. Subsequently, local representations are aligned adaptively with a multi-stage
./validate/v_2.md:10448:reviews/deep_14.md:3977:   164	model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:10449:reviews/deep_14.md:3978:   165	critical pedestrian features using an uncertainty-aware self-attention
./validate/v_2.md:10477:reviews/deep_14.md:5410:   546	uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:10482:reviews/deep_14.md:5764:164:   164	model, which uses un-occluded local features to recover missing features. Zheng et al. (2024) proposed a cascade Transformer reasoning embedded by uncertainty network (CTU) model that progressively extracts
./validate/v_2.md:10483:reviews/deep_14.md:5765:165:   165	critical pedestrian features using an uncertainty-aware self-attention
./validate/v_2.md:10577:reviews/deep_25.md:5735:More recent studies have concentrated on resolving modal differences at a deeper level. S. Li et al. (2024) proposed an uncertaintyaware progressive learning framework to progressively eliminate
./validate/v_2.md:10671:reviews/deep_25.md:14083:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./validate/v_2.md:10673:reviews/deep_25.md:14162:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./validate/v_2.md:10680:reviews/deep_16.md:1019:87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
./validate/v_2.md:10703:reviews/deep_16.md:4873:87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
./validate/v_2.md:10910:reviews/deep_6.md:1750:   168	label purification by adaptively adjusting the pseudo-label distribution.
./validate/v_2.md:10913:reviews/deep_6.md:1816:   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
./validate/v_2.md:10931:reviews/deep_6.md:2550:   759	matching, such as diverse style modeling, uncertainty-aware feature
./validate/v_2.md:10949:reviews/deep_6.md:3665:   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
./validate/v_2.md:11027:reviews/deep_20.md:4695:   462	measures the uncertainty or complexity of the tasks. The mutual information between the task and the injected noise is given by
./validate/v_2.md:11028:reviews/deep_20.md:4716:   483	task by reducing uncertainty rather than introducing disturbance.
./validate/v_2.md:11035:reviews/deep_20.md:5102:   130	of VI-ReID into two aspects: (i) difficulty in aligning lowand high-level semantics across modalities, and (ii) distribution discrepancies between VIS and IR features. To address
./validate/v_2.md:11078:reviews/deep_26.md:3476:As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
./validate/v_2.md:11121:reviews/deep_26.md:8488:As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
./validate/v_2.md:11159:reviews/deep_22.md:648:allowing the model to capture uncertainty in the representations. Second, Cross-Modal Containment (CMC) aligns the
./validate/v_2.md:11167:reviews/deep_22.md:713:while rejecting visually similar but semantically distinct negatives (Icn ). Second, even methods that introduce uncertainty
./validate/v_2.md:11174:reviews/deep_22.md:737:uncertainty but also models the semantic containment relationships between vision and language. As illustrated in Fig. 1(c),
./validate/v_2.md:11181:reviews/deep_22.md:745:captures modality-specific uncertainty. By learning compact
./validate/v_2.md:11192:reviews/deep_22.md:768:semantic scope, distinguishing it from prior uncertaintyaware methods that treat variance primarily as a noise
./validate/v_2.md:11193:reviews/deep_22.md:826:ambiguity, semantic granularity, and distributional uncertainty
./validate/v_2.md:11196:reviews/deep_22.md:830:for representing cross-modal semantics under uncertainty.
./validate/v_2.md:11198:reviews/deep_22.md:832:Probabilistic distribution representations have been increasingly used to model uncertainty and improve robustness
./validate/v_2.md:11200:reviews/deep_22.md:835:encode feature embeddings with uncertainty estimates reflecting representation quality. In domain generalization, DSU [37]
./validate/v_2.md:11201:reviews/deep_22.md:838:pose embeddings as Gaussians to handle input uncertainty
./validate/v_2.md:11203:reviews/deep_22.md:842:incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
./validate/v_2.md:11204:reviews/deep_22.md:843:uncertainty via batch- and identity-level variances.
./validate/v_2.md:11212:reviews/deep_22.md:861:way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
./validate/v_2.md:11214:reviews/deep_22.md:866:can also represent feature uncertainty. Student’s t [50] distribution has heavier tails, which can capture rare or extreme
./validate/v_2.md:11215:reviews/deep_22.md:869:[51] allows modeling multi-modal uncertainty, accommodating features with multiple plausible states; however, learning
./validate/v_2.md:11219:reviews/deep_22.md:880:learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
./validate/v_2.md:11220:reviews/deep_22.md:882:While prior work primarily focuses on uncertainty within
./validate/v_2.md:11231:reviews/deep_22.md:965:uncertainty when only the first two moments are known. Crucially, the mathematical closure and analytical tractability of
./validate/v_2.md:11244:reviews/deep_22.md:1018:texts with low uncertainty is primarily determined by the
./validate/v_2.md:11246:reviews/deep_22.md:1020:while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./validate/v_2.md:11248:reviews/deep_22.md:1047:uncertainty-aware approaches that treat variance merely as a
./validate/v_2.md:11253:reviews/deep_22.md:1059:where µvi and µt j denote the mean features, and σ2vi , σ2t j represent variances that capture feature uncertainty and semantic
./validate/v_2.md:11272:reviews/deep_22.md:1266:enhances robustness by ensuring masked distributions encompass unmasked ones, reflecting increased uncertainty in partial
./validate/v_2.md:11275:reviews/deep_22.md:1317:exhibit greater uncertainty, thereby naturally encompassing the
./validate/v_2.md:11278:reviews/deep_22.md:1336:introduces uncertainty by randomly masking textual tokens.
./validate/v_2.md:11285:reviews/deep_22.md:1447:CUHK PEDES dataset. PDA achieves a Rank-1 accuracy of
./validate/v_2.md:11286:reviews/deep_22.md:1451:(MLLMs) for diverse textual generation, PDA improves Rank1 by 1.25% and mAP by 0.01%. When compared with MUM,
./validate/v_2.md:11288:reviews/deep_22.md:1453:uncertainty in cross modal retrieval, PDA demonstrates a
./validate/v_2.md:11291:reviews/deep_22.md:1477:MUM, our model shows a 3.65% advantage in Rank-1 accuracy, indicating its superior ability to cope with uncertainty.
./validate/v_2.md:11295:reviews/deep_22.md:1487:terms of Rank-1 and mAP. Specifically, PDA surpasses NAM
./validate/v_2.md:11296:reviews/deep_22.md:1490:slightly lower in Rank-5 and Rank-10, PDA demonstrates
./validate/v_2.md:11299:reviews/deep_22.md:1495:a Rank-1 improvement of 7.3%, highlighting PDA’s enhanced
./validate/v_2.md:11303:reviews/deep_22.md:1506:based on Swin B and BERT backbones, PDA shows improvements of 1.54% in Rank-1 and 2.65% in mAP, all while
./validate/v_2.md:11333:reviews/deep_22.md:5367:CLIP and CoOP to address this limitation, the uncertainty and
./validate/v_2.md:11342:reviews/deep_22.md:7178:22:    22	allowing the model to capture uncertainty in the representations. Second, Cross-Modal Containment (CMC) aligns the
./validate/v_2.md:11351:reviews/deep_22.md:7458:   499	CLIP and CoOP to address this limitation, the uncertainty and
./validate/v_2.md:11355:reviews/deep_22.md:7858:   339	uncertainty when only the first two moments are known. Crucially, the mathematical closure and analytical tractability of
./validate/v_2.md:11368:reviews/deep_22.md:7911:   392	texts with low uncertainty is primarily determined by the
./validate/v_2.md:11370:reviews/deep_22.md:7913:   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./validate/v_2.md:11372:reviews/deep_22.md:7940:   421	uncertainty-aware approaches that treat variance merely as a
./validate/v_2.md:11377:reviews/deep_22.md:7952:   433	where µvi and µt j denote the mean features, and σ2vi , σ2t j represent variances that capture feature uncertainty and semantic
./validate/v_2.md:11396:reviews/deep_22.md:8159:   640	enhances robustness by ensuring masked distributions encompass unmasked ones, reflecting increased uncertainty in partial
./validate/v_2.md:11399:reviews/deep_22.md:8210:   691	exhibit greater uncertainty, thereby naturally encompassing the
./validate/v_2.md:11402:reviews/deep_22.md:8229:   710	introduces uncertainty by randomly masking textual tokens.
./validate/v_2.md:11414:reviews/deep_22.md:9247:/bin/zsh -lc "pdftotext 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | rg -n \"Ablation|TABLE|Distribution|Visualization|Parameter|containment|CMC|IMC|DRM|masked|uncertainty|Domain Generalization|Robustness\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./validate/v_2.md:11418:reviews/deep_22.md:9253:22:    22	allowing the model to capture uncertainty in the representations. Second, Cross-Modal Containment (CMC) aligns the
./validate/v_2.md:11420:reviews/deep_22.md:9257:87:    87	while rejecting visually similar but semantically distinct negatives (Icn ). Second, even methods that introduce uncertainty
./validate/v_2.md:11422:reviews/deep_22.md:9259:111:   111	uncertainty but also models the semantic containment relationships between vision and language. As illustrated in Fig. 1(c),
./validate/v_2.md:11425:reviews/deep_22.md:9263:119:   119	captures modality-specific uncertainty. By learning compact
./validate/v_2.md:11431:reviews/deep_22.md:9274:142:   142	semantic scope, distinguishing it from prior uncertaintyaware methods that treat variance primarily as a noise
./validate/v_2.md:11432:reviews/deep_22.md:9280:200:   200	ambiguity, semantic granularity, and distributional uncertainty
./validate/v_2.md:11433:reviews/deep_22.md:9281:204:   204	for representing cross-modal semantics under uncertainty.
./validate/v_2.md:11435:reviews/deep_22.md:9283:206:   206	Probabilistic distribution representations have been increasingly used to model uncertainty and improve robustness
./validate/v_2.md:11436:reviews/deep_22.md:9284:209:   209	encode feature embeddings with uncertainty estimates reflecting representation quality. In domain generalization, DSU [37]
./validate/v_2.md:11437:reviews/deep_22.md:9285:212:   212	pose embeddings as Gaussians to handle input uncertainty
./validate/v_2.md:11438:reviews/deep_22.md:9286:216:   216	incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
./validate/v_2.md:11439:reviews/deep_22.md:9287:217:   217	uncertainty via batch- and identity-level variances.
./validate/v_2.md:11441:reviews/deep_22.md:9290:235:   235	way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
./validate/v_2.md:11442:reviews/deep_22.md:9291:240:   240	can also represent feature uncertainty. Student’s t [50] distribution has heavier tails, which can capture rare or extreme
./validate/v_2.md:11443:reviews/deep_22.md:9292:243:   243	[51] allows modeling multi-modal uncertainty, accommodating features with multiple plausible states; however, learning
./validate/v_2.md:11445:reviews/deep_22.md:9294:254:   254	learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
./validate/v_2.md:11446:reviews/deep_22.md:9295:256:   256	While prior work primarily focuses on uncertainty within
./validate/v_2.md:11449:reviews/deep_22.md:9304:339:   339	uncertainty when only the first two moments are known. Crucially, the mathematical closure and analytical tractability of
./validate/v_2.md:11450:reviews/deep_22.md:9306:392:   392	texts with low uncertainty is primarily determined by the
./validate/v_2.md:11451:reviews/deep_22.md:9307:394:   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
./validate/v_2.md:11452:reviews/deep_22.md:9313:421:   421	uncertainty-aware approaches that treat variance merely as a
./validate/v_2.md:11454:reviews/deep_22.md:9315:433:   433	where µvi and µt j denote the mean features, and σ2vi , σ2t j represent variances that capture feature uncertainty and semantic
./validate/v_2.md:11458:reviews/deep_22.md:9324:640:   640	enhances robustness by ensuring masked distributions encompass unmasked ones, reflecting increased uncertainty in partial
./validate/v_2.md:11461:reviews/deep_22.md:9335:691:   691	exhibit greater uncertainty, thereby naturally encompassing the
./validate/v_2.md:11464:reviews/deep_22.md:9341:710:   710	introduces uncertainty by randomly masking textual tokens.
./validate/v_2.md:11467:reviews/deep_22.md:9356:827:   827	uncertainty in cross modal retrieval, PDA demonstrates a
./validate/v_2.md:11468:reviews/deep_22.md:9359:851:   851	MUM, our model shows a 3.65% advantage in Rank-1 accuracy, indicating its superior ability to cope with uncertainty.
./validate/v_2.md:11470:reviews/deep_22.md:9367:958:   958	uncertainty via variances. This probabilistic modeling helps
./validate/v_2.md:11475:reviews/deep_22.md:9385:1027:  1027	(DRM), we compare it with representative uncertainty-based
./validate/v_2.md:11477:reviews/deep_22.md:9388:1038:  1038	scopes. Compared with MUM, which estimates uncertainty via
./validate/v_2.md:11479:reviews/deep_22.md:9390:1045:  1045	uncertainty via the [CLS] token, effectively avoiding the noise
./validate/v_2.md:11480:reviews/deep_22.md:9391:1048:  1048	AUL employs adaptive loss weighting to handle uncertainty,
./validate/v_2.md:11485:reviews/deep_22.md:9408:1156:  1156	proves that modeling visual-textual uncertainty through probabilistic distributions is a more efficient and powerful paradigm
./validate/v_2.md:11487:reviews/deep_22.md:9410:1203:  1203	across queries, and robustness to modality-specific uncertainty ultimately enabling more comprehensive and reliable
./validate/v_2.md:11490:reviews/deep_22.md:9416:1235:  1235	distributions, capturing uncertainty and enabling probabilistic
./validate/v_2.md:11492:reviews/deep_22.md:9421:1296:  1296	[35] J. Chang, Z. Lan, C. Cheng, and Y. Wei, “Data uncertainty learning
./validate/v_2.md:11493:reviews/deep_22.md:9422:1300:  1300	re-identification by modelling feature uncertainty,” in Proc. IEEE/CVF
./validate/v_2.md:11494:reviews/deep_22.md:9423:1308:  1308	[40] Y. Ji et al., “MAP: Multimodal uncertainty-aware vision-language
./validate/v_2.md:11495:reviews/deep_22.md:9424:1312:  1312	uncertainty modeling and semantic alignment for text-to-image person
./validate/v_2.md:11496:reviews/deep_22.md:9425:1464:  1464	uncertainty-based learning for text-based person retrieval,” in Proc.
./validate/v_2.md:11507:reviews/deep_22.md:9693:   947	These results clearly indicate that PDA strikes an excellent balance between retrieval accuracy and computational overhead,
./validate/v_2.md:11510:reviews/deep_22.md:9704:   958	uncertainty via variances. This probabilistic modeling helps
./validate/v_2.md:11526:reviews/deep_22.md:9764:  1203	across queries, and robustness to modality-specific uncertainty ultimately enabling more comprehensive and reliable
./validate/v_2.md:11556:reviews/deep_22.md:11207:                                                                                distributions, capturing uncertainty and enabling probabilistic
./validate/v_2.md:11573:reviews/deep_22.md:11267:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:11578:reviews/deep_22.md:11344:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:11586:4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
./validate/v_2.md:11603:Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
./validate/v_2.md:11604:Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
./validate/v_2.md:11610:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
./validate/v_2.md:11645:Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
./validate/v_2.md:11677:Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
./validate/v_2.md:11784:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:11792:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:11794:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:11796:web search: "person re-identification" "aleatoric uncertainty"
./validate/v_2.md:11802:web search: "Robust person re-identification" "feature uncertainty"
./validate/v_2.md:11804:web search: "feature uncertainty" "ReID" "Gaussian" "posterior"
./validate/v_2.md:11841:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:11849:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:11851:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:11853:web search: "person re-identification" "aleatoric uncertainty"
./validate/v_2.md:11859:web search: "Robust person re-identification" "feature uncertainty"
./validate/v_2.md:11861:web search: "feature uncertainty" "ReID" "Gaussian" "posterior"
./validate/v_2.md:11865:web search: "uncertainty" "person re-identification" "low resolution"
./validate/v_2.md:11867:web search: "low-resolution" "uncertainty" "person re-identification" "Gaussian"
./validate/v_2.md:11910: exited 1 in 0ms:
./validate/v_2.md:11913: exited 1 in 0ms:
./validate/v_2.md:11916: exited 1 in 0ms:
./validate/v_2.md:11919: exited 1 in 0ms:
./validate/v_2.md:11931: exited 1 in 0ms:
./validate/v_2.md:12063:| 13 | 新协议贴部署 | severe modality imbalance; mix-modality; anytime |
./validate/v_2.md:12068:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
./validate/v_2.md:12170: exited 1 in 0ms:
./validate/v_2.md:12174: exited 2 in 0ms:
./validate/v_2.md:12314:- SeCap(adaptive prompt + local refine): 不做 prompt recalibration。
./validate/v_2.md:12349:1. **token 抽取(复用 maxsim_probe 配方)**: hook `model.layer4`(GeM 前 spatial map, 16×8)→ `adaptive_avg_pool2d` 到 grid(默认 8×4=32 token)→ flatten → **新增 1×1 conv proj 到 256d + 逐 token L2-norm**。
./validate/v_2.md:12357:   - OVLI 是训练期 loss; eval 默认 **global-only 不变**(精确 == baseline)。`--ovli_rerank` 可选额外报 `global + MaxSim` rerank(global 与 rerank 两个数都打印), 用与训练**同一套** proj token + 双向 MaxSim。
./validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./validate/v_2.md:12380:| **OVP+OVLI(full model)** | **52.14**(rerank 52.71, R1 57.74) | **+19.7** | OVLI 互补 prototype **+2.0 over OVP** |
./validate/v_2.md:12382:- 轨迹: 14.91(ep10)→24.24(ep20)→...→45.19(ep60)。rerank(global+MaxSim)≈global(45.17), 收敛后 global 已够, MaxSim rerank 早期 +3 后期收敛。
./validate/v_2.md:12394:- **必做消融**: global-oppview-SupCon vs +OVLI / oppview-only vs all-view / test global vs +MaxSim-rerank / MaxSim vs OT/top-k/avg / vs AlignedReID / α-τ-grid sweep / token-match 可视化(航拍头肩→地面上身/背包)/ AG-ReID.v2 跨数据集。
./validate/v_2.md:12398:- 已写 `OVLIHead`(token proj + 双向 MaxSim + opp-view supcon loss)+ `ovli_rerank_eval`(eval 期 global vs global+MaxSim 双报)。
./validate/v_2.md:12445:validate/v_1.md:30:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:12448:validate/v_1.md:38:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:12449:validate/v_1.md:40:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:12450:validate/v_1.md:56:web search: "low-resolution" "uncertainty" "person re-identification" "Gaussian"
./validate/v_2.md:12474:validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./validate/v_2.md:12481:validate/v_3.md:255:- **必做消融**: global-oppview-SupCon vs +OVLI / oppview-only vs all-view / test global vs +MaxSim-rerank / MaxSim vs OT/top-k/avg / vs AlignedReID / α-τ-grid sweep / token-match 可视化(航拍头肩→地面上身/背包)/ AG-ReID.v2 跨数据集。
./validate/v_2.md:12561:validate/v_3.md:1189:./validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./validate/v_2.md:12567:validate/v_3.md:1203:./validate/v_3.md:255:- **必做消融**: global-oppview-SupCon vs +OVLI / oppview-only vs all-view / test global vs +MaxSim-rerank / MaxSim vs OT/top-k/avg / vs AlignedReID / α-τ-grid sweep / token-match 可视化(航拍头肩→地面上身/背包)/ AG-ReID.v2 跨数据集。
./validate/v_2.md:12644:validate/v_3.md:1812:../claude_review_ovli.md:52:- **事实**: OVLI 相对 OVP 的撞车面**显著缩小**——去掉 prototype/memory/EMA 后，CMPC/MBCE/PDPA 的「per-id per-modality 原型 + momentum」结构不再适用。design 的三条切开(vs OVP/CMPC「无 prototype」、vs ColBERT/MaxSim-rerank「是训练 loss 非 test rerank」、vs GSAlign「无 TPS warp/visibility」)**框架上站得住**。
./validate/v_2.md:12646:validate/v_3.md:1820:../claude_review_ovli.md:123:**审查结论：审查通过。** 无 Critical、无 High。OVLI 载荷设计(proj 进 optimizer 且自检、fp16-map→fp32-proj→encoder 梯度回流、共享 bn 混精单 backward、双向对称 MaxSim + opp-view 多正 logsumexp SupCon、AMP fp32 隔离、floor NaN 安全、--ovli off 精确复现 baseline、ovp/ovli 互斥、eval 默认 global-only + rerank opt-in、边界不崩)逐行 + 5 处本地数值核实**全部正确**。M3(novelty 联网坐实)进稿前处理，不阻断 empirical 训练。可启动 kill-switch #2′ → 交 Codex 第二轮独立审查。
./validate/v_2.md:12649:validate/v_3.md:1875:./reviews/deep_25.md:14041:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:12651:validate/v_3.md:1894:./reviews/deep_25.md:14120:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:12655:validate/v_3.md:1915:../codex_review_dualbranch.md:27:1. **High framing**: `afd_train.py` --airl_dualbranch 注释块(~1474)+ `new_angle_AIRL.md`(~160)framing 改为 **"observation-limited evidence ceiling 下 clean(f_full)/recover(f_rec)evidence head 分化 + 固定先验软融合(fixed-prior fusion)"**,明确写 **不是 query-budget routing / 动态 router**(kill-switch #3 已证硬路由失败 ≤+0.41,增益全来自固定 w 软混合),避 RAR/MRJL/cross-resolution 撞车。"dual-branch routes it to f_rec" 措辞改 "applies it to f_rec"。
./validate/v_2.md:12659:validate/v_3.md:1922:../codex_novelty_airl.md:10:| ④ **检索方向特化双 head** | **★5/5** | query-adaptive late fusion(按 query 估特征有效性);AG-ReID.v2/VDT 分 A→G/G→A 报告但同 embedding/同融合 | **"clean head 强 A→G、robust head 强 G→A + 方向感知融合"无明确先例——最值得当主贡献** |
./validate/v_2.md:12660:validate/v_3.md:1927:../codex_novelty_airl.md:22:- 撞车清单(写 related-work 切开):AG-ReID.v2 / VDT / ViSA / RAIN / DI-REID / MRJL / query-adaptive late fusion。
./validate/v_2.md:12661:validate/v_3.md:1928:../codex_review_ovli.txt:14:Review experiments/cargo_cvpb/afd_train.py, focus on the NEW --ovli OVLIHead (Opposite-View Late-Interaction Evidence Alignment): token proj (Conv2d 2048->256 on a layer4 feature-map hook, adaptive-pool to 8x4 grid, L2-norm per token), opposite-view supervised-contrastive loss score=alpha*cos(global)+(1-alpha)*sym_MaxSim(bidirectional mean-max), logsumexp multi-positive, tau0.05 alpha0.5, lambda warmup --ovli_warmup. 逐行查: bug/runtime/shape/device/AMP数值安全; ★proj新参数是否正确加进optimizer(list(model.parameters())+list(ovli.parameters())+assert自检, 这是关键); MaxSim/logsumexp数值(无NaN, empty-candidate用-1e4 floor); train/test对称(eval默认global-only, --ovli_rerank可选); --ovli off精确复现baseline; --ovp/--ovli互斥。用web search查novelty撞车: opposite-view/cross-view token-set late-interaction(ColBERT/MaxSim)作为training-time cross-view retrieval loss 在 aerial-ground 或 person ReID 的先例(对比CM-EMD/G2DA VI-ReID OT, CVFT geo-localization, DTST token selection, CMPC prototype)。输出 Verdict(approve/needs-attention)+findings(severity Critical/High/Medium/Low)。
./validate/v_2.md:12707:validate/v_3.md:2595:../codex_review_ovli.txt:3794:experiments/cargo_cvpb/design.md:83:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./validate/v_2.md:12729:validate/v_3.md:2764:../codex_review_ovli.txt:3969:experiments/overnight_innovation_log.md:187:- **战略转向**：FM 方向从"唯一活口/大概率 me-too"升级为"有真实正结果"。但需诚实评估**新颖性**：DINOv2+LoRA+ReID 是否 me-too（PersonViT / DINO-as-backbone territory）？competitive 到什么程度（36.72 heavy vs Swin 72.57——仍有大 gap，但这是 pose-part-MaxSim 单分支、无 PSG/GCN/全套）？下一步：等 e30 看上限 + 对照 rank16/large 看 capacity 曲线，再判断是"competitive-FM-adaptation 方法"还是"诊断性结论的正向补充"。
./validate/v_2.md:12732:validate/v_3.md:2771:../codex_review_ovli.txt:3976:experiments/overnight_innovation_log.md:306:- **双对照结论：backbone 容量(base→large)、adaptation 容量(rank16→32)都不是瓶颈** → 瓶颈在**机制/问题结构**（pose-part-MaxSim 5 部位表征上限）。
./validate/v_2.md:12741:validate/v_3.md:2883:../litreview2/analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:12743:validate/v_3.md:2933:../litreview2/analyses/ALL_ANALYSES.txt:1128:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:12790:validate/v_3.md:3452:../litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./validate/v_2.md:12794:validate/v_3.md:3465:../litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./validate/v_2.md:12850:validate/v_3.md:3961:../airl_codex_bundle/reviews/codex_9.md:14752:公开检索补完了：AG-ReID 近作已经有 view-specific / dual-branch local fusion、多流、多尺度，以及 cross-resolution ReID 里已有 recover/identify、feature distillation、resolution-adaptive 表征。AIRL 的“固定先验融合 + clean/recover head 分工”可以作为工程机制讲，但 novelty 不能写成“首个 degradation consistency / recovery branch / multi-branch”。
./validate/v_2.md:12913:validate/v_3.md:4392:../litreview/reviews/lit_19.md:2690:cheap kill-switch：在 CARGO 或 AG-ReID.v2 上取 SOLIDER top-k 错误但高度相似的负样本，人工快速看一小批，判断是否存在真实不可分或标注不确定。再做一个极简 soft negative 训练或只在 triplet 里屏蔽疑似歧义负样本。若 rank1 或 mAP 有明显提升，说明这个设定值得扩成方法稿。
./validate/v_2.md:12916:validate/v_3.md:4401:../litreview/reviews/lit_19.md:2766:cheap kill-switch：在 CARGO 或 AG-ReID.v2 上取 SOLIDER top-k 错误但高度相似的负样本，人工快速看一小批，判断是否存在真实不可分或标注不确定。再做一个极简 soft negative 训练或只在 triplet 里屏蔽疑似歧义负样本。若 rank1 或 mAP 有明显提升，说明这个设定值得扩成方法稿。
./validate/v_2.md:12963:validate/v_2.md:457:/bin/zsh -lc 'rg -n "非对称|包含|probabil|uncertain|uncertainty|OT|optimal|PDA|MaxSim|ColBERT|Aligned|distribution|Gaussian|KL|variance|方差|因果|causal|align|对齐" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./validate/v_2.md:12976:validate/v_2.md:801:./reviews/deep_19.md:3519:Comparison of Rank-k and mAP Performance with State-Of-The-Art (SOTA) Methods in LTCC and PRCC. The best and second-best results are highlighted in bold and underline.
./validate/v_2.md:12977:validate/v_2.md:802:./reviews/deep_19.md:3740:Comparison of Rank-k and mAP Performance with State-Of-The-Art (SOTA) Methods in VC-Clothes. The best and second-best results are
./validate/v_2.md:12989:validate/v_2.md:838:./reviews/deep_19.md:6583:our method achieves 97.8% Rank-1 accuracy, which is significantly better than other SOTA methods.
./validate/v_2.md:13004:validate/v_2.md:957:./reviews/deep_28.md:1619:   293	Table 1: Comparisons with SOTA methods on SYSU-MM01 and RegDB datasets. We report Rank at 𝑟 accuracy (%) and mAP (%).
./validate/v_2.md:13052:validate/v_2.md:1608:./reviews/deep_23.md:8406:the SOTA method IANet [18] by 4.2% Rank-1 and 5.1% mAP
./validate/v_2.md:13132:validate/v_2.md:2197:./reviews/deep_9.md:5344:   744	expanded. For instance, the SOTA performance on Singleshot Rank-1 and Multi-shot Rank-1 is improved by +9.99%
./validate/v_2.md:13137:validate/v_2.md:2222:./reviews/deep_9.md:6902:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./validate/v_2.md:13138:validate/v_2.md:2230:./reviews/deep_9.md:6949:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./validate/v_2.md:13165:validate/v_2.md:2471:./reviews/deep_6.md:1656:  2097	Table 2. Note that none of the methods in this experiment use reranking. Our proposed MCST outperforms existing SOTA methods,
./validate/v_2.md:13166:validate/v_2.md:2472:./reviews/deep_6.md:1658:  2103	improving the average recognition rate. As shown in Table 3, MCST outperforms SOTA methods on the SYSU-MM01 dataset. With re-ranking,
./validate/v_2.md:13199:validate/v_2.md:2614:./reviews/deep_6.md:6141:  1943	cascade dependency between CPC and OTPM, and the coarse-to-fine
./validate/v_2.md:13213:validate/v_2.md:2660:./reviews/deep_14.md:873:546:uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:13221:validate/v_2.md:2784:./reviews/deep_14.md:5410:   546	uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:13327:validate/v_2.md:3830:./reviews/deep_2.md:13237:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./validate/v_2.md:13331:validate/v_2.md:3845:./reviews/deep_2.md:13318:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./validate/v_2.md:13340:validate/v_2.md:3867:./validate/v_1.md:30:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:13343:validate/v_2.md:3871:./validate/v_1.md:38:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:13344:validate/v_2.md:3872:./validate/v_1.md:40:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:13379:validate/v_2.md:4030:./reviews/K-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13382:validate/v_2.md:4045:./reviews/deep_22.md:11344:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13407:validate/v_2.md:4280:./analyses/deep_2.txt:11:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./validate/v_2.md:13455:validate/v_2.md:4734:./analyses/deep_9.txt:7:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./validate/v_2.md:13488:validate/v_2.md:5098:./reviews/deep_25.md:14041:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:13491:validate/v_2.md:5101:./reviews/deep_25.md:14053:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./validate/v_2.md:13493:validate/v_2.md:5110:./reviews/deep_25.md:14120:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:13496:validate/v_2.md:5113:./reviews/deep_25.md:14132:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./validate/v_2.md:13500:validate/v_2.md:5133:./analyses/deep_22.txt:11:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13503:validate/v_2.md:5162:./analyses/deep_25.txt:11:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:13506:validate/v_2.md:5165:./analyses/deep_25.txt:23:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./validate/v_2.md:13564:validate/v_2.md:5864:./analyses/ALL_ANALYSES.txt:705:4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。
./validate/v_2.md:13572:validate/v_2.md:5893:./analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13574:validate/v_2.md:5921:./analyses/ALL_ANALYSES.txt:1128:4. **证据闭环**：关键不是只报 SOTA，而是组件拆开证明重定义。Baseline 在 Event-ReId 上是 36.4 rank-1、21.1 mAP，MSAM 单独到 68.2、42.0，CPDM 单独到 64.8、39.6，完整 SFE-Net 到 75.0、45.0。MSAM 还做了位置、Pre-SA、SIM、Post-SA 的消融，并用特征图可视化说明 SA 让离散语义更连续，SIM 增强上下文。CPDM 和 Dropout、DropBlock、PCB 对比，CPDM 明显更强，支撑“普通 dropout 不适合稀疏事件语义”。
./validate/v_2.md:13577:validate/v_2.md:5924:./analyses/ALL_ANALYSES.txt:1140:4. **证据闭环**：最关键证据是人为构造不同失衡比例。SYSU-MM01 从 2:1 到 4:1、8:1，很多已有方法 rank-1 和 mAP 大幅下滑，而 CMBL 下滑明显小。组件消融也很有说服力：2:1 下 baseline 是 63.79 rank-1、59.87 mAP，完整 CMBL 到 76.36、71.64；8:1 下 baseline 是 40.20、34.39，完整 CMBL 到 70.62、65.72。SSP 插入层位置、MMD 三项、CACS 的 intra/inter 都分别消融。可视化上还对比 SMOTE 和 SSP 的生成分布，并展示分类 logit 更稳定。
./validate/v_2.md:13586:validate/v_2.md:6010:./analyses/ALL_ANALYSES.txt:1836:4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
./validate/v_2.md:13636:validate/v_2.md:6551:./reviews/deep_1.md:4775:841-Rank-1 accuracy among existing SOTA methods. Specifically,
./validate/v_2.md:13643:validate/v_2.md:6579:./reviews/deep_1.md:5656:cloud followed by [68]. For the Spatial-Temporal Aggregation                  Rank-1 accuracy among existing SOTA methods. Specifically,
./validate/v_2.md:13644:validate/v_2.md:6580:./reviews/deep_1.md:5677:competitive performance compare to other SOTA methods.                        Rank-1 accuracy. It demonstrates that as a new and discrim-
./validate/v_2.md:13678:validate/v_2.md:6775:./reviews/deep_11.md:8421:   278	Han, J., Li, Y.-L., & Wang, S. (2022a). Delving into probabilistic uncertainty for unsupervised domain adaptive person re-identiﬁcation. Proceedings of the AAAI Conference on
./validate/v_2.md:13686:validate/v_2.md:6829:/bin/zsh -lc "rg -n \"PDA|probabilistic|Probability|Probabil|distribution|variance|uncertainty|text.*image|文本分布|方差=|⊇|⊆|Optimal Transport|OT-ReID|Wasserstein|EMD|CM-EMD|G2DA\" all_papers.txt analyses reviews validate -g '*.txt' -g '*.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./validate/v_2.md:13705:validate/v_2.md:7019:validate/v_3.md:1928:../codex_review_ovli.txt:14:Review experiments/cargo_cvpb/afd_train.py, focus on the NEW --ovli OVLIHead (Opposite-View Late-Interaction Evidence Alignment): token proj (Conv2d 2048->256 on a layer4 feature-map hook, adaptive-pool to 8x4 grid, L2-norm per token), opposite-view supervised-contrastive loss score=alpha*cos(global)+(1-alpha)*sym_MaxSim(bidirectional mean-max), logsumexp multi-positive, tau0.05 alpha0.5, lambda warmup --ovli_warmup. 逐行查: bug/runtime/shape/device/AMP数值安全; ★proj新参数是否正确加进optimizer(list(model.parameters())+list(ovli.parameters())+assert自检, 这是关键); MaxSim/logsumexp数值(无NaN, empty-candidate用-1e4 floor); train/test对称(eval默认global-only, --ovli_rerank可选); --ovli off精确复现baseline; --ovp/--ovli互斥。用web search查novelty撞车: opposite-view/cross-view token-set late-interaction(ColBERT/MaxSim)作为training-time cross-view retrieval loss 在 aerial-ground 或 person ReID 的先例(对比CM-EMD/G2DA VI-ReID OT, CVFT geo-localization, DTST token selection, CMPC prototype)。输出 Verdict(approve/needs-attention)+findings(severity Critical/High/Medium/Low)。
./validate/v_2.md:13707:validate/v_2.md:7024:validate/v_3.md:2883:../litreview2/analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13714:validate/v_2.md:7182:analyses/deep_22.txt:11:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13719:validate/v_2.md:7271:analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13726:validate/v_2.md:7286:validate/v_1.md:30:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:13729:validate/v_2.md:7290:validate/v_1.md:38:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:13730:validate/v_2.md:7291:validate/v_1.md:40:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:13737:validate/v_2.md:7307:validate/v_2.md:457:/bin/zsh -lc 'rg -n "非对称|包含|probabil|uncertain|uncertainty|OT|optimal|PDA|MaxSim|ColBERT|Aligned|distribution|Gaussian|KL|variance|方差|因果|causal|align|对齐" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
./validate/v_2.md:13760:validate/v_2.md:7938:validate/v_2.md:2660:./reviews/deep_14.md:873:546:uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:13761:validate/v_2.md:7977:validate/v_2.md:2784:./reviews/deep_14.md:5410:   546	uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:13781:validate/v_2.md:8233:validate/v_2.md:3867:./validate/v_1.md:30:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:13784:validate/v_2.md:8237:validate/v_2.md:3871:./validate/v_1.md:38:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:13785:validate/v_2.md:8238:validate/v_2.md:3872:./validate/v_1.md:40:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:13793:validate/v_2.md:8269:validate/v_2.md:4030:./reviews/K-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13796:validate/v_2.md:8273:validate/v_2.md:4045:./reviews/deep_22.md:11344:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13806:validate/v_2.md:8614:validate/v_2.md:5133:./analyses/deep_22.txt:11:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13813:validate/v_2.md:8775:validate/v_2.md:5893:./analyses/ALL_ANALYSES.txt:944:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:13821:validate/v_2.md:9017:validate/v_2.md:6775:./reviews/deep_11.md:8421:   278	Han, J., Li, Y.-L., & Wang, S. (2022a). Delving into probabilistic uncertainty for unsupervised domain adaptive person re-identiﬁcation. Proceedings of the AAAI Conference on
./validate/v_2.md:13855:validate/v_2.md:10373:reviews/deep_11.md:8421:   278	Han, J., Li, Y.-L., & Wang, S. (2022a). Delving into probabilistic uncertainty for unsupervised domain adaptive person re-identiﬁcation. Proceedings of the AAAI Conference on
./validate/v_2.md:13857:validate/v_2.md:10400:reviews/deep_14.md:873:546:uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:13858:validate/v_2.md:10477:reviews/deep_14.md:5410:   546	uncertainty-based transformers (SOT [7], UMSOT [8]). A comparison with state-of-the-art methods is summarized in Table 1.
./validate/v_2.md:13920:validate/v_2.md:11285:reviews/deep_22.md:1447:CUHK PEDES dataset. PDA achieves a Rank-1 accuracy of
./validate/v_2.md:13921:validate/v_2.md:11286:reviews/deep_22.md:1451:(MLLMs) for diverse textual generation, PDA improves Rank1 by 1.25% and mAP by 0.01%. When compared with MUM,
./validate/v_2.md:13923:validate/v_2.md:11288:reviews/deep_22.md:1453:uncertainty in cross modal retrieval, PDA demonstrates a
./validate/v_2.md:13928:validate/v_2.md:11295:reviews/deep_22.md:1487:terms of Rank-1 and mAP. Specifically, PDA surpasses NAM
./validate/v_2.md:13929:validate/v_2.md:11296:reviews/deep_22.md:1490:slightly lower in Rank-5 and Rank-10, PDA demonstrates
./validate/v_2.md:13932:validate/v_2.md:11299:reviews/deep_22.md:1495:a Rank-1 improvement of 7.3%, highlighting PDA’s enhanced
./validate/v_2.md:13936:validate/v_2.md:11303:reviews/deep_22.md:1506:based on Swin B and BERT backbones, PDA shows improvements of 1.54% in Rank-1 and 2.65% in mAP, all while
./validate/v_2.md:13957:validate/v_2.md:11467:reviews/deep_22.md:9356:827:   827	uncertainty in cross modal retrieval, PDA demonstrates a
./validate/v_2.md:13959:validate/v_2.md:11470:reviews/deep_22.md:9367:958:   958	uncertainty via variances. This probabilistic modeling helps
./validate/v_2.md:13963:validate/v_2.md:11485:reviews/deep_22.md:9408:1156:  1156	proves that modeling visual-textual uncertainty through probabilistic distributions is a more efficient and powerful paradigm
./validate/v_2.md:13966:validate/v_2.md:11490:reviews/deep_22.md:9416:1235:  1235	distributions, capturing uncertainty and enabling probabilistic
./validate/v_2.md:13976:validate/v_2.md:11507:reviews/deep_22.md:9693:   947	These results clearly indicate that PDA strikes an excellent balance between retrieval accuracy and computational overhead,
./validate/v_2.md:13978:validate/v_2.md:11510:reviews/deep_22.md:9704:   958	uncertainty via variances. This probabilistic modeling helps
./validate/v_2.md:14001:validate/v_2.md:11556:reviews/deep_22.md:11207:                                                                                distributions, capturing uncertainty and enabling probabilistic
./validate/v_2.md:14011:validate/v_2.md:11573:reviews/deep_22.md:11267:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:14014:validate/v_2.md:11578:reviews/deep_22.md:11344:4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:14021:validate/v_2.md:11784:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:14024:validate/v_2.md:11792:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:14025:validate/v_2.md:11794:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:14031:validate/v_2.md:11841:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:14034:validate/v_2.md:11849:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:14035:validate/v_2.md:11851:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:14036:validate/v_2.md:11867:web search: "low-resolution" "uncertainty" "person re-identification" "Gaussian"
./validate/v_2.md:14077:validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./validate/v_2.md:14084:validate/v_2.md:12394:- **必做消融**: global-oppview-SupCon vs +OVLI / oppview-only vs all-view / test global vs +MaxSim-rerank / MaxSim vs OT/top-k/avg / vs AlignedReID / α-τ-grid sweep / token-match 可视化(航拍头肩→地面上身/背包)/ AG-ReID.v2 跨数据集。
./validate/v_2.md:14123:web search: "probabilistic embeddings" "image-text" retrieval uncertainty asymmetric
./validate/v_2.md:14131:web search: person re-identification probabilistic embedding Gaussian uncertainty
./validate/v_2.md:14133:web search: "uncertainty" "person re-identification" "probabilistic" "embedding" "variance"
./validate/v_2.md:14135:web search: "person re-identification" "aleatoric uncertainty"
./validate/v_2.md:14141:web search: "Robust person re-identification" "feature uncertainty"
./validate/v_2.md:14143:web search: "feature uncertainty" "ReID" "Gaussian" "posterior"
./validate/v_2.md:14147:web search: "uncertainty" "person re-identification" "low resolution"
./validate/v_2.md:14149:web search: "low-resolution" "uncertainty" "person re-identification" "Gaussian"
./validate/v_2.md:14199:web search: "uncertainty-based learning" "text-based person retrieval"
./validate/v_2.md:14217:web search: cross-resolution person re-identification low resolution uncertainty Gaussian distribution
./validate/v_2.md:14219:web search: "Hedged Instance Embedding" "uncertainty" "image retrieval"
./validate/v_2.md:14221:相邻工作已经基本定位：PDA 是最近的“概率包含”先例；PCME/uncertainty ReID 是概率嵌入先例；CM-EMD/G2DA 是 OT 对齐先例；VDT/GSAlign/SeCap 是 AGPReID 红海。现在我收敛成辩护版本，不把 B 包装成“没人做过高斯”，而是钉住“为什么这个包含方向在航拍-地面里不是任意设计”。
./validate/v_2.md:14258:    30	| 13 | 新协议贴部署 | severe modality imbalance; mix-modality; anytime |
./validate/v_2.md:14263:    35	| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
./validate/v_2.md:14328:   420	- ⚠️ **教训(2026-06-23)**: 同步代码到 hyy/lab-4090 时漏了 `maxsim_probe.py`(rerank eval `from maxsim_probe import eval_from_distmat` L626)→ 3 个实验 ep10 首次 rerank 崩 `ModuleNotFoundError`。已补同步 + 重启。**另一坑**: `ssh host "cd X && a & b &"` 的 cd 只作用第一个 `&` 前 → b 在 home 跑找不到脚本; pkill+launch 同 shell 会自杀。多卡分开启动每条带 cd。
./validate/v_2.md:14345: 11267	4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:14365: 11287	2. **重定义动作**：它把“灾难性遗忘”重写成“旧身份原型、新任务特征、更新后特征提取器之间的兼容问题”。关键词是 non-exemplar、prototype-guided knowledge propagation、representation bias、adaptive parameter evolution、compatibility。旧问题是保留旧知识，新说法是让旧知识在新表征空间里继续能区分。
./validate/v_2.md:14369: 11291	4. **证据闭环**：组件消融很对题。baseline 的 Seen-Avg 是 42.8 mAP/57.4 Rank-1，加入 PKP 到 53.4/66.0，加入 APE 到 54.5/67.8，两者一起到 55.4/68.7；Unseen-Avg 也从 50.7/45.0 到 62.3/55.3。t-SNE 显示 PKA 后类内更紧、类间更分开，证明它不是只调分类器，而是在改善检索表征空间。
./validate/v_2.md:14393: 11315	4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。
./validate/v_2.md:14405: 11327	4. **证据闭环**：IMRM 相比全局和局部基线都有提升，F&B-TCRMG 好于只用正序或倒序，CMRM 加上 IMRM 最好。训练证据也对题：Hc-Tri 在约第 20 次迭代就收敛到 0，Acc 和 C2FL 能继续提供监督，C2FL 达到 69.16 Rank-1、86.23 Rank-5、64.57 mAP。热力图和检索例子显示，在外观细节高度相似时，模型能利用“包和人的关系”区分身份。
./validate/v_2.md:14422: 11344	4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。
./validate/v_2.md:14451:> Cross-view ReID is not symmetric alignment, but **physically directed uncertainty containment**: aerial observation is an underdetermined projection whose identity-compatible ground appearances form a broader distribution; correct ground evidence should fall inside that aerial uncertainty envelope.
./validate/v_2.md:14462:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./validate/v_2.md:14505:> Cross-view ReID is not symmetric alignment, but **physically directed uncertainty containment**: aerial observation is an underdetermined projection whose identity-compatible ground appearances form a broader distribution; correct ground evidence should fall inside that aerial uncertainty envelope.
./validate/v_2.md:14516:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_2.md:39:web search: person re-identification uncertainty learning feature quality 2024
./reassess/r_2.md:53:web search: k-reciprocal encoding person re-identification 2017 re-ranking
./reassess/r_2.md:76: exited 1 in 0ms:
./reassess/r_2.md:81: exited 1 in 0ms:
./reassess/r_2.md:86: exited 1 in 0ms:
./reassess/r_2.md:91: exited 1 in 0ms:
./reassess/r_2.md:103: exited 1 in 0ms:
./reassess/r_2.md:164:| 13 | 新协议贴部署 | severe modality imbalance; mix-modality; anytime |
./reassess/r_2.md:169:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
./reassess/r_2.md:439:reassess/r_1.md:959:用联网**尽力查确切先例+撞车**, 逐类判它们是否已做'用几何共同可见表面定same-ID正边可信度并据此重排训练关系': (a)hard-positive mining / sample re-weighting / positive reliability ReID; (b)noisy-correspondence / noisy-label ReID(positive不可靠); (c)curriculum/self-paced ReID(by difficulty); (d)pose/visibility/co-visibility-aware ReID; (e)SMPL/3D-geometry-guided ReID; (f)graph-sampler / relation ReID; (g)metric learning 里'positive不等价'(supcon变体/soft-positive)。撞了说清撞哪+杀伤; 没撞说清切开点。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./reassess/r_2.md:463:reassess/r_1.md:1084:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_2.md:479:reassess/r_1.md:1127:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_2.md:482:reassess/r_1.md:1142:用联网**尽力查确切先例+撞车**, 逐类判它们是否已做'用几何共同可见表面定same-ID正边可信度并据此重排训练关系': (a)hard-positive mining / sample re-weighting / positive reliability ReID; (b)noisy-correspondence / noisy-label ReID(positive不可靠); (c)curriculum/self-paced ReID(by difficulty); (d)pose/visibility/co-visibility-aware ReID; (e)SMPL/3D-geometry-guided ReID; (f)graph-sampler / relation ReID; (g)metric learning 里'positive不等价'(supcon变体/soft-positive)。撞了说清撞哪+杀伤; 没撞说清切开点。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./reassess/r_2.md:506:reassess/r_1.md:1267:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_2.md:522:reassess/r_1.md:1310:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_2.md:626:pivot/clean/novelty_gopl.txt:38:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_2.md:629:pivot/novelty_gopl.md:16:用联网**尽力查确切先例+撞车**, 逐类判它们是否已做'用几何共同可见表面定same-ID正边可信度并据此重排训练关系': (a)hard-positive mining / sample re-weighting / positive reliability ReID; (b)noisy-correspondence / noisy-label ReID(positive不可靠); (c)curriculum/self-paced ReID(by difficulty); (d)pose/visibility/co-visibility-aware ReID; (e)SMPL/3D-geometry-guided ReID; (f)graph-sampler / relation ReID; (g)metric learning 里'positive不等价'(supcon变体/soft-positive)。撞了说清撞哪+杀伤; 没撞说清切开点。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
./reassess/r_2.md:653:pivot/novelty_gopl.md:141:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_2.md:669:pivot/novelty_gopl.md:184:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_2.md:782:analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_2.md:811:analyses/deep_18.txt:8:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
./reassess/r_2.md:816:analyses/deep_18.txt:16:4. **证据闭环**：最关键的消融是和 KNN、相似度阈值、MPLP 比。Market 上相似度阈值是 72.8 Rank-1、39.9 mAP，MPRD 到 83.0、51.1。去掉大量无标签数据训练 GCN 的版本只有 75.8、43.1，说明不是 metadata 本身撑起来的。t-SNE 里加 GCN 后同身份更聚、易混身份分开，直接对应“邻域关系比点对相似度更可靠”。
./reassess/r_2.md:892:analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:905:analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_2.md:915:analyses/ALL_ANALYSES.txt:570:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
./reassess/r_2.md:920:analyses/ALL_ANALYSES.txt:578:4. **证据闭环**：最关键的消融是和 KNN、相似度阈值、MPLP 比。Market 上相似度阈值是 72.8 Rank-1、39.9 mAP，MPRD 到 83.0、51.1。去掉大量无标签数据训练 GCN 的版本只有 75.8、43.1，说明不是 metadata 本身撑起来的。t-SNE 里加 GCN 后同身份更聚、易混身份分开，直接对应“邻域关系比点对相似度更可靠”。
./reassess/r_2.md:927:analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1165:reassess/r_3.md:2389:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/novelty_gopl.txt:38:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_2.md:1263:reassess/r_3.md:2822:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:654:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1287:reassess/r_3.md:2885:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1039:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1306:reassess/r_3.md:2918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1226:./reviews/deep_25.md:1365:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_2.md:1311:reassess/r_3.md:2923:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1259:./reviews/deep_25.md:9584:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_2.md:1332:reassess/r_3.md:2944:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1293:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1336:reassess/r_3.md:2949:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1298:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1371:reassess/r_3.md:3019:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2180:./validate/v_2.md:5106:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1374:reassess/r_3.md:3022:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2183:./validate/v_2.md:5118:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1377:reassess/r_3.md:3026:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2193:./validate/v_2.md:5170:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1381:reassess/r_3.md:3035:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2220:./validate/v_2.md:5929:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:1570:reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_2.md:1580:reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_2.md:1605:reassess/r_3.md:3499:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3262:廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。
./reassess/r_2.md:1615:reassess/r_3.md:3513:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3337:廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。
./reassess/r_2.md:1718:reassess/r_3.md:3794:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2855:最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。
./reassess/r_2.md:1723:reassess/r_3.md:3800:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2928:最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。
./reassess/r_2.md:1756:reassess/r_3.md:3883:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:2891:of non-overlapping distributed cameras. In the early stages
./reassess/r_2.md:1787:reassess/r_3.md:3945:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3639:便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。
./reassess/r_2.md:1798:reassess/r_3.md:3957:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3693:便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。
./reassess/r_2.md:1876:reassess/r_3.md:4271:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_4.md:21478:../airl_related_work.md:26:RAR 用 query resolution **动态路由**选子空间。AIRL **不路由**——#3 oracle 实测硬路由(area/reliability 阈值)失败(≤+0.41),**软固定先验融合**(w=0.25,clean+recover 两证据头)反而 +1.46。所以 AIRL 的 claim 必须是 "fixed-prior fusion of clean/recover evidence heads under an observation-limited ceiling",**绝不吹成 query-budget routing**(那既撞 RAR 又与实现不符)。
./reassess/r_2.md:1917:reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
./reassess/r_2.md:1922:reassess/r_3.md:4495:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:110:"transfer_to_occluded": "Two directly usable ideas (the paper IS occluded ReID, so these are native): (1) The attention-magnitude saliency for foreground/background separation — derive a per-patch foreground probability from a lightweight conv-sigmoid head trained ONLY by a foreground-background contrastive loss (no pose/parsing labels), then suppress occluder/background tokens; this is an external-prior-free occluder gate, attractive because pose estimators fail under occlusion. (2) The importance-ranked cumulative fusion (FHFM): rank patches by CLS-attention*norm and let high-confidence (likely visible) patches contribute to every fusion level while low-importance (likely occluded) patches contribute least — a soft, label-free visibility weighting that downweights corrupted regions without a hard mask. The region-restricted masked attention with learnable part tokens is also a clean parsing-free way to get part features robust to missing regions.",
./reassess/r_2.md:1924:reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
./reassess/r_2.md:1925:reassess/r_3.md:4498:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:127:"novel_component": "Intersection-Alignment Score (IAS): turning an explainability signal into a re-ranking operator — measure how much the model's attention overlaps the visible-body mask (IoU) and how mutually consistent query/gallery attention maps are, then asymmetrically shrink or inflate pairwise distances. It makes \"is the model looking at occluded regions?\" a concrete, label-free matching correction rather than a post-hoc visualization.",
./reassess/r_2.md:1985:reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
./reassess/r_2.md:1986:reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
./reassess/r_2.md:1990:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
./reassess/r_2.md:1991:reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
./reassess/r_2.md:2029:reassess/r_3.md:11564:reviews/deep_25.md:6754:trees) and non-target pedestrian occlusion, which often lead to significant loss of body information and increase the complexity of person
./reassess/r_2.md:2037:reassess/r_3.md:11591:reviews/deep_25.md:7720:datasets, our TTPM achieves 73.4 %/63.4 % and 86.8 %/82.2 % Rank1/mAP performance, respectively. Speciﬁcally, compared to the occlusion augmentation method FED [30], which requires additional computation to simulate occlusion data, our proposed method still shows
./reassess/r_2.md:2152:pivot/p_2.md:654:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2157:pivot/p_2.md:665:./validate/v_4.md:28:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:2162:pivot/p_2.md:680:./validate/v_4.md:105:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:2170:pivot/p_2.md:710:./validate/clean/v_4.txt:1:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:2203:pivot/p_2.md:892:./reviews/deep_13.md:7083:/bin/zsh -lc "pdftotext -f 8 -l 12 'Global aggregated gradient-guided adversarial attacks for person re-identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|gradient consistency|consistency|global aggregated|GAGA|Table|Fig|iterations|early|late|transfer|black-box|white-box|Visualization|parameter|inner|m |beta|Input transformation|ensemble\" -C 3" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_2.md:2218:pivot/p_2.md:1039:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2243:pivot/p_2.md:1226:./reviews/deep_25.md:1365:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_2.md:2248:pivot/p_2.md:1259:./reviews/deep_25.md:9584:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_2.md:2269:pivot/p_2.md:1293:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2273:pivot/p_2.md:1298:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2375:pivot/p_2.md:2180:./validate/v_2.md:5106:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2378:pivot/p_2.md:2183:./validate/v_2.md:5118:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2381:pivot/p_2.md:2193:./validate/v_2.md:5170:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2385:pivot/p_2.md:2220:./validate/v_2.md:5929:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2436:pivot/p_2.md:2457:./validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2451:pivot/p_2.md:2502:./validate/v_2.md:12474:validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2463:pivot/p_2.md:2574:./validate/v_3.md:14416:../litreview2/validate/v_3.md:6601:./validate/v_3.md:1820:../claude_review_ovli.md:123:**审查结论：审查通过。** 无 Critical、无 High。OVLI 载荷设计(proj 进 optimizer 且自检、fp16-map→fp32-proj→encoder 梯度回流、共享 bn 混精单 backward、双向对称 MaxSim + opp-view 多正 logsumexp SupCon、AMP fp32 隔离、floor NaN 安全、--ovli off 精确复现 baseline、ovp/ovli 互斥、eval 默认 global-only + rerank opt-in、边界不崩)逐行 + 5 处本地数值核实**全部正确**。M3(novelty 联网坐实)进稿前处理，不阻断 empirical 训练。可启动 kill-switch #2′ → 交 Codex 第二轮独立审查。
./reassess/r_2.md:2472:pivot/p_2.md:3168:./validate/v_3.md:15011:../litreview2/validate/v_3.md:7211:./validate/v_3.md:2595:../codex_review_ovli.txt:3794:experiments/cargo_cvpb/design.md:83:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2515:pivot/p_2.md:3519:./validate/v_3.md:15366:../litreview2/validate/v_3.md:7905:./validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2523:pivot/p_2.md:3539:./validate/v_3.md:15386:../litreview2/validate/v_3.md:7928:./validate/v_2.md:12474:validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2535:pivot/p_2.md:3561:./validate/v_3.md:15408:../litreview2/validate/v_3.md:7960:./validate/v_2.md:12561:validate/v_3.md:1189:./validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2553:pivot/p_2.md:3586:./validate/v_3.md:15433:../litreview2/validate/v_3.md:7986:./validate/v_2.md:12646:validate/v_3.md:1820:../claude_review_ovli.md:123:**审查结论：审查通过。** 无 Critical、无 High。OVLI 载荷设计(proj 进 optimizer 且自检、fp16-map→fp32-proj→encoder 梯度回流、共享 bn 混精单 backward、双向对称 MaxSim + opp-view 多正 logsumexp SupCon、AMP fp32 隔离、floor NaN 安全、--ovli off 精确复现 baseline、ovp/ovli 互斥、eval 默认 global-only + rerank opt-in、边界不崩)逐行 + 5 处本地数值核实**全部正确**。M3(novelty 联网坐实)进稿前处理，不阻断 empirical 训练。可启动 kill-switch #2′ → 交 Codex 第二轮独立审查。
./reassess/r_2.md:2560:pivot/p_2.md:3629:./validate/v_3.md:15476:../litreview2/validate/v_3.md:8032:./validate/v_2.md:12707:validate/v_3.md:2595:../codex_review_ovli.txt:3794:experiments/cargo_cvpb/design.md:83:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2594:pivot/p_2.md:3697:./validate/v_3.md:15544:../litreview2/validate/v_3.md:8185:./validate/v_2.md:14077:validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2652:pivot/p_2.md:3806:./validate/v_3.md:15943:../litreview2/validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2661:pivot/p_2.md:3830:./validate/v_3.md:15970:../litreview2/validate/v_2.md:12474:validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2678:pivot/p_2.md:3861:./validate/v_3.md:16004:../litreview2/validate/v_2.md:12561:validate/v_3.md:1189:./validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2700:pivot/p_2.md:3894:./validate/v_3.md:16037:../litreview2/validate/v_2.md:12646:validate/v_3.md:1820:../claude_review_ovli.md:123:**审查结论：审查通过。** 无 Critical、无 High。OVLI 载荷设计(proj 进 optimizer 且自检、fp16-map→fp32-proj→encoder 梯度回流、共享 bn 混精单 backward、双向对称 MaxSim + opp-view 多正 logsumexp SupCon、AMP fp32 隔离、floor NaN 安全、--ovli off 精确复现 baseline、ovp/ovli 互斥、eval 默认 global-only + rerank opt-in、边界不崩)逐行 + 5 处本地数值核实**全部正确**。M3(novelty 联网坐实)进稿前处理，不阻断 empirical 训练。可启动 kill-switch #2′ → 交 Codex 第二轮独立审查。
./reassess/r_2.md:2707:pivot/p_2.md:3937:./validate/v_3.md:16080:../litreview2/validate/v_2.md:12707:validate/v_3.md:2595:../codex_review_ovli.txt:3794:experiments/cargo_cvpb/design.md:83:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2751:pivot/p_2.md:4016:./validate/v_3.md:16167:../litreview2/validate/v_2.md:14077:validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:2962:pivot/p_2.md:5693:pivot/p_2.md:654:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:2990:pivot/p_2.md:5778:pivot/p_2.md:1039:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3005:pivot/p_2.md:5811:pivot/p_2.md:1226:./reviews/deep_25.md:1365:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_2.md:3010:pivot/p_2.md:5816:pivot/p_2.md:1259:./reviews/deep_25.md:9584:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_2.md:3031:pivot/p_2.md:5837:pivot/p_2.md:1293:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3035:pivot/p_2.md:5842:pivot/p_2.md:1298:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3075:pivot/p_2.md:5941:pivot/p_2.md:2180:./validate/v_2.md:5106:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3078:pivot/p_2.md:5944:pivot/p_2.md:2183:./validate/v_2.md:5118:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3081:pivot/p_2.md:5949:pivot/p_2.md:2193:./validate/v_2.md:5170:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3085:pivot/p_2.md:5963:pivot/p_2.md:2220:./validate/v_2.md:5929:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3248:pivot/p_2.md:6868:reviews/deep_25.md:1365:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_2.md:3253:pivot/p_2.md:6881:reviews/deep_25.md:9584:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_2.md:3274:pivot/p_2.md:6909:reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3280:pivot/p_2.md:6920:reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3338:pivot/p_2.md:7804:reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:3339:pivot/p_2.md:7808:reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:3348:pivot/p_2.md:7968:analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3364:pivot/p_2.md:8150:analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3387:pivot/p_2.md:8381:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_2.md:3392:pivot/p_2.md:8392:./validate/v_4.md:28:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:3397:pivot/p_2.md:8407:./validate/v_4.md:105:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:3405:pivot/p_2.md:8437:./validate/clean/v_4.txt:1:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:3511:web search: person re-identification uncertainty learning feature quality 2024
./reassess/r_2.md:3525:web search: k-reciprocal encoding person re-identification 2017 re-ranking
./reassess/r_2.md:3548: exited 1 in 0ms:
./reassess/r_2.md:3553: exited 1 in 0ms:
./reassess/r_2.md:3558: exited 1 in 0ms:
./reassess/r_2.md:3563: exited 1 in 0ms:
./reassess/r_2.md:3575: exited 1 in 0ms:
./reassess/r_2.md:3636:| 13 | 新协议贴部署 | severe modality imbalance; mix-modality; anytime |
./reassess/r_2.md:3641:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
./reassess/r_2.md:4091:reassess/r_2.md:811:analyses/deep_18.txt:8:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
./reassess/r_2.md:4131:reassess/r_2.md:915:analyses/ALL_ANALYSES.txt:570:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
./reassess/r_2.md:4307:reassess/r_2.md:1570:reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_2.md:4310:reassess/r_2.md:1580:reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_2.md:4372:reassess/r_2.md:1756:reassess/r_3.md:3883:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:2891:of non-overlapping distributed cameras. In the early stages
./reassess/r_2.md:4416:reassess/r_2.md:1917:reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./reassess/r_2.md:4417:reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
./reassess/r_2.md:4419:reassess/r_2.md:1924:reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
./reassess/r_2.md:4420:reassess/r_2.md:1925:reassess/r_3.md:4498:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:127:"novel_component": "Intersection-Alignment Score (IAS): turning an explainability signal into a re-ranking operator — measure how much the model's attention overlaps the visible-body mask (IoU) and how mutually consistent query/gallery attention maps are, then asymmetrically shrink or inflate pairwise distances. It makes \"is the model looking at occluded regions?\" a concrete, label-free matching correction rather than a post-hoc visualization.",
./reassess/r_2.md:4436:reassess/r_2.md:1985:reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
./reassess/r_2.md:4437:reassess/r_2.md:1986:reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
./reassess/r_2.md:4439:reassess/r_2.md:1990:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
./reassess/r_2.md:4440:reassess/r_2.md:1991:reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
./reassess/r_2.md:4516:reassess/r_2.md:2157:pivot/p_2.md:665:./validate/v_4.md:28:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:4520:reassess/r_2.md:2162:pivot/p_2.md:680:./validate/v_4.md:105:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:4525:reassess/r_2.md:2170:pivot/p_2.md:710:./validate/clean/v_4.txt:1:结论先说：原始 kill-switch “非对称包含距离 beat 对称 cosine”不够干净。它最多说明“某种 uncertainty-aware 打分有用”，不能直接证明“跨视角 ReID 本质是非对称信息包含”。要让它有证伪力，必须把“非对称”“包含方向”“图像级不确定性”分别用破坏性对照隔离出来。
./reassess/r_2.md:4608:reassess/r_2.md:2436:pivot/p_2.md:2457:./validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4622:reassess/r_2.md:2451:pivot/p_2.md:2502:./validate/v_2.md:12474:validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4631:reassess/r_2.md:2463:pivot/p_2.md:2574:./validate/v_3.md:14416:../litreview2/validate/v_3.md:6601:./validate/v_3.md:1820:../claude_review_ovli.md:123:**审查结论：审查通过。** 无 Critical、无 High。OVLI 载荷设计(proj 进 optimizer 且自检、fp16-map→fp32-proj→encoder 梯度回流、共享 bn 混精单 backward、双向对称 MaxSim + opp-view 多正 logsumexp SupCon、AMP fp32 隔离、floor NaN 安全、--ovli off 精确复现 baseline、ovp/ovli 互斥、eval 默认 global-only + rerank opt-in、边界不崩)逐行 + 5 处本地数值核实**全部正确**。M3(novelty 联网坐实)进稿前处理，不阻断 empirical 训练。可启动 kill-switch #2′ → 交 Codex 第二轮独立审查。
./reassess/r_2.md:4638:reassess/r_2.md:2472:pivot/p_2.md:3168:./validate/v_3.md:15011:../litreview2/validate/v_3.md:7211:./validate/v_3.md:2595:../codex_review_ovli.txt:3794:experiments/cargo_cvpb/design.md:83:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4669:reassess/r_2.md:2515:pivot/p_2.md:3519:./validate/v_3.md:15366:../litreview2/validate/v_3.md:7905:./validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4676:reassess/r_2.md:2523:pivot/p_2.md:3539:./validate/v_3.md:15386:../litreview2/validate/v_3.md:7928:./validate/v_2.md:12474:validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4687:reassess/r_2.md:2535:pivot/p_2.md:3561:./validate/v_3.md:15408:../litreview2/validate/v_3.md:7960:./validate/v_2.md:12561:validate/v_3.md:1189:./validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4705:reassess/r_2.md:2553:pivot/p_2.md:3586:./validate/v_3.md:15433:../litreview2/validate/v_3.md:7986:./validate/v_2.md:12646:validate/v_3.md:1820:../claude_review_ovli.md:123:**审查结论：审查通过。** 无 Critical、无 High。OVLI 载荷设计(proj 进 optimizer 且自检、fp16-map→fp32-proj→encoder 梯度回流、共享 bn 混精单 backward、双向对称 MaxSim + opp-view 多正 logsumexp SupCon、AMP fp32 隔离、floor NaN 安全、--ovli off 精确复现 baseline、ovp/ovli 互斥、eval 默认 global-only + rerank opt-in、边界不崩)逐行 + 5 处本地数值核实**全部正确**。M3(novelty 联网坐实)进稿前处理，不阻断 empirical 训练。可启动 kill-switch #2′ → 交 Codex 第二轮独立审查。
./reassess/r_2.md:4712:reassess/r_2.md:2560:pivot/p_2.md:3629:./validate/v_3.md:15476:../litreview2/validate/v_3.md:8032:./validate/v_2.md:12707:validate/v_3.md:2595:../codex_review_ovli.txt:3794:experiments/cargo_cvpb/design.md:83:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4743:reassess/r_2.md:2594:pivot/p_2.md:3697:./validate/v_3.md:15544:../litreview2/validate/v_3.md:8185:./validate/v_2.md:14077:validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4784:reassess/r_2.md:2652:pivot/p_2.md:3806:./validate/v_3.md:15943:../litreview2/validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4792:reassess/r_2.md:2661:pivot/p_2.md:3830:./validate/v_3.md:15970:../litreview2/validate/v_2.md:12474:validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4808:reassess/r_2.md:2678:pivot/p_2.md:3861:./validate/v_3.md:16004:../litreview2/validate/v_2.md:12561:validate/v_3.md:1189:./validate/v_3.md:231:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4830:reassess/r_2.md:2700:pivot/p_2.md:3894:./validate/v_3.md:16037:../litreview2/validate/v_2.md:12646:validate/v_3.md:1820:../claude_review_ovli.md:123:**审查结论：审查通过。** 无 Critical、无 High。OVLI 载荷设计(proj 进 optimizer 且自检、fp16-map→fp32-proj→encoder 梯度回流、共享 bn 混精单 backward、双向对称 MaxSim + opp-view 多正 logsumexp SupCon、AMP fp32 隔离、floor NaN 安全、--ovli off 精确复现 baseline、ovp/ovli 互斥、eval 默认 global-only + rerank opt-in、边界不崩)逐行 + 5 处本地数值核实**全部正确**。M3(novelty 联网坐实)进稿前处理，不阻断 empirical 训练。可启动 kill-switch #2′ → 交 Codex 第二轮独立审查。
./reassess/r_2.md:4837:reassess/r_2.md:2707:pivot/p_2.md:3937:./validate/v_3.md:16080:../litreview2/validate/v_2.md:12707:validate/v_3.md:2595:../codex_review_ovli.txt:3794:experiments/cargo_cvpb/design.md:83:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:4878:reassess/r_2.md:2751:pivot/p_2.md:4016:./validate/v_3.md:16167:../litreview2/validate/v_2.md:14077:validate/v_2.md:12370:- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
./reassess/r_2.md:5162:reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_2.md:5166:reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_2.md:5235:reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
./reassess/r_2.md:5300:reassess/r_3.md:3883:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:2891:of non-overlapping distributed cameras. In the early stages
./reassess/r_2.md:5417:reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./reassess/r_2.md:5420:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
./reassess/r_2.md:5422:reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
./reassess/r_2.md:5423:reassess/r_3.md:4498:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:127:"novel_component": "Intersection-Alignment Score (IAS): turning an explainability signal into a re-ranking operator — measure how much the model's attention overlaps the visible-body mask (IoU) and how mutually consistent query/gallery attention maps are, then asymmetrically shrink or inflate pairwise distances. It makes \"is the model looking at occluded regions?\" a concrete, label-free matching correction rather than a post-hoc visualization.",
./reassess/r_2.md:5447:reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
./reassess/r_2.md:5448:reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
./reassess/r_2.md:5455:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
./reassess/r_2.md:5458:reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
./reassess/r_2.md:5460:reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
./reassess/r_2.md:5462:reassess/r_3.md:4768:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:471:- **为何非已关方向**: 与 visibility 加权**正交**(visibility 问"是否被遮挡"二值; burstiness 问"这个可见特征是否被过度计数"集内冗余统计)。一个 part 可完全可见且高 visibility 却 bursty(大块平躯干)——现有方法升权它, burstiness 说降权。非 completion(不重建)、非 scorer 微变体(改特征集聚合)、非 uncertainty(无 per-feature 置信)。**Novelty 已核: 无 occluded-ReID 做 burst-aware/democratic over part/patch 特征**(最近 cousin = Self-similarity guided probabilistic matching ESWA'23 是去噪/校验, 相反符号)。详见 memory [[burstiness-democratic-aggregation-bet]]。
./reassess/r_2.md:5470:reassess/r_3.md:4815:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:51:**收敛点(e30 matched)双确认**：λ=0 vs λ=1 同 rank16/seed/script 跑到 e30 oracle 仍**完全一致**——Jaccard 0.2646 vs 0.2627、oracle +0.85 vs +0.80、fusion best ALL 75.74 vs 75.73（λ=1 甚至略低）。**早期(e10)+收敛(e30) 双证据：解相关在任何训练阶段对互补性零效果。** 加上 decorr-floor 证据（λ=2 双倍权重只把 0.041 降 ~1% → ~0.041 是 ID-constrained floor，共享判别方向是 ID load-bearing 的、删不掉）。→ **张力对显式干预(e10/e30、λ∈{0,1,2}、λ=10 进行中)全程鲁棒，是 fundamental 的，不是可调超参。**
./reassess/r_2.md:5583:reassess/r_3.md:5622:./reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:5584:reassess/r_3.md:5658:./reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:5695:reassess/r_3.md:7347:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_2.md:5716:reassess/r_3.md:7579:./validate/v_2.md:14462:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_2.md:5719:reassess/r_3.md:7583:./validate/v_2.md:14516:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_2.md:5827:reassess/r_3.md:7999:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./reassess/r_2.md:5830:reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
./reassess/r_2.md:5832:reassess/r_3.md:8009:./reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
./reassess/r_2.md:5838:reassess/r_3.md:8032:./reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
./reassess/r_2.md:5846:reassess/r_3.md:8057:./pivot/p_2.md:700:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_2.md:5881:reassess/r_3.md:8249:./pivot/p_2.md:7804:reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:5882:reassess/r_3.md:8250:./pivot/p_2.md:7808:reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:5895:reassess/r_3.md:8298:./pivot/p_2.md:8427:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_2.md:5947:reassess/r_3.md:8696:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./reassess/r_2.md:5951:reassess/r_3.md:8747:analyses/deep_7.txt:59:4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。
./reassess/r_2.md:5961:reassess/r_3.md:8825:analyses/deep_26.txt:23:4. 证据闭环：数据集 AT-USTC 本身证明旧数据集缺口，21 个月、RGB 和红外、多衣服、多季节支撑“anytime”成立。模型证据包括六场景分别评测、跨数据集泛化、MS-ReID、MoAE、HDW 消融，以及和其他 MoE 方法对比，证明不是单纯多头分类。
./reassess/r_2.md:5962:reassess/r_3.md:8844:analyses/ALL_ANALYSES.txt:73:4. **证据闭环**：它用方法表、数据集表、参数量和 Rank-1 对比、benchmark/protocol 汇总、挑战清单和 3DGait 医疗任务 case study 来证明 SRID 不是零散技巧，而是有共同问题、共同评价和外溢应用的方向。
./reassess/r_2.md:5982:reassess/r_3.md:9211:pivot/p_2.md:7804:reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:5983:reassess/r_3.md:9212:pivot/p_2.md:7808:reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:6068:reassess/r_3.md:10875:reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:6069:reassess/r_3.md:10910:reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_2.md:6201: exited 1 in 0ms:
./reassess/r_2.md:6424:web search: "adaptive" "person re-identification" "computation" "query difficulty"
./reassess/r_2.md:6440:结论：我只推荐 **1 个主线 + 1 个备胎**。不要再救 SMPL / 遮挡 / 航拍-地面；也不要把“相机偏差”当主线，DART³ 已经把 camera bias + distance-aware TTA 写得很近，CA-Jaccard 也占了 camera-aware neighbor/re-ranking 这块。([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2311.10605))
./reassess/r_2.md:6467:- k-reciprocal re-ranking 已经用 reciprocal neighbor 做 test-time ranking，但它不是把 **gallery in-degree hubness** 定义成强 ReID 的失败变量，也不是训练端 anti-hub。([arxiv.org](https://arxiv.org/abs/1701.08398))
./reassess/r_2.md:6471:**方向 2，备胎：Rank-Instability Adaptive ReID**
./reassess/r_2.md:6472:一句话：如果强 backbone 已经很强，继续卷 mAP 可能不现实；换问题轴：**不是所有 query 都需要同等计算/同等复杂匹配**。隐藏变量是 rank instability。
./reassess/r_2.md:6475:同一冻结 Swin，提多种便宜特征：低分辨率、浅层 stage、CLS/avg、final feature。若多数 query 的 cheap rank 与 full rank 高度一致，少数 query 排名剧烈变化且承包错误，就有新问题。
./reassess/r_2.md:6478:ReID 当前默认 uniform inference：每个 query 都走同样 backbone、同样检索策略。实际应是 **rank-stability conditioned retrieval**：稳定 query 早退，unstable query 才走 full feature / rerank / expert。
./reassess/r_2.md:6481:- 多出口 Swin 或低/高分辨率 cascade；
./reassess/r_2.md:6482:- 用 cheap-vs-full top-k overlap、rank entropy、margin stability 作为 gating signal；
./reassess/r_2.md:6483:- 训练时加 rank-consistency / instability predictor；
./reassess/r_2.md:6484:- 目标不是单纯 mAP，而是 mAP-FLOPs Pareto 或 fixed-budget mAP。
./reassess/r_2.md:6488:2. 定义 `R(q)=1-Jaccard(topK_cheap, topK_full)` 或 Kendall rank instability。
./reassess/r_2.md:6490:4. oracle cascade：只对 top p% unstable query 用 full feature，其余用 cheap feature。若 40-60% FLOPs 下 mAP 接近 full，方向活。
./reassess/r_2.md:6491:5. 破坏对照：随机 p%、只用 final margin、只用 norm。如果 instability 不比这些强，杀掉。
./reassess/r_2.md:6494:- AT-ReID 2025 的 “anytime” 是昼夜/长期/换衣多场景，不是 computational early-exit。([arxiv.org](https://arxiv.org/abs/2509.16635))
./reassess/r_2.md:6501:结论：我只推荐 **1 个主线 + 1 个备胎**。不要再救 SMPL / 遮挡 / 航拍-地面；也不要把“相机偏差”当主线，DART³ 已经把 camera bias + distance-aware TTA 写得很近，CA-Jaccard 也占了 camera-aware neighbor/re-ranking 这块。([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2311.10605))
./reassess/r_2.md:6528:- k-reciprocal re-ranking 已经用 reciprocal neighbor 做 test-time ranking，但它不是把 **gallery in-degree hubness** 定义成强 ReID 的失败变量，也不是训练端 anti-hub。([arxiv.org](https://arxiv.org/abs/1701.08398))
./reassess/r_2.md:6532:**方向 2，备胎：Rank-Instability Adaptive ReID**
./reassess/r_2.md:6533:一句话：如果强 backbone 已经很强，继续卷 mAP 可能不现实；换问题轴：**不是所有 query 都需要同等计算/同等复杂匹配**。隐藏变量是 rank instability。
./reassess/r_2.md:6536:同一冻结 Swin，提多种便宜特征：低分辨率、浅层 stage、CLS/avg、final feature。若多数 query 的 cheap rank 与 full rank 高度一致，少数 query 排名剧烈变化且承包错误，就有新问题。
./reassess/r_2.md:6539:ReID 当前默认 uniform inference：每个 query 都走同样 backbone、同样检索策略。实际应是 **rank-stability conditioned retrieval**：稳定 query 早退，unstable query 才走 full feature / rerank / expert。
./reassess/r_2.md:6542:- 多出口 Swin 或低/高分辨率 cascade；
./reassess/r_2.md:6543:- 用 cheap-vs-full top-k overlap、rank entropy、margin stability 作为 gating signal；
./reassess/r_2.md:6544:- 训练时加 rank-consistency / instability predictor；
./reassess/r_2.md:6545:- 目标不是单纯 mAP，而是 mAP-FLOPs Pareto 或 fixed-budget mAP。
./reassess/r_2.md:6549:2. 定义 `R(q)=1-Jaccard(topK_cheap, topK_full)` 或 Kendall rank instability。
./reassess/r_2.md:6551:4. oracle cascade：只对 top p% unstable query 用 full feature，其余用 cheap feature。若 40-60% FLOPs 下 mAP 接近 full，方向活。
./reassess/r_2.md:6552:5. 破坏对照：随机 p%、只用 final margin、只用 norm。如果 instability 不比这些强，杀掉。
./reassess/r_2.md:6555:- AT-ReID 2025 的 “anytime” 是昼夜/长期/换衣多场景，不是 computational early-exit。([arxiv.org](https://arxiv.org/abs/2509.16635))
./reassess/r_3.md:28: exited 1 in 0ms:
./reassess/r_3.md:882:| 062-eq | PSG+GCN + LKU (eq_concat) | 60.7% | 71.2% | 84.1% | 87.4% | mAP+4.1%, R1+4.7% | ❌ 负面。vs 3-seed: -0.03%/-1.37%。Learned uncertainty 损害 R1 |
./reassess/r_3.md:902:| 078-eq | PSG+GCN+PAA+APG (eq) | 60.5% | 72.5% | 84.3% | 87.9% | mAP+3.9%, R1+6.0% | ❌ vs PAA: -1.1%/-1.7%。adaptive gate 负面 |
./reassess/r_3.md:974:| 157+rr | PLBOA+Re-ranking | 78.8% | 79.7% | 87.8% | 90.0% | 🟢🟢🟢🟢🟢 含 re-ranking |
./reassess/r_3.md:1023:1. NFC/Re-ranking 等 test-time 方法所有 SOTA 都可以用，不算公平的对比
./reassess/r_3.md:1976:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_7.md:7025:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
./reassess/r_3.md:1980:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_7.md:7106:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
./reassess/r_3.md:1993:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_19.md:666:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2017:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:300:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2067:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:864:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2153:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:545:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2260:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:213:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2389:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/novelty_gopl.txt:38:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
./reassess/r_3.md:2400:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:470:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2435:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:703:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2466:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp155/design.md:54:- 关键看 `evid_unc`: 如果遮挡样本确实显示更高 uncertainty → 机制生效
./reassess/r_3.md:2483:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:880:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2501:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:991:**方向（用户决策#33=选项1）**：放开 SOLIDER 强主干，回常规 ImageNet 预训练主干(TransReID ViT-base、ImageNet 预训练 Swin-Tiny)刷标准遮挡 ReID mAP/rank1。可靠性/conformal 线归档(决策#28-#32 的负结果+诊断作分析素材)。
./reassess/r_3.md:2533:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:534:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2578:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:603:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2615:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:778:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2637:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:13269:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./reassess/r_3.md:2643:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:13350:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
./reassess/r_3.md:2662:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:534:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2701:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:597:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2732:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:302:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:2763:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:6486:4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。
./reassess/r_3.md:2770:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:6563:4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。
./reassess/r_3.md:2822:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:654:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:2847:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:830:./analyses/deep_14.txt:21:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
./reassess/r_3.md:2869:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1005:./analyses/ALL_ANALYSES.txt:359:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
./reassess/r_3.md:2885:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1039:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:2918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1226:./reviews/deep_25.md:1365:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_3.md:2923:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1259:./reviews/deep_25.md:9584:FrankMocap [61] to recover SMPL mesh sequences from the
./reassess/r_3.md:2944:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1293:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:2949:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1298:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:2989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1760:./reviews/deep_14.md:6594:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
./reassess/r_3.md:2992:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1764:./reviews/deep_14.md:6641:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
./reassess/r_3.md:3019:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2180:./validate/v_2.md:5106:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:3022:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2183:./validate/v_2.md:5118:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:3026:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2193:./validate/v_2.md:5170:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:3035:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2220:./validate/v_2.md:5929:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:3086:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3037:./validate/v_3.md:14880:../litreview2/validate/v_3.md:7080:./validate/v_3.md:2462:../codex_review_ovli.txt:3634:experiments/exp324d/design.md:15:给 DINOv2-base attention 加 LoRA（q/v，rank 8/16，alpha 16），DINO 主权重冻结，只训 LoRA + 轻量头（proj + BNNeck + 全局分类器 + part 分类器），**重遮挡 part-MaxSim mAP 能突破 14 往 competitive 走**。能破 → DINO 创新线可做成；破不动 → 冻结对应特征这条线天花板确实低，止损。
./reassess/r_3.md:3108:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3322:./validate/v_3.md:15165:../litreview2/validate/v_3.md:7365:./validate/v_3.md:2751:../codex_review_ovli.txt:3956:experiments/overnight_innovation_log.md:44:- **exp324d 第2次（本次）已启动**（2026-06-16，lab-3090-d）：`scripts/exp324d_lora.py` 写好——LoRA(q/v rank8 alpha16) 解冻 DINOv2-base + **可微姿态部位池化**(预算每图 row-stochastic 池化矩阵 pool_w，`bmm(pool_w,patch)` 梯度回流 LoRA) + exp324b 同款头/损失/PK64/part-MaxSim eval。**唯一变量=LoRA 解冻 DINO**。双审查 approve（Claude 含 200-trial 池化等价 1.79e-7；Codex 141k tok 确认梯度到 LoRA/full-batch triplet/use_reentrant=False/eval 对称，组合 plausibly new 未找直接先例）。Dry-run：LoRA 294,912+head 413,184 训练，loss↓ acc 0.016→0.766，显存 13.7G/24G，~1.4s/step。全量 35ep 后台跑（micro_bs64 no-ckpt）。**待结果：part-MaxSim 重遮挡 mAP 能否破 exp324b 的 8.65 / 全部破 14.61 往 competitive 走。**【结果待填】
./reassess/r_3.md:3109:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3323:./validate/v_3.md:15166:../litreview2/validate/v_3.md:7366:./validate/v_3.md:2752:../codex_review_ovli.txt:3957:experiments/overnight_innovation_log.md:49:1. **Rank-disagreement oracle（0-GPU，~40行 numpy，⭐立即做，依赖 Swin distmat）**：重遮挡子集上比 Swin top-k vs DINO part-MaxSim top-k 的 Jaccard / P_dino_only（DINO 命中 Swin 漏的真值）/ per-query oracle 上界。判定：P_dino_only<2% 或 oracle<+1mAP → 整条 DINO⊕Swin 当场止损；oracle +3~5 且 Jaccard<0.5 → 正交坐实，进 #2。
./reassess/r_3.md:3110:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3324:./validate/v_3.md:15167:../litreview2/validate/v_3.md:7367:./validate/v_3.md:2753:../codex_review_ovli.txt:3958:experiments/overnight_innovation_log.md:50:2. **遮挡门控 part-MaxSim re-rank（training-free，最可能产出有用主表结果）**：DINO 共可见 distmat 当 `utils/reranking.py re_ranking(local_distmat=)` 输入，只对重遮挡 query 触发。依赖 #1。
./reassess/r_3.md:3113:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3330:./validate/v_3.md:15173:../litreview2/validate/v_3.md:7373:./validate/v_3.md:2759:../codex_review_ovli.txt:3964:experiments/overnight_innovation_log.md:100:- #2（遮挡门控 part-MaxSim re-rank）**依赖 #1 正向 → 现已被砍**，DINO 的共可见 distmat 当 re-rank 输入不会有增益，不做。
./reassess/r_3.md:3287:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_3.md:3569:区别：FLaN-Net 用语言分人、属性、遮挡物；CFPER 用二维部位专家。我们用三维人体表面和投影可见性定义匹配支持区域，重点解决俯视、低清、遮挡时二维部位顺序失真的问题。  
./reassess/r_3.md:3370:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:536:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
./reassess/r_3.md:3464:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:217:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:3499:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3262:廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。
./reassess/r_3.md:3513:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3337:廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。
./reassess/r_3.md:3529:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324h/design.md:27:- **(若 oracle 正)** 顺手 fusion sweep（z-score / min-max，w∈{0..0.5}）+ k-reciprocal re-rank
./reassess/r_3.md:3530:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324h/design.md:28:  （`utils/reranking.py` re_ranking with local_distmat），看重遮挡/全部能否 > Swin 75 单独。
./reassess/r_3.md:3584:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:363:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:3648:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:380:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
./reassess/r_3.md:3718:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:363:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:3776:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:628:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:3794:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2855:最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。
./reassess/r_3.md:3800:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2928:最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。
./reassess/r_3.md:3817:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:460:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:3883:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:2891:of non-overlapping distributed cameras. In the early stages
./reassess/r_3.md:3921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:660:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:3945:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3639:便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。
./reassess/r_3.md:3957:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3693:便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。
./reassess/r_3.md:3990:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:359:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
./reassess/r_3.md:4068:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:94:4. **冷静提醒（吸收陷阱推论）**：inter-person lead 要做成"不被吸收"，必须 test-time + cross-instance（pair 条件化判断 Q 的遮挡人是否就是 G）→ 这又落进 **CVK/re-ranker 家族**（已证测试期仅 +0.8~1.1%）。所以 **lead 本身大概率也是小增量 re-rank**，不是训练端主创新。
./reassess/r_3.md:4095:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp323/design.md:55:- **小模型（3B/7B）B 显著 > A 且集中重遮挡** → 真信号 → exp323b：LoRA 微调小 Qwen verifier + 全量 re-rank + 可控遮挡(PLBOA)消融 + GPT-5.5 蒸馏 teacher。
./reassess/r_3.md:4271:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_4.md:21478:../airl_related_work.md:26:RAR 用 query resolution **动态路由**选子空间。AIRL **不路由**——#3 oracle 实测硬路由(area/reliability 阈值)失败(≤+0.41),**软固定先验融合**(w=0.25,clean+recover 两证据头)反而 +1.46。所以 AIRL 的 claim 必须是 "fixed-prior fusion of clean/recover evidence heads under an observation-limited ceiling",**绝不吹成 query-budget routing**(那既撞 RAR 又与实现不符)。
./reassess/r_3.md:4374:/Users/abdslm/Desktop/SOLIDER-REID/experiments/MORNING_REPORT.md:78:**2. ×4 adaptation 发现** + **可复用诊断工具**（rank-disagreement oracle：0-GPU 判两个表征是否互补 + 出 motivation 图）。
./reassess/r_3.md:4398:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/design.md:1:# 实验 exp324g: rank-disagreement oracle 诊断（0-训练）
./reassess/r_3.md:4404:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/design.md:13:  说明 DINO 在重遮挡上有 Swin 拿不到的独立正确信息，值得做遮挡门控 re-rank。
./reassess/r_3.md:4412:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/design.md:40:- oracle +3~5 且 Jaccard<0.5 → "正交性坐实，值得做 #2 遮挡门控 re-rank"。
./reassess/r_3.md:4419:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/monitor.md:1:# exp324g monitor — rank-disagreement oracle（0-训练诊断）
./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
./reassess/r_3.md:4495:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:110:"transfer_to_occluded": "Two directly usable ideas (the paper IS occluded ReID, so these are native): (1) The attention-magnitude saliency for foreground/background separation — derive a per-patch foreground probability from a lightweight conv-sigmoid head trained ONLY by a foreground-background contrastive loss (no pose/parsing labels), then suppress occluder/background tokens; this is an external-prior-free occluder gate, attractive because pose estimators fail under occlusion. (2) The importance-ranked cumulative fusion (FHFM): rank patches by CLS-attention*norm and let high-confidence (likely visible) patches contribute to every fusion level while low-importance (likely occluded) patches contribute least — a soft, label-free visibility weighting that downweights corrupted regions without a hard mask. The region-restricted masked attention with learnable part tokens is also a clean parsing-free way to get part features robust to missing regions.",
./reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
./reassess/r_3.md:4498:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:127:"novel_component": "Intersection-Alignment Score (IAS): turning an explainability signal into a re-ranking operator — measure how much the model's attention overlaps the visible-body mask (IoU) and how mutually consistent query/gallery attention maps are, then asymmetrically shrink or inflate pairwise distances. It makes \"is the model looking at occluded regions?\" a concrete, label-free matching correction rather than a post-hoc visualization.",
./reassess/r_3.md:4665:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324d_r32/design.md:24:- 假设成立：重遮挡 mAP 明显超过 base-rank16 与冻结 8.65。
./reassess/r_3.md:4691:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:44:- **exp324d 第2次（本次）已启动**（2026-06-16，lab-3090-d）：`scripts/exp324d_lora.py` 写好——LoRA(q/v rank8 alpha16) 解冻 DINOv2-base + **可微姿态部位池化**(预算每图 row-stochastic 池化矩阵 pool_w，`bmm(pool_w,patch)` 梯度回流 LoRA) + exp324b 同款头/损失/PK64/part-MaxSim eval。**唯一变量=LoRA 解冻 DINO**。双审查 approve（Claude 含 200-trial 池化等价 1.79e-7；Codex 141k tok 确认梯度到 LoRA/full-batch triplet/use_reentrant=False/eval 对称，组合 plausibly new 未找直接先例）。Dry-run：LoRA 294,912+head 413,184 训练，loss↓ acc 0.016→0.766，显存 13.7G/24G，~1.4s/step。全量 35ep 后台跑（micro_bs64 no-ckpt）。**待结果：part-MaxSim 重遮挡 mAP 能否破 exp324b 的 8.65 / 全部破 14.61 往 competitive 走。**【结果待填】
./reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
./reassess/r_3.md:4693:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:49:1. **Rank-disagreement oracle（0-GPU，~40行 numpy，⭐立即做，依赖 Swin distmat）**：重遮挡子集上比 Swin top-k vs DINO part-MaxSim top-k 的 Jaccard / P_dino_only（DINO 命中 Swin 漏的真值）/ per-query oracle 上界。判定：P_dino_only<2% 或 oracle<+1mAP → 整条 DINO⊕Swin 当场止损；oracle +3~5 且 Jaccard<0.5 → 正交坐实，进 #2。
./reassess/r_3.md:4694:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:50:2. **遮挡门控 part-MaxSim re-rank（training-free，最可能产出有用主表结果）**：DINO 共可见 distmat 当 `utils/reranking.py re_ranking(local_distmat=)` 输入，只对重遮挡 query 触发。依赖 #1。
./reassess/r_3.md:4696:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:58:**执行计划**：exp324f agent 正在 lab-4090 算 Swin distmat → 它一落地我立刻跑 #1 oracle（0-GPU）。正向 → #2 re-rank（training-free 主表素材）。OT 线(#3)等某 GPU 空了上。
./reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
./reassess/r_3.md:4701:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:80:- **下一步(最高价值)**: 在 lab-3090-d 跑 planner#1 rank-disagreement oracle(Swin vs DINO 正交性 + oracle 上界)，gate 整条 DINO⊕Swin 家族。
./reassess/r_3.md:4702:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:83:- **exp324d LoRA 实际在健康运行**（我先前误判：grep 错了 log 路径 /tmp/exp324d.log，实际是 /tmp/exp324d_r16.log）。第一个 agent 其实完成了全套：写了可微部位池化脚本(bmm pooling matrix，与 exp324 数值等价 1.8e-6)、**过了 Claude+Codex 双审查(都 approve，确认梯度到 LoRA、无遮挡 ReID 先例)**、launch 了 rank16/alpha16/dropout0.05/grad-ckpt/30ep 训练(PID 309591)。两个并发 agent 撞了一下但收敛到一个健康 run。DINO-in-loop 慢(分钟级/epoch)，epoch5 首评出关键数：part 重遮挡能否破 8.65、全部破 14.61。
./reassess/r_3.md:4705:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:87:### [exp324g — planner#1 rank-disagreement oracle 完成 → DINO⊕Swin 家族止损]
./reassess/r_3.md:4710:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:100:- #2（遮挡门控 part-MaxSim re-rank）**依赖 #1 正向 → 现已被砍**，DINO 的共可见 distmat 当 re-rank 输入不会有增益，不做。
./reassess/r_3.md:4721:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:155:- **核心问题**：large 或 rank32 能否把重遮挡 mAP 明显推过冻结 8.65 往 competitive（几十分）走？等 e5 第一次 eval（决定性）。
./reassess/r_3.md:4722:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:158:**base-LoRA rank16，epoch 5/30**：part 重遮挡 mAP **34.92**(vs 冻结 8.65，**×4.0 +26.3**) · part 全部 **42.45**(vs 14.61) · cos 重遮挡 32.38 · cos 全部 41.06。**才 e5/30，还在陡升。**
./reassess/r_3.md:4728:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:207:1. **真新机制**：LoRA↔visibility **交互**（pose/visibility-conditioned LoRA，或 per-part low-rank experts 按遮挡 gate），带消融证明——不是"加了 LoRA 加了 parts"。
./reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
./reassess/r_3.md:4750:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:365:λ=0 vs λ=1 **同 rank16/seed/script，e30 收敛点 matched oracle**：
./reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
./reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
./reassess/r_3.md:4768:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:471:- **为何非已关方向**: 与 visibility 加权**正交**(visibility 问"是否被遮挡"二值; burstiness 问"这个可见特征是否被过度计数"集内冗余统计)。一个 part 可完全可见且高 visibility 却 bursty(大块平躯干)——现有方法升权它, burstiness 说降权。非 completion(不重建)、非 scorer 微变体(改特征集聚合)、非 uncertainty(无 per-feature 置信)。**Novelty 已核: 无 occluded-ReID 做 burst-aware/democratic over part/patch 特征**(最近 cousin = Self-similarity guided probabilistic matching ESWA'23 是去噪/校验, 相反符号)。详见 memory [[burstiness-democratic-aggregation-bet]]。
./reassess/r_3.md:4779:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:510:burstiness 死后调研 agent(带"in-domain 死、frozen 会骗人"教训)Rank-1 过审 bet。**为何逃过训练吸收(结构性)**: held-out (occluder类×部位)组合**按构造从没进训练集**→ 训练模型无可隐式吸收。
./reassess/r_3.md:4807:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:15:DINOv2 pose-anchored part-MaxSim vs Swin MaxSim，重遮挡 oracle 检查：top-10 Jaccard 0.062，**P_dino_only 0.20%**(989 query 里只 2 个 DINO 命中而 Swin 漏)，**oracle 上界仅 +0.12 mAP**。低 Jaccard 是"假正交"——DINO 不是补充，是全局太弱(8.65 vs 72.57)。Swin 错的地方 DINO 也错。→ DINO⊕Swin 融合/re-rank 家族全死（两 agent 独立确认）。
./reassess/r_3.md:4809:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:21:LoRA 解冻 DINOv2-base(~1M 可训：0.6M LoRA + 0.4M 头，DINO 主体冻结) + 可微 pose-part-MaxSim：重遮挡 mAP **8.65 → 36.78**(e10，×4.2)，全部 14.61 → 44.67。rank32 36.72、large 待定。**证明瓶颈是 adaptation 不是特征**——这是 oracle"冻结无独立信息"的对照解释。
./reassess/r_3.md:4815:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:51:**收敛点(e30 matched)双确认**：λ=0 vs λ=1 同 rank16/seed/script 跑到 e30 oracle 仍**完全一致**——Jaccard 0.2646 vs 0.2627、oracle +0.85 vs +0.80、fusion best ALL 75.74 vs 75.73（λ=1 甚至略低）。**早期(e10)+收敛(e30) 双证据：解相关在任何训练阶段对互补性零效果。** 加上 decorr-floor 证据（λ=2 双倍权重只把 0.041 降 ~1% → ~0.041 是 ID-constrained floor，共享判别方向是 ID load-bearing 的、删不掉）。→ **张力对显式干预(e10/e30、λ∈{0,1,2}、λ=10 进行中)全程鲁棒，是 fundamental 的，不是可调超参。**
./reassess/r_3.md:4817:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:61:3. **可复用诊断工具**：rank-disagreement oracle（0-GPU 判两表征是否互补 + motivation 图）、按可见度子集拆 mAP。
./reassess/r_3.md:4840:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp064/design.md:11:- **核心假设**: 遮挡导致 keypoint 特征的可靠性不同。将特征建模为分布而非点向量，让 uncertainty 自然涌现
./reassess/r_3.md:4841:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp064/design.md:26:- 如果成功: uncertainty 与遮挡相关 → 自适应距离 → mAP/R1 提升
./reassess/r_3.md:5019:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp129/monitor.md:47:  2. residual target 没有污染默认 early training，因为 `epoch<=20` 时 `CSRD` 尚未激活
./reassess/r_3.md:5242:./reviews/deep_19.md:5358:can extract fine-grained pedestrian features, they are unable to adaptively segment features based on occlusion
./reassess/r_3.md:5296:./reviews/deep_18.md:1013:76:    76	of model parameters scales linearly with the number of source domains due to the increase of the
./reassess/r_3.md:5345:./reviews/deep_18.md:6659:76:    76	of model parameters scales linearly with the number of source domains due to the increase of the
./reassess/r_3.md:5355:./reviews/deep_18.md:6866:1343:  1343	protocol-3: (1) The imbalance in the quantity of data in the source domain increases the difficulty of
./reassess/r_3.md:5374:./reviews/deep_2.md:1403:Cumulative Matching Characteristics (CMC) Rank-1 accuracy. Our implementation strictly uses only source domain labels and target domain
./reassess/r_3.md:5377:./reviews/deep_2.md:4651:by adaptively calculating the occlusion weights for body parts.
./reassess/r_3.md:5384:./reviews/deep_2.md:4758:occlusion-adaptive local features that accurately reflect the
./reassess/r_3.md:5386:./reviews/deep_2.md:4767:obtains body regions that are adaptive to occlusion by
./reassess/r_3.md:5387:./reviews/deep_2.md:4773:which can adaptively calculate occlusion weights. These
./reassess/r_3.md:5404:./reviews/deep_2.md:5329:adaptively capture key information even in severe occlusion.
./reassess/r_3.md:5421:./reviews/deep_2.md:9363:Cumulative Matching Characteristics (CMC) Rank-1 accuracy. Our implementation strictly uses only source domain labels and target domain
./reassess/r_3.md:5565:./reviews/deep_16.md:1352:1018:[12] C. Zhao, Z. Qu, X. Jiang, Y. Tu, X. Bai, Content-adaptive auto-occlusion network
./reassess/r_3.md:5568:./reviews/deep_16.md:2471:/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|diffusion|inpainting|re-ranking|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:5592:./reviews/deep_16.md:2525:152:use adaptive position extraction with occlusion weighting, while
./reassess/r_3.md:5611:./reviews/deep_16.md:2563:360:reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reassess/r_3.md:5622:./reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_3.md:5627:./reviews/deep_16.md:2983:   360	reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reassess/r_3.md:5642:./reviews/deep_16.md:5451:  1018	[12] C. Zhao, Z. Qu, X. Jiang, Y. Tu, X. Bai, Content-adaptive auto-occlusion network
./reassess/r_3.md:5651:./reviews/deep_16.md:5918:360:reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reassess/r_3.md:5658:./reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_3.md:5660:./reviews/deep_16.md:6096:2420:cases, the ViT-extracted features are passed through DDO filtering and local re-ranking under occlusion. An actionable
./reassess/r_3.md:5677:./reviews/deep_16.md:6705:  2420	cases, the ViT-extracted features are passed through DDO filtering and local re-ranking under occlusion. An actionable
./reassess/r_3.md:5708:./reviews/deep_5.md:2138:adaptively allocates resources based on query difficulty. For
./reassess/r_3.md:5711:./reviews/deep_5.md:2227:allocates computational resources based on the difficulty of
./reassess/r_3.md:5713:./reviews/deep_5.md:2266:mechanism that balances computational efficiency and retrieval accuracy by adaptively allocating resources based
./reassess/r_3.md:5721:./reviews/deep_5.md:2545:adaptive allocation of computational resources between “easy”
./reassess/r_3.md:5736:./reviews/deep_5.md:8399:2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
./reassess/r_3.md:5737:./reviews/deep_5.md:8473:2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
./reassess/r_3.md:5750:./reviews/deep_6.md:3205:   673	environments, lacking real-world complexities such as occlusion, clothing changes, and other appearance variations. To address these challenges, we have developed a new V-I ReID dataset, CMG-P, which incorporates more complex scenarios. As illustrated in Table 1, CMG-P
./reassess/r_3.md:5826:./reviews/deep_23.md:675:KPR (ECCV 2024), ETND (TCSVT 2024) and TTPM (PR 2025), RMPSNet improves the rank-1 performance on
./reassess/r_3.md:5921:./reviews/deep_23.md:1960:without incurring excessive redundancy. RMPSNet justifies its moderate increase in complexity by providing significant gains in occlusion
./reassess/r_3.md:5924:./reviews/deep_23.md:1968:models may remain preferable in highly resource-constrained environments. The practical impact of the increased complexity is a trade-off
./reassess/r_3.md:5970:./reviews/deep_23.md:6944:occlusion, viewpoint variations, and the inherent difficulty of
./reassess/r_3.md:5997:./reviews/deep_4.md:806:adaptive occlusion-aware prompts for occluded ReID.
./reassess/r_3.md:6005:./reviews/deep_4.md:1591:(index 4) yields the highest performance, with Rank-1 reaching 75.2% and mAP 65.5%. These results validate the effectiveness of occlusion-aware fine-grained prompts in guiding
./reassess/r_3.md:6085:./reviews/deep_11.md:4959:   405	By setting different standard deviations, source tasks of different difficulty levels can be obtained.
./reassess/r_3.md:6089:./reviews/deep_11.md:5066:    53	By setting different standard deviations, source tasks of different difficulty levels can be obtained.
./reassess/r_3.md:6115:./reviews/deep_11.md:10684:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:6116:./reviews/deep_11.md:10763:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:6264:./reviews/deep_20.md:5857:    52	difficulty stems from dual occlusion patterns: (1) object-induced occlusion, where environmental elements (e.g., vehicles, buildings) obstruct
./reassess/r_3.md:6335:./reviews/deep_14.md:3445:   803	can adaptively suppress interference from occlusion and facilitate the network in learning discriminative person features.
./reassess/r_3.md:6368:./reviews/deep_14.md:5984:  1235	adaptive dynamic recovery of the occlusion part. Additionally,
./reassess/r_3.md:6372:./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:6589:./reviews/deep_1.md:7731:extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
./reassess/r_3.md:6663:./reviews/deep_1.md:8563:It is observed that mAP and Rank-1 reach the best when the parame-                        In this paper, a semantic-guided occlusion simulation based local
./reassess/r_3.md:6669:./reviews/deep_1.md:8600:lightweight and adaptive occlusion simulation strategies and domain                                    ceedings of the IEEE/CVF International Conference on Computer Vision, 2021,
./reassess/r_3.md:6698:./reviews/deep_1.md:8767:127:extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
./reassess/r_3.md:6774:./reviews/deep_1.md:9241:1374:lightweight and adaptive occlusion simulation strategies and domain
./reassess/r_3.md:6910:./reviews/deep_26.md:11996:introduced additional complexity due to posture variability, clothing diversity, and frequent occlusions between individuals, which negatively
./reassess/r_3.md:6954:./reviews/deep_21.md:1884:   344	information in the intermediate layer and adaptively suppressing the expression of occlusion information. Compared with
./reassess/r_3.md:6959:./reviews/deep_21.md:2015:   475	in the VARA module, which enables adaptive suppression of occlusion information, this alone is insufficient for
./reassess/r_3.md:7007:./reviews/deep_21.md:5288:   118	Fig. 16. The Rank-5 performance of our method under different occlusion
./reassess/r_3.md:7010:./reviews/deep_21.md:5300:   130	pedestrian ranking results under heavy occlusion conditions
./reassess/r_3.md:7012:./reviews/deep_21.md:5304:   134	occlusion issues relatively well, and the Rank-1 results are
./reassess/r_3.md:7092:./analyses/deep_25.txt:71:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:7102:./analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7120:./analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:7130:./analyses/ALL_ANALYSES.txt:1576:2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
./reassess/r_3.md:7199:./reviews/deep_25.md:6754:trees) and non-target pedestrian occlusion, which often lead to significant loss of body information and increase the complexity of person
./reassess/r_3.md:7209:./reviews/deep_25.md:6776:Some methods [6,7] directly use the output of pose estimation models as local masks without adaptive adjustment according to occlusion
./reassess/r_3.md:7258:./reviews/deep_25.md:7720:datasets, our TTPM achieves 73.4 %/63.4 % and 86.8 %/82.2 % Rank1/mAP performance, respectively. Speciﬁcally, compared to the occlusion augmentation method FED [30], which requires additional computation to simulate occlusion data, our proposed method still shows
./reassess/r_3.md:7277:./reviews/deep_25.md:8142:(+8.3 %) on the Partial-REID dataset. The experimental results of Rank1 and Rank-3 validate that our TTPM also exhibits practical applicability
./reassess/r_3.md:7316:./reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:7319:./reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:7327:./analyses/deep_5.txt:28:2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
./reassess/r_3.md:7347:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:7361:./validate/v_2.md:429:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7425:./validate/v_2.md:4706:./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7439:./validate/v_2.md:5821:./analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7468:./validate/v_2.md:6806:./reviews/deep_11.md:10684:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7469:./validate/v_2.md:6814:./reviews/deep_11.md:10763:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7470:./validate/v_2.md:6836:analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7476:./validate/v_2.md:7016:validate/v_3.md:1790:./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7482:./validate/v_2.md:7262:analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7484:./validate/v_2.md:7306:validate/v_2.md:429:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7515:./validate/v_2.md:8483:validate/v_2.md:4706:./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7517:./validate/v_2.md:8769:validate/v_2.md:5821:./analyses/ALL_ANALYSES.txt:195:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7521:./validate/v_2.md:9031:validate/v_2.md:6806:./reviews/deep_11.md:10684:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7522:./validate/v_2.md:9032:validate/v_2.md:6814:./reviews/deep_11.md:10763:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7553:./validate/v_2.md:10387:reviews/deep_11.md:10684:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7554:./validate/v_2.md:10388:reviews/deep_11.md:10763:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7579:./validate/v_2.md:14462:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:7583:./validate/v_2.md:14516:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:7611:./validate/v_3.md:1790:./analyses/deep_11.txt:55:2. 重定义动作：它把 UDA 从“目标域伪标签怎么修”改成“先用多源域学一个更好的、轻量的、可泛化初始化，再进入目标域伪标签学习”。关键词是 lightweight multi-source domain adaptation、self-paced meta-learning、style uncertainty augment、whole-part feature、edge deployment。它还把效率从附加指标提升成问题定义的一部分。
./reassess/r_3.md:7615:./validate/v_3.md:1889:./reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:7618:./validate/v_3.md:1908:./reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:7631:./validate/v_3.md:2947:../litreview2/analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:7637:./validate/v_3.md:3031:../litreview2/analyses/ALL_ANALYSES.txt:1576:2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
./reassess/r_3.md:7647:./validate/v_3.md:5081:./reviews/deep_18.md:1013:76:    76	of model parameters scales linearly with the number of source domains due to the increase of the
./reassess/r_3.md:7648:./validate/v_3.md:5105:./reviews/deep_18.md:6659:76:    76	of model parameters scales linearly with the number of source domains due to the increase of the
./reassess/r_3.md:7999:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
./reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
./reassess/r_3.md:8007:./reassess/r_3.md:4495:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:110:"transfer_to_occluded": "Two directly usable ideas (the paper IS occluded ReID, so these are native): (1) The attention-magnitude saliency for foreground/background separation — derive a per-patch foreground probability from a lightweight conv-sigmoid head trained ONLY by a foreground-background contrastive loss (no pose/parsing labels), then suppress occluder/background tokens; this is an external-prior-free occluder gate, attractive because pose estimators fail under occlusion. (2) The importance-ranked cumulative fusion (FHFM): rank patches by CLS-attention*norm and let high-confidence (likely visible) patches contribute to every fusion level while low-importance (likely occluded) patches contribute least — a soft, label-free visibility weighting that downweights corrupted regions without a hard mask. The region-restricted masked attention with learnable part tokens is also a clean parsing-free way to get part features robust to missing regions.",
./reassess/r_3.md:8009:./reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
./reassess/r_3.md:8032:./reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
./reassess/r_3.md:8037:./reassess/r_3.md:4779:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:510:burstiness 死后调研 agent(带"in-domain 死、frozen 会骗人"教训)Rank-1 过审 bet。**为何逃过训练吸收(结构性)**: held-out (occluder类×部位)组合**按构造从没进训练集**→ 训练模型无可隐式吸收。
./reassess/r_3.md:8057:./pivot/p_2.md:700:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:8166:./pivot/p_2.md:6514:reviews/deep_18.md:6866:1343:  1343	protocol-3: (1) The imbalance in the quantity of data in the source domain increases the difficulty of
./reassess/r_3.md:8180:./pivot/p_2.md:6914:reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:8183:./pivot/p_2.md:6925:reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:8249:./pivot/p_2.md:7804:reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_3.md:8250:./pivot/p_2.md:7808:reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_3.md:8267:./pivot/p_2.md:7973:analyses/deep_25.txt:71:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:8289:./pivot/p_2.md:8155:analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:8298:./pivot/p_2.md:8427:./validate/clean/v_2.txt:16:- vs **probabilistic embedding / uncertainty ReID**：PCME 用概率分布处理图文一对多对应，DUL/ReID uncertainty 用后验/先验高斯建模样本不确定性；它们不是 cross-view 物理方向的包含关系。([arxiv.org](https://arxiv.org/abs/2101.05068?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2007.08785))
./reassess/r_3.md:8374:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:1576:- 🔵 蓝队（方案 B: PPE）核心论点: 重新定义遮挡问题（从 visibility mask 到 uncertainty quantification），pose confidence → feature uncertainty 的因果链是全新机制。论文理论深度更强（概率嵌入+信息几何）。与 PSG 正交互补（PSG 管特征提取，PPE 管匹配）。P3E/DUL 已验证概率嵌入在 ReID 中可训练。攻击 A: PAB 已在同范式失败（attention bias 变体），CLAUDE.md 警告"attention trick"，2.8K 参数太小难以撑论文主贡献，12x4 分辨率太粗糙。信心: 7/10
./reassess/r_3.md:8481: exited 1 in 0ms:
./reassess/r_3.md:8684:4. **证据闭环**：核心消融在 SYSU-MM01。baseline 是 72.25 mAP、75.22 rank-1；加 ATG 到 73.28、76.47；加 CTP 到 73.13、76.21；ATG+CTP 到 74.86、77.85；完整 TRM 到 76.21、78.52。这个表刚好证明三个判断：属性文本有用，轮廓文本也有用，两者互补，TRM 不是多余融合。作者还和传统 VI-ReID 以及语言模型方法对比，强调 richer textual prompts 能更好指导视觉编码器提取 modality-invariant feature。
./reassess/r_3.md:8690:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
./reassess/r_3.md:8692:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
./reassess/r_3.md:8696:4. **证据闭环**：组件消融直接对应两个重定义。CUHK-PEDES baseline 是 70.79 rank-1、64.69 mAP；SRDA 到 74.35、69.63；CSRM 到 72.02、68.13；完整 PSSR 到 75.71、73.34。ICFG-PEDES 从 61.30、37.68 到 65.91、48.11，RSTPReid 从 59.20、46.65 到 62.53、51.77。传统 re-ranking 在 CUHK 上 rank-1 反而从 74.35 降到 68.68，而 CSRM 到 75.71，这很好地证明“同模态 re-ranking 假设不适用于跨模态”。高伪影样本也能涨点，低伪影和高伪影一起最好，支撑“视觉真实不是核心，多样性才是核心”。
./reassess/r_3.md:8698:5. **reviewer 为什么买账**：这篇卖的是两个务实视角：训练时不迷信 diffusion 真实感，测试时不照搬同模态 re-ranking。机制都很简单，但和观察贴得紧，消融也正好打在 reviewer 会怀疑的点上。novelty 来源不是复杂模型，而是对 TBPS 任务目标的重新理解。
./reassess/r_3.md:8708:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:8746:analyses/deep_7.txt:47:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
./reassess/r_3.md:8747:analyses/deep_7.txt:59:4. 证据闭环：Fig.1 的 corruption t-SNE 是重定义的根证据。四种协议 Clean Eval、Corrupt Eval、Corrupt Query、Corrupt Gallery 证明作者没有只测一种有利场景。Market-1501 的 Corrupt Eval 相对 baseline 有大幅提升，mINP 从 18.48 到 32.96，mAP 从 56.36 到 69.74，Rank-1 从 75.86 到 85.68。Clean Eval 略降反而支撑了作者说法：对齐会牺牲部分干净域特定信息，但换来 corruption-invariant 表征。
./reassess/r_3.md:8777:analyses/deep_4.txt:71:4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。
./reassess/r_3.md:8796:analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
./reassess/r_3.md:8801:analyses/deep_25.txt:71:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:8825:analyses/deep_26.txt:23:4. 证据闭环：数据集 AT-USTC 本身证明旧数据集缺口，21 个月、RGB 和红外、多衣服、多季节支撑“anytime”成立。模型证据包括六场景分别评测、跨数据集泛化、MS-ReID、MoAE、HDW 消融，以及和其他 MoE 方法对比，证明不是单纯多头分类。
./reassess/r_3.md:8844:analyses/ALL_ANALYSES.txt:73:4. **证据闭环**：它用方法表、数据集表、参数量和 Rank-1 对比、benchmark/protocol 汇总、挑战清单和 3DGait 医疗任务 case study 来证明 SRID 不是零散技巧，而是有共同问题、共同评价和外溢应用的方向。
./reassess/r_3.md:8848:analyses/ALL_ANALYSES.txt:187:4. 证据闭环：主要证据是消融。加 VTS 后 CARGO 的 A-G 协议 rank1 从 45.00 到 50.63，mAP 从 42.05 到 43.39。head 数、token 数、插入位置的消融说明 token 不是越多越好，2 个 token、2 个 head、最后一层插入最好，这支持“少量关键 token 更有效”的说法。但它缺少更强的 token 可视化或失败案例分析，所以证据闭环比前几篇弱。
./reassess/r_3.md:8899: exited 1 in 0ms:
./reassess/r_3.md:9198:pivot/p_2.md:6914:reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:9201:pivot/p_2.md:6925:reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:9211:pivot/p_2.md:7804:reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_3.md:9212:pivot/p_2.md:7808:reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_3.md:9220:pivot/p_2.md:7973:analyses/deep_25.txt:71:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:9229:pivot/p_2.md:8155:analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:9251:validate/v_3.md:1889:./reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:9255:validate/v_3.md:1908:./reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:9263:validate/v_3.md:2947:../litreview2/analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:9264:validate/v_3.md:3005:../litreview2/analyses/ALL_ANALYSES.txt:1462:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:9608:reviews/deep_3.md:6764:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:9611:reviews/deep_3.md:6819:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:9663:reviews/deep_26.md:11996:introduced additional complexity due to posture variability, clothing diversity, and frequent occlusions between individuals, which negatively
./reassess/r_3.md:9666:analyses/deep_3.txt:46:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:9707:reviews/deep_21.md:1884:   344	information in the intermediate layer and adaptively suppressing the expression of occlusion information. Compared with
./reassess/r_3.md:9712:reviews/deep_21.md:2015:   475	in the VARA module, which enables adaptive suppression of occlusion information, this alone is insufficient for
./reassess/r_3.md:9764:reviews/deep_21.md:5288:   118	Fig. 16. The Rank-5 performance of our method under different occlusion
./reassess/r_3.md:9767:reviews/deep_21.md:5300:   130	pedestrian ranking results under heavy occlusion conditions
./reassess/r_3.md:9769:reviews/deep_21.md:5304:   134	occlusion issues relatively well, and the Rank-1 results are
./reassess/r_3.md:9845:analyses/deep_25.txt:71:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:9869:reviews/deep_25.md:6754:trees) and non-target pedestrian occlusion, which often lead to significant loss of body information and increase the complexity of person
./reassess/r_3.md:9879:reviews/deep_25.md:6776:Some methods [6,7] directly use the output of pose estimation models as local masks without adaptive adjustment according to occlusion
./reassess/r_3.md:9941:reviews/deep_25.md:7720:datasets, our TTPM achieves 73.4 %/63.4 % and 86.8 %/82.2 % Rank1/mAP performance, respectively. Speciﬁcally, compared to the occlusion augmentation method FED [30], which requires additional computation to simulate occlusion data, our proposed method still shows
./reassess/r_3.md:9962:reviews/deep_25.md:8142:(+8.3 %) on the Partial-REID dataset. The experimental results of Rank1 and Rank-3 validate that our TTPM also exhibits practical applicability
./reassess/r_3.md:10025:reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:10030:reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:10046:analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:10048:analyses/ALL_ANALYSES.txt:1462:2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
./reassess/r_3.md:10210:reviews/deep_4.md:806:adaptive occlusion-aware prompts for occluded ReID.
./reassess/r_3.md:10218:reviews/deep_4.md:1591:(index 4) yields the highest performance, with Rank-1 reaching 75.2% and mAP 65.5%. These results validate the effectiveness of occlusion-aware fine-grained prompts in guiding
./reassess/r_3.md:10314:reviews/deep_20.md:5857:    52	difficulty stems from dual occlusion patterns: (1) object-induced occlusion, where environmental elements (e.g., vehicles, buildings) obstruct
./reassess/r_3.md:10376:reviews/deep_19.md:5358:can extract fine-grained pedestrian features, they are unable to adaptively segment features based on occlusion
./reassess/r_3.md:10467:reviews/deep_1.md:7731:extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
./reassess/r_3.md:10541:reviews/deep_1.md:8563:It is observed that mAP and Rank-1 reach the best when the parame-                        In this paper, a semantic-guided occlusion simulation based local
./reassess/r_3.md:10546:reviews/deep_1.md:8600:lightweight and adaptive occlusion simulation strategies and domain                                    ceedings of the IEEE/CVF International Conference on Computer Vision, 2021,
./reassess/r_3.md:10572:reviews/deep_1.md:8767:127:extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
./reassess/r_3.md:10647:reviews/deep_1.md:9241:1374:lightweight and adaptive occlusion simulation strategies and domain
./reassess/r_3.md:10667:reviews/deep_2.md:4651:by adaptively calculating the occlusion weights for body parts.
./reassess/r_3.md:10674:reviews/deep_2.md:4758:occlusion-adaptive local features that accurately reflect the
./reassess/r_3.md:10676:reviews/deep_2.md:4767:obtains body regions that are adaptive to occlusion by
./reassess/r_3.md:10677:reviews/deep_2.md:4773:which can adaptively calculate occlusion weights. These
./reassess/r_3.md:10693:reviews/deep_2.md:5329:adaptively capture key information even in severe occlusion.
./reassess/r_3.md:10766:reviews/deep_14.md:3445:   803	can adaptively suppress interference from occlusion and facilitate the network in learning discriminative person features.
./reassess/r_3.md:10799:reviews/deep_14.md:5984:  1235	adaptive dynamic recovery of the occlusion part. Additionally,
./reassess/r_3.md:10819:reviews/deep_16.md:1352:1018:[12] C. Zhao, Z. Qu, X. Jiang, Y. Tu, X. Bai, Content-adaptive auto-occlusion network
./reassess/r_3.md:10821:reviews/deep_16.md:2471:/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|diffusion|inpainting|re-ranking|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
./reassess/r_3.md:10845:reviews/deep_16.md:2525:152:use adaptive position extraction with occlusion weighting, while
./reassess/r_3.md:10864:reviews/deep_16.md:2563:360:reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reassess/r_3.md:10875:reviews/deep_16.md:2626:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_3.md:10880:reviews/deep_16.md:2983:   360	reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reassess/r_3.md:10894:reviews/deep_16.md:5451:  1018	[12] C. Zhao, Z. Qu, X. Jiang, Y. Tu, X. Bai, Content-adaptive auto-occlusion network
./reassess/r_3.md:10903:reviews/deep_16.md:5918:360:reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
./reassess/r_3.md:10910:reviews/deep_16.md:6034:1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
./reassess/r_3.md:10912:reviews/deep_16.md:6096:2420:cases, the ViT-extracted features are passed through DDO filtering and local re-ranking under occlusion. An actionable
./reassess/r_3.md:10926:reviews/deep_16.md:6705:  2420	cases, the ViT-extracted features are passed through DDO filtering and local re-ranking under occlusion. An actionable
./reassess/r_3.md:10984:reviews/deep_23.md:675:KPR (ECCV 2024), ETND (TCSVT 2024) and TTPM (PR 2025), RMPSNet improves the rank-1 performance on
./reassess/r_3.md:11077:reviews/deep_23.md:1960:without incurring excessive redundancy. RMPSNet justifies its moderate increase in complexity by providing significant gains in occlusion
./reassess/r_3.md:11124:reviews/deep_23.md:6944:occlusion, viewpoint variations, and the inherent difficulty of
./reassess/r_3.md:11133:reviews/deep_6.md:3205:   673	environments, lacking real-world complexities such as occlusion, clothing changes, and other appearance variations. To address these challenges, we have developed a new V-I ReID dataset, CMG-P, which incorporates more complex scenarios. As illustrated in Table 1, CMG-P
./reassess/r_3.md:11156:analyses/deep_25.txt:71:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11169:analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11274:pivot/p_2.md:6914:reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11276:pivot/p_2.md:6925:reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11308:pivot/p_2.md:7973:analyses/deep_25.txt:71:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11310:pivot/p_2.md:8155:analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11336:validate/v_3.md:1889:./reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11338:validate/v_3.md:1908:./reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11342:validate/v_3.md:2947:../litreview2/analyses/ALL_ANALYSES.txt:1188:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11564:reviews/deep_25.md:6754:trees) and non-target pedestrian occlusion, which often lead to significant loss of body information and increase the complexity of person
./reassess/r_3.md:11591:reviews/deep_25.md:7720:datasets, our TTPM achieves 73.4 %/63.4 % and 86.8 %/82.2 % Rank1/mAP performance, respectively. Speciﬁcally, compared to the occlusion augmentation method FED [30], which requires additional computation to simulate occlusion data, our proposed method still shows
./reassess/r_3.md:11609:reviews/deep_25.md:8142:(+8.3 %) on the Partial-REID dataset. The experimental results of Rank1 and Rank-3 validate that our TTPM also exhibits practical applicability
./reassess/r_3.md:11631:reviews/deep_25.md:14101:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11633:reviews/deep_25.md:14180:4. **证据闭环**：消融很干净。Occluded-Duke 上 baseline 是 59.4 rank-1、49.7 mAP；加 MFE 到 62.2、52.4；加 PPM 到 67.3、56.1；MFE+PPM 到 68.9、57.0；完整 TTPM 到 73.4、63.4。Occluded-REID 上从 72.1、67.6 到 86.8、82.2。TAD 还拆 CEL 和 TEL，证明上下文和纹理都贡献。可视化上，PPM 热图在极端遮挡下更集中到目标有效区域；检索图展示非目标行人遮挡时 TAD 能帮忙找回；失败案例也说明当目标和遮挡者纹理高度相似时仍会失败，这反而让机制边界更可信。
./reassess/r_3.md:11647:reviews/deep_23.md:675:KPR (ECCV 2024), ETND (TCSVT 2024) and TTPM (PR 2025), RMPSNet improves the rank-1 performance on
./reassess/r_3.md:11653:reviews/deep_23.md:1639:Rank-1 accuracy and 1.6% in mAP. Compared to ProFD, which achieves
./reassess/r_3.md:11654:reviews/deep_23.md:1651:a Rank-1 accuracy of 93.2% and an mAP of 89.8%, surpassing ProFD by
./reassess/r_3.md:11764:- 泄漏存在性：`cos(f_M, f_D)-cos(f_T, f_D)`、donor pid 进入 top-k 的比例、donor rank 是否显著提前。
./reassess/r_3.md:11815:- 泄漏存在性：`cos(f_M, f_D)-cos(f_T, f_D)`、donor pid 进入 top-k 的比例、donor rank 是否显著提前。

web search: 
web search: CFPER Coarse-to-Fine Part-Experts Retrieval Efficient Person Re-identification query difficulty adaptive compute
web search: 
web search: "Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification"
web search: 
web search: "CFPER" person ReID
web search: 
web search: site:arxiv.org person re-identification early exit dynamic inference
web search: 
web search: https://arxiv.org/abs/2308.11900
web search: 
web search: "Coarse-to-Fine Part" "Efficient Person" "Re-identification"
web search: 
web search: "Coarse-to-Fine Part-Experts"
web search: 
web search: "coarse-to-fine" "part-experts"
web search: 
web search: CFPER Coarse Fine Part Experts Retrieval ReID
web search: 
web search: "DareNet" "Resource aware person re-identification"
web search: 
web search: https://arxiv.org/abs/1805.08805
codex
本地已有的论文清单里出现了 CFPER，但公开搜索暂时没命中对应标题；我会把它作为“未能公开取证”的风险项单列，同时重点核对公开可验证的强先例，比如 DaReNet/HashReID/Anytime ReID。
exec
/bin/zsh -lc "find . -maxdepth 5 -iname '*CFPER*' -o -iname '*Coarse*Fine*Part*'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
exec
/bin/zsh -lc "rg -n \"CFPER|Coarse-to-Fine Part-Experts|Part-Experts Retrieval\" all_papers.txt reviews analyses reassess pivot validate -g '*.md' -g '*.txt'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
all_papers.txt:27:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
pivot/p_3.md:1535:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
reassess/r_1.md:157:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
reassess/r_1.md:504:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
reviews/deep_8.md:162:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_19.md:163:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_19.md:364:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
validate/v_3.md:8366:    35	| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
reviews/deep_6.md:81:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_26.md:76:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_28.md:412:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
analyses/deep_5.txt:25:**CFPER**
reviews/deep_17.md:581:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reassess/r_3.md:3287:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_3.md:3569:区别：FLaN-Net 用语言分人、属性、遮挡物；CFPER 用二维部位专家。我们用三维人体表面和投影可见性定义匹配支持区域，重点解决俯视、低清、遮挡时二维部位顺序失真的问题。  
validate/v_2.md:193:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
validate/v_2.md:11610:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
validate/v_2.md:12068:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
validate/v_2.md:14263:    35	| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
reassess/r_2.md:169:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
reassess/r_2.md:3641:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
reviews/deep_12.md:599:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_9.md:74:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
pivot/p_2.md:4347:| 18 | 主流评测外的轴 | CFPER: 按 query 难度动态分配算力(效率当核心) |
analyses/ALL_ANALYSES.txt:1573:**CFPER**
reviews/deep_5.md:29:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_5.md:85:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_5.md:2110:/bin/zsh -lc "pdftotext -f 1 -l 8 'CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
reviews/deep_5.md:2114:CFPER: Coarse-to-Fine Part-Experts Retrieval for
reviews/deep_5.md:2243:Fig. 1. The framework of CFPER consists of two stages: coarse inference stage and fine inference stage. In the coarse stage, a global feature G and N patch
reviews/deep_5.md:2262:Coarse-to-Fine Part-Experts Retrieval model (CFPER).
reviews/deep_5.md:2274:• The proposed CFPER achieves competitive performance
reviews/deep_5.md:2278:Part-Experts Retrieval (CFPER) in detail. An overview of
reviews/deep_5.md:2279:CFPER is shown in Fig. 1.
reviews/deep_5.md:2514:(mAP) to evaluate the performance of ReID methods. To access the efficiency of our CFPER, we report FLOPs calculated
reviews/deep_5.md:2553:BETWEEN P ERFORMANCE AND E FFICIENCY OF CFPER.
reviews/deep_5.md:2603:CFPER (α = 1)
reviews/deep_5.md:2604:CFPER (α = 0.6)
reviews/deep_5.md:2748:our CFPER (α = 0.6) achieves comparable performance with
reviews/deep_5.md:2752:CFPER (α = 0.6) achieves the highest Rank-1 accuracy
reviews/deep_5.md:2755:CFPER achieves competitive mAP on both datasets. These
reviews/deep_5.md:2756:results validate CFPER’s ability to adaptively handle queries
reviews/deep_5.md:2759:In this section, we conduct ablation studies on OccludedDuke to analyze the effectiveness of components of CFPER.
reviews/deep_5.md:2803:dataset to demonstrate the effectiveness of our CFPER.
reviews/deep_5.md:2805:shows “easy” and “hard” queries identified by CFPER, validating the soundness of our early-exit strategy in the CFDR
reviews/deep_5.md:2837:In this paper, we propose a novel model CFPER to balance
reviews/deep_5.md:2840:difficulty and tailoring the learning of visible body part representations, CFPER achieves competitive performance while
reviews/deep_5.md:2841:significantly reducing computational costs. Extensive experiments demonstrate the effectiveness of CFPER.
reviews/deep_5.md:4191:CFPER 的完整方法和消融已经读完，它的关键不是“部件专家”本身，而是先用难度分流把效率问题改写成检索流程问题。Channel-aware 这篇文本较长，中间被输出截断，我会再针对第 3 节补读。
reviews/deep_5.md:8366:web search: "CFPER" "Coarse-to-Fine Part-Experts Retrieval"
reviews/deep_5.md:8368:web search: "Coarse-to-Fine Part-Experts Retrieval" "person re-identification"
reviews/deep_5.md:8396:**CFPER**
reviews/deep_5.md:8470:**CFPER**
reviews/deep_1.md:259:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_10.md:110:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_20.md:80:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_23.md:77:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_24.md:262:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_18.md:277:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_14.md:93:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_14.md:253:-rw-r--r--@   1 abdslm  staff   5743549 Jun  9 16:51 CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_21.md:87:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_11.md:162:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_4.md:288:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_25.md:208:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_3.md:203:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_15.md:173:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_16.md:821:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_22.md:245:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_27.md:70:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_7.md:173:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_7.md:242:./CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_2.md:108:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_13.md:74:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
reviews/deep_13.md:242:CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf

codex
我在本地审查记录里找到了 CFPER 的 PDF 抽取文本，可以直接核它的路由依据。接下来我会用本地 CFPER 片段 + 联网公开源交叉验证其它类别。
exec
/bin/zsh -lc "sed -n '2110,2860p' reviews/deep_5.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
/bin/zsh -lc "pdftotext -f 1 -l 8 'CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209843

CFPER: Coarse-to-Fine Part-Experts Retrieval for
Efficient Person Re-identification
1st Shiyu Wang

2nd Mingming Lu⋆

School of Computer Science and Engineering
Central South University
ChangSha, China
224711075@csu.edu.cn

School of Computer Science and Engineering
Central South University
ChangSha, China
mingminglu@csu.edu.cn

Abstract—Most existing person re-identification (ReID) methods focus on improving retrieval accuracy by refining features,
which fails to balance accuracy with inference efficiency. We
observe that query difficulty varies: global features suffice for
simple cases, while fine-grained part features are required for
challenging cases, such as occlusion. However, current methods
typically use the same feature extraction network for all queries,
which may limit accuracy on difficult queries or waste computational resources on easier ones. To address this, we propose
a two-stage Coarse-to-Fine Dynamic Retrieval mechanism that
adaptively allocates resources based on query difficulty. For
“easy” queries, only global features are used in the coarse
stage, and inference terminates early. For “hard” queries, part
features are extracted in the fine stage for detailed matching.
To further reduce computational costs, we introduce Mixture
of Experts for part feature extraction, where a router assigns
patches to part experts using topology annotations, and only activates body-relevant experts, enabling accurate part identification
with significant computation reduction. Extensive experiments
demonstrate that our method achieves competitive performance
while significantly reducing computational costs compared to
state-of-the-art methods.
Index Terms—Transformer-based Person Re-identification,
Computational Efficiency, Early-Exit, Mixture of Experts.

I. I NTRODUCTION
Person re-identification (ReID) aims to retrieve a specific
person from a large database of person images captured by
diverse non-overlapping cameras, which is widely applied in
many domains, such as criminal investigation and smart city
[1]. Most existing person ReID methods primarily focus on
improving retrieval accuracy, often overlooking the importance
of computational efficiency. Specifically, these methods [1]–
[7] process all queries with the same network and use uniform
features for retrieval, without accounting for the varying
difficulty of queries, which may limit accuracy on challenging
queries or consume needless computational resources on easier
ones. In real-world applications, the retrieval difficulty of different queries varies: some query images exhibit distinct discriminative features, and global features alone are sufficient for
accurate retrieval. For these “easy” queries, excessive reliance
on fine-grained part features may lead to false matches, as
different pedestrians may share similar body part appearances.
In contrast, “hard” queries, such as those affected by occlusion,
pose variation, or subtle inter-class variation, require finegrained part features to support more detailed matching.

TABLE I
P ERFORMANCE AND FLOP S OF T RANS R E ID WITH D IFFERENT
R ETRIEVAL F EATURES DURING I NFERENCE . “G” D ENOTES THE C ASE
USING ONLY THE G LOBAL F EATURE . “G+P” D ENOTES THE C ASE USING
THE C ONCATENATED F EATURE OF G LOBAL F EATURE AND F OUR PART
F EATURES .

Dataset
Feature
mAP (%)
Rank-1 (%)
FLOPs (G)

Market-1501
G
G+P
87.1
88.2
94.6
95.0
11.35
12.29

DukeMTMC
G
G+P
79.6
80.6
89.0
89.6
11.35
12.29

Occluded-Duke
G
G+P
53.8
55.7
61.6
64.2
11.35
12.29

To verify this, we conducted experiments on three datasets
with increasing retrieval difficulty: Market-1501 (holistic) [8],
DukeMTMC (holistic) [9], and Occluded-DukeMTMC (occluded) [2]. We tested two TransReID [4] configurations: (1)
global features, and (2) concatenated global and four part
features, and report mAP, Rank-1 and FLOPs in Table I. The
results show that introducing part features increases computational cost by 1.1× in terms of FLOPs, but provides only minor
improvements in mAP (1.1% and 1.0%) and Rank-1 (0.4% and
0.6%) on two holistic datasets, while resulting in significant
improvements (1.9% mAP, 3.1% Rank-1) on the more challenging occluded dataset. This indicates that global features
are generally sufficient to support accurate matching for “easy”
queries with distinctive visual features, while fine-grained part
features are more beneficial for challenging queries. Inspired
by the above observations, we propose a two-stage Coarse-toFine Dynamic Retrieval (CFDR) mechanism that adaptively
allocates computational resources based on the difficulty of
queries to balance performance and efficiency. Specifically, an
early-exit threshold is computed to evaluate query difficulty.
For “easy” queries, only global features are used for fast
retrieval in the coarse stage, and inference terminates early to
avoid unnecessary computational resource consumption. Only
“hard” queries proceed to the fine stage, where fine-grained
part features are extracted for more precise matching.
For part feature extraction in the fine inference stage of
CFDR, existing methods face several limitations. Some methods [5] rely on external models (e.g., pose estimation and
human parsing) to extract part features, introducing additional
computational costs. Other methods [4], [10] divide spatially
adjacent patches or pixels into fixed-size groups to learn

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

Fig. 1. The framework of CFPER consists of two stages: coarse inference stage and fine inference stage. In the coarse stage, a global feature G and N patch
embeddings Z are extracted. An early-exit decision is made based on the threshold ⌈N α⌉: if met, the query is considered “easy”, only global feature is used
for retrieval and inference terminates. Otherwise, the query is considered “hard” and proceeds to the fine stage to extract part features using the Part-aware
Mixture of Experts (PMoE). PMoE includes a Patch-to-Part Router (PPR) and M Part Experts (PE). Each patch is routed to the corresponding body part via
PPR, and the fine-grained part feature ei of each body part is learned by the dedicated PE. Here, PWAP refers to Probability Weighted Average Pooling.

part features but ignore human topology priors, leading to
inaccurate part detection. Therefore, to further reduce computational costs and improve part identification accuracy, we
introduce Mixture of Experts and human topology priors to
propose a Part-aware Mixture of Experts (PMoE). The PMoE
consists of a Patch-to-Part Router (PPR) and a set of Part
Experts (PE). The PPR routes patches to corresponding body
parts based on identity labels and human topology annotations, enabling accurate part identification without introducing
additional computational costs. Each body part is assigned
to a dedicated Part Expert for fine-grained feature learning.
And during inference, PMoE only activates body-relevant
experts based on binary visibility routing weights to reduce
computational costs without hurting the performance.
Finally, we combine CFDR and PMoE to construct our
Coarse-to-Fine Part-Experts Retrieval model (CFPER).
The main contributions of our work can be summarized as
follows:
• We propose a novel Coarse-to-Fine dynamic retrieval
mechanism that balances computational efficiency and retrieval accuracy by adaptively allocating resources based
on the query difficulty.
• We innovatively apply MoE in part feature extraction,
which uses human topology annotations to guide the
router for accurate part identification without extra computational costs, and only body-relevant experts are ac-

tivated during inference, significantly reducing inference
costs without sacrificing performance.
• The proposed CFPER achieves competitive performance
compared to SOTA methods on both holistic and occluded person ReID datasets, while also improving inference efficiency.
II. M ETHODOLOGY
In this section, we introduce the proposed Coarse-to-Fine
Part-Experts Retrieval (CFPER) in detail. An overview of
CFPER is shown in Fig. 1.
A. Coarse Inference Stage
Coarse Feature Extractor. We use the pre-trained Vision
Transformer (ViT) [11] as our feature extractor. Given a person
image X, the output of the encoder can be divided into two
parts: a global feature G ∈ R1×D and N patch embeddings
Z = [z1 , . . . , zN ] ∈ RN ×D .
Early-exit strategy. In the coarse inference stage, we
introduce an early-exit strategy to balance performance and
efficiency. Traditional early-exit strategies [12] are typically
used in classification tasks, relying on classifier scores. However, in person ReID, the training and testing identities are
inconsistent and the number of testing identities is unknown,
making the classifier score-based exit metric unsuitable. To
address this, we propose an early-exit strategy for person

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

Fig. 2. Illustration of the early-exit strategy based on the first-order differences
of similarity scores.

ReID, which determines whether to exit early by evaluating
the query difficulty based on the first-order differences of
similarity scores.
To assess query difficulty, our intuition is to count the
number of body-related patches. As shown in Fig. 2, we
first compute the cosine similarity between global feature
G and N patch embeddings Z, obtaining similarity scores
S = [s1 , . . . , sN ]. The scores S are sorted in descending
D
order to obtain S D = [sD
1 , . . . , sN ]. We then compute the
first-order differences D = [D1 , . . . , DN −1 ], where Di =
D
sD
i − si+1 . We assume that body and background/occlusion
features are separate classes in the feature space, so there
should be a noticeable feature transition when a body feature
shifts to a background/occlusion feature. Therefore, we select
the maximum value in D and use its corresponding index
η = arg maxi (D) as the split point between the body and
background/occlusion regions. Patches with the top-η similarity scores are considered as body regions, while the rest
are considered as background/occlusion regions. Finally we
introduce an early-exit threshold α ∈ [0, 1]. If η ≥ ⌈N α⌉, the
query is considered “easy” with sufficient visible body regions
for high-confidence retrieval. In this case, only global features
G are used for retrieval, and the whole inference terminates.
Otherwise, the query is considered “hard” and proceeds to the
fine inference stage to extract fine-grained part features.
Supervision Loss of Coarse Inference Stage. We adopt
cross-entropy loss as ID loss to supervise the learning of
encoder:
LCOARSE = LID (G),
(1)
B. Fine Inference Stage
In the fine inference stage, we employ PMoE to extract finegrained part features for “hard” queries to support detailed
retrieval. PMoE consists of a Patch-to-Part Router and a set
of Part Experts.
1) Patch-to-Part Router. The PPR module learns to route
patches embeddings Z to M+1 parts C = [c0 , c1 , . . . , cM ],
where c0 represents the background, and [c1 , . . . , cM ] corresponds to M body parts. PPR adopts a standard MoE router
structure, consisting of a fully-connected layer with parameters
θ ∈ R(M +1)×D followed by a softmax layer, which produces
the probabilities P ∈ RN ×(M +1) of each patch belonging to
the background and M body parts:
P = {pi ∈ RN | i = 0, 1, ..., M } = Sof tmax(Z · θT ), (2)

where pi represents the probabilities of patches embeddings
Z belonging to part ci .
Next, the router needs to assign patches to each expert based
on P. Traditional MoE patch-level routers [13] commonly
use the Top-k mechanism, where each expert is assigned k
patches (k ≪ N ). However, this fixed Top-k mechanism
is unsuitable for person ReID due to the varying sizes of
body parts. If k is too small, larger body regions (e.g., the
torso) may not be fully covered, leading to information loss.
Conversely, if k is too large, smaller body parts (e.g., the
head) may be polluted by surrounding background/occlusion
patches, introducing noise. To address this, our PPR employs
a soft routing mechanism. First, we aggregate M body parts
probabilities [p1 , . . . , pM ] to compute a foreground probability
pf ∈ RN : pjf = sum(pj1 , . . . , pjM ), where pji represents the
probability of patch embedding zj belonging to body part
ci . We then perform Probability Weighted Average Pooling
(PWAP) on patch embeddings Z to obtain a foreground
feature rf , a background feature r0 and M body part features
[r1 , . . . , rM ]:
PN
j
j=1 zj · pi
(3)
ri = PN j , ∀i ∈ {f, 0, 1, ..., M }
j=1 pi
where zj denotes the j-th patch embedding from the coarse
inference stage.
Supervision Loss of PPR. We introduce human topology
priors in the form of coarse body part labels. For each patch
j ∈ RN , its part label yj ∈ {0, 1, ..., M }. The cross-entropy
loss Lh with label smoothing is computed as:
Lh = −

M
X N
X

qi · log(pji ),

i=0 j=1

with qi =

(
1 − B−1
B ε
ε
B

(4)
if yj = i
otherwise

where B is batch size, ε is label smoothing regularization rate.
Additionally, we propose a Push Loss Lpush to separate the
body regions from background/occlusion noise:
E
D
t
t
B
r
,
r
X
0 f
1
,
(5)
Lpush =
B t=1 ||r0t || · ||rft ||
t
where r0|f
is the background/foreground feature of the t-th
image in a batch.
Finally, we follow the Global-identity Local-triplet (GiLt)
[7] strategy to supervise the training of PPR:

LP P R = λh Lh + Lpush + Lce (rf ) + Lce (rc )
+ Lpart
tri (r1 , . . . , rM ),

(6)

where rc = Concat(r1 , . . . , rM ), Lce is cross-entrophy loss,
Lpart
tri is part-averaged triplet loss [7].
2) Part Experts. Although Transformer excels at capturing
global information, it is less effective at capturing fine-grained
image details. To compensate for this limitation, we assign a

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

III. E XPERIMENTS
A. Datasets and Evaluation Metrics

Fig. 3. The structure of Part Expert Module.

dedicated part-expert module to each body part, enabling deep
learning of its distinctive discriminative features. Futhermore,
we use pedestrian foreground features rf as coarse-grained
cues to provide contextual support for part feature learning.
The structure of PE is shown in Fig. 3. We first implement
1D convolution on body part features ri to obtain r̃i . Then
we apply the multi-head cross-attention (MHCA) mechanism,
where the query matrix Qi is derived from r̃i , and the key
matrix K and value matrix V are derived from the foreground
features rf . The MHCA is computed as:
MHCA(Qi , K, V ) = Concat(head1 , . . . , headh )W O , (7)
headl = Attention(Qli , K l , V l ),

p 
Attention(Q, K, V ) = Sof tmax QK T / dk V ,

(8)
(9)

where headl is the l-th head output, W O is the output transformation matrix for integrating multi-head outputs. Finally,
following layer normalization and two fully-connected layers,
we can obtain the new part feature ei of body part ci .
Additionally, to further reduce inference computational
costs and address occlusion effectively, PE is activated
based on body part visibility during inference, as shown in
Fig. 1. We generate binary visibility routing weights W =
[w1 , . . . , wM ] ∈ RM from the probabilities P. For body part
ci , if at least one patch in pi has a probability greater than
the threshold λ, wi = 1, the corresponding PE is activated;
otherwise, wi = 0, PE remains inactive:
(
1, if max(pni ) > λ
n
(10)
wi =
0, otherwise.
Supervision Loss of PE. We adopt ID loss and part-average
triplet loss to supervise PE:
LP E =

M
X

LID (ei ) + Lpart
tri (e1 , . . . , eM ),

(11)

i=1

C. Loss Functions
In the training stage, we set the early-exit threshold α = 1,
ensuring all images pass through both coarse and fine inference
stages for joint training. In the fine inference stage, all PE
are activated to ensure comprehensive training of each expert
module. The total training loss L is calculated as:
L = LCOARSE + LF IN E
LF IN E = LP P R + LP E

(12)

We evaluate our method on three ReID benchmarks, including one occluded and two holistic person ReID benchmarks.
Occluded-Duke [2] consists of 15,618 training images
from 702 identities, 2,210 occluded query images from 519
identities, and 17,661 gallery images from 1,110 identities.
Market-1501 [8] contains 36,036 images from 1,501 identities captured from 6 camera viewpoints.
DukeMTMC-ReID [9] comprises 36,411 images from
1,404 identities captured from 8 camera viewpoints.
Evaluation Metrics. We adopt the Cumulative Matching
Characteristic (CMC) curve and mean Average Precision
(mAP) to evaluate the performance of ReID methods. To access the efficiency of our CFPER, we report FLOPs calculated
with the fvcore toolkit.
B. Implementation Details
We adopt the ViT-Base [11] as the backbone. Both training
and testing images are resized to 256×128. The training images
are augmented with random cropping, padding, and random
erasing. The batch size is set to 64 with 4 images per ID.
The hidden dimension D is set to 768. The SGD optimizer
is employed with a momentum of 0.9 and a weight decay
of 1e-4. The learning rate is initialized at 0.008 with cosine
learning rate decay. The number of body parts M is set to
13. The threshold λ is empirically set to 0.4. We train our
model for 300 epochs. For a fair comparison, no re-ranking
techniques are used during inference.
C. Analysis of Trade-Off between Performance and Efficiency
The early-exit threshold α is an important factor that
balances model performance and inference efficiency in our
methods. The CFDR uses the early-exit threshold α to assess
the query difficulty and control the number of queries proceed
to the fine inference stage. A larger α imposes stricter criteria
for identifying queries as “easy”, leading to more queries
entering the fine stage. We conduct experiments with different
α on Occluded-Duke, using Rank-1 and mAP as performance
metrics and FLOPs in the fine inference stage as efficiency
metric. The results are shown in Table II. As α decreases,
FLOPs decrease due to fewer queries entering the fine stage,
demonstrating significant computational savings. Rank-1 accuracy initially increases, indicating that global features can
prevent mismatches since not all part features are discriminative, but then decreases as more “hard” queries requiring
fine-grained part features are missed. The decrease in mAP
shows that fine-grained part features are crucial for detailed
matching. The above results indicate that our CFDR enables
adaptive allocation of computational resources between “easy”
and “hard” samples, reducing unnecessary computation while
maintaining performance.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

TABLE II
T HE I MPACT OF E ARLY-E XIT T HRESHOLD α ON THE T RADE - OFF
BETWEEN P ERFORMANCE AND E FFICIENCY OF CFPER.
α
1
0.7
0.6
0.5

Rank-1 (%)
67.8
68.5
68.5
65.9

mAP (%)
58.6
58.4
58.4
57.5

FLOPs (G)
179.05
153.93 (↓14%)
121.53 (↓32%)
102.49 (↓42%)

TABLE III
C OMPARISON WITH SOTA M ETHODS ON O CCLUDED (O CCLUDED -D UKE )
AND H OLISTIC (M ARKET-1501 AND D UKE MTMC) R E ID DATASETS .
T HE S YMBOL † D ENOTES THE M ETHODS INTRODUCING E XTERNAL
M ODELS AND ∗ D ENOTES H YBRID C NN -T RANSFORMER METHODS .
Backbone

CNN

Transformer

Methods
PGFA† (ICCV19) [2]
HOReID† (CVPR20) [3]
IGOAS (TIP21) [14]
BPBReID (WACV23) [7]
RTGAT (TIP23) [15]
GPEOG† (ICME23) [16]
PAT∗ (CVPR21) [17]
TransReID (ICCV21) [4]
FED (CVPR22) [10]
PFD† (AAAI22) [5]
DRL-Net∗ (TMM23) [6]
SCAT (TII23) [18]
SPT (AAAI24) [19]
CFPER (α = 1)
CFPER (α = 0.6)

Occluded-Duke
Rank-1
mAP
51.4
37.3
55.1
43.8
60.1
49.4
66.7
54.1
61.0
50.1
64.1
51.2
64.5
53.6
64.2
55.7
68.1
56.4
67.7
60.1
65.0
50.8
62.8
54.9
68.6
57.4
67.8
58.6
68.5
58.4

Market-1501
Rank-1
mAP
91.2
76.8
94.2
84.9
93.4
84.1
95.1
87.0
93.3
85.1
94.8
87.5
95.4
88.0
95.0
88.2
95.0
86.3
95.5
89.6
94.7
86.9
95.1
88.0
94.5
86.2
95.1
88.7
95.6
88.4

DukeMTMC
Rank-1 mAP
82.6
65.5
86.9
75.6
86.9
75.1
89.6
78.3
88.0
76.9
87.5
75.5
88.8
78.2
89.6
80.6
89.4
78.0
90.6
82.2
88.1
76.6
89.3
79.8
89.4
79.1
90.3
80.7
90.5
80.2

TABLE IV
A BLATION S TUDY ON O CCLUDED -D UKE .
Index
1
2
3
4

PPR
✓
✓
✓

PE
✓
✓

CFDR
✓

Rank-1
60.5
63
67.8
68.5

mAP
53.1
55.4
58.6
58.4

D. Comparison with the State-of-the-Art Methods
We compare our method with SOTA methods on three
benchmarks, covering both occluded and holistic person ReID
scenarios. The comparison includes two kinds of methods:
CNN-based and Transformer-based. Notably, PGFA, HOReID,
GPEOG and PFD incorporate external models for part feature
extraction, while PAT and DRL-Net integrate CNNs within
their Transformer architectures. The comparison results are
shown in Table III. On the challenging Occluded-Duke dataset,
our CFPER (α = 0.6) achieves comparable performance with
a Rank-1 accuracy of 68.5% and mAP of 58.4%, outperforming most of the compared methods on both metrics without
relying on external models. This demonstrates its robustness
in handling occluded scenarios. On the holistic ReID datasets,
CFPER (α = 0.6) achieves the highest Rank-1 accuracy
on Market-1501 and the second-highest Rank-1 accuracy on
DukeMTMC, falling behind PFD by only 0.1%. Additionally,
CFPER achieves competitive mAP on both datasets. These
results validate CFPER’s ability to adaptively handle queries
with varying difficulty, ensuring both efficiency and accuracy.
E. Ablation Study
In this section, we conduct ablation studies on OccludedDuke to analyze the effectiveness of components of CFPER.

TABLE V
C OMPARISON OF C OMPUTATION B ETWEEN PPR AND EXTERNAL MODELS .
Model
PPR(ours)
OpenPose
HRNet32

FLOPs (G)
0.0016
37.80
17.86

Parameters (M )
0.0123
25.94
41.23

Effectiveness of proposed Components. The results are
shown in Table IV. Index-1 represents the baseline model,
which is a standard ViT. Index-2 shows that PPR provides
+2.5% Rank-1 accuracy and +2.3% mAP improvements,
demonstrating the effectiveness of combining human topology priors and identity labels for part localization. Index-3
shows that PE further improve performance by +4.8% Rank-1
accuracy and +3.2% mAP, indicating the benefit of specialized
networks for fine-grained feature learning. Comparing Index-3
and Index-4, the CFDR mechanism optimizes computational
efficiency while maintaining performance.
Comparison of Computation between PPR and external
models. We compare the FLOPs and model parameters of our
PPR with external models commonly used in person ReID for
part feature extraction, including HRNet [20] and OpenPose
[21]. As shown in Table V, PPR significantly outperforms both
OpenPose and HRNet32 in computational efficiency, with only
0.0016G FLOPs, dramatically lower than OpenPose (37.80G)
and HRNet32 (17.86G), demonstrating PPR’s substantial advantage in reducing computational cost. Additionally, PPR has
just 0.0123M parameters, greatly reducing model complexity
compared to OpenPose (25.94M) and HRNet32 (41.23M).
These results demonstrate that our PPR can effectively identify
key body parts using a simple network without adding extra
computation, highlighting its efficiency and practicality.
F. Visualization
In this section, we perform visualizations on Occluded-Duke
dataset to demonstrate the effectiveness of our CFPER.
Visualization of the “Easy” and “Hard” query. Fig. 4
shows “easy” and “hard” queries identified by CFPER, validating the soundness of our early-exit strategy in the CFDR
mechanism. “Easy” queries retain visible discriminative features despite partial occlusion, while “hard” queries involve
severe occlusion and fewer discriminative features, making it
difficult to distinguish the target pedestrian from others with
similar appearance. Fig. 5 shows the Top-10 retrieval results
for “easy” and “hard” queries in the coarse and fine inference
stages. For “easy” queries, global features alone are sufficient
for accurate matching in the coarse stage. In contrast, “hard”
queries benefit from fine-grained part features in the fine stage,
improving retrieval accuracy by compensating for occlusion
noise in global features.
Visualization of Body Part Attention Maps for PPR.
Fig. 6 shows the body part localization results of PPR.
Our method accurately identifies body parts and effectively
addresses occlusions and background noise. Furthermore, the
Part Experts for occluded parts remain inactive, allowing the
model to focus on visible discriminative features while further
reducing computational costs during inference.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

Fig. 4. Visualization of the “easy” and “hard” query images.

Fig. 5. Top-10 Retrieval results of “easy” and “hard” queries. Green/red
borders illustrate correct/false matches. For each query, the first row displays
coarse inference results, the second row shows fine inference results.

Fig. 6. Visualization of body part attention maps for PPR. The second column
denotes foreground attention maps. Green/red border indicates active/inactive
PE during inference.

IV. C ONCLUSION
In this paper, we propose a novel model CFPER to balance
inference efficiency and retrieval accuracy for person ReID.
By dynamically adjusting the retrieval process based on query
difficulty and tailoring the learning of visible body part representations, CFPER achieves competitive performance while
significantly reducing computational costs. Extensive experiments demonstrate the effectiveness of CFPER.
R EFERENCES
[1] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang, “Beyond
part models: Person retrieval with refined part pooling (and a strong
convolutional baseline),” in Proceedings of the European conference on
computer vision (ECCV), 2018, pp. 480–496.
[2] Jiaxu Miao, Yu Wu, Ping Liu, Yuhang Ding, and Yi Yang, “Pose-guided
feature alignment for occluded person re-identification,” in Proceedings
of the IEEE/CVF international conference on computer vision, 2019,
pp. 542–551.

[3] Guan’an Wang, Shuo Yang, Huanyu Liu, Zhicheng Wang, Yang Yang,
Shuliang Wang, Gang Yu, Erjin Zhou, and Jian Sun, “High-order
information matters: Learning relation and topology for occluded person
re-identification,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2020, pp. 6449–6458.
[4] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang,
“Transreid: Transformer-based object re-identification,” in Proceedings
of the IEEE/CVF international conference on computer vision, 2021,
pp. 15013–15022.

exec
/bin/zsh -lc "sed -n '1560,1595p' analyses/ALL_ANALYSES.txt" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
exec
/bin/zsh -lc "sed -n '8390,8495p' reviews/deep_5.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
5. 审稿人为什么买账：这篇卖的主要是视角，其次才是蒸馏机制。它的真实新意来自“联邦 ReID 不应该默认一个全局同构模型”这个部署假设，而不是某个 ReID 特征提取模块。对我们有用的点是，先把现实约束讲成旧范式不成立，再让机制变成补救这个范式缺口的必然选择。

**CCUP**
1. 触发观察：作者注意到换衣 ReID 的核心瓶颈不是模型不够复杂，而是真实数据太少，尤其是同一个身份在大量服装变化下的标注样本太少。PRCC、LTCC 等数据集规模和每人服装数都有限，模型很容易过拟合衣服。

2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。

3. 机制怎么长出来：如果缺的是同一身份多服装数据，那机制自然就是用 3D 人体、服装资产、纹理替换和虚拟相机生成可控数据。裸人体网格定义身份，服装和纹理定义换衣变化，Unreal Engine 模拟监控场景，检测器自动裁框和生成标签。后面的预训练加微调只是让这个大规模合成分布进入普通模型。

4. 证据闭环：最关键的证据是同一模型在 CCUP、UnrealPerson、PersonX、ClonedPerson 等不同合成数据上预训练后的对比。如果只是“多一点数据有用”，通用合成数据也应该同样有效；如果 CCUP 更好，才说明“换衣预训练分布”这个重定义成立。Grad-CAM 可视化进一步说明预训练后模型更关注脸、脖子、肩、手腕、鞋等相对少变的区域，而不是背景或衣服。

5. 审稿人为什么买账：这篇卖的是数据和任务分布视角，不是模型机制。新意来自“换衣 ReID 缺的不是又一个去衣服损失，而是可控的大规模换衣监督”。对我们有用的套路是，发现任务缺少某个关键变化轴，然后自己构造这个轴，并用跨数据集预训练收益证明它不是普通增广。

**CFPER**
1. 触发观察：作者先做了一个很实用的观察，global feature 加 part feature 在 Market、Duke 这种整体行人数据上只带来很小收益，却增加计算量；但在 Occluded-Duke 上收益明显更大。也就是说，查询样本难度不同，统一走细粒度匹配会浪费简单样本，也会让困难样本得不到足够处理。

2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。

3. 机制怎么长出来：先用 ViT 得到全局特征和 patch 特征，用全局特征与 patch 特征的相似度排序，再用一阶差分估计可见人体区域比例。可见人体足够多就判为简单样本，只用全局特征提前退出；否则进入细阶段，用拓扑监督的 patch-to-part router 和 part experts 提取细粒度部件特征。这个机制和重定义绑定很紧，难度分流直接决定是否启用细粒度计算。

4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。

5. 审稿人为什么买账：这篇卖的是视角，部件专家只是服务于视角。它把效率从附属指标变成方法的核心问题，这比单纯说“我又做了一个 part module”更容易成立。对我们有用的是，找一个主流评价以外但真实存在的轴，比如计算、更新、标注、部署，再用一个很小的观察表把问题立住。

**Channel-aware feature mining network**
1. 触发观察：作者的观察是 VI-ReID 中 RGB 和红外不仅有整体模态差异，还有通道层面的不平衡。有些通道包含衣服纹理、轮廓、热分布等身份线索，有些通道贡献很弱甚至引入噪声。现有方法多把通道操作当预处理或数据增强，没有显式挖掘通道关系。

2. 重定义动作：它把 VI-ReID 的模态差距问题改写成“通道级身份信息没有被充分建模”。关键词是 channel-aware、channel-level feature optimization、channel-level feature refinement、channel imbalance、identity-relevant channels。

3. 机制怎么长出来：机制是三段式。CLFO 在早期用深度可分离卷积、归一化、SE 和可学习残差做通道筛选；CLFR 再用通道注意力、非局部融合、空间注意力抑制噪声；MDFO 在高低层特征之间做通道和空间维度的联合优化。这个机制和重定义的绑定中等偏弱，因为一旦说“通道重要”，很多通用注意力堆叠都可以被解释成通道挖掘。

4. 证据闭环：它的闭环主要靠通道激活可视化和模块消融。可视化说明不同通道确实响应不同身份区域；CLFO、CLFR、MDFO 逐步加入带来提升；子模块消融进一步说明各部件有增益。不过它更像“模块有效性证明”，对“通道重定义一定正确”的因果证明不够强，例如缺少更硬的参数匹配注意力对照或随机通道扰动对照。

5. 审稿人为什么买账：这篇主要卖机制和结果，视角相对普通。它的真实新意是把 VI-ReID 里已有的通道增强思路推进到特征学习内部，并用较完整实验支撑 SOTA。对我们来说，这是一个警示样本：如果观察不够尖，机制就容易变成通用模块堆叠，投稿时会更依赖结果强度。

**CLIP-Based Camera-Agnostic Feature Learning**

 succeeded in 0ms:
3. 机制怎么长出来：如果缺的是同一身份多服装数据，那机制自然就是用 3D 人体、服装资产、纹理替换和虚拟相机生成可控数据。裸人体网格定义身份，服装和纹理定义换衣变化，Unreal Engine 模拟监控场景，检测器自动裁框和生成标签。后面的预训练加微调只是让这个大规模合成分布进入普通模型。

4. 证据闭环：最关键的证据是同一模型在 CCUP、UnrealPerson、PersonX、ClonedPerson 等不同合成数据上预训练后的对比。如果只是“多一点数据有用”，通用合成数据也应该同样有效；如果 CCUP 更好，才说明“换衣预训练分布”这个重定义成立。Grad-CAM 可视化进一步说明预训练后模型更关注脸、脖子、肩、手腕、鞋等相对少变的区域，而不是背景或衣服。

5. 审稿人为什么买账：这篇卖的是数据和任务分布视角，不是模型机制。新意来自“换衣 ReID 缺的不是又一个去衣服损失，而是可控的大规模换衣监督”。对我们有用的套路是，发现任务缺少某个关键变化轴，然后自己构造这个轴，并用跨数据集预训练收益证明它不是普通增广。

**CFPER**
1. 触发观察：作者先做了一个很实用的观察，global feature 加 part feature 在 Market、Duke 这种整体行人数据上只带来很小收益，却增加计算量；但在 Occluded-Duke 上收益明显更大。也就是说，查询样本难度不同，统一走细粒度匹配会浪费简单样本，也会让困难样本得不到足够处理。

2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。

3. 机制怎么长出来：先用 ViT 得到全局特征和 patch 特征，用全局特征与 patch 特征的相似度排序，再用一阶差分估计可见人体区域比例。可见人体足够多就判为简单样本，只用全局特征提前退出；否则进入细阶段，用拓扑监督的 patch-to-part router 和 part experts 提取细粒度部件特征。这个机制和重定义绑定很紧，难度分流直接决定是否启用细粒度计算。

4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。

5. 审稿人为什么买账：这篇卖的是视角，部件专家只是服务于视角。它把效率从附属指标变成方法的核心问题，这比单纯说“我又做了一个 part module”更容易成立。对我们有用的是，找一个主流评价以外但真实存在的轴，比如计算、更新、标注、部署，再用一个很小的观察表把问题立住。

**Channel-aware feature mining network**
1. 触发观察：作者的观察是 VI-ReID 中 RGB 和红外不仅有整体模态差异，还有通道层面的不平衡。有些通道包含衣服纹理、轮廓、热分布等身份线索，有些通道贡献很弱甚至引入噪声。现有方法多把通道操作当预处理或数据增强，没有显式挖掘通道关系。

2. 重定义动作：它把 VI-ReID 的模态差距问题改写成“通道级身份信息没有被充分建模”。关键词是 channel-aware、channel-level feature optimization、channel-level feature refinement、channel imbalance、identity-relevant channels。

3. 机制怎么长出来：机制是三段式。CLFO 在早期用深度可分离卷积、归一化、SE 和可学习残差做通道筛选；CLFR 再用通道注意力、非局部融合、空间注意力抑制噪声；MDFO 在高低层特征之间做通道和空间维度的联合优化。这个机制和重定义的绑定中等偏弱，因为一旦说“通道重要”，很多通用注意力堆叠都可以被解释成通道挖掘。

4. 证据闭环：它的闭环主要靠通道激活可视化和模块消融。可视化说明不同通道确实响应不同身份区域；CLFO、CLFR、MDFO 逐步加入带来提升；子模块消融进一步说明各部件有增益。不过它更像“模块有效性证明”，对“通道重定义一定正确”的因果证明不够强，例如缺少更硬的参数匹配注意力对照或随机通道扰动对照。

5. 审稿人为什么买账：这篇主要卖机制和结果，视角相对普通。它的真实新意是把 VI-ReID 里已有的通道增强思路推进到特征学习内部，并用较完整实验支撑 SOTA。对我们来说，这是一个警示样本：如果观察不够尖，机制就容易变成通用模块堆叠，投稿时会更依赖结果强度。

**CLIP-Based Camera-Agnostic Feature Learning**
1. 触发观察：ICS ReID 只在每个相机内部标身份，不给跨相机身份对应。已有方法先做相机内学习，再用相似度或聚类做跨相机关联，但伪标签容易被视角、背景、光照和相机风格污染。CLIP 可以给语义监督，但直接用于跨相机阶段也会受噪声影响。

2. 重定义动作：它把 ICS ReID 从“低标注 ReID 的伪标签问题”改写成“如何利用相机内标签学习 camera-agnostic 特征”。关键词是 camera-agnostic feature learning、intra-camera discriminative learning、inter-camera adversarial learning、prompt learning、semantic supervision。

3. 机制怎么长出来：第一阶段用 CLIP 和可学习 prompt 给每个相机内身份生成隐式文本描述。第二阶段用相机内混合记忆库存身份中心特征和实例特征，同时拉近同身份中心和困难正样本，推开困难负样本。第三阶段做跨相机关联和原型对比，再用 ICAL 惩罚模型区分同一伪身份在不同相机下的全局 ID。机制和重定义绑定很紧，因为每一步都在回答“相机内标签能提供什么，跨相机噪声怎么压”。

4. 证据闭环：最能支撑重定义的不是最终 mAP，而是 ICDL、ICAL、CLIP prompt 的消融，以及全局分类器概率分布可视化。图里从单峰变多峰的现象说明，同一伪身份在不同相机下被拉到一起，模型的相机可分性被削弱。若伪标签质量或跨相机关联准确率也提升，闭环会更完整。

5. 审稿人为什么买账：这篇卖的是视角加协议适配。它不是简单说“CLIP 用在 ReID”，而是把 CLIP 放进 ICS 的特殊标签结构里，用剩余标签解决缺失标签带来的跨相机偏差。对我们有用的套路是，先精确定义“哪些监督缺失，哪些监督还在”，然后让机制围绕剩余监督自然展开。

**CLIP-driven fine-grained mining**
1. 触发观察：作者注意到 CLIP 在文本行人检索里全局对齐很强，但难以区分细粒度人物差异。更关键的是，简单加局部对齐并不一定有效，甚至会伤害 CLIP，因为 CLIP 预训练是实例级图文对齐，随机初始化的局部解码器和粗暴切块会破坏预训练空间。

2. 重定义动作：它把问题从“给 CLIP 加 fine-grained alignment”改写成“在不引入外部工具、不破坏 CLIP 预训练空间的前提下挖细粒度关系”。关键词是 pre-training space compatibility、explicit fine-grained mining、attention bias forward、text extraction strategy、visual bridge、momentum distillation。

3. 机制怎么长出来：ABF 不直接切最后层特征，而是在 ViT 中后层复制 CLS 作为局部 token，并用 attention bias 限制每个局部 token 只关注对应 patch 组，同时保留全局路径。FEL 用共享可学习 token 和多模态解码器从文本中聚合细粒度语义。TES 用局部视觉表示监督 FEL，把 CLIP 空间里的余弦关系和欧氏距离关系转移给细粒度文本表示。MoD 和零初始化用于避免随机局部 token 一开始学噪声。这个机制和重定义绑定非常紧。

4. 证据闭环：这篇的消融很有说服力。普通细粒度 baseline 比 global baseline 还差，证明“直接加局部会伤”这个观察成立；加入 ABF 仍不够，加入 TES 和 MoD 后才稳定提升，说明关键不是局部本身，而是和 CLIP 预训练空间兼容。ABF 对 PCB 的对比也证明 ViT 不能照搬 CNN 的水平切块。

5. 审稿人为什么买账：这篇卖的是诊断视角。它不是声称“我有一个更强 local module”，而是先证明常识动作会失败，再提出兼容预训练空间的约束和机制。对我们最有启发的是，好的创新可以来自“旧办法为什么在新基础模型上不再成立”的反直觉证据。

**共同套路**
这批论文的共同写法是，先找一个具体、可测的失败现象，再把它上升成旧范式的盲点：统一客户端、有限换衣数据、统一查询预算、忽略通道、相机内标签孤岛、粗暴局部 CLIP。真正强的几篇不是模块复杂，而是机制几乎从重定义里自然长出来，并且用消融证明“这个新问题定义是对的”，而不是只证明“加了模块涨点”。

补查来源主要用于核对先例背景：FedUCC/FedReID 相关联邦 ReID 先例见 [Robust Knowledge Adaptation for Federated Unsupervised Person ReID](https://arxiv.org/abs/2301.07320) 和 [FedReID benchmark](https://arxiv.org/abs/2008.11560)；CCUP 与 CCAFL 条目见 [CCUP arXiv](https://arxiv.org/abs/2410.13567) 和 [CCAFL arXiv](https://arxiv.org/abs/2409.19563)。
tokens used
145,807
你列的是 6 篇，不是 5 篇。我按 6 篇都拆了。

**CCFL**
1. 触发观察：作者抓住的不是“联邦 ReID 精度不够”，而是 ReID 的联邦场景天然比普通联邦学习更异质。不同机构的数据来自不同地点、时间、设备和规模，数据非独立同分布很重，而且客户端算力也不同。统一模型、统一训练策略在这种场景下反而不合理。

2. 重定义动作：他们把问题从“怎么在隐私约束下聚合多个 ReID 数据集”改写成“不同客户端应该被允许使用不同训练策略和不同模型结构”。关键词是 customized client、heterogeneous model、interaction model、non-IID、limited server。这个重定义很重要，因为它把 FedAvg 失效从技术限制变成了问题本身。

3. 机制怎么长出来：既然客户端可定制，就不能直接平均本地模型参数，所以自然需要一个结构统一的交互模型作为知识交换媒介。大数据客户端用 DBSCAN 和较大模型，小数据客户端用层次聚类和较小模型，本地模型把知识蒸馏给交互模型，交互模型在服务器聚合后再反向指导本地模型。这个机制和问题定义绑定比较紧，交互模型是异构客户端设定下的自然产物。

4. 证据闭环：关键不是只报 mAP，而是比较统一策略、统一模型、多策略、多模型的差别。它用 8 个 ReID 数据集模拟真实客户端异质性，并和 FedUReID、FedUCC 等联邦 ReID 方法比较。真正支撑重定义的是“多策略、多模型版本优于简化联邦版本”，说明定制客户端确实不是装饰。

5. 审稿人为什么买账：这篇卖的主要是视角，其次才是蒸馏机制。它的真实新意来自“联邦 ReID 不应该默认一个全局同构模型”这个部署假设，而不是某个 ReID 特征提取模块。对我们有用的点是，先把现实约束讲成旧范式不成立，再让机制变成补救这个范式缺口的必然选择。

**CCUP**
1. 触发观察：作者注意到换衣 ReID 的核心瓶颈不是模型不够复杂，而是真实数据太少，尤其是同一个身份在大量服装变化下的标注样本太少。PRCC、LTCC 等数据集规模和每人服装数都有限，模型很容易过拟合衣服。

2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。

3. 机制怎么长出来：如果缺的是同一身份多服装数据，那机制自然就是用 3D 人体、服装资产、纹理替换和虚拟相机生成可控数据。裸人体网格定义身份，服装和纹理定义换衣变化，Unreal Engine 模拟监控场景，检测器自动裁框和生成标签。后面的预训练加微调只是让这个大规模合成分布进入普通模型。

4. 证据闭环：最关键的证据是同一模型在 CCUP、UnrealPerson、PersonX、ClonedPerson 等不同合成数据上预训练后的对比。如果只是“多一点数据有用”，通用合成数据也应该同样有效；如果 CCUP 更好，才说明“换衣预训练分布”这个重定义成立。Grad-CAM 可视化进一步说明预训练后模型更关注脸、脖子、肩、手腕、鞋等相对少变的区域，而不是背景或衣服。

5. 审稿人为什么买账：这篇卖的是数据和任务分布视角，不是模型机制。新意来自“换衣 ReID 缺的不是又一个去衣服损失，而是可控的大规模换衣监督”。对我们有用的套路是，发现任务缺少某个关键变化轴，然后自己构造这个轴，并用跨数据集预训练收益证明它不是普通增广。

**CFPER**
1. 触发观察：作者先做了一个很实用的观察，global feature 加 part feature 在 Market、Duke 这种整体行人数据上只带来很小收益，却增加计算量；但在 Occluded-Duke 上收益明显更大。也就是说，查询样本难度不同，统一走细粒度匹配会浪费简单样本，也会让困难样本得不到足够处理。

2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。

3. 机制怎么长出来：先用 ViT 得到全局特征和 patch 特征，用全局特征与 patch 特征的相似度排序，再用一阶差分估计可见人体区域比例。可见人体足够多就判为简单样本，只用全局特征提前退出；否则进入细阶段，用拓扑监督的 patch-to-part router 和 part experts 提取细粒度部件特征。这个机制和重定义绑定很紧，难度分流直接决定是否启用细粒度计算。

4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。

5. 审稿人为什么买账：这篇卖的是视角，部件专家只是服务于视角。它把效率从附属指标变成方法的核心问题，这比单纯说“我又做了一个 part module”更容易成立。对我们有用的是，找一个主流评价以外但真实存在的轴，比如计算、更新、标注、部署，再用一个很小的观察表把问题立住。

**Channel-aware feature mining network**
1. 触发观察：作者的观察是 VI-ReID 中 RGB 和红外不仅有整体模态差异，还有通道层面的不平衡。有些通道包含衣服纹理、轮廓、热分布等身份线索，有些通道贡献很弱甚至引入噪声。现有方法多把通道操作当预处理或数据增强，没有显式挖掘通道关系。

2. 重定义动作：它把 VI-ReID 的模态差距问题改写成“通道级身份信息没有被充分建模”。关键词是 channel-aware、channel-level feature optimization、channel-level feature refinement、channel imbalance、identity-relevant channels。

3. 机制怎么长出来：机制是三段式。CLFO 在早期用深度可分离卷积、归一化、SE 和可学习残差做通道筛选；CLFR 再用通道注意力、非局部融合、空间注意力抑制噪声；MDFO 在高低层特征之间做通道和空间维度的联合优化。这个机制和重定义的绑定中等偏弱，因为一旦说“通道重要”，很多通用注意力堆叠都可以被解释成通道挖掘。

4. 证据闭环：它的闭环主要靠通道激活可视化和模块消融。可视化说明不同通道确实响应不同身份区域；CLFO、CLFR、MDFO 逐步加入带来提升；子模块消融进一步说明各部件有增益。不过它更像“模块有效性证明”，对“通道重定义一定正确”的因果证明不够强，例如缺少更硬的参数匹配注意力对照或随机通道扰动对照。

5. 审稿人为什么买账：这篇主要卖机制和结果，视角相对普通。它的真实新意是把 VI-ReID 里已有的通道增强思路推进到特征学习内部，并用较完整实验支撑 SOTA。对我们来说，这是一个警示样本：如果观察不够尖，机制就容易变成通用模块堆叠，投稿时会更依赖结果强度。

**CLIP-Based Camera-Agnostic Feature Learning**
1. 触发观察：ICS ReID 只在每个相机内部标身份，不给跨相机身份对应。已有方法先做相机内学习，再用相似度或聚类做跨相机关联，但伪标签容易被视角、背景、光照和相机风格污染。CLIP 可以给语义监督，但直接用于跨相机阶段也会受噪声影响。

2. 重定义动作：它把 ICS ReID 从“低标注 ReID 的伪标签问题”改写成“如何利用相机内标签学习 camera-agnostic 特征”。关键词是 camera-agnostic feature learning、intra-camera discriminative learning、inter-camera adversarial learning、prompt learning、semantic supervision。

web search: 
web search: "Towards Anytime Retrieval" "Anytime Person Re-Identification"
web search: 
web search: "dynamic inference" "person re-identification" "FLOPs"
web search: 
web search: "HashReID" "adaptive computation"
web search: 
web search: https://arxiv.org/abs/2308.11900
web search: 
web search: https://arxiv.org/pdf/2308.11900
web search: 
web search: https://arxiv.org/pdf/2308.11900
web search: 
web search: https://arxiv.org/pdf/1805.08805v3
web search: 
web search: "Resource Aware Person Re-identification across Multiple Resolutions" PDF
web search: 
web search: https://github.com/mileyan/DARENet
web search: 
web search: https://raw.githubusercontent.com/mileyan/DARENet/master/budgeted_stream/README.md
web search: 
web search: https://github.com/mileyan/DARENet
web search: 
web search: "DARENet" "budgeted_stream" "simulation"
web search: 
web search: https://github.com/mileyan/DARENet/tree/master/budgeted_stream
web search: 
web search: https://api.github.com/repos/mileyan/DARENet/contents/budgeted_stream?ref=master
web search: 
web search: "10.1109/ICME59968.2025.11209843"
web search: 
web search: https://doi.org/10.1109/ICME59968.2025.11209843
web search: 
web search: "coarse-to-fine" "person re-identification" "retrieval" "cascade"
web search: 
web search: "coarse-to-fine" "person re-identification"
web search: 
web search: https://arxiv.org/abs/2008.06826
web search: 
web search: "adaptive reranking" retrieval budget query difficulty
web search: 
web search: "query-adaptive" "reranking" retrieval
web search: 
web search: https://arxiv.org/abs/2505.18512
web search: 
web search: "uncertainty" "person re-identification" "hard" "query"
web search: 
web search: "Uncertainty" "Person Re-identification" "CVPR" "ReID"
web search: 
web search: https://arxiv.org/abs/2012.08733
web search: 
web search: "hard sample mining" "person re-identification" "adaptive"
web search: 
web search: https://arxiv.org/abs/2507.11119
web search: 
web search: "Dynamic Token Selective Transformer" "Aerial-Ground Person Re-Identification"
web search: 
web search: https://arxiv.org/abs/2412.00433
web search: 
web search: https://arxiv.org/abs/2509.16635
web search: 
web search: "rank instability" "retrieval"
web search: 
web search: "cheap" "full" "ranking" "retrieval" "disagreement"
web search: 
web search: "ranking consistency" "early exit" retrieval
web search: 
web search: "rank volatility" retrieval
web search: 
web search: "rank uncertainty" "adaptive" retrieval
web search: 
web search: https://ieeexplore.ieee.org/document/11209843
web search: 
web search: "11209843" "CFPER"
web search: 
web search: "CFPER"
codex
CFPER 的关键结论已经很清楚：它已经占了“query difficulty adaptive compute / early exit for ReID”这块地，但它的难度信号是全局特征与 patch 特征的一阶差分估计可见人体比例，不是 cheap-full top-k 排名不一致。下面我继续补通用 reranking 和 rank uncertainty 类先例，判断这个差异能不能单独站住。
exec
/bin/zsh -lc "nl -ba reviews/deep_5.md | sed -n '2114,2295p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
  2114	CFPER: Coarse-to-Fine Part-Experts Retrieval for
  2115	Efficient Person Re-identification
  2116	1st Shiyu Wang
  2117	
  2118	2nd Mingming Lu⋆
  2119	
  2120	School of Computer Science and Engineering
  2121	Central South University
  2122	ChangSha, China
  2123	224711075@csu.edu.cn
  2124	
  2125	School of Computer Science and Engineering
  2126	Central South University
  2127	ChangSha, China
  2128	mingminglu@csu.edu.cn
  2129	
  2130	Abstract—Most existing person re-identification (ReID) methods focus on improving retrieval accuracy by refining features,
  2131	which fails to balance accuracy with inference efficiency. We
  2132	observe that query difficulty varies: global features suffice for
  2133	simple cases, while fine-grained part features are required for
  2134	challenging cases, such as occlusion. However, current methods
  2135	typically use the same feature extraction network for all queries,
  2136	which may limit accuracy on difficult queries or waste computational resources on easier ones. To address this, we propose
  2137	a two-stage Coarse-to-Fine Dynamic Retrieval mechanism that
  2138	adaptively allocates resources based on query difficulty. For
  2139	“easy” queries, only global features are used in the coarse
  2140	stage, and inference terminates early. For “hard” queries, part
  2141	features are extracted in the fine stage for detailed matching.
  2142	To further reduce computational costs, we introduce Mixture
  2143	of Experts for part feature extraction, where a router assigns
  2144	patches to part experts using topology annotations, and only activates body-relevant experts, enabling accurate part identification
  2145	with significant computation reduction. Extensive experiments
  2146	demonstrate that our method achieves competitive performance
  2147	while significantly reducing computational costs compared to
  2148	state-of-the-art methods.
  2149	Index Terms—Transformer-based Person Re-identification,
  2150	Computational Efficiency, Early-Exit, Mixture of Experts.
  2151	
  2152	I. I NTRODUCTION
  2153	Person re-identification (ReID) aims to retrieve a specific
  2154	person from a large database of person images captured by
  2155	diverse non-overlapping cameras, which is widely applied in
  2156	many domains, such as criminal investigation and smart city
  2157	[1]. Most existing person ReID methods primarily focus on
  2158	improving retrieval accuracy, often overlooking the importance
  2159	of computational efficiency. Specifically, these methods [1]–
  2160	[7] process all queries with the same network and use uniform
  2161	features for retrieval, without accounting for the varying
  2162	difficulty of queries, which may limit accuracy on challenging
  2163	queries or consume needless computational resources on easier
  2164	ones. In real-world applications, the retrieval difficulty of different queries varies: some query images exhibit distinct discriminative features, and global features alone are sufficient for
  2165	accurate retrieval. For these “easy” queries, excessive reliance
  2166	on fine-grained part features may lead to false matches, as
  2167	different pedestrians may share similar body part appearances.
  2168	In contrast, “hard” queries, such as those affected by occlusion,
  2169	pose variation, or subtle inter-class variation, require finegrained part features to support more detailed matching.
  2170	
  2171	TABLE I
  2172	P ERFORMANCE AND FLOP S OF T RANS R E ID WITH D IFFERENT
  2173	R ETRIEVAL F EATURES DURING I NFERENCE . “G” D ENOTES THE C ASE
  2174	USING ONLY THE G LOBAL F EATURE . “G+P” D ENOTES THE C ASE USING
  2175	THE C ONCATENATED F EATURE OF G LOBAL F EATURE AND F OUR PART
  2176	F EATURES .
  2177	
  2178	Dataset
  2179	Feature
  2180	mAP (%)
  2181	Rank-1 (%)
  2182	FLOPs (G)
  2183	
  2184	Market-1501
  2185	G
  2186	G+P
  2187	87.1
  2188	88.2
  2189	94.6
  2190	95.0
  2191	11.35
  2192	12.29
  2193	
  2194	DukeMTMC
  2195	G
  2196	G+P
  2197	79.6
  2198	80.6
  2199	89.0
  2200	89.6
  2201	11.35
  2202	12.29
  2203	
  2204	Occluded-Duke
  2205	G
  2206	G+P
  2207	53.8
  2208	55.7
  2209	61.6
  2210	64.2
  2211	11.35
  2212	12.29
  2213	
  2214	To verify this, we conducted experiments on three datasets
  2215	with increasing retrieval difficulty: Market-1501 (holistic) [8],
  2216	DukeMTMC (holistic) [9], and Occluded-DukeMTMC (occluded) [2]. We tested two TransReID [4] configurations: (1)
  2217	global features, and (2) concatenated global and four part
  2218	features, and report mAP, Rank-1 and FLOPs in Table I. The
  2219	results show that introducing part features increases computational cost by 1.1× in terms of FLOPs, but provides only minor
  2220	improvements in mAP (1.1% and 1.0%) and Rank-1 (0.4% and
  2221	0.6%) on two holistic datasets, while resulting in significant
  2222	improvements (1.9% mAP, 3.1% Rank-1) on the more challenging occluded dataset. This indicates that global features
  2223	are generally sufficient to support accurate matching for “easy”
  2224	queries with distinctive visual features, while fine-grained part
  2225	features are more beneficial for challenging queries. Inspired
  2226	by the above observations, we propose a two-stage Coarse-toFine Dynamic Retrieval (CFDR) mechanism that adaptively
  2227	allocates computational resources based on the difficulty of
  2228	queries to balance performance and efficiency. Specifically, an
  2229	early-exit threshold is computed to evaluate query difficulty.
  2230	For “easy” queries, only global features are used for fast
  2231	retrieval in the coarse stage, and inference terminates early to
  2232	avoid unnecessary computational resource consumption. Only
  2233	“hard” queries proceed to the fine stage, where fine-grained
  2234	part features are extracted for more precise matching.
  2235	For part feature extraction in the fine inference stage of
  2236	CFDR, existing methods face several limitations. Some methods [5] rely on external models (e.g., pose estimation and
  2237	human parsing) to extract part features, introducing additional
  2238	computational costs. Other methods [4], [10] divide spatially
  2239	adjacent patches or pixels into fixed-size groups to learn
  2240	
  2241	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.
  2242	
  2243	Fig. 1. The framework of CFPER consists of two stages: coarse inference stage and fine inference stage. In the coarse stage, a global feature G and N patch
  2244	embeddings Z are extracted. An early-exit decision is made based on the threshold ⌈N α⌉: if met, the query is considered “easy”, only global feature is used
  2245	for retrieval and inference terminates. Otherwise, the query is considered “hard” and proceeds to the fine stage to extract part features using the Part-aware
  2246	Mixture of Experts (PMoE). PMoE includes a Patch-to-Part Router (PPR) and M Part Experts (PE). Each patch is routed to the corresponding body part via
  2247	PPR, and the fine-grained part feature ei of each body part is learned by the dedicated PE. Here, PWAP refers to Probability Weighted Average Pooling.
  2248	
  2249	part features but ignore human topology priors, leading to
  2250	inaccurate part detection. Therefore, to further reduce computational costs and improve part identification accuracy, we
  2251	introduce Mixture of Experts and human topology priors to
  2252	propose a Part-aware Mixture of Experts (PMoE). The PMoE
  2253	consists of a Patch-to-Part Router (PPR) and a set of Part
  2254	Experts (PE). The PPR routes patches to corresponding body
  2255	parts based on identity labels and human topology annotations, enabling accurate part identification without introducing
  2256	additional computational costs. Each body part is assigned
  2257	to a dedicated Part Expert for fine-grained feature learning.
  2258	And during inference, PMoE only activates body-relevant
  2259	experts based on binary visibility routing weights to reduce
  2260	computational costs without hurting the performance.
  2261	Finally, we combine CFDR and PMoE to construct our
  2262	Coarse-to-Fine Part-Experts Retrieval model (CFPER).
  2263	The main contributions of our work can be summarized as
  2264	follows:
  2265	• We propose a novel Coarse-to-Fine dynamic retrieval
  2266	mechanism that balances computational efficiency and retrieval accuracy by adaptively allocating resources based
  2267	on the query difficulty.
  2268	• We innovatively apply MoE in part feature extraction,
  2269	which uses human topology annotations to guide the
  2270	router for accurate part identification without extra computational costs, and only body-relevant experts are ac-
  2271	
  2272	tivated during inference, significantly reducing inference
  2273	costs without sacrificing performance.
  2274	• The proposed CFPER achieves competitive performance
  2275	compared to SOTA methods on both holistic and occluded person ReID datasets, while also improving inference efficiency.
  2276	II. M ETHODOLOGY
  2277	In this section, we introduce the proposed Coarse-to-Fine
  2278	Part-Experts Retrieval (CFPER) in detail. An overview of
  2279	CFPER is shown in Fig. 1.
  2280	A. Coarse Inference Stage
  2281	Coarse Feature Extractor. We use the pre-trained Vision
  2282	Transformer (ViT) [11] as our feature extractor. Given a person
  2283	image X, the output of the encoder can be divided into two
  2284	parts: a global feature G ∈ R1×D and N patch embeddings
  2285	Z = [z1 , . . . , zN ] ∈ RN ×D .
  2286	Early-exit strategy. In the coarse inference stage, we
  2287	introduce an early-exit strategy to balance performance and
  2288	efficiency. Traditional early-exit strategies [12] are typically
  2289	used in classification tasks, relying on classifier scores. However, in person ReID, the training and testing identities are
  2290	inconsistent and the number of testing identities is unknown,
  2291	making the classifier score-based exit metric unsuitable. To
  2292	address this, we propose an early-exit strategy for person
  2293	
  2294	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.
  2295	

exec
/bin/zsh -lc "nl -ba reviews/deep_5.md | sed -n '2514,2860p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
  2514	(mAP) to evaluate the performance of ReID methods. To access the efficiency of our CFPER, we report FLOPs calculated
  2515	with the fvcore toolkit.
  2516	B. Implementation Details
  2517	We adopt the ViT-Base [11] as the backbone. Both training
  2518	and testing images are resized to 256×128. The training images
  2519	are augmented with random cropping, padding, and random
  2520	erasing. The batch size is set to 64 with 4 images per ID.
  2521	The hidden dimension D is set to 768. The SGD optimizer
  2522	is employed with a momentum of 0.9 and a weight decay
  2523	of 1e-4. The learning rate is initialized at 0.008 with cosine
  2524	learning rate decay. The number of body parts M is set to
  2525	13. The threshold λ is empirically set to 0.4. We train our
  2526	model for 300 epochs. For a fair comparison, no re-ranking
  2527	techniques are used during inference.
  2528	C. Analysis of Trade-Off between Performance and Efficiency
  2529	The early-exit threshold α is an important factor that
  2530	balances model performance and inference efficiency in our
  2531	methods. The CFDR uses the early-exit threshold α to assess
  2532	the query difficulty and control the number of queries proceed
  2533	to the fine inference stage. A larger α imposes stricter criteria
  2534	for identifying queries as “easy”, leading to more queries
  2535	entering the fine stage. We conduct experiments with different
  2536	α on Occluded-Duke, using Rank-1 and mAP as performance
  2537	metrics and FLOPs in the fine inference stage as efficiency
  2538	metric. The results are shown in Table II. As α decreases,
  2539	FLOPs decrease due to fewer queries entering the fine stage,
  2540	demonstrating significant computational savings. Rank-1 accuracy initially increases, indicating that global features can
  2541	prevent mismatches since not all part features are discriminative, but then decreases as more “hard” queries requiring
  2542	fine-grained part features are missed. The decrease in mAP
  2543	shows that fine-grained part features are crucial for detailed
  2544	matching. The above results indicate that our CFDR enables
  2545	adaptive allocation of computational resources between “easy”
  2546	and “hard” samples, reducing unnecessary computation while
  2547	maintaining performance.
  2548	
  2549	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.
  2550	
  2551	TABLE II
  2552	T HE I MPACT OF E ARLY-E XIT T HRESHOLD α ON THE T RADE - OFF
  2553	BETWEEN P ERFORMANCE AND E FFICIENCY OF CFPER.
  2554	α
  2555	1
  2556	0.7
  2557	0.6
  2558	0.5
  2559	
  2560	Rank-1 (%)
  2561	67.8
  2562	68.5
  2563	68.5
  2564	65.9
  2565	
  2566	mAP (%)
  2567	58.6
  2568	58.4
  2569	58.4
  2570	57.5
  2571	
  2572	FLOPs (G)
  2573	179.05
  2574	153.93 (↓14%)
  2575	121.53 (↓32%)
  2576	102.49 (↓42%)
  2577	
  2578	TABLE III
  2579	C OMPARISON WITH SOTA M ETHODS ON O CCLUDED (O CCLUDED -D UKE )
  2580	AND H OLISTIC (M ARKET-1501 AND D UKE MTMC) R E ID DATASETS .
  2581	T HE S YMBOL † D ENOTES THE M ETHODS INTRODUCING E XTERNAL
  2582	M ODELS AND ∗ D ENOTES H YBRID C NN -T RANSFORMER METHODS .
  2583	Backbone
  2584	
  2585	CNN
  2586	
  2587	Transformer
  2588	
  2589	Methods
  2590	PGFA† (ICCV19) [2]
  2591	HOReID† (CVPR20) [3]
  2592	IGOAS (TIP21) [14]
  2593	BPBReID (WACV23) [7]
  2594	RTGAT (TIP23) [15]
  2595	GPEOG† (ICME23) [16]
  2596	PAT∗ (CVPR21) [17]
  2597	TransReID (ICCV21) [4]
  2598	FED (CVPR22) [10]
  2599	PFD† (AAAI22) [5]
  2600	DRL-Net∗ (TMM23) [6]
  2601	SCAT (TII23) [18]
  2602	SPT (AAAI24) [19]
  2603	CFPER (α = 1)
  2604	CFPER (α = 0.6)
  2605	
  2606	Occluded-Duke
  2607	Rank-1
  2608	mAP
  2609	51.4
  2610	37.3
  2611	55.1
  2612	43.8
  2613	60.1
  2614	49.4
  2615	66.7
  2616	54.1
  2617	61.0
  2618	50.1
  2619	64.1
  2620	51.2
  2621	64.5
  2622	53.6
  2623	64.2
  2624	55.7
  2625	68.1
  2626	56.4
  2627	67.7
  2628	60.1
  2629	65.0
  2630	50.8
  2631	62.8
  2632	54.9
  2633	68.6
  2634	57.4
  2635	67.8
  2636	58.6
  2637	68.5
  2638	58.4
  2639	
  2640	Market-1501
  2641	Rank-1
  2642	mAP
  2643	91.2
  2644	76.8
  2645	94.2
  2646	84.9
  2647	93.4
  2648	84.1
  2649	95.1
  2650	87.0
  2651	93.3
  2652	85.1
  2653	94.8
  2654	87.5
  2655	95.4
  2656	88.0
  2657	95.0
  2658	88.2
  2659	95.0
  2660	86.3
  2661	95.5
  2662	89.6
  2663	94.7
  2664	86.9
  2665	95.1
  2666	88.0
  2667	94.5
  2668	86.2
  2669	95.1
  2670	88.7
  2671	95.6
  2672	88.4
  2673	
  2674	DukeMTMC
  2675	Rank-1 mAP
  2676	82.6
  2677	65.5
  2678	86.9
  2679	75.6
  2680	86.9
  2681	75.1
  2682	89.6
  2683	78.3
  2684	88.0
  2685	76.9
  2686	87.5
  2687	75.5
  2688	88.8
  2689	78.2
  2690	89.6
  2691	80.6
  2692	89.4
  2693	78.0
  2694	90.6
  2695	82.2
  2696	88.1
  2697	76.6
  2698	89.3
  2699	79.8
  2700	89.4
  2701	79.1
  2702	90.3
  2703	80.7
  2704	90.5
  2705	80.2
  2706	
  2707	TABLE IV
  2708	A BLATION S TUDY ON O CCLUDED -D UKE .
  2709	Index
  2710	1
  2711	2
  2712	3
  2713	4
  2714	
  2715	PPR
  2716	✓
  2717	✓
  2718	✓
  2719	
  2720	PE
  2721	✓
  2722	✓
  2723	
  2724	CFDR
  2725	✓
  2726	
  2727	Rank-1
  2728	60.5
  2729	63
  2730	67.8
  2731	68.5
  2732	
  2733	mAP
  2734	53.1
  2735	55.4
  2736	58.6
  2737	58.4
  2738	
  2739	D. Comparison with the State-of-the-Art Methods
  2740	We compare our method with SOTA methods on three
  2741	benchmarks, covering both occluded and holistic person ReID
  2742	scenarios. The comparison includes two kinds of methods:
  2743	CNN-based and Transformer-based. Notably, PGFA, HOReID,
  2744	GPEOG and PFD incorporate external models for part feature
  2745	extraction, while PAT and DRL-Net integrate CNNs within
  2746	their Transformer architectures. The comparison results are
  2747	shown in Table III. On the challenging Occluded-Duke dataset,
  2748	our CFPER (α = 0.6) achieves comparable performance with
  2749	a Rank-1 accuracy of 68.5% and mAP of 58.4%, outperforming most of the compared methods on both metrics without
  2750	relying on external models. This demonstrates its robustness
  2751	in handling occluded scenarios. On the holistic ReID datasets,
  2752	CFPER (α = 0.6) achieves the highest Rank-1 accuracy
  2753	on Market-1501 and the second-highest Rank-1 accuracy on
  2754	DukeMTMC, falling behind PFD by only 0.1%. Additionally,
  2755	CFPER achieves competitive mAP on both datasets. These
  2756	results validate CFPER’s ability to adaptively handle queries
  2757	with varying difficulty, ensuring both efficiency and accuracy.
  2758	E. Ablation Study
  2759	In this section, we conduct ablation studies on OccludedDuke to analyze the effectiveness of components of CFPER.
  2760	
  2761	TABLE V
  2762	C OMPARISON OF C OMPUTATION B ETWEEN PPR AND EXTERNAL MODELS .
  2763	Model
  2764	PPR(ours)
  2765	OpenPose
  2766	HRNet32
  2767	
  2768	FLOPs (G)
  2769	0.0016
  2770	37.80
  2771	17.86
  2772	
  2773	Parameters (M )
  2774	0.0123
  2775	25.94
  2776	41.23
  2777	
  2778	Effectiveness of proposed Components. The results are
  2779	shown in Table IV. Index-1 represents the baseline model,
  2780	which is a standard ViT. Index-2 shows that PPR provides
  2781	+2.5% Rank-1 accuracy and +2.3% mAP improvements,
  2782	demonstrating the effectiveness of combining human topology priors and identity labels for part localization. Index-3
  2783	shows that PE further improve performance by +4.8% Rank-1
  2784	accuracy and +3.2% mAP, indicating the benefit of specialized
  2785	networks for fine-grained feature learning. Comparing Index-3
  2786	and Index-4, the CFDR mechanism optimizes computational
  2787	efficiency while maintaining performance.
  2788	Comparison of Computation between PPR and external
  2789	models. We compare the FLOPs and model parameters of our
  2790	PPR with external models commonly used in person ReID for
  2791	part feature extraction, including HRNet [20] and OpenPose
  2792	[21]. As shown in Table V, PPR significantly outperforms both
  2793	OpenPose and HRNet32 in computational efficiency, with only
  2794	0.0016G FLOPs, dramatically lower than OpenPose (37.80G)
  2795	and HRNet32 (17.86G), demonstrating PPR’s substantial advantage in reducing computational cost. Additionally, PPR has
  2796	just 0.0123M parameters, greatly reducing model complexity
  2797	compared to OpenPose (25.94M) and HRNet32 (41.23M).
  2798	These results demonstrate that our PPR can effectively identify
  2799	key body parts using a simple network without adding extra
  2800	computation, highlighting its efficiency and practicality.
  2801	F. Visualization
  2802	In this section, we perform visualizations on Occluded-Duke
  2803	dataset to demonstrate the effectiveness of our CFPER.
  2804	Visualization of the “Easy” and “Hard” query. Fig. 4
  2805	shows “easy” and “hard” queries identified by CFPER, validating the soundness of our early-exit strategy in the CFDR
  2806	mechanism. “Easy” queries retain visible discriminative features despite partial occlusion, while “hard” queries involve
  2807	severe occlusion and fewer discriminative features, making it
  2808	difficult to distinguish the target pedestrian from others with
  2809	similar appearance. Fig. 5 shows the Top-10 retrieval results
  2810	for “easy” and “hard” queries in the coarse and fine inference
  2811	stages. For “easy” queries, global features alone are sufficient
  2812	for accurate matching in the coarse stage. In contrast, “hard”
  2813	queries benefit from fine-grained part features in the fine stage,
  2814	improving retrieval accuracy by compensating for occlusion
  2815	noise in global features.
  2816	Visualization of Body Part Attention Maps for PPR.
  2817	Fig. 6 shows the body part localization results of PPR.
  2818	Our method accurately identifies body parts and effectively
  2819	addresses occlusions and background noise. Furthermore, the
  2820	Part Experts for occluded parts remain inactive, allowing the
  2821	model to focus on visible discriminative features while further
  2822	reducing computational costs during inference.
  2823	
  2824	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.
  2825	
  2826	Fig. 4. Visualization of the “easy” and “hard” query images.
  2827	
  2828	Fig. 5. Top-10 Retrieval results of “easy” and “hard” queries. Green/red
  2829	borders illustrate correct/false matches. For each query, the first row displays
  2830	coarse inference results, the second row shows fine inference results.
  2831	
  2832	Fig. 6. Visualization of body part attention maps for PPR. The second column
  2833	denotes foreground attention maps. Green/red border indicates active/inactive
  2834	PE during inference.
  2835	
  2836	IV. C ONCLUSION
  2837	In this paper, we propose a novel model CFPER to balance
  2838	inference efficiency and retrieval accuracy for person ReID.
  2839	By dynamically adjusting the retrieval process based on query
  2840	difficulty and tailoring the learning of visible body part representations, CFPER achieves competitive performance while
  2841	significantly reducing computational costs. Extensive experiments demonstrate the effectiveness of CFPER.
  2842	R EFERENCES
  2843	[1] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang, “Beyond
  2844	part models: Person retrieval with refined part pooling (and a strong
  2845	convolutional baseline),” in Proceedings of the European conference on
  2846	computer vision (ECCV), 2018, pp. 480–496.
  2847	[2] Jiaxu Miao, Yu Wu, Ping Liu, Yuhang Ding, and Yi Yang, “Pose-guided
  2848	feature alignment for occluded person re-identification,” in Proceedings
  2849	of the IEEE/CVF international conference on computer vision, 2019,
  2850	pp. 542–551.
  2851	
  2852	[3] Guan’an Wang, Shuo Yang, Huanyu Liu, Zhicheng Wang, Yang Yang,
  2853	Shuliang Wang, Gang Yu, Erjin Zhou, and Jian Sun, “High-order
  2854	information matters: Learning relation and topology for occluded person
  2855	re-identification,” in Proceedings of the IEEE/CVF conference on
  2856	computer vision and pattern recognition, 2020, pp. 6449–6458.
  2857	[4] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang,
  2858	“Transreid: Transformer-based object re-identification,” in Proceedings
  2859	of the IEEE/CVF international conference on computer vision, 2021,
  2860	pp. 15013–15022.

exec
/bin/zsh -lc "nl -ba analyses/ALL_ANALYSES.txt | sed -n '1570,1587p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
  1570	
  1571	5. 审稿人为什么买账：这篇卖的是数据和任务分布视角，不是模型机制。新意来自“换衣 ReID 缺的不是又一个去衣服损失，而是可控的大规模换衣监督”。对我们有用的套路是，发现任务缺少某个关键变化轴，然后自己构造这个轴，并用跨数据集预训练收益证明它不是普通增广。
  1572	
  1573	**CFPER**
  1574	1. 触发观察：作者先做了一个很实用的观察，global feature 加 part feature 在 Market、Duke 这种整体行人数据上只带来很小收益，却增加计算量；但在 Occluded-Duke 上收益明显更大。也就是说，查询样本难度不同，统一走细粒度匹配会浪费简单样本，也会让困难样本得不到足够处理。
  1575	
  1576	2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。
  1577	
  1578	3. 机制怎么长出来：先用 ViT 得到全局特征和 patch 特征，用全局特征与 patch 特征的相似度排序，再用一阶差分估计可见人体区域比例。可见人体足够多就判为简单样本，只用全局特征提前退出；否则进入细阶段，用拓扑监督的 patch-to-part router 和 part experts 提取细粒度部件特征。这个机制和重定义绑定很紧，难度分流直接决定是否启用细粒度计算。
  1579	
  1580	4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。
  1581	
  1582	5. 审稿人为什么买账：这篇卖的是视角，部件专家只是服务于视角。它把效率从附属指标变成方法的核心问题，这比单纯说“我又做了一个 part module”更容易成立。对我们有用的是，找一个主流评价以外但真实存在的轴，比如计算、更新、标注、部署，再用一个很小的观察表把问题立住。
  1583	
  1584	**Channel-aware feature mining network**
  1585	1. 触发观察：作者的观察是 VI-ReID 中 RGB 和红外不仅有整体模态差异，还有通道层面的不平衡。有些通道包含衣服纹理、轮廓、热分布等身份线索，有些通道贡献很弱甚至引入噪声。现有方法多把通道操作当预处理或数据增强，没有显式挖掘通道关系。
  1586	
  1587	2. 重定义动作：它把 VI-ReID 的模态差距问题改写成“通道级身份信息没有被充分建模”。关键词是 channel-aware、channel-level feature optimization、channel-level feature refinement、channel imbalance、identity-relevant channels。

web search: 
web search: 
web search: 
web search: "HashReID" "Early Termination Strategy"
web search: 
web search: https://arxiv.org/abs/2308.11900
web search: 
web search: https://arxiv.org/e-print/2308.11900
web search: 
web search: https://arxiv.org/pdf/2308.11900
web search: 
web search: https://arxiv.org/abs/1805.08805
web search: 
web search: https://arxiv.org/pdf/1805.08805
web search: 
web search: https://arxiv.org/pdf/1805.08805v3
web search: 
web search: https://github.com/mileyan/DARENet
web search: 
web search: https://raw.githubusercontent.com/mileyan/DARENet/master/budgeted_stream/README.md
web search: 
web search: https://github.com/mileyan/DARENet
web search: 
web search: https://arxiv.org/abs/2008.06826
web search: 
web search: https://arxiv.org/abs/2509.16635
web search: 
web search: https://arxiv.org/abs/2505.18512
web search: 
web search: https://arxiv.org/abs/2012.08733
web search: 
web search: https://arxiv.org/abs/2412.00433
codex
我没有找到公开文献直接把“cheap/full top-k overlap 或排序距离”作为 ReID 推理路由信号；接近的先例分成三种：ReID 难度/早退、hash/coarse-to-fine 检索、通用 rerank uncertainty。最后判定会偏“切口存活但容易被审稿人归到 difficulty-adaptive，需要强实验命名和对照”。
exec
/bin/zsh -lc "nl -ba reviews/deep_5.md | sed -n '2280,2335p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
  2280	A. Coarse Inference Stage
  2281	Coarse Feature Extractor. We use the pre-trained Vision
  2282	Transformer (ViT) [11] as our feature extractor. Given a person
  2283	image X, the output of the encoder can be divided into two
  2284	parts: a global feature G ∈ R1×D and N patch embeddings
  2285	Z = [z1 , . . . , zN ] ∈ RN ×D .
  2286	Early-exit strategy. In the coarse inference stage, we
  2287	introduce an early-exit strategy to balance performance and
  2288	efficiency. Traditional early-exit strategies [12] are typically
  2289	used in classification tasks, relying on classifier scores. However, in person ReID, the training and testing identities are
  2290	inconsistent and the number of testing identities is unknown,
  2291	making the classifier score-based exit metric unsuitable. To
  2292	address this, we propose an early-exit strategy for person
  2293	
  2294	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.
  2295	
  2296	Fig. 2. Illustration of the early-exit strategy based on the first-order differences
  2297	of similarity scores.
  2298	
  2299	ReID, which determines whether to exit early by evaluating
  2300	the query difficulty based on the first-order differences of
  2301	similarity scores.
  2302	To assess query difficulty, our intuition is to count the
  2303	number of body-related patches. As shown in Fig. 2, we
  2304	first compute the cosine similarity between global feature
  2305	G and N patch embeddings Z, obtaining similarity scores
  2306	S = [s1 , . . . , sN ]. The scores S are sorted in descending
  2307	D
  2308	order to obtain S D = [sD
  2309	1 , . . . , sN ]. We then compute the
  2310	first-order differences D = [D1 , . . . , DN −1 ], where Di =
  2311	D
  2312	sD
  2313	i − si+1 . We assume that body and background/occlusion
  2314	features are separate classes in the feature space, so there
  2315	should be a noticeable feature transition when a body feature
  2316	shifts to a background/occlusion feature. Therefore, we select
  2317	the maximum value in D and use its corresponding index
  2318	η = arg maxi (D) as the split point between the body and
  2319	background/occlusion regions. Patches with the top-η similarity scores are considered as body regions, while the rest
  2320	are considered as background/occlusion regions. Finally we
  2321	introduce an early-exit threshold α ∈ [0, 1]. If η ≥ ⌈N α⌉, the
  2322	query is considered “easy” with sufficient visible body regions
  2323	for high-confidence retrieval. In this case, only global features
  2324	G are used for retrieval, and the whole inference terminates.
  2325	Otherwise, the query is considered “hard” and proceeds to the
  2326	fine inference stage to extract fine-grained part features.
  2327	Supervision Loss of Coarse Inference Stage. We adopt
  2328	cross-entropy loss as ID loss to supervise the learning of
  2329	encoder:
  2330	LCOARSE = LID (G),
  2331	(1)
  2332	B. Fine Inference Stage
  2333	In the fine inference stage, we employ PMoE to extract finegrained part features for “hard” queries to support detailed
  2334	retrieval. PMoE consists of a Patch-to-Part Router and a set
  2335	of Part Experts.

exec
/bin/zsh -lc "nl -ba reviews/deep_5.md | sed -n '2335,2385p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
 succeeded in 0ms:
  2335	of Part Experts.
  2336	1) Patch-to-Part Router. The PPR module learns to route
  2337	patches embeddings Z to M+1 parts C = [c0 , c1 , . . . , cM ],
  2338	where c0 represents the background, and [c1 , . . . , cM ] corresponds to M body parts. PPR adopts a standard MoE router
  2339	structure, consisting of a fully-connected layer with parameters
  2340	θ ∈ R(M +1)×D followed by a softmax layer, which produces
  2341	the probabilities P ∈ RN ×(M +1) of each patch belonging to
  2342	the background and M body parts:
  2343	P = {pi ∈ RN | i = 0, 1, ..., M } = Sof tmax(Z · θT ), (2)
  2344	
  2345	where pi represents the probabilities of patches embeddings
  2346	Z belonging to part ci .
  2347	Next, the router needs to assign patches to each expert based
  2348	on P. Traditional MoE patch-level routers [13] commonly
  2349	use the Top-k mechanism, where each expert is assigned k
  2350	patches (k ≪ N ). However, this fixed Top-k mechanism
  2351	is unsuitable for person ReID due to the varying sizes of
  2352	body parts. If k is too small, larger body regions (e.g., the
  2353	torso) may not be fully covered, leading to information loss.
  2354	Conversely, if k is too large, smaller body parts (e.g., the
  2355	head) may be polluted by surrounding background/occlusion
  2356	patches, introducing noise. To address this, our PPR employs
  2357	a soft routing mechanism. First, we aggregate M body parts
  2358	probabilities [p1 , . . . , pM ] to compute a foreground probability
  2359	pf ∈ RN : pjf = sum(pj1 , . . . , pjM ), where pji represents the
  2360	probability of patch embedding zj belonging to body part
  2361	ci . We then perform Probability Weighted Average Pooling
  2362	(PWAP) on patch embeddings Z to obtain a foreground
  2363	feature rf , a background feature r0 and M body part features
  2364	[r1 , . . . , rM ]:
  2365	PN
  2366	j
  2367	j=1 zj · pi
  2368	(3)
  2369	ri = PN j , ∀i ∈ {f, 0, 1, ..., M }
  2370	j=1 pi
  2371	where zj denotes the j-th patch embedding from the coarse
  2372	inference stage.
  2373	Supervision Loss of PPR. We introduce human topology
  2374	priors in the form of coarse body part labels. For each patch
  2375	j ∈ RN , its part label yj ∈ {0, 1, ..., M }. The cross-entropy
  2376	loss Lh with label smoothing is computed as:
  2377	Lh = −
  2378	
  2379	M
  2380	X N
  2381	X
  2382	
  2383	qi · log(pji ),
  2384	
  2385	i=0 j=1

web search: 
web search: "rank-stability" "person re-identification"
web search: 
web search: "rank stability" "retrieval" "early exit"
codex
结论：**存活，但要降调。**  
`Rank-Instability Adaptive ReID` 没查到“直接用 cheap-vs-full top-k 排名不一致度作为 ReID 推理路由信号”的确切先例；这个切口有独立新意。  
但它会强烈撞上 **CFPER/DaReNet/HashReID/AcuRank** 的大类叙事：query-adaptive compute / early-exit / uncertain-query extra budget。不能写成“首次发现 ReID query 需要动态算力”，只能写成“首次把动态算力路由从静态难度/置信度改成 cheap-to-full rank preservation / rank regret 的可测信号”。整体信心：**7/10**。

**逐类判定**
| 类别 | 代表先例 | 是否用了 cheap-vs-full rank disagreement 做路由 | 撞车判断 |
|---|---|---:|---|
| (a) CFPER / query-difficulty adaptive ReID | CFPER, ICME 2025，本地 PDF 抽取见 [reviews/deep_5.md](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:2130)，DOI `10.1109/ICME59968.2025.11209843` | **否**。它用 global-patch 相似度一阶差分估计可见人体区域比例，`easy` 只走 global，`hard` 走 part experts，见 [signal lines](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:2296) | **高杀伤但不硬撞**。它已占“ReID query difficulty adaptive compute / early exit”主叙事；你的新意必须落在“rank disagreement 不是可见性/置信度/难度分数” |
| (b) early-exit / dynamic inference / multi-exit ReID | [DaReNet](https://arxiv.org/abs/1805.08805), [HashReID](https://arxiv.org/abs/2308.11900) | **否**。DaReNet 是多分辨率 resource-aware ReID；HashReID 有 ranking regularizer 和 early termination，但不是用 cheap-full 当前 top-k 分歧当路由 | **中高撞**。它们证明 ReID 早退/动态预算不是新问题；HashReID 尤其接近“rank consistency”语义，但用途不同 |
| (c) cascade / coarse-to-fine / hierarchical ReID | [CtF: Faster Person Re-ID](https://arxiv.org/abs/2008.06826), CFPER | **否**。CtF 是短哈希筛候选、长哈希细排的 cascade；CFPER 是 coarse global 到 fine part | **中撞**。如果你只是 cheap top-k 后 full rerank，会被归到 cascade；必须强调“路由依据是 cheap/full rank regret 的预测或估计” |
| (d) anytime ReID | [Towards Anytime Retrieval: Anytime Person Re-ID](https://arxiv.org/abs/2509.16635) | **否**。这里的 anytime 更偏跨时间/长期检索基准，不是推理预算 anytime | **低撞**。不要用 “anytime ReID” 命名，容易误导 |
| (e) hardness / uncertainty-aware ReID | 如 [UMTS](https://arxiv.org/abs/2012.08733) 等 uncertainty ReID | **否**。多数是训练、域泛化、伪标签、特征不确定性，不是按 query 分配推理计算 | **中低撞**。审稿人会说你是 uncertainty/difficulty routing，所以要证明 RI 比 margin、entropy、occlusion、visibility 更预测 cheap-vs-full regret |
| (f) 通用 retrieval adaptive reranking budget | [AcuRank](https://arxiv.org/abs/2505.18512) | **接近但否**。它按 ranking uncertainty 自适应 rerank computation，不是 ReID，也不是 cheap-full rank disagreement | **中高概念撞**。它会削弱“rank uncertainty 分配预算”的新意；但 ReID 网络早退 + cheap/full rank preservation 仍可存活 |

**关键 verdict**
不是硬撞 CFPER。CFPER 的隐藏变量是“query difficulty / visible body sufficiency”，路由信号是图像内部 global-patch 相似度结构；你的隐藏变量是“cheap representation 是否会改变检索排序”，是**关系级、检索结果级**变量。这个差别能成立。

但如果实现最后退化成 `cheap score margin / entropy / top1 gap / occlusion score`，那基本会被归类成 **CFPER-style difficulty-adaptive ReID**，新意会很弱。真正能保住切口的是：

1. 定义 oracle `RI@K = distance(rank_cheap, rank_full)`，例如 top-k overlap loss、RBO、Kendall tau、NDCG regret。
2. 证明 `RI@K` 比静态 difficulty proxy 更能预测 `AP_full - AP_cheap` 或 top-k identity flip。
3. 推理时不能真的先算 full rank 再决定早退，否则没有效率收益；必须用 cheap-only predictor / cheap-stage surrogate 去预测 RI。
4. 强对照必须包括 CFPER-style visibility/body-ratio、cheap margin、top1-top2 gap、entropy、occlusion程度、random routing、fixed budget cascade。

最终建议标题不要叫 “difficulty-adaptive”。更硬的写法是：  
**Rank-Preservation Certified Early Exit for Efficient Person ReID** 或 **Rank-Regret Adaptive ReID**。这样审稿人更难把它直接压成“又一个 CFPER”。
tokens used
266,224
结论：**存活，但要降调。**  
`Rank-Instability Adaptive ReID` 没查到“直接用 cheap-vs-full top-k 排名不一致度作为 ReID 推理路由信号”的确切先例；这个切口有独立新意。  
但它会强烈撞上 **CFPER/DaReNet/HashReID/AcuRank** 的大类叙事：query-adaptive compute / early-exit / uncertain-query extra budget。不能写成“首次发现 ReID query 需要动态算力”，只能写成“首次把动态算力路由从静态难度/置信度改成 cheap-to-full rank preservation / rank regret 的可测信号”。整体信心：**7/10**。

**逐类判定**
| 类别 | 代表先例 | 是否用了 cheap-vs-full rank disagreement 做路由 | 撞车判断 |
|---|---|---:|---|
| (a) CFPER / query-difficulty adaptive ReID | CFPER, ICME 2025，本地 PDF 抽取见 [reviews/deep_5.md](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:2130)，DOI `10.1109/ICME59968.2025.11209843` | **否**。它用 global-patch 相似度一阶差分估计可见人体区域比例，`easy` 只走 global，`hard` 走 part experts，见 [signal lines](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:2296) | **高杀伤但不硬撞**。它已占“ReID query difficulty adaptive compute / early exit”主叙事；你的新意必须落在“rank disagreement 不是可见性/置信度/难度分数” |
| (b) early-exit / dynamic inference / multi-exit ReID | [DaReNet](https://arxiv.org/abs/1805.08805), [HashReID](https://arxiv.org/abs/2308.11900) | **否**。DaReNet 是多分辨率 resource-aware ReID；HashReID 有 ranking regularizer 和 early termination，但不是用 cheap-full 当前 top-k 分歧当路由 | **中高撞**。它们证明 ReID 早退/动态预算不是新问题；HashReID 尤其接近“rank consistency”语义，但用途不同 |
| (c) cascade / coarse-to-fine / hierarchical ReID | [CtF: Faster Person Re-ID](https://arxiv.org/abs/2008.06826), CFPER | **否**。CtF 是短哈希筛候选、长哈希细排的 cascade；CFPER 是 coarse global 到 fine part | **中撞**。如果你只是 cheap top-k 后 full rerank，会被归到 cascade；必须强调“路由依据是 cheap/full rank regret 的预测或估计” |
| (d) anytime ReID | [Towards Anytime Retrieval: Anytime Person Re-ID](https://arxiv.org/abs/2509.16635) | **否**。这里的 anytime 更偏跨时间/长期检索基准，不是推理预算 anytime | **低撞**。不要用 “anytime ReID” 命名，容易误导 |
| (e) hardness / uncertainty-aware ReID | 如 [UMTS](https://arxiv.org/abs/2012.08733) 等 uncertainty ReID | **否**。多数是训练、域泛化、伪标签、特征不确定性，不是按 query 分配推理计算 | **中低撞**。审稿人会说你是 uncertainty/difficulty routing，所以要证明 RI 比 margin、entropy、occlusion、visibility 更预测 cheap-vs-full regret |
| (f) 通用 retrieval adaptive reranking budget | [AcuRank](https://arxiv.org/abs/2505.18512) | **接近但否**。它按 ranking uncertainty 自适应 rerank computation，不是 ReID，也不是 cheap-full rank disagreement | **中高概念撞**。它会削弱“rank uncertainty 分配预算”的新意；但 ReID 网络早退 + cheap/full rank preservation 仍可存活 |

**关键 verdict**
不是硬撞 CFPER。CFPER 的隐藏变量是“query difficulty / visible body sufficiency”，路由信号是图像内部 global-patch 相似度结构；你的隐藏变量是“cheap representation 是否会改变检索排序”，是**关系级、检索结果级**变量。这个差别能成立。

但如果实现最后退化成 `cheap score margin / entropy / top1 gap / occlusion score`，那基本会被归类成 **CFPER-style difficulty-adaptive ReID**，新意会很弱。真正能保住切口的是：

1. 定义 oracle `RI@K = distance(rank_cheap, rank_full)`，例如 top-k overlap loss、RBO、Kendall tau、NDCG regret。
2. 证明 `RI@K` 比静态 difficulty proxy 更能预测 `AP_full - AP_cheap` 或 top-k identity flip。
3. 推理时不能真的先算 full rank 再决定早退，否则没有效率收益；必须用 cheap-only predictor / cheap-stage surrogate 去预测 RI。
4. 强对照必须包括 CFPER-style visibility/body-ratio、cheap margin、top1-top2 gap、entropy、occlusion程度、random routing、fixed budget cascade。

最终建议标题不要叫 “difficulty-adaptive”。更硬的写法是：  
**Rank-Preservation Certified Early Exit for Efficient Person ReID** 或 **Rank-Regret Adaptive ReID**。这样审稿人更难把它直接压成“又一个 CFPER”。
