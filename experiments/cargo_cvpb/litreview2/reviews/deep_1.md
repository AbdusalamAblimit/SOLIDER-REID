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
session id: 019ef835-507f-7d02-96de-bcee285eb285
--------
user
你是 ReID 创新方法论拆解员 1。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(1)==
'Knowledge and experience' for visible-infrared person re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会按你的要求先从 PDF 本文里读方法部分，再在必要处查先例，最后只提炼“创新是怎么被构造出来的”，不把它写成模块复述。当前环境是只读的，所以我只会读取和分析文件，不改动论文库。
hook: PreToolUse
hook: PreToolUse
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

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pwd && rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
我正在使用 PDF 技能里的 Poppler 文本抽取流程，但这里只需要内容拆解，不需要渲染版面。下一步先定位每篇的方法章节边界，避免把摘要或实验段落当成方法来分析。
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
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 \"'Knowledge and experience' for visible-infrared person re-identification.pdf\" - | rg -n \""'^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
42:1. Introduction
148:2. Related work
149:2.1. Single-modality VV-ReID
166:3. Proposed model
174:3.1. Problem formulation
184:2.2. Cross-modality VI-ReID
321:3.2.1. Modality-speciﬁc feature extraction module
360:3.2.2. Modality-shared feature extraction module
379:3.3. Knowledge and experience transformer
385:3.3.1. Knowledge module
410:3.2. VI-ReID subnetwork
459:3.3.2. Reasoning module
627:3.4. Fusion module
660:4. Experiments
673:4.1. Datasets and evaluation metrics
690:4.2. Online batch sampling strategy
712:Methods
762:56.96
763:58.70
764:56.91
765:61.59
766:65.40
767:64.23
768:66.34
769:67.25
770:70.94
771:59.84
772:71.88
773:71.34
774:71.36
775:71.63
778:91.50
779:92.06
780:91.34
782:93.33
783:95.19
785:95.38
786:94.93
787:92.49
788:96.45
789:95.24
790:96.77
791:94.98
792:96.5±0.6
794:96.82
796:96.52
798:97.14
799:98.73
801:98.46
802:96.37
803:97.20
804:98.87
805:98.53
806:98.26
807:98.37
810:54.95
811:56.42
812:55.45
813:59.96
814:61.98
815:61.21
816:62.51
817:64.29
818:65.51
819:57.70
820:68.57
821:66.20
822:68.63
823:67.28
824:67.5±0.7
826:62.09
827:65.85
828:63.51
831:70.34
832:73.44
833:72.95
836:80.23
840:78.4±1.3
842:93.74
843:94.73
844:94.42
847:96.75
849:96.64
852:97.91
856:97.6±0.6
858:97.85
860:97.81
863:99.26
865:99.27
868:99.53
872:99.3±0.3
874:48.02
875:49.56
876:48.56
879:54.36
880:56.06
881:57.62
884:63.13
888:61.3±1
890:59.74
891:63.09
892:62.79
893:67.07
894:72.56
895:64.80
896:68.15
897:69.58
898:73.28
899:62.13
900:76.90
901:75.18
902:77.40
903:79.44
904:75.2±1.3
906:92.07
907:96.61
908:96.53
910:96.98
911:96.62
913:96.66
914:95.20
915:94.83
916:97.68
917:97.01
918:98.69
919:98.32
920:98.0 ±0.4
922:96.22
924:99.07
926:99.20
927:99.31
929:99.03
930:99.43
931:98.24
932:99.64
933:99.09
934:99.82
935:99.91
936:99.5 ±0.2
938:64.91
939:70.31
940:69.79
941:73.58
942:76.58
943:71.53
944:74.09
945:74.37
946:77.18
947:69.03
948:79.91
949:78.30
950:81.31
951:79.85
952:79.3±1
954:69.76
955:72.06
956:70.03
959:76.83
960:78.86
961:80.39
964:84.87
970:95.85
971:97.77
972:97.70
975:98.84
977:98.80
980:99.41
986:98.90
988:99.51
991:99.89
993:99.83
996:99.97
1000:99.9±0.1
1002:57.81
1003:61.69
1004:61.45
1007:65.02
1008:63.82
1009:68.60
1012:74.11
1030:Methods
1063:73.6
1064:80.54
1065:80.31
1066:83.39
1067:87.35
1068:88.78
1069:89.12
1070:89.91
1071:92.10
1072:87.82
1073:90.10
1074:95.17
1075:94.11±0.64
1077:88.6
1078:91.96
1079:92.70
1081:97.10
1082:94.76
1084:96.57
1085:97.97
1086:96.36
1087:97.09
1088:98.82
1089:98.66±1.02
1092:95.07
1093:96.03
1095:98.90
1096:96.37
1098:98.33
1099:99.17
1100:98.20
1101:98.35
1102:99.51
1105:66.9
1106:72.10
1107:73.07
1108:78.75
1109:76.94
1110:88.49
1111:84.43
1112:85.64
1113:82.11
1114:78.35
1115:82.73
1116:87.30
1119:74.6
1120:77.95
1121:75.93
1122:81.07
1123:86.40
1124:84.55
1125:88.38
1126:88.34
1127:91.21
1128:85.73
1129:87.66
1130:91.97
1133:88.7
1134:90.45
1135:90.93
1137:96.36
1138:93.72
1140:96.16
1141:98.20
1142:94.90
1143:96.68
1144:97.92
1148:93.62
1149:94.59
1151:98.20
1152:95.83
1154:97.98
1155:99.08
1156:97.67
1157:99.01
1158:99.07
1161:67.5
1162:69.79
1163:69.49
1164:78.89
1165:75.70
1166:83.64
1167:83.86
1168:84.06
1169:81.62
1170:76.33
1171:81.52
1172:84.46
1194:4.3. Implementation details
1205:4.5. Ablation study
1206:4.5.1. Eﬀectiveness of using the inferred features from our model’s
1214:4.4. Comparison with state-of-the-art methods
1230:Methods
1256:40.3
1257:41.0
1258:54.9
1259:52.1
1260:56.63
1261:55.39±1.44
1263:71.4
1264:73.4
1265:84.9
1266:81.3
1268:80.58±1.06
1270:79.6
1271:81.9
1272:90.9
1273:87.5
1275:85.74±1.32
1277:48.4
1278:49.6
1279:62.9
1280:58.3
1281:62.35
1282:62.07±0.98
1284:48.0
1285:48.5
1286:62.5
1287:65.9
1288:62.42
1291:79.2
1292:81.0
1293:90.3
1294:88.4
1298:86.1
1299:87.8
1300:94.7
1301:94.1
1303:94.12±0.60
1305:52.3
1306:53.0
1307:65.8
1308:50.3
1309:65.33
1310:65.52±0.36
1314:Methods
1335:65.28
1336:67.52
1337:67.62
1338:68.32
1339:70.33
1340:72.10
1342:94.41
1343:95.29
1344:94.84
1345:95.11
1346:96.07
1347:96.53
1349:98.24
1350:98.47
1351:98.26
1352:98.50
1353:98.81
1354:99.02
1356:61.63
1357:63.85
1358:63.68
1359:64.41
1360:66.50
1361:67.53
1364:92.7
1365:92.7
1366:92.7
1367:81.2
1368:67.2
1370:27.8
1371:28.1
1372:28.1
1373:28.1
1374:28.1
1375:29.58
1386:4.5.2. The impact of each component in our proposed model
1434:4.5.3. Evaluations on the number of transformer blocks in the knowledge

exec
/bin/zsh -lc 'pdftotext -f 1 -l 8 "4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc 'pdftotext -f 1 -l 8 "3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc 'pdftotext -f 1 -l 8 "A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
12:Abstract— Cloth-changing person re-identification (ReID) is a
78:I. I NTRODUCTION
208:II. R ELATED W ORK
385:III. P ROPOSED A PPROACH
717:IV. E XPERIMENTS AND D ISCUSSION
817:47.6% and 46.9%, and their corresponding improvements can reach 10.0% (mAP) and 22.5% (rank-1).
843:16.1% and 42.3%, and the improvements of the SAVS

 succeeded in 184ms:
37:ABSTRACT
51:1. Introduction
102:1. When it comes to mining intrinsic attributes within a single modality,
110:2. Regarding feature alignment across different modalities, although several recent studies have introduced frequency-domain modeling,
136:2. For the second problem, the SSM decomposes the characteristic
156:1. For the first issue, the AFF module dynamically generates inputspecific filters to address the limitations of static convolutional
172:2.3. Frequency domain analysis in vision tasks
216:2. Related work
217:2.1. Image-based VI-ReID methods
234:2.2. Feature-based VI-ReID methods
259:3. Proposed 4SNet framework
291:3.1. Overview
324:3.2. Adaptive Frequency Filter (AFF)
345:3.3. Spectrum Synergy Module (SSM)
539:3.4. Loss function
567:4. Experimental results and analysis
637:4.1. Experimental setup
674:Methods
751:67.25
752:67.53
753:63.94
754:74.70
755:66.76
756:69.47
757:68.36
758:74.48
759:75.62
760:75.91
761:75.99
762:72.23
763:76.33
764:78.53
765:71.30
766:75.20
767:77.80
768:76.60
769:77.10
771:95.38
772:95.36
773:93.34
774:97.60
775:96.23
776:96.42
777:96.01
778:96.99
780:97.01
781:97.63
782:97.04
783:97.58
784:97.51
785:95.54
786:96.60
787:97.80
788:97.10
791:64.29
792:64.98
793:60.73
794:71.80
795:64.01
796:66.41
797:66.42
798:71.34
799:74.22
800:72.96
801:70.71
802:70.58
803:69.38
804:74.20
805:67.73
806:71.80
807:75.10
808:73.00
809:73.90
811:69.58
812:71.66
813:71.06
814:80.30
815:72.89
816:71.72
817:74.45
818:82.79
819:84.25
820:79.34
821:80.92
822:83.45
823:81.48
824:83.00
825:77.16
826:82.00
827:87.30
828:82.70
829:82.40
831:96.66
832:96.73
833:97.31
834:99.00
835:97.93
836:97.19
837:97.80
838:98.88
840:98.97
841:98.88
842:98.62
843:98.96
844:98.99
845:98.06
846:98.70
847:99.40
848:98.90
851:74.37
852:76.52
853:76.01
854:83.30
855:76.70
856:76.38
857:79.36
858:85.26
859:86.73
860:84.62
861:82.64
862:84.25
863:83.76
864:85.54
865:80.55
866:85.00
867:89.10
868:85.30
869:86.90
871:89.91
872:84.83
873:73.95
874:91.10
875:85.51
876:85.04
877:89.95
878:91.80
879:94.70
880:92.59
881:95.92
882:89.03
883:94.51
884:90.40
885:94.32
886:89.00
887:95.50
888:91.10
889:81.90
891:96.57
893:89.47
894:97.80
895:94.99
896:94.38
897:97.14
900:98.12
901:99.01
902:98.19
906:96.10
907:99.00
909:92.90
911:85.64
912:76.55
913:65.49
914:85.10
915:79.95
916:82.52
917:83.69
918:86.08
919:90.45
920:86.18
921:89.26
922:83.89
923:88.46
924:81.20
925:89.02
926:84.70
927:90.00
928:81.70
929:78.60
931:88.34
932:84.16
933:69.67
934:89.50
935:80.65
936:81.11
937:90.10
938:88.06
939:93.34
940:91.35
941:93.58
942:87.91
943:93.25
944:87.70
945:93.27
946:88.20
947:98.50
948:89.90
949:80.80
951:96.16
953:86.41
954:96.80
955:92.80
956:92.35
957:97.48
960:98.74
961:98.33
962:97.15
966:95.10
967:94.00
969:91.70
971:84.06
972:75.13
973:61.98
974:83.40
975:75.91
976:79.59
977:82.90
978:83.06
979:89.22
980:84.92
981:86.88
982:83.04
983:87.18
984:78.20
985:88.77
986:82.80
987:88.70
988:80.50
989:77.10
995:77.86
997:97.29
999:74.31
1001:84.96
1003:99.37
1005:86.85
1007:95.92
1009:98.96
1011:92.51
1013:94.31
1015:98.17
1017:90.61
1026:84.96% in Rank-1 and 86.85% in mAP. On the RegDB dataset, 4SNet
1081:4.2. Comparison with state-of-the-art methods
1102:Methods
1140:52.54
1141:60.40
1142:62.51
1143:64.91
1144:61.78
1145:70.20
1146:64.30
1147:62.30
1149:88.50
1151:90.30
1154:91.10
1156:90.60
1158:58.91
1159:63.20
1160:65.82
1161:67.91
1162:65.08
1163:55.80
1164:66.60
1165:68.30
1167:59.96
1168:52.20
1169:54.91
1170:56.38
1171:56.47
1172:56.60
1173:56.50
1174:55.20
1176:81.60
1178:84.90
1181:84.10
1183:83.40
1185:62.72
1186:59.80
1187:62.93
1188:63.21
1189:62.79
1190:62.70
1191:63.50
1192:62.50
1198:64.78
1200:95.53
1202:67.57
1204:56.41
1206:92.00
1208:63.51
1253:72.17
1254:75.48
1255:75.62
1256:77.86
1258:69.86
1259:72.59
1260:72.89
1261:74.31
1263:79.63
1264:82.15
1265:83.21
1266:84.96
1268:81.56
1269:83.45
1270:84.51
1271:86.85
1273:60.98
1274:63.33
1275:63.15
1276:64.78
1278:63.56
1279:65.28
1280:65.14
1281:67.57
1283:53.86
1284:55.15
1285:55.46
1286:56.41
1288:61.30
1289:62.91
1290:62.87
1291:63.51
1324:77.02
1325:76.65
1326:77.86
1328:73.51
1329:73.21
1330:74.31
1332:82.65
1333:82.93
1334:84.96
1336:84.75
1337:85.14
1338:86.85
1351:4.3. Ablation analysis
1366:2.95%. When both AFF and SSM are integrated, the model achieves its

 succeeded in 405ms:
10:Abstract—Video-based person re-identification (Re-ID) aims
33:I. I NTRODUCTION
247:II. R ELATED W ORKS
248:Related works are organized as follows. In Section II-A, we
381:III. M ETHODS
746:IV. E XPERIMENTS
814:93.1% Rank-1 accuracy, outperforming all the mentioned

codex
前四篇的方法边界已经定位到：一篇有排版乱序，需要按小节重组；另外三篇的方法段落都在第 3 节。接下来我继续定位后两篇，然后逐篇抽出方法正文来读。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'pdftotext -f 1 -l 8 "A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related|Background|Taxonomy)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc 'pdftotext -f 1 -l 8 "A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf" - | rg -n "''^(Abstract|ABSTRACT|I'"\\.|II\\.|III\\.|IV\\.|V\\.|[0-9]+\\.|[IVX]+\\.|Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|EXPERIMENT|Experiment|Conclusion|Related)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
59:Abstract
207:2.1
236:2.2
336:2.3
338:Taxonomy of SRID Methods
360:3.1
413:3.2
482:205.8K
492:89.0K
502:188.7K
514:4.8K
526:13.1K
538:26.3K
546:18.0K
600:36.0K
612:706.5K
624:22.9K
634:6667.0K
644:53.6K
706:10.7
707:10.7
721:32.1
722:39.3
737:28.3
738:14.2
739:40.0 43.7 39.4
748:32.6
749:17.0
750:42.7 44.5 51.7
762:28.6
763:10.7
764:45.5 63.6
774:39.3
775:39.3
776:36.4 81.8
789:8.8
790:28.4 28.9 49.4
792:8.93M
800:25.1
801:11.7
802:31.1 31.1 43.2
804:7.15M
812:25.8
813:11.7
814:16.7 22.2 45.0
816:8.47M
824:43.2
825:16.3
826:43.7 47.8 52.3 Transformer
827:0.49M
835:41.7
836:24.5
837:44.8 46.3 66.4
839:0.15M
847:47.5
848:27.3
849:45.6 48.2 69.6
851:3.32M
859:15.1
860:10.8
861:31.4 31.5 22.2
863:11.11M
871:20.1
872:10.8
873:36.4 32.4 46.3 GAT, LSTM
874:0.35M
882:31.3
883:13.2
884:34.0 38.9 45.9 MGRN, LSTM
885:5.58M
893:59.1
894:20.5
895:47.7 48.3 57.4
897:1.42M
905:56.8
906:21.3
907:47.9 50.1 65.9
909:2.06M
917:54.1
918:29.0
919:50.9 60.1 71.4
921:1.30M
929:68.7
930:32.7
931:49.2 59.1 73.6 Transformer
932:0.40M
940:72.0
941:36.0
942:51.9 61.5 76.0 Transformer
943:0.40M
956:Method
985:Methods
1009:3.5
1011:20.8
1013:9.8
1015:37.1
1017:9.6
1019:35.5
1021:3.0
1023:14.6
1025:3.9
1027:32.4
1031:6.6
1033:50.2
1035:9.3
1037:26.6
1039:9.7
1041:27.2
1043:3.0
1045:10.6
1047:3.5
1049:16.6
1053:9.8
1055:71.8
1057:16.5
1059:48.1
1061:7.1
1063:51.2
1065:4.7
1067:15.9
1069:6.7
1071:36.4
1075:10.8
1077:84.8
1079:16.5
1081:69.1
1083:15.7
1085:68.0
1087:5.4
1089:25.6
1091:7.1
1093:42.0
1097:13.1
1099:78.5
1101:17.9
1103:67.1
1105:15.7
1107:65.6
1109:6.7
1111:23.0
1113:8.6
1115:44.1
1119:11.2
1121:85.5
1123:17.0
1125:71.2
1127:14.1
1129:70.2
1131:4.9
1133:27.2
1135:7.5
1137:50.1
1141:16.1
1143:87.9
1145:18.9
1147:73.6
1149:18.1
1151:72.1
1153:7.3
1155:26.5
1157:9.8
1159:50.6
1206:0.15M parameters, followed by the graph-based MG-SCR
1303:6.1
1329:6.2
1331:Methods
1339:35.0
1341:21.9
1345:35.5
1347:28.5
1351:46.2
1353:37.5
1357:45.9
1359:60.2
1363:47.5
1365:64.1
1369:47.4
1371:57.8
1375:48.2
1377:60.2
1407:Conclusion
1445:149. Springer, 2017.
1449:2021.
1486:2016.
1493:2020.
1499:435. Springer, 2022.

 succeeded in 0ms:
45:1. Introduction
146:2. Related work
150:2.1. Person re-identiﬁcation
177:2.2. Occluded person re-identiﬁcation
225:3. Methodology
226:3.1. Overview of the proposed method
243:2.3. Partial person re-identiﬁcation
264:3.2. Semantic-guided occlusion simulation (SGOS)
369:3.3. Local feature semantic expansion (LFSE)
641:3.5. Discuss
648:3.4. Cross center oﬀset loss
672:Methods
674:4. EXPERIMENTAL ANALYSIS
678:4.1. Experimental settings
744:84.1
745:88.0
746:89.0
748:93.4
749:95.4
750:96.1
752:75.1
753:78.2
754:78.8
756:86.9
757:88.8
758:89.3
775:81.3
776:87.3
777:86.3
778:86.9
780:92.7
781:94.4
782:95.0
783:94.7
785:72.6
786:78.0
787:78.0
788:76.6
790:86.2
791:86.7
792:89.4
793:88.1
819:86.0
820:87.2
821:88.5
822:84.0
823:89.8
824:87.0
825:78.6
827:94.5
828:95.3
829:95.0
830:94.6
831:95.5
832:95.1
833:93.5
835:76.5
836:77.9
838:73.1
839:83.0
840:78.3
841:78.1
843:88.9
844:88.5
846:87.3
847:91.7
848:89.6
849:84.2
869:89.9
870:88.0
871:86.8
873:86.2
875:95.7
876:95.1
877:95.0
879:94.5
881:82.3
882:79.8
883:78.8
884:82.8
885:79.1
887:91.2
888:89.3
889:89.7
890:90.8
891:89.4
899:91.8
901:96.9
903:83.8
905:90.8
913:Methods
936:73.5
937:77.1
939:76.4
940:81.4
942:76.7
943:80.8
945:79.4
946:83.2
957:77.9
958:78.1
960:80.9
961:80.9
963:80.8
964:81.1
966:83.9
967:83.5
978:71.6
979:79.3
981:75.1
982:83.2
985:81.5
988:83.2
1002:75.2
1003:80.1
1004:69.3
1006:77.3
1007:82.0
1008:72.8
1010:77.8
1011:82.4
1012:72.7
1014:79.1
1015:84.6
1016:75.6
1024:78.0
1026:80.6
1028:81.5
1030:83.9
1038:82.3
1040:85.0
1042:85.5
1044:86.0
1072:Methods
1096:53.6
1097:46.3
1099:64.5
1100:62.2
1112:43.5
1113:50.8
1114:56.4
1115:57.6
1117:56.3
1118:65.0
1119:68.1
1120:68.1
1137:54.1
1138:42.4
1139:59.7
1140:55.2
1142:66.7
1143:55.4
1144:68.5
1145:68.3
1162:60.9
1163:57.3
1164:54.9
1165:57.4
1167:71.3
1168:70.6
1169:62.8
1170:68.6
1178:61.5
1180:68.7
1198:85.5 % and a Rank-1 score of 86.0 % on CUHK 03-Labeled labeled by a
1211:4.2. Comparison with state-of-the-art methods on datasets.

codex
后两篇也定位完了。注意那篇综述没有“提出方法”的实验章节，它的方法论对象是分类框架和综述组织方式，所以我会把它按“如何把 3D skeleton ReID 这个方向重定义成 taxonomy”来拆。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 \"'Knowledge and experience' for visible-infrared person re-identification.pdf\" - | sed -n '160,660p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2

Pattern Recognition 172 (2026) 112553

N. Huang et al.

3. Proposed model

distance. For instance, considering that optimizing metric losses in a
hand-designed way are sub-optimal and time-consuming, Gu et al. [2]
proposed an AutoML-based methods, termed LFS-ReID, for searching
optimal margin-based softmax loss function. Eventually, their optimized
metric loss achieved signiﬁcant performance.

3.1. Problem formulation
In this section, we take the RGB-to-IR matching as an example to theoretically compare existing VI-ReID models with our proposed model.
𝐾

𝑔
Suppose that 𝑞 is a query image of RGB modality and 𝐺 = {𝑔𝑖 }𝑖=1
denotes the gallery set with a set of IR images. Existing VI-ReID models
will ﬁrst extract their features 𝐅𝑞 and 𝐅𝑔𝑖 from these input images via a
feature extractor 𝑓 (∗), i.e.,

2.2. Cross-modality VI-ReID
VI-ReID has gained increasing interest in research recently, but summarizing the extensive existing research on VI-ReID models is beyond
the scope of this paper. We recommend interested readers to recent surveys on VI-ReID [8] for further introduction.
Generally speaking, there are two major solutions to VI-ReID, i.e.,
modality-shared feature learning and modality-speciﬁc feature compensation. Modality-shared feature learning [9,10] based models aim to
transfer the features extracted from the input image of diﬀerent modalities into the same feature space for extracting their discriminative
modality-shared person features. For example, considering that shape
information is one of the important modality-invariant features, [10]
proposed a multi-task learning-based model that leverages person segmentation to extract modality-invariant shape information for VI-ReID.
Diﬀerently, [9] proposed to exploit the relations among diﬀerent person
parts to complement those modality-shared appearance features, achieving great performance improvements.
Diﬀerently, modality-speciﬁc feature compensation based models
[11–13] ﬁrst generate the missing speciﬁc information from the existing
ones and then use those generated information together with the original images for VI-ReID. For example, [11] ﬁrst analyzed the drawbacks
of existing models and proposed to optimize their image generator in the
network structures and their ReID network in cross-modal feature fusion
strategies. Accordingly, a new two-stage GAN based model is designed,
which achieves new state-of-the-art performance.
Compared with modality-speciﬁc feature compensation based models, modality-shared feature learning based ones are usually more
concise, thus obtaining more attentions [14,15]. However, existing
modality-shared feature learning based models only focus on exploring the features within the input images, ignoring the fact that human
visual systems also employ those information inferred from their ‘knowledge and experience’. Diﬀerently, as that in human visual systems, our
proposed model can not only extract discriminative modality-shared information from the input images but also infer more modality-shared
information from its ‘knowledge and experience’, thus achieving better
results.

𝐅𝑞 = 𝑓 (𝑞, 𝛼), 𝐅𝑔𝑖 = 𝑓 (𝑔𝑖 , 𝛼),

(1)

where 𝛼 denotes the parameters of the feature extractor 𝑓 (∗). Then, they
will achieve person ReID by optimizing the following objective function

𝑖∗ = 𝑎𝑟𝑔𝑚𝑎𝑥𝑖=1,2,…,𝐾𝑔 𝑆𝑖𝑚(𝐅𝑞 , 𝐅𝑔𝑖 ).

(2)

Here, 𝑖∗ denotes that 𝑔𝑖∗ is the matched image. 𝑆𝑖𝑚(∗) denotes a metric
function.
Diﬀerently, our proposed model will use its ‘knowledge and experience’ to infer more discriminative information for VI-ReID. Here, we
denote the ‘knowledge and experience’ of our proposed model by 𝐊𝐄,
which can be seen as a set of person features/attributes for better explanation. Speciﬁcally, after extracting the features 𝐅𝑞 or 𝐅𝑔𝑖 , our proposed
model will further infer more information about the identity from our
model’s ‘knowledge and experience’ according to 𝐅𝑞 or 𝐅𝑔𝑖 by

𝐅𝐾𝑞 = 𝑓𝑧 (𝐅𝑞 , 𝐊𝐄, 𝛽), 𝐅𝐾𝑔𝑖 = 𝑓𝑧 (𝐅𝑔𝑖 , 𝐊𝐄, 𝛽),

(3)

where 𝑓𝑧 (∗, 𝛽) denotes an inference function with its parameters 𝛽. After
that, our proposed model will further fuse those inferred features and
the features extracted from the input images, i.e.,

̂
𝐅𝑞 = ℎ(𝐅𝑞 , 𝐅𝐾𝑞 , 𝛾), ̂
𝐅𝑔𝑖 = ℎ(𝐅𝑔𝑖 , 𝐅𝐾𝑔𝑖 , 𝛾),

(4)

where ℎ(∗, 𝛾) denotes a fusion function with its parameters 𝛾. Accordingly, our proposed model will achieve person ReID via optimizing the
following objective function

𝑖∗ = 𝑎𝑟𝑔𝑚𝑎𝑥𝑖=1,2,…,𝐾𝑔 𝑆𝑖𝑚(̂
𝐅𝑞 , ̂
𝐅𝑔𝑖 ).

(5)

Accordingly, Fig. 2 illustrates the diagram of our proposed Crossmodality Knowledge and Experience Transformer (CKT), which mainly
consists of a VI-ReID subnetwork (corresponding to 𝑓 (∗)), a Knowledge
and Experience Transformer (corresponding to 𝑓𝑧 (∗)) and a fusion module (corresponding to ℎ(∗)).
Speciﬁcally, the VI-ReID subnetwork begins by extracting modalityshared features from input RGB or IR images via two modality-speciﬁc

Fig. 2. Illustration of the proposed model.
3

Pattern Recognition 172 (2026) 112553

N. Huang et al.

feature extraction modules, one for RGB images and one for IR images. Those extracted modality-speciﬁc features are then integrated into
a shared feature extraction module for projecting them into the same
features space to extract their modality-shared features. After that, the
resulting modality-shared features will be fed into the Knowledge and
Experience Transformer (KET) for inferring more modality-shared information about the individual. Finally, the modality-shared features
extracted from input images and their corresponding inferred features
are combined to achieve VI-ReID. We will describe their details in the
following contents.

the proposed VI-ReID subnetwork. Suppose that the training set exists
𝑁 person individuals and captures 𝐾 RGB images and 𝐾 IR images for
each individual. The two loss functions are computed as follows.
The identity loss aims to facilitate the extracted features be identity
discriminative, which is expressed by

Like most current VI-ReID models, our proposed VI-ReID subnetwork
also intends to extract person-related modality-shared features for identifying diﬀerent individuals. As depicted in Fig. 2, it mainly comprises
two modality-speciﬁc feature extraction modules and a modality-shared
feature extraction module.

𝐿ℎ𝑐_𝑡𝑟𝑖 (𝐅𝑠,𝑅 , 𝐅𝑠,𝐼 ) =

(8)

𝑁
∑
[𝜌 +||𝐜𝑛𝑠,𝑅 − 𝐜𝑛𝑠,𝐼 ||2 − 𝑚𝑖𝑛 ||𝐜𝑛𝑠,𝑅 − 𝐜𝑗𝑠,𝐿 ||2 ]+
𝐿∈{𝑅,𝐼}

𝑛=1

+

𝑗≠𝑛

𝑁
∑

[𝜌 + ||𝐜𝑙𝑠,𝐼 − 𝐜𝑙𝑠,𝑅 ||2 −

𝑙=1

𝑚𝑖𝑛

𝐿2 ∈{𝑅,𝐼}

||𝐜𝑙𝑠,𝐼 − 𝐜𝑗𝑠,𝐿 ||2 ]+ ,
2

(9)

𝑗≠𝑙

where 𝜌 is a pre-deﬁned margin. 𝐜𝑛𝑠,𝑅 and 𝐜𝑛𝑠,𝐼 denote the feature centers
of the RGB features and IR features for the 𝑛-th identity, respectively,
which are computed by

3.2.1. Modality-speciﬁc feature extraction module
RGB and IR images have signiﬁcant diﬀerences in their imaging
mechanisms, resulting in heterogeneous modality-speciﬁc information
in their low-level features, such as textures and colors in RGB images
and temperatures and materials in IR images. To tackle such an issue,
we employ two modality-speciﬁc feature extraction modules, shown in
Fig. 2, to extract lower-level modality-speciﬁc features from the input
RGB and IR images. These modules have the same structures as the ﬁrst
three blocks of Res-Net50 [16], but with separate network parameter
optimization to capture distinct modality-speciﬁc information. This allows us to extract two types of modality-speciﬁc features, i.e., 𝐅𝑅 from
RGB images, and 𝐅𝐼 from IR images.

𝐜𝑛𝑠,𝑅 =

𝐾
𝐾
1 ∑ 𝑛,𝑘 𝑛
1 ∑ 𝑛,𝑘
𝐅𝑠,𝑅 , 𝐜𝑠,𝐼 =
𝐅 ,
𝐾 𝑘=1
𝐾 𝑘=1 𝑠,𝐼

(10)

where 𝐅𝑛,𝑘
and 𝐅𝑛,𝑘
are the extracted RGB and IR features, respectively,
𝑠,𝑅
𝑠,𝐼
from the 𝑘-th image of the 𝑛-th identity. Therefore, the total loss for
training the VI-ReID subnetwork is expressed by

𝐿𝑉 𝐼𝑁 = 𝐿𝑖𝑑 (𝑆𝑅 , 𝑆𝑝 ) + 𝐿𝑖𝑑 (𝑆𝐼 , 𝑆𝑝 ) + 𝐿ℎ𝑐_𝑡𝑟𝑖 (𝐅𝑠,𝑅 , 𝐅𝑠,𝐼 ),

(11)

where 𝑆𝑝 denotes their ground true values.

3.2.2. Modality-shared feature extraction module
As previously stated, RGB images and IR images have large modality
discrepancies, which will cause their modality-speciﬁc features to belong to distinct feature spaces. In order to tackle this issue, as depicted
in Fig. 2, a parameter-shared module is further utilized to extract shared
features from the extracted modality-speciﬁc features 𝐅𝑅 and 𝐅𝐼 . More
speciﬁcally, the module ﬁrst projects these modality-speciﬁc features
onto the same feature spaces and subsequently extracts their modalityshared information in the shared feature space, as has been commonly
implemented in existing models.
Speciﬁcally, the shared module adopts a convolutional structure
identical to that of the last two blocks of Res-Net50 [16]. Accordingly, it
extracts two levels of modality-shared features (𝐅4𝑠,𝑚 and 𝐅5𝑠,𝑚 ) by passing the modality-speciﬁc features 𝐅𝑚 through the shared module. Here,
𝑚 ∈ {𝑅, 𝐼} implies the features originating from RGB and IR images,
respectively.
Then, given that the features at diﬀerent levels encompass diverse
types of modality-shared information, which may complement each
other, the modality-shared feature extraction module further integrates
the two levels of modality-shared features 𝐅4𝑠,𝑚 and 𝐅5𝑠,𝑚 to generate the
ﬁnal modality-shared person features 𝐅𝑠,𝑚 ∈ 𝑅1×𝐷 for VI-ReID. This procedure can be mathematically represented as follows:

3.3. Knowledge and experience transformer
On top of the extracted modality-shared features, the proposed
Knowledge and Experience Transformer (KET) aims to imitate the human visual systems for inferring more modality-shared information according to its ‘knowledge and experience’. As shown in Fig. 2, it contains a knowledge module for establishing our model’s ‘knowledge and
experience’, and a reasoning module for inferring more modality-shared
information according to the input features. The details about the two
modules are introduced as follows.
3.3.1. Knowledge module
The knowledge module establishes our model’s ‘knowledge and experience’ via the following two steps.
Establishing our model’s memories: Considering that the feature
center of each identity can well represent those dicriminative information of this identity, the knowledge module employs a feature matrix
𝐌 ∈ 𝑅𝑁×𝐷 (also called memory features) as its memory to store the
modality-shared feature center of each identity. Here, 𝑁 denotes the
number of person identities in the training set and 𝐷 denotes the feature channels of each center. Moreover, the features in the 𝑛-th row
of 𝐌 ∈ 𝑅𝑁×𝐷 , i.e., 𝐌𝑛 ∈ 𝑅1×𝐷 , represent the feature center of the 𝑛-th
person identity. Generally speaking, the memory features 𝐌𝑛 can be
computed by

(6)

where Cat(∗) denotes the concatenation and GAP(∗) denotes the global
average pooling. Moreover, a classiﬁer will be performed on the
modality-shared features 𝐅𝑠,𝑚 in the training stage to predict the identities of input images, i.e.,

𝑆𝑚 = Linear(𝐅𝑠,𝑚 , 𝛼𝑠 ),

𝑁

where 𝑝 is the predicted value and 𝑦 is its ground truth value. While,
the triplet hetero center loss 𝐿𝑡𝑟𝑖_ℎ𝑐 aims at shortening the feature centers of the same identity and enlarging the feature centers of diﬀerent
identities, i.e.,

3.2. VI-ReID subnetwork

𝐅𝑠,𝑚 = Cat(GAP(𝐅4𝑠,𝑚 ), GAP(𝐅5𝑠,𝑚 )),

1 ∑∑
𝑦 Log(𝑝𝑘,𝑐 ),
𝐾 𝑘=1 𝑐=1 𝑘,𝑐
𝐾

𝐿𝑖𝑑 (𝑦, 𝑝) = −

𝐌𝑛 =

𝐾 (
)
1 ∑ 𝑛,𝑘
𝐅 + 𝐅𝑛,𝑘
,
𝑠,𝐼
2𝐾 𝑘=1 𝑠,𝑅

(12)

where 𝐾 denotes the number of the RGB/IR images for each identity.
𝐅𝑛,𝑘
and 𝐅𝑛,𝑘
denote the features extracted from the 𝑘-th RGB and IR im𝑠,𝑅
𝑠,𝐼
ages of the 𝑛-th idetity, respectively, by using the VI-ReID subnetwork.
However, this cannot work well in an end-to-end learning way, since the
features 𝐅𝑛,𝑘
and 𝐅𝑛,𝑘
are changed with diﬀerent epoches. Alternatively,
𝑠,𝑅
𝑠,𝐼
our proposed KET directly learns the memory features 𝐌 by virtue of

(7)

where 𝑆𝑚 denotes the predicted probability of its identity. Linear(∗, 𝛼𝑠 )
denotes a linear function with its parameters 𝛼𝑠 for classiﬁcation.
Loss functions: In this stage, two loss functions, i.e., an identity loss
𝐿𝑖𝑑 and a triplet hetero center loss 𝐿𝑡𝑟𝑖_ℎ𝑐 , are employed for training
4

Pattern Recognition 172 (2026) 112553

N. Huang et al.

3.3.2. Reasoning module
After establishing our model’s ‘knowledge and experience’, our
model can infer more modality-shared information by using a novel reasoning module on top of the modality-shared features extracted from the
input images. As shown in Fig. 4, in the reasoning module, several reasoning blocks are employed to infer more discriminative information
from the knowledge features 𝐊𝐍 ∈ 𝑅𝑁×𝐷 for VI-ReID via the modalityshared features 𝐅𝑠,𝑚 ∈ 𝑅1×𝐷 extracted from the input images.
Here, we take the ﬁrst reasoning block as the example to introduce its
details. Speciﬁcally, in the ﬁrst reasoning module, the input features 𝐅𝑠,𝑚
are taken as the query features 𝐅𝑞,𝑚,1 , which, together with the knowledge features 𝐊𝐍, will be ﬁrst projected into the same feature space via
two linear functions, i.e.,

Fig. 3. Illustration of the knowledge module.

two proposed loss functions, i.e., a feature aggregation loss 𝐿𝐹 𝐴 and a
feature center aggregation loss 𝐿𝐹 𝐶𝐴 .
Speciﬁcally, the feature aggregation loss 𝐿𝐹 𝐴 aims to pull all features
of the same identity close to their corresponding feature center in the
memory features 𝐌, i.e.,

𝐿𝐹 𝐴 =

𝑁 𝐾 ( (
)
(
))
1 ∑∑
𝑛,𝑘
L2 𝐅𝑛,𝑘
−
𝐌
+
L
𝐅
−
𝐌
,
𝑛
2
𝑛
𝑠,𝑅
𝑠,𝐼
2𝑁𝐾 𝑛=1 𝑘=1

̂ 𝑚,1 = Linear(𝐅𝑞,𝑚,1 , 𝛽𝑞 ), 𝐊
̂ 1 = Linear(𝐊𝐍, 𝛽𝑘 ),
𝐐
𝑚,1
1

̂ 𝑚,1 ∈ 𝑅1×𝐷 and 𝐊
̂ 1 ∈ 𝑅𝑁×𝐷 denote the query features and the
where 𝐐
key features, respectively. 𝑚 ∈ {𝑅, 𝐼} denotes the features from diﬀerent modalities. Meanwhile, the knowledge features 𝐊𝐍 will be further
̂1 ∈ 𝑅𝑁×𝐷 by another linear function,
transferred into its value features 𝐕
i.e.,

(13)

̂1 = Linear(𝐊𝐍, 𝛽𝑣 ).
𝐕
1

where L2 (∗) denotes the mean square error function. It should be noted
that the memory features 𝐌 are randomly initialized by Xavier initialization [17]. While, the feature center aggregation loss 𝐿𝐹 𝐶𝐴 aims to
make the feature center of each modality from the same identity close
to their feature center in the memory features 𝐌, i.e.,
𝑁 ( (
)
(
))
1 ∑
𝐿𝐹 𝐶𝐴 =
L 𝐂𝑛 − 𝐌𝑛 + L2 𝐂𝑛𝑠,𝐼 − 𝐌𝑛 ,
2𝑁 𝑛=1 2 𝑠,𝑅

𝐾
𝐾
1 ∑ 𝑛,𝑘 𝑛
1 ∑ 𝑛,𝑘
𝐅𝑠,𝑅 , 𝐂𝑠,𝐼 =
𝐅 .
𝐾 𝑘=1
𝐾 𝑘=1 𝑠,𝐼

̂ 𝑚,1 𝐊
̂𝑇
𝐐
𝐰𝑚,1 = Sof tmax( √ 1 ).
𝐷

(14)

̂1 .
𝐅𝑜,𝑚,1 = 𝐰𝑚,1 𝐕

(21)

Furthermore, the output features 𝐅𝑜,𝑚,1 from the ﬁrst reasoning block
will be taken as the new query features, which, together with the knowledge features 𝐊𝐍, will be further fed into the next reasoning block for
inferring more modality-shared information. After that, the output features of the last reasoning block will be taken as our ﬁnal inferred features 𝐅𝑖𝑛𝑓 ,𝑚 ∈ 𝑅1×𝐷 . Finally, an identity classiﬁer will be further performed on those inferred features 𝐅𝑖𝑛𝑓 ,𝑚 to predict their identities, i.e.,

(15)

𝑆𝑖𝑛𝑓 ,𝑚 = Linear(𝐅𝑖𝑛𝑓 ,𝑚 , 𝛼𝑖𝑛𝑓 ),

(22)

where 𝑆𝑖𝑛𝑓 ,𝑚 denotes the predicted probability of their identities.
Linear(∗, 𝛼𝑖𝑛𝑓 ) denotes a linear layer with its parameters 𝛼𝑖𝑛𝑓 for classiﬁcation.
Moreover, the triplet hetero center loss and the identity loss are also
performed on the inferred features 𝐅𝑖𝑛𝑓 ,𝑚 to make them be identitydiscriminative, i.e.,

𝐐𝟏 = Linear(𝐌, 𝛼𝑞1 ), 𝐊𝟏 = Linear(𝐌, 𝛼𝑘1 ), 𝐕𝟏 = Linear(𝐌, 𝛼𝑣1 ),
𝐐𝟏 𝐊𝟏
𝐅𝑎𝑡𝑡,1 = Sof tmax( √
)𝐕𝟏 ,
𝐷

(20)

Finally, the ouput features 𝐅𝑜,𝑚,1 ∈ 𝑅1×𝐷 of the ﬁrst reasoning block will
be computed by

By virtue of the two loss functions, the memory features 𝐌 can automatically store the information of each identity within the training set
in the training stage.
Establishing our model’s ‘knowledge and experience’: As shown
in Fig. 3, the knowledge module will further feed those memory features
𝐌 into several stacked Transformer blocks to explore their relations and
transfer the memories into the model’s ‘knowledge and experience’.
Here, we take the ﬁrst Transformer block as an example for the introduction. Speciﬁcally, the input memory features 𝐌 will be ﬁrst fed
into a self-attention layer to explore the interactions across the features
of diﬀerent identities, i.e.,

𝑇

(19)

Then, the simility weights 𝐰𝑚,1 ∈ 𝑅1×𝑁 between the query features 𝐅𝑞,𝑚,1
and the knowledge features 𝐊𝐍 are computed by

where 𝐂𝑛𝑠,𝑅 and 𝐂𝑛𝑠,𝐼 denote the feature centers of the RGB images and
the IR images for the 𝑛-th identity, respectively, which are computed by
𝐂𝑛𝑠,𝑅 =

(18)

𝐿𝐾𝐸𝑇 =𝐿𝑖𝑑 (𝑆𝑖𝑛𝑓 ,𝑅 , 𝑆𝑝 ) + 𝐿𝑖𝑑 (𝑆𝑖𝑛𝑓 ,𝐼 , 𝑆𝑝 ) + 𝐿ℎ𝑐_𝑡𝑟𝑖 (𝐅𝑖𝑛𝑓 ,𝑅 , 𝐅𝑖𝑛𝑓 ,𝐼 ).
(16)

(23)

Accordingly, the total loss 𝐿𝑡𝑜𝑡 for training our proposed model is expressed by
𝐿𝑡𝑜𝑡 = 𝐿𝑉 𝐼𝑁 + 𝐿𝐹 𝐴 + 𝐿𝐹 𝐶𝐴 + 𝐿𝐾𝐸𝑇 .

where 𝐐𝟏 , 𝐊𝟏 and 𝐕𝟏 ∈ 𝑅𝑁×𝐷 denote the projected query features, key
features and value features, respectively, in the Transformer block.
𝐅𝑎𝑡𝑡,1 ∈ 𝑅𝑁×𝐷 denotes the output features of the self-attention layer in
the ﬁrst Transformer block. Here, 𝐐𝟏 , 𝐊𝟏 , 𝐕𝟏 and 𝐅𝑎𝑡𝑡,1 ∈ 𝑅𝑁×𝐷 . Then,
the output features 𝐅𝑎𝑡𝑡,1 are further fed into a normalization layer and
a feed-forward layer, i.e.,
(
(
))
𝐅𝐾𝑁,1 = FFN Norm 𝐅𝑎𝑡𝑡,1 ,
(17)

(24)

3.4. Fusion module
A simple fusion module is further designed to fuse the features 𝐅𝑠,𝑚
extracted from the input images and their inferred features 𝐅𝑖𝑛𝑓 ,𝑚 for VIReID. Speciﬁcally, the ﬁnal person features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 for VI-ReID will be
obtained by feeding the features 𝐅𝑠,𝑚 and 𝐅𝑖𝑛𝑓 ,𝑚 into the fusion module,
where they are fused by

where 𝐅𝐾𝑁,1 denotes the transferred knowledge features in the ﬁrst
Transformer block. FFN(∗) denotes a feed-forward layer. Norm(∗) denotes a normalization layer.
Accordingly, the features 𝐅𝐾𝑁,1 will be fed into the next Transformer
block and the output features of the last Transformer block are taken
as the ﬁnal knowledge features 𝐊𝐍 ∈ 𝑅𝑁×𝐷 . By doing so, the memories of our model will be transferred as our model’s ‘knowledge and
experience’.

𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 = 𝐅𝑠,𝑚 + 𝐅𝑖𝑛𝑓 ,𝑚 .

(25)

Fig. 5 shows the distributions of the modality-shared features 𝐅𝑠,𝑚
and the ﬁnal fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 . From Fig. 5, it can be easily seen
that the ﬁnal fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 of the same identities are more compact than the modality-shared features 𝐅𝑠,𝑚 . This indicates that those
inforamtion inferred from our model’s “knowledge and experience” can
enhance the discriminability of the ﬁnal modality-shared features.
5

Pattern Recognition 172 (2026) 112553

N. Huang et al.

Fig. 4. Illustration of the reasoning module.

Fig. 5. Distributions of the modality-shared features 𝐅𝑠,𝑚 (a) and the ﬁnal fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 (b) from diﬀerent identities. Diﬀerent colors denote diﬀerent
identities.

4. Experiments

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 \"'Knowledge and experience' for visible-infrared person re-identification.pdf\" - | sed -n '1,160p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 107ms:
Pattern Recognition 172 (2026) 112553

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

‘Knowledge and experience’ for visible-infrared person re-identiﬁcation
Nianchang Huang a,b , Jialiang Wang a,b , Qiang Zhang

a,b,∗, Jungong Han c , Jin Huang a

a State Key Laboratory of Electromechanical Integrated Manufacturing of High-Performance Electronic Equipment, Xidian University, Xi’an, 710071, Shaanxi, China
b

Center for Complex Systems, School of Mechano-Electronic Engineering, Xidian University, Xi’an, 710071, Shaanxi, China

c Department of Automation, Tsinghua University, Beijing, 100084, China

a r t i c l e

i n f o

Keywords:
Cross-modality person re-identiﬁcation
‘Knowledge and experience’
Modality-invariant person information

a b s t r a c t
Most existing models mainly focus on learning those modality-shared features unique to individuals for crossmodality Visible-Infrared Person Re-Identiﬁcation (VI-ReID). Diﬀerently, the human visual systems identify different persons by using not only the information within the input images but also their learned knowledge and
experience in history. Speciﬁcally, given a query image, our visual systems will ﬁrst extract lots of information
about the person from the given image and then deduce more information about this person with the aid of our
knowledge and experience for searching this person from a set of images. Inspired by that, we will propose a
novel Cross-modality Knowledge and Experience Transformer (CKT) for cross-modality VI-ReID in this paper.
In detail, our CKT will ﬁrst memorize the discriminative modality-shared information of each identity in the
training set, and then convert them into our model’s ‘knowledge and experience’. Accordingly, in addition to the
information within those input images, our CKT can infer more modality-invariant person information from its
‘knowledge and experience’ according to their corresponding modality-shared information for VI-ReID. Comprehensive experiments show that our proposed model achieves signiﬁcant performance improvements over most
state-of-the-art ones by combining the information within the input images and their induced information from
our model’s ‘knowledge and experience’.

1. Introduction
Person Re-Identiﬁcation (ReID) aims to match the given identity in
a query image from a set of gallery images captured by disjoint surveillance camera views [1]. Based on the assumption that the query and
gallery images are captured under good lighting conditions, most existing methods [2] mainly match a visible image from a visible image
based gallery set, i.e., VV-ReID. Although they have achieved great success, these VV-ReID methods usually rely on bright lighting conditions,
thus restricting their applicability in real-life settings. To overcome this
limitation, many surveillance systems now extensively incorporate thermal infrared (IR) cameras with RGB cameras to capture valuable information in low-light environments. As a result of that, cross-modality
Visible-Infrared (VI-) ReID has gained a growing interest recently.
Compared with VV-ReID, the major challenges of VI-ReID is the
modality discrepancy caused by the intrinsically distinct imaging mechanisms of RGB and IR cameras [3–5], which will lead to heterogenous
distributions between single-modality RGB and IR features. The major
idea of addressing such an issue is to extract those common features (i.e.,
modality-shared features) within RGB and IR images for identifying dif-

ferent individuals. However, solely relying on such a limited amount of
modality-shared person information between RGB and IR images may
hinder VI-ReID systems to reach the next milestone.
Alternatively, we will overcome the cross-modality VI-ReID issue by
imitating some characteristics of the human visual system. Speciﬁcally,
if a person wants to tell whether the identities in two images are the
same or not, he (she) uses not only the information obtained from the
two images but also his (her) learned knowledge and experience in the
past. For example, as shown in Fig. 1, when we judge whether person
A in the RGB image and person B in the IR image are the same person
or not, we may ﬁrst extract their modality-invariant person attributes
from the corresponding images, e.g., person A has thin arms and a thin
neck, but person B has a thick neck and a large waist. Then, given these
attributes extracted from input images, alongside our knowledge and
experience, we may infer more modality-invariant attributes about the
two persons in other views or poses, e.g., person A may have a round
face and a slim ﬁgure, and person B may have a stubby neck and a plump
body. By fusing these extracted and inferred person attributes, we will
make our ﬁnal judgment. In this process, besides the person information
extracted from the input images, our knowledge and experience also

∗ Corresponding author.

E-mail addresses: huangnianchang@xidian.edu.cn (N. Huang), Jialiang_Wang@stu.xidian.edu.cn (J. Wang), qzhang@xidian.edu.cn (Q. Zhang),
jungonghan77@gmail.com (J. Han), jhuang@mail.xidian.edu.cn (J. Huang).
https://doi.org/10.1016/j.patcog.2025.112553
Received 30 November 2023; Received in revised form 19 August 2025; Accepted 3 October 2025
Available online 8 October 2025
0031-3203/© 2025 Published by Elsevier Ltd.

Pattern Recognition 172 (2026) 112553

N. Huang et al.

Fig. 1. General process of identifying diﬀerent identities in the human visual system.

play important roles, since they provide more modality-invariant person
information for better identifying diﬀerent persons.
Inspired by that, a novel Cross-modality Knowledge and Experience
Transformer (CKT) will be designed in this paper for VI-ReID, which will
employ its ‘knowledge and experience’ to infer more modality-invariant
information for VI-ReID, in addition to some information extracted from
the input images. Speciﬁcally, as the human visual systems, which ﬁrst
capture information from the given images, our proposed CKT will also
ﬁrst extract those discriminative modality-shared features from the images of diﬀerent modalities by employing a cross-modality VI-ReID subnetwork. On top of that, our proposed CKT will imitate the human visual
system to infer more modality-invariant information for VI-ReID by using its ‘knowledge and experience’.
To this end, we observe that the information within all the training
images not only contains rich discriminative person features/attributes
of a speciﬁc person identity, but also contains the relations among person features/attributes across diﬀerent person identities. This means
that we may infer more information about this person from some existing information by virtue of those learned relations among diﬀerent
person features/attributes. For example, if a person looks overweight in
the given image, it may be inferred that the person probably carries a
heavy belly rather than a ﬂat stomach by exploring the relations among
the features from diﬀerent identities, i.e. such relations may be obtained
by comparing those features from a set of overweight persons and those
from a set of thin persons. Based on such an observation, a novel Knowledge and Experience Transformer (KET) will be further designed in our
proposed CKT, which will transfer the information within the training
set into our VI-ReID model’s ‘knowledge and experience’ and employ
them for inferring more modality-invariant person information based
on the features from the given images as that in the human visual system.
More speciﬁcally, the KET will ﬁrst save the modality-shared feature center of each identity in the training set as our model’s memory,
just like a person’s memory. Here, each modality-shared feature center can represent some unique characteristics of its corresponding identity. Accordingly, the memory features can well save the information of
diﬀerent identities in the training set. Then, a knowledge module will
be designed in the proposed KET for establishing its ‘knowledge and
experience’ by exploring the interactions across the feature centers of
diﬀerent identities. Similar to that in the human visual system, these
‘knowledge and experience’ will enable our model to infer more hidden modality-invariant person information, given such modality-shared
features extracted from the input images. Therefore, the proposed KET
will further design a novel reasoning module, which will utilize the information extracted from the images of one identity for inferring more

modality-shared information about the identity from the ‘knowledge
and experience’ of our proposed CKT. By doing so, our proposed model
can combine the modality-shared features from the input images and
those inferred modality-shared features from its ‘knowledge and experience’ for VI-ReID in the testing stage.
The main contributions of our work are as follows:
(1) Inspired by the human visual system, we endue our proposed VIReID model with the ability to learn its ‘knowledge and experience’ by
exploring the relations among all the identities in the training set. By
virtue of the learned ‘knowledge and experience’, our proposed model
can infer more modality-shared information, thereby helping to reach
new performance milestone for VI-ReID models.
(2) We propose a novel Cross-modality Knowledge and Experience Transformer (CKT) to implement the above idea. Speciﬁcally, the
modality-shared information extracted from each individual in the training set is memorized and transferred into our model’s ‘knowledge and
experience’ via a novel Transformer to infer more modality-invariant
information for VI-ReID during the testing stage.
(3) Extensive experiments prove that our proposed model achieves
a substantial improvement over existing algorithms on the widely-used
SYSU-MM01 and datasets
In the following contents, we will ﬁrst provide a brief overview for
previous research on ReID and VI-ReID, respectively, in Section 2. Then,
we will elaborate our proposed model’s design. After that, we report
several experiments in Section 4 to validate our proposed model. Finally,
we provide a brief conclusion for this study.
2. Related work
2.1. Single-modality VV-ReID
Earlier solutions of VV-ReID mainly rely on some hand-crafted features and probabilistic patch matching algorithms [6]. Recently, with
the remarkable achievements in computer vision, Deep Convolutional
Neural Network (DCNN) based VV-ReID models have taken over as the
mainstream. Generally speaking, there are currently two major categories of DCNN based VV-ReID models: representation learning based
models [7] and metric learning based models [2]. Representation learning based models [7] handle resolution, lighting, view or pose changes
by extracting some invariant features from the inputs. For instance, in
[7], a Pose-Guided Feature Alignment (PGFA) model was introduced
for solving the occluded ReID problem by utilizing pose landmarks to
extract useful information and reduce occlusion noise. Metric learningbased models [2] aim to learn discriminative metric spaces for ReID by
minimizing the intra-person distance and maximizing the inter-person
2

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 \"3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf\" - | sed -n '1,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 306ms:
12830

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025

3D-Aided Pedestrian Representation Learning for
Video-Based Person Re-Identification
Guquan Jing , Graduate Student Member, IEEE, Peng Gao , Graduate Student Member, IEEE, Yujian Lee,
Yiyang Hu, and Hui Zhang , Senior Member, IEEE

Abstract—Video-based person re-identification (Re-ID) aims
to match the target pedestrian from video sequences. Recent
methods perform frame-level feature extraction followed by
temporal aggregation to obtain video representations. However,
they pay insufficient attention to the quality of frame-level
features, which suffer from issues including multi-frame misalignment, partial occlusion and appearance confusion. People
live in a 3D space. 3D pedestrian representations can provide
rich geometric information and shape cues that offer promising
solutions to these challenges in video-based Re-ID. To mitigate these issues, this paper proposes a 3D-Aid Pedestrian
Representation Learning (3DAPRL) network, which introduces
3D modality to video-based Re-ID. Specifically, two novel modules
are designed, i.e., the Cross-Modal Fusion (CMF) module and
the Shape-aware Spatial-Temporal Interaction (SSTI) module,
to enhance pedestrian representation learning. The CMF module generates discriminative fusion representations by utilizing
3D pedestrian data, while the SSTI module learns spatialtemporal 3D shape representation which are distinguishable for
finding the target pedestrian in video scenarios. Both features
generated from the CMF and SSTI modules contribute to
the final video representation. Extensive experiments on four
challenging video-based Re-ID datasets demonstrate that our
3DAPRL network reaches better performance than state-of-thearts methods.
Index Terms—Video-based person re-identification, crossmodal feature fusion, 3D pedestrian representation, video
representation.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) [1], [2] is a critical
task in intelligent surveillance, aiming to identify specific
pedestrians across non-overlapping cameras. Early studies

Received 30 November 2024; revised 19 May 2025; accepted 1 July 2025.
Date of publication 7 July 2025; date of current version 8 December 2025.
This work was supported in part by Guangdong Provincial Key Laboratory of
Interdisciplinary Research and Application for Data Science, Beijing Normal
University-Hong Kong Baptist University United International College (BNUHKBU) United International College under Grant 2022B1212010006; in part
by the Natural Science Foundation of China under Grant 62076029; in part
by the National Key Research and Development Program of China under
Grant 2022YFE0201400; and in part by Guangdong Science and Technology
Department under Grant 2022B1212010006. This article was recommended
by Associate Editor Y. Wang. (Corresponding author: Hui Zhang.)
Guquan Jing, Peng Gao, and Yiyang Hu are with the Department
of Computer Science, Hong Kong Baptist University, Hong Kong, SAR,
China, and also with Beijing Normal University-Hong Kong Baptist University (BNU-HKBU) United International College, Zhuhai 519000, China
(e-mail: guquanjing1@gmail.com; gaopeng1225@gmail.com; yiyanghu124@
gmail.com).
Yujian Lee and Hui Zhang are with Guangdong Provincial Key Laboratory of Interdisciplinary Research and Application for Data Science,
BNU-HKBU United International College, Zhuhai 519087, China (e-mail:
r130034019@mail.uic.edu.cn; amyzhang@uic.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3586808

[3], [4], [5], [6], [7], [8] focused on image-based Re-ID,
i.e., identifying pedestrians from single-shot images. With
the development of video devices, video-based Re-ID (video
Re-ID) [9], [10], [11] that uses multi-frame image sequences
as input has attracted widespread attention. Compared to
image-based Re-ID, video Re-ID offers a more promising scenario for capturing richer temporal and spatial
information.
Early video Re-ID methods primarily adopt models
designed for video recognition (including RNNs [12], [13],
[14], [15], 3D CNNs [16], [17], [18], optical flow [19], [20],
etc.) to learn video temporal information directly, which are
not well-suited for video Re-ID tasks. Recent studies [9],
[21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31],
[32], [33], [34] have made significant progress by proposing
novel modules to aggregate extracted frame-level features
temporally; however, they fail to adequately consider the
quality of the frame-level features. Multi-frame misalignment,
partial occlusion and appearance confusion frequently occur
in video data, compromising frame-level features and causing
the generation of suboptimal video representations. In specific,
some of these methods [35], [36], [37], [38], [39] retain the
local information of pedestrians through designed modules,
such as those that account for the importance of each feature’s
spatial position [39]. Attention mechanisms or local pooling
are used to highlight relevant parts of the pedestrian in these
methods. However, partial occlusion introduces spatial noise,
which integrates irrelevant information into frame-level features and degrades the overall performance of these methods.
In addition, spatial misalignment across frames often yields
uneven feature quality, which in turn can induce temporal
misalignment during feature aggregation. Other methods [21],
[31], [32], [33] attempt to derive global representations by
pooling frame-level features [9]; some [31] also refine framelevel features using these global cues. Nevertheless, their
performance is significantly undermined by the aforementioned multi-frame misalignment and partial occlusion. These
issues can corrupt global features and make it difficult to
spatially emphasize the target pedestrian, leading to a loss of
discriminative spatial information and causing temporal misalignment. Meanwhile, both methods struggle with appearance
confusion, where negative samples with similar appearance
are misidentified as the target pedestrian, ultimately leading to non-discriminative video representations. Figure 1 (a)
shows these challenges in video Re-ID, where two pedestrians with similar appearance are affected by multi-frame

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID

Fig. 1. (a) Two pedestrian sequences from video Re-ID datasets exhibiting
similar appearance, with challenges including multi-frame misalignment (red
box) and partial occlusion (blue box). (b) An example of our 3D pedestrian representation including mesh, point cloud and colored point cloud.
(c) Overview of our proposed framework. Note that in (c), black lines/boxes
denote the basic structure of the previous methods. On this basis, red
lines/boxes represent our additional processes.

misalignment (red boxes) and partial occlusion (blue boxes). In
such scenarios, effective spatial emphasis is hindered directly,
leading to discriminative spatial information loss and temporal
misalignment.
The inherent limitations of relying solely on 2D image to
overcome these issues and other challenges in Re-ID motivate the exploration of auxiliary information. Recent studies
incorporate various modalities (such as text [30], events [28]
and 3D data [40], [41], [42], [43], [44], etc.). Among these,
3D data, such as meshes rendered by models like SMPL [45],
is particularly promising. It offers rich geometric information
and shape cues, which are valuable for extracting detailed and
robust pedestrian features. Motivated by this, Zheng et al. [40]
propose a 3D pedestrian feature generation module that utilizes
both 3D geometric information from reconstructed meshes and
2D appearance cues from images to enhance feature representation. While the extracted 3D feature contributes to pedestrian
representation, their methods lack sufficient consideration for
the effective fusion of cross-modal information, limiting the
potential of fused features. Specifically, their fusion strategies
project RGB information directly into 3D space [40] or simply
concatenate 2D image features with 3D features [41], [42],
leading to coarse feature aggregation and limited cross-modal
interaction. We consider that the main challenge in feature
fusion for Re-ID with 3D representations lies in the modality
gap between 2D images and 3D data, causing inconsistency
in cross-modal features. Such a feature inconsistency are
manifested in feature misalignment across dimensions and
the disparate information captured from different modalities.
On the one hand, the dimensional misalignment between
2D and 3D pedestrian features leads to the loss of crucial
information during the fusion process. On the other hand,
3D pedestrian feature could provide incomplete pedestrian
representation due to the coarse extraction process, resulting
in inconsistency with 2D features and yielding suboptimal
fusion outcomes. Some methods [41], [42] leverage the 3D
shape parameter included with SMPL model to obtain spatialtemporal pedestrian shape cues using attention mechanisms,
but only limited and coarse-grained information is extracted.

12831

In addition, although they utilize 3D information in video
cloth-changing Re-ID (VCCReID) scenarios, its potential in
video Re-ID benchmark remains underexplored.
To address the aforementioned problems, we propose a
novel 3D-Aid Pedestrian Representation Learning (3DAPRL)
network to incoperate 3D modality into video-based Re-ID
(see Figure 1 (b) for an example of the 3D pedestrian representation and (c) for a conceptual illustration of our framework).
Our network exploit a Cross-modal Fusion (CMF) module and a Shape-aware Spatial-Temporal Interaction (SSTI)
module to obtain accurate frame-level fusion representations
and spatial-temporal 3D shape representations, respectively.
Before that, we first convert 3D meshes to point clouds, and
then employ a pillar-based backbone network [46] to extract
pillar-based features (i.e., pseudo-image features), effectively
reducing the feature inconsistency between 2D and 3D modalities. The generated features contain pedestrian geometry,
appearance, and shape information, while being dimensionally aligned with image features. Besides, the pillar-based
backbone network captures the complete pedestrian geometry
in 3D space, enabling a comprehensive feature representation. Consequently, this step narrows the modality gap and
establishes a more consistent feature space for subsequent
processing stages in our framework. After obtaining pillarbased features, the CMF and SSTI modules are employed to
overcome challenges in video Re-ID. Detailed explanations of
these modules are provided below.
The CMF module is introduced to effectively mitigate the
impacts of multi-frame misalignment and partial occlusion.
It achieves this by adaptively fusing the pillar-based features
described above with image features. This fusion process
emphasizes crucial areas of the target pedestrian and integrates
essential geometric and shape cues. In detail, the fusion
strategy exploits deformable cross-attention [47] to learn
correlations between image and 3D modalities, generating
3D-enhanced features. Accurate pedestrian positions derived
from the mapping relationship between 2D images and
3D point clouds serve as reference points, guiding the model
to focus on pedestrian-relevant areas and improving crossmodal feature alignment in the latent space. Additionally, the
CMF module adaptively determines relevant regions for fusion
and capture the local context of pedestrians based on offsets
learned from pillar-based features. Moreover, we retain the
generated pillar-based feature for preserving 3D geometric
and shape cues. Thus, the proposed CMF module generates discriminative fusion pedestrian representations that fuses
frame-level features with 3D geometric and shape information,
mitigating the effects of multi-frame misalignment and partial
occlusion.
To further enhance model ability against these issues, and
in particular to address the challenge of appearance confusion,
it is necessary to utilize a new stable pedestrian characteristic,
i.e., video-level 3D shape. The pedestrian shape in 3D space
offers robustness against similar appearances [41], [42] and
provides a complementary shape representation. Our Shapeaware Spatial-Temporal Interaction (SSTI) module is therefore
proposed to effectively learn and leverage these discriminative 3D shape representations of the target pedestrian from

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

12832

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025

video sequences for improved identification. Specifically, the
3D shape parameters estimated by the SMPL model are
used as the initial input. Subsequently, we create temporal
and spatial streams in parallel to adopt feature contextualization along space and time axes the attention. The spatial
stream focuses on obtaining 3D shape information within each
frame, while the temporal stream models 3D shape variations
across frames. The extracted spatial and temporal features
are then fused to produce spatial-temporal representation.
Our SSTI module allows the network to fully exploit both
spatial and temporal dependencies. Its receptive field captures
spatial-temporal interactions. Hence, discriminative videolevel 3D shape representations are yielded, which contributes
to the final video representation. Extensive experiments on
four challenging video Re-ID datasets demonstrate that the
3DAPRL outperforms the state-of-the-art methods.
In summary, the contributions of this study are as follows:
• We propose a novel 3D-Aid Pedestrian Representation
Learning (3DAPRL) network that introduces 3D modality
to video-based Re-ID. To the best of our knowledge, this
is the first method to leverage 3D modality for addressing
video-based Re-ID (video Re-ID) task.
• We propose a Cross-modal Fusion (CMF) module to
fuse image features and pillar-based features. In the
CMF module, the pillar-based feature helps the model
to focus on the crucial area while providing pedestrian
geometric information and shape cues. Consequently, the
discriminative frame-level pedestrian feature is generated
to reduce the influence of the multi-frame misalignment
and partial occlusion.
• We design a Shape-aware Spatial-Temporal Interaction
(SSTI) module to extract distinctive spatial-temporal
3D shape representations, which contributes to the video
representation. It serves as a new pedestrian characteristic
to further mitigate the impact of appearance confusion,
multi-frame misalignment and partial occlusion.
• Extensive experiments on four widely used video Re-ID
datasets including MARS, iLIDS-VID, PRID-2011 and
DukeMTMC-VID demonstrate that our proposed method
outperforms the state-of-the-art methods.
II. R ELATED W ORKS
Related works are organized as follows. In Section II-A, we
provide a summary of recent video Re-ID methods and discuss
their limitations. Section II-B offers a detailed review of existing methods that incorporate 3D modality for person Re-ID.
We also emphasize the novelty of our approach and highlight
how it differs from these methods. Eventually, methods in the
3D feature learning are introduced in Section II-C.
A. Video-Based Person Re-ID
Remarkable progress has been made in video-based Re-ID
in recent years. Early studies exploit deep learning techniques
for video recognition task including optical flow [19], [20],
3D convolutions [16], [17], [48], [49] and recurrent models
[12], [13], [14], [15], [50] to obtain temporal features directly.
Chung et al. [19] proposed a Siamese network with two

separately streams to extract spatial and temporal features.
Specifically, one stream processes RGB images while the other
processes optical flow. McLaughlin et al. [12] employ CNNs
to extract image features, followed by RNNs for temporal
information extraction. To tackle the spatial misalignment
issue in methods using RNNs, Liao et al. [48] apply 3D convolutions to extract and aggregate spatial and temporal features.
Li et al. [49] propose a multi-scale 3D convolutional layer,
which offers a more compact and efficient solution compared
to traditional 3D convolutions. Despite effectively encoding
temporal information, the techniques used in these methods
are not specifically designed for video Re-ID. They also face
challenges such as multi-frame misalignment, partial occlusion, and appearance confusion in video datasets, which often
lead to performance degradation. Most of methods [1], [20],
[21], [22], [24], [27], [29], [31], [36], [38], [39], [51], [52],
[53], [54], [55], [56], [57], [58], [59], [60] focus on aggregate
frame-level feature temporally by using attention mechanisms
in video Re-ID to obtain discriminative video representations.
For instances, Li et al. [36] build the spatial and temporal
attention to extract useful information from visible body parts.
Wang et al. [21] introduced a pyramid structure that aggregates
spatial and temporal features by utilizing attention-based modules designed to capture spatial and temporal information. In
order to address the spatial misalignment in video-level data,
Chen et al. [27] propose an attention mechanism to interact
each frame with others in the video, enhancing both local
and global features. Despite the effective temporal aggregation of frame-level features using attention, challenges such
as multi-frame misalignment, partial occlusion, and appearance confusion still affect model performance. To mitigate
these issues, some methods incorporate additional modalities
such as event data [28] to provide supplementary information. However, the limited information provided by event
data is insufficient to resolve these challenges. To overcome
these limitations, we introduce a novel network incorporating
3D modality.
B. Person Re-ID With 3D Pedestrian Representation
Different from single-modality Re-ID, cross-modality
Re-ID [28], [40], [41], [43], [61], [62] focuses on retrieving
pedestrians using diverse modality representations, such as
3D pedestrian mesh. In the context of person Re-ID with
3D representation, existing approaches [40], [41], [43], [61],
[62], [63] primarily concentrate on extracting distinctive 3D
features. The incorporation of 3D modality is shown to
enhance the Re-ID performance by providing complementary
3D information, which positively impacts retrieval accuracy.
To obtain distinguishable 3D pedestrian features, some
methods use the SMPL model [45] to generate 3D representation. OGNet [40] first employs 3D representation for image
Re-ID by converting 2D pedestrian images into 3D space via
off-the-shelf 3D human reconstruction networks [64], [65],
[66]. It leverages a GNN structure to build the location
topology and extracts multi-scale semantic information of 3D
pedestrians. Liu et al. [62] reconstruct 3D clothed pedestrians
from images and learn the distinguished feature from the
3D clothed body, naked body and RGB information for

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID

long-term person Re-ID. Some works leverage 3D shape
parameters provided by the SMPL model. For instance,
Chen et al. [43] extract texture-insensitive 3D shape features
to improve image Re-ID and cloth-change Re-ID results. Similarly, Han et al. [41] utilize video-level 3D shape parameters,
incorporating both inter-frame and intra-frame cues from 3D
pedestrian mesh in video cloth-change scenario. Some other
methods propose 3D datasets for Re-ID, Guo et al. [63]
introduce a lidar-based Re-ID dataset containing 3D pedestrian
point clouds and propose a framework to generate comprehensive features from it. Wang et al. [61] propose 3D datasets
with created virtual 3D characters via proposed UV texture
mapping to generate visible clothed parts, and homogeneous
expansion is applied to predict invisible parts. Nevertheless,
most of these methods [40], [41], [43], [62] focus solely
on generating 3D features to represent the target pedestrian,
neglecting the fusion with image features and the specific
fusion strategies, which are crucial for combining complementary visual and geometric cues, leading to improved Re-ID
performance. Since the feature misalignment problem exists
between image and 3D features, generating an accurate and
comprehensive fused feature remains challenging. Our method
aims to utilize a pillar-based backbone and novel fusion
strategy to bridge the domain gap between image and 3D
modalities, thereby generating the distinctive fusion feature
for video-based Re-ID. We also leverage 3D shape parameter
to create the distinguishable video representation.
To the best of our knowledge, there exists no prior 3Dbased solution tailored for video Re-ID. To address this
gap, leveraging an off-the-shelf 3D reconstruction network is
necessary to generate reliable 3D pedestrian representations
as a foundation for our framework. In particular, HMR [64]
reconstructs comprehensive 3D human body meshes from
single RGB images using an adversarial framework trained
to distinguish between real and generated human shape and
pose parameters. Subsequent methods such as PiFu [67] and
LVD [66] can produce highly detailed 3D reconstructions,
including clothing [67] or precise hand surface [66]. Although
high accurate generation of PiFu and LVD, these methods are
not suitable for video Re-ID. The typically low resolution of
video frames in Re-ID datasets is often insufficient for these
advanced methods to reliably recover fine-grained details.
Consequently, any such details, if inaccurately reconstructed
from low-resolution input, are unpromising to serve as robust
identifying features for pedestrians. Therefore, our framework
utilizes HMR for 3D pedestrian generation. Although HMR
lacks the capacity to capture detailed clothing or precise hand
surface as achieved by methods such as PiFu and LVD, it
robustly provides comprehensive geometry and shape information from single images while remaining compatible with the
low-resolution frames typically exist in video Re-ID datasets,
which is suitable for our framework.

12833

method [68], [69] and voxel-based method [46], [70]. Pointbased method [68], [69] process the point cloud directly to
generate representations. PointNet [68] is a unified architecture
that extract point features by using multi-layer perceptron
(MLP) network and max-pooling. Building upon [68], PointNet++ enhances local feature extraction by aggregating local
feature across multiple layers, improving its ability to capture
fine-grained geometric details. Voxel-based method [46], [70],
[71] convert the point cloud into voxels or pillars. VoxelNet
[70] discretizes the point cloud into a structured voxel grid
and then employs 3D convolutional layers to extract geometric
features. In contrast, PointPillar [46] eliminates the need for
computationally expensive 3D convolutions by introducing a
novel pillar-based representation, which organizes the point
cloud into pillars that can be processed using 2D convolutional
layers.
III. M ETHODS
A. Preliminary - Pillar-Based 3D Backbone Network
To extract the geometric and shape information from 3D
pedestrian representations, an efficient 3D backbone network
is significant. Despite point-based backbone networks such as
PointNet++ [69] can extract the shape and semantic information of 3D point clouds, they often suffer from domain gaps
and feature inconsistencies when fused with 2D image features
due to misaligned feature dimensions. Thus, we adopt a pillarbased backbone network [46] without its detection head, which
not only efficiently encodes the full 3D pedestrian cloud but
also aligns the dimensional space of the extracted features
with that of image features. Following by the PointPillar [46],
given a point u with coordinates (x, y, z) and RGB value c
in the 3D pedestrian point cloud h, h is discretized into a
set of pillars that are voxels with unconstrained spatial range
in the z dimension. The distances from u to the center of
the pillar (∆x p , ∆y p ) and to the cluster center in the pillar
(∆xc , ∆yc , ∆zc ) are then calculated to enhance each point.
By discretizing points into pillars, a tensor with dimension
(D, P, N) is generated, where the dimension D consists of
the components (x, y, z, ∆x p , ∆y p , ∆xc , ∆yc , ∆zc , c), P denotes
the number of non-empty pillars and N denotes the number
of points per pillar. Afterwards, the tensor is processed by a
PointNet [68] to create a tensor with dimension (C, P, N), and
then encoded to dimensional size (C, P) by exploiting max
operation. Finally, all encoded tensor is sent to the positions
in the original pillar space by their indices and then generate a
pseudo-image with dimensional size (C, H, W). Consequently,
the 3D pedestrian is thereby encoded into the pseudo-image
which can be operated later by 2D convolutions. Here, the
feature extracted from the pseudo-image by 2D convolutions
is referred to the pillar-based feature.
B. Overview

C. 3D Feature Learning From Point Cloud
The point cloud serves as a data structure for representing
geometric information in 3D space. To extract features from
point clouds, two widely adopted methods are point-based

Video Re-ID aims to retrieve the same pedestrian from
video sequences. As shown in Figure 2, our 3D-Aid Pedestrian
Representation Learning (3DAPRL) network start with three
branches, i.e., the 2D branch, the 3D branch and the shape
branch. Each of the branches extracts their respective modal

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

12834

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025

Fig. 2. The architecture of our 3D-Aid Pedestrian Representation Learning (3DAPRL) network. We use three frames (L = 3) as an example. The 2D
and 3D backbone networks are adopted to obtain image and pillar-based features from video seuqences and colorful 3D SMPL mesh, respectively. The cross
modal fusion (CMF) module is then used to fuse these two information. Here, we use the second feature of the image and the mesh as an example in the CMF
module. Afterwards, the fusion feature is fed into Spatial-Temporal Aggregation (STA) module to obtain spatial-temporal feature. We also extract 3D shape
features from the 3D shape parameter by our proposed Shape-aware Spatial-Temporal Interaction (SSTI) module. Both features from CMF and SSTI modules
are used to build the video representation.

feature. For the 2D branch, given an input video sequence with
L
L frames VI = {Il }l=1
, it is first fed into the 2D image backbone
network:
F I = Θ2D (VI ),

(1)

where Θ2D is a image feature extractor such as ResNet-50
[72]. F I = {F I1 . . . F IL } is the 2D frame-level feature, where
F Il ∈ RC×H×W is the l-th frame feature (C, H, and W denote
its channel, height and width). For the 3D branch, we introduce
the 3D mesh in SMPL to alleviate the impact of multi-frame
misalignment, partial occlusion in video data. Since there
is no existing 3D video Re-ID solution, we use an off-theshelf 3D human reconstruction network [64] to obtain the
3D pedestrian mesh for each frame. The generated pedestrian
meshes are converted into 3D point clouds since each vertex
of the mesh corresponds to a 3D point. Additionally, the RGB
color values from the 2D image are back-projected onto the
corresponding 3D points to integrate appearance information
with geometric representation. To elaborate on this process, we
project every 3D point onto the 2D image plane to determine
its corresponding RGB value. Note that multiple 3D points
could project to the same 2D pixel; therefore, this projection
and color sampling procedure is applied to all 3D points,
regardless of their visibility status in the 2D view. This ensures
that the entire geometric representation is comprehensively
enriched with appearance information derived from the image,
not just the portions directly visible.
Then, we employ a pillar-based backbone network Θ3D
such as PointPillar [46] to extract pillar-based features from
L
3D point clouds sequences VP = {Pl }l=1
for feature alignment

with image features. The process can be expressed as:
F P = Θ3D (VP ).

(2)

Here, F P = {F P1 . . . F PL } is the pillar-based feature, where
F Pl ∈ RC×H×W . It is then fused with the 2D image feature F I in
our proposed Cross-modal Fusion (CMF) module, producing
the discriminative 3D-enhanced feature F̂ I . To preserve both
geometric and shape information, the fusion feature F is
derived by concatenating the pillar-based feature F P with the
result of a shortcut connection between the initial feature F I
and the 3D-enhanced feature F̂ I . Finally, the fusion feature is
fed into the Spatial-Temporal Aggregation (STA) module to
aggregate frame-level features temporally for generating the
video representation. The task flow is shown as follows:
F̂ I = CMF(F I , F P ),
F = Concat((F I + F̂ I ), F P ),
Fout = Ψ(F),

(3)

where Concat is the concatenation operation and Ψ denotes
the STA module. Fout is the output feature which contributes
to the final video representation. The STA module is explained
in Section IV-B.
For the shape branch, we design the Shape-aware SpatialTemporal Interaction (SSTI) module, which learns the spatial
and temporal 3D shape dependencies of pedestrian and generates distinctive video-level 3D shape representation. In detail,
L
given the SMPL shape parameter β = {βl }l=1
, which is a parameter of the pedestrian mesh generated from the 3D human
reconstruction network, we first embed it into the feature space
using a shape embedding layer. Spatial and temporal shape

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID

12835

TABLE I
L OOKUP TABLEFOR N OTATIONS IN THE 3D-A ID P EDESTRIAN R EPRESENTATION L EARNING (3DAPRL) N ETWORK

features and its corresponding pillar-based features are fused
through deformable cross attention [47].
Formally, given the image feature F I and pillar-based
feature F P , we take the fI = F Il and fP = F Pl at l-th
frame as an example. We first initialize the reference point
r through the mapping relationship φ between 2D and 3D
representations obtained from the SMPL parametrization in
3D human reconstruction networks [64]:
r = φ(p),

Fig. 3. The illustration of the fusion process in the CMF module.

features are extracted and subsequently fused into spatialtemporal 3D shape features. We can express it as:
F̂βS T = SSTI(β),

(4)

where F̂βS T is the spatial-temporal 3D shape feature, and it
contributes to the final video representation. The lookup table
of parameters and corresponding notations is shown in Table I.
The detail of our proposed method is described below.
C. Cross-Modal Fusion Module
The 3D pedestrian data provide geometric information,
shape cues and accurate foreground position correspondence.
Previous Re-ID methods [40], [41], [42] utilizing 3D representation mainly focus on generating 3D features while
paying insufficient attention on effective fusion strategies for
fusing 2D and 3D features. Some methods [40], [42] map
RGB information to 3D representations or concatenate 2D and
3D features directly, neglecting the complementarity between
these modalities, resulting in coarse feature aggregation and
limited interaction. Furthermore, the inconsistency between
2D and 3D pedestrian features can lead the network to
focus on noisy or irrelevant features from both modalities,
ultimately degrading performance. To make full use of 3D data
for obtaining discriminative frame-level pedestrian representations, we propose the Cross-Modal Fusion (CMF) module. The
fusion process is shown in Figure 3, in which the 2D image

(5)

where p is the coordinate set of the points on 3D pedestrian
in l-th frame. In this manner, reference points are obtained
to indicate the pedestrian locations on fI . Then, the pillarbased feature fP is regarded as Query, and i indexes the Query
feature and its corresponding reference points r. The Query
feature fPi is fed into two linear layers to create attention
weights Ami j and sampling offsets ∆rmi j (i-th Query feature
to j-th sampling point at m-th head). We generate the Value
feature fV from the image feature fI through a linear layer,
followed by sampling a set of feature fˆVj around the reference
points on fV by employing sampling offsets ∆rmi j . Eventually,
the attention weights Ami j and sampled Value feature fˆVj are
aggregated to acquire the 3D-enhanced feature fˆIi . The fusion
process of module can be formulated as:
Ami j = linear( fPi ),
∆rmi j = linear( fPi ),

(6)

Fusion( fPi , ri , fV )
2
3
M
J
X
X
Wm 4
Ami j · Wm0 fV (ri + ∆rmi j )5 ,
=

(7)

m=1

j=1

where M and J are the number of attention heads and sampling
points, respectively. Wm and Wm0 are learnable weights. To
process all Query features, we obtain the whole 3D-enhanced
feature map fˆI . In this way, our CMF module enables the
model to adaptively select where and what features should
be extracted from the image by the inquire of pillar-based

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

12836

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025

Fig. 4. The illustration of the interaction in the SSTI module.

features. By focusing on the most relevant regions of the
image features via reference points, the model ensures the
effective utilization of crucial information. Through learnable
offsets, our fusion strategy dynamically adjusts the positions
and ranges of sampling points, allowing the model to capture
the local context of the pedestrian accurately while avoiding
irrelevant areas. Thus, 3D-enhanced image features focusing
on the most crucial areas of pedestrians are generated. Finally,
we gain the generated feature f (i.e., F l ) through a shortcut
connection and concatenation of the pillar-based feature fP
(i.e., F Pl ) with the 3D-enhanced feature fˆI (i.e., F̂ Il ) for
retaining geometric and shape information, as demonstrate in
Figure 2. We can express as:
f = Concat(( fI + fˆI ), fP ).

(8)

To apply the whole CMF module to the image feature F I and
the pillar-based feature F P , we can acquire the 3D-enhanced
feature F̂ I and final feature F. Therefore, Our CMF module
reaches cross modality alignment with adaptive fusion. The
frame-level features are strengthened with the help of pillarbased features, and the multi-frame misalignment and partial
occlusion are mitigated via the CMF module.
D. Shape-Aware Spatial-Temporal Interaction Module
The 3D shape parameters control and describe the pedestrian shape in the 3D mesh [45]. Thus, generating distinctive
3D shape representations is critical for distinguishing different
pedestrians with similar appearance. Despite existing methods
[41], [42] utilize 3D shape parameters to create features in
video scenarios, they struggle to effectively obtain high-level
spatial-temporal dependencies. Inspired by spatial-temporal
learning strategies [73], we propose the Shape-aware SpatialTemporal Interaction (SSTI) module to extract distinctive
spatial-temporal 3D shape representations, thereby enhancing
the model’s robustness in capturing pedestrian shapes. As
expressed in Figure 2, we first project 3D shape parameters
into feature space to generate 3D shape features. In specific,
given video-level 3D shape parameters β ∈ RT ×10 of a
pedestrian, we embed them into a high-dimensional space via
the shape embedding layer, generating 3D shape features Fβ
with a shape of T × S × C, where T represents the temporal
dimension, S denotes the spatial dimension, and C denotes the
channel dimension. The shape embedding layer employs linear
layers followed by a activation function and a normalization
layer.
Then, the spatial-temporal interaction process is performed,
as illustrated in Figure 4. Specifically, we derive the Query

Q, Key K and Value V from the 3D shape feature Fβ
through linear layers. Afterwards, they are mapped into two
streams (a spatial stream and a temporal stream) pairwisely to
apply feature contextualization along distinct axes. The spatial
stream captures intra-frame 3D shape, while the temporal
stream models inter-frame 3D shape variations. We process
the time axis and the space axis on two streams in parallel by
employing Multi-Head Self-Attention, enabling the extraction
of both spatial and temporal contexts. In detail, we define QS ,
KS and VS for the spatial stream and QT , KT and VT for the
temporal stream. The attention map AS for the spatial stream
and AT for the temporal stream are generated as follows:


QS · KST
,
AS = softmax
√
C


QT · KTT
.
(9)
AT = softmax
√
C
Then, VS and VT are aggregated to feature maps AS and AT ,
respectively, to obtain contextual feature FS and FT on each
stream:
FS = AS · VS ,
FT = AT · VT .

(10)

Two streams process the shape feature in parallel to obtain
contextual feature within their respective channels. The
acquired spatial feature FS and temporal feature and FT are
subsequently concatenated along the channel dimension to
form the spatial-temporal 3D shape feature Fβ . It can be
expressed as:
FβS T = Concat(FS , FT ).
(11)
The interaction process extracts the spatial and temporal
context of video-level shape parameter β. To further refine the
extracted features, motivated by [74], we normalize them and
pass them through a MLP layer consisting of linear layers and
a activation function. A shortcut connection is then applied,
producing the final spatial-temporal 3D shape feature F̂βS T ,
which can be expressed as:
F̂βS T = MLP(Norm(FβS T )) + FβS T ,

(12)

where Norm denotes the normalization layer. In this manner,
our SSTI module effectively extracts and fuses spatial and
temporal dependencies in parallel. The receptive field captures
the interactions between spatial and temporal dimensions.
As a new pedestrian characteristic, the produced spatialtemporal 3D shape representation is both complementary and
distinguishable, aiding in target pedestrian identification in
video scenarios. Therefore, it helps alleviate the impacts of
appearance confusion and further mitigates issues including
the multi-frame misalignment and partial occlusion.
E. Loss Function
We apply the cross-entropy loss with label smoothing [75]
and the triplet loss with batch hard mining [76] for training.
The Re-ID loss LID for our network can be formulated as:
LID = Lcls + Ltri ,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

(13)

JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID

where Lcls is the classification loss and Ltri is the triplet loss.
We determine the weight of the Re-ID loss for the SSTI
module with a hyper-parameter λ. The overall loss Loverall for
our network can be expressed as:
CMF
SSTI
Loverall = LID
+ λLID
,

(14)

CMF
SSTI
where LID
and LID
indicate the Re-ID loss for the CMF

and the SSTI module, respectively.
IV. E XPERIMENTS

12837

During the training, the Adam [89] optimizer with weight
decay 5 × 10−4 is employed as optimizer. The initial learning
rate is set to 3.5 × 10−4 , and the learning rate decay strategy
[90] is employed. The random horizontal flipping and random
erasing are utilized for image data augmentation. Our network
is trained with maximum 500 epochs. During evaluation, the
final feature representation is obtained by concatenating the
video-level feature representation from the STA module Ψ
with the spatial-temporal 3D shape feature F̂βS T . The cosine
similarity is used as distance metric.


codex
我已经开始读方法正文。为了后面的“证据闭环”不凭印象，我还会额外查每篇的 ablation、visualization、小节标题，但方法主线仍以你指定的前 8 页正文为准。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 \"4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf\" - | sed -n '35,570p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Identity-shared patterns

ABSTRACT
Visible-Infrared Person Re-identification (VI-ReID) is a cross-modality retrieval task that involves searching
for a specific pedestrian across different modalities (visible or infrared). Existing methods have explored crossmodality alignment from different perspectives, including spatial-domain cues and frequency-domain modeling.
Nevertheless, how to adaptively capture modality-specific frequency characteristics and effectively integrate
them into cross-modality feature alignment remains an open challenge. To address this challenge, this work
presents a novel and efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet) for the VI-ReID
task, which incorporates an Adaptive Frequency Filter (AFF) module and a Spectrum Synergy Module (SSM)
to extract robust cross-modality-invariant features. Specifically, in contrast to traditional fixed convolutionkernel operations, the AFF module reconstructs a set of learnable frequency bases by adopting a dual spatialand channel attention-guided strategy, which dynamically synthesizes a customized Spectrum filter for each
input. Complementarily, the SSM decomposes features into multiple frequency bands, which explicitly decouple
identity-shared patterns from modality-specific traits and synergistically enhance them, thereby achieving
consistent feature representation across cross-modal pedestrian images at the cross-modality feature alignment
level. Extensive experiments have been conducted on the public VI-ReID datasets (SYSU-MM01, LLCM, and
RegDB), demonstrating that our proposed method achieves competitive performance when compared to stateof-the-art approaches. Source code implementing 4SNet is publicly available at: https://github.com/dekusaklasdroid/4SNet.

1. Introduction
Visible-Infrared Person Re-identification (VI-ReID) is a crossmodality retrieval task that aims to search for images of the same pedestrian across a non-overlapping, distributed camera network, which
captures visible images during the daytime and infrared images during
nighttime or low-light conditions [1,2]. Due to its cross-perspective
search trait, VI-ReID has found widespread applications in criminal
investigations, target tracking, and robotic systems [3,4], among other
application domains [5,6]. It has also emerged as a significant academic
concern in fields like pattern recognition and artificial intelligence [7,
8]. However, the large differences between daytime and low-light

nighttime conditions, coupled with the constant variations in the
appearance and clothing of the same individual across modalities,
have exacerbated the difficulty and challenges of this modeling task,
remaining as an unresolved task yet [9,10].
Existing VI-ReID methods can be roughly classified into two categories: (1) image-level based methods, and (2) feature-level based
methods. Specifically, image-based methods resort to generative models, such as Generative Adversarial Networks (GANs) [11], and Diffusion models [12], to transform images from one modality to another.
Alternatively, some of them generate an intermediate auxiliary embedding modality that lies between the distributions corresponding to

∗ Corresponding authors.

E-mail addresses: xmf2013@whu.edu.cn (M. Xiong), 2415283016@wtu.edu.cn (F. Luo), jjhuang@wtu.edu.cn (J. Huang), yifeiguo@hust.edu.cn (Y. Guo),
azotaibi@tu.edu.sa (A. Alotaibi), bakshisambit@ieee.org (S. Bakshi), javier.delser@tecnalia.com (J. Del Ser), khan.muhammad@ieee.org (K. Muhammad).
https://doi.org/10.1016/j.patcog.2026.113966
Received 6 January 2026; Received in revised form 26 April 2026; Accepted 12 May 2026
Available online 19 May 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 180 (2026) 113966

M. Xiong et al.

the visible and infrared modalities [13,14], aimed at shrinking the
gap between them. Despite the practical and intuitive nature of these
methods, their heavy reliance on additional models renders the converted (generative) modalities vulnerable to noise [15,16]. Moreover,
the complexity of their underlying learning algorithms adds to the
overall computational burden of the VI-ReID approach, leaving ample
room for further performance enhancements. In contrast, feature-based
methods hinge on projecting data from diverse modalities into a unified
feature embedding space [17,18]. Subsequently, feature enhancement
techniques (e.g., attention-based mechanisms [19,20]) are employed
for recognizing patterns corresponding to the same pedestrian. Although these techniques diminish the reliance on additional models,
direct projection also struggles to entirely bridge the domain gaps
caused by substantial differences in the data distribution characteristics
across various modalities. Several recent methods have embraced this
strategy [21,22]. One notable example is DEEN [23], which employs a
center-guided pair mining strategy to create diverse auxiliary embedding spaces between visible and infrared modalities, thereby reducing
their differences. Another example is FDNM [24], which employs a
salience-guided and amplitude nuance mining strategy to extract crossmodality frequency domain information, thereby decreasing modality
discrepancies and enhancing VI-ReID performance.
Although these methods have achieved promising performance,
they still face two main issues:
1. When it comes to mining intrinsic attributes within a single modality,
existing methods primarily rely on a fixed, pre-learned set of convolution kernels that are uniformly applied to samples from all modalities. As a result, these static operators fail to capture modalityspecific frequency-domain characteristics (such as structural textures and body contours) present within individual modalities. For
instance, a filter that is optimal for extracting fine textures in visible
images may inadvertently amplify interference noise when applied
to infrared images (e.g., strong background–foreground contrast, as
illustrated in Fig. 1(a)). This limitation ultimately constrains the
network’s ability to discover highly discriminative, modality-specific
features.
2. Regarding feature alignment across different modalities, although several recent studies have introduced frequency-domain modeling,
many existing approaches still rely primarily on spatial-domain
cues, such as appearance attributes and relative positional relationships, for cross-modality alignment. However, due to the substantial distributional differences between visible and infrared images,
spatial representations are often sensitive to modality-specific variations, which limits their robustness for alignment. In this context, frequency-domain information (encompassing global structure,
coarse appearance distribution, and complementary edge-like responses) provides features that are relatively more stable across
modalities. Therefore, effectively exploiting these spectral characteristics remains crucial for improving cross-modality feature alignment.

Fig. 1. Motivation for the overall architecture of the 4SNet framework. (a)
Grayscale Fast Fourier Transform (FFT) magnitude spectra illustrate that
infrared images concentrate most of their energy in low-frequency bands,
whereas visible images contain richer medium- and high-frequency details.
This clear spectral divergence suggests that different modalities exhibit distinct
frequency responses, making fixed shared filters inadequate for capturing
modality-specific discriminative features. (b) Motivated by this observation,
4SNet introduces two complementary frequency-domain modules, namely the
Adaptive Frequency Filter (AFF) and the Spectrum Synergy Module (SSM). A
comparison between a baseline fixed stem (denoted as Conv1) and AFF under
visible and infrared inputs shows that fixed shared filters are suboptimal for
early-stage feature extraction, which justifies the use of AFF for input-adaptive
and modality-specific frequency filtering. Meanwhile, the visualization of
low-, middle-, and high-frequency decompositions reveals that different spectral bands preserve complementary structural and fine-grained details. The
SSM explicitly models this band-wise decomposition and their synergistic
relationships to achieve more robust cross-modality feature alignment.. (For
interpretation of the references to colour in this figure legend, the reader is
referred to the web version of this article.)

2. For the second problem, the SSM decomposes the characteristic
spectrum into low-, medium-, and high-frequency components.
These components are then processed in parallel using band-specific
operators (e.g., lightweight convolutional blocks). This strategy retains stable low-frequency body structures and coarse appearances,
while exploiting complementary mid-to-high frequency variations
and fine edge details.

To overcome these limitations, this study introduces a novel and
efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet),
which integrates an Adaptive Frequency Filter (AFF) module and a
Spectrum Synergy Module (SSM) to respectively extract singlemodality-robust features and accomplish cross-modality feature alignment for the VI-ReID task, as shown in Fig. 1(b). Specifically:

Finally, a fusion module is utilized to integrate these features,
with a particular focus on the enhanced low-frequency components to
guide the alignment of cross-domain features. Extensive experiments
on three public VI-ReID datasets (SYSU-MM01 [25], RegDB [26], and
LLCM [23]) demonstrate that 4SNet achieves competitive performance.
Ablation studies and feature visualizations further validate its superior
retrieval capability.

1. For the first issue, the AFF module dynamically generates inputspecific filters to address the limitations of static convolutional
kernels. The module first learns a set of frequency bases from images
using spatial attention. These bases are then adaptively combined
under channel attention guidance, synthesizing customized filters
that extract modality-inherent frequency-domain features.
2

Pattern Recognition 180 (2026) 113966

M. Xiong et al.

The rest of the manuscript is organized as follows: Section 2 comprehensively reviews related work in the VI-ReID field. Section 3
elaborates on the proposed 4SNet framework. Section 4 validates the
effectiveness of the proposed method through a series of extensive experiments described previously. Finally, Section 5 concludes the paper
and outlines potential future research directions rooted on our findings.

2.3. Frequency domain analysis in vision tasks
The frequency domain offers a complementary perspective for image representation and understanding, providing intrinsic robustness
against certain spatial transformations and local perturbations [40].
By decomposing visual signals into components that capture global
structure and fine-grained details at different spectral scales, frequencydomain analysis enables models to disentangle information that is often
entangled in the spatial domain. As a result, frequency-aware modeling
has been successfully applied across a wide range of computer vision
tasks. In particular, studies in domain adaptation and generalization
have demonstrated that low-frequency components, which predominantly encode global shape and coarse structural information, tend to
be more stable and domain-invariant than high-frequency components
that capture texture and local appearance variations [1,41]. These
properties make frequency representations especially suitable for scenarios involving significant domain shifts. Beyond domain adaptation,
frequency-domain insights have also been leveraged to improve adversarial robustness, where sensitivity to high-frequency perturbations
plays a critical role [42].
Motivated by this observation, several works have incorporated
frequency-domain operations directly into deep neural networks. Some
approaches have highlighted the effectiveness of frequency-aware channel attention in selectively emphasizing informative spectral responses
[43,44], while others have gone a step further by partially or fully replacing standard spatial convolutions with frequency-domain counterparts [45,46]. More recently, wavelet-guided designs such as PST [47]
have demonstrated the benefits of frequency-aware decomposition in
surface-defect detection, where carefully separating frequency components helps enhance meaningful details while suppressing irrelevant
noise. Along this line, several recent VI-ReID methods, including FDMNet [48] and FDNM [24], have explored frequency-domain modeling
through amplitude–phase decomposition, motivating further investigation into frequency-aware designs for cross-modality matching
problems.
Contribution. While prior frequency-based methods such as FDMNet [48] and FDNM [24] have achieved promising performance
through amplitude–phase decomposition, they typically treat frequency
modeling as a single, unified mechanism for reducing modality discrepancy. In these approaches, amplitude and phase components are
jointly manipulated to encourage modality-invariant representations,
implicitly assuming that a common frequency processing strategy is
sufficient for both intra-modality feature extraction and inter-modality
alignment. As a consequence, the intrinsic differences in frequency
distribution characteristics between visible and infrared modalities are
not explicitly addressed at different stages of the network. In contrast,
4SNet introduces a role-decoupled frequency modeling strategy, in
which frequency information is explicitly assigned to two complementary and well-defined roles within the framework. Specifically, the
AFF operates at the input stage in a dual-stem, non-shared manner
to perform modality-specific and input-adaptive frequency filtering,
enabling visible and infrared images to be characterized by customized
frequency responses before entering the shared backbone. The Spectrum Synergy Module (SSM) addresses cross-modality alignment as
a separate modeling problem by adopting a coarse low-, medium, and high-frequency band decomposition and performing band-wise
processing and fusion. This design emphasizes robust shared structural
cues while preserving complementary discriminative details, thereby
better meeting the practical requirements of day–night visible–infrared
matching.

2. Related work
2.1. Image-based VI-ReID methods
As previously mentioned, image-based methods depend on generative models (such as GANs [11] or Diffusion models [12]) to synthesize
an auxiliary data modality that bridges the gap between visible and
infrared modalities, minimizing the differences between the two modalities and effectively addressing the VI-ReID task. DiVE [12] is a recent
proposal that adopts this strategy: it generates visible-IR paired images
from samples with the same identity ID using a fine-tuned diffusion
model, thereby extending the text-driven approach to achieve identity
consistency for the VI-ReID task. Pan et al. [27] employ a conditional diffusion model, which is grounded on binary modal indicators
and modal independence, to respectively generate intermediate modal
images of the target modality and pedestrian identity. The bidirectional modal interaction network proposed in [2,28] also follows an
image-based strategy by separately generating samples for each other’s
modality to lower the difficulty of modal recognition. CSDN [29] and
DMA [30] have proposed, respectively, a CLIP-driven semantic discovery network and a dual modality-aware alignment method, designed to
generate intermediate modalities. Other similar approaches, like [31,
32], also adopt the strategy of generating intermediate modalities at
the image level to address the VI-ReID problem.

2.2. Feature-based VI-ReID methods
Unlike their image-based counterparts, feature-based methods [33]
mainly map the features of the two modalities to a shared embedding
space, and subsequently unify the modeling and mapping processes to
achieve cross-modality feature alignment. Some approaches [25,31]
employ a two-stream architecture, which focuses on aligning feature distributions through metric learning and adversarial training.
CPN [34] tackles VIS-NIR person ReID through angular metric learning, projecting features into a modality-invariant hypersphere space.
IDKL [17] explored feature disentanglement to separate modalityinvariant from modality-specific information, enhancing the shared
representation. TSKD [1] proposes a two-stage distillation for progressive cross-modality alignment and intra-modality reduction, robust
under weak supervision. MSCMNet [32] extracts multi-scale shared
semantic features to accomplish cross-modality feature alignment tasks.
DMDL [35] further introduces a dual-level modality debiasing learning
framework for unsupervised VI-ReID, where modality bias is reduced
at both the model and optimization levels. In addition, related works
such as [36,37], aim to achieve cross-modality feature alignment
by extracting intrinsic features and cross-domain invariant features
separately, and using other auxiliary information (such as semantic
text-guided) to bridge the modality gap. Recent endeavors [38,39]
have also pivoted towards part-level alignment, introducing dynamic
and semantically-aware strategies to surpass the limitations of simple
horizontal striping (namely, dividing an image into a fixed number of
equal-sized horizontal segments, which fails to account for changes
in pose or body part location). Different from these methods, 4SNet
addresses supervised VI-ReID from a frequency-modeling perspective.

3. Proposed 4SNet framework
In this section we introduce the proposed 4SNet network, whose
structure is summarized in Fig. 2. Firstly, the AFF module dynamically detailed in Section 3.2 generates a unique filter to capture the
3

Pattern Recognition 180 (2026) 113966

M. Xiong et al.

Fig. 2. Overall architecture of the proposed 4SNet framework. The architecture adopts a dual-stem, shared-backbone design. First, a batch of visible (𝐼𝑉 ) and
infrared (𝐼𝐼 ) images are fed into two independent, non-shared AFF modules (𝐴𝐹 𝐹𝑉 and 𝐴𝐹 𝐹𝐼 ) to dynamically extract initial modality-specific features (𝐹𝑉 and
𝐹𝐼 ). These features are then concatenated along the batch dimension to form 𝐹𝑚𝑖𝑥𝑒𝑑 . Subsequently, 𝐹𝑚𝑖𝑥𝑒𝑑 is processed by a fully shared-weight backbone, where
the SSM is embedded to perform coarse band-wise spectrum synergy for robust cross-modality alignment. Finally, the network is optimized jointly by triplet (𝐿𝑡𝑟𝑖 )
and identity (𝐿𝑖𝑑 ) losses to ensure robust modality-invariant feature alignment.

frequency characteristics of a series of images, thereby extracting the
intrinsic features of a single-modality. Then, the SSM in Section 3.3 implements cross-modality feature alignment. Section 3.4 presents the loss
function and the learning procedure used to train the 4SNet framework.

a fixed convolution kernel size. This rigid and uniform filtering mechanism is often suboptimal, as it fails to capture modality-specific and
personalized features commonly encountered in real-world scenarios.
To address this limitation, we propose the AFF module. As has been
previously mentioned, AFF advances frequency modeling over existing
frequency-based methods by performing modality-specific and inputadaptive filtering at the early feature extraction stage. In this way, the
network can generate sample-specific frequency responses for visible
and infrared inputs before shared-backbone learning. The key idea is
to move away from fixed convolution kernels and instead dynamically
generate customized filters for each input during training. As illustrated
in Fig. 2, the AFF module architecture consists of three main components: attention-guided feature modulation, dynamic kernel synthesis,
and frequency-domain filtering. Each of these components is described
in detail below:

3.1. Overview
Our 4SNet framework employs a unique dual-stem, sharedbackbone architecture designed for cross-modality matching between
visible (𝐼𝑉 ) and infrared (𝐼𝐼 ) data. At the input stage, a batch of 𝐵
visible images 𝐼𝑉 and 𝐵 infrared images 𝐼𝐼 are first fed separately into
two independent and non-shared-weight AFF modules, one for each
modality (AFF𝑉 and AFF𝐼 ). This design ensures the capture of modalityspecific initial features at the earliest stage, yielding outputs 𝐹𝑉 and
𝐹𝐼 therefrom. Distinct from traditional explicit feature or decision
fusion, our core strategy achieves alignment in a shared feature space.
To this end, we concatenate 𝐹𝑉 and 𝐹𝐼 along the batch dimension
(Concat𝐵 (⋅)) to form a mixed batch feature 𝐹mixed = Concat𝐵 (𝐹𝑉 , 𝐹𝐼 ).
This 𝐹mixed is subsequently fed into the downstream fully sharedweight network backbone for unified feature extraction. This backbone
comprises standard convolutional blocks (BN Relu MaxP, Block 1–3),
the SSM module, and Block 4. Since backbone weights are shared
among visible and infrared images, the training losses (e.g., 𝐿𝑡𝑟𝑖 and
𝐿𝑖𝑑 ) are applied over the network, which is forced to learn a robust
modality-invariant feature.

Attention-Guided Feature Modulation. To emphasize prominent features according to their spatial and channel attention mechanisms,
we draw inspiration from CBAM [49] and adopt spatial and channel
attention mechanisms to locate key regions in the original image,
respectively. These mechanisms allow obtaining more accurate highlevel semantic information and preserving spatial domain features.
Firstly, a spatial attention operation is used to calculate a weight map
as:
(
)
𝐴𝑠 = 𝜎 𝑓 7×7 (Concat (AvgPool(𝑋), MaxPool(𝑋))) ,
(1)
where 𝐴𝑠 ∈ R𝐻×𝑊 denotes the weight map after executing the spatial
attention operation, and 𝐻 and 𝑊 are the height and width of the feature map, respectively. Here, 𝑓 7×7 represents a convolutional layer with
a 7 × 7 kernel size, and 𝜎(⋅) denotes the sigmoid function. Subsequently,
this produced weight map is employed to modulate the input feature
𝑋, resulting in a spatially-reweighted feature map 𝑋 ′ given by:

3.2. Adaptive Frequency Filter (AFF)
As mentioned in the introduction, a central challenge in the VIReID task lies in effectively managing the two distinct data distributions
(visible and infrared) in the feature extraction stage of the model. The
standard convolutional neural layer (central to most neural network
architectures) becomes a static operator after training, characterized by

𝑋 ′ = 𝑋 ⊗ 𝐴𝑠 ,
4

(2)

Pattern Recognition 180 (2026) 113966

M. Xiong et al.

where ⊗ denotes element-wise multiplication. In parallel, after Global
Average Pooling (GAP), the original input feature 𝑋 is processed
through a similar channel attention operation to get 𝑋 𝐶 . Subsequently,
to acquire frequency domain information more effectively from singledomain images, a Discrete Fourier Transform (DFT) operation is introduced, which transforms the modulated feature map 𝑋 ′ ∈ R𝐶×𝐻×𝑊
into the frequency domain 𝑋 𝐷 ∈ C𝐶×𝐻×𝑊 .

3.3. Spectrum Synergy Module (SSM)
Although the AFF module addresses the issue of intrinsic feature
mining within a single modality, there remains an inconsistency between modality-invariant identity cues (e.g., global body structure
and coarse appearance patterns) across different modalities. This inconsistency leads to a decline in cross-modality feature alignment
performance. To tackle this problem, the proposed SSM, illustrated in
Fig. 2, first decomposes the feature spectrum into different frequency
bands. Unlike existing frequency-based methods that mainly model
amplitude-phase interactions, SSM treats cross-modality alignment as a
dedicated coarse band-wise synergy problem. By explicitly decomposing the spectrum into low-, mid-, and high-frequency components, SSM
emphasizes robust shared structures while retaining complementary
discriminative cues for inter-modality alignment. Then, it employs
specialized operators to conduct parallel processing on these bands.
Finally, it synergistically fuses them to generate an enhanced feature, thereby improving cross-modality feature alignment, which are
described as follows:

Dynamic Kernel Synthesis. The AFF module synthesizes an adaptive
convolutional filter through the dynamic combination of learnable
frequency bases, which are represented as 𝑁 learnable base kernels denoted as {𝐾1 , 𝐾2 , … , 𝐾𝑁 }. However, differently from recent work using
learnable frequency modules [50], our approach begins by initializing
the filter based on binary values (0 and 1), directly parameterizing it
in the frequency domain, and constructing orthogonal constraints using
non-overlapping Fourier indices. Specifically, the orthogonality of the
base kernel is ensured by employing a set of non-overlapping binary
masks {𝑀1 , … , 𝑀𝑖 , … , 𝑀𝑁 } in the frequency domain. Each mask 𝑀𝑖
is assigned a value of 1 exclusively within a specific, non-overlapping
region of the Fourier spectrum, and 0 elsewhere. The 𝑖th learnable
base kernel 𝐾𝑖 is computed as the element-wise product of a learnable
complex tensor 𝑇𝑖 and its corresponding fixed binary mask 𝑀𝑖 in the
frequency domain. To ensure that each 𝐾𝑖 only captures patterns within
its preassigned frequency subspace, we enforce the condition 𝑀𝑖 ∩𝑀𝑗 =
∅ (𝑖 ≠ 𝑗), i.e., the binary masks are non-overlapping in the Fourier
domain. Inspired by SE-Net [51], we introduce an attention mechanism
to adaptively generate both the channel reweighting vector 𝐰𝑐 and the
kernel aggregation coefficients 𝜶. Specifically, the input feature 𝐗 is
first processed by Global Average Pooling (GAP), followed by a shared
fully connected layer 𝐖1 and a ReLU activation, to obtain a compact
global representation. Based on this shared representation, two separate
branches are introduced. As illustrated in Fig. 2, the shared FC layer
corresponds to 𝐖1 , while FC1 and FC2 denote the two branch-specific
linear layers 𝐖2 and 𝐖3 , respectively. The first branch generates the
channel importance vector 𝐰𝑐 ∈ R𝐶 for channel-wise modulation:
𝐰𝑐 = 𝜎(𝐖2 (𝛿(𝐖1 (GAP(𝐗))))),

Frequency Band Decomposition. Given an input feature map 𝑋mid ∈
R𝐶×𝐻×𝑊 (where 𝐶, 𝐻 and 𝑊 denote the channel, height and width of
the feature map, respectively) from the intermediate block layer, we
first convert it into its frequency-domain representation  (𝑋mid ) via
a two-dimensional discrete Fourier transform (2D DFT). To partition
the spectrum according to frequency content, we employ a set of
predefined, non-overlapping frequency masks: a low-pass mask 𝑀𝐿 ,
a middle-pass mask (𝑀𝑀 ), and a high-pass mask (𝑀𝐻 ). To ensure
smooth, non-binary transitions between frequency bands and prevent
the introduction of ringing artifacts, which are commonly observed
in ideal filters, we use Butterworth filtering [52]. As such, filtering
masks are constructed using 𝑛𝐵 -order Butterworth filters and rely on
the Euclidean distance 𝐷(𝑢, 𝑣) (with 𝑢 and 𝑣 representing the frequencydomain coordinates for the horizontal and vertical spatial frequencies
within the feature map) to the center of the spectrum. We define two
fixed cutoff thresholds in the normalized frequency domain, 𝜓1 and 𝜓2
(𝜓1 < 𝜓2 ). The mask is computed as:

(3)

where 𝐖2 denotes the linear layer of the channel branch, 𝛿(⋅) is the
ReLU activation function, and 𝜎(⋅) is the Sigmoid function. The second
branch generates the aggregation coefficients 𝜶 ∈ R𝑁 for the 𝑁
learnable frequency bases:
𝜶 = Softmax(𝐖3 (𝛿(𝐖1 (GAP(𝐗))))),

1
,
1 + (𝐷norm (𝑢, 𝑣)∕𝜓1 )2𝑛𝐵
1
𝑀𝐻 (𝑢, 𝑣) = 1 −
,
1 + (𝐷norm (𝑢, 𝑣)∕𝜓2 )2𝑛𝐵

(4)

𝑁
∑

𝛼𝑖 ⋅ (IDFT(𝐊𝑖 ) ⊗ 𝐰𝑐 ),

(9)

where 𝐷norm (𝑢, 𝑣) is the normalized Euclidean distance, defined as:
𝐷(𝑢, 𝑣)
,
(10)
𝐷max
√
√
where 𝐷(𝑢, 𝑣) = (𝑢 − 𝐻∕2)2 + (𝑣 − 𝑊 ∕2)2 , 𝐷max = (𝐻∕2)2 + (𝑊 ∕2)2 ;
𝜓1 and 𝜓2 are hyper-parameters; and 𝑛𝐵 denotes the filter order,
which dictates the steepness of its frequency-response curve. A higherorder filter results in a more abrupt transition between the pass-band
and the stop-band, bringing the filter closer to an ideal one. This
construction method ensures smooth, non-binary transitions between
frequency bands, thereby preventing the generation of artifacts. The
computed masks are then applied to the frequency-domain feature
map through element-wise multiplication to isolate the corresponding
spectrum components:
𝐷norm (𝑢, 𝑣) =

(5)

𝑖=1

where ⊗ denotes channel-wise multiplication with broadcasting over
the spatial dimensions. In this way, 𝐰𝑐 controls the channel response
of each base kernel, while 𝜶 determines the contribution of each base
kernel in the final dynamic filter construction.
Frequency-Domain Filtering. The synthesized spatial-domain kernel
𝐾𝑑𝑦𝑛 is transformed into its frequency-domain equivalent,  (𝐾𝑑𝑦𝑛 ), via
a DFT. This dynamic filter is then applied to the modulated feature
representation  (𝑋 ′ ):
 (𝑌 ) =  (𝑋 ′ ) ⊗  (𝐾𝑑𝑦𝑛 ).

(8)

𝑀𝑀 (𝑢, 𝑣) = 1 − 𝑀𝐿 (𝑢, 𝑣) − 𝑀𝐻 (𝑢, 𝑣),

where 𝐖3 denotes a separate linear layer that maps the shared representation to 𝑁 coefficients. The Softmax function ensures that 𝜶 =
∑
[𝛼1 , … , 𝛼𝑁 ] satisfies 𝑁
𝑖=1 𝛼𝑖 = 1. After that, each learnable frequency
base kernel 𝐊𝑖 is first transformed into the spatial domain by the inverse
discrete Fourier transform (IDFT), and then reweighted by 𝐰𝑐 along the
channel dimension. The reweighted kernels are finally aggregated using
the coefficients 𝜶 to form the dynamic kernel 𝐊dyn :
𝐊dyn =

(7)

𝑀𝐿 (𝑢, 𝑣) =

𝐹band =  (𝑋mid ) ⊗ 𝑀band ,

where band ∈ {𝐿, 𝑀, 𝐻},

(11)

where 𝐹band denotes feature representations comprising components
from the low (band = 𝐿), medium (band = 𝑀), and high frequency
(band = 𝐻) bands.

(6)

Parallel Band-Specific Processing. Each frequency component 𝐹𝑏𝑎𝑛𝑑
is subsequently processed by a dedicated parallel branch. This enables the network to learn transformations that are specifically tailored to the nature of the information within each band. Initially,

The final output feature map, denoted as 𝑌 , is acquired by applying
the Inverse Discrete Fourier Transform (Inverse DFT) to the obtained
frequency-domain representation, i.e., 𝑌 = IDFT( (𝑌 )).
5

Pattern Recognition 180 (2026) 113966

M. Xiong et al.

̂band ∈ R𝐵×𝐶×𝐻×𝑊 is the feature tensor for the corresponding
where 𝑋
̂ This dynamic weighting mechanism allows the
band, indexed from 𝑋.
network to adaptively emphasize the most discriminative spectrum
components. A residual connection from the original input 𝑋mid is
subsequently added to 𝑌out to facilitate training.

each 𝐹𝑏𝑎𝑛𝑑 is transformed back into the spatial domain using IDFT.
Subsequently, the resulting spatial representation is fed through a
lightweight, band-specific convolutional block as:
′
𝑋𝑏𝑎𝑛𝑑
= ConvBlockband (IDFT(𝐹band )),

(12)

where ConvBlockband is designed as an efficient sequential block composed of two convolutional layers: first, a 3 × 3 convolutional layer
is employed for spatial feature extraction, followed by Batch Normalization (BN) and Rectified Linear Unit (ReLU). Then, a 1 × 1
convolutional layer is included for channel information mixing, also followed by BN and ReLU. The parameters of ConvBlock𝐿 , ConvBlock𝑀 ,
and ConvBlock𝐻 are learned independently.
In addition, since low-frequency components encapsulate the most
critical and modality-invariant structural information, we apply an
additional channel attention mechanism specifically to them. Concretely, only the output 𝑋𝐿′ from the low-frequency branch undergoes further channel-wise modulation via a Squeeze-and-Excitation
(SE) block [51], resulting in the enhanced feature 𝑋𝐿′′ . Meanwhile,
′ and high-frequency 𝑋 ′ features are
the intermediate-frequency 𝑋𝑀
𝐻
directly passed forward for subsequent fusion without further modification. This differentiated parallel architecture enables the model
to retain robust structural information from the low-frequency branch
and enhance its discriminative capacity through the SE block, while
simultaneously preserving intermediate-frequency local variations and
high-frequency edge-like fine-detail responses as complementary cues,
thereby improving overall feature representation.

3.4. Loss function
To effectively train 4SNet, we employ a composite loss function
total = 𝑖𝑑 + 𝑡𝑟𝑖 that jointly optimizes both intra-modality identity
discrimination and cross-modality feature alignment. The identity loss
𝑖𝑑 [53] and triplet loss 𝑡𝑟𝑖 [54] are described as:
• Identity loss within single modality discrimination. The identity loss 𝑖𝑑
is implemented as the standard cross-entropy loss for person identity classification. It encourages the network to learn discriminative
features that distinguish individuals within each modality (visible or
infrared). Formally, for a batch of 𝐵 samples:

𝑡𝑟𝑖 =

• Feature refinement. The first step is to refine the shared features. To
efficiently learn robust cross-band features, we use a shared fusion
network 𝛷fusion to process the enhanced features. Then, the final
enhanced feature of the low-frequency branch is defined as 𝑋𝐿′′ =
SE-Block(𝑋𝐿′ ), while the mid-frequency and high-frequency features
′ and 𝑋 ′ , respectively. Subsequently, the feaare denoted as 𝑋𝑀
𝐻
tures from all parallel branches are concatenated along the batch
dimension:

exp(𝑤band )
.
b∈{𝐿,𝑀,𝐻} exp(𝑤b )

(18)

4. Experimental results and analysis

(13)
In this section, we present and discuss the experimental results in

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext \"4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf\" - | sed -n '1320,1435p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Rank-1

mAP

77.02
76.65
77.86

73.51
73.21
74.31

82.65
82.93
84.96

84.75
85.14
86.85

strategies on the SYSU-MM01 dataset. The results are reported in
Table 4. Specifically, we evaluate two sequential arrangements (Channel → Spatial and Spatial → Channel) against our proposed parallel
formulation (Channel ∥ Spatial). As shown in the table, the parallel
strategy consistently outperforms both sequential counterparts across
all settings. This empirical evidence supports our theoretical hypothesis: unlike standard spatial-domain features, the channel dimension
(which selects relevant frequency bases) and the spatial dimension
(which locates spatial importance) act as two orthogonal modulators
during frequency filter synthesis. The parallel formulation preserves
both contexts independently and comprehensively, yielding optimal
dynamic frequency adaptation for the AFF module.

4.3. Ablation analysis
We continue our discussion by inspecting the impact of frequency
bands, the location of the SSM, and the hyperparameter values.
Performance contribution of each component. To assess the contribution of the proposed AFF module and SSM, we conduct a comprehensive
ablation study on the SYSU-MM01 and LLCM datasets. The results,
summarized in Table 3, highlight the impact of each component. On the
SYSU-MM01 dataset under the All Search setting, the baseline model
(without AFF and SSM) achieves R1 accuracy of 72.17% and mAP
of 69.86%. Under the Indoor Search setting, these metrics increase
to 79.63% and 81.56%, respectively. Introducing the AFF module
alone leads to improvements across both settings, with scores rising
to 75.48% (R1), 72.59% (mAP), 82.15% (R1), and 83.45% (mAP),
corresponding to gains of 3.31%, 2.73%, 2.52%, and 1.89%. Similarly,
adding only the SSM module yields scores of 75.62%, 72.89%, 83.21%,
and 84.51%, reflecting improvements of 3.45%, 3.03%, 3.58%, and
2.95%. When both AFF and SSM are integrated, the model achieves its
best performance: 77.86% (R1) and 74.31% (mAP) under All Search,
and 84.96% (R1) and 86.85% (mAP) under Indoor Search, marking
overall gains of 5.69%, 4.45%, 5.33%, and 5.29%, respectively. Consistent improvements were also observed on the LLCM dataset. These
results clearly demonstrate the effectiveness of the proposed modules.

Impact of frequency bands. To gain deeper insight into the role of
the low-, mid-, and high-frequency bands employed in the SSM, we
perform a validation study on the SYSU-MM01 dataset. The corresponding results are reported in Table 6. As shown, each frequency
band contributes differently to the overall performance. Importantly,
the best results are obtained only when all three bands are jointly
utilized, underscoring the complementarity of multi-frequency bands
in improving the effectiveness of 4SNet.
Impact of frequency partitioning strategy in SSM. We further clarify
the asymmetric frequency decomposition design between AFF and SSM.
AFF adopts an 𝑁-band split to capture fine-grained, input-specific
details for intra-modality representation learning. In contrast, SSM uses
a coarse 3-band split (Low/Mid/High), since cross-modality alignment
requires more robust and abstract semantic cues. A finer partition
in SSM may overfit to modality-specific noise, while the coarse split
better preserves shared structure and suppresses irrelevant discrepancies. We also validate this design through an ablation study on SSM

Impact of attention aggregation strategies in AFF. To quantitatively
justify the architectural choice within the proposed AFF module, we
conduct a validation study comparing different attention aggregation
8

Pattern Recognition 180 (2026) 113966

M. Xiong et al.

Table 5
Ablation study on the frequency partitioning strategy of the SSM on
the SYSU-MM01 dataset.
All Search

Partitioning Strategy in SSM

Indoor Search

Rank-1

mAP

Rank-1

mAP

73.42
77.86

69.55
74.31

81.34
84.96

80.45
86.85

𝑁-band split (Same as AFF)
Coarse 3-band split (L/M/H)

Regarding the Butterworth filter order 𝑛𝐵 , increasing it from 1 to 2
significantly boosts performance, achieving peak scores at 𝑛𝐵 = 2,
while further increases to 𝑛𝐵 = 4 or 𝑛𝐵 = 8 degrade performance
due to ringing artifacts caused by overly sharp transitions. Based on
these analyses, we configure 𝜓1 = 0.20, 𝜓2 = 0.60, and 𝑛𝐵 = 2
as the optimal hyperparameters for the SSM module, striking the
best balance between feature representativeness and cross-modality

codex
4SNet 的重定义很清楚：它不是泛泛说“跨模态有 gap”，而是把 gap 细分成“单模态内的频率响应要自适应”和“跨模态对齐要按频段协同”。这种拆法会直接决定它后面两个模块的职责。
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
/bin/zsh -lc "pdftotext \"'Knowledge and experience' for visible-infrared person re-identification.pdf\" - | rg -n -C 2 \"Ablation|ablation|visual|Visualization|t-SNE|inferred|knowledge|experience|KET|memory|Fig\\. 5|impact|component\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
6-journal homepage: www.elsevier.com/locate/pr
7-
8:‘Knowledge and experience’ for visible-infrared person re-identiﬁcation
9-Nianchang Huang a,b , Jialiang Wang a,b , Qiang Zhang
10-
--
24-Keywords:
25-Cross-modality person re-identiﬁcation
26:‘Knowledge and experience’
27-Modality-invariant person information
28-
29-a b s t r a c t
30:Most existing models mainly focus on learning those modality-shared features unique to individuals for crossmodality Visible-Infrared Person Re-Identiﬁcation (VI-ReID). Diﬀerently, the human visual systems identify different persons by using not only the information within the input images but also their learned knowledge and
31:experience in history. Speciﬁcally, given a query image, our visual systems will ﬁrst extract lots of information
32-about the person from the given image and then deduce more information about this person with the aid of our
33:knowledge and experience for searching this person from a set of images. Inspired by that, we will propose a
34-novel Cross-modality Knowledge and Experience Transformer (CKT) for cross-modality VI-ReID in this paper.
35-In detail, our CKT will ﬁrst memorize the discriminative modality-shared information of each identity in the
36:training set, and then convert them into our model’s ‘knowledge and experience’. Accordingly, in addition to the
37-information within those input images, our CKT can infer more modality-invariant person information from its
38:‘knowledge and experience’ according to their corresponding modality-shared information for VI-ReID. Comprehensive experiments show that our proposed model achieves signiﬁcant performance improvements over most
39-state-of-the-art ones by combining the information within the input images and their induced information from
40:our model’s ‘knowledge and experience’.
41-
42-1. Introduction
--
58-hinder VI-ReID systems to reach the next milestone.
59-Alternatively, we will overcome the cross-modality VI-ReID issue by
60:imitating some characteristics of the human visual system. Speciﬁcally,
61-if a person wants to tell whether the identities in two images are the
62-same or not, he (she) uses not only the information obtained from the
63:two images but also his (her) learned knowledge and experience in the
64-past. For example, as shown in Fig. 1, when we judge whether person
65-A in the RGB image and person B in the IR image are the same person
--
67-from the corresponding images, e.g., person A has thin arms and a thin
68-neck, but person B has a thick neck and a large waist. Then, given these
69:attributes extracted from input images, alongside our knowledge and
70:experience, we may infer more modality-invariant attributes about the
71-two persons in other views or poses, e.g., person A may have a round
72-face and a slim ﬁgure, and person B may have a stubby neck and a plump
73:body. By fusing these extracted and inferred person attributes, we will
74-make our ﬁnal judgment. In this process, besides the person information
75:extracted from the input images, our knowledge and experience also
76-
77-∗ Corresponding author.
--
88-N. Huang et al.
89-
90:Fig. 1. General process of identifying diﬀerent identities in the human visual system.
91-
92-play important roles, since they provide more modality-invariant person
--
94-Inspired by that, a novel Cross-modality Knowledge and Experience
95-Transformer (CKT) will be designed in this paper for VI-ReID, which will
96:employ its ‘knowledge and experience’ to infer more modality-invariant
97-information for VI-ReID, in addition to some information extracted from
98:the input images. Speciﬁcally, as the human visual systems, which ﬁrst
99-capture information from the given images, our proposed CKT will also
100:ﬁrst extract those discriminative modality-shared features from the images of diﬀerent modalities by employing a cross-modality VI-ReID subnetwork. On top of that, our proposed CKT will imitate the human visual
101:system to infer more modality-invariant information for VI-ReID by using its ‘knowledge and experience’.
102-To this end, we observe that the information within all the training
103-images not only contains rich discriminative person features/attributes
--
105-that we may infer more information about this person from some existing information by virtue of those learned relations among diﬀerent
106-person features/attributes. For example, if a person looks overweight in
107:the given image, it may be inferred that the person probably carries a
108-heavy belly rather than a ﬂat stomach by exploring the relations among
109-the features from diﬀerent identities, i.e. such relations may be obtained
110-by comparing those features from a set of overweight persons and those
111:from a set of thin persons. Based on such an observation, a novel Knowledge and Experience Transformer (KET) will be further designed in our
112-proposed CKT, which will transfer the information within the training
113:set into our VI-ReID model’s ‘knowledge and experience’ and employ
114-them for inferring more modality-invariant person information based
115:on the features from the given images as that in the human visual system.
116:More speciﬁcally, the KET will ﬁrst save the modality-shared feature center of each identity in the training set as our model’s memory,
117:just like a person’s memory. Here, each modality-shared feature center can represent some unique characteristics of its corresponding identity. Accordingly, the memory features can well save the information of
118:diﬀerent identities in the training set. Then, a knowledge module will
119:be designed in the proposed KET for establishing its ‘knowledge and
120:experience’ by exploring the interactions across the feature centers of
121:diﬀerent identities. Similar to that in the human visual system, these
122:‘knowledge and experience’ will enable our model to infer more hidden modality-invariant person information, given such modality-shared
123:features extracted from the input images. Therefore, the proposed KET
124-will further design a novel reasoning module, which will utilize the information extracted from the images of one identity for inferring more
125-
126:modality-shared information about the identity from the ‘knowledge
127:and experience’ of our proposed CKT. By doing so, our proposed model
128-can combine the modality-shared features from the input images and
129:those inferred modality-shared features from its ‘knowledge and experience’ for VI-ReID in the testing stage.
130-The main contributions of our work are as follows:
131:(1) Inspired by the human visual system, we endue our proposed VIReID model with the ability to learn its ‘knowledge and experience’ by
132-exploring the relations among all the identities in the training set. By
133:virtue of the learned ‘knowledge and experience’, our proposed model
134-can infer more modality-shared information, thereby helping to reach
135-new performance milestone for VI-ReID models.
136-(2) We propose a novel Cross-modality Knowledge and Experience Transformer (CKT) to implement the above idea. Speciﬁcally, the
137:modality-shared information extracted from each individual in the training set is memorized and transferred into our model’s ‘knowledge and
138:experience’ via a novel Transformer to infer more modality-invariant
139-information for VI-ReID during the testing stage.
140-(3) Extensive experiments prove that our proposed model achieves
--
203-concise, thus obtaining more attentions [14,15]. However, existing
204-modality-shared feature learning based models only focus on exploring the features within the input images, ignoring the fact that human
205:visual systems also employ those information inferred from their ‘knowledge and experience’. Diﬀerently, as that in human visual systems, our
206-proposed model can not only extract discriminative modality-shared information from the input images but also infer more modality-shared
207:information from its ‘knowledge and experience’, thus achieving better
208-results.
209-
--
221-Here, 𝑖∗ denotes that 𝑔𝑖∗ is the matched image. 𝑆𝑖𝑚(∗) denotes a metric
222-function.
223:Diﬀerently, our proposed model will use its ‘knowledge and experience’ to infer more discriminative information for VI-ReID. Here, we
224:denote the ‘knowledge and experience’ of our proposed model by 𝐊𝐄,
225-which can be seen as a set of person features/attributes for better explanation. Speciﬁcally, after extracting the features 𝐅𝑞 or 𝐅𝑔𝑖 , our proposed
226-model will further infer more information about the identity from our
227:model’s ‘knowledge and experience’ according to 𝐅𝑞 or 𝐅𝑔𝑖 by
228-
229-𝐅𝐾𝑞 = 𝑓𝑧 (𝐅𝑞 , 𝐊𝐄, 𝛽), 𝐅𝐾𝑔𝑖 = 𝑓𝑧 (𝐅𝑔𝑖 , 𝐊𝐄, 𝛽),
--
232-
233-where 𝑓𝑧 (∗, 𝛽) denotes an inference function with its parameters 𝛽. After
234:that, our proposed model will further fuse those inferred features and
235-the features extracted from the input images, i.e.,
236-
--
266-features space to extract their modality-shared features. After that, the
267-resulting modality-shared features will be fed into the Knowledge and
268:Experience Transformer (KET) for inferring more modality-shared information about the individual. Finally, the modality-shared features
269:extracted from input images and their corresponding inferred features
270-are combined to achieve VI-ReID. We will describe their details in the
271-following contents.
--
377-ﬁnal modality-shared person features 𝐅𝑠,𝑚 ∈ 𝑅1×𝐷 for VI-ReID. This procedure can be mathematically represented as follows:
378-
379:3.3. Knowledge and experience transformer
380-On top of the extracted modality-shared features, the proposed
381:Knowledge and Experience Transformer (KET) aims to imitate the human visual systems for inferring more modality-shared information according to its ‘knowledge and experience’. As shown in Fig. 2, it contains a knowledge module for establishing our model’s ‘knowledge and
382:experience’, and a reasoning module for inferring more modality-shared
383-information according to the input features. The details about the two
384-modules are introduced as follows.
385-3.3.1. Knowledge module
386:The knowledge module establishes our model’s ‘knowledge and experience’ via the following two steps.
387-Establishing our model’s memories: Considering that the feature
388:center of each identity can well represent those dicriminative information of this identity, the knowledge module employs a feature matrix
389:𝐌 ∈ 𝑅𝑁×𝐷 (also called memory features) as its memory to store the
390-modality-shared feature center of each identity. Here, 𝑁 denotes the
391-number of person identities in the training set and 𝐷 denotes the feature channels of each center. Moreover, the features in the 𝑛-th row
392-of 𝐌 ∈ 𝑅𝑁×𝐷 , i.e., 𝐌𝑛 ∈ 𝑅1×𝐷 , represent the feature center of the 𝑛-th
393:person identity. Generally speaking, the memory features 𝐌𝑛 can be
394-computed by
395-
--
443-𝑠,𝑅
444-𝑠,𝐼
445:our proposed KET directly learns the memory features 𝐌 by virtue of
446-
447-(7)
--
458-
459-3.3.2. Reasoning module
460:After establishing our model’s ‘knowledge and experience’, our
461-model can infer more modality-shared information by using a novel reasoning module on top of the modality-shared features extracted from the
462-input images. As shown in Fig. 4, in the reasoning module, several reasoning blocks are employed to infer more discriminative information
463:from the knowledge features 𝐊𝐍 ∈ 𝑅𝑁×𝐷 for VI-ReID via the modalityshared features 𝐅𝑠,𝑚 ∈ 𝑅1×𝐷 extracted from the input images.
464-Here, we take the ﬁrst reasoning block as the example to introduce its
465-details. Speciﬁcally, in the ﬁrst reasoning module, the input features 𝐅𝑠,𝑚
466:are taken as the query features 𝐅𝑞,𝑚,1 , which, together with the knowledge features 𝐊𝐍, will be ﬁrst projected into the same feature space via
467-two linear functions, i.e.,
468-
469:Fig. 3. Illustration of the knowledge module.
470-
471-two proposed loss functions, i.e., a feature aggregation loss 𝐿𝐹 𝐴 and a
--
473-Speciﬁcally, the feature aggregation loss 𝐿𝐹 𝐴 aims to pull all features
474-of the same identity close to their corresponding feature center in the
475:memory features 𝐌, i.e.,
476-
477-𝐿𝐹 𝐴 =
--
508-̂ 1 ∈ 𝑅𝑁×𝐷 denote the query features and the
509-where 𝐐
510:key features, respectively. 𝑚 ∈ {𝑅, 𝐼} denotes the features from diﬀerent modalities. Meanwhile, the knowledge features 𝐊𝐍 will be further
511-̂1 ∈ 𝑅𝑁×𝐷 by another linear function,
512-transferred into its value features 𝐕
--
520-
521-where L2 (∗) denotes the mean square error function. It should be noted
522:that the memory features 𝐌 are randomly initialized by Xavier initialization [17]. While, the feature center aggregation loss 𝐿𝐹 𝐶𝐴 aims to
523-make the feature center of each modality from the same identity close
524:to their feature center in the memory features 𝐌, i.e.,
525-𝑁 ( (
526-)
--
555-
556-Furthermore, the output features 𝐅𝑜,𝑚,1 from the ﬁrst reasoning block
557:will be taken as the new query features, which, together with the knowledge features 𝐊𝐍, will be further fed into the next reasoning block for
558:inferring more modality-shared information. After that, the output features of the last reasoning block will be taken as our ﬁnal inferred features 𝐅𝑖𝑛𝑓 ,𝑚 ∈ 𝑅1×𝐷 . Finally, an identity classiﬁer will be further performed on those inferred features 𝐅𝑖𝑛𝑓 ,𝑚 to predict their identities, i.e.,
559-
560-(15)
--
567-Linear(∗, 𝛼𝑖𝑛𝑓 ) denotes a linear layer with its parameters 𝛼𝑖𝑛𝑓 for classiﬁcation.
568-Moreover, the triplet hetero center loss and the identity loss are also
569:performed on the inferred features 𝐅𝑖𝑛𝑓 ,𝑚 to make them be identitydiscriminative, i.e.,
570-
571-𝐐𝟏 = Linear(𝐌, 𝛼𝑞1 ), 𝐊𝟏 = Linear(𝐌, 𝛼𝑘1 ), 𝐕𝟏 = Linear(𝐌, 𝛼𝑣1 ),
--
580-be computed by
581-
582:By virtue of the two loss functions, the memory features 𝐌 can automatically store the information of each identity within the training set
583-in the training stage.
584:Establishing our model’s ‘knowledge and experience’: As shown
585:in Fig. 3, the knowledge module will further feed those memory features
586-𝐌 into several stacked Transformer blocks to explore their relations and
587:transfer the memories into the model’s ‘knowledge and experience’.
588:Here, we take the ﬁrst Transformer block as an example for the introduction. Speciﬁcally, the input memory features 𝐌 will be ﬁrst fed
589-into a self-attention layer to explore the interactions across the features
590-of diﬀerent identities, i.e.,
--
595-
596-Then, the simility weights 𝐰𝑚,1 ∈ 𝑅1×𝑁 between the query features 𝐅𝑞,𝑚,1
597:and the knowledge features 𝐊𝐍 are computed by
598-
599-where 𝐂𝑛𝑠,𝑅 and 𝐂𝑛𝑠,𝐼 denote the feature centers of the RGB images and
--
627-3.4. Fusion module
628-A simple fusion module is further designed to fuse the features 𝐅𝑠,𝑚
629:extracted from the input images and their inferred features 𝐅𝑖𝑛𝑓 ,𝑚 for VIReID. Speciﬁcally, the ﬁnal person features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 for VI-ReID will be
630-obtained by feeding the features 𝐅𝑠,𝑚 and 𝐅𝑖𝑛𝑓 ,𝑚 into the fusion module,
631-where they are fused by
632-
633:where 𝐅𝐾𝑁,1 denotes the transferred knowledge features in the ﬁrst
634-Transformer block. FFN(∗) denotes a feed-forward layer. Norm(∗) denotes a normalization layer.
635-Accordingly, the features 𝐅𝐾𝑁,1 will be fed into the next Transformer
636-block and the output features of the last Transformer block are taken
637:as the ﬁnal knowledge features 𝐊𝐍 ∈ 𝑅𝑁×𝐷 . By doing so, the memories of our model will be transferred as our model’s ‘knowledge and
638:experience’.
639-
640-𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 = 𝐅𝑠,𝑚 + 𝐅𝑖𝑛𝑓 ,𝑚 .
--
642-(25)
643-
644:Fig. 5 shows the distributions of the modality-shared features 𝐅𝑠,𝑚
645:and the ﬁnal fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 . From Fig. 5, it can be easily seen
646-that the ﬁnal fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 of the same identities are more compact than the modality-shared features 𝐅𝑠,𝑚 . This indicates that those
647:inforamtion inferred from our model’s “knowledge and experience” can
648-enhance the discriminability of the ﬁnal modality-shared features.
649-5
--
655-Fig. 4. Illustration of the reasoning module.
656-
657:Fig. 5. Distributions of the modality-shared features 𝐅𝑠,𝑚 (a) and the ﬁnal fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 (b) from diﬀerent identities. Diﬀerent colors denote diﬀerent
658-identities.
659-
--
1203-We stop training our model until convergence or reaching 60 epochs.
1204-
1205:4.5. Ablation study
1206:4.5.1. Eﬀectiveness of using the inferred features from our model’s
1207:‘knowledge and experience’
1208-In this subsection, we qualitatively verify our proposed model.
1209:Speciﬁcally, we ﬁrst use the modality-shared features 𝐅𝑠,𝑚 from the input images, the inferred features 𝐅𝑖𝑛𝑓 ,𝑚 from our proposed KET and
1210-their fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 , respectively, to obtain their own matching results on SYSU-MM01. Then, we statistically analyze their rightly
1211-matched identities according to the average precision (AP) for each identity with a threshold of 75 %. Accordingly, we draw a Venn diagram in
--
1311-
1312-Table 4
1313:Evaluation for each component on SYSU-MM01 dataset.
1314-Methods
1315-
--
1384-the discriminabilities of the modality-shared features 𝐅𝑠,𝑚 , thus helping
1385-to identify more persons.
1386:4.5.2. The impact of each component in our proposed model
1387-Several experiments are conducted to quantitatively evaluate the
1388:eﬀectiveness of each component in our proposed model. As shown in
1389-Table 4, the baseline model (termed by ‘Base’) is obtained by removing
1390:the proposed memory module from our proposed model and only using
1391-the last level of modality-shared features for VI-ReID. ‘Mem’ denotes
1392:the memory features 𝐌 in the Knowledge and Experience Transformer.
1393:‘KM’ denotes the knowledge module in the Knowledge and Experience
1394-Transformer. ‘RM’ denotes the reasoning module in the Knowledge and
1395-Experience Transformer. All the models are trained by using the same
--
1401-𝐿𝐹 𝐶𝐴 can further boost our model’s performance. The results of
1402-‘Base+Mem+KM+RM+𝐿𝐹 𝐴 + 𝐿𝐹 𝐶𝐴 ’ indicate that transferring those
1403:memory features into our model’s experience can further improve the
1404:performance. This proves that our proposed model can save the information of each identity in the training set as our model’s ‘memory’ and
1405:transfer them as our model’s ‘experience’ for inferring more information.
1406-Finally, the results of ‘Base+Mem+KM+RM+ML+𝐿𝐹 𝐴 + 𝐿𝐹 𝐶𝐴 ’ indicate that using multi-level features can further improve our model’s
1407-performance, since there is some complementary information among the
1408:features of diﬀerent levels. Meanwhile, it can also be seen that employing the knowledge module and the reasoning module will signiﬁcantly
1409-reduce the inference speeds of our proposed model. The computational
1410:complexity of the knowledge module is higher than that of the reasoning module because the knowledge module has more layers [See Section 4.5.3 and 4.5.4]. We will discuss this drawbacks in Section 4.6.
1411-
1412-Fig. 6. Venn diagram obtained by computing the intersection and union of the
1413-rightly matched identities with the modality-shared features 𝐅𝑠,𝑚 from the input
1414:images, the inferred features 𝐅𝑖𝑛𝑓 ,𝑚 from our model’s ‘knowledge and experience’
1415-and their fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 , respectively.
1416-
1417-identities for statistic analysis. Here, the percentage values denote the
1418-ratio of those rightly matched identities by using one type of features
1419:to the total rightly matched identities by using the modality-shared features 𝐅𝑠,𝑚 , the inferred features 𝐅𝑖𝑛𝑓 ,𝑚 and their fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 .
1420-From Fig. 6, it can be observed that the modality-shared features 𝐅𝑠,𝑚
1421-obviously play a dominant role for VI-ReID, i.e., they can rightly match
1422:about 80 % identities. This is unsurprising, since the human visual system also mainly uses the information extracted from the input images for
1423:matching. The inferred features 𝐅𝑖𝑛𝑓 ,𝑚 rightly match about 28 % identities among all the identities, which indicates that our proposed model
1424:can indeed infer some usefull information from its ‘knowledge and experience’ for VI-ReID. Moreover, within those 28 % identites, there are
1425:about 4 % identities that can be rightly matched by using the inferred
1426-features 𝐅𝑖𝑛𝑓 ,𝑚 , but are wrongly matched by using the modality-shared
1427-features 𝐅𝑠,𝑚 . This means that our proposed CKT successfully imitates the
1428:human visual system to some extent and can infer some new modalityshared information from its ‘knowledge and experience’ to complement
1429-the modality-shared features 𝐅𝑠,𝑚 from the input images, thus boosting
1430-performance. Finally, in addition to those identities rightly matched by
1431:using the modality-shared features 𝐅𝑠,𝑚 and the inferred features 𝐅𝑖𝑛𝑓 ,𝑚 ,
1432:about 16 % more identiﬁes are rightly matched by using their fused features 𝐅𝑓 𝑢𝑠𝑒𝑑,𝑚 . This also indicates that the inferred features can enhance
1433-
1434:4.5.3. Evaluations on the number of transformer blocks in the knowledge
1435-module
1436-In this subsection, some experiments are further conducted to search
1437:the best number for the Transformer blocks in the knowledge module.
1438-Here, the multi-level features are not employed in this subsection. The
1439:evaluation results for the number of Transformer blocks in the knowledge module are shown in Table 5. It can be seen that our model’s performance has a little ﬂuctuations with the number of Transformer blocks in
1440-8
1441-
--
1445-
1446-Table 5
1447:Evaluation results for the number of Transformer blocks in the knowledge module.
1448-
1449-Table 8
--
1611-
1612-5. Conclusion
1613:In this paper, inspired by the human visual system, we have proposed a novel Cross-modality Knowledge and Experience Transformer
1614-(CKT) for VI-ReID, which can further infer more modality-shared features about the person for boosting performance on top of the widelyused modality-shared features extracted from the input images. Speciﬁcally, this model contains a VI-ReID subnetwork, which can extract those
1615-discriminative modality-shared features from the input images. Moreover, it designs a Knowledge and Experience Transformer to memorize
1616:and transfer the modality-shared person information of all the identities from the training set as its ‘knowledge and experience’ for VI-ReID.
1617:By virtue of the VI-ReID subnetwork and the knowledge, our proposed
1618:model not only extracts the modality-shared information within the input images but also infers more information from its ’knowledge and
1619:experience’, thus signiﬁcantly boosting performance. Extensive experiments have also veriﬁed our proposed model’s superiorities over existing
1620:models as well as the eﬀectiveness of each individual component.
1621-
1622-Table 7
--
1658-26.67
1659-
1660:the knowledge module. Among that, when the number of Transformer
1661-blocks is set to 4, our proposed model exhibits the highest performance
1662-among all the models. Therefore, we set the number of Transformer
1663-blocks to 4 in our ﬁnal model. Besides, the computational complexity of
1664:the knowledge module is increased as the number of layers grows.
1665-
1666-CRediT authorship contribution statement
1667-Nianchang Huang: Writing – original draft, Validation; Jialiang
1668:Wang: Visualization, Formal analysis, Data curation; Qiang Zhang:
1669-Writing – review & editing, Funding acquisition; Jungong Han: Writing
1670-– review & editing; Jin Huang: Writing – review & editing, Supervision.
--
1675-the best number for the reasoning blocks in the reasoning module. Here,
1676-the multi-level features are also not employed in this subsection. The
1677:evaluation results for the number of Transformer blocks in the knowledge module are shown in Table 6. It can be seen that our model also
1678-shows a little performance ﬂuctuation with diﬀerent number of reasoning blocks in the reasoning module. Especially, when the number of
1679-reasoning blocks is set to 1, our proposed model achieves the best performance. Therefore, we set the numbers of Transformer blocks to 1 in
--
1698-which results in a disadvantage in terms of inference speed compared
1699-to existing methods. Therefore, fully leveraging the advantages of the
1700:model’s ’knowledge and experience‘ while simultaneously enhancing its
1701-computational eﬃciency will be the core focus of our future research.
1702-Application expansion: At present, our method is exclusively employed for the VI-ReID task. However, our core concept holds the potential to be generalized to other cross-modal retrieval tasks. As shown
--
1738-[12] Q. Zhang, C. Lai, J. Liu, N. Huang, J. Han, FMCNet: feature-level modality compensation for visible-infrared person re-identiﬁcation, in: Proceedings of the IEEE/CVF
1739-Conference on Computer Vision and Pattern Recognition, 2022, pp. 7349–7358.
1740:[13] Y. Li, T. Zhang, X. Liu, Q. Tian, Y. Zhang, F. Wu, Visible-infrared person reidentiﬁcation with modality-speciﬁc memory network, IEEE Trans. Image Process.
1741-31 (2022) 7165–7178.
1742-[14] X. Li, Y. Lu, B. Liu, Y. Liu, G. Yin, Q. Chu, J. Huang, F. Zhu, R. Zhao, N. Yu, Counterfactual intervention feature transfer for visible-infrared person re-identiﬁcation, in:

exec
/bin/zsh -lc "pdftotext \"4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf\" - | rg -n -C 2 \"visual|Visualization|t-SNE|frequency|band|Fig\\. 6|Fig\\. 7|attention\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
37-ABSTRACT
38-Visible-Infrared Person Re-identification (VI-ReID) is a cross-modality retrieval task that involves searching
39:for a specific pedestrian across different modalities (visible or infrared). Existing methods have explored crossmodality alignment from different perspectives, including spatial-domain cues and frequency-domain modeling.
40:Nevertheless, how to adaptively capture modality-specific frequency characteristics and effectively integrate
41-them into cross-modality feature alignment remains an open challenge. To address this challenge, this work
42-presents a novel and efficient Spatial and Spectrum Self-adaptive Synergy Network (4SNet) for the VI-ReID
43-task, which incorporates an Adaptive Frequency Filter (AFF) module and a Spectrum Synergy Module (SSM)
44:to extract robust cross-modality-invariant features. Specifically, in contrast to traditional fixed convolutionkernel operations, the AFF module reconstructs a set of learnable frequency bases by adopting a dual spatialand channel attention-guided strategy, which dynamically synthesizes a customized Spectrum filter for each
45:input. Complementarily, the SSM decomposes features into multiple frequency bands, which explicitly decouple
46-identity-shared patterns from modality-specific traits and synergistically enhance them, thereby achieving
47-consistent feature representation across cross-modal pedestrian images at the cross-modality feature alignment
--
88-methods hinge on projecting data from diverse modalities into a unified
89-feature embedding space [17,18]. Subsequently, feature enhancement
90:techniques (e.g., attention-based mechanisms [19,20]) are employed
91-for recognizing patterns corresponding to the same pedestrian. Although these techniques diminish the reliance on additional models,
92-direct projection also struggles to entirely bridge the domain gaps
--
96-center-guided pair mining strategy to create diverse auxiliary embedding spaces between visible and infrared modalities, thereby reducing
97-their differences. Another example is FDNM [24], which employs a
98:salience-guided and amplitude nuance mining strategy to extract crossmodality frequency domain information, thereby decreasing modality
99-discrepancies and enhancing VI-ReID performance.
100-Although these methods have achieved promising performance,
101-they still face two main issues:
102-1. When it comes to mining intrinsic attributes within a single modality,
103:existing methods primarily rely on a fixed, pre-learned set of convolution kernels that are uniformly applied to samples from all modalities. As a result, these static operators fail to capture modalityspecific frequency-domain characteristics (such as structural textures and body contours) present within individual modalities. For
104-instance, a filter that is optimal for extracting fine textures in visible
105-images may inadvertently amplify interference noise when applied
--
108-network’s ability to discover highly discriminative, modality-specific
109-features.
110:2. Regarding feature alignment across different modalities, although several recent studies have introduced frequency-domain modeling,
111-many existing approaches still rely primarily on spatial-domain
112-cues, such as appearance attributes and relative positional relationships, for cross-modality alignment. However, due to the substantial distributional differences between visible and infrared images,
113:spatial representations are often sensitive to modality-specific variations, which limits their robustness for alignment. In this context, frequency-domain information (encompassing global structure,
114-coarse appearance distribution, and complementary edge-like responses) provides features that are relatively more stable across
115-modalities. Therefore, effectively exploiting these spectral characteristics remains crucial for improving cross-modality feature alignment.
--
117-Fig. 1. Motivation for the overall architecture of the 4SNet framework. (a)
118-Grayscale Fast Fourier Transform (FFT) magnitude spectra illustrate that
119:infrared images concentrate most of their energy in low-frequency bands,
120:whereas visible images contain richer medium- and high-frequency details.
121-This clear spectral divergence suggests that different modalities exhibit distinct
122:frequency responses, making fixed shared filters inadequate for capturing
123-modality-specific discriminative features. (b) Motivated by this observation,
124:4SNet introduces two complementary frequency-domain modules, namely the
125-Adaptive Frequency Filter (AFF) and the Spectrum Synergy Module (SSM). A
126-comparison between a baseline fixed stem (denoted as Conv1) and AFF under
127-visible and infrared inputs shows that fixed shared filters are suboptimal for
128-early-stage feature extraction, which justifies the use of AFF for input-adaptive
129:and modality-specific frequency filtering. Meanwhile, the visualization of
130:low-, middle-, and high-frequency decompositions reveals that different spectral bands preserve complementary structural and fine-grained details. The
131:SSM explicitly models this band-wise decomposition and their synergistic
132-relationships to achieve more robust cross-modality feature alignment.. (For
133-interpretation of the references to colour in this figure legend, the reader is
--
135-
136-2. For the second problem, the SSM decomposes the characteristic
137:spectrum into low-, medium-, and high-frequency components.
138:These components are then processed in parallel using band-specific
139:operators (e.g., lightweight convolutional blocks). This strategy retains stable low-frequency body structures and coarse appearances,
140:while exploiting complementary mid-to-high frequency variations
141-and fine edge details.
142-
--
147-
148-Finally, a fusion module is utilized to integrate these features,
149:with a particular focus on the enhanced low-frequency components to
150-guide the alignment of cross-domain features. Extensive experiments
151-on three public VI-ReID datasets (SYSU-MM01 [25], RegDB [26], and
152-LLCM [23]) demonstrate that 4SNet achieves competitive performance.
153:Ablation studies and feature visualizations further validate its superior
154-retrieval capability.
155-
156-1. For the first issue, the AFF module dynamically generates inputspecific filters to address the limitations of static convolutional
157:kernels. The module first learns a set of frequency bases from images
158:using spatial attention. These bases are then adaptively combined
159:under channel attention guidance, synthesizing customized filters
160:that extract modality-inherent frequency-domain features.
161-2
162-
--
171-
172-2.3. Frequency domain analysis in vision tasks
173:The frequency domain offers a complementary perspective for image representation and understanding, providing intrinsic robustness
174-against certain spatial transformations and local perturbations [40].
175:By decomposing visual signals into components that capture global
176:structure and fine-grained details at different spectral scales, frequencydomain analysis enables models to disentangle information that is often
177:entangled in the spatial domain. As a result, frequency-aware modeling
178-has been successfully applied across a wide range of computer vision
179-tasks. In particular, studies in domain adaptation and generalization
180:have demonstrated that low-frequency components, which predominantly encode global shape and coarse structural information, tend to
181:be more stable and domain-invariant than high-frequency components
182-that capture texture and local appearance variations [1,41]. These
183:properties make frequency representations especially suitable for scenarios involving significant domain shifts. Beyond domain adaptation,
184:frequency-domain insights have also been leveraged to improve adversarial robustness, where sensitivity to high-frequency perturbations
185-plays a critical role [42].
186-Motivated by this observation, several works have incorporated
187:frequency-domain operations directly into deep neural networks. Some
188:approaches have highlighted the effectiveness of frequency-aware channel attention in selectively emphasizing informative spectral responses
189:[43,44], while others have gone a step further by partially or fully replacing standard spatial convolutions with frequency-domain counterparts [45,46]. More recently, wavelet-guided designs such as PST [47]
190:have demonstrated the benefits of frequency-aware decomposition in
191:surface-defect detection, where carefully separating frequency components helps enhance meaningful details while suppressing irrelevant
192:noise. Along this line, several recent VI-ReID methods, including FDMNet [48] and FDNM [24], have explored frequency-domain modeling
193:through amplitude–phase decomposition, motivating further investigation into frequency-aware designs for cross-modality matching
194-problems.
195:Contribution. While prior frequency-based methods such as FDMNet [48] and FDNM [24] have achieved promising performance
196:through amplitude–phase decomposition, they typically treat frequency
197-modeling as a single, unified mechanism for reducing modality discrepancy. In these approaches, amplitude and phase components are
198-jointly manipulated to encourage modality-invariant representations,
199:implicitly assuming that a common frequency processing strategy is
200-sufficient for both intra-modality feature extraction and inter-modality
201:alignment. As a consequence, the intrinsic differences in frequency
202-distribution characteristics between visible and infrared modalities are
203-not explicitly addressed at different stages of the network. In contrast,
204:4SNet introduces a role-decoupled frequency modeling strategy, in
205:which frequency information is explicitly assigned to two complementary and well-defined roles within the framework. Specifically, the
206-AFF operates at the input stage in a dual-stem, non-shared manner
207:to perform modality-specific and input-adaptive frequency filtering,
208-enabling visible and infrared images to be characterized by customized
209:frequency responses before entering the shared backbone. The Spectrum Synergy Module (SSM) addresses cross-modality alignment as
210:a separate modeling problem by adopting a coarse low-, medium, and high-frequency band decomposition and performing band-wise
211-processing and fusion. This design emphasizes robust shared structural
212-cues while preserving complementary discriminative details, thereby
--
255-equal-sized horizontal segments, which fails to account for changes
256-in pose or body part location). Different from these methods, 4SNet
257:addresses supervised VI-ReID from a frequency-modeling perspective.
258-
259-3. Proposed 4SNet framework
--
269-infrared (𝐼𝐼 ) images are fed into two independent, non-shared AFF modules (𝐴𝐹 𝐹𝑉 and 𝐴𝐹 𝐹𝐼 ) to dynamically extract initial modality-specific features (𝐹𝑉 and
270-𝐹𝐼 ). These features are then concatenated along the batch dimension to form 𝐹𝑚𝑖𝑥𝑒𝑑 . Subsequently, 𝐹𝑚𝑖𝑥𝑒𝑑 is processed by a fully shared-weight backbone, where
271:the SSM is embedded to perform coarse band-wise spectrum synergy for robust cross-modality alignment. Finally, the network is optimized jointly by triplet (𝐿𝑡𝑟𝑖 )
272-and identity (𝐿𝑖𝑑 ) losses to ensure robust modality-invariant feature alignment.
273-
274:frequency characteristics of a series of images, thereby extracting the
275-intrinsic features of a single-modality. Then, the SSM in Section 3.3 implements cross-modality feature alignment. Section 3.4 presents the loss
276-function and the learning procedure used to train the 4SNet framework.
--
279-personalized features commonly encountered in real-world scenarios.
280-To address this limitation, we propose the AFF module. As has been
281:previously mentioned, AFF advances frequency modeling over existing
282:frequency-based methods by performing modality-specific and inputadaptive filtering at the early feature extraction stage. In this way, the
283:network can generate sample-specific frequency responses for visible
284-and infrared inputs before shared-backbone learning. The key idea is
285-to move away from fixed convolution kernels and instead dynamically
286-generate customized filters for each input during training. As illustrated
287:in Fig. 2, the AFF module architecture consists of three main components: attention-guided feature modulation, dynamic kernel synthesis,
288:and frequency-domain filtering. Each of these components is described
289-in detail below:
290-
--
306-modality-invariant feature.
307-
308:Attention-Guided Feature Modulation. To emphasize prominent features according to their spatial and channel attention mechanisms,
309-we draw inspiration from CBAM [49] and adopt spatial and channel
310:attention mechanisms to locate key regions in the original image,
311-respectively. These mechanisms allow obtaining more accurate highlevel semantic information and preserving spatial domain features.
312:Firstly, a spatial attention operation is used to calculate a weight map
313-as:
314-(
--
317-(1)
318-where 𝐴𝑠 ∈ R𝐻×𝑊 denotes the weight map after executing the spatial
319:attention operation, and 𝐻 and 𝑊 are the height and width of the feature map, respectively. Here, 𝑓 7×7 represents a convolutional layer with
320-a 7 × 7 kernel size, and 𝜎(⋅) denotes the sigmoid function. Subsequently,
321-this produced weight map is employed to modulate the input feature
--
339-where ⊗ denotes element-wise multiplication. In parallel, after Global
340-Average Pooling (GAP), the original input feature 𝑋 is processed
341:through a similar channel attention operation to get 𝑋 𝐶 . Subsequently,
342:to acquire frequency domain information more effectively from singledomain images, a Discrete Fourier Transform (DFT) operation is introduced, which transforms the modulated feature map 𝑋 ′ ∈ R𝐶×𝐻×𝑊
343:into the frequency domain 𝑋 𝐷 ∈ C𝐶×𝐻×𝑊 .
344-
345-3.3. Spectrum Synergy Module (SSM)
--
348-and coarse appearance patterns) across different modalities. This inconsistency leads to a decline in cross-modality feature alignment
349-performance. To tackle this problem, the proposed SSM, illustrated in
350:Fig. 2, first decomposes the feature spectrum into different frequency
351:bands. Unlike existing frequency-based methods that mainly model
352-amplitude-phase interactions, SSM treats cross-modality alignment as a
353:dedicated coarse band-wise synergy problem. By explicitly decomposing the spectrum into low-, mid-, and high-frequency components, SSM
354-emphasizes robust shared structures while retaining complementary
355-discriminative cues for inter-modality alignment. Then, it employs
356:specialized operators to conduct parallel processing on these bands.
357-Finally, it synergistically fuses them to generate an enhanced feature, thereby improving cross-modality feature alignment, which are
358-described as follows:
--
360-Dynamic Kernel Synthesis. The AFF module synthesizes an adaptive
361-convolutional filter through the dynamic combination of learnable
362:frequency bases, which are represented as 𝑁 learnable base kernels denoted as {𝐾1 , 𝐾2 , … , 𝐾𝑁 }. However, differently from recent work using
363:learnable frequency modules [50], our approach begins by initializing
364-the filter based on binary values (0 and 1), directly parameterizing it
365:in the frequency domain, and constructing orthogonal constraints using
366-non-overlapping Fourier indices. Specifically, the orthogonality of the
367-base kernel is ensured by employing a set of non-overlapping binary
368:masks {𝑀1 , … , 𝑀𝑖 , … , 𝑀𝑁 } in the frequency domain. Each mask 𝑀𝑖
369-is assigned a value of 1 exclusively within a specific, non-overlapping
370-region of the Fourier spectrum, and 0 elsewhere. The 𝑖th learnable
371-base kernel 𝐾𝑖 is computed as the element-wise product of a learnable
372-complex tensor 𝑇𝑖 and its corresponding fixed binary mask 𝑀𝑖 in the
373:frequency domain. To ensure that each 𝐾𝑖 only captures patterns within
374:its preassigned frequency subspace, we enforce the condition 𝑀𝑖 ∩𝑀𝑗 =
375-∅ (𝑖 ≠ 𝑗), i.e., the binary masks are non-overlapping in the Fourier
376:domain. Inspired by SE-Net [51], we introduce an attention mechanism
377-to adaptively generate both the channel reweighting vector 𝐰𝑐 and the
378-kernel aggregation coefficients 𝜶. Specifically, the input feature 𝐗 is
--
389-R𝐶×𝐻×𝑊 (where 𝐶, 𝐻 and 𝑊 denote the channel, height and width of
390-the feature map, respectively) from the intermediate block layer, we
391:first convert it into its frequency-domain representation  (𝑋mid ) via
392-a two-dimensional discrete Fourier transform (2D DFT). To partition
393:the spectrum according to frequency content, we employ a set of
394:predefined, non-overlapping frequency masks: a low-pass mask 𝑀𝐿 ,
395-a middle-pass mask (𝑀𝑀 ), and a high-pass mask (𝑀𝐻 ). To ensure
396:smooth, non-binary transitions between frequency bands and prevent
397-the introduction of ringing artifacts, which are commonly observed
398-in ideal filters, we use Butterworth filtering [52]. As such, filtering
399-masks are constructed using 𝑛𝐵 -order Butterworth filters and rely on
400:the Euclidean distance 𝐷(𝑢, 𝑣) (with 𝑢 and 𝑣 representing the frequencydomain coordinates for the horizontal and vertical spatial frequencies
401-within the feature map) to the center of the spectrum. We define two
402:fixed cutoff thresholds in the normalized frequency domain, 𝜓1 and 𝜓2
403-(𝜓1 < 𝜓2 ). The mask is computed as:
404-
--
408-ReLU activation function, and 𝜎(⋅) is the Sigmoid function. The second
409-branch generates the aggregation coefficients 𝜶 ∈ R𝑁 for the 𝑁
410:learnable frequency bases:
411-𝜶 = Softmax(𝐖3 (𝛿(𝐖1 (GAP(𝐗))))),
412-
--
437-where 𝐷(𝑢, 𝑣) = (𝑢 − 𝐻∕2)2 + (𝑣 − 𝑊 ∕2)2 , 𝐷max = (𝐻∕2)2 + (𝑊 ∕2)2 ;
438-𝜓1 and 𝜓2 are hyper-parameters; and 𝑛𝐵 denotes the filter order,
439:which dictates the steepness of its frequency-response curve. A higherorder filter results in a more abrupt transition between the pass-band
440:and the stop-band, bringing the filter closer to an ideal one. This
441-construction method ensures smooth, non-binary transitions between
442:frequency bands, thereby preventing the generation of artifacts. The
443:computed masks are then applied to the frequency-domain feature
444-map through element-wise multiplication to isolate the corresponding
445-spectrum components:
--
455-kernel in the final dynamic filter construction.
456-Frequency-Domain Filtering. The synthesized spatial-domain kernel
457:𝐾𝑑𝑦𝑛 is transformed into its frequency-domain equivalent,  (𝐾𝑑𝑦𝑛 ), via
458-a DFT. This dynamic filter is then applied to the modulated feature
459-representation  (𝑋 ′ ):
--
467-∑
468-[𝛼1 , … , 𝛼𝑁 ] satisfies 𝑁
469:𝑖=1 𝛼𝑖 = 1. After that, each learnable frequency
470-base kernel 𝐊𝑖 is first transformed into the spatial domain by the inverse
471-discrete Fourier transform (IDFT), and then reweighted by 𝐰𝑐 along the
--
478-𝑀𝐿 (𝑢, 𝑣) =
479-
480:𝐹band =  (𝑋mid ) ⊗ 𝑀band ,
481-
482:where band ∈ {𝐿, 𝑀, 𝐻},
483-
484-(11)
485-
486:where 𝐹band denotes feature representations comprising components
487:from the low (band = 𝐿), medium (band = 𝑀), and high frequency
488:(band = 𝐻) bands.
489-
490-(6)
491-
492:Parallel Band-Specific Processing. Each frequency component 𝐹𝑏𝑎𝑛𝑑
493:is subsequently processed by a dedicated parallel branch. This enables the network to learn transformations that are specifically tailored to the nature of the information within each band. Initially,
494-
495-The final output feature map, denoted as 𝑌 , is acquired by applying
496-the Inverse Discrete Fourier Transform (Inverse DFT) to the obtained
497:frequency-domain representation, i.e., 𝑌 = IDFT( (𝑌 )).
498-5
499-
--
502-M. Xiong et al.
503-
504:̂band ∈ R𝐵×𝐶×𝐻×𝑊 is the feature tensor for the corresponding
505-where 𝑋
506-̂ This dynamic weighting mechanism allows the
507:band, indexed from 𝑋.
508-network to adaptively emphasize the most discriminative spectrum
509-components. A residual connection from the original input 𝑋mid is
--
512-each 𝐹𝑏𝑎𝑛𝑑 is transformed back into the spatial domain using IDFT.
513-Subsequently, the resulting spatial representation is fed through a
514:lightweight, band-specific convolutional block as:
515-′
516-𝑋𝑏𝑎𝑛𝑑
517:= ConvBlockband (IDFT(𝐹band )),
518-
519-(12)
520-
521:where ConvBlockband is designed as an efficient sequential block composed of two convolutional layers: first, a 3 × 3 convolutional layer
522-is employed for spatial feature extraction, followed by Batch Normalization (BN) and Rectified Linear Unit (ReLU). Then, a 1 × 1
523-convolutional layer is included for channel information mixing, also followed by BN and ReLU. The parameters of ConvBlock𝐿 , ConvBlock𝑀 ,
524-and ConvBlock𝐻 are learned independently.
525:In addition, since low-frequency components encapsulate the most
526-critical and modality-invariant structural information, we apply an
527:additional channel attention mechanism specifically to them. Concretely, only the output 𝑋𝐿′ from the low-frequency branch undergoes further channel-wise modulation via a Squeeze-and-Excitation
528-(SE) block [51], resulting in the enhanced feature 𝑋𝐿′′ . Meanwhile,
529:′ and high-frequency 𝑋 ′ features are
530:the intermediate-frequency 𝑋𝑀
531-𝐻
532-directly passed forward for subsequent fusion without further modification. This differentiated parallel architecture enables the model
533:to retain robust structural information from the low-frequency branch
534-and enhance its discriminative capacity through the SE block, while
535:simultaneously preserving intermediate-frequency local variations and
536:high-frequency edge-like fine-detail responses as complementary cues,
537-thereby improving overall feature representation.
538-
--
550-
551-• Feature refinement. The first step is to refine the shared features. To
552:efficiently learn robust cross-band features, we use a shared fusion
553-network 𝛷fusion to process the enhanced features. Then, the final
554:enhanced feature of the low-frequency branch is defined as 𝑋𝐿′′ =
555:SE-Block(𝑋𝐿′ ), while the mid-frequency and high-frequency features
556-′ and 𝑋 ′ , respectively. Subsequently, the feaare denoted as 𝑋𝑀
557-𝐻
--
559-dimension:
560-
561:exp(𝑤band )
562-.
563-b∈{𝐿,𝑀,𝐻} exp(𝑤b )
--
571-detail. First, we outline the experimental setup (Section 4.1), which
572-includes datasets, evaluation metrics, implementation details, and parameter settings. Next, we compare the performance of our method
573:with existing approaches and analyze failure cases (Section 4.2). Finally, we provide an ablation study (Section 4.3) and a visualization
574-features and retrieval results (Section 4.4) to validate the contributions
575-of key components of the proposed framework.
--
577-(14)
578-
579:Since 𝛷fusion is applied to all band-specific features in the stack, it
580-learns a shared transformation that extracts common salient patterns.
581-• Attention-based aggregation strategy. The output 𝐹fused is reshaped
582-̂ ∈
583:to disentangle the per-band features, resulting in a tensor 𝑋
584:R𝐵×𝑁×𝐶×𝐻×𝑊 . We employ the attention mechanism over the frequency bands to aggregate these refined features. A set of learnable scalar parameters, {𝑤𝐿 , 𝑤𝑀 , 𝑤𝐻 }, is normalized via a softmax
585:function to produce attention weights 𝐚 as:
586:𝑎band = ∑
587-
588-𝐵
--
616-
617-′
618:Synergistic Fusion. The enhanced, band-specific features 𝑋band
619-(with
620:band ∈ {𝐿, 𝑀, 𝐻}) are ultimately fused into a single, comprehensive
621-representation. This process includes a shared feature refinement stage,
622:followed by an attention-based aggregation:
623-
624-′
--
653-
654-The final output of the SSM, 𝑌out , is computed as a weighted sum
655:of the refined, band-specific features:
656-∑
657:̂band ,
658-𝑌out =
659:𝑎band ⋅ 𝑋
660-(16)
661:band∈{𝐿,𝑀,𝐻}
662-
663-6
--
1029-Infrared mode, and 94.31% Rank-1 and 90.61% mAP in the Infrared to
1030-Visible mode, outperforming most comparison baselines. Overall, while
1031:some recent frequency-based methods may achieve slightly higher peak
1032-metrics on specific sub-settings, 4SNet maintains highly competitive
1033-performance across diverse scenarios. The particularly strong mAP
--
1292-
1293-Table 4
1294:Impact of attention aggregation strategies in AFF on the
1295-SYSU-MM01 dataset.
1296-
--
1298-RegDB and competitive Rank-10 performance on LLCM, it is slightly
1299-lower than some recent methods on certain Rank-1 metrics. One possible reason, as supported by the ablation study in Table 5, is that SSM
1300:adopts a relatively coarse three-band decomposition to reduce sensitivity to modality-specific noise. While this design improves overall
1301-retrieval quality and robustness, it may also limit the modeling of some
1302:fine-grained frequency details. As a result, methods that emphasize
1303-finer-grained local detail modeling may achieve slightly better Rank-1
1304-accuracy in relatively simple indoor environments. Overall, 4SNet is designed to favor more robust and generalizable cross-modality retrieval
--
1343-strategy consistently outperforms both sequential counterparts across
1344-all settings. This empirical evidence supports our theoretical hypothesis: unlike standard spatial-domain features, the channel dimension
1345:(which selects relevant frequency bases) and the spatial dimension
1346-(which locates spatial importance) act as two orthogonal modulators
1347:during frequency filter synthesis. The parallel formulation preserves
1348-both contexts independently and comprehensively, yielding optimal
1349:dynamic frequency adaptation for the AFF module.
1350-
1351-4.3. Ablation analysis
1352:We continue our discussion by inspecting the impact of frequency
1353:bands, the location of the SSM, and the hyperparameter values.
1354-Performance contribution of each component. To assess the contribution of the proposed AFF module and SSM, we conduct a comprehensive
1355-ablation study on the SYSU-MM01 and LLCM datasets. The results,
--
1370-results clearly demonstrate the effectiveness of the proposed modules.
1371-
1372:Impact of frequency bands. To gain deeper insight into the role of
1373:the low-, mid-, and high-frequency bands employed in the SSM, we
1374:perform a validation study on the SYSU-MM01 dataset. The corresponding results are reported in Table 6. As shown, each frequency
1375:band contributes differently to the overall performance. Importantly,
1376:the best results are obtained only when all three bands are jointly
1377:utilized, underscoring the complementarity of multi-frequency bands
1378-in improving the effectiveness of 4SNet.
1379:Impact of frequency partitioning strategy in SSM. We further clarify
1380:the asymmetric frequency decomposition design between AFF and SSM.
1381:AFF adopts an 𝑁-band split to capture fine-grained, input-specific
1382-details for intra-modality representation learning. In contrast, SSM uses
1383:a coarse 3-band split (Low/Mid/High), since cross-modality alignment
1384-requires more robust and abstract semantic cues. A finer partition
1385-in SSM may overfit to modality-specific noise, while the coarse split
1386-better preserves shared structure and suppresses irrelevant discrepancies. We also validate this design through an ablation study on SSM
1387-
1388:Impact of attention aggregation strategies in AFF. To quantitatively
1389-justify the architectural choice within the proposed AFF module, we
1390:conduct a validation study comparing different attention aggregation
1391-8
1392-
--
1396-
1397-Table 5
1398:Ablation study on the frequency partitioning strategy of the SSM on
1399-the SYSU-MM01 dataset.
1400-All Search
--
1424-86.85
1425-
1426:𝑁-band split (Same as AFF)
1427:Coarse 3-band split (L/M/H)
1428-
1429-Regarding the Butterworth filter order 𝑛𝐵 , increasing it from 1 to 2
--
1437-
1438-Table 6
1439:Contributions of different frequency bands within the SSM on the
1440-SYSU-MM01 dataset.
1441-SSM Component
--
1465-71.98
1466-
1467:4.4. Separation of learned features and visualization of retrieval results
1468-We end our discussion by offering a more intuitive comprehension
1469-of how 4SNet augments the feature space and enhances the retrieval
1470:accuracy. To this end, we develop three series of visualization analyses:
1471:feature distance distribution, t-SNE feature embedding, and qualitative
1472-retrieval results:
1473-
--
1483-SYSU-MM01
1484-
1485:• Feature distance distribution. First, we visualize the distributions of
1486-intra-class distances (pairs of the same identity across different modalities) and inter-class distances (pairs of different identities). As depicted in Fig. 5(a), the baseline model evinces a substantial overlap
1487-between the two distributions, suggesting poor feature discriminability. In contrast, Fig. 5(b) proves that 4SNet effectively separates the
--
1489-pushing inter-class samples further apart, which is consistent with our
1490-quantitative results.
1491:• t-SNE visualization of feature space. Next, we utilize t-SNE [25] to
1492-project the high-dimensional features of test samples from multiple identities onto a 2D plane for qualitative evaluation. In Fig.
1493-4, each color represents a unique person identity, whereas circles
--
1497-our method effectively diminishes both intra-class variance and the
1498-modality gap, while increasing the inter-class separation.
1499:• Visualization of retrieval results. To demonstrate the practical effectiveness of our enhancements, we present retrieval results from the
1500:SYSU-MM01 dataset in Fig. 6. In each example, the infrared (IR)
1501-query image is shown on the far left, followed by the top-5 retrieved
1502-visible (VIS) images from the gallery for both the baseline model
1503-and our proposed 4SNet. Correct matches are highlighted with green
1504:bounding boxes, while incorrect ones are marked in red. The visualization clearly underscores the superiority of our approach. The
1505-baseline model frequently fails to rank the correct match first and
1506-often includes multiple false positives among its top-5 results. In
--
1538-64.63
1539-
1540:partitioning strategies (Table 5). Compared with a unified 𝑁-band split,
1541:the proposed 3-band split achieves better performance, whereas the
1542-fine-grained strategy leads to a clear drop in mAP (e.g., 4.76% under
1543:All Search). Results show coarse frequency partitioning is optimal for
1544-robust visible-infrared alignment.
1545-Impact of the SSM location. Given that the SSM module operates on the
--
1554-Analysis of Hyperparameters. To investigate the impact of key hyperparameters on the performance of 4SNet, we conduct sensitivity analyses for both the AFF and SSM modules on the SYSU-MM01 (All-Search
1555-mode) dataset:
1556:• For the AFF module, we adjust the number of learnable frequency
1557-bases 𝑁 within the range of 1 to 10, as shown in Fig. 3(a). The results
1558-indicate that performance generally improves as 𝑁 increases from 1
1559:to 4, suggesting that a richer set of frequency bases helps capture
1560-diverse patterns. However, performance degrades when 𝑁 exceeds
1561-4, likely due to overfitting and the introduction of noisy, redundant
1562-bases. We thus uniformly set 𝑁 = 4 to balance representation capacity
1563-and generalization.
1564:• For the SSM module, we examine both the frequency thresholds 𝜓1
1565:and 𝜓2 as well as the Butterworth filter order 𝑛 employed in frequency
1566:band decomposition. As exposed by Table 8 and Fig. 3(b), raising
1567-𝜓1 from 0.10 to 0.20 consistently enhances performance, indicating
1568:that moderately expanding the low-frequency range captures more
1569-robust structural information. Similarly, increasing 𝜓2 from 0.50 to
1570-0.60 improves Rank-1 accuracy and mAP, suggesting that extending
1571:high-frequency coverage enables the network to leverage complementary higher-frequency responses for discrimination. Exceeding these
1572-values (e.g., 𝜓1 = 0.24 or 𝜓2 = 0.66) leads to a slight performance
1573:decline, likely due to redundant or noisy frequency information.
1574-
1575-5. Conclusions and future work
1576-This paper has presented 4SNet, a Spatial and Spectrum Selfadaptive Synergy Network, which addresses the VI-ReID task from
1577:a frequency-aware representation learning perspective. By explicitly
1578-accounting for the inherent spectral discrepancies between visible and
1579-infrared images, 4SNet moves beyond static convolutional modeling
--
1591-M. Xiong et al.
1592-
1593:Fig. 3. Hyperparameters analysis on the SYSU-MM01 dataset. (a) Impact caused by the size of the frequency bases 𝑁 in the AFF module. (b) Impact of the size
1594-of the Butterworth filter order 𝑛𝐵 within the SSM.
1595-
1596:Fig. 4. t-SNE visualization of test features. Circles/triangles denote visible/infrared modalities, and colors indicate identities. Compared to DEEN [23], 4SNet
1597-yields tighter intra-class clusters and clearer inter-class separation, demonstrating superior cross-modality discriminability.
1598-Table 8
--
1718-M. Xiong et al.
1719-
1720:Fig. 6. Qualitative top-5 retrieval comparisons on SYSU-MM01 between DEEN [23] and 4SNet, where green and red boxes denote correct and incorrect matches,
1721-respectively.
1722-
1723:and channel attention guidance, AFF enables modality-specific frequency responses to be emphasized at the earliest stage of feature
1724-extraction. This design allows 4SNet to preserve identity-relevant
1725-spectral patterns while suppressing modality-induced interference,
1726-leading to robust single-modality representations.
1727:• On the other hand and complementarily, the SSM realizes crossmodality alignment by treating it as a coarse, band-wise synergy
1728:problem instead of a fine-grained frequency matching task. Through
1729:explicit low-, mid-, and high-frequency decomposition and attentiondriven aggregation, SSM selectively reinforces modality-invariant
1730-structural cues, while retaining complementary discriminative details,
1731-thereby improving feature alignment consistency across modalities.
--
1787-& editing, Resources, Investigation. Sambit Bakshi: Writing – review
1788-& editing, Validation, Methodology, Investigation. Javier Del Ser:
1789:Writing – review & editing, Visualization, Validation, Project administration, Investigation. Khan Muhammad: Writing – review & editing, Validation, Supervision, Project administration, Formal analysis,
1790-Conceptualization.
1791-Declaration of competing interest
--
1864-alignment and affinity inference, in: IEEE/CVF Conference on Computer Vision
1865-and Pattern Recognition, 2023, pp. 11270–11279.
1866:[40] W. Sun, et al., Visible-infrared person re-identification via adaptive frequency
1867-mining and embedding, Digit. Signal Process. (2025) 105526.
1868-[41] Q. Xu, et al., A Fourier-based framework for domain generalization, in: IEEE/CVF
1869-Conference on Computer Vision and Pattern Recognition, IEEE, 2021, pp.
1870-14383–14392.
1871:[42] H. Wang, X. Wu, Z. Huang, E.P. Xing, High-frequency component helps explain
1872-the generalization of convolutional neural networks, in: IEEE/CVF Conference
1873-on Computer Vision and Pattern Recognition, IEEE, 2020, pp. 8684–8694.
1874:[43] Z. Qin, P. Zhang, F. Wu, X. Li, FCANet: Frequency channel attention networks,
1875-in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE,
1876-2021, pp. 783–792.
1877:[44] H. Yu, et al., Discovering attention-guided cross-modality correlation for
1878-visible–infrared person re-identification, Pattern Recognit. 155 (2024) 110643.
1879-[45] N.K.S. Behera, P.K. Sa, K. Muhammad, S. Bakshi, Large-scale person reidentification for crowd monitoring in emergency, IEEE Trans. Autom. Sci. Eng.
--
1887-for visible-infrared person re-identification, 2024, arXiv preprint arXiv:2401.
1888-01839.
1889:[49] S. Woo, J. Park, J.-Y. Lee, I.S. Kweon, CBAM: Convolutional block attention
1890-module, in: European Conference on Computer Vision, Springer, 2018, pp. 3–19.
1891-[50] L. Chen, et al., Frequency dynamic convolution for dense image prediction, in:

exec
/bin/zsh -lc 'pdftotext "3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf" - | rg -n -C 2 "Ablation|ablation|visual|Visualization|t-SNE|CMF|SSTI|3D|shape|occlusion|misalignment"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 181ms:
462-could project to the same 2D pixel; therefore, this projection
463:and color sampling procedure is applied to all 3D points,
464-regardless of their visibility status in the 2D view. This ensures
465-that the entire geometric representation is comprehensively
466-enriched with appearance information derived from the image,
467-not just the portions directly visible.
468:Then, we employ a pillar-based backbone network Θ3D
469-such as PointPillar [46] to extract pillar-based features from
470-L
471:3D point clouds sequences VP = {Pl }l=1
472-for feature alignment
473-
474-with image features. The process can be expressed as:
475:F P = Θ3D (VP ).
476-
477-(2)
--
479-Here, F P = {F P1 . . . F PL } is the pillar-based feature, where
480-F Pl ∈ RC×H×W . It is then fused with the 2D image feature F I in
481:our proposed Cross-modal Fusion (CMF) module, producing
482:the discriminative 3D-enhanced feature F̂ I . To preserve both
483:geometric and shape information, the fusion feature F is
484-derived by concatenating the pillar-based feature F P with the
485-result of a shortcut connection between the initial feature F I
486:and the 3D-enhanced feature F̂ I . Finally, the fusion feature is
487-fed into the Spatial-Temporal Aggregation (STA) module to
488-aggregate frame-level features temporally for generating the
489-video representation. The task flow is shown as follows:
490:F̂ I = CMF(F I , F P ),
491-F = Concat((F I + F̂ I ), F P ),
492-Fout = Ψ(F),
--
498-to the final video representation. The STA module is explained
499-in Section IV-B.
500:For the shape branch, we design the Shape-aware SpatialTemporal Interaction (SSTI) module, which learns the spatial
501:and temporal 3D shape dependencies of pedestrian and generates distinctive video-level 3D shape representation. In detail,
502-L
503:given the SMPL shape parameter β = {βl }l=1
504:, which is a parameter of the pedestrian mesh generated from the 3D human
505-reconstruction network, we first embed it into the feature space
506:using a shape embedding layer. Spatial and temporal shape
507-
508-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
509-
510:JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID
511-
512-12835
513-
514-TABLE I
515:L OOKUP TABLEFOR N OTATIONS IN THE 3D-A ID P EDESTRIAN R EPRESENTATION L EARNING (3DAPRL) N ETWORK
516-
517-features and its corresponding pillar-based features are fused
--
520-feature F P , we take the fI = F Il and fP = F Pl at l-th
521-frame as an example. We first initialize the reference point
522:r through the mapping relationship φ between 2D and 3D
523-representations obtained from the SMPL parametrization in
524:3D human reconstruction networks [64]:
525-r = φ(p),
526-
527:Fig. 3. The illustration of the fusion process in the CMF module.
528-
529:features are extracted and subsequently fused into spatialtemporal 3D shape features. We can express it as:
530:F̂βS T = SSTI(β),
531-
532-(4)
533-
534:where F̂βS T is the spatial-temporal 3D shape feature, and it
535-contributes to the final video representation. The lookup table
536-of parameters and corresponding notations is shown in Table I.
537-The detail of our proposed method is described below.
538-C. Cross-Modal Fusion Module
539:The 3D pedestrian data provide geometric information,
540:shape cues and accurate foreground position correspondence.
541:Previous Re-ID methods [40], [41], [42] utilizing 3D representation mainly focus on generating 3D features while
542-paying insufficient attention on effective fusion strategies for
543:fusing 2D and 3D features. Some methods [40], [42] map
544:RGB information to 3D representations or concatenate 2D and
545:3D features directly, neglecting the complementarity between
546-these modalities, resulting in coarse feature aggregation and
547-limited interaction. Furthermore, the inconsistency between
548:2D and 3D pedestrian features can lead the network to
549-focus on noisy or irrelevant features from both modalities,
550:ultimately degrading performance. To make full use of 3D data
551:for obtaining discriminative frame-level pedestrian representations, we propose the Cross-Modal Fusion (CMF) module. The
552-fusion process is shown in Figure 3, in which the 2D image
553-
554-(5)
555-
556:where p is the coordinate set of the points on 3D pedestrian
557-in l-th frame. In this manner, reference points are obtained
558-to indicate the pedestrian locations on fI . Then, the pillarbased feature fP is regarded as Query, and i indexes the Query
--
565-points on fV by employing sampling offsets ∆rmi j . Eventually,
566-the attention weights Ami j and sampled Value feature fˆVj are
567:aggregated to acquire the 3D-enhanced feature fˆIi . The fusion
568-process of module can be formulated as:
569-Ami j = linear( fPi ),
--
591-where M and J are the number of attention heads and sampling
592-points, respectively. Wm and Wm0 are learnable weights. To
593:process all Query features, we obtain the whole 3D-enhanced
594:feature map fˆI . In this way, our CMF module enables the
595-model to adaptively select where and what features should
596-be extracted from the image by the inquire of pillar-based
--
602-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025
603-
604:Fig. 4. The illustration of the interaction in the SSTI module.
605-
606-features. By focusing on the most relevant regions of the
--
610-and ranges of sampling points, allowing the model to capture
611-the local context of the pedestrian accurately while avoiding
612:irrelevant areas. Thus, 3D-enhanced image features focusing
613-on the most crucial areas of pedestrians are generated. Finally,
614-we gain the generated feature f (i.e., F l ) through a shortcut
615-connection and concatenation of the pillar-based feature fP
616:(i.e., F Pl ) with the 3D-enhanced feature fˆI (i.e., F̂ Il ) for
617:retaining geometric and shape information, as demonstrate in
618-Figure 2. We can express as:
619-f = Concat(( fI + fˆI ), fP ).
--
621-(8)
622-
623:To apply the whole CMF module to the image feature F I and
624:the pillar-based feature F P , we can acquire the 3D-enhanced
625:feature F̂ I and final feature F. Therefore, Our CMF module
626-reaches cross modality alignment with adaptive fusion. The
627:frame-level features are strengthened with the help of pillarbased features, and the multi-frame misalignment and partial
628:occlusion are mitigated via the CMF module.
629-D. Shape-Aware Spatial-Temporal Interaction Module
630:The 3D shape parameters control and describe the pedestrian shape in the 3D mesh [45]. Thus, generating distinctive
631:3D shape representations is critical for distinguishing different
632-pedestrians with similar appearance. Despite existing methods
633:[41], [42] utilize 3D shape parameters to create features in
634-video scenarios, they struggle to effectively obtain high-level
635-spatial-temporal dependencies. Inspired by spatial-temporal
636:learning strategies [73], we propose the Shape-aware SpatialTemporal Interaction (SSTI) module to extract distinctive
637:spatial-temporal 3D shape representations, thereby enhancing
638:the model’s robustness in capturing pedestrian shapes. As
639:expressed in Figure 2, we first project 3D shape parameters
640:into feature space to generate 3D shape features. In specific,
641:given video-level 3D shape parameters β ∈ RT ×10 of a
642-pedestrian, we embed them into a high-dimensional space via
643:the shape embedding layer, generating 3D shape features Fβ
644:with a shape of T × S × C, where T represents the temporal
645-dimension, S denotes the spatial dimension, and C denotes the
646:channel dimension. The shape embedding layer employs linear
647-layers followed by a activation function and a normalization
648-layer.
--
650-as illustrated in Figure 4. Specifically, we derive the Query
651-
652:Q, Key K and Value V from the 3D shape feature Fβ
653-through linear layers. Afterwards, they are mapped into two
654-streams (a spatial stream and a temporal stream) pairwisely to
655-apply feature contextualization along distinct axes. The spatial
656:stream captures intra-frame 3D shape, while the temporal
657:stream models inter-frame 3D shape variations. We process
658-the time axis and the space axis on two streams in parallel by
659-employing Multi-Head Self-Attention, enabling the extraction
--
685-(10)
686-
687:Two streams process the shape feature in parallel to obtain
688-contextual feature within their respective channels. The
689-acquired spatial feature FS and temporal feature and FT are
690-subsequently concatenated along the channel dimension to
691:form the spatial-temporal 3D shape feature Fβ . It can be
692-expressed as:
693-FβS T = Concat(FS , FT ).
694-(11)
695-The interaction process extracts the spatial and temporal
696:context of video-level shape parameter β. To further refine the
697-extracted features, motivated by [74], we normalize them and
698-pass them through a MLP layer consisting of linear layers and
699-a activation function. A shortcut connection is then applied,
700:producing the final spatial-temporal 3D shape feature F̂βS T ,
701-which can be expressed as:
702-F̂βS T = MLP(Norm(FβS T )) + FβS T ,
--
705-
706-where Norm denotes the normalization layer. In this manner,
707:our SSTI module effectively extracts and fuses spatial and
708-temporal dependencies in parallel. The receptive field captures
709-the interactions between spatial and temporal dimensions.
710:As a new pedestrian characteristic, the produced spatialtemporal 3D shape representation is both complementary and
711-distinguishable, aiding in target pedestrian identification in
712-video scenarios. Therefore, it helps alleviate the impacts of
713-appearance confusion and further mitigates issues including
714:the multi-frame misalignment and partial occlusion.
715-E. Loss Function
716-We apply the cross-entropy loss with label smoothing [75]
--
723-(13)
724-
725:JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID
726-
727-where Lcls is the classification loss and Ltri is the triplet loss.
728:We determine the weight of the Re-ID loss for the SSTI
729-module with a hyper-parameter λ. The overall loss Loverall for
730-our network can be expressed as:
731:CMF
732:SSTI
733-Loverall = LID
734-+ λLID
--
737-(14)
738-
739:CMF
740:SSTI
741-where LID
742-and LID
743:indicate the Re-ID loss for the CMF
744-
745:and the SSTI module, respectively.
746-IV. E XPERIMENTS
747-
--
756-final feature representation is obtained by concatenating the
757-video-level feature representation from the STA module Ψ
758:with the spatial-temporal 3D shape feature F̂βS T . The cosine
759-similarity is used as distance metric.
760-
761-A. Datasets and Evaluation Protocols
762:We evaluate our proposed 3DAPRL on four video-based
763-person Re-ID datasets including MARS [9], iLIDS-VID [10],
764-PRID-2011 [11] and DukeMTMC-VID [84], [85]. Since there
765-is no existing video-based Re-ID datasets with corresponding
766:3D representation data, we generate 3D pedestrian mesh for
767:these three datasets by using off-the-shell 3D human reconstruction networks [64], [65], [66].
768-1) MARS: is a large-scale video-based Re-ID dataset that
769-collected by six cameras, and it is composed of 17,503
--
793-as image backbone network. We sample 8 frames randomly
794-followed by RRS strategy [52], and each selected frame
795:is resized to 256 × 128. For the 3D branch, meshes are
796-transformed to point clouds and we use PointPillar [46]
797-without its detection head as backbone. We normalize the point
--
803-
804-C. Comparison With State-of-the-Art Methods
805:In this section, we compare the 3DAPRL with other methods [21], [24], [25], [27] [28], [29], [30], [31], [32], [35], [38],
806-[39], [49], [51], [54], [55], [56], [59], [77], [78], [79], [80],
807-[81], [82], [83] on three video-based Re-ID datasets including
--
809-since these two datasets have only one correct match in gallery.
810-The results are shown in Table II.
811:1) MARS: On MARS dataset, the proposed 3DAPRL
812-achieves the best performance compared with other SOTA
813:methods. Specifically, the 3DAPRL reaches 90.3% mAP and
814-93.1% Rank-1 accuracy, outperforming all the mentioned
815:SOTA methods. Compared with the second-best method TFCLIP [30], which utilize pre-trained CLIP visual encoder
816-to achieve one-stage CLIP-based video Re-ID learning, our
817:3DAPRL reach 0.9% increase on mAP and 0.1% increase on
818-Rank-1. Besides, SDCL [28] exploit event data to enhance
819:the model performance. Differently, we employ 3D meshes
820:that contains geometric information and shape cues, and yield
821-higher performance than SDCL [28], by 3.8% and 2.0% on
822-mAP and Rank-1 accuracy, respectively.
823:2) iLIDS-VID: On iLIDS-VID dataset, our 3DAPRL also
824-achieves the best performance compared to other SOTA methods. Although we observe a reduction of 0.9% and 0.6%
825-on Rank-5 accuracy when compare to the HASI [27] and
--
828-the best performance. In addition, our method surpasses the
829-methods leveraging other modality information on Rank-1
830:accuracy, e.g., TF-CLIP [30] that uses CLIP visual encoder
831-and SDCL [28] that uses event data, by 0.2% and 1.5%.
832:3) PRID-2011: On PRID-2011 dataset, our 3DAPRL and
833-MFA [79] reach the best Rank-1 accuracy of 96.6% and
834-gain 0.1% higher performance than the second-best method
835:SDCL [28]. It also illustrate that 3D pedestrian representation
836-contain richer pedestrian information, including geometric and
837:shape cues compare to event information which only provide
838-limited motion cues.
839-4) DukeMTMC-VID: On DukeMTMC-VID dataset,
840:our proposed 3DAPRL achieves the highest mAP and
841-Rank-1 accuracy among existing SOTA methods. Specifically,
842-it outperforms PSTA [21] and TMT [82] with improvements
843-of 0.2% in mAP and 0.3% in Rank-1 accuracy, respectively.
844-This performance gain is attributed to the incorporation of
845:3D pedestrian information, which enables our method to
846-
847-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
--
857-generate more discriminative video representations compared
858-to single-modal approaches.
859:Conclusively, it can be observed that the 3DAPRL achieves
860-competitive performance compare to other SOTA methods.
861-The performance improvement is contributed by the rich
862:information provided by the 3D pedestrian mesh, which is not
863-fully considered in previous video Re-ID studies. Particularly,
864:the fusion of 3D and image information in the CMF module
865:enhance the frame-level features to better capture the pedestrian information in image while preserving 3D information.
866:The SSTI module extracts the spatial-temporal information
867:of 3D shape, which contributes to the final video representation. Although our method has larger model parameter size
868-than typical CNN-based models [21], [38], [56], [78], [79],
869:[82], [83], the inclusion of 3D data drives superior results.
870:In addition, the 3DAPRL is significantly more parameterefficient than Transformer-based methods [27], [29], [30],
871-[59], [80] while still demonstrating improved results. Notably,
872-when compared to the other multi-modal technique [30], our
873-method exhibits advantages in both model parameter and
874-overall performance.
875:D. Ablation Study
876-To verify the contribution of each component in our
877:3DAPRL, we conduct experiments on MARS dataset. The
878-baseline model is built upon a ResNet-50 backbone [72] pretrained on ImageNet [88].
879-1) Effectiveness of Each Component: We evaluate the
880:contribution of each component of the proposed 3DAPRL,
881-by adding the components to baseline gradually. As shown
882-in Table III, compared with Index-1 and Index-2, Index2 obtains 0.7% and 0.9% increases on mAP and Rank-1
883-
884:accuracy, respectively. This indicates the benefits of aggregating frame-level feature temporally [21]. Based on the SSTI
885-module, Index-3 outperforms Index-2 by 2.4% mAP and 1.2%
886:Rank-1 accuracy. It demonstrates that as a new and discriminative pedestrian characteristic extracted from SSTI module,
887:the spatial-temporal 3D shape feature contributes to the
888-video representation, thus improving the model performance.
889:It also verifies the effectiveness of our SSTI module. For
890-Index-4, it far exceeds Index-2 by 4.5% and 2.7% on mAP and
891-Rank-1 accuracy, respectively, which shows the significance of
892:our CMF module. This illustrates that fusing image feature
893-with pillar-based feature enables the model to capture key
894:information in image, while assimilating geometric information and shape cues from the 3D pedestrian feature. Eventually,
895-our network reaches the SOTA performance by incorporating
896:the CMF module and the SSTI module as presented in
897-Index-5, which shows the superiority of the integration of all
898-components in our network.
--
903-Table III illustrates, model parameters and FLOPs predictably
904-increase with the integration of more modules. Notably, Our
905:SSTI module requires more parameters than the STA [21]
906-(14.24M vs. 6.9M), but fewer FLOPs (1.08G vs. 1.55G) while
907:delivering superior performance gains. Introducing 3D point
908:cloud features with the CMF module further increases model
909-parameters and FLOPs greatly, yet this is directly correlated
910-with significant performance improvements. Overall, while the
911:proposed 3DAPRL framework incurs an additional 51.31M
912-parameters and 4.8G FLOPs compared to the baseline, it
913-
914-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
915-
916:JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID
917-
918-12839
--
920-TABLE III
921-A BLATION R ESULTS OF THE C OMPONENTS OF O UR P ROPOSED M ETHOD ON MARS DATASET I NCLUDING THE S PATIAL -T EMPORAL AGGREGATION
922:(STA) M ODULE , THE C ROSS M ODAL F USION (CMF) M ODULE AND THE S HAPE -AWARE S PATIAL -T EMPORAL I NTERACTION (SSTI) M ODULE .
923-N UMBERS M ARKED W ITH B OLD I NDICATE THE B EST P ERFORMANCE
924-
--
927-TABLE V
928-
929:A BLATION R ESULTS OF D IFFERENT F USION S TRATEGIES FOR THE CMF
930-M ODULE ON MARS DATASET. N UMBERS M ARKED W ITH B OLD I NDI CATE THE B EST P ERFORMANCE
931-
932-A BLATION R ESULTS OF D IFFERENT ATTENTION M ECHANISMS FOR THE
933:SSTI M ODULE ON MARS DATASET. N UMBERS M ARKED W ITH B OLD
934-I NDICATE THE B EST P ERFORMANCE
935-
936-achieves highest accuracy, demonstrating a compelling tradeoff between computational cost and performance. We further
937:report the inference time of each component. While all components introduce additional computational overhead, the CMF
938-module contributes the most to the overall inference time
939:of 3DAPRL. This is primarily due to the introduction of an
940:additional modality (point cloud) within the CMF, as well as
941:the extra processing time required to load by the 3D backbone
942-during inference.
943-3) Comparative Analysis of Different Fusion Strategies in
944:the CMF Module: To demonstrate the effectiveness of our
945:proposed CMF module, we conduct several experiments with
946-different fusion strategies including the model without fusion
947-(same as Index-3 in Table III), direct concatenation, cross
948:attention and our CMF module. We also conduct experiments
949-to validate the benefits of concatenating pillar-based features.
950:As shown in Table IV, our proposed CMF module surpasses
951-direct concatenation by 2.1% mAP and 2.2% Rank-1 accuracy,
952-exceeds cross-attention mechanism with pillar-based feature
953-concatenation by 1.3% mAP and 1.5% Rank-1 accuracy.
954:Moreover, the CMF module brings the 1.4% mAP and 1.4%
955-Rank-1 accuracy gains compare to cross-attention mechanism
956-when neither concatenates pillar-based features. Additionally,
957-it outperforms the model without fusion by 2.5% mAP and
958:2.3% Rank-1 accuracy. This shows that while using crossattention mechanism can improve the performance, the CMF
959-module is more beneficial for fusion. It demonstrates that
960:our CMF module leverages the mapping relationship between
961:target pedestrian in 2D and 3D space as reference points,
962-enabling the network to accurately focus on key area of pedestrian image. Based on the fusion with pillar-based features,
963-frame-level distinctive pedestrian features are generated.
--
967-+0.6% mAP and +0.4% Rank-1, +0.3% mAP and +0.2%
968-Rank-1 and +0.2% mAP and +0.3% Rank-1 on three strategies (the model without fusion, the cross-attention and the
969:CMF module) without concatenating the pillar-based feature,
970-respectively. The reason is that extracted pillar-based features
971-contain appearance information, geometric information and
972:shape cues, and they are dimensionally aligned with image
973-feature.
974-4) Comparative Analysis of Different Attention Mechanisms
975:in the SSTI Module: In order to evaluate the effectiveness
976:of our SSTI module, we conduct experiments with different
977:variants for extracting the spatial-temporal 3D shape features,
978-as shown in Table V. Figure 5 illustrates the structure of each
979-variant for comparison in detail. Particularly, we use GRUs +
980:Self-Attention [42] to extract spatial-temporal 3D shape features, which are shown in Figure 5 (b). We also compare the
981-different strategies in which spatial and temporal information
982-are composed in Figure 5 (c) and (d). Different from the
983:SSTI module, additive attention fuses the spatial information
984-and temporal information by element-wise summation, and
985-iterative attention combines the spatial layer and the temporal
986-layer. We can observe that the performance drop out by 0.2%
987-on mAP and increase 0.1% on Rank-1 accuracy with only
988:spatial stream in the SSTI compare to the model without SSTI
989:module (same as Index-4 in Table III). Our SSTI module with
990:only temporal stream outperforms the model without SSTI
991-module by 0.3% mAP and 0.3% Rank-1 accuracy, but exhibits
992-a reduction of 0.3% mAP and 0.8% Rank-1 accuracy compare
993:to our SSTI module. These performance variations indicate the
994:significance of modeling video-level 3D shape parameters with
995-
996-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
--
1000-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025
1001-
1002:Fig. 5. Different variations for 3D shape modeling. (a) Our SSTI module. (b) Using GRUs and self-attention [42] to obtain shape representation. (c) Additive
1003-attention fuses the spatial and temporal information by element-wise summation. (d) Iterative attention combines the spatial layer and temporal layer.
1004-
1005:Fig. 6. Visualization of feature distribution of our network with and without
1006:SSTI module. We choose 15 identities with similar appearance from MARS
1007-dataset. Each cross and dot markers of different colors denote a different
1008-pedestrian from gallery and query. We use red dotted circle and blue box to
1009-show some examples of inter-class and intra-class distance, respectively.
1010-
1011:Fig. 7. Visualization of feature distribution of the baseline model and the
1012:3DAPRL. We choose 15 identities from each of iLIDS-VID and MARS
1013-dataset. Each cross and dot markers of different colors denote a different
1014-pedestrian from gallery and query. We use red dotted circle and blue box to
1015-show some examples of inter-class and intra-class distance, respectively.
1016-
1017:both spatial and temporal information. Moreover, our SSTI
1018-exceeds the Self-Attention + GRU and the model without
1019:SSTI module by 0.4% mAP and 0.8% Rank-1 accuracy, and
1020-0.6% mAP and 1.1% Rank-1 accuracy, respectively, which
1021-
1022:Fig. 8. Ablation results of (a) various length of video sequences and
1023-(b) hyperparameter λ on MARS dataset.
1024-
1025:Fig. 9. Comparison of top-10 retrieval results of baseline and the 3DAPRL on
1026-MARS dataset. Results with green boxes represent correctly matched frames,
1027-while those with red boxes represent incorrectly matched frames.
1028-
1029-demonstrates the effectiveness of feature contextualization in
1030:the SSTI. Meanwhile, we conduct comparison with additive
1031-attention and iterative attention, which illustrate that our fusion
1032-way achieve the best accuracy. The comparison results validate
1033:the effectiveness of our SSTI module, which extracts and
1034-fuses spatial and temporal dependencies in parallel, with its
1035-receptive field capturing interactions across both spatial and
--
1037-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
1038-
1039:JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID
1040-
1041-12841
1042-
1043:Fig. 10. Visualization of feature maps of the baseline and our method on iLIDS-VID, PRID-2011 and MARS datasets.
1044-
1045-temporal dimensions. The generated pedestrian characteristics
1046-make a significant contribution to the video representation.
1047:To present the effectiveness of the SSTI module on appearance confusion, we conduct experiments using our networks
1048:with and without the SSTI module on 15 identities with
1049-similar appearance, and the feature distribution results using
1050:t-SNE [92] are shown in Figure 6. We can observe in the
1051:blue boxes that our SSTI module can increase the inter-class
1052-distance. Besides, the intra-class distance is ease greatly which
1053:is shown in red dotted circle. The visual results shows that
1054:our SSTI module can mitigate the influence of appearance
1055-confusion.
1056-5) The Influence of the Hyperparamater λ: The hyperparmeter λ regulates the contribution of the Re-ID loss terms
--
1071-stages of STAM [21] is increased, but only slightly as frame
1072-grows from 4 to 8.
1073:E. Visualization
1074:1) Visualization of Feature Distribution: To validate the
1075:effectiveness of our approach, we employ t-SNE [92] to visualize the feature distributions extracted by the 3DAPRL and the
1076-baseline on MARS and iLIDS-VID datasets, highlighting their
1077-clear distinction. As can be observed in Figure 7, our method
1078-
1079-effectively reduces intra-class distances while expanding interclass distances among different pedestrians, employing the
1080:proposed CMF and SSTI modules.
1081:2) Visualization of Retrieval Results: We compare the
1082-retrieval results of samples with similar appearance between
1083:the baseline method (top two rows) and the 3DAPRL (bottom
1084-two rows) in Figure 9. The baseline method struggles to
1085-accurately distinguish the target pedestrian, often leading to
1086-erroneous predictions. Specifically, it tends to confuse pedestrians with similar appearances, resulting in mis-identifications
1087:due to its limited capability in capturing discriminative features. In contrast, the 3DAPRL achieves consistent accuracy
1088:from Rank-1 to Rank-10 by leveraging the CMF module to
1089:generate distinctive pedestrian features and the SSTI module
1090:to capture spatial-temporal 3D shape cues. These results
1091:demonstrate the effectiveness of the 3DAPRL in alleviating
1092-the appearance confusion problem.
1093:3) Visualization of Feature Maps: We visualize the feature
1094:maps of the baseline and our 3DAPRL on iLIDS-VID, PRID2011 and MARS datasets in Figure 10. We present both
1095-the pedestrian images and point clouds. It can be observed
1096:that the 3DAPRL focuses more on pedestrian features than
1097-the baseline. The discriminative areas of the pedestrian are
1098-highlighted more prominently in our method compared to the
1099:baseline. This demonstrates that by effectively fusing pillarbased features in our the CMF module, the network better
1100-attends to the target pedestrian, leading to the generation of
1101-discriminative features.
1102:To further investigate the performance of the 3DAPRL on
1103:video Re-ID challenges, such as multi-frame misalignment and
1104:partial occlusion, we visualize feature maps of our method
1105-and the baseline on challenging samples from the MARS
1106-dataset, where the target pedestrian is occluded or frames are
1107-misaligned. As shown in Figure 11, compared to the baseline,
1108:the 3DAPRL generates distinctive features, and it focuses more
1109-precisely on the target pedestrian region. The results suggest
1110:that our method can mitigate the misalignment and partially
1111-
1112-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
--
1118-F. Limitations
1119-We observe that the proposed method exhibits reduced
1120:3D reconstruction accuracy in challenging environments such
1121-as severe crowded scenarios. This is particularly evident when
1122-multiple individuals are present and the target lacks prominence or heavily occluded by non-target pedestrians, resulting
--
1124-our framework’s supervision based on accurate identity labels
1125-enables the model to learn to suppress unreliable features
1126:caused by non-target overlap or incorrect 3D geometry, and
1127-the STA module in our framework helps alleviate this issue
1128-through pyramid structure and temporal aggregation [21], such
--
1130-this, we plan to incorporate temporal consistency in the future
1131-work to improve target-specific reconstruction.
1132:Fig. 11. Visualization of hard samples’ feature maps from the baseline and
1133-our method on the MARS dataset. The red box indicate the noisy frame that
1134:have misalignment and partially occlusion problems.
1135-
1136:Fig. 12. Visualization of activation maps with Grad-CAM [91] for challenging
1137-occluded samples from the baseline and our method on the MARS dataset. The
1138-red boxe highlight regions corresponding to non-target objects or pedestrians,
1139-while the green box denote the target pedestrian regions.
1140-
1141:occlusion problems effectively by leveraging the fusion of
1142-pillar-based features.
1143:4) Visualization of Activation Maps: To demonstrate the
1144:ability of our method against challenging occlusion scenarios
1145:such as overlapping scenarios, we visualize activation maps
1146-with Grad-CAM [91] of hard occluded samples. As shown
1147:in Figure 12, each sample involves occlusions caused by
1148-non-target pedestrians or background objects, leaving only
1149-partial visibility of the target pedestrian. The baseline model
1150-fails to accurately localize the target, often activating on
1151-occluded or irrelevant regions. In contrast, benefiting from
1152:the fusion strategy in our CMF module, which incorporates
1153:3D representations, our method effectively suppresses noisy
1154-activations and consistently focuses on the target regions. This
1155:demonstrates the CMF module’s ability to enhance feature
1156:discriminability and robustness under occlusion.
1157-
1158-V. C ONCLUSION
1159:In this paper, we explore the 3D modality for video-based
1160:Re-ID and propose a novel 3D-Aid Pedestrian Representation Learning (3DAPRL) network that effectively leverages
1161:3D information. The 3DAPRL employs a pillar-based backbone network to bridge the modality gap and resolve feature
1162-inconsistency, enabling the extraction of pedestrian appearance
1163:features, 3D geometric information, and shape cues. Moreover,
1164:a Cross-modal Fusion (CMF) module is designed to adaptively
1165-fuse the extracted pillar-based features with image features,
1166-facilitating the creation of distinctive fused representations.
1167:Furthermore, spatial-temporal 3D shape representations are
1168-produced by the Shape-aware Spatial-Temporal Interaction
1169:(SSTI) module. Spatial and temporal dependencies of 3D
1170:shape parameters are captured and integrated. Eventually, the
1171:challenges of multi-frame misalignment, partial occlusion,
1172-and appearance confusion in video data are mitigated by
1173:leveraging the CMF and SSTI modules. Extensive experiments conducted on three widely used video Re-ID datasets
1174-demonstrate the effectiveness of our network, showing that
1175-our proposed network achieves superior results compared to
1176-recent state-of-the-art methods. In the future, we will enhance
1177-our framework according to limitations and further explore its
1178:adaptability to sparse 3D inputs such as depth maps. We also
1179:plan to conduct benchmarking against recent 3D-aware Re-ID
1180-methods including PointNet-based architectures.
1181-R EFERENCES
--
1207-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
1208-
1209:JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID
1210-
1211-[7]
--
1234-motion context,” IEEE Trans. Circuits Syst. Video Technol., vol. 28,
1235-no. 10, pp. 2788–2802, Oct. 2018.
1236:[16] X. Gu, H. Chang, B. Ma, H. Zhang, and X. Chen, “Appearancepreserving 3D convolution for video-based person re-identification,” in
1237-Proc. Eur. Conf. Comput. Vis., 2020, pp. 228–243.
1238-[17] Z. Qiu, T. Yao, and T. Mei, “Learning spatio-temporal representation
1239:with pseudo-3D residual networks,” in Proc. IEEE Int. Conf. Comput.
1240-Vis., Jun. 2017, pp. 5533–5541.
1241-[18] L. Wu, Y. Wang, L. Shao, and M. Wang, “3-D PersonVLAD: Learning
--
1307-(CVPR), Jun. 2020, pp. 10407–10416.
1308-[40] Z. Zheng, X. Wang, N. Zheng, and Y. Yang, “Parameter-efficient person
1309:re-identification in the 3D space,” IEEE Trans. Neural Netw. Learn.
1310-Syst., vol. 35, no. 6, pp. 7534–7534, Jun. 2022.
1311:[41] K. Han, Y. Huang, S. Gong, L. Wang, and T. Tan, “3D shape temporal
1312-aggregation for video-based clothing-change person re-identification,”
1313-in Proc. Asian Conf. Comput. Vis., 2022, pp. 2371–2387.
1314:[42] V. D. Nguyen, P. Mantini, and S. K. Shah, “Temporal 3D shape
1315-modeling for video-based cloth-changing person re-identification,” in
1316-Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. Workshops (WACVW),
1317-Jan. 2024, pp. 173–182.
1318:[43] J. Chen et al., “Learning 3D shape feature for texture-insensitive
1319-person re-identification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
1320-Recognit., Jun. 2021, pp. 8146–8155.
1321-[44] Z. Yu, L. Li, J. Xie, C. Wang, W. Li, and X. Ning, “Pedestrian
1322:3D shape understanding for person re-identification via multi-view
1323-learning,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 7,
1324-pp. 5589–5602, Jul. 2024.
--
1333-DETR: Deformable transformers for end-to-end object detection,” 2020,
1334-arXiv:2010.04159.
1335:[48] X. Liao, L. He, Z. Yang, and C. Zhang, “Video-based person reidentification via 3D convolutional networks and non-local attention,”
1336-in Proc. ACCV, Jan. 2018, pp. 620–634.
1337:[49] J. Li, S. Zhang, and T. Huang, “Multi-scale 3D convolution network for
1338-video based person re-identification,” in Proc. AAAI Conf. Artif. Intell.,
1339-vol. 33, 2019, pp. 8618–8625.
--
1380-Oct. 2024.
1381-[61] Y. Wang, X. Liang, and S. Liao, “Cloning outfits from real-world images
1382:to 3D characters for generalizable person re-identification,” in Proc.
1383-IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022,
1384-pp. 4890–4899.
1385-[62] F. Liu, M. Kim, Z. Gu, A. Jain, and X. Liu, “Learning clothing
1386:and pose invariant 3D shape representation for long-term person reidentification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct.
1387-2023, pp. 19617–19626.
1388-[63] W. Guo et al., “LiDAR-based person re-identification,” in Proc.
1389-IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024,
1390-pp. 17437–17447.
1391:[64] A. Kanazawa, M. J. Black, D. W. Jacobs, and J. Malik, “End-toend recovery of human shape and pose,” in Proc. IEEE/CVF Conf.
1392-Comput. Vis. Pattern Recognit., Salt Lake City, UT, USA, Jun. 2018,
1393-pp. 7122–7131.
1394:[65] Y. Rong, T. Shiratori, and H. Joo, “FrankMocap: A monocular 3D
1395-whole-body pose estimation system via regression and integration,” in
1396-Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW), Oct. 2021,
1397-pp. 1749–1759.
1398-[66] E. Corona, G. Pons-Moll, G. Alenya, and F. Moreno-Noguer, “Learned
1399:vertex descent: A new direction for 3D human model fitting,” in Proc.
1400-Eur. Conf. Comput. Vis., Jan. 2022, pp. 146–165.
1401-[67] S. Saito, Z. Huang, R. Natsume, S. Morishima, H. Li, and A. Kanazawa,
--
1404-2019, pp. 2304–2314.
1405-[68] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “PointNet: Deep
1406:learning on point sets for 3D classification and segmentation,” in
1407-Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017,
1408-pp. 652–660.
--
1411-Inf. Process. Syst., Jan. 2017, pp. 1–12.
1412-[70] Y. Zhou and O. Tuzel, “VoxelNet: End-to-end learning for point cloud
1413:based 3D object detection,” in Proc. IEEE/CVF Conf. Comput. Vis.
1414-Pattern Recognit., Jun. 2018, pp. 4490–4499.
1415:[71] X. Li et al., “LoGoNet: Towards accurate 3D object detection with localto-global cross-modal fusion,” in Proc. IEEE/CVF Conf. Comput. Vis.
1416-Pattern Recognit. (CVPR), Jun. 2023, pp. 17524–17534.
1417-[72] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
1418-image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.
1419-(CVPR), Jun. 2016, pp. 770–778.
1420:[73] Z. Tang, Z. Qiu, Y. Hao, R. Hong, and T. Yao, “3D human
1421-pose estimation with spatio-temporal criss-cross attention,” in Proc.
1422-IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023,
--
1478-gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis.
1479-(ICCV), Oct. 2017, pp. 618–626.
1480:[92] L. V. D. Maaten and G. E. Hinton, “Visualizing data using t-SNE,”
1481-J. Mach. Learn. Res., vol. 9, no. 86, pp. 2579–2605, Jan. 2008.
1482-
--
1494-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
1495-
1496:JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID
1497-
1498-Peng Gao (Graduate Student Member, IEEE)
--
1532-I NTELLIGENCE, IEEE T RANSACTIONS ON I MAGE P ROCESSING, CVPR,
1533-ACM MM, and ECCV. Her research interests include camera calibration,
1534:motion recovery, and 3D reconstruction.
1535-
1536-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext \"3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf\" - | sed -n '880,1058p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 133ms:
contribution of each component of the proposed 3DAPRL,
by adding the components to baseline gradually. As shown
in Table III, compared with Index-1 and Index-2, Index2 obtains 0.7% and 0.9% increases on mAP and Rank-1

accuracy, respectively. This indicates the benefits of aggregating frame-level feature temporally [21]. Based on the SSTI
module, Index-3 outperforms Index-2 by 2.4% mAP and 1.2%
Rank-1 accuracy. It demonstrates that as a new and discriminative pedestrian characteristic extracted from SSTI module,
the spatial-temporal 3D shape feature contributes to the
video representation, thus improving the model performance.
It also verifies the effectiveness of our SSTI module. For
Index-4, it far exceeds Index-2 by 4.5% and 2.7% on mAP and
Rank-1 accuracy, respectively, which shows the significance of
our CMF module. This illustrates that fusing image feature
with pillar-based feature enables the model to capture key
information in image, while assimilating geometric information and shape cues from the 3D pedestrian feature. Eventually,
our network reaches the SOTA performance by incorporating
the CMF module and the SSTI module as presented in
Index-5, which shows the superiority of the integration of all
components in our network.
2) Computation Complexity Analysis: We analyse model
efficiency using parameter count (space complexity) and
FLOPs (time complexity), common in video Re-ID. Note
that we report the FLOPs for a four-frames sequence. As
Table III illustrates, model parameters and FLOPs predictably
increase with the integration of more modules. Notably, Our
SSTI module requires more parameters than the STA [21]
(14.24M vs. 6.9M), but fewer FLOPs (1.08G vs. 1.55G) while
delivering superior performance gains. Introducing 3D point
cloud features with the CMF module further increases model
parameters and FLOPs greatly, yet this is directly correlated
with significant performance improvements. Overall, while the
proposed 3DAPRL framework incurs an additional 51.31M
parameters and 4.8G FLOPs compared to the baseline, it

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID

12839

TABLE III
A BLATION R ESULTS OF THE C OMPONENTS OF O UR P ROPOSED M ETHOD ON MARS DATASET I NCLUDING THE S PATIAL -T EMPORAL AGGREGATION
(STA) M ODULE , THE C ROSS M ODAL F USION (CMF) M ODULE AND THE S HAPE -AWARE S PATIAL -T EMPORAL I NTERACTION (SSTI) M ODULE .
N UMBERS M ARKED W ITH B OLD I NDICATE THE B EST P ERFORMANCE

TABLE IV

TABLE V

A BLATION R ESULTS OF D IFFERENT F USION S TRATEGIES FOR THE CMF
M ODULE ON MARS DATASET. N UMBERS M ARKED W ITH B OLD I NDI CATE THE B EST P ERFORMANCE

A BLATION R ESULTS OF D IFFERENT ATTENTION M ECHANISMS FOR THE
SSTI M ODULE ON MARS DATASET. N UMBERS M ARKED W ITH B OLD
I NDICATE THE B EST P ERFORMANCE

achieves highest accuracy, demonstrating a compelling tradeoff between computational cost and performance. We further
report the inference time of each component. While all components introduce additional computational overhead, the CMF
module contributes the most to the overall inference time
of 3DAPRL. This is primarily due to the introduction of an
additional modality (point cloud) within the CMF, as well as
the extra processing time required to load by the 3D backbone
during inference.
3) Comparative Analysis of Different Fusion Strategies in
the CMF Module: To demonstrate the effectiveness of our
proposed CMF module, we conduct several experiments with
different fusion strategies including the model without fusion
(same as Index-3 in Table III), direct concatenation, cross
attention and our CMF module. We also conduct experiments
to validate the benefits of concatenating pillar-based features.
As shown in Table IV, our proposed CMF module surpasses
direct concatenation by 2.1% mAP and 2.2% Rank-1 accuracy,
exceeds cross-attention mechanism with pillar-based feature
concatenation by 1.3% mAP and 1.5% Rank-1 accuracy.
Moreover, the CMF module brings the 1.4% mAP and 1.4%
Rank-1 accuracy gains compare to cross-attention mechanism
when neither concatenates pillar-based features. Additionally,
it outperforms the model without fusion by 2.5% mAP and
2.3% Rank-1 accuracy. This shows that while using crossattention mechanism can improve the performance, the CMF
module is more beneficial for fusion. It demonstrates that
our CMF module leverages the mapping relationship between
target pedestrian in 2D and 3D space as reference points,
enabling the network to accurately focus on key area of pedestrian image. Based on the fusion with pillar-based features,
frame-level distinctive pedestrian features are generated.

In addition, it is obvious that concatenating pillar-based
feature directly to fusion features is useful, which provides
+0.6% mAP and +0.4% Rank-1, +0.3% mAP and +0.2%
Rank-1 and +0.2% mAP and +0.3% Rank-1 on three strategies (the model without fusion, the cross-attention and the
CMF module) without concatenating the pillar-based feature,
respectively. The reason is that extracted pillar-based features
contain appearance information, geometric information and
shape cues, and they are dimensionally aligned with image
feature.
4) Comparative Analysis of Different Attention Mechanisms
in the SSTI Module: In order to evaluate the effectiveness
of our SSTI module, we conduct experiments with different
variants for extracting the spatial-temporal 3D shape features,
as shown in Table V. Figure 5 illustrates the structure of each
variant for comparison in detail. Particularly, we use GRUs +
Self-Attention [42] to extract spatial-temporal 3D shape features, which are shown in Figure 5 (b). We also compare the
different strategies in which spatial and temporal information
are composed in Figure 5 (c) and (d). Different from the
SSTI module, additive attention fuses the spatial information
and temporal information by element-wise summation, and
iterative attention combines the spatial layer and the temporal
layer. We can observe that the performance drop out by 0.2%
on mAP and increase 0.1% on Rank-1 accuracy with only
spatial stream in the SSTI compare to the model without SSTI
module (same as Index-4 in Table III). Our SSTI module with
only temporal stream outperforms the model without SSTI
module by 0.3% mAP and 0.3% Rank-1 accuracy, but exhibits
a reduction of 0.3% mAP and 0.8% Rank-1 accuracy compare
to our SSTI module. These performance variations indicate the
significance of modeling video-level 3D shape parameters with

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

12840

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025

Fig. 5. Different variations for 3D shape modeling. (a) Our SSTI module. (b) Using GRUs and self-attention [42] to obtain shape representation. (c) Additive
attention fuses the spatial and temporal information by element-wise summation. (d) Iterative attention combines the spatial layer and temporal layer.

Fig. 6. Visualization of feature distribution of our network with and without
SSTI module. We choose 15 identities with similar appearance from MARS
dataset. Each cross and dot markers of different colors denote a different
pedestrian from gallery and query. We use red dotted circle and blue box to
show some examples of inter-class and intra-class distance, respectively.

Fig. 7. Visualization of feature distribution of the baseline model and the
3DAPRL. We choose 15 identities from each of iLIDS-VID and MARS
dataset. Each cross and dot markers of different colors denote a different
pedestrian from gallery and query. We use red dotted circle and blue box to
show some examples of inter-class and intra-class distance, respectively.

both spatial and temporal information. Moreover, our SSTI
exceeds the Self-Attention + GRU and the model without
SSTI module by 0.4% mAP and 0.8% Rank-1 accuracy, and
0.6% mAP and 1.1% Rank-1 accuracy, respectively, which

Fig. 8. Ablation results of (a) various length of video sequences and
(b) hyperparameter λ on MARS dataset.

Fig. 9. Comparison of top-10 retrieval results of baseline and the 3DAPRL on
MARS dataset. Results with green boxes represent correctly matched frames,
while those with red boxes represent incorrectly matched frames.

demonstrates the effectiveness of feature contextualization in
the SSTI. Meanwhile, we conduct comparison with additive
attention and iterative attention, which illustrate that our fusion
way achieve the best accuracy. The comparison results validate
the effectiveness of our SSTI module, which extracts and
fuses spatial and temporal dependencies in parallel, with its
receptive field capturing interactions across both spatial and

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID

12841

Fig. 10. Visualization of feature maps of the baseline and our method on iLIDS-VID, PRID-2011 and MARS datasets.

temporal dimensions. The generated pedestrian characteristics
make a significant contribution to the video representation.
To present the effectiveness of the SSTI module on appearance confusion, we conduct experiments using our networks
with and without the SSTI module on 15 identities with
similar appearance, and the feature distribution results using
t-SNE [92] are shown in Figure 6. We can observe in the
blue boxes that our SSTI module can increase the inter-class
distance. Besides, the intra-class distance is ease greatly which
is shown in red dotted circle. The visual results shows that
our SSTI module can mitigate the influence of appearance
confusion.
5) The Influence of the Hyperparamater λ: The hyperparmeter λ regulates the contribution of the Re-ID loss terms
in Eq. 14. Figure 6 (a) illustrates the results of the model
performance varies with the value of λ. When λ is boosted

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -layout \"3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf\" - | sed -n '900,1005p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 134ms:
[28] C. Cao et al., “Event-guided person re-identification via sparse-dense        [51] Y. Liu, Z. Yuan, W. Zhou, and H. Li, “Spatial and temporal mutual
     complementary learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern              promotion for video-based person re-identification,” in Proc. AAAI Conf.
     Recognit. (CVPR), Jun. 2023, pp. 17990–17999.                                      Artif. Intell., 2019, vol. 33, no. 1, pp. 8786–8793.
[29] P. Wu, L. Wang, S. Zhou, G. Hua, and C. Sun, “Temporal correlation            [52] C.-T. Liu, C.-W. Wu, Y.-C. Frank Wang, and S.-Y. Chien, “Spatially and
     vision transformer for video person re-identification,” in Proc. AAAI              temporally efficient non-local attention network for video-based person
     Conf. Artif. Intell., vol. 38, 2024, pp. 6083–6091.                                re-identification,” 2019, arXiv:1908.01683.

          Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
12844                                       IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025



[53] C. Eom, G. Lee, J. Lee, and B. Ham, “Video-based person re-                  [75] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking
     identification with spatial and temporal memory networks,” in Proc.               the inception architecture for computer vision,” in Proc. IEEE Conf.
     IEEE Int. Conf. Comput. Vis., Jul. 2021, pp. 12036–12045.                         Comput. Vis. Pattern Recognit. (CVPR), Jun. 2016, pp. 2818–2826.
[54] J. Liu, Z.-J. Zha, W. Wu, K. Zheng, and Q. Sun, “Spatial–temporal            [76] A. Hermans, L. Beyer, and B. Leibe, “In defense of the triplet loss for
     correlation and topology learning for person re-identification in videos,”        person re-identification,” 2017, arXiv:1703.07737.
     in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.          [77] T. He, X. Jin, X. Shen, J. Huang, Z. Chen, and X.-S. Hua, “Dense
     2021, pp. 4368–4377.                                                              interaction learning for video-based person re-identification,” in Proc.
[55] Z. Tang, R. Zhang, Z. Peng, J. Chen, and L. Lin, “Multi-stage spatio-             IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 1490–1501.
     temporal aggregation transformer for video person re-identification,”        [78] A. Aich, M. Zheng, S. Karanam, T. Chen, A. K. Roy-Chowdhury, and
     IEEE Trans. Multimedia, vol. 25, pp. 7917–7929, 2023.                             Z. Wu, “Spatio-temporal representation factorization for video-based
[56] L. Liu, X. Yang, N. Wang, and X. Gao, “Frequency information                      person re-identification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis.
     disentanglement network for video-based person re-identification,” IEEE           (ICCV), Oct. 2021, pp. 152–162.
     Trans. Image Process., vol. 32, pp. 4287–4298, 2023.                         [79] X. Gu, H. Chang, B. Ma, and S. Shan, “Motion feature aggregation
[57] J. Li, S. Zhang, J. Wang, W. Gao, and Q. Tian, “Global-local temporal             for video-based person re-identification,” IEEE Trans. Image Process.,
     representations for video person re-identification,” in Proc. IEEE/CVF            vol. 31, pp. 3908–3919, 2022.
     Int. Conf. Comput. Vis. (ICCV), Oct. 2019, pp. 3958–3967.                    [80] X. Zang, G. Li, and W. Gao, “Multidirection and multiscale pyramid
                                                                                       in transformer for video-based pedestrian retrieval,” IEEE Trans. Ind.
[58] D. Wu, M. Ye, G. Lin, X. Gao, and J. Shen, “Person re-identification
                                                                                       Informat., vol. 18, no. 12, pp. 8776–8785, Dec. 2022.
     by context-aware part attention and multi-head collaborative learning,”
                                                                                  [81] H. Pan et al., “Pose-aided video-based person re-identification via
     IEEE Trans. Inf. Forensics Security, vol. 17, pp. 115–126, 2022.
                                                                                       recurrent graph convolutional network,” IEEE Trans. Circuits Syst. Video
[59] J. Wu et al., “CAViT: Contextual alignment vision transformer for video
                                                                                       Technol., vol. 33, no. 12, pp. 7183–7196, Dec. 2023.
     object re-identification,” in Proc. Eur. Conf. Comput. Vis., Jan. 2022,
                                                                                  [82] X. Liu, P. Zhang, C. Yu, X. Qian, X. Yang, and H. Lu, “A video
     pp. 549–566.
                                                                                       is worth three views: Trigeminal transformers for video-based person
[60] X. Liu, C. Yu, P. Zhang, and H. Lu, “Deeply coupled                               re-identification,” IEEE Trans. Intell. Transp. Syst., vol. 25, no. 9,
     convolution–transformer with Spatial–Temporal complementary                       pp. 12818–12828, Sep. 2024.
     learning for video-based person re-identification,” IEEE Trans.              [83] Z. Ran, X. Wei, W. Liu, and X. Lu, “Multi-scale aligned spatial–temporal
     Neural Netw. Learn. Syst., vol. 35, no. 10, pp. 13753–13763,                      interaction for video-based person re-identification,” IEEE Trans. Cir-
     Oct. 2024.                                                                        cuits Syst. Video Technol., vol. 34, no. 9, pp. 8536–8546, Sep. 2024.
[61] Y. Wang, X. Liang, and S. Liao, “Cloning outfits from real-world images      [84] Y. Wu, Y. Lin, X. Dong, Y. Yan, W. Ouyang, and Y. Yang, “Exploit
     to 3D characters for generalizable person re-identification,” in Proc.            the unknown gradually: One-shot video-based person re-identification
     IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022,                  by stepwise learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
     pp. 4890–4899.                                                                    Recognit., Jun. 2018, pp. 5177–5186.
[62] F. Liu, M. Kim, Z. Gu, A. Jain, and X. Liu, “Learning clothing               [85] E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi, “Performance
     and pose invariant 3D shape representation for long-term person re-               measures and a data set for multi-target, multi-camera tracking,” in Proc.
     identification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct.           Eur. Conf. Comput. Vis. Cham, Switzerland: Springer, 2016, pp. 17–35.
     2023, pp. 19617–19626.                                                       [86] P. Felzenszwalb, D. McAllester, and D. Ramanan, “A discriminatively
[63] W. Guo et al., “LiDAR-based person re-identification,” in Proc.                   trained, multiscale, deformable part model,” in Proc. IEEE Conf. Com-
     IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024,                  put. Vis. Pattern Recognit. (CVPR), Jun. 2008, pp. 1–8.
     pp. 17437–17447.                                                             [87] A. Dehghan, S. M. Assari, and M. Shah, “GMMCP tracker: Globally
[64] A. Kanazawa, M. J. Black, D. W. Jacobs, and J. Malik, “End-to-                    optimal generalized maximum multi clique problem for multiple object
     end recovery of human shape and pose,” in Proc. IEEE/CVF Conf.                    tracking,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Jun.
     Comput. Vis. Pattern Recognit., Salt Lake City, UT, USA, Jun. 2018,               2015, pp. 4091–4099.
     pp. 7122–7131.                                                               [88] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet:
[65] Y. Rong, T. Shiratori, and H. Joo, “FrankMocap: A monocular 3D                    A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput.
     whole-body pose estimation system via regression and integration,” in             Vis. Pattern Recognit., Miami, FL, USA, Jun. 2009, pp. 248–255.
     Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW), Oct. 2021,         [89] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
     pp. 1749–1759.                                                                    2014, arXiv:1412.6980.
[66] E. Corona, G. Pons-Moll, G. Alenya, and F. Moreno-Noguer, “Learned           [90] H. Luo, Y. Gu, X. Liao, S. Lai, and W. Jiang, “Bag of tricks and a
     vertex descent: A new direction for 3D human model fitting,” in Proc.             strong baseline for deep person re-identification,” in Proc. IEEE/CVF
     Eur. Conf. Comput. Vis., Jan. 2022, pp. 146–165.                                  Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), Jun. 2019,
[67] S. Saito, Z. Huang, R. Natsume, S. Morishima, H. Li, and A. Kanazawa,             pp. 1487–1495.
     “PIFu: Pixel-aligned implicit function for high-resolution clothed human     [91] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and
     digitization,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct.             D. Batra, “Grad-CAM: Visual explanations from deep networks via
     2019, pp. 2304–2314.                                                              gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis.
[68] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “PointNet: Deep                         (ICCV), Oct. 2017, pp. 618–626.
     learning on point sets for 3D classification and segmentation,” in           [92] L. V. D. Maaten and G. E. Hinton, “Visualizing data using t-SNE,”
     Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017,                J. Mach. Learn. Res., vol. 9, no. 86, pp. 2579–2605, Jan. 2008.
     pp. 652–660.
[69] C. R. Qi, Y. Li, H. Su, and L. Guibas, “PointNet+$: Deep hierarchical
     feature learning on point sets in a metric space,” in Proc. Adv. Neural
     Inf. Process. Syst., Jan. 2017, pp. 1–12.
[70] Y. Zhou and O. Tuzel, “VoxelNet: End-to-end learning for point cloud
     based 3D object detection,” in Proc. IEEE/CVF Conf. Comput. Vis.
     Pattern Recognit., Jun. 2018, pp. 4490–4499.
[71] X. Li et al., “LoGoNet: Towards accurate 3D object detection with local-
     to-global cross-modal fusion,” in Proc. IEEE/CVF Conf. Comput. Vis.
     Pattern Recognit. (CVPR), Jun. 2023, pp. 17524–17534.                                                 Guquan Jing (Graduate Student Member, IEEE)
[72] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for                                      received the B.Eng. degree from Zhejiang Normal
     image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.                                University (ZJNU) and the B.S. degree (Hons.) from
     (CVPR), Jun. 2016, pp. 770–778.                                                                       the University of Worcester in 2020 and the M.S.
[73] Z. Tang, Z. Qiu, Y. Hao, R. Hong, and T. Yao, “3D human                                               degree from the University of Sheffield in 2021.
     pose estimation with spatio-temporal criss-cross attention,” in Proc.                                 He is currently pursuing the Ph.D. degree with
     IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023,                                      the Department of Computer Science, Hong Kong
     pp. 4790–4799.                                                                                        Baptist University (HKBU), Hong Kong, China. His
[74] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf.                             research interests include deep learning, computer
     Process. Syst., vol. 30, Jun. 2017, pp. 5998–6008.                                                    vision, and person re-identification.

          Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID                                                                 12845



                       Peng Gao (Graduate Student Member, IEEE)                                        Yiyang Hu received the B.Eng. degree from
                       received the B.S. degree (Hons.) from Queen Mary                                Chongqing University of Technology (CQUT) in
                       University of London (QMUL), U.K., in 2020, and                                 2021. He is currently pursuing the Ph.D. degree with
                       the M.A. degree from the University of Bristol                                  the Department of Computer Science, Hong Kong
                       (UoB), U.K., in 2022. He is currently pursuing the                              Baptist University (HKBU), Hong Kong, China. His
                       Ph.D. degree with the Department of Computer Sci-                               research interests include deep learning, relighting,

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -layout -f 7 -l 9 \"3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf\" - | sed -n '1,240p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
12836                                      IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025



                                                                               Q, Key K and Value V from the 3D shape feature Fβ
                                                                               through linear layers. Afterwards, they are mapped into two
                                                                               streams (a spatial stream and a temporal stream) pairwisely to
                                                                               apply feature contextualization along distinct axes. The spatial
                                                                               stream captures intra-frame 3D shape, while the temporal
                                                                               stream models inter-frame 3D shape variations. We process
                                                                               the time axis and the space axis on two streams in parallel by
                                                                               employing Multi-Head Self-Attention, enabling the extraction
                                                                               of both spatial and temporal contexts. In detail, we define QS ,
Fig. 4. The illustration of the interaction in the SSTI module.
                                                                               KS and VS for the spatial stream and QT , KT and VT for the
                                                                               temporal stream. The attention map AS for the spatial stream
features. By focusing on the most relevant regions of the                      and AT for the temporal stream are generated as follows:
image features via reference points, the model ensures the
                                                                                                                 QS · KST
                                                                                                                         
effective utilization of crucial information. Through learnable                                  AS = softmax      √        ,
offsets, our fusion strategy dynamically adjusts the positions                                                       C
                                                                                                                 QT · KTT
                                                                                                                         
and ranges of sampling points, allowing the model to capture
                                                                                                 AT = softmax      √        .              (9)
the local context of the pedestrian accurately while avoiding                                                        C
irrelevant areas. Thus, 3D-enhanced image features focusing
                                                                               Then, VS and VT are aggregated to feature maps AS and AT ,
on the most crucial areas of pedestrians are generated. Finally,
                                                                               respectively, to obtain contextual feature FS and FT on each
we gain the generated feature f (i.e., F l ) through a shortcut
                                                                               stream:
connection and concatenation of the pillar-based feature fP
(i.e., F Pl ) with the 3D-enhanced feature fˆI (i.e., F̂ Il ) for                                           FS = AS · VS ,
retaining geometric and shape information, as demonstrate in                                                FT = AT · VT .                          (10)
Figure 2. We can express as:
                                                                               Two streams process the shape feature in parallel to obtain
                       f = Concat(( fI + fˆI ), fP ).                  (8)
                                                                               contextual feature within their respective channels. The
To apply the whole CMF module to the image feature F I and                     acquired spatial feature FS and temporal feature and FT are
the pillar-based feature F P , we can acquire the 3D-enhanced                  subsequently concatenated along the channel dimension to
feature F̂ I and final feature F. Therefore, Our CMF module                    form the spatial-temporal 3D shape feature Fβ . It can be
reaches cross modality alignment with adaptive fusion. The                     expressed as:
frame-level features are strengthened with the help of pillar-                                     FβS T = Concat(FS , FT ).          (11)
based features, and the multi-frame misalignment and partial
occlusion are mitigated via the CMF module.                                    The interaction process extracts the spatial and temporal
                                                                               context of video-level shape parameter β. To further refine the
D. Shape-Aware Spatial-Temporal Interaction Module                             extracted features, motivated by [74], we normalize them and
                                                                               pass them through a MLP layer consisting of linear layers and
   The 3D shape parameters control and describe the pedes-
                                                                               a activation function. A shortcut connection is then applied,
trian shape in the 3D mesh [45]. Thus, generating distinctive
                                                                               producing the final spatial-temporal 3D shape feature F̂βS T ,
3D shape representations is critical for distinguishing different
                                                                               which can be expressed as:
pedestrians with similar appearance. Despite existing methods
[41], [42] utilize 3D shape parameters to create features in                                    F̂βS T = MLP(Norm(FβS T )) + FβS T ,                (12)
video scenarios, they struggle to effectively obtain high-level
spatial-temporal dependencies. Inspired by spatial-temporal                    where Norm denotes the normalization layer. In this manner,
learning strategies [73], we propose the Shape-aware Spatial-                  our SSTI module effectively extracts and fuses spatial and
Temporal Interaction (SSTI) module to extract distinctive                      temporal dependencies in parallel. The receptive field captures
spatial-temporal 3D shape representations, thereby enhancing                   the interactions between spatial and temporal dimensions.
the model’s robustness in capturing pedestrian shapes. As                      As a new pedestrian characteristic, the produced spatial-
expressed in Figure 2, we first project 3D shape parameters                    temporal 3D shape representation is both complementary and
into feature space to generate 3D shape features. In specific,                 distinguishable, aiding in target pedestrian identification in
given video-level 3D shape parameters β ∈ RT ×10 of a                          video scenarios. Therefore, it helps alleviate the impacts of
pedestrian, we embed them into a high-dimensional space via                    appearance confusion and further mitigates issues including
the shape embedding layer, generating 3D shape features Fβ                     the multi-frame misalignment and partial occlusion.
with a shape of T × S × C, where T represents the temporal
dimension, S denotes the spatial dimension, and C denotes the                  E. Loss Function
channel dimension. The shape embedding layer employs linear                      We apply the cross-entropy loss with label smoothing [75]
layers followed by a activation function and a normalization                   and the triplet loss with batch hard mining [76] for training.
layer.                                                                         The Re-ID loss LID for our network can be formulated as:
   Then, the spatial-temporal interaction process is performed,
as illustrated in Figure 4. Specifically, we derive the Query                                             LID = Lcls + Ltri ,                       (13)

         Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
JING et al.: 3D-AIDED PEDESTRIAN REPRESENTATION LEARNING FOR VIDEO-BASED PERSON Re-ID                                                              12837



where Lcls is the classification loss and Ltri is the triplet loss.           During the training, the Adam [89] optimizer with weight
We determine the weight of the Re-ID loss for the SSTI                        decay 5 × 10−4 is employed as optimizer. The initial learning
module with a hyper-parameter λ. The overall loss Loverall for                rate is set to 3.5 × 10−4 , and the learning rate decay strategy
our network can be expressed as:                                              [90] is employed. The random horizontal flipping and random
                                                                              erasing are utilized for image data augmentation. Our network
                     Loverall = LID
                                 CMF
                                     + λLID
                                         SSTI
                                              ,                      (14)
                                                                              is trained with maximum 500 epochs. During evaluation, the
       CMF      SSTI                                                          final feature representation is obtained by concatenating the
where LID  and LID   indicate the Re-ID loss for the CMF
and the SSTI module, respectively.                                            video-level feature representation from the STA module Ψ
                                                                              with the spatial-temporal 3D shape feature F̂βS T . The cosine
                        IV. E XPERIMENTS                                      similarity is used as distance metric.
A. Datasets and Evaluation Protocols
   We evaluate our proposed 3DAPRL on four video-based                        C. Comparison With State-of-the-Art Methods
person Re-ID datasets including MARS [9], iLIDS-VID [10],                        In this section, we compare the 3DAPRL with other meth-
PRID-2011 [11] and DukeMTMC-VID [84], [85]. Since there                       ods [21], [24], [25], [27] [28], [29], [30], [31], [32], [35], [38],
is no existing video-based Re-ID datasets with corresponding                  [39], [49], [51], [54], [55], [56], [59], [77], [78], [79], [80],
3D representation data, we generate 3D pedestrian mesh for                    [81], [82], [83] on three video-based Re-ID datasets including
these three datasets by using off-the-shell 3D human recon-                   MARS [9], iLIDS-VID [10] and PRID-2011 [11]. For iLIDS-
struction networks [64], [65], [66].                                          VID and PRID-2011, we only report the cumulative accuracy
   1) MARS: is a large-scale video-based Re-ID dataset that                   since these two datasets have only one correct match in gallery.
collected by six cameras, and it is composed of 17,503                        The results are shown in Table II.
tracklets corresponding to 1,261 identities and additional 3,248
                                                                                 1) MARS: On MARS dataset, the proposed 3DAPRL
distractors. 625 identities are used for training and the remain-
                                                                              achieves the best performance compared with other SOTA
ing 636 identities are used for testing. The video sequences
                                                                              methods. Specifically, the 3DAPRL reaches 90.3% mAP and
in MARS are detected using DPM [86] detector and tracked
                                                                              93.1% Rank-1 accuracy, outperforming all the mentioned
using GMMCP [87] tracker.
                                                                              SOTA methods. Compared with the second-best method TF-
   2) iLIDS-VID: is captured by two non-overlapping cam-
                                                                              CLIP [30], which utilize pre-trained CLIP visual encoder
eras. It is comprised of 600 tracklets with 300 identities. The
                                                                              to achieve one-stage CLIP-based video Re-ID learning, our
length of each video sequence varies from 23 to 192 frames,
                                                                              3DAPRL reach 0.9% increase on mAP and 0.1% increase on
with an average of 73 frames.
                                                                              Rank-1. Besides, SDCL [28] exploit event data to enhance
   3) PRID-2011: contains 385 and 749 identities captured by
                                                                              the model performance. Differently, we employ 3D meshes
two disjoint cameras, respectively. Only the first 200 pedestri-
                                                                              that contains geometric information and shape cues, and yield
ans appear in both cameras.
                                                                              higher performance than SDCL [28], by 3.8% and 2.0% on
   4) DukeMTMC-VID: is another large-scale video-based
                                                                              mAP and Rank-1 accuracy, respectively.
Re-ID dataset comprising 4,832 tracklets and 1,812 identities.
                                                                                 2) iLIDS-VID: On iLIDS-VID dataset, our 3DAPRL also
Each video sequence contains an average of 168 frames.
                                                                              achieves the best performance compared to other SOTA meth-
For training, 702 identities with 2,196 tracklets are utilized,
                                                                              ods. Although we observe a reduction of 0.9% and 0.6%
while the remaining 702 identities are used for testing, with
                                                                              on Rank-5 accuracy when compare to the HASI [27] and
3,338 tracklets.
                                                                              the second-best performing methods [29], [55], [79], our
   5) Evaluation Protocols.: Following the previous standard
                                                                              method attains 94.7% Rank-1 accuracy which corresponds
of video Re-ID, we evaluate the performance of our method
                                                                              the best performance. In addition, our method surpasses the
by adopting Average Precision (mAP) and the Cumulative
                                                                              methods leveraging other modality information on Rank-1
Matching Characteristics (CMC).
                                                                              accuracy, e.g., TF-CLIP [30] that uses CLIP visual encoder
                                                                              and SDCL [28] that uses event data, by 0.2% and 1.5%.
B. Implementation Details                                                        3) PRID-2011: On PRID-2011 dataset, our 3DAPRL and
   Our method is implemented based on PyTorch platform                        MFA [79] reach the best Rank-1 accuracy of 96.6% and
with NVIDIA Tesla V100 GPUs. For video sequences branch,                      gain 0.1% higher performance than the second-best method
the ResNet-50 [72] pre-trained on ImageNet [88] is employed                   SDCL [28]. It also illustrate that 3D pedestrian representation
as image backbone network. We sample 8 frames randomly                        contain richer pedestrian information, including geometric and
followed by RRS strategy [52], and each selected frame                        shape cues compare to event information which only provide
is resized to 256 × 128. For the 3D branch, meshes are                        limited motion cues.
transformed to point clouds and we use PointPillar [46]                          4) DukeMTMC-VID: On DukeMTMC-VID dataset,
without its detection head as backbone. We normalize the point                our proposed 3DAPRL achieves the highest mAP and
cloud followed by [68]. For the Spatial-Temporal Aggregation                  Rank-1 accuracy among existing SOTA methods. Specifically,
(STA) module Ψ, we employ the STAM module with its                            it outperforms PSTA [21] and TMT [82] with improvements
pyramid architecture as proposed in [21], leveraging its ability              of 0.2% in mAP and 0.3% in Rank-1 accuracy, respectively.
to capture spatial and temporal correlations from frame-level                 This performance gain is attributed to the incorporation of
fusion features to produce spatial-temporal representations.                  3D pedestrian information, which enables our method to

        Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.
12838                                    IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 12, DECEMBER 2025



                                                                 TABLE II
P ERFORMANCE (%) C OMPARISON OF O UR M ETHOD AND S TATE - OF - THE -A RT M ETHODS ON MARS, I LIDS-VID AND PRID-2011 DATASETS . N UMBERS
       M ARKED W ITH B OLD I NDICATE THE B EST P ERFORMANCE , AND U NDERLINED N UMBERS I NDICATE THE S ECOND B EST P ERFORMANCE




generate more discriminative video representations compared                   accuracy, respectively. This indicates the benefits of aggre-
to single-modal approaches.                                                   gating frame-level feature temporally [21]. Based on the SSTI
   Conclusively, it can be observed that the 3DAPRL achieves                  module, Index-3 outperforms Index-2 by 2.4% mAP and 1.2%
competitive performance compare to other SOTA methods.                        Rank-1 accuracy. It demonstrates that as a new and discrim-
The performance improvement is contributed by the rich                        inative pedestrian characteristic extracted from SSTI module,
information provided by the 3D pedestrian mesh, which is not                  the spatial-temporal 3D shape feature contributes to the
fully considered in previous video Re-ID studies. Particularly,               video representation, thus improving the model performance.
the fusion of 3D and image information in the CMF module                      It also verifies the effectiveness of our SSTI module. For
enhance the frame-level features to better capture the pedes-                 Index-4, it far exceeds Index-2 by 4.5% and 2.7% on mAP and
trian information in image while preserving 3D information.                   Rank-1 accuracy, respectively, which shows the significance of
The SSTI module extracts the spatial-temporal information                     our CMF module. This illustrates that fusing image feature
of 3D shape, which contributes to the final video represen-                   with pillar-based feature enables the model to capture key
tation. Although our method has larger model parameter size                   information in image, while assimilating geometric informa-
than typical CNN-based models [21], [38], [56], [78], [79],                   tion and shape cues from the 3D pedestrian feature. Eventually,
[82], [83], the inclusion of 3D data drives superior results.                 our network reaches the SOTA performance by incorporating
In addition, the 3DAPRL is significantly more parameter-                      the CMF module and the SSTI module as presented in
efficient than Transformer-based methods [27], [29], [30],                    Index-5, which shows the superiority of the integration of all
[59], [80] while still demonstrating improved results. Notably,               components in our network.
when compared to the other multi-modal technique [30], our                       2) Computation Complexity Analysis: We analyse model
method exhibits advantages in both model parameter and                        efficiency using parameter count (space complexity) and
overall performance.                                                          FLOPs (time complexity), common in video Re-ID. Note
                                                                              that we report the FLOPs for a four-frames sequence. As
D. Ablation Study                                                             Table III illustrates, model parameters and FLOPs predictably
   To verify the contribution of each component in our                        increase with the integration of more modules. Notably, Our
3DAPRL, we conduct experiments on MARS dataset. The                           SSTI module requires more parameters than the STA [21]
baseline model is built upon a ResNet-50 backbone [72] pre-                   (14.24M vs. 6.9M), but fewer FLOPs (1.08G vs. 1.55G) while
trained on ImageNet [88].                                                     delivering superior performance gains. Introducing 3D point
   1) Effectiveness of Each Component: We evaluate the                        cloud features with the CMF module further increases model
contribution of each component of the proposed 3DAPRL,                        parameters and FLOPs greatly, yet this is directly correlated
by adding the components to baseline gradually. As shown                      with significant performance improvements. Overall, while the
in Table III, compared with Index-1 and Index-2, Index-                       proposed 3DAPRL framework incurs an additional 51.31M
2 obtains 0.7% and 0.9% increases on mAP and Rank-1                           parameters and 4.8G FLOPs compared to the baseline, it

        Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:50 UTC from IEEE Xplore. Restrictions apply.

codex
3D-Aided 这篇的证据链也比较标准：先把视频 ReID 的三类失败归结为“帧级特征质量不足”，再分别用 CMF 证明 3D 对齐能救遮挡和错位，用 SSTI 证明 3D shape 能救相似外观混淆。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 \"A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf\" - | sed -n '1,735p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

1

A Semantic-Aware Attention and Visual Shielding
Network for Cloth-Changing
Person Re-Identification
Zan Gao , Member, IEEE, Hongwei Wei, Weili Guan , Member, IEEE, Jie Nie , Member, IEEE,
Meng Wang , Fellow, IEEE, and Shengyong Chen , Senior Member, IEEE

Abstract— Cloth-changing person re-identification (ReID) is a
newly emerging research topic that aims to retrieve pedestrians
whose clothes are changed. Since the human appearance with
different clothes exhibits large variations, it is very difficult
for existing approaches to extract discriminative and robust
feature representations. Current works mainly focus on body
shape or contour sketches, but the human semantic information
and the potential consistency of pedestrian features before and
after changing clothes are not fully explored or are ignored. To
solve these issues, in this work, a novel semantic-aware attention
and visual shielding network for cloth-changing person ReID
(abbreviated as SAVS) is proposed where the key idea is to shield
clues related to the appearance of clothes and only focus on visual
semantic information that is not sensitive to view/posture changes.
Specifically, a visual semantic encoder is first employed to locate
the human body and clothing regions based on human semantic
segmentation information. Then, a human semantic attention
(HSA) module is proposed to highlight the human semantic
information and reweight the visual feature map. In addition,
a visual clothes shielding (VCS) module is also designed to extract
a more robust feature representation for the cloth-changing task
by covering the clothing regions and focusing the model on
the visual semantic information unrelated to the clothes. Most
importantly, these two modules are jointly explored in an endto-end unified framework. Extensive experiments demonstrate
Manuscript received 17 July 2022; revised 9 October 2023; accepted
29 October 2023. This work was supported in part by the National Natural
Science Foundation of China under Grant 62372325 and Grant 61872270; in
part by the Young Creative Team in universities of Shandong Province under
Grant 2020KJN012; in part by the Jinan 20 Projects in universities under Grant
2020GXRC040; and in part by the Shandong Project toward the Integration
of Education and Industry under Grant 2022PYI001, Grant 2022PY009, and
Grant 2022JBZ01-03. (Corresponding authors: Hongwei Wei; Zan Gao.)
Zan Gao is with the Shandong Artificial Intelligence Institute, Qilu University of Technology (Shandong Academy of Sciences), Jinan 250014, China,
and also with the Key Laboratory of Computer Vision and System, Ministry of
Education, Tianjin University of Technology, Tianjin 300384, China (e-mail:
zangaonsh4522@gmail.com).
Hongwei Wei is with the Shandong Artificial Intelligence Institute, Qilu
University of Technology (Shandong Academy of Sciences), Jinan 250014,
China (e-mail: wmolang@163.com).
Weili Guan is with the Faculty of Information Technology, Monash University, Clayton, VIC 3800, Australia (e-mail: honeyguan@gmail.com).
Jie Nie is with the College of Information Science and Engineering, Ocean
University of China, Qingdao 266100, China (e-mail: niejie@ouc.edu.cn).
Meng Wang is with the School of Computer Science and Information
Engineering, Hefei University of Technology, Hefei 230009, China (e-mail:
eric.mengwang@gmail.com).
Shengyong Chen is with the Key Laboratory of Computer Vision and
System, Ministry of Education, Tianjin University of Technology, Tianjin
300384, China (e-mail: sy@ieee.org).
Color versions of one or more figures in this article are available at
https://doi.org/10.1109/TNNLS.2023.3329384.
Digital Object Identifier 10.1109/TNNLS.2023.3329384

that the proposed method can significantly outperform state-ofthe-art methods, and more robust features can be extracted for
cloth-changing persons. Compared with multibiometric unified
network (MBUNet) (published in TIP2023), this method can
achieve improvements of 17.5% (30.9%) and 8.5% (10.4%) on
the LTCC and Celeb-reID datasets in terms of mean average
precision (mAP) (rank-1), respectively. When compared with the
Swin Transformer (Swin-T), the improvements can reach 28.6%
(17.3%), 22.5% (10.0%), 19.5% (10.2%), and 8.6% (10.1%) on
the PRCC, LTCC, Celeb, and NKUP datasets in terms of rank-1
(mAP), respectively.
Index Terms— Cloth-changing person re-identification (ReID),
human semantic attention (HSA), semantic-aware, visual clothes
shielding (VCS).

I. I NTRODUCTION

T

HE person re-identification (ReID) task is to explore the
usefulness of image retrieval techniques in the public
security domain. It is an upstream task of the person detection
or person localization tasks, while the ReID task needs to
find additional clues about the target person based on the
already-acquired person images or video sequences. A typical
person ReID system aims to discover matching persons from
a gallery library and return the retrieval sequence based on
the query probe. Furthermore, as air pollution continues to
rise, individuals frequently wear face masks as a precautionary
measure in their everyday routines. Additionally, surveillance
cameras often capture face images at a noticeably reduced
size. Consequently, even advanced face recognition methods
often struggle to accurately identify individuals under such
circumstances. To solve this issue, researchers [1], [2], [3],
[4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15],
[16], [17] have developed the person ReID technique, which
is an important supplement to the face recognition technique
[18], [19] and a special case of feature extraction [20], [21],
[22], and several person ReID datasets [23], [24], [25] have
been released. Moreover, Ye et al. [17] have surveyed the
existing person ReID approaches. We can find that these
approaches are very effective for the person ReID task with
short time spans where the human appearance features are
fully used for visual matching, but when the surveillance
acquisition period becomes longer, the complexity of clothing
changes subsequently increases. Fig. 1 shows some examples
of cloth-changing person ReID images, where each row displays the images of the same person wearing different clothes.
From them, we can observe that the differences in the visual
appearances of the same person with different clothes are

2162-237X © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
2

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

Fig. 1. Examples of cloth-changing person ReID images. The images in
each row belong to the same person with different clothes.

very large, and it is also very difficult for humans to identify
them. In other words, when data from different cameras are
collected for a long time, the appearance of the person’s
clothing cannot be used as the feature representation. If the
existing person ReID approaches are directly applied in this
case, their performance deteriorates dramatically, and they
often fail. The reason for this is that the above approaches
mainly rely on visual clothing appearance, and they cannot
provide a robust feature representation for each identity (ID)
with different clothes; thus, discriminative and robust feature
extraction is an urgent issue for the cloth-changing person
ReID task.
To date, a few researchers [26], [27], [28], [29], [30], [31],
[32], [33], [34], [35], [36] have made useful attempts for
the cloth-changing person ReID task. For example, Huang
et al. [27] proposed an augmented representation with vector
neurons for cloth-changing. Yang et al. [29] proposed human
contour information and polar coordinate transformation to
obtain the results of pedestrian matching. Qian et al. [30] proposed a shape embedding module and a clothing-eliminating
shape-distillation module (SE + CESD). To enrich clothing
styles, Zheng et al. [31] proposed a generative adversarial
model (GAM) module to expand the training data. Yu et al.
[32] proposed a new solution by involving rich clothing
templates in training, and in the retrieval, different clothing
templates are added into the query in sequence. Jin et al.
[33] proposed a framework called GI-ReID that leverages gait
recognition as an auxiliary task to learn cloth-agnostic representations for efficient and latency-free person ID matching
in surveillance. Yang et al. [34] proposed SirNet-based on
positive and negative sample clustering to increase interclass
differences and reduce intraclass gaps. These approaches are
very good for trying to solve the cloth-changing person ReID
problem, but since the human appearance exhibits large variations with different clothes, it is very difficult for existing
approaches to extract discriminative and robust feature representations. Moreover, current works mainly focus on body
shape or contour sketches, but the human semantic information
and the potential consistency of pedestrian features before and
after changing clothes are not fully explored or are ignored.

To solve these issues, in this work, we propose a novel
end-to-end SAVS algorithm for the cloth-changing person
ReID task to obtain more discriminative and robust features
that are irrelevant to clothes. The two key points of the
problem to be solved in the field of cloth-changing person
ReID: 1) the same person wearing different clothes and
2) different people wearing the same clothes. The SAVS
method has already eliminated both effects from appearance
simultaneously by attentional weighting and shielding pixels.
Extensive experimental results on four cloth-changing person
ReID datasets demonstrate that SAVS can outperform stateof-the-art person ReID approaches, and more discriminative
and robust features can be obtained that can effectively solve
the cloth-changing issue. The main contributions of this article
are summarized as follows.
1) We develop a novel end-to-end SAVS network for
cloth-changing person ReID that consists of visual
semantic encoding and visual semantic decoding. The
key idea is to shield clues related to the appearance of
clothes and only focus on visual semantic information
that is not sensitive to view/posture changes. In this way,
the negative effect of the clothing information can be
reduced as much as possible.
2) We design a human semantic attention (HSA) module
to highlight the human information and reweight the
visual feature map that is very helpful for obtaining
more discriminative features, and then we develop a
visual clothes shielding (VCS) module to extract a more
robust feature representation by focusing the model
on the visual semantic information unrelated to the
clothes. Most importantly, these two modules are jointly
explored in an end-to-end unified framework. In this
way, more discriminative and robust features can be
extracted that are irrelevant to the cloth-changing or pose
variants.
3) We systematically and comprehensively evaluate the
SAVS algorithm on four public cloth-changing person
ReID datasets, and the experimental results demonstrate
that the SAVS approach can obtain more discriminative and robust features that are irrelevant to clothes;
moreover, it can significantly outperform state-of-theart cloth-changing person ReID methods in terms of the
mean average precision (mAP) and rank-1.
The remainder of this article is organized as follows.
Section II introduces the related work, and Section III
describes the proposed SAVS method. Section IV describes the
experimental settings and the analysis of the results. Section V
presents the details of the ablation study, and concluding
remarks are presented in Section VI.
II. R ELATED W ORK
To date, many person ReID approaches have been proposed.
According to the person’s visual appearance, these methods
can be roughly divided into clothing-consistent person ReID
and cloth-changing person ReID. In the following, we will
separately introduce them.
A. Clothing-Consistent Person ReID
In earlier times, people [31], [37], [38] made efforts to
develop related methods for clothing-consistent person ReID
where the visual appearance of the clothes was consistent
for the same person. For example, Sun et al. [9] proposed

Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
GAO et al.: SEMANTIC-AWARE ATTENTION AND VISUAL SHIELDING NETWORK

a part-based convolutional baseline (PCB) module where a
base feature map was first obtained and then equally divided
into six feature blocks in the horizontal direction. This method
is simple but very effective and has become an important
benchmark in the field of person ReID. Wang et al. [14]
proposed the multiple granularity network (MGN), a multibranch deep network architecture where one branch is built
for the global feature representation and two branches are built
for the local feature representation to capture the preference
information of the pedestrian ID categories from the whole
image. In this way, discriminative information with various
granularities can be obtained via an end-to-end feature learning
strategy. Gao et al. [16] proposed a deep spatial pyramid-based
collaborative feature reconstruction model (DCR) where all
blocks of the person were jointly reconstructed; in this way, the
issues of occlusion, pose changes, and observation perspective
changes can be solved. In addition, some methods employ
human skeleton points or human surface texture as a priori
knowledge to guide pedestrian ReID. Song et al. [39] proposed
a mask-guided contrastive attention model (MGCAM) to learn
features separately from the body and background regions;
moreover, a novel region-level triplet loss was designed to
restrain the features learned from different regions. Miao et al.
[40] proposed a novel pose-guided feature alignment (PGFA)
method where a pose estimator was utilized to detect key
points of the human body in pedestrian images, and then
these keypoints were used to decide whether a specific body
part was occluded. Gao et al. [5] proposed a novel texture
semantic alignment (TSA) approach with visibility awareness
for the partial person ReID task, where the occlusion issue
and changes in poses were simultaneously explored in an endto-end unified framework. Zheng et al. [41] proposed for the
first time to solve the person ReID problem in 3-D space by
learning features from human appearance and 3-D geometric
structure in a coherent manner. Wang et al. [15] proposed
a novel spatial rescaling (SpaRs) layer to help convolutional
neural networks (CNNs) to see more, and it introduced spatial
relations among the feature map activations back to guide
the model to focus on a broad area in the feature map.
Hou et al. [2] proposed a novel interaction-aggregation-update
(IAU) block to comprehensively leverage the spatial–temporal
context information for high-performance person reID. Zhang
et al. [1] introduced a unified attribute-guided collaborative
learning scheme tailored for partial person ReID. In their work,
they proposed an adaptive threshold-guided masked graph
convolutional network, which effectively incorporates human
attributes and a cyclic heterogeneous graph convolutional
network. This integration facilitates the fusion of cross-modal
pedestrian information through both intragraph and intergraph interactions. Ye et al. [17] performed a comprehensive
examination of closed-world person ReID from three distinct
viewpoints, providing an in-depth analysis. They further evaluated the strengths of existing person ReID approaches and
proposed a robust attention generalized mean pooling with
weighted triplet loss (AGW) baseline method. Although these
models are robust to changes caused by poses, lighting, and
viewing angles, they are vulnerable to clothing changes, as the
models heavily rely on the consistency of the appearance of
clothes.
B. Cloth-Changing Person ReID
Since the visual appearance of the pedestrians in the
cloth-changing person ReID task changes dramatically after

3

a long period, it is very difficult to extract more discriminative and robust feature representations. If the existing
clothing-consistent person ReID methods are directly applied
to this task, their performance is unsatisfactory. Therefore,
to accelerate the development of cloth-changing person ReID
techniques, some cloth-changing person ReID datasets have
been built and released, such as LTCC [30], PRCC [29], CelebreID [27], and NKUP [42]. Moreover, several researchers [27],
[28], [29], [30], [35], [36], [43], [44], [45], [46] have made
some attempts to address this problem and then assessed their
performance on a certain dataset. For example, Yang et al. [29]
proposed a spatial polar transformation (SPT) + angle-specific
extractor (ASE) module, where human contour sketching
information was used to substitute for human color information. Moreover, an SPT layer was designed to transform the
contour sketch image, and then a multistream network was
used to aggregate multiple granular features to better discriminate people by changing the sampling range of the SPT layer.
In this way, the changes in visual appearance caused by clothing changes could be reduced. Qian et al. [30] proposed an
SE + CESD module, where the main idea was to completely
delete information related to the appearance of clothes and
only focus on body shape information that is not sensitive to
changes in perspective and posture. The former was used to
encode shape information from human body keypoints, and the
latter was utilized to adaptively distill the ID-relevant shape
features. Huang et al. [27] designed a ReIDCaps module where
a vector neuron concept was proposed. For each vector neuron,
its direction was used to represent the changes in clothing
information, and its length was utilized to identify the people.
In this way, the clothing changes of a specific person can be
perceived, and the auxiliary modules can be used to enhance
the robustness of the module. Zheng et al. [31] proposed a
jointly couples discriminative and generative learning in a
unified network (DG-Net) module where a generative model
was utilized to automatically generate person images with
different appearances regarding clothing. Li et al. [26] and
Yu et al. [32] proposed a new solution for changing clothes
called clothes changing person set (COCAS) and COCAS plus
(COCAS+), respectively, where rich clothing templates were
supplied; thus, in the query, both the clothing template image
and an image of the target person wearing other clothes were
fed into the module to find the target image. Gao et al. [45]
proposed a novel multigranular visual-semantic embedding
algorithm (MVSE) for cloth-changing person ReID, where
visual semantic information and human attributes are embedded into the network. Hong et al. [44] proposed a fine-grained
shape-appearance mutual learning framework that can learn
fine-grained discriminative body shape knowledge in a shaped
stream and transfer it to an appearance stream to complement
the clothing-unrelated knowledge in the appearance features.
Shu et al. [46] proposed a semantic-guided pixel sampling
approach for the cloth-changing person ReID task which
forces the model to automatically learn clothing-irrelevant
cues that are irrelevant to upper clothes and pants. Gu et al.
[47] proposed a clothes-based adversarial loss (CAL) to mine
clothes irrelevant features from the original RGB images by
penalizing the predictive power of the ReID model. Yang
et al. [36] proposed a causality-based autointervention model
(AIM) to mitigate clothing bias for robust cloth-changing
person ReID. Yang et al. [43] proposed an Auxiliaryfree Competitive IDentification (ACID) model to achieve a
win–win situation by enriching the ID-preserving information
conveyed by the appearance and structure features while

Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
4

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

Fig. 2. Pipeline of the SAVS approach is an end-to-end network architecture, and it mainly consists of the visual semantic encoder, the visual semantic
decoder, and the loss function. In the visual semantic encoder, the foreground information and visual clothing shielding information are generated for each ID
to enrich the image representations. The visual semantic decoder focuses the module on human semantic information and visual clothing shielding information;
thus, a more discriminative and robust feature can be extracted. Note that in the SAVS, Swin-T is used as the backbone. GAP indicates the global average
pooling, and FC denotes the fully connected layers.

maintaining holistic efficiency. Zhang et al. [35] proposed a
novel multibiometric unified network (MBUNet) for learning
the robustness of cloth-changing ReID model by exploiting
clothing-independent cues. Since the human appearance with
different clothes exhibits large variations, it is very difficult for
existing approaches to extract discriminative and robust feature
representations, and their performances need to be further
improved. Moreover, the human semantic information and the
potential consistency of pedestrian features before and after
changing clothes are not fully explored or are ignored. Thus,
in this work, we fully explore the available visual semantic
information and the potential consistency of features and then
extract a generalized and robust feature to represent a person
wearing different clothes.

body shape, posture, accompanying markers, and other comprehensive information. In other words, the idea of extracting
human semantic information is to first eliminate the influence
of background changes through HSA to highlight all human
features, and then eliminate the influence of cloth-changing
through VCS to highlight human features unrelated to clothes,
i.e., effective human semantic information for cloth-changing
person ReID. Notice that “Human semantic information” is
summarized as biometric cues that are helpful in identifying
people. In the following, we will introduce the visual semantic
encoder, the visual semantic decoder, and the loss function.
Algorithm 1 shows the complete procedure of the proposed
SAVS model.

III. P ROPOSED A PPROACH
As shown in Fig. 2, our proposed SAVS method consists of
two stages: a visual semantic encoder and a visual semantic decoder. Moreover, the visual semantic decoder mainly
consists of the backbone, the HSA module, and the VCS
module, where these three modules are jointly explored in an
end-to-end unified framework. In addition, the loss function
is used to guide the network optimization. Specifically, the
foreground image of the human body (with background pixel
values of 0), the original image, and the shielding image are
fed into the HSA, the backbone, and VCS streams, to obtain
the enhanced feature, the original feature, and the shielding
feature, respectively, and then the loss functions are further
utilized to mine their relationship among these features. Since
the clothes information is included in the enhanced feature
and the original feature, thus, we invalidate the clothing
information by semantic loss through the VCS stream, at this
point the only human semantic information left is the cues
that are not related to the clothing, such as facial features,

A. Visual Semantic Encoder
With the development of deep learning techniques [48],
[49], researchers have designed different CNNs for the person
ReID task, but since visual appearances are very different
in the cloth-changing person ReID task, it is very difficult for an individual feature to extract a generalized and
robust feature to represent a person with different clothes.
To accommodate clothing variations within the limited data,
enriched feature representations for each ID are needed.
Thus, in the visual semantic encoder, the foreground image
and the shielding image are generated with the help of
human semantic segmentation maps. Specifically, for the
original image, the pretrained self-correction for human parsing (SCHP) module [50] is employed to obtain the human
semantic segmentation information, where the human body
is divided into 18 semantic parts. To make it suitable for
the cloth-changing person ReID task, these 18 semantic parts
are recombined to obtain seven parts, including background,
head, torso, pants, arms, legs, and belongings. Fig. 3 shows

Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
GAO et al.: SEMANTIC-AWARE ATTENTION AND VISUAL SHIELDING NETWORK

5

Algorithm 1 Training Procedure of the SAVS
Input: Input image I ∈ R 3×H ×W
The corresponding semantic segmentation map M ∈ R 1×H ×W
epochs = T , t = 0, the learning rate = 3.5e−3 ,
the shrink rate = 0.1 at 40 to 60
initialize the parameters of the swim-transformer(backbone)
Output: the SAVS parameters θ
while t < T do
1: Extract the foreground image and the shielding image from I
based on the semantics of the segmentation map M
2: Obtain the original feature map Fo ∈ R 7×7×1,024
from I by Swin-Transformer (Swin-T)
3: Obtain the original feature Fo′ ∈ R 1,024
by using global average pooling
4: Obtain the enhanced feature Fe′ ∈ R 1,024
by using equation (1) and (2) in HSA module
5: Obtain the shielding feature Fs′ ∈ R 1,024 in VCS module
6: Compute the semantic loss by using equation (5)
to minimize the distance between Fo′ and Fs′
7: Compute the cross-entropy classification loss
and the circle loss using equation (4) based on Fe′
8: Train the SAVS coaching by the total loss
and update the SAVS parameters θ
end while

the results of the human semantic segmentation maps.
In the following, we provide detailed information on how
the foreground image and the visual shielding image can be
obtained.
1) Foreground Image: Many previous works focus on
obtaining the global or local features for each cloth-changing
person, but in this work, we pay more attention to the
latent association between the foreground and background
information, and the foreground image is considered another
image representation of the original image. Thus, the key step
is to separate the foreground and background for each original
image. Based on the human semantic segmentation maps,
we perform a binarization process to distinguish between
background and nonbackground. All information other than
the background is used as the foreground information, such as
the torso and legs. Thus, the foreground mask can be obtained.
Finally, we associate each body part with its corresponding
mask. These pixels inside the mask boundary or outside the
mask boundary are considered the foreground image or the
background image, respectively. Fig. 3 provides the results of
the foreground images.
2) Visual Shielding Image: In the cloth-changing person
ReID scenario, the most common change occurs for the upper
clothes and pants. To obtain a more discriminative feature for
cloth-changing tasks, the visual shielding image is obtained
as another new image representation of the original image
by covering the clothing regions of the upper clothes and
pants. Specifically, unlike separating foreground information,
at this step, finer-grained segmentation labels are required
to accurately find the local locations of the upper clothes
and pants based on the aforementioned semantic segmentation
information. In this way, the shielding mask can be obtained
where the pixel value is set to one if it belongs to the mask,
or its value is set to zero. Finally, we further combine the original image and the shielding mask by matrix multiplication;
thus, the visual shielding image can be obtained. Fig. 3 also
displays the results of the visual shielding images.

Fig. 3.
Results of the visual semantic encoder process. From top to
bottom: Original images, the corresponding semantic segmentation maps,
the foreground images, the visual shielding images, and the visual rendering
images.

B. Visual Semantic Decoder
The visual semantic decoder mainly consists of the backbone, the HSA module, and the VCS module, where these
three modules are jointly explored in an end-to-end unified framework. In our experiments, the Swin Transformer
(Swin-T) [51] is used as the backbone to obtain the original
feature map Fo ∈ R 7×7×1,024 , whose input is the original
image, and then global average pooling (GAP) is utilized
to obtain the original feature Fo′ ∈ R 1,024 . Note that to
accommodate person feature extraction, in Swin-T, we set the
patch size to 4 for patch partitioning, and we set the window
size to 7 for each patch to compute the self-attention inside
each window. We keep the transformer blocks from stage 1
to stage 4 to extract visual features and discard the final
classification layer of the network. Moreover, the parameters of
the Swin-T pretrained on the ImageNet dataset are further used
in the following joint optimization. In the following, we will
introduce the HSA and VCS modules separately.
1) Human Semantic Attention Module: Current works
mainly focus on body shape or contour sketches, but human
semantic information has not been fully explored. Thus,
in this work, an HSA module is designed to highlight the
human semantic information and reweight the visual feature
map, and the latent correlation between the channels of the
convolved features is mined. In this way, the negative effect
of the background information can be reduced as much as
possible, and more discriminative features can be obtained.

Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
6

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS

The structure of the HSA module is shown in Fig. 2. In this
module, ResNet50 is used as the basic network architecture,
and it is first pretrained on the ImageNet dataset. Then, the
parameters of ResNet50 are further jointly optimized with
other networks. Specifically, in the HSA module, the feature
map FA ∈ R 7×7×1,024 is first obtained by ResNet50, and then
it can be further fed into the GAP to obtain the feature vector
FA′ ∈ R 1024 . Moreover, the feature vector FA′ is subsequently
passed through two fully connected layers, where the first
layer is used to obtain a feature representation with reduced
dimensionality (1, 024/r ) (in our experiment, r is set to
16), and the second layer is used to increase the feature
dimensionality (1, 024). This can be defined as


Fw = σ W2 · δ W1 · f g (FA ) .
(1)
Fw ∈ R 1024 denotes the weighted feature vector with human
semantic information whose input image is the foreground
image. f g indicates the operation of GAP. W1 and W2 represent
the parameters of two fully connected layers. σ and δ indicate
the sigmoid activation and the rectified linear unit (ReLU)
function, respectively. To make the original feature highlight
the human semantic information, the reweight operation is
further used for the original feature map Fo ∈ R 7×7×1,024
obtained from the original image and the backbone (Swin-T)
by the weighted feature vector Fw , and it can be obtained by
Fe′ = f g {Fw ⊗ Fo }

(2)

where ⊗ denotes the channelwise multiplication between the
weight vector Fw and the original feature map Fo . Moreover, the GAP operation is used to obtain the feature vector
that can describe the persons for subsequent classification
training. Fe′ ∈ R 1,024 indicates the enhanced feature, which
is the output of the reweight operation and GAP. Since Fw
is obtained from the foreground image that can selectively
emphasize the human feature channels, Fe′ can pay more
attention to the human semantic information, and the negative
effect of the background information can be reduced as much
as possible. Therefore, the extracted feature is more discriminative and robust.
2) Visual Clothes Shielding Module: In the cloth-changing
person ReID scenario, the clothes of the person often change;
thus, it will be very difficult to obtain more discriminative
features for cloth-changing tasks. We hope that we can extract
clothing-irrelevant features where more attention is given
to human semantic information; thus, the VCS module is
designed, which focuses the model on visual semantic information unrelated to clothes. Specifically, suppose that there are
b original images and b corresponding segmentation maps in
each batch of the training stage, which can be denoted as I =
[I1 , I2 , . . . , Ii , . . . , Ib ] and M = [M1 , M2 , . . . , Mi , . . . , Mb ],
respectively. Ii ∈ R 3×H ×W and Mi ∈ R 1×H ×W are the original
image and the corresponding semantic segmentation map,
respectively, and H and W separately denote the height and
width. According to these segmentation maps, we can obtain
the corresponding clothing regions from the original image;
moreover, all pixel values of these clothing regions can also
be obtained. To confuse these clothes, we build a shielding
pixel pool for each batch where all pixels of these clothing
regions are shuffled (the ellipse of Fig. 2 shows the pixel pool).
Note that in our VCS, we do not care about the pedestrian’s
upper clothes and pants, and all pixels are equally treated.
To reduce the negative effect of the clothes, we specifically
transform each pixel of the clothing regions in the original

image, and its value is replaced by another value randomly
obtained from the shielding pixel pool, but the values of other
pixels from nonclothing areas are kept the same as the original
image. Finally, the visual rendering image can be obtained,
and its results can be observed in the last row of Fig. 3. Then,
the visual rendering image is further fed into the Swin-T and
GAP to obtain the visual shielding feature Fs′ ∈ R 1,024 , where
the network architecture is the same as the backbone of the
SAVS, and the shared network parameters with the backbone
are employed. Moreover, to make the module focus on the
human nonclothing regions, such as the head, face, legs, and
feet, we also seek to ensure that the difference between the
visual shielding feature and the original feature is as small as
possible. In this way, it is difficult for the learning module to
differentiate the clothing regions and nonclothing areas, and
the clothing-irrelevant features can be extracted, where more
attention is devoted to the human semantic information.
C. Loss Function
The person ReID task is often regarded as a person classification problem; thus, the classification loss is often calculated.
To further improve the feature discrimination ability of the proposed method, a metric learning loss is added, and it is used to
narrow the intraclass distance and increase the interclass loss.
Finally, in the HSA module, the human semantic alignment
loss is also utilized. Thus, in total, the loss function of the
SAVS can be defined as follows:
L = λ1 Lid + λ2 Lcir + λ3 Lsem

(3)

where L is the total loss function of the SAVS, Lid denotes
the classification loss, Lcir represents the metric learning loss,
and Lsem is the human semantic alignment loss. λ1 , λ2 , and
λ3 are the trade-off parameters for balancing the contributions
of each term. In our experiments, each term is equally treated;
thus, all λ1 , λ2 , and λ3 values are set to 1. Specifically, for
the classification loss, the public cross-entropy loss is used
as the ID loss to learn discriminative features. To make the
feature more discriminative, the circle loss [52] in metric
learning is employed to measure the distance between sample
pairs. Given a single sample Ie in the feature space, let us
assume that there are K intraclass similarity scores and L
interclass similarity scores associated with Ie . Moreover, these
intraclass similarity scores and interclass similarity scores are
j
denoted as s ip (i = 1, 2, . . . , K ) and sn ( j = 1, 2, . . . , L),
respectively. To maximize the intraclass similarity s p and
minimize the interclass similarity, the metric learning loss can
be calculated by


K X
L
X

Lcir = log1 +
exp γ αnj snj − α ip s ip 
i=1 j=1

(



α ip = O p − s ip +


αnj = snj − On +

(4)

j

where αn and α ip are nonnegative weighting factors for
intraclass similarity scores and interclass similarity scores,
respectively. O p and On indicate the optimization score values
j
for s ip and sn , respectively. [∗]+ indicates the optimization
process. γ is the scale factor where the γ values are set
to 32 for all similarity scores in our experiments. When a
similarity score deviates far from its optimum (i.e., On for

Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
GAO et al.: SEMANTIC-AWARE ATTENTION AND VISUAL SHIELDING NETWORK

j

sn and O p for s ip ), it should obtain a large weighting factor
to obtain an effective update with the large gradient. In this
way, it can make the learned feature more discriminative to
distinguish between different people wearing similar clothing.
The clothing regions in the visual rendering image are
shielded, whose pixels are replaced by the shielding pixel
pool and are very different from those of the original image,
but the ID information is retained. Thus, to make the module
focus on the human nonclothing regions, such as the head,
face, legs, and feet, we also seek to ensure that the difference
between the visual shielding feature and the original feature
is as small as possible. In this way, the clothing-irrelevant
features can be extracted, where more attention is given to the
human semantic information. Therefore, the mean square error
between the visual shielding feature and the original feature
is utilized as the human semantic alignment loss, which can
be calculated by
b

Lsem =

1X
b i=1

Fo′ − Fs′ 2



(5)

where b is the batch size and ||∗||2 indicates the L 2 normalization. Fo′ ∈ R 1,024 and Fs′ ∈ R 1,024 denote the original feature
and the visual shielding feature, respectively. After optimization, the difference between Fo′ and Fs′ is very small, and these
features focus on the clothing-irrelevant regions of the human.
In this way, feature discrimination and generalization can be
further improved, and these features can effectively represent
people with different clothes.
IV. E XPERIMENTS AND D ISCUSSION
To evaluate the performance of our proposed SAVS framework, we perform experiments on four public cloth-changing
person ReID datasets: LTCC [30], PRCC [29], Celeb-reID
[27], and NKUP [42]. Since the cloth-changing person ReID
task is a new and challenging research topic, to the best of
our knowledge, at present, there are no comprehensive experiments with any cloth-changing ReID algorithms on all four
cloth-changing person ReID datasets, and this is the first work
that systematically and comprehensively assesses algorithm
performance in the context of these four cloth-changing person
ReID datasets. The remainder of this section is organized
as follows: 1) the competitors in our experiments are listed;
2) the implementation details are described; and 3) the performance evaluations and comparisons based on these four public
datasets are described.
A. Competitors
Since the cloth-changing person ReID task is a new and
challenging research topic, only a few works have been published, including SPT + ASE (TPAMI2021) [29], SE + CESD
(ACCV 2020) [30], ReIDCaps (TCSVT 2020) [27], Pixel
Sampling (ISPL 2021) [46], fine-grained shape-appearance
mutual learning framework (FSAM) (CVPR 2021) [44], CAL

codex
SAVS 的关键词是“屏蔽”而不是“增强”：它把换衣 ReID 从“找更强外观特征”改写成“主动让衣服线索失效，再保留非衣服的人体语义”。这类论文的说服力主要来自负信息处理是否真的对应问题定义。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -layout -f 8 -l 11 \"A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf\" - | sed -n '1,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
                      This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.


8                                                                                         IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS



                                                                                 TABLE I
                 P ERFORMANCE E VALUATION AND C OMPARISON BASED ON F OUR P UBLIC C LOTH -C HANGING P ERSON R E ID DATASETS ,
                                 W HERE THE B OLD VALUES I NDICATE THE B EST P ERFORMANCE IN E ACH C OLUMN




        but the corresponding performances of the baseline are                                SAVS cannot always obtain the best performance, but
        47.6% and 46.9%, and their corresponding improve-                                     its rank-1 can obtain the first place. The reason for
        ments can reach 10.0% (mAP) and 22.5% (rank-1).                                       this is that these specially designed methods mainly
        Similarly, the mAP and rank-1 accuracies of the SAVS                                  focus on body shape or contour sketch, and the complex
        approach on the LTCC dataset are 32.5% and 71.2%,                                     background and human semantic information are not
        respectively, but the corresponding mAP and rank-1 of                                 fully explored or are ignored, but in the SAVS, the
        the baseline can reach 15.2% and 42.6%, where the                                     HSA and VCS modules are designed to extract clothing-
        maximum improvements are 17.3% (mAP) and 28.6%                                        irrelevant features: more attention is given to the human
        (rank-1). Thus, the SAVS can significantly outperform                                 semantic information, and the negative influences of the
        the baseline. The reason for this is that the HSA and                                 background and the cloth-changing are reduced as much
        VCS modules are embedded into the Swin-T to extract                                   as possible. Thus, the SAVS experimentally exhibits
        the discriminative and robust features, and then the back-                            very good generalization ability, and these experimental
        bone, HSA, and VCS modules are jointly optimized. In                                  results prove the effectiveness and robustness of the
        this way, the human semantic information is fully used,                               SAVS approach.
        and the negative effect of clothing changes is reduced                             2) When comparing clothing-consistent person ReID meth-
        as much as possible. In addition, we also observe that                                ods with cloth-changing person ReID methods, the
        among these approaches, SE + CESD, SPT + ASE,                                         latter often achieve significantly better performance. For
        ReIDCaps, FSAM, Pixel Sampling, AIM, ACID, and                                        instance, the HOReID method outperforms the PCB,
        MBUNet are specially designed for the cloth-changing                                  MGN methods on the LTCC and PRCC datasets, while
        person ReID task, but their performance is still worse                                the MGN method performs best on the Celeb-reID and
        than that of the SAVS (with the exception of mAP on                                   NKUP datasets. Among the clothing-changing person
        the PRCC dataset). For example, ReIDCaps has obtained                                 ReID methods, the SAVS method stands out as the
        good performance (second place) on the Celeb-reID                                     most effective. Therefore, in the following analysis,
        dataset, where the mAP and rank-1 can reach 15.8%                                     we will compare SAVS with the MGN and HOR-
        and 63%, respectively, but when comparing it with the                                 eID methods using different datasets. When evaluat-
        SAVS, the improvements of the SAVS can achieve 5.5%                                   ing the LTCC dataset, SAVS demonstrates remarkable
        (mAP) and 2.9% (rank-1), respectively. Similarly, when                                improvements of 17.4% (mAP) and 24.9% (rank-1)
        the Pixel Sampling method and the LTCC dataset are                                    compared to HOReID. Similarly, on the Celeb-reID
        used, the mAP and rank-1 on the LTCC dataset are                                      dataset, SAVS exhibits a 10.5% improvement in mAP
        16.1% and 42.3%, and the improvements of the SAVS                                     and a 16.9% improvement in rank-1 compared to MGN.
        can attain 16.4% and 28.9%, respectively. We also find                                Similar conclusions can be drawn from the remain-
        that when the PRCC dataset is used, the mAP of the                                    ing two datasets. The reason behind these findings is

    Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.
                    This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.


GAO et al.: SEMANTIC-AWARE ATTENTION AND VISUAL SHIELDING NETWORK                                                                                                      9



     that the HOReID and MGN methods are specifically                                                                      TABLE II
     designed for clothing-consistent person ReID, assuming                                 E FFECTIVENESS OF THE HSA M ODULE W HERE F OUR P UBLIC
     that a person wears the same clothes within a short                                    C LOTH -C HANGING P ERSON R E ID DATASETS A RE E MPLOYED ,
                                                                                             AND THE B OLD VALUES I NDICATE THE B EST P ERFORMANCE
     time interval, resulting in similar visual appearances.                                    IN E ACH C OLUMN . N OTE T HAT O , F , AND B I NDICATE
     However, when clothing changes occur, the person’s                                         THE O RIGINAL I MAGE , THE F OREGROUND I MAGE , AND
     appearance varies significantly, leading to a dramatic                                           THE BACKGROUND I MAGE , R ESPECTIVELY
     decline in performance. In contrast, cloth-changing per-
     son ReID methods aim to disregard clothing-related
     clues and focus solely on visual semantic information
     that remains unaffected by changes in view or posture.
     As a result, these methods extract clothing-irrelevant
     features, which greatly enhance their performance com-
     pared to the appearance-relevant features extracted by
     clothing-consistent person ReID methods.
  3) ResNet50, DenseNet121, and Swin-T modules are
     widely used in many machine learning tasks, but they
     are also often assessed on the cloth-changing person
     ReID task. Although these modules can achieve good
     performances in many related tasks, when they are
     directly applied to the cloth-changing person ReID task,
     their performances are unsatisfactory and much worse
     than that of the SAVS. For example, when the PRCC
     dataset is used, the rank-1 accuracies of ResNet50,
     DenseNet121, Swin-T, and SAVS are 19.4%, 18.5%,
     46.9%, and 69.4%, respectively, and the correspond-
     ing improvements achieved by the SAVS method are
     50.0%, 50.9%, and 22.5%, respectively. Similarly, the
     mAP accuracies of ResNet50, DenseNet121, Swin-T,
     and SAVS on the Celeb-reID dataset are 5.8%, 2.9%,                               Fig. 4. Advantages of the HSA module, where four public cloth-changing
     11.1%, and 21.3%, respectively, and the corresponding                            datasets are utilized, and the evaluation metrics of (a) mAP and (b) rank-1
     improvements achieved by the SAVS method are 15.5%,                              are used. Note that the blue bar indicates the results of the baseline, and the
                                                                                      yellow bar denotes the improvements over the baseline when the HSA module
     18.4%, and 10.2%, respectively. The reason for this is                           is further used.
     that although these modules are widely used in different
     tasks, no cloth-changing characteristics are employed in
     these methods, but in the SAVS, the clothing-irrelevant                          the foreground image, and the background image, this basic
     features are extracted, and the negative influence of                            network is also used to extract the feature representations for
     the cloth-changing is reduced as much as possible. In                            them, and then these features are employed to find the persons
     addition, from these experiments, we can also observe                            from the gallery dataset. Finally, we name them “O,” “F,” and
     that the Swin-T can achieve the best performance among                           “B,” respectively. To further evaluate the importance of human
     the ResNet50, DenseNet121, and Swin-T modules no                                 semantic information, the human semantic information (the
     matter which dataset is used; thus, in our experiments,                          foreground image) is fed into the HSA module, and then its
     the Swin-T is also used as the backbone of the SAVS.                             outputs are used to reweight the original feature extracted by
                                                                                      the Swin-T and the original image (we name it “O + F”).
                    V. A BLATION S TUDY                                               Similarly, the background image is fed into the HSA module
   An ablation study is performed using the SAVS model to                             in place of the foreground image, and the reweight operation
analyze the contribution of each component. In this investiga-                        is also used for the original feature (we call this method
tion, four aspects are considered: 1) the effectiveness of the                        “O + B”). Their results are given in Table II and Fig. 4. From
HSA module; 2) the advantages of the VCS module; 3) a                                 them, we can make the following observations.
convergence analysis; and 4) a qualitative visualization. In the                         1) The original image can achieve the best performance
following, we discuss these four aspects separately.                                        when only the baseline is used, and the foreground
                                                                                            image and the background image can obtain sec-
                                                                                            ond and third places, respectively. For example, the mAP
A. Effectiveness of the HSA Module                                                          and rank-1 accuracies of the original image are 15.2%
   In the few existing cloth-changing person ReID methods,                                  and 42.6%, respectively, on the LTCC dataset, and
the global or local features are often extracted to represent the                           the corresponding accuracies of the foreground image
cloth-changing person, but in this section, we assess the effec-                            are 11.7% and 34.3%, but the corresponding accura-
tiveness of the HSA module on four public cloth-changing                                    cies of the background image are 10.2% and 14.7%,
person ReID datasets where the importance of human semantic                                 respectively; thus, the performance of the original image
information is discussed. Since the Swin-T is the backbone of                               can outperform the foreground and background images.
the SAVS, in our experiments, it is also used as the baseline,                              Similarly, when the Celeb-reID dataset is used, the rank-
where the Swin-T is used to extract features from the original                              1 accuracies of the original, foreground, and background
image, and then the Softmax is used as the classification                                   images are 46.4%, 40.0%, and 7.4% and the improve-
function. To assess the effectiveness of the original image,                                ment of “O” can reach 6.4% (“F”) and 39% (“B”),

  Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.
                       This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.


10                                                                                         IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS



        respectively. Although person ReID is a cross-scene                                                                  TABLE III
        retrieval task and we usually assume that the background                             B ENEFITS OF THE VCS, W HERE F OUR P UBLIC C LOTH -C HANGING
        information is useless, the experimental results show that                           DATASETS A RE U TILIZED . N OTICE T HAT THE S WIN -T I S U SED AS
                                                                                                   THE BASELINE , AND O UR P ROPOSED HSA AND VCS
        the performance of the foreground image is worse than                                               M ODULES A RE E MBEDDED I NTO
        that of the original image no matter which dataset is                                                 THE BASELINE S TEP BY S TEP
        used. For example, when the PRCC dataset is used,
        the mAP accuracies of the former and latter are 47.6%
        and 40.0%, respectively, whose improvement can reach
        7.6%. Thus, these experimental results prove that the
        foreground image is very useful, but the background
        information cannot be eliminated in its entirety.
     2) The performance of “O + F” can obtain a large improve-
        ment when compared with “O” or “F” regardless of the
        dataset used. For example, on the LTCC dataset, the
        mAP accuracies of “O + F,” “O,” and “F” are 24.4%,
        15.2%, and 11.7%, respectively, whose improvements
        can be 9.2% and 12.7%. When the PRCC dataset is
        used, the rank-1 accuracies of “O + F,” “O,” and “F”
        can reach 60.3%, 46.9%, and 45.2%, respectively, and
        the improvement of “O + F” can achieve 13.4% and
        15.1%. The reason for this is that in “O + F,” the
        HSA module is used to emphasize the importance of
        the human semantic information and reweight the visual
        feature map extracted by the original image. In this way,
        the negative effect of the background information can be
        reduced as much as possible, and more discriminative
        features can be obtained. In addition, we also observe
        that when the background information is fed into the
        HSA module, the performance of “O + B” is not stable
        on different datasets; for example, when the PRCC
        and NKUP datasets are used, the rank-1 accuracies of
        “O + B” are worse than those of “O,” but on other
        datasets, the rank-1 accuracies of “O + B” can obtain
        little improvement. From Fig. 4, we can also obtain the                          Fig. 5. Advantage analysis of the VCS module by using CMC curves on
        same conclusions. Thus, these experiments demonstrate                            the LTCC, PRCC, Celeb-reID and NKUP datasets.
        that the HSA module is very effective for enriching
        the feature representation, and the human semantic
        information is very helpful for feature extraction. In                                 5.4% (mAP) and 6.8% (rank-1). On the PRCC dataset,
        addition, the background information is somewhat useful                                the rank-1 accuracies of “+HSA + VCS” and “+HSA”
        for the feature representation, but emphasizing back-                                  are 60.3% and 69.4%, and the improvement can reach
        ground information cannot always obtain a performance                                  9.1%. We can draw similar conclusions from the other
        increase.                                                                              datasets. In addition, when the CMC curves are used
                                                                                               as the metric, we can also observe the same results
                                                                                               in Fig. 5. The reason why the VCS module can be
B. Advantages of the VCS Module                                                                successful is that visual clothing shielding makes it
    To validate the advantages of the VCS module, we perform                                   difficult for appearance features to be learned, and thus,
experiments on the four public cloth-changing person ReID                                      the model loses its reliance on clothing appearance when
datasets, and their results are given in Table III and Fig. 5. Note                            extracting features. Through the contrastive learning of
that in Table III, when the original image is fed into the Swin-                               the original and visual shielding features, the potential
T, the module is considered the baseline. Moreover, when the                                   consistency can be explored to effectively solve the
foreground image (human semantic) is further fed into the                                      cloth-changing problem.
HSA module and its results are combined with the backbone,                                  2) In the SAVS module, the original image information,
it is called “+HSA.” Finally, when the VCS module is further                                   visual shielding information, and human semantic infor-
used, the visual clothing shielding image is further embedded                                  mation are jointly learned in a unified framework. These
into the “+HSA” module; thus, we name it “+HSA + VCS.”                                         modules are complementary, and they can promote each
From them, we can see the following observations.                                              other. When the HSA module and the VCS module
    1) When the VCS module is used, the performance of                                         are embedded into the baseline step by step, their
       “+HSA + VCS” can be greatly improved over that                                          combined performance can yield a stable improvement.
       of the “+HSA” module. For example, when the Celeb-                                      For example, when the LTCC dataset is used, the
       reID dataset is used, the mAP and rank-1 accuracies                                     mAP (rank-1) accuracies of the baseline, “+HSA,” and
       of “+HSA + VCS” are 21.3% and 65.9%, and the                                            “+HSA + VCS” are 15.2% (42.6%), 24.4% (56%),
       mAP and rank-1 accuracies of “+HSA” are 15.9%                                           and 32.5% (71.2%), respectively, whose performance
       and 59.1%, respectively, whose improvements can reach                                   can be improved step by step, and the improvement

     Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.
                    This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.


GAO et al.: SEMANTIC-AWARE ATTENTION AND VISUAL SHIELDING NETWORK                                                                                                      11



     of the “+HSA + VCS” can achieve 17.3% (13.4%,                                                                        TABLE IV
     baseline) and 9.2% (28.6%, “+HSA”). Similarly, on the                                  A DVANTAGE OF THE VCS AND BACKBONE ON THE LTCC AND
     PRCC dataset, the corresponding improvements of the                                         C ELEB - RE ID DATASETS W HEN C OMPARED W ITH
                                                                                                           THE P IXEL S AMPLING M ETHOD
     “+HSA + VCS” module can reach 6.4% (13.4%, base-
     line) and 10% (23.5%, “+HSA”), respectively.
  3) In the SAVS and Pixel Sampling, both focus on the
     surface features of clothes; thus, we also compare
     their performance on two widely used public datasets,
     including LTCC and Celeb-reID. Since ResNet-50 is the
     backbone of Pixel Sampling, to obtain a fair comparison,
     ResNet-50 is also used as the backbone in the SAVS,
     but the human semantic information is ignored, and we
     call this approach “ResNet50 + VCS.” Along another
     line, to assess the advantages of the backbone, the
     backbone of the SAVS is replaced with the Swin-T
     based on the “ResNet50 + VCS,” and we name this
     approach “Swin-T + VCS.” Finally, the HSA module is
     further assessed, and the human semantic information is
     embedded into “Swin-T + VCS,” whose name is “Swin-
     T + VCS + HSA.” Their results are shown in Table IV.
     From this, we can observe that when the same backbone
     and training strategy are used in the Pixel Sampling and
     the “ResNet50 + VCS,” the performance of the latter
     is much better than that of the former. For example, the
     mAP (rank-1) accuracies of the “ResNet50 + VCS” and
     Pixel Sampling on the LTCC dataset are 18.7% (47.7%)
     and 16.1% (42.3%), respectively, whose improvement
     can reach 2.6% (5.4%). We can obtain the same
     results from the Celeb-reID dataset. The reason why
     the “ResNet50 + VCS” is much better than the Pixel                               Fig. 6. Convergence curves of the SAVS method for the LTCC, PRCC,
     Sampling is that the latter only focuses on a single item                        Celeb-reID, and NKUP datasets.
     of clothing, but all clothing regions of the person are
     fully considered in the former. In this way, the nega-
     tive effect of cloth-changing can be reduced as much                             stable regardless of the dataset utilized. Thus, this can further
     as possible, and a more robust feature representation                            prove the effectiveness of the SAVS method.
     can be extracted by covering the clothing regions and
     focusing the model on the visual semantic information                            D. Qualitative Visualization
     unrelated to the clothes. Thus, these experiments can                               To further prove the effectiveness and robustness of the
     further prove that the VCS module is very effective                              SAVS, in this section, we visualize some results of the
     for solving the cloth-changing issue. In addition, when                          proposed SAVS on the different datasets. In this investigation,
     the Swin-T is further used, we can see that the perfor-                          three aspects are considered: 1) visualization of the attention
     mance can be further improved. For example, on the                               maps; 2) visualization of the similarity map; and 3) quali-
     LTCC dataset, the mAP (rank-1) accuracies of “Swin-                              tative visualization of the retrieval results. In the following,
     T + VCS” and Pixel Sampling are 22.8% (54.9%)                                    we discuss these three aspects separately, and their results are
     and 16.1% (42.3%), respectively, whose improvement                               given in Figs. 7–10. From them, we can make the following
     can reach 6.7% (12.6%). Thus, it can be proven that                              observations.
     the Swin-T is also very efficient, and in our following                             1) To further illustrate which part of the focused features
     experiments, it is utilized as the backbone of the SAVS.                                are learned with the help of different modules in the
     Finally, when the HSA module is further employed, its                                   SAVS, the attention maps are visualized and displayed
     performance can significantly outperform the baseline,                                  in Fig. 7. In Fig. 7, the first row indicates the orig-
     the Pixel Sampling, “ResNet50 + VCS,” and “Swin-                                        inal images from different datasets. The second row
     T + VCS” regardless of datasets.                                                        (“+HSA”) represents which channels on the feature map
                                                                                             are activated by the reweighting operation where the
                                                                                             Swin-T and the HSA module are used. The third row
C. Convergence Analysis                                                                      (“+HSA + VCS”) denotes which channels between the
  In this section, we evaluate the convergence of the proposed                               original features and visual shielding features are more
SAVS method on four public cloth-changing person ReID                                        consistently constrained by the semantic alignment loss
datasets, including LTCC, PRCC, Celeb-reID, and NKUP, and                                    and the VCS module, where the Swin-T, the HSA, and
their convergence curves are shown in Fig. 6. From Fig. 6,                                   VCS modules are utilized. From them, we can see that
we can observe that the convergence speeds of the SAVS                                       in the second row, the HSA module is utilized; thus,
method are very fast no matter which dataset is utilized.                                    only the human semantic channels are activated where
Moreover, in the optimization process, only 30–40 epochs are                                 the activated regions focus on the human body, but
required for all datasets, and the convergence curves can be                                 the background information is largely ignored. Thus,

  Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.

exec
/bin/zsh -lc "pdftotext \"A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf\" - | rg -n -C 3 \"Ablation|ablation|HSA|VCS|shield|semantic|visual|Grad|t-SNE|feature distribution|Fig\\. 8|Fig\\. 9\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
15-different clothes exhibits large variations, it is very difficult
16-for existing approaches to extract discriminative and robust
17-feature representations. Current works mainly focus on body
18:shape or contour sketches, but the human semantic information
19-and the potential consistency of pedestrian features before and
20-after changing clothes are not fully explored or are ignored. To
21:solve these issues, in this work, a novel semantic-aware attention
22:and visual shielding network for cloth-changing person ReID
23:(abbreviated as SAVS) is proposed where the key idea is to shield
24:clues related to the appearance of clothes and only focus on visual
25:semantic information that is not sensitive to view/posture changes.
26:Specifically, a visual semantic encoder is first employed to locate
27:the human body and clothing regions based on human semantic
28:segmentation information. Then, a human semantic attention
29:(HSA) module is proposed to highlight the human semantic
30:information and reweight the visual feature map. In addition,
31:a visual clothes shielding (VCS) module is also designed to extract
32-a more robust feature representation for the cloth-changing task
33-by covering the clothing regions and focusing the model on
34:the visual semantic information unrelated to the clothes. Most
35-importantly, these two modules are jointly explored in an endto-end unified framework. Extensive experiments demonstrate
36-Manuscript received 17 July 2022; revised 9 October 2023; accepted
37-29 October 2023. This work was supported in part by the National Natural
--
72-the PRCC, LTCC, Celeb, and NKUP datasets in terms of rank-1
73-(mAP), respectively.
74-Index Terms— Cloth-changing person re-identification (ReID),
75:human semantic attention (HSA), semantic-aware, visual clothes
76:shielding (VCS).
77-
78-I. I NTRODUCTION
79-
--
103-existing person ReID approaches. We can find that these
104-approaches are very effective for the person ReID task with
105-short time spans where the human appearance features are
106:fully used for visual matching, but when the surveillance
107-acquisition period becomes longer, the complexity of clothing
108-changes subsequently increases. Fig. 1 shows some examples
109-of cloth-changing person ReID images, where each row displays the images of the same person wearing different clothes.
110:From them, we can observe that the differences in the visual
111-appearances of the same person with different clothes are
112-
113-2162-237X © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
--
129-existing person ReID approaches are directly applied in this
130-case, their performance deteriorates dramatically, and they
131-often fail. The reason for this is that the above approaches
132:mainly rely on visual clothing appearance, and they cannot
133-provide a robust feature representation for each identity (ID)
134-with different clothes; thus, discriminative and robust feature
135-extraction is an urgent issue for the cloth-changing person
--
155-very good for trying to solve the cloth-changing person ReID
156-problem, but since the human appearance exhibits large variations with different clothes, it is very difficult for existing
157-approaches to extract discriminative and robust feature representations. Moreover, current works mainly focus on body
158:shape or contour sketches, but the human semantic information
159-and the potential consistency of pedestrian features before and
160-after changing clothes are not fully explored or are ignored.
161-
--
167-ReID: 1) the same person wearing different clothes and
168-2) different people wearing the same clothes. The SAVS
169-method has already eliminated both effects from appearance
170:simultaneously by attentional weighting and shielding pixels.
171-Extensive experimental results on four cloth-changing person
172-ReID datasets demonstrate that SAVS can outperform stateof-the-art person ReID approaches, and more discriminative
173-and robust features can be obtained that can effectively solve
174-the cloth-changing issue. The main contributions of this article
175-are summarized as follows.
176-1) We develop a novel end-to-end SAVS network for
177:cloth-changing person ReID that consists of visual
178:semantic encoding and visual semantic decoding. The
179:key idea is to shield clues related to the appearance of
180:clothes and only focus on visual semantic information
181-that is not sensitive to view/posture changes. In this way,
182-the negative effect of the clothing information can be
183-reduced as much as possible.
184:2) We design a human semantic attention (HSA) module
185-to highlight the human information and reweight the
186:visual feature map that is very helpful for obtaining
187-more discriminative features, and then we develop a
188:visual clothes shielding (VCS) module to extract a more
189-robust feature representation by focusing the model
190:on the visual semantic information unrelated to the
191-clothes. Most importantly, these two modules are jointly
192-explored in an end-to-end unified framework. In this
193-way, more discriminative and robust features can be
--
203-Section II introduces the related work, and Section III
204-describes the proposed SAVS method. Section IV describes the
205-experimental settings and the analysis of the results. Section V
206:presents the details of the ablation study, and concluding
207-remarks are presented in Section VI.
208-II. R ELATED W ORK
209-To date, many person ReID approaches have been proposed.
210:According to the person’s visual appearance, these methods
211-can be roughly divided into clothing-consistent person ReID
212-and cloth-changing person ReID. In the following, we will
213-separately introduce them.
214-A. Clothing-Consistent Person ReID
215-In earlier times, people [31], [37], [38] made efforts to
216-develop related methods for clothing-consistent person ReID
217:where the visual appearance of the clothes was consistent
218-for the same person. For example, Sun et al. [9] proposed
219-
220-Authorized licensed use limited to: Tianjin University of Technology. Downloaded on November 11,2023 at 11:52:25 UTC from IEEE Xplore. Restrictions apply.
--
249-points of the human body in pedestrian images, and then
250-these keypoints were used to decide whether a specific body
251-part was occluded. Gao et al. [5] proposed a novel texture
252:semantic alignment (TSA) approach with visibility awareness
253-for the partial person ReID task, where the occlusion issue
254-and changes in poses were simultaneously explored in an endto-end unified framework. Zheng et al. [41] proposed for the
255-first time to solve the person ReID problem in 3-D space by
--
278-models heavily rely on the consistency of the appearance of
279-clothes.
280-B. Cloth-Changing Person ReID
281:Since the visual appearance of the pedestrians in the
282-cloth-changing person ReID task changes dramatically after
283-
284-3
--
297-information was used to substitute for human color information. Moreover, an SPT layer was designed to transform the
298-contour sketch image, and then a multistream network was
299-used to aggregate multiple granular features to better discriminate people by changing the sampling range of the SPT layer.
300:In this way, the changes in visual appearance caused by clothing changes could be reduced. Qian et al. [30] proposed an
301-SE + CESD module, where the main idea was to completely
302-delete information related to the appearance of clothes and
303-only focus on body shape information that is not sensitive to
--
321-supplied; thus, in the query, both the clothing template image
322-and an image of the target person wearing other clothes were
323-fed into the module to find the target image. Gao et al. [45]
324:proposed a novel multigranular visual-semantic embedding
325-algorithm (MVSE) for cloth-changing person ReID, where
326:visual semantic information and human attributes are embedded into the network. Hong et al. [44] proposed a fine-grained
327-shape-appearance mutual learning framework that can learn
328-fine-grained discriminative body shape knowledge in a shaped
329-stream and transfer it to an appearance stream to complement
330-the clothing-unrelated knowledge in the appearance features.
331:Shu et al. [46] proposed a semantic-guided pixel sampling
332-approach for the cloth-changing person ReID task which
333-forces the model to automatically learn clothing-irrelevant
334-cues that are irrelevant to upper clothes and pants. Gu et al.
--
348-
349-IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
350-
351:Fig. 2. Pipeline of the SAVS approach is an end-to-end network architecture, and it mainly consists of the visual semantic encoder, the visual semantic
352:decoder, and the loss function. In the visual semantic encoder, the foreground information and visual clothing shielding information are generated for each ID
353:to enrich the image representations. The visual semantic decoder focuses the module on human semantic information and visual clothing shielding information;
354-thus, a more discriminative and robust feature can be extracted. Note that in the SAVS, Swin-T is used as the backbone. GAP indicates the global average
355-pooling, and FC denotes the fully connected layers.
356-
--
361-different clothes exhibits large variations, it is very difficult for
362-existing approaches to extract discriminative and robust feature
363-representations, and their performances need to be further
364:improved. Moreover, the human semantic information and the
365-potential consistency of pedestrian features before and after
366-changing clothes are not fully explored or are ignored. Thus,
367:in this work, we fully explore the available visual semantic
368-information and the potential consistency of features and then
369-extract a generalized and robust feature to represent a person
370-wearing different clothes.
371-
372-body shape, posture, accompanying markers, and other comprehensive information. In other words, the idea of extracting
373:human semantic information is to first eliminate the influence
374:of background changes through HSA to highlight all human
375-features, and then eliminate the influence of cloth-changing
376:through VCS to highlight human features unrelated to clothes,
377:i.e., effective human semantic information for cloth-changing
378:person ReID. Notice that “Human semantic information” is
379-summarized as biometric cues that are helpful in identifying
380:people. In the following, we will introduce the visual semantic
381:encoder, the visual semantic decoder, and the loss function.
382-Algorithm 1 shows the complete procedure of the proposed
383-SAVS model.
384-
385-III. P ROPOSED A PPROACH
386-As shown in Fig. 2, our proposed SAVS method consists of
387:two stages: a visual semantic encoder and a visual semantic decoder. Moreover, the visual semantic decoder mainly
388:consists of the backbone, the HSA module, and the VCS
389-module, where these three modules are jointly explored in an
390-end-to-end unified framework. In addition, the loss function
391-is used to guide the network optimization. Specifically, the
392-foreground image of the human body (with background pixel
393:values of 0), the original image, and the shielding image are
394:fed into the HSA, the backbone, and VCS streams, to obtain
395:the enhanced feature, the original feature, and the shielding
396-feature, respectively, and then the loss functions are further
397-utilized to mine their relationship among these features. Since
398-the clothes information is included in the enhanced feature
399-and the original feature, thus, we invalidate the clothing
400:information by semantic loss through the VCS stream, at this
401:point the only human semantic information left is the cues
402-that are not related to the clothing, such as facial features,
403-
404-A. Visual Semantic Encoder
405-With the development of deep learning techniques [48],
406-[49], researchers have designed different CNNs for the person
407:ReID task, but since visual appearances are very different
408-in the cloth-changing person ReID task, it is very difficult for an individual feature to extract a generalized and
409-robust feature to represent a person with different clothes.
410-To accommodate clothing variations within the limited data,
411-enriched feature representations for each ID are needed.
412:Thus, in the visual semantic encoder, the foreground image
413:and the shielding image are generated with the help of
414:human semantic segmentation maps. Specifically, for the
415-original image, the pretrained self-correction for human parsing (SCHP) module [50] is employed to obtain the human
416:semantic segmentation information, where the human body
417:is divided into 18 semantic parts. To make it suitable for
418:the cloth-changing person ReID task, these 18 semantic parts
419-are recombined to obtain seven parts, including background,
420-head, torso, pants, arms, legs, and belongings. Fig. 3 shows
421-
--
428-
429-Algorithm 1 Training Procedure of the SAVS
430-Input: Input image I ∈ R 3×H ×W
431:The corresponding semantic segmentation map M ∈ R 1×H ×W
432-epochs = T , t = 0, the learning rate = 3.5e−3 ,
433-the shrink rate = 0.1 at 40 to 60
434-initialize the parameters of the swim-transformer(backbone)
435-Output: the SAVS parameters θ
436-while t < T do
437:1: Extract the foreground image and the shielding image from I
438:based on the semantics of the segmentation map M
439-2: Obtain the original feature map Fo ∈ R 7×7×1,024
440-from I by Swin-Transformer (Swin-T)
441-3: Obtain the original feature Fo′ ∈ R 1,024
442-by using global average pooling
443-4: Obtain the enhanced feature Fe′ ∈ R 1,024
444:by using equation (1) and (2) in HSA module
445:5: Obtain the shielding feature Fs′ ∈ R 1,024 in VCS module
446:6: Compute the semantic loss by using equation (5)
447-to minimize the distance between Fo′ and Fs′
448-7: Compute the cross-entropy classification loss
449-and the circle loss using equation (4) based on Fe′
--
451-and update the SAVS parameters θ
452-end while
453-
454:the results of the human semantic segmentation maps.
455-In the following, we provide detailed information on how
456:the foreground image and the visual shielding image can be
457-obtained.
458-1) Foreground Image: Many previous works focus on
459-obtaining the global or local features for each cloth-changing
--
462-information, and the foreground image is considered another
463-image representation of the original image. Thus, the key step
464-is to separate the foreground and background for each original
465:image. Based on the human semantic segmentation maps,
466-we perform a binarization process to distinguish between
467-background and nonbackground. All information other than
468-the background is used as the foreground information, such as
--
475-2) Visual Shielding Image: In the cloth-changing person
476-ReID scenario, the most common change occurs for the upper
477-clothes and pants. To obtain a more discriminative feature for
478:cloth-changing tasks, the visual shielding image is obtained
479-as another new image representation of the original image
480-by covering the clothing regions of the upper clothes and
481-pants. Specifically, unlike separating foreground information,
482-at this step, finer-grained segmentation labels are required
483-to accurately find the local locations of the upper clothes
484:and pants based on the aforementioned semantic segmentation
485:information. In this way, the shielding mask can be obtained
486-where the pixel value is set to one if it belongs to the mask,
487:or its value is set to zero. Finally, we further combine the original image and the shielding mask by matrix multiplication;
488:thus, the visual shielding image can be obtained. Fig. 3 also
489:displays the results of the visual shielding images.
490-
491-Fig. 3.
492:Results of the visual semantic encoder process. From top to
493:bottom: Original images, the corresponding semantic segmentation maps,
494:the foreground images, the visual shielding images, and the visual rendering
495-images.
496-
497-B. Visual Semantic Decoder
498:The visual semantic decoder mainly consists of the backbone, the HSA module, and the VCS module, where these
499-three modules are jointly explored in an end-to-end unified framework. In our experiments, the Swin Transformer
500-(Swin-T) [51] is used as the backbone to obtain the original
501-feature map Fo ∈ R 7×7×1,024 , whose input is the original
--
505-patch size to 4 for patch partitioning, and we set the window
506-size to 7 for each patch to compute the self-attention inside
507-each window. We keep the transformer blocks from stage 1
508:to stage 4 to extract visual features and discard the final
509-classification layer of the network. Moreover, the parameters of
510-the Swin-T pretrained on the ImageNet dataset are further used
511-in the following joint optimization. In the following, we will
512:introduce the HSA and VCS modules separately.
513-1) Human Semantic Attention Module: Current works
514-mainly focus on body shape or contour sketches, but human
515:semantic information has not been fully explored. Thus,
516:in this work, an HSA module is designed to highlight the
517:human semantic information and reweight the visual feature
518-map, and the latent correlation between the channels of the
519-convolved features is mined. In this way, the negative effect
520-of the background information can be reduced as much as
--
527-
528-IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
529-
530:The structure of the HSA module is shown in Fig. 2. In this
531-module, ResNet50 is used as the basic network architecture,
532-and it is first pretrained on the ImageNet dataset. Then, the
533-parameters of ResNet50 are further jointly optimized with
534:other networks. Specifically, in the HSA module, the feature
535-map FA ∈ R 7×7×1,024 is first obtained by ResNet50, and then
536-it can be further fed into the GAP to obtain the feature vector
537-FA′ ∈ R 1024 . Moreover, the feature vector FA′ is subsequently
--
545-Fw = σ W2 · δ W1 · f g (FA ) .
546-(1)
547-Fw ∈ R 1024 denotes the weighted feature vector with human
548:semantic information whose input image is the foreground
549-image. f g indicates the operation of GAP. W1 and W2 represent
550-the parameters of two fully connected layers. σ and δ indicate
551-the sigmoid activation and the rectified linear unit (ReLU)
552-function, respectively. To make the original feature highlight
553:the human semantic information, the reweight operation is
554-further used for the original feature map Fo ∈ R 7×7×1,024
555-obtained from the original image and the backbone (Swin-T)
556-by the weighted feature vector Fw , and it can be obtained by
--
565-is the output of the reweight operation and GAP. Since Fw
566-is obtained from the foreground image that can selectively
567-emphasize the human feature channels, Fe′ can pay more
568:attention to the human semantic information, and the negative
569-effect of the background information can be reduced as much
570-as possible. Therefore, the extracted feature is more discriminative and robust.
571-2) Visual Clothes Shielding Module: In the cloth-changing
--
573-thus, it will be very difficult to obtain more discriminative
574-features for cloth-changing tasks. We hope that we can extract
575-clothing-irrelevant features where more attention is given
576:to human semantic information; thus, the VCS module is
577:designed, which focuses the model on visual semantic information unrelated to clothes. Specifically, suppose that there are
578-b original images and b corresponding segmentation maps in
579-each batch of the training stage, which can be denoted as I =
580-[I1 , I2 , . . . , Ii , . . . , Ib ] and M = [M1 , M2 , . . . , Mi , . . . , Mb ],
581-respectively. Ii ∈ R 3×H ×W and Mi ∈ R 1×H ×W are the original
582:image and the corresponding semantic segmentation map,
583-respectively, and H and W separately denote the height and
584-width. According to these segmentation maps, we can obtain
585-the corresponding clothing regions from the original image;
586-moreover, all pixel values of these clothing regions can also
587:be obtained. To confuse these clothes, we build a shielding
588-pixel pool for each batch where all pixels of these clothing
589-regions are shuffled (the ellipse of Fig. 2 shows the pixel pool).
590:Note that in our VCS, we do not care about the pedestrian’s
591-upper clothes and pants, and all pixels are equally treated.
592-To reduce the negative effect of the clothes, we specifically
593-transform each pixel of the clothing regions in the original
594-
595-image, and its value is replaced by another value randomly
596:obtained from the shielding pixel pool, but the values of other
597-pixels from nonclothing areas are kept the same as the original
598:image. Finally, the visual rendering image can be obtained,
599-and its results can be observed in the last row of Fig. 3. Then,
600:the visual rendering image is further fed into the Swin-T and
601:GAP to obtain the visual shielding feature Fs′ ∈ R 1,024 , where
602-the network architecture is the same as the backbone of the
603-SAVS, and the shared network parameters with the backbone
604-are employed. Moreover, to make the module focus on the
605-human nonclothing regions, such as the head, face, legs, and
606-feet, we also seek to ensure that the difference between the
607:visual shielding feature and the original feature is as small as
608-possible. In this way, it is difficult for the learning module to
609-differentiate the clothing regions and nonclothing areas, and
610-the clothing-irrelevant features can be extracted, where more
611:attention is devoted to the human semantic information.
612-C. Loss Function
613-The person ReID task is often regarded as a person classification problem; thus, the classification loss is often calculated.
614-To further improve the feature discrimination ability of the proposed method, a metric learning loss is added, and it is used to
615-narrow the intraclass distance and increase the interclass loss.
616:Finally, in the HSA module, the human semantic alignment
617-loss is also utilized. Thus, in total, the loss function of the
618-SAVS can be defined as follows:
619-L = λ1 Lid + λ2 Lcir + λ3 Lsem
--
622-
623-where L is the total loss function of the SAVS, Lid denotes
624-the classification loss, Lcir represents the metric learning loss,
625:and Lsem is the human semantic alignment loss. λ1 , λ2 , and
626-λ3 are the trade-off parameters for balancing the contributions
627-of each term. In our experiments, each term is equally treated;
628-thus, all λ1 , λ2 , and λ3 values are set to 1. Specifically, for
--
682-to obtain an effective update with the large gradient. In this
683-way, it can make the learned feature more discriminative to
684-distinguish between different people wearing similar clothing.
685:The clothing regions in the visual rendering image are
686:shielded, whose pixels are replaced by the shielding pixel
687-pool and are very different from those of the original image,
688-but the ID information is retained. Thus, to make the module
689-focus on the human nonclothing regions, such as the head,
690-face, legs, and feet, we also seek to ensure that the difference
691:between the visual shielding feature and the original feature
692-is as small as possible. In this way, the clothing-irrelevant
693-features can be extracted, where more attention is given to the
694:human semantic information. Therefore, the mean square error
695:between the visual shielding feature and the original feature
696:is utilized as the human semantic alignment loss, which can
697-be calculated by
698-b
699-
--
709-(5)
710-
711-where b is the batch size and ||∗||2 indicates the L 2 normalization. Fo′ ∈ R 1,024 and Fs′ ∈ R 1,024 denote the original feature
712:and the visual shielding feature, respectively. After optimization, the difference between Fo′ and Fs′ is very small, and these
713-features focus on the clothing-irrelevant regions of the human.
714-In this way, feature discrimination and generalization can be
715-further improved, and these features can effectively represent
--
748-B. Implementation Details
749-Since the backbone of the SAVS approach is the Swin-T,
750-it is also used as the baseline in our experiments. Note that
751:the VCS module is only used in the training stage to jointly
752-optimize the network parameters of the backbone, and in
753:the test stage, only the backbone and the HSA module are
754-used to extract the feature representation where the backbone
755-only focuses on the clothing-irrelevant regions of the human.
756-Finally, only the enhanced feature is used to describe each
757-person in the query, where the original feature is used to
758-reweight the enhanced feature. Specifically, an RGB image
759-is first fed into the SCHP [50] module to obtain the human
760:semantic information, the foreground image, and the shielding
761:image, respectively, and then the original image, the foreground image, and the shielding image are further fed into
762:the modules of the visual semantic decoder. Note that the
763:enhanced feature, the original feature, and the shielding feature
764-can be obtained in the SAVS, but only the enhanced feature
765-is used in the query. Moreover, as the module focuses on
766:the human semantic information and visual clothing shielding
767-information, thus, the enhanced feature is more discriminative
768-and robust. In addition, the default settings and divisions of
769-these datasets [27], [29], [30], [42] are used. In our experiments, the Swin-T is first pretrained on the ImageNet dataset,
--
821-the baseline can reach 15.2% and 42.6%, where the
822-maximum improvements are 17.3% (mAP) and 28.6%
823-(rank-1). Thus, the SAVS can significantly outperform
824:the baseline. The reason for this is that the HSA and
825:VCS modules are embedded into the Swin-T to extract
826:the discriminative and robust features, and then the backbone, HSA, and VCS modules are jointly optimized. In
827:this way, the human semantic information is fully used,
828-and the negative effect of clothing changes is reduced
829-as much as possible. In addition, we also observe that
830-among these approaches, SE + CESD, SPT + ASE,
--
848-its rank-1 can obtain the first place. The reason for
849-this is that these specially designed methods mainly
850-focus on body shape or contour sketch, and the complex
851:background and human semantic information are not
852-fully explored or are ignored, but in the SAVS, the
853:HSA and VCS modules are designed to extract clothingirrelevant features: more attention is given to the human
854:semantic information, and the negative influences of the
855-background and the cloth-changing are reduced as much
856-as possible. Thus, the SAVS experimentally exhibits
857-very good generalization ability, and these experimental
--
880-that the HOReID and MGN methods are specifically
881-designed for clothing-consistent person ReID, assuming
882-that a person wears the same clothes within a short
883:time interval, resulting in similar visual appearances.
884-However, when clothing changes occur, the person’s
885-appearance varies significantly, leading to a dramatic
886-decline in performance. In contrast, cloth-changing person ReID methods aim to disregard clothing-related
887:clues and focus solely on visual semantic information
888-that remains unaffected by changes in view or posture.
889-As a result, these methods extract clothing-irrelevant
890-features, which greatly enhance their performance compared to the appearance-relevant features extracted by
--
917-matter which dataset is used; thus, in our experiments,
918-the Swin-T is also used as the backbone of the SAVS.
919-V. A BLATION S TUDY
920:An ablation study is performed using the SAVS model to
921-analyze the contribution of each component. In this investigation, four aspects are considered: 1) the effectiveness of the
922:HSA module; 2) the advantages of the VCS module; 3) a
923:convergence analysis; and 4) a qualitative visualization. In the
924-following, we discuss these four aspects separately.
925:A. Effectiveness of the HSA Module
926-In the few existing cloth-changing person ReID methods,
927-the global or local features are often extracted to represent the
928:cloth-changing person, but in this section, we assess the effectiveness of the HSA module on four public cloth-changing
929:person ReID datasets where the importance of human semantic
930-information is discussed. Since the Swin-T is the backbone of
931-the SAVS, in our experiments, it is also used as the baseline,
932-where the Swin-T is used to extract features from the original
--
936-9
937-
938-TABLE II
939:E FFECTIVENESS OF THE HSA M ODULE W HERE F OUR P UBLIC
940-C LOTH -C HANGING P ERSON R E ID DATASETS A RE E MPLOYED ,
941-AND THE B OLD VALUES I NDICATE THE B EST P ERFORMANCE
942-IN E ACH C OLUMN . N OTE T HAT O , F , AND B I NDICATE
943-THE O RIGINAL I MAGE , THE F OREGROUND I MAGE , AND
944-THE BACKGROUND I MAGE , R ESPECTIVELY
945-
946:Fig. 4. Advantages of the HSA module, where four public cloth-changing
947-datasets are utilized, and the evaluation metrics of (a) mAP and (b) rank-1
948-are used. Note that the blue bar indicates the results of the baseline, and the
949:yellow bar denotes the improvements over the baseline when the HSA module
950-is further used.
951-
952-the foreground image, and the background image, this basic
--
954-them, and then these features are employed to find the persons
955-from the gallery dataset. Finally, we name them “O,” “F,” and
956-“B,” respectively. To further evaluate the importance of human
957:semantic information, the human semantic information (the
958:foreground image) is fed into the HSA module, and then its
959-outputs are used to reweight the original feature extracted by
960-the Swin-T and the original image (we name it “O + F”).
961:Similarly, the background image is fed into the HSA module
962-in place of the foreground image, and the reweight operation
963-is also used for the original feature (we call this method
964-“O + B”). Their results are given in Table II and Fig. 4. From
--
1002-can reach 60.3%, 46.9%, and 45.2%, respectively, and
1003-the improvement of “O + F” can achieve 13.4% and
1004-15.1%. The reason for this is that in “O + F,” the
1005:HSA module is used to emphasize the importance of
1006:the human semantic information and reweight the visual
1007-feature map extracted by the original image. In this way,
1008-the negative effect of the background information can be
1009-reduced as much as possible, and more discriminative
1010-features can be obtained. In addition, we also observe
1011-that when the background information is fed into the
1012:HSA module, the performance of “O + B” is not stable
1013-on different datasets; for example, when the PRCC
1014-and NKUP datasets are used, the rank-1 accuracies of
1015-“O + B” are worse than those of “O,” but on other
1016-datasets, the rank-1 accuracies of “O + B” can obtain
1017-little improvement. From Fig. 4, we can also obtain the
1018-same conclusions. Thus, these experiments demonstrate
1019:that the HSA module is very effective for enriching
1020:the feature representation, and the human semantic
1021-information is very helpful for feature extraction. In
1022-addition, the background information is somewhat useful
1023-for the feature representation, but emphasizing background information cannot always obtain a performance
1024-increase.
1025:B. Advantages of the VCS Module
1026:To validate the advantages of the VCS module, we perform
1027-experiments on the four public cloth-changing person ReID
1028-datasets, and their results are given in Table III and Fig. 5. Note
1029-that in Table III, when the original image is fed into the SwinT, the module is considered the baseline. Moreover, when the
1030:foreground image (human semantic) is further fed into the
1031:HSA module and its results are combined with the backbone,
1032:it is called “+HSA.” Finally, when the VCS module is further
1033:used, the visual clothing shielding image is further embedded
1034:into the “+HSA” module; thus, we name it “+HSA + VCS.”
1035-From them, we can see the following observations.
1036:1) When the VCS module is used, the performance of
1037:“+HSA + VCS” can be greatly improved over that
1038:of the “+HSA” module. For example, when the CelebreID dataset is used, the mAP and rank-1 accuracies
1039:of “+HSA + VCS” are 21.3% and 65.9%, and the
1040:mAP and rank-1 accuracies of “+HSA” are 15.9%
1041-and 59.1%, respectively, whose improvements can reach
1042-
1043-TABLE III
1044:B ENEFITS OF THE VCS, W HERE F OUR P UBLIC C LOTH -C HANGING
1045-DATASETS A RE U TILIZED . N OTICE T HAT THE S WIN -T I S U SED AS
1046:THE BASELINE , AND O UR P ROPOSED HSA AND VCS
1047-M ODULES A RE E MBEDDED I NTO
1048-THE BASELINE S TEP BY S TEP
1049-
1050:Fig. 5. Advantage analysis of the VCS module by using CMC curves on
1051-the LTCC, PRCC, Celeb-reID and NKUP datasets.
1052-
1053-5.4% (mAP) and 6.8% (rank-1). On the PRCC dataset,
1054:the rank-1 accuracies of “+HSA + VCS” and “+HSA”
1055-are 60.3% and 69.4%, and the improvement can reach
1056-9.1%. We can draw similar conclusions from the other
1057-datasets. In addition, when the CMC curves are used
1058-as the metric, we can also observe the same results
1059:in Fig. 5. The reason why the VCS module can be
1060:successful is that visual clothing shielding makes it
1061-difficult for appearance features to be learned, and thus,
1062-the model loses its reliance on clothing appearance when
1063-extracting features. Through the contrastive learning of
1064:the original and visual shielding features, the potential
1065-consistency can be explored to effectively solve the
1066-cloth-changing problem.
1067-2) In the SAVS module, the original image information,
1068:visual shielding information, and human semantic information are jointly learned in a unified framework. These
1069-modules are complementary, and they can promote each
1070:other. When the HSA module and the VCS module
1071-are embedded into the baseline step by step, their
1072-combined performance can yield a stable improvement.
1073-For example, when the LTCC dataset is used, the
1074:mAP (rank-1) accuracies of the baseline, “+HSA,” and
1075:“+HSA + VCS” are 15.2% (42.6%), 24.4% (56%),
1076-and 32.5% (71.2%), respectively, whose performance
1077-can be improved step by step, and the improvement
1078-
--
1081-This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
1082-GAO et al.: SEMANTIC-AWARE ATTENTION AND VISUAL SHIELDING NETWORK
1083-
1084:of the “+HSA + VCS” can achieve 17.3% (13.4%,
1085:baseline) and 9.2% (28.6%, “+HSA”). Similarly, on the
1086-PRCC dataset, the corresponding improvements of the
1087:“+HSA + VCS” module can reach 6.4% (13.4%, baseline) and 10% (23.5%, “+HSA”), respectively.
1088-3) In the SAVS and Pixel Sampling, both focus on the
1089-surface features of clothes; thus, we also compare
1090-their performance on two widely used public datasets,
1091-including LTCC and Celeb-reID. Since ResNet-50 is the
1092-backbone of Pixel Sampling, to obtain a fair comparison,
1093-ResNet-50 is also used as the backbone in the SAVS,
1094:but the human semantic information is ignored, and we
1095:call this approach “ResNet50 + VCS.” Along another
1096-line, to assess the advantages of the backbone, the
1097-backbone of the SAVS is replaced with the Swin-T
1098:based on the “ResNet50 + VCS,” and we name this
1099:approach “Swin-T + VCS.” Finally, the HSA module is
1100:further assessed, and the human semantic information is
1101:embedded into “Swin-T + VCS,” whose name is “SwinT + VCS + HSA.” Their results are shown in Table IV.
1102-From this, we can observe that when the same backbone
1103-and training strategy are used in the Pixel Sampling and
1104:the “ResNet50 + VCS,” the performance of the latter
1105-is much better than that of the former. For example, the
1106:mAP (rank-1) accuracies of the “ResNet50 + VCS” and
1107-Pixel Sampling on the LTCC dataset are 18.7% (47.7%)
1108-and 16.1% (42.3%), respectively, whose improvement
1109-can reach 2.6% (5.4%). We can obtain the same
1110-results from the Celeb-reID dataset. The reason why
1111:the “ResNet50 + VCS” is much better than the Pixel
1112-Sampling is that the latter only focuses on a single item
1113-of clothing, but all clothing regions of the person are
1114-fully considered in the former. In this way, the negative effect of cloth-changing can be reduced as much
1115-as possible, and a more robust feature representation
1116-can be extracted by covering the clothing regions and
1117:focusing the model on the visual semantic information
1118-unrelated to the clothes. Thus, these experiments can
1119:further prove that the VCS module is very effective
1120-for solving the cloth-changing issue. In addition, when
1121-the Swin-T is further used, we can see that the performance can be further improved. For example, on the
1122:LTCC dataset, the mAP (rank-1) accuracies of “SwinT + VCS” and Pixel Sampling are 22.8% (54.9%)
1123-and 16.1% (42.3%), respectively, whose improvement
1124-can reach 6.7% (12.6%). Thus, it can be proven that
1125-the Swin-T is also very efficient, and in our following
1126-experiments, it is utilized as the backbone of the SAVS.
1127:Finally, when the HSA module is further employed, its
1128-performance can significantly outperform the baseline,
1129:the Pixel Sampling, “ResNet50 + VCS,” and “SwinT + VCS” regardless of datasets.
1130-C. Convergence Analysis
1131-In this section, we evaluate the convergence of the proposed
1132-SAVS method on four public cloth-changing person ReID
--
1140-11
1141-
1142-TABLE IV
1143:A DVANTAGE OF THE VCS AND BACKBONE ON THE LTCC AND
1144-C ELEB - RE ID DATASETS W HEN C OMPARED W ITH
1145-THE P IXEL S AMPLING M ETHOD
1146-
--
1151-prove the effectiveness of the SAVS method.
1152-D. Qualitative Visualization
1153-To further prove the effectiveness and robustness of the
1154:SAVS, in this section, we visualize some results of the
1155-proposed SAVS on the different datasets. In this investigation,
1156:three aspects are considered: 1) visualization of the attention
1157:maps; 2) visualization of the similarity map; and 3) qualitative visualization of the retrieval results. In the following,
1158-we discuss these three aspects separately, and their results are
1159-given in Figs. 7–10. From them, we can make the following
1160-observations.
1161-1) To further illustrate which part of the focused features
1162-are learned with the help of different modules in the
1163:SAVS, the attention maps are visualized and displayed
1164-in Fig. 7. In Fig. 7, the first row indicates the original images from different datasets. The second row
1165:(“+HSA”) represents which channels on the feature map
1166-are activated by the reweighting operation where the
1167:Swin-T and the HSA module are used. The third row
1168:(“+HSA + VCS”) denotes which channels between the
1169:original features and visual shielding features are more
1170:consistently constrained by the semantic alignment loss
1171:and the VCS module, where the Swin-T, the HSA, and
1172:VCS modules are utilized. From them, we can see that
1173:in the second row, the HSA module is utilized; thus,
1174:only the human semantic channels are activated where
1175-the activated regions focus on the human body, but
1176-the background information is largely ignored. Thus,
1177-
--
1182-
1183-IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
1184-
1185:Fig. 8. Similarity maps with or without “+HSA + VCS” when the same
1186-person wears different clothes. The cosine similarity is used to calculate the
1187-distance between two images. The color of each square represents the degree
1188-of similarity between the two images indicated by the horizontal and vertical
1189-coordinates. The blue and green colors represent the most and the least similar
1190-pairs, respectively. (a) 14 images of the same person wearing different clothes.
1191:(b) Similarity map without the HSA and VCS modules. (c) Similarity map
1192:with the HSA and VCS modules.
1193-Fig. 7. Visualization of attention maps. The first row, second row, and
1194:third row indicate the original images, the results of “HSA,” and the results
1195:of “+HSA + VCS,” respectively. The “+HSA” focuses on the human
1196:body region, and the “+HSA + VCS” focuses on clothing-irrelevant cues,
1197-e.g., head, legs, and shoes.
1198-
1199-these channels (corresponding to features that play a
1200-major role in later classification) are enhanced, and
1201-more discriminative and robust features can be extracted.
1202:From the third row, we can observe that the VCS module
1203-focuses on human body nonclothing regions, e.g., head,
1204-face, legs, arms, shoes, and belongings, and the human
1205-clothing regions are ignored; thus, the extracted features
1206-are clothing-irrelevant, which can effectively solve the
1207-issue of human cloth-changing, and a more discriminative and robust feature representation can be obtained.
1208:Thus, these experiments can further prove the effectiveness and advantages of the HSA and VCS modules.
1209:2) To intuitively illustrate the effectiveness of the HSA and
1210:VCS modules from another view, the feature similarities
1211-between different images are calculated. Specifically,
1212-14 images of the same person wearing different clothes
1213-are first selected, and then the features are extracted
1214-for each image by the Swin-T (our baseline). Moreover, the cosine similarities between any two images
1215-are calculated by the corresponding features. Finally,
1216-we repeat the above operation in pairs for all 14 images
1217:and visualize their similarities to obtain a 14 × 14 similarity map. Similarly, we can also extract these image
1218:features by Swin-T + HSA + VCS (our proposed SAVS
1219-module), and then the cosine similarities between them
1220-and the similarity map are calculated. The results are
1221:given in Fig. 8, where the blue and green colors represent
1222-the most and the least similar pairs, respectively. From
1223-this, we can determine that when only the baseline
1224-is utilized, the extracted features are very relative to
--
1226-robust; thus, the similarities between the same person
1227-wearing different clothes are very small (the color of
1228-
1229:Fig. 9. Similarity maps with or without “+HSA + VCS” when different
1230-persons wear similar clothes. The cosine similarity is used to calculate the
1231-distance between two images. The color of each square represents the degree
1232-of similarity between the two images indicated by the horizontal and vertical
1233-coordinates. The blue and green colors represent the most and the least similar
1234-pairs, respectively. (a) 14 images of the different person wearing similar
1235:clothes. (b) Similarity map without the HSA and VCS modules. (c) Similarity
1236:map with the HSA and VCS modules.
1237-
1238:most of the squares is green). However, when the HSA
1239:and VCS modules are embedded into the baseline, the
1240:human semantic information and the clothing-irrelevant
1241-clues are fully explored; thus, the extracted features
1242-are more discriminative and robust, where similarities
1243-between the same person wearing different clothes are
1244-added, and the color of most of the squares is blue.
1245:We also can observe the similar conclusions from Fig. 9
1246-where different persons wear similar clothes. Thus, these
1247-experiments can further prove that the SAVS is very
1248-effective, efficient, and robust.
--
1252-This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.
1253-GAO et al.: SEMANTIC-AWARE ATTENTION AND VISUAL SHIELDING NETWORK
1254-
1255:Fig. 10. Qualitative visualization of (a) baseline, (b) baseline + HSA, and
1256:(c) baseline + HSA + VCS (SAVS) on the PRCC dataset. The top-left column
1257-is a single query image, and the other columns represent the top ten retrieval
1258-results. Note that the red boxes indicate the correct results.
1259-
1260:Fig. 11. Qualitative visualization of the SAVS on the NKUP dataset. The
1261-top-left column is a single query image, and the other columns represent the
1262-top ten retrieval results. Note that the red boxes indicate the correct results.
1263-(a) and (b) Full-frontal images are given. (c) and (d) Provided image is a back
--
1270-is a retrieval example including one query image and
1271-the top ten most similar images. We notice that in
1272-Fig. 10(a)–(c), the effectiveness of different parts of
1273:the SAVS, including the baseline, the baseline + HSA,
1274:and the baseline + HSA + VCS, is visualized with
1275-respect to the PRCC dataset. Fig. 10 demonstrates that
1276:since the visual appearance of the cloth-changing person drastically changes, the baseline still has difficulty
1277-obtaining the correct retrieval results when only the
1278:original feature is extracted. When the human semantic
1279-information is further combined, the feature discrimination is improved. Although the number of correct
1280-retrieval results is increased significantly, the first correct
1281-retrieval result lies in the third position, and the average
1282:performance is not unsatisfactory. Finally, when visual
1283:shielding information is employed, more correct retrieval
1284-results can be returned; moreover, their locations are
1285-very close to the front of the images. Thus, these results
1286:can further prove that the HSA and VCS modules are
1287-very useful for describing cloth-changing persons, and
1288-
1289-13
1290-
1291-the extracted feature is effective and robust. To further
1292-demonstrate the effectiveness of the SAVS modules on
1293:difficult samples, we visualize four retrieval results of
1294-the SAVS module on the NKUP dataset, where the
1295-face information is often covered, and the results are
1296-presented in Fig. 11. In the two retrieval cases [see
--
1307-declines dramatically, where only one or two correct
1308-results are returned. These experimental results indicate
1309-that it is quite challenging to perform cloth-changing
1310:person ReID when the provided image lacks visual
1311:semantics to a large extent.
1312-VI. C ONCLUSION
1313-This work proposed a novel SAVS algorithm for the
1314-cloth-changing person ReID task, where the key idea is to
1315:shield clues related to the appearance of clothes and only
1316:focus on visual semantic information that is not sensitive
1317:to view/posture changes. In the SAVS, an HSA module is
1318-designed to highlight human information and reweight the
1319:visual feature map, and a VCS module is proposed to extract
1320-a more robust feature representation for cloth-changing tasks
1321:by covering clothing regions and focusing the model on visual
1322:semantic information unrelated to clothes. Most importantly,
1323-these two modules are jointly explored in an end-to-end unified
1324-framework. The results of extensive experiments conducted
1325-on four cloth-changing person ReID datasets demonstrate
--
1327-cloth-changing person ReID methods in terms of both mAP
1328-and rank-1 accuracy, and more discriminative and robust
1329-features can be extracted to represent the cloth-changing
1330:persons. An ablation study also proves that human semantic
1331:information and visual shielding information are very helpful for solving the cloth-changing person ReID task where
1332-clothes-independent features can be obtained. Additionally,
1333:different qualitative visualizations can further prove the effectiveness and robustness of the HSA and VCS modules.
1334:In the future, we intend to explore how to shield the effects
1335-of clothing changes and design approaches based on novel
1336-dynamic features such as gait that can also describe static
1337-person images.
--
1358-IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
1359-
1360-[5] L. Gao, H. Zhang, Z. Gao, W. Guan, Z. Cheng, and M. Wang,
1361:“Texture semantically aligned with visibility-aware for partial person
1362-re-identification,” in Proc. 28th ACM Int. Conf. Multimedia, Oct. 2020,
1363-pp. 3771–3779.
1364-[6] Y. Lin et al., “Improving person re-identification by attribute and identity
--
1479-in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR),
1480-Jun. 2021, pp. 10508–10517.
1481-[45] Z. Gao, H. Wei, W. Guan, W. Nie, M. Liu, and M. Wang, “Multigranular
1482:visual-semantic embedding for cloth-changing person re-identification,”
1483-in Proc. 30th ACM Int. Conf. Multimedia, Lisboa, Portugal, Oct. 2022,
1484-pp. 3703–3711.
1485-[46] X. Shu, G. Li, X. Wang, W. Ruan, and Q. Tian, “Semantic-guided
--
1580-
1581-Weili Guan (Member, IEEE) received the bachelor’s
1582-degree from Huaqiao University, Quanzhou, China,
1583:in 2009, and the Graduate Diploma and master’s
1584-degrees from the National University of Singapore,
1585-Singapore, in 2011 and 2014, respectively. She is
1586-currently pursuing the Ph.D. degree with the Faculty of Information Technology, Monash University,

codex
SAVS 的 ablation 很有用：它不是只报 HSA 和 VCS 加了会涨，而是把原图、前景、背景、前景重权重、背景重权重都拆开测，证明“人体语义有用，但不能简单丢背景；要用语义去调制原特征”。这就是视角成立的关键证据。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | sed -n '1,675p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112567

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

A Semantic-guided occlusion simulation based local feature semantic
expansion network for person re-identiﬁcation
Zelin Deng a , Mingxuan Tang a , Ke Nai
a

a,∗, Guiji Li b , Shaomiao Chen c , Pei He d

School of Computer, Changsha University of Science and Technology, Changsha, 410114, China

b School of Computer Science and Engineering, Changsha University, Changsha, 410022, China
c

School of Computer Science and Engineering, Hunan University of Science and Technology, Xiangtan, 411201, China

d School of Computer Science and Cyber Engineering, Guangzhou University, Guangzhou, 510000, China

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Local semantic features
Occlusion simulation

a b s t r a c t
Person re-identiﬁcation is a challenging task which aims to retrieve images of a target pedestrian across diﬀerent
cameras. Currently, existing models generally neglect the semantic relevance of local features, which may cause
to limited identiﬁcation performance. To address this issue, we propose a semantic-guided occlusion simulation
based local feature semantic expansion network (FOSENet) to pursue satisfactory performance. Firstly, we propose a semantic-guided occlusion simulation (SGOS) method, which generates semantically relevant occlusion
patches and further simulates the occlusion based on the semantic information of pedestrians to improve the diversity of the occluded pedestrian samples. Then, we propose a local feature semantic expansion (LFSE) method,
which obtains several local areas around each key area and selects some useful local areas as additional cues of
the key area to enhance the discrimination ability. Finally, we introduce a cross-center oﬀset loss function, which
enlarges the distance of a pair of nearest neighbor samples from two classes to optimize the model. Thus, the
samples belonging to diﬀerent identities are pushed further away. Extensive experiments are conducted on four
challenging datasets, and the proposed method can achieve competitive results compared to multiple state-ofthe-art works. The code is publicly available at https://github.com/xuanthan-art/code-with-paper-FOSENet.git

1. Introduction
Person re-identiﬁcation aims to determine whether images captured by diﬀerent cameras contain the same individual by retrieving
pedestrian information across these cameras, which has been widely
used in artiﬁcial intelligence and security monitoring [1,2]. With the
rapid development of deep learning and the release of pedestrians reidentiﬁcation (Re-ID) image datasets, many powerful Re-ID methods
have been proposed to pursue impressive performance, such as multiscale methods [3,4], ﬁne-grained methods [5,6] and adversarial network learning [7–9]. However, person re-identiﬁcation still faces multiple challenges: (1) occlusion. In real-world environments like train
stations, airports, and shopping malls, pedestrian images captured by
stationary surveillance cameras often suﬀer from occlusion caused by
vegetation, vehicles, and other individuals. The occlusion may cause
a pedestrian to exhibit signiﬁcant appearance changes under diﬀerent
environments, which greatly limits the re-identiﬁcation accuracy [17].
(2) local features semantic representation.It is widely acknowledged
that CNN features are extracted from pre-trained models, as the depth

of the layers increases, CNN features tend to overly concentrate on a
single local center, which may lead to the neglect of other important
details. As a result, the semantic discriminability of the extracted features is likely to decrease, potentially aﬀecting the overall performance
of the model. In particular, some regions in pedestrian images, such
as ground surfaces, walls, and background objects, are repeatedly extracted, thereby generating redundant semantic information that is frequently attended by the model but does not contribute to identity discrimination or occlusion interference representations. These redundant
semantics may mislead feature learning and increase similarity across
diﬀerent identities. Therefore, how to remove the interfering semantic information and obtain more additional semantic information to enhance the feature representations of pedestrian identity is a challenge for
person re-identiﬁcation [10,11]. To address these issues, existing methods mainly focus on enhancing the robustness of the model to deal with
occluded samples and the extraction of semantic information from local
features.
On the one hand, to solve the occlusion problem, many methods
try to enhance the robustness of the model by simulating real-world

∗ Corresponding author.

E-mail addresses: zelin.deng@csust.edu.cn (Z. Deng), xuanthan@163.com (M. Tang), naike_hnu@hnu.edu.cn (K. Nai), guiji.li@hnu.edu.cn (G. Li),
csm123@hnust.edu.cn (S. Chen), bk_he@126.com (P. He).
https://doi.org/10.1016/j.patcog.2025.112567
Received 5 August 2024; Received in revised form 20 June 2025; Accepted 3 October 2025
Available online 10 October 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112567

Z. Deng et al.

occlusions through random erasure of image regions [12,13]. Then,
to make the occlusion patches more realistic for simulated occlusion,
Chen et al.[13] proposed a method to crop other images to create occlusion patches, and these patches were used at conventional occlusion
locations (such as top, bottom, left, and right) in pedestrian images to
enhance the robustness of the model to recognize occlusion situations.
Subsequently, in order to better model the feature extraction during occlusion, some works have proposed feature erasure [6]. Wang et al. [14]
introduced a feature erasure and diﬀusion network, which generates an
occlusion mask to guide the model to erase the feature map. As a result,
the model can focus on unobstructed regions and diﬀuse the pedestrian
representation in the feature space to boost the identiﬁcation capability of the target pedestrian. However, existing occlusion simulation or
erasure methods are positionally arbitrary and heavily dependent on
manual design. These methods fail to adequately simulate the occlusions that models encounter in real-world scenarios, where pedestrians
are frequently partially obscured by irregular objects. Moreover, the positions of such occlusions tend to interfere with critical semantic information. Thus, feature extraction and semantic recognition in diﬀerent
occlusion scenarios become more diﬃcult, and the model may focus on
the useless feature areas, leading to a signiﬁcant performance degradation in recognition accuracy.
On the other hand, to extract more semantic information, some methods employ auxiliary models to acquire additional semantic information
to address the issue of limited representation ability in local feature semantics [15]. Yang et al. [16] proposed to introduce discretizing posture
information into visibility labels of body parts to suppress the impact
of interfering information, thereby obtaining more posture information
from visible areas as additional semantic cues to enhance the recognition eﬃciency under various scenes. Subsequently, to extract more
precise semantic information, some methods employ multi-scale information to learn additional semantic information to improve the feature
representation ability of pedestrian images [17]. Gao et al.[18] proposed
a teacher-student ecoder (TSD) framework to aggregate pedestrian area
information through feature distillation and a mask generator was designed to provide discriminative regions for better re-identiﬁcation performance. However, the extraction of semantic information heavily relies on prior training outcomes. To extract more semantic information
that is not focused on by pre-trained models, Zhang et al. [19] introduced a complementary network for person re-identiﬁcation, which utilizes two branches to independently extract potentially discriminative
information that is beneﬁcial for pedestrian recognition from both background and salient regions across the global scope. The proposed network can eﬀectively capture additional semantic information such as
backpacks and handbags to improve the identiﬁcation performance in
pedestrian recognition tasks. However, the loss of partially discriminative features caused by real-world occlusions ampliﬁes the inﬂuence of
unoccluded key regions in recognition decisions, which may increase
the similarity between diﬀerent pedestrians and reduce the recognition
accuracy of the model. Meanwhile, existing methods rely heavily on the
unoccluded key regions, making them highly susceptible to misidentiﬁcation.
Based on the above analysis, we propose a semantic-guided occlusion
simulation based local feature semantic expansion network. Firstly, we
try to exploit semantic information to guide the simulation of occlusions to generate semantically occlusion patches. Specially, occlusion
operations are performed based on their position, shape, and context to
ensure that the generated occlusions are more consistent with the actual content of the images to better simulate real-world scenarios where
pedestrians are obstructed. Subsequently, to obtain richer semantic information for each individual pedestrian, some local areas near the key
area are selected, which are treated as additional cues to enhance semantic diversity. Finally, we introduce a cross-center oﬀset loss function, which enlarges the distance of a pair of nearest neighbor samples
from two classes during the training process to ensure that the samples
belonging to diﬀerent identities are pushed further away. We conducted

extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
realistic occlusion scenarios, which achieves better adaptability to deal
with complex visual interference. In addition, by mining local regions
with rich identity-related cues, it can obtain better discriminative ability
to perform person re-identiﬁcation under occlusion scenes.
The main contributions can be summarized as follows:
•

We propose a semantic-guide occlusion simulation method, which
generates semantically relevant occlusion patches with semantic information of object pedestrians and further realizes occlusion simulation based on their position, shape, and context to generate more
diverse occluded pedestrian samples.
• We develope a local feature semantic extension method, which selects some useful local regions near the key areas to obtain additional
semantic information, thereby enhancing the discriminative ability
of pedestrian features.
• We introduce a cross-center oﬀset loss function, which increases the
distance between a pair of nearest neighbor samples from two classes
to optimize the model and enhance its generalization ability.
• We conduct extensive experiments to demonstrate the eﬀectiveness of our proposed method, which achieves promising performance on four large-scale benchmark datasets: Market-1501
[20], DukeMTMC-REID [21], CUHK03-NP [22], and OccludedDukeMTMC [15].
2. Related work
In this section, we provide a brief overview of methods related to
person re-identiﬁcation, occluded person re-identiﬁcation, and partial
person re-identiﬁcation.
2.1. Person re-identiﬁcation
Pedestrian re-identiﬁcation is a crucial computer vision task that enables the re-identiﬁcation of the same individual across diﬀerent scenes
by analyzing and comparing images of pedestrians, and its research and
applications have obtained substantial progress due to the advancement
of convolutional neural networks [1,2] . One core challenge of person
re-identiﬁcation is how to accurately identify the same pedestrian under varying lighting scenes, viewpoints, and occlusions [23] Typically,
existing pedestrian re-identiﬁcation methods attempt to learn more reﬁned local feature semantic information to achieve ﬁne-grained feature matching [5]. Zhang et al.[24] proposed a method that leverages
graph neural networks to learn global and local features. The proposed
method can utilize the characteristics of graph neural networks to acquire more locally joint semantic information. Subsequently, to enhance
the discriminative ability across diﬀerent regions, some methods have
employed attention mechanisms to eﬀectively focus on pedestrian areas.
Chen et al. [3] proposed an attention pyramid method that employs attention regions in a multi-scale way to learn corresponding attention at
various ﬁne-grained levels to learn beneﬁcial human foreground areas,
thereby producing eﬀective features enriched with identity information.
Additionally, some methods enrich the diversity of features by extracting them from pedestrian images in a region speciﬁc or layered manner
[23]. Zhang et al. [25] proposed a Hierarchical Aggregation Transformer
method, which aggregates diﬀerent feature maps with the same Transformer and further calibrates the features with another Transformer
to enrich high-level semantic information to capture a more comprehensive set of global and local information. However, these methods
overlook the occlusion issue and fail to separate individuals from obstructions when interfering elements are semantically similar to the
pedestrian information, which will signiﬁcantly reduce the recognition
ability.
2

Pattern Recognition 172 (2026) 112567

Z. Deng et al.

2.2. Occluded person re-identiﬁcation

ages. Therefore, the goal of local pedestrian re-identiﬁcation is to match
the partial information in pedestrian images with the overall images in
the database [31]. To better learn diﬀerent local information, Dai et
al. [6] proposed the Batch DropBlock Network (BDB), which erases the
same area of images in a batch to concentrate on other areas and learn
richer local information. Subsequently, to better learn the local features
of pedestrians, Zhou et al. [32] simulated occlusions and introduced a
co-saliency branch to learn those parts of the pedestrians that are not
occluded. Additionally, Dong et al. [33] generated adversarial samples
by randomly erasing, transforming, and adding noise to feature maps
to learn undisturbed areas to obtain more reliable local information.
Recently, Yan et al. [34] leveraged the correlation between local features for aggregation and employed a reﬁned partitioning strategy to
concentrate and learn the visible areas more eﬀectively. Diﬀerent from
the above methods, Huang et al. [35] performed horizontal strip partitioning at various scales to better capture the representable information
of diﬀerent local parts of pedestrians to eﬀectively learn semantic information from local to global levels. To address the issue of partial occlusion, we propose selectively incorporating the features of adjacent local
regions of important local regions as supplementary information into
pedestrian features,to improve the discriminative ability to distinguish
pedestrians in locally similar scenarios.

In real-world scenarios, due to the ﬁxed positions of cameras, occlusions are inevitable when pedestrians are captured, which greatly
limits the re-identiﬁcation performance [26]. To address the issues arising from occluded pedestrians, some works have focused on simulating
occlusion scenarios to enhance the robustness against occlusions [12].
Huang et al [27]. proposed an Ad-Occluded Samples method, which
introduces deliberately designed occlusions into the training data and
trains the model after identifying key areas to better cope with realworld occlusion situations. Subsequently, to enable the model to better
learn with occlusions under diﬀerent scenes, Zhao et al [28]. proposed
an Incremental Generative Occlusion Adversarial Suppression Network,
which simulates occlusions from easy to diﬃcult to make the model
more robust to occlusions. The CBE module [29], which stands for Composite Batch Erasing, combines Batch-level Erasing (BCE) with Imagelevel Erasing (RE). These two erasing operations are applied to the original pedestrian images within the same batch to obtain two batches
of augmented images that are merged into a larger batch for training,
thereby enabling the model to be learned under occlusion scenes which
aﬀect diﬀerent body parts and of varying sizes. Some other methods
focus on directing the attention to visible areas to suppress the interference caused by occlusions. Jia et al. [30]proposed a set-matching
method without a collection, which uses a set matching approach to
concentrate on visible regions of the model. Subsequently, Wang et al.
[17] proposed a Key Point-Aware Occlusion Suppression and Semantic
Alignment (POS) method, which focuses on the unobstructed regions of
pedestrians to achieve semantic alignment to eﬀectively extract semantic information from the visible areas. However, these methods still have
some drawbacks. Firstly, their simulation strategy shows poor adaptability to the interference caused by occlusion in diﬀerent scenarios. Moreover, in real-world scenarios, they are unable to handle occlusions of
various shapes and may mistake the semantic information of occluding
objects for that of pedestrian features, which signiﬁcantly aﬀects the
recognition performance. In our proposed approach, we aim to more
eﬀectively simulate a diverse range of occlusion scenarios in order to
signiﬁcantly enhance recognition performance under various occlusion
interferences.

3. Methodology
3.1. Overview of the proposed method
The overall network structure proposed in this paper is shown in
Fig. 1. We propose a semantic-guided occlusion simulation based local
feature semantic expansion network (FOSENet) to solve the problem
of limited local semantic information representation ability caused by
semantic occlusion in real-world scenarios. Firstly, FOSENet integrates
occlusion patches with the semantic information of pedestrians to generate semantically relevant occlusion patches and further simulates the
occlusion with the generated occlusion patches and the location, shape,
and context of the semantic information of pedestrians to improve the
samples. Then, ResNet-50 is employed as a pretrained extractor to extract features from pedestrian images and obtain the attention weights
for each channel through an attention module to capture the key areas
and surrounding local areas. To ensure that the key areas have more
individual distinctiveness for representing pedestrian information, we
propose a local feature semantic expansion (LFSE) method. Speciﬁcally,
LFSE sorts the surrounding local areas of each key area extracted by the
attention module to select several local areas and fuses them with the

2.3. Partial person re-identiﬁcation
Due to issues such as occlusion, lighting, and changes in viewing
angle, the representable information of pedestrian images is limited,
which means useful information only partially exists on pedestrian im-

Fig. 1. Overall architecture of FOSENet, FOSENet mainly contains SGOS, LFSE, cross-center oﬀset loss function. Then, ResNet-50 is used as a backbone network.
FOSENet is designed to enhance model adaptability under occlusion conditions by generating diverse occlusion samples and mining local features with stronger
identity-discriminative ability.
3

Pattern Recognition 172 (2026) 112567

Z. Deng et al.

Table 1
The key notations of the proposed method.

key area to enhance the semantic richness of the feature areas. Finally,
we introduce a cross-center oﬀset loss function, which enlarges the distance of a pair of nearest neighbor samples from two classes to optimize
the model. Some key notations of the proposed method are listed in
Table 1.
3.2. Semantic-guided occlusion simulation (SGOS)
Existing occlusion simulation methods fail to adapt to various realworld occlusion scenarios as the occlusion areas are determined based
on randomness or pixel thresholds and the generated samples can hardly
achieve eﬀective occlusion simulation. However, occlusions also contain
certain semantic information rather than a random blank space, and this
semantic information can often interfere with the model. We propose a
semantic-guided occlusion simulation (SGOS) method to better simulate
highly complicated environments to improve the recognition ability of
the model. SGOS fuses the pedestrian feature with the occlusion set to
generate a semantic related occlusion patch, which ensures the semantic consistency between the occlusion patch and the pedestrian image
occluded with the location, shape, and context, of the semantic information of the pedestrian.
Speciﬁcally, we utilize the semantic diﬀerence extracted from the input image in diﬀerent convolutional layers to guide the occlusion simulation, which is divided into two phases, as illustrated in Fig. 2 . The ﬁrst
phase involves the extraction of semantic information from the image,
which is obtained from the positions and semantic information occluded
in the image. The second phase is occlusion simulation which fuses the

Notation

Meaning

𝐅𝑖𝐷𝑖𝑓 𝑓
𝐅𝑖,𝑖+1
𝑐
𝐌𝑖𝐵
′
𝐅𝑠𝑜𝑟𝑡
𝐖𝐷𝑖𝑓 𝑓
𝐅exp
′
𝐅𝑠𝑒𝑙𝑒𝑐𝑡
𝐂𝐀 , 𝐂𝐁
𝐋𝐶𝐶𝑂

Semantic diﬀerential features
Convolution feature map
Semantic occlusion patch set
Importance ranking set of surrounding area
Diﬀerential weight set
Semantic extension features
Additional semantic features
Sample anchor
Cross-center oﬀset loss

occlusion patch set with the semantic information patch extracted in
the ﬁrst phase. Then, we further simulate the occlusion with the fused
occlusion patches by applying them to the semantic positions and environments for occlusion in the image to enhance the diversity of samples.
In the ﬁrst phase, we need to acquire the semantic information from
the output feature maps in adjacent convolutional layers. We represent
the input image as X, and the feature maps 𝐅𝑖𝑐 = {𝑐𝑛𝑖 , 𝑛 = 1, … , 𝑁}, 𝐅𝑖+1
=
𝑐
{𝑐𝑛𝑖+1 , 𝑛 = 1, … , 𝑁} are extracted from adjacent convolutional layers as:
𝐹𝑐𝑖 = 𝐶𝑜𝑛𝑣1∼𝑖 (𝑋), 𝐹𝑐𝑖+1 = 𝐶𝑜𝑛𝑣1∼𝑖+1 (𝑋),

(1)

where the convolutional network follows the same structure as the pretrained ResNet-50, 𝑁 represents the maximum number of channels and
1∼i indicates the number of layers used in the feature extraction process. Moreover, we utilize the degree of attention to the regions of in-

Fig. 2. The motivation behind the SGOS module is to generate semantically meaningful occlusion patches for more target simulation of real-world occlusion scenarios.
To achieve this goal, we integrate the semantic diﬀerence information extracted from the input image with a predeﬁned set of occlusion patches, and guide the
placement and fusion process using the semantic diﬀerence map. To illustrate this process more clearly, we provide intermediate examples for each sub-stage in the
ﬁgure. Obviously, the occlusion samples generated by the model better present the real occlusion scenarios.
4

Pattern Recognition 172 (2026) 112567

Z. Deng et al.
𝑖,𝑖+1
terest in the two feature maps to obtain the 𝐹𝑎𝑡𝑡
, which maximizes
the channel diﬀerences to extract the key information from the image
𝑖 and
but remove redundant semantic information. Then, we take 𝐹𝑎𝑡𝑡
𝑖+1
𝑖
𝐹𝑎𝑡𝑡 as inputs to obtain the semantic diﬀerential features 𝐹𝐷𝑖𝑓 𝑓 after
the channel diﬀerentiation operations. The process can be represented
as:
(
(
))
𝛼 𝑖,𝑖+1 = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥 𝑊𝛼 ⋅ 𝑓𝛼 𝐹 𝑖,𝑖+1 ,
(2)
𝑖,𝑖+1
𝐹𝑎𝑡𝑡
=

𝑁
∑

𝛼𝑐𝑖,𝑖+1 ⋅ 𝑐𝑛𝑖,𝑖+1 , 𝑐𝑛𝑖,𝑖+1 ∈ 𝐹 𝑖,𝑖+1 𝑚,

(3)

is sampled from locations that exceed a predeﬁned threshold in 𝐹𝐷𝑖𝑓 𝑓 ,
which ensures semantic relevance during the occlusion process.
Compared to traditional random occlusion strategies such as random erasing, which mask rectangular regions in a semantically agnostic
and unguided manner, our proposed SGOS method introduces semantic
occlusion blocks derived from real-world object shapes to construct occlusion augmentation samples that better resemble actual scenarios. By
guiding both the location and shape of the occlusion patches, SGOS can
eﬀectively generate diverse occluded pedestrian samples, thereby enhancing the model’s adaptability under complex occlusion conditions.

(4)

3.3. Local feature semantic expansion (LFSE)

𝑛=1

( 𝑖 )
( 𝑖+1 )|
|
𝑖
𝐹𝐷𝑖𝑓
𝑓 = ||𝑟𝑒𝑠𝑖𝑧𝑒 𝐹𝑎𝑡𝑡 , 𝑟𝑒𝑠𝑖𝑧𝑒 𝐹𝑎𝑡𝑡 ||.

( )
Here, 𝐖𝛼 represents the weight parameters of the attention model, 𝐟𝛼 ⋅
𝑖,𝑖+1
is the mapping function of the attention model, and 𝛂
denotes the
attention weight corresponding to the channel at layers i and i+1. The
semantic diﬀerential features aims to obtain a degree of attention for the
regions of interest in the two feature maps. Finally, the | ⋅ | represents
the diﬀerentiation operation to extract the semantic diﬀerential features
𝑖
𝐹𝐷𝑖𝑓
. Subsequently, we normalize the obtained semantic diﬀerential
𝑓
𝑖
features 𝐹𝐷𝑖𝑓
through a Batch Normalization (BN) layer as follows:
𝑓
(
)
𝑓𝑖 − 𝜇𝐹
𝑖
𝐵𝑁 𝐹𝐷𝑖𝑓
+ 𝛽𝑖 ,
𝑓 = 𝛾𝑖 √
𝜎𝐹2 + 𝜖

Generally, as current semantic information extraction methods
overly focus on local areas, the obtained semantic information is too singular to capture the unique identity information of the pedestrian and
exhibits a certain degree of similarity among pedestrians. Moreover, the
existence of occlusions can lead to disturbances in local regions, thereby
limiting the representation of local semantic information. To enhance
the identity-representative capability of the extracted features, we propose a method called Local Feature Semantic Extension (LFSE). LFSE
incorporates the semantic information from regions adjacent to the key
regions as supplementary data and exploits the semantic continuity between these surrounding regions and the key regions, thereby strengthening the discriminative power of pedestrian features. This process is
visually demonstrated in Fig. 3.
The proposed LFSE utilizes the spatial and channel attention mechanisms to obtain the channels of the key area and the local areas near the
key area. We denote the input feature as Y and the channels 𝐹𝑖 and 𝐹𝑖′
of the key area and neighboring local areas can be obtained as follows:

(5)

where 𝜇𝐹 and 𝜎𝐹 respectively represent the mean and standard deviation
calculations of the input features.
𝑖
In the second phase, we integrate the diﬀerential features 𝐹𝐷𝑖𝑓
𝑓
which have been normalized to a standard normal distribution in the
ﬁrst phase with the occlusion patch shape sett 𝐌𝑠 = {𝑠𝑛 , 𝑛 = 1, … , 𝑁}
to generate a collection of semantically enriched occlusion patches
𝐌𝑖𝐵 = {𝑚𝑛 , 𝑛 = 1, … , 𝑁}. The process is formulated as:
}𝑁
𝑖
|
𝑀𝐵𝑖 = {𝐹𝐷𝑖𝑓
𝑓 ⊕ 𝑠𝑛 |𝑠𝑛 ∈ 𝑀𝑠 𝑛=1 ,

𝐹𝑖 = 𝐶𝑎𝑚(𝐶𝑜𝑛𝑣(𝑌 )), 𝐹𝑖′ = 𝑆𝐶(𝐶𝑜𝑛𝑣(𝑌 )).

(7)

Here, 𝐶𝑜𝑛𝑣(⋅) follows the same structure as ResNet-50 and the required
channels can be obtained with 𝑆𝐶(⋅) and 𝐶𝑎𝑚(⋅). Moreover, we will
′
get an additional set of leads sorted by importance as 𝐹𝑠𝑜𝑟𝑡
= {𝑓𝑛′′ , 𝑛 =
1, … , 𝑁}. Firstly, the global average pooling operation is performed for
the surrounding region channels. Then, the result obtained by the global
average pooling operation goes through two fully connected layers and
activation functions to obtain the weight matrix. The overall process is

(6)

where ⊕ denotes an operator to fuse the semantic features and a randomly selected shape element from the shape set 𝑀𝑠 . The generated
semantic information patch set 𝑀𝐵𝑖 is used to occlude the input image
X with the positions, Speciﬁcally, the placement of occlusion blocks is
guided by the semantic diﬀerence map 𝐹𝐷𝑖𝑓 𝑓 , where the center position

Fig. 3. The LFSE module aims to enhance the identity-discriminative capability of local pedestrian features by treating the semantic information obtained from
regions adjacent to key areas as additional semantic cues. To better illustrate the rationale and structural logic of LFSE, we provide heatmap visualizations of both
the key local regions and their semantically adjacent areas.
5

Pattern Recognition 172 (2026) 112567

Z. Deng et al.

formulated as:
{ (
(
))|
}
𝑊̂ = 𝜎 𝑊2 𝛿 𝑊1 𝐺𝐴𝑃 (𝑓𝑛 ) |𝑓𝑛 ∈ 𝐹𝑖′ .
(8)
|
Next, these importance weights are utilized to perform a sorting operation on the feature maps in 𝐹 ′ with the following formula:
{
}
|
′
𝐹𝑠𝑜𝑟𝑡
= 𝑠𝑜𝑟𝑡(𝑓𝑛 , 𝑤̂ 𝑛 )|𝑓𝑛 ∈ 𝐹𝑖′ , 𝑤̂ 𝑛 ∈ 𝑊̂ .
(9)
|
Here, 𝐺𝐴𝑃 (⋅) is the global average pooling operation, 𝑊1 and 𝑊2 are
parameter weights for learning, 𝛿 is the ReLu activation function, and
𝑠𝑜𝑟𝑡(⋅) denotes to sort channels with the weight 𝑊̂ .
It is worth noting that the sorted channel set will contain duplicated
elements and interference elements caused by occlusions. To address
this issue, we need to select channels with more semantic information
and are more beneﬁcial for the semantic expansion by considering the
diﬀerentiation between diﬀerent channels. Speciﬁcally, we input the
′
feature channel set 𝐹𝑠𝑜𝑟𝑡
sorted by importance to obtain the diﬀerential
weight set 𝑊𝐷𝑖𝑓 𝑓 = {𝑤𝑛 , 𝑛 = 1, … , 𝑁} as follows:
′′
𝑓𝑎𝑣𝑔
=

𝑁
∑

1
′
𝑓 ′′ , 𝑓 ′′ ∈ 𝐹𝑠𝑜𝑟𝑡
𝑁 𝑛=1 𝑛 𝑛

Fig. 4. The motivation behind the proposed 𝐿𝐶𝐶𝑂 loss function is to increase the
distance between samples from diﬀerent identities, which enhances the model’s
generalization ability. By increasing the distance of Hearest Hard Sample Pair
to push the diﬀerent classes further away, the class domain boundaries become
clearer, and then improving the generalization ability of the model.

(10)

(
)
‖𝑓 ′′ −𝑓 ′′ ‖2
exp − 𝑛 2𝑎𝑣𝑔
𝑒𝑎𝑡𝑡
𝑛 = ∑

2𝜎𝑛

𝑁
𝑘=1,𝑘≠𝑛 exp

(

−

‖𝑓𝑛 −𝑓𝑘 ‖2
2𝜎𝑛2

′
) , 𝑓𝑛′′ ∈ 𝐹𝑠𝑜𝑟𝑡

𝑒𝑎𝑡𝑡
𝑤𝑛 = ∑𝑁 𝑛
, 𝑤𝑛 ∈ 𝑊𝐷𝑖𝑓 𝑓 ,
𝑎𝑡𝑡
𝑗=1 𝑒𝑗

(11)

Deﬁnition 1 (Nearest Hard Sample Pair). Given anchors 𝐶𝐴 and 𝐶𝐵 ,
let 𝑡𝑎 denotes the farthest positive sample from 𝐶𝐴 , 𝑡𝑏 represents the
sample that is closest to 𝑡𝑎 and belongs to 𝐶𝐵 . Here, 𝐶𝐵 is the anchor
point among all the anchor points that is closest to 𝑡𝑎 . Such pair(𝑡𝑎 ,𝑡𝑏 ) is
called nearest hard sample pair.

(12)

where 𝑤𝑛 in 𝑊𝐷𝑖𝑓 𝑓 denotes the diﬀerential weight of n-th channel, 𝑒𝑎𝑡𝑡
𝑛
′′
denotes the dissimilarity between n-th channel and other channels, 𝑓𝑎𝑣𝑔
′
is the mean of the 𝐹𝑠𝑜𝑟𝑡
and 𝜎𝑛 represents a parameter used to regulate
the diﬀerence in features. Then, we combine the diﬀerence and importance of channels to select some useful channels for ﬁnal fusion. The
overall process can be represented as:
u𝑛 = 𝛼𝑤𝑛 + 𝛽 ̂
w𝑛 , 𝑤𝑛 ∈ 𝑊𝐷𝑖𝑓 𝑓 , ̂
w𝑛 ∈ ̂
W,
{
𝑓𝑛′′ 𝑢𝑛 ≥ 𝑦̈
′
′
𝐹𝑠𝑒𝑙𝑒𝑐𝑡
=
, 𝑓𝑛′′ ∈ 𝐹𝑠𝑜𝑟𝑡
,
0
𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒

Speciﬁcally, the sample 𝑡𝑏 and anchor 𝐶𝐵 can be obtained as follows:
(16)

𝑡𝑏 = 𝑎𝑟𝑔𝑚𝑖𝑛 ‖𝑡𝑎 − 𝑡𝑗 ‖.

(17)

𝑗≠𝐴

𝑡𝑗 ∈𝐶𝐵

Next, we calculate the distance between the corresponding sample
anchor and its relative sample by a crossover way to adjust the distance
according to position relation of the center 𝐶𝐴 and the samples 𝑡𝑎 and
𝑡𝑏 . The speciﬁc loss function 𝐿𝐶𝐶𝑂 can be deﬁned as:

(13)
(14)

′
where the 𝛼, 𝛽, 𝑦̈ are learnable hyperparameters, 𝐹𝑠𝑒𝑙𝑒𝑐𝑡
is the channel
feature that is selected and fused as additional semantic information.
′
Finally, we fuse the channel of the key area and 𝐹𝑠𝑒𝑙𝑒𝑐𝑡
to obtain the
ﬁnal semantic extension feature 𝐹exp as follows:
′
𝐹𝑒𝑥𝑝 = 𝐹𝑖 ⊕ 𝐹𝑠𝑒𝑙𝑒𝑐𝑡
,

𝐶𝐵 = 𝑎𝑟𝑔𝑚𝑖𝑛 ‖𝑡𝑎 − 𝐶𝑗 ‖,

𝐿𝐶𝐶𝑂 = max(0, 𝑑1 + 𝑑2 + 𝜌),

(18)

where 𝑑1 = ‖𝑡𝑎 − 𝑡𝑏 ‖2 − ‖𝑡𝑏 − 𝐶𝐴 ‖2 , 𝑑2 = ‖𝑡𝑎 − 𝑡𝑏 ‖2 − ‖𝑡𝑎 − 𝐶𝐵 ‖2 , 𝜌 is an
adjustable number to adjust the distance interval between the negative
sample and the anchor of positive samples to enlarge the distances between diﬀerent classes.
Meanwhile, we also use the center loss 𝐿𝐶𝑒𝑛𝑡𝑒𝑟 and triplet loss 𝐿𝑡𝑟𝑖
to make the pedestrian features more identity distinguishable, and the
model is optimized by minimizing the sum of three losses. The overall
loss is formulated as:

(15)

where ⊕ denotes the fusion operation.
Existing local semantic feature extraction methods under occlusion
typically attempt to enhance feature representations by incorporating
additional semantic information to deal with occlusions. However, such
approaches primarily focus on key regions, which leads to the overemphasis of key regions during the re-identiﬁcation process, thereby
misleading the model toward incorrect predictions. In contrast, the
proposed LFSE module adopts a semantic structure-guided strategy to
expand local features in regions adjacent to identity-relevant areas.
This design conducts the mining of local features based on semantic
relevance and structural consistency, eﬀectively improving the discriminability of occluded pedestrian features. As a result, the model demonstrates greater robustness and recognition accuracy under occlusion scenarios.

𝐿 = 𝜆𝐿𝐶𝐶𝑂 + 𝐿𝐶𝑒𝑛𝑡𝑒𝑟 + 𝐿𝑡𝑟𝑖 ,

(19)

where 𝜆 is a parameter to balance the contribution of individual loss
terms.
3.5. Discuss
To make the model more suitable for actual scenarios, many works
have been proposed based on occlusion simulation to solve the occlusion problem to enhance the recognition ability of the model. However,
existing methods generally use occlusion patches with semantic information for occlusion simulations at random or pre-determined positions
[13,14,29] . Contrarily, our proposed occlusion simulation method employs the occlusion patches with some semantic information to better
simulate the information of occluded objects and the occlusion position is guided by the semantic information in the image to generate occluded pedestrian samples with better diversity. Secondly, to improve

3.4. Cross center oﬀset loss
To enable the model to better choose centroids in the sample space
to form stable class centroids, the samples belonging to diﬀerent identities are pushed further away. Thus, we propose a Cross Center Oﬀset
loss (𝐿𝐶𝐶𝑂 ) to improve the recognition performance, as speciﬁcally illustrated in Fig. 4 .
6

Pattern Recognition 172 (2026) 112567

Z. Deng et al.

Table 2
Comparison of CMC (%) and mAP (%) performances with the state-of-the-art
methods on Market-1501 and DukeMTMC-reID. The best performance value is
marked in bold, and the second best performance value is marked by underline.
The symbol “ - ” indicates that no corresponding value was provided in the
corresponding paper.

the local features semantic information representative ability under occlusion, many works use assistant models to obtain some additional semantic information [11,19,36] . Specially, our method fuses the key
area and several neighboring local areas to generate semantic features
with pedestrian identity information, which enhance the discriminability of occluded pedestrian features. For the loss function, we enlarge the
distance of a pair of nearest neighbor samples from two classes, which
makes the samples belonging to diﬀerent identities are pushed further
away during the training process.

Methods

4. EXPERIMENTAL ANALYSIS
In this section, to validate the eﬀectiveness of our proposed FOSENet

codex
FOSENet 这一篇把两个常见问题连在一起：遮挡不是随机空洞，而是有语义的干扰物；局部特征不是只要盯关键区域，而是关键区域旁边的语义也能补足身份线索。它的机制基本就是从这两个重定义自然长出来的。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -layout -f 7 -l 11 \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | sed -n '1,500p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Z. Deng et al.                                                                                                                        Pattern Recognition 172 (2026) 112567


the local features semantic information representative ability under oc-         Table 2
clusion, many works use assistant models to obtain some additional se-           Comparison of CMC (%) and mAP (%) performances with the state-of-the-art
mantic information [11,19,36] . Specially, our method fuses the key              methods on Market-1501 and DukeMTMC-reID. The best performance value is
                                                                                 marked in bold, and the second best performance value is marked by underline.
area and several neighboring local areas to generate semantic features
                                                                                 The symbol “ - ” indicates that no corresponding value was provided in the
with pedestrian identity information, which enhance the discriminabil-
                                                                                 corresponding paper.
ity of occluded pedestrian features. For the loss function, we enlarge the
distance of a pair of nearest neighbor samples from two classes, which                                                                    Market-1501       DukeMTMC-ReID
                                                                                  Methods                      year       Source
makes the samples belonging to diﬀerent identities are pushed further                                                                     mAp    Rank-1     mAp       Rank-1
away during the training process.
                                                                                  IGOAS [28]                   2021       TCSVT           84.1   93.4       75.1      86.9
                                                                                  PAT [40]                     2021       CVPR            88.0   95.4       78.2      88.8
                                                                                  APNet-S [3]                  2021       TIP             89.0   96.1       78.8      89.3
4. EXPERIMENTAL ANALYSIS
                                                                                  PEFB [41]                    2022       TNNLS           81.3   92.7       72.6      86.2
                                                                                  PAGCN [24]                   2022       SPIC            87.3   94.4       78.0      86.7
   In this section, to validate the eﬀectiveness of our proposed FOSENet
                                                                                  FED [14]                     2022       CVPR            86.3   95.0       78.0      89.4
method and its components, we conduct experiments on four public per-             DRL-Net [42]                 2022       IEEE TMM        86.9   94.7       76.6      88.1
son re-ID datasets: Market1501 [20] , DukeMTMC-reID [21] , CUHK
                                                                                  PRE-Net[34]                  2023       TCSVT           86.0   94.5       76.5      88.9
03-NP [22] and Occluded-DukeMTMC [15] .
                                                                                  ETNDNet [33]                 2023       TCSVT           87.2   95.3       77.9      88.5
                                                                                  AMG [43]                     2023       IEEE TMM        88.5   95.0       –         –
                                                                                  MHSA-Net [44]                2023       TNNLS           84.0   94.6       73.1      87.3
4.1. Experimental settings                                                        AdaSP [45]                   2023       CVPR            89.8   95.5       83.0      91.7
                                                                                  BPBreID [31]                 2023       WACV            87.0   95.1       78.3      89.6
    1)     Datasets. Market-1501 [20] is a dataset for Person re-                 Pose-guided re-ID [46]       2023       PR              78.6   93.5       78.1      84.2
identiﬁcation (PRI) which was released in 2015 by Dr. Li et al. The               OAT [47]                     2024       TIP             89.9   95.7       82.3      91.2
images in the dataset were collected from 6 cameras on Tsinghua Uni-              SCAT [48]                    2024       IEEE TII        88.0   95.1       79.8      89.3
versity campus, and a total of 1501 pedestrians were labeled. Among               FCFormer [11]                2024       IEEE TMM        86.8   95.0       78.8      89.7
them, 751 pedestrian annotations are used as the training set and 750             Gao.et [18]                  2024       ICASSP          –      –          82.8      90.8
                                                                                  VIT-SPT [26]                 2024       AAAI            86.2   94.5       79.1      89.4
pedestrian annotations are used as the test set. Each pedestrian image is
labeled with pedestrian ID and camera ID, where Pedestrian ID is used             our(FOSENet)                 –          –               91.8   96.9       83.8      90.8
to diﬀerentiate between diﬀerent pedestrians and Camera ID is used to
identify which camera the image came from.                                          Table 3
    The DukeMTMC-reID [21] dataset is a large-scale pedestrian re-                  Comparison of CMC (%) and mAP (%) performances with the state-of-
identiﬁcation image dataset provided by the Duke Multi-Target, Multi-               the-art methods on CUHK03-NP. The best performance value is marked
Camera Tracking program at Duke University in 2017. Images and                      in bold, and the second best performance value is marked by underline.
videos in the DukeMTMC-reID dataset were collected from eight cam-                  The symbol “ - ” indicates that no corresponding value was provided in
eras on the Duke University campus, which cover a variety of loca-                  the corresponding paper.
tions and scenarios on campus for both indoors and outdoors. There                                                                 CUHK03-NP
are 16,522 images in the dataset for training, which contains 702 dif-
                                                                                      Methods              Year       Source       Detected               Labeled
ferent identities. Moreover, 17,661 images were randomly sampled to
form the test set, which contains 702 diﬀerent pedestrian identities.                                                              mAp        Rank-1      mAp       Rank-1
    CUHK03 [22] is a classical dataset for pedestrian re-recognition                  BDB+Cut[6]           2019       CVPR         73.5       76.4        76.7      79.4
which is provided by Chinese University of Hong Kong (CUHK). This                     AsNet[49]            2020       SPL          77.1       81.4        80.8      83.2
dataset contains pedestrian images in diﬀerent scenarios and is used                  PAE[50]              2021       PR           77.9       80.9        80.8      83.9
to evaluate the performance of pedestrian re-recognition algorithms in                APNet-S[3]           2021       TIP          78.1       80.9        81.1      83.5
real scenarios. CUHK03-NP contains 1467 pedestrians captured by 10 (5                 PAGCN[24]            2022       SPIC         71.6       75.1        –         –
pairs) diﬀerent cameras, where images of 767 pedestrians are used to                  HRS[51]              2022       PR           79.3       83.2        81.5      83.2
form the training set and images of 700 pedestrians are used to form the
                                                                                      JDK[52]              2023       SPIC         75.2       77.3        77.8      79.1
text set. According to the label generation method, it can be categorized             AdaSP[45]            2023       CVPR         80.1       82.0        82.4      84.6
into “Detected" and “Labeled".                                                        MHSA-Net[44]         2023       TNNLS        69.3       72.8        72.7      75.6
    Occluded-DukeMTMC [15] is an occluded dataset derived from                        OAT[47]              2024       TIP          78.0       80.6        81.5      83.9
the DukeMTMC-reID dataset, which contains 15,618 training images,
                                                                                      our(FOSENet)         –          –            82.3       85.0        85.5      86.0
17,661 gallery images, and 2210 occluded query images to obtain an
extremely challenging dataset. We will train and test our proposed net-
work model on the above datasets to conﬁrm its superiority.
    2) Evaluation metrics. To ensure the reliability and eﬀectiveness                As show in Eq. (6), we introduce a shape set 𝑀𝑠 to simulate real-
of the experiments, Cumulative Matching Characteristics (CMC) curves             world occlusion scenarios during training and enhance model robust-
and mean accuracy (mAP) are used as evaluation metrics to measure the            ness. These occlusion blocks are derived from segmentation masks of
recognition accuracy and mean accuracy of diﬀerent algorithms. The ex-           non-human categories in the COCO dataset, such as backpacks, hand-
perimental setup consists of the use of a speciﬁc deep learning model or         bags, and ﬁre hydrants, which reﬂect common real-world occluders
algorithm for training and testing.                                              [39]. The selected masks are resized as shape templates and then fused
    3) Implementation details. According to existing pedestrian recogni-         with semantic patches extracted from high-response regions, guided by
tion works, we adopt BNNeck (Strong Baseline) [37] as our baseline,              the semantic diﬀerence map 𝐹𝐷𝑖𝑓 𝑓 . The placement centers of the oc-
implemented on PyTorch with a single NVIDIA GeForce RTX 4070 (12                 clusion blocks are sampled from regions in 𝐹𝐷𝑖𝑓 𝑓 that exceed a pre-
GB VRAM). The backbone network is ResNet-50 pre-trained on LUPer-                deﬁned threshold, ensuring that the occlusion process is semantically
son [38]. We use the Adam optimizer, with a warm-up learning rate for            guided. Moreover, to ensure reproducibility and the rigor of experimen-
the ﬁrst 10 epochs, followed by decay at epochs 30 and 70 (decay rate            tal comparisons, a ﬁxed random seed is used across all training runs,
of 0.1). The batch size is set to 64, and training runs for 130 epochs in        and the training set is reshuﬄed at the beginning of each epoch. The
total.                                                                           occlusion simulation ratio is set to 0.35, which falls within the typical

                                                                             7
Z. Deng et al.                                                                                                                 Pattern Recognition 172 (2026) 112567




Fig. 5. Feature distribution with t-SNE on the Market-1501 dataset. (A) feature distribution learned by the Backbone network; (B) distribution under the Baseline
model; (C) results from the proposed FOSENet.


                                                                                           Table 4
                                                                                           Comparison of CMC (%) and mAP (%) performances with
                                                                                           the state-of-the-art methods on Occluded-DukeMTMC. The best
                                                                                           performance value is marked in bold, and the second best per-
                                                                                           formance value is marked by underline.
                                                                                                                                   Occluded-DukeMTMC
                                                                                             Methods          Year    Source
                                                                                                                                   mAP      Rank-1

                                                                                             PAT [40]         2021    CVPR         53.6     64.5
                                                                                             Yang.et [16]     2021    CVPR         46.3     62.2

                                                                                             PEFB [41]        2022    TNNLS        43.5     56.3
                                                                                             DRL-Net [42]     2022    IEEE TMM     50.8     65.0
                                                                                             FED [14]         2022    CVPR         56.4     68.1
                                                                                             ETNDNet [33]     2023    TCSVT        57.6     68.1

                                                                                             BPBreID [31]     2023    WACV         54.1     66.7
                                                                                             MHSA-Net [44]    2023    TNNLS        42.4     55.4
                                                                                             AMG [43]         2023    IEEE TMM     59.7     68.5
                                                                                             PRE-Net [34]     2023    TCSVT        55.2     68.3

                                                                                             FCFormer [11]    2024    IEEE TMM     60.9     71.3
                                                                                             Gao.et [18]      2024    ICASSP       57.3     70.6
                                                                                             SCAT [48]        2024    IEEE TII     54.9     62.8
                                                                                             VIT-SPT [26]     2024    AAAI         57.4     68.6

                                                                                             Our(FOSENet)     –       –            61.5     68.7




                                                                                   commonly used feature extraction strategies in recent studies and are
                                                                                   included as representative references in our comparisons.
                                                                                       1) Comparisons on Market-1501 and DukeMTMC-reID. As shown
Fig. 6. Convergence of CMC (%) and mAP (%) accuracy over epochs on market-         in Table 2, our proposed FOSENet performs better than Baseline and
1501 and occluded-DukeMTMC datasets.
                                                                                   other state-of-the-art methods. Speciﬁcally, for the Market-1501 dataset,
                                                                                   where there are fewer occluded and similar-looking person identities,
range (0.3 ∼ 0.5) used in existing occlusion-based augmentation meth-              FOSENet can eﬀectively extract the person features and obtain a com-
ods [12]. Given the structured and semantically guided nature of our               petitive recognition rate of 96.9 % Rank-1 accuracy and 91.8 % mAP
occlusion blocks, this ratio strikes a balanced trade-oﬀ between intro-            score . For the DukeMTMC-reID dataset, where the data environment is
ducing eﬀective interference and preserving identity-discriminative fea-           much more diﬃcult, our FOSENet still achieves 90.8 % Rank-1 accuracy
tures, thereby supporting fair and consistent performance evaluation.              and 83.8 % mAP score, which conﬁrms the excellent performance of our
                                                                                   proposed FOSENet.
                                                                                       2) Comparisons on CUHK03-NP. CUHK03 is a challenging Person
4.2. Comparison with state-of-the-art methods on datasets.                         Re-ID dataset in recent years, and Table 3 lists the performance of our
                                                                                   method and other state-of-the-art methods. Obviously, our proposed
    In this section, we evaluate the proposed FOSENet model with Mar-              FOSENet achieves leading results on this dataset. To be speciﬁc, FOS-
ket1501, CUHK 03, DukeMTMCreID and Occluded-DukeMTMC, and                          ENet achieves a mAP score of 82.3 % and a Rank-1 score of 85.0 % on
compare FOSENet with state-of-the-art methods. Speciﬁcally, to en-                 CUHK 03-Detected labeled by a machine and obtains a mAP score of
sure a fair comparison and eliminate confounding factors, our FOSENet              85.5 % and a Rank-1 score of 86.0 % on CUHK 03-Labeled labeled by a
model and other baseline models use same initialization parameters,                human, which outperforms other state-of-the-art methods.
including batch size, optimizer conﬁguration, learning rate scheduling,                3) Comparisons on Occluded-DukeMTMC. Occluded-DukeMTMC, as
and the total number of training epochs, to organize experiments. More-            a dataset evolved from DukeMTMC, is more challenging in the occlusion
over, all models in our experiments employ the same data augmentation              aspect. As shown in Table 4, FOSENet still obtains satisfactory results
techniques (e.g., random rotation and random cropping), and most of                and achieves a mAP score of 61.5 % and a Rank-1 score of 68.7 %. Al-
the compared methods, including ours, are based on CNN architectures.              though FOSENet can not achieve a best Rank-1 score, the mAP score
Although a few baseline methods adopt Transformer-based feature ex-                of FOSENet signiﬁcantly surpasses other methods, which proves the ro-
tractors, we did not modify their original implementations but instead             bustness as well as the superiority of our proposed method under the
directly used their oﬃcially reported results. These methods represent             occlusion situations.

                                                                               8
Z. Deng et al.                                                                                                                Pattern Recognition 172 (2026) 112567




      Fig. 7. The impact of the parameter 𝜆 on model performance in terms of CMC (%) and mAP (%) on the Market-1501 and Occluded-DukeMTMC datasets.




Fig. 8. Attention heatmap visualization for the input image, Baseline model, and our proposed FOSENet. (A) shows the original input image; (B) presents the
attention response generated by the Baseline model; and (C) shows the output from FOSENet with our proposed modules integrated. It can be observed that the
proposed method enables the model to attend more accurately to local identity-relevant regions, especially under occlusion conditions, thereby improving feature
discrimination.


                 Table 5                                                                       Table 6
                 Ablation study on diﬀerent modules of the proposed                            Mean and standard deviation of CMC Rank-1 ac-
                 method on the Market-1501 dataset in terms of CMC                             curacy and mAP (%) across ﬁve independent tri-
                 (%) and mAP (%).                                                              als on four public Re-ID datasets (Market-1501,
                                                                                               DukeMTMC-reID, CUHK03-NP, and Occluded-Duke).
                                                   Market-1501
                  Model                                                                          Dataset                 mAP (%)       Rank-1 (%)
                                                   mAP     Rank-1
                                                                                                 Market-1501             91.8 ± 0.1    96.9 ± 0.1
                  Baseline                         85.9    94.5
                                                                                                 DukeMTMC-reID           83.8 ± 0.1    90.8 ± 0.2
                  +SGOS                            87.2    95.0
                                                                                                 CUHK03-NP(Detected)     82.3 ± 0.1    85.0 ± 0.3
                  +LFSE                            88.2    95.5
                                                                                                 CUHK03-NP(Labeled)      85.5 ± 0.2    86.0 ± 0.2
                  +LFSE+SGOS+LOSS(FOSENet)         91.8    96.9
                                                                                                 Occluded-Duke           61.5 ± 0.2    68.7 ± 0.1




4.3. Ablation studies                                                              based occlusion simulation method and local feature semantic extension
                                                                                   simulation method. In addition, we have joined the proposed cross-bias
   In this subsection, we show the ablation experimental results of our            loss function.
proposed FOSENet to demonstrate the eﬀectiveness of each part of our                   1) Eﬀectiveness of the Model. As shown in Table 5, our method
proposed method. In Table 5, the Baseline denotes the underlying back-             achieves 87.2 % mAP accuracy and 95.0 % Rank-1, which signiﬁ-
bone network, SGOS and LFSE sub-tables denote our proposed semantic-               cantly outperforms the baseline by incorporating the semantic based

                                                                               9
Z. Deng et al.                                                                                                                    Pattern Recognition 172 (2026) 112567




Fig. 9. Top-5 retrieval results on the DukeMTMC-reID dataset using Baseline and our proposed FOSENet. Correct matches are marked with green borders, while
incorrect ones are marked with red. (For interpretation of the references to colour in this ﬁgure legend, the reader is referred to the web version of this article.)


occlusion simulation (SGOS) method. Moreover, by considering the pro-                 datasets. The addition of 𝐿𝐶𝐶𝑂 enhances the model’s recognition ability
posed local feature semantic extension module (LFSE) method, 88.2 %                   for diﬀerent situations by separating the clustering of samples with dif-
mAP accuracy and 95.5 % Rank-1 accuracy can be reached. Finally, by                   ferent distances and widening the interclass distances, which helps the
using all components, FOSENet can obtain 96.9 % Rank-1 accuracy and                   model to extract the features eﬃciently.
91.8 % mAP accuracy.
    2) Experimental Data Stability Analysis. As shown in Table 6, the pro-
                                                                                      4.4. Visualization analysis
posed method obtains state-of-the-art performance in terms of mAP and
Rank-1 accuracy on four benchmark datasets and the standard devia-
                                                                                            1) Attention pattern visualization. We use an attention heat map to
tions are within ±0.3. The experiment results indicate that the proposed
                                                                                      represent the regions of interest learned by our model, and color diﬀer-
method reaches impressive stability and consistency under repeated tri-
                                                                                      ences in the labeled regions can show the ﬁnal feature representation,
als. Notably, the performance remains robust even on the heavily oc-
                                                                                      where regions with brighter colors mean that the model pays more at-
cluded Occluded-Duke dataset, demonstrating the eﬀectiveness of our
                                                                                      tentions. As Fig. 8 shows the visualization of the ﬁnal generated fea-
approach under challenging scenarios. The reported results represent
                                                                                      tures. Obviously, the feature map from Baseline shows little attention
the mean and standard deviation calculated from ﬁve independent runs
                                                                                      to pedestrians in the attention heat map. Moreover, it can be observed
under identical experimental settings, ensuring both the reliability and
                                                                                      that the baseline model often focuses on some redundant regions, such
reproducibility of the evaluation.
                                                                                      as background walls or ground surfaces, which are not informative for
    3) Feature distribution. As shown in Fig. 5 . We present visualiza-
                                                                                      identity discrimination. In contrast, our proposed FOSENet model can
tions of the feature distributions optimized by the our method in the
                                                                                      better focus on the local semantic regions of the target pedestrians to ef-
visible modality. It is observed that our proposed method is signiﬁcantly
                                                                                      fectively capture the discriminative features of the person, and suppress-
better than the backbone network.
                                                                                      ing the background interference information around the target person.
    4) Convergence of Accuracy Over Epochs. As shown in Fig. 6 . We pro-
                                                                                      Even when most parts are occluded by other pedestrians, our model can
vide the recognition accuracy data (mAP,Rank-1∼10) of FOSENet in 120
                                                                                      still focus on the localized area of the target pedestrian, which demon-
training epochs. It is observed that the recognition accuracy increased
                                                                                      strates the eﬀectiveness of our proposed method.
rapidly in the early part of the training epoch. The accuracy stabilised
                                                                                           2) Search Results: To verify the retrieval performance improvement
in the middle and late part of the training epoch, and satisfactory results
                                                                                      reached by the proposed model, we compare the proposed FOSENet
were achieved. It is observed that our proposed method(FOSENet) has
                                                                                      model with the Baseline are used to indicate successful results, and red
good generalisation ability without overﬁtting or underﬁtting problems,
                                                                                      borders are used to indicate erroneous results. From the retrieval results
which proves the stability of FOSENet.
                                                                                      shown in Fig. 9, we can ﬁnd that a signiﬁcant improvement of the re-
    5) Evaluation of Parameters. As shown in Eq. (19), we introduced
                                                                                      trieval results is obtained by our FOSENet compared to the Baseline,
the adjustable parameter 𝜆 to balance the contribution of our proposed
                                                                                      which proves the eﬀectiveness of our proposed method.
loss function to the overall model, and thus we evaluated the impact on
the eﬀectiveness of recognition under diﬀerent parameters here. We per-
formed Rank-1 and mAP evaluations on the Market-1501 and Occluded-                    5. Conclusion
DukeMTMC datasets, respectively, and the results are shown in Fig. 7.
It is observed that mAP and Rank-1 reach the best when the parame-                        In this paper, a semantic-guided occlusion simulation based local
ter 𝜆 reaches 0.8, and then show a diﬀerent degree of decline on both                 feature semantic expansion network (FOSENet) is proposed to solve

                                                                                 10
Z. Deng et al.                                                                                                                                       Pattern Recognition 172 (2026) 112567


the problem of limited local semantic information representation abil-                             [3] G. Chen, T. Gu, J. Lu, J.-A. Bao, J. Zhou, Person re-identiﬁcation via attention pyra-
ity caused by semantic occlusion in real-world scenarios. FOSENet in-                                  mid, IEEE Trans. Image Process. 30 (2021) 7663–7676.
                                                                                                   [4] D. Cheng, Y. Gong, S. Zhou, J. Wang, N. Zheng, Person re-identiﬁcation by
tegrates the occluded regions with the semantic information of pedes-                                  multi-channel parts-based cnn with improved triplet loss function, in: Proceed-
trians to systematically generate targeted pedestrian occlusion samples,                               ings of the iEEE Conference on Computer Vision and Pattern Recognition, 2016,
thereby increasing the diversity of pedestrian samples and enhancing                                   pp. 1335–1344.
                                                                                                   [5] Z. Zhang, C. Lan, W. Zeng, X. Jin, Z. Chen, Relation-aware global attention for person
the adaptability of the model. Then, FOSENet has been used to ob-                                      re-identiﬁcation, in: Proceedings of the IEEE/CVF Conference on Computer Vision
tain ﬁne-grained regional semantic details to enhance the discrimina-                                  and Pattern Recognition, 2020, pp. 3186–3195.
tive ability of pedestrian features. In addition, to validate the eﬀective-                        [6] Z. Dai, M. Chen, X. Gu, S. Zhu, P. Tan, Batch dropblock network for person re-
                                                                                                       identiﬁcation and beyond, in: Proceedings of the IEEE/CVF International Conference
ness of our proposed FOSENet method and its components, we conduct                                     on Computer Vision, 2019, pp. 3691–3701.
experiments and obtained satisfactory results on four public person Re-                            [7] Y. Li, H. Yao, C. Xu, TEST: Triplet ensemble student-teacher model for unsupervised
ID datasets: Market1501, DukeMTMC-reID, CUHK 03-NP and Occluded-                                       person re-identiﬁcation, IEEE Trans. Image Process. 30 (2021) 7952–7963.
                                                                                                   [8] W. Xu, H. Liu, W. Shi, Z. Miao, Z. Lu, F. Chen, Adversarial feature disentanglement
DukeMTMC. The experiment results demonstrate the eﬀectiveness of the
                                                                                                       for long-term person re-identiﬁcation, in: IJCAI, 2021, pp. 1201–1207.
proposed method.                                                                                   [9] S. Lin, C.-T. Li, A.C. Kot, Multi-domain adversarial feature generalization for person
    Although signiﬁcant performance improvements have been                                             re-identiﬁcation, IEEE Trans. Image Process. 30 (2020) 1596–1607.
achieved, we claim that the proposed SGOS and LFSE have higher                                    [10] Y. Shi, H. Ling, L. Wu, B. Zhang, P. Li, Attribute disentanglement and reg-
                                                                                                       istration for occluded person re-identiﬁcation, Neurocomputing 470 (2022)
computational cost and training complexity due to their multi-stage                                    226–235.
processing pipeline and the inclusion of attention-based operations.                              [11] T. Wang, M. Liu, H. Liu, W. Li, M. Ban, T. Guo, Y. Li, Feature completion trans-
In other words, the implementation of these modules requires more                                      former for occluded person re-identiﬁcation, IEEE Trans. Multimedia 26 (2024)
                                                                                                       8529–8542.
computational resources and training time compared to the baseline.                               [12] Z. Zhong, L. Zheng, G. Kang, S. Li, Y. Yang, Random erasing data augmenta-
Moreover, under extreme occlusion or cluttered background condi-                                       tion, in: Proceedings of the AAAI Conference on Artiﬁcial Intelligence, 34, 2020,
tions, the consistency of semantic guidance may decrease, limiting the                                 pp. 13001–13008.
                                                                                                  [13] P. Chen, W. Liu, P. Dai, J. Liu, Q. Ye, M. Xu, Q. Chen, R. Ji, Occlude
eﬀectiveness of occlusion simulations. In the future, we will explore                                  them all: occlusion-aware attention network for occluded person re-id, in: Pro-
lightweight and adaptive occlusion simulation strategies and domain                                    ceedings of the IEEE/CVF International Conference on Computer Vision, 2021,
adaptation techniques for pedestrian feature extraction and semantic                                   pp. 11833–11842.
                                                                                                  [14] Z. Wang, F. Zhu, S. Tang, R. Zhao, L. He, J. Song, Feature erasing and diﬀusion net-
representation to further improve the generalization ability and
                                                                                                       work for occluded person re-identiﬁcation, in: Proceedings of the IEEE/CVF Confer-
eﬃciency of the proposed model under various real-world scenarios.                                     ence on Computer Vision and Pattern Recognition, 2022, pp. 4754–4763.
                                                                                                  [15] J. Miao, Y. Wu, P. Liu, Y. Ding, Y. Yang, Pose-guided feature alignment for occluded
CRediT authorship contribution statement                                                               person re-identiﬁcation, in: Proceedings of the IEEE/CVF International Conference
                                                                                                       on Computer Vision, 2019, pp. 542–551.
                                                                                                  [16] J. Yang, J. Zhang, F. Yu, X. Jiang, M. Zhang, X. Sun, Y.-C. Chen, W.-S. Zheng,
     Zelin Deng: Writing – review & editing, Supervision, Resources,                                   Learning to know where to see: a visibility-aware approach for occluded person
Project administration, Investigation, Funding acquisition, Conceptual-                                re-identiﬁcation, in: Proceedings of the IEEE/CVF International Conference on Com-
                                                                                                       puter Vision, 2021, pp. 11885–11894.
ization; Mingxuan Tang: Writing – review & editing, Writing – original                            [17] S. Wang, B. Huang, H. Li, G. Qi, D. Tao, Z. Yu, Key point-aware occlusion suppression
draft, Visualization, Validation, Software, Methodology, Formal analy-                                 and semantic alignment for occluded person re-identiﬁcation, Inf. Sci. 606 (2022)
sis, Data curation; Ke Nai: Writing – review & editing, Visualization, Su-                             669–687.
                                                                                                  [18] S. Gao, C. Yu, P. Zhang, H. Lu, Part representation learning with teacher-Student
pervision, Project administration, Methodology, Investigation, Formal
                                                                                                       decoder for occluded person re-Identiﬁcation, in: ICASSP 2024-2024 IEEE Interna-
analysis, Data curation; Guiji Li: Writing – review & editing, Visualiza-                              tional Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE, 2024,
tion, Validation, Data curation; Shaomiao Chen: Writing – review &                                     pp. 2660–2664.
                                                                                                  [19] G. Zhang, W. Lin, A. kumar Chandran, X. Jing, Inf. Sci. 633 Complementary networks
editing, Visualization, Resources, Project administration, Investigation;
                                                                                                       for person re-identiﬁcation, (2023) 70–84.
Pei He: Writing – review & editing, Resources, Project administration,                            [20] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, Q. Tian, Scalable person re-
Investigation, Data curation.                                                                          identiﬁcation: a benchmark, in: Proceedings of the IEEE International Conference
                                                                                                       on Computer Vision, 2015, pp. 1116–1124.
                                                                                                  [21] E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi, Performance measures and a
Data availability                                                                                      data set for multi-target, multi-camera tracking, in: European Conference on Com-
                                                                                                       puter Vision, Springer, 2016, pp. 17–35.
    Data will be made available on request.                                                       [22] W. Li, R. Zhao, T. Xiao, X. Wang, Deepreid: deep ﬁlter pairing neural network for
                                                                                                       person re-identiﬁcation, in: Proceedings of the IEEE Conference on Computer Vision
                                                                                                       and Pattern Recognition, 2014, pp. 152–159.
Declaration of competing interest                                                                 [23] Y. Sun, L. Zheng, Y. Yang, Q. Tian, S. Wang, Beyond part models: person retrieval
                                                                                                       with reﬁned part pooling (and a strong convolutional baseline), in: Proceedings of
    The authors declare that they have no known competing ﬁnancial                                     the European Conference on Computer Vision (ECCV), 2018, pp. 480–496.
                                                                                                  [24] J. Zhang, J.-P. Ainam, W. Song, L.-h. Zhao, X. Wang, H. Li, Learning global and local
interests or personal relationships that could have appeared to inﬂuence                               features using graph neural networks for person re-identiﬁcation, Signal Process.
the work reported in this paper                                                                        Image Commun. 107 (2022) 116744.
                                                                                                  [25] G. Zhang, P. Zhang, J. Qi, H. Lu, HAT: hierarchical aggregation transformers for
Acknowledgement                                                                                        person re-identiﬁcation, in: Proceedings of the 29th ACM International Conference
                                                                                                       on Multimedia, 2021, pp. 516–525.
                                                                                                  [26] L. Tan, J. Xia, W. Liu, P. Dai, Y. Wu, L. Cao, Occluded person re-identiﬁcation via
   This work was supported in part by the National Natural Science                                     saliency-Guided patch transfer, in: Proceedings of the AAAI Conference on Artiﬁcial
Foundation of China under Grant No. 61977018 and No. 62106071;                                         Intelligence, 38, 2024, pp. 5070–5078.
                                                                                                  [27] H. Huang, D. Li, Z. Zhang, X. Chen, K. Huang, Adversarially occluded samples for
Natural Science Foundation of Changsha under Grant No. kq2202215;                                      person re-identiﬁcation, in: Proceedings of the IEEE Conference on Computer Vision
Practical Innovation and Entrepreneurship Enhancement Program for                                      and Pattern Recognition, 2018, pp. 5098–5107.
Professional Degree Postgraduates of Changsha University of Science                               [28] C. Zhao, X. Lv, S. Dou, S. Zhang, J. Wu, L. Wang, Incremental generative occlu-
                                                                                                       sion adversarial suppression network for person reID, IEEE Trans. Image Process.
and Technology(CLSJCX25072).                                                                           30 (2021) 4212–4224.
                                                                                                  [29] C. Yan, G. Pang, J. Jiao, X. Bai, X. Feng, C. Shen, Occluded person re-identiﬁcation
References                                                                                             with single-scale global representations, in: Proceedings of the IEEE/CVF Interna-
                                                                                                       tional Conference on Computer Vision, 2021, pp. 11875–11884.
 [1] M. Koestinger, M. Hirzer, P. Wohlhart, P.M. Roth, H. Bischof, Large scale metric             [30] M. Jia, X. Cheng, Y. Zhai, S. Lu, S. Ma, Y. Tian, J. Zhang, Matching on sets: conquer
     learning from equivalence constraints, in: 2012 IEEE Conference on Computer Vision                occluded person re-identiﬁcation without alignment, in: Proceedings of the AAAI
     and Pattern Recognition, IEEE, 2012, pp. 2288–2295.                                               Conference on Artiﬁcial Intelligence, 35, 2021, pp. 1673–1681.
 [2] H. Dong, P. Lu, S. Zhong, C. Liu, Y. Ji, S. Gong, Person re-identiﬁcation by enhanced        [31] V. Somers, C. De Vleeschouwer, A. Alahi, Body part-based representation learning
     local maximal occurrence representation and generalized similarity metric learning,               for occluded person re-identiﬁcation, in: Proceedings of the IEEE/CVF Winter Con-
     Neurocomputing 307 (2018) 25–37.                                                                  ference on Applications of Computer Vision, 2023, pp. 1613–1623.


                                                                                             11

exec
/bin/zsh -lc "pdftotext \"A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf\" - | rg -n -C 3 \"Ablation|ablation|SGOS|LFSE|cross|offset|occlusion|semantic|visual|Fig\\. 5|heatmap|t-SNE|Grad|effectiveness|component\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5-Pattern Recognition
6-journal homepage: www.elsevier.com/locate/pr
7-
8:A Semantic-guided occlusion simulation based local feature semantic
9-expansion network for person re-identiﬁcation
10-Zelin Deng a , Mingxuan Tang a , Ke Nai
11-a
--
27-
28-Keywords:
29-Person re-identiﬁcation
30:Local semantic features
31-Occlusion simulation
32-
33-a b s t r a c t
34:Person re-identiﬁcation is a challenging task which aims to retrieve images of a target pedestrian across diﬀerent
35:cameras. Currently, existing models generally neglect the semantic relevance of local features, which may cause
36:to limited identiﬁcation performance. To address this issue, we propose a semantic-guided occlusion simulation
37:based local feature semantic expansion network (FOSENet) to pursue satisfactory performance. Firstly, we propose a semantic-guided occlusion simulation (SGOS) method, which generates semantically relevant occlusion
38:patches and further simulates the occlusion based on the semantic information of pedestrians to improve the diversity of the occluded pedestrian samples. Then, we propose a local feature semantic expansion (LFSE) method,
39-which obtains several local areas around each key area and selects some useful local areas as additional cues of
40:the key area to enhance the discrimination ability. Finally, we introduce a cross-center oﬀset loss function, which
41-enlarges the distance of a pair of nearest neighbor samples from two classes to optimize the model. Thus, the
42-samples belonging to diﬀerent identities are pushed further away. Extensive experiments are conducted on four
43-challenging datasets, and the proposed method can achieve competitive results compared to multiple state-ofthe-art works. The code is publicly available at https://github.com/xuanthan-art/code-with-paper-FOSENet.git
44-
45-1. Introduction
46-Person re-identiﬁcation aims to determine whether images captured by diﬀerent cameras contain the same individual by retrieving
47:pedestrian information across these cameras, which has been widely
48-used in artiﬁcial intelligence and security monitoring [1,2]. With the
49-rapid development of deep learning and the release of pedestrians reidentiﬁcation (Re-ID) image datasets, many powerful Re-ID methods
50:have been proposed to pursue impressive performance, such as multiscale methods [3,4], ﬁne-grained methods [5,6] and adversarial network learning [7–9]. However, person re-identiﬁcation still faces multiple challenges: (1) occlusion. In real-world environments like train
51-stations, airports, and shopping malls, pedestrian images captured by
52:stationary surveillance cameras often suﬀer from occlusion caused by
53:vegetation, vehicles, and other individuals. The occlusion may cause
54-a pedestrian to exhibit signiﬁcant appearance changes under diﬀerent
55-environments, which greatly limits the re-identiﬁcation accuracy [17].
56:(2) local features semantic representation.It is widely acknowledged
57-that CNN features are extracted from pre-trained models, as the depth
58-
59-of the layers increases, CNN features tend to overly concentrate on a
60-single local center, which may lead to the neglect of other important
61:details. As a result, the semantic discriminability of the extracted features is likely to decrease, potentially aﬀecting the overall performance
62-of the model. In particular, some regions in pedestrian images, such
63:as ground surfaces, walls, and background objects, are repeatedly extracted, thereby generating redundant semantic information that is frequently attended by the model but does not contribute to identity discrimination or occlusion interference representations. These redundant
64:semantics may mislead feature learning and increase similarity across
65:diﬀerent identities. Therefore, how to remove the interfering semantic information and obtain more additional semantic information to enhance the feature representations of pedestrian identity is a challenge for
66-person re-identiﬁcation [10,11]. To address these issues, existing methods mainly focus on enhancing the robustness of the model to deal with
67:occluded samples and the extraction of semantic information from local
68-features.
69:On the one hand, to solve the occlusion problem, many methods
70-try to enhance the robustness of the model by simulating real-world
71-
72-∗ Corresponding author.
--
82-
83-Z. Deng et al.
84-
85:occlusions through random erasure of image regions [12,13]. Then,
86:to make the occlusion patches more realistic for simulated occlusion,
87:Chen et al.[13] proposed a method to crop other images to create occlusion patches, and these patches were used at conventional occlusion
88-locations (such as top, bottom, left, and right) in pedestrian images to
89:enhance the robustness of the model to recognize occlusion situations.
90:Subsequently, in order to better model the feature extraction during occlusion, some works have proposed feature erasure [6]. Wang et al. [14]
91-introduced a feature erasure and diﬀusion network, which generates an
92:occlusion mask to guide the model to erase the feature map. As a result,
93-the model can focus on unobstructed regions and diﬀuse the pedestrian
94:representation in the feature space to boost the identiﬁcation capability of the target pedestrian. However, existing occlusion simulation or
95-erasure methods are positionally arbitrary and heavily dependent on
96:manual design. These methods fail to adequately simulate the occlusions that models encounter in real-world scenarios, where pedestrians
97:are frequently partially obscured by irregular objects. Moreover, the positions of such occlusions tend to interfere with critical semantic information. Thus, feature extraction and semantic recognition in diﬀerent
98:occlusion scenarios become more diﬃcult, and the model may focus on
99-the useless feature areas, leading to a signiﬁcant performance degradation in recognition accuracy.
100:On the other hand, to extract more semantic information, some methods employ auxiliary models to acquire additional semantic information
101:to address the issue of limited representation ability in local feature semantics [15]. Yang et al. [16] proposed to introduce discretizing posture
102-information into visibility labels of body parts to suppress the impact
103-of interfering information, thereby obtaining more posture information
104:from visible areas as additional semantic cues to enhance the recognition eﬃciency under various scenes. Subsequently, to extract more
105:precise semantic information, some methods employ multi-scale information to learn additional semantic information to improve the feature
106-representation ability of pedestrian images [17]. Gao et al.[18] proposed
107-a teacher-student ecoder (TSD) framework to aggregate pedestrian area
108:information through feature distillation and a mask generator was designed to provide discriminative regions for better re-identiﬁcation performance. However, the extraction of semantic information heavily relies on prior training outcomes. To extract more semantic information
109-that is not focused on by pre-trained models, Zhang et al. [19] introduced a complementary network for person re-identiﬁcation, which utilizes two branches to independently extract potentially discriminative
110:information that is beneﬁcial for pedestrian recognition from both background and salient regions across the global scope. The proposed network can eﬀectively capture additional semantic information such as
111-backpacks and handbags to improve the identiﬁcation performance in
112:pedestrian recognition tasks. However, the loss of partially discriminative features caused by real-world occlusions ampliﬁes the inﬂuence of
113-unoccluded key regions in recognition decisions, which may increase
114-the similarity between diﬀerent pedestrians and reduce the recognition
115-accuracy of the model. Meanwhile, existing methods rely heavily on the
116-unoccluded key regions, making them highly susceptible to misidentiﬁcation.
117:Based on the above analysis, we propose a semantic-guided occlusion
118:simulation based local feature semantic expansion network. Firstly, we
119:try to exploit semantic information to guide the simulation of occlusions to generate semantically occlusion patches. Specially, occlusion
120-operations are performed based on their position, shape, and context to
121:ensure that the generated occlusions are more consistent with the actual content of the images to better simulate real-world scenarios where
122:pedestrians are obstructed. Subsequently, to obtain richer semantic information for each individual pedestrian, some local areas near the key
123:area are selected, which are treated as additional cues to enhance semantic diversity. Finally, we introduce a cross-center oﬀset loss function, which enlarges the distance of a pair of nearest neighbor samples
124-from two classes during the training process to ensure that the samples
125-belonging to diﬀerent identities are pushed further away. We conducted
126-
127:extensive experiments on four challenging datasets to validate the eﬀectiveness of our proposed method. Notably, the proposed network generates semantically relevant and adaptive occlusion patches to simulate
128:realistic occlusion scenarios, which achieves better adaptability to deal
129:with complex visual interference. In addition, by mining local regions
130-with rich identity-related cues, it can obtain better discriminative ability
131:to perform person re-identiﬁcation under occlusion scenes.
132-The main contributions can be summarized as follows:
133-•
134-
135:We propose a semantic-guide occlusion simulation method, which
136:generates semantically relevant occlusion patches with semantic information of object pedestrians and further realizes occlusion simulation based on their position, shape, and context to generate more
137-diverse occluded pedestrian samples.
138:• We develope a local feature semantic extension method, which selects some useful local regions near the key areas to obtain additional
139:semantic information, thereby enhancing the discriminative ability
140-of pedestrian features.
141:• We introduce a cross-center oﬀset loss function, which increases the
142-distance between a pair of nearest neighbor samples from two classes
143-to optimize the model and enhance its generalization ability.
144-• We conduct extensive experiments to demonstrate the eﬀectiveness of our proposed method, which achieves promising performance on four large-scale benchmark datasets: Market-1501
--
148-person re-identiﬁcation, occluded person re-identiﬁcation, and partial
149-person re-identiﬁcation.
150-2.1. Person re-identiﬁcation
151:Pedestrian re-identiﬁcation is a crucial computer vision task that enables the re-identiﬁcation of the same individual across diﬀerent scenes
152-by analyzing and comparing images of pedestrians, and its research and
153-applications have obtained substantial progress due to the advancement
154-of convolutional neural networks [1,2] . One core challenge of person
155:re-identiﬁcation is how to accurately identify the same pedestrian under varying lighting scenes, viewpoints, and occlusions [23] Typically,
156:existing pedestrian re-identiﬁcation methods attempt to learn more reﬁned local feature semantic information to achieve ﬁne-grained feature matching [5]. Zhang et al.[24] proposed a method that leverages
157-graph neural networks to learn global and local features. The proposed
158:method can utilize the characteristics of graph neural networks to acquire more locally joint semantic information. Subsequently, to enhance
159:the discriminative ability across diﬀerent regions, some methods have
160-employed attention mechanisms to eﬀectively focus on pedestrian areas.
161-Chen et al. [3] proposed an attention pyramid method that employs attention regions in a multi-scale way to learn corresponding attention at
162-various ﬁne-grained levels to learn beneﬁcial human foreground areas,
--
164-Additionally, some methods enrich the diversity of features by extracting them from pedestrian images in a region speciﬁc or layered manner
165-[23]. Zhang et al. [25] proposed a Hierarchical Aggregation Transformer
166-method, which aggregates diﬀerent feature maps with the same Transformer and further calibrates the features with another Transformer
167:to enrich high-level semantic information to capture a more comprehensive set of global and local information. However, these methods
168:overlook the occlusion issue and fail to separate individuals from obstructions when interfering elements are semantically similar to the
169-pedestrian information, which will signiﬁcantly reduce the recognition
170-ability.
171-2
--
182-al. [6] proposed the Batch DropBlock Network (BDB), which erases the
183-same area of images in a batch to concentrate on other areas and learn
184-richer local information. Subsequently, to better learn the local features
185:of pedestrians, Zhou et al. [32] simulated occlusions and introduced a
186-co-saliency branch to learn those parts of the pedestrians that are not
187-occluded. Additionally, Dong et al. [33] generated adversarial samples
188-by randomly erasing, transforming, and adding noise to feature maps
--
190-Recently, Yan et al. [34] leveraged the correlation between local features for aggregation and employed a reﬁned partitioning strategy to
191-concentrate and learn the visible areas more eﬀectively. Diﬀerent from
192-the above methods, Huang et al. [35] performed horizontal strip partitioning at various scales to better capture the representable information
193:of diﬀerent local parts of pedestrians to eﬀectively learn semantic information from local to global levels. To address the issue of partial occlusion, we propose selectively incorporating the features of adjacent local
194-regions of important local regions as supplementary information into
195-pedestrian features,to improve the discriminative ability to distinguish
196-pedestrians in locally similar scenarios.
197-
198:In real-world scenarios, due to the ﬁxed positions of cameras, occlusions are inevitable when pedestrians are captured, which greatly
199-limits the re-identiﬁcation performance [26]. To address the issues arising from occluded pedestrians, some works have focused on simulating
200:occlusion scenarios to enhance the robustness against occlusions [12].
201-Huang et al [27]. proposed an Ad-Occluded Samples method, which
202:introduces deliberately designed occlusions into the training data and
203:trains the model after identifying key areas to better cope with realworld occlusion situations. Subsequently, to enable the model to better
204:learn with occlusions under diﬀerent scenes, Zhao et al [28]. proposed
205-an Incremental Generative Occlusion Adversarial Suppression Network,
206:which simulates occlusions from easy to diﬃcult to make the model
207:more robust to occlusions. The CBE module [29], which stands for Composite Batch Erasing, combines Batch-level Erasing (BCE) with Imagelevel Erasing (RE). These two erasing operations are applied to the original pedestrian images within the same batch to obtain two batches
208-of augmented images that are merged into a larger batch for training,
209:thereby enabling the model to be learned under occlusion scenes which
210-aﬀect diﬀerent body parts and of varying sizes. Some other methods
211:focus on directing the attention to visible areas to suppress the interference caused by occlusions. Jia et al. [30]proposed a set-matching
212-method without a collection, which uses a set matching approach to
213-concentrate on visible regions of the model. Subsequently, Wang et al.
214-[17] proposed a Key Point-Aware Occlusion Suppression and Semantic
215-Alignment (POS) method, which focuses on the unobstructed regions of
216:pedestrians to achieve semantic alignment to eﬀectively extract semantic information from the visible areas. However, these methods still have
217:some drawbacks. Firstly, their simulation strategy shows poor adaptability to the interference caused by occlusion in diﬀerent scenarios. Moreover, in real-world scenarios, they are unable to handle occlusions of
218:various shapes and may mistake the semantic information of occluding
219-objects for that of pedestrian features, which signiﬁcantly aﬀects the
220-recognition performance. In our proposed approach, we aim to more
221:eﬀectively simulate a diverse range of occlusion scenarios in order to
222:signiﬁcantly enhance recognition performance under various occlusion
223-interferences.
224-
225-3. Methodology
226-3.1. Overview of the proposed method
227-The overall network structure proposed in this paper is shown in
228:Fig. 1. We propose a semantic-guided occlusion simulation based local
229:feature semantic expansion network (FOSENet) to solve the problem
230:of limited local semantic information representation ability caused by
231:semantic occlusion in real-world scenarios. Firstly, FOSENet integrates
232:occlusion patches with the semantic information of pedestrians to generate semantically relevant occlusion patches and further simulates the
233:occlusion with the generated occlusion patches and the location, shape,
234:and context of the semantic information of pedestrians to improve the
235-samples. Then, ResNet-50 is employed as a pretrained extractor to extract features from pedestrian images and obtain the attention weights
236-for each channel through an attention module to capture the key areas
237-and surrounding local areas. To ensure that the key areas have more
238-individual distinctiveness for representing pedestrian information, we
239:propose a local feature semantic expansion (LFSE) method. Speciﬁcally,
240:LFSE sorts the surrounding local areas of each key area extracted by the
241-attention module to select several local areas and fuses them with the
242-
243-2.3. Partial person re-identiﬁcation
244:Due to issues such as occlusion, lighting, and changes in viewing
245-angle, the representable information of pedestrian images is limited,
246-which means useful information only partially exists on pedestrian im-
247-
248:Fig. 1. Overall architecture of FOSENet, FOSENet mainly contains SGOS, LFSE, cross-center oﬀset loss function. Then, ResNet-50 is used as a backbone network.
249:FOSENet is designed to enhance model adaptability under occlusion conditions by generating diverse occlusion samples and mining local features with stronger
250-identity-discriminative ability.
251-3
252-
--
257-Table 1
258-The key notations of the proposed method.
259-
260:key area to enhance the semantic richness of the feature areas. Finally,
261:we introduce a cross-center oﬀset loss function, which enlarges the distance of a pair of nearest neighbor samples from two classes to optimize
262-the model. Some key notations of the proposed method are listed in
263-Table 1.
264:3.2. Semantic-guided occlusion simulation (SGOS)
265:Existing occlusion simulation methods fail to adapt to various realworld occlusion scenarios as the occlusion areas are determined based
266-on randomness or pixel thresholds and the generated samples can hardly
267:achieve eﬀective occlusion simulation. However, occlusions also contain
268:certain semantic information rather than a random blank space, and this
269:semantic information can often interfere with the model. We propose a
270:semantic-guided occlusion simulation (SGOS) method to better simulate
271-highly complicated environments to improve the recognition ability of
272:the model. SGOS fuses the pedestrian feature with the occlusion set to
273:generate a semantic related occlusion patch, which ensures the semantic consistency between the occlusion patch and the pedestrian image
274:occluded with the location, shape, and context, of the semantic information of the pedestrian.
275:Speciﬁcally, we utilize the semantic diﬀerence extracted from the input image in diﬀerent convolutional layers to guide the occlusion simulation, which is divided into two phases, as illustrated in Fig. 2 . The ﬁrst
276:phase involves the extraction of semantic information from the image,
277:which is obtained from the positions and semantic information occluded
278:in the image. The second phase is occlusion simulation which fuses the
279-
280-Notation
281-
--
296-
297-Semantic diﬀerential features
298-Convolution feature map
299:Semantic occlusion patch set
300-Importance ranking set of surrounding area
301-Diﬀerential weight set
302-Semantic extension features
303:Additional semantic features
304-Sample anchor
305-Cross-center oﬀset loss
306-
307:occlusion patch set with the semantic information patch extracted in
308:the ﬁrst phase. Then, we further simulate the occlusion with the fused
309:occlusion patches by applying them to the semantic positions and environments for occlusion in the image to enhance the diversity of samples.
310:In the ﬁrst phase, we need to acquire the semantic information from
311-the output feature maps in adjacent convolutional layers. We represent
312-the input image as X, and the feature maps 𝐅𝑖𝑐 = {𝑐𝑛𝑖 , 𝑛 = 1, … , 𝑁}, 𝐅𝑖+1
313-=
--
320-where the convolutional network follows the same structure as the pretrained ResNet-50, 𝑁 represents the maximum number of channels and
321-1∼i indicates the number of layers used in the feature extraction process. Moreover, we utilize the degree of attention to the regions of in-
322-
323:Fig. 2. The motivation behind the SGOS module is to generate semantically meaningful occlusion patches for more target simulation of real-world occlusion scenarios.
324:To achieve this goal, we integrate the semantic diﬀerence information extracted from the input image with a predeﬁned set of occlusion patches, and guide the
325:placement and fusion process using the semantic diﬀerence map. To illustrate this process more clearly, we provide intermediate examples for each sub-stage in the
326:ﬁgure. Obviously, the occlusion samples generated by the model better present the real occlusion scenarios.
327-4
328-
329-Pattern Recognition 172 (2026) 112567
--
334-, which maximizes
335-the channel diﬀerences to extract the key information from the image
336-𝑖 and
337:but remove redundant semantic information. Then, we take 𝐹𝑎𝑡𝑡
338-𝑖+1
339-𝑖
340:𝐹𝑎𝑡𝑡 as inputs to obtain the semantic diﬀerential features 𝐹𝐷𝑖𝑓 𝑓 after
341-the channel diﬀerentiation operations. The process can be represented
342-as:
343-(
--
357-(3)
358-
359-is sampled from locations that exceed a predeﬁned threshold in 𝐹𝐷𝑖𝑓 𝑓 ,
360:which ensures semantic relevance during the occlusion process.
361:Compared to traditional random occlusion strategies such as random erasing, which mask rectangular regions in a semantically agnostic
362:and unguided manner, our proposed SGOS method introduces semantic
363:occlusion blocks derived from real-world object shapes to construct occlusion augmentation samples that better resemble actual scenarios. By
364:guiding both the location and shape of the occlusion patches, SGOS can
365:eﬀectively generate diverse occluded pedestrian samples, thereby enhancing the model’s adaptability under complex occlusion conditions.
366-
367-(4)
368-
369:3.3. Local feature semantic expansion (LFSE)
370-
371-𝑛=1
372-
--
383-is the mapping function of the attention model, and 𝛂
384-denotes the
385-attention weight corresponding to the channel at layers i and i+1. The
386:semantic diﬀerential features aims to obtain a degree of attention for the
387-regions of interest in the two feature maps. Finally, the | ⋅ | represents
388:the diﬀerentiation operation to extract the semantic diﬀerential features
389-𝑖
390-𝐹𝐷𝑖𝑓
391:. Subsequently, we normalize the obtained semantic diﬀerential
392-𝑓
393-𝑖
394-features 𝐹𝐷𝑖𝑓
--
403-𝑓 = 𝛾𝑖 √
404-𝜎𝐹2 + 𝜖
405-
406:Generally, as current semantic information extraction methods
407:overly focus on local areas, the obtained semantic information is too singular to capture the unique identity information of the pedestrian and
408-exhibits a certain degree of similarity among pedestrians. Moreover, the
409:existence of occlusions can lead to disturbances in local regions, thereby
410:limiting the representation of local semantic information. To enhance
411:the identity-representative capability of the extracted features, we propose a method called Local Feature Semantic Extension (LFSE). LFSE
412:incorporates the semantic information from regions adjacent to the key
413:regions as supplementary data and exploits the semantic continuity between these surrounding regions and the key regions, thereby strengthening the discriminative power of pedestrian features. This process is
414:visually demonstrated in Fig. 3.
415:The proposed LFSE utilizes the spatial and channel attention mechanisms to obtain the channels of the key area and the local areas near the
416-key area. We denote the input feature as Y and the channels 𝐹𝑖 and 𝐹𝑖′
417-of the key area and neighboring local areas can be obtained as follows:
418-
--
424-In the second phase, we integrate the diﬀerential features 𝐹𝐷𝑖𝑓
425-𝑓
426-which have been normalized to a standard normal distribution in the
427:ﬁrst phase with the occlusion patch shape sett 𝐌𝑠 = {𝑠𝑛 , 𝑛 = 1, … , 𝑁}
428:to generate a collection of semantically enriched occlusion patches
429-𝐌𝑖𝐵 = {𝑚𝑛 , 𝑛 = 1, … , 𝑁}. The process is formulated as:
430-}𝑁
431-𝑖
--
449-
450-(6)
451-
452:where ⊕ denotes an operator to fuse the semantic features and a randomly selected shape element from the shape set 𝑀𝑠 . The generated
453:semantic information patch set 𝑀𝐵𝑖 is used to occlude the input image
454:X with the positions, Speciﬁcally, the placement of occlusion blocks is
455:guided by the semantic diﬀerence map 𝐹𝐷𝑖𝑓 𝑓 , where the center position
456-
457:Fig. 3. The LFSE module aims to enhance the identity-discriminative capability of local pedestrian features by treating the semantic information obtained from
458:regions adjacent to key areas as additional semantic cues. To better illustrate the rationale and structural logic of LFSE, we provide heatmap visualizations of both
459:the key local regions and their semantically adjacent areas.
460-5
461-
462-Pattern Recognition 172 (2026) 112567
--
484-parameter weights for learning, 𝛿 is the ReLu activation function, and
485-𝑠𝑜𝑟𝑡(⋅) denotes to sort channels with the weight 𝑊̂ .
486-It is worth noting that the sorted channel set will contain duplicated
487:elements and interference elements caused by occlusions. To address
488:this issue, we need to select channels with more semantic information
489:and are more beneﬁcial for the semantic expansion by considering the
490-diﬀerentiation between diﬀerent channels. Speciﬁcally, we input the
491-′
492-feature channel set 𝐹𝑠𝑜𝑟𝑡
--
586-𝑡𝑗 ∈𝐶𝐵
587-
588-Next, we calculate the distance between the corresponding sample
589:anchor and its relative sample by a crossover way to adjust the distance
590-according to position relation of the center 𝐶𝐴 and the samples 𝑡𝑎 and
591-𝑡𝑏 . The speciﬁc loss function 𝐿𝐶𝐶𝑂 can be deﬁned as:
592-
--
596-′
597-where the 𝛼, 𝛽, 𝑦̈ are learnable hyperparameters, 𝐹𝑠𝑒𝑙𝑒𝑐𝑡
598-is the channel
599:feature that is selected and fused as additional semantic information.
600-′
601-Finally, we fuse the channel of the key area and 𝐹𝑠𝑒𝑙𝑒𝑐𝑡
602-to obtain the
603:ﬁnal semantic extension feature 𝐹exp as follows:
604-′
605-𝐹𝑒𝑥𝑝 = 𝐹𝑖 ⊕ 𝐹𝑠𝑒𝑙𝑒𝑐𝑡
606-,
--
622-(15)
623-
624-where ⊕ denotes the fusion operation.
625:Existing local semantic feature extraction methods under occlusion
626-typically attempt to enhance feature representations by incorporating
627:additional semantic information to deal with occlusions. However, such
628-approaches primarily focus on key regions, which leads to the overemphasis of key regions during the re-identiﬁcation process, thereby
629-misleading the model toward incorrect predictions. In contrast, the
630:proposed LFSE module adopts a semantic structure-guided strategy to
631-expand local features in regions adjacent to identity-relevant areas.
632:This design conducts the mining of local features based on semantic
633:relevance and structural consistency, eﬀectively improving the discriminability of occluded pedestrian features. As a result, the model demonstrates greater robustness and recognition accuracy under occlusion scenarios.
634-
635-𝐿 = 𝜆𝐿𝐶𝐶𝑂 + 𝐿𝐶𝑒𝑛𝑡𝑒𝑟 + 𝐿𝑡𝑟𝑖 ,
636-
--
640-terms.
641-3.5. Discuss
642-To make the model more suitable for actual scenarios, many works
643:have been proposed based on occlusion simulation to solve the occlusion problem to enhance the recognition ability of the model. However,
644:existing methods generally use occlusion patches with semantic information for occlusion simulations at random or pre-determined positions
645:[13,14,29] . Contrarily, our proposed occlusion simulation method employs the occlusion patches with some semantic information to better
646:simulate the information of occluded objects and the occlusion position is guided by the semantic information in the image to generate occluded pedestrian samples with better diversity. Secondly, to improve
647-
648-3.4. Cross center oﬀset loss
649-To enable the model to better choose centroids in the sample space
--
662-The symbol “ - ” indicates that no corresponding value was provided in the
663-corresponding paper.
664-
665:the local features semantic information representative ability under occlusion, many works use assistant models to obtain some additional semantic information [11,19,36] . Specially, our method fuses the key
666:area and several neighboring local areas to generate semantic features
667-with pedestrian identity information, which enhance the discriminability of occluded pedestrian features. For the loss function, we enlarge the
668-distance of a pair of nearest neighbor samples from two classes, which
669-makes the samples belonging to diﬀerent identities are pushed further
--
673-
674-4. EXPERIMENTAL ANALYSIS
675-In this section, to validate the eﬀectiveness of our proposed FOSENet
676:method and its components, we conduct experiments on four public person re-ID datasets: Market1501 [20] , DukeMTMC-reID [21] , CUHK
677-03-NP [22] and Occluded-DukeMTMC [15] .
678-4.1. Experimental settings
679-1)
--
1047-
1048-Labeled
1049-
1050:As show in Eq. (6), we introduce a shape set 𝑀𝑠 to simulate realworld occlusion scenarios during training and enhance model robustness. These occlusion blocks are derived from segmentation masks of
1051-non-human categories in the COCO dataset, such as backpacks, handbags, and ﬁre hydrants, which reﬂect common real-world occluders
1052-[39]. The selected masks are resized as shape templates and then fused
1053:with semantic patches extracted from high-response regions, guided by
1054:the semantic diﬀerence map 𝐹𝐷𝑖𝑓 𝑓 . The placement centers of the occlusion blocks are sampled from regions in 𝐹𝐷𝑖𝑓 𝑓 that exceed a predeﬁned threshold, ensuring that the occlusion process is semantically
1055:guided. Moreover, to ensure reproducibility and the rigor of experimental comparisons, a ﬁxed random seed is used across all training runs,
1056-and the training set is reshuﬄed at the beginning of each epoch. The
1057:occlusion simulation ratio is set to 0.35, which falls within the typical
1058-7
1059-
1060-Pattern Recognition 172 (2026) 112567
1061-
1062-Z. Deng et al.
1063-
1064:Fig. 5. Feature distribution with t-SNE on the Market-1501 dataset. (A) feature distribution learned by the Backbone network; (B) distribution under the Baseline
1065-model; (C) results from the proposed FOSENet.
1066-
1067-Table 4
--
1198-85.5 % and a Rank-1 score of 86.0 % on CUHK 03-Labeled labeled by a
1199-human, which outperforms other state-of-the-art methods.
1200-3) Comparisons on Occluded-DukeMTMC. Occluded-DukeMTMC, as
1201:a dataset evolved from DukeMTMC, is more challenging in the occlusion
1202-aspect. As shown in Table 4, FOSENet still obtains satisfactory results
1203-and achieves a mAP score of 61.5 % and a Rank-1 score of 68.7 %. Although FOSENet can not achieve a best Rank-1 score, the mAP score
1204-of FOSENet signiﬁcantly surpasses other methods, which proves the robustness as well as the superiority of our proposed method under the
1205:occlusion situations.
1206-
1207-Fig. 6. Convergence of CMC (%) and mAP (%) accuracy over epochs on market1501 and occluded-DukeMTMC datasets.
1208-
1209:range (0.3 ∼ 0.5) used in existing occlusion-based augmentation methods [12]. Given the structured and semantically guided nature of our
1210:occlusion blocks, this ratio strikes a balanced trade-oﬀ between introducing eﬀective interference and preserving identity-discriminative features, thereby supporting fair and consistent performance evaluation.
1211-4.2. Comparison with state-of-the-art methods on datasets.
1212-In this section, we evaluate the proposed FOSENet model with Market1501, CUHK 03, DukeMTMCreID and Occluded-DukeMTMC, and
1213-compare FOSENet with state-of-the-art methods. Speciﬁcally, to ensure a fair comparison and eliminate confounding factors, our FOSENet
--
1226-
1227-Fig. 7. The impact of the parameter 𝜆 on model performance in terms of CMC (%) and mAP (%) on the Market-1501 and Occluded-DukeMTMC datasets.
1228-
1229:Fig. 8. Attention heatmap visualization for the input image, Baseline model, and our proposed FOSENet. (A) shows the original input image; (B) presents the
1230-attention response generated by the Baseline model; and (C) shows the output from FOSENet with our proposed modules integrated. It can be observed that the
1231:proposed method enables the model to attend more accurately to local identity-relevant regions, especially under occlusion conditions, thereby improving feature
1232-discrimination.
1233-Table 5
1234:Ablation study on diﬀerent modules of the proposed
1235-method on the Market-1501 dataset in terms of CMC
1236-(%) and mAP (%).
1237-
1238-Table 6
1239:Mean and standard deviation of CMC Rank-1 accuracy and mAP (%) across ﬁve independent trials on four public Re-ID datasets (Market-1501,
1240-DukeMTMC-reID, CUHK03-NP, and Occluded-Duke).
1241-
1242-Market-1501
1243-Model
1244-Baseline
1245:+SGOS
1246:+LFSE
1247:+LFSE+SGOS+LOSS(FOSENet)
1248-
1249-mAP
1250-85.9
--
1258-95.5
1259-96.9
1260-
1261:4.3. Ablation studies
1262-
1263-Dataset
1264-
--
1284-86.0 ± 0.2
1285-68.7 ± 0.1
1286-
1287:based occlusion simulation method and local feature semantic extension
1288:simulation method. In addition, we have joined the proposed cross-bias
1289-loss function.
1290-1) Eﬀectiveness of the Model. As shown in Table 5, our method
1291:achieves 87.2 % mAP accuracy and 95.0 % Rank-1, which signiﬁcantly outperforms the baseline by incorporating the semantic based
1292-
1293:In this subsection, we show the ablation experimental results of our
1294-proposed FOSENet to demonstrate the eﬀectiveness of each part of our
1295:proposed method. In Table 5, the Baseline denotes the underlying backbone network, SGOS and LFSE sub-tables denote our proposed semantic9
1296-
1297-Pattern Recognition 172 (2026) 112567
1298-
--
1305-for diﬀerent situations by separating the clustering of samples with different distances and widening the interclass distances, which helps the
1306-model to extract the features eﬃciently.
1307-
1308:occlusion simulation (SGOS) method. Moreover, by considering the proposed local feature semantic extension module (LFSE) method, 88.2 %
1309-mAP accuracy and 95.5 % Rank-1 accuracy can be reached. Finally, by
1310:using all components, FOSENet can obtain 96.9 % Rank-1 accuracy and
1311-91.8 % mAP accuracy.
1312-2) Experimental Data Stability Analysis. As shown in Table 6, the proposed method obtains state-of-the-art performance in terms of mAP and
1313-Rank-1 accuracy on four benchmark datasets and the standard deviations are within ±0.3. The experiment results indicate that the proposed
--
1316-the mean and standard deviation calculated from ﬁve independent runs
1317-under identical experimental settings, ensuring both the reliability and
1318-reproducibility of the evaluation.
1319:3) Feature distribution. As shown in Fig. 5 . We present visualizations of the feature distributions optimized by the our method in the
1320-visible modality. It is observed that our proposed method is signiﬁcantly
1321-better than the backbone network.
1322-4) Convergence of Accuracy Over Epochs. As shown in Fig. 6 . We provide the recognition accuracy data (mAP,Rank-1∼10) of FOSENet in 120
--
1333-It is observed that mAP and Rank-1 reach the best when the parameter 𝜆 reaches 0.8, and then show a diﬀerent degree of decline on both
1334-
1335-4.4. Visualization analysis
1336:1) Attention pattern visualization. We use an attention heat map to
1337-represent the regions of interest learned by our model, and color diﬀerences in the labeled regions can show the ﬁnal feature representation,
1338:where regions with brighter colors mean that the model pays more attentions. As Fig. 8 shows the visualization of the ﬁnal generated features. Obviously, the feature map from Baseline shows little attention
1339-to pedestrians in the attention heat map. Moreover, it can be observed
1340-that the baseline model often focuses on some redundant regions, such
1341-as background walls or ground surfaces, which are not informative for
1342-identity discrimination. In contrast, our proposed FOSENet model can
1343:better focus on the local semantic regions of the target pedestrians to effectively capture the discriminative features of the person, and suppressing the background interference information around the target person.
1344-Even when most parts are occluded by other pedestrians, our model can
1345-still focus on the localized area of the target pedestrian, which demonstrates the eﬀectiveness of our proposed method.
1346-2) Search Results: To verify the retrieval performance improvement
--
1350-shown in Fig. 9, we can ﬁnd that a signiﬁcant improvement of the retrieval results is obtained by our FOSENet compared to the Baseline,
1351-which proves the eﬀectiveness of our proposed method.
1352-5. Conclusion
1353:In this paper, a semantic-guided occlusion simulation based local
1354:feature semantic expansion network (FOSENet) is proposed to solve
1355-10
1356-
1357-Pattern Recognition 172 (2026) 112567
1358-
1359-Z. Deng et al.
1360-
1361:the problem of limited local semantic information representation ability caused by semantic occlusion in real-world scenarios. FOSENet integrates the occluded regions with the semantic information of pedestrians to systematically generate targeted pedestrian occlusion samples,
1362-thereby increasing the diversity of pedestrian samples and enhancing
1363:the adaptability of the model. Then, FOSENet has been used to obtain ﬁne-grained regional semantic details to enhance the discriminative ability of pedestrian features. In addition, to validate the eﬀectiveness of our proposed FOSENet method and its components, we conduct
1364-experiments and obtained satisfactory results on four public person ReID datasets: Market1501, DukeMTMC-reID, CUHK 03-NP and OccludedDukeMTMC. The experiment results demonstrate the eﬀectiveness of the
1365-proposed method.
1366-Although signiﬁcant performance improvements have been
1367:achieved, we claim that the proposed SGOS and LFSE have higher
1368-computational cost and training complexity due to their multi-stage
1369-processing pipeline and the inclusion of attention-based operations.
1370-In other words, the implementation of these modules requires more
1371-computational resources and training time compared to the baseline.
1372:Moreover, under extreme occlusion or cluttered background conditions, the consistency of semantic guidance may decrease, limiting the
1373:eﬀectiveness of occlusion simulations. In the future, we will explore
1374:lightweight and adaptive occlusion simulation strategies and domain
1375:adaptation techniques for pedestrian feature extraction and semantic
1376-representation to further improve the generalization ability and
1377-eﬃciency of the proposed model under various real-world scenarios.
1378-
--
1398-[12] Z. Zhong, L. Zheng, G. Kang, S. Li, Y. Yang, Random erasing data augmentation, in: Proceedings of the AAAI Conference on Artiﬁcial Intelligence, 34, 2020,
1399-pp. 13001–13008.
1400-[13] P. Chen, W. Liu, P. Dai, J. Liu, Q. Ye, M. Xu, Q. Chen, R. Ji, Occlude
1401:them all: occlusion-aware attention network for occluded person re-id, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021,
1402-pp. 11833–11842.
1403-[14] Z. Wang, F. Zhu, S. Tang, R. Zhao, L. He, J. Song, Feature erasing and diﬀusion network for occluded person re-identiﬁcation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 4754–4763.
1404-[15] J. Miao, Y. Wu, P. Liu, Y. Ding, Y. Yang, Pose-guided feature alignment for occluded
--
1407-[16] J. Yang, J. Zhang, F. Yu, X. Jiang, M. Zhang, X. Sun, Y.-C. Chen, W.-S. Zheng,
1408-Learning to know where to see: a visibility-aware approach for occluded person
1409-re-identiﬁcation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 11885–11894.
1410:[17] S. Wang, B. Huang, H. Li, G. Qi, D. Tao, Z. Yu, Key point-aware occlusion suppression
1411:and semantic alignment for occluded person re-identiﬁcation, Inf. Sci. 606 (2022)
1412-669–687.
1413-[18] S. Gao, C. Yu, P. Zhang, H. Lu, Part representation learning with teacher-Student
1414-decoder for occluded person re-Identiﬁcation, in: ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE, 2024,
--
1437-[27] H. Huang, D. Li, Z. Zhang, X. Chen, K. Huang, Adversarially occluded samples for
1438-person re-identiﬁcation, in: Proceedings of the IEEE Conference on Computer Vision
1439-and Pattern Recognition, 2018, pp. 5098–5107.
1440:[28] C. Zhao, X. Lv, S. Dou, S. Zhang, J. Wu, L. Wang, Incremental generative occlusion adversarial suppression network for person reID, IEEE Trans. Image Process.
1441-30 (2021) 4212–4224.
1442-[29] C. Yan, G. Pang, J. Jiao, X. Bai, X. Feng, C. Shen, Occluded person re-identiﬁcation
1443-with single-scale global representations, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 11875–11884.
--
1541-Ke Nai received the PhD degree in Computer Science and Technology from Hunan
1542-University, Changsha, China, in 2019. During 2020 to 2022, he worked as a Postdoctoral Researcher at Hunan University. Currently, he is a lecturer at School of
1543-Computer of Changsha University of Science and Technology. His current research
1544:interests include visual tracking, face recognition, computer vision, pattern recognition and machine learning. He has published several papers in IEEE-TIP, IEEETNNLS, IEEE-TITS, IEEE-TCSVT, IEEE-TMM, Pattern Recognition, Information Sciences,
1545-Knowledge Based Systems, Neural Computing and Applications, ICIP2019 and so
1546-on.
1547-Guiji Li received the PhD degree in computer science and technology from Hunan University, Changsha, China, in 2020. She is currently a lecturer with Changsha
1548:University. Her research focuses on pattern recognition, computer vision, and visual
1549-tracking.
1550-Shaomiao Chen received the MSc and Doctoral degree in Computer Science and Technology from Hunan University, Changsha, China, in 2014 and 2018. He is currently a
1551-lecturer at School of Computer Science and Engineering, Hunan University of Science and

codex
FOSENet 的证据里有一个值得借鉴的点：作者专门把“随机遮挡/人工遮挡”替换成“语义引导遮挡”，再用遮挡数据集和注意力热图证明模型关注点从背景转回目标局部语义。这比只说数据增强涨点更像论文贡献。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 \"A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf\" - | sed -n '1,980p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
A Survey on 3D Skeleton Based Person Re-Identification: Taxonomy, Advances,
Challenges, and Interdisciplinary Prospects
Haocong Rao and Chunyan Miao
College of Computing and Data Science, Nanyang Technological University (NTU), Singapore
Joint NTU-UBC Research Centre of Excellence in Active Living for the Elderly (LILY), NTU, Singapore
{haocong001, ascymiao}@ntu.edu.sg

Person re-identification via 3D skeletons is an important emerging research area that attracts increasing attention within the pattern recognition community. With distinctive advantages across various
application scenarios, numerous 3D skeleton based
person re-identification (SRID) methods with diverse skeleton modeling and learning paradigms
have been proposed in recent years. In this paper, we provide a comprehensive review and analysis of recent SRID advances. First of all, we
define the SRID task and provide an overview
of its origin and major advancements. Secondly,
we formulate a systematic taxonomy that organizes existing methods into three categories centered on hand-crafted, sequence-based, and graphbased modeling. Then, we elaborate on the representative models along these three types with an
illustration of foundational mechanisms. Meanwhile, we provide an overview of mainstream supervised, self-supervised, and unsupervised SRID
learning paradigms and corresponding common
methods. A thorough evaluation of state-of-theart SRID methods is further conducted over various
types of benchmarks and protocols to compare their
effectiveness, efficiency, and key properties. Finally, we present the key challenges and prospects
to advance future research, and highlight interdisciplinary applications of SRID with a case study. A
curated collection of valuable resources is available
at https://github.com/Kali-Hac/3D-SRID-Survey.

1

Introduction

Person re-identification (re-ID) is an essential pattern recognition task of matching and retrieving a person-of-interest
across different views or scenes, which has been widely applied to security authentication, smart surveillance, healthcare, and embodied AI [Nambiar et al., 2019; Ye et al.,
2021]. Recent economical and precise skeleton-tracking devices (e.g., Kinect [Shotton et al., 2011]) have simplified
the acquisition of 3D skeleton data, enabling them to be a
prevalent and versatile data modality for gait analysis and

Hand-Crafted

Similarity

I=1

Sequences

...
3D Skeletons
�1 , �2 , ⋯, ��

Graphs

I=2
Query ID
I=3

Body/Motion Feature Learning
Database Records ��

•••

arXiv:2401.15296v3 [cs.CV] 4 Jan 2026

Abstract

Figure 1: Overview of 3D skeleton based person re-ID (SRID) task
with hand-crafted, sequence-based or graph-based modeling to learn
effective body and motion features for identity recognition.

person re-ID [Liao et al., 2020; Rao et al., 2024]. Unlike conventional person re-ID methods that rely on appearance or facial characteristics [Ye et al., 2021], 3D Skeleton
based person Re-ID (SRID) models typically exploit bodystructure features and motion patterns (e.g., gait [Murray et
al., 1964]) from 3D positions of key body joints to identify
different persons. With unique merits such as small input
data, light-weight models, privacy-preserving without using
appearances, and robustness against view and background
variations [Han et al., 2017], SRID has attracted surging attention from both academia and industry [Rao et al., 2022].
In recent years, research on SRID has gained significant momentum, leading to diverse skeleton modeling and
learning paradigms. Early endeavors [Barbosa et al., 2012;
Munaro et al., 2014a; Andersson and Araujo, 2015; Pala et
al., 2019] mainly extract hand-crafted features such as skeleton descriptors in terms of anthropometric, geometric and gait
attributes of body. As these methods often require domain
expertise such as anatomy and kinematics [Yoo et al., 2002]
for skeleton modeling, they lack the ability to fully mine latent high-level features beyond human cognition. To resolve
this challenge, recent mainstream methods [Liao et al., 2020;
Huynh-The et al., 2020; Rao et al., 2022; Rashmi and Guddeti, 2022] leverage deep neural networks to automatically
perform skeleton representation learning for SRID. One of
exemplar methods (termed “sequence-based modeling”) is to
model sequential dynamics and motion semantics from raw
or normalized skeletons (e.g., joint trajectory) based on long
short-term memory (LSTM) and its variants [Wei et al., 2020;
Rao et al., 2022]. However, they rarely investigate the intrinsic body relationships such as inter-joint motion correlations,

First depth-based
person re-ID method
(Haque et al.) (2016)
First skeleton-based
First deep learning based
person re-ID (SRID) &
SRID method
hand-crafted method
SM-SGE;
(AGE)
(Munaro et al.)
CTR-GCN

First image-based
person re-ID method
(Gheissari et al.)

2006

...

2010

...

First video-based
person re-ID Method
(Farenzana et al.)

2014

...

2017

...

First visible-Infrared
person re-ID Method
(Wu et al.)

First piont cloud based
person re-ID Method
(Munaro et al.)

2020

2021

First Transformer-based
model for SRID (TranSG);
GaitTR; GPGait
SGE-LA

2022

2023

First graph-based
model for SRID
(MG-SCR)

First sequence-based
& self-supervised
SRID method (AGE)

2025

SkeletonGait;
Hi-MPC
(2024)

First unsupervised
SRID method (SimMC)

First motif-based model
for SRID (MoCos)

(a) Origin and Advancements of SRID Research

(b) Model Accuracy & Efficiency Comparison

Figure 2: (a) Overview of research origin and technical advancements of SRID within the person re-ID community (Zoom in and follow the
timeline for the best view). (b) Parameter sizes (Millions (M)), computational complexity (Giga Floating Point Operations (GFLOPs)), and
KS20 Rank-1 accuracy of state-of-the-art deep learning methods for SRID (Red: Sequence-based models; Green: Graph-based models).

thereby possibly overlooking some valuable skeleton patterns. Another paradigm (termed “graph-based modeling”)
mitigates this challenge by constructing skeleton graphs to
model discriminative structural and actional features based on
the interrelations of body parts [Rao and Miao, 2023]. This
often requires multi-granularity body modeling and efficient
relational reasoning mechanisms (e.g., collaborative learning) based on skeleton graphs. Despite the great progress of
SRID, this rapidly evolving technique still lacks a systematic
review, making it difficult for researchers to gain a holistic
view of this field and embark on new research endeavors.
In light of this, we present the first survey on SRID, elucidating recent advancements of skeleton modeling, learning
paradigms, evaluation benchmarks, current challenges, and
interdisciplinary applications. Firstly, we define the SRID
task and provide a milestone overview to illustrate the origin
and key advancements of SRID as shown in Fig. 2a. Secondly, we propose a systematic taxonomy of SRID methods
to categorize them into hand-crafted, sequence-based, and
graph-based modeling, and elaborate on their foundational
mechanisms and representative approaches. We also illustrate the basic definitions and common methods within three
mainstream SRID paradigms (supervised, self-supervised,
unsupervised). Thirdly, we introduce existing public benchmarks, evaluation metrics, and protocols for SRID, while
comprehensively evaluating state-of-the-art methods across
different benchmarks to compare their performance and efficiency. Meanwhile, we conduct qualitative analysis of different methods to compare their key properties with a discussion of advantages and disadvantages. Finally, we discuss the
current challenges in SRID and identify potential directions
for future research. An overview of promising SRID applications in interdisciplinary areas, spanning healthcare, embodied AI, and security, is further provided and illustrated with
a case study. The structure of this survey, including skeleton
modeling (Sec. 3), learning paradigms (Sec. 4), benchmarks
and evaluation (Sec. 5), challenges and prospects (Sec. 6) is
shown in Fig. 3. We hope our survey can bring new insights
to researchers and expedite future research in SRID.

2

Overview

2.1

Task Description

As illustrated in Fig. 1, the input of SRID task is a 3D skeleton sequence that belongs to a certain pedestrian, and the
output is the predicted identity. Formally, we denote a 3D
skeleton sequence as S = (s1 , · · · , sf ) ∈ Rf ×j×3 , where
st ∈ Rj×3 denotes the tth skeleton with 3D coordinates of
j body joints. Each skeleton sequence S corresponds to a
person identity I, where I ∈ {1, · · · , C} and C is the number of different classes (i.e., identities). In the SRID task, we
generally have training set, probe set, and gallery set, respecn oN2
n oN1
, and
, ΦP = S P
tively denoted as ΦT = S Ti
i
i=1
i=1
n oN3
that contain N1 , N2 , and N3 skeleton seΦG = S G
i
i=1
quences of different persons collected from different scenes
or views. The task target is to learn a hand-crafted, sequencebased or graph-based model (detailed in Sec. 3) that maps
3D skeleton sequences into effective representations, so that
we can query the correct identity of an encoded skeleton sequence representation in the probe set via matching it with
the sequence representations in the database (i.e., gallery set).
SRID is essentially a retrieving and matching problem.

2.2

Origin and Advancements

As presented in Fig. 2a, the first SRID research [Munaro
et al., 2014a] using hand-crafted skeleton descriptors commenced in 2014, coming after the first RGB video based
method [Farenzena et al., 2010] and before the first depthbased approach [Haque et al., 2016]. Then, Rao et al. proposed the first deep learning based SRID paradigm in 2020,
followed by the first self-supervised, unsupervised, and supervised paradigms [Rao et al., 2020; Rao and Miao, 2022;
Rao and Miao, 2023]. Over the past five years, an increasing number of innovative models have been devised specifically for SRID and related emerging tasks, including LSTM
models (e.g., AGE [Rao et al., 2020]), CNN models (e.g.,
SkeletonGait [Fan et al., 2024]), Transformer models (e.g.,
TranSG [Rao and Miao, 2023]), GCN models (e.g., CTR-

Body Relations

MoCos

SM-SGE

Hi-MPC

BIWI

Rank-1

Probe-Gallery
Matching

Scarcity

Robustness

Antropometric

Pose Dynamics

Graph Dynamics

TranSG

AGE

SimMC

IAS-Lab

mAP

RGB-Estimated

Imbalance

Generalizability

Geometric

Semantics Learning

Multi-Level
Graphs

GPGait

SGELA

...

KGBD

CMC

Cross-View

Noise

Interpretability

...

...

...

...

...

Gait

Joint Motion

Other Attributes

Importance Inference

Figure 3: Structure of this survey with the taxonomy of SRID research. Representative branches and SRID methods are listed.

GCN [Chen et al., 2021]), MLP models (e.g., SimMC [Rao
and Miao, 2022]), and hybrid/composite models (e.g., SMSGE [Rao et al., 2021a]) (compared in Table 2).

2.3

Taxonomy of SRID Methods

As shown in Fig. 3, we propose a systematic taxonomy
for SRID approaches based on skeleton modeling (see Sec.
3) and learning paradigms (see Sec. 4). For skeleton
modeling, we divide them into three categories, including
(1) hand-crafted modeling using manually-extracted features
(e.g., skeleton descriptors), (2) sequence-based modeling that
focuses on sequential features (e.g., joint trajectory) of 3D
skeletons, and (3) graph-based modeling that represents 3D
skeletons as graphs, and further subcategorize them by different learning focuses such as body relations or graph dynamics. In terms of learning paradigms, we group them
into (1) supervised SRID paradigms that require skeleton labels for feature learning, (2) self-supervised SRID paradigms
that combine pretext tasks for skeleton representation learning with labeled fine-tuning, and (3) unsupervised SRID
paradigms that learn skeleton features without using labels.

3

Skeleton Modeling

We elaborate on different skeleton modeling including their
foundational mechanisms and representative approaches.

3.1

Hand-Crafted Modeling

Gait Attributes. Extracting gait features is a common way
to characterize unique walking patterns of an individual [Cunado et al., 2003], typically including (1) kinematic parameters (e.g., angles of hips, knees, and feet), and (2) spatiotemporal parameters (e.g., stride length, gait cycle time, velocity). They can be manually computed with:
fangles = {(αij , βij ) | (i, j) ∈ Ψ},
(1)
αij = arctan

yi − yj
zi − z j
,
, βij = arctan p
xi − xj
(xi − xj )2 + (yi − yj )2
(2)

where xi , yi , zi denote 3D coordinates of the ith joint, the set
Ψ defines adjacent joints constrained by the human skeleton
model, and two kinds of angles αij and βij are calculated
from these joint pairs. The velocity can be calculated by
Pn strideLength
n
V elocity = i=1
,
(3)
cycleT ime

where
strideLength = 2 ∗ stepLength.
(4)
We calculate the step length by averaging the highest values
of the difference between the right and left feet, and adopt the
mean stride length over all n strides following [Andersson
and Araujo, 2015].
Anthropometric Attributes. The Euclidean distance between two joints, such as bone lengths, limb dimensions,
height, are usually computed as the anthropometric features
(fA ) to differentiate individuals [Barbosa et al., 2012]:
fA = {∥Ji − Jj ∥2 | (i, j) ∈ Ψ∗ },
(5)
where Ji =(xi , yi , zi ), Jj =(xj , yj , zj ), Ψ∗ not only contains
adjacent joints but also can be extended to cover more anthropometric properties, such as 13 (D13 ) or 16 (D16 ) skeleton
descriptors in [Munaro et al., 2014b; Pala et al., 2019].
Geometric Attributes. The incorporation of geometric
skeletal measurements, including body-part ratios and interjoint geodesic distances on the mesh surface, can enhance
feature representations in person re-ID [Barbosa et al., 2012].
The above hand-crafted features are often learned by different classifiers (e.g., KNN) to perform person re-ID [Andersson and Araujo, 2015; Nambiar et al., 2017]. They
are also combined with different metric algorithms [Pala
et al., 2015] or other modalities such as 3D point clouds
[Gharghabi et al., 2015; Bondi et al., 2016; Pala et al., 2019;
Munaro et al., 2014a] to further boost person re-ID accuracy.

3.2

Sequence-Based Modeling

Joint Motion. The walking patterns are characterized by the
motion of joints, which can be defined as the differences between body joint positions [Liao et al., 2020]:
fmotion = st − st−1 ,
(6)
where st = {J1t , J2t , · · · , Jjt }, Jit = (xti , yit , zit ) denotes
the 3D coordinates of ith joint in the tth skeleton, and i ∈
{1, 2, · · · , j}. The occluded or masked joints during walking
can also help models learn useful motion patterns [Rashmi
and Guddeti, 2022].
Pose Dynamics. The consecutive skeletons typically conveys dynamics of unique body poses, which can be encoded
sequentially by temporal learning models (e.g., LSTM):

ϕ(s1 )
if t = 1
ht =
,
(7)
ϕ(ht−1 , st ) if 1 < t ≤ f

3.3

Graph-Based Modeling

Body Relations. Skeletons can be naturally modeled as
graphs based on the physical connections of human body
joints. For each skeleton xt , we represent it as the graph
G t (V t , E t ), where V t = {v1t , v2t , · · · , vjt }, vit ∈ R3 , i ∈
{1, · · · , j} and edges E t = {eti,j | vit , vjt ∈ V t }, eti,j ∈ R.
E t denotes the set of connections between different joints.
Based on the skeleton graphs, we can model the inherent correlations, such as limb collaboration [Rao et al., 2021b] and
channel-specific relations [Chen et al., 2021] for SRID. The
graph motifs can also be devised to enhance the structure and
gait related relation learning [Rao and Miao, 2025].
Graph Dynamics. The dynamics of skeleton graphs can
be leveraged to learn and capture the temporal evolution of
joints’ connection patterns and limbs’ coordination for SRID.
The process can be simplified as

ϕ (fG (s1 ))
if t = 1

,
(9)
gt =
ϕ g t−1 , fG (st ) if 1 < t ≤ f
where fG (·) denotes the graph encoding model (e.g., graph
convolutional networks [Fu et al., 2023]), ϕ(·) is the tempo-

Table 1: Overview of SRID benchmark datasets. The number of
skeletons in training is reported. “S” denotes single or egocentric
view. RGB-estimated 3D and 2D skeleton datasets are listed. ⋆ denotes an interdisciplinary benchmark for gait and disease prediction.
Type Dataset

Reference

Source

# ID # Skeletons # View

[Munaro et al., 2014b]

Kinect V1

50

205.8K

S

IAS-Lab RGBD-ID [Munaro et al., 2014c]

Kinect V1

11

89.0K

S

KGBD

[Andersson and Araujo, 2015] Kinect V1

164

188.7K

S

KinectREID

[Pala et al., 2015]

Kinect V1

71

4.8K

7

UPCV1

[Kastaniotis et al., 2015]

Kinect V1

30

13.1K

S

UPCV2

[Kastaniotis et al., 2016]

Kinect V2

30

26.3K

S

Kinect V2

16

18.0K

S

Sensor-Based

BIWI RGBD-ID

Florence 3D Re-ID [Bondi et al., 2016]

Estimated

where ϕ(·) denotes the model function to encode the pose dynamics of skeletons, ht−1 represents the latent representation
of previous (t−1) poses, which provides the temporal context
information to encode the long-term pose dynamics ht for
person re-ID [Rao et al., 2022]. The joint distances, relative
joint positions, and bones are combined in [Wei et al., 2020;
Zhang et al., 2023] to enhance pose dynamics learning.
Semantics Learning. By encoding the skeleton sequence
into latent high-dimensional representation, latent motion semantics such as motion continuity can be captured to enhance
feature learning. Representative semantics learning tasks include skeleton sequence reconstruction and prediction, which
can be simply formulated as follows:

Plain Recon.
s1 , s2 , · · · , sf
ϕ(s1 , s2 , · · · , sf ) = sf , sf −1 , · · · , s1
Reverse Recon.

sf +1 , sf +2 , · · · , s2f Prediction
(8)
The reverse reconstruction (Recon.) and prediction require
the model function ϕ(·) to capture high-level semantics (e.g.,
order and correlations) to achieve the target output, which
facilitates learning more meaningful gait representations for
person re-ID [Rao et al., 2022]. A few studies also explore
the semantics learning of motion consistency [Rao and Miao,
2022] and cross-scale body relations [Rao et al., 2021a].
Importance Inference. Different skeletons and their feature representations typically possess different importance in
characterizing poses and discriminative patterns of a person,
which can be exploited to mine key skeletons or hard samples [Hermans et al., 2017] for SRID learning. To aggregate
key skeleton features for SRID, AGE [Rao et al., 2020] explores a locality-aware attention mechanism to integrate features of important skeletons in the sequence, while SM-SGE
[Rao et al., 2021a] infers key correlations between different
body-joint nodes. Rao et al. [2024] further propose a hard
skeleton mining mechanism to adaptively infer importance of
multi-level skeleton representations for SRID.

KS20

[Nambiar et al., 2017]

Kinect V2

20

36.0K

5

CAISA-B-3D

[Liao et al., 2020]

Videos

124

706.5K

11

3DGait⋆

[Wang et al., 2023]

Videos

43

22.9K

S

OUMVLP-Pose-2D [Chen et al., 2022]

Videos

10307

6667.0K

14

PoseTrackReID-2D [Chen et al., 2022]

Videos

5350

53.6K

—

ral learning model (e.g., LSTM) to encode the long-term dynamics of graph representations. To facilitate spatio-temporal
graph learning, self-supervised pretext tasks, such as graph
reconstruction and prediction, are incorporated to learn more
effective skeleton features [Rao and Miao, 2023].
Multi-Level Graphs. Multi-level skeleton graph representations are devised to characterize coarse-to-fine body structure and motion [Li et al., 2020], and various graph learning tasks such as sparse graph prediction and cross-scale
graph inference [Rao et al., 2021a] are proposed to help learn
different-level graph semantics. Based on graph transformers, masked reconstruction [He et al., 2022] and motifs [Rao
and Miao, 2025] are further explored for high-level structural
(e.g., locality) and gait semantics learning.

4

Learning Paradigms

Supervised SRID paradigms leverage skeletal annotations or
labels (e.g., identity class) to guide the model to learn discriminative features, typically using cross-entropy (CE) loss
as follows:
N1 X
C
1 X
−Ii,j log(Îi,j ),
(10)
LCE =
N1 i=1 j=1
where Ii,j denotes the ground-truth identity label (Ii,j = 1 iff
the ith sample belongs to the j th identity otherwise 0), Îi,j
is the probability that the ith sample is predicted as the j th
identity, and N1 is the number of training samples. This loss
has been widely applied to classic supervised models such as
SVM [Munaro et al., 2014b], KNN [Munaro et al., 2014a;
Gharghabi et al., 2015; Nambiar et al., 2017], MLP [Andersson and Araujo, 2015], Adaboost [Pala et al., 2019], random
forest [Elaoud et al., 2021] to learn skeleton descriptors to
classify different individuals. Recent studies explore supervised skeleton prototype learning paradigms by employing
the contrastive loss
C nk
1 XX
exp (S k,j · pk /τ )
LProto =
− log PC
, (11)
N1
i=1 exp (S k,j · pi /τ )
k=1 j=1
where S k,j denotes the j th skeleton sequence belonging to
the k th identity, pk represents the prototype generated by

Table 2: Performance (R1 ) and characteristics comparison of existing hand-crafted, sequence-based, and graph-based methods on different
benchmark datasets (BIWI (S/W), IAS-Lab (A/B), KS20). Representative gait recognition methods using skeleton data are also compared
following the same person re-ID evaluation protocol. We report the parameter sizes (Million) and summarize their properties.
Conference/
Algorithms/
Network Require Pretext Multi-Scale Theory for
BIWI-S BIWI-W IAS-A IAS-B KS20
Summary
Journal
Architectures Parameters Labels? Task? Modeling? Analysis?
[Gharghabi et al., 2015]
ICIEV
10.7
10.7
—
—
—
KNN
—
✓
✗
✗
✓
Advantages:
Person
High theoretical explainability
[Munaro et al., 2014a]
32.1
39.3
—
—
—
NN
—
✓
✗
✗
✓
Re-Identification
of features and models;
Generally low complexity.
D13 [Munaro et al., 2014b]
ICRA
28.3
14.2
40.0 43.7 39.4
NN
—
✓
✗
✗
✓
Computers&
D16 [Pala et al., 2019]
32.6
17.0
42.7 44.5 51.7
Adaboost
—
✓
✗
✗
✓
Disadvantages:
Graphics
Require domain knowledge;
[Elaoud et al., 2017]
ACIVS
28.6
10.7
45.5 63.6
—
SVD
—
✓
✗
✗
✓
Labor-expensive;
Multimedia Tools
39.3
39.3
36.4 81.8
—
RF
—
✓
✗
✗
✓
PM [Elaoud et al., 2021]
Low performance.
and Applications
PoseGait [Liao et al., 2020]
Pattern Recognition 14.0
8.8
28.4 28.9 49.4
CNN
8.93M
✓
✗
✗
✗
Advantages:
AGE [Rao et al., 2020]
IJCAI
25.1
11.7
31.1 31.1 43.2
LSTM
7.15M
✓
✓
✗
✗
Learn with raw sequences;
SGELA [Rao et al., 2022]
TPAMI
25.8
11.7
16.7 22.2 45.0
LSTM
8.47M
✓
✓
✗
✗
Capture temporal dynamics.
GaitTR [Zhang et al., 2023]
Expert Systems
43.2
16.3
43.7 47.8 52.3 Transformer
0.49M
✓
✗
✗
✗
Disadvantages:
SimMC [Rao and Miao, 2022]
IJCAI
41.7
24.5
44.8 46.3 66.4
MLP
0.15M
✗
✗
✗
✓
Lack body structure modeling;
Hi-MPC [Rao et al., 2024]
IJCV
47.5
27.3
45.6 48.2 69.6
MLP
3.32M
✗
✗
✓
✓
Ignore joint or limb relations.
SkeletonGait [Fan et al., 2024]
AAAI
15.1
10.8
31.4 31.5 22.2
CNN
11.11M
✓
✗
✗
✗
Advantages:
MG-SCR [Rao et al., 2021b]
IJCAI
20.1
10.8
36.4 32.4 46.3 GAT, LSTM
0.35M
✓
✓
✓
✗
Fully model body structure
SM-SGE [Rao et al., 2021a]
ACM MM
31.3
13.2
34.0 38.9 45.9 MGRN, LSTM
5.58M
✓
✓
✓
✗
and motion connections;
CTR-GCN [Chen et al., 2021]
CVPR
59.1
20.5
47.7 48.3 57.4
GCN
1.42M
✓
✗
✗
✗
Improved interpretability
ST-GCN [Yan et al., 2018]
AAAI
56.8
21.3
47.9 50.1 65.9
GCN
2.06M
✓
✗
✗
✗
with relations.
GPGait [Fu et al., 2023]
ICCV
54.1
29.0
50.9 60.1 71.4
GCN
1.30M
✓
✗
✗
✗
Disadvantages:
TranSG [Rao and Miao, 2023]
CVPR
68.7
32.7
49.2 59.1 73.6 Transformer
0.40M
✓
✓
✗
✓
Graph topology pre-defining;
MoCos [Rao and Miao, 2025]
AAAI
72.0
36.0
51.9 61.5 76.0 Transformer
0.40M
✓
✓
✗
✓
Complex relation modeling.

Graph-Based

Sequence-Based

Hand-Crafted

Method

the feature centroid of k-class skeletons, and τ is the temperature for contrastive learning. Based on the losses in Eq.
(10) and (11), diverse network architectures such as skeletonbased LSTM [Wei et al., 2020; Rashmi and Guddeti, 2022],
CNN [Liao et al., 2020; Huynh-The et al., 2020], and Transformer [Rao and Miao, 2023] are trained for SRID.
Self-Supervised SRID paradigms usually combine unlabeled learning of pretext tasks (e.g., skeleton semantics learning) and labeled fine-tuning. A general form of pretext objective can be formulated as

Table 3: Performance of state-of-the-art SRID methods on different conditions (Normal, Clothes, Bags) of RGB-estimated CASIAB dataset. “C-N” denotes “Clothes” probe set and “Normal” gallery
set. † and ♣ denote sequence-based and graph-based methods.
Probe-Gallery

N

1
1 X
Dis (ϕ (T1 (S i )) , T2 (S i )) ,
LPretext =
N1 i=1

N-N

B-B

C-C


codex
综述这篇不是提出一个模块，而是把一个新兴子领域“3D skeleton based ReID”做成可投稿的知识结构：先说明它区别于外观 ReID 的价值，再用 taxonomy、benchmark、范式和挑战把分散工作组织起来。这个我会按综述论文的创新构造方式拆。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'pdftotext "A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf" - | rg -n -C 3 "Challenges|Prospects|challenge|future|taxonomy|interdisciplinary|application|privacy|benchmark|evaluation|self-supervised|unsupervised"' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1-A Survey on 3D Skeleton Based Person Re-Identification: Taxonomy, Advances,
2:Challenges, and Interdisciplinary Prospects
3-Haocong Rao and Chunyan Miao
4-College of Computing and Data Science, Nanyang Technological University (NTU), Singapore
5-Joint NTU-UBC Research Centre of Excellence in Active Living for the Elderly (LILY), NTU, Singapore
6-{haocong001, ascymiao}@ntu.edu.sg
7-
8-Person re-identification via 3D skeletons is an important emerging research area that attracts increasing attention within the pattern recognition community. With distinctive advantages across various
9:application scenarios, numerous 3D skeleton based
10-person re-identification (SRID) methods with diverse skeleton modeling and learning paradigms
11-have been proposed in recent years. In this paper, we provide a comprehensive review and analysis of recent SRID advances. First of all, we
12-define the SRID task and provide an overview
13-of its origin and major advancements. Secondly,
14:we formulate a systematic taxonomy that organizes existing methods into three categories centered on hand-crafted, sequence-based, and graphbased modeling. Then, we elaborate on the representative models along these three types with an
15:illustration of foundational mechanisms. Meanwhile, we provide an overview of mainstream supervised, self-supervised, and unsupervised SRID
16-learning paradigms and corresponding common
17:methods. A thorough evaluation of state-of-theart SRID methods is further conducted over various
18:types of benchmarks and protocols to compare their
19:effectiveness, efficiency, and key properties. Finally, we present the key challenges and prospects
20:to advance future research, and highlight interdisciplinary applications of SRID with a case study. A
21-curated collection of valuable resources is available
22-at https://github.com/Kali-Hac/3D-SRID-Survey.
23-
--
66-based person Re-ID (SRID) models typically exploit bodystructure features and motion patterns (e.g., gait [Murray et
67-al., 1964]) from 3D positions of key body joints to identify
68-different persons. With unique merits such as small input
69:data, light-weight models, privacy-preserving without using
70-appearances, and robustness against view and background
71-variations [Han et al., 2017], SRID has attracted surging attention from both academia and industry [Rao et al., 2022].
72-In recent years, research on SRID has gained significant momentum, leading to diverse skeleton modeling and
--
76-attributes of body. As these methods often require domain
77-expertise such as anatomy and kinematics [Yoo et al., 2002]
78-for skeleton modeling, they lack the ability to fully mine latent high-level features beyond human cognition. To resolve
79:this challenge, recent mainstream methods [Liao et al., 2020;
80-Huynh-The et al., 2020; Rao et al., 2022; Rashmi and Guddeti, 2022] leverage deep neural networks to automatically
81-perform skeleton representation learning for SRID. One of
82-exemplar methods (termed “sequence-based modeling”) is to
--
148-(MG-SCR)
149-
150-First sequence-based
151:& self-supervised
152-SRID method (AGE)
153-
154-2025
--
157-Hi-MPC
158-(2024)
159-
160:First unsupervised
161-SRID method (SimMC)
162-
163-First motif-based model
--
172-KS20 Rank-1 accuracy of state-of-the-art deep learning methods for SRID (Red: Sequence-based models; Green: Graph-based models).
173-
174-thereby possibly overlooking some valuable skeleton patterns. Another paradigm (termed “graph-based modeling”)
175:mitigates this challenge by constructing skeleton graphs to
176-model discriminative structural and actional features based on
177-the interrelations of body parts [Rao and Miao, 2023]. This
178-often requires multi-granularity body modeling and efficient
--
181-review, making it difficult for researchers to gain a holistic
182-view of this field and embark on new research endeavors.
183-In light of this, we present the first survey on SRID, elucidating recent advancements of skeleton modeling, learning
184:paradigms, evaluation benchmarks, current challenges, and
185:interdisciplinary applications. Firstly, we define the SRID
186-task and provide a milestone overview to illustrate the origin
187:and key advancements of SRID as shown in Fig. 2a. Secondly, we propose a systematic taxonomy of SRID methods
188-to categorize them into hand-crafted, sequence-based, and
189-graph-based modeling, and elaborate on their foundational
190-mechanisms and representative approaches. We also illustrate the basic definitions and common methods within three
191:mainstream SRID paradigms (supervised, self-supervised,
192:unsupervised). Thirdly, we introduce existing public benchmarks, evaluation metrics, and protocols for SRID, while
193-comprehensively evaluating state-of-the-art methods across
194:different benchmarks to compare their performance and efficiency. Meanwhile, we conduct qualitative analysis of different methods to compare their key properties with a discussion of advantages and disadvantages. Finally, we discuss the
195:current challenges in SRID and identify potential directions
196:for future research. An overview of promising SRID applications in interdisciplinary areas, spanning healthcare, embodied AI, and security, is further provided and illustrated with
197-a case study. The structure of this survey, including skeleton
198:modeling (Sec. 3), learning paradigms (Sec. 4), benchmarks
199:and evaluation (Sec. 5), challenges and prospects (Sec. 6) is
200-shown in Fig. 3. We hope our survey can bring new insights
201:to researchers and expedite future research in SRID.
202-
203-2
204-
--
240-As presented in Fig. 2a, the first SRID research [Munaro
241-et al., 2014a] using hand-crafted skeleton descriptors commenced in 2014, coming after the first RGB video based
242-method [Farenzena et al., 2010] and before the first depthbased approach [Haque et al., 2016]. Then, Rao et al. proposed the first deep learning based SRID paradigm in 2020,
243:followed by the first self-supervised, unsupervised, and supervised paradigms [Rao et al., 2020; Rao and Miao, 2022;
244-Rao and Miao, 2023]. Over the past five years, an increasing number of innovative models have been devised specifically for SRID and related emerging tasks, including LSTM
245-models (e.g., AGE [Rao et al., 2020]), CNN models (e.g.,
246-SkeletonGait [Fan et al., 2024]), Transformer models (e.g.,
--
328-
329-Importance Inference
330-
331:Figure 3: Structure of this survey with the taxonomy of SRID research. Representative branches and SRID methods are listed.
332-
333-GCN [Chen et al., 2021]), MLP models (e.g., SimMC [Rao
334-and Miao, 2022]), and hybrid/composite models (e.g., SMSGE [Rao et al., 2021a]) (compared in Table 2).
--
337-
338-Taxonomy of SRID Methods
339-
340:As shown in Fig. 3, we propose a systematic taxonomy
341-for SRID approaches based on skeleton modeling (see Sec.
342-3) and learning paradigms (see Sec. 4). For skeleton
343-modeling, we divide them into three categories, including
--
346-focuses on sequential features (e.g., joint trajectory) of 3D
347-skeletons, and (3) graph-based modeling that represents 3D
348-skeletons as graphs, and further subcategorize them by different learning focuses such as body relations or graph dynamics. In terms of learning paradigms, we group them
349:into (1) supervised SRID paradigms that require skeleton labels for feature learning, (2) self-supervised SRID paradigms
350:that combine pretext tasks for skeleton representation learning with labeled fine-tuning, and (3) unsupervised SRID
351-paradigms that learn skeleton features without using labels.
352-
353-3
--
462-where fG (·) denotes the graph encoding model (e.g., graph
463-convolutional networks [Fu et al., 2023]), ϕ(·) is the tempo-
464-
465:Table 1: Overview of SRID benchmark datasets. The number of
466-skeletons in training is reported. “S” denotes single or egocentric
467:view. RGB-estimated 3D and 2D skeleton datasets are listed. ⋆ denotes an interdisciplinary benchmark for gait and disease prediction.
468-Type Dataset
469-
470-Reference
--
646-—
647-
648-ral learning model (e.g., LSTM) to encode the long-term dynamics of graph representations. To facilitate spatio-temporal
649:graph learning, self-supervised pretext tasks, such as graph
650-reconstruction and prediction, are incorporated to learn more
651-effective skeleton features [Rao and Miao, 2023].
652-Multi-Level Graphs. Multi-level skeleton graph representations are devised to characterize coarse-to-fine body structure and motion [Li et al., 2020], and various graph learning tasks such as sparse graph prediction and cross-scale
--
692-the k th identity, pk represents the prototype generated by
693-
694-Table 2: Performance (R1 ) and characteristics comparison of existing hand-crafted, sequence-based, and graph-based methods on different
695:benchmark datasets (BIWI (S/W), IAS-Lab (A/B), KS20). Representative gait recognition methods using skeleton data are also compared
696:following the same person re-ID evaluation protocol. We report the parameter sizes (Million) and summarize their properties.
697-Conference/
698-Algorithms/
699-Network Require Pretext Multi-Scale Theory for
--
1164-target of a pretext task, ϕ(·) represent the model encoding
1165-function, and Dis(·) is the distance metric (e.g., Euclidean
1166-distance). By employing sequence-based pretext tasks such
1167:as sparse sequential prediction [Rao et al., 2021b], or graphbased tasks such as structure-trajectory prompted reconstruction [Rao and Miao, 2023], self-supervised learning encourages the model to capture high-level motion concepts and
1168-class-related spatio-temporal semantics for SRID.
1169:Unsupervised SRID paradigms perform skeleton representation learning without using any labels. Existing methods mainly adopt unsupervised skeleton prototype learning frameworks, which use the feature centroids of clustering as prototypes (see Eq. (11)) [Rao and Miao, 2022;
1170-Rao et al., 2024]. Their performance could be affected by
1171-the robustness of clustering algorithms and contrastive learning mechanisms, while unlabeled pretext tasks are often combined to enhance general skeleton semantics learning.
1172-
--
1176-
1177-Benchmark Datasets. Table 1 summarizes the data source,
1178-identity number, skeleton amount, and viewpoint number of
1179:commonly-used datasets for SRID evaluation. They can be
1180-mainly categorized to two types: (1) Sensor-based datasets,
1181-where 3D skeleton data are captured from depth sensors such
1182-as Kinect, and (2) RGB-estimated datasets, in which skeleton data are estimated from RGB videos using 2D or 3D
--
1184-al., 2019]. Existing SRID datasets typically contain skeleton data collected from varying scenarios such as multiple
1185-views (KS20, KinectREID), appearance and clothing changes
1186-(BIWI RGBD-ID, IAS-Lab RGBD-ID), and different illumination conditions (KGBD), which enables a comprehensive
1187:evaluation of both short-term and long-term SRID performance. In addition to standard SRID datasets, the 3DGait
1188:dataset provides an interdisciplinary benchmark for evaluating the generalizability of SRID models on healthcarerelated tasks, such as neurodegenerative disease prediction
1189-(see Sec. 6.2) [Rao et al., 2025b].
1190-
1191-Evaluation Metrics and Protocols. In SRID, the performance is typically evaluated based on several mertics, including Cumulative Matching Characteristics (CMC), Rank-1 accuracy (R1 ), Rank-5 accuracy, Rank-10 accuracy, and Mean
1192-Average Precision (mAP) [Zheng et al., 2015]. Multiple
1193:evaluation protocols are employed across varying datasets,
1194:including probe-gallery matching evaluation (main protocol
1195:evaluated in Table 2), RGB-estimated evaluation (representative protocol evaluated in Table 3), random view evaluation,
1196:cross-view evaluation [Rao et al., 2024], zero-shot crossdataset evaluation [Rao et al., 2022], etc.
1197-Comparison of Performance and Efficiency. As shown
1198-in Table 2, D16 and PM are two most competitive handcrafted methods, performing well on both BIWI and IAS
1199:benchmarks. However, recent deep learning based models
1200-significantly surpass them: The latest sequence-based model
1201-(Hi-MPC) and graph-based model (MoCos) achieve superior performance to them across different datasets. Notably,
1202-the top three performers—GPGait, TranSG, and MoCos—all
--
1224-
1225-6
1226-
1227:Challenges and Prospects
1228-
1229:In this section, we elucidate key challenges in SRID data and
1230-models with a discussion of potential directions. We also
1231:present promising interdisciplinary applications of SRID.
1232-Data Scarcity and Imbalance. Existing 3D skeletons are
1233-mainly collected from prevailing depth sensors such as Kinect
1234-[Shotton et al., 2011], while diverse skeleton collection settings (e.g., different devices in uncontrollable environments)
--
1248-factors: The quality of skeleton data may also be affected by
1249-device’s tracking distance, illumination changes (e.g., influence structured light in Kinect V1), and source data quality
1250-(e.g., image resolution). Such inherent noise puts high demand on the model robustness against random perturbations.
1251:To address these two challenges, higher-quality 3D skeleton data should be collected or generated. It is feasible to
1252-devise skeleton denoising models and augmentation strategies for skeleton generation. GAN-based pose generators
1253-[Yan et al., 2017] and diffusion models [Ho et al., 2020]
1254-could be transferred to generate and denoise 3D skeleton
1255:data. The future efforts include (a) collecting and opening new larger-scale SRID datasets, (b) transferring existing
1256-skeleton datasets from other areas to person re-ID datasets
1257:and formulating appropriate evaluation protocols, (c) estimating 2D/3D skeletons from large-scale public person re-ID
1258-datasets to construct new estimated SRID datasets, so as to
1259-advance SRID research and related person re-ID community.
1260-Model Robustness. A few SRID models report unstable
--
1262-parameter initialization, hyper-parameter settings (e.g., clustering parameters) and varying data distributions [Rao and
1263-Miao, 2022]. As these studies opt for the best-trained model
1264-with finely-tuned parameters, this practice often fails to reflect the architecture’s true average performance and shows
1265:limited adaptability in real-world applications.
1266:A key future direction is to systematically investigate
1267-how key factors (e.g., model initialization, data quality) affect robustness, which is essential for developing more reliable models. Theoretical analyses of performance variations
1268-in terms of model-approximated functions and convergence
1269:conditions can also be provided for more robust model design. Moreover, multi-faceted evaluation metrics, such as performance average and standard deviation, should be reported
1270-to better measure the overall robustness of models.
1271-Model Generalizability. Most SRID models are trained
1272-on a single dataset with limited data sizes, views, scenes or
--
1275-scenarios (e.g., RGB-estimated skeleton data).
1276-A potential solution is to exploit larger-scale SRID datasets
1277-to train the model across diverse scenarios, so as to learn more
1278:domain-general skeleton semantics. It is also feasible to explore domain adaptation or generalization techniques to cotrain and transfer models. More benchmarks should be investigated for generality evaluation of SRID models.
1279-Interpretability. Current SRID models typically lack in-
1280-
1281:Table 4: Generalization performance of representative SRID methods on the interdisciplinary benchmark 3DGait — A case study.
1282-Types
1283-Hand-Crafted
1284-
--
1286-
1287-Graph-Based
1288-
1289:Figure 4: Interdisciplinary application landscape of SRID across
1290-three primary domains: healthcare (green box), embodied AI (yellow box), and security (purple box). Please zoom in for better view.
1291-
1292-tuitive explanations for the effectiveness of model architectures, skeleton features, and prediction results. This opacity
1293-not only poses risks of erroneous outcomes but also hinders
1294:their reliable applications at scale.
1295-To this end, different human-friendly explanation including pose/feature visualization and corresponding languagebased description can be considered. There also exist various architecture-specific explanation mechanisms, e.g., class
1296-activation maps (CAM) [Zhou et al., 2016] for CNN, knowledge graphs [Ji et al., 2021] for graph neural networks, which
1297-could be applied to explainable skeleton learning. Moreover,
--
1313-generalize effectively across different data modalities.
1314-Cross-Modality Evaluation Protocol. To enable a fair
1315-comparison of person re-ID methods on multi-modal (e.g.,
1316:RGB-D) benchmarks, it is imperative to formulate a crossmodality evaluation protocol that standardizes re-ID settings
1317-(e.g., probe/gallery settings, single/multi-shot recognition) of
1318-skeleton-based, depth-based, radar-based and other methods.
1319-Skeleton Foundation Model. LLMs and pose generative
--
1321-It can hopefully be fine-tuned for diverse tasks such as skeleton visualization, augmentation, prediction, etc. Such foundation model will help investigate the limit of adaptability,
1322-generality, and interpretability of 3D skeletons and SRID.
1323-Privacy Protection. A unique advantage of SRID models
1324:is that they inherently protect privacy by avoiding the use of
1325-explicit appearance data. However, the illegal or irresponsible deployment of person re-ID technologies could jeopardize
1326-public security, making it crucial to establish SRID-related
1327:laws to protect social safety and personal privacy.
1328-
1329-6.2
1330-
--
1376-
1377-60.2
1378-
1379:Interdisciplinary Prospects
1380-
1381:As shown in Fig. 4, we list several promising applications of
1382-SRID in terms of healthcare, embodied AI, and security:
1383-Healthcare. Since physiological and psychological states
1384-are intrinsically correlated with walking patterns (e.g.,
--
1386-models can be transferred for neurodegenerative disease diagnosis and psychological/affective detection (e.g., depression)
1387-[Lu et al., 2023]. Moreover, they can hopefully support medical gait analysis and rehabilitation assessment (e.g., stroke)
1388-via non-intrusive monitoring in daily environments.
1389:To demonstrate the above interdisciplinary applicability,
1390-we follow [Rao et al., 2025b] to systematically evaluate various SRID models on a representative healthcare task: Neurodegenerative disease prediction. The case study in Table 4
1391-shows that these models can capture generalizable discriminative gait patterns for this task, suggesting their promising
1392:potential for broader real-world healthcare applications.
1393-Embodied AI. With smaller input data and lower resource
1394-requirement than RGB-based models, SRID can serve as a
1395-fundamental semantic link for intelligent agents and their interaction systems: Agents can act via skeleton-based avatar
--
1398-evolve within skeleton-driven synthetic environments, which
1399-facilitate robust identity-aware collaboration in robotics.
1400-Security. Leveraging the robustness of skeletal features
1401:against appearance variations [Han et al., 2017], SRID advances public safety through privacy-preserving authentication and cross-modal (e.g., face, fingerprint) identity verification. It also enables forensic behavioral analysis to track
1402-suspects and detect abnormal events across varying scenarios
1403-particularly where visual details are lacking.
1404-
--
1408-
1409-In this paper, we provide the first comprehensive review of 3D
1410-skeleton based person re-ID (SRID). We first define the SRID
1411:task and present a timeline to overview the major advancements of SRID. Then we formulate a taxonomy of SRID approaches in terms of different skeleton modeling and learning paradigms, systematically presenting foundational mechanisms and reviewing representative approaches. An empirical evaluation of state-of-the-art SRID methods is conducted
1412:across various benchmarks and evaluation protocols to compare their effectiveness, efficiency, and key properties. We
1413:further discuss critical challenges along with potential directions, and highlight interdisciplinary prospects of SRID.
1414-
1415-References
1416-[Andersson and Araujo, 2015] Virginia O Andersson and Ricardo M Araujo. Person identification using anthropometric and
--
1475-Nguyen Anh Tu, and Dong-Seong Kim. Learning 3D spatiotemporal gait feature by convolutional network for person
1476-identification. Neurocomputing, 397:192–202, 2020.
1477-[Ji et al., 2021] Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka
1478:Marttinen, and S Yu Philip. A survey on knowledge graphs: Representation, acquisition, and applications. IEEE Transactions on
1479-Neural Networks and Learning Systems, 33(2):494–514, 2021.
1480-[Kastaniotis et al., 2015] Dimitris Kastaniotis, Ilias Theodorakopoulos, Christos Theoharatos, George Economou, and Spiros
1481-Fotopoulos. A framework for gait-based recognition using
--
1528-Graphics, 79:69–80, 2019.
1529-[Rao and Miao, 2022] Haocong Rao and Chunyan Miao. SimMC:
1530-Simple masked contrastive learning of skeleton representations
1531:for unsupervised person re-identification. In International Joint
1532-Conference on Artificial Intelligence (IJCAI), pages 1290–1297,
1533-2022.
1534-[Rao and Miao, 2023] Haocong Rao and Chunyan Miao. TranSG:
--
1542-Tan, Huang Da, Jun Cheng, and Bin Hu. Self-supervised gait encoding with locality-aware attention for person re-identification.
1543-In International Joint Conference on Artificial Intelligence (IJCAI), volume 1, pages 898–905, 2020.
1544-[Rao et al., 2021a] Haocong Rao, Xiping Hu, Jun Cheng, and Bin
1545:Hu. SM-SGE: A self-supervised multi-scale skeleton graph encoding framework for person re-identification. In Proceedings
1546-of the 29th ACM International Conference on Multimedia, pages
1547-1812–1820, 2021.
1548-[Rao et al., 2021b] Haocong Rao, Shihao Xu, Xiping Hu, Jun
--
1555-
1556-[Rao et al., 2024] Haocong Rao, Cyril Leung, and Chunyan Miao.
1557-Hierarchical skeleton meta-prototype contrastive learning with
1558:hard skeleton mining for unsupervised person re-identification.
1559-International Journal of Computer Vision, 132(1):238–260,
1560-2024.
1561-[Rao et al., 2025a] Haocong Rao, Minlin Zeng, Xuejiao Zhao, and
1562-Chunyan Miao. A survey of artificial intelligence in gait-based
1563-neurodegenerative disease diagnosis. Neurocomputing, 2025.
1564-[Rao et al., 2025b] Haocong Rao, Jiachen Zhao, and Chunyan
1565:Miao. Llm-powered interpretable 3d gait visualization and analysis platform for interdisciplinary ai applications. In 27th International Conference on Human-Computer Interaction (HCII),
1566-2025.
1567-[Rashmi and Guddeti, 2022] M Rashmi and Ram Mohana Reddy
1568-Guddeti. Human identification system using 3D skeleton-based
--
1596-[Zhang et al., 2023] Cun Zhang, Xing-Peng Chen, Guo-Qiang Han,
1597-and Xiang-Jie Liu. Spatial transformer network on skeletonbased gait recognition. Expert Systems, 40(6):e13244, 2023.
1598-[Zheng et al., 2015] Liang Zheng, Liyue Shen, Lu Tian, Shengjin
1599:Wang, Jingdong Wang, and Qi Tian. Scalable person reidentification: A benchmark. In ICCV, pages 1116–1124, 2015.
1600-[Zhou et al., 2016] Bolei Zhou, Aditya Khosla, Agata Lapedriza,
1601-Aude Oliva, and Antonio Torralba. Learning deep features for
1602-discriminative localization. In CVPR, pages 2921–2929, 2016.

exec
/bin/zsh -lc "pdftotext -f 8 -l 14 \"A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf\" - | sed -n '1,900p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
References
[Andersson and Araujo, 2015] Virginia O Andersson and Ricardo M Araujo. Person identification using anthropometric and
gait data from Kinect sensor. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 425–431, 2015.
[Barbosa et al., 2012] Igor Barros Barbosa, Marco Cristani,
Alessio Del Bue, Loris Bazzani, and Vittorio Murino. Reidentification with RGB-D sensors. In ECCV Workshop, pages
433–442. Springer, 2012.
[Bondi et al., 2016] Enrico Bondi, Pietro Pala, Lorenzo Seidenari,
Stefano Berretti, and Alberto Del Bimbo. Long term person
re-identification from depth cameras using facial and skeleton
data. In International Conference on Pattern Recognition (ICPR)
Workshop, pages 29–41, 2016.
[Cao et al., 2019] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En
Wei, and Yaser Sheikh. OpenPose: Realtime multi-person 2D
pose estimation using part affinity fields. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 43(1):172–186, 2019.
[Chen and Ramanan, 2017] Ching-Hang Chen and Deva Ramanan.
3D human pose estimation= 2D pose estimation+ matching. In
CVPR, pages 7035–7043, 2017.
[Chen et al., 2021] Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, Bing
Li, Ying Deng, and Weiming Hu. Channel-wise topology refinement graph convolution for skeleton-based action recognition. In
ICCV, pages 13359–13368, 2021.
[Chen et al., 2022] Di Chen, Andreas Döring, Shanshan Zhang,
Jian Yang, Juergen Gall, and Bernt Schiele. Keypoint message
passing for video-based person re-identification. In Proceedings
of the AAAI Conference on Artificial Intelligence (AAAI), volume 36, pages 239–247, 2022.
[Cunado et al., 2003] David Cunado, Mark S Nixon, and John N
Carter. Automatic extraction and description of human gait models for recognition purposes. Computer vision and image understanding, 90(1):1–41, 2003.
[Elaoud et al., 2017] Amani Elaoud, Walid Barhoumi, Hassen
Drira, and Ezzeddine Zagrouba. Analysis of skeletal shape trajectories for person re-identification. In International Conference
on Advanced Concepts for Intelligent Vision Systems, pages 138–
149. Springer, 2017.
[Elaoud et al., 2021] Amani Elaoud, Walid Barhoumi, Hassen
Drira, and Ezzeddine Zagrouba. Person re-identification from
different views based on dynamic linear combination of distances. Multimedia Tools and Applications, 80:17685–17704,
2021.
[Fan et al., 2024] Chao Fan, Jingzhe Ma, Dongyang Jin, Chuanfu
Shen, and Shiqi Yu. SkeletonGait: Gait recognition using skeleton maps. In Proceedings of the AAAI Conference on Artificial
Intelligence (AAAI), volume 38, pages 1662–1669, 2024.
[Farenzena et al., 2010] Michela Farenzena, Loris Bazzani,
Alessandro Perina, Vittorio Murino, and Marco Cristani. Person
re-identification by symmetry-driven accumulation of local
features. In CVPR, pages 2360–2367. IEEE, 2010.
[Fu et al., 2023] Yang Fu, Shibei Meng, Saihui Hou, Xuecai Hu,
and Yongzhen Huang. GPGait: Generalized pose-based gait
recognition. In ICCV, pages 19595–19604, 2023.
[Gharghabi et al., 2015] Shaghayegh Gharghabi, Faraz Shamshirdar, and Taher Abbas Shangari, et al. People re-identification
using 3D descriptor with skeleton information. In 2015 International Conference on Informatics, Electronics & Vision (ICIEV),
pages 1–5. IEEE, 2015.

[Han et al., 2017] Fei Han, Brian Reily, William Hoff, and Hao
Zhang. Space-time representation of people based on 3D skeletal data: A review. Computer Vision and Image Understanding,
158:85–105, 2017.
[Haque et al., 2016] Albert Haque, Alexandre Alahi, and Li FeiFei. Recurrent attention models for depth-based person identification. In CVPR, pages 1229–1238, 2016.
[He et al., 2022] Kaiming He, Xinlei Chen, Saining Xie, Yanghao
Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are
scalable vision learners. In CVPR, pages 16000–16009, 2022.
[Hermans et al., 2017] Alexander Hermans, Lucas Beyer, and Bastian Leibe.
In defense of the triplet loss for person reidentification. arXiv preprint arXiv:1703.07737, 2017.
[Ho et al., 2020] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 33:6840–6851, 2020.
[Huynh-The et al., 2020] Thien Huynh-The, Cam-Hao Hua,
Nguyen Anh Tu, and Dong-Seong Kim. Learning 3D spatiotemporal gait feature by convolutional network for person
identification. Neurocomputing, 397:192–202, 2020.
[Ji et al., 2021] Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka
Marttinen, and S Yu Philip. A survey on knowledge graphs: Representation, acquisition, and applications. IEEE Transactions on
Neural Networks and Learning Systems, 33(2):494–514, 2021.
[Kastaniotis et al., 2015] Dimitris Kastaniotis, Ilias Theodorakopoulos, Christos Theoharatos, George Economou, and Spiros
Fotopoulos. A framework for gait-based recognition using
Kinect. Pattern Recognition Letters, 68:327–335, 2015.
[Kastaniotis et al., 2016] Dimitris Kastaniotis, Ilias Theodorakopoulos, George Economou, and Spiros Fotopoulos. Gait
based recognition via fusing information from euclidean and
riemannian manifolds. Pattern Recognition Letters, 84:245–251,
2016.
[Li et al., 2020] Maosen Li, Siheng Chen, Yangheng Zhao,
Ya Zhang, Yanfeng Wang, and Qi Tian. Dynamic multiscale
graph neural networks for 3D skeleton based human motion prediction. In CVPR, pages 214–223, 2020.
[Liao et al., 2020] Rijun Liao, Shiqi Yu, Weizhi An, and Yongzhen
Huang. A model-based gait recognition method with body pose
and human prior knowledge. Pattern Recognition, 98:107069,
2020.
[Lu et al., 2023] Haifeng Lu, Xiping Hu, and Bin Hu. See your
emotion from gait using unlabeled skeleton data. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI),
volume 37, pages 1826–1834, 2023.
[Lucas et al., 2022] Thomas Lucas, Fabien Baradel, Philippe Weinzaepfel, and Grégory Rogez. PoseGPT: Quantization-based 3D
human motion generation and forecasting. In ECCV, pages 417–
435. Springer, 2022.
[Munaro et al., 2014a] Matteo Munaro, Alberto Basso, Andrea
Fossati, Luc Van Gool, and Emanuele Menegatti. 3D reconstruction of freely moving persons for re-identification with a depth
sensor. In International Conference on Robotics and Automation
(ICRA), pages 4512–4519. IEEE, 2014.
[Munaro et al., 2014b] Matteo Munaro, Andrea Fossati, Alberto
Basso, Emanuele Menegatti, and Luc Van Gool. One-shot person re-identification with a consumer depth camera. In Person
Re-Identification, pages 161–181. Springer, 2014.

[Munaro et al., 2014c] Matteo Munaro,
Stefano Ghidoni,
Deniz Tartaro Dizmen, and Emanuele Menegatti. A featurebased approach to people re-identification using skeleton
keypoints.
In International Conference on Robotics and
Automation (ICRA), pages 5644–5651. IEEE, 2014.
[Murray et al., 1964] M Pat Murray, A Bernard Drought, and
Ross C Kory. Walking patterns of normal men. Journal of Bone
and Joint Surgery, 46(2):335–360, 1964.
[Nambiar et al., 2017] Athira Nambiar, Alexandre Bernardino, Jacinto C Nascimento, and Ana Fred. Context-aware person reidentification in the wild via fusion of gait and anthropometric
features. In International Conference on Automatic Face & Gesture Recognition, pages 973–980. IEEE, 2017.
[Nambiar et al., 2019] Athira Nambiar, Alexandre Bernardino, and
Jacinto C Nascimento. Gait-based person re-identification: A
survey. ACM Computing Surveys, 52(2):33, 2019.
[Pala et al., 2015] Federico Pala, Riccardo Satta, Giorgio Fumera,
and Fabio Roli. Multimodal person reidentification using RGBD cameras. IEEE Transactions on Circuits and Systems for Video
Technology, 26(4):788–799, 2015.
[Pala et al., 2019] Pietro Pala, Lorenzo Seidenari, Stefano Berretti,
and Alberto Del Bimbo. Enhanced skeleton and face 3D data
for person re-identification from depth cameras. Computers &
Graphics, 79:69–80, 2019.
[Rao and Miao, 2022] Haocong Rao and Chunyan Miao. SimMC:
Simple masked contrastive learning of skeleton representations
for unsupervised person re-identification. In International Joint
Conference on Artificial Intelligence (IJCAI), pages 1290–1297,
2022.
[Rao and Miao, 2023] Haocong Rao and Chunyan Miao. TranSG:
Transformer-based skeleton graph prototype contrastive learning
with structure-trajectory prompted reconstruction for person reidentification. In CVPR, 2023.
[Rao and Miao, 2025] Haocong Rao and Chunyan Miao. Motif
guided graph transformer with combinatorial skeleton prototype
learning for skeleton-based person re-identification. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI),
2025.
[Rao et al., 2020] Haocong Rao, Siqi Wang, Xiping Hu, Mingkui
Tan, Huang Da, Jun Cheng, and Bin Hu. Self-supervised gait encoding with locality-aware attention for person re-identification.
In International Joint Conference on Artificial Intelligence (IJCAI), volume 1, pages 898–905, 2020.
[Rao et al., 2021a] Haocong Rao, Xiping Hu, Jun Cheng, and Bin
Hu. SM-SGE: A self-supervised multi-scale skeleton graph encoding framework for person re-identification. In Proceedings
of the 29th ACM International Conference on Multimedia, pages
1812–1820, 2021.
[Rao et al., 2021b] Haocong Rao, Shihao Xu, Xiping Hu, Jun
Cheng, and Bin Hu. Multi-level graph encoding with structuralcollaborative relation learning for skeleton-based person reidentification. In International Joint Conference on Artificial Intelligence (IJCAI), pages 973–980, 2021.
[Rao et al., 2022] Haocong Rao, Siqi Wang, Xiping Hu, Mingkui
Tan, Yi Guo, Jun Cheng, Xinwang Liu, and Bin Hu. A selfsupervised gait encoding approach with locality-awareness for
3D skeleton based person re-identification. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 44(10):6649–6666,
2022.

[Rao et al., 2024] Haocong Rao, Cyril Leung, and Chunyan Miao.
Hierarchical skeleton meta-prototype contrastive learning with
hard skeleton mining for unsupervised person re-identification.
International Journal of Computer Vision, 132(1):238–260,
2024.
[Rao et al., 2025a] Haocong Rao, Minlin Zeng, Xuejiao Zhao, and
Chunyan Miao. A survey of artificial intelligence in gait-based
neurodegenerative disease diagnosis. Neurocomputing, 2025.
[Rao et al., 2025b] Haocong Rao, Jiachen Zhao, and Chunyan
Miao. Llm-powered interpretable 3d gait visualization and analysis platform for interdisciplinary ai applications. In 27th International Conference on Human-Computer Interaction (HCII),
2025.
[Rashmi and Guddeti, 2022] M Rashmi and Ram Mohana Reddy
Guddeti. Human identification system using 3D skeleton-based
gait features and LSTM model. Journal of Visual Communication
and Image Representation (JVCIR), 82:103416, 2022.
[Shotton et al., 2011] Jamie Shotton, Andrew Fitzgibbon, Mat
Cook, Toby Sharp, Mark J Finocchio, Richard Moore,
Alex Abenathar Kipman, and Andrew Blake. Real-time human
pose recognition in parts from single depth images. In CVPR,
pages 1297–1304, 2011.
[Wang et al., 2023] Diwei Wang, Chaima Zouaoui, Jinhyeok Jang,
Hassen Drira, and Hyewon Seo. Video-based gait analysis for
assessing alzheimer’s disease and dementia with lewy bodies. In
MICCAI Workshop on Applications of Medical AI, pages 72–82.
Springer, 2023.
[Wei et al., 2018] Longhui Wei, Shiliang Zhang, Wen Gao, and
Qi Tian. Person transfer gan to bridge domain gap for person
re-identification. In CVPR, pages 79–88, 2018.
[Wei et al., 2020] Chu-Chien Wei, Li-Huang Tsai, Hsin-Ping Chou,
and Shih-Chieh Chang. Person identification by walking gesture
using skeleton sequences. In Advanced Concepts for Intelligent
Vision Systems, pages 205–214. Springer, 2020.
[Yan et al., 2017] Yichao Yan, Jingwei Xu, Bingbing Ni, Wendong
Zhang, and Xiaokang Yang. Skeleton-aided articulated motion
generation. In Proceedings of the 25th ACM international conference on Multimedia, pages 199–207, 2017.
[Yan et al., 2018] Sijie Yan, Yuanjun Xiong, and Dahua Lin. Spatial temporal graph convolutional networks for skeleton-based action recognition. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 7444–7452, 2018.
[Ye et al., 2021] Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang,
Ling Shao, and Steven CH Hoi. Deep learning for person reidentification: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(6):2872–2893, 2021.
[Yoo et al., 2002] Jang-Hee Yoo, Mark S Nixon, and Chris J Harris. Extracting gait signatures based on anatomical knowledge. In
Proceedings of BMVA Symposium on Advancing Biometric Technologies, pages 596–606. Citeseer, 2002.
[Zhang et al., 2023] Cun Zhang, Xing-Peng Chen, Guo-Qiang Han,
and Xiang-Jie Liu. Spatial transformer network on skeletonbased gait recognition. Expert Systems, 40(6):e13244, 2023.
[Zheng et al., 2015] Liang Zheng, Liyue Shen, Lu Tian, Shengjin
Wang, Jingdong Wang, and Qi Tian. Scalable person reidentification: A benchmark. In ICCV, pages 1116–1124, 2015.
[Zhou et al., 2016] Bolei Zhou, Aditya Khosla, Agata Lapedriza,
Aude Oliva, and Antonio Torralba. Learning deep features for
discriminative localization. In CVPR, pages 2921–2929, 2016.


web search: 
web search: "Knowledge and experience" "visible-infrared person re-identification"
web search: 
web search: "A Semantic-guided occlusion simulation" "local feature semantic expansion" person re-identification
web search: 
web search: "Cross-modality Knowledge and Experience Transformer" VI-ReID
web search: 
web search: https://doi.org/10.1016/j.patcog.2025.112553
web search: 
web search: 10.1016/j.patcog.2025.112553
web search: 
web search: "Occlude Them All" occlusion-aware attention network occluded person re-id
web search: 
web search: "visible-infrared person re-identification" "modality-specific memory network"
web search: 
web search: "FMCNet" "Feature-Level Modality Compensation"
web search: 
web search: "3D shape temporal aggregation" "video-based clothing-change person re-identification"
web search: 
web search: "FOSENet" "person re-identification"
codex
已按你指定的 `pdftotext -f 1 -l 8 '文件名' -` 读了每篇的方法/approach 部分，并额外看了对应 ablation、可视化和必要先例。

**1. ‘Knowledge and experience’ for visible-infrared person re-identification**

1. **触发观察**：现有 VI-ReID 主要从输入 RGB/IR 图像里抽取 modality-shared feature，但 RGB 和 IR 真正共享的信息有限，继续只做对齐会碰到上限。作者借人类识别过程说，人不是只看当前图像，还会用过去经验推断没直接出现的属性。

2. **重定义动作**：它把 VI-ReID 从“跨模态特征对齐”改成“当前图像证据 + 训练历史知识推理”。关键词是 `knowledge and experience`、`memory`、`reasoning`、`inferred modality-shared information`。这一步的核心不是记忆库本身，而是把训练集中不同身份之间的关系说成一种可迁移的“经验”。

3. **机制怎么长出来**：先用普通 VI-ReID 子网抽共享特征，再用每个训练身份的特征中心形成 memory；Transformer 建模 identity center 之间的关系，得到 knowledge；测试时用当前图像特征作为 query，从 knowledge 里推断补充特征，最后和原特征相加。逻辑绑定中等偏强，机制自然，但“人类经验”这层叙述有包装成分，本质接近原型记忆和跨原型注意力推理。

4. **证据闭环**：关键不是只看涨点，而是 Fig. 5/6 和 ablation。Venn 图显示 inferred feature 单独能正确匹配一部分身份，其中还有一小部分是原 modality-shared feature 匹配不到的；融合后又多匹配一些身份。这是在证明“推理出来的信息确实有互补性”，不是单纯多加模块涨点。

5. **reviewer 为什么买账**：这篇卖的是视角多于模块。novelty 来自“VI-ReID 缺的不是又一个对齐损失，而是从历史身份关系中推断隐含共享信息”。风险是 reviewer 会把它看成 memory/prototype attention 的变体，所以它必须靠 inferred feature 的独立有效性和互补性来撑住。

**2. 3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification**

1. **触发观察**：视频 ReID 近期方法多在做 temporal aggregation，但它们默认 frame-level feature 已经可靠。作者指出真正的坏源头是帧级特征质量差：多帧错位、遮挡、相似外观混淆都会污染后面的时序聚合。

2. **重定义动作**：视频 ReID 被改写成“2D 视频外观不足，需要 3D 几何和 shape 作为稳定身份线索”。关键词是 `frame-level feature quality`、`3D modality`、`geometric information`、`shape cues`、`feature inconsistency`、`appearance confusion`。先例上，3D ReID 已有 OGNet 这类图像级 3D 尝试，但本文把位置卡在 video ReID 和 2D-3D 融合策略上。([arxiv.org](https://arxiv.org/abs/2006.04569?utm_source=openai))

3. **机制怎么长出来**：机制基本是自然推出的。既然 2D 帧被遮挡和错位破坏，就用 SMPL/HMR 生成 3D mesh，转成 point cloud，再用 PointPillars 得到和 2D feature 对齐的 pseudo-image feature；CMF 用 deformable cross-attention 和 2D-3D 映射点做融合；SSTI 用 SMPL shape 参数建 spatial/temporal 两条注意力流，专门处理相似外观混淆。

4. **证据闭环**：它的证据链较完整。组件 ablation 显示 SSTI 对 appearance confusion 有贡献，CMF 对 frame-level feature 贡献更大；fusion strategy 对比证明不是随便 concat 3D 就行；SSTI 和 GRU/self-attention、空间/时间单流等变体对比证明“空间和时间 shape 交互”必要；t-SNE、Grad-CAM、hard occlusion feature map 展示模型确实更关注目标区域。

5. **reviewer 为什么买账**：这篇卖的是“新模态切入视频 ReID”的机制和视角结合。真正 novelty 不是 3D 本身，而是把视频 ReID 的失败拆成三类，再让 CMF/SSTI 各自对应。它也主动承认 3D reconstruction 在拥挤场景会不稳，这反而让论证更可信。

**3. 4SNet - Spatial and Spectrum Self-adaptive Synergy Network**

1. **触发观察**：作者观察到 visible 和 infrared 的频谱分布不同，IR 更集中在低频，visible 有更多中高频细节。现有方法要么做图像生成，要么做特征空间对齐；已有频域 VI-ReID 如 FDNM 已提出频域差异，但大多把频率当成一个统一处理工具。([arxiv.org](https://arxiv.org/abs/2401.02162?utm_source=openai))

2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。

3. **机制怎么长出来**：机制和重定义绑定很紧。AFF 放在双 stem 输入端，用空间和通道注意力合成 input-specific frequency filter，解决单模态频率响应不同；SSM 放在共享 backbone 中，把特征频谱分成低、中、高频，用不同分支处理，再注意力融合，解决跨模态对齐。两个模块的职责不重叠，叙事很干净。

4. **证据闭环**：关键证据包括 FFT 可视化、AFF/SSM 单独和联合 ablation、parallel channel-spatial attention 优于顺序结构、SSM 粗三频段优于和 AFF 一样的细分频段、各频段组合实验、t-SNE 和 retrieval 可视化。最重要的是“粗三频段更适合对齐”这个实验，它证明重定义里的“alignment 不该做细粒度频率匹配”是有依据的。

5. **reviewer 为什么买账**：这篇卖的是视角。模块本身都是频域、注意力、分支融合的组合，但它把频率建模分工讲清楚了：AFF 处理模态内，SSM 处理模态间。novelty 的来源是问题拆分，而不是某个算子。

**4. A Semantic-Aware Attention and Visual Shielding Network**

1. **触发观察**：换衣 ReID 的核心失败是模型过度依赖衣服外观。已有方法多做 body shape、contour sketch，但 human semantic information 和换衣前后特征一致性没有被充分利用。

2. **重定义动作**：它把换衣 ReID 从“找更强的外观/形状特征”改成“让衣服线索失效，并强迫模型使用非衣服的人体语义”。关键词是 `visual shielding`、`clothing-irrelevant feature`、`human semantic attention`、`potential consistency`。语义像素替换已有先例，SAVS 的差异是把语义前景注意力和衣服屏蔽一致性联合起来。([arxiv.org](https://arxiv.org/abs/2207.08387?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2107.11522?utm_source=openai))

3. **机制怎么长出来**：机制非常自然。先用 human parsing 得到前景、衣服区域和屏蔽图；HSA 用前景图生成通道权重，去重权原图特征，让模型看人而不是背景；VCS 把衣服区域用 batch 内像素池替换，得到 rendering image，再约束原图特征和屏蔽图特征接近。这样模型要保留身份，只能依赖头、脸、腿、鞋、随身物等非衣服语义。

4. **证据闭环**：它的 ablation 设计很值得学。作者不是只测 `+HSA`、`+VCS`，还测原图、前景、背景、原图加前景注意力、原图加背景注意力，证明“前景本身不够，语义调制原图才有效”。VCS 部分又做 step-by-step，并和 Pixel Sampling 在相同 backbone 下比较。注意力图、同人换衣 similarity map、不同人同衣 similarity map、检索结果共同证明“屏蔽衣服后模型确实不再被衣服主导”。

5. **reviewer 为什么买账**：这篇主要卖视角，但机制也足够贴题。novelty 不在 segmentation，而在“把衣服作为负线索主动破坏，再约束身份一致性”。reviewer 买账的原因是它正面回答了换衣 ReID 的两个硬场景：同人不同衣、不同人同衣。

**5. A Semantic-guided occlusion simulation based local feature semantic expansion network**

1. **触发观察**：遮挡不是随机黑块，真实遮挡物有形状、有语义，而且经常遮住关键身份区域。另一个观察是 CNN 深层局部特征容易盯住单个中心或背景冗余语义，导致局部语义表达不足。

2. **重定义动作**：它把遮挡 ReID 从“增强遮挡鲁棒性”改成两个新问题：遮挡是语义干扰，不是随机缺失；关键局部区域需要邻近语义扩展，不是孤立地看。关键词是 `semantic-guided occlusion simulation`、`semantic differential features`、`local feature semantic expansion`、`semantic relevance`、`semantic continuity`。它和 FED 这类遮挡特征擦除/扩散方法相比，切入点更偏“如何构造更像真实干扰的训练样本”。([arxiv.org](https://arxiv.org/abs/2112.08740?utm_source=openai))

3. **机制怎么长出来**：SGOS 用相邻卷积层的语义差异图找高响应位置，再把非人类 COCO mask 形状和语义 patch 融合成遮挡块，放到语义相关位置。LFSE 用注意力找关键区域和邻近区域，对邻近区域按重要性和差异性筛选，再作为附加语义融合到关键区域。Cross-center offset loss 则把最近的跨类 hard pair 推远。

4. **证据闭环**：组件 ablation 显示 SGOS、LFSE、loss 逐步提升，完整 FOSENet 最好；Occluded-Duke 上 mAP 领先，说明不是只在干净集上有效；五次运行标准差较小，说明结果稳定；t-SNE、注意力热图和检索可视化证明模型从背景/冗余区域转向目标局部语义，尤其在遮挡场景仍能关注目标区域。

5. **reviewer 为什么买账**：它卖的是“真实遮挡语义化”和“局部语义扩展”两个视角。机制略多，有堆模块风险，所以它必须用遮挡数据集、热图和逐组件 ablation 证明每个模块都对应一个具体失败点。

**6. A Survey on 3D Skeleton Based Person Re-Identification**

1. **触发观察**：3D skeleton ReID 已经出现手工特征、序列模型、图模型、监督/自监督/无监督等多条线，但缺少系统综述。传统外观 ReID 依赖 RGB、衣服、人脸，存在隐私、视角、背景和外观变化问题；skeleton 则有轻量、隐私友好、抗背景和视角变化的优势。([arxiv.org](https://arxiv.org/abs/2401.15296?utm_source=openai))

2. **重定义动作**：它把 skeleton ReID 从“ReID 的一个小分支”重新定义成一个独立研究对象 SRID。关键词是 `SRID`、`body/motion feature learning`、`hand-crafted / sequence-based / graph-based modeling`、`supervised / self-supervised / unsupervised paradigms`、`interdisciplinary prospects`。

3. **机制怎么长出来**：综述没有模型机制，它的“方法”是组织机制。先正式定义任务，再做时间线；再按 skeleton modeling 分三类，按 learning paradigm 分三类；然后统一 benchmark、protocol、性能和效率比较；最后提出挑战和跨学科应用。这是典型“领域建构型”论文。

4. **证据闭环**：它用方法表、数据集表、参数量和 Rank-1 对比、benchmark/protocol 汇总、挑战清单和 3DGait 医疗任务 case study 来证明 SRID 不是零散技巧，而是有共同问题、共同评价和外溢应用的方向。

5. **reviewer 为什么买账**：这篇卖的是视角和整理能力。novelty 来自“第一个系统化 SRID 综述 + 清楚 taxonomy + 统一比较 + 未来议程”。对我们做方法稿的启发是：当你能把一堆零散工作重新组织成一个新问题空间时，问题定义本身就有论文价值。

**共同套路**

这批论文都不是先抛模块，而是先把一个旧问题拆出“大家没正面处理的变量”：历史经验、3D 几何、频谱角色、非衣服语义、语义遮挡、骨架模态。然后机制只负责把这个变量强行进入数据流或训练约束。

真正能打动 reviewer 的不是 SOTA 表，而是证明“这个新变量确实按作者说的方式工作”的证据：互补匹配 Venn 图、频谱可视化、相似外观 t-SNE、屏蔽前后 similarity map、遮挡热图、taxonomy 对比。下一步我们做 ReID 方法，也要先找这种能被可视化和消融闭环验证的“重定义变量”。
tokens used
201,351
已按你指定的 `pdftotext -f 1 -l 8 '文件名' -` 读了每篇的方法/approach 部分，并额外看了对应 ablation、可视化和必要先例。

**1. ‘Knowledge and experience’ for visible-infrared person re-identification**

1. **触发观察**：现有 VI-ReID 主要从输入 RGB/IR 图像里抽取 modality-shared feature，但 RGB 和 IR 真正共享的信息有限，继续只做对齐会碰到上限。作者借人类识别过程说，人不是只看当前图像，还会用过去经验推断没直接出现的属性。

2. **重定义动作**：它把 VI-ReID 从“跨模态特征对齐”改成“当前图像证据 + 训练历史知识推理”。关键词是 `knowledge and experience`、`memory`、`reasoning`、`inferred modality-shared information`。这一步的核心不是记忆库本身，而是把训练集中不同身份之间的关系说成一种可迁移的“经验”。

3. **机制怎么长出来**：先用普通 VI-ReID 子网抽共享特征，再用每个训练身份的特征中心形成 memory；Transformer 建模 identity center 之间的关系，得到 knowledge；测试时用当前图像特征作为 query，从 knowledge 里推断补充特征，最后和原特征相加。逻辑绑定中等偏强，机制自然，但“人类经验”这层叙述有包装成分，本质接近原型记忆和跨原型注意力推理。

4. **证据闭环**：关键不是只看涨点，而是 Fig. 5/6 和 ablation。Venn 图显示 inferred feature 单独能正确匹配一部分身份，其中还有一小部分是原 modality-shared feature 匹配不到的；融合后又多匹配一些身份。这是在证明“推理出来的信息确实有互补性”，不是单纯多加模块涨点。

5. **reviewer 为什么买账**：这篇卖的是视角多于模块。novelty 来自“VI-ReID 缺的不是又一个对齐损失，而是从历史身份关系中推断隐含共享信息”。风险是 reviewer 会把它看成 memory/prototype attention 的变体，所以它必须靠 inferred feature 的独立有效性和互补性来撑住。

**2. 3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification**

1. **触发观察**：视频 ReID 近期方法多在做 temporal aggregation，但它们默认 frame-level feature 已经可靠。作者指出真正的坏源头是帧级特征质量差：多帧错位、遮挡、相似外观混淆都会污染后面的时序聚合。

2. **重定义动作**：视频 ReID 被改写成“2D 视频外观不足，需要 3D 几何和 shape 作为稳定身份线索”。关键词是 `frame-level feature quality`、`3D modality`、`geometric information`、`shape cues`、`feature inconsistency`、`appearance confusion`。先例上，3D ReID 已有 OGNet 这类图像级 3D 尝试，但本文把位置卡在 video ReID 和 2D-3D 融合策略上。([arxiv.org](https://arxiv.org/abs/2006.04569?utm_source=openai))

3. **机制怎么长出来**：机制基本是自然推出的。既然 2D 帧被遮挡和错位破坏，就用 SMPL/HMR 生成 3D mesh，转成 point cloud，再用 PointPillars 得到和 2D feature 对齐的 pseudo-image feature；CMF 用 deformable cross-attention 和 2D-3D 映射点做融合；SSTI 用 SMPL shape 参数建 spatial/temporal 两条注意力流，专门处理相似外观混淆。

4. **证据闭环**：它的证据链较完整。组件 ablation 显示 SSTI 对 appearance confusion 有贡献，CMF 对 frame-level feature 贡献更大；fusion strategy 对比证明不是随便 concat 3D 就行；SSTI 和 GRU/self-attention、空间/时间单流等变体对比证明“空间和时间 shape 交互”必要；t-SNE、Grad-CAM、hard occlusion feature map 展示模型确实更关注目标区域。

5. **reviewer 为什么买账**：这篇卖的是“新模态切入视频 ReID”的机制和视角结合。真正 novelty 不是 3D 本身，而是把视频 ReID 的失败拆成三类，再让 CMF/SSTI 各自对应。它也主动承认 3D reconstruction 在拥挤场景会不稳，这反而让论证更可信。

**3. 4SNet - Spatial and Spectrum Self-adaptive Synergy Network**

1. **触发观察**：作者观察到 visible 和 infrared 的频谱分布不同，IR 更集中在低频，visible 有更多中高频细节。现有方法要么做图像生成，要么做特征空间对齐；已有频域 VI-ReID 如 FDNM 已提出频域差异，但大多把频率当成一个统一处理工具。([arxiv.org](https://arxiv.org/abs/2401.02162?utm_source=openai))

2. **重定义动作**：它把“跨模态 gap”拆成两个问题：单模态内部需要自适应频率滤波，跨模态对齐需要按频段协同。关键词是 `role-decoupled frequency modeling`、`input-adaptive filtering`、`low/mid/high frequency synergy`、`identity-shared patterns`、`modality-specific traits`。

3. **机制怎么长出来**：机制和重定义绑定很紧。AFF 放在双 stem 输入端，用空间和通道注意力合成 input-specific frequency filter，解决单模态频率响应不同；SSM 放在共享 backbone 中，把特征频谱分成低、中、高频，用不同分支处理，再注意力融合，解决跨模态对齐。两个模块的职责不重叠，叙事很干净。

4. **证据闭环**：关键证据包括 FFT 可视化、AFF/SSM 单独和联合 ablation、parallel channel-spatial attention 优于顺序结构、SSM 粗三频段优于和 AFF 一样的细分频段、各频段组合实验、t-SNE 和 retrieval 可视化。最重要的是“粗三频段更适合对齐”这个实验，它证明重定义里的“alignment 不该做细粒度频率匹配”是有依据的。

5. **reviewer 为什么买账**：这篇卖的是视角。模块本身都是频域、注意力、分支融合的组合，但它把频率建模分工讲清楚了：AFF 处理模态内，SSM 处理模态间。novelty 的来源是问题拆分，而不是某个算子。

**4. A Semantic-Aware Attention and Visual Shielding Network**

1. **触发观察**：换衣 ReID 的核心失败是模型过度依赖衣服外观。已有方法多做 body shape、contour sketch，但 human semantic information 和换衣前后特征一致性没有被充分利用。

2. **重定义动作**：它把换衣 ReID 从“找更强的外观/形状特征”改成“让衣服线索失效，并强迫模型使用非衣服的人体语义”。关键词是 `visual shielding`、`clothing-irrelevant feature`、`human semantic attention`、`potential consistency`。语义像素替换已有先例，SAVS 的差异是把语义前景注意力和衣服屏蔽一致性联合起来。([arxiv.org](https://arxiv.org/abs/2207.08387?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2107.11522?utm_source=openai))

3. **机制怎么长出来**：机制非常自然。先用 human parsing 得到前景、衣服区域和屏蔽图；HSA 用前景图生成通道权重，去重权原图特征，让模型看人而不是背景；VCS 把衣服区域用 batch 内像素池替换，得到 rendering image，再约束原图特征和屏蔽图特征接近。这样模型要保留身份，只能依赖头、脸、腿、鞋、随身物等非衣服语义。

4. **证据闭环**：它的 ablation 设计很值得学。作者不是只测 `+HSA`、`+VCS`，还测原图、前景、背景、原图加前景注意力、原图加背景注意力，证明“前景本身不够，语义调制原图才有效”。VCS 部分又做 step-by-step，并和 Pixel Sampling 在相同 backbone 下比较。注意力图、同人换衣 similarity map、不同人同衣 similarity map、检索结果共同证明“屏蔽衣服后模型确实不再被衣服主导”。

5. **reviewer 为什么买账**：这篇主要卖视角，但机制也足够贴题。novelty 不在 segmentation，而在“把衣服作为负线索主动破坏，再约束身份一致性”。reviewer 买账的原因是它正面回答了换衣 ReID 的两个硬场景：同人不同衣、不同人同衣。

**5. A Semantic-guided occlusion simulation based local feature semantic expansion network**

1. **触发观察**：遮挡不是随机黑块，真实遮挡物有形状、有语义，而且经常遮住关键身份区域。另一个观察是 CNN 深层局部特征容易盯住单个中心或背景冗余语义，导致局部语义表达不足。

2. **重定义动作**：它把遮挡 ReID 从“增强遮挡鲁棒性”改成两个新问题：遮挡是语义干扰，不是随机缺失；关键局部区域需要邻近语义扩展，不是孤立地看。关键词是 `semantic-guided occlusion simulation`、`semantic differential features`、`local feature semantic expansion`、`semantic relevance`、`semantic continuity`。它和 FED 这类遮挡特征擦除/扩散方法相比，切入点更偏“如何构造更像真实干扰的训练样本”。([arxiv.org](https://arxiv.org/abs/2112.08740?utm_source=openai))

3. **机制怎么长出来**：SGOS 用相邻卷积层的语义差异图找高响应位置，再把非人类 COCO mask 形状和语义 patch 融合成遮挡块，放到语义相关位置。LFSE 用注意力找关键区域和邻近区域，对邻近区域按重要性和差异性筛选，再作为附加语义融合到关键区域。Cross-center offset loss 则把最近的跨类 hard pair 推远。

4. **证据闭环**：组件 ablation 显示 SGOS、LFSE、loss 逐步提升，完整 FOSENet 最好；Occluded-Duke 上 mAP 领先，说明不是只在干净集上有效；五次运行标准差较小，说明结果稳定；t-SNE、注意力热图和检索可视化证明模型从背景/冗余区域转向目标局部语义，尤其在遮挡场景仍能关注目标区域。

5. **reviewer 为什么买账**：它卖的是“真实遮挡语义化”和“局部语义扩展”两个视角。机制略多，有堆模块风险，所以它必须用遮挡数据集、热图和逐组件 ablation 证明每个模块都对应一个具体失败点。

**6. A Survey on 3D Skeleton Based Person Re-Identification**

1. **触发观察**：3D skeleton ReID 已经出现手工特征、序列模型、图模型、监督/自监督/无监督等多条线，但缺少系统综述。传统外观 ReID 依赖 RGB、衣服、人脸，存在隐私、视角、背景和外观变化问题；skeleton 则有轻量、隐私友好、抗背景和视角变化的优势。([arxiv.org](https://arxiv.org/abs/2401.15296?utm_source=openai))

2. **重定义动作**：它把 skeleton ReID 从“ReID 的一个小分支”重新定义成一个独立研究对象 SRID。关键词是 `SRID`、`body/motion feature learning`、`hand-crafted / sequence-based / graph-based modeling`、`supervised / self-supervised / unsupervised paradigms`、`interdisciplinary prospects`。

3. **机制怎么长出来**：综述没有模型机制，它的“方法”是组织机制。先正式定义任务，再做时间线；再按 skeleton modeling 分三类，按 learning paradigm 分三类；然后统一 benchmark、protocol、性能和效率比较；最后提出挑战和跨学科应用。这是典型“领域建构型”论文。

4. **证据闭环**：它用方法表、数据集表、参数量和 Rank-1 对比、benchmark/protocol 汇总、挑战清单和 3DGait 医疗任务 case study 来证明 SRID 不是零散技巧，而是有共同问题、共同评价和外溢应用的方向。

5. **reviewer 为什么买账**：这篇卖的是视角和整理能力。novelty 来自“第一个系统化 SRID 综述 + 清楚 taxonomy + 统一比较 + 未来议程”。对我们做方法稿的启发是：当你能把一堆零散工作重新组织成一个新问题空间时，问题定义本身就有论文价值。

**共同套路**

这批论文都不是先抛模块，而是先把一个旧问题拆出“大家没正面处理的变量”：历史经验、3D 几何、频谱角色、非衣服语义、语义遮挡、骨架模态。然后机制只负责把这个变量强行进入数据流或训练约束。

真正能打动 reviewer 的不是 SOTA 表，而是证明“这个新变量确实按作者说的方式工作”的证据：互补匹配 Venn 图、频谱可视化、相似外观 t-SNE、屏蔽前后 similarity map、遮挡热图、taxonomy 对比。下一步我们做 ReID 方法，也要先找这种能被可视化和消融闭环验证的“重定义变量”。
