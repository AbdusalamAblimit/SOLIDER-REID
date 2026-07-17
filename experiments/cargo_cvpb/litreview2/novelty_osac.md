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
session id: 019ef920-53d9-7120-a67c-fc5bba459cd9
--------
user
我们要做 ReID 方法 **OSAC(Open-Set Spectral Over-Collapse)**。重定义: open-set ReID 的隐藏变量是 **seen-ID neural collapse 过强**——训练末期特征过度对齐 seen-ID prototype 几何, unseen-ID 可迁移身份证据低秩化/各向异性化挤到低能谱尾部; k-reciprocal 只能重排恢复不了被压没的维度。hubness 是症状非根因。机制: 训练期 spectral anti-collapse(spectral floor + Top-PC Dropout)保 unseen 证据, 测试仍单 embedding。

用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'seen-ID 过坍缩定义成 open-set ReID 失败变量 + 训练期抗坍缩保 unseen 迁移': (a)Neural Collapse(Papyan 2020)用到 ReID / open-set / metric learning 没; (b)dimensional collapse / rank collapse in SSL/contrastive(Jing et al, Hua et al); (c)VICReg/Barlow Twins/whitening/decorrelation 泛化零件; (d)ProNet / prototype-projection / classifier-reuse ReID; (e)generalizable/open-set/DG ReID 里有没有'过坍缩伤 unseen 迁移'诊断; (f)feature whitening / group whitening ReID(AMS / SNR / IN 类); (g)spectral/rank regularization metric learning; (h)open-set recognition 里的 feature collapse。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
codex
我会按你列的八类逐一查最近和经典先例，重点不是泛泛找“谱/白化/collapse”，而是确认有没有把它精确定义为 open-set ReID 的 seen-ID 过坍缩失效变量，并且训练期用谱抗坍缩保护 unseen 证据。
web search: 
web search: Neural Collapse person re-identification open-set metric learning prototype ReID
web search: 
web search: "dimensional collapse" contrastive self-supervised learning Jing Hua
web search: 
web search: "ProNet" person re-identification prototype projection classifier reuse ReID
web search: 
web search: "over-collapse" "person re-identification"
web search: 
web search: "domain generalization" "person re-identification" "rank" "feature"
web search: 
web search: "overfitting to source identities" "person re-identification" "unseen"
web search: 
web search: person re-identification feature whitening group whitening AMS SNR IN
web search: 
web search: "Style Normalization and Restitution" person re-identification SNR
web search: 
web search: "spectral regularization" "deep metric learning" rank collapse
web search: 
web search: "spectrum" "deep metric learning" embedding regularization
web search: 
web search: "dimensional collapse" "deep metric learning"
web search: 
web search: "rank regularization" "embedding" "metric learning"
web search: 
web search: "feature collapse" "open set recognition"
web search: 
web search: "Open-Set Recognition" "neural collapse"
web search: 
web search: site:openaccess.thecvf.com "Neural Collapse" "Open Set Recognition"
web search: 
web search: "Neural Collapse" "Deep Metric Learning"
web search: 
web search: "Neural Collapse" "supervised contrastive"
web search: 
web search: "Neural Collapse" transfer learning generalization unseen classes
web search: 
web search: "neural collapse" "hurts" transfer learning unseen classes
web search: 
web search: Papyan Han Donoho Prevalence of Neural Collapse during terminal phase of deep learning PNAS 2020
web search: 
web search: "Neural Collapse" "re-identification"
web search: 
web search: neural collapse reidentification ReID person ID classifier weights prototypes
web search: 
web search: "Top-PC Dropout"
web search: 
web search: "dropout" "top principal components" representations
web search: 
web search: "batch spectral shrinkage" transfer learning singular values representation
web search: 
web search: "Batch Spectral Shrinkage"
web search: 
web search: "Batch Spectral Shrinkage" arxiv
web search: 
web search: "Transferability vs. Discriminability" "Deep Transfer Learning" "Batch Spectral Penalization"
web search: 
web search: "generalizable person re-identification" "overfitting" "source" "style"
web search: 
web search: "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" arxiv
web search: 
web search: "feature decorrelation" "person re-identification"
web search: 
web search: "spectrum" "person re-identification" "feature" "regularization"
web search: 
web search: "spectral floor" deep learning representation
web search: 
web search: "anisotropy" "person re-identification" feature embedding
web search: 
web search: "hubness" "person re-identification"
web search: 
web search: "classifier weights" "person re-identification" "prototype"
web search: 
web search: https://arxiv.org/abs/2008.08186
web search: 
web search: https://arxiv.org/abs/2005.11037
web search: 
web search: https://arxiv.org/abs/2112.15121
web search: 
web search: 'training longer leads to worse last-layer features' in https://arxiv.org/abs/2202.08384
codex
截至 2026-06-24 检索后的判断：**OSAC 主切口存活，但机制零件撞车很多**。不能把“variance/covariance/whitening/谱正则防 collapse”写成主创新；主创新必须写成 **open-set ReID 中 seen-ID prototype 几何过强导致 unseen 身份证据谱尾化，并用训练期抗坍缩保护单 embedding 的可迁移维度**。

| 类别 | Verdict | 撞车点 / 未撞切口 | 信心 |
|---|---|---|---|
| (a) Neural Collapse 到 ReID/open-set/metric learning | **存活，强相邻撞车** | Papyan NC 已定义 train class feature 向 class mean / ETF / classifier self-dual collapse。NC 已用于 transfer/OOD/OSR 解释或检测，且 Galanti 甚至说 NC 可泛化到新类；Hui 则指出 NC 可能让 downstream feature 变差。**没查到把 NC 明确落到 person ReID seen-ID 过坍缩作为 open-set ReID 失败变量**。证据：[Papyan 2020](https://arxiv.org/abs/2008.08186)、[Hui 2022](https://arxiv.org/abs/2202.08384)、[Galanti 2021](https://arxiv.org/abs/2112.15121)、[NC-OOD](https://arxiv.org/abs/2311.01479)、[NECO](https://arxiv.org/abs/2310.06823) | 8/10 |
| (b) SSL/contrastive dimensional collapse | **机制撞车，问题存活** | Jing/Hua 已经把“维度坍缩、强相关、低维子空间”讲得很清楚。OSAC 的“低秩化/各向异性化”概念不能当新现象，只能当迁移到 ReID 的诊断变量。证据：[Jing et al.](https://arxiv.org/abs/2110.09348)、[Hua et al.](https://arxiv.org/abs/2105.00470) | 9/10 |
| (c) VICReg / Barlow / whitening / decorrelation | **零件严重撞车** | spectral floor 类似 VICReg variance floor；去相关/白化类似 Barlow Twins、Whitening SSL。不能声称“防 collapse 的谱地板/去相关”新。切口在 supervised open-set ReID 的 seen-ID prototype over-collapse，而非 SSL collapse。证据：[VICReg](https://arxiv.org/abs/2105.04906)、[Barlow Twins](https://arxiv.org/abs/2103.03230)、[Whitening SSL](https://arxiv.org/abs/2007.06346) | 10/10 |
| (d) ProNet / prototype projection / classifier reuse ReID | **强相邻撞车，有杀伤** | ProNet 直接把 ReID classifier 权重当 seen-ID prototypes，并在推理时保留 prototype similarity 做检索特征。这会挑战“seen prototype 对齐一定伤 unseen”。但它没有诊断过坍缩，也不是训练期谱抗坍缩，且测试不是“仍单原始 embedding”。必须做 ProNet/ProNet-like 对照。证据：[ProNet](https://arxiv.org/abs/2308.10717) | 9/10 |
| (e) open-set / DG ReID 是否已有“过坍缩伤 unseen” | **核心切口存活** | Open-set ReID/GOM 已强调 query ID 可能不在 gallery、ReID 是 retrieval+verification；DG ReID 多数讲 domain/style overfit，不是 seen-ID neural collapse。没看到“训练末期 seen-ID prototype collapse 压掉 unseen identity evidence”的诊断。证据：[GOM](https://arxiv.org/abs/2011.11506)、[SNR](https://arxiv.org/abs/2005.11037)、[MetaBIN](https://arxiv.org/abs/2011.14670)、[Style Interleaved](https://arxiv.org/abs/2207.03132) | 8/10 |
| (f) feature whitening / GW / IN 类 ReID | **中等撞车** | AMS 已用 IN + Group Whitening 做 DG ReID；SNR 明确说 IN 会丢 identity-relevant 信息再 restitution；CC-ReID 也有 feature decorrelation。OSAC 要避开“又一个 whitening ReID”叙事，强调 classifier-prototype over-collapse 与谱尾 unseen 证据。证据：[AMS](https://arxiv.org/abs/2210.10409)、[SNR](https://arxiv.org/abs/2005.11037)、[CC-ReID decorrelation](https://arxiv.org/abs/2410.05536)、[RaMoE decorrelation](https://arxiv.org/abs/2105.09156) | 9/10 |
| (g) spectral/rank regularization metric learning | **局部撞车** | OLE 是 deep metric/classification 的 low-rank orthogonal embedding，方向甚至是主动类内低秩；CLOP 用 orthonormal prototypes 防 contrastive collapse；Top-PC removal 在 NLP embedding 后处理也有先例。没撞到“ReID 训练期 spectral floor + Top-PC Dropout 保 unseen 证据”。证据：[OLÉ](https://arxiv.org/abs/1712.01727)、[CLOP](https://arxiv.org/abs/2403.18699)、[All-but-the-Top](https://arxiv.org/abs/1702.01417) | 7/10 |
| (h) open-set recognition 里的 feature collapse | **相邻撞车，不杀** | OSR/OOD 已把 NC/known-class simplex geometry 用来做 unknown rejection，甚至 Deep Simplex classifier 系列利用 NC。多数是“利用 known collapse 区分 unknown”，不是“known collapse 过强会损害 unseen ReID retrieval”。证据：[NC-OOD](https://arxiv.org/abs/2311.01479)、[NECO](https://arxiv.org/abs/2310.06823)、[DMDSC](https://arxiv.org/abs/2605.00675) | 8/10 |

**总 verdict：存活，信心 7/10。**  
杀伤最大的不是某篇完全同题论文，而是三类 reviewer attack：VICReg/Barlow 说机制不新，AMS/SNR/feature decorrelation ReID 说 ReID 已做白化去相关，ProNet 说 seen prototypes 对 unseen retrieval 也可能有用。

OSAC 要站住，实验必须证明三件事：  
1. baseline 训练末期出现 effective rank 下降、谱尾能量下降、top-PC 与 classifier/prototype subspace 对齐增强，并且这些量和 unseen mAP/R1 负相关。  
2. k-reciprocal/re-ranking 只能重排邻域，不能恢复 embedding 谱维度；OSAC 不靠 test-time trick。  
3. OSAC 胜过 VICReg-like variance/cov、GW/whitening、random dropout、early stopping、label smoothing、ProNet/ProNet-like prototype feature 等强对照。
tokens used
83,709
截至 2026-06-24 检索后的判断：**OSAC 主切口存活，但机制零件撞车很多**。不能把“variance/covariance/whitening/谱正则防 collapse”写成主创新；主创新必须写成 **open-set ReID 中 seen-ID prototype 几何过强导致 unseen 身份证据谱尾化，并用训练期抗坍缩保护单 embedding 的可迁移维度**。

| 类别 | Verdict | 撞车点 / 未撞切口 | 信心 |
|---|---|---|---|
| (a) Neural Collapse 到 ReID/open-set/metric learning | **存活，强相邻撞车** | Papyan NC 已定义 train class feature 向 class mean / ETF / classifier self-dual collapse。NC 已用于 transfer/OOD/OSR 解释或检测，且 Galanti 甚至说 NC 可泛化到新类；Hui 则指出 NC 可能让 downstream feature 变差。**没查到把 NC 明确落到 person ReID seen-ID 过坍缩作为 open-set ReID 失败变量**。证据：[Papyan 2020](https://arxiv.org/abs/2008.08186)、[Hui 2022](https://arxiv.org/abs/2202.08384)、[Galanti 2021](https://arxiv.org/abs/2112.15121)、[NC-OOD](https://arxiv.org/abs/2311.01479)、[NECO](https://arxiv.org/abs/2310.06823) | 8/10 |
| (b) SSL/contrastive dimensional collapse | **机制撞车，问题存活** | Jing/Hua 已经把“维度坍缩、强相关、低维子空间”讲得很清楚。OSAC 的“低秩化/各向异性化”概念不能当新现象，只能当迁移到 ReID 的诊断变量。证据：[Jing et al.](https://arxiv.org/abs/2110.09348)、[Hua et al.](https://arxiv.org/abs/2105.00470) | 9/10 |
| (c) VICReg / Barlow / whitening / decorrelation | **零件严重撞车** | spectral floor 类似 VICReg variance floor；去相关/白化类似 Barlow Twins、Whitening SSL。不能声称“防 collapse 的谱地板/去相关”新。切口在 supervised open-set ReID 的 seen-ID prototype over-collapse，而非 SSL collapse。证据：[VICReg](https://arxiv.org/abs/2105.04906)、[Barlow Twins](https://arxiv.org/abs/2103.03230)、[Whitening SSL](https://arxiv.org/abs/2007.06346) | 10/10 |
| (d) ProNet / prototype projection / classifier reuse ReID | **强相邻撞车，有杀伤** | ProNet 直接把 ReID classifier 权重当 seen-ID prototypes，并在推理时保留 prototype similarity 做检索特征。这会挑战“seen prototype 对齐一定伤 unseen”。但它没有诊断过坍缩，也不是训练期谱抗坍缩，且测试不是“仍单原始 embedding”。必须做 ProNet/ProNet-like 对照。证据：[ProNet](https://arxiv.org/abs/2308.10717) | 9/10 |
| (e) open-set / DG ReID 是否已有“过坍缩伤 unseen” | **核心切口存活** | Open-set ReID/GOM 已强调 query ID 可能不在 gallery、ReID 是 retrieval+verification；DG ReID 多数讲 domain/style overfit，不是 seen-ID neural collapse。没看到“训练末期 seen-ID prototype collapse 压掉 unseen identity evidence”的诊断。证据：[GOM](https://arxiv.org/abs/2011.11506)、[SNR](https://arxiv.org/abs/2005.11037)、[MetaBIN](https://arxiv.org/abs/2011.14670)、[Style Interleaved](https://arxiv.org/abs/2207.03132) | 8/10 |
| (f) feature whitening / GW / IN 类 ReID | **中等撞车** | AMS 已用 IN + Group Whitening 做 DG ReID；SNR 明确说 IN 会丢 identity-relevant 信息再 restitution；CC-ReID 也有 feature decorrelation。OSAC 要避开“又一个 whitening ReID”叙事，强调 classifier-prototype over-collapse 与谱尾 unseen 证据。证据：[AMS](https://arxiv.org/abs/2210.10409)、[SNR](https://arxiv.org/abs/2005.11037)、[CC-ReID decorrelation](https://arxiv.org/abs/2410.05536)、[RaMoE decorrelation](https://arxiv.org/abs/2105.09156) | 9/10 |
| (g) spectral/rank regularization metric learning | **局部撞车** | OLE 是 deep metric/classification 的 low-rank orthogonal embedding，方向甚至是主动类内低秩；CLOP 用 orthonormal prototypes 防 contrastive collapse；Top-PC removal 在 NLP embedding 后处理也有先例。没撞到“ReID 训练期 spectral floor + Top-PC Dropout 保 unseen 证据”。证据：[OLÉ](https://arxiv.org/abs/1712.01727)、[CLOP](https://arxiv.org/abs/2403.18699)、[All-but-the-Top](https://arxiv.org/abs/1702.01417) | 7/10 |
| (h) open-set recognition 里的 feature collapse | **相邻撞车，不杀** | OSR/OOD 已把 NC/known-class simplex geometry 用来做 unknown rejection，甚至 Deep Simplex classifier 系列利用 NC。多数是“利用 known collapse 区分 unknown”，不是“known collapse 过强会损害 unseen ReID retrieval”。证据：[NC-OOD](https://arxiv.org/abs/2311.01479)、[NECO](https://arxiv.org/abs/2310.06823)、[DMDSC](https://arxiv.org/abs/2605.00675) | 8/10 |

**总 verdict：存活，信心 7/10。**  
杀伤最大的不是某篇完全同题论文，而是三类 reviewer attack：VICReg/Barlow 说机制不新，AMS/SNR/feature decorrelation ReID 说 ReID 已做白化去相关，ProNet 说 seen prototypes 对 unseen retrieval 也可能有用。

OSAC 要站住，实验必须证明三件事：  
1. baseline 训练末期出现 effective rank 下降、谱尾能量下降、top-PC 与 classifier/prototype subspace 对齐增强，并且这些量和 unseen mAP/R1 负相关。  
2. k-reciprocal/re-ranking 只能重排邻域，不能恢复 embedding 谱维度；OSAC 不靠 test-time trick。  
3. OSAC 胜过 VICReg-like variance/cov、GW/whitening、random dropout、early stopping、label smoothing、ProNet/ProNet-like prototype feature 等强对照。
