Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019f0b9e-6fc0-7403-835f-42356a4bc30e
--------
user
自主选了 unlock Top1 Wardrobe-Causal Clothes-Changing/Long-Term ReID(8.6)。动手前细查 novelty + 落地 cheap kill-switch。CC-ReID 是 crowded 子领域。三问:
(a)NOVELTY 边界: 'clothing intervention 下 identity-core evidence vs clothing-residual evidence 可控分解'和已有 CAL(clothes adversarial)/AIM(causal intervention CC-ReID CVPR23)/各种 clothes-identity disentangle 本质区别? 能撑 B 类主创新还是已被占满? web search 查 2023-2025 最新 CC-ReID causal/counterfactual/disentangle/intervention 工作, 列最近 3-5 篇 + 我们必须收窄成什么才 novel。
(b)若 #1 太挤, unlock #2 video-gait(8.4)/#4 RGB-event(8.1) 哪个 novelty 更干净更值得换? 一句判断。
(c)#1 cheap kill-switch 可执行 spec: 哪个换装数据集小且公开可下(PRCC/LTCC/Celeb-reID-light, 给下载源 URL)、frozen DINOv3 抽特征、cross-clothes vs same-clothes oracle headroom 怎么算、'去衣服'用什么(SCHP parsing? DINOv3 region mask?)、per-query #false-in-top10 分桶控 trivial。给能直接写脚本的步骤。
务实中文别套话。
codex
我先查最近几年的 CC-ReID 因果、反事实、解耦和 intervention 论文，优先看论文页/代码仓库，避免只凭旧印象判断 novelty。
web search: 
web search: 2023 2024 2025 clothes-changing person re-identification causal intervention counterfactual disentanglement CC-ReID
web search: 
web search: "AIM" "Clothes-Changing Person Re-identification" causal intervention CVPR 2023
web search: 
web search: "AIM" "CVPR 2023" "person re-identification" clothes-changing
web search: 
web search: "ACID" "AIM" "cloth-changing person re-identification"
web search: 
web search: "Clothes-Based Adversarial Loss" "Clothes-Changing Person Re-identification"
web search: 
web search: PRCC dataset clothes changing person re-identification download
web search: 
web search: PRCC dataset official download cloth-changing person re-identification PRCC
web search: 
web search: https://github.com/guxinqian/Simple-CCReID
web search: 
web search: https://github.com/shuxjweb/pixel_sampling
web search: 
web search: https://isee-ai.cn/~yangqize/clothing.html
web search: 
web search: "Celeb-reID-light" dataset download GitHub
web search: 
web search: "Celeb-reID" dataset download
web search: 
web search: site:github.com "Celeb-reID-light" "Download"
web search: 
web search: https://github.com/moon-wh/MADE
web search: 
web search: https://github.com/moon-wh/MADE
web search: 
web search: DINOv3 Meta official GitHub frozen features dense patch embeddings
web search: 
web search: DINOv3 official code github facebookresearch dinov3
web search: 
web search: "ACID" "Cloth-Changing Person Re-Identification"
web search: 
web search: "Adaptive Information" "Clothes-Changing Person Re-identification"
web search: 
web search: "CVPR 2023" "PRCC" "LTCC" "VC-Clothes"
web search: 
web search: "clothes-changing person re-identification" "CVPR 2023" "github"
web search: 
web search: site:openaccess.thecvf.com CVPR 2023 "Clothes" "person re-identification"
web search: 
web search: "AIM" "clothes" "re-identification" "PRCC"
web search: 
web search: "AIM" "CCVID" "PRCC"
web search: 
web search: https://arxiv.org/abs/2305.06145
web search: 
web search: https://arxiv.org/abs/2411.00330
web search: 
web search: https://arxiv.org/abs/2401.05646
web search: 
web search: https://arxiv.org/abs/2204.06890
web search: 
web search: "Good is Bad" "causality inspired cloth-debiasing"
web search: 
web search: 'AIM' in https://arxiv.org/abs/2305.06145
web search: 
web search: "Good is Bad" "Re-ID"
web search: 
web search: "Causality Inspired" "Cloth-Changing"
web search: 
web search: 2023 2024 2025 RGB-event person re-identification event camera reid
web search: 
web search: "Event Re-ID" person re-identification dataset RGB event
web search: 
web search: 2023 2024 2025 video gait clothes-changing person re-identification CCVID
web search: 
web search: PRCC dataset directory structure camera A B C query gallery protocol
web search: 
web search: https://arxiv.org/abs/2508.10104
web search: 
web search: https://arxiv.org/abs/2507.13659
codex
**(a) Novelty 边界**

结论先说：#1 的大表述已经很挤。  
“clothing intervention 下 identity-core evidence vs clothing-residual evidence 可控分解”如果写成训练端 disentangle / clothes-invariant learning，本质上会撞 CAL、CCIL/AIM 类 causal intervention、FRD/DIFFER/MIPL 这条线，不够稳撑 B 类主创新。

最近相关工作：

1. **CAL, CVPR 2022**：已经用 clothes adversarial loss 惩罚模型预测衣服的能力，从 RGB 中挖 clothes-irrelevant feature。也贡献 CCVID，明确指出视频 temporal/gait 有空间。([arxiv.org](https://arxiv.org/abs/2204.06890))  
2. **CCIL, 2023**：直接把 CC-ReID 建成 clothing confounder 问题，用 `P(Y|do(X))` / backdoor adjustment / confounder dictionary 学 clothes-invariant feature。它已经占了“causal intervention 去衣服 shortcut”的核心话语。([arxiv.org](https://arxiv.org/abs/2305.06145))  
3. **MADE, 2024**：用属性描述，显式 mask clothing/color attribute，再注入 Transformer，目标也是强迫模型丢掉衣服信息。([arxiv.org](https://arxiv.org/abs/2401.05646))  
4. **FRD-ReID, 2024**：标题就已经是 controllable disentanglement，用 human parsing mask 做 reconstruction GT，测试时丢弃 clothing-related features。和“可控分解”非常近。([arxiv.org](https://arxiv.org/abs/2407.10694))  
5. **DLCR, WACV 2025 / DIFFER, CVPR 2025**：DLCR 用 diffusion inpainting 生成换装反事实，保留身份特征；DIFFER 用文本语义监督 + adversarial subspace 分离 identity / non-biometric。([arxiv.org](https://arxiv.org/abs/2411.07205)) ([arxiv.org](https://arxiv.org/abs/2503.22912))  

所以必须收窄：不要叫“clothes-invariant disentanglement”。更 defensible 的说法是：

**query-level evidence accounting / reliability under clothing intervention**：不训练一个新 CC-ReID disentangler，而是用 frozen DINOv3 + parsing，把每个 query-gallery pair 的 `identity-core evidence` 和 `clothing-residual evidence` 拆开，判断什么时候 top-k 是被衣服残差误导，什么时候 query 本身缺少 single-image identity-core support。这能接 exp109 的 single-image support incomplete，而不是重复 CC-ReID 的“去衣服表征学习”。

B 类能不能撑：  
可以尝试，但前提是 kill-switch 证明有明显 headroom，并且贡献落在 **可诊断、可控、可拒识/降权的 evidence mechanism**。如果只是“core feature + cloth feature + gate”，不够。

**(b) #2 / #4 一句判断**

若 #1 判挤，novelty 更干净的是 **#4 RGB-event**；#2 video-gait 更贴近现有代码但已经被 ASGL/skeleton-dynamics/CCVID 路线吃掉不少，#4 数据和工程成本高但子问题空间更新。([arxiv.org](https://arxiv.org/abs/2507.13659)) ([arxiv.org](https://arxiv.org/abs/2402.03716)) ([arxiv.org](https://arxiv.org/abs/2503.10759))

**(c) Cheap Kill-Switch Spec**

首选数据集：**PRCC**。小、公开、协议清楚、A/B 同衣服、C 换衣服，33,698 images / 221 IDs，官方页给 Google Drive 和 Baidu。([isee-ai.cn](https://isee-ai.cn/~yangqize/clothing.html))  
备选：**Celeb-reID-light**，10,842 images / 590 IDs，OneDrive/Baidu 直接下，但衣服标签和协议不如 PRCC 干净。([github.com](https://github.com/Huang-3/Celeb-reID))  
LTCC 小但需签 release agreement/邮件申请，适合作第二轮，不适合最快 kill-switch。([naiq.github.io](https://naiq.github.io/LTCC_Perosn_ReID.html))

执行步骤：

1. 建索引表：`img_path,pid,camid,clothid,split`。  
   PRCC：`A/B` 记同一 `clothid`，`C` 记 cross-clothes `clothid`。只保留 gallery 中至少有 cross-clothes positive 的 query。

2. 用 frozen DINOv3 抽 dense patch feature。DINOv3 适合这里，因为它明确强调 no fine-tuning 下的 dense feature 能力。([arxiv.org](https://arxiv.org/abs/2508.10104))  
   每张图保存：
   - `f_full`: foreground/all-patch pooled
   - `f_core`: non-clothing region pooled
   - `f_cloth`: upper/lower/dress clothing region pooled

3. “去衣服”用 **SCHP parsing**，不要先用 DINOv3 自己猜 mask。  
   PRCC pixel-sampling repo 已经把 SCHP 和 PRCC mask 当标准准备项。([github.com](https://github.com/shuxjweb/pixel_sampling))  
   mask 定义：
   - `cloth = upper-clothes | coat | dress | skirt | pants`
   - `core = person_foreground - cloth`
   - 可保留 `head/hair/face/shoes/skin/limbs`，但报告里要单独列 parsing failure rate。

4. 计算三个检索分数：
   - `s_full = cos(f_full_q, f_full_g)`
   - `s_core = cos(f_core_q, f_core_g)`
   - `s_cloth = cos(f_cloth_q, f_cloth_g)`
   - `s_lambda = lambda*s_core + (1-lambda)*s_cloth`，`lambda` 网格取 `0,0.1,...,1`

5. oracle headroom：
   - `same-clothes`: same pid + same clothid positives
   - `cross-clothes`: same pid + different clothid positives
   - `global-best`: 固定一个全局最优 `lambda`
   - `oracle-query`: 每个 query 用 label 选最优 `lambda`
   - `headroom = R1_oracle_query - R1_global_best`，mAP 同算  
   kill 条件：PRCC cross-clothes `headroom < 3pp R1` 或 hard bucket 没增益，直接停 #1。

6. 控 trivial：先用 `s_full` baseline 排名，每个 query 统计：
   - `false_top10 = # {top10 gallery: pid != qid}`
   - 分桶：`0 / 1-2 / 3-5 / 6-10`
   - 所有方法都在同一桶内报 `R1,mAP,false_top10_mean`
   - 只有 easy bucket 涨、`false_top10>=3` 不涨，则不能讲 clothing-residual evidence。

7. 继续条件：
   - `oracle-query` 在 cross-clothes、尤其 `false_top10>=3` 桶有 `>=5pp R1` 或明确 mAP 提升；
   - false top10 的 `s_cloth` 显著高于随机负样本，且 `s_core` 低；
   - 可视化前 50 个失败例确实是衣服残差误导，而不是 pose/blur/face absence。

否则 #1 判死，不进训练端大实验。
tokens used
113,293
**(a) Novelty 边界**

结论先说：#1 的大表述已经很挤。  
“clothing intervention 下 identity-core evidence vs clothing-residual evidence 可控分解”如果写成训练端 disentangle / clothes-invariant learning，本质上会撞 CAL、CCIL/AIM 类 causal intervention、FRD/DIFFER/MIPL 这条线，不够稳撑 B 类主创新。

最近相关工作：

1. **CAL, CVPR 2022**：已经用 clothes adversarial loss 惩罚模型预测衣服的能力，从 RGB 中挖 clothes-irrelevant feature。也贡献 CCVID，明确指出视频 temporal/gait 有空间。([arxiv.org](https://arxiv.org/abs/2204.06890))  
2. **CCIL, 2023**：直接把 CC-ReID 建成 clothing confounder 问题，用 `P(Y|do(X))` / backdoor adjustment / confounder dictionary 学 clothes-invariant feature。它已经占了“causal intervention 去衣服 shortcut”的核心话语。([arxiv.org](https://arxiv.org/abs/2305.06145))  
3. **MADE, 2024**：用属性描述，显式 mask clothing/color attribute，再注入 Transformer，目标也是强迫模型丢掉衣服信息。([arxiv.org](https://arxiv.org/abs/2401.05646))  
4. **FRD-ReID, 2024**：标题就已经是 controllable disentanglement，用 human parsing mask 做 reconstruction GT，测试时丢弃 clothing-related features。和“可控分解”非常近。([arxiv.org](https://arxiv.org/abs/2407.10694))  
5. **DLCR, WACV 2025 / DIFFER, CVPR 2025**：DLCR 用 diffusion inpainting 生成换装反事实，保留身份特征；DIFFER 用文本语义监督 + adversarial subspace 分离 identity / non-biometric。([arxiv.org](https://arxiv.org/abs/2411.07205)) ([arxiv.org](https://arxiv.org/abs/2503.22912))  

所以必须收窄：不要叫“clothes-invariant disentanglement”。更 defensible 的说法是：

**query-level evidence accounting / reliability under clothing intervention**：不训练一个新 CC-ReID disentangler，而是用 frozen DINOv3 + parsing，把每个 query-gallery pair 的 `identity-core evidence` 和 `clothing-residual evidence` 拆开，判断什么时候 top-k 是被衣服残差误导，什么时候 query 本身缺少 single-image identity-core support。这能接 exp109 的 single-image support incomplete，而不是重复 CC-ReID 的“去衣服表征学习”。

B 类能不能撑：  
可以尝试，但前提是 kill-switch 证明有明显 headroom，并且贡献落在 **可诊断、可控、可拒识/降权的 evidence mechanism**。如果只是“core feature + cloth feature + gate”，不够。

**(b) #2 / #4 一句判断**

若 #1 判挤，novelty 更干净的是 **#4 RGB-event**；#2 video-gait 更贴近现有代码但已经被 ASGL/skeleton-dynamics/CCVID 路线吃掉不少，#4 数据和工程成本高但子问题空间更新。([arxiv.org](https://arxiv.org/abs/2507.13659)) ([arxiv.org](https://arxiv.org/abs/2402.03716)) ([arxiv.org](https://arxiv.org/abs/2503.10759))

**(c) Cheap Kill-Switch Spec**

首选数据集：**PRCC**。小、公开、协议清楚、A/B 同衣服、C 换衣服，33,698 images / 221 IDs，官方页给 Google Drive 和 Baidu。([isee-ai.cn](https://isee-ai.cn/~yangqize/clothing.html))  
备选：**Celeb-reID-light**，10,842 images / 590 IDs，OneDrive/Baidu 直接下，但衣服标签和协议不如 PRCC 干净。([github.com](https://github.com/Huang-3/Celeb-reID))  
LTCC 小但需签 release agreement/邮件申请，适合作第二轮，不适合最快 kill-switch。([naiq.github.io](https://naiq.github.io/LTCC_Perosn_ReID.html))

执行步骤：

1. 建索引表：`img_path,pid,camid,clothid,split`。  
   PRCC：`A/B` 记同一 `clothid`，`C` 记 cross-clothes `clothid`。只保留 gallery 中至少有 cross-clothes positive 的 query。

2. 用 frozen DINOv3 抽 dense patch feature。DINOv3 适合这里，因为它明确强调 no fine-tuning 下的 dense feature 能力。([arxiv.org](https://arxiv.org/abs/2508.10104))  
   每张图保存：
   - `f_full`: foreground/all-patch pooled
   - `f_core`: non-clothing region pooled
   - `f_cloth`: upper/lower/dress clothing region pooled

3. “去衣服”用 **SCHP parsing**，不要先用 DINOv3 自己猜 mask。  
   PRCC pixel-sampling repo 已经把 SCHP 和 PRCC mask 当标准准备项。([github.com](https://github.com/shuxjweb/pixel_sampling))  
   mask 定义：
   - `cloth = upper-clothes | coat | dress | skirt | pants`
   - `core = person_foreground - cloth`
   - 可保留 `head/hair/face/shoes/skin/limbs`，但报告里要单独列 parsing failure rate。

4. 计算三个检索分数：
   - `s_full = cos(f_full_q, f_full_g)`
   - `s_core = cos(f_core_q, f_core_g)`
   - `s_cloth = cos(f_cloth_q, f_cloth_g)`
   - `s_lambda = lambda*s_core + (1-lambda)*s_cloth`，`lambda` 网格取 `0,0.1,...,1`

5. oracle headroom：
   - `same-clothes`: same pid + same clothid positives
   - `cross-clothes`: same pid + different clothid positives
   - `global-best`: 固定一个全局最优 `lambda`
   - `oracle-query`: 每个 query 用 label 选最优 `lambda`
   - `headroom = R1_oracle_query - R1_global_best`，mAP 同算  
   kill 条件：PRCC cross-clothes `headroom < 3pp R1` 或 hard bucket 没增益，直接停 #1。

6. 控 trivial：先用 `s_full` baseline 排名，每个 query 统计：
   - `false_top10 = # {top10 gallery: pid != qid}`
   - 分桶：`0 / 1-2 / 3-5 / 6-10`
   - 所有方法都在同一桶内报 `R1,mAP,false_top10_mean`
   - 只有 easy bucket 涨、`false_top10>=3` 不涨，则不能讲 clothing-residual evidence。

7. 继续条件：
   - `oracle-query` 在 cross-clothes、尤其 `false_top10>=3` 桶有 `>=5pp R1` 或明确 mAP 提升；
   - false top10 的 `s_cloth` 显著高于随机负样本，且 `s_core` 低；
   - 可视化前 50 个失败例确实是衣服残差误导，而不是 pose/blur/face absence。

否则 #1 判死，不进训练端大实验。
