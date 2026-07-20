# 模块/方向候选清单（2026-03-13 更新）

## 当前判断标准
- 只保留同时满足“问题定义更清楚”或“机制明显不同于旧模块拼接”的方向。
- 凡是本质上属于 `再学一个权重 / 再加一个小 attention / 再调 pooling` 的候选，默认不进主线。

## 候选 1：共同可见关键点检索（推荐）
**状态**: `推荐进入下一实验`

### 问题定义
- 当前 `PSG+GCN` branch 已证明对 fusion 有价值，但测试时被 `equal_concat` 压成单一向量。
- 对遮挡 ReID 来说，真正需要的是 **query-gallery pair-specific 的共同可见支撑**，而不是固定拼接。

### 机制草案
- 保留 GCN 增强后的 `17 x C` 关键点特征到测试阶段
- 仅在 query/gallery 共同可靠的关键点上计算局部距离
- 再与 global distance 做混合，而不是直接特征拼接

### 为什么值得做
- 问题层面比 `AFF/LKA` 更明确
- 与 BPBreID/KPR/FRT 的主线更一致
- 能直接解释 `exp030a` 为何“global 不涨、fusion 才涨”

## 候选 2：共同可见关键点驱动的 pair-specific fusion
**状态**: `候补`

### 问题定义
- global 和 keypoint branch 的贡献应当依赖于 query-gallery 的共同可见支撑，而不是单图自适应权重。

### 机制草案
- 先算 keypoint overlap / reliability
- 再把它作为 pair-specific 系数去调 global distance 与 local distance 的组合

### 风险
- 如果写法过于接近“质量加权距离”，容易落回已有工作叙事
- 需要先由候选 1 证明 keypoint-level common-support 确实有效

## 候选 2.5：CSGT（Common-Support-Guided Triplet）
**状态**: `推荐进入下一实验`

### 问题定义
- 当前 retrieval-time 证据已经说明：遮挡 ReID 的关键不只是“有没有局部特征”，而是 **batch 内不同 pair 的共同可见支撑并不相同**。
- 但现有 global triplet 仍把所有正负 pair 当成同一可比性假设下的样本来挖 hardest case。

### 机制草案
- 用 skeleton branch 的 `kp_weights` 构造 batch 内 pairwise common-support overlap
- 在 global branch 上增加一条 support-aware triplet：
  - 优先在 overlap 足够高的 pair 上做 hard mining
  - 找不到时回退到标准 mining
- 默认行为不变，完全由 config 开关控制

### 为什么值得做
- 它不是再加一个 branch 模块，而是把 **pair-specific common support** 迁进训练目标
- 相比单纯 test-time `cvk_hybrid`，它更接近训练端创新
- 相比 `exp036` 的逐关键点 triplet，它利用的是 pair 可比性，而不是把每个关键点独立监督一遍

## 候选 3：AFF（Adaptive Feature Fusion）
**状态**: `降级为备选，不作为主线`

### 降级理由
1. 问题定义偏弱，更像在 fixed fusion 上补一个 learnable gate
2. QPM / PAN / RGANet 一类工作早已覆盖质量估计与自适应加权叙事
3. 当前 `exp035b / exp036 / exp037` 已显示 branch 内部权重/损失微调的收益很弱

## 候选 4：继续做 branch 内部 learnable weighting / extra loss
**状态**: `不推荐`

### 不推荐理由
1. `exp035b`: `score*visibility` 负
2. `exp036`: per-kp triplet 负
3. `exp037`: 截至 epoch 100 仍低于 `exp035a` 同期
4. 文献上这类做法也更像局部调参，而不是问题级创新

## 当前结论
- **主线应从“再调 branch 内部模块”切到“如何利用共同可见关键点支撑”。**
- 后续若继续开实验，优先顺序应为：
  1. 共同可见关键点检索诊断
  2. CSGT（训练端 common-support mining）
  3. pair-specific fusion
  4. 若前几者都失败，再回头考虑 AFF 作为纯工程补充

---

## 2026-03-13 新增候选（来自 ProFD / DPEFormer / SSSC-TransReID）

### 候选 5：Random Rectangle Mask 数据增强
**状态**: `推荐验证（低成本）`

**来源**: SSSC-TransReID (arXiv 2410.15613)

**核心机制**:
- 在标准 RandomErasing 基础上替换为多矩形遮挡策略
- 每次生成多个不重叠的矩形遮挡块，总面积达到目标比例（默认 50%）
- 更逼真地模拟真实遮挡（多个独立遮挡物 vs 单个大遮挡物）

**与 Swin-Tiny 兼容性**: 高（纯数据增强）
**额外显存**: 0（CPU 端增强）
**预期增益**: +0.3~0.6% mAP（SSSC 报告 vs Hide-and-Seek +0.6% R1）
**实现难度**: 低（约 30 行代码）
**优先级**: ⭐⭐⭐

**注意事项**: SSSC 中这个增强配合 SimSiam 自监督一起用。单独使用的增益可能低于 0.6%。

---

### 候选 6：Pose-Aware Masking Consistency (PAMC)
**状态**: `推荐作为主线候选`

**来源**: SSSC-TransReID 框架 + 热图引导思路的原创结合

**核心机制**:
1. 用 ViTPose 热图识别低置信度关键点区域（热图响应 < threshold）
2. 用这些区域生成 pose-guided 遮挡 mask
3. 双分支 SimSiam 风格对比：原图 vs 进一步遮挡版本 → stop-gradient consistency loss
4. 训练模型学习"即使关键点被进一步遮挡，也应保持身份一致特征"

**与 Swin-Tiny 兼容性**: 高（在特征层面 SimSiam，不需要修改 backbone）
**额外显存**: ~2GB（双前向传播 + Projector MLP）
**预期增益**: +0.5~1.5% mAP（基于 SSSC 框架效果类比）
**实现难度**: 中（需要修改数据增强 + 训练引擎 + 新增 Projector）
**优先级**: ⭐⭐⭐⭐

**创新差异点**:
- vs SSSC：随机矩形 → 热图引导 body-aware masking（pose 语义更明确）
- vs PSG：PSG 是 feature-level modulation，PAMC 是 training objective level 的遮挡一致性

---

### 候选 7：Dissimilar Loss（部位多样性正则化）
**状态**: `低成本备选，可作为辅助损失`

**来源**: ProFD (ACM MM 2024)

**核心机制**:
- 计算 batch 内所有 part embedding 对之间的 cosine 相似度矩阵
- 用 softmax 加权（高相似度对权重更大），然后最大化平均相似度（等价于最大化多样性）
- 防止 GCN/KPP branch 的多个 keypoint 特征 collapse 到相同方向

**与 Swin-Tiny 兼容性**: 高（只需 part embeddings 作为输入）
**额外显存**: ~50MB
**预期增益**: +0.1~0.3% mAP（作为辅助正则化）
**实现难度**: 低（约 20 行代码，ProFD 代码可直接复用）
**优先级**: ⭐⭐

---

### 候选 8：PartFeatureDecoder（Cross-Attention Part 解码器）
**状态**: `候补（等待 PAMC 验证后考虑）`

**来源**: ProFD (ACM MM 2024)

**核心机制**:
- 把文本 prompt 替换为 pose-heatmap-guided learnable queries（K 个关键点 query）
- 以热图加权的 spatial tokens 作为 K/V，通过双向 cross-attention 解码出每个关键点的 part 特征
- SemiAttentionDecoder 的双向设计（query→memory + memory→query）比单向 cross-attention 更有表达力

**与 Swin-Tiny 兼容性**: 高（输入 Swin Stage 4 的 spatial tokens）
**额外显存**: ~200-400MB（2层 cross-attention decoder）
**预期增益**: 不确定（理论上比 GCN bilinear sampling 更灵活）
**实现难度**: 高（需要大幅修改模型结构）
**优先级**: ⭐⭐

---

## 推荐优先级总结（更新）

| 优先级 | 候选 | 理由 |
|--------|------|------|
| 1 | PAMC（候选 6） | 问题新+机制新+实现可行+与 PSG 正交 |
| 2 | Random Rectangle Mask（候选 5） | 成本极低，可附加验证 |
| 3 | Dissimilar Loss（候选 7） | 辅助正则化，低成本 |
| 4 | PartFeatureDecoder（候选 8） | 高成本高风险，等待更多证据 |

---

## 2026-03-16 更新：PAA/ROA 之后的主候选重排

### 候选 9：TDPC（Target-Distractor Pose Conditioning）
**状态**: `推荐作为下一周主线`

### 问题定义
- 当前 `PSG/PAA` 默认使用 scene-level max-merge 热图。
- 这对抑制背景有效，但在多人图里会把 **target person** 与 **distractor person** 的姿态线索混在一起。
- `exp070` 的负结果只说明“直接切到 target-only”会丢失 scene context，**不等于 target ambiguity 不重要**。

### 机制草案
1. 保留 `PSG` 使用 `scene_heatmap`
2. 在 `PAA` 路径额外构造：
   - `target_heatmap`
   - `distractor_heatmap = max(non-target persons)`
3. 用 ambiguity score 控制额外注入强度：
   - 单人/低歧义图像时近似退回 `exp066`
   - 多人/高歧义图像时启用 `target-distractor` differential conditioning

可行写法：
- `x = x + Adapter(scene) + a * DeltaAdapter(target, distractor)`
- 或 `x = x + Adapter([target, distractor])`

### 为什么当前它比 CVK / 新 decoder 更值得先做
1. **问题层面更强**：
   - 对齐 KPR 的 `target ambiguity`
   - 对齐 TTPM 的 `non-target pedestrian occlusion`
2. **实现成本可控**：
   - `exp033 / exp034` 已把 target-aware 基础设施准备好了
3. **没有被已有负结果直接证伪**：
   - `exp070` 否定的是 naive `target-only`
   - 不是 `scene + target-distractor conditioning`
4. **更容易在一周内形成像样证据**：
   - overall metric
   - multi-person subset metric
   - ambiguous cases 可视化

### 风险
1. Occluded-Duke 中真正高歧义样本比例可能不够高，整体增益未必大
2. ambiguity score 若定义过粗，会退化成又一个 heuristic gate
3. 若结果只在 subset 上好、全量不涨，需要提前接受“问题更强但 benchmark 总分不大涨”的可能性

### 当前建议优先级
1. `TDPC`
2. 若 `TDPC` 单 seed 2-3 天内无正信号，再回退到 retrieval-time `common-support recovery`
3. 不再继续开新的 PAA 小变体

---

## 2026-07-20 新增候选：ELO-CUR（exp403）

**状态**：`SEALED / VALIDITY PASS / SCIENTIFIC ELO_CUR_MECHANISM_NO_GO`

### 问题定义

exp401 route alive但exp402 wrong-RGB/zero不劣于correct，说明当前static expert route没有sample evidence
所有权。ELO-CUR不再增加普通attention/loss，而是要求evidence生成共享低秩production operator系数，
并用matched complete-execution utility训练correct相对control的优势。

### 单变量机制

- 删除slot-specific static experts；
- 保持rank16/rho/batch/seed/epoch不变；
- `H(e)`拥有逐rank operator coefficients，NULL exact identity；
- compatibility直接进入production delta；
- wrong/generic/NULL只作stop-gradient reference；
- final仍为RGB-only single global descriptor。

### 创新与风险

- 问题/机制/证据门槛=`3/3`；
- CAL/AIM/UCT与dynamic filter/hypernetwork/LoRA是明确近邻；
- novelty风险=`6/10`，不能把dynamic/low-rank/counterfactual原子当贡献；
- 只有full超过clean D0且同时通过semantic margin与all-bypass门，才升级为论文候选。

### 当前证据

standalone CPU正反contract两遍`26/26 PASS`、生产合同`34/34 PASS`、真实batch64 CUDA/AMP
preflight=`16/16 PASS`，唯一fresh e120与七臂全量RGB-only终审也自然完成。测量有效，但correct raw
mAP=`0.569929559315091`低于clean D0；wrong/generic/NULL/all-bypass均不低于correct，semantic与route margin
同为`−7.745354277944e-06` raw mAP，七臂R1/R5/R10完全相同。训练期compatibility/CUR没有转化为final
retrieval ownership。

**裁决**：关闭当前ELO-CUR对象，禁止重跑、补跑、续训或通过调rho/loss/batch/stage、mask及删除control
救活。下一候选必须重新定义最终检索对象或结构所有权，不能把ELO-CUR换名继续。

### exp403后候选审计：terminal concept-only / minimal bottleneck

**状态**：`LITERATURE/CODE AUDIT ONLY / INNOVATION GATE FAIL / NO EXP404`

CHAIR已覆盖concept edit后的归一化retrieval；IntCEM已覆盖干预轨迹与干预后task loss；MCBM已覆盖逐概念
minimality/IB；SupCBM、MM-CBM和Caption Bottleneck Models又分别覆盖hard leakage control、concept-only
相似度和严格隔离语义通道。PDiscoNet的无监督part slot也不能提供external evidence ownership。

**裁决**：`terminal concept-only subspace + minimality + intervention-aware loss`是已有原子的组合，不满足
机制创新门；direct-sum/fixed norm也只能强迫数值扰动，不能证明正确语义所有权。继续查source-attributed
representation与interventional path completeness；没有新的结构原理前不编号、不做CPU/CUDA实验。

### exp403后候选审计：multimodal modality-laziness机制

**状态**：`LITERATURE/CODE AUDIT ONLY / ALL SCREENED OUT / NO EXP404`

本轮把exp403现象与多模态学习中的“强模态可完成任务、弱模态route虽存在却未被使用”对齐，完成以下边界审计：

- UniCat：各模态独立ReID训练，测试时固定concat；不能迁移为当前RGB-only evidence ownership；
- MCR（commit `0da29d0`）：batch latent permutation + JSD/MI + game regularizer，最终仍是加性fusion；
- Data Remixing（commit `80898aa`）：warm-up后按unimodal KL拆样本、置零另一模态并分阶段训练；
- ResTacVLA（项目commit `76250e5`，未放方法源码）：视觉预测触觉后取residual、VQ并按不确定性gate；测试仍需触觉；
- SCOPE：matched/mismatched similarity、batch semantic graph、topology alignment与diffusion residual fusion；
- RCL：逐channel suppression得到reliance profile，再匹配旧/新模型依赖；
- VIGIL：用attention-masked blind path直接优化`seeing > blind`；
- MiMIC/VLM2Rec：分别以single-modality mixin/dropout/ANCE和弱模态负项加权/topology KL缓解retrieval collapse。

**裁决**：独立目标、permutation/MI、数据重混、predictive residual、topology preservation、reliance matching和
full-vs-bypass loss都已有直接近邻。共同缺口是matched wrong donor没有保持自己的正目标，故这些方法不能证明
`correct > wrong > generic/NULL`的source ownership。本轮末曾暂定“correct对current ID为正、wrong对donor ID
也为正、两者经过同一最终descriptor路径”的三方合同；第三轮已确认它只适用于身份充分组件，不能作为当前
semantic evidence的普适准入门，修正见下。当前继续查新，不写config、不占GPU。

### exp403后第三轮候选裁决：无条件donor-ID合同不成立

**状态**：`LITERATURE/CODE AUDIT ONLY / CONTRACT REFINED / NO EXP404`

DG-Net、Hi-CMD与CIFT的公式/代码审计表明，交换分支的身份标签必须跟随身份充分组件：DG-Net交换的是由完整
图像训练出的appearance/ID code，所以生成图可跟随该code的identity；Hi-CMD交换style/extrinsic code时，
身份标签明确跟随prototype/content而非style donor；CIFT只替换graph affinity并保持当前身份目标，不存在
donor身份转移。三者的swap/graph路径也都不是当前teacher-free单图固定descriptor的同构实现。

当前16维evidence只定义support/appearance语义，不具备identity sufficiency。强制`wrong -> donor ID`会诱导
身份泄漏或给不存在的A视觉/B语义组合强贴身份，故撤回它作为普适机制准入条件。可定义的donor semantic
reconstruction又退化为已有swap/cycle auxiliary loss，不能保证final retrieval ownership。

后续候选必须同时给semantic donor一个不泄漏身份的正目标，并让该目标与最终identity descriptor共享不可绕过
结构；在找到这种对象前，innovation gate仍失败，不创建exp404、不占GPU。

### exp403后第四轮候选裁决：缺少realized semantic target

**状态**：`LITERATURE/CODE AUDIT ONLY / IDENTIFIABILITY FAIL / NO EXP404`

NeurIPS 2025 Composed Person Retrieval（FAFA，commit `0cc16936`）给出了合法semantic正目标：reference image与
relative caption组成query，直接检索已经实现该修改的同身份target image。其115万SynCPR训练triplet依赖
LLM、Flux identity-consistent image-pair生成和MLLM过滤；正式测试仍需caption，并使用query-to-gallery token
top-k scorer。2026-07的DiCE-CIR也仍需显式edit text与target-caption proxy。

当前different-PID wrong evidence不是对host A的relative edit，official数据里也没有`target(A,e_B)`。在没有
relative annotation、identity-consistent生成器或测试时semantic query时，composition loss的正目标不可识别；
普通same-ID配对又可被ID trunk忽略。三种补法均已有强近邻或违反当前部署边界，故不形成候选模块。

下一候选只有在official RGB内部构造出可验证的realized semantic target，并让它与最终固定descriptor共享路径，
才可重新过创新门；当前不创建exp404、不做CPU/CUDA/GPU。

### exp403后第五轮候选裁决：equivariance/invertibility仍不闭合

**状态**：`LITERATURE/FORMULA AUDIT ONLY / NO EXP404`

DiP已用已知affine矩阵把原图位置`p`变成解析target `Kp`，再做position-equivariance；最终却丢弃位置预测，
使用pair-specific DiP weight distance。当前16维semantic evidence没有已知群作用，different-PID wrong donor也
不是host的已知变换，故无法构造同类解析target。

Normalizing flow/invertible coupling只保证信息可恢复，不保证latent factor归属；即使teacher监督evidence code，
也没有把该code与final identity ranking唯一绑定。exp402/403已实证这种proxy/final gap。

因此affine consistency、augmentation invariance、invertible flow和part-weighted scorer均不进入exp404。只有
同时定义当前evidence的可验证semantic action与固定最终metric的结构才保留，当前GPU NO-START。
