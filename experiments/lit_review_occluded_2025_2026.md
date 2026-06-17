# 遮挡 ReID 2025-2026 SOTA 文献综述（真读，纠正"空间已死"的过度悲观）

> 起因：用户戳穿我**一篇论文没读就宣布 occluded ReID 搬机制证负**。本文件记录**真正读了的论文**学到的有据现状。
> 进度：精读 2 篇（FLaN-Net IJCAI'25、RMPSNet PR'26）+ workflow 中性读 13 篇（synthesis 因 session limit 失败未拿到）。完整 28 篇遮挡论文中性 mine 待 session reset(18:00 Asia/Shanghai)恢复（脚本 occ-reid-sota-read-wf_8510eeab-24c.js，可 resume）。

## 精读 1：FLaN-Net（Categorical Attention，IJCAI 2025）
- **机制**：CLIP-based（冻结 ViT-B/16 图像+文本编码器）。每张遮挡图做 textual inversion 构造细粒度 prompt `"A photo of a [S*] person with [A*] partially occluded by [O*]"`：S*=主体身份 token、A*=可见属性 token（衣着/配饰，learnable queries+cross-attn）、**O*=遮挡物 token（显式在语言空间建模遮挡物本身）**。cross-attn 对齐 prompt 与可见区域 + 熵不确定性加权融合 {视觉,文本,cross-attn}。
- **结果**：Occ-Duke **65.5 mAP / 75.2 R-1**（CLIP 线 SOTA）。消融：S*→72.6/63.7，+A*→74.6/65.2，+O*→75.2/65.5（遮挡物 token +0.6/+0.3）。
- **新意**：CLIP-ReID 把整图映射成 1 个 pseudo-token（粗）；FLaN-Net 分解成主体+属性+**遮挡物** 三 token。**遮挡物 token 是没人做过的点**。

## 精读 2：RMPSNet（PR 2026）
- **机制**：CLIP-based，两阶段。三模块：
  1. **DMPA**（Dual-Masked Prompt Augmentation）：mask 可学**文本 prompt** token（α=0.5 置零）造两个 masked 版本，inter-text 对比。把遮挡视为**文本侧原型的 partial-information condition** → 遮挡图对齐到稳定文本锚。
  2. **RPE**（Region Prioritized Erasure）：**优先擦下半身**（70% 下半身/30% 全图，因真实遮挡 >70% 在腿部）。**图像+特征/token 双层级**都做（patch-token 也擦 70% 下半身 token）。
  3. **MDO**：特征级 erase+noise(σ0.2)+geom 变换 + 逐步对抗优化（minimax）。
- **结果**：Occ-Duke **65.0 mAP / 76.0 R-1**。

## ⭐ 跨论文有据现状（学到什么）
1. **领域很活，但以 +1~2 mAP 增量推进**（KPR 64.8→RMPSNet 65.0；DPM-SPT 63.8→FLaN-Net 65.5）。**一个有用创新不必范式级——增量机制就能发 B/A。我之前"吸收陷阱→空间已死"是过度泛化（overreach）。**
2. **近两年遮挡 SOTA 整条线是 CLIP/视觉-语言的，Occ-Duke 顶 ~65 mAP，且全都不与 SOLIDER 比较。** 我们 exp255 SOLIDER 栈报 **73 mAP / 83 R-1**——**不同、更高的 backbone 谱系**。所以"beat SOTA"取决于谱系：CLIP 线有空间；我们 SOLIDER 线 mAP 已领先。
3. **重叠核对（诚实）**：RMPSNet 的 RPE（下半身优先遮挡）≈ 我们的 **PLBOA**——图像级我们已做。但 RMPSNet **还在特征/token 级**做（训练时擦 70% 下半身 token），**这个我们没做**，是廉价可试点。
4. **没试过的真新点**：(a) 遮挡物 token 语言建模（FLaN-Net O*）；(b) 文本-prompt masking 当遮挡代理（RMPSNet DMPA）；(c) 特征级区域优先 token 擦除（RMPSNet RPE 特征侧）。多为 CLIP-specific，但 (c) 可移植到任何栈。

## 诚实判断（纠正后）
- 我之前的 10 个 NO-GO 是**真做了的负面实验**，但我据此宣布"整个领域搬机制都死"是**未读文献的过度外推**。
- 真读后：领域在增量进步；近年 SOTA 是 CLIP 线（~65 mAP，低于我们 SOLIDER 73）；新机制存在但多 CLIP-specific 或与 PLBOA 重叠。
- **最具体可试（grounded）**：exp332 = 给 SOLIDER 训练加 RMPSNet 式**特征/token 级下半身优先擦除**（我们只有图像级 PLBOA）。单变量、~半天。
- **待办**：session reset 后跑完 28 篇中性 mine + 扩到 334 篇，拿完整 landscape，再定是否有更强的可移植点。

## ⭐⭐ 28 篇全量中性 mine 完成（session reset 后 resume）——有据 landscape + 3 bets

### 6 大方法家族（Occ-Duke 纯模型 mAP 多在 57-66，DDO 81.8 是 open-set 协议虚高）
- **A 姿态/骨架引导部位对齐**：pose 在**推理期**用→偏低(57-63, MTIPE/Texture/PSCR)；pose 只当**训练监督**→偏高(Adaptive Occlusion-Aware 70.6)。2025-26 趋势=pose/parser 转训练监督、推理 aux-free。
- **B 生成/特征补全**：DDO 潜扩散 inpaint(需合成 paired)、HGTDR recovery token(无监督)、MAHATMA DFC——核心弱点"补出来不保证对"。
- **C 语言/属性/prompt on CLIP**：FLaN-Net 65.5、RMPSNet 65.0、AG-ReID。**三者推理时全把文本机器扔了、用纯图像特征检索**。
- **D auxiliary-free 内部 saliency transformer**：HFLAT 64.7(cls-attn×特征norm 排 patch)。
- **E mask/parsing 当训练监督、推理 mask-free**：Adaptive 70.6。
- **F 数据中心遮挡合成**：NIReID(真车 crop 正激励噪声)、FOSENet(COCO mask 语义放置)、RMPSNet-RPE(下半身 70/30)。

### ⭐ 全领域盲点（28 篇几乎都不做的——这是金矿）
1. **推理是 per-image、点估计、单图**：除 DDO 闭集外，**没有方法在检索时联合推理 query-gallery 对**，也**不输出不确定性**。
2. **遮挡物身份从不在测试期建模**：FLaN-Net 学了 O* 遮挡物 token 但推理时扔掉；**人-遮-人(inter-person)是公认未解失败模式**(Texture/FOSENet 明说)。
3. **无 per-image 可见性**：AG-ReID 自陈属性伪标签是 **identity-level**(同 ID 所有图共享)→遮挡图继承它没显示的属性。多数无测试期 per-image/per-region 可见性。
4. **补全非身份条件**：幻觉无正确性保证。

### 3 个有据 bets（judge 排序）
- **⭐ Bet 1（最强新意）= Pair-conditioned common-visible scoring**：扩展 Adaptive 的 pairwise-min OAM，但用**学习的 per-patch 可见性 logit**(cls-attn，推理免 pose)；匹配时**只在双方可见 patch 的交集上算距离**：dist=Σ_patch (v_q·v_g)·patch_dist / 共享可见 mass。**全语料无人做"学习的 per-patch query-gallery 可见性交集"检索**——让度量本身 pair+可见性条件化。kill-switch: 加可见性头(random-erase mask 监督,免 pose)，测试用 common-visible 加权距离，≥+1.5 mAP 过。**注: 这正是我之前红蓝队"理论判死"的 common-visible-support 方向——但文献证它是开放空白，我那次是 armchair 判死、没做实验。该实测。**
- **Bet 2 = Gallery-as-target completion**：用检索到的 gallery 特征当补全目标(检索-补全反馈)。**廉价 kill-switch=oracle 天花板**: 把遮挡 query 特征换成同 ID 最高可见 gallery 特征重评，≥+4 mAP 过。**直接测 exp109"headroom is a wall"——大概率又是墙，0 训练可证否。**
- **Bet 3（增量）= 推理期保留可见性向量**：别扔 FLaN-Net/AG-ReID 的机器，蒸成 per-image 可见性向量拼进距离。最弱。

### 结论 / 下一步
judge 推荐 **Bet 1**(最 literature-defensible)。它攻击全语料盲点(检索 per-image+对称)。**我要实测 Bet 1**(common-visible-support)——用现成 exp255 部位特征+pose 可见性做廉价 re-score 探针，看重遮挡子集是否 +1.5。这是对"我 armchair 判死的方向"的诚实复检。

## ⭐⭐⭐ 文献盲点 vs 我们已做 的完整对账（关键）
查了自己代码+结果后，文献的 3 个"盲点"与我们的工作对账：
- **盲点1 common-visible-support / pair-conditioned 匹配 = 我们的 CVK/LTCS/LPCS，已做透**：
  - **测试期 CVK-hybrid = +0.8~0.9% mAP（exp039b/040b/045b 三 checkpoint 复核）；MaxSim-hybrid = +1.1%**（更强）。→ **作为测试期 re-score 真有效但增量小、且是 re-ranker（项目规则不当主创新）**。
  - **训练期 common-support 失败**：CSGT(exp047) 中止——pos/neg pair 的 support overlap 几乎相同(≈0.65)，机制无法区分；LPCS(exp141) −5.3% mAP 训练干扰。
  - → **文献 #1 盲点是真实的"已发表空白"，但我们已挖：测试增量小、训练判负。Bet 1 对我们不 fresh。**
- **盲点2/3 测试期建模遮挡物 / inter-person 遮挡 = 真正没试**：FLaN-Net 学 O* 遮挡物 token 却推理时扔掉；inter-person(人遮人) 是公认未解失败；**无人在匹配时携带遮挡物表征去 gate 干扰行人特征**。我们也没做这个。← **这才是 fresh 角度**。
- **盲点4 identity-conditioned 补全**：exp109 已证 oracle headroom 不可实现(墙)。

### 修正后的完整诚实结论
1. 我"领域死"是 overreach（错，已撤回）——领域活、+1~2 增量。
2. 但文献 #1 盲点(common-visible-support)**我们已挖**：测试 +0.8~1.1% 小增量、训练判负。读文献 + 对账避免了重做。
3. **真正没试 + 有文献撑 = 测试期遮挡物/inter-person 建模**（gate 干扰行人特征）——下一个该探的真 fresh 点。但需想清机制(避开 occluder-gate 禁区的退化)再设计，不 armchair 判死也不 armchair 上。

## ⭐⭐⭐⭐ 334 篇全量 mine 综合判决（2026-06-18，自做，零 codex / 零子 agent）

> 死掉的 broad-mine workflow（读完 183/334、筛出 23 跨子领域迁移候选、最终 synthesis 撞 session limit 判空）→ 用 Python 直接从 workflow transcript 的 328 个 agent JSONL 里**扒出 163 篇机制 + 23 候选**（`experiments/transfer_candidates.json`），主上下文自己综合。**不重开 workflow、不开任何子 agent**（避免再爆 rate limit）。Codex 综合因长 prompt + 网络代理卡死 20min 弃用（短 prompt codex 正常）。

### 判决：23 个里 ~18 个塌回我们已判死的类
| 候选(来源子领域) | 迁移机制 | 归属我们哪个死类 |
|---|---|---|
| CSSA(video)、AOANet(occ) | pair-wise s_q·s_g / min(可见权重) | = **common-visible-support / CVK/LTCS/LPCS**（测试 +0.8~1.1% re-rank、训练判负） |
| HFLAT、CFPER、Pose-Skeleton SAM、FLaN-Net entropy、AG-ReID Otsu | CLS-attn×norm / 一阶差分 split / 骨架置信 Gaussian / 熵 / Otsu 阈值 → 可见性软权重 | = **visibility/uncertainty weighting** |
| ColorSketchNet、HGT、Shape-centered ISR、OCCNet、DDO、Cloth-debias SD | 注入缺失/距离加权恢复/图引导补全/扩散特征 inpaint | = **feature completion**（扩散/SD 类还叠 **FM-import**） |
| SCI | CLIP 文本正交投影减 occluder 方向 | = **FM-import CLIP** + 特征对齐 |
| CCIL | Occluder 字典(高斯) + 后门调整 NWGM | = **backdoor de-confound** |
| FusionTex IAS | attention-on-visible 不对称 re-rank | = **re-ranker**（项目规则：re-rank 不当主创新） |
| Mixed-degrade FGW | Fused Gromov-Wasserstein 关系拓扑 OT 蒸馏 | ≈ 特征对齐/蒸馏（关系-拓扑角度略新但 train-time 仍可吸收） |

### 数据中心遮挡增强（家族 F，非严格死类但我们已有图像级 PLBOA）
Noise-injection（真车 crop copy-paste，reader 给强度4）、SAVS（batch 像素池腐蚀 + 特征一致性）、SFE CPDM（逐条带 leave-one-out dropout）。均 train-time 增强正则，低新意（data-centric / family F），可能小涨但**不是新机制**。

### 真正逃出吸收陷阱的（外部信息 / 新任务）—— 只有 2 类
1. **外部 3D 身体先验（SMPL mesh）**：3D-Aided video（deformable cross-attn 用 mesh 投影点当 reference point）、Gait Fields（dense 像素→canonical 顶点对应）。靠**外部 3D 重建**提供与像素无关的身体几何（遮挡处 mesh 仍在）→ 真逃逸吸收陷阱。但 (a) **重**（需 off-the-shelf SMPL/HMR，慢、新依赖），(b) "只在共同可见顶点上匹配"那一半又塌回 CVK。P(有用) 中、P(伪死=半 CVK) 中。
2. **人-遮-人（inter-person）遮挡物建模**：TTPM 的纹理判别通道（人遮人时**骨架失效**——遮挡人同样骨架拓扑——只有纹理 appearance distinctiveness 能分目标/干扰人）+ FLaN-Net 的 O* 遮挡物 token（学了却推理时扔掉）。

### ⭐ 结论：全量 mine **没找到能打败现有 lead 的新机制**，反而**确认了 lead + 反证吸收陷阱**
1. **23 个里 18 个 = 我们已判死类** → 强力反证吸收陷阱，且证明**我们之前扫得够全**（近年 SOTA 的机制就是我们实测杀过的家族：可见性加权/补全/CVK/FM-import/backdoor/re-rank）。这本身是 analysis 论文的硬证据。
2. 唯一真逃逸 = 外部 3D（重、半 CVK）+ inter-person 遮挡物（= 我们早先自定的 lead）。
3. **人-遮-人被文献反复点名为公认未解失败模式**（TTPM / FOSENet / Texture 都明说），且**没有任何论文把遮挡物表征带进测试期 gate 干扰行人**。
4. **冷静提醒（吸收陷阱推论）**：inter-person lead 要做成"不被吸收"，必须 test-time + cross-instance（pair 条件化判断 Q 的遮挡人是否就是 G）→ 这又落进 **CVK/re-ranker 家族**（已证测试期仅 +0.8~1.1%）。所以 **lead 本身大概率也是小增量 re-rank**，不是训练端主创新。
5. **本 session 真正的交付 = 这份 334 篇 mine 的诊断结论**（"近年 occluded-ReID 机制系统性塌回少数几个已被端到端训练吸收的家族"），不是新 SOTA。匹配用户早先判断（领域 +1~2 增量）。
