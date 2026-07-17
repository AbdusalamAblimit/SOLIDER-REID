# exp371 CASD：第二轮系统查新与机制红队裁决

日期：2026-07-14

范围：2022–2026 年遮挡行人重识别、multi-view / multi-shot teacher、同 ID support、pose / geometry privileged distillation、跨图局部融合、关系蒸馏，以及与 LGPA 自有化相关的替代问题定义。

本文只记录可复核来源。证据强度分为：

- **全文/代码级**：可核对采样、forward、loss 与推理协议；
- **公开摘要级**：只使用摘要明确写出的机制；
- **题名/元数据级**：只确认论文存在，不从题名推断公式。

## 一、结论先行

CASD 仍是当前唯一值得完成一次 frozen routing screen 的候选，但它还不是已经成立的自有创新。

最窄的条件性机制只能写成：

> 训练期用 target-only raw pose response 对严格排除 current evidence 与 relation endpoint 的同 ID support 做逐 slot donor allocation，并只迁移该 support 相对 same-image self teacher 产生的关系增量；测试 student 只输入单张 RGB，使用标准 descriptor 与 cosine retrieval。

这条联合差分的三个部分都不能单独 claim：

1. strict LOO 首先是反泄漏协议，必须通过 matched anchor-inclusive control 才能证明它还是有效机制；
2. raw pose-response routing 首先是 reliability weighting，必须超过 total-response scalar、equal、appearance agreement 与 response permutation；
3. support-vs-self relation delta 邻近关系/残差/选择性蒸馏，必须通过 routing × transfer 的 2×2 因子交互证明联合价值。

在 `MVCD` 与 `MHSF` 合法全文未审计前，即使 frozen 与 student 数值全部通过，也只能称“内部机制可行、外部新颖性未决”。

## 二、本轮检索协议

本轮在已有 `research_agent_literature.md` 与 `critical_prior_audit_2026.md` 基础上补做：

1. Crossref 题名检索：
   - `occluded person re-identification multi-view distillation`
   - `person re-identification support set knowledge distillation`
   - `person re-identification multi-shot single-shot distillation`
   - `person re-identification multi-view relation distillation`
   - `occluded person re-identification cross-view alignment`
   - `privileged multi-view single-image person re-identification`
   - `person re-identification other views feature completion`
2. 2026 年题名 sweep：`occluded person ReID`、`person re-identification multi-view`、`person re-identification distillation`；
3. DOI 逐篇核对 Crossref、Unpaywall、Semantic Scholar、出版社公开页与 GitHub exact-title repository search；
4. arXiv exact-title 检索；
5. 对有公开摘要的 Springer 页面核对 JSON-LD 与 Abstract；
6. 对 `Semantic-aware multi-view person image generation` 的 SSRN accepted-version 元数据做合法公开访问检查。SSRN 页面被 Cloudflare 安全验证拦截，本轮没有绕过安全页，也没有把未读正文当证据。

检索结果没有发现第三篇公开可审计、同时覆盖 `strict target exclusion + pose-response slot routing + support-relative relation transfer` 的论文；但下述 closed / 未公开全文工作仍阻止最终 novelty GO。

## 三、五个最危险近邻

| 工作 | 证据层级 | 已覆盖 | 尚未确认/未覆盖 | CASD 风险 |
|---|---|---|---|---|
| [MVCD](https://doi.org/10.1016/j.neucom.2026.133015) | 公开摘要；正文/代码未公开 | LUPI training-only multi-view teacher→single-view student；SGFP、CVPA、RGA、consistency distillation；问题表述已使用 information incompleteness paradox | strict LOO、target-level current exclusion、pose-response routing、support-vs-self relation gain | **未决致命近邻** |
| [MHSF](https://doi.org/10.1016/j.displa.2026.103424) | 题名/元数据；无公开摘要/正文 | 只能确认同一高度重合作者组、同任务、multi-view hierarchical semantic fusion | multi-view 单元、训练/推理形态、是否含 anchor、pose/part、distillation 与 gain 均未知 | **未决致命近邻** |
| [MVI²P](https://arxiv.org/abs/2311.03828) | 全文+官方代码 | 同 ID 多图 CAM purification、可靠性加权 comprehensive feature、multi-view→single-image L2 propagation | teacher 含 anchor；无逐 slot anatomy support；无 support-vs-self gain isolation | **公开直接近邻** |
| [LCR²S](https://arxiv.org/abs/2310.11210) | 全文级 | same-ID other-view support set、current+support enriched teacher、feature+relation KD、single-input student | target 仍含 current；support 随机；无 target-only pose-response routing 与 support-relative correction | **公开直接近邻** |
| [UMTS](https://doi.org/10.1609/aaai.v34i07.6774) | 全文级 | multi-shot comprehensive teacher→single-shot student、heteroscedastic uncertainty KD | teacher 包含 student anchor；无 strict part-wise LOO；无 gain isolation | **公开直接近邻** |

因此不能声称：首次 single-image information incomplete、首次 same-ID multi-view support、首次 multi-view→single-view、首次 pose-free KD、首次 feature/relation KD、首次 reliability aggregation、首次 matching 或首次 visibility routing。

## 四、2025–2026 新增条目裁决

### 4.1 CAM2Former

Zichang Tan et al., [*CAM2Former: Fusion of Camera-specific Class Activation Map matters for occluded person re-identification*](https://doi.org/10.1016/j.inffus.2025.103011), Information Fusion 2025。

- Crossref、Unpaywall 与 Semantic Scholar 确认论文存在且 closed access；Semantic Scholar 明确标记 publisher elided abstract；未找到公开代码或 arXiv。
- 当前只能确认题名中的 camera-specific CAM fusion，不能推断它是否在同 ID 多图之间融合、是否 training-only、是否使用 anchor 或如何蒸馏。
- 它是对 `CAM/reliability/cross-camera fusion` claim 的 **unresolved secondary prior**，但在没有摘要/正文时不能升级为已经覆盖 CASD。

### 4.2 Semantic-aware multi-view person image generation

Jiajun Zhang et al., [*Semantic-aware multi-view person image generation for re-identification*](https://doi.org/10.1016/j.imavis.2026.106056), Image and Vision Computing 2026；accepted version 元数据指向 [SSRN 5616970](https://doi.org/10.2139/ssrn.5616970)。

- Unpaywall 标记 green accepted version；当前公开入口被安全验证页阻断，本轮未取得可读正文。
- 只能确认“semantic-aware multi-view person image generation”题名，不能推断生成图是否作为 teacher/support、是否用于单图 student 或是否有 pose routing。
- 它对“首次制造/利用多视角互补证据”的宽 claim 有风险，但目前不是 CASD 联合机制的已证先例。

### 4.3 SiamID

Punit Sohanvi et al., [*SiamID: multi-perspective keypoint fusion for person re-identification*](https://doi.org/10.1007/s11042-026-21289-4), Multimedia Tools and Applications 2026。

Springer 公开摘要明确写出：使用 multi-perspective views 与 keypoints；提出包含 74 个 subject、front/back/left/right 四视角及身高的 SNK-PU 数据集；使用 Siamese Network。公开摘要没有 training-only teacher、strict LOO support、single-image student 或 support-relative relation transfer。

裁决：它进一步封死“首次 multi-perspective keypoint fusion”宽 claim，但不是当前 CASD 联合机制的直接先例。

### 4.4 Pose-Guided Feature Restoration Transformer

Jiaqi Li et al., [*Pose-Guided Feature Restoration Transformer for Occluded Person Re-identification*](https://doi.org/10.1007/978-981-95-7251-9_30), WISE 2025 / LNCS 2026。

Springer 公开摘要明确写出：visibility-score-guided common/unique feature fusion 用于恢复 occluded keypoint features，随后使用 adaptive directional graph convolution 与 Transformer context integration。

裁决：visibility fusion、keypoint feature restoration、GCN higher-order relation 与 Transformer integration 都不能作为 LGPA 改造的新意。摘要未出现 same-ID other-view support 或 strict LOO student distillation，因此不覆盖 CASD 最窄差分。

### 4.5 题名级 geometry / cue transfer / complementary denoising

以下条目本轮只能做到题名/DOI 级核验，不从题名推断公式：

- [GeoReID: Distilling Structured Geometry Priors for Person Re-Identification](https://doi.org/10.1109/cnml68938.2026.11452325)；
- [Multi-Modal Fine-grained Discriminative Cues Transfer for Efficient Person Re-Identification](https://doi.org/10.1109/tcsvt.2026.3701704)；
- [Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification](https://doi.org/10.1109/tcsvt.2025.3609570)。

它们足以要求论文避免“首次 geometry prior distillation / multi-modal cue transfer / part complementarity”一类宽 claim，但没有证据证明其覆盖 strict same-ID LOO support 或 support-vs-self correction。

## 五、CASD 必补的因果对照

### 5.1 `POSE-SCALAR`

对 donor `j` 计算：

```text
q_j = sum_k raw_response[j,k]
w_j = q_j / sum_m q_m
slot_k = sum_j w_j * slot[j,k]
```

同一个 donor 权重复制到所有 slots。它保留 target person 总热图能量、人体大小/检测质量与 donor-level reliability，只删除 part-specific response allocation。

必须满足：

```text
POSE-RESP - POSE-SCALAR >= 0.3 mAP pp
```

否则 raw response 只是在做图像质量加权，不能称 part-response organization。

### 5.2 routing × transfer 的 2×2 因子矩阵

student 阶段不能只排一串 arm，必须显式包含：

|  | full/no-selector relation | support-vs-self advantage |
|---|---:|---:|
| PART-EQUAL | `E+R` | `E+A` |
| POSE-RESP | `P+R` / `R0` | `P+A` / CASD |

交互量：

```text
I = (P+A - E+A) - (P+R - E+R)
```

至少要求三 seed paired mean `I >= +0.3 mAP` 且 PID-grouped bootstrap 95% CI lower `>0`。否则只能说 routing 或 selective KD 的单项有效，不能把二者的联合写成贡献。

### 5.3 matched anchor-inclusive control

在 `POSE-RESP + ADV` 完全相同的条件下，只把 current image 放回 support。该 arm 不是 `MV-INCL` 的 full-feature 替代，而是 strict LOO 的单变量反事实。

若 inclusive 持平或更好，strict LOO 只能保留为协议卫生，不能写成有效机制。

### 5.4 oracle→student 可兑现性

frozen oracle 固定三个 cross-camera donors，而当前普通 `P×K,K=4` sampler 不保证三张 donor 都跨 camera。正式 student 前必须：

1. 统计普通 sampler 中可满足该条件的比例；
2. 若 `<70%`，不得把 oracle GO 直接解释为可训练机制；
3. 设计 matched sampler control，避免把 sampler 改动混入 CASD 增益。

同时应报告 teacher geometry 的三种 sensitivity：`Q-support/R-self`、`Q-self/R-support`、`Q-support/R-support`。若只有单侧 query augmentation 有效，共享 RGB encoder 的可迁移性仍然可疑。

## 六、三个正交备选方向

这些不是 CASD 失败后可以直接开跑的小变体；每条都必须重新专项查新与做廉价 kill-switch。

### A. PLEC：Pose-local Evidence Robustness Certification

- 问题：在任意结构化局部证据擦除下，身份 retrieval margin 能否获得可验证下界；
- 机制：基于 LGPA blocks 优化预注册 part-erasure set 下的 worst-case relation margin，输出 dropped-slot robustness bound；不做重建、多图 KD 或测试期 matching；
- 必要证据：clean mAP 至少保留旧增益的 80%（约 `+0.72`），真实 pose-erasure certified mAP/radius 超过 random erasure、uniform dropout 与普通 DRO；
- 新颖性风险：certified retrieval、robust metric learning、adversarial erasing；需专项查新。相对 NNCL 的差异是“不编码/恢复，只认证最坏 retrieval geometry”。

### B. CAEF：Continuous Anatomical Equivariance Field

- 问题：既有证据不支持精确离散 anatomy correspondence，却支持连续局部结构；
- 机制：pose 只在训练期定义连续坐标场与等变约束，RGB student 预测该场并积分局部 descriptor；不用同 ID support、KD 或 matching；
- 必要证据：超过 fixed bands、5-slot LGPA、random field 与 discovery/equivariant slot controls；正确 pose 必须明确优于 canonical；
- 新颖性风险：PDiscoNet、Invariant Slot Attention、pose transfer、PAFormer。机制与 CASD 正交，但实现与查新风险高。

### C. CED：Causal Evidence-Delta Distillation

- 问题：不蒸馏 LGPA feature，而蒸馏 LGPA 相对同 checkpoint global 稳定修正的 retrieval events；
- 机制：只在 correct/canonical/shuffled 三种干预共同同向的 teacher-win relation 上训练 compact RGB student；无多图 support 与 visibility routing；
- 必要证据：三 seed corrected-event coverage 足够、broken-event 不升，并超过同质量 full relational KD；
- 新颖性风险：expert-exclusive/residual KD、counterfactual distillation、仓库 exp119–131。外部新颖性弱于前两条，只有事件上界很高时才值得推进。

## 七、最终路线裁决

优先级保持：

1. **CASD**：只完成一次严格的 frozen routing screen；
2. frozen 通过后再写 matched RGB-only student design，加入 2×2、matched inclusive 与 sampler-control；
3. frozen 任一核心 gate 失败，CASD 立即 NO-GO，不扫 temperature、queue、slot 数或 response threshold；
4. 若 CASD 失败，先对 PLEC/CAEF/CED 各做专项查新和 frozen feasibility，不允许把 LGPA 换名继续训练。

最终论文级门槛仍为：

```text
CASD - B0 >= 0.8 mAP
CASD - strongest_control >= 0.5 mAP
旧 LGPA +0.9 的 retention >= 80%（约 +0.72 mAP）
三 seed 同向
RGB-only test descriptor 对 pose/support 输入逐元素不变
多数据集与 matched 768-D 结论同向
```

MVCD/MHSF 任一全文显示已覆盖同一联合机制，外部新颖性仍应直接判负，不能靠术语改名继续。
