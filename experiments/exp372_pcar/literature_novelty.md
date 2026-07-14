# exp372 PCAR 文献与代码查新（2026-07-15）

## 最终裁决

PCAR 当前公式不能通过方法级创新门，第一阶段直接 **NO-GO**。

原因不是“字面完全相同的论文已经存在”，而是核心操作仍属于已有的 pose-conditioned additive/masked attention 家族；canonical subtraction 只改变参数化与解释方式，不产生不可归约的新机制。

## 数学归约

候选公式：

\[
L' = L + \gamma[B(P)-B(P_c)]
   = (L-\gamma B(P_c))+\gamma B(P).
\]

当 canonical pose `Pc` 固定时，`-gamma B(Pc)` 是静态 attention bias。普通 additive pose-bias 模块也可以直接把输出定义为 `B(P)-B(Pc)`，因此 canonical subtraction 不扩大表达函数族，只施加 `R(Pc)=0` 的中心化约束。若该项对一行 logits 是常数，则 softmax 后完全抵消。

`zero-init`、少量 heads/layers、保留 untouched heads、可撤销 residual 分别属于常规 adapter 初始化、head selection、语义锚和安全工程属性；它们可提高实验洁净度，但不能单独承担方法级新颖性。

## 直接近邻

| 工作 | 真实机制 | 对 PCAR 的边界影响 |
|---|---|---|
| PeVL, CVPR 2024 | CLIP image encoder + pose encoder；关节构造 weighted pose mask，P2V attention 使用 pose mask 调制 visual attention | 已覆盖“pose 修改 CLIP 内部 attention”的宽泛主张 |
| PAAB/PAAT, 2023 | 在 ViT pose-aware block 中把 pose-pair mask 加到 `QK^T` 后 softmax，再残差写回 token | 已覆盖 pose mask→attention logits→residual token update，并做 random/noise controls |
| PAFormer, 2024 | pose tokens cross-attend patch tokens，姿态热图监督 attention map，输出 part descriptors | 覆盖 pose-supervised attention 与局部聚合，不是 CLIP patch self-attention/global descriptor |
| KPR, ECCV 2024 | keypoint heatmap patch embedding 与 image tokens 逐位置相加，再进入 Swin encoder | 已覆盖 pose-conditioned transformer encoder；不直接改 logits |
| ProFD, MM 2024 | CLIP spatial tokens 出塔后，用 text part prompts 和 hybrid cross-attention decoder提取 part features | 堵住普通“CLIP + spatially supervised part attention”故事 |
| MUVA, 2026 | 在 ReID 中把 grounding body-part mask 转为 `[B×heads,L,L]` 动态 mask，逐层传入 CLIP ViT self-attention | 直接否定“首次在 CLIP-ReID 视觉塔内部动态修改 attention”的主张 |
| PFD, AAAI 2022 | pose heatmap组织 encoder part feature，再作为 transformer decoder K/V | 早期 pose→part transformer decoder 先例 |
| exp012/052/143 | unary pose bias、keypoint pairwise RPE、skeleton geodesic attention | 仓库内部已覆盖 additive/pairwise/骨架 bias，最终分别弱增益或中性 |

## 关键代码核验

### 官方 CLIP-ReID

- 仓库：<https://github.com/Syliz517/CLIP-ReID>，审计 commit `eb1898b72c882875f478bebfc6d41644eece0a5d`
- self-attention：<https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/model/clip/model.py#L165-L186>
- ViT forward：<https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/model/clip/model.py#L200-L240>
- 标准 1280-D descriptor：<https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/model/make_model_clipreid.py#L107-L153>

工程上可在 `ResidualAttentionBlock.attention()` 注入 `[B×12,129,129]` mask；这说明 PCAR 可实现，但“可实现”不等于“有方法级新意”。

### MUVA

- 论文：<https://arxiv.org/abs/2603.14012>
- 仓库：<https://github.com/RikoLi/MUVA>，审计 commit `896526309c3392abc01c4499b792606c3574d3b4`
- 动态 mask 进入 attention：<https://github.com/RikoLi/MUVA/blob/896526309c3392abc01c4499b792606c3574d3b4/model/clip_pat/model.py#L165-L190>
- 构造多头 attention mask：<https://github.com/RikoLi/MUVA/blob/896526309c3392abc01c4499b792606c3574d3b4/model/visual_language_encoder.py#L252-L287>
- 逐层注入：<https://github.com/RikoLi/MUVA/blob/896526309c3392abc01c4499b792606c3574d3b4/model/visual_language_encoder.py#L289-L337>

MUVA 使用 grounding mask、local tokens、全 heads同 mask并输出 CLS+parts，与 PCAR 仍有配置差异；但它已完成 ReID + CLIP ViT + 动态结构 mask + 逐层 self-attention intervention，足以堵住宽泛 headline。仓库无 LICENSE，本项目不得复制其代码。

## 可核验来源

- PeVL：<https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_PeVL_Pose-Enhanced_Vision-Language_Model_for_Fine-Grained_Human_Action_Recognition_CVPR_2024_paper.pdf>
- PAAB/PAAT：<https://arxiv.org/abs/2306.09331>
- PAAB 实现：<https://github.com/dominickrei/PoseAwareVT/blob/main/timesformer/models/vit_poseblock.py#L57-L104>
- PAFormer：<https://arxiv.org/abs/2408.05918>
- KPR：<https://arxiv.org/abs/2407.18112>
- ProFD：<https://arxiv.org/abs/2409.20081>
- PFD：<https://arxiv.org/abs/2112.02466>
- CLIP-ReID：<https://arxiv.org/abs/2211.13977>

## 与“仍有字面差异”的关系

查新未发现一个工作同时满足：official CLIP-ReID、`B(instance)-B(canonical)` 字面公式、少量 heads、标准 global descriptor和六臂因果控制。但这些联合差异主要是实验配置与归因协议，不足以改变以下事实：

1. pose-conditioned attention 已有；
2. pose mask 修改 CLIP attention 已有；
3. ReID 中动态 mask 修改 CLIP ViT attention 已有；
4. canonical subtraction 可归约为普通 additive bias。

因此即便 PCAR 在官方 CLIP-ReID 上涨点，也很容易被审稿人归为 PeVL/PAAB/MUVA 家族的 ReID/CLIP 配置变体，而不是独立的方法贡献。

## 允许保留的资产

六臂反事实协议仍有价值，尤其是 matched cross-image derangement、affine-fitted canonical 和 correct-train/shuffled-train 2×2 对照。但它们是证据设计，不会把已有机制自动变成新机制。

要恢复方法级资格，后续候选必须引入无法写成“实例 pose bias + 静态 bias”的非可分解操作；不得把 canonical subtraction、zero-init、head sparsity 或 global-only 输出重新命名后继续 PCAR 小变体。
