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
