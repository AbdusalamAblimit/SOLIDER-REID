# 论文库挖掘综合结果(20-codex 读 167 篇 + 1-codex 综合,2026-06-24)

## 方法
- 20 lit-codex 各读 ~9 篇 ReID B 类论文(167 篇库,VI-ReID/遮挡/换衣/文本/无监督/3D/终身/diffusion/CLIP)摘要+intro,提创新套路 + 针对团队资产生成候选。
- 1 综合 codex(131k tokens)跨 19 份去重聚类 + 按"避开拥挤可见性 > 切口尖 > 收敛度 > kill-switch 便宜 > 挂资产"重排。

## novelty 约束(关键)
- aerial-ground"可见性"方向已挤死:GSAlign(NeurIPS25 learned 2D visibility mask)/ CVPL(visibility part learning)/ PDPA / geometry-rectify。**收敛最高的"可见表面匹配"必须降权**(论文库 167 篇较老,没覆盖这批最新 AG SOTA,所以 codex 独立想到的人家发了)。

## Top-2 干净候选(非可见性,首推)

### ① Geometry-Aware Ambiguity ReID(问题重定义 + 训练目标机制)★首推
- **headline**: 航拍地面 ReID 不该把所有正样本硬拉近、所有负样本硬推开;航拍低清+3D 视角造成"证据不足/几何不可判别"样本,硬拉硬推 = 学噪声。
- **切口**: vs 假负抑制/标签噪声/难样本挖掘(处理语义假负/标签噪声/通用难样本)→ 我们专处理 **3D 视角造成的几何不可判别正负样本**。
- **kill-switch(零训练)**: 冻 SOLIDER,统计 baseline 错误是否集中在"几何相似负样本 + 几何不相似正样本 + 低清 top-k 歧义集"。错误结构成立 → 立项。
- **挂**: CARGO/AG-ReID.v2/SOLIDER/SMPL。切口尖、成本低、非可见性。

### ② Mixed-View AG-ReID(问题重定义/重设 benchmark)
- **headline**: 真实图库是航拍/地面/多高度/低清混合,非纯 A→G;现有协议问错了问题。
- **切口**: vs VI-ReID mix-modality(二元模态)→ 连续几何观测混合(高度/俯仰/尺度/分辨率)。
- **kill-switch(零训练)**: 构造 mixed-view gallery 扫比例,若比标准协议掉 >2 mAP 且错误来自"同视角异人压过跨视角同人"→ 立项。

## 其余 Top3-5(备选)
3. Geometry-Conditioned Frequency Migration(身份线索随高度/视角在频段迁移,机制)
4. Geometry Coordinate for Lifelong/Unsupervised(存几何坐标非外观,问题重定义+机制)
5. Pair-wise Local Well-posedness(显式 SMPL 物理可解性 + pair-wise 共同可解交集,可见性切口,降权;除非明显赢 GSAlign learned mask 否则当组件)

## 下一步
**先做 ① 零训练错误分析**(frozen SOLIDER eval + 错误结构统计,不需训练):
- 提 baseline 特征 → dist matrix → 排序
- top-1 假阳: 错配 gallery 是否和 query 几何相似(同视角)?
- 真值低 rank(假阴): 真匹配是否几何不相似(跨视角/大尺度差)?
- 若错误显著集中在几何歧义集 → ① 成立,是干净 B 类问题重定义。

## ① kill-switch 结果(2026-06-24, lab-4090 gaitheat env, baseline 复现 A→G 80.73/G→A 81.42≈81.08)
| 检验 | A→G | G→A | 判读 |
|------|-----|-----|------|
| (a) altitude↑难↑ | 83.47→80.35→77.89 | 82.84→81.71→79.37 | ✅成立但**中等**(~3.5-5.6 gap),且 AG-VPReID challenge 已报过 |
| (b) 假阳几何相似 | 仅21%尺度近 | 仅18% | ❌被跨视角混淆(A→G 必然大尺度差),作废 |
| (c) hard-正样本尺度差大 | 2.79 vs 2.20 | 3.61 vs 2.56 | ✅成立,但指向**跨分辨率/尺度不变性**(老赛道) |
| (d) 几何歧义子集更难 | 79.23 vs 83.23=−4.00 | 79.45 vs 85.12=−5.67 | ✅成立但中等 |

**裁决:① 黄灯。动机真实但效应中等(4-5.67 gap),(a)是已知,(c)指向跨分辨率(MRJL 等占),"不该过度对齐"可能误判(模型是提不出尺度不变特征,非过度对齐)。不值得单独主押。**
**kill-switch 基建有效,可复用测 ②。**
