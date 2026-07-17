# AIRL 论文故事线(2026-06-23,双分支训练中)

## 暂定标题
Observation-Limited Identity Recoverability for Aerial-Ground Person Re-Identification

## Motivation(为什么做)
- 现有 aerial-ground ReID 几乎都把跨视角 gap 当成**视角对齐问题**(VDT view-decoupling / GSAlign 空间变形 / ViSA 语义对齐)。
- **我们的洞察**:一个被忽视的主因是**物理像素预算**——航拍视角下行人 bbox 小(CARGO aerial median 仅 ground 的 1/3,跨度 17×),低像素预算让身份信息**物理上不可辨识**,这不是对齐能解决的。
- 诊断证据(kill-switch #1):强 SOLIDER-Swin backbone 上,按 aerial bbox area 分桶,最小桶 A→G mAP 相对顶桶塌陷 +13~19。**强 backbone 也救不了 = 物理问题非 backbone-headroom artifact。**

## 核心贡献(3 点)
1. **重新定义问题**:从"如何跨视角对齐"到"**航拍观测条件下身份信息何时可恢复**"(observation-limited recoverability)。文献空白。
2. **非对称 ground-degradation consistency**:只把高清 ground 图退化到 aerial 像素预算 + 一致性,学"低预算下仍稳定的身份证据"。区别于对称 resolution-invariance(DI-REID/CRReID)。
3. **clean/recover 双证据头 + 固定先验融合**:f_full(保 G→A)+ f_rec(退化一致性,服务低清 A→G),1 forward 软融合。回收方向 trade-off 成净增益。**非动态路由**(区别于 RAR)。

## 实验证据链(kill-switch 阶梯,这是方法稿的硬骨架)
- **#1 诊断**(零训练):area 分桶证明像素预算是主误差源,强 backbone 上仍塌 +13~19。✓
- **#2 最小机制**:最小 AIRL(ground 退化一致性)使 area 最小桶跨 3 粒度 +3.6~+8.4、A→G 方向 +3.15(G→A −3.18,mean 打平)。✓
- **#3 fusion 上界**(零训练):软融合 AIRL+baseline 距离 → mean +1.46(合法 w=0.25)~ per-query oracle +4.96,证明 trade-off 可回收。✓
- **#4 单模型双分支**(进行中):1 forward 内化 #3 融合,验 fuse mean ≥ baseline +1.0。【训练中,门槛 61.84】

## 与 SOTA 对比 narrative(待双分支结果)
- 强 backbone(SOLIDER-Swin)同设置下,AIRL dualbranch vs baseline-Swin(60.84)vs VDT(42.76)/GSAlign/SeCap。
- 卖点不是刷 mean 绝对值(Swin 本身已强),是**机制在低清 A→G 的专项增益 + 净 mean 回收**,且换了问题定义。

## 与 OVLI(已死)的对照——为什么 AIRL 不一样
- OVLI:换 pooling/对比方式,强 backbone 上被 baseline 碾压 = 机制无内在价值。
- AIRL:换**问题定义** + 强 backbone 上诊断证明物理问题 + 三关零训练/训练 kill-switch 全过 + 净增益可达。**第一个三关全过的方法。**

## 待补充实验
- 双分支 fuse mean(#4 决定性)。
- 消融:f_full/f_rec/fuse 三档;kl vs feat(feat 变体在跑);w 敏感性(plateau);非对称 vs 对称退化;有无 consistency。
- 第二数据集 AG-ReID.v2(codex 强调跨数据集必需)。
- 可视化:area 分桶 per-bucket gain;t-SNE clean vs recover 头。

## 现实定位(诚实,codex 共识)
- 双分支若 fuse mean ≥+1.0:CCF-B 候选(+ AG-ReID.v2 + 完整消融)。
- 若 <+1.0:回 ensemble 故事 → 中文核心/workshop。
- **不吹 SOTA mean,主打"observation-limited 新问题 + 低清专项 + 净增益机制"。**
