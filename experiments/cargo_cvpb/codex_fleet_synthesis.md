# 8-Codex 舰队综合(2026-06-23):新方向选定

## 背景
原 4 点 headline 失效(OVP 逐字撞 CMPC/PDPA/MBCE;OVLI-MaxSim 被自己消融证伪——avg 52.37 >> MaxSim 45.19,α 证 MaxSim 贡献≈0)。avg 52.37 是最强单机制但"投影+均值+跨视角对比"故事弱。8 codex 各钻一个 novelty 角度。

## 8 角度 打分 + 裁决
| 角度 | B类分 | 裁决 |
|------|------|----|
| 1 correspondence-free framing | 5→7 | 入口好但要落成具体 CF-OVSP 方法 + 系统反证切开 GSAlign/ViSA |
| **2 可学习集合池化** | — | ★**OVC-SetVLAD**(NetVLAD 残差分布)最可能涨过 52.37;表述必须=跨视角无对应 token 集合的残差分布建模,不能写成普通 NetVLAD |
| 3 方向非对称 A→G≠G→A | 6.5→7.5 | 落在"**非交换 aerial-ground directional retrieval metric**"(不是又一个 view-conditioned projection) |
| 4 OT/Wasserstein | 5 | 撞 CM-EMD/CVFT/DeepEMD,plain OT 不够,不押主线 |
| 5 token reliability gating | 5 | 撞 QPM/PVPM/GSAlign visibility,弱 |
| **6 改装 OVP→ACVP** | 7 | ★prototype 从"对齐目标"改成"**跨视角歧义估计器**",只校准强 contrastive 的负样本(ambiguity-negative relaxation),不做 prototype-positive InfoNCE → 避开 PDPA/CMPC/MBCE |
| **8 综合 meta** | **7.5** | ★★**DCVP**:从点对点跨视角对齐 → **单图反视角身份证据分布预测**;最避撞车,最贴 52.37 实证,kill-switch 清楚(feature-only 推理 >52.37 则成立) |

## ★ 选定方向(三高分角度收敛到同一核心)
核心洞察:**把跨视角身份建成 correspondence-free 的"证据分布",不做点对点对齐。** late-interaction 失败 = 这个洞察的硬证据(负结果驱动正设计)。
- **Headline(角度 8+1)**: **DCVP = Distributional Cross-View identity evidence Prediction**。Motivation: 极端视角差(航拍↔地面)下空间对应失效,所以建分布不建对齐。
- **机制(角度 2)**: **OVC-SetVLAD** = 可学习 VLAD 残差分布聚合 token 集合(分布表示本身就是 distribution),涨过 mean-pool 52.37。
- **加成(角度 6)**: **ACVP** = 把撞车的 OVP 原型改成 ambiguity-negative 估计器,只校准负样本,不撞 PDPA。
- **额外轴(角度 3)**: 非交换 A→G≠G→A directional metric(现有方法都对称处理两视角,这是空白)。

## 下一步(kill-switch 优先级)
1. ★**OVC-SetVLAD**(set-pooling subagent 正在实现 NetVLAD/attn/二阶)→ 跑,看 **>52.37?**(DCVP 分布故事成立的硬判据)
2. 若成:ACVP 负样本校准 + 非对称 directional 轴叠加,看能否再涨
3. Swin/SOLIDER backbone + AG-ReID.v2 跨数据集主表
4. 写 paper 骨架(headline=DCVP,motivation=对应失效的反证,机制=SetVLAD 分布,差异化=非交换+歧义估计)

## 死亡清单(codex 明确否掉,别再碰)
token late-interaction(证伪)、prototype memory 原样(撞车)、visibility/reliability 小补丁(撞 QPM 等)、plain OT(撞 CM-EMD)。
