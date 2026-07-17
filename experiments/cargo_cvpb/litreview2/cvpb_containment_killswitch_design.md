# 候选 B 零训练 kill-switch 设计（cvpb_containment_killswitch.py）

## 动机
检验 re-frame："航拍-地面 ReID 不是对称匹配，而是物理定向的不确定性包含"。
航拍低清俯视=信息欠定(宽分布)，地面高清=确定(窄分布)，正确的地面证据应落入航拍的不确定性包络内。
不训练任何东西（冻结 swin_fix256，mAP 67.33），纯 frozen + TTA σ + numpy。

## 核心假设（B_CONTAINMENT_DESIGN.md 第四节，任一不成立 B 降级）
1. 航拍确实更欠定：trace(σ_A) ≫ trace(σ_G)，σ 随物理量(bbox 面积↓)变化，合成退化单调升 σ。
2. 正确方向有效：`-KL(N_G‖N_A)` 明显优于 cosine，且优于最佳对称分布距离。
3. 收益来自图像级非对称包含，不是混杂：8 个破坏对照全部掉分。

## 技术方案
- μ = 冻结 Swin 的 BN 特征(L2 归一化前)。cosine baseline 对它 L2 归一化 → 复现 67.33 线。
- σ² = TTA/augmentation 方差：每图 K=12~16 个增广(hflip + RandomResizedCrop scale 0.8-1.0 + 轻 ColorJitter)，按维度求方差。**σ 只来自图像，绝不用 ID label。** floor 1e-4。
- 包含距离(A→G, query 航拍 a, gallery 地面 g)：D = KL(N_g‖N_a)，升序检索。
  `KL(N_g‖N_a)=0.5·Σ_d[ln(σ²_a/σ²_g)+(σ²_g+(μ_g-μ_a)²)/σ²_a − 1]`，均值项除 **航拍方差 σ²_a**(非对称来源)。
- G→A：query 地面(窄)，gallery 航拍(宽) → D=KL(N_query_ground‖N_gallery_aerial)。

## 主比较(5 × A→G/G→A，mAP/R1/mINP)
cosine(μ) / sym-KL / JS / Bhattacharyya / KL(g‖a)正向 / KL(a‖g)反向 / equal-var Mahalanobis。

## 8 破坏对照(A→G mAP，预期全掉)
C1 方向破坏(反向) / C2 对称化(sym-KL,JS) / C3 view-mean σ(每视角常量) / C4 同视角 σ 置换 /
C5 hardness-matched 置换(norm 分桶) / C6 图内 σ 维度打乱 / C7 variance-only & norm-only / C8 分桶集中度。

## 分桶
按航拍 query bbox 面积(area_of=h·w)分 4 桶(CARGO 无 A0/A1/A2 altitude 文件夹，故用面积)，报每桶 cosine vs 正确包含 mAP，预期最小面积桶包含收益最大。

## 诊断
trace(σ_A) vs trace(σ_G)；Spearman(bbox 面积, trace σ)(预期负)；
合成退化(downsample×2/×4, blur)地面 σ 单调上升；
覆盖校准：同 ID 地面 positive vs hard-neg 落在航拍 50/80/95% 区间的 per-dim 比例。

## 对照组
cosine(μ) 即对称强 baseline（= 训练模型的 eval，G→A 应 ≈67）。

## 通过标准
-KL(G‖A) > cosine 且 > 最佳对称；正向 >> 反向；C1/C3/C4/C5/C6/C7 全明显掉；σ_A≫σ_G 且退化单调升；
true-pair 包含距离 << impostor。全过 = 隐藏变量证实；任一关键条不过 = B 降级。

## 审查
codex --search exec：**approve**，KL 方向/矩阵朝向确认正确(rows=aerial query, cols=ground gallery, 均值项除航拍方差)，
无 ID label 泄漏，eval_dist 升序+same pid&cam junk removal 正确。唯一 Low: C3 用 per-dim mean 非 median(spec 本就写"均值方差"，已改 label 为 view-mean)。

## 已知坑
- JS/Bhattacharyya 原 block=256 在 Ng=32268 下 (block,Ng,D) float64 ≈30GB OOM → 改成自适应 block(每 temp ≤0.6GB)。
- 远程 relay(lab-3090 ProxyJump)偶尔 banner 超时，重试即可，非宕机。
- smoke gallery 截到 4000 → cosine A→G mAP 偏高(89.7，distractor 少)；full 32268 才是真值。G→A cosine 66.78 ≈ 67.33 复现成功。
- JS/Bhattacharyya 全库(Ng=32k)即使 block 自适应仍极慢(numpy CPU-bound, ~15min)；全 run 总耗时 ~70min(extract 53min + scoring 17min)。

## 全量结果(K=16, full gallery; /tmp/cvpb_full.log)

cosine baseline A→G=67.41 复现训练模型 67.33 ✓。主比较(A→G / G→A mAP):
- cosine 67.41 / 67.25
- sym-KL 56.74 / 52.50；JS 55.66 / 51.14；Bhattacharyya 44.02 / 41.82(对称分布距离全部 << cosine)
- **KL(g‖a) CORRECT 68.62 / 17.37**(A→G 仅 +1.2 over cosine；G→A 崩到 17.37)
- KL(a‖g) REVERSED 21.04 / 65.71
- **equal-var Maha 67.94 / 65.28**(σ 全设常量, A→G 67.94 ≈ CORRECT 68.62 > cosine)

8 破坏对照(A→G; CORRECT=68.62):
- C1 反向 21.04(掉, 但理由错——见下)
- C2 sym-KL 56.74 / JS 55.66
- **C3 view-mean σ 69.07(不掉, 反升)** ← 图像级 σ 无价值
- **C4 同视角 σ 置换 67.47(≈CORRECT, 不掉)** ← σ 非图像级
- **C5 hardness 置换 66.63(≈CORRECT, 不掉)** ← σ 是难度代理
- C6 维度打乱 12.01(掉)；C7 variance-only 0.03 / norm-only 0.09(≈0)

诊断:
- **trace(σ) AERIAL q=156.96 g=167.47 | GROUND q=171.64 g=172.81 → 航拍 σ 反而比地面低(两侧都是)** ← 假设1 核心前提证伪
- 合成退化 clean=118.93 down2=116.19 down4=115.41 blur=117.55 → **σ 不升反降**(monotonic=False) ← σ≠信息欠定
- Spearman(bbox面积, σ)=-0.136(几乎无关)
- 分桶 delta: b0 +3.03 / b1 -2.74 / b2 +6.15 / b3 -1.57 → **散乱不集中**, 最大增益在 b2(中) 非最小面积桶
- 覆盖: positive 略高 hard-neg(0.222/0.200, 0.389/0.351, 0.542/0.498), 但 margin 极小(唯一方向正确的诊断)

## VERDICT: FAIL(B 降级)
三条核心假设全不成立:
1. **假设1 证伪**: 航拍 σ < 地面 σ(与"航拍更欠定"相反); 退化 σ 反降。σ(TTA方差)= 平滑度敏感性, 非信息欠定。
2. **假设2 部分**: KL(g‖a) A→G=68.62 仅 +1.2 over cosine, 但 equal-var Maha(σ全常量)=67.94 同样 > cosine → "增益"来自 σ-free 距离形式, 不是包含。最佳对称分布距离(sym-KL/JS/Bhatt)全 << cosine。
3. **假设3 证伪**: C3/C4/C5 全不掉(图像级 σ 无价值, 是 view-prior + 难度代理)。
- **方向不对称是检索 artifact**: 真正起作用的是"均值项除以 query 端方差"(A→G query=航拍→好; G→A query=地面→好), 除以 gallery 端方差必崩(高方差 gallery 永远排前)。G→A CORRECT(除 gallery-aerial σ)崩到 17.37 = 铁证。这不是物理包含, 是分母选谁的归一化效应。
→ 转 v_3"观测受限可恢复性"变体或换方向。
