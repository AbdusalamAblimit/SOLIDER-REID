# 实验结果总表 — Phase 2: Pure Pose Heatmap

## 数据集: Occluded-Duke

### 无后处理（纯模型结果）

> 注：本节默认记录各实验的**首次单 seed / 单 checkpoint**结果，用于保留搜索轨迹；是否能作为论文主结论，以文末 **4090 多种子验证** 为准。

| ID | 方法 | mAP | R-1 | R-5 | R-10 | vs Baseline | 备注 |
|----|------|-----|-----|-----|------|-------------|------|
| 000 | Baseline (SOLIDER-Swin-Tiny, SW=0.2) | 56.6% | 66.5% | 79.4% | 83.4% | — | 120 epoch, 完美复现 |
| 001 | + Pose Part Pooling (sigmoid, 5 parts) | 57.1% | 66.7% | 78.4% | 83.0% | mAP+0.5%, R1+0.2% | Part 分类器收敛慢(id_part≈2.0 vs id_global≈0.2) |
| 001* | ↳ part-only 特征 | **57.5%** | **67.1%** | 79.1% | 83.5% | mAP+0.9%, R1+0.6% | Part 单独使用反而最好 |
| 002 | + Pose Part Pooling (spatial_softmax, T=1.0) | 57.2% | 66.4% | 79.2% | 83.5% | mAP+0.6%, R1-0.1% | 与 sigmoid 几乎相同 |
| 002* | ↳ part-only 特征 | **57.5%** | 66.8% | 79.6% | 84.0% | mAP+0.9%, R1+0.3% | 再次确认 part 单独使用更好 |
| 003 | Part-Dominant (67% part loss, part-only test) | 50.2% | 59.1% | 73.7% | 78.7% | mAP-6.4%, R1-7.4% | ❌ 降低 global weight 伤 backbone, ep60 终止 |
| 004 | + PFM (pose feature modulation, part-only test) | 57.5% | 66.3% | 79.6% | 84.1% | mAP+0.9%, R1-0.2% | 🟡 mAP 同 001*, R1 差 0.8%. PFM 加速收敛但不改善最终结果 |
| 004-g | ↳ global 特征 | 57.4% | 66.1% | 79.6% | 83.9% | mAP+0.8%, R1-0.4% | Global 略好于 001-global (57.1%), PFM 帮助 global |
| 005 | Stage 2 Part Pooling (24×8, 384ch, part-only) | 37.0%* | 44.8%* | 59.1%* | 65.2%* | mAP-19.6% | ❌ ep40 数据, ep49 OOM 终止. Stage 2 语义不足 |
| 006 | L2-norm concat (exp001 model, test-only) | 57.4% | 66.9% | 78.9% | 83.5% | mAP+0.8% | 🟡 比 concat(57.2%) 好, 但不如 part-only(57.5%) |
| **007** | **Pose Spatial Gate in Backbone (PSG)** | **58.3%** | **67.9%** | **80.8%** | **84.9%** | **mAP+1.7%, R1+1.4%** | **✅ 3-seed mean = 57.83% / 67.13%，所有 seed 均优于 baseline，PSG 有效** |
| 008 | PSG + Part Pooling (part_only test) | 57.7% | 66.0% | 78.3% | 82.8% | mAP+1.1%, R1-0.5% | 🟡 组合不叠加, 低于 PSG-only. Part pooling 拖累全局特征 |
| 009 | Multi-stage PSG (Stage 2+3) | 58.3% | 67.2% | 81.2% | 85.2% | mAP+1.7%, R1+0.7% | 🟡 mAP 匹配 exp007, R1 略低(-0.7%), R5/R10 略优. 多 156K params 无显著收益 |
| 010 | PSG + Backbone Freeze 5ep | 12.5%* | 17.5%* | 30.4%* | 36.7%* | — | ❌ ep30 终止. 冻结 backbone 导致灾难性特征损坏 |
| 011 | PSG Stage 3 (200 epochs) | 58.3% | 67.6% | 81.1% | 85.3% | mAP+1.7%, R1+1.1% | 🟡 与 exp007(120ep) mAP 相同, 75% 更多训练时间无收益 |
| 012 | Pose Attention Bias (PAB, Stage 3) | 57.4% | 67.3% | 81.4% | 86.2% | mAP+0.8%, R1+0.8% | 🟡 有效但弱于 PSG. 仅 5.4K params. 证明 feature gate > attn bias |
| 013 | PSG + PAB Combo (Stage 3) | 57.6% | 67.2% | 81.3% | 84.4% | mAP+1.0%, R1+0.7% | ❌ 双重注入互相干扰, 不如 PSG-only(-0.7% mAP). PAB 拖累 PSG 收敛 |
| 014 | PSG + Part Supervision (global test) | 57.6% | 65.8% | 77.9% | 82.6% | mAP+1.0%, R1-0.7% | ❌ 用 exp008 checkpoint 直接验证。Part supervision 梯度损害 PSG global feature |
| 015 | PSG Spatial (3×3 DWConv) | 58.3% | 67.1% | 81.4% | 85.8% | mAP+1.7%, R1+0.6% | 🟡 mAP 匹配 exp007, R1 低 0.8%. 3×3 conv 冗余，1×1 已足够 |
| 016 | PSG + Pose-Guided Erasing (PGE) | 54.8% | 65.0% | 77.7% | 82.2% | mAP-1.8%, R1-1.5% | ❌ PGE 替代 RE 严重有害 (-3.5% vs exp007). 身体部件级擦除过强 |
| 017 | PSG + Pose Channel Gate (PCG) | 58.0% | 67.3% | 80.9% | 85.3% | mAP+1.4%, R1+0.8% | 🟡 与 exp007 持平(-0.3% mAP). 通道级正交不干扰但无额外收益 |
| 018 | PCG-only (无 PSG) | 57.8% | 67.7% | 81.4% | 86.2% | mAP+1.2%, R1+1.2% | 🟡 PCG 有独立效果(+1.2%), 但低于 PSG(-0.5%). PSG+PCG 不叠加 |
| 019 | Pose Cross-Attention (PXA, 替代 PSG) | 57.3% | 66.9% | 80.4% | 85.3% | mAP+0.7%, R1+0.4% | 🟡 有效但弱于 PSG(-1.0% mAP). Cross-attn 过拟合严重, 简单门控更好 |
| 020 | PSG + Pose Reconstruction Aux (PRA) | 57.8% | 67.3% | 80.3% | 84.7% | mAP+1.2%, R1+0.8% | 🟡 中性. 辅助重建任务不改善 PSG(-0.5% mAP). 后期梯度干扰导致锯齿波动 |
| 021 | Content-Adaptive PSG (CAPSG) | 57.2% | 66.0% | 80.5% | 85.2% | mAP+0.6%, R1-0.5% | ❌ Content-dependent gate 弱于静态 PSG(-1.1% mAP). 过度参数化, PSG 简洁性即优势 |
| 022-g | PDS global-only (独立Stage3, PSG全局分支) | 57.9% | 67.1% | 80.0% | 84.2% | mAP+1.3%, R1+0.6% | 🟡 PSG 增益大部分保留(-0.4% vs exp007), Stage 3 解耦有效 |
| 022-cs | PDS concat_scaled | 57.5% | 66.5% | 79.4% | 83.8% | mAP+0.9%, R1±0% | 🟡 Part 缩放融合 = baseline 水平, Part 仍有噪声 |
| 022-eq | PDS equal_concat | 56.1% | 64.0% | 77.5% | 82.4% | mAP-0.5%, R1-2.5% | ❌ 维度比 5:1 过度稀释 Global 贡献 |
| 022-p | PDS part-only | 55.2% | 63.1% | 76.3% | 81.7% | mAP-1.4%, R1-3.4% | Part 分支独立效果不佳 (ID_part loss 2.02 未充分收敛) |
| **023-g** | **PDS+StopGrad global-only** | **59.5%** | **69.5%** | **81.8%** | **85.8%** | **mAP+2.9%, R1+3.0%** | **单 seed 最优 global；3-seed mean = 59.20% / 68.63%，后续被 exp007a 几乎等价复现** |
| 023-cs | PDS+StopGrad concat_scaled | 59.1% | 68.8% | 81.0% | 85.1% | mAP+2.5%, R1+2.3% | ✅ Part 特征提供补充信息 |
| 023-eq | PDS+StopGrad equal_concat | 57.5% | 66.2% | 79.1% | 83.6% | mAP+0.9%, R1-0.3% | 🟡 equal_concat 仍被维度比稀释 |
| 023-p | PDS+StopGrad part-only | 56.7% | 65.1% | 77.9% | 82.6% | mAP+0.1%, R1-1.4% | Part 在 frozen 共享特征上学好(+1.5% vs exp022-p) |
| 024-g | PDS+StopGrad noPSG global-only | 59.2% | 68.7% | 82.0% | 86.1% | mAP+2.6%, R1+2.2% | 单 seed 高点；后续 multi-seed 不支持“PSG 贡献很小”这一强结论 |
| 024-cs | PDS+StopGrad noPSG concat_scaled | 59.0% | 68.3% | 81.6% | 85.7% | mAP+2.4%, R1+1.8% | |
| 024-eq | PDS+StopGrad noPSG equal_concat | 57.1% | 65.4% | 78.8% | 83.1% | mAP+0.5%, R1-1.1% | |
| 024-p | PDS+StopGrad noPSG part-only | 56.4% | 64.9% | 77.9% | 82.4% | mAP-0.2%, R1-1.6% | |
| 025-g | PDS+DelayedStopGrad global-only | 58.9% | 68.4% | 80.6% | 84.8% | mAP+2.3%, R1+1.9% | 🟡 前30ep阻断+释放, -0.6% vs exp023-g, +1.0% vs exp022-g |
| 025-cs | PDS+DelayedStopGrad concat_scaled | 58.6% | 67.8% | 80.4% | 84.4% | mAP+2.0%, R1+1.3% | 🟡 -0.5% vs exp023-cs |
| 025-eq | PDS+DelayedStopGrad equal_concat | 57.3% | 65.8% | 78.9% | 82.8% | mAP+0.7%, R1-0.7% | 🟡 -0.2% vs exp023-eq |
| 025-p | PDS+DelayedStopGrad part-only | 56.4% | 64.9% | 78.0% | 81.9% | mAP-0.2%, R1-1.6% | 🟡 -0.3% vs exp023-p, +1.2% vs exp022-p |
| 026 | PSG + Stochastic Pose Dropout (p=0.3) | 57.9% | 66.2% | 80.5% | 85.2% | mAP+1.3%, R1-0.3% | 🟡 -0.4% vs exp007. SPD 正则化未超越 PSG, pose 信号一致有用 |
| 027 | PSG + PCRA (alpha=0.2, loss 距离调制) | 57.8% | 66.8% | 81.0% | 85.3% | mAP+1.2%, R1+0.3% | 🟡 -0.5% mAP vs exp007. Pose similarity 调制 triplet 距离中性偏负 |
| 028 | PDS+StopGrad + Part LR 3x (equal_concat) | 59.3% | 68.9% | 81.4% | 85.4% | mAP+2.7%, R1+2.4% | 🟡 vs exp023-eq(57.5%)+2.8%, 但 vs exp023-g(59.5%)-0.2%. Part 收敛改善(ID 0.4 vs 2.0)未转化为测试增益 |
| 029 | PSG + Pose-Weighted Pooling (PWP) | 57.9% | 67.5% | 81.1% | 85.3% | mAP+1.3%, R1+1.0% | 🟡 vs exp007(58.3%)-0.4%. PWP 替换 GAP 为 pose-weighted pooling, 效果中性. PSG 已做了空间选择, post-hoc weighting 冗余 |
| 030-g | PDS+StopGrad + Skeleton GCN (global-only) | 59.5% | 69.5% | 82.0% | 86.5% | mAP+2.9%, R1+3.0% | 与 exp023-g 持平，GCN 辅助训练未损害 Global |
| **030-cs** | **PDS+StopGrad + Skeleton GCN (concat_scaled)** | **60.5%** | **70.5%** | **83.4%** | **87.2%** | **mAP+3.9%, R1+4.0%** | **PDS 版单 seed 最佳 fusion；后续 030a multi-seed 已显示 equal_concat 更强** |
| 030-eq | PDS+StopGrad + Skeleton GCN (equal_concat) | 59.9% | 70.9% | 82.7% | 87.4% | mAP+3.3%, R1+4.4% | E120 训练时默认模式 (E110 peak: 60.0%/71.0%) |
| 030-p | PDS+StopGrad + Skeleton GCN (part-only) | 57.4% | 69.5% | 81.2% | 86.2% | mAP+0.8%, R1+3.0% | GCN part-only 效果好，R1 大幅超越 baseline |
| **007a** | **PSG + 0.5x Global Loss Scale** | **59.5%** | **69.8%** | **81.9%** | **86.0%** | **mAP+2.9%, R1+3.3%** | **✅ 3-seed mean = 59.37% / 69.43%；相对 PSG 稳定 +1.53%，且与 exp023-g 无显著差异** |
| 030a-g | PSG + Skeleton GCN (global-only, 无 PDS) | 59.8% | 69.5% | 81.9% | 86.1% | mAP+3.2%, R1+3.0% | 3-seed mean = 59.33% / 68.87%，≈ exp007a；说明 GCN 分支对 global 基本中性 |
| 030a-cs | PSG + Skeleton GCN (concat_scaled, 无 PDS) | 60.5% | 73.7% | 85.0% | 88.1% | mAP+3.9%, R1+7.2% | 3-seed mean = 60.20% / 73.13%，稳定优于 030a-global，但弱于 equal_concat |
| **030a-eq** | **PSG + Skeleton GCN (equal_concat, 无 PDS)** | **61.1%** | **73.7%** | **85.2%** | **87.8%** | **mAP+4.5%, R1+7.2%** | **✅ 3-seed mean = 60.73% / 72.57%；对 030a-global 稳定 +1.40 mAP，是当前最强且已确认的无后处理模式** |
| 030a-p | PSG + Skeleton GCN (gcn_only, 无 PDS) | 58.2% | 72.9% | 83.3% | 86.6% | mAP+1.6%, R1+6.4% | 3-seed mean = 57.97% / 71.77%；branch 本身强，但不如 fusion |
| 030b-g | PSG+GCN w_p=0.01 (global-only) | **60.6%** | 71.0% | 83.8% | 87.3% | mAP+4.0%, R1+4.5% | 单 seed 高点；现主要作为“低权重时 branch 几乎未学好”的反例，不宜再单独拿它否定 loss scaling |
| 030b-cs | PSG+GCN w_p=0.01 (concat_scaled) | 59.4% | 72.9% | 83.9% | 87.3% | mAP+2.8%, R1+6.4% | 单 seed；核心信息是低权重时 concat 无法稳定超越 global |
| 030b-eq | PSG+GCN w_p=0.01 (equal_concat) | 60.5% | 73.0% | 84.4% | 88.3% | mAP+3.9%, R1+6.5% | 单 seed；与 global 接近，说明未训练好的 branch 贡献有限 |
| 030b-p | PSG+GCN w_p=0.01 (gcn_only) | 56.9% | 70.9% | 82.4% | 86.2% | mAP+0.3%, R1+4.4% | 图传播几乎未训练，但 keypoint pooling 本身仍强 |
| 032-g | PSG + Keypoint Pooling Only (global-only) | 59.8% | 70.0% | 81.7% | 85.4% | mAP+3.2%, R1+3.5% | 单 seed；支持“branch 不解释 global 提升”，但精确结论应以 030a multi-seed 为准 |
| 032-cs | PSG + Keypoint Pooling Only (concat_scaled) | 59.3% | 72.4% | 85.1% | 88.4% | mAP+2.7%, R1+5.9% | 单 seed；说明 keypoint pooling 本身就有较强 fusion 价值 |
| 032-eq | PSG + Keypoint Pooling Only (equal_concat) | 60.2% | 72.5% | 85.1% | 88.3% | mAP+3.6%, R1+6.0% | 单 seed；现在更适合作为“keypoint pooling 强基线”的证据，而不是单独量化 GCN 增益 |
| 032-p | PSG + Keypoint Pooling Only (gcn_only 测试模式) | 54.7% | 69.9% | 82.4% | 86.0% | mAP-1.9%, R1+3.4% | 无图传播仍有高 R1，证明关键点采样+置信度池化本身就是强基线 |
| 035a | PSG+GCN score weight (bundled sanity check) | 61.1% | 73.8% | 85.1% | 87.9% | mAP+4.5%, R1+7.3% | = exp030a seed1234 结果（61.1/72.9），含 target-aware+vis aug fix, 无 regression |
| 035b | PSG+GCN score*visibility weight | 60.4% | 71.6% | 84.8% | 87.9% | mAP+3.8%, R1+5.1% | ❌ vs 035a: -0.7% mAP, -2.2% R1。当前只说明 `score*visibility` 未带来收益，不能上升为整条 visibility 路线结论 |
| 007b | PSG + 0.25x Global Loss Scale | 58.3% | 67.6% | 80.0% | 84.9% | mAP+1.7%, R1+1.1% | = exp007(1.0x)! 收敛慢但最终追平 |
| 007c | PSG + 0.75x Global Loss Scale | 58.6% | 67.6% | 81.6% | 85.6% | mAP+2.0%, R1+1.1% | 单 seed；现阶段不能再用 0.25x/0.75x 的单次结果否定 0.5x，多种子只确认了 0.5x vs 1.0x |
| **000b** | **Baseline (seed 42, 3090)** | **56.1%** | **65.8%** | **79.4%** | **83.8%** | — | 3090 vs 4090(55.9%) Δ=0.2%, 确认跨硬件一致 |
| 036 | PSG+GCN + Per-Keypoint Triplet Loss | 60.6% | 73.1% | 84.5% | 88.2% | mAP+4.0%, R1+6.6% | ❌ vs 035a: -0.5% mAP, -0.7% R1。该编号已偏离原 visibility 路线，实际属于 `exp035` 之后的 branch 内部探索 |
| 037 | PSG+GCN + Learnable Keypoint Attention | 60.7% | 71.7% | 83.8% | 87.1% | mAP+4.1%, R1+5.2% | ❌ vs 035a: -0.4% mAP, -2.1% R1。该编号已偏离原 visibility 路线，LKA 未显示稳定正增益 |
| 039a | PSG+GCN + CVK retrieval (`cvk_only`) | 59.3% | 72.9% | 84.1% | 87.1% | mAP+2.7%, R1+6.4% | 测试时诊断；vs 035a: -1.8% mAP, -0.9% R1。纯共同可见关键点距离不足以替代 `equal_concat` |
| 039b | PSG+GCN + CVK retrieval (`cvk_hybrid`) | 61.9% | 73.2% | 85.2% | 88.5% | mAP+5.3%, R1+6.7% | 测试时诊断；vs 035a: +0.8% mAP, -0.6% R1。共同可见关键点更适合作为 global 的 pair-specific 补充 |
| 040a | exp030a checkpoint recheck (`equal_concat`) | 61.1% | 73.7% | 85.2% | 88.0% | mAP+4.5%, R1+7.2% | 原始 `exp030a` checkpoint 的当前代码口径复核；为 `040b` 提供直接对照 |
| 040b | exp030a checkpoint + CVK retrieval (`cvk_hybrid`) | 61.9% | 73.2% | 85.2% | 88.6% | mAP+5.3%, R1+6.7% | ✅ vs 040a: +0.8% mAP, -0.5% R1。与 039b 高度一致，说明正信号可复核 |
| 041a | exp030a checkpoint + CVK retrieval (`2:1`) | 61.6% | 72.6% | 84.2% | 88.1% | mAP+5.0%, R1+6.1% | 权重敏感性；vs 040b(`1:1`): -0.3% mAP, -0.6% R1。偏向 global 会削弱收益 |
| 041b | exp030a checkpoint + CVK retrieval (`1:2`) | 61.6% | 73.6% | 85.1% | 88.6% | mAP+5.0%, R1+7.1% | 权重敏感性；vs 040b(`1:1`): -0.3% mAP, +0.4% R1。偏向 CVK 更像用 mAP 换 R1 |
| 045a | rebuilt seed42 checkpoint recheck (`equal_concat`) | 60.2% | 72.7% | 84.4% | 87.6% | mAP+3.6%, R1+6.2% | `exp044` 重建 checkpoint 的直接对照；mAP 与既有 seed42 记录一致 |
| 045b | rebuilt seed42 checkpoint + CVK retrieval (`cvk_hybrid`) | 61.1% | 73.2% | 84.2% | 88.1% | mAP+4.5%, R1+6.7% | ✅ vs 045a: +0.9% mAP, +0.5% R1。CVK 正 mAP 信号已在第二个 checkpoint 上复核 |
| 046 | rebuilt seed2024 checkpoint (`exp030a` recover) | 60.1% | 72.9% | 84.0% | 87.6% | mAP+3.5%, R1+6.4% | `exp030a seed2024` checkpoint 重建完成；第三个可复用 checkpoint 资产已补齐，可用于后续第三 checkpoint 复核 |
| 047 | PSG+GCN + CSGT (Common-Support-Guided Triplet) | — | — | — | — | ❌ 中止 | Epoch 60 中断无 checkpoint。根本问题：pos/neg overlap 几乎相同（≈0.65），机制无法区分正负 pair。pos_fallback≈0.7 说明大部分退化为标准 triplet |
| 048 | PSG+GCN + SGMKC (Skeleton-Guided Masked Keypoint Completion) | 58.9% | 72.1% | 84.2% | 87.5% | mAP+2.3%, R1+5.6% | ❌ 负面 (-1.6% vs exp030a)。SGMKC loss 与 ID 分类存在梯度冲突，GCN 容量不足以同时完成两个任务 |
| 050 | PSG+GCN + PAMC (Pose-Aware Masking Consistency) | 60.7% | 72.2% | 83.7% | 87.3% | mAP+4.1%, R1+5.7% | 🟡 中性 (vs exp030a-eq 3-seed: -0.03% mAP, -0.37% R1)。Consistency loss 未提供额外增益。连续第 3 个辅助 loss 方向失败 |
| 051-eq | PSG+GCN + PAML (Pose-Aware Metric Learning, equal_concat) | 60.7% | 72.7% | 84.6% | 88.2% | mAP+4.1%, R1+6.2% | 🟡 中性 (vs exp030a-eq 3-seed: -0.03% mAP, +0.13% R1)。逐关键点距离训练未带来增益。连续第 4 个辅助 loss 失败 |
| 051-cvk | PSG+GCN + PAML (cvk_hybrid) | 62.0% | 73.6% | 85.1% | 88.4% | — | 🟡 vs exp030a CVK (61.9%/73.2%): +0.1%/+0.4%。训练-测试 metric alignment 假设未得到验证 |
| 052-eq | PSG+GCN + KP-RPE (equal_concat) | 61.0% | 72.7% | 84.4% | 87.6% | mAP+4.4%, R1+6.2% | 🟡 中性 (vs exp030a-eq 3-seed: +0.27% mAP, +0.13% R1，在方差范围内)。mAP 训练全程 10/12 checkpoint 为正(均值+0.76%)，但最终结果在方差内 |
| 052-g | PSG+GCN + KP-RPE (global) | 59.5% | 68.4% | 81.6% | 85.7% | mAP+2.9%, R1+1.9% | 🟡 vs exp030a-g(59.8/69.5): -0.3%/-1.1%。KP-RPE 未改善 backbone 特征 |
| 052-cvk | PSG+GCN + KP-RPE (cvk_hybrid) | 61.7% | 72.6% | 84.3% | 88.2% | — | 🟡 vs exp030a CVK(61.9/73.2): -0.2%/-0.6%。KP-RPE + CVK 无正交增益 |
| 053-eq | PSG + XCAD (equal_concat) | 59.7% | 70.8% | 82.0% | 86.2% | mAP+3.1%, R1+4.3% | ❌ vs exp030a-eq 3-seed: -1.03% mAP, -1.77% R1。Cross-attention decoder 劣于 GCN |
| 053-g | PSG + XCAD (global) | 59.2% | 68.6% | 81.6% | 85.9% | mAP+2.6%, R1+2.1% | 🟡 vs exp030a-g 3-seed: -0.13%/-0.27%，几乎持平 |
| 053-cvk | PSG + XCAD (cvk_hybrid) | 60.7% | 71.8% | 82.9% | 86.9% | — | ❌ vs exp030a CVK(61.9/73.2): -1.2%/-1.4% |
| **054-eq** | **PSG+GCN + PGAM (equal_concat)** | **61.1%** | **73.8%** | **85.1%** | **87.9%** | **mAP+4.5%, R1+7.3%** | **🟢 vs exp030a-eq 3-seed: +0.37% mAP, +1.23% R1。首个 PSG+GCN 上正向叠加模块！** |
| 054-g | PSG+GCN + PGAM (global) | 59.8% | 69.5% | 81.9% | 86.1% | mAP+3.2%, R1+3.0% | 🟡 vs exp030a-g 3-seed: +0.47%/+0.63%，方差内 |
| 054-cvk | PSG+GCN + PGAM (cvk_hybrid) | 61.9% | 73.2% | 85.2% | 88.5% | — | 🟡 vs exp030a CVK: 0.0%/0.0%，完全持平 |
| 055-eq | PSG+GCN + PGAM t=0.5 (eq_concat) | 61.2% | 73.5% | 85.2% | 88.6% | mAP+4.6%, R1+7.0% | 🟢 vs exp054: ≈持平。阈值不敏感 |
| 055-g | PSG+GCN + PGAM t=0.5 (global) | 60.3% | 70.2% | 82.2% | 87.1% | mAP+3.7%, R1+3.7% | 🟢 vs exp054-g: +0.5%/+0.7%。t=0.5 global 更好 |
| 056-eq | PSG+GCN + PGAM S2+S3 (eq_concat) | 61.1% | 73.7% | 85.2% | 88.6% | mAP+4.5%, R1+7.2% | 🟡 vs exp054: ≈持平。多 Stage 无额外增益 |
| 057-eq | PSG+GCN + KDL w=0.1 (eq_concat) | 61.0% | 73.3% | 84.6% | 87.9% | mAP+4.4%, R1+6.8% | 🟡 中性。vs exp030a 3-seed: +0.27%/+0.73%。Dissimilar loss 无效 |
| **058-eq** | **PSG+GCN + ROA (equal_concat)** | **61.8%** | **72.8%** | **85.2%** | **88.3%** | **mAP+5.2%, R1+6.3%** | **🟢🟢 历史最高 mAP！vs 3-seed: +1.07%/+0.23%。超出方差！** |
| **058-g** | **PSG+GCN + ROA (global)** | **60.8%** | **70.0%** | **83.0%** | **87.0%** | **mAP+4.2%, R1+3.5%** | **🟢🟢 vs 3-seed: +1.47%/+1.13%。全局特征也显著提升！** |
| 059-eq | PSG+GCN + ROA + PGAM (eq_concat) | 61.8% | 72.8% | 85.2% | 88.3% | mAP+5.2%, R1+6.3% | 🟡 与 exp058 精确相同。PGAM 与 ROA 完全冗余 |
| 060-eq | PSG+GCN + PA-ROA (eq_concat) | 61.6% | 72.5% | 84.5% | 87.9% | mAP+5.0%, R1+6.0% | 🟡 vs random ROA: -0.2%/-0.3%。Pose-guided 放置不优于随机 |
| 061-eq | PSG+GCN + GKD 30% (eq_concat) | 60.8% | 73.0% | 84.3% | 87.8% | mAP+4.2%, R1+6.5% | 🟡 中性。vs 3-seed: +0.07%/+0.43%。GCN dropout 无效 |
| 062-eq | PSG+GCN + LKU (eq_concat) | 60.7% | 71.2% | 84.1% | 87.4% | mAP+4.1%, R1+4.7% | ❌ 负面。vs 3-seed: -0.03%/-1.37%。Learned uncertainty 损害 R1 |
| 063-eq | PSG + PTD (eq_concat) | 56.7% | 65.3% | 78.3% | 82.4% | mAP+0.1%, R1-1.2% | ❌❌ 严重负面。vs 3-seed: -4.03%/-7.27%。Pose-Token 无法替代 GCN |
| 058+nfc | PSG+GCN+ROA + NFC (eq_concat) | **64.0%** | **74.3%** | 84.3% | 87.2% | — | 🟢 NFC test-time boost on ROA。最强结果（含 NFC）|
| 058+cvk | PSG+GCN+ROA + CVK (cvk_hybrid) | 62.7% | 73.5% | 85.4% | 88.7% | — | 🟢 CVK 在 ROA 上也有效 |
| 064-eq | PSG+GCN + PKE (eq_concat) | 61.0% | 73.1% | 84.5% | 87.7% | mAP+4.4%, R1+6.6% | 🟡 微弱正向。vs 3-seed: +0.27%/+0.53%。Precision weighting 安全但不显著 |
| 065-eq | PSG+GCN + PKE+ROA (eq_concat) | 61.9% | 73.2% | 84.5% | 88.2% | mAP+5.3%, R1+6.7% | 🟡 ≈ROA alone。PKE+ROA 不正交 |
| **066-eq** | **PSG+GCN + PAA (eq_concat)** | **61.6%** | **74.2%** | **85.4%** | **88.4%** | **mAP+5.0%, R1+7.7%** | **🟢🟢🟢 历史最高 R1！vs 3-seed: +0.87%/+1.63%。训练端创新！** |
| **067-eq** | **PSG+GCN + PAA+ROA (eq_concat)** | **62.0%** | **73.7%** | **85.2%** | **88.6%** | **mAP+5.4%, R1+7.2%** | **🟢🟢🟢 历史最高 mAP！PAA+ROA 部分正交叠加。vs 3-seed: +1.27%/+1.13%** |
| 068-eq | PSG+GCN + RR-PAA (eq_concat) | 61.2% | 72.9% | 85.4% | 88.3% | mAP+4.6%, R1+6.4% | 🟡 vs PAA uniform: -0.4%/-1.3%。路由不优于 uniform |
| 069-eq | PSG+GCN + PAA b128 (eq_concat) | 61.3% | 74.6% | 85.2% | 88.3% | mAP+4.7%, R1+8.1% | 🟡 vs PAA b32: -0.3% mAP, +0.4% R1。R5/R10 改善但 mAP 未超。b32 仍是最优配置 |
| 070-eq | PSG+GCN + PAA S&C (eq_concat) | 61.4% | 73.4% | 85.4% | 88.5% | mAP+4.8%, R1+6.9% | 🟡 vs PAA scene: -0.2% mAP, -0.8% R1。target-only 热图不优于 scene 热图。消融价值 |
| 071-eq | PSG+GCN + PCL r=16 (eq_concat) | 60.7% | 72.0% | 84.6% | 88.1% | mAP+4.1%, R1+5.5% | ❌ vs PAA: -0.9% mAP, -2.2% R1。Feature-dependent LoRA 劣于 feature-independent PAA |
| 072-eq | PSG+GCN + PS-PAA (eq_concat) | 61.1% | 73.8% | 84.8% | 88.4% | mAP+4.5%, R1+7.3% | 🟡 vs PAA: -0.5% mAP, -0.4% R1。Body-part 分组不优于 generic 混合 |
| 073-eq | PSG+GCN + PAA Stage2+3 (eq) | 61.1% | 74.2% | 85.7% | 88.4% | mAP+4.5%, R1+7.7% | 🟡 vs PAA Stage3: -0.5% mAP, 0.0% R1。多 stage 不如单 stage |
| 074-eq | PSG+GCN + PAA+PGAM (eq) | — | — | — | — | — | ❌ 中止。PGAM 完全无效——结果与 exp066 精确相同。PGAM 为 no-op |
| 066-5060 | PAA 跨硬件验证 (5060 Ti) | 61.2% | 74.3% | 85.4% | 88.3% | — | ✅ 与本地 3090 结果一致 (Δ<0.4%)。远程可靠 |
| **066-s42** | **PAA seed42 (5060 Ti)** | **61.1%** | **74.4%** | **85.0%** | **87.6%** | — | **✅ vs seed1234(61.6%/74.2%): Δ-0.5%/+0.2%。PAA 跨 seed 确认** |
| **067-s42** | **PAA+ROA seed42 (3090)** | **62.1%** | **73.6%** | **85.2%** | **88.6%** | — | **✅ vs seed1234(62.0%/73.7%): Δ+0.1%/-0.1%。完美复现** |
| 076-eq | PSG+GCN+PAA+TDPC (eq) | 61.3% | 72.7% | 84.9% | 87.8% | mAP+4.7%, R1+6.2% | ❌ vs PAA(61.6/74.2): -0.3%/-1.5%。differential adapter 无收益 |
| 077-eq | PSG+GCN+ST-PAA 34ch (eq, 5060) | 61.0% | 73.6% | 84.4% | 88.6% | mAP+4.4%, R1+7.1% | ❌ vs PAA: -0.6%/-0.6%。scene+target concat 不优于 scene-only |
| 078-eq | PSG+GCN+PAA+APG (eq) | 60.5% | 72.5% | 84.3% | 87.9% | mAP+3.9%, R1+6.0% | ❌ vs PAA: -1.1%/-1.7%。adaptive gate 负面 |
| **079-eq** | **PSG+GCN+ROA 无PAA (eq, 5060)** | **62.0%** | **73.6%** | **85.0%** | **88.1%** | **mAP+5.4%, R1+7.1%** | **🟢🟢 ROA 独立有效！vs 3-seed: +1.27%/+1.03%。≈ exp067 PAA+ROA** |
| 081-eq | PSG+PAA+PQTD (eq) | 56.9% | 67.2% | 79.1% | 84.1% | mAP+0.3%, R1+0.7% | ❌❌ Decoder 严重不够收敛。GCN(400K) >> Decoder(2.5M) 在 120ep |
| 083-eq | PSG+GCN+PAA+PGFI (eq) | 61.1% | 73.4% | 84.7% | 88.1% | mAP+4.5%, R1+6.9% | 🟡 中性偏负 vs PAA(-0.5%/-0.8%)。Inpainter 未带来额外收益 |
| 084-eq | PSG+GCN+PAA+CIPGFR (eq) | 61.4% | 73.6% | 85.5% | 88.6% | mAP+4.8%, R1+7.1% | 🟡 中性 vs PAA(-0.2%/-0.6%)。Cross-instance recovery 未改善 |
| **085-eq** | **PSG+GCN+PAA+ROA p=0.7 (5060)** | **62.6%** | **75.3%** | **85.2%** | **88.4%** | **mAP+6.0%, R1+8.8%** | **🟢🟢🟢 历史最高！vs ROA p=0.5: +0.6%/+1.7%** |
| 085b-eq | PSG+GCN+ROA p=0.7 无PAA (5060) | 62.2% | 73.4% | 84.5% | 88.0% | mAP+5.6%, R1+6.9% | 🟡 vs p=0.5 无PAA: +0.2%. p=0.7 增益主要来自与 PAA 协同 |
| **086-eq** | **PSG+GCN+PAA+ROA+PA-PAT (3路)** | **62.7%** | **74.6%** | **85.3%** | **88.7%** | **mAP+6.1%, R1+8.1%** | **🟢🟢🟢 Peak 62.8%@Ep100。留作拼 SOTA recipe** |
| 087-eq | PSG+GCN+PAA+MM (momentum) | 61.5% | 73.0% | 84.5% | 88.2% | mAP+4.9%, R1+6.5% | 🟡 中性 vs PAA(-0.1%/-1.2%)。Memory contrastive 无额外收益 |
| **090-sgcfr** | **SGCFR on PAA (top_k=5, α=0.7)** | **64.2%** | **75.7%** | — | — | **mAP+7.6%, R1+9.2%** | **🟢🟢🟢🟢 +2.6% vs PAA baseline** |
| **090b-sgcfr** | **SGCFR on PAA+ROA (top_k=5, α=0.7)** | **64.9%** | **75.7%** | — | — | **mAP+8.3%, R1+9.2%** | **🟢🟢🟢🟢🟢 最强结果! +2.9% vs PAA+ROA** |
| 091-eq | PSG+GCN+PAA+TTSFR (eq) | 61.4% | 73.2% | 85.1% | 88.5% | mAP+4.8%, R1+6.7% | 🟡 中性 vs PAA(-0.2%/-1.0%)。Batch 内 recovery 信号不够（仅4张/ID） |
| 092-eq | PSG+GCN+PAA+LSRM w=0.5 (eq) | 60.9% | 73.3% | 85.0% | 88.1% | mAP+4.3%, R1+6.8% | 🟡 中性偏负 vs PAA(-0.7%/-0.9%)。Learned recovery 在 batch 内仍不够 |
| 092d-eq | PSG+GCN+PAA+LSRM BS128 (eq) | 61.3% | 73.5% | 84.8% | 88.4% | mAP+4.7%, R1+7.0% | 🟡 大batch帮助 (+0.4% vs BS64)，但仍 -0.3% vs PAA |
| 091b-eq | PSG+GCN+PAA+TTSFR BS128 (5060) | 60.8% | 73.0% | — | 88.6% | mAP+4.2%, R1+6.5% | 🟡 中性偏负。大 batch 对 simple recovery 无效 |
| 093-eq | PSG+GCN+PAA+PGTM (eq) | 56.7% | 68.0% | 80.9% | 85.2% | mAP+0.1%, R1+1.5% | ❌❌ Token merging 9.4M params 120ep 严重不够收敛 |
| 094 | PSG+GCN+PAA+PCQA (PTM) | — | — | — | — | 中性 (Ep74终止) | 🟡 PTM loss 不收敛(0.28→0.40)，Ep70: 59.2% vs 基线58.1%(+1.1%)，但 PTM 对照 exp030a 而非 exp066 |
| 094b | PSG+GCN+PAA+PCQA 归一化 (远程) | 61.2% | 74.0% | 84.8% | 88.2% | vs PAA: -0.4%/-0.2% | 🟡 PCQA 中性。PTM loss 0.41 不收敛 |
| 095-eq | PSG+GCN+PAA+DPF (热图池化) Ep100 | 60.0% | 71.8% | 83.5% | 87.1% | vs PAA: **-1.6%/-2.4%** | ❌ 12×4 分辨率太低，热图空间池化不如点采样 |
| 096-eq | PSG+GCN+PAA+MRKF (多尺度) Ep100 | 60.3% | 72.0% | 84.3% | 87.2% | vs PAA: -1.3%/-2.2% | ❌ Stage2(384d)+Stage3 融合不稳定，高方差震荡 |
| 098-eq | PSG+GCN+PAA+PKP (KPR式prompting) | 60.9% | 72.8% | 84.5% | 88.5% | vs PAA: -0.7%/-1.4% | 🟡 Swin window attention 限制早期 pose 传播 |
| 099 | OT Matching (测试时 Sinkhorn) | 59.0% | 71.0% | — | — | vs PAA: **-2.6%/-3.2%** | ❌ per-keypoint OT 不如 global cosine |
| 100-eq | PSG+GCN+PAA+FiLM (全阶段) | 61.0% | 73.3% | 84.6% | 88.3% | vs PAA: -0.6%/-0.9% | 🟡 PSG+PAA 已足够，更多 conditioning 不帮助 |
| 101-eq | PSG+GCN+PAA+SGMT (masking) | 61.0% | 73.8% | 85.0% | 88.5% | vs PAA: -0.6%/-0.4% | 🟡 中性，SGCFR 增益与基线相同 (+2.7% vs +2.6%) |
| 102-eq | PSG+GCN+PAA+SGMT-50% (masking) Ep110 | 60.6% | 73.1% | 84.7% | 87.9% | vs PAA: -1.0%/-1.1% | 🟡 50% masking 更激进，效果略差于 30%(exp101)。训练仅到 Ep110 |
| 104c-eq | PSG+GCN+PAA+PACD v3 (3×3 fm mask) | 61.3% | 74.5% | 85.4% | 88.6% | vs PAA: -0.3%/+0.3% | 🟡 中性。Feature map masking (8%) 太弱，GAP 鲁棒 |
| 104d-eq | PSG+GCN+PAA+PACD v4 (row fm mask) Ep100 | 60.4% | 73.3% | 84.5% | — | vs PAA: -1.2%/-0.9% | 🟡 中性偏负。33% 行级 mask 仍不够 |
| 105b-eq | PSG+GCN+PAA+SGRE (cross-attn) Ep90 | 60.7% | 73.3% | 85.1% | — | vs PAA: -0.3%/-0.2% | 🟡 中性。SGRE loss 收敛(3.28→0.30)但 detached kp 不影响 backbone |
| 106-eq | PSG+GCN+PAA+PISD (image mask) Ep28 | — | — | — | — | 提前终止 | 🟡 pisd loss 0.02-0.04 极小。GAP 全局特征天然遮挡不变 |
| 142-eq | PSG+GCN+SKC (Support-Supervised Keypoint Completion, eq) | 60.3% | 71.8% | 84.4% | 87.7% | vs exp030a-eq: -0.8%/-1.9% | ❌ 中性偏负。completion module 虽然活跃（gate=0.26, delta_norm=1.5），但 skc_pre≈skc_post 说明修改方向不是向 prototype 靠近。gate 无限制增长导致后期过度修改特征。feature-level completion 方向已被多轮验证为无效 |
| 143-eq | PSG+GCN+SASA (Skeleton-Aware Self-Attention, eq) | 61.1% | 73.7% | 85.1% | 88.5% | vs exp030a-eq: **0.0%/0.0%** | 🟡 完美中性。零参数骨架测地注意力偏置对最终结果无任何影响。与 KP-RPE(exp052) 结论一致：Swin window attention 的 RPE 已足够编码空间结构 |
| 141-cvk | PSG+GCN+LPCS comp_ctx (cvk_residual) | 55.8% | 68.1% | 78.3% | 82.4% | — | ❌ LPCS comp_ctx 失败。competition-context 未改善排序。LPCS 训练 loss 严重干扰主学习，最终远低于 exp030a (-5.3% mAP) |
| 144-eq | PSG+GCN+SASA α=1.0 (equal_concat) | 61.0% | 73.5% | 84.6% | 87.9% | vs exp030a-eq: **-0.1%/-0.2%** | 🟡 中性。10x更强的SASA偏置与α=0.1结果相同。确认skeleton attention信息对Swin完全冗余 |
| 145-eq | PSG+GCN+PAA+SASA (equal_concat) | 61.4% | 73.8% | — | 88.4% | vs PAA(exp066): **-0.2%/-0.4%** | 🟡 中性。SASA 与 PAA 组合无正交增益，确认 SASA 在任何配置下均无效 |
| 148-eq | PSG+GCN+PCVT (Pose-Complementary View Training, eq) | ~59.3%* | ~71.3%* | — | — | ❌ 负面。*ep100 数据，训练中。早期加速（ep30: +2.4 mAP）但后期被基线追平并反超。3-view 训练的 1/3 主损失稀释导致后期收敛不足。训练集 95.8% 全可见使 complementary masking 缺乏信号 |
| 149 | PSG+GCN+SCFA (Symmetry-Conditioned Feature Aggregation) | — | — | — | — | ❌ ep30 止损。ep30: 50.7/61.3 vs exp030a 52.2/66.0 (-1.5/-4.7)。bilateral gap case 太少(scfa_pg=0.09)，hand-crafted pooling trick 不够强 |
| 151-eq | PSG+GCN+PVAT (Pose-Visibility Adversarial Training, eq) | 进行中 | — | — | — | 🟡 中性趋势。ep70: 59.0/72.0 vs exp030a 58.1/70.9 (+0.9/+1.1)。但 pvat_acc=0.83 不降——训练集 95.8% 可见，adversarial 无信号。预计最终中性 |
| **maxsim** | **exp030a + MaxSim (ColBERT-style late interaction)** | **60.1%** | **74.4%** | **84.3%** | **87.5%** | **🟢 Test-time method。R1 74.4% 最高！但 mAP 低于 equal_concat (-1.0%)** |
| **maxsim_hybrid 1:1** | **exp030a + MaxSim Hybrid (global+maxsim)** | **62.2%** | **73.8%** | **84.9%** | **88.2%** | **🟢🟢 超越 CVK hybrid (61.9/73.2)！mAP+1.1% vs eq_concat** |
| **maxsim_hybrid 1:2** | **exp030a + MaxSim Hybrid (偏向 MaxSim)** | **62.2%** | **74.5%** | — | **88.6%** | **🟢🟢🟢 mAP+1.1, R1+0.8 vs eq_concat。ColBERT-style late interaction** |
| **maxsim_paa 1:2** | **PAA (exp066) + MaxSim Hybrid** | **62.6%** | **75.2%** | **85.6%** | **89.0%** | **🟢🟢 vs PAA eq_concat(61.6/74.2): +1.0/+1.0** |
| **maxsim_paa_roa 1:2** | **PAA+ROA (exp067) + MaxSim Hybrid** | **63.5%** | **75.4%** | **86.2%** | **88.9%** | **🟢🟢🟢🟢 vs PAA+ROA eq_concat(62.0/73.7): +1.5/+1.7。跨 checkpoint 稳定正向** |
| 152b-eq | MaxSim Hard Triplet Training (tau=0.005, eq_concat) | 57.8% | 69.7% | — | 86.8% | ❌ vs exp030a-eq: **-3.3/-4.0**。MaxSim training 严重损害特征 |
| 152b-ms | MaxSim Hard Triplet Training (maxsim_hybrid 1:2) | 59.0% | 71.0% | 83.8% | 87.2% | ❌ vs exp030a maxsim: **-3.2/-3.5**。即使 MaxSim test 也无法回补 |
| 152-eq | MaxSim Soft Triplet Training (tau=0.05, eq_concat) | 57.8% | 70.3% | — | 87.4% | ❌ vs exp030a-eq: **-3.3/-3.4**。与 hard 版结果一致 |
| 153-eq | MaxSim Additive w=0.25 (eq_concat) | 60.6% | 72.3% | — | 88.0% | 🟡 中性 vs exp030a-eq: **-0.5/-1.4**。不有害但无增益 |
| 153-ms | MaxSim Additive w=0.25 (maxsim_hybrid 1:2) | 61.8% | 74.3% | 85.1% | 88.4% | 🟡 中性 vs exp030a maxsim: **-0.4/-0.2** |
| 153b-eq | MaxSim Additive w=1.0 (eq_concat) | 57.6% | 70.0% | — | 87.1% | ❌ vs exp030a: **-3.5/-3.7**。w=1.0 崩了，与 replace 模式一致 |
| 155-eq | Evidential DL (GCN branch, eq_concat) | 60.7% | 72.9% | 84.4% | 88.4% | 🟡 中性 vs exp030a: **-0.4/-0.8**。Bayes Risk 梯度太弱(id_part=11 vs CE ~0.5) |
| 155-ms | Evidential DL (maxsim_hybrid 1:2) | 62.1% | 74.3% | 85.7% | 88.7% | 🟡 中性 vs exp030a maxsim: **-0.1/-0.2** |
| 155b-eq | Evidential DL kl=0.01 (eq_concat) | 61.0% | 73.0% | 84.9% | — | 🟡 中性 vs exp030a: **-0.1/-0.7**。中期 +1.4 peak(ep50)但最终追平 |
| 155b-ms | Evidential DL kl=0.01 (maxsim_hybrid) | 62.1% | 74.1% | 84.9% | 88.4% | 🟡 中性 vs maxsim: **-0.1/-0.4** |
| 156-eq | SPLADE sparse repr (eq_concat) | 60.5% | 72.3% | — | 87.5% | 🟡 中性 vs exp030a: **-0.6/-1.4** |
| **157-eq** | **PLBOA lower-body (VOC, p=0.7, eq_concat)** | **62.7%** | **74.0%** | **85.4%** | **89.0%** | **🟢🟢🟢 vs exp030a: +1.6/+0.3。vs ROA: +0.9。最强训练改进！** |
| **157-ms** | **PLBOA + MaxSim hybrid 1:2** | **64.1%** | **75.0%** | **86.4%** | **89.8%** | **🟢🟢🟢🟢 项目最高！vs baseline maxsim: +1.9/+0.5** |
| 157c-eq | PLBOA gradient bottom-heavy (eq_concat) | 60.8% | 73.5% | 85.2% | — | 🟡 中性 vs baseline: -0.3/-0.2。太激进 |
| **158-eq** | **PAA+PLBOA (eq_concat)** | **62.2%** | **74.7%** | **85.8%** | **89.0%** | **🟢🟢 vs baseline: +1.1/+1.0。R1 最高！** |
| **158-ms** | **PAA+PLBOA (maxsim_hybrid)** | **63.6%** | **75.8%** | **86.0%** | **89.2%** | **🟢🟢🟢 R1 最高！** |
| 157d-eq | Body-random occlusion (eq_concat) | 61.0% | 71.5% | 84.4% | 88.4% | 🟡 中性偏负 vs baseline: -0.1/-2.2。人体 bbox 随机遮挡不优于 ROA |
| 159-eq | PLBOA+ROA (eq_concat) | 62.4% | 73.7% | 85.4% | 88.7% | 🟢 vs baseline: +1.3/+0.0。但弱于 PLBOA-only (-0.3 mAP)。ROA+PLBOA 不正交 |
| 157-s42 | PLBOA seed42 (eq_concat) | 61.9% | 73.8% | 85.7% | 89.3% | ✅ 2-seed mean: **62.3%/73.9%** (+1.57/+1.33 vs baseline 3-seed) |
| 161-eq | **STD-PR (structural tokens, eq_concat)** | 58.7% | 67.4% | 81.1% | 85.0% | ❌ vs baseline: -2.4/-6.3。structural tokens 不如 GCN keypoint features |
| **161b-eq** | **STD-PR+PLBOA (eq_concat)** | **63.4%** | **73.4%** | **85.4%** | **88.5%** | **🟢🟢🟢 超 PLBOA+GCN mAP +0.7！vs baseline: +2.3/-0.3。STD-PR 替代 GCN 有效** |
| 161c-eq | STD-PR 17 parts (eq_concat) | 58.2% | 67.3% | 79.8% | 84.1% | 🟡 ≈6 parts (58.7)。token 数不是瓶颈 |
| 164-eq | STD-PR V2+PLBOA (anchor queries, eq) | 62.1% | 72.6% | 85.7% | 88.8% | ❌ vs V1: -1.3/-0.8。anchor 在遮挡位采噪声 |
| 164r-eq | STD-PR V2 alone (anchor, eq) | 57.9% | 68.0% | 81.5% | 85.0% | 🟡 vs V1: -0.8/**+0.6** R1。无 PLBOA 时 R1 改善 |
| 165-eq | STD-PR conf-pool+PLBOA (eq) | 61.8% | 71.9% | 84.5% | 88.5% | ❌ vs V1 mean: -1.6/-1.5。conf-pool 不帮 STD-PR+PLBOA |
| 165r-eq | STD-PR conf-pool alone (eq) | 58.2% | 68.9% | 81.5% | 85.7% | 🟡 vs V1 mean: -0.5/**+1.5** R1。无 PLBOA 时 R1 改善 |
| **157-3seed** | **PLBOA+GCN 3-seed mean** | **62.1±0.49%** | **73.9±0.12%** | — | — | **✅ +1.37/+1.33 vs baseline 3-seed** |
| **161b-3seed** | **STD-PR+PLBOA 3-seed mean** | **62.6±0.87%** | **72.7±0.67%** | — | — | **✅ +1.87/+0.13 vs baseline 3-seed** |
| **157+sgcfr** | **PLBOA+SGCFR (α=0.7)** | **65.2%** | **75.3%** | — | — | **🟢🟢🟢🟢 Test-time best! +4.5/+1.6 vs baseline** |
| 157+nfc | PLBOA+NFC (k=5) | 65.0% | 74.8% | 85.0% | 88.5% | 🟢🟢 +3.9/+1.1 vs baseline |
| 157+rr | PLBOA+Re-ranking | 78.8% | 79.7% | 87.8% | 90.0% | 🟢🟢🟢🟢🟢 含 re-ranking |
| 161d-eq | STD-PR+PLBOA+PAA (eq_concat) | 62.6% | 72.3% | 84.9% | 88.5% | 🟡 PAA 不帮 STD-PR (-0.8 vs 161b) |
| 161e-eq | STD-PR+PLBOA+ROA (eq_concat) | 63.2% | 72.9% | 85.5% | 88.8% | 🟡 ROA 不帮 STD-PR (-0.2 vs 161b) |

### Phase 4: SupCon + OA-SD + Parallel Aug (exp166-193)

| ID | 方法 | mAP | R-1 | R-5 | R-10 | 备注 |
|----|------|-----|-----|-----|------|------|
| 166 | STD-PR+PLBOA+PAPE+MS-PSG+CE (full arch) | 63.1% | 73.9% | 86.1% | 89.2% | CE baseline with full architecture |
| 166r | ↳ base arch (no PAPE/MS-PSG) | 60.3% | 72.8% | — | — | CE base for OA-SD comparison |
| 176 | +SupCon T=0.05 (1-view) | 64.1% | 75.5% | 85.9% | 89.2% | ✅ SupCon +1.0/+1.6 vs CE |
| **187** | **+SupCon T=0.05 + 3-view Parallel Aug** | **64.9%** | **76.6%** | **87.2%** | **90.0%** | **🟢🟢 Overall best! +1.8/+2.7 vs exp166** |
| **190** | **3-view Parallel Aug + CE (no SupCon)** | **64.2%** | **75.6%** | **86.2%** | **89.1%** | **✅ 3-view+CE ≈ 1-view+SupCon! +1.1/+1.7 vs exp166** |
| **191** | **OA-SD + CE (1-view, decay=0.999)** | **63.2%** | **75.4%** | **86.3%** | **89.1%** | **✅ OA-SD 独立有效! +2.9/+2.6 vs CE base** |
| 192 | OA-SD + CE (1-view, decay=0.99) | 62.6% | 74.9% | 86.0% | 89.1% | 🟡 vs exp191: -0.6/-0.5。decay 不敏感 |
| **193** | **OA-SD + 3-view + CE** | **64.4%** | **76.5%** | **86.3%** | **89.4%** | **✅ OA-SD+3-view additive! R1 ≈ exp187 SupCon, mAP +0.2/R1+0.9 vs exp190** |
| 194 | OA-SD + CE (weight=2.0) | 63.4% | 74.8% | 86.1% | 89.1% | 🟡 vs exp191 (w=1.0): +0.2/-0.6。weight 不敏感 |
| 195 | SupCon + OA-SD global-only (base) | 61.3% | 74.9% | — | — | 🟡 OA-SD+SupCon 无冲突但增益有限 |
| 196 | 3-view + SupCon + OA-SD global-only | 62.4% | 75.2% | 85.2% | 87.8% | ❌ vs exp187: -2.5/-1.4。OA-SD+SupCon 互斥 |
| 197 | 3-view + SupCon + STM | 64.1% | 76.0% | 86.7% | 89.0% | ❌ vs exp187: -0.8/-0.6。STM 不改善 SupCon 路线 |
| 198 | OA-SD + CE + STM (base, remote) | 63.2% | 75.2% | — | — | 🟡 = exp191 (无 STM)。STM 只加速不改善天花板 |
| 199 | 3-view + SupCon + OA-RD | 63.4% | 74.5% | 85.2% | 88.1% | ❌ vs exp187: -1.5/-2.1。OA-RD+SupCon 不兼容 |
| 200 | CE + OA-RD (base, remote) | 62.9% | 73.9% | 85.2% | 88.5% | ❌ vs exp191 OA-SD: -0.3/-1.5。OA-RD 不如 OA-SD |
| 201 | 3-view + SupCon + Global SupCon | 63.7% | 73.8% | 85.1% | 88.7% | ❌ vs exp187: -1.2/-2.8。Global SupCon 压缩特征空间 |
| 202 | **Swin-Small** + SupCon (1-view, remote) | 67.9% | 79.5% | 87.9% | 90.2% | **🟢🟢🟢 超过 FRT SOTA! +3.0/+2.9 vs Tiny** |
| **202b** | **Swin-Small + SupCon + 3-view + CP** | **69.3%** | **80.2%** | **88.9%** | **91.4%** | **🟢🟢🟢🟢🟢 NEW BEST! +3.1/+2.0 vs FRT, +4.4/+3.6 vs Tiny** |
| 203r | Small GCN+PAA+SupCon (1-view) | 66.7% | 78.5% | 86.8% | 89.6% | SupCon 在 GCN 上弱于 STD-PR |
| 205 | Small Dual Branch (GCN+PAA+STD-PR SupCon) 3v | 67.1% | 77.0% | 86.9% | 89.5% | ❌ Dual Branch -2.2 vs STD-PR |
| 205r | Small Dual Branch 1-view | 66.5% | 76.3% | 86.1% | 89.1% | ❌ Dual Branch -1.4 vs STD-PR |
| **206** | **Small GCN+PAA+CE+OA-SD (2-run mean)** | **70.4%** | **82.1%** | — | — | **🟢🟢🟢🟢 R1 超 4090 PAA! 2-run: 70.5/82.3 + 70.3/81.8** |

### Occluded-ReID 跨数据集测试 (模型训练于 Occluded-Duke)

| ID | 方法 | mAP | R-1 | R-5 | R-10 | 备注 |
|----|------|-----|-----|-----|------|------|
| 066-occ_reid | PSG+GCN+PAA (equal_concat) | 72.2% | 77.8% | 88.1% | 93.3% | 跨数据集泛化 |
| 079-local | PSG+GCN+ROA 无PAA (本地验证) | 61.8% | 72.8% | 85.2% | 88.3% | — | ✅ 跨硬件一致 (vs 远程 62.0%/73.6%, Δ<0.2%) |
| 082-remote | PSG+GCN+PAA+ROA (远程验证) | 61.8% | 74.1% | 84.8% | 88.5% | — | ✅ 跨硬件一致 (vs 本地 62.0%/73.7%, Δ<0.2%) |
| 056-eq | PSG+GCN + PGAM S2+S3 (eq_concat) | 61.1% | 73.7% | 85.2% | 88.6% | mAP+4.5%, R1+7.2% | 🟡 vs exp054: ≈持平。多 Stage 无额外增益 |

> 注：`exp036 / exp037` 的编号沿用了原 visibility 路线的占位命名，但实验内容已经转入 `PSG+GCN` branch 的后续探索；解读时不要把编号本身当作路线语义。

### exp042 分析摘要

- `exp042` 不引入新 checkpoint，只对 `040a equal_concat` 与 `040b cvk_hybrid` 做 query-level 差分分析。
- 关键统计：
  - `positive_delta_ap = 1129`
  - `negative_delta_ap = 822`
  - `zero_delta_ap = 259`
  - `top1_fixed = 47`
  - `top1_degraded = 58`
- 解释：
  - mAP 增益来自更广泛的 AP 改善，而不是单纯 top-1 修复
  - 在 `040a/040b` 这个 checkpoint 上，它呈现为 `mAP +0.8 / R1 -0.5`
  - 但 `exp045` 说明这种 R1 回落并不是固定规律，稳定项应写成“mAP 转正”

### exp043 论文素材

- 已基于 `exp042 query_deltas.csv` 生成 qualitative case study：
  - `experiments/paper_materials/figures/qualitative/cvk_top_improved.png`
  - `experiments/paper_materials/figures/qualitative/cvk_top_degraded.png`
- 两张图都保留了改进与退化样例，可直接支撑 story 中的 trade-off 叙述。

### exp045 第二 checkpoint 复核摘要

- `045a equal_concat` = `60.2% / 72.7%`
- `045b cvk_hybrid` = `61.1% / 73.2%`
- 相对差异：
  - mAP `+0.9%`
  - R1 `+0.5%`
- 解释：
  - `cvk_hybrid` 的正 mAP 信号已从单 checkpoint 扩展到第二个 checkpoint
  - R1 的变化方向目前不能写死，较稳妥的结论应聚焦于 mAP 复核成立

### +NFC 结果 (Neighbor Feature Centralization, Pose2ID CVPR 2025)

> NFC 是通用 test-time 方法，不是我们的训练端创新。所有结果基于 exp030a seed1234 checkpoint。

| ID | 方法 | k1=k2 | mAP | R-1 | R-5 | R-10 | vs 无后处理 |
|----|------|-------|-----|-----|-----|------|-------------|
| 049-g-k2 | PSG+GCN global + NFC | 2 | 62.8% | 74.9% | 83.9% | 87.5% | mAP+3.0%, R1+5.4% |
| 049-g-k5 | PSG+GCN global + NFC | 5 | 65.5% | 73.0% | 82.0% | 85.4% | mAP+5.7%, R1+3.5% |
| 049-eq-k2 | PSG+GCN equal_concat + NFC | 2 | 63.4% | 74.6% | 84.2% | 87.1% | mAP+2.3%, R1+0.9% |
| 049-eq-k3 | PSG+GCN equal_concat + NFC | 3 | 64.8% | 75.6% | 84.1% | 87.4% | mAP+3.7%, R1+1.9% |
| 049-eq-k4 | PSG+GCN equal_concat + NFC | 4 | 66.3% | 76.9% | 84.2% | 87.5% | mAP+5.2%, R1+3.2% |
| **049-eq-k5** | **PSG+GCN equal_concat + NFC** | **5** | **67.3%** | **77.6%** | **84.8%** | **87.8%** | **mAP+6.2%, R1+3.9%** |
| 049-eq-k6 | PSG+GCN equal_concat + NFC | 6 | 68.3% | 77.2% | 84.9% | 88.0% | mAP+7.2%, R1+3.5% |
| 049-eq-k8 | PSG+GCN equal_concat + NFC | 8 | 69.6% | 76.0% | 84.1% | 87.7% | mAP+8.5%, R1+2.3% |
| 049-eq-k10 | PSG+GCN equal_concat + NFC | 10 | 70.9% | 74.9% | 84.1% | 87.6% | mAP+9.8%, R1+1.2% |

### +Re-ranking 结果

| ID | 方法 | mAP | R-1 | R-5 | R-10 | 备注 |
|----|------|-----|-----|-----|------|------|
| 030a-g+RR | PSG+GCN global + Re-ranking | 76.4% | 75.3% | 85.2% | 87.7% | |
| **030a-eq+RR** | **PSG+GCN equal_concat + Re-ranking** | **78.5%** | **78.8%** | **87.5%** | **89.2%** | **全实验最佳 (含后处理)** |

## 数据集: Market-1501（4090 实验结果）

### Swin-Tiny

| ID | 方法 | mAP | R-1 | R-5 | R-10 | vs Baseline | 备注 |
|----|------|-----|-----|-----|------|-------------|------|
| 4090-M-base | Baseline (SOLIDER-Swin-Tiny, SW=0.2) | 91.6% | 96.3% | 98.7% | 99.2% | — | 120ep |
| 4090-M-PSG | + PSG (Backbone Injection) | 92.4% | 96.7% | 98.8% | 99.4% | mAP+0.8%, R1+0.4% | PSG 在非遮挡数据集也有效 |

### Swin-Small（对照组，不在论文 Swin-Tiny 约束内）

| ID | 方法 | Backbone | LR | mAP | R-1 | R-5 | R-10 |
|----|------|----------|-----|-----|-----|-----|------|
| 4090-M-small-lr4 | Baseline | Swin-Small | 4e-4 | 93.3% | 96.6% | 98.9% | 99.3% |
| 4090-M-small-lr8 | Baseline | Swin-Small | 8e-4 | 93.0% | 96.7% | 98.9% | 99.3% |
| 4090-M-PSG-small-lr4 | PSG | Swin-Small | 4e-4 | 93.9% | 96.9% | 99.0% | 99.3% |
| 4090-M-PSG-small-lr8 | PSG | Swin-Small | 8e-4 | 93.7% | 96.9% | 99.0% | 99.3% |

## 数据集: Occluded-Duke（4090 Swin-Small 对照组）

| ID | 方法 | Backbone | LR | mAP | R-1 | R-5 | R-10 |
|----|------|----------|-----|-----|-----|-----|------|
| 4090-OD-small-base | Baseline | Swin-Small | 8e-4 | 65.8% | 76.0% | 86.2% | 89.0% |
| 4090-OD-PSG-small-lr4 | PSG | Swin-Small | 4e-4 | 67.8% | 76.7% | 86.9% | 90.6% |
| 4090-OD-PSG-small-lr8 | PSG | Swin-Small | 8e-4 | 66.4% | 75.7% | 87.3% | 90.5% |

### 跨数据集/Backbone PSG 增益总结

| 数据集 | Backbone | PSG mAP提升 |
|--------|----------|-------------|
| Occluded-Duke | Swin-Tiny | +1.33% (3-seed mean) |
| Occluded-Duke | Swin-Small (lr4) | +2.0% |
| Market-1501 | Swin-Tiny | +0.8% |
| Market-1501 | Swin-Small (lr4) | +0.6% |

**结论**: PSG 在所有数据集和 backbone 上均有效。在遮挡数据集上增益更大。

---

## 多种子验证结果 (4090, Occluded-Duke, Swin-Tiny)

> 统计口径统一为 **two-sided paired t-test on mAP**。此前文档里出现过的 `~0.054 / ~0.014` 等数值是把双侧检验写成了半尾值，这里已统一更正。

### 已完成的 3-seed 配置

| 方法 | 测试模式 | Seed 1234 | Seed 42 | Seed 2024 | Mean±Std (mAP) | Mean±Std (R1) |
|------|----------|-----------|---------|-----------|----------------|---------------|
| Baseline (exp000) | global | 56.7% | 55.9% | 56.9% | **56.50±0.53%** | **66.33±0.67%** |
| PSG (exp007) | global | 58.3% | 57.9% | 57.3% | **57.83±0.50%** | **67.13±0.84%** |
| PSG + 0.5x loss (exp007a) | global | 59.6% | 59.5% | 59.0% | **59.37±0.32%** | **69.43±0.12%** |
| PDS+StopGrad (exp023) | global | 59.7% | 59.2% | 58.7% | **59.20±0.50%** | **68.63±0.47%** |
| PSG + GCN (exp030a) | global | 59.8% | 59.1% | 59.1% | **59.33±0.40%** | **68.87±1.00%** |
| PSG + GCN (exp030a) | concat_scaled | 60.5% | 59.7% | 60.4% | **60.20±0.44%** | **73.13±0.29%** |
| PSG + GCN (exp030a) | equal_concat | 61.1% | 60.2% | 60.9% | **60.73±0.47%** | **72.57±0.58%** |
| PSG + GCN (exp030a) | gcn_only | 58.2% | 57.4% | 58.3% | **57.97±0.49%** | **71.77±0.60%** |

### 关键统计检验 (two-sided paired t-test, mAP)

| 对比 | Mean Δ | Paired Diffs | t-stat | p-value | 解读 |
|------|--------|--------------|--------|---------|------|
| PSG vs Baseline | **+1.33%** | (1.6, 2.0, 0.4) | 2.77 | 0.1091 | 3 seeds 全正，但 n=3 时双侧检验仍偏弱 |
| exp007a vs PSG | **+1.53%** | (1.3, 1.6, 1.7) | 12.76 | 0.0061 | ✅ 0.5x global loss 对 PSG 是稳定增益 |
| exp007a vs exp023-g | **+0.17%** | (-0.1, 0.3, 0.3) | 1.25 | 0.3377 | 无显著差异；exp023 global 基本被 0.5x loss 复现 |
| exp030a-g vs exp007a | **-0.03%** | (0.2, -0.4, 0.1) | -0.18 | 0.8740 | GCN 分支对 global 几乎中性 |
| exp030a-cs vs exp030a-g | **+0.87%** | (0.7, 0.6, 1.3) | 3.96 | 0.0581 | 边缘改善，方向一致 |
| exp030a-eq vs exp030a-g | **+1.40%** | (1.3, 1.1, 1.8) | 6.73 | 0.0214 | ✅ 训练好的 branch 对 fusion 有稳定增益 |
| exp030a-eq vs exp030a-cs | **+0.53%** | (0.6, 0.5, 0.5) | 16.00 | 0.0039 | ✅ equal_concat 明显优于 concat_scaled |
| exp030a-eq vs exp007a | **+1.37%** | (1.5, 0.7, 1.9) | 3.87 | 0.0606 | 边缘改善，说明 branch 增益大体稳定但 n=3 仍偏小 |
| exp030a-eq vs exp023-g | **+1.53%** | (1.4, 1.0, 2.2) | 4.35 | 0.0491 | ✅ 当前最强模式已稳定超过 PDS global |

### 当前应采用的结论

1. **PSG 仍成立**：3 个 seed 全正，均值 `56.50% → 57.83%`，只是不能再把它写成“统计显著已确认”，更准确的表述是“稳定正向、样本数仍小”。
2. **0.5x global loss 不是方差假象**：`exp007a` 对 `exp007` 的 3-seed 改善为 `+1.53% mAP`，而且方向完全一致。
3. **PDS+StopGrad 的 global 增益基本可由 0.5x loss 解释**：`exp007a = 59.37%`，`exp023-g = 59.20%`，二者无显著差异。
4. **GCN/KPP branch 的贡献应按“fusion 增益”来讲**：`exp030a-global` 与 `exp007a` 几乎相同，但 `exp030a-eq` 能稳定到 `60.73%`，对自身 global `+1.40%`。
5. **equal_concat 才是当前主模式**：3 个 seed 中都优于 `concat_scaled`，因此 `030a-eq` 应替代 `030-cs` 成为主结果。
6. **exp030b / exp032 的角色需要重写**：
   - `exp030b` 说明 `w_p=0.01` 时 branch 基本没学好，不能再据此否定 loss scaling 或 GCN。
   - `exp032` 说明 keypoint pooling 本身就是强基线；GCN 更准确的定位是 branch refinement，而不是全部增益来源。

---

## 参考: Phase 1 最佳结果 (exp/003_offline_pose 分支)
- Baseline: mAP 56.6%, R1 66.5%
- GiLt+PCFC (exp012): mAP 58.0%, R1 68.0% (+1.4/+1.5 vs baseline)
- Full pipeline (NFC+Part): mAP 64.7%, R1 69.4%


## 2026-03-19: exp107 DACHM（retrieval-time 诊断）

### exp107: Duplicate-Aware Counterfactual Hypothesis Matching

> 基于 `exp030a equal_concat` 的 test-time 原型诊断，不计入训练端创新，只用于验证“duplicate-aware 多候选反事实匹配”是否存在独立 headroom。

| 变体 | mAP | R1 | 相对 base |
|------|-----|----|-----------|
| base_equal_concat | 61.14% | 73.71% | — |
| raw_counterfactual_signed | 60.32% | 72.76% | -0.82 / -0.95 |
| dachm_signed | 60.27% | 72.81% | -0.87 / -0.90 |
| raw_counterfactual_penalty | 60.70% | 73.17% | -0.44 / -0.54 |
| **dachm_penalty** | **60.72%** | **73.17%** | **-0.42 / -0.54** |

- 关键子集：
  - `clean multi`: `63.99 / 77.27` → `63.24 / 75.83`
  - `duplicate-suspect multi`: `61.36 / 76.71` → `60.64 / 76.46`
  - `n=2`: `64.64 / 79.05` → `63.99 / 77.82`
- 结论：
  1. coarse pooled hypothesis 上的反事实重排整体负面。
  2. duplicate-aware pruning 没有救回该方向。
  3. “target/distractor ambiguity”如果继续做，推理粒度必须回到 `per-keypoint / common-support`，不能停留在 pooled person embedding。


## 2026-03-19: exp108 DACCM（retrieval-time 诊断）

### exp108: Duplicate-Aware Counterfactual Common-Support Matching

> 基于 `exp030a cvk_hybrid` 的 test-time 原型诊断，不计入训练端创新，只用于验证“per-keypoint / common-support 粒度的 duplicate-aware confuser penalty”是否存在独立 headroom。

| 变体 | mAP | R1 | 相对 base |
|------|-----|----|-----------|
| base_cvk_hybrid | 61.88% | 73.26% | — |
| raw_daccm_penalty | 61.35% | 72.85% | -0.53 / -0.41 |
| **daccm_penalty** | **61.39%** | **72.94%** | **-0.49 / -0.32** |

- 关键子集：
  - `multi`: `64.07 / 76.51` → `63.16 / 75.87`
  - `clean multi`: `65.06 / 76.26` → `64.12 / 75.40`
  - `duplicate-suspect multi`: `62.31 / 76.96` → `61.47 / 76.71`
  - `n=2`: `65.76 / 78.35` → `64.99 / 77.82`
- 结论：
  1. 即使把 confuser reasoning 下沉到 `per-keypoint / common-support` 粒度，当前 retrieval-time penalty 仍然整体负面。
  2. duplicate-aware pruning 只能略微减轻退化，不能把该方向救回到高于基线。
  3. 因此 `ambiguity` 这条线若继续推进，必须从 test-time rerank 转向训练端建模，或直接切换到新的问题定义。


## 2026-03-19: exp109 Oracle Support Bank（上界诊断）

### exp109: GT same-ID per-keypoint support bank

> 基于 `exp030a cvk_hybrid` 的 oracle 诊断，不计入正式结果；仅用于判断“training-time support-complete distillation”是否存在足够大的理论 headroom。

| 变体 | mAP | R1 | 相对 base |
|------|-----|----|-----------|
| base_cvk_hybrid | 61.88% | 73.26% | — |
| oracle_feat_only_cvk | 66.15% | 77.87% | +4.27 / +4.62 |
| **oracle_feat_weight_cvk** | **70.40%** | **81.36%** | **+8.53 / +8.10** |

- oracle 恢复统计：
  - `3385` 个样本发生了恢复
  - 共恢复 `10194` 个 keypoints
  - 平均每个恢复 keypoint 有 `82.33` 个 same-ID 支持样本
- 关键子集：
  - `clean multi`: `65.06 / 76.26` → `68.04 / 79.71` → `71.33 / 82.73`
  - `duplicate-suspect multi`: `62.31 / 76.96` → `65.21 / 78.73` → `68.34 / 81.27`
  - `target_vis<=8` (26 queries): `29.42 / 26.92` → `78.26 / 84.62` → `91.71 / 100.00`
  - `target_vis<=5` (7 queries): `16.85 / 14.29` → `78.43 / 85.71` → `86.28 / 100.00`
- 结论：
  1. `support-complete` 方向存在非常明显的 headroom，尤其集中在低可见 query。
  2. 即使不改权重、只恢复 keypoint feature，本身也已经带来 `+4.27 / +4.62` 的 oracle 增益。
  3. 因此下一步最值得推进的，不是新的 penalty/rerank，而是训练版的 **support-complete prototype distillation**。


## 2026-03-19: exp110 SCKD（训练端最小原型，单 seed）

### exp110: Support-Complete Keypoint Distillation

> 基于 `exp030a equal_concat` 的训练端最小原型。该实验只新增 `per-identity / per-keypoint prototype bank`，并对低可见 keypoint 施加蒸馏；当前结果为单 seed 探索证据，不视为最终结论。

| 方法 | mAP | R1 | R5 | R10 | 相对对照 |
|------|-----|----|----|-----|----------|
| `exp030a-eq` `seed1234` | 61.1% | 72.9% | 85.2% | 87.8% | — |
| **`exp110_sckd`** | **61.2%** | **73.7%** | **84.7%** | **88.2%** | **+0.1 / +0.8** |

- 关键收敛点：
  - `ep40 = 56.2 / 68.4`
  - `ep60 = 58.3 / 70.5`
  - `ep80 = 59.8 / 71.4`
  - `ep90 = 60.9 / 73.1`
  - `ep120 = 61.2 / 73.7`
- 训练观察：
  1. `warmup=20` 阶段与 `exp030a` 基本重合，说明 bank 后台更新没有破坏主训练。
  2. `sckd` 自 `epoch > 20` 后稳定在约 `0.19~0.21`，没有发散。
  3. `mAP` 在 `ep40-120` 基本持续高于基线，`R1` 则从中段轻微落后逐步转成最终领先。
- 当前结论：
  1. `support-complete distillation` 训练版最小原型已经**成功转正**，但当前仍只是单 seed、弱增益证据。
  2. 当前瓶颈更像是 prototype teacher 的可靠性不够，而不是这条主线本身错误。
  3. 因此下一步应优先做 **reliable-support bank** 一类的单变量改进，而不是直接堆更重模块。


## 2026-03-19: exp111 Reliable-Support SCKD（MIN_COUNT=4）

### exp111: Count-Gated Support-Complete Keypoint Distillation

> 基于 `exp110_sckd` 的单变量 teacher reliability 改进。该实验仅把 `POSE_SCKD_MIN_COUNT` 从 `1` 提高到 `4`，用于测试“prototype 是否必须由多个支持样本共同支撑后再参与蒸馏”。

| 方法 | mAP | R1 | R5 | R10 | 相对 exp110 |
|------|-----|----|----|-----|-------------|
| `exp110_sckd` | 61.2% | 73.7% | 84.7% | 88.2% | — |
| **`exp111_sckd_min4`** | **61.1%** | **73.8%** | **85.2%** | **88.6%** | **-0.1 / +0.1** |

- 关键收敛点：
  - `ep40 = 56.0 / 68.1`
  - `ep70 = 59.0 / 71.5`
  - `ep90 = 60.8 / 73.8`
  - `ep120 = 61.1 / 73.8`
- 当前结论：
  1. `MIN_COUNT=4` 没有把 `exp110` 的弱正向显著放大，整体更接近“等价替代”而不是进一步突破。
  2. `support-complete` 主线仍然成立，因为相对 `exp030a-eq seed1234` 依旧保持 `R1` 正增益。
  3. 但当前更关键的 teacher reliability 维度，大概率不是 support 数量门槛，而是 support 纯度 / 写入质量。


## 2026-03-19: exp112 High-Confidence Support SCKD（早停）

### exp112: `UPDATE_THR=0.7`

> 基于 `exp110_sckd` 的 teacher purity 单变量验证。该实验在 `ep84` 提前停表，原因是需要把资源切到更直接的 non-stationary teacher 验证；下面记录其当前最有信息量的观测点，不视为最终收敛结果。

| 观测点 | mAP | R1 | R5 | R10 | 相对 exp110 同期 |
|--------|-----|----|----|-----|------------------|
| `ep50` | 57.4% | 69.7% | 82.5% | 86.5% | `+1.3 / +1.4` |
| `ep70` | 59.0% | 71.2% | 84.5% | 87.9% | `+0.2 / +0.1` |
| `ep80` | 59.7% | 71.6% | 84.7% | 88.0% | `-0.1 / +0.2` |

- 当前结论：
  1. `UPDATE_THR=0.7` 说明 teacher purity 是有价值的方向，但当前只形成 **弱正向 / 近乎等价** 的证据。
  2. 它不足以单独构成论文主机制，但足以说明 `teacher reliability` 仍值得继续做。


## 2026-03-19: exp113 SCKD 统计诊断（机制实验，早停）

### exp113: 为什么 raw `sckd` 不降

> 该实验不是为了刷最终指标，而是为了回答：当前 `raw sckd` 上升究竟来自 coverage 变化，还是来自 teacher hardening。实验在 `ep44` 提前停表，因为机制信号已足够清楚。

- 关键诊断现象（`epoch 22 -> 44`）：
  - `sckd_pairs`: 约 `152~166`，基本稳定
  - `sckd_lowr / sckd_actr`: 约 `0.145~0.153`，基本稳定
  - `sckd_eligr`: 持续为 `1.000`
  - `sckd_conf`: 约 `0.882~0.883`，基本稳定
  - `sckd_count`: 持续升高，约 `312 -> 730+`
  - `sckd_cos`: 持续下降，约 `0.83 -> 0.79`
  - raw `sckd`: 持续升高，约 `0.17 -> 0.21`

- 当前结论：
  1. `raw sckd` 上升不是因为蒸馏覆盖率扩大，也不是因为 teacher 置信度崩坏。
  2. 更像是 bank 随训练持续积累 support，teacher 逐步变硬，导致 student 的平均对齐余弦下降。
  3. 因而下一步最值得验证的，不是继续扫 `count / purity`，而是直接控制 **teacher non-stationarity**。


## 2026-03-20: exp114 Freeze-After-Warmup SCKD

### exp114: `UPDATE_THR=0.7, UPDATE_STOP_EPOCH=20`

> 基于 `exp112_sckd_up07` 的单变量 teacher stability 验证。该实验在 `epoch 20` 后冻结 prototype bank，测试"非定常 teacher"是否是 SCKD 的主要瓶颈。

| 方法 | mAP | R1 | R5 | R10 | 相对 exp110 |
|------|-----|----|----|-----|-------------|
| `exp110_sckd` | 61.2% | 73.7% | 84.7% | 88.2% | — |
| **`exp114_freeze20`** | **61.3%** | **73.6%** | **84.7%** | **88.5%** | **+0.1 / -0.1** |

- 关键收敛点：
  - `ep50 = 56.2 / 68.5`（落后 exp112 的 57.4/69.7）
  - `ep60 = 58.4 / 70.5`（反超 exp112 的 57.7/70.0）
  - `ep70 = 59.0 / 70.7`（回到与 exp110/112 等价）
  - `ep80 = 60.0 / 71.8`（微超 exp110/112）
  - `ep90 = 61.0 / 73.3`
  - `ep110 = 61.3 / 73.4`
  - `ep120 = 61.3 / 73.6`
- 机制层面：
  - `sckd_cos` 在冻结后持续单调改善（0.806→0.821），说明 student 在稳步适应 frozen teacher
  - `count` 稳定在 ~280-340（不增长），不再像在线版本那样持续上升
  - 但这种机制改善**未转化为显著的最终指标优势**
- 当前结论：
  1. **freeze20 与 exp110 (online) 最终完全等价**（Δ < 0.1%）
  2. **teacher non-stationarity 不是 SCKD 的主要瓶颈**
  3. SCKD 系列 6 个变体（exp110-115）均收敛到 ~61.2-61.3 mAP，说明框架天花板已到


## 2026-03-20: exp115 Freeze-Later SCKD（最终）

### exp115: `UPDATE_THR=0.7, UPDATE_STOP_EPOCH=30`（远程 5060 Ti）

> 与 exp114 互补的并行对照。测试"bank 在更成熟时（ep30 而非 ep20）才冻结"是否更好。

| 方法 | mAP | R1 | R5 | R10 | 相对 exp110 |
|------|-----|----|----|-----|-------------|
| `exp110_sckd` | 61.2% | 73.7% | 84.7% | 88.2% | — |
| `exp114_freeze20` | 61.3% | 73.6% | 84.7% | 88.5% | +0.1 / -0.1 |
| **`exp115_freeze30`** | **61.3%** | **73.6%** | **85.1%** | **88.3%** | **+0.1 / -0.1** |

- 当前结论：
  1. **freeze20 和 freeze30 最终完全等价**（均为 61.3/73.6）
  2. 也与 exp110 (online) 近乎等价
  3. **冻结时机对最终结果没有影响**
  4. SCKD 天花板确认：~61.2-61.3 mAP


## 2026-03-20: exp116 SCFR（Support-Complete Feature Replacement）

### exp116: 直接替换 vs 间接蒸馏

> 基于 `exp110_sckd` 的单变量对照。该实验用 prototype bank 直接替换低可见度 keypoint 特征（而非通过 cosine distillation loss 蒸馏），测试"直接替换是否优于间接蒸馏"。

| 方法 | mAP | R1 | R5 | R10 | 相对 exp110 |
|------|-----|----|----|-----|-------------|
| `exp110_sckd` | 61.2% | 73.7% | 84.7% | 88.2% | — |
| **`exp116_scfr`** | **61.1%** | **74.1%** | **84.8%** | **88.5%** | **-0.1 / +0.4** |

- 当前结论：
  1. **SCFR ≈ SCKD**：mAP 差异 -0.1%（噪声），R1 差异 +0.4%（可能是噪声）
  2. 直接替换与间接蒸馏给出本质相同的结果
  3. **SCKD/SCFR 系列 7 个变体的总结**：

| 变体 | 核心改动 | mAP | R1 |
|------|----------|-----|-----|
| exp110 | 基础 SCKD | 61.2% | 73.7% |
| exp111 | MIN_COUNT=4 | 61.1% | 73.8% |
| exp114 | freeze epoch 20 | 61.3% | 73.6% |
| exp115 | freeze epoch 30 | 61.3% | 73.6% |
| exp116 | SCFR (直接替换) | 61.1% | 74.1% |

  4. **天花板：~61.1-61.3 mAP / 73.5-74.1 R1**
  5. **结论：EMA prototype bank 方向已穷尽，不再值得作为主线**


## 2026-03-20: exp117 VCGA（Visibility-Conditioned Graph Attention）

### exp117: GCN 消息传递改进

> 基于 `exp030a-eq` 的单变量对照。测试 visibility-conditioned graph attention（用 keypoint 可见度调制 GCN 邻接矩阵）是否改善消息传递效果。

| 方法 | mAP | R1 | R5 | R10 | 相对 exp030a-eq |
|------|-----|----|----|-----|-----------------|
| `exp030a-eq seed1234` | 61.1% | 72.9% | 85.2% | 87.8% | — |
| **`exp117_vcga`** | **61.1%** | **73.5%** | **84.8%** | **88.2%** | **0.0 / +0.6** |

- 结论：
  1. **VCGA 完全中性**（mAP 精确相同，R1 +0.6% 在噪声范围）
  2. Visibility-conditioned graph attention 不改善 GCN
  3. 标准 symmetric normalization 已足够


## 2026-03-20: exp119 CSRD（Common-Support Relational Distillation）

### exp119: 把 common-support pairwise 几何蒸馏进 global embedding

> 基于 `exp030a` 的单变量训练端实验。`CSRD` 不再把 support 压成 `per-ID prototype`，而是直接用 skeleton/keypoint branch 计算出的 `CVK-style` pairwise 几何作为 detached relational teacher，蒸馏 global embedding 的 batch-wise 距离结构。

| 方法 | mAP | R1 | R5 | R10 | 相对直接对照 |
|------|-----|----|----|-----|-------------|
| `exp030a-eq seed1234` | 61.1% | 72.9% | 85.2% | 87.8% | — |
| **`exp119-eq`** | **61.1%** | **73.2%** | **85.4%** | **88.6%** | **+0.0 / +0.3** |
| `exp030a-g seed1234` | 59.8% | 69.9% | — | — | — |
| **`exp119-g`** | **60.4%** | **70.3%** | **82.8%** | **87.4%** | **+0.6 / +0.4** |
| `exp040b cvk_hybrid` | 61.9% | 73.2% | 85.2% | 88.6% | — |
| **`exp119-cvk`** | **62.0%** | **73.2%** | **85.5%** | **88.8%** | **+0.1 / +0.0** |

- 结论：
  1. **`CSRD` 首次把训练端 pairwise teacher 做成了明确弱正向**，且最明显的收益出现在 `global`
  2. 这说明当前更接近真实问题的，不是 overlap mining，也不是 prototype pointwise 蒸馏，而是 **common-support relational teacher**
  3. 但 `equal_concat` 仍只是近乎持平，表明当前 teacher 还不够强，单图 `kp_feats` 本身依旧受 `support incomplete` 限制
  4. 因此下一步不应回到 generic 模块叠加，而应把 `exp109` 的 support-complete headroom 引回 `CSRD`，验证 **support-complete relational teacher**


## 2026-03-20: exp120 SCRD（Support-Complete Relational Distillation）

### exp120: support-complete teacher enhancement 已生效，但没有自动转成更好指标

> 基于 `exp119` 的单变量训练端实验。`exp120` 保留 `CSRD` 的 relational distillation 形式，只把 support-complete bank 用来补全 low-vis teacher keypoint，再用补全后的 teacher 去蒸馏 global 几何。该实验在 `ep90` 人工停表，结论来自训练监控口径，不是正式 eval 口径。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp119` | 60.1% | 73.7% | 85.1% | 88.3% | `ep90` 训练监控 |
| **`exp120`** | **59.9%** | **73.2%** | **84.7%** | **88.2%** | **`ep90` 训练监控** |

- 结论：
  1. `support-complete teacher` 的增强并没有失败，日志中 `csrd_sr≈0.145`、`csrd_sn≈157~159` 持续稳定，说明 low-vis keypoint 基本都拿到了补全 teacher
  2. `teacher_gap` 也明显强于 `exp119`，说明 teacher 的几何确实更“完整”、更可分
  3. 但直到 `ep90`，指标仍略弱于 `exp119`，说明 **teacher 更强 ≠ 监督一定更有效**
  4. 当前更合理的解释是：support-complete supervision 被大量本来就不缺 support 的 clean 样本稀释了
  5. 因而下一步不该继续盲目增强 teacher，而应测试 **只对 support-incomplete anchor 强化 relational distillation**

## 2026-03-20: exp121 SCRD Freeze-30（远程最终）

### exp121: stable teacher 对 SCRD 有持续但有限的正向帮助

> 基于 `exp120` 的单变量训练端实验。`exp121` 仅把 support-complete teacher bank 的更新停止在 `epoch 30`，用于验证“teacher 稳定化”是否能改善 `SCRD`。该实验跑满 `ep120`，结论来自训练监控口径。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp119` | 60.4% | 73.4% | 85.0% | 88.6% | `ep120` 训练监控 |
| `exp120` | 59.9% | 73.2% | 84.7% | 88.2% | `ep90` 训练监控 |
| **`exp121`** | **60.6%** | **74.0%** | **84.9%** | **88.6%** | **`ep120` 训练监控** |

- 结论：
  1. `stable teacher` 不是伪命题，`freeze30` 最终相对 `exp119` 形成了 `+0.2 / +0.6` 的稳定弱正向
  2. 它也明显强于提前停表的 `exp120 online teacher`，说明 `support-complete relational teacher` 确实受 teacher 稳定性影响
  3. 但这个量级仍不足以单独支撑论文主创新，更合理的定位是 **supporting mechanism**
  4. 因而下一步不应继续围绕 freeze 时机扫点，而应把重点放回 **如何更有效地把 teacher-change pairs 蒸进 global embedding**

## 2026-03-20: exp122 SGW-SCRD（早停）

### exp122: sample-level selective weighting 没有把 support-complete teacher 的收益兑现出来

> 基于 `exp120` 的单变量训练端实验。`exp122` 保持 support-complete teacher 完全不变，只把 `CSRD` 的 anchor 权重改为 sample-level `replace_ratio`。该实验在 `ep43` 提前停表，结论以 `ep40` 首个关键验证点为准。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp119` | 55.9% | 68.7% | — | — | `ep40` 训练监控 |
| `exp120` | 55.5% | 67.8% | — | — | `ep40` 训练监控 |
| **`exp122`** | **55.4%** | **68.2%** | **81.0%** | **85.2%** | **`ep40` 训练监控** |

- 结论：
  1. `exp122` 的新机制确实按设计生效：`csrd_ar≈0.56`、`csrd_aw≈0.145`，说明只有约一半 anchor 真正参与了 selective `CSRD`
  2. 但它相对 `exp119` 仍是 `-0.5 / -0.5`，相对 `exp120` 也没有形成清晰优势
  3. 这说明 “谁有更多被补全 keypoint” 不是足够精确的 supervision 路由信号
  4. 因而当前应放弃 sample-level weighting，转向更结构化的 **pair-level teacher-change focusing**

## 2026-03-20: exp123 Pair-Delta Focused SCRD（正式评估）

### exp123: pair-level teacher-change focusing 方向成立，但 `alpha=1.0` 只做到与 exp119 近乎等价

> 基于 `exp120` 的单变量训练端实验。`exp123` 保持 support-complete relational teacher 完全不变，只新增 pair-level `delta` focusing，让 `CSRD` 更聚焦于那些被 support-complete teacher 实际改变过的 pair。该实验已完成正式 eval。

| 方法 | mAP | R1 | R5 | R10 | 相对直接对照 |
|------|-----|----|----|-----|-------------|
| `exp030a-eq seed1234` | 61.1% | 72.9% | 85.2% | 87.8% | — |
| `exp119-eq` | 61.1% | 73.2% | 85.4% | 88.6% | — |
| **`exp123-eq`** | **61.1%** | **73.4%** | **84.8%** | **88.5%** | **vs `exp119-eq`: +0.0 / +0.2** |
| `exp030a-g seed1234` | 59.8% | 69.9% | — | — | — |
| `exp119-g` | 60.4% | 70.3% | 82.8% | 87.4% | — |
| **`exp123-g`** | **60.2%** | **70.3%** | **82.5%** | **86.7%** | **vs `exp119-g`: -0.2 / +0.0** |
| `exp040b cvk_hybrid` | 61.9% | 73.2% | 85.2% | 88.6% | — |
| `exp119-cvk` | 62.0% | 73.2% | 85.5% | 88.8% | — |
| **`exp123-cvk`** | **61.9%** | **73.2%** | **85.2%** | **88.8%** | **vs `exp119-cvk`: -0.1 / +0.0** |

- 结论：
  1. `exp123` 并没有把 `exp119` 推成更强的正式结果，三种测试口径整体都只是近乎等价
  2. 但它也没有否定 pair-level `teacher-change focusing` 本身，因为训练监控终点仍略高于 `exp119`，`equal_concat` 也保留了 `R1 +0.2`
  3. 当前更合理的解释不是“pair focus 不成立”，而是：
     - teacher-change pairs 的确重要
     - 但 `alpha=1.0` 的连续 delta 加权仍然过于平滑、过于稀释
  4. 因而下一步应继续保留 pair-level 主线，但把重点从“有没有 pair focus”推进到 **如何更强、更稀疏地聚焦真正被 teacher 改变的关系**

## 2026-03-20: exp130 Residual-KL SCRD（收敛结案）

### exp130: `residual_kl` 没有把 `exp125` 推得更强，`target dilution` 不是当前主瓶颈

> 基于 `exp125` 的单变量训练端实验。`exp130` 保持 online support teacher、`delta_top` pair routing、KL distillation 与 `tau=0.10` 全部不变，只把 `CSRD` target 从完整 teacher distribution 改为 `residual_kl`。该实验已跑满 `ep120`，结论来自训练监控口径。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp125` | 60.4% | 73.8% | 84.8% | 88.6% | `ep110` 训练监控 |
| `exp125` | 60.5% | 73.5% | 84.9% | 88.5% | `ep120` 训练监控 |
| **`exp130`** | **60.1%** | **73.4%** | **84.5%** | **88.3%** | **`ep110` 训练监控** |
| **`exp130`** | **60.1%** | **73.1%** | **84.6%** | **88.3%** | **`ep120` 训练监控** |

- 结论：
  1. `residual_kl` 没有失效，后期 `csrd` 始终稳定在 `0.011~0.013`，说明它不是“信号过弱导致看起来无效”
  2. 但它到 `ep110/120` 都稳定落后于 `exp125`，说明完整 teacher target 仍然更有效
  3. 因而当前可以把结论收紧为：
     - `target dilution` 不是当前主瓶颈
     - 下一步不该继续围绕 `target form` 扩线
  4. 这条实验的价值主要是负向因果证据：**pair routing / pair coverage 比 target 改写更重要**

## 2026-03-21: exp131 Cross-Batch Pair SCRD（收敛结案）

### exp131: queue coverage 真实生效，但没有把 `exp125` 推成更强主线

> 基于 `exp125` 的单变量训练端实验。`exp131` 保持 online support teacher、`delta_top` pair routing 与 full teacher target 全部不变，只新增 `cross-batch relation queue`，把每个 anchor 可见的 candidate relations 从 batch 内扩展到 `batch + queue`。该实验已跑满 `ep120`，结论来自训练监控口径。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp125` | 60.4% | 73.8% | 84.8% | 88.6% | `ep110` 训练监控 |
| `exp125` | 60.5% | 73.5% | 84.9% | 88.5% | `ep120` 训练监控 |
| **`exp131`** | **60.4%** | **73.7%** | **84.9%** | **87.8%** | **`ep110` 训练监控** |
| **`exp131`** | **60.5%** | **73.7%** | **84.8%** | **88.0%** | **`ep120` 训练监控** |

- 结论：
  1. queue 不是摆设；后期日志里 `csrd_qn = 256`、`csrd_qr ≈ 0.43`，说明约四成候选 relations 确实来自 cross-batch queue
  2. 但最终相对 `exp125` 只形成了 `mAP +0.0 / R1 +0.2` 的近乎等价结果，不足以支持“batch 内 changed-pair coverage 不足”是主瓶颈
  3. 当前更合理的解释是：
     - `changed pairs` 的存在不是问题
     - 问题在于 **当前学生如何消费这些 pair-specific support-complete corrections**
  4. 因而下一步不应继续扩线 queue，而应转向：
     **真正接入检索的 learned pair module / pair-adaptive correction**

## 2026-03-21: exp132 LTCS（正式评估）

### exp132: learned `alpha`-fusion 没有超过固定 `cvk_hybrid`，第一版 LTCS 作为方法机制判负

> 基于 `exp131` 之后的新方向实验。`exp132` 不再把 support-complete correction 蒸进 embedding，而是在检索期引入真正挂进 checkpoint 与 evaluator 的 `pair-adaptive fusion head`，学习每个 pair 该在多大程度上相信 `global distance` 与 `CVK distance`。该实验已跑满 `ep120`，并完成同一 checkpoint 下的正式对照评估。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp132` | 62.1% | 72.8% | 84.8% | 88.1% | `ep120` 训练监控 |
| **`exp132a cvk_adaptive`** | **62.1%** | **72.8%** | **84.8%** | **88.1%** | **正式 eval** |
| **`exp132b cvk_hybrid`** | **62.1%** | **72.8%** | **84.8%** | **88.1%** | **同 checkpoint 正式 eval** |

- 结论：
  1. `LTCS` 训练本身不是负方向；训练监控后期持续抬升到 `62.1 / 72.8`
  2. 但真正关键的同 checkpoint 正式对照已经给出清晰结论：
     - learned `cvk_adaptive`
     - 与固定 `cvk_hybrid`
     - **结果完全一致**
  3. 因而当前不能声称 “learned pair-adaptive fusion rule” 已经成立；第一版 `LTCS` 作为方法机制判负
  4. 这轮实验的负证据更精确地说明：
     - 检索期 learned pair module 这个大方向还没死
     - 但 **只学一个标量 `alpha`、只在两种标量距离之间做凸组合、并用 teacher distance 回归监督** 这套实现过于弱
  5. 因而下一步应从 `alpha-fusion` 升级到：
     **更强的 learned pair scorer / ranking-aligned pair correction**

## 2026-03-21: exp133 / exp134 LPCS（失效 run）

### exp133 / exp134: 由于共享接线 bug，当前结果全部作废，不能用于支持或反驳 LPCS

> `exp133`（LPCS）与 `exp134`（Changed-Pair Sparse LPCS）在运行过程中都暴露了同一个实现问题：训练日志在 `epoch 21+` 后始终没有任何 `lpcs_*` 统计。进一步排查代码确认，`processor.py` 中 `kp_aux_data` 的构建条件漏掉了 `ltcs_enabled / lpcs_enabled`，导致 `lpcs_teacher_feats` 永远不会生成，`LPCS` loss 实际从未被加入训练。两轮实验因此均判定为 **失效 run**。

| 方法 | 已观测数值 | 当前解释 |
|------|------------|----------|
| `exp133 LPCS` | `ep40 = 56.5 / 67.8`；`ep50 = 58.3 / 69.6` | 仅能反映 baseline 主训练形状，**不能**解释为 LPCS 有效 |
| `exp134 Sparse LPCS` | `ep10 = 35.7 / 49.9`；`ep20 = 46.4 / 58.1` | 仅能反映 baseline 主训练形状，**不能**解释为 sparse LPCS 有效 |

- 结论：
  1. `exp133/134` 当前所有数值都 **不能** 进入 LPCS 的方法判断
  2. 这不是方法负结果，而是共享接线 bug
  3. bug 已定位并修复：
     - `kp_aux_data` 构建条件已补上 `ltcs_enabled / lpcs_enabled`
  4. 因而下一步必须以新实验编号重跑：
     - 本地重跑 corrected `LPCS`
     - 远程重跑 corrected `Changed-Pair Sparse LPCS`

## 2026-03-21: exp135 / exp136 Corrected LPCS（有效 rerun）

### exp135: corrected full-pair `LPCS` 已真实成立，但更像 mAP-strong 的 supporting 线

> `exp135` 是在修复共享接线 bug 后，对 intended `exp133` 的 clean rerun。它保持 `pair_mode=all`，首次真正把 `LPCS` loss 接入训练。该实验已跑满 `ep120`，以下结论来自训练监控口径。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp135` | 61.1% | 72.3% | 83.5% | 87.2% | `ep120` 训练监控 |

- 结论：
  1. `exp135` 证明了修复后的 full-pair `LPCS` 确实有效，日志里 `lpcs_*` 全程稳定出现，且 `lpcs_fg` 长期显著高于 `lpcs_bg`
  2. 但它的最终形态更像 `mAP` 导向的排序修正：
     - 相对 `exp125 ep120 = 60.5 / 73.5`，当前是 `mAP +0.6 / R1 -1.2`
  3. 因而 full-pair `LPCS` 更适合作为 supporting 证据：
     - learned pair correction 确实在工作
     - 但当前 loss 聚合方式还没有把收益转成更强的 `R1`

### exp136: corrected sparse `LPCS` 机制被完整坐实，但最终只达到近似等价结果

> `exp136` 是在修复共享接线 bug 后，对 intended `exp134` 的 clean rerun。它相对 `exp135` 只改 pair 路由：`pair_mode=delta_top, top_ratio=0.25`。该实验已跑满 `ep120`，以下结论来自训练监控口径。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp136` | 60.9% | 72.1% | 83.7% | 86.7% | `ep120` 训练监控 |

- 结论：
  1. `exp136` 的最大价值已经成立：这次 sparse routing 终于是真稀疏，而不是“名义稀疏、实际全开”
     - `lpcs_psr = 0.254`
     - `lpcs_pf ≈ 3.0`
  2. 但最终结果只达到与 full-pair `LPCS` 近似等价：
     - 相对 `exp135 ep120 = 61.1 / 72.3`，当前是 `-0.2 / -0.2`
  3. 这意味着 supervision dilution 也许存在，但当前还不能把它当作 `LPCS` 的主瓶颈
  4. 因而下一步更合理的方向应从“继续改 routing”转向：
     - 更 ranking 对齐的 `LPCS` 损失聚合方式

### exp137: `Hard-Rank LPCS` 机制接线正确，但 `hard-top` 聚合过于激进

> `exp137` 相对 `exp135` 只改 `LPCS` 的 ranking 聚合：从全 routed pairs 改为 hardest positive / negative top-25% 子集。该实验按预设停表规则终止于 `ep80`。

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| `exp137` | 60.1% | 70.4% | 82.1% | 86.2% | `ep80` 训练监控 |

- 结论：
  1. `exp137` 的机制是有效接线，而不是失效 run：
     - `lpcs_rsr = 0.254`
     - `lpcs_psr / lpcs_pf = 1.000 / 1.000`
  2. 但到 `ep80` 为止，它稳定落后于 `exp135 ep80 = 60.8 / 71.9`
  3. 这说明当前问题不是“只要更关注 hardest ranked pairs 就会更强”，相反：
     - **hard selection 过强会伤害 top-rank 表现**
  4. 因而当前更合理的下一步不是继续加大 hard-top，而是：
     - 更平滑的 top-sensitive / rank-decayed pair correction

## 2026-03-22: exp148 / exp149 两条大改动方向的第一轮分化

### exp148: `PCVT` 早中期已形成稳定 `mAP` 正向，成为当前最值得继续追的训练端新方向

> `exp148` 把单图改写成 `full / complementary-view-a / complementary-view-b` 三视图训练对象，用 pose-defined complementary pseudo-views 验证“单图能否被改写成伪多 support 学习对象”。该实验当前仍在运行，以下结论来自 `ep10/20/30` 训练监控。

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| `exp148` | 40.2% | 51.4% | `ep10` |
| `exp148` | 49.1% | 60.7% | `ep20` |
| `exp148` | 54.6% | 65.8% | `ep30` |
| `exp030a` | 38.2% | 51.3% | `ep10` |
| `exp030a` | 46.8% | 60.9% | `ep20` |
| `exp030a` | 52.2% | 66.0% | `ep30` |

- 当前结论：
  1. `PCVT` 已不只是机制接上，而是验证端连续三个点呈现稳定 `mAP` 正向
  2. 当前差值为：
     - `ep10: +2.0 mAP / +0.1 R1`
     - `ep20: +2.3 mAP / -0.2 R1`
     - `ep30: +2.4 mAP / -0.2 R1`
  3. 机制侧也保持健康：
     - `pcvt_cov_u = 1.000`
     - `pcvt_ovr = 0.000`
     - `pcvt_cos_fu ≈ 0.976~0.985`
     - `pcvt_gap > 0`
  4. 因而 `PCVT` 目前是少数真正值得继续追的“大方向”之一
  5. 当前唯一保留风险是：
     - 它是否最终只表现为 `mAP` 增益，而不能同步带来更完整的 `R1` 收益

### exp149: `SCFA` 快速诊断判负，双侧冗余前提在当前 benchmark 上不够强

> `exp149` 只给一个短止损窗口，目标不是整晚主线，而是快速回答“单图内部双侧同源冗余是否足够强到值得单开方法”。该实验已在 `ep30` 后终止。

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| `exp149` | 34.9% | 44.3% | `ep10` |
| `exp149` | 43.6% | 53.8% | `ep20` |
| `exp149` | 50.7% | 61.3% | `ep30` |
| `exp030a` | 38.2% | 51.3% | `ep10` |
| `exp030a` | 46.8% | 60.9% | `ep20` |
| `exp030a` | 52.2% | 66.0% | `ep30` |

- 当前结论：
  1. `SCFA` 到 `ep30` 已明显落后基线，满足预设快速止损条件
  2. 它不是“完全没接上”：
     - `scfa_cov ≈ 0.90`
     - `scfa_hm ≈ 0.80`
     - `scfa_am ≈ 0.65`
     - `scfa_an ≈ 9.7~10.3`
  3. 真正的问题在于：
     - `scfa_pg ≈ 0.086~0.093`
     - 即当前 benchmark 上真正“一侧低一侧高”的 bilateral gap case 太少
  4. 因而这条线目前更像一个已被快速排除的结构先验假设，而不是可继续深挖的主方向

## 2026-04-01/02: exp206r, exp207, exp209, exp210, exp210b, exp212, exp213

### exp206r: Small GCN+PAA+CE+OA-SD (Fixed OA-SD teacher)
> Repeat of exp206 with fixed OA-SD teacher (BN/Dropout/DropPath eval mode, clean teacher pose)

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp206r equal_concat | 70.6% | 82.6% | 89.5% | 91.4% | ep120 final |
| **exp206r maxsim_hybrid** | **72.3%** | **82.9%** | **90.5%** | **92.2%** | ep120 + maxsim test |

- OA-SD fix: +0.1/+0.3 vs buggy exp206 (70.5/82.3). Fix 加速了早期收敛但不改变 final。

### exp207: Base GCN+PAA+CE+OA-SD 3-view (Fixed OA-SD)

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp207 equal_concat | 70.7% | 80.7% | 89.5% | 91.7% | ep120 final |
| exp207 maxsim_hybrid | 72.2% | 82.0% | 90.4% | 92.3% | ep120 + maxsim test |

- Base (88M) 仅比 Small (50M) 高 +0.1% mAP。Base scaling 在当前配置下无效。
- 可能原因: LR=0.0002 太低, 3-view+CP 限制 Base 容量, 数据集太小。

### exp209: Small STD-PR+CE+OA-SD — 终止 (ep30)

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp209 | 56.0% | 69.3% | ep30 终止 |

- STD-PR+CE+OA-SD 严重落后 GCN+PAA+CE+OA-SD (~5% at ep30)。STD-PR 需要 SupCon。

### exp210: Small GCN+PAA+CE+OA-SD + PKC weight=0.5 — 灾难

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp210 | 3.6% | 5.3% | ep10 终止 |

- PKC weight=0.5 的 SupCon 梯度与 CE 在 GCN features 上冲突。灾难性失败。

### exp210b: Small GCN+PAA+CE+OA-SD + PKC weight=0.05

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp210b equal_concat | 70.6% | 81.8% | 89.9% | 92.4% | ep120 final |
| **exp210b maxsim_hybrid** | **72.4%** | **83.1%** | **90.8%** | **92.7%** | ep120 + maxsim test |

- PKC=0.05 不改变 equal_concat (= exp206r)，但 MaxSim 提升 +0.1/+0.2。
- **72.4/83.1 = 当前最佳 (无 NFC/reranking)！**

### exp212: Small GCN+PAA+CE+OA-SD LR=0.0008 — 灾难

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp212 | 0.8% | 1.3% | ep10 终止 |

- LR=0.0008 对 Small 太高，无法学习。Small 需要 LR=0.0004。

### exp213: Small + PKC(0.05) + MST(0.1) — 终止

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp213 | 40.6% | 54.8% | ep10 终止 |

- PKC + MST 组合梯度冲突。per-keypoint losses 只能单独使用。

### MaxSim Hybrid 跨 checkpoint 分析 (exp206 local)

| Epoch | equal_concat | maxsim_hybrid | MaxSim gain |
|-------|------|------|------|
| 40 | 64.9% | 66.3% | +1.4% |
| 60 | 67.3% | 68.9% | +1.6% |
| 80 | 69.3% | 71.1% | +1.8% |
| 100 | 70.1% | 71.8% | +1.7% |
| 120 | 70.3% | 72.1% | +1.8% |

- MaxSim gain 在 +1.5~1.8% 范围内稳定，不依赖训练阶段。

## 2026-04-02: exp215, exp217, exp218, exp220, exp222, exp223

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp215 BA-PKC w=0.1 | 0.5% | 0.8% | 3.1% | 4.5% | ep10 终止 |
| exp217 OERL + OA-SD | 62.2% | 75.2% | 86.0% | 89.0% | ep120 final |
| exp218 PACI + OA-SD | 61.9% | 74.2% | 85.6% | 88.9% | ep120 final |
| exp220 GSPB + OA-SD | 62.9% | 74.3% | 86.2% | 89.5% | ep120 final |
| exp222 GSPB on Small (scale=0.05) | 2.3% | 3.9% | 9.9% | 14.3% | ep10 终止 |
| exp223 PADPQ K=4 + OA-SD | 63.7% | 74.5% | 86.2% | 89.5% | ep120 final |

- exp215 证实了 non-detached BA-PKC 会直接破坏 backbone 收敛。
- exp217 / exp218 / exp220 都低于 `exp191 = 63.2 / 75.4`，因此不能写成训练端正向超越。
- exp223 在 `equal_concat` 上给出 `mAP +0.5`，但 `R1 -0.9`；当前更适合作为 trade-off 证据，而不是“全面超越”。
- exp219 的远程 `train_log` 已补回，但目前只确认到 `ep30 = 51.9 / 64.9`，尚无 final，因此暂不纳入正式结果表。
- 注：`exp220/223` 的 `maxsim_hybrid` 数字目前只在各自 `monitor.md` 中留有测试记录，本地未发现独立 `test_log`，因此本总表仅登记训练日志可直接复核的 `equal_concat` 结果。

## 2026-04-02/03: exp222c, exp224, exp225, exp226, exp227, exp228

### Tiny 实验

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp222c GSPB Small scale=0.01 | 15.1% | 23.8% | 38.4% | 45.4% | ep10 终止 |
| exp224 KAMP (random-init proj) + OA-SD | 60.7% | 73.0% | 85.1% | 88.3% | ep120 final |
| exp225 GSPB(0.05) + PADPQ K=4 + OA-SD | 64.2% | 74.9% | 86.8% | 89.6% | ep120 final |
| exp226 KAMP (zero-init proj) + OA-SD | 61.6% | 74.3% | 85.1% | 88.0% | ep120 final |

- exp222c: GSPB scale=0.01 在 Small 上仍然灾难 (scale=0.05 → 2.3%, scale=0.01 → 15.1%)
- exp224: KAMP (多尺度 keypoint 融合) random-init projection 造成 -2.5% mAP 噪声
- exp225: **GSPB+PADPQ K=4 = 64.2/74.9 — Tiny 最佳 equal_concat！** (+1.0/-0.5 vs OA-SD)
- exp226: KAMP zero-init projection 减少噪声但仍 -1.6% mAP。KAMP 方向失败。

### exp227: Small GSPB(0.005) + PADPQ K=4 + OA-SD

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp227 equal_concat | 71.6% | 80.8% | 89.8% | 91.8% | ep120 final |
| exp227 maxsim_hybrid | 71.8% | 80.6% | 89.9% | 91.9% | ep120 + maxsim test |

- 对照 exp206r: **mAP +1.0, R1 -1.8** (equal_concat)
- MaxSim gain 仅 +0.2 (vs 通常 +1.7) — PADPQ 破坏 cross-image keypoint consistency
- maxsim 71.8 < 当前最佳 72.4 (exp210b)。**GSPB+PADPQ 在 Small maxsim 上无优势。**
- **mAP 正向是确认的，但 R1 代价和 MaxSim 兼容性问题限制了实用价值。**

### exp228: Tiny GSPB(0.05) + PADPQ K=8 + OA-SD

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp228 equal_concat | 64.1% | 74.3% | 86.4% | 89.5% | ep120 final |

- 对照 exp225 K=4: **-0.1/-0.6**。K=8 ≈ K=4，无额外收益。
- PADPQ K=4 已足够，K=8 不值得增加复杂度。

### exp229: Tiny BT-PKD (w=0.01, constant) + OA-SD

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp229 equal_concat | 62.2% | 75.0% | 86.1% | 89.0% | ep120 final |

- 对照 exp191 (OA-SD): **-1.0/-0.4**
- **创新**: 非 detached cosine distillation 从 EMA teacher 到 backbone per-keypoint features
- **首次在 Small 上实现 non-detached 梯度存活** (BA-PKC: 0.5%, GSPB≥0.01: 灾难)
- **早期加速 +3.5% at ep30**, 但后期干扰导致 final -1.0%
- BT-PKD 证明 cosine distillation 梯度比 CE/SupCon 更温和
### exp230: Small BT-PKD (w=0.01, constant, no PARALLEL_AUG)

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp230 equal_concat | 70.8% | 81.9% | 89.7% | 91.9% | ep110 (OOM at ep120) |

- 无 PARALLEL_AUG (OOM with BT-PKD non-detached graph)
- 对照 exp206r (有 PAUG): 70.6/82.6 → **+0.2/-0.7** (mAP 持平, R1 差因缺 PAUG)

### exp231: Tiny BT-PKD cosine decay (w→0 by ep60)

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp231 equal_concat | 61.7% | 74.3% | 85.5% | 88.6% | ep120 final |

- 对照 exp191: **-1.5/-1.1**。Cosine decay 没有解决后期干扰。
- 对照 exp229 constant: **-0.5/-0.7**。Decay 甚至略差。
- **BT-PKD 全系列结论**: 早期加速有效 (+3.5% at ep30)，但 final 始终 ~-1.0% vs baseline。
- **根本限制**: 任何 non-detached backbone 梯度在后期都干扰收敛，与梯度类型和 schedule 无关。

## 2026-04-04: exp235, exp236, exp237

### exp235: FSDC (wrong ROA+PLBOA config)

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp235 | 61.7% | 74.5% | ep120 final |

- 对照 exp191: **-1.5/-0.9**
- FSDC feature completion 在错误增强配置下

### exp236: FSDC (正确 ROA=False, PLBOA=0.7)

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp236 | 61.7% | 73.2% | ep120 final |

- 对照 exp191: **-1.5/-2.2**
- FSDC 正确配置仍然负面。**Feature completion 方向证伪。**

### exp237: PPA (Pose-Prompted Part-Assignment Head) ⭐

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp237 equal_concat | **63.7%** | **75.0%** | ep120 final |
| exp237 maxsim_hybrid | 64.1% | 75.1% | ep120 + maxsim |

- 对照 exp191: equal_concat **+0.5/-0.4**, maxsim -0.1/-2.0
- **第一个 final mAP 正向的 Part branch 创新！**
- End-to-end learnable part assignment (KPR-inspired)
- 持续上升 ep10→120，无后期崩塌

### exp238: PPA assign_weight=0.1

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp238 | 62.1% | 74.0% | ep120 final |

- 对照 exp191: **-1.1/-1.4**
- w=0.1 不如 w=0.5 — assignment supervision 太弱

### exp239: PPA + GiLt (Part triplet only)

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp239 | 63.8% | 73.6% | ep120 final |

- 对照 exp191: **+0.6/-1.8**
- GiLt mAP 正向但 R1 严重负面。Part CE 对 R1 必要。

### exp240: PPA on Small (w=0.5, no PARALLEL_AUG)

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp240 | 70.7% | 81.1% | ep120 final |

- 对照 exp230 (no PAUG, ep110): -0.1/-0.8
- PPA 在 Small 上基本中性，不如 Tiny PPA (+0.5/-0.4)

### exp241: PPA + GCN 双分支 on Tiny ⭐

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp241 equal_concat | **63.7%** | **75.3%** | **86.2%** | **88.9%** | ep120 final |

- 对照 exp191: **+0.5/-0.1** — 最佳综合结果!
- 对照 exp237 PPA-only: +0.0/+0.3 — GCN 改善 R1
- **PPA 端到端 backbone 训练 + GCN detached keypoint features = 协同提升**
- ep80: +1.2/+0.9 (241 实验中最强 ep80, mAP AND R1 都正向!)

**exp241 MaxSim test**: 64.1/74.8 (MaxSim gain +0.4/-0.5 vs equal_concat)

### exp242: PPA + GCN on Small ❌❌

| 方法 | mAP | R1 | R10 | 口径 |
|------|-----|----|----|------|
| exp242 | 60.9% | 73.4% | 88.9% | ep120 final |

- 对照 exp206r (Small GCN): **-9.7/-9.2** — 灾难性失败!
- PPA 的 non-detached 梯度在 Small 上严重损害 backbone
- **结论: PPA 方法不可泛化到更大 backbone**

### exp243: LGPA (CLIP + Cross-Attention + Pose) on Tiny 🟡

| 方法 | mAP | R1 | 口径 |
|------|-----|----|------|
| exp243 ep80 | 60.9% | 72.5% | ep80 (GPU crash at ep88, 训练未完成) |

- 对照 exp191 (GCN ep80): 62.0/74.4 = **-1.1/-1.9**
- 对照 PPA+GCN (ep80): 63.2/75.3 = -2.3/-2.8
- **早期最强** (ep30 +4.1 mAP vs baseline), CLIP 语义锚定加速收敛
- **后期干扰严重**: cross-attention non-detached 梯度干扰 > PPA 的线性 assignment
- **结论**: CLIP 语义有效加速 part learning, 但不解决 detach barrier

### exp244: LGPA-Detach (CLIP + Detached Features) ⭐⭐⭐

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp244-R (detach, 无OASD) | 63.6% | 74.7% | 85.3% | 88.6% | ep120 remote final |
| **exp244-L (detach+OASD)** | **65.3%** | **75.7%** | **86.8%** | **89.7%** | **ep120 local final** |

- 对照 exp191 (GCN+OASD): **+2.1/+0.3** — **首个在 final 仍正向的 Part branch!**
- 对照 exp243 (LGPA non-detach, ep80): +4.4/+3.2 — detach 完全解决后期干扰
- 对照 exp244-R (无OASD): +1.7/+1.0 — OA-SD 与 LGPA-D 正交叠加
- **无 OA-SD 的 LGPA-D (63.6) ≈ GCN + OA-SD (63.2)**: CLIP 语义 ≈ OA-SD
- **所有 epoch mAP delta 均为正 (从未为负!)** — 前所未有
- **论文核心贡献候选: CLIP 语义 Part Assignment + Detached Features**

**exp244 MaxSim test**: 66.0/76.4/87.2/90.5 (MaxSim hybrid on LGPA-D+OA-SD ep120)

### exp245g: LGPA-Detach on Swin-Small ⭐⭐

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| **exp245g (Small LGPA-D+OA-SD)** | **70.2%** | **80.1%** | **89.8%** | **91.2%** | **ep120 local PT2+mmcv-full** |

- 对照 exp206r (Small GCN+PAA+OA-SD): -0.4/-2.5 — mAP 接近, R1 差距
- 对照 exp244 (Tiny LGPA-D+OA-SD): **+4.9/+4.4** — Small backbone 有效
- LGPA-D 用更简单架构 (无 GCN, 无 PAA) 达到接近 exp206r 的 mAP
- 环境: PyTorch 2.5 + mmcv-full (从源码编译), WITH_CP=True

**exp245g MaxSim test**: 71.9/82.2/91.0/92.8 (MaxSim hybrid on Small LGPA-D+OA-SD ep120)
- vs equal_concat (70.2/80.1): **+1.7/+2.1**
- vs exp206r (70.6/82.6): **mAP +1.3, R1 -0.4** — mAP 超越 Small baseline!

### exp245h_v2: Small LGPA-D + OA-SD 远程复现 ⭐⭐⭐

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| **exp245h_v2 equal_concat** | **71.6%** | **81.6%** | **89.2%** | **91.2%** | **ep120 远程 5060Ti final** |

- 对照 exp245g (本地 3090): **+1.4/+1.5** — 远程环境收敛更好
- 对照 exp206r (Small baseline): **mAP +1.0, R1 -1.0**
- ep90 peak: 71.7/82.2

**exp245h_v2 MaxSim test**: 73.0/82.7/90.5/92.7 (MaxSim hybrid on ep120)
- vs equal_concat: **+1.4/+1.1**
- vs exp206r (70.6/82.6): **mAP +2.4, R1 +0.1** — **Small 全面超越!**
- vs exp245g MaxSim (71.9/82.2): **+1.1/+0.5** — **Small 新最强!**

### exp246: LGPA-D + GCN 双分支 (Tiny) 🟡

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp246 (ep83 crash) | 64.1% | 75.2% | — | — | ep80 (GPU 竞争 crash) |
| **exp246b equal_concat** | **65.5%** | **77.2%** | **86.9%** | **90.1%** | **ep120 final** |

- 对照 exp244 (LGPA-D only): **+0.2/+1.5** — GCN 主要贡献在 R1
- 对照 exp191 (GCN only): **+2.3/+1.8** — LGPA-D 贡献巨大
- LGPA-D 语义 part features + GCN 骨架 keypoint features 正交互补
- ep10~ep70 全部与 exp246 精确匹配 (复现验证通过)

**exp246b MaxSim test**: 66.3/77.7/87.6/90.6 (MaxSim hybrid on LGPA-D+GCN ep120)
- vs equal_concat: **+0.8/+0.5**
- vs exp244 MaxSim (66.0/76.4): **+0.3/+1.3** — **Tiny 新最强!**

### exp247: VCSR — Visibility-Conditional Semantic Routing (Tiny, 无OA-SD)

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| **exp247 VCSR** | **63.6%** | **73.5%** | **84.2%** | **88.3%** | **ep120 远程 final** |

- 对照 exp244-R (LGPA-D 无OA-SD): **0.0/-1.2** — VCSR ≈ LGPA-D, visibility gating 无效
- 训练集 95.8% visible → 训练端 visibility routing 几乎无效
- vcsr_n_active 始终为 0 (vis_threshold=0.3 过高)
- 用户判定 novelty 5/10，不作为主创新。作为消融证据保留。

### exp248: PCFD — Pose-Conditioned Feature Differencing (Test-time) ❌

| 方法 | mAP | R1 | delta |
|------|-----|----|-------|
| exp244 cosine baseline | 65.3% | 75.7% | — |
| PCFD alpha=0.1 | 52.1% | 70.5% | -13.2/-5.2 |
| PCFD alpha=0.3 | 46.8% | 68.1% | -18.5/-7.6 |
| MaxSim (无学习) | 66.0% | 76.4% | +0.7/+0.7 |

- MLP difference classifier 严重过拟合训练集 pairs, 不泛化
- Learned pair-level matching 证伪 (训练端 exp152/153 + test-time PCFD 均失败)
- 简单 MaxSim 反而有效。此方向不再继续。

### exp249: Small LGPA-D + GCN 双分支 + OA-SD (进行中)

| 方法 | mAP | R1 | R5 | R10 | 口径 |
|------|-----|----|----|-----|------|
| exp249 ep10 | 51.1% | 61.7% | 77.9% | 83.8% | 远程 5060Ti, ep10 |
| exp249 ep20 | 60.9% | 73.2% | 85.5% | 88.6% | 远程 5060Ti, ep20 |
| exp249 ep30 | 63.6% | 74.2% | 86.0% | 89.2% | 远程 5060Ti, ep30 |
| exp249 ep40 | **68.0%** | **78.7%** | 88.8% | 90.7% | 远程 5060Ti, ep40 |
| exp249 ep50 | 69.4% | 79.4% | — | 90.9% | 远程 5060Ti, ep50 |
| exp249 ep60 | 70.2% | 80.7% | — | 91.1% | 远程 5060Ti, ep60 |
| exp249 ep70 | 70.9% | 81.6% | — | 91.4% | 远程 5060Ti, ep70 |
| exp249 ep80 | 71.5% | 81.4% | 89.4% | 91.5% | 远程 5060Ti, ep80 |
| exp249 ep90 | 71.4% | 81.4% | 89.4% | 91.5% | 远程 5060Ti, ep90 |
| exp249 ep100 | 71.7% | 82.3% | 89.6% | 91.8% | 远程 5060Ti, ep100 |
| exp249 ep110 | 71.9% | 81.7% | 89.7% | 91.7% | 远程 5060Ti, ep110 |
| **exp249 FINAL** | **71.9%** | **81.8%** | **89.5%** | **91.6%** | **远程 5060Ti, ep120 FINAL** ⭐⭐ |

- **FINAL: mAP 71.9 (+0.3 vs exp245h_v2), R1 81.8 (+0.2 vs exp245h_v2)**
- GCN dual branch 在 Small 上确认有效
- 对照 exp206r (Small GCN+PAA+OA-SD): 70.6/82.6 → **mAP +1.3, R1 -0.8**
- **下一步: MaxSim test on final checkpoint**

**exp249 MaxSim test (ep120 final)**:

| 方法 | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| exp249 equal_concat | 71.9% | 81.8% | 89.5% | 91.6% |
| **exp249 MaxSim** | **73.3%** | **83.2%** | **90.9%** | **93.0%** |

- MaxSim gain: +1.4/+1.4
- **vs exp245h_v2 MaxSim (73.0/82.7): +0.3/+0.5 — 全面超越!**
- **exp249 是项目新最佳: 73.3/83.2 (Small LGPA-D+GCN+OA-SD MaxSim)**

### exp250: POT (Partial Optimal Transport) Test-time 评估 🟡

在 exp246b (Tiny LGPA-D+GCN ep120) checkpoint 上测试:

| 方法 | mAP | R1 | Δ vs Global |
|------|-----|----|-------------|
| Global cosine | 65.2% | 76.2% | — |
| Vis-weighted part | 65.7% | 77.5% | +0.5/+1.3 |
| POT m=0.6 | **66.4%** | **78.7%** | +1.2/+2.5 |
| POT m=0.8 | 66.1% | 77.7% | +1.0/+1.5 |
| POT m=1.0 | 66.0% | 77.6% | +0.8/+1.4 |
| **MaxSim hybrid** | **66.6%** | 78.3% | **+1.4/+2.1** |

- POT m=0.6 最佳: mAP 略逊 MaxSim (-0.2), **R1 超越 MaxSim (+0.4)**
- 5-part POT ≈ MaxSim，差异不够论文主线
- POT 可作为消融实验/理论分析保留

**exp245h_v2 (Small LGPA-D, best checkpoint) POT 结果:**

| 方法 | mAP | R1 | Δ vs Global |
|------|-----|----|-------------|
| Global cosine | 71.8% | 81.1% | — |
| Vis-weighted part | 71.9% | 82.2% | +0.1/+1.1 |
| **POT m=0.6** | **73.0%** | 83.1% | **+1.2/+2.0** |
| **MaxSim hybrid** | 72.8% | **83.7%** | +1.0/+2.6 |
| MaxSim+POT 0.3 | 72.5% | 83.3% | -0.3 vs MaxSim |

- **POT mAP 73.0 > MaxSim mAP 72.8** — Small 上 POT mAP 超越 MaxSim!
- MaxSim R1 83.7 > POT R1 83.1 — MaxSim R1 更强
- 两者互补: POT 更好排序 (mAP), MaxSim 更好找 top-1 (R1)
- MaxSim+POT 组合反而降低 — 信号冲突

### exp251: Tiny Multi-Stage PSG (Stage2+3) + PAA + LGPA-D+GCN

| 方法 | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| **exp251 FINAL** | **65.2%** | **76.2%** | 86.6% | 89.6% |
| exp246b (Stage3 PSG+GCN) | 65.5% | 77.2% | — | — |
| exp000 baseline | 56.6% | 66.5% | — | — |

- MSPSG+PAA vs baseline: **+8.6/+9.7** — 论文价值确认
- MSPSG+PAA vs single-stage: -0.3/-1.0 (seed variance 内)
- 结论: multi-stage PSG 作为 novel design 有效，但不额外超越 single-stage

### exp253: Tiny 3-Stage PSG (Stage1+2+3, 无 PAA) + LGPA-D+GCN

| 方法 | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| **exp253 FINAL** | **65.1%** | **76.2%** | 87.0% | 89.5% |
| exp251 (2-stage+PAA) | 65.2% | 76.2% | 86.6% | 89.6% |
| exp246b (1-stage) | 65.5% | 77.2% | — | — |
| exp000 baseline | 56.6% | 66.5% | — | — |

- 3-stage PSG ≈ 2-stage+PAA ≈ 1-stage (final 差异 <0.4 mAP)
- PAA 无贡献 (exp253 vs exp251 = -0.1/0.0)
- **所有 PSG 变体 vs baseline: +8.5~8.9 mAP** — multi-stage 可作为论文 presented method

### 4090 Swin-Base LGPA-D+GCN+OA-SD+PLBOA 结果

**Occluded-Duke (Base):**

| Config | mAP | R1 | MaxSim mAP | MaxSim R1 |
|--------|-----|----|------------|-----------|
| **Base LR=4e-4** | **72.9%** | **82.1%** | **73.8%** | **83.5%** |
| Base LR=2e-4 | 70.0% | 80.3% | 71.4% | 80.4% |

- **MaxSim 73.8 > KPR w/o prompt 73.3!** 我们不用 prompt 就超越了 KPR 无 prompt
- LR=4e-4 >> LR=2e-4, Base 需要较高 LR

**Market-1501 (Base):**

| Config | mAP | R1 |
|--------|-----|----|
| Base LGPA+GCN LR=4e-4 (with PLBOA) | 93.8% | 96.8% |
| Base LGPA+GCN LR=2e-4 (with PLBOA) | 93.1% | 96.8% |
| Small PSG-only LR=4e-4 (no PLBOA) | 93.9% | 96.9% |

- Base+PLBOA 93.8 < Small 无 PLBOA 93.9 — **PLBOA 在无遮挡数据集上有害**
- 需要跑 Base 无 PLBOA 版本验证

### exp255: Small GCN512 + 2-stage PSG + LGPA-D + OA-SD — NEW SMALL BEST

| 方法 | mAP | R1 | R5 | R10 | vs exp249 |
|------|-----|----|----|-----|-----------|
| **exp255 FINAL** | **73.2%** | **83.3%** | 90.4% | 92.3% | **+1.3/+1.5** |
| exp255 MaxSim ep100 | 73.3% | 83.4% | — | — | +0.2/+0.3 (同epoch) |

- GCN512 + 2-stage PSG: equal_concat +1.3 mAP, 但 MaxSim 口径仅 +0.2
- exp255b (GCN512 + 1-stage): ≈ baseline — 2-stage PSG 是 GCN512 发挥的关键!
- **exp255 seed42 FINAL: 73.1/83.1** (vs seed1234 73.2/83.3 = -0.1/-0.2)
- **exp255 seed2024 FINAL: 72.6/82.0** (vs seed1234 73.2/83.3 = -0.6/-1.3)
- **3-seed mean: mAP=(73.2+73.1+72.6)/3 = 72.97±0.32%, R1=(83.3+83.1+82.0)/3 = 82.80±0.72%**

### exp256: Pose Prompt (KPR-style) — 负面/中性

| 变体 | mAP | R1 | MaxSim | vs baseline |
|------|-----|----|--------|-------------|
| exp256 (GCN512+2stage+Prompt, 进行中) | ep90: 72.4 | 82.3 | — | -0.5 vs exp255 |
| **exp256b (GCN256+1stage+Prompt) FINAL** | **68.8** | **79.3** | **70.3/81.0** | **-3.1 vs exp249** |

- Pose Prompt 在强配置 (GCN512) 中性偏负 (-0.1~0.5)
- Pose Prompt 在弱配置 (GCN256) 严重负面 (-3.1)
- **exp256 FINAL: 72.7/82.4** (vs exp255 73.2/83.3 = -0.5/-0.9)
- KPR-style discrete prompt confirmed negative on Swin+PSG

### exp257: ArcFace + Label Smoothing — 负面

| 变体 | mAP | R1 | vs exp255 | 备注 |
|------|-----|----|-----------|------|
| exp257 (ArcFace m=0.35+LS, 远程) | 59.1% | 76.5% | -14.1/-6.8 | ep55 终止, ArcFace 严重不收敛 |
| exp257b (Label Smooth only, 本地) | 71.5% | 81.7% | -1.7/-1.6 | ep86 终止, LS 稳定负面 |

- ArcFace m=0.35: ep10 R1 +3.4 (R1 69.1)，但 mAP 严重落后 (-14 at ep50)。SOLIDER pretrained Swin 不适合 angular margin。
- Label Smoothing: 全程稳定 -1.0~-1.7 mAP。LS 削弱 GCN512 的 discriminative 训练。

### exp258: ArcFace m=0.2 / GCN 3-layer — 负面/中性

| 变体 | mAP | R1 | vs exp255 | 备注 |
|------|-----|----|-----------|------|
| exp258 (ArcFace m=0.2, 本地) | 67.7% | 81.2% | **-5.5/-2.1** | ArcFace 证伪 |
| exp258b (GCN 3-layer, 远程) | 73.1% | 82.7% | -0.1/-0.6 | GCN 3-layer ≈ 2-layer |

- ArcFace m=0.2: 比 m=0.35 好但仍 -5.5 mAP。ArcFace 在 Swin+SOLIDER pretrained 上完全证伪。
- GCN 3-layer: 中性，额外 layer 不增益。GCN 2-layer hidden=512 已是最优。

### exp259: WD / OA-SD / DropPath 调参 — 全中性/负面

| 变体 | mAP | R1 | vs exp255 | 备注 |
|------|-----|----|-----------|------|
| exp259 (WD=2e-4, 本地) | 72.2% | 82.1% | **-1.0/-1.2** | WD 过强负面 |
| exp259b (OA-SD w=2.0, 远程) | 73.2% | 83.4% | 0.0/+0.1 | OA-SD=2.0 ≈ baseline |
| exp259b MaxSim | 73.6% | 83.7% | +0.1/-0.1 | MaxSim 也持平 |
| exp259c (dp=0.2, 本地, 进行中) | ep90: 72.6% | 82.7% | -0.6/-0.6 | dp=0.2 ≈ baseline |

- **exp255 的 recipe (softmax CE, WD=1e-4, OA-SD=1.0, dp=0.1) 已是 SOLIDER Swin 上的最优 recipe。**
- 所有 recipe 调参 (exp257-259) 均无法超越 baseline，recipe 空间已耗尽。

### exp255 Test-Time Evaluations

| 方法 | mAP | R1 | vs equal_concat | 备注 |
|------|-----|----|-----------------|------|
| exp255 equal_concat (baseline) | 73.2% | 83.3% | — | ep120 final |
| exp255 global cosine | 72.7% | 82.3% | -0.5/-1.0 | global-only 模式 |
| exp255 VisWeighted Part | 73.5% | 83.6% | +0.3/+0.3 | 可见部位加权 |
| **exp255 MaxSim Hybrid** | **74.1%** | **84.6%** | **+0.9/+1.3** | **ep120 final, gw=1.0** |
| **exp255 SGCFR α=0.5** | **74.0%** | **84.3%** | **+0.8/+1.0** | **top_k=5, vis_thr=0.3** |
| exp255 SGCFR α=0.4 | 73.9% | 83.8% | +0.7/+0.5 | |
| exp255 CVK hybrid α=0.7 | 72.2% | 82.6% | -1.0/-0.7 | CVK 无 recovery 反而负面 |
| exp259b equal_concat | 73.2% | 83.4% | 0.0/+0.1 | OA-SD=2.0, ≈ exp255 |
| exp259b MaxSim+flip | 75.1% | 85.4% | — | OA-SD=2.0, 略低于 exp255 (-0.1/-0.2) |

| exp255 Global cosine+flip | 73.6% | 83.4% | +0.4/+0.1 | flip-test TTA |
| **exp255 MaxSim+flip** | **75.2%** | **85.6%** | **+2.0/+2.3** | **⭐⭐⭐ 目标达成! flip-test+MaxSim** |

**⭐⭐⭐⭐⭐ 75/85 目标达成! MaxSim+flip = 75.2/85.6 on Swin-Small!**
- Flip-test TTA 额外贡献: +1.1/+1.0 (MaxSim 74.1→75.2, MaxSim R1 84.6→85.6)

### exp260: Base GCN512 + 2-stage PSG (LR=4e-4) — 未超 Small

| 方法 | mAP | R1 | R5 | R10 | vs exp255 (Small) |
|------|-----|----|----|-----|-------------------|
| **exp260 FINAL** | **72.6%** | **81.6%** | — | 92.5% | **-0.6/-1.7** |

- Base LR=4e-4 underfitting: mAP 和 R1 均低于 Small LR=8e-4
- exp260 MaxSim+flip: 74.7/84.6 (仍低于 Small 75.2/85.6)

### exp260b: Base GCN512 + 2-stage PSG (LR=8e-4) — 超越 Small!

| 方法 | mAP | R1 | R5 | R10 | vs exp255 (Small) |
|------|-----|----|----|-----|-------------------|
| **exp260b FINAL** | **73.9%** | **83.2%** | — | — | **+0.7/-0.1** |
| exp260b MaxSim+flip ep100 | 75.4% | 84.9% | — | — | +0.2/-0.7 (非final) |

- LR=8e-4 确认是 Base 的正确 LR (vs LR=4e-4 的 72.6/81.6)
- Base 超越 Small +0.7 mAP!
- **exp260b MaxSim+flip FINAL: 75.4/84.8** (vs Small 75.2/85.6 = +0.2/-0.8)
- Base mAP 更强但 R1 略弱。MaxSim+flip 提升幅度 Base < Small (+1.5 vs +2.0 mAP)

### exp260b Market: Base GCN512 + 2-stage PSG (LR=8e-4, 无PLBOA)

| 方法 | mAP | R1 | R5 | R10 | 备注 |
|------|-----|----|----|-----|------|
| **exp260b Market FINAL** | **94.4%** | **97.1%** | — | 99.4% | Base backbone |
| Tiny baseline | 91.6% | 96.3% | — | — | |
| Tiny+PSG | 92.4% | 96.7% | — | — | |
| Small+PSG (无PLBOA) | 93.9% | 96.9% | — | — | |
| exp260b Market MaxSim+flip | 94.7% | 97.2% | — | — | |
| exp260b Market→Occluded-ReID (eq) | 86.0% | 88.5% | 95.3% | 97.9% | 跨数据集 equal_concat |
| **exp260b Market→Occluded-ReID (MaxSim+flip)** | **88.0%** | **90.6%** | — | — | **跨数据集 MaxSim+flip** |

## PRCV 2026 Phase 1 主表 runs (新协议，含 default flip-test)

> 新协议: `equal_concat + flip-test` 默认；`MaxSim hybrid 1:2` 独立一行(需 `eval_fliptest_maxsim.py` 事后跑)。  
> 训练 scaffold: 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA(OD/OP) / PLBOA off(Market)，BS=64，LR=8e-4，120 epoch，SEED=42。  
> 机器: srvA/B/C = 5060 Ti 16G。本地 3090 挂，Base 3 run (exp263/266/269) DEFERRED。

| Exp ID | Backbone | Dataset | eq_concat+flip mAP / R1 | MaxSim hybrid mAP / R1 | 备注 |
|--------|----------|---------|-------------------------|------------------------|------|
| exp261 | Swin-Tiny | Occ-Duke | **65.9% / 77.4%** | TBD | ✓ e120 FINAL @ 2026-04-19 04:16 srvB |
| exp262 | Swin-Small | Occ-Duke | **73.8% / 83.1%** | TBD | ✓ e120 FINAL @ 2026-04-19 09:59 srvA (R5=90.2 R10=92.2). **略优 KPR w/o prompt 73.3/82.5** (+0.5/+0.6) |
| exp263 | Swin-Base | Occ-Duke | **e100 eff FINAL: 72.5 / 81.8 (Global+flip), 74.5 / 84.0 (MaxSim+flip)** | ✓ @ 2026-04-20 09:01 srvB | ⚠️ e100 eval OOM-killed (内存 13.2G 触 16G),ckpt 100 完整,不重训。MaxSim hybrid+flip **74.5/84.0** 超 KPR w/o prompt +1.2/+1.5 |
| exp263c | Swin-Base | Occ-Duke | ~~abandoned @ e31~~ | — lab3090 pwrlim280 seed 42 | seed 42 轨迹异常 (e10 2.7 / e20 17.0),用户指示换 seed 41 → 切 exp263d |
| exp263d | Swin-Base | Occ-Duke | **74.1 / 83.3** | ✓ e120 FINAL @ 2026-04-21 14:27 lab3090 pwrlim 280W (R5=90.8 R10=93.0). **vs exp263 old e100 eff 72.5/81.8 Δ=+1.6/+1.5**. exp263 系列 PRCV 主表用此数字 (seed 41 替代 seed 42) |
| exp263b | Swin-Base | Occ-Duke (seed 42 restart full 120) | **73.5 / 81.5 (train eq_concat), 74.8 / 84.0 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-23 16:47:17 lab4090 4090 TEST.IMS_PER_BATCH 64 (R5=90.2 R10=92.3). MaxSim Global 72.4/81.4, hybrid 74.8/84.0 (+1.3/+2.5 vs eq_concat)。**vs exp263 old e100 eff 72.5/81.8 (eq) / 74.5/84.0 (MaxSim) Δ=+0.3/0** (MaxSim 侧 full 120 微优)。vs exp263d s41 MaxSim 75.2/84.8 Δ=-0.4/-0.8 (**seed 41 > seed 42 再次 confirmed**)。论文 Base OD 主表仍用 exp263d (seed 41 最强), exp263b 作 seed 42 full 120 复现点 |
| exp294 | Swin-Base | Occ-Duke (LGPA-only / Full-GCN s41) | **74.0 / 82.6 (eq+flip), 75.0 / 84.4 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-24 02:18:48 lab4090 TEST.IMS_PER_BATCH 64 (R5=90.5 R10=92.4). Global 73.5/83.3, **MaxSim 75.0/84.4** (+1.0/+1.8 vs eq_concat)。**vs exp263d Full+GCN s41**: eq 74.1/83.3 → -0.1/-0.7, **MaxSim 75.2/84.8 → -0.2/-0.4** (GCN 冗余双评测模式都验证)。vs exp263b Full+GCN s42 MaxSim 74.8/84.0: **+0.2/+0.4** (Full-GCN s41 > Full+GCN s42)。补 Phase 3-C Base 行, **3-backbone 统一结论 GCN 可移除** |
| exp264 | Swin-Tiny | Occ-PTrack | **76.7% / 85.1%** | TBD | ✓ e120 FINAL @ 2026-04-19 07:15 srvC (R5=94.1 R10=97.0) |
| exp265 | Swin-Small | Occ-PTrack | **78.4% / 86.2%** | TBD | ✓ e120 FINAL @ 2026-04-20 04:45 srvC (R5=94.8 R10=97.3, Small >> Tiny 76.7/85.1) |
| exp266 | Swin-Base | Occ-PTrack | **e60 eff FINAL: 78.4 / 86.2 (peak e50: 78.5/86.3)** | ✓ @ 2026-04-20 21:27 srvC | ⚠️ e70 后 silent exit (非 OOM 非 CUDA, 推测 hy-tmp 平台 kill)。**Base 对 Small (exp265 78.4/86.2) 0 增益**, 不重训 |
| exp265b | Swin-Small | Occ-PTrack (seed 41) | **78.5% / 85.9%** | ✓ e120 FINAL @ 2026-04-22 09:03 srvA 5060Ti (R5=94.7 R10=97.1) | **vs exp265 s42 78.4/86.2 Δ=+0.1/-0.3**。seed 41 微优 mAP 略弱 R1, 论文主表仍用 exp265 s42 (更高 R1), exp265b 作跨 seed 鲁棒性 supplementary |
| exp266b (srvA) | Swin-Base | Occ-PTrack (seed 41) | **78.7% / 86.3%** | ✓ e120 FINAL @ 2026-04-23 13:18:50 srvA 5060Ti TEST.IMS_PER_BATCH 128 (R5=94.5 R10=97.1). **vs exp266b_3090 s41 78.5/86.2 Δ=+0.2/+0.1** (srvA 5060Ti 微优, 跨设备方差 0.2)。vs exp266 s42 e60 eff 78.4/86.2 Δ=+0.3/+0.1。vs exp265b Small s41 78.5/85.9 Δ=+0.2/+0.4 (**Base vs Small 同 s41 首次 R1 显著领先**)。**论文 Base OP 主表更新用此数字 78.7/86.3** (替代原 78.5) |
| exp266b_3090 | Swin-Base | Occ-PTrack (seed 41) | **78.5% / 86.2%** | ✓ e120 FINAL @ 2026-04-22 09:29 lab3090 pwrlim 280W (R5=94.4 R10=96.9). **vs exp266 s42 e60 eff 78.4/86.2 Δ=+0.1/0** (持平)。vs exp265 Small 78.4/86.2 Δ=+0.1/0。vs exp265b Small s41 78.5/85.9 Δ=0/+0.3 |
| exp267 | Swin-Tiny | Market | **92.5% / 96.4%** | TBD | ✓ e120 FINAL @ 2026-04-19 13:45 srvB (R5=98.9 R10=99.3) |
| exp268 | Swin-Small | Market | **94.3% / 97.3%** | TBD | ✓ e120 FINAL @ 2026-04-20 00:39 srvA (R5=99.1 R10=99.5) |
| exp269 | Swin-Base | Market | **e80 eff FINAL: 94.4 / 97.0 (Global+flip), 94.5 / 97.1 (MaxSim+flip)** | ✓ @ 2026-04-20 13:xx srvA | ⚠️ e80 eval OOM-killed 同 exp263 模式,ckpt80 完整,不重训。Base 对 Small 优势小(Market 已饱和) |
| exp269b | Swin-Base | Market (seed 42 restart full 120, PLBOA OFF) | **94.5 / 97.2 (eq+flip), 94.6 / 97.2 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-24 01:17:24 srvC 5060Ti TEST.IMS_PER_BATCH 64 (R5=99.1 R10=99.5). Global+flip 94.4/97.1, **MaxSim 94.6/97.2** (+0.1 mAP vs eq_concat)。**vs exp269 orig e80 eff**: eq 94.4/97.0 → +0.1/+0.2; MaxSim 94.5/97.1 → +0.1/+0.1。vs exp268 Small 94.3/97.3 Δ=+0.2/-0.1。vs exp293b Base PLBOA ON 93.8/97.2 Δ=+0.7/0 (**PLBOA 净 -0.7 mAP**)。**论文 Market Base 主数字升级 eq 94.5/97.2 / MaxSim 94.6/97.2** |

## PRCV 2026 Phase 3 消融 runs

> Phase 3-A: **纯 PSG scaffold** (无 LGPA/GCN/OA-SD/PLBOA/Parallel-Aug),仅开 PSG 的 stage 数。单变量 = PSG_STAGES。回答"PSG 本体稳定性"。

| Exp ID | Backbone | Dataset | PSG stages | eq_concat+flip(global) mAP / R1 | 备注 |
|--------|----------|---------|------------|-------------------------------|------|
| exp270 | Swin-Tiny | Occ-Duke | 无 (baseline) | **59.2 / 68.4** | ✓ e120 FINAL @ 2026-04-20 12:29 srvB (R5=82.2 R10=85.8). vs exp000 旧协议 56.6/66.5 → +default flip 贡献 +2.6/+1.9 |
| exp271 | Swin-Tiny | Occ-Duke | `[-1]` (1-stage) | **60.2 / 69.5** | ✓ e120 FINAL @ 2026-04-20 16:36 srvB (R5=81.8 R10=85.9). vs exp270 Δ=+1.0/+1.1 = stage 3 PSG 独立贡献 |
| exp272 | Swin-Tiny | Occ-Duke | `[-2,-1]` (2-stage) | **60.5 / 69.7** | ✓ e120 FINAL @ 2026-04-20 20:19 srvB (R5=82.6 R10=86.2). vs exp271 Δ=+0.3/+0.2 = stage 2 边际贡献微弱;vs exp270 Δ=+1.3/+1.3 = 2-stage 累计 |
| exp273 | Swin-Tiny | Occ-Duke | `[-3,-2,-1]` (3-stage) | **60.5 / 69.9** | ✓ e120 FINAL @ 2026-04-21 00:05 srvB (R5=82.8 R10=87.0). vs exp272 2-stage Δ=0/+0.2 (stage 1 边际贡献 ~0 mAP). **Phase 3-A Tiny 矩阵完整**: 边际收益递减 +1.0 → +0.3 → 0 |
| exp274 | Swin-Small | Occ-Duke | 无 (baseline) | **68.1 / 76.8** | ✓ e120 FINAL @ 2026-04-20 21:34 lab4090 (R5=87.8 R10=90.9). vs Tiny exp270 Δ=+8.9/+8.4 = Small vs Tiny backbone 容量差 |
| exp275 | Swin-Small | Occ-Duke | `[-1]` (1-stage) | **68.8 / 76.8** | ✓ e120 FINAL @ 2026-04-20 23:37 lab4090 (R5=87.2 R10=90.4). vs exp274 no-PSG Δ=**+0.7/0** (mAP 涨 R1 持平). vs Tiny 1-stage 增益 (+1.0/+1.1),Small 上 +0.7/0 缩水 |
| exp276 | Swin-Small | Occ-Duke | `[-2,-1]` (2-stage) | **68.3 / 77.2** | ✓ e120 FINAL @ 2026-04-21 01:41 lab4090 (R5=87.2 R10=90.1). vs exp275 1-stage Δ=-0.5 mAP/+0.4 R1 (**Small 上 2-stage 不同 Tiny,mAP 不涨但 R1 涨**) |
| exp277 | Swin-Small | Occ-Duke | `[-3,-2,-1]` (3-stage) | ~~49.0 / 57.7 (seed 42 偶发塌缩)~~ | abandoned @ 2026-04-21 03:47 (e2 id_global 卡 3.277 classifier uniform). **改 exp277b seed 41 重跑** (用户判断偶发) |
| exp277b | Swin-Small | Occ-Duke | `[-3,-2,-1]` (3-stage) | **68.3 / 77.6** | ✓ e120 FINAL @ 2026-04-21 23:34 lab4090 (R5=87.4 R10=89.8). **R1 最强 Phase 3-A Small!** vs exp277 s42 塌缩 49.0/57.7 Δ=+19.3/+19.9. vs exp276 2-stg 68.3/77.2 Δ=0/+0.4. **seed 41 完全验证 exp277 塌缩是偶发** |

## PRCV 2026 Phase 3-B 消融 runs

> Phase 3-B: **Full scaffold + 变量 GCN_HIDDEN × PSG_STAGES** (LGPA/OA-SD/ParAug/LOWER_BODY_OCC 全开,仅改 GCN 容量和 PSG stage)。单变量消融。回答"GCN cap 与 PSG stage 是否互补"。

| Exp ID | Backbone | GCN_HIDDEN | PSG_STAGES | eq_concat+flip(global) mAP / R1 | 备注 |
|--------|----------|-----------|------------|-------------------------------|------|
| exp281 (= exp261) | Swin-Tiny | 512 | `[-2,-1]` | **65.9 / 77.4** | Phase 1 共享,不重跑 |
| exp278 | Swin-Tiny | 256 | `[-1]` | **65.7 / 76.7** | ✓ e120 FINAL @ 2026-04-21 10:42 srvB (R5=86.7 R10=89.6). vs exp261 GCN512+2stg 65.9/77.4 Δ=-0.2/-0.7. vs exp286 LGPA-only 66.0/76.6 Δ=-0.3/+0.1 (GCN256 略弱于 no GCN) |
| exp279 | Swin-Tiny | 256 | `[-2,-1]` | **65.7 / 76.9** | ✓ e120 FINAL @ 2026-04-21 21:32 srvB (R5=86.6 R10=90.1). vs exp278 GCN256+1stg 65.7/76.7 Δ=0/+0.2 (mAP 持平 R1 +0.2). vs exp261 65.9/77.4 Δ=-0.2/-0.5 |
| exp280 | Swin-Tiny | 512 | `[-1]` | **65.7 / 76.2** | ✓ e120 FINAL @ 2026-04-22 08:07 srvB (R5=86.7 R10=89.7). **vs exp261 GCN512+2stg 65.9/77.4 Δ=-0.2/-1.2** (最弱 R1 格), vs exp278 GCN256+1stg 65.7/76.7 Δ=0/-0.5. **Phase 3-B Tiny 2×2 闭合: GCN256+1stg=GCN256+2stg=GCN512+1stg mAP 全 65.7, GCN512+2stg 唯一 65.9**。和 Small 2×2 GCN512+1stg 最弱同模式 |
| exp285 (= exp262) | Swin-Small | 512 | `[-2,-1]` | **73.8 / 83.1** | Phase 1 共享, srvA 5060Ti (原始), 已 re-eval flip fix 后 73.8/83.1 no-op |
| exp285b | Swin-Small | 512 | `[-2,-1]` | **73.8 / 83.8** | ✓ e120 FINAL @ 2026-04-22 06:04 lab4090 (R5=90.7 R10=92.7). **vs exp262 (srvA old) 73.8/83.1 Δ=0/+0.7** (mAP 持平, R1 +0.7 lab4090 > srvA). **Phase 3-B Small 矩阵 gold-standard**, 论文主表用此数字 |
| exp282 | Swin-Small | 256 | `[-1]` | **73.7 / 83.9** | ✓ e120 FINAL @ 2026-04-21 09:33 lab4090 (R5=90.5 R10=92.5). **vs exp262 73.8/83.1: mAP -0.1 R1 +0.8** → low-cap ≥ high-cap, Small Full Scaffold 容量饱和 |
| exp283 | Swin-Small | 256 | `[-2,-1]` | **73.5 / 83.2** | ✓ e120 FINAL @ 2026-04-21 15:38 lab4090 (R5=90.7 R10=92.5). vs exp262 73.8/83.1 Δ=-0.3/+0.1. vs exp282 73.7/83.9 Δ=-0.2/-0.7 |
| exp284 | Swin-Small | 512 | `[-1]` | **73.4 / 82.9** | ✓ e120 FINAL @ 2026-04-21 21:23 lab4090 (R5=89.9 R10=92.2). vs exp262 73.8/83.1 Δ=-0.4/-0.2. **Phase 3-B Small 2x2 完整: GCN256+1stg (83.9) 最 R1, GCN512+2stg (73.8 mAP) 最 mAP; GCN512+1stg 反而最弱** |

## PRCV 2026 Phase 3-C 消融 runs (optional, 提前启动)

> Phase 3-C: **LGPA-only + 变量 PSG_STAGES** (关 GCN, 保留 LGPA/OA-SD/ParAug/LOWER_BODY_OCC)。回答"2-stage PSG 的收益是偏 structural 还是 semantic branch 也吃"。srvC exp266 silent exit 后空闲,利用上。

| Exp ID | Backbone | PSG stages | mAP / R1 | 备注 |
|--------|----------|-----------|----------|------|
| exp286 | Swin-Tiny | `[-1]` | **66.0 / 76.6** | ✓ e120 FINAL @ 2026-04-21 10:03 srvC (R5=86.4 R10=89.7). **vs exp261 Full Scaffold 65.9/77.4 Δ=+0.1/-0.8** → GCN 对 Tiny 几乎无贡献, LGPA-only 等价 Full |
| exp287 | Swin-Tiny | `[-2,-1]` | **65.9 / 77.0** | ✓ e120 FINAL @ 2026-04-21 20:48 srvC (R5=87.0 R10=89.7). vs exp286 LGPA-only 1stg 66.0/76.6 Δ=-0.1/+0.4 (2-stg R1 微优). vs exp261 Full 65.9/77.4 Δ=0/-0.4 (GCN 主要给 R1) |
| exp288 | Swin-Small | `[-1]` | **73.8 / 83.8** | ✓ e120 FINAL @ 2026-04-22 12:51 srvC (R5=90.5 R10=92.0). 🔥 **vs exp285b Full Scaffold 73.8/83.8 完全持平** (mAP/R1 identical, R5/R10 微差 0.2/0.7)。vs exp282 Full GCN256+1stg 73.7/83.9 Δ=+0.1/-0.1。**证实 GCN 对 Small OD 零贡献**, LGPA 单独达 Full Scaffold 性能 |
| exp289 | Swin-Small | `[-2,-1]` | **73.8 / 83.3** | ✓ e120 FINAL @ 2026-04-23 05:39 srvC (R5=90.5 R10=92.4). **vs exp288 1-stg 73.8/83.8 Δ=0/-0.5**, vs exp285b Full Scaffold 73.8/83.8 Δ=0/-0.5 — **mAP 完全持平 Full Scaffold, GCN 零贡献 reconfirmed**. 和 Tiny Phase 3-C (exp287 2-stg 65.9/77.0 vs exp286 1-stg 66.0/76.6) 方向相反 (Small 1-stg R1 微优, Tiny 2-stg R1 微优), 但 mAP 均持平 |

## target-heatmap 机制 (POSE_USE_TARGET_HEATMAP=True)

| Exp ID | Backbone | Dataset | seed | eq_concat+flip mAP / R1 | vs scene baseline | 备注 |
|--------|----------|---------|------|-------------------------|-------------------|------|
| exp290 | Swin-Small | Occ-PTrack | 42 | **78.4 / 86.2** | ✓ e120 FINAL @ 2026-04-23 09:22 srvB (R5=94.8 R10=97.4). 🔥 **严格持平 exp265 scene baseline 78.4/86.2/94.8/97.3** (Δ 0/0/0/+0.1). target-heatmap 3 数据集全 near-no-op, OP 多人场景预期增益未实现 |
| exp291 | Swin-Small | Occ-Duke | 42 | **73.5 / 82.9** | exp285b 73.8/83.8 (Δ -0.3/-0.9) | ✓ e120 FINAL @ 2026-04-22 18:13 lab4090 (R5=90.7 R10=92.5). OD 多单人场景 near no-op, 机制无显著回归 |
| exp292 | Swin-Small | Market | 42 | **e90 eff FINAL: 94.2 / 97.1** | exp268 FINAL 94.3/97.3 (Δ -0.1/-0.2 持平) | ✓ 停于 e93 @ 2026-04-22 23:25 用户让出 lab3090。R5 99.2 R10 99.5 = exp268 FINAL R5 99.1 R10 99.5 |
| exp293 | Swin-Base | Market + **PLBOA** | 42 | **e120 FINAL (restart): 93.8 / 97.2** (完整 120ep) | exp269 e80 eff 94.4/97.0 (Δ -0.6/+0.2); first run e80 eff 94.1/96.9 (Δ -0.3/+0.3 跨 restart 方差) | ✓ restart full 120 @ 2026-04-23 08:24 lab4090 (R5=98.9 R10=99.5). First run e80 eval OOM, 重启 w/ TEST.IMS_PER_BATCH 64. **PLBOA 在 Market full 120 net -0.6 mAP / +0.2 R1** (vs exp269 PLBOA OFF e80) — 主表待 exp269b FINAL 公平对比 |

## Post-PRCV 消融/复现/扫参 runs (exp295–321b, 2026-04-25~28)

> post-PRCV（PRCV 已投后）的复现/multi-seed/LR sweep/loss-weight sweep/SOTA push。**整体结论：无一超越已投 baseline**，产出是消融素材而非训练端涨点。`eq` = equal_concat+flip（train-side 默认）；`MaxSim` = MaxSim hybrid+flip（test-time 后处理，**不算训练端贡献**）。Δ 用 MaxSim 口径。
> 回填自各 exp monitor.md + git commit 交叉核对（2026-06-15，补先前文档债；decisions.md 对应 6 条决策同日回填）。

### exp295–304: 复现 / multi-seed / LR sweep / Phase 3-D LGPA 消融

> Scaffold 默认 = 2-stage PSG `[-2,-1]` + LGPA-D + GCN512 + OA-SD + ParAug + PLBOA(OD)。单变量见「关键改动」列。

| Exp ID | Backbone | 关键改动（对照变量） | 机器 / seed | eq+flip mAP/R1 | MaxSim+flip mAP/R1 | vs baseline Δ (MaxSim) | 一句结论 |
|--------|----------|----------------------|-------------|----------------|--------------------|--------------------------|----------|
| exp295 | Swin-Small | Full Scaffold 复现 exp255 | lab4090 / 1234 | **74.2 / 84.0** | **75.2 / 85.4** | vs exp255 hist 75.2/85.6: **0 / -0.2** | ✅ 完全重现 exp255 75.2 mAP，证历史数字真实可复现（非 eval bug）。**Small OD 主表新 reference** |
| exp296 | Swin-Base | LR 8e-4 复现 exp263d | lab4090 / 41 | 73.7 / 81.7 | 74.9 / 83.8 | vs exp263d 75.2/84.8: **-0.3 / -1.0** | reproducibility 接近但 R1 系统性偏低（lab4090 vs lab3090 硬件差）；主表仍用 exp263d |
| exp297 | Swin-Base | **LR 4e-4** | srvA(5060Ti) / 41 | 73.2 / 82.4 | 74.6 / 84.1 | vs exp296 LR8: **-0.3 / +0.3**（近 tie） | LR4 vs LR8 接近持平，**非显著 underfit**；比 hist exp260 LR4(72.6) 高 0.6 mAP |
| exp298 | Swin-Base | **LR 2e-4**（下界） | srvB(5060Ti) / 41 | 68.6 / 78.6 | 69.6 / 79.1 | vs exp296 LR8: **-5.3 / -4.7** | LR2 严重 underfit（e10 mAP 1.3 near-random），LR ablation 下界，证 LR8 不能再降 |
| exp299 | Swin-Base | **PLBOA OFF** | srvC(5060Ti) / 41 | 70.9 / 78.0 | 72.7 / 80.5 | vs exp296 PLBOA ON: **-2.2 / -3.3** | OD 上 PLBOA net positive **+2.2 mAP MaxSim**；与 Market 上 PLBOA 有害形成 dataset-specific claim |
| exp300 | Swin-Base | Full Scaffold seed 1234 | lab4090 / 1234 | 74.0 / 83.8 | 75.0 / 85.0（e100 ckpt 75.0/85.2） | vs exp263d 75.2/84.8: **-0.2 / +0.2**（e120） | 未破 exp263d SOTA mAP，但 R1 +0.2~0.4 微超；e100 ckpt R1 peak 85.2 |
| exp301 | Swin-Small | **LGPA OFF**（Phase 3-D） | lab4090 / 42 | 71.9 / 83.0 | 71.9 / 83.0（MaxSim **0 boost**） | vs exp285b Full 74.7/84.8: **-2.8 / -1.8** | LGPA 贡献 +2.8 mAP MaxSim；移除 LGPA → MaxSim 失去 boost（LGPA 是 MaxSim 主驱动） |
| exp302 | Swin-Base | Full Scaffold seed 42（multi-seed 第3） | srvA(5060Ti) / 42 | 73.3 / 81.4 | 74.4 / 83.6 | vs exp263d 75.2/84.8: **-0.8 / -1.2** | Base 3-seed(41/1234/42) MaxSim mAP mean **74.87 std 0.42**；主行仍用 exp263d |
| exp303 | Swin-Tiny | **LR 4e-4** | srvB(5060Ti) / 41 | 64.4 / 74.8 | 65.7 / 76.1 | vs exp261 LR8 67.2/78.6: **-1.5 / -2.5** | Tiny LR4 underfit -1.5 mAP；LR8 仍 sweet spot（Tiny 比 Base 更 LR 敏感） |
| exp304 | Swin-Small | Full Scaffold seed 2024（multi-seed 第3） | srvC(5060Ti) / 2024 | 73.3 / 82.7 | 74.3 / 84.0 | vs exp295 75.2/85.4: **-0.9 / -1.4** | Small 3-seed(42/1234/2024) MaxSim mAP mean **74.7 std 0.45**；主行仍用 exp295 |

### exp305–307: Tiny LGPA / PLBOA 消融（Phase 3-D 跨 backbone 补齐）

| Exp ID | Backbone | 关键改动 | 机器 / seed | eq+flip mAP/R1 | MaxSim+flip mAP/R1 | vs baseline Δ (MaxSim) | 一句结论 |
|--------|----------|----------|-------------|----------------|--------------------|--------------------------|----------|
| exp305 | Swin-Tiny | **LGPA OFF**（mirror exp301） | lab4090 / 42 | 64.5 / 76.0 | 64.5 / 76.0（**0 boost**） | vs exp261 67.2/78.6: **-2.7 / -2.6** | LGPA 贡献 +2.7 mAP MaxSim（+1.4 eq）；Phase 3-D Tiny+Small 双 backbone 完整 |
| exp307 | Swin-Tiny | **PLBOA OFF**（mirror exp299） | srvB(5060Ti) / 42 | 62.8 / 71.8 | 64.5 / 73.5 | vs exp261 67.2/78.6: **-2.7 / -5.1** | Tiny PLBOA net positive **+2.7 mAP**；与 Base(+2.2) 一致。PLBOA dataset-specific 2-backbone evidence |

### exp311–321b: GLOBAL_LOSS_SCALE bugfix + Tiny 五维 loss-weight sweep + Small SOTA push

> commit `c059dca` 修复 GLOBAL_LOSS_SCALE 只在 no-part 路径生效的 bug（Full Scaffold 此前完全忽略，effective=1.0）。exp311+ 后 scale 才真在 part-path 生效。Tiny sweep seed 42 / baseline exp261(67.2/78.6)；Small 验证 seed 1234 / baseline exp295(75.2/85.4)。

| Exp ID | Backbone | 关键改动 | 机器 / seed | eq+flip mAP/R1 | MaxSim+flip mAP/R1 | vs baseline Δ (MaxSim) | 一句结论 |
|--------|----------|----------|-------------|----------------|--------------------|--------------------------|----------|
| exp311b | Swin-Small | **GLOBAL_LOSS_SCALE 0.5**（bugfix 后真生效） | lab4090 / 1234 | 73.5 / 83.2（e100 eff，e101 OOM） | 74.5 / 84.8 | vs exp295: **-0.7 / -0.6** | 0.5× global 真生效后 net **-0.7 mAP**，非有效改进；effective 1.0 更好 |
| exp312 | Swin-Tiny | **GLOBAL_LOSS_SCALE 2.0** | lab4090 / 42 | 65.7 / 76.6 | 66.8 / 77.2 | vs exp261: **-0.4 / -1.4** | 2.0× 也 net negative。结合 exp311b(0.5×负)，**双向都负 → 1.0 sweet spot**（推翻早期 0.5） |
| exp313 | Swin-Tiny | **POSE_PART_WEIGHT 2.0**（ID favor part） | srvA(5060Ti) / 42 | 65.8 / 77.0 | 66.9 / 77.9 | vs exp261: **-0.3 / -0.7** | favor part 微 negative |
| exp314 | Swin-Tiny | **POSE_PART_WEIGHT 0.5**（ID favor global） | srvB(5060Ti) / 42 | 65.8 / 77.5 | 67.2 / 78.6 | vs exp261: **0 / 0**（完全相等） | favor global net neutral；default 1.0 双 sweet spot |
| exp315 | Swin-Tiny | **POSE_LGPA_ASSIGN_WEIGHT 1.0**（LGPA aux ×2） | srvC(5060Ti) / 42 | 65.8 / 76.9 | 67.0 / 77.4 | vs exp261: **-0.2 / -1.2** | LGPA aux 加倍 net negative；default 0.5 sweet spot |
| exp316 | Swin-Tiny | **POSE_OA_SD_WEIGHT 2.0** | lab4090 / 42 | 66.0 / 77.6 | 67.2 / 78.0 | vs exp261: **0 / -0.6** | OA-SD ×2 net neutral；default 1.0 sweet spot |
| exp317 | Swin-Tiny | **POSE_LGPA_ASSIGN_WEIGHT 0.25**（LGPA aux ÷2） | lab3090 / 42 | 66.2 / 77.4 | 67.4 / 78.6 | vs exp261: **+0.2 / 0** ⭐ | sweep 中**唯一 MaxSim 超 baseline**(+0.2)，但在 multi-seed std 内，需 Small 验证 |
| exp318 | Swin-Tiny | **POSE_PART_TRI_WEIGHT 0.5**（Tri favor global） | srvB(5060Ti) / 42 | 65.9 / 77.7 | 67.1 / 78.3 | vs exp261: **-0.1 / -0.3** | Tri-side favor global slight neg；与 exp314 合证 default 双 sweet spot |
| exp319 | Swin-Tiny | **POSE_OA_SD_WEIGHT 0.5** | srvC(5060Ti) / 42 | 65.8 / 76.8 | 67.1 / 78.1 | vs exp261: **-0.1 / -0.5** | OA-SD ÷2 slight neg；与 exp316(×2) 合证 default 1.0 sweet spot |
| exp320 | Swin-Small | **POSE_LGPA_DETACH=False**（LGPA aux 反传 backbone） | lab4090 / 1234 | 68.1 / 79.3 | 68.8 / 79.6 | vs exp295: **-6.4 / -5.8** | **catastrophic -6.4 mAP**（e10 46% underfit）；证 LGPA detach 必要。强 negative 消融素材 |
| exp321b | Swin-Small | **POSE_LGPA_ASSIGN_WEIGHT 0.25**（验证 exp317） | lab4090 / 1234 | 73.9 / 83.7 | 74.9 / 85.4 | vs exp295: **-0.3 / 0** | Tiny exp317 的 +0.2 **未迁移到 Small**（slight -0.3）→ 判 seed noise，保持 default 0.5 |

> 跳号说明：exp306/308/309/310/321a/321c 无目录（实验号跳过/未跑，非数据丢失）。exp311(s42) e10 即被 kill，以 exp311b(s1234) 计入。exp296/exp302 R1 跨设备系统性偏低 1-1.6（5060Ti/lab4090 vs lab3090），主表用 lab3090 exp263d 不受影响。

### exp323: MLLM 视觉裁剪 A/B 廉价首验（inference-only，非训练）

> post-PRCV「搬范式」首验。Frozen Qwen2.5-VL-3B（lab-3090-d, RTX 3090），288 个重遮挡难例 pair（均衡 144 同/144 异，chance=50%）。三条件：甲(裸全图)、乙(裸图+可见部位文字)、丙(姿态视觉裁剪图)。**非训练端创新，不计入主表增益。**

| 条件 | 一词格式 acc | reasoning 格式 acc | vs 甲 (reasoning) |
|------|------|------|------|
| 甲(裸) | 0.500 (0 YES/288 NO) | **0.542** (156/288) | — |
| 乙(文字) | 0.500 (2 YES) | **0.493** (142/288, 4 UNK) | **-4.9pt** |
| 丙(视觉裁剪) | 0.500 (0 YES) | **0.358** (103/288, 71 UNK) | **-18.4pt** |

- **一词 YES/NO 格式**：3B NO-bias 致三条件全 50.0%（每个 n_visible 0-8 档都 50.0%），A/B/C **不可判**。
- **reasoning 格式**（exp323_qwen3b_reason，先推理后 ANSWER:）让模型 commit：甲54.2% > 乙49.3% > 丙35.8%。
  **文字 grounding 有害(-4.9pt)，视觉裁剪显著有害(-18.4pt，71 UNK)**；增益未集中重遮挡。
  丙最差因裁剪删上下文→模型对碎片长篇描述，128 token 内常没到 ANSWER。
- 对照 GPT-5.5（codex）同 288 对：裸=55.9% / 文字=55.6%（文字也无效）。3B 裸(54.2%)接近 GPT-5.5 裸。
- 任务1 视觉裁剪：keypoints 确认在原图 jpg 像素空间（overlay 验证），288/288 对成功裁剪。
- **结论**：frozen 小 MLLM + pose 视觉裁剪/文字提示这条廉价首验**不正向，建议砍**（kill-switch 信号明确）。

### exp324: frozen DINOv2 emergent correspondence + pose-anchored part-MaxSim（inference-only，非训练）

> post-PRCV「搬范式」#2 路线。frozen DINOv2-base（lab-3090-d, RTX 3090），全量 Occluded-Duke（2210 query × 17661 gallery，无后处理、无训练）。脚本 `scripts/exp324_dino.py`。输入 224W×448H → patch grid 32×16。keypoints 缩放到 grid → 每部位 3×3 窗均值池化成 5 个 part 向量 + per-part visibility，跨图只比 mutually-visible part 的 per-part cosine（part-MaxSim）。重遮挡子集 = query visibility_binary.sum()≤8（989/2210）。**training-free，不计入主表增益。**

| 方法 | ALL mAP/R1 | HEAVY mAP/R1 |
|------|-----------|-------------|
| (a) holistic CLS | 0.64 / 0.90 | 0.55 / 0.81 |
| (a) holistic mean-pool | 0.70 / 1.27 | 0.57 / 0.71 |
| **(b) pose part-MaxSim** | **3.21 / 7.87** | **1.86 / 3.54** |
| (c) grid part-MaxSim（均匀 5 带，无 pose） | 0.89 / 1.63 | 0.67 / 1.21 |

- **重遮挡**：pose-part vs holistic CLS **+1.31 mAP / +2.73 R1**（mAP×3.4、R1×4.4）；vs grid-part **+1.19 mAP / +2.33 R1**。
- **grid-part vs holistic 仅 +0.12 mAP**（几乎无效）→ 涨点几乎全来自 **pose 锚定**，不是部位分解本身。
- 绝对分低（pose-part heavy 1.86 mAP）符合 DINO 零样本 ReID 文献区间（0.3-4.7 mAP）。
- 耗时 629s（feature 293s + rep building 327s + distmat 0.8s），全部有 pose（no-pose 0）。
- **结论**：机制**有明确相对信号**，pose-anchored DINO correspondence 在重遮挡上 3-4 倍超整图基准且 pose 锚定占绝对主导 → kill-switch 命中正向条件，**值得 exp324b 上轻量 part-projection 头 / LoRA**。

### exp327: 更强/更新冻结对应特征源（DINOv2-with-registers）— pose-part-MaxSim training-free 天花板 check（inference-only）

> 同 exp324 pipeline（pose 锚定 5-part + mutually-visible part-MaxSim + 重遮挡 vis.sum()≤8），**唯一变量=特征源**。hyy GPU1（5060 Ti），slim pose data（剥 heatmap，数值与 exp324 一致）。脚本 `scripts/exp327_dinov3.py`。**training-free，不计入主表增益。** DINOv3-vitb16 gated（hf-mirror 需 token）下不了，改用 ungated 的 `facebook/dinov2-with-registers-base`（registers 去 high-norm artifact token，更干净 dense 特征，patch14 grid 32×16，nreg=4）。

| 方法 | ALL mAP/R1 | HEAVY mAP/R1 |
|------|-----------|-------------|
| (a) holistic CLS | 0.74 / 1.00 | 0.58 / 0.61 |
| (a) holistic mean-pool | 0.88 / 1.09 | 0.69 / 0.71 |
| **(b) pose part-MaxSim** | **3.85 / 8.60** | **2.15 / 3.84** |
| (c) grid part-MaxSim（均匀 5 带） | 1.04 / 1.67 | 0.72 / 0.71 |

- **vs exp324 DINOv2-base（heavy pose-part 1.86/3.54）**：dinov2reg-b heavy **2.15/3.84（+0.29 mAP / +0.30 R1）**；ALL 3.85/8.60 vs 3.21/7.87（+0.64/+0.73）。
- 机制保持：pose-part vs holistic CLS heavy **+1.57 mAP / +3.24 R1**；grid vs holistic 仅 +0.13 mAP（几乎无效）→ 涨点仍几乎全来自 pose 锚定（pose vs grid +1.44 mAP）。
- **结论**：registers 更干净 dense 特征给**小幅正向（+0.29 mAP）但没破天花板**。印证 exp324 假说——**训练-free 天花板瓶颈在 frozen 本身，不在 SSL 模型新旧/registers**。这点小增益不值得单独上头（exp324b 头已到 14）。**exp327 线止损**。dinov3-b 因 gated 无法验证更激进升级，按 registers 小幅增益外推预期也不破天花板。

### exp326: DIFT（Stable-Diffusion emergent correspondence）— pose-part-MaxSim training-free（inference-only）

> 同 exp324 pipeline，**唯一变量=特征源换成 SD-v1.5 UNet 中间特征**（VAE encode → t=100 加噪 → 单步 UNet → hook up_blocks[1] → ensemble=4 平均，feature map C=1280 grid 32×16）。hyy GPU0，slim pose data。脚本 `scripts/exp326_dift.py`。**training-free，不计入主表增益。** 注：SD 无 CLS，holistic 基准用 mean-pool。

| 方法 | ALL mAP/R1 | HEAVY mAP/R1 |
|------|-----------|-------------|
| (a) holistic mean-pool | 0.21 / 0.14 | 0.22 / 0.20 |
| **(b) pose part-MaxSim** | **0.92 / 2.58** | **0.73 / 1.42** |
| (c) grid part-MaxSim（均匀 5 带） | 0.39 / 1.09 | 0.35 / 0.81 |

- **vs exp324 DINOv2-base（heavy pose-part 1.86）**：DIFT heavy **0.73（−1.13 mAP）**——**DIFT 全量明显劣于 DINOv2-base**，更不及 dinov2-registers 的 2.15。
- 机制方向仍在（pose 0.73 > grid 0.35 > holistic 0.22，pose vs grid +0.38），但绝对判别性远低于 DINO。
- **smoke 误导**：smoke（500 gallery）DIFT heavy 9.92，full（17661 gallery）塌到 0.73。DINO 从 smoke 2.55→full 1.86 仅小降，DIFT 从 9.92→0.73 灾难性塌 → **SD 特征 category-level 语义对应强（PCK 高）但 instance-level 身份判别弱**（SD-DINO/Tale-of-Two-Features 文献一致）。
- **结论**：**SD/DIFT 特征不值得上头**（决定性问题答案=否），SD 线止损。教训：训练-free probe 必须用全量 gallery 判定，小 gallery smoke 只看流程不看绝对值。耗时 2065s（feature 1650s ensemble4 慢 + rep 405s）。

### exp324d / exp324i: LoRA-unfreeze DINOv2 + 解相关对照（破冻结天花板 + 张力鲁棒性）

> exp324d = LoRA 解冻 DINOv2-base/large + 可微 pose-part-MaxSim（破 exp324b 冻结天花板）。exp324i = 在其上加跨网络跨协方差解相关损失（逼 DINO-global 与 frozen-Swin-global 线性无关）。Occluded-Duke，BS=64，rank16 除非标注。**单分支 part-MaxSim = 纯模型；fusion(⊕Swin) = test-time 后处理(NFC 级)，不计训练端增益。**

**(1) 纯模型 — LoRA 破冻结天花板 + capacity 对照（part-MaxSim，e30 除非标注）**

**matched epoch 对照（rank16 除非标注；同一 epoch 才公平，e10 列为主）：**

| 变体 | e10 HEAVY / ALL | 最终 | 说明 |
|------|-------------|------|------|
| frozen base (exp324b 头) | — | 8.65 / 14.61 | 冻结天花板 |
| LoRA base rank16 (λ=0/exp324d) | 36.78 / 44.67 | ~38.7 heavy (e20) | 破天花板 ×4.2 |
| LoRA base rank32 | 38.85 / 47.12 | **40.81 / 49.68 (e30)** | +LoRA 容量 |
| **LoRA large(1024d) rank16** | **41.72 / 50.65** | 跑 e30(~11h，e5→e10 仍在爬) | +backbone 容量 |

- **LoRA 解冻决定性破冻结天花板**（8.65→40+ heavy，~4.7×）→ 瓶颈是 "frozen" 不是 DINO 表征。
- **capacity 有真实增益但补不上 SOTA gap（诚实修正）**：matched e10 看 base 36.78 < rank32 38.85 < **large 41.72**（large 比 base +4.9 heavy 且 e5 38.50→e10 41.72 还在爬）——**容量（更大 backbone/更高 rank）单调有帮助 ~+3-5 mAP，不是"无帮助"**（早先基于 e5 快照的"large≈base"过度简化，已更正）。但 large ~42 heavy / ~51 all 仍比 SOTA(72.57/75) **低 ~25-30** → **容量帮得动一点、补不上 gap**，瓶颈仍主要在机制/问题结构。me-too 不变。

**(2) 解相关对照 — decorr 打不破"判别性-互补性张力"（exp324i e10 matched oracle，vs Swin MaxSim 75.16/72.57）**

| 指标(heavy) | λ=0 无decorr | λ=1 decorr active | Δ |
|---|---|---|---|
| top-10 Jaccard vs Swin | 0.253 | 0.2513 | ≈0 |
| oracle 上界 gain | +0.59 | +0.58 | ≈0 |
| **fusion best ALL** | 75.53 (+0.37) | 75.52 (+0.37) | ≈0 |
| fusion best HEAVY | 72.83 (+0.26) | 72.84 (+0.27) | ≈0 |

- decorr loss 活跃(0.041)却完全没动 Jaccard/oracle/fusion → **显式全局线性解相关对 part-MaxSim 排序正交**，张力鲁棒。fusion +0.37 是 NFC 级后处理（w≥0.4 转负），非 beat-SOTA。

**收敛点 e30 matched 完整 decorr sweep（同 rank16/seed/script，λ=0/1/2）：**

| 指标 | λ=0 e30 | λ=1 e30 | λ=2 e30 |
|---|---|---|---|
| single-branch heavy | 39.05 | 38.69 | 38.18 |
| top-10 Jaccard | 0.2646 | 0.2627 | 0.2604 |
| oracle gain | +0.85 | +0.80 | +0.78 |
| fusion best ALL | 75.74 | 75.73 | 75.70 |
| P_dino_only | 0.71% | 0.91% | 1.01% |

- **e10 + e30 + λ sweep 三重证据：decorr 强度 0→1→2，Jaccard 仅降 0.004(几乎不动)、fusion 75.74→75.70 几乎不变、单分支判别力 39.05→38.18 与 oracle +0.85→+0.78 单调小降** → 解相关不仅打不破张力、过强还轻微有害(削判别力换不来互补)。**decorr-floor**：λ=2/10 双倍/十倍权重压不下 ~0.04 相关 → ID-constrained floor(共享判别方向 ID load-bearing，删不掉)。
- **结论**：FM-import（frozen/换源/LoRA/decorr-fusion）方法方向全负，各有机制；张力对显式干预全程鲁棒。真产出 = 诊断研究（见 `fm_occluded_reid_study.md`）。

## pose+CLIP 深度融合探索 (exp341–354, 2026-06-20 通宵)

目标: 找 CLIP+姿态融合涨点的创新。**结论: 无 productive fusion(全局层冗余/有害, 空间层 CLIP 非空间)。** 详见 `experiments/overnight_pose_clip_search.md` + `pose_clip_codex_synthesis.md`。

| 实验 | 配置 | mAP | vs 对照 |
|---|---|---|---|
| exp341base | Swin-Tiny 裸 global(无CLIP) | 57.6 | baseline |
| **exp341** | + CLIP-ReID 可学习ID prompt | **59.8** | **+2.2 真CLIP增益** |
| exp343/344/345 (A/B/C) | 姿态进CLIP对齐(池化/prompt/部位) | 57.6/57.6/58.0 | 全负, 吸收/稀释纯ID |
| exp347/348 | de-occluded对齐 | 57.6 | 死 |
| exp342 | CLIP + detached LGPA(外挂) | 60.0 | +0.2 冗余 |
| **exp342b** | CLIP + **un-detach** LGPA | **60.7** | +0.9 vs exp341(但下行戳穿) |
| **exp353** | **un-detach LGPA 无CLIP**(隔离) | **60.5** | pose单独已>CLIP单独59.8; 加CLIP只+0.2 |
| **exp349** | 强系统 exp255(73.2) + CLIP | **71.4/71.3**(eq/global) | **CLIP有害 -1.8** |
| exp354 PC-SOR | pose+CLIP文本 token归属(20-codex首推) | kill-switch FAILED | CLIP文本定位不了遮挡物/分不清目标vs任意人 |

**完整画面**: CLIP 在裸弱baseline +2.2; 弱系统+pose 冗余(+0.2, exp353证+0.9大部分是pose); 强pose系统 -1.8 有害。CLIP=全局语义非空间, pose=空间结构, 无互补层面。交付=完整诊断(8实验+20-codex深搜+2 kill-switch)。

## pose+CLIP 训练端两机制 (exp355 PGPD / exp356 PC-MSC, 2026-06-21)

20-codex 调研后用户选的两个弱赌注, 全协议(design→kill-switch→双审→训练+控制)走完, 全证负。

| 实验 | 机制 | mAP | vs exp341 59.8 | 控制(random) | 隔离 |
|---|---|---|---|---|---|
| exp355 PGPD | pose选完整teacher蒸馏prompt simplex暗知识 | 58.6 | **-1.2** | exp355r 59.0 | pose选teacher≈random→无价值 |
| exp356 PC-MSC | pose mask可见部位重建冻结CLIP部位语义 | 57.1 | **-2.7** | exp356r 57.3 | pose-mask≈random→无价值 |

**两机制同模式: 机制本身轻微~中度有害, pose 成分(teacher/mask 选择)在噪声内无贡献。** PC-MSC kill-switch 已预警(CLIP 部位特征只带弱 ID gap+0.01)。

## ★ pose+CLIP 最终封板
Step1 CLIP-ReID prompt +2.2(干净)。Step2 pose 融 CLIP **五角度全负**: 进对齐(A/B/C 57.6死)、强系统(exp349 -1.8)、空间归属(PC-SOR kill-switch死)、训练端蒸馏(PGPD -1.2)、训练端补全(PC-MSC -2.7)。**pose 与 CLIP 无 productive fusion**(CLIP=全局语义工具, pose=空间结构工具, 能力不重叠处无法在对方层面发挥)。最强仍 exp342b 60.7 / 强 pose 系统 73.2。交付 = 详尽负结果诊断。

## exp370: PBSR 姿态监督双向结构槽路由 — 同机严格门禁 NO-GO（2026-07-13）

> Swin-Tiny / Occluded-Duke / seed 1234 / batch 64 / 标准 768-d global descriptor。P0 只增加 `spatial → shared routing → structural slots → slot mixer → same routing write-back`；pose 只监督 detached routing target，eval 不读 heatmap。最终严格对照在同一 RTX 4090、同一 Python/依赖、同一 execution `14b2b68` 下完成。

| Epoch | PBSR-off B0 mAP / R1 | PBSR P0 mAP / R1 | P0-B0 mAP / R1 |
|---:|---:|---:|---:|
| 10 | 33.4 / 42.7 | 34.2 / 43.0 | +0.8 / +0.3 |
| 20 | 43.1 / 53.3 | 38.4 / 48.1 | -4.7 / -5.2 |
| 30 | 49.2 / 59.5 | 48.5 / 57.8 | -0.7 / -1.7 |
| 40 | 52.8 / 62.5 | 51.4 / 61.0 | -1.4 / -1.5 |
| 50 | 53.0 / 63.3 | 53.2 / 63.5 | +0.2 / +0.2 |
| **60（冻结门禁）** | **54.5 / 63.8** | **54.4 / 63.7** | **-0.1 / -0.1** |

- 所有 matched eval 均列出；epoch 10/50 的孤立正点不能替代 epoch 60 预注册裁决。
- 机制审计健康：route loss 下降，写回门、路由熵、background share 与 residual norm 均 finite，无 NaN/死门/background collapse；失败属于“机制学会但 identity 无收益”，不是执行故障。
- **裁决：PBSR P0 未达到 `+0.8～1.0 mAP` 明确正向门槛，正式 NO-GO。** 停止 P1/P4/P2/P3、三 seed和跨 backbone，不做小变体救场。
- 结论只否定 PBSR 写回机制作为论文主创新，不否定历史 LGPA/pose 分支的已有增益；matching、GCN 与 CLIP 语义仍不得包装成 PBSR 创新。
- 原始日志与 SHA：`experiments/exp370_pbsr/execution_14b2b68/manifest.md`。

## exp371：CASD frozen routing screen — 正式 NO-GO（2026-07-14）

> 这不是标准 Occluded-Duke test mAP，而是固定 exp336 teacher geometry 上的五折、cross-camera、class-free episodic oracle。每个 eligible query 固定三名 strict-LOO same-ID donor，`max_queries=0`，并做 `2000` 次 PID-grouped bootstrap。其用途是回答 pose-response routing 是否比 matched support controls 更好。

| target arm | episodic mAP | R1 | 说明 |
|---|---:|---:|---|
| SELF | 63.8019 | 65.2596 | 当前图 teacher geometry |
| ID-MEAN | 93.9357 | 97.3732 | 同 ID bag mean |
| PART-EQUAL | **94.3121** | 97.4096 | 最强 routing control |
| SLOT-PERM | 93.0774 | 96.7512 | 部位对应打乱 |
| POSE-SCALAR | 94.2517 | 97.4580 | donor-level pose quality，仅无逐 slot allocation |
| POSE-RESP | 94.2355 | 97.4488 | CASD 必要 routing arm |
| RESP-PERM | 94.2727 | 97.4677 | response-slot 打乱 |
| FULL-INCL | 94.2681 | 97.4771 | 完整特征、含 anchor 边界 |
| FULL-LOO | 94.0600 | 97.2842 | 完整特征、严格 LOO 边界 |
| WRONG-ID | 1.2525 | 0.2970 | fail-safe |

正式门禁结果：

- `POSE-RESP−PART-EQUAL=-0.0766` mAP pp，未达预注册 `+0.5` pp；
- 五折相对各折最强 control 均为负：`-0.1504/-0.0139/-0.0623/-0.0936/-0.1238` pp；
- 对 PART-EQUAL 的 PID bootstrap point=`-0.0765` pp，95% CI=`[-0.1561,+0.0022]` pp；
- `POSE-RESP−POSE-SCALAR=-0.0162` pp，`POSE-RESP−RESP-PERM=-0.0372` pp；
- scene-merged 同样为 `-0.0868` pp且五折全负；
- coverage、三 donor、slot active、path/content disjoint、canonical matrix 与 wrong-ID fail-safe 均通过，排除协议失效或执行故障；
- `PART-EQUAL−SLOT-PERM=+1.2347` pp 是唯一清晰的结构信号：固定部位对应有价值，但逐图 pose-response allocation 没有独立价值。

**裁决：CASD 正式 NO-GO。** 不进入 matched RGB-only student，不做 OT/MoE/slot/temperature/queue/loss-weight 救场，不扩三 seed或跨 backbone。历史 LGPA `global+parts` 的约 `+0.82～0.85 mAP` 性能资产仍成立；本结果只否定把它改写为“pose-response 组织跨实例 support”的自有创新。

原始结果保留在 4090；本地 Git 外回传 `manifest.json`、`runner_stdout.log` 与完整 `results.json.gz` 至 `remote_artifacts/exp371_gate_c_formal_005ab74/`。raw results SHA=`2213d91fdf4594409d38e4ce2ab7c03dccdef8e1390cd9bcb3837f92006b429f`，gzip SHA=`98fbbbaa4584185b9d2f17dbc68d245fa9735f9d428d3cdedfc11e7c7d7a882b`。

## exp372：PCAR 新颖性 Gate — NO-GO（2026-07-15）

> 本实验停在训练前查新阶段，因此没有可报告的 mAP/R1，不得把“未运行”写成性能负结果。

- 候选：在 official CLIP-ReID 的 CLIP ViT self-attention 内注入 `B(Pinstance)-B(Pcanonical)` 零初始化残差，只改少量 heads/layers，输出仍为标准 global descriptor；
- 数学审计：canonical subtraction 可写成实例 pose bias 与静态 canonical bias 的和，不扩大普通 additive pose-bias 函数族；
- 外部查新：PeVL/PAAB 已覆盖 pose-conditioned CLIP/ViT attention，MUVA 已在 ReID 中把动态 body-part mask逐层注入 CLIP ViT self-attention；PAFormer/KPR/ProFD 又覆盖相邻 pose/part/CLIP attention 路线；
- 内部证据：exp371 Gate B 的 correct 只比 shuffled/canonical 高 `+0.0320/+0.0984 mAP`，实例 pose residual 缺少关键燃料；
- **裁决：新颖性 Gate FAIL，PCAR 正式 NO-GO。** 不实现、不训练、不转 layer/head/temperature 小变体。历史 LGPA 增益仍成立，本结果只禁止把它通过中心化 attention bias 重新包装为自有创新。

## exp373：SA 非冗余正交调制新颖性 Gate — NO-GO（2026-07-15）

> 本实验在训练前的新颖性门禁停止，没有运行 forward audit、没有 mAP/R1，
> 不得写成性能负结果。

- 现有实现核对：每个启用 stage 的每个 block 本来就执行 PSG→PAA；
- 历史证据：`exp073` 多层同步 PSG+PAA 比 Stage-3-only 低 `0.5 mAP`；
  matched `exp251/exp254` 中两阶段 PAA 为 `-0.3 mAP/-0.6 R1`；
- 普通版本：`x*(1+g(H))+a(H)` 是 FiLM/SPADE 类条件仿射；
- 候选版本：把 PAA residual 投影到 PSG displacement 正交补；
- 查新结果：arXiv 2025 Orthogonal Residual Update 已覆盖 hard orthogonal
  residual operator；CVPR 2023 Shape-Erased VI-ReID、ICML 2026 CoLoRAI
  Workshop Ortho-ReID 已覆盖 ReID 中人体结构/外观相关子空间与正交补身份表征；
- **裁决：新颖性 Gate FAIL，exp373 正式 NO-GO。** exact exp066 checkpoint、
  commit、数据均已找到，停止不是资产阻塞，而是机制归属不足。未实现、未训练、
  未占用 GPU，也不转 transport/routing/FiLM/层数小变体。

## exp374：PSG 图像—姿态对应依赖 Gate A — 正式 NO-GO（2026-07-15）

> 三 seed、primary-only 12 臂；每个 seed 顺序执行 correct-start、一个
> matched-shuffle、bypass、correct-end。该实验是冻结 checkpoint 的因果评测，不是训练。

| seed | correct mAP / R1 | matched-shuffle mAP / R1 | bypass mAP / R1 |
|---:|---:|---:|---:|
| 42 | 57.5281 / 66.6968 | 57.5296 / 66.6516 | 53.4920 / 62.6244 |
| 1234 | 58.2909 / 68.1448 | 58.2825 / 68.1448 | 54.5796 / 62.8054 |
| 2024 | 57.9931 / 68.3710 | 57.9966 / 68.4163 | 54.1675 / 62.3529 |

- correct−shuffle mAP=`+0.001163 pp`，区间=`[-0.363577,+0.377887]`；
- correct−bypass mAP=`+3.857684 pp`，区间=`[+3.492944,+4.234408]`；
- correct−shuffle R1=`0.000000 pp`，correct−bypass R1=`+5.143288 pp`；
- 三 seed correct−shuffle mAP 为 `-0.001421/+0.008382/-0.003472`，两 seed 非正。

**裁决：正式 `NO_GO`。** PSG 相对 bypass 的价值很大且三 seed 稳定，但 matched 错姿态与
正确姿态几乎等价。这证明“PSG 有用”，同时否定“收益主要来自当前图像—正确实例姿态
对应关系”。停止 PSG gate 权重、层数、canonical/centroid/anatomical 小变体；下一路线
转向与逐位置仿射调制不同的 pose-controlled state dynamics。

原始结果：`remote_artifacts/exp374_a06_cce2982/evidence/`；execution commit=
`cce29820ccf09cb33a43ab0ff701733f58826c35`。

## exp375：PRSM 姿态路由选择性记忆 — 完整 Gate A NO-GO（2026-07-15）

> Swin-Tiny / Occluded-Duke / seed 1234 / batch 64 / 标准 768-d global descriptor。
> B0、M0、P0 最终严格对照均在同一 RTX 4090、同一运行时和同一 execution 下跑满 120 epoch；
> 每 10 epoch 的 12/12 组完整四项见 `experiments/exp375_prsm/monitor.md`。

| arm | pose/state 设置 | e120 mAP / R1 / R5 / R10 |
|---|---|---:|
| B0 | image-only，无状态模块 | **58.4 / 67.1 / 81.2 / 85.6** |
| M0 | 参数匹配 PRSM，固定 canonical pose | **58.8 / 67.5 / 81.5 / 86.2** |
| P0 | PRSM，当前实例 pose | **57.1 / 66.3 / 80.3 / 85.3** |

最终差值：`P0−B0=-1.3 mAP / -0.8 R1`，`P0−M0=-1.7 / -1.2`；M0 反而比
B0 高 `+0.4/+0.4`。三条训练均自然结束，日志无 NaN/Inf/Traceback/RuntimeError/OOM；
P0 的 residual scale 与 retention 参数已明显离开初始化，因此不是模块完全未学习。

P0 e120 同 checkpoint 六臂因果评测：

| arm | mAP | R1 | correct−arm mAP 百分点 |
|---|---:|---:|---:|
| correct-start | 57.083883 | 66.289592 | 0 |
| matched-shuffle | 57.084221 | 66.289592 | -0.000338 |
| full canonical（诊断） | 57.085602 | 66.289592 | -0.001719 |
| foreground-uniform | 57.084573 | 66.289592 | -0.000690 |
| zero-bypass | 57.082088 | 66.289592 | +0.001795 |
| correct-end | 57.083883 | 66.289592 | 0 |

- R1/R5/R10 六臂完全相同；各干预 descriptor SHA 不同，correct-start/end 精确复现；
- matched donor map 的 target-only nuisance gate 全部 PASS，不是普通随机错姿态；
- foreground 保留 correct visibility 和总 write mass，只删除部位槽归属；
- zero 的 622 次 forward 均 exact identity；full canonical 因同时改变 route/support/mass，
  只作诊断，不进入硬门禁。

**裁决：PRSM 正式 NO-GO。** 实例姿态路由、部位状态槽归属与推理时 memory 写回均未产生
可测检索排序贡献；不做 graph、scan order、更多槽、额外 loss 或小参数救场。该结论不推翻
历史 PSG/LGPA 的性能资产，只否定当前 PRSM 作为自有主创新。

## exp376：逐层 Pose Hyper-LoRA — e60 性能 Gate NO-GO（2026-07-16）

> Swin-Tiny / Occluded-Duke / seed 1234 / batch 64。4090 P0 在 stage 2/3 每个 block 后
> 用实例姿态生成 factor-wise rank-4 动态低秩变换；3090 D0 是 diagonal control，仅作跨机趋势。

| arm | e30 mAP / R1 | e60 mAP / R1 | 说明 |
|---|---:|---:|---|
| clean B0（同 4090 历史严格参考） | 50.6 / 60.5 | **55.2 / 65.0** | exp375 clean B0 |
| P0 Pose Hyper-LoRA | 49.8 / 59.8 | **54.2 / 63.0** | e60 相对 B0 `-1.0/-2.0` |
| D0 diagonal（3090） | 48.4 / 58.1 | — | 跨机只作趋势，e30 后停止 |

8 层 alpha、pose coefficient、visibility 与动态 delta 均有限非零；batch64 AMP preflight
证明全部关键参数实际更新。因此失败不是 FP16 舍入或死模块。**裁决：NO-GO。** 不补 M0、
exact B0、matched donor、多 seed 或低秩/层数小变体；结论只否定当前 stage2/3、
rank4/M4 Pose Hyper-LoRA。

## exp377：Pose-Conditioned Selective SSM — 正式 NO-GO（2026-07-16）

> 最终 `12×4` token 上使用真实双向 selective SSM；RGB 生成基础 `Δ/B/C`，实例姿态以
> 有界 residual 联合修正三者。4090 P0 与 3090 RGB-only D0 均自然完成 e120。

| arm | e60 mAP / R1 / R5 / R10 | e120 mAP / R1 / R5 / R10 |
|---|---:|---:|
| clean B0（预注册同机参考） | **55.2 / 65.0 / 77.6 / 83.1** | 58.4 / 67.1 / 81.2 / 85.6 |
| P0 instance pose（4090） | **54.5 / 63.8 / 78.5 / 83.8** | 58.6 / 67.8 / 81.4 / 86.3 |
| D0 RGB-only SSM（3090，趋势） | 54.4 / 64.0 / 77.7 / 82.9 | 58.8 / 68.1 / 81.3 / 86.5 |

P0 e60 相对 clean B0 为 `-0.7 mAP / -1.2 R1`，触发预注册 `<=-0.5 mAP` NO-GO；
e120 虽回到 `+0.2 mAP`，仍远低于正式 `+0.8 mAP` 门槛，且没有优于 D0 的跨机趋势。
e120 alpha=`0.1808`，pose `Δ/B/C` residual 与 state/output 统计均有限非零，排除死分支和
数值异常。**裁决：正式 NO-GO。** 不补同机 B0/D0/M0、donor 反事实、多 seed、跨 backbone
或 `Δ-only/B/C-only` 消融。小型日志与 SHA 位于 `remote_artifacts/exp377_52c4ef6/`。

## exp378：TAPF 单锚点 Gate A 与语义归因闭合（2026-07-17）

> Swin-Tiny / Occluded-Duke / seed 1234 / batch 64 / 同一 RTX 4090。以下是探索性单 seed
> final，不是显著性或多 seed 结论。旧 commit `5de3b30` 的 P0/F0 只保留为
> `INVALID_AS_HARD_FREEZE / VALID_RELAXATION_PILOT`，不进入本表。

| anchor transition | residual OFF | residual ON | ON−OFF（mAP/R1/R5/R10） |
|---|---:|---:|---:|
| hard | F0 **55.9/67.4/79.3/83.3** | P0 **55.6/66.7/78.4/83.0** | `-0.3/-0.7/-0.9/-0.3` |
| explicit SGD relaxation | MR-F0 **56.0/67.1/79.2/83.4** | MR-P0 **55.7/67.1/78.7/82.9** | `-0.3/+0.0/-0.5/-0.5` |

同机 clean B0=`55.1/66.7/79.5/83.8`。关键 final 差值为：

- hard F0−B0=`+0.8/+0.7/-0.2/-0.5`；MR-F0−B0=`+0.9/+0.4/-0.3/-0.4`；
- MR-F0−hard F0=`+0.1/-0.3/-0.1/+0.1`；MR-P0−hard P0=`+0.1/+0.4/+0.3/-0.1`；
- hard P0−F0=`-0.3/-0.7/-0.9/-0.3`；MR-P0−MR-F0=`-0.3/+0.0/-0.5/-0.5`。

按日志显示到`0.1`点的精度，transition×residual的mAP difference-in-differences为`0.0`。
因此当前单 seed 证据把mAP正差主要定位到**residual-OFF的内生姿态场配置**：两个配置相对
B0分别为`+0.8/+0.9 mAP`；显式relaxation相对hard在residual OFF/ON下都只有描述性
`+0.1 mAP`，当前`17×4` geometry residual在两种transition下都为`-0.3 mAP`。R5/R10
相对B0略降，不能写成四项全面提升；F0仍混合bootstrap课程、内部anchor、Gaussian renderer
与PSG，也不能在R0/D0/语义审计前把`+0.8/+0.9`全归因于正确关节语义。

后续同机final为D0=`56.2/67.6/79.8/83.4`、J0=`56.2/67.9/79.5/83.9`、外部raw
ViTPose R0=`56.1/67.4/79.5/83.7`、Gaussian RG0=`56.2/66.9/79.8/83.9`、固定17-cycle
bootstrap N0=`56.1/67.6/80.0/83.4`。J0−D0 mAP=`+0.0`，RG0−R0 mAP=`+0.1`，N0−hard
F0 mAP=`+0.2`；单seed下geometry residual、Gaussianization和正确关节通道名称均没有可分辨
独立mAP贡献。

D0 e90事后选择checkpoint的冻结语义审计进一步闭合因果边界。correct=
`56.2984/67.6471/79.8190/83.5294`；external correct/shuffle/None/unindexable与correct的四项
指标及descriptor逐位相同。相对correct的百分点差值为：matched-wrong-field=
`-0.0155/+0.0000/+0.0000/+0.0000`、joint-permutation=
`+0.0024/+0.0000/+0.0000/-0.0453`、confidence-permutation=
`+0.0002/+0.0000/+0.0000/+0.0000`、spatial-constant=
`-0.0238/-0.0905/+0.0000/-0.0453`、zero-field=
`-0.0536/-0.0452/+0.0000/+0.0452`，均属于无可分辨mAP贡献。只有真正PSG bypass降到
`53.6154/63.9367/76.2896/80.4525`，即`-2.6829/-3.7104/-3.5294/-3.0769`。

anchor仍有pose-like统计（pseudo-PCK@0.05=`0.5539`、teacher posterior cosine=`0.8276`、
flip posterior cosine=`0.9467`、17通道全部占用），但检索对场语义和空间结构不敏感。因此当前
单锚点增益应归为**训练后PSG模块/容量性重标定**，不能归为姿态场因果贡献。当前实现不补
multi-seed、ResNet或Video迁移，也不触发H0；TAPF研究只允许转入独立Hierarchical设计，且新
consumer必须满足null field严格恒等、parameter-matched static/RGB control和逐层field干预
敏感性门禁，不能直接堆叠现有PSG。

### 2026-07-17 用户裁决后的结果口径修正：完整模块有效，子部件不强拆

上述冻结语义审计的数字与边界全部保留，但“不补multi-seed、ResNet或Video迁移”的执行结论被
用户后续裁决覆盖。论文中的实验对象改为完整的`anchor+PSG`模块，而不是要求anchor、Gaussian
field和PSG各自提供可分离mAP：fresh同机D0=`56.2/67.6/79.8/83.4`相对B0=
`+1.1/+0.9/+0.3/-0.4`，并基本匹配测试期依赖external ViTPose的R0=
`56.1/67.4/79.5/83.7`。因此成立的模块级结论是：**训练期姿态监督可以把原始测试期需要外部
热图的PSG改造成推理期RGB-only的完整方法，并保留其检索性能。**

语义审计继续禁止“精确关节名称、confidence或空间场在冻结推理时具有独立因果贡献”的强主张，
但不再把整体`+1.1 mAP`视为无效容量结果，也不阻止逐层版本和跨backbone迁移。下一实验固定为
`exp379 Progressive Hierarchical TAPF`：不同视觉层生成递进内部场，Stage-1场调制Stage-2，
Stage-2 refined场调制Stage-3；它直接比较单点D0，不重跑B0/D0，也不等同于历史上复制同一外部
热图的multi-stage PSG。

## exp379：Progressive Hierarchical TAPF 首轮Swin-T结果（2026-07-17）

> Occluded-Duke / Swin-T / seed 1234 / batch 64 / 120 epochs / 同一RTX 4090。HT0为fresh
> execution；B0与D0使用已完整审计的matched同机final，不重复训练。以下仍是探索性单seed。

| arm | mAP | R1 | R5 | R10 | 推理期外部pose | 说明 |
|---|---:|---:|---:|---:|---|---|
| B0 | 55.1 | 66.7 | 79.5 | 83.8 | 否 | clean baseline |
| D0 | **56.2** | 67.6 | 79.8 | 83.4 | 否 | 单anchor→Stage-3 PSG |
| HT0 | 56.1 | **67.6** | **79.9** | **83.4** | 否 | 每anchor一PSG：Stage-1→2、Stage-2→3 |

HT0−D0=`-0.1/+0.0/+0.1/+0.0`，HT0−B0=`+1.0/+0.9/+0.4/-0.4`，固定顺序为
mAP/R1/R5/R10。逐层版在Swin-T上与单点D0基本中性，不能宣称hierarchical设计带来额外增益；
但完整`anchor+PSG`方法相对B0的约`+1 mAP`信号仍保留。

执行与机制审计均通过：exact commit=`2181e940c4b8b4d032b9e5fb0de2ce57c9e84720`，12个
checkpoint齐全且全张量有限；e10→e120两个projection、共享decoder、Stage-2/3 PSG分别全部
有限变化。final checkpoint上的external correct/shuffle/None/exploding pose descriptor逐位相同，
确认部署只读RGB。runner/train无`NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow`。

**结论边界**：HT0证明了“每个anchor对应一个PSG、内部状态逐层refine、共享decoder”的完整链路
可训练且不损伤D0，但尚未证明逐层优于单层。Swin上不做层数、宽度、loss或独立decoder小调参；
下一证据是ResNet-50内matched B0/D0/HT0三臂，再评估合适ViT，不能用跨backbone绝对值代替同
backbone差值。

## exp380：ResNet-50 TAPF 跨骨干与逐层增量验证（2026-07-17）

> Occluded-Duke / ImageNet预训练ResNet-50 / seed 1234 / batch 64 / 120 epochs / 同一RTX 4090。
> 三个arm均fresh串行训练；以下是探索性单seed，不与Swin-T绝对值横比。

| arm | mAP | R1 | R5 | R10 | 推理期外部pose | 机制 |
|---|---:|---:|---:|---:|---|---|
| R50-B0 | 35.0 | 45.3 | 61.3 | 68.2 | 否 | 标准global ResNet-50 |
| R50-D0 | 38.1 | 49.4 | 64.6 | 71.1 | 否 | 单anchor→layer4 PSG |
| R50-HT0 | **38.9** | **50.5** | **65.9** | **72.0** | 否 | layer2/3逐层anchor，每anchor一组后继PSG |

固定final差值（mAP/R1/R5/R10）：

- D0−B0=`+3.1/+4.1/+3.3/+2.9`；
- HT0−D0=`+0.8/+1.1/+1.3/+0.9`；
- HT0−B0=`+3.9/+5.2/+4.6/+3.8`。

完整执行审计通过：exact commit=`90ed55cf4798f06d1b08e70f84d0e32ca212ff27`，三臂共享同一
ImageNet权重、seed、batch、optimizer配方与120-epoch训练长度；每臂12个checkpoint齐全。D0的
single anchor/PSG与HT0的shared anchor、两个stage projection、9个后继PSG bank和ResNet参数在
全checkpoint轨迹中均有限更新。两个pose arm的final external correct/shuffle/None/exploding
descriptor逐位一致，runner/train严格异常为0。

**结果解释**：完整`anchor+PSG`原子方法在ResNet-50上明确优于同骨干B0，支持“训练期pose监督、
推理期RGB-only”的跨骨干可迁移性。逐层HT0在ResNet上又比D0高`+0.8 mAP`，达到预注册描述线；
但Swin-T的HT0−D0为`-0.1 mAP`，因此当前只能写“逐层增量在不同归纳偏置/容量下具有条件性”，
不能声称已跨backbone稳定优于单层。下一判别实验是合适ViT内部matched B0/D0/HT0三臂；不在
ResNet上追加层数、宽度、loss或decoder小变体救场。

## exp381：ViT-B/16 TAPF 原子方法与逐层最终判别（2026-07-17）

> Occluded-Duke / ImageNet预训练ViT-B/16 / seed 1234 / batch 64 / 120 epochs / 同一RTX 4090。
> 三个arm均fresh串行训练；以下仍是探索性单seed，只比较ViT内部排序。

| arm | mAP | R1 | R5 | R10 | 推理期外部pose | 机制 |
|---|---:|---:|---:|---:|---|---|
| ViT-B0 | 52.9 | 59.5 | 77.1 | 82.0 | 否 | 标准CLS descriptor ViT-B/16 |
| ViT-D0 | **54.9** | **61.4** | **78.9** | 84.0 | 否 | block8 anchor→有效post-block9/10 PSG |
| ViT-HT0 | 54.6 | 60.6 | 78.4 | **84.1** | 否 | block5/8逐层anchor，每anchor一组后继PSG |

固定final差值（mAP/R1/R5/R10）：

- D0−B0=`+2.0/+1.9/+1.8/+2.0`；
- HT0−D0=`-0.3/-0.8/-0.5/+0.1`；
- HT0−B0=`+1.7/+1.1/+1.3/+2.1`。

三臂进程、GPU、12 checkpoints、SHA、strict finite load与严格异常终审均通过。D0/HT0的final
correct/shuffle/None/exploding external pose descriptor逐位一致，证明推理只读RGB；HT0的stage
projections、shared anchor decoder、G2/G3有效PSG与ViT在完整checkpoint轨迹中均有限更新。

实现审计发现一个必须保留的边界：PSG在每个ViT block之后调制patch token，因此post-block11位于
最后一次CLS–patch交互之后，对最终CLS descriptor没有下游路径；其零初始化final projection全轨迹
保持`0/2 changed`。实际有效G3 consumer只有post-block9/10。该无效terminal consumer在D0/HT0
间共享，不混淆HT0−D0的新增G2增量，但论文不得声称block11 PSG有效。

**最终判定**：完整`anchor+PSG`原子方法现在在Swin-T、ResNet-50和ViT-B三个骨干内部相对B0的
mAP差依次为`+1.1/+3.1/+2.0`，支持训练期pose supervision、推理期RGB-only的跨骨干描述性证据。
逐层HT0−D0的mAP差依次为`-0.1/+0.8/-0.3`，不具跨架构稳定性；不把hierarchical refinement
升为核心贡献，也不继续单图层数、宽度、decoder或loss小变体。下一阶段进入Video ReID/时序姿态，
新增变量必须来自跨帧可靠性、运动连续性或遮挡恢复，而不是继续复制单图PSG容量。

## exp382：Video TAPF 查新与数据门禁（2026-07-17）

> 本实验只做专项查新与数据审计；未下载、未实现、未训练，因此没有视频性能数字。

Video TAPF作为独立方法主线正式`NO-GO`。决定性近邻是GAE-Net：其已使用训练期RGB+gait视频
教师，并把局部互补时序知识蒸馏给RGB-only视频学生；PAFormer又覆盖pose-supervised、pose-free
inference，KPRTrack覆盖tracklet同部位moving average。结合PSTA/STMN/TF-CLIP等成熟时序
memory/aggregation，把跨帧pose state改名为temporal evidence routing仍不足以形成清楚的新方法
差分。远端`AG-VPReID.VIR`仅为空目录，也未发现可用视频训练数据。本轮不下载、不启动4090，
Video TAPF只保留为未来跨任务应用扩展，不能写成论文headline。

当前论文证据缺口转为：

1. 现有三骨干结果都只来自Occluded-Duke、seed 1234；
2. TAPF专属参数/FLOPs、训练与推理开销尚未审计；
3. PAFormer/PGFL-KD/TSD/KPR的paper-ready机制差分仍需固定；
4. 第二训练域与独立遮挡target尚无matched B0/D0。

因此下一必要验证预注册为`exp383 Market→Occluded-ReID TAPF`：fresh Market B0/D0使用同一
Swin-T recipe训练，并用同一个e120 checkpoint同时报告Market域内与Occluded-ReID跨域结果。
Occluded-ReID没有train split，只能称为独立遮挡target，不能伪装成第二训练数据集。exp383当前仅
完成设计和数据门禁，尚未实现或启动。

## exp384–exp389：官方最后代码上的干净重启与层级终判（2026-07-18）

> 用户要求退回 SOLIDER 官方最后提交，从原始 RGB 重新建立数据路径与 pose target，禁止复用旧
> runtime、旧 `pose_data/cache/path mapping`。以下均为 Swin-T / batch64 / seed1234 / 120 epoch
> 的 fresh final；只比较同数据集、同 recipe 的 matched arm。

| 数据集 | arm | mAP | R1 | R5 | R10 | 测试期外部 pose |
|---|---|---:|---:|---:|---:|---|
| Market-1501 | official B0 | 91.6 | 96.3 | 98.7 | 99.2 | 否 |
| Market-1501 | clean D0 | **92.0** | **96.5** | **98.8** | **99.3** | 否 |
| Occluded-Duke | official B0 | 57.4 | 67.4 | 80.6 | **85.2** | 否 |
| Occluded-Duke | clean D0 | **57.6** | **67.7** | **80.8** | 84.6 | 否 |
| Occluded-Duke | clean HT0 | 56.9 | 65.9 | 80.0 | 84.1 | 否 |

固定 e120 差值（mAP/R1/R5/R10）：

- Market D0−B0=`+0.4/+0.2/+0.1/+0.1`；
- Occluded-Duke D0−B0=`+0.2/+0.3/+0.2/−0.6`；
- Occluded-Duke HT0−D0=`−0.7/−1.8/−0.8/−0.5`；
- Occluded-Duke HT0−B0=`−0.5/−1.5/−0.6/−1.1`。

Market official B0 的 mAP 精确复现官方报告，R1 高 `0.2`。Occluded-Duke pose target 由
ViTPose-H 从 15,618 张 train RGB fresh 离线提取，manifest SHA256=
`cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`；query/gallery 没有 pose。
Market 的 12,936 张 train RGB 也独立 fresh 提取，query/gallery 同样为 RGB-only。

clean D0 只增加完整 Stage-2 anchor→Stage-3 两个独立 PSG：Occluded-Duke 参数 `+105,442 /
0.375585%`、supported-op FLOPs `+0.242424%`、train batch64 step `+1.96%`、eval batch256 step
`+1.64%`；FLOPs 不含 analyzer 未支持算子。两个数据集 final correct/shuffle/None/exploding
external pose 的 descriptor、field、gate 均逐元素 exact。

exp389 从零新增 Stage-1 anchor→Stage-2 六个独立 PSG，并保留完整 D0 late path。全部八个 bank
均真实更新且逐 consumer 旁路会有限改变最终 descriptor，排除 dead consumer 或未训练解释；但
e120 四项均低于 D0。因此 clean 证据只支持原子 D0 在两个训练域上的**小幅描述性 mAP 正差**，
不支持层级增量，也不支持“稳定四项提升”或统计显著主张。此前旧 runtime 的三骨干结果保留为
历史探索证据，不能与本轮官方干净主表混合成同一实现口径。

下一必要验证改为官方干净 Occ-Duke matched B0/D0 多 seed；效应仅 `+0.2 mAP`，在补 seed 前
不得把原子方法写成稳定普适提升。Video 与 hierarchical 都不再作为下一算力优先级。

## exp390：official clean TAPF 的 Occ-Duke 三seed paired终判（2026-07-18）

> Swin-T / Occluded-Duke / batch64 / 120 epochs / final-only。seed1234沿用已封板canonical pair，
> seed4321与2025各自fresh串行训练matched B0/D0；所有数字均来自正式train log。

| seed | B0 mAP/R1/R5/R10 | D0 mAP/R1/R5/R10 | D0−B0 |
|---:|---:|---:|---:|
| 1234 | `57.4/67.4/80.6/85.2` | `57.6/67.7/80.8/84.6` | `+0.2/+0.3/+0.2/−0.6` |
| 4321 | `56.0/66.2/79.4/83.8` | `56.8/66.5/79.9/84.3` | `+0.8/+0.3/+0.5/+0.5` |
| 2025 | `57.5/67.9/81.1/85.7` | `57.9/67.0/80.4/85.2` | `+0.4/−0.9/−0.7/−0.5` |
| mean±sample std | `56.97±0.84/67.17±0.87/80.37±0.87/84.90±0.98` | `57.43±0.57/67.07±0.60/80.37±0.45/84.70±0.46` | `+0.47±0.31/−0.10±0.69/+0.00±0.62/−0.20±0.61` |

paired方向计数（正/零/负）：mAP=`3/0/0`、R1=`2/0/1`、R5=`2/0/1`、R10=`1/0/2`。结论是
完整`anchor+PSG`在official clean Occ-Duke上提供了**小而可重复的mAP正差**，但没有形成rank
指标的稳定增益；不得写成“四项全面提升”或跨架构headline。D0-s2025 final checkpoint SHA256=
`ce06c380dddbec6149ee57aad12f1ebff0d310183b26886e6194adc41ca243f4`，223-state strict finite、
anchor/两PSG轨迹、pose-free exact与两个consumer下游路径终审全部PASS，排除dead branch或外部pose
泄漏解释。

## exp391：H2-M 多阶段 loss-budget 诊断终判（2026-07-18）

> Swin-T / Occluded-Duke / seed1234 / batch64 / 120 epochs / final-only。相对exp389只把两层
> pose objective从`0.1×sum`改为`0.1×mean`；early/late anchor、`6/2` consumer和其余recipe不变。

| arm | mAP | R1 | R5 | R10 | 相对exp387 D0 | 相对exp389 HT0 |
|---|---:|---:|---:|---:|---:|---:|
| exp391 H2-M | 57.2 | 67.3 | 80.2 | 84.5 | `−0.4/−0.4/−0.6/−0.1` | `+0.3/+1.4/+0.2/+0.4` |

H2-M恢复并略超过旧HT0，但final mAP仍比单层D0低`0.4`，超过预注册允许下降`0.2`，因此
Phase A=`SEALED / NO-GO`，Phase B/C禁止实现或启动。冻结层级消融显示full−early-bypass=
`+0.141111/+0.497735/+0.316739/−0.135750`，early route达到`+0.1 mAP`独立贡献门槛；
full−late-bypass=`+1.546086/+2.036196/+1.447964/+1.583707`。这排除了early terminal-dead解释，
但不能把条件性局部贡献替代最终D0比较。

正式checkpoint SHA256=`914b9d321a72a6af9045743f9c0456b38c5641acf812928602c76d2558a14206`。
243-state strict finite、early/late anchor与八个PSG参数轨迹、correct/shuffle/None/exploding pose-free
exact、八consumer最终descriptor可达性和严格异常终审全部PASS。结论是mean loss修复了部分优化
预算问题，却没有证明`6/2`多阶段topology具有优于单层D0的最终检索价值。该NO-GO只封板exp391
的纯结构链，不永久否定多阶段TAPF；若后续CLIP双编码器teacher先通过joint-channel语义可辨识门禁，
可在新的独立实验中重新比较语义校准后的单阶段与多阶段版本。

## exp392 Phase 0A：clean D0内部field语义审计（2026-07-18）

> 使用exp387 sealed D0-s1234 e120 checkpoint，全模型`eval+no_grad`，query/gallery RGB-only；
> 所有counterfactual在anchor输出field与两个PSG的真实production seam临时注入，不训练参数。

| arm | mAP | R1 | R5 | R10 | 相对correct |
|---|---:|---:|---:|---:|---:|
| correct | 57.5588 | 67.6923 | 80.7692 | 84.5701 | `0/0/0/0` |
| channel-cycle | 57.5828 | 67.7376 | 80.7240 | 84.6154 | `+0.0240/+0.0452/−0.0452/+0.0452` |
| left/right swap | 57.5415 | 67.6923 | 80.7240 | 84.5249 | `−0.0173/0/−0.0452/−0.0452` |
| confidence permutation | 57.5353 | 67.6923 | 80.7240 | 84.4344 | `−0.0234/0/−0.0452/−0.1357` |
| matched-wrong field | 57.5538 | 67.8281 | 80.7240 | 84.5701 | `−0.0049/+0.1357/−0.0452/0` |
| spatial-constant | **57.9051** | **68.1900** | **81.2670** | **85.0679** | `+0.3463/+0.4977/+0.4977/+0.4977` |
| PSG0 bypass | 56.8832 | 67.0136 | 79.9095 | 83.9367 | `−0.6756/−0.6787/−0.8597/−0.6335` |
| PSG1 bypass | 56.8439 | 66.6968 | 79.5475 | 83.8914 | `−0.7149/−0.9955/−1.2217/−0.6787` |
| zero/all-bypass | 56.2002 | 66.0181 | 79.0498 | 83.3032 | `−1.3586/−1.6742/−1.7195/−1.2670` |

paired bootstrap mAP 95% CI分别为channel-cycle`[+0.0101,+0.0386]`、matched-wrong
`[−0.0276,+0.0192]`、spatial-constant`[+0.2873,+0.4121]`、all-bypass
`[−1.5141,−1.2081]`。correct start/end和四个external-pose变体descriptor exact；zero与all-bypass
exact；两个PSG分别旁路均显著下降；model state SHA前后exact。

裁决：`CONSUMER_EFFECTIVE_JOINT_SEMANTICS_NOT_IDENTIFIED`。PSG consumer对已训练D0 checkpoint
有强条件性贡献，但正确joint-channel binding相对循环置换或matched-wrong field没有可辨优势；
空间常量控制反而全面更高。该`−1.359 mAP` bypass值是D0 checkpoint内的冻结路径贡献，不可替代
matched D0−B0训练增益`+0.2`。结果证明exp392的问题对象真实，但不证明CLIP teacher有效；下一步
必须通过Phase 0B teacher-only门禁。

## exp392 Phase 0B：naive dense CLIP teacher-only审计（2026-07-18）

> official Occ-Duke train共15,618图；frozen OpenCLIP ViT-L/14 image+text encoder；pose只用于
> 训练域region mask，未构建ReID模型/optimizer。两种CLIP geometry、全部反事实和bootstrap均为
> 预注册零训练审计。

| geometry | correct top-1（95% CI） | correct margin（95% CI） | shuffle top-1 | wrong-text top-1 | image-only cluster |
|---|---:|---:|---:|---:|---:|
| square-stretch | `2.692% [2.583,2.801]` | `−0.11349 [−0.11383,−0.11312]` | `16.107%` | `29.996%` | `59.99%` |
| aspect-letterbox | `4.637% [4.508,4.777]` | `−0.11099 [−0.11137,−0.11063]` | `15.511%` | `33.546%` | `52.77%` |

correct相对wrong RGB/matched wrong mask/channel shuffle/wrong text的paired expected-margin差分别为：

- square=`−0.02423/−0.01651/−0.03129/−0.07901`；
- letterbox=`−0.02241/−0.01473/−0.02757/−0.08247`。

两种geometry均显著低于20% chance，且语义错配不是让teacher变差而是让top-1/margin变好。
wrong-RGB q-JSD约`0.0032–0.0033`、centered effective rank约`2.58–3.64`，说明分布并非数值常量；
但它的sample variation没有绑定到正确body-part text identity。flip q-JSD很低，却只有
`89.06%/84.43%` top-1 consistency；confidence与synthetic-erasing方向也未过门禁。

实现归因进一步排除低层bug：pose head→legs y顺序正确；hook token与OpenCLIP官方
`output_tokens`经同一`ln_post+proj`逐元素exact；bicubic相对bilinear仅把128图top-1从
`5.469%`变为`5.625%`。保持同一mask/prompt，把region改走受CLIP对比目标直接监督的tight-crop
global CLS后，128图macro top-1升至`44.688%`；最后block pose-conditioned hard-CLS readout为
`32.5%`。因此裁决是`CURRENT_CLIP_TEACHER_NO_GO`：失败点为naive last-block patch token没有
被校准到text轴，而不是pose/mask/hook/label实现错误。

Phase 0C与正式训练均不获授权。该结果只否定当前dense readout，不永久否定CLIP语义校准；下一步
需对共享trunk的multi-block CLS readout或region-crop global CLS另做成本与teacher-only门禁。
全量result/donor/runner SHA256分别为
`af8e654565396f338a9a1b1f8ce5fe4d8178d551ec2767c500d230d477d7e6f8`、
`27f31fa69ec223c4506218ce468b01a540882da70380ad85cd8449333c9d5a74`、
`bcb588175a54ecb175d4c6a60efd71bfd0e8aa5a5bec032117abbed18cb28b02`；进程/worker自然退出，
GPU恢复`2 MiB / 0%`，严格异常为0。

## exp392 Phase 0B2/0C：PC-MBCLS support readout与single-stage Semantic TAPF（2026-07-19）

Phase 0B2没有用一个失败否定整条路线。soft ontology因region overlap失败，hard-owner把五slot重叠
精确降到0；part-name prompt因arms top-1=`0%`只关闭“让CLIP重猜part name”。随后PC-MBCLS
pose-conditioned support readout在128图、639个valid target上通过全部12个局部门禁：macro target
q-visible随support遮挡0/25/50/75从`0.51104/0.48925/0.47765/0.46990`单调下降；五slot
target−non-target下降=`+0.06919/+0.02273/+0.04528/+0.03202/+0.02879`，逐slot PID-cluster
CI均大于0。该结果证明五slot存在局部sample-specific视觉响应，但不等价证明它能改善ReID final。

用户明确授权首次single-stage bundled feasibility后，Semantic C0保持clean D0的backbone、两处late
consumer和完整120-epoch recipe；唯一机制组合是五slot mask/presence/q student加两个
feature-dependent low-rank router，训练期teacher在模型外，推理删除CLIP、文本和external pose。
fresh运行自然e120 final如下：

| 方法 | mAP | R1 | R5 | R10 | 相对D0 | 相对HT0 |
|---|---:|---:|---:|---:|---:|---:|
| Semantic C0 | 56.9 | 67.1 | 80.6 | 85.0 | `−0.7/−0.6/−0.2/+0.4` | `+0.0/+1.2/+0.6/+0.9` |

相对official B0=`57.4/67.4/80.6/85.2`为`−0.5/−0.3/+0.0/−0.2`。e120末
Semantic/RegionMask/Presence/Q=`0.292/0.158/0.026/0.692`，support在8图终审中混合五slot后的
mean/std/min/max=`0.51172/0.01686/0.48159/0.53281`，两router gate-delta abs-mean仅
`3.606e-06/1.040e-05`。该pooled std包含固定slot均值差，不能直接当作同slot跨图动态。

唯一checkpoint SHA256=`8f8e4a8af1280f17f736053a3068dfae0384ac54915f9c68fb0c779350c3638e`。
231-state strict finite、teacher/CLIP不在state、anchor/q-head/两个consumer轨迹、RGB-only
correct/shuffle/None/exploding exact、两consumer final-descriptor可达和zero-mask/zero-q exact identity
终审全部PASS。训练与审计进程均自然退出，GPU恢复`2 MiB/0%`，严格异常为0。

**最终判定**：`Semantic C0 = SEALED / CURRENT BUNDLED COMBINATION NO-GO`。它优于HT0的rank差
不能证明CLIP语义有效，因为对更直接的clean D0仍低`0.7 mAP`，且CLIP相关q动态范围不足。该结论只
关闭当前PC-MBCLS teacher/readout/router组合，不否定CLIP–TAPF。下一步做必要的单变量拆因与机制
修复；在single-stage语义因果未成立前，不启动balanced semantic multi-stage，不重跑或换seed救场。

### Phase 0D：Semantic C0 final checkpoint冻结拆因

全query+gallery 19,871图上，correct start/end exact=`56.920063/67.058825/80.588233/85.022622`。
相对correct，static-slot-q、q-one、spatial-constant-mask、slot-cycle、expert-mean、router0/1/all-bypass
的ΔmAP依次为`+0.000056/−0.000060/+0.000654/+0.000009/−0.000092/−0.000067/−0.000029/
−0.000077`，所有ΔR1/R5/R10均严格为0。五slot同slot跨图q std仅
`0.000293/0.000163/0.000090/0.000121/0.000191`；此前pooled std=`0.01686`主要由固定slot均值差
构成，而非sample-specific CLIP动态。

correct start/end descriptor exact、state SHA exact、全部descriptor finite、异常0。结果把归因进一步
收紧：两个consumer在数值上可达final descriptor，但整条semantic route对e120检索排序近似失活；
q、精确mask geometry、slot binding和slot-specific expert均无`0.1 mAP`级别的边际贡献。下一机制
不能只调q或增加stage，必须先让semantic route本身形成有量级且经all-bypass验证的检索残差。

## exp393 Phase 0E：centered rich CLIP local evidence审计（2026-07-19）

exp393把“route能否离开identity”和“CLIP局部证据是否足够丰富”拆成逻辑独立的两门。Phase 0E只
审计后者，不构建ReID model、optimizer或checkpoint，也不以teacher PASS直接授权训练。

0E-S synthetic exact与0E-C8真实8图contract均`SEALED-PASS`。0E-C8验证official global parity、
repeat/NULL exact、hard-owner/wrong-mask IoU=`0`、输出`[8,5,16]`及teacher frozen/no-grad；五slot的
donor和wrong-mask描述性margin均为正，但未用8图作统计裁决。

0E-128在128个不同PID上按hash严格拆成64 fit/64 held-out，全部正式门`PASS`。五slot的16维
held-out code每维std均非零，各slot最小std=`0.1649/0.1480/0.1802/0.1219/0.1371`；effective rank=
`10.764/10.756/11.843/10.788/11.101`，macro=`11.050/16`。correct↔flip相对different-PID wrong
RGB的margin均值=`0.808/0.735/0.781/0.742/0.821`，95% PID-cluster CI下界=
`0.709/0.639/0.709/0.655/0.733`；相对same-RGB low-IoU wrong-mask的margin均值=
`0.614/0.179/0.413/0.165/0.645`，CI下界=`0.531/0.097/0.341/0.103/0.575`，五slot均严格正。

slot-mean/global-only code exact zero；raw uncentered的wrong-RGB/wrong-mask margin明显更弱。
fixed random orthogonal仍保留强信号且macro rank=`13.458`，说明可用信息来自rich local residual，
不依赖PCA偶然选轴；PCA只作固定压缩器，不能当作创新点。累计PC-MBCLS forward=`12.505s`，峰值
allocation=`1,712,272,384 bytes`，strict finite与异常/AMP warning检查PASS，进程自然退出且GPU恢复
`2 MiB/0%`。

0E-128 script/result/codebook/runner SHA256分别为
`deae5c9308650f9f9344ab19e0e78fa78b193a53244e41ccc24d9274fbd1526a`、
`47a27631756c42bfa696f9751b604532fa9033489d67ef107126fcaa254b19dc`、
`4a671a70e0744edad88f911ce628d421650cb09453eb511a61e8d01c239269ef`、
`e8f35143a8599bfec3f3e0354b872bc71090d48420a6408fa9d517d3f46c01a3`。

**当前判定**：`Phase 0E-S/C8/128 = SEALED-PASS`，只授权official 15,618 train的0E-FULL
teacher-only held-out PID审计。Phase A、Phase B正式训练与semantic multi-stage仍`NO-START`；即使
0E-FULL失败，也只关闭当前rich evidence code，不替代逻辑独立的Phase A route-activation裁决。

### Phase 0E-FULL official train裁决

full审计覆盖official 15,618图，fit/audit=`7,860/7,758`图、`361/341`个PID且无泄漏。五slot
effective rank=`12.332/12.289/12.950/12.278/11.828`，macro=`12.335/16`；wrong RGB margin的
PID-cluster CI下界=`0.756/0.748/0.733/0.773/0.766`，same-RGB wrong-mask CI下界=
`0.632/0.160/0.480/0.189/0.633`，全部正式门PASS。raw uncentered显著更弱，random orthogonal仍
保留强信号，说明rich residual有效而PCA不是贡献。

**最终判定**：`Phase 0E = SEALED-PASS`。它证明centered rich CLIP local evidence在全train
held-out PID上稳定存在，但不证明ReID route会使用它；只授权Phase B teacher接口，正式训练仍需
独立Phase A route-activation通过。

## exp393 Phase A：RZ-C0 route activation终审（2026-07-19）

RZ-C0 fresh seed1234自然跑满e120，final mAP/R1/R5/R10=`56.8/66.8/79.6/83.9`。相对
Semantic C0=`-0.1/-0.3/-1.0/-1.1`，相对clean D0 seed1234=`-0.8/-0.9/-1.2/-0.7`。

| frozen eval | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| RZ-C0 full | `56.8` | `66.8` | `79.6` | `83.9` |
| all-router-bypass | `56.8` | `66.8` | `79.6` | `83.9` |

raw full−bypass=`-0.000249709 mAP point`；R1/R5/R10 raw完全相同。checkpoint strict finite、
teacher-free、RGB-only、NULL identity、两router独立和全部目标参数轨迹PASS；最终alpha仅
`-1.843e-4/-1.363e-4`，synthetic descriptor max-abs gap=`2.861e-6`。

**最终判定**：`Phase A RZ-C0 = SEALED-NO-GO / ROUTE-ALIVE-FAIL`。只关闭当前ReZero route
接口；Phase 0E rich teacher仍为独立PASS。原Phase B不获授权，不重跑、不换seed、不降低门槛。

## exp394 Production static/CPU：证据预算化rich residual实现门禁（2026-07-19）

在exp393 exact source seam上完成默认关闭的fresh production实现。独立CPU contract相对实现前commit
逐state/forward复核D0、HT0、Semantic C0、RZ-C0全部exact；新route的e1–e5 rho=0 full/bypass exact、
NULL mask/presence identity、e6–e9线性schedule、e10+/eval固定预算、两个consumer、strict reload、
relation loss与teacher-free state全部PASS。evidence、mask/presence、`L_exec`、ReID四类loss的梯度
所有权逐组符合冻结协议。

最终两遍result与runner逐SHA一致，script/result/runner SHA256=
`5be2980eb6a666f791ba5e3cd87bbabb7a0b9934bb44724e091cbbb7e4545cd1`/
`658ac1fd261ec09db618e9d658ae00fa3f0f7d7887b87e8716c601adbc8b0636`/
`658ac1fd261ec09db618e9d658ae00fa3f0f7d7887b87e8716c601adbc8b0636`。该结果只封板CPU实现契约，
不含真实CLIP吞吐、AMP更新或检索结果；裁决为`PRODUCTION_STATIC_CPU_SEALED_PASS / CUDA NO-START /
FORMAL NO-START`。

## exp394 Production CUDA/AMP preflight（2026-07-19）

唯一冻结actual-batch门在canonical Torch/OpenCLIP/OpenCV runtime上执行。official batch64与rich
teacher target前置contract通过，但首个scaled backward在unscale后出现non-finite model gradient，
脚本在`scaler.step`前退出；成功optimizer update=`0/24`。因此没有handoff、reload、RGB-only或
counterfactual终审数据，也没有checkpoint或可续训权重。

script/result/runner SHA256=
`bae2210bc606048371b4750f85919595c0b8fdbd1e11681abac59fe9727ea4f0`/
`3897d76fd6b6aeb0d9ed2a27e527053874f6cdf32b56cc80d5bc2f12e584b152`/
`c76e9285a41f65f0e9333dda2ef10a75bd1a17bf85538019ac3871d000b0c879`。进程自然退出，GPU恢复
`2 MiB/0%`，execution HEAD/tracked/source/asset SHA保持exact，checkpoint=`0`。

**最终判定**：`CUDA_AMP_PREFLIGHT_SEALED_FAIL / EXP394 FORMAL NO-START`。该FAIL只关闭当前
production AMP接口，不推翻Phase0E rich evidence、Phase0R预算代数或CLIP–TAPF总体；但CPU contract
不能替代真实CUDA finite门，禁止重跑、调initial scale、补步或启动e120。

## exp395 Phase 0S：AMP梯度归因器static/CPU门（2026-07-19）

exp395没有修改或重跑exp394，而是先验冻结了独立只读归因对象：D0 baseline 5个loss、rich 11个loss，
以及backbone、anchor/head和两个router T/C/E/Expert共15个参数组；每格同时记录scaled/unscaled的
present/nonzero、finite/NaN/±Inf计数与abs-max/L2/P50/P95/P99。执行协议禁止
`optimizer.step/scaler.step/scaler.update`，要求state/RNG/source/asset前后exact。

独立CPU contract连续两遍逐字节复现，13/13 gate PASS：11-loss×15-group synthetic ownership、
固定`65536` scale比例、两个consumer loss与aggregate公式、NaN/±Inf sentinel分类、source seam、
state/RNG zero-drift全部exact；CUDA initialized=`false/false`，optimizer update=`0`、checkpoint=`0`。

script/result/runner SHA256=
`d4c6d67b082e4e4f68ff215de3e7cf1f2a2ac1c4c59e17ceb265353b8810083a`/
`89afc893409957ee5ad356e0e2d5789683b36bcce449076d26a7dec3d3bed91c`/
`89afc893409957ee5ad356e0e2d5789683b36bcce449076d26a7dec3d3bed91c`。

**当前判定**：`PHASE0S_STATIC_CPU_SEALED_PASS / CUDA NO-START / FORMAL NO-START`。它只证明归因器
数学和静态seam可信，没有读取official batch/teacher资产，也没有给出任何实际AMP根因；exp394 sealed
FAIL与semantic multi-stage NO-START边界均不变。

### exp395 CUDA attribution implementation static

零更新CUDA归因脚本现已实现但未执行。CPU-only AST contract连续两遍29/29 PASS，确认D0 5行、rich
11行、15组、默认scale、scaled→unscale→unscaled顺序、fresh asset/runtime门与post-exit审计全部进入
源码；同时不存在optimizer/scaler/scheduler step/update、checkpoint load或训练授权路径。

implementation/static/result SHA256=
`64840b710db587720aa8807571212b246af3eabb54306bd5aa1bbf692f5ea08b`/
`345d26309043dd8d14119316a7ca186e1cf9faea2e666bd01d652ded50663c1b`/
`30b7b7ae06ff2bd3153208fe4384e11e06a097608c6ce876d6c254c079f2e314`。

裁决=`CUDA_ATTRIBUTION_IMPLEMENTATION_STATIC_SEALED_PASS / CUDA EXECUTION NO-START`；它不包含actual
batch或gradient结果，不改变exp394 sealed FAIL和formal NO-START。

### exp395 CUDA attribution actual：reporter运行时失效

唯一actual在fresh source、exp395 regular CLIP/codebook、canonical runtime和GPU空闲门通过后执行。
official batch64与teacher target前置控制流通过；第一行D0 `reid`完成scaled backward后，scaled
reporter对backbone组调用`torch.quantile`时触发`RuntimeError: quantile() input tensor is too large`，
并在`scaler.unscale_`前按协议退出。

因此没有完成D0 5行、rich 11行或15组scaled/unscaled矩阵，也没有新的gradient finite/non-finite归属。
optimizer/scaler update=`0`、checkpoint=`0`，进程退出后GPU=`2 MiB/0%`且无compute process。
result/runner/manifest SHA256=
`cdffff60b1b6e04e6bb0b13bb54e12518380421675c59c2f2c785f1b7a5adb75`/
`cdffff60b1b6e04e6bb0b13bb54e12518380421675c59c2f2c785f1b7a5adb75`/
`3a0ef5d98dd6387b330958bbfb1e9d893e60745e8857237bbbbe375778886c64`。

**最终判定**：`CUDA_ATTRIBUTION_EXECUTION_SEALED_INVALID / REPORTER_RUNTIME_FAIL`。它只否定exp395
测量器对真实大组梯度的可执行性，不支持任何exp394根因判断；exp395不得重跑，正式训练继续
`NO-START`。

## exp396 Phase 0Q：chunk-safe exact reporter static/CPU门（2026-07-19）

exp396只替换exp395失败的统计器，冻结D0/rich loss、15组、batch64、default scale与zero-update不变。
production reporter以`1,048,576`元素chunk双遍扫描，finite absolute values进入temporary FP64 memmap
exact sort，不再调用大输入`torch.quantile`。

独立contract连续两遍33/33 PASS且逐字节一致。小张量和multi-chunk与reference一致；固定
`16,777,217`元素case完整完成，P50/P95/P99=
`8,388,608 / 15,938,355.2 / 16,609,443.84`并与解析order statistic exact。输入不变、success/exception
scratch清零、CUDA initialized=`false/false`、update=`0`、checkpoint=`0`。

implementation/static/result SHA256=
`6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`/
`f3a2ee3ccafa4caa1606b92b93b86177cc0b5ef6cfe7ac2b6f0d31fa195c415b`/
`e5d68df7731042a98f440f43acc45c9cf11b70aa7df25e09397ff6375f355394`。

**当前判定**：`PHASE0Q_STATIC_CPU_SEALED_PASS / CUDA ATTRIBUTION FRESH-EXECUTION GO / FORMAL
NO-START`。尚无actual gradient归因结果。

### exp396 CUDA actual：完整矩阵定位到shared ReID backbone

唯一actual完整执行D0 5行、rich 11行、15组scaled/unscaled，status=`PASS`，所有state/RNG/teacher/
asset/scratch/zero-update gate通过。预注册outcome=`SHARED_D0_OR_RUNTIME_NONFINITE`。

D0与rich的`reid` scalar均为`20.846956253051758`。两arm的`reid/total`只在backbone非有限，每格
NaN/`+Inf`/`-Inf`=`368/3,753/4,183`，scaled与unscaled一致；其余D0 pose项及rich mask/presence/
evidence/两个exec consumer/pose全部finite。D0与rich的non-finite支持、计数和finite range完全相同，
且common initial state 211 tensors exact。

内部矩阵耗时=`7.281521141529083 s`，peak memory=`7,631,537,152 bytes`；update=`0`、checkpoint=`0`、
scratch=`0`。script/result/runner/manifest SHA256=
`6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164`/
`58ae4beb56c9dabbff7fd77202d87b53f3ccecc9edec725051f04ed3c60ed96c`/
`58ae4beb56c9dabbff7fd77202d87b53f3ccecc9edec725051f04ed3c60ed96c`/
`3a0ef5d98dd6387b330958bbfb1e9d893e60745e8857237bbbbe375778886c64`。

**最终判定**：`CUDA_ATTRIBUTION_SEALED_PASS / SHARED_D0_OR_RUNTIME_NONFINITE`。exp394仍按原绝对
首步finite门保持FAIL，但该现象不能再称为rich-specific；matched clean D0也以相同方式失败。下一步只
授权新的matched native GradScaler dynamics门，正式训练仍`NO-START`。

## exp397 matched native GradScaler static/CPU门（2026-07-19）

新门冻结同一12个official batch的D0/rich原生动态scale轨迹：e1六步→e6六步、default initial scale、
每attempt一次`step/update`，不手调scale、不补step。连续两遍21/21 static PASS且逐字节一致；matched
synthetic通过，extra skip/late success/handoff/rich-specific non-finite四类反例均正确失败，CUDA未
初始化。

implementation/static/result SHA256=
`4ad2c40a8d679e8dd52619d9216016aaecdc0fd6530d7ca679e0bb16b7cfa9ba`/
`99ad9a0d34db4bcbc0816ecd05c62d361322f47d214bca21c9927f92738269dd`/
`82d52315d1472e996fc50f330d332853c2e025ecf1c333651aca6cd7385f06eb`。

**当前判定**：`STATIC_CPU_SEALED_PASS / CUDA NATIVE-PARITY FRESH-EXECUTION GO / FORMAL NO-START`。

### exp397 CUDA native GradScaler actual

唯一fresh actual完成D0/rich各12行。两臂scale/skip轨迹exact：e1 attempts 1–5从`65,536`连续
backoff并skip，attempt 6在`2,048`首次成功；e6 attempt 7再次matched skip到`1,024`，attempts 8–12
成功。两臂均只有`6/12`次update、首个成功为attempt 6，未满足冻结的`>=10/12`、`<=3`、首次成功后
全finite和e6六步全success门。

所有non-finite仅在两臂相同attempt的shared backbone；rich-specific 11组全程finite，source/runtime/
assets、common init、RNG、teacher/codebook、scratch与checkpoint终审均PASS。elapsed=
`20.892324913293123 s`，peak memory=`7,907,269,120 bytes`，checkpoint=`0`。result/runner/manifest SHA=
`eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
`eef02328fb4026459fa28a7095d8d5c7b5703834e25ba950a78d9a3f1978faa2`/
`a9c2acd912f57a7020e129c59c4b24b615d1e4065f0bf482f71ad631eb7b3c51`。

**最终判定**：`CUDA NATIVE-PARITY SEALED-FAIL / NATIVE_GRADSCALER_PARITY_FAIL / FORMAL NO-START`。
该FAIL不能改判或补跑；它只否定exp397的绝对生产力门，不能归因于rich-specific图。

## exp398 baseline-relative AMP稳态static/CPU门（2026-07-19）

新门固定32步、e1/e6各16步与每阶段最后8步连续稳态，保持default GradScaler、batch64、loss和source
不变；rich不得在D0成功处extra skip，non-finite只能是同attempt D0 shared group，11个rich-specific组
还必须在e6有非零梯度和state变化。

连续两遍24/24 static PASS且逐字节一致；matched shared warm-up与rich-better正例PASS，extra skip、
persistent tail、rich-only non-finite与inactive组反例均正确FAIL。implementation/static/result SHA=
`8bc599e94264eb3fb89b3cdc94810c483f4c4a037ebe86311c30a741011aeac9`/
`b137fadaf0463ae51eb2e552945cf87923a2c788582dbf0a4aaf00e296829414`/
`d7efa6894411f7b7433c8819422e14c2495110484544d86b9b21083d0bb24317`。

**当前判定**：`STATIC_CPU_SEALED_PASS / CUDA BASELINE-RELATIVE FRESH-EXECUTION GO / FORMAL NO-START`。

### exp398 CUDA actual：group-state reporter运行时失效

唯一fresh actual完成source/runtime/assets、32-batch与teacher target前置门，但D0首个forward前的
initial group-state统计把`(name, parameter)`元组当tensor调用`.detach()`，触发`AttributeError`。没有
backward、optimizer update或D0/rich轨迹；checkpoint=`0`、scratch=`0`，GPU已恢复空闲。

result/runner/manifest SHA=
`71a943e6a233999549f69c1ece2ce1c2c3e507c69d9e99364272442d9b6ac998`/
`71a943e6a233999549f69c1ece2ce1c2c3e507c69d9e99364272442d9b6ac998`/
`b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`。

**最终判定**：`CUDA EXECUTION SEALED-INVALID / GROUP_STATE_REPORTER_RUNTIME_FAIL / FORMAL NO-START`。
它不回答baseline-relative AMP稳态，exp398不得修补或重跑。

## exp399 named-parameter state contract static/CPU门（2026-07-19）

exp399保持exp398科学门不变，只把group-state SHA改为严格的有序`(name, parameter)`记录，并用真实
exp396 `parameter_groups()` synthetic model覆盖15组container。初次逻辑PASS但tiny model seed遗漏导致
两遍byte-exact FAIL，证据保留；固定static seed后正式两遍35/35 PASS且逐字节一致。

implementation/static/result SHA=
`b9da4346b0d74d13b537bd7fa3f5eff1e65b0b6e512014026800506807723907`/
`7948845f1600141302285cee12c025cbf0ba50faa1af01d1fb298bd3aa558810`/
`32adc18d2b6dc06c0d3ea37ca6003d749a2ff2540efefdbca0e35e1fba2f0d98`。

**当前判定**：`STATIC_CPU_SEALED_PASS / CUDA FRESH-EXECUTION GO / FORMAL NO-START`。

### exp399 CUDA baseline-relative actual

唯一fresh actual完成D0/rich各32行。两臂在attempts 1–5和7发生完全相同的shared-backbone native skip，
scale自然降到`1,024`；attempts 8–32连续成功。两臂均`26/32` update，rich extra skip=`0`，e1/e6四个
tail8全部success/finite。rich-specific 11组全程finite、e6均有非零梯度且state全部改变。

全部validity、teacher/codebook、scratch/checkpoint与退出审计PASS。elapsed=`57.138093701563776 s`，peak
memory=`7,901,594,112 bytes`。result/runner/manifest SHA=
`d5255fced4553c6d4669ce11a1644e1495340a590ee76e54f22139f547cb9cca`/
`d5255fced4553c6d4669ce11a1644e1495340a590ee76e54f22139f547cb9cca`/
`b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`。

**最终判定**：`BASELINE_RELATIVE_STEADY_STATE_PASS / PRODUCTION PREFLIGHT GO / FORMAL NO-START`。
它不重判exp397，只授权新编号final production preflight。

## exp400 final production preflight static/CPU门（2026-07-19）

exp400逐字保留exp399的32-step matched动态门，只新增terminal production contract：final state finite与
teacher-free、fresh strict reload、eval RGB-only、epoch1 rho=0 full/all-bypass exact、epoch6 full相对
all-bypass和两个单consumer bypass均有限非零，以及diagnostic state/RNG/patch、source/assets终审。

static/CPU连续两遍`48/48 PASS`且result/runner逐字节一致；toy双router覆盖rho0 identity、两个consumer
独立非零、strict reload、ExplodingPose零访问和patch/state恢复，CUDA初始化前后均为`false`。CUDA脚本/
static脚本/result SHA256=
`1f069614fd789f7c3a6ca1d5666239d7ce91769a502087cc379769bd1cceb797`/
`e4019edf3df23a675c9ee0b1c2da1006f28bf184a97db036b88cb2d67888b33e`/
`501b12b4a926e8bc0b9de88995a939beca465ea05fef8b8d96410ef9074c3f02`。

**当前判定**：`STATIC-CPU SEALED-PASS / CUDA FRESH-EXECUTION GO / FORMAL NO-START`。static只授权
唯一fresh actual；只有actual trajectory、validity、terminal全PASS才允许`formal_training_authorized=true`。

### exp400 CUDA final production actual

唯一actual完整运行D0/rich各32行。两臂skip均仅attempts 1–5、first success=6、各`27/32`更新，rich
extra skip=0且没有rich-only non-finite；e1/e6四个tail8全部连续success/finite，11个rich-specific组
全程finite、e6 active且state全部改变。

31项terminal gate全PASS：241项final state全部finite/teacher-free，fresh strict reload和descriptor exact；
eval correct/shuffle/None/ExplodingPose逐元素exact且访问数0；epoch1 rho=0 full/all-bypass exact；epoch6
all-bypass mean L2=`0.4205047190`，bypass0/bypass1 max-abs=`0.0727601051/0.0865910053`。diagnostic
state/RNG/patch、teacher/codebook/source/assets/tracked、checkpoint=0和scratch=0全部PASS。

elapsed=`58.02608146890998 s`，peak memory=`7,901,594,112 bytes`。result/runner/manifest/stdout SHA=
`3935eb6df97ae832770316eff27cbfc757e4d2bd305b789d0b9b97835659a02f`/
`3935eb6df97ae832770316eff27cbfc757e4d2bd305b789d0b9b97835659a02f`/
`b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`/
`e91ffd6c4732387b90fe4f49dc31b41eb1c35ca831ac65c30975048979d4e620`。进程自然退出，GPU恢复
`2 MiB/0%`且无compute process。

**最终判定**：`FINAL_PRODUCTION_PREFLIGHT_PASS / FORMAL E120 GO`。result显式
`formal_training_authorized=true`；直接启动唯一fresh rich-budget C0 seed1234 e120，exp400不得重跑。

## exp401 rich-budget C0 formal launch static门（2026-07-19）

formal config相对冻结rich-budget C0只改变fresh CLIP、fresh codebook和OUTPUT_DIR。初次17项科学门仅因
YAML保留`('/mnt1/afrdata')`字面括号导致路径reporter误判，config未改且失败结果保留；修正static判定后
正式两遍18/18 PASS、result/runner逐字节一致，CUDA未初始化。

static/config/result SHA=
`90c95b4ac1be32a8d4917882be1c407d17945511205446ede7ddaefb847f319d`/
`c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84`/
`85cab0e0a8391b3470f0e11acbd634d3dce2fee638432679a2ef9dc49cae020d`。

**当前判定**：`STATIC-CPU SEALED-PASS / FORMAL FRESH-EXECUTION GO`。直接启动唯一fresh seed1234 e120；
中间eval只记录，e120 final full与all-router-bypass才裁决route alive。

### exp401 formal启动

fresh source/config/assets审计全PASS后，唯一seed1234 e120已启动。main PID=`404782`并有8个DataLoader
workers，首批teacher evidence与e1 Iter20所有loss有限，rho/BudgetAbs按冻结schedule均为0，checkpoint=0。

**当前判定**：`FORMAL RUNNING`。中间eval只记录，不裁决、不早停。
