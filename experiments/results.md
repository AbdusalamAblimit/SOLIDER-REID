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
