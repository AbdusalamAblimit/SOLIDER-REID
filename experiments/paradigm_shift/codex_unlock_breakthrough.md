Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019f0b96-3e93-71c3-a8c5-a9759a555a0c
--------
user
用户 UNLOCK: **不限 SOLIDER backbone(可换任意 foundation)+可换数据规模/task**。之前训练侧 Top5 被框在 SOLIDER+Market+死区, 天花板 6.2-6.8(你自己说要换数据/task 才有 8-9)。现在不限——找能突破到 **8-9 分 CCF-B** 的 person-ReID(或紧邻 person-centric retrieval)**训练侧/方法创新**, 必须是真能发 B 类的方法稿。可换: backbone(DINOv2/CLIP/SAM/EVA/任意)+数据规模(下新大规模数据集, 列具体哪个 public 数据)+task framing(person-centric 相邻)。仍避死区: occluded 内部/aerial-ground 视频/domain-gen foundation-preserving/gallery-side trick(hubness/rank-regret)/open-set/SMPL 几何/纯 test-time marginalization/single-support-CVaR/camera-transport(强 backbone 已 camera-invariant)。仍守六点定律(训练端塑形 K 变体破坏 test-time marginalization 多样性)。每方向: 机制一句/**为何能到 8-9 而非 6.5**/换什么 backbone+数据+task/cheap kill-switch(per-query 必控 #false-in-topk)/先例风险(web search 查)/CCF-B 分。排序 Top5。**这次目标突破天花板, 别再给 6.5 级保守货**。务实中文。
codex
我先按仓库接手规则快速读当前实验与 story，再做外部先例检索；这次会把目标限定在“训练侧/方法创新”，并把能不能真正够 B 类放在排序核心。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
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

exec
/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 决策记录 — Phase 2: Pure Pose Heatmap

### [2026-03-07 19:30] 决策 #1

**上下文**: Phase 1 (33 个实验) 已充分探索了基于 ViTPose visibility 向量的所有方向。最佳训练端改进仅 +1.4%（GiLt+PCFC）。用户指示放弃 visibility 方向，转向纯 pose heatmap + mmpose 更鲁棒模型。

**选择**: 从 SOLIDER 作者原始代码重新开始，纯 pose heatmap 方向
**理由**:
1. Visibility 向量不够可靠（与 AP 相关性仅 0.237）
2. PCFC alpha suppression 限制了所有后续改进
3. 纯 pose 热图更原始、更可靠、更多论文验证过有效性
4. 从干净代码开始避免 33 个实验的代码污染

**执行计划**:
1. 先跑 baseline 确认可复现（预期 mAP ~56.6%）
2. 选择并部署 mmpose 模型进行 pose heatmap 提取
3. 设计新的 heatmap 利用方式

### [2026-03-07 19:22] 决策 #2

**上下文**: 用户补充了多条重要指导：
1. 姿态模型可以参与训练（不限于离线热图）
2. 尽可能做训练侧创新（NFC/RR 等 test-time 方法不够公平）
3. 可以大胆修改 backbone 中间层
4. 数据必须从 log 文件读取，不能凭记忆

**选择**: 以训练侧创新为核心方向
**理由**:
1. NFC/Re-ranking 等 test-time 方法所有 SOTA 都可以用，不算公平的对比
2. 训练侧创新才是论文的核心贡献
3. 用户允许 pose 模型参与训练 + 修改中间层 → 创新空间大大增加
4. 可以考虑：在线 pose 特征注入、pose-guided attention、pose 结构约束等

**潜在技术方向**:
- 冻结的 pose 模型提取特征 → 通过 cross-attention 注入 Swin 中间层
- Pose heatmap 作为 spatial attention bias 直接修改 window attention
- Pose 骨骼结构约束 part features 之间的关系

### [2026-03-09 11:55] 决策 #3

**上下文**: exp001 (Pose Part Pooling with sigmoid) 完成。结果：mAP 57.1% (+0.5%), R1 66.7% (+0.2%)。有效但提升有限。关键发现：id_part 收敛极慢（最终仍在 2.0 vs id_global 0.2），说明 sigmoid 热图在 12×4 分辨率的 soft attention pooling 不够 discriminative。

**选项**:
  A. 改进 part pooling：使用 spatial softmax 代替 sigmoid，增强热图峰值对比度
  B. 放弃 part pooling，转向 pose heatmap 作为 attention bias 注入 Swin backbone
  C. 使用更高分辨率的中间层特征（stage2: 24×8 而非 stage3: 12×4）

**选择**: 先试 A（spatial softmax 改进），如果 id_part 收敛改善但最终结果仍有限，再转 B
**理由**:
1. exp001 证明 part pooling 方向有效（+0.5% mAP），但 id_part 是瓶颈
2. Spatial softmax 是最小改动，只改一行代码就能验证 "热图对比度不够" 的假设
3. 如果 id_part 收敛问题解决，part pooling 可能有更大提升空间
4. 如果 A 验证后仍不够，则 B 是完全不同的方向，有更大的创新性

**执行结果**: exp002 结果 mAP 57.2% vs exp001 57.1%，几乎无差异。id_part 训练中期收敛更快但最终效果相同。**结论：归一化方式不是瓶颈，转向方案 B。**

### [2026-03-09 14:13] 决策 #4

**上下文**: exp001 和 exp002 结果对比完成。两种归一化方式（sigmoid vs spatial_softmax）效果几乎一致。特征模式消融发现 part-only > concat > global，说明 part 特征有效但融合方式有问题。

**关键发现**:
1. 两种 normalization 最终 part-only mAP 都是 57.5%（+0.9% vs baseline）
2. Concat 融合反而比 part-only 差（1/N scaling 稀释信号）
3. id_part 收敛慢不是 normalization 的问题，而是 12×4 分辨率下 part 区分度本身有限

**选项**:
  A. 改进 part pooling 的融合方式（如 learnable weights, attention-based fusion）
  B. 转向 pose heatmap attention bias 注入 Swin backbone 中间层
  C. 提高 part 特征图分辨率（使用 stage2 特征 24×8）

**选择**: A — 改进特征融合方式。Part 特征已被证明有效（+0.9% mAP），但融合方式拖累了整体效果。这是最直接的改进方向。

**理由**:
1. Part-only 已经超 baseline 0.9%，说明 part 学到了有用信息
2. 当前 concat 的 1/N scaling 太朴素，直接稀释了 part 信号
3. 改进融合方式是低风险高回报：不需要改 backbone，只需修改测试时的特征组合
4. 如果简单的融合改进有效，可以作为消融实验的重要证据
5. B 和 C 是更大的改动，作为备选

**具体方案**: exp003 — 移除 1/N scaling，等权拼接 global + parts；或测试只用 part-only 作为最终特征

**执行结果**: exp003 在 ep60 终止，mAP 50.2%（-6.4% vs baseline）。降低 global loss weight 严重伤害 backbone 特征质量。Part 分类器虽学得更快（id_part 2.08 vs exp001 ~3.3），但池化的 backbone 特征变差了。**结论：global 和 part 是共生关系，不能通过削弱 global 来强化 part。**

### [2026-03-09 15:32] 决策 #5

**上下文**: exp001-003 完成。核心发现：
1. Part pooling 有效（+0.9% mAP with part-only feature）
2. 归一化方式（sigmoid vs spatial_softmax）无差异
3. 融合方式：part-only > concat（1/N scaling 有害）
4. 降低 global weight 反而伤害 part（因为 backbone 质量下降）
5. id_part 收敛极慢是核心瓶颈（id_part≈2.0 vs id_global≈0.2）

**关键问题**: 如何在不削弱 backbone 的前提下增强 part 学习？

**选项**:
  A. 独立 Part BN+分类器 + 更高 LR（加速 part 收敛，不改 global loss weight）
  B. 转向 Direction B：Pose heatmap 作为 attention bias 注入 Swin backbone 中间层（全新方向）
  C. Part feature 使用更高分辨率特征图（stage2: 24×8 而非 stage3: 12×4）
  D. 在 Part head 加入额外的 self-attention 层增强 part 特征表达
  E. 改进 Part 学习信号：per-part triplet loss (GiLt) + part-specific augmentation

**选择**: E — Per-part triplet loss (GiLt)
**理由**:
1. Phase 1 中 GiLt 已证明有效（+0.5% on top of PCFC），这次在 pure heatmap 框架下重试
2. 当前 part triplet 是"所有 part 特征拼起来做一个 triplet"，每个 part 没有独立的 hard positive/negative mining
3. Per-part triplet 让每个 part 独立学习判别性特征，直接解决 id_part 收敛慢的问题
4. 最小改动：只改 loss 计算方式，不改 backbone 或 part pooling 模块
5. 如果 GiLt 有效，可以作为消融实验的重要证据（"per-part triplet vs global triplet"）

**方案 B 作为备选**：如果 GiLt + part pooling 组合无法超过 +1.5% mAP，则转向全新的 backbone attention 方向。

**执行结果**: exp004 PFM 是中性结果。mAP 与 exp001 part-only 相同（57.5%），R1 反而下降 0.8%。PFM 加速收敛但不改善最终表征。**结论：不要在同一处重复使用 pose 信息（PFM+part pooling 是冗余的）。**

### [2026-03-09 17:52] 决策 #6

**上下文**: exp001-004 已探索了当前 part pooling 架构的多个变体：
- exp001/002: 不同热图归一化（sigmoid vs spatial_softmax）→ 无差异
- exp003: 改变 loss 权重 → 负面
- exp004: 加 PFM feature modulation → 中性

当前最佳：mAP 57.5% (part-only), R1 67.1% (+0.9%/+0.6% vs baseline)

**核心瓶颈**: id_part ≈ 2.0 无法进一步降低，part 特征质量受限于 12×4 分辨率。

**选项**:
  A. 使用 stage 2 特征 (24×8, 384ch) 做 part pooling — 4× spatial resolution
  B. Part diversity loss — 惩罚 part 特征间的相似度
  C. 转向 backbone attention 注入 — 修改 Swin 中间层
  D. Part-specific data augmentation — 基于 pose 的部位级数据增强
  E. Adaptive global-part fusion — 学习动态融合权重

**选择**: A — 使用 stage 2 高分辨率特征做 part pooling

**理由**:
1. 当前 12×4 分辨率对 5 个 part 来说太粗（每个 part 只能覆盖 2-3 个 spatial position）
2. Stage 2 (24×8 = 192 positions) 提供 4× 空间分辨率，pose heatmap attention 可以更精确
3. 384 channels 虽然比 768 少，但仍然有丰富的语义信息
4. 实验简单：只需改一下 part pooling 使用的特征图来源
5. 如果分辨率是瓶颈，这个实验会看到 id_part 明显改善

**风险**: Stage 2 特征可能不够 semantic（还没经过 stage 3 的进一步抽象）

**执行结果**: exp005 明确负面。ep40 mAP 仅 37.0%（baseline 56.6%）。id_part 到 ep49 才降到 4.70（exp001 同期 ~2.0）。**确认：Stage 2 特征语义不足以支撑 part-level identity 分类。** 更高空间分辨率无法补偿语义信息的缺失。

### [2026-03-09 19:10] 决策 #7

**上下文**: exp005 证明浅层（stage 2）特征不够 semantic。exp001-005 总结：
- Part pooling 方向的上限在 +0.9% mAP（part-only mode）
- Fusion 方式（concat_scaled vs equal_concat）对最终结果影响约 0.4%
- PFM 是冗余的；stage 2 太浅
- id_part 收敛始终慢于 id_global

**核心问题**：如何突破 +0.9% 的瓶颈？

**选项**:
  A. 改进 test-time 融合（L2-norm concat）— 可能挤出 0.2-0.5%，但不需要训练
  B. 转向 backbone attention 注入 — pose 信息参与特征形成过程
  C. 多尺度 part pooling — stage 2 spatial + stage 3 semantic 的融合
  D. Part feature diversity loss — 惩罚 part 间的冗余
  E. Pose-guided token selection — 剔除无关 token 提高效率

**选择**: 先做 A（test-time L2-norm fusion，零成本验证），然后转 B（backbone attention）

**理由**:
1. A 不需要训练，5 分钟可验证，如果能把 +0.9% 提升到 +1.2%，对论文有价值
2. B 是全新方向，改变特征形成过程本身，可能突破当前 part pooling 的上限
3. C-E 仍在 part pooling 框架内优化，上限有限
4. B 的创新性更好（"pose-conditioned attention" vs "better pooling"），更适合论文

**执行结果**:
- exp006 (A): L2-norm concat 57.4% vs concat 57.2%，小改进但仍不如 part-only (57.5%)。融合方向上限已到。
- **exp007 (B): PSG backbone injection → mAP 58.3%, R1 67.9%。Phase 2 最佳结果！+1.7% mAP, +1.4% R1。超过 Phase 1 最佳 (58.0%/68.0%)。**

### [2026-03-09 21:25] 决策 #8

**上下文**: exp007 PSG 取得突破性结果 (58.3%/67.9%)。关键发现：
1. Backbone-level pose injection (+1.7%) 远优于 post-hoc part pooling (+0.9%)
2. 纯 global feature，无需 part branch，架构极简
3. 额外参数仅 102K（两个 PSG 模块），几乎不增加计算量
4. 已超过 Phase 1 最佳

**下一步方向**:
  A. PSG + Part Pooling 组合 — 让 backbone 和 part branch 同时利用 pose
  B. PSG 消融实验 — 证明 PSG 每个组件的必要性
  C. PSG 在不同 stage 注入 — Stage 2 vs Stage 3 vs 全部 stages
  D. PSG 超参数分析 — hidden_dim, 是否 sigmoid, etc.

**选择**: A — PSG + Part Pooling 组合

**理由**:
1. PSG global feat (58.3%) 和 part-only feat (57.5%) 都有各自的优势
2. PSG 改善了 backbone 特征质量 → part features 也应该受益
3. 组合后可能进一步提升（PSG backbone + enhanced part features）
4. 如果组合有效，这就是完整的方法（backbone injection + part pooling = 全方位 pose 利用）

**执行结果**: exp008 mAP 57.7%, R1 66.0%。**组合不叠加**，低于 PSG-only (58.3%/67.9%)。Part pooling 的 part_only 测试模式丢弃了 PSG 增强的 global feature，而 part features 本身无法匹配 PSG-global 的质量。**结论：backbone-level injection 是更有效的 pose 利用方式，post-hoc pooling 在 PSG 基础上没有增量价值。**

### [2026-03-09 23:35] 决策 #9

**上下文**: exp007 (PSG) 和 exp008 (PSG+Part) 的对比揭示了重要规律：
1. PSG backbone injection: mAP 58.3% (+1.7%) — 全局特征，无 part branch
2. PSG + Part Pooling: mAP 57.7% (+1.1%) — part_only 测试，丢弃 global
3. Part Pooling alone: mAP 57.5% (+0.9%) — exp001

**核心洞察**:
- PSG 的增益主要来自改善全局特征质量，而 part pooling 依赖的是局部特征
- 两种方法的增益来源有重叠：都利用 pose heatmap 做 spatial attention
- 在 part_only 测试模式下，PSG 增强的 global 特征被浪费了

**选项**:
  A. PSG + concat 融合 — 保留 PSG global + part features，不丢弃全局特征
  B. 多 stage PSG — 在 Stage 2 也注入 PSG，更早引入 pose 先验
  C. PSG 改进 — 更强的 gate 机制（如 channel attention, multi-head gate）
  D. Backbone freeze warmup — 冻结 backbone 前 5 epochs，防止随机初始化模块破坏预训练

**选择**: 先做 B（多 stage PSG），这是架构级改进，有更大创新潜力

**理由**:
1. 当前 PSG 只在 Stage 3（2 个 block）注入，信息利用有限
2. 多 stage 注入可以让 pose 信息更早参与特征形成（Stage 2 的 24×8 分辨率对 pose heatmap 更有利）

exec
/bin/zsh -lc "sed -n '1,260p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 创新点头脑风暴 — Phase 2: Pure Pose Heatmap

## 2026-04-15 PRCV 重审：回到 PSG 主线，重做 multi-stage 消融

### 这轮重审后的核心判断

1. `PSG` 仍然是当前最稳的主创新点  
   - `exp007` 单次 `58.3 / 67.9`
   - 3-seed mean `57.83 / 67.13`
   - backbone injection 明确优于 post-hoc pooling

2. `2-stage PSG` 有希望作为最终版本，但现有证据还不够干净  
   - `exp009 / exp251 / exp253` 说明 multi-stage **不是普遍自动更优**
   - `exp255 vs exp255b` 又强烈说明：在 `GCN512` 结构分支下，`2-stage PSG` 是关键条件

3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
   - `LGPA-D` 像 detached semantic part asset
   - `MaxSim/POT/flip` 主要是 test-time
   - `exp257-259` 已说明 recipe 空间基本耗尽

### 当前真正该补的不是新故事，而是干净消融

用户已明确说明所有实验都可以重跑，因此下一步最该做的是：

1. 不再把旧结果当最终消融闭环
2. 重新设计 `PSG` / `2-stage PSG` / `3-stage PSG` 的干净对照
3. 把“multi-stage PSG 什么时候有用”这件事说清楚

### 当前推荐验证顺序

1. **基础 PSG 消融**
   - no PSG
   - 1-stage PSG
   - 2-stage PSG
   - 3-stage PSG

2. **结构分支依赖性消融**
   - GCN256 + 1-stage
   - GCN256 + 2-stage
   - GCN512 + 1-stage
   - GCN512 + 2-stage

3. **必要时再补 semantic 分支依赖性**
   - LGPA-only + 1-stage / 2-stage
   - LGPA+GCN + 1-stage / 2-stage

### 当前主线口径

从现在开始，PRCV 方向优先写成：

- `PSG` = 主创新
- `2-stage PSG` = scalable extension / 当前最终版本
- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets

### 参考

详细文献压缩与路线说明见：
`experiments/paper_notes/2026-04-15_prcv_reset.md`

## 前轮教训 (Phase 1, 33 experiments)
- ViTPose visibility 向量不够可靠（AP 相关性仅 0.237）
- 中间层 visibility modulation 有害（破坏预训练空间结构），但这是 visibility 特有问题
- PCFC alpha suppression 是该框架特有的脆弱平衡点
- NFC test-time 方法有效但不算训练端创新
- **关键结论**: 不要用 visibility 向量，用原始 pose 热图

---

## Phase 2 实验总结 (exp001-exp021, 21 个实验)

### 已证实的核心发现

**1. Backbone Injection > Post-hoc Pooling**
- PSG (特征形成阶段注入 pose) +1.7% mAP，仅 102K params
- Part Pooling (特征形成后用 pose 选择) +0.9% mAP，~2.6M params
- PFM (后置调制) 中性效果
- **结论**: 让 backbone 在特征提取过程中知道人体结构，比事后选择更有效

**2. PSG 对空间级干扰敏感，但通道级正交不干扰**
- PSG + PAB Combo: ❌ (-0.7% vs PSG-only)
- PSG + Part Pooling: ❌ (-0.6% vs PSG-only)
- PSG + Part Supervision (global test): ❌ (-0.7% mAP, -2.1% R1)
- PSG + PCG (通道级 gate): 🟡 持平 (mAP 58.0%, -0.3% vs PSG-only)
- **核心瓶颈**: 在同一个 Stage 3 上叠加任何模块都会梯度干扰

**3. 复杂度越高，效果越差**
- 有效方法排序: PSG(58.3%) > PCG-only(57.8%) > PRA(57.8%) > Part Pooling(57.5%) > PAB(57.4%) > PXA(57.3%) > CAPSG(57.2%)
- PXA/CAPSG 最复杂，效果最差
- **PSG 的极简性就是它的优势**

**4. PSG 跨数据集/backbone 均有效（4090 验证）**

| 数据集 | Backbone | PSG mAP 提升 |
|--------|----------|-------------|
| Occluded-Duke | Swin-Tiny | +1.7% |
| Occluded-Duke | Swin-Small (lr4) | +2.0% |
| Market-1501 | Swin-Tiny | +0.8% |
| Market-1501 | Swin-Small (lr4) | +0.6% |

### 关键教训
- **梯度干扰是核心瓶颈**: 21 个实验中所有在 PSG 基础上"加东西"的尝试都失败了，原因是同一个 Stage 3 内的模块共享梯度流
- **要突破 PSG，必须用独立的处理路径**，避免梯度干扰
- **可以添加大模块**: 用户确认不限于轻量模块，可以加 ResNet 分支、Decoder、GCN 等

---

## Phase 3: 新方向候选 (2025.03.11 Web 搜索 + 文献调研)

### ★ 方向 K: 双分支架构 — Pose-Guided Dual-Stream (PDS)
**优先级: ⭐⭐⭐⭐⭐**

```
Input → Swin Stage 1-2 (共享)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  Stage 3-A           Stage 3-B (独立权重)
  + PSG               + Pose Part Processing
    ↓                   ↓
   GAP              Part Pooling (基于 heatmap)
    ↓                   ↓
 Global Feat        Part Feats
    ↓                   ↓
 ID + Triplet      Part ID + Part Triplet
    ↓                   ↓
    └── concat (test) ──┘
```

- **核心想法**: 解决 exp008/014 暴露的梯度干扰问题。复制独立的 Stage 3 给 Part 分支。
- **为什么与前轮不同**: exp008 在同一 Stage 3 上做 PSG+Part，梯度干扰。PDS 用独立 Stage 3，各自优化。
- **代价**: ~6M 额外 params（Stage 3 ≈ 2 SwinBlocks × 768ch）
- **优势**: Part 分支可以集成更多 pose 操作（GCN、cross-attention），而不影响 Global 分支
- **参考**: PGFL-KD (ACM MM 2021) 三分支架构, FCFormer (TPAMI 2024) 双流设计
- **论文定位**: 主贡献之一 — "dual-stream pose-guided architecture with gradient-isolated part learning"

### ★ 方向 L: 关键点相对位置编码 (KP-RPE)
**优先级: ⭐⭐⭐⭐**

- **来源**: CVPR 2024 人脸识别 (Kim et al., "KeyPoint Relative Position Encoding for Face Recognition")
- **核心想法**: 将 Swin 的 attention bias 从像素相对位置改为关键点相对位置
  - 标准 RPE: bias(i,j) = table[xi-xj, yi-yj]
  - KP-RPE: bias(i,j) = MLP(dist(i,kp1),...,dist(j,kp1),...)
- **与 PSG 完全正交**: PSG 调制 feature 幅度（乘法），KP-RPE 调制 attention 路由（加法 bias）
- **与 exp012 PAB 的区别**: PAB 是单像素 spatial map，KP-RPE 编码 token 对之间的**结构关系**
- **参数量**: ~5-10K，几乎零开销
- **风险**: Swin 用 window attention (7×7 window)，12×4 feature map 上 window 划分可能限制效果
- **论文定位**: 与 PSG 叠加使用 — "spatial gating + structural attention routing"

### ★ 方向 M: 骨架图卷积特征传播 (Skeleton GCN)
**优先级: ⭐⭐⭐⭐**

```
PSG-enhanced Stage 3 features (12×4, 768ch)
              ↓
Bilinear sample at 17 keypoint locations → (17, 768)
              ↓
Skeleton GCN (2-3 layers, COCO 19 bone edges)
              ↓
可见部位特征 沿骨骼边传播到遮挡部位
              ↓
Part Feat Pool → concat with Global
```

- **核心价值**: 遮挡补全 — 当下半身被遮挡时，GCN 沿"髋→膝→踝"传播上半身特征
- **参考**: Tran-GCN (IET 2025), skeleton action recognition 领域成熟技术
- **参数量**: ~3-4M (17 nodes × 768 features × 2-3 GCN layers)
- **与 PDS (方向 K) 的结合**: 可以作为 Part 分支的核心模块
- **风险**: 前 21 个实验显示 post-backbone 方法收益有限，但 GCN 提供了全新的结构推理能力
- **论文定位**: "skeleton-topology-aware feature propagation for occlusion recovery"

### 方向 N: ControlNet-Style 加法注入
**优先级: ⭐⭐⭐**

- **来源**: ControlNet (ICCV 2023), LLaMA-Adapter
- **核心想法**: 复制 Stage 3 为 "pose encoder branch"，处理 pose heatmaps，通过 zero-conv 加法注入主干
- **与 PSG 的区别**: PSG 是乘法 x*(1+gate)，ControlNet 是加法 x + zero_conv(pose_feat)
- **可以和 PDS 整合**: 就是 PDS 的 Part 分支不做独立 pooling，而是加法 inject 回 Global 分支
- **参数量**: ~6M（完整 Stage 3 clone）或 ~100K（轻量 conv encoder）

### 方向 O: Pose Attention Supervision (PAS)
**优先级: ⭐⭐⭐**

- **来源**: PAFormer (arXiv 2024)
- **核心想法**: 不把 pose 热图作为输入，而是作为 attention 的**监督信号**。训练时让 attention map 匹配 pose heatmap 分布，推理时不需要 pose。
- **优势**: 零推理开销，改变 backbone 的内在表征
- **风险**: Swin window attention 使监督复杂化

### 方向 P: Pose-Guided Token Pruning
**优先级: ⭐⭐⭐**

- **来源**: PrATo (2025), HeatViT, Zero-TPrune (CVPR 2024)
- **核心想法**: 用 pose 热图计算 token 重要性，剪掉背景/遮挡物 token，只保留人体区域
- **效果**: 不仅加速推理，还从根本上消除遮挡物对特征的污染
- **与 PSG 的区别**: PSG 给所有 token 不同权重，token pruning 直接删除无关 token
- **挑战**: Swin window attention 对 token 数量有要求

### 方向 Q: 特征补全 (Feature Completion)
**优先级: ⭐⭐⭐**

- **来源**: FCFormer (TPAMI 2024)
- **核心想法**: 用 pose 热图识别遮挡区域（热图响应低），用 learnable tokens + decoder 重建遮挡区域特征
- **与 Skeleton GCN 的区别**: GCN 沿骨架边传播，Feature Completion 用 decoder 直接预测
- **可以结合 PDS**: 作为 Part 分支的特征补全模块

---

## 推荐实验路线

### Round 1: 大架构实验
1. **exp022: PDS (双分支)** — 最有潜力，解决核心梯度干扰问题
2. **exp023: KP-RPE** — 最轻量，与 PSG 正交，快速验证

### Round 2: 基于 Round 1 结果深化
- 如果 PDS 有效 → 在 Part 分支内集成 Skeleton GCN (方向 M)
- 如果 KP-RPE 有效 → PSG + KP-RPE 组合
- 如果都有效 → PDS + KP-RPE 组合

### Round 3: 论文补充实验
- 消融实验（PDS 各组件贡献）
- 效率分析（参数量、FLOPs、推理速度）
- 可视化（attention map、t-SNE、检索结果）

---

## exp022 PDS 结果反馈 (2026-03-11)

### PDS 实验结论
- **global-only 57.9%**: 接近但未超过 PSG-only 58.3% (-0.4%)
- **concat_scaled 57.5%**: Part 有微弱贡献
- **equal_concat 56.1%**: 5:1 维度比稀释 Global，不可用
- **part-only 55.2%**: Part 分支独立效果差

### 关键洞察
1. **Stage 3 权重解耦确实有效**: PDS global (57.9%) > exp008 PSG+Part same Stage3 (57.7%)，证明独立 Stage 3 保护了 PSG
2. **但共享 Stage 0-2 仍有轻微干扰**: 57.9% vs 58.3% 的 -0.4% gap 来自 Part 分支经共享层的反向传播
3. **Part 分支学习太慢**: 120 epoch 后 Part ID loss 仍高达 2.02（Global 为 0.17）。5 个独立分类器需要更多训练容量
4. **fusion 策略需要优化**: Part 维度是 Global 的 5 倍，等权 concat 本质上是给 Part 5 倍的投票权

### 方向修正

PDS 实验证明了 **"梯度干扰是可以通过架构解耦缓解的"** 这一核心假设。但也暴露了新问题：

**问题 1: Part 分支需要更好的训练策略**
- 可以尝试：stop_gradient 阻断 Part→共享层梯度，或 Part 分支延迟启动
- 但考虑到复杂度增加和收益不确定，这个方向的性价比可能不高

**问题 2: 当前方法组合天花板**
- PSG-only 58.3% 已经是非常好的单模块结果
- 所有组合实验都未能在此基础上叠加增益
- 也许应该接受 PSG 作为核心贡献，转向其他维度（如 test-time fusion、NFC 定制化）

**修正后的优先级**:
1. **exp023: 先尝试 stop_gradient 隔离** — 最简单的 fix，验证是否能消除 -0.4% gap
2. **如果 stop_gradient 有效** → PDS + stop_gradient 作为论文的 full model
3. **如果 stop_gradient 无效** → 放弃 dual-stream，PSG 作为核心贡献 + 其他正交方向 (KP-RPE, Skeleton GCN)

---

## exp023 PDS+StopGrad 结果反馈 (2026-03-11) 🎉


exec
/bin/zsh -lc "sed -n '1,240p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 论文故事线（持续更新）

> **⚠️ Phase 1-4 内容保留在下方。Phase 5 更新如下。**

## PRCV Reset (2026-04-15) — PSG 主线 + GCN 结构补充

### 当前一句话故事

现有 pose-guided occluded ReID 大多在特征形成之后再使用 pose 信息；我们提出 `PSG`，将 pose 先验前移到 backbone 表征学习阶段，并在最终系统中引入 `GCN` 结构分支做显式 skeleton relational reasoning，形成 semantic-structural complementary evidence。

### 当前重审结论

这轮重审后，PRCV 主故事优先回到 `PSG`，而不是继续把 `LGPA-D + MaxSim + flip` 当主创新。

当前更稳的写法是：
1. **PSG** 是主创新点
2. **2-stage PSG** 只作为 `PSG` 的最终 instantiation / final configuration，不单独抢主贡献位置
3. **GCN 必须明确写进方法**，但定位为 *structural pose branch*，不是与 `PSG` 并列的第二主创新
4. `LGPA-D / OA-SD / PLBOA` 作为完整系统资产
5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献

### 当前主判断

1. `exp007` 已经足够支撑 `PSG` 本体：
   - 单次 `58.3 / 67.9`
   - 3-seed mean `57.83 / 67.13`
   - backbone-level pose injection 明确优于 post-hoc part pooling

2. `GCN` 应该被强调，但应强调其**作用位置**而不是单独吹成主创新：
   - `GCN` 的价值是提供显式 skeleton structure evidence
   - 更适合作为 `PSG` 支撑下的结构分支，而不是与 `PSG` 平行的主贡献
   - `exp249` 与 `exp246` 已经说明 `LGPA-D + GCN` 双分支具备稳定互补性

3. `2-stage PSG` 可以作为最终版本，但**不必在主叙事里和 1-stage 正面对打**
   - `exp009`、`exp251`、`exp253` 都说明：multi-stage 不会在所有 scaffold 上自动更强
   - 但 `exp255 vs exp255b` 明确说明：在 `GCN512` 这类高容量结构分支上，`2-stage PSG` 是关键条件

4. 因为实验都可以重跑，接下来不把旧消融当最终版，而是重新设计干净的 `PSG` stage 消融矩阵

### 当前最强系统 scaffold

当前训练端最强实验是 `exp255`：
- `Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA`
- `FINAL = 73.2 / 83.3`

当前最关键的结构证据是：
- `exp255`: `GCN512 + 2-stage PSG = 73.2 / 83.3`
- `exp255b`: `GCN512 + 1-stage PSG = 71.5 / 81.9`

这组对照最适合在**消融**里写成：
> 最终实现采用 `2-stage PSG`；进一步对照显示，在高容量结构分支上，它比 `1-stage` 更能稳定支撑结构证据的发挥。

### 推荐写作口径

1. **标题 / 摘要 / 引言**
   - 只讲 `PSG`
   - 可以写：我们在 backbone 中间 stage 之间注入 pose 信息
   - 最多补一句：最终实现采用 two-stage instantiation

2. **方法部分**
   - 把 `PSG` 定义成一个通用的 pose-guided spatial gating 机制
   - 再说明：实际实验中采用 `2-stage PSG` 作为最终配置

3. **消融部分**
   - 再回答为什么最终选 `2-stage`
   - 用 `1-stage / 2-stage / 3-stage` 小表说明选择依据即可

### 论文里哪些模块要重点提及

1. **第一层：主贡献**
   - `PSG`
   - 写法：backbone 内的 pose-guided spatial gating

2. **第二层：关键支撑机制**
   - `GCN`
   - 写法：`GCN` 是 explicit structural pose reasoning branch
   - `2-stage PSG` 放在最终实现与消融选择中说明，不单列为主贡献

3. **第三层：完整系统资产**
   - `LGPA-D`
   - `OA-SD`
   - `PLBOA`
   - 写法：semantic branch + training recipe，不抢主创新位

4. **第四层：附加评测资产**
   - `MaxSim / POT / flip`
   - 写法：test-time supporting evaluations

### 推荐贡献点写法

1. 提出 `PSG`，在 backbone 内进行 pose-guided spatial gating，而不是在特征形成后再做 pose-aware pooling 或 filtering
2. 构建 semantic-structural complementary occluded ReID system，其中 `GCN` 提供显式 skeleton relational evidence，`LGPA-D` 提供语义 part evidence，与 `PSG` 形成互补
3. 在 Occluded-Duke 上系统验证该框架，并采用 `2-stage PSG` 作为最终实现；实验表明该设计能够更稳定地支撑高容量结构分支，最终在 Swin-Small 上得到当前最佳训练端结果之一

### 推荐摘要骨架

可按下面 4 句展开：

1. **问题句**
   - 现有 pose-guided occluded ReID 往往在特征提取完成后才利用 pose，因而对表征学习阶段的结构先验注入不足。

2. **方法句**
   - 我们提出 `PSG`，在 backbone 中间层通过 pose-conditioned spatial gating 直接调制特征形成过程。

3. **扩展句**
   - 在此基础上，我们结合 `GCN` 结构分支，以显式建模 skeleton relational evidence，并在最终实现中采用 `2-stage PSG` 作为具体配置，从而形成 semantic-structural complementary representation。

4. **结果句**
   - 在 Occluded-Duke 等基准上，该框架取得了当前项目最优结果之一，其中 `Swin-Small` 配置达到 `73.2 / 83.3`；消融进一步表明，最终采用的 `2-stage PSG` 更适合支撑高容量结构分支。

### 执行优先级

1. 重新设计 `PSG` 的干净 stage 消融：
   - no PSG
   - 1-stage PSG
   - 2-stage PSG
   - 3-stage PSG
2. 固定 branch 容量，单独验证 `2-stage PSG` 是否是高容量 `GCN` branch 的必要条件
3. 在此基础上，再决定最终论文标题更偏 `PSG` 还是 `Hierarchical PSG`

### 说明

详细重审与文献压缩总结见：
`experiments/paper_notes/2026-04-15_prcv_reset.md`

## Phase 5 Story Update (2026-04-08) — LGPA-D 时代

### 暂定标题
Language-Grounded Part Assembly for Occluded Person Re-Identification

### 当前最佳结果

| Backbone | Method | mAP (eq) | R1 (eq) | mAP (MaxSim) | R1 (MaxSim) |
|----------|--------|------|------|------|------|
| Tiny | LGPA-D+OA-SD | 65.3% | 75.7% | 66.0% | 76.4% |
| **Tiny** | **LGPA-D+GCN+OA-SD** | **65.5%** | **77.2%** | **66.3%** | **77.7%** |
| Small | LGPA-D+OA-SD (local) | 70.2% | 80.1% | 71.9% | 82.2% |
| **Small** | **LGPA-D+OA-SD (remote)** | **71.6%** | **81.6%** | **73.0%** | **82.7%** |
| Small | GCN+PAA+OA-SD (old baseline) | 70.6% | 82.6% | 72.3% | 82.9% |
| *Small* | *LGPA-D+GCN+OA-SD (exp249, 进行中)* | *TBD* | *TBD* | *TBD* | *TBD* |

### 核心贡献

1. **LGPA-D (Language-Grounded Part Assignment, Detached)**
   - 首次将 VLM (CLIP) 语义知识用于 occluded ReID 的 part assignment
   - 5 个语义 body parts: head, torso, arms, upper_legs, lower_legs
   - CLIP frozen text prototypes + cross-attention + pose heatmap bias
   - Detached from backbone → 不干扰训练, 全程 delta 为正
   - vs GCN skeleton features: +2.1% mAP (语义 > 结构)
   - vs PPA (non-detached): +4.4% (detach 消除后期干扰)

2. **PSG (Pose Spatial Gate)**
   - Backbone 内部 pose 注入 (Stage 3 block 间)
   - 轻量 102K params, +1.7% mAP
   - 改变特征形成方式, 不只是 post-hoc pooling

3. **Dual-Branch Architecture (LGPA-D + GCN)**
   - 语义 part features (LGPA-D) + 骨架 keypoint features (GCN) 正交互补
   - Tiny: +0.2 mAP / +1.5 R1 vs LGPA-D only
   - 两个 branch 都在 detached features 上操作

4. **MaxSim Hybrid Matching**
   - ColBERT-style late interaction 首次引入 person ReID
   - +1.0~1.5% mAP across all checkpoints
   - 理论框架: partial-set-to-partial-set matching

### 关键消融发现

1. **Detach barrier 是根本性约束**: 
   - Non-detached (exp243): ep80 -1.1 mAP → 后期干扰
   - Detached (exp244): ep120 +2.1 mAP → 全程正向
   - 250 实验验证: backbone 必须完全由主 loss 驱动

2. **CLIP 语义 > GCN 结构**:
   - LGPA-D 无 OA-SD (63.6) ≈ GCN + OA-SD (63.2)
   - CLIP 的 part assignment 能力 ≈ OA-SD 的训练增强

3. **训练集 95.8% visible**: 
   - 所有 visibility-dependent 训练方法失败 (VCSR, routing)
   - PLBOA (pixel-level occlusion augmentation) 是唯一有效补充

### 论文叙事

> Occluded person ReID 的核心挑战不是"如何处理遮挡"而是"如何定义和匹配不完整的身份证据"。
> 我们提出 LGPA (Language-Grounded Part Assembly): 利用 CLIP 的语义理解能力，
> 将 backbone 空间特征分解为语义 body parts，在 detached 特征上安全操作。
> 配合 PSG (backbone 内 pose 注入) 和 MaxSim (part-level late interaction matching)，
> 形成完整的 "语义引导提取 → 部分集合匹配" 框架。

---

## Phase 4 Story Update (2026-04-02)

### 当前最佳结果 (Phase 4 时期)

| Backbone | Method | mAP (eq) | R1 (eq) | mAP (maxsim) | R1 (maxsim) |
|----------|--------|------|------|------|------|
| Tiny | GCN+PAA+OA-SD | 63.2% | 75.4% | 64.2% | 77.1% |
| Tiny | **GCN+PAA+OA-SD+GSPB** | 62.9% | 74.3% | **64.6%** | **76.0%** |
| Small | GCN+PAA+OA-SD | 70.6% | 82.6% | 72.3% | 82.9% |
| Small | GCN+PAA+OA-SD+PKC | 70.6% | 81.8% | **72.4%** | **83.1%** |

### Phase 4 发现

1. **MaxSim Behavior on Tiny**: `MaxSim` 的收益更依赖 per-keypoint consistency，而不是简单取决于 global 强弱。

2. **GSPB (Gradient-Scaled Part Branch)**: 5% Part→Backbone 梯度大幅加速早期收敛 (+5.8% at ep10!) 但不改善 final。首次发现 detach 与 non-detach 之间的中间解。

3. **OA-SD Teacher Fix**: 修复了 EMA teacher 的 Dropout/DropPath/BN 噪声问题。修复后 teacher 更稳定，但 final 结果不变（EMA 的自修正性）。

4. **per-keypoint training loss 全面证伪**: PKC, MST, PACI, OERL, BA-PKC — 10 个实验全部失败。根本原因: detached GCN 阻断梯度到 backbone，non-detached 与 CE 冲突。

---

## Phase 3 Story Update (2026-03-23)

### 暂定标题
Pose-Guided Structural Token Decomposition for Occluded Person Re-Identification

### 核心贡献（更新 2026-03-24）

1. **STD-PR (Structural Token Decomposition with Pose-guided Routing)**
   - 用 pose-biased cross-attention 将 spatial tokens 转为 structural body-part tokens
   - 替代 GCN keypoint sampling：cross-attention 比 bilinear sampling 更善于利用 data augmentation
   - 3-seed mean: 62.6%±0.87 mAP (+1.87 vs baseline)

2. **PLBOA (Pose-guided Lower-Body Occlusion Augmentation)**
   - 基于 train-test occlusion gap 分析（1.8% vs 24.4% lower-body occluded）
   - 用真实 VOC 物体贴到 hip 以下区域
   - 2-seed mean: 62.3% mAP with GCN (+1.57 vs baseline)

3. **STD-PR+PLBOA Synergy**
   - STD-PR alone: -2.4 vs GCN
   - STD-PR+PLBOA: **+0.7 vs GCN+PLBOA**
   - PLBOA 增益：GCN +1.6 vs STD-PR **+4.7** (3x 差距！)
   - 核心发现：cross-attention 比 keypoint sampling 更善于利用 augmentation

4. **MaxSim (ColBERT Late Interaction, 辅助)**
   - 零训练成本 test-time method
   - +1.0~1.5% mAP across checkpoints

codex
本地文档显示后续已经从 exp148/149 往 LGPA-D、OA-SD、PSG 重审走了不少，且现有路线的强点仍主要局限在 Occluded-Duke/SOLIDER 系统资产。下面我会把“解锁 backbone/数据/task”当成重新开题来查先例，重点避开你列的死区。
exec
/bin/zsh -lc 'rg -n "exp25|exp24|exp23|exp22|exp21|exp20|exp19|exp18|exp17|exp16|exp15|exp148|exp149|exp151|exp257|exp259|LGPA|PSG|OA-SD" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/paper_materials/story.md:5:## PRCV Reset (2026-04-15) — PSG 主线 + GCN 结构补充
experiments/paper_materials/story.md:9:现有 pose-guided occluded ReID 大多在特征形成之后再使用 pose 信息；我们提出 `PSG`，将 pose 先验前移到 backbone 表征学习阶段，并在最终系统中引入 `GCN` 结构分支做显式 skeleton relational reasoning，形成 semantic-structural complementary evidence。
experiments/paper_materials/story.md:13:这轮重审后，PRCV 主故事优先回到 `PSG`，而不是继续把 `LGPA-D + MaxSim + flip` 当主创新。
experiments/paper_materials/story.md:16:1. **PSG** 是主创新点
experiments/paper_materials/story.md:17:2. **2-stage PSG** 只作为 `PSG` 的最终 instantiation / final configuration，不单独抢主贡献位置
experiments/paper_materials/story.md:18:3. **GCN 必须明确写进方法**，但定位为 *structural pose branch*，不是与 `PSG` 并列的第二主创新
experiments/paper_materials/story.md:19:4. `LGPA-D / OA-SD / PLBOA` 作为完整系统资产
experiments/paper_materials/story.md:24:1. `exp007` 已经足够支撑 `PSG` 本体：
experiments/paper_materials/story.md:31:   - 更适合作为 `PSG` 支撑下的结构分支，而不是与 `PSG` 平行的主贡献
experiments/paper_materials/story.md:32:   - `exp249` 与 `exp246` 已经说明 `LGPA-D + GCN` 双分支具备稳定互补性
experiments/paper_materials/story.md:34:3. `2-stage PSG` 可以作为最终版本，但**不必在主叙事里和 1-stage 正面对打**
experiments/paper_materials/story.md:35:   - `exp009`、`exp251`、`exp253` 都说明：multi-stage 不会在所有 scaffold 上自动更强
experiments/paper_materials/story.md:36:   - 但 `exp255 vs exp255b` 明确说明：在 `GCN512` 这类高容量结构分支上，`2-stage PSG` 是关键条件
experiments/paper_materials/story.md:38:4. 因为实验都可以重跑，接下来不把旧消融当最终版，而是重新设计干净的 `PSG` stage 消融矩阵
experiments/paper_materials/story.md:42:当前训练端最强实验是 `exp255`：
experiments/paper_materials/story.md:43:- `Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA`
experiments/paper_materials/story.md:47:- `exp255`: `GCN512 + 2-stage PSG = 73.2 / 83.3`
experiments/paper_materials/story.md:48:- `exp255b`: `GCN512 + 1-stage PSG = 71.5 / 81.9`
experiments/paper_materials/story.md:51:> 最终实现采用 `2-stage PSG`；进一步对照显示，在高容量结构分支上，它比 `1-stage` 更能稳定支撑结构证据的发挥。
experiments/paper_materials/story.md:56:   - 只讲 `PSG`
experiments/paper_materials/story.md:61:   - 把 `PSG` 定义成一个通用的 pose-guided spatial gating 机制
experiments/paper_materials/story.md:62:   - 再说明：实际实验中采用 `2-stage PSG` 作为最终配置
experiments/paper_materials/story.md:71:   - `PSG`
experiments/paper_materials/story.md:77:   - `2-stage PSG` 放在最终实现与消融选择中说明，不单列为主贡献
experiments/paper_materials/story.md:80:   - `LGPA-D`
experiments/paper_materials/story.md:81:   - `OA-SD`
experiments/paper_materials/story.md:91:1. 提出 `PSG`，在 backbone 内进行 pose-guided spatial gating，而不是在特征形成后再做 pose-aware pooling 或 filtering
experiments/paper_materials/story.md:92:2. 构建 semantic-structural complementary occluded ReID system，其中 `GCN` 提供显式 skeleton relational evidence，`LGPA-D` 提供语义 part evidence，与 `PSG` 形成互补
experiments/paper_materials/story.md:93:3. 在 Occluded-Duke 上系统验证该框架，并采用 `2-stage PSG` 作为最终实现；实验表明该设计能够更稳定地支撑高容量结构分支，最终在 Swin-Small 上得到当前最佳训练端结果之一
experiments/paper_materials/story.md:103:   - 我们提出 `PSG`，在 backbone 中间层通过 pose-conditioned spatial gating 直接调制特征形成过程。
experiments/paper_materials/story.md:106:   - 在此基础上，我们结合 `GCN` 结构分支，以显式建模 skeleton relational evidence，并在最终实现中采用 `2-stage PSG` 作为具体配置，从而形成 semantic-structural complementary representation。
experiments/paper_materials/story.md:109:   - 在 Occluded-Duke 等基准上，该框架取得了当前项目最优结果之一，其中 `Swin-Small` 配置达到 `73.2 / 83.3`；消融进一步表明，最终采用的 `2-stage PSG` 更适合支撑高容量结构分支。
experiments/paper_materials/story.md:113:1. 重新设计 `PSG` 的干净 stage 消融：
experiments/paper_materials/story.md:114:   - no PSG
experiments/paper_materials/story.md:115:   - 1-stage PSG
experiments/paper_materials/story.md:116:   - 2-stage PSG
experiments/paper_materials/story.md:117:   - 3-stage PSG
experiments/paper_materials/story.md:118:2. 固定 branch 容量，单独验证 `2-stage PSG` 是否是高容量 `GCN` branch 的必要条件
experiments/paper_materials/story.md:119:3. 在此基础上，再决定最终论文标题更偏 `PSG` 还是 `Hierarchical PSG`
experiments/paper_materials/story.md:126:## Phase 5 Story Update (2026-04-08) — LGPA-D 时代
experiments/paper_materials/story.md:135:| Tiny | LGPA-D+OA-SD | 65.3% | 75.7% | 66.0% | 76.4% |
experiments/paper_materials/story.md:136:| **Tiny** | **LGPA-D+GCN+OA-SD** | **65.5%** | **77.2%** | **66.3%** | **77.7%** |
experiments/paper_materials/story.md:137:| Small | LGPA-D+OA-SD (local) | 70.2% | 80.1% | 71.9% | 82.2% |
experiments/paper_materials/story.md:138:| **Small** | **LGPA-D+OA-SD (remote)** | **71.6%** | **81.6%** | **73.0%** | **82.7%** |
experiments/paper_materials/story.md:139:| Small | GCN+PAA+OA-SD (old baseline) | 70.6% | 82.6% | 72.3% | 82.9% |
experiments/paper_materials/story.md:140:| *Small* | *LGPA-D+GCN+OA-SD (exp249, 进行中)* | *TBD* | *TBD* | *TBD* | *TBD* |
experiments/paper_materials/story.md:144:1. **LGPA-D (Language-Grounded Part Assignment, Detached)**
experiments/paper_materials/story.md:152:2. **PSG (Pose Spatial Gate)**
experiments/paper_materials/story.md:157:3. **Dual-Branch Architecture (LGPA-D + GCN)**
experiments/paper_materials/story.md:158:   - 语义 part features (LGPA-D) + 骨架 keypoint features (GCN) 正交互补
experiments/paper_materials/story.md:159:   - Tiny: +0.2 mAP / +1.5 R1 vs LGPA-D only
experiments/paper_materials/story.md:170:   - Non-detached (exp243): ep80 -1.1 mAP → 后期干扰
experiments/paper_materials/story.md:171:   - Detached (exp244): ep120 +2.1 mAP → 全程正向
experiments/paper_materials/story.md:175:   - LGPA-D 无 OA-SD (63.6) ≈ GCN + OA-SD (63.2)
experiments/paper_materials/story.md:176:   - CLIP 的 part assignment 能力 ≈ OA-SD 的训练增强
experiments/paper_materials/story.md:185:> 我们提出 LGPA (Language-Grounded Part Assembly): 利用 CLIP 的语义理解能力，
experiments/paper_materials/story.md:187:> 配合 PSG (backbone 内 pose 注入) 和 MaxSim (part-level late interaction matching)，
experiments/paper_materials/story.md:198:| Tiny | GCN+PAA+OA-SD | 63.2% | 75.4% | 64.2% | 77.1% |
experiments/paper_materials/story.md:199:| Tiny | **GCN+PAA+OA-SD+GSPB** | 62.9% | 74.3% | **64.6%** | **76.0%** |
experiments/paper_materials/story.md:200:| Small | GCN+PAA+OA-SD | 70.6% | 82.6% | 72.3% | 82.9% |
experiments/paper_materials/story.md:201:| Small | GCN+PAA+OA-SD+PKC | 70.6% | 81.8% | **72.4%** | **83.1%** |
experiments/paper_materials/story.md:209:3. **OA-SD Teacher Fix**: 修复了 EMA teacher 的 Dropout/DropPath/BN 噪声问题。修复后 teacher 更稳定，但 final 结果不变（EMA 的自修正性）。
experiments/paper_materials/story.md:250:4. **PSG + GCN pipeline**: 提供高质量 body-part token set 的提取方法
experiments/paper_materials/story.md:254:- MaxSim training: exp152 进行中
experiments/paper_materials/story.md:255:- Ablation: soft vs hard MaxSim (exp152 vs exp152b) 进行中
experiments/paper_materials/story.md:265:1. **PSG (Pose Spatial Gate)**
experiments/paper_materials/story.md:270:   `exp007a` 三 seed 均值 `59.37%`，相对 PSG 稳定 `+1.53% mAP`。
experiments/paper_materials/story.md:277:4. **PSG + KPP/GCN branch 的贡献应写成 fusion 增益**
experiments/paper_materials/story.md:323:这是 “recover” 范式，不同于 “suppress”(PSG) / “inject”(PAA) / “select”(pruning)。
experiments/paper_materials/story.md:326:1. **PSG + 0.5x loss + GCN**: 基础三件套，已确认
experiments/paper_materials/story.md:330:5. **问题层面**: 如果 PGFI 也失败，应考虑把 story 转向 “pose-guided multi-granularity representation”（PSG+GCN+equal_concat 的整体范式叙事），而非继续追求单一新模块
experiments/paper_materials/story.md:337:| PSG | global | 57.83±0.50% | 67.13±0.84% | 3-seed |
experiments/paper_materials/story.md:338:| PSG + 0.5x loss | global | 59.37±0.32% | 69.43±0.12% | 3-seed |
experiments/paper_materials/story.md:340:| PSG + GCN | global | 59.33±0.40% | 68.87±1.00% | 3-seed |
experiments/paper_materials/story.md:341:| PSG + GCN | concat_scaled | 60.20±0.44% | 73.13±0.29% | 3-seed |
experiments/paper_materials/story.md:342:| **PSG + GCN** | **equal_concat** | **60.73±0.47%** | **72.57±0.58%** | **当前最强且已确认的无后处理模式** |
experiments/paper_materials/story.md:348:| PSG vs Baseline | +1.33% | (1.6, 2.0, 0.4) | 0.1091 | 3 个 seed 全正，样本数仍小 |
experiments/paper_materials/story.md:349:| exp007a vs PSG | +1.53% | (1.3, 1.6, 1.7) | 0.0061 | ✅ `0.5x loss` 是稳定增益 |
experiments/paper_materials/story.md:355:1. Baseline `56.50%` → PSG `57.83%`
experiments/paper_materials/story.md:358:2. PSG `57.83%` → PSG + `0.5x loss` `59.37%`
experiments/paper_materials/story.md:361:3. PSG + `0.5x loss` `59.37%` ≈ PDS+StopGrad `59.20%`
experiments/paper_materials/story.md:364:4. PSG + GCN(global) `59.33%` ≈ PSG + `0.5x loss` `59.37%`
experiments/paper_materials/story.md:367:5. PSG + GCN(equal_concat) `60.73%` > PSG + GCN(global) `59.33%`
experiments/paper_materials/story.md:371:1. **PSG**: 极简的 backbone 内 pose 注入，稳定提升 Occluded-Duke，并在 Market / Swin-Small 上可复现。
experiments/paper_materials/story.md:397:   - PSG 提升 backbone 全局表征
experiments/paper_materials/story.md:417:- PSG 负责 backbone 级 pose prior
experiments/paper_materials/story.md:513:- PSG 提供 backbone-level pose prior
experiments/paper_materials/story.md:548:| 数据集 | Backbone | Baseline mAP | PSG mAP | Δ |
experiments/paper_materials/story.md:555:→ PSG 在所有组合上均有效，且在遮挡数据集上的增益更大。
experiments/paper_materials/story.md:727:  1. PSG/PAA 负责 suppress noisy context / inject pose prior
experiments/paper_materials/story.md:1406:1. `exp148 PCVT`
experiments/paper_materials/story.md:1415:2. `exp149 SCFA`
experiments/innovation_brainstorm.md:3:## 2026-04-15 PRCV 重审：回到 PSG 主线，重做 multi-stage 消融
experiments/innovation_brainstorm.md:7:1. `PSG` 仍然是当前最稳的主创新点  
experiments/innovation_brainstorm.md:12:2. `2-stage PSG` 有希望作为最终版本，但现有证据还不够干净  
experiments/innovation_brainstorm.md:13:   - `exp009 / exp251 / exp253` 说明 multi-stage **不是普遍自动更优**
experiments/innovation_brainstorm.md:14:   - `exp255 vs exp255b` 又强烈说明：在 `GCN512` 结构分支下，`2-stage PSG` 是关键条件
experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/innovation_brainstorm.md:17:   - `LGPA-D` 像 detached semantic part asset
experiments/innovation_brainstorm.md:19:   - `exp257-259` 已说明 recipe 空间基本耗尽
experiments/innovation_brainstorm.md:26:2. 重新设计 `PSG` / `2-stage PSG` / `3-stage PSG` 的干净对照
experiments/innovation_brainstorm.md:27:3. 把“multi-stage PSG 什么时候有用”这件事说清楚
experiments/innovation_brainstorm.md:31:1. **基础 PSG 消融**
experiments/innovation_brainstorm.md:32:   - no PSG
experiments/innovation_brainstorm.md:33:   - 1-stage PSG
experiments/innovation_brainstorm.md:34:   - 2-stage PSG
experiments/innovation_brainstorm.md:35:   - 3-stage PSG
experiments/innovation_brainstorm.md:44:   - LGPA-only + 1-stage / 2-stage
experiments/innovation_brainstorm.md:45:   - LGPA+GCN + 1-stage / 2-stage
experiments/innovation_brainstorm.md:51:- `PSG` = 主创新
experiments/innovation_brainstorm.md:52:- `2-stage PSG` = scalable extension / 当前最终版本
experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/innovation_brainstorm.md:74:- PSG (特征形成阶段注入 pose) +1.7% mAP，仅 102K params
experiments/innovation_brainstorm.md:79:**2. PSG 对空间级干扰敏感，但通道级正交不干扰**
experiments/innovation_brainstorm.md:80:- PSG + PAB Combo: ❌ (-0.7% vs PSG-only)
experiments/innovation_brainstorm.md:81:- PSG + Part Pooling: ❌ (-0.6% vs PSG-only)
experiments/innovation_brainstorm.md:82:- PSG + Part Supervision (global test): ❌ (-0.7% mAP, -2.1% R1)
experiments/innovation_brainstorm.md:83:- PSG + PCG (通道级 gate): 🟡 持平 (mAP 58.0%, -0.3% vs PSG-only)
experiments/innovation_brainstorm.md:87:- 有效方法排序: PSG(58.3%) > PCG-only(57.8%) > PRA(57.8%) > Part Pooling(57.5%) > PAB(57.4%) > PXA(57.3%) > CAPSG(57.2%)
experiments/innovation_brainstorm.md:88:- PXA/CAPSG 最复杂，效果最差
experiments/innovation_brainstorm.md:89:- **PSG 的极简性就是它的优势**
experiments/innovation_brainstorm.md:91:**4. PSG 跨数据集/backbone 均有效（4090 验证）**
experiments/innovation_brainstorm.md:93:| 数据集 | Backbone | PSG mAP 提升 |
experiments/innovation_brainstorm.md:101:- **梯度干扰是核心瓶颈**: 21 个实验中所有在 PSG 基础上"加东西"的尝试都失败了，原因是同一个 Stage 3 内的模块共享梯度流
experiments/innovation_brainstorm.md:102:- **要突破 PSG，必须用独立的处理路径**，避免梯度干扰
experiments/innovation_brainstorm.md:118:  + PSG               + Pose Part Processing
experiments/innovation_brainstorm.md:130:- **为什么与前轮不同**: exp008 在同一 Stage 3 上做 PSG+Part，梯度干扰。PDS 用独立 Stage 3，各自优化。
experiments/innovation_brainstorm.md:143:- **与 PSG 完全正交**: PSG 调制 feature 幅度（乘法），KP-RPE 调制 attention 路由（加法 bias）
experiments/innovation_brainstorm.md:147:- **论文定位**: 与 PSG 叠加使用 — "spatial gating + structural attention routing"
experiments/innovation_brainstorm.md:153:PSG-enhanced Stage 3 features (12×4, 768ch)
experiments/innovation_brainstorm.md:176:- **与 PSG 的区别**: PSG 是乘法 x*(1+gate)，ControlNet 是加法 x + zero_conv(pose_feat)
experiments/innovation_brainstorm.md:194:- **与 PSG 的区别**: PSG 给所有 token 不同权重，token pruning 直接删除无关 token
experiments/innovation_brainstorm.md:211:2. **exp023: KP-RPE** — 最轻量，与 PSG 正交，快速验证
experiments/innovation_brainstorm.md:215:- 如果 KP-RPE 有效 → PSG + KP-RPE 组合
experiments/innovation_brainstorm.md:228:- **global-only 57.9%**: 接近但未超过 PSG-only 58.3% (-0.4%)
experiments/innovation_brainstorm.md:234:1. **Stage 3 权重解耦确实有效**: PDS global (57.9%) > exp008 PSG+Part same Stage3 (57.7%)，证明独立 Stage 3 保护了 PSG
experiments/innovation_brainstorm.md:248:- PSG-only 58.3% 已经是非常好的单模块结果
experiments/innovation_brainstorm.md:250:- 也许应该接受 PSG 作为核心贡献，转向其他维度（如 test-time fusion、NFC 定制化）
experiments/innovation_brainstorm.md:255:3. **如果 stop_gradient 无效** → 放弃 dual-stream，PSG 作为核心贡献 + 其他正交方向 (KP-RPE, Skeleton GCN)
experiments/innovation_brainstorm.md:262:- **global-only 59.5%**: 超越 PSG-only 58.3% (+1.2%)，超越 baseline +2.9%
experiments/innovation_brainstorm.md:271:1. **共享 Stage 0-2 只被 Global loss 优化** → 特征更适合全局 ID 任务 → PSG 获得更好的输入
experiments/innovation_brainstorm.md:278:1. **PSG**: 在 backbone 内部注入 pose spatial prior → +1.7% (简洁有效)
experiments/innovation_brainstorm.md:279:2. **PDS**: 双分支解耦 Stage 3 权重 → 让 PSG 和 Part 不冲突
experiments/innovation_brainstorm.md:284:- Baseline 56.6% → +PSG 58.3% (+1.7%) → +PDS 57.9% (-0.4%, 有 Part 干扰) → +StopGrad 59.5% (+1.6%, 消除干扰)
experiments/innovation_brainstorm.md:289:**exp022-025 PDS 系列**: PDS+StopGrad 达到 59.5%，但 exp024 (无 PSG 版) 达到 59.2%（仅 -0.3%），暗示提升可能来自训练随机性。多 seed 实验待验证。
experiments/innovation_brainstorm.md:292:- mAP 57.9% vs PSG 58.3% (-0.4%)
experiments/innovation_brainstorm.md:294:- **推论**: PSG 不存在过度依赖问题 → 正则化方向价值有限 → 应该探索"让 loss 函数也感知 pose"的方向
experiments/innovation_brainstorm.md:298:1. ~~完善消融实验~~: exp024 已完成 (PDS+StopGrad 无 PSG = 59.2%)
experiments/innovation_brainstorm.md:300:3. ~~PCRA (Pose-Contrastive Representation Alignment)~~: exp027 验证 mAP 57.8% (-0.5% vs PSG)。17 维 pose signature 不够精确区分姿态差异，引入训练不稳定性。
experiments/innovation_brainstorm.md:303:6. **跨数据集验证**: Market-1501 上跑 PSG (需准备 pose 数据)
experiments/innovation_brainstorm.md:307:**exp027 PCRA**: mAP 57.8%, R1 66.8% (-0.5%/-1.1% vs PSG)
experiments/innovation_brainstorm.md:311:- **推论**: 在 PSG 基础上的所有单点改进（forward/loss/regularization）均已失败。应转向 PDS+StopGrad 的改进或全新范式
experiments/innovation_brainstorm.md:316:1. PSG + forward path 添加: exp008-021 全部失败
experiments/innovation_brainstorm.md:317:2. PSG + 正则化: exp026 SPD 中性
experiments/innovation_brainstorm.md:344:- exp152 (soft MaxSim, tau=0.05) → 远程
experiments/innovation_brainstorm.md:345:- exp152b (hard MaxSim, tau=0.005) → 本地
experiments/innovation_brainstorm.md:352:4. **Evidence**: test-time 已验证; training alignment 待 exp152
experiments/innovation_brainstorm.md:392:1. **exp148 PCVT 早期加速但后期无效**: complementary masking 在前期提供多样性，但 backbone 最终收敛到的表示已经隐式假设"几乎全可见"
experiments/innovation_brainstorm.md:393:2. **exp151 PVAT pvat_acc 不下降**: visibility GT 几乎全 1，predictor 只需猜 "全可见" 就有 83% accuracy
experiments/innovation_brainstorm.md:490:3. PSG + loss 调制: exp027 PCRA 中性
experiments/innovation_brainstorm.md:493:1. **PDS+StopGrad**: 唯一超越 PSG 的方法 (+2.9% mAP)，但 PSG 在其中贡献很小
experiments/innovation_brainstorm.md:511:1. PSG + forward path 添加: exp008-021 全部失败
experiments/innovation_brainstorm.md:512:2. PSG + 正则化: exp026 SPD 中性
experiments/innovation_brainstorm.md:513:3. PSG + loss 调制: exp027 PCRA 中性
experiments/innovation_brainstorm.md:525:**与 PSG 的区别**:
experiments/innovation_brainstorm.md:526:- PSG: 在 feature map 上做乘法调制（soft gate），所有 token 都保留
experiments/innovation_brainstorm.md:539:**预期**: 在遮挡场景中，PGTS 直接排除遮挡/背景 token 的干扰，比 PSG 的 soft gate 更直接有效
experiments/innovation_brainstorm.md:548:**exp029 PSG + Pose-Weighted Pooling (PWP)**: mAP 57.9%, R1 67.5% (-0.4%/-0.4% vs PSG)
experiments/innovation_brainstorm.md:550:- **重要启示**: Post-backbone 的 weighted pooling 在 PSG 已做空间调制后是冗余操作
experiments/innovation_brainstorm.md:551:- PSG 在 Stage 3 内部已完成空间选择 → pooling 阶段再加权只是重复工作
experiments/innovation_brainstorm.md:557:1. PSG + forward path 添加: exp008-021 全部失败
experiments/innovation_brainstorm.md:558:2. PSG + 正则化: exp026 SPD 中性
experiments/innovation_brainstorm.md:559:3. PSG + loss 调制: exp027 PCRA 中性
experiments/innovation_brainstorm.md:561:5. **PSG + post-hoc pooling 改进: exp029 PWP 中性** ← NEW
experiments/innovation_brainstorm.md:564:- PSG 在 Stage 3 内做的空间门控已经是 pose heatmap 利用的最优方式之一
experiments/innovation_brainstorm.md:565:- 所有"在 PSG 之上/之后加东西"的尝试都失败了（21 个 PSG 改进实验 + 1 个 PWP）
experiments/innovation_brainstorm.md:567:- **下一步需要全新的框架思路**，而不是继续在 PSG/PDS 上微调
experiments/innovation_brainstorm.md:584:- exp007a (PSG + 0.5x global loss) 将验证 loss weighting 是否是 PDS 增益的主因
experiments/innovation_brainstorm.md:590:**exp007a (PSG + 0.5x Global Loss Scale)**: mAP **59.5%**, R1 **69.8%**
experiments/innovation_brainstorm.md:594:| exp007 (PSG, 1.0x loss) | 58.3% | 67.9% | +102K | PSG 基线 |
experiments/innovation_brainstorm.md:596:| **exp007a (PSG, 0.5x loss)** | **59.5%** | **69.8%** | **+102K** | **仅改 loss scale** |
experiments/innovation_brainstorm.md:602:2. PSG + loss scaling 是更简洁、更本质的方法
experiments/innovation_brainstorm.md:606:- PSG: backbone 内 pose 注入 (+1.7%)
experiments/innovation_brainstorm.md:609:- 总计: baseline 56.6% → PSG+LS+GCN 60.5% (+3.9% mAP, +4.0% R1)
experiments/innovation_brainstorm.md:613:**exp030a (PSG + Skeleton GCN, 无 PDS)**:
experiments/innovation_brainstorm.md:615:| 模式 | exp030a (PSG+GCN, ~500K) | exp030 (PDS+SG+GCN, ~6.3M) | Δ |
experiments/innovation_brainstorm.md:622:**核心发现**: 独立 Stage 3 完全不必要！共享 Stage 3 的 PSG 特征对 GCN 更好（因为 PSG 已做了 pose-aware modulation），且参数减少 92%。
experiments/innovation_brainstorm.md:624:**R1 大幅提升的可能原因**: exp030a 的 GCN 特征质量更高（来自 PSG 增强的特征），导致 concat 后的组合特征在 Rank-1 检索上大幅优于 PDS 方案。
experiments/innovation_brainstorm.md:627:1. PSG: +102K params, +1.7% mAP
experiments/innovation_brainstorm.md:635:**exp030b (PSG+GCN, w_p=0.01, ≈1.0x loss)**:
experiments/innovation_brainstorm.md:642:| exp007 | PSG only | 1.0x | 58.3% |
experiments/innovation_brainstorm.md:643:| exp007a | PSG only | 0.5x | 59.5% |
experiments/innovation_brainstorm.md:644:| exp030a | PSG+GCN | ~0.5x | 59.8% |
experiments/innovation_brainstorm.md:645:| exp030b | PSG+GCN | ~1.0x | **60.6%** |
experiments/innovation_brainstorm.md:650:1. **PSG 增益 (+1.7%) 的确认需要多种子数据**
experiments/innovation_brainstorm.md:656:- PSG: +1.0~2.0% (需 multi-seed 确认)
experiments/innovation_brainstorm.md:682:| PSG | 58.3% | 57.9% | 57.3% | 57.83±0.50% |
experiments/innovation_brainstorm.md:685:**PSG 增益 +1.33% 确认** — 所有 3 seeds 正向 (paired: +1.6/+2.0/+0.4), p≈0.054
experiments/innovation_brainstorm.md:687:**PDS+SG vs PSG +1.37% 极一致** — paired diffs (1.4/1.3/1.4), p<0.001
experiments/innovation_brainstorm.md:692:1. exp023 (PDS+SG, global) = 59.5% ≈ exp007a (PSG, 0.5x loss) = 59.5% → 增益来自 loss*0.5
experiments/innovation_brainstorm.md:697:- PDS+SG mean = 59.20% vs PSG mean = 57.83% → +1.37% 极一致
experiments/innovation_brainstorm.md:706:- **PSG 是核心贡献** — +1.33% 确认
experiments/innovation_brainstorm.md:712:- PSG: **+1.33%** (3-seed confirmed)
experiments/innovation_brainstorm.md:713:- 0.5x Loss Scaling: **+1.37%** over PSG (PDS+SG multi-seed 暗示, 需 exp007a multi-seed 直接确认)
experiments/innovation_brainstorm.md:914:我们有 ViTPose 热图，直接提供空间重要性 — 但 exp029（PWP）已证明 post-backbone 的 token 加权是冗余操作（PSG 已在 Stage 3 内部做了空间调制）。
experiments/innovation_brainstorm.md:915:**因此 DPSM 思路对我们而言价值有限**。即使我们用热图代替 CLS 相似度做 token 选择，exp029 的负结论已经说明 post-backbone pooling 阶段做任何 pose-weighted 操作都不如 PSG 内部调制有效。
experiments/innovation_brainstorm.md:939:- 但其核心信号（遮挡一致性）与我们的 PSG 目标高度重叠
experiments/innovation_brainstorm.md:999:1. PSG + forward path 添加: exp008-021 全部失败
experiments/innovation_brainstorm.md:1000:2. PSG + 正则化: exp026 SPD 中性
experiments/innovation_brainstorm.md:1001:3. PSG + loss 调制: exp027 PCRA 中性
experiments/innovation_brainstorm.md:1003:5. PSG + post-hoc pooling 改进: exp029 PWP 中性
experiments/innovation_brainstorm.md:1018:- 它在 backbone 层面操作（PSG 成功的关键洞察）
experiments/innovation_brainstorm.md:1032:1. **PSG 已提供充分的遮挡鲁棒性**：PSG 在 Stage 3 内部做 pose-aware spatial gating，模型已经"知道"人体结构。额外的 consistency loss 没有提供新的训练信号。
experiments/innovation_brainstorm.md:1045:1. PSG + forward path 添加: exp008-021 全部失败
experiments/innovation_brainstorm.md:1046:2. PSG + 正则化: exp026 SPD 中性
experiments/innovation_brainstorm.md:1047:3. PSG + loss 调制: exp027 PCRA 中性
experiments/innovation_brainstorm.md:1049:5. PSG + post-hoc pooling 改进: exp029 PWP 中性
experiments/innovation_brainstorm.md:1056:**"在 PSG+GCN 基础上添加训练端辅助 loss"这个方向已被连续 3 次实验否定**：
experiments/innovation_brainstorm.md:1061:**核心教训**：PSG+GCN 的训练已经高度优化。任何辅助 loss 都无法在不干扰主 ID+triplet 目标的情况下提供额外增益。训练端改进这条路线应当彻底关闭。
experiments/innovation_brainstorm.md:1196:这与我们的核心发现一致：**backbone 中有效的 pose 注入方式是 PSG（乘性门控），不是注意力偏置（加性偏移）**。PSG 直接抑制/增强特征值，而注意力偏置只能微调注意力权重——后者的影响力更小。
experiments/innovation_brainstorm.md:1202:| PSG | +1.33% mAP | ✅ | backbone 乘性门控 |
experiments/innovation_brainstorm.md:1212:| Part pooling | 弱正面(+0.9%) | 被 PSG 取代 |
experiments/innovation_brainstorm.md:1216:**我们已经耗尽了"在现有 PSG+GCN 框架上做增量修改"的空间。** 包括：
experiments/innovation_brainstorm.md:1246:2. **Global 模式几乎持平**：说明 XCAD 没有损害 backbone 特征（PSG 工作正常），问题完全出在 keypoint branch 的特征质量上。
experiments/innovation_brainstorm.md:1261:PSG + 0.5x loss + Skeleton GCN（exp030a）仍然是最强方法，无一竞争者能超越。
experiments/innovation_brainstorm.md:1270:**下一步必须是完全不同的思路**，不能再在 PSG+GCN 框架上做任何修改。
experiments/innovation_brainstorm.md:1286:- PSG: +1.33% mAP（已 3-seed 确认）
experiments/innovation_brainstorm.md:1313:| Auxiliary Loss | 7 个 (CSGT/SGMKC/PAMC/PAML/Per-KP Tri/KDL/LKU) | 全部中性或负面 | PSG+GCN 的训练已充分，增量训练信号无效 |
experiments/innovation_brainstorm.md:1320:| Backbone 注入 | PSG 已验证 | +1.33% | 唯一成功的训练端创新 |
experiments/innovation_brainstorm.md:1326:**1. PSG+GCN 框架已达到优化天花板**
experiments/innovation_brainstorm.md:1327:62 个实验中，只有 PSG（backbone injection）、GCN（independent branch）、0.5x loss 和 ROA 是有效的。所有在此基础上的增量修改（loss/attention/weighting/dropout/uncertainty）都失败了。
experiments/innovation_brainstorm.md:1331:- PSG：在 backbone feature 形成阶段引入 pose（之前没有）
experiments/innovation_brainstorm.md:1337:不能再在 PSG/GCN 上做任何修改。需要的是一个全新的信息维度或全新的问题定义。
experiments/innovation_brainstorm.md:1363:1. PSG (backbone injection) — +1.33% mAP ✅
experiments/innovation_brainstorm.md:1379:**exp066**: PSG + GCN + PAA = mAP 61.6% / **R1 74.2%** (+1.63% R1 vs 3-seed!)
experiments/innovation_brainstorm.md:1380:**exp067**: PSG + GCN + PAA + ROA = **mAP 62.0%** / R1 73.7% (+1.27% mAP vs 3-seed!)
experiments/innovation_brainstorm.md:1382:**PAA 的核心创新**：在 PSG 乘性门控之后，用加法 adapter 注入 pose-derived content
experiments/innovation_brainstorm.md:1383:- PSG: `x = x * (1 + gate)` — 调制幅值（哪里重要/哪里抑制）
experiments/innovation_brainstorm.md:1390:1. **Suppress-and-Complete**：PSG 抑制非目标人区域，PAA 只补全目标人的缺失区域
experiments/innovation_brainstorm.md:1441:- PSG (乘性门控) + PAA (加性适配器) + GCN (骨架分支) + 0.5x loss
experiments/innovation_brainstorm.md:1476:  - `PSG` 仍保留 `scene` 路径
experiments/innovation_brainstorm.md:1547:1. **PSG**: 已确认，稳定 +1.33%
experiments/innovation_brainstorm.md:1748:2. **接受当前配置**：PSG+GCN+PAA+ROA 作为训练端最强配置（~62.7% 单 seed），SGCFR 作为测试端独特创新（+2.6%）
experiments/innovation_brainstorm.md:1784:- 与 PSG 的区别: PSG 是 soft gating，OGFR 是 hard replacement with learned tokens
experiments/innovation_brainstorm.md:2430:- PSG/PAA/SKC: 修改 **what** (特征本身)
experiments/innovation_brainstorm.md:2441:1. PSG + forward path 添加 (21 实验)
experiments/innovation_brainstorm.md:2442:2. PSG + 正则化/dropout
experiments/innovation_brainstorm.md:2443:3. PSG + loss 调制 (PCRA, CSGT, GKD, DPF, etc.)
experiments/innovation_brainstorm.md:2559:1. Pose-aware backbone injection (PSG)
experiments/innovation_brainstorm.md:2575:- exp166 per-token training + pooled test: **63.1/73.9** ← 实际运行结果
experiments/innovation_brainstorm.md:2576:- exp166 per-token training + per-token concat test: **61.8/72.5** ← 修复后 test.py 评估
experiments/innovation_brainstorm.md:2588:3. DPTL (exp167) 仍然合理：self-attention refined tokens → better pooled feature
experiments/innovation_brainstorm.md:2589:4. 17-token (exp168) 的 test 也应该用 pooled，不用 per-token concat
experiments/innovation_brainstorm.md:2591:### exp166 完整结果表
experiments/innovation_brainstorm.md:2595:| exp166 (per-token+PLBOA) ep120 | 63.1% | 73.9% | 86.1% | 89.5% | pooled test (best) |
experiments/innovation_brainstorm.md:2596:| exp166 (per-token+PLBOA) ep100 | 62.9% | 74.5% | 86.2% | 89.3% | peak R1 |
experiments/innovation_brainstorm.md:2597:| exp166 bugfix (concat test) ep120 | 61.8% | 72.5% | 85.0% | 89.1% | per-token concat test |
experiments/innovation_brainstorm.md:2598:| exp166r (per-token, 无PLBOA) ep120 | 60.3% | 72.8% | 83.6% | 87.0% | PLBOA 贡献: +2.8/+1.1 |
experiments/innovation_brainstorm.md:2608:| PSG (backbone pose gate) | 低 — 无直接先例 | Swin block 内乘法 gate |
experiments/innovation_brainstorm.md:2628:| +PSG | 58.3% | 67.9% | 贡献 1: backbone injection |
experiments/innovation_brainstorm.md:2658:## 2026-03-26: OA-SD 系列实验总结 + Global-Only 新发现
experiments/innovation_brainstorm.md:2660:### OA-SD 核心特性（exp191-194 消融）
experiments/innovation_brainstorm.md:2662:1. **OA-SD + CE = 强正向**: +2.9/+2.6 vs CE base (exp191)
experiments/innovation_brainstorm.md:2663:2. **OA-SD + SupCon (all-token) = 负向**: -0.7/-0.4 (exp188) — 梯度冲突
experiments/innovation_brainstorm.md:2664:3. **EMA decay 不敏感**: 0.99 vs 0.999 最终差异 <1% (exp192)
experiments/innovation_brainstorm.md:2665:4. **Loss weight 不敏感**: 1.0 vs 2.0 最终差异 <1% (exp194)
experiments/innovation_brainstorm.md:2666:5. **OA-SD + 3-view 是 additive**: exp193 = 64.4/76.5 vs exp190 = 64.2/75.6 (+0.2/+0.9)
experiments/innovation_brainstorm.md:2667:6. **OA-SD late-stage boost**: ep40 前拖累 → ep40 crossover → ep60+ 大幅正向
experiments/innovation_brainstorm.md:2669:### 关键新发现：OA-SD Global-Only 解决 SupCon 梯度冲突
experiments/innovation_brainstorm.md:2671:**问题**: OA-SD all-token distillation 与 SupCon 在 per-token features 上产生梯度冲突
experiments/innovation_brainstorm.md:2674:- OA-SD distillation 鼓励 student tokens 逼近 teacher tokens（不管 ID）
experiments/innovation_brainstorm.md:2677:**解决方案**: OA-SD GLOBAL_ONLY — 只在 global (GAP后) feature 上做 distillation
experiments/innovation_brainstorm.md:2678:- Global feature: CE + triplet + OA-SD distill（三者协同）
experiments/innovation_brainstorm.md:2679:- Per-token features: CE + triplet + SupCon（三者协同，无 OA-SD 干扰）
experiments/innovation_brainstorm.md:2681:**验证** (exp195): SupCon + OA-SD global-only ep70=60.2/73.4
experiments/innovation_brainstorm.md:2682:- 没有出现 exp188 的负向效应
experiments/innovation_brainstorm.md:2683:- R1 稳定领先 CE+OA-SD（SupCon 的 R1 优势保持）
experiments/innovation_brainstorm.md:2687:- 全局遮挡不变性 (OA-SD) 和局部判别力 (SupCon) 在不同特征级别独立优化
experiments/innovation_brainstorm.md:2688:- 消融链: exp188 (冲突) → exp195 (分离) 是清晰的证据
experiments/innovation_brainstorm.md:2694:| 1 | exp187 | 3-view + SupCon | 64.9% | 76.6% |
experiments/innovation_brainstorm.md:2695:| 2 | exp193 | 3-view + OA-SD + CE | 64.4% | 76.5% |
experiments/innovation_brainstorm.md:2696:| 3 | exp190 | 3-view + CE | 64.2% | 75.6% |
experiments/innovation_brainstorm.md:2697:| 4 | exp176 | SupCon (1-view) | 64.1% | 75.5% |
experiments/innovation_brainstorm.md:2698:| 5 | exp194 | OA-SD + CE (w=2.0) | 63.4% | 74.8% |
experiments/innovation_brainstorm.md:2699:| 6 | exp191 | OA-SD + CE | 63.2% | 75.4% |
experiments/innovation_brainstorm.md:2700:| 7 | exp166 | CE baseline (full) | 63.1% | 73.9% |
experiments/innovation_brainstorm.md:2702:### 待验证: exp196 终极配置
experiments/innovation_brainstorm.md:2704:3-view + SupCon + OA-SD global-only — 预计 65.0-65.5/77.0-77.5
experiments/innovation_brainstorm.md:2722:- **直接解决 OA-SD vs SupCon 冲突**: 不 match 个体特征(会冲突) → match 关系结构(不冲突)
experiments/innovation_brainstorm.md:2736:- STM (Token Mixup): 只加速不改善天花板 (exp197/198)
experiments/innovation_brainstorm.md:2737:- OA-SD + SupCon: 互斥，即使 global-only 也无法叠加 (exp195/196)
experiments/innovation_brainstorm.md:2739:### 选定方向: OA-RD (exp199)
experiments/innovation_brainstorm.md:2754:| exp188 | OA-SD all-token + SupCon | SupCon | -0.7/-0.4 | ❌ 梯度冲突 |
experiments/innovation_brainstorm.md:2755:| exp195 | OA-SD global-only + SupCon | SupCon | ~-2.8 mAP | ❌ 信号太弱 |
experiments/innovation_brainstorm.md:2756:| exp196 | OA-SD global-only + SupCon + 3v | SupCon | -2.5/-1.4 | ❌ 同上 |
experiments/innovation_brainstorm.md:2757:| exp199 | OA-RD relational + SupCon + 3v | SupCon | ~-1.5/-3.4 | ❌ 关系级也冲突 |
experiments/innovation_brainstorm.md:2758:| exp191 | OA-SD all-token + CE | CE | +2.9/+2.6 | ✅ CE 兼容 |
experiments/innovation_brainstorm.md:2759:| exp193 | OA-SD all-token + CE + 3v | CE | +0.2/+0.9 | ✅ CE 兼容 |
experiments/innovation_brainstorm.md:2760:| exp200 | OA-RD relational + CE | CE | ~-1.0/-3.4 | ❌ OA-RD 不如 OA-SD |
experiments/innovation_brainstorm.md:2765:2. **OA-SD (feature distillation) 在 CE 路线有效**，但 OA-RD (relational) 在 CE 路线不如 OA-SD
experiments/innovation_brainstorm.md:2767:   - SupCon 路线: exp187 = 64.9/76.6 (不加任何 distillation)
experiments/innovation_brainstorm.md:2768:   - OA-SD 路线: exp193 = 64.4/76.5 (不加 SupCon)
experiments/innovation_brainstorm.md:2788:**MaxSim Hybrid 在 exp206 checkpoint 上无需重训即可获得 +1.8% mAP！**
experiments/innovation_brainstorm.md:2822:1. **Swin-Small + GCN+PAA+OA-SD + maxsim_hybrid = 72.4%** (exp210b with PKC=0.05)
experiments/innovation_brainstorm.md:2823:2. 训练端改进: 目前所有尝试均未超过 OA-SD-only ceiling
experiments/innovation_brainstorm.md:2834:| 实验 | 方法 | detach? | vs OA-SD-only |
experiments/innovation_brainstorm.md:2836:| exp210 | PKC w=0.5 | detached | 灾难 3.6% |
experiments/innovation_brainstorm.md:2837:| exp210b | PKC w=0.05 | detached | 无效 (=baseline) |
experiments/innovation_brainstorm.md:2838:| exp211 | MST w=0.5 | detached | 完全无效 (所有 loss 一致) |
experiments/innovation_brainstorm.md:2839:| exp213 | PKC+MST 组合 | detached | 灾难 40.6% |
experiments/innovation_brainstorm.md:2840:| exp215 | BA-PKC non-detach | non-det | 灾难 0.5% |
experiments/innovation_brainstorm.md:2841:| exp217 | OERL non-detach cosine | non-det | `62.2/75.2`，相对 `exp191 63.2/75.4` 为 `-1.0/-0.2` |
experiments/innovation_brainstorm.md:2842:| exp218 | PACI prototype bank | detached | `61.9/74.2`，相对 `exp191 63.2/75.4` 为 `-1.3/-1.2` |
experiments/innovation_brainstorm.md:2843:| exp219 | PACI without OA-SD | detached | 远程日志当前只确认到 `ep30=51.9/64.9`，早期即落后 baseline `52.2/65.2` |
experiments/innovation_brainstorm.md:2844:| exp220 | GSPB gradient scale 5% | 5% scale | `62.9/74.3`，相对 `exp191 63.2/75.4` 为 `-0.3/-1.1` |
experiments/innovation_brainstorm.md:2853:早期只看 `OA-SD / OERL / PACI` 三条 Tiny 线时，`maxsim_hybrid` 确实都落在 `64.1~64.3`；
experiments/innovation_brainstorm.md:2858:| OA-SD-only | 63.2 | 64.2 |
experiments/innovation_brainstorm.md:2859:| OERL+OA-SD | 62.2 | 64.3 |
experiments/innovation_brainstorm.md:2860:| PACI+OA-SD | 61.9 | 64.1 |
experiments/innovation_brainstorm.md:2861:| GSPB+OA-SD | 62.9 | 64.6 |
experiments/innovation_brainstorm.md:2862:| PADPQ+OA-SD | 63.7 | 63.9 |
experiments/innovation_brainstorm.md:2865:1. `MaxSim` 对 OA-SD 本身仍然有效（`63.2 -> 64.2`）
experiments/innovation_brainstorm.md:2871:exp220 (scale=0.05) 完整对照:
experiments/innovation_brainstorm.md:2873:| Epoch | GSPB mAP | OA-SD mAP | delta |
experiments/innovation_brainstorm.md:2885:**GCN+PAA+OA-SD 在 Tiny 上已达极限 ~63%。** 需要:
experiments/innovation_brainstorm.md:2895:- exp229: BT-PKD constant (Tiny) → -1.0/-0.4
experiments/innovation_brainstorm.md:2896:- exp230: BT-PKD constant (Small, no PAUG) → ~0/-0.7 (ep110)
experiments/innovation_brainstorm.md:2897:- exp231: BT-PKD cosine decay (Tiny) → -1.5/-1.1
experiments/innovation_brainstorm.md:2898:- exp232: BT-PKD cosine decay (Small) → terminated ep37
experiments/innovation_brainstorm.md:2927:2. **改善 Global branch** 而不是 Part branch — PSG 已证明有效，还有什么可以在 Global 上做？
experiments/innovation_brainstorm.md:2940:| Pose injection | Additive prompt tokens | PSG multiplicative gating |
experiments/innovation_brainstorm.md:2978:**唯一有效的方向是改变 backbone 本身**: PSG (+1.7%), OA-SD (+2-3%), PLBOA (+1.5%)
experiments/innovation_brainstorm.md:3000:PPA 替换 GCN Part branch，但保留 PSG + OA-SD + PLBOA。
experiments/innovation_brainstorm.md:3009:1. **只有 backbone 修改有效**: PSG (+1.7%), OA-SD (+2-3%), LGPA-D (+2.1%)
experiments/innovation_brainstorm.md:3032:1. **短期**: 完成 exp249 (Small LGPA-D+GCN) → 可能 73-74% mAP
experiments/innovation_brainstorm.md:3034:3. **论文策略**: LGPA-D (CLIP 语义 part assignment) 作为核心贡献, 配合完整 pipeline + 详细消融
experiments/results.md:21:| **007** | **Pose Spatial Gate in Backbone (PSG)** | **58.3%** | **67.9%** | **80.8%** | **84.9%** | **mAP+1.7%, R1+1.4%** | **✅ 3-seed mean = 57.83% / 67.13%，所有 seed 均优于 baseline，PSG 有效** |
experiments/results.md:22:| 008 | PSG + Part Pooling (part_only test) | 57.7% | 66.0% | 78.3% | 82.8% | mAP+1.1%, R1-0.5% | 🟡 组合不叠加, 低于 PSG-only. Part pooling 拖累全局特征 |
experiments/results.md:23:| 009 | Multi-stage PSG (Stage 2+3) | 58.3% | 67.2% | 81.2% | 85.2% | mAP+1.7%, R1+0.7% | 🟡 mAP 匹配 exp007, R1 略低(-0.7%), R5/R10 略优. 多 156K params 无显著收益 |
experiments/results.md:24:| 010 | PSG + Backbone Freeze 5ep | 12.5%* | 17.5%* | 30.4%* | 36.7%* | — | ❌ ep30 终止. 冻结 backbone 导致灾难性特征损坏 |
experiments/results.md:25:| 011 | PSG Stage 3 (200 epochs) | 58.3% | 67.6% | 81.1% | 85.3% | mAP+1.7%, R1+1.1% | 🟡 与 exp007(120ep) mAP 相同, 75% 更多训练时间无收益 |
experiments/results.md:26:| 012 | Pose Attention Bias (PAB, Stage 3) | 57.4% | 67.3% | 81.4% | 86.2% | mAP+0.8%, R1+0.8% | 🟡 有效但弱于 PSG. 仅 5.4K params. 证明 feature gate > attn bias |
experiments/results.md:27:| 013 | PSG + PAB Combo (Stage 3) | 57.6% | 67.2% | 81.3% | 84.4% | mAP+1.0%, R1+0.7% | ❌ 双重注入互相干扰, 不如 PSG-only(-0.7% mAP). PAB 拖累 PSG 收敛 |
experiments/results.md:28:| 014 | PSG + Part Supervision (global test) | 57.6% | 65.8% | 77.9% | 82.6% | mAP+1.0%, R1-0.7% | ❌ 用 exp008 checkpoint 直接验证。Part supervision 梯度损害 PSG global feature |
experiments/results.md:29:| 015 | PSG Spatial (3×3 DWConv) | 58.3% | 67.1% | 81.4% | 85.8% | mAP+1.7%, R1+0.6% | 🟡 mAP 匹配 exp007, R1 低 0.8%. 3×3 conv 冗余，1×1 已足够 |
experiments/results.md:30:| 016 | PSG + Pose-Guided Erasing (PGE) | 54.8% | 65.0% | 77.7% | 82.2% | mAP-1.8%, R1-1.5% | ❌ PGE 替代 RE 严重有害 (-3.5% vs exp007). 身体部件级擦除过强 |
experiments/results.md:31:| 017 | PSG + Pose Channel Gate (PCG) | 58.0% | 67.3% | 80.9% | 85.3% | mAP+1.4%, R1+0.8% | 🟡 与 exp007 持平(-0.3% mAP). 通道级正交不干扰但无额外收益 |
experiments/results.md:32:| 018 | PCG-only (无 PSG) | 57.8% | 67.7% | 81.4% | 86.2% | mAP+1.2%, R1+1.2% | 🟡 PCG 有独立效果(+1.2%), 但低于 PSG(-0.5%). PSG+PCG 不叠加 |
experiments/results.md:33:| 019 | Pose Cross-Attention (PXA, 替代 PSG) | 57.3% | 66.9% | 80.4% | 85.3% | mAP+0.7%, R1+0.4% | 🟡 有效但弱于 PSG(-1.0% mAP). Cross-attn 过拟合严重, 简单门控更好 |
experiments/results.md:34:| 020 | PSG + Pose Reconstruction Aux (PRA) | 57.8% | 67.3% | 80.3% | 84.7% | mAP+1.2%, R1+0.8% | 🟡 中性. 辅助重建任务不改善 PSG(-0.5% mAP). 后期梯度干扰导致锯齿波动 |
experiments/results.md:35:| 021 | Content-Adaptive PSG (CAPSG) | 57.2% | 66.0% | 80.5% | 85.2% | mAP+0.6%, R1-0.5% | ❌ Content-dependent gate 弱于静态 PSG(-1.1% mAP). 过度参数化, PSG 简洁性即优势 |
experiments/results.md:36:| 022-g | PDS global-only (独立Stage3, PSG全局分支) | 57.9% | 67.1% | 80.0% | 84.2% | mAP+1.3%, R1+0.6% | 🟡 PSG 增益大部分保留(-0.4% vs exp007), Stage 3 解耦有效 |
experiments/results.md:44:| 024-g | PDS+StopGrad noPSG global-only | 59.2% | 68.7% | 82.0% | 86.1% | mAP+2.6%, R1+2.2% | 单 seed 高点；后续 multi-seed 不支持“PSG 贡献很小”这一强结论 |
experiments/results.md:45:| 024-cs | PDS+StopGrad noPSG concat_scaled | 59.0% | 68.3% | 81.6% | 85.7% | mAP+2.4%, R1+1.8% | |
experiments/results.md:46:| 024-eq | PDS+StopGrad noPSG equal_concat | 57.1% | 65.4% | 78.8% | 83.1% | mAP+0.5%, R1-1.1% | |
experiments/results.md:47:| 024-p | PDS+StopGrad noPSG part-only | 56.4% | 64.9% | 77.9% | 82.4% | mAP-0.2%, R1-1.6% | |
experiments/results.md:52:| 026 | PSG + Stochastic Pose Dropout (p=0.3) | 57.9% | 66.2% | 80.5% | 85.2% | mAP+1.3%, R1-0.3% | 🟡 -0.4% vs exp007. SPD 正则化未超越 PSG, pose 信号一致有用 |
experiments/results.md:53:| 027 | PSG + PCRA (alpha=0.2, loss 距离调制) | 57.8% | 66.8% | 81.0% | 85.3% | mAP+1.2%, R1+0.3% | 🟡 -0.5% mAP vs exp007. Pose similarity 调制 triplet 距离中性偏负 |
experiments/results.md:55:| 029 | PSG + Pose-Weighted Pooling (PWP) | 57.9% | 67.5% | 81.1% | 85.3% | mAP+1.3%, R1+1.0% | 🟡 vs exp007(58.3%)-0.4%. PWP 替换 GAP 为 pose-weighted pooling, 效果中性. PSG 已做了空间选择, post-hoc weighting 冗余 |
experiments/results.md:60:| **007a** | **PSG + 0.5x Global Loss Scale** | **59.5%** | **69.8%** | **81.9%** | **86.0%** | **mAP+2.9%, R1+3.3%** | **✅ 3-seed mean = 59.37% / 69.43%；相对 PSG 稳定 +1.53%，且与 exp023-g 无显著差异** |
experiments/results.md:61:| 030a-g | PSG + Skeleton GCN (global-only, 无 PDS) | 59.8% | 69.5% | 81.9% | 86.1% | mAP+3.2%, R1+3.0% | 3-seed mean = 59.33% / 68.87%，≈ exp007a；说明 GCN 分支对 global 基本中性 |
experiments/results.md:62:| 030a-cs | PSG + Skeleton GCN (concat_scaled, 无 PDS) | 60.5% | 73.7% | 85.0% | 88.1% | mAP+3.9%, R1+7.2% | 3-seed mean = 60.20% / 73.13%，稳定优于 030a-global，但弱于 equal_concat |
experiments/results.md:63:| **030a-eq** | **PSG + Skeleton GCN (equal_concat, 无 PDS)** | **61.1%** | **73.7%** | **85.2%** | **87.8%** | **mAP+4.5%, R1+7.2%** | **✅ 3-seed mean = 60.73% / 72.57%；对 030a-global 稳定 +1.40 mAP，是当前最强且已确认的无后处理模式** |
experiments/results.md:64:| 030a-p | PSG + Skeleton GCN (gcn_only, 无 PDS) | 58.2% | 72.9% | 83.3% | 86.6% | mAP+1.6%, R1+6.4% | 3-seed mean = 57.97% / 71.77%；branch 本身强，但不如 fusion |
experiments/results.md:65:| 030b-g | PSG+GCN w_p=0.01 (global-only) | **60.6%** | 71.0% | 83.8% | 87.3% | mAP+4.0%, R1+4.5% | 单 seed 高点；现主要作为“低权重时 branch 几乎未学好”的反例，不宜再单独拿它否定 loss scaling |
experiments/results.md:66:| 030b-cs | PSG+GCN w_p=0.01 (concat_scaled) | 59.4% | 72.9% | 83.9% | 87.3% | mAP+2.8%, R1+6.4% | 单 seed；核心信息是低权重时 concat 无法稳定超越 global |
experiments/results.md:67:| 030b-eq | PSG+GCN w_p=0.01 (equal_concat) | 60.5% | 73.0% | 84.4% | 88.3% | mAP+3.9%, R1+6.5% | 单 seed；与 global 接近，说明未训练好的 branch 贡献有限 |
experiments/results.md:68:| 030b-p | PSG+GCN w_p=0.01 (gcn_only) | 56.9% | 70.9% | 82.4% | 86.2% | mAP+0.3%, R1+4.4% | 图传播几乎未训练，但 keypoint pooling 本身仍强 |
experiments/results.md:69:| 032-g | PSG + Keypoint Pooling Only (global-only) | 59.8% | 70.0% | 81.7% | 85.4% | mAP+3.2%, R1+3.5% | 单 seed；支持“branch 不解释 global 提升”，但精确结论应以 030a multi-seed 为准 |
experiments/results.md:70:| 032-cs | PSG + Keypoint Pooling Only (concat_scaled) | 59.3% | 72.4% | 85.1% | 88.4% | mAP+2.7%, R1+5.9% | 单 seed；说明 keypoint pooling 本身就有较强 fusion 价值 |
experiments/results.md:71:| 032-eq | PSG + Keypoint Pooling Only (equal_concat) | 60.2% | 72.5% | 85.1% | 88.3% | mAP+3.6%, R1+6.0% | 单 seed；现在更适合作为“keypoint pooling 强基线”的证据，而不是单独量化 GCN 增益 |
experiments/results.md:72:| 032-p | PSG + Keypoint Pooling Only (gcn_only 测试模式) | 54.7% | 69.9% | 82.4% | 86.0% | mAP-1.9%, R1+3.4% | 无图传播仍有高 R1，证明关键点采样+置信度池化本身就是强基线 |
experiments/results.md:73:| 035a | PSG+GCN score weight (bundled sanity check) | 61.1% | 73.8% | 85.1% | 87.9% | mAP+4.5%, R1+7.3% | = exp030a seed1234 结果（61.1/72.9），含 target-aware+vis aug fix, 无 regression |
experiments/results.md:74:| 035b | PSG+GCN score*visibility weight | 60.4% | 71.6% | 84.8% | 87.9% | mAP+3.8%, R1+5.1% | ❌ vs 035a: -0.7% mAP, -2.2% R1。当前只说明 `score*visibility` 未带来收益，不能上升为整条 visibility 路线结论 |
experiments/results.md:75:| 007b | PSG + 0.25x Global Loss Scale | 58.3% | 67.6% | 80.0% | 84.9% | mAP+1.7%, R1+1.1% | = exp007(1.0x)! 收敛慢但最终追平 |
experiments/results.md:76:| 007c | PSG + 0.75x Global Loss Scale | 58.6% | 67.6% | 81.6% | 85.6% | mAP+2.0%, R1+1.1% | 单 seed；现阶段不能再用 0.25x/0.75x 的单次结果否定 0.5x，多种子只确认了 0.5x vs 1.0x |
experiments/results.md:78:| 036 | PSG+GCN + Per-Keypoint Triplet Loss | 60.6% | 73.1% | 84.5% | 88.2% | mAP+4.0%, R1+6.6% | ❌ vs 035a: -0.5% mAP, -0.7% R1。该编号已偏离原 visibility 路线，实际属于 `exp035` 之后的 branch 内部探索 |
experiments/results.md:79:| 037 | PSG+GCN + Learnable Keypoint Attention | 60.7% | 71.7% | 83.8% | 87.1% | mAP+4.1%, R1+5.2% | ❌ vs 035a: -0.4% mAP, -2.1% R1。该编号已偏离原 visibility 路线，LKA 未显示稳定正增益 |
experiments/results.md:80:| 039a | PSG+GCN + CVK retrieval (`cvk_only`) | 59.3% | 72.9% | 84.1% | 87.1% | mAP+2.7%, R1+6.4% | 测试时诊断；vs 035a: -1.8% mAP, -0.9% R1。纯共同可见关键点距离不足以替代 `equal_concat` |
experiments/results.md:81:| 039b | PSG+GCN + CVK retrieval (`cvk_hybrid`) | 61.9% | 73.2% | 85.2% | 88.5% | mAP+5.3%, R1+6.7% | 测试时诊断；vs 035a: +0.8% mAP, -0.6% R1。共同可见关键点更适合作为 global 的 pair-specific 补充 |
experiments/results.md:89:| 047 | PSG+GCN + CSGT (Common-Support-Guided Triplet) | — | — | — | — | ❌ 中止 | Epoch 60 中断无 checkpoint。根本问题：pos/neg overlap 几乎相同（≈0.65），机制无法区分正负 pair。pos_fallback≈0.7 说明大部分退化为标准 triplet |
experiments/results.md:90:| 048 | PSG+GCN + SGMKC (Skeleton-Guided Masked Keypoint Completion) | 58.9% | 72.1% | 84.2% | 87.5% | mAP+2.3%, R1+5.6% | ❌ 负面 (-1.6% vs exp030a)。SGMKC loss 与 ID 分类存在梯度冲突，GCN 容量不足以同时完成两个任务 |
experiments/results.md:91:| 050 | PSG+GCN + PAMC (Pose-Aware Masking Consistency) | 60.7% | 72.2% | 83.7% | 87.3% | mAP+4.1%, R1+5.7% | 🟡 中性 (vs exp030a-eq 3-seed: -0.03% mAP, -0.37% R1)。Consistency loss 未提供额外增益。连续第 3 个辅助 loss 方向失败 |
experiments/results.md:92:| 051-eq | PSG+GCN + PAML (Pose-Aware Metric Learning, equal_concat) | 60.7% | 72.7% | 84.6% | 88.2% | mAP+4.1%, R1+6.2% | 🟡 中性 (vs exp030a-eq 3-seed: -0.03% mAP, +0.13% R1)。逐关键点距离训练未带来增益。连续第 4 个辅助 loss 失败 |
experiments/results.md:93:| 051-cvk | PSG+GCN + PAML (cvk_hybrid) | 62.0% | 73.6% | 85.1% | 88.4% | — | 🟡 vs exp030a CVK (61.9%/73.2%): +0.1%/+0.4%。训练-测试 metric alignment 假设未得到验证 |
experiments/results.md:94:| 052-eq | PSG+GCN + KP-RPE (equal_concat) | 61.0% | 72.7% | 84.4% | 87.6% | mAP+4.4%, R1+6.2% | 🟡 中性 (vs exp030a-eq 3-seed: +0.27% mAP, +0.13% R1，在方差范围内)。mAP 训练全程 10/12 checkpoint 为正(均值+0.76%)，但最终结果在方差内 |
experiments/results.md:95:| 052-g | PSG+GCN + KP-RPE (global) | 59.5% | 68.4% | 81.6% | 85.7% | mAP+2.9%, R1+1.9% | 🟡 vs exp030a-g(59.8/69.5): -0.3%/-1.1%。KP-RPE 未改善 backbone 特征 |
experiments/results.md:96:| 052-cvk | PSG+GCN + KP-RPE (cvk_hybrid) | 61.7% | 72.6% | 84.3% | 88.2% | — | 🟡 vs exp030a CVK(61.9/73.2): -0.2%/-0.6%。KP-RPE + CVK 无正交增益 |
experiments/results.md:97:| 053-eq | PSG + XCAD (equal_concat) | 59.7% | 70.8% | 82.0% | 86.2% | mAP+3.1%, R1+4.3% | ❌ vs exp030a-eq 3-seed: -1.03% mAP, -1.77% R1。Cross-attention decoder 劣于 GCN |
experiments/results.md:98:| 053-g | PSG + XCAD (global) | 59.2% | 68.6% | 81.6% | 85.9% | mAP+2.6%, R1+2.1% | 🟡 vs exp030a-g 3-seed: -0.13%/-0.27%，几乎持平 |
experiments/results.md:99:| 053-cvk | PSG + XCAD (cvk_hybrid) | 60.7% | 71.8% | 82.9% | 86.9% | — | ❌ vs exp030a CVK(61.9/73.2): -1.2%/-1.4% |
experiments/results.md:100:| **054-eq** | **PSG+GCN + PGAM (equal_concat)** | **61.1%** | **73.8%** | **85.1%** | **87.9%** | **mAP+4.5%, R1+7.3%** | **🟢 vs exp030a-eq 3-seed: +0.37% mAP, +1.23% R1。首个 PSG+GCN 上正向叠加模块！** |
experiments/results.md:101:| 054-g | PSG+GCN + PGAM (global) | 59.8% | 69.5% | 81.9% | 86.1% | mAP+3.2%, R1+3.0% | 🟡 vs exp030a-g 3-seed: +0.47%/+0.63%，方差内 |
experiments/results.md:102:| 054-cvk | PSG+GCN + PGAM (cvk_hybrid) | 61.9% | 73.2% | 85.2% | 88.5% | — | 🟡 vs exp030a CVK: 0.0%/0.0%，完全持平 |
experiments/results.md:103:| 055-eq | PSG+GCN + PGAM t=0.5 (eq_concat) | 61.2% | 73.5% | 85.2% | 88.6% | mAP+4.6%, R1+7.0% | 🟢 vs exp054: ≈持平。阈值不敏感 |
experiments/results.md:104:| 055-g | PSG+GCN + PGAM t=0.5 (global) | 60.3% | 70.2% | 82.2% | 87.1% | mAP+3.7%, R1+3.7% | 🟢 vs exp054-g: +0.5%/+0.7%。t=0.5 global 更好 |
experiments/results.md:105:| 056-eq | PSG+GCN + PGAM S2+S3 (eq_concat) | 61.1% | 73.7% | 85.2% | 88.6% | mAP+4.5%, R1+7.2% | 🟡 vs exp054: ≈持平。多 Stage 无额外增益 |
experiments/results.md:106:| 057-eq | PSG+GCN + KDL w=0.1 (eq_concat) | 61.0% | 73.3% | 84.6% | 87.9% | mAP+4.4%, R1+6.8% | 🟡 中性。vs exp030a 3-seed: +0.27%/+0.73%。Dissimilar loss 无效 |
experiments/results.md:107:| **058-eq** | **PSG+GCN + ROA (equal_concat)** | **61.8%** | **72.8%** | **85.2%** | **88.3%** | **mAP+5.2%, R1+6.3%** | **🟢🟢 历史最高 mAP！vs 3-seed: +1.07%/+0.23%。超出方差！** |
experiments/results.md:108:| **058-g** | **PSG+GCN + ROA (global)** | **60.8%** | **70.0%** | **83.0%** | **87.0%** | **mAP+4.2%, R1+3.5%** | **🟢🟢 vs 3-seed: +1.47%/+1.13%。全局特征也显著提升！** |
experiments/results.md:109:| 059-eq | PSG+GCN + ROA + PGAM (eq_concat) | 61.8% | 72.8% | 85.2% | 88.3% | mAP+5.2%, R1+6.3% | 🟡 与 exp058 精确相同。PGAM 与 ROA 完全冗余 |
experiments/results.md:110:| 060-eq | PSG+GCN + PA-ROA (eq_concat) | 61.6% | 72.5% | 84.5% | 87.9% | mAP+5.0%, R1+6.0% | 🟡 vs random ROA: -0.2%/-0.3%。Pose-guided 放置不优于随机 |
experiments/results.md:111:| 061-eq | PSG+GCN + GKD 30% (eq_concat) | 60.8% | 73.0% | 84.3% | 87.8% | mAP+4.2%, R1+6.5% | 🟡 中性。vs 3-seed: +0.07%/+0.43%。GCN dropout 无效 |
experiments/results.md:112:| 062-eq | PSG+GCN + LKU (eq_concat) | 60.7% | 71.2% | 84.1% | 87.4% | mAP+4.1%, R1+4.7% | ❌ 负面。vs 3-seed: -0.03%/-1.37%。Learned uncertainty 损害 R1 |
experiments/results.md:113:| 063-eq | PSG + PTD (eq_concat) | 56.7% | 65.3% | 78.3% | 82.4% | mAP+0.1%, R1-1.2% | ❌❌ 严重负面。vs 3-seed: -4.03%/-7.27%。Pose-Token 无法替代 GCN |
experiments/results.md:114:| 058+nfc | PSG+GCN+ROA + NFC (eq_concat) | **64.0%** | **74.3%** | 84.3% | 87.2% | — | 🟢 NFC test-time boost on ROA。最强结果（含 NFC）|
experiments/results.md:115:| 058+cvk | PSG+GCN+ROA + CVK (cvk_hybrid) | 62.7% | 73.5% | 85.4% | 88.7% | — | 🟢 CVK 在 ROA 上也有效 |
experiments/results.md:116:| 064-eq | PSG+GCN + PKE (eq_concat) | 61.0% | 73.1% | 84.5% | 87.7% | mAP+4.4%, R1+6.6% | 🟡 微弱正向。vs 3-seed: +0.27%/+0.53%。Precision weighting 安全但不显著 |
experiments/results.md:117:| 065-eq | PSG+GCN + PKE+ROA (eq_concat) | 61.9% | 73.2% | 84.5% | 88.2% | mAP+5.3%, R1+6.7% | 🟡 ≈ROA alone。PKE+ROA 不正交 |
experiments/results.md:118:| **066-eq** | **PSG+GCN + PAA (eq_concat)** | **61.6%** | **74.2%** | **85.4%** | **88.4%** | **mAP+5.0%, R1+7.7%** | **🟢🟢🟢 历史最高 R1！vs 3-seed: +0.87%/+1.63%。训练端创新！** |
experiments/results.md:119:| **067-eq** | **PSG+GCN + PAA+ROA (eq_concat)** | **62.0%** | **73.7%** | **85.2%** | **88.6%** | **mAP+5.4%, R1+7.2%** | **🟢🟢🟢 历史最高 mAP！PAA+ROA 部分正交叠加。vs 3-seed: +1.27%/+1.13%** |
experiments/results.md:120:| 068-eq | PSG+GCN + RR-PAA (eq_concat) | 61.2% | 72.9% | 85.4% | 88.3% | mAP+4.6%, R1+6.4% | 🟡 vs PAA uniform: -0.4%/-1.3%。路由不优于 uniform |
experiments/results.md:121:| 069-eq | PSG+GCN + PAA b128 (eq_concat) | 61.3% | 74.6% | 85.2% | 88.3% | mAP+4.7%, R1+8.1% | 🟡 vs PAA b32: -0.3% mAP, +0.4% R1。R5/R10 改善但 mAP 未超。b32 仍是最优配置 |
experiments/results.md:122:| 070-eq | PSG+GCN + PAA S&C (eq_concat) | 61.4% | 73.4% | 85.4% | 88.5% | mAP+4.8%, R1+6.9% | 🟡 vs PAA scene: -0.2% mAP, -0.8% R1。target-only 热图不优于 scene 热图。消融价值 |
experiments/results.md:123:| 071-eq | PSG+GCN + PCL r=16 (eq_concat) | 60.7% | 72.0% | 84.6% | 88.1% | mAP+4.1%, R1+5.5% | ❌ vs PAA: -0.9% mAP, -2.2% R1。Feature-dependent LoRA 劣于 feature-independent PAA |
experiments/results.md:124:| 072-eq | PSG+GCN + PS-PAA (eq_concat) | 61.1% | 73.8% | 84.8% | 88.4% | mAP+4.5%, R1+7.3% | 🟡 vs PAA: -0.5% mAP, -0.4% R1。Body-part 分组不优于 generic 混合 |
experiments/results.md:125:| 073-eq | PSG+GCN + PAA Stage2+3 (eq) | 61.1% | 74.2% | 85.7% | 88.4% | mAP+4.5%, R1+7.7% | 🟡 vs PAA Stage3: -0.5% mAP, 0.0% R1。多 stage 不如单 stage |
experiments/results.md:126:| 074-eq | PSG+GCN + PAA+PGAM (eq) | — | — | — | — | — | ❌ 中止。PGAM 完全无效——结果与 exp066 精确相同。PGAM 为 no-op |
experiments/results.md:130:| 076-eq | PSG+GCN+PAA+TDPC (eq) | 61.3% | 72.7% | 84.9% | 87.8% | mAP+4.7%, R1+6.2% | ❌ vs PAA(61.6/74.2): -0.3%/-1.5%。differential adapter 无收益 |
experiments/results.md:131:| 077-eq | PSG+GCN+ST-PAA 34ch (eq, 5060) | 61.0% | 73.6% | 84.4% | 88.6% | mAP+4.4%, R1+7.1% | ❌ vs PAA: -0.6%/-0.6%。scene+target concat 不优于 scene-only |
experiments/results.md:132:| 078-eq | PSG+GCN+PAA+APG (eq) | 60.5% | 72.5% | 84.3% | 87.9% | mAP+3.9%, R1+6.0% | ❌ vs PAA: -1.1%/-1.7%。adaptive gate 负面 |
experiments/results.md:133:| **079-eq** | **PSG+GCN+ROA 无PAA (eq, 5060)** | **62.0%** | **73.6%** | **85.0%** | **88.1%** | **mAP+5.4%, R1+7.1%** | **🟢🟢 ROA 独立有效！vs 3-seed: +1.27%/+1.03%。≈ exp067 PAA+ROA** |
experiments/results.md:134:| 081-eq | PSG+PAA+PQTD (eq) | 56.9% | 67.2% | 79.1% | 84.1% | mAP+0.3%, R1+0.7% | ❌❌ Decoder 严重不够收敛。GCN(400K) >> Decoder(2.5M) 在 120ep |
experiments/results.md:135:| 083-eq | PSG+GCN+PAA+PGFI (eq) | 61.1% | 73.4% | 84.7% | 88.1% | mAP+4.5%, R1+6.9% | 🟡 中性偏负 vs PAA(-0.5%/-0.8%)。Inpainter 未带来额外收益 |
experiments/results.md:136:| 084-eq | PSG+GCN+PAA+CIPGFR (eq) | 61.4% | 73.6% | 85.5% | 88.6% | mAP+4.8%, R1+7.1% | 🟡 中性 vs PAA(-0.2%/-0.6%)。Cross-instance recovery 未改善 |
experiments/results.md:137:| **085-eq** | **PSG+GCN+PAA+ROA p=0.7 (5060)** | **62.6%** | **75.3%** | **85.2%** | **88.4%** | **mAP+6.0%, R1+8.8%** | **🟢🟢🟢 历史最高！vs ROA p=0.5: +0.6%/+1.7%** |
experiments/results.md:138:| 085b-eq | PSG+GCN+ROA p=0.7 无PAA (5060) | 62.2% | 73.4% | 84.5% | 88.0% | mAP+5.6%, R1+6.9% | 🟡 vs p=0.5 无PAA: +0.2%. p=0.7 增益主要来自与 PAA 协同 |
experiments/results.md:139:| **086-eq** | **PSG+GCN+PAA+ROA+PA-PAT (3路)** | **62.7%** | **74.6%** | **85.3%** | **88.7%** | **mAP+6.1%, R1+8.1%** | **🟢🟢🟢 Peak 62.8%@Ep100。留作拼 SOTA recipe** |
experiments/results.md:140:| 087-eq | PSG+GCN+PAA+MM (momentum) | 61.5% | 73.0% | 84.5% | 88.2% | mAP+4.9%, R1+6.5% | 🟡 中性 vs PAA(-0.1%/-1.2%)。Memory contrastive 无额外收益 |
experiments/results.md:143:| 091-eq | PSG+GCN+PAA+TTSFR (eq) | 61.4% | 73.2% | 85.1% | 88.5% | mAP+4.8%, R1+6.7% | 🟡 中性 vs PAA(-0.2%/-1.0%)。Batch 内 recovery 信号不够（仅4张/ID） |
experiments/results.md:144:| 092-eq | PSG+GCN+PAA+LSRM w=0.5 (eq) | 60.9% | 73.3% | 85.0% | 88.1% | mAP+4.3%, R1+6.8% | 🟡 中性偏负 vs PAA(-0.7%/-0.9%)。Learned recovery 在 batch 内仍不够 |
experiments/results.md:145:| 092d-eq | PSG+GCN+PAA+LSRM BS128 (eq) | 61.3% | 73.5% | 84.8% | 88.4% | mAP+4.7%, R1+7.0% | 🟡 大batch帮助 (+0.4% vs BS64)，但仍 -0.3% vs PAA |
experiments/results.md:146:| 091b-eq | PSG+GCN+PAA+TTSFR BS128 (5060) | 60.8% | 73.0% | — | 88.6% | mAP+4.2%, R1+6.5% | 🟡 中性偏负。大 batch 对 simple recovery 无效 |
experiments/results.md:147:| 093-eq | PSG+GCN+PAA+PGTM (eq) | 56.7% | 68.0% | 80.9% | 85.2% | mAP+0.1%, R1+1.5% | ❌❌ Token merging 9.4M params 120ep 严重不够收敛 |
experiments/results.md:148:| 094 | PSG+GCN+PAA+PCQA (PTM) | — | — | — | — | 中性 (Ep74终止) | 🟡 PTM loss 不收敛(0.28→0.40)，Ep70: 59.2% vs 基线58.1%(+1.1%)，但 PTM 对照 exp030a 而非 exp066 |
experiments/results.md:149:| 094b | PSG+GCN+PAA+PCQA 归一化 (远程) | 61.2% | 74.0% | 84.8% | 88.2% | vs PAA: -0.4%/-0.2% | 🟡 PCQA 中性。PTM loss 0.41 不收敛 |
experiments/results.md:150:| 095-eq | PSG+GCN+PAA+DPF (热图池化) Ep100 | 60.0% | 71.8% | 83.5% | 87.1% | vs PAA: **-1.6%/-2.4%** | ❌ 12×4 分辨率太低，热图空间池化不如点采样 |
experiments/results.md:151:| 096-eq | PSG+GCN+PAA+MRKF (多尺度) Ep100 | 60.3% | 72.0% | 84.3% | 87.2% | vs PAA: -1.3%/-2.2% | ❌ Stage2(384d)+Stage3 融合不稳定，高方差震荡 |
experiments/results.md:152:| 098-eq | PSG+GCN+PAA+PKP (KPR式prompting) | 60.9% | 72.8% | 84.5% | 88.5% | vs PAA: -0.7%/-1.4% | 🟡 Swin window attention 限制早期 pose 传播 |
experiments/results.md:154:| 100-eq | PSG+GCN+PAA+FiLM (全阶段) | 61.0% | 73.3% | 84.6% | 88.3% | vs PAA: -0.6%/-0.9% | 🟡 PSG+PAA 已足够，更多 conditioning 不帮助 |
experiments/results.md:155:| 101-eq | PSG+GCN+PAA+SGMT (masking) | 61.0% | 73.8% | 85.0% | 88.5% | vs PAA: -0.6%/-0.4% | 🟡 中性，SGCFR 增益与基线相同 (+2.7% vs +2.6%) |
experiments/results.md:156:| 102-eq | PSG+GCN+PAA+SGMT-50% (masking) Ep110 | 60.6% | 73.1% | 84.7% | 87.9% | vs PAA: -1.0%/-1.1% | 🟡 50% masking 更激进，效果略差于 30%(exp101)。训练仅到 Ep110 |
experiments/results.md:157:| 104c-eq | PSG+GCN+PAA+PACD v3 (3×3 fm mask) | 61.3% | 74.5% | 85.4% | 88.6% | vs PAA: -0.3%/+0.3% | 🟡 中性。Feature map masking (8%) 太弱，GAP 鲁棒 |
experiments/results.md:158:| 104d-eq | PSG+GCN+PAA+PACD v4 (row fm mask) Ep100 | 60.4% | 73.3% | 84.5% | — | vs PAA: -1.2%/-0.9% | 🟡 中性偏负。33% 行级 mask 仍不够 |
experiments/results.md:159:| 105b-eq | PSG+GCN+PAA+SGRE (cross-attn) Ep90 | 60.7% | 73.3% | 85.1% | — | vs PAA: -0.3%/-0.2% | 🟡 中性。SGRE loss 收敛(3.28→0.30)但 detached kp 不影响 backbone |
experiments/results.md:160:| 106-eq | PSG+GCN+PAA+PISD (image mask) Ep28 | — | — | — | — | 提前终止 | 🟡 pisd loss 0.02-0.04 极小。GAP 全局特征天然遮挡不变 |
experiments/results.md:161:| 142-eq | PSG+GCN+SKC (Support-Supervised Keypoint Completion, eq) | 60.3% | 71.8% | 84.4% | 87.7% | vs exp030a-eq: -0.8%/-1.9% | ❌ 中性偏负。completion module 虽然活跃（gate=0.26, delta_norm=1.5），但 skc_pre≈skc_post 说明修改方向不是向 prototype 靠近。gate 无限制增长导致后期过度修改特征。feature-level completion 方向已被多轮验证为无效 |
experiments/results.md:162:| 143-eq | PSG+GCN+SASA (Skeleton-Aware Self-Attention, eq) | 61.1% | 73.7% | 85.1% | 88.5% | vs exp030a-eq: **0.0%/0.0%** | 🟡 完美中性。零参数骨架测地注意力偏置对最终结果无任何影响。与 KP-RPE(exp052) 结论一致：Swin window attention 的 RPE 已足够编码空间结构 |
experiments/results.md:163:| 141-cvk | PSG+GCN+LPCS comp_ctx (cvk_residual) | 55.8% | 68.1% | 78.3% | 82.4% | — | ❌ LPCS comp_ctx 失败。competition-context 未改善排序。LPCS 训练 loss 严重干扰主学习，最终远低于 exp030a (-5.3% mAP) |
experiments/results.md:164:| 144-eq | PSG+GCN+SASA α=1.0 (equal_concat) | 61.0% | 73.5% | 84.6% | 87.9% | vs exp030a-eq: **-0.1%/-0.2%** | 🟡 中性。10x更强的SASA偏置与α=0.1结果相同。确认skeleton attention信息对Swin完全冗余 |
experiments/results.md:165:| 145-eq | PSG+GCN+PAA+SASA (equal_concat) | 61.4% | 73.8% | — | 88.4% | vs PAA(exp066): **-0.2%/-0.4%** | 🟡 中性。SASA 与 PAA 组合无正交增益，确认 SASA 在任何配置下均无效 |
experiments/results.md:166:| 148-eq | PSG+GCN+PCVT (Pose-Complementary View Training, eq) | ~59.3%* | ~71.3%* | — | — | ❌ 负面。*ep100 数据，训练中。早期加速（ep30: +2.4 mAP）但后期被基线追平并反超。3-view 训练的 1/3 主损失稀释导致后期收敛不足。训练集 95.8% 全可见使 complementary masking 缺乏信号 |
experiments/results.md:167:| 149 | PSG+GCN+SCFA (Symmetry-Conditioned Feature Aggregation) | — | — | — | — | ❌ ep30 止损。ep30: 50.7/61.3 vs exp030a 52.2/66.0 (-1.5/-4.7)。bilateral gap case 太少(scfa_pg=0.09)，hand-crafted pooling trick 不够强 |
experiments/results.md:168:| 151-eq | PSG+GCN+PVAT (Pose-Visibility Adversarial Training, eq) | 进行中 | — | — | — | 🟡 中性趋势。ep70: 59.0/72.0 vs exp030a 58.1/70.9 (+0.9/+1.1)。但 pvat_acc=0.83 不降——训练集 95.8% 可见，adversarial 无信号。预计最终中性 |
experiments/results.md:208:### Phase 4: SupCon + OA-SD + Parallel Aug (exp166-193)
experiments/results.md:212:| 166 | STD-PR+PLBOA+PAPE+MS-PSG+CE (full arch) | 63.1% | 73.9% | 86.1% | 89.2% | CE baseline with full architecture |
experiments/results.md:213:| 166r | ↳ base arch (no PAPE/MS-PSG) | 60.3% | 72.8% | — | — | CE base for OA-SD comparison |
experiments/results.md:215:| **187** | **+SupCon T=0.05 + 3-view Parallel Aug** | **64.9%** | **76.6%** | **87.2%** | **90.0%** | **🟢🟢 Overall best! +1.8/+2.7 vs exp166** |
experiments/results.md:216:| **190** | **3-view Parallel Aug + CE (no SupCon)** | **64.2%** | **75.6%** | **86.2%** | **89.1%** | **✅ 3-view+CE ≈ 1-view+SupCon! +1.1/+1.7 vs exp166** |
experiments/results.md:217:| **191** | **OA-SD + CE (1-view, decay=0.999)** | **63.2%** | **75.4%** | **86.3%** | **89.1%** | **✅ OA-SD 独立有效! +2.9/+2.6 vs CE base** |
experiments/results.md:218:| 192 | OA-SD + CE (1-view, decay=0.99) | 62.6% | 74.9% | 86.0% | 89.1% | 🟡 vs exp191: -0.6/-0.5。decay 不敏感 |
experiments/results.md:219:| **193** | **OA-SD + 3-view + CE** | **64.4%** | **76.5%** | **86.3%** | **89.4%** | **✅ OA-SD+3-view additive! R1 ≈ exp187 SupCon, mAP +0.2/R1+0.9 vs exp190** |
experiments/results.md:220:| 194 | OA-SD + CE (weight=2.0) | 63.4% | 74.8% | 86.1% | 89.1% | 🟡 vs exp191 (w=1.0): +0.2/-0.6。weight 不敏感 |
experiments/results.md:221:| 195 | SupCon + OA-SD global-only (base) | 61.3% | 74.9% | — | — | 🟡 OA-SD+SupCon 无冲突但增益有限 |
experiments/results.md:222:| 196 | 3-view + SupCon + OA-SD global-only | 62.4% | 75.2% | 85.2% | 87.8% | ❌ vs exp187: -2.5/-1.4。OA-SD+SupCon 互斥 |
experiments/results.md:223:| 197 | 3-view + SupCon + STM | 64.1% | 76.0% | 86.7% | 89.0% | ❌ vs exp187: -0.8/-0.6。STM 不改善 SupCon 路线 |
experiments/results.md:224:| 198 | OA-SD + CE + STM (base, remote) | 63.2% | 75.2% | — | — | 🟡 = exp191 (无 STM)。STM 只加速不改善天花板 |
experiments/results.md:225:| 199 | 3-view + SupCon + OA-RD | 63.4% | 74.5% | 85.2% | 88.1% | ❌ vs exp187: -1.5/-2.1。OA-RD+SupCon 不兼容 |
experiments/results.md:226:| 200 | CE + OA-RD (base, remote) | 62.9% | 73.9% | 85.2% | 88.5% | ❌ vs exp191 OA-SD: -0.3/-1.5。OA-RD 不如 OA-SD |
experiments/results.md:227:| 201 | 3-view + SupCon + Global SupCon | 63.7% | 73.8% | 85.1% | 88.7% | ❌ vs exp187: -1.2/-2.8。Global SupCon 压缩特征空间 |
experiments/results.md:233:| **206** | **Small GCN+PAA+CE+OA-SD (2-run mean)** | **70.4%** | **82.1%** | — | — | **🟢🟢🟢🟢 R1 超 4090 PAA! 2-run: 70.5/82.3 + 70.3/81.8** |
experiments/results.md:239:| 066-occ_reid | PSG+GCN+PAA (equal_concat) | 72.2% | 77.8% | 88.1% | 93.3% | 跨数据集泛化 |
experiments/results.md:240:| 079-local | PSG+GCN+ROA 无PAA (本地验证) | 61.8% | 72.8% | 85.2% | 88.3% | — | ✅ 跨硬件一致 (vs 远程 62.0%/73.6%, Δ<0.2%) |
experiments/results.md:241:| 082-remote | PSG+GCN+PAA+ROA (远程验证) | 61.8% | 74.1% | 84.8% | 88.5% | — | ✅ 跨硬件一致 (vs 本地 62.0%/73.7%, Δ<0.2%) |
experiments/results.md:242:| 056-eq | PSG+GCN + PGAM S2+S3 (eq_concat) | 61.1% | 73.7% | 85.2% | 88.6% | mAP+4.5%, R1+7.2% | 🟡 vs exp054: ≈持平。多 Stage 无额外增益 |
experiments/results.md:244:> 注：`exp036 / exp037` 的编号沿用了原 visibility 路线的占位命名，但实验内容已经转入 `PSG+GCN` branch 的后续探索；解读时不要把编号本身当作路线语义。
experiments/results.md:284:| 049-g-k2 | PSG+GCN global + NFC | 2 | 62.8% | 74.9% | 83.9% | 87.5% | mAP+3.0%, R1+5.4% |
experiments/results.md:285:| 049-g-k5 | PSG+GCN global + NFC | 5 | 65.5% | 73.0% | 82.0% | 85.4% | mAP+5.7%, R1+3.5% |
experiments/results.md:286:| 049-eq-k2 | PSG+GCN equal_concat + NFC | 2 | 63.4% | 74.6% | 84.2% | 87.1% | mAP+2.3%, R1+0.9% |
experiments/results.md:287:| 049-eq-k3 | PSG+GCN equal_concat + NFC | 3 | 64.8% | 75.6% | 84.1% | 87.4% | mAP+3.7%, R1+1.9% |
experiments/results.md:288:| 049-eq-k4 | PSG+GCN equal_concat + NFC | 4 | 66.3% | 76.9% | 84.2% | 87.5% | mAP+5.2%, R1+3.2% |
experiments/results.md:289:| **049-eq-k5** | **PSG+GCN equal_concat + NFC** | **5** | **67.3%** | **77.6%** | **84.8%** | **87.8%** | **mAP+6.2%, R1+3.9%** |
experiments/results.md:290:| 049-eq-k6 | PSG+GCN equal_concat + NFC | 6 | 68.3% | 77.2% | 84.9% | 88.0% | mAP+7.2%, R1+3.5% |
experiments/results.md:291:| 049-eq-k8 | PSG+GCN equal_concat + NFC | 8 | 69.6% | 76.0% | 84.1% | 87.7% | mAP+8.5%, R1+2.3% |
experiments/results.md:292:| 049-eq-k10 | PSG+GCN equal_concat + NFC | 10 | 70.9% | 74.9% | 84.1% | 87.6% | mAP+9.8%, R1+1.2% |
experiments/results.md:298:| 030a-g+RR | PSG+GCN global + Re-ranking | 76.4% | 75.3% | 85.2% | 87.7% | |
experiments/results.md:299:| **030a-eq+RR** | **PSG+GCN equal_concat + Re-ranking** | **78.5%** | **78.8%** | **87.5%** | **89.2%** | **全实验最佳 (含后处理)** |
experiments/results.md:308:| 4090-M-PSG | + PSG (Backbone Injection) | 92.4% | 96.7% | 98.8% | 99.4% | mAP+0.8%, R1+0.4% | PSG 在非遮挡数据集也有效 |
experiments/results.md:316:| 4090-M-PSG-small-lr4 | PSG | Swin-Small | 4e-4 | 93.9% | 96.9% | 99.0% | 99.3% |
experiments/results.md:317:| 4090-M-PSG-small-lr8 | PSG | Swin-Small | 8e-4 | 93.7% | 96.9% | 99.0% | 99.3% |
experiments/results.md:324:| 4090-OD-PSG-small-lr4 | PSG | Swin-Small | 4e-4 | 67.8% | 76.7% | 86.9% | 90.6% |
experiments/results.md:325:| 4090-OD-PSG-small-lr8 | PSG | Swin-Small | 8e-4 | 66.4% | 75.7% | 87.3% | 90.5% |
experiments/results.md:327:### 跨数据集/Backbone PSG 增益总结
experiments/results.md:329:| 数据集 | Backbone | PSG mAP提升 |
experiments/results.md:336:**结论**: PSG 在所有数据集和 backbone 上均有效。在遮挡数据集上增益更大。
experiments/results.md:349:| PSG (exp007) | global | 58.3% | 57.9% | 57.3% | **57.83±0.50%** | **67.13±0.84%** |
experiments/results.md:350:| PSG + 0.5x loss (exp007a) | global | 59.6% | 59.5% | 59.0% | **59.37±0.32%** | **69.43±0.12%** |
experiments/results.md:352:| PSG + GCN (exp030a) | global | 59.8% | 59.1% | 59.1% | **59.33±0.40%** | **68.87±1.00%** |
experiments/results.md:353:| PSG + GCN (exp030a) | concat_scaled | 60.5% | 59.7% | 60.4% | **60.20±0.44%** | **73.13±0.29%** |
experiments/results.md:354:| PSG + GCN (exp030a) | equal_concat | 61.1% | 60.2% | 60.9% | **60.73±0.47%** | **72.57±0.58%** |
experiments/results.md:355:| PSG + GCN (exp030a) | gcn_only | 58.2% | 57.4% | 58.3% | **57.97±0.49%** | **71.77±0.60%** |
experiments/results.md:361:| PSG vs Baseline | **+1.33%** | (1.6, 2.0, 0.4) | 2.77 | 0.1091 | 3 seeds 全正，但 n=3 时双侧检验仍偏弱 |
experiments/results.md:362:| exp007a vs PSG | **+1.53%** | (1.3, 1.6, 1.7) | 12.76 | 0.0061 | ✅ 0.5x global loss 对 PSG 是稳定增益 |
experiments/results.md:373:1. **PSG 仍成立**：3 个 seed 全正，均值 `56.50% → 57.83%`，只是不能再把它写成“统计显著已确认”，更准确的表述是“稳定正向、样本数仍小”。
experiments/results.md:888:## 2026-03-22: exp148 / exp149 两条大改动方向的第一轮分化
experiments/results.md:890:### exp148: `PCVT` 早中期已形成稳定 `mAP` 正向，成为当前最值得继续追的训练端新方向
experiments/results.md:892:> `exp148` 把单图改写成 `full / complementary-view-a / complementary-view-b` 三视图训练对象，用 pose-defined complementary pseudo-views 验证“单图能否被改写成伪多 support 学习对象”。该实验当前仍在运行，以下结论来自 `ep10/20/30` 训练监控。
experiments/results.md:896:| `exp148` | 40.2% | 51.4% | `ep10` |
experiments/results.md:897:| `exp148` | 49.1% | 60.7% | `ep20` |
experiments/results.md:898:| `exp148` | 54.6% | 65.8% | `ep30` |
experiments/results.md:918:### exp149: `SCFA` 快速诊断判负，双侧冗余前提在当前 benchmark 上不够强
experiments/results.md:920:> `exp149` 只给一个短止损窗口，目标不是整晚主线，而是快速回答“单图内部双侧同源冗余是否足够强到值得单开方法”。该实验已在 `ep30` 后终止。
experiments/results.md:924:| `exp149` | 34.9% | 44.3% | `ep10` |
experiments/results.md:925:| `exp149` | 43.6% | 53.8% | `ep20` |
experiments/results.md:926:| `exp149` | 50.7% | 61.3% | `ep30` |
experiments/results.md:943:## 2026-04-01/02: exp206r, exp207, exp209, exp210, exp210b, exp212, exp213
experiments/results.md:945:### exp206r: Small GCN+PAA+CE+OA-SD (Fixed OA-SD teacher)
experiments/results.md:946:> Repeat of exp206 with fixed OA-SD teacher (BN/Dropout/DropPath eval mode, clean teacher pose)
experiments/results.md:950:| exp206r equal_concat | 70.6% | 82.6% | 89.5% | 91.4% | ep120 final |
experiments/results.md:951:| **exp206r maxsim_hybrid** | **72.3%** | **82.9%** | **90.5%** | **92.2%** | ep120 + maxsim test |
experiments/results.md:953:- OA-SD fix: +0.1/+0.3 vs buggy exp206 (70.5/82.3). Fix 加速了早期收敛但不改变 final。
experiments/results.md:955:### exp207: Base GCN+PAA+CE+OA-SD 3-view (Fixed OA-SD)
experiments/results.md:959:| exp207 equal_concat | 70.7% | 80.7% | 89.5% | 91.7% | ep120 final |
experiments/results.md:960:| exp207 maxsim_hybrid | 72.2% | 82.0% | 90.4% | 92.3% | ep120 + maxsim test |
experiments/results.md:965:### exp209: Small STD-PR+CE+OA-SD — 终止 (ep30)
experiments/results.md:969:| exp209 | 56.0% | 69.3% | ep30 终止 |
experiments/results.md:971:- STD-PR+CE+OA-SD 严重落后 GCN+PAA+CE+OA-SD (~5% at ep30)。STD-PR 需要 SupCon。
experiments/results.md:973:### exp210: Small GCN+PAA+CE+OA-SD + PKC weight=0.5 — 灾难
experiments/results.md:977:| exp210 | 3.6% | 5.3% | ep10 终止 |
experiments/results.md:981:### exp210b: Small GCN+PAA+CE+OA-SD + PKC weight=0.05
experiments/results.md:985:| exp210b equal_concat | 70.6% | 81.8% | 89.9% | 92.4% | ep120 final |
experiments/results.md:986:| **exp210b maxsim_hybrid** | **72.4%** | **83.1%** | **90.8%** | **92.7%** | ep120 + maxsim test |
experiments/results.md:988:- PKC=0.05 不改变 equal_concat (= exp206r)，但 MaxSim 提升 +0.1/+0.2。
experiments/results.md:991:### exp212: Small GCN+PAA+CE+OA-SD LR=0.0008 — 灾难
experiments/results.md:995:| exp212 | 0.8% | 1.3% | ep10 终止 |
experiments/results.md:999:### exp213: Small + PKC(0.05) + MST(0.1) — 终止
experiments/results.md:1003:| exp213 | 40.6% | 54.8% | ep10 终止 |
experiments/results.md:1007:### MaxSim Hybrid 跨 checkpoint 分析 (exp206 local)
experiments/results.md:1019:## 2026-04-02: exp215, exp217, exp218, exp220, exp222, exp223
experiments/results.md:1023:| exp215 BA-PKC w=0.1 | 0.5% | 0.8% | 3.1% | 4.5% | ep10 终止 |
experiments/results.md:1024:| exp217 OERL + OA-SD | 62.2% | 75.2% | 86.0% | 89.0% | ep120 final |
experiments/results.md:1025:| exp218 PACI + OA-SD | 61.9% | 74.2% | 85.6% | 88.9% | ep120 final |
experiments/results.md:1026:| exp220 GSPB + OA-SD | 62.9% | 74.3% | 86.2% | 89.5% | ep120 final |
experiments/results.md:1027:| exp222 GSPB on Small (scale=0.05) | 2.3% | 3.9% | 9.9% | 14.3% | ep10 终止 |
experiments/results.md:1028:| exp223 PADPQ K=4 + OA-SD | 63.7% | 74.5% | 86.2% | 89.5% | ep120 final |
experiments/results.md:1030:- exp215 证实了 non-detached BA-PKC 会直接破坏 backbone 收敛。
experiments/results.md:1031:- exp217 / exp218 / exp220 都低于 `exp191 = 63.2 / 75.4`，因此不能写成训练端正向超越。
experiments/results.md:1032:- exp223 在 `equal_concat` 上给出 `mAP +0.5`，但 `R1 -0.9`；当前更适合作为 trade-off 证据，而不是“全面超越”。
experiments/results.md:1033:- exp219 的远程 `train_log` 已补回，但目前只确认到 `ep30 = 51.9 / 64.9`，尚无 final，因此暂不纳入正式结果表。
experiments/results.md:1034:- 注：`exp220/223` 的 `maxsim_hybrid` 数字目前只在各自 `monitor.md` 中留有测试记录，本地未发现独立 `test_log`，因此本总表仅登记训练日志可直接复核的 `equal_concat` 结果。
experiments/results.md:1036:## 2026-04-02/03: exp222c, exp224, exp225, exp226, exp227, exp228
experiments/results.md:1042:| exp222c GSPB Small scale=0.01 | 15.1% | 23.8% | 38.4% | 45.4% | ep10 终止 |
experiments/results.md:1043:| exp224 KAMP (random-init proj) + OA-SD | 60.7% | 73.0% | 85.1% | 88.3% | ep120 final |
experiments/results.md:1044:| exp225 GSPB(0.05) + PADPQ K=4 + OA-SD | 64.2% | 74.9% | 86.8% | 89.6% | ep120 final |
experiments/results.md:1045:| exp226 KAMP (zero-init proj) + OA-SD | 61.6% | 74.3% | 85.1% | 88.0% | ep120 final |
experiments/results.md:1047:- exp222c: GSPB scale=0.01 在 Small 上仍然灾难 (scale=0.05 → 2.3%, scale=0.01 → 15.1%)
experiments/results.md:1048:- exp224: KAMP (多尺度 keypoint 融合) random-init projection 造成 -2.5% mAP 噪声
experiments/results.md:1049:- exp225: **GSPB+PADPQ K=4 = 64.2/74.9 — Tiny 最佳 equal_concat！** (+1.0/-0.5 vs OA-SD)
experiments/results.md:1050:- exp226: KAMP zero-init projection 减少噪声但仍 -1.6% mAP。KAMP 方向失败。
experiments/results.md:1052:### exp227: Small GSPB(0.005) + PADPQ K=4 + OA-SD
experiments/results.md:1056:| exp227 equal_concat | 71.6% | 80.8% | 89.8% | 91.8% | ep120 final |
experiments/results.md:1057:| exp227 maxsim_hybrid | 71.8% | 80.6% | 89.9% | 91.9% | ep120 + maxsim test |
experiments/results.md:1059:- 对照 exp206r: **mAP +1.0, R1 -1.8** (equal_concat)
experiments/results.md:1061:- maxsim 71.8 < 当前最佳 72.4 (exp210b)。**GSPB+PADPQ 在 Small maxsim 上无优势。**
experiments/results.md:1064:### exp228: Tiny GSPB(0.05) + PADPQ K=8 + OA-SD
experiments/results.md:1068:| exp228 equal_concat | 64.1% | 74.3% | 86.4% | 89.5% | ep120 final |
experiments/results.md:1070:- 对照 exp225 K=4: **-0.1/-0.6**。K=8 ≈ K=4，无额外收益。
experiments/results.md:1073:### exp229: Tiny BT-PKD (w=0.01, constant) + OA-SD
experiments/results.md:1077:| exp229 equal_concat | 62.2% | 75.0% | 86.1% | 89.0% | ep120 final |
experiments/results.md:1079:- 对照 exp191 (OA-SD): **-1.0/-0.4**
experiments/results.md:1084:### exp230: Small BT-PKD (w=0.01, constant, no PARALLEL_AUG)
experiments/results.md:1088:| exp230 equal_concat | 70.8% | 81.9% | 89.7% | 91.9% | ep110 (OOM at ep120) |
experiments/results.md:1091:- 对照 exp206r (有 PAUG): 70.6/82.6 → **+0.2/-0.7** (mAP 持平, R1 差因缺 PAUG)
experiments/results.md:1093:### exp231: Tiny BT-PKD cosine decay (w→0 by ep60)
experiments/results.md:1097:| exp231 equal_concat | 61.7% | 74.3% | 85.5% | 88.6% | ep120 final |
experiments/results.md:1099:- 对照 exp191: **-1.5/-1.1**。Cosine decay 没有解决后期干扰。
experiments/results.md:1100:- 对照 exp229 constant: **-0.5/-0.7**。Decay 甚至略差。
experiments/results.md:1104:## 2026-04-04: exp235, exp236, exp237
experiments/results.md:1106:### exp235: FSDC (wrong ROA+PLBOA config)
experiments/results.md:1110:| exp235 | 61.7% | 74.5% | ep120 final |
experiments/results.md:1112:- 对照 exp191: **-1.5/-0.9**
experiments/results.md:1115:### exp236: FSDC (正确 ROA=False, PLBOA=0.7)
experiments/results.md:1119:| exp236 | 61.7% | 73.2% | ep120 final |
experiments/results.md:1121:- 对照 exp191: **-1.5/-2.2**
experiments/results.md:1124:### exp237: PPA (Pose-Prompted Part-Assignment Head) ⭐
experiments/results.md:1128:| exp237 equal_concat | **63.7%** | **75.0%** | ep120 final |
experiments/results.md:1129:| exp237 maxsim_hybrid | 64.1% | 75.1% | ep120 + maxsim |
experiments/results.md:1131:- 对照 exp191: equal_concat **+0.5/-0.4**, maxsim -0.1/-2.0
experiments/results.md:1136:### exp238: PPA assign_weight=0.1
experiments/results.md:1140:| exp238 | 62.1% | 74.0% | ep120 final |
experiments/results.md:1142:- 对照 exp191: **-1.1/-1.4**
experiments/results.md:1145:### exp239: PPA + GiLt (Part triplet only)
experiments/results.md:1149:| exp239 | 63.8% | 73.6% | ep120 final |
experiments/results.md:1151:- 对照 exp191: **+0.6/-1.8**
experiments/results.md:1154:### exp240: PPA on Small (w=0.5, no PARALLEL_AUG)
experiments/results.md:1158:| exp240 | 70.7% | 81.1% | ep120 final |
experiments/results.md:1160:- 对照 exp230 (no PAUG, ep110): -0.1/-0.8
experiments/results.md:1163:### exp241: PPA + GCN 双分支 on Tiny ⭐
experiments/results.md:1167:| exp241 equal_concat | **63.7%** | **75.3%** | **86.2%** | **88.9%** | ep120 final |
experiments/results.md:1169:- 对照 exp191: **+0.5/-0.1** — 最佳综合结果!
experiments/results.md:1170:- 对照 exp237 PPA-only: +0.0/+0.3 — GCN 改善 R1
experiments/results.md:1174:**exp241 MaxSim test**: 64.1/74.8 (MaxSim gain +0.4/-0.5 vs equal_concat)
experiments/results.md:1176:### exp242: PPA + GCN on Small ❌❌
experiments/results.md:1180:| exp242 | 60.9% | 73.4% | 88.9% | ep120 final |
experiments/results.md:1182:- 对照 exp206r (Small GCN): **-9.7/-9.2** — 灾难性失败!
experiments/results.md:1186:### exp243: LGPA (CLIP + Cross-Attention + Pose) on Tiny 🟡
experiments/results.md:1190:| exp243 ep80 | 60.9% | 72.5% | ep80 (GPU crash at ep88, 训练未完成) |
experiments/results.md:1192:- 对照 exp191 (GCN ep80): 62.0/74.4 = **-1.1/-1.9**
experiments/results.md:1198:### exp244: LGPA-Detach (CLIP + Detached Features) ⭐⭐⭐
experiments/results.md:1202:| exp244-R (detach, 无OASD) | 63.6% | 74.7% | 85.3% | 88.6% | ep120 remote final |
experiments/results.md:1203:| **exp244-L (detach+OASD)** | **65.3%** | **75.7%** | **86.8%** | **89.7%** | **ep120 local final** |
experiments/results.md:1205:- 对照 exp191 (GCN+OASD): **+2.1/+0.3** — **首个在 final 仍正向的 Part branch!**
experiments/results.md:1206:- 对照 exp243 (LGPA non-detach, ep80): +4.4/+3.2 — detach 完全解决后期干扰
experiments/results.md:1207:- 对照 exp244-R (无OASD): +1.7/+1.0 — OA-SD 与 LGPA-D 正交叠加
experiments/results.md:1208:- **无 OA-SD 的 LGPA-D (63.6) ≈ GCN + OA-SD (63.2)**: CLIP 语义 ≈ OA-SD
experiments/results.md:1212:**exp244 MaxSim test**: 66.0/76.4/87.2/90.5 (MaxSim hybrid on LGPA-D+OA-SD ep120)
experiments/results.md:1214:### exp245g: LGPA-Detach on Swin-Small ⭐⭐
experiments/results.md:1218:| **exp245g (Small LGPA-D+OA-SD)** | **70.2%** | **80.1%** | **89.8%** | **91.2%** | **ep120 local PT2+mmcv-full** |
experiments/results.md:1220:- 对照 exp206r (Small GCN+PAA+OA-SD): -0.4/-2.5 — mAP 接近, R1 差距
experiments/results.md:1221:- 对照 exp244 (Tiny LGPA-D+OA-SD): **+4.9/+4.4** — Small backbone 有效
experiments/results.md:1222:- LGPA-D 用更简单架构 (无 GCN, 无 PAA) 达到接近 exp206r 的 mAP
experiments/results.md:1225:**exp245g MaxSim test**: 71.9/82.2/91.0/92.8 (MaxSim hybrid on Small LGPA-D+OA-SD ep120)
experiments/results.md:1227:- vs exp206r (70.6/82.6): **mAP +1.3, R1 -0.4** — mAP 超越 Small baseline!
experiments/results.md:1229:### exp245h_v2: Small LGPA-D + OA-SD 远程复现 ⭐⭐⭐
experiments/results.md:1233:| **exp245h_v2 equal_concat** | **71.6%** | **81.6%** | **89.2%** | **91.2%** | **ep120 远程 5060Ti final** |
experiments/results.md:1235:- 对照 exp245g (本地 3090): **+1.4/+1.5** — 远程环境收敛更好
experiments/results.md:1236:- 对照 exp206r (Small baseline): **mAP +1.0, R1 -1.0**
experiments/results.md:1239:**exp245h_v2 MaxSim test**: 73.0/82.7/90.5/92.7 (MaxSim hybrid on ep120)
experiments/results.md:1241:- vs exp206r (70.6/82.6): **mAP +2.4, R1 +0.1** — **Small 全面超越!**
experiments/results.md:1242:- vs exp245g MaxSim (71.9/82.2): **+1.1/+0.5** — **Small 新最强!**
experiments/results.md:1244:### exp246: LGPA-D + GCN 双分支 (Tiny) 🟡
experiments/results.md:1248:| exp246 (ep83 crash) | 64.1% | 75.2% | — | — | ep80 (GPU 竞争 crash) |
experiments/results.md:1249:| **exp246b equal_concat** | **65.5%** | **77.2%** | **86.9%** | **90.1%** | **ep120 final** |
experiments/results.md:1251:- 对照 exp244 (LGPA-D only): **+0.2/+1.5** — GCN 主要贡献在 R1
experiments/results.md:1252:- 对照 exp191 (GCN only): **+2.3/+1.8** — LGPA-D 贡献巨大
experiments/results.md:1253:- LGPA-D 语义 part features + GCN 骨架 keypoint features 正交互补
experiments/results.md:1254:- ep10~ep70 全部与 exp246 精确匹配 (复现验证通过)
experiments/results.md:1256:**exp246b MaxSim test**: 66.3/77.7/87.6/90.6 (MaxSim hybrid on LGPA-D+GCN ep120)
experiments/results.md:1258:- vs exp244 MaxSim (66.0/76.4): **+0.3/+1.3** — **Tiny 新最强!**
experiments/results.md:1260:### exp247: VCSR — Visibility-Conditional Semantic Routing (Tiny, 无OA-SD)
experiments/results.md:1264:| **exp247 VCSR** | **63.6%** | **73.5%** | **84.2%** | **88.3%** | **ep120 远程 final** |
experiments/results.md:1266:- 对照 exp244-R (LGPA-D 无OA-SD): **0.0/-1.2** — VCSR ≈ LGPA-D, visibility gating 无效
experiments/results.md:1271:### exp248: PCFD — Pose-Conditioned Feature Differencing (Test-time) ❌
experiments/results.md:1275:| exp244 cosine baseline | 65.3% | 75.7% | — |
experiments/results.md:1281:- Learned pair-level matching 证伪 (训练端 exp152/153 + test-time PCFD 均失败)
experiments/results.md:1284:### exp249: Small LGPA-D + GCN 双分支 + OA-SD (进行中)
experiments/results.md:1288:| exp249 ep10 | 51.1% | 61.7% | 77.9% | 83.8% | 远程 5060Ti, ep10 |
experiments/results.md:1289:| exp249 ep20 | 60.9% | 73.2% | 85.5% | 88.6% | 远程 5060Ti, ep20 |
experiments/results.md:1290:| exp249 ep30 | 63.6% | 74.2% | 86.0% | 89.2% | 远程 5060Ti, ep30 |
experiments/results.md:1291:| exp249 ep40 | **68.0%** | **78.7%** | 88.8% | 90.7% | 远程 5060Ti, ep40 |
experiments/results.md:1292:| exp249 ep50 | 69.4% | 79.4% | — | 90.9% | 远程 5060Ti, ep50 |
experiments/results.md:1293:| exp249 ep60 | 70.2% | 80.7% | — | 91.1% | 远程 5060Ti, ep60 |
experiments/results.md:1294:| exp249 ep70 | 70.9% | 81.6% | — | 91.4% | 远程 5060Ti, ep70 |
experiments/results.md:1295:| exp249 ep80 | 71.5% | 81.4% | 89.4% | 91.5% | 远程 5060Ti, ep80 |
experiments/results.md:1296:| exp249 ep90 | 71.4% | 81.4% | 89.4% | 91.5% | 远程 5060Ti, ep90 |
experiments/results.md:1297:| exp249 ep100 | 71.7% | 82.3% | 89.6% | 91.8% | 远程 5060Ti, ep100 |
experiments/results.md:1298:| exp249 ep110 | 71.9% | 81.7% | 89.7% | 91.7% | 远程 5060Ti, ep110 |
experiments/results.md:1299:| **exp249 FINAL** | **71.9%** | **81.8%** | **89.5%** | **91.6%** | **远程 5060Ti, ep120 FINAL** ⭐⭐ |
experiments/results.md:1301:- **FINAL: mAP 71.9 (+0.3 vs exp245h_v2), R1 81.8 (+0.2 vs exp245h_v2)**
experiments/results.md:1303:- 对照 exp206r (Small GCN+PAA+OA-SD): 70.6/82.6 → **mAP +1.3, R1 -0.8**
experiments/results.md:1306:**exp249 MaxSim test (ep120 final)**:
experiments/results.md:1310:| exp249 equal_concat | 71.9% | 81.8% | 89.5% | 91.6% |
experiments/results.md:1311:| **exp249 MaxSim** | **73.3%** | **83.2%** | **90.9%** | **93.0%** |
experiments/results.md:1314:- **vs exp245h_v2 MaxSim (73.0/82.7): +0.3/+0.5 — 全面超越!**
experiments/results.md:1315:- **exp249 是项目新最佳: 73.3/83.2 (Small LGPA-D+GCN+OA-SD MaxSim)**
experiments/results.md:1317:### exp250: POT (Partial Optimal Transport) Test-time 评估 🟡
experiments/results.md:1319:在 exp246b (Tiny LGPA-D+GCN ep120) checkpoint 上测试:
experiments/results.md:1334:**exp245h_v2 (Small LGPA-D, best checkpoint) POT 结果:**
experiments/results.md:1349:### exp251: Tiny Multi-Stage PSG (Stage2+3) + PAA + LGPA-D+GCN
experiments/results.md:1353:| **exp251 FINAL** | **65.2%** | **76.2%** | 86.6% | 89.6% |
experiments/results.md:1354:| exp246b (Stage3 PSG+GCN) | 65.5% | 77.2% | — | — |
experiments/results.md:1357:- MSPSG+PAA vs baseline: **+8.6/+9.7** — 论文价值确认
experiments/results.md:1358:- MSPSG+PAA vs single-stage: -0.3/-1.0 (seed variance 内)
experiments/results.md:1359:- 结论: multi-stage PSG 作为 novel design 有效，但不额外超越 single-stage
experiments/results.md:1361:### exp253: Tiny 3-Stage PSG (Stage1+2+3, 无 PAA) + LGPA-D+GCN
experiments/results.md:1365:| **exp253 FINAL** | **65.1%** | **76.2%** | 87.0% | 89.5% |
experiments/results.md:1366:| exp251 (2-stage+PAA) | 65.2% | 76.2% | 86.6% | 89.6% |
experiments/results.md:1367:| exp246b (1-stage) | 65.5% | 77.2% | — | — |
experiments/results.md:1370:- 3-stage PSG ≈ 2-stage+PAA ≈ 1-stage (final 差异 <0.4 mAP)
experiments/results.md:1371:- PAA 无贡献 (exp253 vs exp251 = -0.1/0.0)
experiments/results.md:1372:- **所有 PSG 变体 vs baseline: +8.5~8.9 mAP** — multi-stage 可作为论文 presented method
experiments/results.md:1374:### 4090 Swin-Base LGPA-D+GCN+OA-SD+PLBOA 结果
experiments/results.md:1390:| Base LGPA+GCN LR=4e-4 (with PLBOA) | 93.8% | 96.8% |
experiments/results.md:1391:| Base LGPA+GCN LR=2e-4 (with PLBOA) | 93.1% | 96.8% |
experiments/results.md:1392:| Small PSG-only LR=4e-4 (no PLBOA) | 93.9% | 96.9% |
experiments/results.md:1397:### exp255: Small GCN512 + 2-stage PSG + LGPA-D + OA-SD — NEW SMALL BEST
experiments/results.md:1399:| 方法 | mAP | R1 | R5 | R10 | vs exp249 |
experiments/results.md:1401:| **exp255 FINAL** | **73.2%** | **83.3%** | 90.4% | 92.3% | **+1.3/+1.5** |
experiments/results.md:1402:| exp255 MaxSim ep100 | 73.3% | 83.4% | — | — | +0.2/+0.3 (同epoch) |
experiments/results.md:1404:- GCN512 + 2-stage PSG: equal_concat +1.3 mAP, 但 MaxSim 口径仅 +0.2
experiments/results.md:1405:- exp255b (GCN512 + 1-stage): ≈ baseline — 2-stage PSG 是 GCN512 发挥的关键!
experiments/results.md:1406:- **exp255 seed42 FINAL: 73.1/83.1** (vs seed1234 73.2/83.3 = -0.1/-0.2)
experiments/results.md:1407:- **exp255 seed2024 FINAL: 72.6/82.0** (vs seed1234 73.2/83.3 = -0.6/-1.3)
experiments/results.md:1410:### exp256: Pose Prompt (KPR-style) — 负面/中性
experiments/results.md:1414:| exp256 (GCN512+2stage+Prompt, 进行中) | ep90: 72.4 | 82.3 | — | -0.5 vs exp255 |
experiments/results.md:1415:| **exp256b (GCN256+1stage+Prompt) FINAL** | **68.8** | **79.3** | **70.3/81.0** | **-3.1 vs exp249** |
experiments/results.md:1419:- **exp256 FINAL: 72.7/82.4** (vs exp255 73.2/83.3 = -0.5/-0.9)
experiments/results.md:1420:- KPR-style discrete prompt confirmed negative on Swin+PSG
experiments/results.md:1422:### exp257: ArcFace + Label Smoothing — 负面
experiments/results.md:1424:| 变体 | mAP | R1 | vs exp255 | 备注 |
experiments/results.md:1426:| exp257 (ArcFace m=0.35+LS, 远程) | 59.1% | 76.5% | -14.1/-6.8 | ep55 终止, ArcFace 严重不收敛 |
experiments/results.md:1427:| exp257b (Label Smooth only, 本地) | 71.5% | 81.7% | -1.7/-1.6 | ep86 终止, LS 稳定负面 |
experiments/results.md:1432:### exp258: ArcFace m=0.2 / GCN 3-layer — 负面/中性
experiments/results.md:1434:| 变体 | mAP | R1 | vs exp255 | 备注 |
experiments/results.md:1436:| exp258 (ArcFace m=0.2, 本地) | 67.7% | 81.2% | **-5.5/-2.1** | ArcFace 证伪 |
experiments/results.md:1437:| exp258b (GCN 3-layer, 远程) | 73.1% | 82.7% | -0.1/-0.6 | GCN 3-layer ≈ 2-layer |
experiments/results.md:1442:### exp259: WD / OA-SD / DropPath 调参 — 全中性/负面
experiments/results.md:1444:| 变体 | mAP | R1 | vs exp255 | 备注 |
experiments/results.md:1446:| exp259 (WD=2e-4, 本地) | 72.2% | 82.1% | **-1.0/-1.2** | WD 过强负面 |
experiments/results.md:1447:| exp259b (OA-SD w=2.0, 远程) | 73.2% | 83.4% | 0.0/+0.1 | OA-SD=2.0 ≈ baseline |
experiments/results.md:1448:| exp259b MaxSim | 73.6% | 83.7% | +0.1/-0.1 | MaxSim 也持平 |
experiments/results.md:1449:| exp259c (dp=0.2, 本地, 进行中) | ep90: 72.6% | 82.7% | -0.6/-0.6 | dp=0.2 ≈ baseline |
experiments/results.md:1451:- **exp255 的 recipe (softmax CE, WD=1e-4, OA-SD=1.0, dp=0.1) 已是 SOLIDER Swin 上的最优 recipe。**
experiments/results.md:1452:- 所有 recipe 调参 (exp257-259) 均无法超越 baseline，recipe 空间已耗尽。
experiments/results.md:1454:### exp255 Test-Time Evaluations
experiments/results.md:1458:| exp255 equal_concat (baseline) | 73.2% | 83.3% | — | ep120 final |
experiments/results.md:1459:| exp255 global cosine | 72.7% | 82.3% | -0.5/-1.0 | global-only 模式 |
experiments/results.md:1460:| exp255 VisWeighted Part | 73.5% | 83.6% | +0.3/+0.3 | 可见部位加权 |
experiments/results.md:1461:| **exp255 MaxSim Hybrid** | **74.1%** | **84.6%** | **+0.9/+1.3** | **ep120 final, gw=1.0** |
experiments/results.md:1462:| **exp255 SGCFR α=0.5** | **74.0%** | **84.3%** | **+0.8/+1.0** | **top_k=5, vis_thr=0.3** |
experiments/results.md:1463:| exp255 SGCFR α=0.4 | 73.9% | 83.8% | +0.7/+0.5 | |
experiments/results.md:1464:| exp255 CVK hybrid α=0.7 | 72.2% | 82.6% | -1.0/-0.7 | CVK 无 recovery 反而负面 |
experiments/results.md:1465:| exp259b equal_concat | 73.2% | 83.4% | 0.0/+0.1 | OA-SD=2.0, ≈ exp255 |
experiments/results.md:1466:| exp259b MaxSim+flip | 75.1% | 85.4% | — | OA-SD=2.0, 略低于 exp255 (-0.1/-0.2) |
experiments/results.md:1468:| exp255 Global cosine+flip | 73.6% | 83.4% | +0.4/+0.1 | flip-test TTA |
experiments/results.md:1469:| **exp255 MaxSim+flip** | **75.2%** | **85.6%** | **+2.0/+2.3** | **⭐⭐⭐ 目标达成! flip-test+MaxSim** |
experiments/results.md:1474:### exp260: Base GCN512 + 2-stage PSG (LR=4e-4) — 未超 Small
experiments/results.md:1476:| 方法 | mAP | R1 | R5 | R10 | vs exp255 (Small) |
experiments/results.md:1483:### exp260b: Base GCN512 + 2-stage PSG (LR=8e-4) — 超越 Small!
experiments/results.md:1485:| 方法 | mAP | R1 | R5 | R10 | vs exp255 (Small) |
experiments/results.md:1495:### exp260b Market: Base GCN512 + 2-stage PSG (LR=8e-4, 无PLBOA)
experiments/results.md:1501:| Tiny+PSG | 92.4% | 96.7% | — | — | |
experiments/results.md:1502:| Small+PSG (无PLBOA) | 93.9% | 96.9% | — | — | |
experiments/results.md:1510:> 训练 scaffold: 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA(OD/OP) / PLBOA off(Market)，BS=64，LR=8e-4，120 epoch，SEED=42。  
experiments/results.md:1521:| exp294 | Swin-Base | Occ-Duke (LGPA-only / Full-GCN s41) | **74.0 / 82.6 (eq+flip), 75.0 / 84.4 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-24 02:18:48 lab4090 TEST.IMS_PER_BATCH 64 (R5=90.5 R10=92.4). Global 73.5/83.3, **MaxSim 75.0/84.4** (+1.0/+1.8 vs eq_concat)。**vs exp263d Full+GCN s41**: eq 74.1/83.3 → -0.1/-0.7, **MaxSim 75.2/84.8 → -0.2/-0.4** (GCN 冗余双评测模式都验证)。vs exp263b Full+GCN s42 MaxSim 74.8/84.0: **+0.2/+0.4** (Full-GCN s41 > Full+GCN s42)。补 Phase 3-C Base 行, **3-backbone 统一结论 GCN 可移除** |
experiments/results.md:1535:> Phase 3-A: **纯 PSG scaffold** (无 LGPA/GCN/OA-SD/PLBOA/Parallel-Aug),仅开 PSG 的 stage 数。单变量 = PSG_STAGES。回答"PSG 本体稳定性"。
experiments/results.md:1537:| Exp ID | Backbone | Dataset | PSG stages | eq_concat+flip(global) mAP / R1 | 备注 |
experiments/results.md:1540:| exp271 | Swin-Tiny | Occ-Duke | `[-1]` (1-stage) | **60.2 / 69.5** | ✓ e120 FINAL @ 2026-04-20 16:36 srvB (R5=81.8 R10=85.9). vs exp270 Δ=+1.0/+1.1 = stage 3 PSG 独立贡献 |
experiments/results.md:1544:| exp275 | Swin-Small | Occ-Duke | `[-1]` (1-stage) | **68.8 / 76.8** | ✓ e120 FINAL @ 2026-04-20 23:37 lab4090 (R5=87.2 R10=90.4). vs exp274 no-PSG Δ=**+0.7/0** (mAP 涨 R1 持平). vs Tiny 1-stage 增益 (+1.0/+1.1),Small 上 +0.7/0 缩水 |
experiments/results.md:1551:> Phase 3-B: **Full scaffold + 变量 GCN_HIDDEN × PSG_STAGES** (LGPA/OA-SD/ParAug/LOWER_BODY_OCC 全开,仅改 GCN 容量和 PSG stage)。单变量消融。回答"GCN cap 与 PSG stage 是否互补"。
experiments/results.md:1553:| Exp ID | Backbone | GCN_HIDDEN | PSG_STAGES | eq_concat+flip(global) mAP / R1 | 备注 |
experiments/results.md:1556:| exp278 | Swin-Tiny | 256 | `[-1]` | **65.7 / 76.7** | ✓ e120 FINAL @ 2026-04-21 10:42 srvB (R5=86.7 R10=89.6). vs exp261 GCN512+2stg 65.9/77.4 Δ=-0.2/-0.7. vs exp286 LGPA-only 66.0/76.6 Δ=-0.3/+0.1 (GCN256 略弱于 no GCN) |
experiments/results.md:1567:> Phase 3-C: **LGPA-only + 变量 PSG_STAGES** (关 GCN, 保留 LGPA/OA-SD/ParAug/LOWER_BODY_OCC)。回答"2-stage PSG 的收益是偏 structural 还是 semantic branch 也吃"。srvC exp266 silent exit 后空闲,利用上。
experiments/results.md:1569:| Exp ID | Backbone | PSG stages | mAP / R1 | 备注 |
experiments/results.md:1571:| exp286 | Swin-Tiny | `[-1]` | **66.0 / 76.6** | ✓ e120 FINAL @ 2026-04-21 10:03 srvC (R5=86.4 R10=89.7). **vs exp261 Full Scaffold 65.9/77.4 Δ=+0.1/-0.8** → GCN 对 Tiny 几乎无贡献, LGPA-only 等价 Full |
experiments/results.md:1572:| exp287 | Swin-Tiny | `[-2,-1]` | **65.9 / 77.0** | ✓ e120 FINAL @ 2026-04-21 20:48 srvC (R5=87.0 R10=89.7). vs exp286 LGPA-only 1stg 66.0/76.6 Δ=-0.1/+0.4 (2-stg R1 微优). vs exp261 Full 65.9/77.4 Δ=0/-0.4 (GCN 主要给 R1) |
experiments/results.md:1573:| exp288 | Swin-Small | `[-1]` | **73.8 / 83.8** | ✓ e120 FINAL @ 2026-04-22 12:51 srvC (R5=90.5 R10=92.0). 🔥 **vs exp285b Full Scaffold 73.8/83.8 完全持平** (mAP/R1 identical, R5/R10 微差 0.2/0.7)。vs exp282 Full GCN256+1stg 73.7/83.9 Δ=+0.1/-0.1。**证实 GCN 对 Small OD 零贡献**, LGPA 单独达 Full Scaffold 性能 |
experiments/results.md:1590:### exp295–304: 复现 / multi-seed / LR sweep / Phase 3-D LGPA 消融
experiments/results.md:1592:> Scaffold 默认 = 2-stage PSG `[-2,-1]` + LGPA-D + GCN512 + OA-SD + ParAug + PLBOA(OD)。单变量见「关键改动」列。
experiments/results.md:1596:| exp295 | Swin-Small | Full Scaffold 复现 exp255 | lab4090 / 1234 | **74.2 / 84.0** | **75.2 / 85.4** | vs exp255 hist 75.2/85.6: **0 / -0.2** | ✅ 完全重现 exp255 75.2 mAP，证历史数字真实可复现（非 eval bug）。**Small OD 主表新 reference** |
experiments/results.md:1602:| exp301 | Swin-Small | **LGPA OFF**（Phase 3-D） | lab4090 / 42 | 71.9 / 83.0 | 71.9 / 83.0（MaxSim **0 boost**） | vs exp285b Full 74.7/84.8: **-2.8 / -1.8** | LGPA 贡献 +2.8 mAP MaxSim；移除 LGPA → MaxSim 失去 boost（LGPA 是 MaxSim 主驱动） |
experiments/results.md:1607:### exp305–307: Tiny LGPA / PLBOA 消融（Phase 3-D 跨 backbone 补齐）
experiments/results.md:1611:| exp305 | Swin-Tiny | **LGPA OFF**（mirror exp301） | lab4090 / 42 | 64.5 / 76.0 | 64.5 / 76.0（**0 boost**） | vs exp261 67.2/78.6: **-2.7 / -2.6** | LGPA 贡献 +2.7 mAP MaxSim（+1.4 eq）；Phase 3-D Tiny+Small 双 backbone 完整 |
experiments/results.md:1624:| exp315 | Swin-Tiny | **POSE_LGPA_ASSIGN_WEIGHT 1.0**（LGPA aux ×2） | srvC(5060Ti) / 42 | 65.8 / 76.9 | 67.0 / 77.4 | vs exp261: **-0.2 / -1.2** | LGPA aux 加倍 net negative；default 0.5 sweet spot |
experiments/results.md:1625:| exp316 | Swin-Tiny | **POSE_OA_SD_WEIGHT 2.0** | lab4090 / 42 | 66.0 / 77.6 | 67.2 / 78.0 | vs exp261: **0 / -0.6** | OA-SD ×2 net neutral；default 1.0 sweet spot |
experiments/results.md:1626:| exp317 | Swin-Tiny | **POSE_LGPA_ASSIGN_WEIGHT 0.25**（LGPA aux ÷2） | lab3090 / 42 | 66.2 / 77.4 | 67.4 / 78.6 | vs exp261: **+0.2 / 0** ⭐ | sweep 中**唯一 MaxSim 超 baseline**(+0.2)，但在 multi-seed std 内，需 Small 验证 |
experiments/results.md:1628:| exp319 | Swin-Tiny | **POSE_OA_SD_WEIGHT 0.5** | srvC(5060Ti) / 42 | 65.8 / 76.8 | 67.1 / 78.1 | vs exp261: **-0.1 / -0.5** | OA-SD ÷2 slight neg；与 exp316(×2) 合证 default 1.0 sweet spot |
experiments/results.md:1629:| exp320 | Swin-Small | **POSE_LGPA_DETACH=False**（LGPA aux 反传 backbone） | lab4090 / 1234 | 68.1 / 79.3 | 68.8 / 79.6 | vs exp295: **-6.4 / -5.8** | **catastrophic -6.4 mAP**（e10 46% underfit）；证 LGPA detach 必要。强 negative 消融素材 |
experiments/results.md:1630:| exp321b | Swin-Small | **POSE_LGPA_ASSIGN_WEIGHT 0.25**（验证 exp317） | lab4090 / 1234 | 73.9 / 83.7 | 74.9 / 85.4 | vs exp295: **-0.3 / 0** | Tiny exp317 的 +0.2 **未迁移到 Small**（slight -0.3）→ 判 seed noise，保持 default 0.5 |
experiments/results.md:1751:| exp342 | CLIP + detached LGPA(外挂) | 60.0 | +0.2 冗余 |
experiments/results.md:1752:| **exp342b** | CLIP + **un-detach** LGPA | **60.7** | +0.9 vs exp341(但下行戳穿) |
experiments/results.md:1753:| **exp353** | **un-detach LGPA 无CLIP**(隔离) | **60.5** | pose单独已>CLIP单独59.8; 加CLIP只+0.2 |
experiments/results.md:1754:| **exp349** | 强系统 exp255(73.2) + CLIP | **71.4/71.3**(eq/global) | **CLIP有害 -1.8** |
experiments/decisions.md:172:- **exp007 (B): PSG backbone injection → mAP 58.3%, R1 67.9%。Phase 2 最佳结果！+1.7% mAP, +1.4% R1。超过 Phase 1 最佳 (58.0%/68.0%)。**
experiments/decisions.md:176:**上下文**: exp007 PSG 取得突破性结果 (58.3%/67.9%)。关键发现：
experiments/decisions.md:179:3. 额外参数仅 102K（两个 PSG 模块），几乎不增加计算量
experiments/decisions.md:183:  A. PSG + Part Pooling 组合 — 让 backbone 和 part branch 同时利用 pose
experiments/decisions.md:184:  B. PSG 消融实验 — 证明 PSG 每个组件的必要性
experiments/decisions.md:185:  C. PSG 在不同 stage 注入 — Stage 2 vs Stage 3 vs 全部 stages
experiments/decisions.md:186:  D. PSG 超参数分析 — hidden_dim, 是否 sigmoid, etc.
experiments/decisions.md:188:**选择**: A — PSG + Part Pooling 组合
experiments/decisions.md:191:1. PSG global feat (58.3%) 和 part-only feat (57.5%) 都有各自的优势
experiments/decisions.md:192:2. PSG 改善了 backbone 特征质量 → part features 也应该受益
experiments/decisions.md:193:3. 组合后可能进一步提升（PSG backbone + enhanced part features）
experiments/decisions.md:196:**执行结果**: exp008 mAP 57.7%, R1 66.0%。**组合不叠加**，低于 PSG-only (58.3%/67.9%)。Part pooling 的 part_only 测试模式丢弃了 PSG 增强的 global feature，而 part features 本身无法匹配 PSG-global 的质量。**结论：backbone-level injection 是更有效的 pose 利用方式，post-hoc pooling 在 PSG 基础上没有增量价值。**
experiments/decisions.md:200:**上下文**: exp007 (PSG) 和 exp008 (PSG+Part) 的对比揭示了重要规律：
experiments/decisions.md:201:1. PSG backbone injection: mAP 58.3% (+1.7%) — 全局特征，无 part branch
experiments/decisions.md:202:2. PSG + Part Pooling: mAP 57.7% (+1.1%) — part_only 测试，丢弃 global
experiments/decisions.md:206:- PSG 的增益主要来自改善全局特征质量，而 part pooling 依赖的是局部特征
experiments/decisions.md:208:- 在 part_only 测试模式下，PSG 增强的 global 特征被浪费了
experiments/decisions.md:211:  A. PSG + concat 融合 — 保留 PSG global + part features，不丢弃全局特征
experiments/decisions.md:212:  B. 多 stage PSG — 在 Stage 2 也注入 PSG，更早引入 pose 先验
experiments/decisions.md:213:  C. PSG 改进 — 更强的 gate 机制（如 channel attention, multi-head gate）
experiments/decisions.md:216:**选择**: 先做 B（多 stage PSG），这是架构级改进，有更大创新潜力
experiments/decisions.md:219:1. 当前 PSG 只在 Stage 3（2 个 block）注入，信息利用有限
experiments/decisions.md:221:3. exp005 证明 Stage 2 特征不足以直接做 identity classification，但这不代表 Stage 2 不适合做 spatial attention（PSG 不做分类，只做 spatial gating）
experiments/decisions.md:222:4. 多 stage PSG 是论文中可以画出更好架构图的设计
experiments/decisions.md:225:**执行结果**: exp009 mAP 58.3%, R1 67.2%, R5 81.2%, R10 85.2%。Multi-stage PSG (Stage 2+3) 与 single-stage (Stage 3 only) mAP 持平，R1 略低（-0.7%），R5/R10 略优（+0.4%/+0.3%），但增加了 156K 额外参数。**结论：Stage 2 PSG 无显著收益，pose spatial gating 在 Stage 3 已足够。后续聚焦于改进 PSG 机制本身，而非扩展注入范围。**
experiments/decisions.md:231:2. PSG backbone injection 最佳: +1.7% mAP (exp007)
experiments/decisions.md:232:3. PSG + Part Pooling 组合: 不叠加 (exp008)
experiments/decisions.md:233:4. Multi-stage PSG: 无额外收益 (exp009)
experiments/decisions.md:235:**核心发现**: PSG Stage 3 (2 blocks, 102K params) 是当前最优配置。进一步改进需要改变 PSG 的内部机制或训练策略。
experiments/decisions.md:238:  A. PSG Channel Attention — 在 spatial gate 基础上加 channel-wise attention (SE-style)
experiments/decisions.md:239:  B. Backbone Freeze Warmup — 冻结 backbone 前 N epochs，让 PSG 先学稳定的 gate 模式
experiments/decisions.md:240:  C. PSG + Global-Part Concat — concat PSG-global + part features (不是 part_only 测试)
experiments/decisions.md:241:  D. PSG 超参数搜索 — hidden_dim, sigmoid vs tanh, gate 初始化方式
experiments/decisions.md:248:2. 当前 PSG 零初始化，但训练初期 backbone 的梯度（来自 ID loss 和 triplet loss）会同时更新 backbone 和 PSG，可能让 PSG 来不及学到好的 gate pattern 就被 backbone 适应掉了
experiments/decisions.md:249:3. 冻结 backbone warmup 让 PSG 先在"固定"的特征空间上学习 pose-to-gate mapping，之后解冻 backbone 时 PSG 已有良好初始化
experiments/decisions.md:253:**执行结果**: exp010 彻底失败。ep30 mAP 仅 12.5%（exp007 ep30 ~49%），提前终止。冻结 backbone 导致：(1) 解冻后特征空间剧变使 PSG/classifier 失效 (2) PSG 学到错误的 gate pattern 产生负面干扰 (3) 训练指标看似正常但测试表现灾难性。**教训：PSG 必须和 backbone 端到端同时训练。**
experiments/decisions.md:258:- exp007 PSG Stage 3: mAP 58.3% (+1.7%) — 当前最佳
experiments/decisions.md:259:- exp008 PSG+Part: 不叠加
experiments/decisions.md:260:- exp009 Multi-stage PSG: 无额外收益
experiments/decisions.md:262:- **PSG 的基本配置（Stage 3, zero-init, 102K params）已被证明是最优的**
experiments/decisions.md:266:  A. PSG + Channel Attention (SE-style) — 增加 channel-wise gating 维度
experiments/decisions.md:268:  C. PSG + Global-Part Concat — 保留 PSG-global + part features 的拼接测试
experiments/decisions.md:275:1. 所有 PSG 实验都在 ep100-120 还在上升，曲线没有完全收敛
experiments/decisions.md:278:4. 200 epochs 可以测试 PSG 是否有更大的后段潜力
experiments/decisions.md:280:6. 如果 120→200 epochs 带来 0.5-1% 的额外提升，说明 PSG 确实需要更长训练
experiments/decisions.md:283:**执行结果**: exp011 最终 mAP 58.3%, R1 67.6%。与 exp007 (120ep) 完全相同的 mAP，75% 更多训练时间无收益。**结论：PSG 的性能上限由架构决定（~58.3% mAP），120ep 已足够。需要架构创新来突破。**
experiments/decisions.md:288:- **有效**: PSG Stage 3 (+1.7%), Part Pooling (+0.9%)
experiments/decisions.md:289:- **无效/中性**: Multi-stage PSG (=), PFM (=), Stage 2 Parts (❌), Part-Dominant (❌)
experiments/decisions.md:293:PSG Stage 3 (mAP 58.3%) 是确认的性能上限。所有尝试过的改进方向都无法突破这个上限。需要全新的方法。
experiments/decisions.md:295:**核心反思**: 当前 PSG 是一个简单的 spatial gate (17→64→768 的 1×1 conv)。它做的是"根据 pose heatmap 在空间维度上调制特征"。这个方法的局限性：
experiments/decisions.md:298:3. 不影响 self-attention — PSG 在 SwinBlock 之后作用，不改变 attention 计算
experiments/decisions.md:303:  B. PSG 3x3 conv — 在 PSG 中加 3×3 conv（depthwise），让 gate 有空间感受野
experiments/decisions.md:304:  C. Channel Attention PSG — 在 spatial gate 基础上加 channel attention (SE-style)
experiments/decisions.md:305:  D. PSG + Global-Part Concat Test — 测试时 concat PSG-global + part features（非 part_only）
experiments/decisions.md:314:4. 这改变了 token 之间的 attention 权重，比只改变 token 值（PSG 的做法）更根本
experiments/decisions.md:315:5. 如果 PCA > PSG，这是一个强有力的消融证据
experiments/decisions.md:317:**执行结果**: exp012 最终 mAP 57.4%, R1 67.3%。PAB 有效但弱于 PSG（-0.9% mAP, -0.6% R1）。尽管只有 5.4K 参数，attention bias 的调制效果不如 feature gate。**结论：在 Swin 的 window attention 中，additive bias decomposition (bias(i,j) = val[i] + val[j]) 的表达能力有限，softmax 压缩了 bias 的效果。Feature-level gating (PSG) 仍是更有效的 pose 注入方式。**
experiments/decisions.md:321:**上下文**: 12 个实验已完成。PSG 仍是最佳方法 (mAP 58.3%)。已验证：
experiments/decisions.md:323:- Backbone feature gating (PSG): +1.7% (exp007) ← BEST
experiments/decisions.md:325:- Multi-stage PSG: 无额外收益 (exp009)
experiments/decisions.md:326:- PSG+Part 组合: 不叠加 (exp008)
experiments/decisions.md:330:**关键排序**: Feature gating (PSG) > Post-hoc pooling > Attention bias (PAB)
experiments/decisions.md:335:  A. PSG + PAB 组合 — 同时做 feature gating 和 attention bias，双重 pose 注入
experiments/decisions.md:336:  B. PSG + 3×3 Depthwise Conv — 给 PSG 加空间感受野
experiments/decisions.md:338:  D. Stronger PSG Gate — 增大 hidden_dim 或加深 gate 网络
experiments/decisions.md:339:  E. Dual-Stream PSG — 分离 body-part gate 和 structure gate
experiments/decisions.md:341:**选择**: A — PSG + PAB 组合
experiments/decisions.md:344:1. PSG 和 PAB 作用在不同层面：PSG 调制 feature values, PAB 调制 attention patterns
experiments/decisions.md:345:2. 两者理论上互补：PAB 让 attention 关注正确的位置，PSG 增强这些位置的特征
experiments/decisions.md:350:**执行结果**: exp013 最终 mAP 57.6%, R1 67.2%。**组合未能超越 PSG-only (-0.7% mAP, -0.7% R1)**。与 exp008 类似的规律：两种 pose 注入方式在同一层（Stage 3）互相干扰。PAB 修改了 attention 分布导致 PSG 基于的特征分布发生变化，PSG 的 gate 不再最优。**结论：单一高质量 pose 注入点（PSG）比多个中等质量注入点更好。**
experiments/decisions.md:355:- PSG Stage 3 only: +1.7% mAP (exp007) ← BEST
experiments/decisions.md:356:- Multi-stage PSG: 无额外收益 (exp009)
experiments/decisions.md:358:- PSG + PAB combo: +1.0% mAP (exp013，不如 PSG alone)
experiments/decisions.md:359:- PSG + Part Pooling: +1.1% mAP (exp008，不如 PSG alone)
experiments/decisions.md:362:**核心认知**: PSG 的 58.3% 性能上限是由**pose heatmap 本身的信息量**和**Stage 3 的 2 个 block**共同决定的。在 backbone 内部叠加更多 pose 模块只会互相干扰。
experiments/decisions.md:367:  C. PSG 内部机制改进 — 3×3 depthwise conv, channel attention
experiments/decisions.md:369:  E. PSG + 不同 loss 组合 — 如 per-part triplet loss (GiLt) 在 PSG backbone 上
experiments/decisions.md:371:**选择**: E — PSG + per-part triplet loss (GiLt)
experiments/decisions.md:374:1. PSG 只改善了特征质量，但训练信号仍然是标准的 ID+triplet loss
experiments/decisions.md:375:2. Per-part triplet loss 可以给 backbone 提供更细粒度的梯度信号，可能进一步提升 PSG 增强的特征
experiments/decisions.md:376:3. Phase 1 中 GiLt 在 PCFC 基础上额外 +0.5%，PSG 基础上可能也有类似增益
experiments/decisions.md:377:4. 这是正交方向的改进（module 级 vs loss 级），不会像 PAB 那样干扰 PSG
experiments/decisions.md:378:5. 实现简单：在 PSG 增强的特征图上做 pose-guided part pooling，然后每个 part 独立 triplet loss
experiments/decisions.md:379:6. 论文价值：训练信号层面的 pose 利用，与 backbone 层面的 PSG 互补
experiments/decisions.md:381:**执行结果**: 用户指出 exp014 与 exp008 训练完全相同，直接加载 exp008 checkpoint 做 global test 即可。结果：mAP 57.6%, R1 65.8%。**比 PSG-only 低 0.7% mAP, 2.1% R1。Part supervision 确实损害了 PSG global feature。** "loss 级正交改进"的假设不成立——part supervision 的梯度回传到 backbone 时会干扰 PSG gate 的学习。
experiments/decisions.md:386:- **有效**: PSG Stage 3 only (+1.7% mAP) — 唯一突破
experiments/decisions.md:388:- **所有组合尝试都失败**: PSG+Part, PSG+PAB, PSG+Part(global test)
experiments/decisions.md:391:**关键规律**: 任何修改 PSG 训练时梯度流的改动都会降低 PSG 效果。PSG 的 58.3% 是"全局 ID+triplet + pose spatial gating"的最优解。
experiments/decisions.md:393:**战略性反思**: 继续在 PSG 基础上小修小补已经穷尽了选项。需要跳出"在 backbone 里加东西"的思路。
experiments/decisions.md:396:  A. PSG 机制改进 — 3×3 depthwise conv, channel attention (仍在 backbone 内部)
experiments/decisions.md:398:  C. PSG + Pose-Guided Data Augmentation — 训练数据层面的 pose 利用
experiments/decisions.md:399:  D. 全新方向: Pose-Guided Feature Disentangling — 用 PSG feature 做 pose/appearance 解耦
experiments/decisions.md:402:**选择**: A — PSG 内部机制改进（3×3 depthwise conv），而非 C
experiments/decisions.md:405:1. 这是对 PSG gate 本身的改进，不增加新模块或新 loss，不会像 exp008-014 那样干扰梯度流
experiments/decisions.md:411:**执行结果**: exp015 训练完成。mAP 58.3% 与 exp007 完全持平, R1 67.1% 低 0.8%。3×3 depthwise conv 是冗余的，1×1 gate 已是最优。训练过程中波动极大（差距从 -2.2% 到 +2.3%），但最终收敛到相同水平。PSG 的瓶颈不在感受野。
experiments/decisions.md:415:**上下文**: exp015（PSG 空间卷积改进）完成，与原始 PSG 持平。至此：
experiments/decisions.md:416:- PSG 内部结构改进（depthwise conv）：无效
experiments/decisions.md:417:- PSG 外部组合（PAB, Part Pooling, Part Supervision）：全部有害
experiments/decisions.md:418:- PSG 训练策略（freeze, 200ep, multi-stage）：无效
experiments/decisions.md:420:**已穷尽的方向**: 在"PSG + 全局 ID/Triplet Loss"框架内的所有优化都已探索完毕。PSG 58.3% mAP 是该框架的理论上限。
experiments/decisions.md:422:**需要根本性的方向转变**: 不再在 PSG 上修修补补，需要全新的利用 pose heatmap 的方式。
experiments/decisions.md:428:  D. Adaptive PSG — 根据遮挡程度动态调节 PSG 强度
experiments/decisions.md:429:  E. Deformable PSG — 可变形卷积替代固定网格，对齐到关键点位置
experiments/decisions.md:433:**执行结果**: exp016 完成。PGE 严重有害（mAP 54.8% vs exp007 58.3%，-3.5%）。身体部件级擦除过强+削弱 PSG 输入。数据增强层面的 pose 利用方向失败。
experiments/decisions.md:438:- PSG 58.3% 是唯一有效方法
experiments/decisions.md:442:**关键反思**: 已经把 PSG 本身和所有"加法"都试遍了。需要完全不同的思路。
experiments/decisions.md:446:  B. 全新模型: Deformable PSG — 可变形卷积对齐关键点
experiments/decisions.md:448:  D. 测试端优化: 基于 PSG 的 NFC/Re-ranking
experiments/decisions.md:449:  E. 跨步思考: 不再改 PSG，而是替换整个 Part Pooling 方案
experiments/decisions.md:451:**选择**: 先尝试 Pose-Conditioned Channel Gate (PCG)——在 GAP 后做通道级 pose gating，与 PSG 的空间级正交
experiments/decisions.md:453:**执行结果**: exp017 完成。PCG 与 exp007 基本持平（mAP 58.0% vs 58.3%，-0.3%）。证实了通道级正交操作不干扰 PSG（不像之前的 combo 实验那样降性能），但 GAP 后的 pose 信息太弱无法提供额外收益。
experiments/decisions.md:458:- PSG 58.3% 仍是最佳，所有扩展都无法超越
experiments/decisions.md:459:- **新发现**: 正交维度操作不干扰 PSG（PCG 仅 -0.3%），但空间级操作干扰 PSG（PAB combo -0.7%, Part pooling -0.6%）
experiments/decisions.md:460:- 这说明 PSG 的空间 gating 格局已定，但通道维度仍有空间
experiments/decisions.md:466:  B. PCG-only (无 PSG) — 验证 PCG 单独效果，排除 PSG 已经做了足够 conditioning 的可能
experiments/decisions.md:467:  C. 完全新方向：跳出 PSG 框架
experiments/decisions.md:471:**执行结果**: exp018 完成。**PCG-only mAP 57.8%, R1 67.7%（+1.2%/+1.2% vs baseline）。** 出乎意料，PCG 有独立效果！推翻了"PCG 无效"的初步判断。但仍低于 PSG (58.3%/-0.5%)，且 PSG+PCG (exp017) 不叠加。结论：PCG 和 PSG 各自有效但捕获相似的 pose 信号，组合不互补。
experiments/decisions.md:477:**上下文**: exp019 PXA (Pose Cross-Attention) 训练完成。最终 mAP 57.3%, R1 66.9%。超过 baseline (+0.7%) 但落后 PSG (-1.0%)。
experiments/decisions.md:480:- PXA 证明了 cross-attention 方式注入 pose 信息有效（>baseline），但效果不如 PSG 的简单空间门控
experiments/decisions.md:481:- 原因：cross-attention 表达能力过强导致过拟合（训练 acc 99.5%，loss 0.10），PSG 的逐元素乘法有更强归纳偏置
experiments/decisions.md:482:- PXA 和 PSG 不太可能组合使用（都在 Stage 3 blocks 中间注入，已有先例显示 PSG+其他会互相干扰）
experiments/decisions.md:484:**核心结论**: 三种 pose 注入机制的排序确认：PSG(空间门控) > PXA(跨注意力) > PCG(通道门控, GAP后)。简单的空间逐元素操作始终优于复杂机制。
experiments/decisions.md:486:**下一步**: 启动 exp020 (PSG + Pose Reconstruction Auxiliary)，测试**正交方向**——不修改 PSG 前向传播，仅通过辅助重建任务提供额外梯度信号。这是突破 58.3% 上限的希望所在。
experiments/decisions.md:492:**上下文**: exp020 PRA 完成。mAP 57.8%, R1 67.3%。中性结果，不如 PSG (58.3%)。至此，20 个实验全部无法超越 PSG (exp007)。
experiments/decisions.md:494:**总结过去尝试突破 PSG 的失败**:
experiments/decisions.md:495:- exp008 PSG+Part Pooling: 57.7% ❌ 组合干扰
experiments/decisions.md:496:- exp009 Multi-stage PSG: 58.3% 🟡 匹配但不超越
experiments/decisions.md:497:- exp012 PAB: 57.4% ❌ 弱于 PSG
experiments/decisions.md:498:- exp013 PSG+PAB: 57.6% ❌ 互相干扰
experiments/decisions.md:499:- exp015 PSG Spatial 3×3: 58.3% 🟡 匹配但不超越
experiments/decisions.md:500:- exp017 PSG+PCG: 58.0% ❌ 不叠加
experiments/decisions.md:502:- exp020 PSG+PRA: 57.8% ❌ 梯度干扰
experiments/decisions.md:504:**核心问题**: 所有**添加额外模块/任务**到 PSG 的尝试都失败。PSG 似乎已经是"局部最优"——它简单有效，但不接受增强。
experiments/decisions.md:506:**新思路**: 不添加模块到 PSG，而是**改进 PSG 本身**。核心问题：PSG 的门控是**静态的**——给定相同的 heatmap，不同图像得到相同的 gate。如果让 gate 同时依赖 pose 和当前特征内容（Content-Adaptive PSG / CAPSG），可能打破这个限制。
experiments/decisions.md:509:A. Content-Adaptive PSG (CAPSG): gate = f(pose, features) 而非 gate = f(pose)
experiments/decisions.md:510:B. PSG + 超参调优 (weight decay, dropout, label smoothing)
experiments/decisions.md:511:C. 接受 PSG 58.3% 作为最终方法，开始写论文
experiments/decisions.md:513:**选择**: A — CAPSG
experiments/decisions.md:516:1. CAPSG 是对 PSG 机制本身的改进，而非外挂模块，避免了"组合干扰"问题
experiments/decisions.md:517:2. 与 PXA (cross-attention) 不同，CAPSG 保持了 PSG 的逐元素乘法范式，只是让乘法因子变成 content-dependent
experiments/decisions.md:519:4. 零初始化保证初始行为等同 PSG，只有学到有用的 content-feature 交互才会偏离
experiments/decisions.md:521:**执行结果**: exp021 完成。**CAPSG mAP 57.2%, R1 66.0%（-1.1% vs PSG）。** Content-adaptive gate 不如静态 pose-only gate。CAPSG 前期慢启动（ep20 落后 -4.2%），虽多次追近但从未在后段超越 PSG。关键洞察：PSG 的静态 pose-only gating 不是局限而是优势——ReID 需要的是一致的空间先验，不是动态调制。
experiments/decisions.md:527:**上下文**: 21 个实验完成，PSG (exp007) 仍是唯一超过 baseline 1.5%+ 的方法。已尝试的所有突破方向均失败。
experiments/decisions.md:530:- **有效**: PSG (+1.7%), Part Pooling (+0.9%), PCG-only (+1.2%), PXA (+0.7%), PRA (+1.2%)
experiments/decisions.md:531:- **最佳**: PSG 58.3% — 简单空间门控
experiments/decisions.md:532:- **组合全部失败**: PSG+Part, PSG+PAB, PSG+PCG, PSG+PRA — 要么中性要么有害
experiments/decisions.md:533:- **改进 PSG 也失败**: Multi-stage PSG, PSG Spatial, CAPSG — 匹配或弱于 PSG
experiments/decisions.md:534:- **复杂机制全部不如简单门控**: PXA < PSG, CAPSG < PSG
experiments/decisions.md:536:**核心洞察**: 对于 Swin-Tiny backbone + pose heatmap 的组合，简单的逐元素空间门控（PSG）已经是最优解。所有增加复杂度的尝试都是负面的。
experiments/decisions.md:539:A. 接受 PSG 作为最终方法，转向完善论文（跨数据集实验、可视化、效率分析）
experiments/decisions.md:541:C. 探索 PSG 在更长训练、不同 LR 下的潜力
experiments/decisions.md:542:D. 在不同数据集（Market-1501）上验证 PSG 泛化性
experiments/decisions.md:544:**选择**: D — Market-1501 上验证 PSG 泛化性（已由用户在 4090 完成）
experiments/decisions.md:546:**执行结果**: 用户已在 4090 上完成所有跨数据集实验。PSG 在所有配置上均有效（Occluded-Duke Swin-Small +2.0%, Market-1501 Swin-Tiny +0.8%, Market-1501 Swin-Small +0.6%）。
experiments/decisions.md:554:2. 可以完全放弃 PSG 框架
experiments/decisions.md:559:- KP-RPE (关键点相对位置编码): CVPR24 人脸识别，与 PSG 正交
experiments/decisions.md:570:1. 21 个实验证明梯度干扰是 PSG 组合失败的根本原因。PDS 从架构层面解决这个问题。
experiments/decisions.md:573:4. 如果 PDS 有效，结合 PSG 在 Global 分支 + 结构化 Part 分支，论文故事非常完整
experiments/decisions.md:576:**执行结果**: PDS 训练完成。global-only mAP 57.9% (vs PSG-only 58.3%)，Stage 3 解耦有效但共享 Stage 0-2 仍有轻微干扰。Part 分支独立效果一般 (55.2%)。PDS 未超过 PSG-only。
experiments/decisions.md:580:**上下文**: exp022 PDS 结果分析 — global-only 57.9% (接近但未超过 PSG-only 58.3%)。Part 分支 ID loss (2.02) 远未收敛到 Global (0.17) 水平，Part 特征质量不足。
experiments/decisions.md:594:**执行结果**: 🎉 **突破性成功！** exp023 global-only mAP 59.5% (+1.2% vs PSG-only 58.3%, +2.9% vs baseline)。
experiments/decisions.md:603:1. PDS+StopGrad global-only (59.5%) > PSG-only (58.3%) > PDS global (57.9%)
experiments/decisions.md:609:A. 对 exp023 结果做完整消融：单独评估 PSG 贡献、Part 贡献、stop_grad 贡献
experiments/decisions.md:617:**上下文**: exp024 (PDS+StopGrad 无 PSG 消融) 在 epoch 60 达到 equal_concat mAP 53.9%（vs exp023 54.4%，-0.5%），趋势明确但被误杀了 DataLoader worker 导致终止。
experiments/decisions.md:619:用户提出关键质疑：**exp023 的 +1.2% 提升（vs exp007 PSG-only）在理论上说不通**——如果 Part 梯度被阻断，Global 分支应该和 PSG-only 完全等价。分析后认为 +1.2% 很可能是训练随机性差异（不同模型类的初始化消耗不同随机状态），而非架构真实贡献。
experiments/decisions.md:632:**理由**: exp025 是用户指定的高优先级实验，且 exp024 的趋势数据已足够（6 个 epoch 评估点都确认了无 PSG 的影响）
experiments/decisions.md:648:- 新证据：exp025 global 58.9% > exp007 PSG-only 58.3% (+0.6%)，说明 Part 分支确实有正面影响
experiments/decisions.md:656:4. exp024 (No PSG) — 消融：证明 PSG 在 Global 分支中的贡献
experiments/decisions.md:667:  C. Stochastic Pose Dropout (SPD) — 全新方向，在 PSG 基础上做正则化
experiments/decisions.md:673:1. 最简单实现（~5 行代码），最低风险（最差情况 = PSG baseline 58.3%）
experiments/decisions.md:675:3. 基于确认有效的 PSG 架构（58.3%），不依赖尚未验证的 PDS
experiments/decisions.md:677:5. 论文 story 清晰："PSG 教 backbone 在哪里关注，SPD 防止过度依赖"
experiments/decisions.md:679:**执行结果**: mAP 57.9% (-0.4% vs PSG)。SPD 略微负面，证明 pose 信号在 Occluded-Duke 上一致有用
experiments/decisions.md:693:- 🔴 红队（方案 A: PCRA）核心论点: PCRA 是唯一操作在未被探索维度（loss 距离度量）上的方案。26 个实验证明所有 forward path 改动都干扰 PSG，但 PCRA 不修改 forward path 也不添加 aux loss——它只改变 hard mining 中的距离计算。这是论文方法的"第三层"贡献（特征层 PSG + 架构层 PDS + 度量层 PCRA）。实现仅~20 行代码，0 新参数。风险极低：最差退化为 PSG baseline。Occluded-Duke 中 pose 相似的 negative 才是真正的 hard case（两个只露上半身的不同人），标准 triplet loss 对此毫无感知。PVR 的弱点在于 exp020 (PRA) 已证明辅助 loss 方向中性/负面(-0.5%)，且需要 part feature 提取（与 exp008 的失败模式重叠）。SPD 调参无论文价值。信心: 8/10
experiments/decisions.md:694:- 🔵 蓝队（方案 B: PVR）核心论点: PVR 零 forward path 改动，在 26 个实验证明"所有 forward 改动都干扰 PSG"的历史下最安全。与 PSG 理论互补（PSG 约束幅度分布，PVR 约束语义结构）。复用现有 heatmaps_to_parts() 基础设施。exp020 (PRA) 失败是因为重建任务梯度方向与 ID loss 不一致，PVR 的结构正则化方向与 ID loss 一致（同部位同 ID → 应相似）。PCRA 的弱点在于修改了核心度量学习过程（exp003 修改 loss 权重导致 -6.4%），且引入 O(B²) 批内 pose 比较和新的超参数设计空间。信心: 7/10
experiments/decisions.md:703:**执行结果**: ❌ PCRA (alpha=0.2) 得到 mAP 57.8%, R1 66.8%，低于 PSG -0.5%/-1.1%。pose similarity 调制在 triplet 距离中引入了训练不稳定性（锯齿形 mAP 波动）。17 维 pose signature 不够精确区分姿态差异。
experiments/decisions.md:709:**上下文**: exp027 PCRA 结果中性偏负 (-0.5% mAP)。至此，所有在 PSG 基础上的单点改进（forward path 添加、aux loss、距离度量调制、dropout 正则化）均未能超越 PSG。唯一成功的方向是 PDS+StopGrad (exp023, +2.9% mAP)，但 exp024 证明其中 PSG 的贡献很小（仅 0.3%）。需要决定下一步方向。
experiments/decisions.md:719:1. PDS+StopGrad 是唯一成功超越 PSG 的方向，直接在其上改进是最高 ROI 的选择
experiments/decisions.md:723:5. 全新范式（20 个 PSG 改进全失败的历史）成功概率不高
experiments/decisions.md:742:  B. 回归 PSG 路线但走全新范式（如 Pose-Guided Feature Selection — 基于热图做 token pruning/routing）
experiments/decisions.md:743:  C. 探索 PSG + PDS 的更深层结合（如 Part 分支用不同的 pose modulation）
experiments/decisions.md:747:1. PSG 和 PDS 都是在 backbone 输出后处理，但 token-level 操作是 Transformer 的原生语言
experiments/decisions.md:753:**执行结果**: exp029 (PWP) 完成。mAP 57.9%, R1 67.5% — 低于 exp007 (58.3%) -0.4%。**PWP 是 PGTS 的 soft 版本，结果中性偏负。** Post-backbone 的加权 pooling 在 PSG 已完成空间选择的情况下是冗余操作。关键启示：如果要做 token-level 操作，必须在 Stage 3 **内部** 做（如 hard token pruning），而不是在 pooling 阶段。
experiments/decisions.md:759:**上下文**: exp029 (PWP) 完成后，29 个实验全部结束。PDS+StopGrad global-only 59.5% 仍是最佳。所有 PSG 改进（21 个）和 Part 改进（exp028, exp029）均失败。需要决定下一步方向。
experiments/decisions.md:805:**上下文**: 用户分享了 codex 对 PDS+StopGrad 的分析——发现 loss 聚合中 global loss 被隐式乘以 0.5。multi-seed 实验确认 PDS+StopGrad 相对 PSG 有稳定的 +1.4% 增益。需要验证这个增益是来自架构还是 loss weighting。
experiments/decisions.md:808:  A. 在 exp007 (PSG-only) 上显式乘 0.5x global loss，验证 loss weighting 假设
experiments/decisions.md:811:**选择**: A — exp007a (PSG + 0.5x global loss)
experiments/decisions.md:820:- **重大发现**: 仅 +102K params (PSG) + 0.5x loss = 6.3M params (PDS) + StopGrad 的相同效果
experiments/decisions.md:831:  B. PSG + 0.5x Loss + Skeleton GCN 无 PDS (exp030a): 验证 GCN 能否不需要独立 Stage 3
experiments/decisions.md:835:- 🔵 蓝队（方案 B）核心论点: 架构简化问题的信息价值远高于超参调优。如果成功，整个方法简化为 PSG+loss+GCN (~0.6M params vs PDS 6M+)，论文 story 极其简洁。仅 1 个实验 ~5h。即使失败也有价值（说明 PDS 必要）。信心: 8/10
experiments/decisions.md:838:**选择**: B — exp030a (PSG + 0.5x loss + GCN, 无 PDS)
experiments/decisions.md:843:**上下文**: exp030a 训练完成。结果：PSG+GCN (无 PDS) equal_concat 达到 mAP 61.1%, R1 73.7%，为全实验最佳。完全验证 GCN 不需要独立 Stage 3。
experiments/decisions.md:852:  A. 消融实验：分离 loss scaling 和 GCN 的独立贡献（exp030b: PSG+GCN+1.0x loss，即不使用 0.5x scaling）
experiments/decisions.md:865:| exp007 | PSG only | 58.3% |
experiments/decisions.md:866:| exp007a | PSG + 0.5x loss | 59.5% |
experiments/decisions.md:867:| exp030a | PSG + GCN (w_p=1.0, 隐式 0.5x) | 59.8% |
experiments/decisions.md:868:| exp030b | PSG + GCN (w_p=0.01, ≈1.0x) | 60.6% |
experiments/decisions.md:879:  A. 立即设计多种子实验 (3-5 seeds)，在 3090 上跑 baseline/PSG/PSG+GCN 各 3 次
experiments/decisions.md:886:1. exp007 (PSG only, 1.0x loss) — 基线
experiments/decisions.md:887:2. exp007a (PSG + 0.5x loss) — loss scaling 效果
experiments/decisions.md:888:3. exp030a (PSG + GCN, equal_concat) — GCN + loss scaling 组合
experiments/decisions.md:903:| PSG | 57.83% | 0.50% | +1.33% | ~0.054 |
experiments/decisions.md:907:1. PSG +1.33% 所有 seeds 正向，但边缘显著
experiments/decisions.md:909:3. PDS+SG vs PSG = +1.37%，极一致 (1.4/1.3/1.4)
experiments/decisions.md:914:- PSG 确认为有效贡献 (+1.33% mean)
experiments/decisions.md:916:- 但 PDS+StopGrad 增加 ~8.8M 参数（完整 Stage 3 复制），效率比 PSG 差很多
experiments/decisions.md:918:- 需要理解 PDS+StopGrad 为何在 loss scaling 无效的情况下仍优于 PSG
experiments/decisions.md:962:- PSG 仍是主创新点之一（稳定正向、参数极轻）
experiments/decisions.md:968:1. PSG
experiments/decisions.md:1107:**上下文**: exp148 PCVT 和 exp151 PVAT 全部失败。训练集 95.8% 可见率使得所有 visibility-dependent 训练方法无效。
experiments/decisions.md:1121:- exp152 (soft MaxSim, tau=0.05) → 远程
experiments/decisions.md:1122:- exp152b (hard MaxSim, tau=0.005) → 本地
experiments/decisions.md:1136:### [2026-03-23 02:30] 决策 #N: exp148/149/151 结论与训练集 visibility 关键发现
experiments/decisions.md:1138:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/decisions.md:1143:1. **exp148 PCVT**: 早期加速（ep30 +2.4 mAP），后期被基线追平并反超（ep100 -0.9 mAP）。3-view 训练的 1/3 主损失稀释 + 训练数据缺乏 visibility 多样性。
experiments/decisions.md:1144:2. **exp149 SCFA**: ep30 即止损（-1.5 mAP / -4.7 R1）。bilateral gap case 太少。
experiments/decisions.md:1145:3. **exp151 PVAT**: 中性。pvat_acc 从未下降（0.83 constant），gradient reversal 无法影响 backbone，因为训练数据几乎全可见。
experiments/decisions.md:1206:- 🔴 红队（方案 A - LKA）核心论点: 最低风险最低成本（~600 params, ~20 行），exp035b 证明权重方案敏感（score→score*vis -0.7%），confidence ≠ 判别重要性。可解释性高（训练后可可视化哪些关键点最重要）。攻击 B: CAPSG 前车之鉴（内容自适应门控失败 -1.1%），fusion 已接近最优。攻击 C: exp005 灾难。信心: 7/10
experiments/decisions.md:1207:- 🔵 蓝队（方案 B - AFF）核心论点: exp036 证明 GCN 特征已饱和，继续优化 GCN 内部（LKA/C）是优化已饱和子系统。equal_concat vs concat_scaled p=0.0039 证明融合权重极敏感。AFF 完成 PSG→GCN→AFF 的"感知→补全→融合"完整链条，论文故事最佳。攻击 A: alpha suppression 前车之鉴，LKA 在已饱和系统上增量有限。信心: 8/10
experiments/decisions.md:1532:2. 失败的共同根因：PSG+GCN 已将 ID+Triplet 的训练优化到一个局部最优，任何辅助 loss 要么干扰主目标（SGMKC），要么信号太弱被主目标淹没（PAMC），要么机制不成立（CSGT）
experiments/decisions.md:1576:- 🔵 蓝队（方案 B: PPE）核心论点: 重新定义遮挡问题（从 visibility mask 到 uncertainty quantification），pose confidence → feature uncertainty 的因果链是全新机制。论文理论深度更强（概率嵌入+信息几何）。与 PSG 正交互补（PSG 管特征提取，PPE 管匹配）。P3E/DUL 已验证概率嵌入在 ReID 中可训练。攻击 A: PAB 已在同范式失败（attention bias 变体），CLAUDE.md 警告"attention trick"，2.8K 参数太小难以撑论文主贡献，12x4 分辨率太粗糙。信心: 7/10
experiments/decisions.md:1581:1. KP-RPE 是 backbone 级改动（与 PSG 同类），不在 CLAUDE.md 警告的"branch 内 trick"范围内
experiments/decisions.md:1584:4. 即使失败，诊断信息清晰（确认 attention bias 在 PSG 存在时是否冗余）
experiments/decisions.md:1595:- 在 PSG+GCN 框架上的增量修改空间已耗尽
experiments/decisions.md:1598:  A. 继续扩展 KP-RPE（更大容量、更多层、与 PSG 更紧密集成）
experiments/decisions.md:1618:- 有效：PSG（乘性门控）、GCN（结构特征 fusion）、CVK（test-time pairwise matching）
experiments/decisions.md:1629:  A. **Pose-Guided Token Selection + Cross-Attention (PGTCA)**: 用 PSG 热图做 token 重要性评分，选出可靠 token，再用 keypoint-guided cross-attention 提取 part 特征。本质上替换当前 GCN branch 为更强大的 cross-attention 解码器。
experiments/decisions.md:1636:3. 可以利用已有的 PSG 基础设施，在其上构建解码器
experiments/decisions.md:1637:4. 论文 story：PSG (backbone 注入) + Cross-Attention Decoder (结构化解码) 形成完整的 encode-decode 范式
experiments/decisions.md:1670:  - 保留 `PSG + GCN + PAA + 0.5x loss`
experiments/decisions.md:1674:1. `PSG` 继续使用 `scene_heatmap`
experiments/decisions.md:1708:4. 如果有效：可以讲 "PSG suppress + PGFI recover" 的互补 story
experiments/decisions.md:3003:  - `exp148 PCVT`
experiments/decisions.md:3004:  - `exp149 SCFA`
experiments/decisions.md:3012:- `exp148 PCVT` 已给出连续 `ep10/20/30` 的稳定 `mAP` 正向
experiments/decisions.md:3013:- `exp149 SCFA` 已在预设的 `ep30` 快速止损窗口内明确判负
experiments/decisions.md:3039:- 已终止远程 `exp149`
experiments/decisions.md:3041:- 本地主线继续保留 `exp148 PCVT`
experiments/decisions.md:3046:**上下文**: exp190-195 系列实验完成，揭示了 OA-SD 和 3-view parallel aug 的组合关系，以及 OA-SD global-only 解决 SupCon 梯度冲突的新机制。
experiments/decisions.md:3049:- exp190 (3-view+CE): 64.2/75.6 — 3-view 是最强单一技术
experiments/decisions.md:3050:- exp191 (OA-SD+CE): 63.2/75.4 — OA-SD 独立有效
experiments/decisions.md:3051:- exp192 (decay=0.99): 62.6/74.9 — decay 不敏感
experiments/decisions.md:3052:- exp193 (3-view+OA-SD+CE): 64.4/76.5 — additive! R1 追平 SupCon
experiments/decisions.md:3053:- exp194 (weight=2.0): 63.4/74.8 — weight 不敏感
experiments/decisions.md:3054:- exp195 (SupCon+OA-SD global-only): ep70=60.2/73.4 — 梯度冲突解决!
experiments/decisions.md:3057:  A. exp196: 3-view + SupCon + OA-SD global-only（终极组合，验证所有创新 additive）
experiments/decisions.md:3063:1. OA-SD global-only 是一个新机制（满足创新门槛 #2 和 #3），不是简单组合
experiments/decisions.md:3064:2. 如果 exp196 > exp187 (64.9/76.6)，则创论文主表新高
experiments/decisions.md:3065:3. exp195 已验证 SupCon+OA-SD global-only 兼容，exp193 已验证 3-view+OA-SD additive
experiments/decisions.md:3072:**上下文**: exp196 (3-view + SupCon + OA-SD global-only) 在 ep70 持续落后 exp187 (3-view + SupCon) -1.8/-0.9。OA-SD global-only 的 distillation 信号 (oa_sd=0.01) 过弱。
experiments/decisions.md:3075:- OA-SD + CE 有效 (+2.9/+2.6)
experiments/decisions.md:3076:- OA-SD + SupCon (all-token) 有梯度冲突 (exp188 负向)
experiments/decisions.md:3077:- OA-SD + SupCon (global-only) 无梯度冲突但信号太弱 (exp195/196)
experiments/decisions.md:3078:- 结论: **OA-SD 和 SupCon 本质上互斥**——选一个用
experiments/decisions.md:3081:  A. 最终配置用 SupCon (exp187: 64.9/76.6) — 测试时最强
experiments/decisions.md:3082:  B. 最终配置用 OA-SD+CE (exp193: 64.4/76.5) — R1 几乎一样
experiments/decisions.md:3090:3. OA-SD 路线: R1 几乎一样，且 OA-SD 概念更新颖 (self-distillation 范式)
experiments/decisions.md:3096:**上下文**: 5 个研究 agent 并行调研后，发现 Relational Knowledge Distillation (RKD, CVPR 2019) 可能解决 OA-SD vs SupCon 互斥问题。OA-SD 在 token 级别 match 个体特征 → 与 SupCon 冲突。RKD match pairwise similarity structure → 不碰个体特征 → 理论上与 SupCon 兼容。
experiments/decisions.md:3106:1. 直接解决已知问题 (OA-SD vs SupCon 冲突)
experiments/decisions.md:3107:2. 满足创新门槛: 问题层面(关系级 vs 特征级) + 机制层面(RKD in occluded ReID) + 证据链(OA-SD→OA-RD)
experiments/decisions.md:3113:**上下文**: exp199 (OA-RD+SupCon) ep60=-1.5/-3.4 vs exp187，exp200 (OA-RD+CE) ep60=-1.1/-3.4 vs exp191。OA-RD (relational distillation) 也是负结果。
experiments/decisions.md:3115:**核心发现**: 任何形式的 EMA self-distillation (OA-SD/OA-RD) 都与 SupCon 不兼容。
experiments/decisions.md:3116:- OA-SD (feature-level): exp188/196 失败
experiments/decisions.md:3117:- OA-RD (relation-level): exp199 失败
experiments/decisions.md:3137:**上下文**: exp197-201 连续 5 个负结果。所有在 exp187 (64.9/76.6) 基础上的改进尝试都失败。
experiments/decisions.md:3140:- exp197 (STM + SupCon): -0.8/-0.6 — token mixup 只加速不改善
experiments/decisions.md:3141:- exp198 (STM + OA-SD): ±0 — 同上
experiments/decisions.md:3142:- exp199 (OA-RD + SupCon): -1.5/-2.1 — relational distillation 也与 SupCon 冲突
experiments/decisions.md:3143:- exp200 (OA-RD + CE): -0.3/-1.5 — OA-RD 不如 OA-SD
experiments/decisions.md:3144:- exp201 (global SupCon): ~-1.5/-3.6 — global SupCon 压缩特征空间
experiments/decisions.md:3147:**结论**: exp187 的配置已接近 Swin-Tiny 的 performance ceiling (~65% mAP)。
experiments/decisions.md:3173:| 当前 | Small GCN+PAA+CE+OA-SD | 70.5% |
experiments/decisions.md:3175:| +2 | **Swin-Base** (exp207 进行中) | 74-75% |
experiments/decisions.md:3180:1. exp207 Base 跑完后确认 Base 增益
experiments/decisions.md:3185:- exp208 (0.5x global loss) = NO-OP（GCN list-loss 已隐含 0.5x），取消
experiments/decisions.md:3186:- exp209 (STD-PR+CE+OA-SD) ep30=56.0/69.3，落后 5%，终止
experiments/decisions.md:3188:- OA-SD teacher Critical bug 修复并部署
experiments/decisions.md:3190:### [2026-04-01 10:30] 决策 #X — MaxSim + PKC + Fixed OA-SD 路线
experiments/decisions.md:3192:**上下文**: MaxSim hybrid 在 exp206 checkpoint 上无需重训给 +1.8% mAP (70.3→72.1)。OA-SD teacher bug 已修复。PKC (Per-Keypoint Contrastive) 开始测试。
experiments/decisions.md:3198:| 已确认 | Small GCN+PAA+CE+OA-SD + maxsim_hybrid | **72.1%** |
experiments/decisions.md:3199:| exp210 | + PKC (进行中) | 73-74% |
experiments/decisions.md:3200:| exp207 | Swin-Base 3-view (进行中) | 74-76% |
experiments/decisions.md:3211:| exp210 | PKC weight=0.5 (detached GCN) | 灾难 (3.6%) |
experiments/decisions.md:3212:| exp210b | PKC weight=0.05 (detached GCN) | 无效 (= baseline) |
experiments/decisions.md:3213:| exp211 | MST weight=0.5 (detached GCN) | 无效 (= baseline, 所有 loss 完全一致) |
experiments/decisions.md:3214:| exp213 | PKC+MST 组合 (detached) | 灾难 (40.6%) |
experiments/decisions.md:3215:| exp215 | BA-PKC weight=0.1 (NON-detached backbone) | 灾难 (0.5%) |
experiments/decisions.md:3224:3. **当前最佳: 72.4/83.1 (exp210b + maxsim)**
experiments/decisions.md:3231:4. 回到 STD-PR+SupCon 路线（已知 67.9+maxsim ≈ 69.7，不如 GCN+OA-SD）
experiments/decisions.md:3240:| exp210 | PKC w=0.5 on detached GCN | Yes | 灾难 3.6% |
experiments/decisions.md:3241:| exp210b | PKC w=0.05 on detached GCN | Yes | 无效 (=baseline) |
experiments/decisions.md:3242:| exp211 | MST w=0.5 on detached GCN | Yes | 无效 (所有 loss 完全一致) |
experiments/decisions.md:3243:| exp213 | PKC+MST combo | Yes | 灾难 40.6% |
experiments/decisions.md:3244:| exp215 | BA-PKC w=0.1 non-detached | No | 灾难 0.5% |
experiments/decisions.md:3245:| exp212 | LR=0.0008 | — | 灾难 0.8% |
experiments/decisions.md:3246:| exp217 | OERL w=1.0 non-detached cosine | No | `62.2/75.2`，相对 `exp191 63.2/75.4` 为 `-1.0/-0.2` |
experiments/decisions.md:3251:3. non-detached cosine alignment: 与 OA-SD 竞争 → 负面
experiments/decisions.md:3261:**PACI (exp218/219) 结果:**
experiments/decisions.md:3262:- PACI + OA-SD (exp218): `61.9 / 74.2` (vs `exp191 63.2 / 75.4` = **-1.3 / -1.2**)
experiments/decisions.md:3263:- PACI-only (exp219): 已从远程补回 `train_log`，当前可直接复核到 `ep10=37.7/50.5`、`ep20=47.5/60.4`、`ep30=51.9/64.9`；但尚无 final，因此它仍只能作为 early stop-loss 证据，不能写成正式最终结果
experiments/decisions.md:3270:| OA-SD-only | **63.2** | 64.2 |
experiments/decisions.md:3271:| OERL+OA-SD | 62.2 | 64.3 |
experiments/decisions.md:3272:| PACI+OA-SD | 61.9 | 64.1 |
experiments/decisions.md:3274:这一步更准确的结论不是 “OA-SD 已达 64.4 ceiling”，而是：
experiments/decisions.md:3275:1. 在 `OA-SD / OERL / PACI` 这三条已完成 Tiny 线内部，`maxsim_hybrid` 都落在 `64.1~64.3`
experiments/decisions.md:3276:2. `MaxSim` 对 `OA-SD` 本身仍是正向的（`63.2 -> 64.2`），只是 `OERL/PACI` 并没有把这个 test-time 上限继续抬高
experiments/decisions.md:3283:2. non-detached losses 与 CE/OA-SD 冲突 → 灾难
experiments/decisions.md:3285:4. 后续 `exp220` 已把 Tiny `maxsim_hybrid` 推到 `64.6`，因此这里原先的 `~64.4` / `~64.2` ceiling 表述应视为阶段性误判
experiments/decisions.md:3298:| OA-SD-only | 63.2/75.4 | 64.2/77.1 |
experiments/decisions.md:3299:| GSPB+OA-SD (scale=0.05) | 62.9/74.3 | **64.6/76.0** |
experiments/decisions.md:3300:| PADPQ K=4+OA-SD | **63.7/74.5** | 63.9/74.8 |
experiments/decisions.md:3301:| PADPQ K=8+OA-SD | 进行中 | 进行中 |
experiments/decisions.md:3304:1. GSPB: 早期加速 +5.8% at ep10，按当前测试记录 `maxsim_hybrid` 相对 OA-SD 为 `+0.4`，是目前 Tiny 线上最高的 `maxsim` mAP
experiments/decisions.md:3317:**上下文**: exp229-232 全面测试了 BT-PKD (Backbone-Through Per-Keypoint Distillation):
experiments/decisions.md:3340:### [2026-04-04 15:40] 决策: exp242 PPA+GCN Small 灾难性失败
experiments/decisions.md:3347:2. 对比: PPA on Small (exp240) 也是中性 (70.7/81.1 vs 70.6/82.6 = +0.1/-1.5)
experiments/decisions.md:3351:**选择**: 放弃 PPA 作为主线方向。转向 LGPA (CLIP-based part assignment)
experiments/decisions.md:3353:**理由**: LGPA 使用 CLIP frozen text prototypes 作为语义锚, cross-attention 机制与 PPA 不同, 可能在梯度控制上更好。
experiments/decisions.md:3355:### [2026-04-04 15:40] 决策: 启动 exp243 LGPA
experiments/decisions.md:3358:**选择**: LGPA = CLIP text embeddings + cross-attention + pose masks
experiments/decisions.md:3365:### [2026-04-04 21:10] exp243 LGPA 结果分析 (GPU crash at ep88)
experiments/decisions.md:3375:- LGPA with detached features (仅用 CLIP 做更好的 part pooling, 不传梯度)
experiments/decisions.md:3379:### [2026-04-05 04:10] exp244 LGPA-Detach — 突破性结果! ⭐⭐⭐
experiments/decisions.md:3381:**结果**: 65.3/75.7 (+2.1/+0.3 vs exp191 GCN+OA-SD)
experiments/decisions.md:3386:1. LGPA-D 全程 mAP delta 均为正 (ep10~ep120), 前所未有
experiments/decisions.md:3387:2. detach 完全消除了 non-detach 的后期干扰 (exp243 -1.1 → exp244 +2.1)
experiments/decisions.md:3389:4. LGPA-D 无 OA-SD (63.6) ≈ GCN + OA-SD (63.2): CLIP 价值 ≈ OA-SD
experiments/decisions.md:3393:- 消融故事清晰: non-detach (exp243) vs detach (exp244) 证明 detach 必要性
experiments/decisions.md:3395:- 与 OA-SD 正交: 可叠加
experiments/decisions.md:3408:- 与 LGPA-D 完美配合: LGPA-D 提供 per-part features, PCFD 做 pair-level 精细比较
experiments/decisions.md:3416:3. PCFD 与现有 LGPA-D pipeline 正交叠加
experiments/decisions.md:3418:### [2026-04-08 16:45] 决策 — exp249 完成后下一步
experiments/decisions.md:3420:**上下文**: exp249 (Small LGPA-D+GCN) 完成: 71.9/81.8 equal_concat, 73.3/83.2 MaxSim。
experiments/decisions.md:3429:1. Tiny 消融数据 (exp244, exp246b) 已经足够完整
experiments/decisions.md:3440:- LGPA-D: +2.1 mAP (Tiny), +0.3 mAP (Small) vs GCN baseline
experiments/decisions.md:3448:**上下文**: LGPA-D novelty 4.5/10, 需要更深层创新达到 CCF-B 级别。
experiments/decisions.md:3449:已完成 VCSR (exp247, 失败) 和 PCFD (exp248, 失败) 两个创新尝试。
experiments/decisions.md:3461:1. 只有 backbone 修改有效 (PSG, OA-SD)
experiments/decisions.md:3466:A. 短期: 完成 exp249, 快速测试 POT (test-time, 无训练需求)
experiments/decisions.md:3467:B. 论文策略: 以 LGPA-D (CLIP 语义 part assignment) 为核心贡献, 配合完整 pipeline 消融
experiments/decisions.md:3472:2. LGPA-D 虽然 single novelty 4.5/10, 但与 PSG+OA-SD+MaxSim 组成完整 framework novelty 更高
experiments/decisions.md:3473:3. exp249 (LGPA-D+GCN on Small) 有潜力达到 73-74% → 与 SOTA 竞争力足够
experiments/decisions.md:3475:### [2026-04-15 18:30] 决策 — PRCV 方向重审，停止把 LGPA/MaxSim 当主故事
experiments/decisions.md:3480:- `exp257-259` 已基本说明当前 `exp255` recipe 空间耗尽
experiments/decisions.md:3481:- `LGPA-D + GCN + OA-SD + MaxSim + flip` 虽然结果强，但主问题定义仍偏弱
experiments/decisions.md:3485:  A. 继续沿 `LGPA-D + MaxSim/POT + test-time` 故事收论文
experiments/decisions.md:3492:2. `LGPA-D` 更像 detached semantic part asset，不足以单独撑起新的问题定义
experiments/decisions.md:3509:3. 若 Tiny 为正，再上 `exp255` Small scaffold
experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/decisions.md:3520:### [2026-04-15 19:20] 决策 — 用户确认：PRCV 先收敛到 PSG 主线，旧实验允许重跑
experiments/decisions.md:3529:  B. 回到 `PSG` 主线，把 `multi-stage PSG` 重新做成干净可辩护的扩展版本
experiments/decisions.md:3534:1. `PSG` 本体已有最稳的证据链：`exp007` 单次正向，且 3-seed mean 明确成立
experiments/decisions.md:3535:2. 当前最强系统 `exp255` 使用的就是 `2-stage PSG`
experiments/decisions.md:3536:3. `exp255 vs exp255b` 给出最强信息：在 `GCN512` 高容量结构分支下，`2-stage PSG` 带来 `+1.7 / +1.4`
experiments/decisions.md:3537:4. 虽然 `exp009 / exp251 / exp253` 不支持“multi-stage 普遍更强”，但这恰好说明需要**重跑干净消融**，而不是放弃 PSG 主线
experiments/decisions.md:3538:5. 对 PRCV 来说，“PSG 为主创新，2-stage PSG 为 scalable extension” 比临时强切新问题定义更稳
experiments/decisions.md:3541:1. `PSG` = 主创新
experiments/decisions.md:3542:2. `2-stage PSG` = 当前最终版本 / scalable extension
experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/decisions.md:3546:1. 设计并重跑基础 PSG stage 消融：
experiments/decisions.md:3547:   - no PSG
experiments/decisions.md:3548:   - 1-stage PSG
experiments/decisions.md:3549:   - 2-stage PSG
experiments/decisions.md:3550:   - 3-stage PSG
experiments/decisions.md:3558:**执行结果**: 待后续新一轮 PSG 消融实验补充。
experiments/decisions.md:3586:**上下文**: 用户审查发现 `processor/processor.py::_extract_feat_flip` 在 `equal_concat` 模式下用 whole-vector 平均,让 evaluator 单次 L2-normalize,破坏了 `equal_concat` 在 model 端每块 L2-normed concat 的"均等贡献" cosine 语义。其他 pose 模块(如 OA-SD 打破训练端 flip 对称 + GCN per_part 的 L/R 非完全对齐)导致每块的 flip-后 norm `r_k` 不同,whole-vector 重归一化 `sqrt(Σ r_k²)` 给各块的有效权重被扭曲。
experiments/decisions.md:3679:1. 确保 pose_data 在 lab4090 与 srvB(训练集 ground truth 源)完全一致,避免未来 Full Scaffold (OA-SD/PLBOA) 因 visibility 用 score proxy 出现微小精度漂移
experiments/decisions.md:3812:**立即决策**: srvC 启动 **Phase 3-C exp286/287** (LGPA-only Tiny 2 runs,phase3_design.md L111-134 已规划),填补 srvC 空闲。
experiments/decisions.md:3815:- exp286 (LGPA-only + 1-stg PSG + Tiny, PID 59845) @ 23:32 CST,config load + dataset load OK
experiments/decisions.md:3816:- daemon 59846 挂 exp286 → exp287 (2-stg PSG) auto-chain
experiments/decisions.md:3820:**Phase 3-C 科学价值**: 回答 phase3_design.md 核心问题 3 — "2-stage PSG 收益是偏 structural branch (GCN) 还是 semantic branch (LGPA) 也吃"。和 Phase 3-B (GCN on) 对照,Table 4 (optional) 的 4 行素材。
experiments/decisions.md:3880:### [2026-04-21 03:47] 事件 — exp277 Small 3-stage PSG 训练塌缩 (negative result,不重训)
experiments/decisions.md:3883:- exp277 Small + PSG 3-stage `[-3,-2,-1]` 自 01:42 CST 启动
experiments/decisions.md:3884:- e10 eval **0.3 / 0.3** (接近 random), e120 FINAL **49.0 / 57.7** (远低 exp274 no-PSG 68.1/76.8, Δ=-19.1)
experiments/decisions.md:3889:- triplet loss (`tri_global`) 仍在下降 (7→3), 说明 **仅 feature space 在学**, **BNNeck/classifier 梯度通路被 3-stage PSG gate 截断**
experiments/decisions.md:3898:- negative result 本身有价值:支持 "default 选 2-stage" 论述,展示 "PSG stage × backbone 容量" 交互效应
experiments/decisions.md:3910:1. PSG 本体在 Tiny 上 monotonic 增益至 2-stage, 在 Small 上 1-stage 已达 peak
experiments/decisions.md:3917:- 3:47 CST exp277 FINAL 49.0/57.7 归因为 "Small 3-stage PSG 系统塌缩"
experiments/decisions.md:3939:- exp280 Swin-Tiny + GCN512 + PSG `[-1]` FINAL @ 08:07 CST srvB
experiments/decisions.md:3941:- Phase 3-B Tiny 2×2 最后一格, 补齐 GCN{256,512} × PSG{1,2} stage 矩阵
experiments/decisions.md:3946:| PSG `[-1]` | 65.7/76.7 (exp278) | **65.7/76.2** (exp280, **weakest R1**) |
experiments/decisions.md:3947:| PSG `[-2,-1]` | 65.7/76.9 (exp279) | **65.9/77.4** (exp261) |
experiments/decisions.md:3953:- **大 GCN 容量必须配 2-stg PSG 才完整 exploit**, 1-stg gate 浪费 GCN 容量
experiments/decisions.md:4001:- srvB (5+4 batches): exp261 Tiny OD, exp267 Tiny Market (retry 后成功), exp278/279/280 Phase 3-B Tiny, exp271/272/273 Phase 3-A pure PSG Tiny
experiments/decisions.md:4003:- lab4090 (4+5 batches): exp282/283/284/285b Phase 3-B Small, exp275/276/277/277b Phase 3-A pure PSG Small (exp274 POSE_ENABLED False crash)
experiments/decisions.md:4016:**跨 eval 验证**: Phase 3-A pure PSG 所有 Global+flip 数字和训练 FINAL eq+flip 精确对齐 (差 ≤ 0.1 R1), **exp277 seed 42 塌缩 49.0/57.6 跨 eval 复现确认偶发 seed 训练塌缩**。
experiments/decisions.md:4023:- srvC local: exp264 Tiny OP, exp265 Small OP, exp286/287 Phase 3-C Tiny LGPA-only, exp288/289 Phase 3-C Small LGPA-only
experiments/decisions.md:4039:- exp288 Swin-Small + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + PSG `[-1]` (**无 GCN**) FINAL @ 12:51 CST srvC
experiments/decisions.md:4043:- exp285b Full Scaffold (GCN512 + LGPA + 2-stg PSG): **73.8 / 83.8 / 90.7 / 92.7** → Δ 0/0/-0.2/-0.7
experiments/decisions.md:4045:- exp284 Full GCN512+1stg: 73.4/82.9 → Δ +0.4/+0.9 (LGPA-only 反超!)
experiments/decisions.md:4048:1. **GCN 对 Swin-Small OD 零或负贡献** — LGPA 单独即满配性能
experiments/decisions.md:4049:2. 和 Tiny 结论 (exp286 LGPA-only 66.0 ≈ exp261 Full 65.9) **跨 backbone 一致**
experiments/decisions.md:4050:3. **Phase 3-B GCN cap × PSG 矩阵 方差 ≤ 0.4 mAP 本质是因为 GCN 不起作用**, 方差来自 PSG/LGPA 随机性
experiments/decisions.md:4054:- **main contribution**: PSG + LGPA (semantic branch) — GCN 不抢主位
experiments/decisions.md:4059:- exp289 LGPA-only 2-stg 自动启动 (srvC PID 86783), FINAL ~16:50 对照 PSG stage in LGPA-only 配置
experiments/decisions.md:4114:- exp293 config 确认 PLBOA=True 激活, OA-SD WARNING 消失 (teacher/student 现有差异)
experiments/decisions.md:4125:**exp293 Base Market + PLBOA** (lab4090, OA-SD 激活):
experiments/decisions.md:4132:  - 假设验证: 第 3 情景 "两力相抵, 微 net 负" (OA-SD 收益 < 分布偏差)
experiments/decisions.md:4144:- OD (exp285b etc): PLBOA True, OA-SD 蒸馏有效, +性能
experiments/decisions.md:4145:- OP (exp265 etc): PLBOA True, OA-SD 蒸馏有效, +性能
experiments/decisions.md:4163:- Swin-Small LGPA-only + 2-stage PSG (无 GCN) @ 2026-04-23 05:39:56 CST srvC
experiments/decisions.md:4165:- vs exp288 LGPA-only 1-stg 73.8/83.8/90.5/92.0: Δ 0 / -0.5 / 0 / +0.4
experiments/decisions.md:4171:| LGPA-only | exp288 73.8/83.8 | exp289 73.8/**83.3** |
experiments/decisions.md:4184:- Phase 3-A (pure PSG stage): 8/8 FINAL ✓
experiments/decisions.md:4185:- Phase 3-B (GCN cap × PSG): 6/6 FINAL ✓
experiments/decisions.md:4186:- Phase 3-C (LGPA-only × PSG): **4/4 FINAL ✓** (刚刚 exp289 闭合!)
experiments/decisions.md:4238:- 说明 PSG/LGPA gate 在现有训练中已 implicitly 学会部分 disambiguation, 显式 target 换 scene 无额外增益
experiments/decisions.md:4367:3. **LGPA 已捕获足够 pose 结构信息**, GCN branch 冗余
experiments/decisions.md:4413:**上下文**：exp317（Tiny，LGPA_ASSIGN_WEIGHT 0.25）是 Tiny sweep 中唯一 MaxSim 超 baseline 的点（+0.2 mAP），需在 Small 验证。
experiments/decisions.md:4420:### [2026-04-28] 决策 #exp320 LGPA_DETACH=False -6.4 mAP catastrophic → detach 是必要设计（强 negative 消融）
experiments/decisions.md:4422:**上下文**：SOTA push 探索——让 LGPA aux loss 反传 backbone（default DETACH=True），测是否能让 LGPA shape backbone features。
experiments/decisions.md:4425:**理由**：DETACH=False → catastrophic underfit（e10 46% near-random，e80 plateau 68.3）。LGPA 须 detach，作为 frozen pose-spatial-gated features 上的 downstream attention head。
experiments/decisions.md:4426:**执行结果**：强 negative，写入论文消融（"LGPA must be detached; allowing backprop causes -6.4 mAP severe underfitting"）。POSE_LGPA_DETACH=True 保持。
experiments/decisions.md:4540:**资产**: 现成 Market-trained ckpt `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`(Swin-Base+PSG+LGPA+GCN512), 现成跨域 eval `test_on_occluded_reid.py`(Market→Occluded-ReID 86.0/88.5 已存), 两域数据+pose 齐。脚本 `scripts/uce_calib_probe.py`, 结果 `log/uce_calib_probe.json`。
experiments/decisions.md:4551:**资产**: Market-trained ckpt `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`(Swin-Base+PSG+LGPA+GCN512, Occ-ReID baseline 88.0 mAP MaxSim+flip), 数据 Occluded-ReID(1000q/1000g+pose), env solider-reid(torch1.13+mmcv)。脚本 `scripts/vcnorm_probe.py`(主)+`scripts/vcnorm_probe_control.py`(对照), 结果 `experiments/vcnorm_probe/*.json` + README。
experiments/decisions.md:4552:**方法(无训练)**: per-part token = SkeletonGCNHead 在 PSG-modulated Stage-3 图上按 17 COCO kp bilinear 采样的 token(dim 1024, pre/post-GCN)。按 pose 置信度 score 把每 kp 的 token 分 high-vis(≥0.7) vs low-vis/遮挡(≤0.2), 算逐通道对角高斯对称 KL + 2-Wasserstein + Fisher-LDA held-out AUC。三对照排伪影: %border 坐标、KL(hi,**rand** 体内随机采样)、KL(hi,**lo_onbody** 剔边界坐标)。
experiments/decisions.md:4597:### [2026-06-18 ~19:1x] 决策: CLIP/LGPA-D 复现弧线 — 增益是 pose 不是 CLIP 文本(拆解证据)
experiments/decisions.md:4598:**上下文**: post-SMPL,用户想把 LGPA-D 包装成新 CLIP 模块创新。先复现 ViT-baseline+LGPA-D 确认 CLIP +X。ViT 上 equalcat < global(负)。用户坚持"是你的复现 bug 非 backbone"。
experiments/decisions.md:4599:**调查**: 派 10 个 Codex 并行深挖(用户 rate-limit 不让开 300 Claude 子agent,Codex token 无限)→ 挖出**热图 bug**:exp335 喂 target-only 热图(`heatmaps[:,0]`+POSE_USE_TARGET_HEATMAP=True)→ LGPA assign KL 坍缩=0 → 部位退化。修(scene-merged)→ assign 0→7.02≈原版。但 ViT 仍只 +0.5、不翻盘。深挖发现 **LGPA-D 从未单独跑过**(exp244/245g 全是 PSG+LGPA+OASD+aug+384+Swin 全系统)。
experiments/decisions.md:4601:- exp336(Swin 纯 LGPA-D,关 PSG/OASD/aug,`POSE_PSG_STAGES=[]`):equalcat 59.6 vs global 58.5 = **+1.1**(e60 时 +1.7)。→ CLIP 模块 standalone 在 Swin 上涨;**ViT 失败=ViT-specific**(单尺度池不出强部位)。
experiments/decisions.md:4602:- exp337(同配置 + `POSE_LGPA_NO_POSE=True`,LGPA 收 heatmaps=None,纯 CLIP-text 部位,assign=0):equalcat 58.7 vs global 58.8 = **≈0**。→ **那 +1.1 全来自 pose 注入,不是 CLIP 文本语义**。CLIP 文本"head/torso/legs"是 query 壳、冗余于 global;pose-bias 引导注意力到身体区才让部位有判别力。
experiments/decisions.md:4610:**理由**: detach 保证 global==无-LGPA baseline,within-ckpt equalcat-vs-global 干净;no-pose ablation 单变量隔离 pose。
experiments/decisions.md:4640:- 姿态**外挂** (exp342 detached LGPA): 60.0 +0.2 marginal
experiments/decisions.md:4641:- 姿态 **un-detach LGPA** + CLIP (exp342b): 60.7 +0.9 ← 一度以为是突破
experiments/decisions.md:4642:- **un-detach LGPA 单独无 CLIP (exp353): 60.5 = +2.9, 已 > CLIP 单独 59.8!**
experiments/decisions.md:4647:2. exp342b 的 +0.9 **大部分是 pose(un-detach LGPA), CLIP 只 +0.2**(exp353 隔离戳穿)。
experiments/decisions.md:4648:3. **CLIP(+2.2)与 un-detach LGPA(+2.9)冗余**: 合 +3.1 << 5.1, 都塑造 backbone 学 ID, 重叠。
experiments/decisions.md:4650:5. un-detach 在简单设置(纯 LGPA)涨, 但破坏全系统(exp349b 65.7<<73.2)。
experiments/decisions.md:4652:**决策**: pose+CLIP 不存在有意义的融合增益(冗余, 非互补)。CLIP-ReID(+2.2)和 pose(LGPA)各自是干净贡献, 但二选一即可, 叠加无收益。这与 [[fm-import]]/[[occluded-reid-four-classes]] 一脉: 强判别 backbone 上"互补"信号总冗余。诚实负结果 = 这一夜的真交付。
experiments/decisions.md:4691:**上下文**: 三个独立 codex(终身 d3 / 开集 d9 / 长尾 d10)收敛到同一 re-framing: ReID 失败由 GALLERY 组成(规模/膨胀/分布)驱动, 非只看 query/模型。用户要求零训练验证, ★铁律=每个 per-query 相关都控 trivial 代理(吸取 HUBNESS §7.6 教训: 上个诊断被漏控 #false-in-topk 证伪)。脚本 `cvpb_gallery_killswitch.py`, 复用 hubness 缓存特征, Market exp260b + Occluded-Duke exp255。双审(Claude broad 5 blocking 全修 + Codex)。
experiments/decisions.md:4731:**出路(用户拍板)**: ①收 LM-ReID(#99, 6.5 现有最强 B 类, 文档已全)/PSG 已成型线(最务实) ②换真实新数据规模(下新大规模数据集) ③换问题领域。

exec
/bin/zsh -lc 'find experiments -maxdepth 2 -name monitor.md | sort | tail -n 20' in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc 'find experiments -maxdepth 2 -name design.md | sort | tail -n 20' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp351_undetach_deocc/design.md
experiments/exp352_undetach_morepartw/design.md
experiments/exp353_undetach_noclip/design.md
experiments/exp354_pc_sor/design.md
experiments/exp355_pgpd/design.md
experiments/exp355r_pgpd_random/design.md
experiments/exp356_pcmsc/design.md
experiments/exp356r_pcmsc_random/design.md
experiments/exp357_pose_shuffle_ks/design.md
experiments/exp358_pose_channel_shuffle/design.md
experiments/exp359_lm_reid/design.md
experiments/exp360_intruder/design.md
experiments/exp361_psc_jepa/design.md
experiments/exp363_ag_foundation/design.md
experiments/exp364_dg_foundation/design.md
experiments/exp365_wildlife/design.md
experiments/exp366_active_evidence/design.md
experiments/exp367_single_support/design.md
experiments/exp368_camtransport/design.md
experiments/smpl_anchor_vireid/design.md

 succeeded in 0ms:
experiments/exp324h/monitor.md
experiments/exp325/monitor.md
experiments/exp326_dift/monitor.md
experiments/exp327_dinov3/monitor.md
experiments/exp328_vcnorm/monitor.md
experiments/exp333_smpl3d/monitor.md
experiments/exp334_smpl_geom/monitor.md
experiments/exp335_vit_lgpa/monitor.md
experiments/exp336_swin_lgpa_nopsg/monitor.md
experiments/exp359_lm_reid/monitor.md
experiments/exp360_intruder/monitor.md
experiments/exp361_psc_jepa/monitor.md
experiments/exp362_genoccl/monitor.md
experiments/exp363_ag_foundation/monitor.md
experiments/exp364_dg_foundation/monitor.md
experiments/exp365_wildlife/monitor.md
experiments/exp366_active_evidence/monitor.md
experiments/exp367_single_support/monitor.md
experiments/exp368_camtransport/monitor.md
experiments/smpl_anchor_vireid/monitor.md

codex
我看到仓库里其实已经有一批“越出 SOLIDER 死区”的草案/试验：LM-ReID、intruder、JEPA、foundation、wildlife、active-evidence、single-support、camera-transport 等。我会先抽读这些，避免把已判死或你明确禁掉的方向重新包装。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp360_intruder/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp360 — Intruder Identity Suppression（范式转向第一个 build）

## 选定经过（2026-06-26 深夜，用户授权范式转向后）

3 路 codex gap analysis → 2 候选终局对比（codex `decision_tscd_vs_intruder.md`）：

| 候选 | codex 综合 | 否/选理由 |
|---|---|---|
| T-SCD（tracklet support 蒸馏） | 5.0/10 ❌ | 撞项目自己的 `fgeu_realizability_result.md`（posetrack tracklet 每条≤2帧、同机位冗余，只恢复 oracle 16.3% < 40% 门槛）+ MVI²P/UMTS/VKD 先例 |
| **Intruder Identity Suppression** | **7.0/10 ✅** | 避开 completion/visibility/occluder-gate 死区；novelty 窄缝（KPR/QPM/DPEFormer/OGFR 相邻但无"显式 donor-ID 泄漏可测+训练对抗+测试单图"直接同构） |

## 动机 / 问题重定义（核心）

遮挡 ReID 的根症结，**不是**"target 信息缺失要补全"（completion / support-complete 这条线已反复证负：exp109 墙、fgeu 16.3%、各种 feature completion 小残差），**而是**：

> **遮挡物（尤其另一个行人 donor）把 donor 的身份信息泄漏进了 target 的 embedding，污染了检索。要做的是 source separation——把 donor-ID 从 target embedding 里分离/压制掉。**

这是"换问题定义"的范式动作，不是在强 backbone 上加模块。

## 核心假设（可证伪）

H1: 行人遮挡的 target crop，其 embedding 里**可测地**含 donor-ID 信息（donor-ID probe 显著 > 随机）。
H2: donor-ID 泄漏量与 target 检索错误**正相关**（cos(f_mix,f_donor) − cos(f_clean,f_donor) 越大，AP drop 越大）。
H3: 训练时显式压制 donor-ID（对抗）能降低泄漏，**且**降低泄漏带来真实人遮挡 split 的 ReID 提升（不是只压表征不涨点）。

## Pipeline（codex 终选）

1. 合成样本：target 图 + donor 行人 crop/mask（donor 有已知 PID），遮挡比例分档。
2. 三路 forward：clean target `x_t` / intruded `x_t+d` / donor alone `x_d`。
3. 主 ReID loss：`x_t`、`x_t+d` 都用 **target** PID 做 CE/triplet。
4. clean-occluded consistency：`f(x_t+d)` → `stopgrad(f(x_t))`，保护 target identity。
5. donor suppression：在 `f(x_t+d)`（或 residual `f(x_t+d)−f(x_t)`）接 donor-ID classifier + **GRL**；加 margin/contrastive 让 mixed feature 不靠近 donor feature。
6. 测试：纯单图，去掉 donor head，无外部信息。

## Build 阶段（deep work，非 cheap-收敛）

- **阶段 0（地基机制验证，先做）**：frozen strong baseline 上验证 H1+H2——合成 target+donor，测 donor-ID probe acc vs 随机、测泄漏量 vs AP drop 相关性。**这是 build 地基不是 cheap 逃避**：H1/H2 不成立则对抗压制无的放矢，需调整 donor 合成或换机制变体。
- **阶段 1**：小规模训练验证 H3（donor suppression head + GRL + consistency），20-40 epoch 看 donor probe 是否降 + 真实人遮挡 split 是否涨。
- **阶段 2（成立则）**：全量训练 + 多数据集（Occluded-Duke/Occluded-ReID/Market/MSMT）+ 消融（GRL on/off、consistency on/off、donor margin）+ 迭代。deep work，允许多日训练。

## Kill-switch / 风险（codex）

- **头号风险**：donor-ID probe 证明泄漏存在，但压掉后 ReID 不涨 = 退化成 **target ambiguity 墙**（项目 memory `post-prcv-directions-exhausted` 记该方向红蓝队判死过；本方向靠"训练端 source separation + 可测泄漏"区别于 KPR test-time prompt，窄缝）。
- **硬阈值**：donor 泄漏下降 ≥30% 但真实人遮挡 mAP 没 ≥+0.5 → kill（说明压泄漏不是有效 ReID 机制）。
- **对照**：matched synthetic-only baseline（同样合成数据但不压 donor）、Market/物体遮挡 split（不该涨，涨了说明是通用增强非 source-separation）。

## 与死区的关系（诚实标注）

- target ambiguity（memory 判死）相邻 → 靠"训练端可测 donor 泄漏 + 对抗压制"区别，阶段 0 的 H1/H2 就是验证这个区别真实。
- 不碰：SMPL 几何 / FM-import / test-time prompt(KPR) / completion 残差。

关联：`experiments/paradigm_shift/`（README + 3 路 gap + 决策）、memory [[post-prcv-directions-exhausted]] [[exp109-headroom-is-a-wall]] [[fm-import-occluded-reid-closed]]。

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp363_ag_foundation/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp363 — Aerial-Ground / RGB-IR Video Foundation Adaptation（跳出盒子第一个 build）

## 选定经过（2026-06-27，用户点醒"别自我设限 occluded ReID+SOLIDER"后）

- 换量级 deep work 在 occluded ReID 内部多 build 全证伪接近墙（[[paradigm-shift-occluded-reid-wall]]）。用户点醒：没限定我在 occluded ReID+SOLIDER。
- codex 全 ReID 范式级 gap analysis（`paradigm_shift/codex_full_reid_gap.md`）：判"别再押 occluded ReID/SOLIDER/pose-part"，选 **Aerial-Ground / RGB-IR Video Foundation Adaptation**（换问题+换 backbone+换监督）。
- codex AG 核查（`paradigm_shift/codex_ag_verify.md`）：避 CARGO 死区确认，数据可及，cheap kill-switch 清晰，**6.5-7/10**，值 1-2 天 kill-switch 不直接开大工程。

## 真正跳出盒子（vs 之前自我设限）
- **换问题**：occluded ReID → aerial-ground / video / RGB-IR（AG-VPReID.VIR）
- **换 backbone**：SOLIDER → DINOv3 ViT-B/16（foundation，非 SOLIDER）
- **换监督**：identity CE/triplet → 跨视角/尺度/模态/时间 multi-axis consistency + foundation-teacher anchoring

## 核心 idea（novelty 窄缝，codex）
> 极端视角/模态/时间证据缺失下，低秩适配（LoRA）只学**跨帧/跨模态可验证的 residual identity evidence**，同时用**冻结 foundation teacher 的关系结构锚定 identity prior**，防 direct fine-tune 把 foundation prior 覆盖成数据集偏置。

**避死区**（codex + 项目历史）：不做 uncertainty containment（CARGO σ 前提错已死）/ 不做 avg-vs-MaxSim late-interaction（已死）/ 不做 view-aware semantic experts（撞 ViSA/SD-ReID/GSAlign）/ 不做几何 alignment / 不生成 view-specific feature / 不靠 SD / 不把 CLIP prompt 当主贡献。

## 数据（codex 核查 + 下载报告 codex_download_cmds.md）
**AG-VPReID.VIR**：GitHub `agvpreid25/AG-VPReID.VIR`（**只 assets+README，无代码** → dataloader 自己写）+ Google Drive folder `1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7`（gdown --folder）。**目录结构 README 未公开**（下载后 find 看真实结构）。
- **★规模修正（重要）**：train **326 ID / 978 tracklets / 24793 frames**（偏小！之前误写 1837 ID/124855 frames 是全集含 test）。小数据 + frozen foundation + LoRA 适合，但 kill-switch 要注意样本量（hard bucket 可能样本少，统计噪声）。
- test 协议：G→G / A→A / G→A / A→G，各 V2I+I2V；源 = UAV RGB + UAV IR/NIR + CCTV RGB + CCTV IR/NIR + wearable RGB。
（备：AG-VPReID 全集 9.6M frames；CARGO 已踩过；LAGPeR 申请制；G2APS 不可直接释放。）

## 第一步 cheap kill-switch（1-2 天，codex）
1. 下载 AG-VPReID.VIR。foundation = **DINOv2-reg-B 先用**（DINOv3 ViT-B/16 是 gated 需 HF 登录同意 dinov3-license；**DINOv2-reg-B `vit_base_patch14_reg4_dinov2.lvd142m` Apache-2.0 无 gate，timm 1.0.25 可加载，torch 1.13.1 兼容**）。DINOv3 等 HF token 到位再换对比。CLIP-L 只做 baseline（X-TFCLIP 贴脸）。
   - 环境：4090/3090 均 torch 1.13.1 + timm 1.0.25（缺 peft/gdown/transformers，已 pip install）。
2. frozen frame encoder，每 tracklet 采样 N=1/4/8 帧，frame embedding L2 norm。
3. temporal pooling 诊断：mean / quality top-k mean / score max（**不上 attention**）。
4. **4 baseline**：single center/random frame / frozen mean pooling / direct LoRA fine-tune CE+Triplet / direct fine-tune without anchor。
5. **method**：LoRA rank 8/16（qkv/proj/MLP adapter）+ frozen-teacher relational anchoring + tracklet/modality consistency。

**硬判定线（任一不过立即杀，不补 LoRA rank/attention pooling/view gate 小变体）**：
- frozen temporal mean vs single frame，hard bucket ≥ **+5 mAP/R1**（视频证据积累成立）
- oracle/top-k vs mean ≥ **+3 mAP**（选择/校准有空间）
- anchored-LoRA vs direct-LoRA/direct fine-tune ≥ **+2 overall 或 +3 hard bucket**（只赢 frozen 不算）
- 增益必须集中高海拔 A↔G / RGB↔IR / 短低清 tracklet（全桶平均涨=普通 adapter，杀）

## 风险
- 数据存储/IO（视频帧大）；DINOv3 权重下载/许可；又撞 AG-VPReID-Net/TCC-VPReID/X-TFCLIP（leading method 已打 temporal/CLIP）—— 靠 anchored residual-evidence 窄缝区别。
- backbone 训练（LoRA fine-tune）= 启动前 codex 三审 diff[[pre-experiment-review-discipline]]。

关联：`paradigm_shift/codex_full_reid_gap.md` + `codex_ag_verify.md`，memory [[paradigm-shift-occluded-reid-wall]] [[aerial-ground-containment-bet]]（CARGO 死区，本方向避开）。

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp359_lm_reid/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 实验 exp359: LM-ReID (Lattice-Marginalized ReID)

> 后 PRCV「换问题」阶段第一个 GO 的 method 候选（9 个零训练 cheap-kill 全死后）。零训练 kill-switch 已 GO（d8 lattice, agent a68e6），method-design codex 判 **CCF-B 7/10**（训练版超 frozen ensemble → 8/10）。本实验做训练版，证明 lattice-marginalization 是 **method 不是 ensemble trick**。
> 设计来源：`experiments/cargo_cvpb/litreview2/pivot/clean/lattice_method_design.txt`（完整 codex 设计）+ `cvpb_lattice_result.md`（kill-switch 数据）。

## 动机

低分辨率 ReID 的传统视角：LR = 模糊/缺细节，解法 = SR / resolution-invariant feature。**我们重定义**：一部分 LR 失败不是"缺信息"，而是 **采样格点不确定性（sampling-lattice uncertainty）**——同一个 HR 身份在不同合法的 LR 采样格点（sub-pixel phase / bbox alignment / downsample kernel）下，落到不同 embedding 区域，导致 rank-1 身份翻转。

### 零训练 kill-switch 证据（GO）
frozen exp260b Market，K=9 lattice variants ensemble，HR gallery / LR query：

| h | rank-1 flip% | single LR | lat-MaxSim | **LATgain** | **LAT−TTA** |
|---|---|---|---|---|---|
| 16 | 74.9% | 42.65 | 46.87 | **+4.23** | **+3.04** |
| 24 | 31.3% | 69.31 | 72.98 | **+3.67** | **+2.68** |
| 32 | 9.7% | 81.93 | 83.98 | +2.05 | +1.44 |
| 48 | 1.2% | 90.44 | 91.02 | +0.58 | +0.41 |

两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。

### 诚实 caveat（写法要求）
phase-var 作 per-query 失败预测器**不干净**（控 LR-severity 后 partial 塌到 ≈0，与 per-image LR 失真共线）。**GO 靠的是 interventional 结果**（ensemble +4.2 / LAT−TTA +3.0 是直接测量）。故事写成 "lattice sensitivity 是 **mechanism-level nuisance**, 不是 standalone failure predictor"，方法是对所有 severe-LR query 做 marginalization（不是预测哪张失败）。

## 核心假设

训练一个 lattice-marginalized embedding（对 lattice variants 身份稳定）+ 推理 K-marginalization，在 h=16 上比 frozen lattice ensemble **再高 +0.8~2.0 mAP** → 证明它学到了 lattice-invariance（是 method），不是 ensemble trick。

## 技术方案

### 数据流
1. 正常 ReID baseline（Market，exp260b 同配置）。
2. fine-tune：HR train image 在线生成 LR lattice variants `x^l = U(D_l(x))`，l ∈ {sub-pixel phase, bbox jitter, downsample kernel}。每图每 iter 采样 M=2-4 variants，eval 用 K=9。
3. h 混合训练 h ∈ {16,24,32}，severe-biased（但不只训 h=16）。

### LM-ReID loss
```
z^l = norm(fθ(T_l(x)));  p^l = softmax(W z^l);  z^μ = norm(mean_l z^l);  p^μ = mean_l p^l
L_id   = mean_l [ CE(p^l, y) + Triplet(z^l, y) ]
L_marg = -log[ mean_l p^l[y] ] + Triplet(z^μ, y)                 # marginal likelihood（主贡献）
L_cons = mean_l (1 - cos(z^l, sg(z^μ))) + β·mean_l KL(p^l || sg(p^μ))  # consistency to mean
L_adv  = GRL-CE(Dφ(z^l), lattice_label_l)                        # 弱：去掉 embedding 中可预测 lattice label
L = L_id + λ_m·L_marg + λ_c·L_cons + λ_a·L_adv
```
默认 λ_m=1.0, λ_c=0.2, β=0.5, λ_a=0.02–0.05（warmup 后开）。**L_adv 弱辅助非主贡献**（太强会擦身份边缘细节，必须 ablation）。

### 推理 K-marginalization
```
s(q, g) = τ·log[ 1/K Σ_l exp( cos(f(T_l(q)), f(g)) / τ ) ]
```
τ→0 接近 lat-MaxSim（主推，因 lat-MaxSim 46.9 > mean），τ 大接近 mean（消融）。

## 预期结果

**过线（决定 method vs trick）**：
- h=16：训练版 > frozen ensemble **+0.8~2.0 mAP**；> single +5~7；> TTA +2~3.5。
- h=24：稳定收益。
- h=32：允许 marginal 不负。

失败最可能原因：训练版只 ≈ frozen ensemble（没学到额外 lattice-invariance）→ 沦为 test-time ensemble trick，不成方法稿。备选投稿角度：同等 mAP 下 K 从 9 降到 3 或 single inference 保留大部分收益。

## 对照组

- single LR（canonical bicubic，固定一个）。
- 普通 K-TTA（同 K，random crop/flip/color/resize）。
- **frozen lattice ensemble**（零训练 K=9，= kill-switch 的 +4.23，这是训练版必须超过的硬线）。
- （成稿）k-reciprocal / SR-based / VPFA。

消融：marg only / marg+cons / marg+cons+adv；τ sweep；K=1/3/5/9 曲线；phase-only vs +bbox+kernel。

## 协议 / benchmark

- 合成：Market/MSMT，gallery HR，query LR h=16/24/32，canonical LR single baseline，K=9（3×3 phase 主，bbox/kernel ablation，不无限扩 K）。所有 TTA 对照 **K-matched**。
- 标准 CR-ReID（成稿补）：MLR-Market / MLR-CUHK03 / CAVIAR（PS-HRNet 用过）。
- 新指标：PRF@1（phase rank-flip rate）、Flip Entropy、LEG（lattice ensemble gain）、LOTG（lattice-over-TTA gain）、query ΔAP。按 h 分报 + paired bootstrap 95% CI + K=1/3/5/9 曲线 + compute cost。PRF 随 h（74.9%→31%→10%→1.2%）是强故事线。

## 撞车边界（novelty，codex 5 路联网）

- **VPFA**（2510.00936，最近邻）：CR-ReID = feature-space resolution direction，Vector Panning LR→pseudo-HR。hidden variable 是 resolution gap/feature direction，**不是 sampling lattice，不做 lattice marginalization**。必须正面对比。
- LRAR（2207.13037）：resolution-adaptive representation，占"resolution adaptive"词 → 我们 novelty 写 sampling-lattice uncertainty。
- RFD（2109.07871）：multi-res gallery distillation，占"resolution-invariant distillation"，没占 lattice intervention。
- BlurPool（1904.11486）：anti-aliasing/shift-invariance 已老 → **不能说首次发现 aliasing 影响**。**能声称**：首次在 LR person ReID 把采样格点作隐藏变量 + rank-level intervention 证明 + lattice-specific marginalization 解决检索身份翻转。
- FlipReID（2105.05639）：ReID 常用 flip-mean TTA → **必须反复强调非 TTA 换名**（同 K vs-TTA 控制，lattice 多 +3.04）。

## 风险与定位

operating point 低（只在 h≤24 强 +3-4 mAP，h=32 marginal）→ 主动收窄定位 **"severe low-resolution / cross-resolution ReID under sampling-lattice uncertainty"**，不写成通用 ReID 鲁棒性稿。h=32 marginal 反而支持机制（分辨率升高 → lattice uncertainty 消退）。真正风险 = 训练版打不过 frozen ensemble，那就只是 ensemble trick。

## 审查 / 训练协议

1. 本 design.md（已写）。
2. 实现 LM-ReID（插件式，config 开关，lattice aug 复用 `cvpb_lattice_killswitch.py` 的 LR 生成）。
3. Claude 广审（Opus 子代理）→ `claude_review.md`；Codex 审（`codex --search exec`）→ `codex_review.md`。**两层通过才训练**。
4. smoke（几 iter 确认各 loss 分量下降无泄漏）。
5. 训练（lab-3090，Market，h 混合，BS=64 不改，TEST.IMS_PER_BATCH 64，PYTHONUNBUFFERED=1 nohup）。
6. eval（**test.py 不用 train.py**）：LM-ReID(K marg) vs single / TTA / frozen-ensemble，h=16/24/32 分报，LEG/LOTG/PRF。

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp361_psc_jepa/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp361 — PSC-JEPA：Pose-defined Support-Complete JEPA pretraining（范式转向核心 build）

## 选定经过（2026-06-26）

范式转向（用户授权换量级）→ 3 路 gap → Intruder(C-#2) 选中 → **Intruder DEAD**（exp360 阶段0.5 因果证伪：donor-ID 可读但压它不救排序，H2 #false 控制 + rank-r projection 双证）→ codex 裁决转 **B PSC-JEPA(6.5)**。这是真正的 continued-pretrain 换量级，不是 frozen probe。

## 核心 idea

给一张**不完整人体图**（pose-defined 部分遮挡），让 backbone 在 **latent body-part token 空间预测"完整身份 support"**——target 来自 EMA full-view teacher + 高置信 pseudo same-ID support bank。**不补像素、不补语义比例，而是补"身份证据"**。JEPA 路线：预测 embedding target 而非像素，避开 ReID 中重建背景/遮挡物的污染。

贴项目 exp109 oracle 证据：support-complete teacher 有真实 headroom（oracle 61.88/73.26 → 70.40/81.36），PSC-JEPA 把这个 headroom 尝试**蒸进预训练**（不像 exp109/FGEU 是测试端不可得的 oracle）。

## Novelty 边界（codex 标，诚实）

**已被占**：SOLIDER(human SSL+语义可控) / PersonMAE(occlusion masking + pixel+semantic 重建, 报 Occ-Duke) / HAP(part-guided MIM) / PersonViT 2024(MIM+contrastive) / SAIP 2025(cross-scale)。
**没吃掉的窄缝**：① 补身份 support 非像素 ② support bank/pseudo cross-view teacher 非 single-image MAE ③ pretraining 学"缺部位时如何形成可检索身份证据" ④ JEPA latent prediction。
→ **B 类空间只在"latent support completion 讲清 + 赢过 plain continued-pretrain/random-mask/PersonMAE-lite"才成立**。

## Pipeline（codex 设计）

1. **init**：SOLIDER/Swin-Tiny continued-pretrain（保持 fine-tune 兼容）；DINOv2 可选 frozen dense teacher（稳的 latent target，不全量 fine-tune）。
2. **data**：**train split only**（防 query/gallery 泄漏）Market+MSMT17+Occluded-Duke+Occluded-ReID+Occluded-PoseTrack；预缓存 pose/keypoint visibility/body-group mask + pose-defined complementary masks。
3. **pretext**：输入 partial view（保留一部分 body support）；teacher target = `T_full`(EMA full-view body-part latent) + `T_bank`(pseudo same-ID/NN support bank 的 body-part prototype)；student 输出 visible/missing/union tokens。
4. **loss**：`L_part_jepa`(missing token 预测 teacher/support, cos/L2) + `L_union`(union token ≈ full-view identity) + `L_gram`(part-token 关系矩阵对齐) + `L_visible_anchor`(可见 part 不被改坏) + `L_solider/dino_anchor`(可见区蒸原 backbone, 防遗忘) + 可选 `L_cluster_contrast`(高置信 pseudo cluster)。
5. **fine-tune**：continued-pretrain 完 → 标准 ReID fine-tune → 评估。

## 4-slot 排布（codex；对照是 novelty 生命线）

| slot | 任务 | 作用 |
|---|---|---|
| 4090 | **PSC-JEPA 主跑**（SOLIDER/Swin-Tiny multi-dataset 50-100ep）| 主结果 |
| 3090 | 去 support bank，只 same-image full teacher | 对照"是否只是 OA-SD/PCVT 换名" |
| 5060Ti-1 | random mask / PersonMAE-lite latent baseline | 对照"是否只是普通 MIM" |
| 5060Ti-2 | support bank 质量诊断 / DINOv2 frozen teacher variant / Occ-Duke smoke | 诊断 |

## Build 阶段（deep work，v0 5-8 天）

- **阶段 A 骨架**：continued-pretrain loop（SOLIDER init + EMA teacher + partial-view pretext + `L_part_jepa`+`L_union`+`L_visible_anchor`，先不加 support bank）→ smoke 跑通 + 不崩。
- **阶段 B support bank**：加 pseudo same-ID support bank（`T_bank`）+ 质量诊断（pseudo same-ID top-k 精度）+ `L_solider_anchor` 防遗忘。
- **阶段 C 对照 + 全量**：4-slot 并行主跑 + 3 对照 → fine-tune Occ-Duke/Market/MSMT → 判 novelty（主跑赢过对照 + Occ-Duke ≥+0.7）。

## Kill-switch（codex）

- continued-pretrain 后 Occluded-Duke fine-tune **≥+0.7 mAP** 或 heavy-occ 子集明显正向；
- **plain continued-pretrain / random mask / same-image teacher 不能同涨**（否则 novelty 塌，只是"更好的 MIM/continued-pretrain"）；
- pseudo support bank top-k 精度过线（否则噪声主导）。

## 风险

- pseudo support bank 噪声（第一风险）；novelty 拥挤（PersonMAE/HAP/PersonViT，靠对照证差异）；continued-pretrain 不涨（45-55% 成功率, codex）。

## 训练前纪律

continued-pretrain = backbone 训练 → **启动前 codex 审 diff**（[[pre-experiment-review-discipline]]，用户分工：审查交 codex）。

关联：`experiments/paradigm_shift/`、`experiments/exp360_intruder/`（DEAD 因果记录）、memory [[exp109-headroom-is-a-wall]]。

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp364_dg_foundation/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp365_wildlife/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp364 — DG/Lifelong Foundation-Preserving Adaptation（跳出盒子第二个 build）

## 由来（2026-06-27）
- AG（exp363）杀（视频证据积累路死：frozen DINOv2-reg 全 8 protocol mean-single < +5；cheap kill-switch 一天验死）。
- codex 全 ReID gap analysis 排序 #2：DG/Lifelong foundation-preserving adaptation，信心 **6.5**（不是 7.5，窄缝窄，Not All Starting Points 抬高 direct-FT baseline）。

## 问题重定义（novelty 窄缝，codex）
DG-ReID 失败不只是 dataset-level domain shift，而是 **directed camera-pair / source-pair matching risk**——fine-tune 为源域 ID 判别，覆盖 foundation 保留的跨相机/跨数据集局部邻域几何，尤其伤高风险 camera-pair。

## 方法：Camera-Pair Foundation-Preserved Residual DG
- frozen F0（跨域 prior，不更新）+ 低秩 residual Rθ
- `z = norm(P(F0(x)) + β(pair-risk, x)·Rθ(x))`
- residual 只补源域判别，不重写高风险 pair 的 foundation 邻域拓扑
- preservation target = **cross-camera local rank topology**（保 F0 跨相机邻居 top-k soft dist，非单图 L2）
- β 由 source camera-pair risk 控（高风险小 residual）
- 避：normalization/IBN/MoE/meta/CLIP-prompt/CILP-FGDI

## ★PSC-JEPA 同质性 + U-shaped 判死（关键，避免 no-op 重演）
有同质风险（都"训练破坏 foundation prior + anchor 越强越 no-op"）。区别：PSC-JEPA continued-pretrain 改 backbone 保单域判别 vs DG frozen+residual 保跨域邻域结构。
**判死标准 = U-shaped sweet spot**：λ（preservation 权重）中间有峰值 = 真有救；单调 frozen↔FT 无峰值 = no-op 死（和 PSC-JEPA 一样）。

## cheap kill-switch（先验证前提，非写方法）
### 第一步：frozen cross-domain probe（零训练，本轮）
DINOv2-reg frozen 提特征，Market/MSMT/Duke in-domain mAP baseline。验证 frozen foundation 的 ReID 基线（fine-tune 要超它才有意义，破坏它则 fine-tune-harm）。

### 第二步：4 卡矩阵（frozen / head-only 短训 / direct-FT 30ep）
多源训 → held-out（Market+Duke→MSMT 等）。记：source gain / held-out gap vs frozen / neighbor overlap@20 topology drift / drift-failure Spearman。
- **Go**：direct-FT source 大涨但 held-out 比 frozen/head-only 低 ≥3 且 ≥2 协议 / head-only 优 full-FT held-out ≥2
- **Kill**：direct-FT 所有 held-out 赢 frozen/head-only ≥2（foundation-preserving 没燃料）→ 转 open-set/gallery-growth/distractor-aware lifelong

## 数据/基建
Market/MSMT17/occluded_duke 现成（3090 /root/work/SOLIDER-REID/data/）。DINOv2-reg timm `vit_base_patch14_reg4_dinov2.lvd142m`（transformers python3.8 不兼容用 timm）。foundation fine-tune 前 codex 三审 diff。

关联：`paradigm_shift/codex_dg_deepen.md`，memory [[paradigm-shift-occluded-reid-wall]]，exp363（AG negative control）。

 succeeded in 0ms:
# exp365 Wildlife Species-Conditioned ReID（换领域换量级，2026-06-27）

## 动机

LM-ReID 训练端真 measure 六点穷尽（塑造/对齐/分离 K 变体全死，见 design_lcrs/lrfd + memory [[lm-reid-trainside-shaping-kills-marg]]）。现有标准行人 ReID 数据 + 4 单卡训练端机制窄。**探透一层 = 自主转下一层**：codex 换量级调研第二名 = Wildlife/Animal ReID（真训练端换领域，避全死区）。

codex 深化核查（codex_wildlife_check.md）：**不是伪命题**，真价值不在"animal 数据集"，而在 **多物种少样本场景下，训练端如何把 shared animal ReID prior 分解成 species-conditioned identity evidence**。主线风险中等。

## 核心假设

动物 ReID 的难点是 **same-species hard negatives**（同物种个体极像）+ **rare/unseen species 少样本**。统一 backbone 给的是 shared-animal embedding；若能 **species-conditioned** 地分解出"该物种内部真正区分个体的证据"（per-species metric/centering），应能在 rare/unseen species 上超过 species-agnostic 的 global fine-tune。

## 候选机制（过 kill-switch 后才写）

SCREA: Species-Conditioned Residual Evidence Adapter —— shared identity core + species-conditioned residual（per-species centering/whitening 的可学版），adapter/LoRA 不碰大规模 full fine-tune。

## cheap kill-switch（第一步零训练 frozen probe，codex 建议）

1. 下 WildlifeReID-10k（214k 图 / 10k 个体；wildlife-datasets 库统一 dataframe + similarity-aware split）。
2. 抽特征：**MegaDescriptor-L**（动物 ReID SOTA）/ **MiewID-msv2** / DINOv2 / SOLIDER。
3. similarity-aware split + 构造 gallery：rare species / unseen species / 5-shot / same-species-only / all-species。
4. 指标（控 trivial 代理）：mAP/R1 + **false_in_top5/top10** + wrong-species-in-topk + same-species hard-neg error + per-species mAP 方差。

**Go 线**：best frozen 在 rare/unseen 5-shot 仍明显低，错误集中 same-species hard negatives；per-species centering/whitening **oracle >= +3 mAP**；all-species gallery 明显比 same-species-only 差（species 干扰真实存在）；MegaDescriptor/MiewID 没饱和。

**Kill 线**：MegaDescriptor/MiewID frozen 已很强，per-species normalization <+1；错误主要来自标注噪声/裁剪/单图不可辨（非 species-conditioned）；ordinary ArcFace/LoRA 已吃掉 rare/unseen 增益；species conditioning 只在 seen/common 涨、不迁移 unseen。

## 单卡可行性

可行。先缓存 embedding，零训练 probe 一张 3090/4090 足够。过线后训练只做 adapter/LoRA。下载走本地代理 → rsync 3090。

## 状态

待：① 探查 WildlifeReID-10k 下载（wildlife-datasets 库）② 装库 + 抽 MegaDescriptor/DINO/SOLIDER 特征 ③ frozen probe（per-species oracle）。过线 → 写 SCREA design；不过线 → 直接杀，不补 adapter rank/head/loss 小变体（codex 明确）。

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp366_active_evidence/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp367_single_support/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp366 Active Evidence Acquisition ReID（范式级方向 #1，2026-06-28）

## 动机

用户指令"放下 LM-ReID，找新 ReID 范式级创新"。codex 范式级调研 #1（7/10，最值得）：传统 ReID 给一张 query 就必须排序；**范式重定义=系统可花 1-3 次预算主动获取下一条视觉证据**（请求另一帧/另一 camera 视角/操作员二值 VQ）。先例 LLaVA-ReID/ChatReID/Inter-ReID 是**文本对话补全**，主动获取**视觉证据**（camera-view evidence acquisition）是空白；旧 human-in-loop 偏人工标注反馈，不是主动传感/证据预算。避所有探死方向（occluded/AG/DG/gallery/open-set/Wildlife/VI-ReID/lattice/SMPL/FM/test-time）。

## 核心假设

ReID 真实场景=多相机网络：query 在 camera A，系统可主动调 camera B 获取同人证据（预算受限）。**难 query 值得花预算获取第二证据，简单 query 不值；policy（预算分配给哪些 query）是真问题**。

## cheap kill-switch（零训练，cvpb_active_evidence_probe.py，frozen SOLIDER）

- baseline：single query mAP
- oracle-all：每 query + 同 ID 不同 camera 第二证据（multi-query mean）→ upper-bound
- **★policy**：只对 hard query（top1-top2 margin 小=不确定）花预算 20% 获取第二证据
- random：随机 20%（同 has_second 池公平对照）

**判定 GO**：policy gain / oracle-all gain ≥ 0.5 且 policy−random > 0.3 → 主动获取证据 policy 有真价值。
**DEAD**：policy ≈ random → trivial multi-query 无 policy 价值。

★**诚实设计**：避 codex 的 trivial oracle（multi-query 必涨 = upper-bound 不是创新），真验 policy（预算分配 vs random）。控 margin（top1-top2 = #false-in-topk 的代理）。自查抓到 2 个 bug（margins 长度 != len(qf) 退化 policy；policy hard 应只在 has_second 池选）已 fix。

## 预期

- GO → 设计轻量训练端 active-acquisition policy（学"选哪个 query/候选获取证据"），范式级第二 contribution。
- DEAD → 主动获取证据无 policy 价值（等 trivial multi-query），转 Generative Index（codex #2，6.5/10 真空白）。

## 状态

probe 跑中（3090，b984dv1y8，frozen SOLIDER exp260b 抽 Market query/gallery 特征 + camera split + oracle/policy/random）。

 succeeded in 0ms:
# exp367 Single-Support CVaR Episodic Loss（训练侧创新，2026-06-28）

## 动机

用户 goal：找训练侧创新发 CCF-B，不收手，不轻易说穷尽，审查调研交 codex，严谨，文档记好。codex 训练侧深度调研 #1（最务实）：训练时每 ID 只用单图 support 定义身份，对 worst-support 选择做 CVaR 优化。回应 exp109 根问题（single-image support incomplete）。**纯训练侧**（episodic loss，输出常规 descriptor），严格非 test-time/检索侧/范式重定义。

codex 已避所有死区（visibility/masked/CLIP-align/synthetic/topology/DG-foundation/noisy-label/long-tail），2024-26 novelty 空白：few-shot/DG 有先例，但**标准 Market/MSMT/Occluded 监督训练里"单图 support 是否足够定义身份"做成主训练目标，2024-2026 没看到直接占位**。

## 核心假设

ReID 训练用 multi-shot gallery（每 ID 多图），但模型学到的身份边界可能依赖"见过该 ID 多个 view"。部署常 single-shot（单图 support 定义新身份）。训练时**强制单图 support + CVaR worst-support 优化**，逼模型学"从任意单图恢复完整身份边界"的鲁棒表征，而非依赖 multi-view 平均。

## cheap kill-switch（零训练，cvpb_single_support_probe.py）

复用 Market 特征 cache（frozen SOLIDER exp260b）。每 gallery ID 只留 1 图：
- full-gallery：上界
- best-support：每 ID 选最好单图（同 ID query 平均 sim 最高，oracle 上界）
- random-support：每 ID 随机 1 图
- worst-support：每 ID 选最差单图（CVaR worst-case 目标针对的）

**GO**（support 选择是真训练瓶颈）：worst 比 full 掉 > 3 mAP 且 **best−worst gap > 3 mAP**（哪张 support 图很重要 = support 选择 matters，值得 CVaR 优化）。
**DEAD**：best≈worst（哪张 support 都一样，没 support 选择价值）或 single≈full（单图够）。

★诚实设计要点：单图 vs 多图必掉 mAP（少正样本）是 trivial，所以**关键判据是 best−worst gap**（同样单图，选择重不重要），不是 single<full。codex 审 probe 验这个设计是否真有意义（用户要审查交 codex）。

## 审查（codex，用户要求）

codex 审 probe（codex_review.md）：kill-switch 设计是否有意义、best/worst per-ID 选择逻辑、#false-in-topk 控制。

## 预期

- GO → 设计 Single-Support CVaR episodic loss 训练（每 ID 单图 support + worst-case 风险优化），训练侧第一 contribution，full fine-tune 前 codex 三审 diff。
- DEAD → support 选择无训练价值，转 Equivariant Routing（codex 训练侧 #2，routing 等变非 embedding 一致）。

## 训练设计（codex 调研 63517，probe GO 后）

★**novelty 真空白（codex 确认）**：2024-26 标准监督 person ReID 没有"episodic single-support training + CVaR worst-support tail optimization"直接先例（检索 single-support/worst-support/CVaR-ReID/support-selection 都没命中）。邻近但不同：CFReID(continual few-shot)/DG-episodic(domain-invariant)/ProtoNet(novel-class 优化 prototype 平均非 tail)/batch-hard(hard pair mining 非 support tail)。**claim 写窄**：不发明 episodic/CVaR，是"标准 ReID 优化单图 support 定义身份的 tail risk"。

★**训练设计（two-level CVaR，加项不替换）**：
- episode N ID × K 图，每 ID 1 support + K-1 query。
- `risk(y,s)` = 该 support 对同 ID 多 query 的 CE 失败。
- `L_cvar_y = CVaR_α(support risks)`，`L_ss_cvar = mean_y`。α=0.7/0.8。
- `L = L_id_ce + L_triplet + λ·L_ss_cvar`（λ 0.1→0.3 warmup）。
- support/query 都梯度 + 保 CE+Triplet 防 collapse。两级聚合(support risk→CVaR)非 CVaR over all pairs(避 batch-hard)。

★**避坑（避六点定律）**：不写 support completion/feature alignment/prototype compression/query-dependent selector。训练用 label 算 worst support 可以，测试不选 best/不用 query label，输出常规 descriptor。

★**cheap 验证路径**：① frozen head smoke(10-20ep, worst/random +0.8~1.0, 失败不判死) ② last-stage(20-40ep, worst+2/random+1/gap 缩≥2/full 不降>0.5, 对照普通 CE+Triplet continued FT) ③ full FT。

★**风险+对照（防退化 hard-mining）**：报 batch-hard/pair-CVaR/random episodic CE 三对照，只 support-level CVaR 独立赢才站得住。报 missing-positive/cross-cam 覆盖/false10 random std。

★CCF-B 6.5/10（last-stage/full FT 抬 worst/random + 赢 hard-mining → 7.5；只改 diagnostic 但 full-gallery 不涨 → 4 附录）。

## 状态

probe v2 GO（best-worst 12.27 不被 #false 解释，codex 两轮审）。codex 训练设计 GO（novelty 空白 6.5/10）。下一步：写 frozen head smoke 训练（cheap 第一步，复用 Market cache + projection head + episodic CVaR loss）+ codex 三审 diff（full FT 铁律 + 审查交 codex）。

codex
这些本地草案里，AG/DG/Single-support/Camera-transport 已经被你这次点名排除或接近排除；LM-ReID 本地判断也卡在训练端塑形破坏 K-marginalization。我会把它们当负边界，不进最终 Top5，除非作为“先例风险/不要做”的证据。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp368_camtransport/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,180p' experiments/exp366_active_evidence/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp366 Active Evidence Acquisition — monitor

## cheap kill-switch（frozen SOLIDER, 零训练, 2026-06-28）

验 codex 范式 #1：query 主动获取第二证据（同 ID 不同 camera），policy（margin 选 hard query 花预算）能否接近 oracle 且 >> random。

| 数据集 | baseline | oracle-all (2nd) | policy (hard 20%) | random (20%) | frac | verdict |
|---|---|---|---|---|---|---|
| Market（exp260b 强）| 94.43 | +2.51 | **+0.31** | +0.48 | 0.12 | DEAD（policy<random）|
| Occluded-Duke（exp004_pfm⚠️）| 3.09⚠️ | +6.45 | **+1.29** | +1.31 | 0.20 | DEAD（policy≈random）|

⚠️ Occluded baseline 3.09 异常低（exp004_pfm ckpt 配 market config 的 FrozenExtractor 加载不匹配）；但 policy vs random 是相对比较、不依赖 baseline 绝对值，结论成立。

## ★VERDICT DEAD（两数据集坐实）

margin（top1-top2 检索不确定性）**不是好的"值得获取证据"预算信号**：Market policy+0.31<random+0.48；Occluded policy+1.29≈random+1.31。两数据集 policy 都 ≈/< random。

**★范式根本困难（诚实诊断）**：
1. 系统不知道哪个 query 的第二证据有用（要获取才知道=鸡生蛋）。
2. margin 小（难 query）给证据没用——第二证据也可能难（occluded query 另一张也 occluded）。
3. 任何"检索不确定性"policy（margin/entropy 同质）都救不了，因为不确定 ≠ 第二证据能救。

oracle headroom 真实（occluded +6.45 R1 大涨），但**没有 cheap policy 信号能逼近它**。主动获取证据要价值，需要预测"第二证据质量"，而那本身要先获取（鸡生蛋）。

## 决定

Active Evidence policy DEAD（cheap kill-switch 半小时验透，没浪费训练）。转 **Generative Index ReID（codex 范式 #2，6.5/10 真空白）**：gallery identity 离散 token，query 生成 code prefix，kill-switch=PQ code recall<95% 则杀。

 succeeded in 0ms:
# 实验 exp368: Camera-Pose Transport Operators

## 动机
- Single-Support CVaR DEAD（exp367, worst 跨 camera/pose gap 不可训练改善, oracle-headroom 墙）→ 转 codex 训练侧 Top3。
- 现有 ReID 学 camera-invariant descriptor（抹平 camera nuisance）。但 invariance 可能丢判别信息。
- codex Top3（CCF-B 6.2）：不学 invariance, 学 low-rank transport 把 descriptor 从一 camera cell 映到另一 cell 后再比（comparability operator, not invariance）。

## 核心假设
- camera 间有系统性 feature shift（非随机），train ID 可拟合 transport map W_{a→b}，test query transport 后比 gallery 更准。

## cheap kill-switch（frozen, 零训练, 豁免审查）
- frozen global feat（/tmp/ae_feats.npz query/gallery + cam, exp260b SOLIDER）
- train ID 拟合 cam pair ID-mean ridge transport W_{a→b}（cam a ID-mean → cam b ID-mean, lam=1.0）
- test query(cam a) transport 到 gallery cam b 后 cosine
- 脚本 cvpb_camtransport_probe.py

## 成功线
- transport mAP Δ>+0.5 vs baseline cosine, 且明显 > camera-centering 对照（证不只是去 camera bias）
- 按 baseline #false@10 分桶 ΔAP（控 trivial: transport 增益应在难桶更显著, 非均匀偏移）
- 若 transport 抬 → camera invariance 不够 transport 有 headroom → codex 细查 novelty + 训练版; 不抬 → DEAD 转 Top2 Counterfactual Part-Contradiction

## 对照
- baseline: 直接 cosine（无 transport）
- camera-centering: per-cam mean 减（弱对照, 证 transport 不只是去 camera bias = 仅 first-order）

## 先例风险（codex Top5）
- CamStyle/camera-aware 先例密; 新意只在"transport not invariant"。GO 后 codex 细查 novelty 避先例。

## 诚实标注
- codex 明说训练侧天花板 6.2-6.8（没 8-9, 要换数据/task）。Top3 6.2 是 cheap 验先（有 headroom 再深入）。一方向证伪→不停→下一个。

exec
/bin/zsh -lc "sed -n '1,180p' experiments/exp359_lm_reid/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp359 LM-ReID monitor

## 训练配置
- 脚本 `cvpb_lm_reid_train.py`，微调 exp260b backbone（pose-OFF global feat，pose_dict=None）
- Market 751 id / 12936 img，M=3 lattice variants/img（1 canonical + 2 随机轴），h∈{16,24,32} severe-biased（0.5/0.3/0.2）
- lr 1e-3（cosine + warmup5），40 epochs，BS=64（P16×K4），save_every 10，AMP
- `L = L_id + 1.0·L_marg + 0.2·L_cons`（L_adv off，lam_adv=0）
- lab-3090 PID 1335155，log `/tmp/exp359_lm_reid.log`

## 双审
- Claude 自审（主 Opus 循环）+ Codex 独立审 round-2 = approve（round-1 抓 5 findings 全修：High-1 RNG per-epoch / High-2 随机轴 / Medium-3 GRL 单 λ / Medium-4 per-slot triplet）。

## smoke 结论
- **pipeline 健康**（no crash，4 loss 计算，backward，save 工作）。
- 150-id smoke（lr 1.5e-3, 2 epoch）：iter 打印是**累积平均**（agg/n，注意别误读）；反推瞬时 loss epoch 末 6.5→5.1（在学），但 epoch 边界有 spike（每 epoch fresh 随机 variants 冲击 + warmup LR 升）。
- 判断：32/150-id 微子集 spike 偏大（模型过拟合那几十 iter 的 variants）；真实 751-id × ~202 iter/epoch 见足够多 variants → 应更稳。acc 是分类 acc 非检索 mAP，真判据=eval。

## 进度
- **[启动 2026-06-25]** 真实训练 40ep，lr 1e-3。
- **[epoch 0 健康 ✓]** GPU 100%/13GB（AMP，无 OOM），it200 L=3.30 / acc=0.845（累积平均），loss 降、acc 高、**完全无 spike**——**坐实 smoke 的 spike 是 32/150-id 微子集 artifact（classifier 不平衡+BN 偏移），full 751-id 训练稳定**（smoke L 飙 9-13，这里仅 3.3）。
- 速度: **~5min/epoch**（306-314s/epoch；上轮误判 18min）→ **epoch-10 ckpt ~50min**。
- **[epoch 1-3 健康]** L 稳定 ~3.3（warmup LR 升轻微波动），acc 0.846→0.888。
- **[epoch 5-7 收敛漂亮]** warmup 结束 cosine 段：**L 3.3→1.66，acc 0.888→0.979，cons loss 0.035→0.020（模型在变 lattice-invariant，机制在工作）**，marg 1.14→0.28。无崩溃。
- 触发器 b3e9ya110 等 epoch-10 ckpt → eval。训练 epoch 40 全部完成（transformer_40.pth），acc→1.000，cons→0.011。

## ★epoch-10 eval 结果（method-vs-trick 关键，2026-06-25）

fine-tuned ckpt transformer_10，h=16（SANITY HR mAP 83.31，frozen 94.43）：

| | mAP | 增益分解 |
|---|---|---|
| frozen single | 42.65 | baseline |
| frozen lattice ensemble | 46.87 | 零训练上界 |
| **LM-ReID single** | **69.96** | +27.3（"在 LR 上训"=标准 CR-ReID，**非我们创新**） |
| **LM-ReID lattice-mean** | **72.60** | +2.63 over single（lattice-marg，**真创新**） |
| LM-ReID ordinary-TTA | 69.09 | −0.87 |
| **LATTICE−TTA** | | **+3.504**（lattice-specific，干净 beat trivial TTA） |

h=24：single 78.67 → lattice 79.67（+1.00），LATTICE−TTA +0.978。

**诚实分解**：①+27 大头是 LR fine-tune（标准），②lattice-marg 真贡献 +2.63 但 robust beat TTA +3.504（frozen +3.04 / trained +3.504 都成立）。**HR-gallery 退化担心是多虑**（SANITY 掉但 LR mAP 高=模型学了 LR↔HR 对应，牺牲 HR-HR 换 LR-HR 正是 CR-ReID 目标，已纠正误判）。codex 过线被远超**但大头是 LR fine-tune 不是 lattice 机制** → 真判据靠 ablation。

## ★ablation + final eval（2026-06-25 在跑）

- **no-LM-loss ablation**（PID 1347672）：M=3 保留 LR 增强，关 L_marg/L_cons（lam_marg=0 lam_cons=0），只 L_id。隔离 consistency 训练独立价值。out exp359_abl_noLMloss，等待 bw6sm13th。
- **final eval**（PID 1347722）：epoch 40 ckpt，h=16/24/32 headline。log /tmp/exp359_eval_ep40.log，等待 bju02h54q。
- 判：LM-ReID single > ablation single → consistency 训练有独立价值（强 method）；≈ → 只是"LR 增强 + 测试时 lattice ensemble"（弱，test-time trick 级）。

## ★final eval（epoch 40 ckpt，2026-06-25，headline）

SANITY HR 86.09（epoch-10 83.31 回升，frozen 94.43）。

| h | single | lattice-mean | lattice-MaxSim | TTA-MaxSim | **LATTICE−TTA** |
|---|---|---|---|---|---|
| **16** | 75.71 | 78.01 (+2.30) | **78.04 (+2.33)** | 75.04 | **+3.006** |
| 24 | 81.99 | 82.74 | 83.00 (+1.01) | 81.82 | +1.174 |
| 32 | 84.10 | 84.60 | 84.81 (+0.71) | 84.11 | +0.697 |

- epoch 40 > epoch 10（h=16 single 69.96→75.71，+5.7），lattice-marg 增益随 h 递减（+2.33→+1.01→+0.71）= 符合"分辨率升高 lattice uncertainty 消退"机制，**三个 h 全 beat TTA**。
- **诚实**：大头是 LR fine-tune（single 75.71 vs frozen single 42.65=+33，标准 CR-ReID，非创新）；**lattice-marg 真贡献 +2.33（robust，frozen +3.04/trained +3.006 一致 beat TTA）= 方法的 lattice-specific 贡献，但是 test-time-trick 量级**。
- **method-vs-trick 终判仍靠 ablation（no-LM-loss）**：若 LM-ReID single > ablation single 则 consistency 训练有独立价值。

## ★ablation ladder（2026-06-25，method-vs-trick；OSS 打通后 ablation 移 4090 mmpose-abu，3090 跑 M=1）

h=16 single / lattice(best) / LATTICE−TTA：

| 配置 | single | lattice | LATTICE−TTA |
|---|---|---|---|
| **M=1 plain**（标准 CR-ReID，M=1，lam_marg/cons=0） | 73.62 | 75.95 | +3.727 |
| **M=3 no-LM-loss**（增强但关 L_marg/L_cons） | **77.44** | **79.90** | **+2.576**（SANITY 88.92） |
| **M=3 full LM-ReID** | 75.71 | 78.04 | +3.006 |

- **full − M=1 = +2.09**（single+lattice 都 +2.09）= M=3 增强 + consistency 合计。
- **★★★命门结果（2026-06-25）= 证伪 method**：no-LM-loss single **77.44 > full 75.71（+1.73）**！**consistency 训练（L_marg+L_cons）= HARMFUL 有害**，不是无用。分解：M=3 增强 +3.82（73.62→77.44，大）；consistency −1.73（77.44→75.71，过度正则压判别性）。SANITY 也证 no-LM HR 88.92 > full 86.09。**LM-ReID 作 method 死**（以 consistency 为创新，比简单 M=3 增强+测试时边缘化更差）。剩 test-time lattice-marg robust（五配置 beat TTA）但 trick（codex 5/10）。
- lattice-marg 三配置都 robust beat TTA（M=1 +3.727 > full +3.006：越不 lattice-invariant 的模型、测试时边缘化获益越多 = "训练内化测试时边缘化"干净机制故事）。
- cons-only（lam_cons=0.2 only）在 3090 跑（bsc27k3vn，补全消融）。

## ★salvage（2026-06-25 codex 后）：LM-ReID 没全死，是 consistency 版死了

salvage codex（`litreview2/lmreid_salvage.md`）：主救法 = **重构**，不卖 consistency，卖"**lattice-aware augmentation 增判别性 + test-time 对 lattice hidden variable 边缘化**（非压成同一 embedding）"。信心 6→7/10 after 命门对照，CCF-B 有条件可行。核心：no-LM-loss 77.44 已最强 = "invariance 死、lattice hidden variable + marginalization 没死"。

**★命门对照（4090 跑中 `b46h62df9`）= M=3 ordinary-aug no-LM**（同 3 views/h/步数，variants 换普通 random crop/flip/color，非 lattice）：
- ≈77+（=lattice no-LM）→ M=3 增益只是"更多增广"，training-side 创新缩水 → 纯 test-time。
- 明显低（−1.0+）→ **lattice-aug 是 lattice-specific** → 改名 **Lattice-Aware Augmentation + Lattice-Marginalized Retrieval** 重投。
- 脚本加 `--aug_mode ordinary`（make_ordinary_variants），smoke 过。

**★命门结果（2026-06-26）= lattice-aug 训练端不特殊（仅 +0.54）**：

| h=16 | single | lattice | LATTICE−TTA | SANITY |
|---|---|---|---|---|
| ordinary-aug（M=3 随机增广） | **76.90** | 79.28 | +1.624 | 88.96 |
| no-LM-loss（M=3 lattice） | **77.44** | 79.90 | +2.576 | 88.92 |

- lattice-aug 仅 **+0.54** over ordinary-aug（阈值 −1.0+ 才算 lattice-specific）；高 h 更平（h=24 84.00≈84.28，h=32 86.48≈86.44）→ **M=3 训练端增益主要是"更多增广"，lattice 训练不特殊**，salvage 主路 (a) 弱。
- **★不对称（干净）**：test-time lattice-marg 四配置全 beat TTA（M=1 +3.727 / ordinary-aug +1.624 / lattice no-LM +2.576 / full +3.006）→ **lattice 在 test-time 是真 hidden variable，在 training 不特殊**。
- 剩：(b) 纯 test-time（robust 但 codex 4.5/10）/ (c) Hard-Lattice ERM（被命门削弱）/ (d) pivot。**post-命门 codex（53466，bulrvjdg2）定夺**：含"test-time 特殊 / training 不特殊"不对称能否成更干净卖点。cons-only（3090）仍跑（诊断 L_cons vs L_marg，现次要）。

## ★LM-S2 inference 主实验（2026-06-26，重定位后第一个实验）= test-time 故事干净

no-LM-loss ckpt，h=12/16/20/24/32，single vs lattice-marg vs 普通TTA（MaxSim）：

| h | single | lattice | LATTICE−TTA | 普通TTA gain |
|---|---|---|---|---|
| 12 | ~66.7 | ~72.2 | **+6.534** | **−1.005** |
| 16 | 77.44 | 80.00 | +2.734 | −0.177 |
| 20 | 82.49 | 83.69 | +1.052 | +0.153 |
| 24 | 84.28 | 85.28 | +0.979 | +0.018 |
| 32 | 86.44 | 87.18 | +0.537 | +0.206 |

- ① lattice-marg **全 5 分辨率 beat 普通TTA**（robust）；② **优势随 h 单调递减**（+6.5→+0.5）= sampling-lattice 是 severe-LR nuisance 的干净证据；③ **severe LR（h=12/16）普通 TTA 反而有害**，lattice-marg 大涨 = clean contrast（防"普通 TTA"质疑）。
- 强化 test-time decision-marginalization 故事。下一步 LM-S3（logsumexp 聚合）/ LM-S4（phase/bbox/kernel 因子）+ 更强 TTA baseline（kill-switch make_tta_variants 加 color/resize）。
- **cons-only eval 结果（L_cons 诊断，2026-06-26）**：single **76.895**（h=16）vs no-LM-loss 77.44 / full 75.71 → **L_cons 单独只 −0.55，L_marg 才是大元凶（≈−1.18）**（codex 预测"L_cons 主杀"错）。**强化 asymmetry：训练时边缘化 L_marg 有害 / 测试时边缘化有益**——decision-level marginalization 故事更锐。
- **LM-S4 因子分解（h=16，single 77.44）**：phase +1.758 / **bbox +2.842（最大）** / zoom +1.702 / all +2.557 → **bbox 轴（±1 LR-pixel 检测框偏移）主导**，9 bbox > 3+3+3 混合。codex 预测 phase 不对，但 bbox 故事更直观（severe LR 检测框 ±1 LR 像素 = 几个 HR 像素 = 真实 crop 不确定性）。三轴全 beat TTA。→ nuisance 主要是 **LR 检测框/crop 不确定性**。
- LM-S2-strong（强 TTA 防御）还在跑；**LM-S5 Hard-Lattice ERM**（3090，bqd5skbum，loss_mode=hard 已实现+smoke 过）epoch 0，待 Hard-ordinary 对照（需 Hard-Lat ≥+0.8 over Hard-ord 才算训练端活）。

novelty：无 exact prior（sampling-lattice as LR-ReID hidden variable + marginalization）；BlurPool(aliasing)/FlipReID(flip consistency) 是相关先例；"consistency 有害"不单独成 paper 但强消融卖点。
training-side 转向备选（若 ordinary-aug 抹平 lattice 特异性）：Hard-Lattice ERM/CVaR（优化最难 variant 的 CE+triplet，非压 embedding）/ set-wise retrieval training。

### [2026-06-26] 训练端三大类全死（LSRC eval 确认）

| 指标 | no-LM-loss | LSRC lam0.5 | Δ |
|---|---|---|---|
| HR sanity | 88.92 | **85.84** | **−3.08** |
| h16 single | 77.44 | 75.70 | −1.74 |
| h16 lattice(MaxSim) | 79.90 | **77.98** | **−1.92** |
| h24 single | — | 82.31 | |
| h24 lattice(MaxSim) | — | 83.27 | |

- **LSRC（backbone set-loss，bag-to-bag set-supcon+neg-tail，4090 lam0.5 full fine-tune）死**：训练 acc 1.000 过拟合训练集，测试全掉，backbone 被训坏（HR sanity 掉 3 点）。marginalization 在受损 backbone 上仍 +2.288（证机制本身没问题，是 backbone 被训坏）。**asym 不用试**（对称 M×M 给 gallery-side oracle = 宽松上界，对称死→非对称必死）。3090 lam1.0 必死。
- **训练端三大类全死**：① frozen 重投影/重加权（LS-MRT +0.028 / LPA +0.075 — 无 headroom）；② backbone 改 loss（LSRC −1.9 / consistency −1.73 — 损判别力）；③ robust ERM（Hard-Lattice 76.9<77.44）。
- **强结论**：no-LM-loss backbone 已是 LR-ReID 好特征，**test-time decision marginalization 是唯一有效杠杆**。论文 = test-time 6/10 核心 + 训练端系统反例。备选 BLC（input canonicalize，design_blc.md）market 受限未验。启 codex（train3_{fourthclass,paperstrategy}）。审查纪律见 codex_review_lsrc.md（审出 Critical+High 已修）。
- **codex final（train4，8.5/10）判训练端确定无空间，别硬凑**：4 类全封（frozen/sidecar 无 headroom含LATS / backbone-loss 伤判别 / robust-ERM 没赢 / BLC 逻辑封住）。训练端定论穷尽（8 机制 + 4 codex）→ 转 test-time 论文 + "Why Training-Time Invariance Fails" 反例节。

### [2026-06-26] K-sweep compute-accuracy（no-LM-loss baseline，h=16）

| K | mAP | gain over single | 收益% |
|---|---|---|---|
| 1 (single) | 77.44 | — | 0% |
| 3 | 78.73 | +1.29 | 53% |
| 5 | 79.61 | +2.14 | 87% |
| 9 | 79.90 | +2.46 | 100% |

- **K=5 已达 87% 收益（79.61≈79.90）**，K=3 中等（53%）。compute-accuracy 边际递减 → 论文"防 compute"论点：K=5 性价比高（省 4/9 compute 保 87% 收益），K=9 完整。marginalization 的 compute 可调，K=5 是 sweet spot。

### [2026-06-26] LM-S3 聚合消融（no-LM-loss，K=9，5 分辨率）

| h | single | mean-feat(embed) | MaxSim(hard) | logsumexp(soft) |
|---|---|---|---|---|
| 12 | 66.72 | 72.25 | 72.03 | **73.01** |
| 16 | 77.44 | 79.84 | 80.00 | **80.28** |
| 20 | 82.49 | 83.41 | **83.69** | 83.62 |
| 24 | 84.28 | 85.02 | **85.28** | 85.16 |
| 32 | 86.44 | 86.87 | **87.18** | 87.02 |

- **soft decision marginalization（logsumexp，LM-ReID 公式 s=τlog[1/KΣexp(cos/τ)]）在 severe LR（h12/16）最优**；mild LR（h≥20）hard-max 略超。三种聚合都 >> single；decision-level（max/logsumexp）≥ embedding-mean（h≥16）。差距虽小（±0.3）但完整 controlled ablation，logsumexp 是理论 motivated 的 sweet spot。

### [2026-06-26] LM-S2-strong 5 分辨率（强 TTA 防御，no-LM-loss）

| h | phase-lattice best | strong-TTA best | LATTICE−TTA |
|---|---|---|---|
| 12 | +5.494 | −1.781 | **+7.275** |
| 16 | +2.811 | −0.526 | +3.337 |
| 20 | +1.195 | −0.035 | +1.230 |
| 24 | +0.991 | −0.144 | +1.135 |
| 32 | +0.759 | +0.003 | +0.757 |

- **lattice marg 全 5 分辨率 beat 强 TTA（pad-crop+resize-jitter+color），+0.76~7.28，h 越低优势越大**。severe LR（h12）强 TTA 本身 −1.781（有害），lattice +5.494 → 干净堵死"不就是多裁几次"质疑。论文核心防线。

### [2026-06-26] backbone 泛化（Swin-small baseline，market，LATTICE−strong-TTA）

| h | single | MaxSim | LATTICE−TTA |
|---|---|---|---|
| 12 | (弱) | — | +0.778 |
| 16 | 41.41 | 46.20(+4.80) | +3.061 |
| 20 | 61.92 | 66.43(+4.51) | +3.162 |
| 24 | 70.31 | 73.90(+3.60) | +2.370 |
| 32 | 81.97 | 83.93(+1.96) | +0.883 |

- **lattice marginalization 在 Swin-small backbone 上也 beat 强 TTA（h16/20 +3.06/+3.16）**，证机制不依赖 SOLIDER backbone（push-7.0 kill-switch②：两 backbone 成立 ✓ Swin+SOLIDER）。Swin baseline LR 更弱（h16 single 41.4 vs SOLIDER 77.4）→ lattice 敏感度更高。


exec
/bin/zsh -lc "sed -n '1,180p' experiments/exp367_single_support/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp367 Single-Support CVaR — monitor

## cheap kill-switch（frozen SOLIDER Market exp260b, 零训练, 2026-06-28）

### v1 → codex 审抓 3 High → v2 修

| | full | best-support | random-support | worst-support | best-worst gap |
|---|---|---|---|---|---|
| v1（污染）| 94.43 | 98.61（>full!）| 96.46 | 88.17 | 10.44 |
| **v2（干净）** | 94.43 | 76.08 | 73.36±0.22(20seed) | 63.82 | **12.27** |

★v1 污染（codex 审抓）：single-support 跳无 positive query（比不同子集）+ distractor 压 1 张（负样本池变）→ best/random>full 假象。
★v2 修（codex 3 High）：common-valid query 共用 + distractor 全量 + 20 seed + missing 记 0。**single-support 都 <full（合理少正样本），best-worst 12.27 + random-worst 9.54，false10 best0.923≈worst0.927 → gap 不被 #false-in-topk 解释**。

### codex 两轮审（用户要审查交 codex）
- v1：needs-attention，3 个 High（valid-query 污染 / 负样本池变 / kill-switch 不硬）。
- v2：needs-attention（轻微残留非致命）：false10 没给 random mean/std + go 没检查 false10 + missing 可能混 camera-coverage。best/worst oracle 用 query-label 可接受。

## ★VERDICT GO（基本可信）

support 选择有 oracle headroom（best-worst 12.27，不被 #false 解释），单图 support representation 是真训练瓶颈。**诚实标注**：best/worst 用 query-label oracle 上下界，证 headroom 存在；训练能否学到（不用 query）要 Single-Support CVaR train 验。

## 下一步

codex 调研 Single-Support CVaR 训练设计 + novelty 确认（63517）：novelty 真空白（episodic single-support+CVaR worst-support tail 标准 ReID 无直接先例），two-level CVaR 设计，cheap 验证路径，CCF-B 6.5/10。详见 design.md。

## frozen head smoke（codex cheap 路径 #1，2026-06-28）——失败

frozen backbone + projection head 训 episodic single-support CVaR 20ep（codex 审 loss 实现基本对，two-level 一致，不退化 hard-mining）：

| | frozen baseline(probe v2) | frozen head CVaR smoke | Δ |
|---|---|---|---|
| full-gallery | 94.43 | 93.89 | **−0.54** |
| random-support | 73.36 | 72.98±0.28 | **−0.38** |
| worst-support | 63.82 | 62.09 | **−1.73** |

**全部掉**（codex 成功线 worst/random +0.8~1.0 未达，反而掉）。

★诚实诊断：① **train loss 几乎 0（0.004）= episode 太易**（N=16 id 分类，support-query 同 id 分类到 16 id 太易）→ **CVaR worst tail≈0，CVaR term 没起作用**；② head 学 episode 分类过拟合 → eval 掉（frozen+projection 只能旋转特征，codex 预言）。

★codex 明确"frozen 失败不判死"（frozen 不够，可能要改 backbone）。但 loss 0 是 episode 设计问题，要修（增大 N / 用 gallery distractor 当负样本，让分类难、CVaR 起作用）才能真验机制。

## frozen head N=128（episode 修难，2026-06-28）——cvar≈random

N=16 loss 0（CVaR 空转）→ 增 N=128 让 episode 难、CVaR 起作用（loss 0.085→0.056）：

| mode | full | random-support | worst-support |
|---|---|---|---|
| frozen baseline(probe v2) | 94.43 | 73.36 | 63.82 |
| N=128 **cvar** | 94.25 | 73.28 | 63.36 |
| N=128 **random**(无 CVaR) | 94.24 | 73.26 | 63.29 |

★**cvar ≈ random**（三项几乎一样）→ CVaR term 在 frozen 特征上不带来差异。cvar/random 都 ≈ baseline（略掉 0.1-0.5）→ frozen head 训练没提升。

★诊断：① frozen head（projection）不够（≈baseline，codex 预言单线性头只能旋转改不了特征）② CVaR term 在 frozen 旋转空间没用（cvar≈random）。只有 last-stage（解冻 backbone 改特征）能区分"frozen 不够"vs"CVaR 机制本身弱"。

## last-stage backbone 训练（codex 四轮审 approve, 2026-06-28）

解冻 swin base.stages[-1]+norm3+bottleneck/classifier，episodic single-support CVaR loss，3 mode。codex 四轮审 approve（Critical make_optimizer → High eval/train 口径 → 修 → approve；ss_cvar_laststage.py + codex_review_laststage1-4.md）。

### cvar mode（epoch 20）—— DEAD

| | full | best | random | worst | best-worst gap |
|---|---|---|---|---|---|
| frozen baseline(probe v2) | 94.43 | 76.08 | 73.36 | 63.82 | 12.27 |
| cvar e20 | 94.41 | 76.05 | 73.40 | **63.62** | **12.42** |

★ss_cvar 不空转（~0.025-0.14，比 frozen smoke N=16 的 0 好），但 **worst 63.62<63.82（略掉）+ gap 12.42>12.27（略增）**——Single-Support CVaR 没改善 single-support 鲁棒性，反略负。codex 成功线 worst+2 完全未达（反向）。

★机理：worst-support 难来自 query-support 跨 camera/pose gap，训练优化 support 选择改不了本质难度（像 exp109 oracle headroom 墙：best-worst gap 是 identity-conditioned 不可训练实现）。lam=0.3 base 主导 + ss_cvar 信号太小。

### random 对照（epoch 20）—— 坐实 DEAD

| mode | full | random | worst | gap |
|---|---|---|---|---|
| cvar | 94.41 | 73.40 | 63.62 | 12.42 |
| random(无CVaR) | 94.41 | 73.44 | 63.75 | 12.32 |
| plain(CE+triplet) | 94.45 | 73.38 | 63.90 | 12.10 |
| baseline | 94.43 | 73.36 | 63.82 | 12.27 |

★cvar≈random≈plain≈baseline（worst 63.62/63.75/63.90/63.82 全 ≈，差<0.3 噪声）：CVaR term 无用（cvar≈random）+ last-stage FT 不改善 single-support（cvar/random/plain 都≈baseline）。**3 mode 完整坐实 Single-Support CVaR DEAD**。

## 决定

cvar DEAD（worst 不可训练改善）。等 random/plain 对照坐实 → 记 memory（Single-Support CVaR：probe oracle headroom 12.27 存在但训练不可达，worst 跨 camera/pose gap 不可训练改善，又一个 exp109-style oracle-headroom 墙）→ 转 codex 训练侧 #2 Equivariant Routing。严谨 build + 四轮 codex 审跑出干净负结果，比脏 GO 强。

exec
/bin/zsh -lc "sed -n '1,180p' experiments/exp361_psc_jepa/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp361 PSC-JEPA — monitor

## Stage-A continued-pretrain（2026-06-26，4090，50 ep）
- 健康收敛：L 3.0→0.13，**tok_std 0.49→0.998（C1 防坍缩成功，全程不坍缩）**，cos_drop 0.95，var hinge 满足。
- ckpt pscjepa_10/20/30/40/50.pth（205 keys，backbone. prefix，H1 修复生效）。
- smoke 抓到 SOLIDER swin `.train()/.eval()` 不返回 self 的运行时 bug（审查覆盖不到），已修。

## Stage-A fine-tune kill-switch（2026-06-26，4090 PSC-JEPA vs 3090 plain，Occ-Duke 120ep，同 config 只差 PRETRAIN_PATH）

**★负结果（同 epoch 对比）**：
| epoch | PSC-JEPA mAP | plain mAP | 差 |
|---|---|---|---|
| 10 | 15.9% | 33.1% | **−17.2** |
| 20 | 27.8% | 42.8% | **−15.0** |
| 70 | 39.9%（平台）| — | — |

PSC-JEPA epoch 70 才 39.9%（平台），plain epoch 20 就 42.8%（趋势更高）。**continued-pretrain 让 fine-tune 更差，不是更好。**

**诊断：Stage-A 裸 continued-pretrain 破坏 SOLIDER backbone 判别性（catastrophic forgetting）** —— Stage-A **无 L_solider_anchor 防遗忘**（design 里 Stage-B 才加）。partial-view JEPA 把 backbone 从 ReID 判别表征拉偏，fine-tune 50+ ep 拉不回。

**★final 确认（2026-06-26）**：PSC-JEPA **120ep final mAP = 41.0%**（已 ENDED，平台）；plain @ epoch 60 已 **52.9%**（还在涨，120ep ~55%）。差 **−12 且会更大**。

**结论**：kill-switch Stage-A **FAIL（严重）**（PSC-JEPA 41.0 << plain ~55，差 ~−14，远不是 ≥+0.7）。**catastrophic forgetting 坐实** = 裸 continued-pretrain（无防遗忘）严重破坏 SOLIDER 判别性，fine-tune 拉不回。但**诊断清楚 = forgetting**，design 预期内（Stage-A = 骨架/防坍缩验证，不主张 novelty；Stage-B 才防遗忘 + support bank）。不是死路，是诊断明确的迭代。

## Stage-B 修复方向（防遗忘 + 真 novelty）
1. **L_solider_anchor（防遗忘，关键）**：frozen SOLIDER backbone（swin_tiny.pth 不更新）= anchor teacher；student 可见区 part token 蒸 frozen SOLIDER token（cos，gvis 掩码 visible）。锚住可见区判别性不遗忘，JEPA 只在 dropped 区学 completion。
2. **pseudo same-ID support bank（B 类 novelty）**：T_bank 同 ID NN 的 body-part prototype，dropped 区预测 support。
3. 重训 continued-pretrain（3 backbone：student + EMA teacher + frozen SOLIDER anchor）→ fine-tune 再验 kill-switch（≥+0.7 vs plain）。

## Stage-B 重训 + fine-tune（防遗忘 v2，2026-06-27）

- **continued-pretrain 50ep 健康**：防遗忘 sol_p 0.6→0.11 / sol_g 0.05→0.03 活跃，tok_std 不坍缩，L 收敛。codex 三审（R1 抓"只锚 5 part token 覆盖窄"→补 global GAP distillation→R2/R3 approve）。
- **★fine-tune early signal（epoch 10）**：Stage-B **23.0%** vs Stage-A 15.9%（防遗忘 **+7.1**，机制部分生效）vs plain 33.1%（仍 **−10.1**，没完全修）。
- **诚实判读**：防遗忘 anchor（part + global GAP）**减轻 forgetting 但不充分**——continued-pretrain 仍损害判别性。完全符合 codex 守的诚实"修复尝试成立非 forgetting 已解决"。
- 可能原因：anchor 权重不够 / part+global GAP 还不够（codex 提 dense/stage-wise distill）/ partial-view JEPA 与 ReID 判别性本质张力。
- 待：epoch 50 趋势（追平 plain or 平台卡 < plain）→ final 判 kill-switch。趋势平行 plain（差固定 −10）=防遗忘不够；收敛追平=够。

## ★PSC-JEPA continued-pretrain 主范式判死（2026-06-27，codex 诊断 8/10）

**kill-switch FAIL**：Stage-B 防遗忘 fine-tune 趋势 23→36.7→44→46.2→46.7（epoch 10-50 平台）<< plain 58.5（−11.5 平台差）。防遗忘缩 early gap（−10→−2）但**没改平台**。

**codex 诊断（非代码问题，范式本质冲突）**：
1. partial-view JEPA 补"**不可观测身份细节**"（被遮 part 真实 identity 单图不存在）→ 学生学上下文均值/人体先验 → 特征推向"可预测平滑不变"，而 ReID 要"细粒度可区分"（纹理/颜色/局部差异）。**目标方向冲突**。
2. continued-pretrain 覆盖 SOLIDER 已调好的 appearance/semantic 平衡，anchor 拉不回全部 dense feature/attention/层间几何/margin。
3. 防遗忘越强 JEPA 越 no-op（终点≈原 SOLIDER 不超 plain）。
- 外部先例支持：TransReID-SSL(Catastrophic Forgetting Score) / PersonMAE/HAP(需完整预训练体系) / continued-pretrain 普遍报 forgetting。

**结论：不再 pretrain backbone**。尸检 4-run mini grid（救回 2/10，跳过）。
**Pivot（codex）**：① support bank 改 fine-tune 侧 **detached auxiliary**（5.5，plain SOLIDER fine-tune + support 只监督轻量 completion head/part branch/pair scorer，global backbone hard guard ≥plain−0.3，避 pretrain harm 保 exp109 信号）② **生成数据引擎**（6.5，扩训练分布让监督 fine-tune 自学遮挡鲁棒，不蒸不可观测 support，更长期范式空间）。

exec
/bin/zsh -lc "sed -n '1,180p' experiments/exp365_wildlife/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp365 Wildlife species-conditioned — monitor

## GiraffeZebraID cheap kill-switch（frozen MegaDescriptor-L-384, 零训练, 2026-06-28）

数据：GiraffeZebraID（lila, 非 kaggle, 9.7G）6925 图 / 2056 个体 / 2 物种（zebra_plains 6286 + giraffe_masai 639）。split q=1142 g=4869（每 id 第1张 query 其余 gallery）。本地 MPS 抽 MegaDescriptor 特征(1536维, bbox crop)。

| 测量 | 值 | 判定 |
|---|---|---|
| baseline all-species mAP | 70.95 (R1 78.02) | |
| **wrong-species in top10** | **0.001** | MegaDescriptor 已把 species 分开 |
| per-species centering oracle | **+0.15** | <+3 → Kill |
| same-species vs all-species (species 干扰) | **+0.05** | <+1 → Kill |
| false in top10 | 0.712 | 错误多但都同物种内 |

**VERDICT DEAD**：codex Kill 线命中——MegaDescriptor（动物 ReID SOTA）frozen 已 species-agnostic 强（wrong-species 0.001），per-species centering 没空间（+0.15），species 干扰不存在（+0.05）。

## ★核心发现

Wildlife ReID 真难点 = **同物种内细粒度个体区分**（false_top10=0.712 都是 same-species hard negatives，wrong-species 仅 0.001），**不是 species 干扰**。codex 的 species-conditioning（SCREA: Species-Conditioned Residual Evidence Adapter）**解决错方向**——species 已被 MegaDescriptor 分开，需要解决的是同物种内细粒度区分，那是标准 ReID 问题，无 species-conditioning 新机制。

## 局限（诚实标注）

GiraffeZebraID 只 2 物种（zebra/giraffe 差异大，species 本来好分 → wrong-species 0.001 符合预期）。近缘多物种（WildlifeReID-10k 10k 个体含近缘 species，kaggle 需 key 没有）的 species 干扰没验。**但即使近缘物种 species 干扰更大，真难点（同物种内细粒度）species-conditioning 也解决不了**——core argument 不依赖物种数。

## 决定

species-conditioning 方向偏死（不补 SCREA adapter rank/head/loss 小变体，codex 明确）。换量级 codex 给的方向（第一名 lattice real-data 偏 test-time + 第二名 Wildlife species-conditioning cheap probe 偏死）都偏弱。LM-ReID 6.5（现有最强 B 类，exp359 文档全）是兜底。

下一步：① 可求用户 kaggle key 严谨验 WildlifeReID-10k 近缘多物种（确认 species-conditioning 死，但 core argument 已指向同物种内细粒度）；② 或转新方向（codex 再调研换量级）。cheap kill-switch 价值：零训练几分钟（GiraffeZebraID lila）证 species-conditioning 偏死，没下 kaggle 全量 24.7G 才发现。

## local-verifier cheap probe（codex 建议最后一搏, LoFTR rerank, 2026-06-28）

针对真难点（同物种内细粒度 hard neg），验 LoFTR local matching 能否纠正 MegaDescriptor 的 same-species false。180 query（top-k 内有正样本的）采样：

| | false_top10 | R1(in-topk) |
|---|---|---|
| baseline (MegaDescriptor) | 0.677 | 90.56 |
| LoFTR rerank | 0.677 | **97.22** |
| Δ | +0.000 | **+6.67** |

**★脚本 verdict bug 诚实纠正**：脚本 GO 条件用了 false_top10<0.60，但 **false_top10 对 rerank 无意义**（rerank 只重排 top-k 内顺序，集合不变 → false 比例必然 +0.000），导致误判 DEAD。**真信号 = ΔR1(in-topk)=+6.67**（LoFTR 把 top-k 内正样本提到 top1，远超 codex Top1 +1~2 标准）→ **local verifier 实际有效**：LoFTR local matching 在 same-species hard neg 上有 MegaDescriptor 没用好的判别信息。

**但诚实面对**（为何仍不主推）：
1. upper-bound（LoFTR expensive 双图 matching），训练端要蒸馏成 pattern-token 单次前向。
2. **撞 WildFusion**（arxiv 2408.12934：MegaDescriptor + LoFTR/LightGlue 融合，17 数据集 84.0% +8.5pp）—— local-feature 纠正 same-species 这条 codex 明确"先例太近，主创新风险高"。
3. codex 主线诚实判：LM-ReID 6.5 现实最强 B 类，收尾投稿；Wildlife 即使工程增益也撞 WildFusion。

## ★最终决定（2026-06-28）

收 LM-ReID 6.5（codex 判现实最强 B 类）。Wildlife species-conditioning 死 + 真难点（同物种内细粒度）local-verifier upper-bound 有（ΔR1+6.67）但撞 WildFusion → **记 future work，不主推**。换量级三方向（AG/DG/Wildlife）cheap probe 全证伪/偏弱，LM-ReID 6.5 兜底。教训：local-verifier 脚本 false_top10 指标对 rerank 无意义（栽了指标设计坑），ΔR1 才是真信号 —— 差点被错误 verdict 埋没真信号。

web search: 
web search: person re-identification foundation model DINOv2 CLIP large-scale dataset LUPerson self-supervised 2024 2025
web search: 
web search: LUPerson dataset person re-identification self-supervised large scale 4 million images
web search: 
web search: LLaVA-ReID ChatReID Inter-ReID person re-identification 2024 2025
web search: 
web search: generative retrieval person re-identification discrete code index 2024 2025
web search: 
web search: LaST person re-identification dataset clothes changing 10862 identities 228k images public
web search: 
web search: gait recognition foundation model large dataset GREW Gait3D SUSTech1K person retrieval 2024
web search: 
web search: InternVideo2 foundation model video retrieval 2024 open source
web search: 
web search: PersonPath22 dataset person re-identification tracklet path public 2022 2024
web search: 
web search: Ego4D person retrieval re-identification benchmark dataset person-centric search
web search: 
web search: Segment Anything person re-identification SAM ReID 2024 2025
web search: 
web search: event based person re-identification dataset Event-ReID 2024 public
web search: 
web search: LaST person ReID GitHub dataset 10862 identities 228k images public
web search: 
web search: person search foundation model DINOv2 SAM CLIP 2024 2025 CUHK-SYSU PRW
web search: 
web search: visible infrared person re-identification dataset HITSZ-VCM BUPT VCM 2024 large scale
web search: 
web search: RandPerson UnrealPerson SynPerson synthetic person re-identification dataset 2024 diffusion
web search: 
web search: MovieNet person search dataset character search body face person retrieval 2024
web search: 
web search: https://arxiv.org/abs/2410.13567
web search: 
web search: OmniReID Instruct-ReID dataset 2024 person re-identification instruction large scale
web search: 
web search: DINOv2 learning robust visual features without supervision arxiv 2023 2024
web search: 
web search: DINOv3 Meta AI August 2025 official release ViT person reid
web search: 
web search: Event-AHU Neuromorphic_ReID GitHub EvReID dataset release
web search: 
web search: 'MovieNet-PS' in https://arxiv.org/abs/2112.02500
web search: 
web search: MovieNet-PS dataset number identities images person search
web search: 
web search: dense correspondence local matching person re-identification LoFTR ReID 2024
codex
结论：这次真要冲 8-9，不该再从 SOLIDER/Occluded-Duke 内部挖模块。Top5 应该押“换任务 + 换数据规模 + foundation dense/video/multimodal prior”。我不会把 LM-ReID 训练版、active evidence、single-support CVaR、camera-transport、AG/DG foundation-preserving 放进 Top5，它们要么已被本地证伪，要么被你明确禁掉。

**1. Wardrobe-Causal Long-Term ReID**
机制一句：用真实/合成换装干预训练“身份核心证据”和“服装残留证据”的可控分解，而不是简单丢弃衣服特征。  
为何 8-9：衣服变化是标准短期 ReID 的真实任务迁移，不是 Occluded-Duke 小修；CCUP 有 6000 ID、约 118 万图、100 camera、平均 26.5 outfits/ID，DLCR 又把多套 CC-ReID 数据扩到 210 万级，数据规模足够支撑方法稿。  
换什么：backbone 用 `DINOv3-L/SigLIP2-L` dense feature；数据用 `CCUP + DLCR generated PRCC/CCVID/LaST/VC-Clothes/LTCC + DeepChange + LaST`；task 改成 long-term / clothes-changing ReID。  
Cheap kill-switch：冻结 DINOv3/SigLIP2，按 query 的 `#false-in-top10` 分桶，比较 same-clothes、cross-clothes、outfit-oracle mask/parse 后的 AP；若 cross-clothes hard bucket 有 ≥5 mAP oracle headroom 且不是由 `#false-in-topk` 解释，GO。  
先例风险：CAL/FRD/CCUP/DLCR 已很近，不能写成“换装增强”；必须写成 clothing intervention 下的 evidence factorization。参考：CCUP、DLCR、DeepChange、LaST。  
CCF-B：**8.6**

**2. Gait-Appearance Evidence Routing for Video ReID**
机制一句：视频里同时学习 appearance token 和 phase-aligned gait token，训练一个证据路由器决定何时信衣服、何时信步态。  
为何 8-9：这是从 image crop ReID 换到 video/person biometric retrieval；GREW 26K ID/128K sequences/233K distractors，MEVID 有 10M frames、33 views、clothing changes，头寸比 Market/Occluded-Duke 大得多。  
换什么：`InternVideo2` 或 `DINOv3 frame encoder + OpenGait/GaitBase-style silhouette branch`；数据 `GREW + Gait3D + MEVID + CCVID`；task 为 video ReID / clothes-changing video ReID。  
Cheap kill-switch：冻结 appearance/gait 特征，按 `#false-in-top10`、tracklet length、clothes-change 标记分桶；若 gait branch 在高 false bucket 降低 false-top1/false-top10，并带来 ≥5 mAP/R1 hard-bucket gain，GO。  
先例风险：RealGait、SMPLGait、ASGL、DinoGRL 已占“gait helps ReID”；要避 SMPL，主张应收窄为 foundation video + evidence routing，而非又一个 gait branch。  
CCF-B：**8.4**

**3. Target-Bound Person Search with SAM2/DINOv3**
机制一句：在 uncropped 多人场景中，先用 query-conditioned target-binding token 绑定目标实例/mask，再学习 ReID descriptor。  
为何 8-9：直接跳出 clean bbox crop，变成 person search / character search；MovieNet 有 1100 movies 和 110 万 character boxes，MovieNet-PS/PRW/CUHK-SYSU 能支撑 full-image 检索问题。  
换什么：`SAM2 + DINOv3/DEIMv2/GroundingDINO-style detector`；数据 `MovieNet-PS + MovieNet + PRW + CUHK-SYSU`；task 改成 person search / character search in the wild。  
Cheap kill-switch：冻结 detector/SAM2/DINOv3，对比 bbox crop、oracle mask、query-conditioned mask；每 query 控 `#false-in-topk` 和同帧人数，若 target-binding oracle 在 high-crowd bucket 有 ≥5 mAP 或 false-top10 降 ≥20%，GO。  
先例风险：PSTR、GLCNet、KPR 已占 person search/context/promptable ReID；不能写 scene context trick，只能写“target binding before identity embedding”。  
CCF-B：**8.2**

**4. RGB-Event Person ReID Foundation Fusion**
机制一句：把 event stream 当低光/高速/隐私场景下的身份边缘与运动证据，训练 RGB-event reciprocal distillation 和 reliability gate。  
为何 8-9：EvReID 是新 RGB-event ReID benchmark，约 1200 ID、118988 RGB-event pairs，且 event ReID 还没被 foundation 模型充分吃掉。  
换什么：RGB 侧 `DINOv3/SigLIP2`，event 侧 event transformer/spiking ViT；数据 `EvReID`，可补 `SYSU-MM01/RegDB/LLCM` 做跨模态外延；task 为 RGB-event / privacy-preserving ReID。  
Cheap kill-switch：冻结 RGB-only、event-only、naive fusion，按 lighting/motion/event sparsity 和 `#false-in-top10` 分桶；若 event 在低光/运动 hard bucket 降 false-top10 ≥15% 且 +3 mAP 以上，GO。  
先例风险：SFE-Net、TriPro-ReID、event anonymization 已存在；新意必须是 foundation reciprocal distillation + reliability，不是 attribute-guided contrastive。  
CCF-B：**8.1**

**5. Dense Correspondence Distillation ReID**
机制一句：训练时用 DINOv3/LoFTR/SAM2 的离线 dense correspondence 作为 hard-pair teacher，把局部匹配/不匹配证据蒸进单次前向 descriptor。  
为何 8-9：它直接打 top-k false positives，而不是平均 mAP 调参；方法仍是训练侧，测试不跑 LoFTR、不做 MaxSim/rerank。  
换什么：teacher 用 `DINOv3 dense features + LoFTR/SAM2 masks`，student 用 DINOv3/ViT ReID head；数据 `LaST + DeepChange + MSMT17 + Market + MovieNet-PS`；task 为 hard-negative person retrieval/person search。  
Cheap kill-switch：在 frozen baseline 的 top20 上跑 dense matcher 只做诊断，按原始 `#false-in-top10` 分桶；若 dense oracle 在高 false bucket +5 mAP 或 +10 R1，并且正样本本就在 top20，说明可蒸馏，GO。  
先例风险：LoFTR、早期 correspondence-structure ReID、局部 part matching 都相关；边界是“pairwise dense teacher 只训练用，测试单 descriptor”。  
CCF-B：**8.0**

优先级建议：先做 **#1 和 #2 的 frozen kill-switch**。它们同时换 task、换数据规模、换 foundation，而且不是你列的死区；若 kill-switch 过，才值得开真正训练工程。
tokens used
166,687
结论：这次真要冲 8-9，不该再从 SOLIDER/Occluded-Duke 内部挖模块。Top5 应该押“换任务 + 换数据规模 + foundation dense/video/multimodal prior”。我不会把 LM-ReID 训练版、active evidence、single-support CVaR、camera-transport、AG/DG foundation-preserving 放进 Top5，它们要么已被本地证伪，要么被你明确禁掉。

**1. Wardrobe-Causal Long-Term ReID**
机制一句：用真实/合成换装干预训练“身份核心证据”和“服装残留证据”的可控分解，而不是简单丢弃衣服特征。  
为何 8-9：衣服变化是标准短期 ReID 的真实任务迁移，不是 Occluded-Duke 小修；CCUP 有 6000 ID、约 118 万图、100 camera、平均 26.5 outfits/ID，DLCR 又把多套 CC-ReID 数据扩到 210 万级，数据规模足够支撑方法稿。  
换什么：backbone 用 `DINOv3-L/SigLIP2-L` dense feature；数据用 `CCUP + DLCR generated PRCC/CCVID/LaST/VC-Clothes/LTCC + DeepChange + LaST`；task 改成 long-term / clothes-changing ReID。  
Cheap kill-switch：冻结 DINOv3/SigLIP2，按 query 的 `#false-in-top10` 分桶，比较 same-clothes、cross-clothes、outfit-oracle mask/parse 后的 AP；若 cross-clothes hard bucket 有 ≥5 mAP oracle headroom 且不是由 `#false-in-topk` 解释，GO。  
先例风险：CAL/FRD/CCUP/DLCR 已很近，不能写成“换装增强”；必须写成 clothing intervention 下的 evidence factorization。参考：CCUP、DLCR、DeepChange、LaST。  
CCF-B：**8.6**

**2. Gait-Appearance Evidence Routing for Video ReID**
机制一句：视频里同时学习 appearance token 和 phase-aligned gait token，训练一个证据路由器决定何时信衣服、何时信步态。  
为何 8-9：这是从 image crop ReID 换到 video/person biometric retrieval；GREW 26K ID/128K sequences/233K distractors，MEVID 有 10M frames、33 views、clothing changes，头寸比 Market/Occluded-Duke 大得多。  
换什么：`InternVideo2` 或 `DINOv3 frame encoder + OpenGait/GaitBase-style silhouette branch`；数据 `GREW + Gait3D + MEVID + CCVID`；task 为 video ReID / clothes-changing video ReID。  
Cheap kill-switch：冻结 appearance/gait 特征，按 `#false-in-top10`、tracklet length、clothes-change 标记分桶；若 gait branch 在高 false bucket 降低 false-top1/false-top10，并带来 ≥5 mAP/R1 hard-bucket gain，GO。  
先例风险：RealGait、SMPLGait、ASGL、DinoGRL 已占“gait helps ReID”；要避 SMPL，主张应收窄为 foundation video + evidence routing，而非又一个 gait branch。  
CCF-B：**8.4**

**3. Target-Bound Person Search with SAM2/DINOv3**
机制一句：在 uncropped 多人场景中，先用 query-conditioned target-binding token 绑定目标实例/mask，再学习 ReID descriptor。  
为何 8-9：直接跳出 clean bbox crop，变成 person search / character search；MovieNet 有 1100 movies 和 110 万 character boxes，MovieNet-PS/PRW/CUHK-SYSU 能支撑 full-image 检索问题。  
换什么：`SAM2 + DINOv3/DEIMv2/GroundingDINO-style detector`；数据 `MovieNet-PS + MovieNet + PRW + CUHK-SYSU`；task 改成 person search / character search in the wild。  
Cheap kill-switch：冻结 detector/SAM2/DINOv3，对比 bbox crop、oracle mask、query-conditioned mask；每 query 控 `#false-in-topk` 和同帧人数，若 target-binding oracle 在 high-crowd bucket 有 ≥5 mAP 或 false-top10 降 ≥20%，GO。  
先例风险：PSTR、GLCNet、KPR 已占 person search/context/promptable ReID；不能写 scene context trick，只能写“target binding before identity embedding”。  
CCF-B：**8.2**

**4. RGB-Event Person ReID Foundation Fusion**
机制一句：把 event stream 当低光/高速/隐私场景下的身份边缘与运动证据，训练 RGB-event reciprocal distillation 和 reliability gate。  
为何 8-9：EvReID 是新 RGB-event ReID benchmark，约 1200 ID、118988 RGB-event pairs，且 event ReID 还没被 foundation 模型充分吃掉。  
换什么：RGB 侧 `DINOv3/SigLIP2`，event 侧 event transformer/spiking ViT；数据 `EvReID`，可补 `SYSU-MM01/RegDB/LLCM` 做跨模态外延；task 为 RGB-event / privacy-preserving ReID。  
Cheap kill-switch：冻结 RGB-only、event-only、naive fusion，按 lighting/motion/event sparsity 和 `#false-in-top10` 分桶；若 event 在低光/运动 hard bucket 降 false-top10 ≥15% 且 +3 mAP 以上，GO。  
先例风险：SFE-Net、TriPro-ReID、event anonymization 已存在；新意必须是 foundation reciprocal distillation + reliability，不是 attribute-guided contrastive。  
CCF-B：**8.1**

**5. Dense Correspondence Distillation ReID**
机制一句：训练时用 DINOv3/LoFTR/SAM2 的离线 dense correspondence 作为 hard-pair teacher，把局部匹配/不匹配证据蒸进单次前向 descriptor。  
为何 8-9：它直接打 top-k false positives，而不是平均 mAP 调参；方法仍是训练侧，测试不跑 LoFTR、不做 MaxSim/rerank。  
换什么：teacher 用 `DINOv3 dense features + LoFTR/SAM2 masks`，student 用 DINOv3/ViT ReID head；数据 `LaST + DeepChange + MSMT17 + Market + MovieNet-PS`；task 为 hard-negative person retrieval/person search。  
Cheap kill-switch：在 frozen baseline 的 top20 上跑 dense matcher 只做诊断，按原始 `#false-in-top10` 分桶；若 dense oracle 在高 false bucket +5 mAP 或 +10 R1，并且正样本本就在 top20，说明可蒸馏，GO。  
先例风险：LoFTR、早期 correspondence-structure ReID、局部 part matching 都相关；边界是“pairwise dense teacher 只训练用，测试单 descriptor”。  
CCF-B：**8.0**

优先级建议：先做 **#1 和 #2 的 frozen kill-switch**。它们同时换 task、换数据规模、换 foundation，而且不是你列的死区；若 kill-switch 过，才值得开真正训练工程。
