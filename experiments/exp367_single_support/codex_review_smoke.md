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
session id: 019f0ac3-f069-78c2-b3ee-586e8509a89d
--------
user
审查 experiments/exp367_single_support/cvpb_ss_cvar_smoke.py(Single-Support CVaR frozen-head smoke 训练, codex 设计的 two-level CVaR)。逐行查 bug + 训练逻辑。审: (a)episodic single-support CVaR loss 对吗——episode N id×K 图, 每 support-slot 当 prototype 其余 query 分类到 N proto 的 CE, per-id K 个 support risk 做 CVaR_α worst tail, L=mean_CE+lam·CVaR? (b)和 codex 设计(two-level: support risk→CVaR per id)一致吗? (c)random mode(无 CVaR)对照对吗? (d)评估 projected query/gallery→full/random/worst-support 对吗(复用 probe v2 口径)? (e)会退化普通 hard-mining 吗? frozen head(init eye)+backbone 冻对吗? loss 数值稳定(topk/CE shape)? verdict approve/needs-attention + 简短理由。务实中文。
codex
我先按仓库规则补齐当前研究上下文，然后直接看 `exp367_single_support` 的设计和 smoke 脚本。重点会放在 loss 形状、episode 采样、CVaR 定义、random 对照和 eval 口径是否真的对应设计。
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
/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID
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

exec
/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID
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

codex
上下文看完，最新主线已经转到 single-support / 不完整证据问题。下面我会把 `design.md` 和脚本按行号对齐看，必要时查它调用的 probe/eval 口径，避免只凭函数名判断。
exec
/bin/zsh -lc "nl -ba experiments/exp367_single_support/cvpb_ss_cvar_smoke.py | sed -n '1,260p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Single-Support CVaR frozen-head smoke — codex cheap 路径 #1 (不动 backbone).
     3	
     4	冻 backbone, 只训 projection head, 用 episodic single-support CVaR loss。验机制方向:
     5	worst/random-support 是否提升? codex 明确: frozen 失败不判死(可能要改 backbone), 只 smoke。
     6	成功线: worst 或 random-support +0.8~1.0 mAP, full-gallery 掉 <0.5。
     7	对照(防退化 hard-mining): --mode random (episodic CE 无 CVaR), 证不是 episode 本身涨。
     8	
     9	训练设计(codex two-level CVaR):
    10	  episode N id × K 图; 每 id 枚举 K 个候选 support, 算该 support 当 prototype 时同 id query 的 CE risk;
    11	  对每 id 的 K 个 support risk 做 CVaR_α(worst tail); L = mean_CE + lam·mean_CVaR。
    12	  (加项不替换主任务; frozen smoke 先只 CE+CVaR, full FT 再加 triplet。)
    13	
    14	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_ss_cvar_smoke.py
    15	"""
    16	import os, sys, argparse
    17	import numpy as np, torch, torch.nn.functional as F
    18	from collections import defaultdict
    19	
    20	ap = argparse.ArgumentParser()
    21	ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
    22	ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
    23	ap.add_argument('--eval_cache', default='/tmp/ae_feats.npz')          # query/gallery 特征(复用)
    24	ap.add_argument('--train_cache', default='/tmp/ss_train_feats.npz')
    25	ap.add_argument('--epochs', type=int, default=20)
    26	ap.add_argument('--N', type=int, default=16)        # ids per episode
    27	ap.add_argument('--K', type=int, default=4)         # imgs per id
    28	ap.add_argument('--alpha', type=float, default=0.7) # CVaR tail
    29	ap.add_argument('--lam', type=float, default=0.3)   # CVaR weight
    30	ap.add_argument('--tau', type=float, default=0.1)   # softmax temp
    31	ap.add_argument('--mode', default='cvar', choices=['cvar', 'random'])
    32	cli = ap.parse_args()
    33	sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data', '--K', '1',
    34	            '--reuse_gallery', '--cache_gallery', '/tmp/ss_g.npz']
    35	sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments', 'cargo_cvpb'))
    36	import cvpb_lattice_killswitch as ks
    37	from datasets.bases import read_image
    38	REPO = ks._repo; DEV = 'cuda'
    39	
    40	# ---- 1. 抽 train 特征 (frozen) ----
    41	if os.path.exists(cli.train_cache):
    42	    z = np.load(cli.train_cache); tf, tp = z['tf'], z['tp']
    43	    print(f'[train feat] cached {tf.shape}', flush=True)
    44	else:
    45	    ext = ks.FrozenExtractor()
    46	    its = ks.list_split(os.path.join(REPO, 'data', 'market1501', 'bounding_box_train'))
    47	    pils = [ks._to_target_aspect(read_image(it[0])) for it in its]
    48	    tf = ext.feats_from_pil(pils).astype(np.float32); tf /= np.linalg.norm(tf, axis=1, keepdims=True) + 1e-9
    49	    tp = np.array([it[1] for it in its])
    50	    np.savez(cli.train_cache, tf=tf, tp=tp); print(f'[train feat] extracted {tf.shape}', flush=True)
    51	D = tf.shape[1]
    52	ft = torch.tensor(tf, device=DEV); yt = torch.tensor(tp, device=DEV)
    53	id2idx = defaultdict(list)
    54	for i, p in enumerate(tp): id2idx[int(p)].append(i)
    55	ids = [p for p in id2idx if len(id2idx[p]) >= cli.K]
    56	print(f'[ss-cvar smoke] mode={cli.mode} train-ids={len(ids)} D={D}', flush=True)
    57	
    58	# ---- 2. projection head (Linear+BN, init eye) ----
    59	head = torch.nn.Sequential(torch.nn.Linear(D, D, bias=False), torch.nn.BatchNorm1d(D)).to(DEV)
    60	head[0].weight.data.copy_(torch.eye(D))
    61	opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    62	
    63	
    64	def episode_loss():
    65	    bids = np.random.choice(ids, min(cli.N, len(ids)), replace=False)
    66	    idxs = np.stack([np.random.choice(id2idx[int(y)], cli.K, replace=False) for y in bids])  # [N,K]
    67	    z = F.normalize(head(ft[idxs.reshape(-1)]), dim=1).reshape(len(bids), cli.K, D)            # [N,K,D]
    68	    # 对每个 support-slot s: 用每 id 的第 s 图当 prototype(N 个), 其余 K-1 当 query, 分类到 N prototypes
    69	    id_risks = []                                   # [N] each id 的 CVaR over K support
    70	    ce_terms = []
    71	    for s in range(cli.K):
    72	        proto = z[:, s]                             # [N,D] 每 id 的 support-s 当 prototype
    73	        q_slots = [j for j in range(cli.K) if j != s]
    74	        qz = z[:, q_slots].reshape(len(bids), len(q_slots), D)  # [N,K-1,D]
    75	        logit = torch.einsum('nqd,md->nqm', qz, proto) / cli.tau   # [N,K-1,N] query vs N proto
    76	        tgt = torch.arange(len(bids), device=DEV)[:, None].expand(-1, len(q_slots))  # 正样本=自己 id
    77	        ce = F.cross_entropy(logit.reshape(-1, len(bids)), tgt.reshape(-1), reduction='none').reshape(len(bids), -1)
    78	        ce_terms.append(ce.mean(1))                 # [N] 该 support-s 对每 id 的 risk
    79	    risks = torch.stack(ce_terms, dim=1)            # [N, K] 每 id 的 K 个 support risk
    80	    l_ce = risks.mean()
    81	    if cli.mode == 'random':
    82	        return l_ce                                  # 对照: 无 CVaR
    83	    # CVaR_alpha over K support risks per id (worst tail)
    84	    k_tail = max(1, int(np.ceil((1 - cli.alpha) * cli.K)))
    85	    cvar = torch.topk(risks, k_tail, dim=1).values.mean()
    86	    return l_ce + cli.lam * cvar
    87	
    88	
    89	for ep in range(cli.epochs):
    90	    head.train(); last = 0.0
    91	    for _ in range(max(1, len(ids) // cli.N)):
    92	        loss = episode_loss()
    93	        opt.zero_grad(); loss.backward(); opt.step(); last = float(loss.item())
    94	    if ep % 5 == 0 or ep == cli.epochs - 1: print(f'  ep{ep} loss={last:.4f}', flush=True)
    95	
    96	# ---- 3. 评估: projected query/gallery → full / single-support diagnostic ----
    97	head.eval()
    98	z = np.load(cli.eval_cache)
    99	qf, qp, qc, gf, gp, gc = z['qf'], z['qp'], z['qc'], z['gf'], z['gp'], z['gc']
   100	with torch.no_grad():
   101	    qf = F.normalize(head(torch.tensor(qf, device=DEV)), dim=1).cpu().numpy()
   102	    gf = F.normalize(head(torch.tensor(gf, device=DEV)), dim=1).cpu().numpy()
   103	
   104	
   105	def eval_fixed(g_idx, valid_q):
   106	    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
   107	    aps = []
   108	    for i in valid_q:
   109	        s = qf[i] @ gff.T; keep = ~((gpp == qp[i]) & (gcc == qc[i]))
   110	        ss = s[keep]; gpk = gpp[keep]; o = np.argsort(-ss); m = (gpk[o] == qp[i])
   111	        aps.append((np.cumsum(m)[m] / np.arange(1, len(m)+1)[m]).mean() if m.any() else 0.0)
   112	    return 100*np.mean(aps)
   113	
   114	
   115	id2g = defaultdict(list)
   116	for i, p in enumerate(gp): id2g[p].append(i)
   117	q_ids = set(qp.tolist())
   118	distractor_g = np.array([i for p in id2g if p not in q_ids for i in id2g[p]], dtype=int)
   119	valid_q = np.array([i for i in range(len(qf)) if (gp[~((gp == qp[i]) & (gc == qc[i]))] == qp[i]).any()])
   120	
   121	
   122	def supp_g(sidx): return np.concatenate([np.array(sidx, dtype=int), distractor_g])
   123	full_mAP = eval_fixed(np.arange(len(gf)), valid_q)
   124	hasq = [p for p in id2g if p in q_ids]
   125	rand_mAPs = []
   126	for sd in range(10):
   127	    rng = np.random.RandomState(sd); rand_mAPs.append(eval_fixed(supp_g([rng.choice(id2g[p]) for p in hasq]), valid_q))
   128	worst_s = []
   129	for p in hasq:
   130	    gi = id2g[p]; qs = np.where(qp == p)[0]
   131	    qual = [(qf[qs[qc[qs] != gc[g]]] @ gf[g]).mean() if (qc[qs] != gc[g]).any() else -1 for g in gi]
   132	    worst_s.append(gi[int(np.argmin(qual))])
   133	worst_mAP = eval_fixed(supp_g(worst_s), valid_q)
   134	print(f'\n[SS-CVAR SMOKE RESULT mode={cli.mode}]')
   135	print(f'  full-gallery   : mAP={full_mAP:.2f}')
   136	print(f'  random-support : mAP={np.mean(rand_mAPs):.2f}±{np.std(rand_mAPs):.2f}')
   137	print(f'  worst-support  : mAP={worst_mAP:.2f}')
   138	print(f'  ※ 与 frozen baseline(probe v2: full 94.43 / random 73.36 / worst 63.82) 比, 看 head 是否抬 worst/random', flush=True)
   139	print('[done]', flush=True)

exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
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

exec
/bin/zsh -lc 'rg --files experiments/exp367_single_support' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/design.md
experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review_smoke.md
experiments/exp367_single_support/monitor.md
experiments/exp367_single_support/codex_train_design.md
experiments/exp367_single_support/codex_review2.md
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py

exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
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

codex 调研 Single-Support CVaR 训练设计 + novelty 确认（63517）：episodic 单图 support + CVaR worst-case 如何写 loss、避六点定律陷阱（不塑造/对齐/压缩变体）、cheap 验证路径（frozen head 小训 or full FT）。GO 则训练侧创新动手，full fine-tune 前 codex 三审 diff。

codex
脚本主体不长，关键风险集中在两个地方：训练 loss 的“support risk”是不是按设计定义，以及评估是否复用了 probe v2 的 common-valid/distractor 口径。我现在对照 probe v2 的实现逐项比。
exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/codex_review_smoke.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
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
session id: 019f0ac3-f069-78c2-b3ee-586e8509a89d
--------
user
审查 experiments/exp367_single_support/cvpb_ss_cvar_smoke.py(Single-Support CVaR frozen-head smoke 训练, codex 设计的 two-level CVaR)。逐行查 bug + 训练逻辑。审: (a)episodic single-support CVaR loss 对吗——episode N id×K 图, 每 support-slot 当 prototype 其余 query 分类到 N proto 的 CE, per-id K 个 support risk 做 CVaR_α worst tail, L=mean_CE+lam·CVaR? (b)和 codex 设计(two-level: support risk→CVaR per id)一致吗? (c)random mode(无 CVaR)对照对吗? (d)评估 projected query/gallery→full/random/worst-support 对吗(复用 probe v2 口径)? (e)会退化普通 hard-mining 吗? frozen head(init eye)+backbone 冻对吗? loss 数值稳定(topk/CE shape)? verdict approve/needs-attention + 简短理由。务实中文。
codex
我先按仓库规则补齐当前研究上下文，然后直接看 `exp367_single_support` 的设计和 smoke 脚本。重点会放在 loss 形状、episode 采样、CVaR 定义、random 对照和 eval 口径是否真的对应设计。
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

exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/codex_train_design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
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
session id: 019f0aaa-519e-73c2-b19b-4cbd670effd5
--------
user
Single-Support CVaR ReID 训练侧创新, cheap probe 已 GO(frozen SOLIDER: support 选择 oracle headroom best-worst 12.27 / random-worst 9.54, false10 best≈worst 不被 #false-in-topk 解释; best/worst 用 query-label oracle 上下界)。现在要**设计训练**(把 oracle headroom 学成不用 query 的鲁棒表征)。**深度调研+设计**: (a)**novelty 严格确认**: 'episodic single-support training + CVaR worst-support 优化' for standard person ReID, 2024-26 web search 有直接先例吗? few-shot ReID(CFReID/ReMix)/episodic metric learning/DRO-ReID/hard-example mining 区别在哪? 这个角度真空白吗还是撞 episodic few-shot? (b)**训练设计**: 怎么把'每 ID 单图 support 定义身份 + worst-case 鲁棒'写成 loss? episodic batch 采样(每 ID 1 support + 多 query)? CVaR over worst support-query pairs? 和标准 triplet+CE 怎么结合(加项还是替换)? 避六点定律陷阱(不塑造/对齐/压缩变体)? (c)**cheap 验证路径**: 先 frozen backbone + 单 support episodic head 小训(不动 backbone)能否提升 worst-support? 还是必须 full fine-tune? 给最小训练验证(1 单卡半天)。(d)风险: 会不会退化成普通 hard-mining / 普通 single-shot eval? 给设计+novelty 判定+CCF-B 潜力分。务实中文严谨。
codex
我先按仓库接手规则读当前实验与 story，再做 2024-26 的论文/代码检索。这个请求主要是研究判定和训练设计，我会把“是否撞车”和“怎么最小验证”分开给结论。
exec
/bin/zsh -lc "sed -n '1,260p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,240p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,260p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,260p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID
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

exec
/bin/zsh -lc "nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py | sed -n '1,260p'" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/codex_review2.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Single-Support ReID — cheap kill-switch (零训练) — v2 (codex needs-attention 修).
     3	
     4	codex 训练侧 #1: 训练时每 ID 单图 support 定义身份, CVaR worst-support 优化。回应 exp109
     5	根问题(single-image support incomplete)。纯训练侧(episodic loss 输出常规 descriptor)。
     6	
     7	★codex 审 v1 抓 3 个 High, v2 修:
     8	  1. common-valid query mask: 用 full-gallery 下有 positive 的 query 子集, 所有 support 设置同子集(否则比不同难度)。
     9	  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
    10	  3. 主判据 best-random / random-worst 多 seed(20) 均值±std; 报 #false-in-topk(top10 错样本数)。
    11	  4. best/worst 用 query-label oracle(诊断上下界, 不证训练可学, 诚实标注)。
    12	  5. cache provenance: 校验 full-gallery mAP sanity(=exp260b ref 94.4)。
    13	
    14	GO(support 选择是真训练瓶颈): random-worst gap > 3 mAP(多 seed 稳, 同负样本池同 valid query) AND
    15	   #false-in-topk 不被 trivial 解释。DEAD: gap 小或被负样本池/valid-query 变化解释。
    16	
    17	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
    18	"""
    19	import numpy as np
    20	from collections import defaultdict
    21	import argparse
    22	
    23	ap = argparse.ArgumentParser()
    24	ap.add_argument('--cache', default='/tmp/ae_feats.npz')
    25	ap.add_argument('--seeds', type=int, default=20)
    26	cli = ap.parse_args()
    27	
    28	import os
    29	if not os.path.exists(cli.cache):
    30	    raise SystemExit(f'[FATAL] cache {cli.cache} 不存在, 先跑 active_evidence probe 生成 Market 特征 cache')
    31	z = np.load(cli.cache)
    32	qf, qp, qc = z['qf'], z['qp'], z['qc']
    33	gf, gp, gc = z['gf'], z['gp'], z['gc']
    34	assert np.isfinite(qf).all() and np.isfinite(gf).all(), 'feat 含 nan/inf'
    35	print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)
    36	
    37	# pid 分类: has-query ID(在 query 出现) vs distractor(只在 gallery)
    38	q_ids = set(qp.tolist())
    39	id2g = defaultdict(list)
    40	for i, p in enumerate(gp): id2g[p].append(i)
    41	hasq_ids = [p for p in id2g if p in q_ids]
    42	distractor_g = np.array([i for p in id2g if p not in q_ids for i in id2g[p]], dtype=int)
    43	print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
    44	
    45	
    46	def support_gallery(support_idx_per_id):
    47	    """has-query ID 用单 support, distractor 全量 → 负样本池不变。"""
    48	    return np.concatenate([np.array(support_idx_per_id, dtype=int), distractor_g])
    49	
    50	
    51	def eval_fixed(g_idx, valid_q):
    52	    """对固定 valid_q 子集 eval mAP/R1 + #false-in-top10。g_idx=gallery 子集。"""
    53	    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
    54	    aps, r1s, false10 = [], [], []
    55	    for i in valid_q:
    56	        sim_i = qf[i] @ gff.T
    57	        keep = ~((gpp == qp[i]) & (gcc == qc[i]))
    58	        s = sim_i[keep]; gpk = gpp[keep]
    59	        o = np.argsort(-s); m = (gpk[o] == qp[i])
    60	        if not m.any():
    61	            aps.append(0.0); r1s.append(0.0); false10.append(1.0); continue   # missing-positive 记 0(codex)
    62	        cum = np.cumsum(m); r = np.arange(1, len(m) + 1)
    63	        aps.append((cum[m] / r[m]).mean()); r1s.append(float(m[0]))
    64	        false10.append(float((gpk[o[:10]] != qp[i]).mean()))
    65	    return 100*np.mean(aps), 100*np.mean(r1s), np.mean(false10)
    66	
    67	
    68	# common-valid query: full-gallery 下有 positive 的 query (固定子集, 所有 support 设置共用)
    69	full_g = np.arange(len(gf))
    70	valid_q = []
    71	for i in range(len(qf)):
    72	    keep = ~((gp == qp[i]) & (gc == qc[i]))
    73	    if (gp[keep] == qp[i]).any(): valid_q.append(i)
    74	valid_q = np.array(valid_q)
    75	print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
    76	
    77	full_mAP, full_R1, full_f10 = eval_fixed(full_g, valid_q)
    78	print(f'  full-gallery sanity mAP={full_mAP:.2f} (provenance check vs exp260b ref ~94.4)', flush=True)
    79	
    80	# best/worst-support oracle (用 query-label, 诊断上下界 — 诚实: 不证训练可学)
    81	best_s, worst_s = [], []
    82	for p in hasq_ids:
    83	    gidxs = id2g[p]; q_same = np.where(qp == p)[0]
    84	    qual = []
    85	    for g in gidxs:
    86	        qs = q_same[qc[q_same] != gc[g]]
    87	        qual.append((qf[qs] @ gf[g]).mean() if len(qs) else -1.0)
    88	    qual = np.array(qual)
    89	    best_s.append(gidxs[int(np.argmax(qual))]); worst_s.append(gidxs[int(np.argmin(qual))])
    90	best_mAP, best_R1, best_f10 = eval_fixed(support_gallery(best_s), valid_q)
    91	worst_mAP, worst_R1, worst_f10 = eval_fixed(support_gallery(worst_s), valid_q)
    92	
    93	# random-support 多 seed
    94	rand_mAPs = []
    95	for sd in range(cli.seeds):
    96	    rng = np.random.RandomState(sd)
    97	    rs = [rng.choice(id2g[p]) for p in hasq_ids]
    98	    rand_mAPs.append(eval_fixed(support_gallery(rs), valid_q)[0])
    99	rand_mean, rand_std = np.mean(rand_mAPs), np.std(rand_mAPs)
   100	
   101	print(f'\n[SINGLE-SUPPORT v2 RESULT] (common-valid q={len(valid_q)}, distractor 全量, {cli.seeds} seeds)')
   102	print(f'  full-gallery   : mAP={full_mAP:.2f} R1={full_R1:.2f} false10={full_f10:.3f}')
   103	print(f'  best-support   : mAP={best_mAP:.2f} (oracle 上界, 用 query-label)  false10={best_f10:.3f}')
   104	print(f'  random-support : mAP={rand_mean:.2f}±{rand_std:.2f}')
   105	print(f'  worst-support  : mAP={worst_mAP:.2f} (oracle 下界)  false10={worst_f10:.3f}')
   106	print(f'  best-random gap = {best_mAP-rand_mean:.2f}  random-worst gap = {rand_mean-worst_mAP:.2f}  best-worst = {best_mAP-worst_mAP:.2f}')
   107	# 主判据: random-worst gap(同负样本池同valid query, 单support内选择) > 3 且 false10 同向变化(非trivial少正样本)
   108	go = (rand_mean - worst_mAP) > 3.0 and (best_mAP - rand_mean) > 1.0
   109	print(f'  [verdict] {"GO (单 support 内选择 matters, support representation 是真训练瓶颈)" if go else "DEAD (support 选择价值小/被负样本池-valid-query 解释)"}')
   110	print('  ※ best/worst 是 query-label oracle 诊断上下界, 证 headroom 存在; 训练能否学到要 CVaR train 验', flush=True)
   111	print('[done]', flush=True)

 succeeded in 0ms:
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
session id: 019f0aa6-3af5-70d1-8e85-fee19b29bf6c
--------
user
再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
codex
我先按仓库接手规则读最新实验上下文，再聚焦审 `cvpb_single_support_probe.py` 的 v2 逻辑和潜在泄漏点。
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

codex
对照下来，训练部分基本是 two-level CVaR 的骨架，但我看到几个会影响解释强度的点：`random` 对照并不是真“随机 support”训练，只是同一个 episode CE 去掉 CVaR；eval 也只算 mAP，不含 probe v2 的 R1/false10。这些不一定阻塞 smoke，但要在 verdict 里区分“可跑”和“证据不完整”。
exec
/bin/zsh -lc "nl -ba experiments/cargo_cvpb/cvpb_lattice_killswitch.py | sed -n '1,240p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Lattice-Marginalized ReID — ZERO-TRAINING kill-switch  (Market, frozen SOLIDER).
     3	
     4	HYPOTHESIS UNDER TEST (d_8.txt 机会1):
     5	    Low-resolution ReID failure is NOT "blur / missing detail" but SAMPLING-LATTICE
     6	    UNCERTAINTY. A single person at h=16-32px is a *family* of alias/crop-lattice
     7	    observations (sub-pixel sampling phase, +/-1 LR-pixel bbox quantization, antialias
     8	    kernel, slight detector crop error). The model has only ever seen ONE member of that
     9	    family. Identity matching should MARGINALIZE over the family, not treat one
    10	    deterministic LR image as ground truth.
    11	
    12	ZERO-TRAINING (no backward, frozen ckpt + numpy/PIL).  Standard CR-ReID setting:
    13	    HR gallery (enrolled), LR query (far/small).  Gallery features extracted ONCE at HR.
    14	
    15	WHAT WE MEASURE per LR height h in {16,24,32,48}:
    16	    (A) same-image phase feature variance: feed K lattice variants of the SAME hr query,
    17	        measure mean pairwise (1-cos) of their frozen features.  (does the lattice move
    18	        the embedding at all?)
    19	    (B) rank volatility: top1 agreement + top10 Jaccard ACROSS the K lattice variants
    20	        (do retrieved IDs flip between phases?).
    21	    (C) does phase variance EXPLAIN LR false matches?  per-query Spearman(phase-var, AP-error)
    22	        AND -- decisive, Hubness-§7.6 lesson -- partial-Spearman CONTROLLING the trivial
    23	        proxy #false-in-topk (and LR severity).  If phase-var is just a proxy for
    24	        "#wrong-in-topk", it has no independent value.
    25	    (D) ensemble mAP: K-phase feature-mean / MaxSim vs a SINGLE deterministic bicubic LR.
    26	
    27	  ******  THE LIFE/DEATH CONTROLS  ******
    28	    (C1) vs ORDINARY TTA: the SAME K, the SAME fusion (mean / MaxSim), but the K views are
    29	         ordinary test-time augmentation (pad+RandomCrop + hflip) of ONE bicubic LR image
    30	         -- NOT lattice variants.  phase-lattice ensemble MUST clearly beat ordinary-TTA
    31	         ensemble, else it is just TTA renamed.
    32	    (C2) vs #false-in-topk: phase-variance explaining failure MUST survive partialling out
    33	         #false-in-topk (Hubness lesson: a trivial proxy must not silently do the work).
    34	
    35	VERDICT:
    36	    GO   if  h<=32: rank volatility clearly nonzero  AND  phase-ensemble >= +2 mAP over single
    37	         LR  AND that gain CLEARLY exceeds ordinary-TTA  AND phase-var explains failure
    38	         INDEPENDENTLY of #false-in-topk.
    39	    DEAD if  phase variance tiny  /  ensemble ~ single LR  /  ensemble ~ ordinary TTA  /
    40	         phase-var absorbed by #false-in-topk.
    41	
    42	Run on lab-3090-d:
    43	    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
    44	      /root/miniconda3/envs/solider-reid/bin/python \
    45	      experiments/cargo_cvpb/cvpb_lattice_killswitch.py \
    46	      --config configs/market/pose_psg_lgpa_gcn_base.yml \
    47	      --ckpt   log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth \
    48	      --K 9  2>&1 | tee /tmp/cvpb_lattice_market.log
    49	    # smoke first:  --smoke 150 --heights 32   (fast)
    50	"""
    51	import os, sys, time, argparse
    52	import numpy as np
    53	from PIL import Image
    54	
    55	_here = os.path.dirname(os.path.abspath(__file__))
    56	_repo = os.path.abspath(os.path.join(_here, '..', '..'))
    57	sys.path.insert(0, _repo)
    58	
    59	ap = argparse.ArgumentParser()
    60	ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
    61	ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
    62	ap.add_argument('--data_root', default='data')
    63	ap.add_argument('--heights', type=int, nargs='+', default=[16, 24, 32, 48],
    64	                help='LR person heights to test')
    65	ap.add_argument('--K', type=int, default=9, help='#lattice (phase) variants per LR query')
    66	ap.add_argument('--smoke', type=int, default=0, help='cap #query for a fast smoke run')
    67	ap.add_argument('--batch', type=int, default=128)
    68	ap.add_argument('--seed', type=int, default=42)
    69	ap.add_argument('--cache_gallery', default='/tmp/lattice_gallery_hr.npz')
    70	ap.add_argument('--reuse_gallery', action='store_true')
    71	ap.add_argument('--lattice_axis', type=int, default=-1, help='LM-S4: restrict lattice variants to ONE axis (0=phase,1=bbox,2=zoom); -1=all (round-robin)')
    72	ap.add_argument('--strong_tta', action='store_true', help='LM-S2 defense: richer ordinary-TTA (resize-jitter+color) so lattice must beat a STRONG baseline')
    73	ap.add_argument('--jitter_mode', default='lattice', choices=['lattice', 'detector'], help='push-7.0: lattice=uniform +-1 LR-px theoretical sampling lattice; detector=continuous Gaussian center+scale jitter calibrated to detector localization error (tests if marginalization holds under realistic detector bbox uncertainty, NOT just synthetic lattice). Market has no source frame so this is a literature-informed proxy, not real detector boxes.')
    74	ap.add_argument('--jitter_sigma', type=float, default=0.5, help='detector jitter translation sigma in LR-px (scale sigma=0.2*this). smaller=closer to precise lattice. sweep to map marginalization gain vs detector-error magnitude.')
    75	ap.add_argument('--dataset', default='market1501', choices=['market1501', 'msmt17'], help='cross-dataset push-7.0 kill-switch②: market1501 (dir split) or msmt17 (list-file split). MSMT17 needs its own ckpt+config (num_class differs).')
    76	ap.add_argument('--semantic_weight', type=float, default=-1.0, help='override MODEL.SEMANTIC_WEIGHT to match ckpt training (MSMT17 swin ckpt trained sw=0.6 but config has 0.2 -> SOLIDER backbone feature mismatch). -1=use config.')
    77	ap.add_argument('--adaptive_k', action='store_true', help='supporting: per-query phase-volatility selects K (high-vol query marginalize over K, low-vol use K=1). Reduces avg compute keeping most marginalization gain -> rebut "K=9 too expensive".')
    78	cli = ap.parse_args()
    79	np.random.seed(cli.seed)
    80	RNG = np.random.RandomState(cli.seed)
    81	
    82	SIZE_TEST = (384, 128)       # (H, W) the model input
    83	PIXEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    84	PIXEL_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    85	
    86	# PIL resample kernels for the "different antialias kernel" lattice axis
    87	_KERNELS = {
    88	    'bicubic': Image.BICUBIC,
    89	    'bilinear': Image.BILINEAR,
    90	    'lanczos': Image.LANCZOS,
    91	    'box': Image.BOX,
    92	    'hamming': Image.HAMMING,
    93	    'nearest': Image.NEAREST,
    94	}
    95	
    96	
    97	# =========================================================================== #
    98	# dataset list (parse Market dirs directly; no dataloader needed)
    99	# =========================================================================== #
   100	import re, glob
   101	_PAT = re.compile(r'([-\d]+)_c(\d)')
   102	
   103	
   104	def list_split(dir_path):
   105	    items = []
   106	    for p in sorted(glob.glob(os.path.join(dir_path, '*.jpg'))):
   107	        pid, cam = map(int, _PAT.search(p).groups())
   108	        if pid == -1:
   109	            continue
   110	        items.append((p, pid, cam - 1))
   111	    return items
   112	
   113	
   114	def msmt17_split(data_root, list_file):
   115	    """MSMT17 list-file split: each line 'relpath pid'; cam parsed from filename
   116	    (pid_seq_CAM_time_...), images under <data_root>/MSMT17/test/<relpath>."""
   117	    items = []
   118	    base = os.path.join(data_root, 'MSMT17')
   119	    with open(os.path.join(base, list_file)) as f:
   120	        for line in f:
   121	            rel, pid = line.strip().split(' ')
   122	            cam = int(os.path.basename(rel).split('_')[2]) - 1
   123	            items.append((os.path.join(base, 'test', rel), int(pid), cam))
   124	    return items
   125	
   126	
   127	# =========================================================================== #
   128	# LR + lattice variant generation  (all in PIL space, from the ORIGINAL image)
   129	# =========================================================================== #
   130	def _to_target_aspect(img):
   131	    """Resize the original crop to the model's 384x128 (3:1) HR canvas with BICUBIC.
   132	    This is the 'HR' reference everything is degraded from (gallery also uses this)."""
   133	    return img.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
   134	
   135	
   136	def make_lr(hr_img, h, kernel='bicubic'):
   137	    """Deterministic synthetic LR: HR(384x128) --down--> (h, w) --up--> 384x128.
   138	    w preserves the 3:1 canvas aspect: w = round(h/3).  Returns a 384x128 PIL image
   139	    (degrade-then-restore-size, the standard CR-ReID synthetic LR convention)."""
   140	    w = max(1, int(round(h * SIZE_TEST[1] / SIZE_TEST[0])))   # h*128/384 = h/3
   141	    k = _KERNELS[kernel]
   142	    small = hr_img.resize((w, h), k)
   143	    return small.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
   144	
   145	
   146	def make_lattice_variants(hr_img, h, K, rng, fixed_axis=None, jitter_mode='lattice'):
   147	    """K plausible PHASE/CROP/KERNEL variants of the SAME hr image at height h.
   148	    LM-S4 factor ablation: fixed_axis (0=phase,1=bbox,2=zoom) restricts ALL variants to ONE
   149	    lattice axis, isolating which axis drives the test-time gain (cleanest story=phase).
   150	
   151	    Each variant perturbs the SAMPLING LATTICE relative to the scene by a SUB-LR-pixel
   152	    amount, then forms the LR image.  The depicted person is (almost) the same extent;
   153	    only WHICH hr pixels land on each LR sample point changes.  Axes:
   154	      - sub-pixel phase shift  (fractional HR translate before downsample)
   155	      - +/-1 LR-pixel bbox crop shift / expand (integer LR-pixel = h/.. HR pixels)
   156	      - antialias kernel choice
   157	
   158	    variant 0 is ALWAYS the canonical deterministic bicubic LR (no perturbation) so the
   159	    single-LR baseline == variants[0].
   160	    Returns list of K PIL images (each 384x128)."""
   161	    W_hr, H_hr = hr_img.size                      # 128, 384
   162	    # how many HR pixels correspond to 1 LR pixel at this height
   163	    hr_per_lr_y = H_hr / float(h)                  # 384/h
   164	    hr_per_lr_x = W_hr / float(max(1, round(h / 3.0)))  # 128/(h/3) ~ 3
   165	    variants = [make_lr(hr_img, h, 'bicubic')]     # 0: canonical
   166	    kernels_cycle = ['bicubic', 'bilinear', 'lanczos', 'box', 'hamming']
   167	    for j in range(1, K):
   168	        # --- pick a lattice perturbation type round-robin so the K cover all axes ---
   169	        mode = fixed_axis if fixed_axis is not None else (j % 3)
   170	        kern = kernels_cycle[j % len(kernels_cycle)]
   171	        if jitter_mode == 'detector':
   172	            # detector-like localization jitter: continuous Gaussian center-shift + scale error,
   173	            # calibrated to typical detector bbox localization error (sigma ~0.5 LR-px translate,
   174	            # ~10% scale). NOT real detector boxes (Market has no source frame) -- a literature-
   175	            # informed proxy for deployment uncertainty, replacing the uniform +-1 LR-px lattice.
   176	            dx = rng.normal(0, cli.jitter_sigma) * hr_per_lr_x
   177	            dy = rng.normal(0, cli.jitter_sigma) * hr_per_lr_y
   178	            sc = float(np.clip(1.0 + rng.normal(0, cli.jitter_sigma * 0.2), 0.7, 1.3))
   179	            cw, ch = W_hr / sc, H_hr / sc
   180	            cx, cy = W_hr / 2.0 + dx, H_hr / 2.0 + dy
   181	            l, u = int(round(cx - cw / 2)), int(round(cy - ch / 2))
   182	            r, b = int(round(cx + cw / 2)), int(round(cy + ch / 2))
   183	            pad = max(0, -l, -u, r - W_hr, b - H_hr) + 1
   184	            canvas = Image.new('RGB', (W_hr + 2 * pad, H_hr + 2 * pad), (0, 0, 0))
   185	            canvas.paste(hr_img, (pad, pad))
   186	            cropped = canvas.crop((l + pad, u + pad, r + pad, b + pad)).resize(
   187	                (W_hr, H_hr), Image.BICUBIC)
   188	            variants.append(make_lr(cropped, h, kern)); continue
   189	        if mode == 0:
   190	            # sub-pixel phase: fractional shift of up to +/-0.5 LR pixel (in HR px)
   191	            dx = rng.uniform(-0.5, 0.5) * hr_per_lr_x
   192	            dy = rng.uniform(-0.5, 0.5) * hr_per_lr_y
   193	            shifted = hr_img.transform(
   194	                (W_hr, H_hr), Image.AFFINE, (1, 0, dx, 0, 1, dy),
   195	                resample=Image.BICUBIC)
   196	            v = make_lr(shifted, h, kern)
   197	        elif mode == 1:
   198	            # +/-1 LR-pixel bbox crop shift: crop the HR by an integer # of LR pixels on
   199	            # each side then resize back to the HR canvas (== shifting the bbox window).
   200	            sx = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_x))
   201	            sy = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_y))
   202	            left = max(0, sx); upper = max(0, sy)
   203	            right = W_hr + min(0, sx); lower = H_hr + min(0, sy)
   204	            if right - left < 4 or lower - upper < 4:
   205	                left, upper, right, lower = 0, 0, W_hr, H_hr
   206	            cropped = hr_img.crop((left, upper, right, lower)).resize(
   207	                (W_hr, H_hr), Image.BICUBIC)
   208	            v = make_lr(cropped, h, kern)
   209	        else:
   210	            # bbox expand / contract by 1 LR pixel (zoom in/out a touch) + kernel swap
   211	            ez = rng.choice([-1, 1]) * 0.5 * hr_per_lr_y   # expand/contract in HR px
   212	            box = (-ez, -ez * (W_hr / H_hr), W_hr + ez, H_hr + ez * (W_hr / H_hr)) \
   213	                if ez > 0 else (abs(ez), abs(ez) * (W_hr / H_hr),
   214	                                W_hr - abs(ez), H_hr - abs(ez) * (W_hr / H_hr))
   215	            # PIL crop on a fractional/negative box: emulate via paste on padded canvas
   216	            l, u, r, b = box
   217	            l, u, r, b = int(round(l)), int(round(u)), int(round(r)), int(round(b))
   218	            pad = max(0, -l, -u, r - W_hr, b - H_hr) + 1
   219	            canvas = Image.new('RGB', (W_hr + 2 * pad, H_hr + 2 * pad), (0, 0, 0))
   220	            canvas.paste(hr_img, (pad, pad))
   221	            cropped = canvas.crop((l + pad, u + pad, r + pad, b + pad)).resize(
   222	                (W_hr, H_hr), Image.BICUBIC)
   223	            v = make_lr(cropped, h, kern)
   224	        variants.append(v)
   225	    return variants
   226	
   227	
   228	def make_tta_variants(lr_img, K, rng, pad=10, strong=False):
   229	    """ORDINARY TTA control: K views of ONE bicubic LR image via pad+RandomCrop (+ hflip).
   230	    NO lattice/phase semantics -- the standard cheap test-time augmentation.  variant 0 ==
   231	    the un-augmented LR so the single-LR baseline is shared with the lattice path.
   232	    strong=True (LM-S2 defense): ALSO add resize-jitter + brightness/contrast = a RICHER TTA,
   233	    so the lattice ensemble must beat a STRONG (not just crop+flip) ordinary baseline."""
   234	    from PIL import ImageEnhance
   235	    W, H = lr_img.size
   236	    out = [lr_img]                                  # 0: identity (== single LR)
   237	    for j in range(1, K):
   238	        canvas = Image.new('RGB', (W + 2 * pad, H + 2 * pad), (0, 0, 0))
   239	        canvas.paste(lr_img, (pad, pad))
   240	        cx = rng.randint(0, 2 * pad + 1)

exec
/bin/zsh -lc "rg -n \"class FrozenExtractor|def list_split|def feats_from_pil|ae_feats|qf|gf|cache\" experiments/cargo_cvpb/cvpb_lattice_killswitch.py experiments -g'*.py'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:69:ap.add_argument('--cache_gallery', default='/tmp/lattice_gallery_hr.npz')
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:104:def list_split(dir_path):
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:267:class FrozenExtractor:
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:300:    def feats_from_pil(self, pil_list):
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:432:    if cli.reuse_gallery and os.path.exists(cli.cache_gallery) and cli.smoke == 0:
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:433:        z = np.load(cli.cache_gallery, allow_pickle=True)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:434:        gf = z['gf']; assert len(gf) == len(g_items)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:435:        print(f"[gallery] reuse HR feats {cli.cache_gallery}  ({gf.shape})")
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:438:        gf = ext.feats_from_pil(hr_g)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:440:            np.savez(cli.cache_gallery, gf=gf)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:442:        print(f"[gallery] HR feats {gf.shape}  ({time.time()-t0:.0f}s)")
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:445:    hr_qf = ext.feats_from_pil(hr_q)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:446:    dist_hr = 1.0 - hr_qf @ gf.T
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:490:        lr_hr_drift = 1.0 - (f_single * hr_qf).sum(1)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:501:        sims = f_lat @ gf.T                          # [Nq,K,Ng]  (~1.9GB f32)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:530:        d_single = 1.0 - f_single @ gf.T
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:535:        d_lat_mean = 1.0 - f_lat_mean @ gf.T
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:538:        sim_lat_full = f_lat @ gf.T                  # [Nq,K,Ng]
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:555:            sim_adapt = np.where(use_marg[:, None], sim_lat_max, f_single @ gf.T)  # [Nq,Ng]
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:575:        r_tta_mean = eval_map(1.0 - f_tta_mean @ gf.T, q_pid, q_cam, g_pid, g_cam)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:576:        sim_tta_max = (f_tta @ gf.T).max(1)
experiments/afd_reid/verify_agreid_v2.py:128:    qf = torch.tensor(rng.standard_normal((len(q), D)), dtype=torch.float32)
experiments/afd_reid/verify_agreid_v2.py:129:    gf = torch.tensor(rng.standard_normal((len(g), D)), dtype=torch.float32)
experiments/afd_reid/verify_agreid_v2.py:130:    mAP, cmc, mINP = eval_market(qf, qp, qc, gf, gp, gc)
experiments/afd_reid/afd_killswitch.py:10:m = torchvision.models.resnet50(weights='IMAGENET1K_V1'); m.fc = torch.nn.Identity(); m = m.to(device).eval()  # V1=cached resnet50-19c8e357
experiments/afd_reid/afd_killswitch.py:38:def cmap(qf, qp, gf, gp):
experiments/afd_reid/afd_killswitch.py:39:    qf = torch.nn.functional.normalize(qf); gf = torch.nn.functional.normalize(gf)
experiments/afd_reid/afd_killswitch.py:40:    sim = (qf @ gf.t()).numpy(); aps = []; r1 = 0; n = 0
experiments/afd_reid/afd_train.py:139:def eval_market(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=50):
experiments/afd_reid/afd_train.py:144:    if qf.numel() == 0 or gf.numel() == 0:
experiments/afd_reid/afd_train.py:147:    qf = F.normalize(qf, dim=1)
experiments/afd_reid/afd_train.py:148:    gf = F.normalize(gf, dim=1)
experiments/afd_reid/afd_train.py:149:    distmat = (2 - 2 * qf @ gf.t()).numpy()   # cosine distance
experiments/afd_reid/afd_train.py:216:        qf, qp, qc = extract_features(model, ql, device, args.use_afd)
experiments/afd_reid/afd_train.py:217:        gf, gp, gc = extract_features(model, gl, device, args.use_afd)
experiments/afd_reid/afd_train.py:218:        mAP, cmc, mINP = eval_market(qf, qp, qc, gf, gp, gc)
experiments/afd_reid/afd_train.py:344:                gfeat = out['global_feat']
experiments/afd_reid/afd_train.py:346:                loss_tri = tri(gfeat, labels)
experiments/exp367_single_support/cvpb_single_support_probe.py:12:  5. cache provenance: 校验 full-gallery mAP sanity(=exp260b ref 94.4)。
experiments/exp367_single_support/cvpb_single_support_probe.py:24:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/cvpb_single_support_probe.py:29:if not os.path.exists(cli.cache):
experiments/exp367_single_support/cvpb_single_support_probe.py:30:    raise SystemExit(f'[FATAL] cache {cli.cache} 不存在, 先跑 active_evidence probe 生成 Market 特征 cache')
experiments/exp367_single_support/cvpb_single_support_probe.py:31:z = np.load(cli.cache)
experiments/exp367_single_support/cvpb_single_support_probe.py:32:qf, qp, qc = z['qf'], z['qp'], z['qc']
experiments/exp367_single_support/cvpb_single_support_probe.py:33:gf, gp, gc = z['gf'], z['gp'], z['gc']
experiments/exp367_single_support/cvpb_single_support_probe.py:34:assert np.isfinite(qf).all() and np.isfinite(gf).all(), 'feat 含 nan/inf'
experiments/exp367_single_support/cvpb_single_support_probe.py:35:print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/cvpb_single_support_probe.py:53:    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
experiments/exp367_single_support/cvpb_single_support_probe.py:56:        sim_i = qf[i] @ gff.T
experiments/exp367_single_support/cvpb_single_support_probe.py:69:full_g = np.arange(len(gf))
experiments/exp367_single_support/cvpb_single_support_probe.py:71:for i in range(len(qf)):
experiments/exp367_single_support/cvpb_single_support_probe.py:75:print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/cvpb_single_support_probe.py:87:        qual.append((qf[qs] @ gf[g]).mean() if len(qs) else -1.0)
experiments/afd_reid/band_analysis.py:123:    qf, qp, qc = extract(model, ql, device)
experiments/afd_reid/band_analysis.py:124:    gf, gp, gc = extract(model, gl, device)
experiments/afd_reid/band_analysis.py:125:    mAP, cmc, mINP = eval_market(qf, qp, qc, gf, gp, gc)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:29:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:32:            '--reuse_gallery', '--cache_gallery', '/tmp/ae_g.npz']
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:49:if os.path.exists(cli.cache):
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:50:    z = np.load(cli.cache)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:51:    qf, qp, qc, gf, gp, gc = z['qf'], z['qp'], z['qc'], z['gf'], z['gp'], z['gc']
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:52:    print('[feat] cached', flush=True)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:55:    qf, qp, qc = extract('query')
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:56:    gf, gp, gc = extract('bounding_box_test')
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:57:    np.savez(cli.cache, qf=qf, qp=qp, qc=qc, gf=gf, gp=gp, gc=gc)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:58:print(f'[AE] q={len(qf)} g={len(gf)}', flush=True)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:61:def eval_market(qfeat, qp, qc, gf, gp, gc):
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:63:    sim = qfeat @ gf.T
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:65:    margins = np.ones(len(qfeat))                        # 每 query 都算(难度, 不依赖 match)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:66:    for i in range(len(qfeat)):
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:81:for i in range(len(qf)): idc2q[(qp[i], qc[i])].append(i)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:82:second = -np.ones(len(qf), dtype=int)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:83:for i in range(len(qf)):
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:84:    cands = [j for j in range(len(qf)) if qp[j] == qp[i] and qc[j] != qc[i]]
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:87:print(f'[AE] queries with 2nd-evidence available: {has_second.sum()}/{len(qf)}', flush=True)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:92:    qq = qf.copy()
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:94:        qq[i] = (qf[i] + qf[second[i]]); qq[i] /= (np.linalg.norm(qq[i]) + 1e-9)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:99:base_mAP, base_R1, margins = eval_market(qf, qp, qc, gf, gp, gc)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:101:orc_mAP, orc_R1, _ = eval_market(with_evidence(np.ones(len(qf), bool)), qp, qc, gf, gp, gc)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:106:hard = np.zeros(len(qf), bool); hard[cand[np.argsort(margins[cand])[:n_budget]]] = True
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:107:pol_mAP, pol_R1, _ = eval_market(with_evidence(hard), qp, qc, gf, gp, gc)
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:110:rmask = np.zeros(len(qf), bool); ridx = rng.choice(np.where(has_second)[0], min(n_budget, has_second.sum()), replace=False); rmask[ridx] = True
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:111:rnd_mAP, rnd_R1, _ = eval_market(with_evidence(rmask), qp, qc, gf, gp, gc)
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:23:ap.add_argument('--eval_cache', default='/tmp/ae_feats.npz')          # query/gallery 特征(复用)
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:24:ap.add_argument('--train_cache', default='/tmp/ss_train_feats.npz')
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:34:            '--reuse_gallery', '--cache_gallery', '/tmp/ss_g.npz']
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:41:if os.path.exists(cli.train_cache):
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:42:    z = np.load(cli.train_cache); tf, tp = z['tf'], z['tp']
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:43:    print(f'[train feat] cached {tf.shape}', flush=True)
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:50:    np.savez(cli.train_cache, tf=tf, tp=tp); print(f'[train feat] extracted {tf.shape}', flush=True)
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:98:z = np.load(cli.eval_cache)
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:99:qf, qp, qc, gf, gp, gc = z['qf'], z['qp'], z['qc'], z['gf'], z['gp'], z['gc']
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:101:    qf = F.normalize(head(torch.tensor(qf, device=DEV)), dim=1).cpu().numpy()
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:102:    gf = F.normalize(head(torch.tensor(gf, device=DEV)), dim=1).cpu().numpy()
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:106:    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:109:        s = qf[i] @ gff.T; keep = ~((gpp == qp[i]) & (gcc == qc[i]))
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:119:valid_q = np.array([i for i in range(len(qf)) if (gp[~((gp == qp[i]) & (gc == qc[i]))] == qp[i]).any()])
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:123:full_mAP = eval_fixed(np.arange(len(gf)), valid_q)
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:131:    qual = [(qf[qs[qc[qs] != gc[g]]] @ gf[g]).mean() if (qc[qs] != gc[g]).any() else -1 for g in gi]
experiments/afd_reid/afd_model.py:337:            _gfeat, outs = self.swin(x)
experiments/exp365_wildlife/cvpb_wildlife_localverify.py:20:ap.add_argument('--cache', default='/tmp/wl_feats_gz.npz')
experiments/exp365_wildlife/cvpb_wildlife_localverify.py:29:feats = np.load(cli.cache)['feats']
experiments/exp365_wildlife/cvpb_wildlife_localverify.py:38:qf, gf = feats[q_idx], feats[g_idx]; qp, gp = pid[q_idx], pid[g_idx]
experiments/exp365_wildlife/cvpb_wildlife_localverify.py:39:sim = qf @ gf.T; order = np.argsort(-sim, axis=1)
experiments/exp365_wildlife/cvpb_wildlife_probe.py:32:ap.add_argument('--cache', default='/tmp/wl_feats_gz.npz')
experiments/exp365_wildlife/cvpb_wildlife_probe.py:43:if os.path.exists(cli.cache):
experiments/exp365_wildlife/cvpb_wildlife_probe.py:44:    feats = np.load(cli.cache)['feats']
experiments/exp365_wildlife/cvpb_wildlife_probe.py:45:    print(f'[feat] cached {feats.shape}', flush=True)
experiments/exp365_wildlife/cvpb_wildlife_probe.py:64:    np.savez(cli.cache, feats=feats)
experiments/exp365_wildlife/cvpb_wildlife_probe.py:78:def eval_map(qf, gf, qp, gp, qs, gs, topk=10):
experiments/exp365_wildlife/cvpb_wildlife_probe.py:80:    sim = qf @ gf.T                                  # [Nq,Ng] cos
experiments/exp365_wildlife/cvpb_wildlife_probe.py:83:    for i in range(len(qf)):
experiments/exp365_wildlife/cvpb_wildlife_probe.py:97:qf, gf = feats[q_idx], feats[g_idx]
experiments/exp365_wildlife/cvpb_wildlife_probe.py:102:base = eval_map(qf, gf, qp, gp, qs, gs)
experiments/exp365_wildlife/cvpb_wildlife_probe.py:114:qf_c = species_center(feats, species)[q_idx]; gf_c = species_center(feats, species)[g_idx]
experiments/exp365_wildlife/cvpb_wildlife_probe.py:115:cen = eval_map(qf_c, gf_c, qp, gp, qs, gs)
experiments/exp365_wildlife/cvpb_wildlife_probe.py:121:sim = qf @ gf.T
experiments/exp365_wildlife/cvpb_wildlife_probe.py:122:for i in range(len(qf)):
experiments/afd_reid/smoke_agreid_v2_wiring.py:118:qf = torch.eye(3)
experiments/afd_reid/smoke_agreid_v2_wiring.py:119:gf = torch.cat([torch.eye(3), torch.eye(3)], 0)
experiments/afd_reid/smoke_agreid_v2_wiring.py:124:mAP, cmc, mINP = afd_train.eval_market(qf, q_pids, q_cam, gf, g_pids, g_cam)
experiments/cargo_cvpb/airl_gate_oracle.py:177:def cosdist(qf, gf):
experiments/cargo_cvpb/airl_gate_oracle.py:180:    qfn = F.normalize(qf, dim=1)
experiments/cargo_cvpb/airl_gate_oracle.py:181:    gfn = F.normalize(gf, dim=1)
experiments/cargo_cvpb/airl_gate_oracle.py:182:    return (2 - 2 * qfn @ gfn.t()).numpy()
experiments/cargo_cvpb/airl_gate_oracle.py:185:def cossim(qf, gf):
experiments/cargo_cvpb/airl_gate_oracle.py:186:    qfn = F.normalize(qf, dim=1)
experiments/cargo_cvpb/airl_gate_oracle.py:187:    gfn = F.normalize(gf, dim=1)
experiments/cargo_cvpb/airl_gate_oracle.py:188:    return (qfn @ gfn.t()).numpy()
experiments/cargo_cvpb/airl_gate_oracle.py:377:    for d in ('A->G',):  # area gate is meaningful when query=aerial
experiments/cargo_cvpb/maxsim_probe.py:30:and is not what gives a meaningful delta, so we combine in similarity space
experiments/cargo_cvpb/maxsim_probe.py:95:            gfeat = self.model(imgs)                     # (B,2048) L2-normed
experiments/cargo_cvpb/maxsim_probe.py:102:            g_list.append(gfeat.cpu())
experiments/cargo_cvpb/maxsim_probe.py:158:        torch.cuda.empty_cache()
experiments/cargo_cvpb/maxsim_probe.py:231:    # so the default sweeps the meaningful pooled grids 8x4 (32 tok) and 4x2 (8 tok).
experiments/cargo_cvpb/smoke_ovli_setpool.py:115:def original_loss(ovli, OVLIHead, gfeat, tok, labels, views):
experiments/cargo_cvpb/smoke_ovli_setpool.py:118:    B = gfeat.size(0)
experiments/cargo_cvpb/smoke_ovli_setpool.py:119:    device = gfeat.device
experiments/cargo_cvpb/smoke_ovli_setpool.py:120:    gsim = gfeat @ gfeat.t()
experiments/cargo_cvpb/smoke_ovli_setpool.py:132:        z = gfeat.new_zeros(())
experiments/cargo_cvpb/smoke_ovli_setpool.py:168:    return head.tokens_from_cached_map(), fmap
experiments/cargo_cvpb/smoke_ovli_setpool.py:188:    gfeat = F.normalize(torch.randn(B, Dg), dim=1)
experiments/cargo_cvpb/smoke_ovli_setpool.py:214:        l_new, p_new, n_new = head.loss(gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_setpool.py:215:        l_ref, p_ref, n_ref = original_loss(head, OVLIHead, gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_setpool.py:272:        l, _, _ = head.loss(gfeat, tok_g, labels, views)
experiments/cargo_cvpb/smoke_ovli_setpool.py:300:    l_mean, _, _ = head_avg.loss(gfeat, tok_id, labels, views)
experiments/cargo_cvpb/smoke_ovli_setpool.py:305:        l_sp, _, _ = head.loss(gfeat, tok_id, labels, views)
experiments/exp364_dg_foundation/solider_frozen_probe.py:46:        gf = out[0] if isinstance(out, (tuple, list)) else out   # SOLIDER swin out[0] = GAP global feat
experiments/exp364_dg_foundation/solider_frozen_probe.py:47:        feats.append(F.normalize(gf, dim=1).cpu())
experiments/exp364_dg_foundation/solider_frozen_probe.py:79:def eval_reid(qf, qp, qc, gf, gp, gc):
experiments/exp364_dg_foundation/solider_frozen_probe.py:80:    dist = (1 - qf @ gf.t()).numpy()
experiments/exp364_dg_foundation/solider_frozen_probe.py:102:    qf = encode(net, tf, [x[0] for x in q_items], device)
experiments/exp364_dg_foundation/solider_frozen_probe.py:103:    gf = encode(net, tf, [x[0] for x in g_items], device)
experiments/exp364_dg_foundation/solider_frozen_probe.py:104:    mAP, r1 = eval_reid(qf, qp, qc, gf, gp, gc)
experiments/exp364_dg_foundation/solider_frozen_probe.py:106:          f"  [dim={qf.shape[1]}]", flush=True)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:21:KPR mot_inter_intra_video protocol). So the cached q_cam/g_cam are NOT the tracklet id — we
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:57:ap.add_argument('--cache_feat', default='/tmp/realiz_posetrack_feats.npz')
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:142:    np.savez(cli.cache_feat,
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:145:    print(f"[extract] cached -> {cli.cache_feat}", flush=True)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:150:    if cli.reuse_feat and os.path.exists(cli.cache_feat):
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:151:        z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:154:        print(f"[data] reused {cli.cache_feat}", flush=True)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:160:        # video_id parsed from filename c{VID} (the TRACKLET id; cached cam is per-image unique)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:176:def per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam):
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:177:    sim = qf @ gf.T
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:216:def kreciprocal_rerank(qf, gf, k1=20, k2=6, lam=0.3):
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:218:    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:219:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:220:    allf = np.concatenate([qf, gf], 0)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:280:    qf, q_pid, q_cam, q_vid, q_name = q['feat'], q['pid'], q['cam'], q['vid'], q['name']
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:281:    gf, g_pid, g_cam, g_vid = g['feat'], g['pid'], g['cam'], g['vid']
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:284:    base_all = per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:298:    ap_single = per_query_ap(qf[single_rows], gf, q_pid[single_rows], q_cam[single_rows], g_pid, g_cam)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:306:    sim_qg = qf[single_rows] @ gf.T
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:341:        ap_b = per_query_ap(qf[i:i+1], gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0]
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:343:        pack = np.concatenate([qf[i][None], qf[extra]], 0)      # (1+k, D)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:346:        ap_Am = per_query_ap(f_mean[None], gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0]
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:347:        ap_Ax = per_query_ap(f_max[None],  gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0]
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:355:            f_or = qf[i] + gf[gj]; f_or /= (np.linalg.norm(f_or) + 1e-12)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:356:            ap_B = per_query_ap(f_or[None], gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0]
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:364:            f_rd = qf[i] + qf[jr]; f_rd /= (np.linalg.norm(f_rd) + 1e-12)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:365:            rand_draws.append(per_query_ap(f_rd[None], gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0])
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:400:    # recovery RATE (fraction of queries improved by a meaningful margin)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:413:        rr = kreciprocal_rerank(qf[single_rows], gf, cli.krecip_k1, cli.krecip_k2, cli.krecip_lambda)
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:414:        base_dist = 1.0 - qf[single_rows] @ gf.T
experiments/cargo_cvpb/error_analysis_geom.py:64:        gf = model(batch['img'].to(device, non_blocking=True))
experiments/cargo_cvpb/error_analysis_geom.py:65:        if isinstance(gf, (tuple, list)):
experiments/cargo_cvpb/error_analysis_geom.py:66:            gf = gf[0]
experiments/cargo_cvpb/error_analysis_geom.py:67:        feats.append(gf.detach().cpu())
experiments/cargo_cvpb/error_analysis_geom.py:79:_scache = {}
experiments/cargo_cvpb/error_analysis_geom.py:82:    if p not in _scache:
experiments/cargo_cvpb/error_analysis_geom.py:85:            _scache[p] = float(h * w)
experiments/cargo_cvpb/error_analysis_geom.py:87:            _scache[p] = -1.0
experiments/cargo_cvpb/error_analysis_geom.py:88:    return _scache[p]
experiments/cargo_cvpb/error_analysis_geom.py:113:    qf, qp, qc = extract(q)
experiments/cargo_cvpb/error_analysis_geom.py:114:    gf, gp, gc = extract(g)
experiments/cargo_cvpb/error_analysis_geom.py:115:    qf = F.normalize(qf, dim=1); gf = F.normalize(gf, dim=1)
experiments/cargo_cvpb/error_analysis_geom.py:116:    sim = (qf @ gf.t()).numpy()
experiments/cargo_cvpb/diag_swin_eval.py:124:    gfeat_swin, outs = swin(x)
experiments/cargo_cvpb/diag_swin_eval.py:127:    describe('swin.gfeat', gfeat_swin)                # avgpool inside swin.forward
experiments/exp364_dg_foundation/frozen_xdomain_probe.py:65:def eval_reid(qf, qp, qc, gf, gp, gc):
experiments/exp364_dg_foundation/frozen_xdomain_probe.py:66:    dist = (1 - qf @ gf.t()).numpy()
experiments/exp364_dg_foundation/frozen_xdomain_probe.py:88:    qf = encode(model, tf, [x[0] for x in q_items], device)
experiments/exp364_dg_foundation/frozen_xdomain_probe.py:89:    gf = encode(model, tf, [x[0] for x in g_items], device)
experiments/exp364_dg_foundation/frozen_xdomain_probe.py:90:    mAP, r1 = eval_reid(qf, qp, qc, gf, gp, gc)
experiments/exp364_dg_foundation/frozen_xdomain_probe.py:92:          f"  [qcam={list(np.unique(qc))[:6]} gcam={list(np.unique(gc))[:6]} dim={qf.shape[1]} "
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:35:Reuses the kill-switch's feature cache + eval helpers + core/pool split.
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:36:Run on lab-3090-d (cached frozen features, pure numpy, no GPU training):
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:39:    --dataset market1501 --cache_feat /tmp/hub_market_feats.npz \
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:41:  # OD: --dataset occluded_duke --cache_feat /tmp/hub_oduke_feats.npz
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:52:ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz')
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:74:# data load (reuse kill-switch cache + normalization)
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:77:    z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:87:    print(f"[data] {cli.cache_feat}: Nq={len(q['name'])} Ng={len(g['name'])} dim={q['feat'].shape[1]} "
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:95:def per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam):
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:97:    sim = qf @ gf.T
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:259:    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:260:    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:275:    cqf, cq_pid, cq_cam = qf[qsel], q_pid[qsel], q_cam[qsel]
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:319:    base_aps = per_query_ap(cqf, gf[core_idx], cq_pid, cq_cam, g_pid[core_idx], g_cam[core_idx])
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:332:        aps_10x_runs.append(per_query_ap(cqf, gf[gidx], cq_pid, cq_cam, g_pid[gidx], g_cam[gidx]))
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:360:    sim_core = cqf @ gf[core_idx].T                    # core queries x core gallery
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:361:    sim_vis = cqf @ gf[pool_visible].T                 # core queries x visible distractors
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:367:    triv_norm = np.linalg.norm(qf[qsel], axis=1)       # raw query feature norm (pre-normalize)
experiments/cargo_cvpb/cvpb_lsmrt_probe.py:7:classifier head (that was L_marg, which collapsed).  Frozen backbone + cached feats => cheap probe.
experiments/cargo_cvpb/cvpb_lsmrt_probe.py:27:ap.add_argument('--cache_gallery', default='/tmp/g_lpa.npz')
experiments/cargo_cvpb/cvpb_lsmrt_probe.py:30:            '--K', str(cli.K), '--reuse_gallery', '--cache_gallery', cli.cache_gallery]
experiments/cargo_cvpb/cvpb_lsmrt_probe.py:62:# ---- 1. TRAIN: cache variant feats, fit P with set-retrieval SupCon ----
experiments/cargo_cvpb/cvpb_lsmrt_probe.py:100:gf = np.load(cli.cache_gallery, allow_pickle=True)['gf']
experiments/cargo_cvpb/cvpb_lsmrt_probe.py:105:    zg = F.normalize(P(torch.tensor(gf, device=DEV)), dim=-1).cpu().numpy()      # [Ng,D]
experiments/cargo_cvpb/cvpb_lsmrt_probe.py:107:ug = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/afd_train.py:557:    def tokens_from_cached_map(self):
experiments/cargo_cvpb/afd_train.py:566:            raise RuntimeError("OVLIHead: no cached layer4 map; run model "
experiments/cargo_cvpb/afd_train.py:567:                               "forward before tokens_from_cached_map().")
experiments/cargo_cvpb/afd_train.py:569:        # (the cached map may be fp16 under autocast).
experiments/cargo_cvpb/afd_train.py:733:    def acvp_neg_bias(self, gfeat, labels, views, neg, proto, inited,
experiments/cargo_cvpb/afd_train.py:767:        B = gfeat.size(0)
experiments/cargo_cvpb/afd_train.py:768:        # work in fp32 for the cos/sigmoid/clamp/log numerics (gfeat may be fp32
experiments/cargo_cvpb/afd_train.py:770:        z = gfeat.float()                                           # (B,D) L2-normed
experiments/cargo_cvpb/afd_train.py:812:    def loss(self, gfeat, tok, labels, views,
experiments/cargo_cvpb/afd_train.py:817:        gfeat:(B,D) L2-normed global feature (gradient flows -> encoder).
experiments/cargo_cvpb/afd_train.py:840:        B = gfeat.size(0)
experiments/cargo_cvpb/afd_train.py:841:        device = gfeat.device
experiments/cargo_cvpb/afd_train.py:843:        gsim = gfeat @ gfeat.t()                                   # (B,B)
experiments/cargo_cvpb/afd_train.py:865:            z = gfeat.new_zeros(())
experiments/cargo_cvpb/afd_train.py:869:                self._acvp_stats = (z.detach(), gfeat.new_ones(()),
experiments/cargo_cvpb/afd_train.py:870:                                    gfeat.new_zeros((), dtype=torch.long))
experiments/cargo_cvpb/afd_train.py:887:                gfeat, labels, views, neg, acvp_proto, acvp_inited,
experiments/cargo_cvpb/afd_train.py:930:        gfs, tks, pids, cams = [], [], [], []
experiments/cargo_cvpb/afd_train.py:936:            gf = model(imgs, view_idx=vidx)              # (b,D) L2-normed BN
experiments/cargo_cvpb/afd_train.py:937:            tok = ovli.tokens_from_cached_map()           # (b,K,Dp) L2-normed
experiments/cargo_cvpb/afd_train.py:938:            gfs.append(gf.cpu())
experiments/cargo_cvpb/afd_train.py:942:        if not gfs:
experiments/cargo_cvpb/afd_train.py:945:        return (torch.cat(gfs, 0), torch.cat(tks, 0),
experiments/cargo_cvpb/afd_train.py:996:            torch.cuda.empty_cache()
experiments/cargo_cvpb/afd_train.py:1006:        qf, qt, qp, qc = extract(q)
experiments/cargo_cvpb/afd_train.py:1007:        gf, gt, gp, gc = extract(g)
experiments/cargo_cvpb/afd_train.py:1008:        if qf.numel() == 0 or gf.numel() == 0:
experiments/cargo_cvpb/afd_train.py:1012:        qf = F.normalize(qf, dim=1)
experiments/cargo_cvpb/afd_train.py:1013:        gf = F.normalize(gf, dim=1)
experiments/cargo_cvpb/afd_train.py:1014:        gsim = (qf @ gf.t()).numpy()                      # (Nq,Ng) cosine
experiments/cargo_cvpb/afd_train.py:1016:        gmap, gcmc, _ = eval_market(qf, qp, qc, gf, gp, gc)
experiments/cargo_cvpb/afd_train.py:2032:                gfeat = out['global_feat']
experiments/cargo_cvpb/afd_train.py:2035:                loss_tri = tri(gfeat, labels)
experiments/cargo_cvpb/afd_train.py:2072:            # the autocast forward already cached the fp16 layer4 map) keeps the
experiments/cargo_cvpb/afd_train.py:2082:                    tok = ovli.tokens_from_cached_map()          # (B,K,Dp) fp32
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:45:      --cache_feat /tmp/hub_market_feats.npz 2>&1 | tee /tmp/cvpb_hubness_market.log
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:58:ap.add_argument('--dataset', default='market1501', help='label only (for headers/cache name)')
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:59:ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz',
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:61:ap.add_argument('--reuse_feat', action='store_true', help='reuse --cache_feat if present')
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:142:    np.savez(cli.cache_feat,
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:145:    print(f"[extract] cached -> {cli.cache_feat}", flush=True)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:305:    if cli.reuse_feat and os.path.exists(cli.cache_feat):
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:306:        z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:309:        print(f"[reuse] features from {cli.cache_feat}: q={len(q['name'])} g={len(g['name'])}")
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:323:    qf = q['feat'].astype(np.float32); gf = g['feat'].astype(np.float32)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:324:    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:325:    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:328:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:329:    print(f"[data] Nq={Nq} Ng={Ng} dim={qf.shape[1]}  "
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:333:    sim = qf @ gf.T                 # (Nq,Ng)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:555:    rr = kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=False)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:557:    rr_ca = kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=True)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:652:def kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=False):
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:653:    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:654:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:655:    allf = np.concatenate([qf, gf], 0)            # (Nq+Ng, D)
experiments/cargo_cvpb/diag_swin_ckpt.py:74:        gfeat = model._pool(feat_map)
experiments/cargo_cvpb/diag_swin_ckpt.py:75:        describe('global_feat', gfeat)
experiments/cargo_cvpb/diag_swin_ckpt.py:76:        bn = model.bottleneck(gfeat)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:38:      --cache_feat /tmp/rma_rstp_feats.npz 2>&1 | tee experiments/cargo_cvpb/cvpb_rma.log
experiments/cargo_cvpb/cvpb_rma_killswitch.py:51:ap.add_argument('--cache_feat', default='/tmp/rma_rstp_feats.npz')
experiments/cargo_cvpb/cvpb_rma_killswitch.py:214:    if cli.reuse_feat and os.path.exists(cli.cache_feat):
experiments/cargo_cvpb/cvpb_rma_killswitch.py:215:        z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:216:        print(f"[reuse] features from {cli.cache_feat}")
experiments/cargo_cvpb/cvpb_rma_killswitch.py:229:    np.savez(cli.cache_feat,
experiments/cargo_cvpb/cvpb_rma_killswitch.py:232:    print(f"[extract] cached -> {cli.cache_feat}", flush=True)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:360:def run_t2i(rec_test, gf, g_pid, proto, idf, label, color_only=False, token_subset=None):
experiments/cargo_cvpb/cvpb_rma_killswitch.py:366:    sim = M[valid] @ gf.T
experiments/cargo_cvpb/cvpb_rma_killswitch.py:387:    gf = test_feat / (np.linalg.norm(test_feat, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:389:    print(f"[data] test gallery={gf.shape[0]} imgs dim={gf.shape[1]}; "
experiments/cargo_cvpb/cvpb_rma_killswitch.py:398:    sim_ii = gf @ gf.T
experiments/cargo_cvpb/cvpb_rma_killswitch.py:418:        rec['test'], gf, g_pid, proto, idf, 'token-proto (ALL)')
experiments/cargo_cvpb/cvpb_rma_killswitch.py:423:        rec['test'], gf, g_pid, proto, idf, 'color-only', color_only=True)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:430:        rec['test'], gf, g_pid, proto_shuf, idf, 'token-shuffle', token_subset=proto_shuf)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:432:    D = gf.shape[1]
experiments/cargo_cvpb/cvpb_rma_killswitch.py:436:        rec['test'], gf, g_pid, proto_rand, idf, 'random-prototype', token_subset=proto_rand)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:442:    gperm = RNG.permutation(gf.shape[0])
experiments/cargo_cvpb/cvpb_rma_killswitch.py:443:    sim_fs = M[valid] @ gf[gperm].T          # gallery features permuted vs their pids
experiments/cargo_cvpb/cvpb_rma_killswitch.py:452:    res_rq, _ = eval_rank(Rq @ gf.T, qp[valid], g_pid)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:481:    r1_ii = np.zeros(gf.shape[0], bool)
experiments/cargo_cvpb/cvpb_rma_killswitch.py:482:    for i in range(gf.shape[0]):
experiments/cargo_cvpb/cvpb_rma_killswitch.py:492:    sim_t = Mf[validf] @ gf.T
experiments/exp363_ag_foundation/ag_frozen_baseline.py:79:def eval_map_r1(qf, qp, qc, gf, gp, gc):
experiments/exp363_ag_foundation/ag_frozen_baseline.py:81:    dist = 1 - qf @ gf.t()                                   # [Q,G]，feat 已 L2 norm
experiments/exp363_ag_foundation/ag_frozen_baseline.py:98:def oracle_map_r1(q_fr_list, qp, qc, gf, gp, gc):
experiments/exp363_ag_foundation/ag_frozen_baseline.py:106:            d = (1 - (f.unsqueeze(0) @ gf.t())).numpy()[0]
experiments/exp363_ag_foundation/ag_frozen_baseline.py:137:    gf, gp, gc = encode_tracklets(ds.gallery, model, proc, cli.device, cli.nframes, 'mean')
experiments/exp363_ag_foundation/ag_frozen_baseline.py:142:        qf, qp, qc = encode_tracklets(ds.query, model, proc, cli.device, cli.nframes, mode)
experiments/exp363_ag_foundation/ag_frozen_baseline.py:143:        mAP, r1 = eval_map_r1(qf, qp, qc, gf, gp, gc)
experiments/exp363_ag_foundation/ag_frozen_baseline.py:153:    oap, or1 = oracle_map_r1(q_fr, qp, qc, gf, gp, gc)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:31:NOTHING is trained: frozen cached features + numpy only. Two datasets (market + od).
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:33:Run on lab-3090-d (features already cached by the hubness script):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:37:      --cache_feat /tmp/hub_market_feats.npz --dataset market1501 \
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:39:    # od: --cache_feat /tmp/hub_oduke_feats.npz --dataset occluded_duke
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:45:ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz',
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:46:                help='cached frozen features (q_feat/q_pid/q_cam, g_feat/g_pid/g_cam)')
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:104:def kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=False):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:105:    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:106:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:107:    allf = np.concatenate([qf, gf], 0)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:152:def build_bags(qf, q_pid, q_cam, gf, g_pid, g_cam, c_count, m_true):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:166:    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:167:    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:188:            true_feats = [qf[anchor]]
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:195:            pool_feats = [qf[i] for i in other_q] + [gf[j] for j in cross_cam_g]
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:206:                    true_feats.append(qf[anchor])
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:212:                sims = gf @ qf[anchor]
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:229:                    contam_feats.append(gf[cand[s]])
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:251:def fuse_average(bag, gf):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:254:    return 1.0 - gf @ v                                  # distance row
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:256:def fuse_median(bag, gf):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:259:    return 1.0 - gf @ v
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:261:def fuse_trimmed(bag, gf, trim=0.25):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:270:    return 1.0 - gf @ v
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:272:def fuse_single_best(bag, gf):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:276:    sims = bag @ gf.T                                    # (B, Ng)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:279:def bag_agreement_matrix(bag, gf, topL):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:283:    sims = bag @ gf.T                                    # (B, Ng)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:293:def consensus_select(bag, gf, topL, overlap_thr, mode='medoid'):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:306:    A = bag_agreement_matrix(bag, gf, topL)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:343:    agreement matrix from those sets. Avoids recomputing bag @ gf.T per sweep config."""
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:384:def fuse_consensus(bag, gf, topL, overlap_thr, trim=0.25, mode='medoid'):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:385:    keep = consensus_select(bag, gf, topL, overlap_thr, mode=mode)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:388:        return fuse_trimmed(sub, gf, trim=trim)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:390:    return 1.0 - gf @ v
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:396:def run_cell(bags, gf, g_pid, g_cam):
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:400:    Ng = gf.shape[0]
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:407:    fused_avg_vecs = np.empty((Nb, gf.shape[1]), np.float32)   # for k-recip / camera
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:410:        D['avg'][bi] = fuse_average(bag, gf)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:411:        D['single-best'][bi] = fuse_single_best(bag, gf)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:412:        D['median'][bi] = fuse_median(bag, gf)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:413:        D['trimmed'][bi] = fuse_trimmed(bag, gf, trim=cli.trim_frac)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:429:        sims = b['bag_feat'] @ gf.T                      # (B, Ng)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:446:                        Dc[bi] = fuse_trimmed(sub, gf, trim=cli.trim_frac)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:449:                        Dc[bi] = 1.0 - gf @ v
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:459:    rr = kreciprocal_rerank(fused_avg_vecs, gf, bag_cam, g_cam, k1=20, k2=6, lam=0.3,
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:464:    sim_avg = fused_avg_vecs @ gf.T
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:483:    print(f"# AMBIGUOUS QUERY-BAG KILL-SWITCH  dataset={DS}  cache={cli.cache_feat}")
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:488:    z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:489:    qf, q_pid, q_cam = z['q_feat'].astype(np.float32), z['q_pid'], z['q_cam']
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:490:    gf, g_pid, g_cam = z['g_feat'].astype(np.float32), z['g_pid'], z['g_cam']
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:492:    gf, g_pid, g_cam = gf[keep_g], g_pid[keep_g], g_cam[keep_g]
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:493:    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:494:    print(f"[data] Nq={qf.shape[0]} Ng={gf.shape[0]} dim={qf.shape[1]} "
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:510:        bags = build_bags(qf, q_pid, q_cam, gf, g_pid, g_cam, c_count, m_true)
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:511:        res, diag = run_cell(bags, gf, g_pid, g_cam)
experiments/cargo_cvpb/hub_verify_p3_mask.py:12:  1. cached features -> H_k(g) -> pick top-N hub gallery + N camera-matched low-H_k ctrl.
experiments/cargo_cvpb/hub_verify_p3_mask.py:28:  Background-mask sanity: re-extracting with NO mask must reproduce the cached feature
experiments/cargo_cvpb/hub_verify_p3_mask.py:34:        --cache_feat /tmp/hub_oduke_feats.npz 2>&1 | tee /tmp/hub_p3_oduke.log
experiments/cargo_cvpb/hub_verify_p3_mask.py:49:ap.add_argument('--cache_feat', default='/tmp/hub_oduke_feats.npz')
experiments/cargo_cvpb/hub_verify_p3_mask.py:61:# --------------------------------------------------------- H_k from cached feats
experiments/cargo_cvpb/hub_verify_p3_mask.py:144:    # ---- cached feats -> H_k -> select hub + ctrl ----
experiments/cargo_cvpb/hub_verify_p3_mask.py:145:    z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/hub_verify_p3_mask.py:146:    qf = z['q_feat'].astype(np.float32); gf = z['g_feat'].astype(np.float32)
experiments/cargo_cvpb/hub_verify_p3_mask.py:149:    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/hub_verify_p3_mask.py:150:    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/hub_verify_p3_mask.py:151:    Nq, Ng = qf.shape[0], gf.shape[0]; km = cli.k_main
experiments/cargo_cvpb/hub_verify_p3_mask.py:152:    sim = qf @ gf.T
experiments/cargo_cvpb/hub_verify_p3_mask.py:158:    print(f"[data] Nq={Nq} Ng={Ng} dim={qf.shape[1]}; hub H_k range "
experiments/cargo_cvpb/hub_verify_p3_mask.py:281:    # ---- sanity: re-extract 'orig' for a few hub items, compare to cached ----
experiments/cargo_cvpb/hub_verify_p3_mask.py:282:    print("\n[sanity] re-extracted 'orig' vs cached cosine (want ~1.0):", flush=True)
experiments/cargo_cvpb/hub_verify_p3_mask.py:287:            c = float(f @ gf[gi]); cos_list.append(c)
experiments/cargo_cvpb/hub_verify_p3_mask.py:309:    # (All OTHER gallery features stay at their original cached value.)
experiments/cargo_cvpb/hub_verify_p3_mask.py:311:        gf_mod = gf.copy()
experiments/cargo_cvpb/hub_verify_p3_mask.py:315:                gf_mod[gi] = feats[cond][int(gi)]
experiments/cargo_cvpb/hub_verify_p3_mask.py:317:        sim_mod = qf @ gf_mod.T                 # (Nq,Ng)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:49:Reuses the kill-switch feature caches + eval/stat helpers + core/pool split.
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:50:Run on lab-3090-d (cached frozen features, pure numpy, no GPU training):
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:53:    --dataset market1501 --cache_feat /tmp/hub_market_feats.npz \
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:55:  # OD: --dataset occluded_duke --cache_feat /tmp/hub_oduke_feats.npz
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:66:ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz')
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:92:# DATA LOAD (reuse kill-switch cache + normalization)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:95:    z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:105:    print(f"[data] {cli.cache_feat}: Nq={len(q['name'])} Ng={len(g['name'])} dim={q['feat'].shape[1]} "
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:114:def per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam, topk=None, return_false=False):
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:115:    sim = qf @ gf.T
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:281:def positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, a_temp):
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:285:    sim = qf @ gf.T                                     # nq x Ng (positives sparse)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:307:    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:308:    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:316:    cqf, cq_pid, cq_cam = qf[qsel], q_pid[qsel], q_cam[qsel]
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:337:    return dict(qsel=qsel, cqf=cqf, cq_pid=cq_pid, cq_cam=cq_cam,
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:351:    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:353:    cqf, cq_pid, cq_cam = cp['cqf'], cp['cq_pid'], cp['cq_cam']
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:372:    base_aps, base_false = per_query_ap(cqf, gf[core_idx], cq_pid, cq_cam,
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:386:        runs.append(per_query_ap(cqf, gf[gidx], cq_pid, cq_cam, g_pid[gidx], g_cam[gidx]))
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:394:    ps = positive_support(cqf, cq_pid, cq_cam, gf[core_idx], g_pid[core_idx], g_cam[core_idx], cli.a_temp)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:401:    sim_core = cqf @ gf[core_idx].T
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:402:    sim_vis = cqf @ gf[pool_visible].T
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:466:    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:467:    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:468:    aps, false_k = per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam,
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:483:    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:484:    # NOTE: cached features are L2-normalized at extraction (F.normalize in extract_features),
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:495:    sim = qf @ gf.T
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:546:def kreciprocal_rerank(qf, gf, k1=20, k2=6, lam=0.3):
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:552:    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:553:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:554:    allf = np.concatenate([qf, gf], 0)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:606:    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:607:    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:610:    aps, _ = per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam, return_false=True)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:612:    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:664:        ap1 = per_query_ap(qf[qi:qi+1], gf, q_pid[qi:qi+1], q_cam[qi:qi+1], g_pid, g_cam)[0]
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:666:        f_mean = qf[qi] + qf[j2]; f_mean /= (np.linalg.norm(f_mean) + 1e-12)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:667:        f_max = np.maximum(qf[qi], qf[j2]); f_max /= (np.linalg.norm(f_max) + 1e-12)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:668:        ap_mean = per_query_ap(f_mean[None], gf, q_pid[qi:qi+1], q_cam[qi:qi+1], g_pid, g_cam)[0]
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:669:        ap_max = per_query_ap(f_max[None], gf, q_pid[qi:qi+1], q_cam[qi:qi+1], g_pid, g_cam)[0]
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:674:            f_rand = qf[qi] + qf[jr]; f_rand /= (np.linalg.norm(f_rand) + 1e-12)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:675:            ap_rand_draws.append(per_query_ap(f_rand[None], gf, q_pid[qi:qi+1], q_cam[qi:qi+1],
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:692:    # recovery rate: fraction of queries whose AP rises by a meaningful margin
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:705:        rr_dist = kreciprocal_rerank(qf, gf, cli.krecip_k1, cli.krecip_k2, cli.krecip_lambda)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:706:        base_dist = 1.0 - qf @ gf.T
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:732:    print(f"# EVIDENCE-SUFFICIENCY KILL-SWITCH  dataset={cli.dataset}  cache={cli.cache_feat}")
experiments/cargo_cvpb/cvpb_osac_killswitch.py:53:      --cache_dir /tmp/osac_od 2>&1 | tee /tmp/cvpb_osac_od.log
experiments/cargo_cvpb/cvpb_osac_killswitch.py:57:      --dataset market1501 --cache_dir /tmp/osac_mk
experiments/cargo_cvpb/cvpb_osac_killswitch.py:71:ap.add_argument('--dataset', default='occluded_duke', help='label only (headers/cache)')
experiments/cargo_cvpb/cvpb_osac_killswitch.py:74:ap.add_argument('--cache_dir', default='/tmp/osac_od',
experiments/cargo_cvpb/cvpb_osac_killswitch.py:75:                help='per-epoch feature cache dir (q/g/train features per epoch)')
experiments/cargo_cvpb/cvpb_osac_killswitch.py:76:ap.add_argument('--reuse_feat', action='store_true', help='reuse cached features if present')
experiments/cargo_cvpb/cvpb_osac_killswitch.py:92:os.makedirs(cli.cache_dir, exist_ok=True)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:196:    cache = os.path.join(cli.cache_dir, f'feat_ep{epoch}.npz')
experiments/cargo_cvpb/cvpb_osac_killswitch.py:197:    if cli.reuse_feat and os.path.exists(cache):
experiments/cargo_cvpb/cvpb_osac_killswitch.py:198:        z = np.load(cache, allow_pickle=True)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:205:        print(f"[reuse] ep{epoch} from {cache}: q={len(q['name'])} g={len(g['name'])} "
experiments/cargo_cvpb/cvpb_osac_killswitch.py:216:    np.savez(cache, **save)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:217:    print(f"[cache] ep{epoch} -> {cache}")
experiments/cargo_cvpb/cvpb_osac_killswitch.py:480:def kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=False):
experiments/cargo_cvpb/cvpb_osac_killswitch.py:481:    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:482:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/cvpb_osac_killswitch.py:483:    allf = np.concatenate([qf, gf], 0)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:532:def hub_and_map(qf, gf, q_pid, q_cam, g_pid, g_cam, k_main):
experiments/cargo_cvpb/cvpb_osac_killswitch.py:533:    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:534:    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:535:    sim = qf @ gf.T
experiments/cargo_cvpb/cvpb_osac_killswitch.py:573:        gf = g['feat'].astype(np.float64)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:574:        gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:575:        qf = q['feat'].astype(np.float64)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:576:        qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:578:        eigs_g = covariance_eigvals(gf)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:584:        res = eval_map(1.0 - (qf @ gf.T), q['pid'], q['cam'], g['pid'], g['cam'])
experiments/cargo_cvpb/cvpb_osac_killswitch.py:647:    qf = q['feat'].astype(np.float64); gf = g['feat'].astype(np.float64)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:648:    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:649:    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:652:    Nq, Ng, D = qf.shape[0], gf.shape[0], qf.shape[1]
experiments/cargo_cvpb/cvpb_osac_killswitch.py:653:    sim = qf @ gf.T
experiments/cargo_cvpb/cvpb_osac_killswitch.py:663:    g_mu, g_V, g_eigs = fit_pca(gf)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:683:    qe_top1 = query_topPC_energy(qf, g_mu, g_V, 1)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:684:    qe_top10 = query_topPC_energy(qf, g_mu, g_V, 10)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:690:        q_proto_max = (qf @ P.T).max(1)           # (Nq,)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:691:        q_proto_mean = (qf @ P.T).mean(1)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:758:    base_res, base_skew, base_hmax, _, _ = hub_and_map(qf, gf, q_pid, q_cam, g_pid, g_cam, cli.k_main)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:762:    best_abtt = dict(m=0, mAP=base_res['mAP'], qf=qf, gf=gf)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:765:        qf_a = abtt_remove(qf, g_mu, g_V, which)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:766:        gf_a = abtt_remove(gf, g_mu, g_V, which)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:767:        r, sk, hm, _, _ = hub_and_map(qf_a, gf_a, q_pid, q_cam, g_pid, g_cam, cli.k_main)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:768:        abtt_results[m] = dict(mAP=r['mAP'], r1=r['r1'], skew=sk, hmax=hm, qf=qf_a, gf=gf_a)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:771:            best_abtt = dict(m=m, mAP=r['mAP'], r1=r['r1'], qf=qf_a, gf=gf_a); flag = '  <== best'
experiments/cargo_cvpb/cvpb_osac_killswitch.py:776:    best_white = dict(nk=0, mAP=base_res['mAP'], qf=qf, gf=gf)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:780:        qf_w, gf_w = zca_whiten(qf, gf, g_mu, g_V, g_eigs, eps=1e-3, n_keep=nk)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:781:        r, sk, hm, _, _ = hub_and_map(qf_w, gf_w, q_pid, q_cam, g_pid, g_cam, cli.k_main)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:784:            best_white = dict(nk=nk, mAP=r['mAP'], r1=r['r1'], qf=qf_w, gf=gf_w); flag = '  <== best'
experiments/cargo_cvpb/cvpb_osac_killswitch.py:794:        aps_a = per_query_ap(1.0 - best_abtt['qf'] @ best_abtt['gf'].T,
experiments/cargo_cvpb/cvpb_osac_killswitch.py:814:    qf_top = abtt_remove(qf, g_mu, g_V, np.arange(m_star))
experiments/cargo_cvpb/cvpb_osac_killswitch.py:815:    gf_top = abtt_remove(gf, g_mu, g_V, np.arange(m_star))
experiments/cargo_cvpb/cvpb_osac_killswitch.py:816:    r_top = eval_map(1.0 - qf_top @ gf_top.T, q_pid, q_cam, g_pid, g_cam)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:819:    qf_bot = abtt_remove(qf, g_mu, g_V, which_bot)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:820:    gf_bot = abtt_remove(gf, g_mu, g_V, which_bot)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:821:    r_bot = eval_map(1.0 - qf_bot @ gf_bot.T, q_pid, q_cam, g_pid, g_cam)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:826:        qf_r = abtt_remove(qf, g_mu, g_V, which_rnd)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:827:        gf_r = abtt_remove(gf, g_mu, g_V, which_rnd)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:828:        rand_maps.append(eval_map(1.0 - qf_r @ gf_r.T, q_pid, q_cam, g_pid, g_cam)['mAP'])
experiments/cargo_cvpb/cvpb_osac_killswitch.py:840:    dm_rr_base = kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:843:    qf_b, gf_b = best_abtt['qf'], best_abtt['gf']
experiments/cargo_cvpb/cvpb_osac_killswitch.py:844:    dm_rr_abtt = kreciprocal_rerank(qf_b, gf_b, q_cam, g_cam, k1=20, k2=6, lam=0.3)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:847:    qf_w, gf_w = best_white['qf'], best_white['gf']
experiments/cargo_cvpb/cvpb_osac_killswitch.py:848:    dm_rr_white = kreciprocal_rerank(qf_w, gf_w, q_cam, g_cam, k1=20, k2=6, lam=0.3)
experiments/cargo_cvpb/cvpb_osac_killswitch.py:910:    jpath = os.path.join(cli.cache_dir, f'osac_summary_{DS}.json')
experiments/cargo_cvpb/hub_verify_p0_p4.py:4:ZERO-TRAINING: cached frozen features (.npz) + numpy only. No model, no backward.
experiments/cargo_cvpb/hub_verify_p0_p4.py:32:    --cache_feat /tmp/hub_oduke_feats.npz --dataset occluded_duke 2>&1 | tee /tmp/hub_p0p4_oduke.log
experiments/cargo_cvpb/hub_verify_p0_p4.py:33:  ... --cache_feat /tmp/hub_market_feats.npz --dataset market1501 ...
experiments/cargo_cvpb/hub_verify_p0_p4.py:39:ap.add_argument('--cache_feat', required=True)
experiments/cargo_cvpb/hub_verify_p0_p4.py:238:def kreciprocal_rerank(qf, gf, k1=20, k2=6, lam=0.3):
experiments/cargo_cvpb/hub_verify_p0_p4.py:239:    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
experiments/cargo_cvpb/hub_verify_p0_p4.py:240:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/hub_verify_p0_p4.py:241:    allf = np.concatenate([qf, gf], 0)
experiments/cargo_cvpb/hub_verify_p0_p4.py:277:    print(f"# HUBNESS VERIFY P0+P4   dataset={cli.dataset}   k_main={cli.k_main}   feat={cli.cache_feat}")
experiments/cargo_cvpb/hub_verify_p0_p4.py:280:    z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/hub_verify_p0_p4.py:281:    qf = z['q_feat'].astype(np.float32); gf = z['g_feat'].astype(np.float32)
experiments/cargo_cvpb/hub_verify_p0_p4.py:286:    gf, g_pid, g_cam = gf[keep_g], g_pid[keep_g], g_cam[keep_g]
experiments/cargo_cvpb/hub_verify_p0_p4.py:287:    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/hub_verify_p0_p4.py:288:    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/hub_verify_p0_p4.py:289:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/hub_verify_p0_p4.py:291:    print(f"[data] Nq={Nq} Ng={Ng} dim={qf.shape[1]} "
experiments/cargo_cvpb/hub_verify_p0_p4.py:294:    sim = qf @ gf.T
experiments/cargo_cvpb/hub_verify_p0_p4.py:387:    rr = kreciprocal_rerank(qf, gf, k1=cli.rr_k1, k2=cli.rr_k2, lam=cli.rr_lam)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:44:Run on lab-3090-d (reuse the hubness feature caches):
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:47:    --dataset market1501 --cache_feat /tmp/hub_market_feats.npz --reuse_feat \
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:49:  # occluded_duke: --dataset occluded_duke --cache_feat /tmp/hub_oduke_feats.npz --reuse_feat
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:62:ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz')
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:134:    np.savez(cli.cache_feat,
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:143:def per_query_ap_cmc(qf, gf, q_pid, q_cam, g_pid, g_cam, max_rank=10, return_falsecnt=False):
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:147:    sim = qf @ gf.T
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:236:    if cli.reuse_feat and os.path.exists(cli.cache_feat):
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:237:        z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:240:        print(f"[reuse] {cli.cache_feat}: q={len(q['name'])} g={len(g['name'])}", flush=True)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:262:    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:263:    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:282:    cqf, cq_pid, cq_cam = qf[qsel], q_pid[qsel], q_cam[qsel]
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:315:        cqf, gf[core_idx], cq_pid, cq_cam, g_pid[core_idx], g_cam[core_idx],
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:335:                cqf, gf[gidx], cq_pid, cq_cam, g_pid[gidx], g_cam[gidx],
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:406:        res_real, _, _ = per_query_ap_cmc(cqf, gf[gidx], cq_pid, cq_cam,
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:409:        add_feat = gf[add_idx].copy()
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:413:        g_feat_mix = np.concatenate([gf[core_idx], add_feat])
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:416:        res_shuf, _, _ = per_query_ap_cmc(cqf, g_feat_mix, cq_pid, cq_cam,
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:418:        assert g_feat_mix.shape[0] == gf[gidx].shape[0], "CONTROL2 count mismatch"
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:430:def enroll_score(qf_one, enrolled_feat):
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:432:    return float((enrolled_feat @ qf_one).max())
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:459:    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:460:    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:496:                    out[s]['gen'].append((t, enroll_score(qf[qi], ef)))
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:504:                    out[s]['imp'].append((t, enroll_score(qf[qi], ef)))
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:509:        return gf[mask]
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:519:        rows = rs.choice(gf.shape[0], n, replace=False)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:520:        tpl = gf[rows].copy()
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:562:        gd1, sd1, gd5, sd5, gf90, sf90 = [], [], [], [], [], []
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:580:            gf90.append(f90g); sf90.append(f90s)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:582:        return rows, (gd1, sd1, gd5, sd5, gf90, sf90)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:592:    gd1, sd1, gd5, sd5, gf90, sf90 = agg_real
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:595:    drift_red_real = float(np.std(gf90) - np.std(sf90))   # >0 means size-cond flattens drift
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:596:    print(f"  REAL: FPIR@TPIR90 std  global={np.std(gf90):.4f}  size-cond={np.std(sf90):.4f}  "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:604:    gd1r, sd1r, gd5r, sd5r, gf90r, sf90r = agg_rnd
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:607:    drift_red_rnd = float(np.std(gf90r) - np.std(sf90r))
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:608:    print(f"  RANDOM: FPIR@TPIR90 std global={np.std(gf90r):.4f} size-cond={np.std(sf90r):.4f} "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:619:                glob_f90_std=float(np.std(gf90)), sc_f90_std=float(np.std(sf90)),
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:631:    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:632:    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:694:        zgf = gf[gidx]; zg_pid = g_pid[gidx]
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:710:            sim = zgf[keep] @ qf[qi]
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:776:        zgf = gf[gidx]; zg_pid = g_pid[gidx]
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:783:            sim = zgf @ qf[qi]; own = (zg_pid == hid)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:794:            sub_pid = zg_pid[keep]; sim = zgf[keep] @ qf[qi]
experiments/cargo_cvpb/smoke_ovli_ablations.py:100:def original_loss(ovli, OVLIHead, gfeat, tok, labels, views):
experiments/cargo_cvpb/smoke_ovli_ablations.py:103:    B = gfeat.size(0)
experiments/cargo_cvpb/smoke_ovli_ablations.py:104:    device = gfeat.device
experiments/cargo_cvpb/smoke_ovli_ablations.py:105:    gsim = gfeat @ gfeat.t()
experiments/cargo_cvpb/smoke_ovli_ablations.py:120:        z = gfeat.new_zeros(())
experiments/cargo_cvpb/smoke_ovli_ablations.py:174:    return head.tokens_from_cached_map(), fmap
experiments/cargo_cvpb/smoke_ovli_ablations.py:213:    gfeat = F.normalize(torch.randn(B, Dg), dim=1)
experiments/cargo_cvpb/smoke_ovli_ablations.py:214:    l_def, p_def, n_def = head_def.loss(gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_ablations.py:215:    l_ref, p_ref, n_ref = original_loss(head_def, OVLIHead, gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_ablations.py:267:    l_ord, _, _ = head_ord2.loss(gfeat, tok_og, labels, views)
experiments/cargo_cvpb/smoke_ovli_ablations.py:275:    l_def_ord, _, _ = head_def.loss(gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_ablations.py:301:    l_avg, _, _ = head_avg2.loss(gfeat, tok_ag, labels, views)
experiments/cargo_cvpb/smoke_ovli_ablations.py:308:    l_def_avg, _, _ = head_def.loss(gfeat, tok, labels, views)
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:20:trained eval feature), aligns features to the SMPL cache BY IMAGE NAME, and runs
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:32:      --smpl_dir cache/smpl_geom 2>&1 | tee /tmp/cvpb_gopl.log
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:46:ap.add_argument('--smpl_dir', default='cache/smpl_geom')
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:47:ap.add_argument('--cache_feat', default='/tmp/gopl_feats.npz',
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:49:ap.add_argument('--reuse_feat', action='store_true', help='reuse --cache_feat if present')
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:52:                help='SMPL per-IMAGE conf gate (scalar in this cache); images below are still '
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:131:    np.savez(cli.cache_feat,
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:134:    print(f"[extract] cached -> {cli.cache_feat}", flush=True)
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:141:def cosine_distmat(qf, gf):
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:142:    return 1.0 - (qf @ gf.T)
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:186:# pj2d: (N,71,2) in INPUT-image pixel coords (the cache stores ROMP 2D joints in the
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:361:    if cli.reuse_feat and os.path.exists(cli.cache_feat):
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:362:        z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:365:        print(f"[reuse] features from {cli.cache_feat}: q={len(q['name'])} g={len(g['name'])}")
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:370:    qf = q['feat'].astype(np.float64); gf = g['feat'].astype(np.float64)
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:371:    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:372:    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:373:    dm = cosine_distmat(qf, gf)
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:441:    cos_full = 1.0 - (qf @ gf.T)   # (Nq,Ng) cosine distance
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:69:ap.add_argument('--cache_gallery', default='/tmp/lattice_gallery_hr.npz')
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:104:def list_split(dir_path):
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:267:class FrozenExtractor:
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:300:    def feats_from_pil(self, pil_list):
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:432:    if cli.reuse_gallery and os.path.exists(cli.cache_gallery) and cli.smoke == 0:
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:433:        z = np.load(cli.cache_gallery, allow_pickle=True)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:434:        gf = z['gf']; assert len(gf) == len(g_items)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:435:        print(f"[gallery] reuse HR feats {cli.cache_gallery}  ({gf.shape})")
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:438:        gf = ext.feats_from_pil(hr_g)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:440:            np.savez(cli.cache_gallery, gf=gf)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:442:        print(f"[gallery] HR feats {gf.shape}  ({time.time()-t0:.0f}s)")
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:445:    hr_qf = ext.feats_from_pil(hr_q)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:446:    dist_hr = 1.0 - hr_qf @ gf.T
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:490:        lr_hr_drift = 1.0 - (f_single * hr_qf).sum(1)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:501:        sims = f_lat @ gf.T                          # [Nq,K,Ng]  (~1.9GB f32)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:530:        d_single = 1.0 - f_single @ gf.T
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:535:        d_lat_mean = 1.0 - f_lat_mean @ gf.T
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:538:        sim_lat_full = f_lat @ gf.T                  # [Nq,K,Ng]
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:555:            sim_adapt = np.where(use_marg[:, None], sim_lat_max, f_single @ gf.T)  # [Nq,Ng]
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:575:        r_tta_mean = eval_map(1.0 - f_tta_mean @ gf.T, q_pid, q_cam, g_pid, g_cam)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:576:        sim_tta_max = (f_tta @ gf.T).max(1)
experiments/cargo_cvpb/cvpb_containment_killswitch.py:342:def area_of(path, _cache={}):
experiments/cargo_cvpb/cvpb_containment_killswitch.py:343:    if path not in _cache:
experiments/cargo_cvpb/cvpb_containment_killswitch.py:346:            _cache[path] = float(h * w)
experiments/cargo_cvpb/cvpb_containment_killswitch.py:348:            _cache[path] = -1.0
experiments/cargo_cvpb/cvpb_containment_killswitch.py:349:    return _cache[path]
experiments/cargo_cvpb/smoke_ovli_allview.py:64:def ref_oppview_loss(ovli, gfeat, tok, labels, views):
experiments/cargo_cvpb/smoke_ovli_allview.py:65:    B = gfeat.size(0)
experiments/cargo_cvpb/smoke_ovli_allview.py:66:    device = gfeat.device
experiments/cargo_cvpb/smoke_ovli_allview.py:67:    gsim = gfeat @ gfeat.t()
experiments/cargo_cvpb/smoke_ovli_allview.py:78:        z = gfeat.new_zeros(())
experiments/cargo_cvpb/smoke_ovli_allview.py:125:    gfeat = torch.nn.functional.normalize(torch.randn(B, Dg, device=device), dim=1)
experiments/cargo_cvpb/smoke_ovli_allview.py:128:    tok = head_off.tokens_from_cached_map()           # (B,K,proj), proj in graph
experiments/cargo_cvpb/smoke_ovli_allview.py:130:    l_off, ps_off, ns_off = head_off.loss(gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_allview.py:131:    l_ref, ps_ref, ns_ref = ref_oppview_loss(head_off, gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_allview.py:156:    gfeat2 = torch.nn.functional.normalize(
experiments/cargo_cvpb/smoke_ovli_allview.py:160:    tok2 = head_av.tokens_from_cached_map()
experiments/cargo_cvpb/smoke_ovli_allview.py:161:    l_av, ps_av, ns_av = head_av.loss(gfeat2, tok2, labels, views)
experiments/cargo_cvpb/smoke_ovli_allview.py:177:    l_flip, _, _ = head_off.loss(gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_swin_backbone.py:167:    tok = ovli.tokens_from_cached_map()              # (B, K, ovli_dim)
experiments/cargo_cvpb/hub_failure_characterize.py:4:Reuse the cached frozen exp255 global features (/tmp/hub_oduke_feats.npz) to:
experiments/cargo_cvpb/hub_failure_characterize.py:21:ap.add_argument('--cache_feat', default='/tmp/hub_oduke_feats.npz')
experiments/cargo_cvpb/hub_failure_characterize.py:120:    z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/hub_failure_characterize.py:121:    qf = z['q_feat'].astype(np.float32); gf = z['g_feat'].astype(np.float32)
experiments/cargo_cvpb/hub_failure_characterize.py:122:    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/hub_failure_characterize.py:123:    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
experiments/cargo_cvpb/hub_failure_characterize.py:127:    Nq, Ng = qf.shape[0], gf.shape[0]
experiments/cargo_cvpb/hub_failure_characterize.py:128:    print(f"[data] Nq={Nq} Ng={Ng} dim={qf.shape[1]}")
experiments/cargo_cvpb/hub_failure_characterize.py:130:    sim = qf @ gf.T
experiments/cargo_cvpb/cvpb_lcrs_probe.py:19:frozen backbone + cached K=9 feats => cheap, 复用 cvpb_lattice_killswitch / lsmrt framework.
experiments/cargo_cvpb/cvpb_lcrs_probe.py:45:ap.add_argument('--cache_gallery', default='/tmp/g_lcrs.npz')
experiments/cargo_cvpb/cvpb_lcrs_probe.py:48:            '--K', str(cli.K), '--reuse_gallery', '--cache_gallery', cli.cache_gallery]
experiments/cargo_cvpb/cvpb_lcrs_probe.py:156:    qf, yq, cq = variant_feats(items('query'))
experiments/cargo_cvpb/cvpb_lcrs_probe.py:157:    gf, yg, cg = variant_feats(items('bounding_box_test'))
experiments/cargo_cvpb/cvpb_lcrs_probe.py:159:        zq, _ = net(torch.tensor(qf, device=DEV)); zq = zq.cpu().numpy()
experiments/cargo_cvpb/cvpb_lcrs_probe.py:160:        zg_all, _ = net(torch.tensor(gf, device=DEV)); zg = zg_all[:, 0].cpu().numpy()  # gallery canonical-0
experiments/cargo_cvpb/cvpb_lcrs_probe.py:161:        gf0 = F.normalize(torch.tensor(gf[:, 0], device=DEV), dim=-1).cpu().numpy()      # no-P gallery-0
experiments/cargo_cvpb/cvpb_lcrs_probe.py:168:    r_u = ks.eval_map(-setscore(qf / (np.linalg.norm(qf, axis=-1, keepdims=True) + 1e-9), gf0),
experiments/cargo_cvpb/cvpb_lcrs_probe.py:173:                  qf[i] / (np.linalg.norm(qf[i], axis=-1, keepdims=True) + 1e-9),
experiments/cargo_cvpb/cvpb_lcrs_probe.py:174:                  qf[i] / (np.linalg.norm(qf[i], axis=-1, keepdims=True) + 1e-9))) for i in range(min(500, len(qf)))]))
experiments/cbcl_t2i/irra_encoder.py:76:        # openai checkpoint to ~/.cache/clip.
experiments/cargo_cvpb/cvpb_d17_killswitch.py:54:qf = ext.feats_from_pil([ks._to_target_aspect(read_image(x[0])) for x in q_it])   # [Nq,D] L2
experiments/cargo_cvpb/cvpb_d17_killswitch.py:55:gf = ext.feats_from_pil([ks._to_target_aspect(read_image(x[0])) for x in g_it])   # [Ng,D] L2
experiments/cargo_cvpb/cvpb_d17_killswitch.py:56:dist = 1.0 - qf @ gf.T
experiments/cargo_cvpb/cvpb_d17_killswitch.py:70:    qf_s = ext.feats_from_pil([stripe_masked(read_image(x[0]), s, N) for x in q_it])  # [Nq,D]
experiments/cargo_cvpb/cvpb_d17_killswitch.py:72:        full_sim = qf[i] @ gf[idx[i]].T                                          # [topk]
experiments/cargo_cvpb/cvpb_d17_killswitch.py:73:        mask_sim = qf_s[i] @ gf[idx[i]].T
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:49:      --cache_feat /tmp/rr_market_feats.npz 2>&1 | tee /tmp/cvpb_rr_market.log
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:53:    --dataset occluded_duke --cache_feat /tmp/rr_od_feats.npz
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:66:ap.add_argument('--dataset', default='market1501', help='label only (headers/cache name)')
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:67:ap.add_argument('--cache_feat', default='/tmp/rr_market_feats.npz',
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:69:ap.add_argument('--reuse_feat', action='store_true', help='reuse --cache_feat if present')
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:183:    qf_full, gf_full = split(full)
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:184:    q = dict(full=qf_full, pid=pids[:nq], cam=camids[:nq], name=names[:nq])
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:185:    g = dict(full=gf_full, pid=pids[nq:], cam=camids[nq:], name=names[nq:])
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:204:    np.savez(cli.cache_feat, **save)
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:205:    print(f"[extract] cached -> {cli.cache_feat}", flush=True)
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:464:    if cli.reuse_feat and os.path.exists(cli.cache_feat):
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:465:        z = np.load(cli.cache_feat, allow_pickle=True)
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:478:        print(f"[reuse] features from {cli.cache_feat}: q={len(q['name'])} g={len(g['name'])} "
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:501:    qf_full = l2(q['full'].astype(np.float32)); gf_full = l2(g['full'].astype(np.float32))
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:502:    sim_full = qf_full @ gf_full.T
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:507:    qf_cheap = l2(q[f'stage{cs}'].astype(np.float32)); gf_cheap = l2(g[f'stage{cs}'].astype(np.float32))
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:508:    sim_cheap = qf_cheap @ gf_cheap.T
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:511:    print(f"[data] Nq={Nq} Ng={Ng}  full_dim={qf_full.shape[1]} cheap_dim={qf_cheap.shape[1]}  "
experiments/exp362_genoccl/gap_audit.py:39:    qf, qh, nq, qnv = occ_profile('pose_query.npz')
experiments/exp362_genoccl/gap_audit.py:43:        print(f"{gk:8}{tf[gi]*100:11.1f}{qf[gi]*100:11.1f}{(qf[gi]-tf[gi])*100:+11.1f}")
experiments/exp362_genoccl/gap_audit.py:47:    gaps = qf - tf
experiments/exp361_psc_jepa/psc_jepa_pretrain.py:136:    gf = F.normalize(global_feat, dim=1) if global_feat is not None else None
experiments/exp361_psc_jepa/psc_jepa_pretrain.py:137:    return part_pool(featmaps, gmask), gf                                    # (part [B,G,C], global [B,D] L2)
experiments/exp361_psc_jepa/psc_jepa_pretrain.py:200:                    A, A_gf = fwd_tokens(anchor, x, gmask)                 # frozen SOLIDER full-view part [B,G,C] + global [B,D]
experiments/exp361_psc_jepa/psc_jepa_pretrain.py:201:                _, S_gf_full = fwd_tokens(student, x, gmask)              # student FULL-view global (有grad, 锚全局判别几何 codex-R1)
experiments/exp361_psc_jepa/psc_jepa_pretrain.py:203:                L_glob = (1.0 - (S_gf_full * A_gf).sum(1)).mean()        # ★全局 GAP 判别几何 (防 forgetting 核心 codex-R1)
experiments/cargo_cvpb/cvpb_lats_probe.py:32:ap.add_argument('--cache_gallery', default='/tmp/g_lats.npz')
experiments/cargo_cvpb/cvpb_lats_probe.py:95:# ---- 1. TRAIN: cache (global,stripes) variant feats, fit Sidecar with set-retrieval SupCon ----
experiments/cargo_cvpb/smoke_acvp.py:96:def ref_oppview_loss(ovli, gfeat, tok, labels, views):
experiments/cargo_cvpb/smoke_acvp.py:97:    B = gfeat.size(0)
experiments/cargo_cvpb/smoke_acvp.py:98:    device = gfeat.device
experiments/cargo_cvpb/smoke_acvp.py:99:    gsim = gfeat @ gfeat.t()
experiments/cargo_cvpb/smoke_acvp.py:110:        z = gfeat.new_zeros(())
experiments/cargo_cvpb/smoke_acvp.py:171:    Dg = proj_dim  # gfeat / prototype dim must match (cos(z_i, P[..]))
experiments/cargo_cvpb/smoke_acvp.py:174:    gfeat = F.normalize(torch.randn(B, Dg, device=device), dim=1)
experiments/cargo_cvpb/smoke_acvp.py:177:    tok = head.tokens_from_cached_map()                # (B,K,proj), proj in graph
experiments/cargo_cvpb/smoke_acvp.py:180:    l_off, ps_off, ns_off = head.loss(gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_acvp.py:181:    l_ref, ps_ref, ns_ref = ref_oppview_loss(head, gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_acvp.py:190:    l_none, _, _ = head.loss(gfeat, tok, labels, views, acvp_proto=None)
experiments/cargo_cvpb/smoke_acvp.py:197:    l_on, _, _ = head.loss(gfeat, tok, labels, views,
experiments/cargo_cvpb/smoke_acvp.py:220:        gfeat, labels, views, neg, bank, inited,
experiments/cargo_cvpb/smoke_acvp.py:237:        gfeat, labels, views, neg, zbank, zinit,
experiments/cargo_cvpb/smoke_acvp.py:255:    tok_g = head.tokens_from_cached_map()
experiments/cargo_cvpb/smoke_acvp.py:256:    gfeat_g = F.normalize(torch.randn(B, Dg, device=device, requires_grad=True),
experiments/cargo_cvpb/smoke_acvp.py:258:    l_g, _, _ = head.loss(gfeat_g, tok_g, labels, views,
experiments/cargo_cvpb/smoke_acvp.py:278:    l_u, _, _ = head.loss(gfeat, tok, labels, views,
experiments/cargo_cvpb/smoke_acvp.py:288:    l_g0, _, _ = head.loss(gfeat, tok, labels, views,
experiments/cargo_cvpb/smoke_acvp.py:297:    _ = head.loss(gfeat, tok, labels, views,
experiments/cargo_cvpb/smoke_acvp.py:351:        _l, _, _ = head.loss(gfeat, tok, labels, views,
experiments/cargo_cvpb/cvpb_cache_feats.py:2:"""One-time cache of K-lattice-variant features (frozen no-LM-loss backbone) for train/query +
experiments/cargo_cvpb/cvpb_cache_feats.py:34:ft = vfeats(tr); print(f'[cache] train {ft.shape} ({time.time()-t0:.0f}s)', flush=True)
experiments/cargo_cvpb/cvpb_cache_feats.py:35:fq = vfeats(q); print(f'[cache] query {fq.shape} ({time.time()-t0:.0f}s)', flush=True)
experiments/cargo_cvpb/cvpb_cache_feats.py:36:gf = ext.feats_from_pil([ks._to_target_aspect(read_image(it[0])) for it in g])
experiments/cargo_cvpb/cvpb_cache_feats.py:40:         gf=gf, g_pid=np.array([it[1] for it in g]), g_cam=np.array([it[2] for it in g]))
experiments/cargo_cvpb/cvpb_cache_feats.py:41:print(f'[done] cached feats to {cli.out}  ({time.time()-t0:.0f}s)', flush=True)
experiments/cargo_cvpb/airl_scale_diag.py:147:def per_query_top1(qf, gf, q_pids, q_camids, g_pids, g_camids):
experiments/cargo_cvpb/airl_scale_diag.py:150:    qfn = F.normalize(qf, dim=1)
experiments/cargo_cvpb/airl_scale_diag.py:151:    gfn = F.normalize(gf, dim=1)
experiments/cargo_cvpb/airl_scale_diag.py:152:    sims = (qfn @ gfn.t()).numpy()             # cosine sim, higher = closer
experiments/cargo_cvpb/airl_scale_diag.py:226:    qf, qp, qc, qpaths = extract_features(model, loader(q_aerial), device)
experiments/cargo_cvpb/airl_scale_diag.py:228:    gf, gp, gc, _ = extract_features(model, loader(g_ground), device)
experiments/cargo_cvpb/airl_scale_diag.py:231:    full_map, full_cmc, full_minp = eval_market(qf, qp, qc, gf, gp, gc)
experiments/cargo_cvpb/airl_scale_diag.py:260:    conf, correct = per_query_top1(qf, gf, qp, qc, gp, gc)
experiments/cargo_cvpb/airl_scale_diag.py:276:        bqf = qf[sel]; bqp = qp[sel]; bqc = qc[sel]
experiments/cargo_cvpb/airl_scale_diag.py:277:        mAP, cmc, minp = eval_market(bqf, bqp, bqc, gf, gp, gc)
experiments/cross_view_cargo/cv_train.py:315:                gfeat = out['global_feat']
experiments/cross_view_cargo/cv_train.py:317:                loss_tri = tri(gfeat, labels)
experiments/cross_view_cargo/cv_train.py:324:                    loss_cv, cov = cv_tri(gfeat.float(), labels, views)
experiments/cargo_cvpb/hub_verify_p0c_deep.py:18:ZERO-TRAINING: cached features + numpy.
experiments/cargo_cvpb/hub_verify_p0c_deep.py:23:ap.add_argument('--cache_feat', required=True)
experiments/cargo_cvpb/hub_verify_p0c_deep.py:53:z=np.load(cli.cache_feat,allow_pickle=True)
experiments/cargo_cvpb/hub_verify_p0c_deep.py:54:qf=z['q_feat'].astype(np.float32); gf=z['g_feat'].astype(np.float32)
experiments/cargo_cvpb/hub_verify_p0c_deep.py:56:keep=g_pid!=-1; gf,g_pid,g_cam=gf[keep],g_pid[keep],g_cam[keep]
experiments/cargo_cvpb/hub_verify_p0c_deep.py:57:qf/=(np.linalg.norm(qf,axis=1,keepdims=True)+1e-12); gf/=(np.linalg.norm(gf,axis=1,keepdims=True)+1e-12)
experiments/cargo_cvpb/hub_verify_p0c_deep.py:58:Nq,Ng=qf.shape[0],gf.shape[0]; km=cli.k_main
experiments/cargo_cvpb/hub_verify_p0c_deep.py:59:sim=qf@gf.T
experiments/cargo_cvpb/cvpb_lm_reid_train.py:12:        L_id   = mean_l [ CE(cls^l, y) + Triplet(gf^l, y) ]            (per-variant ReID)
experiments/cargo_cvpb/cvpb_lm_reid_train.py:13:        L_marg = -log[ mean_l softmax(cls^l)[y] ] + Triplet(mean_l gf^l, y)  (marginal lik.)
experiments/cargo_cvpb/cvpb_lm_reid_train.py:407:            gf = (out[1][0] if isinstance(out[1], (list, tuple)) else out[1]).float()    # [B*M, D]
experiments/cargo_cvpb/cvpb_lm_reid_train.py:408:            D = gf.shape[1]
experiments/cargo_cvpb/cvpb_lm_reid_train.py:412:            gf_bm = gf.view(B, M, D)
experiments/cargo_cvpb/cvpb_lm_reid_train.py:418:            L_tri = torch.stack([batch_hard_triplet(gf_bm[:, m], y, cli.margin)
experiments/cargo_cvpb/cvpb_lm_reid_train.py:437:                zb = torch.nn.functional.normalize(gf_bm, dim=-1)               # [B,M,D]
experiments/cargo_cvpb/cvpb_lm_reid_train.py:465:            gf_mean = gf_bm.mean(1)                          # [B,D]
experiments/cargo_cvpb/cvpb_lm_reid_train.py:466:            L_marg = -ll.mean() + batch_hard_triplet(gf_mean, y, cli.margin)
experiments/cargo_cvpb/cvpb_lm_reid_train.py:469:            z = torch.nn.functional.normalize(gf_bm, dim=-1)            # [B,M,D]
experiments/cargo_cvpb/cvpb_lm_reid_train.py:470:            z_mu = torch.nn.functional.normalize(gf_bm.mean(1), dim=-1).detach()  # [B,D]
experiments/cargo_cvpb/cvpb_lm_reid_train.py:481:            # not the slot index (now a random axis per slot), so the adversary label is meaningful.
experiments/cargo_cvpb/cvpb_lm_reid_train.py:484:                zr = GradReverse.apply(torch.nn.functional.normalize(gf, dim=-1), adv_lamb)
experiments/cargo_cvpb/cvpb_lpa_head.py:24:ap.add_argument('--cache_gallery', default='/tmp/g_lpa_head.npz')
experiments/cargo_cvpb/cvpb_lpa_head.py:29:            '--K', str(lpa.K), '--reuse_gallery', '--cache_gallery', lpa.cache_gallery]
experiments/cargo_cvpb/cvpb_lpa_head.py:101:if os.path.exists(lpa.cache_gallery):
experiments/cargo_cvpb/cvpb_lpa_head.py:102:    gf = np.load(lpa.cache_gallery, allow_pickle=True)['gf']
experiments/cargo_cvpb/cvpb_lpa_head.py:105:    gf = ext.feats_from_pil([ks._to_target_aspect(read_image(it[0])) for it in gits])
experiments/cargo_cvpb/cvpb_lpa_head.py:106:    np.savez(lpa.cache_gallery, gf=gf)
experiments/cargo_cvpb/cvpb_lpa_head.py:114:sim_q = fq @ gf.T                                              # [Nq,K,Ng]
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:557:    def tokens_from_cached_map(self):
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:566:            raise RuntimeError("OVLIHead: no cached layer4 map; run model "
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:567:                               "forward before tokens_from_cached_map().")
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:569:        # (the cached map may be fp16 under autocast).
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:733:    def acvp_neg_bias(self, gfeat, labels, views, neg, proto, inited,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:767:        B = gfeat.size(0)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:768:        # work in fp32 for the cos/sigmoid/clamp/log numerics (gfeat may be fp32
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:770:        z = gfeat.float()                                           # (B,D) L2-normed
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:812:    def loss(self, gfeat, tok, labels, views,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:817:        gfeat:(B,D) L2-normed global feature (gradient flows -> encoder).
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:840:        B = gfeat.size(0)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:841:        device = gfeat.device
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:843:        gsim = gfeat @ gfeat.t()                                   # (B,B)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:865:            z = gfeat.new_zeros(())
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:869:                self._acvp_stats = (z.detach(), gfeat.new_ones(()),
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:870:                                    gfeat.new_zeros((), dtype=torch.long))
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:887:                gfeat, labels, views, neg, acvp_proto, acvp_inited,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:930:        gfs, tks, pids, cams = [], [], [], []
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:936:            gf = model(imgs, view_idx=vidx)              # (b,D) L2-normed BN
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:937:            tok = ovli.tokens_from_cached_map()           # (b,K,Dp) L2-normed
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:938:            gfs.append(gf.cpu())
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:942:        if not gfs:
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:945:        return (torch.cat(gfs, 0), torch.cat(tks, 0),
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:996:            torch.cuda.empty_cache()
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1006:        qf, qt, qp, qc = extract(q)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1007:        gf, gt, gp, gc = extract(g)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1008:        if qf.numel() == 0 or gf.numel() == 0:
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1012:        qf = F.normalize(qf, dim=1)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1013:        gf = F.normalize(gf, dim=1)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1014:        gsim = (qf @ gf.t()).numpy()                      # (Nq,Ng) cosine
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1016:        gmap, gcmc, _ = eval_market(qf, qp, qc, gf, gp, gc)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:2032:                gfeat = out['global_feat']
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:2035:                loss_tri = tri(gfeat, labels)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:2072:            # the autocast forward already cached the fp16 layer4 map) keeps the
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:2082:                    tok = ovli.tokens_from_cached_map()          # (B,K,Dp) fp32
experiments/cargo_cvpb/cvpb_lrfd_probe.py:43:ap.add_argument('--cache_gallery', default='/tmp/g_lrfd.npz')
experiments/cargo_cvpb/cvpb_lrfd_probe.py:46:            '--K', str(cli.K), '--reuse_gallery', '--cache_gallery', cli.cache_gallery]
experiments/cargo_cvpb/cvpb_lrfd_probe.py:146:    qf, yq, cq = variant_feats(items('query'))
experiments/cargo_cvpb/cvpb_lrfd_probe.py:147:    gf, yg, cg = variant_feats(items('bounding_box_test'))
experiments/cargo_cvpb/cvpb_lrfd_probe.py:149:        zq, _ = net(torch.tensor(qf, device=DEV)); zq = zq.cpu().numpy()
experiments/cargo_cvpb/cvpb_lrfd_probe.py:150:        zg_all, _ = net(torch.tensor(gf, device=DEV)); zg = zg_all[:, 0].cpu().numpy()  # gallery canonical-0
experiments/cargo_cvpb/cvpb_lrfd_probe.py:151:        gf0 = F.normalize(torch.tensor(gf[:, 0], device=DEV), dim=-1).cpu().numpy()      # no-P gallery-0
experiments/cargo_cvpb/cvpb_lrfd_probe.py:153:        _, rq = net(torch.tensor(qf[:min(1000, len(qf))], device=DEV))
experiments/cargo_cvpb/cvpb_lrfd_probe.py:163:    r_u = ks.eval_map(-setscore(qf / (np.linalg.norm(qf, axis=-1, keepdims=True) + 1e-9), gf0),
experiments/cargo_cvpb/cvpb_lrfd_probe.py:167:                  qf[i] / (np.linalg.norm(qf[i], axis=-1, keepdims=True) + 1e-9),
experiments/cargo_cvpb/cvpb_lrfd_probe.py:168:                  qf[i] / (np.linalg.norm(qf[i], axis=-1, keepdims=True) + 1e-9))) for i in range(min(500, len(qf)))]))
experiments/cargo_cvpb/smoke_ovli_residual.py:140:def reference_loss(ovli, gfeat, tok, labels, views):
experiments/cargo_cvpb/smoke_ovli_residual.py:143:    B = gfeat.size(0)
experiments/cargo_cvpb/smoke_ovli_residual.py:144:    gsim = gfeat @ gfeat.t()
experiments/cargo_cvpb/smoke_ovli_residual.py:149:    eye = torch.eye(B, dtype=torch.bool, device=gfeat.device)
experiments/cargo_cvpb/smoke_ovli_residual.py:155:        z = gfeat.new_zeros(())
experiments/cargo_cvpb/smoke_ovli_residual.py:192:    return head.tokens_from_cached_map(), fmap
experiments/cargo_cvpb/smoke_ovli_residual.py:212:    gfeat = F.normalize(torch.randn(B, Dg), dim=1)
experiments/cargo_cvpb/smoke_ovli_residual.py:281:        l_new, p_new, n_new = head.loss(gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_residual.py:282:        l_ref, p_ref, n_ref = reference_loss(head, gfeat, tok, labels, views)
experiments/cargo_cvpb/smoke_ovli_residual.py:340:        l, _, _ = head.loss(gfeat, tok_g, labels, views)
experiments/cargo_cvpb/smoke_ovli_residual.py:415:        tok_r = head_r.tokens_from_cached_map()
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:337:            _gfeat, outs = self.swin(x)

exec
/bin/zsh -lc "rg -n \"exp367|single-support|CVaR|cvpb_ss_cvar\" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md experiments/exp367_single_support -g'*.md'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp367_single_support/monitor.md:1:# exp367 Single-Support CVaR — monitor
experiments/exp367_single_support/monitor.md:12:★v1 污染（codex 审抓）：single-support 跳无 positive query（比不同子集）+ distractor 压 1 张（负样本池变）→ best/random>full 假象。
experiments/exp367_single_support/monitor.md:13:★v2 修（codex 3 High）：common-valid query 共用 + distractor 全量 + 20 seed + missing 记 0。**single-support 都 <full（合理少正样本），best-worst 12.27 + random-worst 9.54，false10 best0.923≈worst0.927 → gap 不被 #false-in-topk 解释**。
experiments/exp367_single_support/monitor.md:21:support 选择有 oracle headroom（best-worst 12.27，不被 #false 解释），单图 support representation 是真训练瓶颈。**诚实标注**：best/worst 用 query-label oracle 上下界，证 headroom 存在；训练能否学到（不用 query）要 Single-Support CVaR train 验。
experiments/exp367_single_support/monitor.md:25:codex 调研 Single-Support CVaR 训练设计 + novelty 确认（63517）：episodic 单图 support + CVaR worst-case 如何写 loss、避六点定律陷阱（不塑造/对齐/压缩变体）、cheap 验证路径（frozen head 小训 or full FT）。GO 则训练侧创新动手，full fine-tune 前 codex 三审 diff。
experiments/exp367_single_support/design.md:1:# exp367 Single-Support CVaR Episodic Loss（训练侧创新，2026-06-28）
experiments/exp367_single_support/design.md:5:用户 goal：找训练侧创新发 CCF-B，不收手，不轻易说穷尽，审查调研交 codex，严谨，文档记好。codex 训练侧深度调研 #1（最务实）：训练时每 ID 只用单图 support 定义身份，对 worst-support 选择做 CVaR 优化。回应 exp109 根问题（single-image support incomplete）。**纯训练侧**（episodic loss，输出常规 descriptor），严格非 test-time/检索侧/范式重定义。
experiments/exp367_single_support/design.md:11:ReID 训练用 multi-shot gallery（每 ID 多图），但模型学到的身份边界可能依赖"见过该 ID 多个 view"。部署常 single-shot（单图 support 定义新身份）。训练时**强制单图 support + CVaR worst-support 优化**，逼模型学"从任意单图恢复完整身份边界"的鲁棒表征，而非依赖 multi-view 平均。
experiments/exp367_single_support/design.md:19:- worst-support：每 ID 选最差单图（CVaR worst-case 目标针对的）
experiments/exp367_single_support/design.md:21:**GO**（support 选择是真训练瓶颈）：worst 比 full 掉 > 3 mAP 且 **best−worst gap > 3 mAP**（哪张 support 图很重要 = support 选择 matters，值得 CVaR 优化）。
experiments/exp367_single_support/design.md:32:- GO → 设计 Single-Support CVaR episodic loss 训练（每 ID 单图 support + worst-case 风险优化），训练侧第一 contribution，full fine-tune 前 codex 三审 diff。
experiments/exp367_single_support/design.md:37:★**novelty 真空白（codex 确认）**：2024-26 标准监督 person ReID 没有"episodic single-support training + CVaR worst-support tail optimization"直接先例（检索 single-support/worst-support/CVaR-ReID/support-selection 都没命中）。邻近但不同：CFReID(continual few-shot)/DG-episodic(domain-invariant)/ProtoNet(novel-class 优化 prototype 平均非 tail)/batch-hard(hard pair mining 非 support tail)。**claim 写窄**：不发明 episodic/CVaR，是"标准 ReID 优化单图 support 定义身份的 tail risk"。
experiments/exp367_single_support/design.md:39:★**训练设计（two-level CVaR，加项不替换）**：
experiments/exp367_single_support/design.md:42:- `L_cvar_y = CVaR_α(support risks)`，`L_ss_cvar = mean_y`。α=0.7/0.8。
experiments/exp367_single_support/design.md:44:- support/query 都梯度 + 保 CE+Triplet 防 collapse。两级聚合(support risk→CVaR)非 CVaR over all pairs(避 batch-hard)。
experiments/exp367_single_support/design.md:50:★**风险+对照（防退化 hard-mining）**：报 batch-hard/pair-CVaR/random episodic CE 三对照，只 support-level CVaR 独立赢才站得住。报 missing-positive/cross-cam 覆盖/false10 random std。
experiments/exp367_single_support/design.md:56:probe v2 GO（best-worst 12.27 不被 #false 解释，codex 两轮审）。codex 训练设计 GO（novelty 空白 6.5/10）。下一步：写 frozen head smoke 训练（cheap 第一步，复用 Market cache + projection head + episodic CVaR loss）+ codex 三审 diff（full FT 铁律 + 审查交 codex）。
experiments/exp367_single_support/codex_review2.md:14:再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review2.md:914:上下文里最新主线已经不是早期 visibility，而是 single-support / partial evidence 这类诊断问题。我现在直接看 exp367 的设计、监控和脚本实现，重点查四个你列的修复点是否真正落到代码路径里。
experiments/exp367_single_support/codex_review2.md:916:/bin/zsh -lc 'ls -la experiments/exp367_single_support && rg -n "common|valid|distractor|support|seed|false|missing|positive|top" experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review2.md:925:4:codex 训练侧 #1: 训练时每 ID 单图 support 定义身份, CVaR worst-support 优化。回应 exp109
experiments/exp367_single_support/codex_review2.md:928:9:  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_review2.md:932:17:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review2.md:934:35:print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review2.md:937:43:print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_review2.md:952:75:print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_review2.md:969:/bin/zsh -lc "nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py | sed -n '261,560p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review2.md:971:/bin/zsh -lc 'for f in experiments/exp367_single_support/design.md experiments/exp367_single_support/monitor.md experiments/exp367_single_support/claude_review.md; do [ -f "$f" ] && printf '"'%s\\n' \""'$f" && sed -n '"'1,220p' \""'$f"; done' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review2.md:974:experiments/exp367_single_support/design.md
experiments/exp367_single_support/codex_review2.md:975:# exp367 Single-Support CVaR Episodic Loss（训练侧创新，2026-06-28）
experiments/exp367_single_support/codex_review2.md:979:用户 goal：找训练侧创新发 CCF-B，不收手，不轻易说穷尽，审查调研交 codex，严谨，文档记好。codex 训练侧深度调研 #1（最务实）：训练时每 ID 只用单图 support 定义身份，对 worst-support 选择做 CVaR 优化。回应 exp109 根问题（single-image support incomplete）。**纯训练侧**（episodic loss，输出常规 descriptor），严格非 test-time/检索侧/范式重定义。
experiments/exp367_single_support/codex_review2.md:985:ReID 训练用 multi-shot gallery（每 ID 多图），但模型学到的身份边界可能依赖"见过该 ID 多个 view"。部署常 single-shot（单图 support 定义新身份）。训练时**强制单图 support + CVaR worst-support 优化**，逼模型学"从任意单图恢复完整身份边界"的鲁棒表征，而非依赖 multi-view 平均。
experiments/exp367_single_support/codex_review2.md:993:- worst-support：每 ID 选最差单图（CVaR worst-case 目标针对的）
experiments/exp367_single_support/codex_review2.md:995:**GO**（support 选择是真训练瓶颈）：worst 比 full 掉 > 3 mAP 且 **best−worst gap > 3 mAP**（哪张 support 图很重要 = support 选择 matters，值得 CVaR 优化）。
experiments/exp367_single_support/codex_review2.md:1006:- GO → 设计 Single-Support CVaR episodic loss 训练（每 ID 单图 support + worst-case 风险优化），训练侧第一 contribution，full fine-tune 前 codex 三审 diff。
experiments/exp367_single_support/codex_review2.md:1014:/bin/zsh -lc "nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py | sed -n '1,260p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review2.md:1019:     4	codex 训练侧 #1: 训练时每 ID 单图 support 定义身份, CVaR worst-support 优化。回应 exp109
experiments/exp367_single_support/codex_review2.md:1024:     9	  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_review2.md:1032:    17	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review2.md:1050:    35	print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review2.md:1058:    43	print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_review2.md:1090:    75	print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_review2.md:1125:   110	print('  ※ best/worst 是 query-label oracle 诊断上下界, 证 headroom 存在; 训练能否学到要 CVaR train 验', flush=True)
experiments/exp367_single_support/codex_review2.md:1131:/bin/zsh -lc "sed -n '1,220p' experiments/exp367_single_support/codex_review2.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review2.md:1146:再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review2.md:1355:/bin/zsh -lc 'rg -n "High|false10|random|common-valid|distractor|missing|verdict|needs|approve|trivial|污染" experiments/exp367_single_support/codex_review2.md' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review2.md:1357:14:再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review2.md:1362:916:/bin/zsh -lc 'ls -la experiments/exp367_single_support && rg -n "common|valid|distractor|support|seed|false|missing|positive|top" experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review2.md:1364:928:9:  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_review2.md:1370:937:43:print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_review2.md:1378:952:75:print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_review2.md:1392:1024:     9	  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_review2.md:1398:1058:    43	print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_review2.md:1406:1090:    75	print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_review2.md:1420:/bin/zsh -lc 'git status --short && git diff -- experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review2.md:1619:?? experiments/exp367_single_support/
experiments/exp367_single_support/codex_review2.md:1640:(d) **对**。single-support 下某 query 没有可用 positive 时记 AP/R1=0 是合理的，能惩罚 support 选到同 camera 或无法覆盖 cross-camera positive 的情况。
experiments/exp367_single_support/codex_review2.md:1655:(d) **对**。single-support 下某 query 没有可用 positive 时记 AP/R1=0 是合理的，能惩罚 support 选到同 camera 或无法覆盖 cross-camera positive 的情况。
experiments/exp367_single_support/codex_review_smoke.md:14:审查 experiments/exp367_single_support/cvpb_ss_cvar_smoke.py(Single-Support CVaR frozen-head smoke 训练, codex 设计的 two-level CVaR)。逐行查 bug + 训练逻辑。审: (a)episodic single-support CVaR loss 对吗——episode N id×K 图, 每 support-slot 当 prototype 其余 query 分类到 N proto 的 CE, per-id K 个 support risk 做 CVaR_α worst tail, L=mean_CE+lam·CVaR? (b)和 codex 设计(two-level: support risk→CVaR per id)一致吗? (c)random mode(无 CVaR)对照对吗? (d)评估 projected query/gallery→full/random/worst-support 对吗(复用 probe v2 口径)? (e)会退化普通 hard-mining 吗? frozen head(init eye)+backbone 冻对吗? loss 数值稳定(topk/CE shape)? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review_smoke.md:16:我先按仓库规则补齐当前研究上下文，然后直接看 `exp367_single_support` 的设计和 smoke 脚本。重点会放在 loss 形状、episode 采样、CVaR 定义、random 对照和 eval 口径是否真的对应设计。
experiments/exp367_single_support/codex_review_smoke.md:914:上下文看完，最新主线已经转到 single-support / 不完整证据问题。下面我会把 `design.md` 和脚本按行号对齐看，必要时查它调用的 probe/eval 口径，避免只凭函数名判断。
experiments/exp367_single_support/codex_review_smoke.md:916:/bin/zsh -lc "nl -ba experiments/exp367_single_support/cvpb_ss_cvar_smoke.py | sed -n '1,260p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review_smoke.md:919:     2	"""Single-Support CVaR frozen-head smoke — codex cheap 路径 #1 (不动 backbone).
experiments/exp367_single_support/codex_review_smoke.md:921:     4	冻 backbone, 只训 projection head, 用 episodic single-support CVaR loss。验机制方向:
experiments/exp367_single_support/codex_review_smoke.md:924:     7	对照(防退化 hard-mining): --mode random (episodic CE 无 CVaR), 证不是 episode 本身涨。
experiments/exp367_single_support/codex_review_smoke.md:926:     9	训练设计(codex two-level CVaR):
experiments/exp367_single_support/codex_review_smoke.md:928:    11	  对每 id 的 K 个 support risk 做 CVaR_α(worst tail); L = mean_CE + lam·mean_CVaR。
experiments/exp367_single_support/codex_review_smoke.md:929:    12	  (加项不替换主任务; frozen smoke 先只 CE+CVaR, full FT 再加 triplet。)
experiments/exp367_single_support/codex_review_smoke.md:931:    14	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_ss_cvar_smoke.py
experiments/exp367_single_support/codex_review_smoke.md:945:    28	ap.add_argument('--alpha', type=float, default=0.7) # CVaR tail
experiments/exp367_single_support/codex_review_smoke.md:946:    29	ap.add_argument('--lam', type=float, default=0.3)   # CVaR weight
experiments/exp367_single_support/codex_review_smoke.md:986:    69	    id_risks = []                                   # [N] each id 的 CVaR over K support
experiments/exp367_single_support/codex_review_smoke.md:999:    82	        return l_ce                                  # 对照: 无 CVaR
experiments/exp367_single_support/codex_review_smoke.md:1000:    83	    # CVaR_alpha over K support risks per id (worst tail)
experiments/exp367_single_support/codex_review_smoke.md:1013:    96	# ---- 3. 评估: projected query/gallery → full / single-support diagnostic ----
experiments/exp367_single_support/codex_review_smoke.md:1059:/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review_smoke.md:1061:# exp367 Single-Support CVaR Episodic Loss（训练侧创新，2026-06-28）
experiments/exp367_single_support/codex_review_smoke.md:1065:用户 goal：找训练侧创新发 CCF-B，不收手，不轻易说穷尽，审查调研交 codex，严谨，文档记好。codex 训练侧深度调研 #1（最务实）：训练时每 ID 只用单图 support 定义身份，对 worst-support 选择做 CVaR 优化。回应 exp109 根问题（single-image support incomplete）。**纯训练侧**（episodic loss，输出常规 descriptor），严格非 test-time/检索侧/范式重定义。
experiments/exp367_single_support/codex_review_smoke.md:1071:ReID 训练用 multi-shot gallery（每 ID 多图），但模型学到的身份边界可能依赖"见过该 ID 多个 view"。部署常 single-shot（单图 support 定义新身份）。训练时**强制单图 support + CVaR worst-support 优化**，逼模型学"从任意单图恢复完整身份边界"的鲁棒表征，而非依赖 multi-view 平均。
experiments/exp367_single_support/codex_review_smoke.md:1079:- worst-support：每 ID 选最差单图（CVaR worst-case 目标针对的）
experiments/exp367_single_support/codex_review_smoke.md:1081:**GO**（support 选择是真训练瓶颈）：worst 比 full 掉 > 3 mAP 且 **best−worst gap > 3 mAP**（哪张 support 图很重要 = support 选择 matters，值得 CVaR 优化）。
experiments/exp367_single_support/codex_review_smoke.md:1092:- GO → 设计 Single-Support CVaR episodic loss 训练（每 ID 单图 support + worst-case 风险优化），训练侧第一 contribution，full fine-tune 前 codex 三审 diff。
experiments/exp367_single_support/codex_review_smoke.md:1097:★**novelty 真空白（codex 确认）**：2024-26 标准监督 person ReID 没有"episodic single-support training + CVaR worst-support tail optimization"直接先例（检索 single-support/worst-support/CVaR-ReID/support-selection 都没命中）。邻近但不同：CFReID(continual few-shot)/DG-episodic(domain-invariant)/ProtoNet(novel-class 优化 prototype 平均非 tail)/batch-hard(hard pair mining 非 support tail)。**claim 写窄**：不发明 episodic/CVaR，是"标准 ReID 优化单图 support 定义身份的 tail risk"。
experiments/exp367_single_support/codex_review_smoke.md:1099:★**训练设计（two-level CVaR，加项不替换）**：
experiments/exp367_single_support/codex_review_smoke.md:1102:- `L_cvar_y = CVaR_α(support risks)`，`L_ss_cvar = mean_y`。α=0.7/0.8。
experiments/exp367_single_support/codex_review_smoke.md:1104:- support/query 都梯度 + 保 CE+Triplet 防 collapse。两级聚合(support risk→CVaR)非 CVaR over all pairs(避 batch-hard)。
experiments/exp367_single_support/codex_review_smoke.md:1110:★**风险+对照（防退化 hard-mining）**：报 batch-hard/pair-CVaR/random episodic CE 三对照，只 support-level CVaR 独立赢才站得住。报 missing-positive/cross-cam 覆盖/false10 random std。
experiments/exp367_single_support/codex_review_smoke.md:1116:probe v2 GO（best-worst 12.27 不被 #false 解释，codex 两轮审）。codex 训练设计 GO（novelty 空白 6.5/10）。下一步：写 frozen head smoke 训练（cheap 第一步，复用 Market cache + projection head + episodic CVaR loss）+ codex 三审 diff（full FT 铁律 + 审查交 codex）。
experiments/exp367_single_support/codex_review_smoke.md:1119:/bin/zsh -lc 'rg --files experiments/exp367_single_support' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review_smoke.md:1121:experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/codex_review_smoke.md:1122:experiments/exp367_single_support/design.md
experiments/exp367_single_support/codex_review_smoke.md:1123:experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review_smoke.md:1124:experiments/exp367_single_support/codex_review_smoke.md
experiments/exp367_single_support/codex_review_smoke.md:1125:experiments/exp367_single_support/monitor.md
experiments/exp367_single_support/codex_review_smoke.md:1126:experiments/exp367_single_support/codex_train_design.md
experiments/exp367_single_support/codex_review_smoke.md:1127:experiments/exp367_single_support/codex_review2.md
experiments/exp367_single_support/codex_review_smoke.md:1128:experiments/exp367_single_support/cvpb_ss_cvar_smoke.py
experiments/exp367_single_support/codex_review_smoke.md:1131:/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review_smoke.md:1133:# exp367 Single-Support CVaR — monitor
experiments/exp367_single_support/codex_review_smoke.md:1144:★v1 污染（codex 审抓）：single-support 跳无 positive query（比不同子集）+ distractor 压 1 张（负样本池变）→ best/random>full 假象。
experiments/exp367_single_support/codex_review_smoke.md:1145:★v2 修（codex 3 High）：common-valid query 共用 + distractor 全量 + 20 seed + missing 记 0。**single-support 都 <full（合理少正样本），best-worst 12.27 + random-worst 9.54，false10 best0.923≈worst0.927 → gap 不被 #false-in-topk 解释**。
experiments/exp367_single_support/codex_review_smoke.md:1153:support 选择有 oracle headroom（best-worst 12.27，不被 #false 解释），单图 support representation 是真训练瓶颈。**诚实标注**：best/worst 用 query-label oracle 上下界，证 headroom 存在；训练能否学到（不用 query）要 Single-Support CVaR train 验。
experiments/exp367_single_support/codex_review_smoke.md:1157:codex 调研 Single-Support CVaR 训练设计 + novelty 确认（63517）：episodic 单图 support + CVaR worst-case 如何写 loss、避六点定律陷阱（不塑造/对齐/压缩变体）、cheap 验证路径（frozen head 小训 or full FT）。GO 则训练侧创新动手，full fine-tune 前 codex 三审 diff。
experiments/exp367_single_support/codex_review_smoke.md:1162:/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/codex_review_smoke.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review_smoke.md:1177:审查 experiments/exp367_single_support/cvpb_ss_cvar_smoke.py(Single-Support CVaR frozen-head smoke 训练, codex 设计的 two-level CVaR)。逐行查 bug + 训练逻辑。审: (a)episodic single-support CVaR loss 对吗——episode N id×K 图, 每 support-slot 当 prototype 其余 query 分类到 N proto 的 CE, per-id K 个 support risk 做 CVaR_α worst tail, L=mean_CE+lam·CVaR? (b)和 codex 设计(two-level: support risk→CVaR per id)一致吗? (c)random mode(无 CVaR)对照对吗? (d)评估 projected query/gallery→full/random/worst-support 对吗(复用 probe v2 口径)? (e)会退化普通 hard-mining 吗? frozen head(init eye)+backbone 冻对吗? loss 数值稳定(topk/CE shape)? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review_smoke.md:1179:我先按仓库规则补齐当前研究上下文，然后直接看 `exp367_single_support` 的设计和 smoke 脚本。重点会放在 loss 形状、episode 采样、CVaR 定义、random 对照和 eval 口径是否真的对应设计。
experiments/exp367_single_support/codex_review_smoke.md:1426:/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/codex_train_design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review_smoke.md:1441:Single-Support CVaR ReID 训练侧创新, cheap probe 已 GO(frozen SOLIDER: support 选择 oracle headroom best-worst 12.27 / random-worst 9.54, false10 best≈worst 不被 #false-in-topk 解释; best/worst 用 query-label oracle 上下界)。现在要**设计训练**(把 oracle headroom 学成不用 query 的鲁棒表征)。**深度调研+设计**: (a)**novelty 严格确认**: 'episodic single-support training + CVaR worst-support 优化' for standard person ReID, 2024-26 web search 有直接先例吗? few-shot ReID(CFReID/ReMix)/episodic metric learning/DRO-ReID/hard-example mining 区别在哪? 这个角度真空白吗还是撞 episodic few-shot? (b)**训练设计**: 怎么把'每 ID 单图 support 定义身份 + worst-case 鲁棒'写成 loss? episodic batch 采样(每 ID 1 support + 多 query)? CVaR over worst support-query pairs? 和标准 triplet+CE 怎么结合(加项还是替换)? 避六点定律陷阱(不塑造/对齐/压缩变体)? (c)**cheap 验证路径**: 先 frozen backbone + 单 support episodic head 小训(不动 backbone)能否提升 worst-support? 还是必须 full fine-tune? 给最小训练验证(1 单卡半天)。(d)风险: 会不会退化成普通 hard-mining / 普通 single-shot eval? 给设计+novelty 判定+CCF-B 潜力分。务实中文严谨。
experiments/exp367_single_support/codex_review_smoke.md:1690:/bin/zsh -lc "nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py | sed -n '1,260p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review_smoke.md:1692:/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/codex_review2.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review_smoke.md:1697:     4	codex 训练侧 #1: 训练时每 ID 单图 support 定义身份, CVaR worst-support 优化。回应 exp109
experiments/exp367_single_support/codex_review_smoke.md:1702:     9	  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_review_smoke.md:1710:    17	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review_smoke.md:1728:    35	print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review_smoke.md:1736:    43	print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_review_smoke.md:1768:    75	print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_review_smoke.md:1803:   110	print('  ※ best/worst 是 query-label oracle 诊断上下界, 证 headroom 存在; 训练能否学到要 CVaR train 验', flush=True)
experiments/exp367_single_support/codex_review_smoke.md:1820:再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review_smoke.md:2069:对照下来，训练部分基本是 two-level CVaR 的骨架，但我看到几个会影响解释强度的点：`random` 对照并不是真“随机 support”训练，只是同一个 episode CE 去掉 CVaR；eval 也只算 mAP，不含 probe v2 的 R1/false10。这些不一定阻塞 smoke，但要在 verdict 里区分“可跑”和“证据不完整”。
experiments/exp367_single_support/codex_train_design.md:14:Single-Support CVaR ReID 训练侧创新, cheap probe 已 GO(frozen SOLIDER: support 选择 oracle headroom best-worst 12.27 / random-worst 9.54, false10 best≈worst 不被 #false-in-topk 解释; best/worst 用 query-label oracle 上下界)。现在要**设计训练**(把 oracle headroom 学成不用 query 的鲁棒表征)。**深度调研+设计**: (a)**novelty 严格确认**: 'episodic single-support training + CVaR worst-support 优化' for standard person ReID, 2024-26 web search 有直接先例吗? few-shot ReID(CFReID/ReMix)/episodic metric learning/DRO-ReID/hard-example mining 区别在哪? 这个角度真空白吗还是撞 episodic few-shot? (b)**训练设计**: 怎么把'每 ID 单图 support 定义身份 + worst-case 鲁棒'写成 loss? episodic batch 采样(每 ID 1 support + 多 query)? CVaR over worst support-query pairs? 和标准 triplet+CE 怎么结合(加项还是替换)? 避六点定律陷阱(不塑造/对齐/压缩变体)? (c)**cheap 验证路径**: 先 frozen backbone + 单 support episodic head 小训(不动 backbone)能否提升 worst-support? 还是必须 full fine-tune? 给最小训练验证(1 单卡半天)。(d)风险: 会不会退化成普通 hard-mining / 普通 single-shot eval? 给设计+novelty 判定+CCF-B 潜力分。务实中文严谨。
experiments/exp367_single_support/codex_train_design.md:1054:本地文档里主线已经推进到多阶段/多分支，但你现在给的是一个新的训练侧问题定义：把 single-support oracle headroom 训练成无需 query 的鲁棒表征。我会用 exp030a/现有 strongest scaffold 作为参照，但判新意时只按“standard person ReID 训练目标”来审，不把 test-time trick 当贡献。
experiments/exp367_single_support/codex_train_design.md:1135:experiments/exp367_single_support/design.md
experiments/exp367_single_support/codex_train_design.md:1339:?? experiments/exp367_single_support/
experiments/exp367_single_support/codex_train_design.md:1346:/bin/zsh -lc 'rg -n "support|single|CVaR|oracle|worst|headroom|cheap|exp109|exp148|exp149|exp15[0-9]|exp20[0-9]" experiments -S' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:1975:experiments/exp367_single_support/codex_review2.md:14:再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:1976:experiments/exp367_single_support/codex_review2.md:16:我先按仓库接手规则读最新实验上下文，再聚焦审 `cvpb_single_support_probe.py` 的 v2 逻辑和潜在泄漏点。
experiments/exp367_single_support/codex_train_design.md:1977:experiments/exp367_single_support/codex_review2.md:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:1978:experiments/exp367_single_support/codex_review2.md:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:1979:experiments/exp367_single_support/codex_review2.md:711:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/exp367_single_support/codex_train_design.md:1980:experiments/exp367_single_support/codex_review2.md:778:   - 写法：test-time supporting evaluations
experiments/exp367_single_support/codex_train_design.md:1981:experiments/exp367_single_support/codex_review2.md:914:上下文里最新主线已经不是早期 visibility，而是 single-support / partial evidence 这类诊断问题。我现在直接看 exp367 的设计、监控和脚本实现，重点查四个你列的修复点是否真正落到代码路径里。
experiments/exp367_single_support/codex_train_design.md:1982:experiments/exp367_single_support/codex_review2.md:916:/bin/zsh -lc 'ls -la experiments/exp367_single_support && rg -n "common|valid|distractor|support|seed|false|missing|positive|top" experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:1983:experiments/exp367_single_support/codex_review2.md:923:-rw-r--r--@   1 abdslm  staff     5895 Jun 28 03:54 cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:1984:experiments/exp367_single_support/codex_review2.md:925:4:codex 训练侧 #1: 训练时每 ID 单图 support 定义身份, CVaR worst-support 优化。回应 exp109
experiments/exp367_single_support/codex_train_design.md:1985:experiments/exp367_single_support/codex_review2.md:926:5:根问题(single-image support incomplete)。纯训练侧(episodic loss 输出常规 descriptor)。
experiments/exp367_single_support/codex_train_design.md:1986:experiments/exp367_single_support/codex_review2.md:927:8:  1. common-valid query mask: 用 full-gallery 下有 positive 的 query 子集, 所有 support 设置同子集(否则比不同难度)。
experiments/exp367_single_support/codex_train_design.md:1987:experiments/exp367_single_support/codex_review2.md:928:9:  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_train_design.md:1988:experiments/exp367_single_support/codex_review2.md:929:10:  3. 主判据 best-random / random-worst 多 seed(20) 均值±std; 报 #false-in-topk(top10 错样本数)。
experiments/exp367_single_support/codex_train_design.md:1989:experiments/exp367_single_support/codex_review2.md:930:14:GO(support 选择是真训练瓶颈): random-worst gap > 3 mAP(多 seed 稳, 同负样本池同 valid query) AND
experiments/exp367_single_support/codex_train_design.md:1990:experiments/exp367_single_support/codex_review2.md:932:17:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:1991:experiments/exp367_single_support/codex_review2.md:934:35:print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_train_design.md:1992:experiments/exp367_single_support/codex_review2.md:937:43:print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:1993:experiments/exp367_single_support/codex_review2.md:938:46:def support_gallery(support_idx_per_id):
experiments/exp367_single_support/codex_train_design.md:1994:experiments/exp367_single_support/codex_review2.md:939:47:    """has-query ID 用单 support, distractor 全量 → 负样本池不变。"""
experiments/exp367_single_support/codex_train_design.md:1995:experiments/exp367_single_support/codex_review2.md:940:48:    return np.concatenate([np.array(support_idx_per_id, dtype=int), distractor_g])
experiments/exp367_single_support/codex_train_design.md:1996:experiments/exp367_single_support/codex_review2.md:948:68:# common-valid query: full-gallery 下有 positive 的 query (固定子集, 所有 support 设置共用)
experiments/exp367_single_support/codex_train_design.md:1997:experiments/exp367_single_support/codex_review2.md:952:75:print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:1998:experiments/exp367_single_support/codex_review2.md:954:80:# best/worst-support oracle (用 query-label, 诊断上下界 — 诚实: 不证训练可学)
experiments/exp367_single_support/codex_train_design.md:1999:experiments/exp367_single_support/codex_review2.md:955:90:best_mAP, best_R1, best_f10 = eval_fixed(support_gallery(best_s), valid_q)
experiments/exp367_single_support/codex_train_design.md:2000:experiments/exp367_single_support/codex_review2.md:956:91:worst_mAP, worst_R1, worst_f10 = eval_fixed(support_gallery(worst_s), valid_q)
experiments/exp367_single_support/codex_train_design.md:2001:experiments/exp367_single_support/codex_review2.md:957:93:# random-support 多 seed
experiments/exp367_single_support/codex_train_design.md:2002:experiments/exp367_single_support/codex_review2.md:959:98:    rand_mAPs.append(eval_fixed(support_gallery(rs), valid_q)[0])
experiments/exp367_single_support/codex_train_design.md:2003:experiments/exp367_single_support/codex_review2.md:962:103:print(f'  best-support   : mAP={best_mAP:.2f} (oracle 上界, 用 query-label)  false10={best_f10:.3f}')
experiments/exp367_single_support/codex_train_design.md:2004:experiments/exp367_single_support/codex_review2.md:963:104:print(f'  random-support : mAP={rand_mean:.2f}±{rand_std:.2f}')
experiments/exp367_single_support/codex_train_design.md:2005:experiments/exp367_single_support/codex_review2.md:964:105:print(f'  worst-support  : mAP={worst_mAP:.2f} (oracle 下界)  false10={worst_f10:.3f}')
experiments/exp367_single_support/codex_train_design.md:2006:experiments/exp367_single_support/codex_review2.md:965:107:# 主判据: random-worst gap(同负样本池同valid query, 单support内选择) > 3 且 false10 同向变化(非trivial少正样本)
experiments/exp367_single_support/codex_train_design.md:2007:experiments/exp367_single_support/codex_review2.md:966:109:print(f'  [verdict] {"GO (单 support 内选择 matters, support representation 是真训练瓶颈)" if go else "DEAD (support 选择价值小/被负样本池-valid-query 解释)"}')
experiments/exp367_single_support/codex_train_design.md:2008:experiments/exp367_single_support/codex_review2.md:969:/bin/zsh -lc "nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py | sed -n '261,560p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2009:experiments/exp367_single_support/codex_review2.md:971:/bin/zsh -lc 'for f in experiments/exp367_single_support/design.md experiments/exp367_single_support/monitor.md experiments/exp367_single_support/claude_review.md; do [ -f "$f" ] && printf '"'%s\\n' \""'$f" && sed -n '"'1,220p' \""'$f"; done' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2010:experiments/exp367_single_support/codex_review2.md:974:experiments/exp367_single_support/design.md
experiments/exp367_single_support/codex_train_design.md:2011:experiments/exp367_single_support/codex_review2.md:975:# exp367 Single-Support CVaR Episodic Loss（训练侧创新，2026-06-28）
experiments/exp367_single_support/codex_train_design.md:2012:experiments/exp367_single_support/codex_review2.md:979:用户 goal：找训练侧创新发 CCF-B，不收手，不轻易说穷尽，审查调研交 codex，严谨，文档记好。codex 训练侧深度调研 #1（最务实）：训练时每 ID 只用单图 support 定义身份，对 worst-support 选择做 CVaR 优化。回应 exp109 根问题（single-image support incomplete）。**纯训练侧**（episodic loss，输出常规 descriptor），严格非 test-time/检索侧/范式重定义。
experiments/exp367_single_support/codex_train_design.md:2013:experiments/exp367_single_support/codex_review2.md:981:codex 已避所有死区（visibility/masked/CLIP-align/synthetic/topology/DG-foundation/noisy-label/long-tail），2024-26 novelty 空白：few-shot/DG 有先例，但**标准 Market/MSMT/Occluded 监督训练里"单图 support 是否足够定义身份"做成主训练目标，2024-2026 没看到直接占位**。
experiments/exp367_single_support/codex_train_design.md:2014:experiments/exp367_single_support/codex_review2.md:985:ReID 训练用 multi-shot gallery（每 ID 多图），但模型学到的身份边界可能依赖"见过该 ID 多个 view"。部署常 single-shot（单图 support 定义新身份）。训练时**强制单图 support + CVaR worst-support 优化**，逼模型学"从任意单图恢复完整身份边界"的鲁棒表征，而非依赖 multi-view 平均。
experiments/exp367_single_support/codex_train_design.md:2015:experiments/exp367_single_support/codex_review2.md:987:## cheap kill-switch（零训练，cvpb_single_support_probe.py）
experiments/exp367_single_support/codex_train_design.md:2016:experiments/exp367_single_support/codex_review2.md:991:- best-support：每 ID 选最好单图（同 ID query 平均 sim 最高，oracle 上界）
experiments/exp367_single_support/codex_train_design.md:2017:experiments/exp367_single_support/codex_review2.md:992:- random-support：每 ID 随机 1 图
experiments/exp367_single_support/codex_train_design.md:2018:experiments/exp367_single_support/codex_review2.md:993:- worst-support：每 ID 选最差单图（CVaR worst-case 目标针对的）
experiments/exp367_single_support/codex_train_design.md:2019:experiments/exp367_single_support/codex_review2.md:995:**GO**（support 选择是真训练瓶颈）：worst 比 full 掉 > 3 mAP 且 **best−worst gap > 3 mAP**（哪张 support 图很重要 = support 选择 matters，值得 CVaR 优化）。
experiments/exp367_single_support/codex_train_design.md:2020:experiments/exp367_single_support/codex_review2.md:996:**DEAD**：best≈worst（哪张 support 都一样，没 support 选择价值）或 single≈full（单图够）。
experiments/exp367_single_support/codex_train_design.md:2021:experiments/exp367_single_support/codex_review2.md:998:★诚实设计要点：单图 vs 多图必掉 mAP（少正样本）是 trivial，所以**关键判据是 best−worst gap**（同样单图，选择重不重要），不是 single<full。codex 审 probe 验这个设计是否真有意义（用户要审查交 codex）。
experiments/exp367_single_support/codex_train_design.md:2022:experiments/exp367_single_support/codex_review2.md:1002:codex 审 probe（codex_review.md）：kill-switch 设计是否有意义、best/worst per-ID 选择逻辑、#false-in-topk 控制。
experiments/exp367_single_support/codex_train_design.md:2023:experiments/exp367_single_support/codex_review2.md:1006:- GO → 设计 Single-Support CVaR episodic loss 训练（每 ID 单图 support + worst-case 风险优化），训练侧第一 contribution，full fine-tune 前 codex 三审 diff。
experiments/exp367_single_support/codex_train_design.md:2024:experiments/exp367_single_support/codex_review2.md:1007:- DEAD → support 选择无训练价值，转 Equivariant Routing（codex 训练侧 #2，routing 等变非 embedding 一致）。
experiments/exp367_single_support/codex_train_design.md:2025:experiments/exp367_single_support/codex_review2.md:1014:/bin/zsh -lc "nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py | sed -n '1,260p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2026:experiments/exp367_single_support/codex_review2.md:1017:     2	"""Single-Support ReID — cheap kill-switch (零训练) — v2 (codex needs-attention 修).
experiments/exp367_single_support/codex_train_design.md:2027:experiments/exp367_single_support/codex_review2.md:1019:     4	codex 训练侧 #1: 训练时每 ID 单图 support 定义身份, CVaR worst-support 优化。回应 exp109
experiments/exp367_single_support/codex_train_design.md:2028:experiments/exp367_single_support/codex_review2.md:1020:     5	根问题(single-image support incomplete)。纯训练侧(episodic loss 输出常规 descriptor)。
experiments/exp367_single_support/codex_train_design.md:2029:experiments/exp367_single_support/codex_review2.md:1023:     8	  1. common-valid query mask: 用 full-gallery 下有 positive 的 query 子集, 所有 support 设置同子集(否则比不同难度)。
experiments/exp367_single_support/codex_train_design.md:2030:experiments/exp367_single_support/codex_review2.md:1024:     9	  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_train_design.md:2031:experiments/exp367_single_support/codex_review2.md:1025:    10	  3. 主判据 best-random / random-worst 多 seed(20) 均值±std; 报 #false-in-topk(top10 错样本数)。
experiments/exp367_single_support/codex_train_design.md:2032:experiments/exp367_single_support/codex_review2.md:1026:    11	  4. best/worst 用 query-label oracle(诊断上下界, 不证训练可学, 诚实标注)。
experiments/exp367_single_support/codex_train_design.md:2033:experiments/exp367_single_support/codex_review2.md:1029:    14	GO(support 选择是真训练瓶颈): random-worst gap > 3 mAP(多 seed 稳, 同负样本池同 valid query) AND
experiments/exp367_single_support/codex_train_design.md:2034:experiments/exp367_single_support/codex_review2.md:1032:    17	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:2035:experiments/exp367_single_support/codex_review2.md:1050:    35	print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2036:experiments/exp367_single_support/codex_review2.md:1058:    43	print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2037:experiments/exp367_single_support/codex_review2.md:1061:    46	def support_gallery(support_idx_per_id):
experiments/exp367_single_support/codex_train_design.md:2038:experiments/exp367_single_support/codex_review2.md:1062:    47	    """has-query ID 用单 support, distractor 全量 → 负样本池不变。"""
experiments/exp367_single_support/codex_train_design.md:2039:experiments/exp367_single_support/codex_review2.md:1063:    48	    return np.concatenate([np.array(support_idx_per_id, dtype=int), distractor_g])
experiments/exp367_single_support/codex_train_design.md:2040:experiments/exp367_single_support/codex_review2.md:1083:    68	# common-valid query: full-gallery 下有 positive 的 query (固定子集, 所有 support 设置共用)
experiments/exp367_single_support/codex_train_design.md:2041:experiments/exp367_single_support/codex_review2.md:1090:    75	print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2042:experiments/exp367_single_support/codex_review2.md:1095:    80	# best/worst-support oracle (用 query-label, 诊断上下界 — 诚实: 不证训练可学)
experiments/exp367_single_support/codex_train_design.md:2043:experiments/exp367_single_support/codex_review2.md:1096:    81	best_s, worst_s = [], []
experiments/exp367_single_support/codex_train_design.md:2044:experiments/exp367_single_support/codex_review2.md:1104:    89	    best_s.append(gidxs[int(np.argmax(qual))]); worst_s.append(gidxs[int(np.argmin(qual))])
experiments/exp367_single_support/codex_train_design.md:2045:experiments/exp367_single_support/codex_review2.md:1105:    90	best_mAP, best_R1, best_f10 = eval_fixed(support_gallery(best_s), valid_q)
experiments/exp367_single_support/codex_train_design.md:2046:experiments/exp367_single_support/codex_review2.md:1106:    91	worst_mAP, worst_R1, worst_f10 = eval_fixed(support_gallery(worst_s), valid_q)
experiments/exp367_single_support/codex_train_design.md:2047:experiments/exp367_single_support/codex_review2.md:1108:    93	# random-support 多 seed
experiments/exp367_single_support/codex_train_design.md:2048:experiments/exp367_single_support/codex_review2.md:1113:    98	    rand_mAPs.append(eval_fixed(support_gallery(rs), valid_q)[0])
experiments/exp367_single_support/codex_train_design.md:2049:experiments/exp367_single_support/codex_review2.md:1118:   103	print(f'  best-support   : mAP={best_mAP:.2f} (oracle 上界, 用 query-label)  false10={best_f10:.3f}')
experiments/exp367_single_support/codex_train_design.md:2050:experiments/exp367_single_support/codex_review2.md:1119:   104	print(f'  random-support : mAP={rand_mean:.2f}±{rand_std:.2f}')
experiments/exp367_single_support/codex_train_design.md:2051:experiments/exp367_single_support/codex_review2.md:1120:   105	print(f'  worst-support  : mAP={worst_mAP:.2f} (oracle 下界)  false10={worst_f10:.3f}')
experiments/exp367_single_support/codex_train_design.md:2052:experiments/exp367_single_support/codex_review2.md:1121:   106	print(f'  best-random gap = {best_mAP-rand_mean:.2f}  random-worst gap = {rand_mean-worst_mAP:.2f}  best-worst = {best_mAP-worst_mAP:.2f}')
experiments/exp367_single_support/codex_train_design.md:2053:experiments/exp367_single_support/codex_review2.md:1122:   107	# 主判据: random-worst gap(同负样本池同valid query, 单support内选择) > 3 且 false10 同向变化(非trivial少正样本)
experiments/exp367_single_support/codex_train_design.md:2054:experiments/exp367_single_support/codex_review2.md:1123:   108	go = (rand_mean - worst_mAP) > 3.0 and (best_mAP - rand_mean) > 1.0
experiments/exp367_single_support/codex_train_design.md:2055:experiments/exp367_single_support/codex_review2.md:1124:   109	print(f'  [verdict] {"GO (单 support 内选择 matters, support representation 是真训练瓶颈)" if go else "DEAD (support 选择价值小/被负样本池-valid-query 解释)"}')
experiments/exp367_single_support/codex_train_design.md:2056:experiments/exp367_single_support/codex_review2.md:1125:   110	print('  ※ best/worst 是 query-label oracle 诊断上下界, 证 headroom 存在; 训练能否学到要 CVaR train 验', flush=True)
experiments/exp367_single_support/codex_train_design.md:2057:experiments/exp367_single_support/codex_review2.md:1131:/bin/zsh -lc "sed -n '1,220p' experiments/exp367_single_support/codex_review2.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2058:experiments/exp367_single_support/codex_review2.md:1146:再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:2059:experiments/exp367_single_support/codex_review2.md:1148:我先按仓库接手规则读最新实验上下文，再聚焦审 `cvpb_single_support_probe.py` 的 v2 逻辑和潜在泄漏点。
experiments/exp367_single_support/codex_train_design.md:2060:experiments/exp367_single_support/codex_review2.md:1355:/bin/zsh -lc 'rg -n "High|false10|random|common-valid|distractor|missing|verdict|needs|approve|trivial|污染" experiments/exp367_single_support/codex_review2.md' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2061:experiments/exp367_single_support/codex_review2.md:1357:14:再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:2062:experiments/exp367_single_support/codex_review2.md:1362:916:/bin/zsh -lc 'ls -la experiments/exp367_single_support && rg -n "common|valid|distractor|support|seed|false|missing|positive|top" experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2063:experiments/exp367_single_support/codex_review2.md:1363:927:8:  1. common-valid query mask: 用 full-gallery 下有 positive 的 query 子集, 所有 support 设置同子集(否则比不同难度)。
experiments/exp367_single_support/codex_train_design.md:2064:experiments/exp367_single_support/codex_review2.md:1364:928:9:  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_train_design.md:2065:experiments/exp367_single_support/codex_review2.md:1365:929:10:  3. 主判据 best-random / random-worst 多 seed(20) 均值±std; 报 #false-in-topk(top10 错样本数)。
experiments/exp367_single_support/codex_train_design.md:2066:experiments/exp367_single_support/codex_review2.md:1366:930:14:GO(support 选择是真训练瓶颈): random-worst gap > 3 mAP(多 seed 稳, 同负样本池同 valid query) AND
experiments/exp367_single_support/codex_train_design.md:2067:experiments/exp367_single_support/codex_review2.md:1370:937:43:print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2068:experiments/exp367_single_support/codex_review2.md:1371:939:47:    """has-query ID 用单 support, distractor 全量 → 负样本池不变。"""
experiments/exp367_single_support/codex_train_design.md:2069:experiments/exp367_single_support/codex_review2.md:1372:940:48:    return np.concatenate([np.array(support_idx_per_id, dtype=int), distractor_g])
experiments/exp367_single_support/codex_train_design.md:2070:experiments/exp367_single_support/codex_review2.md:1377:948:68:# common-valid query: full-gallery 下有 positive 的 query (固定子集, 所有 support 设置共用)
experiments/exp367_single_support/codex_train_design.md:2071:experiments/exp367_single_support/codex_review2.md:1378:952:75:print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2072:experiments/exp367_single_support/codex_review2.md:1379:957:93:# random-support 多 seed
experiments/exp367_single_support/codex_train_design.md:2073:experiments/exp367_single_support/codex_review2.md:1382:962:103:print(f'  best-support   : mAP={best_mAP:.2f} (oracle 上界, 用 query-label)  false10={best_f10:.3f}')
experiments/exp367_single_support/codex_train_design.md:2074:experiments/exp367_single_support/codex_review2.md:1383:963:104:print(f'  random-support : mAP={rand_mean:.2f}±{rand_std:.2f}')
experiments/exp367_single_support/codex_train_design.md:2075:experiments/exp367_single_support/codex_review2.md:1384:964:105:print(f'  worst-support  : mAP={worst_mAP:.2f} (oracle 下界)  false10={worst_f10:.3f}')
experiments/exp367_single_support/codex_train_design.md:2076:experiments/exp367_single_support/codex_review2.md:1385:965:107:# 主判据: random-worst gap(同负样本池同valid query, 单support内选择) > 3 且 false10 同向变化(非trivial少正样本)
experiments/exp367_single_support/codex_train_design.md:2077:experiments/exp367_single_support/codex_review2.md:1386:966:109:print(f'  [verdict] {"GO (单 support 内选择 matters, support representation 是真训练瓶颈)" if go else "DEAD (support 选择价值小/被负样本池-valid-query 解释)"}')
experiments/exp367_single_support/codex_train_design.md:2078:experiments/exp367_single_support/codex_review2.md:1387:992:- random-support：每 ID 随机 1 图
experiments/exp367_single_support/codex_train_design.md:2079:experiments/exp367_single_support/codex_review2.md:1388:998:★诚实设计要点：单图 vs 多图必掉 mAP（少正样本）是 trivial，所以**关键判据是 best−worst gap**（同样单图，选择重不重要），不是 single<full。codex 审 probe 验这个设计是否真有意义（用户要审查交 codex）。
experiments/exp367_single_support/codex_train_design.md:2080:experiments/exp367_single_support/codex_review2.md:1389:1017:     2	"""Single-Support ReID — cheap kill-switch (零训练) — v2 (codex needs-attention 修).
experiments/exp367_single_support/codex_train_design.md:2081:experiments/exp367_single_support/codex_review2.md:1391:1023:     8	  1. common-valid query mask: 用 full-gallery 下有 positive 的 query 子集, 所有 support 设置同子集(否则比不同难度)。
experiments/exp367_single_support/codex_train_design.md:2082:experiments/exp367_single_support/codex_review2.md:1392:1024:     9	  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_train_design.md:2083:experiments/exp367_single_support/codex_review2.md:1393:1025:    10	  3. 主判据 best-random / random-worst 多 seed(20) 均值±std; 报 #false-in-topk(top10 错样本数)。
experiments/exp367_single_support/codex_train_design.md:2084:experiments/exp367_single_support/codex_review2.md:1394:1029:    14	GO(support 选择是真训练瓶颈): random-worst gap > 3 mAP(多 seed 稳, 同负样本池同 valid query) AND
experiments/exp367_single_support/codex_train_design.md:2085:experiments/exp367_single_support/codex_review2.md:1398:1058:    43	print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2086:experiments/exp367_single_support/codex_review2.md:1399:1062:    47	    """has-query ID 用单 support, distractor 全量 → 负样本池不变。"""
experiments/exp367_single_support/codex_train_design.md:2087:experiments/exp367_single_support/codex_review2.md:1400:1063:    48	    return np.concatenate([np.array(support_idx_per_id, dtype=int), distractor_g])
experiments/exp367_single_support/codex_train_design.md:2088:experiments/exp367_single_support/codex_review2.md:1405:1083:    68	# common-valid query: full-gallery 下有 positive 的 query (固定子集, 所有 support 设置共用)
experiments/exp367_single_support/codex_train_design.md:2089:experiments/exp367_single_support/codex_review2.md:1406:1090:    75	print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2090:experiments/exp367_single_support/codex_review2.md:1407:1108:    93	# random-support 多 seed
experiments/exp367_single_support/codex_train_design.md:2091:experiments/exp367_single_support/codex_review2.md:1411:1118:   103	print(f'  best-support   : mAP={best_mAP:.2f} (oracle 上界, 用 query-label)  false10={best_f10:.3f}')
experiments/exp367_single_support/codex_train_design.md:2092:experiments/exp367_single_support/codex_review2.md:1412:1119:   104	print(f'  random-support : mAP={rand_mean:.2f}±{rand_std:.2f}')
experiments/exp367_single_support/codex_train_design.md:2093:experiments/exp367_single_support/codex_review2.md:1413:1120:   105	print(f'  worst-support  : mAP={worst_mAP:.2f} (oracle 下界)  false10={worst_f10:.3f}')
experiments/exp367_single_support/codex_train_design.md:2094:experiments/exp367_single_support/codex_review2.md:1414:1121:   106	print(f'  best-random gap = {best_mAP-rand_mean:.2f}  random-worst gap = {rand_mean-worst_mAP:.2f}  best-worst = {best_mAP-worst_mAP:.2f}')
experiments/exp367_single_support/codex_train_design.md:2095:experiments/exp367_single_support/codex_review2.md:1415:1122:   107	# 主判据: random-worst gap(同负样本池同valid query, 单support内选择) > 3 且 false10 同向变化(非trivial少正样本)
experiments/exp367_single_support/codex_train_design.md:2096:experiments/exp367_single_support/codex_review2.md:1416:1124:   109	print(f'  [verdict] {"GO (单 support 内选择 matters, support representation 是真训练瓶颈)" if go else "DEAD (support 选择价值小/被负样本池-valid-query 解释)"}')
experiments/exp367_single_support/codex_train_design.md:2097:experiments/exp367_single_support/codex_review2.md:1420:/bin/zsh -lc 'git status --short && git diff -- experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2098:experiments/exp367_single_support/codex_review2.md:1441:?? experiments/cargo_cvpb/airl_gate_oracle.py
experiments/exp367_single_support/codex_train_design.md:2099:experiments/exp367_single_support/codex_review2.md:1619:?? experiments/exp367_single_support/
experiments/exp367_single_support/codex_train_design.md:2100:experiments/exp367_single_support/codex_review2.md:1632:(a) **对**。`valid_q` 是 full-gallery 下去掉 same-camera 后仍有 positive 的 query 子集，之后 full/best/random/worst 都共用同一个 `valid_q`。
experiments/exp367_single_support/codex_train_design.md:2101:experiments/exp367_single_support/codex_review2.md:1634:(b) **基本对，但别说成 exact negative pool 不变**。`distractor/no-query pid` 确实全量保留；`has-query ID` 确实每 ID 单 support。  
experiments/exp367_single_support/codex_train_design.md:2102:experiments/exp367_single_support/codex_review2.md:1635:但对任意 query 来说，其他 has-query ID 也是负样本，它们的具体 support 图会随 best/random/worst 改变。所以“负样本池不变”只能理解为 distractor 全量、负 ID/cardinality 可比，不是负样本图像完全固定。
experiments/exp367_single_support/codex_train_design.md:2103:experiments/exp367_single_support/codex_review2.md:1637:(c) **部分对**。random 默认 20 seeds，主判据 `random-worst > 3` + `best-random > 1` 已实现。  
experiments/exp367_single_support/codex_train_design.md:2104:experiments/exp367_single_support/codex_review2.md:1638:问题是 `false10` 没有给 random mean/std，只报了 full/best/worst；而且当前 `false10` 是 top10 错误比例，不是 `#false-in-topk` 计数。代码注释说 false10 参与排除 trivial，但 `go` 实际没检查 false10。
experiments/exp367_single_support/codex_train_design.md:2105:experiments/exp367_single_support/codex_review2.md:1640:(d) **对**。single-support 下某 query 没有可用 positive 时记 AP/R1=0 是合理的，能惩罚 support 选到同 camera 或无法覆盖 cross-camera positive 的情况。
experiments/exp367_single_support/codex_train_design.md:2106:experiments/exp367_single_support/codex_review2.md:1642:残留 trivial 污染：**有轻微残留，不是 v1 那种致命污染**。主要是 has-query 负样本图像随 support 策略变化，以及 missing-positive 可能混入 camera-coverage 效应。建议补报 `random false10 mean±std`、`missing-positive rate`，并把 `false10` 明确成 rate 或改成 count。Best/worst oracle 用 query label 已标注为诊断上/下界，这点可以接受。
experiments/exp367_single_support/codex_train_design.md:2107:experiments/exp367_single_support/codex_review2.md:1647:(a) **对**。`valid_q` 是 full-gallery 下去掉 same-camera 后仍有 positive 的 query 子集，之后 full/best/random/worst 都共用同一个 `valid_q`。
experiments/exp367_single_support/codex_train_design.md:2108:experiments/exp367_single_support/codex_review2.md:1649:(b) **基本对，但别说成 exact negative pool 不变**。`distractor/no-query pid` 确实全量保留；`has-query ID` 确实每 ID 单 support。  
experiments/exp367_single_support/codex_train_design.md:2109:experiments/exp367_single_support/codex_review2.md:1650:但对任意 query 来说，其他 has-query ID 也是负样本，它们的具体 support 图会随 best/random/worst 改变。所以“负样本池不变”只能理解为 distractor 全量、负 ID/cardinality 可比，不是负样本图像完全固定。
experiments/exp367_single_support/codex_train_design.md:2110:experiments/exp367_single_support/codex_review2.md:1652:(c) **部分对**。random 默认 20 seeds，主判据 `random-worst > 3` + `best-random > 1` 已实现。  
experiments/exp367_single_support/codex_train_design.md:2111:experiments/exp367_single_support/codex_review2.md:1653:问题是 `false10` 没有给 random mean/std，只报了 full/best/worst；而且当前 `false10` 是 top10 错误比例，不是 `#false-in-topk` 计数。代码注释说 false10 参与排除 trivial，但 `go` 实际没检查 false10。
experiments/exp367_single_support/codex_train_design.md:2112:experiments/exp367_single_support/codex_review2.md:1655:(d) **对**。single-support 下某 query 没有可用 positive 时记 AP/R1=0 是合理的，能惩罚 support 选到同 camera 或无法覆盖 cross-camera positive 的情况。
experiments/exp367_single_support/codex_train_design.md:2113:experiments/exp367_single_support/codex_review2.md:1657:残留 trivial 污染：**有轻微残留，不是 v1 那种致命污染**。主要是 has-query 负样本图像随 support 策略变化，以及 missing-positive 可能混入 camera-coverage 效应。建议补报 `random false10 mean±std`、`missing-positive rate`，并把 `false10` 明确成 rate 或改成 count。Best/worst oracle 用 query label 已标注为诊断上/下界，这点可以接受。
experiments/exp367_single_support/codex_train_design.md:2120:experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:2121:experiments/exp367_single_support/codex_review.md:16:我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_train_design.md:2122:experiments/exp367_single_support/codex_review.md:18:/bin/zsh -lc 'nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2123:experiments/exp367_single_support/codex_review.md:20:/bin/zsh -lc 'find experiments/exp367_single_support -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2124:experiments/exp367_single_support/codex_review.md:22:experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:2125:experiments/exp367_single_support/codex_review.md:23:experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/codex_train_design.md:2126:experiments/exp367_single_support/codex_review.md:27:     2	"""Single-Support ReID — cheap kill-switch (零训练).
experiments/exp367_single_support/codex_train_design.md:2127:experiments/exp367_single_support/codex_review.md:29:     4	codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_train_design.md:2128:experiments/exp367_single_support/codex_review.md:30:     5	回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_train_design.md:2129:experiments/exp367_single_support/codex_review.md:35:    10	  - random-support (每 ID 随机 1 图) : 随机单 support
experiments/exp367_single_support/codex_train_design.md:2130:experiments/exp367_single_support/codex_review.md:36:    11	  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_train_design.md:2131:experiments/exp367_single_support/codex_review.md:37:    12	  - best-support (每 ID 选最好 1 图)  : support 选择 oracle 上界
experiments/exp367_single_support/codex_train_design.md:2132:experiments/exp367_single_support/codex_review.md:39:    14	GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_train_design.md:2133:experiments/exp367_single_support/codex_review.md:40:    15	  worst 比 full 掉 > 3 mAP  AND  best - worst gap > 3 mAP (support 选择 matters)。
experiments/exp367_single_support/codex_train_design.md:2134:experiments/exp367_single_support/codex_review.md:41:    16	DEAD: best≈worst (哪张 support 都一样, 没 support 选择价值) 或 single≈full (单图够)。
experiments/exp367_single_support/codex_train_design.md:2135:experiments/exp367_single_support/codex_review.md:44:    19	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:2136:experiments/exp367_single_support/codex_review.md:57:    32	print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2137:experiments/exp367_single_support/codex_review.md:82:    57	# random-support: 每 ID 随机 1 图
experiments/exp367_single_support/codex_train_design.md:2138:experiments/exp367_single_support/codex_review.md:87:    62	# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_train_design.md:2139:experiments/exp367_single_support/codex_review.md:88:    63	# 用 该 ID 的 gallery 图 与 该 ID 所有 query 的平均 sim 作为 support quality (高=好 support)
experiments/exp367_single_support/codex_train_design.md:2140:experiments/exp367_single_support/codex_review.md:89:    64	best_idx, worst_idx = [], []
experiments/exp367_single_support/codex_train_design.md:2141:experiments/exp367_single_support/codex_review.md:93:    68	        best_idx.append(gidxs[0]); worst_idx.append(gidxs[0]); continue
experiments/exp367_single_support/codex_train_design.md:2142:experiments/exp367_single_support/codex_review.md:94:    69	    # 每个候选 support 图 g 对 同 ID query 的平均 cos (排同 cam)
experiments/exp367_single_support/codex_train_design.md:2143:experiments/exp367_single_support/codex_review.md:101:    76	    worst_idx.append(gidxs[int(np.argmin(quality))])
experiments/exp367_single_support/codex_train_design.md:2144:experiments/exp367_single_support/codex_review.md:103:    78	worst_mAP, worst_R1 = eval_gallery(np.array(worst_idx))
experiments/exp367_single_support/codex_train_design.md:2145:experiments/exp367_single_support/codex_review.md:107:    82	print(f'  best-support     : mAP={best_mAP:.2f} R1={best_R1:.2f}  (vs full {best_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2146:experiments/exp367_single_support/codex_review.md:108:    83	print(f'  random-support   : mAP={rand_mAP:.2f} R1={rand_R1:.2f}  (vs full {rand_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2147:experiments/exp367_single_support/codex_review.md:109:    84	print(f'  worst-support    : mAP={worst_mAP:.2f} R1={worst_R1:.2f}  (vs full {worst_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2148:experiments/exp367_single_support/codex_review.md:110:    85	print(f'  worst-vs-full drop = {full_mAP-worst_mAP:.2f} (GO>3)  best-worst gap = {best_mAP-worst_mAP:.2f} (GO>3)')
experiments/exp367_single_support/codex_train_design.md:2149:experiments/exp367_single_support/codex_review.md:111:    86	go = (full_mAP - worst_mAP) > 3.0 and (best_mAP - worst_mAP) > 3.0
experiments/exp367_single_support/codex_train_design.md:2150:experiments/exp367_single_support/codex_review.md:112:    87	print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_train_design.md:2151:experiments/exp367_single_support/codex_review.md:116:/bin/zsh -lc "rg -n \"cvpb_single_support|single_support|ae_feats|false-in-topk|false|topk|exp367|CVPB|support\" experiments/exp367_single_support experiments -g '"'!*.npz'"' -g '"'!*.pth'"'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2152:experiments/exp367_single_support/codex_review.md:118:experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:2153:experiments/exp367_single_support/codex_review.md:119:experiments/exp367_single_support/codex_review.md:16:我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_train_design.md:2154:experiments/exp367_single_support/codex_review.md:120:experiments/exp130/design.md:24:3. 那么这才是对 “support-complete 新增 correction 是否被完整 target 稀释” 的有效检验
experiments/exp367_single_support/codex_train_design.md:2155:experiments/exp367_single_support/codex_review.md:121:experiments/exp130/design.md:35:- online support-complete teacher 不 freeze
experiments/exp367_single_support/codex_train_design.md:2156:experiments/exp367_single_support/codex_review.md:122:experiments/exp367_single_support/cvpb_single_support_probe.py:4:codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_train_design.md:2157:experiments/exp367_single_support/codex_review.md:123:experiments/exp367_single_support/cvpb_single_support_probe.py:5:回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_train_design.md:2158:experiments/exp367_single_support/codex_review.md:124:experiments/exp367_single_support/cvpb_single_support_probe.py:10:  - random-support (每 ID 随机 1 图) : 随机单 support
experiments/exp367_single_support/codex_train_design.md:2159:experiments/exp367_single_support/codex_review.md:125:experiments/exp367_single_support/cvpb_single_support_probe.py:11:  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_train_design.md:2160:experiments/exp367_single_support/codex_review.md:126:experiments/exp367_single_support/cvpb_single_support_probe.py:12:  - best-support (每 ID 选最好 1 图)  : support 选择 oracle 上界
experiments/exp367_single_support/codex_train_design.md:2161:experiments/exp367_single_support/codex_review.md:127:experiments/exp367_single_support/cvpb_single_support_probe.py:14:GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_train_design.md:2162:experiments/exp367_single_support/codex_review.md:128:experiments/exp367_single_support/cvpb_single_support_probe.py:15:  worst 比 full 掉 > 3 mAP  AND  best - worst gap > 3 mAP (support 选择 matters)。
experiments/exp367_single_support/codex_train_design.md:2163:experiments/exp367_single_support/codex_review.md:129:experiments/exp367_single_support/cvpb_single_support_probe.py:16:DEAD: best≈worst (哪张 support 都一样, 没 support 选择价值) 或 single≈full (单图够)。
experiments/exp367_single_support/codex_train_design.md:2164:experiments/exp367_single_support/codex_review.md:130:experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_train_design.md:2165:experiments/exp367_single_support/codex_review.md:131:experiments/exp367_single_support/cvpb_single_support_probe.py:19:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:2166:experiments/exp367_single_support/codex_review.md:132:experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_train_design.md:2167:experiments/exp367_single_support/codex_review.md:133:experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2168:experiments/exp367_single_support/codex_review.md:134:experiments/exp367_single_support/cvpb_single_support_probe.py:57:# random-support: 每 ID 随机 1 图
experiments/exp367_single_support/codex_train_design.md:2169:experiments/exp367_single_support/codex_review.md:135:experiments/exp367_single_support/cvpb_single_support_probe.py:62:# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_train_design.md:2170:experiments/exp367_single_support/codex_review.md:136:experiments/exp367_single_support/cvpb_single_support_probe.py:63:# 用 该 ID 的 gallery 图 与 该 ID 所有 query 的平均 sim 作为 support quality (高=好 support)
experiments/exp367_single_support/codex_train_design.md:2171:experiments/exp367_single_support/codex_review.md:137:experiments/exp367_single_support/cvpb_single_support_probe.py:69:    # 每个候选 support 图 g 对 同 ID query 的平均 cos (排同 cam)
experiments/exp367_single_support/codex_train_design.md:2172:experiments/exp367_single_support/codex_review.md:138:experiments/exp367_single_support/cvpb_single_support_probe.py:82:print(f'  best-support     : mAP={best_mAP:.2f} R1={best_R1:.2f}  (vs full {best_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2173:experiments/exp367_single_support/codex_review.md:139:experiments/exp367_single_support/cvpb_single_support_probe.py:83:print(f'  random-support   : mAP={rand_mAP:.2f} R1={rand_R1:.2f}  (vs full {rand_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2174:experiments/exp367_single_support/codex_review.md:140:experiments/exp367_single_support/cvpb_single_support_probe.py:84:print(f'  worst-support    : mAP={worst_mAP:.2f} R1={worst_R1:.2f}  (vs full {worst_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2175:experiments/exp367_single_support/codex_review.md:141:experiments/exp367_single_support/cvpb_single_support_probe.py:87:print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_train_design.md:2176:experiments/exp367_single_support/codex_review.md:142:experiments/exp212/claude_review.md:28:这是纯超参数调整，不是创新实验。但 design.md 并未声称这是创新，而是定位为"LR 消融"。作为 supporting experiment 是可接受的。只要不作为主线创新方向即可。
experiments/exp367_single_support/codex_train_design.md:2177:experiments/exp367_single_support/codex_review.md:143:experiments/exp139/design.md:14:- 这个 query 当前的 support 完整度如何
experiments/exp367_single_support/codex_train_design.md:2178:experiments/exp367_single_support/codex_review.md:144:experiments/exp139/design.md:15:- 这个 query 的 global / common-support 分歧有多大
experiments/exp367_single_support/codex_train_design.md:2179:experiments/exp367_single_support/codex_review.md:145:experiments/exp139/design.md:24:4. 当前 query 的平均 common support
experiments/exp367_single_support/codex_train_design.md:2180:experiments/exp367_single_support/codex_review.md:146:experiments/exp139/design.md:25:5. 当前 query 的平均 global / common-support 分歧
experiments/exp367_single_support/codex_train_design.md:2181:experiments/exp367_single_support/codex_review.md:147:experiments/exp139/design.md:46:   - `support_ratio`
experiments/exp367_single_support/codex_train_design.md:2182:experiments/exp367_single_support/codex_review.md:148:experiments/exp139/design.md:53:   - `row_support_mean`
experiments/exp367_single_support/codex_train_design.md:2183:experiments/exp367_single_support/codex_review.md:149:experiments/module_candidates.md:36:- 需要先由候选 1 证明 keypoint-level common-support 确实有效
experiments/exp367_single_support/codex_train_design.md:2184:experiments/exp367_single_support/codex_review.md:150:experiments/module_candidates.md:46:- 用 skeleton branch 的 `kp_weights` 构造 batch 内 pairwise common-support overlap
experiments/exp367_single_support/codex_train_design.md:2185:experiments/exp367_single_support/codex_review.md:151:experiments/module_candidates.md:47:- 在 global branch 上增加一条 support-aware triplet：
experiments/exp367_single_support/codex_train_design.md:2186:experiments/exp367_single_support/codex_review.md:152:experiments/module_candidates.md:53:- 它不是再加一个 branch 模块，而是把 **pair-specific common support** 迁进训练目标
experiments/exp367_single_support/codex_train_design.md:2187:experiments/exp367_single_support/codex_review.md:153:experiments/module_candidates.md:78:  2. CSGT（训练端 common-support mining）
experiments/exp367_single_support/codex_train_design.md:2188:experiments/exp367_single_support/codex_review.md:154:experiments/module_candidates.md:220:2. 若 `TDPC` 单 seed 2-3 天内无正信号，再回退到 retrieval-time `common-support recovery`
experiments/exp367_single_support/codex_train_design.md:2189:experiments/exp367_single_support/codex_review.md:155:experiments/exp130/monitor.md:43:  1. 已在 `processor.py` 补上 `residual_kl requires support teacher` 的保护
experiments/exp367_single_support/codex_train_design.md:2190:experiments/exp367_single_support/codex_review.md:156:experiments/exp130/monitor.md:55:  2. support-complete teacher 仍为在线版本：
experiments/exp367_single_support/codex_train_design.md:2191:experiments/exp367_single_support/codex_review.md:157:experiments/exp130/monitor.md:185:     - 至少在 `delta_top + online support teacher` 这条线上，完整 teacher target 比 `residual_kl` 更有效
experiments/exp367_single_support/codex_train_design.md:2192:experiments/exp367_single_support/codex_review.md:158:experiments/exp130/monitor.md:205:  1. 已在 `processor.py` 中补上 `residual_kl requires support teacher` 的保护
experiments/exp367_single_support/codex_train_design.md:2193:experiments/exp367_single_support/codex_review.md:159:experiments/exp139/claude_review_v2.md:13:2. **Label-dependent context** — 已修复。新版 `build_query_context_descriptors()` (`pair_adaptive_fusion.py:48-74`) 的 5 个特征 (`row_mean`, `row_std`, `row_min`, `row_support_mean`, `row_change_mean`) 全部来自距离矩阵和 support ratio 统计，不依赖任何 label 信息。
experiments/exp367_single_support/codex_train_design.md:2194:experiments/exp367_single_support/codex_review.md:160:experiments/exp130/claude_review.md:3:### 1. MEDIUM — Missing validation for `residual_kl` + no support teacher
experiments/exp367_single_support/codex_train_design.md:2195:experiments/exp367_single_support/codex_review.md:161:experiments/exp130/claude_review.md:7:if csrd_target_mode == 'residual' and not csrd_support_teacher:
experiments/exp367_single_support/codex_train_design.md:2196:experiments/exp367_single_support/codex_review.md:162:experiments/exp130/claude_review.md:10:but no equivalent guard for `'residual_kl'`. Without a support teacher, `dist_t == dist_base`, making all teacher residual logits exactly zero → uniform teacher distribution → KL loss pushes student to uniform → actively harmful.
experiments/exp367_single_support/codex_train_design.md:2197:experiments/exp367_single_support/codex_review.md:163:experiments/exp130/claude_review.md:86:| Missing validation for residual_kl without support teacher | Medium | Low (config is correct) | Add one-line guard after launch |
experiments/exp367_single_support/codex_train_design.md:2198:experiments/exp367_single_support/codex_review.md:164:experiments/exp130/claude_review.md:96:   if csrd_target_mode == 'residual_kl' and not csrd_support_teacher:
experiments/exp367_single_support/codex_train_design.md:2199:experiments/exp367_single_support/codex_review.md:165:experiments/exp109/design.md:6:- 但 `SGCFR` 明确证明：**跨图 support recovery** 确实能带来大增益
experiments/exp367_single_support/codex_train_design.md:2200:experiments/exp367_single_support/codex_review.md:166:experiments/exp109/design.md:8:  **batch 内没有足够稳定的 same-ID support**
experiments/exp367_single_support/codex_train_design.md:2201:experiments/exp367_single_support/codex_review.md:167:experiments/exp109/design.md:9:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:2202:experiments/exp367_single_support/codex_review.md:168:experiments/exp109/design.md:10:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp367_single_support/codex_train_design.md:2203:experiments/exp367_single_support/codex_review.md:169:experiments/exp109/design.md:14:1. 若单图表征真的受限于“support 不完整”，那么用同 ID 多图构造 oracle prototype 后，matching 应明显优于原始 `cvk_hybrid`
experiments/exp367_single_support/codex_train_design.md:2204:experiments/exp367_single_support/codex_review.md:170:experiments/exp109/design.md:19:3. 若 oracle 上界都很小，则说明 training-time support-complete distillation 很难成为主线，应立即止损
experiments/exp367_single_support/codex_train_design.md:2205:experiments/exp367_single_support/codex_review.md:171:experiments/exp109/design.md:30:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:2206:experiments/exp367_single_support/codex_review.md:172:experiments/exp109/design.md:55:  - 说明 “support-complete teacher” 这条训练主线有真实 headroom
experiments/exp367_single_support/codex_train_design.md:2207:experiments/exp367_single_support/codex_review.md:173:experiments/exp109/design.md:59:  - support bank 训练线应止损
experiments/exp367_single_support/codex_train_design.md:2208:experiments/exp367_single_support/codex_review.md:174:experiments/exp139/monitor.md:63:  - `row_support_mean`
experiments/exp367_single_support/codex_train_design.md:2209:experiments/exp367_single_support/codex_review.md:175:experiments/exp139/monitor.md:254:### [2026-03-22 02:00] `exp139` 到 `ep80`：已追平当前最强 supporting 线，继续保持主候选
experiments/exp367_single_support/codex_train_design.md:2210:experiments/exp367_single_support/codex_review.md:176:experiments/exp246/claude_review.md:43:- 但此处作为 supporting evidence (语义+结构是否互补), 可接受
experiments/exp367_single_support/codex_train_design.md:2211:experiments/exp367_single_support/codex_review.md:177:experiments/exp109/monitor.md:14:  - `SGCFR` 说明 cross-image support recovery 确有价值
experiments/exp367_single_support/codex_train_design.md:2212:experiments/exp367_single_support/codex_review.md:178:experiments/exp109/monitor.md:16:  1. 新增 `scripts/eval_oracle_support_bank.py`
experiments/exp367_single_support/codex_train_design.md:2213:experiments/exp367_single_support/codex_review.md:179:experiments/exp109/monitor.md:19:  4. 若 headroom 明显，再进入训练版 support-complete distillation 设计
experiments/exp367_single_support/codex_train_design.md:2214:experiments/exp367_single_support/codex_review.md:180:experiments/exp109/monitor.md:22:- 结果文件: `log/occluded_duke/exp109_oracle_support_bank_exp030a/summary.json`
experiments/exp367_single_support/codex_train_design.md:2215:experiments/exp367_single_support/codex_review.md:181:experiments/exp109/monitor.md:26:  - `avg_support_count = 82.33`
experiments/exp367_single_support/codex_train_design.md:2216:experiments/exp367_single_support/codex_review.md:182:experiments/exp109/monitor.md:39:  2. 说明“support-complete latent representation”不是空想，而是存在巨大 headroom
experiments/exp367_single_support/codex_train_design.md:2217:experiments/exp367_single_support/codex_review.md:183:experiments/exp139/claude_review.md:60:1. **最优方案**：重新设计 context 特征使其不依赖 labels。例如用 row-wise distance statistics（row mean、row std、row min、row max、row support mean）替代 pos/neg 统计。这样训练和测试一致，不需要 labels。
experiments/exp367_single_support/codex_train_design.md:2218:experiments/exp367_single_support/codex_review.md:184:experiments/exp136/design.md:22:3. 应更集中地把梯度打到真正被 support-complete teacher 改变的关系上
experiments/exp367_single_support/codex_train_design.md:2219:experiments/exp367_single_support/codex_review.md:185:experiments/exp136/monitor.md:149:### [2026-03-21 20:25] `exp136` 到 `ep90`：稀疏机制稳定，但当前更像 supporting 线
experiments/exp367_single_support/codex_train_design.md:2220:experiments/exp367_single_support/codex_review.md:186:experiments/exp136/monitor.md:185:- 当前判断: `exp136` 结案，保留为 supporting 线
experiments/exp367_single_support/codex_train_design.md:2221:experiments/exp367_single_support/codex_review.md:188:experiments/exp108/design.md:8:- 因此，`exp108` 的核心不是继续调 `exp107` 的公式，而是把同一问题重新落在 **per-keypoint / common-support** 粒度：
experiments/exp367_single_support/codex_train_design.md:2222:experiments/exp367_single_support/codex_review.md:189:experiments/exp108/design.md:9:  **只有在关键点可见性和 common-support 层面，target-target 与 target-distractor 的差异才可能被稳定表达。**
experiments/exp367_single_support/codex_train_design.md:2223:experiments/exp367_single_support/codex_review.md:190:experiments/exp108/design.md:16:   的 common-support 距离比 `target ↔ target` 更小，则该 pair 应被惩罚。
experiments/exp367_single_support/codex_train_design.md:2224:experiments/exp367_single_support/codex_review.md:191:experiments/exp108/design.md:17:3. 与 `exp107` 不同，duplicate-aware pruning 在 per-keypoint 层面才可能真正发挥作用，因为 duplicate detection 与 visibility/common-support 是同一层面的结构信息。
experiments/exp367_single_support/codex_train_design.md:2225:experiments/exp367_single_support/codex_review.md:192:experiments/exp108/design.md:36:### 4. Counterfactual common-support penalty
experiments/exp367_single_support/codex_train_design.md:2226:experiments/exp367_single_support/codex_review.md:193:experiments/exp108/design.md:41:  - `support_gap = min(d_q_gd, d_qd_g) - d_tt`
experiments/exp367_single_support/codex_train_design.md:2227:experiments/exp367_single_support/codex_review.md:194:experiments/exp108/design.md:43:  - 当 `support_gap < 0` 时，说明 confuser 比 target-target 更占优，增加距离惩罚
experiments/exp367_single_support/codex_train_design.md:2228:experiments/exp367_single_support/codex_review.md:195:experiments/exp108/design.md:59:  - 说明 ambiguity 这条 retrieval-time 线即使下沉到 per-keypoint/common-support，也还不足以形成稳定可用的排名信号
experiments/exp367_single_support/codex_train_design.md:2229:experiments/exp367_single_support/codex_review.md:196:experiments/exp108/design.md:63:1. `cvk_hybrid` 已经吃掉了 target-target 的主要 common-support 信号，confuser penalty 额外增益不足
experiments/exp367_single_support/codex_train_design.md:2230:experiments/exp367_single_support/codex_review.md:197:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2231:experiments/exp367_single_support/codex_review.md:198:experiments/exp108/monitor.md:7:- 核心变量: per-keypoint common-support 层面的 duplicate-aware confuser penalty
experiments/exp367_single_support/codex_train_design.md:2232:experiments/exp367_single_support/codex_review.md:199:experiments/exp108/monitor.md:14:  - 若继续 ambiguity 主线，必须回到 `per-keypoint / common-support`
experiments/exp367_single_support/codex_train_design.md:2233:experiments/exp367_single_support/codex_review.md:200:experiments/exp108/monitor.md:33:  1. per-keypoint / common-support 粒度下的 penalty 仍然整体负面
experiments/exp367_single_support/codex_train_design.md:2234:experiments/exp367_single_support/codex_review.md:201:experiments/exp131/design.md:15:如果当前瓶颈主要是 batch 内 changed-pair coverage 不足，那么在保持 `exp125` 的 online support teacher 与 `delta_top` routing 不变的前提下，引入 cross-batch relation queue 应当带来更强的 late-stage 收益。
experiments/exp367_single_support/codex_train_design.md:2235:experiments/exp367_single_support/codex_review.md:202:experiments/exp131/design.md:33:   - support-complete teacher kp feats
experiments/exp367_single_support/codex_train_design.md:2236:experiments/exp367_single_support/codex_review.md:203:experiments/exp137/design.md:24:- online support teacher
experiments/exp367_single_support/codex_train_design.md:2237:experiments/exp367_single_support/codex_review.md:204:experiments/exp131/monitor.md:116:     - 真正卡住的更像是 **pair-specific support-complete correction 不能被当前单向量学生充分吸收**
experiments/exp367_single_support/codex_train_design.md:2238:experiments/exp367_single_support/codex_review.md:205:experiments/exp357_pose_shuffle_ks/design.md:38:- Medium-2(判读): NO-DROP 侧被裁剪对齐混淆(别人 pose 仍带粗糙 canonical 头/躯干/腿先验)。Codex/Claude 一致: 掉点=干净铁证(图特定 pose correspondence 重要); 不掉=只能说"精确图特定 pose 在对齐裁剪下非必需", 需补 **cross-PART(17关键点通道)shuffle** 二次确认(测解剖通道身份是否重要, 同图空间 support 不变)。最佳矩阵: cross-image + per-image channel-shuffle + no-pose/fixed-canonical control。
experiments/exp367_single_support/codex_train_design.md:2239:experiments/exp367_single_support/codex_review.md:206:experiments/exp357_pose_shuffle_ks/design.md:44:- 下一步: cross-PART(通道)shuffle exp358 二次确认——打乱17关键点通道(破坏解剖部位身份, 保留同图空间 support)。若 exp358 也只小掉→解剖身份也不重要, 只是"某种空间池化结构"在涨→故事进一步塌; 若 exp358 大掉→解剖部位结构重要。
experiments/exp367_single_support/codex_train_design.md:2240:experiments/exp367_single_support/codex_review.md:207:experiments/exp138/monitor.md:137:- 当前判断: 继续，但当前更像 supporting 线
experiments/exp367_single_support/codex_train_design.md:2241:experiments/exp367_single_support/codex_review.md:208:experiments/exp138/monitor.md:158:  - 这已经足够说明 `rank-decay` 是有效但偏弱的 supporting 机制，不值得继续占用本地主卡
experiments/exp367_single_support/codex_train_design.md:2242:experiments/exp367_single_support/codex_review.md:209:experiments/exp138/monitor.md:168:- 当前判断: `exp138` 结案，定性为 supporting 线
experiments/exp367_single_support/codex_train_design.md:2243:experiments/exp367_single_support/codex_review.md:210:experiments/exp042/design.md:13:- 如果 `cvk_hybrid` 真的是 common-support correction，那么它应主要改善那些：
experiments/exp367_single_support/codex_train_design.md:2244:experiments/exp367_single_support/codex_review.md:211:experiments/afd_reid/afd_model.py:618:            # exist in Swin -> AFD is unsupported here (OVLI is the headline and
experiments/exp367_single_support/codex_train_design.md:2245:experiments/exp367_single_support/codex_review.md:212:experiments/afd_reid/afd_model.py:620:            assert not use_afd, ("backbone='swin_small' does not support the AFD "
experiments/exp367_single_support/codex_train_design.md:2246:experiments/exp367_single_support/codex_review.md:213:experiments/exp042/monitor.md:51:5. 同时也存在 `top1_degraded` 样例，说明当前 common-support reasoning 还不是无代价增强。
experiments/exp367_single_support/codex_train_design.md:2247:experiments/exp367_single_support/codex_review.md:214:experiments/afd_reid/design_airl_iso_agreidv2_4090.md:58:   - 注：OSS 客户端只收 .zip（拒 .tgz "Unsupported file type"），lab-3090 又无 zip 命令 → 改走 base64-over-ssh 本地中转。
experiments/exp367_single_support/codex_train_design.md:2248:experiments/exp367_single_support/codex_review.md:215:experiments/exp107/design.md:41:  - `support_gap = min(d_q_gd, d_qd_g) - d_tt`
experiments/exp367_single_support/codex_train_design.md:2249:experiments/exp367_single_support/codex_review.md:216:experiments/exp107/design.md:43:  - 若 `support_gap` 小，说明该 pair 的 target-target 优势不足，属于高歧义 pair
experiments/exp367_single_support/codex_train_design.md:2250:experiments/exp367_single_support/codex_review.md:217:experiments/exp107/design.md:44:  - 用 `support_gap` 对 top-K 基线距离做 margin-based 调整
experiments/exp367_single_support/codex_train_design.md:2251:experiments/exp367_single_support/codex_review.md:218:experiments/exp247/design.md:5:**重新定义问题**: "Occluded ReID fails because fixed part vocabularies assume complete semantic support. Under occlusion, the model should instantiate only the semantic groups actually supported by visible evidence."
experiments/exp367_single_support/codex_train_design.md:2252:experiments/exp367_single_support/codex_review.md:219:experiments/exp107/monitor.md:46:  1. 有符号 support-gap 重排明确负面，说明“奖励安全 pair + 惩罚危险 pair”的粗糙公式不成立。
experiments/exp367_single_support/codex_train_design.md:2253:experiments/exp367_single_support/codex_review.md:220:experiments/exp107/monitor.md:54:- 若继续研究 target/distractor ambiguity，必须把推理粒度拉回 `per-keypoint / common-support`，而不是继续在 pooled person embedding 上做文章。
experiments/exp367_single_support/codex_train_design.md:2254:experiments/exp367_single_support/codex_review.md:221:experiments/exp225/claude_review.md:78:- **创新门槛**：这是组合实验，不是新创新。design.md 没有声称是创新，只是验证叠加效果。符合"supporting evidence"角色
experiments/exp367_single_support/codex_train_design.md:2255:experiments/exp367_single_support/codex_review.md:222:experiments/decisions.md:1138:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/exp367_single_support/codex_train_design.md:2256:experiments/exp367_single_support/codex_review.md:223:experiments/decisions.md:1224:1. 近年的强路线把问题定义在 **target ambiguity / common visible support / retrieval-time reasoning**，而不是“再学一个融合权重”。
experiments/exp367_single_support/codex_train_design.md:2257:experiments/exp367_single_support/codex_review.md:224:experiments/decisions.md:1289:   这符合“keypoint common-support 更适合作为补充项”的判断。
experiments/exp367_single_support/codex_train_design.md:2258:experiments/exp367_single_support/codex_review.md:225:experiments/decisions.md:1413:   - common-support reasoning 对整体排序的修正作用
experiments/exp367_single_support/codex_train_design.md:2259:experiments/exp367_single_support/codex_review.md:226:experiments/decisions.md:1441:   - 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
experiments/exp367_single_support/codex_train_design.md:2260:experiments/exp367_single_support/codex_review.md:227:experiments/decisions.md:1442:   - 在 global triplet 上增加 support-aware hard mining 约束
experiments/exp367_single_support/codex_train_design.md:2261:experiments/exp367_single_support/codex_review.md:228:experiments/decisions.md:1499:5. 但我认同蓝队的核心判断：SGMKC 更可能是 supporting experiment 而非 main contribution
experiments/exp367_single_support/codex_train_design.md:2262:experiments/exp367_single_support/codex_review.md:229:experiments/decisions.md:1690:3. 若 `TDPC` 在 2-3 天内无明显正信号，则 fallback 到 retrieval-time `common-support recovery`，不继续做 `TDPC` 小修小补。
experiments/exp367_single_support/codex_train_design.md:2263:experiments/exp367_single_support/codex_review.md:230:experiments/decisions.md:1738:   真正有效的 pair-specific reasoning 很可能必须发生在 `per-keypoint / common-support` 粒度，而不是 pooled person feature 粒度。
experiments/exp367_single_support/codex_train_design.md:2264:experiments/exp367_single_support/codex_review.md:232:experiments/decisions.md:1749:  **duplicate-aware / confuser-aware 的 per-keypoint common-support reasoning**
experiments/exp367_single_support/codex_train_design.md:2265:experiments/exp367_single_support/codex_review.md:233:experiments/decisions.md:1755:**上下文**: `exp108 DACCM` 完成了第二轮 retrieval-time 原型验证。该实验把 `exp107` 的思路从 pooled person embedding 下沉到 `per-keypoint / common-support` 粒度，并以 `exp030a cvk_hybrid` 为主基线，比较：
experiments/exp367_single_support/codex_train_design.md:2266:experiments/exp367_single_support/codex_review.md:234:experiments/decisions.md:1771:   - per-keypoint common-support penalty 仍负面
experiments/exp367_single_support/codex_train_design.md:2267:experiments/exp367_single_support/codex_review.md:236:experiments/decisions.md:1803:   **当前性能缺口里有一大块确实来自“support 不完整”，而不是 confuser suppression 失败。**
experiments/exp367_single_support/codex_train_design.md:2268:experiments/exp367_single_support/codex_review.md:237:experiments/decisions.md:1804:3. 因而 `support-complete distillation` 已从“想法”升级为“有强 headroom 支撑的训练主线候选”。
experiments/exp367_single_support/codex_train_design.md:2269:experiments/exp367_single_support/codex_review.md:238:experiments/decisions.md:1838:   - 这和“support-complete”要表达的 multi-view support 概念并不完全一致
experiments/exp367_single_support/codex_train_design.md:2270:experiments/exp367_single_support/codex_review.md:239:experiments/decisions.md:1840:**选择**: 继续 `support-complete` 主线，但下一步只做 teacher reliability 的单变量改动。
experiments/exp367_single_support/codex_train_design.md:2271:experiments/exp367_single_support/codex_review.md:240:experiments/decisions.md:1873:3. 这与当前论文主线也更一致：关键不只是“有多少 support”，而是“teacher support 是否足够干净可信”。
experiments/exp367_single_support/codex_train_design.md:2272:experiments/exp367_single_support/codex_review.md:241:experiments/decisions.md:1892:1. 当前 `support-complete` 主线没有被否定；相反，它的瓶颈已比之前更清楚。
experiments/exp367_single_support/codex_train_design.md:2273:experiments/exp367_single_support/codex_review.md:242:experiments/decisions.md:1903:   **reliable support-complete learning**
experiments/exp367_single_support/codex_train_design.md:2274:experiments/exp367_single_support/codex_review.md:243:experiments/decisions.md:1936:   - 围绕已确认的 `support incomplete` 问题重新设计新机制
experiments/exp367_single_support/codex_train_design.md:2275:experiments/exp367_single_support/codex_review.md:244:experiments/decisions.md:1950:4. 这条线会把 story 从 `support incomplete / support-complete learning` 拉回到“GCN 小模块 + 组合扫点”。
experiments/exp367_single_support/codex_train_design.md:2276:experiments/exp367_single_support/codex_review.md:245:experiments/decisions.md:1966:2. 若要切到新方向，必须先说明它相对 `support incomplete` 主线的关系，而不是直接跳到模块叠加。
experiments/exp367_single_support/codex_train_design.md:2277:experiments/exp367_single_support/codex_review.md:246:experiments/decisions.md:1974:- `exp109-116` 则说明 `support-complete` 若被压成 `per-ID prototype`，会丢失太多 pair-specific 细节
experiments/exp367_single_support/codex_train_design.md:2278:experiments/exp367_single_support/codex_review.md:247:experiments/decisions.md:1979:   **用已经被 `cvk_hybrid` 验证过的 common-support pairwise 几何，直接蒸馏 global embedding 的关系结构。**
experiments/exp367_single_support/codex_train_design.md:2279:experiments/exp367_single_support/codex_review.md:248:experiments/decisions.md:1992:   - global embedding 需要被蒸馏成更符合 common-support geometry 的空间
experiments/exp367_single_support/codex_train_design.md:2280:experiments/exp367_single_support/codex_review.md:249:experiments/decisions.md:2014:2. 当前最清楚的增益落在 `global`（`+0.6 / +0.4`），说明它确实把 common-support 几何迁进了 backbone/global 空间。
experiments/exp367_single_support/codex_train_design.md:2281:experiments/exp367_single_support/codex_review.md:250:experiments/decisions.md:2015:3. `equal_concat` 仍接近持平，说明第一版 teacher 还不够强；瓶颈更像 teacher 的 `support incompleteness`，而不是 relational distillation 这件事本身无效。
experiments/exp367_single_support/codex_train_design.md:2282:experiments/exp367_single_support/codex_review.md:251:experiments/decisions.md:2016:4. 因而 `exp109` 的高价值结论仍应保留：真正缺的不是再换一个 loss 形式，而是 **更 support-complete 的 teacher**。
experiments/exp367_single_support/codex_train_design.md:2283:experiments/exp367_single_support/codex_review.md:252:experiments/decisions.md:2019:**把 `exp109` 的 support-complete bank 降级为 teacher enhancer，而不是 pointwise distillation target，构造 support-complete relational teacher。**
experiments/exp367_single_support/codex_train_design.md:2284:experiments/exp367_single_support/codex_review.md:253:experiments/decisions.md:2052:1. `support-complete teacher` 并没有“没生效”，相反，它已经稳定地增强了 teacher 几何。
experiments/exp367_single_support/codex_train_design.md:2285:experiments/exp367_single_support/codex_review.md:254:experiments/decisions.md:2055:   **support-complete 监督的价值集中在 support-incomplete 样本；如果对所有 anchor 等权蒸馏，clean 样本会稀释掉这份增益。**
experiments/exp367_single_support/codex_train_design.md:2286:experiments/exp367_single_support/codex_review.md:255:experiments/decisions.md:2065:   - 单图遮挡带来 support incomplete
experiments/exp367_single_support/codex_train_design.md:2287:experiments/exp367_single_support/codex_review.md:256:experiments/decisions.md:2066:   - pose branch 提供 support-complete relational teacher
experiments/exp367_single_support/codex_train_design.md:2288:experiments/exp367_single_support/codex_review.md:257:experiments/decisions.md:2067:   - 但 distillation 必须 **selective**，聚焦真正存在 support gap 的 anchor
experiments/exp367_single_support/codex_train_design.md:2289:experiments/exp367_single_support/codex_review.md:258:experiments/decisions.md:2090:2. 但它没有把 `support-complete teacher` 的增强转成更好的指标，反而更像削弱了有效监督总量。
experiments/exp367_single_support/codex_train_design.md:2290:experiments/exp367_single_support/codex_review.md:259:experiments/decisions.md:2093:4. `support-complete` 主线本身仍然成立；被否定的只是 sample-level `replace_ratio` 作为路由信号太粗。
experiments/exp367_single_support/codex_train_design.md:2291:experiments/exp367_single_support/codex_review.md:260:experiments/decisions.md:2099:2. 它直接回应 `exp122` 的失败：真正该被强调的不是“这个样本补了多少 keypoint”，而是 **support-complete teacher 实际改变了哪些 pair 几何**。
experiments/exp367_single_support/codex_train_design.md:2292:experiments/exp367_single_support/codex_review.md:261:experiments/decisions.md:2101:   - 单图遮挡带来 support incomplete
experiments/exp367_single_support/codex_train_design.md:2293:experiments/exp367_single_support/codex_review.md:262:experiments/decisions.md:2102:   - support-complete teacher 改变一部分 pairwise comparability
experiments/exp367_single_support/codex_train_design.md:2294:experiments/exp367_single_support/codex_review.md:263:experiments/decisions.md:2128:1. `stable teacher` 已经被 `exp121` 明确坐实为有效 supporting mechanism，但它不是当前主突破口。
experiments/exp367_single_support/codex_train_design.md:2295:experiments/exp367_single_support/codex_review.md:264:experiments/decisions.md:2180:   **只把被 support completion 真正改变过的 comparability relations 蒸进 global embedding。**
experiments/exp367_single_support/codex_train_design.md:2296:experiments/exp367_single_support/codex_review.md:265:experiments/decisions.md:2185:3. 不同时改 `alpha`、不改 teacher bank、不断开 `support-complete` teacher，避免再次混入多个变量。
experiments/exp367_single_support/codex_train_design.md:2297:experiments/exp367_single_support/codex_review.md:266:experiments/decisions.md:2237:2. `exp124` 证明了单纯增大 focus 强度也有效，但最终不如 `exp125`，因此它应退居 supporting branch。
experiments/exp367_single_support/codex_train_design.md:2298:experiments/exp367_single_support/codex_review.md:267:experiments/decisions.md:2270:  - 二者都没有把 oracle support-complete 上界真正兑现出来
experiments/exp367_single_support/codex_train_design.md:2299:experiments/exp367_single_support/codex_review.md:268:experiments/decisions.md:2274:2. 但这不意味着要离开 `exp109`；相反，最合理的下一步仍然是沿 `support incomplete -> support-complete learning` 这条主线，直接测试更强的 feature-level 兑现机制。
experiments/exp367_single_support/codex_train_design.md:2300:experiments/exp367_single_support/codex_review.md:269:experiments/decisions.md:2275:3. `SCFR≈SCKD` 只能说明 “hard replace 不优于 loss-only”，不能说明 “feature-level support completion 整体无效”。
experiments/exp367_single_support/codex_train_design.md:2301:experiments/exp367_single_support/codex_review.md:270:experiments/decisions.md:2279:2. 该实验保持 `bank`、`warmup`、`threshold` 与 `exp116` 同量级，只改 low-vis keypoint 如何利用 support-complete prototype：
experiments/exp367_single_support/codex_train_design.md:2302:experiments/exp367_single_support/codex_review.md:271:experiments/decisions.md:2310:1. `SCRC` 没有把 feature-level support completion 推成更强结果，反而 late-stage 基本塌成了“近似 hard replace”。
experiments/exp367_single_support/codex_train_design.md:2303:experiments/exp367_single_support/codex_review.md:272:experiments/decisions.md:2311:2. 因而 `exp109` 被否定的不是 `support incomplete` 问题定义，而是：
experiments/exp367_single_support/codex_train_design.md:2304:experiments/exp367_single_support/codex_review.md:273:experiments/decisions.md:2313:3. `freeze20/30` 的既有证据已经足够说明它只是弱 supporting mechanism，不值得继续占用本地算力。
experiments/exp367_single_support/codex_train_design.md:2305:experiments/exp367_single_support/codex_review.md:274:experiments/decisions.md:2315:   **support-complete teacher 的新增 correction 仍被完整 teacher target 稀释。**
experiments/exp367_single_support/codex_train_design.md:2306:experiments/exp367_single_support/codex_review.md:275:experiments/decisions.md:2324:   - support-complete teacher 的增量信息是真实存在的
experiments/exp367_single_support/codex_train_design.md:2307:experiments/exp367_single_support/codex_review.md:276:experiments/decisions.md:2401:   **pair-specific support-complete correction 不能被当前单向量 student 充分吸收。**
experiments/exp367_single_support/codex_train_design.md:2308:experiments/exp367_single_support/codex_review.md:277:experiments/decisions.md:2422:3. `exp040/045` 的固定 `cvk_hybrid` 已经证明 pair-specific common-support correction 在检索时能转成稳定正信号。
experiments/exp367_single_support/codex_train_design.md:2309:experiments/exp367_single_support/codex_review.md:278:experiments/decisions.md:2563:1. `exp136` 到此结案，保留为 supporting 证据
experiments/exp367_single_support/codex_train_design.md:2310:experiments/exp367_single_support/codex_review.md:279:experiments/decisions.md:2713:1. `exp138` 已经提供了足够的负边界：平滑 top-sensitive 只能算 supporting 机制
experiments/exp367_single_support/codex_train_design.md:2311:experiments/exp367_single_support/codex_review.md:280:experiments/decisions.md:2716:   - pose 定义 common support
experiments/exp367_single_support/codex_train_design.md:2312:experiments/exp367_single_support/codex_review.md:281:experiments/decisions.md:2717:   - query context 决定 pair correction 应如何解释该 support
experiments/exp367_single_support/codex_train_design.md:2313:experiments/exp367_single_support/codex_review.md:282:experiments/decisions.md:2722:- `exp138` 已停表，结论为 supporting 线
experiments/exp367_single_support/codex_train_design.md:2314:experiments/exp367_single_support/codex_review.md:283:experiments/decisions.md:2829:1. `exp139` 到 `ep80` 为止，已经基本追平当前最强 supporting 线 `exp135`
experiments/exp367_single_support/codex_train_design.md:2315:experiments/exp367_single_support/codex_review.md:284:experiments/decisions.md:2920:  - 真正 headroom 来自 `single-image support incomplete`
experiments/exp367_single_support/codex_train_design.md:2316:experiments/exp367_single_support/codex_review.md:285:experiments/decisions.md:2925:   - 但它当前更像 supporting 机制，而不是确定的论文主方法
experiments/exp367_single_support/codex_train_design.md:2317:experiments/exp367_single_support/codex_review.md:286:experiments/decisions.md:2930:   - 而在特征层直接补全 keypoint-level support
experiments/exp367_single_support/codex_train_design.md:2318:experiments/exp367_single_support/codex_review.md:287:experiments/decisions.md:2947:**上下文**: exp142 SKC 训练完成。最终结果 mAP 60.3% / R1 71.8%（equal_concat），相对 exp030a -0.8% mAP / -1.9% R1。feature-level support-supervised completion 方向确认失败。
experiments/exp367_single_support/codex_train_design.md:2319:experiments/exp367_single_support/codex_review.md:288:experiments/decisions.md:2950:1. SKC completion 模块虽然活跃（gate≈0.26, delta_norm≈1.5），但 skc_pre≈skc_post 说明修改方向不是向 support prototype 靠近
experiments/exp367_single_support/codex_train_design.md:2320:experiments/exp367_single_support/codex_review.md:289:experiments/decisions.md:2980:1. `single-image support incomplete` 这个问题定义没有被推翻
experiments/exp367_single_support/codex_train_design.md:2321:experiments/exp367_single_support/codex_review.md:290:experiments/decisions.md:2996:   - 单图能否被改写成“伪多 support 学习”对象
experiments/exp367_single_support/codex_train_design.md:2322:experiments/exp367_single_support/codex_review.md:291:experiments/decisions.md:3479:- `exp109` oracle support bank 仍是仓库内最强问题证据
experiments/exp367_single_support/codex_train_design.md:2323:experiments/exp367_single_support/codex_review.md:292:experiments/decisions.md:3486:  B. 回到 `exp109`，把主线改成“single-image support incomplete”的训练对象重写
experiments/exp367_single_support/codex_train_design.md:2324:experiments/exp367_single_support/codex_review.md:293:experiments/decisions.md:3493:3. `MaxSim / POT / flip` 主要仍是 test-time supporting evidence，不能作为训练端主贡献
experiments/exp367_single_support/codex_train_design.md:2325:experiments/exp367_single_support/codex_review.md:294:experiments/decisions.md:3501:1. 用 pose 定义互补 support 伪视图，而不是随机多视图分类
experiments/exp367_single_support/codex_train_design.md:2326:experiments/exp367_single_support/codex_review.md:295:experiments/decisions.md:3502:2. 用互补视图组装 support-complete teacher token set
experiments/exp367_single_support/codex_train_design.md:2327:experiments/exp367_single_support/codex_review.md:296:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:2328:experiments/exp367_single_support/codex_review.md:297:experiments/decisions.md:3528:  A. 继续沿刚提出的 `PSCD/support-complete` 新路线展开
experiments/exp367_single_support/codex_train_design.md:2329:experiments/exp367_single_support/codex_review.md:298:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2330:experiments/exp367_single_support/codex_review.md:301:experiments/decisions.md:4696:- **测试 C Singleton Merge = DEAD**: NN-is-head 0.72 只反映 head 占 72% 图像质量。per-head-ID(n=450/311 真功效)Spearman(support, attraction-PER-IMAGE)+0.003/+0.005≈0, 分箱 per-image 甚至下降。support-calibrated 阈值几乎无增益(d≈−0.003)且 40-60% level 退回 global。被 "head 图多→NN 彩票多" trivial count 吃掉。
experiments/exp367_single_support/codex_train_design.md:2331:experiments/exp367_single_support/codex_review.md:307:experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:2332:experiments/exp367_single_support/codex_review.md:308:experiments/exp367_single_support/codex_review.md:16:我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_train_design.md:2333:experiments/exp367_single_support/codex_review.md:309:experiments/exp367_single_support/cvpb_single_support_probe.py:4:codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_train_design.md:2334:experiments/exp367_single_support/codex_review.md:310:experiments/exp367_single_support/cvpb_single_support_probe.py:5:回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_train_design.md:2335:experiments/exp367_single_support/codex_review.md:311:experiments/exp367_single_support/cvpb_single_support_probe.py:10:  - random-support (每 ID 随机 1 图) : 随机单 support
experiments/exp367_single_support/codex_train_design.md:2336:experiments/exp367_single_support/codex_review.md:312:experiments/exp367_single_support/cvpb_single_support_probe.py:11:  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_train_design.md:2337:experiments/exp367_single_support/codex_review.md:313:experiments/exp367_single_support/cvpb_single_support_probe.py:12:  - best-support (每 ID 选最好 1 图)  : support 选择 oracle 上界
experiments/exp367_single_support/codex_train_design.md:2338:experiments/exp367_single_support/codex_review.md:314:experiments/exp367_single_support/cvpb_single_support_probe.py:14:GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_train_design.md:2339:experiments/exp367_single_support/codex_review.md:315:experiments/exp367_single_support/cvpb_single_support_probe.py:15:  worst 比 full 掉 > 3 mAP  AND  best - worst gap > 3 mAP (support 选择 matters)。
experiments/exp367_single_support/codex_train_design.md:2340:experiments/exp367_single_support/codex_review.md:316:experiments/exp367_single_support/cvpb_single_support_probe.py:16:DEAD: best≈worst (哪张 support 都一样, 没 support 选择价值) 或 single≈full (单图够)。
experiments/exp367_single_support/codex_train_design.md:2341:experiments/exp367_single_support/codex_review.md:317:experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_train_design.md:2342:experiments/exp367_single_support/codex_review.md:318:experiments/exp367_single_support/cvpb_single_support_probe.py:19:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:2343:experiments/exp367_single_support/codex_review.md:319:experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_train_design.md:2344:experiments/exp367_single_support/codex_review.md:320:experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_train_design.md:2345:experiments/exp367_single_support/codex_review.md:321:experiments/exp367_single_support/cvpb_single_support_probe.py:57:# random-support: 每 ID 随机 1 图
experiments/exp367_single_support/codex_train_design.md:2346:experiments/exp367_single_support/codex_review.md:322:experiments/exp367_single_support/cvpb_single_support_probe.py:62:# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_train_design.md:2347:experiments/exp367_single_support/codex_review.md:323:experiments/exp367_single_support/cvpb_single_support_probe.py:63:# 用 该 ID 的 gallery 图 与 该 ID 所有 query 的平均 sim 作为 support quality (高=好 support)
experiments/exp367_single_support/codex_train_design.md:2348:experiments/exp367_single_support/codex_review.md:324:experiments/exp367_single_support/cvpb_single_support_probe.py:69:    # 每个候选 support 图 g 对 同 ID query 的平均 cos (排同 cam)
experiments/exp367_single_support/codex_train_design.md:2349:experiments/exp367_single_support/codex_review.md:325:experiments/exp367_single_support/cvpb_single_support_probe.py:82:print(f'  best-support     : mAP={best_mAP:.2f} R1={best_R1:.2f}  (vs full {best_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2350:experiments/exp367_single_support/codex_review.md:326:experiments/exp367_single_support/cvpb_single_support_probe.py:83:print(f'  random-support   : mAP={rand_mAP:.2f} R1={rand_R1:.2f}  (vs full {rand_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2351:experiments/exp367_single_support/codex_review.md:327:experiments/exp367_single_support/cvpb_single_support_probe.py:84:print(f'  worst-support    : mAP={worst_mAP:.2f} R1={worst_R1:.2f}  (vs full {worst_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:2352:experiments/exp367_single_support/codex_review.md:328:experiments/exp367_single_support/cvpb_single_support_probe.py:87:print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_train_design.md:2353:experiments/exp367_single_support/codex_review.md:329:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:2354:experiments/exp367_single_support/codex_review.md:330:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2355:experiments/exp367_single_support/codex_review.md:331:experiments/innovation_brainstorm.md:408:经过 `exp110-126`，当前最重要的收束不是“support-complete 有没有价值”，而是：
experiments/exp367_single_support/codex_train_design.md:2356:experiments/exp367_single_support/codex_review.md:332:experiments/innovation_brainstorm.md:417:**让 support-complete prototype 以“可学习残差 prior”的形式进入 keypoint branch。**
experiments/exp367_single_support/codex_train_design.md:2357:experiments/exp367_single_support/codex_review.md:333:experiments/innovation_brainstorm.md:433:  1. 问题层面仍锚定 `single-image support incomplete`
experiments/exp367_single_support/codex_train_design.md:2358:experiments/exp367_single_support/codex_review.md:334:experiments/innovation_brainstorm.md:434:  2. 机制层面从 “memory bank / routing trick” 升级成了真正的 **support-conditioned completion**
experiments/exp367_single_support/codex_train_design.md:2359:experiments/exp367_single_support/codex_review.md:335:experiments/innovation_brainstorm.md:444:- 这说明 common-support 不是噪声，而是真实的 pairwise 证据
experiments/exp367_single_support/codex_train_design.md:2360:experiments/exp367_single_support/codex_review.md:336:experiments/innovation_brainstorm.md:462:1. 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
experiments/exp367_single_support/codex_train_design.md:2361:experiments/exp367_single_support/codex_review.md:337:experiments/innovation_brainstorm.md:463:2. 在 global branch 上增加一条 support-aware triplet
experiments/exp367_single_support/codex_train_design.md:2362:experiments/exp367_single_support/codex_review.md:338:experiments/innovation_brainstorm.md:481:**核心教训**: 把 retrieval-time 的 common-support 信号迁到训练端，不能简单用 overlap 做 mining filter。retrieval-time CVK 有效是因为它改变了距离计算方式（只在共同可见关键点上计算距离），而不是因为它筛选了更好的 pair。
experiments/exp367_single_support/codex_train_design.md:2363:experiments/exp367_single_support/codex_review.md:339:experiments/innovation_brainstorm.md:485:2. 如果要做训练端 common-support，必须改变 loss 本身的距离计算（如只在共同可见区域上计算 triplet 距离）
experiments/exp367_single_support/codex_train_design.md:2364:experiments/exp367_single_support/codex_review.md:340:experiments/innovation_brainstorm.md:858:   **global identity space + balanced common-support correction**
experiments/exp367_single_support/codex_train_design.md:2365:experiments/exp367_single_support/codex_review.md:341:experiments/innovation_brainstorm.md:885:   - **CVK 主要做 deeper-rank common-support correction**
experiments/exp367_single_support/codex_train_design.md:2366:experiments/exp367_single_support/codex_review.md:342:experiments/innovation_brainstorm.md:1498:4. 若首轮无正信号，立即止损，回退到 retrieval-time `common-support recovery`
experiments/exp367_single_support/codex_train_design.md:2367:experiments/exp367_single_support/codex_review.md:343:experiments/innovation_brainstorm.md:1569:   **在 per-keypoint / common-visible support 层面做 duplicate-aware confuser reasoning**。
experiments/exp367_single_support/codex_train_design.md:2368:experiments/exp367_single_support/codex_review.md:344:experiments/innovation_brainstorm.md:1577:  3. per-keypoint / common-support 粒度
experiments/exp367_single_support/codex_train_design.md:2369:experiments/exp367_single_support/codex_review.md:345:experiments/innovation_brainstorm.md:1589:  - 就连 `per-keypoint / common-support` 层面的 test-time confuser penalty 也不稳定
experiments/exp367_single_support/codex_train_design.md:2370:experiments/exp367_single_support/codex_review.md:347:experiments/innovation_brainstorm.md:1611:  - “support-complete latent representation” 的 headroom 非常大
experiments/exp367_single_support/codex_train_design.md:2371:experiments/exp367_single_support/codex_review.md:348:experiments/innovation_brainstorm.md:1616:   **模型没有学会从单图中逼近完整 identity support。**
experiments/exp367_single_support/codex_train_design.md:2372:experiments/exp367_single_support/codex_review.md:349:experiments/innovation_brainstorm.md:1618:   - support 来源太弱
experiments/exp367_single_support/codex_train_design.md:2373:experiments/exp367_single_support/codex_review.md:350:experiments/innovation_brainstorm.md:1622:   **same-ID support bank → single-image support-complete distillation**
experiments/exp367_single_support/codex_train_design.md:2374:experiments/exp367_single_support/codex_review.md:351:experiments/innovation_brainstorm.md:1640:  - `support-complete` 不是只存在于上界分析里的幻觉
experiments/exp367_single_support/codex_train_design.md:2375:experiments/exp367_single_support/codex_review.md:352:experiments/innovation_brainstorm.md:1644:1. 当前最值得继续赌的，不再是“有没有必要做 support-complete”，而是：
experiments/exp367_single_support/codex_train_design.md:2376:experiments/exp367_single_support/codex_review.md:353:experiments/innovation_brainstorm.md:1645:   **怎样让 prototype teacher 更可靠、更接近真正的 multi-view support。**
experiments/exp367_single_support/codex_train_design.md:2377:experiments/exp367_single_support/codex_review.md:354:experiments/innovation_brainstorm.md:1651:   **reliable-support bank / teacher reliability gating**
experiments/exp367_single_support/codex_train_design.md:2378:experiments/exp367_single_support/codex_review.md:355:experiments/innovation_brainstorm.md:1655:  1. 问题不是简单 occlusion comparison，而是 single-image support incomplete
experiments/exp367_single_support/codex_train_design.md:2379:experiments/exp367_single_support/codex_review.md:356:experiments/innovation_brainstorm.md:1656:  2. 方法不是通用补全 decoder，而是 identity-level support-complete distillation
experiments/exp367_single_support/codex_train_design.md:2380:experiments/exp367_single_support/codex_review.md:357:experiments/innovation_brainstorm.md:1666:- 结果几乎等价，说明“要求多个 support 样本共同支撑 teacher”这件事本身，并没有把当前增益显著放大。
experiments/exp367_single_support/codex_train_design.md:2381:experiments/exp367_single_support/codex_review.md:358:experiments/innovation_brainstorm.md:1669:1. 当前 `support-complete` 主线并没有被否定，因为结果仍保持正向区间。
experiments/exp367_single_support/codex_train_design.md:2382:experiments/exp367_single_support/codex_review.md:359:experiments/innovation_brainstorm.md:1672:   **teacher purity / write quality / support cleanliness**
experiments/exp367_single_support/codex_train_design.md:2383:experiments/exp367_single_support/codex_review.md:360:experiments/innovation_brainstorm.md:1676:- 基于 support 置信度的 soft reliability weighting
experiments/exp367_single_support/codex_train_design.md:2384:experiments/exp367_single_support/codex_review.md:361:experiments/innovation_brainstorm.md:1683:- `exp112` 说明更干净的 support 写入有用，但当前只形成弱正向：
experiments/exp367_single_support/codex_train_design.md:2385:experiments/exp367_single_support/codex_review.md:362:experiments/innovation_brainstorm.md:1691:1. 当前最值得讲的主创新，已经不只是 “support-complete distillation”。
experiments/exp367_single_support/codex_train_design.md:2386:experiments/exp367_single_support/codex_review.md:363:experiments/innovation_brainstorm.md:1693:   **如何在 pose-aligned support-complete learning 中控制 teacher hardening / non-stationary target。**
experiments/exp367_single_support/codex_train_design.md:2387:experiments/exp367_single_support/codex_review.md:364:experiments/innovation_brainstorm.md:1701:- Lagged / stale support bank
experiments/exp367_single_support/codex_train_design.md:2388:experiments/exp367_single_support/codex_review.md:365:experiments/innovation_brainstorm.md:1802:  1. `cvk_hybrid` 说明 common-support 的 pairwise 几何是真实的
experiments/exp367_single_support/codex_train_design.md:2389:experiments/exp367_single_support/codex_review.md:366:experiments/innovation_brainstorm.md:1805:  4. `exp109-116` 说明 `support-complete` 若被压成 `per-ID prototype`，会损失 pair-specific 细节
experiments/exp367_single_support/codex_train_design.md:2390:experiments/exp367_single_support/codex_review.md:367:experiments/innovation_brainstorm.md:1812:- 不再把 support 压成 prototype
experiments/exp367_single_support/codex_train_design.md:2391:experiments/exp367_single_support/codex_review.md:368:experiments/innovation_brainstorm.md:1826:2. 机制层面：pose/keypoint branch 作为 **common-support relational teacher**
experiments/exp367_single_support/codex_train_design.md:2392:experiments/exp367_single_support/codex_review.md:369:experiments/innovation_brainstorm.md:1827:3. 训练目标：把 global embedding 蒸馏成更符合 common-support 几何的空间
experiments/exp367_single_support/codex_train_design.md:2393:experiments/exp367_single_support/codex_review.md:370:experiments/innovation_brainstorm.md:1840:   **teacher 自身还是单图 `kp_feats`，并不 support-complete**
experiments/exp367_single_support/codex_train_design.md:2394:experiments/exp367_single_support/codex_review.md:371:experiments/innovation_brainstorm.md:1849:3. 而是先用 `exp109` 方向的 support bank 补全 low-vis keypoint teacher，再用补全后的 teacher 去做 `CSRD`
experiments/exp367_single_support/codex_train_design.md:2395:experiments/exp367_single_support/codex_review.md:372:experiments/innovation_brainstorm.md:1852:1. `exp109` 已证明 support-complete teacher 有巨大 headroom
experiments/exp367_single_support/codex_train_design.md:2396:experiments/exp367_single_support/codex_review.md:373:experiments/innovation_brainstorm.md:1856:   **support-complete teacher + relational distillation**
experiments/exp367_single_support/codex_train_design.md:2397:experiments/exp367_single_support/codex_review.md:374:experiments/innovation_brainstorm.md:1861:- 但这次不能简单说 `support-complete teacher` 失败，因为机制统计很清楚：
experiments/exp367_single_support/codex_train_design.md:2398:experiments/exp367_single_support/codex_review.md:375:experiments/innovation_brainstorm.md:1870:   **support-complete 监督的收益主要属于 support-incomplete 样本，被 clean 样本等权平均后稀释掉了**
experiments/exp367_single_support/codex_train_design.md:2399:experiments/exp367_single_support/codex_review.md:376:experiments/innovation_brainstorm.md:1877:1. 保持 `exp120` 的 support-complete relational teacher 完全不变
experiments/exp367_single_support/codex_train_design.md:2400:experiments/exp367_single_support/codex_review.md:377:experiments/innovation_brainstorm.md:1880:   - 它有多少 keypoint 真正被 support-complete teacher 补全
experiments/exp367_single_support/codex_train_design.md:2401:experiments/exp367_single_support/codex_review.md:378:experiments/innovation_brainstorm.md:1903:   **support-complete teacher 实际只改变了一部分 pairwise 关系，distillation 应聚焦这些 pair-change relations**
experiments/exp367_single_support/codex_train_design.md:2402:experiments/exp367_single_support/codex_review.md:379:experiments/innovation_brainstorm.md:1910:1. 保持 `exp120` 的 support-complete teacher 完全不变
experiments/exp367_single_support/codex_train_design.md:2403:experiments/exp367_single_support/codex_review.md:380:experiments/innovation_brainstorm.md:1914:   - support-complete teacher 几何
experiments/exp367_single_support/codex_train_design.md:2404:experiments/exp367_single_support/codex_review.md:381:experiments/innovation_brainstorm.md:1915:4. 对那些 **被 support completion 真正改变过的 pair** 赋予更高 distillation focus
experiments/exp367_single_support/codex_train_design.md:2405:experiments/exp367_single_support/codex_review.md:382:experiments/innovation_brainstorm.md:1927:  **teacher stability = supporting mechanism**
experiments/exp367_single_support/codex_train_design.md:2406:experiments/exp367_single_support/codex_review.md:383:experiments/innovation_brainstorm.md:1949:1. `exp121` 已说明 freeze 只是 supporting，不值得再扩成一条线
experiments/exp367_single_support/codex_train_design.md:2407:experiments/exp367_single_support/codex_review.md:384:experiments/innovation_brainstorm.md:1979:1. 保持 `exp123/124` 的 support-complete relational teacher 完全不变
experiments/exp367_single_support/codex_train_design.md:2408:experiments/exp367_single_support/codex_review.md:385:experiments/innovation_brainstorm.md:1995:  **stable teacher 只是 supporting mechanism，不再值得单独扩线**
experiments/exp367_single_support/codex_train_design.md:2409:experiments/exp367_single_support/codex_review.md:386:experiments/innovation_brainstorm.md:1998:1. `support-complete teacher` 的新增信息是真实存在的
experiments/exp367_single_support/codex_train_design.md:2410:experiments/exp367_single_support/codex_review.md:387:experiments/innovation_brainstorm.md:2001:4. 于是 support-complete 带来的那部分新增 correction，极可能被 base teacher 的主体结构稀释掉
experiments/exp367_single_support/codex_train_design.md:2411:experiments/exp367_single_support/codex_review.md:388:experiments/innovation_brainstorm.md:2012:4. 让 global embedding 学习的不是“再复刻一遍 skeleton teacher”，而是只学 **support completion 真正带来的关系修正**
experiments/exp367_single_support/codex_train_design.md:2412:experiments/exp367_single_support/codex_review.md:389:experiments/innovation_brainstorm.md:2041:1. 保留 `exp125` 当前最强的 online support teacher 与 `delta_top` routing
experiments/exp367_single_support/codex_train_design.md:2413:experiments/exp367_single_support/codex_review.md:390:experiments/innovation_brainstorm.md:2053:   **让 student 在更大的 relation support 上学习 support-complete comparability correction**
experiments/exp367_single_support/codex_train_design.md:2414:experiments/exp367_single_support/codex_review.md:391:experiments/innovation_brainstorm.md:2084:1. 不再强迫单个 global embedding 吃下 support-complete correction
experiments/exp367_single_support/codex_train_design.md:2415:experiments/exp367_single_support/codex_review.md:392:experiments/innovation_brainstorm.md:2088:   - 该在多大程度上相信 common-support distance
experiments/exp367_single_support/codex_train_design.md:2416:experiments/exp367_single_support/codex_review.md:393:experiments/innovation_brainstorm.md:2090:   - 用 `support-complete teacher` 提供更理想的 pairwise target
experiments/exp367_single_support/codex_train_design.md:2417:experiments/exp367_single_support/codex_review.md:394:experiments/innovation_brainstorm.md:2131:   - 必要时再加入更细的 keypoint-wise common-support statistics
experiments/exp367_single_support/codex_train_design.md:2418:experiments/exp367_single_support/codex_review.md:395:experiments/innovation_brainstorm.md:2166:### 2026-03-21 晚间更新：`LPCS` 已经真正成立，但 sparse routing 最终只是 supporting 机制
experiments/exp367_single_support/codex_train_design.md:2419:experiments/exp367_single_support/codex_review.md:396:experiments/innovation_brainstorm.md:2199:- `pose-defined common support`
experiments/exp367_single_support/codex_train_design.md:2420:experiments/exp367_single_support/codex_review.md:397:experiments/innovation_brainstorm.md:2238:   - 方案：给每个 pair descriptor 追加 query 的正负均值距离、margin、support 完整度与 teacher change 统计
experiments/exp367_single_support/codex_train_design.md:2421:experiments/exp367_single_support/codex_review.md:398:experiments/innovation_brainstorm.md:2241:- `pose-defined common support`
experiments/exp367_single_support/codex_train_design.md:2422:experiments/exp367_single_support/codex_review.md:399:experiments/innovation_brainstorm.md:2266:### 2026-03-22 当前收敛：`rank-decay` 退为 supporting，`query-context correction` 升为主候选
experiments/exp367_single_support/codex_train_design.md:2423:experiments/exp367_single_support/codex_review.md:400:experiments/innovation_brainstorm.md:2271:  - 它证明了“平滑 top-sensitive”比 `hard-rank` 合理，但最终只形成 supporting 级别的改进
experiments/exp367_single_support/codex_train_design.md:2424:experiments/exp367_single_support/codex_review.md:401:experiments/innovation_brainstorm.md:2283:3. 从而让同样的 common support 在不同 query 上被不同地解释
experiments/exp367_single_support/codex_train_design.md:2425:experiments/exp367_single_support/codex_review.md:402:experiments/innovation_brainstorm.md:2307:1. pose 定义 common support
experiments/exp367_single_support/codex_train_design.md:2426:experiments/exp367_single_support/codex_review.md:403:experiments/innovation_brainstorm.md:2308:2. support-complete teacher 提供 correction 方向
experiments/exp367_single_support/codex_train_design.md:2427:experiments/exp367_single_support/codex_review.md:404:experiments/innovation_brainstorm.md:2322:     - 同一份 common support，是否需要放在 query-level 语境里解释
experiments/exp367_single_support/codex_train_design.md:2428:experiments/exp367_single_support/codex_review.md:405:experiments/innovation_brainstorm.md:2351:3. `support_rank`
experiments/exp367_single_support/codex_train_design.md:2429:experiments/exp367_single_support/codex_review.md:406:experiments/innovation_brainstorm.md:2357:- pose 定义 common support
experiments/exp367_single_support/codex_train_design.md:2430:experiments/exp367_single_support/codex_review.md:407:experiments/innovation_brainstorm.md:2362:- 都还紧扣 `exp109` 的核心发现：单图 support 不完整
experiments/exp367_single_support/codex_train_design.md:2431:experiments/exp367_single_support/codex_review.md:408:experiments/innovation_brainstorm.md:2366:- `exp139` 强调 **如何解释 common support**
experiments/exp367_single_support/codex_train_design.md:2432:experiments/exp367_single_support/codex_review.md:409:experiments/innovation_brainstorm.md:2369:### 2026-03-21 本地大转向：从 pair correction 切回 feature-space support completion
experiments/exp367_single_support/codex_train_design.md:2433:experiments/exp367_single_support/codex_review.md:410:experiments/innovation_brainstorm.md:2394:1. pose 不再只是用来构造 `common support distance`
experiments/exp367_single_support/codex_train_design.md:2434:experiments/exp367_single_support/codex_review.md:411:experiments/innovation_brainstorm.md:2398:   - 哪些 support prototype 可作为跨图补全证据
experiments/exp367_single_support/codex_train_design.md:2435:experiments/exp367_single_support/codex_review.md:412:experiments/innovation_brainstorm.md:2403:- `exp109` 暴露出的单图 support incomplete，能否在编码阶段被修复
experiments/exp367_single_support/codex_train_design.md:2436:experiments/exp367_single_support/codex_review.md:413:experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:2437:experiments/exp367_single_support/codex_review.md:414:experiments/innovation_brainstorm.md:2470:- 单图 support incomplete 能否通过 **pose-defined complementary pseudo-views** 改写成“伪多 support 学习”？
experiments/exp367_single_support/codex_train_design.md:2438:experiments/exp367_single_support/codex_review.md:415:experiments/innovation_brainstorm.md:2481:- PCVT 直接改写训练对象，把单图变成“互补 support 组合体”
experiments/exp367_single_support/codex_train_design.md:2439:experiments/exp367_single_support/codex_review.md:416:experiments/innovation_brainstorm.md:2522:2. `single-image support incomplete` 可能确实更适合被改写成“伪多 support 学习对象”，而不是继续做 scorer / completion 小修补
experiments/exp367_single_support/codex_train_design.md:2440:experiments/exp367_single_support/codex_review.md:417:experiments/exp140/design.md:26:并用 support-complete teacher 诱导 `conf` 对齐“teacher 实际改动有多大”的 soft target，那么：
experiments/exp367_single_support/codex_train_design.md:2441:experiments/exp367_single_support/codex_review.md:418:experiments/exp140/design.md:42:3. 保留 support-complete teacher bank
experiments/exp367_single_support/codex_train_design.md:2442:experiments/exp367_single_support/codex_review.md:419:experiments/exp175/claude_review.md:14:**Is this just a small config change?**: This is a legitimate ablation/extension experiment. The multi-stage PSG code already exists (validated in exp173 with stages [2,3]). Extending to [0,1,2,3] is a valid ablation to answer "does full-stage PSG beat partial PSG and PAPE?" This is a fine experiment as a supporting/ablation result.
experiments/exp367_single_support/codex_train_design.md:2443:experiments/exp367_single_support/codex_review.md:420:experiments/clip_reid_compare/CLIP-ReID/datasets/make_dataloader.py:96:        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
experiments/exp367_single_support/codex_train_design.md:2444:experiments/exp367_single_support/codex_review.md:421:experiments/exp252/claude_review.md:17:- **Innovation concern**: This is a config combination experiment, NOT a new mechanism. Per CLAUDE.md rules, combination experiments should NOT be the main line. However, since this tests multi-stage injection (never tested before for PSG), it provides useful ablation evidence for the paper narrative. Acceptable as a supporting experiment.
experiments/exp367_single_support/codex_train_design.md:2445:experiments/exp367_single_support/codex_review.md:422:experiments/exp255/claude_review.md:175:This experiment is a hyperparameter sweep (GCN hidden 256 -> 512). It is not an innovation experiment. However, the context is clear: this is part of pushing the Small backbone recipe toward best results, not a main-line creative experiment. As a supporting capacity ablation for the paper's ablation table, it is acceptable.
experiments/exp367_single_support/codex_train_design.md:2446:experiments/exp367_single_support/codex_review.md:423:experiments/exp140/claude_review.md:63:- `teacher_dist`：support-complete bank 替换后的 global+kp 加权距离（label-free）
experiments/exp367_single_support/codex_train_design.md:2447:experiments/exp367_single_support/codex_review.md:424:experiments/clip_reid_compare/CLIP-ReID/datasets/sampler_ddp.py:57:    # we pad the tensor because torch all_gather does not support
experiments/exp367_single_support/codex_train_design.md:2448:experiments/exp367_single_support/codex_review.md:425:experiments/cargo_cvpb/fgeu_realizability_result.md:51:**fragility gate (只融弱 support 失败 vs 全融):**
experiments/exp367_single_support/codex_train_design.md:2449:experiments/exp367_single_support/codex_review.md:426:experiments/cargo_cvpb/fgeu_realizability_result.md:53:- fuse-FRAGILE-only (bottom-50% support) dAP = +5.51 (n=45)
experiments/exp367_single_support/codex_train_design.md:2450:experiments/exp367_single_support/codex_review.md:427:experiments/cargo_cvpb/claude_review_gallery_killswitch.md:14:| C2 | High | test_C 阈值 | support-calibrated false-merge 单 Zipf draw + sparse level 静默 fallback → 单点高方差 | ✅ 已修: 跨 n_zipf_seeds 平均, 报 fallback-to-global 比例 |
experiments/exp367_single_support/codex_train_design.md:2451:experiments/exp367_single_support/codex_review.md:439:experiments/exp141/design.md:8:2. `row_support_mean / row_gap_mean`
experiments/exp367_single_support/codex_train_design.md:2452:experiments/exp367_single_support/codex_review.md:440:experiments/exp141/design.md:13:- 当前这个 pair 的 common-support 改善是普遍现象还是稀有现象
experiments/exp367_single_support/codex_train_design.md:2453:experiments/exp367_single_support/codex_review.md:441:experiments/exp141/design.md:23:3. 当前 pair 的 `support_ratio` 相对排名
experiments/exp367_single_support/codex_train_design.md:2454:experiments/exp367_single_support/codex_review.md:442:experiments/exp141/design.md:28:- 什么时候 common-support correction 值得强用
experiments/exp367_single_support/codex_train_design.md:2455:experiments/exp367_single_support/codex_review.md:443:experiments/exp141/design.md:37:3. `support_rank`
experiments/exp367_single_support/codex_train_design.md:2456:experiments/exp367_single_support/codex_review.md:455:experiments/exp141/claude_review_v2.md:53:| `support_ratio` | `support_ratio.detach()` (batch x batch) | `support_ratio[start:end]` (chunk_q x gallery) |
experiments/exp367_single_support/codex_train_design.md:2457:experiments/exp367_single_support/codex_review.md:456:experiments/exp141/claude_review_v2.md:67:| 追加 5 维内容 | `row_mean, row_std, row_min, row_support_mean, row_gap_mean` | `base_rank, kp_rank, support_rank, gain_rank, gain_zscore` |
experiments/exp367_single_support/codex_train_design.md:2458:experiments/exp367_single_support/codex_review.md:457:experiments/exp141/claude_review_v2.md:80:- `kp_dist`: 来自 common-support 距离计算（detached）
experiments/exp367_single_support/codex_train_design.md:2459:experiments/exp367_single_support/codex_review.md:458:experiments/exp141/claude_review_v2.md:81:- `support_ratio`: 来自 keypoint weight 计算（detached）
experiments/exp367_single_support/codex_train_design.md:2460:experiments/exp367_single_support/codex_review.md:459:experiments/exp141/claude_review_v2.md:100:- ascending（距离越小排名越前）用 `inf` 填充 invalid；descending（support_ratio 越大排名越前）用 `-inf` 填充 invalid：正确
experiments/exp367_single_support/codex_train_design.md:2461:experiments/exp367_single_support/codex_review.md:460:experiments/exp141/claude_review_v2.md:115:- `base_dist.detach()`, `kp_dist.detach()`, `support_ratio.detach()` -> 所有 descriptor 输入无梯度
experiments/exp367_single_support/codex_train_design.md:2462:experiments/exp367_single_support/codex_review.md:463:experiments/exp126/design.md:34:- support-complete teacher 构造不变
experiments/exp367_single_support/codex_train_design.md:2463:experiments/exp367_single_support/codex_review.md:466:experiments/exp141/monitor.md:26:     - `support_rank`
experiments/exp367_single_support/codex_train_design.md:2464:experiments/exp367_single_support/codex_review.md:467:experiments/exp141/monitor.md:85:     - `exp142`: feature-space support-supervised completion
experiments/exp367_single_support/codex_train_design.md:2465:experiments/exp367_single_support/codex_review.md:470:experiments/cargo_cvpb/hub_verify_p0c_deep.py:8:exactly those false neighbors). This script isolates that single control cleanly on the
experiments/exp367_single_support/codex_train_design.md:2466:experiments/exp367_single_support/codex_review.md:491:experiments/exp126/monitor.md:18:- [x] support-complete teacher、bank 更新、主 loss 配比全部保持不变
experiments/exp367_single_support/codex_train_design.md:2467:experiments/exp367_single_support/codex_review.md:496:experiments/cargo_cvpb/cvpb_gallery_killswitch_DESIGN.md:33:  是否错并入 head prototype, false-merge rate 是否随 head support 单调上升。
experiments/exp367_single_support/codex_train_design.md:2468:experiments/exp367_single_support/codex_review.md:497:experiments/cargo_cvpb/cvpb_gallery_killswitch_DESIGN.md:34:- 比 GLOBAL 阈值 vs SUPPORT-CALIBRATED（按 support 分层校准）在同 head-recall 下的 tail false-merge。
experiments/exp367_single_support/codex_train_design.md:2469:experiments/exp367_single_support/codex_review.md:498:experiments/cargo_cvpb/cvpb_gallery_killswitch_DESIGN.md:36:  per-image rate 若 FLAT 则纯机械, 若仍随 support 升才是非平凡 over-attraction。
experiments/exp367_single_support/codex_train_design.md:2470:experiments/exp367_single_support/codex_review.md:500:experiments/cargo_cvpb/cvpb_gallery_killswitch_DESIGN.md:49:- A: 唯一变量=gallery size（注入 distractor）; B: 唯一变量=watchlist size; C: 唯一变量=head support。
experiments/exp367_single_support/codex_train_design.md:2471:experiments/exp367_single_support/codex_review.md:501:experiments/exp366_active_evidence/design.md:21:★**诚实设计**：避 codex 的 trivial oracle（multi-query 必涨 = upper-bound 不是创新），真验 policy（预算分配 vs random）。控 margin（top1-top2 = #false-in-topk 的代理）。自查抓到 2 个 bug（margins 长度 != len(qf) 退化 policy；policy hard 应只在 has_second 池选）已 fix。
experiments/exp367_single_support/codex_train_design.md:2472:experiments/exp367_single_support/codex_review.md:502:experiments/exp141/claude_review.md:62:- exp139 (`query_ctx`)：追加 5 维 **query 级常量**（`row_mean, row_std, row_min, row_support_mean, row_gap_mean`），同一 query 内所有 pair 共享同一组 context 值
experiments/exp367_single_support/codex_train_design.md:2473:experiments/exp367_single_support/codex_review.md:503:experiments/exp141/claude_review.md:63:- exp141 (`comp_ctx`)：追加 5 维 **pair-specific 相对竞争位置**（`base_rank, kp_rank, support_rank, gain_rank, gain_zscore`），同一 query 内每个 pair 的 context 值不同
experiments/exp367_single_support/codex_train_design.md:2474:experiments/exp367_single_support/codex_review.md:504:experiments/exp141/claude_review.md:81:**是的，无泄漏。** `build_query_competition_descriptors` 的所有输入（`base_dist, kp_dist, support_ratio`）均来自特征距离计算，不涉及任何标签。排名和 z-score 也都是纯统计量。
experiments/exp367_single_support/codex_train_design.md:2475:experiments/exp367_single_support/codex_review.md:505:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:14:  * the d14 "Evidence-Sufficient ReID" backup (single-image support insufficiency,
experiments/exp367_single_support/codex_train_design.md:2476:experiments/exp367_single_support/codex_review.md:506:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:19:  TEST 1 (positive-support explains the TAX residual):
experiments/exp367_single_support/codex_train_design.md:2477:experiments/exp367_single_support/codex_review.md:507:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:25:      Does positive-support explain the gallery-growth tax residual (the 1x->10x AP
experiments/exp367_single_support/codex_train_design.md:2478:experiments/exp367_single_support/codex_review.md:509:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:28:      positive-support must show different tax (partial Spearman survives controls).
experiments/exp367_single_support/codex_train_design.md:2479:experiments/exp367_single_support/codex_review.md:510:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:30:  TEST 2 (positive-support predicts per-query FAILURE):
experiments/exp367_single_support/codex_train_design.md:2480:experiments/exp367_single_support/codex_review.md:511:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:31:      ROC-AUC of positive-support predicting per-query AP-failure on the FULL gallery,
experiments/exp367_single_support/codex_train_design.md:2481:experiments/exp367_single_support/codex_review.md:513:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:36:      For LOW positive-support FAILURE queries, add a 2nd same-ID image and form an
experiments/exp367_single_support/codex_train_design.md:2482:experiments/exp367_single_support/codex_review.md:514:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:44:      -> the failure is EVIDENCE INSUFFICIENCY (more evidence fixes it), supporting d14.
experiments/exp367_single_support/codex_train_design.md:2483:experiments/exp367_single_support/codex_review.md:515:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:46:VERDICT: positive-support has an INDEPENDENT-of-trivial signal (survives TEST1 partial
experiments/exp367_single_support/codex_train_design.md:2484:experiments/exp367_single_support/codex_review.md:517:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:81:ap.add_argument('--low_support_quant', type=float, default=0.30,
experiments/exp367_single_support/codex_train_design.md:2485:experiments/exp367_single_support/codex_review.md:518:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:82:                help='bottom-q by positive-support among FAILURE queries -> the rescue target set')
experiments/exp367_single_support/codex_train_design.md:2486:experiments/exp367_single_support/codex_review.md:527:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:281:def positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, a_temp):
experiments/exp367_single_support/codex_train_design.md:2487:experiments/exp367_single_support/codex_review.md:533:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:394:    ps = positive_support(cqf, cq_pid, cq_cam, gf[core_idx], g_pid[core_idx], g_cam[core_idx], cli.a_temp)
experiments/exp367_single_support/codex_train_design.md:2488:experiments/exp367_single_support/codex_review.md:534:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:395:    # weak-positive RISK convention: higher = weaker support = predict bigger tax
experiments/exp367_single_support/codex_train_design.md:2489:experiments/exp367_single_support/codex_review.md:537:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:415:    print(f"\n[1] raw Spearman(positive-support risk, tax) over {int(ev.sum())} valid core queries:")
experiments/exp367_single_support/codex_train_design.md:2490:experiments/exp367_single_support/codex_review.md:539:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:422:    # ★ LIFE-OR-DEATH partials: positive-support vs tax controlling 1x-margin AND #false
experiments/exp367_single_support/codex_train_design.md:2491:experiments/exp367_single_support/codex_review.md:541:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:424:    print(f"\n[1] ★PARTIAL Spearman(support-risk, tax | 1x-top1-margin + #false-in-topk):")
experiments/exp367_single_support/codex_train_design.md:2492:experiments/exp367_single_support/codex_review.md:542:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:431:    # reverse direction (do the trivials survive controlling support? — fairness check)
experiments/exp367_single_support/codex_train_design.md:2493:experiments/exp367_single_support/codex_review.md:544:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:435:    print(f"     [reverse] 1x-margin | support  = {pr_m:+.4f}   #false | support = {pr_f:+.4f}")
experiments/exp367_single_support/codex_train_design.md:2494:experiments/exp367_single_support/codex_review.md:545:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:437:    # combined: does support add to a logistic predicting big-tax over trivials? (OOF AUC)
experiments/exp367_single_support/codex_train_design.md:2495:experiments/exp367_single_support/codex_review.md:547:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:451:    print(f"\n[1] big-tax (top-30% tax) OOF-AUC: trivials={a_triv:.4f}  +support={a_both:.4f}  "
experiments/exp367_single_support/codex_train_design.md:2496:experiments/exp367_single_support/codex_review.md:548:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:452:          f"support-solo={a_supp:.4f}  >> INCREMENT={a_both-a_triv:+.4f}")
experiments/exp367_single_support/codex_train_design.md:2497:experiments/exp367_single_support/codex_review.md:551:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:482:    # positive-support on the FULL gallery (cross-cam positives only)
experiments/exp367_single_support/codex_train_design.md:2498:experiments/exp367_single_support/codex_review.md:552:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:483:    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
experiments/exp367_single_support/codex_train_design.md:2499:experiments/exp367_single_support/codex_review.md:554:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:488:        'lowtail-pos(soft-min)': -ps['lowtail'],       # ★ support: weak = high risk
experiments/exp367_single_support/codex_train_design.md:2500:experiments/exp367_single_support/codex_review.md:555:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:489:        'pos-dispersion':         ps['disp'],          # ★ support
experiments/exp367_single_support/codex_train_design.md:2501:experiments/exp367_single_support/codex_review.md:556:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:490:        '#cross-cam-pos(neg)':   -ps['ncc'],           # ★ support
experiments/exp367_single_support/codex_train_design.md:2502:experiments/exp367_single_support/codex_review.md:559:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:530:    print(f"     trivials + support          OOF-AUC = {a_both:.4f}")
experiments/exp367_single_support/codex_train_design.md:2503:experiments/exp367_single_support/codex_review.md:560:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:531:    print(f"     support-only (3 proxies)    OOF-AUC = {a_supp:.4f}")
experiments/exp367_single_support/codex_train_design.md:2504:experiments/exp367_single_support/codex_review.md:561:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:532:    print(f"     >> INCREMENT support adds on top of trivials = {a_both-a_triv:+.4f}")
experiments/exp367_single_support/codex_train_design.md:2505:experiments/exp367_single_support/codex_review.md:562:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:533:    print(f"     >> best support AUC - best trivial AUC        = {best_supp-best_triv:+.4f}")
experiments/exp367_single_support/codex_train_design.md:2506:experiments/exp367_single_support/codex_review.md:563:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:534:    # partial spearman: best support var vs continuous (-AP) controlling all trivials
experiments/exp367_single_support/codex_train_design.md:2507:experiments/exp367_single_support/codex_review.md:564:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:609:    # full-gallery per-query AP + failure + positive-support
experiments/exp367_single_support/codex_train_design.md:2508:experiments/exp367_single_support/codex_review.md:566:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:612:    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
experiments/exp367_single_support/codex_train_design.md:2509:experiments/exp367_single_support/codex_review.md:567:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:613:    support = ps['lowtail']                             # higher = stronger support
experiments/exp367_single_support/codex_train_design.md:2510:experiments/exp367_single_support/codex_review.md:568:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:615:    # FAILURE = bottom-30% AP; among failures, LOW-SUPPORT = bottom-q by support
experiments/exp367_single_support/codex_train_design.md:2511:experiments/exp367_single_support/codex_review.md:569:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:621:    # low-support subset among failures (need >=2 same-ID query imgs to do union -> see below)
experiments/exp367_single_support/codex_train_design.md:2512:experiments/exp367_single_support/codex_review.md:570:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:622:    supp_fail = support[fail_idx]
experiments/exp367_single_support/codex_train_design.md:2513:experiments/exp367_single_support/codex_review.md:571:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:625:    nlow = int(round(cli.low_support_quant * len(fidx2)))
experiments/exp367_single_support/codex_train_design.md:2514:experiments/exp367_single_support/codex_review.md:572:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:626:    low_order = np.argsort(supp2)                       # weakest support first
experiments/exp367_single_support/codex_train_design.md:2515:experiments/exp367_single_support/codex_review.md:573:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:627:    low_support_fail = fidx2[low_order[:nlow]]
experiments/exp367_single_support/codex_train_design.md:2516:experiments/exp367_single_support/codex_review.md:574:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:629:          f"low-support failures(bot-{cli.low_support_quant:.0%})={len(low_support_fail)}")
experiments/exp367_single_support/codex_train_design.md:2517:experiments/exp367_single_support/codex_review.md:575:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:631:    # for each low-support failure query, we need a SECOND same-ID query image to union.
experiments/exp367_single_support/codex_train_design.md:2518:experiments/exp367_single_support/codex_review.md:576:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:658:    for qi in low_support_fail:
experiments/exp367_single_support/codex_train_design.md:2519:experiments/exp367_single_support/codex_review.md:577:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:684:    print(f"\n[3] oracle multi-query on {n} low-support failure queries (mean AP, %):")
experiments/exp367_single_support/codex_train_design.md:2520:experiments/exp367_single_support/codex_review.md:578:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:699:    # then index out our selected low-support failure rows. We compare against base AP computed
experiments/exp367_single_support/codex_train_design.md:2521:experiments/exp367_single_support/codex_review.md:579:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:703:          f"lam={cli.krecip_lambda}) then index the SAME {n} low-support failure queries:")
experiments/exp367_single_support/codex_train_design.md:2522:experiments/exp367_single_support/codex_review.md:582:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:751:    print(f"[T2] failure-AUC: best support-trivial gap={T2['best_supp_minus_triv']:+.3f}  "
experiments/exp367_single_support/codex_train_design.md:2523:experiments/exp367_single_support/codex_review.md:583:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:752:          f"OOF incr={T2['incr']:+.3f}  support-solo AUC={T2['supp_solo']:.3f}  "
experiments/exp367_single_support/codex_train_design.md:2524:experiments/exp367_single_support/codex_review.md:584:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:754:    print(f"[T3] oracle on n={T3['n']} low-support failures: base={T3['base']*100:.2f} -> "
experiments/exp367_single_support/codex_train_design.md:2525:experiments/exp367_single_support/codex_review.md:585:experiments/exp119/design.md:6:- `exp051 PAML` 中性，说明 **只改 part triplet 的距离形式** 也不足以把 pairwise common-support 几何传给 global embedding。
experiments/exp367_single_support/codex_train_design.md:2526:experiments/exp367_single_support/codex_review.md:586:experiments/exp119/design.md:7:- `exp109-116` 又说明：把 support 压成 `per-ID EMA prototype` 会损失太多 pair-specific 细节。
experiments/exp367_single_support/codex_train_design.md:2527:experiments/exp367_single_support/codex_review.md:587:experiments/exp119/design.md:10:**如何把 keypoint/common-support 分支已经掌握的 pairwise 比较几何，直接蒸馏给 global embedding。**
experiments/exp367_single_support/codex_train_design.md:2528:experiments/exp367_single_support/codex_review.md:588:experiments/exp119/design.md:51:  - `exp110-116`：prototype-bank support-complete 路线天花板已现
experiments/exp367_single_support/codex_train_design.md:2529:experiments/exp367_single_support/codex_review.md:589:experiments/exp119/design.md:58:  - 后续 `cvk_hybrid` 增益缩小，说明训练端已吸收部分 common-support 几何
experiments/exp367_single_support/codex_train_design.md:2530:experiments/exp367_single_support/codex_review.md:596:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:75:[C] P(tail-probe NN is a HEAD of support s) by support bin (DESCRIPTIVE, n=4 bins):
experiments/exp367_single_support/codex_train_design.md:2531:experiments/exp367_single_support/codex_review.md:597:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:81:  [per-IMAGE rate FLAT across support -> purely mechanical count; RISING -> heads over-attract disproportionately.]
experiments/exp367_single_support/codex_train_design.md:2532:experiments/exp367_single_support/codex_review.md:598:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:82:  binned Spearman(support, rate/headID)=+0.4000 (trivially >0)  rate/IMAGE=-0.8000  [n=4 bins, descriptive only]
experiments/exp367_single_support/codex_train_design.md:2533:experiments/exp367_single_support/codex_review.md:599:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:85:     Spearman(support, attraction-count)     = +0.0428  [trivially >0: more imgs = more NN tickets]
experiments/exp367_single_support/codex_train_design.md:2534:experiments/exp367_single_support/codex_review.md:600:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:86:     Spearman(support, attraction-PER-IMAGE) = -0.0127  [NON-TRIVIAL: >0 means heads over-attract beyond count; ~0 means purely count]
experiments/exp367_single_support/codex_train_design.md:2535:experiments/exp367_single_support/codex_review.md:601:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:88:[C] support-calibrated vs global threshold (OVERALL tail->head false-merge at matched head-recall), CAL=even seeds / EVAL=odd seeds:
experiments/exp367_single_support/codex_train_design.md:2536:experiments/exp367_single_support/codex_review.md:602:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:89:  head-recall=0.90: OVERALL false-merge  global=0.0373  support-calibrated=0.0373  (d=+0.0000; want NEGATIVE)  [5 eval seeds, 1500 tail probes]
experiments/exp367_single_support/codex_train_design.md:2537:experiments/exp367_single_support/codex_review.md:603:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:90:  head-recall=0.95: OVERALL false-merge  global=0.0853  support-calibrated=0.0840  (d=-0.0013; want NEGATIVE)  [5 eval seeds, 1500 tail probes]
experiments/exp367_single_support/codex_train_design.md:2538:experiments/exp367_single_support/codex_review.md:604:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:91:  support-level fallback-to-global fraction (sparse levels) = 0.000  [high -> 'support-calibrated' is mostly global]
experiments/exp367_single_support/codex_train_design.md:2539:experiments/exp367_single_support/codex_review.md:606:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:99:[C] per-head-ID Spearman(support, attraction)=+0.043 (trivial)  PER-IMAGE=-0.013 (non-trivial claim)  [n=450 IDs]
experiments/exp367_single_support/codex_train_design.md:2540:experiments/exp367_single_support/codex_review.md:608:experiments/exp119/monitor.md:28:  - `CSRD` 是当前最直接的新机制验证：不用 prototype，而是直接蒸馏 common-support 关系
experiments/exp367_single_support/codex_train_design.md:2541:experiments/exp367_single_support/codex_review.md:609:experiments/exp119/monitor.md:214:  4. 这说明 `CSRD` 的作用更像是把 common-support pairwise 几何蒸进 backbone/global，而不是直接替代 fusion 或 test-time correction
experiments/exp367_single_support/codex_train_design.md:2542:experiments/exp367_single_support/codex_review.md:610:experiments/exp119/monitor.md:215:  5. 同时它也暴露了当前版本的瓶颈：teacher 仍来自单图 `kp_feats`，还不够 support-complete
experiments/exp367_single_support/codex_train_design.md:2543:experiments/exp367_single_support/codex_review.md:611:experiments/exp119/monitor.md:218:  - 下一步最合理的单变量不是扫 `CSRD` 权重/温度，而是把 `exp109` 的 support-complete teacher headroom 引回 `CSRD`，做更强的 relational teacher
experiments/exp367_single_support/codex_train_design.md:2544:experiments/exp367_single_support/codex_review.md:618:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:75:[C] P(tail-probe NN is a HEAD of support s) by support bin (DESCRIPTIVE, n=4 bins):
experiments/exp367_single_support/codex_train_design.md:2545:experiments/exp367_single_support/codex_review.md:619:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:81:  [per-IMAGE rate FLAT across support -> purely mechanical count; RISING -> heads over-attract disproportionately.]
experiments/exp367_single_support/codex_train_design.md:2546:experiments/exp367_single_support/codex_review.md:620:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:82:  binned Spearman(support, rate/headID)=+0.4000 (trivially >0)  rate/IMAGE=-0.4000  [n=4 bins, descriptive only]
experiments/exp367_single_support/codex_train_design.md:2547:experiments/exp367_single_support/codex_review.md:621:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:85:     Spearman(support, attraction-count)     = +0.0590  [trivially >0: more imgs = more NN tickets]
experiments/exp367_single_support/codex_train_design.md:2548:experiments/exp367_single_support/codex_review.md:622:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:86:     Spearman(support, attraction-PER-IMAGE) = -0.0093  [NON-TRIVIAL: >0 means heads over-attract beyond count; ~0 means purely count]
experiments/exp367_single_support/codex_train_design.md:2549:experiments/exp367_single_support/codex_review.md:623:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:88:[C] support-calibrated vs global threshold (OVERALL tail->head false-merge at matched head-recall), CAL=even seeds / EVAL=odd seeds:
experiments/exp367_single_support/codex_train_design.md:2550:experiments/exp367_single_support/codex_review.md:624:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:89:  head-recall=0.90: OVERALL false-merge  global=0.2490  support-calibrated=0.2481  (d=-0.0010; want NEGATIVE)  [5 eval seeds, 1040 tail probes]
experiments/exp367_single_support/codex_train_design.md:2551:experiments/exp367_single_support/codex_review.md:625:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:90:  head-recall=0.95: OVERALL false-merge  global=0.5654  support-calibrated=0.5519  (d=-0.0135; want NEGATIVE)  [5 eval seeds, 1040 tail probes]
experiments/exp367_single_support/codex_train_design.md:2552:experiments/exp367_single_support/codex_review.md:626:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:91:  support-level fallback-to-global fraction (sparse levels) = 0.143  [high -> 'support-calibrated' is mostly global]
experiments/exp367_single_support/codex_train_design.md:2553:experiments/exp367_single_support/codex_review.md:628:experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:99:[C] per-head-ID Spearman(support, attraction)=+0.059 (trivial)  PER-IMAGE=-0.009 (non-trivial claim)  [n=311 IDs]
experiments/exp367_single_support/codex_train_design.md:2554:experiments/exp367_single_support/codex_review.md:630:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:33:  Fragility gate : fuse only when lowtail-positive-support is weak (fragile) vs fuse-all.
experiments/exp367_single_support/codex_train_design.md:2555:experiments/exp367_single_support/codex_review.md:631:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:61:ap.add_argument('--a_temp', type=float, default=20.0, help='soft-min temp for lowtail positive support')
experiments/exp367_single_support/codex_train_design.md:2556:experiments/exp367_single_support/codex_review.md:632:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:65:                help='fragility gate: among failures, bottom-q lowtail-support = "fragile" (fuse only these)')
experiments/exp367_single_support/codex_train_design.md:2557:experiments/exp367_single_support/codex_review.md:633:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:303:    # lowtail positive support of each single frame (cross-VIDEO positives in gallery only;
experiments/exp367_single_support/codex_train_design.md:2558:experiments/exp367_single_support/codex_review.md:634:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:305:    # support meter reflects deployable cross-camera identity evidence, not same-video repeats.
experiments/exp367_single_support/codex_train_design.md:2559:experiments/exp367_single_support/codex_review.md:635:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:431:    # Fragility gate: fuse only FRAGILE (weak lowtail support) failures vs fuse-all.
experiments/exp367_single_support/codex_train_design.md:2560:experiments/exp367_single_support/codex_review.md:636:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:432:    # Among the used failures, split by lowtail support (computed earlier per single_rows row).
experiments/exp367_single_support/codex_train_design.md:2561:experiments/exp367_single_support/codex_review.md:637:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:435:    print(f"[gate] fragility-weighted (fuse only weak-support failures) vs fuse-all")
experiments/exp367_single_support/codex_train_design.md:2562:experiments/exp367_single_support/codex_review.md:638:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:440:        fragile = okg & (lt_used <= thr)               # weakest support = most fragile
experiments/exp367_single_support/codex_train_design.md:2563:experiments/exp367_single_support/codex_review.md:639:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:446:        print(f"     fuse-FRAGILE-only (bottom-{cli.frag_quant:.0%} support) dAP={dA_frag:+.3f} "
experiments/exp367_single_support/codex_train_design.md:2564:experiments/exp367_single_support/codex_review.md:640:experiments/cargo_cvpb/cvpb_realizability_killswitch.py:451:        print(f"     [too few finite-support rows to gate: {int(okg.sum())}]")
experiments/exp367_single_support/codex_train_design.md:2565:experiments/exp367_single_support/codex_review.md:648:experiments/cargo_cvpb/cvpb_lattice_killswitch.py:77:ap.add_argument('--adaptive_k', action='store_true', help='supporting: per-query phase-volatility selects K (high-vol query marginalize over K, low-vol use K=1). Reduces avg compute keeping most marginalization gain -> rebut "K=9 too expensive".')
experiments/exp367_single_support/codex_train_design.md:2566:experiments/exp367_single_support/codex_review.md:650:experiments/cargo_cvpb/cvpb_lattice_killswitch.py:548:        # ---- adaptive-K (supporting): per-query phase volatility -> spend K only where it helps ----
experiments/exp367_single_support/codex_train_design.md:2567:experiments/exp367_single_support/codex_review.md:651:experiments/cargo_cvpb/cvpb_lattice_killswitch.py:606:        nfalse = n_false_in_topk(d_single, q_pid, q_cam, g_pid, g_cam, k=10)
experiments/exp367_single_support/codex_train_design.md:2568:experiments/exp367_single_support/codex_review.md:657:experiments/cargo_cvpb/cvpb_lattice_killswitch.py:615:        # also control per-image LR severity (single-LR->HR drift) jointly with #false,
experiments/exp367_single_support/codex_train_design.md:2569:experiments/exp367_single_support/codex_review.md:667:experiments/cargo_cvpb/hub_verify_p0_p4.py:20:  P0c stronger cheap controls: partial corr of M(q) | {#false-in-topk, topk-precision,
experiments/exp367_single_support/codex_train_design.md:2570:experiments/exp367_single_support/codex_review.md:680:experiments/cargo_cvpb/litreview/reviews/lit_4.md:926:date of current version 7 May 2025. This work was supported in part by the
experiments/exp367_single_support/codex_train_design.md:2571:experiments/exp367_single_support/codex_review.md:682:experiments/cargo_cvpb/litreview/reviews/lit_4.md:2727:2026. This work was supported in part by the Natural Science Foundation
experiments/exp367_single_support/codex_train_design.md:2572:experiments/exp367_single_support/codex_review.md:683:experiments/cargo_cvpb/litreview/reviews/lit_4.md:3305:This work is supported by the University of Macau Start-up Research Grant SRG2024-00002-FST and Multi-Year Research Grant MYRGGRG2024-00077-FST-UMDF
experiments/exp367_single_support/codex_train_design.md:2573:experiments/exp367_single_support/codex_review.md:684:experiments/exp360_intruder/design.md:9:| T-SCD（tracklet support 蒸馏） | 5.0/10 ❌ | 撞项目自己的 `fgeu_realizability_result.md`（posetrack tracklet 每条≤2帧、同机位冗余，只恢复 oracle 16.3% < 40% 门槛）+ MVI²P/UMTS/VKD 先例 |
experiments/exp367_single_support/codex_train_design.md:2574:experiments/exp367_single_support/codex_review.md:685:experiments/exp360_intruder/design.md:14:遮挡 ReID 的根症结，**不是**"target 信息缺失要补全"（completion / support-complete 这条线已反复证负：exp109 墙、fgeu 16.3%、各种 feature completion 小残差），**而是**：
experiments/exp367_single_support/codex_train_design.md:2575:experiments/exp367_single_support/codex_review.md:686:experiments/clip_reid_compare/CLIP-ReID/datasets/make_dataloader_clipreid.py:96:        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
experiments/exp367_single_support/codex_train_design.md:2576:experiments/exp367_single_support/codex_review.md:688:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1016:This work was supported in part by the National Science Foundation Program of China (NSFC) (grant number: 61976241),
experiments/exp367_single_support/codex_train_design.md:2577:experiments/exp367_single_support/codex_review.md:689:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1170:2025. This work was supported in part by the National Natural Science
experiments/exp367_single_support/codex_train_design.md:2578:experiments/exp367_single_support/codex_review.md:690:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1511:(MML). SML constructs diverse query and support sets in each training cycle, allowing the model to learn
experiments/exp367_single_support/codex_train_design.md:2579:experiments/exp367_single_support/codex_review.md:691:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1562:adopting a meta-learning perspective lies in the way query-support
experiments/exp367_single_support/codex_train_design.md:2580:experiments/exp367_single_support/codex_review.md:692:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1565:To fully leverage the advantages of the query-support paradigm in
experiments/exp367_single_support/codex_train_design.md:2581:experiments/exp367_single_support/codex_review.md:693:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1567:utility by dynamically constructing diverse query-support pairs in
experiments/exp367_single_support/codex_train_design.md:2582:experiments/exp367_single_support/codex_review.md:694:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1574:diverse query and support sets during training, enabling the model to
experiments/exp367_single_support/codex_train_design.md:2583:experiments/exp367_single_support/codex_review.md:695:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1578:support sets among the samples of each identity. Specifically, in each
experiments/exp367_single_support/codex_train_design.md:2584:experiments/exp367_single_support/codex_review.md:696:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1580:samples form the support set. This ensures that every sample is utilized
experiments/exp367_single_support/codex_train_design.md:2585:experiments/exp367_single_support/codex_review.md:697:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1582:to systematically explore all possible combinations of query-support
experiments/exp367_single_support/codex_train_design.md:2586:experiments/exp367_single_support/codex_review.md:698:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1587:and support sets. Concurrently, the MML method captures long-term
experiments/exp367_single_support/codex_train_design.md:2587:experiments/exp367_single_support/codex_review.md:699:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1712:subtask, SML constructs diverse query and support sets during each
experiments/exp367_single_support/codex_train_design.md:2588:experiments/exp367_single_support/codex_review.md:700:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1743:metric space for VIPR by introducing a shuffling strategy that dynamically constructs query-support pairs while leveraging memory banks
experiments/exp367_single_support/codex_train_design.md:2589:experiments/exp367_single_support/codex_review.md:701:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1787:learned to learn from a given support set to minimize loss over a batch
experiments/exp367_single_support/codex_train_design.md:2590:experiments/exp367_single_support/codex_review.md:702:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1843:This work was supported by the National Nature Science Foundation of China (No. 62376201). This research was financially
experiments/exp367_single_support/codex_train_design.md:2591:experiments/exp367_single_support/codex_review.md:703:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1844:supported by funds from Key Laboratory of Social Computing and Cognitive Intelligence (Dalian University of Technology),
experiments/exp367_single_support/codex_train_design.md:2592:experiments/exp367_single_support/codex_review.md:704:experiments/cargo_cvpb/litreview/reviews/lit_12.md:1992:The work is partially supported by Shanghai Artificial Intelligence Innovation and Development Fund (No. 2020-RGZN02026).
experiments/exp367_single_support/codex_train_design.md:2593:experiments/exp367_single_support/codex_review.md:705:experiments/cargo_cvpb/litreview/reviews/lit_12.md:3365:能发原因：VI-ReID 的普通 triplet 或 center loss 只看 batch 内关系，数据少时学不到全局度量。作者把训练改成 query-support 检索小任务，并用 memory bank 引入历史特征。证据链是 sample-based、center-based、meta-based 三类对比，加 RegDB/SYSU 实验。  
experiments/exp367_single_support/codex_train_design.md:2594:experiments/exp367_single_support/codex_review.md:706:experiments/cargo_cvpb/litreview/reviews/lit_12.md:3442:能发原因：VI-ReID 的普通 triplet 或 center loss 只看 batch 内关系，数据少时学不到全局度量。作者把训练改成 query-support 检索小任务，并用 memory bank 引入历史特征。证据链是 sample-based、center-based、meta-based 三类对比，加 RegDB/SYSU 实验。  
experiments/exp367_single_support/codex_train_design.md:2595:experiments/exp367_single_support/codex_review.md:707:experiments/cargo_cvpb/litreview/reviews/lit_7.md:687:supported in part by the National Natural Science Foundation of China under
experiments/exp367_single_support/codex_train_design.md:2596:experiments/exp367_single_support/codex_review.md:708:experiments/cargo_cvpb/litreview/reviews/lit_7.md:1169:This work is supported by the National Natural Science Foundation of
experiments/exp367_single_support/codex_train_design.md:2597:experiments/exp367_single_support/codex_review.md:709:experiments/cargo_cvpb/litreview/reviews/lit_7.md:1908:This work is supported by the National Natural Science Foundation
experiments/exp367_single_support/codex_train_design.md:2598:experiments/exp367_single_support/codex_review.md:710:experiments/cargo_cvpb/litreview/reviews/lit_7.md:2398:⋆ This work is supported by the National Natural Science Foundation of China (Grant No. 62272430).
experiments/exp367_single_support/codex_train_design.md:2599:experiments/exp367_single_support/codex_review.md:711:experiments/cargo_cvpb/litreview/reviews/lit_7.md:2494:This work is supported by the National Natural Science Foundation of China (No. 62276221, No. 62376232, No. 62466003);
experiments/exp367_single_support/codex_train_design.md:2600:experiments/exp367_single_support/codex_review.md:712:experiments/cargo_cvpb/litreview/reviews/lit_7.md:2811:ISE (Zhang et al., 2022) employs a progressive linear interpolation strategy to create support samples from real samples and adjacent clusters in
experiments/exp367_single_support/codex_train_design.md:2601:experiments/exp367_single_support/codex_review.md:713:experiments/cargo_cvpb/litreview/reviews/lit_7.md:3170:cheap kill-switch：拿已有有标签数据，模拟无监督聚类后统计 false split 和 false merge 是否被 pose visibility mismatch 显著解释。如果可见性指标不能预测聚类错误，或者简单几何重加权不能提升伪标签纯度，就不继续。
experiments/exp367_single_support/codex_train_design.md:2602:experiments/exp367_single_support/codex_review.md:714:experiments/cargo_cvpb/litreview/reviews/lit_7.md:3248:cheap kill-switch：拿已有有标签数据，模拟无监督聚类后统计 false split 和 false merge 是否被 pose visibility mismatch 显著解释。如果可见性指标不能预测聚类错误，或者简单几何重加权不能提升伪标签纯度，就不继续。
experiments/exp367_single_support/codex_train_design.md:2603:experiments/exp367_single_support/codex_review.md:715:experiments/exp121/design.md:5:- `exp120` 正在验证：support-complete teacher 是否能把 `exp109` 的 headroom 接到 `exp119` 的 relational distillation 上
experiments/exp367_single_support/codex_train_design.md:2604:experiments/exp367_single_support/codex_review.md:716:experiments/exp121/design.md:8:  **support-complete relational teacher 是否也需要稳定化**
experiments/exp367_single_support/codex_train_design.md:2605:experiments/exp367_single_support/codex_review.md:717:experiments/exp121/design.md:16:1. 如果 `exp120` 的主要风险也来自 online teacher non-stationarity，那么在 support-complete bank 已初步成熟后冻结更新，可能会比持续在线更新更稳
experiments/exp367_single_support/codex_train_design.md:2606:experiments/exp367_single_support/codex_review.md:718:experiments/exp121/design.md:20:3. 若 freeze30 优于 exp120，说明后续主方法应把“support-complete”与“stable teacher”一起写
experiments/exp367_single_support/codex_train_design.md:2607:experiments/exp367_single_support/codex_review.md:719:experiments/exp121/design.md:35:- 本地主实验：`exp120 SCRD`（online support-complete teacher）
experiments/exp367_single_support/codex_train_design.md:2608:experiments/exp367_single_support/codex_review.md:720:experiments/exp121/design.md:41:  - 说明 support-complete teacher 也存在 non-stationary / hardening 问题
experiments/exp367_single_support/codex_train_design.md:2609:experiments/exp367_single_support/codex_review.md:721:experiments/exp121/design.md:42:  - 下一步应继续沿“stable support-complete relational teacher”写主方法
experiments/exp367_single_support/codex_train_design.md:2610:experiments/exp367_single_support/codex_review.md:722:experiments/exp121/design.md:50:3. 冻结过早可能让 bank 还没积累够 support，导致 teacher 反而变弱
experiments/exp367_single_support/codex_train_design.md:2611:experiments/exp367_single_support/codex_review.md:723:experiments/exp121/monitor.md:17:  2. 当前最有信息量的远程并行对照，不是重复一份 `exp120`，而是测试 support-complete teacher 的稳定化
experiments/exp367_single_support/codex_train_design.md:2612:experiments/exp367_single_support/codex_review.md:724:experiments/exp121/monitor.md:25:  - 需要形成“本地 online support-complete teacher + 远程 freeze30 support-complete teacher”的并行对照
experiments/exp367_single_support/codex_train_design.md:2613:experiments/exp367_single_support/codex_review.md:725:experiments/exp121/monitor.md:94:  - `exp121` 仍处于有信息量阶段；如果后续继续保持对 `exp120` 的领先，就能更明确支持 “support-complete relational teacher 也需要稳定化”
experiments/exp367_single_support/codex_train_design.md:2614:experiments/exp367_single_support/codex_review.md:726:experiments/exp121/monitor.md:106:  3. 因而 “support-complete relational teacher 也需要稳定化” 这条判断仍成立，但幅度还不够大，暂时不足以单独构成主方法
experiments/exp367_single_support/codex_train_design.md:2615:experiments/exp367_single_support/codex_review.md:727:experiments/exp121/monitor.md:124:  - `exp121` 仍值得跑到更后期完成对照，但从当前形态看，它更像 supporting mechanism，而不是单独的主创新突破口
experiments/exp367_single_support/codex_train_design.md:2616:experiments/exp367_single_support/codex_review.md:728:experiments/exp121/monitor.md:134:  - 这条线更像 supporting evidence：support-complete relational teacher 确实受稳定性影响，但单靠 freeze 还不足以构成论文主方法
experiments/exp367_single_support/codex_train_design.md:2617:experiments/exp367_single_support/codex_review.md:729:experiments/exp121/monitor.md:147:     - 对 `SCRD online teacher` 的稳定化 supporting evidence
experiments/exp367_single_support/codex_train_design.md:2618:experiments/exp367_single_support/codex_review.md:730:experiments/exp121/monitor.md:151:  - 目前最合理的用途是把 `freeze30` 作为 supporting mechanism 收尾，而不是再围绕它单独扩展新分支
experiments/exp367_single_support/codex_train_design.md:2619:experiments/exp367_single_support/codex_review.md:731:experiments/exp121/monitor.md:165:  4. 但这个量级仍更像 supporting mechanism，而不是足以单独支撑论文主创新的核心版本
experiments/exp367_single_support/codex_train_design.md:2620:experiments/exp367_single_support/codex_review.md:732:experiments/exp121/monitor.md:168:  - 这条线现在最有价值的角色是作为“teacher stability 有帮助”的最终 supporting evidence
experiments/exp367_single_support/codex_train_design.md:2621:experiments/exp367_single_support/codex_review.md:733:experiments/exp121/monitor.md:180:  3. 这说明 `stable teacher` 不是伪命题，它确实能把 `support-complete relational teacher` 从中性偏负拉回到弱正向
experiments/exp367_single_support/codex_train_design.md:2622:experiments/exp367_single_support/codex_review.md:734:experiments/exp121/monitor.md:181:  4. 但这个量级仍明显更像 supporting mechanism，而不是足以单独撑起论文主创新的主方法
experiments/exp367_single_support/codex_train_design.md:2623:experiments/exp367_single_support/codex_review.md:735:experiments/exp121/monitor.md:182:- 当前判断: 实验完成，作为 supporting evidence 收口
experiments/exp367_single_support/codex_train_design.md:2624:experiments/exp367_single_support/codex_review.md:736:experiments/exp149/design.md:5:`exp142` 之后，cross-image support completion 基本已经做尽。  
experiments/exp367_single_support/codex_train_design.md:2625:experiments/exp367_single_support/codex_review.md:737:experiments/exp149/design.md:21:1. 单图 support incomplete 不只体现在“缺少更多图”，也体现在“没有利用好同一张图里的 homologous evidence”
experiments/exp367_single_support/codex_train_design.md:2626:experiments/exp367_single_support/codex_review.md:738:experiments/cargo_cvpb/litreview/reviews/lit_19.md:767:October 16, 2024, January 31, 2025. This work was supported by the National
experiments/exp367_single_support/codex_train_design.md:2627:experiments/exp367_single_support/codex_review.md:739:experiments/cargo_cvpb/litreview/reviews/lit_19.md:1119:This work is supported by the National Key R&D Program of China (Grant No. 2023YFC3321600) and funds of South Central
experiments/exp367_single_support/codex_train_design.md:2628:experiments/exp367_single_support/codex_review.md:740:experiments/cargo_cvpb/litreview/reviews/lit_19.md:1898:of current version 7 May 2025. This work was supported in part by the
experiments/exp367_single_support/codex_train_design.md:2629:experiments/exp367_single_support/codex_review.md:741:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2215:This work was supported by the National Natural Science Foundation of
experiments/exp367_single_support/codex_train_design.md:2630:experiments/exp367_single_support/codex_review.md:742:experiments/cargo_cvpb/litreview2/reviews/deep_8.md:361:This work was supported by the National Natural Science Foundation of China (No. 62302080), Guangxi Key Research and
experiments/exp367_single_support/codex_train_design.md:2631:experiments/exp367_single_support/codex_review.md:743:experiments/cargo_cvpb/litreview2/reviews/deep_8.md:2249:This work was supported in part by the National Natural
experiments/exp367_single_support/codex_train_design.md:2632:experiments/exp367_single_support/codex_review.md:744:experiments/cargo_cvpb/litreview2/reviews/deep_8.md:5076:This work is supported by the National Natural Science Foundation of China (Grants Nos. 62202061 and 62171043), the
experiments/exp367_single_support/codex_train_design.md:2633:experiments/exp367_single_support/codex_review.md:746:experiments/cargo_cvpb/litreview2/reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
experiments/exp367_single_support/codex_train_design.md:2634:experiments/exp367_single_support/codex_review.md:747:experiments/cargo_cvpb/litreview2/reviews/deep_8.md:11079:This work was supported by National Natural Science Foundation of China (62102003), Anhui Postdoctoral Science Foundation
experiments/exp367_single_support/codex_train_design.md:2635:experiments/exp367_single_support/codex_review.md:748:experiments/cargo_cvpb/litreview2/reviews/deep_8.md:12018:This work is supported by the National Natural Science Foundation of China (No. 62466003, No. 62276221, No. 62376232), the
experiments/exp367_single_support/codex_train_design.md:2636:experiments/exp367_single_support/codex_review.md:749:experiments/cargo_cvpb/litreview2/reviews/deep_8.md:12023:resources and support.
experiments/exp367_single_support/codex_train_design.md:2637:experiments/exp367_single_support/codex_review.md:752:experiments/cargo_cvpb/cvpb_lattice_killswitch_DESIGN.md:18:2. **vs #false-in-topk**（Hubness §7.6 教训）：phase-var 解释失败必须在 partial out 这个 trivial 计数后仍 >0。另加 partial out LR severity（single-LR→HR drift）。Hubness 的 M(q) 正是没控这个代理被判死。
experiments/exp367_single_support/codex_train_design.md:2638:experiments/exp367_single_support/codex_review.md:756:experiments/cargo_cvpb/litreview/reviews/lit_16.md:671:2026. This work was supported in part by Guangdong Science and Technology
experiments/exp367_single_support/codex_train_design.md:2639:experiments/exp367_single_support/codex_review.md:757:experiments/cargo_cvpb/litreview/reviews/lit_16.md:1348:+ This work was supported in part by the National Natural Science
experiments/exp367_single_support/codex_train_design.md:2640:experiments/exp367_single_support/codex_review.md:758:experiments/cargo_cvpb/litreview/reviews/lit_16.md:1864:This work was supported by the National Natural Science Foundation of China (Nos. 62272461, 62172417, 62276266, and
experiments/exp367_single_support/codex_train_design.md:2641:experiments/exp367_single_support/codex_review.md:761:experiments/cargo_cvpb/litreview/reviews/lit_16.md:2022:This work was supported by the National Natural Science Foundation of
experiments/exp367_single_support/codex_train_design.md:2642:experiments/exp367_single_support/codex_review.md:762:experiments/cargo_cvpb/litreview/reviews/lit_16.md:2496:Prior works, however, considered the problems of continuously updating models and decentralized training models separately. They are still unable to support distributed edge clients
experiments/exp367_single_support/codex_train_design.md:2643:experiments/exp367_single_support/codex_review.md:763:experiments/cargo_cvpb/litreview/reviews/lit_16.md:2787:current version 5 February 2026. This work was supported in part by the
experiments/exp367_single_support/codex_train_design.md:2644:experiments/exp367_single_support/codex_review.md:767:experiments/cargo_cvpb/litreview/reviews/lit_11.md:1759:2026. This work was supported in part by the National Natural Science
experiments/exp367_single_support/codex_train_design.md:2645:experiments/exp367_single_support/codex_review.md:770:experiments/cargo_cvpb/litreview/reviews/lit_11.md:2488:This work was supported in part by the NSFC Key Project of International (Regional) Cooperation and Exchanges under Grant 61860206004, and in part by
experiments/exp367_single_support/codex_train_design.md:2646:experiments/exp367_single_support/codex_review.md:771:experiments/cargo_cvpb/litreview/reviews/lit_11.md:2809:This research was supported by Bourns Endowment funds.
experiments/exp367_single_support/codex_train_design.md:2647:experiments/exp367_single_support/codex_review.md:773:experiments/cargo_cvpb/litreview/reviews/lit_11.md:3142:version 6 October 2025. This work was supported in part by the Research
experiments/exp367_single_support/codex_train_design.md:2648:experiments/exp367_single_support/codex_review.md:774:experiments/cargo_cvpb/litreview/reviews/lit_11.md:3755:This work was partially supported by the National Natural Science Foundation of China under Grant No. 62301315, Startup
experiments/exp367_single_support/codex_train_design.md:2649:experiments/exp367_single_support/codex_review.md:775:experiments/cargo_cvpb/litreview/reviews/lit_11.md:3941:This work is supported by the Guangdong Basic and Applied Basic Research Foundation (No.2025A1515011465), the National Natural Science Foundation of
experiments/exp367_single_support/codex_train_design.md:2650:experiments/exp367_single_support/codex_review.md:779:experiments/cargo_cvpb/litreview/reviews/lit_8.md:914:This study is supported in part by the Key Technologies Research and Development Program (grant no. 2024YFF0617200),
experiments/exp367_single_support/codex_train_design.md:2651:experiments/exp367_single_support/codex_review.md:780:experiments/cargo_cvpb/litreview/reviews/lit_8.md:951:network architectures. Concurrently, researchers have progressively developed expanded benchmark datasets with growing image volumes to support methodological innovations in this domain
experiments/exp367_single_support/codex_train_design.md:2652:experiments/exp367_single_support/codex_review.md:781:experiments/cargo_cvpb/litreview/reviews/lit_8.md:975:in processing power and memory capacity, making it difficult to support complex and largescale models [21]. Additionally, many IoT applications require local data processing at the edge
experiments/exp367_single_support/codex_train_design.md:2653:experiments/exp367_single_support/codex_review.md:782:experiments/cargo_cvpb/litreview/reviews/lit_8.md:1325:This work was supported in part by the National Natural Science Foundation
experiments/exp367_single_support/codex_train_design.md:2654:experiments/exp367_single_support/codex_review.md:783:experiments/cargo_cvpb/litreview/reviews/lit_8.md:2054:This work is supported in part by the Natural Science Foundation of
experiments/exp367_single_support/codex_train_design.md:2655:experiments/exp367_single_support/codex_review.md:831:experiments/cargo_cvpb/litreview/reviews/lit_8.md:2926:13 February 2025. This work was supported in part by the National High
experiments/exp367_single_support/codex_train_design.md:2656:experiments/exp367_single_support/codex_review.md:832:experiments/cargo_cvpb/litreview/reviews/lit_8.md:3060:published to support this task, such as PRID-2011 [12],
experiments/exp367_single_support/codex_train_design.md:2657:experiments/exp367_single_support/codex_review.md:833:experiments/cargo_cvpb/litreview/reviews/lit_8.md:3347:version 31 October 2025. This work was supported in part by the National
experiments/exp367_single_support/codex_train_design.md:2658:experiments/exp367_single_support/codex_review.md:834:experiments/exp117/design.md:5:SCKD 系列（exp110-116, 7 个变体）已经证明：EMA prototype bank 的增量上限仅 ~+0.1% mAP。这说明从外部（memory bank）向模型注入 support-complete 信号行不通。
experiments/exp367_single_support/codex_train_design.md:2659:experiments/exp367_single_support/codex_review.md:836:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:1874:This work was supported in part by the Natural Science Foundation
experiments/exp367_single_support/codex_train_design.md:2660:experiments/exp367_single_support/codex_review.md:837:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:2089:This work was supported by the Natural Science Foundation (NSF) of
experiments/exp367_single_support/codex_train_design.md:2661:experiments/exp367_single_support/codex_review.md:838:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:2101:Compared to UDA, USL is more challenging to train directly on unlabeled data due to the lack of pretraining support
experiments/exp367_single_support/codex_train_design.md:2662:experiments/exp367_single_support/codex_review.md:839:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:2926:I This research was supported by National Natural Science Foundation of China (Grant Nos. 62376089, U23A20318, 62302154, 62472149), and Young and
experiments/exp367_single_support/codex_train_design.md:2663:experiments/exp367_single_support/codex_review.md:840:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:4093:This work is supported in part by the Natural Science Foundation of
experiments/exp367_single_support/codex_train_design.md:2664:experiments/exp367_single_support/codex_review.md:843:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:8087:retrieval accuracy. This result fully demonstrates the retrieval superiority of MCCAN and strongly supports the feasibility of its practical
experiments/exp367_single_support/codex_train_design.md:2665:experiments/exp367_single_support/codex_review.md:844:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:8354:This study is partially supported by the National Key R&D Program of China (No. 2022YFC3803600), the National Natural Science
experiments/exp367_single_support/codex_train_design.md:2666:experiments/exp367_single_support/codex_review.md:845:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:8356:Key Laboratory of Software Development Environment (No. SKLSDE2023ZX-11). This research was supported by the Research Start-up
experiments/exp367_single_support/codex_train_design.md:2667:experiments/exp367_single_support/codex_review.md:846:experiments/cargo_cvpb/litreview2/reviews/deep_19.md:8357:Funds of Hangzhou International Innovation Institute of Beihang University under Grant No. 2024KQ012. Thank you for the support from
experiments/exp367_single_support/codex_train_design.md:2668:experiments/exp367_single_support/codex_review.md:850:experiments/cargo_cvpb/litreview/reviews/lit_17.md:825:This work was supported in part by the Research Project of ZJULeague Research and Development Center, Zhejiang Laboratory under Grant
experiments/exp367_single_support/codex_train_design.md:2669:experiments/exp367_single_support/codex_review.md:851:experiments/cargo_cvpb/litreview/reviews/lit_17.md:1330:thereby supporting persistent object segmentation. In the domain of motion tracking, [38] develop a salient event blob detector that identiﬁes regions with consistent optical ﬂow through a novel Field of Active Flow Directions (FAFD) representation constructed from the Surface
experiments/exp367_single_support/codex_train_design.md:2670:experiments/exp367_single_support/codex_review.md:852:experiments/cargo_cvpb/litreview/reviews/lit_17.md:1890:at enabling retrieval at any time moment and across different time intervals. We contribute for the first time a largescale dataset named AT-USTC to support the study of ATReID. Compared to existing datasets, AT-USTC stands out
experiments/exp367_single_support/codex_train_design.md:2671:experiments/exp367_single_support/codex_review.md:853:experiments/cargo_cvpb/litreview/reviews/lit_17.md:2043:4 June 2025. This work was supported in part by the National Natural
experiments/exp367_single_support/codex_train_design.md:2672:experiments/exp367_single_support/codex_review.md:854:experiments/cargo_cvpb/litreview/reviews/lit_17.md:2327:support for tracking suspects and ﬁnding lost people. Due to the impact
experiments/exp367_single_support/codex_train_design.md:2673:experiments/exp367_single_support/codex_review.md:855:experiments/cargo_cvpb/litreview/reviews/lit_17.md:2741:that text lacks stable local structural support in feature space. These
experiments/exp367_single_support/codex_train_design.md:2674:experiments/exp367_single_support/codex_review.md:857:experiments/exp128/design.md:33:- support-complete teacher 其余配置不变
experiments/exp367_single_support/codex_train_design.md:2675:experiments/exp367_single_support/codex_review.md:858:experiments/exp128/design.md:39:3. supporting 对照: `exp121`
experiments/exp367_single_support/codex_train_design.md:2676:experiments/exp367_single_support/codex_review.md:860:experiments/exp128/design.md:54:   **exact sparse routing + stable support-complete teacher**
experiments/exp367_single_support/codex_train_design.md:2677:experiments/exp367_single_support/codex_review.md:862:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:16:[1] raw Spearman(positive-support risk, tax) over 476 valid core queries:
experiments/exp367_single_support/codex_train_design.md:2678:experiments/exp367_single_support/codex_review.md:864:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:23:[1] ★PARTIAL Spearman(support-risk, tax | 1x-top1-margin + #false-in-topk):
experiments/exp367_single_support/codex_train_design.md:2679:experiments/exp367_single_support/codex_review.md:865:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:27:     [reverse] 1x-margin | support  = +0.4244   #false | support = -0.1277
experiments/exp367_single_support/codex_train_design.md:2680:experiments/exp367_single_support/codex_review.md:866:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:29:[1] big-tax (top-30% tax) OOF-AUC: trivials=0.7989  +support=0.8416  support-solo=0.7642  >> INCREMENT=+0.0427
experiments/exp367_single_support/codex_train_design.md:2681:experiments/exp367_single_support/codex_review.md:868:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:45:     trivials + support          OOF-AUC = 0.9852
experiments/exp367_single_support/codex_train_design.md:2682:experiments/exp367_single_support/codex_review.md:869:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:46:     support-only (3 proxies)    OOF-AUC = 0.8641
experiments/exp367_single_support/codex_train_design.md:2683:experiments/exp367_single_support/codex_review.md:870:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:47:     >> INCREMENT support adds on top of trivials = +0.0236
experiments/exp367_single_support/codex_train_design.md:2684:experiments/exp367_single_support/codex_review.md:871:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:48:     >> best support AUC - best trivial AUC        = -0.1181
experiments/exp367_single_support/codex_train_design.md:2685:experiments/exp367_single_support/codex_review.md:872:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:54:[3] full mAP=73.055  failures(bot-30%)=663  low-support failures(bot-30%)=199
experiments/exp367_single_support/codex_train_design.md:2686:experiments/exp367_single_support/codex_review.md:873:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:56:[3] oracle multi-query on 198 low-support failure queries (mean AP, %):
experiments/exp367_single_support/codex_train_design.md:2687:experiments/exp367_single_support/codex_review.md:874:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:63:[3] k-reciprocal (full-set re-rank, k1=20 k2=6 lam=0.3) then index the SAME 198 low-support failure queries:
experiments/exp367_single_support/codex_train_design.md:2688:experiments/exp367_single_support/codex_review.md:877:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:73:[T2] failure-AUC: best support-trivial gap=-0.118  OOF incr=+0.024  support-solo AUC=0.864  partial(best-supp,-AP|triv)=+0.747
experiments/exp367_single_support/codex_train_design.md:2689:experiments/exp367_single_support/codex_review.md:878:experiments/cargo_cvpb/rr_logs/cvpb_evidence_oduke.log:74:[T3] oracle on n=198 low-support failures: base=17.16 -> same-ID-union=52.47 (d=+35.31)  random-ID=4.61 (d=-12.56)  k-recip=27.88
experiments/exp367_single_support/codex_train_design.md:2690:experiments/exp367_single_support/codex_review.md:879:experiments/exp128/monitor.md:18:  3. `freeze30` 是已验证的 supporting mechanism，因此值得接到 exact-topk 稀疏版本上做 full-model 候选验证
experiments/exp367_single_support/codex_train_design.md:2691:experiments/exp367_single_support/codex_review.md:880:experiments/exp128/monitor.md:24:  2. `freeze20 / freeze30` 的既有证据已经足够说明它只是弱 supporting mechanism，不值得继续消耗本地 3090
experiments/exp367_single_support/codex_train_design.md:2692:experiments/exp367_single_support/codex_review.md:884:experiments/cargo_cvpb/litreview/reviews/lit_15.md:940:2026. This work was supported in part by the National Natural Science
experiments/exp367_single_support/codex_train_design.md:2693:experiments/exp367_single_support/codex_review.md:885:experiments/cargo_cvpb/litreview/reviews/lit_15.md:1058:to support finer-grained retrieval tasks. With the advent of
experiments/exp367_single_support/codex_train_design.md:2694:experiments/exp367_single_support/codex_review.md:886:experiments/cargo_cvpb/litreview/reviews/lit_15.md:2016:This work was supported by the Ministry of Education of Singapore under
experiments/exp367_single_support/codex_train_design.md:2695:experiments/exp367_single_support/codex_review.md:887:experiments/cargo_cvpb/litreview/reviews/lit_15.md:2407:Manuscript received May 17, 2025. This work was supported in part by
experiments/exp367_single_support/codex_train_design.md:2696:experiments/exp367_single_support/codex_review.md:888:experiments/cargo_cvpb/litreview/reviews/lit_15.md:3358:2025. This work was supported in part by the National Key Research
experiments/exp367_single_support/codex_train_design.md:2697:experiments/exp367_single_support/codex_review.md:889:experiments/cargo_cvpb/litreview/reviews/lit_15.md:4157:shared feature extractor simultaneously supports image restoration and
experiments/exp367_single_support/codex_train_design.md:2698:experiments/exp367_single_support/codex_review.md:890:experiments/exp324h/design.md:24:- **oracle 数学逐行复用 exp324g**：`topk_excluded`（top-10 Jaccard）、`per_query_ap`
experiments/exp367_single_support/codex_train_design.md:2699:experiments/exp367_single_support/codex_review.md:892:experiments/cargo_cvpb/litreview/reviews/lit_9.md:640:capture clear images under those poor illumination conditions. Moreover, most cameras in modern surveillance systems support autoswitch between the visible and infrared
experiments/exp367_single_support/codex_train_design.md:2700:experiments/exp367_single_support/codex_review.md:893:experiments/cargo_cvpb/litreview/reviews/lit_9.md:1440:2022; date of current version 1 March 2025. This work was supported by
experiments/exp367_single_support/codex_train_design.md:2701:experiments/exp367_single_support/codex_review.md:894:experiments/cargo_cvpb/litreview/reviews/lit_9.md:2385:and 13 January 2026; accepted 17 April 2026. This work was supported in part
experiments/exp367_single_support/codex_train_design.md:2702:experiments/exp367_single_support/codex_review.md:895:experiments/cargo_cvpb/litreview/reviews/lit_9.md:2575:paradigm for segmentation by supporting diverse prompt
experiments/exp367_single_support/codex_train_design.md:2703:experiments/exp367_single_support/codex_review.md:896:experiments/cargo_cvpb/litreview/reviews/lit_9.md:2713:2025. This work was supported in part by the National Natural Science
experiments/exp367_single_support/codex_train_design.md:2704:experiments/exp367_single_support/codex_review.md:897:experiments/exp128/claude_review.md:48:**Partially support stop exp127 + start exp128, with reservations.**
experiments/exp367_single_support/codex_train_design.md:2705:experiments/exp367_single_support/codex_review.md:902:experiments/cargo_cvpb/codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:2706:experiments/exp367_single_support/codex_review.md:903:experiments/cargo_cvpb/codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2707:experiments/exp367_single_support/codex_review.md:904:experiments/cargo_cvpb/codex_review_ovli.txt:711:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/exp367_single_support/codex_train_design.md:2708:experiments/exp367_single_support/codex_review.md:905:experiments/cargo_cvpb/codex_review_ovli.txt:778:   - 写法：test-time supporting evaluations
experiments/exp367_single_support/codex_train_design.md:2709:experiments/exp367_single_support/codex_review.md:906:experiments/cargo_cvpb/codex_review_ovli.txt:945:exp358 disambiguate: **打乱 17 关键点通道(per-image)→ 破坏解剖部位身份, 但保留同图自己的空间 support**(关键点位置是本图的, 只是哪个点属于哪个部位被打乱)。无裁剪对齐 rescue。
experiments/exp367_single_support/codex_train_design.md:2710:experiments/exp367_single_support/codex_review.md:913:experiments/cargo_cvpb/codex_review_ovli.txt:3374:experiments/decisions.md:3493:3. `MaxSim / POT / flip` 主要仍是 test-time supporting evidence，不能作为训练端主贡献
experiments/exp367_single_support/codex_train_design.md:2711:experiments/exp367_single_support/codex_review.md:914:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:2712:experiments/exp367_single_support/codex_review.md:915:experiments/cargo_cvpb/codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2713:experiments/exp367_single_support/codex_review.md:916:experiments/cargo_cvpb/codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2714:experiments/exp367_single_support/codex_review.md:917:experiments/cargo_cvpb/codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:2715:experiments/exp367_single_support/codex_review.md:918:experiments/cargo_cvpb/codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2716:experiments/exp367_single_support/codex_review.md:919:experiments/cargo_cvpb/codex_review_ovli.txt:3972:experiments/overnight_innovation_log.md:209:3. **新问题框定**：CLAUDE.md 已列对方向——**common-visible support / pair comparability / reliability-aware matching**。把"互见部位 MaxSim"形式化成新匹配目标（理论+消融），不是当 scoring trick。
experiments/exp367_single_support/codex_train_design.md:2717:experiments/exp367_single_support/codex_review.md:920:experiments/cargo_cvpb/codex_review_ovli.txt:3975:experiments/overnight_innovation_log.md:268:- **新颖性裁决 + plateau 双确认**：novelty agent 的"路线2=打平/超SOTA"对单分支 pose-part-MaxSim 已**实质不可达**（~48 all-query vs 需 ≥62）。剩可走路线只有 **(1) 机制重组 LoRA↔visibility** 或 **(3) 问题 reframe（common-visible support / reliability-aware matching，CLAUDE.md 钦定方向）**。
experiments/exp367_single_support/codex_train_design.md:2718:experiments/exp367_single_support/codex_review.md:922:experiments/cargo_cvpb/codex_review_ovli.txt:4112:experiments/paper_notes/2026-04-15_prcv_reset.md:10:4. `GCN` 虽然也属于 pose 信息利用，但应统一写成 **structural pose branch**；`LGPA-D / OA-SD / MaxSim / POT / flip-test` 仍作为 supporting assets，不再抢主创新位置
experiments/exp367_single_support/codex_train_design.md:2719:experiments/exp367_single_support/codex_review.md:923:experiments/cargo_cvpb/codex_review_ovli.txt:4115:experiments/paper_notes/2026-04-15_prcv_reset.md:201:- `LGPA-D / GCN / OA-SD / MaxSim` = system assets / supporting modules
experiments/exp367_single_support/codex_train_design.md:2720:experiments/exp367_single_support/codex_review.md:924:experiments/cargo_cvpb/codex_review_ovli.txt:4128:experiments/paper_materials/story.md:20:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/exp367_single_support/codex_train_design.md:2721:experiments/exp367_single_support/codex_review.md:929:experiments/exp110/design.md:5:- `exp109` 的 oracle support bank 诊断给出极强 headroom：
experiments/exp367_single_support/codex_train_design.md:2722:experiments/exp367_single_support/codex_review.md:930:experiments/exp110/design.md:9:  **单图关键点表征缺少 support-complete identity 信息**
experiments/exp367_single_support/codex_train_design.md:2723:experiments/exp367_single_support/codex_review.md:931:experiments/exp110/design.md:11:  - `CIPGFR / LSRM / TTSFR` 依赖 batch 内 same-ID support，太弱
experiments/exp367_single_support/codex_train_design.md:2724:experiments/exp367_single_support/codex_review.md:932:experiments/exp110/design.md:15:用 identity-level prototype bank 把 low-visibility keypoints 蒸馏向更完整的 support teacher。
experiments/exp367_single_support/codex_train_design.md:2725:experiments/exp367_single_support/codex_review.md:933:experiments/exp110/design.md:19:1. 若 batch-local recovery 失败的原因真是 support 来源太弱，那么换成持久的 `per-ID / per-keypoint prototype bank` 后，训练端应该更稳定。
experiments/exp367_single_support/codex_train_design.md:2726:experiments/exp367_single_support/codex_review.md:934:experiments/exp110/design.md:59:  - support-complete 方向需重写为更强 teacher / recovered pooling，而不是继续在当前最小版小修小补
experiments/exp367_single_support/codex_train_design.md:2727:experiments/exp367_single_support/codex_review.md:935:experiments/exp360_intruder/codex_h2fail_decision.md:14:exp360 Intruder Identity Suppression 阶段0 frozen probe 结果, 帮判方向(调研/决策交你, 别捧场)。机制: 遮挡 ReID 重定义为 donor 行人身份污染 target embedding, 训练端对抗压 donor-ID, 测试单图。**阶段0 frozen 强 baseline(market) 合成 target+donor**: H1 donor-ID linear probe acc 73% vs chance 2% = **36.5x PASS(泄漏巨大确凿)**; person leak 0.15 >> rand patch -0.01 PASS; **但 H2 FAIL: leak=cos(f_mix,f_donor)-cos(f_clean,f_donor) vs per-query AP drop, raw spearman +0.120, 控 #false-in-topk(top-k混错ID平凡代理)后 partial=-0.028 约0**。AP drop 大(clean 0.835→mix 0.409)。诚实含义: donor 泄漏存在且大, 但泄漏量不独立于 #false 预测检索损害。问: (a)H2 FAIL 是否**致命否定** Intruder 因果链(压泄漏→救检索)? 还是 H2(frozen per-query 相关)≠H3(训练干预), donor suppression 训练仍值得试? (b)若值得试, 阶段1怎么设计让'压 donor-ID'真涨不退化成 target ambiguity 墙(KPR已占)? (c)若致命, 3候选回选下一个: B PSC-JEPA(从SOLIDER continued-pretrain pseudo-support-bank latent JEPA) / C-#3 support-set continued pretrain / 还是另起? 联网查 identity disentangle/feature purification/source separation occluded ReID 里'泄漏可测但压它未必涨'的先例教训(OGFR DPEFormer 等)。务实中文, 每选项信心1-10。
experiments/exp367_single_support/codex_train_design.md:2728:experiments/exp367_single_support/codex_review.md:936:experiments/exp360_intruder/codex_h2fail_decision.md:261:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:2729:experiments/exp367_single_support/codex_review.md:937:experiments/exp360_intruder/codex_h2fail_decision.md:298:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2730:experiments/exp367_single_support/codex_review.md:943:experiments/exp360_intruder/codex_h2fail_decision.md:787:experiments/exp360_intruder/codex_h2fail_decision.md:14:exp360 Intruder Identity Suppression 阶段0 frozen probe 结果, 帮判方向(调研/决策交你, 别捧场)。机制: 遮挡 ReID 重定义为 donor 行人身份污染 target embedding, 训练端对抗压 donor-ID, 测试单图。**阶段0 frozen 强 baseline(market) 合成 target+donor**: H1 donor-ID linear probe acc 73% vs chance 2% = **36.5x PASS(泄漏巨大确凿)**; person leak 0.15 >> rand patch -0.01 PASS; **但 H2 FAIL: leak=cos(f_mix,f_donor)-cos(f_clean,f_donor) vs per-query AP drop, raw spearman +0.120, 控 #false-in-topk(top-k混错ID平凡代理)后 partial=-0.028 约0**。AP drop 大(clean 0.835→mix 0.409)。诚实含义: donor 泄漏存在且大, 但泄漏量不独立于 #false 预测检索损害。问: (a)H2 FAIL 是否**致命否定** Intruder 因果链(压泄漏→救检索)? 还是 H2(frozen per-query 相关)≠H3(训练干预), donor suppression 训练仍值得试? (b)若值得试, 阶段1怎么设计让'压 donor-ID'真涨不退化成 target ambiguity 墙(KPR已占)? (c)若致命, 3候选回选下一个: B PSC-JEPA(从SOLIDER continued-pretrain pseudo-support-bank latent JEPA) / C-#3 support-set continued pretrain / 还是另起? 联网查 identity disentangle/feature purification/source separation occluded ReID 里'泄漏可测但压它未必涨'的先例教训(OGFR DPEFormer 等)。务实中文, 每选项信心1-10。
experiments/exp367_single_support/codex_train_design.md:2731:experiments/exp367_single_support/codex_review.md:944:experiments/exp360_intruder/codex_h2fail_decision.md:794:experiments/decisions.md:1803:   **当前性能缺口里有一大块确实来自“support 不完整”，而不是 confuser suppression 失败。**
experiments/exp367_single_support/codex_train_design.md:2732:experiments/exp367_single_support/codex_review.md:947:experiments/exp360_intruder/codex_h2fail_decision.md:903:experiments/cargo_cvpb/litreview2/explore20.sh:6:CTX="一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。"
experiments/exp367_single_support/codex_train_design.md:2733:experiments/exp367_single_support/codex_review.md:948:experiments/exp360_intruder/codex_h2fail_decision.md:972:experiments/cargo_cvpb/litreview2/lattice_method_design.md:14:一个 ReID 团队的 **Lattice-Marginalized ReID**(重定义: 低分辨率 ReID = 采样格点不确定性, 非'模糊缺细节')零训练 kill-switch = GO。证据: frozen 模型, K=9 phase/bbox/kernel variants 的 ensemble 在 h=16 LR query 上 +4.23 mAP, **比同 K 的普通随机 TTA 多 +3.04**(lattice-specific 非 TTA 换名: lat-MaxSim 46.9>tta 43.8>single 42.6), 74.9% LR query 跨 lattice 变体翻转 rank-1 身份(h升→31%→10%), 过 vs-TTA + vs-#false 双控。**诚实: phase-var 作 per-query 失败预测器与 LR-severity 共线(correlational 弱), GO 靠 interventional ensemble 结果(直接测量)。**
experiments/exp367_single_support/codex_train_design.md:2734:experiments/exp367_single_support/codex_review.md:1026:experiments/exp360_intruder/codex_h2fail_decision.md:1127:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:7569:experiments/cargo_cvpb/litreview2/train_more_import.md:1886:experiments/cargo_cvpb/litreview2/lattice_method_design.md:14:一个 ReID 团队的 **Lattice-Marginalized ReID**(重定义: 低分辨率 ReID = 采样格点不确定性, 非'模糊缺细节')零训练 kill-switch = GO。证据: frozen 模型, K=9 phase/bbox/kernel variants 的 ensemble 在 h=16 LR query 上 +4.23 mAP, **比同 K 的普通随机 TTA 多 +3.04**(lattice-specific 非 TTA 换名: lat-MaxSim 46.9>tta 43.8>single 42.6), 74.9% LR query 跨 lattice 变体翻转 rank-1 身份(h升→31%→10%), 过 vs-TTA + vs-#false 双控。**诚实: phase-var 作 per-query 失败预测器与 LR-severity 共线(correlational 弱), GO 靠 interventional ensemble 结果(直接测量)。**
experiments/exp367_single_support/codex_train_design.md:2735:experiments/exp367_single_support/codex_review.md:1027:experiments/exp360_intruder/codex_h2fail_decision.md:1132:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8085:experiments/cargo_cvpb/litreview2/lattice_method_design.md:14:一个 ReID 团队的 **Lattice-Marginalized ReID**(重定义: 低分辨率 ReID = 采样格点不确定性, 非'模糊缺细节')零训练 kill-switch = GO。证据: frozen 模型, K=9 phase/bbox/kernel variants 的 ensemble 在 h=16 LR query 上 +4.23 mAP, **比同 K 的普通随机 TTA 多 +3.04**(lattice-specific 非 TTA 换名: lat-MaxSim 46.9>tta 43.8>single 42.6), 74.9% LR query 跨 lattice 变体翻转 rank-1 身份(h升→31%→10%), 过 vs-TTA + vs-#false 双控。**诚实: phase-var 作 per-query 失败预测器与 LR-severity 共线(correlational 弱), GO 靠 interventional ensemble 结果(直接测量)。**
experiments/exp367_single_support/codex_train_design.md:2736:experiments/exp367_single_support/codex_review.md:1030:experiments/exp360_intruder/codex_h2fail_decision.md:1609:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2737:experiments/exp367_single_support/codex_review.md:1031:experiments/exp360_intruder/codex_h2fail_decision.md:1628:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2738:experiments/exp367_single_support/codex_review.md:1043:experiments/exp360_intruder/codex_h2fail_decision.md:1844:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2739:experiments/exp367_single_support/codex_review.md:1044:experiments/exp360_intruder/codex_h2fail_decision.md:1868:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2740:experiments/exp367_single_support/codex_review.md:1045:experiments/exp360_intruder/codex_h2fail_decision.md:1990:experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3982:../../cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2741:experiments/exp367_single_support/codex_review.md:1048:experiments/exp360_intruder/codex_h2fail_decision.md:2169:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15257:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5763:./experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2742:experiments/exp367_single_support/codex_review.md:1049:experiments/exp360_intruder/codex_h2fail_decision.md:2182:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15270:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5782:./experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2743:experiments/exp367_single_support/codex_review.md:1050:experiments/exp360_intruder/codex_h2fail_decision.md:2204:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15297:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5872:./experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2744:experiments/exp367_single_support/codex_review.md:1051:experiments/exp360_intruder/codex_h2fail_decision.md:2217:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15310:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5888:./experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2745:experiments/exp367_single_support/codex_review.md:1052:experiments/exp360_intruder/codex_h2fail_decision.md:2242:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15338:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5938:./experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3982:../../cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2746:experiments/exp367_single_support/codex_review.md:1064:experiments/exp360_intruder/codex_h2fail_decision.md:2267:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15373:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6169:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17863:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:953:./reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2747:experiments/exp367_single_support/codex_review.md:1065:experiments/exp360_intruder/codex_h2fail_decision.md:2268:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15375:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6171:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17873:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:3957:./reassess/r_2.md:5969:reassess/r_3.md:8894:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2748:experiments/exp367_single_support/codex_review.md:1066:experiments/exp360_intruder/codex_h2fail_decision.md:2269:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15376:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6172:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17915:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4001:./reassess/r_2.md:6196:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2749:experiments/exp367_single_support/codex_review.md:1067:experiments/exp360_intruder/codex_h2fail_decision.md:2270:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15377:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6173:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4020:./reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2750:experiments/exp367_single_support/codex_review.md:1068:experiments/exp360_intruder/codex_h2fail_decision.md:2271:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15378:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6174:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4046:./reassess/r_3.md:1954:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2751:experiments/exp367_single_support/codex_review.md:1069:experiments/exp360_intruder/codex_h2fail_decision.md:2272:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15379:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6175:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17926:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4254:./reassess/r_3.md:3334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2752:experiments/exp367_single_support/codex_review.md:1070:experiments/exp360_intruder/codex_h2fail_decision.md:2273:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15380:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6176:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17943:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4479:./reassess/r_3.md:6129:./reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2753:experiments/exp367_single_support/codex_review.md:1071:experiments/exp360_intruder/codex_h2fail_decision.md:2274:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15381:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6177:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4598:./reassess/r_3.md:7723:./reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2754:experiments/exp367_single_support/codex_review.md:1072:experiments/exp360_intruder/codex_h2fail_decision.md:2275:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15382:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6178:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17991:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4600:./reassess/r_3.md:7734:./reassess/r_3.md:1954:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2755:experiments/exp367_single_support/codex_review.md:1073:experiments/exp360_intruder/codex_h2fail_decision.md:2276:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15383:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6179:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:17996:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4634:./reassess/r_3.md:7897:./reassess/r_3.md:3334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2756:experiments/exp367_single_support/codex_review.md:1074:experiments/exp360_intruder/codex_h2fail_decision.md:2277:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15385:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6181:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:18026:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4734:./reassess/r_3.md:8476:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2757:experiments/exp367_single_support/codex_review.md:1075:experiments/exp360_intruder/codex_h2fail_decision.md:2278:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15387:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6183:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:18031:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4770:./reassess/r_3.md:8894:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2758:experiments/exp367_single_support/codex_review.md:1090:experiments/exp360_intruder/codex_h2fail_decision.md:2328:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15444:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6406:./experiments/cargo_cvpb/litreview2/reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2759:experiments/exp367_single_support/codex_review.md:1091:experiments/exp360_intruder/codex_h2fail_decision.md:2349:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15471:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6470:./experiments/cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2760:experiments/exp367_single_support/codex_review.md:1092:experiments/exp360_intruder/codex_h2fail_decision.md:2402:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15528:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6636:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2761:experiments/exp367_single_support/codex_review.md:1093:experiments/exp360_intruder/codex_h2fail_decision.md:2421:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15547:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6660:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2762:experiments/exp367_single_support/codex_review.md:1094:experiments/exp360_intruder/codex_h2fail_decision.md:2509:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15641:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6955:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:14:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2763:experiments/exp367_single_support/codex_review.md:1095:experiments/exp360_intruder/codex_h2fail_decision.md:2521:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15653:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6992:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:2929:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2764:experiments/exp367_single_support/codex_review.md:1096:experiments/exp360_intruder/codex_h2fail_decision.md:2534:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15666:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7011:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:2973:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2765:experiments/exp367_single_support/codex_review.md:1102:experiments/exp360_intruder/codex_h2fail_decision.md:2553:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15685:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7051:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3824:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17863:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:953:./reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2766:experiments/exp367_single_support/codex_review.md:1103:experiments/exp360_intruder/codex_h2fail_decision.md:2554:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15686:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7052:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3828:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17873:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:3957:./reassess/r_2.md:5969:reassess/r_3.md:8894:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2767:experiments/exp367_single_support/codex_review.md:1104:experiments/exp360_intruder/codex_h2fail_decision.md:2555:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15687:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7053:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3859:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17915:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4001:./reassess/r_2.md:6196:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2768:experiments/exp367_single_support/codex_review.md:1105:experiments/exp360_intruder/codex_h2fail_decision.md:2556:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15688:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7054:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3860:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4020:./reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2769:experiments/exp367_single_support/codex_review.md:1106:experiments/exp360_intruder/codex_h2fail_decision.md:2557:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15689:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7055:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3862:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4046:./reassess/r_3.md:1954:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2770:experiments/exp367_single_support/codex_review.md:1107:experiments/exp360_intruder/codex_h2fail_decision.md:2558:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15690:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7056:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3863:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17926:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4254:./reassess/r_3.md:3334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2771:experiments/exp367_single_support/codex_review.md:1108:experiments/exp360_intruder/codex_h2fail_decision.md:2559:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15691:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7057:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3867:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17943:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4479:./reassess/r_3.md:6129:./reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2772:experiments/exp367_single_support/codex_review.md:1109:experiments/exp360_intruder/codex_h2fail_decision.md:2560:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15692:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7058:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3879:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4598:./reassess/r_3.md:7723:./reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2773:experiments/exp367_single_support/codex_review.md:1110:experiments/exp360_intruder/codex_h2fail_decision.md:2561:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15693:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7059:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3881:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17991:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4600:./reassess/r_3.md:7734:./reassess/r_3.md:1954:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2774:experiments/exp367_single_support/codex_review.md:1111:experiments/exp360_intruder/codex_h2fail_decision.md:2562:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15694:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7060:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3882:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17996:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4634:./reassess/r_3.md:7897:./reassess/r_3.md:3334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2775:experiments/exp367_single_support/codex_review.md:1112:experiments/exp360_intruder/codex_h2fail_decision.md:2563:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15695:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7061:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3890:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18026:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4734:./reassess/r_3.md:8476:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2776:experiments/exp367_single_support/codex_review.md:1113:experiments/exp360_intruder/codex_h2fail_decision.md:2564:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15696:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7062:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3893:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18031:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4770:./reassess/r_3.md:8894:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2777:experiments/exp367_single_support/codex_review.md:1124:experiments/exp360_intruder/codex_h2fail_decision.md:2606:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15742:experiments/paradigm_shift/decision_tscd_vs_intruder.md:11555:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2778:experiments/exp367_single_support/codex_review.md:1146:experiments/exp360_intruder/codex_h2fail_decision.md:2750:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15909:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2779:experiments/exp367_single_support/codex_review.md:1147:experiments/exp360_intruder/codex_h2fail_decision.md:2763:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15924:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2780:experiments/exp367_single_support/codex_review.md:1148:experiments/exp360_intruder/codex_h2fail_decision.md:2789:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15955:experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3982:../../cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2781:experiments/exp367_single_support/codex_review.md:1149:experiments/exp360_intruder/codex_h2fail_decision.md:2805:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15981:experiments/cargo_cvpb/litreview2/reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2782:experiments/exp367_single_support/codex_review.md:1150:experiments/exp360_intruder/codex_h2fail_decision.md:2826:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16012:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2783:experiments/exp367_single_support/codex_review.md:1153:experiments/exp360_intruder/codex_h2fail_decision.md:2864:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16063:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2784:experiments/exp367_single_support/codex_review.md:1154:experiments/exp360_intruder/codex_h2fail_decision.md:2877:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16078:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2785:experiments/exp367_single_support/codex_review.md:1155:experiments/exp360_intruder/codex_h2fail_decision.md:2928:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16139:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2786:experiments/exp367_single_support/codex_review.md:1156:experiments/exp360_intruder/codex_h2fail_decision.md:2947:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16162:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2787:experiments/exp367_single_support/codex_review.md:1166:experiments/exp360_intruder/codex_h2fail_decision.md:2961:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16183:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17863:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:953:./reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2788:experiments/exp367_single_support/codex_review.md:1167:experiments/exp360_intruder/codex_h2fail_decision.md:2962:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16185:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17873:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:3957:./reassess/r_2.md:5969:reassess/r_3.md:8894:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2789:experiments/exp367_single_support/codex_review.md:1168:experiments/exp360_intruder/codex_h2fail_decision.md:2963:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16186:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17915:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4001:./reassess/r_2.md:6196:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2790:experiments/exp367_single_support/codex_review.md:1169:experiments/exp360_intruder/codex_h2fail_decision.md:2964:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16187:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4020:./reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2791:experiments/exp367_single_support/codex_review.md:1170:experiments/exp360_intruder/codex_h2fail_decision.md:2965:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16189:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4046:./reassess/r_3.md:1954:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2792:experiments/exp367_single_support/codex_review.md:1171:experiments/exp360_intruder/codex_h2fail_decision.md:2966:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16190:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17926:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4254:./reassess/r_3.md:3334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2793:experiments/exp367_single_support/codex_review.md:1172:experiments/exp360_intruder/codex_h2fail_decision.md:2967:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16192:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17943:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4479:./reassess/r_3.md:6129:./reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2794:experiments/exp367_single_support/codex_review.md:1173:experiments/exp360_intruder/codex_h2fail_decision.md:2968:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16193:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4598:./reassess/r_3.md:7723:./reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2795:experiments/exp367_single_support/codex_review.md:1174:experiments/exp360_intruder/codex_h2fail_decision.md:2969:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16195:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17991:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4600:./reassess/r_3.md:7734:./reassess/r_3.md:1954:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess.sh:19:ROLES[3]="角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。"
experiments/exp367_single_support/codex_train_design.md:2796:experiments/exp367_single_support/codex_review.md:1175:experiments/exp360_intruder/codex_h2fail_decision.md:2970:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16196:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17996:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4634:./reassess/r_3.md:7897:./reassess/r_3.md:3334:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:23:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2797:experiments/exp367_single_support/codex_review.md:1176:experiments/exp360_intruder/codex_h2fail_decision.md:2971:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16199:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18026:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4734:./reassess/r_3.md:8476:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2798:experiments/exp367_single_support/codex_review.md:1177:experiments/exp360_intruder/codex_h2fail_decision.md:2972:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16201:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18031:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4770:./reassess/r_3.md:8894:角色=**遮挡最后一搏**。p_3提的遮挡source-separation(occlusion=别人身份污染target embedding, donor-ID泄漏)值不值得测? 但必须面对历史'occluder-gate是墙'(exp109: 即使gate掉遮挡, headroom是identity-conditioned不可实现)。设计**最锋利最便宜的kill-switch, 且必须同时回答'压制donor-ID泄漏能不能真涨ReID'**(不只是证泄漏存在——泄漏存在但压了不涨=撞墙=没用)。联网查novelty(撞TTPM/non-target-pedestrian-occlusion那批没)。如果连这都救不动或必撞墙, 诚实说'遮挡这块彻底关'。
experiments/exp367_single_support/codex_train_design.md:2799:experiments/exp367_single_support/codex_review.md:1188:experiments/exp360_intruder/codex_h2fail_decision.md:3014:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16288:Intruder 的优势不是它一定涨点，而是它把问题从“缺失 support 要补全”改成“外来身份源污染 target embedding 要分离”。这避开了 completion / visibility / occluder-gate 死区。
experiments/exp367_single_support/codex_train_design.md:2800:experiments/exp367_single_support/codex_review.md:1189:experiments/exp360_intruder/codex_h2fail_decision.md:3029:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16349:Intruder 的优势不是它一定涨点，而是它把问题从“缺失 support 要补全”改成“外来身份源污染 target embedding 要分离”。这避开了 completion / visibility / occluder-gate 死区。
experiments/exp367_single_support/codex_train_design.md:2801:experiments/exp367_single_support/codex_review.md:1190:experiments/exp360_intruder/codex_h2fail_decision.md:3056:| T-SCD（tracklet support 蒸馏） | 5.0/10 ❌ | 撞项目自己的 `fgeu_realizability_result.md`（posetrack tracklet 每条≤2帧、同机位冗余，只恢复 oracle 16.3% < 40% 门槛）+ MVI²P/UMTS/VKD 先例 |
experiments/exp367_single_support/codex_train_design.md:2802:experiments/exp367_single_support/codex_review.md:1191:experiments/exp360_intruder/codex_h2fail_decision.md:3061:遮挡 ReID 的根症结，**不是**"target 信息缺失要补全"（completion / support-complete 这条线已反复证负：exp109 墙、fgeu 16.3%、各种 feature completion 小残差），**而是**：
experiments/exp367_single_support/codex_train_design.md:2803:experiments/exp367_single_support/codex_review.md:1195:experiments/exp360_intruder/codex_h2fail_decision.md:3127:- codex 评估中（`codex_h2fail_decision.md`，PID 91395）：H2 FAIL 致命否 + push H3 训练 vs 回 candidate（B PSC-JEPA continued-pretrain / C-#3 support-set）重选。
experiments/exp367_single_support/codex_train_design.md:2804:experiments/exp367_single_support/codex_review.md:1196:experiments/exp360_intruder/codex_h2fail_decision.md:3145:exp360 Intruder Identity Suppression 阶段0 frozen probe 结果, 帮判方向(调研/决策交你, 别捧场)。机制: 遮挡 ReID 重定义为 donor 行人身份污染 target embedding, 训练端对抗压 donor-ID, 测试单图。**阶段0 frozen 强 baseline(market) 合成 target+donor**: H1 donor-ID linear probe acc 73% vs chance 2% = **36.5x PASS(泄漏巨大确凿)**; person leak 0.15 >> rand patch -0.01 PASS; **但 H2 FAIL: leak=cos(f_mix,f_donor)-cos(f_clean,f_donor) vs per-query AP drop, raw spearman +0.120, 控 #false-in-topk(top-k混错ID平凡代理)后 partial=-0.028 约0**。AP drop 大(clean 0.835→mix 0.409)。诚实含义: donor 泄漏存在且大, 但泄漏量不独立于 #false 预测检索损害。问: (a)H2 FAIL 是否**致命否定** Intruder 因果链(压泄漏→救检索)? 还是 H2(frozen per-query 相关)≠H3(训练干预), donor suppression 训练仍值得试? (b)若值得试, 阶段1怎么设计让'压 donor-ID'真涨不退化成 target ambiguity 墙(KPR已占)? (c)若致命, 3候选回选下一个: B PSC-JEPA(从SOLIDER continued-pretrain pseudo-support-bank latent JEPA) / C-#3 support-set continued pretrain / 还是另起? 联网查 identity disentangle/feature purification/source separation occluded ReID 里'泄漏可测但压它未必涨'的先例教训(OGFR DPEFormer 等)。务实中文, 每选项信心1-10。
experiments/exp367_single_support/codex_train_design.md:2805:experiments/exp367_single_support/codex_review.md:1197:experiments/exp360_intruder/codex_h2fail_decision.md:3354:/bin/zsh -lc 'rg -n "PSC-JEPA|support-set|continued pretrain|pseudo-support|JEPA|exp35[0-9]|exp360|Intruder|OGFR|DPEFormer|KPR|QPM|target ambiguity|frozen kill-switch|kill-switch" experiments/paradigm_shift experiments/cargo_cvpb experiments/exp35* experiments/exp360_intruder -S' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2806:experiments/exp367_single_support/codex_review.md:1198:experiments/exp360_intruder/codex_h2fail_decision.md:3457:experiments/exp357_pose_shuffle_ks/design.md:44:- 下一步: cross-PART(通道)shuffle exp358 二次确认——打乱17关键点通道(破坏解剖部位身份, 保留同图空间 support)。若 exp358 也只小掉→解剖身份也不重要, 只是"某种空间池化结构"在涨→故事进一步塌; 若 exp358 大掉→解剖部位结构重要。
experiments/exp367_single_support/codex_train_design.md:2807:experiments/exp367_single_support/codex_review.md:1209:experiments/exp360_intruder/codex_h2fail_decision.md:3627:experiments/paradigm_shift/paradigm_B_pretraining.md:2091:experiments/exp357_pose_shuffle_ks/design.md:38:- Medium-2(判读): NO-DROP 侧被裁剪对齐混淆(别人 pose 仍带粗糙 canonical 头/躯干/腿先验)。Codex/Claude 一致: 掉点=干净铁证(图特定 pose correspondence 重要); 不掉=只能说"精确图特定 pose 在对齐裁剪下非必需", 需补 **cross-PART(17关键点通道)shuffle** 二次确认(测解剖通道身份是否重要, 同图空间 support 不变)。最佳矩阵: cross-image + per-image channel-shuffle + no-pose/fixed-canonical control。
experiments/exp367_single_support/codex_train_design.md:2808:experiments/exp367_single_support/codex_review.md:1210:experiments/exp360_intruder/codex_h2fail_decision.md:3628:experiments/paradigm_shift/paradigm_B_pretraining.md:2092:experiments/exp357_pose_shuffle_ks/design.md:44:- 下一步: cross-PART(通道)shuffle exp358 二次确认——打乱17关键点通道(破坏解剖部位身份, 保留同图空间 support)。若 exp358 也只小掉→解剖身份也不重要, 只是"某种空间池化结构"在涨→故事进一步塌; 若 exp358 大掉→解剖部位结构重要。
experiments/exp367_single_support/codex_train_design.md:2809:experiments/exp367_single_support/codex_review.md:1211:experiments/exp360_intruder/codex_h2fail_decision.md:3629:experiments/paradigm_shift/paradigm_B_pretraining.md:2116:experiments/decisions.md:1224:1. 近年的强路线把问题定义在 **target ambiguity / common visible support / retrieval-time reasoning**，而不是“再学一个融合权重”。
experiments/exp367_single_support/codex_train_design.md:2810:experiments/exp367_single_support/codex_review.md:1212:experiments/exp360_intruder/codex_h2fail_decision.md:3634:experiments/paradigm_shift/paradigm_B_pretraining.md:2770:experiments/cargo_cvpb/codex_review_ovli.txt:945:exp358 disambiguate: **打乱 17 关键点通道(per-image)→ 破坏解剖部位身份, 但保留同图自己的空间 support**(关键点位置是本图的, 只是哪个点属于哪个部位被打乱)。无裁剪对齐 rescue。
experiments/exp367_single_support/codex_train_design.md:2811:experiments/exp367_single_support/codex_review.md:1214:experiments/exp360_intruder/codex_h2fail_decision.md:3652:experiments/paradigm_shift/paradigm_B_pretraining.md:3503:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2158:./ondisk_pivot.md:4421:./reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/exp367_single_support/codex_train_design.md:2812:experiments/exp367_single_support/codex_review.md:1215:experiments/exp360_intruder/codex_h2fail_decision.md:3661:experiments/paradigm_shift/paradigm_B_pretraining.md:3569:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5576:./reassess/r_2.md:1990:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/exp367_single_support/codex_train_design.md:2813:experiments/exp367_single_support/codex_review.md:1216:experiments/exp360_intruder/codex_h2fail_decision.md:3666:experiments/paradigm_shift/paradigm_B_pretraining.md:3582:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5652:./reassess/r_2.md:4439:reassess/r_2.md:1990:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/exp367_single_support/codex_train_design.md:2814:experiments/exp367_single_support/codex_review.md:1217:experiments/exp360_intruder/codex_h2fail_decision.md:3671:experiments/paradigm_shift/paradigm_B_pretraining.md:3594:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5683:./reassess/r_2.md:5455:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/exp367_single_support/codex_train_design.md:2815:experiments/exp367_single_support/codex_review.md:1218:experiments/exp360_intruder/codex_h2fail_decision.md:3676:experiments/paradigm_shift/paradigm_B_pretraining.md:3694:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5835:./reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/exp367_single_support/codex_train_design.md:2816:experiments/exp367_single_support/codex_review.md:1219:experiments/exp360_intruder/codex_h2fail_decision.md:3920:experiments/paradigm_shift/paradigm_B_pretraining.md:5677:最大风险不是算力，而是 pseudo support bank 噪声。第一 kill-switch 要看：
experiments/exp367_single_support/codex_train_design.md:2817:experiments/exp367_single_support/codex_review.md:1220:experiments/exp360_intruder/codex_h2fail_decision.md:3922:experiments/paradigm_shift/paradigm_B_pretraining.md:5688:- Novelty：`6.5/10`，如果 support bank + latent JEPA 做干净，可到 `7/10`
experiments/exp367_single_support/codex_train_design.md:2818:experiments/exp367_single_support/codex_review.md:1221:experiments/exp360_intruder/codex_h2fail_decision.md:3929:experiments/paradigm_shift/paradigm_B_pretraining.md:5769:最大风险不是算力，而是 pseudo support bank 噪声。第一 kill-switch 要看：
experiments/exp367_single_support/codex_train_design.md:2819:experiments/exp367_single_support/codex_review.md:1222:experiments/exp360_intruder/codex_h2fail_decision.md:3931:experiments/paradigm_shift/paradigm_B_pretraining.md:5780:- Novelty：`6.5/10`，如果 support bank + latent JEPA 做干净，可到 `7/10`
experiments/exp367_single_support/codex_train_design.md:2820:experiments/exp367_single_support/codex_review.md:1223:experiments/exp360_intruder/codex_h2fail_decision.md:3954:experiments/exp360_intruder/codex_h2fail_decision.md:14:exp360 Intruder Identity Suppression 阶段0 frozen probe 结果, 帮判方向(调研/决策交你, 别捧场)。机制: 遮挡 ReID 重定义为 donor 行人身份污染 target embedding, 训练端对抗压 donor-ID, 测试单图。**阶段0 frozen 强 baseline(market) 合成 target+donor**: H1 donor-ID linear probe acc 73% vs chance 2% = **36.5x PASS(泄漏巨大确凿)**; person leak 0.15 >> rand patch -0.01 PASS; **但 H2 FAIL: leak=cos(f_mix,f_donor)-cos(f_clean,f_donor) vs per-query AP drop, raw spearman +0.120, 控 #false-in-topk(top-k混错ID平凡代理)后 partial=-0.028 约0**。AP drop 大(clean 0.835→mix 0.409)。诚实含义: donor 泄漏存在且大, 但泄漏量不独立于 #false 预测检索损害。问: (a)H2 FAIL 是否**致命否定** Intruder 因果链(压泄漏→救检索)? 还是 H2(frozen per-query 相关)≠H3(训练干预), donor suppression 训练仍值得试? (b)若值得试, 阶段1怎么设计让'压 donor-ID'真涨不退化成 target ambiguity 墙(KPR已占)? (c)若致命, 3候选回选下一个: B PSC-JEPA(从SOLIDER continued-pretrain pseudo-support-bank latent JEPA) / C-#3 support-set continued pretrain / 还是另起? 联网查 identity disentangle/feature purification/source separation occluded ReID 里'泄漏可测但压它未必涨'的先例教训(OGFR DPEFormer 等)。务实中文, 每选项信心1-10。
experiments/exp367_single_support/codex_train_design.md:2821:experiments/exp367_single_support/codex_review.md:1224:experiments/exp360_intruder/codex_h2fail_decision.md:3984:experiments/exp360_intruder/codex_h2fail_decision.md:787:experiments/exp360_intruder/codex_h2fail_decision.md:14:exp360 Intruder Identity Suppression 阶段0 frozen probe 结果, 帮判方向(调研/决策交你, 别捧场)。机制: 遮挡 ReID 重定义为 donor 行人身份污染 target embedding, 训练端对抗压 donor-ID, 测试单图。**阶段0 frozen 强 baseline(market) 合成 target+donor**: H1 donor-ID linear probe acc 73% vs chance 2% = **36.5x PASS(泄漏巨大确凿)**; person leak 0.15 >> rand patch -0.01 PASS; **但 H2 FAIL: leak=cos(f_mix,f_donor)-cos(f_clean,f_donor) vs per-query AP drop, raw spearman +0.120, 控 #false-in-topk(top-k混错ID平凡代理)后 partial=-0.028 约0**。AP drop 大(clean 0.835→mix 0.409)。诚实含义: donor 泄漏存在且大, 但泄漏量不独立于 #false 预测检索损害。问: (a)H2 FAIL 是否**致命否定** Intruder 因果链(压泄漏→救检索)? 还是 H2(frozen per-query 相关)≠H3(训练干预), donor suppression 训练仍值得试? (b)若值得试, 阶段1怎么设计让'压 donor-ID'真涨不退化成 target ambiguity 墙(KPR已占)? (c)若致命, 3候选回选下一个: B PSC-JEPA(从SOLIDER continued-pretrain pseudo-support-bank latent JEPA) / C-#3 support-set continued pretrain / 还是另起? 联网查 identity disentangle/feature purification/source separation occluded ReID 里'泄漏可测但压它未必涨'的先例教训(OGFR DPEFormer 等)。务实中文, 每选项信心1-10。
experiments/exp367_single_support/codex_train_design.md:2822:experiments/exp367_single_support/codex_review.md:1225:experiments/exp360_intruder/codex_h2fail_decision.md:3994:experiments/exp360_intruder/codex_h2fail_decision.md:903:experiments/cargo_cvpb/litreview2/explore20.sh:6:CTX="一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。"
experiments/exp367_single_support/codex_train_design.md:2823:experiments/exp367_single_support/codex_review.md:1226:experiments/exp360_intruder/codex_h2fail_decision.md:3998:experiments/exp360_intruder/codex_h2fail_decision.md:972:experiments/cargo_cvpb/litreview2/lattice_method_design.md:14:一个 ReID 团队的 **Lattice-Marginalized ReID**(重定义: 低分辨率 ReID = 采样格点不确定性, 非'模糊缺细节')零训练 kill-switch = GO。证据: frozen 模型, K=9 phase/bbox/kernel variants 的 ensemble 在 h=16 LR query 上 +4.23 mAP, **比同 K 的普通随机 TTA 多 +3.04**(lattice-specific 非 TTA 换名: lat-MaxSim 46.9>tta 43.8>single 42.6), 74.9% LR query 跨 lattice 变体翻转 rank-1 身份(h升→31%→10%), 过 vs-TTA + vs-#false 双控。**诚实: phase-var 作 per-query 失败预测器与 LR-severity 共线(correlational 弱), GO 靠 interventional ensemble 结果(直接测量)。**
experiments/exp367_single_support/codex_train_design.md:2824:experiments/exp367_single_support/codex_review.md:1290:experiments/exp360_intruder/codex_h2fail_decision.md:4070:experiments/exp360_intruder/codex_h2fail_decision.md:1127:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:7569:experiments/cargo_cvpb/litreview2/train_more_import.md:1886:experiments/cargo_cvpb/litreview2/lattice_method_design.md:14:一个 ReID 团队的 **Lattice-Marginalized ReID**(重定义: 低分辨率 ReID = 采样格点不确定性, 非'模糊缺细节')零训练 kill-switch = GO。证据: frozen 模型, K=9 phase/bbox/kernel variants 的 ensemble 在 h=16 LR query 上 +4.23 mAP, **比同 K 的普通随机 TTA 多 +3.04**(lattice-specific 非 TTA 换名: lat-MaxSim 46.9>tta 43.8>single 42.6), 74.9% LR query 跨 lattice 变体翻转 rank-1 身份(h升→31%→10%), 过 vs-TTA + vs-#false 双控。**诚实: phase-var 作 per-query 失败预测器与 LR-severity 共线(correlational 弱), GO 靠 interventional ensemble 结果(直接测量)。**
experiments/exp367_single_support/codex_train_design.md:2825:experiments/exp367_single_support/codex_review.md:1291:experiments/exp360_intruder/codex_h2fail_decision.md:4071:experiments/exp360_intruder/codex_h2fail_decision.md:1132:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8085:experiments/cargo_cvpb/litreview2/lattice_method_design.md:14:一个 ReID 团队的 **Lattice-Marginalized ReID**(重定义: 低分辨率 ReID = 采样格点不确定性, 非'模糊缺细节')零训练 kill-switch = GO。证据: frozen 模型, K=9 phase/bbox/kernel variants 的 ensemble 在 h=16 LR query 上 +4.23 mAP, **比同 K 的普通随机 TTA 多 +3.04**(lattice-specific 非 TTA 换名: lat-MaxSim 46.9>tta 43.8>single 42.6), 74.9% LR query 跨 lattice 变体翻转 rank-1 身份(h升→31%→10%), 过 vs-TTA + vs-#false 双控。**诚实: phase-var 作 per-query 失败预测器与 LR-severity 共线(correlational 弱), GO 靠 interventional ensemble 结果(直接测量)。**
experiments/exp367_single_support/codex_train_design.md:2826:experiments/exp367_single_support/codex_review.md:1292:experiments/exp360_intruder/codex_h2fail_decision.md:4207:experiments/exp360_intruder/codex_h2fail_decision.md:1609:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2827:experiments/exp367_single_support/codex_review.md:1293:experiments/exp360_intruder/codex_h2fail_decision.md:4225:experiments/exp360_intruder/codex_h2fail_decision.md:1628:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2828:experiments/exp367_single_support/codex_review.md:1294:experiments/exp360_intruder/codex_h2fail_decision.md:4284:experiments/exp360_intruder/codex_h2fail_decision.md:1844:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2829:experiments/exp367_single_support/codex_review.md:1295:experiments/exp360_intruder/codex_h2fail_decision.md:4302:experiments/exp360_intruder/codex_h2fail_decision.md:1868:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2830:experiments/exp367_single_support/codex_review.md:1307:experiments/exp360_intruder/codex_h2fail_decision.md:4777:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21695:experiments/exp359_lm_reid/codex_review_raw_v2.md:3871:experiments/exp359_lm_reid/design.md:20:两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。
experiments/exp367_single_support/codex_train_design.md:2831:experiments/exp367_single_support/codex_review.md:1309:experiments/exp360_intruder/codex_h2fail_decision.md:4819:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21737:experiments/exp359_lm_reid/codex_review_raw_v2.md:3913:experiments/exp359_lm_reid/codex_review_raw.md:49:两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。
experiments/exp367_single_support/codex_train_design.md:2832:experiments/exp367_single_support/codex_review.md:1314:experiments/exp360_intruder/codex_h2fail_decision.md:5297:experiments/cargo_cvpb/litreview2/explore20/d_5.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2833:experiments/exp367_single_support/codex_review.md:1315:experiments/exp360_intruder/codex_h2fail_decision.md:5305:experiments/cargo_cvpb/litreview2/explore20/d_18.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2834:experiments/exp367_single_support/codex_review.md:1316:experiments/exp360_intruder/codex_h2fail_decision.md:5313:experiments/cargo_cvpb/litreview2/explore20/d_3.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2835:experiments/exp367_single_support/codex_review.md:1317:experiments/exp360_intruder/codex_h2fail_decision.md:5319:experiments/cargo_cvpb/litreview2/explore20/d_1.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2836:experiments/exp367_single_support/codex_review.md:1320:experiments/exp360_intruder/codex_h2fail_decision.md:5325:experiments/cargo_cvpb/litreview2/explore20/d_9.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2837:experiments/exp367_single_support/codex_review.md:1321:experiments/exp360_intruder/codex_h2fail_decision.md:5333:experiments/cargo_cvpb/litreview2/explore20/d_19.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2838:experiments/exp367_single_support/codex_review.md:1322:experiments/exp360_intruder/codex_h2fail_decision.md:5339:experiments/cargo_cvpb/litreview2/explore20/d_6.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2839:experiments/exp367_single_support/codex_review.md:1323:experiments/exp360_intruder/codex_h2fail_decision.md:5347:experiments/cargo_cvpb/litreview2/explore20/d_7.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2840:experiments/exp367_single_support/codex_review.md:1324:experiments/exp360_intruder/codex_h2fail_decision.md:5356:experiments/cargo_cvpb/litreview2/explore20/d_10.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2841:experiments/exp367_single_support/codex_review.md:1325:experiments/exp360_intruder/codex_h2fail_decision.md:5358:experiments/cargo_cvpb/litreview2/explore20/d_10.md:88:- 零训练 kill-switch：用 frozen SOLIDER/exp030a 特征，按 query-camera 到 gallery-camera 的 train support 分桶，算 pair-conditioned mAP/R1；控制 positive 数量、gallery size、occlusion 后，tail-pair 比 head-pair 仍低 5-8 mAP 才继续。再跑一次 2025 camera-specific feature normalization 后 gap 仍在，才说明不是普通 camera bias。([arxiv.org](https://arxiv.org/abs/2502.10195))
experiments/exp367_single_support/codex_train_design.md:2842:experiments/exp367_single_support/codex_review.md:1326:experiments/exp360_intruder/codex_h2fail_decision.md:5359:experiments/cargo_cvpb/litreview2/explore20/d_10.md:96:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_train_design.md:2843:experiments/exp367_single_support/codex_review.md:1327:experiments/exp360_intruder/codex_h2fail_decision.md:5360:experiments/cargo_cvpb/litreview2/explore20/d_10.md:107:- 零训练 kill-switch：用 frozen SOLIDER/exp030a 特征，按 query-camera 到 gallery-camera 的 train support 分桶，算 pair-conditioned mAP/R1；控制 positive 数量、gallery size、occlusion 后，tail-pair 比 head-pair 仍低 5-8 mAP 才继续。再跑一次 2025 camera-specific feature normalization 后 gap 仍在，才说明不是普通 camera bias。([arxiv.org](https://arxiv.org/abs/2502.10195))
experiments/exp367_single_support/codex_train_design.md:2844:experiments/exp367_single_support/codex_review.md:1328:experiments/exp360_intruder/codex_h2fail_decision.md:5361:experiments/cargo_cvpb/litreview2/explore20/d_10.md:115:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_train_design.md:2845:experiments/exp367_single_support/codex_review.md:1330:experiments/exp360_intruder/codex_h2fail_decision.md:5369:experiments/cargo_cvpb/litreview2/explore20/d_4.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2846:experiments/exp367_single_support/codex_review.md:1333:experiments/exp360_intruder/codex_h2fail_decision.md:5375:experiments/cargo_cvpb/litreview2/explore20/d_15.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2847:experiments/exp367_single_support/codex_review.md:1334:experiments/exp360_intruder/codex_h2fail_decision.md:5381:experiments/cargo_cvpb/litreview2/explore20/d_2.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2848:experiments/exp367_single_support/codex_review.md:1335:experiments/exp360_intruder/codex_h2fail_decision.md:5393:experiments/cargo_cvpb/litreview2/explore20/d_11.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2849:experiments/exp367_single_support/codex_review.md:1336:experiments/exp360_intruder/codex_h2fail_decision.md:5399:experiments/cargo_cvpb/litreview2/explore20/d_20.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2850:experiments/exp367_single_support/codex_review.md:1337:experiments/exp360_intruder/codex_h2fail_decision.md:5424:experiments/cargo_cvpb/litreview2/explore20/d_14.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:2851:experiments/exp367_single_support/codex_review.md:1338:experiments/exp360_intruder/codex_h2fail_decision.md:5504:**fragility gate (只融弱 support 失败 vs 全融):**
experiments/exp367_single_support/codex_train_design.md:2852:experiments/exp367_single_support/codex_review.md:1339:experiments/exp360_intruder/codex_h2fail_decision.md:5506:- fuse-FRAGILE-only (bottom-50% support) dAP = +5.51 (n=45)
experiments/exp367_single_support/codex_train_design.md:2853:experiments/exp367_single_support/codex_review.md:1340:experiments/exp360_intruder/codex_h2fail_decision.md:5560:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2854:experiments/exp367_single_support/codex_review.md:1341:experiments/exp360_intruder/codex_h2fail_decision.md:5583:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:2855:experiments/exp367_single_support/codex_review.md:1342:experiments/exp360_intruder/codex_h2fail_decision.md:5620:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:2856:experiments/exp367_single_support/codex_review.md:1343:experiments/exp360_intruder/codex_h2fail_decision.md:5839:- [✅] 范式 gap analysis：3 路 codex（A 生成 / B 预训练 / C 自由）。三路 + 项目 exp109 oracle 收敛到根问题=**single-image support incomplete**。
experiments/exp367_single_support/codex_train_design.md:2857:experiments/exp367_single_support/codex_review.md:1344:experiments/exp360_intruder/codex_h2fail_decision.md:5851:/bin/zsh -lc 'rg -n "PSC-JEPA|support-set continued|support set continued|pseudo-support|JEPA|#3|continued-pretrain|continued pretrain|support-bank|support bank|candidate B|候选 B|候选.*C" experiments/paradigm_shift experiments/cargo_cvpb experiments -S' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:2858:experiments/exp367_single_support/codex_review.md:1345:experiments/exp360_intruder/codex_h2fail_decision.md:5870:experiments/paradigm_shift/paradigm_C_free.md:996:experiments/exp109/design.md:9:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:2859:experiments/exp367_single_support/codex_review.md:1346:experiments/exp360_intruder/codex_h2fail_decision.md:5871:experiments/paradigm_shift/paradigm_C_free.md:997:experiments/exp109/design.md:10:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp367_single_support/codex_train_design.md:2860:experiments/exp367_single_support/codex_review.md:1347:experiments/exp360_intruder/codex_h2fail_decision.md:5872:experiments/paradigm_shift/paradigm_C_free.md:1000:experiments/exp109/design.md:30:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:2861:experiments/exp367_single_support/codex_review.md:1348:experiments/exp360_intruder/codex_h2fail_decision.md:5873:experiments/paradigm_shift/paradigm_C_free.md:1002:experiments/exp109/design.md:59:  - support bank 训练线应止损
experiments/exp367_single_support/codex_train_design.md:2862:experiments/exp367_single_support/codex_review.md:1349:experiments/exp360_intruder/codex_h2fail_decision.md:5875:experiments/paradigm_shift/paradigm_C_free.md:1312:experiments/decisions.md:3479:- `exp109` oracle support bank 仍是仓库内最强问题证据
experiments/exp367_single_support/codex_train_design.md:2863:experiments/exp367_single_support/codex_review.md:1350:experiments/exp360_intruder/codex_h2fail_decision.md:5876:experiments/paradigm_shift/paradigm_C_free.md:1599:experiments/innovation_brainstorm.md:1622:   **same-ID support bank → single-image support-complete distillation**
experiments/exp367_single_support/codex_train_design.md:2864:experiments/exp367_single_support/codex_review.md:1351:experiments/exp360_intruder/codex_h2fail_decision.md:5877:experiments/paradigm_shift/paradigm_C_free.md:1604:experiments/innovation_brainstorm.md:1651:   **reliable-support bank / teacher reliability gating**
experiments/exp367_single_support/codex_train_design.md:2865:experiments/exp367_single_support/codex_review.md:1352:experiments/exp360_intruder/codex_h2fail_decision.md:5878:experiments/paradigm_shift/paradigm_C_free.md:1614:experiments/innovation_brainstorm.md:1701:- Lagged / stale support bank
experiments/exp367_single_support/codex_train_design.md:2866:experiments/exp367_single_support/codex_review.md:1353:experiments/exp360_intruder/codex_h2fail_decision.md:5879:experiments/paradigm_shift/paradigm_C_free.md:1623:experiments/innovation_brainstorm.md:1849:3. 而是先用 `exp109` 方向的 support bank 补全 low-vis keypoint teacher，再用补全后的 teacher 去做 `CSRD`
experiments/exp367_single_support/codex_train_design.md:2867:experiments/exp367_single_support/codex_review.md:1354:experiments/exp360_intruder/codex_h2fail_decision.md:5880:experiments/paradigm_shift/paradigm_C_free.md:1669:experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:2868:experiments/exp367_single_support/codex_review.md:1355:experiments/exp360_intruder/codex_h2fail_decision.md:5894:experiments/paradigm_shift/paradigm_C_free.md:2391:experiments/exp110/design.md:5:- `exp109` 的 oracle support bank 诊断给出极强 headroom：
experiments/exp367_single_support/codex_train_design.md:2869:experiments/exp367_single_support/codex_review.md:1356:experiments/exp360_intruder/codex_h2fail_decision.md:5900:experiments/paradigm_shift/paradigm_C_free.md:2464:experiments/exp110/monitor.md:45:  3. 但至少排除了“为了维护 support bank，前 10 个 epoch 就明显掉点”的风险
experiments/exp367_single_support/codex_train_design.md:2870:experiments/exp367_single_support/codex_review.md:1357:experiments/exp360_intruder/codex_h2fail_decision.md:5920:experiments/paradigm_shift/paradigm_C_free.md:4182:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14742:experiments/cargo_cvpb/codex_review_raw.txt:4140:experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:2871:experiments/exp367_single_support/codex_review.md:1358:experiments/exp360_intruder/codex_h2fail_decision.md:5921:experiments/paradigm_shift/paradigm_C_free.md:4195:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14789:experiments/cargo_cvpb/codex_review_raw.txt:4896:./experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:2872:experiments/exp367_single_support/codex_review.md:1359:experiments/exp360_intruder/codex_h2fail_decision.md:5939:experiments/paradigm_shift/paradigm_C_free.md:5212:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:2873:experiments/exp367_single_support/codex_review.md:1360:experiments/exp360_intruder/codex_h2fail_decision.md:5940:experiments/paradigm_shift/paradigm_C_free.md:5213:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp367_single_support/codex_train_design.md:2874:experiments/exp367_single_support/codex_review.md:1361:experiments/exp360_intruder/codex_h2fail_decision.md:5941:experiments/paradigm_shift/paradigm_C_free.md:5233:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:2875:experiments/exp367_single_support/codex_review.md:1362:experiments/exp360_intruder/codex_h2fail_decision.md:5942:experiments/paradigm_shift/paradigm_C_free.md:5262:  - support bank 训练线应止损
experiments/exp367_single_support/codex_train_design.md:2876:experiments/exp367_single_support/codex_review.md:1363:experiments/exp360_intruder/codex_h2fail_decision.md:6150:experiments/paradigm_shift/paradigm_C_free.md:8046:experiments/paradigm_shift/paradigm_C_free.md:5233:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:2877:experiments/exp367_single_support/codex_review.md:1364:experiments/exp360_intruder/codex_h2fail_decision.md:6159:experiments/paradigm_shift/paradigm_C_free.md:8416:我会选 **#1 Temporal Support-Complete Distillation** 去 build。它最贴你们自己的硬证据：`exp109` oracle support bank 从 `61.88/73.26` 到 `66.15/77.87`，加 weight 到 `70.40/81.36`，说明根问题不是小模块，而是 **single-image support incomplete** [monitor](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp109/monitor.md:27)。
experiments/exp367_single_support/codex_train_design.md:2878:experiments/exp367_single_support/codex_review.md:1365:experiments/exp360_intruder/codex_h2fail_decision.md:6164:experiments/paradigm_shift/paradigm_C_free.md:8485:我会选 **#1 Temporal Support-Complete Distillation** 去 build。它最贴你们自己的硬证据：`exp109` oracle support bank 从 `61.88/73.26` 到 `66.15/77.87`，加 weight 到 `70.40/81.36`，说明根问题不是小模块，而是 **single-image support incomplete** [monitor](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp109/monitor.md:27)。
experiments/exp367_single_support/codex_train_design.md:2879:experiments/exp367_single_support/codex_review.md:1366:experiments/exp360_intruder/codex_h2fail_decision.md:6251:experiments/paradigm_shift/decision_tscd_vs_intruder.md:14:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2880:experiments/exp367_single_support/codex_review.md:1367:experiments/exp360_intruder/codex_h2fail_decision.md:6253:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1241:experiments/exp109/design.md:9:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:2881:experiments/exp367_single_support/codex_review.md:1368:experiments/exp360_intruder/codex_h2fail_decision.md:6254:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1242:experiments/exp109/design.md:10:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp367_single_support/codex_train_design.md:2882:experiments/exp367_single_support/codex_review.md:1369:experiments/exp360_intruder/codex_h2fail_decision.md:6255:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1245:experiments/exp109/design.md:30:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:2883:experiments/exp367_single_support/codex_review.md:1370:experiments/exp360_intruder/codex_h2fail_decision.md:6256:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1254:experiments/exp109/design.md:59:  - support bank 训练线应止损
experiments/exp367_single_support/codex_train_design.md:2884:experiments/exp367_single_support/codex_review.md:1371:experiments/exp360_intruder/codex_h2fail_decision.md:6257:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1391:experiments/decisions.md:3479:- `exp109` oracle support bank 仍是仓库内最强问题证据
experiments/exp367_single_support/codex_train_design.md:2885:experiments/exp367_single_support/codex_review.md:1372:experiments/exp360_intruder/codex_h2fail_decision.md:6258:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1499:experiments/innovation_brainstorm.md:1622:   **same-ID support bank → single-image support-complete distillation**
experiments/exp367_single_support/codex_train_design.md:2886:experiments/exp367_single_support/codex_review.md:1373:experiments/exp360_intruder/codex_h2fail_decision.md:6259:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1505:experiments/innovation_brainstorm.md:1651:   **reliable-support bank / teacher reliability gating**
experiments/exp367_single_support/codex_train_design.md:2887:experiments/exp367_single_support/codex_review.md:1374:experiments/exp360_intruder/codex_h2fail_decision.md:6260:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1515:experiments/innovation_brainstorm.md:1701:- Lagged / stale support bank
experiments/exp367_single_support/codex_train_design.md:2888:experiments/exp367_single_support/codex_review.md:1375:experiments/exp360_intruder/codex_h2fail_decision.md:6261:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1526:experiments/innovation_brainstorm.md:1849:3. 而是先用 `exp109` 方向的 support bank 补全 low-vis keypoint teacher，再用补全后的 teacher 去做 `CSRD`
experiments/exp367_single_support/codex_train_design.md:2889:experiments/exp367_single_support/codex_review.md:1376:experiments/exp360_intruder/codex_h2fail_decision.md:6262:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1571:experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:2890:experiments/exp367_single_support/codex_review.md:1377:experiments/exp360_intruder/codex_h2fail_decision.md:6268:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1956:experiments/exp148/design.md:6:但 `exp110-142` 也反复证明：把 same-ID 跨图 support bank 直接蒸到单图特征里，很难在 15K 数据上学成。
experiments/exp367_single_support/codex_train_design.md:2891:experiments/exp367_single_support/codex_review.md:1378:experiments/exp360_intruder/codex_h2fail_decision.md:6271:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2341:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5675:./reassess/r_2.md:5235:reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:2892:experiments/exp367_single_support/codex_review.md:1379:experiments/exp360_intruder/codex_h2fail_decision.md:6272:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2348:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5684:./reassess/r_2.md:5458:reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
experiments/exp367_single_support/codex_train_design.md:2893:experiments/exp367_single_support/codex_review.md:1380:experiments/exp360_intruder/codex_h2fail_decision.md:6273:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2365:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5797:./reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:2894:experiments/exp367_single_support/codex_review.md:1381:experiments/exp360_intruder/codex_h2fail_decision.md:6276:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2382:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5837:./reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
experiments/exp367_single_support/codex_train_design.md:2895:experiments/exp367_single_support/codex_review.md:1382:experiments/exp360_intruder/codex_h2fail_decision.md:6279:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2675:experiments/exp110/design.md:5:- `exp109` 的 oracle support bank 诊断给出极强 headroom：
experiments/exp367_single_support/codex_train_design.md:2896:experiments/exp367_single_support/codex_review.md:1383:experiments/exp360_intruder/codex_h2fail_decision.md:6280:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2691:experiments/exp110/monitor.md:45:  3. 但至少排除了“为了维护 support bank，前 10 个 epoch 就明显掉点”的风险
experiments/exp367_single_support/codex_train_design.md:2897:experiments/exp367_single_support/codex_review.md:1384:experiments/exp360_intruder/codex_h2fail_decision.md:6286:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2929:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2898:experiments/exp367_single_support/codex_review.md:1385:experiments/exp360_intruder/codex_h2fail_decision.md:6287:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2973:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2899:experiments/exp367_single_support/codex_review.md:1394:experiments/exp360_intruder/codex_h2fail_decision.md:6297:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3585:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14543:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5235:reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:2900:experiments/exp367_single_support/codex_review.md:1395:experiments/exp360_intruder/codex_h2fail_decision.md:6298:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3599:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14575:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5458:reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
experiments/exp367_single_support/codex_train_design.md:2901:experiments/exp367_single_support/codex_review.md:1398:experiments/exp360_intruder/codex_h2fail_decision.md:6301:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3656:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14980:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:2902:experiments/exp367_single_support/codex_review.md:1399:experiments/exp360_intruder/codex_h2fail_decision.md:6302:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3673:experiments/cargo_cvpb/litreview2/false_negative_audit.md:15079:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:4696:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:58:**执行计划**：exp324f agent 正在 lab-4090 算 Swin distmat → 它一落地我立刻跑 #1 oracle（0-GPU）。正向 → #2 re-rank（training-free 主表素材）。OT 线(#3)等某 GPU 空了上。
experiments/exp367_single_support/codex_train_design.md:2903:experiments/exp367_single_support/codex_review.md:1400:experiments/exp360_intruder/codex_h2fail_decision.md:6303:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3679:experiments/cargo_cvpb/litreview2/false_negative_audit.md:15093:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:4763:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:437:**KILL**: backdoor 要消的"协变量冒充身份"虚高**没侵蚀决策边界**(SOTA exp255 已把跨cam/跨遮挡同人正样本压得离边界数量级远), 去掉不改变任何排序=无可回收错误=无 headroom。**深刻含义: SOTA ReID 特征上, in-domain 去混淆类训练端机制无 headroom** — 呼应张力+三堵墙+MEMORY"别在 ReID 内部找机制"。#3 拓扑(也 in-domain)大概率同命; #2 UCE(测跨域/开集分数尺度漂移)测不同维度, 待定。
experiments/exp367_single_support/codex_train_design.md:2904:experiments/exp367_single_support/codex_review.md:1401:experiments/exp360_intruder/codex_h2fail_decision.md:6304:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4339:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14742:experiments/cargo_cvpb/codex_review_raw.txt:4140:experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:2905:experiments/exp367_single_support/codex_review.md:1402:experiments/exp360_intruder/codex_h2fail_decision.md:6305:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4340:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14789:experiments/cargo_cvpb/codex_review_raw.txt:4896:./experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:2906:experiments/exp367_single_support/codex_review.md:1403:experiments/exp360_intruder/codex_h2fail_decision.md:6306:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4477:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:2907:experiments/exp367_single_support/codex_review.md:1404:experiments/exp360_intruder/codex_h2fail_decision.md:6307:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4478:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp367_single_support/codex_train_design.md:2908:experiments/exp367_single_support/codex_review.md:1405:experiments/exp360_intruder/codex_h2fail_decision.md:6308:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4498:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:2909:experiments/exp367_single_support/codex_review.md:1406:experiments/exp360_intruder/codex_h2fail_decision.md:6309:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4527:  - support bank 训练线应止损
experiments/exp367_single_support/codex_train_design.md:2910:experiments/exp367_single_support/codex_review.md:1407:experiments/exp360_intruder/codex_h2fail_decision.md:6310:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4854:但 `exp110-142` 也反复证明：把 same-ID 跨图 support bank 直接蒸到单图特征里，很难在 15K 数据上学成。
experiments/exp367_single_support/codex_train_design.md:2911:experiments/exp367_single_support/codex_review.md:1408:experiments/exp360_intruder/codex_h2fail_decision.md:6311:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5763:./experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2912:experiments/exp367_single_support/codex_review.md:1409:experiments/exp360_intruder/codex_h2fail_decision.md:6312:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5782:./experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2913:experiments/exp367_single_support/codex_review.md:1410:experiments/exp360_intruder/codex_h2fail_decision.md:6313:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5872:./experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2914:experiments/exp367_single_support/codex_review.md:1411:experiments/exp360_intruder/codex_h2fail_decision.md:6314:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5888:./experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2915:experiments/exp367_single_support/codex_review.md:1412:experiments/exp360_intruder/codex_h2fail_decision.md:6315:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5938:./experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3982:../../cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2916:experiments/exp367_single_support/codex_review.md:1413:experiments/exp360_intruder/codex_h2fail_decision.md:6316:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6406:./experiments/cargo_cvpb/litreview2/reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2917:experiments/exp367_single_support/codex_review.md:1414:experiments/exp360_intruder/codex_h2fail_decision.md:6317:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6470:./experiments/cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2918:experiments/exp367_single_support/codex_review.md:1415:experiments/exp360_intruder/codex_h2fail_decision.md:6318:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6636:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2919:experiments/exp367_single_support/codex_review.md:1416:experiments/exp360_intruder/codex_h2fail_decision.md:6319:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6660:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2920:experiments/exp367_single_support/codex_review.md:1417:experiments/exp360_intruder/codex_h2fail_decision.md:6326:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6955:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:14:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2921:experiments/exp367_single_support/codex_review.md:1418:experiments/exp360_intruder/codex_h2fail_decision.md:6327:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6992:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:2929:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2922:experiments/exp367_single_support/codex_review.md:1419:experiments/exp360_intruder/codex_h2fail_decision.md:6328:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7011:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:2973:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2923:experiments/exp367_single_support/codex_review.md:1420:experiments/exp360_intruder/codex_h2fail_decision.md:6330:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7649:experiments/innovation_brainstorm.md:1622:   **same-ID support bank → single-image support-complete distillation**
experiments/exp367_single_support/codex_train_design.md:2924:experiments/exp367_single_support/codex_review.md:1421:experiments/exp360_intruder/codex_h2fail_decision.md:6331:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7655:experiments/innovation_brainstorm.md:1651:   **reliable-support bank / teacher reliability gating**
experiments/exp367_single_support/codex_train_design.md:2925:experiments/exp367_single_support/codex_review.md:1422:experiments/exp360_intruder/codex_h2fail_decision.md:6332:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7679:experiments/innovation_brainstorm.md:1849:3. 而是先用 `exp109` 方向的 support bank 补全 low-vis keypoint teacher，再用补全后的 teacher 去做 `CSRD`
experiments/exp367_single_support/codex_train_design.md:2926:experiments/exp367_single_support/codex_review.md:1423:experiments/exp360_intruder/codex_h2fail_decision.md:6333:experiments/paradigm_shift/decision_tscd_vs_intruder.md:8016:experiments/exp109/design.md:9:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:2927:experiments/exp367_single_support/codex_review.md:1424:experiments/exp360_intruder/codex_h2fail_decision.md:6335:experiments/paradigm_shift/decision_tscd_vs_intruder.md:8301:experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:2928:experiments/exp367_single_support/codex_review.md:1425:experiments/exp360_intruder/codex_h2fail_decision.md:6337:experiments/paradigm_shift/decision_tscd_vs_intruder.md:10150:experiments/cargo_cvpb/litreview2/train4_final.md:4187:experiments/exp120/design.md:17:2. 如果用 same-ID support bank 补全 low-vis keypoint teacher，再做 relational distillation，收益应强于 `exp119`
experiments/exp367_single_support/codex_train_design.md:2929:experiments/exp367_single_support/codex_review.md:1426:experiments/exp360_intruder/codex_h2fail_decision.md:6338:experiments/paradigm_shift/decision_tscd_vs_intruder.md:11555:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2930:experiments/exp367_single_support/codex_review.md:1427:experiments/exp360_intruder/codex_h2fail_decision.md:6341:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15127:experiments/paradigm_shift/paradigm_C_free.md:8416:我会选 **#1 Temporal Support-Complete Distillation** 去 build。它最贴你们自己的硬证据：`exp109` oracle support bank 从 `61.88/73.26` 到 `66.15/77.87`，加 weight 到 `70.40/81.36`，说明根问题不是小模块，而是 **single-image support incomplete** [monitor](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp109/monitor.md:27)。
experiments/exp367_single_support/codex_train_design.md:2931:experiments/exp367_single_support/codex_review.md:1428:experiments/exp360_intruder/codex_h2fail_decision.md:6342:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15135:experiments/paradigm_shift/paradigm_C_free.md:8485:我会选 **#1 Temporal Support-Complete Distillation** 去 build。它最贴你们自己的硬证据：`exp109` oracle support bank 从 `61.88/73.26` 到 `66.15/77.87`，加 weight 到 `70.40/81.36`，说明根问题不是小模块，而是 **single-image support incomplete** [monitor](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp109/monitor.md:27)。
experiments/exp367_single_support/codex_train_design.md:2932:experiments/exp367_single_support/codex_review.md:1429:experiments/exp360_intruder/codex_h2fail_decision.md:6343:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15143:experiments/paradigm_shift/decision_tscd_vs_intruder.md:14:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2933:experiments/exp367_single_support/codex_review.md:1430:experiments/exp360_intruder/codex_h2fail_decision.md:6344:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15158:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2929:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2934:experiments/exp367_single_support/codex_review.md:1431:experiments/exp360_intruder/codex_h2fail_decision.md:6345:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15173:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2973:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2935:experiments/exp367_single_support/codex_review.md:1432:experiments/exp360_intruder/codex_h2fail_decision.md:6346:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15257:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5763:./experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2936:experiments/exp367_single_support/codex_review.md:1433:experiments/exp360_intruder/codex_h2fail_decision.md:6347:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15270:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5782:./experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2937:experiments/exp367_single_support/codex_review.md:1434:experiments/exp360_intruder/codex_h2fail_decision.md:6348:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15297:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5872:./experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2938:experiments/exp367_single_support/codex_review.md:1435:experiments/exp360_intruder/codex_h2fail_decision.md:6349:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15310:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5888:./experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2939:experiments/exp367_single_support/codex_review.md:1436:experiments/exp360_intruder/codex_h2fail_decision.md:6350:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15338:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5938:./experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3982:../../cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2940:experiments/exp367_single_support/codex_review.md:1437:experiments/exp360_intruder/codex_h2fail_decision.md:6351:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15444:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6406:./experiments/cargo_cvpb/litreview2/reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2941:experiments/exp367_single_support/codex_review.md:1438:experiments/exp360_intruder/codex_h2fail_decision.md:6352:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15471:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6470:./experiments/cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2942:experiments/exp367_single_support/codex_review.md:1439:experiments/exp360_intruder/codex_h2fail_decision.md:6353:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15528:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6636:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2943:experiments/exp367_single_support/codex_review.md:1440:experiments/exp360_intruder/codex_h2fail_decision.md:6354:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15547:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6660:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2944:experiments/exp367_single_support/codex_review.md:1441:experiments/exp360_intruder/codex_h2fail_decision.md:6355:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15641:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6955:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:14:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2945:experiments/exp367_single_support/codex_review.md:1442:experiments/exp360_intruder/codex_h2fail_decision.md:6356:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15653:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6992:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:2929:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2946:experiments/exp367_single_support/codex_review.md:1443:experiments/exp360_intruder/codex_h2fail_decision.md:6357:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15666:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7011:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:2973:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2947:experiments/exp367_single_support/codex_review.md:1444:experiments/exp360_intruder/codex_h2fail_decision.md:6358:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15742:experiments/paradigm_shift/decision_tscd_vs_intruder.md:11555:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2948:experiments/exp367_single_support/codex_review.md:1445:experiments/exp360_intruder/codex_h2fail_decision.md:6359:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15909:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2949:experiments/exp367_single_support/codex_review.md:1446:experiments/exp360_intruder/codex_h2fail_decision.md:6360:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15924:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2950:experiments/exp367_single_support/codex_review.md:1447:experiments/exp360_intruder/codex_h2fail_decision.md:6361:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15955:experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3982:../../cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2951:experiments/exp367_single_support/codex_review.md:1448:experiments/exp360_intruder/codex_h2fail_decision.md:6362:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15981:experiments/cargo_cvpb/litreview2/reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2952:experiments/exp367_single_support/codex_review.md:1449:experiments/exp360_intruder/codex_h2fail_decision.md:6363:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16012:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2953:experiments/exp367_single_support/codex_review.md:1450:experiments/exp360_intruder/codex_h2fail_decision.md:6364:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16063:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2954:experiments/exp367_single_support/codex_review.md:1451:experiments/exp360_intruder/codex_h2fail_decision.md:6365:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16078:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2955:experiments/exp367_single_support/codex_review.md:1452:experiments/exp360_intruder/codex_h2fail_decision.md:6366:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16139:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2956:experiments/exp367_single_support/codex_review.md:1453:experiments/exp360_intruder/codex_h2fail_decision.md:6367:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16162:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5843:reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2957:experiments/exp367_single_support/codex_review.md:1454:experiments/exp360_intruder/codex_h2fail_decision.md:6411:experiments/decisions.md:3479:- `exp109` oracle support bank 仍是仓库内最强问题证据
experiments/exp367_single_support/codex_train_design.md:2958:experiments/exp367_single_support/codex_review.md:1455:experiments/exp360_intruder/codex_h2fail_decision.md:6597:experiments/paradigm_shift/paradigm_B_pretraining.md:4799:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:2959:experiments/exp367_single_support/codex_review.md:1456:experiments/exp360_intruder/codex_h2fail_decision.md:6598:experiments/paradigm_shift/paradigm_B_pretraining.md:4800:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp367_single_support/codex_train_design.md:2960:experiments/exp367_single_support/codex_review.md:1457:experiments/exp360_intruder/codex_h2fail_decision.md:6599:experiments/paradigm_shift/paradigm_B_pretraining.md:4820:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:2961:experiments/exp367_single_support/codex_review.md:1458:experiments/exp360_intruder/codex_h2fail_decision.md:6600:experiments/paradigm_shift/paradigm_B_pretraining.md:4849:  - support bank 训练线应止损
experiments/exp367_single_support/codex_train_design.md:2962:experiments/exp367_single_support/codex_review.md:1459:experiments/exp360_intruder/codex_h2fail_decision.md:6601:experiments/paradigm_shift/paradigm_B_pretraining.md:5157:本地 exp109 给出的上界很强：GT same-ID per-keypoint support bank 从 61.88/73.26 提到 70.40/81.36，低可见 query 上几乎是数量级提升。这说明“support-complete 表征”不能判死；问题是已有训练端实现没有足够稳定的 teacher/support，而不是目标没有 headroom。
experiments/exp367_single_support/codex_train_design.md:2963:experiments/exp367_single_support/codex_review.md:1460:experiments/exp360_intruder/codex_h2fail_decision.md:6604:experiments/paradigm_shift/paradigm_B_pretraining.md:5606:> 给模型一张不完整人体图，要求它在 latent body-part token 空间里预测“完整身份 support”，target 来自 EMA full-view teacher + 高置信跨图/伪同 ID support bank。
experiments/exp367_single_support/codex_train_design.md:2964:experiments/exp367_single_support/codex_review.md:1461:experiments/exp360_intruder/codex_h2fail_decision.md:6606:experiments/paradigm_shift/paradigm_B_pretraining.md:5623:- **不是 single-image MAE，而是 support bank / pseudo cross-view teacher**。
experiments/exp367_single_support/codex_train_design.md:2965:experiments/exp367_single_support/codex_review.md:1462:experiments/exp360_intruder/codex_h2fail_decision.md:6609:experiments/paradigm_shift/paradigm_B_pretraining.md:5644:     - `T_bank`: 高置信 pseudo same-ID / nearest-neighbor support bank 中对应 body part prototype。
experiments/exp367_single_support/codex_train_design.md:2966:experiments/exp367_single_support/codex_review.md:1463:experiments/exp360_intruder/codex_h2fail_decision.md:6611:experiments/paradigm_shift/paradigm_B_pretraining.md:5662:   - 对 exp109：把 oracle support bank 的 headroom 尝试蒸进预训练。你本地 exp109 显示 oracle support 从 `61.88/73.26` 到 `70.40/81.36`，这条线有真实上界。
experiments/exp367_single_support/codex_train_design.md:2967:experiments/exp367_single_support/codex_review.md:1464:experiments/exp360_intruder/codex_h2fail_decision.md:6613:experiments/paradigm_shift/paradigm_B_pretraining.md:5666:- 3090：去掉 support bank，只做 same-image full teacher，对照“是否只是 OA-SD/PCVT 换名”。
experiments/exp367_single_support/codex_train_design.md:2968:experiments/exp367_single_support/codex_review.md:1465:experiments/exp360_intruder/codex_h2fail_decision.md:6614:experiments/paradigm_shift/paradigm_B_pretraining.md:5668:- 5060Ti-2：pseudo support bank 质量诊断、DINOv2 frozen teacher variant、小规模 Occluded-Duke smoke。
experiments/exp367_single_support/codex_train_design.md:2969:experiments/exp367_single_support/codex_review.md:1466:experiments/exp360_intruder/codex_h2fail_decision.md:6615:experiments/paradigm_shift/paradigm_B_pretraining.md:5677:最大风险不是算力，而是 pseudo support bank 噪声。第一 kill-switch 要看：
experiments/exp367_single_support/codex_train_design.md:2970:experiments/exp367_single_support/codex_review.md:1467:experiments/exp360_intruder/codex_h2fail_decision.md:6618:experiments/paradigm_shift/paradigm_B_pretraining.md:5688:- Novelty：`6.5/10`，如果 support bank + latent JEPA 做干净，可到 `7/10`
experiments/exp367_single_support/codex_train_design.md:2971:experiments/exp367_single_support/codex_review.md:1468:experiments/exp360_intruder/codex_h2fail_decision.md:6621:experiments/paradigm_shift/paradigm_B_pretraining.md:5698:> 给模型一张不完整人体图，要求它在 latent body-part token 空间里预测“完整身份 support”，target 来自 EMA full-view teacher + 高置信跨图/伪同 ID support bank。
experiments/exp367_single_support/codex_train_design.md:2972:experiments/exp367_single_support/codex_review.md:1469:experiments/exp360_intruder/codex_h2fail_decision.md:6623:experiments/paradigm_shift/paradigm_B_pretraining.md:5715:- **不是 single-image MAE，而是 support bank / pseudo cross-view teacher**。
experiments/exp367_single_support/codex_train_design.md:2973:experiments/exp367_single_support/codex_review.md:1470:experiments/exp360_intruder/codex_h2fail_decision.md:6626:experiments/paradigm_shift/paradigm_B_pretraining.md:5736:     - `T_bank`: 高置信 pseudo same-ID / nearest-neighbor support bank 中对应 body part prototype。
experiments/exp367_single_support/codex_train_design.md:2974:experiments/exp367_single_support/codex_review.md:1471:experiments/exp360_intruder/codex_h2fail_decision.md:6628:experiments/paradigm_shift/paradigm_B_pretraining.md:5754:   - 对 exp109：把 oracle support bank 的 headroom 尝试蒸进预训练。你本地 exp109 显示 oracle support 从 `61.88/73.26` 到 `70.40/81.36`，这条线有真实上界。
experiments/exp367_single_support/codex_train_design.md:2975:experiments/exp367_single_support/codex_review.md:1472:experiments/exp360_intruder/codex_h2fail_decision.md:6630:experiments/paradigm_shift/paradigm_B_pretraining.md:5758:- 3090：去掉 support bank，只做 same-image full teacher，对照“是否只是 OA-SD/PCVT 换名”。
experiments/exp367_single_support/codex_train_design.md:2976:experiments/exp367_single_support/codex_review.md:1473:experiments/exp360_intruder/codex_h2fail_decision.md:6631:experiments/paradigm_shift/paradigm_B_pretraining.md:5760:- 5060Ti-2：pseudo support bank 质量诊断、DINOv2 frozen teacher variant、小规模 Occluded-Duke smoke。
experiments/exp367_single_support/codex_train_design.md:2977:experiments/exp367_single_support/codex_review.md:1474:experiments/exp360_intruder/codex_h2fail_decision.md:6632:experiments/paradigm_shift/paradigm_B_pretraining.md:5769:最大风险不是算力，而是 pseudo support bank 噪声。第一 kill-switch 要看：
experiments/exp367_single_support/codex_train_design.md:2978:experiments/exp367_single_support/codex_review.md:1475:experiments/exp360_intruder/codex_h2fail_decision.md:6635:experiments/paradigm_shift/paradigm_B_pretraining.md:5780:- Novelty：`6.5/10`，如果 support bank + latent JEPA 做干净，可到 `7/10`
experiments/exp367_single_support/codex_train_design.md:2979:experiments/exp367_single_support/codex_review.md:1476:experiments/exp360_intruder/codex_h2fail_decision.md:6793:experiments/paradigm_shift/decision_tscd_vs_intruder.md:14:ReID 范式转向终局决策(调研交给你, 不判死, 帮选+审视可行性)。3 路 gap analysis 收敛: 遮挡 ReID 根问题=single-image support incomplete(项目 exp109 oracle 证 +8.5 mAP headroom)。两个最强候选二选一去 build: **#1 T-SCD(Temporal Support-Complete Distillation, 7.5/10)**: 训练用 occluded_posetrack tracklet(同 ID 多帧)构 support-complete teacher 蒸馏进单图 student, 测试单图。**#2 Intruder Identity Suppression(6.8/10)**: 遮挡重定义为另一行人身份泄漏进 target embedding, 训练合成 target+donor 显式压 donor-ID(GRL/adversarial)+clean-occluded consistency。**关键审视**: 项目 memory 记 exp109 的 +8.5 oracle headroom 是'identity-conditioned 不可实现的墙'(测试时不知 ID 没法 complete support; 三堵墙 completion/occluder-gate/95.8%可见; 当年结论'别在 ReID 内部找机制')。问: (a)T-SCD 把 identity-conditioned support 放训练端(tracklet privileged info)蒸馏进单图 student, 是真绕过 exp109 测试端墙, 还是又撞同一堵墙(privileged distillation gap/student 学不会)? 诚实判断别捧场。(b)T-SCD vs Intruder 哪个更值得 build(野心×可行×novelty×B类×绕开项目死区)? (c)选定方向的最终 pipeline + 头号风险 + 头号 kill-switch(第一周能看的)。联网查 privileged/temporal distillation to single-image occluded ReID + identity source separation ReID 的 2024-26 novelty 先例。务实中文, 信心 1-10。
experiments/exp367_single_support/codex_train_design.md:2980:experiments/exp367_single_support/codex_review.md:1477:experiments/exp360_intruder/codex_h2fail_decision.md:6795:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1241:experiments/exp109/design.md:9:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:2981:experiments/exp367_single_support/codex_review.md:1478:experiments/exp360_intruder/codex_h2fail_decision.md:6796:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1242:experiments/exp109/design.md:10:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp367_single_support/codex_train_design.md:2982:experiments/exp367_single_support/codex_review.md:1479:experiments/exp360_intruder/codex_h2fail_decision.md:6797:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1245:experiments/exp109/design.md:30:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:2983:experiments/exp367_single_support/codex_review.md:1480:experiments/exp360_intruder/codex_h2fail_decision.md:6798:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1254:experiments/exp109/design.md:59:  - support bank 训练线应止损
experiments/exp367_single_support/codex_train_design.md:2984:experiments/exp367_single_support/codex_review.md:1481:experiments/exp360_intruder/codex_h2fail_decision.md:6799:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1391:experiments/decisions.md:3479:- `exp109` oracle support bank 仍是仓库内最强问题证据
experiments/exp367_single_support/codex_train_design.md:2985:experiments/exp367_single_support/codex_review.md:1482:experiments/exp360_intruder/codex_h2fail_decision.md:6800:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1499:experiments/innovation_brainstorm.md:1622:   **same-ID support bank → single-image support-complete distillation**
experiments/exp367_single_support/codex_train_design.md:2986:experiments/exp367_single_support/codex_review.md:1483:experiments/exp360_intruder/codex_h2fail_decision.md:6801:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1505:experiments/innovation_brainstorm.md:1651:   **reliable-support bank / teacher reliability gating**
experiments/exp367_single_support/codex_train_design.md:2987:experiments/exp367_single_support/codex_review.md:1484:experiments/exp360_intruder/codex_h2fail_decision.md:6802:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1515:experiments/innovation_brainstorm.md:1701:- Lagged / stale support bank
experiments/exp367_single_support/codex_train_design.md:2988:experiments/exp367_single_support/codex_review.md:1485:experiments/exp360_intruder/codex_h2fail_decision.md:6803:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1526:experiments/innovation_brainstorm.md:1849:3. 而是先用 `exp109` 方向的 support bank 补全 low-vis keypoint teacher，再用补全后的 teacher 去做 `CSRD`
experiments/exp367_single_support/codex_train_design.md:2989:experiments/exp367_single_support/codex_review.md:1486:experiments/exp360_intruder/codex_h2fail_decision.md:6804:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1571:experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:2990:experiments/exp367_single_support/codex_review.md:1487:experiments/exp360_intruder/codex_h2fail_decision.md:6810:experiments/paradigm_shift/decision_tscd_vs_intruder.md:1956:experiments/exp148/design.md:6:但 `exp110-142` 也反复证明：把 same-ID 跨图 support bank 直接蒸到单图特征里，很难在 15K 数据上学成。
experiments/exp367_single_support/codex_train_design.md:2991:experiments/exp367_single_support/codex_review.md:1488:experiments/exp360_intruder/codex_h2fail_decision.md:6813:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2341:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5675:./reassess/r_2.md:5235:reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:2992:experiments/exp367_single_support/codex_review.md:1489:experiments/exp360_intruder/codex_h2fail_decision.md:6814:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2348:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5684:./reassess/r_2.md:5458:reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
experiments/exp367_single_support/codex_train_design.md:2993:experiments/exp367_single_support/codex_review.md:1490:experiments/exp360_intruder/codex_h2fail_decision.md:6815:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2365:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5797:./reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:2994:experiments/exp367_single_support/codex_review.md:1491:experiments/exp360_intruder/codex_h2fail_decision.md:6818:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2382:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5837:./reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
experiments/exp367_single_support/codex_train_design.md:2995:experiments/exp367_single_support/codex_review.md:1492:experiments/exp360_intruder/codex_h2fail_decision.md:6821:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2675:experiments/exp110/design.md:5:- `exp109` 的 oracle support bank 诊断给出极强 headroom：
experiments/exp367_single_support/codex_train_design.md:2996:experiments/exp367_single_support/codex_review.md:1493:experiments/exp360_intruder/codex_h2fail_decision.md:6822:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2691:experiments/exp110/monitor.md:45:  3. 但至少排除了“为了维护 support bank，前 10 个 epoch 就明显掉点”的风险
experiments/exp367_single_support/codex_train_design.md:2997:experiments/exp367_single_support/codex_review.md:1494:experiments/exp360_intruder/codex_h2fail_decision.md:6828:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2929:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2998:experiments/exp367_single_support/codex_review.md:1495:experiments/exp360_intruder/codex_h2fail_decision.md:6829:experiments/paradigm_shift/decision_tscd_vs_intruder.md:2973:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:2999:experiments/exp367_single_support/codex_review.md:1504:experiments/exp360_intruder/codex_h2fail_decision.md:6839:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3585:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14543:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5235:reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:3000:experiments/exp367_single_support/codex_review.md:1505:experiments/exp360_intruder/codex_h2fail_decision.md:6840:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3599:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14575:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5458:reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
experiments/exp367_single_support/codex_train_design.md:3001:experiments/exp367_single_support/codex_review.md:1508:experiments/exp360_intruder/codex_h2fail_decision.md:6843:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3656:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14980:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/exp367_single_support/codex_train_design.md:3002:experiments/exp367_single_support/codex_review.md:1509:experiments/exp360_intruder/codex_h2fail_decision.md:6844:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3673:experiments/cargo_cvpb/litreview2/false_negative_audit.md:15079:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:4696:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:58:**执行计划**：exp324f agent 正在 lab-4090 算 Swin distmat → 它一落地我立刻跑 #1 oracle（0-GPU）。正向 → #2 re-rank（training-free 主表素材）。OT 线(#3)等某 GPU 空了上。
experiments/exp367_single_support/codex_train_design.md:3003:experiments/exp367_single_support/codex_review.md:1510:experiments/exp360_intruder/codex_h2fail_decision.md:6845:experiments/paradigm_shift/decision_tscd_vs_intruder.md:3679:experiments/cargo_cvpb/litreview2/false_negative_audit.md:15093:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:4763:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:437:**KILL**: backdoor 要消的"协变量冒充身份"虚高**没侵蚀决策边界**(SOTA exp255 已把跨cam/跨遮挡同人正样本压得离边界数量级远), 去掉不改变任何排序=无可回收错误=无 headroom。**深刻含义: SOTA ReID 特征上, in-domain 去混淆类训练端机制无 headroom** — 呼应张力+三堵墙+MEMORY"别在 ReID 内部找机制"。#3 拓扑(也 in-domain)大概率同命; #2 UCE(测跨域/开集分数尺度漂移)测不同维度, 待定。
experiments/exp367_single_support/codex_train_design.md:3004:experiments/exp367_single_support/codex_review.md:1511:experiments/exp360_intruder/codex_h2fail_decision.md:6846:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4339:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14742:experiments/cargo_cvpb/codex_review_raw.txt:4140:experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:3005:experiments/exp367_single_support/codex_review.md:1512:experiments/exp360_intruder/codex_h2fail_decision.md:6847:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4340:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14789:experiments/cargo_cvpb/codex_review_raw.txt:4896:./experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:3006:experiments/exp367_single_support/codex_review.md:1513:experiments/exp360_intruder/codex_h2fail_decision.md:6848:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4477:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp367_single_support/codex_train_design.md:3007:experiments/exp367_single_support/codex_review.md:1514:experiments/exp360_intruder/codex_h2fail_decision.md:6849:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4478:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp367_single_support/codex_train_design.md:3008:experiments/exp367_single_support/codex_review.md:1515:experiments/exp360_intruder/codex_h2fail_decision.md:6850:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4498:### 2. Oracle support bank 构造
experiments/exp367_single_support/codex_train_design.md:3009:experiments/exp367_single_support/codex_review.md:1516:experiments/exp360_intruder/codex_h2fail_decision.md:6851:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4527:  - support bank 训练线应止损
experiments/exp367_single_support/codex_train_design.md:3010:experiments/exp367_single_support/codex_review.md:1517:experiments/exp360_intruder/codex_h2fail_decision.md:6852:experiments/paradigm_shift/decision_tscd_vs_intruder.md:4854:但 `exp110-142` 也反复证明：把 same-ID 跨图 support bank 直接蒸到单图特征里，很难在 15K 数据上学成。
experiments/exp367_single_support/codex_train_design.md:3011:experiments/exp367_single_support/codex_review.md:1518:experiments/exp360_intruder/codex_h2fail_decision.md:6853:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5763:./experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4448:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:3012:experiments/exp367_single_support/codex_review.md:1519:experiments/exp360_intruder/codex_h2fail_decision.md:6854:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5782:./experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4659:./reassess/r_3.md:8051:./reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能不能被检测”，必须在 frozen 特征上直接测“无 GT 身份、只按遮挡源压制后，排序是否上升”。
experiments/exp367_single_support/codex_train_design.md:3013:experiments/exp367_single_support/codex_review.md:1520:experiments/exp360_intruder/codex_h2fail_decision.md:6855:experiments/paradigm_shift/decision_tscd_vs_intruder.md:5872:./experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3622:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5546:reassess/r_3.md:5139:本地 exp109 的原始记录比当前叙述更细：oracle support bank 上界是强阳性，但它用 GT same-ID prototype，是“知道目标身份以后怎么补”的上界；这正是 donor-source-separation 必须绕开的墙。真正的 kill-switch 不能只测“泄漏能丼�原 rho+0.60/+0.65 被 circular self-loop 高估, 且 M(q) 控住 trivial 代理 `#false-in-topk` 后无独立信号——诊断的"hub/拓扑"框架不成立, 退化为"top-k 里错的多"。诚实记录。**
experiments/exp367_single_support/codex_train_design.md:3014:experiments/exp367_single_support/codex_review.md:1529:experiments/cargo_cvpb/litreview2/d17_eval.md:9187:experiments/cargo_cvpb/litreview2/false_negative_audit.md:989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/hubness_paper_review.md:212:3. 需要控制更强的 cheap baselines：`#false in top-k`、top-k precision、first positive rank、mean negative similarity、top-1 correctness、positive count、camera pair、feature norm、margin。现在只控 norm/margin/camera/#pos 还不够。
experiments/exp367_single_support/codex_train_design.md:3015:experiments/exp367_single_support/codex_review.md:1530:experiments/cargo_cvpb/litreview2/d17_eval.md:9188:experiments/cargo_cvpb/litreview2/false_negative_audit.md:998:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/hubness_paper_review.md:266:3. 需要控制更强的 cheap baselines：`#false in top-k`、top-k precision、first positive rank、mean negative similarity、top-1 correctness、positive count、camera pair、feature norm、margin。现在只控 norm/margin/camera/#pos 还不够。
experiments/exp367_single_support/codex_train_design.md:3016:experiments/exp367_single_support/codex_review.md:1531:experiments/cargo_cvpb/litreview2/d17_eval.md:9189:experiments/cargo_cvpb/litreview2/false_negative_audit.md:1117:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:39:| 6 | **Rank-Regret** | 效率/Pareto | cheap-vs-full 排名不一致路由算力 | 撞 CFPER（RI partial 控 #false 后≈0）+ Swin 无 cheap exit（算力集中 stage2 92%，cascade 省 1-5% 无意义） |
experiments/exp367_single_support/codex_train_design.md:3017:experiments/exp367_single_support/codex_review.md:1626:experiments/cargo_cvpb/litreview2/d17_eval.md:9284:experiments/cargo_cvpb/litreview2/false_negative_audit.md:3518:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:7390:./validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3018:experiments/exp367_single_support/codex_review.md:1628:experiments/cargo_cvpb/litreview2/d17_eval.md:9286:experiments/cargo_cvpb/litreview2/false_negative_audit.md:3644:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:8058:./validate/v_2.md:12734:validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3019:experiments/exp367_single_support/codex_review.md:1632:experiments/cargo_cvpb/litreview2/d17_eval.md:9290:experiments/cargo_cvpb/litreview2/false_negative_audit.md:4223:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:14206:./validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3020:experiments/exp367_single_support/codex_review.md:1636:experiments/cargo_cvpb/litreview2/d17_eval.md:9294:experiments/cargo_cvpb/litreview2/false_negative_audit.md:4342:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:15190:../litreview2/validate/v_3.md:7390:./validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3021:experiments/exp367_single_support/codex_review.md:1638:experiments/cargo_cvpb/litreview2/d17_eval.md:9296:experiments/cargo_cvpb/litreview2/false_negative_audit.md:4445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:15502:../litreview2/validate/v_3.md:8058:./validate/v_2.md:12734:validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3022:experiments/exp367_single_support/codex_review.md:1640:experiments/cargo_cvpb/litreview2/d17_eval.md:9298:experiments/cargo_cvpb/litreview2/false_negative_audit.md:4543:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:16106:../litreview2/validate/v_2.md:12734:validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3023:experiments/exp367_single_support/codex_review.md:1669:experiments/cargo_cvpb/litreview2/d17_eval.md:9327:experiments/cargo_cvpb/litreview2/false_negative_audit.md:7696:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_2.md:12734:validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3024:experiments/exp367_single_support/codex_review.md:1674:experiments/cargo_cvpb/litreview2/d17_eval.md:9332:experiments/cargo_cvpb/litreview2/false_negative_audit.md:7899:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3025:experiments/exp367_single_support/codex_review.md:1679:experiments/cargo_cvpb/litreview2/d17_eval.md:9337:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8067:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:7390:./validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3026:experiments/exp367_single_support/codex_review.md:1680:experiments/cargo_cvpb/litreview2/d17_eval.md:9338:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8095:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:8058:./validate/v_2.md:12734:validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3027:experiments/exp367_single_support/codex_review.md:1681:experiments/cargo_cvpb/litreview2/d17_eval.md:9339:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8610:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:14206:./validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3028:experiments/exp367_single_support/codex_review.md:1682:experiments/cargo_cvpb/litreview2/d17_eval.md:9340:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8616:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:15190:../litreview2/validate/v_3.md:7390:./validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3029:experiments/exp367_single_support/codex_review.md:1683:experiments/cargo_cvpb/litreview2/d17_eval.md:9341:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8618:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:15502:../litreview2/validate/v_3.md:8058:./validate/v_2.md:12734:validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3030:experiments/exp367_single_support/codex_review.md:1684:experiments/cargo_cvpb/litreview2/d17_eval.md:9342:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8620:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:16106:../litreview2/validate/v_2.md:12734:validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3031:experiments/exp367_single_support/codex_review.md:1787:experiments/cargo_cvpb/litreview2/d17_eval.md:9445:experiments/cargo_cvpb/litreview2/false_negative_audit.md:11030:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:3151:../litreview/reviews/lit_17.md:3140:cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。
experiments/exp367_single_support/codex_train_design.md:3032:experiments/exp367_single_support/codex_review.md:1788:experiments/cargo_cvpb/litreview2/d17_eval.md:9446:experiments/cargo_cvpb/litreview2/false_negative_audit.md:11034:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:3162:../litreview/reviews/lit_17.md:3219:cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。
experiments/exp367_single_support/codex_train_design.md:3033:experiments/exp367_single_support/codex_review.md:1797:experiments/cargo_cvpb/litreview2/d17_eval.md:9455:experiments/cargo_cvpb/litreview2/false_negative_audit.md:12201:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_2.md:12752:validate/v_3.md:3151:../litreview/reviews/lit_17.md:3140:cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。
experiments/exp367_single_support/codex_train_design.md:3034:experiments/exp367_single_support/codex_review.md:1798:experiments/cargo_cvpb/litreview2/d17_eval.md:9456:experiments/cargo_cvpb/litreview2/false_negative_audit.md:12203:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_2.md:12754:validate/v_3.md:3162:../litreview/reviews/lit_17.md:3219:cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。
experiments/exp367_single_support/codex_train_design.md:3035:experiments/exp367_single_support/codex_review.md:1801:experiments/cargo_cvpb/litreview2/d17_eval.md:9459:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14335:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2774:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2755:../../cargo_cvpb/litreview/reviews/lit_17.md:3140:cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。
experiments/exp367_single_support/codex_train_design.md:3036:experiments/exp367_single_support/codex_review.md:1803:experiments/cargo_cvpb/litreview2/d17_eval.md:9461:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14405:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3156:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:4716:reassess/r_2.md:2564:pivot/p_2.md:3655:./validate/v_3.md:15502:../litreview2/validate/v_3.md:8058:./validate/v_2.md:12734:validate/v_3.md:2776:../codex_review_ovli.txt:3981:experiments/overnight_innovation_log.md:405:按用户睡前"不停+cheap kill-switch first"，自主把 TBPS 唯一幸存候选 PartNC 推到"数据一到就能判生死"：agent 读懂 RDE 接口(checkpoint 加载 / image 192-patch+text 77-token 特征 / CCD 的 min-max+2-GMM+低均值簇后验)，写 `scripts/partnc_killswitch.py`(本地+lab-3090-d，md5 一致)：(a) spaCy+lexicon 切 caption head/torso/legs + 部位级噪声注入(参数化, 有 regex fallback)；(b) RDE token 特征算 noun-phrase×patch per-part MaxSim cleanliness(复刻 eval_fliptest_maxsim einsum.max)；(c) AUROC(part-level) vs AUROC(实例级CCD)；(d) verdict。逻辑全验证通过(切分/注入/加载/形状/MaxSim/GMM/AUROC 方向性单测)。
experiments/exp367_single_support/codex_train_design.md:3037:experiments/exp367_single_support/codex_review.md:1905:experiments/cargo_cvpb/litreview2/d17_eval.md:9563:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17930:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4401:./reassess/r_3.md:4483:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:22:"transfer_to_occluded": "Two directly usable ideas for occluded ReID. (1) The pose loss L_p that explicitly minimizes cosine similarity between body-part features and non-human/background features — a cheap regularizer to suppress occluder/background leakage into the identity embedding, applicable to any pose- or part-based occluded model. (2) The texture-distinctiveness decoder channel aimed specifically at non-target PEDESTRIAN occlusion: when the occluder is itself a person, pose/structure cues fail (same skeleton topology), so using texture appearance distinctiveness (via pose-filtered queries cross-attending to contextual features) to push apart target vs distractor-person is a concrete mechanism for the under-addressed person-on-person occlusion case. Also transferable: replacing coarse pose-to-global mapping with a per-keypoint-to-patch confidence-gated argmax correspondence (Mahalanobis+cosine affine similarity) to localize visible regions.",
experiments/exp367_single_support/codex_train_design.md:3038:experiments/exp367_single_support/codex_review.md:1923:experiments/cargo_cvpb/litreview2/d17_eval.md:9581:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18000:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4651:./reassess/r_3.md:7995:./reassess/r_3.md:4483:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:22:"transfer_to_occluded": "Two directly usable ideas for occluded ReID. (1) The pose loss L_p that explicitly minimizes cosine similarity between body-part features and non-human/background features — a cheap regularizer to suppress occluder/background leakage into the identity embedding, applicable to any pose- or part-based occluded model. (2) The texture-distinctiveness decoder channel aimed specifically at non-target PEDESTRIAN occlusion: when the occluder is itself a person, pose/structure cues fail (same skeleton topology), so using texture appearance distinctiveness (via pose-filtered queries cross-attending to contextual features) to push apart target vs distractor-person is a concrete mechanism for the under-addressed person-on-person occlusion case. Also transferable: replacing coarse pose-to-global mapping with a per-keypoint-to-patch confidence-gated argmax correspondence (Mahalanobis+cosine affine similarity) to localize visible regions.",
experiments/exp367_single_support/codex_train_design.md:3039:experiments/exp367_single_support/codex_review.md:2074:experiments/cargo_cvpb/litreview2/d17_eval.md:9732:experiments/cargo_cvpb/litreview2/false_negative_audit.md:20396:| ③ Gallery Hubness remedy | Kill robust，但要分清：单说“被 k-reciprocal 占”略过强，因为训练成 single embedding 理论上还能争部署价值；但 §7.6 新控制更致命，`M(q)` 控 `#false-in-topk` 后偏相关≈0，说明 hubness 诊断本身无增量。k-reciprocal 本来就是 ReID 里标准、无监督、邻域拓扑 re-ranking。([arxiv.org](https://arxiv.org/abs/1701.08398?utm_source=openai)) Hubness 及 hub-aware retrieval 也不是新问题。([arxiv.org](https://arxiv.org/abs/2503.10526?utm_source=openai)) | 9/10 |
experiments/exp367_single_support/codex_train_design.md:3040:experiments/exp367_single_support/codex_review.md:2075:experiments/cargo_cvpb/litreview2/d17_eval.md:9733:experiments/cargo_cvpb/litreview2/false_negative_audit.md:20424:| ③ Gallery Hubness remedy | Kill robust，但要分清：单说“被 k-reciprocal 占”略过强，因为训练成 single embedding 理论上还能争部署价值；但 §7.6 新控制更致命，`M(q)` 控 `#false-in-topk` 后偏相关≈0，说明 hubness 诊断本身无增量。k-reciprocal 本来就是 ReID 里标准、无监督、邻域拓扑 re-ranking。([arxiv.org](https://arxiv.org/abs/1701.08398?utm_source=openai)) Hubness 及 hub-aware retrieval 也不是新问题。([arxiv.org](https://arxiv.org/abs/2503.10526?utm_source=openai)) | 9/10 |
experiments/exp367_single_support/codex_train_design.md:3041:experiments/exp367_single_support/codex_review.md:2103:experiments/cargo_cvpb/litreview2/explore20/clean/d_20.txt:4:**触发观察/失败**：标准 ReID 默认 query 是单张正确目标，multi-query 默认多张全是同一 ID。真实部署里更常见的是“用户/检索员给一包候选截图”，其中可能混入跟踪漂移、检测错框、相邻行人、错误截图。大家以为“query support 越多越稳”，其实隐藏变量可能是 **query-bag purity**：多给错图会比单图更坏。
experiments/exp367_single_support/codex_train_design.md:3042:experiments/exp367_single_support/codex_review.md:2104:experiments/cargo_cvpb/litreview2/explore20/clean/d_2.txt:21:触发观察：CC-ReID 统一算 mAP/Rank-1，但不是每张换衣 query 都有足够身份信息。正面脸、清晰头部、稳定肢体、鞋包 carry-over、多视角 support 都会极大改变可匹配性。很多失败不是模型不够强，而是单图证据不足。
experiments/exp367_single_support/codex_train_design.md:3043:experiments/exp367_single_support/codex_review.md:2105:experiments/cargo_cvpb/litreview2/explore20/clean/d_2.txt:25:机制草案：`Biometric Evidence Routing`。为 query-gallery 对构建 cue support：face/head visible、limb visible、body contour quality、artifact volatility、same-clothes support、gallery support density。高证据走对应 cue expert；低证据样本输出低置信或请求 set-query/tracklet/人工复核。训练端可用 evidence-conditioned contrastive：只在共同可见且可靠 cue 上拉近，不对低证据样本过度监督。
experiments/exp367_single_support/codex_train_design.md:3044:experiments/exp367_single_support/codex_review.md:2107:experiments/cargo_cvpb/litreview2/explore20/clean/d_10.txt:5:- 重定义：把 long-tail 从 `identity frequency` 改成 `identity-camera-pair edge / camera-transition support`。大家以为长尾是类频次，其实在 ReID 里关键是“某些跨摄像头转换从训练到测试都没被充分覆盖”。
experiments/exp367_single_support/codex_train_design.md:3045:experiments/exp367_single_support/codex_review.md:2108:experiments/cargo_cvpb/litreview2/explore20/clean/d_10.txt:6:- 机制草案：构建 camera-pair support graph；batch/loss 按低支持 camera-pair 采样和加权；约束不同 camera-pair support bin 的 positive margin 分布一致。不要讲 camera-bias normalization，要讲“跨摄像头证据覆盖不均”。
experiments/exp367_single_support/codex_train_design.md:3046:experiments/exp367_single_support/codex_review.md:2109:experiments/cargo_cvpb/litreview2/explore20/clean/d_10.txt:7:- 零训练 kill-switch：用 frozen SOLIDER/exp030a 特征，按 query-camera 到 gallery-camera 的 train support 分桶，算 pair-conditioned mAP/R1；控制 positive 数量、gallery size、occlusion 后，tail-pair 比 head-pair 仍低 5-8 mAP 才继续。再跑一次 2025 camera-specific feature normalization 后 gap 仍在，才说明不是普通 camera bias。([arxiv.org](https://arxiv.org/abs/2502.10195))
experiments/exp367_single_support/codex_train_design.md:3047:experiments/exp367_single_support/codex_review.md:2110:experiments/cargo_cvpb/litreview2/explore20/clean/d_10.txt:8:- 撞车核查：APRA 已做 camera performance imbalance 和指标；2025 camera bias/DART³ 已做 test-time camera debias；3C 2024 用 camera information entropy 做无监督聚类置信度。撞车风险中高，但还没看到“camera-pair support long-tail”作为 supervised ReID 主问题。([arxiv.org](https://arxiv.org/abs/2207.01204)) ([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2408.09464))
experiments/exp367_single_support/codex_train_design.md:3048:experiments/exp367_single_support/codex_review.md:2111:experiments/cargo_cvpb/litreview2/explore20/clean/d_10.txt:13:- 重定义：long-tail ReID = open-world gallery 中的“尾部身份保全”问题。大家以为 open-set 是阈值问题，其实隐藏变量是 prototype support count：head prototype 越大，越容易吸走未知 tail。
experiments/exp367_single_support/codex_train_design.md:3049:experiments/exp367_single_support/codex_review.md:2112:experiments/cargo_cvpb/litreview2/explore20/clean/d_10.txt:15:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_train_design.md:3050:experiments/exp367_single_support/codex_review.md:2113:experiments/cargo_cvpb/litreview2/explore20/clean/d_10.txt:16:- 撞车核查：CFReID 2025 是 continual few-shot domain adaptation；LReID backward-compatibility 2024 关注旧 gallery 特征兼容；MICRO-TRACK 2024 做 open-set industrial tracking 和 centroid gallery，但没有把 Zipf singleton false-consolidation 定义成 long-tail 主问题。([arxiv.org](https://arxiv.org/abs/2503.18469)) ([arxiv.org](https://arxiv.org/abs/2403.10022)) ([arxiv.org](https://arxiv.org/abs/2409.03879))
experiments/exp367_single_support/codex_train_design.md:3051:experiments/exp367_single_support/codex_review.md:2114:experiments/cargo_cvpb/litreview2/explore20/d_9.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:3052:experiments/exp367_single_support/codex_review.md:2119:experiments/cargo_cvpb/litreview2/explore20/d_10.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:3053:experiments/exp367_single_support/codex_review.md:2120:experiments/cargo_cvpb/litreview2/explore20/d_10.md:86:- 重定义：把 long-tail 从 `identity frequency` 改成 `identity-camera-pair edge / camera-transition support`。大家以为长尾是类频次，其实在 ReID 里关键是“某些跨摄像头转换从训练到测试都没被充分覆盖”。
experiments/exp367_single_support/codex_train_design.md:3054:experiments/exp367_single_support/codex_review.md:2121:experiments/cargo_cvpb/litreview2/explore20/d_10.md:87:- 机制草案：构建 camera-pair support graph；batch/loss 按低支持 camera-pair 采样和加权；约束不同 camera-pair support bin 的 positive margin 分布一致。不要讲 camera-bias normalization，要讲“跨摄像头证据覆盖不均”。
experiments/exp367_single_support/codex_train_design.md:3055:experiments/exp367_single_support/codex_review.md:2122:experiments/cargo_cvpb/litreview2/explore20/d_10.md:88:- 零训练 kill-switch：用 frozen SOLIDER/exp030a 特征，按 query-camera 到 gallery-camera 的 train support 分桶，算 pair-conditioned mAP/R1；控制 positive 数量、gallery size、occlusion 后，tail-pair 比 head-pair 仍低 5-8 mAP 才继续。再跑一次 2025 camera-specific feature normalization 后 gap 仍在，才说明不是普通 camera bias。([arxiv.org](https://arxiv.org/abs/2502.10195))
experiments/exp367_single_support/codex_train_design.md:3056:experiments/exp367_single_support/codex_review.md:2123:experiments/cargo_cvpb/litreview2/explore20/d_10.md:89:- 撞车核查：APRA 已做 camera performance imbalance 和指标；2025 camera bias/DART³ 已做 test-time camera debias；3C 2024 用 camera information entropy 做无监督聚类置信度。撞车风险中高，但还没看到“camera-pair support long-tail”作为 supervised ReID 主问题。([arxiv.org](https://arxiv.org/abs/2207.01204)) ([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2408.09464))
experiments/exp367_single_support/codex_train_design.md:3057:experiments/exp367_single_support/codex_review.md:2124:experiments/cargo_cvpb/litreview2/explore20/d_10.md:94:- 重定义：long-tail ReID = open-world gallery 中的“尾部身份保全”问题。大家以为 open-set 是阈值问题，其实隐藏变量是 prototype support count：head prototype 越大，越容易吸走未知 tail。
experiments/exp367_single_support/codex_train_design.md:3058:experiments/exp367_single_support/codex_review.md:2125:experiments/cargo_cvpb/litreview2/explore20/d_10.md:96:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_train_design.md:3059:experiments/exp367_single_support/codex_review.md:2126:experiments/cargo_cvpb/litreview2/explore20/d_10.md:97:- 撞车核查：CFReID 2025 是 continual few-shot domain adaptation；LReID backward-compatibility 2024 关注旧 gallery 特征兼容；MICRO-TRACK 2024 做 open-set industrial tracking 和 centroid gallery，但没有把 Zipf singleton false-consolidation 定义成 long-tail 主问题。([arxiv.org](https://arxiv.org/abs/2503.18469)) ([arxiv.org](https://arxiv.org/abs/2403.10022)) ([arxiv.org](https://arxiv.org/abs/2409.03879))
experiments/exp367_single_support/codex_train_design.md:3060:experiments/exp367_single_support/codex_review.md:2127:experiments/cargo_cvpb/litreview2/explore20/d_10.md:105:- 重定义：把 long-tail 从 `identity frequency` 改成 `identity-camera-pair edge / camera-transition support`。大家以为长尾是类频次，其实在 ReID 里关键是“某些跨摄像头转换从训练到测试都没被充分覆盖”。
experiments/exp367_single_support/codex_train_design.md:3061:experiments/exp367_single_support/codex_review.md:2128:experiments/cargo_cvpb/litreview2/explore20/d_10.md:106:- 机制草案：构建 camera-pair support graph；batch/loss 按低支持 camera-pair 采样和加权；约束不同 camera-pair support bin 的 positive margin 分布一致。不要讲 camera-bias normalization，要讲“跨摄像头证据覆盖不均”。
experiments/exp367_single_support/codex_train_design.md:3062:experiments/exp367_single_support/codex_review.md:2129:experiments/cargo_cvpb/litreview2/explore20/d_10.md:107:- 零训练 kill-switch：用 frozen SOLIDER/exp030a 特征，按 query-camera 到 gallery-camera 的 train support 分桶，算 pair-conditioned mAP/R1；控制 positive 数量、gallery size、occlusion 后，tail-pair 比 head-pair 仍低 5-8 mAP 才继续。再跑一次 2025 camera-specific feature normalization 后 gap 仍在，才说明不是普通 camera bias。([arxiv.org](https://arxiv.org/abs/2502.10195))
experiments/exp367_single_support/codex_train_design.md:3063:experiments/exp367_single_support/codex_review.md:2130:experiments/cargo_cvpb/litreview2/explore20/d_10.md:108:- 撞车核查：APRA 已做 camera performance imbalance 和指标；2025 camera bias/DART³ 已做 test-time camera debias；3C 2024 用 camera information entropy 做无监督聚类置信度。撞车风险中高，但还没看到“camera-pair support long-tail”作为 supervised ReID 主问题。([arxiv.org](https://arxiv.org/abs/2207.01204)) ([arxiv.org](https://arxiv.org/abs/2505.18337)) ([arxiv.org](https://arxiv.org/abs/2408.09464))
experiments/exp367_single_support/codex_train_design.md:3064:experiments/exp367_single_support/codex_review.md:2131:experiments/cargo_cvpb/litreview2/explore20/d_10.md:113:- 重定义：long-tail ReID = open-world gallery 中的“尾部身份保全”问题。大家以为 open-set 是阈值问题，其实隐藏变量是 prototype support count：head prototype 越大，越容易吸走未知 tail。
experiments/exp367_single_support/codex_train_design.md:3065:experiments/exp367_single_support/codex_review.md:2132:experiments/cargo_cvpb/litreview2/explore20/d_10.md:115:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_train_design.md:3066:experiments/exp367_single_support/codex_review.md:2133:experiments/cargo_cvpb/litreview2/explore20/d_10.md:116:- 撞车核查：CFReID 2025 是 continual few-shot domain adaptation；LReID backward-compatibility 2024 关注旧 gallery 特征兼容；MICRO-TRACK 2024 做 open-set industrial tracking 和 centroid gallery，但没有把 Zipf singleton false-consolidation 定义成 long-tail 主问题。([arxiv.org](https://arxiv.org/abs/2503.18469)) ([arxiv.org](https://arxiv.org/abs/2403.10022)) ([arxiv.org](https://arxiv.org/abs/2409.03879))
experiments/exp367_single_support/codex_train_design.md:3067:experiments/exp367_single_support/codex_review.md:2134:experiments/cargo_cvpb/litreview2/explore20/clean/d_19.txt:18:2. 做 oracle evidence gate：把低 evidence-overlap 的 pair/query 降权或拒识，看 false top-k 是否明显下降。  
experiments/exp367_single_support/codex_train_design.md:3068:experiments/exp367_single_support/codex_review.md:2135:experiments/cargo_cvpb/litreview2/explore20/d_20.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:3069:experiments/exp367_single_support/codex_review.md:2136:experiments/cargo_cvpb/litreview2/explore20/d_20.md:101:../../exp109/monitor.md:22:- 结果文件: `log/occluded_duke/exp109_oracle_support_bank_exp030a/summary.json`
experiments/exp367_single_support/codex_train_design.md:3070:experiments/exp367_single_support/codex_review.md:2137:experiments/cargo_cvpb/litreview2/explore20/d_20.md:262:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3071:experiments/exp367_single_support/codex_review.md:2138:experiments/cargo_cvpb/litreview2/explore20/d_20.md:299:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3072:experiments/exp367_single_support/codex_review.md:2139:experiments/cargo_cvpb/litreview2/explore20/d_20.md:490:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/exp367_single_support/codex_train_design.md:3073:experiments/exp367_single_support/codex_review.md:2140:experiments/cargo_cvpb/litreview2/explore20/d_20.md:557:   - 写法：test-time supporting evaluations
experiments/exp367_single_support/codex_train_design.md:3074:experiments/exp367_single_support/codex_review.md:2141:experiments/cargo_cvpb/litreview2/explore20/d_20.md:979:**触发观察/失败**：标准 ReID 默认 query 是单张正确目标，multi-query 默认多张全是同一 ID。真实部署里更常见的是“用户/检索员给一包候选截图”，其中可能混入跟踪漂移、检测错框、相邻行人、错误截图。大家以为“query support 越多越稳”，其实隐藏变量可能是 **query-bag purity**：多给错图会比单图更坏。
experiments/exp367_single_support/codex_train_design.md:3075:experiments/exp367_single_support/codex_review.md:2142:experiments/cargo_cvpb/litreview2/explore20/d_20.md:1017:**触发观察/失败**：标准 ReID 默认 query 是单张正确目标，multi-query 默认多张全是同一 ID。真实部署里更常见的是“用户/检索员给一包候选截图”，其中可能混入跟踪漂移、检测错框、相邻行人、错误截图。大家以为“query support 越多越稳”，其实隐藏变量可能是 **query-bag purity**：多给错图会比单图更坏。
experiments/exp367_single_support/codex_train_design.md:3076:experiments/exp367_single_support/codex_review.md:2146:experiments/cargo_cvpb/litreview2/explore20/d_1.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:3077:experiments/exp367_single_support/codex_review.md:2149:experiments/cargo_cvpb/litreview2/explore20/d_4.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:3078:experiments/exp367_single_support/codex_review.md:2156:experiments/cargo_cvpb/litreview2/explore20/d_3.md:14:一个 ReID 团队找 CCF-B 方法稿。post-PRCV 的 frozen-image+on-disk 路已穷尽证负: 6 方向 cheap-kill 全死(B航拍不确定性/GOPL-SMPL可靠性/Gallery-Hubness/OSAC谱坍缩/RMA文本锚定/Rank-Regret效率)+视频no-go+Hubness诊断被 trivial 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:3079:experiments/exp367_single_support/codex_review.md:2159:experiments/exp111/design.md:5:`exp110` 证明了训练端的 `support-complete prototype distillation` 能在单 seed 上转正，但增益仍然偏弱。当前实现里 `POSE_SCKD_MIN_COUNT=1`，意味着某个 identity 的某个 keypoint 只要出现过一次 high-visibility 观测，就会被当作 teacher 使用。这种 teacher 过于宽松，容易把偶然噪声、姿态抽取误差、遮挡下的不稳定局部都写入 bank。
experiments/exp367_single_support/codex_train_design.md:3080:experiments/exp367_single_support/codex_review.md:2160:experiments/exp111/design.md:7:用户也明确强调了当前阶段应优先探索“真正有效且足够支撑论文的创新点”，而不是提前做多 seed 收尾。因此下一步最合理的单变量推进，不是重复验证，而是把 `support-complete` 主线里的关键机制继续做实。
experiments/exp367_single_support/codex_train_design.md:3081:experiments/exp367_single_support/codex_review.md:2161:experiments/exp111/design.md:50:2. 若结果变差，不代表 `support-complete` 主线错误，更可能说明：
experiments/exp367_single_support/codex_train_design.md:3082:experiments/exp367_single_support/codex_review.md:2162:experiments/exp111/monitor.md:15:  3. `MIN_COUNT=1` 过于宽松，不足以体现真正的 multi-view support
experiments/exp367_single_support/codex_train_design.md:3083:experiments/exp367_single_support/codex_review.md:2163:experiments/exp111/monitor.md:60:  3. 这说明“support 数量门槛”不是当前最主要的增益来源
experiments/exp367_single_support/codex_train_design.md:3084:experiments/exp367_single_support/codex_review.md:2164:experiments/exp111/monitor.md:77:  2. 相对基线 `exp030a-eq seed1234`，仍保持 **R1 +0.9**，说明 `support-complete` 主线本身没有被破坏
experiments/exp367_single_support/codex_train_design.md:3085:experiments/exp367_single_support/codex_review.md:2165:experiments/exp111/monitor.md:78:  3. `MIN_COUNT=4` 没有把当前弱正向显著放大，因此“teacher reliability”里更关键的可能不是 support 数量，而是 support 纯度
experiments/exp367_single_support/codex_train_design.md:3086:experiments/exp367_single_support/codex_review.md:2166:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:41:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3087:experiments/exp367_single_support/codex_review.md:2167:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:78:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3088:experiments/exp367_single_support/codex_review.md:2168:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:489:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/exp367_single_support/codex_train_design.md:3089:experiments/exp367_single_support/codex_review.md:2169:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:556:   - 写法：test-time supporting evaluations
experiments/exp367_single_support/codex_train_design.md:3090:experiments/exp367_single_support/codex_review.md:2170:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:1616:experiments/exp148/design.md:17:1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
experiments/exp367_single_support/codex_train_design.md:3091:experiments/exp367_single_support/codex_review.md:2171:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:1641:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3092:experiments/exp367_single_support/codex_review.md:2172:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:1746:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3093:experiments/exp367_single_support/codex_review.md:2173:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:1747:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3094:experiments/exp367_single_support/codex_review.md:2174:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2192:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3095:experiments/exp367_single_support/codex_review.md:2175:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2193:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3096:experiments/exp367_single_support/codex_review.md:2176:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2809:experiments/cargo_cvpb/codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3097:experiments/exp367_single_support/codex_review.md:2177:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2810:experiments/cargo_cvpb/codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3098:experiments/exp367_single_support/codex_review.md:2178:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2844:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3099:experiments/exp367_single_support/codex_review.md:2179:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2845:experiments/cargo_cvpb/codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3100:experiments/exp367_single_support/codex_review.md:2180:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2848:experiments/cargo_cvpb/codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3101:experiments/exp367_single_support/codex_review.md:2181:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2862:experiments/cargo_cvpb/codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3102:experiments/exp367_single_support/codex_review.md:2182:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2863:experiments/cargo_cvpb/codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3103:experiments/exp367_single_support/codex_review.md:2183:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2885:experiments/cargo_cvpb/codex_review_ovli.txt:4112:experiments/paper_notes/2026-04-15_prcv_reset.md:10:4. `GCN` 虽然也属于 pose 信息利用，但应统一写成 **structural pose branch**；`LGPA-D / OA-SD / MaxSim / POT / flip-test` 仍作为 supporting assets，不再抢主创新位置
experiments/exp367_single_support/codex_train_design.md:3104:experiments/exp367_single_support/codex_review.md:2184:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2887:experiments/cargo_cvpb/codex_review_ovli.txt:4115:experiments/paper_notes/2026-04-15_prcv_reset.md:201:- `LGPA-D / GCN / OA-SD / MaxSim` = system assets / supporting modules
experiments/exp367_single_support/codex_train_design.md:3105:experiments/exp367_single_support/codex_review.md:2185:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3208:experiments/cargo_cvpb/litreview2/meta_converge.md:545:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3106:experiments/exp367_single_support/codex_review.md:2186:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3209:experiments/cargo_cvpb/litreview2/meta_converge.md:582:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3107:experiments/exp367_single_support/codex_review.md:2187:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3343:experiments/cargo_cvpb/litreview2/meta_converge.md:2067:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3108:experiments/exp367_single_support/codex_review.md:2188:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3387:experiments/cargo_cvpb/litreview2/meta_converge.md:2820:experiments/exp148/design.md:17:1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
experiments/exp367_single_support/codex_train_design.md:3109:experiments/exp367_single_support/codex_review.md:2189:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3394:experiments/cargo_cvpb/litreview2/meta_converge.md:3286:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3110:experiments/exp367_single_support/codex_review.md:2190:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3454:experiments/cargo_cvpb/litreview2/d17_eval.md:41:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3111:experiments/exp367_single_support/codex_review.md:2191:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3455:experiments/cargo_cvpb/litreview2/d17_eval.md:78:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3112:experiments/exp367_single_support/codex_review.md:2192:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3572:experiments/cargo_cvpb/litreview2/d17_eval.md:1350:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3113:experiments/exp367_single_support/codex_review.md:2193:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3580:experiments/cargo_cvpb/litreview2/d17_eval.md:1684:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3114:experiments/exp367_single_support/codex_review.md:2194:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3584:experiments/cargo_cvpb/litreview2/d17_eval.md:3591:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:2526:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3115:experiments/exp367_single_support/codex_review.md:2195:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3588:experiments/cargo_cvpb/litreview2/d17_eval.md:5425:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3116:experiments/exp367_single_support/codex_review.md:2196:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3589:experiments/cargo_cvpb/litreview2/d17_eval.md:5462:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3117:experiments/exp367_single_support/codex_review.md:2197:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3590:experiments/cargo_cvpb/litreview2/d17_eval.md:5733:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3118:experiments/exp367_single_support/codex_review.md:2198:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3591:experiments/cargo_cvpb/litreview2/d17_eval.md:5770:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3119:experiments/exp367_single_support/codex_review.md:2199:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3741:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:615:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3120:experiments/exp367_single_support/codex_review.md:2200:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3742:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:652:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3121:experiments/exp367_single_support/codex_review.md:2201:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3881:experiments/cargo_cvpb/litreview2/train3_paperstrategy.md:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3122:experiments/exp367_single_support/codex_review.md:2202:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3882:experiments/cargo_cvpb/litreview2/train3_paperstrategy.md:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3123:experiments/exp367_single_support/codex_review.md:2203:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4004:experiments/cargo_cvpb/litreview2/reassess2/x_2.md:312:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3124:experiments/exp367_single_support/codex_review.md:2204:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4005:experiments/cargo_cvpb/litreview2/reassess2/x_2.md:349:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3125:experiments/exp367_single_support/codex_review.md:2205:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4358:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:54:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3126:experiments/exp367_single_support/codex_review.md:2206:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4359:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:91:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3127:experiments/exp367_single_support/codex_review.md:2207:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4474:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:3237:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:5572:./pivot/p_3.md:1202:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3128:experiments/exp367_single_support/codex_review.md:2208:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4475:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:3238:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:5574:./pivot/p_3.md:1239:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3129:experiments/exp367_single_support/codex_review.md:2209:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4631:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4104:experiments/cargo_cvpb/litreview2/reassess/r_3.md:1538:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3130:experiments/exp367_single_support/codex_review.md:2210:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4632:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4105:experiments/cargo_cvpb/litreview2/reassess/r_3.md:1575:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3131:experiments/exp367_single_support/codex_review.md:2211:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4771:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4591:experiments/cargo_cvpb/litreview2/explore20/d_5.md:711:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3132:experiments/exp367_single_support/codex_review.md:2212:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4772:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4592:experiments/cargo_cvpb/litreview2/explore20/d_5.md:748:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3133:experiments/exp367_single_support/codex_review.md:2233:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4796:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:5086:experiments/cargo_cvpb/litreview2/explore20/d_20.md:262:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3134:experiments/exp367_single_support/codex_review.md:2234:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4797:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:5087:experiments/cargo_cvpb/litreview2/explore20/d_20.md:299:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3135:experiments/exp367_single_support/codex_review.md:2235:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4840:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:5225:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3136:experiments/exp367_single_support/codex_review.md:2236:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4841:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:5262:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3137:experiments/exp367_single_support/codex_review.md:2237:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4922:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:5934:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3138:experiments/exp367_single_support/codex_review.md:2238:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4939:experiments/cargo_cvpb/litreview2/train_lens2_uncertainty.md:38:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3139:experiments/exp367_single_support/codex_review.md:2239:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4940:experiments/cargo_cvpb/litreview2/train_lens2_uncertainty.md:75:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3140:experiments/exp367_single_support/codex_review.md:2240:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5144:experiments/cargo_cvpb/litreview2/pivot/p_2.md:125:./pivot/p_3.md:1202:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3141:experiments/exp367_single_support/codex_review.md:2241:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5145:experiments/cargo_cvpb/litreview2/pivot/p_2.md:127:./pivot/p_3.md:1239:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3142:experiments/exp367_single_support/codex_review.md:2242:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5149:experiments/cargo_cvpb/litreview2/pivot/p_2.md:2591:./validate/v_3.md:14434:../litreview2/validate/v_3.md:6625:./validate/v_3.md:1954:../codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3143:experiments/exp367_single_support/codex_review.md:2243:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5150:experiments/cargo_cvpb/litreview2/pivot/p_2.md:2593:./validate/v_3.md:14436:../litreview2/validate/v_3.md:6627:./validate/v_3.md:1956:../codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3144:experiments/exp367_single_support/codex_review.md:2244:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5166:experiments/cargo_cvpb/litreview2/pivot/p_2.md:2780:./validate/v_3.md:14623:../litreview2/validate/v_3.md:6823:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3145:experiments/exp367_single_support/codex_review.md:2245:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5167:experiments/cargo_cvpb/litreview2/pivot/p_2.md:2781:./validate/v_3.md:14624:../litreview2/validate/v_3.md:6824:./validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3146:experiments/exp367_single_support/codex_review.md:2246:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5170:experiments/cargo_cvpb/litreview2/pivot/p_2.md:2832:./validate/v_3.md:14675:../litreview2/validate/v_3.md:6875:./validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3147:experiments/exp367_single_support/codex_review.md:2247:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5184:experiments/cargo_cvpb/litreview2/pivot/p_2.md:2979:./validate/v_3.md:14822:../litreview2/validate/v_3.md:7022:./validate/v_3.md:2404:../codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3148:experiments/exp367_single_support/codex_review.md:2248:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5185:experiments/cargo_cvpb/litreview2/pivot/p_2.md:2981:./validate/v_3.md:14824:../litreview2/validate/v_3.md:7024:./validate/v_3.md:2406:../codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3149:experiments/exp367_single_support/codex_review.md:2249:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5203:experiments/cargo_cvpb/litreview2/pivot/p_2.md:3601:./validate/v_3.md:15448:../litreview2/validate/v_3.md:8004:./validate/v_2.md:12678:validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3150:experiments/exp367_single_support/codex_review.md:2250:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5204:experiments/cargo_cvpb/litreview2/pivot/p_2.md:3605:./validate/v_3.md:15452:../litreview2/validate/v_3.md:8008:./validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3151:experiments/exp367_single_support/codex_review.md:2251:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5205:experiments/cargo_cvpb/litreview2/pivot/p_2.md:3909:./validate/v_3.md:16052:../litreview2/validate/v_2.md:12678:validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3152:experiments/exp367_single_support/codex_review.md:2252:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5206:experiments/cargo_cvpb/litreview2/pivot/p_2.md:3913:./validate/v_3.md:16056:../litreview2/validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3153:experiments/exp367_single_support/codex_review.md:2253:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5239:experiments/cargo_cvpb/litreview2/pivot/p_2.md:4599:./pivot/p_3.md:1202:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3154:experiments/exp367_single_support/codex_review.md:2254:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5240:experiments/cargo_cvpb/litreview2/pivot/p_2.md:4601:./pivot/p_3.md:1239:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3155:experiments/exp367_single_support/codex_review.md:2255:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5290:experiments/cargo_cvpb/litreview2/pivot/p_2.md:5098:pivot/p_3.md:1202:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3156:experiments/exp367_single_support/codex_review.md:2256:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5291:experiments/cargo_cvpb/litreview2/pivot/p_2.md:5099:pivot/p_3.md:1239:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3157:experiments/exp367_single_support/codex_review.md:2257:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5293:experiments/cargo_cvpb/litreview2/pivot/p_2.md:5646:pivot/p_2.md:125:./pivot/p_3.md:1202:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3158:experiments/exp367_single_support/codex_review.md:2258:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5294:experiments/cargo_cvpb/litreview2/pivot/p_2.md:5647:pivot/p_2.md:127:./pivot/p_3.md:1239:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3159:experiments/exp367_single_support/codex_review.md:2259:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5296:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6025:pivot/p_2.md:2591:./validate/v_3.md:14434:../litreview2/validate/v_3.md:6625:./validate/v_3.md:1954:../codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3160:experiments/exp367_single_support/codex_review.md:2260:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5297:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6026:pivot/p_2.md:2593:./validate/v_3.md:14436:../litreview2/validate/v_3.md:6627:./validate/v_3.md:1956:../codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3161:experiments/exp367_single_support/codex_review.md:2261:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5305:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6036:pivot/p_2.md:2780:./validate/v_3.md:14623:../litreview2/validate/v_3.md:6823:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3162:experiments/exp367_single_support/codex_review.md:2262:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5306:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6037:pivot/p_2.md:2781:./validate/v_3.md:14624:../litreview2/validate/v_3.md:6824:./validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3163:experiments/exp367_single_support/codex_review.md:2263:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5309:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6041:pivot/p_2.md:2832:./validate/v_3.md:14675:../litreview2/validate/v_3.md:6875:./validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3164:experiments/exp367_single_support/codex_review.md:2264:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5323:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6063:pivot/p_2.md:2979:./validate/v_3.md:14822:../litreview2/validate/v_3.md:7022:./validate/v_3.md:2404:../codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3165:experiments/exp367_single_support/codex_review.md:2265:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5324:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6064:pivot/p_2.md:2981:./validate/v_3.md:14824:../litreview2/validate/v_3.md:7024:./validate/v_3.md:2406:../codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3166:experiments/exp367_single_support/codex_review.md:2266:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5341:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6137:pivot/p_2.md:3601:./validate/v_3.md:15448:../litreview2/validate/v_3.md:8004:./validate/v_2.md:12678:validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3167:experiments/exp367_single_support/codex_review.md:2267:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5342:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6138:pivot/p_2.md:3605:./validate/v_3.md:15452:../litreview2/validate/v_3.md:8008:./validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3168:experiments/exp367_single_support/codex_review.md:2268:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5343:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6190:pivot/p_2.md:3909:./validate/v_3.md:16052:../litreview2/validate/v_2.md:12678:validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3169:experiments/exp367_single_support/codex_review.md:2269:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5344:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6191:pivot/p_2.md:3913:./validate/v_3.md:16056:../litreview2/validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3170:experiments/exp367_single_support/codex_review.md:2270:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5373:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6305:pivot/p_2.md:4599:./pivot/p_3.md:1202:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3171:experiments/exp367_single_support/codex_review.md:2271:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5374:experiments/cargo_cvpb/litreview2/pivot/p_2.md:6306:pivot/p_2.md:4601:./pivot/p_3.md:1239:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3172:experiments/exp367_single_support/codex_review.md:2272:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5434:experiments/cargo_cvpb/litreview2/validate/v_3.md:1471:../codex_review_raw.txt:485:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3173:experiments/exp367_single_support/codex_review.md:2273:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5435:experiments/cargo_cvpb/litreview2/validate/v_3.md:1473:../codex_review_raw.txt:522:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3174:experiments/exp367_single_support/codex_review.md:2274:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5441:experiments/cargo_cvpb/litreview2/validate/v_3.md:1954:../codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3175:experiments/exp367_single_support/codex_review.md:2275:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5442:experiments/cargo_cvpb/litreview2/validate/v_3.md:1956:../codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3176:experiments/exp367_single_support/codex_review.md:2276:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5459:experiments/cargo_cvpb/litreview2/validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3177:experiments/exp367_single_support/codex_review.md:2277:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5460:experiments/cargo_cvpb/litreview2/validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3178:experiments/exp367_single_support/codex_review.md:2278:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5463:experiments/cargo_cvpb/litreview2/validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3179:experiments/exp367_single_support/codex_review.md:2279:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5477:experiments/cargo_cvpb/litreview2/validate/v_3.md:2404:../codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3180:experiments/exp367_single_support/codex_review.md:2280:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5478:experiments/cargo_cvpb/litreview2/validate/v_3.md:2406:../codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3181:experiments/exp367_single_support/codex_review.md:2281:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5499:experiments/cargo_cvpb/litreview2/validate/v_3.md:6527:./validate/v_3.md:1471:../codex_review_raw.txt:485:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3182:experiments/exp367_single_support/codex_review.md:2282:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5500:experiments/cargo_cvpb/litreview2/validate/v_3.md:6529:./validate/v_3.md:1473:../codex_review_raw.txt:522:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3183:experiments/exp367_single_support/codex_review.md:2283:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5504:experiments/cargo_cvpb/litreview2/validate/v_3.md:6625:./validate/v_3.md:1954:../codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3184:experiments/exp367_single_support/codex_review.md:2284:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5505:experiments/cargo_cvpb/litreview2/validate/v_3.md:6627:./validate/v_3.md:1956:../codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3185:experiments/exp367_single_support/codex_review.md:2285:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5521:experiments/cargo_cvpb/litreview2/validate/v_3.md:6823:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3186:experiments/exp367_single_support/codex_review.md:2286:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5522:experiments/cargo_cvpb/litreview2/validate/v_3.md:6824:./validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3187:experiments/exp367_single_support/codex_review.md:2287:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5525:experiments/cargo_cvpb/litreview2/validate/v_3.md:6875:./validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3188:experiments/exp367_single_support/codex_review.md:2288:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5539:experiments/cargo_cvpb/litreview2/validate/v_3.md:7022:./validate/v_3.md:2404:../codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3189:experiments/exp367_single_support/codex_review.md:2289:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5540:experiments/cargo_cvpb/litreview2/validate/v_3.md:7024:./validate/v_3.md:2406:../codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3190:experiments/exp367_single_support/codex_review.md:2290:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5558:experiments/cargo_cvpb/litreview2/validate/v_3.md:8004:./validate/v_2.md:12678:validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3191:experiments/exp367_single_support/codex_review.md:2291:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5559:experiments/cargo_cvpb/litreview2/validate/v_3.md:8008:./validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3192:experiments/exp367_single_support/codex_review.md:2292:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5569:experiments/cargo_cvpb/litreview2/validate/v_3.md:13361:./validate/v_3.md:1471:../codex_review_raw.txt:485:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3193:experiments/exp367_single_support/codex_review.md:2293:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5570:experiments/cargo_cvpb/litreview2/validate/v_3.md:13363:./validate/v_3.md:1473:../codex_review_raw.txt:522:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3194:experiments/exp367_single_support/codex_review.md:2294:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5574:experiments/cargo_cvpb/litreview2/validate/v_3.md:13449:./validate/v_3.md:1954:../codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3195:experiments/exp367_single_support/codex_review.md:2295:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5575:experiments/cargo_cvpb/litreview2/validate/v_3.md:13451:./validate/v_3.md:1956:../codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3196:experiments/exp367_single_support/codex_review.md:2296:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5591:experiments/cargo_cvpb/litreview2/validate/v_3.md:13639:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3197:experiments/exp367_single_support/codex_review.md:2297:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5592:experiments/cargo_cvpb/litreview2/validate/v_3.md:13640:./validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3198:experiments/exp367_single_support/codex_review.md:2298:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5595:experiments/cargo_cvpb/litreview2/validate/v_3.md:13691:./validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3199:experiments/exp367_single_support/codex_review.md:2299:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5609:experiments/cargo_cvpb/litreview2/validate/v_3.md:13838:./validate/v_3.md:2404:../codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3200:experiments/exp367_single_support/codex_review.md:2300:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5610:experiments/cargo_cvpb/litreview2/validate/v_3.md:13840:./validate/v_3.md:2406:../codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3201:experiments/exp367_single_support/codex_review.md:2301:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5631:experiments/cargo_cvpb/litreview2/validate/v_3.md:14434:../litreview2/validate/v_3.md:6625:./validate/v_3.md:1954:../codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3202:experiments/exp367_single_support/codex_review.md:2302:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5632:experiments/cargo_cvpb/litreview2/validate/v_3.md:14436:../litreview2/validate/v_3.md:6627:./validate/v_3.md:1956:../codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3203:experiments/exp367_single_support/codex_review.md:2303:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5648:experiments/cargo_cvpb/litreview2/validate/v_3.md:14623:../litreview2/validate/v_3.md:6823:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3204:experiments/exp367_single_support/codex_review.md:2304:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5649:experiments/cargo_cvpb/litreview2/validate/v_3.md:14624:../litreview2/validate/v_3.md:6824:./validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3205:experiments/exp367_single_support/codex_review.md:2305:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5652:experiments/cargo_cvpb/litreview2/validate/v_3.md:14675:../litreview2/validate/v_3.md:6875:./validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3206:experiments/exp367_single_support/codex_review.md:2306:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5666:experiments/cargo_cvpb/litreview2/validate/v_3.md:14822:../litreview2/validate/v_3.md:7022:./validate/v_3.md:2404:../codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3207:experiments/exp367_single_support/codex_review.md:2307:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5667:experiments/cargo_cvpb/litreview2/validate/v_3.md:14824:../litreview2/validate/v_3.md:7024:./validate/v_3.md:2406:../codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3208:experiments/exp367_single_support/codex_review.md:2308:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5685:experiments/cargo_cvpb/litreview2/validate/v_3.md:15448:../litreview2/validate/v_3.md:8004:./validate/v_2.md:12678:validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3209:experiments/exp367_single_support/codex_review.md:2309:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5686:experiments/cargo_cvpb/litreview2/validate/v_3.md:15452:../litreview2/validate/v_3.md:8008:./validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3210:experiments/exp367_single_support/codex_review.md:2310:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5687:experiments/cargo_cvpb/litreview2/validate/v_3.md:16052:../litreview2/validate/v_2.md:12678:validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3211:experiments/exp367_single_support/codex_review.md:2311:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5688:experiments/cargo_cvpb/litreview2/validate/v_3.md:16056:../litreview2/validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3212:experiments/exp367_single_support/codex_review.md:2312:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5710:experiments/paper_notes/2026-04-15_prcv_reset.md:10:4. `GCN` 虽然也属于 pose 信息利用，但应统一写成 **structural pose branch**；`LGPA-D / OA-SD / MaxSim / POT / flip-test` 仍作为 supporting assets，不再抢主创新位置
experiments/exp367_single_support/codex_train_design.md:3213:experiments/exp367_single_support/codex_review.md:2313:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5713:experiments/paper_notes/2026-04-15_prcv_reset.md:201:- `LGPA-D / GCN / OA-SD / MaxSim` = system assets / supporting modules
experiments/exp367_single_support/codex_train_design.md:3214:experiments/exp367_single_support/codex_review.md:2314:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5740:experiments/paper_notes/2026-03-19_support_complete_direction.md:61:1. `PAA` 对多人 query 更有效，说明 scene-level pose 的价值主要在复杂遮挡场景
experiments/exp367_single_support/codex_train_design.md:3215:experiments/exp367_single_support/codex_review.md:2315:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5775:experiments/cargo_cvpb/litreview2/validate/v_2.md:12678:validate/v_3.md:2206:../codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3216:experiments/exp367_single_support/codex_review.md:2316:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5776:experiments/cargo_cvpb/litreview2/validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3217:experiments/exp367_single_support/codex_review.md:2317:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5795:experiments/exp359_lm_reid/codex_review_raw_v2.md:229:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3218:experiments/exp367_single_support/codex_review.md:2318:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5796:experiments/exp359_lm_reid/codex_review_raw_v2.md:266:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3219:experiments/exp367_single_support/codex_review.md:2319:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6302:experiments/exp324b/_codex_review2.log:707:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3220:experiments/exp367_single_support/codex_review.md:2320:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6303:experiments/exp324b/_codex_review2.log:744:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3221:experiments/exp367_single_support/codex_review.md:2321:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6306:experiments/cargo_cvpb/litreview2/lmreid_push7.md:39:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3222:experiments/exp367_single_support/codex_review.md:2322:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6307:experiments/cargo_cvpb/litreview2/lmreid_push7.md:76:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3223:experiments/exp367_single_support/codex_review.md:2323:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6447:experiments/cargo_cvpb/litreview2/lmreid_push7.md:4708:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3224:experiments/exp367_single_support/codex_review.md:2324:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6448:experiments/cargo_cvpb/litreview2/lmreid_push7.md:4745:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3225:experiments/exp367_single_support/codex_review.md:2325:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6531:experiments/cargo_cvpb/litreview2/train_more_import.md:485:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3226:experiments/exp367_single_support/codex_review.md:2326:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6532:experiments/cargo_cvpb/litreview2/train_more_import.md:522:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3227:experiments/exp367_single_support/codex_review.md:2327:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6607:experiments/cargo_cvpb/litreview2/train_more_import.md:5555:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3228:experiments/exp367_single_support/codex_review.md:2328:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6608:experiments/cargo_cvpb/litreview2/train_more_import.md:5592:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3229:experiments/exp367_single_support/codex_review.md:2329:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6718:experiments/cargo_cvpb/litreview2/reassess/r_3.md:1538:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/exp367_single_support/codex_train_design.md:3230:experiments/exp367_single_support/codex_review.md:2330:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6719:experiments/cargo_cvpb/litreview2/reassess/r_3.md:1575:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3231:experiments/exp367_single_support/codex_review.md:2331:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6813:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4877:/Users/abdslm/Desktop/SOLIDER-REID/experiments/paper_notes/2026-03-19_support_complete_direction.md:61:1. `PAA` 对多人 query 更有效，说明 scene-level pose 的价值主要在复杂遮挡场景
experiments/exp367_single_support/codex_train_design.md:3232:experiments/exp367_single_support/codex_review.md:2332:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6917:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5477:reassess/r_3.md:4877:/Users/abdslm/Desktop/SOLIDER-REID/experiments/paper_notes/2026-03-19_support_complete_direction.md:61:1. `PAA` 对多人 query 更有效，说明 scene-level pose 的价值主要在复杂遮挡场景
experiments/exp367_single_support/codex_train_design.md:3233:experiments/exp367_single_support/codex_review.md:2333:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:7055:./experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3234:experiments/exp367_single_support/codex_review.md:2334:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:7758:./experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3235:experiments/exp367_single_support/codex_review.md:2335:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:8394:./experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:2526:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3236:experiments/exp367_single_support/codex_review.md:2336:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:8743:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:1261:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3237:experiments/exp367_single_support/codex_review.md:2337:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:8822:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:1517:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3238:experiments/exp367_single_support/codex_review.md:2356:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9869:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:2028:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3239:experiments/exp367_single_support/codex_review.md:2357:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10098:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:2483:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3240:experiments/exp367_single_support/codex_review.md:2376:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10589:./experiments/cargo_cvpb/litreview2/d17_eval.md:1350:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3241:experiments/exp367_single_support/codex_review.md:2377:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10595:./experiments/cargo_cvpb/litreview2/d17_eval.md:1684:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3242:experiments/exp367_single_support/codex_review.md:2378:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10599:./experiments/cargo_cvpb/litreview2/d17_eval.md:3591:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:2526:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3243:experiments/exp367_single_support/codex_review.md:2379:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10654:./experiments/exp148/design.md:17:1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
experiments/exp367_single_support/codex_train_design.md:3244:experiments/exp367_single_support/codex_review.md:2380:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10663:./experiments/cargo_cvpb/litreview2/pivot/p_2.md:2780:./validate/v_3.md:14623:../litreview2/validate/v_3.md:6823:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3245:experiments/exp367_single_support/codex_review.md:2381:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10736:./experiments/cargo_cvpb/litreview2/pivot/p_2.md:6036:pivot/p_2.md:2780:./validate/v_3.md:14623:../litreview2/validate/v_3.md:6823:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3246:experiments/exp367_single_support/codex_review.md:2382:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10866:./experiments/cargo_cvpb/litreview2/validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3247:experiments/exp367_single_support/codex_review.md:2383:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10890:./experiments/cargo_cvpb/litreview2/validate/v_3.md:6823:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3248:experiments/exp367_single_support/codex_review.md:2384:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10919:./experiments/cargo_cvpb/litreview2/validate/v_3.md:13639:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3249:experiments/exp367_single_support/codex_review.md:2385:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10941:./experiments/cargo_cvpb/litreview2/validate/v_3.md:14623:../litreview2/validate/v_3.md:6823:./validate/v_3.md:2205:../codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3250:experiments/exp367_single_support/codex_review.md:2424:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:11150:./experiments/cargo_cvpb/litreview2/meta_converge.md:2067:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3251:experiments/exp367_single_support/codex_review.md:2425:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:11185:./experiments/cargo_cvpb/litreview2/meta_converge.md:2820:experiments/exp148/design.md:17:1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
experiments/exp367_single_support/codex_train_design.md:3252:experiments/exp367_single_support/codex_review.md:2426:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:11191:./experiments/cargo_cvpb/litreview2/meta_converge.md:3286:experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/exp367_single_support/codex_train_design.md:3253:experiments/exp367_single_support/codex_review.md:2427:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12141:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/exp367_single_support/codex_train_design.md:3254:experiments/exp367_single_support/codex_review.md:2428:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12414:### 3. 不能再假设“同 ID 跨图 support 一定比单图更好学”
experiments/exp367_single_support/codex_train_design.md:3255:experiments/exp367_single_support/codex_review.md:2429:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12416:`exp109` 的 oracle 证明跨图完整 support **存在**
experiments/exp367_single_support/codex_train_design.md:3256:experiments/exp367_single_support/codex_review.md:2430:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12426:### Gap A: 单图内部能否合成“伪多 support”？
experiments/exp367_single_support/codex_train_design.md:3257:experiments/exp367_single_support/codex_review.md:2431:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12428:`exp109` 暴露的是 single-image support incomplete。  
experiments/exp367_single_support/codex_train_design.md:3258:experiments/exp367_single_support/codex_review.md:2432:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12429:如果 cross-image support 太难学，一个更合理的问题是：
experiments/exp367_single_support/codex_train_design.md:3259:experiments/exp367_single_support/codex_review.md:2433:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12431:**能否用同一张图自己构造出互补的 partial views，把单图训练成“伪多 support 学习”？**
experiments/exp367_single_support/codex_train_design.md:3260:experiments/exp367_single_support/codex_review.md:2434:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12976:experiments/decisions.md:1138:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/exp367_single_support/codex_train_design.md:3261:experiments/exp367_single_support/codex_review.md:2435:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12979:experiments/decisions.md:1755:**上下文**: `exp108 DACCM` 完成了第二轮 retrieval-time 原型验证。该实验把 `exp107` 的思路从 pooled person embedding 下沉到 `per-keypoint / common-support` 粒度，并以 `exp030a cvk_hybrid` 为主基线，比较：
experiments/exp367_single_support/codex_train_design.md:3262:experiments/exp367_single_support/codex_review.md:2436:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13011:experiments/decisions.md:2713:1. `exp138` 已经提供了足够的负边界：平滑 top-sensitive 只能算 supporting 机制
experiments/exp367_single_support/codex_train_design.md:3263:experiments/exp367_single_support/codex_review.md:2437:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13012:experiments/decisions.md:2722:- `exp138` 已停表，结论为 supporting 线
experiments/exp367_single_support/codex_train_design.md:3264:experiments/exp367_single_support/codex_review.md:2438:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13015:experiments/decisions.md:2947:**上下文**: exp142 SKC 训练完成。最终结果 mAP 60.3% / R1 71.8%（equal_concat），相对 exp030a -0.8% mAP / -1.9% R1。feature-level support-supervised completion 方向确认失败。
experiments/exp367_single_support/codex_train_design.md:3265:experiments/exp367_single_support/codex_review.md:2439:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13163:experiments/cargo_cvpb/codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3266:experiments/exp367_single_support/codex_review.md:2441:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13432:experiments/exp138/monitor.md:168:- 当前判断: `exp138` 结案，定性为 supporting 线
experiments/exp367_single_support/codex_train_design.md:3267:experiments/exp367_single_support/codex_review.md:2442:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13501:experiments/exp108/design.md:8:- 因此，`exp108` 的核心不是继续调 `exp107` 的公式，而是把同一问题重新落在 **per-keypoint / common-support** 粒度：
experiments/exp367_single_support/codex_train_design.md:3268:experiments/exp367_single_support/codex_review.md:2443:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13688:experiments/exp128/claude_review.md:48:**Partially support stop exp127 + start exp128, with reservations.**
experiments/exp367_single_support/codex_train_design.md:3269:experiments/exp367_single_support/codex_review.md:2445:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13935:experiments/results.md:892:> `exp148` 把单图改写成 `full / complementary-view-a / complementary-view-b` 三视图训练对象，用 pose-defined complementary pseudo-views 验证“单图能否被改写成伪多 support 学习对象”。该实验当前仍在运行，以下结论来自 `ep10/20/30` 训练监控。
experiments/exp367_single_support/codex_train_design.md:3270:experiments/exp367_single_support/codex_review.md:2449:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14741:experiments/cargo_cvpb/codex_review_raw.txt:4055:./model/modules/support_complete_bank.py:108:        Returns scaled prototypes plus a valid mask. Scaling follows SCFR:
experiments/exp367_single_support/codex_train_design.md:3271:experiments/exp367_single_support/codex_review.md:2450:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14742:experiments/cargo_cvpb/codex_review_raw.txt:4140:experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:3272:experiments/exp367_single_support/codex_review.md:2459:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:14789:experiments/cargo_cvpb/codex_review_raw.txt:4896:./experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/exp367_single_support/codex_train_design.md:3273:experiments/exp367_single_support/codex_review.md:2470:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:15416:experiments/exp142/claude_review.md:131:1. Loss 计算（L1081: `skc_bank.get_support()` → 用于算 loss）
experiments/exp367_single_support/codex_train_design.md:3274:experiments/exp367_single_support/codex_review.md:2473:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:17075:experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_5.md:10130:../../decisions.md:2947:**上下文**: exp142 SKC 训练完成。最终结果 mAP 60.3% / R1 71.8%（equal_concat），相对 exp030a -0.8% mAP / -1.9% R1。feature-level support-supervised completion 方向确认失败。
experiments/exp367_single_support/codex_train_design.md:3275:experiments/exp367_single_support/codex_review.md:2474:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:17088:experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_5.md:10291:../../../experiments/decisions.md:2947:**上下文**: exp142 SKC 训练完成。最终结果 mAP 60.3% / R1 71.8%（equal_concat），相对 exp030a -0.8% mAP / -1.9% R1。feature-level support-supervised completion 方向确认失败。
experiments/exp367_single_support/codex_train_design.md:3276:experiments/exp367_single_support/codex_review.md:2479:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:18621:experiments/cargo_cvpb/litreview2/validate/v_2.md:12588:validate/v_3.md:1486:../codex_review_raw.txt:711:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/exp367_single_support/codex_train_design.md:3277:experiments/exp367_single_support/codex_review.md:2480:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:18634:experiments/cargo_cvpb/litreview2/validate/v_2.md:12682:validate/v_3.md:2257:../codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/exp367_single_support/codex_train_design.md:3278:experiments/exp367_single_support/codex_review.md:2486:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19283:两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。
experiments/exp367_single_support/codex_train_design.md:3279:experiments/exp367_single_support/codex_review.md:2487:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19414:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/exp367_single_support/codex_train_design.md:3280:experiments/exp367_single_support/codex_review.md:2488:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19481:   - 写法：test-time supporting evaluations
experiments/exp367_single_support/codex_train_design.md:3281:experiments/exp367_single_support/codex_review.md:2500:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20115:experiments/exp359_lm_reid/codex_review_raw_v2.md:3871:experiments/exp359_lm_reid/design.md:20:两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。
experiments/exp367_single_support/codex_train_design.md:3282:experiments/exp367_single_support/codex_review.md:2502:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20157:experiments/exp359_lm_reid/codex_review_raw_v2.md:3913:experiments/exp359_lm_reid/codex_review_raw.md:49:两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。
experiments/exp367_single_support/codex_train_design.md:3283:experiments/exp367_single_support/codex_review.md:2506:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20657:experiments/exp359_lm_reid/design_blc.md:22:- **⚠️ market 约束（codex 警告）**：market 图是已裁好的人框，无原图上下文 → 要先 **pad crops 人工制造 bbox 不确定性**，否则 refiner 没有可校正的 support 偏移。这是 BLC 在 market 上先天降到 6.5 的原因。
experiments/exp367_single_support/codex_train_design.md:3284:experiments/exp367_single_support/codex_review.md:2509:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20991:experiments/exp359_lm_reid/design_blc.md:22:- **⚠️ market 约束（codex 警告）**：market 图是已裁好的人框，无原图上下文 → 要先 **pad crops 人工制造 bbox 不确定性**，否则 refiner 没有可校正的 support 偏移。这是 BLC 在 market 上先天降到 6.5 的原因。
experiments/exp367_single_support/codex_train_design.md:3285:experiments/exp367_single_support/codex_review.md:2522:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21695:experiments/exp359_lm_reid/codex_review_raw_v2.md:3871:experiments/exp359_lm_reid/design.md:20:两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。
experiments/exp367_single_support/codex_train_design.md:3286:experiments/exp367_single_support/codex_review.md:2524:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21737:experiments/exp359_lm_reid/codex_review_raw_v2.md:3913:experiments/exp359_lm_reid/codex_review_raw.md:49:两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。
experiments/exp367_single_support/codex_train_design.md:3287:experiments/exp367_single_support/codex_review.md:2528:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:22206:experiments/paper_notes/paper_15_common_support_training.md:67:## 这件事够不够支撑 B 类论文主贡献？
experiments/exp367_single_support/codex_train_design.md:3288:experiments/exp367_single_support/codex_review.md:2531:脚本本体很短，主要风险不在运行时 shape，而在评价定义：best/worst 的选择用了 query label 做 oracle，`eval_gallery` 的 Market 过滤基本像标准流程，但 kill-switch 当前把“每 ID 只留 1 张导致正样本变少”和“support 质量选择”混在一起。接下来我再核对 cache 和已有记录里是否有结果数字/生成来源。
experiments/exp367_single_support/codex_train_design.md:3289:experiments/exp367_single_support/codex_review.md:2533:/bin/zsh -lc 'nl -ba experiments/exp367_single_support/codex_review.md' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:3290:experiments/exp367_single_support/codex_review.md:2553:    14	审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:3291:experiments/exp367_single_support/codex_review.md:2555:    16	我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_train_design.md:3292:experiments/exp367_single_support/codex_review.md:2557:    18	/bin/zsh -lc 'nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:3293:experiments/exp367_single_support/codex_review.md:2559:    20	/bin/zsh -lc 'find experiments/exp367_single_support -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:3294:experiments/exp367_single_support/codex_review.md:2561:    22	experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:3295:experiments/exp367_single_support/codex_review.md:2562:    23	experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/codex_train_design.md:3296:experiments/exp367_single_support/codex_review.md:2566:    27	     2	"""Single-Support ReID — cheap kill-switch (零训练).
experiments/exp367_single_support/codex_train_design.md:3297:experiments/exp367_single_support/codex_review.md:2568:    29	     4	codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_train_design.md:3298:experiments/exp367_single_support/codex_review.md:2569:    30	     5	回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_train_design.md:3299:experiments/exp367_single_support/codex_reviewl 代理 #false-in-topk 证伪。**死区别推**: ①SMPL/人体几何对ReID无独特信号(四连证); ②遮挡ReID三墙(completion/occluder-gate/visibility); ③航拍-地面被VDT/GSAlign/ViSA/AG-VPReID占满; ④通用FM-import(MLLM/DINO/SD)赢SOTA已证负; ⑤frozen-image找accuracy隐藏变量被k-reciprocal/camera碾。资产: 强Swin/SOLIDER backbone + occluded_duke/market/MSMT/CARGO/AG-ReID.v2/RSTPReid数据 + 算力。**用户指令: 换问题/换ReID其他领域/换情况/主动构造特殊情况, 广撒网找新B类机会。** 铁律: ①问题或机制有新意(重定义'大家以为X其实Y'); ②有廉价(最好零训练frozen)kill-switch; ③联网查撞车不能是红海。
experiments/exp367_single_support/codex_train_design.md:4287:我在仓库里找到了 `exp367_single_support`，这和你描述的 cheap probe 直接相关。下面我会优先读取它和 `exp109` 的设计/监控，再把外部检索和本地证据合起来判断。
experiments/exp367_single_support/codex_train_design.md:4291:/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:4293:/bin/zsh -lc "sed -n '1,220p' experiments/exp367_single_support/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:4295:# exp367 Single-Support CVaR Episodic Loss（训练侧创新，2026-06-28）
experiments/exp367_single_support/codex_train_design.md:4299:用户 goal：找训练侧创新发 CCF-B，不收手，不轻易说穷尽，审查调研交 codex，严谨，文档记好。codex 训练侧深度调研 #1（最务实）：训练时每 ID 只用单图 support 定义身份，对 worst-support 选择做 CVaR 优化。回应 exp109 根问题（single-image support incomplete）。**纯训练侧**（episodic loss，输出常规 descriptor），严格非 test-time/检索侧/范式重定义。
experiments/exp367_single_support/codex_train_design.md:4305:ReID 训练用 multi-shot gallery（每 ID 多图），但模型学到的身份边界可能依赖"见过该 ID 多个 view"。部署常 single-shot（单图 support 定义新身份）。训练时**强制单图 support + CVaR worst-support 优化**，逼模型学"从任意单图恢复完整身份边界"的鲁棒表征，而非依赖 multi-view 平均。
experiments/exp367_single_support/codex_train_design.md:4313:- worst-support：每 ID 选最差单图（CVaR worst-case 目标针对的）
experiments/exp367_single_support/codex_train_design.md:4315:**GO**（support 选择是真训练瓶颈）：worst 比 full 掉 > 3 mAP 且 **best−worst gap > 3 mAP**（哪张 support 图很重要 = support 选择 matters，值得 CVaR 优化）。
experiments/exp367_single_support/codex_train_design.md:4326:- GO → 设计 Single-Support CVaR episodic loss 训练（每 ID 单图 support + worst-case 风险优化），训练侧第一 contribution，full fine-tune 前 codex 三审 diff。
experiments/exp367_single_support/codex_train_design.md:4404:# exp367 Single-Support CVaR — monitor
experiments/exp367_single_support/codex_train_design.md:4415:★v1 污染（codex 审抓）：single-support 跳无 positive query（比不同子集）+ distractor 压 1 张（负样本池变）→ best/random>full 假象。
experiments/exp367_single_support/codex_train_design.md:4416:★v2 修（codex 3 High）：common-valid query 共用 + distractor 全量 + 20 seed + missing 记 0。**single-support 都 <full（合理少正样本），best-worst 12.27 + random-worst 9.54，false10 best0.923≈worst0.927 → gap 不被 #false-in-topk 解释**。
experiments/exp367_single_support/codex_train_design.md:4424:support 选择有 oracle headroom（best-worst 12.27，不被 #false 解释），单图 support representation 是真训练瓶颈。**诚实标注**：best/worst 用 query-label oracle 上下界，证 headroom 存在；训练能否学到（不用 query）要 Single-Support CVaR train 验。
experiments/exp367_single_support/codex_train_design.md:4428:codex 调研 Single-Support CVaR 训练设计 + novelty 确认（63517）：episodic 单图 support + CVaR worst-case 如何写 loss、避六点定律陷阱（不塑造/对齐/压缩变体）、cheap 验证路径（frozen head 小训 or full FT）。GO 则训练侧创新动手，full fine-tune 前 codex 三审 diff。
experiments/exp367_single_support/codex_train_design.md:4869:web search: 2024 2025 person re-identification episodic training single support CVaR worst support ReID
experiments/exp367_single_support/codex_train_design.md:4889:web search: CVaR distributionally robust optimization deep learning worst-case loss paper
experiments/exp367_single_support/codex_train_design.md:4891:web search: "CVaR" "distributionally robust" "deep learning" "tail" "loss" "ICLR"
experiments/exp367_single_support/codex_train_design.md:4905:web search: "CVaR" "few-shot learning"
experiments/exp367_single_support/codex_train_design.md:4911:调研初步结论已经比较清楚：外部有 episodic few-shot / DG episodic / one-shot semi-supervised ReID，也有通用 CVaR/DRO；但我还没看到“标准监督 ReID 中把每个 ID 的单张 support 当身份定义，并对 support 选择尾部风险做 CVaR 优化”的直接先例。接下来我会把 loss 写到能落地的粒度，并给半天单卡验证路径。
experiments/exp367_single_support/codex_train_design.md:4915:experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/codex_train_design.md:4916:experiments/exp367_single_support/design.md
experiments/exp367_single_support/codex_train_design.md:4917:experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:4918:experiments/exp367_single_support/monitor.md
experiments/exp367_single_support/codex_train_design.md:4919:experiments/exp367_single_support/codex_train_design.md
experiments/exp367_single_support/codex_train_design.md:4920:experiments/exp367_single_support/codex_review2.md
experiments/exp367_single_support/codex_train_design.md:4923:/bin/zsh -lc 'rg -n "CVaR|single-support|single support|support|episodic|headroom|best-worst|false10" experiments/exp367_single_support experiments/cargo_cvpb experiments/paradigm_shift -S' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:5387:experiments/paradigm_shift/codex_lmreid_elevate.md:3785:training-side 转向备选（若 ordinary-aug 抹平 lattice 特异性）：Hard-Lattice ERM/CVaR（优化最难 variant 的 CE+triplet，非压 embedding）/ set-wise retrieval training。
experiments/exp367_single_support/codex_train_design.md:5568:experiments/cargo_cvpb/litreview2/pivot/clean/lmreid_salvage.txt:15:| (c) 换 training-side 机制 | 可作为第二条线，但必须避开 feature collapse；做 hard-lattice / CVaR / set-wise loss | **5.5/10** |
experiments/exp367_single_support/codex_train_design.md:5569:experiments/cargo_cvpb/litreview2/pivot/clean/lmreid_salvage.txt:58:1. **Hard-Lattice ERM / CVaR**
experiments/exp367_single_support/codex_train_design.md:5570:experiments/cargo_cvpb/litreview2/pivot/clean/lmreid_salvage.txt:81:先跑 `M=3 ordinary-aug no-LM`。如果它明显低于 lattice-noLM，比如低 1.0 mAP 以上，LM-ReID 可以改名重投为 **Lattice-Aware Augmentation + Lattice-Marginalized Retrieval**。如果它接近 `77.44`，训练端创新就收缩，转向 pure test-time + protocol，或者上 `Hard-Lattice ERM/CVaR` 抢回 training-side 主贡献。
experiments/exp367_single_support/codex_train_design.md:6074:experiments/cargo_cvpb/litreview2/oasd_mechanism.md:8057:experiments/cargo_cvpb/litreview2/lmreid_push7.md:3263:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:6656:先跑 `M=3 ordinary-aug no-LM`。如果它明显低于 lattice-noLM，比如低 1.0 mAP 以上，LM-ReID 可以改名重投为 **Lattice-Aware Augmentation + Lattice-Marginalized Retrieval**。如果它接近 `77.44`，训练端创新就收缩，转向 pure test-time + protocol，或者上 `Hard-Lattice ERM/CVaR` 抢回 training-side 主贡献。
experiments/exp367_single_support/codex_train_design.md:6075:experiments/cargo_cvpb/litreview2/oasd_mechanism.md:8059:experiments/cargo_cvpb/litreview2/lmreid_push7.md:3273:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:6740:先跑 `M=3 ordinary-aug no-LM`。如果它明显低于 lattice-noLM，比如低 1.0 mAP 以上，LM-ReID 可以改名重投为 **Lattice-Aware Augmentation + Lattice-Marginalized Retrieval**。如果它接近 `77.44`，训练端创新就收缩，转向 pure test-time + protocol，或者上 `Hard-Lattice ERM/CVaR` 抢回 training-side 主贡献。
experiments/exp367_single_support/codex_train_design.md:6121:experiments/cargo_cvpb/litreview2/oasd_mechanism.md:12770:experiments/cargo_cvpb/litreview2/pivot/clean/lmreid_salvage.txt:81:先跑 `M=3 ordinary-aug no-LM`。如果它明显低于 lattice-noLM，比如低 1.0 mAP 以上，LM-ReID 可以改名重投为 **Lattice-Aware Augmentation + Lattice-Marginalized Retrieval**。如果它接近 `77.44`，训练端创新就收缩，转向 pure test-time + protocol，或者上 `Hard-Lattice ERM/CVaR` 抢回 training-side 主贡献。
experiments/exp367_single_support/codex_train_design.md:6168:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:6590:| (c) 换 training-side 机制 | 可作为第二条线，但必须避开 feature collapse；做 hard-lattice / CVaR / set-wise loss | **5.5/10** |
experiments/exp367_single_support/codex_train_design.md:6169:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:6633:1. **Hard-Lattice ERM / CVaR**
experiments/exp367_single_support/codex_train_design.md:6170:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:6656:先跑 `M=3 ordinary-aug no-LM`。如果它明显低于 lattice-noLM，比如低 1.0 mAP 以上，LM-ReID 可以改名重投为 **Lattice-Aware Augmentation + Lattice-Marginalized Retrieval**。如果它接近 `77.44`，训练端创新就收缩，转向 pure test-time + protocol，或者上 `Hard-Lattice ERM/CVaR` 抢回 training-side 主贡献。
experiments/exp367_single_support/codex_train_design.md:6171:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:6674:| (c) 换 training-side 机制 | 可作为第二条线，但必须避开 feature collapse；做 hard-lattice / CVaR / set-wise loss | **5.5/10** |
experiments/exp367_single_support/codex_train_design.md:6172:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:6717:1. **Hard-Lattice ERM / CVaR**
experiments/exp367_single_support/codex_train_design.md:6173:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:6740:先跑 `M=3 ordinary-aug no-LM`。如果它明显低于 lattice-noLM，比如低 1.0 mAP 以上，LM-ReID 可以改名重投为 **Lattice-Aware Augmentation + Lattice-Marginalized Retrieval**。如果它接近 `77.44`，训练端创新就收缩，转向 pure test-time + protocol，或者上 `Hard-Lattice ERM/CVaR` 抢回 training-side 主贡献。
experiments/exp367_single_support/codex_train_design.md:6194:experiments/cargo_cvpb/litreview2/train3_paperstrategy.md:1951:experiments/exp359_lm_reid/monitor.md:117:training-side 转向备选（若 ordinary-aug 抹平 lattice 特异性）：Hard-Lattice ERM/CVaR（优化最难 variant 的 CE+triplet，非压 embedding）/ set-wise retrieval training。
experiments/exp367_single_support/codex_train_design.md:6197:experiments/cargo_cvpb/litreview2/train3_paperstrategy.md:2167:experiments/exp359_lm_reid/train_time_options.md:25:- raw Hard-Lattice ERM/CVaR（太像更狠 lattice-aug，续压判别性）→ 只做 cheap ablation（= 正在跑的 LM-S5，验证后即停）
experiments/exp367_single_support/codex_train_design.md:6207:experiments/cargo_cvpb/litreview2/train3_paperstrategy.md:2905:./experiments/exp359_lm_reid/train_time_options.md:25:- raw Hard-Lattice ERM/CVaR（太像更狠 lattice-aug，续压判别性）→ 只做 cheap ablation（= 正在跑的 LM-S5，验证后即停）
experiments/exp367_single_support/codex_train_design.md:6213:experiments/cargo_cvpb/litreview2/train3_paperstrategy.md:3960:./experiments/exp359_lm_reid/monitor.md:117:training-side 转向备选（若 ordinary-aug 抹平 lattice 特异性）：Hard-Lattice ERM/CVaR（优化最难 variant 的 CE+triplet，非压 embedding）/ set-wise retrieval training。
experiments/exp367_single_support/codex_train_design.md:6217:experiments/cargo_cvpb/litreview2/train3_paperstrategy.md:4185:- raw Hard-Lattice ERM/CVaR（太像更狠 lattice-aug，续压判别性）→ 只做 cheap ablation（= 正在跑的 LM-S5，验证后即停）
experiments/exp367_single_support/codex_train_design.md:6595:experiments/cargo_cvpb/litreview2/train_lens3_import.md:1180:training-side 转向备选（若 ordinary-aug 抹平 lattice 特异性）：Hard-Lattice ERM/CVaR（优化最难 variant 的 CE+triplet，非压 embedding）/ set-wise retrieval training。
experiments/exp367_single_support/codex_train_design.md:6598:experiments/cargo_cvpb/litreview2/train_lens3_import.md:4657:web search: site:arxiv.org "Distributionally Robust Optimization" "deep learning" "CVaR" "group DRO"
experiments/exp367_single_support/codex_train_design.md:6599:experiments/cargo_cvpb/litreview2/train_lens3_import.md:4730:- **raw Hard-Lattice ERM / CVaR**：太像更狠的 lattice-aug，容易继续压判别性。最多做 cheap ablation。
experiments/exp367_single_support/codex_train_design.md:6600:experiments/cargo_cvpb/litreview2/train_lens3_import.md:4808:- **raw Hard-Lattice ERM / CVaR**：太像更狠的 lattice-aug，容易继续压判别性。最多做 cheap ablation。
experiments/exp367_single_support/codex_train_design.md:6711:experiments/cargo_cvpb/litreview2/meta_converge.md:6323:training-side 转向备选（若 ordinary-aug 抹平 lattice 特异性）：Hard-Lattice ERM/CVaR（优化最难 variant 的 CE+triplet，非压 embedding）/ set-wise retrieval training。
experiments/exp367_single_support/codex_train_design.md:7085:/bin/zsh -lc "find log -maxdepth 4 -iname '*single*support*' -o -iname '*support*probe*' -o -iname '*exp367*' | sort | head -80" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:7090:/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/codex_train_design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:7105:Single-Support CVaR ReID 训练侧创新, cheap probe 已 GO(frozen SOLIDER: support 选择 oracle headroom best-worst 12.27 / random-worst 9.54, false10 best≈worst 不被 #false-in-topk 解释; best/worst 用 query-label oracle 上下界)。现在要**设计训练**(把 oracle headroom 学成不用 query 的鲁棒表征)。**深度调研+设计**: (a)**novelty 严格确认**: 'episodic single-support training + CVaR worst-support 优化' for standard person ReID, 2024-26 web search 有直接先例吗? few-shot ReID(CFReID/ReMix)/episodic metric learning/DRO-ReID/hard-example mining 区别在哪? 这个角度真空白吗还是撞 episodic few-shot? (b)**训练设计**: 怎么把'每 ID 单图 support 定义身份 + worst-case 鲁棒'写成 loss? episodic batch 采样(每 ID 1 support + 多 query)? CVaR over worst support-query pairs? 和标准 triplet+CE 怎么结合(加项还是替换)? 避六点定律陷阱(不塑造/对齐/压缩变体)? (c)**cheap 验证路径**: 先 frozen backbone + 单 support episodic head 小训(不动 backbone)能否提升 worst-support? 还是必须 full fine-tune? 给最小训练验证(1 单卡半天)。(d)风险: 会不会退化成普通 hard-mining / 普通 single-shot eval? 给设计+novelty 判定+CCF-B 潜力分。务实中文严谨。
experiments/exp367_single_support/codex_train_design.md:7354:/bin/zsh -lc "sed -n '1,240p' experiments/exp367_single_support/codex_review.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:7369:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:7373:/bin/zsh -lc 'nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:7375:/bin/zsh -lc 'find experiments/exp367_single_support -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:7377:experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:7378:experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/codex_train_design.md:7384:     4	codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_train_design.md:7391:    11	  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_train_design.md:7394:    14	GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_train_design.md:7399:    19	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:7412:    32	print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_train_design.md:7442:    62	# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_train_design.md:7467:    87	print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_train_design.md:7471:/bin/zsh -lc "rg -n \"cvpb_single_support|single_support|ae_feats|false-in-topk|false|topk|exp367|CVPB|support\" experiments/exp367_single_support experiments -g '"'!*.npz'"' -g '"'!*.pth'"'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:7473:experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:7474:experiments/exp367_single_support/codex_review.md:16:我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_train_design.md:7477:experiments/exp367_single_support/cvpb_single_support_probe.py:4:codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_train_design.md:7478:experiments/exp367_single_support/cvpb_single_support_probe.py:5:回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_train_design.md:7479:experiments/exp367_single_support/cvpb_single_support_probe.py:10:  - random-support (每 ID 随机 1 图) : 随机单 support
experiments/exp367_single_support/codex_train_design.md:7480:experiments/exp367_single_support/cvpb_single_support_probe.py:11:  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_train_design.md:7481:experiments/exp367_single_support/cvpb_single_support_probe.py:12:  - best-support (每 ID 选最好 1 图)  : support 选择 oracle 上界
experiments/exp367_single_support/codex_train_design.md:7482:experiments/exp367_single_support/cvpb_single_support_probe.py:14:GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_train_design.md:7483:experiments/exp367_single_support/cvpb_single_support_probe.py:15:  worst 比 full 掉 > 3 mAP  AND  best - worst gap > 3 mAP (support 选择 matters)。
experiments/exp367_single_support/codex_train_design.md:7484:experiments/exp367_single_support/cvpb_single_support_probe.py:16:DEAD: best≈worst (哪张 support 都一样, 没 support 选择价值) 或 single≈full (单图够)。
experiments/exp367_single_support/codex_train_design.md:7485:experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_train_design.md:7486:experiments/exp367_single_support/cvpb_single_support_probe.py:19:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:7487:experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_train_design.md:7488:experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_train_design.md:7489:experiments/exp367_single_support/cvpb_single_support_probe.py:57:# random-support: 每 ID 随机 1 图
experiments/exp367_single_support/codex_train_design.md:7490:experiments/exp367_single_support/cvpb_single_support_probe.py:62:# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_train_design.md:7491:experiments/exp367_single_support/cvpb_single_support_probe.py:63:# 用 该 ID 的 gallery 图 与 该 ID 所有 query 的平均 sim 作为 support quality (高=好 support)
experiments/exp367_single_support/codex_train_design.md:7492:experiments/exp367_single_support/cvpb_single_support_probe.py:69:    # 每个候选 support 图 g 对 同 ID query 的平均 cos (排同 cam)
experiments/exp367_single_support/codex_train_design.md:7493:experiments/exp367_single_support/cvpb_single_support_probe.py:82:print(f'  best-support     : mAP={best_mAP:.2f} R1={best_R1:.2f}  (vs full {best_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:7494:experiments/exp367_single_support/cvpb_single_support_probe.py:83:print(f'  random-support   : mAP={rand_mAP:.2f} R1={rand_R1:.2f}  (vs full {rand_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:7495:experiments/exp367_single_support/cvpb_single_support_probe.py:84:print(f'  worst-support    : mAP={worst_mAP:.2f} R1={worst_R1:.2f}  (vs full {worst_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_train_design.md:7496:experiments/exp367_single_support/cvpb_single_support_probe.py:87:print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_train_design.md:7598:/bin/zsh -lc "sed -n '1,240p' experiments/exp367_single_support/cvpb_single_support_probe.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:7603:codex 训练侧 #1: 训练时每 ID 单图 support 定义身份, CVaR worst-support 优化。回应 exp109
experiments/exp367_single_support/codex_train_design.md:7608:  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
experiments/exp367_single_support/codex_train_design.md:7616:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_train_design.md:7634:print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_train_design.md:7642:print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:7674:print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)
experiments/exp367_single_support/codex_train_design.md:7709:print('  ※ best/worst 是 query-label oracle 诊断上下界, 证 headroom 存在; 训练能否学到要 CVaR train 验', flush=True)
experiments/exp367_single_support/codex_train_design.md:7713:/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/codex_review2.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:7728:再审 experiments/exp367_single_support/cvpb_single_support_probe.py v2(修了 v1 的 3 个 High: common-valid query mask + distractor 全量 + 多 seed + #false-in-topk)。确认修对吗: (a)common-valid query(full-gallery 有 positive 子集)所有 support 设置共用对吗? (b)distractor 全量 + 只 has-query ID 单 support, 负样本池不变对吗? (c)random 20 seed + 主判据 random-worst gap + best-random + 报 false10 对吗? (d)missing-positive 记 0 对吗? 还有残留 trivial 污染吗? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_train_design.md:7987:/bin/zsh -lc 'tail -120 experiments/exp367_single_support/codex_review2.md' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_train_design.md:8070:?? experiments/exp367_single_support/
experiments/exp367_single_support/codex_train_design.md:8091:(d) **对**。single-support 下某 query 没有可用 positive 时记 AP/R1=0 是合理的，能惩罚 support 选到同 camera 或无法覆盖 cross-camera positive 的情况。
experiments/exp367_single_support/codex_train_design.md:8106:(d) **对**。single-support 下某 query 没有可用 positive 时记 AP/R1=0 是合理的，能惩罚 support 选到同 camera 或无法覆盖 cross-camera positive 的情况。
experiments/exp367_single_support/codex_train_design.md:8127:web search: "CVaR" "Distributionally Robust Optimization" deep learning
experiments/exp367_single_support/codex_train_design.md:8131:web search: "single-support" "ReID"
experiments/exp367_single_support/codex_train_design.md:8133:web search: 2026 "person re-identification" "CVaR"
experiments/exp367_single_support/codex_train_design.md:8140:`episodic single-support training` + `CVaR / worst-support tail optimization`
experiments/exp367_single_support/codex_train_design.md:8142:作为训练目标来做常规 ReID descriptor。它会撞到“episodic / support-query / few-shot / DRO / hard mining”的邻域，但不是同一个问题定义。可推进，但必须把 claim 写窄：不是“发明 episodic training / CVaR”，而是“标准 person ReID 中优化单张 support 定义身份时的 tail risk”。
experiments/exp367_single_support/codex_train_design.md:8145:直接先例：未检到。检索词包括 `single-support ReID`, `worst-support ReID`, `CVaR person re-identification`, `support selection person re-identification`, `episodic single support person ReID`，没有命中直接方法。
experiments/exp367_single_support/codex_train_design.md:8152:- CVaR/DRO 是通用鲁棒优化工具，不新；但检索未见其被用于 person ReID 的 single-support support-selection tail。见 [DRO uniform performance](https://arxiv.org/abs/1810.08750) 和 [large-scale CVaR DRO](https://arxiv.org/abs/2010.05893)。
experiments/exp367_single_support/codex_train_design.md:8162:基础 single-support episodic CE：
experiments/exp367_single_support/codex_train_design.md:8178:然后对同一 ID 的 support 风险做 CVaR：
experiments/exp367_single_support/codex_train_design.md:8196:- 同时做一个 `L_ss_random` 对照，即随机 single-support episodic CE 但无 CVaR，用来证明不是“episode 本身”在涨。
experiments/exp367_single_support/codex_train_design.md:8198:不要一开始做 `CVaR over all support-query pairs`。那会太像 batch-hard positive mining。更干净的是“两级聚合”：先把某张 support 对该 ID 的多 query 失败合成 support risk，再对 support risk 做 CVaR。
experiments/exp367_single_support/codex_train_design.md:8205:- single-support random / worst / best 诊断，证明 tail risk 改善。
experiments/exp367_single_support/codex_train_design.md:8228:最大风险：它退化成普通 hard positive mining。解决方式是报告 `batch-hard triplet 加强版`、`pair-CVaR`、`random episodic CE` 三个对照；只有 `support-level CVaR` 独立赢，claim 才站得住。
experiments/exp367_single_support/codex_train_design.md:8230:第二风险：single-support probe 的 gap 来自 camera coverage / missing positive，而不是身份表征。必须在训练和评估里报 missing-positive、cross-camera support 覆盖、false10 random mean/std。
experiments/exp367_single_support/codex_train_design.md:8232:CCF-B 潜力：当前 novelty 7/10，机制清晰度 7/10，实证不确定性 5.5/10，综合 **6.5/10**。如果 last-stage/full fine-tune 能在 standard mAP 不掉的前提下显著抬 worst/random single-support，并且赢过 hard-mining 对照，可升到 **7.5/10**；如果只改善 single-shot diagnostic 但 full-gallery 不涨或下跌，降到 **4/10**，只能做分析附录。
experiments/exp367_single_support/codex_train_design.md:8238:`episodic single-support training` + `CVaR / worst-support tail optimization`
experiments/exp367_single_support/codex_train_design.md:8240:作为训练目标来做常规 ReID descriptor。它会撞到“episodic / support-query / few-shot / DRO / hard mining”的邻域，但不是同一个问题定义。可推进，但必须把 claim 写窄：不是“发明 episodic training / CVaR”，而是“标准 person ReID 中优化单张 support 定义身份时的 tail risk”。
experiments/exp367_single_support/codex_train_design.md:8243:直接先例：未检到。检索词包括 `single-support ReID`, `worst-support ReID`, `CVaR person re-identification`, `support selection person re-identification`, `episodic single support person ReID`，没有命中直接方法。
experiments/exp367_single_support/codex_train_design.md:8250:- CVaR/DRO 是通用鲁棒优化工具，不新；但检索未见其被用于 person ReID 的 single-support support-selection tail。见 [DRO uniform performance](https://arxiv.org/abs/1810.08750) 和 [large-scale CVaR DRO](https://arxiv.org/abs/2010.05893)。
experiments/exp367_single_support/codex_train_design.md:8260:基础 single-support episodic CE：
experiments/exp367_single_support/codex_train_design.md:8276:然后对同一 ID 的 support 风险做 CVaR：
experiments/exp367_single_support/codex_train_design.md:8294:- 同时做一个 `L_ss_random` 对照，即随机 single-support episodic CE 但无 CVaR，用来证明不是“episode 本身”在涨。
experiments/exp367_single_support/codex_train_design.md:8296:不要一开始做 `CVaR over all support-query pairs`。那会太像 batch-hard positive mining。更干净的是“两级聚合”：先把某张 support 对该 ID 的多 query 失败合成 support risk，再对 support risk 做 CVaR。
experiments/exp367_single_support/codex_train_design.md:8303:- single-support random / worst / best 诊断，证明 tail risk 改善。
experiments/exp367_single_support/codex_train_design.md:8326:最大风险：它退化成普通 hard positive mining。解决方式是报告 `batch-hard triplet 加强版`、`pair-CVaR`、`random episodic CE` 三个对照；只有 `support-level CVaR` 独立赢，claim 才站得住。
experiments/exp367_single_support/codex_train_design.md:8328:第二风险：single-support probe 的 gap 来自 camera coverage / missing positive，而不是身份表征。必须在训练和评估里报 missing-positive、cross-camera support 覆盖、false10 random mean/std。
experiments/exp367_single_support/codex_train_design.md:8330:CCF-B 潜力：当前 novelty 7/10，机制清晰度 7/10，实证不确定性 5.5/10，综合 **6.5/10**。如果 last-stage/full fine-tune 能在 standard mAP 不掉的前提下显著抬 worst/random single-support，并且赢过 hard-mining 对照，可升到 **7.5/10**；如果只改善 single-shot diagnostic 但 full-gallery 不涨或下跌，降到 **4/10**，只能做分析附录。
experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:18:/bin/zsh -lc 'nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:20:/bin/zsh -lc 'find experiments/exp367_single_support -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:22:experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:23:experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/codex_review.md:29:     4	codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_review.md:36:    11	  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_review.md:39:    14	GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_review.md:44:    19	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:57:    32	print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:87:    62	# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_review.md:112:    87	print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_review.md:116:/bin/zsh -lc "rg -n \"cvpb_single_support|single_support|ae_feats|false-in-topk|false|topk|exp367|CVPB|support\" experiments/exp367_single_support experiments -g '"'!*.npz'"' -g '"'!*.pth'"'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:118:experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:119:experiments/exp367_single_support/codex_review.md:16:我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_review.md:122:experiments/exp367_single_support/cvpb_single_support_probe.py:4:codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_review.md:123:experiments/exp367_single_support/cvpb_single_support_probe.py:5:回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_review.md:124:experiments/exp367_single_support/cvpb_single_support_probe.py:10:  - random-support (每 ID 随机 1 图) : 随机单 support
experiments/exp367_single_support/codex_review.md:125:experiments/exp367_single_support/cvpb_single_support_probe.py:11:  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_review.md:126:experiments/exp367_single_support/cvpb_single_support_probe.py:12:  - best-support (每 ID 选最好 1 图)  : support 选择 oracle 上界
experiments/exp367_single_support/codex_review.md:127:experiments/exp367_single_support/cvpb_single_support_probe.py:14:GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_review.md:128:experiments/exp367_single_support/cvpb_single_support_probe.py:15:  worst 比 full 掉 > 3 mAP  AND  best - worst gap > 3 mAP (support 选择 matters)。
experiments/exp367_single_support/codex_review.md:129:experiments/exp367_single_support/cvpb_single_support_probe.py:16:DEAD: best≈worst (哪张 support 都一样, 没 support 选择价值) 或 single≈full (单图够)。
experiments/exp367_single_support/codex_review.md:130:experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:131:experiments/exp367_single_support/cvpb_single_support_probe.py:19:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:132:experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:133:experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:134:experiments/exp367_single_support/cvpb_single_support_probe.py:57:# random-support: 每 ID 随机 1 图
experiments/exp367_single_support/codex_review.md:135:experiments/exp367_single_support/cvpb_single_support_probe.py:62:# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_review.md:136:experiments/exp367_single_support/cvpb_single_support_probe.py:63:# 用 该 ID 的 gallery 图 与 该 ID 所有 query 的平均 sim 作为 support quality (高=好 support)
experiments/exp367_single_support/codex_review.md:137:experiments/exp367_single_support/cvpb_single_support_probe.py:69:    # 每个候选 support 图 g 对 同 ID query 的平均 cos (排同 cam)
experiments/exp367_single_support/codex_review.md:138:experiments/exp367_single_support/cvpb_single_support_probe.py:82:print(f'  best-support     : mAP={best_mAP:.2f} R1={best_R1:.2f}  (vs full {best_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:139:experiments/exp367_single_support/cvpb_single_support_probe.py:83:print(f'  random-support   : mAP={rand_mAP:.2f} R1={rand_R1:.2f}  (vs full {rand_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:140:experiments/exp367_single_support/cvpb_single_support_probe.py:84:print(f'  worst-support    : mAP={worst_mAP:.2f} R1={worst_R1:.2f}  (vs full {worst_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:141:experiments/exp367_single_support/cvpb_single_support_probe.py:87:print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_review.md:307:experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:308:experiments/exp367_single_support/codex_review.md:16:我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_review.md:309:experiments/exp367_single_support/cvpb_single_support_probe.py:4:codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_review.md:310:experiments/exp367_single_support/cvpb_single_support_probe.py:5:回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_review.md:311:experiments/exp367_single_support/cvpb_single_support_probe.py:10:  - random-support (每 ID 随机 1 图) : 随机单 support
experiments/exp367_single_support/codex_review.md:312:experiments/exp367_single_support/cvpb_single_support_probe.py:11:  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_review.md:313:experiments/exp367_single_support/cvpb_single_support_probe.py:12:  - best-support (每 ID 选最好 1 图)  : support 选择 oracle 上界
experiments/exp367_single_support/codex_review.md:314:experiments/exp367_single_support/cvpb_single_support_probe.py:14:GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_review.md:315:experiments/exp367_single_support/cvpb_single_support_probe.py:15:  worst 比 full 掉 > 3 mAP  AND  best - worst gap > 3 mAP (support 选择 matters)。
experiments/exp367_single_support/codex_review.md:316:experiments/exp367_single_support/cvpb_single_support_probe.py:16:DEAD: best≈worst (哪张 support 都一样, 没 support 选择价值) 或 single≈full (单图够)。
experiments/exp367_single_support/codex_review.md:317:experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:318:experiments/exp367_single_support/cvpb_single_support_probe.py:19:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:319:experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:320:experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:321:experiments/exp367_single_support/cvpb_single_support_probe.py:57:# random-support: 每 ID 随机 1 图
experiments/exp367_single_support/codex_review.md:322:experiments/exp367_single_support/cvpb_single_support_probe.py:62:# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_review.md:323:experiments/exp367_single_support/cvpb_single_support_probe.py:63:# 用 该 ID 的 gallery 图 与 该 ID 所有 query 的平均 sim 作为 support quality (高=好 support)
experiments/exp367_single_support/codex_review.md:324:experiments/exp367_single_support/cvpb_single_support_probe.py:69:    # 每个候选 support 图 g 对 同 ID query 的平均 cos (排同 cam)
experiments/exp367_single_support/codex_review.md:325:experiments/exp367_single_support/cvpb_single_support_probe.py:82:print(f'  best-support     : mAP={best_mAP:.2f} R1={best_R1:.2f}  (vs full {best_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:326:experiments/exp367_single_support/cvpb_single_support_probe.py:83:print(f'  random-support   : mAP={rand_mAP:.2f} R1={rand_R1:.2f}  (vs full {rand_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:327:experiments/exp367_single_support/cvpb_single_support_probe.py:84:print(f'  worst-support    : mAP={worst_mAP:.2f} R1={worst_R1:.2f}  (vs full {worst_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:328:experiments/exp367_single_support/cvpb_single_support_probe.py:87:print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_review.md:2533:/bin/zsh -lc 'nl -ba experiments/exp367_single_support/codex_review.md' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:2553:    14	审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:2557:    18	/bin/zsh -lc 'nl -ba experiments/exp367_single_support/cvpb_single_support_probe.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:2559:    20	/bin/zsh -lc 'find experiments/exp367_single_support -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:2561:    22	experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:2562:    23	experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/codex_review.md:2568:    29	     4	codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_review.md:2575:    36	    11	  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_review.md:2578:    39	    14	GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_review.md:2583:    44	    19	Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:2596:    57	    32	print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:2626:    87	    62	# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_review.md:2651:   112	    87	print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_review.md:2655:   116	/bin/zsh -lc "rg -n \"cvpb_single_support|single_support|ae_feats|false-in-topk|false|topk|exp367|CVPB|support\" experiments/exp367_single_support experiments -g '"'!*.npz'"' -g '"'!*.pth'"'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:2657:   118	experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:2658:   119	experiments/exp367_single_support/codex_review.md:16:我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_review.md:2661:   122	experiments/exp367_single_support/cvpb_single_support_probe.py:4:codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_review.md:2662:   123	experiments/exp367_single_support/cvpb_single_support_probe.py:5:回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_review.md:2663:   124	experiments/exp367_single_support/cvpb_single_support_probe.py:10:  - random-support (每 ID 随机 1 图) : 随机单 support
experiments/exp367_single_support/codex_review.md:2664:   125	experiments/exp367_single_support/cvpb_single_support_probe.py:11:  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_review.md:2665:   126	experiments/exp367_single_support/cvpb_single_support_probe.py:12:  - best-support (每 ID 选最好 1 图)  : support 选择 oracle 上界
experiments/exp367_single_support/codex_review.md:2666:   127	experiments/exp367_single_support/cvpb_single_support_probe.py:14:GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_review.md:2667:   128	experiments/exp367_single_support/cvpb_single_support_probe.py:15:  worst 比 full 掉 > 3 mAP  AND  best - worst gap > 3 mAP (support 选择 matters)。
experiments/exp367_single_support/codex_review.md:2668:   129	experiments/exp367_single_support/cvpb_single_support_probe.py:16:DEAD: best≈worst (哪张 support 都一样, 没 support 选择价值) 或 single≈full (单图够)。
experiments/exp367_single_support/codex_review.md:2669:   130	experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:2670:   131	experiments/exp367_single_support/cvpb_single_support_probe.py:19:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:2671:   132	experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:2672:   133	experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:2673:   134	experiments/exp367_single_support/cvpb_single_support_probe.py:57:# random-support: 每 ID 随机 1 图
experiments/exp367_single_support/codex_review.md:2674:   135	experiments/exp367_single_support/cvpb_single_support_probe.py:62:# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_review.md:2675:   136	experiments/exp367_single_support/cvpb_single_support_probe.py:63:# 用 该 ID 的 gallery 图 与 该 ID 所有 query 的平均 sim 作为 support quality (高=好 support)
experiments/exp367_single_support/codex_review.md:2676:   137	experiments/exp367_single_support/cvpb_single_support_probe.py:69:    # 每个候选 support 图 g 对 同 ID query 的平均 cos (排同 cam)
experiments/exp367_single_support/codex_review.md:2677:   138	experiments/exp367_single_support/cvpb_single_support_probe.py:82:print(f'  best-support     : mAP={best_mAP:.2f} R1={best_R1:.2f}  (vs full {best_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:2678:   139	experiments/exp367_single_support/cvpb_single_support_probe.py:83:print(f'  random-support   : mAP={rand_mAP:.2f} R1={rand_R1:.2f}  (vs full {rand_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:2679:   140	experiments/exp367_single_support/cvpb_single_support_probe.py:84:print(f'  worst-support    : mAP={worst_mAP:.2f} R1={worst_R1:.2f}  (vs full {worst_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:2680:   141	experiments/exp367_single_support/cvpb_single_support_probe.py:87:print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_review.md:2846:   307	experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:2847:   308	experiments/exp367_single_support/codex_review.md:16:我先按 review 模式读脚本本体和相邻实验文档/缓存生成路径，重点核对 mAP 过滤、support 选择、cache 语义和 kill-switch 指标是否真的回答问题。
experiments/exp367_single_support/codex_review.md:2848:   309	experiments/exp367_single_support/cvpb_single_support_probe.py:4:codex 训练侧 #1: 训练时每 ID 只用单图 support 定义身份, 对 worst-support 选择做 CVaR 优化。
experiments/exp367_single_support/codex_review.md:2849:   310	experiments/exp367_single_support/cvpb_single_support_probe.py:5:回应 exp109 根问题(single-image support incomplete)。纯训练侧(episodic loss, 输出常规 descriptor),
experiments/exp367_single_support/codex_review.md:2850:   311	experiments/exp367_single_support/cvpb_single_support_probe.py:10:  - random-support (每 ID 随机 1 图) : 随机单 support
experiments/exp367_single_support/codex_review.md:2851:   312	experiments/exp367_single_support/cvpb_single_support_probe.py:11:  - worst-support (每 ID 选最差 1 图) : CVaR worst-case 目标针对的
experiments/exp367_single_support/codex_review.md:2852:   313	experiments/exp367_single_support/cvpb_single_support_probe.py:12:  - best-support (每 ID 选最好 1 图)  : support 选择 oracle 上界
experiments/exp367_single_support/codex_review.md:2853:   314	experiments/exp367_single_support/cvpb_single_support_probe.py:14:GO(support 选择是真训练瓶颈, 值得 CVaR 优化):
experiments/exp367_single_support/codex_review.md:2854:   315	experiments/exp367_single_support/cvpb_single_support_probe.py:15:  worst 比 full 掉 > 3 mAP  AND  best - worst gap > 3 mAP (support 选择 matters)。
experiments/exp367_single_support/codex_review.md:2855:   316	experiments/exp367_single_support/cvpb_single_support_probe.py:16:DEAD: best≈worst (哪张 support 都一样, 没 support 选择价值) 或 single≈full (单图够)。
experiments/exp367_single_support/codex_review.md:2856:   317	experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:2857:   318	experiments/exp367_single_support/cvpb_single_support_probe.py:19:Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:2858:   319	experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:2859:   320	experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:2860:   321	experiments/exp367_single_support/cvpb_single_support_probe.py:57:# random-support: 每 ID 随机 1 图
experiments/exp367_single_support/codex_review.md:2861:   322	experiments/exp367_single_support/cvpb_single_support_probe.py:62:# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_review.md:2862:   323	experiments/exp367_single_support/cvpb_single_support_probe.py:63:# 用 该 ID 的 gallery 图 与 该 ID 所有 query 的平均 sim 作为 support quality (高=好 support)
experiments/exp367_single_support/codex_review.md:2863:   324	experiments/exp367_single_support/cvpb_single_support_probe.py:69:    # 每个候选 support 图 g 对 同 ID query 的平均 cos (排同 cam)
experiments/exp367_single_support/codex_review.md:2864:   325	experiments/exp367_single_support/cvpb_single_support_probe.py:82:print(f'  best-support     : mAP={best_mAP:.2f} R1={best_R1:.2f}  (vs full {best_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:2865:   326	experiments/exp367_single_support/cvpb_single_support_probe.py:83:print(f'  random-support   : mAP={rand_mAP:.2f} R1={rand_R1:.2f}  (vs full {rand_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:2866:   327	experiments/exp367_single_support/cvpb_single_support_probe.py:84:print(f'  worst-support    : mAP={worst_mAP:.2f} R1={worst_R1:.2f}  (vs full {worst_mAP-full_mAP:+.2f})')
experiments/exp367_single_support/codex_review.md:2867:   328	experiments/exp367_single_support/cvpb_single_support_probe.py:87:print(f'  [verdict] {"GO (support 选择是真训练瓶颈, 值得 CVaR worst-support 优化)" if go else "DEAD (support 选择无价值, best≈worst 或单图够)"}')
experiments/exp367_single_support/codex_review.md:5500:./experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:5501:./experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:5502:./experiments/exp367_single_support/cvpb_single_support_probe.py:30:qf, qp, qc = z['qf'], z['qp'], z['qc']
experiments/exp367_single_support/codex_review.md:5503:./experiments/exp367_single_support/cvpb_single_support_probe.py:31:gf, gp, gc = z['gf'], z['gp'], z['gc']
experiments/exp367_single_support/codex_review.md:5504:./experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:5505:./experiments/exp367_single_support/cvpb_single_support_probe.py:35:def eval_gallery(g_idx):
experiments/exp367_single_support/codex_review.md:5506:./experiments/exp367_single_support/cvpb_single_support_probe.py:36:    """标准 Market mAP: gallery 只用 g_idx, 排同 cam 同 id。"""
experiments/exp367_single_support/codex_review.md:5507:./experiments/exp367_single_support/cvpb_single_support_probe.py:37:    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
experiments/exp367_single_support/codex_review.md:5508:./experiments/exp367_single_support/cvpb_single_support_probe.py:38:    sim = qf @ gff.T
experiments/exp367_single_support/codex_review.md:5509:./experiments/exp367_single_support/cvpb_single_support_probe.py:40:    for i in range(len(qf)):
experiments/exp367_single_support/codex_review.md:5510:./experiments/exp367_single_support/cvpb_single_support_probe.py:55:full_mAP, full_R1 = eval_gallery(np.arange(len(gf)))
experiments/exp367_single_support/codex_review.md:5511:./experiments/exp367_single_support/cvpb_single_support_probe.py:60:rand_mAP, rand_R1 = eval_gallery(rand_idx)
experiments/exp367_single_support/codex_review.md:5512:./experiments/exp367_single_support/cvpb_single_support_probe.py:73:        quality.append((qf[qs] @ gf[g]).mean() if len(qs) else -1.0)
experiments/exp367_single_support/codex_review.md:5513:./experiments/exp367_single_support/cvpb_single_support_probe.py:77:best_mAP, best_R1 = eval_gallery(np.array(best_idx))
experiments/exp367_single_support/codex_review.md:5514:./experiments/exp367_single_support/cvpb_single_support_probe.py:78:worst_mAP, worst_R1 = eval_gallery(np.array(worst_idx))
experiments/exp367_single_support/codex_review.md:5574:./experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:5575:./experiments/exp367_single_support/codex_review.md:43:    18	复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:5576:./experiments/exp367_single_support/codex_review.md:51:    26	ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:5577:./experiments/exp367_single_support/codex_review.md:55:    30	qf, qp, qc = z['qf'], z['qp'], z['qc']
experiments/exp367_single_support/codex_review.md:5578:./experiments/exp367_single_support/codex_review.md:56:    31	gf, gp, gc = z['gf'], z['gp'], z['gc']
experiments/exp367_single_support/codex_review.md:5579:./experiments/exp367_single_support/codex_review.md:57:    32	print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:5580:./experiments/exp367_single_support/codex_review.md:60:    35	def eval_gallery(g_idx):
experiments/exp367_single_support/codex_review.md:5581:./experiments/exp367_single_support/codex_review.md:61:    36	    """标准 Market mAP: gallery 只用 g_idx, 排同 cam 同 id。"""
experiments/exp367_single_support/codex_review.md:5582:./experiments/exp367_single_support/codex_review.md:62:    37	    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
experiments/exp367_single_support/codex_review.md:5583:./experiments/exp367_single_support/codex_review.md:63:    38	    sim = qf @ gff.T
experiments/exp367_single_support/codex_review.md:5584:./experiments/exp367_single_support/codex_review.md:65:    40	    for i in range(len(qf)):
experiments/exp367_single_support/codex_review.md:5585:./experiments/exp367_single_support/codex_review.md:80:    55	full_mAP, full_R1 = eval_gallery(np.arange(len(gf)))
experiments/exp367_single_support/codex_review.md:5586:./experiments/exp367_single_support/codex_review.md:85:    60	rand_mAP, rand_R1 = eval_gallery(rand_idx)
experiments/exp367_single_support/codex_review.md:5587:./experiments/exp367_single_support/codex_review.md:98:    73	        quality.append((qf[qs] @ gf[g]).mean() if len(qs) else -1.0)
experiments/exp367_single_support/codex_review.md:5588:./experiments/exp367_single_support/codex_review.md:102:    77	best_mAP, best_R1 = eval_gallery(np.array(best_idx))
experiments/exp367_single_support/codex_review.md:5589:./experiments/exp367_single_support/codex_review.md:103:    78	worst_mAP, worst_R1 = eval_gallery(np.array(worst_idx))
experiments/exp367_single_support/codex_review.md:5590:./experiments/exp367_single_support/codex_review.md:118:experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:5591:./experiments/exp367_single_support/codex_review.md:130:experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:5592:./experiments/exp367_single_support/codex_review.md:132:experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:5593:./experiments/exp367_single_support/codex_review.md:133:experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:5594:./experiments/exp367_single_support/codex_review.md:299:experiments/decisions.md:4691:**上下文**: 三个独立 codex(终身 d3 / 开集 d9 / 长尾 d10)收敛到同一 re-framing: ReID 失败由 GALLERY 组成(规模/膨胀/分布)驱动, 非只看 query/模型。用户要求零训练验证, ★铁律=每个 per-query 相关都控 trivial 代理(吸取 HUBNESS §7.6 教训: 上个诊断被漏控 #false-in-topk 证伪)。脚本 `cvpb_gallery_killswitch.py`, 复用 hubness 缓存特征, Market exp260b + Occluded-Duke exp255。双审(Claude broad 5 blocking 全修 + Codex)。
experiments/exp367_single_support/codex_review.md:5595:./experiments/exp367_single_support/codex_review.md:300:experiments/decisions.md:4694:- **测试 A Gallery-Growth Tax = LIVE**: frozen 模型旧 query mAP 随同域 gallery 膨胀结构性下降(Market 1x→10x −4.4, **OD −12.9**, 量级≈LReID 报的 forgetting)。CONTROL1(#false-in-topk, 杀 Hubness 的代理): ρ(−dAP,d#false)+0.74 大部分是 trivial 计数, 但"#false 完全不变"子集仍 −1.2(Market)/−2.6(OD) mAP, partial(OD)+0.28——结构成分过了致命代理。CONTROL2 ★决定性: real distractor −4.45(Market)/−13.16(OD) vs 列洗牌毁方向同 count −0.00 → tax 是结构性(distractor 身份几何咬人), 非机械 count。
experiments/exp367_single_support/codex_review.md:5596:./experiments/exp367_single_support/codex_review.md:307:experiments/exp367_single_support/codex_review.md:14:审查 experiments/exp367_single_support/cvpb_single_support_probe.py(Single-Support ReID cheap kill-switch 零训练, codex 训练侧#1)。逐行查 bug + kill-switch 设计是否有意义。审: (a)best/worst-support per-ID 选择逻辑对吗(用同 ID query 平均 sim 选 support quality 排同 cam)? (b)eval_gallery 标准 Market mAP 对吗(排同 cam 同 id)? (c)★kill-switch 判定(worst 比 full 掉>3 + best-worst gap>3)有意义还是 trivial(单图必掉 mAP)? best-worst gap 是否真反映 support 选择价值(非单纯少正样本)? (d)复用 ae_feats.npz cache 对吗? 有没有 #false-in-topk 该控但没控? verdict approve/needs-attention + 简短理由。务实中文。
experiments/exp367_single_support/codex_review.md:5597:./experiments/exp367_single_support/codex_review.md:317:experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:5598:./experiments/exp367_single_support/codex_review.md:319:experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:5599:./experiments/exp367_single_support/codex_review.md:320:experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:5600:./experiments/exp367_single_support/codex_review.md:465:experiments/exp366_active_evidence/cvpb_active_evidence_probe.py:29:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:5601:./experiments/exp367_single_support/codex_review.md:477:experiments/cargo_cvpb/hub_verify_p0c_deep.py:61:# per-query AP (junk removed) and raw #false-in-topk
experiments/exp367_single_support/codex_review.md:5602:./experiments/exp367_single_support/codex_review.md:501:experiments/exp366_active_evidence/design.md:21:★**诚实设计**：避 codex 的 trivial oracle（multi-query 必涨 = upper-bound 不是创新），真验 policy（预算分配 vs random）。控 margin（top1-top2 = #false-in-topk 的代理）。自查抓到 2 个 bug（margins 长度 != len(qf) 退化 policy；policy hard 应只在 has_second 池选）已 fix。
experiments/exp367_single_support/codex_review.md:5603:./experiments/exp367_single_support/codex_review.md:519:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:111:# per-query AP (+ optional #false-in-topk). Market protocol: drop same pid&cam junk.
experiments/exp367_single_support/codex_review.md:5604:./experiments/exp367_single_support/codex_review.md:520:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:114:def per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam, topk=None, return_false=False):
experiments/exp367_single_support/codex_review.md:5605:./experiments/exp367_single_support/codex_review.md:527:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:281:def positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, a_temp):
experiments/exp367_single_support/codex_review.md:5606:./experiments/exp367_single_support/codex_review.md:531:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:372:    base_aps, base_false = per_query_ap(cqf, gf[core_idx], cq_pid, cq_cam,
experiments/exp367_single_support/codex_review.md:5607:./experiments/exp367_single_support/codex_review.md:533:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:394:    ps = positive_support(cqf, cq_pid, cq_cam, gf[core_idx], g_pid[core_idx], g_cam[core_idx], cli.a_temp)
experiments/exp367_single_support/codex_review.md:5608:./experiments/exp367_single_support/codex_review.md:549:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:468:    aps, false_k = per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam,
experiments/exp367_single_support/codex_review.md:5609:./experiments/exp367_single_support/codex_review.md:552:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:483:    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
experiments/exp367_single_support/codex_review.md:5610:./experiments/exp367_single_support/codex_review.md:565:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:610:    aps, _ = per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam, return_false=True)
experiments/exp367_single_support/codex_review.md:5611:./experiments/exp367_single_support/codex_review.md:566:experiments/cargo_cvpb/cvpb_evidence_killswitch.py:612:    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
experiments/exp367_single_support/codex_review.md:5612:./experiments/exp367_single_support/codex_review.md:670:experiments/cargo_cvpb/hub_verify_p0_p4.py:126:    topk_prec = np.zeros(Nq)              # fraction same-id within EVAL top-k (junk removed)
experiments/exp367_single_support/codex_review.md:5613:./experiments/exp367_single_support/codex_review.md:817:experiments/cargo_cvpb/litreview/reviews/lit_8.md:2751:primary challenge is to identify and remove the potential false
experiments/exp367_single_support/codex_review.md:5614:./experiments/exp367_single_support/codex_review.md:945:experiments/exp360_intruder/codex_h2fail_decision.md:814:experiments/decisions.md:4694:- **测试 A Gallery-Growth Tax = LIVE**: frozen 模型旧 query mAP 随同域 gallery 膨胀结构性下降(Market 1x→10x −4.4, **OD −12.9**, 量级≈LReID 报的 forgetting)。CONTROL1(#false-in-topk, 杀 Hubness 的代理): ρ(−dAP,d#false)+0.74 大部分是 trivial 计数, 但"#false 完全不变"子集仍 −1.2(Market)/−2.6(OD) mAP, partial(OD)+0.28——结构成分过了致命代理。CONTROL2 ★决定性: real distractor −4.45(Market)/−13.16(OD) vs 列洗牌毁方向同 count −0.00 → tax 是结构性(distractor 身份几何咬人), 非机械 count。
experiments/exp367_single_support/codex_review.md:5615:./experiments/exp367_single_support/codex_review.md:950:experiments/exp360_intruder/codex_h2fail_decision.md:1023:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4632:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8892:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/HUBNESS_PAPER_DRAFT.md:43:We tested the obvious remedies and report them honestly. A zero-training hub penalty `score' = cos − λ·log(1+H_k)` gives only +0.31 (Market) / +1.51 (Occluded-Duke) mAP, and is **dominated** by same-camera down-weighting (+0.67 / +3.13) and k-reciprocal (+1.26 / **+10.98**)—the gap *widens* on the harder set. A training-side anti-hub embedding sits in the same space already covered by re-ranking. The mechanism (scene over-encoding) points to background/region suppression, which is non-generalizable here (a dataset-specific scene) and overlaps prior pose-masked suppression. We therefore present negative in-degree as a **diagnostic**: it tells you *where* strong ReID fails and *why* (gallery topology / non-identity factor), while the *fix* remains the province of established test-time tools.
experiments/exp367_single_support/codex_review.md:5616:./experiments/exp367_single_support/codex_review.md:960:experiments/exp360_intruder/codex_h2fail_decision.md:1033:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4738:experiments/cargo_cvpb/litreview2/false_negative_audit.md:12950:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/HUBNESS_PAPER_DRAFT.md:43:We tested the obvious remedies and report them honestly. A zero-training hub penalty `score' = cos − λ·log(1+H_k)` gives only +0.31 (Market) / +1.51 (Occluded-Duke) mAP, and is **dominated** by same-camera down-weighting (+0.67 / +3.13) and k-reciprocal (+1.26 / **+10.98**)—the gap *widens* on the harder set. A training-side anti-hub embedding sits in the same space already covered by re-ranking. The mechanism (scene over-encoding) points to background/region suppression, which is non-generalizable here (a dataset-specific scene) and overlaps prior pose-masked suppression. We therefore present negative in-degree as a **diagnostic**: it tells you *where* strong ReID fails and *why* (gallery topology / non-identity factor), while the *fix* remains the province of established test-time tools.
experiments/exp367_single_support/codex_review.md:5617:./experiments/exp367_single_support/codex_review.md:1028:experiments/exp360_intruder/codex_h2fail_decision.md:1200:experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3963:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4730:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5618:./experiments/exp367_single_support/codex_review.md:1029:experiments/exp360_intruder/codex_h2fail_decision.md:1201:experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3964:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4732:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5619:./experiments/exp367_single_support/codex_review.md:1040:experiments/exp360_intruder/codex_h2fail_decision.md:1748:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:1195:experiments/decisions.md:4694:- **测试 A Gallery-Growth Tax = LIVE**: frozen 模型旧 query mAP 随同域 gallery 膨胀结构性下降(Market 1x→10x −4.4, **OD −12.9**, 量级≈LReID 报的 forgetting)。CONTROL1(#false-in-topk, 杀 Hubness 的代理): ρ(−dAP,d#false)+0.74 大部分是 trivial 计数, 但"#false 完全不变"子集仍 −1.2(Market)/−2.6(OD) mAP, partial(OD)+0.28——结构成分过了致命代理。CONTROL2 ★决定性: real distractor −4.45(Market)/−13.16(OD) vs 列洗牌毁方向同 count −0.00 → tax 是结构性(distractor 身份几何咬人), 非机械 count。
experiments/exp367_single_support/codex_review.md:5620:./experiments/exp367_single_support/codex_review.md:1047:experiments/exp360_intruder/codex_h2fail_decision.md:2151:experiments/cargo_cvpb/litreview2/train_lens3_import.md:4452:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14255:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2333:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1755:../../cargo_cvpb/new_angle_AIRL.md:169:2. **头分化**: consistency 只读 f_rec(logits_rec/bn_feat_rec), f_full BNNeck/classifier 零 consistency **梯度**(smoke D4); clean f_rec 侧 detach(稳定目标)。**已知并接受的次要项(codex round-2 Medium)**: 退化 forward 是整模型 `model(deg_imgs)`(无 rec-only 路径), 故 f_full 的 frozen-bias BNNeck running mean/var 仍会"看到"退化 ground 图(仅统计跟踪, 非梯度泄漏)——与 `--airl` 单头路径完全一�lm_reid/codex_review_raw_v2.md:3138:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4190:reassess/r_2.md:1109:reassess/r_3.md:2244:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2185:- 关键 kill-switch 很硬：冻结强 Swin embedding，训练 probe 预测遮挡者 / donor ID。若 probe 显著高于随机，且在人遮挡 split 高、物体遮挡/Market 低，这就是 B 类方法稿需要的隐藏变量。
experiments/exp367_single_support/codex_review.md:5621:./experiments/exp367_single_support/codex_review.md:1053:experiments/exp360_intruder/codex_h2fail_decision.md:2256:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15359:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6009:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3963:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4730:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5622:./experiments/exp367_single_support/codex_review.md:1054:experiments/exp360_intruder/codex_h2fail_decision.md:2257:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15360:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6010:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3964:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4732:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5623:./experiments/exp367_single_support/codex_review.md:1060:experiments/exp360_intruder/codex_h2fail_decision.md:2263:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15366:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6108:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5624:./experiments/exp367_single_support/codex_review.md:1062:experiments/exp360_intruder/codex_h2fail_decision.md:2265:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15369:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6111:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5625:./experiments/exp367_single_support/codex_review.md:1086:experiments/exp360_intruder/codex_h2fail_decision.md:2302:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15414:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6297:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4730:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5626:./experiments/exp367_single_support/codex_review.md:1087:experiments/exp360_intruder/codex_h2fail_decision.md:2303:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15415:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6298:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4732:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5627:./experiments/exp367_single_support/codex_review.md:1088:experiments/exp360_intruder/codex_h2fail_decision.md:2307:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15419:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6313:./experiments/cargo_cvpb/litreview2/d17_eval.md:3886:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4730:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5628:./experiments/exp367_single_support/codex_review.md:1089:experiments/exp360_intruder/codex_h2fail_decision.md:2308:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15420:experiments/paradigm_shift/decision_tscd_vs_intruder.md:6314:./experiments/cargo_cvpb/litreview2/d17_eval.md:3887:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4732:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5629:./experiments/exp367_single_support/codex_review.md:1098:experiments/exp360_intruder/codex_h2fail_decision.md:2549:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15681:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7043:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3489:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5630:./experiments/exp367_single_support/codex_review.md:1100:experiments/exp360_intruder/codex_h2fail_decision.md:2551:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15683:experiments/paradigm_shift/decision_tscd_vs_intruder.md:7045:./experiments/paradigm_shift/decision_tscd_vs_intruder.md:3491:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5631:./experiments/exp367_single_support/codex_review.md:1142:experiments/exp360_intruder/codex_h2fail_decision.md:2712:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15866:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4730:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5632:./experiments/exp367_single_support/codex_review.md:1143:experiments/exp360_intruder/codex_h2fail_decision.md:2713:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15867:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4732:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5633:./experiments/exp367_single_support/codex_review.md:1144:experiments/exp360_intruder/codex_h2fail_decision.md:2735:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15892:experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3963:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4730:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5634:./experiments/exp367_single_support/codex_review.md:1145:experiments/exp360_intruder/codex_h2fail_decision.md:2736:experiments/paradigm_shift/decision_tscd_vs_intruder.md:15893:experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3964:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4732:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5635:./experiments/exp367_single_support/codex_review.md:1151:experiments/exp360_intruder/codex_h2fail_decision.md:2848:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16043:experiments/cargo_cvpb/litreview2/d17_eval.md:3886:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4730:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5636:./experiments/exp367_single_support/codex_review.md:1152:experiments/exp360_intruder/codex_h2fail_decision.md:2849:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16044:experiments/cargo_cvpb/litreview2/d17_eval.md:3887:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4732:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5637:./experiments/exp367_single_support/codex_review.md:1162:experiments/exp360_intruder/codex_h2fail_decision.md:2957:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16176:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10916:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11752:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5638:./experiments/exp367_single_support/codex_review.md:1164:experiments/exp360_intruder/codex_h2fail_decision.md:2959:experiments/paradigm_shift/decision_tscd_vs_intruder.md:16179:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:11803:从 Market/Occluded-Duke 中选高可见 target query `T` 和 donor 行人 `D`，把 donor 的人体 mask 或 bbox 贴到 `T` 上，生成 `M=T⊕D`。gallery 保持原始，里面同时有 target-ID 和 donor-ID。这样 donor 身份、target 身份、遮挡 mask 全部已知，但 eval 仍按 target pid 算。
experiments/exp367_single_support/codex_review.md:5639:./experiments/exp367_single_support/codex_review.md:1326:experiments/exp360_intruder/codex_h2fail_decision.md:5359:experiments/cargo_cvpb/litreview2/explore20/d_10.md:96:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_review.md:5640:./experiments/exp367_single_support/codex_review.md:1328:experiments/exp360_intruder/codex_h2fail_decision.md:5361:experiments/cargo_cvpb/litreview2/explore20/d_10.md:115:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_review.md:5641:./experiments/exp367_single_support/codex_review.md:1527:experiments/cargo_cvpb/litreview2/d17_eval.md:9185:experiments/cargo_cvpb/litreview2/false_negative_audit.md:708:连续 rho 只 ~+0.25（被大量 M=0 query 稀释, 5 箱里 3 箱 M≈0）, 但**分箱趋势干净**: 高 M(q) query 正是 k-reciprocal 修复增益最大的（OD 4.3×, Market 4.5×, 两集一致）。→ **P4 reframe 在分箱意义上成立**: "M(q) 标记了现成 re-rank 工具受益最大的 query"。这是验证后唯一仍站得住、可写的正向叙事（但注意 §P0c: M 此处可能也只是 `#false-in-topk` 的代理, reframe 同样可用 #false 复述, 严谨写作需在 P4 也加 `#false` 对照)。
experiments/exp367_single_support/codex_review.md:5642:./experiments/exp367_single_support/codex_review.md:1528:experiments/cargo_cvpb/litreview2/d17_eval.md:9186:experiments/cargo_cvpb/litreview2/false_negative_audit.md:711:~~M(q) 干净解释 AP 误差(rho+0.60, 控代理后仍在)~~ → **修订**: (1) rho+0.60/+0.65 含 circular self-loop, 去 circular（LOO/held-out）后降到 +0.33（OD）/+0.25（Market）; (2) **决定性**: 控 trivial 代理 `#false-in-topk` 后 M(q) 偏相关 ≈0（两集一致 −0.06/−0.05）, 即"gallery 负向 in-degree / 拓扑"框架相对"top-k 里错几个"无增量解释力——原 D3 漏控了这个最致命代理; (3) P3 机制只半支持（去背景 −47%, 但去人 −87% 反更大, 非纯场景因子）; (4) **唯一仍正向**: P4 高 M(q) query 正是 k-reciprocal 修复最多的（分箱 4.3×/4.5×, 两集一致）。→ **作为"诊断变量"的 headline 站不住**（被 trivial 代理吃掉）; 若要保留, 只能定位成"M(q)/负向 in-degree 标记 re-rank 高收益 query"的弱 reframe, 且必须诚实写明它相对 `#false-in-topk` 无增量。**当前形态不足以撑 analysis short 的核心 claim。**
experiments/exp367_single_support/codex_review.md:5643:./experiments/exp367_single_support/codex_review.md:1533:experiments/cargo_cvpb/litreview2/d17_eval.md:9191:experiments/cargo_cvpb/litreview2/false_negative_audit.md:1122:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:95:106:连续 rho 只 ~+0.25（被大量 M=0 query 稀释, 5 箱里 3 箱 M≈0）, 但**分箱趋势干净**: 高 M(q) query 正是 k-reciprocal 修复增益最大的（OD 4.3×, Market 4.5×, 两集一致）。→ **P4 reframe 在分箱意义上成立**: "M(q) 标记了现成 re-rank 工具受益最大的 query"。这是验证后唯一仍站得住、可写的正向叙事（但注意 §P0c: M 此处可能也只是 `#false-in-topk` 的代理, reframe 同样可用 #false 复述, 严谨写作需在 P4 也加 `#false` 对照)。
experiments/exp367_single_support/codex_review.md:5644:./experiments/exp367_single_support/codex_review.md:1534:experiments/cargo_cvpb/litreview2/d17_eval.md:9192:experiments/cargo_cvpb/litreview2/false_negative_audit.md:1123:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:96:109:~~M(q) 干净解释 AP 误差(rho+0.60, 控代理后仍在)~~ → **修订**: (1) rho+0.60/+0.65 含 circular self-loop, 去 circular（LOO/held-out）后降到 +0.33（OD）/+0.25（Market）; (2) **决定性**: 控 trivial 代理 `#false-in-topk` 后 M(q) 偏相关 ≈0（两集一致 −0.06/−0.05）, 即"gallery 负向 in-degree / 拓扑"框架相对"top-k 里错几个"无增量解释力——原 D3 漏控了这个最致命代理; (3) P3 机制只半支持（去背景 −47%, 但去人 −87% 反更大, 非纯场景因子）; (4) **唯一仍正向**: P4 高 M(q) query 正是 k-reciprocal 修复最多的（分箱 4.3×/4.5×, 两集一致）。→ **作为"诊断变量"的 headline 站不住**（被 trivial 代理吃掉）; 若要保留, 只能定位成"M(q)/负向 in-degree 标记 re-rank 高收益 query"的弱 reframe, 且必须诚实写明它相对 `#false-in-topk` 无增量。**当前形态不足以撑 analysis short 的核心 claim。**
experiments/exp367_single_support/codex_review.md:5645:./experiments/exp367_single_support/codex_review.md:1535:experiments/cargo_cvpb/litreview2/d17_eval.md:9193:experiments/cargo_cvpb/litreview2/false_negative_audit.md:1126:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:151:连续 rho 只 ~+0.25（被大量 M=0 query 稀释, 5 箱里 3 箱 M≈0）, 但**分箱趋势干净**: 高 M(q) query 正是 k-reciprocal 修复增益最大的（OD 4.3×, Market 4.5×, 两集一致）。→ **P4 reframe 在分箱意义上成立**: "M(q) 标记了现成 re-rank 工具受益最大的 query"。这是验证后唯一仍站得住、可写的正向叙事（但注意 §P0c: M 此处可能也只是 `#false-in-topk` 的代理, reframe 同样可用 #false 复述, 严谨写作需在 P4 也加 `#false` 对照)。
experiments/exp367_single_support/codex_review.md:5646:./experiments/exp367_single_support/codex_review.md:1536:experiments/cargo_cvpb/litreview2/d17_eval.md:9194:experiments/cargo_cvpb/litreview2/false_negative_audit.md:1127:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:154:~~M(q) 干净解释 AP 误差(rho+0.60, 控代理后仍在)~~ → **修订**: (1) rho+0.60/+0.65 含 circular self-loop, 去 circular（LOO/held-out）后降到 +0.33（OD）/+0.25（Market）; (2) **决定性**: 控 trivial 代理 `#false-in-topk` 后 M(q) 偏相关 ≈0（两集一致 −0.06/−0.05）, 即"gallery 负向 in-degree / 拓扑"框架相对"top-k 里错几个"无增量解释力——原 D3 漏控了这个最致命代理; (3) P3 机制只半支持（去背景 −47%, 但去人 −87% 反更大, 非纯场景因子）; (4) **唯一仍正向**: P4 高 M(q) query 正是 k-reciprocal 修复最多的（分箱 4.3×/4.5×, 两集一致）。→ **作为"诊断变量"的 headline 站不住**（被 trivial 代理吃掉）; 若要保留, 只能定位成"M(q)/负向 in-degree 标记 re-rank 高收益 query"的弱 reframe, 且必须诚实写明它相对 `#false-in-topk` 无增量。**当前形态不足以撑 analysis short 的核心 claim。**
experiments/exp367_single_support/codex_review.md:5647:./experiments/exp367_single_support/codex_review.md:1598:experiments/cargo_cvpb/litreview2/d17_eval.md:9256:experiments/cargo_cvpb/litreview2/false_negative_audit.md:2163:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/HUBNESS_ANALYSIS.md:106:连续 rho 只 ~+0.25（被大量 M=0 query 稀释, 5 箱里 3 箱 M≈0）, 但**分箱趋势干净**: 高 M(q) query 正是 k-reciprocal 修复增益最大的（OD 4.3×, Market 4.5×, 两集一致）。→ **P4 reframe 在分箱意义上成立**: "M(q) 标记了现成 re-rank 工具受益最大的 query"。这是验证后唯一仍站得住、可写的正向叙事（但注意 §P0c: M 此处可能也只是 `#false-in-topk` 的代理, reframe 同样可用 #false 复述, 严谨写作需在 P4 也加 `#false` 对照)。
experiments/exp367_single_support/codex_review.md:5648:./experiments/exp367_single_support/codex_review.md:1599:experiments/cargo_cvpb/litreview2/d17_eval.md:9257:experiments/cargo_cvpb/litreview2/false_negative_audit.md:2164:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/HUBNESS_ANALYSIS.md:109:~~M(q) 干净解释 AP 误差(rho+0.60, 控代理后仍在)~~ → **修订**: (1) rho+0.60/+0.65 含 circular self-loop, 去 circular（LOO/held-out）后降到 +0.33（OD）/+0.25（Market）; (2) **决定性**: 控 trivial 代理 `#false-in-topk` 后 M(q) 偏相关 ≈0（两集一致 −0.06/−0.05）, 即"gallery 负向 in-degree / 拓扑"框架相对"top-k 里错几个"无增量解释力——原 D3 漏控了这个最致命代理; (3) P3 机制只半支持（去背景 −47%, 但去人 −87% 反更大, 非纯场景因子）; (4) **唯一仍正向**: P4 高 M(q) query 正是 k-reciprocal 修复最多的（分箱 4.3×/4.5×, 两集一致）。→ **作为"诊断变量"的 headline 站不住**（被 trivial 代理吃掉）; 若要保留, 只能定位成"M(q)/负向 in-degree 标记 re-rank 高收益 query"的弱 reframe, 且必须诚实写明它相对 `#false-in-topk` 无增量。**当前形态不足以撑 analysis short 的核心 claim。**
experiments/exp367_single_support/codex_review.md:5649:./experiments/exp367_single_support/codex_review.md:1720:experiments/cargo_cvpb/litreview2/d17_eval.md:9378:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_1.md:7927:Fig. 2. The motivation behind the SGOS module is to generate semantically meaningful occlusion patches for more target simulation of real-world occlusion scenarios.
experiments/exp367_single_support/codex_review.md:5650:./experiments/exp367_single_support/codex_review.md:1739:experiments/cargo_cvpb/litreview2/d17_eval.md:9397:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10134:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_1.md:8931:323:Fig. 2. The motivation behind the SGOS module is to generate semantically meaningful occlusion patches for more target simulation of real-world occlusion scenarios.
experiments/exp367_single_support/codex_review.md:5651:./experiments/exp367_single_support/codex_review.md:1995:experiments/cargo_cvpb/litreview2/d17_eval.md:9653:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18559:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:10020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_1.md:7927:Fig. 2. The motivation behind the SGOS module is to generate semantically meaningful occlusion patches for more target simulation of real-world occlusion scenarios.
experiments/exp367_single_support/codex_review.md:5652:./experiments/exp367_single_support/codex_review.md:2014:experiments/cargo_cvpb/litreview2/d17_eval.md:9672:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18662:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:10134:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_1.md:8931:323:Fig. 2. The motivation behind the SGOS module is to generate semantically meaningful occlusion patches for more target simulation of real-world occlusion scenarios.
experiments/exp367_single_support/codex_review.md:5653:./experiments/exp367_single_support/codex_review.md:2112:experiments/cargo_cvpb/litreview2/explore20/clean/d_10.txt:15:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_review.md:5654:./experiments/exp367_single_support/codex_review.md:2125:experiments/cargo_cvpb/litreview2/explore20/d_10.md:96:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_review.md:5655:./experiments/exp367_single_support/codex_review.md:2132:experiments/cargo_cvpb/litreview2/explore20/d_10.md:115:- 零训练 kill-switch：用 Market/MSMT/CARGO 构造 Zipf arrival stream，frozen 特征跑 nearest-centroid/global-threshold/DBSCAN/k-reciprocal baseline。看 false-merge 是否随 head support count 单调上升；再测一个 support-calibrated threshold 是否在相同 known-ID recall 下显著降 tail false merge。若全局阈值已解决，直接杀。
experiments/exp367_single_support/codex_review.md:5656:./experiments/exp367_single_support/codex_review.md:2213:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4773:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4689:experiments/cargo_cvpb/litreview2/false_negative_audit.md:7895:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5657:./experiments/exp367_single_support/codex_review.md:2214:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4774:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4693:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8063:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:6784:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5658:./experiments/exp367_single_support/codex_review.md:2215:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4775:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4696:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8600:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:12332:../airl_codex_bundle/reviews/codex_7.md:8517:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5659:./experiments/exp367_single_support/codex_review.md:2216:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4776:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4697:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8601:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:12335:../airl_codex_bundle/reviews/codex_7.md:8579:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5660:./experiments/exp367_single_support/codex_review.md:2217:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4777:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4698:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8606:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:13600:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5661:./experiments/exp367_single_support/codex_review.md:2218:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4778:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4699:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8612:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:14584:../litreview2/validate/v_3.md:6784:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5662:./experiments/exp367_single_support/codex_review.md:2338:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9364:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3951:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4689:experiments/cargo_cvpb/litreview2/false_negative_audit.md:7895:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5663:./experiments/exp367_single_support/codex_review.md:2339:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9365:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3952:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4693:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8063:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:6784:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5664:./experiments/exp367_single_support/codex_review.md:2340:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9366:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3953:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4696:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8600:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:12332:../airl_codex_bundle/reviews/codex_7.md:8517:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5665:./experiments/exp367_single_support/codex_review.md:2341:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9367:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3954:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4697:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8601:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:12335:../airl_codex_bundle/reviews/codex_7.md:8579:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5666:./experiments/exp367_single_support/codex_review.md:2342:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9368:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3955:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4698:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8606:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:13600:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5667:./experiments/exp367_single_support/codex_review.md:2343:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9369:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3956:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4699:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8612:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:14584:../litreview2/validate/v_3.md:6784:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5668:./experiments/exp367_single_support/codex_review.md:2358:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10403:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4689:experiments/cargo_cvpb/litreview2/false_negative_audit.md:7895:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5669:./experiments/exp367_single_support/codex_review.md:2359:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10404:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4693:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8063:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:6784:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5670:./experiments/exp367_single_support/codex_review.md:2360:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10405:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4696:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8600:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:12332:../airl_codex_bundle/reviews/codex_7.md:8517:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5671:./experiments/exp367_single_support/codex_review.md:2361:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10406:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4697:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8601:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:12335:../airl_codex_bundle/reviews/codex_7.md:8579:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5672:./experiments/exp367_single_support/codex_review.md:2362:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10407:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4698:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8606:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:13600:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5673:./experiments/exp367_single_support/codex_review.md:2363:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10408:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4699:experiments/cargo_cvpb/litreview2/false_negative_audit.md:8612:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:14584:../litreview2/validate/v_3.md:6784:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5674:./experiments/exp367_single_support/codex_review.md:2394:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10981:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:7895:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5675:./experiments/exp367_single_support/codex_review.md:2395:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10982:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:8063:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:6784:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5676:./experiments/exp367_single_support/codex_review.md:2396:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10983:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:8600:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:12332:../airl_codex_bundle/reviews/codex_7.md:8517:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5677:./experiments/exp367_single_support/codex_review.md:2397:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10984:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:8601:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:12335:../airl_codex_bundle/reviews/codex_7.md:8579:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5678:./experiments/exp367_single_support/codex_review.md:2398:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10985:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:8606:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:13600:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5679:./experiments/exp367_single_support/codex_review.md:2399:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10986:./experiments/cargo_cvpb/litreview2/false_negative_audit.md:8612:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:14584:../litreview2/validate/v_3.md:6784:./validate/v_3.md:2166:../codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp367_single_support/codex_review.md:5680:./experiments/exp367_single_support/codex_review.md:2531:脚本本体很短，主要风险不在运行时 shape，而在评价定义：best/worst 的选择用了 query label 做 oracle，`eval_gallery` 的 Market 过滤基本像标准流程，但 kill-switch 当前把“每 ID 只留 1 张导致正样本变少”和“support 质量选择”混在一起。接下来我再核对 cache 和已有记录里是否有结果数字/生成来源。
experiments/exp367_single_support/codex_review.md:8028:/bin/zsh -lc "rg -n \"ae_feats\\.npz|np\\.savez\\(|savez_compressed|qf.*gf|gf.*qf|exp260b_base_gcn512_2stage|single-support|SINGLE-SUPPORT\" . -S -g '"'!experiments/exp367_single_support/codex_review.md'"' -g '"'!experiments/cargo_cvpb/litreview2/**'"' -g '"'!*.npz'"' -g '"'!*.pth'"' -g '"'!*.pt'"' -g '"'!*.pkl'"'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:8069:./experiments/exp367_single_support/cvpb_single_support_probe.py:18:复用 Market 特征 cache (/tmp/ae_feats.npz, frozen SOLIDER exp260b)。
experiments/exp367_single_support/codex_review.md:8070:./experiments/exp367_single_support/cvpb_single_support_probe.py:26:ap.add_argument('--cache', default='/tmp/ae_feats.npz')
experiments/exp367_single_support/codex_review.md:8071:./experiments/exp367_single_support/cvpb_single_support_probe.py:32:print(f'[single-support] q={len(qf)} g={len(gf)} ids={len(set(gp))}', flush=True)
experiments/exp367_single_support/codex_review.md:8072:./experiments/exp367_single_support/cvpb_single_support_probe.py:38:    sim = qf @ gff.T
experiments/exp367_single_support/codex_review.md:8073:./experiments/exp367_single_support/cvpb_single_support_probe.py:62:# best/worst-support: 每 ID 选 single-support 使该 ID 的"被同 ID query 检索质量"最高/最低
experiments/exp367_single_support/codex_review.md:8074:./experiments/exp367_single_support/cvpb_single_support_probe.py:73:        quality.append((qf[qs] @ gf[g]).mean() if len(qs) else -1.0)
experiments/exp367_single_support/codex_review.md:8075:./experiments/exp367_single_support/cvpb_single_support_probe.py:80:print(f'\n[SINGLE-SUPPORT RESULT]')
experiments/exp367_single_support/codex_review.md:8506:./experiments/paradigm_shift/codex_trainside_innovation.md:6008:- few-shot / mixed data：CFReID、ReMix 说明少样本/混合单摄像头数据已有先例，但“标准监督 ReID 中强制 single-support 表征学习”仍有空白。来源：[CFReID](https://arxiv.org/abs/2503.18469)、[ReMix](https://arxiv.org/abs/2410.21938)
experiments/exp367_single_support/codex_review.md:8507:./experiments/paradigm_shift/codex_trainside_innovation.md:6018:第一步 kill-switch：零训练。用现有 checkpoint 做 single-support gallery probe：每个 gallery ID 只留 1 张图，分别报 random-support / worst-support / best-support / full-gallery mAP。若 random 或 worst 比 full 掉 >3 mAP，且 best-support 能追回 >=2 mAP，说明 support 选择是真瓶颈。否则 kill。
experiments/exp367_single_support/codex_review.md:8508:./experiments/paradigm_shift/codex_trainside_innovation.md:6044:下一步最务实：先做候选 1 的零训练 single-support probe。它最贴合 `exp109` 根问题，且不需要新依赖、不改 batch size、不碰测试侧。
experiments/exp367_single_support/codex_review.md:8509:./experiments/paradigm_shift/codex_trainside_innovation.md:6055:- few-shot / mixed data：CFReID、ReMix 说明少样本/混合单摄像头数据已有先例，但“标准监督 ReID 中强制 single-support 表征学习”仍有空白。来源：[CFReID](https://arxiv.org/abs/2503.18469)、[ReMix](https://arxiv.org/abs/2410.21938)
experiments/exp367_single_support/codex_review.md:8510:./experiments/paradigm_shift/codex_trainside_innovation.md:6065:第一步 kill-switch：零训练。用现有 checkpoint 做 single-support gallery probe：每个 gallery ID 只留 1 张图，分别报 random-support / worst-support / best-support / full-gallery mAP。若 random 或 worst 比 full 掉 >3 mAP，且 best-support 能追回 >=2 mAP，说明 support 选择是真瓶颈。否则 kill。
experiments/exp367_single_support/codex_review.md:8511:./experiments/paradigm_shift/codex_trainside_innovation.md:6091:下一步最务实：先做候选 1 的零训练 single-support probe。它最贴合 `exp109` 根问题，且不需要新依赖、不改 batch size、不碰测试侧。
experiments/exp367_single_support/codex_review.md:8559:我找到了最初的 single-support kill-switch 说明：原始建议里其实有 “best 至少追回 >=2 mAP”，这比脚本现在的 `best-worst > 3` 更接近 support 选择价值，但仍然要和 random/one-positive 控制分开。
experiments/exp367_single_support/codex_review.md:8583:  6008	- few-shot / mixed data：CFReID、ReMix 说明少样本/混合单摄像头数据已有先例，但“标准监督 ReID 中强制 single-support 表征学习”仍有空白。来源：[CFReID](https://arxiv.org/abs/2503.18469)、[ReMix](https://arxiv.org/abs/2410.21938)
experiments/exp367_single_support/codex_review.md:8588:  6013	1. **Single-Support CVaR Episodic Loss**
experiments/exp367_single_support/codex_review.md:8589:  6014	核心：训练时每个 ID 只允许一个 support 图代表身份，query 必须从这个单图 support 中恢复身份边界；再对最差 support 选择做 CVaR/worst-case 优化。它直接回应 `single-image support incomplete`，不是测试侧，也不是范式重定义，最终仍输出常规 ReID descriptor。
experiments/exp367_single_support/codex_review.md:8593:  6018	第一步 kill-switch：零训练。用现有 checkpoint 做 single-support gallery probe：每个 gallery ID 只留 1 张图，分别报 random-support / worst-support / best-support / full-gallery mAP。若 random 或 worst 比 full 掉 >3 mAP，且 best-support 能追回 >=2 mAP，说明 support 选择是真瓶颈。否则 kill。
experiments/exp367_single_support/codex_review.md:8609:  6034	4. **Natural Camera-Pose Subgroup CVaR**
experiments/exp367_single_support/codex_review.md:8619:  6044	下一步最务实：先做候选 1 的零训练 single-support probe。它最贴合 `exp109` 根问题，且不需要新依赖、不改 batch size、不碰测试侧。
experiments/exp367_single_support/codex_review.md:8630:  6055	- few-shot / mixed data：CFReID、ReMix 说明少样本/混合单摄像头数据已有先例，但“标准监督 ReID 中强制 single-support 表征学习”仍有空白。来源：[CFReID](https://arxiv.org/abs/2503.18469)、[ReMix](https://arxiv.org/abs/2410.21938)
experiments/exp367_single_support/codex_review.md:8635:  6060	1. **Single-Support CVaR Episodic Loss**
experiments/exp367_single_support/codex_review.md:8636:  6061	核心：训练时每个 ID 只允许一个 support 图代表身份，query 必须从这个单图 support 中恢复身份边界；再对最差 support 选择做 CVaR/worst-case 优化。它直接回应 `single-image support incomplete`，不是测试侧，也不是范式重定义，最终仍输出常规 ReID descriptor。
experiments/exp367_single_support/codex_review.md:8640:  6065	第一步 kill-switch：零训练。用现有 checkpoint 做 single-support gallery probe：每个 gallery ID 只留 1 张图，分别报 random-support / worst-support / best-support / full-gallery mAP。若 random 或 worst 比 full 掉 >3 mAP，且 best-support 能追回 >=2 mAP，说明 support 选择是真瓶颈。否则 kill。
experiments/exp367_single_support/codex_review.md:9008:/bin/zsh -lc "find . -name '*ae_feats*' -o -name '*single_support*' -o -name '*exp367*' | sed -n '1,120p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp367_single_support/codex_review.md:9010:./experiments/exp367_single_support
experiments/exp367_single_support/codex_review.md:9011:./experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/codex_review.md:9018:- **High** [cvpb_single_support_probe.py:41](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:41): `eval_gallery` 对“固定 gallery”是标准 Market 过滤：排 same pid + same cam，再算 AP/R1。问题是 single-support 会制造“该 query 没有跨 cam positive”的情况，line 45 会直接跳过 query。这样 best/worst 可能比较的是不同 query 子集，不是同一任务难度。应报告 `n_valid_q`，并考虑 missing-positive 记 0，或固定 common-valid query mask。
experiments/exp367_single_support/codex_review.md:9020:- **High** [cvpb_single_support_probe.py:51](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:51): 每个 `pid` 只留 1 张会把 Market 的 distractor-only / no-query pid 也压成 1 张。若 cache 里有 `pid=0` 或其它 gallery-only ID，负样本池被大幅削弱，`#false-in-topk` 会变，single-support mAP 不能和 full-gallery 直接解释。no-query/distractor pid 应保持全量，或单独做等负样本数控制。
experiments/exp367_single_support/codex_review.md:9022:- **High** [cvpb_single_support_probe.py:85](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:85): kill-switch 条件 `worst-full drop >3 && best-worst >3` 还不够干净。`worst-full drop` 很容易混入“少正样本/换负样本池/跳过无 positive query”的 trivial 变化；`best-worst gap` 比 full-drop 更有意义，因为两边都是单 support，但仍受 valid-query 覆盖和负样本池变化影响。建议主判据改成 `best-random`、`random-worst` 多 seed 均值/std，并控制 `#false-in-topk` / top1 margin。
experiments/exp367_single_support/codex_review.md:9024:- **Medium** [cvpb_single_support_probe.py:62](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:62): best/worst 选择逻辑作为“positive support quality oracle”基本对：同 ID query、按候选 support 相似度均值、排 same cam。但它不是严格的 best/worst retrieval AP oracle，因为没看 distractor、margin、false positives；且用 query label 做 oracle，只能当诊断上/下界，不能直接证明训练时可学到 support 选择。
experiments/exp367_single_support/codex_review.md:9026:- **Medium** [cvpb_single_support_probe.py:26](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:26): `/tmp/ae_feats.npz` 复用只有在 provenance 明确时才可接受。当前机器这个文件不存在；脚本也没校验 keys、shape、feature norm、finite、q/g 数量、ckpt/config/hash、full-gallery mAP sanity。并且若是 exp260b Base cache，就不能直接当 exp030a/Swin-Tiny 主线证据。
experiments/exp367_single_support/codex_review.md:9028:- **Low** [cvpb_single_support_probe.py:57](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:57): random-support 只跑 seed=42，不能作为判据基准。至少 20-100 seeds 报均值/std。
experiments/exp367_single_support/codex_review.md:9042:- **High** [cvpb_single_support_probe.py:41](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:41): `eval_gallery` 对“固定 gallery”是标准 Market 过滤：排 same pid + same cam，再算 AP/R1。问题是 single-support 会制造“该 query 没有跨 cam positive”的情况，line 45 会直接跳过 query。这样 best/worst 可能比较的是不同 query 子集，不是同一任务难度。应报告 `n_valid_q`，并考虑 missing-positive 记 0，或固定 common-valid query mask。
experiments/exp367_single_support/codex_review.md:9044:- **High** [cvpb_single_support_probe.py:51](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:51): 每个 `pid` 只留 1 张会把 Market 的 distractor-only / no-query pid 也压成 1 张。若 cache 里有 `pid=0` 或其它 gallery-only ID，负样本池被大幅削弱，`#false-in-topk` 会变，single-support mAP 不能和 full-gallery 直接解释。no-query/distractor pid 应保持全量，或单独做等负样本数控制。
experiments/exp367_single_support/codex_review.md:9046:- **High** [cvpb_single_support_probe.py:85](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:85): kill-switch 条件 `worst-full drop >3 && best-worst >3` 还不够干净。`worst-full drop` 很容易混入“少正样本/换负样本池/跳过无 positive query”的 trivial 变化；`best-worst gap` 比 full-drop 更有意义，因为两边都是单 support，但仍受 valid-query 覆盖和负样本池变化影响。建议主判据改成 `best-random`、`random-worst` 多 seed 均值/std，并控制 `#false-in-topk` / top1 margin。
experiments/exp367_single_support/codex_review.md:9048:- **Medium** [cvpb_single_support_probe.py:62](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:62): best/worst 选择逻辑作为“positive support quality oracle”基本对：同 ID query、按候选 support 相似度均值、排 same cam。但它不是严格的 best/worst retrieval AP oracle，因为没看 distractor、margin、false positives；且用 query label 做 oracle，只能当诊断上/下界，不能直接证明训练时可学到 support 选择。
experiments/exp367_single_support/codex_review.md:9050:- **Medium** [cvpb_single_support_probe.py:26](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:26): `/tmp/ae_feats.npz` 复用只有在 provenance 明确时才可接受。当前机器这个文件不存在；脚本也没校验 keys、shape、feature norm、finite、q/g 数量、ckpt/config/hash、full-gallery mAP sanity。并且若是 exp260b Base cache，就不能直接当 exp030a/Swin-Tiny 主线证据。
experiments/exp367_single_support/codex_review.md:9052:- **Low** [cvpb_single_support_probe.py:57](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_single_support_probe.py:57): random-support 只跑 seed=42，不能作为判据基准。至少 20-100 seeds 报均值/std。

exec
/bin/zsh -lc 'git status --short' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-n3s5JkOb' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-zfxQEinG' (errno=Operation not permitted)
 M .claude/hooks/check_design.sh
 M .claude/rules/experiment_protocol.md
 M CLAUDE.md
 M experiments/decisions.md
 M model/backbones/swin_transformer.py
?? configs/market/swin_tiny_dg_directft.yml
?? configs/occluded_duke/exp361_ft.yml
?? experiments/afd_reid/
?? experiments/cargo_cvpb/INNOVATION_METHOD.md
?? experiments/cargo_cvpb/afd_train.py
?? experiments/cargo_cvpb/agreid_dataset.py
?? experiments/cargo_cvpb/agreid_smoke.py
?? experiments/cargo_cvpb/airl_agreidv2_result.md
?? experiments/cargo_cvpb/airl_codex_bundle/
?? experiments/cargo_cvpb/airl_dualbranch_result.md
?? experiments/cargo_cvpb/airl_gate_oracle.py
?? experiments/cargo_cvpb/airl_iso_result.md
?? experiments/cargo_cvpb/airl_main_table.md
?? experiments/cargo_cvpb/airl_related_work.md
?? experiments/cargo_cvpb/airl_scale_diag.py
?? experiments/cargo_cvpb/airl_story.md
?? experiments/cargo_cvpb/claude_review.md
?? experiments/cargo_cvpb/claude_review_ablations.md
?? experiments/cargo_cvpb/claude_review_acvp.md
?? experiments/cargo_cvpb/claude_review_agreid_v2.md
?? experiments/cargo_cvpb/claude_review_airl.md
?? experiments/cargo_cvpb/claude_review_airl_iso.md
?? experiments/cargo_cvpb/claude_review_dualbranch.md
?? experiments/cargo_cvpb/claude_review_iso_trunkrecce.md
?? experiments/cargo_cvpb/claude_review_ovli.md
?? experiments/cargo_cvpb/claude_review_setpool.md
?? experiments/cargo_cvpb/claude_review_swin.md
?? experiments/cargo_cvpb/codex_fleet_synthesis.md
?? experiments/cargo_cvpb/codex_novelty_airl.md
?? experiments/cargo_cvpb/codex_review.md
?? experiments/cargo_cvpb/codex_review_ablations.md
?? experiments/cargo_cvpb/codex_review_acvp.md
?? experiments/cargo_cvpb/codex_review_agreid_v2.md
?? experiments/cargo_cvpb/codex_review_airl_iso.md
?? experiments/cargo_cvpb/codex_review_dualbranch.md
?? experiments/cargo_cvpb/codex_review_iso_trunkrecce.md
?? experiments/cargo_cvpb/codex_review_ovli.md
?? experiments/cargo_cvpb/codex_review_ovli.txt
?? experiments/cargo_cvpb/codex_review_raw.txt
?? experiments/cargo_cvpb/codex_review_raw2.txt
?? experiments/cargo_cvpb/codex_review_resfix.md
?? experiments/cargo_cvpb/codex_review_setpool.md
?? experiments/cargo_cvpb/cvpb_cache_feats.py
?? experiments/cargo_cvpb/cvpb_containment_killswitch.py
?? experiments/cargo_cvpb/cvpb_d17_killswitch.py
?? experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py
?? experiments/cargo_cvpb/cvpb_gopl_killswitch.py
?? experiments/cargo_cvpb/cvpb_hubness_killswitch.py
?? experiments/cargo_cvpb/cvpb_intruder_probe.py
?? experiments/cargo_cvpb/cvpb_lats_diag.py
?? experiments/cargo_cvpb/cvpb_lats_probe.py
?? experiments/cargo_cvpb/cvpb_lattice_killswitch.py
?? experiments/cargo_cvpb/cvpb_lattice_killswitch_DESIGN.md
?? experiments/cargo_cvpb/cvpb_lattice_result.md
?? experiments/cargo_cvpb/cvpb_lcrs_probe.py
?? experiments/cargo_cvpb/cvpb_lm_reid_train.py
?? experiments/cargo_cvpb/cvpb_lpa_head.py
?? experiments/cargo_cvpb/cvpb_lrfd_probe.py
?? experiments/cargo_cvpb/cvpb_lsmrt_probe.py
?? experiments/cargo_cvpb/cvpb_osac_killswitch.py
?? experiments/cargo_cvpb/cvpb_osac_mk.log
?? experiments/cargo_cvpb/cvpb_osac_od.log
?? experiments/cargo_cvpb/cvpb_rankregret_killswitch.py
?? experiments/cargo_cvpb/cvpb_realizability_killswitch.py
?? experiments/cargo_cvpb/cvpb_rma.log
?? experiments/cargo_cvpb/cvpb_rma_killswitch.py
?? experiments/cargo_cvpb/design.md
?? experiments/cargo_cvpb/design_agreid_v2.md
?? experiments/cargo_cvpb/diag_swin_ckpt.py
?? experiments/cargo_cvpb/diag_swin_eval.py
?? experiments/cargo_cvpb/error_analysis_geom.py
?? experiments/cargo_cvpb/fgeu_realizability_result.md
?? experiments/cargo_cvpb/honest_assessment.md
?? experiments/cargo_cvpb/hub_chars_local/
?? experiments/cargo_cvpb/hub_failure_characterize.py
?? experiments/cargo_cvpb/hub_failure_grid_FINAL.png
?? experiments/cargo_cvpb/hub_verify_p0_p4.py
?? experiments/cargo_cvpb/hub_verify_p0c_deep.py
?? experiments/cargo_cvpb/hub_verify_p3_mask.py
?? experiments/cargo_cvpb/hub_vs_control_grid.png
?? experiments/cargo_cvpb/hub_zoom_top18.png
?? experiments/cargo_cvpb/hubness_logs/
?? experiments/cargo_cvpb/litreview/
?? experiments/cargo_cvpb/litreview2/B_CONTAINMENT_DESIGN.md
?? experiments/cargo_cvpb/litreview2/GOPL_KILLSWITCH_DESIGN.md
?? experiments/cargo_cvpb/litreview2/HUBNESS_ANALYSIS.md
?? experiments/cargo_cvpb/litreview2/HUBNESS_KILLSWITCH_DESIGN.md
?? experiments/cargo_cvpb/litreview2/HUBNESS_PAPER_DRAFT.md
?? experiments/cargo_cvpb/litreview2/OSAC_KILLSWITCH_DESIGN.md
?? experiments/cargo_cvpb/litreview2/RANKREGRET_KILLSWITCH_DESIGN.md
?? experiments/cargo_cvpb/litreview2/RANKREGRET_RESULT.md
?? experiments/cargo_cvpb/litreview2/SESSION_NEGATIVE_RESULTS.md
?? experiments/cargo_cvpb/litreview2/SYNTHESIS_METHODOLOGY.md
?? experiments/cargo_cvpb/litreview2/all_papers.txt
?? experiments/cargo_cvpb/litreview2/analyses/
?? experiments/cargo_cvpb/litreview2/batches/
?? experiments/cargo_cvpb/litreview2/claude_review_rankregret.md
?? experiments/cargo_cvpb/litreview2/codex_lsrc_review.md
?? experiments/cargo_cvpb/litreview2/codex_review_rankregret.md
?? experiments/cargo_cvpb/litreview2/cvpb_containment_full.log
?? experiments/cargo_cvpb/litreview2/cvpb_containment_killswitch_design.md
?? experiments/cargo_cvpb/litreview2/d17_eval.md
?? experiments/cargo_cvpb/litreview2/debate/
?? experiments/cargo_cvpb/litreview2/evidence_method_design.md
?? experiments/cargo_cvpb/litreview2/explore20.sh
?? experiments/cargo_cvpb/litreview2/explore20/
?? experiments/cargo_cvpb/litreview2/false_negative_audit.md
?? experiments/cargo_cvpb/litreview2/gallery_growth_method_design.md
?? experiments/cargo_cvpb/litreview2/hubness_debate.sh
?? experiments/cargo_cvpb/litreview2/hubness_paper_review.md
?? experiments/cargo_cvpb/litreview2/hubness_verify_logs/
?? experiments/cargo_cvpb/litreview2/lattice_method_design.md
?? experiments/cargo_cvpb/litreview2/launch_30deep.sh
?? experiments/cargo_cvpb/litreview2/lmreid_post_kanmen.md
?? experiments/cargo_cvpb/litreview2/lmreid_push7.md
?? experiments/cargo_cvpb/litreview2/lmreid_salvage.md
?? experiments/cargo_cvpb/litreview2/meta_converge.md
?? experiments/cargo_cvpb/litreview2/novelty_gallery_lens.md
?? experiments/cargo_cvpb/litreview2/novelty_hubness.md
?? experiments/cargo_cvpb/litreview2/novelty_osac.md
?? experiments/cargo_cvpb/litreview2/novelty_rankinstab.md
?? experiments/cargo_cvpb/litreview2/oasd_mechanism.md
?? experiments/cargo_cvpb/litreview2/ondisk_pivot.md
?? experiments/cargo_cvpb/litreview2/pivot/
?? experiments/cargo_cvpb/litreview2/plboa_mechanism.md
?? experiments/cargo_cvpb/litreview2/reassess.sh
?? experiments/cargo_cvpb/litreview2/reassess/
?? experiments/cargo_cvpb/litreview2/reassess2.sh
?? experiments/cargo_cvpb/litreview2/reassess2/
?? experiments/cargo_cvpb/litreview2/remedy_rethink.md
?? experiments/cargo_cvpb/litreview2/reviews/
?? experiments/cargo_cvpb/litreview2/rr_logs/
?? experiments/cargo_cvpb/litreview2/strategic_pivot.sh
?? experiments/cargo_cvpb/litreview2/train2_backbone.md
?? experiments/cargo_cvpb/litreview2/train2_input.md
?? experiments/cargo_cvpb/litreview2/train3_fourthclass.md
?? experiments/cargo_cvpb/litreview2/train3_paperstrategy.md
?? experiments/cargo_cvpb/litreview2/train4_final.md
?? experiments/cargo_cvpb/litreview2/train_lens1_align.md
?? experiments/cargo_cvpb/litreview2/train_lens2_uncertainty.md
?? experiments/cargo_cvpb/litreview2/train_lens3_import.md
?? experiments/cargo_cvpb/litreview2/train_more_disentangle.md
?? experiments/cargo_cvpb/litreview2/train_more_diversity.md
?? experiments/cargo_cvpb/litreview2/train_more_import.md
?? experiments/cargo_cvpb/litreview2/train_more_setwise.md
?? experiments/cargo_cvpb/litreview2/validate/
?? experiments/cargo_cvpb/litreview2/validate_candidates.sh
?? experiments/cargo_cvpb/litreview2/video_feasibility.md
?? experiments/cargo_cvpb/lmS3S4_driver.sh
?? experiments/cargo_cvpb/maxsim_probe.py
?? experiments/cargo_cvpb/monitor.md
?? experiments/cargo_cvpb/monitor_agreid_v2.md
?? experiments/cargo_cvpb/new_angle_AIRL.md
?? experiments/cargo_cvpb/osac_summary_market1501.json
?? experiments/cargo_cvpb/osac_summary_occluded_duke.json
?? experiments/cargo_cvpb/precheck_market.log
?? experiments/cargo_cvpb/precheck_oduke.log
?? experiments/cargo_cvpb/refrand_grid.png
?? experiments/cargo_cvpb/rr_logs/cvpb_rr_market_s1.log
?? experiments/cargo_cvpb/rr_logs/cvpb_rr_market_s2.log
?? experiments/cargo_cvpb/rr_logs/cvpb_rr_od_s0.log
?? experiments/cargo_cvpb/rr_logs/cvpb_rr_od_s1.log
?? experiments/cargo_cvpb/rr_logs/cvpb_rr_od_s2.log
?? experiments/cargo_cvpb/rr_logs/cvpb_rr_od_smoke.log
?? experiments/cargo_cvpb/rr_logs/rr_market_stage1.json
?? experiments/cargo_cvpb/rr_logs/rr_market_stage2.json
?? experiments/cargo_cvpb/rr_logs/rr_od_stage0.json
?? experiments/cargo_cvpb/rr_logs/rr_od_stage1.json
?? experiments/cargo_cvpb/rr_logs/rr_od_stage2.json
?? experiments/cargo_cvpb/smoke_acvp.py
?? experiments/cargo_cvpb/smoke_airl.py
?? experiments/cargo_cvpb/smoke_airl_dualbranch.py
?? experiments/cargo_cvpb/smoke_airl_iso.py
?? experiments/cargo_cvpb/smoke_ovli_ablations.py
?? experiments/cargo_cvpb/smoke_ovli_allview.py
?? experiments/cargo_cvpb/smoke_ovli_residual.py
?? experiments/cargo_cvpb/smoke_ovli_setpool.py
?? experiments/cargo_cvpb/smoke_swin_backbone.py
?? experiments/cbcl_t2i/
?? experiments/cross_view_cargo/
?? experiments/exp359_lm_reid/
?? experiments/exp360_intruder/
?? experiments/exp361_psc_jepa/
?? experiments/exp362_genoccl/
?? experiments/exp363_ag_foundation/
?? experiments/exp364_dg_foundation/
?? experiments/exp365_wildlife/
?? experiments/exp366_active_evidence/
?? experiments/exp367_single_support/
?? experiments/library_167_synthesis.md
?? experiments/paradigm_shift/
?? experiments/publishable_angles_synthesis.md
?? experiments/smpl_anchor_vireid/

codex
我还查了特征抽取器：train 特征是一次性缓存的 frozen descriptor，训练循环里只有 `head` 参数进优化器，backbone 没有反传路径。现在可以给结论；主要是 needs-attention，不是因为核心 loss 写错，而是因为 smoke 的对照和评估口径还有两个解释漏洞。
exec
/bin/zsh -lc "nl -ba experiments/cargo_cvpb/cvpb_lattice_killswitch.py | sed -n '260,320p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   260	    arr = (arr - PIXEL_MEAN) / PIXEL_STD
   261	    return arr.transpose(2, 0, 1)                            # CHW
   262	
   263	
   264	# =========================================================================== #
   265	# frozen model (pose DISABLED at test: pose_dict=None -> plain SOLIDER global feat)
   266	# =========================================================================== #
   267	class FrozenExtractor:
   268	    def __init__(self):
   269	        import torch
   270	        from config import cfg
   271	        from model import make_model
   272	        from datasets.market1501 import Market1501
   273	        from datasets.msmt17 import MSMT17
   274	        self.torch = torch
   275	        cfg.merge_from_file(os.path.join(_repo, cli.config))
   276	        _overrides = [
   277	            'TEST.WEIGHT', os.path.join(_repo, cli.ckpt),
   278	            'MODEL.POSE_TEST_FEAT', 'global',     # single clean global vector
   279	            'TEST.NECK_FEAT', 'after',            # trained BN-neck eval feature
   280	            'TEST.FEAT_NORM', 'yes',
   281	        ]
   282	        if cli.semantic_weight >= 0:               # match ckpt training (SOLIDER semantic-aware backbone)
   283	            _overrides += ['MODEL.SEMANTIC_WEIGHT', str(cli.semantic_weight)]
   284	        cfg.merge_from_list(_overrides)  # leave PRETRAIN_* as in config; load_param(ckpt) overwrites all weights anyway
   285	        cfg.freeze()
   286	        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
   287	        # need num_classes/camera_num/view_num to build the head; read from Market meta.
   288	        ds = {'market1501': Market1501, 'msmt17': MSMT17}[cli.dataset](
   289	            root=os.path.join(_repo, cli.data_root), verbose=False)
   290	        model = make_model(cfg, num_class=ds.num_train_pids,
   291	                           camera_num=ds.num_train_cams, view_num=1,
   292	                           semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
   293	        model.load_param(os.path.join(_repo, cli.ckpt))
   294	        self.model = model.cuda().eval()
   295	        self.cam_dummy = None
   296	        print(f"[model] loaded {cli.ckpt}; POSE_TEST_FEAT=global; pose DISABLED at test "
   297	              f"(pose_dict=None -> plain backbone global feat). num_cls={ds.num_train_pids}",
   298	              flush=True)
   299	
   300	    def feats_from_pil(self, pil_list):
   301	        """Batched frozen features for a list of 384x128 PIL images. L2-normalized."""
   302	        torch = self.torch
   303	        feats = []
   304	        B = cli.batch
   305	        for s in range(0, len(pil_list), B):
   306	            chunk = pil_list[s:s + B]
   307	            arr = np.stack([pil_to_tensor_np(im) for im in chunk], 0)
   308	            t = torch.from_numpy(arr).cuda(non_blocking=True)
   309	            cam = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
   310	            view = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
   311	            with torch.no_grad():
   312	                out = self.model(t, cam_label=cam, view_label=view, pose_dict=None)
   313	            feat = out[0] if isinstance(out, (tuple, list)) else out
   314	            assert torch.is_tensor(feat) and feat.dim() == 2, \
   315	                f"expected single global vector, got {type(feat)} {getattr(feat,'shape',None)}"
   316	            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
   317	            feats.append(feat.cpu().numpy().astype(np.float32))
   318	        return np.concatenate(feats, 0)
   319	
   320	

exec
/bin/zsh -lc "nl -ba experiments/exp366_active_evidence/cvpb_active_evidence_probe.py | sed -n '1,90p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Active Evidence Acquisition ReID — cheap kill-switch (零训练).
     3	
     4	codex 范式级方向 #1 (7/10): query 不只被动排序, 系统可花预算主动获取视觉证据(另一 camera 视角)。
     5	★真 kill-switch(非 codex 的 trivial oracle, multi-query 必涨): policy(hard query 选预算)能否接近 oracle?
     6	
     7	  - baseline      : single query mAP
     8	  - oracle-all    : 每 query + 同 ID 不同 camera 第二证据(multi-query mean) → upper-bound
     9	  - **policy**    : 只对 hard query(top-1 margin 小=不确定) 花预算 budget% 获取第二证据
    10	  - random        : 随机 budget% query 给第二证据 (对照)
    11	
    12	GO: policy gain / oracle-all gain >= 0.5  AND  policy 明显 > random → 主动获取证据 policy 有真价值。
    13	DEAD: policy ≈ random → 没 policy 价值(等于 trivial multi-query, 预算分配无效)。
    14	控 margin(top1-top2 sim) = #false-in-topk 的代理。frozen SOLIDER, 零训练。
    15	
    16	Run on lab-3090-d:
    17	  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 /root/miniconda3/envs/solider-reid/bin/python \
    18	    experiments/exp366_active_evidence/cvpb_active_evidence_probe.py \
    19	    --ckpt log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth 2>&1 | tee /tmp/cvpb_ae.log
    20	"""
    21	import sys, os, argparse
    22	import numpy as np
    23	
    24	ap = argparse.ArgumentParser()
    25	ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
    26	ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
    27	ap.add_argument('--budget', type=float, default=0.2)       # 20% query 可获取第二证据
    28	ap.add_argument('--data_dir', default='market1501')        # market1501 / occluded_duke
    29	ap.add_argument('--cache', default='/tmp/ae_feats.npz')
    30	cli = ap.parse_args()
    31	sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data', '--K', '1',
    32	            '--reuse_gallery', '--cache_gallery', '/tmp/ae_g.npz']
    33	sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments', 'cargo_cvpb'))
    34	import cvpb_lattice_killswitch as ks
    35	from datasets.bases import read_image
    36	
    37	REPO = ks._repo; ext = ks.FrozenExtractor()
    38	
    39	
    40	def extract(split):
    41	    items = ks.list_split(os.path.join(REPO, 'data', cli.data_dir, split))
    42	    pils = [ks._to_target_aspect(read_image(it[0])) for it in items]
    43	    feats = ext.feats_from_pil(pils).astype(np.float32)
    44	    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    45	    pid = np.array([it[1] for it in items]); cam = np.array([it[2] for it in items])
    46	    return feats, pid, cam
    47	
    48	
    49	if os.path.exists(cli.cache):
    50	    z = np.load(cli.cache)
    51	    qf, qp, qc, gf, gp, gc = z['qf'], z['qp'], z['qc'], z['gf'], z['gp'], z['gc']
    52	    print('[feat] cached', flush=True)
    53	else:
    54	    print('[AE] extract query/gallery feats ...', flush=True)
    55	    qf, qp, qc = extract('query')
    56	    gf, gp, gc = extract('bounding_box_test')
    57	    np.savez(cli.cache, qf=qf, qp=qp, qc=qc, gf=gf, gp=gp, gc=gc)
    58	print(f'[AE] q={len(qf)} g={len(gf)}', flush=True)
    59	
    60	
    61	def eval_market(qfeat, qp, qc, gf, gp, gc):
    62	    """标准 Market mAP/R1: 排除同 camera 同 ID gallery。返回 mAP + per-query margin(top1-top2)。"""
    63	    sim = qfeat @ gf.T
    64	    aps, r1s = [], []
    65	    margins = np.ones(len(qfeat))                        # 每 query 都算(难度, 不依赖 match)
    66	    for i in range(len(qfeat)):
    67	        keep = ~((gp == qp[i]) & (gc == qc[i]))         # 排同 cam 同 id
    68	        s = sim[i][keep]; gp_k = gp[keep]
    69	        ss = np.sort(s)[::-1]
    70	        margins[i] = float(ss[0] - ss[1]) if len(ss) > 1 else 1.0
    71	        o = np.argsort(-s); gp_o = gp_k[o]; match = (gp_o == qp[i])
    72	        if not match.any(): continue
    73	        cum = np.cumsum(match); ranks = np.arange(1, len(gp_o) + 1)
    74	        aps.append((cum[match] / ranks[match]).mean()); r1s.append(float(match[0]))
    75	    return 100*np.mean(aps), 100*np.mean(r1s), margins
    76	
    77	
    78	# 每 query 的第二证据 = 同 ID 不同 camera 的另一张 query 图 (无则无证据)
    79	from collections import defaultdict
    80	idc2q = defaultdict(list)
    81	for i in range(len(qf)): idc2q[(qp[i], qc[i])].append(i)
    82	second = -np.ones(len(qf), dtype=int)
    83	for i in range(len(qf)):
    84	    cands = [j for j in range(len(qf)) if qp[j] == qp[i] and qc[j] != qc[i]]
    85	    if cands: second[i] = cands[0]
    86	has_second = second >= 0
    87	print(f'[AE] queries with 2nd-evidence available: {has_second.sum()}/{len(qf)}', flush=True)
    88	
    89	
    90	def with_evidence(use_mask):

exec
/bin/zsh -lc "nl -ba experiments/exp367_single_support/design.md | sed -n '37,52p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
    37	★**novelty 真空白（codex 确认）**：2024-26 标准监督 person ReID 没有"episodic single-support training + CVaR worst-support tail optimization"直接先例（检索 single-support/worst-support/CVaR-ReID/support-selection 都没命中）。邻近但不同：CFReID(continual few-shot)/DG-episodic(domain-invariant)/ProtoNet(novel-class 优化 prototype 平均非 tail)/batch-hard(hard pair mining 非 support tail)。**claim 写窄**：不发明 episodic/CVaR，是"标准 ReID 优化单图 support 定义身份的 tail risk"。
    38	
    39	★**训练设计（two-level CVaR，加项不替换）**：
    40	- episode N ID × K 图，每 ID 1 support + K-1 query。
    41	- `risk(y,s)` = 该 support 对同 ID 多 query 的 CE 失败。
    42	- `L_cvar_y = CVaR_α(support risks)`，`L_ss_cvar = mean_y`。α=0.7/0.8。
    43	- `L = L_id_ce + L_triplet + λ·L_ss_cvar`（λ 0.1→0.3 warmup）。
    44	- support/query 都梯度 + 保 CE+Triplet 防 collapse。两级聚合(support risk→CVaR)非 CVaR over all pairs(避 batch-hard)。
    45	
    46	★**避坑（避六点定律）**：不写 support completion/feature alignment/prototype compression/query-dependent selector。训练用 label 算 worst support 可以，测试不选 best/不用 query label，输出常规 descriptor。
    47	
    48	★**cheap 验证路径**：① frozen head smoke(10-20ep, worst/random +0.8~1.0, 失败不判死) ② last-stage(20-40ep, worst+2/random+1/gap 缩≥2/full 不降>0.5, 对照普通 CE+Triplet continued FT) ③ full FT。
    49	
    50	★**风险+对照（防退化 hard-mining）**：报 batch-hard/pair-CVaR/random episodic CE 三对照，只 support-level CVaR 独立赢才站得住。报 missing-positive/cross-cam 覆盖/false10 random std。
    51	
    52	★CCF-B 6.5/10（last-stage/full FT 抬 worst/random + 赢 hard-mining → 7.5；只改 diagnostic 但 full-gallery 不涨 → 4 附录）。

exec
/bin/zsh -lc "nl -ba experiments/exp367_single_support/monitor.md | sed -n '1,25p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
     1	# exp367 Single-Support CVaR — monitor
     2	
     3	## cheap kill-switch（frozen SOLIDER Market exp260b, 零训练, 2026-06-28）
     4	
     5	### v1 → codex 审抓 3 High → v2 修
     6	
     7	| | full | best-support | random-support | worst-support | best-worst gap |
     8	|---|---|---|---|---|---|
     9	| v1（污染）| 94.43 | 98.61（>full!）| 96.46 | 88.17 | 10.44 |
    10	| **v2（干净）** | 94.43 | 76.08 | 73.36±0.22(20seed) | 63.82 | **12.27** |
    11	
    12	★v1 污染（codex 审抓）：single-support 跳无 positive query（比不同子集）+ distractor 压 1 张（负样本池变）→ best/random>full 假象。
    13	★v2 修（codex 3 High）：common-valid query 共用 + distractor 全量 + 20 seed + missing 记 0。**single-support 都 <full（合理少正样本），best-worst 12.27 + random-worst 9.54，false10 best0.923≈worst0.927 → gap 不被 #false-in-topk 解释**。
    14	
    15	### codex 两轮审（用户要审查交 codex）
    16	- v1：needs-attention，3 个 High（valid-query 污染 / 负样本池变 / kill-switch 不硬）。
    17	- v2：needs-attention（轻微残留非致命）：false10 没给 random mean/std + go 没检查 false10 + missing 可能混 camera-coverage。best/worst oracle 用 query-label 可接受。
    18	
    19	## ★VERDICT GO（基本可信）
    20	
    21	support 选择有 oracle headroom（best-worst 12.27，不被 #false 解释），单图 support representation 是真训练瓶颈。**诚实标注**：best/worst 用 query-label oracle 上下界，证 headroom 存在；训练能否学到（不用 query）要 Single-Support CVaR train 验。
    22	
    23	## 下一步
    24	
    25	codex 调研 Single-Support CVaR 训练设计 + novelty 确认（63517）：episodic 单图 support + CVaR worst-case 如何写 loss、避六点定律陷阱（不塑造/对齐/压缩变体）、cheap 验证路径（frozen head 小训 or full FT）。GO 则训练侧创新动手，full fine-tune 前 codex 三审 diff。

codex
**Verdict: needs-attention，但核心 loss 可跑。**

主要问题：

- **Medium** [cvpb_ss_cvar_smoke.py:81](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:81): `--mode random` 实际不是“随机选 1 个 support-slot 做 CE”，而是对全部 K 个 support-slot 的 CE 做均值，只是去掉 CVaR。作为“无 CVaR / mean episodic CE”对照是对的，但名字叫 `random` 容易误导；它不能代表随机 support 训练对照。

- **Medium** [cvpb_ss_cvar_smoke.py:105](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:105): eval 基本复用了 probe v2 的核心口径：common-valid query、distractor 全量、single-support missing-positive 记 0、worst-support 按同 ID 跨 cam query 相似度选。但 smoke 只报 mAP，不报 R1 / false10 / missing-positive 覆盖，也只跑 random 10 seed。若要和 probe v2 严格对齐，建议补这些诊断。

- **Low** [cvpb_ss_cvar_smoke.py:41](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:41): `train_cache` / `eval_cache` 没有 ckpt/config/dim provenance 校验。若 `/tmp/ss_train_feats.npz` 或 `/tmp/ae_feats.npz` 是旧 checkpoint 生成，会静默污染或维度报错。

核心判断：

(a) **episodic single-support CVaR loss 基本对。** [64-86](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:64) 实现的是 N ID × K 图；每个 slot `s` 当所有 ID 的 prototype，其余 K-1 图作 query；CE shape `[N,K-1,N]` 正确，target 是本 ID；`risks=[N,K]` 后 per-ID top tail 做 CVaR，`L=mean_CE+lam*CVaR` 正确。

(b) **和 two-level 设计一致。** 是 `support risk -> per-id CVaR`，不是 all-pair CVaR，也不是 batch-hard pair mining。注意它不是组合枚举“某 ID support=s、其它 ID support 任意”的全组合版本，而是 slot 同步枚举；这和脚本注释里的“每 support-slot 当 prototype”一致。

(c) **random/no-CVaR 对照方向对，但命名不准。** 它能回答“CVaR tail term 是否额外有用”，不能回答“随机单 support CE 是否足够”。

(d) **评估大体对，非完整 probe v2。** full/random/worst 都是在 projected query/gallery 上算；distractor 保持全量；single-support 缺 positive 记 AP=0。缺的是 false10/R1/random 20 seed/best-support 诊断。

(e) **不算普通 hard-mining 退化。** hard 的单位是“每 ID 的 support candidate risk”，且 risk 先平均该 support 对同 ID query 的 CE，再做 per-ID tail；不是直接挑 hardest pair。但 K=4、tail 很短时确实会有 hard-example flavor，后续 full 训练仍需要 batch-hard/pair-CVaR 对照来守住论文 claim。

frozen head / 数值稳定：

- backbone 确实 frozen：训练用 cached `ft`，optimizer 只含 `head.parameters()`。
- `Linear+BN` eye init 可以跑，但 BN 训练态会更新 running stats，所以“init eye”不是训练中严格 identity；用 random/mean-CE 对照即可抵消。
- CE/topk shape 默认 `N=16,K=4,alpha=0.7` 稳定；建议加 `assert K >= 2`、`0 < alpha < 1`、`k_tail <= K`。当前默认不会炸。
tokens used
137,783
**Verdict: needs-attention，但核心 loss 可跑。**

主要问题：

- **Medium** [cvpb_ss_cvar_smoke.py:81](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:81): `--mode random` 实际不是“随机选 1 个 support-slot 做 CE”，而是对全部 K 个 support-slot 的 CE 做均值，只是去掉 CVaR。作为“无 CVaR / mean episodic CE”对照是对的，但名字叫 `random` 容易误导；它不能代表随机 support 训练对照。

- **Medium** [cvpb_ss_cvar_smoke.py:105](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:105): eval 基本复用了 probe v2 的核心口径：common-valid query、distractor 全量、single-support missing-positive 记 0、worst-support 按同 ID 跨 cam query 相似度选。但 smoke 只报 mAP，不报 R1 / false10 / missing-positive 覆盖，也只跑 random 10 seed。若要和 probe v2 严格对齐，建议补这些诊断。

- **Low** [cvpb_ss_cvar_smoke.py:41](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:41): `train_cache` / `eval_cache` 没有 ckpt/config/dim provenance 校验。若 `/tmp/ss_train_feats.npz` 或 `/tmp/ae_feats.npz` 是旧 checkpoint 生成，会静默污染或维度报错。

核心判断：

(a) **episodic single-support CVaR loss 基本对。** [64-86](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp367_single_support/cvpb_ss_cvar_smoke.py:64) 实现的是 N ID × K 图；每个 slot `s` 当所有 ID 的 prototype，其余 K-1 图作 query；CE shape `[N,K-1,N]` 正确，target 是本 ID；`risks=[N,K]` 后 per-ID top tail 做 CVaR，`L=mean_CE+lam*CVaR` 正确。

(b) **和 two-level 设计一致。** 是 `support risk -> per-id CVaR`，不是 all-pair CVaR，也不是 batch-hard pair mining。注意它不是组合枚举“某 ID support=s、其它 ID support 任意”的全组合版本，而是 slot 同步枚举；这和脚本注释里的“每 support-slot 当 prototype”一致。

(c) **random/no-CVaR 对照方向对，但命名不准。** 它能回答“CVaR tail term 是否额外有用”，不能回答“随机单 support CE 是否足够”。

(d) **评估大体对，非完整 probe v2。** full/random/worst 都是在 projected query/gallery 上算；distractor 保持全量；single-support 缺 positive 记 AP=0。缺的是 false10/R1/random 20 seed/best-support 诊断。

(e) **不算普通 hard-mining 退化。** hard 的单位是“每 ID 的 support candidate risk”，且 risk 先平均该 support 对同 ID query 的 CE，再做 per-ID tail；不是直接挑 hardest pair。但 K=4、tail 很短时确实会有 hard-example flavor，后续 full 训练仍需要 batch-hard/pair-CVaR 对照来守住论文 claim。

frozen head / 数值稳定：

- backbone 确实 frozen：训练用 cached `ft`，optimizer 只含 `head.parameters()`。
- `Linear+BN` eye init 可以跑，但 BN 训练态会更新 running stats，所以“init eye”不是训练中严格 identity；用 random/mean-CE 对照即可抵消。
- CE/topk shape 默认 `N=16,K=4,alpha=0.7` 稳定；建议加 `assert K >= 2`、`0 < alpha < 1`、`k_tail <= K`。当前默认不会炸。
