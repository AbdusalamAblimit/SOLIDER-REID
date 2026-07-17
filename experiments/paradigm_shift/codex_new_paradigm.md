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
session id: 019f0a63-d3fe-7f01-9a63-725083e0d0dc
--------
user
用户明确指令: 放下 LM-ReID, 找**新的 person ReID 范式级别创新**(不是小修小补/模块/变体, 是问题重定义 or 全新机制 or 新监督范式)。**这些已探透全证伪/偏弱/撞先例, 严禁推回**: occluded 内部(Intruder/PSC-JEPA continued-pretrain/生成引擎/support set)/AG 视频证据积累/DG foundation-preserving/gallery(hubness/rank-regret/OSAC)/open-set 阈值-watchlist/Wildlife species-conditioning/VI-ReID(SMPL-anchor)/LM-ReID lattice marginalization 训练端六点定律/SMPL 几何(对 ReID 无独特信号)/FM-import(MLLM/DINO/SD frozen 弱)/test-time trick。**深度调研真没探+范式级(2024-2026 最新, web search)**: (a)**问题重定义**: 主动 ReID(active query)/因果 ReID(去 camera-confounder, do-calculus)/联邦隐私 ReID/终身 continual ReID/生成式检索(generative retrieval, ID 当 token)/可解释 ReID/逆问题 ReID——逐个判 novelty+2024-26 先例+4 单卡可行+cheap kill-switch。(b)**新监督范式**: 自监督已 SOLIDER/LUPerson 占, 还有啥? 合成数据 scaling(3D 引擎/diffusion 造 ID)/弱监督/跨模态(audio-gait-language)/test-time training。(c)**新机制范式**: diffusion-ReID/in-context ReID(检索增强 ICL)/equivariant ReID/神经渲染 3D-person canonical。**要求**: 给 3-5 个真没探+范式级+有 2024-26 novelty 空白+4 单卡可行+有 cheap kill-switch 的, 排序+每个第一步零训练 kill-switch。务实中文, 诚实判别硬凑, **不要推回上面已死方向**。
codex
我先按仓库规则快速读当前实验结论文档，同时做 2024-2026 先例检索。重点会放在“能不能算范式变化”和“零训练能不能先杀掉”上，不把已经证伪的路线重新包装。
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
/bin/zsh -lc "sed -n '1,260p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,240p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID
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
3. exp005 证明 Stage 2 特征不足以直接做 identity classification，但这不代表 Stage 2 不适合做 spatial attention（PSG 不做分类，只做 spatial gating）
4. 多 stage PSG 是论文中可以画出更好架构图的设计
5. 如果多 stage 有效，这构成了一个"层次化姿态注入"的创新点

**执行结果**: exp009 mAP 58.3%, R1 67.2%, R5 81.2%, R10 85.2%。Multi-stage PSG (Stage 2+3) 与 single-stage (Stage 3 only) mAP 持平，R1 略低（-0.7%），R5/R10 略优（+0.4%/+0.3%），但增加了 156K 额外参数。**结论：Stage 2 PSG 无显著收益，pose spatial gating 在 Stage 3 已足够。后续聚焦于改进 PSG 机制本身，而非扩展注入范围。**

### [2026-03-10 01:45] 决策 #10

**上下文**: Phase 2 九个实验的系统总结：
1. Post-hoc part pooling 上限: +0.9% mAP (exp001)
2. PSG backbone injection 最佳: +1.7% mAP (exp007)
3. PSG + Part Pooling 组合: 不叠加 (exp008)
4. Multi-stage PSG: 无额外收益 (exp009)

**核心发现**: PSG Stage 3 (2 blocks, 102K params) 是当前最优配置。进一步改进需要改变 PSG 的内部机制或训练策略。

**选项**:
  A. PSG Channel Attention — 在 spatial gate 基础上加 channel-wise attention (SE-style)
  B. Backbone Freeze Warmup — 冻结 backbone 前 N epochs，让 PSG 先学稳定的 gate 模式
  C. PSG + Global-Part Concat — concat PSG-global + part features (不是 part_only 测试)

exec
/bin/zsh -lc "sed -n '1,260p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID
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
   - 首次将 NLP late interaction 引入 person ReID

### 核心范式论点
Occluded person ReID 不应该是 "extract one vector, compare vectors"，而是 "extract a set of body-part tokens, match sets"。这直接类比 NLP 从 sentence embeddings (BERT) 到 token-level late interaction (ColBERT) 的范式迁移。

### 贡献（候选）
1. **问题重构**: 将 occluded ReID 形式化为 partial-set-to-partial-set matching
2. **MaxSim matching**: 首次将 ColBERT-style late interaction 引入 person ReID
3. **Set-level metric learning**: 用 Soft-MaxSim 距离替换 pooled triplet，实现 train-test metric symmetry
4. **PSG + GCN pipeline**: 提供高质量 body-part token set 的提取方法

### 实验证据链（待补）
- MaxSim test-time: +1.0~1.5% mAP across all checkpoints ✅
- MaxSim training: exp152 进行中
- Ablation: soft vs hard MaxSim (exp152 vs exp152b) 进行中

---

## Phase 2 Story Update (2026-03-13)


codex
当前文档显示仓库已经大量验证过 pose/part/late-interaction/遮挡增强/语言部件等方向，且最近 story 又回到 PSG/GCN 系统线。用户这次明确要求跳出 LM-ReID 和已死方向，我会把这些只作为负面边界，不再建议沿它们加模块。
exec
/bin/zsh -lc "find experiments -maxdepth 1 -type d -name 'exp*' | sort -V | tail -n 12" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp355_pgpd
experiments/exp356r_pcmsc_random
experiments/exp356_pcmsc
experiments/exp357_pose_shuffle_ks
experiments/exp358_pose_channel_shuffle
experiments/exp359_lm_reid
experiments/exp360_intruder
experiments/exp361_psc_jepa
experiments/exp362_genoccl
experiments/exp363_ag_foundation
experiments/exp364_dg_foundation
experiments/exp365_wildlife

exec
/bin/zsh -lc 'rg -n "exp19[0-9]|exp2[0-9][0-9]|LM-ReID|lattice|single-image|范式|证伪|已死|不得" experiments/innovation_brainstorm.md experiments/decisions.md experiments/paper_materials/story.md experiments/results.md' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/paper_materials/story.md:32:   - `exp249` 与 `exp246` 已经说明 `LGPA-D + GCN` 双分支具备稳定互补性
experiments/paper_materials/story.md:35:   - `exp009`、`exp251`、`exp253` 都说明：multi-stage 不会在所有 scaffold 上自动更强
experiments/paper_materials/story.md:36:   - 但 `exp255 vs exp255b` 明确说明：在 `GCN512` 这类高容量结构分支上，`2-stage PSG` 是关键条件
experiments/paper_materials/story.md:42:当前训练端最强实验是 `exp255`：
experiments/paper_materials/story.md:47:- `exp255`: `GCN512 + 2-stage PSG = 73.2 / 83.3`
experiments/paper_materials/story.md:48:- `exp255b`: `GCN512 + 1-stage PSG = 71.5 / 81.9`
experiments/paper_materials/story.md:140:| *Small* | *LGPA-D+GCN+OA-SD (exp249, 进行中)* | *TBD* | *TBD* | *TBD* | *TBD* |
experiments/paper_materials/story.md:170:   - Non-detached (exp243): ep80 -1.1 mAP → 后期干扰
experiments/paper_materials/story.md:171:   - Detached (exp244): ep120 +2.1 mAP → 全程正向
experiments/paper_materials/story.md:211:4. **per-keypoint training loss 全面证伪**: PKC, MST, PACI, OERL, BA-PKC — 10 个实验全部失败。根本原因: detached GCN 阻断梯度到 backbone，non-detached 与 CE 冲突。
experiments/paper_materials/story.md:243:### 核心范式论点
experiments/paper_materials/story.md:244:Occluded person ReID 不应该是 "extract one vector, compare vectors"，而是 "extract a set of body-part tokens, match sets"。这直接类比 NLP 从 sentence embeddings (BERT) 到 token-level late interaction (ColBERT) 的范式迁移。
experiments/paper_materials/story.md:314:#### 发现 3: TDPC 方向全面证伪
experiments/paper_materials/story.md:323:这是 “recover” 范式，不同于 “suppress”(PSG) / “inject”(PAA) / “select”(pruning)。
experiments/paper_materials/story.md:330:5. **问题层面**: 如果 PGFI 也失败，应考虑把 story 转向 “pose-guided multi-granularity representation”（PSG+GCN+equal_concat 的整体范式叙事），而非继续追求单一新模块
experiments/paper_materials/story.md:731:     **用 same-ID multi-view support 作为 teacher，把 single-image representation 蒸馏成更接近 support-complete 的关键点表征**
experiments/paper_materials/story.md:957:  1. 继续坚持 `single-image support incomplete` 这个问题定义
experiments/paper_materials/story.md:971:  1. `single-image support incomplete` 的问题定义仍然成立
experiments/paper_materials/story.md:1066:  - 所以“learned pair module”并没有被做过，更谈不上被证伪
experiments/innovation_brainstorm.md:13:   - `exp009 / exp251 / exp253` 说明 multi-stage **不是普遍自动更优**
experiments/innovation_brainstorm.md:14:   - `exp255 vs exp255b` 又强烈说明：在 `GCN512` 结构分支下，`2-stage PSG` 是关键条件
experiments/innovation_brainstorm.md:19:   - `exp257-259` 已说明 recipe 空间基本耗尽
experiments/innovation_brainstorm.md:311:- **推论**: 在 PSG 基础上的所有单点改进（forward/loss/regularization）均已失败。应转向 PDS+StopGrad 的改进或全新范式
experiments/innovation_brainstorm.md:321:## 2026-03-23: MaxSim — Set-to-Set Metric Learning 范式确立
experiments/innovation_brainstorm.md:323:### 核心范式转变
experiments/innovation_brainstorm.md:433:  1. 问题层面仍锚定 `single-image support incomplete`
experiments/innovation_brainstorm.md:1017:- 它不依赖 GCN branch 改进（避开已证伪的方向）
experiments/innovation_brainstorm.md:1081:   - 重点关注不是 "加模块" 而是 "改范式" 的工作
experiments/innovation_brainstorm.md:1101:#### 1. 领域范式转移
experiments/innovation_brainstorm.md:1125:- 如果中性/失败：证明距离计算方式不是瓶颈，需要更根本的范式转变
experiments/innovation_brainstorm.md:1226:1. **全新架构方向**：探索完全不同的 pose 信息利用范式，而非在现有框架上微调
experiments/innovation_brainstorm.md:1341:1. **深度文献调研**（正在进行）：寻找 2025 年最新的 ReID 范式
experiments/innovation_brainstorm.md:1524:### 已证伪的方向（exp076-078）
experiments/innovation_brainstorm.md:1533:### 已证伪的方向（exp081 PQTD）
experiments/innovation_brainstorm.md:1542:这是一个不同于"suppress/inject/select"的新范式："recover"。
experiments/innovation_brainstorm.md:1622:   **same-ID support bank → single-image support-complete distillation**
experiments/innovation_brainstorm.md:1655:  1. 问题不是简单 occlusion comparison，而是 single-image support incomplete
experiments/innovation_brainstorm.md:2077:3. 因而“learned pair module”这条线 **还没有被真正做过，更没有被证伪**
experiments/innovation_brainstorm.md:2409:## 2026-03-22: feature-level completion 方向彻底证伪，转入注意力 inductive bias
experiments/innovation_brainstorm.md:2522:2. `single-image support incomplete` 可能确实更适合被改写成“伪多 support 学习对象”，而不是继续做 scorer / completion 小修补
experiments/innovation_brainstorm.md:2660:### OA-SD 核心特性（exp191-194 消融）
experiments/innovation_brainstorm.md:2662:1. **OA-SD + CE = 强正向**: +2.9/+2.6 vs CE base (exp191)
experiments/innovation_brainstorm.md:2664:3. **EMA decay 不敏感**: 0.99 vs 0.999 最终差异 <1% (exp192)
experiments/innovation_brainstorm.md:2665:4. **Loss weight 不敏感**: 1.0 vs 2.0 最终差异 <1% (exp194)
experiments/innovation_brainstorm.md:2666:5. **OA-SD + 3-view 是 additive**: exp193 = 64.4/76.5 vs exp190 = 64.2/75.6 (+0.2/+0.9)
experiments/innovation_brainstorm.md:2681:**验证** (exp195): SupCon + OA-SD global-only ep70=60.2/73.4
experiments/innovation_brainstorm.md:2688:- 消融链: exp188 (冲突) → exp195 (分离) 是清晰的证据
experiments/innovation_brainstorm.md:2695:| 2 | exp193 | 3-view + OA-SD + CE | 64.4% | 76.5% |
experiments/innovation_brainstorm.md:2696:| 3 | exp190 | 3-view + CE | 64.2% | 75.6% |
experiments/innovation_brainstorm.md:2698:| 5 | exp194 | OA-SD + CE (w=2.0) | 63.4% | 74.8% |
experiments/innovation_brainstorm.md:2699:| 6 | exp191 | OA-SD + CE | 63.2% | 75.4% |
experiments/innovation_brainstorm.md:2702:### 待验证: exp196 终极配置
experiments/innovation_brainstorm.md:2735:**4. 已证伪的方向更新**
experiments/innovation_brainstorm.md:2736:- STM (Token Mixup): 只加速不改善天花板 (exp197/198)
experiments/innovation_brainstorm.md:2737:- OA-SD + SupCon: 互斥，即使 global-only 也无法叠加 (exp195/196)
experiments/innovation_brainstorm.md:2739:### 选定方向: OA-RD (exp199)
experiments/innovation_brainstorm.md:2755:| exp195 | OA-SD global-only + SupCon | SupCon | ~-2.8 mAP | ❌ 信号太弱 |
experiments/innovation_brainstorm.md:2756:| exp196 | OA-SD global-only + SupCon + 3v | SupCon | -2.5/-1.4 | ❌ 同上 |
experiments/innovation_brainstorm.md:2757:| exp199 | OA-RD relational + SupCon + 3v | SupCon | ~-1.5/-3.4 | ❌ 关系级也冲突 |
experiments/innovation_brainstorm.md:2758:| exp191 | OA-SD all-token + CE | CE | +2.9/+2.6 | ✅ CE 兼容 |
experiments/innovation_brainstorm.md:2759:| exp193 | OA-SD all-token + CE + 3v | CE | +0.2/+0.9 | ✅ CE 兼容 |
experiments/innovation_brainstorm.md:2760:| exp200 | OA-RD relational + CE | CE | ~-1.0/-3.4 | ❌ OA-RD 不如 OA-SD |
experiments/innovation_brainstorm.md:2768:   - OA-SD 路线: exp193 = 64.4/76.5 (不加 SupCon)
experiments/innovation_brainstorm.md:2788:**MaxSim Hybrid 在 exp206 checkpoint 上无需重训即可获得 +1.8% mAP！**
experiments/innovation_brainstorm.md:2822:1. **Swin-Small + GCN+PAA+OA-SD + maxsim_hybrid = 72.4%** (exp210b with PKC=0.05)
experiments/innovation_brainstorm.md:2836:| exp210 | PKC w=0.5 | detached | 灾难 3.6% |
experiments/innovation_brainstorm.md:2837:| exp210b | PKC w=0.05 | detached | 无效 (=baseline) |
experiments/innovation_brainstorm.md:2838:| exp211 | MST w=0.5 | detached | 完全无效 (所有 loss 一致) |
experiments/innovation_brainstorm.md:2839:| exp213 | PKC+MST 组合 | detached | 灾难 40.6% |
experiments/innovation_brainstorm.md:2840:| exp215 | BA-PKC non-detach | non-det | 灾难 0.5% |
experiments/innovation_brainstorm.md:2841:| exp217 | OERL non-detach cosine | non-det | `62.2/75.2`，相对 `exp191 63.2/75.4` 为 `-1.0/-0.2` |
experiments/innovation_brainstorm.md:2842:| exp218 | PACI prototype bank | detached | `61.9/74.2`，相对 `exp191 63.2/75.4` 为 `-1.3/-1.2` |
experiments/innovation_brainstorm.md:2843:| exp219 | PACI without OA-SD | detached | 远程日志当前只确认到 `ep30=51.9/64.9`，早期即落后 baseline `52.2/65.2` |
experiments/innovation_brainstorm.md:2844:| exp220 | GSPB gradient scale 5% | 5% scale | `62.9/74.3`，相对 `exp191 63.2/75.4` 为 `-0.3/-1.1` |
experiments/innovation_brainstorm.md:2871:exp220 (scale=0.05) 完整对照:
experiments/innovation_brainstorm.md:2887:2. 重新定义训练范式（不是 loss 增改）
experiments/innovation_brainstorm.md:2892:## 2026-04-03: BT-PKD 系列 — Non-Detached Gradient 全面证伪
experiments/innovation_brainstorm.md:2895:- exp229: BT-PKD constant (Tiny) → -1.0/-0.4
experiments/innovation_brainstorm.md:2896:- exp230: BT-PKD constant (Small, no PAUG) → ~0/-0.7 (ep110)
experiments/innovation_brainstorm.md:2897:- exp231: BT-PKD cosine decay (Tiny) → -1.5/-1.1
experiments/innovation_brainstorm.md:2898:- exp232: BT-PKD cosine decay (Small) → terminated ep37
experiments/innovation_brainstorm.md:2926:1. **不再试图让 Part 梯度到达 backbone** — 这条路已彻底证伪
experiments/innovation_brainstorm.md:2968:## 2026-04-04: PPA — Pose-Prompted Part-Assignment Head (范式级创新候选)
experiments/innovation_brainstorm.md:3032:1. **短期**: 完成 exp249 (Small LGPA-D+GCN) → 可能 73-74% mAP
experiments/results.md:218:| 192 | OA-SD + CE (1-view, decay=0.99) | 62.6% | 74.9% | 86.0% | 89.1% | 🟡 vs exp191: -0.6/-0.5。decay 不敏感 |
experiments/results.md:219:| **193** | **OA-SD + 3-view + CE** | **64.4%** | **76.5%** | **86.3%** | **89.4%** | **✅ OA-SD+3-view additive! R1 ≈ exp187 SupCon, mAP +0.2/R1+0.9 vs exp190** |
experiments/results.md:220:| 194 | OA-SD + CE (weight=2.0) | 63.4% | 74.8% | 86.1% | 89.1% | 🟡 vs exp191 (w=1.0): +0.2/-0.6。weight 不敏感 |
experiments/results.md:224:| 198 | OA-SD + CE + STM (base, remote) | 63.2% | 75.2% | — | — | 🟡 = exp191 (无 STM)。STM 只加速不改善天花板 |
experiments/results.md:226:| 200 | CE + OA-RD (base, remote) | 62.9% | 73.9% | 85.2% | 88.5% | ❌ vs exp191 OA-SD: -0.3/-1.5。OA-RD 不如 OA-SD |
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
experiments/results.md:1122:- FSDC 正确配置仍然负面。**Feature completion 方向证伪。**
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
experiments/results.md:1361:### exp253: Tiny 3-Stage PSG (Stage1+2+3, 无 PAA) + LGPA-D+GCN
experiments/results.md:1365:| **exp253 FINAL** | **65.1%** | **76.2%** | 87.0% | 89.5% |
experiments/results.md:1366:| exp251 (2-stage+PAA) | 65.2% | 76.2% | 86.6% | 89.6% |
experiments/results.md:1367:| exp246b (1-stage) | 65.5% | 77.2% | — | — |
experiments/results.md:1371:- PAA 无贡献 (exp253 vs exp251 = -0.1/0.0)
experiments/results.md:1397:### exp255: Small GCN512 + 2-stage PSG + LGPA-D + OA-SD — NEW SMALL BEST
experiments/results.md:1399:| 方法 | mAP | R1 | R5 | R10 | vs exp249 |
experiments/results.md:1401:| **exp255 FINAL** | **73.2%** | **83.3%** | 90.4% | 92.3% | **+1.3/+1.5** |
experiments/results.md:1402:| exp255 MaxSim ep100 | 73.3% | 83.4% | — | — | +0.2/+0.3 (同epoch) |
experiments/results.md:1405:- exp255b (GCN512 + 1-stage): ≈ baseline — 2-stage PSG 是 GCN512 发挥的关键!
experiments/results.md:1406:- **exp255 seed42 FINAL: 73.1/83.1** (vs seed1234 73.2/83.3 = -0.1/-0.2)
experiments/results.md:1407:- **exp255 seed2024 FINAL: 72.6/82.0** (vs seed1234 73.2/83.3 = -0.6/-1.3)
experiments/results.md:1410:### exp256: Pose Prompt (KPR-style) — 负面/中性
experiments/results.md:1414:| exp256 (GCN512+2stage+Prompt, 进行中) | ep90: 72.4 | 82.3 | — | -0.5 vs exp255 |
experiments/results.md:1415:| **exp256b (GCN256+1stage+Prompt) FINAL** | **68.8** | **79.3** | **70.3/81.0** | **-3.1 vs exp249** |
experiments/results.md:1419:- **exp256 FINAL: 72.7/82.4** (vs exp255 73.2/83.3 = -0.5/-0.9)
experiments/results.md:1422:### exp257: ArcFace + Label Smoothing — 负面
experiments/results.md:1424:| 变体 | mAP | R1 | vs exp255 | 备注 |
experiments/results.md:1426:| exp257 (ArcFace m=0.35+LS, 远程) | 59.1% | 76.5% | -14.1/-6.8 | ep55 终止, ArcFace 严重不收敛 |
experiments/results.md:1427:| exp257b (Label Smooth only, 本地) | 71.5% | 81.7% | -1.7/-1.6 | ep86 终止, LS 稳定负面 |
experiments/results.md:1432:### exp258: ArcFace m=0.2 / GCN 3-layer — 负面/中性
experiments/results.md:1434:| 变体 | mAP | R1 | vs exp255 | 备注 |
experiments/results.md:1436:| exp258 (ArcFace m=0.2, 本地) | 67.7% | 81.2% | **-5.5/-2.1** | ArcFace 证伪 |
experiments/results.md:1437:| exp258b (GCN 3-layer, 远程) | 73.1% | 82.7% | -0.1/-0.6 | GCN 3-layer ≈ 2-layer |
experiments/results.md:1439:- ArcFace m=0.2: 比 m=0.35 好但仍 -5.5 mAP。ArcFace 在 Swin+SOLIDER pretrained 上完全证伪。
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
experiments/results.md:1478:| **exp260 FINAL** | **72.6%** | **81.6%** | — | 92.5% | **-0.6/-1.7** |
experiments/results.md:1481:- exp260 MaxSim+flip: 74.7/84.6 (仍低于 Small 75.2/85.6)
experiments/results.md:1483:### exp260b: Base GCN512 + 2-stage PSG (LR=8e-4) — 超越 Small!
experiments/results.md:1485:| 方法 | mAP | R1 | R5 | R10 | vs exp255 (Small) |
experiments/results.md:1487:| **exp260b FINAL** | **73.9%** | **83.2%** | — | — | **+0.7/-0.1** |
experiments/results.md:1488:| exp260b MaxSim+flip ep100 | 75.4% | 84.9% | — | — | +0.2/-0.7 (非final) |
experiments/results.md:1492:- **exp260b MaxSim+flip FINAL: 75.4/84.8** (vs Small 75.2/85.6 = +0.2/-0.8)
experiments/results.md:1495:### exp260b Market: Base GCN512 + 2-stage PSG (LR=8e-4, 无PLBOA)
experiments/results.md:1499:| **exp260b Market FINAL** | **94.4%** | **97.1%** | — | 99.4% | Base backbone |
experiments/results.md:1503:| exp260b Market MaxSim+flip | 94.7% | 97.2% | — | — | |
experiments/results.md:1504:| exp260b Market→Occluded-ReID (eq) | 86.0% | 88.5% | 95.3% | 97.9% | 跨数据集 equal_concat |
experiments/results.md:1505:| **exp260b Market→Occluded-ReID (MaxSim+flip)** | **88.0%** | **90.6%** | — | — | **跨数据集 MaxSim+flip** |
experiments/results.md:1511:> 机器: srvA/B/C = 5060 Ti 16G。本地 3090 挂，Base 3 run (exp263/266/269) DEFERRED。
experiments/results.md:1515:| exp261 | Swin-Tiny | Occ-Duke | **65.9% / 77.4%** | TBD | ✓ e120 FINAL @ 2026-04-19 04:16 srvB |
experiments/results.md:1516:| exp262 | Swin-Small | Occ-Duke | **73.8% / 83.1%** | TBD | ✓ e120 FINAL @ 2026-04-19 09:59 srvA (R5=90.2 R10=92.2). **略优 KPR w/o prompt 73.3/82.5** (+0.5/+0.6) |
experiments/results.md:1517:| exp263 | Swin-Base | Occ-Duke | **e100 eff FINAL: 72.5 / 81.8 (Global+flip), 74.5 / 84.0 (MaxSim+flip)** | ✓ @ 2026-04-20 09:01 srvB | ⚠️ e100 eval OOM-killed (内存 13.2G 触 16G),ckpt 100 完整,不重训。MaxSim hybrid+flip **74.5/84.0** 超 KPR w/o prompt +1.2/+1.5 |
experiments/results.md:1518:| exp263c | Swin-Base | Occ-Duke | ~~abandoned @ e31~~ | — lab3090 pwrlim280 seed 42 | seed 42 轨迹异常 (e10 2.7 / e20 17.0),用户指示换 seed 41 → 切 exp263d |
experiments/results.md:1519:| exp263d | Swin-Base | Occ-Duke | **74.1 / 83.3** | ✓ e120 FINAL @ 2026-04-21 14:27 lab3090 pwrlim 280W (R5=90.8 R10=93.0). **vs exp263 old e100 eff 72.5/81.8 Δ=+1.6/+1.5**. exp263 系列 PRCV 主表用此数字 (seed 41 替代 seed 42) |
experiments/results.md:1520:| exp263b | Swin-Base | Occ-Duke (seed 42 restart full 120) | **73.5 / 81.5 (train eq_concat), 74.8 / 84.0 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-23 16:47:17 lab4090 4090 TEST.IMS_PER_BATCH 64 (R5=90.2 R10=92.3). MaxSim Global 72.4/81.4, hybrid 74.8/84.0 (+1.3/+2.5 vs eq_concat)。**vs exp263 old e100 eff 72.5/81.8 (eq) / 74.5/84.0 (MaxSim) Δ=+0.3/0** (MaxSim 侧 full 120 微优)。vs exp263d s41 MaxSim 75.2/84.8 Δ=-0.4/-0.8 (**seed 41 > seed 42 再次 confirmed**)。论文 Base OD 主表仍用 exp263d (seed 41 最强), exp263b 作 seed 42 full 120 复现点 |
experiments/results.md:1521:| exp294 | Swin-Base | Occ-Duke (LGPA-only / Full-GCN s41) | **74.0 / 82.6 (eq+flip), 75.0 / 84.4 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-24 02:18:48 lab4090 TEST.IMS_PER_BATCH 64 (R5=90.5 R10=92.4). Global 73.5/83.3, **MaxSim 75.0/84.4** (+1.0/+1.8 vs eq_concat)。**vs exp263d Full+GCN s41**: eq 74.1/83.3 → -0.1/-0.7, **MaxSim 75.2/84.8 → -0.2/-0.4** (GCN 冗余双评测模式都验证)。vs exp263b Full+GCN s42 MaxSim 74.8/84.0: **+0.2/+0.4** (Full-GCN s41 > Full+GCN s42)。补 Phase 3-C Base 行, **3-backbone 统一结论 GCN 可移除** |
experiments/results.md:1522:| exp264 | Swin-Tiny | Occ-PTrack | **76.7% / 85.1%** | TBD | ✓ e120 FINAL @ 2026-04-19 07:15 srvC (R5=94.1 R10=97.0) |
experiments/results.md:1523:| exp265 | Swin-Small | Occ-PTrack | **78.4% / 86.2%** | TBD | ✓ e120 FINAL @ 2026-04-20 04:45 srvC (R5=94.8 R10=97.3, Small >> Tiny 76.7/85.1) |
experiments/results.md:1524:| exp266 | Swin-Base | Occ-PTrack | **e60 eff FINAL: 78.4 / 86.2 (peak e50: 78.5/86.3)** | ✓ @ 2026-04-20 21:27 srvC | ⚠️ e70 后 silent exit (非 OOM 非 CUDA, 推测 hy-tmp 平台 kill)。**Base 对 Small (exp265 78.4/86.2) 0 增益**, 不重训 |
experiments/results.md:1525:| exp265b | Swin-Small | Occ-PTrack (seed 41) | **78.5% / 85.9%** | ✓ e120 FINAL @ 2026-04-22 09:03 srvA 5060Ti (R5=94.7 R10=97.1) | **vs exp265 s42 78.4/86.2 Δ=+0.1/-0.3**。seed 41 微优 mAP 略弱 R1, 论文主表仍用 exp265 s42 (更高 R1), exp265b 作跨 seed 鲁棒性 supplementary |
experiments/results.md:1526:| exp266b (srvA) | Swin-Base | Occ-PTrack (seed 41) | **78.7% / 86.3%** | ✓ e120 FINAL @ 2026-04-23 13:18:50 srvA 5060Ti TEST.IMS_PER_BATCH 128 (R5=94.5 R10=97.1). **vs exp266b_3090 s41 78.5/86.2 Δ=+0.2/+0.1** (srvA 5060Ti 微优, 跨设备方差 0.2)。vs exp266 s42 e60 eff 78.4/86.2 Δ=+0.3/+0.1。vs exp265b Small s41 78.5/85.9 Δ=+0.2/+0.4 (**Base vs Small 同 s41 首次 R1 显著领先**)。**论文 Base OP 主表更新用此数字 78.7/86.3** (替代原 78.5) |
experiments/results.md:1527:| exp266b_3090 | Swin-Base | Occ-PTrack (seed 41) | **78.5% / 86.2%** | ✓ e120 FINAL @ 2026-04-22 09:29 lab3090 pwrlim 280W (R5=94.4 R10=96.9). **vs exp266 s42 e60 eff 78.4/86.2 Δ=+0.1/0** (持平)。vs exp265 Small 78.4/86.2 Δ=+0.1/0。vs exp265b Small s41 78.5/85.9 Δ=0/+0.3 |
experiments/results.md:1528:| exp267 | Swin-Tiny | Market | **92.5% / 96.4%** | TBD | ✓ e120 FINAL @ 2026-04-19 13:45 srvB (R5=98.9 R10=99.3) |
experiments/results.md:1529:| exp268 | Swin-Small | Market | **94.3% / 97.3%** | TBD | ✓ e120 FINAL @ 2026-04-20 00:39 srvA (R5=99.1 R10=99.5) |
experiments/results.md:1530:| exp269 | Swin-Base | Market | **e80 eff FINAL: 94.4 / 97.0 (Global+flip), 94.5 / 97.1 (MaxSim+flip)** | ✓ @ 2026-04-20 13:xx srvA | ⚠️ e80 eval OOM-killed 同 exp263 模式,ckpt80 完整,不重训。Base 对 Small 优势小(Market 已饱和) |
experiments/results.md:1531:| exp269b | Swin-Base | Market (seed 42 restart full 120, PLBOA OFF) | **94.5 / 97.2 (eq+flip), 94.6 / 97.2 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-24 01:17:24 srvC 5060Ti TEST.IMS_PER_BATCH 64 (R5=99.1 R10=99.5). Global+flip 94.4/97.1, **MaxSim 94.6/97.2** (+0.1 mAP vs eq_concat)。**vs exp269 orig e80 eff**: eq 94.4/97.0 → +0.1/+0.2; MaxSim 94.5/97.1 → +0.1/+0.1。vs exp268 Small 94.3/97.3 Δ=+0.2/-0.1。vs exp293b Base PLBOA ON 93.8/97.2 Δ=+0.7/0 (**PLBOA 净 -0.7 mAP**)。**论文 Market Base 主数字升级 eq 94.5/97.2 / MaxSim 94.6/97.2** |
experiments/results.md:1539:| exp270 | Swin-Tiny | Occ-Duke | 无 (baseline) | **59.2 / 68.4** | ✓ e120 FINAL @ 2026-04-20 12:29 srvB (R5=82.2 R10=85.8). vs exp000 旧协议 56.6/66.5 → +default flip 贡献 +2.6/+1.9 |
experiments/results.md:1540:| exp271 | Swin-Tiny | Occ-Duke | `[-1]` (1-stage) | **60.2 / 69.5** | ✓ e120 FINAL @ 2026-04-20 16:36 srvB (R5=81.8 R10=85.9). vs exp270 Δ=+1.0/+1.1 = stage 3 PSG 独立贡献 |
experiments/results.md:1541:| exp272 | Swin-Tiny | Occ-Duke | `[-2,-1]` (2-stage) | **60.5 / 69.7** | ✓ e120 FINAL @ 2026-04-20 20:19 srvB (R5=82.6 R10=86.2). vs exp271 Δ=+0.3/+0.2 = stage 2 边际贡献微弱;vs exp270 Δ=+1.3/+1.3 = 2-stage 累计 |
experiments/results.md:1542:| exp273 | Swin-Tiny | Occ-Duke | `[-3,-2,-1]` (3-stage) | **60.5 / 69.9** | ✓ e120 FINAL @ 2026-04-21 00:05 srvB (R5=82.8 R10=87.0). vs exp272 2-stage Δ=0/+0.2 (stage 1 边际贡献 ~0 mAP). **Phase 3-A Tiny 矩阵完整**: 边际收益递减 +1.0 → +0.3 → 0 |
experiments/results.md:1543:| exp274 | Swin-Small | Occ-Duke | 无 (baseline) | **68.1 / 76.8** | ✓ e120 FINAL @ 2026-04-20 21:34 lab4090 (R5=87.8 R10=90.9). vs Tiny exp270 Δ=+8.9/+8.4 = Small vs Tiny backbone 容量差 |
experiments/results.md:1544:| exp275 | Swin-Small | Occ-Duke | `[-1]` (1-stage) | **68.8 / 76.8** | ✓ e120 FINAL @ 2026-04-20 23:37 lab4090 (R5=87.2 R10=90.4). vs exp274 no-PSG Δ=**+0.7/0** (mAP 涨 R1 持平). vs Tiny 1-stage 增益 (+1.0/+1.1),Small 上 +0.7/0 缩水 |
experiments/results.md:1545:| exp276 | Swin-Small | Occ-Duke | `[-2,-1]` (2-stage) | **68.3 / 77.2** | ✓ e120 FINAL @ 2026-04-21 01:41 lab4090 (R5=87.2 R10=90.1). vs exp275 1-stage Δ=-0.5 mAP/+0.4 R1 (**Small 上 2-stage 不同 Tiny,mAP 不涨但 R1 涨**) |
experiments/results.md:1546:| exp277 | Swin-Small | Occ-Duke | `[-3,-2,-1]` (3-stage) | ~~49.0 / 57.7 (seed 42 偶发塌缩)~~ | abandoned @ 2026-04-21 03:47 (e2 id_global 卡 3.277 classifier uniform). **改 exp277b seed 41 重跑** (用户判断偶发) |
experiments/results.md:1547:| exp277b | Swin-Small | Occ-Duke | `[-3,-2,-1]` (3-stage) | **68.3 / 77.6** | ✓ e120 FINAL @ 2026-04-21 23:34 lab4090 (R5=87.4 R10=89.8). **R1 最强 Phase 3-A Small!** vs exp277 s42 塌缩 49.0/57.7 Δ=+19.3/+19.9. vs exp276 2-stg 68.3/77.2 Δ=0/+0.4. **seed 41 完全验证 exp277 塌缩是偶发** |
experiments/results.md:1555:| exp281 (= exp261) | Swin-Tiny | 512 | `[-2,-1]` | **65.9 / 77.4** | Phase 1 共享,不重跑 |
experiments/results.md:1556:| exp278 | Swin-Tiny | 256 | `[-1]` | **65.7 / 76.7** | ✓ e120 FINAL @ 2026-04-21 10:42 srvB (R5=86.7 R10=89.6). vs exp261 GCN512+2stg 65.9/77.4 Δ=-0.2/-0.7. vs exp286 LGPA-only 66.0/76.6 Δ=-0.3/+0.1 (GCN256 略弱于 no GCN) |
experiments/results.md:1557:| exp279 | Swin-Tiny | 256 | `[-2,-1]` | **65.7 / 76.9** | ✓ e120 FINAL @ 2026-04-21 21:32 srvB (R5=86.6 R10=90.1). vs exp278 GCN256+1stg 65.7/76.7 Δ=0/+0.2 (mAP 持平 R1 +0.2). vs exp261 65.9/77.4 Δ=-0.2/-0.5 |
experiments/results.md:1558:| exp280 | Swin-Tiny | 512 | `[-1]` | **65.7 / 76.2** | ✓ e120 FINAL @ 2026-04-22 08:07 srvB (R5=86.7 R10=89.7). **vs exp261 GCN512+2stg 65.9/77.4 Δ=-0.2/-1.2** (最弱 R1 格), vs exp278 GCN256+1stg 65.7/76.7 Δ=0/-0.5. **Phase 3-B Tiny 2×2 闭合: GCN256+1stg=GCN256+2stg=GCN512+1stg mAP 全 65.7, GCN512+2stg 唯一 65.9**。和 Small 2×2 GCN512+1stg 最弱同模式 |
experiments/results.md:1559:| exp285 (= exp262) | Swin-Small | 512 | `[-2,-1]` | **73.8 / 83.1** | Phase 1 共享, srvA 5060Ti (原始), 已 re-eval flip fix 后 73.8/83.1 no-op |
experiments/results.md:1560:| exp285b | Swin-Small | 512 | `[-2,-1]` | **73.8 / 83.8** | ✓ e120 FINAL @ 2026-04-22 06:04 lab4090 (R5=90.7 R10=92.7). **vs exp262 (srvA old) 73.8/83.1 Δ=0/+0.7** (mAP 持平, R1 +0.7 lab4090 > srvA). **Phase 3-B Small 矩阵 gold-standard**, 论文主表用此数字 |
experiments/results.md:1561:| exp282 | Swin-Small | 256 | `[-1]` | **73.7 / 83.9** | ✓ e120 FINAL @ 2026-04-21 09:33 lab4090 (R5=90.5 R10=92.5). **vs exp262 73.8/83.1: mAP -0.1 R1 +0.8** → low-cap ≥ high-cap, Small Full Scaffold 容量饱和 |
experiments/results.md:1562:| exp283 | Swin-Small | 256 | `[-2,-1]` | **73.5 / 83.2** | ✓ e120 FINAL @ 2026-04-21 15:38 lab4090 (R5=90.7 R10=92.5). vs exp262 73.8/83.1 Δ=-0.3/+0.1. vs exp282 73.7/83.9 Δ=-0.2/-0.7 |
experiments/results.md:1563:| exp284 | Swin-Small | 512 | `[-1]` | **73.4 / 82.9** | ✓ e120 FINAL @ 2026-04-21 21:23 lab4090 (R5=89.9 R10=92.2). vs exp262 73.8/83.1 Δ=-0.4/-0.2. **Phase 3-B Small 2x2 完整: GCN256+1stg (83.9) 最 R1, GCN512+2stg (73.8 mAP) 最 mAP; GCN512+1stg 反而最弱** |
experiments/results.md:1567:> Phase 3-C: **LGPA-only + 变量 PSG_STAGES** (关 GCN, 保留 LGPA/OA-SD/ParAug/LOWER_BODY_OCC)。回答"2-stage PSG 的收益是偏 structural 还是 semantic branch 也吃"。srvC exp266 silent exit 后空闲,利用上。
experiments/results.md:1571:| exp286 | Swin-Tiny | `[-1]` | **66.0 / 76.6** | ✓ e120 FINAL @ 2026-04-21 10:03 srvC (R5=86.4 R10=89.7). **vs exp261 Full Scaffold 65.9/77.4 Δ=+0.1/-0.8** → GCN 对 Tiny 几乎无贡献, LGPA-only 等价 Full |
experiments/results.md:1572:| exp287 | Swin-Tiny | `[-2,-1]` | **65.9 / 77.0** | ✓ e120 FINAL @ 2026-04-21 20:48 srvC (R5=87.0 R10=89.7). vs exp286 LGPA-only 1stg 66.0/76.6 Δ=-0.1/+0.4 (2-stg R1 微优). vs exp261 Full 65.9/77.4 Δ=0/-0.4 (GCN 主要给 R1) |
experiments/results.md:1573:| exp288 | Swin-Small | `[-1]` | **73.8 / 83.8** | ✓ e120 FINAL @ 2026-04-22 12:51 srvC (R5=90.5 R10=92.0). 🔥 **vs exp285b Full Scaffold 73.8/83.8 完全持平** (mAP/R1 identical, R5/R10 微差 0.2/0.7)。vs exp282 Full GCN256+1stg 73.7/83.9 Δ=+0.1/-0.1。**证实 GCN 对 Small OD 零贡献**, LGPA 单独达 Full Scaffold 性能 |
experiments/results.md:1574:| exp289 | Swin-Small | `[-2,-1]` | **73.8 / 83.3** | ✓ e120 FINAL @ 2026-04-23 05:39 srvC (R5=90.5 R10=92.4). **vs exp288 1-stg 73.8/83.8 Δ=0/-0.5**, vs exp285b Full Scaffold 73.8/83.8 Δ=0/-0.5 — **mAP 完全持平 Full Scaffold, GCN 零贡献 reconfirmed**. 和 Tiny Phase 3-C (exp287 2-stg 65.9/77.0 vs exp286 1-stg 66.0/76.6) 方向相反 (Small 1-stg R1 微优, Tiny 2-stg R1 微优), 但 mAP 均持平 |
experiments/results.md:1580:| exp290 | Swin-Small | Occ-PTrack | 42 | **78.4 / 86.2** | ✓ e120 FINAL @ 2026-04-23 09:22 srvB (R5=94.8 R10=97.4). 🔥 **严格持平 exp265 scene baseline 78.4/86.2/94.8/97.3** (Δ 0/0/0/+0.1). target-heatmap 3 数据集全 near-no-op, OP 多人场景预期增益未实现 |
experiments/results.md:1581:| exp291 | Swin-Small | Occ-Duke | 42 | **73.5 / 82.9** | exp285b 73.8/83.8 (Δ -0.3/-0.9) | ✓ e120 FINAL @ 2026-04-22 18:13 lab4090 (R5=90.7 R10=92.5). OD 多单人场景 near no-op, 机制无显著回归 |
experiments/results.md:1582:| exp292 | Swin-Small | Market | 42 | **e90 eff FINAL: 94.2 / 97.1** | exp268 FINAL 94.3/97.3 (Δ -0.1/-0.2 持平) | ✓ 停于 e93 @ 2026-04-22 23:25 用户让出 lab3090。R5 99.2 R10 99.5 = exp268 FINAL R5 99.1 R10 99.5 |
experiments/results.md:1583:| exp293 | Swin-Base | Market + **PLBOA** | 42 | **e120 FINAL (restart): 93.8 / 97.2** (完整 120ep) | exp269 e80 eff 94.4/97.0 (Δ -0.6/+0.2); first run e80 eff 94.1/96.9 (Δ -0.3/+0.3 跨 restart 方差) | ✓ restart full 120 @ 2026-04-23 08:24 lab4090 (R5=98.9 R10=99.5). First run e80 eval OOM, 重启 w/ TEST.IMS_PER_BATCH 64. **PLBOA 在 Market full 120 net -0.6 mAP / +0.2 R1** (vs exp269 PLBOA OFF e80) — 主表待 exp269b FINAL 公平对比 |
experiments/results.md:1585:## Post-PRCV 消融/复现/扫参 runs (exp295–321b, 2026-04-25~28)
experiments/results.md:1590:### exp295–304: 复现 / multi-seed / LR sweep / Phase 3-D LGPA 消融
experiments/results.md:1596:| exp295 | Swin-Small | Full Scaffold 复现 exp255 | lab4090 / 1234 | **74.2 / 84.0** | **75.2 / 85.4** | vs exp255 hist 75.2/85.6: **0 / -0.2** | ✅ 完全重现 exp255 75.2 mAP，证历史数字真实可复现（非 eval bug）。**Small OD 主表新 reference** |
experiments/results.md:1597:| exp296 | Swin-Base | LR 8e-4 复现 exp263d | lab4090 / 41 | 73.7 / 81.7 | 74.9 / 83.8 | vs exp263d 75.2/84.8: **-0.3 / -1.0** | reproducibility 接近但 R1 系统性偏低（lab4090 vs lab3090 硬件差）；主表仍用 exp263d |
experiments/results.md:1598:| exp297 | Swin-Base | **LR 4e-4** | srvA(5060Ti) / 41 | 73.2 / 82.4 | 74.6 / 84.1 | vs exp296 LR8: **-0.3 / +0.3**（近 tie） | LR4 vs LR8 接近持平，**非显著 underfit**；比 hist exp260 LR4(72.6) 高 0.6 mAP |
experiments/results.md:1599:| exp298 | Swin-Base | **LR 2e-4**（下界） | srvB(5060Ti) / 41 | 68.6 / 78.6 | 69.6 / 79.1 | vs exp296 LR8: **-5.3 / -4.7** | LR2 严重 underfit（e10 mAP 1.3 near-random），LR ablation 下界，证 LR8 不能再降 |
experiments/results.md:1600:| exp299 | Swin-Base | **PLBOA OFF** | srvC(5060Ti) / 41 | 70.9 / 78.0 | 72.7 / 80.5 | vs exp296 PLBOA ON: **-2.2 / -3.3** | OD 上 PLBOA net positive **+2.2 mAP MaxSim**；与 Market 上 PLBOA 有害形成 dataset-specific claim |
experiments/results.md:1601:| exp300 | Swin-Base | Full Scaffold seed 1234 | lab4090 / 1234 | 74.0 / 83.8 | 75.0 / 85.0（e100 ckpt 75.0/85.2） | vs exp263d 75.2/84.8: **-0.2 / +0.2**（e120） | 未破 exp263d SOTA mAP，但 R1 +0.2~0.4 微超；e100 ckpt R1 peak 85.2 |
experiments/results.md:1602:| exp301 | Swin-Small | **LGPA OFF**（Phase 3-D） | lab4090 / 42 | 71.9 / 83.0 | 71.9 / 83.0（MaxSim **0 boost**） | vs exp285b Full 74.7/84.8: **-2.8 / -1.8** | LGPA 贡献 +2.8 mAP MaxSim；移除 LGPA → MaxSim 失去 boost（LGPA 是 MaxSim 主驱动） |
experiments/results.md:1603:| exp302 | Swin-Base | Full Scaffold seed 42（multi-seed 第3） | srvA(5060Ti) / 42 | 73.3 / 81.4 | 74.4 / 83.6 | vs exp263d 75.2/84.8: **-0.8 / -1.2** | Base 3-seed(41/1234/42) MaxSim mAP mean **74.87 std 0.42**；主行仍用 exp263d |
experiments/results.md:1604:| exp303 | Swin-Tiny | **LR 4e-4** | srvB(5060Ti) / 41 | 64.4 / 74.8 | 65.7 / 76.1 | vs exp261 LR8 67.2/78.6: **-1.5 / -2.5** | Tiny LR4 underfit -1.5 mAP；LR8 仍 sweet spot（Tiny 比 Base 更 LR 敏感） |
experiments/results.md:1605:| exp304 | Swin-Small | Full Scaffold seed 2024（multi-seed 第3） | srvC(5060Ti) / 2024 | 73.3 / 82.7 | 74.3 / 84.0 | vs exp295 75.2/85.4: **-0.9 / -1.4** | Small 3-seed(42/1234/2024) MaxSim mAP mean **74.7 std 0.45**；主行仍用 exp295 |
experiments/results.md:1611:| exp305 | Swin-Tiny | **LGPA OFF**（mirror exp301） | lab4090 / 42 | 64.5 / 76.0 | 64.5 / 76.0（**0 boost**） | vs exp261 67.2/78.6: **-2.7 / -2.6** | LGPA 贡献 +2.7 mAP MaxSim（+1.4 eq）；Phase 3-D Tiny+Small 双 backbone 完整 |
experiments/results.md:1612:| exp307 | Swin-Tiny | **PLBOA OFF**（mirror exp299） | srvB(5060Ti) / 42 | 62.8 / 71.8 | 64.5 / 73.5 | vs exp261 67.2/78.6: **-2.7 / -5.1** | Tiny PLBOA net positive **+2.7 mAP**；与 Base(+2.2) 一致。PLBOA dataset-specific 2-backbone evidence |
experiments/results.md:1616:> commit `c059dca` 修复 GLOBAL_LOSS_SCALE 只在 no-part 路径生效的 bug（Full Scaffold 此前完全忽略，effective=1.0）。exp311+ 后 scale 才真在 part-path 生效。Tiny sweep seed 42 / baseline exp261(67.2/78.6)；Small 验证 seed 1234 / baseline exp295(75.2/85.4)。
experiments/results.md:1620:| exp311b | Swin-Small | **GLOBAL_LOSS_SCALE 0.5**（bugfix 后真生效） | lab4090 / 1234 | 73.5 / 83.2（e100 eff，e101 OOM） | 74.5 / 84.8 | vs exp295: **-0.7 / -0.6** | 0.5× global 真生效后 net **-0.7 mAP**，非有效改进；effective 1.0 更好 |
experiments/results.md:1621:| exp312 | Swin-Tiny | **GLOBAL_LOSS_SCALE 2.0** | lab4090 / 42 | 65.7 / 76.6 | 66.8 / 77.2 | vs exp261: **-0.4 / -1.4** | 2.0× 也 net negative。结合 exp311b(0.5×负)，**双向都负 → 1.0 sweet spot**（推翻早期 0.5） |
experiments/results.md:1622:| exp313 | Swin-Tiny | **POSE_PART_WEIGHT 2.0**（ID favor part） | srvA(5060Ti) / 42 | 65.8 / 77.0 | 66.9 / 77.9 | vs exp261: **-0.3 / -0.7** | favor part 微 negative |
experiments/results.md:1623:| exp314 | Swin-Tiny | **POSE_PART_WEIGHT 0.5**（ID favor global） | srvB(5060Ti) / 42 | 65.8 / 77.5 | 67.2 / 78.6 | vs exp261: **0 / 0**（完全相等） | favor global net neutral；default 1.0 双 sweet spot |
experiments/results.md:1624:| exp315 | Swin-Tiny | **POSE_LGPA_ASSIGN_WEIGHT 1.0**（LGPA aux ×2） | srvC(5060Ti) / 42 | 65.8 / 76.9 | 67.0 / 77.4 | vs exp261: **-0.2 / -1.2** | LGPA aux 加倍 net negative；default 0.5 sweet spot |
experiments/results.md:1625:| exp316 | Swin-Tiny | **POSE_OA_SD_WEIGHT 2.0** | lab4090 / 42 | 66.0 / 77.6 | 67.2 / 78.0 | vs exp261: **0 / -0.6** | OA-SD ×2 net neutral；default 1.0 sweet spot |
experiments/results.md:1626:| exp317 | Swin-Tiny | **POSE_LGPA_ASSIGN_WEIGHT 0.25**（LGPA aux ÷2） | lab3090 / 42 | 66.2 / 77.4 | 67.4 / 78.6 | vs exp261: **+0.2 / 0** ⭐ | sweep 中**唯一 MaxSim 超 baseline**(+0.2)，但在 multi-seed std 内，需 Small 验证 |
experiments/results.md:1627:| exp318 | Swin-Tiny | **POSE_PART_TRI_WEIGHT 0.5**（Tri favor global） | srvB(5060Ti) / 42 | 65.9 / 77.7 | 67.1 / 78.3 | vs exp261: **-0.1 / -0.3** | Tri-side favor global slight neg；与 exp314 合证 default 双 sweet spot |
experiments/results.md:1628:| exp319 | Swin-Tiny | **POSE_OA_SD_WEIGHT 0.5** | srvC(5060Ti) / 42 | 65.8 / 76.8 | 67.1 / 78.1 | vs exp261: **-0.1 / -0.5** | OA-SD ÷2 slight neg；与 exp316(×2) 合证 default 1.0 sweet spot |
experiments/results.md:1629:| exp320 | Swin-Small | **POSE_LGPA_DETACH=False**（LGPA aux 反传 backbone） | lab4090 / 1234 | 68.1 / 79.3 | 68.8 / 79.6 | vs exp295: **-6.4 / -5.8** | **catastrophic -6.4 mAP**（e10 46% underfit）；证 LGPA detach 必要。强 negative 消融素材 |
experiments/results.md:1630:| exp321b | Swin-Small | **POSE_LGPA_ASSIGN_WEIGHT 0.25**（验证 exp317） | lab4090 / 1234 | 73.9 / 83.7 | 74.9 / 85.4 | vs exp295: **-0.3 / 0** | Tiny exp317 的 +0.2 **未迁移到 Small**（slight -0.3）→ 判 seed noise，保持 default 0.5 |
experiments/results.md:1632:> 跳号说明：exp306/308/309/310/321a/321c 无目录（实验号跳过/未跑，非数据丢失）。exp311(s42) e10 即被 kill，以 exp311b(s1234) 计入。exp296/exp302 R1 跨设备系统性偏低 1-1.6（5060Ti/lab4090 vs lab3090），主表用 lab3090 exp263d 不受影响。
experiments/results.md:1636:> post-PRCV「搬范式」首验。Frozen Qwen2.5-VL-3B（lab-3090-d, RTX 3090），288 个重遮挡难例 pair（均衡 144 同/144 异，chance=50%）。三条件：甲(裸全图)、乙(裸图+可见部位文字)、丙(姿态视觉裁剪图)。**非训练端创新，不计入主表增益。**
experiments/results.md:1654:> post-PRCV「搬范式」#2 路线。frozen DINOv2-base（lab-3090-d, RTX 3090），全量 Occluded-Duke（2210 query × 17661 gallery，无后处理、无训练）。脚本 `scripts/exp324_dino.py`。输入 224W×448H → patch grid 32×16。keypoints 缩放到 grid → 每部位 3×3 窗均值池化成 5 个 part 向量 + per-part visibility，跨图只比 mutually-visible part 的 per-part cosine（part-MaxSim）。重遮挡子集 = query visibility_binary.sum()≤8（989/2210）。**training-free，不计入主表增益。**
experiments/results.md:1754:| **exp349** | 强系统 exp255(73.2) + CLIP | **71.4/71.3**(eq/global) | **CLIP有害 -1.8** |
experiments/decisions.md:517:2. 与 PXA (cross-attention) 不同，CAPSG 保持了 PSG 的逐元素乘法范式，只是让乘法因子变成 content-dependent
experiments/decisions.md:713:  B. 全新范式（PGFS 硬 token 选择等）
experiments/decisions.md:723:5. 全新范式（20 个 PSG 改进全失败的历史）成功概率不高
experiments/decisions.md:742:  B. 回归 PSG 路线但走全新范式（如 Pose-Guided Feature Selection — 基于热图做 token pruning/routing）
experiments/decisions.md:750:4. 这个方向有更强的论文 story：从"全局调制"到"局部选择"的范式升级
experiments/decisions.md:766:- 🔴 红队（方案 A: LPQ）核心论点: Part 分支失败不是方向错误而是方法不对。29 个实验中从未在梯度隔离的 Part 分支测试过 cross-attention 查询机制。LPQ 从"被动聚合"到"主动查询"的范式转换，与热图 mask 池化有本质区别。成本仅 2-3 天，StopGrad 提供安全网（global-only 59.5% 下限不变）。论文最大软肋是 Part 特征无效，LPQ 可能补全这个 gap。信心: 6/10
experiments/decisions.md:1118:**论文范式定义**: Occluded ReID = partial-set-to-partial-set matching, 不是 vector-to-vector matching
experiments/decisions.md:1138:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/decisions.md:1152:3. 连续 3 条不同范式（data augmentation / structural / adversarial）全部失败，排除了实验噪声
experiments/decisions.md:1175:- 需要继续思考真正的范式级创新
experiments/decisions.md:1531:1. 连续 3 个辅助 loss 实验（CSGT/SGMKC/PAMC）分别代表了 mining filter、自监督重建、一致性正则化这三种完全不同的辅助 loss 范式，全部失败/中性
experiments/decisions.md:1535:**下一步**: 进入文献学习，寻找不是"加辅助 loss"而是"改变特征提取或匹配范式"的新方向
experiments/decisions.md:1576:- 🔵 蓝队（方案 B: PPE）核心论点: 重新定义遮挡问题（从 visibility mask 到 uncertainty quantification），pose confidence → feature uncertainty 的因果链是全新机制。论文理论深度更强（概率嵌入+信息几何）。与 PSG 正交互补（PSG 管特征提取，PPE 管匹配）。P3E/DUL 已验证概率嵌入在 ReID 中可训练。攻击 A: PAB 已在同范式失败（attention bias 变体），CLAUDE.md 警告"attention trick"，2.8K 参数太小难以撑论文主贡献，12x4 分辨率太粗糙。信心: 7/10
experiments/decisions.md:1637:4. 论文 story：PSG (backbone 注入) + Cross-Attention Decoder (结构化解码) 形成完整的 encode-decode 范式
experiments/decisions.md:1684:3. 它没有被 `exp070` 直接证伪，因为 `exp070` 试的是 hard switch，不是 `scene + target-distractor` 的联合机制。
experiments/decisions.md:1705:1. "recover" 是与 "suppress/inject/select" 完全不同的范式
experiments/decisions.md:1733:1. `target/distractor ambiguity` 这个问题定义本身还没有被证伪。
experiments/decisions.md:2395:  - `exp089 PAMN` 只有 design/review 草案，从未真正接入 checkpoint 与测试检索流程，因此**不能**算作“learned pair module 已被证伪”
experiments/decisions.md:2447:2. 这不等于 learned pair module 大方向被证伪；真正被证伪的是更具体的实现：
experiments/decisions.md:2534:   - 不得再引用其数值作为方法证据
experiments/decisions.md:2920:  - 真正 headroom 来自 `single-image support incomplete`
experiments/decisions.md:2961:1. feature-level completion 方向已被彻底证伪（5+ 次尝试），不值得继续做 ablation
experiments/decisions.md:2980:1. `single-image support incomplete` 这个问题定义没有被推翻
experiments/decisions.md:3046:**上下文**: exp190-195 系列实验完成，揭示了 OA-SD 和 3-view parallel aug 的组合关系，以及 OA-SD global-only 解决 SupCon 梯度冲突的新机制。
experiments/decisions.md:3049:- exp190 (3-view+CE): 64.2/75.6 — 3-view 是最强单一技术
experiments/decisions.md:3050:- exp191 (OA-SD+CE): 63.2/75.4 — OA-SD 独立有效
experiments/decisions.md:3051:- exp192 (decay=0.99): 62.6/74.9 — decay 不敏感
experiments/decisions.md:3052:- exp193 (3-view+OA-SD+CE): 64.4/76.5 — additive! R1 追平 SupCon
experiments/decisions.md:3053:- exp194 (weight=2.0): 63.4/74.8 — weight 不敏感
experiments/decisions.md:3054:- exp195 (SupCon+OA-SD global-only): ep70=60.2/73.4 — 梯度冲突解决!
experiments/decisions.md:3057:  A. exp196: 3-view + SupCon + OA-SD global-only（终极组合，验证所有创新 additive）
experiments/decisions.md:3064:2. 如果 exp196 > exp187 (64.9/76.6)，则创论文主表新高
experiments/decisions.md:3065:3. exp195 已验证 SupCon+OA-SD global-only 兼容，exp193 已验证 3-view+OA-SD additive
experiments/decisions.md:3072:**上下文**: exp196 (3-view + SupCon + OA-SD global-only) 在 ep70 持续落后 exp187 (3-view + SupCon) -1.8/-0.9。OA-SD global-only 的 distillation 信号 (oa_sd=0.01) 过弱。
experiments/decisions.md:3077:- OA-SD + SupCon (global-only) 无梯度冲突但信号太弱 (exp195/196)
experiments/decisions.md:3082:  B. 最终配置用 OA-SD+CE (exp193: 64.4/76.5) — R1 几乎一样
experiments/decisions.md:3090:3. OA-SD 路线: R1 几乎一样，且 OA-SD 概念更新颖 (self-distillation 范式)
experiments/decisions.md:3091:4. 论文可以展示："两条互补但互斥的训练范式"
experiments/decisions.md:3113:**上下文**: exp199 (OA-RD+SupCon) ep60=-1.5/-3.4 vs exp187，exp200 (OA-RD+CE) ep60=-1.1/-3.4 vs exp191。OA-RD (relational distillation) 也是负结果。
experiments/decisions.md:3117:- OA-RD (relation-level): exp199 失败
experiments/decisions.md:3137:**上下文**: exp197-201 连续 5 个负结果。所有在 exp187 (64.9/76.6) 基础上的改进尝试都失败。
experiments/decisions.md:3140:- exp197 (STM + SupCon): -0.8/-0.6 — token mixup 只加速不改善
experiments/decisions.md:3141:- exp198 (STM + OA-SD): ±0 — 同上
experiments/decisions.md:3142:- exp199 (OA-RD + SupCon): -1.5/-2.1 — relational distillation 也与 SupCon 冲突
experiments/decisions.md:3143:- exp200 (OA-RD + CE): -0.3/-1.5 — OA-RD 不如 OA-SD
experiments/decisions.md:3144:- exp201 (global SupCon): ~-1.5/-3.6 — global SupCon 压缩特征空间
experiments/decisions.md:3175:| +2 | **Swin-Base** (exp207 进行中) | 74-75% |
experiments/decisions.md:3180:1. exp207 Base 跑完后确认 Base 增益
experiments/decisions.md:3185:- exp208 (0.5x global loss) = NO-OP（GCN list-loss 已隐含 0.5x），取消
experiments/decisions.md:3186:- exp209 (STD-PR+CE+OA-SD) ep30=56.0/69.3，落后 5%，终止
experiments/decisions.md:3192:**上下文**: MaxSim hybrid 在 exp206 checkpoint 上无需重训给 +1.8% mAP (70.3→72.1)。OA-SD teacher bug 已修复。PKC (Per-Keypoint Contrastive) 开始测试。
experiments/decisions.md:3199:| exp210 | + PKC (进行中) | 73-74% |
experiments/decisions.md:3200:| exp207 | Swin-Base 3-view (进行中) | 74-76% |
experiments/decisions.md:3211:| exp210 | PKC weight=0.5 (detached GCN) | 灾难 (3.6%) |
experiments/decisions.md:3212:| exp210b | PKC weight=0.05 (detached GCN) | 无效 (= baseline) |
experiments/decisions.md:3213:| exp211 | MST weight=0.5 (detached GCN) | 无效 (= baseline, 所有 loss 完全一致) |
experiments/decisions.md:3214:| exp213 | PKC+MST 组合 (detached) | 灾难 (40.6%) |
experiments/decisions.md:3215:| exp215 | BA-PKC weight=0.1 (NON-detached backbone) | 灾难 (0.5%) |
experiments/decisions.md:3222:1. **per-keypoint loss 路线已证伪** — 架构约束使其不可能有效
experiments/decisions.md:3224:3. **当前最佳: 72.4/83.1 (exp210b + maxsim)**
experiments/decisions.md:3240:| exp210 | PKC w=0.5 on detached GCN | Yes | 灾难 3.6% |
experiments/decisions.md:3241:| exp210b | PKC w=0.05 on detached GCN | Yes | 无效 (=baseline) |
experiments/decisions.md:3242:| exp211 | MST w=0.5 on detached GCN | Yes | 无效 (所有 loss 完全一致) |
experiments/decisions.md:3243:| exp213 | PKC+MST combo | Yes | 灾难 40.6% |
experiments/decisions.md:3244:| exp215 | BA-PKC w=0.1 non-detached | No | 灾难 0.5% |
experiments/decisions.md:3245:| exp212 | LR=0.0008 | — | 灾难 0.8% |
experiments/decisions.md:3246:| exp217 | OERL w=1.0 non-detached cosine | No | `62.2/75.2`，相对 `exp191 63.2/75.4` 为 `-1.0/-0.2` |
experiments/decisions.md:3252:4. **per-keypoint training loss 路线已全面证伪**
experiments/decisions.md:3259:### [2026-04-02 09:45] 决策 — PACI 证伪 + MaxSim Ceiling 发现
experiments/decisions.md:3261:**PACI (exp218/219) 结果:**
experiments/decisions.md:3262:- PACI + OA-SD (exp218): `61.9 / 74.2` (vs `exp191 63.2 / 75.4` = **-1.3 / -1.2**)
experiments/decisions.md:3263:- PACI-only (exp219): 已从远程补回 `train_log`，当前可直接复核到 `ep10=37.7/50.5`、`ep20=47.5/60.4`、`ep30=51.9/64.9`；但尚无 final，因此它仍只能作为 early stop-loss 证据，不能写成正式最终结果
experiments/decisions.md:3264:- **PACI 证伪。Consistency loss on detached GCN = 无效。**
experiments/decisions.md:3285:4. 后续 `exp220` 已把 Tiny `maxsim_hybrid` 推到 `64.6`，因此这里原先的 `~64.4` / `~64.2` ceiling 表述应视为阶段性误判
experiments/decisions.md:3289:2. **或者找到不依赖 detach/non-detach 的全新训练范式**
experiments/decisions.md:3315:### [2026-04-03 20:40] 决策 — BT-PKD 系列证伪，Non-Detached Gradient 方向关闭
experiments/decisions.md:3317:**上下文**: exp229-232 全面测试了 BT-PKD (Backbone-Through Per-Keypoint Distillation):
experiments/decisions.md:3330:**已证伪的 non-detached 变体汇总**:
experiments/decisions.md:3340:### [2026-04-04 15:40] 决策: exp242 PPA+GCN Small 灾难性失败
experiments/decisions.md:3347:2. 对比: PPA on Small (exp240) 也是中性 (70.7/81.1 vs 70.6/82.6 = +0.1/-1.5)
experiments/decisions.md:3355:### [2026-04-04 15:40] 决策: 启动 exp243 LGPA
experiments/decisions.md:3357:**上下文**: 寻找范式级创新, 结合 VLM + pose 做 part assignment。
experiments/decisions.md:3365:### [2026-04-04 21:10] exp243 LGPA 结果分析 (GPU crash at ep88)
experiments/decisions.md:3379:### [2026-04-05 04:10] exp244 LGPA-Detach — 突破性结果! ⭐⭐⭐
experiments/decisions.md:3381:**结果**: 65.3/75.7 (+2.1/+0.3 vs exp191 GCN+OA-SD)
experiments/decisions.md:3387:2. detach 完全消除了 non-detach 的后期干扰 (exp243 -1.1 → exp244 +2.1)
experiments/decisions.md:3393:- 消融故事清晰: non-detach (exp243) vs detach (exp244) 证明 detach 必要性
experiments/decisions.md:3418:### [2026-04-08 16:45] 决策 — exp249 完成后下一步
experiments/decisions.md:3420:**上下文**: exp249 (Small LGPA-D+GCN) 完成: 71.9/81.8 equal_concat, 73.3/83.2 MaxSim。
experiments/decisions.md:3429:1. Tiny 消融数据 (exp244, exp246b) 已经足够完整
experiments/decisions.md:3435:1. 所有 "安全" 创新方向已试完或被证伪
experiments/decisions.md:3449:已完成 VCSR (exp247, 失败) 和 PCFD (exp248, 失败) 两个创新尝试。
experiments/decisions.md:3466:A. 短期: 完成 exp249, 快速测试 POT (test-time, 无训练需求)
experiments/decisions.md:3473:3. exp249 (LGPA-D+GCN on Small) 有潜力达到 73-74% → 与 SOTA 竞争力足够
experiments/decisions.md:3480:- `exp257-259` 已基本说明当前 `exp255` recipe 空间耗尽
experiments/decisions.md:3486:  B. 回到 `exp109`，把主线改成“single-image support incomplete”的训练对象重写
experiments/decisions.md:3509:3. 若 Tiny 为正，再上 `exp255` Small scaffold
experiments/decisions.md:3535:2. 当前最强系统 `exp255` 使用的就是 `2-stage PSG`
experiments/decisions.md:3536:3. `exp255 vs exp255b` 给出最强信息：在 `GCN512` 高容量结构分支下，`2-stage PSG` 带来 `+1.7 / +1.4`
experiments/decisions.md:3537:4. 虽然 `exp009 / exp251 / exp253` 不支持“multi-stage 普遍更强”，但这恰好说明需要**重跑干净消融**，而不是放弃 PSG 主线
experiments/decisions.md:3562:**上下文**: 本地 3090 已挂；`phase1_design.md` 原把 Base 3 个 run（exp263/266/269）全部排在 3090 上。剩余资源仅 srvA/B/C 三台 5060 Ti 16G，已在跑 Phase 1 前 3 个 Tiny/Small run（exp261/262/264）。
experiments/decisions.md:3575:2. `exp260b Base = 73.9/83.2`（旧协议，本地 3090）可作 Base 行 reference
experiments/decisions.md:3582:- Phase 1 当前运行: srvA=exp262(Small OD) e70, srvB=exp261(Tiny OD) e106, srvC=exp264(Tiny OP) e83；接下来按 srvB→exp267, srvC→exp265, srvA→exp268 顺序排队；Tiny/Small 6 run 完成后立即评估是否把 Base 3 run 并入 Phase 1
experiments/decisions.md:3609:- **Phase 3-A exp271** 刚起 15min,kill + restart 用新代码(`POSE_TEST_FEAT='global'` 单块,实际上受 bug 影响极小,但 restart 是对的)
experiments/decisions.md:3610:- **exp269 / exp266** 还在训中,Python 进程里缓存的是旧 code,e120 eval 会走 broken path → 完成后 test.py 重测
experiments/decisions.md:3611:- **exp270** `POSE_ENABLED=False` 单块模式,bug 不生效,数字 59.2/68.4 仍有效
experiments/decisions.md:3612:- **Phase 1 其余已完成** (exp261/262/264/265/267/268 + exp263 e100): 全部 test.py + 新 code 重测,在机器空闲时批量跑
experiments/decisions.md:3615:- [x] exp262 Small OD transformer_120.pth re-eval → **73.8/83.1 (与原训练内部 eval 完全一致)**
experiments/decisions.md:3616:- [x] exp268 Small Market transformer_120.pth re-eval → **94.3/97.3 (与原训练内部 eval 完全一致)**
experiments/decisions.md:3624:- exp262 Small OD: 73.8/83.1 (fixed) vs 73.8/83.1 (broken) — 完全一致
experiments/decisions.md:3625:- exp268 Small Market: 94.3/97.3 (fixed) vs 94.3/97.3 (broken) — 完全一致
experiments/decisions.md:3635:- exp263 e100: Global+flip 72.5/81.8, MaxSim+flip 74.5/84.0
experiments/decisions.md:3636:- exp269 e80: Global+flip 94.4/97.0, MaxSim+flip 94.5/97.1
experiments/decisions.md:3641:- srvA (gpushare i-2:29162) 用户忘续费,SSH refused 持续 >1.5h。ckpt (exp262/268/269.pth) 和原始 train_log.txt 在 /hy-tmp/log/ 上,是否保留取决于 gpushare 平台对 expired 实例的处理策略(未确认)
experiments/decisions.md:3642:- exp274(Phase 3-A Small baseline)刚启 40min 丢失,无重要损失
experiments/decisions.md:3643:- 同时用户的实验室 3090 复活(tailscale 100.115.252.80:22,容器 `18fbbab202e1`),git pull 到 `f69b61c`(flip fix 版),正在跑 `exp263b_best_b_od_s42_3090`(Base OD 完整重跑)
experiments/decisions.md:3649:   - srvB: Phase 3-A Tiny 全 4 格(exp270 ✓ / exp271 → exp272 → exp273)
experiments/decisions.md:3650:   - srvC: exp266 Base OP(完成后 → Phase 3-B 6 格)
experiments/decisions.md:3651:   - lab3090: exp263b Base OD 完成后(~2026-04-21 02:30 CST)→ Phase 3-A Small 4 格(exp274 重启+275/276/277)
experiments/decisions.md:3652:4. Phase 1 数字:exp262/268/269 FINAL 已 committed,不受 ckpt 丢失影响;若 gpushare 宽限期内能救回 ckpt 再说
experiments/decisions.md:3653:5. lab3090 3090 24GB 显存足以容纳 Base + full scaffold + default flip eval 不 OOM,原来在 5060Ti 16GB 上 OOM 的问题在 3090 上不复现;exp263b 完成后将给出干净的 e120 FINAL,作为 exp263 e100 salvage 的升级替代
experiments/decisions.md:3660:**exp263b vs exp263 对照**:
experiments/decisions.md:3661:- exp263 新协议 e100 eff-FINAL(5060Ti srvB,OOM 后 salvage):Global+flip 72.5/81.8, MaxSim 74.5/84.0
experiments/decisions.md:3662:- exp263b 3090 完整 e120 FINAL(将来):预期 MaxSim 75+,超 KPR w/ prompt 75.1/84.3 可能性大
experiments/decisions.md:3693:| srvC | 5060Ti 16G | exp266 Base OP + Phase 3-B |
experiments/decisions.md:3694:| lab3090 | 3090 24G | exp263b Base OD + Phase 3-A Small 4 格(exp274-277 重启) |
experiments/decisions.md:3704:- lab3090 (tailscale 100.115.252.80 docker container `18fbbab202e1`) 跑 exp263b (Base OD, Full Scaffold) 从 2026-04-20 ~08:00 本地起,到 e42 Iter 100 (10:14:56 UTC = 18:14 local) 卡住
experiments/decisions.md:3711:1. kill -9 189605 + `pkill -9 -f 'exp263b_best_b_od_s42_3090'` ✅
experiments/decisions.md:3714:4. e40 eval 未产生(本 run 只到 e30=某 mAP),需 GPU 恢复后用 `test.py` 跑 ckpt40 得 interim FINAL 作为 exp263 e100 salvage 的升级替代
experiments/decisions.md:3718:- 如短期未恢复,exp263b e40 作为可接受 fallback(Base OD 中段,预期 ~70-72 mAP,低于完整 e120 但比 srvA exp263 e100 salvage 稍低)
experiments/decisions.md:3719:- 如长期挂机,lab4090(24G,pose_data 同步完成后)可接替 exp263b resume from ckpt40
experiments/decisions.md:3754:**结论**: lab4090 Occluded-Duke pose_data **production-ready**,可接 Phase 3-A Small baseline(exp274 重启)。
experiments/decisions.md:3756:### [2026-04-20 21:35] 事件 — lab4090 queue_on_ckpt daemon python3 bug,exp275 crash 重启
experiments/decisions.md:3759:- 21:34 exp274 FINAL (68.1/76.8/87.8/90.9) ckpt 生成
experiments/decisions.md:3760:- daemon 3580255 立即触发 exp275,但 1 分钟内 crash
experiments/decisions.md:3761:- `/tmp/exp275.log` 只有 `ModuleNotFoundError: No module named 'torch'`
experiments/decisions.md:3766:- exp274 当初**手动**启动用的是完整 conda path,没用 daemon,所以 OK
experiments/decisions.md:3773:5. 手动启动 exp275 用 mmpose-abu python (PID 3653199)
experiments/decisions.md:3780:### [2026-04-20 22:41] 事件 — srvC exp266 silent exit @ e70 (非 OOM)
experiments/decisions.md:3783:- exp266 Base OP Full Scaffold 从 04:46 启动,稳定跑到 e70 (~21:27 CST)
experiments/decisions.md:3795:**决策**: **不重训 exp266**。
experiments/decisions.md:3797:- 与 exp265 Small FINAL 78.4 / 86.2 **完全持平** → Base 对 Small 在 Occ-PTrack 上 0 增益
experiments/decisions.md:3801:**同 exp263/exp269 OOM 处理模式**: effective FINAL 用最后一次 eval 数字,不重训。
experiments/decisions.md:3805:- rsync Occ-Duke + pose_data ~5.5GB from srvB 会影响 exp273 磁盘 I/O
experiments/decisions.md:3812:**立即决策**: srvC 启动 **Phase 3-C exp286/287** (LGPA-only Tiny 2 runs,phase3_design.md L111-134 已规划),填补 srvC 空闲。
experiments/decisions.md:3815:- exp286 (LGPA-only + 1-stg PSG + Tiny, PID 59845) @ 23:32 CST,config load + dataset load OK
experiments/decisions.md:3816:- daemon 59846 挂 exp286 → exp287 (2-stg PSG) auto-chain
experiments/decisions.md:3818:- Small 2 runs (exp288/289) 等 lab4090 Phase 3-B 完成后接
experiments/decisions.md:3822:### [2026-04-20 23:30] 决策 — lab3090 exp263 系列 seed 切换
experiments/decisions.md:3825:- exp263c (lab3090 Base OD Full Scaffold pwrlim 280W seed 42) 跑到 e31,trajectory 异常:
experiments/decisions.md:3832:**决策**: 切换 **seed 42 → seed 41**,新命名 `exp263d_best_b_od_s41_3090_pwrlim`。
experiments/decisions.md:3833:- kill exp263c main PID 266
experiments/decisions.md:3834:- 启动 exp263d seed 41 at 23:34 CST (PID 8248)
experiments/decisions.md:3837:**用户指示**: "报告时就报告这个是 seed 41 就行" — PRCV 主表 exp263 行用 exp263d seed 41 的数字。
experiments/decisions.md:3844:**Monitor 更新**: stop b9h22bdiy (old exp263c tail) → bizb8v35k (new exp263d tail)
experiments/decisions.md:3851:- 启动 `exp265b_best_s_op_s41` (Small Full Scaffold OP seed 41) on srvA @ 12:00:30 CST, PID 633
experiments/decisions.md:3853:- 相对 exp265 (seed 42, srvC) 单变量 SEED 42→41
experiments/decisions.md:3855:- 用途: 和 exp265 组成 2-seed ensemble 或 max, 强化 OP SOTA 声明 (vs KPR w/o prompt 73.3/82.5)
experiments/decisions.md:3858:- 历史 exp263 Base OD 在 5060Ti e100 eval OOM (13.2G → 16G),  exp269 Base Market e80 eval OOM, exp266 Base OP silent exit (不确定 OOM)
experiments/decisions.md:3865:- 立即 apply: kill exp265b (12:00 版 TEST=256) + restart (12:08 版 TEST=128 PID 1151)
experiments/decisions.md:3868:**挂 daemon exp265b → exp266b (5060Ti Base OP seed 41, 带 TEST BATCH 降)**:
experiments/decisions.md:3870:queue_on_ckpt.sh /hy-tmp/log/occluded_posetrack/exp265b_.../transformer_120.pth \
experiments/decisions.md:3872:  /hy-tmp/log/occluded_posetrack/exp266b_best_b_op_s41 \
experiments/decisions.md:3873:  /tmp/exp266b.log exp265b_to_266b \
experiments/decisions.md:3876:exp266b FINAL 预计后天上午,覆盖 exp266 silent exit 留下的 OP 主表瑕疵。
experiments/decisions.md:3878:Monitor b8y4oohc4 arm for srvA exp265b。
experiments/decisions.md:3880:### [2026-04-21 03:47] 事件 — exp277 Small 3-stage PSG 训练塌缩 (negative result,不重训)
experiments/decisions.md:3883:- exp277 Small + PSG 3-stage `[-3,-2,-1]` 自 01:42 CST 启动
experiments/decisions.md:3884:- e10 eval **0.3 / 0.3** (接近 random), e120 FINAL **49.0 / 57.7** (远低 exp274 no-PSG 68.1/76.8, Δ=-19.1)
experiments/decisions.md:3893:- Tiny 3-stage (exp273) **60.5/69.9 正常** — Tiny backbone 容量小,features 较稀疏不易被 gate 归零
experiments/decisions.md:3894:- Small 3-stage (exp277) **49.0/57.7 塌缩** — Small backbone 容量大,features dense 更易被 multi-stage gate 压缩
experiments/decisions.md:3896:**决策**: **不重训**, 用 exp277 FINAL 作为 **negative result** 有效数据点。
experiments/decisions.md:3909:**Phase 3-A 科学结论** (初版, exp277 seed 42 塌缩):
experiments/decisions.md:3914:### [2026-04-21 04:30] 决策更新 — exp277 塌缩重审为偶发 seed 问题,exp277b seed 41 重跑
experiments/decisions.md:3917:- 3:47 CST exp277 FINAL 49.0/57.7 归因为 "Small 3-stage PSG 系统塌缩"
experiments/decisions.md:3922:- 新建 `exp277b_psg3_s_od_s41` 用 seed 41 重跑 (其他参数同 exp277)
experiments/decisions.md:3923:- daemon 3909905 挂 lab4090: exp284/transformer_120.pth → exp277b
experiments/decisions.md:3924:- 预计 tmr 11:50 CST FINAL (exp284 ~tmr 10:00 + 1h50min)
experiments/decisions.md:3925:- **exp277b 数字替代 exp277 作为 PRCV Table 2 Small 3-stage 行的数字**
experiments/decisions.md:3926:- exp277 (seed 42) 降级为 decisions.md 里 "偶发 seed 塌缩" 记录, results.md 标 strikethrough
experiments/decisions.md:3934:**不预判**, 等 exp277b 数据再下结论。当前 Phase 3-A 结论暂定 (基于 exp275/276 稳定的 1/2-stage 收益)。
experiments/decisions.md:3936:### [2026-04-22 08:08] 事件 — exp280 FINAL 65.7/76.2, Phase 3-B Tiny 2×2 闭合 + srvB idle
experiments/decisions.md:3939:- exp280 Swin-Tiny + GCN512 + PSG `[-1]` FINAL @ 08:07 CST srvB
experiments/decisions.md:3946:| PSG `[-1]` | 65.7/76.7 (exp278) | **65.7/76.2** (exp280, **weakest R1**) |
experiments/decisions.md:3947:| PSG `[-2,-1]` | 65.7/76.9 (exp279) | **65.9/77.4** (exp261) |
experiments/decisions.md:3950:- **GCN512+1stg 必弱**: Tiny 76.2 R1 最弱, Small exp284 82.9 R1 最弱
experiments/decisions.md:3957:**srvB GPU 状态**: exp280 是 Phase 3-B Tiny chain 最后一个, 无 daemon 继承 → **srvB idle**。
experiments/decisions.md:3960:- Task #12 (批量 MaxSim+flip) 用户指令"等当前队列跑完再起", srvC Phase 3-C exp288→exp289 ~12.5h 后才全 FINAL
experiments/decisions.md:3964:**监控链 idle 判定**: monitor `boairmoh9` (srvB 事件) 保持 armed, 将捕获任意意外事件。exp280 FINAL 处理完毕 (monitor.md + results.md + ablation.md + decisions.md + memory + git push `08de230`)。
experiments/decisions.md:3966:### [2026-04-22 09:29] 事件 — exp266b_3090 FINAL 78.5/86.2 (Base OP s41 完整 120ep) + lab3090 idle
experiments/decisions.md:3969:- exp266b_3090 Swin-Base + Full Scaffold + Occ-PTrack + seed 41 FINAL @ 09:29 CST lab3090 (docker, pwrlim 280W)
experiments/decisions.md:3973:- exp266 s42 srvC e60 eff: 78.4/86.2 → Δ +0.1/0 (持平, seed 41 微优 mAP)
experiments/decisions.md:3974:- exp265 Small OP s42: 78.4/86.2 → Δ +0.1/0 (**Base vs Small OP 0 mAP 增益**)
experiments/decisions.md:3975:- exp265b Small OP s41: 78.5/85.9 → Δ 0/+0.3 (Base 略优 R1 over Small 同 seed)
experiments/decisions.md:3982:**论文主数字**: Base OP 用 exp266b_3090 78.5/86.2 (完整 120ep), 替换 exp266 s42 e60 eff。
experiments/decisions.md:3985:- Phase 3-B Small GCN512+2stg rerun (exp285b) ✓
experiments/decisions.md:3986:- Base OD seed 41 (exp263d) ✓
experiments/decisions.md:3987:- Base OP seed 41 (exp266b_3090) ✓
experiments/decisions.md:3990:- Task #12 MaxSim eval (Base ckpts exp263d + exp266b_3090, 只需 test.py, ~5 min/ckpt)
experiments/decisions.md:3991:- 等 srvA exp266b + srvC Phase 3-C 都 FINAL 后统一批跑 (用户 wait 指令)
experiments/decisions.md:3994:**当前五机**: srvA exp266b (刚启动 e2), srvB idle, srvC exp288 (e95), lab3090 idle, lab4090 idle。3 idle。
experiments/decisions.md:4001:- srvB (5+4 batches): exp261 Tiny OD, exp267 Tiny Market (retry 后成功), exp278/279/280 Phase 3-B Tiny, exp271/272/273 Phase 3-A pure PSG Tiny
experiments/decisions.md:4002:- lab3090 (2 ckpts): exp263d Base OD, exp266b_3090 Base OP
experiments/decisions.md:4003:- lab4090 (4+5 batches): exp282/283/284/285b Phase 3-B Small, exp275/276/277/277b Phase 3-A pure PSG Small (exp274 POSE_ENABLED False crash)
experiments/decisions.md:4016:**跨 eval 验证**: Phase 3-A pure PSG 所有 Global+flip 数字和训练 FINAL eq+flip 精确对齐 (差 ≤ 0.1 R1), **exp277 seed 42 塌缩 49.0/57.6 跨 eval 复现确认偶发 seed 训练塌缩**。
experiments/decisions.md:4019:- Tiny 2×2: GCN512+2stg peak (exp261 66.4/77.7), GCN512+1stg 最弱 (exp280 66.1/76.7)
experiments/decisions.md:4020:- Small 2×2: GCN512+2stg peak mAP (exp285b 74.0/84.1), 四格方差 ≤ 0.3 mAP / 0.4 R1
experiments/decisions.md:4023:- srvC local: exp264 Tiny OP, exp265 Small OP, exp286/287 Phase 3-C Tiny LGPA-only, exp288/289 Phase 3-C Small LGPA-only
experiments/decisions.md:4024:- srvA local: exp262 (原始 srvA), exp265b Small OP s41, exp268 Small Market, exp269 Base Market
experiments/decisions.md:4032:- 等 srvA exp266b FINAL ~14:00 → srvA idle → 补 exp262/265b/268/269 eval
experiments/decisions.md:4033:- 等 srvC exp288/289 FINAL ~17:00 → srvC idle → 补 exp264/265/286/287/288/289 eval
experiments/decisions.md:4034:- 可选: lab3090 上跑 cross-domain Market→Occ-ReID (Occ-ReID 数据集已解压, 需 rsync exp267/268/269 Market ckpt)
experiments/decisions.md:4036:### [2026-04-22 12:51] 🔥 exp288 FINAL 73.8/83.8 — GCN 对 Small OD 零贡献确认
experiments/decisions.md:4039:- exp288 Swin-Small + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + PSG `[-1]` (**无 GCN**) FINAL @ 12:51 CST srvC
experiments/decisions.md:4043:- exp285b Full Scaffold (GCN512 + LGPA + 2-stg PSG): **73.8 / 83.8 / 90.7 / 92.7** → Δ 0/0/-0.2/-0.7
experiments/decisions.md:4044:- exp282 Full GCN256+1stg: 73.7/83.9 → Δ +0.1/-0.1
experiments/decisions.md:4045:- exp284 Full GCN512+1stg: 73.4/82.9 → Δ +0.4/+0.9 (LGPA-only 反超!)
experiments/decisions.md:4049:2. 和 Tiny 结论 (exp286 LGPA-only 66.0 ≈ exp261 Full 65.9) **跨 backbone 一致**
experiments/decisions.md:4059:- exp289 LGPA-only 2-stg 自动启动 (srvC PID 86783), FINAL ~16:50 对照 PSG stage in LGPA-only 配置
experiments/decisions.md:4060:- 建议: 跑完 exp289 → 评估是否也加 GCN 做 Market/OP 对照 (exp267 + exp264 本就无 GCN 配置?)
experiments/decisions.md:4063:- results.md Phase 3-C section 已填 exp288 FINAL
experiments/decisions.md:4067:### [2026-04-22 14:31] ⚠️ exp292 CUDA OOM @ e20 eval, restart with TEST.IMS_PER_BATCH 64
experiments/decisions.md:4070:- exp292 Small Market target-heatmap 启动 12:52, 训练 e1-e20 顺利, Loss 14.77→4.08, Acc 0.001→0.607
experiments/decisions.md:4081:- 重启 exp292 with `TEST.IMS_PER_BATCH 64` (从 default 256 降 4x)
experiments/decisions.md:4087:- 新启动 PID 通过 /tmp/exp292.log 验证 e1-e20 都过关, 特别关注 e20 eval
experiments/decisions.md:4092:- lab4090 exp291 目前 TEST 默认 256, 如果 e20 eval 失败也同样降
experiments/decisions.md:4094:### [2026-04-22 18:13] exp291 FINAL 73.5/82.9 (target-heatmap OD) + exp293 auto-chain launched
experiments/decisions.md:4096:**exp291 FINAL** @ 18:13:30 CST lab4090:
experiments/decisions.md:4098:- vs exp285b Full Scaffold scene baseline 73.8/83.8 → Δ -0.3/-0.9/0/-0.2
experiments/decisions.md:4101:**三数据集 target-heatmap 横向对比 (partial, exp290/exp292 还在跑)**:
experiments/decisions.md:4104:| OP (多人, exp290 e30) | -0.1 / +0.1 | R1 持平/微优, 符合预期机制有效场景 |
experiments/decisions.md:4105:| OD (多单人, exp291 FINAL) | -0.3 / -0.9 | 接近 no-op, 微差 eval noise |
experiments/decisions.md:4106:| Market (全单人, exp292 e30) | 对照待 FINAL | 预期严格持平 (目前 e30 92.7 正常轨迹) |
experiments/decisions.md:4110:- 作为 supplementary 消融: 机制在 single-person 数据集无回归, 论文主表 Small OD 仍用 exp285b 73.8/83.8
experiments/decisions.md:4112:**auto-chain → exp293 触发成功**:
experiments/decisions.md:4113:- daemon 706372 detected ckpt @ 10:14:09 UTC (18:14 CST), 20s 安全 + no-crash 检查 → launch exp293 PID 724112 @ 10:14:29 UTC
experiments/decisions.md:4114:- exp293 config 确认 PLBOA=True 激活, OA-SD WARNING 消失 (teacher/student 现有差异)
experiments/decisions.md:4117:### [2026-04-22 23:25] exp292 e90 eff FINAL + exp293 e80 eff FINAL — target-heatmap Market + PLBOA Base 双消融收尾
experiments/decisions.md:4119:**exp292 Small Market target-heatmap** (lab3090 PLBOA OFF default):
experiments/decisions.md:4122:- vs exp268 FINAL 94.3/97.3: Δ **-0.1 / -0.2** (essentially 持平)
experiments/decisions.md:4123:- 结论: target-heatmap 在 Market 全 single-person 严格 no-op, 和 exp291 OD (-0.3/-0.9) / exp290 OP (-0.1/0) 结论一致 — **机制 3 数据集都 near-持平**
experiments/decisions.md:4125:**exp293 Base Market + PLBOA** (lab4090, OA-SD 激活):
experiments/decisions.md:4130:- vs exp269 e80 eff FINAL 94.4/97.0 (PLBOA OFF): Δ -0.3 / -0.1 (Global)
experiments/decisions.md:4133:  - 主表 Base Market 主数字 **仍用 exp269 94.4/97.0**
experiments/decisions.md:4134:  - exp293 作 supplementary "PLBOA on Market" 消融
experiments/decisions.md:4141:- 不作主创新, 主表 Small OD/OP/Market 仍用 exp285b/exp265/exp268 scene baseline
experiments/decisions.md:4144:- OD (exp285b etc): PLBOA True, OA-SD 蒸馏有效, +性能
experiments/decisions.md:4145:- OP (exp265 etc): PLBOA True, OA-SD 蒸馏有效, +性能
experiments/decisions.md:4146:- **Market: PLBOA False** (exp293 验证), 分布不匹配 → 保留关闭
experiments/decisions.md:4155:- exp289 完成后 Phase 3-C 2x2 闭合 (srvC ~05:30 tmr FINAL)
experiments/decisions.md:4156:- exp266b srvA FINAL (~13:00 tmr) 作 Base OP seed 41 srvA 对照 (cross-device with lab3090)
experiments/decisions.md:4157:- exp290 srvB FINAL (~09:00 tmr)
experiments/decisions.md:4160:### [2026-04-23 05:40] exp289 FINAL 73.8/83.3 — Phase 3-C Small 2×2 闭合 + exp269b auto-chain 启动
experiments/decisions.md:4162:**exp289 FINAL**:
experiments/decisions.md:4165:- vs exp288 LGPA-only 1-stg 73.8/83.8/90.5/92.0: Δ 0 / -0.5 / 0 / +0.4
experiments/decisions.md:4166:- vs exp285b Full Scaffold 73.8/83.8: Δ 0 / -0.5
experiments/decisions.md:4171:| LGPA-only | exp288 73.8/83.8 | exp289 73.8/**83.3** |
experiments/decisions.md:4177:**exp269b auto-chain 启动成功** @ 05:40 srvC via daemon 94420:
experiments/decisions.md:4178:- Base Market PLBOA OFF full 120 epoch (公平对比 exp293 restart PLBOA ON)
experiments/decisions.md:4186:- Phase 3-C (LGPA-only × PSG): **4/4 FINAL ✓** (刚刚 exp289 闭合!)
experiments/decisions.md:4188:- PLBOA Market (exp293 restart + exp269b) 进行中 (~06:00-11:40 FINAL)
experiments/decisions.md:4189:- exp263b (Base OD s42 restart) queued on lab4090 after exp293
experiments/decisions.md:4191:srvC 接下来: exp269b FINAL ~11:40 → 再 idle (无 chain). 或可 queue exp263b_s42 之类。
experiments/decisions.md:4193:### [2026-04-23 08:24] exp293 FINAL 93.8/97.2 (restart full 120) + exp263b auto-chain launched
experiments/decisions.md:4195:**exp293 restart FINAL**:
experiments/decisions.md:4200:**对比 original exp269 (PLBOA OFF, e80 eff 94.4/97.0)**:
experiments/decisions.md:4202:- 但 exp269 只有 e80, 对比不公平 — 等 exp269b FINAL (~11:40) 才有公平 120ep vs 120ep
experiments/decisions.md:4204:**cross-restart noise (exp293 first run e80 eff vs restart e80)**:
experiments/decisions.md:4209:**exp263b auto-chain 启动 @ 08:24 lab4090**:
experiments/decisions.md:4210:- Base OD seed 42 full 120 restart (原 exp263 e100 eff 72.5/81.8 OOM 截断)
experiments/decisions.md:4215:1. ✅ exp293 (PLBOA ON Base Market) FINAL 93.8/97.2
experiments/decisions.md:4216:2. 🔄 exp269b (PLBOA OFF Base Market) srvC e17, FINAL ~11:40
experiments/decisions.md:4217:3. 🔄 exp263b (Base OD s42) lab4090 e1 NEW, FINAL ~15:30
experiments/decisions.md:4218:4. ⏳ exp266c (Base OP s42) queued srvB after exp290 FINAL (~09:15)
experiments/decisions.md:4221:### [2026-04-23 09:22] exp290 FINAL 78.4/86.2 — target-heatmap OP 严格持平 scene + exp266c chain
experiments/decisions.md:4223:**exp290 FINAL**:
experiments/decisions.md:4226:- **严格持平 exp265 scene baseline 78.4/86.2/94.8/97.3** (Δ 0/0/0/+0.1)
experiments/decisions.md:4231:| OD (exp291) | 73.5/82.9 | 73.8/83.8 | -0.3/-0.9 |
experiments/decisions.md:4232:| **OP (exp290)** | **78.4/86.2** | **78.4/86.2** | **0/0 严格持平** |
experiments/decisions.md:4233:| Market (exp292 e90 eff) | 94.2/97.1 | 94.3/97.3 | -0.1/-0.2 |
experiments/decisions.md:4240:- 主表 Small OP 数字用 exp265 78.4/86.2 (= exp290, 等价)
experiments/decisions.md:4242:**exp266c chain**:
experiments/decisions.md:4243:- daemon 109773 detected exp290 ckpt @ 09:21
experiments/decisions.md:4248:- ✅ exp289 FINAL 73.8/83.3 (Phase 3-C Small 2-stg)
experiments/decisions.md:4249:- ✅ exp290 FINAL 78.4/86.2 (target-heatmap OP)
experiments/decisions.md:4250:- ✅ exp293 FINAL 93.8/97.2 (Base Market PLBOA ON full 120)
experiments/decisions.md:4251:- 🔄 exp269b e20 (Base Market PLBOA OFF full 120)
experiments/decisions.md:4252:- 🔄 exp263b e8 (Base OD s42 full 120)
experiments/decisions.md:4253:- ⏳ exp266c queued (Base OP s42 full 120) chain soon
experiments/decisions.md:4255:### [2026-04-23 13:20] 决策 #exp266b FINAL 78.7/86.3 — Base OP 新 SOTA
experiments/decisions.md:4257:**exp266b srvA s41 FINAL (2026-04-23 13:18:50 CST)**:
experiments/decisions.md:4263:| **srvA 5060Ti** | exp266b | **78.7/86.3** | baseline |
experiments/decisions.md:4264:| lab3090 | exp266b_3090 | 78.5/86.2 | -0.2/-0.1 |
experiments/decisions.md:4272:| Small (exp265/265b) | 78.4/86.2 | 78.5/85.9 |
experiments/decisions.md:4273:| Base (exp266/266b) | 78.4/86.2 e60 eff | **78.7/86.3** (srvA) |
experiments/decisions.md:4276:- 原方案: exp266b_3090 78.5/86.2 (lab3090 完整 120 epoch)
experiments/decisions.md:4277:- **更新方案**: **exp266b srvA 78.7/86.3** (srvA 完整 120 epoch, +0.2 mAP / +0.1 R1 更强)
experiments/decisions.md:4281:- exp266b 78.7/86.3 vs exp265b 78.5/85.9 → Δ **+0.2 mAP / +0.4 R1**
experiments/decisions.md:4288:### [2026-04-23 16:50] 决策 #exp263b FINAL 73.5/81.5 — seed 42 full 120 restart 有效但不如 seed 41
experiments/decisions.md:4290:**exp263b lab4090 s42 FINAL (2026-04-23 16:47:17 CST)**:
experiments/decisions.md:4292:- ckpt: `/home/afr/SOLIDER-REID/log/occluded_duke/exp263b_best_b_od_s42_full120/transformer_120.pth`
experiments/decisions.md:4297:| exp263 orig | 42 | e100 eff (OOM) | 72.5/81.8 | baseline |
experiments/decisions.md:4298:| **exp263b restart** | **42** | **e120 FINAL** | **73.5/81.5** | +1.0/-0.3 |
experiments/decisions.md:4299:| exp263d | 41 | e120 FINAL | 74.1/83.3 | +1.6/+1.5 |
experiments/decisions.md:4302:1. **full 120 epoch > e100 eff**: +1.0 mAP 提升, 说明原 exp263 因 OOM 中断的确损失了数字
experiments/decisions.md:4304:3. **R1 异常**: exp263b R1 (81.5) 略弱于 exp263 orig (81.8) 尽管 mAP 更高。可能 full 120 epoch 在末期轻微 overfit R1 top-1。
experiments/decisions.md:4308:- 主表仍用 **exp263d 74.1/83.3** (seed 41, 最强)
experiments/decisions.md:4309:- exp263b 作 **seed 42 full 120 复现数据点** (证明 restart 机制有效, 证明 seed 42 天然弱)
experiments/decisions.md:4314:- 对照 exp263 orig MaxSim 74.5/84.0, exp263d MaxSim 75.2/84.8
experiments/decisions.md:4317:- ✅ exp263b (Base OD s42 full 120) lab4090 FINAL 73.5/81.5
experiments/decisions.md:4318:- 🔄 exp266c (Base OP s42 full 120) srvB e30 eval 76.5/84.7
experiments/decisions.md:4319:- 🔄 exp269b (Base Market PLBOA OFF full 120) srvC e60 eval 94.1/97.0
experiments/decisions.md:4321:### [2026-04-24 01:20] 决策 #exp269b FINAL 94.5/97.2 — Market Base 新 SOTA, full 120 restart 策略验证
experiments/decisions.md:4323:**exp269b srvC s42 FINAL (2026-04-24 01:17:24 CST)**:
experiments/decisions.md:4326:**vs exp269 original (OOM 前 e80 eff)**:
experiments/decisions.md:4330:**vs exp268 Small**: Δ +0.2/-0.1 (Base vs Small Market 已饱和)
experiments/decisions.md:4331:**vs exp293b Base PLBOA ON**: Δ +0.7/0 → **PLBOA 净 -0.7 mAP 代价确认**
experiments/decisions.md:4334:- 原: exp269 orig 94.4/97.0 (Global+flip) 或 94.5/97.1 (MaxSim+flip)
experiments/decisions.md:4335:- **新: exp269b 94.5/97.2** (两者等价, 直接 eq_concat 就达 MaxSim 水平)
experiments/decisions.md:4338:- ✅ exp263b (Base OD s42 full 120) lab4090 FINAL 73.5/81.5, MaxSim 74.8/84.0
experiments/decisions.md:4339:- ✅ exp266c (Base OP s42 full 120) srvB **running** (e60 77.9/85.6)
experiments/decisions.md:4340:- ✅ exp269b (Base Market PLBOA OFF full 120) srvC FINAL 94.5/97.2
experiments/decisions.md:4342:**等等, 是 2/3 FINAL. srvB exp266c 仍在训练, FINAL ETA ~13:22 today。**
experiments/decisions.md:4346:### [2026-04-24 02:20] 决策 #exp294 FINAL 74.0/82.6 — GCN 冗余假设 3-backbone 统一验证
experiments/decisions.md:4348:**exp294 lab4090 s41 FINAL (2026-04-24 02:18:48 CST)**:
experiments/decisions.md:4352:| Exp | GCN | mAP | R1 | Δ vs exp263d |
experiments/decisions.md:4354:| exp263d | **ON** | **74.1** | **83.3** | baseline |
experiments/decisions.md:4355:| **exp294 (本)** | **OFF** | **74.0** | **82.6** | **-0.1 / -0.7** |
experiments/decisions.md:4360:| Tiny | exp287 65.9/77.0 | exp261 65.9/77.4 | **0/-0.4** |
experiments/decisions.md:4361:| Small | exp289 73.8/83.3 | exp285b 73.8/83.8 | **0/-0.5** |
experiments/decisions.md:4362:| Base | exp294 74.0/82.6 | exp263d 74.1/83.3 | **-0.1/-0.7** |
experiments/decisions.md:4374:- Base OD 主数字仍用 **exp263d 74.1/83.3** (最强)
experiments/decisions.md:4375:- exp294 作 **Phase 3-C Base 补齐行 + GCN 冗余 claim 证据**
experiments/decisions.md:4378:- lab4090 idle, 启动 exp294 MaxSim+flip eval (预期 ~74.8-75.2/83-84, 对标 exp263b 74.8 / exp263d 75.2)
experiments/decisions.md:4379:- 若 MaxSim < exp263d, 补 claim: "GCN 对 MaxSim 也冗余"
experiments/decisions.md:4382:- ✅ exp263b (Base OD s42) 73.5/81.5
experiments/decisions.md:4383:- ✅ exp266b (Base OP s41) 78.7/86.3 (SOTA)
experiments/decisions.md:4384:- ✅ exp269b (Market Base PLBOA OFF) 94.5/97.2
experiments/decisions.md:4385:- 🔄 exp266c (Base OP s42 full 120) srvB running, FINAL ~13:22 today
experiments/decisions.md:4386:- ⭐ **exp294 (Base Full-GCN s41 ablation)** FINAL 74.0/82.6 (用户新加 ablation)
experiments/decisions.md:4390:# Post-PRCV exp295–321b 决策回填（2026-06-15 补文档债）
experiments/decisions.md:4398:**双向 sweep 证据**：0.5 真生效(exp311b Small) **-0.7 mAP**；1.0 default(exp295/exp261) baseline ⭐；2.0(exp312 Tiny) **-0.4 mAP**
experiments/decisions.md:4406:**8 个 sweep 点（vs exp261 67.2/78.6 MaxSim）**：GLS2.0 -0.4；PartW2.0 -0.3；PartW0.5 0；lgpaW1.0 -0.2；oasdW2.0 0；lgpaW0.25 **+0.2**；partTriW0.5 -0.1；oasdW0.5 -0.1
experiments/decisions.md:4423:**结果（exp320 Small s1234 vs exp295）**：eq 68.1/79.3 vs 74.2/84.0（-6.1/-4.7）；MaxSim **68.8/79.6 vs 75.2/85.4（-6.4/-5.8）**
experiments/decisions.md:4430:**上下文**：overnight Base OD LR sweep（exp296/297/298）+ PLBOA 消融（exp299）。
experiments/decisions.md:4431:**LR sweep（Base s41 vs exp296 LR8 74.9/83.8 MaxSim）**：LR8 baseline；LR4(exp297) -0.3/+0.3（近 tie）；LR2(exp298) **-5.3/-4.7**（下界）
experiments/decisions.md:4432:**PLBOA 消融**：exp299(OFF) 72.7/80.5 vs exp296(ON) 74.9/83.8 → OD 上 **PLBOA net +2.2 mAP**；配 Tiny exp307(+2.7) 2-backbone 一致
experiments/decisions.md:4434:**理由**：LR4≈LR8（非显著 underfit），LR2 严重 underfit -5.3。PLBOA 在 Occ-Duke +2.2-2.7 mAP，但 Market→Occ-ReID 跨域 -25.4 mAP（exp293 vs exp269）。
experiments/decisions.md:4435:**执行结果**：Base OD 主表保持 exp263d 75.2/84.8；exp296-298 作 LR ablation，exp299/exp307 作 PLBOA dataset-specific evidence。
experiments/decisions.md:4440:**multi-seed 统计（MaxSim+flip）**：Small(42/1234/2024) mean **74.7 std 0.45** 主行 exp295；Base(41/1234/42) mean **74.87 std 0.42** 主行 exp263d
experiments/decisions.md:4447:**上下文**：post-PRCV「搬范式」首验（frozen Qwen2.5-VL-3B，零训练，lab-3090-d）。
experiments/decisions.md:4454:**选择**：(a) always-NO 使一个词格式下 A/B/C **不可判**，非方法被证伪；
experiments/decisions.md:4475:**上下文**：exp323 frozen-MLLM 线偏负后，按搬范式 #2 路线做 frozen DINOv2-base 廉价首验（training-free）：
experiments/decisions.md:4500:**执行结果**：exp327 线止损。若要上头优先选 DIFT（不同范式，smoke 趋势更强）。dinov3-b 因 gated 无法验证；按 registers 小幅增益外推预期也不破天花板，不为它申请 token。slim pose data pipeline 经 dinov2-b sanity（复现 exp324 数字）+ heavy-occ 989/2210 完全一致核验无损，可复用于后续 hyy 实验。
experiments/decisions.md:4511:**执行结果**：SD/DIFT 线止损，不上头。**重要方法论教训写入铁律：训练-free probe 必须用全量 gallery 判定绝对值，小 gallery smoke 只验流程不验数值**——DIFT 是活教材（smoke 排第一、full 垫底）。结合 exp327（registers +0.29 小幅、不破天花板）：**换特征源（更新 DINO / 换 SD 范式）都不是 frozen 天花板的解**，瓶颈在 frozen 本身（需 LoRA/解冻，即 exp324d 线）或换"DINO 补 Swin"重量级角度（planner #1 oracle）。
experiments/decisions.md:4540:**资产**: 现成 Market-trained ckpt `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`(Swin-Base+PSG+LGPA+GCN512), 现成跨域 eval `test_on_occluded_reid.py`(Market→Occluded-ReID 86.0/88.5 已存), 两域数据+pose 齐。脚本 `scripts/uce_calib_probe.py`, 结果 `log/uce_calib_probe.json`。
experiments/decisions.md:4551:**资产**: Market-trained ckpt `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`(Swin-Base+PSG+LGPA+GCN512, Occ-ReID baseline 88.0 mAP MaxSim+flip), 数据 Occluded-ReID(1000q/1000g+pose), env solider-reid(torch1.13+mmcv)。脚本 `scripts/vcnorm_probe.py`(主)+`scripts/vcnorm_probe_control.py`(对照), 结果 `experiments/vcnorm_probe/*.json` + README。
experiments/decisions.md:4562:### [2026-06-17 05:2x] 决策: burstiness 范式 bet 判 KILL + 收窄"in-domain 特征机制"整类
experiments/decisions.md:4563:**上下文**: 夜间范式调研唯一过审强 bet=burstiness(VLAD-BuFF/face-set import)。0-GPU 前提在 frozen DINO 成立(occluded +0.0206 更 bursty)。e120 弱 baseline(TransReID 53.5)训练模型真实判据。
experiments/decisions.md:4567:**下一步**: 不再碰 in-domain 特征机制。转(a)改问题定义/评测协议 (b)改监督/目标 (c)跨域泛化 (d)新匹配范式。新 bet 必须有**训练模型 kill-switch**(非 frozen)。已启动 informed 调研 agent。VC-Norm(唯一在训的训练端改表征机制)跨域判据待定。
experiments/decisions.md:4599:**调查**: 派 10 个 Codex 并行深挖(用户 rate-limit 不让开 300 Claude 子agent,Codex token 无限)→ 挖出**热图 bug**:exp335 喂 target-only 热图(`heatmaps[:,0]`+POSE_USE_TARGET_HEATMAP=True)→ LGPA assign KL 坍缩=0 → 部位退化。修(scene-merged)→ assign 0→7.02≈原版。但 ViT 仍只 +0.5、不翻盘。深挖发现 **LGPA-D 从未单独跑过**(exp244/245g 全是 PSG+LGPA+OASD+aug+384+Swin 全系统)。
experiments/decisions.md:4691:**上下文**: 三个独立 codex(终身 d3 / 开集 d9 / 长尾 d10)收敛到同一 re-framing: ReID 失败由 GALLERY 组成(规模/膨胀/分布)驱动, 非只看 query/模型。用户要求零训练验证, ★铁律=每个 per-query 相关都控 trivial 代理(吸取 HUBNESS §7.6 教训: 上个诊断被漏控 #false-in-topk 证伪)。脚本 `cvpb_gallery_killswitch.py`, 复用 hubness 缓存特征, Market exp260b + Occluded-Duke exp255。双审(Claude broad 5 blocking 全修 + Codex)。
experiments/decisions.md:4702:### [2026-06-26] 决策 #99: LM-ReID(低分辨率=采样格点) session — 6.5 成稿 + 训练端穷尽 + 冲 7.0 失败 + d17 KILL = 探索收敛
experiments/decisions.md:4704:**上下文**: 探索 LM-ReID(d8 演化): 低分辨率 ReID 重定义为采样格点 sampling-lattice 隐变量, test-time decision marginalization(K=9 phase/bbox/kernel 变体边缘化)。autonomous mandate 找 B 类方法稿, 全自主无休止。脚本 cvpb_lattice_killswitch.py(全参数)/cvpb_lm_reid_train.py/cvpb_d17_killswitch.py。
experiments/decisions.md:4707:- **LM-ReID test-time 成立(6.5/10)**: LM-S2 5 分辨率全 beat 普通 TTA / LM-S2-strong 全 beat 强 TTA(+0.76~7.28, severe LR 处强 TTA 反有害) / LM-S4 bbox 检测框不确定性主导 +2.84 / K-sweep K=5 达 87% / LM-S3 logsumexp(soft decision marginalization)severe LR 最优 / backbone 泛化 Swin +3。
experiments/decisions.md:4708:- **训练端穷尽(8 机制 + 4 codex 8.5/10 无空间)**: embedding-invariance(consistency −1.73)/frozen-adaptation(LS-MRT +0.028/LPA +0.075/LATS)/backbone-loss(LSRC −1.9 损判别)/robust-ERM(Hard-Lattice 76.9<77.44)/input-canonicalize(BLC 数据证伪)全负 → "Why Training-Time Invariance Fails" 强论点。审查纪律: LSRC full-finetune codex 审出 Critical(默认混旧 loss)+High(train/test 不对称)已修[[pre-experiment-review-discipline]]。
experiments/decisions.md:4709:- **冲 7.0 三条腿全失败(codex push7 6/10 路径)**: ①detector-jitter σ-sweep 单调衰减到负(h12 +5.49→+2.18→−5.85, marginalization 是 sub-pixel sampling-lattice 非 detector 鲁棒性=诚实机制范围界定) ②MSMT17 跨数据集 config 缺失止损(msmt17_split 数据读对但 swin_small_pose.yml 被删 SANITY 2.67) ③adaptive-K 中性(per-query≈fixed K=5)。
experiments/decisions.md:4711:- **codex meta-eval 确认探索充分收敛**: 全新范式(event-camera 5.5 需新数据/federated 4.5/text 3 撞 FM/3D-SMPL 2.5 撞 SMPL/group 2 撞#false)非当前代码线方法点。
experiments/decisions.md:4713:**决策**: **LM-ReID 6.5 收尾投 B 类(唯一存活候选; codex 三层 push7/d17-eval/meta 都判务实)**。
experiments/decisions.md:4714:**理由**: 探索充分收敛——训练端无空间(8 机制+4 codex)/冲 7.0 机制范围被 detector 误差封顶/d17 撞 #false 坑/全新范式需新数据非方法点。6.5 是诚实天花板=中等偏强 B 类(重定义+干净 test-time 机制+训练端系统反例+多维证据+σ-sweep 诚实界定)。负结果大多成论文素材。
experiments/decisions.md:4717:### [2026-06-27] 决策 #100: 换量级跳出盒子探索诚实终点 — AG+DG+open-set 全证伪, 现有 ReID 训练端机制探透
experiments/decisions.md:4719:**上下文**: 用户点醒"没限定 occreid+solider"→换量级跳出盒子(训新预训练范式/换 backbone/换问题, 不限现有 occluded+SOLIDER)。codex 全 ReID gap analysis 选 AG(8/10)/DG(#2 6.5)。
experiments/decisions.md:4721:**探索链(一整天全证伪)**:
experiments/decisions.md:4731:**出路(用户拍板)**: ①收 LM-ReID(#99, 6.5 现有最强 B 类, 文档已全)/PSG 已成型线(最务实) ②换真实新数据规模(下新大规模数据集) ③换问题领域。
experiments/decisions.md:4732:**方法论价值**: cheap kill-switch(frozen probe 零训练几分钟 + 一次 30ep bounded)一天验死 AG+DG 两范式, 没浪费多日训练; codex 审 backbone fine-tune(抓预处理口径不一致+ckpt+checkpoint); 诚实定位每个工程 bug(patch14/.eval返回None/MSMT camid/0.5 norm/python env mmcv); 不绑架沉没成本。负向结论(钉死共同墙)避免后续重复盲试。

codex
最新 decisions 里已经把 LM-ReID、AG、DG、open-set 等写成探索收敛或证伪。接下来我会把 web 检索分成三组：问题重定义、监督范式、机制范式，并优先查 2024-2026 是否已有强先例。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp359_lm_reid/design.md && sed -n '1,220p' experiments/exp359_lm_reid/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
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

### [2026-06-26] push-7.0 实验（codex push7 给冲 7.0 路径 6/10）

- **LM-S4 bbox 5-height**（完整）：bbox-only ≈ all-axis 跨所有分辨率（h12/16/20/24/32 = 72.01/80.34/83.94/85.53/87.33，all 72.03/80.00/83.69/85.28/87.18）→ **bbox 是主因子，跨分辨率确认**（好素材）。

**★★detector-jitter σ-sweep（冲 7.0 detector 腿确定失败）**：均匀 ±1 离散格点改成连续 Gaussian center+scale（模拟真实检测器 localization error），扫 σ：

| detector σ | h12 marg | h16 marg | h16 LATTICE−TTA | h20 LATTICE−TTA |
|---|---|---|---|---|
| 0（均匀离散） | +5.49 | +2.81 | +3.34 | +1.23 |
| 0.25（理想精确 detector） | +3.68 | +1.55 | +2.15 ✓ | +0.76 ✗ |
| 0.5（真实 COCO detector） | +2.18 | +0.86 | +1.46 ✗ | +0.37 ✗ |
| 1.0（大误差） | **−5.85** | **−3.11** | **−2.52** ✗ | **−1.39** ✗ |

- **marginalization 增益随 detector 误差 σ 单调衰减**。codex kill-switch③（h12/16/20 都 ≥+2）**确定不过**：σ=0.25（理想）h20 不过，σ=0.5（真实 detector localization error ~0.5-1.0 LR-px）h16/20 都不过。
- **→ 冲 7.0 的 detector 腿失败，LM-ReID 诚实定位 6.5。** 但 σ-sweep 是**有价值的诚实诊断**：精确界定机制范围 = **sub-pixel sampling-lattice 边缘化（小精确扰动），不是对大 detector 框误差的鲁棒性**。论文 Discussion 用这个诚实界定（比硬吹 7.0 扎实）。
- 6.5 后续（巩固泛化非冲 7.0）：跨数据集 MSMT17（kill-switch②）+ adaptive-K（compute）+ paper 6.5 定位写作。

### [2026-06-26] 跨数据集 MSMT17 = 止损（config 配对深坑）

参数化 cvpb 加 `--dataset msmt17`（msmt17_split 读 list 文件）+ `--semantic_weight` override。**msmt17_split 数据读取完全正确**（q 11659/g 82161 标准 count，pid 0-3059 同空间，img 存在，q⊂g）。但 **MSMT17 ckpt 的 model config 配对是深坑**：正确 config `swin_small_pose.yml` 已被删（缺失），现有 `pose_backbone_psg_small.yml`（sw=0.2）/swin_small.yml（baseline 无 pose_dict 报错）都不匹配 ckpt 训练（sw=0.6 + pose-mul-scale0.3 + llw0.7）；SANITY 持续 2.67~4.29（特征垃圾，非数据问题）。**止损留用户**（用正确 config 或重训），不无限 debug 缺失 config。

**→ 跨数据集（kill-switch②第二维度）做不了；但 backbone 泛化（Swin market +3）已是一个泛化维度，且冲 7.0 已失败（detector 腿），跨数据集对 6.5 是 nice-to-have 非必需。LM-ReID 诚实定位 6.5（核心齐 + 训练端反例 + backbone 泛化 + σ-sweep 机制范围界定）。**

### [2026-06-26] adaptive-K（最后 supporting，中性）

per-query phase volatility（median 阈值）选 K：高 vol marginalize K=9，低 vol K=1，avg_K=5（56% compute）。

| h | single | adaptive-K(56%) | fixed K=9 | 保留率 |
|---|---|---|---|---|
| 12 | 66.79 | 70.76 | 72.01 | 76% |
| 16 | 77.47 | 79.66 | 80.10 | 83% |
| 20 | 82.48 | 83.58 | 83.72 | 88% |

- **adaptive-K（per-query）≈ fixed K=5（uniform）at 56% compute**（h16 adaptive 79.66 ≈ K-sweep K=5 79.61）→ per-query volatility selection 没明显优势 over uniform K。和 LPA 死因一致（query-side 预测 lattice 受益度做不到）。compute 故事用 K-sweep（fixed K=5 87%）就够。

**★★→ LM-ReID 6.5 实验链完整**：LM-S2/S2-strong/S4/K-sweep/LM-S3/backbone/σ-sweep/adaptive-K + 训练端反例（4 类穷尽）。冲 7.0 失败（detector 腿 σ-sweep 单调衰减到负 + MSMT17 config 缺失止损）。**6.5 是诚实天花板，中等偏强 B 类候选。下一步=完善 paper 6.5 + 用户醒决定收尾投 vs 换方向（d17 备胎）。**

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp360_intruder/design.md && sed -n '1,220p' experiments/exp360_intruder/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
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
# exp360 Intruder — monitor

## 阶段 0 地基机制验证（frozen probe，2026-06-26）

**脚本**: `experiments/cargo_cvpb/cvpb_intruder_probe.py`（market noLMloss frozen baseline + 合成 target+donor，occ_frac 0.45 下半身遮挡）
**log**: 4090 `/tmp/exp360_intruder_full.log`（50 donor×20 + 700 query vs full gallery）

| 判据 | 结果 | 通过 |
|---|---|---|
| H1 donor-ID 泄漏可测 | probe acc 73% vs chance 2% = **36.5x** | ✅ PASS（泄漏巨大确凿）|
| person >> rand control | leak person 0.15 >> rand −0.01 | ✅ PASS |
| **H2 leak ↔ AP drop（控 #false）** | raw spearman **+0.120** → **partial\|#false = −0.028（≈0）** | ❌ **FAIL** |

AP drop 大（clean 0.835 → mix 0.409，−0.43，遮挡确实大幅损害检索）。

### 诚实判读
- donor 泄漏**存在且巨大**（H1 36.5x，能从被遮挡 target 的 embedding 73% 认出遮挡者），但**泄漏量不独立于 #false-in-topk 预测检索损害**（H2 控 #false 后 ≈0）。
- 即：leak 和检索损害都是遮挡的"症状"，但 leak 不是损害的**独立原因**（损害主由 top-k 混入错 ID 驱动）。
- **memory 铁律再次发挥**（#false 控制把 +0.120 打回 −0.028），避免 over-claim "泄漏导致损害"。Hubness/evidence/d17 同款。
- 对 Intruder 的实质打击：核心假设"压 donor 泄漏 → 救检索"因果地基不稳（= codex 头号风险=退化 target ambiguity 墙）。

### 不收敛停（deep work 模式）
- H2 是 frozen per-query 相关（哪个 query 损害大），FAIL ≠ "训练压 donor-ID 无效"（相关 ≠ 干预效果，H3 训练才是终判）。
- codex 评估（`codex_h2fail_decision.md`）：H2 FAIL 不数学杀死(但杀强叙事)，建议 Stage0.5 frozen donor-null projection 因果测试（GRL 上界代理）再裁决。

## 阶段 0.5 donor-null projection 因果测试（2026-06-26，codex 7/10）

frozen rank-r 抠掉 donor 判别方向，看 mix AP 是否回升（= GRL 软压的上界代理）：

| r | donor acc | mixAP Δ | cleanAP Δ | #false |
|---|---|---|---|---|
| 5 | 0.72→0.575 | +0.018 | −0.002 | 5.89→5.71 |
| 10 | →0.500 | +0.025 | −0.005 | →5.63 |
| 20 | →0.480 | +0.028 | −0.008 | →5.57 |
| 40 | →0.480 | **+0.031** | −0.009 | →5.58 |

**裁决：Intruder DEAD（因果证伪）**：
- donor 信息**高度分布式**：r=40 抠 40 维，donor acc 只 0.72→0.48（r=20→40 饱和，抠不干净）。
- 即使抠 40 维，mixAP 只回 **+0.031**（mix 0.409→0.440 vs clean 0.835，gap 0.4 几乎没动），clean 被伤 −0.009，#false 几乎不降。
- **坐实 codex 头号风险 + memory target ambiguity 墙**：donor-ID 可读(H1 36x)但压它不救排序。
- **codex Stage0.5 完美奏效**：frozen 因果测试(零训练)裁决 Intruder，省多日 GRL 训练 + 审查。上界 projection 救 0.031，GRL 软压更不行。

**→ Intruder DEAD（有据：H2 #false 控制 + Stage0.5 r-sweep 双重因果证伪）。按 codex 裁决转 B PSC-JEPA（6.5，从 SOLIDER continued-pretrain + pose-defined pseudo-support-bank latent JEPA，真正换量级）。**

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp362_genoccl/design.md && sed -n '1,220p' experiments/exp362_genoccl/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 exited 1 in 0ms:
sed: experiments/exp362_genoccl/design.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp361_psc_jepa/design.md && sed -n '1,220p' experiments/exp361_psc_jepa/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
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
/bin/zsh -lc 'find experiments/exp362_genoccl -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp362_genoccl/gap_audit.py
experiments/exp362_genoccl/monitor.md
experiments/exp362_genoccl/codex_gap_decision.md

exec
/bin/zsh -lc 'find experiments/exp363_ag_foundation -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp363_ag_foundation/codex_download_cmds.md
experiments/exp363_ag_foundation/codex_frozen_verdict.md
experiments/exp363_ag_foundation/monitor.md
experiments/exp363_ag_foundation/ag_frozen_baseline.py
experiments/exp363_ag_foundation/design.md

exec
/bin/zsh -lc 'find experiments/exp364_dg_foundation -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp364_dg_foundation/codex_directft_review.md
experiments/exp364_dg_foundation/monitor.md
experiments/exp364_dg_foundation/frozen_xdomain_probe.py
experiments/exp364_dg_foundation/design.md
experiments/exp364_dg_foundation/codex_directft_review2.md
experiments/exp364_dg_foundation/codex_solider_verdict.md
experiments/exp364_dg_foundation/solider_frozen_probe.py
experiments/exp364_dg_foundation/codex_review.md
experiments/exp364_dg_foundation/codex_dinov2_verdict.md

exec
/bin/zsh -lc 'find experiments/exp365_wildlife -maxdepth 2 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp365_wildlife/.venv-wl/.lock
experiments/exp365_wildlife/.venv-wl/pyvenv.cfg
experiments/exp365_wildlife/.venv-wl/CACHEDIR.TAG
experiments/exp365_wildlife/.venv-wl/.gitignore
experiments/exp365_wildlife/__pycache__/cvpb_wildlife_localverify.cpython-311.pyc
experiments/exp365_wildlife/__pycache__/cvpb_wildlife_probe.cpython-311.pyc
experiments/exp365_wildlife/monitor.md
experiments/exp365_wildlife/cvpb_wildlife_probe.py
experiments/exp365_wildlife/cvpb_wildlife_localverify.py
experiments/exp365_wildlife/design.md

codex
本地最新几条已经覆盖了 intruder、continued-pretrain、生成/AG/DG/wildlife 等，且多条是通过零训练或短训 kill-switch 砍掉的。web 检索会避免这些同构：不再推荐 support-completion、生成遮挡引擎、foundation feature import、DG gallery 修补、watchlist/open-set。
exec
/bin/zsh -lc "sed -n '1,240p' experiments/exp362_genoccl/monitor.md && sed -n '1,240p' experiments/exp362_genoccl/codex_gap_decision.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp362 gap-measured occlusion engine — gap 审计（negative control）

## 由来
PSC-JEPA continued-pretrain（exp361）判死后，codex pivot 决策选生成数据引擎（6/10，窄缝=gap-measured occlusion distribution engine，必须赢 PLBOA）。第一步 cheap kill-switch：先 gap 审计验证前提（train-test 遮挡分布 gap 是否存在）。

## gap 审计（2026-06-27，cheap，no training，no diffusion）

**pose visibility threshold sweep（occluded_duke train N=15618 vs query N=2210）**：

| VIS_THR | legs gap | arm gap | heavy-occ(可见组≤2) gap |
|---|---|---|---|
| 0.3 | +9.2% (q9.5/t0.4) | rarm +0.5 | +0.1% |
| 0.5 | +16.4% (q17.2/t0.8) | larm +2.0 / rarm +2.1 | +1.0% |
| 0.7 | +26.1% (q28.1/t2.0) | larm +8.9 / rarm +5.9 | +5.1% |

**gap 形态确认：各 threshold 下始终主导 lower-body（legs），arm 次之（高 thr 才显现），heavy-occ 始终少。**

## ★结论：生成引擎窄缝被 PLBOA 占 → 转 LM-ReID（codex 2.5-3/10）

- gap 几乎全在 legs（lower-body），**正是 PLBOA（Pose-guided Lower-Body Occlusion，3-seed +1.37 mAP）已经在补的**。
- 生成引擎要成立 = 在 PLBOA 已覆盖的 lower-body gap 上**净增益**（否则只是"更贵的 PLBOA"）。窄缝太窄。
- caveat：pose visibility ≠ 真遮挡（codex），但 occluded_duke **无官方 mask**（真 mask audit 要跑 human parsing，中等成本）；且最强相对信号始终是 legs（非全身 heavy occlusion），caveat 不足以救生成线。
- **codex 综合判：转 LM-ReID**（exp359，诚实 6.5 B 类候选，表链全：强 TTA 对照/聚合消融/因子消融/K-sweep/backbone 泛化/σ-sweep/训练端反例）。exp362 保留为 audit/negative control。

## ★★换量级在 occluded ReID 内部的诚实困境（重大节点）

范式转向（换量级）在 occluded ReID 内部探索了多个 build，**都接近墙**：
- Intruder（exp360）DEAD：donor 可读但压它不救排序。
- PSC-JEPA continued-pretrain（exp361）DEAD：partial-view JEPA 与 ReID 判别性本质冲突（41/50 << 58.5）。
- 生成引擎（exp362）2.5-3/10：gap 被 PLBOA 占。
- support-bank aux 4/10 死：撞 SCKD 穷尽（+0.1）/FGEU 16.3%。

cheap kill-switch（Stage0.5 frozen 因果 / gap 审计纯 numpy）省了多次大投入——这是分工（先验证再大 build）的价值。**用户拍板节点**：接受"occluded ReID 内部换量级接近墙"+ 回 LM-ReID 6.5 收尾投 B 类（最务实）/ 生成引擎 human parsing 最后一搏 / 转非遮挡 reframe。
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
session id: 019f05dd-9071-7ff0-b498-1c531abaae38
--------
user
exp362 生成数据引擎 gap 审计结果(调研/决策交你, 诚实判别硬凑)。PSC-JEPA continued-pretrain 死后 pivot 生成数据引擎(gap-measured occlusion engine, 你之前说窄缝=必须赢 PLBOA)。**gap 审计(occluded_duke train N=15618 vs query N=2210 遮挡分布, pose visibility thr 0.3)**: per-group 遮挡频率 gap(query-train): head +2.3% / torso +0.1 / larm +0.1 / rarm +0.5 / **legs +9.2%(query 9.5% vs train 0.4%)**。heavy-occ(可见组<=2) gap +0.1%(train/query 都~0.1-0.2% 几乎无重遮挡)。可见组数分布 train 93% 5组全可见, query 82% 5组全可见。**关键发现: gap 几乎全在下半身 legs, 而这正是 PLBOA(Pose-guided Lower-Body Occlusion +1.37 3-seed)已经在补的**。**问**: (a)gap 主要被 PLBOA 占→生成引擎窄缝更窄(要在 PLBOA 已补的 legs gap 上赢它), 还值得短训 kill-switch 吗(生成引擎更真实/可控腿部遮挡+ID filter 比 PLBOA VOC 物体粘贴好在哪)? (b)visibility caveat: pose visibility(keypoint confidence)!=真遮挡, occluded_duke 大多 5 组全可见(thr 0.3 低估遮挡?), gap 审计可信吗? 该用 occluded_duke masks/anns 真 mask 重测遮挡分布吗? (c)综合判: 生成引擎继续短训对 PLBOA, 还是 gap 太被 PLBOA 占+occluded ReID 内部接近墙该转非遮挡 reframe(LM-ReID 6.5 现有最强 B 类候选, paper 素材全)? 诚实务实中文。
codex
我先按仓库接手规则读最新结果、决策、story 和 exp362 附近文档，避免只凭这段摘要下判断。当前沙箱只读，所以这轮我会做审计和决策判断，不会改文档。
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
/bin/zsh -lc "sed -n '1,220p' experiments/exp363_ag_foundation/design.md && sed -n '1,260p' experiments/exp363_ag_foundation/monitor.md && sed -n '1,220p' experiments/exp363_ag_foundation/codex_frozen_verdict.md" in /Users/abdslm/Desktop/SOLIDER-REID
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
# exp363 AG Foundation Adaptation — monitor

## 数据/foundation 打通（2026-06-27，3090）
AG-VPReID.VIR（本地代理 gdown 下载 → rsync 3090，all/ 54739 图像 547M）+ DINOv2-reg-B（timm `vit_base_patch14_reg4_dinov2.lvd142m`；transformers 4.46/python3.8 不支持 dinov2_with_registers → timm 退路）+ 官方 dataset_reader.py（AG_ReID_IR_Enhanced，8 protocol exp5-12）。工程坎：4090 连不上 Google→本地代理 / gdown --remaining-ok 错参数 / transformers→timm。

## frozen DINOv2-reg baseline — cheap kill-switch 第一步（无训练）

### exp7 (cross_platform aerial_ground, visible_infrared) nframes=8
| pooling | mAP | R1 |
|---|---|---|
| single | 6.87 | 3.08 |
| mean | 8.62 | 4.62 |
| max | 8.43 | 5.00 |
| topk | 7.90 | 2.69 |
| oracle | 15.11 | 8.46 |

**codex 硬判定（复杂，一过一不过）**：
- mean − single = **+1.76 < +5**（不过，视频证据积累弱）
- oracle − mean = **+6.48 > +3**（过，选择空间大）

**诚实判读（不 cherry-pick）**：严格"任一不过即杀"→ mean-single 不过 = 简单 temporal mean 没多少视频增益。但 oracle-mean +6.48 大 = 帧质量差异大、选好帧空间存在（anchored-LoRA 机会），简单 mean 没利用。整体 mAP 极低（8.62）= frozen DINOv2 在 AG cross A-G IR 域 gap 大（aerial 视角 + IR 模态 vs DINOv2 RGB 自然图预训练）。exp7 = 最难 protocol（双域 gap）。

### 全 protocol（exp5-12，nframes=8）
| protocol | mean−single | oracle−mean | mAP(mean) |
|---|---|---|---|
| exp5 cross G-A V→IR | **−1.82** | +10.07 | 7.15 |
| exp7 cross A-G V→IR | +1.76 | +6.48 | 8.62 |
| exp8 cross A-G IR→V | +0.32 | +3.36 | 1.52 |
| exp9 same G-G V→IR | +1.60 | +10.16 | 9.13 |
| exp10 same G-G IR→V | +1.82 | +6.21 | 4.89 |
| exp11 same A-A V→IR | +0.28 | +5.28 | 6.04 |
| exp12 same A-A IR→V | +0.09 | +0.63 | 0.59 |

## ★结论：视频证据积累（temporal mean）路死（2026-06-27）
- **mean−single 全 8 protocol < +5**（最高 +1.82，exp5 还 −1.82）→ codex 硬判定全不过，简单 temporal mean 无视频增益。
- oracle−mean 大部分 +3~+10（选好帧空间大），但 oracle 用真 label（upper bound），anchored-LoRA 要无 test-label 学 frame quality（可能另一个坑：quality estimation 没监督）。
- 整体 mAP 极低（IR→V 0.59-1.52，V→IR 6-9）= frozen DINOv2-reg 对 AG aerial+IR 域太弱。
- **cheap kill-switch 价值**：几小时 frozen baseline 验死"视频证据积累"核心假设，没闷头训 anchored-LoRA（省多日）。
- codex 判方向生死（codex_frozen_verdict.md）见下。

## ★★codex 判：杀 AG 主线，保留资产，切 DG/Lifelong（2026-06-27）
- (a) temporal mean 路**确认死**（7 protocol mean-single 平均 +0.58 << +5，简单视频证据积累不成立）。
- (b) frame-quality selection **是坑**（oracle=retrieval-label upper bound，无监督 quality 学成清晰度/模态/中心度非 identity utility；generic selector 不新）。除非零训练 probe（label-free top-k 稳定 +2）否则别上。
- (c) 换 foundation **不救方向**（CLIP/EVA 仍 RGB prior 不解 IR；抬绝对值 ≠ 方法创新）。最多 CLIP-L frozen sanity 对照。
- **执行**：停 AG 主线（不补 attention/view-gate/LoRA-rank 小变体=移动 kill-switch）；不先做 anchored-LoRA（oracle-chasing）；**主资源切 #2 DG/Lifelong foundation-preserving adaptation**（问题=fine-tune 时保住跨域泛化 prior，数据 Market/MSMT/Duke 现成，kill-switch 清楚）。
- 沉没成本：半天不亏（数据+DINOv2+dataloader 链路 + 干净负结论），诚实止损。
- **AG 资产保留**（数据+DINOv2+dataloader+frozen baseline 脚本 on 3090），可当 negative control / 未来对照。
- codex 深化 DG/Lifelong 中（paradigm_shift/codex_dg_deepen.md）：novelty 窄缝 / **PSC-JEPA 同质性核查（DG 会不会同样 fine-tune-harm 死）** / cheap kill-switch / 信心。
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
session id: 019f0687-33d3-7f62-ad23-13f892fe866b
--------
user
exp363 AG frozen DINOv2-reg baseline 全 protocol 结果(方向生死判, 交你, 诚实别硬撑)。你之前定的 cheap kill-switch: frozen mean vs single hard +5(视频证据积累成立)/oracle vs mean +3(选择空间留 anchored-LoRA)。**全 8 protocol [mean-single / oracle-mean / mAP(mean)]**: exp5(cross G-A V→IR) -1.82/+10.07/7.15; exp7(cross A-G V→IR) +1.76/+6.48/8.62; exp8(cross A-G IR→V) +0.32/+3.36/1.52; exp9(same G-G V→IR) +1.60/+10.16/9.13; exp10(same G-G IR→V) +1.82/+6.21/4.89; exp11(same A-A V→IR) +0.28/+5.28/6.04; exp12(same A-A IR→V) +0.09/+0.63/0.59。**关键判**: (a)mean-single 全 protocol <+5(最高+1.82, exp5 还-1.82)→视频证据积累(temporal mean)路确认死? (b)oracle-mean 大部分+3~+10(选好帧空间大), 但 oracle 用真 label 选(upper bound), anchored-LoRA 要无 test-label 学 frame quality/selection——这是真 novelty 轴还是另一个坑(quality estimation 没监督难学)? (c)整体 mAP 极低(IR→V 0.59-1.52, V→IR 6-9)=frozen DINOv2-reg 对 AG aerial+IR 域太弱, 换 foundation(CLIP/EVA-02/SigLIP)能救还是 IR 模态本身 DINO 系全弱? (d)**方向生死**: AG foundation adaptation 还值得继续吗(转 frame-quality selection novelty / 换 foundation 重测 frozen / LoRA fine-tune 直接拉域)还是诚实换你排序第2 DG/Lifelong foundation-preserving(7.5)? 我已花~半天打通 AG 全链路(数据+DINOv2+dataloader on 3090), 沉没成本不该绑架判断。务实中文诚实判。
codex
我先按仓库接手规则快速读最新结果/决策/story 和 exp363 文档，再给结论。你给的数据已经足够做生死判，但我需要确认它和现有主线叙事是否冲突。
exec
/bin/zsh -lc "pwd && rg --files experiments | rg '("'^experiments/(results|decisions|innovation_brainstorm)'"\\.md"'$|''^experiments/paper_materials/story'"\\.md"'$|''^experiments/exp363/)'"'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID
experiments/decisions.md
experiments/innovation_brainstorm.md
experiments/results.md
experiments/paper_materials/story.md

exec
/bin/zsh -lc 'git status --short' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-wq4Sc2Zf' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-osVF0fOF' (errno=Operation not permitted)
 M .claude/hooks/check_design.sh
 M .claude/rules/experiment_protocol.md
 M CLAUDE.md
 M experiments/decisions.md
 M model/backbones/swin_transformer.py
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
?? experiments/cargo_cvpb/cvpb_lm_reid_train.py
?? experiments/cargo_cvpb/cvpb_lpa_head.py
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
?? experiments/library_167_synthesis.md

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp364_dg_foundation/design.md && sed -n '1,260p' experiments/exp364_dg_foundation/monitor.md && sed -n '1,220p' experiments/exp364_dg_foundation/codex_dinov2_verdict.md" in /Users/abdslm/Desktop/SOLIDER-REID
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
# exp364 DG Foundation-Preserving Adaptation — monitor

## DG cheap kill-switch 第一步：frozen cross-domain probe（零训练，2026-06-27）

验证前提：frozen DINOv2-reg 有无跨域 ReID prior（DG 方法的核心假设是"frozen foundation 有 prior、fine-tune 破坏它"）。

### debug 历程（多轮排除假设）
1. patch14 倍数 bug（256x128 → AssertionError）→ 改 252x126
2. 252x126 mAP 1.35（反常）→ 怀疑非方形 dynamic interpolate 退化 → 改 224x224
3. 224x224 CLS mAP 1.55（还低）→ 排除尺寸；debug 确认 camid(1-6)/dim(768)/pid(750/751 Market 标准)都对
4. CLS → patch tokens GAP（skip CLS+4register）→ mAP 2.71（略好还低）

### ★结果：frozen DINOv2-reg 对行人 instance ReID 弱
| domain | CLS mAP | patch GAP mAP |
|---|---|---|
| Market | 1.55 | 2.71 (R1 8.88) |
| Occ-Duke | 0.68 | 0.89 |

- 排除尺寸/camid/dim/pooling 后，一致低（接近随机，Market frozen ReID 文献该 20-40）→ 指向 **DINOv2-reg（通用自监督）对行人 instance ReID 弱**。
- AG（exp363）frozen DINOv2 也低（8.62）一致。
- codex 查文献判中（codex_dinov2_verdict.md）：2.71 是否正常 / DG 用 DINOv2 前提是否存疑（frozen prior 弱谈何破坏）/ 换 CLIP-ReID/SOLIDER / DG 降级转 open-set。

### cheap probe 价值
零训练几轮 debug 就暴露"DINOv2 不是好 ReID foundation"，没闷头训 direct-FT 才发现 base 弱（同 AG frozen baseline 验死视频证据积累）。

## SOLIDER frozen base bounded retest（codex 首选，2026-06-27）
| foundation | Market mAP | Market R1 | Occ-Duke mAP |
|---|---|---|---|
| DINOv2-reg（通用） | 2.71 | 8.88 | 0.89 |
| **SOLIDER（行人 LUPerson 预训练）** | **15.62** | **39.96** | 3.30 |

- SOLIDER frozen 5-6x DINOv2 → 行人预训练 foundation frozen 邻域确实有身份结构。
- codex 门槛 frozen base Market >10 → SOLIDER **15.62 勉强过线**（DINOv2 2.71 死）。
- 但 15.62 不强（fine-tuned 91.6；R1 40 但 mAP 15 = top1 有信号、ranking 弱）。DG 前提靠 SOLIDER 勉强成立、prior 不强。
- `.eval()` 返回 None 坑（PSC-JEPA memory 记过）又踩一次，已 fix（分开 .to/.eval）。
## ★codex 判第二步（2026-06-27，DG 完整方向降 5.5）
- DG 完整方向 **5.5**（SOLIDER 15.62 再降）；bounded second-step kill-switch **7/10 值得做**；直接写 preservation 方法**不值得**。
- **务实：再花一次 30ep 验证，不再为 DG 追加无条件投入**。"不因 15.62 直接弃，也不因 15.62 写方法——15.62 只买一次廉价判决机会"。
- **第二步（暂不写 preservation 代码）**：head-only + direct-FT 30ep，多源 Market+Duke → held-out MSMT，SOLIDER swin。
- **判定**：direct-FT 赢 head-only/frozen（无 held-out harm）→ DG 降 3/10，转 open-set/gallery-growth；direct-FT source overfit + held-out 输 head-only → 再 λ sweep（preservation 有戏）；干净 U-shape → DG 回 6.5。
- 项目 datasets/ 有 market1501/msmt17/occluded_duke 单源 dataloader。config swin_tiny。

## ★★DG 判死（Kill，2026-06-27）：direct-FT held-out 反涨，foundation-preserving 没燃料

direct-FT SOLIDER swin_tiny on Market 30ep → probe --ckpt transformer_30 eval（同口径 0.5 norm）：

| domain | F0(frozen) | FT(direct-FT) | Δ |
|---|---|---|---|
| Market(source) | 15.56 | **88.70** | +73.1 |
| MSMT(held-out) | 4.18 | **11.37** | **+7.19** |
| Occ-Duke | 3.27 | 14.47 | +11.2 |

**codex 判定 Kill**：FT_MSMT 11.37 >> F0_MSMT+2（6.18）。full FT 不但没破坏 held-out 反而大幅提升（+7.19）→ **foundation-preserving 没燃料**（fine-tune 让所有行人域都涨，保护弱 frozen topology 15.56 无意义，没 U-shape sweet spot = DG no-op，PSC-JEPA 同质死另一形式）。

★**DG 降 3/10，转 open-set/gallery-growth/distractor-aware lifelong**（Market/MSMT/Duke 构造协议，零训练先验，不依赖 frozen prior 强）。

**今天 AG（exp363 杀）+ DG（exp364 杀）两个跳出盒子的范式方向都诚实证伪**，但全程 cheap 先验前提（frozen probe 零训练 + 一次 30ep bounded）、codex 审、诚实定位每个 bug、不被沉没成本绑架。DG 这个负结果干净（direct-FT held-out 反涨，因果清楚）。
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
session id: 019f06e4-69da-7d43-b340-28774d84bb42
--------
user
exp364 DG cheap kill-switch 第一步 frozen probe 结果异常(调研判, 交你, 诚实)。frozen DINOv2-reg-B(timm vit_base_patch14_reg4_dinov2.lvd142m)在 Market in-domain ReID mAP 极低: CLS token mAP 1.55, patch tokens GAP mAP 2.71/R1 8.88(Occ-Duke 0.89)。已排除: 尺寸(224 方形 patch14 ok)/camid 解析(1-6 对)/feat dim(768)/pid(750/751 Market 标准)/pooling(CLS 和 patch GAP 都试)。**4 问**: (a)frozen DINOv2(无 ReID fine-tune)在 Market in-domain mAP 文献正常值多少? 我的 2.71 是正常(DINOv2 通用自监督对 instance ReID 本就弱)还是 preprocessing bug(DINOv2 需特定 resize/mean-std/哪层特征)? 有没有论文报 frozen DINOv2/DINOv2-reg ReID linear/retrieval baseline 数字? (b)若 2.71 正常(DINOv2 frozen ReID 弱), DG 方向用 DINOv2 当 foundation 的核心前提'frozen foundation 有跨域 ReID prior、fine-tune 破坏它'还站得住吗? frozen prior 本就弱(2.71)谈何破坏? (c)该换 foundation 吗——CLIP(CLIP-ReID 系)/SOLIDER(行人专门预训练, 我项目有 swin)是不是更该当 DG frozen base? 还是 DG(foundation-preserving)本身因'frozen 行人 ReID prior 弱'就该降级/重评估? (d)若 DINOv2 不行, DG 第一步该换什么 frozen base 重测? 还是直接转 open-set/gallery-growth(不依赖 foundation ReID prior 强)? 务实中文诚实判。
codex
我先按仓库接手规则读最新实验文档和 exp364 相关材料，同时查公开文献里 frozen DINOv2/CLIP/SOLIDER 在 ReID 上的实际 baseline。这个问题不能靠印象判，尤其是 2.71 mAP 到底是机制失败还是预处理问题。
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

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp365_wildlife/design.md && sed -n '1,260p' experiments/exp365_wildlife/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
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
web search: 2024 2025 active query person re-identification active learning query selection person ReID
web search: 
web search: "active" "person re-identification" "query" 2024
web search: 
web search: "causal" "person re-identification" 2024
web search: 
web search: "Federated" "person re-identification" 2024 2025 "ReID"
web search: 
web search: "generative retrieval" "person re-identification"
web search: 
web search: generative retrieval identity token image retrieval 2024 paper
web search: 
web search: 2024 explainable person re-identification interpretable ReID concept bottleneck
web search: 
web search: "inverse problem" "person re-identification"
web search: 
web search: 2024 2025 synthetic data person re-identification diffusion 3D engine
web search: 
web search: 2024 weakly supervised person re-identification "ReID"
web search: 
web search: "test-time adaptation" "person re-identification" 2024
web search: 
web search: "in-context" "person re-identification"
web search: 
web search: "equivariant" "person re-identification"
web search: 
web search: "NeRF" "person re-identification" 2024
web search: 
web search: "3D" "person re-identification" "canonical"
web search: 
web search: "active person re-identification" "query" "camera"
web search: 
web search: "EvReID" person re-identification dataset code 2025
web search: 
web search: "audio" "person re-identification" "2024"
web search: 
web search: "footstep" "person identification" "gait" deep learning 2024
web search: 
web search: site:github.com Event-AHU Neuromorphic_ReID
web search: 
web search: "Re-index Free Lifelong person Re-IDentification" "Bi-C2R"
web search: 
web search: 2025 long-tail person re-identification new benchmark
web search: 
web search: "sketch person re-identification" 2025
web search: 
web search: "Diffusion" "Person Re-Identification" "ReID" 2025
web search: 
web search: "Generative Person Re-Identification"
web search: 
web search: 2025 VLM person ReID in-context instance-level recognition IIR-VLM ReID
web search: 
web search: https://arxiv.org/abs/2504.10174
web search: 
web search: https://arxiv.org/abs/2505.20540
web search: 
web search: "multi-query" "person re-identification" 2024 "active"
web search: 
web search: "multi-query" "person re-identification"
web search: 
web search: "budgeted" "person re-identification"
codex
我做完本地负结果边界和 2024-2026 web 检索后的判断：**不要硬凑 5 个。严格看“没探过 + 范式级 + 四卡可行 + 有零训练 kill-switch”，真正值得进下一步的只有 3 个，其中第 3 个有明显先例压力。**

**排序结论**

| 排名 | 方向 | 结论 | 强度 |
|---|---|---|---|
| 1 | **Active Evidence Acquisition ReID** | 最值得先做。不是 text dialogue，而是“预算化主动获取下一条身份视觉证据” | 7/10 |
| 2 | **Generative Index / ID-token ReID** | ReID 里几乎没直接先例，范式新，但高风险 | 6.5/10 |
| 3 | **Stale-index / Re-index-free Lifelong ReID** | 问题强，但 2025 已有 Bi-C2R/CFReID， novelty 要换角度 | 5.5/10 |
| 备选 | **Event/privacy ReID** | 新模态有数据趋势，但已有 SFE-Net/EvReID，且数据链路不确定 | 5/10 |

**1. Active Evidence Acquisition ReID**
问题重定义：传统 ReID 给一张 query 就必须排序；这里改成 **系统可以花 1-3 次预算主动拿更多证据**，例如请求另一帧、另一个相机视角、或让操作员回答一个二值 visual question。

2024-2026 先例：LLaVA-ReID/Inter-ReID 已经做了 witness text dialogue；ChatReID 也做 open-ended interactive retrieval。但这些主要是**文本描述补全**，不是“主动获取下一条视觉证据 / camera-view evidence acquisition”。旧 human-in-loop ReID 也有，但偏人工标注反馈，不是主动传感/证据预算。

零训练 kill-switch：
- 在 Market/MSMT/Occluded-Duke 上模拟：每个 query 允许 oracle 额外给 1 张同 ID 不同 camera 图片。
- 用现有 checkpoint 做 multi-query aggregation / MaxSim / mean feature。
- 判据：hard queries 上 `+5 mAP` 或 `+5 R1` 以上才 GO。
- 再模拟非 oracle policy：按 top-k margin/entropy 选择“最该问哪一个候选/哪一个属性”。若 policy 能拿到 oracle gain 的 50% 以上，才进入训练。

杀法很便宜：不训练，只跑特征和重排。若 oracle second evidence 都救不动，直接杀。

**2. Generative Index / ID-token ReID**
问题重定义：不再把 ReID 当 embedding nearest-neighbor，而是把 gallery identity/index 表示成离散 token/code，query 生成候选 code，再局部 rerank。注意不是 closed-set 训练 ID 分类，而是 **gallery-code generation**，否则 test ID unseen 会直接崩。

2024-2026 先例：generative retrieval 在文本/图文检索里很热，例如 AVG 把 text-to-image retrieval 改成 token-to-voken generation，DiffuGR 用 diffusion 生成 DocID；但我没查到 person ReID 里有直接“ID/code as generated tokens”的主线。这个空白是真空白，但风险也最大。

零训练 kill-switch：
- 用现有 strong ReID features 对 gallery 做 hierarchical k-means / PQ code。
- query 只预测最近 code prefix，取 top-B buckets 后再 cosine rerank。
- 判据：扫描 <10% gallery 时，正样本 recall ≥95%，mAP/R1 不明显低于全 gallery。
- 如果 code bucket recall 差，说明 ReID identity 不适合离散生成索引，杀。

这个方向成功才像范式；失败也快。

**3. Stale-index / Re-index-free Lifelong ReID**
问题重定义：真实系统不能每次更新模型就重抽历史 gallery 特征。任务变成：**新 query feature 必须能匹配旧模型抽出来的 gallery feature**。

2024-2026 先例压力很强：Bi-C2R 已经明确提出 Re-index Free Lifelong ReID；CFReID 提出 continual few-shot ReID。因此不能声称大空白。可争的窄缝是：做更严格的 **privacy/stale-gallery feature-only compatibility**，不保存旧图，只保存旧 embedding。

零训练 kill-switch：
- 取两个不同训练阶段/不同域 checkpoint：A 抽 gallery，B 抽 query。
- 比较 `B-query vs A-gallery` 与 `B-query vs B-gallery` 的 mAP drop。
- 再只用 train split feature 学一个线性/Procrustes bridge。
- 判据：stale drop ≥10 mAP 且 feature-only bridge 能回收 ≥3 mAP，才 GO。
- 如果 direct fine-tune 后 stale gallery 也没明显掉，或 bridge 没空间，杀。

这个不是最“空白”，但工程现实强，四卡可控。

**明确不推的方向**

- 因果 ReID：2025 已有 causal ReID survey，2021 DIR-ReID 已做 backdoor/domain-invariant causal framing，2026 还有 causal VI/video 方向；再做 camera-confounder 很容易变 DG 换皮。
- 联邦隐私 ReID：FedUReID、FedDG、FedARKS、CO-EVO 已把 2025-2026 空间占得很密。
- 扩散/合成数据：DLCR、DiVE、SD-ReID 已有强先例；项目 exp362 也说明 occluded 内部 gap 被 PLBOA 吃掉。
- TTT/TTA：TEMP、DART³ 已经是 ReID-specific test-time adaptation，而且用户已禁 test-time trick。
- In-context ReID：IIR-VLM 2026 已贴近 instance-level/person ReID；再做 MLLM import 会撞已死 FM-import。
- Neural rendering / 3D canonical：VAFA、CSCL 等 2D-3D/canonical view 已有，且项目已判 SMPL/3D 对 ReID 无独特信号。
- Explainable ReID：AMD、MoSAIC-ReID 等属性解释先例明显，更像分析工具，不像主性能范式。

**我建议下一步**
先开两个零训练 probe 并行：

1. `Active Evidence Acquisition` oracle second-view kill-switch。
2. `Generative Index ReID` gallery-code recall kill-switch。

两者都不用训练，半天内能知道有没有真实 headroom。若二者都死，再考虑第 3 个 stale-index lifelong；不要先碰 causal/federated/diffusion/TTT/3D。
tokens used
190,027
我做完本地负结果边界和 2024-2026 web 检索后的判断：**不要硬凑 5 个。严格看“没探过 + 范式级 + 四卡可行 + 有零训练 kill-switch”，真正值得进下一步的只有 3 个，其中第 3 个有明显先例压力。**

**排序结论**

| 排名 | 方向 | 结论 | 强度 |
|---|---|---|---|
| 1 | **Active Evidence Acquisition ReID** | 最值得先做。不是 text dialogue，而是“预算化主动获取下一条身份视觉证据” | 7/10 |
| 2 | **Generative Index / ID-token ReID** | ReID 里几乎没直接先例，范式新，但高风险 | 6.5/10 |
| 3 | **Stale-index / Re-index-free Lifelong ReID** | 问题强，但 2025 已有 Bi-C2R/CFReID， novelty 要换角度 | 5.5/10 |
| 备选 | **Event/privacy ReID** | 新模态有数据趋势，但已有 SFE-Net/EvReID，且数据链路不确定 | 5/10 |

**1. Active Evidence Acquisition ReID**
问题重定义：传统 ReID 给一张 query 就必须排序；这里改成 **系统可以花 1-3 次预算主动拿更多证据**，例如请求另一帧、另一个相机视角、或让操作员回答一个二值 visual question。

2024-2026 先例：LLaVA-ReID/Inter-ReID 已经做了 witness text dialogue；ChatReID 也做 open-ended interactive retrieval。但这些主要是**文本描述补全**，不是“主动获取下一条视觉证据 / camera-view evidence acquisition”。旧 human-in-loop ReID 也有，但偏人工标注反馈，不是主动传感/证据预算。

零训练 kill-switch：
- 在 Market/MSMT/Occluded-Duke 上模拟：每个 query 允许 oracle 额外给 1 张同 ID 不同 camera 图片。
- 用现有 checkpoint 做 multi-query aggregation / MaxSim / mean feature。
- 判据：hard queries 上 `+5 mAP` 或 `+5 R1` 以上才 GO。
- 再模拟非 oracle policy：按 top-k margin/entropy 选择“最该问哪一个候选/哪一个属性”。若 policy 能拿到 oracle gain 的 50% 以上，才进入训练。

杀法很便宜：不训练，只跑特征和重排。若 oracle second evidence 都救不动，直接杀。

**2. Generative Index / ID-token ReID**
问题重定义：不再把 ReID 当 embedding nearest-neighbor，而是把 gallery identity/index 表示成离散 token/code，query 生成候选 code，再局部 rerank。注意不是 closed-set 训练 ID 分类，而是 **gallery-code generation**，否则 test ID unseen 会直接崩。

2024-2026 先例：generative retrieval 在文本/图文检索里很热，例如 AVG 把 text-to-image retrieval 改成 token-to-voken generation，DiffuGR 用 diffusion 生成 DocID；但我没查到 person ReID 里有直接“ID/code as generated tokens”的主线。这个空白是真空白，但风险也最大。

零训练 kill-switch：
- 用现有 strong ReID features 对 gallery 做 hierarchical k-means / PQ code。
- query 只预测最近 code prefix，取 top-B buckets 后再 cosine rerank。
- 判据：扫描 <10% gallery 时，正样本 recall ≥95%，mAP/R1 不明显低于全 gallery。
- 如果 code bucket recall 差，说明 ReID identity 不适合离散生成索引，杀。

这个方向成功才像范式；失败也快。

**3. Stale-index / Re-index-free Lifelong ReID**
问题重定义：真实系统不能每次更新模型就重抽历史 gallery 特征。任务变成：**新 query feature 必须能匹配旧模型抽出来的 gallery feature**。

2024-2026 先例压力很强：Bi-C2R 已经明确提出 Re-index Free Lifelong ReID；CFReID 提出 continual few-shot ReID。因此不能声称大空白。可争的窄缝是：做更严格的 **privacy/stale-gallery feature-only compatibility**，不保存旧图，只保存旧 embedding。

零训练 kill-switch：
- 取两个不同训练阶段/不同域 checkpoint：A 抽 gallery，B 抽 query。
- 比较 `B-query vs A-gallery` 与 `B-query vs B-gallery` 的 mAP drop。
- 再只用 train split feature 学一个线性/Procrustes bridge。
- 判据：stale drop ≥10 mAP 且 feature-only bridge 能回收 ≥3 mAP，才 GO。
- 如果 direct fine-tune 后 stale gallery 也没明显掉，或 bridge 没空间，杀。

这个不是最“空白”，但工程现实强，四卡可控。

**明确不推的方向**

- 因果 ReID：2025 已有 causal ReID survey，2021 DIR-ReID 已做 backdoor/domain-invariant causal framing，2026 还有 causal VI/video 方向；再做 camera-confounder 很容易变 DG 换皮。
- 联邦隐私 ReID：FedUReID、FedDG、FedARKS、CO-EVO 已把 2025-2026 空间占得很密。
- 扩散/合成数据：DLCR、DiVE、SD-ReID 已有强先例；项目 exp362 也说明 occluded 内部 gap 被 PLBOA 吃掉。
- TTT/TTA：TEMP、DART³ 已经是 ReID-specific test-time adaptation，而且用户已禁 test-time trick。
- In-context ReID：IIR-VLM 2026 已贴近 instance-level/person ReID；再做 MLLM import 会撞已死 FM-import。
- Neural rendering / 3D canonical：VAFA、CSCL 等 2D-3D/canonical view 已有，且项目已判 SMPL/3D 对 ReID 无独特信号。
- Explainable ReID：AMD、MoSAIC-ReID 等属性解释先例明显，更像分析工具，不像主性能范式。

**我建议下一步**
先开两个零训练 probe 并行：

1. `Active Evidence Acquisition` oracle second-view kill-switch。
2. `Generative Index ReID` gallery-code recall kill-switch。

两者都不用训练，半天内能知道有没有真实 headroom。若二者都死，再考虑第 3 个 stale-index lifelong；不要先碰 causal/federated/diffusion/TTT/3D。
