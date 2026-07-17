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
session id: 019f0429-ad21-7bb1-b14a-ed0341cb9fac
--------
user
ReID 训练端范式级创新评估(重大转向, 不要 cheap 不要判死)。背景: 守 SOLIDER/Swin 强 backbone 加小模块训练端已穷尽, 用户授权换量级可花周级时间。**算力 4 单卡 slot(4090/3090/5060Ti×2)无 from-scratch foundation 算力**。数据 occluded_duke/market/msmt17/occluded_reid/occluded_posetrack。**评估范式=新人体自监督预训练 pretext**(从 SOLIDER/DINOv2 权重 continued-pretraining 新 pretext task, person 数据, 非 from-scratch 省算力)。问: (a)SOLIDER(2023 semantic-controllable SSL on LUPerson)之后, person-specific SSL 预训练新 pretext 还有什么空间(occlusion-aware completion pretext? cross-view correspondence? part-whole consistency? identity-preserving masked-modeling? 多粒度?)? (b)我 4-slot 从 SOLIDER/DINOv2 continued-pretrain(中等算力 market+msmt+occluded 数据)能 build 的最有野心新 pretext(loss/data/规模)? (c)避开 SOLIDER/LUPerson/MAE-ReID/各种 SSL-ReID, 联网查 2024-26 novelty。**不判死, 选最能 build 方案+pipeline+工作量+风险+B类潜力。信心。务实中文。**
codex
我会先做一轮近年论文/代码检索，只用论文主页、arXiv/CVF/IEEE/OpenReview 和代码仓库这类一手来源，避免凭印象判断 novelty。重点看 2024-2026 的 person-specific SSL、masked modeling、跨视角/遮挡/part consistency 是否已经被系统做过。
web search: 
web search: 2024 person re-identification self-supervised pretraining masked modeling pretext arXiv
web search: 
web search: site:openaccess.thecvf.com 2024 person re-identification self-supervised pre-training ReID
web search: 
web search: "SSL-ReID" person re-identification self-supervised learning
web search: 
web search: "Person Re-Identification" "Masked Autoencoder" "2024" "pre-training"
web search: 
web search: "PersonViT" "PersonMAE" "ReID" 2025
web search: 
web search: "DINOv2" "person re-identification" "2024"
web search: 
web search: "HumanBench" "self-supervised" "human-centric" "ReID"
web search: 
web search: "Part-Aware Self-Supervised" "ReID" 2024 follow-up
web search: 
web search: "ReID-specific pre-training" "2024" "person"
web search: 
web search: "Diffusion-ReID" "pre-training" "person re-identification" "2024"
web search: 
web search: https://arxiv.org/abs/2303.17602
web search: 
web search: 'RegionA' in https://arxiv.org/abs/2311.04496
web search: 
web search: CVPR 2025 person re-identification self-supervised pretraining
web search: 
web search: https://arxiv.org/abs/2601.20598
web search: 
web search: DINOv2 Learning Robust Visual Features without Supervision arXiv 2023
web search: 
web search: DINOv3 Self-supervised learning for vision at unprecedented scale paper
web search: 
web search: https://arxiv.org/abs/2508.10104
web search: 
web search: ReMix Training Generalized Person Re-identification on a Mixture of Data 2024 arxiv
web search: 
web search: "occluded" "person re-identification" "pre-training" 2024
web search: 
web search: HAP Structure-Aware Masked Image Modeling Human-Centric Perception code github
web search: 
web search: https://arxiv.org/abs/2310.20695
web search: 
web search: https://arxiv.org/abs/2410.15613
web search: 
web search: "Focus and imagine" "Occlusion suppression and repairing transformer" 2024 occluded person re-identification
web search: 
web search: I-JEPA self-supervised learning from images joint embedding predictive architecture arxiv 2023
web search: 
web search: HumanBench projector assisted pretraining human-centric perception arxiv 2023
web search: 
web search: https://arxiv.org/abs/2503.08201
web search: 
web search: "identity-preserving" "masked image modeling" person re-identification
codex
检索到几个关键边界：2023 后不是一片空白，HAP/PersonMAE 已经把 human part-guided MIM、遮挡式区域预测、结构一致性做过；2024 PersonViT 又把 MIM + discriminative contrastive 做成 ReID 大预训练。接下来我会对照你仓库里的 exp109/exp148 语境，把“还能做什么”收紧到不和这些工作正面重名的方案。
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
本地结果表比接手说明更新，已经推进到 exp194/Phase 4；但你这次问题是“训练端范式级转向”，我不会继续沿已有小模块链条判断。下一步我只抽取与 pretext 相关的失败/正信号，尤其是 PCVT、masking/completion、OA-SD/parallel aug 这几类。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp148/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 实验 exp148: PCVT（Pose-Complementary View Training）

## 动机

`exp109` 证明真正的 headroom 在 `single-image support incomplete`。  
但 `exp110-142` 也反复证明：把 same-ID 跨图 support bank 直接蒸到单图特征里，很难在 15K 数据上学成。

因此这次不再做 cross-image completion，也不再做 retrieval scorer 小修补，而是直接改训练对象：

**让一张图在训练时变成两张“互补可见”的伪视图。**

核心问题不是“如何从别的图补我”，而是：
**能否让同一张图自己提供两份互补 support，从而逼迫编码器学会对 partial support 稳定？**

## 核心假设

1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
2. 如果两张增强视图是由 pose 定义的互补遮挡，那么：
   - 单独看每张视图都不完整
   - 合起来却接近完整 support
3. 在这种设置下训练得到的 backbone，应比普通 `exp030a` 更擅长处理真实遮挡

## 技术方案

### 1. 数据视图构造：从随机增强改成 pose-defined complementary views

对每张训练图：

1. 用 scene heatmap / person-0 keypoint response 找到可见 body groups  
   默认 groups:
   - head
   - left_arm
   - right_arm
   - torso
   - left_leg
   - right_leg

2. 将“当前可见”的 groups 做一次平衡划分，得到互补两组 `A/B`
   - 尽量让两组总响应面积接近
   - 保证 `A` 与 `B` 不重叠

3. 生成三张训练视图
   - `view_full`: 原图标准增强
   - `view_a`: 遮掉 `A`
   - `view_b`: 遮掉 `B`

这样 `view_a` 与 `view_b` 的 body support 是互补的，而不是两个随机遮挡版本。

### 2. 三视图共享 backbone

三张图都走同一个 `exp030a` 主干：
- full view
- complement view A
- complement view B

### 3. 训练目标

总损失分为两部分：

#### (a) ReID 主损失
对三张视图都计算标准 `ID + Triplet`

目的：
- 不是只让 masked view 对 full 做对齐
- 而是要求每个 partial view 本身也能保留身份判别性

#### (b) Complement-Union Consistency
定义：
- `f_full`
- `f_a`
- `f_b`
- `f_union = 0.5 * (f_a + f_b)`（第一版先不用新参数）

约束：
- `f_union` 应比 `f_a`、`f_b` 单独更接近 `f_full`

实现第一版用：
- cosine consistency / MSE consistency（二选一，优先 cosine）

这一步的核心不是“重建像素”，而是：
**让互补 partial supports 在表示空间里重新合成接近完整 support 的身份特征。**

## 对照组

- 主基线：`exp030a-eq`
- 间接历史参考：
  - `exp050 PAMC`：body-aware masking consistency，中性
  - `exp067 ROA`：遮挡增强 recipe
  - `exp142 SKC`：feature-level completion，负

本实验的价值在于它与这三条都不同：
- 不只是 consistency
- 不只是增强
- 不只是 completion module

## 预期结果

若方向成立，应该出现：

1. `view_a/view_b` 单独性能较低，但 full 主分支最终更强
2. `union consistency` 的统计明显优于单视图 consistency
3. 最终 `exp148` 应优于 `exp030a`
4. 若成立，它会比 retrieval-side 小 scorer 更像论文主贡献

## 关键日志

训练期必须额外记录：

- `pcvt_lc`: complement consistency loss
- `pcvt_cov_a`: A 视图保留的可见 support 比例
- `pcvt_cov_b`: B 视图保留的可见 support 比例
- `pcvt_cov_u`: A/B 联合 support 比例
- `pcvt_ovr`: A/B overlap 比例
- `pcvt_mga`: A 视图平均 mask 面积
- `pcvt_mgb`: B 视图平均 mask 面积
- `pcvt_cos_fa`: `cos(f_full, f_a)`
- `pcvt_cos_fb`: `cos(f_full, f_b)`
- `pcvt_cos_fu`: `cos(f_full, f_union)`
- `pcvt_gap`: `cos(f_full, f_union) - 0.5*(cos_fa + cos_fb)`

如果这些日志缺失，这次实验直接算不可解释 run。

## 风险与失败解释

1. 若 `pcvt_cov_u` 明显高于单视图，但结果仍无提升  
   说明“伪多 support”这个训练对象本身不够有用

2. 若 `pcvt_gap <= 0`  
   说明简单 `avg(f_a, f_b)` 并没有真的形成 union support

3. 若 masked views 太难，主损失明显恶化  
   说明 complementary masking 过重，需要收紧 partition / mask 策略

4. 若结果只是和 `PAMC` 一样中性  
   说明单图伪多 support 这条训练范式也很可能不足以支撑主创新

exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp148/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp148 PCVT 监控

## 实验信息
- 方法: PCVT（Pose-Complementary View Training）
- 类型: 训练范式大改动
- 主基线: `exp030a-eq`
- 当前状态: 已通过审查并正式启动训练

## 启动前检查清单
- [ ] 数据管线支持生成 `full / view_a / view_b`
- [ ] complement partition 只依赖当前图 pose
- [ ] 三视图主损失与 `exp030a` 对齐
- [ ] `pcvt_*` 行为日志接好
- [ ] Claude 广范围审查通过

## 当前判断
- 这是对 `support incomplete` 的训练范式级重写，不是小调参
- 若实现失败，也应能明确排除“单图伪多 support”这条主线

## 启动记录

### [2026-03-22 09:58] 代码接入完成，开始工程自检
- 已修改:
  - `config/defaults.py`
  - `datasets/make_dataloader.py`
  - `datasets/pose_dataset.py`
  - `processor/processor.py`
  - `configs/occluded_duke/pose_psg_gcn_pcvt.yml`
- 关键实现点:
  1. 数据侧新增 `PCVT` 三视图：`full / view_a / view_b`
  2. `view_a/view_b` 不再是随机增强，而是由 person-0 pose heatmap 的 body-group 响应做互补划分
  3. 训练侧新增 `pcvt_lc`，约束 `0.5*(f_a+f_b)` 逼近 `f_full`
  4. 行为日志已接入 `pcvt_*`

### [2026-03-22 10:08] 数据级 probe 通过，并修正了“伪互补”问题
- 初版 probe 发现:
  - `pcvt_cov_u ≈ 0.74`
  - `pcvt_ovr ≈ 0.26`
- 原因:
  - part mask 直接取各自 active 区域，空间上会重叠，导致两张视图并不是真正互补
- 已修正:
  - 改为像素级独占分配：每个可见像素只归属响应最大的 body group
- 修正后新 probe:
  - `pcvt_cov_a ≈ 0.499`
  - `pcvt_cov_b ≈ 0.501`
  - `pcvt_cov_u = 1.000`
  - `pcvt_ovr = 0.000`
  - `pcvt_fb = 0.000`
- 当前判断: 现在才算真正实现了“互补 support”

### [2026-03-22 10:18] 模型级 probe 通过，训练路径闭合
- 使用 `pose_psg_gcn_pcvt.yml` 构造:
  - dataloader
  - `PSG+GCN` 模型
  - `loss_fn`
- 取一个 `bs=8` 的真实 batch 做三视图前向
- 结果:
  - `main_loss = 12.296`
  - `pcvt_lc = 0.317`
  - `pcvt_gap = 0.047`
  - `pcvt_cov_u = 1.000`
- 解释:
  1. 主损失与新增 `pcvt_lc` 可以同时正常计算
  2. `pcvt_gap > 0`，说明 union 表示初始就略优于单个 partial view
  3. 当前已具备送 Claude 做广范围审查的工程完整性

### [2026-03-22 10:20] 当前判断
- 继续
- 原因:
  1. 这不是空设计，数据/模型/日志三条链已打通
  2. 下一步不是直接开跑，而是按规则先做广范围 Claude 审查

### [2026-03-22 10:36] Claude 广范围审查通过，可启动
- 审查文件:
  - `experiments/exp148/claude_review.md`
- 审查结论:
  - 无阻塞问题
  - 可以启动，但必须显式记录风险并按风险监控

### 审查指出的关键风险

1. **主损失被 1/3 稀释**
   - 当前训练总损失是 `(L_full + L_a + L_b) / 3 + 0.25 * L_pcvt`
   - 这意味着 full view 主任务权重从 `1.0` 变成了 `1/3`
   - 如果最终结果为负，不能直接说 `PCVT` 想法失败，必须先考虑主损失 dilution

2. **3x 前向导致 OOM 风险**
   - 三个视图都保留梯度图
   - 首个 epoch 必须重点盯 3090 显存

3. **masked views 仍共享完整 pose heatmap**
   - 这不是 bug，而是设计选择
   - 如果 `pcvt_cos_fa/fb` 异常低，需要优先怀疑这个矛盾信号

### 明确止损判据

1. 若首个 epoch OOM，立即停，先考虑降低 partial-view 反向强度，而不是继续硬跑
2. 若 `pcvt_cov_u` 明显偏离 `1.0` 或 `pcvt_ovr` 明显高于 `0`，说明互补视图构造失效，立即停
3. 若 `pcvt_gap` 长期 `<= 0`，说明 union 表示没有优于单视图，这条线的核心假设会被直接动摇

### [2026-03-22 10:40] 官方训练启动，本地主卡占用正常
- 启动方式:
  - PTY 持续训练会话
- 输出目录:
  - `log/occluded_duke/exp148_pcvt`
- 启动确认:
  - 日志明确打印 `[PCVT] enabled: weight=0.25, resp_thr=0.1, act_thr=0.3, min_parts=2`
- 显存观察:
  - RTX 3090 占用约 `18.8GB`
- 当前判断:
  - 继续
  - 原因:
    1. 三视图前向已真实接上
    2. 显存虽高但未 OOM，仍在可承受区间

### [2026-03-22 10:44] 首个 epoch 早期形态健康，PCVT 机制已真实生效
- 训练日志摘录:
  - `Epoch[1] Iter[20/227] Loss: 21.733 | pcvt_lc: 0.320 | pcvt_cov_u: 1.000 | pcvt_ovr: 0.000 | pcvt_cos_fu: 0.680 | pcvt_gap: 0.049`
  - `Epoch[1] Iter[200/227] Loss: 14.700 | pcvt_lc: 0.205 | pcvt_cov_u: 1.000 | pcvt_ovr: 0.000 | pcvt_cos_fu: 0.795 | pcvt_gap: 0.047`
  - `Epoch[2] Iter[120/227] Loss: 10.729 | pcvt_lc: 0.105 | pcvt_cov_u: 1.000 | pcvt_ovr: 0.000 | pcvt_cos_fu: 0.895 | pcvt_gap: 0.032`
- 关键观察:
  1. `pcvt_cov_a/b` 始终稳定在 `0.496~0.504`
  2. `pcvt_cov_u=1.000`、`pcvt_ovr=0.000` 持续成立，说明互补视图构造没有漂
  3. `pcvt_lc` 从 `0.320 -> 0.105` 明显下降
  4. `pcvt_cos_fu` 从 `0.680 -> 0.895` 快速上升
  5. `pcvt_gap` 仍为正，但从 `0.049` 收到 `0.032`
- 当前判断:
  - 继续
  - 原因:
    1. `PCVT` 已被真实测试到，不是失效 run
    2. 目前 union 表示确实优于单 partial view
    3. 下一关键点是 `ep10/20`，看主任务会不会被 1/3 dilution 拖住

### [2026-03-22 10:50] Epoch 1-5 持续健康，PCVT 不是只会优化额外一致性
- 当前进度:
  - `Epoch 5 Iter 120/227`
- 关键轨迹:
  - `Epoch1 Iter20`: `Loss=21.733`, `pcvt_lc=0.320`, `pcvt_cos_fu=0.680`, `pcvt_gap=0.049`
  - `Epoch2 Iter120`: `Loss=10.729`, `pcvt_lc=0.105`, `pcvt_cos_fu=0.895`, `pcvt_gap=0.032`
  - `Epoch3 Iter200`: `Loss=8.358`, `pcvt_lc=0.049`, `pcvt_cos_fu=0.951`, `pcvt_gap=0.017`
  - `Epoch4 Iter200`: `Loss=7.523`, `pcvt_lc=0.030`, `pcvt_cos_fu=0.970`, `pcvt_gap=0.011`
  - `Epoch5 Iter120`: `Loss=7.068`, `pcvt_lc=0.025`, `pcvt_cos_fu=0.975`, `pcvt_gap=0.010`
- 稳定不变的结构指标:
  - `pcvt_cov_a/b ≈ 0.496~0.504`
  - `pcvt_cov_u = 1.000`
  - `pcvt_ovr = 0.000`
  - `pcvt_fb = 0.000`
- 当前观察:
  1. union-consistency 在持续变强，但 `pcvt_gap` 正在快速收窄
  2. 这说明 `f_a / f_b / f_union` 都在一起向 `f_full` 靠近，而不是只有 union 单独变好
  3. 主损失也在同步下降，所以现在还看不出“三视图主任务 dilution”已经把优化拖坏
- 当前判断:
  - 继续
  - 原因:
    1. 目前没有出现 OOM、互补失效、或 `pcvt_gap<=0`
    2. 下一真正有信息量的点仍是 `ep10`

### [2026-03-22 10:53] `ep10/20/30` 已出，PCVT 形成稳定的 `mAP` 正向
- 验证结果:
  - `ep10 = 40.2 / 51.4`
  - `ep20 = 49.1 / 60.7`
  - `ep30 = 54.6 / 65.8`
- 对照 `exp030a`:
  - `ep10 = 38.2 / 51.3`
  - `ep20 = 46.8 / 60.9`
  - `ep30 = 52.2 / 66.0`
- 现阶段差值:
  - `ep10: +2.0 mAP / +0.1 R1`
  - `ep20: +2.3 mAP / -0.2 R1`
  - `ep30: +2.4 mAP / -0.2 R1`
- 机制侧观察:
  1. `pcvt_lc` 已收敛到 `0.015~0.024`
  2. `pcvt_cos_fu` 稳在 `0.976~0.985`
  3. `pcvt_gap` 仍为正，但已缩到 `0.006~0.009`
  4. `pcvt_cov_u=1.000` 与 `pcvt_ovr=0.000` 仍稳定成立
- 当前判断:
  - 继续，优先级升高
  - 原因:
    1. 这不是单纯一致性 loss 好看，验证端已经连续 3 个点呈现 `mAP` 正向
    2. 目前风险不在“方法没接上”，而在于它会不会最终变成 `mAP` 换 `R1` 的 trade-off
    3. 下一关键点是 `ep40/50`

### [2026-03-22 15:48] `epoch32` 仍稳定推进，机制未塌
- 当前进度:
  - 已运行到 `Epoch 32`
- 训练侧观察:
  1. `pcvt_cov_u = 1.000`、`pcvt_ovr = 0.000` 持续成立
  2. `pcvt_lc ≈ 0.024~0.026`
  3. `pcvt_cos_fu ≈ 0.974~0.976`
  4. `pcvt_gap ≈ 0.008~0.009`，仍为正
- 当前判断:
  - 继续
  - 原因:
    1. `PCVT` 到中期仍没有出现 collapse、视图互补失效、或 `gap<=0`
    2. 目前最关键的已不是”接没接上”，而是后续验证是否能把 `mAP` 正向转成更完整的收益形态

### [2026-03-22 23:55] 新会话接手，`epoch38` 稳定推进中
- 当前进度:
  - 已运行到 `Epoch 38`（约 31.7%）
  - ETA ≈ 3h，预计凌晨 3 点左右完成
- 训练侧观察（ep33-38 取样）:
  1. `pcvt_cov_u = 1.000`、`pcvt_ovr = 0.000` 仍完美成立
  2. `pcvt_lc ≈ 0.024~0.026`，稳定未变
  3. `pcvt_cos_fu ≈ 0.974~0.976`，稳定
  4. `pcvt_gap ≈ 0.008~0.009`，仍为正
  5. 主损失正常下降：`ep33 Iter200 Loss=0.802` → `ep38 Iter120 Loss=0.655`
  6. `id_global` 从 `~1.0` 降到 `~0.84`，`id_part` 从 `~0.55` 降到 `~0.47`
  7. GPU 显存 `18790 MiB`（93% utilization），稳定
- 与 ep30 对比:
  - 训练正常推进，机制指标无任何恶化
  - 下一关键点是 `ep40` 验证
- 当前判断:
  - 继续
  - 原因:
    1. 所有 PCVT 行为指标仍在健康范围
    2. 目前需要关注的是 `ep40` 验证结果是否延续 `mAP` 正向
    3. 远程服务器空闲，应启动一个不同方向实验

### [2026-03-23 00:25] `ep40/50` 验证出炉，PCVT 正向在收窄

| Epoch | PCVT | exp030a | Δ mAP | Δ R1 |
|-------|------|---------|-------|------|
| 10 | 40.2 / 51.4 | 38.2 / 51.3 | **+2.0** | +0.1 |
| 20 | 49.1 / 60.7 | 46.8 / 60.9 | **+2.3** | -0.2 |
| 30 | 54.6 / 65.8 | 52.2 / 66.0 | **+2.4** | -0.2 |
| 40 | 56.7 / 67.9 | 55.6 / 68.6 | **+1.1** | -0.7 |
| 50 | 57.0 / 68.5 | 55.7 / 68.8 | **+1.3** | -0.3 |

- 关键观察:
  1. mAP 正向仍然成立（ep40 +1.1, ep50 +1.3）
  2. 但正向从 ep30 的 +2.4 收窄到 ep40-50 的 +1.1~1.3
  3. R1 始终为微负或持平（-0.2 ~ -0.7）
  4. 这与 exp148 design.md 中预警的"mAP 换 R1 trade-off"一致
  5. PCVT 仍有 ~1% mAP 净正向
- 当前判断:
  - 继续
  - 原因:
    1. +1.1~1.3 mAP 虽然收窄了，但仍是正向
    2. exp030a 在 ep50→120 还会从 55.7→61.1，PCVT 也可能继续提升
    3. 如果最终差值保持 +1%，对论文有消融价值
    4. 下一关键点：ep60

### [2026-03-23 00:50] `ep60` 验证出炉，mAP 正向继续收窄，R1 转负

| Epoch | PCVT | exp030a | Δ mAP | Δ R1 |
|-------|------|---------|-------|------|
| 10 | 40.2 / 51.4 | 38.2 / 51.3 | **+2.0** | +0.1 |
| 20 | 49.1 / 60.7 | 46.8 / 60.9 | **+2.3** | -0.2 |
| 30 | 54.6 / 65.8 | 52.2 / 66.0 | **+2.4** | -0.2 |
| 40 | 56.7 / 67.9 | 55.6 / 68.6 | **+1.1** | -0.7 |
| 50 | 57.0 / 68.5 | 55.7 / 68.8 | **+1.3** | -0.3 |
| 60 | 58.5 / 69.7 | 57.7 / 70.8 | **+0.8** | **-1.1** |

- 关键观察:
  1. mAP 正向从 ep30 的 +2.4 持续下降到 ep60 的 +0.8
  2. R1 差距扩大到 -1.1，呈现明确 mAP vs R1 trade-off
  3. 趋势判断：如果最终收敛到 mAP +0.5~1.0 / R1 -1.0~-1.5 的区间，对论文价值有限
  4. 但还不确定这个趋势是否会在 ep80-120 阶段反转
- 风险评估:
  1. 如果 ep80 mAP 正向进一步收窄到 < +0.5 → 基本说明 PCVT 不如预期

exec
/bin/zsh -lc "find experiments -maxdepth 2 -type f \\( -name 'design.md' -o -name 'monitor.md' \\) | sort | tail -30" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp336_swin_lgpa_nopsg/monitor.md
experiments/exp337_swin_lgpa_nopose/design.md
experiments/exp340_swin_lgpa_fixedbands/design.md
experiments/exp340b_fixedbands_undetach/design.md
experiments/exp340c_randomtext/design.md
experiments/exp341_clip_id_prompt/design.md
experiments/exp341base_noprompt/design.md
experiments/exp342_clip_id_pose/design.md
experiments/exp343_pose_guided_clip/design.md
experiments/exp344_pose_cond_prompt/design.md
experiments/exp345_pose_part_clip/design.md
experiments/exp347_noparam_deocc/design.md
experiments/exp348_occ_repel/design.md
experiments/exp349_small_full_clip/design.md
experiments/exp349b_small_undetach_clip/design.md
experiments/exp350_undetach_clean/design.md
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
experiments/exp359_lm_reid/monitor.md
experiments/smpl_anchor_vireid/design.md
experiments/smpl_anchor_vireid/monitor.md

exec
/bin/zsh -lc 'rg -n "exp109|single-image|support|completion|PCVT|OA-SD|Parallel Aug|pretrain|pretext|DINO|SOLIDER|MAE" experiments -S' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp296_base_lr8_s41/monitor.md:5:- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp296_base_lr8_s41/train_log.txt`
experiments/exp296_base_lr8_s41/monitor.md:7:- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG `[-2,-1]`)
experiments/exp212/design.md:1:# 实验 exp212: Small GCN+PAA+CE+OA-SD LR=0.0008 (Higher LR)
experiments/module_candidates.md:36:- 需要先由候选 1 证明 keypoint-level common-support 确实有效
experiments/module_candidates.md:46:- 用 skeleton branch 的 `kp_weights` 构造 batch 内 pairwise common-support overlap
experiments/module_candidates.md:47:- 在 global branch 上增加一条 support-aware triplet：
experiments/module_candidates.md:53:- 它不是再加一个 branch 模块，而是把 **pair-specific common support** 迁进训练目标
experiments/module_candidates.md:78:  2. CSGT（训练端 common-support mining）
experiments/module_candidates.md:220:2. 若 `TDPC` 单 seed 2-3 天内无正信号，再回退到 retrieval-time `common-support recovery`
experiments/exp350_undetach_clean/design.md:4:- Step1（exp341）成功：CLIP-ReID 可学习 ID prompt 在 SOLIDER 上 **+2.2**（59.8 vs 57.6 matched）。找到了能涨的 CLIP 机制。
experiments/exp198/design.md:1:# 实验 exp198: STM + OA-SD + CE (远程)
experiments/exp198/design.md:5:- 远程 16GB 无法跑 3-view，用 1-view + OA-SD + CE + STM
experiments/exp198/design.md:6:- 测试 STM 在 OA-SD 路线下的效果（与 exp191 对照）
experiments/exp198/design.md:9:STM（token mixup）和 OA-SD（self-distillation）是正交的增强手段，应能叠加。
experiments/exp198/design.md:12:- 配置 = exp191 (OA-SD + CE) + POSE_STM=True
experiments/exp198/design.md:14:- 远程 16GB: 1-view + OA-SD + STM
experiments/exp198/design.md:17:- exp191 (OA-SD + CE, no STM): 63.2/75.4
experiments/exp265b_best_s_op_s41/codex_review.md:11:srvA 刚 resume (GPU 空闲 + OP 数据齐全 + pretrained 齐全), 立即利用。
experiments/exp188/design.md:1:# 实验 exp188: Occlusion-Asymmetric Self-Distillation (OA-SD)
experiments/exp188/design.md:10:- 与 PersonMAE 的区别：distill identity-level tokens 而非 reconstruct pixels
experiments/exp188/design.md:35:2. `datasets/pose_dataset.py`: OA-SD mode 保存 pre-PLBOA clean image
experiments/exp188/design.md:53:- decay = 0.999 (标准 DINO/BYOL 值)
experiments/exp188/design.md:60:- 消融变量: 仅增加 OA-SD distillation loss
experiments/exp213/design.md:1:# 实验 exp213: Small GCN+PAA+CE+OA-SD + PKC(0.05) + MST(0.1)
experiments/exp241/monitor.md:1:# exp241 Tiny + PPA + GCN 双分支 + OA-SD 监控
experiments/exp241/monitor.md:3:配置: Tiny + PSG + PPA (w=0.5) + GCN (detached) + OA-SD + PLBOA(0.7)
experiments/exp212/monitor.md:1:# exp212 Small + GCN+PAA+CE+OA-SD LR=0.0008 监控
experiments/exp246/design.md:17:- 其他: OA-SD + PLBOA(0.7) + PSG
experiments/exp198/monitor.md:1:# exp198 STM + OA-SD + CE (远程) 监控
experiments/exp198/monitor.md:3:配置: base arch + OA-SD + STM (num_swap=2, prob=0.5, weight=0.5)
experiments/exp198/monitor.md:4:对照: exp191 (OA-SD + CE, no STM): 63.2/75.4
experiments/exp198/monitor.md:21:**观察**: STM + OA-SD 正常共存。stm_n=64 ✓。
experiments/exp198/monitor.md:35:| **exp198 (STM+OA-SD+CE)** | **36.7/49.1** | **48.7/62.0** |
experiments/exp198/monitor.md:36:| exp191 (OA-SD+CE) | 34.3/46.8 | 46.0/58.0 |
experiments/exp198/monitor.md:39:**STM 在 OA-SD+CE 路线效果巨大！增益持续扩大。**
experiments/exp198/monitor.md:40:token 级别的 cross-instance mixup 与 OA-SD self-distillation 高度协同。
experiments/exp198/monitor.md:50:| Epoch | exp198 (STM+OA-SD) | exp191 (OA-SD only) | delta |
experiments/exp198/monitor.md:76:2. **STM 在 OA-SD+CE 路线不改善最终结果** — 只加速收敛
experiments/exp213/monitor.md:1:# exp213 Small + GCN+PAA+CE+OA-SD + PKC(0.05) + MST(0.1) 监控
experiments/exp109/design.md:1:# 实验 exp109: Oracle Support Bank 上界诊断
experiments/exp109/design.md:6:- 但 `SGCFR` 明确证明：**跨图 support recovery** 确实能带来大增益
experiments/exp109/design.md:8:  **batch 内没有足够稳定的 same-ID support**
experiments/exp109/design.md:9:- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
experiments/exp109/design.md:10:  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**
experiments/exp109/design.md:14:1. 若单图表征真的受限于“support 不完整”，那么用同 ID 多图构造 oracle prototype 后，matching 应明显优于原始 `cvk_hybrid`
experiments/exp109/design.md:19:3. 若 oracle 上界都很小，则说明 training-time support-complete distillation 很难成为主线，应立即止损
experiments/exp109/design.md:30:### 2. Oracle support bank 构造
experiments/exp109/design.md:55:  - 说明 “support-complete teacher” 这条训练主线有真实 headroom
experiments/exp109/design.md:59:  - support bank 训练线应止损
experiments/exp109/design.md:66:   - 说明关键在 latent feature completion，而不是 visibility correction
experiments/exp241/claude_review.md:9:e. OA-SD 兼容性：teacher model 同结构
experiments/exp241/claude_review.md:72:## e. OA-SD 兼容性
experiments/exp241/claude_review.md:76:OA-SD distillation 路径（行 756-764）：
experiments/exp241/claude_review.md:91:代码逻辑正确，梯度隔离清晰，loss 处理无误，测试路径输出一致，OA-SD 兼容。
experiments/exp265b_best_s_op_s41/design.md:13:- 其他参数完全不变: Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC + PSG [-2,-1])
experiments/exp212/claude_review.md:1:# exp212 Claude Review — Small GCN+PAA+CE+OA-SD LR=0.0008
experiments/exp212/claude_review.md:28:这是纯超参数调整，不是创新实验。但 design.md 并未声称这是创新，而是定位为"LR 消融"。作为 supporting experiment 是可接受的。只要不作为主线创新方向即可。
experiments/exp212/claude_review.md:89:| OA-SD | Yes | Yes |
experiments/exp188/monitor.md:1:# exp188 OA-SD (Occlusion-Asymmetric Self-Distillation) 监控
experiments/exp188/monitor.md:24:**观察**: OA-SD 正常工作。oa_sd loss 从 0.40 上升到 0.48（EMA teacher 尚未充分更新，student 快速学习导致差距增大）。
experiments/exp188/monitor.md:48:| 指标 | exp188 (OA-SD+SupCon) | exp176 (SupCon only) | delta |
experiments/exp188/monitor.md:59:| 指标 | exp188 (OA-SD+SupCon) | exp176 (SupCon only) | delta |
experiments/exp188/monitor.md:65:**观察**: ep20 追平 exp176。OA-SD 不再落后。oa_sd=0.10 说明 teacher-student 已接近。
experiments/exp188/monitor.md:71:| Epoch | exp188 (OA-SD+SupCon) | exp176 (SupCon only) | delta |
experiments/exp188/monitor.md:77:**观察**: ep30 OA-SD R1 +4.2！类似 exp187 (parallel aug) 的模式但只需 2x forward（不是 3x）。
experiments/exp188/monitor.md:84:| Epoch | OA-SD+SupCon | SupCon only | delta |
experiments/exp188/monitor.md:90:**观察**: OA-SD 持续产生 R1 正向信号（+2.5~4.2）。类似 parallel aug 的效果但更轻量（2x vs 3x forward）。
experiments/exp188/monitor.md:96:| Epoch | OA-SD+SupCon | SupCon only | delta |
experiments/exp188/monitor.md:109:| Epoch | OA-SD+SupCon | SupCon only | delta |
experiments/exp188/monitor.md:115:**观察**: R1 从 +4.2(ep30) 到 -1.0(ep60)。OA-SD 早期加速但后期消退。类似 DPTL、PLTD 的模式。
experiments/exp188/monitor.md:116:Remote exp191 (OA-SD+CE) ep30 = 50.6/61.7（也落后 -1.0 R1）。
experiments/exp188/monitor.md:122:| Epoch | OA-SD+SupCon | SupCon only | delta |
experiments/exp188/monitor.md:128:**观察**: ep70 持平。OA-SD 的 ep30 R1 +4.2 优势完全消退。
experiments/exp188/monitor.md:129:OA-SD 只加速了早期收敛，没有改善最终质量。类似 DPTL、PLTD 的模式。
experiments/exp188/monitor.md:130:但 Remote exp191 (OA-SD+CE) ep40 反超了 CE (+2.0/+1.6)！OA-SD 可能在 CE 下后期有效。
experiments/exp188/monitor.md:136:| Epoch | OA-SD+SupCon | SupCon only | delta |
experiments/exp188/monitor.md:142:**观察**: OA-SD+SupCon 在 ep80 落后 exp176。确认中性偏负。
experiments/exp188/monitor.md:143:OA-SD 只加速了早期收敛（ep30 +4.2 R1），但没有改善最终质量。
experiments/exp188/monitor.md:148:| Epoch | OA-SD+SupCon | SupCon only | delta |
experiments/exp188/monitor.md:154:**观察**: OA-SD+SupCon 持续落后。但 Remote exp191 (OA-SD+CE) ep50=59.0/70.6 (+1.6/+1.5 vs CE)!
experiments/exp188/monitor.md:155:OA-SD 与 CE 正向，与 SupCon 负向。可能 distillation 与 contrastive 梯度冲突。
experiments/exp188/monitor.md:168:**结论**: OA-SD + SupCon 中性偏负。Distillation 与 contrastive loss 有梯度冲突。
experiments/exp188/monitor.md:169:但 Remote exp191 (OA-SD + CE) 有巨大正效果 (+3.8/+4.4@ep60)！OA-SD 与 CE 正 synergy。
experiments/exp199/design.md:4:- OA-SD (per-token feature distillation) 与 SupCon 互斥 — exp188/196 已证明
experiments/exp199/design.md:5:- 根本原因：OA-SD 在 token 级别 match 个体特征，与 SupCon 在 token 级别做对比学习梯度冲突
experiments/exp199/design.md:10:将 self-distillation 从 "feature-level matching" (OA-SD) 升级为 "relation-level matching" (OA-RD)，可以在不与 SupCon 冲突的情况下获得 occlusion invariance。
experiments/exp199/design.md:33:1. EMA teacher 机制复用 OA-SD 的代码（PLBOA asymmetry、decay、EMA update）
experiments/exp199/design.md:41:2. `processor/processor.py`: 在 OA-SD 代码块旁边新增 OA-RD 逻辑
experiments/exp199/design.md:56:- exp196 (3-view + SupCon + OA-SD global-only): 62.4/75.2 (失败)
experiments/exp199/design.md:57:- exp193 (3-view + OA-SD + CE): 64.4/76.5
experiments/exp199/design.md:62:3. ✅ 证据层面：OA-SD 失败 (exp188/196) → OA-RD 成功 = 清晰的对照链
experiments/exp198/claude_review.md:1:# exp198 STM + OA-SD + CE — Claude Review
experiments/exp198/claude_review.md:5:本实验为纯配置组合实验：将 exp197 的 STM 功能叠加到 exp191 的 OA-SD + CE 配置上。
experiments/exp198/claude_review.md:6:无新增/修改代码。审查重点是两个已审查通过的功能（STM、OA-SD）在组合使用时的交互安全性。
experiments/exp198/claude_review.md:10:- processor/processor.py — 训练循环中 STM 和 OA-SD 的执行流
experiments/exp198/claude_review.md:16:在 OA-SD 模式下（`len(img)==2`），训练循环的执行顺序如下：
experiments/exp198/claude_review.md:23:6. **OA-SD**（行 680-728）：EMA teacher forward `img_teacher`，计算蒸馏 loss，加入 `loss`
experiments/exp198/claude_review.md:29:## b. STM 与 OA-SD 交互安全性
experiments/exp198/claude_review.md:34:- OA-SD 操作的也是同一份 `feat`（行 704），但只读取不修改（`F.normalize` 创建新 tensor）
experiments/exp198/claude_review.md:42:- 主 loss（行 533）→ STM loss 累加（行 592）→ OA-SD loss 累加（行 726）
experiments/exp198/claude_review.md:46:- OA-SD 写入 keys: `oa_sd`
experiments/exp198/claude_review.md:54:- OA-SD loss 的梯度：`oa_sd_loss` → `F.normalize(feat[i])` → `feat[i]` → 模型参数
experiments/exp198/claude_review.md:73:- OA-SD 模式下：student forward + teacher forward（no_grad）
experiments/exp198/claude_review.md:75:- 远程 16GB GPU：exp191 (OA-SD + CE) 已验证可运行，STM 额外开销 < 150MB
experiments/exp198/claude_review.md:83:- exp198 通过命令行启用 `POSE_STM True`，与 OA-SD 配置项（`POSE_OA_SD True`）无冲突
experiments/exp198/claude_review.md:89:无新问题。STM 和 OA-SD 是完全独立的 loss 组件，操作不同阶段，无数据/梯度/内存冲突。
experiments/exp198/claude_review.md:95:**审查通过。** STM（exp197 已审查通过）和 OA-SD（多次审查通过）在 processor 中的执行路径完全独立：STM 在主 loss 后立即执行（行 535-595），OA-SD 在 parallel_aug 之后执行（行 680-728），两者均只读取原始 `score`/`feat`，各自计算独立的 loss 分量并累加。无数据修改冲突、无 key 冲突、无梯度干扰。可以启动训练。
experiments/exp246/monitor.md:1:# exp246b Tiny + LGPA-D + GCN 双分支 + OA-SD (重跑)
experiments/exp246/monitor.md:13:同时远程启动 exp245 Small LGPA-D + OA-SD 满血版。
experiments/exp246/monitor.md:75:Agent 审查结论: 7.6% 不是代码 bug, 是 OA-SD 训练不稳定性 + LGPA features 稀释。
experiments/exp246/monitor.md:107:| OA-SD | 0.036~0.038 | 稳定 |
experiments/exp246/monitor.md:125:| OA-SD | 0.038 | 稳定 |
experiments/exp246/monitor.md:164:| OA-SD | 0.038 | 稳定 |
experiments/exp246/monitor.md:178:| OA-SD | 0.039 | 稳定 |
experiments/exp246/monitor.md:253:**exp246b (Tiny LGPA-D + GCN 双分支 + OA-SD): 65.5/77.2/86.9/90.1**
experiments/exp213/claude_review.md:68:5. **OA-SD** (self-distillation): 通过 feat list → GCN params
experiments/exp109/monitor.md:1:# exp109 Oracle Support Bank 监控
experiments/exp109/monitor.md:14:  - `SGCFR` 说明 cross-image support recovery 确有价值
experiments/exp109/monitor.md:16:  1. 新增 `scripts/eval_oracle_support_bank.py`
experiments/exp109/monitor.md:19:  4. 若 headroom 明显，再进入训练版 support-complete distillation 设计
experiments/exp109/monitor.md:22:- 结果文件: `log/occluded_duke/exp109_oracle_support_bank_exp030a/summary.json`
experiments/exp109/monitor.md:26:  - `avg_support_count = 82.33`
experiments/exp109/monitor.md:39:  2. 说明“support-complete latent representation”不是空想，而是存在巨大 headroom
experiments/exp265b_best_s_op_s41/monitor.md:7:- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC) + 2-stage PSG `[-2,-1]` (default)
experiments/exp188/claude_review.md:1:# Claude Broad Review: exp188 OA-SD with EMA Teacher (Opus 4.6)
experiments/exp188/claude_review.md:6:- [High] OA-SD 无 PLBOA 无 warning → **已修复**（line 405-406 添加 warning）
experiments/exp188/claude_review.md:9:- [Low] OA-SD + PCVT/parallel_aug 静默禁用 → 已记录
experiments/exp188/claude_review.md:25:OA_SD=False 时：ema_teacher=None，所有 OA-SD 代码短路，行为不变。
experiments/exp236/monitor.md:3:配置: Tiny + GCN+PAA+OA-SD+PLBOA(0.7)+**无ROA** + FSDC(w=0.5)
experiments/exp236/monitor.md:235:**FSDC 方向证伪 — detached feature completion 无法改善 final ReID。**
experiments/exp246/claude_review.md:30:## OA-SD Teacher
experiments/exp246/claude_review.md:36:- LGPA head (~5.5M params) + GCN head (~400K params) + OA-SD teacher 
experiments/exp246/claude_review.md:43:- 但此处作为 supporting evidence (语义+结构是否互补), 可接受
experiments/exp199/monitor.md:6:- exp196 (3-view+SupCon+OA-SD global-only): 62.4/75.2 — OA-SD 失败对照
experiments/exp199/monitor.md:10:OA-SD (feature distillation) 失败了，OA-RD 是否能成功？
experiments/exp199/monitor.md:50:| exp196 (OA-SD global+SupCon+3v) | 37.9/51.2 |
experiments/exp199/monitor.md:52:**OA-RD 在 ep10 大幅领先！**但需谨慎——OA-SD 也曾在早期领先后落后。
experiments/exp199/monitor.md:83:| exp196 (OA-SD gl+SupCon+3v) | 37.9/51.2 | 50.8/63.0 | +1.0/+1.2 |
experiments/exp199/monitor.md:86:关键：exp196 (OA-SD) 在 ep20 只领先 +1.0/+1.2，后续完全消失变负。
experiments/exp199/monitor.md:105:| Epoch | exp199 (OA-RD) | exp187 (no distill) | exp196 (OA-SD gl) | delta vs 187 |
experiments/exp199/monitor.md:111:**观察**: ep30 落后 exp187 -0.8/-2.3。与 OA-SD 的 early drag 一致。
experiments/exp199/monitor.md:112:但关键区别：exp196 (OA-SD) 在 ep30 落后 -1.6/-1.2 且之后持续恶化。
experiments/exp199/monitor.md:134:| Epoch | exp199 (OA-RD) | exp187 (no distill) | exp196 (OA-SD gl) | delta vs 187 |
experiments/exp199/monitor.md:142:对比 exp196 (OA-SD) ep40: -0.7/-1.2 → OA-RD (+0.6/-1.3) vs OA-SD (-0.7/-1.2)
experiments/exp199/monitor.md:143:**OA-RD mAP 行为完全不同于 OA-SD：OA-RD 在 ep40 反超，OA-SD 在 ep40 仍落后！**
experiments/exp199/monitor.md:168:与 exp196 (OA-SD) ep50 (-2.0/-1.7) 对比：OA-RD 的 mAP 明显更好 (±0 vs -2.0)，
experiments/exp199/monitor.md:181:远程 exp200 ep60: 59.5/69.5 (vs exp191 OA-SD: 60.6/72.9, **-1.1/-3.4!!**)
experiments/exp199/monitor.md:182:OA-RD 在 CE 路线 ep60 开始落后 OA-SD！早期优势完全消失且反转。
experiments/exp199/monitor.md:197:OA-RD 在 SupCon+3-view 下与 OA-SD global-only (exp196) 的失败 pattern 类似：
experiments/exp199/monitor.md:201:**初步结论**: OA-RD (relational distillation) 虽然比 OA-SD (feature distillation)
experiments/exp199/monitor.md:308:| exp196 | 3-view + SupCon + OA-SD gl | 62.4% | 75.2% | -2.5/-1.4 |
experiments/exp199/monitor.md:313:2. **比 OA-SD global-only (exp196) 的 mAP 好** (-1.5 vs -2.5)，但 **R1 更差** (-2.1 vs -1.4)
experiments/exp265b_best_s_op_s41/claude_review.md:23:- pretrained swin_{tiny,small,base}, clip_part_text_features ✓
experiments/exp265b_best_s_op_s41/claude_review.md:30:- 预估显存 10-12GB (Small + Full + OA-SD), flip eval 峰值 13GB, < 16G 安全
experiments/exp197/design.md:4:- 当前最强: exp187 (3v+SupCon) = 64.9/76.6, exp193 (3v+OA-SD+CE) = 64.4/76.5
experiments/exp197/design.md:5:- OA-SD 和 SupCon 互斥（exp196 负结果）
experiments/exp197/design.md:48:- exp193 (3v+OA-SD+CE): 64.4/76.5 — 在此基础上加 STM
experiments/exp136/design.md:22:3. 应更集中地把梯度打到真正被 support-complete teacher 改变的关系上
experiments/exp272_psg2_t_od_s42/design.md:8:- 其他 pose 模块仍关(LGPA/GCN/OA-SD/PLBOA/ParAug)
experiments/exp199/claude_review.md:15:- OA-SD (feature distillation) 作用于 individual features (global 或 token)，约束它们向 teacher 的方向收敛 — 这与 SupCon 在同一空间上施加相反方向的力
experiments/exp199/claude_review.md:18:**关键区别**：OA-SD 的梯度形如 ∂L/∂f_i ∝ (f_i - f_teacher_i)，直接推动 f_i 向特定方向移动。OA-RD 的梯度形如 ∂L/∂f_i ∝ Σ_j w_j * (f_j - <f_i,f_j>*f_i)，它通过 pairwise cosine 传播，更关心保持相对结构。而且 OA-RD 仅在 global feature 上操作，SupCon 在 token features 上操作，特征空间完全不重叠。
experiments/exp199/claude_review.md:25:- 与 OA-SD 失败 (exp188/196) 形成对照 → 如果 OA-RD 成功，证据链完整
experiments/exp199/claude_review.md:72:- 当 OA-SD 禁用但 OA-RD 启用时，需要单独运行 teacher forward。**逻辑正确。**
experiments/exp199/claude_review.md:77:- `teacher_feat` 变量：如果 OA-SD 已经运行 (L684-701)，`teacher_feat` 已经赋值。如果 OA-SD 未运行，OA-RD 自己赋值 (L750-755)。**正确，无 NameError 风险。**
experiments/exp199/claude_review.md:79:### b.4 双模式兼容：OA-SD + OA-RD 同时启用
experiments/exp199/claude_review.md:81:- OA-SD (L684) 先运行 teacher forward → 设置 `teacher_feat`
experiments/exp199/claude_review.md:92:- 条件检查 `ema_teacher is not None`，对 OA-SD 和 OA-RD 共用。**正确。**
experiments/exp199/claude_review.md:93:- EMA decay 使用 `POSE_OA_SD_EMA_DECAY`（即使只启用 OA-RD）。这是一个设计选择而非 bug — OA-RD 没有自己的 decay 参数，复用 OA-SD 的。**可接受。** 如果未来需要区分，可以添加 `POSE_OA_RD_EMA_DECAY`，但目前不影响正确性。
experiments/exp199/claude_review.md:103:### c.1 与已有 OA-SD 实验的兼容性
experiments/exp199/claude_review.md:106:- OA-SD 代码路径 (L684-732) 完全独立于 OA-RD (L734-776)，通过 `if oa_sd_enabled` 和 `if oa_rd_enabled` 分开。**正确。**
experiments/exp199/claude_review.md:121:- 3-view 平均 (L681) → OA-SD loss (L730) → OA-RD loss (L774)
experiments/exp199/claude_review.md:138:- 如果同时启用 OA-SD，会分别记录 `oa_sd` 和 `oa_rd`。**可区分。**
experiments/exp199/claude_review.md:156:- `oa_sd_mode`, `parallel_oa_sd`, `_oa_sd_mode` 这些变量名已经同时服务于 OA-SD 和 OA-RD。建议后续重命名为更通用的名称（如 `teacher_mode`, `parallel_teacher`），但不影响正确性。
experiments/exp199/claude_review.md:168:所有代码路径正确。Teacher forward 在 OA-RD-only 和 OA-SD+OA-RD 两种模式下都正确处理。KL divergence 计算正确。Config 默认值安全。日志可观察。梯度流分析表明与 SupCon 不冲突。
experiments/afd_reid/verify_agreid_v2.py:18:    cd /root/work/SOLIDER-REID/experiments/afd_reid
experiments/afd_reid/verify_agreid_v2.py:19:    python verify_agreid_v2.py --data_root /root/work/SOLIDER-REID/data
experiments/afd_reid/verify_agreid_v2.py:137:    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/exp247/design.md:5:**重新定义问题**: "Occluded ReID fails because fixed part vocabularies assume complete semantic support. Under occlusion, the model should instantiate only the semantic groups actually supported by visible evidence."
experiments/exp197/claude_review_v2.md:85:### e7. 与 OA-SD 的交互
experiments/exp197/claude_review_v2.md:87:OA-SD 在 parallel_aug 之后，操作 teacher forward 的输出，与 STM 完全独立。无冲突。
experiments/exp197/claude_review_v2.md:128:**审查通过。** v1 中发现的全部 3 个需修复问题（1 Critical + 2 Medium）均已正确修复。固定数量生成策略保证了 triplet loss 的等正样本数假设，梯度流正确，与 SupCon/OA-SD/parallel_aug 等现有功能无冲突。可以启动训练。
experiments/exp191/design.md:1:# 实验 exp191: OA-SD + CE (不用 SupCon)
experiments/exp191/design.md:4:- exp188 测试 OA-SD + SupCon
experiments/exp191/design.md:5:- 需要消融：OA-SD 在 CE (不用 SupCon) 下是否也有效？
experiments/exp191/design.md:6:- 如果有效 → OA-SD 是独立有效的 paradigm
experiments/exp191/design.md:7:- 如果无效 → OA-SD 依赖 SupCon
experiments/exp191/design.md:16:- exp166 (CE, 无 OA-SD): 63.1/73.9
experiments/exp191/design.md:17:- exp188 (SupCon + OA-SD): 运行中
experiments/exp136/monitor.md:149:### [2026-03-21 20:25] `exp136` 到 `ep90`：稀疏机制稳定，但当前更像 supporting 线
experiments/exp136/monitor.md:185:- 当前判断: `exp136` 结案，保留为 supporting 线
experiments/exp267_best_t_m_s42/monitor.md:1:# exp267 monitor — Swin-Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) @ Market-1501
experiments/exp267_best_t_m_s42/monitor.md:66:- Swin-Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF,Market 非 occluded) + default flip-test @ Market-1501 = **92.5 / 96.4**
experiments/exp272_psg2_t_od_s42/monitor.md:7:- Scaffold: Swin-Tiny + PSG 2-stage (LGPA/GCN/OA-SD/PLBOA/ParAug 全关)
experiments/afd_reid/design.md:14:CARGO(+AG-ReID.v2)上, pretrained Swin/CLIP 抽特征, 对 原图 / low-pass(FFT保低频) / high-pass(FFT保高频)三套各算 Aerial↔Ground 的 mAP/R1。
experiments/afd_reid/design.md:20:CARGO 29M 子集(Mac 抽 3000 aerial+3000 ground, A∩G=1045 pid)。pretrained resnet50 V1(ImageNet)抽特征, A↔G mAP%:
experiments/afd_reid/design.md:50:- backbone: SOLIDER-Swin(团队资产)。
experiments/afd_reid/design.md:53:baseline(裸 pretrained A↔G)→ +Router → +Counterfactual → +both; 频带切分消融; 有无 view-adversarial。kill-switch 的三路特征(原图/low/high)对比即第一张图。
experiments/exp247/monitor.md:5:远程 (Tiny, 无 OA-SD): 首轮快速验证
experiments/exp247/monitor.md:6:本地: 等 exp246 完成后启动 VCSR + OA-SD 版本
experiments/exp247/monitor.md:8:对照: exp244 (LGPA-D + OA-SD): 65.3/75.7
experiments/exp247/monitor.md:9:对照: exp244-R (LGPA-D 无 OA-SD): 63.6/74.7
experiments/exp247/monitor.md:39:**远程 ep10 (VCSR, 无OA-SD): 37.3/51.2**
experiments/exp247/monitor.md:40:vs exp244-R (LGPA-D 无OA-SD) ep10: 37.1/50.7 = +0.2/+0.5。
experiments/exp247/monitor.md:68:**exp247 VCSR (Tiny, 无OA-SD): 63.6/73.5/84.2/88.3**
experiments/exp247/monitor.md:72:| exp244-R (LGPA-D 无OA-SD) | 63.6 | 74.7 | 85.3 | 88.6 |
experiments/exp247/monitor.md:73:| **exp247 VCSR (无OA-SD)** | **63.6** | **73.5** | **84.2** | **88.3** |
experiments/exp247/monitor.md:77:1. VCSR ≈ LGPA-D 无 OA-SD (mAP 完全持平, R1 -1.2)
experiments/exp197/monitor.md:121:2. 与 exp198 (OA-SD+CE 路线) 一致：**STM 不改善最终天花板**
experiments/exp330_compositional_dro/codex_review.md:11:- **High — make_model 签名/返回值不符**。**FALSE ALARM**：Codex 第 1 轮（本地 read-only）读的是**本地 SOLIDER-REID** `model/make_model.py`（Swin + semantic_weight + 3 返回值）。脚本实际 `--repo /hy-tmp/transreid` + `sys.path.insert(0, repo)` 导入的是**vanilla TransReID** make_model（已验证：(cfg,num_class,camera_num,view_num) → train 返回 (cls_score, global_feat)、eval 返回单 tensor；burstiness_probe 已在该树成功用过）。第 2 轮已明确告知，Codex 不再 re-flag。
experiments/exp191/monitor.md:1:# exp191 OA-SD + CE 监控
experiments/exp191/monitor.md:20:**观察**: OA-SD 在 CE 环境下正常工作。oa_sd loss 0.56 比 exp188 (SupCon) 的 0.48 略高。
experiments/exp191/monitor.md:26:| 指标 | exp191 (OA-SD+CE) | exp166r (CE only) | delta |
experiments/exp191/monitor.md:32:**观察**: ep10 偏低，但 exp188 (OA-SD+SupCon) 也是 ep10 偏低后来 ep30 反超。
experiments/exp191/monitor.md:38:| Epoch | OA-SD+CE | CE only (exp166r) | delta |
experiments/exp191/monitor.md:43:**观察**: OA-SD 在 CE 下 ep20 仍落后 (-1.5 R1)。可能 OA-SD 需要 SupCon 才有效。
experiments/exp191/monitor.md:49:| Epoch | OA-SD+CE | CE only | delta |
experiments/exp191/monitor.md:55:**观察**: OA-SD 在 CE 下持续落后 R1（-1.0~-1.7）。mAP 已追平。
experiments/exp191/monitor.md:56:OA-SD 可能只在 SupCon 环境下有效（exp188 ep30=+4.2 R1），CE 下无效。
experiments/exp191/monitor.md:62:| Epoch | OA-SD+CE | CE only | delta |
experiments/exp191/monitor.md:68:**观察**: ep40 OA-SD 反超 CE！+2.0/+1.6。早期落后但后期发力。
experiments/exp191/monitor.md:74:| Epoch | OA-SD+CE | CE only | delta |
experiments/exp191/monitor.md:80:**观察**: OA-SD+CE 持续正向（+1.5~1.6 R1）！与 exp188 (OA-SD+SupCon = 负) 形成对比。
experiments/exp191/monitor.md:81:OA-SD 与 CE 有正 synergy，与 SupCon 有负 synergy（梯度冲突）。
experiments/exp191/monitor.md:87:| Epoch | OA-SD+CE | CE only | delta |
experiments/exp191/monitor.md:93:**观察**: OA-SD+CE ep60 增益巨大！+3.8/+4.4 vs CE alone。比 SupCon 的增益还大。
experiments/exp191/monitor.md:94:OA-SD 与 CE 有强正 synergy！
experiments/exp191/monitor.md:100:| Epoch | OA-SD+CE | CE only | delta |
experiments/exp191/monitor.md:106:**观察**: OA-SD+CE 持续巨大增益。ep70 已接近 exp166 最终 (63.1/73.9)。
experiments/exp191/monitor.md:107:如果 ep120 超过 63.1 → OA-SD 在 CE 下独立有效，是范式级模块。
experiments/exp191/monitor.md:113:| Epoch | OA-SD+CE | CE only | exp166 final | delta vs CE |
experiments/exp191/monitor.md:118:**观察**: R1 73.8% ≈ exp166 最终 73.9%！OA-SD+CE 在 ep80 几乎达到完整系统的最终值。
experiments/exp191/monitor.md:121:**状态**: 🔥🔥 OA-SD+CE 超越 exp166!
experiments/exp191/monitor.md:124:| Epoch | OA-SD+CE (base) | CE only (base) | exp166 final (full) | delta vs exp166 |
experiments/exp191/monitor.md:129:**R1 75.1% 超过 exp166 最终 73.9%！** OA-SD 在 base 架构 + CE 上就达到超越！
experiments/exp191/monitor.md:134:| Epoch | OA-SD+CE | exp166 final | delta |
experiments/exp191/monitor.md:144:| Epoch | OA-SD+CE | exp166 final | delta |
experiments/exp191/monitor.md:161:| 实验 | 架构 | 训练 | OA-SD | mAP | R1 |
experiments/exp191/monitor.md:167:**OA-SD 在 base 架构 + CE 上超过了 full 架构 + CE + 全部增强！**
experiments/exp191/monitor.md:171:**OA-SD 是独立有效的范式级创新！** 只需 EMA teacher + PLBOA asymmetry，无需额外架构。
experiments/exp267_best_t_m_s42/claude_review.md:12:H1. Swin-Base + WITH_CP + POSE_PARALLEL_AUG (4 views) + OA-SD on 5060Ti 16G：Base 首 epoch OOM 风险。exp260b 历史 3090 运行 OK，但 5060Ti 16G 需要监控。若 OOM：关 OA-SD 或 PARALLEL_AUG，不改 BS。
experiments/exp267_best_t_m_s42/claude_review.md:22:M3. Market 的 OA-SD 因为没 PLBOA，teacher/student view 1 几乎相同；蒸馏信号弱但不出错。可以接受作为一致性。
experiments/exp267_best_t_m_s42/claude_review.md:28:L1-L8. OA-SD 参数跨数据集统一（可接受）；PLBOA 在 Occ-PTrack 开（数据已 occluded 但一致性保持）；POSE_PFM_ENABLED 默认 False 即使 POSE_PFM_HIDDEN=64 存在（无 bug，命名小瑕疵）；DEVICE_ID=('0')、BIAS_LR_FACTOR=2、WARMUP_EPOCHS=20 都与 exp255 scaffold 一致。
experiments/exp209/design.md:1:# 实验 exp209: Small + STD-PR + CE + OA-SD (远程 1-view)
experiments/exp209/design.md:4:- OA-SD 在 GCN+CE 路线 = +2.9/+2.6 (Tiny), +0 (Small vs 4090 PAA)
experiments/exp209/design.md:5:- OA-SD 在 STD-PR+SupCon 路线 = 负向 (互斥)
experiments/exp209/design.md:6:- **但 OA-SD + STD-PR + CE 从未测试过！**
experiments/exp209/design.md:7:- 如果 OA-SD 在 STD-PR+CE 上也有效: STD-PR+CE base ~65.8 + OA-SD +2-3 = **68-69%**
experiments/exp209/design.md:8:- 这与 GCN+PAA+OA-SD (70.5) 比较，确认哪个架构+OA-SD 更好
experiments/exp209/design.md:11:OA-SD 在 CE 路线下与架构无关——在 STD-PR 上也应该有效。
experiments/exp209/design.md:16:- 启用 OA-SD: POSE_OA_SD True
experiments/exp209/design.md:21:- exp206 (GCN+PAA+OA-SD 1v Small): 70.5/82.3
experiments/afd_reid/cargo_dataset.py:6:    /root/work/SOLIDER-REID/data/CARGO/{train,query,gallery}/Cam{1..13}/*.jpg
experiments/afd_reid/cargo_dataset.py:71:    def __init__(self, root='/root/work/SOLIDER-REID/data', verbose=True):
experiments/afd_reid/cargo_dataset.py:268:    ds = CARGO(root='/root/work/SOLIDER-REID/data', verbose=True)
experiments/exp191/claude_review.md:1:# Claude Broad Review: exp191 OA-SD + CE (Opus 4.6)
experiments/exp191/claude_review.md:6:OA-SD with CE (no SupCon)。消融 OA-SD 的独立有效性。
experiments/exp191/claude_review.md:13:- Loss: CE path (SUPCON=False default)。OA-SD distillation 独立于 CE/SupCon。
experiments/exp191/claude_review.md:30:如果 OA-SD+CE > CE alone → OA-SD 独立于 SupCon 有效
experiments/exp191/claude_review.md:31:如果 OA-SD+CE ≈ CE alone → OA-SD 需要 SupCon 才有效
experiments/exp197/claude_review.md:16:**单变量原则**: 满足。在 exp187 (3v+SupCon) 或 exp193 (3v+OA-SD+CE) 基础上只加 STM。
experiments/exp197/claude_review.md:94:- **OA-SD**: STM 在 main loss 计算后、OA-SD 在 parallel_aug 处理后。两者独立。但 STM 对 `score` 和 `feat` 操作，OA-SD 对 teacher forward 输出操作。无冲突。
experiments/exp247/claude_review.md:22:- OA-SD: teacher deepcopy + list distillation ✅
experiments/exp330_compositional_dro/design.md:32:- **GO**：group-DRO 在 3 个 held-out 组合 cell 上比 ERM **≥ +1.5 mAP**（主判据；副判据 standard Occ-Duke 不掉 >0.5 脚本外单独验）。→ 真组合泛化机制，升级（pose-anchored 放置 + 更细 cells + SOLIDER 强栈）。
experiments/exp131/design.md:15:如果当前瓶颈主要是 batch 内 changed-pair coverage 不足，那么在保持 `exp125` 的 online support teacher 与 `delta_top` routing 不变的前提下，引入 cross-batch relation queue 应当带来更强的 late-stage 收益。
experiments/exp131/design.md:33:   - support-complete teacher kp feats
experiments/exp209/monitor.md:1:# exp209 Small + STD-PR + CE + OA-SD (远程 1-view) 监控
experiments/exp209/monitor.md:3:配置: Swin-Small + STD-PR + CE + OA-SD (no SupCon)
experiments/exp209/monitor.md:4:对照: exp202 (STD-PR+SupCon 1v Small): 67.9/79.5, exp206 (GCN+PAA+OA-SD): 70.5/82.3
experiments/exp209/monitor.md:22:STD-PR+CE+OA-SD 启动正常。oa_sd=0.785 (比 GCN 路线的 0.48 高)。
experiments/exp209/monitor.md:63:**ep10: 32.7/45.2** — 远低于 GCN+PAA+OA-SD (exp206 ep10: 47.9/60.3)!
experiments/exp209/monitor.md:79:| Epoch | exp209 STD-PR+CE+OA-SD | exp206 GCN+PAA+CE+OA-SD |
experiments/exp209/monitor.md:114:| Epoch | exp209 STD-PR+CE+OA-SD | exp206 GCN+PAA+CE+OA-SD |
experiments/exp209/monitor.md:120:**结论**: STD-PR+CE+OA-SD 在所有 epoch 持续落后 GCN+PAA+CE+OA-SD ~5%。
experiments/exp209/monitor.md:122:**STD-PR 需要 SupCon 才能发挥，但 SupCon 与 OA-SD 互斥。**
experiments/exp209/monitor.md:124:**实验终止。远程 GPU 转给 exp210 (GCN+PAA+CE+OA-SD+PKC)。**
experiments/afd_reid/agreid_v2_dataset.py:8:    ds = AGReIDV2(root='/root/work/SOLIDER-REID/data', protocol='exp1')
experiments/afd_reid/agreid_v2_dataset.py:17:Data layout (lab-3090, /root/work/SOLIDER-REID/data/AG-ReID.v2/):
experiments/afd_reid/agreid_v2_dataset.py:130:    def __init__(self, root='/root/work/SOLIDER-REID/data',
experiments/afd_reid/agreid_v2_dataset.py:255:    ds = AGReIDV2(root='/root/work/SOLIDER-REID/data', protocol='exp1', verbose=True)
experiments/exp196/design.md:1:# 实验 exp196: 3-view + SupCon + OA-SD Global-Only（终极配置）
experiments/exp196/design.md:5:- exp193 (3-view + OA-SD + CE): 64.4/76.5 — 3-view+OA-SD+CE 最佳
experiments/exp196/design.md:6:- exp195 (SupCon + OA-SD global-only): 验证了 global-only 解决梯度冲突
experiments/exp196/design.md:7:- **问题**: 能否把三大创新（3-view + SupCon + OA-SD）合为一体？
experiments/exp196/design.md:8:- exp188 证明 all-token OA-SD + SupCon 失败（梯度冲突）
experiments/exp196/design.md:9:- exp195 证明 global-only OA-SD + SupCon 成功（职责分离）
experiments/exp196/design.md:10:- **本实验验证**: 在 3-view 环境下，SupCon + OA-SD global-only 是否 additive
experiments/exp196/design.md:13:通过 OA-SD global-only distillation 实现"职责分离"——global feature 学遮挡不变性（OA-SD），per-token features 学判别力（SupCon）——在 3-view parallel augmentation 下三者增益 additive。
experiments/exp196/design.md:16:- 配置 = exp187 (3-view + SupCon) + OA-SD global-only
experiments/exp196/design.md:21:- OA-SD distillation 仅在 `feat[0]` (global) 上计算
experiments/exp196/design.md:28:   - 1 × OA-SD cosine distillation on global feat → added after averaging
experiments/exp196/design.md:29:4. Backward: OA-SD 梯度只流过 global feature，不影响 per-token SupCon
experiments/exp196/design.md:32:- exp193 (3-view + OA-SD all-token): 20.9GB — OK
experiments/exp196/design.md:38:- 如果中性: ~64.9/76.6 (= exp187)，说明 OA-SD global 在 SupCon 下冗余
experiments/exp196/design.md:39:- 如果失败: < 64.5，说明 3-view 下 OA-SD + SupCon 仍有某种干扰
experiments/exp196/design.md:42:- exp187 (3-view + SupCon, no OA-SD): 64.9/76.6 — 主对照
experiments/exp196/design.md:43:- exp193 (3-view + OA-SD + CE): 64.4/76.5
experiments/exp196/design.md:44:- exp195 (SupCon + OA-SD global-only, 1-view): 进行中
experiments/exp209/claude_review.md:1:# exp209 Review: Small + STD-PR + CE + OA-SD (远程 1-view)
experiments/exp209/claude_review.md:7:- OA-SD 与 STD-PR per-token list feat 的兼容性 (zip 对齐)
experiments/exp209/claude_review.md:8:- dataset 2-view 生成 (OA-SD mode, 非 parallel_aug)
experiments/exp209/claude_review.md:10:- PLBOA 启用验证 (OA-SD 前提)
experiments/exp209/claude_review.md:16:本实验是 recipe 交叉验证：OA-SD 已在 GCN+CE (exp191) 上验证有效 (+2.9/+2.6)，STD-PR+CE 已在 exp166 中验证。这里组合两者看 OA-SD 是否与架构无关。作为 ablation 数据点可接受，非主线创新。
experiments/exp209/claude_review.md:40:## 3. OA-SD 与 STD-PR per-token feat 兼容性 — 完全兼容
experiments/exp209/claude_review.md:60:`POSE_OA_SD_GLOBAL_ONLY` 默认 False — 这里无 SupCon 冲突，distill 所有 tokens 是正确选择（CE 梯度与 OA-SD 梯度不冲突）。
experiments/exp209/claude_review.md:68:`pose_dataset.py:171-172`: OA-SD mode 下，PLBOA 前保存 `img_clean_for_oa_sd = img.copy()`。
experiments/exp209/claude_review.md:74:不使用 `POSE_PARALLEL_AUG` (3-view)，只有 2-view OA-SD。设计文档说 "1-view" 应理解为无 parallel_aug 的 3-view，实际是 2-view (student + teacher)。预期行为。
experiments/exp209/claude_review.md:82:这是 OA-SD 的核心前提。如果关闭，teacher 和 student 看几乎相同的图，distillation 退化。
experiments/exp209/claude_review.md:93:exp206 (同为 Small + OA-SD) 的 review 建议 `WITH_CP: True`。保持一致。
experiments/exp209/claude_review.md:103:- `MODEL.POSE_OA_SD True` — 启用 OA-SD ✓
experiments/exp209/claude_review.md:113:- `PRETRAIN_PATH: pretrained/swin_small.pth` ✓
experiments/exp209/claude_review.md:128:**审查通过**。SupCon 关闭后正确回退到 per-token CE，OA-SD 与 STD-PR per-token 输出完全兼容 (7 元素 list zip 对齐)，dataset 2-view 生成正确，PLBOA 已启用。唯一建议是确保 `WITH_CP True` 在命令行中以避免 16GB OOM。无需新代码。
experiments/exp328_vcnorm/codex_review.md:60:**原问题**：VCA 嵌在 OA-SD 分支内（`processor.py:783` 的 `if oa_sd_enabled ...`），
experiments/exp328_vcnorm/codex_review.md:64:`POSE_VCNORM=True` 必须 `POSE_OA_SD=True`，否则报错并指明原因（VCA 消费 OA-SD EMA teacher 的
experiments/afd_reid/afd_killswitch.py:2:# 无训练: pretrained resnet50 抽特征, 对 原图/low-pass/high-pass 各算 A<->G mAP。
experiments/afd_reid/afd_killswitch.py:47:imgs = glob.glob('/home/afr/SOLIDER-REID/data/**/Cam*/*.jpg', recursive=True)
experiments/exp224/design.md:47:- exp191 OA-SD (single stage): 63.2/75.4
experiments/exp131/monitor.md:116:     - 真正卡住的更像是 **pair-specific support-complete correction 不能被当前单向量学生充分吸收**
experiments/exp196/monitor.md:1:# exp196 3-view + SupCon + OA-SD Global-Only 监控
experiments/exp196/monitor.md:3:配置: full arch (PAPE+MS-PSG) + 3-view parallel aug + SupCon T=0.05 + OA-SD global-only
experiments/exp196/monitor.md:5:- exp187 (3-view+SupCon, no OA-SD): 64.9/76.6 — 主对照
experiments/exp196/monitor.md:6:- exp193 (3-view+OA-SD+CE): 64.4/76.5
experiments/exp196/monitor.md:7:- exp195 (SupCon+OA-SD global-only, 1-view, base config): 61.3/74.9
experiments/exp196/monitor.md:25:SupCon 和 OA-SD 同时出现在 log 中——两者共存！
experiments/exp196/monitor.md:60:| **exp196 (3v+SupCon+OA-SD global)** | **37.9/51.2** |
experiments/exp196/monitor.md:62:| exp193 (3v+OA-SD+CE) | 39.3/52.4 |
experiments/exp196/monitor.md:65:与 OA-SD 的 early-stage drag 一致（exp193 也在 ep30 曾拖累）。
experiments/exp196/monitor.md:89:| **exp196 (3v+SupCon+OA-SD global)** | **37.9/51.2** | **50.8/63.0** |
experiments/exp196/monitor.md:91:| exp193 (3v+OA-SD+CE) | 39.3/52.4 | 50.7/62.1 |
experiments/exp196/monitor.md:95:- OA-SD global-only 没有拖累 SupCon，反而提供了额外增益
experiments/exp196/monitor.md:96:- 也超过了 exp193 (CE+OA-SD all-token) +0.1/+0.9
experiments/exp196/monitor.md:110:### [17:44] 检查点 #17 — ep30 OA-SD drag
experiments/exp196/monitor.md:114:| Epoch | exp196 (终极) | exp187 (3v+SupCon) | exp193 (3v+OA-SD+CE) | delta vs 187 |
experiments/exp196/monitor.md:120:**观察**: ep30 落后 exp187 -1.6/-1.2。与 OA-SD 的 early drag 一致：
experiments/exp196/monitor.md:123:- OA-SD crossover 通常在 ep40
experiments/exp196/monitor.md:149:OA-SD drag 在消退。crossover 可能在 ep50-60。
experiments/exp196/monitor.md:176:这不像正常的 OA-SD late-stage boost — 反而在恶化。
experiments/exp196/monitor.md:179:2. OA-SD EMA teacher 在 3-view+SupCon 下的某种干扰
experiments/exp196/monitor.md:206:Gap 缩小到 -1.4/-0.7。OA-SD late boost 开始显现。
experiments/exp196/monitor.md:232:OA-SD global-only 在 SupCon+3-view 配置下不仅没有正向贡献，
experiments/exp196/monitor.md:236:**初步结论**: OA-SD global-only + SupCon + 3-view ≠ additive。
experiments/exp196/monitor.md:237:最可能原因：global feature 的 OA-SD 信号太弱（oa_sd~0.01），不如 all-token。
experiments/exp196/monitor.md:238:而 all-token OA-SD 又与 SupCon 冲突。所以 OA-SD 和 SupCon 在同一配置下无法共存。
experiments/exp196/monitor.md:318:| exp193 | 3-view + OA-SD + CE | 64.4% | 76.5% | -0.5/-0.1 |
experiments/exp196/monitor.md:319:| **exp196** | **3-view + SupCon + OA-SD global-only** | **62.4%** | **75.2%** | **-2.5/-1.4** |
experiments/exp196/monitor.md:323:1. **OA-SD global-only + SupCon + 3-view = 负向** (-2.5/-1.4 vs exp187)
experiments/exp196/monitor.md:324:2. **OA-SD 和 SupCon 本质互斥**：即使用 global-only 避免梯度冲突，OA-SD 仍然拖累 SupCon
experiments/exp196/monitor.md:325:3. **可能原因**：OA-SD 的 EMA teacher forward 占用 GPU 资源，但 global-only distillation (oa_sd=0.008) 几乎无信号
experiments/exp196/monitor.md:326:4. **论文定位**：负结果 ablation——证明 OA-SD 和 SupCon 是两条互补但互斥的路线
experiments/exp284_gcn512_1stg_s_od_s42/monitor.md:5:- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp284_gcn512_1stg_s_od_s42/train_log.txt`
experiments/exp284_gcn512_1stg_s_od_s42/monitor.md:7:- Scaffold: Swin-Small + Full Scaffold (LGPA + **GCN512** + OA-SD + ParAug + LOWER_BODY_OCC) + 1-stage PSG
experiments/exp277_psg3_s_od_s42/monitor.md:5:- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp277_psg3_s_od_s42/train_log.txt`
experiments/exp277_psg3_s_od_s42/monitor.md:7:- Scaffold: Swin-Small + PSG 3-stage (LGPA/GCN/OA-SD/PLBOA/ParAug 全关)
experiments/exp328_vcnorm/design.md:4:> **性质**：训练端实验（**本任务只到 dry-run，不正式训练**；正式训练前须过双审查）。**机器**：lab-3090-d，repo `/root/work/SOLIDER-REID`。
experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/exp328_vcnorm/design.md:9:exp260b 已有 OA-SD（occluded student ↔ clean EMA teacher 双前向）+ PLBOA（下肢遮挡增广），但其蒸馏只作用在 **pooled global 特征**（cosine）。BT-PKD 进一步蒸馏 per-keypoint 特征**方向**（cosine），仍是「让被遮挡 token 的特征向 clean token 看齐」。两者都没有触碰探针指认的真正病灶：**被遮挡 per-part token 的归一化统计（一阶/二阶矩）整体漂移到一个可分离的子空间**——这是一条「有/无遮挡」的 domain 轴，叠加在身份信号上。OA-SD/BT-PKD 把个体特征拉近，但没有显式消掉这条 domain 轴；探针显示这条轴在跨域时大量残留（pre-GCN AUC 0.97），GCN 只部分修复。
experiments/exp328_vcnorm/design.md:28:   - **机制（核心新意）**：不是 per-instance 拉特征（那是 OA-SD/BT-PKD），而是**对齐 per-keypoint 的"可见性条件归一化统计"**。对每个 keypoint k，在 batch 内取 student 路该 keypoint 的 token 群，估计其 per-channel mean/var（一阶/二阶矩），与 teacher 路同 keypoint 的 mean/var 做对齐（Gaussian/统计距离，**与探针同一把尺子**：对角高斯一阶/二阶矩匹配）。
experiments/exp328_vcnorm/design.md:52:5. **processor 接线** `processor/processor.py`：在 OA-SD 段（已算好 `teacher_kp_data`）后追加 VCA loss：
experiments/exp328_vcnorm/design.md:61:   - **复用现成 dual-forward**：exp260b 已是 `parallel_oa_sd`（3 occluded student + 1 clean teacher），teacher 前向已在 OA-SD 段算出 `teacher_kp_data`，**VCA 不新增前向、不增显存峰值的额外 backbone pass**。
experiments/exp328_vcnorm/design.md:94:- **PLBOA 说明（重要）**：exp260b 原 config `POSE_LOWER_BODY_OCC=False`，此时 OA-SD 的 teacher≈student（代码已 warning「near-identical images」），OA-SD 近退化。VC-Norm 要 occluded-vs-clean 的对比信号**必须开 PLBOA**，故本实验与对照都设 `POSE_LOWER_BODY_OCC=True`。因此严格单变量对照是「同 PLBOA、VC-Norm 开/关」，**不是** vs 原 exp260b 88.0（后者 PLBOA 关）。88.0 仅作量级参照；若开 PLBOA 的 VC-Norm-OFF 对照已偏离 88.0，以对照为准。
experiments/exp328_vcnorm/design.md:106:- **机制层面已防**：VCA 对齐的是 **batch 级 per-keypoint per-channel 矩（mean/var）**，不是 instance 级特征。身份来自单个 token 在该统计坐标下的**相对位置**，矩对齐只搬动整群分布的中心/尺度，不直接监督「谁是谁」。这与 OA-SD/BT-PKD（instance 级拉近）正交。
experiments/exp224/monitor.md:1:# exp224 Tiny + GCN+PAA+CE+OA-SD+ROA + 2-Stage Keypoint Fusion 监控
experiments/exp224/monitor.md:3:配置: 基于 pose_psg_gcn_paa_roa.yml (含 ROA) + OA-SD + PLBOA + POSE_MULTI_SCALE_KP=True
experiments/exp224/monitor.md:9:对照: exp191 OA-SD+ROA (single scale): 63.2/75.4
experiments/exp224/monitor.md:31:**ep10: 36.6/49.0** (vs OA-SD 34.3/46.8 = +2.3/+2.2)
experiments/exp224/monitor.md:33:| Method | ep10 mAP/R1 | vs OA-SD |
experiments/exp224/monitor.md:35:| OA-SD | 34.3/46.8 | — |
experiments/exp224/monitor.md:57:**ep20: 47.0/59.4** (vs OA-SD 46.0/58.0 = +1.0/+1.4)
experiments/exp224/monitor.md:59:| Epoch | KAMP mAP/R1 | OA-SD mAP/R1 | delta |
experiments/exp224/monitor.md:80:**ep30: 51.4/63.8** (vs OA-SD 50.6/61.7 = +0.8/+2.1)
experiments/exp224/monitor.md:82:| Epoch | KAMP mAP/R1 | OA-SD mAP/R1 | delta |
experiments/exp224/monitor.md:105:**ep40: 55.8/68.2** (vs OA-SD 57.2/69.2 = -1.4/-1.0)
experiments/exp224/monitor.md:107:| Epoch | KAMP mAP/R1 | OA-SD mAP/R1 | delta |
experiments/exp224/monitor.md:121:**exp223b (PADPQ K=8) FINAL: 63.9/74.3** (vs OA-SD 63.2/75.4 = +0.7/-1.1)
experiments/exp224/monitor.md:129:vs OA-SD ep50: 59.0 = **-3.6!** 2-stage fusion 在 ep50 大幅落后。
experiments/exp224/monitor.md:131:| Epoch | KAMP mAP | OA-SD mAP | delta |
experiments/exp224/monitor.md:149:**ep60: 57.3/70.2** (vs OA-SD 60.6/72.9 = **-3.3/-2.7**)
experiments/exp224/monitor.md:151:| Epoch | KAMP mAP | OA-SD mAP | delta |
experiments/exp224/monitor.md:159:预计 final ~60-61%，远低于 OA-SD 63.2。
experiments/exp224/monitor.md:171:**ep70: 58.2/71.1** (vs OA-SD 61.8/73.1 = -3.6/-2.0)
experiments/exp224/monitor.md:188:**ep80: 59.1/71.0** (vs OA-SD 62.0/73.8 = -2.9/-2.8)
experiments/exp224/monitor.md:190:预计 final ~61-62%，低于 OA-SD 63.2。
experiments/exp224/monitor.md:202:**ep90: 60.2/73.2** (vs OA-SD 62.4/75.1 = -2.2/-1.9)
experiments/exp224/monitor.md:219:**ep100: 60.4/72.5** (vs OA-SD 63.0/75.0 = -2.6/-2.5)
experiments/exp224/monitor.md:245:**exp224 (2-stage keypoint fusion + OA-SD) FINAL: 60.7/73.0**
experiments/exp224/monitor.md:250:| exp191 OA-SD-only | 63.2% | 75.4% | — | — |
experiments/exp196/claude_review.md:1:# exp196 审查报告: 3-view + SupCon + OA-SD Global-Only
experiments/exp196/claude_review.md:46:- `img_teacher` 在此赋值，后续 OA-SD 块 (L624) 可访问
experiments/exp196/claude_review.md:73:**Step 4**: OA-SD distillation (L617-665):
experiments/exp196/claude_review.md:77:- OA-SD loss 加在 averaged loss 之后 (L663): `loss = loss + oa_sd_weight * oa_sd_loss`
experiments/exp196/claude_review.md:82:- **OA-SD global-only**: 操作 `feat[0]` (global pooled feature)，梯度流向 backbone + global pooling
experiments/exp196/claude_review.md:86:SupCon 和 OA-SD 在 feature 层面完全分离: SupCon on tokens, OA-SD on global。二者梯度不冲突。
experiments/exp196/claude_review.md:87:Global CE 和 OA-SD 都作用于 global feature，但这是 by design -- CE 提供判别力，OA-SD 提供遮挡不变性。
experiments/exp196/claude_review.md:106:- 对照组完整: exp187 (SupCon only)、exp193 (OA-SD+CE)、exp195 (SupCon+OA-SD)
experiments/exp196/claude_review.md:109:**关于"小调参"质疑**: 本实验是纯配置组合，无新代码。但它组合的三个组件各自已单独验证有效，且 exp188 证明了 naive 全 token OA-SD + SupCon 失败，exp195 证明了 global-only 版本成功。因此本实验是对"职责分离"假说在完整 3-view 环境下的最终验证，具有论文消融表价值。可以接受。
experiments/exp196/claude_review.md:113:唯一风险: 3-view 下 OA-SD global-only 的增益可能被 3-view 本身的 regularization 效果吸收（3-view 已提供多样遮挡模式）。这属于科学问题而非代码问题。
experiments/afd_reid/afd_train.py:15:    cd /root/work/SOLIDER-REID/experiments/afd_reid
experiments/afd_reid/afd_train.py:17:        --data_root /root/work/SOLIDER-REID/data \
experiments/afd_reid/afd_train.py:18:        --out_dir   /root/work/SOLIDER-REID/log/cargo/afd_baseline \
experiments/afd_reid/afd_train.py:249:    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/exp284_gcn512_1stg_s_od_s42/claude_review.md:19:- 该结论在 Phase 1 新协议 (flip-fix, 新 OA-SD EMA) 下是否复现
experiments/exp284_gcn512_1stg_s_od_s42/claude_review.md:47:与 exp262 完全同除 `POSE_PSG_STAGES` 一字段 (从 `[-2,-1]` → `[-1]`)。Swin-Small Occ-Duke SGD lr 8e-4 120 epoch seed 42 flip-test equal_concat GLOBAL_LOSS_SCALE=0.5 GCN_HIDDEN=512 LGPA/OA-SD/PLBOA/ParAug 全开。Δ 可直接归因 PSG stage 2 注入。
experiments/exp284_gcn512_1stg_s_od_s42/claude_review.md:67:- 若需要精确复现 exp255 vs exp255b 的历史差距 (约 Δ = +1.3/+1.5 FINAL),本 exp 的 Δ 应接近该值;若显著缩小,反映 Phase 1 新协议 (OA-SD + PLBOA + LGPA + ParAug) 改变了 GCN cap vs PSG stage 的互补关系
experiments/exp231/design.md:35:- exp191: OA-SD only (63.2/75.4) — 无 BT-PKD baseline
experiments/overnight_pose_clip_search.md:47:un-detach 破坏了全系统(强系统为 detached LGPA 调好, un-detach 让 LGPA 塑造 backbone 干扰 PSG/GCN/OA-SD 平衡)。
experiments/overnight_pose_clip_search.md:65:exp349 = exp255 全 pose 系统(Swin-Small + 2-stage PSG + LGPA + GCN512 + OA-SD + PLBOA)+ CLIP prompt, e120 训练eval mAP = **71.4%** vs exp255 **73.2%** → **CLIP 拉低强系统 -1.8**(test.py poll 待精确分解)。
experiments/exp328_vcnorm/monitor.md:4:- 主：`configs/market/pose_vcnorm_base.yml`（POSE_VCNORM=True, WARMUP=20, WEIGHT=0.5, VIS_THR=0.3, OA_SD=True, PLBOA=True, TEST_FEAT=equal_concat, SIZE_TEST=[384,128]）@ lab-3090-d `/root/work/SOLIDER-REID`, OUTPUT `./log/market1501/exp328_vcnorm`, log `/tmp/exp328_vcnorm.log`。
experiments/exp341base_noprompt/design.md:4:exp341（CLIP-ReID 可学习 ID prompt）的**精确单变量对照**：完全相同的 SOLIDER 配置（GLOBAL_LOSS_SCALE 1.0、无 LGPA/PSG），仅 `POSE_CLIP_ID_PROMPT: False`（关掉 prompt 分支）。
experiments/exp231/monitor.md:3:配置: Tiny + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01, decay_epoch=60)
experiments/exp231/monitor.md:4:对照: exp191 (Tiny OA-SD): 63.2/75.4
experiments/exp356r_pcmsc_random/codex_review.md:22:先例(Codex web search): PersonMAE(2311.04496)、MVP(2203.05175)、RILS(2301.06958)、RFCnet、MaskCLIP(2208.12262)。PC-MSC 差异化 = pose 部位选择 mask + 冻结 CLIP dense 区域语义目标 + 遮挡 ReID + 训练端正则的组合, 是 plausible distinct mechanism(非孤立"masked CLIP completion")。
experiments/exp190/design.md:1:# 实验 exp190: 3-view Parallel Aug + CE (不用 SupCon)
experiments/exp328_vcnorm/claude_review.md:25:- **新于 OA-SD/BT-PKD 确认**：OA-SD 蒸 pooled global cosine（instance 级），BT-PKD 蒸 per-keypoint 方向 cosine（instance 级）。VCA 对齐的是 **batch 级 per-keypoint per-channel mean/var**，不监督"谁是谁"——机制上确实正交，不是换皮。代码里也确实如此（`_weighted_moments` 在 dim=0/batch 上聚合）。
experiments/exp328_vcnorm/claude_review.md:49:- 复用现成前向、不新增 backbone pass：teacher 前向是 OA-SD 既有的，VCA 只读它的 kp_data，无额外前向。✅
experiments/exp328_vcnorm/claude_review.md:86:代码质量高：插件式、zero-init 恒等、teacher detach、train/test 单点对称、AMP dtype 处理、优化器自动纳管、baseline 复现性（关开关=no-op）全部正确，dryrun 的 load-bearing 正确性主张本地实跑通过。机制确实新于 OA-SD/BT-PKD（batch 级统计矩对齐 vs instance 级特征蒸馏），有探针支撑，是 problem-level 创新而非小调参。
experiments/exp305_full_noLGPA_t_od_s42/monitor.md:6:- Scaffold: Swin-Tiny + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG (**LGPA OFF**)
experiments/afd_reid/agreid_v2_combined.py:76:    def __init__(self, root='/root/work/SOLIDER-REID/data', verbose=True):
experiments/afd_reid/agreid_v2_combined.py:173:    root = sys.argv[1] if len(sys.argv) > 1 else '/root/work/SOLIDER-REID/data'
experiments/exp341base_noprompt/claude_review.md:18:- open_clip 接受本地路径/safetensors 作为 `pretrained`（已查证）。
experiments/exp341base_noprompt/claude_review.md:71:12. **pretrained 路径**：config `POSE_CLIP_ID_PRETRAINED='/home/afr/SOLIDER-REID/clip_l14_openclip.safetensors'`（lab-4090 已确认该文件存在，1.7GB），open_clip `create_model_and_transforms(pretrained=<本地 .safetensors>)` 支持 ✓。
experiments/exp341base_noprompt/claude_review.md:78:- **单变量**：对照组就是同 config 关 `POSE_CLIP_ID_PROMPT`（= 纯 global ID+triplet）。本实验所有 PSG/LGPA/GCN/OA-SD/PLBOA/parallel_aug 都显式关闭，隔离干净。✓
experiments/exp341base_noprompt/claude_review.md:82:- **R1 — LR schedule 与 CoOp 不匹配**：CoOp prompt 通常用较高的**常数** LR（原论文 SGD 0.002 cosine，但 prompt 单独高 LR）。本 config 让 `cls_ctx` 跟 SOLIDER backbone 同一套（BASE_LR=0.0008，20 epoch warmup→cosine 衰减），warmup 早期 LR 极小，prompt 可能学得慢/学不动。**这是 tuning 风险不是 bug**：若 clip_id_loss 不降或 global 不涨，第一件事就是给 `cls_ctx`/`clip_id_proj` 单独抬 LR（可仿 make_optimizer.py:20 的 part_lr_factor 加一个 clip-id 分支，或 LARGE_FC_LR 思路）。建议训练时把 `details['clip_id']` 打进 log 盯它是否单调下降——日志已接（processor.py:1301），够观察。
experiments/exp341base_noprompt/claude_review.md:95:代码正确性逐项核对通过：手写 CLIP 文本编码器与 open_clip 2.32.0 `encode_text` 在标准 ViT-L-14 配置下完全一致（batch_first 无 permute、EOT argmax、causal mask、text_projection Parameter 分支均已对源码确认）；优化器确实优化 cls_ctx + clip_id_proj（#1 风险排除）；clip_id_loss 不重复计、test 端不受影响、维度全动态无 hardcode、pretrained 本地 safetensors 路径有效。无 Critical/High 阻断项。
experiments/exp249/design.md:1:# exp249: Small LGPA-D + GCN 双分支 + OA-SD
experiments/exp249/design.md:16:- OA-SD + PLBOA(0.7) + PSG
experiments/exp249/design.md:24:- MODEL.PRETRAIN_PATH pretrained/swin_small.pth
experiments/exp249/design.md:32:- exp206r (Small GCN+PAA+OA-SD): 70.6/82.6
experiments/exp356r_pcmsc_random/design.md:8:2. SOLIDER backbone → feature map F + global。
experiments/exp356r_pcmsc_random/design.md:25:- CLIP 预处理: SOLIDER 输入(384×128, 0.5 norm)→ un-norm → CLIP norm → resize 224(畸变, 但 pose 热图同步 resize 一致)。
experiments/exp356r_pcmsc_random/design.md:32:- ablation: 重建 CLIP 特征 vs 重建 SOLIDER 自身特征(隔离"CLIP 语义目标"价值, 后者=PersonMAE 类已知)。
experiments/exp190/monitor.md:1:# exp190 3-view Parallel Aug + CE 监控
experiments/exp190/monitor.md:45:Remote exp191 ep90: 62.4/75.1 (OA-SD+CE 超过 exp166!)
experiments/exp190/monitor.md:58:Remote exp191 ep110: 63.1/75.3 (OA-SD+CE ±0/+1.4 vs exp166 final!)
experiments/exp190/monitor.md:87:**观察**: ep40=60.3/73.2，比 exp191 (OA-SD+CE) ep40=57.2/69.2 高很多 (+3.1/+4.0)。
experiments/exp190/monitor.md:119:ep50 已达 61.1/73.5，exp191 (OA-SD+CE) ep50=59.0/70.6，3-view 加速效果明显强于 OA-SD。
experiments/exp190/monitor.md:160:| Epoch | 3-view+CE (exp190) | 3-view+SupCon (exp187) | OA-SD+CE (exp191) | delta vs exp187 |
experiments/exp190/monitor.md:171:3-view 效果 >> OA-SD 效果 (ep60: +1.7/+1.0)。
experiments/exp190/monitor.md:209:| Epoch | 3-view+CE (exp190) | 3-view+SupCon (exp187) | OA-SD+CE (exp191) | delta vs 187 |
experiments/exp190/monitor.md:240:| Epoch | 3-view+CE (exp190) | 3-view+SupCon (exp187) | OA-SD+CE (exp191) | exp166 final |
experiments/exp190/monitor.md:268:| Epoch | 3-view+CE (exp190) | exp191 final (OA-SD+CE) | exp187 final (3-view+SupCon) |
experiments/exp190/monitor.md:294:| Epoch | 3-view+CE (exp190) | exp176 final (SupCon) | exp187 final (3v+SupCon) | exp191 final (OA-SD+CE) |
experiments/exp190/monitor.md:349:| exp191 | OA-SD + CE | 63.2% | 75.4% | -1.0/-0.2 |
experiments/exp190/monitor.md:356:4. **3-view > OA-SD**: exp190 (+1.1/+1.7) vs exp191 (+2.9/+2.6 from CE base, but lower abs)
experiments/exp190/monitor.md:357:5. **exp193 (3-view + OA-SD + CE) 是合理的下一步**
experiments/exp237/design.md:9:| Detached feature completion | FSDC (exp235/236) | ~-1.5% |
experiments/exp237/design.md:14:**唯一有效的方向是改变 backbone 本身的行为**: PSG (+1.7%), OA-SD (+2-3%)。
experiments/exp237/design.md:91:- exp191 (OA-SD, detached GCN): 63.2/75.4
experiments/exp269b_best_b_m_s42_full120/monitor.md:7:- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA **OFF** + 2-stage PSG `[-2,-1]`)
experiments/afd_reid/band_analysis.py:25:    cd /root/work/SOLIDER-REID/experiments/afd_reid
experiments/afd_reid/band_analysis.py:27:        --data_root /root/work/SOLIDER-REID/data \
experiments/afd_reid/band_analysis.py:28:        --ckpt /root/work/SOLIDER-REID/log/cargo/afd_baseline/model_best.pth
experiments/afd_reid/band_analysis.py:131:    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/exp214/design.md:1:# 实验 exp214: Small GCN+PAA + 3-view (无 OA-SD)
experiments/exp214/design.md:5:- 但 Small 的 3-view+CP+OA-SD 出过学习停滞 bug (exp206 3-view)
experiments/exp214/design.md:6:- 假设: 不用 OA-SD 的纯 3-view 可能在 Small 上也有效
experiments/exp214/design.md:10:3-view CE/triplet 训练在 Small GCN+PAA 上提供 +1-2% mAP，不需要 OA-SD。
experiments/exp214/design.md:14:- 不用 OA-SD 节省显存，3-view 不需要 CP
experiments/exp214/design.md:17:- exp206r (1-view + OA-SD): 70.6/82.6 (eq), 72.3/82.9 (maxsim)
experiments/exp214/design.md:18:- exp210b (1-view + OA-SD + PKC): 70.6/81.8 (eq), 72.4/83.1 (maxsim)
experiments/exp138/claude_review_request.txt:4:- `/root/work/SOLIDER-REID/experiments/exp138/design.md`
experiments/exp138/claude_review_request.txt:5:- `/root/work/SOLIDER-REID/experiments/exp138/monitor.md`
experiments/exp138/claude_review_request.txt:6:- `/root/work/SOLIDER-REID/configs/occluded_duke/pose_psg_gcn_lpcs_rank_decay.yml`
experiments/exp138/claude_review_request.txt:7:- `/root/work/SOLIDER-REID/config/defaults.py`
experiments/exp138/claude_review_request.txt:8:- `/root/work/SOLIDER-REID/processor/processor.py`
experiments/exp138/claude_review_request.txt:9:- `/root/work/SOLIDER-REID/model/pose_backbone_model.py`
experiments/exp138/claude_review_request.txt:10:- `/root/work/SOLIDER-REID/model/modules/pair_adaptive_fusion.py`
experiments/exp138/claude_review_request.txt:11:- `/root/work/SOLIDER-REID/experiments/exp135/design.md`
experiments/exp138/claude_review_request.txt:12:- `/root/work/SOLIDER-REID/experiments/exp137/design.md`
experiments/exp138/claude_review_request.txt:13:- `/root/work/SOLIDER-REID/experiments/decisions.md`
experiments/exp237/monitor.md:1:# exp237 Tiny + PPA (Pose-Prompted Part-Assignment) + OA-SD 监控
experiments/exp237/monitor.md:3:配置: Tiny + PSG + PPA (替换 GCN) + OA-SD + PLBOA(0.7) + 无 ROA
experiments/exp237/monitor.md:5:对照: exp191 (Tiny OA-SD, detached GCN): 63.2/75.4
experiments/exp237/monitor.md:32:- FSDC: -1.5/-2.2 (detached feature completion ineffective)
experiments/exp356r_pcmsc_random/claude_review.md:17:- **SOLIDER→CLIP 重归一化**: line 221-223 `img.float()*0.5+0.5`（un-norm mean=std=0.5 → [0,1]）→ `(x-mean)/std`（CLIP mean/std buffer，line 209-210）→ `interpolate` 224。正确。
experiments/exp356r_pcmsc_random/claude_review.md:58:`_run_backbone_with_psg(x,...)` 首行 `x, hw_shape = self.base.patch_embed(x)` 仅**重绑函数内局部 x**，不改写 caller 的 `x`。扫描 forward (699-783) 无对 `x` 的重绑/in-place。→ line 782 传给 `_pcmsc_loss` 的 `x` 是原始未改输入图（SOLIDER 0.5/0.5 归一），正是 `part_targets` 所需。
experiments/exp356r_pcmsc_random/claude_review.md:68:- **L1 (启动机/可移植性)**: `configs/occluded_duke/exp356_pcmsc.yml:23` `POSE_CLIP_ID_PRETRAINED='/home/afr/SOLIDER-REID/clip_l14_openclip.safetensors'` 是 **lab-4090 专属路径**（1.7GB safetensors）。`CLIPVisualEncoder.__init__` 传给 `open_clip.create_model_and_transforms(pretrained=...)`（open_clip 2.32.0 支持本地 safetensors，已查证）。**启动机选择（子代理实测，2026-06-21）**：
experiments/exp215/monitor.md:1:# exp215 Small + GCN+PAA+CE+OA-SD + BA-PKC (weight=0.1) 监控
experiments/exp249/monitor.md:1:# exp249 Small + LGPA-D + GCN 双分支 + OA-SD 监控
experiments/exp249/monitor.md:3:配置: Swin-Small + PSG + LGPA-D (CLIP, detached) + GCN (detached) + OA-SD + PLBOA(0.7) + WITH_CP
experiments/exp249/monitor.md:6:对照: exp206r (Small GCN+PAA+OA-SD): 70.6/82.6
experiments/exp249/monitor.md:16:**决策**: 等 ep5 看 oa_sd 趋势 (之前 3090 上 Small OA-SD 会暴降, PT2+mmcv-full 已解决)
experiments/exp249/monitor.md:41:| OA-SD | 0.457 | ↑ 正常 (0.26→0.46) |
experiments/exp249/monitor.md:62:| OA-SD | 0.387 | ↓ (teacher 变好) |
experiments/exp249/monitor.md:82:| OA-SD | 0.298 | ↓ (teacher 进步) |
experiments/exp249/monitor.md:100:| OA-SD | 0.138 | ↓ (teacher 快速进步) |
experiments/exp249/monitor.md:104:**观察**: 训练进入快速学习阶段。OA-SD 从 0.46 快速降到 0.14。ep10 eval 预计 ~01:27。
experiments/exp249/monitor.md:118:| OA-SD | 0.061 | ↓↓ (teacher 已非常强) |
experiments/exp249/monitor.md:123:**观察**: ID loss 开始加速下降 (6.53→6.35 in 4 epochs)。OA-SD 已从 0.46 降到 0.06，teacher 很强。ep10 eval 预计 ~01:28。
experiments/exp249/monitor.md:136:| OA-SD | 0.057 | ↓↓ teacher 极强 |
experiments/exp249/monitor.md:151:| OA-SD | 0.081 | ~0.05 |
experiments/exp249/monitor.md:166:| OA-SD | 0.041 | ↓ (teacher 接近 final) |
experiments/exp249/monitor.md:182:| OA-SD | 0.038 |
experiments/exp249/monitor.md:192:**ep9 final**: Loss 10.35, Acc 0.261, ID_G 6.206, OA-SD 0.037, LGPA 6.961
experiments/exp249/monitor.md:200:**当前**: Loss 10.34, ID_G 6.194, Acc 0.175, OA-SD 0.030, LGPA 6.949
experiments/exp249/monitor.md:236:**当前**: Loss 10.14, ID_G 6.050, Acc 0.204, OA-SD 0.024, LGPA 6.862
experiments/exp249/monitor.md:251:| OA-SD | 0.028 | 0.019 | ↓ 接近 0 |
experiments/exp249/monitor.md:255:**观察**: ID loss 快速下降 (6.10→5.67 in 3ep)。OA-SD 接近 0 表示 teacher 和 student 已非常接近。LGPA assign 从 7.23→6.71 持续改善。ep20 eval ~02:54。
experiments/exp249/monitor.md:268:| OA-SD | 0.019 | 0.016 | → 接近 0 |
experiments/exp249/monitor.md:278:**当前**: Loss 8.86, ID_G 4.720, Acc 0.274, OA-SD 0.016, LGPA 6.397
experiments/exp249/monitor.md:307:| OA-SD | 0.015 | 0.015 | → 稳定极低 |
experiments/exp214/claude_review.md:1:# exp214 审查: Small GCN+PAA + 3-view (无 OA-SD)
experiments/exp214/claude_review.md:20:- 动机清晰：3-view 在 Tiny 有效 (+1.4%)，尝试在 Small 上去掉 OA-SD 的纯 3-view
experiments/exp214/claude_review.md:22:- 对照组正确：exp206r (1-view + OA-SD) 作为参照
experiments/exp214/claude_review.md:23:- 单变量原则：相比 exp206r，仅 OA-SD → off，parallel_aug → on
experiments/exp214/claude_review.md:62:**OA-SD block (line 693):**
experiments/exp214/claude_review.md:89:注意：严格来说改了两个变量（OA-SD off + parallel_aug on），不是纯单变量实验。但设计意图是用 3-view 替换 OA-SD 作为正则化手段，所以两个变量同时改是合理的。
experiments/exp214/claude_review.md:96:此前 exp206 3-view+OA-SD 在 WITH_CP=True 下成功运行（4 views），但出现学习停滞。exp214 去掉了 OA-SD（3 views）且不用 CP，内存应该是够的（3 with-grad < 3 with-grad + 1 no-grad + CP overhead），但建议监控前几个 iter 的 GPU 内存。
experiments/exp214/claude_review.md:99:没有 OA-SD 的 teacher-student 不对称，PLBOA 只作为数据增强。3-view 中所有 view 共享相同的 PLBOA 变换（在 branching 前应用），这是正确的。
experiments/exp214/claude_review.md:103:代码逻辑正确。3-view 无 OA-SD 路径经过完整追踪：
experiments/exp214/claude_review.md:108:5. OA-SD/OA-RD/EMA 全部正确跳过
experiments/exp353_undetach_noclip/design.md:4:- Step1（exp341）成功：CLIP-ReID 可学习 ID prompt 在 SOLIDER 上 **+2.2**（59.8 vs 57.6 matched）。找到了能涨的 CLIP 机制。
experiments/prcv_2026_psg/decisions.md:9:2. `exp109` 所代表的问题证据仍然强，说明旧故事并不牢
experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/afd_reid/afd_model.py:42:# Swin-Small backbone (SOLIDER) -- optional, team asset, for SOTA push.
experiments/afd_reid/afd_model.py:46:# Repo root = .../SOLIDER-REID (this file lives at experiments/afd_reid/).
experiments/afd_reid/afd_model.py:52:    """The SOLIDER swin_transformer.py does, at import time,
experiments/afd_reid/afd_model.py:78:    """Thin wrapper around the SOLIDER swin_small backbone for the AFD/OVLI
experiments/afd_reid/afd_model.py:95:    def __init__(self, img_size=(256, 128), pretrain_path='', semantic_weight=0.2,
experiments/afd_reid/afd_model.py:111:            semantic_weight=semantic_weight,   # SOLIDER ReID default 0.2
experiments/afd_reid/afd_model.py:115:        if pretrain_path:
experiments/afd_reid/afd_model.py:116:            # loads the SOLIDER 'teacher' checkpoint (backbone.* keys), strict=False
experiments/afd_reid/afd_model.py:117:            self.swin.init_weights(pretrain_path)
experiments/afd_reid/afd_model.py:167:            # deep-copy preserves the pretrained weights as the f_rec init (same
experiments/afd_reid/afd_model.py:530:                 pretrained=True,
experiments/afd_reid/afd_model.py:537:                 swin_pretrain='', swin_semantic_weight=0.2,
experiments/afd_reid/afd_model.py:585:            weights = 'IMAGENET1K_V1' if pretrained else None
experiments/afd_reid/afd_model.py:616:            # SOLIDER Swin-Small (team asset, SOTA push).  AFD frequency modules
experiments/afd_reid/afd_model.py:618:            # exist in Swin -> AFD is unsupported here (OVLI is the headline and
experiments/afd_reid/afd_model.py:620:            assert not use_afd, ("backbone='swin_small' does not support the AFD "
experiments/afd_reid/afd_model.py:625:                img_size=tuple(img_size), pretrain_path=swin_pretrain,
experiments/afd_reid/afd_model.py:636:            # (== SOLIDER's native avgpool head over the same map).
experiments/afd_reid/afd_model.py:874:    builds the SOLIDER Swin-Small backbone instead (img_size / swin_pretrain /
experiments/afd_reid/afd_model.py:875:    swin_semantic_weight are read from args, with the SOLIDER ReID defaults).
experiments/afd_reid/afd_model.py:881:        pretrained=True,
experiments/afd_reid/afd_model.py:891:        swin_pretrain=getattr(args, 'swin_pretrain', ''),
experiments/exp237/claude_review.md:19:**动机清晰**: 236 个实验的系统总结证明 detached feature 上的所有操作都无法改善最终结果。只有改变 backbone 本身行为的方法（PSG, OA-SD）有效。PPA 尝试用 learnable part-assignment head 取代 detached GCN sampling，让 part loss 梯度端到端流过 backbone。
experiments/exp237/claude_review.md:127:### C4. OA-SD 兼容性
experiments/exp237/claude_review.md:129:**Teacher 也会运行 PPA**: OA-SD teacher 在 `train()` mode (line 714) + `no_grad()` 下执行 forward。由于 `self.training=True`，PPA 的训练路径会被触发。Teacher 产生相同结构的输出: `[global_cls, pooled_cls, part1_cls, ..., part5_cls]`。
experiments/exp237/claude_review.md:183:- 结合 Global ID + Global triplet + OA-SD distillation，backbone 同时接收 6-7 种 loss 的梯度。
experiments/exp237/claude_review.md:213:- OA-SD: teacher 结构匹配，distillation 安全
experiments/exp138/monitor.md:137:- 当前判断: 继续，但当前更像 supporting 线
experiments/exp138/monitor.md:158:  - 这已经足够说明 `rank-decay` 是有效但偏弱的 supporting 机制，不值得继续占用本地主卡
experiments/exp138/monitor.md:168:- 当前判断: `exp138` 结案，定性为 supporting 线
experiments/exp249/claude_review.md:13:e. processor — loss 计算, OA-SD distillation 正确 ✅
experiments/exp249/claude_review.md:28:4. **OA-SD**: 正确处理 list features (global + LGPA parts + GCN parts) ✅
experiments/exp207/design.md:1:# 实验 exp207: Swin-Base + GCN+PAA+CE+OA-SD (Backbone Scaling)
experiments/exp207/design.md:5:- 当前最佳: Small GCN+PAA+OA-SD = 70.5/82.3
experiments/exp207/design.md:9:- 再加 OA-SD 等创新 → 可能 **75%+**
experiments/exp207/design.md:13:我们的 GCN+PAA+OA-SD 方法应在更大 backbone 上有类似或更大的增益。
experiments/exp207/design.md:17:- `pretrained/swin_base.pth` 权重
experiments/exp207/design.md:18:- 配置同 exp206: GCN+PAA+ROA+CE+OA-SD+PLBOA
experiments/exp207/design.md:32:- exp206 (Small + GCN+PAA+OA-SD): 70.5/82.3
experiments/prcv_2026_psg/story.md:25:   - `OA-SD`
experiments/prcv_2026_psg/story.md:73:   - `Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA`
experiments/prcv_2026_psg/story.md:144:   - 第二目标：至少超过 `SOLIDER / BPBreID`
experiments/afd_reid/design_airl_iso_agreidv2_4090.md:27:  --data_root /home/afr/SOLIDER-REID/data \
experiments/afd_reid/design_airl_iso_agreidv2_4090.md:29:  --out_dir /home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090 \
experiments/afd_reid/design_airl_iso_agreidv2_4090.md:31:  --swin_pretrain /home/afr/SOLIDER-REID/pretrained/swin_small.pth \
experiments/afd_reid/design_airl_iso_agreidv2_4090.md:51:- swin_small.pth 1.15GB 已在 pretrained/。
experiments/afd_reid/design_airl_iso_agreidv2_4090.md:56:1. **python 环境**：lab-4090 系统 `/usr/bin/python3` 无 torch/numpy。SOLIDER-REID 自己没 .venv/pyproject。解决：用 afr 现成 uv venv `/home/afr/reid-clean/.venv/bin/python`（torch 2.6.0+cu124，含 numpy/timm/yacs，cuda 可用）。**今后 lab-4090 跑这套必须用这个 python，不是 python3。**
experiments/afd_reid/design_airl_iso_agreidv2_4090.md:58:   - 注：OSS 客户端只收 .zip（拒 .tgz "Unsupported file type"），lab-3090 又无 zip 命令 → 改走 base64-over-ssh 本地中转。
experiments/exp208/design.md:1:# 实验 exp208: Small + GCN+PAA+CE+OA-SD + 0.5x Global Loss
experiments/exp208/design.md:4:- exp206 (Small GCN+PAA+CE+OA-SD, GLOBAL_LOSS_SCALE=1.0) = 70.5/82.3
experiments/exp287_lgpaOnly_2stg_t_od_s42/design.md:9:- 保持 LGPA True, OA-SD True, ParAug True, LOWER_BODY_OCC True
experiments/exp287_lgpaOnly_2stg_t_od_s42/design.md:14:- exp287 vs exp286: 同 semantic branch (LGPA + OA-SD) 下 2-stage vs 1-stage PSG
experiments/prcv_2026_psg/phase1_design.md:8:- exp255 (old protocol, 不含 default flip-test): Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA = 73.2/83.3
experiments/prcv_2026_psg/phase1_design.md:14:一句话：**Swin + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA** 在 3 个 backbone × 3 个训练集（OccDuke / OccPTrack / Market）上都能给出有竞争力的 SOTA 数字；默认测试协议含 flip-test。
experiments/exp207/monitor.md:1:# exp207 Swin-Base + GCN+PAA+CE+OA-SD 监控
experiments/exp207/monitor.md:3:配置: Swin-Base (88M) + GCN+PAA+ROA + CE + OA-SD + PLBOA + WITH_CP
experiments/exp207/monitor.md:32:本地 Base 3-view+CP+OA-SD: **9.5GB/24GB** — 完美！
experiments/exp207/monitor.md:377:远程 exp209 (STD-PR+CE+OA-SD) 已启动, ep2, ETA 6h49m。
experiments/exp207/monitor.md:444:### [10:10] 检查点 #64 — OA-SD teacher bug 发现 & 修复
experiments/exp207/monitor.md:457:**重要**: 之前所有 OA-SD 实验 (exp191, exp193, exp200, exp206, exp207) 的 teacher 都是有噪声的。
experiments/exp207/monitor.md:458:修复后 OA-SD 应该更强——teacher target 更稳定。
experiments/exp207/monitor.md:464:第二次修复 (pose_test_feat='global') 修了 crash 但让 OA-SD 变成了 global-only distillation。
experiments/exp207/monitor.md:472:远程 exp210 (GCN+PAA+CE+OA-SD+PKC Small) 已启动，pkc=3.725。
experiments/exp207/monitor.md:514:已终止 exp210，远程改跑 exp206r (同配置无 PKC) 作为 OA-SD fix 对照。
experiments/exp207/monitor.md:516:**决策**: 继续 exp207，等 exp206r ep10 确认 OA-SD fix 无问题
experiments/exp207/monitor.md:522:确认: OA-SD fix 无问题, PKC weight=0.5 是灾难原因。
experiments/exp207/monitor.md:534:exp206r: id_global=6.278, Acc=0.184 — 学习正常！OA-SD fix confirmed OK。
experiments/exp207/monitor.md:539:**两台 ep10 结果！OA-SD fix 带来显著提升！**
experiments/exp207/monitor.md:548:**OA-SD teacher fix 确认有效！** 修复后的 teacher (无 Dropout/DropPath 噪声 + 干净 pose) 产生更好的 distillation target。
experiments/exp207/monitor.md:595:**OA-SD fix 加速了早期收敛但 ep20 已追平。** 
experiments/exp207/monitor.md:642:**OA-SD fix 在 ep30 重新拉开差距！** ep20 的追平只是暂时的。
experiments/exp207/monitor.md:739:OA-SD fix 在 ep50 保持 +0.4 mAP, +0.7 R1 优势。不大但一致。
experiments/exp207/monitor.md:878:ep70 的落后在 ep80 追回并反超！OA-SD fix 在后期再次领先。
experiments/exp207/monitor.md:932:**结论**: OA-SD fix 正确但不改变 final 结果。主要 lever 仍是 maxsim_hybrid (+1.8%)。
experiments/exp207/monitor.md:977:**最终结论: OA-SD fix 不改变 final 结果。**
experiments/exp207/monitor.md:1001:**exp206r (fixed OA-SD) FINAL: 70.6/82.6**
experiments/exp207/monitor.md:1201:可能原因: LR=0.0002 对 Base 偏低，3-view+OA-SD 交互不够好。
experiments/exp207/monitor.md:1314:**exp207 Base 3-view + GCN+PAA+CE+OA-SD (fixed): 70.7/80.7**
experiments/afd_reid/smoke_agreid_v2_wiring.py:21:    python3 <repo>/experiments/afd_reid/smoke_agreid_v2_wiring.py /root/work/SOLIDER-REID/data
experiments/afd_reid/smoke_agreid_v2_wiring.py:29:ROOT = sys.argv[1] if len(sys.argv) > 1 else '/root/work/SOLIDER-REID/data'
experiments/exp107/design.md:41:  - `support_gap = min(d_q_gd, d_qd_g) - d_tt`
experiments/exp107/design.md:43:  - 若 `support_gap` 小，说明该 pair 的 target-target 优势不足，属于高歧义 pair
experiments/exp107/design.md:44:  - 用 `support_gap` 对 top-K 基线距离做 margin-based 调整
experiments/exp208/monitor.md:1:# exp208 Small + GCN+PAA+CE+OA-SD + 0.5x Global Loss 监控
experiments/exp287_lgpaOnly_2stg_t_od_s42/monitor.md:7:- Scaffold: Swin-Tiny + LGPA + OA-SD + ParAug + LOWER_BODY_OCC - GCN + PSG `[-2,-1]`
experiments/exp207/claude_review.md:1:# exp207 审查: Swin-Base + GCN+PAA+CE+OA-SD
experiments/exp207/claude_review.md:6:b. Backbone 可用性 (swin_base_patch4_window7_224 + pretrained weights)
experiments/exp207/claude_review.md:7:c. 模块兼容性 (PSG / PAA / GCN / OA-SD vs 1024-dim features)
experiments/exp207/claude_review.md:19:- `pretrained/swin_base.pth` 存在 (1.77GB)
experiments/exp207/claude_review.md:20:- `configs/occluded_duke/swin_base.yml` 已有正确的 `TRANSFORMER_TYPE: 'swin_base_patch4_window7_224'` 和 `PRETRAIN_PATH: 'pretrained/swin_base.pth'`
experiments/exp207/claude_review.md:33:- **OA-SD**: 使用 `F.normalize` + cosine distance -- 完全 dim-agnostic
experiments/exp207/claude_review.md:37:## 3. OA-SD 兼容性: PASS
experiments/exp207/claude_review.md:42:- exp206 已验证 OA-SD 在 GCN 模式 + Small backbone 下正常工作 (70.5/82.3)
experiments/exp207/claude_review.md:51:| PRETRAIN_PATH | pretrained/swin_tiny.pth | pretrained/swin_base.pth |
experiments/exp207/claude_review.md:59:**PLBOA 必须启用** (POSE_LOWER_BODY_OCC: True)。无 PLBOA 时 OA-SD 的 teacher/student 看到近乎相同的图像，exp206 monitor 中有此警告。
experiments/exp207/claude_review.md:68:- OA-SD 需要额外 EMA teacher (完整模型副本) -- **这是关键约束**
experiments/exp207/claude_review.md:75:- + OA-SD teacher forward: +4-6 GB (no_grad 但需激活)
experiments/exp207/claude_review.md:87:- exp206 (Small + OA-SD) 使用多少 LR? 需确认
experiments/exp207/claude_review.md:100:### Medium: Swin-Base pretrained weights 来源
experiments/exp207/claude_review.md:101:- `pretrained/swin_base.pth` (1.77GB) 已存在
experiments/exp207/claude_review.md:102:- 需确认是否为 SOLIDER pretrained (与 Tiny/Small 一致) 还是 ImageNet-22K
experiments/exp207/claude_review.md:103:- 如果是 ImageNet-22K 而非 SOLIDER, 基线可能不同, 对比不完全公平
experiments/exp207/claude_review.md:106:- Base + OA-SD + CP 估计 17-22GB, **5060 Ti 16GB 必定 OOM**
experiments/exp207/claude_review.md:122:| OA-SD 维度兼容 | PASS |
experiments/exp207/claude_review.md:133:1. 必须创建 exp207 专用 config 文件 (基于 pose_psg_gcn_paa_plboa_roa.yml + Base backbone + OA-SD + LR 0.0002 + WITH_CP=True)
experiments/exp207/claude_review.md:136:4. 确认 swin_base.pth 的预训练来源 (SOLIDER vs ImageNet-22K)
experiments/exp107/monitor.md:46:  1. 有符号 support-gap 重排明确负面，说明“奖励安全 pair + 惩罚危险 pair”的粗糙公式不成立。
experiments/exp107/monitor.md:54:- 若继续研究 target/distractor ambiguity，必须把推理粒度拉回 `per-keypoint / common-support`，而不是继续在 pooled person embedding 上做文章。
experiments/exp340b_fixedbands_undetach/claude_review.md:23:L5–30、32、34–79 **逐字相同**：detach 以外的 LGPA 超参（CLIP_DIM 512 / NUM_HEADS 8 / POSE_TEMP 1.0 / ASSIGN_WEIGHT 0.5 / FIXED_BANDS True / TEST_FEAT equal_concat）、384×128、SGD/LR 0.0008/120ep、PLBOA/OA-SD/parallel-aug 全关、Swin-Tiny。**无任何意外差异**，确认是「只翻两个相关旋钮」的受控变体。
experiments/exp340b_fixedbands_undetach/claude_review.md:74:- **分支隔离**：exp340b 配置下 GCN（POSE_PSG_STAGES 空、无 GCN flag）、OA-SD、PLBOA、parallel-aug 全关，前向走纯 LGPA 分支（`:635-659`），无 dual-branch 交互。**隔离干净**。
experiments/prcv_2026_psg/todo.md:15:- 最强配置：Swin-{T/S/B} + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA
experiments/prcv_2026_psg/todo.md:91:- [ ] Method section（PSG + GCN + LGPA-D + OA-SD + PLBOA + 测试协议）
experiments/prcv_2026_psg/todo.md:115:这组不含 LGPA/GCN/OA-SD/PLBOA（纯 PSG scaffold），回答 "PSG 本体是否稳定"。
experiments/prcv_2026_psg/todo.md:130:full scaffold（LGPA-D + OA-SD + PLBOA + GCN + PSG）。
experiments/exp287_lgpaOnly_2stg_t_od_s42/claude_review.md:25:- `MODEL.POSE_SKELETON_GCN False` 单变量 override,scaffold 其他 (LGPA/OA-SD/ParAug/LOWER_BODY_OCC/PSG_STAGES default `[-2,-1]`) 自动继承
experiments/exp269_best_b_m_s42/monitor.md:1:# exp269 monitor — Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) @ Market-1501
experiments/exp269_best_b_m_s42/monitor.md:69:- Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) + default flip-test @ Market-1501 (e80 eff FINAL) = **94.4 / 97.0 (Global+flip)**, **94.5 / 97.1 (MaxSim+flip)**
experiments/prcv_2026_psg/experiment_route.md:67:- feature-level residual completion 小变体
experiments/prcv_2026_psg/experiment_route.md:91:2. `SOLIDER`
experiments/exp286_lgpaOnly_1stg_t_od_s42/codex_review.md:15:srvC 上 Occ-Duke 4.9GB 数据 + pretrained 全齐,无数据缺失风险。
experiments/exp238/monitor.md:1:# exp238 Tiny + PPA (assign_weight=0.1) + OA-SD 监控
experiments/exp143/monitor.md:15:  1. `exp142 SKC` 已确认为中性偏负（-0.8% mAP），feature-level completion 方向再次失败
experiments/cargo_cvpb/airl_story.md:9:- 诊断证据(kill-switch #1):强 SOLIDER-Swin backbone 上,按 aerial bbox area 分桶,最小桶 A→G mAP 相对顶桶塌陷 +13~19。**强 backbone 也救不了 = 物理问题非 backbone-headroom artifact。**
experiments/cargo_cvpb/airl_story.md:23:- 强 backbone(SOLIDER-Swin)同设置下,AIRL dualbranch vs baseline-Swin(60.84)vs VDT(42.76)/GSAlign/SeCap。
experiments/exp269_best_b_m_s42/claude_review.md:12:H1. Swin-Base + WITH_CP + POSE_PARALLEL_AUG (4 views) + OA-SD on 5060Ti 16G：Base 首 epoch OOM 风险。exp260b 历史 3090 运行 OK，但 5060Ti 16G 需要监控。若 OOM：关 OA-SD 或 PARALLEL_AUG，不改 BS。
experiments/exp269_best_b_m_s42/claude_review.md:22:M3. Market 的 OA-SD 因为没 PLBOA，teacher/student view 1 几乎相同；蒸馏信号弱但不出错。可以接受作为一致性。
experiments/exp269_best_b_m_s42/claude_review.md:28:L1-L8. OA-SD 参数跨数据集统一（可接受）；PLBOA 在 Occ-PTrack 开（数据已 occluded 但一致性保持）；POSE_PFM_ENABLED 默认 False 即使 POSE_PFM_HIDDEN=64 存在（无 bug，命名小瑕疵）；DEVICE_ID=('0')、BIAS_LR_FACTOR=2、WARMUP_EPOCHS=20 都与 exp255 scaffold 一致。
experiments/exp275_psg1_s_od_s42/design.md:11:- 其他 pose 模块关(LGPA/GCN/OA-SD/PLBOA/ParAug)
experiments/exp286_lgpaOnly_1stg_t_od_s42/design.md:10:- 保持 LGPA True, OA-SD True, ParAug True, LOWER_BODY_OCC True
experiments/exp286_lgpaOnly_1stg_t_od_s42/design.md:18:- exp286 vs exp271 (Tiny pure PSG 1-stage 60.2/69.5): 加入 LGPA + OA-SD + ParAug 对 PSG 1-stage 的增益
experiments/prcv_2026_psg/result.md:126:### Table A — Pure PSG stage (no LGPA / no GCN / no OA-SD / no ParAug / no PLBOA)
experiments/exp068/review.md:26:`exp068` 当前没有真正实现“Reliability-Routed”。它测试到的是一个被全局软缩放后的 PAA，而不是面向遮挡区域的 selective completion。
experiments/exp200/design.md:6:- 与 exp191 (OA-SD + CE) 对比：relational distillation vs feature distillation
experiments/exp200/design.md:9:OA-RD (relational distillation) 在 CE 路线下应与 OA-SD 效果相当或更好。
experiments/exp200/design.md:17:- exp191 (OA-SD + CE): 63.2/75.4
experiments/exp286_lgpaOnly_1stg_t_od_s42/monitor.md:9:- Scaffold: Swin-Tiny + LGPA + OA-SD + ParAug + LOWER_BODY_OCC **- GCN** + 1-stage PSG
experiments/exp286_lgpaOnly_1stg_t_od_s42/monitor.md:54:1. Tiny backbone 的 semantic branch (LGPA + OA-SD + ParAug + LOWER_BODY_OCC) 已提供全部关键增益
experiments/exp139/claude_review_request_v2.txt:10:- `/root/work/SOLIDER-REID/experiments/exp139/design.md`
experiments/exp139/claude_review_request_v2.txt:11:- `/root/work/SOLIDER-REID/experiments/exp139/monitor.md`
experiments/exp139/claude_review_request_v2.txt:12:- `/root/work/SOLIDER-REID/configs/occluded_duke/pose_psg_gcn_lpcs_query_ctx.yml`
experiments/exp139/claude_review_request_v2.txt:13:- `/root/work/SOLIDER-REID/config/defaults.py`
experiments/exp139/claude_review_request_v2.txt:14:- `/root/work/SOLIDER-REID/processor/processor.py`
experiments/exp139/claude_review_request_v2.txt:15:- `/root/work/SOLIDER-REID/model/pose_backbone_model.py`
experiments/exp139/claude_review_request_v2.txt:16:- `/root/work/SOLIDER-REID/model/modules/pair_adaptive_fusion.py`
experiments/exp139/claude_review_request_v2.txt:17:- `/root/work/SOLIDER-REID/utils/metrics.py`
experiments/exp139/claude_review_request_v2.txt:18:- `/root/work/SOLIDER-REID/experiments/exp135/design.md`
experiments/exp139/claude_review_request_v2.txt:19:- `/root/work/SOLIDER-REID/experiments/decisions.md`
experiments/exp248/monitor.md:5:Test-time 实验, 在已有 checkpoint (exp244 LGPA-D+OA-SD) 上验证。
experiments/exp275_psg1_s_od_s42/monitor.md:5:- Log: `/tmp/exp275.log` → `/home/afr/SOLIDER-REID/log/occluded_duke/exp275_psg1_s_od_s42/train_log.txt`
experiments/exp275_psg1_s_od_s42/monitor.md:7:- Scaffold: Swin-Small + PSG 1-stage (LGPA/GCN/OA-SD/PLBOA/ParAug 全关)
experiments/exp275_psg1_s_od_s42/monitor.md:63:- Ckpt: `/home/afr/SOLIDER-REID/log/occluded_duke/exp275_psg1_s_od_s42/transformer_120.pth` (199MB)
experiments/exp200/monitor.md:4:对照: exp191 (OA-SD + CE): 63.2/75.4
experiments/exp200/monitor.md:24:**ep10 eval**: 39.9/51.3 (vs exp191 OA-SD: 34.3/46.8, **+5.6/+4.5!!**)
experiments/exp200/monitor.md:30:| Epoch | exp200 (OA-RD+CE) | exp191 (OA-SD+CE) | delta |
experiments/exp200/monitor.md:40:| Epoch | exp200 (OA-RD+CE) | exp191 (OA-SD+CE) | delta |
experiments/exp200/monitor.md:56:OA-RD 在 CE 路线 ep60 反转为负！不如 OA-SD！
experiments/exp200/monitor.md:57:OA-RD 的 relational distillation 在 CE 路线不如 OA-SD 的 feature distillation。
experiments/exp200/monitor.md:85:### vs exp191 (OA-SD + CE)
experiments/exp200/monitor.md:90:| exp191 (OA-SD) | 63.2% | 75.4% | +0.3/+1.5 |
experiments/exp200/monitor.md:93:OA-RD (relational distillation) 不如 OA-SD (feature distillation) 在 CE 路线。
experiments/cargo_cvpb/fgeu_realizability_result.md:1:# FGEU Realizability Kill-Switch — 结果 (DEAD 3/10, 撞 exp109 query-side 墙)
experiments/cargo_cvpb/fgeu_realizability_result.md:14:**realizable 同 camera tracklet union 只拿到 16.3% 的 oracle 恢复 (远低于 40% 门槛), 且只比免费 k-reciprocal 强 ~2.3x。FGEU 撞 exp109 query-side 墙 — 大 headroom 只存在于部署不可得的 cross-camera 同 ID 证据里, realizable 的同机位多帧拿不到。判 DEAD 3/10。**
experiments/cargo_cvpb/fgeu_realizability_result.md:42:| **B_oracle** cross-video gallery union | 76.58 | **+26.77** | ❌ 部署不可得 (exp109 上界) |
experiments/cargo_cvpb/fgeu_realizability_result.md:51:**fragility gate (只融弱 support 失败 vs 全融):**
experiments/cargo_cvpb/fgeu_realizability_result.md:53:- fuse-FRAGILE-only (bottom-50% support) dAP = +5.51 (n=45)
experiments/cargo_cvpb/fgeu_realizability_result.md:62:2. **但远够不到 headroom**: 在能直接和 oracle 比的 90 个 tracklet 上, realizable 只恢复 oracle 的 **16.3%**。大 headroom (oracle +26.77, 91% query 被救) 来自**跨 video / 跨机位的同 ID 证据**, 而那正是部署不可得的 (= exp109 墙)。
experiments/cargo_cvpb/fgeu_realizability_result.md:87:**DEAD 3/10 — exp109 query-side 墙。**
experiments/cargo_cvpb/fgeu_realizability_result.md:89:realizable 同 camera tracklet union 拿不到 ≥40% 的 oracle 恢复 (只有 16.3%)。FGEU 的大 headroom 是 cross-camera 同 ID 证据 (oracle, 部署不可得) 独有的, realizable 的同机位多帧复制不了。这正是 query-side oracle 变体, 撞 exp109 墙。
experiments/exp357_pose_shuffle_ks/design.md:38:- Medium-2(判读): NO-DROP 侧被裁剪对齐混淆(别人 pose 仍带粗糙 canonical 头/躯干/腿先验)。Codex/Claude 一致: 掉点=干净铁证(图特定 pose correspondence 重要); 不掉=只能说"精确图特定 pose 在对齐裁剪下非必需", 需补 **cross-PART(17关键点通道)shuffle** 二次确认(测解剖通道身份是否重要, 同图空间 support 不变)。最佳矩阵: cross-image + per-image channel-shuffle + no-pose/fixed-canonical control。
experiments/exp357_pose_shuffle_ks/design.md:44:- 下一步: cross-PART(通道)shuffle exp358 二次确认——打乱17关键点通道(破坏解剖部位身份, 保留同图空间 support)。若 exp358 也只小掉→解剖身份也不重要, 只是"某种空间池化结构"在涨→故事进一步塌; 若 exp358 大掉→解剖部位结构重要。
experiments/exp010/monitor.md:69:1. 冻结期间，classifier/PSG 学习适应了 SOLIDER 预训练的固定特征空间
experiments/prcv_2026_psg/phase3_design.md:17:**scaffold（纯 PSG，全部无 LGPA / GCN / OA-SD / PLBOA）**:
experiments/prcv_2026_psg/phase3_design.md:77:基于 `prcv_best_{t,s}.yml`，其余模块全开（LGPA-D + GCN + OA-SD + PLBOA）。
experiments/prcv_2026_psg/phase3_design.md:147:- Phase 3-A 的 exp270（no-PSG + 关一切）不是 `exp260b / exp262` 的 baseline，它是一个**纯 baseline**（SOLIDER + default data aug）。严格来说和 `4090-OD-PSG-small-lr8` 系列 baseline 一致。已有历史数据 `baseline ≈ 56/66 for Tiny, ~65/76 for Small`，但若时间允许可以重跑一次在 5060 Ti 上拿一致硬件基线。
experiments/exp286_lgpaOnly_1stg_t_od_s42/claude_review.md:24:- 其他 scaffold 模块 (LGPA / OA-SD / ParAug / LOWER_BODY_OCC) 自动从 `prcv_best_tiny.yml` 继承 default True
experiments/exp286_lgpaOnly_1stg_t_od_s42/claude_review.md:43:- pretrained swin_tiny/small/base + clip_part_text_features 全齐
experiments/exp335_vit_lgpa/CODEX_BRIEF.md:22:| 系统 | PSG+LGPA-D+OA-SD+GCN | 纯 LGPA-D |
experiments/exp139/claude_review_request.txt:4:- `/root/work/SOLIDER-REID/experiments/exp139/design.md`
experiments/exp139/claude_review_request.txt:5:- `/root/work/SOLIDER-REID/experiments/exp139/monitor.md`
experiments/exp139/claude_review_request.txt:6:- `/root/work/SOLIDER-REID/configs/occluded_duke/pose_psg_gcn_lpcs_query_ctx.yml`
experiments/exp139/claude_review_request.txt:7:- `/root/work/SOLIDER-REID/config/defaults.py`
experiments/exp139/claude_review_request.txt:8:- `/root/work/SOLIDER-REID/processor/processor.py`
experiments/exp139/claude_review_request.txt:9:- `/root/work/SOLIDER-REID/model/pose_backbone_model.py`
experiments/exp139/claude_review_request.txt:10:- `/root/work/SOLIDER-REID/model/modules/pair_adaptive_fusion.py`
experiments/exp139/claude_review_request.txt:11:- `/root/work/SOLIDER-REID/experiments/exp135/design.md`
experiments/exp139/claude_review_request.txt:12:- `/root/work/SOLIDER-REID/experiments/decisions.md`
experiments/exp319_tiny_oasdW05_s42/monitor.md:6:- 动机: OA-SD self-distillation 减半 (与 exp316 oasdW=2.0 互为反向 sweep)
experiments/exp319_tiny_oasdW05_s42/monitor.md:26:加上 exp316 (oasdW=2.0) MaxSim 0/-0.6, OA-SD weight default 1.0 验证为 sweet spot, 上下 1× 都微负。
experiments/decisions.md:7:**选择**: 从 SOLIDER 作者原始代码重新开始，纯 pose heatmap 方向
experiments/decisions.md:1107:**上下文**: exp148 PCVT 和 exp151 PVAT 全部失败。训练集 95.8% 可见率使得所有 visibility-dependent 训练方法无效。
experiments/decisions.md:1138:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/decisions.md:1143:1. **exp148 PCVT**: 早期加速（ep30 +2.4 mAP），后期被基线追平并反超（ep100 -0.9 mAP）。3-view 训练的 1/3 主损失稀释 + 训练数据缺乏 visibility 多样性。
experiments/decisions.md:1168:- PCVT (neutral)
experiments/decisions.md:1224:1. 近年的强路线把问题定义在 **target ambiguity / common visible support / retrieval-time reasoning**，而不是“再学一个融合权重”。
experiments/decisions.md:1289:   这符合“keypoint common-support 更适合作为补充项”的判断。
experiments/decisions.md:1413:   - common-support reasoning 对整体排序的修正作用
experiments/decisions.md:1441:   - 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
experiments/decisions.md:1442:   - 在 global triplet 上增加 support-aware hard mining 约束
experiments/decisions.md:1491:- 🔴 红队（方案 A）核心论点: SGMKC 实现成本极低（~15 行代码，无新参数），属于不同类别的改进（训练方法论 vs 架构/loss 添加），skeleton graph + masked prediction 组合是真正新颖的（FCFormer 用 transformer decoder，MAE 用 random patches，没有人在 skeleton graph 上做过 masked completion for ReID）。47 个失败实验都是架构添加或 loss 变体，SGMKC 是 self-supervised 训练策略——不同搜索空间。即使失败，负面结论也有论文价值。信心: 6/10
experiments/decisions.md:1492:- 🔵 蓝队（方案 B）核心论点: 47 个实验中 21/21 训练端改进全部失败，贝叶斯后验 P(失败) ≈ 95.7%。CLAUDE.md 明确说"不要做 GCN 小变体"。SkeletonMAE (2023) 已在动作识别上做过 skeleton graph masked autoencoding，novelty 受限。即使涨 1%，"在 GCN 上加 MAE 辅助任务"的 story 不够支撑 B 会。应该花时间精读 2024-2025 新工作（uncertainty-aware ReID、diffusion-based augmentation 等），找到 paradigm shift 级别的创新。信心: 8/10
experiments/decisions.md:1499:5. 但我认同蓝队的核心判断：SGMKC 更可能是 supporting experiment 而非 main contribution
experiments/decisions.md:1690:3. 若 `TDPC` 在 2-3 天内无明显正信号，则 fallback 到 retrieval-time `common-support recovery`，不继续做 `TDPC` 小修小补。
experiments/decisions.md:1738:   真正有效的 pair-specific reasoning 很可能必须发生在 `per-keypoint / common-support` 粒度，而不是 pooled person feature 粒度。
experiments/decisions.md:1749:  **duplicate-aware / confuser-aware 的 per-keypoint common-support reasoning**
experiments/decisions.md:1755:**上下文**: `exp108 DACCM` 完成了第二轮 retrieval-time 原型验证。该实验把 `exp107` 的思路从 pooled person embedding 下沉到 `per-keypoint / common-support` 粒度，并以 `exp030a cvk_hybrid` 为主基线，比较：
experiments/decisions.md:1771:   - per-keypoint common-support penalty 仍负面
experiments/decisions.md:1790:**上下文**: `exp109` 完成了 `Oracle Support Bank` 上界诊断。该实验使用 `exp030a cvk_hybrid` 的 target keypoint features，在 query+gallery 上用 GT same-ID 样本构造 leave-one-out 的 per-keypoint prototype。
experiments/decisions.md:1803:   **当前性能缺口里有一大块确实来自“support 不完整”，而不是 confuser suppression 失败。**
experiments/decisions.md:1804:3. 因而 `support-complete distillation` 已从“想法”升级为“有强 headroom 支撑的训练主线候选”。
experiments/decisions.md:1809:1. `oracle_feat_only` 已经大幅转正，说明关键不只是 weight 修正，而是 feature completion 本身。
experiments/decisions.md:1838:   - 这和“support-complete”要表达的 multi-view support 概念并不完全一致
experiments/decisions.md:1840:**选择**: 继续 `support-complete` 主线，但下一步只做 teacher reliability 的单变量改动。
experiments/decisions.md:1873:3. 这与当前论文主线也更一致：关键不只是“有多少 support”，而是“teacher support 是否足够干净可信”。
experiments/decisions.md:1892:1. 当前 `support-complete` 主线没有被否定；相反，它的瓶颈已比之前更清楚。
experiments/decisions.md:1903:   **reliable support-complete learning**
experiments/decisions.md:1935:   - 文献/代码学习（寻找不同的 feature completion / cross-view learning 机制）
experiments/decisions.md:1936:   - 围绕已确认的 `support incomplete` 问题重新设计新机制
experiments/decisions.md:1950:4. 这条线会把 story 从 `support incomplete / support-complete learning` 拉回到“GCN 小模块 + 组合扫点”。
experiments/decisions.md:1966:2. 若要切到新方向，必须先说明它相对 `support incomplete` 主线的关系，而不是直接跳到模块叠加。
experiments/decisions.md:1974:- `exp109-116` 则说明 `support-complete` 若被压成 `per-ID prototype`，会丢失太多 pair-specific 细节
experiments/decisions.md:1979:   **用已经被 `cvk_hybrid` 验证过的 common-support pairwise 几何，直接蒸馏 global embedding 的关系结构。**
experiments/decisions.md:1992:   - global embedding 需要被蒸馏成更符合 common-support geometry 的空间
experiments/decisions.md:2014:2. 当前最清楚的增益落在 `global`（`+0.6 / +0.4`），说明它确实把 common-support 几何迁进了 backbone/global 空间。
experiments/decisions.md:2015:3. `equal_concat` 仍接近持平，说明第一版 teacher 还不够强；瓶颈更像 teacher 的 `support incompleteness`，而不是 relational distillation 这件事本身无效。
experiments/decisions.md:2016:4. 因而 `exp109` 的高价值结论仍应保留：真正缺的不是再换一个 loss 形式，而是 **更 support-complete 的 teacher**。
experiments/decisions.md:2019:**把 `exp109` 的 support-complete bank 降级为 teacher enhancer，而不是 pointwise distillation target，构造 support-complete relational teacher。**
experiments/decisions.md:2022:1. 这条线同时保留了 `exp109` 的核心 headroom 和 `exp119` 已验证的 pairwise 机制。
experiments/decisions.md:2052:1. `support-complete teacher` 并没有“没生效”，相反，它已经稳定地增强了 teacher 几何。
experiments/decisions.md:2054:3. 结合 `exp109` 的 oracle 结论，更合理的新解释是：
experiments/decisions.md:2055:   **support-complete 监督的价值集中在 support-incomplete 样本；如果对所有 anchor 等权蒸馏，clean 样本会稀释掉这份增益。**
experiments/decisions.md:2056:4. 因此，`exp120` 否定的不是 `exp109 -> exp119` 这条主线，而只是：
experiments/decisions.md:2063:2. 它直接把 `exp109` 的低可见 headroom 转译成训练机制，而不是再做 generic loss 调参。
experiments/decisions.md:2065:   - 单图遮挡带来 support incomplete
experiments/decisions.md:2066:   - pose branch 提供 support-complete relational teacher
experiments/decisions.md:2067:   - 但 distillation 必须 **selective**，聚焦真正存在 support gap 的 anchor
experiments/decisions.md:2090:2. 但它没有把 `support-complete teacher` 的增强转成更好的指标，反而更像削弱了有效监督总量。
experiments/decisions.md:2093:4. `support-complete` 主线本身仍然成立；被否定的只是 sample-level `replace_ratio` 作为路由信号太粗。
experiments/decisions.md:2099:2. 它直接回应 `exp122` 的失败：真正该被强调的不是“这个样本补了多少 keypoint”，而是 **support-complete teacher 实际改变了哪些 pair 几何**。
experiments/decisions.md:2101:   - 单图遮挡带来 support incomplete
experiments/decisions.md:2102:   - support-complete teacher 改变一部分 pairwise comparability
experiments/decisions.md:2128:1. `stable teacher` 已经被 `exp121` 明确坐实为有效 supporting mechanism，但它不是当前主突破口。
experiments/decisions.md:2174:1. 这一步仍然严格锚定 `exp109 -> exp119` 主线，不回到 sample-level，也不回到 generic 模块叠加。
experiments/decisions.md:2180:   **只把被 support completion 真正改变过的 comparability relations 蒸进 global embedding。**
experiments/decisions.md:2185:3. 不同时改 `alpha`、不改 teacher bank、不断开 `support-complete` teacher，避免再次混入多个变量。
experiments/decisions.md:2237:2. `exp124` 证明了单纯增大 focus 强度也有效，但最终不如 `exp125`，因此它应退居 supporting branch。
experiments/decisions.md:2267:- 同时当前 `exp109` 主线已经暴露出另一个未被打透的缺口：
experiments/decisions.md:2270:  - 二者都没有把 oracle support-complete 上界真正兑现出来
experiments/decisions.md:2274:2. 但这不意味着要离开 `exp109`；相反，最合理的下一步仍然是沿 `support incomplete -> support-complete learning` 这条主线，直接测试更强的 feature-level 兑现机制。
experiments/decisions.md:2275:3. `SCFR≈SCKD` 只能说明 “hard replace 不优于 loss-only”，不能说明 “feature-level support completion 整体无效”。
experiments/decisions.md:2279:2. 该实验保持 `bank`、`warmup`、`threshold` 与 `exp116` 同量级，只改 low-vis keypoint 如何利用 support-complete prototype：
experiments/decisions.md:2284:1. 这是沿 `exp109` 主线的下一阶段机制，而不是换题。
experiments/decisions.md:2291:1. `exp127` 只测试 `residual completion` 本身，不同时叠加 `CSRD` 或新的 pair routing。
experiments/decisions.md:2310:1. `SCRC` 没有把 feature-level support completion 推成更强结果，反而 late-stage 基本塌成了“近似 hard replace”。
experiments/decisions.md:2311:2. 因而 `exp109` 被否定的不是 `support incomplete` 问题定义，而是：
experiments/decisions.md:2312:   - per-ID prototype 的 direct feature completion 兑现方式
experiments/decisions.md:2313:3. `freeze20/30` 的既有证据已经足够说明它只是弱 supporting mechanism，不值得继续占用本地算力。
experiments/decisions.md:2315:   **support-complete teacher 的新增 correction 仍被完整 teacher target 稀释。**
experiments/decisions.md:2318:1. 关闭 `SCRC` 这条本地主线，不再追加 direct completion 变体。
experiments/decisions.md:2324:   - support-complete teacher 的增量信息是真实存在的
experiments/decisions.md:2330:3. 这一步比继续扫 `alpha/top_ratio/freeze` 更接近方法机制，也更贴合 `exp109` 的问题定义。
experiments/decisions.md:2401:   **pair-specific support-complete correction 不能被当前单向量 student 充分吸收。**
experiments/decisions.md:2422:3. `exp040/045` 的固定 `cvk_hybrid` 已经证明 pair-specific common-support correction 在检索时能转成稳定正信号。
experiments/decisions.md:2563:1. `exp136` 到此结案，保留为 supporting 证据
experiments/decisions.md:2713:1. `exp138` 已经提供了足够的负边界：平滑 top-sensitive 只能算 supporting 机制
experiments/decisions.md:2716:   - pose 定义 common support
experiments/decisions.md:2717:   - query context 决定 pair correction 应如何解释该 support
experiments/decisions.md:2722:- `exp138` 已停表，结论为 supporting 线
experiments/decisions.md:2829:1. `exp139` 到 `ep80` 为止，已经基本追平当前最强 supporting 线 `exp135`
experiments/decisions.md:2911:## [2026-03-21 19:02] 决策：不启动 `exp141`，本地转向更大的 feature completion 主线 `exp142`
experiments/decisions.md:2919:- `exp109` 的 oracle 结论始终没有被推翻：
experiments/decisions.md:2920:  - 真正 headroom 来自 `single-image support incomplete`
experiments/decisions.md:2925:   - 但它当前更像 supporting 机制，而不是确定的论文主方法
experiments/decisions.md:2928:3. 当前更合理的本地大转向应回到 `exp109` 根问题本身：
experiments/decisions.md:2930:   - 而在特征层直接补全 keypoint-level support
experiments/decisions.md:2940:2. 它直接回应 `exp109` 的核心发现，而不是继续在 scoring 层修修补补
experiments/decisions.md:2947:**上下文**: exp142 SKC 训练完成。最终结果 mAP 60.3% / R1 71.8%（equal_concat），相对 exp030a -0.8% mAP / -1.9% R1。feature-level support-supervised completion 方向确认失败。
experiments/decisions.md:2950:1. SKC completion 模块虽然活跃（gate≈0.26, delta_norm≈1.5），但 skc_pre≈skc_post 说明修改方向不是向 support prototype 靠近
experiments/decisions.md:2952:3. 这与 SGMKC, SCRC, SCKD 的结论一致：feature-level completion 在 15K 数据集上不可行
experiments/decisions.md:2961:1. feature-level completion 方向已被彻底证伪（5+ 次尝试），不值得继续做 ablation
experiments/decisions.md:2971:### [2026-03-22 09:35] 决策：停止沿 `exp141-147` 的小修补漂移，重新回到 `exp109` 的大问题定义
experiments/decisions.md:2975:- `SKC` 再次确认 feature-level completion 失败
experiments/decisions.md:2980:1. `single-image support incomplete` 这个问题定义没有被推翻
experiments/decisions.md:2982:   - cross-image prototype completion 的兑现方式
experiments/decisions.md:2988:  A. 继续在已有 retrieval scorer / completion / bias 路线上小修小补  
experiments/decisions.md:2989:  B. 回到 `exp109` 的问题定义，重新设计两个真正不同的大方向  
experiments/decisions.md:2996:   - 单图能否被改写成“伪多 support 学习”对象
experiments/decisions.md:3003:  - `exp148 PCVT`
experiments/decisions.md:3009:### [2026-03-22 15:48] 决策：停止 `SCFA`，将本轮方向收缩为“继续追 PCVT，远程空出来给下一条真正不同的新机制”
experiments/decisions.md:3012:- `exp148 PCVT` 已给出连续 `ep10/20/30` 的稳定 `mAP` 正向
experiments/decisions.md:3017:   - `PCVT` 至少跑出了连续可见的验证正信号
experiments/decisions.md:3022:3. `PCVT` 当前最主要风险不是方法失效，而是：
experiments/decisions.md:3034:   - 继续把 `PCVT` 跑清楚
experiments/decisions.md:3035:   - 并尽快设计一个与 `PCVT` 真正不同的新大方向占用远程卡
experiments/decisions.md:3041:- 本地主线继续保留 `exp148 PCVT`
experiments/decisions.md:3046:**上下文**: exp190-195 系列实验完成，揭示了 OA-SD 和 3-view parallel aug 的组合关系，以及 OA-SD global-only 解决 SupCon 梯度冲突的新机制。
experiments/decisions.md:3050:- exp191 (OA-SD+CE): 63.2/75.4 — OA-SD 独立有效
experiments/decisions.md:3052:- exp193 (3-view+OA-SD+CE): 64.4/76.5 — additive! R1 追平 SupCon
experiments/decisions.md:3054:- exp195 (SupCon+OA-SD global-only): ep70=60.2/73.4 — 梯度冲突解决!
experiments/decisions.md:3057:  A. exp196: 3-view + SupCon + OA-SD global-only（终极组合，验证所有创新 additive）
experiments/decisions.md:3063:1. OA-SD global-only 是一个新机制（满足创新门槛 #2 和 #3），不是简单组合
experiments/decisions.md:3065:3. exp195 已验证 SupCon+OA-SD global-only 兼容，exp193 已验证 3-view+OA-SD additive
experiments/decisions.md:3072:**上下文**: exp196 (3-view + SupCon + OA-SD global-only) 在 ep70 持续落后 exp187 (3-view + SupCon) -1.8/-0.9。OA-SD global-only 的 distillation 信号 (oa_sd=0.01) 过弱。
experiments/decisions.md:3075:- OA-SD + CE 有效 (+2.9/+2.6)
experiments/decisions.md:3076:- OA-SD + SupCon (all-token) 有梯度冲突 (exp188 负向)
experiments/decisions.md:3077:- OA-SD + SupCon (global-only) 无梯度冲突但信号太弱 (exp195/196)
experiments/decisions.md:3078:- 结论: **OA-SD 和 SupCon 本质上互斥**——选一个用
experiments/decisions.md:3082:  B. 最终配置用 OA-SD+CE (exp193: 64.4/76.5) — R1 几乎一样
experiments/decisions.md:3090:3. OA-SD 路线: R1 几乎一样，且 OA-SD 概念更新颖 (self-distillation 范式)
experiments/decisions.md:3096:**上下文**: 5 个研究 agent 并行调研后，发现 Relational Knowledge Distillation (RKD, CVPR 2019) 可能解决 OA-SD vs SupCon 互斥问题。OA-SD 在 token 级别 match 个体特征 → 与 SupCon 冲突。RKD match pairwise similarity structure → 不碰个体特征 → 理论上与 SupCon 兼容。
experiments/decisions.md:3106:1. 直接解决已知问题 (OA-SD vs SupCon 冲突)
experiments/decisions.md:3107:2. 满足创新门槛: 问题层面(关系级 vs 特征级) + 机制层面(RKD in occluded ReID) + 证据链(OA-SD→OA-RD)
experiments/decisions.md:3115:**核心发现**: 任何形式的 EMA self-distillation (OA-SD/OA-RD) 都与 SupCon 不兼容。
experiments/decisions.md:3116:- OA-SD (feature-level): exp188/196 失败
experiments/decisions.md:3141:- exp198 (STM + OA-SD): ±0 — 同上
experiments/decisions.md:3143:- exp200 (OA-RD + CE): -0.3/-1.5 — OA-RD 不如 OA-SD
experiments/decisions.md:3173:| 当前 | Small GCN+PAA+CE+OA-SD | 70.5% |
experiments/decisions.md:3186:- exp209 (STD-PR+CE+OA-SD) ep30=56.0/69.3，落后 5%，终止
experiments/decisions.md:3188:- OA-SD teacher Critical bug 修复并部署
experiments/decisions.md:3190:### [2026-04-01 10:30] 决策 #X — MaxSim + PKC + Fixed OA-SD 路线
experiments/decisions.md:3192:**上下文**: MaxSim hybrid 在 exp206 checkpoint 上无需重训给 +1.8% mAP (70.3→72.1)。OA-SD teacher bug 已修复。PKC (Per-Keypoint Contrastive) 开始测试。
experiments/decisions.md:3198:| 已确认 | Small GCN+PAA+CE+OA-SD + maxsim_hybrid | **72.1%** |
experiments/decisions.md:3231:4. 回到 STD-PR+SupCon 路线（已知 67.9+maxsim ≈ 69.7，不如 GCN+OA-SD）
experiments/decisions.md:3251:3. non-detached cosine alignment: 与 OA-SD 竞争 → 负面
experiments/decisions.md:3262:- PACI + OA-SD (exp218): `61.9 / 74.2` (vs `exp191 63.2 / 75.4` = **-1.3 / -1.2**)
experiments/decisions.md:3270:| OA-SD-only | **63.2** | 64.2 |
experiments/decisions.md:3271:| OERL+OA-SD | 62.2 | 64.3 |
experiments/decisions.md:3272:| PACI+OA-SD | 61.9 | 64.1 |
experiments/decisions.md:3274:这一步更准确的结论不是 “OA-SD 已达 64.4 ceiling”，而是：
experiments/decisions.md:3275:1. 在 `OA-SD / OERL / PACI` 这三条已完成 Tiny 线内部，`maxsim_hybrid` 都落在 `64.1~64.3`
experiments/decisions.md:3276:2. `MaxSim` 对 `OA-SD` 本身仍是正向的（`63.2 -> 64.2`），只是 `OERL/PACI` 并没有把这个 test-time 上限继续抬高
experiments/decisions.md:3283:2. non-detached losses 与 CE/OA-SD 冲突 → 灾难
experiments/decisions.md:3298:| OA-SD-only | 63.2/75.4 | 64.2/77.1 |
experiments/decisions.md:3299:| GSPB+OA-SD (scale=0.05) | 62.9/74.3 | **64.6/76.0** |
experiments/decisions.md:3300:| PADPQ K=4+OA-SD | **63.7/74.5** | 63.9/74.8 |
experiments/decisions.md:3301:| PADPQ K=8+OA-SD | 进行中 | 进行中 |
experiments/decisions.md:3304:1. GSPB: 早期加速 +5.8% at ep10，按当前测试记录 `maxsim_hybrid` 相对 OA-SD 为 `+0.4`，是目前 Tiny 线上最高的 `maxsim` mAP
experiments/decisions.md:3381:**结果**: 65.3/75.7 (+2.1/+0.3 vs exp191 GCN+OA-SD)
experiments/decisions.md:3389:4. LGPA-D 无 OA-SD (63.6) ≈ GCN + OA-SD (63.2): CLIP 价值 ≈ OA-SD
experiments/decisions.md:3395:- 与 OA-SD 正交: 可叠加
experiments/decisions.md:3461:1. 只有 backbone 修改有效 (PSG, OA-SD)
experiments/decisions.md:3472:2. LGPA-D 虽然 single novelty 4.5/10, 但与 PSG+OA-SD+MaxSim 组成完整 framework novelty 更高
experiments/decisions.md:3479:- `exp109` oracle support bank 仍是仓库内最强问题证据
experiments/decisions.md:3481:- `LGPA-D + GCN + OA-SD + MaxSim + flip` 虽然结果强，但主问题定义仍偏弱
experiments/decisions.md:3486:  B. 回到 `exp109`，把主线改成“single-image support incomplete”的训练对象重写
experiments/decisions.md:3491:1. `exp109` 已给出巨大 headroom：`61.88 -> 66.15 -> 70.40 mAP`，而后续任何实验都没有正面回应这个 gap
experiments/decisions.md:3493:3. `MaxSim / POT / flip` 主要仍是 test-time supporting evidence，不能作为训练端主贡献
experiments/decisions.md:3501:1. 用 pose 定义互补 support 伪视图，而不是随机多视图分类
experiments/decisions.md:3502:2. 用互补视图组装 support-complete teacher token set
experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/decisions.md:3528:  A. 继续沿刚提出的 `PSCD/support-complete` 新路线展开
experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/decisions.md:3586:**上下文**: 用户审查发现 `processor/processor.py::_extract_feat_flip` 在 `equal_concat` 模式下用 whole-vector 平均,让 evaluator 单次 L2-normalize,破坏了 `equal_concat` 在 model 端每块 L2-normed concat 的"均等贡献" cosine 语义。其他 pose 模块(如 OA-SD 打破训练端 flip 对称 + GCN per_part 的 L/R 非完全对齐)导致每块的 flip-后 norm `r_k` 不同,whole-vector 重归一化 `sqrt(Σ r_k²)` 给各块的有效权重被扭曲。
experiments/decisions.md:3668:- lab4090 有完整代码(`/home/afr/SOLIDER-REID`)、pretrained(swin_*, ViTPose-Huge, VisPredictHead, clip_part_text)、数据集(Occluded-Duke原图+pose_data)、mmpose-abu conda env(torch 1.13.1+cu,mmpose 1.3.2,mmdet 3.2.0)
experiments/decisions.md:3679:1. 确保 pose_data 在 lab4090 与 srvB(训练集 ground truth 源)完全一致,避免未来 Full Scaffold (OA-SD/PLBOA) 因 visibility 用 score proxy 出现微小精度漂移
experiments/decisions.md:3732:- lab3090 checkpoint `pretrained/best_coco_AP_epoch_210.pth` md5 = `90496f7405b61228dde244657c357c7a`
experiments/decisions.md:3848:**srvA resume**: 用户重新续费, ssh 通, GPU 0 MiB / 15849 free / 0% util, Occ-Duke + Occ-PoseTrack + Market + ReID 数据齐全, pretrained swin_{tiny,small,base} + clip 齐全。
experiments/decisions.md:4039:- exp288 Swin-Small + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + PSG `[-1]` (**无 GCN**) FINAL @ 12:51 CST srvC
experiments/decisions.md:4114:- exp293 config 确认 PLBOA=True 激活, OA-SD WARNING 消失 (teacher/student 现有差异)
experiments/decisions.md:4125:**exp293 Base Market + PLBOA** (lab4090, OA-SD 激活):
experiments/decisions.md:4132:  - 假设验证: 第 3 情景 "两力相抵, 微 net 负" (OA-SD 收益 < 分布偏差)
experiments/decisions.md:4144:- OD (exp285b etc): PLBOA True, OA-SD 蒸馏有效, +性能
experiments/decisions.md:4145:- OP (exp265 etc): PLBOA True, OA-SD 蒸馏有效, +性能
experiments/decisions.md:4292:- ckpt: `/home/afr/SOLIDER-REID/log/occluded_duke/exp263b_best_b_od_s42_full120/transformer_120.pth`
experiments/decisions.md:4470:**执行结果**：建议砍 frozen-MLLM-reasoner 廉价首验，转 exp324（DINO-correspondence，更 frontier-independent）或换机制。
experiments/decisions.md:4473:### [2026-06-16] 决策 #exp324 — DINO emergent correspondence + pose-anchored part-MaxSim 首验偏正
experiments/decisions.md:4475:**上下文**：exp323 frozen-MLLM 线偏负后，按搬范式 #2 路线做 frozen DINOv2-base 廉价首验（training-free）：
experiments/decisions.md:4482:ALL 子集同向更明显（pose-part 3.21/7.87 vs holistic 0.64/0.90）。绝对分低（heavy 1.86 mAP）但落在 DINO 零样本 ReID 文献区间（0.3-4.7）。
experiments/decisions.md:4484:**理由**：(1) 三种表征**单变量隔离干净**——(b)/(c) 都是 5 同序 part 向量在 common-visible part 求均值，唯一差别是锚定方式（pose vs 固定带），grid 几乎不涨而 pose 大涨，直接证明"姿态把 DINO token 约束到身体部位语义"是涨点来源，不是部位分解 trivial 效果；
experiments/decisions.md:4486:(3) 与 exp323（frozen 干预无效）形成对照——同样 frozen + 同样 pose，但 DINO dense correspondence 这条**有信号**，差别在表征端而非 LLM-reasoning 端。
experiments/decisions.md:4487:**执行结果**：exp324b 候选——冻结 DINO，仅训一个轻量 part-projection 头（或 LoRA）把 token 投到 ReID-judiciable 空间，
experiments/decisions.md:4491:### [2026-06-16] 决策 #exp327 — 更强冻结对应源（DINOv2-with-registers）止损
experiments/decisions.md:4493:**上下文**：exp324 frozen DINOv2-base pose-part 重遮挡 1.86，天花板低。问"换更新/更干净的冻结 SSL 源能否抬过 1.86"。hyy GPU1，唯一变量=特征源。DINOv3-vitb16 gated（hf-mirror 需 token）下不了，改用 ungated 的 dinov2-with-registers-base（registers 去 high-norm artifact token，更干净 dense 特征）。
experiments/decisions.md:4499:**理由**：registers 更干净特征只蹭出 +0.29 mAP（heavy），远不足以独立可用（exp324b 头已到 14）；印证 exp324 假说**训练-free 天花板瓶颈在 "frozen" 本身，不在 SSL 模型新旧/registers**。换更强冻结 DINO 源不是天花板解。
experiments/decisions.md:4504:**上下文**：exp324 frozen DINOv2-base pose-part 重遮挡 1.86。对应特征综述称 SD UNet 中间特征（DIFT）在遮挡/姿态对应基准上比 DINO 高 14-19 PCK。问"换 SD-DIFT 特征源能否超 1.86"。hyy GPU0，唯一变量=特征源（DINOv2→SD-v1.5 UNet up_blocks[1] DIFT，t=100 ensemble=4）。
experiments/decisions.md:4507:  B. 不超 → SD 训练-free 不优于 DINO，止损。
experiments/decisions.md:4508:**结果**：DIFT smoke（500 gallery）pose-part heavy **9.92**（趋势第一，误导），但 **FULL（17661 gallery）塌到 0.73（−1.13 vs 1.86）**，更不及 dinov2-registers 2.15。机制方向仍在（pose 0.73 > grid 0.35 > holistic 0.22）但绝对判别性远低于 DINO。
experiments/decisions.md:4510:**理由**：(1) DINO 从 smoke 2.55→full 1.86 仅小降，DIFT 从 9.92→0.73 **灾难性塌**——证明 **SD/DIFT 特征 category-level 语义对应强（PCK 高）但 instance-level 身份判别弱**（与 SD-DINO / Tale-of-Two-Features 文献一致：SD 与 DINO 互补、SD 不主导 instance retrieval）；(2) instance-discrimination 是 SD 特征的**结构性短板**（非超参问题），扫 t/up_block/ensemble 不会救；(3) 训头起点 0.73 远低于 DINO（1.86→14），不值得上 exp326b。
experiments/decisions.md:4511:**执行结果**：SD/DIFT 线止损，不上头。**重要方法论教训写入铁律：训练-free probe 必须用全量 gallery 判定绝对值，小 gallery smoke 只验流程不验数值**——DIFT 是活教材（smoke 排第一、full 垫底）。结合 exp327（registers +0.29 小幅、不破天花板）：**换特征源（更新 DINO / 换 SD 范式）都不是 frozen 天花板的解**，瓶颈在 frozen 本身（需 LoRA/解冻，即 exp324d 线）或换"DINO 补 Swin"重量级角度（planner #1 oracle）。
experiments/decisions.md:4513:### [2026-06-16] 决策 #exp324i — 做"解相关感知 DINO-LoRA"作 FM-import 方向最后一个真 method shot
experiments/decisions.md:4515:**上下文**：夜间 FM-import 全线证负，headline = 判别性-互补性张力（adaptation 让 DINO 判别化但趋同 Swin，融合只 +0.37）。lab-3090-d 空闲。用户睡前铁令"整夜不停务必找一个有用创新点"。问：直接用解相关损失攻击该张力，能否换来真互补、融合超 SOTA？
experiments/decisions.md:4517:  A. 跑 exp324i（跨协方差解相关 DINO-LoRA，λ=0 vs λ=1 单变量）——真机制介入，成则 method、败则把张力升级为强结论。
experiments/decisions.md:4526:**上下文**：exp324i（解相关感知 DINO-LoRA）e10 matched oracle 出（λ=0 vs λ=1）。
experiments/decisions.md:4563:**上下文**: 夜间范式调研唯一过审强 bet=burstiness(VLAD-BuFF/face-set import)。0-GPU 前提在 frozen DINO 成立(occluded +0.0206 更 bursty)。e120 弱 baseline(TransReID 53.5)训练模型真实判据。
experiments/decisions.md:4608:- 基建: 清 hyy /hy-tmp(47→25G 用),传 pose_data 上 hyy(torch2.7+cu128 在 sm_120 上验证可跑 SOLIDER)。
experiments/decisions.md:4611:**基建**: Clash 两修(都 live,非永久):① `PROCESS-NAME,tailscaled,DIRECT`(tailscaled 直连 DERP→4090 relay 复活)② `tun.dns-hijack: ['any:53']`(原为空→gpushare/hyy DNS 解析失败)。hyy(5060Ti sm_120)需 torch2.x,跑不了 SOLIDER 的 torch1.13+mmcv2.1。详见 [[lgpa-d-reproduction-gotchas]] memory。
experiments/decisions.md:4696:- **测试 C Singleton Merge = DEAD**: NN-is-head 0.72 只反映 head 占 72% 图像质量。per-head-ID(n=450/311 真功效)Spearman(support, attraction-PER-IMAGE)+0.003/+0.005≈0, 分箱 per-image 甚至下降。support-calibrated 阈值几乎无增益(d≈−0.003)且 40-60% level 退回 global。被 "head 图多→NN 彩票多" trivial count 吃掉。
experiments/exp335_vit_lgpa/design.md:32:lab-3090-d（ControlMaster；CLIP 文本特征缓存 pretrained/clip_part_text_features.pt；open_clip 已装）。
experiments/exp200/claude_review.md:11:- 与 exp191 (OA-SD + CE, 63.2/75.4) 直接对比：relational vs feature distillation
experiments/exp200/claude_review.md:74:- L741: `if not oa_sd_enabled` → True（OA-SD 未启用）
experiments/exp200/claude_review.md:81:### c.6 OA-SD 块 (L685) 不触发
experiments/exp200/claude_review.md:83:- `if oa_sd_enabled and ...` — `oa_sd_enabled=False` → 整个 OA-SD 块跳过
experiments/exp200/claude_review.md:119:- Teacher forward 独立运行（不依赖 OA-SD）
experiments/exp200/claude_review.md:120:- OA-SD 块不触发，OA-RD 块正确触发
experiments/exp222/design.md:1:# 实验 exp222: GSPB on Small — GCN+PAA+CE+OA-SD + gradient scale=0.05
experiments/exp222/design.md:4:- exp220 (GSPB on Tiny): equal_concat 62.9 (-0.3 vs OA-SD), **maxsim_hybrid 64.6 (+0.4 vs OA-SD!)**
experiments/cargo_cvpb/cvpb_rma.log:2:# RMA-TIReID ZERO-TRAINING KILL-SWITCH  (frozen SOLIDER exp260b, RSTPReid)
experiments/cargo_cvpb/cvpb_rma.log:6:/root/work/SOLIDER-REID/model/backbones/swin_transformer.py:1193: UserWarning: DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead
experiments/cargo_cvpb/cvpb_rma.log:7:  warnings.warn('DeprecationWarning: pretrained is deprecated, '
experiments/cargo_cvpb/cvpb_rma.log:14:Loading pretrained model from /root/work/SOLIDER-REID/log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth
experiments/cargo_cvpb/cvpb_rma.log:37:TEST A — image-image identity manifold on RSTPReid test (frozen SOLIDER)
experiments/cargo_cvpb/cvpb_rma.log:94:SUMMARY / VERDICT  (RSTPReid, frozen SOLIDER exp260b)
experiments/exp141/claude_review_request_v2.txt:1:请对 `/root/work/SOLIDER-REID` 中准备启动的 `exp141` 再做一次**全面代码审查**，目标是判断修复后的 `Competition-Context LPCS` 是否允许启动训练。
experiments/exp141/claude_review_request_v2.txt:11:1. `/root/work/SOLIDER-REID/experiments/exp141/design.md`
experiments/exp141/claude_review_request_v2.txt:12:2. `/root/work/SOLIDER-REID/experiments/exp141/monitor.md`
experiments/exp141/claude_review_request_v2.txt:13:3. `/root/work/SOLIDER-REID/configs/occluded_duke/pose_psg_gcn_lpcs_comp_ctx.yml`
experiments/exp141/claude_review_request_v2.txt:14:4. `/root/work/SOLIDER-REID/configs/occluded_duke/pose_psg_gcn_lpcs_fix.yml`
experiments/exp141/claude_review_request_v2.txt:15:5. `/root/work/SOLIDER-REID/model/modules/pair_adaptive_fusion.py`
experiments/exp141/claude_review_request_v2.txt:16:6. `/root/work/SOLIDER-REID/model/pose_backbone_model.py`
experiments/exp141/claude_review_request_v2.txt:17:7. `/root/work/SOLIDER-REID/processor/processor.py`
experiments/exp141/claude_review_request_v2.txt:18:8. `/root/work/SOLIDER-REID/utils/metrics.py`
experiments/exp141/claude_review_request_v2.txt:19:9. `/root/work/SOLIDER-REID/experiments/exp139/design.md`
experiments/exp141/claude_review_request_v2.txt:33:- `/root/work/SOLIDER-REID/experiments/exp141/claude_review_v2.md`
experiments/exp042/design.md:13:- 如果 `cvk_hybrid` 真的是 common-support correction，那么它应主要改善那些：
experiments/exp223/design.md:71:- exp191 OA-SD (fixed sampling): 63.2/75.4
experiments/exp139/design.md:14:- 这个 query 当前的 support 完整度如何
experiments/exp139/design.md:15:- 这个 query 的 global / common-support 分歧有多大
experiments/exp139/design.md:24:4. 当前 query 的平均 common support
experiments/exp139/design.md:25:5. 当前 query 的平均 global / common-support 分歧
experiments/exp139/design.md:46:   - `support_ratio`
experiments/exp139/design.md:53:   - `row_support_mean`
experiments/exp335_vit_lgpa/monitor.md:91:   - **LGPA-D 从未单独跑过**。exp244(+4.4)/exp245g(70.2)全是 **POSE_BACKBONE_PSG=True + LGPA + OA-SD + PARALLEL_AUG + 384 + Swin** 完整系统。
experiments/exp335_vit_lgpa/monitor.md:98:**重要 reframe**:LGPA detached → global-only == no-LGPA baseline,故 equalcat vs global = LGPA-D vs baseline(正确判据)。equalcat < baseline → 这套 ViT 设置上 LGPA-D 描述子未超 baseline。原版 70.2>baseline 很可能也靠 OA-SD+parallel-aug+384+Swin 全系统。
experiments/exp335_vit_lgpa/monitor.md:103:- **exp245g(70.2) = Swin-Small + LGPA-D + OA-SD + GCN（全系统）,不是纯 LGPA-D。** 纯 LGPA-D(无 OASD)Swin-Tiny ≈63.6(baseline 56.6 → +7 真涨)。
experiments/exp222/monitor.md:1:# exp222 Small + GCN+PAA+CE+OA-SD + GSPB (scale=0.05) 监控
experiments/exp222/monitor.md:3:配置: Small GCN+PAA+CE+OA-SD + POSE_PART_GRAD_SCALE=0.05
experiments/exp222/monitor.md:7:**动机**: exp220 (Tiny) 证明 GSPB+MaxSim=64.6 > OA-SD+MaxSim=64.2 (+0.4)
experiments/exp222/monitor.md:86:**Small 的 GCN+PAA+OA-SD+MaxSim 72.4% 仍是最佳。**
experiments/cargo_cvpb/error_analysis_geom.py:4:冻结 AG-ReID.v2 baseline(swin_small SOLIDER),提 BN 特征 → cosine distmat,
experiments/cargo_cvpb/error_analysis_geom.py:12:  cd /home/afr/SOLIDER-REID/experiments/cargo_cvpb && \
experiments/cargo_cvpb/error_analysis_geom.py:13:  python error_analysis_geom.py --ckpt /home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090/model_final.pth \
experiments/cargo_cvpb/error_analysis_geom.py:14:    --data_root /home/afr/SOLIDER-REID/data \
experiments/cargo_cvpb/error_analysis_geom.py:15:    --swin_pretrain /home/afr/SOLIDER-REID/pretrained/swin_small.pth 2>&1 | tee /tmp/err_analysis_geom.log
experiments/cargo_cvpb/error_analysis_geom.py:28:ap.add_argument('--data_root', default='/home/afr/SOLIDER-REID/data')
experiments/cargo_cvpb/error_analysis_geom.py:29:ap.add_argument('--swin_pretrain', default='/home/afr/SOLIDER-REID/pretrained/swin_small.pth')
experiments/cargo_cvpb/error_analysis_geom.py:36:    backbone='swin_small', swin_pretrain=cli.swin_pretrain, swin_semantic_weight=0.2,
experiments/exp042/monitor.md:51:5. 同时也存在 `top1_degraded` 样例，说明当前 common-support reasoning 还不是无代价增强。
experiments/exp141/claude_review_request.txt:1:请对 `/root/work/SOLIDER-REID` 中准备启动的 `exp141` 做一次**全面代码审查**，目标是判断它是否允许启动训练。
experiments/exp141/claude_review_request.txt:9:1. `/root/work/SOLIDER-REID/experiments/exp141/design.md`
experiments/exp141/claude_review_request.txt:10:2. `/root/work/SOLIDER-REID/experiments/exp141/monitor.md`
experiments/exp141/claude_review_request.txt:11:3. `/root/work/SOLIDER-REID/configs/occluded_duke/pose_psg_gcn_lpcs_comp_ctx.yml`
experiments/exp141/claude_review_request.txt:12:4. `/root/work/SOLIDER-REID/config/defaults.py`
experiments/exp141/claude_review_request.txt:13:5. `/root/work/SOLIDER-REID/model/modules/pair_adaptive_fusion.py`
experiments/exp141/claude_review_request.txt:14:6. `/root/work/SOLIDER-REID/model/pose_backbone_model.py`
experiments/exp141/claude_review_request.txt:15:7. `/root/work/SOLIDER-REID/processor/processor.py`
experiments/exp141/claude_review_request.txt:16:8. `/root/work/SOLIDER-REID/utils/metrics.py`
experiments/exp141/claude_review_request.txt:17:9. `/root/work/SOLIDER-REID/experiments/exp135/design.md`
experiments/exp141/claude_review_request.txt:18:10. `/root/work/SOLIDER-REID/experiments/exp139/design.md`
experiments/exp141/claude_review_request.txt:38:- `/root/work/SOLIDER-REID/experiments/exp141/claude_review.md`
experiments/exp292_target_s_m_s42/codex_review.md:19:Code = exp290 approved diff. All structural concerns (shape, dtype, backward compat, OA-SD, flip test, pose dropout) resolved in exp290 codex review.
experiments/exp139/claude_review_v2.md:13:2. **Label-dependent context** — 已修复。新版 `build_query_context_descriptors()` (`pair_adaptive_fusion.py:48-74`) 的 5 个特征 (`row_mean`, `row_std`, `row_min`, `row_support_mean`, `row_change_mean`) 全部来自距离矩阵和 support ratio 统计，不依赖任何 label 信息。
experiments/exp223/monitor.md:1:# exp223 Tiny + GCN+PAA+CE+OA-SD + PADPQ K=4 监控
experiments/exp223/monitor.md:3:配置: Tiny GCN+PAA+CE+OA-SD + Deformable keypoint sampling (K=4)
experiments/exp223/monitor.md:4:对照: exp191 OA-SD (fixed sampling): 63.2/75.4, exp220 GSPB+MaxSim: 64.6
experiments/exp223/monitor.md:27:**ep10: 37.5/47.3** (vs OA-SD 34.3/46.8 = **+3.2/+0.5**)
experiments/exp223/monitor.md:29:| Method | ep10 mAP/R1 | vs OA-SD |
experiments/exp223/monitor.md:31:| OA-SD-only | 34.3/46.8 | — |
experiments/exp223/monitor.md:33:| PACI+OA-SD | 39.2/52.5 | +4.9 |
experiments/exp223/monitor.md:52:**ep20: 47.7/59.1** (vs OA-SD 46.0/58.0 = **+1.7/+1.1**)
experiments/exp223/monitor.md:54:| Epoch | PADPQ mAP/R1 | OA-SD mAP/R1 | GSPB mAP/R1 | delta PADPQ |
experiments/exp223/monitor.md:75:**ep30: 52.8/62.9** (vs OA-SD 50.6/61.7 = **+2.2/+1.2**)
experiments/exp223/monitor.md:77:| Epoch | PADPQ | OA-SD | delta |
experiments/exp223/monitor.md:96:**ep40: 55.9/66.7** (vs OA-SD 57.2/69.2 = **-1.3/-2.5!**)
experiments/exp223/monitor.md:98:| Epoch | PADPQ | OA-SD | delta |
experiments/exp223/monitor.md:121:**ep50: 60.3/71.7!** (vs OA-SD 59.0/70.6 = **+1.3/+1.1!**)
experiments/exp223/monitor.md:123:| Epoch | PADPQ | OA-SD | GSPB | delta PADPQ |
experiments/exp223/monitor.md:142:**ep60: 60.7/71.8** (vs OA-SD 60.6/72.9 = +0.1/-1.1)
experiments/exp223/monitor.md:146:| Epoch | PADPQ mAP | OA-SD mAP | delta |
experiments/exp223/monitor.md:156:PADPQ 预计 final ~63.0-63.5 (vs OA-SD 63.2)。
experiments/exp223/monitor.md:164:| Epoch | K=4 | K=8 | OA-SD | delta K=8 vs OA-SD |
experiments/exp223/monitor.md:171:**K=8 在 ep40 领先 OA-SD +0.7!** (K=4 同期 -1.3)
experiments/exp223/monitor.md:182:**K=4 ep70: 61.2/71.7** (vs OA-SD 61.8/73.1 = -0.6/-1.4)
experiments/exp223/monitor.md:184:K=8 still hope — was +0.7 vs OA-SD at ep40 (vs K=4's -1.3).
experiments/exp223/monitor.md:189:K=4 ep~75. K=8 ep50: 59.5 (vs OA-SD 59.0 = +0.5, vs K=4 60.3 = -0.8).
experiments/exp223/monitor.md:190:两个 K 值都在 OA-SD 附近震荡。
experiments/exp223/monitor.md:200:**K=4 ep80: 62.0%** (vs OA-SD 62.0 = **0.0! 完全持平！**)
experiments/exp223/monitor.md:201:PADPQ 跟踪 OA-SD 非常紧密。
experiments/exp223/monitor.md:202:预计 final ~63.0-63.2 (= OA-SD 63.2)。
experiments/exp223/monitor.md:219:**K=4 ep90: 63.1/74.3!** (vs OA-SD 62.4/75.1 = **+0.7/-0.8!**)
experiments/exp223/monitor.md:221:| Epoch | PADPQ | OA-SD | delta |
experiments/exp223/monitor.md:227:**63.1% at ep90 已接近 OA-SD final (63.2)!**
experiments/exp223/monitor.md:228:如果趋势持续，PADPQ final 可能在 `equal_concat mAP` 上超过 OA-SD；但是否能形成综合超越，仍取决于 R1 和后续 `MaxSim`。
experiments/exp223/monitor.md:243:**K=4 ep100: 63.3/73.8!** (vs OA-SD 63.0/75.0 = **+0.3/-1.2**)
experiments/exp223/monitor.md:245:**63.3% 已超过 OA-SD final (63.2)!!**
experiments/exp223/monitor.md:247:| Epoch | PADPQ | OA-SD | delta |
experiments/exp223/monitor.md:252:**到 ep100 为止，PADPQ 首次在 `equal_concat mAP` 上超过 OA-SD final；但当时还不能把它写成综合超越。**
experiments/exp223/monitor.md:263:**K=4 ep110: 63.5/74.2** (vs OA-SD 63.1/75.3 = **+0.4/-1.1**)
experiments/exp223/monitor.md:265:| Epoch | PADPQ | OA-SD | delta |
experiments/exp223/monitor.md:270:**63.5% 超越 OA-SD final (63.2) by +0.3!**
experiments/exp223/monitor.md:272:但 **R1 持续落后 OA-SD ~1.1%**，mAP 微涨不足以弥补 R1 下降。
experiments/exp223/monitor.md:273:总体看 PADPQ ≈ OA-SD，不是真正的突破。
experiments/exp223/monitor.md:283:**exp223 (PADPQ K=4 + OA-SD) FINAL: 63.7/74.5**
experiments/exp223/monitor.md:287:| **exp223 PADPQ+OA-SD** | **63.7%** | 74.5% | 86.2% | 89.5% |
experiments/exp223/monitor.md:288:| exp191 OA-SD-only | 63.2% | **75.4%** | — | — |
experiments/exp223/monitor.md:303:| OA-SD | 63.2/75.4 | 64.2/77.1 | +1.0/+1.7 |
experiments/exp223/monitor.md:307:**PADPQ + MaxSim = 63.9 — 低于 GSPB (64.6) 和 OA-SD (64.2)!**
experiments/exp223/monitor.md:327:- 也没有证据表明 `K=8` 能形成对 `OA-SD final = 63.2/75.4` 的稳定综合超越
experiments/exp141/design.md:8:2. `row_support_mean / row_gap_mean`
experiments/exp141/design.md:13:- 当前这个 pair 的 common-support 改善是普遍现象还是稀有现象
experiments/exp141/design.md:23:3. 当前 pair 的 `support_ratio` 相对排名
experiments/exp141/design.md:28:- 什么时候 common-support correction 值得强用
experiments/exp141/design.md:37:3. `support_rank`
experiments/exp292_target_s_m_s42/design.md:20:- Market pose_data on lab3090 (`/root/work/SOLIDER-REID/data/market1501/pose_data`, 4.3GB, 46635 npz):
experiments/exp139/monitor.md:63:  - `row_support_mean`
experiments/exp139/monitor.md:254:### [2026-03-22 02:00] `exp139` 到 `ep80`：已追平当前最强 supporting 线，继续保持主候选
experiments/exp223/claude_review.md:127:- **exp191 OA-SD** (baseline): Uses `POSE_SKELETON_GCN=True`, `POSE_DEFORMABLE_SAMPLE=False` (default). Unaffected. **OK**
experiments/cargo_cvpb/agreid_dataset.py:146:    def __init__(self, root='/root/work/SOLIDER-REID/data', verbose=True,
experiments/exp141/claude_review_v2.md:53:| `support_ratio` | `support_ratio.detach()` (batch x batch) | `support_ratio[start:end]` (chunk_q x gallery) |
experiments/exp141/claude_review_v2.md:67:| 追加 5 维内容 | `row_mean, row_std, row_min, row_support_mean, row_gap_mean` | `base_rank, kp_rank, support_rank, gain_rank, gain_zscore` |
experiments/exp141/claude_review_v2.md:80:- `kp_dist`: 来自 common-support 距离计算（detached）
experiments/exp141/claude_review_v2.md:81:- `support_ratio`: 来自 keypoint weight 计算（detached）
experiments/exp141/claude_review_v2.md:100:- ascending（距离越小排名越前）用 `inf` 填充 invalid；descending（support_ratio 越大排名越前）用 `-inf` 填充 invalid：正确
experiments/exp141/claude_review_v2.md:115:- `base_dist.detach()`, `kp_dist.detach()`, `support_ratio.detach()` -> 所有 descriptor 输入无梯度
experiments/exp225/design.md:5:- PADPQ K=4 (exp223): **equal_concat 63.7** (+0.5 mAP vs OA-SD), R1 74.5 (-0.9)
experiments/exp225/design.md:13:- 基于 pose_psg_gcn_paa_roa.yml (含 ROA) + OA-SD + PLBOA
experiments/exp225/design.md:17:- exp191 OA-SD-only: 63.2/75.4 (eq), 64.2/77.1 (maxsim)
experiments/exp292_target_s_m_s42/monitor.md:5:- Log: `/tmp/exp292.log` + `/root/work/SOLIDER-REID/log/market1501/exp292_target_s_m_s42/train_log.txt`
experiments/exp292_target_s_m_s42/monitor.md:7:- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + **NO PLBOA** Market default) + 2-stage PSG + target-heatmap
experiments/exp240/design.md:15:- exp206r (Small OA-SD GCN): 70.6/82.6
experiments/exp139/claude_review.md:60:1. **最优方案**：重新设计 context 特征使其不依赖 labels。例如用 row-wise distance statistics（row mean、row std、row min、row max、row support mean）替代 pos/neg 统计。这样训练和测试一致，不需要 labels。
experiments/exp225/claude_review.md:78:- **创新门槛**：这是组合实验，不是新创新。design.md 没有声称是创新，只是验证叠加效果。符合"supporting evidence"角色
experiments/exp225/claude_review.md:82:design.md 说"基于 pose_psg_gcn_paa_roa.yml (含 ROA) + OA-SD + PLBOA"，但没有列出具体配置文件路径或完整的命令行 override。需要确认最终运行命令包含：
experiments/exp265_best_s_op_s42/monitor.md:1:# exp265 monitor — Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-PoseTrack-ReID
experiments/exp265_best_s_op_s42/monitor.md:60:- Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA + default flip-test @ Occ-PTrack-ReID = **78.4 / 86.2**
experiments/exp141/monitor.md:26:     - `support_rank`
experiments/exp141/monitor.md:85:     - `exp142`: feature-space support-supervised completion
experiments/exp141/monitor.md:90:  2. 已从本地精确同步以下文件到远程 `/root/work/SOLIDER-REID`：
experiments/exp225/monitor.md:3:配置: 基于 pose_psg_gcn_paa_roa.yml + OA-SD + PLBOA + GSPB(0.05) + PADPQ(K=4)
experiments/exp225/monitor.md:4:对照: exp220 GSPB(maxsim 64.6), exp223 PADPQ(eq 63.7), exp191 OA-SD(63.2/75.4)
experiments/exp225/monitor.md:14:**ep10: 38.3%** (vs GSPB 40.1, PADPQ 37.5, OA-SD 34.3)
experiments/exp225/monitor.md:20:**ep20: 49.4%** (vs GSPB 49.1, PADPQ 47.7, OA-SD 46.0)
experiments/exp225/monitor.md:27:ep70: 62.3 (+0.5 vs OA-SD), ep80: 62.8 (+0.8 vs OA-SD) — 稳定正向!
experiments/exp225/monitor.md:32:**ep90: 63.6%** (vs OA-SD 62.4 = **+1.2!**)
experiments/exp225/monitor.md:34:| Epoch | GSPB+PADPQ | OA-SD | delta |
experiments/exp225/monitor.md:41:ep90 已超过 OA-SD final (63.2) by +0.4。
experiments/exp225/monitor.md:47:**ep100: 63.9%** (vs OA-SD 63.0 = +0.9)
experiments/exp225/monitor.md:49:| Epoch | GSPB+PADPQ | OA-SD | delta |
experiments/exp225/monitor.md:55:63.9% 已超过 OA-SD final (63.2) by +0.7。
experiments/exp225/monitor.md:67:**ep110: 64.1/74.6** (vs OA-SD 63.1/75.3 = +1.0/-0.7)
experiments/exp225/monitor.md:79:**exp225 (GSPB scale=0.05 + PADPQ K=4 + OA-SD) FINAL: 64.2/74.9**
experiments/exp225/monitor.md:84:| exp191 OA-SD-only | 63.2% | **75.4%** | — | — |
experiments/exp225/monitor.md:87:**mAP +1.0, R1 -0.5 vs OA-SD-only。**
experiments/exp225/monitor.md:88:mAP 在 Tiny 上首次显著超过 OA-SD by +1.0%！
experiments/exp225/monitor.md:93:| Epoch | GSPB+PADPQ | OA-SD | delta mAP |
experiments/exp225/monitor.md:110:| OA-SD | 63.2/75.4 | 64.2/77.1 | +1.0/+1.7 |
experiments/exp225/monitor.md:116:**但 GSPB+PADPQ 是 equal_concat 最佳 (64.2, +1.0 vs OA-SD)。**
experiments/exp240/monitor.md:1:# exp240 Small + PPA (w=0.5) + OA-SD 监控
experiments/exp240/monitor.md:3:配置: Small + PSG + PPA (w=0.5) + OA-SD + PLBOA(0.7) + 无 ROA
experiments/exp240/monitor.md:4:对照: exp206r (Small OA-SD GCN): 70.6/82.6
experiments/exp292_target_s_m_s42/claude_review.md:59:cd /root/work/SOLIDER-REID
experiments/exp240/claude_review.md:21:对照组 exp206r (Small OA-SD GCN): 70.6/82.6 是正确的 baseline。
experiments/exp240/claude_review.md:38:| PRETRAIN_PATH | pretrained/swin_tiny.pth | pretrained/swin_small.pth | 已确认文件存在 (1.1GB) |
experiments/exp240/claude_review.md:95:exp206r 使用: GCN + PAA + OA-SD + PLBOA + **ROA**
experiments/exp240/claude_review.md:96:exp240 使用: PPA + PAA + OA-SD + PLBOA + **无 ROA**
experiments/exp240/claude_review.md:100:更公平的对照应该是找一个 Small + OA-SD + 无 ROA 的实验。
experiments/exp240/claude_review.md:134:- 单变量: 只有 backbone Tiny→Small (+ 对应的 LR 和 pretrained 调整)
experiments/exp265_best_s_op_s42/claude_review.md:12:H1. Swin-Base + WITH_CP + POSE_PARALLEL_AUG (4 views) + OA-SD on 5060Ti 16G：Base 首 epoch OOM 风险。exp260b 历史 3090 运行 OK，但 5060Ti 16G 需要监控。若 OOM：关 OA-SD 或 PARALLEL_AUG，不改 BS。
experiments/exp265_best_s_op_s42/claude_review.md:22:M3. Market 的 OA-SD 因为没 PLBOA，teacher/student view 1 几乎相同；蒸馏信号弱但不出错。可以接受作为一致性。
experiments/exp265_best_s_op_s42/claude_review.md:28:L1-L8. OA-SD 参数跨数据集统一（可接受）；PLBOA 在 Occ-PTrack 开（数据已 occluded 但一致性保持）；POSE_PFM_ENABLED 默认 False 即使 POSE_PFM_HIDDEN=64 存在（无 bug，命名小瑕疵）；DEVICE_ID=('0')、BIAS_LR_FACTOR=2、WARMUP_EPOCHS=20 都与 exp255 scaffold 一致。
experiments/exp141/claude_review.md:62:- exp139 (`query_ctx`)：追加 5 维 **query 级常量**（`row_mean, row_std, row_min, row_support_mean, row_gap_mean`），同一 query 内所有 pair 共享同一组 context 值
experiments/exp141/claude_review.md:63:- exp141 (`comp_ctx`)：追加 5 维 **pair-specific 相对竞争位置**（`base_rank, kp_rank, support_rank, gain_rank, gain_zscore`），同一 query 内每个 pair 的 context 值不同
experiments/exp141/claude_review.md:81:**是的，无泄漏。** `build_query_competition_descriptors` 的所有输入（`base_dist, kp_dist, support_ratio`）均来自特征距离计算，不涉及任何标签。排名和 z-score 也都是纯统计量。
experiments/cargo_cvpb/cvpb_osac_mk.log:18:/root/work/SOLIDER-REID/model/backbones/swin_transformer.py:1193: UserWarning: DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead
experiments/cargo_cvpb/cvpb_osac_mk.log:19:  warnings.warn('DeprecationWarning: pretrained is deprecated, '
experiments/cargo_cvpb/cvpb_osac_mk.log:26:Loading pretrained model from /root/work/SOLIDER-REID/log/market1501/exp260b_base_gcn512_2stage/transformer_20.pth
experiments/cargo_cvpb/cvpb_osac_mk.log:57:Loading pretrained model from /root/work/SOLIDER-REID/log/market1501/exp260b_base_gcn512_2stage/transformer_40.pth
experiments/cargo_cvpb/cvpb_osac_mk.log:88:Loading pretrained model from /root/work/SOLIDER-REID/log/market1501/exp260b_base_gcn512_2stage/transformer_60.pth
experiments/cargo_cvpb/cvpb_osac_mk.log:119:Loading pretrained model from /root/work/SOLIDER-REID/log/market1501/exp260b_base_gcn512_2stage/transformer_80.pth
experiments/cargo_cvpb/cvpb_osac_mk.log:150:Loading pretrained model from /root/work/SOLIDER-REID/log/market1501/exp260b_base_gcn512_2stage/transformer_100.pth
experiments/cargo_cvpb/cvpb_osac_mk.log:181:Loading pretrained model from /root/work/SOLIDER-REID/log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth
experiments/clip_reid_compare/CODEX_BRIEF.md:4:- 在 SOLIDER/Swin ReID backbone 上加一个 CLIP 部位分支:
experiments/exp349b_small_undetach_clip/codex_review.md:7:codex 审查通过。clip_id_loss 经 LGPA+GCN dual 分支正确透传、processor 只加一次(parallel_aug+OA-SD 4-view 下仅 view0 kp_data 被用、loss_func 不读 clip_id_loss,无双计数);Swin-Small in_planes=768 = clip_id_proj 输入 = ViT-L clip_dim=768 维度匹配;OA-SD EMA teacher deepcopy 冻结 CLIP 安全;两次 CLIP 加载(LGPA ViT-B-32 buffer + clip_id ViT-L-14 submodule)无 state_dict key 冲突;GLOBAL_LOSS_SCALE 0.5 全系统 list-path 一致(M1);单变量 vs exp255。run349.sh 含 TEST.IMS_PER_BATCH 64 防 OOM。Verdict: approve。
experiments/exp327_dinov3/codex_review.md:1:# Codex Review — exp326 (DIFT) + exp327 (DINOv3)
experiments/exp327_dinov3/codex_review.md:11:- **Medium — exp327 token assert 不够严**：原 `assert patch_tok.shape[1] == n_tok` 只查切片长度，若 `nreg` 读错仍可能切出 n_tok 个 token 但混入 register、丢尾部 patch。HF DINOv3 文档要求先 assert 完整布局 `[CLS, registers, patches]` 再切。
experiments/exp327_dinov3/codex_review.md:25:DIFT "SD 特征在语义对应上超 DINO/OpenCLIP" 前提对 SPair/PCK 成立（project page 报 +19 vs DINO / +14 vs OpenCLIP，含遮挡/姿态变体）。但 SD-DINO / Tale-of-Two-Features 把 SD 与 DINO 视作**互补**，并非对每个下游 retrieval 指标都 SD 主导 → 与 design.md 预期一致，DIFT 未必必胜，故训练-free 首验有必要。
experiments/exp253/design.md:16:- 其余与 exp246b 相同：Tiny LGPA-D+GCN+OA-SD+PLBOA
experiments/exp086/design.md:1:# 实验 exp086: Pose-Aware Parallel Augmentation Training (PA-PAT)
experiments/exp349b_small_undetach_clip/claude_review.md:3:**审查范围**：configs/occluded_duke/exp349_small_full_clip.yml、model/pose_backbone_model.py（forward 多分支 + clip_id_loss 注入 line 733/906）、model/modules/clip_id_prompt.py、processor/processor.py（clip_id_loss 消费 line 1297、parallel_aug line 790、OA-SD EMA line 470/801/1317）、config/defaults.py、loss/make_loss.py、model/make_model.py（in_planes）、model/modules/clip_part_head.py（LGPA CLIP 加载）。
experiments/exp349b_small_undetach_clip/claude_review.md:48:### 5. OA-SD EMA teacher 与 CLIP prompt —— 安全（含一处可接受的内存代价） ✓ / Low
experiments/exp349b_small_undetach_clip/claude_review.md:58:PLBOA 是数据增强（下半身遮挡），作用在输入图像/student view 上，OA-SD 用 clean（pre-PLBOA）图像与 pose 喂 teacher（line 819-824 `teacher_pose = pose_dict.get('teacher_pose', pose_dict)`，line 557 `img_teacher = img[3]` 为 clean view）。CLIP prompt 对齐的是 student 的 global_feat（带 PLBOA 遮挡）到 per-ID 文本原型 —— 这反而是「让被遮挡图也对齐纯 ID 语义」，与 PLBOA 目标方向一致、不冲突。✓ 纯增强，与 CLIP loss 计算路径无耦合。
experiments/exp349b_small_undetach_clip/claude_review.md:98:逐行核对了 clip_id_loss 在 LGPA+GCN dual 分支的注入与透传（model line 733/747-749 → processor line 1297）、parallel_aug 4-view 下的单次计数（view0 only + loss_fn 不读 clip_id）、Swin-Small in_planes=768 与 clip_id_proj 维度匹配、OA-SD EMA 深拷贝对冻结 CLIP 的恒等性、两次 CLIP 加载（ViT-B-32 buffer vs ViT-L-14 子模块）无 key 冲突、PLBOA 正交、单变量隔离（仅 POSE_CLIP_ID_PROMPT）。未发现 Critical/High 阻断项。唯一硬性提醒：**启动训练命令必须 `TEST.IMS_PER_BATCH 64` override**（流程铁律，防 eval OOM）。其余为 Low 级继承自 exp341 的已验证项。
experiments/exp253/monitor.md:3:配置: Swin-Tiny + PSG Stage1+2+3 (无 PAA) + LGPA-D (detach) + GCN (detach) + OA-SD + PLBOA(0.7) + WITH_CP
experiments/exp086/monitor.md:1:# exp086 PA-PAT (Parallel Augmentation Training) 监控
experiments/exp262_best_s_od_s42/monitor.md:1:# exp262 monitor — Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-Duke
experiments/exp262_best_s_od_s42/monitor.md:63:- 远远超过 BPBreID (62.5/75.1) 和 SOLIDER (61.9/71.2)
experiments/exp262_best_s_od_s42/monitor.md:69:- Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA + default flip-test @ Occ-Duke = **73.8 / 83.1**
experiments/exp349b_small_undetach_clip/design.md:4:最现实的组合模型 / 论文交付:用户最强 pose 系统 exp255(Swin-Small + 2-stage PSG + LGPA + GCN512 + OA-SD + PLBOA = 73.2/83.3)+ CLIP-ReID prompt(Swin-Tiny 上 +2.2)。问:CLIP 能否给这个强 pose 系统再加涨 → 一个 CLIP+pose 都有的更强模型。
experiments/exp349b_small_undetach_clip/design.md:16:exp349 equal_concat > exp255 73.2。失败可能:CLIP 与 OA-SD 自蒸馏/PLBOA 多 loss 冲突;global 0.5x 稀释 CLIP 增益;强系统已饱和(冗余)。
experiments/exp349b_small_undetach_clip/design.md:22:CLIP prompt 与全系统(PSG/LGPA/GCN/OA-SD/PLBOA)多分支多 loss 共存无冲突;clip_id_loss 经 dual 分支正确回传不重复;Swin-Small backbone 下 clip_id_proj 维度(in_planes)对;OA-SD 的 EMA teacher 不受 CLIP 干扰;单变量 vs exp255;Swin-Small 显存(+CLIP ViT-L)够。
experiments/exp327_dinov3/design.md:1:# 实验 exp327: DINOv3 / DINOv2-with-registers pose-anchored part-MaxSim 训练-free 天花板 check
experiments/exp327_dinov3/design.md:3:> **来源**：post-PRCV「搬范式」路线，exp324（frozen DINOv2-base）的更强/更新特征源探针。
experiments/exp327_dinov3/design.md:5:> **决定性问题**：更新/更强的冻结对应模型（DINOv3-B / DINOv2-with-registers-B）训练-free 重遮挡 pose-part mAP 能否抬过 exp324 DINOv2-base 的 **1.86**？
experiments/exp327_dinov3/design.md:9:- exp324 用 frozen DINOv2-base 验证了机制（pose 锚定 dense token + mutually-visible MaxSim），但天花板低（重遮挡 1.86，训头 e20 到顶 14）。
experiments/exp327_dinov3/design.md:10:- DINOv3（2025）与 DINOv2-with-registers 都改进了 dense feature 质量：registers 去掉了 ViT 的 high-norm artifact token（更干净的 patch 特征），DINOv3 用更大数据 + Gram anchoring 进一步提纯 dense 特征。**若更干净的 dense 对应能直接抬训练-free 重遮挡 mAP，则更强冻结源的天花板更高，值得优先上头。**
experiments/exp327_dinov3/design.md:11:- 与 exp326（SD-DIFT）并行，从「更新 DINO 系」与「换范式（SD）」两条腿同时探天花板。
experiments/exp327_dinov3/design.md:15:把 exp324 的特征源从 DINOv2-base 换成 DINOv3-vitb16 或 DINOv2-with-registers-base（其余 pipeline 完全不变），训练-free 重遮挡 pose-part mAP **超过 1.86**。
experiments/exp327_dinov3/design.md:19:1. **特征源**：`AutoModel.from_pretrained`，frozen，`--model ∈ {dinov3-b, dinov2reg-b, dinov2-b}`。
experiments/exp327_dinov3/design.md:20:   - dinov3-b: `facebook/dinov3-vitb16-pretrain-lvd1689m`（patch 16, hidden 768）
experiments/exp327_dinov3/design.md:33:- 主跑 dinov3-b（最新最强）；如 hf-mirror / transformers 版本对 DINOv3 不兼容，回退 dinov2reg-b。
experiments/exp327_dinov3/design.md:37:- 成立：DINOv3/registers 重遮挡 pose-part mAP > 1.86，pose>grid → 更强冻结源天花板更高 → 优先上头。
experiments/exp327_dinov3/design.md:38:- 失败最可能：(1) 更干净 dense 特征在 ReID 零样本判别性上仍弱（PCK 强 ≠ ReID 强）；(2) DINOv3 patch16 在 224 输入 grid 更粗，空间分辨率降低抵消特征提纯；(3) transformers 对 DINOv3 输出格式不一致导致 token 切片错位（已加 assert 拦截）。
experiments/exp327_dinov3/design.md:42:- baseline = exp324 frozen DINOv2-base 重遮挡 pose-part 1.86。唯一变量 = **特征源**（DINOv2-base → DINOv3-B / registers-B）。
experiments/exp327_dinov3/design.md:48:- 重遮挡 > 1.86 且 pose>grid → exp327b：该更强冻结源上轻量 part 头，天花板 check vs DINOv2-base 头的 14。
experiments/exp294_lgpaOnly_2stg_b_od_s41/codex_review.md:33:本 exp 仅 backbone 从 Small → Base, PSG/LGPA/OA-SD/ParAug/PLBOA/数据 pipeline 完全一致。
experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/innovation_brainstorm.md:392:1. **exp148 PCVT 早期加速但后期无效**: complementary masking 在前期提供多样性，但 backbone 最终收敛到的表示已经隐式假设"几乎全可见"
experiments/innovation_brainstorm.md:404:## 2026-03-20: `exp109` 主线的下一阶段机制收束
experiments/innovation_brainstorm.md:408:经过 `exp110-126`，当前最重要的收束不是“support-complete 有没有价值”，而是：
experiments/innovation_brainstorm.md:410:1. `exp109` 的 oracle 上界仍然极强，问题定义成立
experiments/innovation_brainstorm.md:415:因此，`exp109` 这条线最自然的下一跳不是再扫权重，而是：
experiments/innovation_brainstorm.md:417:**让 support-complete prototype 以“可学习残差 prior”的形式进入 keypoint branch。**
experiments/innovation_brainstorm.md:433:  1. 问题层面仍锚定 `single-image support incomplete`
experiments/innovation_brainstorm.md:434:  2. 机制层面从 “memory bank / routing trick” 升级成了真正的 **support-conditioned completion**
experiments/innovation_brainstorm.md:444:- 这说明 common-support 不是噪声，而是真实的 pairwise 证据
experiments/innovation_brainstorm.md:462:1. 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
experiments/innovation_brainstorm.md:463:2. 在 global branch 上增加一条 support-aware triplet
experiments/innovation_brainstorm.md:481:**核心教训**: 把 retrieval-time 的 common-support 信号迁到训练端，不能简单用 overlap 做 mining filter。retrieval-time CVK 有效是因为它改变了距离计算方式（只在共同可见关键点上计算距离），而不是因为它筛选了更好的 pair。
experiments/innovation_brainstorm.md:485:2. 如果要做训练端 common-support，必须改变 loss 本身的距离计算（如只在共同可见区域上计算 triplet 距离）
experiments/innovation_brainstorm.md:858:   **global identity space + balanced common-support correction**
experiments/innovation_brainstorm.md:885:   - **CVK 主要做 deeper-rank common-support correction**
experiments/innovation_brainstorm.md:1498:4. 若首轮无正信号，立即止损，回退到 retrieval-time `common-support recovery`
experiments/innovation_brainstorm.md:1569:   **在 per-keypoint / common-visible support 层面做 duplicate-aware confuser reasoning**。
experiments/innovation_brainstorm.md:1577:  3. per-keypoint / common-support 粒度
experiments/innovation_brainstorm.md:1589:  - 就连 `per-keypoint / common-support` 层面的 test-time confuser penalty 也不稳定
experiments/innovation_brainstorm.md:1603:## 2026-03-19: exp109 Oracle Support Bank 后的新主线收敛
experiments/innovation_brainstorm.md:1606:- `exp109` 用 GT same-ID per-keypoint prototype 做 leave-one-out oracle recovery：
experiments/innovation_brainstorm.md:1611:  - “support-complete latent representation” 的 headroom 非常大
experiments/innovation_brainstorm.md:1616:   **模型没有学会从单图中逼近完整 identity support。**
experiments/innovation_brainstorm.md:1618:   - support 来源太弱
experiments/innovation_brainstorm.md:1622:   **same-ID support bank → single-image support-complete distillation**
experiments/innovation_brainstorm.md:1639:- 虽然幅度不大，但这和 `exp109` 的 oracle headroom 连起来后，意义很明确：
experiments/innovation_brainstorm.md:1640:  - `support-complete` 不是只存在于上界分析里的幻觉
experiments/innovation_brainstorm.md:1644:1. 当前最值得继续赌的，不再是“有没有必要做 support-complete”，而是：
experiments/innovation_brainstorm.md:1645:   **怎样让 prototype teacher 更可靠、更接近真正的 multi-view support。**
experiments/innovation_brainstorm.md:1650:3. 因此下一步不应直接堆 decoder / completion block，而应先做：
experiments/innovation_brainstorm.md:1651:   **reliable-support bank / teacher reliability gating**
experiments/innovation_brainstorm.md:1655:  1. 问题不是简单 occlusion comparison，而是 single-image support incomplete
experiments/innovation_brainstorm.md:1656:  2. 方法不是通用补全 decoder，而是 identity-level support-complete distillation
experiments/innovation_brainstorm.md:1666:- 结果几乎等价，说明“要求多个 support 样本共同支撑 teacher”这件事本身，并没有把当前增益显著放大。
experiments/innovation_brainstorm.md:1669:1. 当前 `support-complete` 主线并没有被否定，因为结果仍保持正向区间。
experiments/innovation_brainstorm.md:1672:   **teacher purity / write quality / support cleanliness**
experiments/innovation_brainstorm.md:1676:- 基于 support 置信度的 soft reliability weighting
experiments/innovation_brainstorm.md:1683:- `exp112` 说明更干净的 support 写入有用，但当前只形成弱正向：
experiments/innovation_brainstorm.md:1691:1. 当前最值得讲的主创新，已经不只是 “support-complete distillation”。
experiments/innovation_brainstorm.md:1693:   **如何在 pose-aligned support-complete learning 中控制 teacher hardening / non-stationary target。**
experiments/innovation_brainstorm.md:1701:- Lagged / stale support bank
experiments/innovation_brainstorm.md:1734:oracle experiment (exp109) 给出 +8.5% mAP 的 headroom，但 SCKD 只能捕获 1%。核心原因：
experiments/innovation_brainstorm.md:1802:  1. `cvk_hybrid` 说明 common-support 的 pairwise 几何是真实的
experiments/innovation_brainstorm.md:1805:  4. `exp109-116` 说明 `support-complete` 若被压成 `per-ID prototype`，会损失 pair-specific 细节
experiments/innovation_brainstorm.md:1812:- 不再把 support 压成 prototype
experiments/innovation_brainstorm.md:1826:2. 机制层面：pose/keypoint branch 作为 **common-support relational teacher**
experiments/innovation_brainstorm.md:1827:3. 训练目标：把 global embedding 蒸馏成更符合 common-support 几何的空间
experiments/innovation_brainstorm.md:1828:4. 证据层面：可直接和 `exp047 / exp051 / exp109-116` 构成一条非常清晰的对照链
experiments/innovation_brainstorm.md:1840:   **teacher 自身还是单图 `kp_feats`，并不 support-complete**
experiments/innovation_brainstorm.md:1849:3. 而是先用 `exp109` 方向的 support bank 补全 low-vis keypoint teacher，再用补全后的 teacher 去做 `CSRD`
experiments/innovation_brainstorm.md:1852:1. `exp109` 已证明 support-complete teacher 有巨大 headroom
experiments/innovation_brainstorm.md:1856:   **support-complete teacher + relational distillation**
experiments/innovation_brainstorm.md:1861:- 但这次不能简单说 `support-complete teacher` 失败，因为机制统计很清楚：
experiments/innovation_brainstorm.md:1867:1. `teacher completion` 已经真实发生
experiments/innovation_brainstorm.md:1870:   **support-complete 监督的收益主要属于 support-incomplete 样本，被 clean 样本等权平均后稀释掉了**
experiments/innovation_brainstorm.md:1877:1. 保持 `exp120` 的 support-complete relational teacher 完全不变
experiments/innovation_brainstorm.md:1880:   - 它有多少 keypoint 真正被 support-complete teacher 补全
experiments/innovation_brainstorm.md:1884:1. `exp109` 的 headroom 本来就主要集中在低可见样本
experiments/innovation_brainstorm.md:1903:   **support-complete teacher 实际只改变了一部分 pairwise 关系，distillation 应聚焦这些 pair-change relations**
experiments/innovation_brainstorm.md:1910:1. 保持 `exp120` 的 support-complete teacher 完全不变
experiments/innovation_brainstorm.md:1914:   - support-complete teacher 几何
experiments/innovation_brainstorm.md:1915:4. 对那些 **被 support completion 真正改变过的 pair** 赋予更高 distillation focus
experiments/innovation_brainstorm.md:1919:2. `exp109` 的 headroom 本质上是 pairwise comparability 被修正
experiments/innovation_brainstorm.md:1927:  **teacher stability = supporting mechanism**
experiments/innovation_brainstorm.md:1949:1. `exp121` 已说明 freeze 只是 supporting，不值得再扩成一条线
experiments/innovation_brainstorm.md:1979:1. 保持 `exp123/124` 的 support-complete relational teacher 完全不变
experiments/innovation_brainstorm.md:1993:- 同时其 gate 几乎塌到 `1.0`，说明 per-ID prototype 的 direct feature completion 这条兑现线暂时可以收住
experiments/innovation_brainstorm.md:1995:  **stable teacher 只是 supporting mechanism，不再值得单独扩线**
experiments/innovation_brainstorm.md:1998:1. `support-complete teacher` 的新增信息是真实存在的
experiments/innovation_brainstorm.md:2001:4. 于是 support-complete 带来的那部分新增 correction，极可能被 base teacher 的主体结构稀释掉
experiments/innovation_brainstorm.md:2012:4. 让 global embedding 学习的不是“再复刻一遍 skeleton teacher”，而是只学 **support completion 真正带来的关系修正**
experiments/innovation_brainstorm.md:2041:1. 保留 `exp125` 当前最强的 online support teacher 与 `delta_top` routing
experiments/innovation_brainstorm.md:2053:   **让 student 在更大的 relation support 上学习 support-complete comparability correction**
experiments/innovation_brainstorm.md:2084:1. 不再强迫单个 global embedding 吃下 support-complete correction
experiments/innovation_brainstorm.md:2088:   - 该在多大程度上相信 common-support distance
experiments/innovation_brainstorm.md:2090:   - 用 `support-complete teacher` 提供更理想的 pairwise target
experiments/innovation_brainstorm.md:2131:   - 必要时再加入更细的 keypoint-wise common-support statistics
experiments/innovation_brainstorm.md:2166:### 2026-03-21 晚间更新：`LPCS` 已经真正成立，但 sparse routing 最终只是 supporting 机制
experiments/innovation_brainstorm.md:2199:- `pose-defined common support`
experiments/innovation_brainstorm.md:2238:   - 方案：给每个 pair descriptor 追加 query 的正负均值距离、margin、support 完整度与 teacher change 统计
experiments/innovation_brainstorm.md:2241:- `pose-defined common support`
experiments/innovation_brainstorm.md:2266:### 2026-03-22 当前收敛：`rank-decay` 退为 supporting，`query-context correction` 升为主候选
experiments/innovation_brainstorm.md:2271:  - 它证明了“平滑 top-sensitive”比 `hard-rank` 合理，但最终只形成 supporting 级别的改进
experiments/innovation_brainstorm.md:2283:3. 从而让同样的 common support 在不同 query 上被不同地解释
experiments/innovation_brainstorm.md:2307:1. pose 定义 common support
experiments/innovation_brainstorm.md:2308:2. support-complete teacher 提供 correction 方向
experiments/innovation_brainstorm.md:2322:     - 同一份 common support，是否需要放在 query-level 语境里解释
experiments/innovation_brainstorm.md:2351:3. `support_rank`
experiments/innovation_brainstorm.md:2357:- pose 定义 common support
experiments/innovation_brainstorm.md:2362:- 都还紧扣 `exp109` 的核心发现：单图 support 不完整
experiments/innovation_brainstorm.md:2366:- `exp139` 强调 **如何解释 common support**
experiments/innovation_brainstorm.md:2369:### 2026-03-21 本地大转向：从 pair correction 切回 feature-space support completion
experiments/innovation_brainstorm.md:2394:1. pose 不再只是用来构造 `common support distance`
experiments/innovation_brainstorm.md:2398:   - 哪些 support prototype 可作为跨图补全证据
experiments/innovation_brainstorm.md:2399:   - 哪些 skeleton 邻接关系约束 completion
experiments/innovation_brainstorm.md:2403:- `exp109` 暴露出的单图 support incomplete，能否在编码阶段被修复
experiments/innovation_brainstorm.md:2409:## 2026-03-22: feature-level completion 方向彻底证伪，转入注意力 inductive bias
experiments/innovation_brainstorm.md:2415:这是 feature-level completion 方向的第 5+ 次尝试，全部失败：
experiments/innovation_brainstorm.md:2423:1. 15K 数据集无法学习复杂的 completion 函数
experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/innovation_brainstorm.md:2425:3. completion module 的 gate 无法学会"该修改多少"——要么太保守（不起作用），要么太激进（破坏特征）
experiments/innovation_brainstorm.md:2446:6. Feature-level completion (SGMKC, CIPGFR, TTSFR, LSRM, SKC)
experiments/innovation_brainstorm.md:2452:## 2026-03-22 上午方向重置：从“小修 scorer / completion”转向两条更大的新机制
experiments/innovation_brainstorm.md:2457:2. feature-level completion 也不是“没接上”，而是反复接上后依然不成立
experiments/innovation_brainstorm.md:2467:### 候选 A: PCVT（Pose-Complementary View Training）
experiments/innovation_brainstorm.md:2470:- 单图 support incomplete 能否通过 **pose-defined complementary pseudo-views** 改写成“伪多 support 学习”？
experiments/innovation_brainstorm.md:2481:- PCVT 直接改写训练对象，把单图变成“互补 support 组合体”
experiments/innovation_brainstorm.md:2493:为什么比 direct completion 更合理：
experiments/innovation_brainstorm.md:2502:- feature completion 小变体
experiments/innovation_brainstorm.md:2507:## 2026-03-22 下午更新：`PCVT` 与 `SCFA` 已出现第一轮明显分化
experiments/innovation_brainstorm.md:2509:### `PCVT` 的当前价值
experiments/innovation_brainstorm.md:2511:`PCVT` 现在是少数真正跑出连续验证正信号的新方向：
experiments/innovation_brainstorm.md:2522:2. `single-image support incomplete` 可能确实更适合被改写成“伪多 support 学习对象”，而不是继续做 scorer / completion 小修补
experiments/innovation_brainstorm.md:2539:   - `PCVT` 这类直接改写训练对象的方向
experiments/innovation_brainstorm.md:2543:   - 即便 `PCVT` 成立，也必须证明它不是“更复杂的数据增广/一致性 recipe”
experiments/innovation_brainstorm.md:2549:- KPR+SOLIDER (ECCV24): ~82/73 (Swin-L, keypoint prompt at test time)
experiments/innovation_brainstorm.md:2658:## 2026-03-26: OA-SD 系列实验总结 + Global-Only 新发现
experiments/innovation_brainstorm.md:2660:### OA-SD 核心特性（exp191-194 消融）
experiments/innovation_brainstorm.md:2662:1. **OA-SD + CE = 强正向**: +2.9/+2.6 vs CE base (exp191)
experiments/innovation_brainstorm.md:2663:2. **OA-SD + SupCon (all-token) = 负向**: -0.7/-0.4 (exp188) — 梯度冲突
experiments/innovation_brainstorm.md:2666:5. **OA-SD + 3-view 是 additive**: exp193 = 64.4/76.5 vs exp190 = 64.2/75.6 (+0.2/+0.9)
experiments/innovation_brainstorm.md:2667:6. **OA-SD late-stage boost**: ep40 前拖累 → ep40 crossover → ep60+ 大幅正向
experiments/innovation_brainstorm.md:2669:### 关键新发现：OA-SD Global-Only 解决 SupCon 梯度冲突
experiments/innovation_brainstorm.md:2671:**问题**: OA-SD all-token distillation 与 SupCon 在 per-token features 上产生梯度冲突
experiments/innovation_brainstorm.md:2674:- OA-SD distillation 鼓励 student tokens 逼近 teacher tokens（不管 ID）
experiments/innovation_brainstorm.md:2677:**解决方案**: OA-SD GLOBAL_ONLY — 只在 global (GAP后) feature 上做 distillation
experiments/innovation_brainstorm.md:2678:- Global feature: CE + triplet + OA-SD distill（三者协同）
experiments/innovation_brainstorm.md:2679:- Per-token features: CE + triplet + SupCon（三者协同，无 OA-SD 干扰）
experiments/innovation_brainstorm.md:2681:**验证** (exp195): SupCon + OA-SD global-only ep70=60.2/73.4
experiments/innovation_brainstorm.md:2683:- R1 稳定领先 CE+OA-SD（SupCon 的 R1 优势保持）
experiments/innovation_brainstorm.md:2687:- 全局遮挡不变性 (OA-SD) 和局部判别力 (SupCon) 在不同特征级别独立优化
experiments/innovation_brainstorm.md:2695:| 2 | exp193 | 3-view + OA-SD + CE | 64.4% | 76.5% |
experiments/innovation_brainstorm.md:2698:| 5 | exp194 | OA-SD + CE (w=2.0) | 63.4% | 74.8% |
experiments/innovation_brainstorm.md:2699:| 6 | exp191 | OA-SD + CE | 63.2% | 75.4% |
experiments/innovation_brainstorm.md:2704:3-view + SupCon + OA-SD global-only — 预计 65.0-65.5/77.0-77.5
experiments/innovation_brainstorm.md:2722:- **直接解决 OA-SD vs SupCon 冲突**: 不 match 个体特征(会冲突) → match 关系结构(不冲突)
experiments/innovation_brainstorm.md:2737:- OA-SD + SupCon: 互斥，即使 global-only 也无法叠加 (exp195/196)
experiments/innovation_brainstorm.md:2754:| exp188 | OA-SD all-token + SupCon | SupCon | -0.7/-0.4 | ❌ 梯度冲突 |
experiments/innovation_brainstorm.md:2755:| exp195 | OA-SD global-only + SupCon | SupCon | ~-2.8 mAP | ❌ 信号太弱 |
experiments/innovation_brainstorm.md:2756:| exp196 | OA-SD global-only + SupCon + 3v | SupCon | -2.5/-1.4 | ❌ 同上 |
experiments/innovation_brainstorm.md:2758:| exp191 | OA-SD all-token + CE | CE | +2.9/+2.6 | ✅ CE 兼容 |
experiments/innovation_brainstorm.md:2759:| exp193 | OA-SD all-token + CE + 3v | CE | +0.2/+0.9 | ✅ CE 兼容 |
experiments/innovation_brainstorm.md:2760:| exp200 | OA-RD relational + CE | CE | ~-1.0/-3.4 | ❌ OA-RD 不如 OA-SD |
experiments/innovation_brainstorm.md:2765:2. **OA-SD (feature distillation) 在 CE 路线有效**，但 OA-RD (relational) 在 CE 路线不如 OA-SD
experiments/innovation_brainstorm.md:2768:   - OA-SD 路线: exp193 = 64.4/76.5 (不加 SupCon)
experiments/innovation_brainstorm.md:2774:1. **BMKCA** — batch-mate cross-attention part completion
experiments/innovation_brainstorm.md:2822:1. **Swin-Small + GCN+PAA+OA-SD + maxsim_hybrid = 72.4%** (exp210b with PKC=0.05)
experiments/innovation_brainstorm.md:2823:2. 训练端改进: 目前所有尝试均未超过 OA-SD-only ceiling
experiments/innovation_brainstorm.md:2834:| 实验 | 方法 | detach? | vs OA-SD-only |
experiments/innovation_brainstorm.md:2843:| exp219 | PACI without OA-SD | detached | 远程日志当前只确认到 `ep30=51.9/64.9`，早期即落后 baseline `52.2/65.2` |
experiments/innovation_brainstorm.md:2853:早期只看 `OA-SD / OERL / PACI` 三条 Tiny 线时，`maxsim_hybrid` 确实都落在 `64.1~64.3`；
experiments/innovation_brainstorm.md:2858:| OA-SD-only | 63.2 | 64.2 |
experiments/innovation_brainstorm.md:2859:| OERL+OA-SD | 62.2 | 64.3 |
experiments/innovation_brainstorm.md:2860:| PACI+OA-SD | 61.9 | 64.1 |
experiments/innovation_brainstorm.md:2861:| GSPB+OA-SD | 62.9 | 64.6 |
experiments/innovation_brainstorm.md:2862:| PADPQ+OA-SD | 63.7 | 63.9 |
experiments/innovation_brainstorm.md:2865:1. `MaxSim` 对 OA-SD 本身仍然有效（`63.2 -> 64.2`）
experiments/innovation_brainstorm.md:2873:| Epoch | GSPB mAP | OA-SD mAP | delta |
experiments/innovation_brainstorm.md:2885:**GCN+PAA+OA-SD 在 Tiny 上已达极限 ~63%。** 需要:
experiments/innovation_brainstorm.md:2939:| Backbone | ViT + SOLIDER pre-training | Swin + SOLIDER pre-training |
experiments/innovation_brainstorm.md:2974:- FSDC: detached 上 feature completion → 无效
experiments/innovation_brainstorm.md:2978:**唯一有效的方向是改变 backbone 本身**: PSG (+1.7%), OA-SD (+2-3%), PLBOA (+1.5%)
experiments/innovation_brainstorm.md:2996:KPR 证明这个 exact mechanism 在 SOLIDER+Swin 上有效 (75.1%)。
experiments/innovation_brainstorm.md:3000:PPA 替换 GCN Part branch，但保留 PSG + OA-SD + PLBOA。
experiments/innovation_brainstorm.md:3009:1. **只有 backbone 修改有效**: PSG (+1.7%), OA-SD (+2-3%), LGPA-D (+2.1%)
experiments/innovation_brainstorm.md:3028:考虑到 Swin vs ViT 的 backbone 差异, 73% 可能接近 Swin+SOLIDER 的天花板。
experiments/exp270_psg0_t_od_s42/monitor.md:7:- Scaffold: **pure Swin-Tiny** (no PSG / LGPA / GCN / OA-SD / PLBOA / Parallel-Aug)
experiments/exp270_psg0_t_od_s42/monitor.md:69:- **exp000 SOLIDER-Tiny 旧协议 (无 default flip)**: 56.6 / 66.5 → 本 run 加 default flip **+2.6 / +1.9**,新协议默认 flip-test 的 baseline 加成验证
experiments/smpl_anchor_vireid/design.md:35:→ **转 fallback: Swin-VI 机制**。VI-ReID 至今 ResNet50 主导、无 Swin/CLIP 赢家(纯 ViT PMT 才 67.5 R1), 团队是 Swin/SOLIDER 专家。CAJ ResNet baseline 已搭好(POOL 76.80/69.14)→ 换 Swin-Small(SOLIDER pretrain)→ 强 backbone 在 VI-ReID 帮不帮? 帮=机制论文; 不帮=为什么不帮(模态 gap 吞掉 backbone 红利)也是机制问题。先做这个经验首验。
experiments/exp270_psg0_t_od_s42/design.md:11:- LGPA / GCN / OA-SD / PLBOA / Parallel-Aug: **全部关闭**(纯 baseline,排除其他模块干扰)
experiments/exp270_psg0_t_od_s42/design.md:17:去除所有 pose 模块,等同 SOLIDER baseline + Swin-Tiny。期望 mAP ~56/67 (对照 exp000 历史 baseline 56.6/66.5,新协议 default flip-test 期望 +0.5-0.9 → ~57/67-68)。
experiments/exp270_psg0_t_od_s42/design.md:24:**改用 `POSE_ENABLED=False`** 直接走 pure `build_transformer`(纯 Swin-Tiny + ID + triplet loss,零 pose branch,等同 SOLIDER baseline),彻底绕开 bug。最终命令:
experiments/exp294_lgpaOnly_2stg_b_od_s41/design.md:11:**假设**: Base + LGPA + OA-SD + ParAug + PLBOA + 2-stage PSG (无 GCN) ≈ Base Full (带 GCN)。若成立, 证明 GCN 在所有 backbone 容量下都冗余, 简化模型。
experiments/exp294_lgpaOnly_2stg_b_od_s41/design.md:25:3. OA-SD 蒸馏 student/teacher
experiments/exp294_lgpaOnly_2stg_b_od_s41/design.md:44:cd /home/afr/SOLIDER-REID
experiments/exp294_lgpaOnly_2stg_b_od_s41/design.md:50:  OUTPUT_DIR /home/afr/SOLIDER-REID/log/occluded_duke/exp294_lgpaOnly_2stg_b_od_s41 \
experiments/exp294_lgpaOnly_2stg_b_od_s41/design.md:86:其他与 exp263d 完全相同: Base + 2-stage PSG + LGPA + OA-SD + ParAug + PLBOA + seed 41 + Occ-Duke
experiments/exp327_dinov3/monitor.md:1:# exp327 监控记录 — DINOv3 / DINOv2-with-registers pose-part-MaxSim
experiments/exp327_dinov3/monitor.md:3:脚本：`scripts/exp327_dinov3.py`（training-free，frozen DINO 系，纯推理）
experiments/exp327_dinov3/monitor.md:5:特征源（--model）：`dinov2reg-b`=facebook/dinov2-with-registers-base（默认，ungated）；`dinov3-b`=facebook/dinov3-vitb16-pretrain-lvd1689m（gated，需 token）；`dinov2-b`=facebook/dinov2-base（复现 exp324 sanity）
experiments/exp327_dinov3/monitor.md:39:**vs exp324 DINOv2-base（heavy pose-part 1.86/3.54）**：dinov2reg-b heavy **2.15/3.84（+0.29 mAP / +0.30 R1）**。ALL：3.85/8.60 vs exp324 3.21/7.87（+0.64/+0.73）。
experiments/exp327_dinov3/monitor.md:45:- 印证 exp324 假说：**训练-free 天花板瓶颈在"frozen"本身，不在 SSL 模型新旧/registers 干净度**。换更新 DINO 系只能蹭出 +0.3 量级，不足以独立可用。
experiments/exp327_dinov3/monitor.md:47:- **下一步**：registers 这点小增益**不值得单独上头**（vs exp324b 头已到 14）；若要上头优先用 DIFT（若 full 超 1.86 更多）。exp327 线**判定：更强冻结 DINO 源非天花板瓶颈解，止损**。
experiments/exp270_psg0_t_od_s42/claude_review.md:7:1. `design.md` — 单变量 ablation,关所有 pose 模块,等同 SOLIDER Tiny baseline
experiments/exp270_psg0_t_od_s42/claude_review.md:11:5. processor / forward / loss 路径: `POSE_ENABLED=True` 下会构造 PSG/LGPA/GCN/OA-SD,但 `POSE_BACKBONE_PSG=False` 时 PSG gate 退化 identity,`POSE_LGPA=False` 时 LGPA head 不构造,`POSE_SKELETON_GCN=False` 时 GCN head 不构造,`POSE_OA_SD=False` 时 EMA teacher 不启用,`POSE_LOWER_BODY_OCC=False` 时 PLBOA augmentation 跳过
experiments/exp270_psg0_t_od_s42/claude_review.md:12:6. 与前序对照: exp000 (旧 baseline,120 epoch SOLIDER-Tiny, SW=0.2) = 56.6/66.5;本次新协议加默认 flip-test 预期 +0.5-0.9
experiments/exp270_psg0_t_od_s42/claude_review.md:16:相对 exp261 (Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA) 本 run 关了 5 个模块。**多于单变量**,但 Phase 3-A 的目的不是单变量对照 exp261,而是建立一个**纯 baseline**用以对比后续 exp271-273 的 PSG 变体(单变量: PSG stages 数)。这个 baseline 本身必须没有其他模块才能隔离 PSG 的贡献。Phase 3-A 内部的 4 格(exp270/271/272/273)之间才是严格单变量。
experiments/exp270_psg0_t_od_s42/claude_review.md:29:- **收敛性**: 已在 exp000 验证过 SOLIDER-Tiny baseline 稳定收敛到 56-57 mAP
experiments/exp262_best_s_od_s42/claude_review.md:12:H1. Swin-Base + WITH_CP + POSE_PARALLEL_AUG (4 views) + OA-SD on 5060Ti 16G：Base 首 epoch OOM 风险。exp260b 历史 3090 运行 OK，但 5060Ti 16G 需要监控。若 OOM：关 OA-SD 或 PARALLEL_AUG，不改 BS。
experiments/exp262_best_s_od_s42/claude_review.md:22:M3. Market 的 OA-SD 因为没 PLBOA，teacher/student view 1 几乎相同；蒸馏信号弱但不出错。可以接受作为一致性。
experiments/exp262_best_s_od_s42/claude_review.md:28:L1-L8. OA-SD 参数跨数据集统一（可接受）；PLBOA 在 Occ-PTrack 开（数据已 occluded 但一致性保持）；POSE_PFM_ENABLED 默认 False 即使 POSE_PFM_HIDDEN=64 存在（无 bug，命名小瑕疵）；DEVICE_ID=('0')、BIAS_LR_FACTOR=2、WARMUP_EPOCHS=20 都与 exp255 scaffold 一致。
experiments/smpl_anchor_vireid/baseline_integration_plan.md:22:### (a) ResNet50→Swin-Small(团队 SOLIDER 权重)
experiments/smpl_anchor_vireid/baseline_integration_plan.md:23:- 复用 `/Users/abdslm/Desktop/SOLIDER-REID/model/backbones/swin_transformer.py` 的 `swin_small_patch4_window7_224`(embed96, depths(2,2,18,2), 最终768)。forward 返回(global_feat[B,768], outs)。`base.init_weights(model_path)` 加载。
experiments/smpl_anchor_vireid/baseline_integration_plan.md:26:- 三份 Swin(2 stem+1 shared)同一 SOLIDER ckpt 初始化。输入288×144→patch4→72×36 OK。
experiments/exp327_dinov3/claude_review.md:1:# Claude Broad Review — exp326 (DIFT) + exp327 (DINOv3)
experiments/exp327_dinov3/claude_review.md:8:- exp327 register-token 切片 `out[:, 1+nreg : 1+nreg+n_tok]`：DINOv2-with-registers / DINOv3-ViT 的 `last_hidden_state` 布局确为 `[CLS, registers, patches]`，DINOv3-ViT-B = 1 CLS + 4 reg + 196 patch，切片正确；`assert` 正确；`dinov3_vit` 在 transformers 5.12 的 `MODEL_MAPPING_NAMES` 中，`AutoModel.from_pretrained` 可解析。
experiments/exp327_dinov3/claude_review.md:19:**H2 (High) — DINOv3 是 gated 模型**：`facebook/dinov3-*` 需接受 license + token，匿名 `from_pretrained` 可能 401/404。
experiments/exp254/monitor.md:3:配置: Swin-Tiny + PSG Stage2+3 (无 PAA) + LGPA-D+GCN+OA-SD+PLBOA + WITH_CP
experiments/exp294_lgpaOnly_2stg_b_od_s41/monitor.md:5:- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp294_lgpaOnly_2stg_b_od_s41/train_log.txt` (UTC clock)
experiments/exp294_lgpaOnly_2stg_b_od_s41/monitor.md:7:- Scaffold: Swin-Base + **LGPA + OA-SD + ParAug + PLBOA + 2-stage PSG `[-2,-1]`** (**NO GCN**)
experiments/exp254/claude_review.md:145:语义权重（SOLIDER 预训练特有）在每个 stage 循环之后应用。PSG 注入发生在 block 内部（stage 循环内部）。两者层级不冲突。semantic weight 作用于 PSG-gated 后的特征。
experiments/cargo_cvpb/codex_review_ovli.txt:4:workdir: /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:18:/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:30:| 000 | Baseline (SOLIDER-Swin-Tiny, SW=0.2) | 56.6% | 66.5% | 79.4% | 83.4% | — | 120 epoch, 完美复现 |
experiments/cargo_cvpb/codex_review_ovli.txt:180:| 142-eq | PSG+GCN+SKC (Support-Supervised Keypoint Completion, eq) | 60.3% | 71.8% | 84.4% | 87.7% | vs exp030a-eq: -0.8%/-1.9% | ❌ 中性偏负。completion module 虽然活跃（gate=0.26, delta_norm=1.5），但 skc_pre≈skc_post 说明修改方向不是向 prototype 靠近。gate 无限制增长导致后期过度修改特征。feature-level completion 方向已被多轮验证为无效 |
experiments/cargo_cvpb/codex_review_ovli.txt:185:| 148-eq | PSG+GCN+PCVT (Pose-Complementary View Training, eq) | ~59.3%* | ~71.3%* | — | — | ❌ 负面。*ep100 数据，训练中。早期加速（ep30: +2.4 mAP）但后期被基线追平并反超。3-view 训练的 1/3 主损失稀释导致后期收敛不足。训练集 95.8% 全可见使 complementary masking 缺乏信号 |
experiments/cargo_cvpb/codex_review_ovli.txt:227:### Phase 4: SupCon + OA-SD + Parallel Aug (exp166-193)
experiments/cargo_cvpb/codex_review_ovli.txt:232:| 166r | ↳ base arch (no PAPE/MS-PSG) | 60.3% | 72.8% | — | — | CE base for OA-SD comparison |
experiments/cargo_cvpb/codex_review_ovli.txt:234:| **187** | **+SupCon T=0.05 + 3-view Parallel Aug** | **64.9%** | **76.6%** | **87.2%** | **90.0%** | **🟢🟢 Overall best! +1.8/+2.7 vs exp166** |
experiments/cargo_cvpb/codex_review_ovli.txt:235:| **190** | **3-view Parallel Aug + CE (no SupCon)** | **64.2%** | **75.6%** | **86.2%** | **89.1%** | **✅ 3-view+CE ≈ 1-view+SupCon! +1.1/+1.7 vs exp166** |
experiments/cargo_cvpb/codex_review_ovli.txt:236:| **191** | **OA-SD + CE (1-view, decay=0.999)** | **63.2%** | **75.4%** | **86.3%** | **89.1%** | **✅ OA-SD 独立有效! +2.9/+2.6 vs CE base** |
experiments/cargo_cvpb/codex_review_ovli.txt:237:| 192 | OA-SD + CE (1-view, decay=0.99) | 62.6% | 74.9% | 86.0% | 89.1% | 🟡 vs exp191: -0.6/-0.5。decay 不敏感 |
experiments/cargo_cvpb/codex_review_ovli.txt:238:| **193** | **OA-SD + 3-view + CE** | **64.4%** | **76.5%** | **86.3%** | **89.4%** | **✅ OA-SD+3-view additive! R1 ≈ exp187 SupCon, mAP +0.2/R1+0.9 vs exp190** |
experiments/cargo_cvpb/codex_review_ovli.txt:239:| 194 | OA-SD + CE (weight=2.0) | 63.4% | 74.8% | 86.1% | 89.1% | 🟡 vs exp191 (w=1.0): +0.2/-0.6。weight 不敏感 |
experiments/cargo_cvpb/codex_review_ovli.txt:242:/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:250:**选择**: 从 SOLIDER 作者原始代码重新开始，纯 pose heatmap 方向
experiments/cargo_cvpb/codex_review_ovli.txt:466:/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:483:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/cargo_cvpb/codex_review_ovli.txt:520:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/cargo_cvpb/codex_review_ovli.txt:690:/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:710:4. `LGPA-D / OA-SD / PLBOA` 作为完整系统资产
experiments/cargo_cvpb/codex_review_ovli.txt:711:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/cargo_cvpb/codex_review_ovli.txt:734:- `Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA`
experiments/cargo_cvpb/codex_review_ovli.txt:772:   - `OA-SD`
experiments/cargo_cvpb/codex_review_ovli.txt:778:   - 写法：test-time supporting evaluations
experiments/cargo_cvpb/codex_review_ovli.txt:826:| Tiny | LGPA-D+OA-SD | 65.3% | 75.7% | 66.0% | 76.4% |
experiments/cargo_cvpb/codex_review_ovli.txt:827:| **Tiny** | **LGPA-D+GCN+OA-SD** | **65.5%** | **77.2%** | **66.3%** | **77.7%** |
experiments/cargo_cvpb/codex_review_ovli.txt:828:| Small | LGPA-D+OA-SD (local) | 70.2% | 80.1% | 71.9% | 82.2% |
experiments/cargo_cvpb/codex_review_ovli.txt:829:| **Small** | **LGPA-D+OA-SD (remote)** | **71.6%** | **81.6%** | **73.0%** | **82.7%** |
experiments/cargo_cvpb/codex_review_ovli.txt:830:| Small | GCN+PAA+OA-SD (old baseline) | 70.6% | 82.6% | 72.3% | 82.9% |
experiments/cargo_cvpb/codex_review_ovli.txt:831:| *Small* | *LGPA-D+GCN+OA-SD (exp249, 进行中)* | *TBD* | *TBD* | *TBD* | *TBD* |
experiments/cargo_cvpb/codex_review_ovli.txt:866:   - LGPA-D 无 OA-SD (63.6) ≈ GCN + OA-SD (63.2)
experiments/cargo_cvpb/codex_review_ovli.txt:867:   - CLIP 的 part assignment 能力 ≈ OA-SD 的训练增强
experiments/cargo_cvpb/codex_review_ovli.txt:889:| Tiny | GCN+PAA+OA-SD | 63.2% | 75.4% | 64.2% | 77.1% |
experiments/cargo_cvpb/codex_review_ovli.txt:890:| Tiny | **GCN+PAA+OA-SD+GSPB** | 62.9% | 74.3% | **64.6%** | **76.0%** |
experiments/cargo_cvpb/codex_review_ovli.txt:891:| Small | GCN+PAA+OA-SD | 70.6% | 82.6% | 72.3% | 82.9% |
experiments/cargo_cvpb/codex_review_ovli.txt:892:| Small | GCN+PAA+OA-SD+PKC | 70.6% | 81.8% | **72.4%** | **83.1%** |
experiments/cargo_cvpb/codex_review_ovli.txt:900:3. **OA-SD Teacher Fix**: 修复了 EMA teacher 的 Dropout/DropPath/BN 噪声问题。修复后 teacher 更稳定，但 final 结果不变（EMA 的自修正性）。
experiments/cargo_cvpb/codex_review_ovli.txt:914:/bin/zsh -lc 'find experiments -maxdepth 2 -name design.md | sort -V | tail -5' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:923:/bin/zsh -lc 'find experiments -maxdepth 2 -name monitor.md | sort -V | tail -5' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:934:/bin/zsh -lc "sed -n '1,220p' experiments/exp358_pose_channel_shuffle/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:936:/bin/zsh -lc "sed -n '1,220p' experiments/exp358_pose_channel_shuffle/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:945:exp358 disambiguate: **打乱 17 关键点通道(per-image)→ 破坏解剖部位身份, 但保留同图自己的空间 support**(关键点位置是本图的, 只是哪个点属于哪个部位被打乱)。无裁剪对齐 rescue。
experiments/cargo_cvpb/codex_review_ovli.txt:970:/bin/zsh -lc "sed -n '1,220p' experiments/smpl_anchor_vireid/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:1009:/bin/zsh -lc "sed -n '1,220p' experiments/smpl_anchor_vireid/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:1011:/bin/zsh -lc "sed -n '1,240p' experiments/cargo_cvpb/afd_train.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:1047:→ **转 fallback: Swin-VI 机制**。VI-ReID 至今 ResNet50 主导、无 Swin/CLIP 赢家(纯 ViT PMT 才 67.5 R1), 团队是 Swin/SOLIDER 专家。CAJ ResNet baseline 已搭好(POOL 76.80/69.14)→ 换 Swin-Small(SOLIDER pretrain)→ 强 backbone 在 VI-ReID 帮不帮? 帮=机制论文; 不帮=为什么不帮(模态 gap 吞掉 backbone 红利)也是机制问题。先做这个经验首验。
experiments/cargo_cvpb/codex_review_ovli.txt:1149:    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/codex_review_ovli.txt:1152:        --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/codex_review_ovli.txt:1153:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
experiments/cargo_cvpb/codex_review_ovli.txt:1158:        --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/codex_review_ovli.txt:1159:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/codex_review_ovli.txt:1314:/bin/zsh -lc 'rg -n "ovli|OVLI|ovp|OVP|MaxSim|late|optimizer|optim|hook|register|layer4|GradScaler|autocast|amp|rerank|alpha|tau" experiments/cargo_cvpb/afd_train.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:1348:82:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
experiments/cargo_cvpb/codex_review_ovli.txt:1352:88:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/codex_review_ovli.txt:1556:/bin/zsh -lc "nl -ba experiments/cargo_cvpb/afd_train.py | sed -n '240,360p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:1681:/bin/zsh -lc "nl -ba experiments/cargo_cvpb/afd_train.py | sed -n '360,460p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:1765:   442	    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/cargo_cvpb/codex_review_ovli.txt:1786:/bin/zsh -lc "nl -ba experiments/cargo_cvpb/afd_train.py | sed -n '460,580p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:1911:/bin/zsh -lc "nl -ba experiments/cargo_cvpb/afd_train.py | sed -n '580,730p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2060:/bin/zsh -lc 'rg -n "class |def |return|loss =|optimizer|autocast|GradScaler|run_cross_view_eval|print_eval|torch.save|load|state_dict|args.ovli|args.ovp|ovli" experiments/cargo_cvpb/afd_train.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2069:88:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/codex_review_ovli.txt:2205:/bin/zsh -lc 'rg -n "class .*Model|def build_model|layer4|in_planes|forward|return" experiments/afd_reid/afd_model.py experiments/afd_reid/afd_train.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2271:/bin/zsh -lc "nl -ba experiments/afd_reid/afd_model.py | sed -n '215,340p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2273:/bin/zsh -lc "nl -ba experiments/afd_reid/afd_train.py | sed -n '120,245p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2405:   217	                 pretrained=True,
experiments/cargo_cvpb/codex_review_ovli.txt:2418:   230	        weights = 'IMAGENET1K_V1' if pretrained else None
experiments/cargo_cvpb/codex_review_ovli.txt:2531:/bin/zsh -lc 'rg -n "class RandomIdentitySampler|view|Aerial|Ground|pid|camid|__getitem__|return" experiments/afd_reid/cargo_dataset.py' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2611:/bin/zsh -lc "nl -ba experiments/afd_reid/cargo_dataset.py | sed -n '1,240p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2618:     6	    /root/work/SOLIDER-REID/data/CARGO/{train,query,gallery}/Cam{1..13}/*.jpg
experiments/cargo_cvpb/codex_review_ovli.txt:2683:    71	    def __init__(self, root='/root/work/SOLIDER-REID/data', verbose=True):
experiments/cargo_cvpb/codex_review_ovli.txt:2855:/bin/zsh -lc "nl -ba experiments/afd_reid/afd_train.py | sed -n '245,380p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2861:   249	    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/cargo_cvpb/codex_review_ovli.txt:2995:/bin/zsh -lc "rg -n \"maxsim_probe|eval_from_distmat|def eval_from_distmat|def .*maxsim|OVLI|cvpb\" experiments -g '*.py'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:2997:experiments/cargo_cvpb/maxsim_probe.py:40:    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/codex_review_ovli.txt:3009:experiments/cargo_cvpb/afd_train.py:78:    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/codex_review_ovli.txt:3010:experiments/cargo_cvpb/afd_train.py:82:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
experiments/cargo_cvpb/codex_review_ovli.txt:3013:experiments/cargo_cvpb/afd_train.py:88:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/codex_review_ovli.txt:3050:/bin/zsh -lc "nl -ba experiments/afd_reid/cargo_dataset.py | sed -n '226,266p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:3095:/bin/zsh -lc "nl -ba experiments/cargo_cvpb/maxsim_probe.py | sed -n '120,210p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:3097:/bin/zsh -lc "nl -ba experiments/cargo_cvpb/maxsim_probe.py | sed -n '300,335p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:3230:/bin/zsh -lc 'find experiments/cargo_cvpb -maxdepth 2 -type f | sort' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:3243:/bin/zsh -lc "rg -n \"ovli|OVLI|cvpb|cargo_cvpb|opposite-view|late-interaction|MaxSim|ColBERT\" experiments -g '*.md'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:3266:experiments/exp327_dinov3/design.md:1:# 实验 exp327: DINOv3 / DINOv2-with-registers pose-anchored part-MaxSim 训练-free 天花板 check
experiments/cargo_cvpb/codex_review_ovli.txt:3267:experiments/exp327_dinov3/design.md:9:- exp324 用 frozen DINOv2-base 验证了机制（pose 锚定 dense token + mutually-visible MaxSim），但天花板低（重遮挡 1.86，训头 e20 到顶 14）。
experiments/cargo_cvpb/codex_review_ovli.txt:3291:experiments/exp327_dinov3/monitor.md:1:# exp327 监控记录 — DINOv3 / DINOv2-with-registers pose-part-MaxSim
experiments/cargo_cvpb/codex_review_ovli.txt:3337:experiments/exp328_vcnorm/design.md:5:> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。
experiments/cargo_cvpb/codex_review_ovli.txt:3353:experiments/decisions.md:3190:### [2026-04-01 10:30] 决策 #X — MaxSim + PKC + Fixed OA-SD 路线
experiments/cargo_cvpb/codex_review_ovli.txt:3354:experiments/decisions.md:3192:**上下文**: MaxSim hybrid 在 exp206 checkpoint 上无需重训给 +1.8% mAP (70.3→72.1)。OA-SD teacher bug 已修复。PKC (Per-Keypoint Contrastive) 开始测试。
experiments/cargo_cvpb/codex_review_ovli.txt:3359:experiments/decisions.md:3276:2. `MaxSim` 对 `OA-SD` 本身仍是正向的（`63.2 -> 64.2`），只是 `OERL/PACI` 并没有把这个 test-time 上限继续抬高
experiments/cargo_cvpb/codex_review_ovli.txt:3370:experiments/decisions.md:3472:2. LGPA-D 虽然 single novelty 4.5/10, 但与 PSG+OA-SD+MaxSim 组成完整 framework novelty 更高
experiments/cargo_cvpb/codex_review_ovli.txt:3372:experiments/decisions.md:3481:- `LGPA-D + GCN + OA-SD + MaxSim + flip` 虽然结果强，但主问题定义仍偏弱
experiments/cargo_cvpb/codex_review_ovli.txt:3374:experiments/decisions.md:3493:3. `MaxSim / POT / flip` 主要仍是 test-time supporting evidence，不能作为训练端主贡献
experiments/cargo_cvpb/codex_review_ovli.txt:3376:experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/cargo_cvpb/codex_review_ovli.txt:3377:experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/cargo_cvpb/codex_review_ovli.txt:3409:experiments/decisions.md:4473:### [2026-06-16] 决策 #exp324 — DINO emergent correspondence + pose-anchored part-MaxSim 首验偏正
experiments/cargo_cvpb/codex_review_ovli.txt:3428:experiments/prcv_2026_psg/decisions.md:24:4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/cargo_cvpb/codex_review_ovli.txt:3513:experiments/exp223/monitor.md:4:对照: exp191 OA-SD (fixed sampling): 63.2/75.4, exp220 GSPB+MaxSim: 64.6
experiments/cargo_cvpb/codex_review_ovli.txt:3520:experiments/exp223/monitor.md:228:如果趋势持续，PADPQ final 可能在 `equal_concat mAP` 上超过 OA-SD；但是否能形成综合超越，仍取决于 R1 和后续 `MaxSim`。
experiments/cargo_cvpb/codex_review_ovli.txt:3526:experiments/exp223/monitor.md:307:**PADPQ + MaxSim = 63.9 — 低于 GSPB (64.6) 和 OA-SD (64.2)!**
experiments/cargo_cvpb/codex_review_ovli.txt:3533:experiments/exp222/monitor.md:7:**动机**: exp220 (Tiny) 证明 GSPB+MaxSim=64.6 > OA-SD+MaxSim=64.2 (+0.4)
experiments/cargo_cvpb/codex_review_ovli.txt:3534:experiments/exp222/monitor.md:86:**Small 的 GCN+PAA+OA-SD+MaxSim 72.4% 仍是最佳。**
experiments/cargo_cvpb/codex_review_ovli.txt:3537:experiments/MORNING_REPORT.md:97:**decorr loss 全程活跃却完全没动 Jaccard/fusion。** 原因（这就是洞察）：强迫全局**线性**解相关，对"模型给 query 排哪些 gallery"（part-MaxSim 排序）是**正交**的——检索由 part-MaxSim over 相同可见身体部位证据决定，两模型受**同一份可见证据**约束、犯**同样的错**（Swin 对 370/989=37% 重遮挡 query 对、DINO 只补 8 个=0.81%）。
experiments/cargo_cvpb/codex_review_ovli.txt:3540:experiments/exp269_best_b_m_s42/monitor.md:69:- Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) + default flip-test @ Market-1501 (e80 eff FINAL) = **94.4 / 97.0 (Global+flip)**, **94.5 / 97.1 (MaxSim+flip)**
experiments/cargo_cvpb/codex_review_ovli.txt:3545:experiments/exp324h/design.md:7:- **但 exp324d 让 DINO 经 LoRA adaptation 变判别了**：base-r16 重遮挡 part-MaxSim mAP
experiments/cargo_cvpb/codex_review_ovli.txt:3546:experiments/exp324h/design.md:13:- 一句话：LoRA-adapted DINO 的 part-MaxSim 检索在重遮挡 query 上携带 Swin 漏掉的独立正确信息
experiments/cargo_cvpb/codex_review_ovli.txt:3547:experiments/exp324h/design.md:22:  - `part_maxsim_distmat`（exp324_dino）→ adapted-DINO part-MaxSim distmat（Q×G）
experiments/cargo_cvpb/codex_review_ovli.txt:3548:experiments/exp324h/design.md:32:→ L2-norm 5-part → part-MaxSim distmat（adapted-DINO）→ 对齐 Swin 顺序 → oracle 对照。
experiments/cargo_cvpb/codex_review_ovli.txt:3557:experiments/exp324h/monitor.md:22:adapted-DINO part-MaxSim ALONE: **all mAP=44.67 R1=57.01**（= exp324d e10，验证 LoRA 加载+encode+distmat 全链路正确）。
experiments/cargo_cvpb/codex_review_ovli.txt:3575:experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/cargo_cvpb/codex_review_ovli.txt:3577:experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/cargo_cvpb/codex_review_ovli.txt:3605:experiments/innovation_brainstorm.md:2865:1. `MaxSim` 对 OA-SD 本身仍然有效（`63.2 -> 64.2`）
experiments/cargo_cvpb/codex_review_ovli.txt:3619:experiments/exp319_tiny_oasdW05_s42/monitor.md:26:加上 exp316 (oasdW=2.0) MaxSim 0/-0.6, OA-SD weight default 1.0 验证为 sweet spot, 上下 1× 都微负。
experiments/cargo_cvpb/codex_review_ovli.txt:3623:experiments/exp324d/codex_review.md:21:LoRA / DINOv2 / pose-visible-part matching 各自有先例（PVPM/PFD），DINOv2-for-ReID 新兴，LoRA-DINOv2 在 ReID 外存在。**未找到 LoRA-finetuned DINOv2 + 可微姿态锚定部位池化 + mutually-visible part-MaxSim 用于遮挡单图行人 ReID 的直接先例** → 组合 plausibly new。
experiments/cargo_cvpb/codex_review_ovli.txt:3633:experiments/exp324d/design.md:9:- exp324b 证明：把冻结 DINO 部位特征投到 ReID 空间能从 1.86 → 14.61（×7.9），机制（姿态锚定 + 只比可见部位 part-MaxSim）干净有效。
experiments/cargo_cvpb/codex_review_ovli.txt:3634:experiments/exp324d/design.md:15:给 DINOv2-base attention 加 LoRA（q/v，rank 8/16，alpha 16），DINO 主权重冻结，只训 LoRA + 轻量头（proj + BNNeck + 全局分类器 + part 分类器），**重遮挡 part-MaxSim mAP 能突破 14 往 competitive 走**。能破 → DINO 创新线可做成；破不动 → 冻结对应特征这条线天花板确实低，止损。
experiments/cargo_cvpb/codex_review_ovli.txt:3636:experiments/exp324d/design.md:50:- 假设成立：重遮挡 part-MaxSim mAP 破 14 明显上升（进入 20-40+ → DINO 线有救、competitive 可期）。
experiments/cargo_cvpb/codex_review_ovli.txt:3653:experiments/exp324d/claude_review.md:1:# Claude Broad Review — exp324d (LoRA-unfrozen DINOv2 + pose part-MaxSim)
experiments/cargo_cvpb/codex_review_ovli.txt:3654:experiments/exp324d/claude_review.md:29:heavy-occ mask（vis≤8）、same-cam 排除（eval_func 内）、distmat（cosine + part-MaxSim）全不变。无 autocast，DINO float32，pool_w float32 → bmm dtype 一致；bn/pp `.float()` 后转 numpy。无 dtype mismatch。PASS。
experiments/cargo_cvpb/codex_review_ovli.txt:3659:experiments/exp324d_large/design.md:1:# 实验 exp324d_large: LoRA-解冻 DINOv2-large + pose-anchored part-MaxSim
experiments/cargo_cvpb/codex_review_ovli.txt:3671:experiments/exp324d_large/monitor.md:9:- 冻结 DINOv2-base (exp324b)：part-MaxSim 重遮挡 mAP **8.65** / 全部 **14.61**（e20 触顶）
experiments/cargo_cvpb/codex_review_ovli.txt:3698:experiments/exp324f/design.md:3:> **来源**：过夜创新探索（overnight_innovation_log.md）。exp324b 已证冻结 DINO + 姿态部位 part-MaxSim 在重遮挡上有机制信号（part 重遮挡 8.65 / 全部 14.61），但**绝对分远低于 exp255 Swin SOTA（75 mAP MaxSim）**。本实验换一个**重量级角度**：不再单独评 DINO，而是问"**给 75-mAP 的 SOTA Swin 模型加上 DINO 的遮挡部位对应，重遮挡 query 能否比 Swin 单独更好？**"——建在 75 上而非 14 上。
experiments/cargo_cvpb/codex_review_ovli.txt:3700:experiments/exp324f/design.md:13:- DINO 姿态部位对应（exp324b）提供一种**正交的、自监督的、姿态显式锚定的**遮挡鲁棒匹配信号：跨图只比双方可见部位（mutually-visible part-MaxSim），天然对遮挡部位免疫。
experiments/cargo_cvpb/codex_review_ovli.txt:3701:experiments/exp324f/design.md:18:DINO 姿态部位对应的距离矩阵与 Swin MaxSim 距离矩阵**互补**：在重遮挡 query 子集上，加权融合 `d = (1-w)·d_swin + w·d_dino`（w>0）的 mAP/R1 高于 w=0（Swin 单独）。
experiments/cargo_cvpb/codex_review_ovli.txt:3704:experiments/exp324f/design.md:27:2. **DINO part-MaxSim distmat**（`d_dino`，Q×G）：
experiments/cargo_cvpb/codex_review_ovli.txt:3709:experiments/exp325/codex_review.md:20:- **新颖性（web search）**：DINOv2/基础模型用于 ReID、score-level/late fusion、姿态引导可见部位匹配均有先例。未找到完全相同组合"frozen DINOv2 dense + 姿态锚定 5 部位轻量头 + mutually-visible part-MaxSim on Occluded-Duke"，但应**窄框为实验机制/诊断**，非广义新颖。来源：DINOv2 HF config、ECHO-BID、DinoGRL、Query Adaptive Late Fusion、PFD、PGFL-KD。
experiments/cargo_cvpb/codex_review_ovli.txt:3730:experiments/exp325/design.md:3:> **来源**：过夜创新探索。exp324b 证冻结 DINOv2-**base** 训轻量头 part-MaxSim 重遮挡 8.65 / 全部 14.61，但 e20 即到顶 → 冻结 base 特征天花板低。本实验换**更强冻结 backbone（DINOv2-large，hidden 1024，patch 14）**，问"**更强冻结模型重遮挡 mAP 能否抬过 exp324b 的 8.65 / 全部 14.61？**"。
experiments/cargo_cvpb/codex_review_ovli.txt:3731:experiments/exp325/design.md:14:把 exp324b 的冻结 backbone 从 DINOv2-base 换成 DINOv2-large（其余 pipeline、损失、超参、采样、评测**完全不变**，单变量），part-MaxSim 重遮挡 mAP > 8.65、全部 > 14.61。
experiments/cargo_cvpb/codex_review_ovli.txt:3737:experiments/exp324f/monitor.md:12:- **DINO part-MaxSim ALONE: mAP=14.61 R1=21.99** —— 与 exp324b e60 完全一致 ✓
experiments/cargo_cvpb/codex_review_ovli.txt:3738:experiments/exp324f/monitor.md:42:- 冻结 DINO part-MaxSim 信号（14.61 全部 / 8.65 重遮挡，量级远低于 Swin）**噪声太大**，对 SOTA Swin 是**严格冗余且有害**的距离源，任何正权重都拖垮排序。
experiments/cargo_cvpb/codex_review_ovli.txt:3752:experiments/cargo_cvpb/monitor.md:33:cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/codex_review_ovli.txt:3753:experiments/cargo_cvpb/monitor.md:36:  --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
experiments/cargo_cvpb/codex_review_ovli.txt:3764:experiments/cargo_cvpb/monitor.md:56:cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/codex_review_ovli.txt:3765:experiments/cargo_cvpb/monitor.md:59:  --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/codex_review_ovli.txt:3832:experiments/exp324i/claude_review.md:57:- 但 headline 张力（Jaccard 0.062→0.253）与最终 fusion eval 用的是 **FULL MaxSim = global + 5 part 子空间**。decorr 只把 DINO-global 推离 Swin-**global**，**没有**推离 Swin 的 part 子空间。
experiments/cargo_cvpb/codex_review_ovli.txt:3834:experiments/exp324g/design.md:4:- "DINO 补 Swin" 整条创新家族（frozen DINOv2 pose-part-MaxSim 给 75-mAP SOTA Swin 补遮挡信息）需要一个 gate：
experiments/cargo_cvpb/codex_review_ovli.txt:3835:experiments/exp324g/design.md:5:  **在重遮挡 query 上，DINO part-MaxSim 是否携带 Swin 漏掉的独立正确信息？**
experiments/cargo_cvpb/codex_review_ovli.txt:3836:experiments/exp324g/design.md:17:  - `exp324f_fuse.get_dino_distmat`：从 exp324b 缓存部位特征 + head_60.pth 算 DINO part-MaxSim distmat（无 DINO 前向，几乎不吃 GPU）。
experiments/cargo_cvpb/codex_review_ovli.txt:3839:experiments/exp324i/design.md:23:4. **eval**：decorr-DINO part-MaxSim（heavy/all） + 复用 `exp324h_lora_oracle.py` 的 **adapted-DINO ⊕ Swin oracle + fusion sweep**，看 fusion 重遮挡/全部能否 **> Swin 72.57/75**。
experiments/cargo_cvpb/codex_review_ovli.txt:3840:experiments/exp324i/design.md:29:- **已知 scope 限制（Claude review Medium#1）**：缓存的 `s` 只是 Swin 的 **holistic global**（前 768 维），decorr 只把 DINO-global 推离 Swin-global，**没动 Swin 的 part 子空间**；而 tension 指标 / fusion eval 是**完整 MaxSim（global+parts）**。所以 null 结果可能是"张力打不破"或"只解相关了 global 不够"二义。v1 先做 global-vs-global（最干净的单变量介入）；若 λ>0 在全 MaxSim Jaccard 上有移动→值得再做 part-level decorr v2。设计里诚实标注，不夸大。
experiments/cargo_cvpb/codex_review_ovli.txt:3849:experiments/exp324g/monitor.md:13:- DINO part-MaxSim ALONE（全 query）：mAP=14.61 R1=21.99（与 exp324f 报的 DINO-only 数一致，确认 head/缓存加载正确）。
experiments/cargo_cvpb/codex_review_ovli.txt:3850:experiments/exp324g/monitor.md:37:- Jaccard 极低（0.06）看似"正交"，但这是**虚假正交**：DINO part-MaxSim 整体判别力太弱
experiments/cargo_cvpb/codex_review_ovli.txt:3855:experiments/exp211/design.md:1:# 实验 exp211: GCN+PAA+CE+OA-SD + MaxSim Triplet (MST)
experiments/cargo_cvpb/codex_review_ovli.txt:3871:experiments/exp220/monitor.md:334:**64.6% = Tiny 新最佳！ GSPB + MaxSim 超过 OA-SD + MaxSim by +0.4%!**
experiments/cargo_cvpb/codex_review_ovli.txt:3873:experiments/exp220/monitor.md:338:**MaxSim gain: GSPB +1.7 vs OA-SD +1.0 → GSPB 产生了更好的 per-keypoint features!**
experiments/cargo_cvpb/codex_review_ovli.txt:3877:experiments/exp316_tiny_oasdW2_s42/monitor.md:41:OA-SD weight 2.0 在 Tiny 上 **net neutral**: eq slight +, MaxSim mAP =, MaxSim R1 -0.6。Default 1.0 is sweet spot for Tiny。
experiments/cargo_cvpb/codex_review_ovli.txt:3878:experiments/exp211/monitor.md:1:# exp211 Small + GCN+PAA+CE+OA-SD + MST (MaxSim Triplet) 监控
experiments/cargo_cvpb/codex_review_ovli.txt:3892:experiments/exp218/design.md:6:当前所有方法（包括我们的 PSG+GCN+OA-SD+MaxSim）都把 identity 当作单一概念：
experiments/cargo_cvpb/codex_review_ovli.txt:3900:experiments/exp218/monitor.md:319:**就当时已完成的 OA-SD / OERL / PACI 三条 Tiny 线而言，MaxSim 都落在 64.1-64.3。**
experiments/cargo_cvpb/codex_review_ovli.txt:3901:experiments/exp218/monitor.md:328:- `MaxSim` 对 OA-SD 本身仍是正向的（`63.2 -> 64.2`）
experiments/cargo_cvpb/codex_review_ovli.txt:3927:experiments/exp326_dift/design.md:9:- exp324 证明 frozen DINOv2-base + 姿态锚定 5-part + mutually-visible MaxSim 在重遮挡子集有干净相对信号（pose 锚定 ×3.4 vs 整图），但绝对天花板低（重遮挡 1.86 mAP，训头后 e20 即到顶 14）。
experiments/cargo_cvpb/codex_review_ovli.txt:3928:experiments/exp326_dift/design.md:15:frozen SD-v1.5 UNet，对图像 VAE 编码 → 小 timestep t 加噪 → 单步 UNet forward → 取 up_block 中间激活作 dense 特征，按 pose 锚定 5-part + mutually-visible part-MaxSim，在重遮挡子集 mAP **超过 exp324 DINOv2-base 的 1.86**。超了 → SD 特征值得上轻量头；不超 → SD 训练-free 不优于 DINO，路线降级。
experiments/cargo_cvpb/codex_review_ovli.txt:3931:experiments/exp326_dift/design.md:40:- baseline = exp324 frozen DINOv2-base 重遮挡 pose-part 1.86（同 pose data、同 5-part、同 MaxSim、同重遮挡口径）。唯一变量 = **特征源**（DINOv2 → SD-DIFT）。
experiments/cargo_cvpb/codex_review_ovli.txt:3932:experiments/exp324d_r32/design.md:1:# 实验 exp324d_r32: LoRA-解冻 DINOv2-base rank32 + pose-anchored part-MaxSim
experiments/cargo_cvpb/codex_review_ovli.txt:3942:experiments/exp324d_r32/monitor.md:9:- 冻结 DINOv2-base (exp324b)：part-MaxSim 重遮挡 mAP **8.65** / 全部 **14.61**（e20 触顶）
experiments/cargo_cvpb/codex_review_ovli.txt:3956:experiments/overnight_innovation_log.md:44:- **exp324d 第2次（本次）已启动**（2026-06-16，lab-3090-d）：`scripts/exp324d_lora.py` 写好——LoRA(q/v rank8 alpha16) 解冻 DINOv2-base + **可微姿态部位池化**(预算每图 row-stochastic 池化矩阵 pool_w，`bmm(pool_w,patch)` 梯度回流 LoRA) + exp324b 同款头/损失/PK64/part-MaxSim eval。**唯一变量=LoRA 解冻 DINO**。双审查 approve（Claude 含 200-trial 池化等价 1.79e-7；Codex 141k tok 确认梯度到 LoRA/full-batch triplet/use_reentrant=False/eval 对称，组合 plausibly new 未找直接先例）。Dry-run：LoRA 294,912+head 413,184 训练，loss↓ acc 0.016→0.766，显存 13.7G/24G，~1.4s/step。全量 35ep 后台跑（micro_bs64 no-ckpt）。**待结果：part-MaxSim 重遮挡 mAP 能否破 exp324b 的 8.65 / 全部破 14.61 往 competitive 走。**【结果待填】
experiments/cargo_cvpb/codex_review_ovli.txt:3957:experiments/overnight_innovation_log.md:49:1. **Rank-disagreement oracle（0-GPU，~40行 numpy，⭐立即做，依赖 Swin distmat）**：重遮挡子集上比 Swin top-k vs DINO part-MaxSim top-k 的 Jaccard / P_dino_only（DINO 命中 Swin 漏的真值）/ per-query oracle 上界。判定：P_dino_only<2% 或 oracle<+1mAP → 整条 DINO⊕Swin 当场止损；oracle +3~5 且 Jaccard<0.5 → 正交坐实，进 #2。
experiments/cargo_cvpb/codex_review_ovli.txt:3958:experiments/overnight_innovation_log.md:50:2. **遮挡门控 part-MaxSim re-rank（training-free，最可能产出有用主表结果）**：DINO 共可见 distmat 当 `utils/reranking.py re_ranking(local_distmat=)` 输入，只对重遮挡 query 触发。依赖 #1。
experiments/cargo_cvpb/codex_review_ovli.txt:3960:experiments/overnight_innovation_log.md:65:- Swin MaxSim ALONE **75.16/85.57**（=主线 75.2 ✓）；DINO part-MaxSim ALONE **14.61/21.99**（=exp324b e60 ✓）；对齐 sanity 全过（文件名 join，pid 全等，camid 偏移恒=1，w=0=75.16）。
experiments/cargo_cvpb/codex_review_ovli.txt:3963:experiments/overnight_innovation_log.md:97:机理：Jaccard 0.06 是**虚假正交**——DINO part-MaxSim 整体太弱（heavy 8.65 vs Swin 72.57，差 64 mAP），top-10 基本是噪声，自然不与 Swin 重叠，但不重叠 ≠ 命中。决定性证据是 oracle 上界：连"每 query 取 Swin/DINO 更优 AP"都只 +0.12，说明 **Swin 失败的 query 上 DINO 几乎也失败**。exp324f late-fusion 涨不动不是融合方式问题，而是**信息上限不存在**。
experiments/cargo_cvpb/codex_review_ovli.txt:3964:experiments/overnight_innovation_log.md:100:- #2（遮挡门控 part-MaxSim re-rank）**依赖 #1 正向 → 现已被砍**，DINO 的共可见 distmat 当 re-rank 输入不会有增益，不做。
experiments/cargo_cvpb/codex_review_ovli.txt:3965:experiments/overnight_innovation_log.md:101:- #3/#5 OT 线同理：把 part-MaxSim 平均换 Sinkhorn 只改聚合方式，无法凭空造出 DINO 没有的判别信息，**优先级降到底**（除非先有证据 DINO part 判别力可被大幅拉高）。
experiments/cargo_cvpb/codex_review_ovli.txt:3966:experiments/overnight_innovation_log.md:107:**机制结论**：低 Jaccard 是"假正交"——DINO part-MaxSim 不是"补充"，是"全局太弱"(8.65 vs 72.57)，top-10 大多噪声、天然不重叠 Swin 但≠正确。Swin 错的地方 DINO 也错。**信息天花板根本不存在**——不是融合配方问题。
experiments/cargo_cvpb/codex_review_ovli.txt:3968:experiments/overnight_innovation_log.md:176:exp324d_r32（DINOv2-base + LoRA rank32，hyy GPU1）e5 part-MaxSim：
experiments/cargo_cvpb/codex_review_ovli.txt:3969:experiments/overnight_innovation_log.md:187:- **战略转向**：FM 方向从"唯一活口/大概率 me-too"升级为"有真实正结果"。但需诚实评估**新颖性**：DINOv2+LoRA+ReID 是否 me-too（PersonViT / DINO-as-backbone territory）？competitive 到什么程度（36.72 heavy vs Swin 72.57——仍有大 gap，但这是 pose-part-MaxSim 单分支、无 PSG/GCN/全套）？下一步：等 e30 看上限 + 对照 rank16/large 看 capacity 曲线，再判断是"competitive-FM-adaptation 方法"还是"诊断性结论的正向补充"。
experiments/cargo_cvpb/codex_review_ovli.txt:3970:experiments/overnight_innovation_log.md:199:并行跑实验时启动联网研究 agent 查 "DINOv2+LoRA+pose-part-MaxSim for occluded ReID" 的先例。grounded 结论（带 paper+数字）：
experiments/cargo_cvpb/codex_review_ovli.txt:3971:experiments/overnight_innovation_log.md:203:- **pose-part + 只匹配互见部位：最成熟的轴**。PVPM (CVPR'20) 已有 visibility predictor + 只匹配可见部位；KPR (ECCV'24, SOLIDER/Swin) 是当前最强同款。我们的差异仅 **MaxSim**（ColBERT late-interaction 借来）替代固定 part-to-part 对齐——mechanism 小 delta，不是新问题定义。
experiments/cargo_cvpb/codex_review_ovli.txt:3972:experiments/overnight_innovation_log.md:209:3. **新问题框定**：CLAUDE.md 已列对方向——**common-visible support / pair comparability / reliability-aware matching**。把"互见部位 MaxSim"形式化成新匹配目标（理论+消融），不是当 scoring trick。
experiments/cargo_cvpb/codex_review_ovli.txt:3973:experiments/overnight_innovation_log.md:221:query/gallery → adapted-DINO part-MaxSim distmat（验证 = exp324d e10 44.67 all / 36.78 heavy，链路对）；
experiments/cargo_cvpb/codex_review_ovli.txt:3975:experiments/overnight_innovation_log.md:268:- **新颖性裁决 + plateau 双确认**：novelty agent 的"路线2=打平/超SOTA"对单分支 pose-part-MaxSim 已**实质不可达**（~48 all-query vs 需 ≥62）。剩可走路线只有 **(1) 机制重组 LoRA↔visibility** 或 **(3) 问题 reframe（common-visible support / reliability-aware matching，CLAUDE.md 钦定方向）**。
experiments/cargo_cvpb/codex_review_ovli.txt:3977:experiments/overnight_innovation_log.md:350:- **机制解读（关键）**：强迫 DINO-global 与 Swin-global **线性解相关**，对"模型给 query 排哪些 gallery"（part-MaxSim 排序）是**正交的**——决定检索的是 part-MaxSim over 相同可见身体部位证据，两模型受**同一份可见证据**约束而犯**同样的错**（Swin-only-r1-hit 370/989=37%，DINO 只补 8=0.81%）。global 线性相关只是排序的"装饰"，解它不改排序。
experiments/cargo_cvpb/codex_review_ovli.txt:3984:experiments/overnight_innovation_log.md:490:- **加 part-MaxSim kill-switch(防 false-KILL)**: pooled 判据可能误杀——cls 是训练检索特征, patch-pool 天生弱于它。加 token 级 part-MaxSim 版本(128 patch→6 水平 part-band, 全局 burst 权重降权 band 内 bursty patch, part-MaxSim 检索 burst vs uniform)。pretrained smoke 全跑通: pooled burst−uniform=-0.01, part-MaxSim burst−uniform=-0.03(均 KILL, 未训练应如此=harness 校准正确)。e120 自动触发用更新脚本, 一次出 pooled+part-MaxSim 双判据 + 训练模型诊断。
experiments/cargo_cvpb/codex_review_ovli.txt:3993:experiments/fm_occluded_reid_study.md:15:DINOv2 pose-anchored part-MaxSim vs Swin MaxSim，重遮挡 oracle 检查：top-10 Jaccard 0.062，**P_dino_only 0.20%**(989 query 里只 2 个 DINO 命中而 Swin 漏)，**oracle 上界仅 +0.12 mAP**。低 Jaccard 是"假正交"——DINO 不是补充，是全局太弱(8.65 vs 72.57)。Swin 错的地方 DINO 也错。→ DINO⊕Swin 融合/re-rank 家族全死（两 agent 独立确认）。
experiments/cargo_cvpb/codex_review_ovli.txt:3994:experiments/fm_occluded_reid_study.md:21:LoRA 解冻 DINOv2-base(~1M 可训：0.6M LoRA + 0.4M 头，DINO 主体冻结) + 可微 pose-part-MaxSim：重遮挡 mAP **8.65 → 36.78**(e10，×4.2)，全部 14.61 → 44.67。rank32 36.72、large 待定。**证明瓶颈是 adaptation 不是特征**——这是 oracle"冻结无独立信息"的对照解释。
experiments/cargo_cvpb/codex_review_ovli.txt:3995:experiments/fm_occluded_reid_study.md:49:**decorr loss 全程活跃(稳 0.041)却完全没移动 Jaccard/oracle/fusion。** 机制：强迫 global 线性解相关对"排哪些 gallery"(part-MaxSim 排序)是**正交**的——检索由 part-MaxSim over 相同可见身体部位证据决定，两模型受**同一份可见证据**约束犯**同样的错**(Swin-only-r1-hit 370/989=37%，DINO 补 8=0.81%)。global 线性相关只是排序"装饰"。→ **显式施压也打不破张力 = 张力鲁棒、fundamental**。这是张力洞察最强的对照证据(诊断论文核心实验)。
experiments/cargo_cvpb/codex_review_ovli.txt:3998:experiments/fm_occluded_reid_study.md:60:2. **×4 adaptation 发现**："冻结基础模型对 occluded ReID 无用，但 ~1M 参数 pose-anchored adaptation 解锁 ×4"——bottleneck-is-adaptation 的干净论点（Codex 称 LoRA-DINO+可微 pose-part-MaxSim 组合无直接先例）。
experiments/cargo_cvpb/codex_review_ovli.txt:4034:experiments/exp259/monitor.md:89:**R1 83.7 = exp255 MaxSim R1!** OA-SD=2.0 的 equal_concat R1 已达 exp255 MaxSim 水平!
experiments/cargo_cvpb/codex_review_ovli.txt:4035:experiments/exp259/monitor.md:106:**R1 83.8 matches exp255 MaxSim!** OA-SD=2.0 R1 improvement confirmed stable.
experiments/cargo_cvpb/codex_review_ovli.txt:4037:experiments/exp259/monitor.md:119:**MaxSim: 73.6/83.7** (vs exp255 73.5/83.8 = +0.1/-0.1). 持平。OA-SD=2.0 中性。
experiments/cargo_cvpb/codex_review_ovli.txt:4045:experiments/exp259/design.md:11:exp255 (WD=1e-4, OA-SD=1.0): 73.2/83.3, MaxSim 73.5/83.8
experiments/cargo_cvpb/codex_review_ovli.txt:4112:experiments/paper_notes/2026-04-15_prcv_reset.md:10:4. `GCN` 虽然也属于 pose 信息利用，但应统一写成 **structural pose branch**；`LGPA-D / OA-SD / MaxSim / POT / flip-test` 仍作为 supporting assets，不再抢主创新位置
experiments/cargo_cvpb/codex_review_ovli.txt:4114:experiments/paper_notes/2026-04-15_prcv_reset.md:108:   - `GCN / LGPA-D / OA-SD / MaxSim`
experiments/cargo_cvpb/codex_review_ovli.txt:4115:experiments/paper_notes/2026-04-15_prcv_reset.md:201:- `LGPA-D / GCN / OA-SD / MaxSim` = system assets / supporting modules
experiments/cargo_cvpb/codex_review_ovli.txt:4120:experiments/exp324b/design.md:9:- exp324（frozen，无训练）已证：用姿态把 DINOv2 dense token 锚定到 5 个身体部位、跨图只比双方可见部位（part-MaxSim），在 Occluded-Duke 重遮挡 query 上 mAP 1.86 vs 整图 0.55（×3.4），且**均匀网格对照只 0.67（几乎不涨）→ 涨点几乎全来自"姿态锚定"机制本身**（单变量隔离干净）。
experiments/cargo_cvpb/codex_review_ovli.txt:4121:experiments/exp324b/design.md:15:冻结 DINOv2-base、**仅训一个轻量 per-part 投影头**（先线性，必要时小 MLP），用 ID 分类 + triplet loss 训练、用 mutually-visible part-MaxSim 匹配，能把重遮挡 mAP 从 1.86 **大幅拉高**（先看能否进入"几十分"量级证明冻结特征有救），同时保留"姿态锚定 + 只比可见部位"的机制优势。
experiments/cargo_cvpb/codex_review_ovli.txt:4122:experiments/exp324b/design.md:19:- **数据流**：图 → 冻结 DINOv2-base dense tokens → 姿态锚定 5 部位池化 + per-part visibility（exp324 已实现）→ **轻量共享线性投影头** 768→512 + BNNeck → 全局特征 = 可见部位投影向量的 masked mean；测试用 mutually-visible part-MaxSim 出 distmat。
experiments/cargo_cvpb/codex_review_ovli.txt:4128:experiments/paper_materials/story.md:20:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/cargo_cvpb/codex_review_ovli.txt:4164:experiments/exp324b/claude_review.md:3:实验：冻结 DINOv2-base + 轻量共享线性投影头 + 姿态部位匹配，训练把 exp324 frozen 的 part-MaxSim 信号拉上去。
experiments/cargo_cvpb/codex_review_ovli.txt:4207:experiments/exp324/design.md:1:# 实验 exp324: DINOv2 Emergent Correspondence — pose-anchored part-MaxSim 廉价首验
experiments/cargo_cvpb/codex_review_ovli.txt:4208:experiments/exp324/design.md:11:- **我们的咬合**：MaxSim 天然吃 DINO dense tokens（late-interaction，非全局池化）；pose+5-part 把 DINO 在低分辨率人物上 noisy 的 correspondence 约束到 body-part 语义上降噪；只比 mutually-visible part 对症遮挡。24G 最稳、不依赖大模型、最复用现成 MaxSim 全套。
experiments/cargo_cvpb/codex_review_ovli.txt:4209:experiments/exp324/design.md:15:frozen DINOv2-B dense patch tokens，按 pose 锚定成 5 个 body-part 表征（带 per-part visibility），跨图只比 mutually-visible part 的 part-MaxSim，在**重遮挡子集**上**超过 holistic 基线**（DINO 全局 cosine 以及/或 exp255 holistic）——全程 training-free。
experiments/cargo_cvpb/codex_review_ovli.txt:4211:experiments/exp324/design.md:23:5. **对照基线**：① DINO 全局 cls/mean-pool cosine（holistic DINO）；② exp255 holistic（equal_concat）；③ DINO part-MaxSim **不带 pose 锚定**（均匀网格 part，证 pose 锚定的必要）。
experiments/cargo_cvpb/codex_review_ovli.txt:4212:experiments/exp324/design.md:27:- 假设成立：DINO pose-anchored part-MaxSim 在重遮挡子集超 holistic DINO + 接近/超 exp255，且 pose 锚定 > 均匀网格 → emergent correspondence + pose 降噪有效 → 上轻量 part-projection 头训练。
experiments/cargo_cvpb/codex_review_ovli.txt:4213:experiments/exp324/design.md:28:- 失败最可能原因：(1) DINO correspondence 在 128×256 低分辨率脏 crop 上漂移严重，pose 降噪不够；(2) part-MaxSim 不带训练直接 eval 太弱（DINO 特征非 ReID-judiciable）；(3) 只在整体涨、重遮挡组不涨。任一即降级。
experiments/cargo_cvpb/codex_review_ovli.txt:4214:experiments/exp324/design.md:32:- baseline = holistic DINO 全局 cosine（无 part、无 pose）。treatment = pose-anchored part-MaxSim。
experiments/cargo_cvpb/codex_review_ovli.txt:4226:experiments/exp324/monitor.md:50:- **机制有明确相对信号**：frozen DINOv2 dense token + pose 锚定 5-part + mutually-visible part-MaxSim 在重遮挡子集**显著超过整图基准**（mAP/R1 提升 3-4 倍），且 **pose 锚定 >> 均匀网格**——证明涨点来自"姿态把 token 约束到身体部位语义"，不是部位分解本身的 trivial 效果。
experiments/cargo_cvpb/codex_review_ovli.txt:4246:experiments/results.md:1212:**exp244 MaxSim test**: 66.0/76.4/87.2/90.5 (MaxSim hybrid on LGPA-D+OA-SD ep120)
experiments/cargo_cvpb/codex_review_ovli.txt:4247:experiments/results.md:1225:**exp245g MaxSim test**: 71.9/82.2/91.0/92.8 (MaxSim hybrid on Small LGPA-D+OA-SD ep120)
experiments/cargo_cvpb/codex_review_ovli.txt:4259:experiments/results.md:1315:- **exp249 是项目新最佳: 73.3/83.2 (Small LGPA-D+GCN+OA-SD MaxSim)**
experiments/cargo_cvpb/codex_review_ovli.txt:4276:experiments/results.md:1466:| exp259b MaxSim+flip | 75.1% | 85.4% | — | OA-SD=2.0, 略低于 exp255 (-0.1/-0.2) |
experiments/cargo_cvpb/codex_review_ovli.txt:4303:experiments/results.md:1652:### exp324: frozen DINOv2 emergent correspondence + pose-anchored part-MaxSim（inference-only，非训练）
experiments/cargo_cvpb/codex_review_ovli.txt:4304:experiments/results.md:1654:> post-PRCV「搬范式」#2 路线。frozen DINOv2-base（lab-3090-d, RTX 3090），全量 Occluded-Duke（2210 query × 17661 gallery，无后处理、无训练）。脚本 `scripts/exp324_dino.py`。输入 224W×448H → patch grid 32×16。keypoints 缩放到 grid → 每部位 3×3 窗均值池化成 5 个 part 向量 + per-part visibility，跨图只比 mutually-visible part 的 per-part cosine（part-MaxSim）。重遮挡子集 = query visibility_binary.sum()≤8（989/2210）。**training-free，不计入主表增益。**
experiments/cargo_cvpb/codex_review_ovli.txt:4307:experiments/results.md:1669:### exp327: 更强/更新冻结对应特征源（DINOv2-with-registers）— pose-part-MaxSim training-free 天花板 check（inference-only）
experiments/cargo_cvpb/codex_review_ovli.txt:4308:experiments/results.md:1671:> 同 exp324 pipeline（pose 锚定 5-part + mutually-visible part-MaxSim + 重遮挡 vis.sum()≤8），**唯一变量=特征源**。hyy GPU1（5060 Ti），slim pose data（剥 heatmap，数值与 exp324 一致）。脚本 `scripts/exp327_dinov3.py`。**training-free，不计入主表增益。** DINOv3-vitb16 gated（hf-mirror 需 token）下不了，改用 ungated 的 `facebook/dinov2-with-registers-base`（registers 去 high-norm artifact token，更干净 dense 特征，patch14 grid 32×16，nreg=4）。
experiments/cargo_cvpb/codex_review_ovli.txt:4314:experiments/results.md:1701:> exp324d = LoRA 解冻 DINOv2-base/large + 可微 pose-part-MaxSim（破 exp324b 冻结天花板）。exp324i = 在其上加跨网络跨协方差解相关损失（逼 DINO-global 与 frozen-Swin-global 线性无关）。Occluded-Duke，BS=64，rank16 除非标注。**单分支 part-MaxSim = 纯模型；fusion(⊕Swin) = test-time 后处理(NFC 级)，不计训练端增益。**
experiments/cargo_cvpb/codex_review_ovli.txt:4326:/bin/zsh -lc "sed -n '1,240p' experiments/cargo_cvpb/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:4328:/bin/zsh -lc "sed -n '1,260p' experiments/cargo_cvpb/claude_review.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:4541:/bin/zsh -lc "sed -n '1,260p' experiments/cargo_cvpb/codex_review_ovli.txt" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:4546:workdir: /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:4560:/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:4572:| 000 | Baseline (SOLIDER-Swin-Tiny, SW=0.2) | 56.6% | 66.5% | 79.4% | 83.4% | — | 120 epoch, 完美复现 |
experiments/cargo_cvpb/codex_review_ovli.txt:4722:| 142-eq | PSG+GCN+SKC (Support-Supervised Keypoint Completion, eq) | 60.3% | 71.8% | 84.4% | 87.7% | vs exp030a-eq: -0.8%/-1.9% | ❌ 中性偏负。completion module 虽然活跃（gate=0.26, delta_norm=1.5），但 skc_pre≈skc_post 说明修改方向不是向 prototype 靠近。gate 无限制增长导致后期过度修改特征。feature-level completion 方向已被多轮验证为无效 |
experiments/cargo_cvpb/codex_review_ovli.txt:4727:| 148-eq | PSG+GCN+PCVT (Pose-Complementary View Training, eq) | ~59.3%* | ~71.3%* | — | — | ❌ 负面。*ep100 数据，训练中。早期加速（ep30: +2.4 mAP）但后期被基线追平并反超。3-view 训练的 1/3 主损失稀释导致后期收敛不足。训练集 95.8% 全可见使 complementary masking 缺乏信号 |
experiments/cargo_cvpb/codex_review_ovli.txt:4769:### Phase 4: SupCon + OA-SD + Parallel Aug (exp166-193)
experiments/cargo_cvpb/codex_review_ovli.txt:4774:| 166r | ↳ base arch (no PAPE/MS-PSG) | 60.3% | 72.8% | — | — | CE base for OA-SD comparison |
experiments/cargo_cvpb/codex_review_ovli.txt:4776:| **187** | **+SupCon T=0.05 + 3-view Parallel Aug** | **64.9%** | **76.6%** | **87.2%** | **90.0%** | **🟢🟢 Overall best! +1.8/+2.7 vs exp166** |
experiments/cargo_cvpb/codex_review_ovli.txt:4777:| **190** | **3-view Parallel Aug + CE (no SupCon)** | **64.2%** | **75.6%** | **86.2%** | **89.1%** | **✅ 3-view+CE ≈ 1-view+SupCon! +1.1/+1.7 vs exp166** |
experiments/cargo_cvpb/codex_review_ovli.txt:4778:| **191** | **OA-SD + CE (1-view, decay=0.999)** | **63.2%** | **75.4%** | **86.3%** | **89.1%** | **✅ OA-SD 独立有效! +2.9/+2.6 vs CE base** |
experiments/cargo_cvpb/codex_review_ovli.txt:4779:| 192 | OA-SD + CE (1-view, decay=0.99) | 62.6% | 74.9% | 86.0% | 89.1% | 🟡 vs exp191: -0.6/-0.5。decay 不敏感 |
experiments/cargo_cvpb/codex_review_ovli.txt:4780:| **193** | **OA-SD + 3-view + CE** | **64.4%** | **76.5%** | **86.3%** | **89.4%** | **✅ OA-SD+3-view additive! R1 ≈ exp187 SupCon, mAP +0.2/R1+0.9 vs exp190** |
experiments/cargo_cvpb/codex_review_ovli.txt:4781:| 194 | OA-SD + CE (weight=2.0) | 63.4% | 74.8% | 86.1% | 89.1% | 🟡 vs exp191 (w=1.0): +0.2/-0.6。weight 不敏感 |
experiments/cargo_cvpb/codex_review_ovli.txt:4784:/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:4792:**选择**: 从 SOLIDER 作者原始代码重新开始，纯 pose heatmap 方向
experiments/cargo_cvpb/codex_review_ovli.txt:4805:/bin/zsh -lc "sed -n '1,240p' experiments/cargo_cvpb/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:4839:cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/codex_review_ovli.txt:4841:  --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/codex_review_ovli.txt:4842:  --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
experiments/cargo_cvpb/codex_review_ovli.txt:4862:cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/codex_review_ovli.txt:4864:  --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/codex_review_ovli.txt:4865:  --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/codex_review_ovli.txt:4901:/bin/zsh -lc "nl -ba experiments/afd_reid/afd_model.py | sed -n '1,90p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:4995:/bin/zsh -lc "rg -n \"def weights_init_kaiming|Conv|BatchNorm|Linear\" experiments/afd_reid/afd_model.py experiments -g '*.py'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:5032:/bin/zsh -lc pwd in /Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:5034:/Users/abdslm/Desktop/SOLIDER-REID
experiments/cargo_cvpb/codex_review_ovli.txt:5047:Static review found no Critical/High blocker. The key requirement passes: OVLI projection params are added via `list(model.parameters()) + list(ovli.parameters())`, and the optimizer assert checks `ovli.proj.parameters()` are present: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:559).
experiments/cargo_cvpb/codex_review_ovli.txt:5053:- **Medium:** OVLI loss can silently have reduced effective signal because `RandomIdentitySampler` samples `K` images per PID but does not force opposite-view composition: [cargo_dataset.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/cargo_dataset.py:226). The loss correctly skips anchors without both opposite-view positive and negative candidates: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:316), but logs only loss/pos/neg/gap: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:677). Add `valid_anchor_frac`, `pos_per_valid`, `neg_per_valid` before long training so low OVLI loss is not confused with “no valid anchors.”
experiments/cargo_cvpb/codex_review_ovli.txt:5059:- Token path is shape/device safe: `layer4` is 2048 channels in the model, OVLI hooks it, adaptive-pools to grid, applies `Conv2d(2048 -> ovli_dim)`, flattens to `(B,K,D)`, and L2-normalizes per token: [afd_model.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_model.py:228), [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:248).
experiments/cargo_cvpb/codex_review_ovli.txt:5060:- AMP/numerics are handled correctly: cached fmap may be fp16, but OVLI token projection/loss run with autocast disabled and explicit `.float()`: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:619).
experiments/cargo_cvpb/codex_review_ovli.txt:5061:- MaxSim/logsumexp path is NaN-safe: symmetric bidirectional mean-max is correct, invalid logits use finite `-1e4` floor, and valid rows are selected after logsumexp: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:270), [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:322).
experiments/cargo_cvpb/codex_review_ovli.txt:5062:- Eval behavior is clean: default eval remains `run_cross_view_eval` global-only; `--ovli_rerank` only adds optional global+MaxSim reporting: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:687).
experiments/cargo_cvpb/codex_review_ovli.txt:5063:- `--ovli` off does not construct the head/hook and optimizer stays model-only. `--ovp/--ovli` are mutually exclusive: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:503).
experiments/cargo_cvpb/codex_review_ovli.txt:5082:Static review found no Critical/High blocker. The key requirement passes: OVLI projection params are added via `list(model.parameters()) + list(ovli.parameters())`, and the optimizer assert checks `ovli.proj.parameters()` are present: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:559).
experiments/cargo_cvpb/codex_review_ovli.txt:5088:- **Medium:** OVLI loss can silently have reduced effective signal because `RandomIdentitySampler` samples `K` images per PID but does not force opposite-view composition: [cargo_dataset.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/cargo_dataset.py:226). The loss correctly skips anchors without both opposite-view positive and negative candidates: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:316), but logs only loss/pos/neg/gap: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:677). Add `valid_anchor_frac`, `pos_per_valid`, `neg_per_valid` before long training so low OVLI loss is not confused with “no valid anchors.”
experiments/cargo_cvpb/codex_review_ovli.txt:5094:- Token path is shape/device safe: `layer4` is 2048 channels in the model, OVLI hooks it, adaptive-pools to grid, applies `Conv2d(2048 -> ovli_dim)`, flattens to `(B,K,D)`, and L2-normalizes per token: [afd_model.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_model.py:228), [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:248).
experiments/cargo_cvpb/codex_review_ovli.txt:5095:- AMP/numerics are handled correctly: cached fmap may be fp16, but OVLI token projection/loss run with autocast disabled and explicit `.float()`: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:619).
experiments/cargo_cvpb/codex_review_ovli.txt:5096:- MaxSim/logsumexp path is NaN-safe: symmetric bidirectional mean-max is correct, invalid logits use finite `-1e4` floor, and valid rows are selected after logsumexp: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:270), [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:322).
experiments/cargo_cvpb/codex_review_ovli.txt:5097:- Eval behavior is clean: default eval remains `run_cross_view_eval` global-only; `--ovli_rerank` only adds optional global+MaxSim reporting: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:687).
experiments/cargo_cvpb/codex_review_ovli.txt:5098:- `--ovli` off does not construct the head/hook and optimizer stays model-only. `--ovp/--ovli` are mutually exclusive: [afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py:503).
experiments/cargo_cvpb/airl_main_table.md:10:| **baseline-Swin**(我方,同 backbone) | SOLIDER-Swin | **60.84** | — | 强基线(团队资产) |
experiments/cargo_cvpb/airl_main_table.md:11:| **AIRL iso (fixed w=0.25)** | SOLIDER-Swin | **63.21** | — | **+2.37 over baseline** |
experiments/cargo_cvpb/airl_main_table.md:12:| **AIRL iso (rec 单头)** | SOLIDER-Swin | 63.38 | — | +2.54 |
experiments/cargo_cvpb/airl_main_table.md:13:| **AIRL iso (方向感知融合)** | SOLIDER-Swin | **~64.60** | — | **+3.76**,待 val-tuned w 实现 |
experiments/cargo_cvpb/diag_swin_ckpt.py:11:        --ckpt /root/work/SOLIDER-REID/log/cargo/cvpb_swin_ovli/model_best.pth \
experiments/cargo_cvpb/diag_swin_ckpt.py:12:        --swin_pretrain /root/work/SOLIDER-REID/pretrained/swin_small.pth \
experiments/cargo_cvpb/diag_swin_ckpt.py:13:        --data_root /root/work/SOLIDER-REID/data
experiments/cargo_cvpb/diag_swin_ckpt.py:35:    ap.add_argument('--swin_pretrain', default='')
experiments/cargo_cvpb/diag_swin_ckpt.py:36:    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/cargo_cvpb/diag_swin_ckpt.py:50:                        args=_args('swin_small', args.swin_pretrain, img_size, 0.2)).to(device)
experiments/exp294_lgpaOnly_2stg_b_od_s41/claude_review.md:29:  - OA-SD: True, WEIGHT=1.0, EMA=0.999
experiments/exp294_lgpaOnly_2stg_b_od_s41/claude_review.md:58:OUTPUT_DIR /home/afr/SOLIDER-REID/log/.../exp294_...     ← 独立输出目录
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:2:"""Lattice-Marginalized ReID — ZERO-TRAINING kill-switch  (Market, frozen SOLIDER).
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:43:    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:76:ap.add_argument('--semantic_weight', type=float, default=-1.0, help='override MODEL.SEMANTIC_WEIGHT to match ckpt training (MSMT17 swin ckpt trained sw=0.6 but config has 0.2 -> SOLIDER backbone feature mismatch). -1=use config.')
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:77:ap.add_argument('--adaptive_k', action='store_true', help='supporting: per-query phase-volatility selects K (high-vol query marginalize over K, low-vol use K=1). Reduces avg compute keeping most marginalization gain -> rebut "K=9 too expensive".')
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:265:# frozen model (pose DISABLED at test: pose_dict=None -> plain SOLIDER global feat)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:282:        if cli.semantic_weight >= 0:               # match ckpt training (SOLIDER semantic-aware backbone)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:548:        # ---- adaptive-K (supporting): per-query phase volatility -> spend K only where it helps ----
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:9:BUT that union used CROSS-CAMERA same-ID images (oracle, NOT deployable = the exp109 wall).
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:14:ORACLE has headroom and the realizable same-camera tracklet does not -> exp109 query-side
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:28:  B_oracle     : union a CROSS-VIDEO same-ID *gallery* image — NOT deployable (exp109 upper bound).
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:33:  Fragility gate : fuse only when lowtail-positive-support is weak (fragile) vs fuse-all.
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:37:has headroom, realizable cannot -> exp109 query-side wall (3/10, DEAD). Be honest.
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:40:  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:61:ap.add_argument('--a_temp', type=float, default=20.0, help='soft-min temp for lowtail positive support')
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:65:                help='fragility gate: among failures, bottom-q lowtail-support = "fragile" (fuse only these)')
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:303:    # lowtail positive support of each single frame (cross-VIDEO positives in gallery only;
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:305:    # support meter reflects deployable cross-camera identity evidence, not same-video repeats.
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:329:    # ARM B (ORACLE):     union a cross-video same-ID GALLERY image — exp109 upper bound
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:384:          f"(dAP={ (Bo[okB].mean()-base_a[okB].mean())*100:+.3f})  [NOT deployable; exp109 upper bound]")
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:431:    # Fragility gate: fuse only FRAGILE (weak lowtail support) failures vs fuse-all.
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:432:    # Among the used failures, split by lowtail support (computed earlier per single_rows row).
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:435:    print(f"[gate] fragility-weighted (fuse only weak-support failures) vs fuse-all")
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:440:        fragile = okg & (lt_used <= thr)               # weakest support = most fragile
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:446:        print(f"     fuse-FRAGILE-only (bottom-{cli.frag_quant:.0%} support) dAP={dA_frag:+.3f} "
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:451:        print(f"     [too few finite-support rows to gate: {int(okg.sum())}]")
experiments/cargo_cvpb/cvpb_realizability_killswitch.py:469:        print(f"\n  [{'LIVE 7/10' if live else 'DEAD 3/10 — exp109 query-side wall'}] "
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:14:  * the d14 "Evidence-Sufficient ReID" backup (single-image support insufficiency,
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:19:  TEST 1 (positive-support explains the TAX residual):
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:25:      Does positive-support explain the gallery-growth tax residual (the 1x->10x AP
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:28:      positive-support must show different tax (partial Spearman survives controls).
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:30:  TEST 2 (positive-support predicts per-query FAILURE):
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:31:      ROC-AUC of positive-support predicting per-query AP-failure on the FULL gallery,
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:36:      For LOW positive-support FAILURE queries, add a 2nd same-ID image and form an
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:44:      -> the failure is EVIDENCE INSUFFICIENCY (more evidence fixes it), supporting d14.
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:46:VERDICT: positive-support has an INDEPENDENT-of-trivial signal (survives TEST1 partial
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:51:  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:81:ap.add_argument('--low_support_quant', type=float, default=0.30,
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:82:                help='bottom-q by positive-support among FAILURE queries -> the rescue target set')
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:281:def positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, a_temp):
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:394:    ps = positive_support(cqf, cq_pid, cq_cam, gf[core_idx], g_pid[core_idx], g_cam[core_idx], cli.a_temp)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:395:    # weak-positive RISK convention: higher = weaker support = predict bigger tax
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:415:    print(f"\n[1] raw Spearman(positive-support risk, tax) over {int(ev.sum())} valid core queries:")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:422:    # ★ LIFE-OR-DEATH partials: positive-support vs tax controlling 1x-margin AND #false
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:424:    print(f"\n[1] ★PARTIAL Spearman(support-risk, tax | 1x-top1-margin + #false-in-topk):")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:431:    # reverse direction (do the trivials survive controlling support? — fairness check)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:435:    print(f"     [reverse] 1x-margin | support  = {pr_m:+.4f}   #false | support = {pr_f:+.4f}")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:437:    # combined: does support add to a logistic predicting big-tax over trivials? (OOF AUC)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:451:    print(f"\n[1] big-tax (top-30% tax) OOF-AUC: trivials={a_triv:.4f}  +support={a_both:.4f}  "
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:452:          f"support-solo={a_supp:.4f}  >> INCREMENT={a_both-a_triv:+.4f}")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:482:    # positive-support on the FULL gallery (cross-cam positives only)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:483:    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:488:        'lowtail-pos(soft-min)': -ps['lowtail'],       # ★ support: weak = high risk
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:489:        'pos-dispersion':         ps['disp'],          # ★ support
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:490:        '#cross-cam-pos(neg)':   -ps['ncc'],           # ★ support
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:530:    print(f"     trivials + support          OOF-AUC = {a_both:.4f}")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:531:    print(f"     support-only (3 proxies)    OOF-AUC = {a_supp:.4f}")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:532:    print(f"     >> INCREMENT support adds on top of trivials = {a_both-a_triv:+.4f}")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:533:    print(f"     >> best support AUC - best trivial AUC        = {best_supp-best_triv:+.4f}")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:534:    # partial spearman: best support var vs continuous (-AP) controlling all trivials
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:609:    # full-gallery per-query AP + failure + positive-support
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:612:    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:613:    support = ps['lowtail']                             # higher = stronger support
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:615:    # FAILURE = bottom-30% AP; among failures, LOW-SUPPORT = bottom-q by support
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:621:    # low-support subset among failures (need >=2 same-ID query imgs to do union -> see below)
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:622:    supp_fail = support[fail_idx]
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:625:    nlow = int(round(cli.low_support_quant * len(fidx2)))
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:626:    low_order = np.argsort(supp2)                       # weakest support first
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:627:    low_support_fail = fidx2[low_order[:nlow]]
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:629:          f"low-support failures(bot-{cli.low_support_quant:.0%})={len(low_support_fail)}")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:631:    # for each low-support failure query, we need a SECOND same-ID query image to union.
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:658:    for qi in low_support_fail:
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:684:    print(f"\n[3] oracle multi-query on {n} low-support failure queries (mean AP, %):")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:699:    # then index out our selected low-support failure rows. We compare against base AP computed
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:703:          f"lam={cli.krecip_lambda}) then index the SAME {n} low-support failure queries:")
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:751:    print(f"[T2] failure-AUC: best support-trivial gap={T2['best_supp_minus_triv']:+.3f}  "
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:752:          f"OOF incr={T2['incr']:+.3f}  support-solo AUC={T2['supp_solo']:.3f}  "
experiments/cargo_cvpb/cvpb_evidence_killswitch.py:754:    print(f"[T3] oracle on n={T3['n']} low-support failures: base={T3['base']*100:.2f} -> "
experiments/cargo_cvpb/monitor_agreid_v2.md:5:- backbone: Swin-Small + SOLIDER pretrain (swin_small.pth), 256×128, bs64, 60ep, lr3.5e-4(backbone×0.1)
experiments/cargo_cvpb/monitor_agreid_v2.md:6:- baseline: `--dataset agreid_v2 --backbone swin_small --swin_pretrain ...` (无 AIRL)
experiments/cargo_cvpb/monitor_agreid_v2.md:29:- 启动验证:config use_afd=False 全 AIRL 关(纯 baseline);SOLIDER pretrain All keys matched;
experiments/exp130/design.md:24:3. 那么这才是对 “support-complete 新增 correction 是否被完整 target 稀释” 的有效检验
experiments/exp130/design.md:35:- online support-complete teacher 不 freeze
experiments/cargo_cvpb/airl_iso_result.md:60:Swin-Small + SOLIDER pretrain,256×128,bs64,60ep。lab-3090(4090 数据上传链路死,顺序跑)。
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:8:    operator mistakes) whose purity is unknown. People assume "more query support =
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:34:    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
experiments/cargo_cvpb/cvpb_querybag_killswitch.py:497:    # sanity: single-image standard mAP (anchor only) for reference
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:75:[C] P(tail-probe NN is a HEAD of support s) by support bin (DESCRIPTIVE, n=4 bins):
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:81:  [per-IMAGE rate FLAT across support -> purely mechanical count; RISING -> heads over-attract disproportionately.]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:82:  binned Spearman(support, rate/headID)=+0.4000 (trivially >0)  rate/IMAGE=-0.8000  [n=4 bins, descriptive only]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:85:     Spearman(support, attraction-count)     = +0.0428  [trivially >0: more imgs = more NN tickets]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:86:     Spearman(support, attraction-PER-IMAGE) = -0.0127  [NON-TRIVIAL: >0 means heads over-attract beyond count; ~0 means purely count]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:88:[C] support-calibrated vs global threshold (OVERALL tail->head false-merge at matched head-recall), CAL=even seeds / EVAL=odd seeds:
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:89:  head-recall=0.90: OVERALL false-merge  global=0.0373  support-calibrated=0.0373  (d=+0.0000; want NEGATIVE)  [5 eval seeds, 1500 tail probes]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:90:  head-recall=0.95: OVERALL false-merge  global=0.0853  support-calibrated=0.0840  (d=-0.0013; want NEGATIVE)  [5 eval seeds, 1500 tail probes]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:91:  support-level fallback-to-global fraction (sparse levels) = 0.000  [high -> 'support-calibrated' is mostly global]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_market.log:99:[C] per-head-ID Spearman(support, attraction)=+0.043 (trivial)  PER-IMAGE=-0.013 (non-trivial claim)  [n=450 IDs]
experiments/cargo_cvpb/airl_gate_oracle.py:38:    cd /home/afr/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/airl_gate_oracle.py:40:        --base_ckpt /home/afr/SOLIDER-REID/log/cargo/cvpb_swin_baseline256/model_best.pth \
experiments/cargo_cvpb/airl_gate_oracle.py:41:        --airl_ckpt /home/afr/SOLIDER-REID/log/cargo/cvpb_airl_swin2/model_best.pth \
experiments/cargo_cvpb/airl_gate_oracle.py:42:        --swin_pretrain /home/afr/SOLIDER-REID/pretrained/swin_small.pth \
experiments/cargo_cvpb/airl_gate_oracle.py:43:        --data_root /home/afr/SOLIDER-REID/data \
experiments/cargo_cvpb/airl_gate_oracle.py:68:def _args(backbone, swin_pretrain, img_size, semantic_weight=0.2):
experiments/cargo_cvpb/airl_gate_oracle.py:79:    ns.swin_pretrain = swin_pretrain
experiments/cargo_cvpb/airl_gate_oracle.py:103:def load_model(ckpt, backbone, swin_pretrain, img_size, num_classes, device,
experiments/cargo_cvpb/airl_gate_oracle.py:106:                        args=_args(backbone, swin_pretrain, img_size)).to(device)
experiments/cargo_cvpb/airl_gate_oracle.py:237:    ap.add_argument('--swin_pretrain', default='')
experiments/cargo_cvpb/airl_gate_oracle.py:238:    ap.add_argument('--data_root', default='/home/afr/SOLIDER-REID/data')
experiments/cargo_cvpb/airl_gate_oracle.py:274:    base = load_model(args.base_ckpt, args.backbone, args.swin_pretrain,
experiments/cargo_cvpb/airl_gate_oracle.py:276:    airl = load_model(args.airl_ckpt, args.backbone, args.swin_pretrain,
experiments/exp175/claude_review.md:14:**Is this just a small config change?**: This is a legitimate ablation/extension experiment. The multi-stage PSG code already exists (validated in exp173 with stages [2,3]). Extending to [0,1,2,3] is a valid ablation to answer "does full-stage PSG beat partial PSG and PAPE?" This is a fine experiment as a supporting/ablation result.
experiments/exp130/monitor.md:43:  1. 已在 `processor.py` 补上 `residual_kl requires support teacher` 的保护
experiments/exp130/monitor.md:55:  2. support-complete teacher 仍为在线版本：
experiments/exp130/monitor.md:185:     - 至少在 `delta_top + online support teacher` 这条线上，完整 teacher target 比 `residual_kl` 更有效
experiments/exp130/monitor.md:205:  1. 已在 `processor.py` 中补上 `residual_kl requires support teacher` 的保护
experiments/cargo_cvpb/claude_review_agreid_v2.md:47:唯一变量 = AIRL on/off。两实验同机(尽量)、同 Swin pretrain、同 256×128/bs64/60ep。AIRL-iso 用
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:75:[C] P(tail-probe NN is a HEAD of support s) by support bin (DESCRIPTIVE, n=4 bins):
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:81:  [per-IMAGE rate FLAT across support -> purely mechanical count; RISING -> heads over-attract disproportionately.]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:82:  binned Spearman(support, rate/headID)=+0.4000 (trivially >0)  rate/IMAGE=-0.4000  [n=4 bins, descriptive only]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:85:     Spearman(support, attraction-count)     = +0.0590  [trivially >0: more imgs = more NN tickets]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:86:     Spearman(support, attraction-PER-IMAGE) = -0.0093  [NON-TRIVIAL: >0 means heads over-attract beyond count; ~0 means purely count]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:88:[C] support-calibrated vs global threshold (OVERALL tail->head false-merge at matched head-recall), CAL=even seeds / EVAL=odd seeds:
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:89:  head-recall=0.90: OVERALL false-merge  global=0.2490  support-calibrated=0.2481  (d=-0.0010; want NEGATIVE)  [5 eval seeds, 1040 tail probes]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:90:  head-recall=0.95: OVERALL false-merge  global=0.5654  support-calibrated=0.5519  (d=-0.0135; want NEGATIVE)  [5 eval seeds, 1040 tail probes]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:91:  support-level fallback-to-global fraction (sparse levels) = 0.143  [high -> 'support-calibrated' is mostly global]
experiments/cargo_cvpb/gallery_logs/cvpb_gallery_oduke.log:99:[C] per-head-ID Spearman(support, attraction)=+0.059 (trivial)  PER-IMAGE=-0.009 (non-trivial claim)  [n=311 IDs]
experiments/cargo_cvpb/cvpb_gopl_killswitch.py:28:    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
experiments/exp130/claude_review.md:3:### 1. MEDIUM — Missing validation for `residual_kl` + no support teacher
experiments/exp130/claude_review.md:7:if csrd_target_mode == 'residual' and not csrd_support_teacher:
experiments/exp130/claude_review.md:10:but no equivalent guard for `'residual_kl'`. Without a support teacher, `dist_t == dist_base`, making all teacher residual logits exactly zero → uniform teacher distribution → KL loss pushes student to uniform → actively harmful.
experiments/exp130/claude_review.md:86:| Missing validation for residual_kl without support teacher | Medium | Low (config is correct) | Add one-line guard after launch |
experiments/exp130/claude_review.md:96:   if csrd_target_mode == 'residual_kl' and not csrd_support_teacher:
experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py:37:  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
experiments/cargo_cvpb/cvpb_containment_killswitch.py:28:    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb && \
experiments/cargo_cvpb/cvpb_containment_killswitch.py:30:      --ckpt /root/work/SOLIDER-REID/log/cargo/cvpb_swin_fix256/model_best.pth \
experiments/cargo_cvpb/cvpb_containment_killswitch.py:31:      --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/cvpb_containment_killswitch.py:32:      --swin_pretrain /root/work/SOLIDER-REID/pretrained/swin_small.pth \
experiments/cargo_cvpb/cvpb_containment_killswitch.py:48:ap.add_argument('--ckpt', default='/root/work/SOLIDER-REID/log/cargo/cvpb_swin_fix256/model_best.pth')
experiments/cargo_cvpb/cvpb_containment_killswitch.py:49:ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/cargo_cvpb/cvpb_containment_killswitch.py:50:ap.add_argument('--swin_pretrain', default='/root/work/SOLIDER-REID/pretrained/swin_small.pth')
experiments/cargo_cvpb/cvpb_containment_killswitch.py:73:    backbone='swin_small', swin_pretrain=cli.swin_pretrain, swin_semantic_weight=0.2,
experiments/cargo_cvpb/cvpb_osac_killswitch.py:48:    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
experiments/cargo_cvpb/claude_review_swin.md:16:弱 resnet50 都到 OVLI 52.37 超 VDT 42.76 → 换 SOLIDER Swin 强 backbone 冲 SOTA。两个涨点模块(SetVLAD/ACVP)证伪后的正确转向: 不加模块, 换 backbone。
experiments/cargo_cvpb/claude_review_swin.md:21:- **init_weights(pretrain_path)**(L113-115): SOLIDER teacher ckpt(backbone.* keys), strict=False。smoke[2] `<All keys matched successfully>`。✓
experiments/cargo_cvpb/claude_review_swin.md:37:审查通过。Swin backbone 字节级保留 resnet50 默认 + OVLI hook 适配 NCHW + grad 流到 Swin + 预训练 teacher ckpt 加载(all keys matched)+ avg-pool 强制(GeM 不破坏 LayerNorm 负值)。smoke 4 组全过(lab-3090 真 swin_small.pth)。codex 审 + GPU 空即跑 Swin OVLI(`--backbone swin_small --swin_pretrain <repo>/pretrained/swin_small.pth`,可选 `--img_size 384 128` SOLIDER 原生分辨率)。
experiments/exp341_clip_id_prompt/design.md:1:# 实验 exp341: CLIP-ReID 式可学习 ID prompt 移植到 SOLIDER（Step 1 — 找能涨的 CLIP 机制）
experiments/exp341_clip_id_prompt/design.md:9:**给 SOLIDER global 加一个 CLIP-ReID 式可学习 ID 文本原型分支（CoOp prompt + SupCon i2t/t2i 对比），让文本原型监督/正则化 global 特征 → global 涨点。**
experiments/exp341_clip_id_prompt/design.md:14:  - 投影：SOLIDER global(768) → CLIP dim(512)，可学习 Linear。
experiments/exp341_clip_id_prompt/design.md:16:- 测试描述子：仍用 SOLIDER global（不动 LGPA），看 i2t/t2i 正则有没有让 global 涨。
experiments/exp341_clip_id_prompt/design.md:20:- 失败最可能原因：SOLIDER 已 73 很强，ID 文本原型正则边际为零；或 1-stage joint 不如 2-stage 稳（prompt 与特征互相追）。若 1-stage 平，再试 2-stage（stage1 冻特征学 prompt，stage2 冻 prompt 微调）。
experiments/exp341_clip_id_prompt/design.md:23:- baseline：同 SOLIDER 配置但关 CLIP_ID_PROMPT（= 纯 global ID+triplet）。
experiments/exp341_clip_id_prompt/design.md:41:**CLIP-ReID 可学习 ID prompt（CoOp + i2t/t2i）在 SOLIDER 上真涨 +2.2**——这是死掉的固定文本（exp340 壳）做不到的。clip_id loss 8.7→2.83（prompt 学得好）。
experiments/cargo_cvpb/litreview2/validate_candidates.sh:3:OUT=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview/reviews/lit_4.md:17:- SOLIDER-Swin 强 backbone(自监督人体预训练,in_planes=768)
experiments/cargo_cvpb/litreview/reviews/lit_4.md:126:`SOLIDER-REID/` 是干净的上游代码，来源是 `github.com/tinyvision/SOLIDER-REID`，版本是 `8c08e1c`。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这个目录里从零写。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:298:## SOLIDER-REID 代码底子
experiments/cargo_cvpb/litreview/reviews/lit_4.md:302:用 SOLIDER 的预训练权重。从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:331:- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:419:## 8. 代码底子（上游 SOLIDER-REID）
experiments/cargo_cvpb/litreview/reviews/lit_4.md:422:- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:434:- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:507:**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:509:**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:511:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:538:方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:555:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:557:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:653:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:654:- **exp017 探针放行候选 A**：SOLIDER 特征对 CLIP 遮挡物语义零线性解码（R²=-0.01）、嵌入级超锥零假设解释力仅约 26%、残差与 AP 相关 -0.164（控 s_top1 后 -0.097 贴线）。按预注册中间地带规则的属性级判据放行。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:670:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:926:date of current version 7 May 2025. This work was supported in part by the
experiments/cargo_cvpb/litreview/reviews/lit_4.md:1552:images. While this approach can enhance model performance, as pretraining and fine-tuning have emerged as a new paradigm, we suggest
experiments/cargo_cvpb/litreview/reviews/lit_4.md:1677:(2023) directly fine-tuned the CLIP model, and the remarkable results demonstrated that the visual-language pretraining model has high
experiments/cargo_cvpb/litreview/reviews/lit_4.md:1707:Inspired by the success of unimodal pretraining models in Transformer-based language pretraining, such as BERT (Devlin et al.,
experiments/cargo_cvpb/litreview/reviews/lit_4.md:1788:Given that pretraining a model from scratch is too expensive, we adopt
experiments/cargo_cvpb/litreview/reviews/lit_4.md:2727:2026. This work was supported in part by the Natural Science Foundation
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3277:to ensure the quality of the synthesized data, we employ pretrained vision-language models to filter out substandard images.
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3305:This work is supported by the University of Macau Start-up Research Grant SRG2024-00002-FST and Multi-Year Research Grant MYRGGRG2024-00077-FST-UMDF
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3734:挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3736:cheap kill-switch：冻结 SOLIDER，用 SMPL 或 2D pose 把特征池化到身体部位，只比较双方共同可见的部位。若在 CARGO 或 AG-ReID.v2 上不能超过全局特征的噪声线，就停。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3739:挂靠资产：aerial-ground 数据、SOLIDER 特征、SMPL 可见性。  
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3741:cheap kill-switch：先不训练，直接用 SOLIDER 聚类，统计同身份跨高度或跨俯仰角距离是否显著大于同视角距离，再做一个几何加权的原型合并。若碎裂不随几何变化，方向判弱。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3744:挂靠资产：SOLIDER-Swin 的 768 维特征、SMPL mesh、joints 和投影基建。  
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3808:挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3810:cheap kill-switch：冻结 SOLIDER，用 SMPL 或 2D pose 把特征池化到身体部位，只比较双方共同可见的部位。若在 CARGO 或 AG-ReID.v2 上不能超过全局特征的噪声线，就停。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3813:挂靠资产：aerial-ground 数据、SOLIDER 特征、SMPL 可见性。  
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3815:cheap kill-switch：先不训练，直接用 SOLIDER 聚类，统计同身份跨高度或跨俯仰角距离是否显著大于同视角距离，再做一个几何加权的原型合并。若碎裂不随几何变化，方向判弱。
experiments/cargo_cvpb/litreview/reviews/lit_4.md:3818:挂靠资产：SOLIDER-Swin 的 768 维特征、SMPL mesh、joints 和投影基建。  
experiments/cargo_cvpb/cvpb_rma_killswitch.py:2:"""RMA-TIReID  —  ZERO-TRAINING kill-switch (frozen SOLIDER + numpy only).
experiments/cargo_cvpb/cvpb_rma_killswitch.py:8:    encoder (SOLIDER/Swin, exp260b market mAP 94.4). RSTPReid person crops are the
experiments/cargo_cvpb/cvpb_rma_killswitch.py:14:     manifold is good, same-ID images are highly retrievable in frozen SOLIDER space.
experiments/cargo_cvpb/cvpb_rma_killswitch.py:32:    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
experiments/cargo_cvpb/cvpb_rma_killswitch.py:44:_repo = os.path.abspath(os.path.join(_here, '..', '..'))   # repo root .../SOLIDER-REID
experiments/cargo_cvpb/cvpb_rma_killswitch.py:380:    print("# RMA-TIReID ZERO-TRAINING KILL-SWITCH  (frozen SOLIDER exp260b, RSTPReid)")
experiments/cargo_cvpb/cvpb_rma_killswitch.py:396:    print("TEST A — image-image identity manifold on RSTPReid test (frozen SOLIDER)")
experiments/cargo_cvpb/cvpb_rma_killswitch.py:521:    print("SUMMARY / VERDICT  (RSTPReid, frozen SOLIDER exp260b)")
experiments/exp069/design.md:12:- 这让 PAA 从 "generic pose injection" 升级为 "part-structured semantic completion"
experiments/cargo_cvpb/cvpb_gallery_killswitch_DESIGN.md:33:  是否错并入 head prototype, false-merge rate 是否随 head support 单调上升。
experiments/cargo_cvpb/cvpb_gallery_killswitch_DESIGN.md:34:- 比 GLOBAL 阈值 vs SUPPORT-CALIBRATED（按 support 分层校准）在同 head-recall 下的 tail false-merge。
experiments/cargo_cvpb/cvpb_gallery_killswitch_DESIGN.md:36:  per-image rate 若 FLAT 则纯机械, 若仍随 support 升才是非平凡 over-attraction。
experiments/cargo_cvpb/cvpb_gallery_killswitch_DESIGN.md:49:- A: 唯一变量=gallery size（注入 distractor）; B: 唯一变量=watchlist size; C: 唯一变量=head support。
experiments/exp341_clip_id_prompt/claude_review.md:18:- open_clip 接受本地路径/safetensors 作为 `pretrained`（已查证）。
experiments/exp341_clip_id_prompt/claude_review.md:71:12. **pretrained 路径**：config `POSE_CLIP_ID_PRETRAINED='/home/afr/SOLIDER-REID/clip_l14_openclip.safetensors'`（lab-4090 已确认该文件存在，1.7GB），open_clip `create_model_and_transforms(pretrained=<本地 .safetensors>)` 支持 ✓。
experiments/exp341_clip_id_prompt/claude_review.md:78:- **单变量**：对照组就是同 config 关 `POSE_CLIP_ID_PROMPT`（= 纯 global ID+triplet）。本实验所有 PSG/LGPA/GCN/OA-SD/PLBOA/parallel_aug 都显式关闭，隔离干净。✓
experiments/exp341_clip_id_prompt/claude_review.md:82:- **R1 — LR schedule 与 CoOp 不匹配**：CoOp prompt 通常用较高的**常数** LR（原论文 SGD 0.002 cosine，但 prompt 单独高 LR）。本 config 让 `cls_ctx` 跟 SOLIDER backbone 同一套（BASE_LR=0.0008，20 epoch warmup→cosine 衰减），warmup 早期 LR 极小，prompt 可能学得慢/学不动。**这是 tuning 风险不是 bug**：若 clip_id_loss 不降或 global 不涨，第一件事就是给 `cls_ctx`/`clip_id_proj` 单独抬 LR（可仿 make_optimizer.py:20 的 part_lr_factor 加一个 clip-id 分支，或 LARGE_FC_LR 思路）。建议训练时把 `details['clip_id']` 打进 log 盯它是否单调下降——日志已接（processor.py:1301），够观察。
experiments/exp341_clip_id_prompt/claude_review.md:95:代码正确性逐项核对通过：手写 CLIP 文本编码器与 open_clip 2.32.0 `encode_text` 在标准 ViT-L-14 配置下完全一致（batch_first 无 permute、EOT argmax、causal mask、text_projection Parameter 分支均已对源码确认）；优化器确实优化 cls_ctx + clip_id_proj（#1 风险排除）；clip_id_loss 不重复计、test 端不受影响、维度全动态无 hardcode、pretrained 本地 safetensors 路径有效。无 Critical/High 阻断项。
experiments/cargo_cvpb/litreview2/remedy_rethink.md:4:workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
experiments/cargo_cvpb/litreview2/remedy_rethink.md:91:唯一可救方向是换成 **positive-support / evidence-sufficiency**，不是 distractor remedy：例如看同一 query 在 gallery growth 下有多少“可见证据兼容的正样本支撑质量”，用 lower-tail positive similarity、positive dispersion、visible-part overlap、query-pair uncertainty 去解释 tax residual。它的独立信号应是：两个 query 1x top1 margin 相近，但 positive support mass 不同，growth tax 显著不同。这个方向不应 hallucinate 被遮挡区域，只做“可见证据是否足够”的校准或局部匹配。若零训练 residual 也没有，立即杀。
experiments/cargo_cvpb/litreview2/remedy_rethink.md:107:最终动作：保留 Tax 为诊断贡献，停止 Tax-Aware Distractor Training 主线；只允许做一个零训练 positive-support residual 预检。预检不过，直接 d14。
experiments/cargo_cvpb/litreview2/remedy_rethink.md:117:唯一可救方向是换成 **positive-support / evidence-sufficiency**，不是 distractor remedy：例如看同一 query 在 gallery growth 下有多少“可见证据兼容的正样本支撑质量”，用 lower-tail positive similarity、positive dispersion、visible-part overlap、query-pair uncertainty 去解释 tax residual。它的独立信号应是：两个 query 1x top1 margin 相近，但 positive support mass 不同，growth tax 显著不同。这个方向不应 hallucinate 被遮挡区域，只做“可见证据是否足够”的校准或局部匹配。若零训练 residual 也没有，立即杀。
experiments/cargo_cvpb/litreview2/remedy_rethink.md:133:最终动作：保留 Tax 为诊断贡献，停止 Tax-Aware Distractor Training 主线；只允许做一个零训练 positive-support residual 预检。预检不过，直接 d14。
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:361:This work was supported by the National Natural Science Foundation of China (No. 62302080), Guangxi Key Research and
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:501:automatically annotate large-scale ReID datasets in a human-like manner. The CLIP models pretrained on large-scale ReID datasets exhibit strong zero-shot performance. Their compatibility
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:1257:contrastive representation learning into Vision-language pretraining. These methods encode images and texts separately
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:1891:Table 1: Performance comparison with state-of-the-art methods on three TIPR benchmarks. The first part lists non-pretrained methods, while
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:1892:the following section presents pretrained methods. R@1, R@5, and R@10 are listed.
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:2249:This work was supported in part by the National Natural
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:2430:(MAE) decoder [14] to reconstruct masked image patch sequences into their original unmasked
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:2564:Visual Modality. Given an input image 𝐼 ∈ R𝐻 ×𝑊 ×𝐶 , we employ a CLIP-pretrained Vision
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:5076:This work is supported by the National Natural Science Foundation of China (Grants Nos. 62202061 and 62171043), the
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:5140:To address the aforementioned limitations, we leverage the strong cross-modal alignment capabilities of CLIP as the foundation of our method. CLIP is pretrained on a large-scale corpus of
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:5158:stages for AGPReID task. CVAF leverages the powerful vision-language pretrained model CLIP [10]
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:5266:person ReID, marking the first application of vision-language pretraining in this domain. CFine
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:5272:As vision-language pretrained models continue to evolve, their superior cross-modal alignment
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:6889:Implementation Details. We utilize either the original CLIP model [37] or its ReID-domain pretrained variants [22, 45] as encoders tailored to each modality. To maintain consistency, we employ
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:7173:“G,” “L,” and “P” in the “Type” column stand for global-matching method, local-matching method, and pretrained model with ReID-domain, respectively. “Image Enc.” and “Text Enc.” mean the backbone of image
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:7650:contrastive representation learning into Vision-language pretraining. These methods encode images and texts separately
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:8309:Evaluation Results on CUHK-PEDES We measure the performance of FMFA on the CUHKPEDES dataset, as presented in Table 3. When using the VL-Backbones without ReID-domain pretraining, FMFA achieves superior performance over advanced global matching methods, attaining
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:11079:This work was supported by National Natural Science Foundation of China (62102003), Anhui Postdoctoral Science Foundation
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:12018:This work is supported by the National Natural Science Foundation of China (No. 62466003, No. 62276221, No. 62376232), the
experiments/cargo_cvpb/litreview2/reviews/deep_8.md:12023:resources and support.
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:43:    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
experiments/cargo_cvpb/cvpb_rankregret_killswitch.py:60:_repo = os.path.abspath(os.path.join(_here, '..', '..'))   # repo root = .../SOLIDER-REID
experiments/cargo_cvpb/cvpb_lats_probe.py:8:the OA-SD episode warned against, so MEASURE it.
experiments/cargo_cvpb/smoke_airl_dualbranch.py:117:    """Tiny CPU AFDModel (resnet50, NO pretrained download)."""
experiments/cargo_cvpb/smoke_airl_dualbranch.py:118:    return AFDModel(num_classes=num_classes, pretrained=False,
experiments/exp108/design.md:8:- 因此，`exp108` 的核心不是继续调 `exp107` 的公式，而是把同一问题重新落在 **per-keypoint / common-support** 粒度：
experiments/exp108/design.md:9:  **只有在关键点可见性和 common-support 层面，target-target 与 target-distractor 的差异才可能被稳定表达。**
experiments/exp108/design.md:16:   的 common-support 距离比 `target ↔ target` 更小，则该 pair 应被惩罚。
experiments/exp108/design.md:17:3. 与 `exp107` 不同，duplicate-aware pruning 在 per-keypoint 层面才可能真正发挥作用，因为 duplicate detection 与 visibility/common-support 是同一层面的结构信息。
experiments/exp108/design.md:36:### 4. Counterfactual common-support penalty
experiments/exp108/design.md:41:  - `support_gap = min(d_q_gd, d_qd_g) - d_tt`
experiments/exp108/design.md:43:  - 当 `support_gap < 0` 时，说明 confuser 比 target-target 更占优，增加距离惩罚
experiments/exp108/design.md:59:  - 说明 ambiguity 这条 retrieval-time 线即使下沉到 per-keypoint/common-support，也还不足以形成稳定可用的排名信号
experiments/exp108/design.md:63:1. `cvk_hybrid` 已经吃掉了 target-target 的主要 common-support 信号，confuser penalty 额外增益不足
experiments/cargo_cvpb/design_agreid_v2.md:33:## 两个实验(均 AG-ReID.v2,均 Swin-Small + SOLIDER pretrain,256×128,bs64,60ep)
experiments/cargo_cvpb/design_agreid_v2.md:35:   `--dataset agreid_v2 --backbone swin_small --swin_pretrain .../swin_small.pth --img_size 256 128`
experiments/cargo_cvpb/design_agreid_v2.md:38:    --airl_fuse_w 0.25 --backbone swin_small --swin_pretrain .../swin_small.pth --img_size 256 128`
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:41:      report false-merge vs support AND vs a degree-matched random-label null;
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:42:      support-calibration must help BEYOND the mechanical support effect.
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:45:  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:78:ap.add_argument('--zipf_a', type=float, default=1.2, help='Zipf exponent for head support sizes')
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:186:    discrete support); plain double-argsort breaks ties by position and can
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:625:# TEST C — SINGLETON MERGE (Zipf gallery, tail false-merge vs head support)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:641:    # Build a Zipf gallery: assign target support sizes via Zipf, cap by availability.
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:654:        # head supports ~ Zipf, clamp to [2, available]; tail support = 1 (singleton)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:657:        # map to support sizes 2..maxsupp
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:677:    # is a HEAD id (tail wrongly absorbed). We measure attraction vs head support, with
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:686:    # per-head-ID accumulation across seeds: attraction-count, support-sum, appearances
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:688:    headid_supp_sum = {hid: 0 for hid in head_ids}     # sum of its support over seeds it appeared
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:727:    print("\n[C] P(tail-probe NN is a HEAD of support s) by support bin (DESCRIPTIVE, n=4 bins):")
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:738:    print("  [per-IMAGE rate FLAT across support -> purely mechanical count; RISING -> "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:743:    print(f"  binned Spearman(support, rate/headID)={rho_id_bin:+.4f} (trivially >0)  "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:748:    # support = its mean support; n_imgs ~ support (same thing here). We test whether
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:749:    # attraction RISES WITH SUPPORT AFTER CONTROLLING the image-count (support) itself is
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:751:    #   (a) Spearman(attraction, support)            -- includes the count effect
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:752:    #   (b) Spearman(attraction-per-image, support)  -- count removed; the non-trivial claim
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:760:    print(f"     Spearman(support, attraction-count)     = {rho_attr_supp:+.4f}  "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:762:    print(f"     Spearman(support, attraction-PER-IMAGE) = {rho_attrpi_supp:+.4f}  "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:765:    # ---- support-calibrated vs global threshold, CAL/EVAL FOLD across Zipf seeds ----
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:770:    print("\n[C] support-calibrated vs global threshold (OVERALL tail->head false-merge at "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:786:        # tail probes: record (score, nn-support, is-head, n_tail_probes_total)
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:817:            # per-support thresholds from CAL fold
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:834:                # support-calibrated: per-probe threshold by its nn-support
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:839:                  f"support-calibrated={fm_s:.4f}  (d={fm_s-fm_g:+.4f}; want NEGATIVE)  "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:841:        print(f"  support-level fallback-to-global fraction (sparse levels) = "
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:842:              f"{n_fallback/max(1,n_levels):.3f}  [high -> 'support-calibrated' is mostly global]")
experiments/cargo_cvpb/cvpb_gallery_killswitch.py:880:    print(f"[C] per-head-ID Spearman(support, attraction)={C['rho_attr_supp']:+.3f} (trivial)  "
experiments/exp056/design.md:12:- 与 PSG Stage 2 失败的区别: PSG 在 Stage 2 修改特征值（可能破坏 SOLIDER 预训练的低级特征），PGAM 只修改注意力路由（不改变特征值本身）
experiments/cargo_cvpb/smoke_swin_backbone.py:8:  2. swin_small path builds (optionally loading the SOLIDER teacher checkpoint),
experiments/cargo_cvpb/smoke_swin_backbone.py:17:Run on a CUDA box that has the SOLIDER swin_small.pth (lab-3090-d / lab-4090):
experiments/cargo_cvpb/smoke_swin_backbone.py:19:    python3 smoke_swin_backbone.py --swin_pretrain <repo>/pretrained/swin_small.pth
experiments/cargo_cvpb/smoke_swin_backbone.py:63:def _args(backbone, swin_pretrain, img_size):
experiments/cargo_cvpb/smoke_swin_backbone.py:75:    ns.swin_pretrain = swin_pretrain
experiments/cargo_cvpb/smoke_swin_backbone.py:118:def check_swin(device, swin_pretrain, img_size):
experiments/cargo_cvpb/smoke_swin_backbone.py:120:          f"pretrain={'yes' if swin_pretrain else 'NO -> scratch'})")
experiments/cargo_cvpb/smoke_swin_backbone.py:122:                        args=_args('swin_small', swin_pretrain, img_size)).to(device)
experiments/cargo_cvpb/smoke_swin_backbone.py:220:    ap.add_argument('--swin_pretrain', default='',
experiments/cargo_cvpb/smoke_swin_backbone.py:221:                    help='path to SOLIDER swin_small.pth (empty -> scratch init)')
experiments/cargo_cvpb/smoke_swin_backbone.py:227:        "smoke needs CUDA (the SOLIDER Swin forward hard-codes .cuda() for the " \
experiments/cargo_cvpb/smoke_swin_backbone.py:235:    print(f"  swin_pretrain = {args.swin_pretrain or '(none -> scratch)'}")
experiments/cargo_cvpb/smoke_swin_backbone.py:241:    model = check_swin(device, args.swin_pretrain, img_size)
experiments/cargo_cvpb/monitor.md:33:cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/monitor.md:35:  --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/monitor.md:36:  --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
experiments/cargo_cvpb/monitor.md:56:cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/monitor.md:58:  --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/monitor.md:59:  --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/monitor.md:137:- **★★★ 战略结论(2026-06-23)**: **两个涨点机制(OVC-SetVLAD 角度2 + ACVP 角度6)全失败 → resnet50 + 现有 OVLI loss 框架内"再加一个涨点模块"这条路走完。** headline 定为 **OVLI 52.37 单机制 + 4 个干净消融**(setpool/allview/α sweep + OVP 对照),5-codex 评 B 类可行。**下一步换思路: Swin/SOLIDER 强 backbone 冲 SOTA**(弱 resnet50 都 52.37 超 VDT 42.76,强 backbone 应上台阶)→ AG-ReID.v2 跨数据集 → paper 骨架。
experiments/cargo_cvpb/monitor.md:151:**根因**: **resnet50 调出来的峰值 LR(3.5e-4 AdamW 均一施加到 ~50M 参数 SOLIDER Swin transformer)对 Swin 过大 → warmup 升过 ~2.5e-4 时几步大更新把 backbone 推进"常数输出"退化吸引子**。resnet50 不塌(对它 3.5e-4 安全), 仓库主 SOLIDER config 训 Swin 用 SGD BASE_LR=8e-4 + 20-epoch warmup(对 transformer 等效步长温和得多)。这是 LR 问题, 不是 AMP/forward 问题(AMP 仅可能放大, 但 LR 是触发器; 全程无 inf/scale 警告)。
experiments/cargo_cvpb/monitor.md:165:- lab-4090 python = `/home/afr/vireid/.venv/bin/python`(torch 2.4.1+cu121); data_root `/home/afr/SOLIDER-REID/data`。**lab-4090 缺 afd_reid/afd_train.py 会循环 import**(cargo_cvpb 从它 import CE/Triplet/LR), 已 OSS 同步修复。
experiments/cargo_cvpb/monitor.md:203:**out_dir**: `/root/work/SOLIDER-REID/log/cargo/cvpb_airl_dualbranch`
experiments/cargo_cvpb/monitor.md:206:cd /root/work/SOLIDER-REID/experiments/cargo_cvpb && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 setsid python3 afd_train.py \
experiments/cargo_cvpb/monitor.md:207:  --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/monitor.md:208:  --out_dir /root/work/SOLIDER-REID/log/cargo/cvpb_airl_dualbranch \
experiments/cargo_cvpb/monitor.md:210:  --backbone swin_small --swin_pretrain /root/work/SOLIDER-REID/pretrained/swin_small.pth \
experiments/cargo_cvpb/monitor.md:213:(数据 `/root/work/SOLIDER-REID/data` = 默认; swin = `pretrained/swin_small.pth` 1.15GB; 与 OVLI-Swin `cvpb_swin_fix256` 同路径同 img_size 256x128。)
experiments/cargo_cvpb/monitor.md:224:- 配置 banner: `airl_dualbranch=True (fuse_w=0.25 ...)`; swin pretrain `All keys matched successfully`; CARGO train 51451/2500pid, query 312/149pid。
experiments/cargo_cvpb/monitor.md:275:- **与 oracle 矛盾的根因**: airl_gate_oracle 的 +1.46 上界是用**两个独立模型**(baseline + AIRL)的 score fusion 测的; 单模型双分支共享 backbone, 两头无法分化到独立模型那种互补度 → oracle headroom 不可在单模型内实现（与 exp109 identity-conditioned headroom 不可实现同类陷阱）。
experiments/cargo_cvpb/afd_train.py:113:    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/afd_train.py:116:        --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/afd_train.py:117:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
experiments/cargo_cvpb/afd_train.py:122:        --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/afd_train.py:123:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/afd_train.py:128:        --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/afd_train.py:129:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp_ovli \
experiments/cargo_cvpb/afd_train.py:1139:#     swin_small are both supported (the second forward just reuses `model`).
experiments/cargo_cvpb/afd_train.py:1235:    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/cargo_cvpb/afd_train.py:1263:    # 'swin_small' = SOLIDER Swin-Small (team asset, SOTA push): SOLIDER teacher
experiments/cargo_cvpb/afd_train.py:1264:    # pretrain + avg-pool + BNNeck, in_planes=768; AFD freq modules are NOT
experiments/cargo_cvpb/afd_train.py:1265:    # supported on swin (OVP/OVLI are independent and DO work). OVLI hooks the
experiments/cargo_cvpb/afd_train.py:1270:                         "or swin_small (SOLIDER Swin-Small, in_planes=768)")
experiments/cargo_cvpb/afd_train.py:1271:    ap.add_argument('--swin_pretrain', default='',
experiments/cargo_cvpb/afd_train.py:1272:                    help="path to the SOLIDER swin_small.pth teacher checkpoint "
experiments/cargo_cvpb/afd_train.py:1273:                         "(e.g. <repo>/pretrained/swin_small.pth). Empty -> train "
experiments/cargo_cvpb/afd_train.py:1277:                    help="SOLIDER semantic weight for the Swin backbone "
experiments/cargo_cvpb/afd_train.py:1559:        ap.error("--backbone swin_small does not support --use_afd (AFD modules "
experiments/cargo_cvpb/afd_train.py:1564:    # are supported and back-compatible:
experiments/cargo_cvpb/afd_train.py:1696:        print(f"  backbone=swin_small (SOLIDER, in_planes=768) "
experiments/cargo_cvpb/afd_train.py:1697:              f"pretrain={args.swin_pretrain or 'NONE (from scratch)'} "
experiments/cargo_cvpb/afd_train.py:1810:    # SAFE for resnet50 but DIVERGES the ~50M-param SOLIDER Swin transformer --
experiments/cargo_cvpb/afd_train.py:1816:    # backbone LR than the randomly-initialised heads (the repo's main SOLIDER
experiments/cargo_cvpb/afd_train.py:1931:    # LR group (pretrained weights, same as f_full's stages).  Assert both placements
experiments/cargo_cvpb/afd_train.py:1948:        # the rec late stage must be on the SCALED Swin LR group (it is pretrained
experiments/cargo_cvpb/afd_train.py:1958:                                       "Swin LR group (it is pretrained backbone "
experiments/exp126/design.md:34:- support-complete teacher 构造不变
experiments/exp108/monitor.md:7:- 核心变量: per-keypoint common-support 层面的 duplicate-aware confuser penalty
experiments/exp108/monitor.md:14:  - 若继续 ambiguity 主线，必须回到 `per-keypoint / common-support`
experiments/exp108/monitor.md:33:  1. per-keypoint / common-support 粒度下的 penalty 仍然整体负面
experiments/cargo_cvpb/litreview/reviews/lit_12.md:17:- SOLIDER-Swin 强 backbone(自监督人体预训练,in_planes=768)
experiments/cargo_cvpb/litreview/reviews/lit_12.md:82:- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:170:## 8. 代码底子（上游 SOLIDER-REID）
experiments/cargo_cvpb/litreview/reviews/lit_12.md:173:- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:185:- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:254:**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:256:**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:258:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:285:方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:302:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:304:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:400:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:401:- **exp017 探针放行候选 A**：SOLIDER 特征对 CLIP 遮挡物语义零线性解码（R²=-0.01）、嵌入级超锥零假设解释力仅约 26%、残差与 AP 相关 -0.164（控 s_top1 后 -0.097 贴线）。按预注册中间地带规则的属性级判据放行。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:417:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1016:This work was supported in part by the National Science Foundation Program of China (NSFC) (grant number: 61976241),
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1160:Index Terms— Occluded person re-identification, vision transformer, feature aggregation, feature completion.
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1170:2025. This work was supported in part by the National Natural Science
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1283:• A diverse feature completion (DFC) module that can
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1421:we design a diverse feature completion (DFC) module, which
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1511:(MML). SML constructs diverse query and support sets in each training cycle, allowing the model to learn
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1562:adopting a meta-learning perspective lies in the way query-support
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1565:To fully leverage the advantages of the query-support paradigm in
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1567:utility by dynamically constructing diverse query-support pairs in
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1574:diverse query and support sets during training, enabling the model to
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1578:support sets among the samples of each identity. Specifically, in each
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1580:samples form the support set. This ensures that every sample is utilized
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1582:to systematically explore all possible combinations of query-support
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1587:and support sets. Concurrently, the MML method captures long-term
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1712:subtask, SML constructs diverse query and support sets during each
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1743:metric space for VIPR by introducing a shuffling strategy that dynamically constructs query-support pairs while leveraging memory banks
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1787:learned to learn from a given support set to minimize loss over a batch
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1843:This work was supported by the National Nature Science Foundation of China (No. 62376201). This research was financially
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1844:supported by funds from Key Laboratory of Social Computing and Cognitive Intelligence (Dalian University of Technology),
experiments/cargo_cvpb/litreview/reviews/lit_12.md:1992:The work is partially supported by Shanghai Artificial Intelligence Innovation and Development Fund (No. 2020-RGZN02026).
experiments/cargo_cvpb/litreview/reviews/lit_12.md:2045:Adapter [10] in our method. CLIP is a powerful pretrained model which aligns texts and images
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3365:能发原因：VI-ReID 的普通 triplet 或 center loss 只看 batch 内关系，数据少时学不到全局度量。作者把训练改成 query-support 检索小任务，并用 memory bank 引入历史特征。证据链是 sample-based、center-based、meta-based 三类对比，加 RegDB/SYSU 实验。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3399:挂团队资产：CARGO / AG-ReID.v2，SOLIDER-Swin，SMPL。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3401:cheap kill-switch：不训练，直接用现有 SOLIDER 或 AG-ReID 模型构造 mixed-view query/gallery，按航拍比例、地面比例、低清比例扫一遍。如果混合协议比纯跨视角掉超过 2 mAP，并且错误主要来自同视角不同人压过跨视角同人，就成立；否则杀掉。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3404:挂团队资产：SMPL 3D 几何，pose 热图门控，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3409:挂团队资产：aerial-ground 数据，SMPL，pose gating，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3410:和最像工作的区别：最像 MSP-ReID，但 MSP 只讲换衣后的发型捷径；我们讲航拍地面里的多级捷径链。先证明 SOLIDER 特征能预测相机高度、视角、背景块或尺度，再用结构保留的擦除、尺度反事实、背景替换、SMPL 轮廓保持来压掉这些非身份线索。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3411:cheap kill-switch：冻结 SOLIDER 特征训练几个线性探针，预测视角、相机、高度、背景类别。如果这些 nuisance 可预测性很高，并且错误检索按这些因素聚类，方向成立。若探针预测不出来，说明捷径链证据不足，杀掉。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3414:挂团队资产：SOLIDER-Swin，CARGO / AG-ReID.v2，SMPL。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3442:能发原因：VI-ReID 的普通 triplet 或 center loss 只看 batch 内关系，数据少时学不到全局度量。作者把训练改成 query-support 检索小任务，并用 memory bank 引入历史特征。证据链是 sample-based、center-based、meta-based 三类对比，加 RegDB/SYSU 实验。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3476:挂团队资产：CARGO / AG-ReID.v2，SOLIDER-Swin，SMPL。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3478:cheap kill-switch：不训练，直接用现有 SOLIDER 或 AG-ReID 模型构造 mixed-view query/gallery，按航拍比例、地面比例、低清比例扫一遍。如果混合协议比纯跨视角掉超过 2 mAP，并且错误主要来自同视角不同人压过跨视角同人，就成立；否则杀掉。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3481:挂团队资产：SMPL 3D 几何，pose 热图门控，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3486:挂团队资产：aerial-ground 数据，SMPL，pose gating，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3487:和最像工作的区别：最像 MSP-ReID，但 MSP 只讲换衣后的发型捷径；我们讲航拍地面里的多级捷径链。先证明 SOLIDER 特征能预测相机高度、视角、背景块或尺度，再用结构保留的擦除、尺度反事实、背景替换、SMPL 轮廓保持来压掉这些非身份线索。  
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3488:cheap kill-switch：冻结 SOLIDER 特征训练几个线性探针，预测视角、相机、高度、背景类别。如果这些 nuisance 可预测性很高，并且错误检索按这些因素聚类，方向成立。若探针预测不出来，说明捷径链证据不足，杀掉。
experiments/cargo_cvpb/litreview/reviews/lit_12.md:3491:挂团队资产：SOLIDER-Swin，CARGO / AG-ReID.v2，SMPL。  
experiments/exp126/monitor.md:18:- [x] support-complete teacher、bank 更新、主 loss 配比全部保持不变
experiments/cargo_cvpb/cvpb_gallery_result.md:70:Zipf gallery(head 多图 head support 2-12, tail singleton), tail query 移除自身 singleton→真 unknown。
experiments/cargo_cvpb/cvpb_gallery_result.md:77:| per-head-ID Spearman(support, attraction-count) | +0.043 | +0.059 |
experiments/cargo_cvpb/cvpb_gallery_result.md:78:| per-head-ID Spearman(support, attraction-PER-IMAGE) | **−0.013** | **−0.009** |
experiments/cargo_cvpb/cvpb_gallery_result.md:79:| support-cal vs global OVERALL false-merge d (recall0.90) | +0.000 | −0.001 |
experiments/cargo_cvpb/cvpb_gallery_result.md:84:heads **不超额吸附**(ρ≈0 甚至略负)。CAL/EVAL 折后 support-calibrated 阈值**零增益**(d≈0)。
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:40:    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
experiments/cargo_cvpb/cvpb_hubness_killswitch.py:52:_repo = os.path.abspath(os.path.join(_here, '..', '..'))   # repo root = .../SOLIDER-REID
experiments/exp137/design.md:24:- online support teacher
experiments/pose_clip_codex_synthesis.md:17:3. **PC-MSC (file16)**: pose mask 可见部位→小decoder从可见证据重建被mask部位的CLIP语义token(对齐frozen CLIP clean teacher)。**被删的token吸不走**(从输入删除)。先例: PersonMAE/PersonViT, MVP/RILS(CLIP-token MIM)。有kill-switch(random-mask vs pose-mask)。
experiments/cargo_cvpb/codex_fleet_synthesis.md:27:3. Swin/SOLIDER backbone + AG-ReID.v2 跨数据集主表
experiments/exp119/design.md:6:- `exp051 PAML` 中性，说明 **只改 part triplet 的距离形式** 也不足以把 pairwise common-support 几何传给 global embedding。
experiments/exp119/design.md:7:- `exp109-116` 又说明：把 support 压成 `per-ID EMA prototype` 会损失太多 pair-specific 细节。
experiments/exp119/design.md:10:**如何把 keypoint/common-support 分支已经掌握的 pairwise 比较几何，直接蒸馏给 global embedding。**
experiments/exp119/design.md:51:  - `exp110-116`：prototype-bank support-complete 路线天花板已现
experiments/exp119/design.md:58:  - 后续 `cvk_hybrid` 增益缩小，说明训练端已吸收部分 common-support 几何
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:577:**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:579:**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:581:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:608:方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:625:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:627:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:698:- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:786:## 8. 代码底子（上游 SOLIDER-REID）
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:789:- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:801:- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:858:Clothes Caption Generation (MCG), and Multi-Branch ClothesAgnostic Feature Extraction (MAE). MCG leverages Large VisionLanguage Models to generate diverse coarse-to-ﬁne clothing descriptions, reducing the impact of clothing on feature extraction. MAE
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:982:and the Multi-Branch Clothes-Agnostic Feature Extraction (MAE).
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:984:generates coarse-to-ﬁne clothes captions to guide the MAE module
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:985:in suppressing clothing-related features. The MAE module employs
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:1013:Multi-Branch Clothes-Agnostic Feature Extraction (MAE) module.
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:1040:input the descriptions and their corresponding images into the pretrained CLIP model’s text and image encoders, generating text features Ft and image features Fv . We calculate the cosine similarity to
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:1205:In the MAE module, both branches adopt two core loss functions
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:1868:(MAE). MCG uses LVLMs to generate diverse clothing descriptions, reducing the inﬂuence of clothing changes. MAE combines
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:1874:This work was supported in part by the Natural Science Foundation
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:2009:synthetic data generation pipeline for pretraining cloth-changing person
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:2089:This work was supported by the Natural Science Foundation (NSF) of
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:2101:Compared to UDA, USL is more challenging to train directly on unlabeled data due to the lack of pretraining support
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:2926:I This research was supported by National Natural Science Foundation of China (Grant Nos. 62376089, U23A20318, 62302154, 62472149), and Young and
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:4093:This work is supported in part by the Natural Science Foundation of
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:5328:capabilities. For example, Yu et al. [20] proposed TFCLIP, which utilizes a combination of vit [21] and pretrained CLIP to obtain robust video features. While this
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:8087:retrieval accuracy. This result fully demonstrates the retrieval superiority of MCCAN and strongly supports the feasibility of its practical
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:8354:This study is partially supported by the National Key R&D Program of China (No. 2022YFC3803600), the National Natural Science
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:8356:Key Laboratory of Software Development Environment (No. SKLSDE2023ZX-11). This research was supported by the Research Start-up
experiments/cargo_cvpb/litreview2/reviews/deep_19.md:8357:Funds of Hangzhou International Innovation Institute of Beihang University under Grant No. 2024KQ012. Thank you for the support from
experiments/cargo_cvpb/litreview/reviews/lit_16.md:17:- SOLIDER-Swin 强 backbone(自监督人体预训练,in_planes=768)
experiments/cargo_cvpb/litreview/reviews/lit_16.md:318:**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:320:**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:322:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:349:方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:366:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:368:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:464:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:465:- **exp017 探针放行候选 A**：SOLIDER 特征对 CLIP 遮挡物语义零线性解码（R²=-0.01）、嵌入级超锥零假设解释力仅约 26%、残差与 AP 相关 -0.164（控 s_top1 后 -0.097 贴线）。按预注册中间地带规则的属性级判据放行。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:481:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:499:- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:587:## 8. 代码底子（上游 SOLIDER-REID）
experiments/cargo_cvpb/litreview/reviews/lit_16.md:590:- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:602:- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
experiments/cargo_cvpb/litreview/reviews/lit_16.md:671:2026. This work was supported in part by Guangdong Science and Technology
experiments/cargo_cvpb/litreview/reviews/lit_16.md:1348:+ This work was supported in part by the National Natural Science
experiments/cargo_cvpb/litreview/reviews/lit_16.md:1864:This work was supported by the National Natural Science Foundation of China (Nos. 62272461, 62172417, 62276266, and
experiments/cargo_cvpb/litreview/reviews/lit_16.md:2022:This work was supported by the National Natural Science Foundation of
experiments/cargo_cvpb/litreview/reviews/lit_16.md:2496:Prior works, however, considered the problems of continuously updating models and decentralized training models separately. They are still unable to support distributed edge clients
experiments/cargo_cvpb/litreview/reviews/lit_16.md:2787:current version 5 February 2026. This work was supported in part by the
experiments/cargo_cvpb/litreview/reviews/lit_16.md:3319:ReID has been ongoing to expand the scope of training and handle diverse cases. Zhang et al. [45] addressed overfitting by pretraining with
experiments/cargo_cvpb/litreview/reviews/lit_16.md:3457:挂团队资产：aerial-ground + SMPL 3D 几何 + SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_16.md:3462:挂团队资产：aerial-ground + pose 热图门控 + SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_16.md:3472:挂团队资产：aerial-ground + 现有终身 ReID 基建 + SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_16.md:3530:挂团队资产：aerial-ground + SMPL 3D 几何 + SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_16.md:3535:挂团队资产：aerial-ground + pose 热图门控 + SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_16.md:3545:挂团队资产：aerial-ground + 现有终身 ReID 基建 + SOLIDER-Swin。  
experiments/cargo_cvpb/codex_review_gallery_killswitch.md:9:- **High**: `spearman()` uses double `argsort` ranks, not tie-aware ranks. With tied/constant support or `#false-in-topk`, it can fabricate correlations instead of returning proper tied-rank Spearman/nan. This affects A controls and C headline.
experiments/cargo_cvpb/codex_review_gallery_killswitch.md:29:  - Test C: per-image Spearman Market −0.013 / OD −0.009(≈0); CAL/EVAL 折后 support-cal d≈0.000/−0.013, fallback 0%/14%。**C DEAD。**
experiments/exp119/monitor.md:28:  - `CSRD` 是当前最直接的新机制验证：不用 prototype，而是直接蒸馏 common-support 关系
experiments/exp119/monitor.md:214:  4. 这说明 `CSRD` 的作用更像是把 common-support pairwise 几何蒸进 backbone/global，而不是直接替代 fusion 或 test-time correction
experiments/exp119/monitor.md:215:  5. 同时它也暴露了当前版本的瓶颈：teacher 仍来自单图 `kp_feats`，还不够 support-complete
experiments/exp119/monitor.md:218:  - 下一步最合理的单变量不是扫 `CSRD` 权重/温度，而是把 `exp109` 的 support-complete teacher headroom 引回 `CSRD`，做更强的 relational teacher
experiments/exp324d/codex_review.md:12:- `bmm(pool_w, patch)` 与 `build_part_pose` cell 等价：同 part groups / visibility skip / (0,0) sentinel / rounded grid cell / 3×3 窗 / set-union / mean 权重。`r*GRID_W+c` 匹配 DINO patch row-major 顺序。
experiments/exp324d/codex_review.md:13:- Train path 无 detach / no_grad / numpy 转换；梯度流 `DINO LoRA → patch → bmm → PartHead → losses`。
experiments/exp324d/codex_review.md:14:- `micro_bs` 只切 DINO forward 再 concat；batch-hard triplet 见完整逻辑 batch；每逻辑 batch 恰一次 optimizer.step()。
experiments/exp324d/codex_review.md:21:LoRA / DINOv2 / pose-visible-part matching 各自有先例（PVPM/PFD），DINOv2-for-ReID 新兴，LoRA-DINOv2 在 ReID 外存在。**未找到 LoRA-finetuned DINOv2 + 可微姿态锚定部位池化 + mutually-visible part-MaxSim 用于遮挡单图行人 ReID 的直接先例** → 组合 plausibly new。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4:workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:20:/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:22:/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:24:/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:35:/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:42:/bin/zsh -lc pwd in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:44:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:47:/bin/zsh -lc ls in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:79:/bin/zsh -lc "rg --files -g 'results.md' -g 'decisions.md' -g 'innovation_brainstorm.md' -g 'story.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:81:/bin/zsh -lc "find .. -maxdepth 4 \\( -name results.md -o -name decisions.md -o -name innovation_brainstorm.md -o -name story.md \\)" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:87:/bin/zsh -lc "sed -n '1,260p' novelty_rankinstab.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:92:workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:108:/bin/zsh -lc "sed -n '1,220p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:110:/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:112:/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:123:/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:130:/bin/zsh -lc pwd in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:132:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:135:/bin/zsh -lc ls in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:167:/bin/zsh -lc "rg --files -g 'results.md' -g 'decisions.md' -g 'innovation_brainstorm.md' -g 'story.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:169:/bin/zsh -lc "find .. -maxdepth 4 \\( -name results.md -o -name decisions.md -o -name innovation_brainstorm.md -o -name story.md \\)" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:176:/bin/zsh -lc "sed -n '1,220p' all_papers.txt" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:347:/bin/zsh -lc 'rg -n "Rank|rank|instability|adaptive|early|exit|anytime|CFPER|cascade|re-ranking|rerank|difficulty|uncertainty|budget" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:373:./reviews/deep_3.md:512:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:688:./reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:846:./reviews/deep_13.md:618:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:960:./reviews/deep_17.md:217:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1089:./reviews/deep_19.md:581:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1343:./reviews/deep_11.md:423:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1551:./reviews/deep_28.md:215:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1705:./reviews/deep_15.md:4420:a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1766:./reviews/deep_15.md:10485:   198	a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1807:./reviews/deep_15.md:13260:   198	a pretrained weight 𝑊0 ∈ R𝑑×𝑘 , LoRA use low-rank decomposition to
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1822:./reviews/deep_18.md:385:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1963:./reviews/deep_18.md:7298:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:1966:./reviews/deep_18.md:7353:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2109:./ondisk_pivot.md:3301:./reviews/deep_22.md:842:incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2124:./ondisk_pivot.md:3656:./reviews/deep_22.md:9286:216:   216	incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2153:./ondisk_pivot.md:4266:./reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2154:./ondisk_pivot.md:4269:./reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2155:./ondisk_pivot.md:4402:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2156:./ondisk_pivot.md:4412:./reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2157:./ondisk_pivot.md:4413:./reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2158:./ondisk_pivot.md:4421:./reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2163:./ondisk_pivot.md:4652:./reassess/r_3.md:7999:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2198:./reviews/deep_16.md:128:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2471:./reviews/deep_6.md:518:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2597:./reviews/deep_26.md:451:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2846:./reviews/deep_23.md:445:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:2978:./reviews/deep_9.md:779:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:3228:./reviews/deep_21.md:6648:   170	regions and performs adaptive cross-modality completion,
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:3229:./reviews/deep_21.md:6664:   186	adaptive cross-modality feature completion, effectively alleviating spatial misalignment and modality-specific degradation.
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:3293:./reviews/deep_1.md:4308:1299-lower than some recent methods on certain Rank-1 metrics. One possible reason, as supported by the ablation study in Table 5, is that SSM
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:3469:./reviews/deep_2.md:693:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:3788:./reviews/deep_27.md:460:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:3912:./reviews/deep_12.md:387:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4096:./reviews/deep_22.md:449:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4114:./reviews/deep_22.md:842:incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4269:./reviews/deep_20.md:779:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4276:./reviews/deep_20.md:975:**方向（用户决策#33=选项1）**：放开 SOLIDER 强主干，回常规 ImageNet 预训练主干(TransReID ViT-base、ImageNet 预训练 Swin-Tiny)刷标准遮挡 ReID mAP/rank1。可靠性/conformal 线归档(决策#28-#32 的负结果+诊断作分析素材)。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4454:./analyses/ALL_ANALYSES.txt:570:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4519:./debate/d_1.md:65:web search: SOLIDER person re-identification re-ranking results GitHub
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4545:./reviews/deep_5.md:762:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4792:./validate/v_2.md:457:/bin/zsh -lc 'rg -n "非对称|包含|probabil|uncertain|uncertainty|OT|optimal|PDA|MaxSim|ColBERT|Aligned|distribution|Gaussian|KL|variance|方差|因果|causal|align|对齐" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:4801:./validate/v_2.md:715:./reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5083:./validate/v_2.md:6829:/bin/zsh -lc "rg -n \"PDA|probabilistic|Probability|Probabil|distribution|variance|uncertainty|text.*image|文本分布|方差=|⊇|⊆|Optimal Transport|OT-ReID|Wasserstein|EMD|CM-EMD|G2DA\" all_papers.txt analyses reviews validate -g '*.txt' -g '*.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5089:./validate/v_2.md:6969:reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5109:./validate/v_2.md:7307:validate/v_2.md:457:/bin/zsh -lc 'rg -n "非对称|包含|probabil|uncertain|uncertainty|OT|optimal|PDA|MaxSim|ColBERT|Aligned|distribution|Gaussian|KL|variance|方差|因果|causal|align|对齐" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5112:./validate/v_2.md:7360:validate/v_2.md:715:./reviews/deep_8.md:10572:identities clearly supports that TCL improves the robustness and view-invariance of the learned
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5266:./validate/v_2.md:11203:reviews/deep_22.md:842:incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5311:./validate/v_2.md:11438:reviews/deep_22.md:9286:216:   216	incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5401:./validate/v_2.md:12729:validate/v_3.md:2764:../codex_review_ovli.txt:3969:experiments/overnight_innovation_log.md:187:- **战略转向**：FM 方向从"唯一活口/大概率 me-too"升级为"有真实正结果"。但需诚实评估**新颖性**：DINOv2+LoRA+ReID 是否 me-too（PersonViT / DINO-as-backbone territory）？competitive 到什么程度（36.72 heavy vs Swin 72.57——仍有大 gap，但这是 pose-part-MaxSim 单分支、无 PSG/GCN/全套）？下一步：等 e30 看上限 + 对照 rank16/large 看 capacity 曲线，再判断是"competitive-FM-adaptation 方法"还是"诊断性结论的正向补充"。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5405:./validate/v_2.md:12790:validate/v_3.md:3452:../litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5406:./validate/v_2.md:12794:validate/v_3.md:3465:../litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5408:./validate/v_2.md:12913:validate/v_3.md:4392:../litreview/reviews/lit_19.md:2690:cheap kill-switch：在 CARGO 或 AG-ReID.v2 上取 SOLIDER top-k 错误但高度相似的负样本，人工快速看一小批，判断是否存在真实不可分或标注不确定。再做一个极简 soft negative 训练或只在 triplet 里屏蔽疑似歧义负样本。若 rank1 或 mAP 有明显提升，说明这个设定值得扩成方法稿。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5409:./validate/v_2.md:12916:validate/v_3.md:4401:../litreview/reviews/lit_19.md:2766:cheap kill-switch：在 CARGO 或 AG-ReID.v2 上取 SOLIDER top-k 错误但高度相似的负样本，人工快速看一小批，判断是否存在真实不可分或标注不确定。再做一个极简 soft negative 训练或只在 triplet 里屏蔽疑似歧义负样本。若 rank1 或 mAP 有明显提升，说明这个设定值得扩成方法稿。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5410:./validate/v_2.md:12963:validate/v_2.md:457:/bin/zsh -lc 'rg -n "非对称|包含|probabil|uncertain|uncertainty|OT|optimal|PDA|MaxSim|ColBERT|Aligned|distribution|Gaussian|KL|variance|方差|因果|causal|align|对齐" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5449:./validate/v_2.md:13686:validate/v_2.md:6829:/bin/zsh -lc "rg -n \"PDA|probabilistic|Probability|Probabil|distribution|variance|uncertainty|text.*image|文本分布|方差=|⊇|⊆|Optimal Transport|OT-ReID|Wasserstein|EMD|CM-EMD|G2DA\" all_papers.txt analyses reviews validate -g '*.txt' -g '*.md'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5457:./validate/v_2.md:13737:validate/v_2.md:7307:validate/v_2.md:457:/bin/zsh -lc 'rg -n "非对称|包含|probabil|uncertain|uncertainty|OT|optimal|PDA|MaxSim|ColBERT|Aligned|distribution|Gaussian|KL|variance|方差|因果|causal|align|对齐" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5541:./reassess/r_2.md:811:analyses/deep_18.txt:8:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5545:./reassess/r_2.md:915:analyses/ALL_ANALYSES.txt:570:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5548:./reassess/r_2.md:1165:reassess/r_3.md:2389:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/novelty_gopl.txt:38:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5549:./reassess/r_2.md:1263:reassess/r_3.md:2822:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:654:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5550:./reassess/r_2.md:1287:reassess/r_3.md:2885:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1039:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5551:./reassess/r_2.md:1306:reassess/r_3.md:2918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1226:./reviews/deep_25.md:1365:FrankMocap [61] to recover SMPL mesh sequences from the
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5552:./reassess/r_2.md:1311:reassess/r_3.md:2923:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1259:./reviews/deep_25.md:9584:FrankMocap [61] to recover SMPL mesh sequences from the
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5553:./reassess/r_2.md:1332:reassess/r_3.md:2944:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1293:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5554:./reassess/r_2.md:1336:reassess/r_3.md:2949:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1298:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5555:./reassess/r_2.md:1371:reassess/r_3.md:3019:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2180:./validate/v_2.md:5106:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5556:./reassess/r_2.md:1374:reassess/r_3.md:3022:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2183:./validate/v_2.md:5118:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5557:./reassess/r_2.md:1377:reassess/r_3.md:3026:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2193:./validate/v_2.md:5170:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5558:./reassess/r_2.md:1381:reassess/r_3.md:3035:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2220:./validate/v_2.md:5929:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5559:./reassess/r_2.md:1570:reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5560:./reassess/r_2.md:1580:reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5561:./reassess/r_2.md:1605:reassess/r_3.md:3499:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3262:廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5562:./reassess/r_2.md:1615:reassess/r_3.md:3513:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3337:廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5563:./reassess/r_2.md:1718:reassess/r_3.md:3794:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2855:最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5564:./reassess/r_2.md:1723:reassess/r_3.md:3800:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2928:最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5565:./reassess/r_2.md:1756:reassess/r_3.md:3883:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:2891:of non-overlapping distributed cameras. In the early stages
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5566:./reassess/r_2.md:1787:reassess/r_3.md:3945:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3639:便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5567:./reassess/r_2.md:1798:reassess/r_3.md:3957:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3693:便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5568:./reassess/r_2.md:1876:reassess/r_3.md:4271:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_4.md:21478:../airl_related_work.md:26:RAR 用 query resolution **动态路由**选子空间。AIRL **不路由**——#3 oracle 实测硬路由(area/reliability 阈值)失败(≤+0.41),**软固定先验融合**(w=0.25,clean+recover 两证据头)反而 +1.46。所以 AIRL 的 claim 必须是 "fixed-prior fusion of clean/recover evidence heads under an observation-limited ceiling",**绝不吹成 query-budget routing**(那既撞 RAR 又与实现不符)。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5569:./reassess/r_2.md:1917:reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5570:./reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5571:./reassess/r_2.md:1922:reassess/r_3.md:4495:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:110:"transfer_to_occluded": "Two directly usable ideas (the paper IS occluded ReID, so these are native): (1) The attention-magnitude saliency for foreground/background separation — derive a per-patch foreground probability from a lightweight conv-sigmoid head trained ONLY by a foreground-background contrastive loss (no pose/parsing labels), then suppress occluder/background tokens; this is an external-prior-free occluder gate, attractive because pose estimators fail under occlusion. (2) The importance-ranked cumulative fusion (FHFM): rank patches by CLS-attention*norm and let high-confidence (likely visible) patches contribute to every fusion level while low-importance (likely occluded) patches contribute least — a soft, label-free visibility weighting that downweights corrupted regions without a hard mask. The region-restricted masked attention with learnable part tokens is also a clean parsing-free way to get part features robust to missing regions.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5572:./reassess/r_2.md:1924:reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5573:./reassess/r_2.md:1925:reassess/r_3.md:4498:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:127:"novel_component": "Intersection-Alignment Score (IAS): turning an explainability signal into a re-ranking operator — measure how much the model's attention overlaps the visible-body mask (IoU) and how mutually consistent query/gallery attention maps are, then asymmetrically shrink or inflate pairwise distances. It makes \"is the model looking at occluded regions?\" a concrete, label-free matching correction rather than a post-hoc visualization.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5574:./reassess/r_2.md:1985:reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5575:./reassess/r_2.md:1986:reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5576:./reassess/r_2.md:1990:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5577:./reassess/r_2.md:1991:reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5641:./reassess/r_2.md:4091:reassess/r_2.md:811:analyses/deep_18.txt:8:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5642:./reassess/r_2.md:4131:reassess/r_2.md:915:analyses/ALL_ANALYSES.txt:570:4. **证据闭环**：关键不是总表，而是消融。CE 到 CE+CM 后，RegDB V2I Rank-1 从 59.0 到 88.0，SYSU all-search 从 56.5 到 67.1，证明 meta metric 本身有效。加 SMR 后 RegDB 从 88.0 到 94.1，证明双向一致性有用。去掉 shuffling 后 RegDB 只有 89.3，加回后到 94.1，SYSU 从 63.6 到 68.6，说明“全样本轮流 query-support”不是装饰。MM 的收益较小，更多证明 memory 是辅助增强。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5643:./reassess/r_2.md:4307:reassess/r_2.md:1570:reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5644:./reassess/r_2.md:4310:reassess/r_2.md:1580:reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5645:./reassess/r_2.md:4372:reassess/r_2.md:1756:reassess/r_3.md:3883:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:2891:of non-overlapping distributed cameras. In the early stages
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5646:./reassess/r_2.md:4416:reassess/r_2.md:1917:reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5647:./reassess/r_2.md:4417:reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5648:./reassess/r_2.md:4419:reassess/r_2.md:1924:reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5649:./reassess/r_2.md:4420:reassess/r_2.md:1925:reassess/r_3.md:4498:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:127:"novel_component": "Intersection-Alignment Score (IAS): turning an explainability signal into a re-ranking operator — measure how much the model's attention overlaps the visible-body mask (IoU) and how mutually consistent query/gallery attention maps are, then asymmetrically shrink or inflate pairwise distances. It makes \"is the model looking at occluded regions?\" a concrete, label-free matching correction rather than a post-hoc visualization.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5650:./reassess/r_2.md:4436:reassess/r_2.md:1985:reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5651:./reassess/r_2.md:4437:reassess/r_2.md:1986:reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5652:./reassess/r_2.md:4439:reassess/r_2.md:1990:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5653:./reassess/r_2.md:4440:reassess/r_2.md:1991:reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5673:./reassess/r_2.md:5162:reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5674:./reassess/r_2.md:5166:reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5675:./reassess/r_2.md:5235:reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5676:./reassess/r_2.md:5300:reassess/r_3.md:3883:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:2891:of non-overlapping distributed cameras. In the early stages
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5677:./reassess/r_2.md:5417:reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5678:./reassess/r_2.md:5420:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5679:./reassess/r_2.md:5422:reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5680:./reassess/r_2.md:5423:reassess/r_3.md:4498:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:127:"novel_component": "Intersection-Alignment Score (IAS): turning an explainability signal into a re-ranking operator — measure how much the model's attention overlaps the visible-body mask (IoU) and how mutually consistent query/gallery attention maps are, then asymmetrically shrink or inflate pairwise distances. It makes \"is the model looking at occluded regions?\" a concrete, label-free matching correction rather than a post-hoc visualization.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5681:./reassess/r_2.md:5447:reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5682:./reassess/r_2.md:5448:reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5683:./reassess/r_2.md:5455:reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5684:./reassess/r_2.md:5458:reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5685:./reassess/r_2.md:5460:reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5686:./reassess/r_2.md:5462:reassess/r_3.md:4768:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:471:- **为何非已关方向**: 与 visibility 加权**正交**(visibility 问"是否被遮挡"二值; burstiness 问"这个可见特征是否被过度计数"集内冗余统计)。一个 part 可完全可见且高 visibility 却 bursty(大块平躯干)——现有方法升权它, burstiness 说降权。非 completion(不重建)、非 scorer 微变体(改特征集聚合)、非 uncertainty(无 per-feature 置信)。**Novelty 已核: 无 occluded-ReID 做 burst-aware/democratic over part/patch 特征**(最近 cousin = Self-similarity guided probabilistic matching ESWA'23 是去噪/校验, 相反符号)。详见 memory [[burstiness-democratic-aggregation-bet]]。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5687:./reassess/r_2.md:5470:reassess/r_3.md:4815:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:51:**收敛点(e30 matched)双确认**：λ=0 vs λ=1 同 rank16/seed/script 跑到 e30 oracle 仍**完全一致**——Jaccard 0.2646 vs 0.2627、oracle +0.85 vs +0.80、fusion best ALL 75.74 vs 75.73（λ=1 甚至略低）。**早期(e10)+收敛(e30) 双证据：解相关在任何训练阶段对互补性零效果。** 加上 decorr-floor 证据（λ=2 双倍权重只把 0.041 降 ~1% → ~0.041 是 ID-constrained floor，共享判别方向是 ID load-bearing 的、删不掉）。→ **张力对显式干预(e10/e30、λ∈{0,1,2}、λ=10 进行中)全程鲁棒，是 fundamental 的，不是可调超参。**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5693:./reassess/r_2.md:5827:reassess/r_3.md:7999:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5694:./reassess/r_2.md:5830:reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5695:./reassess/r_2.md:5832:reassess/r_3.md:8009:./reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5696:./reassess/r_2.md:5838:reassess/r_3.md:8032:./reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5744:./reassess/r_3.md:1976:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_7.md:7025:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5745:./reassess/r_3.md:1980:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_7.md:7106:4. 证据闭环：消融很直接。CSSA1 相对硬条带 baseline 在三个数据集上 mAP 分别提升 9.5%、9.8%、4.6%，证明“自语义 part 对齐”比条带强。CARE 在 LS-VID 上相对 CSSA1 提升 5.6% mAP 和 7.1% Rank-1，证明视频上下文能压噪。SGFA 把 LS-VID Rank-1 从 78.6% 提到 83.7%，证明 pair-wise semantic-aligned global feature 有用。可视化也对应检测错误和遮挡。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5746:./reassess/r_3.md:1993:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_19.md:666:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5747:./reassess/r_3.md:2017:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:300:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5748:./reassess/r_3.md:2067:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:864:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5749:./reassess/r_3.md:2153:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:545:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5750:./reassess/r_3.md:2260:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:213:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5751:./reassess/r_3.md:2389:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/novelty_gopl.txt:38:- “GOPL builds a geometry-ordered positive graph where high co-visible-surface positives provide early strong constraints and low-overlap positives are delayed, weakened, or connected through bridge positives.”
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5752:./reassess/r_3.md:2400:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:470:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5753:./reassess/r_3.md:2435:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:703:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5754:./reassess/r_3.md:2466:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp155/design.md:54:- 关键看 `evid_unc`: 如果遮挡样本确实显示更高 uncertainty → 机制生效
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5755:./reassess/r_3.md:2483:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:880:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5756:./reassess/r_3.md:2501:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:991:**方向（用户决策#33=选项1）**：放开 SOLIDER 强主干，回常规 ImageNet 预训练主干(TransReID ViT-base、ImageNet 预训练 Swin-Tiny)刷标准遮挡 ReID mAP/rank1。可靠性/conformal 线归档(决策#28-#32 的负结果+诊断作分析素材)。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5757:./reassess/r_3.md:2533:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:534:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5758:./reassess/r_3.md:2578:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:603:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5759:./reassess/r_3.md:2615:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:778:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5760:./reassess/r_3.md:2637:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:13269:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5761:./reassess/r_3.md:2643:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:13350:2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5762:./reassess/r_3.md:2662:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:534:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5763:./reassess/r_3.md:2701:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:597:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5764:./reassess/r_3.md:2732:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:302:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5765:./reassess/r_3.md:2763:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:6486:4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5766:./reassess/r_3.md:2770:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:6563:4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5767:./reassess/r_3.md:2822:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:654:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5768:./reassess/r_3.md:2847:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:830:./analyses/deep_14.txt:21:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5769:./reassess/r_3.md:2869:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1005:./analyses/ALL_ANALYSES.txt:359:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5770:./reassess/r_3.md:2885:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1039:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5771:./reassess/r_3.md:2918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1226:./reviews/deep_25.md:1365:FrankMocap [61] to recover SMPL mesh sequences from the
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5772:./reassess/r_3.md:2923:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1259:./reviews/deep_25.md:9584:FrankMocap [61] to recover SMPL mesh sequences from the
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5773:./reassess/r_3.md:2944:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1293:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5774:./reassess/r_3.md:2949:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1298:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5775:./reassess/r_3.md:2989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1760:./reviews/deep_14.md:6594:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5776:./reassess/r_3.md:2992:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:1764:./reviews/deep_14.md:6641:4. 证据闭环：消融围绕三段重定义做。SPG 比基线涨 2.2 Rank-1、2.1 mAP；AHSTG 单独涨 4.4、3.5；没有 DFR 时是 68.2 Rank-1、58.9 mAP，完整模型到 71.8、62.3。Grad-CAM 显示 SPG 看细粒度语义，AHSTG 看可见强响应人体，DFR 后对遮挡响应下降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5777:./reassess/r_3.md:3019:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2180:./validate/v_2.md:5106:./reviews/deep_25.md:14065:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5778:./reassess/r_3.md:3022:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2183:./validate/v_2.md:5118:./reviews/deep_25.md:14144:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5779:./reassess/r_3.md:3026:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2193:./validate/v_2.md:5170:./analyses/deep_25.txt:35:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5780:./reassess/r_3.md:3035:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2220:./validate/v_2.md:5929:./analyses/ALL_ANALYSES.txt:1152:4. **证据闭环**：最强闭环是“换辅助模态”消融。dense correspondence embeddings 全面超过 2D keypoints、silhouettes、part segmentation、3D skeletons、point clouds、SMPL parameters，证明重定义的“稠密对齐表示比稀疏参数更适合”是对的。组件表里，appearance baseline 在 VCCR CC 是 49.7 rank-1、40.2 mAP，完整方法到 59.8、48.6；CCVID 从 80.9、75.3 到 93.9、91.2。MRIE 相比简单融合带来大幅提升，说明不是单纯 gait 分支有效，而是 gait-appearance interaction 有价值。ECS、PCA 可视化、t-SNE 和激活图进一步证明 dense correspondence embedding 稳定，3D gait 比 SMPL 参数更成簇，融合后类间更清楚。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5781:./reassess/r_3.md:3086:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3037:./validate/v_3.md:14880:../litreview2/validate/v_3.md:7080:./validate/v_3.md:2462:../codex_review_ovli.txt:3634:experiments/exp324d/design.md:15:给 DINOv2-base attention 加 LoRA（q/v，rank 8/16，alpha 16），DINO 主权重冻结，只训 LoRA + 轻量头（proj + BNNeck + 全局分类器 + part 分类器），**重遮挡 part-MaxSim mAP 能突破 14 往 competitive 走**。能破 → DINO 创新线可做成；破不动 → 冻结对应特征这条线天花板确实低，止损。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5782:./reassess/r_3.md:3108:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3322:./validate/v_3.md:15165:../litreview2/validate/v_3.md:7365:./validate/v_3.md:2751:../codex_review_ovli.txt:3956:experiments/overnight_innovation_log.md:44:- **exp324d 第2次（本次）已启动**（2026-06-16，lab-3090-d）：`scripts/exp324d_lora.py` 写好——LoRA(q/v rank8 alpha16) 解冻 DINOv2-base + **可微姿态部位池化**(预算每图 row-stochastic 池化矩阵 pool_w，`bmm(pool_w,patch)` 梯度回流 LoRA) + exp324b 同款头/损失/PK64/part-MaxSim eval。**唯一变量=LoRA 解冻 DINO**。双审查 approve（Claude 含 200-trial 池化等价 1.79e-7；Codex 141k tok 确认梯度到 LoRA/full-batch triplet/use_reentrant=False/eval 对称，组合 plausibly new 未找直接先例）。Dry-run：LoRA 294,912+head 413,184 训练，loss↓ acc 0.016→0.766，显存 13.7G/24G，~1.4s/step。全量 35ep 后台跑（micro_bs64 no-ckpt）。**待结果：part-MaxSim 重遮挡 mAP 能否破 exp324b 的 8.65 / 全部破 14.61 往 competitive 走。**【结果待填】
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5783:./reassess/r_3.md:3109:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3323:./validate/v_3.md:15166:../litreview2/validate/v_3.md:7366:./validate/v_3.md:2752:../codex_review_ovli.txt:3957:experiments/overnight_innovation_log.md:49:1. **Rank-disagreement oracle（0-GPU，~40行 numpy，⭐立即做，依赖 Swin distmat）**：重遮挡子集上比 Swin top-k vs DINO part-MaxSim top-k 的 Jaccard / P_dino_only（DINO 命中 Swin 漏的真值）/ per-query oracle 上界。判定：P_dino_only<2% 或 oracle<+1mAP → 整条 DINO⊕Swin 当场止损；oracle +3~5 且 Jaccard<0.5 → 正交坐实，进 #2。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5784:./reassess/r_3.md:3110:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3324:./validate/v_3.md:15167:../litreview2/validate/v_3.md:7367:./validate/v_3.md:2753:../codex_review_ovli.txt:3958:experiments/overnight_innovation_log.md:50:2. **遮挡门控 part-MaxSim re-rank（training-free，最可能产出有用主表结果）**：DINO 共可见 distmat 当 `utils/reranking.py re_ranking(local_distmat=)` 输入，只对重遮挡 query 触发。依赖 #1。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5785:./reassess/r_3.md:3113:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3330:./validate/v_3.md:15173:../litreview2/validate/v_3.md:7373:./validate/v_3.md:2759:../codex_review_ovli.txt:3964:experiments/overnight_innovation_log.md:100:- #2（遮挡门控 part-MaxSim re-rank）**依赖 #1 正向 → 现已被砍**，DINO 的共可见 distmat 当 re-rank 输入不会有增益，不做。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5786:./reassess/r_3.md:3287:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_3.md:3569:区别：FLaN-Net 用语言分人、属性、遮挡物；CFPER 用二维部位专家。我们用三维人体表面和投影可见性定义匹配支持区域，重点解决俯视、低清、遮挡时二维部位顺序失真的问题。  
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5787:./reassess/r_3.md:3370:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:536:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5788:./reassess/r_3.md:3430:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2806:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5789:./reassess/r_3.md:3445:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_14.md:2866:cheap kill-switch：不训练新大模型，先用现有 SMPL 或 2D joints 把图像分成若干 canonical 身体区域，用 SOLIDER 特征做 visibility-aware pooling 和距离重加权。若在 AG-ReID 小子集上最后指标没有超过普通全局特征至少 mAP 0.4、rank1 0.5，先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5790:./reassess/r_3.md:3464:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:217:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5791:./reassess/r_3.md:3499:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3262:廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5792:./reassess/r_3.md:3513:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:3337:廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5793:./reassess/r_3.md:3529:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324h/design.md:27:- **(若 oracle 正)** 顺手 fusion sweep（z-score / min-max，w∈{0..0.5}）+ k-reciprocal re-rank
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5794:./reassess/r_3.md:3530:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324h/design.md:28:  （`utils/reranking.py` re_ranking with local_distmat），看重遮挡/全部能否 > Swin 75 单独。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5795:./reassess/r_3.md:3584:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:363:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5796:./reassess/r_3.md:3648:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:380:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5797:./reassess/r_3.md:3697:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp142/design.md:5:- `exp109` 的 oracle support bank 已明确说明：当前真正的 headroom 不在简单 rerank，而在 **single-image support incomplete**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5798:./reassess/r_3.md:3718:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:363:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5799:./reassess/r_3.md:3776:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:628:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5800:./reassess/r_3.md:3794:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2855:最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5801:./reassess/r_3.md:3800:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2928:最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5802:./reassess/r_3.md:3817:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:460:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5803:./reassess/r_3.md:3883:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:2891:of non-overlapping distributed cameras. In the early stages
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5804:./reassess/r_3.md:3921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:660:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5805:./reassess/r_3.md:3945:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3639:便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5806:./reassess/r_3.md:3957:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3693:便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5807:./reassess/r_3.md:3990:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:359:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5808:./reassess/r_3.md:4068:/Users/abdslm/Desktop/SOLIDER-REID/experiments/lit_review_occluded_2025_2026.md:94:4. **冷静提醒（吸收陷阱推论）**：inter-person lead 要做成"不被吸收"，必须 test-time + cross-instance（pair 条件化判断 Q 的遮挡人是否就是 G）→ 这又落进 **CVK/re-ranker 家族**（已证测试期仅 +0.8~1.1%）。所以 **lead 本身大概率也是小增量 re-rank**，不是训练端主创新。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5809:./reassess/r_3.md:4095:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp323/design.md:55:- **小模型（3B/7B）B 显著 > A 且集中重遮挡** → 真信号 → exp323b：LoRA 微调小 Qwen verifier + 全量 re-rank + 可控遮挡(PLBOA)消融 + GPT-5.5 蒸馏 teacher。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5810:./reassess/r_3.md:4271:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_4.md:21478:../airl_related_work.md:26:RAR 用 query resolution **动态路由**选子空间。AIRL **不路由**——#3 oracle 实测硬路由(area/reliability 阈值)失败(≤+0.41),**软固定先验融合**(w=0.25,clean+recover 两证据头)反而 +1.46。所以 AIRL 的 claim 必须是 "fixed-prior fusion of clean/recover evidence heads under an observation-limited ceiling",**绝不吹成 query-budget routing**(那既撞 RAR 又与实现不符)。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5811:./reassess/r_3.md:4374:/Users/abdslm/Desktop/SOLIDER-REID/experiments/MORNING_REPORT.md:78:**2. ×4 adaptation 发现** + **可复用诊断工具**（rank-disagreement oracle：0-GPU 判两个表征是否互补 + 出 motivation 图）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5812:./reassess/r_3.md:4398:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/design.md:1:# 实验 exp324g: rank-disagreement oracle 诊断（0-训练）
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5813:./reassess/r_3.md:4404:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/design.md:13:  说明 DINO 在重遮挡上有 Swin 拿不到的独立正确信息，值得做遮挡门控 re-rank。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5814:./reassess/r_3.md:4412:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/design.md:40:- oracle +3~5 且 Jaccard<0.5 → "正交性坐实，值得做 #2 遮挡门控 re-rank"。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5815:./reassess/r_3.md:4419:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324g/monitor.md:1:# exp324g monitor — rank-disagreement oracle（0-训练诊断）
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5816:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5817:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5818:./reassess/r_3.md:4495:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:110:"transfer_to_occluded": "Two directly usable ideas (the paper IS occluded ReID, so these are native): (1) The attention-magnitude saliency for foreground/background separation — derive a per-patch foreground probability from a lightweight conv-sigmoid head trained ONLY by a foreground-background contrastive loss (no pose/parsing labels), then suppress occluder/background tokens; this is an external-prior-free occluder gate, attractive because pose estimators fail under occlusion. (2) The importance-ranked cumulative fusion (FHFM): rank patches by CLS-attention*norm and let high-confidence (likely visible) patches contribute to every fusion level while low-importance (likely occluded) patches contribute least — a soft, label-free visibility weighting that downweights corrupted regions without a hard mask. The region-restricted masked attention with learnable part tokens is also a clean parsing-free way to get part features robust to missing regions.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5819:./reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5820:./reassess/r_3.md:4498:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:127:"novel_component": "Intersection-Alignment Score (IAS): turning an explainability signal into a re-ranking operator — measure how much the model's attention overlaps the visible-body mask (IoU) and how mutually consistent query/gallery attention maps are, then asymmetrically shrink or inflate pairwise distances. It makes \"is the model looking at occluded regions?\" a concrete, label-free matching correction rather than a post-hoc visualization.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5821:./reassess/r_3.md:4665:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324d_r32/design.md:24:- 假设成立：重遮挡 mAP 明显超过 base-rank16 与冻结 8.65。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5822:./reassess/r_3.md:4691:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:44:- **exp324d 第2次（本次）已启动**（2026-06-16，lab-3090-d）：`scripts/exp324d_lora.py` 写好——LoRA(q/v rank8 alpha16) 解冻 DINOv2-base + **可微姿态部位池化**(预算每图 row-stochastic 池化矩阵 pool_w，`bmm(pool_w,patch)` 梯度回流 LoRA) + exp324b 同款头/损失/PK64/part-MaxSim eval。**唯一变量=LoRA 解冻 DINO**。双审查 approve（Claude 含 200-trial 池化等价 1.79e-7；Codex 141k tok 确认梯度到 LoRA/full-batch triplet/use_reentrant=False/eval 对称，组合 plausibly new 未找直接先例）。Dry-run：LoRA 294,912+head 413,184 训练，loss↓ acc 0.016→0.766，显存 13.7G/24G，~1.4s/step。全量 35ep 后台跑（micro_bs64 no-ckpt）。**待结果：part-MaxSim 重遮挡 mAP 能否破 exp324b 的 8.65 / 全部破 14.61 往 competitive 走。**【结果待填】
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5823:./reassess/r_3.md:4692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:47:**最关键洞察：先做 0-GPU 的"rank-disagreement oracle"（#1），它是整条"DINO 补 Swin"家族的天花板+kill-switch + 论文 motivation 图。**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5824:./reassess/r_3.md:4693:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:49:1. **Rank-disagreement oracle（0-GPU，~40行 numpy，⭐立即做，依赖 Swin distmat）**：重遮挡子集上比 Swin top-k vs DINO part-MaxSim top-k 的 Jaccard / P_dino_only（DINO 命中 Swin 漏的真值）/ per-query oracle 上界。判定：P_dino_only<2% 或 oracle<+1mAP → 整条 DINO⊕Swin 当场止损；oracle +3~5 且 Jaccard<0.5 → 正交坐实，进 #2。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5825:./reassess/r_3.md:4694:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:50:2. **遮挡门控 part-MaxSim re-rank（training-free，最可能产出有用主表结果）**：DINO 共可见 distmat 当 `utils/reranking.py re_ranking(local_distmat=)` 输入，只对重遮挡 query 触发。依赖 #1。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5826:./reassess/r_3.md:4696:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:58:**执行计划**：exp324f agent 正在 lab-4090 算 Swin distmat → 它一落地我立刻跑 #1 oracle（0-GPU）。正向 → #2 re-rank（training-free 主表素材）。OT 线(#3)等某 GPU 空了上。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5827:./reassess/r_3.md:4699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:69:- 这正是 planner #1 oracle 要量化的"P_dino_only / oracle 上界"的 fusion 侧旁证：简单加权死路 → 下一步直接做 **#1 rank-disagreement oracle** 量化是否存在任何正交信号（DINO 捞到 Swin 漏的真值），它是整条"DINO⊕Swin"家族的天花板 + kill-switch。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5828:./reassess/r_3.md:4701:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:80:- **下一步(最高价值)**: 在 lab-3090-d 跑 planner#1 rank-disagreement oracle(Swin vs DINO 正交性 + oracle 上界)，gate 整条 DINO⊕Swin 家族。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5829:./reassess/r_3.md:4702:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:83:- **exp324d LoRA 实际在健康运行**（我先前误判：grep 错了 log 路径 /tmp/exp324d.log，实际是 /tmp/exp324d_r16.log）。第一个 agent 其实完成了全套：写了可微部位池化脚本(bmm pooling matrix，与 exp324 数值等价 1.8e-6)、**过了 Claude+Codex 双审查(都 approve，确认梯度到 LoRA、无遮挡 ReID 先例)**、launch 了 rank16/alpha16/dropout0.05/grad-ckpt/30ep 训练(PID 309591)。两个并发 agent 撞了一下但收敛到一个健康 run。DINO-in-loop 慢(分钟级/epoch)，epoch5 首评出关键数：part 重遮挡能否破 8.65、全部破 14.61。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5830:./reassess/r_3.md:4705:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:87:### [exp324g — planner#1 rank-disagreement oracle 完成 → DINO⊕Swin 家族止损]
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5831:./reassess/r_3.md:4710:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:100:- #2（遮挡门控 part-MaxSim re-rank）**依赖 #1 正向 → 现已被砍**，DINO 的共可见 distmat 当 re-rank 输入不会有增益，不做。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5832:./reassess/r_3.md:4721:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:155:- **核心问题**：large 或 rank32 能否把重遮挡 mAP 明显推过冻结 8.65 往 competitive（几十分）走？等 e5 第一次 eval（决定性）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5833:./reassess/r_3.md:4722:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:158:**base-LoRA rank16，epoch 5/30**：part 重遮挡 mAP **34.92**(vs 冻结 8.65，**×4.0 +26.3**) · part 全部 **42.45**(vs 14.61) · cos 重遮挡 32.38 · cos 全部 41.06。**才 e5/30，还在陡升。**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5834:./reassess/r_3.md:4728:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:207:1. **真新机制**：LoRA↔visibility **交互**（pose/visibility-conditioned LoRA，或 per-part low-rank experts 按遮挡 gate），带消融证明——不是"加了 LoRA 加了 parts"。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5835:./reassess/r_3.md:4741:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:293:**判断**：FM-import / generic-paradigm-import 方向保持关闭（各有机制，已入 memory+study）。真正没撞墙的下一前沿 = **问题 reframe**（reliability/uncertainty-aware matching 或 common-visible support / pair comparability，CLAUDE.md 钦定）——level-1 重定义创新，多日研究线，非一夜 kill-switch。**拒绝在 3am 编造注定撞同墙的 FM-import 实验充数**（CLAUDE.md：不做组合实验逃避创新）。当前真决策点 = λ=1 e10 oracle（Jaccard vs 0.253）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5836:./reassess/r_3.md:4750:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:365:λ=0 vs λ=1 **同 rank16/seed/script，e30 收敛点 matched oracle**：
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5837:./reassess/r_3.md:4755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:388:TBPS 新主线 + gait/face/跨模态搬运调研(29 agent)。**7 候选，6 死于真实顶会先例(落进已证伪区)**：VG-TBPS/common-support≈MGCC(AAAI24 遮挡TBPS)/PLOT(AAAI25)/PMA(AAAI20 pose-短语对齐)/ProFD(MM24)；其余撞 Visual-Perturbation(AAAI25)、uncertainty-aware TBPS。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5838:./reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5839:./reassess/r_3.md:4768:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:471:- **为何非已关方向**: 与 visibility 加权**正交**(visibility 问"是否被遮挡"二值; burstiness 问"这个可见特征是否被过度计数"集内冗余统计)。一个 part 可完全可见且高 visibility 却 bursty(大块平躯干)——现有方法升权它, burstiness 说降权。非 completion(不重建)、非 scorer 微变体(改特征集聚合)、非 uncertainty(无 per-feature 置信)。**Novelty 已核: 无 occluded-ReID 做 burst-aware/democratic over part/patch 特征**(最近 cousin = Self-similarity guided probabilistic matching ESWA'23 是去噪/校验, 相反符号)。详见 memory [[burstiness-democratic-aggregation-bet]]。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5840:./reassess/r_3.md:4779:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:510:burstiness 死后调研 agent(带"in-domain 死、frozen 会骗人"教训)Rank-1 过审 bet。**为何逃过训练吸收(结构性)**: held-out (occluder类×部位)组合**按构造从没进训练集**→ 训练模型无可隐式吸收。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5841:./reassess/r_3.md:4807:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:15:DINOv2 pose-anchored part-MaxSim vs Swin MaxSim，重遮挡 oracle 检查：top-10 Jaccard 0.062，**P_dino_only 0.20%**(989 query 里只 2 个 DINO 命中而 Swin 漏)，**oracle 上界仅 +0.12 mAP**。低 Jaccard 是"假正交"——DINO 不是补充，是全局太弱(8.65 vs 72.57)。Swin 错的地方 DINO 也错。→ DINO⊕Swin 融合/re-rank 家族全死（两 agent 独立确认）。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5842:./reassess/r_3.md:4809:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:21:LoRA 解冻 DINOv2-base(~1M 可训：0.6M LoRA + 0.4M 头，DINO 主体冻结) + 可微 pose-part-MaxSim：重遮挡 mAP **8.65 → 36.78**(e10，×4.2)，全部 14.61 → 44.67。rank32 36.72、large 待定。**证明瓶颈是 adaptation 不是特征**——这是 oracle"冻结无独立信息"的对照解释。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5843:./reassess/r_3.md:4815:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:51:**收敛点(e30 matched)双确认**：λ=0 vs λ=1 同 rank16/seed/script 跑到 e30 oracle 仍**完全一致**——Jaccard 0.2646 vs 0.2627、oracle +0.85 vs +0.80、fusion best ALL 75.74 vs 75.73（λ=1 甚至略低）。**早期(e10)+收敛(e30) 双证据：解相关在任何训练阶段对互补性零效果。** 加上 decorr-floor 证据（λ=2 双倍权重只把 0.041 降 ~1% → ~0.041 是 ID-constrained floor，共享判别方向是 ID load-bearing 的、删不掉）。→ **张力对显式干预(e10/e30、λ∈{0,1,2}、λ=10 进行中)全程鲁棒，是 fundamental 的，不是可调超参。**
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5844:./reassess/r_3.md:4817:/Users/abdslm/Desktop/SOLIDER-REID/experiments/fm_occluded_reid_study.md:61:3. **可复用诊断工具**：rank-disagreement oracle（0-GPU 判两表征是否互补 + motivation 图）、按可见度子集拆 mAP。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5845:./reassess/r_3.md:4840:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp064/design.md:11:- **核心假设**: 遮挡导致 keypoint 特征的可靠性不同。将特征建模为分布而非点向量，让 uncertainty 自然涌现
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5846:./reassess/r_3.md:4841:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp064/design.md:26:- 如果成功: uncertainty 与遮挡相关 → 自适应距离 → mAP/R1 提升
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5847:./reassess/r_3.md:5019:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp129/monitor.md:47:  2. residual target 没有污染默认 early training，因为 `epoch<=20` 时 `CSRD` 尚未激活
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5938:./reassess/r_3.md:7999:./reassess/r_3.md:4487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:45:"mechanism": "FLaN-Net builds a CLIP-based pipeline that constructs an occlusion-aware fine-grained textual prompt \"A photo of a [S*] person with [A*] partially occluded by [O*],\" where three categorical pseudo-word tokens are learned by textual inversion from the image: S* (subject/identity description), A* (visible attributes), and O* (the occluding object). A* and O* are each produced by a separate fine-grained noise-filtering network in which a small set of learnable queries cross-attend to local patch features (so different query categories latch onto visible-attribute regions vs. occluder regions), then average-pool into a token. A cross-attention module uses the textual prompt as query and visual features as key/value (s_att) to steer the image encoder toward visible, identity-relevant regions, and a dynamic weighting fusion module combines global-visual, textual, and cross-attention features with weights derived from each feature's prediction-entropy (uncertainty), giving reliable features larger weight. Trained with contrastive + triplet + ID + ArcFace losses; inference uses only the image-encoder feature.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5939:./reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5940:./reassess/r_3.md:8007:./reassess/r_3.md:4495:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:110:"transfer_to_occluded": "Two directly usable ideas (the paper IS occluded ReID, so these are native): (1) The attention-magnitude saliency for foreground/background separation — derive a per-patch foreground probability from a lightweight conv-sigmoid head trained ONLY by a foreground-background contrastive loss (no pose/parsing labels), then suppress occluder/background tokens; this is an external-prior-free occluder gate, attractive because pose estimators fail under occlusion. (2) The importance-ranked cumulative fusion (FHFM): rank patches by CLS-attention*norm and let high-confidence (likely visible) patches contribute to every fusion level while low-importance (likely occluded) patches contribute least — a soft, label-free visibility weighting that downweights corrupted regions without a hard mask. The region-restricted masked attention with learnable part tokens is also a clean parsing-free way to get part features robust to missing regions.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5941:./reassess/r_3.md:8009:./reassess/r_3.md:4497:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:126:"transfer_to_occluded": "The Intersection-Alignment Score (IAS) explainable re-ranking is directly occlusion-relevant: compute IoU between the model's activation/attention map and a visible-region (non-occluded) mask, plus a cosine alignment between query and gallery activation maps, then refine the distance matrix — multiply distance by alpha<1 when attention focuses on visible regions AND query/gallery attentions agree, and by beta>1 when attention falls on occluded/missing regions. In occluded ReID this is a self-supervised matching-side trust mechanism: down-weight matches whose discriminative evidence sits on occluder/missing pixels and reward query-gallery pairs whose visible-region attention co-aligns, without needing identity-level supervision. The canonical-UV idea (mapping body regions to fixed coordinates so the same part always aligns) is also a clean way to make two partially-occluded crops compare only on their commonly-visible parts.",
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5942:./reassess/r_3.md:8032:./reassess/r_3.md:4760:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:420:1. ⭐ **因果 backdoor-triplet**(搬 gait GaitC3I TIP25/GaitSCM CVIU24)：把 camera-id+occlusion-state 当显式混淆变量 C, backdoor adjustment 改 triplet 正样本采样, 让跨遮挡/跨摄像头正样本对不再发协变量差异冒充身份的有害梯度。仓库零因果/SCM 代码(唯一命中 DACHM/DACCM 是 training-free reranker, 另一 lever)。**唯一同时满足: 问题重定义+机制新+零代码+避所有已死框架**。遮挡状态从 pose 可见性直接得。0.5天无训练首验。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5943:./reassess/r_3.md:8037:./reassess/r_3.md:4779:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:510:burstiness 死后调研 agent(带"in-domain 死、frozen 会骗人"教训)Rank-1 过审 bet。**为何逃过训练吸收(结构性)**: held-out (occluder类×部位)组合**按构造从没进训练集**→ 训练模型无可隐式吸收。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5953:./reassess/r_3.md:8374:/Users/abdslm/Desktop/SOLIDER-REID/experiments/decisions.md:1576:- 🔵 蓝队（方案 B: PPE）核心论点: 重新定义遮挡问题（从 visibility mask 到 uncertainty quantification），pose confidence → feature uncertainty 的因果链是全新机制。论文理论深度更强（概率嵌入+信息几何）。与 PSG 正交互补（PSG 管特征提取，PPE 管匹配）。P3E/DUL 已验证概率嵌入在 ReID 中可训练。攻击 A: PAB 已在同范式失败（attention bias 变体），CLAUDE.md 警告"attention trick"，2.8K 参数太小难以撑论文主贡献，12x4 分辨率太粗糙。信心: 7/10
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6074:/bin/zsh -lc "find . -maxdepth 5 -iname '*CFPER*' -o -iname '*Coarse*Fine*Part*'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6077:/bin/zsh -lc "rg -n \"CFPER|Coarse-to-Fine Part-Experts|Part-Experts Retrieval\" all_papers.txt reviews analyses reassess pivot validate -g '*.md' -g '*.txt'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6092:reassess/r_3.md:3287:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_3.md:3569:区别：FLaN-Net 用语言分人、属性、遮挡物；CFPER 用二维部位专家。我们用三维人体表面和投影可见性定义匹配支持区域，重点解决俯视、低清、遮挡时二维部位顺序失真的问题。  
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6157:/bin/zsh -lc "sed -n '2110,2860p' reviews/deep_5.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6218:pose variation, or subtle inter-class variation, require finegrained part features to support more detailed matching.
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6272:are generally sufficient to support accurate matching for “easy”
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6382:In the fine inference stage, we employ PMoE to extract finegrained part features for “hard” queries to support detailed
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6496:cues to provide contextual support for part feature learning.
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6912:/bin/zsh -lc "sed -n '1560,1595p' analyses/ALL_ANALYSES.txt" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6914:/bin/zsh -lc "sed -n '8390,8495p' reviews/deep_5.md" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:6921:2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7026:2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7140:/bin/zsh -lc "nl -ba reviews/deep_5.md | sed -n '2114,2295p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7197:  2169	pose variation, or subtle inter-class variation, require finegrained part features to support more detailed matching.
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7251:  2223	are generally sufficient to support accurate matching for “easy”
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7326:/bin/zsh -lc "nl -ba reviews/deep_5.md | sed -n '2514,2860p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7677:/bin/zsh -lc "nl -ba analyses/ALL_ANALYSES.txt | sed -n '1570,1587p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7733:/bin/zsh -lc "nl -ba reviews/deep_5.md | sed -n '2280,2335p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7788:  2333	In the fine inference stage, we employ PMoE to extract finegrained part features for “hard” queries to support detailed
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7793:/bin/zsh -lc "nl -ba reviews/deep_5.md | sed -n '2335,2385p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7859:| (a) CFPER / query-difficulty adaptive ReID | CFPER, ICME 2025，本地 PDF 抽取见 [reviews/deep_5.md](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:2130)，DOI `10.1109/ICME59968.2025.11209843` | **否**。它用 global-patch 相似度一阶差分估计可见人体区域比例，`easy` 只走 global，`hard` 走 part experts，见 [signal lines](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:2296) | **高杀伤但不硬撞**。它已占“ReID query difficulty adaptive compute / early exit”主叙事；你的新意必须落在“rank disagreement 不是可见性/置信度/难度分数” |
experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:7887:| (a) CFPER / query-difficulty adaptive ReID | CFPER, ICME 2025，本地 PDF 抽取见 [reviews/deep_5.md](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:2130)，DOI `10.1109/ICME59968.2025.11209843` | **否**。它用 global-patch 相似度一阶差分估计可见人体区域比例，`easy` 只走 global，`hard` 走 part experts，见 [signal lines](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:2296) | **高杀伤但不硬撞**。它已占“ReID query difficulty adaptive compute / early exit”主叙事；你的新意必须落在“rank disagreement 不是可见性/置信度/难度分数” |
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:113:    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:116:        --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:117:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:122:        --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:123:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:128:        --data_root /root/work/SOLIDER-REID/data \
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:129:        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp_ovli \
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1139:#     swin_small are both supported (the second forward just reuses `model`).
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1235:    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1263:    # 'swin_small' = SOLIDER Swin-Small (team asset, SOTA push): SOLIDER teacher
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1264:    # pretrain + avg-pool + BNNeck, in_planes=768; AFD freq modules are NOT
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1265:    # supported on swin (OVP/OVLI are independent and DO work). OVLI hooks the
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1270:                         "or swin_small (SOLIDER Swin-Small, in_planes=768)")
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1271:    ap.add_argument('--swin_pretrain', default='',
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1272:                    help="path to the SOLIDER swin_small.pth teacher checkpoint "
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1273:                         "(e.g. <repo>/pretrained/swin_small.pth). Empty -> train "
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1277:                    help="SOLIDER semantic weight for the Swin backbone "
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1559:        ap.error("--backbone swin_small does not support --use_afd (AFD modules "
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1564:    # are supported and back-compatible:
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1696:        print(f"  backbone=swin_small (SOLIDER, in_planes=768) "
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1697:              f"pretrain={args.swin_pretrain or 'NONE (from scratch)'} "
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1810:    # SAFE for resnet50 but DIVERGES the ~50M-param SOLIDER Swin transformer --
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1816:    # backbone LR than the randomly-initialised heads (the repo's main SOLIDER
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1931:    # LR group (pretrained weights, same as f_full's stages).  Assert both placements
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1948:        # the rec late stage must be on the SCALED Swin LR group (it is pretrained
experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1958:                                       "Swin LR group (it is pretrained backbone "
experiments/exp252/design.md:12:- 其余与 exp249 相同：Small LGPA-D+GCN+OA-SD+PLBOA+WITH_CP
experiments/cargo_cvpb/hubness_logs/hub_smoke.log:12:/root/work/SOLIDER-REID/model/backbones/swin_transformer.py:1193: UserWarning: DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead
experiments/cargo_cvpb/hubness_logs/hub_smoke.log:13:  warnings.warn('DeprecationWarning: pretrained is deprecated, '
experiments/cargo_cvpb/hubness_logs/hub_smoke.log:20:Loading pretrained model from /root/work/SOLIDER-REID/log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth
experiments/exp324d/design.md:1:# 实验 exp324d: LoRA 解冻 DINOv2-base + 姿态部位匹配（破 14-mAP 天花板）
experiments/exp324d/design.md:3:> **来源**：exp324b（冻结 DINO + 433K 轻量头）e20 即到顶：part 重遮挡 8.65 / 全部 14.61 mAP。机制对（姿态部位匹配 > 整图、可训 ×4.6），**但冻结特征天花板低**（vs exp255 Swin 75）。
experiments/exp324d/design.md:5:> **机器**：lab-3090-d（DINOv2-base 已下、RTX 3090 24G idle、peft 0.19.1 已装）。
experiments/exp324d/design.md:9:- exp324b 证明：把冻结 DINO 部位特征投到 ReID 空间能从 1.86 → 14.61（×7.9），机制（姿态锚定 + 只比可见部位 part-MaxSim）干净有效。
experiments/exp324d/design.md:10:- **但天花板是冻结特征本身**：DINO 自监督预训练没见过 ReID 判别目标，dense token 对"同一人不同图"的判别力有上限，轻量头拉不动（e20 到顶）。
experiments/exp324d/design.md:11:- 要 competitive，必须让 backbone 适应 ReID。全量微调 12 层 768d ×15K 数据会过拟合 + 成本大。**LoRA**：只在 attention q/v 注入低秩适配（rank 8/16），DINO 主权重冻结，参数量小、过拟合风险低，是"让冻结特征端动起来"的最小代价方案。
experiments/exp324d/design.md:15:给 DINOv2-base attention 加 LoRA（q/v，rank 8/16，alpha 16），DINO 主权重冻结，只训 LoRA + 轻量头（proj + BNNeck + 全局分类器 + part 分类器），**重遮挡 part-MaxSim mAP 能突破 14 往 competitive 走**。能破 → DINO 创新线可做成；破不动 → 冻结对应特征这条线天花板确实低，止损。
experiments/exp324d/design.md:20:- exp324b 在**缓存的部位特征**上训（DINO 不在循环里 forward，飞快）。
experiments/exp324d/design.md:21:- exp324d 中 LoRA 改变 DINO 输出 → **不能再用缓存特征**。每 step 必须：图 → DINO(LoRA) forward → 姿态部位池化（**可微**）→ 轻量头 → 损失。慢但 3090 可承受。
experiments/exp324d/design.md:26:2. DINO(LoRA) forward → `last_hidden_state` → 去 CLS → `(B, GRID_H*GRID_W, 768)`。
experiments/exp324d/design.md:32:- DINO forward in loop + 448×224 输入，3090 24G 显存可能不够喂满 BS64。
experiments/exp324d/design.md:33:- 策略：物理 micro-batch 可降（如 16 或 32），用**梯度累积**凑有效 BS=64。PK 采样仍按 P16×K4=64 一个完整逻辑 batch 出，micro-batch 是对这 64 个样本切片做累积。**triplet 必须在完整 64 个样本上算**（batch-hard 需要全 batch 的正负对）→ 所以策略：先 forward 全 64 个样本的 DINO（no_grad 分块？不行，要梯度）。
experiments/exp324d/design.md:34:- **正确做法**：对 64 样本分 micro-chunk forward+保留计算图会爆显存。改用 **gradient checkpointing**（DINO encoder layer 级）让 64 样本一次 forward 显存可控；若仍不够，micro-batch 累积时 triplet 单独处理（见下）。
experiments/exp324d/design.md:38:- `peft.LoraConfig(target_modules=["query","value"], r=8/16, lora_alpha=16, lora_dropout=0.0)`，DINOv2 attention 是分离的 query/key/value Linear（已验证），suffix 匹配命中全 12 层 q/v。
experiments/exp324d/design.md:39:- DINO 主权重 `requires_grad_(False)`；peft `get_peft_model` 自动只放开 LoRA。轻量头单独 requires_grad。
experiments/exp324d/design.md:46:- rank 8（先），alpha 16，lora_dropout 0；LoRA LR 1e-4，head LR 3.5e-4（Adam）；cosine LR；epoch 30-40（带 DINO forward 慢，看趋势够）；part_weight 0.5；margin soft。eval period 5（epoch 少，密点看轨迹）。
experiments/exp324d/design.md:50:- 假设成立：重遮挡 part-MaxSim mAP 破 14 明显上升（进入 20-40+ → DINO 线有救、competitive 可期）。
experiments/exp324d/design.md:55:- **baseline**：exp324b 冻结头 e60（part 重遮挡 8.65 / 全部 14.61 / cos 全部 13.51 / cos 重遮挡 7.32）。**唯一变量 = LoRA 解冻 DINO**（头/损失/采样/eval 全同）。
experiments/cargo_cvpb/smoke_airl_iso.py:27:The swin backbone is built FROM SCRATCH (swin_pretrain='') on CPU at a small input so
experiments/cargo_cvpb/smoke_airl_iso.py:58:        param-group split; the rec late stage (pretrained-recipe backbone weight)
experiments/cargo_cvpb/smoke_airl_iso.py:127:# imported-but-unused in the SOLIDER swin_transformer.py -> stub it so the swin
experiments/cargo_cvpb/smoke_airl_iso.py:174:    return AFDModel(num_classes=num_classes, pretrained=False,
experiments/cargo_cvpb/smoke_airl_iso.py:175:                    backbone='swin_small', swin_pretrain='', img_size=img_size,
experiments/cargo_cvpb/smoke_airl_iso.py:532:    except Exception as e:                      # CPU autocast unsupported -> fp32 only
experiments/cargo_cvpb/litreview/reviews/lit_17.md:17:- SOLIDER-Swin 强 backbone(自监督人体预训练,in_planes=768)
experiments/cargo_cvpb/litreview/reviews/lit_17.md:107:- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:195:## 8. 代码底子（上游 SOLIDER-REID）
experiments/cargo_cvpb/litreview/reviews/lit_17.md:198:- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:210:- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:279:**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:281:**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:283:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:310:方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:327:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:329:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:425:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:426:- **exp017 探针放行候选 A**：SOLIDER 特征对 CLIP 遮挡物语义零线性解码（R²=-0.01）、嵌入级超锥零假设解释力仅约 26%、残差与 AP 相关 -0.164（控 s_top1 后 -0.097 贴线）。按预注册中间地带规则的属性级判据放行。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:442:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:825:This work was supported in part by the Research Project of ZJULeague Research and Development Center, Zhejiang Laboratory under Grant
experiments/cargo_cvpb/litreview/reviews/lit_17.md:966:a comprehensive pipeline for generating dense paired 2D3D data, comprising 3D vertex embedding pretraining,
experiments/cargo_cvpb/litreview/reviews/lit_17.md:1330:thereby supporting persistent object segmentation. In the domain of motion tracking, [38] develop a salient event blob detector that identiﬁes regions with consistent optical ﬂow through a novel Field of Active Flow Directions (FAFD) representation constructed from the Surface
experiments/cargo_cvpb/litreview/reviews/lit_17.md:1890:at enabling retrieval at any time moment and across different time intervals. We contribute for the first time a largescale dataset named AT-USTC to support the study of ATReID. Compared to existing datasets, AT-USTC stands out
experiments/cargo_cvpb/litreview/reviews/lit_17.md:2043:4 June 2025. This work was supported in part by the National Natural
experiments/cargo_cvpb/litreview/reviews/lit_17.md:2128:bn1, relu, and maxpool layers of the pretrained ResNet50 on
experiments/cargo_cvpb/litreview/reviews/lit_17.md:2327:support for tracking suspects and ﬁnding lost people. Due to the impact
experiments/cargo_cvpb/litreview/reviews/lit_17.md:2526:propose a feature completion decoder that aggregates potential information to generate occluded features. Li et al. [9] introduced a secondorder attention mechanism to model high-order semantic relationships
experiments/cargo_cvpb/litreview/reviews/lit_17.md:2741:that text lacks stable local structural support in feature space. These
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3133:这批能发的工作，大多不是“模块更复杂”，而是先把一个默认假设打掉：模态一定平衡、数据一定同场景、3D 表征够用、pose 一定可靠、ReID 只发生在单一时间尺度。我们的强点也应该这么讲，优先用团队已有的 aerial-ground、SMPL、pose gating 和 SOLIDER，而不是再做普通注意力模块。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3137:挂靠资产：aerial-ground 数据和代码，SMPL mesh 投影，SOLIDER-Swin 特征。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3140:cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3143:挂靠资产：aerial-ground，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3144:核心想法：现实里地面图多、航拍图少且低清，不能假设双视角平衡。用少量航拍样本估计 view-style 分布，在 SOLIDER 特征空间做航拍式扰动，而不是图像生成。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3149:挂靠资产：SMPL 3D 几何，pose 热图门控，当前 lifelong ReID 台子，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3155:挂靠资产：pose 热图门控，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3156:核心想法：专攻非目标行人遮挡，不把车、树、包和另一个人混成一种遮挡。多人体 pose heatmap 负责分人，SOLIDER 纹理负责保留身份细节。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3212:这批能发的工作，大多不是“模块更复杂”，而是先把一个默认假设打掉：模态一定平衡、数据一定同场景、3D 表征够用、pose 一定可靠、ReID 只发生在单一时间尺度。我们的强点也应该这么讲，优先用团队已有的 aerial-ground、SMPL、pose gating 和 SOLIDER，而不是再做普通注意力模块。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3216:挂靠资产：aerial-ground 数据和代码，SMPL mesh 投影，SOLIDER-Swin 特征。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3219:cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3222:挂靠资产：aerial-ground，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3223:核心想法：现实里地面图多、航拍图少且低清，不能假设双视角平衡。用少量航拍样本估计 view-style 分布，在 SOLIDER 特征空间做航拍式扰动，而不是图像生成。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3228:挂靠资产：SMPL 3D 几何，pose 热图门控，当前 lifelong ReID 台子，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3234:挂靠资产：pose 热图门控，SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_17.md:3235:核心想法：专攻非目标行人遮挡，不把车、树、包和另一个人混成一种遮挡。多人体 pose heatmap 负责分人，SOLIDER 纹理负责保留身份细节。  
experiments/exp252/monitor.md:3:配置: Swin-Small + PSG Stage2+3 + PAA + LGPA-D (detach) + GCN (detach) + OA-SD + PLBOA(0.7) + WITH_CP
experiments/exp252/monitor.md:18:**当前**: Loss 11.75, ID_G 6.537, Acc 0.016, OA-SD 0.398, LGPA 7.14
experiments/exp252/monitor.md:27:**当前**: Loss 10.48, ID_G 6.294, Acc 0.253, OA-SD 0.047, LGPA 7.01
experiments/exp252/monitor.md:42:**当前**: Loss 9.18, ID_G 5.088, Acc 0.158, OA-SD 0.015, LGPA 6.491
experiments/exp324d/monitor.md:1:# exp324d 监控 — LoRA 解冻 DINOv2-base + 姿态部位匹配（破 14-mAP 天花板）
experiments/exp324d/monitor.md:6:变量：唯一 = LoRA 解冻 DINO（q/v rank8 alpha16），头/损失/采样/eval 全同 exp324b。
experiments/exp324d/monitor.md:49:- **e2 强正信号**：loss 10.36→4.63，acc 0.409→**0.831**；**d_ap=30.9 / d_an=36.5**（正负已分开 Δ5.6，e1 时几乎相等）→ embedding 快速变 ReID-判别。LoRA 让 DINO 适应 ReID 的迹象明确。
experiments/cargo_cvpb/hub_failure_characterize.py:22:ap.add_argument('--gallery_dir', default='/root/work/SOLIDER-REID/data/occluded_duke/bounding_box_test')
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:211:**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:213:**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:215:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:242:方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:259:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:261:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:357:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:358:- **exp017 探针放行候选 A**：SOLIDER 特征对 CLIP 遮挡物语义零线性解码（R²=-0.01）、嵌入级超锥零假设解释力仅约 26%、残差与 AP 相关 -0.164（控 s_top1 后 -0.097 贴线）。按预注册中间地带规则的属性级判据放行。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:374:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:624:- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:712:## 8. 代码底子（上游 SOLIDER-REID）
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:715:- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:727:- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:877:`SOLIDER-REID/` 是干净的上游代码，来源是 `github.com/tinyvision/SOLIDER-REID`，版本是 `8c08e1c`。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这个目录里从零写。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:1049:## SOLIDER-REID 代码底子
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:1053:用 SOLIDER 的预训练权重。从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:3199:    43	October 16, 2024, January 31, 2025. This work was supported by the National
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:3772:    45	This work was supported by the National Natural Science Foundation of
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:4596:    69	ResNet-50 [11] pretrained on ImageNet as the backbone network, with the convolutional stride of
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:5095:12:and fully fine-tune the entire network, which have shown notable performance improvements compared to uni-modal pretraining models. However, full-tuning a large model is prone
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:5497:764:from pretrained ResNet50; (b) represents the feature distribution visualization after passing through the AGW
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:6052:  1790	layernorm will directly destroy the knowledge of the pretrained CLIP.
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:6134:   750	distribution from pretrained ResNet50, has the least ideal effect. The 𝐵𝑎𝑠𝑒 shows the feature distribution extracted by the AGW [48] baseline, where the clustering effect is poor, with significant
experiments/cargo_cvpb/litreview2/reviews/deep_28.md:6148:   764	from pretrained ResNet50; (b) represents the feature distribution visualization after passing through the AGW
experiments/exp252/claude_review.md:17:- **Innovation concern**: This is a config combination experiment, NOT a new mechanism. Per CLAUDE.md rules, combination experiments should NOT be the main line. However, since this tests multi-stage injection (never tested before for PSG), it provides useful ablation evidence for the paper narrative. Acceptable as a supporting experiment.
experiments/exp252/claude_review.md:223:- exp251 inherits from exp246b recipe (Tiny LGPA-D+GCN+OA-SD+PLBOA). Only changes: PSG_STAGES and ADDITIVE_ADAPTER. CORRECT.
experiments/exp252/claude_review.md:224:- exp252 inherits from exp249 recipe (Small LGPA-D+GCN+OA-SD+PLBOA+WITH_CP). Only changes: PSG_STAGES and ADDITIVE_ADAPTER. CORRECT.
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:42:# Swin-Small backbone (SOLIDER) -- optional, team asset, for SOTA push.
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:46:# Repo root = .../SOLIDER-REID (this file lives at experiments/afd_reid/).
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:52:    """The SOLIDER swin_transformer.py does, at import time,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:78:    """Thin wrapper around the SOLIDER swin_small backbone for the AFD/OVLI
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:95:    def __init__(self, img_size=(256, 128), pretrain_path='', semantic_weight=0.2,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:111:            semantic_weight=semantic_weight,   # SOLIDER ReID default 0.2
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:115:        if pretrain_path:
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:116:            # loads the SOLIDER 'teacher' checkpoint (backbone.* keys), strict=False
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:117:            self.swin.init_weights(pretrain_path)
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:167:            # deep-copy preserves the pretrained weights as the f_rec init (same
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:530:                 pretrained=True,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:537:                 swin_pretrain='', swin_semantic_weight=0.2,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:585:            weights = 'IMAGENET1K_V1' if pretrained else None
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:616:            # SOLIDER Swin-Small (team asset, SOTA push).  AFD frequency modules
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:618:            # exist in Swin -> AFD is unsupported here (OVLI is the headline and
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:620:            assert not use_afd, ("backbone='swin_small' does not support the AFD "
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:625:                img_size=tuple(img_size), pretrain_path=swin_pretrain,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:636:            # (== SOLIDER's native avgpool head over the same map).
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:874:    builds the SOLIDER Swin-Small backbone instead (img_size / swin_pretrain /
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:875:    swin_semantic_weight are read from args, with the SOLIDER ReID defaults).
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:881:        pretrained=True,
experiments/cargo_cvpb/airl_codex_bundle/code/afd_model.py:891:        swin_pretrain=getattr(args, 'swin_pretrain', ''),
experiments/exp324d/claude_review.md:1:# Claude Broad Review — exp324d (LoRA-unfrozen DINOv2 + pose part-MaxSim)
experiments/exp324d/claude_review.md:14:eval `encode_split` 与 exp324b 完全相同的 pooling+head+normalization，唯一差异是 parts 来自 live DINO+pool 而非缓存（即预期单变量）。distmats/eval_func/heavy mask 全是同一 import 函数。PASS。
experiments/exp324d/claude_review.md:17:`assert bs==64` 硬守卫。micro_bs 只切 DINO forward 再 `torch.cat` 拼回完整 64 → batch-hard triplet 见全 64 样本。每逻辑 batch 恰好一次 zero_grad/backward/step；micro-batch 是激活内存分块非梯度累积。PASS。
experiments/exp324d/claude_review.md:29:heavy-occ mask（vis≤8）、same-cam 排除（eval_func 内）、distmat（cosine + part-MaxSim）全不变。无 autocast，DINO float32，pool_w float32 → bmm dtype 一致；bn/pp `.float()` 后转 numpy。无 dtype mismatch。PASS。
experiments/exp324d/claude_review.md:32:唯一偏离 exp324b = LoRA 解冻 DINO（+ 机制必需的"缓存→live forward + 可微池化"）。head/损失/权重/采样/eval/distmat/heavy mask 全 import 复用。缓存目录(`exp324d/_cache`)+tag前缀(`_pool_`)与 exp324b 不冲突。干净单变量。
experiments/exp324d/claude_review.md:39:- L2 save_pretrained 只存 LoRA adapter（peft 行为，小 ckpt，预期）。
experiments/exp324d/claude_review.md:44:**审查通过**（approve）。所有硬门通过：可微池化数值等价（1.79e-7）、梯度证实到达 LoRA、train/eval 对称、有效 BS=64 + 完整 batch-hard triplet + 单次 step、use_reentrant=False 正确且 dry-run 验证、优化器分组完整不相交 BN 排除 WD、peft 仅 LoRA 可训、eval 复用 exp324b/exp324 函数。LoRA 解冻 DINO 为相对 exp324b 唯一变量。
experiments/cargo_cvpb/airl_codex_bundle/launch_10codex.sh:3:BUNDLE=/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
experiments/exp255/design.md:1:# exp255: Small LGPA-D + GCN hidden 512 + 2-stage PSG + OA-SD
experiments/exp255/design.md:14:- 其余与 exp249 相同: Small + LGPA-D + GCN + OA-SD + PLBOA
experiments/cargo_cvpb/cvpb_osac_od.log:20:/root/work/SOLIDER-REID/model/backbones/swin_transformer.py:1193: UserWarning: DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead
experiments/cargo_cvpb/cvpb_osac_od.log:21:  warnings.warn('DeprecationWarning: pretrained is deprecated, '
experiments/cargo_cvpb/cvpb_osac_od.log:28:Loading pretrained model from /root/work/SOLIDER-REID/log/occluded_duke/exp260_base_gcn512_2stage/transformer_20.pth
experiments/cargo_cvpb/cvpb_osac_od.log:61:Loading pretrained model from /root/work/SOLIDER-REID/log/occluded_duke/exp260_base_gcn512_2stage/transformer_40.pth
experiments/cargo_cvpb/cvpb_osac_od.log:94:Loading pretrained model from /root/work/SOLIDER-REID/log/occluded_duke/exp260_base_gcn512_2stage/transformer_60.pth
experiments/cargo_cvpb/cvpb_osac_od.log:127:Loading pretrained model from /root/work/SOLIDER-REID/log/occluded_duke/exp260_base_gcn512_2stage/transformer_80.pth
experiments/cargo_cvpb/cvpb_osac_od.log:160:Loading pretrained model from /root/work/SOLIDER-REID/log/occluded_duke/exp260_base_gcn512_2stage/transformer_100.pth
experiments/cargo_cvpb/cvpb_osac_od.log:193:Loading pretrained model from /root/work/SOLIDER-REID/log/occluded_duke/exp260_base_gcn512_2stage/transformer_120.pth
experiments/exp255/monitor.md:1:# exp255 Small GCN hidden 512 + 2-stage PSG + LGPA-D + OA-SD 监控
experiments/exp255/monitor.md:3:配置: Swin-Small + 2-stage PSG + LGPA-D + GCN hidden=512 + OA-SD + PLBOA + WITH_CP
experiments/exp255/monitor.md:242:exp256 (Pose Prompt) ep2, healthy. OA-SD 0.527 (teacher starting).
experiments/cargo_cvpb/litreview/reviews/lit_1.md:17:- SOLIDER-Swin 强 backbone(自监督人体预训练,in_planes=768)
experiments/cargo_cvpb/litreview/reviews/lit_1.md:410:**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:412:**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:414:重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:441:方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:458:- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:460:- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:556:- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:557:- **exp017 探针放行候选 A**：SOLIDER 特征对 CLIP 遮挡物语义零线性解码（R²=-0.01）、嵌入级超锥零假设解释力仅约 26%、残差与 AP 相关 -0.164（控 s_top1 后 -0.097 贴线）。按预注册中间地带规则的属性级判据放行。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:573:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:591:- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:679:## 8. 代码底子（上游 SOLIDER-REID）
experiments/cargo_cvpb/litreview/reviews/lit_1.md:682:- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:694:- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:772:29 October 2023. This work was supported in part by the National Natural
experiments/cargo_cvpb/litreview/reviews/lit_1.md:1127:This work was supported in part by Guangdong Provincial Key Laboratory of
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2042:that combine pretext tasks for skeleton representation learning with labeled fine-tuning, and (3) unsupervised SRID
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2366:samples. Then, ResNet-50 is employed as a pretrained extractor to extract features from pedestrian images and obtain the attention weights
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2422:given a natural-language description, yet existing methods still rely on instance-level similarity from pretrained
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2449:ALBEF [5], pretrained with contrastive objectives [2,6], have provided
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2488:itself, allowing extraction without modifying the pretrained backbone
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2495:combines QP-Similarity and QG-Similarity to correct mismatched rankings without modifying pretrained backbones. Its cascaded extension,
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2509:training data or parameter updates. The framework can be seamlessly attached to existing pretrained VLM-based TI-ReID models.
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2537:and domain-aligned adaptation to reduce the synthetic-to-real pretraining gap [18]. Collectively, these methods primarily advance TI-ReID by
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2555:mainly exploit prototypes to shape training signals rather than to support identity-level reasoning at inference. Combined with the modality
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2567:large-scale pretraining. T5 [8] unifies text-to-text and image-to-text
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2951:• To support effective attribute-level optimization, we identify and
experiments/cargo_cvpb/litreview/reviews/lit_1.md:2978:To effectively support attribute-level alignment, we propose a dualdimensional and dual-directional attribute pairing strategy that defines
experiments/cargo_cvpb/litreview/reviews/lit_1.md:3425:挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:3427:低成本首验：冻结 SOLIDER，在 CARGO 或 AG-ReID.v2 小子集上用现有 SMPL 投影做表面分区池化，对比全局特征、水平条带、2D pose part。若规范表面池化不能稳定提高跨视角正负间隔，先杀。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:3430:挂靠资产：aerial-ground、SMPL、SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:3435:挂靠资产：aerial-ground、SOLIDER-Swin、pose 热图门控，必要时加 SMPL 可见性。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:3499:挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:3501:低成本首验：冻结 SOLIDER，在 CARGO 或 AG-ReID.v2 小子集上用现有 SMPL 投影做表面分区池化，对比全局特征、水平条带、2D pose part。若规范表面池化不能稳定提高跨视角正负间隔，先杀。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:3504:挂靠资产：aerial-ground、SMPL、SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:3509:挂靠资产：aerial-ground、SOLIDER-Swin、pose 热图门控，必要时加 SMPL 可见性。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4433:挂靠资产：CARGO/AG-ReID.v2、SMPL、pose 热图门控、SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4435:便宜首验：冻结 SOLIDER，在 AG-ReID/CARGO 上只做 SMPL 部位投影池化和可见性重加权，不训练或只训一层线性融合。若跨视角 mAP/rank1 连噪声线都过不了，直接杀。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4438:挂靠资产：aerial-ground、SOLIDER-Swin、SMPL。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4440:便宜首验：把地面图按航拍尺度退化，比较 SOLIDER 特征在低频、中频、高频分量上的同身份稳定性。若高频降权不能提升正负样本间隔，杀。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4448:挂靠资产：SOLIDER-Swin、aerial-ground、SMPL。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4502:挂靠资产：CARGO/AG-ReID.v2、SMPL、pose 热图门控、SOLIDER-Swin。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4504:便宜首验：冻结 SOLIDER，在 AG-ReID/CARGO 上只做 SMPL 部位投影池化和可见性重加权，不训练或只训一层线性融合。若跨视角 mAP/rank1 连噪声线都过不了，直接杀。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4507:挂靠资产：aerial-ground、SOLIDER-Swin、SMPL。  
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4509:便宜首验：把地面图按航拍尺度退化，比较 SOLIDER 特征在低频、中频、高频分量上的同身份稳定性。若高频降权不能提升正负样本间隔，杀。
experiments/cargo_cvpb/litreview/reviews/lit_1.md:4517:挂靠资产：SOLIDER-Swin、aerial-ground、SMPL。  
experiments/exp344_pose_cond_prompt/claude_review.md:103:其余（ViT-L-14 权重、GLOBAL_LOSS_SCALE 1.0、POSE_TEST_FEAT global、关 PSG/LGPA/PLBOA/OA-SD/多视图）与 exp341 一致。
experiments/exp255/claude_review.md:1:# Claude Review -- exp255: Small LGPA-D + GCN hidden 512 + 2-stage PSG + OA-SD
experiments/exp255/claude_review.md:157:GCN forward pass is identical in train and test. PSG modules are residual gating (no dropout, no stochastic behavior outside standard train/eval mode switching). LGPA, OA-SD, PLBOA only affect training. Test uses `pose_test_feat` config (inherited as `equal_concat` from base config). **CORRECT.**
experiments/exp255/claude_review.md:175:This experiment is a hyperparameter sweep (GCN hidden 256 -> 512). It is not an innovation experiment. However, the context is clear: this is part of pushing the Small backbone recipe toward best results, not a main-line creative experiment. As a supporting capacity ablation for the paper's ablation table, it is acceptable.
experiments/library_167_synthesis.md:7:Occluded-Duke 的遮挡**大比例是"另一个行人"而非物体**。全领域(167篇)把遮挡当"信息缺失→补全(completion/pose/part/CLIP)"。**真正失败不是缺信息, 而是 target embedding 被遮挡那个人的身份污染**——一个 distractor identity 泄漏进来。这是 source-separation 问题, 与全场"补全"范式正交。**没有一篇这么做。**
experiments/library_167_synthesis.md:34:库里所有遮挡论文几乎全踩你死亡清单: [090/124] pose-guided+mask、[101/080/093/120] feature completion、[132/067/015/022] CLIP遮挡prompt、[009/081] part/visibility加权。**说明"backbone 上加抗遮挡模块"连 B 类都在原地踏步——但反过来: 没人质疑这些增益是不是通用正则(你的 wall-2 没人碰)=空白。**
experiments/exp121/design.md:5:- `exp120` 正在验证：support-complete teacher 是否能把 `exp109` 的 headroom 接到 `exp119` 的 relational distillation 上
experiments/exp121/design.md:8:  **support-complete relational teacher 是否也需要稳定化**
experiments/exp121/design.md:16:1. 如果 `exp120` 的主要风险也来自 online teacher non-stationarity，那么在 support-complete bank 已初步成熟后冻结更新，可能会比持续在线更新更稳
experiments/exp121/design.md:20:3. 若 freeze30 优于 exp120，说明后续主方法应把“support-complete”与“stable teacher”一起写
experiments/exp121/design.md:35:- 本地主实验：`exp120 SCRD`（online support-complete teacher）
experiments/exp121/design.md:41:  - 说明 support-complete teacher 也存在 non-stationary / hardening 问题
experiments/exp121/design.md:42:  - 下一步应继续沿“stable support-complete relational teacher”写主方法
experiments/exp121/design.md:50:3. 冻结过早可能让 bank 还没积累够 support，导致 teacher 反而变弱
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:4:workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:46:/bin/zsh -lc 'rg --files code logs | sort' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:57:/bin/zsh -lc 'wc -l code/afd_model.py code/afd_train.py logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log logs/agreidv2_baseline_4090.log logs/agreidv2_baseline_4090_s2.log logs/agreidv2_baseline.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:69:/bin/zsh -lc 'git status --short' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:88:/bin/zsh -lc 'rg -n "airl|dualbranch|iso|rec|full|fuse|do_inference|seed|manual_seed|DataLoader|worker|sampler|init|kaiming|random|cudnn|Generator|set_seed|make_dataloader|build" code/afd_model.py' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:94:95:    def __init__(self, img_size=(256, 128), pretrain_path='', semantic_weight=0.2,
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:98:117:            self.swin.init_weights(pretrain_path)
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:133:167:            # deep-copy preserves the pretrained weights as the f_rec init (same
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:374:874:    builds the SOLIDER Swin-Small backbone instead (img_size / swin_pretrain /
experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_6.md:381:/bin/zsh -lc 'rg -n "airl|dualbranch|iso|rec|full|FUSE|do_inference|seed|manual_seed|DataLoader|worker|sampler|init|kaiming|random|cudnn|Generator|set_seed|make_dataloader|train_loader|shuffle|drop_last|--airl" code/afd_train.py' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
experiments/cargo_cvpb/airl_codex_bund和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1237:experiments/cargo_cvpb/litreview2/train_more_import.md:1392:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1239:experiments/cargo_cvpb/litreview2/train_more_import.md:1394:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1240:experiments/cargo_cvpb/litreview2/train_more_import.md:1589:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5570:./reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1241:experiments/cargo_cvpb/litreview2/train_more_import.md:1590:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5647:./reassess/r_2.md:4417:reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1242:experiments/cargo_cvpb/litreview2/train_more_import.md:1591:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5678:./reassess/r_2.md:5420:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1243:experiments/cargo_cvpb/litreview2/train_more_import.md:1592:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5694:./reassess/r_2.md:5830:reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1244:experiments/cargo_cvpb/litreview2/train_more_import.md:1595:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5817:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1245:experiments/cargo_cvpb/litreview2/train_more_import.md:1596:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5939:./reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1253:experiments/cargo_cvpb/litreview2/train_more_import.md:1857:experiments/cargo_cvpb/litreview2/explore20/clean/d_14.txt:1:结论先放前面：**CLIP/DINO/SAM 直接当 backbone、prompt learner、text anchor、SAM 遮挡增强、LVLM 生成语义 token 这条线基本红了**。2024-2026 已经有 CLIP 语义注入 VI-ReID、CLIP selective prompt tuning、LVLM-ReID、LLaVA/MLLM interactive ReID、SAM occlusion augmentation、KPR promptable ReID 等工作在占位。继续“导入 FM 提精度”不建议碰。参考：CSDN/CLIP-VIReID 已做语义桥接与 prompt learner([arxiv.org](https://arxiv.org/abs/2401.05806))，SCING 做 selective cross-modal prompt tuning([arxiv.org](https://arxiv.org/abs/2507.00506))，LVLM-ReID 做 LVLM semantic token([arxiv.org](https://arxiv.org/abs/2411.18111))，LLaVA-ReID/ICL 已做交互式文本 ReID([arxiv.org](https://arxiv.org/abs/2504.10174))，DPEFormer 已用 SAM 做 realistic occlusion augmentation([arxiv.org](https://arxiv.org/abs/2402.10435))。
experiments/paradigm_shift/paradigm_A_generation.md:1257:experiments/cargo_cvpb/litreview2/train_more_import.md:2111:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3563:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5333:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1258:experiments/cargo_cvpb/litreview2/train_more_import.md:2112:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3568:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5338:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1259:experiments/cargo_cvpb/litreview2/train_more_import.md:2141:experiments/cargo_cvpb/litreview2/explore20/d_14.md:68:结论先放前面：**CLIP/DINO/SAM 直接当 backbone、prompt learner、text anchor、SAM 遮挡增强、LVLM 生成语义 token 这条线基本红了**。2024-2026 已经有 CLIP 语义注入 VI-ReID、CLIP selective prompt tuning、LVLM-ReID、LLaVA/MLLM interactive ReID、SAM occlusion augmentation、KPR promptable ReID 等工作在占位。继续“导入 FM 提精度”不建议碰。参考：CSDN/CLIP-VIReID 已做语义桥接与 prompt learner([arxiv.org](https://arxiv.org/abs/2401.05806))，SCING 做 selective cross-modal prompt tuning([arxiv.org](https://arxiv.org/abs/2507.00506))，LVLM-ReID 做 LVLM semantic token([arxiv.org](https://arxiv.org/abs/2411.18111))，LLaVA-ReID/ICL 已做交互式文本 ReID([arxiv.org](https://arxiv.org/abs/2504.10174))，DPEFormer 已用 SAM 做 realistic occlusion augmentation([arxiv.org](https://arxiv.org/abs/2402.10435))。
experiments/paradigm_shift/paradigm_A_generation.md:1260:experiments/cargo_cvpb/litreview2/train_more_import.md:2142:experiments/cargo_cvpb/litreview2/explore20/d_14.md:104:结论先放前面：**CLIP/DINO/SAM 直接当 backbone、prompt learner、text anchor、SAM 遮挡增强、LVLM 生成语义 token 这条线基本红了**。2024-2026 已经有 CLIP 语义注入 VI-ReID、CLIP selective prompt tuning、LVLM-ReID、LLaVA/MLLM interactive ReID、SAM occlusion augmentation、KPR promptable ReID 等工作在占位。继续“导入 FM 提精度”不建议碰。参考：CSDN/CLIP-VIReID 已做语义桥接与 prompt learner([arxiv.org](https://arxiv.org/abs/2401.05806))，SCING 做 selective cross-modal prompt tuning([arxiv.org](https://arxiv.org/abs/2507.00506))，LVLM-ReID 做 LVLM semantic token([arxiv.org](https://arxiv.org/abs/2411.18111))，LLaVA-ReID/ICL 已做交互式文本 ReID([arxiv.org](https://arxiv.org/abs/2504.10174))，DPEFormer 已用 SAM 做 realistic occlusion augmentation([arxiv.org](https://arxiv.org/abs/2402.10435))。
experiments/paradigm_shift/paradigm_A_generation.md:1265:experiments/cargo_cvpb/litreview2/train_more_import.md:2389:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:2076:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1267:experiments/cargo_cvpb/litreview2/train_more_import.md:2391:experiments/cargo_cvpb/litreview2/lmreid_salvage.md:2078:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1285:experiments/cargo_cvpb/litreview2/train_more_import.md:2928:experiments/exp359_lm_reid/codex_review_raw_v2.md:2973:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1286:experiments/cargo_cvpb/litreview2/train_more_import.md:2929:experiments/exp359_lm_reid/codex_review_raw_v2.md:2979:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1820:reassess/r_3.md:4016:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1287:experiments/cargo_cvpb/litreview2/train_more_import.md:2930:experiments/exp359_lm_reid/codex_review_raw_v2.md:2983:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1288:experiments/cargo_cvpb/litreview2/train_more_import.md:2931:experiments/exp359_lm_reid/codex_review_raw_v2.md:2989:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1830:reassess/r_3.md:4027:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1289:experiments/cargo_cvpb/litreview2/train_more_import.md:2934:experiments/exp359_lm_reid/codex_review_raw_v2.md:3332:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4385:reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1290:experiments/cargo_cvpb/litreview2/train_more_import.md:2935:experiments/exp359_lm_reid/codex_review_raw_v2.md:3337:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4390:reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1291:experiments/cargo_cvpb/litreview2/train_more_import.md:2937:experiments/exp359_lm_reid/codex_review_raw_v2.md:3689:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5333:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1292:experiments/cargo_cvpb/litreview2/train_more_import.md:2938:experiments/exp359_lm_reid/codex_review_raw_v2.md:3694:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5338:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1313:experiments/cargo_cvpb/litreview2/train_more_import.md:3477:./experiments/exp359_lm_reid/codex_review_raw_v2.md:1617:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1315:experiments/cargo_cvpb/litreview2/train_more_import.md:3479:./experiments/exp359_lm_reid/codex_review_raw_v2.md:1619:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1326:experiments/cargo_cvpb/litreview2/train_more_import.md:3805:./experiments/exp359_lm_reid/codex_review_raw_v2.md:2973:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1327:experiments/cargo_cvpb/litreview2/train_more_import.md:3806:./experiments/exp359_lm_reid/codex_review_raw_v2.md:2979:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1820:reassess/r_3.md:4016:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1328:experiments/cargo_cvpb/litreview2/train_more_import.md:3807:./experiments/exp359_lm_reid/codex_review_raw_v2.md:2983:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1329:experiments/cargo_cvpb/litreview2/train_more_import.md:3808:./experiments/exp359_lm_reid/codex_review_raw_v2.md:2989:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1830:reassess/r_3.md:4027:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:1330:experiments/cargo_cvpb/litreview2/train_more_import.md:3811:./experiments/exp359_lm_reid/codex_review_raw_v2.md:3332:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4385:reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1331:experiments/cargo_cvpb/litreview2/train_more_import.md:3812:./experiments/exp359_lm_reid/codex_review_raw_v2.md:3337:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4390:reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1332:experiments/cargo_cvpb/litreview2/train_more_import.md:3814:./experiments/exp359_lm_reid/codex_review_raw_v2.md:3689:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5333:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1333:experiments/cargo_cvpb/litreview2/train_more_import.md:3815:./experiments/exp359_lm_reid/codex_review_raw_v2.md:3694:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5338:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1373:experiments/cargo_cvpb/litreview2/train_more_import.md:5307:ReID团队 LM-ReID(低分辨率person ReID 重定义为采样格点sampling-lattice隐变量,非模糊): test-time decision-marginalization已成立(LR-query vs HR-gallery, 同图K=9个lattice变体[sub-pixel phase/±1 LR-pixel bbox/antialias kernel]做边缘化, Market h=12/16/20/24/32全beat普通TTA, 优势随分辨率单调递减h12 +6.5→h32 +0.5; 因子分解=**bbox检测框±1 LR-pixel的crop不确定性主导+2.84** > phase +1.76 > zoom +1.70)。**训练端已试全负**: ①consistency(拉K变体特征到均值)有害single -1.73还掉HR sanity; ②lattice-aug(训练时见lattice变体)≈普通random增广只差+0.54不lattice-specific; ③marginal-likelihood L_marg是训练有害大元凶。**铁律: 训练端做invariance/feature-collapse压判别性=死路**。目标=一个**互补的训练端创新**和test-time marginalization结合成train+test完整方法(更有把握, 6/10→7-8/10)。 ★lens=从相邻领域搬训练范式: 'LR bbox/采样格点是要边缘化的隐变量'这设定, 哪个相邻领域训练机制能映射且避开invariance-collapse? 候选评估: amortized-variational-inference(把marginalization学进网络)/detection-jitter-robust-training/test-time-training训练侧/super-resolution-as-auxiliary/deep-equilibrium/EM-style-latent。求: 最值1-2个 + 具体机制 + 廉价kill-switch + novelty(联网查) + 信心1-10 + 务实中文。死区避开通用FM-import(MLLM/DINO/SD)。
experiments/paradigm_shift/paradigm_A_generation.md:1403:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4263:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5570:./reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1404:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4280:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5647:./reassess/r_2.md:4417:reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1405:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4297:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5678:./reassess/r_2.md:5420:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1406:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4298:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5694:./reassess/r_2.md:5830:reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1407:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4307:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5817:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1408:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4308:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5939:./reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1410:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4427:experiments/cargo_cvpb/litreview2/false_negative_audit.md:3149:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/validate/v_3.md:2942:../litreview2/analyses/ALL_ANALYSES.txt:1172:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
experiments/paradigm_shift/paradigm_A_generation.md:1411:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4687:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10629:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5570:./reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1412:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4688:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10669:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5647:./reassess/r_2.md:4417:reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1413:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4689:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10677:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5678:./reassess/r_2.md:5420:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1414:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4690:experiments/cargo_cvpb/litreview2/false_negative_audit.md:10684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5694:./reassess/r_2.md:5830:reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1415:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4730:experiments/cargo_cvpb/litreview2/false_negative_audit.md:12661:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_hubness.md:16:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'gallery 负向 in-degree 定义成 ReID 失败主变量 + 训练端 anti-hub': (a)hubness in 高维检索(Radovanovic hubness / CSLS / Mutual Proximity / local scaling)用到 ReID 没; (b)k-reciprocal re-ranking / neighbor-based re-ranking ReID; (c)hard-negative mining / anti-hub margin / proxy metric learning; (d)Pose2ID / feature centralization / neighbor aggregation ReID; (e)graph/relation ReID(SGGNN等); (f)long-tail / popularity-bias / false-match-rate retrieval。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
experiments/paradigm_shift/paradigm_A_generation.md:1416:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4799:experiments/cargo_cvpb/litreview2/false_negative_audit.md:14348:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2943:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3112:../../cargo_cvpb/litreview2/novelty_hubness.md:16:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'gallery 负向 in-degree 定义成 ReID 失败主变量 + 训练端 anti-hub': (a)hubness in 高维检索(Radovanovic hubness / CSLS / Mutual Proximity / local scaling)用到 ReID 没; (b)k-reciprocal re-ranking / neighbor-based re-ranking ReID; (c)hard-negative mining / anti-hub margin / proxy metric learning; (d)Pose2ID / feature centralization / neighbor aggregation ReID; (e)graph/relation ReID(SGGNN等); (f)long-tail / popularity-bias / false-match-rate retrieval。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
experiments/paradigm_shift/paradigm_A_generation.md:1417:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4907:experiments/cargo_cvpb/litreview2/false_negative_audit.md:15494:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_3.md:8692:2. **重定义动作**：他们把数据增强从“生成逼真的人图”重定义为“基于行人结构语义单元做组合扩展”。关键词是 semantic reorganization、semantic restructuring-based data augmentation、structural combinatorial enhancement、training enhancement + testing calibration、cross-modal similarity re-ranking。旧问题从“缺真实图片”变成“缺足够多上身、下身、语义组合变化”；从“排序不准”变成“文本 query 需要可靠跨模态邻域来校准”。
experiments/paradigm_shift/paradigm_A_generation.md:1418:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4952:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17699:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5570:./reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1419:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4953:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17708:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5647:./reassess/r_2.md:4417:reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1420:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4954:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17712:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5678:./reassess/r_2.md:5420:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1421:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4955:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17718:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5694:./reassess/r_2.md:5830:reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1422:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4956:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17729:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5817:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1423:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:4957:experiments/cargo_cvpb/litreview2/false_negative_audit.md:17773:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5939:./reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1424:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:5013:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18809:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:10629:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5570:./reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1425:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:5014:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18818:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:10669:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5647:./reassess/r_2.md:4417:reassess/r_2.md:1918:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1426:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:5015:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18822:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:10677:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5678:./reassess/r_2.md:5420:reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1427:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:5016:experiments/cargo_cvpb/litreview2/false_negative_audit.md:18828:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/false_negative_audit.md:10684:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5694:./reassess/r_2.md:5830:reassess/r_3.md:8002:./reassess/r_3.md:4490:/Users/abdslm/Desktop/SOLIDER-REID/experiments/transfer_candidates.json:78:"transfer_to_occluded": "Recast the occluder as a confounder and apply the same backdoor-adjustment recipe: build an \"Occluder Dictionary\" of occlusion patterns/regions stored as Gaussians with priors P(occluder), then train with P(Y|do(X))=Σ_occluder P(Y|X,occluder)·P(occluder), marginalizing over occluders so the model cannot exploit occluder-identity spurious correlations (e.g. a particular bicycle/car/pole co-occurring with certain galleries or cameras). Two concretely transferable tricks: (1) the reverse-attention masking that isolates the confounder feature from the identity feature map — directly usable to separate occluder-region features from body features without external segmentation; (2) representing each confounder as a sampled Gaussian rather than a stored feature, giving cheap synthetic occluder augmentation at the feature level. The NWGM trick keeps the marginalization a single classification, so it's nearly free at training time and zero-cost at inference.",
experiments/paradigm_shift/paradigm_A_generation.md:1443:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8134:experiments/cargo_cvpb/litreview2/reassess/r_3.md:2776:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:35:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1444:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8160:experiments/cargo_cvpb/litreview2/reassess/r_3.md:3071:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:2431:./validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1445:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8217:experiments/cargo_cvpb/litreview2/reassess/r_3.md:3128:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3506:./validate/v_3.md:15353:../litreview2/validate/v_3.md:7891:./validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1446:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8244:experiments/cargo_cvpb/litreview2/reassess/r_3.md:3155:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:3791:./validate/v_3.md:15923:../litreview2/validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1447:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8261:experiments/cargo_cvpb/litreview2/reassess/r_3.md:3206:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4509:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1448:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8448:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4172:./reassess/r_3.md:2776:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:35:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1449:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8457:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4236:./reassess/r_3.md:3206:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4509:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1450:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:8497:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:5482:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1451:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:9324:experiments/cargo_cvpb/litreview2/validate/v_3.md:15353:../litreview2/validate/v_3.md:7891:./validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1452:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:9595:experiments/cargo_cvpb/litreview2/validate/v_3.md:15923:../litreview2/validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1453:experiments/cargo_cvpb/litreview2/codex_lsrc_review.md:9888:experiments/cargo_cvpb/litreview2/validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1468:experiments/cargo_cvpb/litreview2/reviews/deep_9.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1524:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1385:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:374:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1525:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1409:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1526:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1425:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:544:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1527:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1455:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:852:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1528:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1482:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_20.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1529:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1495:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_12.md:546:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1530:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1511:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:677:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1531:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1521:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/B_CONTAINMENT_DESIGN.md:4:> - ① **σ_aerial < σ_ground**(航拍q156.96/g167.47 < 地面q171.64/g172.81 双侧; 合成退化σ反降）——"航拍更欠定=宽分布"前提**根本错了**, 航拍低清=更平滑=低TTA方差。
experiments/paradigm_shift/paradigm_A_generation.md:1532:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1543:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_23.md:604:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1533:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1563:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:777:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1534:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1597:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:287:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1535:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1636:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_1.md:10875:3. **机制怎么长出来**：机制基本是自然推出的。既然 2D 帧被遮挡和错位破坏，就用 SMPL/HMR 生成 3D mesh，转成 point cloud，再用 PointPillars 得到和 2D feature 对齐的 pseudo-image feature；CMF 用 deformable cross-attention 和 2D-3D 映射点做融合；SSTI 用 SMPL shape 参数建 spatial/temporal 两条注意力流，专门处理相似外观混淆。
experiments/paradigm_shift/paradigm_A_generation.md:1536:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1637:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_1.md:10956:3. **机制怎么长出来**：机制基本是自然推出的。既然 2D 帧被遮挡和错位破坏，就用 SMPL/HMR 生成 3D mesh，转成 point cloud，再用 PointPillars 得到和 2D feature 对齐的 pseudo-image feature；CMF 用 deformable cross-attention 和 2D-3D 映射点做融合；SSTI 用 SMPL shape 参数建 spatial/temporal 两条注意力流，专门处理相似外观混淆。
experiments/paradigm_shift/paradigm_A_generation.md:1537:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1646:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:619:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1538:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1690:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:376:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1539:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1724:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1190:外部核查方向基本清楚：2024-2026 的新工作没有停在固定特征后处理，而是在数据/模态/视频/预训练/生成式跨视角上扩展问题。接下来我只再看当前子目录有没有你这轮三负的落盘记录，避免漏掉你们自己的 kill-switch 细节。
experiments/paradigm_shift/paradigm_A_generation.md:1540:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1735:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:608:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1541:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1740:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_15.md:4433:ControlNet [33] is proposed to control DMs with task-specific conditions by an additional trainable copy of pre-trained models. Imagen [34] demonstrates the efficacy of utilizing pre-trained, frozen large
experiments/paradigm_shift/paradigm_A_generation.md:1542:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1743:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_15.md:10498:   211	ControlNet [33] is proposed to control DMs with task-specific conditions by an additional trainable copy of pre-trained models. Imagen [34] demonstrates the efficacy of utilizing pre-trained, frozen large
experiments/paradigm_shift/paradigm_A_generation.md:1543:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1745:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_15.md:13273:   211	ControlNet [33] is proposed to control DMs with task-specific conditions by an additional trainable copy of pre-trained models. Imagen [34] demonstrates the efficacy of utilizing pre-trained, frozen large
experiments/paradigm_shift/paradigm_A_generation.md:1544:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1768:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/clean/x_2.txt:518:**可复用 checklist**: ①先命名一个 confound(比模块新更重要); ②只需 2 模块(结构模块 + 训练正则/增强); ③实验像"系统验证"(SOTA+消融+敏感性+可视化+复杂度+跨数据集), 单卡抓手=多数据集非大模型; ④小正则写成"旧目标忽略 confound X→引入 X-aware regularization"; ⑤生成/增强很好发(离线缓存); ⑥特权信息(LUPI: 训练用pose/SMPL/CLIP/RGB老师, 推理丢掉)很吃香。
experiments/paradigm_shift/paradigm_A_generation.md:1545:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1833:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:954:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1546:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1847:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_26.md:610:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1547:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1911:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:608:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1548:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1933:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_25.md:470:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1549:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1984:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_25.md:14083:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
experiments/paradigm_shift/paradigm_A_generation.md:1550:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1985:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_25.md:14087:3. **机制怎么长出来**：SRDA 很直接：先用 MLLM 判断性别，在同一性别内随机选两个 ID，把一个人的上半身和另一个人的下半身横向拼接，生成新 ID，再用微调 BLIP 为新图生成文本。它不追求自然，而追求结构组合多样性。CSRM 则在测试时为文本 query 找 cross-modal k-reciprocal 图像邻居，用 Jaccard 权重把可靠图像特征融合进文本特征，再检索。训练阶段用 CLIP 双流、ID loss、SDM loss 和 image triplet；不加 text triplet，因为 MLLM 文本有噪声。
experiments/paradigm_shift/paradigm_A_generation.md:1551:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1990:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_25.md:14162:1. **触发观察**：TBPS 数据少，原因是隐私和文本标注成本高。已有增强要么靠大规模外部数据和 MLLM 预训练，要么靠 diffusion 编辑图像，成本高，还追求视觉真实但不一定提高检索。作者还观察到传统 re-ranking 的 k-reciprocal 假设在文本到图像里不成立，因为文本特征没有稳定的同模态局部邻域，text-image 距离分布也不平衡。
experiments/paradigm_shift/paradigm_A_generation.md:1552:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:1991:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_25.md:14166:3. **机制怎么长出来**：SRDA 很直接：先用 MLLM 判断性别，在同一性别内随机选两个 ID，把一个人的上半身和另一个人的下半身横向拼接，生成新 ID，再用微调 BLIP 为新图生成文本。它不追求自然，而追求结构组合多样性。CSRM 则在测试时为文本 query 找 cross-modal k-reciprocal 图像邻居，用 Jaccard 权重把可靠图像特征融合进文本特征，再检索。训练阶段用 CLIP 双流、ID loss、SDM loss 和 image triplet；不加 text triplet，因为 MLLM 文本有噪声。
experiments/paradigm_shift/paradigm_A_generation.md:1553:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2003:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_5.md:921:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1554:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2028:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_11.md:582:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1555:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2112:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1182:../../afd_reid/design.md:3:> 全自动重评估选出的 #1 角度(8 codex + 合成裁决)。死亡清单 0 风险、novelty 空白、kill-switch 最廉价。
experiments/paradigm_shift/paradigm_A_generation.md:1556:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2139:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1231:../../overnight_innovation_log.md:4:> 战略：搬外部范式（CLIP-ReID/Pose2ID 打法），冻结 kill-switch 先验 → 赢家训头 → 破天花板。已死：MLLM-reasoner（姿态提示无效）。已证伪：exp109 内部机制（三堵墙）。
experiments/paradigm_shift/paradigm_A_generation.md:1557:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2291:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1689:../../cargo_cvpb/codex_novelty_airl.md:15:- 主张排序:① view alignment → aerial pixel-budget recoverability;② 非对称降质 + 梯度隔离双 head 生成两类证据;③ **核心发现:两 head 非简单互补,而是按检索方向特化,direction-aware fusion +2.4~3.8**。
experiments/paradigm_shift/paradigm_A_generation.md:1558:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2354:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1785:../../exp322/design.md:76:- **D4 → 源域合成跨域视角**：distractor 注入在源域即可制造多人歧义监督，Market→Occ-ReID 跨域无需 target 遮挡标注 → 第二张表，绕开 Occ-Duke multi-query 比例可能偏低的天花板。
experiments/paradigm_shift/paradigm_A_generation.md:1559:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2359:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1792:../../pose_clip_codex_synthesis.md:18:4. **KCD-CLIP (file17)**: pose生成动态卷积核调制CLIP token + role头(目标/遮挡/别人/bg) + **pose-shuffle负样本**(打乱关键点role必须崩→强制真用pose)。先例: CondConv/DynamicConv。便宜kill-switch。
experiments/paradigm_shift/paradigm_A_generation.md:1560:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2375:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1812:../../cargo_cvpb/airl_dualbranch_result.md:48:- novelty:无 exact prior("detached fork into independent copied late backbone stage + recover branch + fixed cosine prior fusion" for ReID);成分接近 MGN/GreyReID/cross-res ReID/ControlNet-locked-copy/SimSiam-stopgrad → novelty 钉在 CARGO aerial-ground 失败模式 + 证据,不吹全新 primitive。
experiments/paradigm_shift/paradigm_A_generation.md:1561:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2417:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1903:../../library_167_synthesis.md:15:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:1562:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2429:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1956:../../cargo_cvpb/litreview/reviews/lit_4.md:3748:4. **SMPL 约束的反事实视角生成：生成不是为了扩数据，而是为了制造同一 3D 身份在航拍和地面之间的可控视角反事实。**  
experiments/paradigm_shift/paradigm_A_generation.md:1563:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2431:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1958:../../cargo_cvpb/litreview/reviews/lit_4.md:3751:cheap kill-switch：只做小规模，选几十个身份生成有限反事实样本，用现有 ReID 训练短跑对比普通增广。若过滤后保留率低，或收益被普通模糊、裁剪、随机擦除吃掉，就不继续。
experiments/paradigm_shift/paradigm_A_generation.md:1564:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2437:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1968:../../cargo_cvpb/litreview/reviews/lit_4.md:3822:4. **SMPL 约束的反事实视角生成：生成不是为了扩数据，而是为了制造同一 3D 身份在航拍和地面之间的可控视角反事实。**  
experiments/paradigm_shift/paradigm_A_generation.md:1565:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2439:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:1970:../../cargo_cvpb/litreview/reviews/lit_4.md:3825:cheap kill-switch：只做小规模，选几十个身份生成有限反事实样本，用现有 ReID 训练短跑对比普通增广。若过滤后保留率低，或收益被普通模糊、裁剪、随机擦除吃掉，就不继续。
experiments/paradigm_shift/paradigm_A_generation.md:1566:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2487:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2150:../../cargo_cvpb/litreview/reviews/lit_12.md:3406:cheap kill-switch：先不训练，只在一小批 AG-ReID 样本上生成 2D pose 或 SMPL 可见部位，把全局特征或部位特征按可见性重加权做检索。如果航拍 query 的 AP 明显提升，或者错误样本从“姿态/视角相似”中被拉开，再进入训练；如果 train-free 重加权都没信号，杀掉。
experiments/paradigm_shift/paradigm_A_generation.md:1567:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2494:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2162:../../cargo_cvpb/litreview/reviews/lit_12.md:3483:cheap kill-switch：先不训练，只在一小批 AG-ReID 样本上生成 2D pose 或 SMPL 可见部位，把全局特征或部位特征按可见性重加权做检索。如果航拍 query 的 AP 明显提升，或者错误样本从“姿态/视角相似”中被拉开，再进入训练；如果 train-free 重加权都没信号，杀掉。
experiments/paradigm_shift/paradigm_A_generation.md:1568:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2570:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2288:../../cargo_cvpb/litreview2/B_CONTAINMENT_DESIGN.md:4:> - ① **σ_aerial < σ_ground**(航拍q156.96/g167.47 < 地面q171.64/g172.81 双侧; 合成退化σ反降）——"航拍更欠定=宽分布"前提**根本错了**, 航拍低清=更平滑=低TTA方差。
experiments/paradigm_shift/paradigm_A_generation.md:1569:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2674:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2573:../../cargo_cvpb/litreview2/cvpb_containment_killswitch_design.md:9:1. 航拍确实更欠定：trace(σ_A) ≫ trace(σ_G)，σ 随物理量(bbox 面积↓)变化，合成退化单调升 σ。
experiments/paradigm_shift/paradigm_A_generation.md:1570:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2806:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2824:../../cargo_cvpb/litreview2/pivot/p_3.md:17:- ①σ_aerial < σ_ground(航拍q156.96/g167.47 < 地面q171.64/g172.81双侧; 合成退化σ反而降115.41<118.93)。'航拍更欠定=宽分布'前提**错的**——航拍低清=少高频细节=更平滑=低TTA方差。
experiments/paradigm_shift/paradigm_A_generation.md:1571:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2848:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2903:../../cargo_cvpb/litreview2/pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:1572:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2860:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2915:../../cargo_cvpb/litreview2/pivot/p_3.md:2164:2. containment kill-switch 杀得很干净。它不是某个实现失败，而是物理前提失败：航拍 TTA 方差更低、合成退化方差反降、view-mean / 同视角置换不掉分。这个方向不能再包装成“信息欠定包含”。
experiments/paradigm_shift/paradigm_A_generation.md:1573:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2867:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:2922:../../cargo_cvpb/litreview2/pivot/p_3.md:2198:2. containment kill-switch 杀得很干净。它不是某个实现失败，而是物理前提失败：航拍 TTA 方差更低、合成退化方差反降、view-mean / 同视角置换不掉分。这个方向不能再包装成“信息欠定包含”。
experiments/paradigm_shift/paradigm_A_generation.md:1574:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2912:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3043:../../cargo_cvpb/litreview/reviews/lit_1.md:3442:低成本首验：用 SMPL 或 pose 估计得到真实 aerial-ground 的部位可见分布，再把地面图按该分布合成航拍式缺失，做冻结特征检索或很短训练。若合成分布和真实航拍错误模式对不上，杀。
experiments/paradigm_shift/paradigm_A_generation.md:1575:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2920:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3057:../../cargo_cvpb/litreview/reviews/lit_1.md:3516:低成本首验：用 SMPL 或 pose 估计得到真实 aerial-ground 的部位可见分布，再把地面图按该分布合成航拍式缺失，做冻结特征检索或很短训练。若合成分布和真实航拍错误模式对不上，杀。
experiments/paradigm_shift/paradigm_A_generation.md:1576:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:2943:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess2/x_2.md:3112:../../cargo_cvpb/litreview2/novelty_hubness.md:16:用联网**尽力查确切先例+撞车**, 逐类判它们是否已把'gallery 负向 in-degree 定义成 ReID 失败主变量 + 训练端 anti-hub': (a)hubness in 高维检索(Radovanovic hubness / CSLS / Mutual Proximity / local scaling)用到 ReID 没; (b)k-reciprocal re-ranking / neighbor-based re-ranking ReID; (c)hard-negative mining / anti-hub margin / proxy metric learning; (d)Pose2ID / feature centralization / neighbor aggregation ReID; (e)graph/relation ReID(SGGNN等); (f)long-tail / popularity-bias / false-match-rate retrieval。撞了说撞哪+杀伤; 没撞说切口。务实中文, verdict(撞车/存活)+证据链接+信心1-10。
experiments/paradigm_shift/paradigm_A_generation.md:1577:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3388:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:4952:reassess/r_2.md:2908:pivot/p_2.md:5143:pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:1578:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3511:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5176:reassess/r_3.md:3476:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:291:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1579:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3517:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5201:reassess/r_3.md:3596:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1580:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3520:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5209:reassess/r_3.md:3614:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3474:区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
experiments/paradigm_shift/paradigm_A_generation.md:1581:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3525:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5215:reassess/r_3.md:3627:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3530:区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
experiments/paradigm_shift/paradigm_A_generation.md:1582:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3529:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5228:reassess/r_3.md:3660:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:454:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1583:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3531:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5245:reassess/r_3.md:3730:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1584:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3543:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5268:reassess/r_3.md:3788:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:702:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1585:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3545:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5280:reassess/r_3.md:3829:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:534:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1586:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3547:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5286:reassess/r_3.md:3842:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3494:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:1587:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3549:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5288:reassess/r_3.md:3850:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3567:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:1588:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3558:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5312:reassess/r_3.md:3933:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:734:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1589:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3562:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5330:reassess/r_3.md:4002:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:433:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1590:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3563:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5333:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1591:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3568:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5338:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:1592:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3594:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5446:reassess/r_3.md:4686:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:4:> 战略：搬外部范式（CLIP-ReID/Pose2ID 打法），冻结 kill-switch 先验 → 赢家训头 → 破天花板。已死：MLLM-reasoner（姿态提示无效）。已证伪：exp109 内部机制（三堵墙）。
experiments/paradigm_shift/paradigm_A_generation.md:1593:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3633:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5645:reassess/r_3.md:6502:./pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:1594:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3664:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5759:reassess/r_3.md:7775:./reassess/r_3.md:2223:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:1595:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3688:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:5860:reassess/r_3.md:8117:./pivot/p_2.md:5143:pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:1596:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3717:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6082:reassess/r_3.md:11153:reviews/deep_28.md:374:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1597:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3718:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6084:reassess/r_3.md:11162:reviews/deep_9.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1598:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3719:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6085:reassess/r_3.md:11167:reviews/deep_18.md:544:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1599:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3720:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6087:reassess/r_3.md:11207:pivot/p_2.md:5196:reviews/deep_12.md:546:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1600:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3721:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6088:reassess/r_3.md:11211:pivot/p_2.md:5313:reviews/deep_11.md:582:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1601:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3722:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6089:reassess/r_3.md:11214:pivot/p_2.md:5418:reviews/deep_24.md:954:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1602:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3723:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6090:reassess/r_3.md:11217:pivot/p_2.md:5472:reviews/deep_10.md:608:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1603:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3724:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6091:reassess/r_3.md:11221:pivot/p_2.md:5610:reviews/deep_5.md:921:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1604:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3725:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6092:reassess/r_3.md:11257:pivot/p_2.md:6383:reviews/deep_13.md:777:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1605:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3726:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6093:reassess/r_3.md:11260:pivot/p_2.md:6474:reviews/deep_18.md:544:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1606:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3727:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6094:reassess/r_3.md:11266:pivot/p_2.md:6693:reviews/deep_26.md:610:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1607:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3728:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6095:reassess/r_3.md:11269:pivot/p_2.md:6739:reviews/deep_28.md:374:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1608:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3729:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6096:reassess/r_3.md:11272:pivot/p_2.md:6850:reviews/deep_25.md:470:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1609:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3730:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6099:reassess/r_3.md:11281:pivot/p_2.md:7135:reviews/deep_6.md:677:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1610:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3731:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6100:reassess/r_3.md:11284:pivot/p_2.md:7206:reviews/deep_20.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1611:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3732:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6101:reassess/r_3.md:11287:pivot/p_2.md:7328:reviews/deep_23.md:604:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1612:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3733:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6102:reassess/r_3.md:11290:pivot/p_2.md:7429:reviews/deep_2.md:852:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1613:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3734:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6103:reassess/r_3.md:11293:pivot/p_2.md:7498:reviews/deep_17.md:376:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1614:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3735:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6104:reassess/r_3.md:11296:pivot/p_2.md:7569:reviews/deep_9.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1615:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3736:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6105:reassess/r_3.md:11299:pivot/p_2.md:7619:reviews/deep_27.md:619:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1616:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3737:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6106:reassess/r_3.md:11302:pivot/p_2.md:7793:reviews/deep_16.md:287:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1617:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3738:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6107:reassess/r_3.md:11305:pivot/p_2.md:7876:reviews/deep_22.md:608:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1618:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3739:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6110:reassess/r_3.md:11317:validate/v_3.md:753:./reviews/deep_23.md:604:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1619:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3740:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6111:reassess/r_3.md:11320:validate/v_3.md:886:./reviews/deep_20.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1620:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3742:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6113:reassess/r_3.md:11324:validate/v_3.md:1046:./reviews/deep_28.md:374:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1621:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3743:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6114:reassess/r_3.md:11327:validate/v_3.md:1305:./reviews/deep_16.md:287:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1622:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3744:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6115:reassess/r_3.md:11330:validate/v_3.md:1383:./reviews/deep_9.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1623:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3745:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6116:reassess/r_3.md:11333:validate/v_3.md:1862:./reviews/deep_25.md:470:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1624:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3746:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6120:reassess/r_3.md:11346:validate/v_3.md:3143:../litreview/reviews/lit_17.md:442:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1625:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3747:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6121:reassess/r_3.md:11349:validate/v_3.md:3411:../litreview/reviews/lit_7.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1626:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3748:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6122:reassess/r_3.md:11352:validate/v_3.md:3499:../litreview/reviews/lit_18.md:702:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1627:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3749:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6123:reassess/r_3.md:11355:validate/v_3.md:3563:../litreview/reviews/lit_6.md:534:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1628:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3750:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6124:reassess/r_3.md:11358:validate/v_3.md:4033:../litreview/reviews/lit_15.md:862:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1629:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3751:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6125:reassess/r_3.md:11362:validate/v_3.md:4148:../litreview/reviews/lit_10.md:454:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1630:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3752:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6126:reassess/r_3.md:11365:validate/v_3.md:4207:../litreview/reviews/lit_11.md:1693:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1631:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3753:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6127:reassess/r_3.md:11368:validate/v_3.md:4264:../litreview/reviews/lit_8.md:734:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1632:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3754:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6128:reassess/r_3.md:11371:validate/v_3.md:4321:../litreview/reviews/lit_2.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1633:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3755:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6129:reassess/r_3.md:11374:validate/v_3.md:4385:../litreview/reviews/lit_19.md:433:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1634:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3756:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6130:reassess/r_3.md:11377:validate/v_3.md:4426:../litreview/reviews/lit_1.md:573:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1635:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3757:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6131:reassess/r_3.md:11380:validate/v_3.md:4507:../litreview/reviews/lit_13.md:291:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1636:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3758:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6132:reassess/r_3.md:11394:reviews/deep_2.md:852:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1637:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3762:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6136:reassess/r_3.md:11408:reviews/deep_27.md:619:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1638:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3763:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6138:reassess/r_3.md:11461:reviews/deep_16.md:287:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1639:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3764:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6139:reassess/r_3.md:11467:reviews/deep_13.md:777:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1640:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3765:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6140:reassess/r_3.md:11474:reviews/deep_22.md:608:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1641:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3766:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6141:reassess/r_3.md:11480:reviews/deep_6.md:677:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1642:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3768:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6143:reassess/r_3.md:11486:reviews/deep_26.md:610:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1643:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3774:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6153:reassess/r_3.md:11534:reviews/deep_17.md:376:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1644:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3776:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6155:reassess/r_3.md:11543:reviews/deep_11.md:582:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1645:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3777:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6156:reassess/r_3.md:11548:reviews/deep_20.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1646:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3779:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6159:reassess/r_3.md:11560:reviews/deep_25.md:470:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1647:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6162:reassess/r_3.md:11638:reviews/deep_5.md:921:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1648:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3781:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6163:reassess/r_3.md:11646:reviews/deep_23.md:604:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1649:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3782:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6166:reassess/r_3.md:11663:reviews/deep_10.md:608:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1650:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3783:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6167:reassess/r_3.md:11697:reviews/deep_24.md:954:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1651:experiments/cargo_cvpb/litreview2/reassess2/x_1.md:3785:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reassess/r_2.md:6169:reassess/r_3.md:11715:reviews/deep_12.md:546:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1660:experiments/exp187/design.md:4:- 之前 PCVT (exp148) 三视图训练早期 +2.4 mAP 但后期被追平
experiments/paradigm_shift/paradigm_A_generation.md:1672:experiments/cargo_cvpb/litreview2/reviews/deep_18.md:544:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1674:experiments/cargo_cvpb/litreview2/reviews/deep_18.md:2471:   529	distribution by reusing historical data as synthetic support sets, circumventing the need for additional raw samples. During training, the top-𝐿
experiments/paradigm_shift/paradigm_A_generation.md:1686:experiments/cargo_cvpb/litreview2/validate/v_3.md:28:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1705:experiments/cargo_cvpb/litreview2/validate/v_3.md:753:./reviews/deep_23.md:604:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1709:experiments/cargo_cvpb/litreview2/validate/v_3.md:886:./reviews/deep_20.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1730:experiments/cargo_cvpb/litreview2/validate/v_3.md:1046:./reviews/deep_28.md:374:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1739:experiments/cargo_cvpb/litreview2/validate/v_3.md:1164:./validate/v_3.md:28:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1747:experiments/cargo_cvpb/litreview2/validate/v_3.md:1305:./reviews/deep_16.md:287:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1751:experiments/cargo_cvpb/litreview2/validate/v_3.md:1383:./reviews/deep_9.md:938:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1760:experiments/cargo_cvpb/litreview2/validate/v_3.md:1862:./reviews/deep_25.md:470:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1795:experiments/cargo_cvpb/litreview2/validate/v_3.md:2999:../litreview2/analyses/ALL_ANALYSES.txt:1438:2. 重定义动作：作者把“生成更多图”重定义为“给同一身份生成不同姿态、且保持 ReID 身份嵌入一致的训练数据”。关键词是 Identity Diffuser、identity-specific embedding、target pose heatmap、identity consistency、synthetic pretraining。
experiments/paradigm_shift/paradigm_A_generation.md:1802:experiments/cargo_cvpb/litreview2/validate/v_3.md:3029:../litreview2/analyses/ALL_ANALYSES.txt:1565:2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。
experiments/paradigm_shift/paradigm_A_generation.md:1816:experiments/cargo_cvpb/litreview2/validate/v_3.md:3143:../litreview/reviews/lit_17.md:442:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1818:experiments/cargo_cvpb/litreview2/validate/v_3.md:3411:../litreview/reviews/lit_7.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1822:experiments/cargo_cvpb/litreview2/validate/v_3.md:3499:../litreview/reviews/lit_18.md:702:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1824:experiments/cargo_cvpb/litreview2/validate/v_3.md:3563:../litreview/reviews/lit_6.md:534:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1826:experiments/cargo_cvpb/litreview2/validate/v_3.md:4033:../litreview/reviews/lit_15.md:862:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1835:experiments/cargo_cvpb/litreview2/validate/v_3.md:4148:../litreview/reviews/lit_10.md:454:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1837:experiments/cargo_cvpb/litreview2/validate/v_3.md:4207:../litreview/reviews/lit_11.md:1693:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1839:experiments/cargo_cvpb/litreview2/validate/v_3.md:4264:../litreview/reviews/lit_8.md:734:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1845:experiments/cargo_cvpb/litreview2/validate/v_3.md:4321:../litreview/reviews/lit_2.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1850:experiments/cargo_cvpb/litreview2/validate/v_3.md:4385:../litreview/reviews/lit_19.md:433:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1852:experiments/cargo_cvpb/litreview2/validate/v_3.md:4426:../litreview/reviews/lit_1.md:573:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1854:experiments/cargo_cvpb/litreview2/validate/v_3.md:4507:../litreview/reviews/lit_13.md:291:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:1885:experiments/cargo_cvpb/litreview2/validate/v_3.md:5503:./reviews/deep_5.md:1020:scalable pretrain-finetune framework leveraging our largescale synthetic dataset to enhance the model performance
experiments/paradigm_shift/paradigm_A_generation.md:1888:experiments/cargo_cvpb/litreview2/validate/v_3.md:5674:./reviews/deep_16.md:6592:  1143	pretraining. Once the hyperparameters were finalised using a 40,000 synthetic high-resolution frontal images with diverse
experiments/paradigm_shift/paradigm_A_generation.md:1907:experiments/cargo_cvpb/litreview2/validate/v_3.md:6338:./validate/v_3.md:28:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1910:experiments/cargo_cvpb/litreview2/validate/v_3.md:6472:./validate/v_3.md:1164:./validate/v_3.md:28:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1931:experiments/cargo_cvpb/litreview2/validate/v_3.md:7891:./validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1951:experiments/cargo_cvpb/litreview2/validate/v_3.md:8944:./validate/v_3.md:28:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1952:experiments/cargo_cvpb/litreview2/validate/v_3.md:8960:./validate/v_3.md:1164:./validate/v_3.md:28:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1956:experiments/cargo_cvpb/litreview2/validate/v_3.md:9002:./validate/v_3.md:6338:./validate/v_3.md:28:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1957:experiments/cargo_cvpb/litreview2/validate/v_3.md:9008:./validate/v_3.md:6472:./validate/v_3.md:1164:./validate/v_3.md:28:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:1972:experiments/cargo_cvpb/litreview2/validate/v_3.md:9070:./validate/v_3.md:7891:./validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:2061:experiments/cargo_cvpb/litreview/reviews/lit_10.md:454:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2199:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:1616:experiments/exp148/design.md:17:1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
experiments/paradigm_shift/paradigm_A_generation.md:2201:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:1644:experiments/exp197/claude_review_v2.md:128:**审查通过。** v1 中发现的全部 3 个需修复问题（1 Critical + 2 Medium）均已正确修复。固定数量生成策略保证了 triplet loss 的等正样本数假设，梯度流正确，与 SupCon/OA-SD/parallel_aug 等现有功能无冲突。可以启动训练。
experiments/paradigm_shift/paradigm_A_generation.md:2202:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:1653:experiments/exp209/claude_review.md:8:- dataset 2-view 生成 (OA-SD mode, 非 parallel_aug)
experiments/paradigm_shift/paradigm_A_generation.md:2203:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:1665:experiments/exp209/claude_review.md:128:**审查通过**。SupCon 关闭后正确回退到 per-token CE，OA-SD 与 STD-PR per-token 输出完全兼容 (7 元素 list zip 对齐)，dataset 2-view 生成正确，PLBOA 已启用。唯一建议是确保 `WITH_CP True` 在命令行中以避免 16GB OOM。无需新代码。
experiments/paradigm_shift/paradigm_A_generation.md:2211:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:2310:experiments/innovation_brainstorm.md:2698:| 5 | exp194 | OA-SD + CE (w=2.0) | 63.4% | 74.8% |
experiments/paradigm_shift/paradigm_A_generation.md:2213:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3008:experiments/exp194/design.md:1:# 实验 exp194: OA-SD + CE + oa_sd_weight=2.0
experiments/paradigm_shift/paradigm_A_generation.md:2214:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3009:experiments/exp194/design.md:4:- exp191 (OA-SD + CE, weight=1.0) = 63.2/75.4 — 在 CE base (+2.9/+2.6)
experiments/paradigm_shift/paradigm_A_generation.md:2215:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3010:experiments/exp194/design.md:5:- exp192 (OA-SD + CE, decay=0.99) = 62.6/74.9 — decay 不敏感
experiments/paradigm_shift/paradigm_A_generation.md:2216:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3011:experiments/exp194/design.md:7:- 如果 weight=2.0 能进一步提升，说明 OA-SD 的 distillation 信号被低估了
experiments/paradigm_shift/paradigm_A_generation.md:2217:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3012:experiments/exp194/design.md:10:OA-SD distillation loss 当前权重 1.0 可能太弱，增大到 2.0 可以提供更强的 occlusion invariance 学习信号。
experiments/paradigm_shift/paradigm_A_generation.md:2218:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3013:experiments/exp194/design.md:14:- 即: 1-view + CE + PLBOA + OA-SD (EMA decay=0.999, weight=2.0)
experiments/paradigm_shift/paradigm_A_generation.md:2219:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3014:experiments/exp194/design.md:22:- exp191 (OA-SD weight=1.0): 63.2/75.4
experiments/paradigm_shift/paradigm_A_generation.md:2220:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3016:experiments/exp194/monitor.md:1:# exp194 OA-SD + CE + weight=2.0 监控
experiments/paradigm_shift/paradigm_A_generation.md:2221:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3017:experiments/exp194/monitor.md:171:2. **OA-SD 对 weight 超参不敏感**
experiments/paradigm_shift/paradigm_A_generation.md:2222:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3018:experiments/exp194/monitor.md:172:3. **论文价值**: 与 decay 消融一起，证明 OA-SD 超参鲁棒
experiments/paradigm_shift/paradigm_A_generation.md:2223:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3023:experiments/exp194/claude_review.md:4:超参数实验: OA-SD weight 1.0 -> 2.0, 无代码修改
experiments/paradigm_shift/paradigm_A_generation.md:2224:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3024:experiments/exp194/claude_review.md:9:- 动机清晰: 验证 OA-SD distillation loss 权重敏感性
experiments/paradigm_shift/paradigm_A_generation.md:2227:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3387:experiments/cargo_cvpb/litreview2/meta_converge.md:2820:experiments/exp148/design.md:17:1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
experiments/paradigm_shift/paradigm_A_generation.md:2231:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3594:experiments/cargo_cvpb/litreview2/d17_eval.md:8186:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2232:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:3598:experiments/cargo_cvpb/litreview2/d17_eval.md:8287:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8038:./reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2239:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4571:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:3920:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1982:reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2240:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4676:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4251:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4618:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:20:v2 完全重写：不再依赖外部遮挡。使用 `torch.randperm` 随机选择 keypoints 作为 `occ_mask`(line 899-902)，然后 `visible = (~occ_mask) & (kp_scores > 0.3)` (line 923) 正确排除了被遮挡的 keypoints。遮挡是自生成的，不依赖 PLBOA。
experiments/paradigm_shift/paradigm_A_generation.md:2241:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4677:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4252:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4619:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:26:v2 条件 (line 872): `if oerl_enabled and use_pose and feat_maps is not None and kp_data is not None`。不再依赖 OA-SD 模式或 teacher 图像。OERL 在单次 forward 的 feature map 上操作，用 heatmap 合成遮挡。
experiments/paradigm_shift/paradigm_A_generation.md:2242:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4678:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4253:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4627:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:128:v1 审查指出 design.md 描述了 5 种遮挡模式但实现仅依赖 PLBOA。v2 实现使用 **随机 keypoint 遮挡**（`torch.randperm` 选择 ~50% keypoints），这与 design 中的"随机 40-60% keypoints"模式一致。不再需要 PLBOA 来生成遮挡。
experiments/paradigm_shift/paradigm_A_generation.md:2243:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4685:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4262:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2244:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4690:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4284:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4774:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:487:- **真正判据 = 跨域 Occluded-ReID eval**(有真遮挡处), 非 Market。训练用 PLBOA 合成遮挡, VCA 训练时对齐合成遮挡 token, 收益只在遮挡测试集显现。待 e40 Market(确认 VCA ramp 不进一步伤整体) + 训练完跨域 Occ-ReID(决定性)。
experiments/paradigm_shift/paradigm_A_generation.md:2245:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4696:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4298:experiments/cargo_cvpb/litreview2/reassess/r_3.md:5090:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp299_base_lr8_noplboa_s41/monitor.md:32:**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。
experiments/paradigm_shift/paradigm_A_generation.md:2246:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4713:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4348:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8024:./reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2247:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:4714:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4350:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8048:./reassess/r_3.md:5090:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp299_base_lr8_noplboa_s41/monitor.md:32:**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。
experiments/paradigm_shift/paradigm_A_generation.md:2248:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:5088:experiments/exp203/claude_review.md:57:如果同时启用 OA-SD + parallel_aug，processor.py:444 检查 `len(img) == 4`。dataset 需要返回 4 个视图。make_dataloader.py:123-127 显示 `parallel_aug` 和 `_oa_sd_mode` 是独立设置的。需要确认 dataset 是否同时生成 4 个视图。
experiments/paradigm_shift/paradigm_A_generation.md:2254:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6782:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4618:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:20:v2 完全重写：不再依赖外部遮挡。使用 `torch.randperm` 随机选择 keypoints 作为 `occ_mask`(line 899-902)，然后 `visible = (~occ_mask) & (kp_scores > 0.3)` (line 923) 正确排除了被遮挡的 keypoints。遮挡是自生成的，不依赖 PLBOA。
experiments/paradigm_shift/paradigm_A_generation.md:2255:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6783:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4619:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:26:v2 条件 (line 872): `if oerl_enabled and use_pose and feat_maps is not None and kp_data is not None`。不再依赖 OA-SD 模式或 teacher 图像。OERL 在单次 forward 的 feature map 上操作，用 heatmap 合成遮挡。
experiments/paradigm_shift/paradigm_A_generation.md:2256:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6784:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4627:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:128:v1 审查指出 design.md 描述了 5 种遮挡模式但实现仅依赖 PLBOA。v2 实现使用 **随机 keypoint 遮挡**（`torch.randperm` 选择 ~50% keypoints），这与 design 中的"随机 40-60% keypoints"模式一致。不再需要 PLBOA 来生成遮挡。
experiments/paradigm_shift/paradigm_A_generation.md:2257:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6795:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2258:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6803:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4774:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:487:- **真正判据 = 跨域 Occluded-ReID eval**(有真遮挡处), 非 Market。训练用 PLBOA 合成遮挡, VCA 训练时对齐合成遮挡 token, 收益只在遮挡测试集显现。待 e40 Market(确认 VCA ramp 不进一步伤整体) + 训练完跨域 Occ-ReID(决定性)。
experiments/paradigm_shift/paradigm_A_generation.md:2259:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6804:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2260:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6818:experiments/cargo_cvpb/litreview2/reassess/r_3.md:5090:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp299_base_lr8_noplboa_s41/monitor.md:32:**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。
experiments/paradigm_shift/paradigm_A_generation.md:2261:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6846:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8024:./reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2262:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6849:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8038:./reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2263:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6851:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8048:./reassess/r_3.md:5090:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp299_base_lr8_noplboa_s41/monitor.md:32:**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。
experiments/paradigm_shift/paradigm_A_generation.md:2264:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6906:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1982:reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2265:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6916:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5467:reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2266:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:6918:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5841:reassess/r_3.md:8038:./reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2274:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:7688:./experiments/exp209/claude_review.md:128:**审查通过**。SupCon 关闭后正确回退到 per-token CE，OA-SD 与 STD-PR per-token 输出完全兼容 (7 元素 list zip 对齐)，dataset 2-view 生成正确，PLBOA 已启用。唯一建议是确保 `WITH_CP True` 在命令行中以避免 16GB OOM。无需新代码。
experiments/paradigm_shift/paradigm_A_generation.md:2278:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:8565:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:158:experiments/exp194/design.md:14:- 即: 1-view + CE + PLBOA + OA-SD (EMA decay=0.999, weight=2.0)
experiments/paradigm_shift/paradigm_A_generation.md:2279:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:8626:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:561:experiments/exp209/claude_review.md:128:**审查通过**。SupCon 关闭后正确回退到 per-token CE，OA-SD 与 STD-PR per-token 输出完全兼容 (7 元素 list zip 对齐)，dataset 2-view 生成正确，PLBOA 已启用。唯一建议是确保 `WITH_CP True` 在命令行中以避免 16GB OOM。无需新代码。
experiments/paradigm_shift/paradigm_A_generation.md:2285:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9290:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3664:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4251:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4618:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:20:v2 完全重写：不再依赖外部遮挡。使用 `torch.randperm` 随机选择 keypoints 作为 `occ_mask`(line 899-902)，然后 `visible = (~occ_mask) & (kp_scores > 0.3)` (line 923) 正确排除了被遮挡的 keypoints。遮挡是自生成的，不依赖 PLBOA。
experiments/paradigm_shift/paradigm_A_generation.md:2286:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9291:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3666:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4253:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4627:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:128:v1 审查指出 design.md 描述了 5 种遮挡模式但实现仅依赖 PLBOA。v2 实现使用 **随机 keypoint 遮挡**（`torch.randperm` 选择 ~50% keypoints），这与 design 中的"随机 40-60% keypoints"模式一致。不再需要 PLBOA 来生成遮挡。
experiments/paradigm_shift/paradigm_A_generation.md:2287:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9297:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3673:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4262:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2288:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9302:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3678:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4284:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4774:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:487:- **真正判据 = 跨域 Occluded-ReID eval**(有真遮挡处), 非 Market。训练用 PLBOA 合成遮挡, VCA 训练时对齐合成遮挡 token, 收益只在遮挡测试集显现。待 e40 Market(确认 VCA ramp 不进一步伤整体) + 训练完跨域 Occ-ReID(决定性)。
experiments/paradigm_shift/paradigm_A_generation.md:2289:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9306:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3684:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4298:experiments/cargo_cvpb/litreview2/reassess/r_3.md:5090:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp299_base_lr8_noplboa_s41/monitor.md:32:**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。
experiments/paradigm_shift/paradigm_A_generation.md:2290:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9323:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3717:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4348:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8024:./reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2291:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9324:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:3718:experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4350:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8048:./reassess/r_3.md:5090:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp299_base_lr8_noplboa_s41/monitor.md:32:**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。
experiments/paradigm_shift/paradigm_A_generation.md:2292:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9568:./experiments/cargo_cvpb/litreview2/oasd_mechanism.md:6681:experiments/exp194/design.md:14:- 即: 1-view + CE + PLBOA + OA-SD (EMA decay=0.999, weight=2.0)
experiments/paradigm_shift/paradigm_A_generation.md:2294:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:9718:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:1453:experiments/exp209/claude_review.md:128:**审查通过**。SupCon 关闭后正确回退到 per-token CE，OA-SD 与 STD-PR per-token 输出完全兼容 (7 元素 list zip 对齐)，dataset 2-view 生成正确，PLBOA 已启用。唯一建议是确保 `WITH_CP True` 在命令行中以避免 16GB OOM。无需新代码。
experiments/paradigm_shift/paradigm_A_generation.md:2298:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10070:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:2390:experiments/exp194/design.md:14:- 即: 1-view + CE + PLBOA + OA-SD (EMA decay=0.999, weight=2.0)
experiments/paradigm_shift/paradigm_A_generation.md:2300:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10253:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:3920:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1982:reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2301:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10329:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4251:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4618:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:20:v2 完全重写：不再依赖外部遮挡。使用 `torch.randperm` 随机选择 keypoints 作为 `occ_mask`(line 899-902)，然后 `visible = (~occ_mask) & (kp_scores > 0.3)` (line 923) 正确排除了被遮挡的 keypoints。遮挡是自生成的，不依赖 PLBOA。
experiments/paradigm_shift/paradigm_A_generation.md:2302:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10330:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4253:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4627:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp217/claude_review_v2.md:128:v1 审查指出 design.md 描述了 5 种遮挡模式但实现仅依赖 PLBOA。v2 实现使用 **随机 keypoint 遮挡**（`torch.randperm` 选择 ~50% keypoints），这与 design 中的"随机 40-60% keypoints"模式一致。不再需要 PLBOA 来生成遮挡。
experiments/paradigm_shift/paradigm_A_generation.md:2303:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10336:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4262:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2304:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10341:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4284:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4774:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:487:- **真正判据 = 跨域 Occluded-ReID eval**(有真遮挡处), 非 Market。训练用 PLBOA 合成遮挡, VCA 训练时对齐合成遮挡 token, 收益只在遮挡测试集显现。待 e40 Market(确认 VCA ramp 不进一步伤整体) + 训练完跨域 Occ-ReID(决定性)。
experiments/paradigm_shift/paradigm_A_generation.md:2305:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10345:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4298:experiments/cargo_cvpb/litreview2/reassess/r_3.md:5090:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp299_base_lr8_noplboa_s41/monitor.md:32:**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。
experiments/paradigm_shift/paradigm_A_generation.md:2306:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10362:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4348:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8024:./reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2307:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10363:./experiments/cargo_cvpb/litreview2/train3_fourthclass.md:4350:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8048:./reassess/r_3.md:5090:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp299_base_lr8_noplboa_s41/monitor.md:32:**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。
experiments/paradigm_shift/paradigm_A_generation.md:2308:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10504:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:1982:reassess/r_3.md:4670:/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp204/design.md:4:- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
experiments/paradigm_shift/paradigm_A_generation.md:2309:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10514:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:5467:reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2310:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10515:./experiments/cargo_cvpb/litreview2/reassess/r_2.md:5841:reassess/r_3.md:8038:./reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2312:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10605:./experiments/cargo_cvpb/litreview2/d17_eval.md:8186:experiments/cargo_cvpb/litreview2/reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2313:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10609:./experiments/cargo_cvpb/litreview2/d17_eval.md:8287:experiments/cargo_cvpb/litreview2/reassess/r_3.md:8038:./reassess/r_3.md:4780:/Users/abdslm/Desktop/SOLIDER-REID/experiments/overnight_innovation_log.md:511:- **设计**: 3 occluder类(car/bicycle/person, VOC2012 分割patch)× 3 部位(head/torso/legs=上/中/下三分区, region-based 免pose)=9 cells; 训练见 6, hold out 3 对角(car-legs/bicycle-torso/person-head); ERM(mean CE) vs online group-DRO(7组=6cell+clean, present-group 重归一保 CE 尺度=单变量)。eval=Market 整体 query 施 cell 合成遮挡(按(cell,image)确定性seed, ERM/DRO 同遮挡), clean gallery, per-cell mAP。
experiments/paradigm_shift/paradigm_A_generation.md:2314:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:10654:./experiments/exp148/design.md:17:1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
experiments/paradigm_shift/paradigm_A_generation.md:2318:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:11185:./experiments/cargo_cvpb/litreview2/meta_converge.md:2820:experiments/exp148/design.md:17:1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
experiments/paradigm_shift/paradigm_A_generation.md:2320:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:11311:./experiments/exp194/design.md:14:- 即: 1-view + CE + PLBOA + OA-SD (EMA decay=0.999, weight=2.0)
experiments/paradigm_shift/paradigm_A_generation.md:2325:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:11686:./experiments/exp219/claude_review.md:46:- → PLBOA 仍然正常工作（PLBOA 不依赖 OA-SD），但 teacher_pose 不会生成
experiments/paradigm_shift/paradigm_A_generation.md:2331:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12141:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/paradigm_shift/paradigm_A_generation.md:2332:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12146:1. **exp148 PCVT**: 早期加速（ep30 +2.4 mAP），后期被基线追平并反超（ep100 -0.9 mAP）。3-view 训练的 1/3 主损失稀释 + 训练数据缺乏 visibility 多样性。
experiments/paradigm_shift/paradigm_A_generation.md:2337:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12426:### Gap A: 单图内部能否合成“伪多 support”？
experiments/paradigm_shift/paradigm_A_generation.md:2343:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12754:experiments/exp151/design.md:7:PCVT（exp148）从"数据增强"角度解决：创建互补视图训练多样性。
experiments/paradigm_shift/paradigm_A_generation.md:2344:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12755:experiments/exp151/design.md:77:- 直接对照：`exp148 PCVT`（数据增强范式 vs 表示学习范式）
experiments/paradigm_shift/paradigm_A_generation.md:2351:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12974:experiments/decisions.md:1107:**上下文**: exp148 PCVT 和 exp151 PVAT 全部失败。训练集 95.8% 可见率使得所有 visibility-dependent 训练方法无效。
experiments/paradigm_shift/paradigm_A_generation.md:2353:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12976:experiments/decisions.md:1138:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/paradigm_shift/paradigm_A_generation.md:2354:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:12977:experiments/decisions.md:1143:1. **exp148 PCVT**: 早期加速（ep30 +2.4 mAP），后期被基线追平并反超（ep100 -0.9 mAP）。3-view 训练的 1/3 主损失稀释 + 训练数据缺乏 visibility 多样性。
experiments/paradigm_shift/paradigm_A_generation.md:2355:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13016:experiments/decisions.md:3003:  - `exp148 PCVT`
experiments/paradigm_shift/paradigm_A_generation.md:2356:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13017:experiments/decisions.md:3012:- `exp148 PCVT` 已给出连续 `ep10/20/30` 的稳定 `mAP` 正向
experiments/paradigm_shift/paradigm_A_generation.md:2357:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13018:experiments/decisions.md:3041:- 本地主线继续保留 `exp148 PCVT`
experiments/paradigm_shift/paradigm_A_generation.md:2358:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13271:experiments/exp150/design.md:5:`exp148 PCVT` 在 ep30 展示了 +2.4 mAP 的强正向信号。但我们必须回答一个关键机制问题：
experiments/paradigm_shift/paradigm_A_generation.md:2363:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13277:experiments/exp150/design.md:54:- 主对照：`exp148 PCVT`（pose-guided complementary masking）
experiments/paradigm_shift/paradigm_A_generation.md:2370:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13307:experiments/exp150/monitor.md:5:- 类型: exp148 PCVT 的关键机制对照
experiments/paradigm_shift/paradigm_A_generation.md:2371:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13308:experiments/exp150/monitor.md:7:- 直接对照: `exp148 PCVT`（pose-guided complementary masking）
experiments/paradigm_shift/paradigm_A_generation.md:2375:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13317:experiments/exp150/claude_review.md:6:- 设计合理，是 exp148 PCVT 的核心机制对照
experiments/paradigm_shift/paradigm_A_generation.md:2380:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13436:experiments/innovation_brainstorm.md:392:1. **exp148 PCVT 早期加速但后期无效**: complementary masking 在前期提供多样性，但 backbone 最终收敛到的表示已经隐式假设"几乎全可见"
experiments/paradigm_shift/paradigm_A_generation.md:2384:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13591:experiments/exp148/claude_review_request.txt:1:请对 `/root/work/SOLIDER-REID` 中的 `exp148 PCVT` 做一次**广范围、严格、偏保守**的审查。
experiments/paradigm_shift/paradigm_A_generation.md:2385:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13592:experiments/exp148/claude_review_request.txt:17:- `/root/work/SOLIDER-REID/experiments/exp148/design.md`
experiments/paradigm_shift/paradigm_A_generation.md:2386:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13593:experiments/exp148/claude_review_request.txt:18:- `/root/work/SOLIDER-REID/experiments/exp148/monitor.md`
experiments/paradigm_shift/paradigm_A_generation.md:2387:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13594:experiments/exp148/claude_review_request.txt:89:`/root/work/SOLIDER-REID/experiments/exp148/claude_review.md`
experiments/paradigm_shift/paradigm_A_generation.md:2388:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13601:experiments/exp148/design.md:1:# 实验 exp148: PCVT（Pose-Complementary View Training）
experiments/paradigm_shift/paradigm_A_generation.md:2390:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13615:experiments/exp148/monitor.md:1:# exp148 PCVT 监控
experiments/paradigm_shift/paradigm_A_generation.md:2394:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13627:experiments/exp148/claude_review.md:1:# exp148 PCVT 广范围审查报告
experiments/paradigm_shift/paradigm_A_generation.md:2398:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13753:experiments/exp187/design.md:4:- 之前 PCVT (exp148) 三视图训练早期 +2.4 mAP 但后期被追平
experiments/paradigm_shift/paradigm_A_generation.md:2401:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13934:experiments/results.md:890:### exp148: `PCVT` 早中期已形成稳定 `mAP` 正向，成为当前最值得继续追的训练端新方向
experiments/paradigm_shift/paradigm_A_generation.md:2402:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:13935:experiments/results.md:892:> `exp148` 把单图改写成 `full / complementary-view-a / complementary-view-b` 三视图训练对象，用 pose-defined complementary pseudo-views 验证“单图能否被改写成伪多 support 学习对象”。该实验当前仍在运行，以下结论来自 `ep10/20/30` 训练监控。
experiments/paradigm_shift/paradigm_A_generation.md:2435:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:18614:experiments/cargo_cvpb/litreview2/validate/v_2.md:12167:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|avg|MaxSim|ColBERT|Aligned|CARGO|AG-ReID|67.33|52.37|45.19|single-image|PCVT|SCFA|非对称|包含|因果|对齐" experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:2456:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19610:experiments/exp359_lm_reid/codex_review_raw_v2.md:1617:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:2458:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19612:experiments/exp359_lm_reid/codex_review_raw_v2.md:1619:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:2461:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19636:experiments/exp359_lm_reid/codex_review_raw_v2.md:1826:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5759:./reassess/r_3.md:2615:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:778:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:2465:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19729:experiments/exp359_lm_reid/codex_review_raw_v2.md:2216:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4285:./reassess/r_3.md:3602:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:2174:前 5 篇里已经出现几个高频模式：遮挡论文还在围绕“局部对齐加遮挡权重”，红外论文围绕“跨模态生成或伪标签净化”，泛化论文把“困难样本”重新定义成度量关系被扰动的样本。后面 4 篇我继续按同样口径抽取，最后只保留对我们能发 B 类有用的反推结论。
experiments/paradigm_shift/paradigm_A_generation.md:2470:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19960:experiments/exp359_lm_reid/codex_review_raw_v2.md:2973:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2471:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19961:experiments/exp359_lm_reid/codex_review_raw_v2.md:2979:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1820:reassess/r_3.md:4016:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:2472:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19963:experiments/exp359_lm_reid/codex_review_raw_v2.md:2983:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2473:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:19964:experiments/exp359_lm_reid/codex_review_raw_v2.md:2989:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1830:reassess/r_3.md:4027:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:2474:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20008:experiments/exp359_lm_reid/codex_review_raw_v2.md:3332:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4385:reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2475:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20010:experiments/exp359_lm_reid/codex_review_raw_v2.md:3337:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4390:reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2476:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20035:experiments/exp359_lm_reid/codex_review_raw_v2.md:3541:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5176:reassess/r_3.md:3476:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:291:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2477:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20039:experiments/exp359_lm_reid/codex_review_raw_v2.md:3565:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5201:reassess/r_3.md:3596:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2478:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20044:experiments/exp359_lm_reid/codex_review_raw_v2.md:3589:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5228:reassess/r_3.md:3660:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:454:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2479:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20047:experiments/exp359_lm_reid/codex_review_raw_v2.md:3604:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5245:reassess/r_3.md:3730:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2480:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20049:experiments/exp359_lm_reid/codex_review_raw_v2.md:3627:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5268:reassess/r_3.md:3788:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:702:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2481:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20051:experiments/exp359_lm_reid/codex_review_raw_v2.md:3639:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5280:reassess/r_3.md:3829:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:534:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2482:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20053:experiments/exp359_lm_reid/codex_review_raw_v2.md:3670:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5312:reassess/r_3.md:3933:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:734:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2483:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20055:experiments/exp359_lm_reid/codex_review_raw_v2.md:3686:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5330:reassess/r_3.md:4002:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:433:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2484:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20056:experiments/exp359_lm_reid/codex_review_raw_v2.md:3689:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5333:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2485:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:20058:experiments/exp359_lm_reid/codex_review_raw_v2.md:3694:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5338:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2511:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21190:experiments/exp359_lm_reid/codex_review_raw_v2.md:1617:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:2513:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21192:experiments/exp359_lm_reid/codex_review_raw_v2.md:1619:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:2516:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21216:experiments/exp359_lm_reid/codex_review_raw_v2.md:1826:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5759:./reassess/r_3.md:2615:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:778:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:2520:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21309:experiments/exp359_lm_reid/codex_review_raw_v2.md:2216:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4285:./reassess/r_3.md:3602:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:2174:前 5 篇里已经出现几个高频模式：遮挡论文还在围绕“局部对齐加遮挡权重”，红外论文围绕“跨模态生成或伪标签净化”，泛化论文把“困难样本”重新定义成度量关系被扰动的样本。后面 4 篇我继续按同样口径抽取，最后只保留对我们能发 B 类有用的反推结论。
experiments/paradigm_shift/paradigm_A_generation.md:2525:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21540:experiments/exp359_lm_reid/codex_review_raw_v2.md:2973:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2526:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21541:experiments/exp359_lm_reid/codex_review_raw_v2.md:2979:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1820:reassess/r_3.md:4016:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:2527:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21543:experiments/exp359_lm_reid/codex_review_raw_v2.md:2983:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2528:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21544:experiments/exp359_lm_reid/codex_review_raw_v2.md:2989:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1830:reassess/r_3.md:4027:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:2529:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21588:experiments/exp359_lm_reid/codex_review_raw_v2.md:3332:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4385:reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2530:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21590:experiments/exp359_lm_reid/codex_review_raw_v2.md:3337:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4390:reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2531:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21615:experiments/exp359_lm_reid/codex_review_raw_v2.md:3541:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5176:reassess/r_3.md:3476:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:291:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2532:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21619:experiments/exp359_lm_reid/codex_review_raw_v2.md:3565:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5201:reassess/r_3.md:3596:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2533:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21624:experiments/exp359_lm_reid/codex_review_raw_v2.md:3589:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5228:reassess/r_3.md:3660:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:454:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2534:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21627:experiments/exp359_lm_reid/codex_review_raw_v2.md:3604:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5245:reassess/r_3.md:3730:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2535:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21629:experiments/exp359_lm_reid/codex_review_raw_v2.md:3627:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5268:reassess/r_3.md:3788:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:702:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2536:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21631:experiments/exp359_lm_reid/codex_review_raw_v2.md:3639:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5280:reassess/r_3.md:3829:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:534:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2537:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21633:experiments/exp359_lm_reid/codex_review_raw_v2.md:3670:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5312:reassess/r_3.md:3933:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:734:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2538:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21635:experiments/exp359_lm_reid/codex_review_raw_v2.md:3686:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5330:reassess/r_3.md:4002:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:433:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2539:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21636:experiments/exp359_lm_reid/codex_review_raw_v2.md:3689:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5333:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2540:experiments/cargo_cvpb/litreview2/plboa_mechanism.md:21638:experiments/exp359_lm_reid/codex_review_raw_v2.md:3694:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5338:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:2570:experiments/cargo_cvpb/litreview2/reviews/deep_11.md:582:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2602:experiments/cargo_cvpb/litreview2/reviews/deep_25.md:470:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2654:experiments/cargo_cvpb/litreview/reviews/lit_7.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2701:experiments/results.md:890:### exp148: `PCVT` 早中期已形成稳定 `mAP` 正向，成为当前最值得继续追的训练端新方向
experiments/paradigm_shift/paradigm_A_generation.md:2702:experiments/results.md:892:> `exp148` 把单图改写成 `full / complementary-view-a / complementary-view-b` 三视图训练对象，用 pose-defined complementary pseudo-views 验证“单图能否被改写成伪多 support 学习对象”。该实验当前仍在运行，以下结论来自 `ep10/20/30` 训练监控。
experiments/paradigm_shift/paradigm_A_generation.md:2755:experiments/cargo_cvpb/litreview/reviews/lit_11.md:664:../../paper_materials/arxiv_reid_papers/pdfs/2025_icme_2410.13567v3_ccup_a_controllable_synthetic_data_generation_pipeline_for_pretraining_cloth_changing_person_re_identification.pdf
experiments/paradigm_shift/paradigm_A_generation.md:2757:experiments/cargo_cvpb/litreview/reviews/lit_11.md:891:../arxiv_reid_papers/pdfs/2025_icme_2410.13567v3_ccup_a_controllable_synthetic_data_generation_pipeline_for_pretraining_cloth_changing_person_re_identification.pdf
experiments/paradigm_shift/paradigm_A_generation.md:2771:experiments/cargo_cvpb/litreview/reviews/lit_11.md:1693:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2806:experiments/cargo_cvpb/litreview/reviews/lit_3.md:671:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2812:experiments/cargo_cvpb/litreview/reviews/lit_3.md:3187:scalable pretrain-finetune framework leveraging our largescale synthetic dataset to enhance the model performance
experiments/paradigm_shift/paradigm_A_generation.md:2837:experiments/cargo_cvpb/litreview/reviews/lit_15.md:862:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:2855:experiments/cargo_cvpb/litreview2/analyses/deep_3.txt:22:2. 重定义动作：作者把“生成更多图”重定义为“给同一身份生成不同姿态、且保持 ReID 身份嵌入一致的训练数据”。关键词是 Identity Diffuser、identity-specific embedding、target pose heatmap、identity consistency、synthetic pretraining。
experiments/paradigm_shift/paradigm_A_generation.md:2871:experiments/exp206/claude_review.md:7:- dataset 2-view 生成（OA-SD mode）
experiments/paradigm_shift/paradigm_A_generation.md:2909:experiments/cargo_cvpb/litreview2/analyses/deep_5.txt:17:2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。
experiments/paradigm_shift/paradigm_A_generation.md:3063:experiments/cargo_cvpb/litreview2/analyses/ALL_ANALYSES.txt:1438:2. 重定义动作：作者把“生成更多图”重定义为“给同一身份生成不同姿态、且保持 ReID 身份嵌入一致的训练数据”。关键词是 Identity Diffuser、identity-specific embedding、target pose heatmap、identity consistency、synthetic pretraining。
experiments/paradigm_shift/paradigm_A_generation.md:3075:experiments/cargo_cvpb/litreview2/analyses/ALL_ANALYSES.txt:1565:2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。
experiments/paradigm_shift/paradigm_A_generation.md:3107:experiments/exp203/claude_review.md:57:如果同时启用 OA-SD + parallel_aug，processor.py:444 检查 `len(img) == 4`。dataset 需要返回 4 个视图。make_dataloader.py:123-127 显示 `parallel_aug` 和 `_oa_sd_mode` 是独立设置的。需要确认 dataset 是否同时生成 4 个视图。
experiments/paradigm_shift/paradigm_A_generation.md:3116:experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_7.md:8224:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/design_agreid_v2.md:13:CARGO(合成跨视角)小,涨幅可能缩;若不超,如实报。
experiments/paradigm_shift/paradigm_A_generation.md:3118:experiments/cargo_cvpb/airl_codex_bundle/reviews/codex_7.md:9133:/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/smoke_agreid_v2_wiring.py:125:print(f'  eval_market synthetic: mAP={mAP*100:.1f} R1={cmc[0]*100:.1f} '
experiments/paradigm_shift/paradigm_A_generation.md:3119:experiments/paper_notes/paper_20_PADE.md:7:1. **Parallel Augmentation**: 同一图片生成3个增强版本同时训练
experiments/paradigm_shift/paradigm_A_generation.md:3120:experiments/paper_notes/paper_21_20260322_direction_reset.md:70:### Gap A: 单图内部能否合成“伪多 support”？
experiments/paradigm_shift/paradigm_A_generation.md:3121:experiments/paper_notes/paper_21_20260322_direction_reset.md:109:**一句话**: 用姿态热图生成两张“互补可见”的伪视图，不再做随机遮挡，而是训练模型从互补 body support 中学习稳定身份表示。
experiments/paradigm_shift/paradigm_A_generation.md:3129:experiments/paper_notes/2026-03-19_support_complete_direction.md:42:### 3. NFR / FRT / Pose2ID / SGCFR：recover / centralize 往往是对的，但多数停留在 test-time
experiments/paradigm_shift/paradigm_A_generation.md:3130:experiments/paper_notes/2026-03-19_support_complete_direction.md:44:- NFR、FRT、Pose2ID、以及我们自己的 SGCFR，都说明：
experiments/paradigm_shift/paradigm_A_generation.md:3131:experiments/paper_notes/2026-03-19_support_complete_direction.md:46:  - 利用邻居、生成视角、或 gallery support 做恢复/中心化，经常会带来明显增益
experiments/paradigm_shift/paradigm_A_generation.md:3132:experiments/paper_notes/2026-03-22_claude_takeover_prompt.md:36:1. `exp148 PCVT` 是当前最值得继续追的主线
experiments/paradigm_shift/paradigm_A_generation.md:3147:experiments/overnight_innovation_log.md:4:> 战略：搬外部范式（CLIP-ReID/Pose2ID 打法），冻结 kill-switch 先验 → 赢家训头 → 破天花板。已死：MLLM-reasoner（姿态提示无效）。已证伪：exp109 内部机制（三堵墙）。
experiments/paradigm_shift/paradigm_A_generation.md:3151:experiments/overnight_innovation_log.md:111:**若 LoRA 也低**：DINO paradigm-import 判为负，剩余夜转生成式遮挡增强(攻 95.8% 墙)或 test-time adaptation，或收敛到 analysis 报告。
experiments/paradigm_shift/paradigm_A_generation.md:3152:experiments/overnight_innovation_log.md:202:- **LoRA-for-ReID 已有**：diffusion-ReID (2502.06619) 用 LoRA adapt Q/K/V/O+FFN。而且"frozen DINOv2 缺判别性、需 LoRA 中间地带"这个发现**文献已近乎明说** → 我们的 8.65→37 **印证已知现象**，不是新发现。
experiments/paradigm_shift/paradigm_A_generation.md:3153:experiments/overnight_innovation_log.md:290:- **Pose2ID 式生成补全**：撞 exp109 墙（completion = identity-conditioned 不可实现）→ 不做。
experiments/paradigm_shift/paradigm_A_generation.md:3160:experiments/exp219/claude_review.md:46:- → PLBOA 仍然正常工作（PLBOA 不依赖 OA-SD），但 teacher_pose 不会生成
experiments/paradigm_shift/paradigm_A_generation.md:3168:experiments/paper_materials/story.md:1406:1. `exp148 PCVT`
experiments/paradigm_shift/paradigm_A_generation.md:3188:experiments/exp359_lm_reid/codex_review_raw_v2.md:1617:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:3190:experiments/exp359_lm_reid/codex_review_raw_v2.md:1619:experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:3193:experiments/exp359_lm_reid/codex_review_raw_v2.md:1814:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5746:./reassess/r_3.md:1993:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_19.md:666:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3194:experiments/exp359_lm_reid/codex_review_raw_v2.md:1815:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5747:./reassess/r_3.md:2017:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:300:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3195:experiments/exp359_lm_reid/codex_review_raw_v2.md:1816:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5748:./reassess/r_3.md:2067:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:864:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3196:experiments/exp359_lm_reid/codex_review_raw_v2.md:1817:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5749:./reassess/r_3.md:2153:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:545:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3197:experiments/exp359_lm_reid/codex_review_raw_v2.md:1818:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5750:./reassess/r_3.md:2260:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:213:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3198:experiments/exp359_lm_reid/codex_review_raw_v2.md:1820:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5752:./reassess/r_3.md:2400:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:470:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3199:experiments/exp359_lm_reid/codex_review_raw_v2.md:1821:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5753:./reassess/r_3.md:2435:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:703:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3200:experiments/exp359_lm_reid/codex_review_raw_v2.md:1822:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5755:./reassess/r_3.md:2483:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:880:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3201:experiments/exp359_lm_reid/codex_review_raw_v2.md:1824:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5757:./reassess/r_3.md:2533:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:534:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3202:experiments/exp359_lm_reid/codex_review_raw_v2.md:1825:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5758:./reassess/r_3.md:2578:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:603:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3203:experiments/exp359_lm_reid/codex_review_raw_v2.md:1826:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5759:./reassess/r_3.md:2615:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:778:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3204:experiments/exp359_lm_reid/codex_review_raw_v2.md:1829:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5762:./reassess/r_3.md:2662:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:534:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3205:experiments/exp359_lm_reid/codex_review_raw_v2.md:1830:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5763:./reassess/r_3.md:2701:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:597:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3206:experiments/exp359_lm_reid/codex_review_raw_v2.md:1831:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5764:./reassess/r_3.md:2732:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:302:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3207:experiments/exp359_lm_reid/codex_review_raw_v2.md:1854:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5787:./reassess/r_3.md:3370:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:536:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3208:experiments/exp359_lm_reid/codex_review_raw_v2.md:1857:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5790:./reassess/r_3.md:3464:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:217:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3209:experiments/exp359_lm_reid/codex_review_raw_v2.md:1860:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5795:./reassess/r_3.md:3584:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:363:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3210:experiments/exp359_lm_reid/codex_review_raw_v2.md:1861:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5796:./reassess/r_3.md:3648:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:380:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3211:experiments/exp359_lm_reid/codex_review_raw_v2.md:1862:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5798:./reassess/r_3.md:3718:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:363:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3212:experiments/exp359_lm_reid/codex_review_raw_v2.md:1863:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5799:./reassess/r_3.md:3776:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:628:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3213:experiments/exp359_lm_reid/codex_review_raw_v2.md:1866:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5802:./reassess/r_3.md:3817:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:460:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3214:experiments/exp359_lm_reid/codex_review_raw_v2.md:1868:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5804:./reassess/r_3.md:3921:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:660:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3215:experiments/exp359_lm_reid/codex_review_raw_v2.md:1871:experiments/cargo_cvpb/litreview2/novelty_rankinstab.md:5807:./reassess/r_3.md:3990:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:359:**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
experiments/paradigm_shift/paradigm_A_generation.md:3220:experiments/exp359_lm_reid/codex_review_raw_v2.md:1977:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4042:./reassess/r_3.md:1943:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_8.md:12133:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
experiments/paradigm_shift/paradigm_A_generation.md:3221:experiments/exp359_lm_reid/codex_review_raw_v2.md:1978:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4043:./reassess/r_3.md:1944:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_8.md:12180:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
experiments/paradigm_shift/paradigm_A_generation.md:3222:experiments/exp359_lm_reid/codex_review_raw_v2.md:1987:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4052:./reassess/r_3.md:1992:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_19.md:664:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3223:experiments/exp359_lm_reid/codex_review_raw_v2.md:1991:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4056:./reassess/r_3.md:2016:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_28.md:298:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3224:experiments/exp359_lm_reid/codex_review_raw_v2.md:2002:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4067:./reassess/r_3.md:2066:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_9.md:862:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3225:experiments/exp359_lm_reid/codex_review_raw_v2.md:2008:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4073:./reassess/r_3.md:2152:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_27.md:543:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3226:experiments/exp359_lm_reid/codex_review_raw_v2.md:2018:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4084:./reassess/r_3.md:2223:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:3227:experiments/exp359_lm_reid/codex_review_raw_v2.md:2025:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4091:./reassess/r_3.md:2259:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_16.md:211:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3228:experiments/exp359_lm_reid/codex_review_raw_v2.md:2040:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4108:./reassess/r_3.md:2399:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_18.md:468:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3229:experiments/exp359_lm_reid/codex_review_raw_v2.md:2047:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4115:./reassess/r_3.md:2434:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_13.md:701:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3230:experiments/exp359_lm_reid/codex_review_raw_v2.md:2055:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4123:./reassess/r_3.md:2482:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_24.md:878:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3231:experiments/exp359_lm_reid/codex_review_raw_v2.md:2061:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4129:./reassess/r_3.md:2532:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_22.md:532:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3232:experiments/exp359_lm_reid/codex_review_raw_v2.md:2069:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4137:./reassess/r_3.md:2577:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_6.md:601:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3233:experiments/exp359_lm_reid/codex_review_raw_v2.md:2079:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4147:./reassess/r_3.md:2614:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_2.md:776:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3234:experiments/exp359_lm_reid/codex_review_raw_v2.md:2085:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4153:./reassess/r_3.md:2661:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_10.md:532:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3235:experiments/exp359_lm_reid/codex_review_raw_v2.md:2091:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4159:./reassess/r_3.md:2700:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_3.md:595:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3236:experiments/exp359_lm_reid/codex_review_raw_v2.md:2097:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4165:./reassess/r_3.md:2731:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/reviews/deep_17.md:300:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3237:experiments/exp359_lm_reid/codex_review_raw_v2.md:2104:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4172:./reassess/r_3.md:2776:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:35:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:3238:experiments/exp359_lm_reid/codex_review_raw_v2.md:2168:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4236:./reassess/r_3.md:3206:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:4509:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:3239:experiments/exp359_lm_reid/codex_review_raw_v2.md:2179:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4247:./reassess/r_3.md:3252:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:5043:reviews/deep_8.md:12133:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
experiments/paradigm_shift/paradigm_A_generation.md:3240:experiments/exp359_lm_reid/codex_review_raw_v2.md:2180:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4248:./reassess/r_3.md:3253:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_2.md:5045:reviews/deep_8.md:12180:1. 触发观察：CCReID 里视觉外观被衣服强烈污染，传统方法容易困在视觉瓶颈；但直接用 CLIP 也有问题，因为 CLIP 生成或学习的文本往往偏向衣服颜色，而衣服颜色在换衣场景不稳定。
experiments/paradigm_shift/paradigm_A_generation.md:3241:experiments/exp359_lm_reid/codex_review_raw_v2.md:2191:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4259:./reassess/r_3.md:3369:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_5.md:534:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3242:experiments/exp359_lm_reid/codex_review_raw_v2.md:2206:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4274:./reassess/r_3.md:3463:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:215:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3243:experiments/exp359_lm_reid/codex_review_raw_v2.md:2213:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4282:./reassess/r_3.md:3583:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:361:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3244:experiments/exp359_lm_reid/codex_review_raw_v2.md:2216:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4285:./reassess/r_3.md:3602:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:2174:前 5 篇里已经出现几个高频模式：遮挡论文还在围绕“局部对齐加遮挡权重”，红外论文围绕“跨模态生成或伪标签净化”，泛化论文把“困难样本”重新定义成度量关系被扰动的样本。后面 4 篇我继续按同样口径抽取，最后只保留对我们能发 B 类有用的反推结论。
experiments/paradigm_shift/paradigm_A_generation.md:3245:experiments/exp359_lm_reid/codex_review_raw_v2.md:2227:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4296:./reassess/r_3.md:3647:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:378:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3246:experiments/exp359_lm_reid/codex_review_raw_v2.md:2232:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4301:./reassess/r_3.md:3717:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:361:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3247:experiments/exp359_lm_reid/codex_review_raw_v2.md:2246:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4315:./reassess/r_3.md:3775:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:626:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3248:experiments/exp359_lm_reid/codex_review_raw_v2.md:2254:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4323:./reassess/r_3.md:3816:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:458:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3249:experiments/exp359_lm_reid/codex_review_raw_v2.md:2259:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4328:./reassess/r_3.md:3842:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3494:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:3250:experiments/exp359_lm_reid/codex_review_raw_v2.md:2262:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4331:./reassess/r_3.md:3850:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3567:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:3251:experiments/exp359_lm_reid/codex_review_raw_v2.md:2276:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4345:./reassess/r_3.md:3920:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:658:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3252:experiments/exp359_lm_reid/codex_review_raw_v2.md:2279:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4348:./reassess/r_3.md:3949:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3648:和最像工作的区别：MIP 的 prompt 是模态和实例向量，DTST 是 learned token 选择，π-VL 是 parsing 文本监督。我们的切开点是显式建模“这张图哪些人体证据可靠”，由 pose 置信度、遮挡、视角、mesh 可见性生成 reliability prompt 或 gating，指导 Swin 只强化可信身体证据。  
experiments/paradigm_shift/paradigm_A_generation.md:3253:experiments/exp359_lm_reid/codex_review_raw_v2.md:2280:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4349:./reassess/r_3.md:3961:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:3702:和最像工作的区别：MIP 的 prompt 是模态和实例向量，DTST 是 learned token 选择，π-VL 是 parsing 文本监督。我们的切开点是显式建模“这张图哪些人体证据可靠”，由 pose 置信度、遮挡、视角、mesh 可见性生成 reliability prompt 或 gating，指导 Swin 只强化可信身体证据。  
experiments/paradigm_shift/paradigm_A_generation.md:3254:experiments/exp359_lm_reid/codex_review_raw_v2.md:2283:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4352:./reassess/r_3.md:3989:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:357:**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。
experiments/paradigm_shift/paradigm_A_generation.md:3255:experiments/exp359_lm_reid/codex_review_raw_v2.md:2318:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:4606:./reassess/r_3.md:7775:./reassess/r_3.md:2223:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:3260:experiments/exp359_lm_reid/codex_review_raw_v2.md:2514:experiments/cargo_cvpb/litreview2/ondisk_pivot.md:5482:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:3261:experiments/exp359_lm_reid/codex_review_raw_v2.md:2800:experiments/cargo_cvpb/codex_review_raw.txt:3691:experiments/cargo_cvpb/claude_review.md:100:### L8.（已确认p/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3474:区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
experiments/paradigm_shift/paradigm_A_generation.md:3262:experiments/exp359_lm_reid/codex_review_raw_v2.md:2810:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1650:reassess/r_3.md:3627:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3530:区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
experiments/paradigm_shift/paradigm_A_generation.md:3263:experiments/exp359_lm_reid/codex_review_raw_v2.md:2826:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1666:reassess/r_3.md:3678:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:3908:headline：合成数据不追求数量，而是生成同一 SMPL 身份在航拍和地面视角下的成对反事实样本。  
experiments/paradigm_shift/paradigm_A_generation.md:3264:experiments/exp359_lm_reid/codex_review_raw_v2.md:2836:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1676:reassess/r_3.md:3692:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:3986:headline：合成数据不追求数量，而是生成同一 SMPL 身份在航拍和地面视角下的成对反事实样本。  
experiments/paradigm_shift/paradigm_A_generation.md:3265:experiments/exp359_lm_reid/codex_review_raw_v2.md:2880:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1721:reassess/r_3.md:3797:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2867:我最建议先验一号。它的问题定义最硬，也最能把团队的 SMPL、pose gating 和 aerial-ground 资产合成一个别人不容易复刻的机制。二号可以作为一号的低成本前置实验，三号适合在有无监督设定价值时另开。
experiments/paradigm_shift/paradigm_A_generation.md:3266:experiments/exp359_lm_reid/codex_review_raw_v2.md:2885:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1726:reassess/r_3.md:3803:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:2940:我最建议先验一号。它的问题定义最硬，也最能把团队的 SMPL、pose gating 和 aerial-ground 资产合成一个别人不容易复刻的机制。二号可以作为一号的低成本前置实验，三号适合在有无监督设定价值时另开。
experiments/paradigm_shift/paradigm_A_generation.md:3267:experiments/exp359_lm_reid/codex_review_raw_v2.md:2899:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1740:reassess/r_3.md:3842:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3494:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:3268:experiments/exp359_lm_reid/codex_review_raw_v2.md:2906:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1747:reassess/r_3.md:3850:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3567:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:3269:experiments/exp359_lm_reid/codex_review_raw_v2.md:2920:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1761:reassess/r_3.md:3890:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:3373:和本批最像工作的区别：最像 `FMCNet+`。它补偿可见光和红外的模态特异特征，我们补偿航拍和地面视角下缺失的身体表面和局部身份残差。关键切开点是用 SMPL 可见性和姿态热图约束生成什么，不做无约束特征幻觉。  
experiments/paradigm_shift/paradigm_A_generation.md:3270:experiments/exp359_lm_reid/codex_review_raw_v2.md:2930:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1771:reassess/r_3.md:3902:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_9.md:3447:和本批最像工作的区别：最像 `FMCNet+`。它补偿可见光和红外的模态特异特征，我们补偿航拍和地面视角下缺失的身体表面和局部身份残差。关键切开点是用 SMPL 可见性和姿态热图约束生成什么，不做无约束特征幻觉。  
experiments/paradigm_shift/paradigm_A_generation.md:3271:experiments/exp359_lm_reid/codex_review_raw_v2.md:2973:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:3272:experiments/exp359_lm_reid/codex_review_raw_v2.md:2977:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1818:reassess/r_3.md:4014:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2694:一句话 headline：用 SMPL 生成训练期身份几何 token，把 LVLM 式“语义 token”改成不依赖语言、不增加测试成本的 3D 人体 token。
experiments/paradigm_shift/paradigm_A_generation.md:3273:experiments/exp359_lm_reid/codex_review_raw_v2.md:2979:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1820:reassess/r_3.md:4016:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2698:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:3274:experiments/exp359_lm_reid/codex_review_raw_v2.md:2983:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:3275:experiments/exp359_lm_reid/codex_review_raw_v2.md:2987:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1828:reassess/r_3.md:4025:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2770:一句话 headline：用 SMPL 生成训练期身份几何 token，把 LVLM 式“语义 token”改成不依赖语言、不增加测试成本的 3D 人体 token。
experiments/paradigm_shift/paradigm_A_generation.md:3276:experiments/exp359_lm_reid/codex_review_raw_v2.md:2989:experiments/cargo_cvpb/litreview2/reassess/r_2.md:1830:reassess/r_3.md:4027:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2774:和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。
experiments/paradigm_shift/paradigm_A_generation.md:3277:experiments/exp359_lm_reid/codex_review_raw_v2.md:3043:experiments/cargo_cvpb/litreview2/reassess/r_2.md:2054:pivot/p_2.md:35:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:3278:experiments/exp359_lm_reid/codex_review_raw_v2.md:3076:experiments/cargo_cvpb/litreview2/reassess/r_2.md:2794:pivot/p_2.md:4509:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:3279:experiments/exp359_lm_reid/codex_review_raw_v2.md:3079:experiments/cargo_cvpb/litreview2/reassess/r_2.md:2935:pivot/p_2.md:5631:pivot/p_2.md:35:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:3280:experiments/exp359_lm_reid/codex_review_raw_v2.md:3080:experiments/cargo_cvpb/litreview2/reassess/r_2.md:3188:pivot/p_2.md:6290:pivot/p_2.md:4509:/bin/zsh -lc 'rg -n "exp109|exp148|exp149|exp333|SMPL|beta|β|avg|MaxSim|CARGO|AG-ReID|kill|contain|包含|visibility|可见" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2
experiments/paradigm_shift/paradigm_A_generation.md:3281:experiments/exp359_lm_reid/codex_review_raw_v2.md:3134:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4186:reassess/r_2.md:1097:reassess/r_3.md:2223:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:3282:experiments/exp359_lm_reid/codex_review_raw_v2.md:3149:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4202:reassess/r_2.md:1158:reassess/r_3.md:2382:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/novelty_gopl.txt:12:| (e) SMPL / 3D-geometry-guided ReID | 存活但不能大喊 first | [OG-Net / 3D Space ReID](https://arxiv.org/abs/2006.04569) 已把 2D 图像映射到 3D point cloud / visible surface 上学习 ReID 表征；[GCL](https://arxiv.org/abs/2012.09071) 用 HMR mesh rotation 生成新视角并做 contrastive；[CASE-Net / SMPL-reID](https://arxiv.org/abs/2003.07340) 用 SMPL 合成数据研究换衣/shape。 | 已有人用 SMPL/mesh/3D 做 ReID 表征、生成、合成数据。GOPL 的切口是：**SMPL 只作 pair-reliability meter，不作 identity feature、不作 alignment、不作 augmentation**。这个切口目前没看到确切先例。 | 8/10 |
experiments/paradigm_shift/paradigm_A_generation.md:3283:experiments/exp359_lm_reid/codex_review_raw_v2.md:3278:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4331:reassess/r_2.md:1640:reassess/r_3.md:3614:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3474:区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
experiments/paradigm_shift/paradigm_A_generation.md:3284:experiments/exp359_lm_reid/codex_review_raw_v2.md:3283:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4336:reassess/r_2.md:1650:reassess/r_3.md:3627:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3530:区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
experiments/paradigm_shift/paradigm_A_generation.md:3285:experiments/exp359_lm_reid/codex_review_raw_v2.md:3311:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4364:reassess/r_2.md:1740:reassess/r_3.md:3842:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3494:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:3286:experiments/exp359_lm_reid/codex_review_raw_v2.md:3313:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4366:reassess/r_2.md:1747:reassess/r_3.md:3850:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3567:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:3287:experiments/exp359_lm_reid/codex_review_raw_v2.md:3332:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4385:reassess/r_2.md:1814:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:3288:experiments/exp359_lm_reid/codex_review_raw_v2.md:3337:experiments/cargo_cvpb/litreview2/reassess/r_2.md:4390:reassess/r_2.md:1824:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:3289:experiments/exp359_lm_reid/codex_review_raw_v2.md:3541:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5176:reassess/r_3.md:3476:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_13.md:291:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:3290:experiments/exp359_lm_reid/codex_review_raw_v2.md:3565:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5201:reassess/r_3.md:3596:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:3291:experiments/exp359_lm_reid/codex_review_raw_v2.md:3573:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5209:reassess/r_3.md:3614:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3474:区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
experiments/paradigm_shift/paradigm_A_generation.md:3292:experiments/exp359_lm_reid/codex_review_raw_v2.md:3579:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5215:reassess/r_3.md:3627:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_2.md:3530:区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
experiments/paradigm_shift/paradigm_A_generation.md:3293:experiments/exp359_lm_reid/codex_review_raw_v2.md:3589:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5228:reassess/r_3.md:3660:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:454:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:3294:experiments/exp359_lm_reid/codex_review_raw_v2.md:3594:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5233:reassess/r_3.md:3680:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:3910:区别：Identity-aware diffusion 解决 RGB 红外生成，InfinitePerson 解决通用合成数据质量；我们专门控制相机高度、俯仰角、低清、遮挡和同一身体几何，目标是航拍地面跨视角不变性。  
experiments/paradigm_shift/paradigm_A_generation.md:3295:experiments/exp359_lm_reid/codex_review_raw_v2.md:3595:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5234:reassess/r_3.md:3694:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_10.md:3988:区别：Identity-aware diffusion 解决 RGB 红外生成，InfinitePerson 解决通用合成数据质量；我们专门控制相机高度、俯仰角、低清、遮挡和同一身体几何，目标是航拍地面跨视角不变性。  
experiments/paradigm_shift/paradigm_A_generation.md:3296:experiments/exp359_lm_reid/codex_review_raw_v2.md:3604:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5245:reassess/r_3.md:3730:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_7.md:437:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:3297:experiments/exp359_lm_reid/codex_review_raw_v2.md:3627:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5268:reassess/r_3.md:3788:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_18.md:702:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:3298:experiments/exp359_lm_reid/codex_review_raw_v2.md:3639:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5280:reassess/r_3.md:3829:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:534:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:3299:experiments/exp359_lm_reid/codex_review_raw_v2.md:3645:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5286:reassess/r_3.md:3842:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3494:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:3300:experiments/exp359_lm_reid/codex_review_raw_v2.md:3647:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5288:reassess/r_3.md:3850:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_6.md:3567:cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。
experiments/paradigm_shift/paradigm_A_generation.md:3301:experiments/exp359_lm_reid/codex_review_raw_v2.md:3670:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5312:reassess/r_3.md:3933:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_8.md:734:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:3302:experiments/exp359_lm_reid/codex_review_raw_v2.md:3686:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5330:reassess/r_3.md:4002:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:433:阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。
experiments/paradigm_shift/paradigm_A_generation.md:3303:experiments/exp359_lm_reid/codex_review_raw_v2.md:3689:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5333:reassess/r_3.md:4009:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2668:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:3304:experiments/exp359_lm_reid/codex_review_raw_v2.md:3694:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5338:reassess/r_3.md:4020:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview/reviews/lit_19.md:2744:和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。
experiments/paradigm_shift/paradigm_A_generation.md:3305:experiments/exp359_lm_reid/codex_review_raw_v2.md:3755:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5759:reassess/r_3.md:7775:./reassess/r_3.md:2223:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2023:- **★廉价 kill-switch(一晚)**: 合成人遮挡上, 用强 baseline(Swin-Small)embedding 训入侵者-ID 探针。≈随机→强主干已抑制无泄漏可挖→kill; >>随机→有故事。
experiments/paradigm_shift/paradigm_A_generation.md:3306:experiments/exp359_lm_reid/codex_review_raw_v2.md:3760:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5764:reassess/r_3.md:7784:./reassess/r_3.md:2243:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2184:- 这能自然长出方法：训练期合成已知 donor identity 的人遮挡、压制 donor-ID 可预测性、保持 target identity 一致；推理期不加任何外部输入。
experiments/paradigm_shift/paradigm_A_generation.md:3307:experiments/exp359_lm_reid/codex_review_raw_v2.md:3763:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5767:reassess/r_3.md:7788:./reassess/r_3.md:2249:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/p_3.md:2218:- 这能自然长出方法：训练期合成已知 donor identity 的人遮挡、压制 donor-ID 可预测性、保持 target identity 一致；推理期不加任何外部输入。
experiments/paradigm_shift/paradigm_A_generation.md:3308:experiments/exp359_lm_reid/codex_review_raw_v2.md:3772:experiments/cargo_cvpb/litreview2/reassess/r_2.md:5776:reassess/r_3.md:7807:./reassess/r_3.md:2351:/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/litreview2/pivot/clean/p_3.txt:28:- 这能自然长出方法：训练期合成已知 donor identity 的人遮挡、压制 donor-ID 可预测性、保持 target identity 一致；推理期不加任何外部输入。
experiments/paradigm_shift/paradigm_A_generation.md:3332:experiments/paradigm_shift/README.md:10:LM-ReID(exp359) 6.5 是最强 B 类候选，但训练端探索（守 SOLIDER/Swin 强 backbone 加小模块 + 历史模块）已 **100% 实测穷尽**。诚实结论：**不是"ReID 没训练端创新"，而是"在不换 backbone、不换量级、cheap kill-switch 的约束下，小修小补探到底了"**。真正的训练端创新是"范式级"的（换预训练/监督/数据来源）——CLIP-ReID(language)/SOLIDER(SSL pretraining)/Pose2ID(generation) 都是。用户授权松开所有约束去够这个量级。
experiments/paradigm_shift/paradigm_A_generation.md:3334:experiments/paradigm_shift/paradigm_C_free.md:14:ReID 训练端范式级创新评估(重大转向, 不要 cheap 不要判死)。背景: 守 SOLIDER/Swin 强 backbone 加小模块训练端已穷尽, 用户授权换量级可花周级时间。**算力 4 单卡 slot 无 from-scratch foundation 算力**。数据 occluded_duke/market/msmt17/occluded_reid/occluded_posetrack。**评估范式=自由(新监督信号/跨界 import/新训练目标)**。避死区: CLIP-ReID(language 占)/SMPL 几何(我项目多次证负无独特 ReID 信号)/FM-import MLLM-DINO-SD(我项目证负 判别性-互补性张力)/test-time trick。问: 2026 ReID 训练端哪个范式级方向(新监督信号如 video-temporal/attribute/synthetic-label, 或跨界 import 如 face-recognition/3D/retrieval-pretraining 训练范式, 或新训练目标)最有野心 × 4-slot 单卡周级可行 × 真 novelty × 能 actually build? 给 3 候选排序。**不判死, 选去 build + pipeline + 工作量 + 风险 + B类潜力。信心。务实中文。**
experiments/paradigm_shift/paradigm_A_generation.md:3338:experiments/paradigm_shift/paradigm_A_generation.md:15:ReID 训练端范式级创新评估(重大转向, 明确不要 cheap 不要判死)。背景: 遮挡 ReID 项目守 SOLIDER/Swin 强 backbone 加小模块训练端已 100% 实测穷尽。用户授权换量级, 可不 cheap 花周级时间。**算力: 4 单卡 slot(4090 24G/3090 24G/5060Ti 16G×2), 无 from-scratch foundation model 算力**。数据: occluded_duke/market/msmt17/occluded_reid/occluded_posetrack。**评估范式=生成式数据引擎**(diffusion SD/ControlNet + pose/SMPL/depth 控制造大规模 ID-consistent 跨视角跨遮挡合成 ReID 训练数据训 backbone)。问: (a)Pose2ID(2024)/GAN-DG 之后, '生成数据当大规模训练源'范式还有什么真 novelty(可控遮挡合成 occlusion-curriculum? 无限合成 ID 解决数据稀缺? sim2real domain-gap 系统方法? identity-disentangled generation?)? (b)我 4-slot 单卡(diffusion 造数据数天可累积)能 actually build 的最有野心版本(pipeline: 生成模型+控制+造多少+怎么训)? (c)避开 Pose2ID/DG-Market/各种 ReID-GAN-aug, 联网查 2024-26 generative-ReID novelty。**不判死, 选最能 build 方案+pipeline+工作量(天/周)+风险+B类以上潜力。信心 1-10。务实中文。**
experiments/paradigm_shift/paradigm_A_generation.md:3343:experiments/paradigm_shift/paradigm_A_generation.md:915:本地文档显示项目已经从“小模块穷尽”进入过更大的训练范式探索，尤其 exp148/151 类 single-image support 问题和后续 PLBOA/OA-SD/多视图增强。接下来我会重点查外部 2024-2026：Pose2ID、DG-Market/GAN-DG、diffusion/ControlNet 造 ReID 数据、ID 一致生成和 synthetic-to-real ReID。
experiments/paradigm_shift/paradigm_A_generation.md:3352:/bin/zsh -lc "sed -n '1,220p' experiments/paradigm_shift/README.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/paradigm_shift/paradigm_A_generation.md:3363:LM-ReID(exp359) 6.5 是最强 B 类候选，但训练端探索（守 SOLIDER/Swin 强 backbone 加小模块 + 历史模块）已 **100% 实测穷尽**。诚实结论：**不是"ReID 没训练端创新"，而是"在不换 backbone、不换量级、cheap kill-switch 的约束下，小修小补探到底了"**。真正的训练端创新是"范式级"的（换预训练/监督/数据来源）——CLIP-ReID(language)/SOLIDER(SSL pretraining)/Pose2ID(generation) 都是。用户授权松开所有约束去够这个量级。
experiments/paradigm_shift/paradigm_A_generation.md:3369:2. **新自监督预训练 pretext**：从 SOLIDER/DINOv2 权重 continued-pretraining（省算力，非 from-scratch）
experiments/paradigm_shift/paradigm_A_generation.md:3376:- ✅ 允许多日/多周训练、允许 from-checkpoint continued-pretraining、允许造数据
experiments/paradigm_shift/paradigm_A_generation.md:3378:- ✅ 死区仍避：SMPL 几何无独特 ReID 信号(多次证负)/FM-import MLLM-DINO-SD(判别性-互补性张力证负)/test-time trick 当主创新
experiments/paradigm_shift/paradigm_A_generation.md:3382:- [进行中] 范式 gap analysis：3 路 codex（generation-data / new-pretraining / free-supervision），导向"选一个能 build 的去做"，**不判死**。
experiments/paradigm_shift/paradigm_A_generation.md:3390:/bin/zsh -lc "sed -n '1,260p' experiments/paradigm_shift/paradigm_A_generation.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/paradigm_shift/paradigm_A_generation.md:3396:workdir: /Users/abdslm/Desktop/SOLIDER-REID
experiments/paradigm_shift/paradigm_A_generation.md:3406:ReID 训练端范式级创新评估(重大转向, 明确不要 cheap 不要判死)。背景: 遮挡 ReID 项目守 SOLIDER/Swin 强 backbone 加小模块训练端已 100% 实测穷尽。用户授权换量级, 可不 cheap 花周级时间。**算力: 4 单卡 slot(4090 24G/3090 24G/5060Ti 16G×2), 无 from-scratch foundation model 算力**。数据: occluded_duke/market/msmt17/occluded_reid/occluded_posetrack。**评估范式=生成式数据引擎**(diffusion SD/ControlNet + pose/SMPL/depth 控制造大规模 ID-consistent 跨视角跨遮挡合成 ReID 训练数据训 backbone)。问: (a)Pose2ID(2024)/GAN-DG 之后, '生成数据当大规模训练源'范式还有什么真 novelty(可控遮挡合成 occlusion-curriculum? 无限合成 ID 解决数据稀缺? sim2real domain-gap 系统方法? identity-disentangled generation?)? (b)我 4-slot 单卡(diffusion 造数据数天可累积)能 actually build 的最有野心版本(pipeline: 生成模型+控制+造多少+怎么训)? (c)避开 Pose2ID/DG-Market/各种 ReID-GAN-aug, 联网查 2024-26 generative-ReID novelty。**不判死, 选最能 build 方案+pipeline+工作量(天/周)+风险+B类以上潜力。信心 1-10。务实中文。**
experiments/paradigm_shift/paradigm_A_generation.md:3410:/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/paradigm_shift/paradigm_A_generation.md:3422:| 000 | Baseline (SOLIDER-Swin-Tiny, SW=0.2) | 56.6% | 66.5% | 79.4% | 83.4% | — | 120 epoch, 完美复现 |
experiments/paradigm_shift/paradigm_A_generation.md:3572:| 142-eq | PSG+GCN+SKC (Support-Supervised Keypoint Completion, eq) | 60.3% | 71.8% | 84.4% | 87.7% | vs exp030a-eq: -0.8%/-1.9% | ❌ 中性偏负。completion module 虽然活跃（gate=0.26, delta_norm=1.5），但 skc_pre≈skc_post 说明修改方向不是向 prototype 靠近。gate 无限制增长导致后期过度修改特征。feature-level completion 方向已被多轮验证为无效 |
experiments/paradigm_shift/paradigm_A_generation.md:3577:| 148-eq | PSG+GCN+PCVT (Pose-Complementary View Training, eq) | ~59.3%* | ~71.3%* | — | — | ❌ 负面。*ep100 数据，训练中。早期加速（ep30: +2.4 mAP）但后期被基线追平并反超。3-view 训练的 1/3 主损失稀释导致后期收敛不足。训练集 95.8% 全可见使 complementary masking 缺乏信号 |
experiments/paradigm_shift/paradigm_A_generation.md:3619:### Phase 4: SupCon + OA-SD + Parallel Aug (exp166-193)
experiments/paradigm_shift/paradigm_A_generation.md:3624:| 166r | ↳ base arch (no PAPE/MS-PSG) | 60.3% | 72.8% | — | — | CE base for OA-SD comparison |
experiments/paradigm_shift/paradigm_A_generation.md:3626:| **187** | **+SupCon T=0.05 + 3-view Parallel Aug** | **64.9%** | **76.6%** | **87.2%** | **90.0%** | **🟢🟢 Overall best! +1.8/+2.7 vs exp166** |
experiments/paradigm_shift/paradigm_A_generation.md:3627:| **190** | **3-view Parallel Aug + CE (no SupCon)** | **64.2%** | **75.6%** | **86.2%** | **89.1%** | **✅ 3-view+CE ≈ 1-view+SupCon! +1.1/+1.7 vs exp166** |
experiments/paradigm_shift/paradigm_A_generation.md:3628:| **191** | **OA-SD + CE (1-view, decay=0.999)** | **63.2%** | **75.4%** | **86.3%** | **89.1%** | **✅ OA-SD 独立有效! +2.9/+2.6 vs CE base** |
experiments/paradigm_shift/paradigm_A_generation.md:3629:| 192 | OA-SD + CE (1-view, decay=0.99) | 62.6% | 74.9% | 86.0% | 89.1% | 🟡 vs exp191: -0.6/-0.5。decay 不敏感 |
experiments/paradigm_shift/paradigm_A_generation.md:3630:| **193** | **OA-SD + 3-view + CE** | **64.4%** | **76.5%** | **86.3%** | **89.4%** | **✅ OA-SD+3-view additive! R1 ≈ exp187 SupCon, mAP +0.2/R1+0.9 vs exp190** |
experiments/paradigm_shift/paradigm_A_generation.md:3631:| 194 | OA-SD + CE (weight=2.0) | 63.4% | 74.8% | 86.1% | 89.1% | 🟡 vs exp191 (w=1.0): +0.2/-0.6。weight 不敏感 |
experiments/paradigm_shift/paradigm_A_generation.md:3634:/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/paradigm_shift/paradigm_A_generation.md:3636:/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/paradigm_shift/paradigm_A_generation.md:3654:/bin/zsh -lc "sed -n '1,180p' experiments/exp359_lm_reid/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/paradigm_shift/paradigm_A_generation.md:3754:/bin/zsh -lc "sed -n '1,220p' experiments/smpl_anchor_vireid/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/paradigm_shift/paradigm_A_generation.md:3790:→ **转 fallback: Swin-VI 机制**。VI-ReID 至今 ResNet50 主导、无 Swin/CLIP 赢家(纯 ViT PMT 才 67.5 R1), 团队是 Swin/SOLIDER 专家。CAJ ResNet baseline 已搭好(POOL 76.80/69.14)→ 换 Swin-Small(SOLIDER pretrain)→ 强 backbone 在 VI-ReID 帮不帮? 帮=机制论文; 不帮=为什么不帮(模态 gap 吞掉 backbone 红利)也是机制问题。先做这个经验首验。
experiments/exp324b/_codex_review.log:4:workdir: /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:14:Review two files for a person-ReID experiment exp324b: experiments/exp324b/design.md and scripts/exp324b_train_head.py (it imports helpers from scripts/exp324_dino.py — read that for context). Setup: FROZEN DINOv2-base dense tokens, pose-anchored into 5 body parts (cached), then a LIGHTWEIGHT shared Linear(768->512)+BNNeck+classifier head (~413K params, DINO never back-propped). Loss = ID CrossEntropy on BNNeck feat + batch-hard SOFT-MARGIN triplet on pre-BN global feat (global = masked-mean over visible projected parts), weight 1:1. PK sampler P=16 x K=4 = BS64. Eval = two distmats: global cosine + mutually-visible part-MaxSim (L2-normalized projected parts); mAP/R1 on ALL query and HEAVY-OCC (query pose visibility_binary.sum()<=8). Goal: lift exp324's frozen 1.86 heavy-occ mAP via training the head. Review line-by-line: bugs / runtime errors / shape-device-dtype mismatches / edge cases (zero-visible-part images, NaN); train/test asymmetry (head is supervised via GLOBAL masked-mean feature but tested via per-part normalized part-MaxSim — is this coupling sound? should there be a part-level loss?); triplet correctness (pos/neg masks, hardest mining, soft-margin); PK sampler correctness; eval correctness (same-pid-same-cam exclusion, heavy-occ mask slicing, query/gallery pose split mapping); single-variable isolation vs exp324; no LR scheduler concern. Use web search to check novelty/precedent of 'frozen foundation-model features + lightweight head + pose-part matching for occluded person ReID'. Output: Verdict (approve / needs-attention) + findings with severity (Critical/High/Medium/Low).
experiments/exp324b/_codex_review.log:16:I’ll read the experiment design, training script, and DINO helper first, then cross-check the review points against the actual code paths. After that I’ll do a focused literature/code search for close precedents before giving a severity-ranked verdict.
experiments/exp324b/_codex_review.log:18:/bin/zsh -lc "sed -n '1,260p' experiments/exp324b/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:20:# 实验 exp324b: 冻结 DINOv2 + 轻量投影头 + 姿态部位匹配
experiments/exp324b/_codex_review.log:22:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
experiments/exp324b/_codex_review.log:24:> **机器**：lab-3090-d（DINOv2-base 已下、exp324 已缓存部位特征、RTX 3090 idle）。
experiments/exp324b/_codex_review.log:28:- exp324（frozen，无训练）已证：用姿态把 DINOv2 dense token 锚定到 5 个身体部位、跨图只比双方可见部位（part-MaxSim），在 Occluded-Duke 重遮挡 query 上 mAP 1.86 vs 整图 0.55（×3.4），且**均匀网格对照只 0.67（几乎不涨）→ 涨点几乎全来自"姿态锚定"机制本身**（单变量隔离干净）。
experiments/exp324b/_codex_review.log:29:- 但绝对分低（1.86 mAP），符合 DINO 零样本 ReID 文献（0.3-4.7）。training-free 不可用。
experiments/exp324b/_codex_review.log:30:- 与 exp323 对照：同样 frozen + 同样 pose，**MLLM-reasoning 那条无信号、DINO dense-correspondence 这条有信号**——差别在"特征表示端"而非"大模型推理端"。下一步应在特征端使劲：用最小训练把冻结特征投到 ReID 判别空间。
experiments/exp324b/_codex_review.log:34:冻结 DINOv2-base、**仅训一个轻量 per-part 投影头**（先线性，必要时小 MLP），用 ID 分类 + triplet loss 训练、用 mutually-visible part-MaxSim 匹配，能把重遮挡 mAP 从 1.86 **大幅拉高**（先看能否进入"几十分"量级证明冻结特征有救），同时保留"姿态锚定 + 只比可见部位"的机制优势。
experiments/exp324b/_codex_review.log:38:- **数据流**：图 → 冻结 DINOv2-base dense tokens（已缓存）→ 姿态锚定 5 部位池化 + per-part visibility（exp324 已实现）→ **轻量投影头**（per-part 线性 768→D，或共享线性 + 可选 BN/L2-norm）→ 训练时每部位接 ID 分类头（CE）+ part/global triplet；测试用 mutually-visible part-MaxSim 出 distmat。
experiments/exp324b/_codex_review.log:39:- **冻结边界**：DINO **不反传**（不解冻 backbone，避免大成本 + 防过拟合 15K 数据）；只训投影头 + 分类器。**在缓存特征上训** → 每 step 无 backbone forward，飞快。
experiments/exp324b/_codex_review.log:46:- 假设成立：重遮挡 mAP 从 1.86 显著上升（若进入几十分量级 → 冻结特征有救、方向成立 → 再加容量/解冻部分 DINO 冲 SOTA）。
experiments/exp324b/_codex_review.log:47:- 失败最可能：冻结 DINO 特征 ReID 判别力有天花板，轻量头拉不动（停在个位数）→ 则下一步换 LoRA 解冻部分 DINO，或换更强 DINO（large/v3）；若仍不动则这条线天花板低、止损。
experiments/exp324b/_codex_review.log:64:/bin/zsh -lc "sed -n '1,360p' scripts/exp324b_train_head.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:66:/bin/zsh -lc "sed -n '1,360p' scripts/exp324_dino.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:69:"""exp324: Training-free DINOv2 dense-correspondence + pose-anchored part matching
experiments/exp324b/_codex_review.log:72:Hypothesis: frozen DINOv2-base dense patch features, anchored into 5 body parts by
experiments/exp324b/_codex_review.log:81:  1. Extract DINOv2 patch tokens for all query+gallery imgs (resize 224W x 448H,
experiments/exp324b/_codex_review.log:102:ROOT = '/root/work/SOLIDER-REID'
experiments/exp324b/_codex_review.log:152:# ---- DINOv2 input / grid geometry ----
experiments/exp324b/_codex_review.log:203:# DINOv2 feature extraction (cached)
experiments/exp324b/_codex_review.log:208:    m = AutoModel.from_pretrained('facebook/dinov2-base').eval().cuda()
experiments/exp324b/_codex_review.log:431:"""exp324b: Train a LIGHTWEIGHT projection head on top of FROZEN DINOv2-base
experiments/exp324b/_codex_review.log:434:Context (exp324): frozen DINOv2-base dense tokens, anchored into 5 body parts by
experiments/exp324b/_codex_review.log:441:  * DINOv2 is FROZEN, never back-propped. We pre-extract pose-anchored 5-part
experiments/exp324b/_codex_review.log:443:    step operates on cached features -> NO DINO forward in the loop -> fast.
experiments/exp324b/_codex_review.log:471:ROOT = '/root/work/SOLIDER-REID'
experiments/exp324b/_codex_review.log:490:CACHE_QG = os.path.join(ROOT, 'experiments/exp324/_cache')  # query/gallery DINO patch cache (reuse)
experiments/exp324b/_codex_review.log:508:      parts   : (N,5,768) float32  -- raw part-mean DINO vectors (NOT normalized)
experiments/exp324b/_codex_review.log:794:/bin/zsh -lc "sed -n '360,760p' scripts/exp324b_train_head.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:796:/bin/zsh -lc "sed -n '360,760p' scripts/exp324_dino.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:983:    # ---- DINO features ----
experiments/exp324b/_codex_review.log:1048:/bin/zsh -lc 'rg -n "exp324b|exp324|heavy_mask|part_maxsim|PKSampler|batch_hard_triplet|CACHE_QG|_cache_train|vis.*sum|visibility_binary" experiments scripts -S' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1053:scripts/exp324_dino.py:2:"""exp324: Training-free DINOv2 dense-correspondence + pose-anchored part matching
experiments/exp324b/_codex_review.log:1076:scripts/eval_oracle_support_bank.py:55:    if vis.sum() < 2:
experiments/exp324b/_codex_review.log:1077:scripts/eval_oracle_support_bank.py:57:    if vis.sum() == 0:
experiments/exp324b/_codex_review.log:1078:scripts/eval_oracle_support_bank.py:65:    if vis.sum() < 4:
experiments/exp324b/_codex_review.log:1079:scripts/eval_oracle_support_bank.py:211:            target_visible = (test_feat['kp_weights'].detach().cpu() > vis_thr).sum(dim=1).numpy()
experiments/exp324b/_codex_review.log:1080:scripts/eval_oracle_support_bank.py:258:        sum_w = (weights_pid * vis_mask.float()).sum(dim=0)
experiments/exp324b/_codex_review.log:1081:scripts/eval_oracle_support_bank.py:259:        count = vis_mask.sum(dim=0)
experiments/exp324b/_codex_review.log:1082:scripts/eval_oracle_support_bank.py:273:            recover_mask = (kp_weights[sample_idx] <= vis_thr) & (local_count > 0) & (local_sum_w > 1e-12)
experiments/exp324b/_codex_review.log:1083:scripts/eval_oracle_support_bank.py:396:            'query_target_vis<=8': int((q_visible <= 8).sum()),
experiments/exp324b/_codex_review.log:1084:scripts/eval_oracle_support_bank.py:397:            'query_target_vis<=5': int((q_visible <= 5).sum()),
experiments/exp324b/_codex_review.log:1106:scripts/exp324b_train_head.py:2:"""exp324b: Train a LIGHTWEIGHT projection head on top of FROZEN DINOv2-base
experiments/exp324b/_codex_review.log:1107:scripts/exp324b_train_head.py:5:Context (exp324): frozen DINOv2-base dense tokens, anchored into 5 body parts by
experiments/exp324b/_codex_review.log:1117:scripts/exp324b_train_head.py:61:CACHE_QG = os.path.join(ROOT, 'experiments/exp324/_cache')  # query/gallery DINO patch cache (reuse)
experiments/exp324b/_codex_review.log:1142:experiments/decisions.md:4470:**执行结果**：建议砍 frozen-MLLM-reasoner 廉价首验，转 exp324（DINO-correspondence，更 frontier-independent）或换机制。
experiments/exp324b/_codex_review.log:1143:experiments/decisions.md:4473:### [2026-06-16] 决策 #exp324 — DINO emergent correspondence + pose-anchored part-MaxSim 首验偏正
experiments/exp324b/_codex_review.log:1145:experiments/decisions.md:4487:**执行结果**：exp324b 候选——冻结 DINO，仅训一个轻量 part-projection 头（或 LoRA）把 token 投到 ReID-judiciable 空间，
experiments/exp324b/_codex_review.log:1158:experiments/results.md:1652:### exp324: frozen DINOv2 emergent correspondence + pose-anchored part-MaxSim（inference-only，非训练）
experiments/exp324b/_codex_review.log:1159:experiments/results.md:1654:> post-PRCV「搬范式」#2 路线。frozen DINOv2-base（lab-3090-d, RTX 3090），全量 Occluded-Duke（2210 query × 17661 gallery，无后处理、无训练）。脚本 `scripts/exp324_dino.py`。输入 224W×448H → patch grid 32×16。keypoints 缩放到 grid → 每部位 3×3 窗均值池化成 5 个 part 向量 + per-part visibility，跨图只比 mutually-visible part 的 per-part cosine（part-MaxSim）。重遮挡子集 = query visibility_binary.sum()≤8（989/2210）。**training-free，不计入主表增益。**
experiments/exp324b/_codex_review.log:1160:experiments/results.md:1667:- **结论**：机制**有明确相对信号**，pose-anchored DINO correspondence 在重遮挡上 3-4 倍超整图基准且 pose 锚定占绝对主导 → kill-switch 命中正向条件，**值得 exp324b 上轻量 part-projection 头 / LoRA**。
experiments/exp324b/_codex_review.log:1164:experiments/exp323/design.md:21:1. **重遮挡子集定义**：每张 query 的可见关键点数 = `visibility_binary.sum()`。取**可见数最低的子集**（如底部 30%，或 visible≤8，复用 exp109 分桶口径）作 heavy-occ query。
experiments/exp324b/_codex_review.log:1167:experiments/exp323/design.md:51:**战略警告**：MLLM-reasoner 有点在和前沿赛跑（模型越强缺口越小，卖点随时间衰减）→ 方法须落到"蒸馏进小可部署网络"；exp324 DINO-correspondence 更 frontier-independent。梯度实验顺带量化衰减速度。
experiments/exp324b/_codex_review.log:1168:experiments/exp323/design.md:56:- 小模型也 B≈A → 砍，转 exp324（DINO-correspondence），不沉没成本。（GPT-5.5 单独 B≈A 不算 kill，可能天花板。）
experiments/exp324b/_codex_review.log:1169:experiments/exp324b/design.md:1:# 实验 exp324b: 冻结 DINOv2 + 轻量投影头 + 姿态部位匹配
experiments/exp324b/_codex_review.log:1170:experiments/exp324b/design.md:3:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
experiments/exp324b/_codex_review.log:1171:experiments/exp324b/design.md:5:> **机器**：lab-3090-d（DINOv2-base 已下、exp324 已缓存部位特征、RTX 3090 idle）。
experiments/exp324b/_codex_review.log:1172:experiments/exp324b/design.md:9:- exp324（frozen，无训练）已证：用姿态把 DINOv2 dense token 锚定到 5 个身体部位、跨图只比双方可见部位（part-MaxSim），在 Occluded-Duke 重遮挡 query 上 mAP 1.86 vs 整图 0.55（×3.4），且**均匀网格对照只 0.67（几乎不涨）→ 涨点几乎全来自"姿态锚定"机制本身**（单变量隔离干净）。
experiments/exp324b/_codex_review.log:1173:experiments/exp324b/design.md:19:- **数据流**：图 → 冻结 DINOv2-base dense tokens（已缓存）→ 姿态锚定 5 部位池化 + per-part visibility（exp324 已实现）→ **轻量投影头**（per-part 线性 768→D，或共享线性 + 可选 BN/L2-norm）→ 训练时每部位接 ID 分类头（CE）+ part/global triplet；测试用 mutually-visible part-MaxSim 出 distmat。
experiments/exp324b/_codex_review.log:1181:experiments/exp324b/_codex_review.log:14:Review two files for a person-ReID experiment exp324b: experiments/exp324b/design.md and scripts/exp324b_train_head.py (it imports helpers from scripts/exp324_dino.py — read that for context). Setup: FROZEN DINOv2-base dense tokens, pose-anchored into 5 body parts (cached), then a LIGHTWEIGHT shared Linear(768->512)+BNNeck+classifier head (~413K params, DINO never back-propped). Loss = ID CrossEntropy on BNNeck feat + batch-hard SOFT-MARGIN triplet on pre-BN global feat (global = masked-mean over visible projected parts), weight 1:1. PK sampler P=16 x K=4 = BS64. Eval = two distmats: global cosine + mutually-visible part-MaxSim (L2-normalized projected parts); mAP/R1 on ALL query and HEAVY-OCC (query pose visibility_binary.sum()<=8). Goal: lift exp324's frozen 1.86 heavy-occ mAP via training the head. Review line-by-line: bugs / runtime errors / shape-device-dtype mismatches / edge cases (zero-visible-part images, NaN); train/test asymmetry (head is supervised via GLOBAL masked-mean feature but tested via per-part normalized part-MaxSim — is this coupling sound? should there be a part-level loss?); triplet correctness (pos/neg masks, hardest mining, soft-margin); PK sampler correctness; eval correctness (same-pid-same-cam exclusion, heavy-occ mask slicing, query/gallery pose split mapping); single-variable isolation vs exp324; no LR scheduler concern. Use web search to check novelty/precedent of 'frozen foundation-model features + lightweight head + pose-part matching for occluded person ReID'. Output: Verdict (approve / needs-attention) + findings with severity (Critical/High/Medium/Low).
experiments/exp324b/_codex_review.log:1182:experiments/exp324b/_codex_review.log:18:/bin/zsh -lc "sed -n '1,260p' experiments/exp324b/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1183:experiments/exp324b/_codex_review.log:20:# 实验 exp324b: 冻结 DINOv2 + 轻量投影头 + 姿态部位匹配
experiments/exp324b/_codex_review.log:1184:experiments/exp324b/_codex_review.log:22:> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
experiments/exp324b/_codex_review.log:1185:experiments/exp324b/_codex_review.log:24:> **机器**：lab-3090-d（DINOv2-base 已下、exp324 已缓存部位特征、RTX 3090 idle）。
experiments/exp324b/_codex_review.log:1186:experiments/exp324b/_codex_review.log:28:- exp324（frozen，无训练）已证：用姿态把 DINOv2 dense token 锚定到 5 个身体部位、跨图只比双方可见部位（part-MaxSim），在 Occluded-Duke 重遮挡 query 上 mAP 1.86 vs 整图 0.55（×3.4），且**均匀网格对照只 0.67（几乎不涨）→ 涨点几乎全来自"姿态锚定"机制本身**（单变量隔离干净）。
experiments/exp324b/_codex_review.log:1187:experiments/exp324b/_codex_review.log:38:- **数据流**：图 → 冻结 DINOv2-base dense tokens（已缓存）→ 姿态锚定 5 部位池化 + per-part visibility（exp324 已实现）→ **轻量投影头**（per-part 线性 768→D，或共享线性 + 可选 BN/L2-norm）→ 训练时每部位接 ID 分类头（CE）+ part/global triplet；测试用 mutually-visible part-MaxSim 出 distmat。
experiments/exp324b/_codex_review.log:1193:experiments/exp324b/_codex_review.log:64:/bin/zsh -lc "sed -n '1,360p' scripts/exp324b_train_head.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1194:experiments/exp324b/_codex_review.log:66:/bin/zsh -lc "sed -n '1,360p' scripts/exp324_dino.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1195:experiments/exp324b/_codex_review.log:69:"""exp324: Training-free DINOv2 dense-correspondence + pose-anchored part matching
experiments/exp324b/_codex_review.log:1200:experiments/exp324b/_codex_review.log:431:"""exp324b: Train a LIGHTWEIGHT projection head on top of FROZEN DINOv2-base
experiments/exp324b/_codex_review.log:1201:experiments/exp324b/_codex_review.log:434:Context (exp324): frozen DINOv2-base dense tokens, anchored into 5 body parts by
experiments/exp324b/_codex_review.log:1211:experiments/exp324b/_codex_review.log:490:CACHE_QG = os.path.join(ROOT, 'experiments/exp324/_cache')  # query/gallery DINO patch cache (reuse)
experiments/exp324b/_codex_review.log:1228:experiments/exp324b/_codex_review.log:794:/bin/zsh -lc "sed -n '360,760p' scripts/exp324b_train_head.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1229:experiments/exp324b/_codex_review.log:796:/bin/zsh -lc "sed -n '360,760p' scripts/exp324_dino.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1245:experiments/exp324/design.md:1:# 实验 exp324: DINOv2 Emergent Correspondence — pose-anchored part-MaxSim 廉价首验
experiments/exp324b/_codex_review.log:1247:experiments/exp324/design.md:37:- 重遮挡组超 holistic 且 pose 锚定有效 → exp324b：轻量 part-projection 头 / LoRA 微调 DINO，全量评测 vs KPR。
experiments/exp324b/_codex_review.log:1249:experiments/exp324/monitor.md:3:脚本：`scripts/exp324_dino.py`（training-free，frozen DINOv2-base，纯推理）
experiments/exp324b/_codex_review.log:1252:experiments/exp324/monitor.md:52:- **值得下一步**：上轻量 part-projection 头 / LoRA 把 DINO 特征投到 ReID-judiciable 空间，再全量对比 KPR（design.md kill-switch 命中"重遮挡组超 holistic 且 pose 锚定有效" → exp324b）。
experiments/exp324b/_codex_review.log:1257:/bin/zsh -lc "nl -ba scripts/exp324b_train_head.py | sed -n '1,220p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1260:     2	"""exp324b: Train a LIGHTWEIGHT projection head on top of FROZEN DINOv2-base
experiments/exp324b/_codex_review.log:1263:     5	Context (exp324): frozen DINOv2-base dense tokens, anchored into 5 body parts by
experiments/exp324b/_codex_review.log:1270:    12	  * DINOv2 is FROZEN, never back-propped. We pre-extract pose-anchored 5-part
experiments/exp324b/_codex_review.log:1272:    14	    step operates on cached features -> NO DINO forward in the loop -> fast.
experiments/exp324b/_codex_review.log:1300:    42	ROOT = '/root/work/SOLIDER-REID'
experiments/exp324b/_codex_review.log:1319:    61	CACHE_QG = os.path.join(ROOT, 'experiments/exp324/_cache')  # query/gallery DINO patch cache (reuse)
experiments/exp324b/_codex_review.log:1337:    79	      parts   : (N,5,768) float32  -- raw part-mean DINO vectors (NOT normalized)
experiments/exp324b/_codex_review.log:1481:/bin/zsh -lc "nl -ba experiments/exp324b/design.md | sed -n '1,140p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1483:/bin/zsh -lc "nl -ba scripts/exp324b_train_head.py | sed -n '220,520p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1485:     1	# 实验 exp324b: 冻结 DINOv2 + 轻量投影头 + 姿态部位匹配
experiments/exp324b/_codex_review.log:1487:     3	> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
experiments/exp324b/_codex_review.log:1489:     5	> **机器**：lab-3090-d（DINOv2-base 已下、exp324 已缓存部位特征、RTX 3090 idle）。
experiments/exp324b/_codex_review.log:1493:     9	- exp324（frozen，无训练）已证：用姿态把 DINOv2 dense token 锚定到 5 个身体部位、跨图只比双方可见部位（part-MaxSim），在 Occluded-Duke 重遮挡 query 上 mAP 1.86 vs 整图 0.55（×3.4），且**均匀网格对照只 0.67（几乎不涨）→ 涨点几乎全来自"姿态锚定"机制本身**（单变量隔离干净）。
experiments/exp324b/_codex_review.log:1494:    10	- 但绝对分低（1.86 mAP），符合 DINO 零样本 ReID 文献（0.3-4.7）。training-free 不可用。
experiments/exp324b/_codex_review.log:1495:    11	- 与 exp323 对照：同样 frozen + 同样 pose，**MLLM-reasoning 那条无信号、DINO dense-correspondence 这条有信号**——差别在"特征表示端"而非"大模型推理端"。下一步应在特征端使劲：用最小训练把冻结特征投到 ReID 判别空间。
experiments/exp324b/_codex_review.log:1499:    15	冻结 DINOv2-base、**仅训一个轻量 per-part 投影头**（先线性，必要时小 MLP），用 ID 分类 + triplet loss 训练、用 mutually-visible part-MaxSim 匹配，能把重遮挡 mAP 从 1.86 **大幅拉高**（先看能否进入"几十分"量级证明冻结特征有救），同时保留"姿态锚定 + 只比可见部位"的机制优势。
experiments/exp324b/_codex_review.log:1503:    19	- **数据流**：图 → 冻结 DINOv2-base dense tokens（已缓存）→ 姿态锚定 5 部位池化 + per-part visibility（exp324 已实现）→ **轻量投影头**（per-part 线性 768→D，或共享线性 + 可选 BN/L2-norm）→ 训练时每部位接 ID 分类头（CE）+ part/global triplet；测试用 mutually-visible part-MaxSim 出 distmat。
experiments/exp324b/_codex_review.log:1504:    20	- **冻结边界**：DINO **不反传**（不解冻 backbone，避免大成本 + 防过拟合 15K 数据）；只训投影头 + 分类器。**在缓存特征上训** → 每 step 无 backbone forward，飞快。
experiments/exp324b/_codex_review.log:1511:    27	- 假设成立：重遮挡 mAP 从 1.86 显著上升（若进入几十分量级 → 冻结特征有救、方向成立 → 再加容量/解冻部分 DINO 冲 SOTA）。
experiments/exp324b/_codex_review.log:1512:    28	- 失败最可能：冻结 DINO 特征 ReID 判别力有天花板，轻量头拉不动（停在个位数）→ 则下一步换 LoRA 解冻部分 DINO，或换更强 DINO（large/v3）；若仍不动则这条线天花板低、止损。
experiments/exp324b/_codex_review.log:1816:/bin/zsh -lc "nl -ba scripts/exp324_dino.py | sed -n '1,340p'" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:1819:     2	"""exp324: Training-free DINOv2 dense-correspondence + pose-anchored part matching
experiments/exp324b/_codex_review.log:1822:     5	Hypothesis: frozen DINOv2-base dense patch features, anchored into 5 body parts by
experiments/exp324b/_codex_review.log:1831:    14	  1. Extract DINOv2 patch tokens for all query+gallery imgs (resize 224W x 448H,
experiments/exp324b/_codex_review.log:1852:    35	ROOT = '/root/work/SOLIDER-REID'
experiments/exp324b/_codex_review.log:1902:    85	# ---- DINOv2 input / grid geometry ----
experiments/exp324b/_codex_review.log:1953:   136	# DINOv2 feature extraction (cached)
experiments/exp324b/_codex_review.log:1958:   141	    m = AutoModel.from_pretrained('facebook/dinov2-base').eval().cuda()
experiments/exp324b/_codex_review.log:2162:/bin/zsh -lc "sed -n '1,260p' loss/triplet_loss.py" in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:2306:/bin/zsh -lc 'rg -n "class TripletLoss|SoftMarginLoss|margin_ranking_loss|def hard_example_mining|euclidean_dist" loss layers solver model -S' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:2318:/bin/zsh -lc 'rg -n "RandomIdentitySampler|PK|num_instances|batch_size|P=|K=" . -S' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:2341:./config/defaults.py:260:_C.MODEL.POSE_BT_PKD = False              # Enable BT-PKD (requires OA-SD)
experiments/exp324b/_codex_review.log:2501:./experiments/exp224/monitor.md:3:配置: 基于 pose_psg_gcn_paa_roa.yml (含 ROA) + OA-SD + PLBOA + POSE_MULTI_SCALE_KP=True
experiments/exp324b/_codex_review.log:2505:./experiments/exp224/monitor.md:121:**exp223b (PADPQ K=8) FINAL: 63.9/74.3** (vs OA-SD 63.2/75.4 = +0.7/-1.1)
experiments/exp324b/_codex_review.log:2511:./experiments/exp213/design.md:1:# 实验 exp213: Small GCN+PAA+CE+OA-SD + PKC(0.05) + MST(0.1)
experiments/exp324b/_codex_review.log:2517:./experiments/exp213/monitor.md:1:# exp213 Small + GCN+PAA+CE+OA-SD + PKC(0.05) + MST(0.1) 监控
experiments/exp324b/_codex_review.log:2563:./experiments/decisions.md:3190:### [2026-04-01 10:30] 决策 #X — MaxSim + PKC + Fixed OA-SD 路线
experiments/exp324b/_codex_review.log:2564:./experiments/decisions.md:3192:**上下文**: MaxSim hybrid 在 exp206 checkpoint 上无需重训给 +1.8% mAP (70.3→72.1)。OA-SD teacher bug 已修复。PKC (Per-Keypoint Contrastive) 开始测试。
experiments/exp324b/_codex_review.log:2576:./experiments/decisions.md:3300:| PADPQ K=4+OA-SD | **63.7/74.5** | 63.9/74.8 |
experiments/exp324b/_codex_review.log:2577:./experiments/decisions.md:3301:| PADPQ K=8+OA-SD | 进行中 | 进行中 |
experiments/exp324b/_codex_review.log:2586:./experiments/exp223/monitor.md:1:# exp223 Tiny + GCN+PAA+CE+OA-SD + PADPQ K=4 监控
experiments/exp324b/_codex_review.log:2587:./experiments/exp223/monitor.md:3:配置: Tiny GCN+PAA+CE+OA-SD + Deformable keypoint sampling (K=4)
experiments/exp324b/_codex_review.log:2595:./experiments/exp223/monitor.md:164:| Epoch | K=4 | K=8 | OA-SD | delta K=8 vs OA-SD |
experiments/exp324b/_codex_review.log:2596:./experiments/exp223/monitor.md:171:**K=8 在 ep40 领先 OA-SD +0.7!** (K=4 同期 -1.3)
experiments/exp324b/_codex_review.log:2599:./experiments/exp223/monitor.md:182:**K=4 ep70: 61.2/71.7** (vs OA-SD 61.8/73.1 = -0.6/-1.4)
experiments/exp324b/_codex_review.log:2600:./experiments/exp223/monitor.md:184:K=8 still hope — was +0.7 vs OA-SD at ep40 (vs K=4's -1.3).
experiments/exp324b/_codex_review.log:2602:./experiments/exp223/monitor.md:189:K=4 ep~75. K=8 ep50: 59.5 (vs OA-SD 59.0 = +0.5, vs K=4 60.3 = -0.8).
experiments/exp324b/_codex_review.log:2604:./experiments/exp223/monitor.md:200:**K=4 ep80: 62.0%** (vs OA-SD 62.0 = **0.0! 完全持平！**)
experiments/exp324b/_codex_review.log:2608:./experiments/exp223/monitor.md:219:**K=4 ep90: 63.1/74.3!** (vs OA-SD 62.4/75.1 = **+0.7/-0.8!**)
experiments/exp324b/_codex_review.log:2611:./experiments/exp223/monitor.md:243:**K=4 ep100: 63.3/73.8!** (vs OA-SD 63.0/75.0 = **+0.3/-1.2**)
experiments/exp324b/_codex_review.log:2612:./experiments/exp223/monitor.md:263:**K=4 ep110: 63.5/74.2** (vs OA-SD 63.1/75.3 = **+0.4/-1.1**)
experiments/exp324b/_codex_review.log:2613:./experiments/exp223/monitor.md:283:**exp223 (PADPQ K=4 + OA-SD) FINAL: 63.7/74.5**
experiments/exp324b/_codex_review.log:2618:./experiments/exp223/monitor.md:327:- 也没有证据表明 `K=8` 能形成对 `OA-SD final = 63.2/75.4` 的稳定综合超越
experiments/exp324b/_codex_review.log:2631:./experiments/exp214/design.md:18:- exp210b (1-view + OA-SD + PKC): 70.6/81.8 (eq), 72.4/83.1 (maxsim)
experiments/exp324b/_codex_review.log:2634:./experiments/exp214/claude_review.md:96:此前 exp206 3-view+OA-SD 在 WITH_CP=True 下成功运行（4 views），但出现学习停滞。exp214 去掉了 OA-SD（3 views）且不用 CP，内存应该是够的（3 with-grad < 3 with-grad + 1 no-grad + CP overhead），但建议监控前几个 iter 的 GPU 内存。
experiments/exp324b/_codex_review.log:2643:./experiments/exp210b/design.md:1:# 实验 exp210b: GCN+PAA+CE+OA-SD + PKC (weight=0.05)
experiments/exp324b/_codex_review.log:2650:./experiments/exp210b/design.md:17:- exp206r (fixed OA-SD, no PKC): 70.6/82.6 (equal_concat), 72.3/82.9 (maxsim_hybrid)
experiments/exp324b/_codex_review.log:2652:./experiments/exp210b/monitor.md:1:# exp210b Small + GCN+PAA+CE+OA-SD + PKC (weight=0.05) 监控
experiments/exp324b/_codex_review.log:2702:./experiments/exp215/monitor.md:1:# exp215 Small + GCN+PAA+CE+OA-SD + BA-PKC (weight=0.1) 监控
experiments/exp324b/_codex_review.log:2773:./experiments/innovation_brainstorm.md:2822:1. **Swin-Small + GCN+PAA+OA-SD + maxsim_hybrid = 72.4%** (exp210b with PKC=0.05)
experiments/exp324b/_codex_review.log:2789:./experiments/exp225/design.md:5:- PADPQ K=4 (exp223): **equal_concat 63.7** (+0.5 mAP vs OA-SD), R1 74.5 (-0.9)
experiments/exp324b/_codex_review.log:2792:./experiments/exp225/monitor.md:3:配置: 基于 pose_psg_gcn_paa_roa.yml + OA-SD + PLBOA + GSPB(0.05) + PADPQ(K=4)
experiments/exp324b/_codex_review.log:2793:./experiments/exp225/monitor.md:79:**exp225 (GSPB scale=0.05 + PADPQ K=4 + OA-SD) FINAL: 64.2/74.9**
experiments/exp324b/_codex_review.log:2829:./experiments/exp230/design.md:1:# 实验 exp230: BT-PKD on Small (OA-SD baseline)
experiments/exp324b/_codex_review.log:2835:./experiments/exp230/monitor.md:1:# exp230 Small + BT-PKD(0.01) + OA-SD 监控
experiments/exp324b/_codex_review.log:2836:./experiments/exp230/monitor.md:3:配置: Small + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01)
experiments/exp324b/_codex_review.log:2857:./experiments/exp230/claude_review.md:13:**单变量原则**: 满足。相对于 exp206r (Small OA-SD) 只添加了 BT-PKD。
experiments/exp324b/_codex_review.log:2875:./experiments/results.md:973:### exp210: Small GCN+PAA+CE+OA-SD + PKC weight=0.5 — 灾难
experiments/exp324b/_codex_review.log:2877:./experiments/results.md:981:### exp210b: Small GCN+PAA+CE+OA-SD + PKC weight=0.05
experiments/exp324b/_codex_review.log:2882:./experiments/results.md:1028:| exp223 PADPQ K=4 + OA-SD | 63.7% | 74.5% | 86.2% | 89.5% | ep120 final |
experiments/exp324b/_codex_review.log:2884:./experiments/results.md:1044:| exp225 GSPB(0.05) + PADPQ K=4 + OA-SD | 64.2% | 74.9% | 86.8% | 89.6% | ep120 final |
experiments/exp324b/_codex_review.log:2885:./experiments/results.md:1049:- exp225: **GSPB+PADPQ K=4 = 64.2/74.9 — Tiny 最佳 equal_concat！** (+1.0/-0.5 vs OA-SD)
experiments/exp324b/_codex_review.log:2886:./experiments/results.md:1052:### exp227: Small GSPB(0.005) + PADPQ K=4 + OA-SD
experiments/exp324b/_codex_review.log:2887:./experiments/results.md:1064:### exp228: Tiny GSPB(0.05) + PADPQ K=8 + OA-SD
experiments/exp324b/_codex_review.log:2890:./experiments/results.md:1073:### exp229: Tiny BT-PKD (w=0.01, constant) + OA-SD
experiments/exp324b/_codex_review.log:2919:./experiments/exp282_gcn256_1stg_s_od_s42/claude_review.md:42:lab4090 RTX 4090 24GB 显存 + Swin-Small + Full Scaffold (LGPA + GCN + OA-SD + ParAug) + WITH_CP=True。exp262 在原 srvA (5060 Ti 16GB) 也跑通,4090 裕度大得多。GCN hidden 降半 + PSG stage 减半 → 显存进一步减少。OOM 概率 ≈ 0。
experiments/exp324b/_codex_review.log:2936:./experiments/exp227/monitor.md:1:# exp227 Small + GSPB(0.005) + PADPQ K=4 + OA-SD 监控
experiments/exp324b/_codex_review.log:2937:./experiments/exp227/monitor.md:3:配置: Small + GCN+PAA+OA-SD+PLBOA+ROA + GSPB(0.005) + PADPQ(K=4)
experiments/exp324b/_codex_review.log:2950:./experiments/exp245/monitor.md:44:**下一步**: 尝试 Small + LGPA-D + OA-SD + WITH_CP=True (gradient checkpointing 节省显存)
experiments/exp324b/_codex_review.log:2951:./experiments/exp245/monitor.md:61:远程 Small + LGPA-D + OA-SD + WITH_CP=True 启动成功!
experiments/exp324b/_codex_review.log:2971:./experiments/exp209/monitor.md:124:**实验终止。远程 GPU 转给 exp210 (GCN+PAA+CE+OA-SD+PKC)。**
experiments/exp324b/_codex_review.log:2986:./experiments/exp229/monitor.md:1:# exp229 Tiny + BT-PKD(0.01) + OA-SD 监控
experiments/exp324b/_codex_review.log:2987:./experiments/exp229/monitor.md:3:配置: Tiny + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01)
experiments/exp324b/_codex_review.log:2989:./experiments/exp229/monitor.md:20:vs OA-SD-only (exp191) ep6: Acc ~0.12 — **BT-PKD 与 baseline 一致或略好！**
experiments/exp324b/_codex_review.log:3003:./experiments/exp229/claude_review.md:7:**单变量原则**: 满足。在 OA-SD baseline (exp191) 基础上只增加了 BT-PKD 一个变量。
experiments/exp324b/_codex_review.log:3006:./experiments/exp229/claude_review.md:22:_C.MODEL.POSE_BT_PKD = False              # Enable BT-PKD (requires OA-SD)
experiments/exp324b/_codex_review.log:3018:./experiments/exp229/claude_review.md:115:**代码位置**: 在 OA-SD loss 计算之后（line 775 之后），仍在 `if oa_sd_enabled and ...` 块内。这意味着 BT-PKD 只在 OA-SD 激活时才运行，符合设计要求。
experiments/exp324b/_codex_review.log:3024:./experiments/exp229/claude_review.md:142:- **vs exp191 (OA-SD only)**: 仅新增 BT-PKD loss，单变量隔离
experiments/exp324b/_codex_review.log:3028:./experiments/exp210/design.md:1:# 实验 exp210: GCN+PAA+CE+OA-SD + Per-Keypoint Contrastive (PKC)
experiments/exp324b/_codex_review.log:3031:./experiments/exp210/design.md:31:3. **与 OA-SD 共存**: PKC 在 student forward 的 kp_feats 上计算，不涉及 teacher
experiments/exp324b/_codex_review.log:3036:./experiments/exp210/design.md:45:- 如果 PKC 与 OA-SD 冲突: 降 PKC weight 或只对高 visibility keypoint 计算
experiments/exp324b/_codex_review.log:3039:./experiments/exp210/monitor.md:1:# exp210 Small + GCN+PAA+CE+OA-SD + PKC (远程 1-view) 监控
experiments/exp324b/_codex_review.log:3054:./experiments/exp231/design.md:35:- exp191: OA-SD only (63.2/75.4) — 无 BT-PKD baseline
experiments/exp324b/_codex_review.log:3062:./experiments/exp210/claude_review.md:26:- 在 exp206 (GCN+PAA+CE+OA-SD) 基础上只增加 PKC loss，单一变量。
experiments/exp324b/_codex_review.log:3074:./experiments/exp210/claude_review.md:149:### PKC + OA-SD: OK
experiments/exp324b/_codex_review.log:3077:./experiments/exp210/claude_review.md:159:- GCN 参数同时接收 CE + triplet (via pooled feat) + PKC (per-keypoint) + OA-SD (via pooled feat) 的梯度
experiments/exp324b/_codex_review.log:3082:./experiments/exp231/monitor.md:3:配置: Tiny + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01, decay_epoch=60)
experiments/exp324b/_codex_review.log:3114:./experiments/exp228/design.md:4:- exp225 (GSPB+PADPQ K=4): 64.2/74.9 (+1.0 vs OA-SD)
experiments/exp324b/_codex_review.log:3129:./experiments/exp228/monitor.md:1:# exp228 Tiny + GSPB(0.05) + PADPQ K=8 + OA-SD 监控
experiments/exp324b/_codex_review.log:3130:./experiments/exp228/monitor.md:3:配置: Tiny + GCN+PAA+OA-SD+PLBOA+ROA + GSPB(0.05) + PADPQ(K=8)
experiments/exp324b/_codex_review.log:3135:./experiments/exp228/monitor.md:28:**ep30: 54.1%** (vs K=4 exp225 ~53.5, vs OA-SD 50.6 = +3.5)
experiments/exp324b/_codex_review.log:3141:./experiments/exp228/monitor.md:55:**ep70: 62.2%** (vs K=4 62.3, vs OA-SD 61.8 = +0.4)
experiments/exp324b/_codex_review.log:3143:./experiments/exp228/monitor.md:66:**ep80: 62.9/73.7** (vs K=4 62.8/73.5, vs OA-SD 62.0/74.4 = +0.9/-0.7)
experiments/exp324b/_codex_review.log:3144:./experiments/exp228/monitor.md:67:K=8 ≈ K=4 on mAP, both -0.7 R1 vs OA-SD (PADPQ's R1 cost persists)。
experiments/exp324b/_codex_review.log:3145:./experiments/exp228/monitor.md:78:**ep90: 62.8/73.2** (vs K=4 exp225 ~63.3, vs OA-SD 62.8/75.1)
experiments/exp324b/_codex_review.log:3146:./experiments/exp228/monitor.md:80:| Epoch | exp228 mAP/R1 | exp225 K=4 mAP | OA-SD mAP/R1 |
experiments/exp324b/_codex_review.log:3147:./experiments/exp228/monitor.md:86:K=8 开始落后 K=4。mAP 持平 OA-SD (62.8 = 62.8)，R1 -1.9 vs OA-SD (73.2 vs 75.1)。
experiments/exp324b/_codex_review.log:3150:./experiments/exp228/monitor.md:95:| Epoch | exp228 K=8 mAP/R1 | exp225 K=4 mAP/R1 | OA-SD mAP/R1 |
experiments/exp324b/_codex_review.log:3153:./experiments/exp228/monitor.md:123:**ep110: 63.9/73.8** (vs K=4 exp225 ~64.0, vs OA-SD 63.2/75.4)
experiments/exp324b/_codex_review.log:3154:./experiments/exp228/monitor.md:125:| Epoch | exp228 K=8 | exp225 K=4 | delta vs OA-SD |
experiments/exp324b/_codex_review.log:3158:./experiments/exp228/monitor.md:142:**PADPQ 的 R1 损失在 K=4 和 K=8 上都持续存在** (vs OA-SD: -1.1 and -1.1)。
experiments/exp324b/_codex_review.log:3179:./experiments/exp207/monitor.md:472:远程 exp210 (GCN+PAA+CE+OA-SD+PKC Small) 已启动，pkc=3.725。
experiments/exp324b/_codex_review.log:3181:./experiments/exp207/monitor.md:514:已终止 exp210，远程改跑 exp206r (同配置无 PKC) 作为 OA-SD fix 对照。
experiments/exp324b/_codex_review.log:3182:./experiments/exp207/monitor.md:522:确认: OA-SD fix 无问题, PKC weight=0.5 是灾难原因。
experiments/exp324b/_codex_review.log:3188:./experiments/exp207/claude_review.md:133:1. 必须创建 exp207 专用 config 文件 (基于 pose_psg_gcn_paa_plboa_roa.yml + Base backbone + OA-SD + LR 0.0002 + WITH_CP=True)
experiments/exp324b/_codex_review.log:3238:./experiments/exp259/monitor.md:116:**FINAL**: mAP=73.2%, R1=83.4% (vs exp255 73.2/83.3 = 0.0/+0.1). OA-SD=2.0 ≈ baseline at final.
experiments/exp324b/_codex_review.log:3262:./experiments/exp232/claude_review.md:88:- OA-SD distillation 不受 BT-PKD decay 影响，始终以 `POSE_OA_SD_WEIGHT=1.0` 运行。正确
experiments/exp324b/_codex_review.log:3266:./experiments/exp232/monitor.md:3:配置: Small + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01, decay_ep=60), 无 PARALLEL_AUG
experiments/exp324b/_codex_review.log:3276:./experiments/exp324b/_codex_review.log:14:Review two files for a person-ReID experiment exp324b: experiments/exp324b/design.md and scripts/exp324b_train_head.py (it imports helpers from scripts/exp324_dino.py — read that for context). Setup: FROZEN DINOv2-base dense tokens, pose-anchored into 5 body parts (cached), then a LIGHTWEIGHT shared Linear(768->512)+BNNeck+classifier head (~413K params, DINO never back-propped). Loss = ID CrossEntropy on BNNeck feat + batch-hard SOFT-MARGIN triplet on pre-BN global feat (global = masked-mean over visible projected parts), weight 1:1. PK sampler P=16 x K=4 = BS64. Eval = two distmats: global cosine + mutually-visible part-MaxSim (L2-normalized projected parts); mAP/R1 on ALL query and HEAVY-OCC (query pose visibility_binary.sum()<=8). Goal: lift exp324's frozen 1.86 heavy-occ mAP via training the head. Review line-by-line: bugs / runtime errors / shape-device-dtype mismatches / edge cases (zero-visible-part images, NaN); train/test asymmetry (head is supervised via GLOBAL masked-mean feature but tested via per-part normalized part-MaxSim — is this coupling sound? should there be a part-level loss?); triplet correctness (pos/neg masks, hardest mining, soft-margin); PK sampler correctness; eval correctness (same-pid-same-cam exclusion, heavy-occ mask slicing, query/gallery pose split mapping); single-variable isolation vs exp324; no LR scheduler concern. Use web search to check novelty/precedent of 'frozen foundation-model features + lightweight head + pose-part matching for occluded person ReID'. Output: Verdict (approve / needs-attention) + findings with severity (Critical/High/Medium/Low).
experiments/exp324b/_codex_review.log:3296:./experiments/exp324b/_codex_review.log:1048:/bin/zsh -lc 'rg -n "exp324b|exp324|heavy_mask|part_maxsim|PKSampler|batch_hard_triplet|CACHE_QG|_cache_train|vis.*sum|visibility_binary" experiments scripts -S' in /Users/abdslm/Desktop/SOLIDER-REID
experiments/exp324b/_codex_review.log:3299:./experiments/exp324b/_codex_review.log:1181:experiments/exp324b/_codex_review.log:14:Review two files for a person-ReID experiment exp324b: experiments/exp324b/design.md and scripts/exp324b_train_head.py (it imports helpers from scripts/exp324_dino.py — read that for context). Setup: FROZEN DINOv2-base dense tokens, pose-anchored into 5 body parts (cached), then a LIGHTWEIGHT shared Linear(768->512)+BNNeck+classifier head (~413K params, DINO never back-propped). Loss = ID CrossEntropy on BNNeck feat + batch-hard SOFT-MARGIN triplet on pre-BN global feat (global = masked-mean over visible projected parts), weight 1:1. PK sampler P=16 x K=4 = BS64. Eval = two distmats: global cosine + mutually-visible part-MaxSim (L2-normalized projected parts); mAP/R1 on ALL query and HEAVY-OCC (query pose visibility_binary.sum()<=8). Goal: lift exp324's frozen 1.86 heavy-occ mAP via training the head. Review line-by-line: bugs / runtime errors / shape-device-dtype mismatches / edge cases (zero-visible-part images, NaN); train/test asymmetry (head is supervised via GLOBAL masked-mean feature but tested via per-part normalized part-MaxSim — is this coupling sound? should there be a part-level loss?); triplet correctness (pos/neg masks, hardest mining, soft-margin); PK sampler correctness; eval correctness (same-pid-same-cam exclusion, heavy-occ mask slicing, query/gallery pose split mapping); single-variable isolation vs exp324; no LR scheduler concern. Use web search to check novelty/precedent of 'frozen foundation-model features + lightweight head + pose-part matching for occluded person ReID'. Output: Verdict (approve / needs-attention) + findings with severity (Critical/High/Medium/Low).
experiments/exp324b/_codex_review.log:3326:./experiments/paper_materials/story.md:201:| Small | GCN+PAA+OA-SD+PKC | 70.6% | 81.8% | **72.4%** | **83.1%** |
experiments/exp324b/_codex_review.log:3351:web search: arXiv frozen foundation model features lightweight head person re-identification DINOv2
experiments/exp324b/_codex_review.log:3353:web search: DINOv2 features person re-identification frozen feature extraction arxiv 2024
experiments/exp324b/_codex_review.log:3355:web search: "DINOv2" "person re-identification"
experiments/exp324b/_codex_review.log:3363:web search: "occluded person re-identification" "DINOv2"
experiments/exp324b/_codex_review.log:3365:web search: DINOv2 paper frozen features dense correspondence arxiv
experiments/exp324b/_codex_review.log:3367:web search: DINOv2 Learning Robust Visual Features without Supervision arxiv
experiments/exp324b/_codex_review.log:3379:- **High**: Train/test objective mismatch for the main part-MaxSim metric. Training supervises only masked-mean global features: CE on BN global and triplet on pre-BN global [scripts/exp324b_train_head.py:176](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:176), [scripts/exp324b_train_head.py:427](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:427). Evaluation’s key metric uses individually L2-normalized projected parts and same-index mutually-visible part matching [scripts/exp324b_train_head.py:182](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:182), [scripts/exp324b_train_head.py:305](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:305). Because only the average part vector is optimized, individual part vectors can remain weakly aligned. Either add a part-level ID/triplet/contrastive auxiliary, or make global cosine the primary metric for exp324b and explicitly mark part-MaxSim as diagnostic.
experiments/exp324b/_codex_review.log:3381:- **Medium**: `design.md` and code disagree on the training head/loss. The design says “训练时每部位接 ID 分类头（CE）+ part/global triplet” [experiments/exp324b/design.md:19](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/design.md:19), but code has one global BNNeck/classifier and no per-part loss. This is not a runtime bug, but it affects the experiment’s evidence chain.
experiments/exp324b/_codex_review.log:3383:- **Medium**: Query/gallery cache isolation is not exactly what the design claims. `CACHE_QG` is declared as reuse of exp324 cache but is unused [scripts/exp324b_train_head.py:61](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:61); query/gallery parts are re-extracted into exp324b cache [scripts/exp324b_train_head.py:462](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:462). This is probably numerically close, but it weakens single-variable comparison to exp324’s cached float16 patch pipeline.
experiments/exp324b/_codex_review.log:3385:- **Medium**: PK sampler is acceptable for full Occluded-Duke, but not robust for smoke limits. If `--limit_train` gives fewer than 64 images, `num_batches=0` and dry-run can fail after repeated `StopIteration` [scripts/exp324b_train_head.py:245](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:245), [scripts/exp324b_train_head.py:441](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:441). If identities `< P`, batch size silently drops below 64 [scripts/exp324b_train_head.py:254](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:254).
experiments/exp324b/_codex_review.log:3387:- **Low**: Zero-visible images do not NaN: masked mean clamps denominator [scripts/exp324b_train_head.py:166](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:166), `F.normalize` has eps, and no-common part pairs become distance 2.0 in exp324 helper [scripts/exp324_dino.py:280](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324_dino.py:280). But zero-visible train samples become identical zero global features, so log their count.
experiments/exp324b/_codex_review.log:3389:- **Low**: Triplet implementation matches the repo’s soft-margin semantics: `SoftMarginLoss(dist_an - dist_ap, +1)` is correct [scripts/exp324b_train_head.py:223](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:223), consistent with [loss/triplet_loss.py](/Users/abdslm/Desktop/SOLIDER-REID/loss/triplet_loss.py:120). Pos/neg masks are also correct for PK batches.
experiments/exp324b/_codex_review.log:3391:- **Low**: Eval same-pid-same-cam exclusion is correct via reused Market metric [scripts/exp324_dino.py:56](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324_dino.py:56). Heavy-occ slicing uses query-only mask and correct pose split [scripts/exp324b_train_head.py:334](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:334).
experiments/exp324b/_codex_review.log:3395:Close precedents exist: PVPM already does pose-guided visible part matching for occluded ReID with learned visibility; KPR uses keypoints to disambiguate occluded/multi-person ReID; PFD explicitly matches visible body parts and adds pose-guided part loss. Foundation-model ReID adaptation is also active: CLIP-ReID and PCL-based CLIP fine-tuning adapt pretrained vision-language models for ReID, and SAGA-ReID specifically argues global CLIP aggregation is fragile under occlusion and moves toward patch/local aggregation. DINOv2 itself is positioned as general-purpose frozen visual features.
experiments/exp324b/_codex_review.log:3397:So the exact combination “frozen DINOv2 dense tokens + lightweight trained projection head + pose-anchored mutually-visible part matching” still looks plausibly distinct, but not novel as “pose visible-part matching” or “foundation-model ReID adaptation” alone. The strongest claim would be about a low-trainable-parameter frozen-DINO dense correspondence route for occluded ReID, with careful controls against PVPM/KPR/PFD/CLIP-local aggregation.
experiments/exp324b/_codex_review.log:3399:Sources: PVPM arXiv, KPR arXiv, PFD arXiv, CLIP-ReID arXiv, SAGA-ReID arXiv, DINOv2 arXiv.
experiments/exp324b/_codex_review.log:3408:- **High**: Train/test objective mismatch for the main part-MaxSim metric. Training supervises only masked-mean global features: CE on BN global and triplet on pre-BN global [scripts/exp324b_train_head.py:176](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:176), [scripts/exp324b_train_head.py:427](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:427). Evaluation’s key metric uses individually L2-normalized projected parts and same-index mutually-visible part matching [scripts/exp324b_train_head.py:182](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:182), [scripts/exp324b_train_head.py:305](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:305). Because only the average part vector is optimized, individual part vectors can remain weakly aligned. Either add a part-level ID/triplet/contrastive auxiliary, or make global cosine the primary metric for exp324b and explicitly mark part-MaxSim as diagnostic.
experiments/exp324b/_codex_review.log:3410:- **Medium**: `design.md` and code disagree on the training head/loss. The design says “训练时每部位接 ID 分类头（CE）+ part/global triplet” [experiments/exp324b/design.md:19](/Users/abdslm/Desktop/SOLIDER-REID/experiments/exp324b/design.md:19), but code has one global BNNeck/classifier and no per-part loss. This is not a runtime bug, but it affects the experiment’s evidence chain.
experiments/exp324b/_codex_review.log:3412:- **Medium**: Query/gallery cache isolation is not exactly what the design claims. `CACHE_QG` is declared as reuse of exp324 cache but is unused [scripts/exp324b_train_head.py:61](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:61); query/gallery parts are re-extracted into exp324b cache [scripts/exp324b_train_head.py:462](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:462). This is probably numerically close, but it weakens single-variable comparison to exp324’s cached float16 patch pipeline.
experiments/exp324b/_codex_review.log:3414:- **Medium**: PK sampler is acceptable for full Occluded-Duke, but not robust for smoke limits. If `--limit_train` gives fewer than 64 images, `num_batches=0` and dry-run can fail after repeated `StopIteration` [scripts/exp324b_train_head.py:245](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:245), [scripts/exp324b_train_head.py:441](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:441). If identities `< P`, batch size silently drops below 64 [scripts/exp324b_train_head.py:254](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:254).
experiments/exp324b/_codex_review.log:3416:- **Low**: Zero-visible images do not NaN: masked mean clamps denominator [scripts/exp324b_train_head.py:166](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:166), `F.normalize` has eps, and no-common part pairs become distance 2.0 in exp324 helper [scripts/exp324_dino.py:280](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324_dino.py:280). But zero-visible train samples become identical zero global features, so log their count.
experiments/exp324b/_codex_review.log:3418:- **Low**: Triplet implementation matches the repo’s soft-margin semantics: `SoftMarginLoss(dist_an - dist_ap, +1)` is correct [scripts/exp324b_train_head.py:223](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:223), consistent with [loss/triplet_loss.py](/Users/abdslm/Desktop/SOLIDER-REID/loss/triplet_loss.py:120). Pos/neg masks are also correct for PK batches.
experiments/exp324b/_codex_review.log:3420:- **Low**: Eval same-pid-same-cam exclusion is correct via reused Market metric [scripts/exp324_dino.py:56](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324_dino.py:56). Heavy-occ slicing uses query-only mask and correct pose split [scripts/exp324b_train_head.py:334](/Users/abdslm/Desktop/SOLIDER-REID/scripts/exp324b_train_head.py:334).
experiments/exp324b/_codex_review.log:3424:Close precedents exist: PVPM already does pose-guided visible part matching for occluded ReID with learned visibility; KPR uses keypoints to disambiguate occluded/multi-person ReID; PFD explicitly matches visible body parts and adds pose-guided part loss. Foundation-model ReID adaptation is also active: CLIP-ReID and PCL-based CLIP fine-tuning adapt pretrained vision-language models for ReID, and SAGA-ReID specifically argues global CLIP aggregation is fragile under occlusion and moves toward patch/local aggregation. DINOv2 itself is positioned as general-purpose frozen visual features.
experiments/exp324b/_codex_review.log:3426:So the exact combination “frozen DINOv2 dense tokens + lightweight trained projection head + pose-anchored mutually-visible part matching” still looks plausibly distinct, but not novel as “pose visible-part matching” or “foundation-model ReID adaptation” alone. The strongest claim would be about a low-trainable-parameter frozen-DINO dense correspondence route for occluded ReID, with careful controls against PVPM/KPR/PFD/CLIP-local aggregation.
experiments/exp324b/_codex_review.log:3428:Sources: PVPM arXiv, KPR arXiv, PFD arXiv, CLIP-ReID arXiv, SAGA-ReID arXiv, DINOv2 arXiv.

web search: 
web search: "PersonViT" "Self-supervised Vision Transformer" person re-identification arxiv
web search: 
web search: "SAIP" "Scale-Aware Image Pre-training" human-centric perception arxiv 2025
web search: 
web search: "Scale-Aware Image Pre-training" "human-centric perception"
web search: 
web search: https://arxiv.org/abs/2503.08201
web search: 
web search: https://arxiv.org/abs/2508.10104
web search: 
web search: https://arxiv.org/abs/2304.07193
web search: 
web search: "cross-view" "masked image modeling" "person re-identification"
web search: 
web search: "SSL-ReID" "Unsupervised Person Re-Identification" "self-supervised" arxiv
web search: 
web search: "LUPerson" "self-supervised" "person re-identification"
web search: 
web search: "SAGA-ReID" "occluded" "CLIP" "person re-identification" arXiv
web search: 
web search: "CLIP-ReID" "Exploiting Vision-Language Model" "Person Re-Identification" arxiv
web search: 
web search: "CLIP-ReID" "person re-identification"
web search: 
web search: "V-JEPA" "self-supervised" video prediction arxiv 2024
web search: 
web search: "Person Re-Identification" "Masked Autoencoder" "CVPR 2024" "Occluded-Duke"
web search: 
web search: https://github.com/hustvl/PersonViT
codex
我又确认了一个重要外部边界：2025 的 SAIP 已经把“cross-scale consistency/reconstruction/search”用于 human-centric pretraining，2026 的 CLIP/SAGA 类工作也在强调遮挡下 global token 脆弱、需要局部结构聚合。所以可讲的新 pretext 不能只是“多尺度 + 局部聚合”，必须落在“身份证据不完整时的可见 support 合成/预测”这个更具体问题上。
exec
/bin/zsh -lc "sed -n '1,180p' experiments/exp109/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 实验 exp109: Oracle Support Bank 上界诊断

## 动机

- `exp107/108` 已基本否定 retrieval-time `ambiguity/confuser penalty` 主线
- 但 `SGCFR` 明确证明：**跨图 support recovery** 确实能带来大增益
- 训练端 recovery 系列 (`exp091/092/101/105/106`) 之所以没有成功，核心问题很可能不是“recover 这个想法错了”，而是：
  **batch 内没有足够稳定的 same-ID support**
- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**

## 核心假设

1. 若单图表征真的受限于“support 不完整”，那么用同 ID 多图构造 oracle prototype 后，matching 应明显优于原始 `cvk_hybrid`
2. 收益应主要集中在：
   - `multi`
   - `clean multi`
   - 低可见性 query
3. 若 oracle 上界都很小，则说明 training-time support-complete distillation 很难成为主线，应立即止损

## 技术方案

### 1. 特征提取
- 基于 `exp030a` checkpoint
- 提取：
  - `global_feat`
  - `kp_feats`
  - `kp_weights`

### 2. Oracle support bank 构造
- 在 query+gallery 范围内，用 GT `pid` 构造每个 identity、每个 keypoint 的 prototype
- prototype 只由高可见性 keypoint 聚合
- 对当前样本做 leave-one-out，避免把自身直接抄回去

### 3. 恢复方式
- 对低可见 keypoint，用对应 ID 的 prototype 替换其 `kp_feat`
- 比较两种版本：
  - `oracle_feat_only`: 只替换 feature，保留原始 `kp_weight`
  - `oracle_feat_weight`: 替换 feature，并把该 keypoint 的 weight 提升到 prototype confidence

### 4. 评估
- 主基线: `exp030a cvk_hybrid`
- 比较：
  - `base_cvk_hybrid`
  - `oracle_feat_only_cvk`
  - `oracle_feat_weight_cvk`

## 对照组

- `exp030a cvk_hybrid`

## 预期结果

- 如果 oracle 明显为正：
  - 说明 “support-complete teacher” 这条训练主线有真实 headroom
  - 可继续做最小训练版 prototype distillation
- 如果 oracle 只有极小增益或直接负面：
  - 说明 SGCFR 的收益更接近 retrieval-time 邻域特性，而不是可蒸馏到单图编码器的潜在表征
  - support bank 训练线应止损

## 风险与失败解释

1. GT same-ID prototype 也不提升：
   - 说明当前 `kp_feats` 本身不具备可恢复性
2. 只替换 feature 有用，但替换 weight 无用：
   - 说明关键在 latent feature completion，而不是 visibility correction
3. 只在极少数 query 上提升：
   - 说明该方向可能只适合作为 case-specific trick，不足以支撑主线

exec
/bin/zsh -lc "sed -n '1,140p' experiments/exp109/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp109 Oracle Support Bank 监控

## 实验信息
- 方法: Oracle Support Bank 上界诊断
- 类型: retrieval-time oracle 分析，不计入正式方法结果
- 主基线: `exp030a cvk_hybrid`
- 核心变量: GT same-ID per-keypoint prototype 是否提供明显 headroom

## 启动记录

### [2026-03-19 15:30] 实验启动
- 来自上一阶段的直接教训：
  - `DACHM/DACCM` 说明 confuser penalty 不是对的路
  - `SGCFR` 说明 cross-image support recovery 确有价值
- 当前执行内容：
  1. 新增 `scripts/eval_oracle_support_bank.py`
  2. 在 `exp030a` 上提取 `global_feat / kp_feats / kp_weights`
  3. 构造 GT same-ID oracle prototype，比较 `base / oracle_feat_only / oracle_feat_weight`
  4. 若 headroom 明显，再进入训练版 support-complete distillation 设计

### [2026-03-19 16:08] 实验完成
- 结果文件: `log/occluded_duke/exp109_oracle_support_bank_exp030a/summary.json`
- oracle 恢复统计:
  - `samples_recovered = 3385`
  - `keypoints_recovered = 10194`
  - `avg_support_count = 82.33`
- 整体结果:
  - `base_cvk_hybrid = 61.88% mAP / 73.26% R1`
  - `oracle_feat_only_cvk = 66.15% / 77.87%`
  - `oracle_feat_weight_cvk = 70.40% / 81.36%`
- 关键子集:
  - `clean multi`: `65.06 / 76.26` → `68.04 / 79.71` → `71.33 / 82.73`
  - `duplicate-suspect multi`: `62.31 / 76.96` → `65.21 / 78.73` → `68.34 / 81.27`
  - `target_vis<=8` (26 queries): `29.42 / 26.92` → `78.26 / 84.62` → `91.71 / 100.00`
  - `target_vis<=5` (7 queries): `16.85 / 14.29` → `78.43 / 85.71` → `86.28 / 100.00`
- 当前判断: 强阳性上界，继续
- 原因:
  1. 即使只替换低可见 keypoint feature、保留原始 weight，也有 `+4.27 mAP / +4.62 R1`
  2. 说明“support-complete latent representation”不是空想，而是存在巨大 headroom
  3. 下一步应做训练版 prototype distillation，而不是继续 retrieval-time penalty

exec
/bin/zsh -lc "sed -n '1,160p' experiments/exp188/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 实验 exp188: Occlusion-Asymmetric Self-Distillation (OA-SD)

## 动机
- 范式级创新方向：把 occluded ReID 重新定义为"学习遮挡不变表示"
- Teacher 看 clean image (pre-PLBOA) → clean structural tokens
- Student 看 occluded image (post-PLBOA) → degraded tokens
- Distillation: student tokens 逼近 teacher tokens（cosine distance）
- **使用 EMA teacher**：teacher 是 student 的 exponential moving average (decay=0.999)
- Teacher 用 eval mode 做 forward (no dropout/droppath)，每个 optimizer step 后 EMA 更新
- 与 PersonMAE 的区别：distill identity-level tokens 而非 reconstruct pixels
- 与我们失败的 exp048/091/092 的区别：软目标 + 同一图像的 clean/occluded 版本

## 核心假设
通过让 student (看到遮挡图像) 的 structural tokens 逼近 teacher (看到完整图像) 的 tokens，
模型学会产生"遮挡不变"的 body-part 表示。

## 技术方案

### 数据流
```
同一张图像:
  ├─ clean_img (pre-PLBOA) ─→ EMA Teacher forward (eval, no_grad) → clean tokens
  └─ occluded_img (post-PLBOA) ─→ Student forward (train) → degraded tokens
                                      ↓
                               Standard loss (CE/SupCon + triplet)
                                      +
                               Distillation loss (cosine distance to EMA teacher)
                                      ↓
                               After optimizer.step():
                               EMA update: teacher = 0.999 * teacher + 0.001 * student
```

### 修改文件
1. `config/defaults.py`: POSE_OA_SD, POSE_OA_SD_WEIGHT
2. `datasets/pose_dataset.py`: OA-SD mode 保存 pre-PLBOA clean image
3. `datasets/make_dataloader.py`: 设置 _oa_sd_mode flag
4. `processor/processor.py`: teacher forward (no_grad) + distillation loss

### Distillation Loss
对 per-token features (global + 6 structural tokens):
```python
d_loss = (1 - cosine_sim(student_token, teacher_token.detach())).mean()
```
逐 token 计算 cosine distance，平均后作为 distillation loss。

### 显存
- 2x model weights (student + EMA teacher copy)
- 2x forward pass (student with graph + teacher no graph)
- 预估 ~18-20GB (student ~10GB + teacher ~3GB + model copies ~4GB)
- **需要在本地 3090 24GB 上跑**

### EMA 参数
- decay = 0.999 (标准 DINO/BYOL 值)
- teacher 初始化 = student 的 deepcopy
- teacher 在 eval mode (无 dropout/droppath)
- 每个 optimizer step 后更新

## 对照组
- exp176 (SupCon T=0.05, 无 distillation): 64.1/75.5
- 消融变量: 仅增加 OA-SD distillation loss

## 预期
- 如果成立: 遮挡不变表示 → R1 提升（更鲁棒的 top-1 matching）
- 如果失败: distillation 信号可能与 SupCon/CE 梯度冲突

exec
/bin/zsh -lc "sed -n '1,180p' experiments/exp324b/_codex_review.log" in /Users/abdslm/Desktop/SOLIDER-REID
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
session id: 019ecd47-c052-7c11-806b-f9429bbc2ac7
--------
user
Review two files for a person-ReID experiment exp324b: experiments/exp324b/design.md and scripts/exp324b_train_head.py (it imports helpers from scripts/exp324_dino.py — read that for context). Setup: FROZEN DINOv2-base dense tokens, pose-anchored into 5 body parts (cached), then a LIGHTWEIGHT shared Linear(768->512)+BNNeck+classifier head (~413K params, DINO never back-propped). Loss = ID CrossEntropy on BNNeck feat + batch-hard SOFT-MARGIN triplet on pre-BN global feat (global = masked-mean over visible projected parts), weight 1:1. PK sampler P=16 x K=4 = BS64. Eval = two distmats: global cosine + mutually-visible part-MaxSim (L2-normalized projected parts); mAP/R1 on ALL query and HEAVY-OCC (query pose visibility_binary.sum()<=8). Goal: lift exp324's frozen 1.86 heavy-occ mAP via training the head. Review line-by-line: bugs / runtime errors / shape-device-dtype mismatches / edge cases (zero-visible-part images, NaN); train/test asymmetry (head is supervised via GLOBAL masked-mean feature but tested via per-part normalized part-MaxSim — is this coupling sound? should there be a part-level loss?); triplet correctness (pos/neg masks, hardest mining, soft-margin); PK sampler correctness; eval correctness (same-pid-same-cam exclusion, heavy-occ mask slicing, query/gallery pose split mapping); single-variable isolation vs exp324; no LR scheduler concern. Use web search to check novelty/precedent of 'frozen foundation-model features + lightweight head + pose-part matching for occluded person ReID'. Output: Verdict (approve / needs-attention) + findings with severity (Critical/High/Medium/Low).
codex
I’ll read the experiment design, training script, and DINO helper first, then cross-check the review points against the actual code paths. After that I’ll do a focused literature/code search for close precedents before giving a severity-ranked verdict.
exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp324b/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 实验 exp324b: 冻结 DINOv2 + 轻量投影头 + 姿态部位匹配

> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
> **性质**：**训练实验**。开训前必须过 Claude broad review + Codex review（hook 阻断）。
> **机器**：lab-3090-d（DINOv2-base 已下、exp324 已缓存部位特征、RTX 3090 idle）。

## 动机

- exp324（frozen，无训练）已证：用姿态把 DINOv2 dense token 锚定到 5 个身体部位、跨图只比双方可见部位（part-MaxSim），在 Occluded-Duke 重遮挡 query 上 mAP 1.86 vs 整图 0.55（×3.4），且**均匀网格对照只 0.67（几乎不涨）→ 涨点几乎全来自"姿态锚定"机制本身**（单变量隔离干净）。
- 但绝对分低（1.86 mAP），符合 DINO 零样本 ReID 文献（0.3-4.7）。training-free 不可用。
- 与 exp323 对照：同样 frozen + 同样 pose，**MLLM-reasoning 那条无信号、DINO dense-correspondence 这条有信号**——差别在"特征表示端"而非"大模型推理端"。下一步应在特征端使劲：用最小训练把冻结特征投到 ReID 判别空间。

## 核心假设

冻结 DINOv2-base、**仅训一个轻量 per-part 投影头**（先线性，必要时小 MLP），用 ID 分类 + triplet loss 训练、用 mutually-visible part-MaxSim 匹配，能把重遮挡 mAP 从 1.86 **大幅拉高**（先看能否进入"几十分"量级证明冻结特征有救），同时保留"姿态锚定 + 只比可见部位"的机制优势。

## 技术方案

- **数据流**：图 → 冻结 DINOv2-base dense tokens（已缓存）→ 姿态锚定 5 部位池化 + per-part visibility（exp324 已实现）→ **轻量投影头**（per-part 线性 768→D，或共享线性 + 可选 BN/L2-norm）→ 训练时每部位接 ID 分类头（CE）+ part/global triplet；测试用 mutually-visible part-MaxSim 出 distmat。
- **冻结边界**：DINO **不反传**（不解冻 backbone，避免大成本 + 防过拟合 15K 数据）；只训投影头 + 分类器。**在缓存特征上训** → 每 step 无 backbone forward，飞快。
- **修改文件**：新增独立训练脚本 `scripts/exp324b_train_head.py`（不碰 model/ 核心、不碰 train.py）：读 exp324 缓存的 train/query/gallery 部位特征（train 部位特征需补抽一次）→ 训头 → 评测。loss 复用项目思路（ID CE label-smooth off + triplet no-margin），可直接 import `loss/` 里的 triplet 或自带最简实现。
- **关键超参（先最小）**：投影头 = 线性（per-part 独立 vs 共享，先共享）；投影维 D=512；ID loss + triplet（权重 1:1）；优化器 SGD/Adam，LR 小、epoch 少（缓存特征训得快，先 30-60 epoch 看趋势）；**BS=64（项目硬约束，不可改）**；mutually-visible part-MaxSim 同 exp324。
- **评测**：`test.py` 风格（永不用 train.py 评估）。mAP/R1，全量 + 重遮挡子集（vis≤8）。

## 预期结果

- 假设成立：重遮挡 mAP 从 1.86 显著上升（若进入几十分量级 → 冻结特征有救、方向成立 → 再加容量/解冻部分 DINO 冲 SOTA）。
- 失败最可能：冻结 DINO 特征 ReID 判别力有天花板，轻量头拉不动（停在个位数）→ 则下一步换 LoRA 解冻部分 DINO，或换更强 DINO（large/v3）；若仍不动则这条线天花板低、止损。

## 对照组

- **baseline**：exp324 frozen 无训练（重遮挡 1.86 / 全量 3.21）。
- **消融**：投影头 on 姿态部位 vs on 均匀网格（训练后再证姿态锚定贡献仍在）；线性头 vs 小 MLP（容量敏感性）；ID-only vs ID+triplet。
- **对标**：最终对 KPR（Occluded-Duke）；与项目 Swin 主线（75 mAP）做"frontier-independent 新表征"定位（不必正面超，讲机制新颖 + 参数/范式角度）。

## 协议待办（开训前硬性）

1. [x] exp324 frozen 首验通过 kill-switch
2. [ ] 写训练脚本 `scripts/exp324b_train_head.py`（+ 补抽 train 部位特征）
3. [ ] **Claude broad review**（Opus 子代理，全范围）→ `claude_review.md`
4. [ ] **Codex review**（`codex --search exec`，查 frozen-FM + light-head ReID 先例 / 训练正确性）→ `codex_review.md`
5. [ ] 双审查 approve 后才开训（hook 阻断）

exec
/bin/zsh -lc "sed -n '1,360p' scripts/exp324b_train_head.py" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,360p' scripts/exp324_dino.py" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
#!/usr/bin/env python3
"""exp324: Training-free DINOv2 dense-correspondence + pose-anchored part matching
for occluded person ReID on Occluded-Duke.

Hypothesis: frozen DINOv2-base dense patch features, anchored into 5 body parts by
pose keypoints, and matched cross-image only over parts that are visible in BOTH
images (part-MaxSim), give a RELATIVE gain over a holistic (CLS / mean-pool) baseline
on heavily-occluded queries. A uniform-grid variant (5 horizontal bands, no pose)
isolates the contribution of pose anchoring.

Pure inference / eval. No training. Does not touch model/ core code.

Pipeline:
  1. Extract DINOv2 patch tokens for all query+gallery imgs (resize 224W x 448H,
     patch=14 -> 16 cols x 32 rows = 512 patch tokens + 1 CLS). Cache to disk.
  2. Build three representations:
       (a) holistic   : CLS token (also report mean-pool patches)
       (b) part_pose  : 5 pose-anchored part vectors + per-part visibility
       (c) part_grid  : 5 horizontal-band part vectors (uniform, no pose)
  3. Score:
       (a) cosine distance on holistic vector
       (b)/(c) part-MaxSim: only common-visible parts, per-part cosine, averaged
  4. eval_func (utils.metrics) -> mAP/R1, two slices: all-query, heavy-occ (vis<=8).

Coordinate scaling: keypoints are already in ORIGINAL JPG PIXEL space (verified
exp323). Map (x,y) -> grid (col = x/W * GRID_W, row = y/H * GRID_H). Pool patch
tokens within Chebyshev radius POOL_RADIUS of the mapped cell(s).
"""
import os, sys, glob, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = '/root/work/SOLIDER-REID'
sys.path.insert(0, ROOT)


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Market-1501 metric. Verbatim copy of utils.metrics.eval_func (which we
    cannot import directly: utils/metrics.py top-level imports the model package,
    requiring mmcv/mmengine not present in this system python). Pure numpy."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP

DATA = os.path.join(ROOT, 'data/occluded_duke')
QUERY_DIR = os.path.join(DATA, 'query')
GALLERY_DIR = os.path.join(DATA, 'bounding_box_test')
POSE_DIR = os.path.join(DATA, 'pose_data')
CACHE_DIR = os.path.join(ROOT, 'experiments/exp324/_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ---- DINOv2 input / grid geometry ----
IN_W, IN_H = 224, 448          # resize target (W,H), keeps ~1:2 person aspect
PATCH = 14
GRID_W, GRID_H = IN_W // PATCH, IN_H // PATCH   # 16 cols, 32 rows
HIDDEN = 768

# ---- COCO-17 -> 5 parts ----
PART_GROUPS = {
    'head':  [0, 1, 2, 3, 4],
    'torso': [5, 6, 11, 12],
    'arms':  [7, 8, 9, 10],
    'legs':  [13, 14],
    'feet':  [15, 16],
}
PART_NAMES = list(PART_GROUPS.keys())  # fixed order
NPARTS = len(PART_NAMES)
POOL_RADIUS = 1                # Chebyshev radius around each kp cell (3x3 window)
HEAVY_OCC_THR = 8             # query is "heavy-occluded" if visibility_binary.sum() <= 8

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------
# data listing
# ----------------------------------------------------------------------------
def list_imgs(d):
    return sorted([f for f in os.listdir(d) if f.endswith('.jpg')])


codex
本地 exp109 给出的上界很强：GT same-ID per-keypoint support bank 从 61.88/73.26 提到 70.40/81.36，低可见 query 上几乎是数量级提升。这说明“support-complete 表征”不能判死；问题是已有训练端实现没有足够稳定的 teacher/support，而不是目标没有 headroom。
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
/bin/zsh -lc "sed -n '1,220p' experiments/smpl_anchor_vireid/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp359_lm_reid/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 实验 SMPL-Anchor VI-ReID (v0 preliminary) — SMPL 几何当模态无关锚

> 状态: v0 草案(远程宕机期间写, 待 kill-switch + baseline 复现后refine)。文档先行铁律。

## 动机
- VI-ReID(可见光-红外行人ReID)核心 confound = **模态 gap**: RGB 有颜色/纹理, IR 有热强度, 外观差异巨大。
- 现有去模态-gap 手段: 共享-特有特征解耦、频率域、中间模态(灰图/X-modality/MTRL)、**轮廓/体形(MSO2021/GSMEN2023/Contour-MMN2024 已占满)**。
- ★洞察: **SMPL 3D 几何(姿态参数 + 投影关节 + mesh)是从人体结构恢复的, 与外观(颜色/热强度)无关 → 天然模态不变**。比 2D 轮廓更"光谱无关"(纯几何, RGB/IR 投影一致, 不受 IR 边缘检测退化影响)。
- ★空白: 文献里没人用 **SMPL 3D 几何**当 VI-ReID 模态锚(轮廓/体形有人做, 3D mesh 几何是空白)。团队有 SMPL 基建(ROMP/SMPLer-X)。
- 区别于团队 exp333(SMPL-β 随机判负): 这里用**几何(姿态/关节)做结构对齐锚**, 不是 β 当身份特征。绕开 exp333 死因。

## 核心假设
SMPL 3D 几何当**训练期特权(LUPI)模态共享锚**, 把同人的 RGB/IR 特征拉到同一几何上 → 缩小模态 gap → VI-ReID 涨点; 且增益来自几何对齐(非通用正则, 由 shuffle 对照证伪)。

## 技术方案
1. **离线**: 对 RGB 和 IR 行人 crop 各跑 SMPL fit(ROMP/SMPLer-X) → 几何(姿态参数 / 投影 2D 关节热图 / mesh 顶点)。缓存成 .pth。
2. **训练**: 共享 backbone(Swin-Small, 团队资产)双流(RGB/IR)→ baseline ID+triplet loss + **SMPL 几何对齐 loss**: 同人 RGB/IR 特征对齐到共享几何(privileged 监督)。
3. **测试**: 只用 RGB/IR encoder, **丢 SMPL** → 单 embedding 零外部(LUPI 干净口径)。
4. **baseline**: mangye16/Cross-Modal-Re-ID-baseline 的 CAJ(~67 mAP) → 换 Swin backbone。

## ★ kill-switch (step 0, 远程一回来先跑)
**低对比 IR 图上 SMPL 能不能 fit 准?** 取 RegDB / SYSU-MM01 的 IR crop 样本(~200), 跑 SMPL fit, 量: valid rate / 2D 关节 reprojection error / 人工视觉检查 N 张。
- fit 是 garbage(valid<60% 或 reproj 大) → **SMPL-anchor 死**, 转 Swin-VI 机制(领域无 Swin 赢家那条)。
- fit 尚可 → 进 design v1 + 训练。

## ★★ kill-switch 结果 (2026-06-22): SMPL 锚 **死**
torchvision keypointrcnn(COCO 2D pose)在 RegDB 上:
- RGB-visible: 检测率@0.5=85% / @0.7=69%, kp_conf=0.74(正常)
- **IR-thermal: 检测率@0.5=12% / @0.7=5%, kp_conf=-0.10(几乎全失效)**
- IR/RGB 检测率比 0.14, 置信比 -0.13。

**判定: 2D pose 在热成像上完全 OOD(所有 pose/SMPL 工具都 RGB 训), IR 提不出人体几何。SMPL 几何锚需两模态都提几何 → 死。** 加上 RegDB RGB/IR 非同时拍(几何不对应), 双重证死。
**kill-switch 价值兑现: 5min GPU 在实现前拦住, 省了 Swin 改造+几何对齐几天工。**

→ **转 fallback: Swin-VI 机制**。VI-ReID 至今 ResNet50 主导、无 Swin/CLIP 赢家(纯 ViT PMT 才 67.5 R1), 团队是 Swin/SOLIDER 专家。CAJ ResNet baseline 已搭好(POOL 76.80/69.14)→ 换 Swin-Small(SOLIDER pretrain)→ 强 backbone 在 VI-ReID 帮不帮? 帮=机制论文; 不帮=为什么不帮(模态 gap 吞掉 backbone 红利)也是机制问题。先做这个经验首验。

## 对照组 / 消融(扛"通用正则"质疑)
- baseline CAJ(双流, 无 SMPL)。
- +SMPL 几何对齐(完整)。
- ★**shuffle SMPL 几何**(跨人乱配几何)→ 增益应塌(证明是几何对齐, 非通用正则)。
- 单变量: 只加 SMPL 对齐 loss, 其余不动。

## 预期结果
- 过 CAJ baseline(~67 mAP), 冲及格线 ~73-75 mAP(SYSU all-search single-shot)。
- 天花板参考 IDKL 79.85。
- 失败最可能原因: (a) IR SMPL fit 太差(kill-switch 拦); (b) 几何对齐增益是通用正则(shuffle 对照拦)。

## 数据 / 设施
- RegDB(今天可下, github githubXin89/RegDB-dataset Drive) 先原型; SYSU-MM01(邮件 wuanc@mail.sysu.edu.cn)+ LLCM(zhangyk@stu.xmu.edu.cn)申请中。
- SMPL infra 在 lab-3090(ROMP/SMPLer-X)。
- venue: ICME/ICPR/Neurocomputing/PR Letters → 冲 ACCV/BMVC。

## 待办(远程回来后)
1. kill-switch: IR SMPL fit 可行性。
2. 复现 CAJ baseline(Swin backbone)。
3. design v1: 几何对齐 loss 的具体形式(关节热图对齐 / mesh 顶点对齐 / 姿态参数一致)。
4. 双审(Claude + Codex)→ 训练。

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
/bin/zsh -lc "sed -n '1,220p' experiments/smpl_anchor_vireid/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# Monitor — CAJ RegDB baseline 复现 (lab-4090)

## 目的
确认 CAJ baseline 在 lab-4090 跑通 + 数字对得上(官方 RegDB R1 85.03 / mAP 79.14 / mINP 65.33),作为换 Swin + 加 SMPL 几何锚之前的地基。

## 环境
- lab-4090(4090D, CUDA 12.4), uv 项目 /home/afr/vireid, torch 2.4.1+cu121。
- RegDB(Mac 下载 zip → scp, 8240 图, symlink Datasets/RegDB→data/RegDB, trial 1)。
- baseline: mangye16 ICCV21_CAJ, method=adp(Enhanced Squared Diff + Channel Aug + KL)。
- 启动: `train_ext.py --dataset regdb --method adp --augc 1 --rande 0.5 --alpha 1 --square 1 --gamma 1 --trial 1 --gpu 0`。
- resnet50 ImageNet 权重: Mac 下载 scp 到 /root/.cache/torch/hub/checkpoints/(绕过 pytorch.org 国内慢)。

## 进度记录

### [启动] Epoch 0
数据加载 OK(query/gallery 各 206 ID / 2060 图, Data Loading 3.4s)。
- Epoch[0][0/64]: Loss 93.33, iLoss 5.33, TLoss 88.0, KLoss 0.0025, Accu 0.00
- Epoch[0][50/64]: Loss 92.77, iLoss 4.77, Accu 4.49
正常(iLoss 降, Accu 升)。addmm_ deprecation warning 无害。Monitor bekcaem3s 盯后续。

### eval 进度(每 2 epoch, Monitor bekcaem3s)
- Epoch 4: POOL R1 12.62/mAP 12.40, FC R1 11.26/mAP 9.74(早期低, warmup 阶段正常)
- Epoch 9: Accu 64.27, Loss 3.02
- Epoch 10: Accu 81.12, Loss 1.69(Accu e0→e10: 4.5→81, 飞涨)

趋势健康。lr 现 0.100(warmup 完), 20/50 epoch 各 ×0.1 衰减后 mAP 会爬向目标。

### ★ 终值(best epoch 52, 训练完成)
**POOL Rank-1 76.80 / mAP 69.14 / mINP 53.61**(FC 75.58/68.20)。训练内 eval, **无 TTA, 单 trial 1**。
- 官方 85.03/79.14/65.33 = **testa.py flip-TTA + 10-trial 平均**。无 TTA 单 trial 文献本就 ~77/69 → **复现确认, pipeline 正常**。
- ⚠️ testa.py 直接跑给垃圾(5-7%): 它循环测 trial 1-10 期望 10 个 per-trial 模型, 我只训了 trial 1 → 其余 9 trial 随机权重拉垮平均。**非 pipeline 问题**, 是 testa.py 设计(要 10 trial 全训)。
- **结论: 用训练内 eval(76.80/69.14)当一致 baseline, 和 SMPL 锚同口径对比。** 不纠结绝对匹配 79(TTA+多trial差异)。

## 下一步: SMPL kill-switch(GPU 已空)
torchvision keypointrcnn(2D pose)在 RegDB IR 热 crop 上跑 → 检测率/关键点质量。答"IR 上能否提人体几何"。garbage→SMPL 锚死→转 Swin-VI 机制。

web search: 
web search: PersonViT Self-supervised Vision Transformer for Person Re-identification arxiv
web search: 
web search: SOLIDER Semantic Controllable Self-Supervised Learning Human-Centric Visual Tasks arxiv
web search: 
web search: HAP Structure-Aware Masked Image Modeling for Human-Centric Perception arxiv
web search: 
web search: PersonMAE masked autoencoder person re-identification occluded duke arxiv
web search: 
web search: Diffusion-ReID Synthesizing Efficient Data for Person Re-Identification pre-training arxiv 2024
web search: 
web search: DINOv2 Learning Robust Visual Features without Supervision arxiv
web search: 
web search: "DINOv3: Self-supervised learning for vision at unprecedented scale" arxiv
web search: 
web search: SAGA-ReID CLIP occluded person re-identification arxiv 2024 2025
web search: 
web search: "CLIP-ReID" "Exploiting Vision-Language Model" "arXiv" "person re-identification"
web search: 
web search: "Cross-view" "Person Re-identification" "self-supervised" "arXiv"
web search: 
web search: "V-JEPA" "Revisiting Feature Prediction for Learning Visual Representations" arxiv 2024
web search: 
web search: PVPM pose-guided visible part matching occluded person re-identification arxiv
codex
**结论**
不判死。`occlusion-aware completion` 这个词本身已经不新了，但“**pose-defined support-complete latent pretraining**”还有空间。最值得做的不是 MAE 式像素补全，也不是再加 part/GCN/attention，而是把 pretext 改成：

> 给模型一张不完整人体图，要求它在 latent body-part token 空间里预测“完整身份 support”，target 来自 EMA full-view teacher + 高置信跨图/伪同 ID support bank。

我建议主线叫暂名：**PSC-JEPA: Pose-defined Support-Complete JEPA pretraining**。

**Novelty 边界**
已被明显占掉的部分：

- SOLIDER 已做 human-centric SSL + pseudo semantic labels + semantic controller，不宜再讲“人体语义可控预训练”本身是新东西。([arxiv.org](https://arxiv.org/abs/2303.17602?utm_source=openai))
- PersonMAE 已把 occlusion-like masking、RegionA/RegionB、pixel + semantic feature prediction 用在 ReID pretraining，且直接报 Occluded-Duke/MSMT17。([arxiv.org](https://arxiv.org/abs/2311.04496?utm_source=openai))
- HAP 已做 human part-guided MIM + structure-invariant alignment，用人体结构先验指导 mask。([arxiv.org](https://arxiv.org/abs/2310.20695?utm_source=openai))
- PersonViT 2024 已做大规模 ReID SSL，组合 MIM + discriminative contrastive，并覆盖 MSMT17、Market、Occluded-Duke。([arxiv.org](https://arxiv.org/abs/2408.05398?utm_source=openai))
- SAIP 2025 又占了 cross-scale consistency/reconstruction/search 这条 human-centric pretraining 线。([arxiv.org](https://arxiv.org/abs/2503.08201?utm_source=openai))
- 下游 pose/visible-part matching 也不是空白：PVPM/PFD/KPR/ProFD 都覆盖了 pose-visible part、prompt、语义 part 对齐等方向。([arxiv.org](https://arxiv.org/abs/2004.00230?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2112.02466?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2407.18112?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2409.20081?utm_source=openai))

还没被完全吃掉的空间：

- **不是补像素，而是补身份 support**。
- **不是 single-image MAE，而是 support bank / pseudo cross-view teacher**。
- **不是 part pooling，而是 pretraining 阶段让 backbone 学会“缺哪个部位时如何形成可检索身份证据”**。
- JEPA/latent prediction 路线可借鉴，因为它预测 embedding target 而非像素，适合避免 ReID 中“重建背景/遮挡物”污染。([arxiv.org](https://arxiv.org/abs/2301.08243?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2404.08471?utm_source=openai))

**最推荐方案**
PSC-JEPA 具体做法：

1. **初始化**
   - 主干：SOLIDER/Swin-Tiny 继续预训练，保持后续 fine-tune 兼容。
   - DINOv2 可作为 frozen dense teacher 或并行 LoRA 试验，不建议一开始全量 fine-tune DINOv2-B。DINOv2 的价值是提供稳的 dense latent target，而不是换成另一个大工程。([arxiv.org](https://arxiv.org/abs/2304.07193?utm_source=openai))

2. **数据**
   - 只用 train split，避免 query/gallery 泄漏。
   - Market + MSMT17 + Occluded-Duke + Occluded-ReID + Occluded-PoseTrack。
   - 每张图预缓存 pose/keypoint visibility/body-group mask。
   - 额外构造 pose-defined complementary masks，继承 exp148 的正确方向，但不再把三视图 CE 混进下游训练，避免主损失 dilution。

3. **pretext**
   - 输入：partial view，只保留一部分 body support。
   - teacher target：
     - `T_full`: EMA teacher 看原图/full view 的 body-part latent tokens。
     - `T_bank`: 高置信 pseudo same-ID / nearest-neighbor support bank 中对应 body part prototype。
   - student 输出：
     - visible part tokens
     - missing part predicted tokens
     - union identity token

4. **loss**
   - `L_part_jepa`: missing part token 预测 teacher/support token，cosine/L2。
   - `L_union`: predicted union token 接近 full-view teacher identity token。
   - `L_gram`: part-token Gram/关系矩阵对齐，保留身份内部结构。
   - `L_visible_anchor`: 可见 part 不许被改坏，防止 completion 过度平滑。
   - `L_dino/solider_anchor`: 对可见区域蒸馏原始 backbone/DINOv2 dense feature，防 catastrophic forgetting。
   - 可选 `L_cluster_contrast`: 只用高置信 pseudo cluster，低置信跳过。

5. **关键区别**
   - 对 PersonMAE/HAP：不是 pixel/region reconstruction，是 latent support completion。
   - 对 SOLIDER：不是语义比例控制，是 missing identity evidence prediction。
   - 对 exp148 PCVT：不在 supervised ReID training 里三视图稀释主损失，而是单独 continued-pretrain，再 fine-tune。
   - 对 exp109：把 oracle support bank 的 headroom 尝试蒸进预训练。你本地 exp109 显示 oracle support 从 `61.88/73.26` 到 `70.40/81.36`，这条线有真实上界。

**4-slot 实验排布**
- 4090：PSC-JEPA 主跑，SOLIDER/Swin-Tiny，multi-dataset，50-100 epoch。
- 3090：去掉 support bank，只做 same-image full teacher，对照“是否只是 OA-SD/PCVT 换名”。
- 5060Ti-1：random mask / PersonMAE-lite latent baseline。
- 5060Ti-2：pseudo support bank 质量诊断、DINOv2 frozen teacher variant、小规模 Occluded-Duke smoke。

**工作量**
- v0 工程：5-8 天。
- 首轮预训练 + fine-tune：1-2 周。
- 有意义 ablation：3-4 周。
- 可投稿证据链：4-6 周。

**风险**
最大风险不是算力，而是 pseudo support bank 噪声。第一 kill-switch 要看：

- support target 覆盖率是否足够。
- pseudo same-ID top-k 精度诊断是否过线。
- 10-20 epoch continued pretrain 后，Occluded-Duke fine-tune 是否至少有 `+0.7 mAP` 或 heavy-occ 子集明显正向。
- plain continued-pretrain / random mask 不能同涨，否则 novelty 会塌。

**B 类潜力与信心**
我的判断：

- Buildability：`7/10`
- Novelty：`6.5/10`，如果 support bank + latent JEPA 做干净，可到 `7/10`
- 性能成功概率：`45%-55%`
- B 类潜力：`6.5/10`，成功条件是跨 Occluded-Duke/MSMT/Market 至少两处稳定正向，并且 heavy-occlusion subset 明显强于普通 MIM/continued-pretrain。

最终建议：**做 PSC-JEPA，不做普通 occlusion MAE，不做单纯 part-whole consistency。** 这条线最贴 exp109 的 headroom，也最像真正训练端范式转向。
tokens used
212,009
**结论**
不判死。`occlusion-aware completion` 这个词本身已经不新了，但“**pose-defined support-complete latent pretraining**”还有空间。最值得做的不是 MAE 式像素补全，也不是再加 part/GCN/attention，而是把 pretext 改成：

> 给模型一张不完整人体图，要求它在 latent body-part token 空间里预测“完整身份 support”，target 来自 EMA full-view teacher + 高置信跨图/伪同 ID support bank。

我建议主线叫暂名：**PSC-JEPA: Pose-defined Support-Complete JEPA pretraining**。

**Novelty 边界**
已被明显占掉的部分：

- SOLIDER 已做 human-centric SSL + pseudo semantic labels + semantic controller，不宜再讲“人体语义可控预训练”本身是新东西。([arxiv.org](https://arxiv.org/abs/2303.17602?utm_source=openai))
- PersonMAE 已把 occlusion-like masking、RegionA/RegionB、pixel + semantic feature prediction 用在 ReID pretraining，且直接报 Occluded-Duke/MSMT17。([arxiv.org](https://arxiv.org/abs/2311.04496?utm_source=openai))
- HAP 已做 human part-guided MIM + structure-invariant alignment，用人体结构先验指导 mask。([arxiv.org](https://arxiv.org/abs/2310.20695?utm_source=openai))
- PersonViT 2024 已做大规模 ReID SSL，组合 MIM + discriminative contrastive，并覆盖 MSMT17、Market、Occluded-Duke。([arxiv.org](https://arxiv.org/abs/2408.05398?utm_source=openai))
- SAIP 2025 又占了 cross-scale consistency/reconstruction/search 这条 human-centric pretraining 线。([arxiv.org](https://arxiv.org/abs/2503.08201?utm_source=openai))
- 下游 pose/visible-part matching 也不是空白：PVPM/PFD/KPR/ProFD 都覆盖了 pose-visible part、prompt、语义 part 对齐等方向。([arxiv.org](https://arxiv.org/abs/2004.00230?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2112.02466?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2407.18112?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2409.20081?utm_source=openai))

还没被完全吃掉的空间：

- **不是补像素，而是补身份 support**。
- **不是 single-image MAE，而是 support bank / pseudo cross-view teacher**。
- **不是 part pooling，而是 pretraining 阶段让 backbone 学会“缺哪个部位时如何形成可检索身份证据”**。
- JEPA/latent prediction 路线可借鉴，因为它预测 embedding target 而非像素，适合避免 ReID 中“重建背景/遮挡物”污染。([arxiv.org](https://arxiv.org/abs/2301.08243?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2404.08471?utm_source=openai))

**最推荐方案**
PSC-JEPA 具体做法：

1. **初始化**
   - 主干：SOLIDER/Swin-Tiny 继续预训练，保持后续 fine-tune 兼容。
   - DINOv2 可作为 frozen dense teacher 或并行 LoRA 试验，不建议一开始全量 fine-tune DINOv2-B。DINOv2 的价值是提供稳的 dense latent target，而不是换成另一个大工程。([arxiv.org](https://arxiv.org/abs/2304.07193?utm_source=openai))

2. **数据**
   - 只用 train split，避免 query/gallery 泄漏。
   - Market + MSMT17 + Occluded-Duke + Occluded-ReID + Occluded-PoseTrack。
   - 每张图预缓存 pose/keypoint visibility/body-group mask。
   - 额外构造 pose-defined complementary masks，继承 exp148 的正确方向，但不再把三视图 CE 混进下游训练，避免主损失 dilution。

3. **pretext**
   - 输入：partial view，只保留一部分 body support。
   - teacher target：
     - `T_full`: EMA teacher 看原图/full view 的 body-part latent tokens。
     - `T_bank`: 高置信 pseudo same-ID / nearest-neighbor support bank 中对应 body part prototype。
   - student 输出：
     - visible part tokens
     - missing part predicted tokens
     - union identity token

4. **loss**
   - `L_part_jepa`: missing part token 预测 teacher/support token，cosine/L2。
   - `L_union`: predicted union token 接近 full-view teacher identity token。
   - `L_gram`: part-token Gram/关系矩阵对齐，保留身份内部结构。
   - `L_visible_anchor`: 可见 part 不许被改坏，防止 completion 过度平滑。
   - `L_dino/solider_anchor`: 对可见区域蒸馏原始 backbone/DINOv2 dense feature，防 catastrophic forgetting。
   - 可选 `L_cluster_contrast`: 只用高置信 pseudo cluster，低置信跳过。

5. **关键区别**
   - 对 PersonMAE/HAP：不是 pixel/region reconstruction，是 latent support completion。
   - 对 SOLIDER：不是语义比例控制，是 missing identity evidence prediction。
   - 对 exp148 PCVT：不在 supervised ReID training 里三视图稀释主损失，而是单独 continued-pretrain，再 fine-tune。
   - 对 exp109：把 oracle support bank 的 headroom 尝试蒸进预训练。你本地 exp109 显示 oracle support 从 `61.88/73.26` 到 `70.40/81.36`，这条线有真实上界。

**4-slot 实验排布**
- 4090：PSC-JEPA 主跑，SOLIDER/Swin-Tiny，multi-dataset，50-100 epoch。
- 3090：去掉 support bank，只做 same-image full teacher，对照“是否只是 OA-SD/PCVT 换名”。
- 5060Ti-1：random mask / PersonMAE-lite latent baseline。
- 5060Ti-2：pseudo support bank 质量诊断、DINOv2 frozen teacher variant、小规模 Occluded-Duke smoke。

**工作量**
- v0 工程：5-8 天。
- 首轮预训练 + fine-tune：1-2 周。
- 有意义 ablation：3-4 周。
- 可投稿证据链：4-6 周。

**风险**
最大风险不是算力，而是 pseudo support bank 噪声。第一 kill-switch 要看：

- support target 覆盖率是否足够。
- pseudo same-ID top-k 精度诊断是否过线。
- 10-20 epoch continued pretrain 后，Occluded-Duke fine-tune 是否至少有 `+0.7 mAP` 或 heavy-occ 子集明显正向。
- plain continued-pretrain / random mask 不能同涨，否则 novelty 会塌。

**B 类潜力与信心**
我的判断：

- Buildability：`7/10`
- Novelty：`6.5/10`，如果 support bank + latent JEPA 做干净，可到 `7/10`
- 性能成功概率：`45%-55%`
- B 类潜力：`6.5/10`，成功条件是跨 Occluded-Duke/MSMT/Market 至少两处稳定正向，并且 heavy-occlusion subset 明显强于普通 MIM/continued-pretrain。

最终建议：**做 PSC-JEPA，不做普通 occlusion MAE，不做单纯 part-whole consistency。** 这条线最贴 exp109 的 headroom，也最像真正训练端范式转向。
