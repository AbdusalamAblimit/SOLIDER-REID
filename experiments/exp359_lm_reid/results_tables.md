# LM-ReID 6.5 — Paper-Ready 核心实验表（从 monitor.md 精确复制，2026-06-26）

> Backbone = SOLIDER (exp359_abl_noLMloss, market HR ReID, M=3 aug no-LM-loss)，HR sanity 88.92。
> 数据 = Market-1501 合成 LR（h = LR query 高度 px，K=9 lattice 变体）。所有数字 mAP(%)，无后处理。

## Table 1 — Main result: lattice marginalization vs strong TTA（LM-S2-strong）

decision = MaxSim；strong TTA = pad-crop + resize-jitter + color，同 K 同 backbone 同 compute。

| h (LR px) | single | phase-lattice gain | strong-TTA gain | **LATTICE−TTA** |
|---|---|---|---|---|
| 12 | 66.79 | +5.494 | −1.781 | **+7.275** |
| 16 | 77.44 | +2.811 | −0.526 | **+3.337** |
| 20 | 82.49 | +1.195 | −0.035 | **+1.230** |
| 24 | 84.28 | +0.991 | −0.144 | **+1.135** |
| 32 | 86.44 | +0.759 | +0.003 | **+0.757** |

→ lattice marginalization **全 5 分辨率 beat 强 TTA**（+0.76~7.28），severe LR（h12）强 TTA 本身有害（−1.78）。堵死"strong TTA / ensemble"质疑。

## Table 2 — Aggregation ablation（LM-S3，K=9，decision-level vs embedding-level）

| h | single | mean-feat (embed) | MaxSim (hard) | **logsumexp (soft)** |
|---|---|---|---|---|
| 12 | 66.72 | 72.25 | 72.03 | **73.01** |
| 16 | 77.44 | 79.84 | 80.00 | **80.28** |
| 20 | 82.49 | 83.41 | **83.69** | 83.62 |
| 24 | 84.28 | 85.02 | **85.28** | 85.16 |
| 32 | 86.44 | 86.87 | **87.18** | 87.02 |

→ soft decision marginalization（logsumexp，LM-ReID 公式）severe LR 最优；decision-level（max/logsumexp）≥ embedding-mean（h≥16）。

## Table 3 — Factor ablation（LM-S4，bbox 主导跨分辨率）

bbox-only marginalize (lattice_axis=1, MaxSim) ≈ all-axis → bbox 检测框 crop 不确定性主导。

| h | bbox-only | all-axis | (h16 单轴对比: phase +1.76 / bbox +2.84 / zoom +1.70) |
|---|---|---|---|
| 12 | 72.01 | 72.03 | bbox-only ≈ all |
| 16 | 80.34 | 80.00 | bbox 略高 |
| 20 | 83.94 | 83.69 | |
| 24 | 85.53 | 85.28 | |
| 32 | 87.33 | 87.18 | |

## Table 4 — Compute-accuracy（K-sweep, h16）

| K | mAP | gain | 收益% of K=9 |
|---|---|---|---|
| 1 (single) | 77.44 | — | 0% |
| 3 | 78.73 | +1.29 | 53% |
| 5 | 79.61 | +2.14 | **87%** |
| 9 | 79.90 | +2.46 | 100% |

→ K=5 sweet spot（87% 收益 56% compute）。adaptive-K（per-query volatility）≈ fixed K=5（无优势，query-side 预测受益度做不到，同 LPA 死因）。

## Table 5 — Backbone generalization（Swin-small market baseline，LATTICE−strong-TTA）

| h | single | MaxSim | LATTICE−TTA |
|---|---|---|---|
| 16 | 41.41 | 46.20 | **+3.061** |
| 20 | 61.92 | 66.43 | **+3.162** |
| 24 | 70.31 | 73.90 | +2.370 |
| 32 | 81.97 | 83.93 | +0.883 |

→ 机制不依赖 SOLIDER backbone（Swin 上也 beat 强 TTA ~+3）。

## Table 6 — 机制范围界定（detector-jitter σ-sweep，诚实 Discussion）

均匀 ±1 LR-px lattice → 连续 Gaussian center+scale（模拟检测器 localization error）。marg gain：

| detector σ (LR-px) | h12 | h16 | h20 |
|---|---|---|---|
| 0 (均匀离散) | +5.49 | +2.81 | +1.24 |
| 0.25 | +3.68 | +1.55 | +0.68 |
| 0.5 (真实 COCO detector) | +2.18 | +0.86 | +0.29 |
| 1.0 (大误差) | **−5.85** | **−3.11** | **−1.44** |

→ marginalization 增益随 detector 误差**单调衰减、大误差有害**。机制是 **sub-pixel sampling-lattice 边缘化、不是 detector 鲁棒性**（这是 6.5 而非 7.0 的诚实原因；真实 detector-calibrated 验证需有原图数据集 CUHK-SYSU/PoseTrack，留未来）。

## Table 7 — Why Training-Time Invariance Fails（controlled alternatives，4 类全负）

| 类 | 代表 | 结果 |
|---|---|---|
| embedding invariance | full LM-ReID consistency | 75.71 < no-LM 77.44；HR sanity 86.09 < 88.92（L_marg 主害）|
| frozen adaptation | LS-MRT / LPA | +0.028 / +0.075（无 headroom）|
| backbone set/robust | LSRC / Hard-Lattice | LSRC HR 88.92→85.84；Hard-Lattice 76.9 < 77.44（train acc 1.0 但 test 掉）|
| input canonicalization | BLC | 数据证伪（canonicalize bbox 主因子 < marginalize 它，market 框已 canonical→退化 single）|

→ codex final 8.5/10 判训练端无空间。论点：*Learning invariance is the wrong objective; marginalizing decisions over plausible observations is the right one.*

---

**定位**：6.5/10 中等偏强 B 类。test-time decision marginalization + 训练端系统反例 + 多维证据（5 分辨率 / 强 TTA / 因子 / compute / backbone）+ σ-sweep 诚实机制范围界定。留用户：multi-seed + 正式 train + MLR benchmark + 真实 detector 数据集。
