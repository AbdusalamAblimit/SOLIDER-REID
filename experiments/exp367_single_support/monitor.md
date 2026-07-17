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
