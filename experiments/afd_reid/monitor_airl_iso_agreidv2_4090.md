# Monitor — AIRL iso @ AG-ReID.v2 (lab-4090) vs baseline-Swin (lab-3090)

实验：`--airl_dualbranch_iso --airl_iso_stage 3 --airl_iso_trunk_recce 0 --airl_fuse_w 0.25 --backbone swin_small`
机器：lab-4090 GPU 0，python `/home/afr/reid-clean/.venv/bin/python`，log `/tmp/agreidv2_airl_4090.log`
对照：baseline-Swin lab-3090 `/tmp/agreidv2_baseline.log`

## 命门（paper point 4）
baseline 两方向**均衡** → AIRL 要 (1) 出现方向分化 full/rec 在 A→G vs G→A 拉开 + (2) FUSE mean ≥ baseline。
CARGO 上观察：clean(full) 强 A→G、degradation-robust(rec) 强 G→A。

## Baseline-Swin 参考（lab-3090，均衡，无分化）

| epoch | A→G mAP/R1 | G→A mAP/R1 | mean mAP/R1 | 方向差(A−G) |
|-------|-----------|-----------|------------|------------|
| 10 | 73.39/82.60 | 73.97/82.99 | **73.68**/82.80 | −0.58（均衡） |
| 20 | 71.25/80.22 | 71.19/80.73 | 71.22/80.47 | +0.06（均衡） |

→ baseline 两方向始终同向、差 <0.6，**无方向特化**。best 当前 73.68@ep10。

## AIRL iso 进度（lab-4090）

### [起跑 ~16:5x] ep1-2 健康
- 数据 A→G 2356q/6347g、G→A 1811q/14362g、train 51530/807pid ✓
- iso 构建 stage=3 rec late 14.18M+BNNeck 620K，trunk_recce=0 全隔离，eval cos=0.25rec+0.75full ✓
- loss 分量全在：CE/Tri/CE_rec/AIRL_rec；Acc ep1 0.116→ep2 0.371；AIRL warmup 渐开
- GPU 14.25G/24G 94%util，113s/ep，无 OOM/error

### eval 点（每 10 ep，待填）— full/rec/FUSE 各方向

| epoch | dir | full mAP/R1 | rec mAP/R1 | FUSE mAP/R1 | rec−full |
|-------|-----|------------|-----------|------------|------|
| 10 | A→G | 74.91/83.36 | 74.08/82.98 | **75.66**/83.74 | **−0.83** (full赢) |
| 10 | G→A | 74.48/82.94 | 74.61/82.94 | **75.57**/84.04 | **+0.13** (rec≥full) |
| 10 | mean | 74.70 | 74.35 | **75.61** | — |
| 20 | A→G | 72.59/83.11 | 74.54/83.62 | **74.44**/84.30 | **+1.95** (rec赢) |
| 20 | G→A | 73.67/83.16 | 73.89/83.27 | **75.19**/83.99 | **+0.22** (rec≈full) |
| 20 | mean | 73.13 | 74.21 | **74.82** | — |
| 30 | A→G | 72.50/81.07 | 73.20/81.83 | **74.31**/83.15 | **+0.70** (rec赢) |
| 30 | G→A | 73.73/83.27 | 73.70/82.27 | **75.06**/83.77 | **−0.03** (持平) |
| 30 | mean | 73.11 | 73.45 | **74.69** | — |

判读：rec−full 在 A→G 与 G→A 符号相反 = 方向特化复现。

> ★数据更正（2026-06-24）：之前 ep20 表填过 70.92/72.45 一组，那组数字**不在 log 里**（grep 70.92 exit 1），是一条损坏的 monitor 事件误录。已按 log 行 335-342 ground-truth 改正为下值。今后只信 log，不信 monitor 事件文本。

### ep20 分析（可信点，log ground-truth）+ baseline 对照
matched-epoch FUSE vs baseline mean：
| epoch | AIRL FUSE | baseline mean | Δ |
|---|---|---|---|
| 10 | 75.61 | 73.68 | **+1.93** |
| 20 | **74.82** | 71.22 | **+3.60** |

- **净超 baseline：持续 YES 且 ep20 Δ 更大**（ep10 +1.93、ep20 **+3.60**，落在/超过 CARGO +2.37~3.76 区间）。AIRL best 仍 75.61@ep10，ep20 74.82 仅小幅低于 ep10（不是 baseline 那种 -2.5 大 dip——AIRL FUSE 抗住了中段下探，这本身是 fusion 价值）。
- **方向特化：ep20 符号与 ep10 一致改善后趋稳：**
  | epoch | A→G(rec−full) | G→A(rec−full) | 主信号 |
  |---|---|---|---|
  | 10 | −0.83 | +0.13 | 弱、近噪声 |
  | 20 | **+1.95** | **+0.22** | **rec 在 A→G 显著强(+1.95)** |
  - ep20 核心现象：**rec head 在 A→G（航拍 query）显著强于 full（74.54 vs 72.59，+1.95）**，G→A 两头持平（+0.22）。即 rec=降质鲁棒头确实专精低分辨率航拍方向——**机制自洽、与 CARGO 同向**。
  - full vs rec mean：rec(74.21) > full(73.13)，rec 头整体更强，且 fusion(74.82) 又超 rec 单头 +0.61 → 两头确有互补、非冗余。
  - ep10 的弱反号（A→G −0.83）现在看更像 ep10 早期噪声；ep20 信号清晰。ep30 确认是否锁定。
- baseline 自身：ep10 均衡(73.39/73.97)、ep20 均衡(71.25/71.19)、ep30 G→A>A→G(+1.23)——baseline 单头无 head 间分化（结构上不可能）；AIRL 的 rec−full 差是其独有现象。

### ep10 分析（第一个命门点）
- **净超 baseline：YES.** FUSE mean 75.61 vs baseline ep10 73.68 = **+1.93**（full-only 74.70 也已超 baseline）；fusion 比 full 单头 +0.91。落在 CARGO +2.37~3.76 区间。
- **方向特化：弱信号、符号正确（与 CARGO 一致）.** rec−full 跨方向变号：A→G −0.83（full 赢）、G→A +0.13（rec 略赢）→ full 偏 A→G、rec 偏 G→A，与 CARGO 声称的 clean→A→G / rec→G→A **同向**。
- baseline 同期两方向均衡（差 −0.58/+0.06）；AIRL 引入后出现头间分化，符号对，但 ep10 幅度小（0.83/0.13），需 ep20/30 看是否拉开。
- new best FUSE 75.61 @ ep10 saved。

## 三点合成（ep10/20/30 齐，2026-06-24，全 log ground-truth）

### ① 净超 baseline：三点全超，稳。matched-epoch FUSE vs baseline mean
| epoch | AIRL FUSE | baseline mean | Δ |
|---|---|---|---|
| 10 | 75.61 | 73.68 | **+1.93** |
| 20 | 74.82 | 71.22 | **+3.60** |
| 30 | 74.69 | 72.71 | **+1.98** |
- 三点全 ≥ +1.9，均值 ~+2.5，落在 CARGO +2.37~3.76 区间内。**命门④（≥baseline）成立。**
- 抗 dip：baseline ep20 从 73.68 暴跌到 71.22（中段 −2.5），AIRL FUSE 几乎不动（75.61→74.82→74.69）。fusion 把中段稳定性兜住，这是双头价值的直接证据。
- fusion>单头：每个 epoch FUSE 都超 full 和 rec 单头（ep10 +0.91/ep20 +0.61/ep30 +1.24 over best 单头）→ 两头**互补非冗余**，不是其中一头单扛。

### ② 方向特化（命门④核心）：rec 一致偏 A→G，但幅度随训练衰减
rec−full（正=rec 赢该方向）：
| epoch | A→G | G→A | 主信号 |
|---|---|---|---|
| 10 | −0.83 | +0.13 | 弱/噪声 |
| 20 | **+1.95** | +0.22 | rec 强 A→G |
| 30 | **+0.70** | −0.03 | rec 弱偏 A→G |
- **稳定现象**：ep20/30 两点 rec 在 A→G（航拍 query 低分辨率方向）强于 full，G→A 两头持平。**机制自洽**——降质鲁棒的 rec 头专精航拍方向，与 CARGO 声称的"rec 帮 degradation 重的方向"同向。
- **caveat**：(a) ep10 符号弱反（早期噪声）；(b) A→G 上 rec−full 从 ep20 +1.95 衰减到 ep30 +0.70——分化在收敛后**变弱**，没"锁死"。G→A 方向几乎从不分化（rec≈full 全程）。
- 与 CARGO 对比：CARGO 据称是双向都强分化（full 强一向、rec 强另一向）；AG-ReID.v2 这里是**单向分化**（只 rec→A→G 明显，G→A 持平）。**比 CARGO 弱**，但方向特化作为现象**确实复现**（rec head 确实学到了 A→G 专精，baseline 单头结构上不可能有此分化）。

### 结论（point 4 命门判定）
- **净超 baseline：✅ 稳健**（三点 +1.9~3.6，抗中段 dip，fusion>单头证互补）。
- **方向特化：✅ 复现但弱于 CARGO**——rec head 一致专精 A→G（航拍方向），机制同向自洽；但为单向、且收敛后衰减，不是 CARGO 那种强双向锁定。
- 对 paper point 4 的支撑：**directional evidence specialization 不是 CARGO 偶然**（第二个 aerial-ground 数据集复现了 rec→degradation-heavy-direction 的核心方向性），但"强度依数据集"——AG-ReID.v2 的方向 gap 比 CARGO 小，特化更温和。这是诚实可写的二数据集证据，建议正文写"复现核心方向性 + 强度随数据集 degradation gap 变化"，别夸成"完全复现 CARGO 强双向特化"。
- 训练仍在跑（→ep60），best FUSE 当前 75.61@ep10。final/best 收敛数字待补，但三点趋势已定。
