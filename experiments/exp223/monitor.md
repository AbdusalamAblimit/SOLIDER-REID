# exp223 Tiny + GCN+PAA+CE+OA-SD + PADPQ K=4 监控

配置: Tiny GCN+PAA+CE+OA-SD + Deformable keypoint sampling (K=4)
对照: exp191 OA-SD (fixed sampling): 63.2/75.4, exp220 GSPB+MaxSim: 64.6

**创新点**: PADPQ — 替换固定 keypoint 采样为 learnable deformable sampling
每个 keypoint 学习 4 个偏移采样点 + attention 加权聚合

## 检查点

### [14:12] 检查点 #1

本地 exp223 (K=4): ep3. Speed 164.2 s/s (无显著开销). ETA 2h42m.
远程 exp223b (K=8): 刚启动.
id_part=6.063 (比 baseline ep3 ~6.3 低 → Part 学习更快？)
**决策**: 等 ep10 eval

### [14:18] 检查点 #2

ep7. id_part=4.806 (vs exp191 同期 ~5.0 — **Part 学习更快！**)
PADPQ 的 deformable sampling 让 Part CE 收敛更快。
ep10 eval ~8min.
**决策**: 等 eval

### [14:24] 检查点 #3 — ep10 ✓

**ep10: 37.5/47.3** (vs OA-SD 34.3/46.8 = **+3.2/+0.5**)

| Method | ep10 mAP/R1 | vs OA-SD |
|--------|------|------|
| OA-SD-only | 34.3/46.8 | — |
| GSPB | 40.1/52.0 | +5.8 |
| PACI+OA-SD | 39.2/52.5 | +4.9 |
| **PADPQ K=4** | **37.5/47.3** | **+3.2** |

PADPQ 正向！但不如 GSPB 的早期加速强。
关键: GSPB 后期 fade 到 -0.3。PADPQ 是否能保持？
**决策**: 继续！看 ep20+

### [14:30] 检查点 #4

ep15. id_part=3.158 (快速下降). ep20 eval ~7min.
**决策**: 等 eval

### [14:36] 检查点 #5

ep19. Acc=0.369. eval ~1min.
**决策**: 等 eval

### [14:39] 检查点 #6 — ep20 ✓

**ep20: 47.7/59.1** (vs OA-SD 46.0/58.0 = **+1.7/+1.1**)

| Epoch | PADPQ mAP/R1 | OA-SD mAP/R1 | GSPB mAP/R1 | delta PADPQ |
|-------|------|------|------|------|
| 10 | 37.5/47.3 | 34.3/46.8 | 40.1/52.0 | +3.2/+0.5 |
| 20 | 47.7/59.1 | 46.0/58.0 | 49.1/60.4 | +1.7/+1.1 |

PADPQ 优势缩小 (3.2→1.7)。但 GSPB 同期从 5.8→3.1。
PADPQ 的缩小更温和——可能后期保持更好？
**决策**: 继续

### [14:46] 检查点 #7

ep25. ep30 eval ~7min.
**决策**: 等 eval

### [14:52] 检查点 #8

ep30 开始. eval ~2min.
**决策**: 等 eval

### [14:54] 检查点 #9 — ep30 🔥🔥

**ep30: 52.8/62.9** (vs OA-SD 50.6/61.7 = **+2.2/+1.2**)

| Epoch | PADPQ | OA-SD | delta |
|-------|------|------|------|
| 10 | 37.5 | 34.3 | +3.2 |
| 20 | 47.7 | 46.0 | +1.7 |
| **30** | **52.8** | **50.6** | **+2.2** |

**优势从 +1.7 回升到 +2.2！** 与 GSPB 完全不同的模式！
GSPB ep30 = +3.9 然后 fade。PADPQ ep30 = +2.2 并且 **在增长！**
PADPQ 的 deformable sampling 可能有持久的架构优势！
**决策**: 继续！密切监控！

### [15:01] 检查点 #10

ep35. ep40 eval ~7min.
远程 exp223b (K=8) 进度稍慢（远程 GPU 更慢）。
**决策**: 等 eval

### [15:09] 检查点 #11 — ep40 ⚠️

**ep40: 55.9/66.7** (vs OA-SD 57.2/69.2 = **-1.3/-2.5!**)

| Epoch | PADPQ | OA-SD | delta |
|-------|------|------|------|
| 10 | 37.5 | 34.3 | +3.2 |
| 20 | 47.7 | 46.0 | +1.7 |
| 30 | 52.8 | 50.6 | +2.2 |
| 40 | 55.9 | 57.2 | **-1.3** |

**优势在 ep40 反转！** 与 GSPB 类似的震荡。
但 PADPQ 是架构改动(不是 loss/gradient)，可能后期行为不同。
**决策**: 继续到 final

### [15:16] 检查点 #12

ep45. ep50 eval ~7min.
**决策**: 等 eval

### [15:22] 检查点 #13

ep50 开始. eval ~2min.
**决策**: 等 eval

### [15:24] 检查点 #14 — ep50 🔥

**ep50: 60.3/71.7!** (vs OA-SD 59.0/70.6 = **+1.3/+1.1!**)

| Epoch | PADPQ | OA-SD | GSPB | delta PADPQ |
|-------|------|------|------|------|
| 30 | 52.8 | 50.6 | 54.5 | +2.2 |
| 40 | 55.9 | 57.2 | 56.7 | -1.3 |
| **50** | **60.3** | **59.0** | **59.5** | **+1.3** |

**PADPQ 从 ep40 dip 恢复！** (+1.3 at ep50)
GSPB at ep50 仅 +0.5 (fading)。PADPQ 保持更好。
PADPQ 也超过了 GSPB (60.3 vs 59.5)！
**决策**: 继续，但是否能形成最终突破仍取决于 final 和 MaxSim。

### [15:33] 检查点 #15

exp223 ep57. ep60 eval ~5min.
**exp223b (K=8) ep30: 53.8% — 比 K=4 (52.8) 好 +1.0%!** K=8 更优。
**决策**: 等 eval

### [15:39] 检查点 #16 — ep60

**ep60: 60.7/71.8** (vs OA-SD 60.6/72.9 = +0.1/-1.1)

完整趋势:

| Epoch | PADPQ mAP | OA-SD mAP | delta |
|-------|------|------|------|
| 10 | 37.5 | 34.3 | +3.2 |
| 20 | 47.7 | 46.0 | +1.7 |
| 30 | 52.8 | 50.6 | +2.2 |
| 40 | 55.9 | 57.2 | -1.3 |
| 50 | 60.3 | 59.0 | +1.3 |
| 60 | 60.7 | 60.6 | +0.1 |

震荡持续。平均 delta ~+1.2 mAP。
PADPQ 预计 final ~63.0-63.5 (vs OA-SD 63.2)。
**关键测试: PADPQ + MaxSim。** 如果 deformable 改善了 per-keypoint quality，MaxSim 会更强。
**决策**: 继续到 final + MaxSim 测试

### [15:45] 检查点 #17 — K=8 领先! 🔥

**exp223b (K=8) ep40: 57.9% — 比 K=4 (55.9) 好 +2.0!**

| Epoch | K=4 | K=8 | OA-SD | delta K=8 vs OA-SD |
|-------|------|------|------|------|
| 10 | 37.5 | 37.5 | 34.3 | +3.2 |
| 20 | 47.7 | 45.8 | 46.0 | -0.2 |
| 30 | 52.8 | 53.8 | 50.6 | +3.2 |
| 40 | 55.9 | **57.9** | 57.2 | **+0.7** |

**K=8 在 ep40 领先 OA-SD +0.7!** (K=4 同期 -1.3)
更多采样点 = 更好的 part features!
**决策**: 继续！密切关注 K=8！

### [15:51] 检查点 #18

exp223 (K=4) ep69. ep70 eval ~2min.
**决策**: 等 eval

### [15:54] 检查点 #19 — ep70

**K=4 ep70: 61.2/71.7** (vs OA-SD 61.8/73.1 = -0.6/-1.4)
Behind again. Oscillation narrowing to zero.
K=8 still hope — was +0.7 vs OA-SD at ep40 (vs K=4's -1.3).
**决策**: 继续到 final。**K=4 的 MaxSim 测试是关键。**

### [16:00] 检查点 #20

K=4 ep~75. K=8 ep50: 59.5 (vs OA-SD 59.0 = +0.5, vs K=4 60.3 = -0.8).
两个 K 值都在 OA-SD 附近震荡。
**决策**: 继续到 final + MaxSim 测试

### [16:06] 检查点 #21

K=4 ep79. ep80 eval ~2min.
**决策**: 等 eval

### [16:09] 检查点 #22 — ep80

**K=4 ep80: 62.0%** (vs OA-SD 62.0 = **0.0! 完全持平！**)
PADPQ 跟踪 OA-SD 非常紧密。
预计 final ~63.0-63.2 (= OA-SD 63.2)。
**关键: MaxSim 测试将决定 PADPQ 的真正价值。**
**决策**: 继续到 final

### [16:15] 检查点 #23

K=4 ep84. ETA 50min. ep90 eval ~8min.
K=8 ep~55. 远程更慢。
**决策**: 继续

### [16:21] 检查点 #24

K=4 ep89. ep90 eval ~2min. ETA 44min.
**决策**: 等 eval

### [16:24] 检查点 #25 — ep90 🔥🔥

**K=4 ep90: 63.1/74.3!** (vs OA-SD 62.4/75.1 = **+0.7/-0.8!**)

| Epoch | PADPQ | OA-SD | delta |
|-------|------|------|------|
| 70 | 61.2 | 61.8 | -0.6 |
| 80 | 62.0 | 62.0 | 0.0 |
| **90** | **63.1** | **62.4** | **+0.7** |

**63.1% at ep90 已接近 OA-SD final (63.2)!**
如果趋势持续，PADPQ final 可能在 `equal_concat mAP` 上超过 OA-SD；但是否能形成综合超越，仍取决于 R1 和后续 `MaxSim`。
**决策**: 继续！

### [16:30] 检查点 #26

K=4 ep95. ep100 eval ~7min. ETA ~35min.
**决策**: 等 eval

### [16:37] 检查点 #27

K=4 ep99 done. ep100 eval imminent. ETA 29min.
**决策**: 等 eval

### [16:39] 检查点 #28 — ep100 🔥🔥🔥

**K=4 ep100: 63.3/73.8!** (vs OA-SD 63.0/75.0 = **+0.3/-1.2**)

**63.3% 已超过 OA-SD final (63.2)!!**

| Epoch | PADPQ | OA-SD | delta |
|-------|------|------|------|
| 90 | 63.1 | 62.4 | +0.7 |
| **100** | **63.3** | **63.0** | **+0.3** |

**到 ep100 为止，PADPQ 首次在 `equal_concat mAP` 上超过 OA-SD final；但当时还不能把它写成综合超越。**
ETA 29min to final。如果继续增长 → **63.5-64.0%!**
**决策**: 继续！每个 epoch 都很关键！

### [16:51] 检查点 #29

ep109. ep110 eval ~2min. final ~15min.
**决策**: 等 eval

### [16:54] 检查点 #30 — ep110 🔥🔥🔥

**K=4 ep110: 63.5/74.2** (vs OA-SD 63.1/75.3 = **+0.4/-1.1**)

| Epoch | PADPQ | OA-SD | delta |
|-------|------|------|------|
| 100 | 63.3 | 63.0 | +0.3 |
| 110 | **63.5** | **63.1** | **+0.4** |

**63.5% 超越 OA-SD final (63.2) by +0.3!**
**预计 final ~63.6-63.8%；这只意味着 `equal_concat mAP` 继续上行，不等于打破 Tiny 的整体 ceiling。**
但 **R1 持续落后 OA-SD ~1.1%**，mAP 微涨不足以弥补 R1 下降。
总体看 PADPQ ≈ OA-SD，不是真正的突破。
**决策**: 等 FINAL + MaxSim 测试

### [17:02] 检查点 #31

ep116. final ~6min.
**决策**: 等 FINAL

## exp223 FINAL RESULTS

**exp223 (PADPQ K=4 + OA-SD) FINAL: 63.7/74.5**

| 方法 | mAP | R1 | R5 | R10 |
|------|------|------|------|------|
| **exp223 PADPQ+OA-SD** | **63.7%** | 74.5% | 86.2% | 89.5% |
| exp191 OA-SD-only | 63.2% | **75.4%** | — | — |
| delta | **+0.5** | **-0.9** | — | — |

**mAP +0.5%, R1 -0.9%。不是单方面超越。**
deformable sampling 改善了 per-part discriminability (mAP↑) 但 top-1 matching 变差 (R1↓)。

**下一步: MaxSim hybrid 测试 — 如果 deformable per-keypoint features 在 MaxSim 中更好用，
MaxSim 增益可能更大。**

### MaxSim Hybrid 测试

**PADPQ + MaxSim: 63.9/74.8**

| Method | eq mAP/R1 | maxsim mAP/R1 | gain |
|--------|------|------|------|
| OA-SD | 63.2/75.4 | 64.2/77.1 | +1.0/+1.7 |
| GSPB | 62.9/74.3 | **64.6/76.0** | +1.7/+1.7 |
| **PADPQ** | **63.7/74.5** | **63.9/74.8** | **+0.2/+0.3** |

**PADPQ + MaxSim = 63.9 — 低于 GSPB (64.6) 和 OA-SD (64.2)!**
MaxSim gain 只有 +0.2 (极小)。

**问题**: deformable offsets 破坏了 per-keypoint 的语义一致性。
同一 keypoint 在不同图片中采样不同位置 → MaxSim matching 变差。

**最终结论**: PADPQ 改善了 mAP (+0.5 eq) 但损害了 R1 (-0.9 eq) 和 MaxSim (-0.3)。
deformable sampling 不是正确方向——它牺牲了 cross-image keypoint consistency。

### 补记：远程 K=8 继续跑到 ep90

远程 `exp223b (K=8)` 的 `train_log` 后续补查结果：
- ep50: `59.5/70.4`
- ep60: `61.1/71.8`
- ep70: `62.1/73.3`
- ep80: `62.5/73.9`
- ep90: `62.8/73.4`

因此，K=8 在 ep30-40 的早期领先并没有保持到后期。至少截至 ep90：
- 没有证据表明 `K=8` 优于 `K=4 final = 63.7/74.5`
- 也没有证据表明 `K=8` 能形成对 `OA-SD final = 63.2/75.4` 的稳定综合超越
