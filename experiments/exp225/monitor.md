# exp225 Tiny + GSPB + PADPQ Combined 监控

配置: 基于 pose_psg_gcn_paa_roa.yml + OA-SD + PLBOA + GSPB(0.05) + PADPQ(K=4)
对照: exp220 GSPB(maxsim 64.6), exp223 PADPQ(eq 63.7), exp191 OA-SD(63.2/75.4)

## 检查点

### [18:33] 检查点 #1

远程刚启动。等 ep10 eval。

### [02:59] 检查点 #2 — ep10

**ep10: 38.3%** (vs GSPB 40.1, PADPQ 37.5, OA-SD 34.3)
介于 GSPB 和 PADPQ 之间——没有明显的叠加效果。
**决策**: 继续到 final

### [03:20] 检查点 #3 — ep20

**ep20: 49.4%** (vs GSPB 49.1, PADPQ 47.7, OA-SD 46.0)
微幅超过 GSPB (+0.3)。组合效果不明显。
**决策**: 继续

### [05:48] 检查点 #4

exp225 ep89. ep90 eval ~1min.
ep70: 62.3 (+0.5 vs OA-SD), ep80: 62.8 (+0.8 vs OA-SD) — 稳定正向!
**决策**: 等 eval

### [05:51] 检查点 #5 — ep90

**ep90: 63.6%** (vs OA-SD 62.4 = **+1.2!**)

| Epoch | GSPB+PADPQ | OA-SD | delta |
|-------|------|------|------|
| 70 | 62.3 | 61.8 | +0.5 |
| 80 | 62.8 | 62.0 | +0.8 |
| 90 | **63.6** | **62.4** | **+1.2** |

**优势在持续扩大！** +0.5→+0.8→+1.2。
ep90 已超过 OA-SD final (63.2) by +0.4。
如果趋势持续: final 可能达到 **64.0-64.5%!**
**决策**: 继续！

### [06:13] 检查点 #6 — ep100

**ep100: 63.9%** (vs OA-SD 63.0 = +0.9)

| Epoch | GSPB+PADPQ | OA-SD | delta |
|-------|------|------|------|
| 80 | 62.8 | 62.0 | +0.8 |
| 90 | 63.6 | 62.4 | +1.2 |
| 100 | 63.9 | 63.0 | +0.9 |

63.9% 已超过 OA-SD final (63.2) by +0.7。
平均 delta ep70-100 ≈ +0.9 mAP。
预计 final ~64.0-64.2%。
**决策**: 等 final

### [06:33] 检查点 #7

exp225 ep110. eval running. ETA 20min.
**决策**: 等 final

### [06:34] 检查点 #8 — ep110

**ep110: 64.1/74.6** (vs OA-SD 63.1/75.3 = +1.0/-0.7)
mAP 持续正向 (+1.0)，R1 略负 (-0.7)。
**决策**: 等 FINAL

### [06:52] 检查点 #9

exp225 ep119. FINAL ~2min!
exp226 ep99. ep100 eval ~3min.
**决策**: 等 FINAL

## exp225 FINAL RESULTS

**exp225 (GSPB scale=0.05 + PADPQ K=4 + OA-SD) FINAL: 64.2/74.9**

| 方法 | mAP | R1 | R5 | R10 |
|------|------|------|------|------|
| **exp225 GSPB+PADPQ** | **64.2%** | 74.9% | 86.8% | 89.6% |
| exp191 OA-SD-only | 63.2% | **75.4%** | — | — |
| delta | **+1.0** | **-0.5** | — | — |

**mAP +1.0, R1 -0.5 vs OA-SD-only。**
mAP 在 Tiny 上首次显著超过 OA-SD by +1.0%！
但 R1 仍略负 (-0.5)，这是 PADPQ deformable sampling 的代价。

GSPB+PADPQ 组合完整趋势:

| Epoch | GSPB+PADPQ | OA-SD | delta mAP |
|-------|------|------|------|
| 70 | 62.3 | 61.8 | +0.5 |
| 80 | 62.8 | 62.0 | +0.8 |
| 90 | 63.6 | 62.4 | +1.2 |
| 100 | 63.9 | 63.0 | +0.9 |
| 110 | 64.1 | 63.1 | +1.0 |
| **120** | **64.2** | **63.2** | **+1.0** |

**后半段持续 +0.8~+1.2 优势——最稳定的正向创新！**

### MaxSim Hybrid 测试

**GSPB+PADPQ + MaxSim: 64.5/75.2**

| Method | eq mAP/R1 | maxsim mAP/R1 | gain |
|--------|------|------|------|
| OA-SD | 63.2/75.4 | 64.2/77.1 | +1.0/+1.7 |
| GSPB | 62.9/74.3 | **64.6/76.0** | +1.7/+1.7 |
| **GSPB+PADPQ** | **64.2/74.9** | **64.5/75.2** | +0.3/+0.3 |

MaxSim gain 仅 +0.3 (PADPQ 破坏了 cross-image keypoint consistency)。
GSPB-only 仍是 MaxSim 最佳 (64.6)。
**但 GSPB+PADPQ 是 equal_concat 最佳 (64.2, +1.0 vs OA-SD)。**
