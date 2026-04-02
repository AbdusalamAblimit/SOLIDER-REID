# exp226 Tiny + 2-Stage Keypoint Fusion (Zero-Init) + OA-SD 监控

配置: 基于 pose_psg_gcn_paa_roa.yml + OA-SD + PLBOA + POSE_MULTI_SCALE_KP
修复: kamp_s2_proj 零初始化 (vs exp224 Kaiming init → -2.5% vs OA-SD)
对照: exp224 (random init): 60.7/73.0, exp191 OA-SD: 63.2/75.4

## 检查点

### [20:27] 检查点 #1

刚启动。exp225 (GSPB+PADPQ) 在远程跑。
**决策**: 等 ep10 eval

### [20:32] 检查点 #2

ep4. tri_part=0.856 (vs exp224 ep4 tri_part~5.4 — 6x 更低！)
零初始化 working: Part triplet loss 从更好的起点开始。
exp225 ep50: 60.2% (≈ PADPQ 60.3, 无叠加效果).
ep10 eval ~15min.
**决策**: 继续

### [20:37] 检查点 #3

ep8. Acc=0.200. ep10 eval ~3min.
**决策**: 等 eval

### [20:41] 检查点 #4 — ep10

**ep10: 36.2/45.4** (vs OA-SD 34.3/46.8 = +1.9/-1.4)
vs exp224 (random-init) ep10: 36.6/49.0 — 零初始化反而略低！
但 exp224 在 ep40+ 恶化到 -3.6%。零初始化应该更稳定。
**决策**: 继续看趋势

### [20:48] 检查点 #5

ep15. ep20 eval ~7min.
exp225 ep60: 60.2% (flat from ep50). GSPB+PADPQ 无叠加效果。
**决策**: 继续

### [20:55] 检查点 #6

ep20 done. eval running.
**决策**: 等 eval

### [20:56] 检查点 #7 — ep20

**ep20: 47.1/58.4** (vs OA-SD 46.0/58.0 = +1.1/+0.4)
vs exp224 (random) ep20: 47.0/59.4 — 几乎一致。
真正差异将在 ep40+ 出现（exp224 在那里恶化到 -3.6%）。
**决策**: 继续

### [21:03] 检查点 #8

exp226 ep~26. exp225 ep69.
**决策**: 继续

### [21:09] 检查点 #9

exp226 ep30 开始. eval ~2min.
**exp225 ep70: 62.3%** — GSPB+PADPQ 在 ep70 超过所有单独方法 (+0.5 vs OA-SD)!
**决策**: 等 eval

### [21:11] 检查点 #10 — ep30

**ep30: 52.1/64.8** (vs OA-SD 50.6/61.7 = +1.5/+3.1!)

| Epoch | zero-init | random-init | OA-SD | delta zero vs OA-SD |
|-------|------|------|------|------|
| 10 | 36.2/45.4 | 36.6/49.0 | 34.3/46.8 | +1.9/-1.4 |
| 20 | 47.1/58.4 | 47.0/59.4 | 46.0/58.0 | +1.1/+0.4 |
| 30 | 52.1/64.8 | 51.4/63.8 | 50.6/61.7 | +1.5/+3.1 |

零初始化在 ep30 优于随机初始化 (+0.7/+1.0)。
**R1 +3.1 vs OA-SD — 非常强！**
关键: ep40-60 是否避免了 exp224 的 -3.6 dip?
**决策**: 继续，密切关注 ep40

### [21:18] 检查点 #11

exp226 ep~36. exp225 ep75.
**决策**: 继续

### [21:24] 检查点 #12

ep40 mid. eval ~1min.
**决策**: 等 eval — 这是关键 checkpoint (exp224 在 ep40 开始恶化)

### [21:26] 检查点 #13 — ep40 ✓

**ep40: 56.1/68.6** (vs OA-SD 57.2/69.2 = -1.1/-0.6)

| Epoch | zero-init | random-init | OA-SD |
|-------|------|------|------|
| 30 | 52.1/64.8 | 51.4/63.8 | 50.6/61.7 |
| 40 | 56.1/68.6 | 55.8/68.2 | 57.2/69.2 |

**ep40 dip 比 exp224 温和！** (-1.1 vs exp224 -1.4)。
exp224 从这里开始恶化到 -3.6 at ep50。零初始化应该更稳定。
**决策**: 继续！ep50 是最关键的 checkpoint。

### [21:32] 检查点 #14

exp226 ep45. ep50 eval ~7min.
**exp225 ep80: 62.8%** (vs OA-SD 62.0 = +0.8). GSPB+PADPQ 稳定正向。
**决策**: 等 eval

### [21:40] 检查点 #15 — ep50

**ep50: 56.5/68.7** (vs OA-SD 59.0/70.6 = -2.5/-1.9)

| Epoch | zero-init | random-init | OA-SD |
|-------|------|------|------|
| 40 | 56.1/68.6 | 55.8/68.2 | 57.2/69.2 |
| 50 | **56.5/68.7** | **55.4/67.6** | **59.0/70.6** |

**零初始化避免了 ep50 collapse!** 56.1→56.5 (上升) vs exp224 55.8→55.4 (下降)。
但仍落后 OA-SD -2.5。2-stage fusion 本身可能仍是负效果。
**决策**: 继续到 final

### [21:55] 检查点 #16 — ep60

**ep60: 58.3%** (vs exp224 57.3 = +1.0, vs OA-SD 60.6 = -2.3)
零初始化持续优于随机初始化。Gap vs OA-SD 在缩小 (-2.5→-2.3)。
exp225 (GSPB+PADPQ) ep90: 63.6% (+1.2 vs OA-SD — 持续扩大!)
**决策**: 继续

### [22:03] 检查点 #17

exp226 ep66. exp225 ep96. 两台都约 1h to final.
**决策**: 继续

### [22:09] 检查点 #18

exp226 ep70 almost done. eval ~1min. exp225 ep~99.
**决策**: 等 evals

### [22:10] 检查点 #19 — ep70

**ep70: 59.1/71.7** (vs OA-SD 61.8/73.1 = -2.7/-1.4)
vs exp224 random-init ep70: 58.2/71.1 → +0.9/+0.6
仍落后 OA-SD ~2.5。2-stage fusion 整体仍是负效果。
**决策**: 继续到 final

### [22:15] 检查点 #20

exp226 ep~76. exp225 ep100: 63.9% (+0.9 vs OA-SD).
**决策**: 继续

### [22:26] 检查点 #21

exp226 ep~80 (eval soon). exp225 ep~106.
**决策**: 等 evals

### [22:25] 检查点 #22 — ep80

**ep80: 60.0/72.3** (vs OA-SD 62.0/73.8 = -2.0/-1.5)
Gap 在缩小 (-2.7 → -2.0)。零初始化持续 +0.9 ahead of random-init。
预计 final ~61.5-62 (vs OA-SD 63.2, exp224 60.7)。
**决策**: 继续到 final

### [22:40] 检查点 #23 — ep90

**ep90: 61.1%** (vs OA-SD 62.4 = -1.3, vs exp224 60.2 = +0.9)
Gap vs OA-SD: -2.7 → -2.0 → -1.3 — 持续缩小！
exp225 ep116. final ~6min.
**决策**: 继续

### [22:55] 检查点 #24 — ep100

**ep100: 61.4/73.8** (vs OA-SD 63.0/75.0 = -1.6/-1.2)
**exp225 FINAL: 64.2/74.9** (+1.0/-0.5 vs OA-SD)!
**exp225 + MaxSim: 64.5/75.2** (+0.3 gain)
**决策**: 继续 exp226 到 final

### [23:15] 检查点 #25

ep114. ep110: 61.6/74.2. final ~8min.
**决策**: 等 FINAL

### [23:21] 检查点 #26

ep119. final ~3min.
**决策**: 等 FINAL

## exp226 FINAL RESULTS

**exp226 (zero-init 2-stage fusion + OA-SD) FINAL: 61.6/74.3**

| 方法 | mAP | R1 |
|------|------|------|
| **exp226 (zero-init)** | 61.6% | 74.3% |
| exp224 (random-init) | 60.7% | 73.0% |
| exp191 OA-SD-only | 63.2% | 75.4% |

**零初始化 vs 随机初始化: +0.9/+1.3 — 确认零初始化更好。**
**但 2-stage fusion 整体仍落后 OA-SD -1.6/-1.1。**
多尺度 keypoint fusion (当前实现) 不如单尺度 + OA-SD。
