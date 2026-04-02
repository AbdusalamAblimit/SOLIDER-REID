# exp218 Tiny + GCN+PAA+CE+OA-SD + PACI (Part Prototype Bank) 监控

配置: Tiny GCN+PAA+CE+OA-SD + PACI (per-ID per-part prototype bank + consistency loss)
对照: exp191 (Tiny + OA-SD): 64.4/76.5, exp187 (Tiny + SupCon 3v): 64.9/76.6

**创新点**: Per-identity per-part momentum prototype bank
- 训练时记忆每个 ID 的 body part appearance
- Consistency loss: 拉向自身 prototype, 推离其他 ID

## 检查点

### [06:08] 检查点 #1

本地 exp218 (w=0.5): ep3. PACI warmup (ep<5, consistency loss 未激活).
远程 exp218b (w=1.0): ep1.
正常启动。PACI 将在 ep6 开始激活。
**决策**: 等 ep10 eval (PACI 应在 ep6 开始影响训练)

### [06:14] 检查点 #2 — PACI 激活! 🔥

**ep6: PACI consistency loss 已激活!**
- `paci: 0.253` (triplet loss 有效)
- `paci_pos: 0.911` (与自身 ID prototype 相似度高)
- `paci_neg: 0.864` (与其他 ID prototype 相似度)
- `pos - neg = 0.047` → margin=0.3 未满足 → loss=0.253 → 有效梯度!

**与 detached PKC/MST 的关键区别:**
PACI 的 prototype bank 积累了跨 batch 的 identity-specific 信息。
即使 kp_feats 是 detached 的, consistency loss 提供了 NEW information
(来自 prototype bank 的跨 batch 记忆), 而不是 self-referential。
ep10 eval ~10min。
**决策**: 等 eval

### [06:20] 检查点 #3

ep10 mid. paci=0.250, pos-neg=0.050 (improving from 0.047). eval ~2min.
**决策**: 等 eval

### [06:22] 检查点 #4 — ep10 ✓

**ep10: 39.2/52.5** (vs baseline 38.2/51.3 = **+1.0/+1.2**)
vs OERL ep10: 38.9/52.3 → PACI 略好 (+0.3/+0.2)

PACI 在 ep10 表现最好！prototype bank 已开始发挥作用。
**关键**: ep20+ 随着 bank 积累更多 prototypes, 优势是否扩大？
**决策**: 继续！

### [06:29] 检查点 #5

ep15. paci=0.228 (下降中!), pos-neg=0.072 (从 0.047→0.050→0.072 持续扩大!)
GCN 正在学习区分不同 identity 的 part prototypes!
ep20 eval ~14min.
**决策**: 等 eval

### [06:35] 检查点 #6

ep19. paci=0.201, pos-neg=0.099 (持续扩大! 0.047→0.072→0.099)
GCN features 越来越 identity-specific! ep20 eval ~3min.
exp218b (w=1.0) ep10: 39.2 (= exp218 w=0.5).
**决策**: 等 eval

### [06:39] 检查点 #7 — ep20 🔥

**ep20: 48.4%!** (vs baseline 46.8 = **+1.6%!**)

| Epoch | PACI | baseline | delta |
|-------|------|------|------|
| 10 | 39.2 | 38.2 | +1.0 |
| 20 | 48.4 | 46.8 | **+1.6** |

**优势在扩大！** (+1.0 → +1.6)
PACI 的 prototype bank 积累了更多 identity-specific 信息 → 更强的信号！
这是第一个在 Tiny 上显示出持续增长优势的创新！
**决策**: 继续！非常有前途！

### [06:45] 检查点 #8

ep25. paci=0.154, pos-neg=**0.147** (0.047→0.072→0.099→0.147 — 持续扩大!)
GCN keypoint features 正在快速分化不同 identity 的 prototypes。
ep30 eval ~8min.
**决策**: 等 eval

### [06:51] 检查点 #9

ep28. paci=0.131, pos-neg=0.170. ep30 eval ~3min.
**决策**: 等 eval

### [06:56] 检查点 #10 — ep30 🔥🔥

**ep30: 54.3%!** (vs baseline 52.2 = **+2.1%!!**)

| Epoch | PACI | baseline | delta |
|-------|------|------|------|
| 10 | 39.2 | 38.2 | +1.0 |
| 20 | 48.4 | 46.8 | +1.6 |
| 30 | **54.3** | **52.2** | **+2.1** |

**优势持续扩大!! +1.0 → +1.6 → +2.1!**
PACI prototype bank 越积累越有效——这正是 PACI 设计的核心价值。
如果趋势持续: final 可能达到 **63-65% (+3-4% vs baseline)!**
甚至可能超过 exp191 OA-SD-only (64.4)!

**这是目前最有前途的创新方向！**
**决策**: 继续！密切监控！

### [07:02] 检查点 #11

ep33. paci=0.111, pos-neg=0.191 (持续扩大！). ep40 eval ~10min.
**决策**: 等 eval

### [07:07] 检查点 #12

ep36. paci=0.101, pos-neg=0.202 (突破 0.2！). ep40 eval ~6min.
**决策**: 等 eval

### [07:13] 检查点 #13

ep40 mid. paci=0.088, pos-neg=0.217. eval ~2min.
**决策**: 等 eval

### [07:15] 检查点 #14 — ep40

**ep40: 56.1%** (vs baseline ~55.5 = +0.6)

| Epoch | PACI | baseline | delta |
|-------|------|------|------|
| 10 | 39.2 | 38.2 | +1.0 |
| 20 | 48.4 | 46.8 | +1.6 |
| 30 | 54.3 | 52.2 | +2.1 |
| 40 | 56.1 | ~55.5 | +0.6 |

**优势缩小！** 与 OERL/OA-SD fix 类似的震荡模式。
但 PACI 的 consistency loss 是在 detached GCN 上的，不应该干扰 backbone。
可能是 paci loss 本身在 ep40 开始 saturate (paci=0.088 接近 0)。
**决策**: 继续看趋势

### [07:23] 检查点 #15

ep45. paci=0.081, pos-neg=0.226. ep50 eval ~8min.
exp218b (w=1.0) ep20: 47.6% (vs exp218 w=0.5 ep20: 48.4% — w=0.5 更好)。
**决策**: 等 eval

### [07:28] 检查点 #16

ep48. paci=0.079, pos-neg=0.229. ep50 eval ~3min.
**决策**: 等 eval

### [07:32] 检查点 #17 — ep50

**ep50: 58.5%** (vs baseline ~57.5 = +1.0)

| Epoch | PACI | baseline | delta |
|-------|------|------|------|
| 10 | 39.2 | 38.2 | +1.0 |
| 20 | 48.4 | 46.8 | +1.6 |
| 30 | 54.3 | 52.2 | +2.1 |
| 40 | 56.1 | ~55.5 | +0.6 |
| 50 | 58.5 | ~57.5 | +1.0 |

优势回升到 +1.0。震荡但平均正。
vs OERL ep50: 58.3 → PACI 略好。
预计 final ~62-63% (vs baseline 60.7, vs OA-SD 64.4)。
**决策**: 继续

### [07:40] 检查点 #18

exp218 ep55. paci=0.071, pos-neg=0.239.
exp218b ep34. paci=0.097, pos-neg=0.207.
两台正常。exp218 ep60 eval ~12min.
**同时开始设计 PACI Phase 2 (test-time prototype completion)。**
**决策**: 继续

### [07:46] 检查点 #19

ep59. paci=0.072. ep60 eval ~2min.
**决策**: 等 eval

### [07:49] 检查点 #20 — ep60 ⚠️

**ep60: 58.4%** — 比 ep50 (58.5) 还低! 停滞/轻微下降!

| Epoch | PACI | OERL | baseline |
|-------|------|------|------|
| 50 | 58.5 | 58.3 | ~57.5 |
| 60 | **58.4** | **59.5** | ~59.0 |

PACI 在 ep60 被 OERL 反超！可能是 paci loss=0.072 已 saturate，
triplet margin 几乎满足，梯度接近零。
**决策**: 继续到 final 确认

### [07:55] 检查点 #21

ep64. paci=0.067. ep70 eval ~9min.
exp218b (w=1.0) ep30: 53.4 (vs w=0.5 ep30: 54.3 — w=0.5 更好).
**决策**: 等 eval

### [08:01] 检查点 #22

ep68. paci=0.065. ep70 eval ~3min. ETA 1h24m.
**决策**: 等 eval

### [08:07] 检查点 #23 — ep70

**ep70: 59.8%** (recovered from ep60 dip!)

| Epoch | PACI | OERL | baseline |
|-------|------|------|------|
| 50 | 58.5 | 58.3 | ~57.5 |
| 60 | 58.4 | 59.5 | ~59.0 |
| 70 | 59.8 | 60.5 | ~59.5 |

PACI +0.3 vs baseline. OERL was +1.0 at ep70。
预计 final ~62-63% (类似 OERL 62.2)。
**决策**: 继续到 final

### [08:12] 检查点 #24

exp218b (w=1.0) ep40: 55.4 (vs w=0.5 ep40: 56.1 — w=0.5 领先 +0.7)。
Higher PACI weight 略差。
exp218 ep~75. ETA ~1h15m.
**决策**: 继续

### [08:18] 检查点 #25

ep78. paci=0.062. ep80 eval ~4min.
**决策**: 等 eval

### [08:24] 检查点 #26 — ep80

**ep80: 60.7%** (= OERL ep80 60.7, = baseline final 60.7)

| Epoch | PACI | OERL |
|-------|------|------|
| 70 | 59.8 | 60.5 |
| 80 | 60.7 | 60.7 |

PACI 和 OERL 在 ep80 完全一致！
预计 final ~62-63% (与 OERL 62.2 类似)。
**决策**: 继续到 final

### [08:29] 检查点 #27

ep84. ep90 eval ~9min. ETA ~1h.
**决策**: 等 final

### [08:35] 检查点 #28

ep88. paci=0.064. ep90 eval ~3min. ETA ~50min.
**决策**: 等 eval

### [08:41] 检查点 #29 — ep90

**ep90: 61.0%** (vs OERL 61.2 — 几乎一致)
ETA 48min. 预计 final ~62%.
**决策**: 继续到 final

### [08:50] 检查点 #30

exp218 ~ep95. ep100 eval ~15min.
远程: exp218b 已终止，改跑 exp219 (PACI-only, no OA-SD). ep4.
**决策**: 继续

### [08:56] 检查点 #31

ep100 mid. eval ~2min.
**决策**: 等 eval

### [08:58] 检查点 #32 — ep100

**ep100: 61.7%** (vs OERL 62.0 — PACI 略低)
ETA ~33min. 预计 final ~62.0-62.5%.
**决策**: 继续到 final

### [09:05] 检查点 #33

ep105. final ~25min.
**决策**: 等 final

### [09:11] 检查点 #34

ep109. ep110 eval ~2min, final ~15min.
**决策**: 等 final

### [09:17] 检查点 #35

ep112. ep110: 61.9/74.5. final ~13min.
**决策**: 等 FINAL

### [09:23] 检查点 #36

ep116. final ~6min.
**决策**: 等 FINAL

### [09:28] 检查点 #37

ep119. FINAL ~2min!
**决策**: 等 FINAL

## exp218 FINAL RESULTS

**exp218 (PACI w=0.5 + OA-SD) FINAL: 61.9/74.2**

| 方法 | mAP | R1 |
|------|------|------|
| exp030a baseline | 60.7% | 72.6% |
| **exp218 PACI+OA-SD** | **61.9%** | **74.2%** |
| exp217 OERL+OA-SD | 62.2% | 75.2% |
| exp191 OA-SD only | 64.4% | 76.5% |

**PACI+OA-SD: +1.2% vs baseline, -2.5% vs OA-SD-only.**
PACI 与 OERL 在最终结果上几乎一致 (61.9 vs 62.2)。
两者都低于 OA-SD-only，confirm 额外 loss 干扰 OA-SD。

**下一步**: exp219 (PACI-only, no OA-SD) 正在远程跑。
如果 PACI-only > baseline (60.7)，PACI 仍有论文价值。

### MaxSim Hybrid 对比 (Tiny) 🔥🔥🔥

**关键发现: 所有方法 + MaxSim 都收敛到 ~64.1-64.3%！**

| 方法 | equal_concat | maxsim_hybrid |
|------|------|------|
| OA-SD-only (exp191) | **64.4** | 64.2 |
| OERL+OA-SD (exp217) | 62.2 | 64.3 |
| PACI+OA-SD (exp218) | 61.9 | 64.1 |

**MaxSim 在 OA-SD 模型上无效 (-0.2%)！OA-SD 的 global feature 已经最优。**
**OERL/PACI 的模型 global 较弱 → MaxSim 弥补 → 最终趋同。**
**~64.2% 是 GCN+PAA on Tiny 的 MaxSim ceiling。**
