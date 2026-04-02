# exp217 Tiny + GCN+PAA+CE+OA-SD + OERL (Part Occlusion Invariance) 监控

配置: Tiny GCN+PAA+CE+OA-SD + OERL (feature-map-level soft occlusion + per-part invariance)
对照: exp191 (Tiny + OA-SD): 64.4/76.5, exp187 (Tiny + SupCon 3v): 64.9/76.6

**创新点**: 
- Feature-map-level pose-guided occlusion (不需要第二次 forward)
- Per-part cosine invariance loss on NON-detached backbone features
- 与 CE 方向一致 (都是 push same-ID together)

## 检查点

### [02:46] 检查点 #1

本地 exp217 (weight=1.0): ep3. oerl=0.012, oa_sd=0.325. ETA 2h44m.
远程 exp217b (weight=0.5): ep2. oerl=0.016, oa_sd=0.439. ETA 3h57m.

**oerl loss 非常低 (0.01)!** 可能 heatmap masking 太温和。
oerl_nv=7.2 (50% occlusion ratio → 8.5 expected visible, 实际 7.2 合理)。
**决策**: 等 ep10 eval

### [02:52] 检查点 #2

exp217 ep7. oerl=0.006 (继续下降), id_global=6.308, Acc=0.140.
exp217b ep5. oerl=0.010.
OERL loss 非常低——特征几乎已经对 soft occlusion invariant。
ep10 eval ~8min (local), ~15min (remote).
**决策**: 等 ep10 eval

### [02:58] 检查点 #3 — ep10

**exp217 (OERL w=1.0) ep10: 38.9/52.3**
vs exp030a baseline ep10: 38.2/51.3 (+0.7/+1.0)

OERL loss 仅 0.006 — 几乎不影响训练。
但 ep10 比纯 baseline 略高。需要与 OA-SD-only 对比。
**决策**: 继续到 ep30+ 看趋势

### [03:04] 检查点 #4

exp217 ep15. oerl=0.003 (继续下降). id_global=4.772.
exp217b ep10 done, eval running.
**决策**: 等 ep217b eval + 继续 exp217

### [11:05] 检查点 #5

exp217b (w=0.5) ep10: 37.2/50.9 — 低于 exp217 (w=1.0) 38.9/52.3。
两者都低于 baseline OA-SD (~40 at ep10)。

**OERL loss ~0.003 — 基本是 no-op。**
Swin 的 self-attention 让 feature map 高度 spatially mixed，
soft heatmap masking 几乎不改变临近 keypoint 的特征。

需要更 aggressive 的 occlusion 机制（如 token pruning 或 hard spatial zeroing）。
**决策**: 继续跑完看 final，同时思考更好的方案

### [03:13] 检查点 #6 — ep20

**exp217 ep20: 47.7/60.5** (vs baseline ep20: 46.8/60.9 = +0.9/-0.4)
OERL 仍然 ~0.003。几乎无效。继续跑完看 final。
**决策**: 继续

### [03:19] 检查点 #7

ep25. oerl=0.003 (平台). ep30 eval ~10min.
**决策**: 等 eval

### [03:26] 检查点 #8

ep29. oerl=0.003 (flat). ep30 eval ~2min.
**决策**: 等 eval

### [03:28] 检查点 #9 — ep30

**exp217 ep30: 53.8/66.0** (vs baseline 52.2/66.0 = **+1.6/0.0**)

mAP 优势在增长 (ep10 +0.7, ep20 +0.9, ep30 +1.6)!
但需要对比 exp191 (OA-SD only) 才能判断 OERL 的额外贡献。
**决策**: 继续到 ep120 final

### [03:34] 检查点 #10

exp217b (w=0.5) ep20: 48.2/60.9 (vs exp217 w=1.0 ep20: 47.7/60.5 — w=0.5 略好)。
**决策**: 继续

### [03:40] 检查点 #11

exp217 ep38. ep40 eval ~4min. ETA 1h58m.
**决策**: 等 eval

### [03:44] 检查点 #12 — ep40

**exp217 ep40: 56.9/69.1**
mAP 优势持续增长 (+0.7→+0.9→+1.6→...)
**决策**: 继续

### [03:49] 检查点 #13

exp217b (w=0.5) ep30: 53.9 (= exp217 w=1.0 ep30: 53.8 — 一致)。
两个 weight 给出相同结果，confirm OERL loss=0.003 基本无影响。
exp217 ~ep46. exp217b ~ep34.
**决策**: 继续跑完

### [03:56] 检查点 #14

exp217 ep49. oerl=0.002. ep50 eval ~3min.
**决策**: 等 eval

### [03:59] 检查点 #15 — ep50

**exp217 ep50: 58.3%** (vs baseline ~57.5 = +0.8)

| Epoch | OERL | baseline | delta |
|-------|------|------|------|
| 10 | 38.9 | 38.2 | +0.7 |
| 20 | 47.7 | 46.8 | +0.9 |
| 30 | 53.8 | 52.2 | +1.6 |
| 40 | 56.9 | ~55.5 | +1.4 |
| 50 | 58.3 | ~57.5 | +0.8 |

优势在 ep30 达峰后收窄。可能 OERL 主要帮助早中期收敛。
**决策**: 继续到 final

### [04:09] 检查点 #16

exp217 ep57. ETA ~55min.
exp217b ep40. ETA 2h40m.
**决策**: 继续

### [04:15] 检查点 #17 — ep60

**exp217 ep60: 59.5%** (+1.2 from ep50)
Trending toward ~62-63% at ep120. ETA ~50min.
**决策**: 继续

### [04:21] 检查点 #18

ep65. ep70 eval ~7min.
**决策**: 等 eval

### [04:27] 检查点 #19

ep69. ep70 eval ~2min. ETA 1h15m.
**决策**: 等 eval

### [04:30] 检查点 #20 — ep70

**exp217 ep70: 60.5%**
Growth: +1.2 (ep60), +1.0 (ep70) — 放缓。
预计 final ~63-64% (vs exp191 OA-SD-only: 64.4)。
OERL 可能略低于 OA-SD-only — 因为 OERL loss=0.002 基本无效。
**决策**: 继续到 final 确认

### [04:36] 检查点 #21

exp217b (w=0.5) ep50: 58.4% (= exp217 w=1.0 ep50: 58.3%)。
两个 weight 完全一致——confirm OERL 无效。
exp217 ep~78. ep80 eval ~8min.
**决策**: 继续

### [04:41] 检查点 #22

ep78. ep80 eval ~4min.
**决策**: 等 eval

### [04:46] 检查点 #23 — ep80

**exp217 ep80: 60.7%** (仅 +0.2 from ep70)
vs baseline ~60.0 (+0.7), vs OA-SD-only (exp191) ~62% (**-1.3!**)

**OERL + OA-SD 比 OA-SD-only 更差！** 
non-detached OERL 梯度与 OA-SD 竞争。
**决策**: 继续到 final 确认，但预计负结果

### [04:52] 检查点 #24

ep85. final ~32min。
**决策**: 继续

### [04:58] 检查点 #25

ep89. ep90 eval ~2min. ETA 46min.
**决策**: 等 eval

### [05:01] 检查点 #26 — ep90

**ep90: 61.2%** (+0.5 from ep80)
预计 final ~62-63% (vs exp191 OA-SD: 64.4 = **-1.5 to -2.5%**)
OERL v2 是负结果 — non-detached invariance loss 干扰了 OA-SD。
**决策**: 继续到 final

### [05:07] 检查点 #27

ep94. ETA 37min. ep100 eval ~10min.
exp217b ep60: 59.7 (= exp217 59.5).
**决策**: 等 final

### [05:13] 检查点 #28

ep99. ep100 eval ~2min. ETA 31min.
**决策**: 等 eval

### [05:17] 检查点 #29 — ep100

**ep100: 62.0%** (+0.8 from ep90)
预计 final ~63% (vs OA-SD-only 64.4 = **-1.4%**)
**OERL 是负结果。** 但比 baseline 60.7 好 (+1.3)——说明 OERL 有效果但不如 OA-SD。
ETA 28min.
**决策**: 等 final

### [05:23] 检查点 #30

ep105. ep110 eval ~7min. final ~20min.
**决策**: 等 final

### [05:31] 检查点 #31

ep110 mid. eval ~2min. final ~12min.
**决策**: 等 eval

### [05:35] 检查点 #32

ep112. ep110: 62.2%. final ~10min.
**决策**: 等 FINAL

### [05:41] 检查点 #33

ep116. final ~5min.
**决策**: 等 FINAL

## exp217 FINAL RESULTS

**exp217 (OERL w=1.0 + OA-SD) FINAL: 62.2/75.2**

| 方法 | mAP | R1 |
|------|------|------|
| exp030a baseline | 60.7% | 72.6% |
| **exp217 (OERL+OA-SD)** | **62.2%** | **75.2%** |
| exp191 (OA-SD only) | 64.4% | 76.5% |

**结论**: OERL+OA-SD (62.2) < OA-SD-only (64.4) by **-2.2%**。
OERL 的 non-detached invariance loss 干扰了 OA-SD 的训练。
OERL 本身比 baseline 好 (+1.5)，但与 OA-SD 不兼容。

**OERL v2 soft masking 太弱 (loss=0.002)，同时 non-detached 梯度与 OA-SD 竞争。**
需要更 aggressive 的 occlusion mechanism 或完全不同的方法。
