# exp233 Tiny + Per-Body-Part Independent Training + OA-SD 监控

配置: Tiny + GCN+PAA+OA-SD+PLBOA+ROA + POSE_GCN_PER_PART=True
对照: exp191 (Tiny OA-SD, pooled GCN): 63.2/75.4
**创新**: 17 keypoints → 6 body parts, 每个有独立 BN+classifier (KPR-inspired)

## 检查点

### [21:00] 检查点 #1

远程启动成功。ep1 done. id_part=6.711, tri_part=0.789。
id_part 是 7 个 part classifiers (pooled + 6 body parts) 的平均 CE loss。
ETA ~4h14m。
**决策**: 等 ep10 eval

### [21:03] 检查点 #2

ep3. id_part=6.544 (下降中). ep10 eval ~14min。
**决策**: 等 ep10 eval

### [21:06] 检查点 #3

ep4. id_part=6.384 (稳定下降). ep10 eval ~12min。
**决策**: 等 ep10 eval

### [21:09] 检查点 #4

ep5. Acc=0.160, id_part=6.210. ep10 eval ~10min。
**决策**: 等 ep10 eval

### [21:11] 检查点 #5

ep7. id_part=6.122. ep10 eval ~6min。
**决策**: 等 ep10 eval

### [21:14] 检查点 #6

ep8. id_part=6.034. ep10 eval ~4min。
**决策**: 等 ep10 eval

### [21:21] 检查点 #7 — ep10

**ep10: 34.0/49.2** (vs exp191 34.3/46.8 = **-0.3/+2.4**)

mAP 基本持平 (-0.3), **R1 +2.4!** Per-part 训练在早期就显示 R1 优势。
7 个独立 classifier 学习较慢 (id_part 5.82 vs baseline ~5.0), 但特征多样性更高。
远高于 25% 早停线。
ETA ~3h50m。
**决策**: 继续！R1 +2.4 是积极信号

### [21:23] 检查点 #8

ep11. id_part=5.714. ep20 eval ~18min。
**决策**: 等 ep20 eval

### [21:26] 检查点 #9

ep13. id_part=5.763 (7 classifiers 收敛中). ep20 eval ~14min。
**决策**: 等 ep20 eval

### [21:29] 检查点 #10

ep14. id_part=5.546. ep20 eval ~12min。
**决策**: 等 ep20 eval

### [21:31] 检查点 #11

ep15. id_part=5.435, id_global=4.729. ep20 eval ~10min。
注意 id_part > id_global: 7 个独立 part classifiers 学习比 global 慢。
**决策**: 等 ep20 eval

### [21:34] 检查点 #12

ep17. id_part=5.429, id_global=4.478. ep20 eval ~6min。
**决策**: 等 ep20 eval

### [21:37] 检查点 #13

ep18. id_part=5.272. ep20 eval ~4min。
**决策**: 等 ep20 eval

### [21:44] 检查点 #14 — ep20

**ep20: 44.7/58.8** (vs exp191 46.0/58.0 = **-1.3/+0.8**)

| Epoch | exp233 mAP/R1 | exp191 mAP/R1 | delta |
|-------|------|------|------|
| 10 | 34.0/49.2 | 34.3/46.8 | -0.3/+2.4 |
| **20** | **44.7/58.8** | **46.0/58.0** | **-1.3/+0.8** |

mAP 轻微落后 (-1.3), R1 仍正向 (+0.8)。
per-part 的 7 个 classifiers 学习更慢 (id_part 5.0 vs baseline ~3.5 at ep20)。
这解释了 mAP 落后 — 需要更多训练时间让 part classifiers 收敛。
**关键**: 如果 mAP 在 ep40-60 追上, 而 R1 优势持续, 那就是正向结果。
ETA ~3h30m。
**决策**: 继续！R1 持续正向是好信号

### [21:45] 检查点 #15

ep21. id_part=4.993 (开始接近 5.0 以下). ep30 eval ~18min。
**决策**: 等 ep30 eval

### [21:48] 检查点 #16

ep22. ETA ~3h25m. ep30 eval ~16min。
**决策**: 等 ep30 eval

### [21:50] 检查点 #17

ep24. id_global=2.914 (快速下降), id_part=4.953 (仍然慢). ep30 eval ~12min。
Part classifiers 学习速度是 global 的约 1/3。
**决策**: 等 ep30 eval

### [21:53] 检查点 #18

ep25. id_global=2.529, id_part=4.752. ep30 eval ~10min。
**决策**: 等 ep30 eval

### [21:56] 检查点 #19

ep26. id_global=2.158, id_part=4.516. ep30 eval ~8min。
**决策**: 等 ep30 eval

### [22:01] 检查点 #20

ep29. id_global=2.077, id_part=4.645. ep30 eval ~2min。
**决策**: 等 ep30 eval

### [22:06] 检查点 #21 — ep30

**ep30: 49.4/62.9** (vs exp191 50.6/64.8 = **-1.2/-1.9**)

| Epoch | exp233 mAP/R1 | exp191 mAP/R1 | delta |
|-------|------|------|------|
| 10 | 34.0/49.2 | 34.3/46.8 | -0.3/+2.4 |
| 20 | 44.7/58.8 | 46.0/58.0 | -1.3/+0.8 |
| **30** | **49.4/62.9** | **50.6/64.8** | **-1.2/-1.9** |

R1 优势逆转！从 ep10 +2.4 到 ep30 -1.9。
per-part classifiers 仍在学习中 (id_part=4.2 vs baseline 已收敛)。
可能需要更多 epoch 让 7 classifiers 追上。
ETA ~3h11m。
**决策**: 继续，观察 ep50+ 是否追上

### [22:08] 检查点 #22

ep31. ETA ~3h8m. 下次检查 ep40 eval (~19min)。
**决策**: 继续

### [22:11] 检查点 #23

ep33 附近. ep40 eval ~14min。
**决策**: 等 ep40 eval

### [22:14] 检查点 #24

ep35. id_part=4.213. ep40 eval ~10min。
**决策**: 等 ep40 eval

### [22:19] 检查点 #25

ep36. ETA ~3h2m. ep40 eval ~8min。
**决策**: 等 ep40 eval

### [22:22] 检查点 #26

ep38. id_part=4.166. ep40 eval ~4min。
**决策**: 等 ep40 eval

### [22:25] 检查点 #27

ep39. id_part=4.017. ep40 eval ~2min。
**决策**: 等 ep40 eval

### [22:29] 检查点 #28 — ep40

**ep40: 53.2/65.6** (vs exp191 55.1/68.7 = **-1.9/-3.1**)

| Epoch | exp233 mAP/R1 | exp191 mAP/R1 | delta |
|-------|------|------|------|
| 10 | 34.0/49.2 | 34.3/46.8 | -0.3/+2.4 |
| 20 | 44.7/58.8 | 46.0/58.0 | -1.3/+0.8 |
| 30 | 49.4/62.9 | 50.6/64.8 | -1.2/-1.9 |
| **40** | **53.2/65.6** | **55.1/68.7** | **-1.9/-3.1** |

Gap 扩大到 -1.9/-3.1。Per-part classifiers 仍未收敛 (id_part=4.0 vs baseline ~1.0)。
7 个独立 classifier 学习效率太低 — 120 epoch 可能不够。
如果 ep60+ 不回升，per-part training 在当前配置下负面。
ETA ~2h50m。
**决策**: 继续，但趋势不乐观

### [22:33] 检查点 #29

ep42. id_part=3.900. ep50 eval ~16min。
**决策**: 等 ep50 eval

### [22:36] 检查点 #30

ep43. id_part=3.610 (持续下降但仍然高). ep50 eval ~14min。
**决策**: 等 ep50 eval

### [22:39] 检查点 #31

ep46. id_part=3.903. ep50 eval ~8min。
**决策**: 等 ep50 eval

### [22:44] 检查点 #32

ep47. id_part=3.644. ep50 eval ~5min。
**决策**: 等 ep50 eval

### [22:49] 检查点 #33

ep50 iter20. eval ~2min。
**决策**: 等 ep50 eval

### [22:53] 检查点 #34 — ep50

**ep50: 55.6/68.3** (vs exp191 57.6/71.0 = **-2.0/-2.7**)

| Epoch | exp233 mAP/R1 | exp191 mAP/R1 | delta |
|-------|------|------|------|
| 10 | 34.0/49.2 | 34.3/46.8 | -0.3/+2.4 |
| 20 | 44.7/58.8 | 46.0/58.0 | -1.3/+0.8 |
| 30 | 49.4/62.9 | 50.6/64.8 | -1.2/-1.9 |
| 40 | 53.2/65.6 | 55.1/68.7 | -1.9/-3.1 |
| **50** | **55.6/68.3** | **57.6/71.0** | **-2.0/-2.7** |

持续落后 ~-2.0 mAP, -2.7 R1。
7 per-part classifiers 的 id_part=3.6 (baseline 的 pooled 版在 ep50 已 <1.0)。
**预计 final 可能 -2~-3% 低于 baseline。**
ETA ~2h30m。
**决策**: 继续到 final 收集完整证据

### [22:57] 检查点 #35

ep53. id_part=3.688. ep60 eval ~14min。
**决策**: 等 ep60 eval

### [23:00] 检查点 #36

ep57-60. id_part=3.474. ep60 eval imminent。
**决策**: 等 ep60 eval

### [23:16] 检查点 #37 — ep60

**ep60: 56.1/69.4** (vs exp191 60.6/73.9 = **-4.5/-4.5**)

| Epoch | exp233 mAP/R1 | exp191 mAP/R1 | delta |
|-------|------|------|------|
| 10 | 34.0/49.2 | 34.3/46.8 | -0.3/+2.4 |
| 30 | 49.4/62.9 | 50.6/64.8 | -1.2/-1.9 |
| 50 | 55.6/68.3 | 57.6/71.0 | -2.0/-2.7 |
| **60** | **56.1/69.4** | **60.6/73.9** | **-4.5/-4.5** |

**Gap 大幅扩大！** -2.0 → -4.5。Per-part 训练严重落后。
7 个独立 classifiers 收敛太慢，detached features 上的 per-body-part 判别力不足。
预计 final ~-3 到 -5% 低于 baseline。
ETA ~2h。
**决策**: 继续到 final 但已经确认负面

### [23:21] 检查点 #38

ep63. id_part=3.338. ep70 eval ~14min, ep80 ~28min。
结论已清楚：per-part 在当前配置下负面。继续跑完收集完整数据。
**决策**: 继续到 final

### [23:24] 检查点 #39

ep68-70. id_part 降到 <3.0。

### [23:39] 检查点 #40 — ep70

**ep70: 57.9/70.4** (vs exp191 61.4/74.6 = **-3.5/-4.2**)
Gap 从 ep60 -4.5 略缩小到 -3.5 mAP, 但 R1 仍 -4.2。
ETA ~1h47m。
**决策**: 继续到 final

### [23:58] 检查点 #41

ep79. ETA ~1h26m. ep80 eval ~2min。
**决策**: 等 ep80 eval

### [00:02] 检查点 #42 — ep80

**ep80: 58.1/71.0** (vs exp191 62.0/74.4 = **-3.9/-3.4**)

| Epoch | delta mAP | delta R1 |
|-------|------|------|
| 10 | -0.3 | +2.4 |
| 30 | -1.2 | -1.9 |
| 50 | -2.0 | -2.7 |
| 60 | -4.5 | -4.5 |
| 70 | -3.5 | -4.2 |
| **80** | **-3.9** | **-3.4** |

Per-part training 确认严重负面。预计 final ~-3%.
ETA ~1h24m。
**决策**: 继续到 final 收集完整数据

### [00:07] 检查点 #43

ep83. id_part=2.900. ETA ~1h15m。
正在搜索范式级别创新方向（agent 后台运行）。
**决策**: 继续到 final

### [00:13] 检查点 #44

ep86. id_part=2.994. ep90 eval ~8min。ETA ~1h12m。
范式级创新搜索完成 — Feature-Space Diffusion Completion 被选为下一方向。
**决策**: 继续到 final

### [00:23] 检查点 #45

ep90 iter200. eval imminent。id_part=2.609。
**决策**: 等 ep90 eval

### [00:24] 检查点 #46 — ep90

**ep90: 59.6/72.6** (vs exp191 62.8/75.1 = **-3.2/-2.5**)
Gap 从 ep80 -3.9 缩小到 -3.2。part classifiers 正在追赶。
ETA ~1h。
**决策**: 继续到 final

### [00:30] 检查点 #47

ep93. id_part=2.871. ep100 eval ~14min。ETA ~56min。
**决策**: 继续到 final

### [00:36] 检查点 #48

ep96. id_part=3.015. ep100 eval ~8min。ETA ~50min。
exp235 (FSDC) 已在本地启动。
**决策**: 继续到 final

### [00:48] 检查点 #49 — ep100

**ep100: 60.2/72.4** (vs exp191 63.2/75.4 = **-3.0/-3.0**)
Gap 从 ep90 -3.2 缩小到 -3.0。仍然显著负面。
ETA ~43min。
**决策**: 继续到 final

### [01:04] 检查点 #50

ep108. ETA ~26min. ep110 eval ~4min。
**决策**: 等 ep110 eval

### [01:08] 检查点 #51

ep110. eval imminent. ETA ~22min。
**决策**: 等 ep110 eval

### [01:11] 检查点 #52 — ep110

**ep110: 60.4/73.3** (vs exp191 63.2/75.4 = **-2.8/-2.1**)
Gap 仍 -2.8 mAP。预计 final ~60.5/73.5 (-2.7/-1.9)。
ETA ~22min。
**决策**: 继续到 final
