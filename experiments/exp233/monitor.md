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
