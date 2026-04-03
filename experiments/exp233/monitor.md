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
