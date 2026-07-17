# exp208 Small + GCN+PAA+CE+OA-SD + 0.5x Global Loss 监控

配置: exp206 + GLOBAL_LOSS_SCALE=0.5
对照: exp206 (SCALE=1.0): 70.5/82.3

## 检查点

### [15:10] 检查点 #1

**状态**: 正常
**进度**: Epoch 2/120, ETA 5h12m
oa_sd=0.518. 训练正常。
**决策**: 继续

### [15:36] 检查点 #2

**ep10: 42.3/56.4** (vs exp206 1.0x: 47.9/60.3 = -5.6/-3.9)
0.5x 初期落后，但 Tiny 上也是如此（0.5x 前期慢，后期追上并超越 +1.5%）。
**决策**: 继续

### [08:11] 远程磁盘满 crash! 已清理 8.4GB。重启 exp208。

远程 30GB 磁盘满 → inline_container.cc error → crash。
已清理旧实验 checkpoints。8.4GB 可用。exp208 重启。
**决策**: 继续

### [16:20] 检查点 #3

重启后正常（清理旧 output dir 解决 inf 问题）。
ep1 done, ETA 5h09m。
**决策**: 继续

### [08:26] 实验终止 — NO-OP!

审查发现 GLOBAL_LOSS_SCALE 不影响 GCN list-loss path！
GCN 路径已经隐含 w_g=0.5 (通过 POSE_PART_WEIGHT=1.0)。
exp208 = exp206 的完全重复。已 kill。
**远程 GPU 空闲。需要新实验。**
