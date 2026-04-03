# exp237 Tiny + PPA (Pose-Prompted Part-Assignment) + OA-SD 监控

配置: Tiny + PSG + PPA (替换 GCN) + OA-SD + PLBOA(0.7) + 无 ROA
**范式创新**: 从 detached GCN sampling 到 end-to-end learnable part assignment
对照: exp191 (Tiny OA-SD, detached GCN): 63.2/75.4

## 检查点

### [22:17] 检查点 #1

本地启动成功。ep1.
ppa_assign=1.63 (assignment CE loss, 下降中)
ppa_bg_ratio=0.916 (大部分 token 分配为背景, zero-init 预期行为)
ppa_entropy=1.752 (assignment 开始变 confident)
**决策**: 等 ep10 eval

### [22:23] 检查点 #2

ep5. ppa_assign=0.69 (**从 1.77 快速下降!** assignment 在学习)
ppa_bg_ratio=0.52 (从 0.92 下降 — tokens 开始被分配到 body parts)
ppa_entropy=0.81 (assignment 变 confident)
ep10 eval ~10min。
**决策**: 继续

### [22:32] 检查点 #3 — ep10

**ep10: 36.5/47.8** (vs exp191 34.3/46.8 = **+2.2/+1.0**)

**PPA 正向！** mAP +2.2, R1 +1.0。
ppa_assign=0.53 (assignment 在学习，loss 从 1.77→0.53)
ppa_bg_ratio=0.48 (约一半 tokens 被分配为 body parts)
ppa_entropy=0.58 (assignment 变得 confident)

**对比 FSDC ep10**: exp236 +4.6/+6.3, exp235 +2.0/+2.2。
PPA 的早期加速 (+2.2) 与 FSDC (exp235 +2.0) 类似。
**关键差异**: PPA 操作在 NON-detached features 上 → 可能有不同的后期行为。
ETA ~3h。
**决策**: 继续！密切监控
