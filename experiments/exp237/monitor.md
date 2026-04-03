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
