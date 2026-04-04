# exp239 Tiny + PPA + GiLt Loss (Part triplet only) 监控

配置: PPA + GiLt (Part CE=0, Part triplet only) + OA-SD + PLBOA(0.7)
**创新**: KPR-style loss — global CE + part triplet, 防止梯度竞争
对照: exp237 (PPA, full CE): 63.7/75.0 (+0.5/-0.4)
对照: exp191 (GCN): 63.2/75.4

## 检查点

### [02:22] 检查点 #1

本地启动。ep1. id_part=0.000 (GiLt 模式确认: Part CE = 0)。
ppa_assign=1.77. ETA ~3h.
**决策**: 等 ep10 eval
