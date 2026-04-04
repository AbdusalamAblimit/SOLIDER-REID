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

### [02:35] 检查点 #2

### [02:37] 检查点 #3 — ep10

**ep10: 34.5/46.2** (vs exp191 34.3/46.8 = **+0.2/-0.6**)

GiLt 模式在 ep10 比 full PPA (exp237 +2.2/+1.0) 弱。
没有 Part CE → Part features 学得更慢 → 早期贡献小。
需要看后期是否追上。
**决策**: 继续

### [02:46] 检查点 #4

### [02:53] 检查点 #5 — ep20

**ep20: 44.6/55.2** (vs exp191 46.0/58.0 = **-1.4/-2.8**)

| Epoch | exp239 (GiLt) | exp237 (full CE) | exp191 | delta vs base |
|-------|------|------|------|------|
| 10 | 34.5/46.2 | 36.5/47.8 | 34.3/46.8 | +0.2/-0.6 |
| **20** | **44.6/55.2** | **48.4/59.5** | **46.0/58.0** | **-1.4/-2.8** |

GiLt 模式落后 full PPA 和 baseline。Part CE 对 PPA 是必要的。
**决策**: 继续到 final 收集完整数据
