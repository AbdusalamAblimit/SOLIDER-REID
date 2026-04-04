# exp241 Tiny + PPA + GCN 双分支 + OA-SD 监控

配置: Tiny + PSG + PPA (w=0.5) + GCN (detached) + OA-SD + PLBOA(0.7)
**组合创新**: PPA 端到端训练 backbone + GCN 额外 detached keypoint features
对照: exp237 (PPA only): 63.7/75.0 (+0.5/-0.4)
对照: exp191 (GCN only): 63.2/75.4

## 检查点

### [11:27] 检查点 #1

本地启动。ep1。PPA + GCN 双分支都在训练。
ppa_assign=1.64, ppa_bg_ratio=0.88。
**决策**: 等 ep10 eval

### [11:37] 检查点 #2

### [11:42] 检查点 #3 — ep10

**ep10: 37.3/49.4** (vs exp191 34.3/46.8 = **+3.0/+2.6**)
vs exp237 PPA-only: 36.5/47.8 = **+0.8/+1.6** — GCN 添加了额外价值！

| 实验 | ep10 delta |
|------|------|
| exp191 GCN only | baseline |
| exp237 PPA only | +2.2/+1.0 |
| **exp241 PPA+GCN** | **+3.0/+2.6** |

双分支比任一单分支都强！

### [11:53] 检查点 #4

### [11:58] 检查点 #5 — ep20

**ep20: 46.6/57.8** (vs exp191 46.0/58.0 = **+0.6/-0.2**)
vs exp237 PPA-only ep20: 48.4/59.5 = -1.8/-1.7

| Epoch | exp241 PPA+GCN | exp237 PPA | exp191 GCN | delta vs base |
|-------|------|------|------|------|
| 10 | 37.3/49.4 | 36.5/47.8 | 34.3/46.8 | +3.0/+2.6 |
| **20** | **46.6/57.8** | **48.4/59.5** | **46.0/58.0** | **+0.6/-0.2** |

ep20: dual 落后 PPA-only。更多 loss 减慢收敛。
但仍正向 vs baseline (+0.6)。需要看后期。
**决策**: 继续
