# exp243 Tiny + LGPA + OA-SD 监控

配置: Tiny + PSG + LGPA (CLIP + cross-attn + pose mask) + OA-SD + PLBOA(0.7)
对照: exp191 (GCN only): 63.2/75.4
对照: exp237 (PPA only): 63.7/75.0 (+0.5/-0.4)
对照: exp241 (PPA+GCN): 63.7/75.3 (+0.5/-0.1)

## 检查点

### [15:35] 检查点 #1

ep1 iter40. 训练正常启动。lgpa_assign=6.94 (raw, 0.5x后=3.47)。
Loss: 16.6, id_global: 6.55, id_part: 6.72, tri_global: 12.3, tri_part: 0.79, oa_sd: 0.27。
**决策**: 等 ep10 eval

### [15:41] 检查点 #2

ep3 iter80. 训练正常。

| 指标 | ep1 | ep3 | 趋势 |
|------|-----|-----|------|
| Total Loss | 17.2 | 11.6 | 正常下降 |
| id_global | 6.555 | 6.537 | 缓慢下降(正常,warmup阶段) |
| id_part | 6.727 | 6.680 | 缓慢下降 |
| tri_global | 14.05 | 2.01 | 快速下降 |
| tri_part | 0.80 | 0.70 | 缓慢下降 |
| oa_sd | 0.26 | 0.22 | 正常 |
| lgpa_assign | 6.94 | 6.63 | 稳步下降 |

**关注**: Speed=90.9 samples/s, 比 PPA (~160) 慢 44%。CLIP cross-attention 较贵。ETA ~5h。
**决策**: 继续等 ep10 eval

### [15:46] 检查点 #3

ep5 iter120. Acc=15.1%。lgpa_assign=5.61 (持续下降)。
id_global=6.45, id_part=6.60, tri_global=1.06, oa_sd=0.13。
所有指标正常下降。
**决策**: 等 ep10 eval (~12min)

### [15:56] 检查点 #4

ep8 完成。Speed=84 samples/s (比 PPA 慢 ~47%)。ETA ~5h。
**决策**: 等 ep10 eval (~5min)

### [16:02] 检查点 #5 — ep10 eval

**ep10: 38.3/48.1** (vs exp191 34.3/46.8 = **+4.0/+1.3**)

| 实验 | ep10 mAP/R1 | delta vs GCN baseline |
|------|------|------|
| exp191 GCN only | 34.3/46.8 | baseline |
| exp237 PPA only | 36.5/47.8 | +2.2/+1.0 |
| exp241 PPA+GCN | 37.3/49.4 | +3.0/+2.6 |
| **exp243 LGPA** | **38.3/48.1** | **+4.0/+1.3** |

**LGPA ep10 mAP 最高！** +4.0 超过所有 PPA 变体!
R1 +1.3 弱于 PPA+GCN (+2.6) 但仍正向。

**决策**: 非常有希望。继续等 ep20

### [16:14] 检查点 #6

ep15. lgpa_assign=1.89 (从 ep1 6.94 下降至 1.89, 降了 73%)。
id_global=4.72, id_part=5.60, Acc=21.9%。
所有指标正常收敛。
**决策**: 等 ep20 eval (~14min)

### [16:24] 检查点 #7

ep18 完成。ETA 4h38m。
**决策**: 等 ep20 eval (~6min)

### [16:30] 检查点 #8 — ep20 eval

**ep20: 49.0/58.2** (vs exp191 46.0/58.0 = **+3.0/+0.2**)

| Epoch | LGPA | PPA | PPA+GCN | GCN baseline | LGPA delta |
|-------|------|-----|---------|------|------|
| 10 | **38.3/48.1** | 36.5/47.8 | 37.3/49.4 | 34.3/46.8 | **+4.0/+1.3** |
| **20** | **49.0/58.2** | 48.4/59.5 | 46.6/57.8 | 46.0/58.0 | **+3.0/+0.2** |

LGPA mAP 在 ep10 和 ep20 都是最高！
R1 在 ep20 弱于 PPA (+0.2 vs +1.5) 但仍正向。
mAP 增益从 +4.0 下降到 +3.0 — 需要观察后续是否继续收窄。

**决策**: 继续，等 ep30

### [16:40] 训练中止 — 代码重大问题修复

用户指出 4 个问题（全部有道理）:
1. **H1**: Pose-conditioned attention 未按设计实现 (vanilla MHA + 后混合)
2. **H2**: 用 scene_heatmaps 而非 target_heatmaps (CLAUDE.md 明确禁止)
3. **M3**: 4 parts vs PPA 的 5 parts (非单变量)
4. **M4**: CLIP 失败静默退化为随机

**决策**: Kill 训练, 全部修复, 重新审查, 重新启动

---

## 修复后重启
