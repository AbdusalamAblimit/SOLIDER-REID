# exp219 Tiny + GCN+PAA + PACI (no OA-SD) 监控

配置: Tiny GCN+PAA + PACI (no OA-SD, no EMA teacher)
对照: exp030a baseline (60.7), exp191 OA-SD (64.4), exp218 PACI+OA-SD (61.9)

## 检查点

### [17:26] 检查点 #1

ep10: 37.7%, ep20: 47.5%
vs baseline: ep10 38.2, ep20 46.8 → PACI-only 几乎 = baseline
PACI 的 consistency loss 在 detached GCN 上单独贡献极小。
**决策**: 继续到 final 确认

### [17:47] 检查点 #2

ep30: 51.9% (vs baseline 52.2 = **-0.3!**)
PACI-only 比 baseline 更差！consistency loss 在 detached GCN 上没有正面效果。
**PACI 的值完全来自 OA-SD 的 distillation，不是来自 prototype bank。**
**决策**: 继续到 final 确认
