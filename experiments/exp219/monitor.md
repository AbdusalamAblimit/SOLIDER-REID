# exp219 Tiny + GCN+PAA + PACI (no OA-SD) 监控

配置: Tiny GCN+PAA + PACI (no OA-SD, no EMA teacher)
对照: exp030a baseline (60.7), exp191 OA-SD (63.2/75.4), exp218 PACI+OA-SD (61.9/74.2)

补记：已从远程补回 `train_log`，当前可直接复核到 `ep30=51.9/64.9`；尚未见更后续 eval，因此这份监控只能支撑 early stop-loss 判断，不能上升成正式 final 结果。

## 检查点

### [17:26] 检查点 #1

ep10: 37.7%, ep20: 47.5%
vs baseline: ep10 38.2, ep20 46.8 → PACI-only 几乎 = baseline
PACI 的 consistency loss 在 detached GCN 上单独贡献极小。
**决策**: 继续到 final 确认

### [17:47] 检查点 #2

ep30: 51.9% (vs baseline 52.2 = **-0.3!**)
截至 ep30，PACI-only 仍比 baseline 更差；至少在早期阶段，consistency loss 没有显示出独立正收益。
更稳妥的表述应是：**prototype bank 的独立价值很弱，现有证据不足以支持它在无 OA-SD 时带来收益。**
**决策**: 继续到 final 确认
