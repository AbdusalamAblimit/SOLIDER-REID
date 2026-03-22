# exp144 监控

## 实验信息
- 方法: `SASA alpha=1.0（强骨架注意力偏置）`
- 类型: exp143 的 alpha 扫参（0.1→1.0）
- 运行位置: 本地 (3090)
- 直接对照: `exp143 (alpha=0.1, 中性)`，`exp030a`
- 唯一差异: `POSE_SASA_ALPHA: 1.0`

## 启动记录

### [2026-03-22 00:06] 正式启动

- 启动原因:
  1. exp143 (alpha=0.1) 完美中性，需要测试是否是偏置强度不够
  2. alpha=1.0 让骨架偏置与 attention score 同量级
  3. 零代码改动（仅 config 差异），可立即启动
- 配置: `configs/occluded_duke/pose_psg_gcn_sasa_alpha1.yml`
- 输出: `log/occluded_duke/exp144_sasa_a1/`

### [2026-03-22 00:20] E10 = 39.1%, E20 = 46.7%（与 alpha=0.1 几乎相同）

- E10: 39.1% / 52.0%（vs exp143 E10: 37.4%/50.2%, vs exp030a: 38.2%/51.3%）
- E20: 46.7% / 59.7%（vs exp143 E20: 46.8%/58.9%, vs exp030a: 46.8%/60.9%）
- 判断: alpha=1.0 与 alpha=0.1 结果几乎相同，说明 SASA 偏置强度不是问题
- 骨架测地距离信息在 Swin attention 中确实是冗余的
- 继续跑到完成以获取最终确认

### [2026-03-22 00:51] E30/E40 tracking exp143 closely

- E30: 53.7% (vs exp143: 52.8%, exp030a: 52.2%)
- E40: 56.4% (vs exp143: 55.4%, exp030a: 55.6%)
- alpha=1.0 shows slightly higher early values than alpha=0.1
- But exp143 also had early positive signals that vanished by E120
- Expect same pattern here — continue monitoring

### [2026-03-22 01:01] E50 = 56.5% — 低于 alpha=0.1 的 E50

- E50: 56.5% / 68.9%（vs exp143 alpha=0.1 E50: 57.6%/69.8%, vs exp030a: 55.7%/68.8%）
- **alpha=1.0 在 E50 反而不如 alpha=0.1**（56.5% vs 57.6%）
- 但仍高于 exp030a（+0.8%）
- 结论趋势明确：SASA alpha 值不影响最终结果，中间差异是正常训练噪声
- 继续跑到 E120 获取最终确认

### [2026-03-22 01:12] E60 = 58.2% — 低于 alpha=0.1

- E60: 58.2% / 70.6%（vs exp143: 58.9%/71.0%, vs exp030a cs: 57.7%/70.8%）
- alpha=1.0 持续低于 alpha=0.1 的中间结果
- 但仍高于 exp030a（+0.5%），这个差异会在 E120 消失
- 结论不变：SASA 是完全中性的，alpha 不影响最终结果

### [2026-03-22 01:33] E70/E80 持续跟踪基线

- E70: 58.6% / 71.4%（vs exp030a cs: 58.1%, vs exp143: 58.9%）
- E80: 59.5% / 71.4%（vs exp030a cs: 59.4%, vs exp143: 59.7%）
- alpha=1.0 跟踪 exp030a 和 exp143 都在 ±0.3% 以内
- exp145 PAA+SASA E40 = 56.2% = exp066 PAA exactly
- 结论确定：SASA 完全中性，不影响训练，不影响最终结果

### [2026-03-22 02:15] 训练完成：SASA alpha=1.0 确认中性

- **最终结果**: E120 = 61.0% mAP / 73.5% R1 / 84.6% R5 / 87.9% R10
- vs exp143 (α=0.1): 61.1% / 73.7% → Δ = -0.1% / -0.2%（噪声范围）
- vs exp030a: 61.1% / 73.7% → Δ = -0.1% / -0.2%
- **结论**: 10x 更强的 SASA 偏置与 α=0.1 结果完全相同，确认 skeleton geodesic attention 信息在 Swin 中完全冗余
