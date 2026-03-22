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
