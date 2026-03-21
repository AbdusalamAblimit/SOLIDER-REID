# exp143 监控

## 实验信息
- 方法: `SASA（Skeleton-Aware Self-Attention）`
- 类型: 零参数注意力偏置，基于骨架测地距离
- 运行位置: 本地 (3090)
- 直接对照: `exp030a`（PSG+GCN）
- 关键差异: 唯一差异是 `POSE_SASA: True`

## 启动记录

### [2026-03-21 21:59] 正式启动

- 启动原因:
  1. `exp142 SKC` 已确认为中性偏负（-0.8% mAP），feature-level completion 方向再次失败
  2. SASA 代表一个全新方向：通过骨架拓扑结构修改注意力路由，而非修改特征值
  3. 零参数（纯归纳偏置），不需要从数据中学习复杂函数
- Claude 审查结论: 通过（无 Critical/High 问题）
- 配置: `configs/occluded_duke/pose_psg_gcn_sasa.yml`
- 输出: `log/occluded_duke/exp143_sasa/`
- 与 exp030a 的唯一差异:
  - `POSE_SASA: True`
  - `POSE_SASA_ALPHA: 0.1`
  - `POSE_TEST_FEAT: 'equal_concat'`（注意：exp030a 用 `concat_scaled`）
  - `OUTPUT_DIR`

### [2026-03-21 22:00] 启动健康确认

- `[SASA] Skeleton-Aware Self-Attention enabled: alpha=0.1, params=0`
- E1 Iter[20-140] Loss 正常下降（22.2 → 16.5）
- 没有 NaN / OOM / shape 错误
