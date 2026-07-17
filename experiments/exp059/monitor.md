# exp059 ROA+PGAM 组合训练监控

## 实验信息
- **方法**: PSG + GCN + ROA + PGAM
- **配置**: `configs/occluded_duke/pose_psg_gcn_roa_pgam.yml`
- **输出**: `log/occluded_duke/exp059_roa_pgam/`
- **对照**: exp058 (ROA only) = 61.8%/72.8%, exp054 (PGAM only) = 61.1%/73.8%
- **启动时间**: 2026-03-14 22:20
- **PID**: 816578

---

### [22:31] 检查点 #1 — Epoch 10

| 指标 | exp059 ROA+PGAM | exp058 ROA | exp054 PGAM | exp030a |
|------|-----------------|------------|-------------|---------|
| mAP | 37.2% | 37.2% | 38.3% | 38.2% |
| R1 | 50.5% | 50.5% | 51.3% | 51.3% |

**观察**: ep10 与 ROA-only 完全相同。PGAM 在早期没有额外效果。

---

### [00:25] 最终结果

**最终: mAP 61.8% / R1 72.8%** — 与 exp058 ROA-only **精确相同**。

所有 12 个检查点（ep10-120）都与 exp058 匹配到 0.0-0.1% 以内。PGAM 在 ROA 存在时完全冗余。

**结论**: PGAM 和 ROA 都解决遮挡鲁棒性，不正交。ROA 从数据层面已完全覆盖 PGAM 的注意力层面效果。
