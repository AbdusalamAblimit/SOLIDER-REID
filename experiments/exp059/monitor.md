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

**观察**: ep10 与 ROA-only 完全相同。PGAM 在早期没有额外效果（与之前观察一致——PGAM 效果在后期才显现）。继续监控。
