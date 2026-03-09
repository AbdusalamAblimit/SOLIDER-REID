# exp009: Multi-Stage PSG — 训练监控

## 实验配置
- **核心改动**: PSG 注入 Stage 2 (6 blocks) + Stage 3 (2 blocks)
- **Config**: `configs/occluded_duke/pose_multi_psg.yml`
- **Output**: `./log/occluded_duke/exp009_multi_psg`
- **模型**: PoseBackboneModel (multi-stage PSG)
- **Test feat**: global (768-dim)
- **Extra params**: 258K (vs exp007 102K)
- **GPU**: 8.2GB

---
### [23:40] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| id_global | 6.555 |
| tri_global | 9.6-13.6 |

**观察**: 初始阶段正常。与 exp007（同为 PoseBackboneModel，仅 Stage 3）对比，tri_global 初始值类似。8.2GB GPU 内存可接受。
**决策**: 继续监控
