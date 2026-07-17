# exp065 PKE+ROA 训练监控

## 实验信息
- **方法**: PSG + GCN + PKE + ROA
- **配置**: `configs/occluded_duke/pose_psg_gcn_pke_roa.yml`
- **对照**: exp058 (ROA only) = 61.8%/72.8%, exp064 (PKE only) = 61.0%/73.1%
- **启动时间**: 2026-03-15 12:39
- **PID**: 1561182

---

### [12:50] Epoch 10: mAP 37.1% / R1 51.0%

---

### [14:48] 最终: mAP **61.9%** / R1 **73.2%**

vs ROA only: +0.1%/+0.4% — PKE 额外贡献可忽略
vs 3-seed mean: +1.17%/+0.63% — 主要来自 ROA

**结论**: PKE 和 ROA 不正交。组合效果 ≈ ROA alone。
