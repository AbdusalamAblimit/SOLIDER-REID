# exp058 ROA 训练监控

## 实验信息
- **方法**: PSG + GCN + ROA (Realistic Occlusion Augmentation)
- **配置**: `configs/occluded_duke/pose_psg_gcn_roa.yml`
- **输出**: `log/occluded_duke/exp058_roa/`
- **对照**: exp030a (PSG+GCN, 标准RE) 3-seed mean = 60.73% / 72.57%
- **核心改动**: 50% 概率在训练图像上粘贴 VOC 2012 真实物体（1289 个 RGBA patch）
- **启动时间**: 2026-03-14 20:12 (第二次启动，修复了 PIL→numpy 转换)
- **PID**: 775035

---

### [20:13] 检查点 #1
**状态**: 🟢正常
**进度**: Epoch 1/120
**观察**: Loss 18.1 (ep1 iter100)，略高于 exp030a 的 17.5（可能因为 ROA 增加了遮挡难度）。训练正常。
