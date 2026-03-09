# 实验 exp002: Spatial Softmax Part Pooling

## 动机
- exp001 证明 pose part pooling 有效（mAP +0.5%），但 id_part 收敛极慢（最终 id_part≈2.0 vs id_global≈0.2）
- 原因分析：sigmoid 将热图值映射到 [0.25, 0.99] 范围，在 12×4=48 个位置上差异不够大，导致 part attention ≈ 均匀分布 ≈ GAP
- 假设：使用 spatial softmax（在每个 part 的 12×4 位置上做 softmax）可以产生更尖锐的 attention，让 part 特征与 GAP 有本质区别

## 创新点 / 核心想法
- **核心假设**：spatial softmax 使热图 attention 更集中于关键区域（如头部热图只关注上方 2-3 个位置），从而产生真正不同于 GAP 的 part 特征
- 与 exp001 的唯一区别：将 `sigmoid(heatmap)` 改为 `spatial_softmax(heatmap, temperature=T)`

## 技术方案
- **修改文件**: model/modules/pose_part_pooling.py — forward 方法
- 改动一行：从 `torch.sigmoid(scene_heatmaps)` 改为 `spatial_softmax(scene_heatmaps)`
- Spatial softmax: 对每个 part heatmap 的 (fH, fW) 维度做 softmax，使所有位置权重和为 1
- Temperature 参数可调：T 越小 attention 越尖锐（默认试 T=1.0）
- 其他所有配置不变

## 预期结果
- 如果成立：id_part 应显著下降（从 2.0 → <1.5），最终 mAP ≥ 57.5%
- 如果失败：说明 12×4 分辨率下即使 attention 更尖锐也不够，需要转向更高分辨率或完全不同的方法

## 对照组
- Baseline 对照：exp000 (mAP 56.6%, R1 66.5%)
- 消融对照：exp001 (mAP 57.1%, R1 66.7%) — 唯一区别是 softmax vs sigmoid

## 论文定位
- 消融表中的一行：验证 attention 归一化方式的选择
