# 实验 exp034: Target-Aware Dataloader / Branch

## 动机
- exp033 已完成 target person assignment（98.2% 与 person 0 一致，但有 ~2% 边界修正）
- 当前 GCN/KPP 分支硬编码使用 `keypoints[:, 0, :, :]`，即 person 0 的关键点
- 需要确保 person 0 始终是 target person（按 targetness 排序后放到 index 0）
- 这是一个代码健壮性修正 + visibility 实验的基础设施准备

## 核心想法
- 在 dataset `_load_persons()` 中，读取 `target_person_idx` 并将 target person 移到列表第一位
- GCN 分支无需任何修改（它已经使用 person 0）
- 保持向后兼容：如果 index.json 无 `target_person_idx` 字段，默认使用原始顺序

## 技术方案
1. 修改 `datasets/pose_dataset.py` 的 `_load_persons()`:
   - 从 entry 中读取 `target_person_idx`
   - 如果 target_person_idx != 0，将 target person 移到列表第一位
   - 其他 person 顺序保持不变

2. 不修改 GCN 分支代码（`skeleton_gcn.py`）— 它已经正确使用 person 0

3. 训练对比实验：用 exp030a 的配置跑一次，验证 target-aware reordering 的影响

## 预期结果
- 性能变化在噪声范围内（因为只有 ~2% 的图片受影响）
- 但代码正确性提升，为后续 visibility 实验打基础

## 对照组
- Baseline: exp030a (PSG + GCN, equal_concat, mAP 60.73% 3-seed mean)
- 本实验只改一个变量：dataset 中 target person 的位置
