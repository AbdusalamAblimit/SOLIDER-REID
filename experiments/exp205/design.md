# 实验 exp205: GCN+PAA + STD-PR per-token SupCon (Dual Part Branch)

## 动机
- GCN+PAA (CE only) 在 Small 上 = 70.8/81.7 (4090 PAA)
- STD-PR+SupCon 在 Small 上 = 69.3/80.2 (exp202b)
- GCN+PAA+SupCon 在 Small 上 = ? (exp203, 进行中但 ep10 落后 STD-PR)
- **根本问题**: SupCon 在 GCN 上效果差因为只有 1 个 pooled skeleton feat
- **解决方案**: 同时启用 GCN (用于 equal_concat test feat) + STD-PR (用于 per-token SupCon)

## 核心假设
GCN+PAA 提供更强的架构 (70.8 CE ceiling)，STD-PR 提供 per-token SupCon 的训练信号。
两者组合：GCN 的测试特征 + STD-PR 的训练 loss → 可能超过两者各自的 best。

## 技术方案

### 关键改动: model/pose_backbone_model.py
目前 GCN 和 STD-PR 是 `elif` 关系 (line 340 vs 432)。需要改为可同时启用。

当 `use_structural_routing AND use_skeleton_gcn` 同时为 True 时：
1. 运行 STD-PR → 产生 structural_tokens (6 per-token features)
2. 运行 GCN → 产生 gcn_feats (1 pooled skeleton feature)
3. 训练时返回: scores = [global_cls, str_cls_1...6, gcn_cls], feats = [global, str_1...6, gcn]
4. SupCon 在 str_1...6 上计算 (per-token)
5. CE+triplet 在所有 features 上计算
6. 测试时: equal_concat 包含 global + gcn + str_tokens

### 修改文件
- `model/pose_backbone_model.py`: 修改 forward() 的分支逻辑，允许 GCN+STD-PR 同时运行
- `config/defaults.py`: 新 flag `POSE_DUAL_PART_BRANCH = False`

### 代码改动量
~50-80 行修改（主要在 forward 的分支逻辑合并）

## 预期结果
- GCN 架构优势 (70.8 CE) + STD-PR SupCon 优势 (+3-4%) → 73-74%?
- 最保守估计: 至少超过 exp202b (69.3) 和 4090 PAA (70.8)

## 对照组
- 4090 PAA (GCN+PAA, CE): 70.8/81.7
- exp202b (STD-PR+SupCon+3-view): 69.3/80.2
- exp203 (GCN+PAA+SupCon, 进行中)
