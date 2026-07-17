# 实验 exp101: SGMT (Skeleton-Guided Masked Training)

## 统一框架: SGFR (Skeleton-Guided Feature Recovery)

经过 100 个实验的穷举探索：
- Backbone conditioning (PSG/PAA/PKP/FiLM): 天花板已达到 (+2.2%)
- GCN 分支改进: 近最优
- 辅助损失: 全部失败
- **SGCFR (测试时恢复)**: 我们最强且最独特的创新 (+2.6%)

**核心洞察**: 骨架图是处理遮挡的统一工具。它既可以在训练时引导特征恢复（SGMT），
也可以在测试时从gallery恢复特征（SGCFR）。两者共同构成一个完整框架。

## SGMT: 训练端创新

### 核心想法
训练时随机遮蔽关键点特征，让 GCN 通过骨架消息传播恢复被遮蔽的信息。
已有的 ID/triplet 损失自然训练 GCN 学会遮挡恢复 — 无需辅助损失。

### 与之前 SMKC 的区别
- SMKC 是一个独立的小实验（被用户要求回退）
- SGMT 是 **SGFR 框架的一部分**：训练时 SGMT → 测试时 SGCFR
- 论文价值: 不是一个小 trick，而是统一框架的一半

### 技术方案
```
Training:
  backbone → keypoint features (B, 17, 768)
      ↓
  Random mask 30% keypoints → replace with learnable [MASK] token
      ↓
  GCN propagation → visible neighbors fill masked positions
      ↓
  Weighted pool → BN → ID loss + triplet (same as before)

Testing:
  backbone → keypoint features (B, 17, 768)
      ↓
  Low-confidence keypoints → replace with [MASK] token
      ↓
  GCN propagation → within-image recovery
      ↓
  SGCFR: cross-image recovery from gallery candidates
      ↓
  Final matching
```

### 修改
1. `model/modules/skeleton_gcn.py`: 添加 mask_token + masking 逻辑
2. `config/defaults.py`: POSE_SGMT flags
3. 新 config 文件

### 参数: 768 (一个 learnable mask token)
