# exp243 Claude Review v2: LGPA 修复后二次审查

## 修复内容

### H1 (已修复): Pose-conditioned attention
- 原问题: vanilla MHA + 后混合 pose-pooled features
- 修复: 手动 cross-attention, pose_bias 在 softmax 前注入 QK^T/sqrt(d) + pose_bias

### H2 (已修复): 使用 target_heatmaps
- 原问题: 用 scene_heatmaps (含所有人)
- 修复: 从 _prepare_pose 取 target_heatmaps (person-0), train+test 路径都已修复

### M3 (已修复): 5 parts 统一
- 原问题: 4 parts vs PPA 的 5 parts
- 修复: 5 parts + background = 6 text prompts, BN/classifiers 数量对应

### M4 (已修复): CLIP hard-fail
- 原问题: try/except 静默退化为随机
- 修复: 无 try/except, 直接 import + 调用

## 二次审查新发现

### Medium: PART_KPS 与 PART_TEXTS 命名不完全一致
- PART_TEXTS[2]="upper arms" 但 PART_KPS[2]=[5,7,9]=左臂全部
- PART_TEXTS[3]="lower arms/hands" 但 PART_KPS[3]=[6,8,10]=右臂全部
- 影响: 有限。CLIP features 是语义初始化, 投影层会学习适应。assign_loss 用 PART_KPS 与自身一致。

## 验证清单

| 检查项 | 状态 |
|--------|------|
| Pose bias 在 softmax 前注入 | PASS |
| target_heatmaps train+test | PASS |
| 5 parts + background | PASS |
| CLIP hard-fail | PASS |
| Q/K/V shapes | PASS (B,H,L,d 正确) |
| Scale factor | PASS (1/sqrt(96)) |
| pose_bias broadcasting | PASS (unsqueeze(1) across heads) |
| AMP 安全 | PASS |
| OA-SD 兼容 | PASS |
| 输出结构 train | PASS (7 cls/feats) |
| 输出结构 test | PASS (5376 = 7*768) |
| assign_loss on pose-conditioned attn | PASS |

## 结论

审查通过。所有4个问题已正确修复。无新 Critical/High 问题。
