# exp243 Claude Review: LGPA (Language-Grounded Part Assignment)

## 审查范围
1. `experiments/exp243/design.md` — 设计文档
2. `model/modules/clip_part_head.py` — 新模块 (276行)
3. `model/pose_backbone_model.py` — LGPA 集成 (init + forward + test)
4. `processor/processor.py` — LGPA assign_loss
5. `config/defaults.py` — LGPA 配置
6. `configs/occluded_duke/pose_psg_lgpa.yml` — 实验配置

## 第一轮审查发现 & 修复

### H1 (已修复): Visibility weighting 退化为均匀分布
- 原因: `attn_weights.sum(dim=-1)` 对 softmax attention 求和恒等于 1.0
- 修复: 改用 pose heatmap response 计算 per-part visibility

### M1 (已修复): Dead code (Q, K, V, d_k 赋值但未使用)
- 修复: 删除无用变量

### M2 (已修复): PPA+LGPA 同时启用时 assign_loss 重复计算
- 修复: 添加 `not ppa_enabled` guard

### L1 (已修复): 缺少 LGPA/PPA 互斥检查
- 修复: 添加 `raise ValueError` 互斥检查

## 验证清单

| 检查项 | 状态 | 说明 |
|--------|------|------|
| CLIP text features frozen (buffer) | PASS | register_buffer, 不参与训练 |
| Cross-attention shapes Q/K/V | PASS | Q=(B,5,768), K=V=(B,48,768) |
| Pose mask computation | PASS | COCO keypoint-to-part mapping 正确 |
| Per-part BN + classifier init | PASS | bias frozen, weight=1, bias=0 |
| KL divergence 方向 | PASS | input=log(pred), target=GT |
| Test path 结构 | PASS | 返回 [pooled, part1..4] |
| OA-SD 兼容性 | PASS | deepcopy + buffer 保持 |
| AMP 安全 | PASS | 无 float16 兼容问题 |
| Non-detached gradient | PASS | featmaps[-1] 直接传入 |
| open_clip 可用 | PASS | v2.32.0 |
| Config defaults 安全 | PASS | 所有新默认值为 False |
| 单变量原则 | PASS | 仅替换 PPA 为 LGPA |

## 结论

审查通过。所有发现的问题已修复。实验可以启动。
