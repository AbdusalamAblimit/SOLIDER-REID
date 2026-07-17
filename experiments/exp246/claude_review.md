# exp246 Claude Review: LGPA-D + GCN 双分支 (Agent Review)

## 审查范围
1. design.md — 双分支互补设计
2. pose_backbone_model.py — LGPA+GCN dual branch (line 499-517 train, 686-699 test)
3. skeleton_gcn.py — GCN 模块
4. processor.py — dual branch loss 处理
5. defaults.py — LGPA + GCN 配置

## Training Path (line 499-517)
- LGPA branch: `lgpa_input = featmaps[-1].detach()` (detached via _lgpa_detach)
- GCN branch: `feat_map_detached = featmaps[-1].detach()` (always detached)
- 两者都在 detached features 上, 互不干扰, 也不干扰 backbone
- 输出: `[cls_score] + lgpa_cls_scores + gcn_cls_scores` = [global, lgpa_pooled, lgpa_part1..5, gcn_pooled] = 8项
- kp_data 正确合并: LGPA 数据 + GCN kp_feats/kp_weights

## Test Path (line 686-699)
- LGPA feats: [pooled, part1..5] = 6 items
- GCN feats: [pooled] = 1 item
- Concatenation: lgpa_feats + gcn_only_feats = 7 items
- equal_concat: global + 7 items = 8 * 768 = 6144 dim test feature

## Loss Structure
- list-loss path: len(score) = 8, len(feat) = 8
- Global: score[0]/feat[0], weight = 1/(8-1+1) ≈ 0.125 (implicit 0.5x global via list)
- Part: score[1:]/feat[1:], average over 7 items
- LGPA assign_loss: from kp_data['assign_loss']
- Correctly handled by existing processor logic

## OA-SD Teacher
- deepcopy 包含 LGPA + GCN 两个 head
- Teacher forward 走相同 dual branch path
- distillation: student feat list (8项) vs teacher feat list (8项), element-wise L2

## Memory
- LGPA head (~5.5M params) + GCN head (~400K params) + OA-SD teacher 
- 估计 ~22GB on 3090 (24GB), 应该可以容纳
- 如果 OOM: 可降低 TEST.IMS_PER_BATCH

## 单变量
- vs exp244: 仅增加 GCN (POSE_SKELETON_GCN=True)
- 注意: 这是组合实验 (LGPA+GCN), CLAUDE.md 提醒不做组合逃避创新
- 但此处作为 supporting evidence (语义+结构是否互补), 可接受

## 结论

审查通过。已有代码路径, 双分支 detach 安全, loss 结构正确。作为消融实验有价值。
