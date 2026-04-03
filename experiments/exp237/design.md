# 实验 exp237: PPA — Pose-Prompted Part-Assignment Head (范式级创新)

## 动机

236 个实验的根本发现: **在 detached features 上操作的所有方法都无法改善 final 结果。**

| 类别 | 方法 | Final delta |
|------|------|------|
| Detached feature completion | FSDC (exp235/236) | ~-1.5% |
| Detached per-keypoint loss | PKC, OERL, PACI | 0 ~ -1% |
| Non-detached gradient | GSPB, BT-PKD | ~-1% (late interference) |
| Per-part independent | exp233 | -2.8% |

**唯一有效的方向是改变 backbone 本身的行为**: PSG (+1.7%), OA-SD (+2-3%)。

KPR (ECCV 2024, 75.1% SOTA) 的核心差异: **learnable part assignment head, end-to-end training**。
KPR 用 softmax attention 让每个 spatial token 分配到一个 body part，梯度端到端流过 backbone。

## 核心创新

**PPA: 用 pose heatmap 监督的 learnable part-assignment head 替换 detached GCN sampling。**

```
当前 pipeline:
backbone → featmap → DETACH → GCN sampling → pool → Part loss (不影响 backbone)

PPA pipeline:
backbone → featmap → Part Assignment Head → per-part pool → Part loss (影响 backbone!)
                   ↓
           pose heatmap supervision (cross-entropy)
```

## 技术方案

### Part Assignment Head
```python
class PartAssignmentHead(nn.Module):
    # Input: spatial tokens (B, 48, 768)
    # Output: assignment probs (B, 48, K+1), K=5 body parts + background
    
    self.part_proj = nn.Linear(768, K+1)  # assignment logits
    self.part_bns = nn.ModuleList([BN1d(768) for _ in range(K)])
    self.part_classifiers = nn.ModuleList([Linear(768, num_classes) for _ in range(K)])
    
    def forward(self, tokens, pose_heatmaps):
        # Part assignment
        logits = self.part_proj(tokens)  # (B, 48, K+1)
        probs = F.softmax(logits, dim=-1)  # (B, 48, K+1)
        
        # Supervision: pose heatmaps → GT part labels
        gt_parts = self._heatmaps_to_part_labels(pose_heatmaps)  # (B, 48)
        assign_loss = F.cross_entropy(logits.transpose(1,2), gt_parts)
        
        # Per-part weighted pooling
        part_feats = []
        for k in range(K):
            weights = probs[:, :, k].unsqueeze(-1)  # (B, 48, 1)
            part_feat = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
            part_feats.append(part_feat)  # (B, 768)
        
        # Visibility: max assignment prob per part
        visibility = probs[:, :, :K].max(dim=1)[0]  # (B, K)
        
        return part_feats, assign_loss, visibility
```

### 与 GCN 的关键区别
1. **端到端梯度**: assignment loss (CE on part labels) 流过 backbone
2. **Soft assignment**: 每个 token 可以部分属于多个 part (vs GCN 的 hard bilinear sampling)
3. **Clean gradient**: 单个 softmax CE，不是 17 个独立 per-keypoint losses
4. **Implicit visibility**: 如果某 part 没有高概率 token → visibility low → 自动降权

### GT Part Labels 生成
```
COCO 17 keypoints → 5 body part groups:
0: head (nose, eyes, ears) 
1: torso (shoulders, hips)
2: arms (elbows, wrists)
3: legs (knees, ankles)
4: background (no keypoint nearby)

对每个 spatial token (12x4), 找到 heatmap activation 最高的 body part → GT label
```

### Loss
```
total = CE_global + w_p * (CE_part_avg + triplet_part_avg) + w_a * assign_loss + OA_SD
```

## 对照组
- exp191 (OA-SD, detached GCN): 63.2/75.4

## 预期结果
- 成功: +1.5~3.0% mAP (KPR 证明此机制有效)
- 失败: ~0% (softmax CE 仍然干扰 backbone ID loss)

## 论文价值
- **核心贡献**: 首次在 pose-guided ReID 中用 pose heatmap 监督 part assignment (vs KPR 用 parsing labels)
- **叙事**: "从 detached keypoint sampling 到 end-to-end part assignment"
- **消融**: vs detached GCN, vs KPR parsing supervision, assignment visualization

## 早停
- ep10 < 20% → 终止 (assignment loss 可能冲突)
- ep30 < 45% → 终止
