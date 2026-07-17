# 实验 exp229: BT-PKD (Backbone-Through Per-Keypoint Distillation) on Tiny

## 动机

**核心问题**: detach 屏障阻止 Part branch 梯度到达 backbone (50M params)。所有 per-keypoint losses 因此失败。GSPB 部分解决（5% gradient scale），但在 Small 上灾难，且 final 不改善。

**BA-PKC 教训**: 非 detached 的 SupCon/对比学习梯度（sharp, hard mining）与 CE 冲突导致灾难 (exp215: 0.5%)。但这不意味着 ALL 非 detached 梯度都会灾难 — 关键是梯度的 **性质**。

**核心洞察**: L2 distillation from EMA teacher ≠ Contrastive loss from hard mining
- SupCon/triplet: 梯度来自 hard mining，高方差，与 CE 冲突
- L2/cosine distillation toward EMA teacher: 梯度平滑、低方差、目标是 student 自身的缓慢移动平均
- EMA decay 0.999 → teacher 与 student 特征差异小 → 梯度幅度小

## 核心假设

通过 OA-SD teacher 的 per-keypoint features 作为 distillation target，让平滑的 L2 梯度流过非 detached 的 backbone features 到达 backbone。这给 backbone 提供了直接的、per-keypoint 级别的学习信号，但梯度足够温和不会灾难。

## 技术方案

### 数据流
```
Student (occluded image):
  backbone → featmaps[-1] (NON-detached) → grid_sample at 17 keypoints → bt_kp_feats (B, 17, C)

Teacher (clean image, EMA):
  backbone → GCN → kp_feats (B, 17, C) from teacher's kp_data

Loss:
  per_kp_dist = 1 - cosine(bt_kp_feats, teacher_kp_feats.detach())  # (B, 17)
  weighted by teacher_kp_weights (confidence)
  bt_pkd_loss = weighted_mean(per_kp_dist)
  total_loss += bt_pkd_weight * bt_pkd_loss
```

### 修改文件
1. `config/defaults.py`: `POSE_BT_PKD = True`, `POSE_BT_PKD_WEIGHT = 0.01`
2. `model/pose_backbone_model.py`: 复用 BA-PKC 的 non-detached sampling 代码，当 bt_pkd=True 时 sample
3. `processor/processor.py`: OA-SD teacher forward 后，提取 teacher 的 per-keypoint features，计算 cosine distillation loss

### 关键设计选择
- Weight=0.01: 极低权重，因为梯度直接到 backbone。BA-PKC 用 0.1 但灾难。
- Cosine distillation: 比 L2 更稳定，梯度只和角度差相关
- Teacher confidence weighting: 低置信 keypoint 的 distillation 信号降权
- Student features 来自 backbone (非 GCN): 直接让 backbone 对齐 teacher

### 与前序实验对比
| 实验 | 梯度到 backbone? | 损失函数 | 结果 |
|------|------|------|------|
| exp210 PKC w=0.5 | No (detached) | SupCon | 灾难 3.6% |
| exp210b PKC w=0.05 | No (detached) | SupCon | 无效 (= baseline) |
| exp215 BA-PKC w=0.1 | Yes (non-detached) | SupCon | 灾难 0.5% |
| exp220 GSPB 5% | Yes (scaled) | CE+triplet (all) | -0.3 (neutral) |
| **exp229 BT-PKD** | **Yes (non-detached)** | **Cosine distillation** | **?** |

### 为什么这次不同
1. **梯度性质**: 不是 CE/SupCon 的分类梯度，而是 L2/cosine distillation 的回归梯度
2. **梯度幅度**: EMA teacher 与 student 特征接近 → 差异小 → 梯度小
3. **梯度方向**: 指向 teacher (自身 EMA)，不与 CE 竞争
4. **仅在 17 个空间位置**: 不是全部 12×4=48 个 token

## 预期结果

- 假设成立: mAP +0.5~1.5%。per-keypoint distillation 让 backbone 直接学到 body-part 对齐
- 假设失败: 即使 cosine distillation 梯度也太强 → mAP 下降或灾难
- 最可能失败原因: weight=0.01 太高/太低需要调整

## 对照组
- exp191: OA-SD only (63.2/75.4)
- exp220: GSPB 5% (62.9/74.3)

## 早停
- ep10 < 25% → 终止 (灾难检测)
- ep30 < 48% → 终止 (明显负面)
