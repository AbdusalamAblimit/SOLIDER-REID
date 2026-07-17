# 实验 exp218: PACI — Pose-Anchored Compositional Identity (Tiny)

## 动机

### 问题重新定义
当前所有方法（包括我们的 PSG+GCN+OA-SD+MaxSim）都把 identity 当作单一概念：
一张图 → 一个向量（或固定 part 向量集）。遮挡时，这个表示不完整，且没有办法知道
"这个人的被遮挡部分应该长什么样"。

**PACI 将 identity 重新定义为"per-part prototype 的组合"。**
每个 identity 不是一个向量，而是由 17 个（或更少个） body-part prototypes 组成的。
这些 prototypes 在训练过程中通过 EMA 积累同一个 identity 在不同图片中的 part features。

### 与已有方法的本质区别
| 方法 | 做什么 | 局限 |
|------|--------|------|
| GCN | 从同一张图的可见 parts propagate | 无法想象没见过的 parts |
| OA-SD | teacher 给 clean target | 不记忆 identity-specific 信息 |
| TTSFR/SKC | batch 内 same-ID recovery | 只用当前 batch，信息有限 |
| NFC | gallery neighbors | test-time hack，用户禁止 |
| **PACI** | **training-time per-ID per-part memory bank** | **跨整个训练集记忆每个 identity 的 parts** |

### 创新门槛检查
1. ✅ **问题层面**: 重新定义 identity 为 per-part prototype composition (vs single vector)
2. ✅ **机制层面**: Per-identity per-part momentum prototype bank 在 ReID 中是新的
3. ✅ **证据层面**: 清晰的消融设计：(a) baseline, (b) +prototype bank, (c) +consistency loss, (d) +test-time completion

## 核心假设
Per-identity per-part prototypes 能让模型学习 identity-specific 的 part appearance，
在测试时通过 prototype substitution 补全遮挡的 parts，提升 matching accuracy。

## 技术方案

### Phase 1: Prototype Bank (本实验)
```python
class PartPrototypeBank:
    # Shape: (num_ids, 17, feat_dim)
    # Momentum update: P[id][k] = alpha * P[id][k] + (1-alpha) * feat_k
    
    def update(self, kp_feats, kp_scores, labels):
        # For each sample in batch:
        #   For each visible keypoint k (score > threshold):
        #     P[label][k] = 0.9 * P[label][k] + 0.1 * kp_feats[k]
        
    def get_prototypes(self, labels):
        # Return prototypes for given labels
        # Shape: (B, 17, C)
        return self.bank[labels]
```

### Phase 2: Part-Prototype Consistency Loss
```python
# For each visible keypoint k of identity j:
#   Pull: cos_sim(feat_k, P[j][k]) should be high (same identity's part)
#   Push: cos_sim(feat_k, P[j'][k]) should be low (different identity's same part)
# This teaches: "this person's left arm should look like THIS"

consistency_loss = 0
for k in range(17):
    if visible[k]:
        pos = cosine_sim(feat_k, prototype[j][k])  # positive: same ID
        neg = cosine_sim(feat_k, prototype[rand_j'][k])  # negative: diff ID
        consistency_loss += max(0, margin - pos + neg)  # triplet-style
```

### Phase 3: Test-Time Prototype Completion (future)
在测试时，对遮挡的 query：
1. 先用可见 parts 做初步 matching
2. 从 top-K gallery matches 提取 prototypes
3. 用 prototypes 补全 query 的遮挡 parts
4. Re-rank with completed features

### 修改文件
- `model/modules/part_prototype_bank.py`: 新模块
- `config/defaults.py`: POSE_PACI, POSE_PACI_MOMENTUM, POSE_PACI_WEIGHT
- `processor/processor.py`: Bank update + consistency loss
- `model/pose_backbone_model.py`: 无需修改

### 关键设计决策
1. **Bank 在 detached features 上工作** — 不需要 non-detached，因为 bank 只是记忆，不参与梯度
2. **Consistency loss 也用 detached bank** — bank prototypes 作为 target (不传梯度到 bank)
3. **Consistency loss 通过 GCN kp_feats 传梯度到 GCN** — 与现有 CE/triplet 一致
4. **不与 OA-SD 冲突** — PACI 和 OA-SD 是正交的（PACI 记忆，OA-SD distill）

## 预期结果
- Phase 1 (bank only): 可能无直接效果（bank 只是记忆）
- Phase 2 (+consistency loss): 相对 `exp191 = 63.2/75.4` 再涨 `+1-2%` mAP（约 `64-65%`）
- Phase 3 (+test-time completion): 额外 +1-2%

## 对照组
- exp030a baseline: 60.7%
- exp191 OA-SD: 63.2/75.4
- exp187 SupCon 3v: 64.9%
