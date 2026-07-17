# 实验 exp116: Support-Complete Feature Replacement (SCFR)

## 动机

exp110-115 的 SCKD 系列（6 个变体）已经给出了一个清晰信号：

1. prototype bank 作为 cosine distillation target，稳定产生 ~+0.1 mAP / +0.8 R1 的弱正向
2. 无论是提高写入纯度（exp112）、冻结 teacher（exp114/115）、还是提升计数门槛（exp111），都无法突破这个天花板
3. oracle experiment（exp109）的 +8.5% mAP 说明 headroom 巨大，但 SCKD 只能捕获其中 ~1%

**根本问题**：SCKD 是一个 gradient-based indirect signal — 它通过辅助 loss 推动 student 的梯度，但不直接改变 GCN/classifier 看到的输入。oracle 之所以有效，是因为它**直接替换**了低可见度 keypoint 的特征。

## 核心假设

如果在训练时直接用 prototype bank 中的特征**替换**低可见度 keypoint 的采样特征（而不是通过 loss 蒸馏），GCN 和 classifier 将直接受益于更完整的输入，可能带来比 SCKD 更大的提升。

关键区别：
- **SCKD**: `loss += cosine_distance(student_feat, prototype)` → 间接梯度信号
- **SCFR**: `kp_feats[low_vis] = prototype.detach()` → 直接特征替换

## 技术方案

在 `SupportCompleteBank` 中新增 `replace()` 方法：

```python
def replace(self, kp_feats, kp_weights, labels):
    """Replace low-visibility keypoint features with prototypes."""
    replaced = kp_feats.clone()
    low_mask = kp_weights <= self.low_thr
    counts = self.count_bank[labels]
    conf = self.confidence_bank[labels]
    support_mask = (counts >= self.min_count) & (conf > 0)
    replace_mask = low_mask & support_mask

    if replace_mask.any():
        proto = F.normalize(self.prototype_bank[labels], dim=2)
        replaced[replace_mask] = proto[replace_mask].detach()

    return replaced, replace_mask
```

在 `SkeletonGCNHead.forward()` 中，在 GCN 之前调用 replace：

```python
# After sampling, before GCN:
if self.training and self.scfr_bank is not None:
    kp_feats, replace_mask = self.scfr_bank.replace(kp_feats, kp_scores, labels)
    # 注意：替换的特征是 detached 的，不会回传梯度到 bank
    # 但它们参与 GCN 的 forward，提供完整的上下文信息
```

保留 bank 的后台 EMA 更新（同 SCKD），但移除 distillation loss。

关键配置：
- `POSE_SCFR = True`
- `POSE_SCKD = True`（保留 bank 创建和更新，但 SCFR 模式下 loss 被跳过）
- `POSE_SCKD_UPDATE_THR = 0.5`（与 exp110 一致，保证单变量）
- `POSE_SCKD_WARMUP = 20`（前 20 epoch 只更新 bank，不替换）

## 对照组

1. 主对照: `exp110_sckd`（SCKD 蒸馏版）
2. 次对照: `exp030a-eq seed1234`（无 SCKD 基线）

## 预期结果

如果假设成立：
1. GCN 在训练时接收更完整的输入 → 学到更好的消息传递模式
2. 分类器在训练时看到更完整的 pooled feature → 学到更鲁棒的决策边界
3. 最终 mAP 可能超过 SCKD 的 ~61.2（目标 ~61.5+）

如果失败，最可能原因：
1. 替换的 detached 特征切断了梯度，导致被替换 keypoint 完全不被优化
2. Prototype 特征和当前 batch 特征分布不匹配，导致 GCN 输入不一致
3. Bank 在 warmup 阶段不够成熟，替换的特征质量太差

## 风险与缓解

1. **梯度断裂**: 被替换的 keypoint 不再接收梯度。但这些本来就是低可见度 keypoint，其原始特征主要来自遮挡物/背景，梯度信号本来就是噪声。
2. **训练/测试不一致**: 训练时有替换，测试时没有。但测试时有 SGCFR 可以做类似的 recovery。也可以在测试时也用 bank（但需要 gallery 先跑一遍 bank）。
3. **替换比例过高**: 如果大部分 keypoint 都被替换，GCN 的输入几乎全是 prototype，失去了 instance-specific 信息。可以通过 `low_thr` 控制。
